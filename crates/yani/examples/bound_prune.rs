//! How much of the reachable closure can a transmutation actually populate,
//! and what does a driver have to load to find out?
//!
//! Two questions, both answered against the ENDF/B-VIII.1 fixture chain seeded
//! from Fe56. Run from the repo root:
//!
//! ```text
//! cargo run --release -p yani --example bound_prune
//! ```
//!
//! The per-edge rate is flat here, standing in for the per-nuclide rates the
//! driver folds from a tallied flux spectrum
//! (`TransmutationTallies::bounding_reaction_rates`). A flat sweep is the right
//! shape for these two questions: neither depends on which nuclide carries
//! which cross section, only on how hard the network is driven.

use std::collections::{HashMap, HashSet};

const CHAIN: &str = "crates/yamc/tests/transmutation-endf-b8.1-sfr.arrow";
const FLOOR: f64 = 1e-30; // yani_transmute::DENSITY_FLOOR

/// The 49 nuclides a real coupled solve of this sphere populates (one year at
/// 1e14 n/s). Anything the bound drops that appears here is a lost nuclide.
const POPULATED: &[&str] = &[
    "Ca44", "Ca45", "Ca46", "Ca47", "Co58", "Co59", "Cr50", "Cr51", "Cr52", "Cr53", "Cr54", "Cr55",
    "Cr56", "Fe53", "Fe54", "Fe55", "Fe56", "Fe57", "Fe58", "Fe59", "H1", "H2", "H3", "He3", "He4",
    "Mn53", "Mn54", "Mn55", "Mn56", "Mn57", "Mn58", "Sc45", "Sc46", "Sc47", "Sc48", "Sc49", "Sc50",
    "Ti46", "Ti47", "Ti48", "Ti49", "Ti50", "Ti51", "V49", "V50", "V51", "V52", "V53", "V54",
];

fn main() {
    let root = std::path::Path::new(CHAIN);
    let (chain, _branch) = yani::parse_chain_parts(
        &root.join("decay"),
        Some(&root.join("reactions")),
        Some(&root.join("fission_yields")),
        None,
    )
    .expect("fixture chain");

    let seeds: HashMap<String, f64> = [("Fe56".to_string(), 8.477e-2)].into_iter().collect();
    let seed_refs: Vec<&str> = seeds.keys().map(|s| s.as_str()).collect();
    let reachable = yani::reachable_nuclides(&chain, &seed_refs);
    let reactive = |set: &HashSet<String>| {
        set.iter()
            .filter(|n| chain.get(*n).is_some_and(|cn| !cn.reactions.is_empty()))
            .count()
    };
    let missing = |kept: &HashSet<String>| POPULATED.iter().filter(|n| !kept.contains(**n)).count();
    let year = 365.0 * 86400.0;

    println!(
        "reachable closure: {} nodes, {} reactive\nfloor = {FLOOR:e}\n",
        reachable.len(),
        reactive(&reachable)
    );

    // ---- 1. What the bound keeps, against how hard the network is driven ----
    //
    // `missing` counts the 49 against a bound driven at each row's rate, so the
    // weakest rows drop some: at a hundred-thousandth of the fluence those 49
    // were measured at, fewer nuclides are populated, and tracking that is the
    // point. Driven harder the bound returns the whole closure rather than
    // truncating, so it degrades safely in the other direction too.
    println!("Rating the whole closure:\n");
    println!(
        "{:>10} {:>9} {:>8} {:>9} {:>11} {:>8}",
        "flux", "sigma_b", "kept", "reactive", "of_closure", "missing"
    );
    for flux in [1e12_f64, 1e14, 1e16] {
        for sigma_b in [0.001_f64, 0.1, 1.0, 100.0] {
            let rate = flux * sigma_b * 1e-24;
            let kept = yani::populated_nuclides(&chain, &seeds, year, FLOOR, |_, _| rate);
            // Every kept nuclide must be reachable; the bound cannot invent one.
            assert!(
                kept.is_subset(&reachable),
                "bound kept something outside the closure"
            );
            println!(
                "{:>10.0e} {:>9.3} {:>8} {:>9} {:>10.1}% {:>8}",
                flux,
                sigma_b,
                kept.len(),
                reactive(&kept),
                100.0 * kept.len() as f64 / reachable.len() as f64,
                missing(&kept),
            );
        }
    }

    // ---- 2. Could a driver reach the same answer while loading less? ----
    //
    // Rating a nuclide costs its cross sections, so the tempting move is to
    // grow the rated set outwards from the seeds, rating the unrated at zero,
    // and finish with a verification sweep that rates them at a ceiling no
    // cross section can exceed instead. That sweep is what makes the answer
    // sound: nothing outside a set it leaves untouched can reach the floor.
    //
    // It also has to load the whole closure before it will leave anything
    // untouched, which is the finding: there is no cheaper sound route, so the
    // driver rates the closure outright.
    println!("\nGrowing the rated set from the seeds instead:\n");
    println!(
        "{:>10} {:>9} {:>8} {:>9} {:>8} {:>9} {:>7}",
        "flux", "sigma_b", "answer", "reactive", "missing", "rated", "rounds"
    );
    let ceiling = 1.0 / year; // an atom transmutes at most once
    for flux in [1e12_f64, 1e14, 1e16] {
        for sigma_b in [0.1_f64, 1.0] {
            let rate = flux * sigma_b * 1e-24;
            let mut rated: HashSet<String> = seeds.keys().cloned().collect();
            let mut answer = HashSet::new();
            let mut rounds = 0;
            for _ in 0..64 {
                // Grow: unrated parents contribute nothing.
                loop {
                    rounds += 1;
                    let kept = yani::populated_nuclides(&chain, &seeds, year, FLOOR, |p, _| {
                        if rated.contains(p) {
                            rate
                        } else {
                            0.0
                        }
                    });
                    let grew = !kept.is_subset(&rated);
                    rated.extend(kept.iter().cloned());
                    if !grew {
                        break;
                    }
                }
                // Verify: unrated parents at the ceiling.
                rounds += 1;
                answer = yani::populated_nuclides(&chain, &seeds, year, FLOOR, |p, _| {
                    if rated.contains(p) {
                        rate
                    } else {
                        ceiling
                    }
                });
                let grew = !answer.is_subset(&rated);
                rated.extend(answer.iter().cloned());
                if !grew {
                    break;
                }
            }
            let direct = yani::populated_nuclides(&chain, &seeds, year, FLOOR, |_, _| rate);
            assert_eq!(
                answer, direct,
                "growing and verifying must land on the same set as rating everything"
            );
            println!(
                "{:>10.0e} {:>9.3} {:>8} {:>9} {:>8} {:>9} {:>7}",
                flux,
                sigma_b,
                answer.len(),
                reactive(&answer),
                missing(&answer),
                format!("{}/{}", rated.len(), reachable.len()),
                rounds,
            );
        }
    }
}
