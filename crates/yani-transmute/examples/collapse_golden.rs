//! The collapse, printed bit for bit, so a speed change can be shown not to
//! have moved a number.
//!
//! Run as: `cargo run --release -p yani-transmute --example collapse_golden`
//!
//! Issue #576 rewrites the multigroup collapse for speed under a hard
//! "no accuracy loss" constraint. Decimal printing hides the last bits, which
//! is exactly what such a claim is about, so this prints `f64::to_bits` in hex
//! -- stable Rust has no hex-float format, and the bit pattern is the same
//! statement -- and the pass condition is that the output is byte-identical
//! before and after:
//!
//! ```text
//! cargo run --release -p yani-transmute --example collapse_golden > /tmp/before.txt
//! # ... change the collapse ...
//! cargo run --release -p yani-transmute --example collapse_golden > /tmp/after.txt
//! diff /tmp/before.txt /tmp/after.txt
//! ```
//!
//! What it covers, all three of the things the collapse produces:
//!
//! * the per-group terms (`per_group_reaction_rates`), which are `sigma_g`
//!   scaled by two exact-in-the-last-bit factors, so a group average that moves
//!   shows here group by group rather than washed out in a sum;
//! * the one-group rates, which are the flux-weighted sum of those, and so the
//!   guard on the accumulation order as well as on the values;
//! * the fission-yield weights, which come from the second, finer point set in
//!   `add_group_fission_xs_by_yield_point`.
//!
//! Skips when the nuclear-data fixtures are not cached, the way the test suites
//! do. `python scripts/fetch_test_fixtures.py` fetches them.

use std::collections::HashMap;
use std::path::PathBuf;

use yamc_materials::Material;
use yani_transmute::multigroup::per_group_reaction_rates;

/// CCFE-709 is where the collapse cost lives, and where a point-set change has
/// the most chances to differ.
const GROUPS: &str = "CCFE-709";

/// The nuclides the golden is taken over. Structural steel, plus a fissionable
/// one so the yield fold is exercised rather than skipped.
const NUCLIDES: [(&str, f64); 6] = [
    ("Fe54", 0.0585),
    ("Fe56", 0.9175),
    ("Fe57", 0.0212),
    ("Fe58", 0.0028),
    ("Cr52", 0.8379),
    ("U240", 1.0e-6),
];

fn main() {
    let boundaries = match yamc_nuclide::group_structures::get_group_structure(GROUPS) {
        Ok(b) => b.to_vec(),
        Err(e) => {
            eprintln!("no {GROUPS} group structure: {e}");
            return;
        }
    };
    let flux = fusion_spectrum(&boundaries);

    let mut paths = HashMap::new();
    for (name, _) in NUCLIDES {
        match yamc_test_cache::nuclide(name) {
            Some(path) => {
                paths.insert(name.to_string(), path);
            }
            None => {
                eprintln!("skipping: {name} is not cached (nuclear-data fixtures missing)");
                return;
            }
        }
    }

    let mut material = Material::new(
        NUCLIDES
            .iter()
            .map(|(n, f)| (n.to_string(), *f))
            .collect::<HashMap<_, _>>(),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    material.density = Some(7.9);
    material.set_temperature("294");
    material
        .read_nuclear_data(&paths, None)
        .expect("read nuclear data");

    let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = yani::parse_chain_arrow(&chain_path).expect("parse chain");

    let total: f64 = flux.iter().sum();
    let masses: Vec<f64> = flux.iter().map(|f| f / total).collect();

    // Per group, so a single group average that moves is visible where it moved.
    let per_group = per_group_reaction_rates(&material, &chain, &masses, &boundaries, None);
    let mut nuclides: Vec<&String> = per_group.keys().collect();
    nuclides.sort();
    for name in nuclides {
        let mut kinds: Vec<&String> = per_group[name].keys().collect();
        kinds.sort();
        for kind in kinds {
            for (g, term) in per_group[name][kind].iter().enumerate() {
                println!("term {name} {kind} {g} {:#018x}", term.to_bits());
            }
        }
    }

    // And the sums, which are also the guard on the accumulation order.
    let (rates, fy_weights) = yani_transmute::compute_multigroup_reaction_rates(
        &material,
        &chain,
        &masses,
        &boundaries,
        1.0,
    );
    let mut names: Vec<&String> = rates.keys().collect();
    names.sort();
    for name in names {
        let mut kinds: Vec<&String> = rates[name].keys().collect();
        kinds.sort();
        for kind in kinds {
            println!("rate {name} {kind} {:#018x}", rates[name][kind].to_bits());
        }
    }

    let mut names: Vec<&String> = fy_weights.keys().collect();
    names.sort();
    for name in names {
        for (k, w) in fy_weights[name].iter().enumerate() {
            println!("fy {name} {k} {:#018x}", w.to_bits());
        }
    }
}

/// A 1/E slowing-down tail with a 14 MeV peak, the same shape
/// `tools/bench_transmute.py` times against. Every group carries flux, so
/// nothing in the collapse can be skipped as a zero.
fn fusion_spectrum(boundaries: &[f64]) -> Vec<f64> {
    boundaries
        .windows(2)
        .map(|w| {
            let mid = (w[0].max(1.0e-5) * w[1]).sqrt();
            let value = (w[1] - w[0]) / mid;
            if (1.3e7..1.5e7).contains(&mid) {
                value + 50.0
            } else {
                value
            }
        })
        .collect()
}
