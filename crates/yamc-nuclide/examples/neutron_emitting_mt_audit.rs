//! Audit: enumerate every reaction MT that emits at least one neutron on
//! the GPU verification nuclide set, and flag which ones are NOT covered
//! by the canonical slot table (`INELASTIC_MT_SLOTS`, which yamc-gpu
//! re-exports as `MT_SLOTS`). Run with:
//!   cargo run -p yamc-nuclide --example neutron_emitting_mt_audit --release

use std::collections::BTreeMap;
use yamc_nuclide::nuclide::{load_nuclide, INELASTIC_MT_SLOTS as MT_SLOTS};
use yamc_nuclide::particle_type::ParticleType;

const NUCLIDES: &[&str] = &[
    "H1", "Li6", "Li7", "Be9", "B10", "C12", "N14", "O16", "F19", "Na23", "Al27", "Si28", "Cr52",
    "Fe56", "Ni58", "Cu63", "W186", "Pb208", "U235", "U238",
];

fn main() {
    // missing_mt -> (max_yield_at_14MeV, list of nuclides carrying it)
    let mut missing: BTreeMap<i32, (f64, Vec<String>)> = BTreeMap::new();

    for &nm in NUCLIDES {
        let path = yamc_test_cache::nuclide_path(nm);
        let nuclide = match load_nuclide(&path, &yamc_nuclide::LoadScope::full()) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("skip {nm}: {e}");
                continue;
            }
        };
        let Some(reactions) = nuclide.reactions.first() else {
            continue;
        };
        for (&mt, rxn) in reactions {
            // Skip fission (handled separately on GPU) and redundant
            // aggregates (MT 1/4/etc are summed, not sampled).
            if rxn.redundant {
                continue;
            }
            if matches!(mt, 18 | 19 | 20 | 21 | 38) {
                continue;
            }
            // First neutron product carries the multiplicity (mirror of
            // sample_from_products_with_awr).
            let Some(neutron) = rxn
                .products
                .iter()
                .find(|p| p.is_particle_type(&ParticleType::Neutron))
            else {
                continue;
            };
            // Yield at 14.06 MeV (the source energy in the verification scan).
            let y = neutron
                .product_yield
                .as_ref()
                .map(|c| c.evaluate(14.06e6))
                .unwrap_or(1.0);
            if y < 1e-10 {
                continue; // no neutron emitted at this energy -> absorption-like
            }
            if MT_SLOTS.contains(&mt) {
                continue;
            }
            let e = missing.entry(mt).or_insert((0.0, Vec::new()));
            if y > e.0 {
                e.0 = y;
            }
            e.1.push(nm.to_string());
        }
    }

    println!("\nMissing neutron-emitting MTs (not in GPU MT_SLOTS):");
    let header = format!("{:<5} {:<12} {:<8} nuclides", "MT", "yield@14MeV", "round");
    println!("{header}");
    for (mt, (y, nucs)) in &missing {
        let round = y.round() as i32;
        let nuc_list = nucs.join(", ");
        println!("{mt:<5} {y:<12.4} {round:<8} {nuc_list}");
    }
    if missing.is_empty() {
        println!("(none -- MT_SLOTS already covers every neutron-emitting MT)");
    }
}
