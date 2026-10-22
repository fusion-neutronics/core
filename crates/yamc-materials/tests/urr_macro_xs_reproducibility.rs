//! The URR-perturbed macroscopic cross sections must be bit-reproducible.
//!
//! The same defect `macro_xs_reproducibility.rs` guards the smooth tables
//! against, in the per-step sum the URR reaction-rate scores and the Woodcock
//! rejection read. `Material::compute_urr_macro_xs` walked `nuclide_data`, a
//! `HashMap` seeded per instance, and every nuclide accumulated into the same
//! five sums, so the last bit of each depended on the map's iteration order
//! and moved between runs. It now sums in name order.
//!
//! Rebuilt rather than cloned: `Material::new` gives every material a fresh
//! `nuclide_data` map with its own hasher, while a cloned material's map
//! iterates in its source's order, which would hold the order fixed and let
//! the unfixed code pass.

use std::collections::HashMap;

use yamc_materials::Material;

/// The fixture-set steel of `macro_xs_reproducibility.rs`: eight nuclides
/// spanning four orders of magnitude in fraction, with Fe58's probability
/// tables covering 350 keV to 3 MeV.
fn steel() -> HashMap<String, f64> {
    HashMap::from(
        [
            ("Fe54", 0.0380),
            ("Fe56", 0.5966),
            ("Fe57", 0.0138),
            ("Fe58", 0.0018),
            ("Cr52", 0.1423),
            ("Si28", 0.0092),
            ("Si29", 0.0005),
            ("Si30", 0.0003),
        ]
        .map(|(n, f)| (n.to_string(), f)),
    )
}

fn build(paths: &HashMap<String, String>) -> Material {
    let mut m = Material::new(steel(), "atom", "g/cm3", Some(7.9)).expect("material");
    m.set_temperature("294");
    m.read_nuclear_data(paths, None).expect("nuclear data");
    m.calculate_macroscopic_xs(&vec![1, 2, 18, 101, 102], true);
    m
}

/// Every field of every sample, as raw bits.
fn samples(material: &Material) -> Vec<Option<[u64; 5]>> {
    let mut out = Vec::new();
    for energy in [1.0e6, 2.0e6] {
        for i in 0..200 {
            let base = (i as f64 + 0.5) / 200.0;
            out.push(material.compute_urr_macro_xs(energy, base).map(|u| {
                [u.total, u.elastic, u.fission, u.capture, u.absorption].map(f64::to_bits)
            }));
        }
    }
    out
}

#[test]
fn the_urr_macroscopic_cross_sections_are_bit_identical_across_builds() {
    let Some(paths) = steel()
        .keys()
        .map(|n| yamc_test_cache::nuclide(n).map(|p| (n.clone(), p)))
        .collect::<Option<HashMap<String, String>>>()
    else {
        eprintln!("SKIP: the steel nuclides are not in the test cache; this test checked nothing");
        return;
    };

    let expected = samples(&build(&paths));
    assert!(
        expected.iter().all(Option::is_some),
        "Fe58's tables cover 1 and 2 MeV, so every sample is perturbed"
    );
    for i in 1..8 {
        let got = samples(&build(&paths));
        let differing = expected.iter().zip(&got).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{differing} of {} samples differ on build {i}",
            expected.len()
        );
    }
}
