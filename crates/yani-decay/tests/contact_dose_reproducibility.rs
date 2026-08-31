//! A contact dose must be the same number twice running.
//!
//! `linear_attenuation` summed each element's partial mass density out of the
//! caller's `HashMap` in iteration order. An element with several isotopes --
//! iron in anything activated -- therefore had its bucket accumulated in an
//! order that is a property of the map INSTANCE rather than of its contents, so
//! two maps built from the same material walked differently, and so did two
//! calls that each built their own.
//!
//! `mu` divides every photon-line term, so that reached every per-nuclide dose
//! and the total. Measured before the fix, `Material.contact_dose()` returned
//! three distinct values across six processes, and three across eight calls in
//! a single process. Same defect as issue #502 and issue #576.
//!
//! The maps here are rebuilt from scratch on every iteration on purpose: a
//! fresh `HashMap` is the only way to get a fresh iteration order, so reusing
//! one would test nothing.

use std::collections::HashMap;

use yani::{ChainNuclide, DecaySource, DecaySourceDistribution};
use yani_decay::{contact_dose_by_nuclide, contact_dose_total, DoseQuantity};

/// How many independent builds to compare. Enough that a seed-dependent order
/// is overwhelmingly likely to differ at least once across them.
const BUILDS: usize = 25;

/// An inventory where TWO elements each carry several isotopes, so more than
/// one bucket is a multi-term sum. With one isotope per element every order is
/// the same order and the test would pass without the fix.
fn inventory() -> HashMap<String, f64> {
    HashMap::from([
        ("Fe54".to_string(), 4.9e-3),
        ("Fe55".to_string(), 1.1e-4),
        ("Fe56".to_string(), 7.7e-2),
        ("Fe57".to_string(), 1.8e-3),
        ("Fe58".to_string(), 2.4e-4),
        ("Co58".to_string(), 3.3e-6),
        ("Co60".to_string(), 7.1e-7),
        ("Mn54".to_string(), 5.2e-6),
        ("Mn56".to_string(), 9.4e-8),
    ])
}

/// A photon emitter for each nuclide that needs one, so the dose is non-zero.
/// The energies straddle the tabulations rather than sitting on a grid point.
fn chain() -> HashMap<String, ChainNuclide> {
    let mut chain = HashMap::new();
    for (name, lines) in [
        ("Co60", vec![(1_173_228.0, 0.9985), (1_332_492.0, 0.9998)]),
        ("Co58", vec![(810_759.0, 0.9945)]),
        ("Mn54", vec![(834_848.0, 0.9998)]),
        ("Mn56", vec![(846_771.0, 0.9885), (1_810_726.0, 0.2719)]),
        ("Fe55", vec![(5_900.0, 0.281)]),
    ] {
        let (energies, intensities): (Vec<f64>, Vec<f64>) = lines.into_iter().unzip();
        chain.insert(
            name.to_string(),
            ChainNuclide {
                name: name.to_string(),
                half_life: Some(1.0e6),
                half_life_uncertainty: None,
                decay_energy: 1.0e6,
                decay_energy_uncertainty: None,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies,
                        intensities,
                    },
                }],
            },
        );
    }
    chain
}

#[test]
fn the_total_is_bit_identical_across_builds() {
    let chain = chain();
    for quantity in [DoseQuantity::AbsorbedAir, DoseQuantity::Effective] {
        let first = contact_dose_total(&inventory(), &chain, quantity, 2.0).expect("dose");
        assert!(first > 0.0, "the fixture must produce a dose to compare");
        for i in 1..BUILDS {
            let again = contact_dose_total(&inventory(), &chain, quantity, 2.0).expect("dose");
            assert_eq!(
                again.to_bits(),
                first.to_bits(),
                "{quantity:?}: build {i} gave {again} where build 0 gave {first}"
            );
        }
    }
}

#[test]
fn every_nuclides_share_is_bit_identical_across_builds() {
    let chain = chain();
    let bits = |d: &HashMap<String, f64>| {
        let mut out: Vec<(String, u64)> = d.iter().map(|(n, v)| (n.clone(), v.to_bits())).collect();
        out.sort();
        out
    };

    let first = contact_dose_by_nuclide(&inventory(), &chain, DoseQuantity::AbsorbedAir, 2.0)
        .expect("dose");
    assert!(
        first.len() > 1,
        "more than one nuclide must contribute for this to mean anything"
    );
    let expected = bits(&first);
    for i in 1..BUILDS {
        let again = contact_dose_by_nuclide(&inventory(), &chain, DoseQuantity::AbsorbedAir, 2.0)
            .expect("dose");
        assert_eq!(bits(&again), expected, "build {i} differs from build 0");
    }
}

/// The property the failing Python tests were really asserting: the total is
/// the sum of the parts, whichever way it is reached.
#[test]
fn the_total_agrees_with_the_by_nuclide_breakdown() {
    let chain = chain();
    let total =
        contact_dose_total(&inventory(), &chain, DoseQuantity::AbsorbedAir, 2.0).expect("dose");
    let parts = contact_dose_by_nuclide(&inventory(), &chain, DoseQuantity::AbsorbedAir, 2.0)
        .expect("dose");
    assert_eq!(total.to_bits(), yani_decay::total(&parts).to_bits());
}
