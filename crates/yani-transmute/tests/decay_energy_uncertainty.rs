//! Decay-energy uncertainty on decay heat (issue #140, item 2).
//!
//! Co60 cooling on its own, with sigmas placed on its beta and gamma energies.
//! A decay energy does not enter the Bateman matrix, so the inventory, and with
//! it the activity, must not move at all, while the decay heat moves by
//! exactly the components' quadrature over the total:
//! `sqrt(sigma_beta^2 + sigma_gamma^2) / (E_beta + E_gamma)`.
//!
//! Self-skips when the Fe56 fixture (used only to load the material's data
//! path) is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, TransmuteStep};

const BETA: (f64, f64) = (0.096e6, 0.096e6 * 0.10);
const GAMMA: (f64, f64) = (2.503e6, 2.503e6 * 0.05);

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let co60 = chain.get_mut("Co60").expect("Co60");
    co60.decay_energy = BETA.0 + GAMMA.0;
    co60.decay_energy_components = [
        Some(yani::DecayEnergyComponent {
            energy: BETA.0,
            uncertainty: Some(BETA.1),
        }),
        Some(yani::DecayEnergyComponent {
            energy: GAMMA.0,
            uncertainty: Some(GAMMA.1),
        }),
        Some(yani::DecayEnergyComponent {
            energy: 0.0,
            uncertainty: None,
        }),
    ];
    Arc::new(chain)
}

fn cobalt() -> Material {
    let mut m = Material::new(
        HashMap::from([("Co60".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Co60 material");
    m.nuclides.insert("Co60".to_string(), 1.0e-6);
    m.volume = Some(1.0);
    m
}

#[test]
fn decay_energy_moves_the_heat_and_nothing_else() {
    let chain = chain();
    let mut material = cobalt();
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: None,
    }];
    let request = DataUncertainty {
        seed: 6,
        samples: Some(4096),
        sources: vec![Source::DecayEnergy],
        attribution: false,
    };
    let results = transmute_material(
        &mut material,
        &[],
        &steps,
        Arc::clone(&chain),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute");

    let info = &results.uncertainty_info[&0];
    assert!(info.decay_energies_perturbed.contains("Co60"), "{info:?}");
    assert!(!info.not_perturbed.iter().any(|s| s == "decay energy"));

    let heat = results
        .decay_heat_uncertainty(0, 1, &chain)
        .unwrap()
        .unwrap();
    let expected = (BETA.1 * BETA.1 + GAMMA.1 * GAMMA.1).sqrt() / (BETA.0 + GAMMA.0);
    let got = heat.relative_std_dev().expect("a spread");
    assert!(
        (got / expected - 1.0).abs() < 0.05,
        "decay heat moves by the components' quadrature: {got:.4} against {expected:.4}"
    );
    // The inventory never saw a decay energy.
    let activity = results.activity_uncertainty(0, 1, &chain).unwrap().unwrap();
    assert_eq!(activity.std_dev, Some(0.0));
}

/// A sigma the data states but no draw can carry is reported as a gap of its
/// own: held at nominal, and never listed as perturbed or as stating none.
#[test]
fn a_sigma_no_draw_can_carry_is_reported_as_a_gap() {
    let mut chain = Arc::unwrap_or_clone(chain());
    let co60 = chain.get_mut("Co60").expect("Co60");
    co60.half_life_uncertainty = Some(f64::INFINITY);
    // One component with an infinite sigma, one with a sigma on a zero
    // energy: neither can be drawn, so the nuclide has nothing to perturb.
    co60.decay_energy_components = [
        Some(yani::DecayEnergyComponent {
            energy: BETA.0,
            uncertainty: Some(f64::INFINITY),
        }),
        Some(yani::DecayEnergyComponent {
            energy: 0.0,
            uncertainty: Some(GAMMA.1),
        }),
        None,
    ];
    let chain = Arc::new(chain);
    let mut material = cobalt();
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: None,
    }];
    let request = DataUncertainty {
        seed: 6,
        samples: Some(8),
        sources: vec![Source::HalfLife, Source::DecayEnergy],
        attribution: false,
    };
    let results = transmute_material(
        &mut material,
        &[],
        &steps,
        Arc::clone(&chain),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute");

    let info = &results.uncertainty_info[&0];
    assert!(
        info.half_life_uncertainty_not_carried.contains("Co60"),
        "{info:?}"
    );
    assert!(!info.half_lives_perturbed.contains("Co60"));
    assert!(!info.no_half_life_uncertainty.contains("Co60"));
    assert!(
        info.decay_energy_uncertainty_not_carried.contains("Co60"),
        "{info:?}"
    );
    assert!(!info.decay_energies_perturbed.contains("Co60"));
    assert!(!info.no_decay_energy_uncertainty.contains("Co60"));
    assert!(info.has_gaps());

    // Nothing was drawn, so neither the heat nor the activity has a spread:
    // with no replica to solve the ensemble has none to measure one from.
    let heat = results
        .decay_heat_uncertainty(0, 1, &chain)
        .unwrap()
        .unwrap();
    assert!(heat.std_dev.is_none_or(|s| s == 0.0), "{heat:?}");
    let activity = results.activity_uncertainty(0, 1, &chain).unwrap().unwrap();
    assert!(activity.std_dev.is_none_or(|s| s == 0.0), "{activity:?}");
}
