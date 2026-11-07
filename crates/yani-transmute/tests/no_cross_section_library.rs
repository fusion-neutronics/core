//! Irradiating a material with no neutron cross-section library configured is
//! refused, naming the setting, rather than solved as pure decay.
//!
//! The material's own nuclides with no cross sections cannot activate at all,
//! and before this the solve returned the decay of an unirradiated material,
//! zero activity, without a word. In a file of its own because it needs the
//! global configuration empty, which a test sharing a binary with one that
//! configures it could not rely on.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let nuclide = |name: &str, reactions: Vec<yani::ChainReaction>| yani::ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_evaluated: true,
        excitation_energy: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: Vec::new(),
        fission_yields: None,
        sources: Vec::new(),
    };
    let capture = yani::ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("Fe57".to_string()),
        branching: 1.0,
        q_value: Some(0.0),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    Arc::new(HashMap::from([
        ("Fe56".to_string(), nuclide("Fe56", vec![capture])),
        ("Fe57".to_string(), nuclide("Fe57", Vec::new())),
    ]))
}

fn run(irradiation: Option<(usize, f64)>) -> Result<(), String> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.density = Some(7.87);
    m.set_temperature("294");
    transmute_material(
        &mut m,
        &[MultigroupSpectrum {
            boundaries: vec![1.0e-5, 1.0e6, 2.0e7],
            masses: vec![0.5, 0.5],
            flux_error: None,
        }],
        &[TransmuteStep {
            dt: 3600.0,
            irradiation,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        None,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn an_irradiation_with_no_library_is_refused_and_a_cooldown_is_not() {
    yamc_nuclide::config::CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
    let err = run(Some((0, 1.0e14))).expect_err("no library, nothing to activate");
    assert!(
        err.contains("no neutron cross-section library is configured"),
        "{err}"
    );
    assert!(
        err.contains("cross_section_data"),
        "must name the setting: {err}"
    );
    assert!(err.contains("'Fe56'"), "must name the nuclide: {err}");

    // A cooldown reads no cross section, so it needs no library.
    run(None).expect("a decay-only schedule needs no cross sections");
}
