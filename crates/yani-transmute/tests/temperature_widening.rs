//! Issue #481, yani side: a relabelled material silently activated nothing.
//!
//! `transmute_material` loads cross sections with a scope narrowed to the
//! material's temperature, but skips nuclides already present in
//! `nuclide_data`, so the narrowing never reaches them. Every rate lookup then
//! resolves `material.temperature()` through `reactions_for_temp`, which
//! answers `None` for a temperature that was never parsed in, and each call
//! site handles that with `return 0.0` or `continue`.
//!
//! The result was an irradiation that reported no activation, with no error --
//! quieter than the panic the CPU gave and the hard error the GPU gives.
//!
//! Needs a multi-temperature nuclide, and no such fixture is committed, so this
//! self-skips.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn fe56_cache() -> Option<String> {
    yamc_test_cache::nuclide("Fe56")
}

/// Fe56 loaded at 294 K, then relabelled, i.e. the state that used to activate
/// nothing.
fn fe56(cache: &str, relabel: Option<&str>) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), cache.to_string())]),
        None,
    )
    .expect("read Fe56");
    if let Some(t) = relabel {
        m.set_temperature(t);
    }
    m
}

/// Irradiate for a year at a high flux and report how much of the original
/// nuclide is left. Under a working rate calculation Fe56 burns measurably;
/// with every rate zero the composition cannot move at all, because Fe56 is
/// stable and the decay half of the step has nothing to act on.
fn fraction_remaining(m: &Material, chain: Arc<HashMap<String, yani::ChainNuclide>>) -> f64 {
    let spectra = [MultigroupSpectrum {
        boundaries: vec![1.0e-5, 2.0e7],
        masses: vec![1.0],
        relative_std_dev: None,
    }];
    let steps = [TransmuteStep {
        dt: 365.0 * 24.0 * 3600.0,
        irradiation: Some((0, 1.0e15)),
    }];
    let branch = HashMap::new();
    let out = transmute_material(
        &mut m.clone(),
        &spectra,
        &steps,
        chain,
        &branch,
        Default::default(),
        None,
    )
    .expect("transmute");
    let final_material = out
        .get_final_material(m.material_id.unwrap_or(0))
        .expect("one material per step");
    *final_material.nuclides.get("Fe56").unwrap_or(&0.0)
}

#[test]
fn a_relabelled_material_still_activates() {
    let Some(cache) = fe56_cache() else {
        eprintln!("a_relabelled_material_still_activates: skip (data missing)");
        return;
    };
    let chain = chain();

    let baseline = fraction_remaining(&fe56(&cache, None), Arc::clone(&chain));
    assert!(
        baseline < 1.0,
        "baseline: Fe56 at its loaded temperature must burn ({baseline})"
    );

    let relabelled = fe56(&cache, Some("900"));
    // Precondition: this really is the narrow state the bug needs. If loading
    // stops narrowing, this test should be rewritten rather than left passing
    // for the wrong reason.
    assert_eq!(
        relabelled.nuclide_data["Fe56"].loaded_temperatures,
        vec!["294".to_string()],
        "expected the narrow load that creates the bug"
    );

    let hot = fraction_remaining(&relabelled, chain);
    assert!(
        hot < 1.0,
        "a material relabelled to 900 K activated nothing: every reaction rate \
         came back zero, so the composition never moved ({hot})"
    );
}
