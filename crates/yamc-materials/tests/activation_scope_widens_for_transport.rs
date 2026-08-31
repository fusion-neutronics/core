//! A material transmuted first and transported second must still get its
//! transport data (issue #576, finding 3).
//!
//! `Material::transmute` now loads the cross sections it needs INTO the
//! material rather than into a private clone, so those entries survive the
//! call. They are activation-scope entries: the transmutation network's MTs
//! and none of the transport sections. `ensure_nuclides_loaded` used to skip
//! any nuclide already present in `nuclide_data`, whatever it was loaded for,
//! so transport would then look for an MT that is not there.
//!
//! This drives the loader directly rather than through a transmute, because
//! the property is about `Material`'s own loading and belongs beside it.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::{HashMap, HashSet};

use yamc_materials::material::Material;
use yamc_nuclide::load_scope::LoadScope;
use yamc_nuclide::nuclide::get_or_load_nuclide;

const NUCLIDE: &str = "Fe56";
/// (n,gamma) alone, the way an activation preload asks.
const ACTIVATION_MTS: [i32; 1] = [102];

fn iron(path: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    m.density = Some(7.87);
    // Registered so `ensure_nuclides_loaded` can resolve a path for it the way
    // it would for any configured material.
    let mut cfg = yamc_nuclide::config::CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    cfg.set_cross_section(NUCLIDE, Some(path));
    drop(cfg);
    m
}

#[test]
fn an_activation_scope_entry_is_widened_rather_than_taken_as_loaded() {
    let Some(path) = yamc_test_cache::nuclide(NUCLIDE) else {
        eprintln!("skipping: {NUCLIDE} fixture missing");
        return;
    };
    let mut material = iron(&path);

    // What a transmutation preload leaves behind.
    let scope = LoadScope::activation(HashSet::from(ACTIVATION_MTS));
    let sources = HashMap::from([(NUCLIDE.to_string(), path.clone())]);
    let preloaded = get_or_load_nuclide(NUCLIDE, &sources, &scope).expect("activation load");
    assert!(
        preloaded.load_scope.mts.is_some(),
        "the fixture must give an MT-filtered entry for this test to mean anything"
    );
    material.nuclide_data.insert(NUCLIDE.to_string(), preloaded);

    material
        .ensure_nuclides_loaded()
        .expect("widen for transport");

    let after = &material.nuclide_data[NUCLIDE];
    assert!(
        after.load_scope.mts.is_none(),
        "an activation-scope entry must be widened, not taken as already loaded: \
         still {:?}",
        after.load_scope
    );
}

/// And the converse: a nuclide already loaded for transport is not re-read.
///
/// A trimmed data directory that carries none of the transport sections is
/// recorded as `XsOnly` even when the caller asked for everything, so testing
/// `SectionScope::Full` here would re-read such a nuclide on every call.
#[test]
fn a_fully_loaded_nuclide_is_left_alone() {
    let Some(path) = yamc_test_cache::nuclide(NUCLIDE) else {
        eprintln!("skipping: {NUCLIDE} fixture missing");
        return;
    };
    let mut material = iron(&path);
    material
        .read_nuclear_data(&HashMap::from([(NUCLIDE.to_string(), path.clone())]), None)
        .expect("read");

    let before = std::sync::Arc::as_ptr(&material.nuclide_data[NUCLIDE]);
    material.ensure_nuclides_loaded().expect("no-op");
    let after = std::sync::Arc::as_ptr(&material.nuclide_data[NUCLIDE]);
    assert_eq!(
        before, after,
        "a nuclide loaded as fully as its directory allows must not be re-read"
    );
}
