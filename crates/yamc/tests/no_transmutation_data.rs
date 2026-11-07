//! A D1S decay-photon run reads the decay data and the reactions, and there
//! is no default library for either: with neither set the run is refused,
//! naming both, before any nuclear data is loaded. The material's own data is
//! not loaded and no cross-section library is set, so a refusal that came
//! after the data load would name the cross sections instead.
//!
//! In a file of its own because it needs the global configuration empty,
//! which a test sharing a binary with one that configures it could not rely
//! on.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;

#[test]
fn a_decay_photon_run_names_its_unset_sources_before_loading_data() {
    yamc_nuclide::config::Config::global().clear();
    let sphere = Arc::new(Surface::sphere(
        0.0,
        0.0,
        0.0,
        10.0,
        Some(1),
        Some(BoundaryType::Vacuum),
    ));
    let iron = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(sphere)),
        None,
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(iron)]).expect("geometry");
    let mut model = Model::new(geometry, vec![], vec![]);
    model.transport_secondary_photons = true;
    model.use_decay_photons = true;

    let err = model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10),
            ..Default::default()
        })
        .expect_err("no transmutation source is set");
    assert!(
        err.starts_with("use_decay_photons: no transmutation data is configured"),
        "{err}"
    );
    assert!(err.contains("transmutation_decay_data is not set"), "{err}");
    assert!(err.contains("transmutation_reactions is not set"), "{err}");
    // Fission yields are not read by D1S, so they are not asked for.
    assert!(!err.contains("transmutation_fission_yields"), "{err}");
}
