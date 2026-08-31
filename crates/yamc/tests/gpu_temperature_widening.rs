//! Issue #481, GPU side: the GPU dispatch skips the CPU's material prep, so it
//! never picked up the temperature widening that prep does as a side effect.
//!
//! `read_nuclear_data` narrows the load to the material's temperature at the
//! time it runs. Relabelling afterwards (`material.temperature = 900`, which is
//! a Python setter) leaves the label naming reactions that were never parsed in.
//! `run_internal` recovers via `calculate_macroscopic_xs`; the GPU dispatch does
//! not run that block, so it reached `extract_material_xs` with the narrow data
//! and failed with `TemperatureNotLoaded`.
//!
//! Needs a multi-temperature nuclide, and no such fixture is committed, so this
//! self-skips. It needs no GPU: the assertion is on the material state the GPU
//! extractor reads, via the same `get_temp_idx` the extractor calls.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc_materials::Material;
use yamc_tallies::tally::Tally;

fn h1_cache() -> Option<String> {
    // Presence is an exists() check: a cache entry is a directory, so this
    // covers everything the old is_dir() did plus the legacy single-file entry.
    yamc_test_cache::nuclide("H1")
}

/// A material relabelled after its data was loaded, i.e. the state the GPU
/// dispatch used to choke on.
fn relabelled(cache: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("H1".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("H1 material");
    m.density = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("H1".to_string(), cache.to_string())]),
        None,
    )
    .expect("read H1");
    m.set_temperature("900");
    m
}

#[test]
fn the_gpu_extractor_finds_the_relabelled_temperature() {
    let Some(cache) = h1_cache() else {
        eprintln!("the_gpu_extractor_finds_the_relabelled_temperature: skip (data missing)");
        return;
    };

    let material = relabelled(&cache);

    // Precondition: this really is the narrow state. If loading ever stops
    // narrowing, this test is no longer exercising anything and should be
    // rewritten rather than left passing vacuously.
    assert_eq!(
        material.nuclide_data["H1"].loaded_temperatures,
        vec!["294".to_string()],
        "expected the narrow load that creates the bug"
    );
    assert_eq!(
        material.temperature(),
        "900",
        "and a label pointing outside it"
    );

    let mut material = material;
    material.set_material_id(1);
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("h".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let mut model = Model::new(geometry, vec![], Vec::<Arc<Tally>>::new());

    // Exactly what the GPU dispatch calls before translating.
    model
        .ensure_neutron_temperatures_for_gpu()
        .expect("widen to the labelled temperature");

    let m = &model.geometry.materials()[0];

    // What `extract_material_xs` does per nuclide
    // (yamc-gpu/src/neutron/xs/extract.rs:67 and :304): a miss here is
    // `NuclideXsError::TemperatureNotLoaded`.
    let idx = m.nuclide_data["H1"].get_temp_idx(m.temperature());
    assert!(
        idx.is_some(),
        "the GPU extractor would fail with TemperatureNotLoaded: label {:?} \
         is not in loaded_temperatures {:?}",
        m.temperature(),
        m.nuclide_data["H1"].loaded_temperatures
    );

    // And the energy grid the extractor pairs with it must exist too, since a
    // missing grid is a separate error variant (`MissingEnergyGrid`).
    let has_grid = m.nuclide_data["H1"]
        .energy
        .as_ref()
        .is_some_and(|e| e.contains_key(m.temperature()));
    assert!(has_grid, "no energy grid at the widened temperature");
}
