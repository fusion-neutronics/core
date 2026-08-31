//! End-to-end check of the wasm-bindgen `WasmSimulation` path on a
//! native target.
//!
//! Builds a Li-6 sphere `Model` in Rust, serializes it to JSON, then
//! drives the wasm-bindgen entry points exactly as JS would:
//!
//!     sim.load_model_json(model_json)
//!     sim.add_file("/Li6.arrow/<file>", bytes)   // per file
//!     sim.simulate_transport(particles, batches, seed)
//!
//! Runs as part of `cargo test -p yamc --features wasm-test`. The test
//! installs a fresh `InMemoryStorage` as the process-global; don't add
//! anything to this file that expects `NativeStorage`-style filesystem
//! reads -- they'd be redirected.

#![cfg(feature = "wasm")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
use yamc::wasm::simulation_wasm::WasmSimulation;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mt::Mt;
use yamc_tallies::score::{ReactionRateScore, Score};
use yamc_tallies::tally::Tally;

fn li6_sphere_model() -> Model {
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

    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");

    let cell = Cell::new(Some(1), region, Some("li6_sphere".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    tally.scores = vec![Score::ReactionRate(ReactionRateScore::from_mt(
        Mt::try_from(105).unwrap(),
    ))];
    tally.name = Some("tritium_production".into());

    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

fn load_fixture_into(sim: &WasmSimulation, nuclide: &str) {
    let dir = std::path::PathBuf::from(format!("tests/{nuclide}.arrow"));
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
        let entry = entry.unwrap();
        let bytes = std::fs::read(entry.path()).unwrap();
        let virtual_path = format!("/{nuclide}.arrow/{}", entry.file_name().to_string_lossy());
        sim.add_file(virtual_path, bytes);
    }
}

#[test]
fn li6_sphere_runs_via_loaded_model_json() {
    // Build the model in Rust, serialize to JSON -- this is the same
    // step `model.save()` / `model.export()` will do in Python.
    let model_json = serde_json::to_string(&li6_sphere_model()).expect("serialize");

    // Drive WasmSimulation exactly as the JS host would.
    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).expect("load_model_json");
    load_fixture_into(&sim, "Li6");

    // `model_required_nuclides` is the JS-side hook that tells the host
    // which `<Nuclide>.arrow/` section sets to fetch.
    assert_eq!(sim.model_required_nuclides(), "Li6");

    let result_json = sim.simulate_transport(200, 5, 42);
    eprintln!("loaded-model result: {result_json}");
    let parsed: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    assert_eq!(
        parsed["status"], "ok",
        "simulate_transport reported error: {result_json}"
    );

    // New JSON shape: tallies[0].scores[0].{mean,std}.
    let scores = &parsed["tallies"][0]["scores"];
    assert_eq!(scores.as_array().unwrap().len(), 1, "expected one score");
    let mean = scores[0]["mean"].as_f64().unwrap();
    let std = scores[0]["std"].as_f64().unwrap();
    // 14 MeV neutrons into 10 cm of Li-6 at 0.46 g/cc -- ~3 × 10⁻²
    // tritium reactions per source neutron at the configured sample size.
    assert!(mean > 1e-3, "tritium mean too low: {mean:.4e}");
    assert!(mean < 1e-1, "tritium mean implausibly high: {mean:.4e}");
    assert!(std > 0.0, "tritium std must be positive, got {std:.4e}");
}
