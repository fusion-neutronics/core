//! End-to-end check that a multi-nuclide model survives JSON
//! round-trip + runs through the wasm `simulate_transport`. Natural
//! lithium (Li-6 + Li-7) in the same 10 cm sphere geometry.
//!
//! Separate `tests/*.rs` file from the pure Li-6 case because cargo
//! parallelises tests within one binary; both tests call
//! `WasmSimulation::new()` which replaces the process-global storage,
//! so two tests in the same binary would race on which set of nuclide
//! files the global ends up holding.

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

fn natural_li_sphere_model() -> Model {
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

    // Natural lithium: 7.5 % Li-6, 92.5 % Li-7. Density 0.534 g/cm³.
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 0.075), ("Li7".into(), 0.925)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");

    let cell = Cell::new(Some(1), region, Some("li_sphere".into()), Some(0));
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
fn natural_lithium_model_round_trips_and_runs() {
    let model_json = serde_json::to_string(&natural_li_sphere_model()).expect("serialize");

    let sim = WasmSimulation::new();
    sim.load_model_json(model_json).expect("load_model_json");

    // The model knows which nuclides it needs.
    assert_eq!(sim.model_required_nuclides(), "Li6,Li7");

    load_fixture_into(&sim, "Li6");
    load_fixture_into(&sim, "Li7");

    let result_json = sim.simulate_transport(200, 5, 42);
    eprintln!("natural Li result: {result_json}");
    let parsed: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    assert_eq!(
        parsed["status"], "ok",
        "simulate_transport reported error: {result_json}"
    );

    let mean = parsed["tallies"][0]["scores"][0]["mean"].as_f64().unwrap();
    // Natural-Li tritium yield from 14 MeV neutrons -- mostly
    // Li-7(n,n′α)T + small Li-6(n,t)α share.
    assert!(mean > 1e-4, "natural Li tritium mean too low: {mean:.4e}");
    assert!(
        mean < 1e-1,
        "natural Li tritium mean implausibly high: {mean:.4e}"
    );
}
