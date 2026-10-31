//! The smallest model the wasm tests can run end to end: a 10 cm Li-6 sphere
//! with a 14 MeV point source at its centre and a tritium-production tally.
//!
//! Shared by the native `WasmSimulation` tests and the browser test, each of
//! which is its own crate and pulls this in with `#[path]`, so that the model
//! the browser fetches data for is the one the native tests already know the
//! answer to. Not part of `common/mod.rs`: that module is for the transport
//! tests and drags in builders the wasm32 test target does not need.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::Model;
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

/// Build the model with its one material at `temperature` (a bare Kelvin label,
/// `"294"`).
pub fn li6_sphere_model(temperature: &str) -> Model {
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
    material.set_temperature(temperature);

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

/// The model as the JSON `model.save()` writes, which is what
/// `WasmSimulation.load_model_json` takes.
pub fn li6_sphere_model_json(temperature: &str) -> String {
    serde_json::to_string(&li6_sphere_model(temperature)).expect("serialize the model")
}

/// Check a `simulate_transport` result JSON: status ok, and a tritium
/// production per source neutron in the range 14 MeV neutrons into 10 cm of
/// Li-6 at 0.46 g/cm3 give at a few hundred histories.
pub fn assert_tritium_result(result_json: &str) {
    let parsed: serde_json::Value = serde_json::from_str(result_json).unwrap();
    assert_eq!(
        parsed["status"], "ok",
        "simulate_transport reported error: {result_json}"
    );
    let scores = &parsed["tallies"][0]["scores"];
    assert_eq!(scores.as_array().unwrap().len(), 1, "expected one score");
    let mean = scores[0]["mean"].as_f64().unwrap();
    let std = scores[0]["std"].as_f64().unwrap();
    assert!(mean > 1e-3, "tritium mean too low: {mean:.4e}");
    assert!(mean < 1e-1, "tritium mean implausibly high: {mean:.4e}");
    assert!(std > 0.0, "tritium std must be positive, got {std:.4e}");
}
