//! Changing a material's temperature or density on an existing model, then
//! running again, gives the answer of a model built at the new state -- the
//! coupling loop of a thermal-hydraulic feedback iteration -- and leaves the
//! other materials and the loaded nuclear data alone.
//!
//! Data: endf-b8.1 Fe56 in the fixture cache. Self-skips when absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};

fn fe56(path: &str, temperature: &str, density: f64) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_temperature(temperature);
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), path.to_string())]),
        None,
    )
    .unwrap();
    m
}

fn sphere(id: usize, radius: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary,
        name: None,
    })
}

/// Two concentric iron shells (inner r < 5, outer 5 < r < 10, each its own
/// material) around a 14.06 MeV point source, with a flux tally per cell.
fn shells(inner: Material, outer: Material) -> Model {
    let s1 = sphere(1, 5.0, BoundaryType::Transmission);
    let s2 = sphere(2, 10.0, BoundaryType::Vacuum);
    let inner_region = Region::new_from_halfspace(HalfspaceType::Below(s1.clone()));
    let outer_region = Region::new_from_halfspace(HalfspaceType::Above(s1))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(s2)));
    let cells = vec![
        Cell::new(Some(1), inner_region, Some("inner".into()), Some(0)),
        Cell::new(Some(2), outer_region, Some("outer".into()), Some(1)),
    ];
    let geometry = Geometry::new(cells, vec![Arc::new(inner), Arc::new(outer)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut tally = Tally::new();
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.filters = vec![Filter::Cell(CellFilter::from_ids(vec![1, 2]))];
    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

fn run(model: &mut Model) -> Vec<f64> {
    let settings = TransportSettings {
        total_particles: Some(2000),
        seed: 7,
        ..Default::default()
    };
    model.simulate_transport(&settings).unwrap();
    model.tallies[0].get_mean()
}

/// Same histories, so equal up to the thread-order rounding of the reduction;
/// a physics difference would show at the statistical level, ~1e-2 here.
fn assert_same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() <= 1e-12 * x.abs().max(y.abs()),
            "{a:?} != {b:?}"
        );
    }
}

fn differs(a: &[f64], b: &[f64]) -> bool {
    a.iter()
        .zip(b)
        .any(|(x, y)| (x - y).abs() > 1e-9 * x.abs().max(y.abs()))
}

fn id(model: &Model, slot: usize) -> u32 {
    model.geometry.materials()[slot].get_material_id().unwrap()
}

#[test]
fn a_new_temperature_gives_the_fresh_model_answer() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let mut model = shells(fe56(&path, "294", 7.8), fe56(&path, "294", 7.8));
    let before = run(&mut model);
    let outer_before = Arc::clone(&model.geometry.materials()[1]);

    model
        .set_material_temperature(id(&model, 0), "900")
        .unwrap();
    let updated = run(&mut model);
    assert!(
        differs(&before, &updated),
        "the change must reach transport"
    );

    let mut fresh = shells(fe56(&path, "900", 7.8), fe56(&path, "294", 7.8));
    assert_same(&updated, &run(&mut fresh));
    assert_eq!(model.geometry.materials()[0].temperature(), "900");
    assert!(
        Arc::ptr_eq(&outer_before, &model.geometry.materials()[1]),
        "the untouched material must keep its prepared cross sections"
    );
}

#[test]
fn a_new_density_gives_the_fresh_model_answer_without_reloading_data() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let mut model = shells(fe56(&path, "294", 7.8), fe56(&path, "294", 7.8));
    let before = run(&mut model);
    let data_before = Arc::clone(&model.geometry.materials()[1].nuclide_data["Fe56"]);

    model.set_material_density(id(&model, 1), 3.9).unwrap();
    let updated = run(&mut model);
    assert!(
        differs(&before, &updated),
        "the change must reach transport"
    );

    let mut fresh = shells(fe56(&path, "294", 7.8), fe56(&path, "294", 3.9));
    assert_same(&updated, &run(&mut fresh));
    assert!(
        Arc::ptr_eq(
            &data_before,
            &model.geometry.materials()[1].nuclide_data["Fe56"]
        ),
        "a density change must not reload nuclear data"
    );
}

#[test]
fn bad_requests_are_errors() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let mut model = shells(fe56(&path, "294", 7.8), fe56(&path, "294", 7.8));
    let missing = u32::MAX - 1;
    assert!(model
        .set_material_temperature(missing, "600")
        .unwrap_err()
        .contains(&missing.to_string()));
    assert!(model.set_material_density(missing, 1.0).is_err());
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(model.set_material_density(id(&model, 0), bad).is_err());
    }
}
