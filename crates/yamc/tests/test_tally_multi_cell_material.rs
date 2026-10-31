//! End-to-end scoring tests for multi-cell and multi-material tally filters.
//!
//! These assert the headline guarantee: a single tally binning over `cells = [c1, c2]`
//! produces the same per-bin means as two independent single-cell tallies (same RNG,
//! same geometry, deterministic scoring path).

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::{CellFilter, MaterialFilter};

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

fn inside(surf: &Arc<Surface>) -> RegionExpr {
    RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
        surf.clone(),
    ))))
}

fn outside(surf: &Arc<Surface>) -> RegionExpr {
    RegionExpr::Halfspace(HalfspaceType::Above(surf.clone()))
}

fn make_material(id: u32) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.46),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
    m.read_nuclear_data(&nuclide_map, None).unwrap();
    Arc::new(m)
}

fn make_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

/// Two nested spherical shells sharing one material. Returns (geometry, cell_id_1, cell_id_2).
fn two_cell_geometry(material: Arc<Material>) -> (Geometry, u32, u32) {
    let s1 = sphere(1, 1.0, BoundaryType::Transmission);
    let s2 = sphere(2, 5.0, BoundaryType::Vacuum);

    let c1 = Cell::new(
        Some(1),
        Region { expr: inside(&s1) },
        Some("inner".to_string()),
        Some(0),
    );
    let c2 = Cell::new(
        Some(2),
        Region {
            expr: RegionExpr::Intersection(Box::new(outside(&s1)), Box::new(inside(&s2))),
        },
        Some("outer".to_string()),
        Some(0),
    );

    let geometry = Geometry::new(vec![c1, c2], vec![material]).unwrap();
    (geometry, 1, 2)
}

/// A single-cell flux tally filtered on one cell_id.
fn single_cell_tally(cell_id: u32) -> Tally {
    let mut t = Tally::new();
    // Filters must be pushed before `set_scores_mixed` so that storage
    // is sized to the final `num_bins()`.
    t.filters = vec![Filter::Cell(CellFilter {
        cell_ids: vec![cell_id],
    })];
    t.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    t
}

/// A flux tally binning over a list of cells.
fn multi_cell_tally(cell_ids: Vec<u32>) -> Tally {
    let mut t = Tally::new();
    t.filters = vec![Filter::Cell(CellFilter { cell_ids })];
    t.set_scores_mixed(vec![Score::Flux(FluxScore)]);
    t
}

#[test]
fn multi_cell_tally_matches_two_single_cell_tallies() {
    let material = make_material(1);
    let (geometry, c1_id, c2_id) = two_cell_geometry(material);

    let multi = multi_cell_tally(vec![c1_id, c2_id]);
    let single_c1 = single_cell_tally(c1_id);
    let single_c2 = single_cell_tally(c2_id);

    let mut model = Model::new(
        geometry,
        vec![make_source()],
        vec![Arc::new(multi), Arc::new(single_c1), Arc::new(single_c2)],
    );
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(2000 * 3),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let multi_result = &model.tallies[0];
    let c1_result = &model.tallies[1];
    let c2_result = &model.tallies[2];

    let multi_mean = multi_result.get_mean();
    let c1_mean = c1_result.get_mean()[0];
    let c2_mean = c2_result.get_mean()[0];

    assert_eq!(multi_mean.len(), 2, "multi-cell tally must expose 2 bins");

    // Same RNG seed, identical scoring logic -- bins must match bitwise
    // (or at least within f64 round-off from reassociated summation).
    assert!(
        (multi_mean[0] - c1_mean).abs() <= 1e-12 * c1_mean.abs().max(1e-30),
        "bin 0 ({}) vs single-c1 tally ({})",
        multi_mean[0],
        c1_mean
    );
    assert!(
        (multi_mean[1] - c2_mean).abs() <= 1e-12 * c2_mean.abs().max(1e-30),
        "bin 1 ({}) vs single-c2 tally ({})",
        multi_mean[1],
        c2_mean
    );

    // And both bins must be non-trivial (flux is being scored).
    assert!(c1_mean > 0.0);
    assert!(c2_mean > 0.0);
}

#[test]
fn multi_material_tally_matches_two_single_material_tallies() {
    let m1 = make_material(1);
    let m2 = make_material(2);

    // Two separate materials, one per shell.
    let s1 = sphere(1, 1.0, BoundaryType::Transmission);
    let s2 = sphere(2, 5.0, BoundaryType::Vacuum);

    let c1 = Cell::new(
        Some(1),
        Region { expr: inside(&s1) },
        Some("inner".to_string()),
        Some(0),
    );
    let c2 = Cell::new(
        Some(2),
        Region {
            expr: RegionExpr::Intersection(Box::new(outside(&s1)), Box::new(inside(&s2))),
        },
        Some("outer".to_string()),
        Some(1),
    );
    let geometry = Geometry::new(vec![c1, c2], vec![m1, m2]).unwrap();

    let mut multi = Tally::new();
    multi.filters = vec![Filter::Material(MaterialFilter {
        material_ids: vec![1, 2],
    })];
    multi.set_scores_mixed(vec![Score::Flux(FluxScore)]);

    let mut single_m1 = Tally::new();
    single_m1.filters = vec![Filter::Material(MaterialFilter {
        material_ids: vec![1],
    })];
    single_m1.set_scores_mixed(vec![Score::Flux(FluxScore)]);

    let mut single_m2 = Tally::new();
    single_m2.filters = vec![Filter::Material(MaterialFilter {
        material_ids: vec![2],
    })];
    single_m2.set_scores_mixed(vec![Score::Flux(FluxScore)]);

    let mut model = Model::new(
        geometry,
        vec![make_source()],
        vec![Arc::new(multi), Arc::new(single_m1), Arc::new(single_m2)],
    );
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(2000 * 3),
            seed: 42,
            ..Default::default()
        })
        .unwrap();

    let multi_mean = model.tallies[0].get_mean();
    let m1_mean = model.tallies[1].get_mean()[0];
    let m2_mean = model.tallies[2].get_mean()[0];

    assert_eq!(multi_mean.len(), 2);
    assert!(
        (multi_mean[0] - m1_mean).abs() <= 1e-12 * m1_mean.abs().max(1e-30),
        "material bin 0 ({}) vs single-m1 tally ({})",
        multi_mean[0],
        m1_mean
    );
    assert!(
        (multi_mean[1] - m2_mean).abs() <= 1e-12 * m2_mean.abs().max(1e-30),
        "material bin 1 ({}) vs single-m2 tally ({})",
        multi_mean[1],
        m2_mean
    );
    assert!(m1_mean > 0.0);
    assert!(m2_mean > 0.0);
}
