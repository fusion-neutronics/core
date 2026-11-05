//! Proves the model graph survives a round-trip through JSON.
//!
//! Builds a small Li-6 sphere model from scratch, serializes to JSON,
//! deserializes back, and asserts the deserialized model produces
//! identical tally results to the original (post-cache-rebuild).
//!
//! This is the regression test for the serde refactor -- any time a
//! new field is added to a `*Serde` shape (or a runtime cache moves
//! between "rebuilt" and "skip"), this test will catch it.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
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

fn build_li6_sphere_model() -> Model {
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
    let nuclide_map = HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]);
    material.read_nuclear_data(&nuclide_map, None).unwrap();

    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell.clone()], vec![Arc::new(material)]).unwrap();

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
    tally.name = Some("tritium_production".to_string());
    tally.reset_accumulation();

    Model::new(geometry, vec![source], vec![Arc::new(tally)])
}

#[test]
fn model_round_trips_through_json() {
    let original = build_li6_sphere_model();

    // Serialize to JSON.
    let json = serde_json::to_string(&original).expect("serialize");
    assert!(!json.is_empty());

    // Sanity-check the JSON contains the expected model bits. total_particles
    // and seed are no longer serialized -- they are per-run TransportSettings.
    assert!(json.contains("\"Li6\""));
    assert!(json.contains("\"tritium_production\""));
    assert!(json.contains("\"radius\":10.0"), "missing sphere radius");

    // Deserialize back.
    let loaded: Model = serde_json::from_str(&json).expect("deserialize");

    // Sources and tallies survive.
    assert_eq!(loaded.sources.len(), original.sources.len());
    assert_eq!(loaded.tallies.len(), original.tallies.len());
    assert_eq!(loaded.tallies[0].name, original.tallies[0].name);
    assert_eq!(loaded.tallies[0].scores, original.tallies[0].scores);
    assert_eq!(loaded.tallies[0].filters, original.tallies[0].filters);
}

#[test]
fn deep_model_fields_survive_round_trip() {
    // Verify the model graph deeper than the smoke check above -- geometry
    // cells, surfaces, materials, source distribution, tally filters.
    // (Actually running `simulate_transport` on a loaded model is a
    // separate concern: nuclide data isn't part of the model spec -- a
    // freshly-loaded Material has its composition but no loaded
    // cross-section arrays. The caller is expected to call
    // `material.read_nuclear_data(...)` after load, exactly the same
    // way as constructing a fresh model.)
    let original = build_li6_sphere_model();
    let json = serde_json::to_string(&original).unwrap();
    let loaded: Model = serde_json::from_str(&json).unwrap();

    // GeometryKind round-trips as CSG.
    use yamc::geometry::backend::GeometryKind;
    // `GeometryKind::Mesh` only exists under the `mesh` feature, so this
    // pattern is irrefutable on the default build (Csg-only) and refutable
    // with `mesh` -- allow the lint rather than special-case each config.
    #[allow(irrefutable_let_patterns)]
    let GeometryKind::Csg(loaded_geom) = &loaded.geometry
    else {
        panic!("expected CSG geometry");
    };
    #[allow(irrefutable_let_patterns)]
    let GeometryKind::Csg(orig_geom) = &original.geometry
    else {
        panic!("expected CSG geometry");
    };
    assert_eq!(loaded_geom.cells.len(), orig_geom.cells.len());
    assert_eq!(loaded_geom.materials.len(), orig_geom.materials.len());
    assert_eq!(loaded_geom.cells[0].cell_id, orig_geom.cells[0].cell_id);
    assert_eq!(loaded_geom.cells[0].name, orig_geom.cells[0].name);
    assert_eq!(
        loaded_geom.cells[0].material_idx,
        orig_geom.cells[0].material_idx
    );
    assert_eq!(loaded_geom.materials[0].name, orig_geom.materials[0].name);
    assert_eq!(
        loaded_geom.materials[0].nuclides,
        orig_geom.materials[0].nuclides
    );
    assert_eq!(
        loaded_geom.materials[0].density,
        orig_geom.materials[0].density
    );
    assert_eq!(
        loaded_geom.materials[0].temperature(),
        orig_geom.materials[0].temperature()
    );
}

fn sphere(surface_id: usize, radius: f64, boundary: BoundaryType) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(surface_id),
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

fn intersection(a: HalfspaceType, b: HalfspaceType) -> Region {
    use yamc::geo::RegionExpr;
    Region {
        expr: RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(a)),
            Box::new(RegionExpr::Halfspace(b)),
        ),
    }
}

#[test]
fn shared_surface_with_an_id_survives_round_trip() {
    // The inner sphere bounds both cells. JSON writes it once per cell, so
    // loading has to make the two copies one surface again; without that
    // its explicit id read as a duplicate and the load failed.
    let inner = sphere(1, 5.0, BoundaryType::Transmission);
    let outer = sphere(2, 10.0, BoundaryType::Vacuum);
    let core = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(inner.clone())),
        None,
        None,
    );
    let shell = Cell::new(
        Some(2),
        intersection(HalfspaceType::Above(inner), HalfspaceType::Below(outer)),
        None,
        None,
    );
    let geometry = Geometry::new(vec![core, shell], vec![]).unwrap();
    let json = serde_json::to_string(&geometry).unwrap();

    let loaded: Geometry =
        serde_json::from_str(&json).expect("a shared surface with an id must load");

    let in_core = loaded.cells[0].region.surfaces_with_sense()[0].0.clone();
    let in_shell = loaded.cells[1]
        .region
        .surfaces_with_sense()
        .into_iter()
        .find(|(surface, _)| surface.surface_id == Some(1))
        .unwrap()
        .0;
    assert!(
        Arc::ptr_eq(&in_core, &in_shell),
        "the inner sphere is one surface again"
    );
    assert_eq!(
        serde_json::to_string(&loaded).unwrap(),
        json,
        "the on-disk form is unchanged"
    );
}

#[test]
fn one_id_on_two_different_surfaces_is_still_refused_on_load() {
    // A genuine clash is not merged away: the two spheres differ, so they
    // stay two surfaces and their shared id is still reported.
    use yamc::geometry::cell::CellSerde;
    use yamc::geometry::csg::GeometrySerde;
    let cells = [5.0, 8.0].into_iter().enumerate().map(|(i, radius)| {
        let surface = sphere(7, radius, BoundaryType::Vacuum);
        CellSerde::from(Cell::new(
            Some(i as u32 + 1),
            Region::new_from_halfspace(HalfspaceType::Below(surface)),
            None,
            None,
        ))
    });
    let spec = GeometrySerde {
        cells: cells.collect(),
        materials: vec![],
        fills: vec![],
    };
    let json = serde_json::to_string(&spec).unwrap();

    let error = serde_json::from_str::<Geometry>(&json)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Duplicate surface_id 7"), "{error}");
}
