//! Hybrid geometry: CSG cells filled by surface-mesh bodies (issue #232).
//!
//! Uses the two-region Arrow fixture (a unit cube split at x=0.5 into
//! "fuel" (x in [0, 0.5]) and "moderator" (x in [0.5, 1]) volumes) embedded
//! in analytic CSG containers. Layer 1 tests exercise point location, the
//! min-of-two-boundaries query and the placement transform with no
//! transport; the transport tests pin the hybrid representation against an
//! exact pure-CSG twin of the same model.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::fill::CellFillSpec;
use yamc::geometry::mesh::MeshGeometry;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;

const TWO_REGION: &str = "../yamt/tests/data/two_region.arrow";
const CUBE: &str = "../yamt/tests/data/cube.arrow";

/// Material with no nuclear data: fine for geometry-only tests.
fn dummy_material(name: &str, id: u32) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("H1".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.0),
    )
    .unwrap();
    m.set_name(name);
    m.set_material_id(id);
    Arc::new(m)
}

fn two_region_mesh(fuel: Arc<Material>, moderator: Arc<Material>) -> MeshGeometry {
    let materials = HashMap::from([
        ("fuel".to_string(), fuel),
        ("moderator".to_string(), moderator),
    ]);
    MeshGeometry::from_arrow(std::path::Path::new(TWO_REGION), &materials)
        .expect("load two_region.arrow")
}

fn sphere_region(radius: f64, centre: [f64; 3], boundary: BoundaryType) -> Region {
    let sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: centre[0],
            y0: centre[1],
            z0: centre[2],
            radius,
        },
        boundary,
        name: None,
    });
    Region::new_from_halfspace(HalfspaceType::Below(sphere))
}

/// Sphere host cell around the two-region cube, built with
/// `Geometry::new_with_fills`. Returns the geometry; the host cell is
/// index 0 and the two embedded cells follow.
fn filled_sphere_geometry(
    translation: [f64; 3],
    rotation_degrees: [f64; 3],
) -> Result<Geometry, String> {
    let complement = dummy_material("complement", 1);
    let mesh = two_region_mesh(dummy_material("fuel", 2), dummy_material("moderator", 3));
    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        Some("chamber".to_string()),
        Some(0),
    );
    Geometry::new_with_fills(
        vec![host],
        vec![complement],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation,
            rotation_degrees,
            allow_clipping: false,
        }],
    )
}

fn material_name_of(geometry: &Geometry, cell_index: usize) -> Option<String> {
    geometry.cells[cell_index]
        .material_idx
        .and_then(|i| geometry.materials.get(i as usize))
        .and_then(|m| m.get_name().map(|n| n.to_string()))
}

// --------------------------- Layer 1: geometry ---------------------------

#[test]
fn point_location_resolves_fill() {
    let g = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
    assert_eq!(g.cells.len(), 3, "host + two embedded volumes");

    // Inside the fuel half of the cube.
    let fuel_idx = g.find_cell_index((0.25, 0.5, 0.5)).unwrap();
    assert_eq!(material_name_of(&g, fuel_idx).as_deref(), Some("fuel"));
    assert_ne!(fuel_idx, 0, "point in the body must not stay in the host");

    // Inside the moderator half.
    let mod_idx = g.find_cell_index((0.75, 0.5, 0.5)).unwrap();
    assert_eq!(material_name_of(&g, mod_idx).as_deref(), Some("moderator"));

    // In the gap between the cube and the sphere: the host (complement).
    let gap_idx = g.find_cell_index((0.5, 0.5, 2.0)).unwrap();
    assert_eq!(gap_idx, 0);
    assert_eq!(material_name_of(&g, gap_idx).as_deref(), Some("complement"));

    // Outside the region entirely.
    assert_eq!(g.find_cell_index((0.5, 0.5, 5.0)), None);

    // Embedded cells got their own unique auto-assigned ids.
    let ids: Vec<u32> = g.cells.iter().map(|c| c.cell_id.unwrap()).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "cell ids must be unique: {ids:?}");
    // Embedded cells carry the mesh volume, prefixed by the host name.
    assert!(g.cells[fuel_idx].name.as_deref().unwrap().contains("fuel"));
    assert!(g.cells[fuel_idx]
        .name
        .as_deref()
        .unwrap()
        .starts_with("chamber/"));
    // Embedded volumes carry the mesh's analytic volume measure (0.5 cm^3).
    assert!((g.cells[fuel_idx].volume.unwrap() - 0.5).abs() < 0.05);
}

#[test]
fn closest_boundary_takes_min_of_csg_and_mesh() {
    let g = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
    let fuel_idx = g.find_cell_index((0.25, 0.5, 0.5)).unwrap();
    let mod_idx = g.find_cell_index((0.75, 0.5, 0.5)).unwrap();

    // From the gap towards the cube: the mesh face at z=1 is nearer than
    // the sphere, so the crossing enters the fuel volume.
    let hit = g
        .closest_boundary(0, [0.25, 0.5, 2.0], [0.0, 0.0, -1.0])
        .unwrap();
    assert!((hit.distance - 1.0).abs() < 1e-9, "got {}", hit.distance);
    assert_eq!(hit.boundary, BoundaryType::Transmission);
    assert_eq!(hit.next_cell_index, Some(fuel_idx));

    // From the same point away from the cube: the sphere (centre
    // (0.5, 0.5, 0.5), r=3) is the only boundary (vacuum). Relative to
    // the centre the ray starts at (-0.25, 0, 1.5), so the exit solves
    // (1.5 + t)^2 + 0.0625 = 9.
    let hit = g
        .closest_boundary(0, [0.25, 0.5, 2.0], [0.0, 0.0, 1.0])
        .unwrap();
    let expected = (9.0_f64 - 0.0625).sqrt() - 1.5;
    assert!(
        (hit.distance - expected).abs() < 1e-9,
        "got {}",
        hit.distance
    );
    assert_eq!(hit.boundary, BoundaryType::Vacuum);
    assert_eq!(hit.next_cell_index, None);

    // From inside the fuel volume upwards: exit through the skin back to
    // the host complement.
    let hit = g
        .closest_boundary(fuel_idx, [0.25, 0.5, 0.5], [0.0, 0.0, 1.0])
        .unwrap();
    assert!((hit.distance - 0.5).abs() < 1e-9);
    assert_eq!(hit.next_cell_index, Some(0));

    // From fuel towards moderator: internal face at x=0.5, same CSG cell,
    // material switches.
    let hit = g
        .closest_boundary(fuel_idx, [0.25, 0.5, 0.5], [1.0, 0.0, 0.0])
        .unwrap();
    assert!((hit.distance - 0.25).abs() < 1e-9);
    assert_eq!(hit.next_cell_index, Some(mod_idx));
}

#[test]
fn translation_and_rotation_place_the_body() {
    // Translated up by 1.2: the cube occupies z in [1.2, 2.2].
    let g = filled_sphere_geometry([0.0, 0.0, 1.2], [0.0; 3]).unwrap();
    let idx = g.find_cell_index((0.25, 0.5, 1.7)).unwrap();
    assert_eq!(material_name_of(&g, idx).as_deref(), Some("fuel"));
    assert_eq!(g.find_cell_index((0.25, 0.5, 0.5)), Some(0), "gap now");
    // Boundary from below the translated cube: skin at z=1.2.
    let hit = g
        .closest_boundary(0, [0.25, 0.5, 0.0], [0.0, 0.0, 1.0])
        .unwrap();
    assert!((hit.distance - 1.2).abs() < 1e-9, "got {}", hit.distance);
    assert_eq!(hit.next_cell_index, Some(idx));

    // Rotated 90 degrees about z: mesh point (x, y) lands at (-y, x), so
    // the cube occupies x in [-1, 0], y in [0, 1] and the fuel half
    // (x_mesh < 0.5) is the y_world < 0.5 half.
    let g = filled_sphere_geometry([0.0; 3], [0.0, 0.0, 90.0]).unwrap();
    let idx = g.find_cell_index((-0.5, 0.25, 0.5)).unwrap();
    assert_eq!(material_name_of(&g, idx).as_deref(), Some("fuel"));
    let idx = g.find_cell_index((-0.5, 0.75, 0.5)).unwrap();
    assert_eq!(material_name_of(&g, idx).as_deref(), Some("moderator"));
    assert_eq!(g.find_cell_index((0.5, 0.5, 0.5)), Some(0), "gap now");
}

#[test]
fn protrusion_errors_unless_clipping_allowed() {
    let complement = dummy_material("complement", 1);
    let mesh = two_region_mesh(dummy_material("fuel", 2), dummy_material("moderator", 3));
    // Sphere of radius 0.4 centred in the unit cube: the cube pokes out.
    let host = Cell::new(
        Some(1),
        sphere_region(0.4, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        Some("small".to_string()),
        Some(0),
    );
    let spec = |mesh: MeshGeometry, allow: bool| CellFillSpec {
        host_cell_index: 0,
        mesh_geometry: mesh,
        translation: [0.0; 3],
        rotation_degrees: [0.0; 3],
        allow_clipping: allow,
    };

    let err = Geometry::new_with_fills(
        vec![host.clone()],
        vec![complement.clone()],
        vec![spec(mesh, false)],
    )
    .unwrap_err();
    assert!(err.contains("protrudes"), "got: {err}");
    assert!(err.contains("allow_clipping"), "got: {err}");

    // With clipping allowed the geometry builds, and the CSG surface wins
    // the min: from inside the body the nearest boundary is the sphere
    // (vacuum exit), not the clipped-off mesh skin.
    let mesh = two_region_mesh(dummy_material("fuel", 2), dummy_material("moderator", 3));
    let g = Geometry::new_with_fills(vec![host], vec![complement], vec![spec(mesh, true)]).unwrap();
    let idx = g.find_cell_index((0.45, 0.5, 0.5)).unwrap();
    assert_eq!(material_name_of(&g, idx).as_deref(), Some("fuel"));
    // Clipping invalidates the analytic mesh volumes: the effective body
    // is smaller, so volume-normalized physics must not use them.
    assert_eq!(g.cells[idx].volume, None, "clipped cell volume dropped");
    let mat = &g.materials[g.cells[idx].material_idx.unwrap() as usize];
    assert_eq!(mat.volume, None, "clipped material volume dropped");
    let hit = g
        .closest_boundary(idx, [0.45, 0.5, 0.5], [-1.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(hit.boundary, BoundaryType::Vacuum, "CSG surface clips");
    assert!(hit.distance < 0.45, "got {}", hit.distance);
    assert_eq!(hit.next_cell_index, None);
}

/// Unit-cube box host whose planes coincide exactly with the mesh skin,
/// with the body placed at `offset` (host planes shifted to match).
fn touching_box_geometry(offset: f64) -> Result<Geometry, String> {
    let complement = dummy_material("complement", 1);
    let mesh = two_region_mesh(dummy_material("fuel", 2), dummy_material("moderator", 3));
    let above = |s: Surface| Region::new_from_halfspace(HalfspaceType::Above(Arc::new(s)));
    let below = |s: Surface| Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s)));
    let vac = Some(BoundaryType::Vacuum);
    let box_region = above(Surface::x_plane(offset, None, vac.clone()))
        .intersection(&below(Surface::x_plane(offset + 1.0, None, vac.clone())))
        .intersection(&above(Surface::y_plane(offset, None, vac.clone())))
        .intersection(&below(Surface::y_plane(offset + 1.0, None, vac.clone())))
        .intersection(&above(Surface::z_plane(offset, None, vac.clone())))
        .intersection(&below(Surface::z_plane(offset + 1.0, None, vac)));
    let host = Cell::new(Some(1), box_region, Some("box".to_string()), Some(0));
    Geometry::new_with_fills(
        vec![host],
        vec![complement],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [offset; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
}

#[test]
fn body_touching_region_boundary_is_not_a_protrusion() {
    // Every mesh vertex lies exactly ON the region boundary.
    let g = touching_box_geometry(0.0);
    assert!(g.is_ok(), "coincident skin must build: {:?}", g.err());
}

#[test]
fn coincident_skin_crossings_go_to_the_csg_surface() {
    // With the mesh skin coincident with the host's vacuum planes, the
    // complement has zero width: every skin crossing must be reported as
    // the CSG (vacuum) exit, never as a mesh crossing into the host.
    // Moller-Trumbore and the plane intersection round differently for
    // oblique rays, so without the crossing margin the mesh would win
    // about half the time and strand the particle outside the region
    // with a stale topological cell. The 0.1 offset makes the plane
    // coordinates inexact in binary, maximizing the float noise.
    for offset in [0.0, 0.1] {
        let g = touching_box_geometry(offset).unwrap();
        let mod_idx = g
            .find_cell_index((offset + 0.85, offset + 0.6, offset + 0.6))
            .unwrap();
        let start = [offset + 0.85, offset + 0.6, offset + 0.6];
        for k in 0..2000usize {
            // Golden-ratio direction sphere.
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / 2000.0;
            let r = (1.0 - z * z).sqrt();
            let phi = 2.399963229728653 * k as f64;
            let dir = [r * phi.cos(), r * phi.sin(), z];
            let hit = g.closest_boundary(mod_idx, start, dir).unwrap();
            assert_ne!(
                hit.next_cell_index,
                Some(0),
                "offset {offset}, ray {k}: skin crossing resolved into the \
                 zero-width complement instead of the CSG boundary"
            );
        }
    }
}

#[test]
fn non_watertight_fill_is_rejected() {
    let mut topo =
        yamt::build_topology(yamt::read_arrow_mesh(std::path::Path::new(CUBE)).unwrap()).unwrap();
    // Drop one triangle from the first surface: opens a hole.
    let range = topo.surface_tri_ranges[0].clone();
    topo.surface_tri_ranges[0] = range.start..(range.end - 1);
    let mesh = yamt::MeshGeometry::from_topology(topo);
    let materials = HashMap::from([("water".to_string(), dummy_material("water", 2))]);
    let mesh = MeshGeometry::new(mesh, &materials, None).unwrap();

    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        None,
        Some(0),
    );
    let err = Geometry::new_with_fills(
        vec![host],
        vec![dummy_material("complement", 1)],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap_err();
    assert!(err.contains("watertight"), "got: {err}");
}

#[test]
fn fill_with_implicit_complement_material_is_rejected() {
    let materials = HashMap::from([
        ("fuel".to_string(), dummy_material("fuel", 2)),
        ("moderator".to_string(), dummy_material("moderator", 3)),
    ]);
    let mesh = yamt::MeshGeometry::from_arrow(std::path::Path::new(TWO_REGION)).unwrap();
    let mesh = MeshGeometry::new(mesh, &materials, Some(dummy_material("air", 4))).unwrap();
    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        None,
        Some(0),
    );
    let err = Geometry::new_with_fills(
        vec![host],
        vec![dummy_material("complement", 1)],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap_err();
    assert!(err.contains("implicit_complement_material"), "got: {err}");
}

/// Two disjoint sphere hosts, each filled by its own two-region body.
fn two_fill_geometry(
    fill_a: (Arc<Material>, Arc<Material>),
    fill_b: (Arc<Material>, Arc<Material>),
) -> Result<Geometry, String> {
    let host_a = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        Some("a".to_string()),
        Some(0),
    );
    let host_b = Cell::new(
        Some(2),
        sphere_region(3.0, [10.5, 0.5, 0.5], BoundaryType::Vacuum),
        Some("b".to_string()),
        Some(0),
    );
    let spec = |host, mesh, tx| CellFillSpec {
        host_cell_index: host,
        mesh_geometry: mesh,
        translation: [tx, 0.0, 0.0],
        rotation_degrees: [0.0; 3],
        allow_clipping: false,
    };
    Geometry::new_with_fills(
        vec![host_a, host_b],
        vec![dummy_material("complement", 1)],
        vec![
            spec(0, two_region_mesh(fill_a.0, fill_a.1), 0.0),
            spec(1, two_region_mesh(fill_b.0, fill_b.1), 10.0),
        ],
    )
}

#[test]
fn two_fills_with_colliding_material_ids_are_rejected() {
    // Different materials sharing an id across two fills (the default
    // outcome of independent auto-assignment) must fail fast: material
    // filters, plots and transmutation are keyed by material id.
    let err = two_fill_geometry(
        (dummy_material("fuel", 2), dummy_material("moderator", 3)),
        (dummy_material("water", 2), dummy_material("oil", 3)),
    )
    .unwrap_err();
    assert!(err.contains("collides"), "got: {err}");
}

#[test]
fn two_fills_sharing_the_same_materials_are_allowed() {
    let fuel = dummy_material("fuel", 2);
    let moderator = dummy_material("moderator", 3);
    let g = two_fill_geometry(
        (Arc::clone(&fuel), Arc::clone(&moderator)),
        (fuel, moderator),
    )
    .unwrap();
    assert_eq!(g.cells.len(), 6, "2 hosts + 2x2 embedded volumes");
    let a = g.find_cell_index((0.25, 0.5, 0.5)).unwrap();
    let b = g.find_cell_index((10.25, 0.5, 0.5)).unwrap();
    assert_ne!(a, b);
    assert_eq!(material_name_of(&g, a).as_deref(), Some("fuel"));
    assert_eq!(material_name_of(&g, b).as_deref(), Some("fuel"));
}

#[test]
fn rebuilding_a_geometry_from_filled_cells_is_rejected() {
    // Cells of a filled geometry do not carry the fill; reusing them
    // would silently drop the mesh bodies (or index a missing fill).
    let g = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
    let err = Geometry::new(g.cells.clone(), g.materials.clone()).unwrap_err();
    assert!(err.contains("cannot seed"), "got: {err}");
}

#[test]
fn fill_material_id_collision_with_different_material_is_rejected() {
    let complement = dummy_material("steel", 2);
    // Fill's "fuel" material deliberately reuses id 2 with another name.
    let mesh = two_region_mesh(dummy_material("fuel", 2), dummy_material("moderator", 3));
    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        None,
        Some(0),
    );
    let err = Geometry::new_with_fills(
        vec![host],
        vec![complement],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap_err();
    assert!(err.contains("collides"), "got: {err}");
}

#[test]
fn sample_slice_shows_the_body() {
    let g = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
    // 1-pixel probes through the fuel half and through the gap.
    let (cells, mats) = g.sample_slice((0.25, 0.5, 0.5), (0.01, 0.01), (1, 1), "xy");
    let fuel_idx = g.find_cell_index((0.25, 0.5, 0.5)).unwrap();
    assert_eq!(cells[0][0], g.cells[fuel_idx].cell_id.unwrap() as i32);
    assert_eq!(mats[0][0], 2, "fuel material id");
    let (cells, mats) = g.sample_slice((0.5, 0.5, 2.0), (0.01, 0.01), (1, 1), "xy");
    assert_eq!(cells[0][0], 1, "host cell id in the gap");
    assert_eq!(mats[0][0], 1, "complement material id in the gap");
}

#[test]
fn geometry_with_fills_fingerprints_but_does_not_deserialize() {
    let g = filled_sphere_geometry([0.0, 0.0, 1.2], [0.0; 3]).unwrap();
    let v1 = serde_json::to_value(&g).expect("serialize");
    let v2 = serde_json::to_value(&g).expect("serialize again");
    assert_eq!(v1, v2, "fingerprint must be deterministic");
    let fills = v1["fills"].as_array().expect("fills fingerprint present");
    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0]["translation"][2], 1.2);
    assert_eq!(fills[0]["num_volumes"], 2);

    let err = serde_json::from_value::<Geometry>(v1).unwrap_err();
    assert!(
        err.to_string().contains("cannot be deserialized"),
        "got: {err}"
    );

    // A translation change alters the fingerprint (combine_results identity).
    let g2 = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
    assert_ne!(serde_json::to_value(&g2).unwrap(), v2);
}

// --------------------------- Transport ---------------------------

fn data_material(name: &str, id: u32, nuclide: &str, density: f64) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_name(name);
    m.set_material_id(id);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(nuclide.to_string(), format!("tests/{nuclide}.arrow"))]),
        None,
    )
    .unwrap();
    Arc::new(m)
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.5, 0.5, 2.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn flux_tally(name: &str, cell_id: Option<u32>) -> Arc<Tally> {
    let mut t = Tally::new();
    if let Some(id) = cell_id {
        t.filters.push(Filter::Cell(CellFilter::from_id(id)));
    }
    t.scores = vec!["flux".parse::<Score>().unwrap()];
    t.name = Some(name.to_string());
    t.initialize_batches(1);
    Arc::new(t)
}

fn run(model: &mut Model, n: usize) {
    model.verbose = Verbose::silent();
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(n),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        })
        .expect("transport run");
}

/// The highest-value test from the issue: a hybrid model (mesh cube in a
/// CSG sphere) against a pure-CSG twin of the identical configuration
/// (the two-region fixture is an exact axis-aligned cube, so the twin is
/// exact). Per-region and total flux must agree within statistics (a
/// combined-standard-error z test, see the criterion below); this
/// exercises location, tracking, material assignment and tally
/// addressing against an independent representation.
#[test]
fn hybrid_matches_pure_csg_twin() {
    let complement = data_material("complement", 1, "Fe56", 2.0);
    let fuel = data_material("fuel", 2, "Li6", 1.5);
    let moderator = data_material("moderator", 3, "Fe56", 7.8);
    const N: usize = 40_000;

    // Hybrid: sphere host (id 1) + mesh cube; embedded ids auto-assign
    // to 2 (fuel volume) and 3 (moderator volume) after the host's 1.
    let mesh = two_region_mesh(Arc::clone(&fuel), Arc::clone(&moderator));
    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        Some("chamber".to_string()),
        Some(0),
    );
    let g = Geometry::new_with_fills(
        vec![host],
        vec![Arc::clone(&complement)],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap();
    let fuel_idx = g.find_cell_index((0.25, 0.5, 0.5)).unwrap();
    let mod_idx = g.find_cell_index((0.75, 0.5, 0.5)).unwrap();
    let hybrid_ids = [
        1u32,
        g.cells[fuel_idx].cell_id.unwrap(),
        g.cells[mod_idx].cell_id.unwrap(),
    ];
    let hybrid_tallies: Vec<Arc<Tally>> = ["complement", "fuel", "moderator"]
        .iter()
        .zip(hybrid_ids)
        .map(|(n, id)| flux_tally(n, Some(id)))
        .collect();
    let mut model = Model::new(g, vec![neutron_source()], hybrid_tallies.clone());
    run(&mut model, N);
    assert!(
        model.lost_particles.is_empty(),
        "hybrid run lost particles: {:?}",
        model.lost_particles
    );
    let hybrid_flux: Vec<f64> = hybrid_tallies
        .iter()
        .map(|t| t.get_mean().iter().sum())
        .collect();
    let hybrid_sigma: Vec<f64> = hybrid_tallies.iter().map(|t| t.total_std()).collect();

    // Pure-CSG twin: same sphere, the cube as two half boxes of planes.
    let above = |s: Surface| Region::new_from_halfspace(HalfspaceType::Above(Arc::new(s)));
    let below = |s: Surface| Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s)));
    let x0 = Arc::new(Surface::x_plane(0.0, None, None));
    let xm = Arc::new(Surface::x_plane(0.5, None, None));
    let x1 = Arc::new(Surface::x_plane(1.0, None, None));
    let slab = |lo: &Arc<Surface>, hi: &Arc<Surface>| {
        Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(lo))).intersection(
            &Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(hi))),
        )
    };
    let yz_bounds = above(Surface::y_plane(0.0, None, None))
        .intersection(&below(Surface::y_plane(1.0, None, None)))
        .intersection(&above(Surface::z_plane(0.0, None, None)))
        .intersection(&below(Surface::z_plane(1.0, None, None)));
    let fuel_box = slab(&x0, &xm).intersection(&yz_bounds);
    let mod_box = slab(&xm, &x1).intersection(&yz_bounds);
    let whole_box = slab(&x0, &x1).intersection(&yz_bounds);
    let chamber = sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum)
        .intersection(&whole_box.complement());

    let twin_cells = vec![
        Cell::new(Some(1), chamber, Some("chamber".to_string()), Some(0)),
        Cell::new(Some(2), fuel_box, Some("fuel".to_string()), Some(1)),
        Cell::new(Some(3), mod_box, Some("moderator".to_string()), Some(2)),
    ];
    let twin = Geometry::new(twin_cells, vec![complement, fuel, moderator]).unwrap();
    let twin_tallies: Vec<Arc<Tally>> = ["complement", "fuel", "moderator"]
        .iter()
        .zip([1u32, 2, 3])
        .map(|(n, id)| flux_tally(n, Some(id)))
        .collect();
    let mut model = Model::new(twin, vec![neutron_source()], twin_tallies.clone());
    run(&mut model, N);
    let twin_flux: Vec<f64> = twin_tallies
        .iter()
        .map(|t| t.get_mean().iter().sum())
        .collect();
    let twin_sigma: Vec<f64> = twin_tallies.iter().map(|t| t.total_std()).collect();

    // Agreement criterion. The two models are physically identical but run
    // independent random streams, so their tallies can only agree to within
    // their own Monte Carlo error: compare the difference against the
    // combined standard error, `z = |hybrid - twin| / sqrt(sigma_h^2 +
    // sigma_t^2)`, not against a fixed percentage.
    //
    // Recalibrated from a flat `rel < 0.02` (issue #111): "fuel" is a thin
    // low-flux Li6 region whose own relative error is ~3% at this N, so a 2%
    // bound was tighter than the statistics and only held for one particular
    // RNG realization. Moving the continuum inelastic kinematics off `FastRng`
    // onto the shared PCG stream renumbers that stream and pushed the fuel
    // realization to 2.14% while its z was UNCHANGED at 0.45 (and the
    // discrepancy shrank as 1/sqrt(N): 2.14% -> 0.61% -> 0.43% at N x 4, x 16),
    // i.e. noise, not bias. Measured z: complement 0.74, fuel 0.45,
    // moderator 0.09. `Z_MAX = 4` leaves ample headroom for a future stream
    // renumbering while still catching a real geometry / tally-addressing bias
    // (which would show up as many sigma); `REL_MAX` is a coarse backstop for
    // the case where a regression destroys the statistics instead of the mean.
    const Z_MAX: f64 = 4.0;
    const REL_MAX: f64 = 0.25;
    for (i, name) in ["complement", "fuel", "moderator"].iter().enumerate() {
        assert!(
            twin_flux[i] > 0.0 && hybrid_flux[i] > 0.0,
            "{name}: fluxes must be positive (twin {}, hybrid {})",
            twin_flux[i],
            hybrid_flux[i]
        );
        let diff = (hybrid_flux[i] - twin_flux[i]).abs();
        let rel = diff / twin_flux[i];
        let combined_sigma = (hybrid_sigma[i].powi(2) + twin_sigma[i].powi(2)).sqrt();
        assert!(
            combined_sigma > 0.0,
            "{name}: both runs must report a non-zero standard error \
             (hybrid {}, twin {})",
            hybrid_sigma[i],
            twin_sigma[i]
        );
        let z = diff / combined_sigma;
        assert!(
            z < Z_MAX && rel < REL_MAX,
            "{name}: hybrid {} +/- {} vs twin {} +/- {} differ by {:.2}% ({z:.2} sigma)",
            hybrid_flux[i],
            hybrid_sigma[i],
            twin_flux[i],
            twin_sigma[i],
            rel * 100.0
        );
    }
}

/// Fill with the complement's own material: results must match the
/// unfilled cell within statistics (the geometry machinery is invisible
/// when it separates identical materials).
#[test]
fn same_material_fill_is_invisible() {
    let mat = data_material("iron", 1, "Fe56", 7.8);
    const N: usize = 30_000;

    let mesh = two_region_mesh(Arc::clone(&mat), Arc::clone(&mat));
    let host = Cell::new(
        Some(1),
        sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        None,
        Some(0),
    );
    let g = Geometry::new_with_fills(
        vec![host],
        vec![Arc::clone(&mat)],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap();
    // Filter-free tally: total flux over everything.
    let t_filled = flux_tally("flux", None);
    let mut model = Model::new(g, vec![neutron_source()], vec![t_filled.clone()]);
    run(&mut model, N);
    assert!(model.lost_particles.is_empty());
    let filled: f64 = t_filled.get_mean().iter().sum();

    let plain = Geometry::new(
        vec![Cell::new(
            Some(1),
            sphere_region(3.0, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
            None,
            Some(0),
        )],
        vec![mat],
    )
    .unwrap();
    let t_plain = flux_tally("flux", None);
    let mut model = Model::new(plain, vec![neutron_source()], vec![t_plain.clone()]);
    run(&mut model, N);
    let unfilled: f64 = t_plain.get_mean().iter().sum();

    let rel = (filled - unfilled).abs() / unfilled;
    assert!(
        rel < 0.02,
        "filled {filled} vs unfilled {unfilled} differ by {:.2}%",
        rel * 100.0
    );
}

/// Clipped protrusion: particles cross the CSG surface cleanly, never
/// getting lost, even though the mesh pokes through it.
#[test]
fn clipped_protrusion_loses_no_particles() {
    let complement = data_material("complement", 1, "Fe56", 2.0);
    let fuel = data_material("fuel", 2, "Li6", 1.5);
    let moderator = data_material("moderator", 3, "Fe56", 7.8);

    let mesh = two_region_mesh(fuel, moderator);
    // Sphere r=0.4 inside the unit cube: most of the cube is clipped off.
    let host = Cell::new(
        Some(1),
        sphere_region(0.4, [0.5, 0.5, 0.5], BoundaryType::Vacuum),
        None,
        Some(0),
    );
    let g = Geometry::new_with_fills(
        vec![host],
        vec![complement],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh,
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: true,
        }],
    )
    .unwrap();
    let t = flux_tally("flux", None);
    let mut model = Model::new(g, vec![neutron_source()], vec![t.clone()]);
    // Source sits outside the sphere region entirely; use one inside.
    model.sources = vec![ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.5, 0.5, 0.5])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })];
    run(&mut model, 10_000);
    assert!(
        model.lost_particles.is_empty(),
        "clipping must never lose particles: {:?}",
        model.lost_particles
    );
    let flux: f64 = t.get_mean().iter().sum();
    assert!(flux.is_finite() && flux > 0.0);
}

/// Woodcock and hybrid tracking cannot see mesh fills: hard error.
#[test]
fn woodcock_tracking_rejects_mesh_fills() {
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let g = filled_sphere_geometry([0.0; 3], [0.0; 3]).unwrap();
        let mut model = Model::new(g, vec![neutron_source()], vec![flux_tally("flux", None)]);
        model.verbose = Verbose::silent();
        model.tracking_mode = mode;
        let err = model
            .simulate_transport(&TransportSettings {
                total_particles: Some(100),
                seed: 1,
                threads: Some(1),
                ..Default::default()
            })
            .unwrap_err();
        assert!(
            err.contains("mesh-filled") && err.contains("surface"),
            "{mode:?}: got {err}"
        );
    }
}
