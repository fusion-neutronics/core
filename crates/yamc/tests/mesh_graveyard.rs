//! The vacuum box a model adds around a whole-model mesh geometry.
//!
//! A mesh made from CAD has no vacuum surface, and a particle in its
//! implicit complement with nothing ahead is a lost particle. So before
//! transport the model closes such a mesh with a vacuum box
//! `DEFAULT_GRAVEYARD_OFFSET` cm outside its bounding box when the complement
//! is void, and refuses the run when the complement holds a material (where
//! the box sits would change the answer). A mesh used as a CSG fill gets no
//! box: the CSG geometry bounds it.
//!
//! Uses the two-region Arrow fixture (a unit cube split at x = 0.5). The
//! transport tests self-skip if the Fe56 fixture is absent.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::fill::CellFillSpec;
use yamc::geometry::mesh::{MeshGeometry, DEFAULT_GRAVEYARD_OFFSET};
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::{CylindricalMesh, RegularRectangularMesh};
use yamc_tallies::tally::Tally;
use yamc_tallies::MaterialFilter;

const TWO_REGION: &str = "../yamt/tests/data/two_region.arrow";
const NUCLIDE: &str = "Fe56";

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

/// Iron with nuclear data, or without it for tests that never transport.
fn iron(id: u32, with_data: bool) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    if with_data {
        let nm = HashMap::from([(NUCLIDE.to_string(), yamc_test_cache::nuclide_path(NUCLIDE))]);
        m.read_nuclear_data(&nm, None).unwrap();
    }
    Arc::new(m)
}

fn materials(with_data: bool) -> HashMap<String, Arc<Material>> {
    HashMap::from([
        ("fuel".to_string(), iron(1, with_data)),
        ("moderator".to_string(), iron(2, with_data)),
    ])
}

/// The two-region cube, with a `graveyard_offset`-style box of `offset` cm
/// built into the mesh when given, and an optional complement material.
fn mesh(
    with_data: bool,
    graveyard_offset: Option<f64>,
    complement: Option<Arc<Material>>,
) -> MeshGeometry {
    let mut data = yamt::read_arrow_mesh(std::path::Path::new(TWO_REGION)).unwrap();
    if let Some(offset) = graveyard_offset {
        yamt::add_vacuum_boundary(&mut data, offset);
    }
    let topo = yamt::build_topology(data).unwrap();
    let mesh = yamt::MeshGeometry::from_topology(topo);
    MeshGeometry::new(mesh, &materials(with_data), complement).unwrap()
}

fn source(position: [f64; 3]) -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new(position)),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn rect_mesh_tally(lower: f64, upper: f64, shape: usize) -> Arc<Tally> {
    let mut t = Tally::new();
    let mesh = RegularRectangularMesh::new([lower; 3], [upper; 3], [shape; 3]);
    t.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
    t.scores = vec!["flux".parse().unwrap()];
    t.name = Some("voxels".into());
    t.reset_accumulation();
    Arc::new(t)
}

fn iron_tally(material: &Arc<Material>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Material(MaterialFilter::new(material)));
    t.scores = vec!["flux".parse().unwrap()];
    t.reset_accumulation();
    Arc::new(t)
}

fn run(model: &mut Model, mode: TrackingMode) -> Result<(), String> {
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.simulate_transport(&TransportSettings {
        total_particles: Some(2_000),
        seed: 11,
        threads: Some(1),
        ..Default::default()
    })
}

/// A void complement gets the box, nothing is lost, and every tally (a
/// mesh tally that scores the complement included) matches the same mesh
/// with a `graveyard_offset` box of the same offset, in every tracking mode.
#[test]
fn automatic_graveyard_matches_an_explicit_one() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let expected_box = mesh(false, None, None).graveyard_box(DEFAULT_GRAVEYARD_OFFSET);
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        let mut means = Vec::new();
        for explicit in [false, true] {
            let geometry = mesh(true, explicit.then_some(DEFAULT_GRAVEYARD_OFFSET), None);
            let fuel = geometry.materials[0].clone();
            let tallies = vec![iron_tally(&fuel), rect_mesh_tally(-4.0, 5.0, 9)];
            let mut model =
                Model::new_with_mesh(geometry, vec![source([0.5, 0.5, 0.5])], tallies.clone());
            let extent = model.mesh_transport_extent().unwrap().unwrap();
            for k in 0..6 {
                assert!((extent[k] - expected_box[k]).abs() < 1e-9, "{extent:?}");
            }
            run(&mut model, mode).unwrap();
            assert!(
                model.lost_particles.is_empty(),
                "{mode:?} explicit={explicit}"
            );
            means.push(
                tallies
                    .iter()
                    .flat_map(|t| t.get_mean())
                    .collect::<Vec<f64>>(),
            );
        }
        assert!(means[0].iter().any(|&m| m > 0.0));
        for (auto, explicit) in means[0].iter().zip(&means[1]) {
            assert!(
                (auto - explicit).abs() <= 1e-9 * explicit.abs().max(1e-30),
                "{mode:?}: automatic box {auto:e} vs graveyard_offset box {explicit:e}"
            );
        }
    }
}

/// With a material in the complement the box position is physics, so the
/// run is refused before transport, and an explicit box makes it run.
#[test]
fn material_complement_needs_an_explicit_graveyard() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let air = || Some(iron(3, true));
    let mut model = Model::new_with_mesh(mesh(true, None, air()), vec![source([0.5; 3])], vec![]);
    let err = model.mesh_transport_extent().unwrap_err();
    assert!(err.contains("graveyard_offset"), "{err}");
    let err = run(&mut model, TrackingMode::Surface).unwrap_err();
    assert!(err.contains("graveyard_offset"), "{err}");

    let mut model =
        Model::new_with_mesh(mesh(true, Some(2.0), air()), vec![source([0.5; 3])], vec![]);
    run(&mut model, TrackingMode::Surface).unwrap();
    assert!(model.lost_particles.is_empty());
}

/// A mesh filling a CSG cell gets no box: particles leaving the mesh
/// cross the host cell all the way to its vacuum sphere, well past where an
/// automatic box would have stopped them.
#[test]
fn mesh_fill_gets_no_graveyard() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let host_radius = 4.0 * DEFAULT_GRAVEYARD_OFFSET;
    let sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.5,
            y0: 0.5,
            z0: 0.5,
            radius: host_radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let host = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(sphere)),
        Some("chamber".to_string()),
        None,
    );
    let geometry = Geometry::new_with_fills(
        vec![host],
        vec![],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: mesh(true, None, None),
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap();
    // Voxels from 2 to 3 offsets out along +x: beyond any automatic box.
    let far = {
        let mut t = Tally::new();
        let lower = 0.5 + 2.0 * DEFAULT_GRAVEYARD_OFFSET;
        let mesh = RegularRectangularMesh::new(
            [lower, -10.0, -10.0],
            [lower + DEFAULT_GRAVEYARD_OFFSET, 11.0, 11.0],
            [1, 1, 1],
        );
        t.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
        t.scores = vec!["flux".parse().unwrap()];
        t.reset_accumulation();
        Arc::new(t)
    };
    let mut model = Model::new(geometry, vec![source([0.5; 3])], vec![far.clone()]);
    assert_eq!(model.mesh_transport_extent(), Ok(None));
    assert!(model.mesh_tally_extent_warnings().is_empty());
    run(&mut model, TrackingMode::Surface).unwrap();
    assert!(model.lost_particles.is_empty());
    assert!(far.get_mean()[0] > 0.0, "nothing reached past the mesh");
}

/// A mesh tally reaching outside the vacuum boundary is reported (its outer
/// voxels read zero); one inside it is not. Covers the automatic box, a
/// `graveyard_offset` box, and a cylindrical tally mesh.
#[test]
fn mesh_tally_beyond_the_vacuum_boundary_is_reported() {
    let model = |offset: Option<f64>, tally: Arc<Tally>| {
        Model::new_with_mesh(
            mesh(false, offset, None),
            vec![source([0.5; 3])],
            vec![tally],
        )
    };
    let inside = rect_mesh_tally(-5.0, 6.0, 2);
    let outside = rect_mesh_tally(-15.0, 16.0, 2);
    assert!(model(None, inside.clone())
        .mesh_tally_extent_warnings()
        .is_empty());
    let warnings = model(None, outside.clone()).mesh_tally_extent_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("'voxels'") && warnings[0].contains("graveyard_offset"));
    // A 20 cm `graveyard_offset` box holds the larger tally mesh, a 2 cm
    // one does not hold even the smaller.
    assert!(model(Some(20.0), outside)
        .mesh_tally_extent_warnings()
        .is_empty());
    assert_eq!(
        model(Some(2.0), inside).mesh_tally_extent_warnings().len(),
        1
    );

    let cylinder = |r_max: f64| {
        let mut t = Tally::new();
        let mesh = CylindricalMesh::uniform(
            [0.5, 0.5, 0.0],
            (0.0, r_max),
            (0.0, std::f64::consts::TAU),
            (0.0, 1.0),
            [2, 4, 1],
        );
        t.filters
            .push(Filter::Mesh(MeshFilter::new_cylindrical(mesh)));
        t.scores = vec!["flux".parse().unwrap()];
        t.reset_accumulation();
        Arc::new(t)
    };
    assert!(model(None, cylinder(8.0))
        .mesh_tally_extent_warnings()
        .is_empty());
    assert_eq!(
        model(None, cylinder(12.0))
            .mesh_tally_extent_warnings()
            .len(),
        1
    );
}
