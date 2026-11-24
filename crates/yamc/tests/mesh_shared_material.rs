//! Mesh volumes given one material share one slot of the material store.
//!
//! Transport builds one set of cross-section tables per slot, so a mesh whose
//! volumes all hold the same material needs one set, not one per volume. The
//! tests here show that sharing a slot changes nothing a tally can see, and
//! that a mesh filling a CSG cell hands its shared slot on to the host
//! geometry.
//!
//! Uses the two-region Arrow fixture (a unit cube split at x = 0.5 into
//! volumes named "fuel" and "moderator", 0.5 cm3 each). The transport test
//! self-skips if the Fe56 fixture is absent.
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
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::Tally;
use yamc_tallies::MaterialFilter;

const TWO_REGION: &str = "../yamt/tests/data/two_region.arrow";
const NUCLIDE: &str = "Fe56";

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

/// Iron with nuclear data, or without it for tests that never transport.
fn iron(id: u32, with_data: bool) -> Material {
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
    m
}

/// The two-region cube with `fuel` and `moderator` as given.
fn two_region_mesh(fuel: Arc<Material>, moderator: Arc<Material>) -> MeshGeometry {
    let materials = HashMap::from([
        ("fuel".to_string(), fuel),
        ("moderator".to_string(), moderator),
    ]);
    MeshGeometry::from_arrow(std::path::Path::new(TWO_REGION), &materials)
        .expect("load two_region.arrow")
}

/// The cube with one material under both names: one slot for both volumes.
fn shared_mesh(material: Material) -> MeshGeometry {
    let material = Arc::new(material);
    two_region_mesh(Arc::clone(&material), material)
}

/// The cube with a copy of the material under each name. Two `Arc`s are two
/// materials, so each volume has a slot of its own (with one id between
/// them), which is the layout a slot per volume gives.
fn separate_mesh(material: Material) -> MeshGeometry {
    two_region_mesh(Arc::new(material.clone()), Arc::new(material))
}

/// A 14.06 MeV point source in the middle of the fuel half.
fn source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.25, 0.5, 0.5])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

fn flux_tally(name: &str, filter: Filter) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(filter);
    t.scores = vec!["flux".parse().unwrap()];
    t.name = Some(name.to_string());
    t.reset_accumulation();
    Arc::new(t)
}

/// Flux in the iron, in each of the two volumes, and on voxels 0.6 cm wide
/// that straddle the cube's faces and the face between its halves.
fn tallies(geometry: &MeshGeometry) -> Vec<Arc<Tally>> {
    let iron = &geometry.materials[0];
    let voxels = RegularRectangularMesh::new([-1.0; 3], [2.0; 3], [5; 3]);
    vec![
        flux_tally("iron", Filter::Material(MaterialFilter::new(iron).unwrap())),
        flux_tally(
            "fuel",
            Filter::Cell(CellFilter::from_id(geometry.cells[0].cell_id.unwrap())),
        ),
        flux_tally(
            "moderator",
            Filter::Cell(CellFilter::from_id(geometry.cells[1].cell_id.unwrap())),
        ),
        flux_tally("voxels", Filter::Mesh(MeshFilter::new(voxels))),
    ]
}

fn run(model: &mut Model, mode: TrackingMode) {
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(3_000),
            seed: 17,
            threads: Some(1),
            ..Default::default()
        })
        .expect("transport run");
}

/// Two volumes on one slot give the tallies that a slot each gives, in every
/// tracking mode: the slot a volume reads its cross sections from is not
/// something transport or a tally can see.
#[test]
fn sharing_a_slot_does_not_change_transport() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        let mut means: Vec<Vec<Vec<f64>>> = Vec::new();
        for (shared, slots) in [(true, 1), (false, 2)] {
            let geometry = if shared {
                shared_mesh(iron(1, true))
            } else {
                separate_mesh(iron(1, true))
            };
            assert_eq!(geometry.materials.len(), slots, "shared={shared}");
            let tallies = tallies(&geometry);
            let mut model = Model::new_with_mesh(geometry, vec![source()], tallies.clone());
            run(&mut model, mode);
            // Preparing the cross sections kept the slots as they were built.
            assert_eq!(
                model.geometry.materials().len(),
                slots,
                "{mode:?} shared={shared}"
            );
            assert!(
                model.lost_particles.is_empty(),
                "{mode:?} shared={shared}: {:?}",
                model.lost_particles
            );
            means.push(tallies.iter().map(|t| t.get_mean()).collect());
        }
        let (shared, separate) = (&means[0], &means[1]);
        // The iron, each half of it, and some voxel all saw flux, and the
        // two halves add up to the iron.
        for tally in shared {
            assert!(tally.iter().any(|&m| m > 0.0), "{mode:?}: {tally:?}");
        }
        let (iron, halves) = (shared[0][0], shared[1][0] + shared[2][0]);
        assert!(
            (iron - halves).abs() <= 1e-12 * iron,
            "{mode:?}: iron {iron:e} vs fuel + moderator {halves:e}"
        );
        // Exact equality, not a tolerance: one seed on one thread gives the
        // same histories, and tables built twice from the same material are
        // the same numbers.
        assert_eq!(shared, separate, "{mode:?}");
    }
}

fn sphere_region(radius: f64) -> Region {
    let sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.5,
            y0: 0.5,
            z0: 0.5,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    Region::new_from_halfspace(HalfspaceType::Below(sphere))
}

/// A vacuum sphere of `radius` centred on the cube, holding material 1 and
/// filled by the cube with material 2 in both volumes. The host cell is
/// index 0 and the two embedded cells follow; the user's one material is
/// slot 0 and the fill's follow.
fn filled_sphere_geometry(radius: f64, allow_clipping: bool) -> Geometry {
    let host = Cell::new(
        Some(1),
        sphere_region(radius),
        Some("chamber".to_string()),
        Some(0),
    );
    Geometry::new_with_fills(
        vec![host],
        vec![Arc::new(iron(1, false))],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: shared_mesh(iron(2, false)),
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping,
        }],
    )
    .unwrap()
}

/// A mesh filling a CSG cell brings one slot for its shared material, not
/// one per volume. The embedded cells keep their own volumes and the slot
/// has their total.
#[test]
fn mesh_fill_brings_one_slot_for_a_shared_material() {
    let g = filled_sphere_geometry(3.0, false);
    assert_eq!(g.cells.len(), 3, "host + two embedded volumes");
    assert_eq!(g.materials.len(), 2, "the user's material + the fill's one");
    assert_eq!(g.cells[0].material_idx, Some(0));
    assert_eq!(g.materials[1].get_material_id(), Some(2));

    let mut total = 0.0;
    for cell in &g.cells[1..] {
        assert_eq!(cell.material_idx, Some(1), "{:?}", cell.name);
        let volume = cell.volume.unwrap();
        assert!((volume - 0.5).abs() < 1e-9, "{:?}: {volume}", cell.name);
        total += volume;
    }
    let slot_volume = g.materials[1].volume.unwrap();
    assert!((slot_volume - total).abs() < 1e-12, "{slot_volume}");
    assert!((slot_volume - 1.0).abs() < 1e-9, "{slot_volume}");
}

/// With clipping allowed the host's surface cuts the body, so neither the
/// embedded cells nor the shared slot keep the mesh's volumes.
#[test]
fn clipped_mesh_fill_drops_the_shared_volume() {
    // A sphere of radius 0.4 centred in the unit cube: the cube pokes out.
    let g = filled_sphere_geometry(0.4, true);
    assert_eq!(g.cells.len(), 3, "host + two embedded volumes");
    assert_eq!(g.materials.len(), 2, "the user's material + the fill's one");
    for cell in &g.cells[1..] {
        assert_eq!(cell.material_idx, Some(1), "{:?}", cell.name);
        assert_eq!(cell.volume, None, "{:?}", cell.name);
    }
    assert_eq!(g.materials[1].volume, None);
}
