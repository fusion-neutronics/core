//! Transmutation of one material spread over several cells.
//!
//! Transmutation tallies a material over every cell holding its id and
//! divides by the total `Material::volume` of the slots of the geometry's
//! material store those cells use. A mesh gives volumes that share a material
//! one slot whose volume is their total; two copies of a material with one id,
//! or a material that is both a fill's and its host's, give several slots.
//! Either way the reaction rates are those of the whole material.
//!
//! Uses the two-region Arrow fixture (a unit cube split at x = 0.5 into two
//! volumes of 0.5 cm3) against the one-volume cube fixture (the same cube
//! uncut), and a CSG sphere around the cube. Self-skips when the Fe56 or chain
//! fixture is absent.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::fill::CellFillSpec;
use yamc::geometry::mesh::MeshGeometry;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yani_transmute::{Schedule, ScheduleStep, TransmutationResults};

const TWO_REGION: &str = "../yamt/tests/data/two_region.arrow";
const CUBE: &str = "../yamt/tests/data/cube.arrow";
const NUCLIDE: &str = "Fe56";
const MATERIAL_ID: u32 = 1;
const PARTICLES: usize = 20_000;
const SEED: u64 = 11;
/// The cut and uncut cubes agree to better than one part in a thousand at
/// this particle count. Dividing by one volume's measure in place of the
/// material's total is out by a factor of 2.
const TOLERANCE: f64 = 0.02;
/// Threshold reactions of Fe56 at 14 MeV and the product each one makes.
const CHANNELS: [(&str, &str); 2] = [("(n,2n)", "Fe55"), ("(n,p)", "Mn56")];

/// The chain's decay and reaction data, or `None` when it is not fetched.
fn chain() -> Option<yani::LoadedChain> {
    let chain =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/transmutation-endf-b8.1-sfr.arrow");
    if !chain.join("reactions").is_dir() {
        return None;
    }
    let dir = |d: PathBuf| d.to_string_lossy().into_owned();
    Some(
        yani::load_chain_parts(
            &dir(chain.join("decay")),
            Some(&dir(chain.join("reactions"))),
            None,
            &[],
        )
        .expect("chain"),
    )
}

/// Transmutable iron with its nuclear data read.
fn iron(path: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .expect("material");
    m.set_material_id(MATERIAL_ID);
    m.set_temperature("294");
    m.transmutable = true;
    m.read_nuclear_data(
        &HashMap::from([(NUCLIDE.to_string(), path.to_string())]),
        None,
    )
    .expect("read Fe56");
    m
}

fn mesh(arrow: &str, materials: &HashMap<String, Arc<Material>>) -> MeshGeometry {
    let mesh = yamt::MeshGeometry::from_arrow(Path::new(arrow)).expect("mesh fixture");
    MeshGeometry::new(mesh, materials, None).expect("mesh geometry")
}

/// The uncut cube, one volume named `water`, filled with iron.
fn one_volume(path: &str) -> MeshGeometry {
    let materials = HashMap::from([("water".to_string(), Arc::new(iron(path)))]);
    mesh(CUBE, &materials)
}

/// The cut cube with one iron material under both volume names.
fn two_volumes_one_material(path: &str) -> MeshGeometry {
    let iron = Arc::new(iron(path));
    let materials = HashMap::from([
        ("fuel".to_string(), Arc::clone(&iron)),
        ("moderator".to_string(), iron),
    ]);
    mesh(TWO_REGION, &materials)
}

/// A one-second pulse of a 14 MeV point source inside the cube, off the plane
/// that cuts it. A second is short enough that each product is its reaction
/// rate times the pulse, and no product earns cross sections of its own.
fn transmute(
    geometry: MeshGeometry,
    method: &str,
    loaded: &yani::LoadedChain,
) -> TransmutationResults {
    run(
        Model::new_with_mesh(geometry, vec![], vec![]),
        method,
        loaded,
    )
}

/// The pulse of [`transmute`] on any model.
fn run(mut model: Model, method: &str, loaded: &yani::LoadedChain) -> TransmutationResults {
    model.sources = vec![ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.25, 0.5, 0.5])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })];
    model.verbose = Verbose::silent();
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e10,
        dt: 1.0,
        is_pulse: true,
    }])
    .expect("schedule");
    let results = model
        .transmute(
            method,
            &schedule,
            Arc::clone(&loaded.chain),
            Arc::clone(&loaded.branch),
            loaded.parts,
            &TransportSettings {
                total_particles: Some(PARTICLES),
                seed: SEED,
                threads: Some(1),
                ..Default::default()
            },
            None,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
            None,
        )
        .expect("transmute");
    assert!(model.lost_particles.is_empty());
    results
}

/// Fe56's rate per atom [1/s] in `kind` over the pulse, all products summed.
fn rate(results: &TransmutationResults, kind: &str) -> f64 {
    results
        .get_reaction_rates(MATERIAL_ID, 0)
        .expect("the pulse has rates")[NUCLIDE][kind]
        .iter()
        .map(|(_, rate)| rate)
        .sum()
}

/// A product's atom density [atoms/b-cm] after the pulse.
fn product(results: &TransmutationResults, nuclide: &str) -> f64 {
    results
        .get_nuclide_density(MATERIAL_ID, nuclide, 1)
        .unwrap_or(0.0)
}

fn fixtures() -> Option<(String, yani::LoadedChain)> {
    let found = yamc_test_cache::transport_nuclide(NUCLIDE).zip(chain());
    if found.is_none() {
        eprintln!("skipping: {NUCLIDE} or chain fixture not fetched");
    }
    found
}

/// Each channel's rate and product in `geometry`, over the same in the uncut
/// cube, with the material's volume as the results report it.
fn against_the_uncut_cube(
    geometry: fn(&str) -> MeshGeometry,
    method: &str,
    path: &str,
    loaded: &yani::LoadedChain,
) -> (Vec<f64>, Option<f64>) {
    let reference = transmute(one_volume(path), method, loaded);
    let cut = transmute(geometry(path), method, loaded);
    let mut ratios = Vec::new();
    for (kind, nuclide) in CHANNELS {
        let (r, p) = (rate(&reference, kind), product(&reference, nuclide));
        assert!(r > 0.0 && p > 0.0, "{method} {kind}: the uncut cube reacts");
        let rates = rate(&cut, kind) / r;
        let products = product(&cut, nuclide) / p;
        eprintln!("{method} {kind}: rate x{rates:.6}, {nuclide} x{products:.6} of the uncut cube");
        ratios.extend([rates, products]);
    }
    let volume = cut
        .get_final_material(MATERIAL_ID)
        .expect("material")
        .volume;
    (ratios, volume)
}

/// One material under both volume names has one slot holding the whole
/// cube's volume, so its rates and products are the uncut cube's.
#[test]
fn one_material_over_two_volumes_transmutes_as_the_uncut_cube() {
    let Some((path, loaded)) = fixtures() else {
        return;
    };
    let geometry = two_volumes_one_material(&path);
    assert_eq!(geometry.materials.len(), 1);
    assert!((geometry.materials[0].volume.unwrap() - 1.0).abs() < 1e-9);

    for method in ["coupled", "independent"] {
        let (ratios, volume) =
            against_the_uncut_cube(two_volumes_one_material, method, &path, &loaded);
        for ratio in ratios {
            assert!(
                (ratio - 1.0).abs() < TOLERANCE,
                "{method}: x{ratio} of the uncut cube"
            );
        }
        let volume = volume.expect("the transmuted material has a volume");
        assert!((volume - 1.0).abs() < 1e-9, "{method}: volume {volume}");
    }
}

/// The cut cube with two iron materials of one id: two slots of 0.5 cm3.
fn two_volumes_two_copies(path: &str) -> MeshGeometry {
    let materials = HashMap::from([
        ("fuel".to_string(), Arc::new(iron(path))),
        ("moderator".to_string(), Arc::new(iron(path))),
    ]);
    mesh(TWO_REGION, &materials)
}

/// One id in two slots is normalized by the total of their volumes, not by
/// the first slot's, and each slot keeps its own volume through the run.
#[test]
fn one_id_in_two_slots_transmutes_as_the_uncut_cube() {
    let Some((path, loaded)) = fixtures() else {
        return;
    };
    let geometry = two_volumes_two_copies(&path);
    assert_eq!(geometry.materials.len(), 2);
    for m in &geometry.materials {
        assert!((m.volume.unwrap() - 0.5).abs() < 1e-9);
    }

    for method in ["coupled", "independent"] {
        let (ratios, volume) =
            against_the_uncut_cube(two_volumes_two_copies, method, &path, &loaded);
        for ratio in ratios {
            assert!(
                (ratio - 1.0).abs() < TOLERANCE,
                "{method}: x{ratio} of the uncut cube"
            );
        }
        let volume = volume.expect("the transmuted material has a volume");
        assert!((volume - 1.0).abs() < 1e-9, "{method}: volume {volume}");
    }
}

/// A sphere of radius `R` around the cube, filled with iron.
const R: f64 = 1.5;

fn sphere() -> Region {
    let sphere = Arc::new(Surface {
        surface_id: None,
        kind: SurfaceKind::Sphere {
            x0: 0.5,
            y0: 0.5,
            z0: 0.5,
            radius: R,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    Region::new_from_halfspace(HalfspaceType::Below(sphere))
}

fn iron_with_volume(path: &str, volume: f64) -> Arc<Material> {
    let mut m = iron(path);
    m.volume = Some(volume);
    Arc::new(m)
}

/// The iron sphere as one CSG cell.
fn csg_sphere(path: &str) -> Model {
    let cell = Cell::new(Some(1), sphere(), None, Some(0));
    let volume = 4.0 / 3.0 * std::f64::consts::PI * R.powi(3);
    let geometry = Geometry::new(vec![cell], vec![iron_with_volume(path, volume)]).unwrap();
    Model::new(geometry, vec![], vec![])
}

/// The same sphere as a host cell of iron filled by the iron cube: one id in
/// a host slot (the complement's volume) and a fill slot (the cube's).
fn host_plus_fill(path: &str) -> Model {
    let host = Cell::new(Some(1), sphere(), None, Some(0));
    let complement = 4.0 / 3.0 * std::f64::consts::PI * R.powi(3) - 1.0;
    let geometry = Geometry::new_with_fills(
        vec![host],
        vec![iron_with_volume(path, complement)],
        vec![CellFillSpec {
            host_cell_index: 0,
            mesh_geometry: one_volume(path),
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            allow_clipping: false,
        }],
    )
    .unwrap();
    Model::new(geometry, vec![], vec![])
}

/// A material that is both a fill's and its host's is normalized by both
/// volumes, so it transmutes as the one CSG cell it amounts to.
#[test]
fn a_host_and_fill_material_transmutes_as_one_cell() {
    let Some((path, loaded)) = fixtures() else {
        return;
    };
    assert_eq!(host_plus_fill(&path).geometry.materials().len(), 2);
    for method in ["coupled", "independent"] {
        let reference = run(csg_sphere(&path), method, &loaded);
        let filled = run(host_plus_fill(&path), method, &loaded);
        for (kind, nuclide) in CHANNELS {
            let rates = rate(&filled, kind) / rate(&reference, kind);
            let products = product(&filled, nuclide) / product(&reference, nuclide);
            eprintln!("{method} {kind}: rate x{rates:.6}, {nuclide} x{products:.6} of one cell");
            for ratio in [rates, products] {
                assert!((ratio - 1.0).abs() < TOLERANCE, "{method} {kind}: x{ratio}");
            }
        }
    }
}
