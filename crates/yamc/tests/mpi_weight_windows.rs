//! DeGVR weight-window generation under MPI returns the same bounds on every
//! rank, and those bounds are the serial answer at the same total budget.
//!
//! `cargo test --features mpi` runs this binary as a single process, so the
//! test relaunches itself under `mpirun -np 2`. Each rank writes its bounds to
//! a file; the parent then checks the two ranks against each other and against
//! a serial generation it runs in-process.
#![cfg(feature = "mpi")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc::variance_reduction::WeightWindowGeneratorDeGVR;
use yamc_materials::Material;
use yamc_particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::RegularRectangularMesh;

const TEST_NAME: &str = "weight_windows_are_identical_on_every_rank";
const OUT_DIR_ENV: &str = "YAMC_MPI_WW_OUT_DIR";
const RANKS: usize = 2;

/// A 14 MeV point source at the centre of a 30 cm iron sphere, with the
/// window mesh inside the sphere.
fn iron_sphere_model() -> Model {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 30.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/Fe56.arrow");
    let nuclide_map = HashMap::from([("Fe56".to_string(), data.to_string_lossy().into_owned())]);
    material.read_nuclear_data(&nuclide_map, None).unwrap();

    let cell = Cell::new(Some(1), region, Some("iron".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    Model::new(geometry, vec![source], Vec::new())
}

fn generator() -> WeightWindowGeneratorDeGVR {
    WeightWindowGeneratorDeGVR {
        mesh: RegularRectangularMesh::new([-20.0, -20.0, -20.0], [20.0, 20.0, 20.0], [4, 4, 4]),
        energy_bins: None,
        particles: vec![ParticleType::Neutron],
        photon_energy: None,
        density_reduction: Some(4.0),
        ratio: 5.0,
        survival_factor: 3.0,
        max_split: 10,
        weight_floor: 1e-38,
    }
}

/// Lower bounds of the generated neutron window, at a fixed total budget.
fn lower_bounds() -> Vec<f64> {
    let settings = TransportSettings {
        total_particles: Some(4000),
        seed: 7,
        ..Default::default()
    };
    let windows = iron_sphere_model()
        .generate_weight_windows(&generator(), &settings)
        .expect("generate weight windows");
    windows[0].lower_bounds.clone()
}

fn write_bounds(path: &Path, bounds: &[f64]) {
    let text: Vec<String> = bounds
        .iter()
        .map(|b| format!("{:016x}", b.to_bits()))
        .collect();
    std::fs::write(path, text.join("\n")).expect("write rank bounds");
}

fn read_bounds(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .map(|l| f64::from_bits(u64::from_str_radix(l, 16).expect("hex f64 bits")))
        .collect()
}

fn under_mpi_launcher() -> bool {
    std::env::var("OMPI_COMM_WORLD_SIZE").is_ok() || std::env::var("PMI_SIZE").is_ok()
}

fn rank_file(dir: &Path, rank: i32) -> PathBuf {
    dir.join(format!("rank{rank}.txt"))
}

#[test]
fn weight_windows_are_identical_on_every_rank() {
    if under_mpi_launcher() {
        // One rank of the relaunch: generate and record this rank's bounds.
        let dir = PathBuf::from(std::env::var(OUT_DIR_ENV).expect("output dir from the parent"));
        let bounds = lower_bounds();
        write_bounds(&rank_file(&dir, yamc::mpi_context::mpi_rank()), &bounds);
        yamc::mpi_context::mpi_finalize();
        return;
    }

    let dir = std::env::temp_dir().join(format!("yamc_mpi_ww_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = std::env::current_exe().unwrap();
    let status = Command::new("mpirun")
        .args(["-np", &RANKS.to_string()])
        .arg(&exe)
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(OUT_DIR_ENV, &dir)
        .status()
        .expect("run mpirun: the mpi feature needs an MPI launcher on PATH");
    assert!(status.success(), "mpirun -np {RANKS} failed: {status}");

    let per_rank: Vec<Vec<f64>> = (0..RANKS as i32)
        .map(|r| read_bounds(&rank_file(&dir, r)))
        .collect();
    std::fs::remove_dir_all(&dir).ok();
    let serial = lower_bounds();

    let sum = |b: &[f64]| b.iter().sum::<f64>();
    eprintln!("sum of lower bounds: serial {:.6}", sum(&serial));
    for (r, b) in per_rank.iter().enumerate() {
        let max_rel = b
            .iter()
            .zip(&serial)
            .map(|(m, s)| ((m - s) / s).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "sum of lower bounds: rank {r} {:.6} (max per-voxel relative difference from serial {max_rel:.3e})",
            sum(b)
        );
    }

    for (r, b) in per_rank.iter().enumerate().skip(1) {
        assert_eq!(
            b, &per_rank[0],
            "rank {r} returned different weight-window bounds from rank 0"
        );
    }
    // Same histories as the serial run, folded in a different order, so the
    // bounds agree to rounding rather than bit for bit.
    assert_eq!(per_rank[0].len(), serial.len());
    for (i, (m, s)) in per_rank[0].iter().zip(&serial).enumerate() {
        let tol = 1e-9 * s.abs().max(f64::MIN_POSITIVE);
        assert!(
            (m - s).abs() <= tol,
            "voxel {i}: MPI lower bound {m:e} differs from serial {s:e}"
        );
    }
}
