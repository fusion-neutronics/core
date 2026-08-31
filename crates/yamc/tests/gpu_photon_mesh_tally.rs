//! GPU rectangular mesh-tally support on the PHOTON path (issue #234),
//! on-hardware parity check.
//!
//! A `Tally(mesh=RegularRectangularMesh, scores=[flux])` carries a MeshFilter
//! and NO CellFilter: it voxel-bins the photon flux by spatial position. The
//! GPU photon kernel runs the Amanatides-Woo voxel walk per track-length step
//! (per-source direct-to-`src_acc` variance, issue #234) and must reproduce the
//! CPU photon mesh tally. Runs the SAME Fe-sphere / Co60 photon-source model on
//! the CPU (`simulate_transport`) and the GPU (`run_on_gpu`) and compares the
//! per-voxel flux mean and total.
//!
//! Self-skips if the bundled Fe photon data or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu,mesh --release \
//!       --test gpu_photon_mesh_tally -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", feature = "mesh", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::GpuRunResult;
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
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260724;
const RADIUS: f64 = 3.0;
const SOURCE_E: f64 = 1.25e6; // Co60 mean line
const MAX_STEPS: u32 = 5_000;

fn fe_data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").is_dir()
        && std::path::Path::new("tests/Fe.arrow").is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn fe_sphere() -> Geometry {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(1), region, Some("fe".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// A `mesh=`-only photon flux tally (no CellFilter): `shape^3` voxels over the
/// sphere bounding box, row-major (the GPU-supported layout).
fn mesh_flux_tally(shape: usize, n_batches: usize) -> Arc<Tally> {
    let mesh = RegularRectangularMesh::new(
        [-RADIUS, -RADIUS, -RADIUS],
        [RADIUS, RADIUS, RADIUS],
        [shape, shape, shape],
    );
    let mut t = Tally::new();
    t.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(n_batches);
    Arc::new(t)
}

fn build_model(tallies: Vec<Arc<Tally>>, total_particles: usize) -> (Model, TransportSettings) {
    let mut model = Model::new(fe_sphere(), vec![photon_source()], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(total_particles),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<GpuRunResult, String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(r) => return Ok(r),
            Err(e) => {
                last = e.to_string();
                let transient = last.contains("BufferAsync") || last.contains("buffer async");
                if !transient {
                    return Err(last);
                }
                eprintln!("GPU transient error (attempt {attempt}): {last}; retrying");
            }
        }
    }
    Err(last)
}

#[test]
fn gpu_cpu_photon_mesh_flux_parity() {
    if !fe_data_present() || !gpu_available() {
        eprintln!("skipping gpu_cpu_photon_mesh_flux_parity: Fe data / GPU absent");
        return;
    }
    let shape = 4usize; // 4x4x4 = 64 voxels
    let total: usize = 200_000;

    // CPU reference.
    let cpu_t = mesh_flux_tally(shape, 10);
    let (mut cpu_model, csettings) = build_model(vec![Arc::clone(&cpu_t)], total);
    cpu_model
        .simulate_transport(&csettings)
        .expect("CPU run failed");
    let cpu_mean = cpu_t.get_mean().to_vec();

    // GPU.
    let gpu_t = mesh_flux_tally(shape, 10);
    let (mut gpu_model, gsettings) = build_model(vec![Arc::clone(&gpu_t)], total);
    run_gpu_retry(&mut gpu_model, &gsettings).expect("GPU run failed");
    let gpu_mean = gpu_t.get_mean().to_vec();

    assert_eq!(cpu_mean.len(), shape * shape * shape, "voxel count");
    assert_eq!(
        cpu_mean.len(),
        gpu_mean.len(),
        "CPU/GPU voxel counts differ"
    );

    let cpu_total: f64 = cpu_mean.iter().sum();
    let gpu_total: f64 = gpu_mean.iter().sum();
    assert!(cpu_total > 0.0, "no CPU photon flux scored into the mesh");
    assert!(gpu_total > 0.0, "no GPU photon flux scored into the mesh");
    let total_ratio = gpu_total / cpu_total;
    eprintln!(
        "photon mesh total flux: CPU={cpu_total:.5e} GPU={gpu_total:.5e} (ratio {total_ratio:.4})"
    );
    assert!(
        (0.97..=1.03).contains(&total_ratio),
        "GPU/CPU total photon mesh flux ratio {total_ratio:.4} outside [0.97, 1.03]"
    );

    // Per-voxel parity on well-populated voxels (>= 10% of the peak voxel).
    let max_mean = cpu_mean.iter().cloned().fold(0.0f64, f64::max);
    let mut worst_mean_dev = 0.0f64;
    let mut n_checked = 0usize;
    for i in 0..cpu_mean.len() {
        if cpu_mean[i] < 0.10 * max_mean {
            continue;
        }
        n_checked += 1;
        let mean_dev = (gpu_mean[i] - cpu_mean[i]).abs() / cpu_mean[i];
        worst_mean_dev = worst_mean_dev.max(mean_dev);
    }
    assert!(
        n_checked >= 6,
        "too few well-populated voxels ({n_checked}) to judge parity"
    );
    eprintln!(
        "photon mesh per-voxel: {n_checked} well-populated voxels, worst mean dev {:.1}%",
        100.0 * worst_mean_dev
    );
    // Independent MC estimates of the same per-voxel photon flux; a few % apart.
    assert!(
        worst_mean_dev < 0.08,
        "worst well-populated-voxel photon mean deviation {:.1}% exceeds 8%",
        100.0 * worst_mean_dev
    );
}
