//! GPU cylindrical mesh-tally support (issue #279), on-hardware parity check.
//!
//! A `Tally(mesh=CylindricalMesh, scores=[flux])` carries a MeshFilter and NO
//! CellFilter: it voxel-bins by `(r, phi, z)` position. The GPU path runs the
//! analytic cylindrical voxel walk in the kernel (`cyl_mesh_score_src_acc`,
//! per-source direct-to-`src_acc` variance) and must reproduce the CPU mesh
//! tally. This test runs the SAME single-nuclide sphere model on the CPU
//! (`simulate_transport`) and the GPU (`run_on_gpu`) and compares the per-voxel
//! flux mean and std_dev, for a full-2pi mesh and a central-hole mesh (whose
//! source-at-origin tracks all enter through the inner shell, exercising the
//! `first_entry` re-entry path).
//!
//! Self-skips if the endf-b8.1 C12 cache or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu,mesh --release \
//!       --test gpu_cyl_mesh_tally -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::f64::consts::TAU;
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
use yamc_tallies::mesh::CylindricalMesh;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 90210;
const RADIUS: f64 = 30.0;
const SOURCE_E: f64 = 14_000_000.0;
const MAX_STEPS: u32 = 10_000;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn nuclide_sphere(nuclide: &str, density: f64) -> Geometry {
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
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(nuclide.to_string(), cache_dir(nuclide))]);
    material.read_nuclear_data(&nm, None).unwrap();

    let cell = Cell::new(Some(1), region, Some("mat".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// A `mesh=`-only flux tally (no CellFilter) over the given cylindrical mesh.
fn cyl_mesh_flux_tally(mesh: CylindricalMesh, n_batches: usize) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters
        .push(Filter::Mesh(MeshFilter::new_cylindrical(mesh)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(n_batches);
    Arc::new(t)
}

fn build_model(
    geometry: Geometry,
    tallies: Vec<Arc<Tally>>,
    total_particles: usize,
) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![neutron_source()], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
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

/// Run the sphere model on CPU and GPU with the given cylindrical mesh and
/// assert per-voxel flux parity. `label` names the case; `worst_dev_bound` is
/// the max allowed worst well-populated-voxel mean deviation.
fn assert_cyl_parity(mesh: CylindricalMesh, total: usize, label: &str, worst_dev_bound: f64) {
    let nuclide = "C12";
    let density = 2.0;
    let n_voxels = mesh.num_bins();

    // CPU reference.
    let cpu_t = cyl_mesh_flux_tally(mesh.clone(), 10);
    let (mut cpu_model, csettings) = build_model(
        nuclide_sphere(nuclide, density),
        vec![Arc::clone(&cpu_t)],
        total,
    );
    cpu_model
        .simulate_transport(&csettings)
        .expect("CPU run failed");
    let cpu_mean = cpu_t.get_mean().to_vec();
    let cpu_sd = cpu_t.get_std_dev().to_vec();

    // GPU.
    let gpu_t = cyl_mesh_flux_tally(mesh, 10);
    let (mut gpu_model, gsettings) = build_model(
        nuclide_sphere(nuclide, density),
        vec![Arc::clone(&gpu_t)],
        total,
    );
    run_gpu_retry(&mut gpu_model, &gsettings).expect("GPU run failed");
    let gpu_mean = gpu_t.get_mean().to_vec();
    let gpu_sd = gpu_t.get_std_dev().to_vec();

    assert_eq!(cpu_mean.len(), n_voxels, "{label}: voxel count");
    assert_eq!(
        cpu_mean.len(),
        gpu_mean.len(),
        "{label}: CPU/GPU voxel counts"
    );

    let cpu_total: f64 = cpu_mean.iter().sum();
    let gpu_total: f64 = gpu_mean.iter().sum();
    assert!(cpu_total > 0.0, "{label}: no CPU flux scored into the mesh");
    assert!(gpu_total > 0.0, "{label}: no GPU flux scored into the mesh");
    let total_ratio = gpu_total / cpu_total;
    eprintln!(
        "[{label}] total flux: CPU={cpu_total:.5e} GPU={gpu_total:.5e} (ratio {total_ratio:.4})"
    );
    assert!(
        (0.98..=1.02).contains(&total_ratio),
        "{label}: GPU/CPU total mesh flux ratio {total_ratio:.4} outside [0.98, 1.02]"
    );

    // Per-voxel parity on well-populated voxels (>= 5% of the peak voxel).
    let max_mean = cpu_mean.iter().cloned().fold(0.0f64, f64::max);
    let mut worst_mean_dev = 0.0f64;
    let mut sd_ratios = Vec::new();
    let mut n_checked = 0usize;
    for i in 0..cpu_mean.len() {
        if cpu_mean[i] < 0.05 * max_mean || cpu_sd[i] <= 0.0 || gpu_sd[i] <= 0.0 {
            continue;
        }
        n_checked += 1;
        let mean_dev = (gpu_mean[i] - cpu_mean[i]).abs() / cpu_mean[i];
        worst_mean_dev = worst_mean_dev.max(mean_dev);
        sd_ratios.push(gpu_sd[i] / cpu_sd[i]);
    }
    assert!(
        n_checked >= 6,
        "{label}: too few well-populated voxels ({n_checked}) to judge parity"
    );
    sd_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_sd = sd_ratios[sd_ratios.len() / 2];
    eprintln!(
        "[{label}] per-voxel: {n_checked} well-populated voxels, worst mean dev {:.1}%, \
         median GPU/CPU std_dev ratio {median_sd:.3}",
        100.0 * worst_mean_dev
    );
    assert!(
        worst_mean_dev < worst_dev_bound,
        "{label}: worst well-populated-voxel mean deviation {:.1}% exceeds {:.0}%",
        100.0 * worst_mean_dev,
        100.0 * worst_dev_bound
    );
    assert!(
        (0.85..=1.18).contains(&median_sd),
        "{label}: median GPU/CPU std_dev ratio {median_sd:.3} not near 1"
    );
}

/// Full-2pi cylindrical mesh whose innermost ring touches the axis (r_min = 0):
/// exercises the seam wrap and the through-axis radial-perigee event.
#[test]
fn gpu_cpu_cyl_flux_parity_full_phi() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_cpu_cyl_flux_parity_full_phi: data/GPU absent");
        return;
    }
    // 4 radial rings x 4 azimuthal sectors x 4 z layers = 64 voxels.
    let mesh = CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (0.0, RADIUS),
        (0.0, TAU),
        (-RADIUS, RADIUS),
        [4, 4, 4],
    );
    assert_cyl_parity(mesh, 200_000, "full_phi", 0.06);
}

/// Central-hole cylindrical mesh (r_min = 10): the source at the origin sits in
/// the untallied hole, so every track enters the mesh through the inner shell,
/// stressing the `first_entry` re-entry path on the GPU.
#[test]
fn gpu_cpu_cyl_flux_parity_central_hole() {
    if !data_present("C12") || !gpu_available() {
        eprintln!("skipping gpu_cpu_cyl_flux_parity_central_hole: data/GPU absent");
        return;
    }
    // 3 radial rings (r in [10, 30]) x 4 sectors x 3 z layers = 36 voxels.
    let mesh = CylindricalMesh::uniform(
        [0.0, 0.0, 0.0],
        (10.0, RADIUS),
        (0.0, TAU),
        (-RADIUS, RADIUS),
        [3, 4, 3],
    );
    // More scatter (fewer particles reach the outer rings), so a looser bound.
    assert_cyl_parity(mesh, 400_000, "central_hole", 0.08);
}
