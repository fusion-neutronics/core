//! GPU rectangular mesh tallies on the COUPLED neutron->photon path (issue
//! #234), on-hardware parity check.
//!
//! A 14 MeV neutron point source in an Fe56 sphere with
//! `transport_secondary_photons = true`. The coupled dispatch scores a
//! neutron-filtered mesh flux tally in the neutron kernel (per-source direct,
//! `has_mesh_n`) and a photon-filtered mesh flux tally in the drained photon
//! sub-pass (`has_mesh_p`). Both must reproduce the CPU coupled mesh tallies:
//!   - neutron mesh flux: tight parity (same transport, matched stream);
//!   - photon mesh flux: the wider coupled-photon band the `gpu_coupled_photon`
//!     test already documents (the GPU photon kernel is an approximate mirror).
//!
//! Self-skips if the bundled Fe data or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu,mesh --release \
//!       --test gpu_coupled_mesh_tally -- --nocapture --test-threads=1
#![cfg(all(feature = "gpu", feature = "mesh", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::GpuRunResult;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 424242;
const RADIUS: f64 = 10.0;
const MAX_STEPS: u32 = 10_000;

fn data_present() -> bool {
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

fn neutron_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14_000_000.0], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// A `mesh=`-only flux tally filtered to `particle`: `shape^3` voxels over the
/// sphere bounding box, row-major (the GPU-supported layout).
fn mesh_flux_tally(particle: ParticleType, shape: usize, n_batches: usize) -> Arc<Tally> {
    let mesh = RegularRectangularMesh::new(
        [-RADIUS, -RADIUS, -RADIUS],
        [RADIUS, RADIUS, RADIUS],
        [shape, shape, shape],
    );
    let mut t = Tally::new();
    t.filters.push(Filter::Mesh(MeshFilter::new(mesh)));
    t.filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(particle)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(n_batches);
    Arc::new(t)
}

fn build_model(tallies: Vec<Arc<Tally>>, total_particles: usize) -> (Model, TransportSettings) {
    let mut model = Model::new(fe_sphere(), vec![neutron_source()], tallies);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
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
fn gpu_cpu_coupled_mesh_flux_parity() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping gpu_cpu_coupled_mesh_flux_parity: Fe data / GPU absent");
        return;
    }
    let shape = 4usize; // 4x4x4 = 64 voxels
    let total: usize = 200_000;

    // CPU reference: neutron + photon mesh flux from one coupled run.
    let cpu_n = mesh_flux_tally(ParticleType::Neutron, shape, 10);
    let cpu_p = mesh_flux_tally(ParticleType::Photon, shape, 10);
    let (mut cpu_model, csettings) =
        build_model(vec![Arc::clone(&cpu_n), Arc::clone(&cpu_p)], total);
    cpu_model
        .simulate_transport(&csettings)
        .expect("CPU run failed");
    let cpu_n_mean = cpu_n.get_mean().to_vec();
    let cpu_p_mean = cpu_p.get_mean().to_vec();

    // GPU: same coupled run through the two-pass dispatch.
    let gpu_n = mesh_flux_tally(ParticleType::Neutron, shape, 10);
    let gpu_p = mesh_flux_tally(ParticleType::Photon, shape, 10);
    let (mut gpu_model, gsettings) =
        build_model(vec![Arc::clone(&gpu_n), Arc::clone(&gpu_p)], total);
    run_gpu_retry(&mut gpu_model, &gsettings).expect("GPU run failed");
    let gpu_n_mean = gpu_n.get_mean().to_vec();
    let gpu_p_mean = gpu_p.get_mean().to_vec();

    let nvox = shape * shape * shape;
    assert_eq!(cpu_n_mean.len(), nvox, "neutron voxel count");
    assert_eq!(cpu_p_mean.len(), nvox, "photon voxel count");
    assert_eq!(gpu_n_mean.len(), nvox);
    assert_eq!(gpu_p_mean.len(), nvox);

    // --- Neutron mesh: tight parity (same transport). ---
    let cpu_n_total: f64 = cpu_n_mean.iter().sum();
    let gpu_n_total: f64 = gpu_n_mean.iter().sum();
    assert!(
        cpu_n_total > 0.0 && gpu_n_total > 0.0,
        "no neutron mesh flux"
    );
    let n_ratio = gpu_n_total / cpu_n_total;
    eprintln!(
        "coupled neutron mesh flux: CPU={cpu_n_total:.5e} GPU={gpu_n_total:.5e} (ratio {n_ratio:.4})"
    );
    assert!(
        (0.97..=1.03).contains(&n_ratio),
        "coupled neutron mesh flux ratio {n_ratio:.4} outside [0.97, 1.03]"
    );

    // --- Photon mesh: the wider coupled-photon band (approximate GPU kernel). ---
    let cpu_p_total: f64 = cpu_p_mean.iter().sum();
    let gpu_p_total: f64 = gpu_p_mean.iter().sum();
    assert!(
        cpu_p_total > 0.0 && gpu_p_total > 0.0,
        "no coupled photon mesh flux scored"
    );
    let p_ratio = gpu_p_total / cpu_p_total;
    eprintln!(
        "coupled photon mesh flux: CPU={cpu_p_total:.5e} GPU={gpu_p_total:.5e} (ratio {p_ratio:.4})"
    );
    assert!(
        (0.75..=1.25).contains(&p_ratio),
        "coupled photon mesh flux ratio {p_ratio:.4} outside the coupled band [0.75, 1.25]"
    );

    // Per-voxel neutron parity on well-populated voxels (>= 10% of the peak).
    let max_n = cpu_n_mean.iter().cloned().fold(0.0f64, f64::max);
    let mut worst_n = 0.0f64;
    let mut n_checked = 0usize;
    for i in 0..nvox {
        if cpu_n_mean[i] < 0.10 * max_n {
            continue;
        }
        n_checked += 1;
        worst_n = worst_n.max((gpu_n_mean[i] - cpu_n_mean[i]).abs() / cpu_n_mean[i]);
    }
    assert!(n_checked >= 6, "too few neutron voxels ({n_checked})");
    eprintln!(
        "coupled neutron per-voxel: {n_checked} voxels, worst mean dev {:.1}%",
        100.0 * worst_n
    );
    assert!(
        worst_n < 0.08,
        "worst neutron voxel dev {:.1}% exceeds 8%",
        100.0 * worst_n
    );
}
