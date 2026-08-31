//! GPU rectangular mesh tallies on the MIXED neutron+photon primary-source path
//! (issue #234), on-hardware parity check.
//!
//! An Fe56 sphere with BOTH a 14 MeV neutron source and a 2 MeV photon source
//! (equal strength). The mixed dispatch splits each chunk into a neutron share
//! (coupled kernel, `has_mesh_n`) and a photon share (the primary photon pass,
//! `has_mesh_p`), and also drains the neutron-induced secondary photons into the
//! photon sub-pass. A photon-filtered mesh flux tally therefore exercises BOTH
//! the primary and the secondary photon mesh scoring; a neutron-filtered mesh
//! tally exercises the neutron pass. Both must reproduce the CPU mixed mesh
//! tallies within the mixed-source statistical band.
//!
//! Self-skips if the bundled Fe data or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu,mesh --release \
//!       --test gpu_mixed_mesh_tally -- --nocapture --test-threads=1
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

const SEED: u64 = 7_654_321;
const RADIUS: f64 = 5.0;
const MAX_STEPS: u32 = 5_000;

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

fn mixed_sources() -> Vec<ParticleSource> {
    vec![
        ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![14.0e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        }),
        ParticleSource::Photon(Source {
            space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![2.0e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        }),
    ]
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
    let mut model = Model::new(fe_sphere(), mixed_sources(), tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model.photon_cutoff_energy = 1000.0;
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
fn gpu_cpu_mixed_mesh_flux_parity() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping gpu_cpu_mixed_mesh_flux_parity: Fe data / GPU absent");
        return;
    }
    let shape = 4usize; // 4x4x4 = 64 voxels
    let total: usize = 200_000;

    // CPU reference.
    let cpu_n = mesh_flux_tally(ParticleType::Neutron, shape, 10);
    let cpu_p = mesh_flux_tally(ParticleType::Photon, shape, 10);
    let (mut cpu_model, csettings) =
        build_model(vec![Arc::clone(&cpu_n), Arc::clone(&cpu_p)], total);
    cpu_model
        .simulate_transport(&csettings)
        .expect("CPU run failed");
    let cpu_n_mean = cpu_n.get_mean().to_vec();
    let cpu_p_mean = cpu_p.get_mean().to_vec();

    // GPU mixed run through the three-pass dispatch.
    let gpu_n = mesh_flux_tally(ParticleType::Neutron, shape, 10);
    let gpu_p = mesh_flux_tally(ParticleType::Photon, shape, 10);
    let (mut gpu_model, gsettings) =
        build_model(vec![Arc::clone(&gpu_n), Arc::clone(&gpu_p)], total);
    run_gpu_retry(&mut gpu_model, &gsettings).expect("GPU run failed");
    let gpu_n_mean = gpu_n.get_mean().to_vec();
    let gpu_p_mean = gpu_p.get_mean().to_vec();

    let nvox = shape * shape * shape;
    assert_eq!(cpu_n_mean.len(), nvox);
    assert_eq!(cpu_p_mean.len(), nvox);
    assert_eq!(gpu_n_mean.len(), nvox);
    assert_eq!(gpu_p_mean.len(), nvox);

    let cpu_n_total: f64 = cpu_n_mean.iter().sum();
    let gpu_n_total: f64 = gpu_n_mean.iter().sum();
    let cpu_p_total: f64 = cpu_p_mean.iter().sum();
    let gpu_p_total: f64 = gpu_p_mean.iter().sum();
    assert!(
        cpu_n_total > 0.0 && gpu_n_total > 0.0 && cpu_p_total > 0.0 && gpu_p_total > 0.0,
        "both source types must fill both mesh tallies on both backends"
    );
    let n_ratio = gpu_n_total / cpu_n_total;
    let p_ratio = gpu_p_total / cpu_p_total;
    eprintln!(
        "mixed neutron mesh flux: CPU={cpu_n_total:.5e} GPU={gpu_n_total:.5e} (ratio {n_ratio:.4})"
    );
    eprintln!(
        "mixed photon  mesh flux: CPU={cpu_p_total:.5e} GPU={gpu_p_total:.5e} (ratio {p_ratio:.4})"
    );
    // Neutron mesh: tight (same transport). Photon mesh: the wider coupled band
    // (primary + neutron-induced secondaries through the approximate GPU kernel).
    assert!(
        (0.96..=1.04).contains(&n_ratio),
        "mixed neutron mesh flux ratio {n_ratio:.4} outside [0.96, 1.04]"
    );
    assert!(
        (0.75..=1.25).contains(&p_ratio),
        "mixed photon mesh flux ratio {p_ratio:.4} outside the coupled band [0.75, 1.25]"
    );
}
