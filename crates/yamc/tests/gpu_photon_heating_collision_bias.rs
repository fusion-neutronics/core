//! Task #68 decisive measurement: is the GPU-high COLLISION-estimator photon
//! heating (~1.008 in pure-photon mode) a real bias or just the collision
//! estimator's higher variance?
//!
//! Setup: a 1.25 MeV (Co-60 mean) isotropic photon point source at the centre
//! of an Fe sphere (r=5 cm, CSG, vacuum), photon data `tests/Fe.arrow`. ONE
//! collision-estimator photon-heating tally. CPU (`Model::simulate_transport`)
//! vs GPU (`yamc::gpu::run_on_gpu`), at high statistics over several seeds.
//!
//! Decision rule (the task's): if GPU/CPU collision-estimator photon heating
//! averages ~1.0 across seeds / converges to ~1.0 with statistics, the ~1.008
//! was variance -> close, no fix. If it stays systematically ~1.008 high across
//! seeds AND high statistics, it is a real bias.
//!
//! Run it (single-threaded -- the shared cubecl client serialises launches):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_photon_heating_collision_bias -- --nocapture --test-threads=1
//! Self-skips if the Arrow data is absent or no f64 GPU adapter is present.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const RADIUS: f64 = 5.0;
const MAX_STEPS: u32 = 5_000;

fn data_present() -> bool {
    std::path::Path::new("tests/Fe.arrow").exists()
        && std::path::Path::new("tests/Fe56.arrow").exists()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn fe_sphere(cell_id: u32) -> (Geometry, u32) {
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
    // Mirror gpu_cpu_comparison_matrix::fe_sphere: load neutron + photon data
    // through read_nuclear_data (this populates photon_data_paths so the
    // auto-enabled secondary-photon validation passes), then init photon data.
    let nm = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);
    let photon_paths = HashMap::from([("Fe".to_string(), "tests/Fe.arrow".to_string())]);
    material
        .read_nuclear_data(&nm, Some(&photon_paths))
        .unwrap();
    material.init_photon_data(&photon_paths).unwrap();

    let cell = Cell::new(Some(cell_id), region, Some("fe".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_id)
}

fn photon_source() -> ParticleSource {
    ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.25e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

/// One photon-heating tally on the cell with the given estimator.
fn heating_tally_est(cell_id: u32, n_batches: usize, estimator: Estimator) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Photon,
    )));
    t.scores = vec!["heating".parse::<Score>().unwrap()];
    t.estimator = estimator;
    t.initialize_batches(n_batches);
    Arc::new(t)
}

fn heating_tally(cell_id: u32, n_batches: usize) -> Arc<Tally> {
    heating_tally_est(cell_id, n_batches, Estimator::Collision)
}

fn build_model(
    geometry: Geometry,
    tallies: Vec<Arc<Tally>>,
    seed: u64,
    n_per_batch: usize,
    n_batches: usize,
) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![photon_source()], tallies);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(n_per_batch * n_batches),
        seed,
        ..Default::default()
    };
    (model, settings)
}

fn tally_sum(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum::<f64>()
}

fn run_cpu(seed: u64, n_per_batch: usize, n_batches: usize) -> f64 {
    let (geometry, cell_id) = fe_sphere(1);
    let t = heating_tally(cell_id, n_batches);
    let (mut model, settings) =
        build_model(geometry, vec![Arc::clone(&t)], seed, n_per_batch, n_batches);
    model
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU run");
    tally_sum(&t)
}

/// GPU run with bounded retry on transient errors (shared AMD GPU can return a
/// `BufferAsyncError` under contention).
fn run_gpu(seed: u64, n_per_batch: usize, n_batches: usize) -> f64 {
    let mut last_err = String::new();
    for attempt in 0..6 {
        let (geometry, cell_id) = fe_sphere(1);
        let t = heating_tally(cell_id, n_batches);
        let (mut model, settings) =
            build_model(geometry, vec![Arc::clone(&t)], seed, n_per_batch, n_batches);
        match yamc::gpu::run_on_gpu(&mut model, &settings) {
            Ok(_) => return tally_sum(&t),
            Err(e) => {
                last_err = e.to_string();
                eprintln!("  GPU attempt {attempt} failed ({last_err}); retrying");
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
    panic!("GPU run failed after retries: {last_err}");
}

#[test]
fn gpu_photon_heating_collision_bias() {
    if !data_present() {
        eprintln!("skipping -- tests/Fe.arrow not found");
        return;
    }
    if !gpu_available() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    // High statistics: 1M per seed in the quick pass, plus a convergence check.
    let seeds: [u64; 5] = [1, 7, 42, 1234, 98765];
    let n_per_batch = 100_000usize;
    let n_batches = 10usize; // 1M histories per seed

    println!("\n# Task #68: collision-estimator photon-heating GPU/CPU, 1.25 MeV Fe sphere");
    println!(
        "\n{} histories per seed ({n_per_batch} x {n_batches}).",
        n_per_batch * n_batches
    );
    println!("\n| seed | CPU heating | GPU heating | GPU/CPU |");
    println!("|---|---|---|---|");

    let mut ratios = Vec::new();
    for &seed in &seeds {
        let cpu = run_cpu(seed, n_per_batch, n_batches);
        let gpu = run_gpu(seed, n_per_batch, n_batches);
        let ratio = gpu / cpu;
        ratios.push(ratio);
        println!("| {seed} | {cpu:.6e} | {gpu:.6e} | {ratio:.5} |");
    }

    let n = ratios.len() as f64;
    let mean = ratios.iter().sum::<f64>() / n;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    let sem = sd / n.sqrt();
    println!(
        "\nmean GPU/CPU = {mean:.5}  (sd {sd:.5}, sem {sem:.5}, n={})",
        ratios.len()
    );
    println!("mean deviation from 1.0 = {:.3}%", (mean - 1.0) * 100.0);

    // Convergence check at one seed: 5M histories.
    let seed = 42u64;
    let cpu_hi = run_cpu(seed, n_per_batch, 50);
    let gpu_hi = run_gpu(seed, n_per_batch, 50);
    println!(
        "\nconvergence (seed {seed}, 5M histories): CPU {cpu_hi:.6e}  GPU {gpu_hi:.6e}  ratio {:.5}",
        gpu_hi / cpu_hi
    );

    // Diagnostic: CPU collision (analog deposit) vs CPU track-length (KERMA).
    // If these disagree by ~the same ~0.9%, the GPU-high collision heating is
    // the analog-vs-KERMA estimator gap (GPU collision uses KERMA; CPU
    // collision uses the analog deposit).
    {
        let (g, cid) = fe_sphere(1);
        let t_coll = heating_tally_est(cid, n_batches, Estimator::Collision);
        let t_tl = heating_tally_est(cid, n_batches, Estimator::TrackLength);
        let (mut m, settings) = build_model(
            g,
            vec![Arc::clone(&t_coll), Arc::clone(&t_tl)],
            42,
            n_per_batch,
            n_batches,
        );
        m.simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU two-estimator run");
        let coll = tally_sum(&t_coll);
        let tl = tally_sum(&t_tl);
        println!(
            "\nCPU collision (analog) = {coll:.6e}  CPU track-length (KERMA) = {tl:.6e}  \
             collision/track-length = {:.5}",
            coll / tl
        );
        println!(
            "CPU track-length (KERMA) vs GPU collision: GPU/CPU_TL = {:.5}",
            gpu_hi / tl
        );
    }

    // Informational only -- the test never fails; the printed numbers drive the
    // bias-vs-variance decision. A wide guard catches a gross regression.
    assert!(
        (mean - 1.0).abs() < 0.1,
        "mean GPU/CPU {mean} wildly off -- not the ~1.008 effect under study"
    );
}
