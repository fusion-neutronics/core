//! Convergence targets on the GPU (fusion-neutronics/core#29).
//!
//! A `Model.convergence_targets` entry is defined on a tally's per-history
//! AGGREGATE moments (the total score of each history across the tally's bins),
//! which the GPU's per-bin `sum` / `sum_sq` cannot recover. The neutron kernel
//! now emits each history's per-tally total at its history-end flush and the
//! dispatch folds those into `AggMoments` launch by launch; the fissile path
//! folds each source's per-bin grand total instead (a source's sample spans
//! its fission generations). The launch loop then decides the targets at every
//! chunk boundary, the third stop condition beside the particle cap and the
//! time budget, and the finalized tallies carry the aggregate moments the CPU
//! reports (`Tally::get_agg`).
//!
//! These tests hold the GPU to that contract: an uncapped run stops on the
//! target and reports an aggregate error that honours it; a tighter target runs
//! longer; on a single-bin tally the aggregate moments reproduce the bin
//! statistics exactly (both fold the same per-history sample); on a multi-bin
//! tally the aggregate agrees with the CPU's (same seed, lockstep histories);
//! and a model that transports photons is still refused, in words, before any
//! device work.
//!
//! Run them (needs an f64 GPU and the Fe56 fixture; the fissile case takes
//! U235 from the cache):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_convergence_targets -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::GpuDispatchError;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::welford::AggMoments;
use yamc_tallies::{ConvergenceMetric, ConvergenceTarget, Estimator};

const SEED: u64 = 20260913;
const SOURCE_E: f64 = 14.06e6;
/// Small launch chunk so a stop after a handful of launches is a real
/// decision, not the first chunk overshooting every target at once.
const CHUNK: usize = 2_000;

/// Serialises every test in this binary and scopes the launch-chunk override
/// to it (the override is a process-global env var read inside the dispatch),
/// restoring the previous value on drop. Same shape as `gpu_coupled_variance`.
struct GpuTest {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<String>,
}

impl GpuTest {
    fn with_chunk(chunk: usize) -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let previous = std::env::var("YAMC_GPU_LAUNCH_CHUNK").ok();
        std::env::set_var("YAMC_GPU_LAUNCH_CHUNK", chunk.to_string());
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for GpuTest {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => std::env::set_var("YAMC_GPU_LAUNCH_CHUNK", v),
            None => std::env::remove_var("YAMC_GPU_LAUNCH_CHUNK"),
        }
    }
}

struct Case {
    nuclide: &'static str,
    data: String,
    density: f64,
    radius: f64,
}

fn fe56() -> Option<Case> {
    std::path::Path::new("tests/Fe56.arrow")
        .exists()
        .then(|| Case {
            nuclide: "Fe56",
            data: "tests/Fe56.arrow".to_string(),
            density: 7.874,
            radius: 10.0,
        })
}

fn u235() -> Option<Case> {
    yamc_test_cache::nuclide("U235").map(|path| Case {
        nuclide: "U235",
        data: path,
        density: 18.7,
        radius: 5.0,
    })
}

/// Flux tally named `t`, on the cell; optionally split by `energy_edges`.
fn flux_tally(cell_id: u32, energy_edges: Option<Vec<f64>>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.name = Some("t".to_string());
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    if let Some(edges) = energy_edges {
        t.filters.push(Filter::Energy(EnergyFilter::new(edges)));
    }
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model(
    case: &Case,
    energy_edges: Option<Vec<f64>>,
    targets: Vec<ConvergenceTarget>,
) -> (Model, Arc<Tally>) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: case.radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let mut material = Material::new(
        HashMap::from([(case.nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(case.density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material
        .read_nuclear_data(
            &HashMap::from([(case.nuclide.to_string(), case.data.clone())]),
            None,
        )
        .unwrap();
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let t = flux_tally(cell_id, energy_edges);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&t)]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_max_steps_per_particle = 10_000;
    model.convergence_targets = targets;
    (model, t)
}

fn rel_error(t: &ConvergenceTarget) -> Vec<ConvergenceTarget> {
    vec![t.clone().for_name("t")]
}

fn uncapped() -> TransportSettings {
    TransportSettings {
        total_particles: None,
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    }
}

fn capped(n: usize) -> TransportSettings {
    TransportSettings {
        total_particles: Some(n),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    }
}

/// The relative error of the aggregate mean from its moments, the quantity a
/// `RelativeError` target is judged on.
fn agg_rel_error(agg: &AggMoments) -> f64 {
    let n = agg.n as f64;
    (agg.m2 / ((n - 1.0) * n)).max(0.0).sqrt() / agg.mean.abs()
}

fn gpu_ready() -> bool {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping: no f64 GPU");
        return false;
    }
    true
}

#[test]
fn an_uncapped_neutron_run_stops_when_the_target_is_met() {
    let _env = GpuTest::with_chunk(CHUNK);
    let (Some(case), true) = (fe56(), gpu_ready()) else {
        return;
    };
    let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.01);
    let (mut model, tally) = build_model(&case, None, rel_error(&target));
    let result = yamc::gpu::run_on_gpu(&mut model, &uncapped()).expect("GPU run");
    let agg = tally.get_agg();
    let rel = agg_rel_error(&agg);
    eprintln!(
        "1% target: stopped after {} histories ({} launches), aggregate rel err {rel:.4}",
        result.n_particles,
        result.n_particles / CHUNK
    );
    assert!(result.n_particles > 0);
    assert_eq!(
        result.n_particles % CHUNK,
        0,
        "the stop is decided at a launch boundary"
    );
    assert_eq!(agg.n, result.n_particles as u64);
    assert!(
        rel > 0.0 && rel <= 0.01,
        "aggregate rel err {rel} misses the target"
    );

    // A tighter target needs more histories: about four times as many for half
    // the error, and never fewer.
    let tighter = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.005);
    let (mut model2, tally2) = build_model(&case, None, rel_error(&tighter));
    let result2 = yamc::gpu::run_on_gpu(&mut model2, &uncapped()).expect("GPU run");
    let rel2 = agg_rel_error(&tally2.get_agg());
    eprintln!(
        "0.5% target: stopped after {} histories, aggregate rel err {rel2:.4}",
        result2.n_particles
    );
    assert!(rel2 > 0.0 && rel2 <= 0.005);
    assert!(
        result2.n_particles > result.n_particles,
        "halving the target should take more launches ({} vs {})",
        result2.n_particles,
        result.n_particles
    );
}

#[test]
fn a_particle_cap_alongside_the_target_does_not_change_where_it_stops() {
    let _env = GpuTest::with_chunk(CHUNK);
    let (Some(case), true) = (fe56(), gpu_ready()) else {
        return;
    };
    let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.01);
    let (mut a, _) = build_model(&case, None, rel_error(&target));
    let ra = yamc::gpu::run_on_gpu(&mut a, &uncapped()).expect("GPU run");
    let (mut b, _) = build_model(&case, None, rel_error(&target));
    let rb = yamc::gpu::run_on_gpu(&mut b, &capped(10_000_000)).expect("GPU run");
    assert_eq!(ra.n_particles, rb.n_particles);
    // And a cap below the convergence point wins, as the first stop condition.
    let (mut c, tally_c) = build_model(&case, None, rel_error(&target));
    let rc = yamc::gpu::run_on_gpu(&mut c, &capped(CHUNK)).expect("GPU run");
    assert_eq!(rc.n_particles, CHUNK);
    assert_eq!(tally_c.get_agg().n, CHUNK as u64);
}

/// On a single-bin tally the per-history total IS the bin value, so the
/// aggregate moments and the bin statistics fold the same sample and must
/// agree to rounding: this pins the kernel's per-history emission to the
/// per-bin `sum` / `sum_sq` it sits beside.
#[test]
fn single_bin_aggregate_moments_reproduce_the_bin_statistics() {
    let _env = GpuTest::with_chunk(CHUNK);
    let (Some(case), true) = (fe56(), gpu_ready()) else {
        return;
    };
    let n = 3 * CHUNK;
    let (mut model, tally) = build_model(&case, None, Vec::new());
    let result = yamc::gpu::run_on_gpu(&mut model, &capped(n)).expect("GPU run");
    assert_eq!(result.n_particles, n);
    let agg = tally.get_agg();
    let mean = tally.get_mean()[0];
    let std = tally.get_std_dev()[0];
    let agg_std = (agg.m2 / ((agg.n as f64 - 1.0) * agg.n as f64)).sqrt();
    eprintln!(
        "single bin: mean {mean:.6e} vs agg {:.6e}; std {std:.4e} vs agg {agg_std:.4e}; VoV {:.3e}",
        agg.mean,
        agg.variance_of_variance()
    );
    assert_eq!(agg.n, n as u64);
    assert!(
        ((agg.mean - mean) / mean).abs() < 1e-9,
        "aggregate mean {} vs bin mean {mean}",
        agg.mean
    );
    assert!(
        ((agg_std - std) / std).abs() < 1e-6,
        "aggregate std {agg_std} vs bin std {std}"
    );
    assert!(agg.variance_of_variance() > 0.0);
}

/// The fissile path folds each SOURCE neutron's grand total (its progeny fold
/// into the same sample, so the sample count is the source count): the same
/// identity holds there, and the run stops on the target.
#[test]
fn fissile_per_source_aggregate_stops_on_the_target_and_matches_the_bin() {
    let _env = GpuTest::with_chunk(CHUNK);
    let (Some(case), true) = (u235(), gpu_ready()) else {
        eprintln!("skipping: U235 not in the cache or no GPU");
        return;
    };
    let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.01);
    let (mut model, tally) = build_model(&case, None, rel_error(&target));
    assert!(model.gpu_fission_bank, "the bank is on by default");
    let result = yamc::gpu::run_on_gpu(&mut model, &uncapped()).expect("GPU run");
    let agg = tally.get_agg();
    let rel = agg_rel_error(&agg);
    let mean = tally.get_mean()[0];
    let std = tally.get_std_dev()[0];
    let agg_std = (agg.m2 / ((agg.n as f64 - 1.0) * agg.n as f64)).sqrt();
    eprintln!(
        "U235: stopped after {} sources ({} progeny re-launched), rel err {rel:.4}; \
         mean {mean:.6e} vs agg {:.6e}; std {std:.4e} vs agg {agg_std:.4e}",
        result.n_particles, result.n_bank_relaunched, agg.mean
    );
    assert!(result.n_bank_relaunched > 0, "the fissile loop did not run");
    assert_eq!(result.n_particles % CHUNK, 0);
    assert_eq!(agg.n, result.n_particles as u64);
    assert!(rel > 0.0 && rel <= 0.01);
    assert!(((agg.mean - mean) / mean).abs() < 1e-9);
    assert!(((agg_std - std) / std).abs() < 1e-6);
}

/// On a multi-bin tally the aggregate variance is NOT the sum of the bin
/// variances (one history scores several bins), which is why the kernel emits
/// the per-history total instead of the host deriving it. The CPU folds the
/// same per-history totals on the same seed-keyed histories, so the two
/// aggregates agree within statistics.
#[test]
fn multi_bin_aggregate_agrees_with_the_cpu() {
    let _env = GpuTest::with_chunk(CHUNK);
    let (Some(case), true) = (fe56(), gpu_ready()) else {
        return;
    };
    let edges = vec![1.0e3, 1.0e5, 1.0e6, 5.0e6, 1.5e7];
    let n = 5 * CHUNK;
    let (mut gpu, gpu_t) = build_model(&case, Some(edges.clone()), Vec::new());
    yamc::gpu::run_on_gpu(&mut gpu, &capped(n)).expect("GPU run");
    let (mut cpu, cpu_t) = build_model(&case, Some(edges), Vec::new());
    cpu.simulate_transport(&capped(n)).expect("CPU run");
    let (ga, ca) = (gpu_t.get_agg(), cpu_t.get_agg());
    let bin_mean_sum: f64 = gpu_t.get_mean().iter().sum();
    let bin_var_sum: f64 = gpu_t.get_std_dev().iter().map(|s| s * s).sum::<f64>();
    let agg_var = ga.m2 / ((ga.n as f64 - 1.0) * ga.n as f64);
    eprintln!(
        "multi bin: GPU agg mean {:.6e} (bins sum {bin_mean_sum:.6e}) CPU {:.6e}; \
         GPU agg var {agg_var:.4e} (bin var sum {bin_var_sum:.4e}) CPU {:.4e}",
        ga.mean,
        ca.mean,
        ca.m2 / ((ca.n as f64 - 1.0) * ca.n as f64)
    );
    assert_eq!(ga.n, n as u64);
    assert_eq!(ca.n, n as u64);
    assert!(((ga.mean - bin_mean_sum) / bin_mean_sum).abs() < 1e-9);
    // Two estimates of the same aggregate on the same seed. The histories are
    // only partly in lockstep here ((n,xn) secondaries reorder the streams), so
    // the means are held to three combined standard errors and the variance to
    // the sampling spread of a 10k-history second moment (a few percent).
    let cpu_var = ca.m2 / ((ca.n as f64 - 1.0) * ca.n as f64);
    let z = (ga.mean - ca.mean) / (agg_var + cpu_var).sqrt();
    assert!(
        z.abs() < 3.0,
        "GPU aggregate mean {} vs CPU {} ({z:+.2} sigma)",
        ga.mean,
        ca.mean
    );
    assert!(
        ((ga.m2 - ca.m2) / ca.m2).abs() < 0.10,
        "GPU aggregate m2 {} vs CPU {}",
        ga.m2,
        ca.m2
    );
    // The aggregate variance is not the sum of the bin variances: a history's
    // track length is shared out between the energy bins, so they are
    // correlated (negatively, on this sphere) and the difference is large.
    assert!(
        ((agg_var - bin_var_sum) / bin_var_sum).abs() > 0.05,
        "aggregate variance {agg_var} indistinguishable from the sum of bin variances \
         {bin_var_sum}; the per-history emit would be redundant"
    );
}

#[test]
fn photon_models_are_still_refused_in_words() {
    let _env = GpuTest::with_chunk(CHUNK);
    let Some(case) = fe56() else {
        return;
    };
    let target = ConvergenceTarget::new(ConvergenceMetric::RelativeError, 0.01);
    let (mut model, _) = build_model(&case, None, rel_error(&target));
    model.transport_secondary_photons = true;
    match yamc::gpu::run_on_gpu(&mut model, &uncapped()) {
        Err(GpuDispatchError::ConvergenceTargetsUnsupported { n_targets }) => {
            assert_eq!(n_targets, 1);
        }
        other => panic!("expected the photon refusal, got {other:?}"),
    }
}
