//! Batch-free GPU per-history variance (issue #233 Stage 1).
//!
//! Two on-hardware checks for the NON-fissile neutron path:
//!
//! 1. `total_particles_invariance` -- the per-particle stream (and therefore
//!    every tally) is invariant to `total_particles`. With a FIXED launch chunk
//!    the per-particle PCG seed keys off the global history index, so history
//!    `h` is identical whether the run requests `N` or `2N` histories. Verified
//!    bit-exactly on the returned per-history diagnostics
//!    (`final_energies` / `n_steps` / `alive`): the first `N` of a `2N` run
//!    equal the `N` run.
//!
//! 2. `gpu_cpu_per_history_std_dev_parity` -- the GPU per-history `std_dev`
//!    matches the CPU per-history `std_dev` within Monte-Carlo tolerance, on a
//!    fine (64-bin) energy spectrum. A 14 MeV neutron slowing in a dense C12
//!    sphere sweeps far more than `PERHIST_K = 32` distinct `(cell, energy)`
//!    bins per history, so this exercises the exact-variance SPILL path (a
//!    broken spill would drop bins from `sum_sq` and bias `std_dev` low). The
//!    CPU (unbounded per-history hashmap) is the exact reference.
//!
//! Entirely in Rust (`Model::simulate_transport` vs `yamc::gpu::run_on_gpu`);
//! self-skips if the endf-b8.1 C12 cache or an f64 GPU is absent. Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_batch_free_variance -- --nocapture
//!
//! `--test-threads=1` used to be required: the launch-chunk override is a
//! process-global env var, so a test that set it changed what its concurrent
//! siblings launched (issue #344). `GpuTest` serialises the binary instead.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

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
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
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

/// Serialises every test in this binary, and scopes the launch-chunk override
/// to the one test that sets it.
///
/// `YAMC_GPU_LAUNCH_CHUNK` is a process-global env var read inside
/// `launch_chunk_size`, so a test that sets it changes what every test running
/// CONCURRENTLY in the same binary launches. The module header used to tell the
/// reader to pass `--test-threads=1`; a plain `cargo test` does not, so the
/// documented local GPU gate failed three tests that all passed serially
/// (issue #344).
///
/// Two things make this correct where a lock around the set/remove pair alone
/// would not:
///
/// * every test takes the guard, including the ones that do not set the
///   override, because they still READ it and must be kept out of the window;
/// * the previous value is restored in `Drop`, so a panic between setting it
///   and clearing it cannot leak the override into every later test.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Holds the binary lock and the launch-chunk override together, restoring the
/// override when it drops. Construct it FIRST in a test, and keep it alive for
/// the whole body.
struct GpuTest {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<String>,
}

impl GpuTest {
    /// Serialise against the other tests without touching the override.
    fn new() -> Self {
        Self {
            _guard: exclusive(),
            previous: std::env::var("YAMC_GPU_LAUNCH_CHUNK").ok(),
        }
    }

    /// Serialise, and force the launch chunk for the duration.
    fn with_chunk(chunk: usize) -> Self {
        let held = Self::new();
        std::env::set_var("YAMC_GPU_LAUNCH_CHUNK", chunk.to_string());
        held
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

/// Single-nuclide sphere (r=RADIUS, `Below` so the GPU AABB pass accepts it),
/// vacuum boundary. Non-fissile nuclides only (this path is Stage 1).
fn nuclide_sphere(nuclide: &str, density: f64) -> (Geometry, u32) {
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
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, 1)
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

/// 64 log-spaced energy bins from 1 meV to 15 MeV -- fine enough that a
/// moderated history sweeps well past `PERHIST_K = 32` distinct bins.
fn fine_spectrum_bins() -> Vec<f64> {
    let n = 64usize;
    let lo = 1e-3f64.ln();
    let hi = 1.5e7f64.ln();
    (0..=n)
        .map(|i| (lo + (hi - lo) * i as f64 / n as f64).exp())
        .collect()
}

fn spectral_tally(cell_id: u32, n_batches: usize) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Neutron,
    )));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(fine_spectrum_bins())));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(n_batches);
    Arc::new(t)
}

/// A cell-only neutron heating (KERMA, MT 301) tally. Its linear fixed-point
/// scale is 1.0, so it exercises the dedicated KERMA `sum_sq` scale on the
/// per-history path (the generic `s^2/2^40` form would underflow its
/// eV-magnitude squares and collapse the variance to ~0).
fn heating_tally(cell_id: u32, n_batches: usize) -> Arc<Tally> {
    use yamc_tallies::score::{HeatingScore, Score};
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.scores = vec![Score::Heating(HeatingScore)];
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

/// Run the GPU with a small retry loop (the shared AMD adapter intermittently
/// surfaces a transient `BufferAsyncError`), returning the per-history
/// diagnostics.
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
fn total_particles_invariance() {
    let nuclide = "C12";
    if !data_present(nuclide) || !gpu_available() {
        eprintln!("skipping total_particles_invariance: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let n: usize = 3000;

    // Run N histories.
    let (g1, c1) = nuclide_sphere(nuclide, 2.0);
    let (mut m1, s1) = build_model(g1, vec![spectral_tally(c1, 1)], n);
    let r1 = run_gpu_retry(&mut m1, &s1).expect("GPU run N");

    // Run 2N histories (same fixed chunk, same seed).
    let (g2, c2) = nuclide_sphere(nuclide, 2.0);
    let (mut m2, s2) = build_model(g2, vec![spectral_tally(c2, 1)], 2 * n);
    let r2 = run_gpu_retry(&mut m2, &s2).expect("GPU run 2N");

    assert_eq!(r1.final_energies.len(), n);
    assert_eq!(r2.final_energies.len(), 2 * n);

    // The first N histories of the 2N run must be BIT-IDENTICAL to the N run:
    // the per-particle seed keys off the global history index, which a fixed
    // chunk makes independent of total_particles.
    for h in 0..n {
        assert_eq!(
            r1.final_energies[h].to_bits(),
            r2.final_energies[h].to_bits(),
            "history {h}: final_energy differs between N and 2N (total leaked into the RNG key)"
        );
        assert_eq!(r1.n_steps[h], r2.n_steps[h], "history {h}: n_steps differs");
        assert_eq!(r1.alive[h], r2.alive[h], "history {h}: alive differs");
    }
    eprintln!("total_particles_invariance: {n} histories bit-identical across N and 2N");
}

/// Multi-CHUNK invariance (issue #233 Stage 2; addresses a Stage 1 review gap
/// where the invariance test only used a single launch chunk). `YAMC_GPU_LAUNCH_CHUNK`
/// forces a small chunk so N and 2N each span several launches; the shared
/// prefix (the first N source histories) must still be bit-identical, proving
/// the global-history-index seeding is total-independent ACROSS chunk
/// boundaries, not just within one chunk.
#[test]
fn total_particles_invariance_multi_chunk() {
    let nuclide = "C12";
    if !data_present(nuclide) || !gpu_available() {
        eprintln!("skipping total_particles_invariance_multi_chunk: data/GPU absent");
        return;
    }
    // Force an 800-history launch chunk: N = 2400 spans 3 chunks, 2N = 4800 spans 6.
    let chunk = 800usize;
    let n = 3 * chunk;
    let _held = GpuTest::with_chunk(chunk);

    let (g1, c1) = nuclide_sphere(nuclide, 2.0);
    let (mut m1, s1) = build_model(g1, vec![spectral_tally(c1, 1)], n);
    let r1 = run_gpu_retry(&mut m1, &s1);

    let (g2, c2) = nuclide_sphere(nuclide, 2.0);
    let (mut m2, s2) = build_model(g2, vec![spectral_tally(c2, 1)], 2 * n);
    let r2 = run_gpu_retry(&mut m2, &s2);

    let r1 = r1.expect("GPU run N");
    let r2 = r2.expect("GPU run 2N");

    assert_eq!(r1.final_energies.len(), n);
    assert_eq!(r2.final_energies.len(), 2 * n);
    for h in 0..n {
        assert_eq!(
            r1.final_energies[h].to_bits(),
            r2.final_energies[h].to_bits(),
            "multi-chunk history {h}: final_energy differs across N and 2N \
             (chunk-boundary seeding not total-independent)"
        );
        assert_eq!(
            r1.n_steps[h], r2.n_steps[h],
            "multi-chunk history {h}: n_steps differs"
        );
    }
    eprintln!(
        "total_particles_invariance_multi_chunk: {n} histories (span {} chunks) bit-identical",
        n / chunk
    );
}

#[test]
fn gpu_cpu_per_history_std_dev_parity() {
    let nuclide = "C12";
    if !data_present(nuclide) || !gpu_available() {
        eprintln!("skipping gpu_cpu_per_history_std_dev_parity: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    // Dense C12 => many collisions => a history sweeps >32 distinct energy bins
    // (exercises the exact-variance spill). Enough histories that the std_dev
    // ESTIMATE itself is well-resolved on both backends.
    let density = 3.0;
    let total: usize = 200_000;

    // CPU (production per-history Welford = exact reference). A fine flux
    // spectrum (exercises the spill) plus a cell-only heating tally (exercises
    // the dedicated KERMA sum_sq scale).
    let (gc, cc) = nuclide_sphere(nuclide, density);
    let cpu_t = spectral_tally(cc, 10);
    let cpu_h = heating_tally(cc, 10);
    let (mut cpu_model, csettings) =
        build_model(gc, vec![Arc::clone(&cpu_t), Arc::clone(&cpu_h)], total);
    cpu_model
        .simulate_transport(&csettings)
        .expect("CPU run failed");
    let cpu_mean = cpu_t.get_mean().to_vec();
    let cpu_sd = cpu_t.get_std_dev().to_vec();
    let cpu_heat_mean: f64 = cpu_h.get_mean().iter().sum();
    let cpu_heat_sd: f64 = cpu_h.get_std_dev().iter().sum();

    // GPU (batch-free per-history sum + sum_sq).
    let (gg, cg) = nuclide_sphere(nuclide, density);
    let gpu_t = spectral_tally(cg, 10);
    let gpu_h = heating_tally(cg, 10);
    let (mut gpu_model, gsettings) =
        build_model(gg, vec![Arc::clone(&gpu_t), Arc::clone(&gpu_h)], total);
    run_gpu_retry(&mut gpu_model, &gsettings).expect("GPU run failed");
    let gpu_mean = gpu_t.get_mean().to_vec();
    let gpu_sd = gpu_t.get_std_dev().to_vec();
    let gpu_heat_mean: f64 = gpu_h.get_mean().iter().sum();
    let gpu_heat_sd: f64 = gpu_h.get_std_dev().iter().sum();

    assert_eq!(cpu_mean.len(), gpu_mean.len());
    let max_mean = cpu_mean.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_mean > 0.0, "no flux scored");

    // Compare only well-populated bins (>=5% of the peak bin): sparse bins have
    // an ill-resolved std_dev estimate on either backend.
    let mut sd_ratios = Vec::new();
    let mut worst_mean_dev = 0.0f64;
    for i in 0..cpu_mean.len() {
        if cpu_mean[i] < 0.05 * max_mean || cpu_sd[i] <= 0.0 || gpu_sd[i] <= 0.0 {
            continue;
        }
        let mean_dev = (gpu_mean[i] - cpu_mean[i]).abs() / cpu_mean[i];
        worst_mean_dev = worst_mean_dev.max(mean_dev);
        let sd_ratio = gpu_sd[i] / cpu_sd[i];
        sd_ratios.push(sd_ratio);
        eprintln!(
            "bin {i:2}: mean cpu={:.4e} gpu={:.4e} ({:+.1}%)  std cpu={:.3e} gpu={:.3e} (ratio {:.3})",
            cpu_mean[i],
            gpu_mean[i],
            100.0 * (gpu_mean[i] / cpu_mean[i] - 1.0),
            cpu_sd[i],
            gpu_sd[i],
            sd_ratio,
        );
        // Per-bin std_dev must be sane: neither collapsed (broken spill would
        // drop sum_sq contributions -> ratio << 1) nor doubled.
        assert!(
            (0.5..=2.0).contains(&sd_ratio),
            "bin {i}: GPU/CPU std_dev ratio {sd_ratio:.3} out of [0.5, 2.0] \
             (exact-variance spill likely broken)"
        );
    }
    assert!(
        sd_ratios.len() >= 8,
        "too few well-populated bins ({}) to judge parity",
        sd_ratios.len()
    );

    // The MEDIAN std_dev ratio must sit near 1 -- a systematic bias (e.g. spill
    // dropping bins) would pull the whole distribution off 1.
    sd_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sd_ratios[sd_ratios.len() / 2];
    eprintln!(
        "median GPU/CPU std_dev ratio = {median:.3} over {} bins; worst mean dev {:.1}%",
        sd_ratios.len(),
        100.0 * worst_mean_dev
    );
    assert!(
        (0.85..=1.18).contains(&median),
        "median GPU/CPU std_dev ratio {median:.3} not near 1 (per-history variance biased)"
    );
    // Means must also agree closely (both estimate the same true flux).
    assert!(
        worst_mean_dev < 0.06,
        "worst well-populated-bin mean deviation {:.1}% exceeds 6%",
        100.0 * worst_mean_dev
    );

    // KERMA/heating tally (linear scale 1.0): the dedicated KERMA sum_sq scale
    // must resolve the per-history eV-magnitude squares. A broken (generic
    // 2^-40) scale underflows them and collapses the heating std_dev to ~0,
    // giving a heat_sd ratio far below 1. Both mean and std_dev must track CPU.
    let heat_mean_ratio = gpu_heat_mean / cpu_heat_mean;
    let heat_sd_ratio = gpu_heat_sd / cpu_heat_sd;
    eprintln!(
        "heating: CPU mean={cpu_heat_mean:.4e} sd={cpu_heat_sd:.4e}  \
         GPU mean={gpu_heat_mean:.4e} sd={gpu_heat_sd:.4e}  \
         (mean ratio {heat_mean_ratio:.3}, sd ratio {heat_sd_ratio:.3})"
    );
    assert!(
        cpu_heat_mean > 0.0 && cpu_heat_sd > 0.0,
        "no CPU heating scored"
    );
    assert!(
        (0.95..=1.05).contains(&heat_mean_ratio),
        "heating mean GPU/CPU ratio {heat_mean_ratio:.3} outside [0.95, 1.05]"
    );
    assert!(
        (0.80..=1.25).contains(&heat_sd_ratio),
        "heating std_dev GPU/CPU ratio {heat_sd_ratio:.3} outside [0.80, 1.25] \
         (KERMA sum_sq scale likely underflowing the per-history variance)"
    );
}
