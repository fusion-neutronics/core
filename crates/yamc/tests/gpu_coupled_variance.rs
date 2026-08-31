//! Batch-free per-history variance for the photon / coupled paths (issue #233
//! Stage 3). The photon-source path is per-history (each source photon is a
//! history, cascade in-thread); the coupled path is per-source (a source
//! neutron's secondary photons, transported in the photon sub-pass, fold back
//! into the source neutron's variance sample via a `source_idx` threaded through
//! the bank).
//!
//! Checks on an Fe56 sphere (14 MeV neutron source, secondary photons on):
//! 1. `coupled_total_particles_invariance` -- the per-source stream is invariant
//!    to `total_particles`: the source neutrons of an N run and the first N of a
//!    2N run are bit-identical.
//! 2. `coupled_gpu_cpu_std_dev_parity` -- GPU per-history std_dev matches CPU
//!    within (physics-gap-tolerant) tolerance for the neutron and photon flux.
//!
//! Self-skips if the local Fe56 / Fe arrow files or an f64 GPU are absent. Run:
//!   cargo test -p yamc --features gpu --release --test gpu_coupled_variance
//!
//! `--test-threads=1` used to be required, because the launch-chunk override is
//! a process-global env var and a test that set it changed what its concurrent
//! siblings launched (issue #344). `GpuTest` serialises the binary instead, so
//! the plain invocation above is now the right one.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::GpuRunResult;
use yamc::model::{Model, TransportSettings, Verbose};
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
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

fn data_present() -> bool {
    std::path::Path::new("tests/Fe56.arrow").exists()
        && std::path::Path::new("tests/Fe.arrow").exists()
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

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn fe_sphere_geo() -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 12.0,
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
    let cell_id = cell.cell_id.unwrap();
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_id)
}

fn point_source(ptype: ParticleType, energy_ev: f64, strength: f64) -> ParticleSource {
    let s = Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength,
    };
    match ptype {
        ParticleType::Photon => ParticleSource::Photon(s),
        _ => ParticleSource::Neutron(s),
    }
}

fn flux_tally(cell_id: u32, ptype: ParticleType, bins: Option<Vec<f64>>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(ptype)));
    if let Some(b) = bins {
        t.filters.push(Filter::Energy(EnergyFilter::new(b)));
    }
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(10);
    Arc::new(t)
}

fn energy_bins(n: usize) -> Vec<f64> {
    let lo = 1e2f64.ln();
    let hi = 1.5e7f64.ln();
    (0..=n)
        .map(|i| (lo + (hi - lo) * i as f64 / n as f64).exp())
        .collect()
}

fn settings(total: usize) -> TransportSettings {
    TransportSettings {
        total_particles: Some(total),
        seed: 7654321,
        threads: Some(1),
        ..Default::default()
    }
}

fn build_coupled(
    neutron_t: Arc<Tally>,
    photon_t: Arc<Tally>,
    total: usize,
) -> (Model, TransportSettings) {
    let (geo, _cid) = fe_sphere_geo();
    let src = point_source(ParticleType::Neutron, 14_000_000.0, 1.0);
    let mut model = Model::new(geo, vec![src], vec![neutron_t, photon_t]);
    model.verbose = Verbose::silent();
    model.transport_secondary_photons = true;
    model.max_steps_per_particle = 10_000;
    (model, settings(total))
}

/// Photon-source model (routes to `run_on_gpu_photon`, PER-HISTORY): a 2 MeV
/// isotropic photon point source, its full in-thread cascade scored per source
/// photon.
fn build_photon(photon_t: Arc<Tally>, total: usize) -> (Model, TransportSettings) {
    let (geo, _cid) = fe_sphere_geo();
    let src = point_source(ParticleType::Photon, 2_000_000.0, 1.0);
    let mut model = Model::new(geo, vec![src], vec![photon_t]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = 10_000;
    (model, settings(total))
}

/// Mixed-source model (routes to `run_on_gpu_mixed`, PER-SOURCE unified index):
/// a 14 MeV neutron source AND a 2 MeV photon source of equal strength, with
/// secondary photons on so the neutron pass, the secondary sub-pass, and the
/// primary photon pass all fold into the same per-source accumulator.
fn build_mixed(
    neutron_t: Arc<Tally>,
    photon_t: Arc<Tally>,
    total: usize,
) -> (Model, TransportSettings) {
    let (geo, _cid) = fe_sphere_geo();
    let neutron_src = point_source(ParticleType::Neutron, 14_000_000.0, 1.0);
    let photon_src = point_source(ParticleType::Photon, 2_000_000.0, 1.0);
    let mut model = Model::new(
        geo,
        vec![neutron_src, photon_src],
        vec![neutron_t, photon_t],
    );
    model.verbose = Verbose::silent();
    model.transport_secondary_photons = true;
    model.max_steps_per_particle = 10_000;
    (model, settings(total))
}

fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<GpuRunResult, String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(r) => return Ok(r),
            Err(e) => {
                last = e.to_string();
                if !(last.contains("BufferAsync") || last.contains("buffer async")) {
                    return Err(last);
                }
                eprintln!("GPU transient (attempt {attempt}): {last}; retrying");
            }
        }
    }
    Err(last)
}

#[test]
fn coupled_total_particles_invariance() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping coupled invariance: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let n = 3000usize;
    let cid = 1;
    let (mut m1, s1) = build_coupled(
        flux_tally(cid, ParticleType::Neutron, None),
        flux_tally(cid, ParticleType::Photon, None),
        n,
    );
    let r1 = run_gpu_retry(&mut m1, &s1).expect("GPU run N");
    let (mut m2, s2) = build_coupled(
        flux_tally(cid, ParticleType::Neutron, None),
        flux_tally(cid, ParticleType::Photon, None),
        2 * n,
    );
    let r2 = run_gpu_retry(&mut m2, &s2).expect("GPU run 2N");

    // Diagnostics carry the SOURCE neutrons (ordered by global source index).
    assert_eq!(r1.final_energies.len(), n);
    assert!(r2.final_energies.len() >= 2 * n);
    for h in 0..n {
        assert_eq!(
            r1.final_energies[h].to_bits(),
            r2.final_energies[h].to_bits(),
            "coupled source history {h}: final_energy differs between N and 2N"
        );
        assert_eq!(
            r1.n_steps[h], r2.n_steps[h],
            "source history {h}: n_steps differs"
        );
    }
    assert_eq!(
        r1.n_particles, n,
        "coupled n_particles must be the source-neutron count"
    );
    eprintln!(
        "coupled_total_particles_invariance: {n} source histories bit-identical across N and 2N"
    );
}

#[test]
fn coupled_gpu_cpu_std_dev_parity() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping coupled std_dev parity: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let total = 120_000usize;
    let cid = 1;
    let bins = energy_bins(20);

    // CPU (production per-history Welford = reference).
    let cpu_n = flux_tally(cid, ParticleType::Neutron, Some(bins.clone()));
    let cpu_p = flux_tally(cid, ParticleType::Photon, Some(bins.clone()));
    let (mut cpu_m, cs) = build_coupled(Arc::clone(&cpu_n), Arc::clone(&cpu_p), total);
    cpu_m.simulate_transport(&cs).expect("CPU run");
    let (cn_mean, cn_sd) = (cpu_n.get_mean().to_vec(), cpu_n.get_std_dev().to_vec());
    let (cp_mean, cp_sd) = (cpu_p.get_mean().to_vec(), cpu_p.get_std_dev().to_vec());

    // GPU (batch-free per-source).
    let gpu_n = flux_tally(cid, ParticleType::Neutron, Some(bins.clone()));
    let gpu_p = flux_tally(cid, ParticleType::Photon, Some(bins.clone()));
    let (mut gpu_m, gs) = build_coupled(Arc::clone(&gpu_n), Arc::clone(&gpu_p), total);
    run_gpu_retry(&mut gpu_m, &gs).expect("GPU run");
    let (gn_mean, gn_sd) = (gpu_n.get_mean().to_vec(), gpu_n.get_std_dev().to_vec());
    let (gp_mean, gp_sd) = (gpu_p.get_mean().to_vec(), gpu_p.get_std_dev().to_vec());

    check_parity("neutron", &cn_mean, &cn_sd, &gn_mean, &gn_sd);
    check_parity("photon", &cp_mean, &cp_sd, &gp_mean, &gp_sd);
}

/// The source-particle diagnostics (`final_energies` / `n_steps`) of an N run
/// and the first N of a 2N run must be bit-identical: proof the per-source seed
/// stream is keyed to the GLOBAL source index with a fixed launch chunk, so it
/// is invariant to `total_particles`.
fn assert_prefix_bit_identical(r1: &GpuRunResult, r2: &GpuRunResult, n: usize, label: &str) {
    assert!(
        r1.final_energies.len() >= n,
        "{label}: N run produced {} diagnostics (< {n})",
        r1.final_energies.len()
    );
    assert!(
        r2.final_energies.len() >= n,
        "{label}: 2N run produced {} diagnostics (< {n})",
        r2.final_energies.len()
    );
    for h in 0..n {
        assert_eq!(
            r1.final_energies[h].to_bits(),
            r2.final_energies[h].to_bits(),
            "{label} history {h}: final_energy differs between N and 2N"
        );
        assert_eq!(
            r1.n_steps[h], r2.n_steps[h],
            "{label} history {h}: n_steps differs between N and 2N"
        );
    }
}

#[test]
fn photon_source_total_particles_invariance() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping photon-source invariance: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let n = 3000usize;
    let (mut m1, s1) = build_photon(flux_tally(1, ParticleType::Photon, None), n);
    let r1 = run_gpu_retry(&mut m1, &s1).expect("GPU run N");
    let (mut m2, s2) = build_photon(flux_tally(1, ParticleType::Photon, None), 2 * n);
    let r2 = run_gpu_retry(&mut m2, &s2).expect("GPU run 2N");
    assert_prefix_bit_identical(&r1, &r2, n, "photon-source");
    assert_eq!(
        r1.n_particles, n,
        "photon N must be the source-photon count"
    );
    eprintln!("photon_source_total_particles_invariance: {n} source photons bit-identical across N and 2N");
}

/// Multi-CHUNK invariance for the per-history photon path: a small launch chunk
/// makes N and 2N each span several launches, so the bit-identical prefix proves
/// the seeding is total-independent ACROSS chunk boundaries.
#[test]
fn photon_source_multi_chunk_invariance() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping photon-source multi-chunk invariance: data/GPU absent");
        return;
    }
    let chunk = 700usize;
    let n = 3 * chunk;
    let _held = GpuTest::with_chunk(chunk);
    let (mut m1, s1) = build_photon(flux_tally(1, ParticleType::Photon, None), n);
    let r1 = run_gpu_retry(&mut m1, &s1);
    let (mut m2, s2) = build_photon(flux_tally(1, ParticleType::Photon, None), 2 * n);
    let r2 = run_gpu_retry(&mut m2, &s2);
    let (r1, r2) = (r1.expect("GPU run N"), r2.expect("GPU run 2N"));
    assert_prefix_bit_identical(&r1, &r2, n, "photon-source multi-chunk");
    eprintln!("photon_source_multi_chunk_invariance: {n} photons bit-identical across N and 2N over {} chunks", n / chunk);
}

#[test]
fn photon_source_gpu_cpu_std_dev_parity() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping photon-source std_dev parity: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let total = 120_000usize;
    let bins = energy_bins(20);
    let cpu = flux_tally(1, ParticleType::Photon, Some(bins.clone()));
    let (mut cpu_m, cs) = build_photon(Arc::clone(&cpu), total);
    cpu_m.simulate_transport(&cs).expect("CPU run");
    let (cmean, csd) = (cpu.get_mean().to_vec(), cpu.get_std_dev().to_vec());
    let gpu = flux_tally(1, ParticleType::Photon, Some(bins.clone()));
    let (mut gpu_m, gs) = build_photon(Arc::clone(&gpu), total);
    run_gpu_retry(&mut gpu_m, &gs).expect("GPU run");
    let (gmean, gsd) = (gpu.get_mean().to_vec(), gpu.get_std_dev().to_vec());
    check_parity("photon-source", &cmean, &csd, &gmean, &gsd);
}

/// Mixed-source invariance: the neutron + photon passes append their diagnostics
/// per chunk, so a bit-identical prefix needs aligned chunk boundaries. Forcing
/// the launch chunk to exactly N makes the N run one chunk and the first chunk of
/// the 2N run identical to it, so `final_energies[..N]` must match bit-for-bit.
#[test]
fn mixed_source_total_particles_invariance() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping mixed-source invariance: data/GPU absent");
        return;
    }
    let n = 3000usize;
    let _held = GpuTest::with_chunk(n);
    let (mut m1, s1) = build_mixed(
        flux_tally(1, ParticleType::Neutron, None),
        flux_tally(1, ParticleType::Photon, None),
        n,
    );
    let r1 = run_gpu_retry(&mut m1, &s1);
    let (mut m2, s2) = build_mixed(
        flux_tally(1, ParticleType::Neutron, None),
        flux_tally(1, ParticleType::Photon, None),
        2 * n,
    );
    let r2 = run_gpu_retry(&mut m2, &s2);
    let (r1, r2) = (r1.expect("GPU run N"), r2.expect("GPU run 2N"));
    assert_prefix_bit_identical(&r1, &r2, n, "mixed-source");
    assert_eq!(
        r1.n_particles, n,
        "mixed N must be the source-particle count"
    );
    eprintln!("mixed_source_total_particles_invariance: {n} source particles bit-identical across N and 2N");
}

#[test]
fn mixed_source_gpu_cpu_std_dev_parity() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping mixed-source std_dev parity: data/GPU absent");
        return;
    }
    let _held = GpuTest::new();
    let total = 120_000usize;
    let bins = energy_bins(20);
    let cpu_n = flux_tally(1, ParticleType::Neutron, Some(bins.clone()));
    let cpu_p = flux_tally(1, ParticleType::Photon, Some(bins.clone()));
    let (mut cpu_m, cs) = build_mixed(Arc::clone(&cpu_n), Arc::clone(&cpu_p), total);
    cpu_m.simulate_transport(&cs).expect("CPU run");
    let (cn_mean, cn_sd) = (cpu_n.get_mean().to_vec(), cpu_n.get_std_dev().to_vec());
    let (cp_mean, cp_sd) = (cpu_p.get_mean().to_vec(), cpu_p.get_std_dev().to_vec());
    let gpu_n = flux_tally(1, ParticleType::Neutron, Some(bins.clone()));
    let gpu_p = flux_tally(1, ParticleType::Photon, Some(bins.clone()));
    let (mut gpu_m, gs) = build_mixed(Arc::clone(&gpu_n), Arc::clone(&gpu_p), total);
    run_gpu_retry(&mut gpu_m, &gs).expect("GPU run");
    check_parity(
        "mixed neutron",
        &cn_mean,
        &cn_sd,
        &gpu_n.get_mean(),
        &gpu_n.get_std_dev(),
    );
    check_parity(
        "mixed photon",
        &cp_mean,
        &cp_sd,
        &gpu_p.get_mean(),
        &gpu_p.get_std_dev(),
    );
}

/// Multi-CHUNK invariance for the per-source coupled path (the single-chunk test
/// above uses the default large chunk): a small chunk makes N and 2N span several
/// launches, so the bit-identical source-neutron prefix proves the per-source
/// seeding is total-independent across chunk boundaries.
#[test]
fn coupled_total_particles_invariance_multi_chunk() {
    if !data_present() || !gpu_available() {
        eprintln!("skipping coupled multi-chunk invariance: data/GPU absent");
        return;
    }
    let chunk = 700usize;
    let n = 3 * chunk;
    let _held = GpuTest::with_chunk(chunk);
    let (mut m1, s1) = build_coupled(
        flux_tally(1, ParticleType::Neutron, None),
        flux_tally(1, ParticleType::Photon, None),
        n,
    );
    let r1 = run_gpu_retry(&mut m1, &s1);
    let (mut m2, s2) = build_coupled(
        flux_tally(1, ParticleType::Neutron, None),
        flux_tally(1, ParticleType::Photon, None),
        2 * n,
    );
    let r2 = run_gpu_retry(&mut m2, &s2);
    let (r1, r2) = (r1.expect("GPU run N"), r2.expect("GPU run 2N"));
    assert_prefix_bit_identical(&r1, &r2, n, "coupled multi-chunk");
    eprintln!("coupled_total_particles_invariance_multi_chunk: {n} source neutrons bit-identical over {} chunks", n / chunk);
}

/// Per-bin GPU/CPU std_dev parity over well-populated bins. Bands are wide
/// enough to clear the known GPU-vs-CPU photon-physics gap (a few %) while still
/// catching a broken per-source grouping (which would move the ratio far off 1).
fn check_parity(label: &str, cmean: &[f64], csd: &[f64], gmean: &[f64], gsd: &[f64]) {
    assert_eq!(cmean.len(), gmean.len());
    let max_mean = cmean.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_mean > 0.0, "{label}: no flux scored");
    let mut ratios = Vec::new();
    let mut worst_mean = 0.0f64;
    // Restrict to genuinely well-populated bins (>= 8% of the peak): sparse bins
    // carry large MC noise on the mean that would flake the mean-deviation check
    // without telling us anything about the per-source variance grouping.
    for i in 0..cmean.len() {
        if cmean[i] < 0.08 * max_mean || csd[i] <= 0.0 || gsd[i] <= 0.0 {
            continue;
        }
        let md = (gmean[i] - cmean[i]).abs() / cmean[i];
        worst_mean = worst_mean.max(md);
        let r = gsd[i] / csd[i];
        ratios.push(r);
        eprintln!(
            "{label} bin {i:2}: mean cpu={:.3e} gpu={:.3e} ({:+.1}%) std ratio {:.3}",
            cmean[i],
            gmean[i],
            100.0 * (gmean[i] / cmean[i] - 1.0),
            r,
        );
        assert!(
            (0.4..=2.5).contains(&r),
            "{label} bin {i}: GPU/CPU std_dev ratio {r:.3} out of [0.4, 2.5] (grouping broken)"
        );
    }
    assert!(
        ratios.len() >= 4,
        "{label}: too few well-populated bins ({})",
        ratios.len()
    );
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ratios[ratios.len() / 2];
    eprintln!(
        "{label}: median std_dev ratio {med:.3} over {} bins; worst mean dev {:.1}%",
        ratios.len(),
        100.0 * worst_mean
    );
    assert!(
        (0.7..=1.4).contains(&med),
        "{label}: median GPU/CPU std_dev ratio {med:.3} outside [0.7, 1.4] (per-history variance biased)"
    );
    assert!(
        worst_mean < 0.10,
        "{label}: worst mean dev {:.1}% exceeds 10%",
        100.0 * worst_mean
    );
}
