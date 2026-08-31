//! Parity regression for the GPU multi-component evaporation sampler.
//!
//! A neutron product may carry several Evaporation sub-distributions gated by
//! equal `applicability(E_in)` (1/N each). The CPU
//! (`ReactionProduct::sample_distribution_index`) draws one component per
//! collision; the GPU mirrors this with a per-collision uniform selector over
//! the components stored in `EvapSlot` (see `EvapSlot::from_evaps` /
//! `COL_EVAP_N_COMPONENTS`). This test confirms the GPU reproduces the CPU's
//! outgoing (n,xn) spectrum. Before the multi-component sampler, the GPU
//! collapsed the mixture to its applicability-dominant curve, biasing Ba140's
//! (n,2n) continuum by up to 47% (hot/cold mis-shaping; total flux unaffected
//! since (n,2n) conserves particle count).
//!
//! O17 and Ba140 both carry (n,2n) (MT16) with FRACTIONAL applicability in
//! endf-b8.1. Ba140's two components have well-separated theta (8.2e5 vs 5.5e5
//! at 14 MeV) so the mixture is clearly distinct from either curve; O17's two
//! components are nearly identical, so it was already in band (a useful
//! null control).
//!
//! Ar38 covers the OTHER multi-Evaporation shape: MT91 carries two laws
//! step-gated by 0/1 applicability WINDOWS (u = 5.9 MeV below 11 MeV
//! incident, u = 2.2 MeV above). The collapse grid must include the 11 MeV
//! window edge; before that fix the GPU kept u = 5.9 MeV at 14.06 MeV and
//! hard-truncated the (n,n') continuum at E_in - 5.9 (V&V broomstick: zero
//! flux across 8.2--11.9 MeV, reduced chi2 28). Same pattern: Ar36, Na22,
//! Co58_m1 MT91.
//!
//! The probe uses an OPTICALLY THICK sphere (high density) so a large fraction
//! of source neutrons actually collide and feed the (n,2n) continuum: in a thin
//! sphere ~99% of the flux is the uncollided 14 MeV peak, which is not shaped by
//! evaporation at all and would swamp the comparison. For each nuclide it
//! compares on both backends:
//!   * total neutron track-length flux (cell-integrated), and
//!   * an energy-binned neutron flux spectrum, EXCLUDING the source-energy
//!     (uncollided) bin, which carries no evaporation signal.
//!
//! Each spectral bin is judged two ways: the raw GPU/CPU ratio, and a z-score
//! (`|GPU-CPU| / sqrt(se_cpu^2 + se_gpu^2)`) so a large ratio in a poorly-sampled
//! bin is correctly recognised as MC noise rather than a real bias. The verdict
//! uses bins that are both reasonably populated AND statistically resolved.
//!
//! It is ENTIRELY in Rust (`Model::simulate_transport` vs `yamc::gpu::run_on_gpu`)
//! and self-skips if the endf-b8.1 cache files or an f64 GPU are absent. The GPU
//! launch is retried on transient `BufferAsyncError` (the shared AMD adapter).
//!
//! Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_cpu_evap_applicability -- --nocapture --test-threads=1

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
use yamc_tallies::filter::energy::EnergyFilter;
use yamc_tallies::filter::particle_type::ParticleTypeFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 4242;
const RADIUS: f64 = 30.0;
const N_PER_BATCH: usize = 50_000;
const N_BATCHES: usize = 8; // 400k histories total.
const MAX_STEPS: u32 = 10_000;

/// Decision band from the task brief: <=2% on total flux AND <=4% on the worst
/// spectral bin -> the argmax approximation is good enough.
const FLUX_BAND: f64 = 0.02;
const SPECTRAL_BAND: f64 = 0.04;
/// A bin only counts toward the spectral verdict if its CPU mean is at least
/// this fraction of the largest collided-bin mean (so a near-empty, noise-only
/// bin does not dominate) AND it is statistically resolved (z below).
const POP_FRAC: f64 = 0.05;
/// And only if the CPU/GPU disagreement is statistically significant: a ratio
/// far from 1 in a poorly-sampled bin (high relative error) is MC noise, not a
/// systematic evaporation bias. Require >3 sigma combined to call it a bias.
const Z_SIGNIFICANT: f64 = 3.0;

/// The source energy and the (n,2n) continuum below it. The TOP bin
/// (`SOURCE_E` upward) is the uncollided source peak and is excluded from the
/// spectral verdict (no evaporation signal). Everything below is the collided /
/// (n,2n)-fed continuum where a per-component bias would appear.
const SOURCE_E: f64 = 14_000_000.0;
fn spectrum_bins() -> Vec<f64> {
    vec![
        1e-3, 1e3, 1e5, 3e5, 5e5, 1e6, 2e6, 3e6, 4e6, 6e6, 8e6, 1e7, 1.2e7, 1.39e7, 1.45e7,
    ]
}
/// Index of the bin that holds the uncollided 14 MeV source line; excluded from
/// the spectral verdict.
fn source_bin_index() -> usize {
    let bins = spectrum_bins();
    (0..bins.len() - 1)
        .find(|&i| bins[i] <= SOURCE_E && SOURCE_E < bins[i + 1])
        .unwrap_or(bins.len() - 2)
}

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&cache_dir(nuclide)).is_dir()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Single-nuclide sphere (r=RADIUS, `Below` so the GPU AABB pass accepts it),
/// vacuum boundary, neutron data from the endf-b8.1 cache directory.
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

/// A neutron-filtered track-length flux tally over the spectrum bins on the
/// single cell.
fn spectral_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Neutron,
    )));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(spectrum_bins())));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn total_tally(cell_id: u32) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        ParticleType::Neutron,
    )));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

fn build_model(geometry: Geometry, tallies: Vec<Arc<Tally>>) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![neutron_source()], tallies);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

/// Run the GPU with a small retry loop: the shared AMD adapter intermittently
/// surfaces `BufferAsyncError` under contention.
fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<(), String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(_) => return Ok(()),
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

struct NuclideResult {
    nuclide: String,
    cpu_flux: f64,
    gpu_flux: f64,
    cpu_spec: Vec<f64>,
    gpu_spec: Vec<f64>,
    cpu_se: Vec<f64>,
    gpu_se: Vec<f64>,
}

fn run_nuclide(nuclide: &str, density: f64) -> NuclideResult {
    // CPU.
    let (geometry, cell_id) = nuclide_sphere(nuclide, density);
    let cpu_total = total_tally(cell_id);
    let cpu_spec_t = spectral_tally(cell_id);
    let (mut cpu_model, settings) = build_model(
        geometry,
        vec![Arc::clone(&cpu_total), Arc::clone(&cpu_spec_t)],
    );
    cpu_model
        .simulate_transport(&settings)
        .unwrap_or_else(|e| panic!("CPU run for {nuclide} failed: {e}"));
    let cpu_flux = cpu_total.get_mean().iter().sum();
    let cpu_spec = cpu_spec_t.get_mean().to_vec();
    let cpu_se = cpu_spec_t.get_std_dev().to_vec();

    // GPU (rebuilt model).
    let (g2, c2) = nuclide_sphere(nuclide, density);
    let gpu_total = total_tally(c2);
    let gpu_spec_t = spectral_tally(c2);
    let (mut gpu_model, settings) =
        build_model(g2, vec![Arc::clone(&gpu_total), Arc::clone(&gpu_spec_t)]);
    run_gpu_retry(&mut gpu_model, &settings)
        .unwrap_or_else(|e| panic!("GPU run for {nuclide} failed: {e}"));
    let gpu_flux = gpu_total.get_mean().iter().sum();
    let gpu_spec = gpu_spec_t.get_mean().to_vec();
    let gpu_se = gpu_spec_t.get_std_dev().to_vec();

    NuclideResult {
        nuclide: nuclide.to_string(),
        cpu_flux,
        gpu_flux,
        cpu_spec,
        gpu_spec,
        cpu_se,
        gpu_se,
    }
}

/// Print the comparison and return `(flux_dev, worst_significant_spectral_dev)`.
/// The spectral worst-case considers only collided continuum bins (excludes the
/// uncollided source bin) that are well populated AND show a statistically
/// significant (>3 sigma) CPU/GPU disagreement.
fn report(r: &NuclideResult) -> (f64, f64) {
    println!("\n## {} sphere (14 MeV neutron source)\n", r.nuclide);
    let flux_dev = if r.cpu_flux.abs() > 0.0 {
        (r.gpu_flux / r.cpu_flux - 1.0).abs()
    } else {
        0.0
    };
    println!("Total neutron flux:");
    println!(
        "  CPU = {:.6e}   GPU = {:.6e}   GPU/CPU = {:.4}   |dev| = {:.2}%",
        r.cpu_flux,
        r.gpu_flux,
        r.gpu_flux / r.cpu_flux,
        flux_dev * 100.0
    );

    let bins = spectrum_bins();
    let src_bin = source_bin_index();
    // Reference population: largest CPU mean among the COLLIDED bins (exclude
    // the uncollided source peak) so the population threshold is relative to
    // the real (n,2n) continuum, not the dominant source line.
    let max_collided = r
        .cpu_spec
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != src_bin)
        .map(|(_, &v)| v)
        .fold(0.0_f64, f64::max);

    println!(
        "\nNeutron flux spectrum (track-length, per source particle); \
         source/uncollided bin marked [src] and excluded from verdict:"
    );
    println!("| bin (eV) | CPU+-se | GPU+-se | GPU/CPU | z | counts |");
    println!("|---|---|---|---|---|---|");
    let mut worst_dev = 0.0_f64;
    for i in 0..r.cpu_spec.len() {
        let lo = bins[i];
        let hi = bins[i + 1];
        let c = r.cpu_spec[i];
        let g = r.gpu_spec[i];
        let cse = r.cpu_se[i];
        let gse = r.gpu_se[i];
        let ratio = if c != 0.0 { g / c } else { f64::NAN };
        let comb_se = (cse * cse + gse * gse).sqrt();
        let z = if comb_se > 0.0 {
            (g - c).abs() / comb_se
        } else {
            f64::NAN
        };
        let is_src = i == src_bin;
        let well_pop = c.abs() > 0.0 && max_collided > 0.0 && c >= POP_FRAC * max_collided;
        // A bin counts toward the verdict when it is a collided continuum bin,
        // well populated, AND the disagreement is statistically significant.
        let counts = !is_src && well_pop && z.is_finite() && z > Z_SIGNIFICANT;
        if counts {
            worst_dev = worst_dev.max((ratio - 1.0).abs());
        }
        let tag = if is_src {
            "[src]"
        } else if !well_pop {
            "(low pop)"
        } else if counts {
            "**counts**"
        } else {
            "noise"
        };
        println!(
            "| {lo:.2e}-{hi:.2e} | {c:.3e}+-{cse:.1e} | {g:.3e}+-{gse:.1e} | {ratio:.4} | {z:.1} | {tag} |",
        );
    }
    println!(
        "\nWorst spectral |dev| over well-populated, statistically-significant \
         collided bins: {:.2}%",
        worst_dev * 100.0
    );
    (flux_dev, worst_dev)
}

#[test]
fn gpu_cpu_evap_applicability() {
    // High densities -> optically thick sphere -> most neutrons collide and
    // feed the (n,2n) continuum, so the evaporation spectrum is well sampled.
    // The comparison is CPU-vs-GPU on identical geometry, so absolute density
    // is immaterial to the verdict.
    let cases: &[(&str, f64)] = &[("O17", 3.0), ("Ba140", 6.0), ("Ar38", 5.0)];

    let mut missing = Vec::new();
    for (n, _) in cases {
        if !data_present(n) {
            missing.push(*n);
        }
    }
    if !missing.is_empty() {
        eprintln!(
            "skipping gpu_cpu_evap_applicability -- missing endf-b8.1 cache dirs: {missing:?}"
        );
        return;
    }
    if !gpu_available() {
        eprintln!("skipping gpu_cpu_evap_applicability -- no GPU with f64 compute available");
        return;
    }

    println!("\n# CPU vs GPU: multi-component evaporation (n,2n) applicability");
    println!(
        "\nSingle-nuclide spheres r={RADIUS} cm (CSG, vacuum, optically thick); \
         14 MeV neutron point source; seed {SEED}; {} histories ({N_PER_BATCH} x {N_BATCHES} batches).",
        N_PER_BATCH * N_BATCHES
    );

    let mut overall_flux = 0.0_f64;
    let mut overall_spec = 0.0_f64;
    for (n, rho) in cases {
        let r = run_nuclide(n, *rho);
        let (fd, sd) = report(&r);
        overall_flux = overall_flux.max(fd);
        overall_spec = overall_spec.max(sd);
    }

    println!("\n# Result");
    println!(
        "Worst flux |dev| = {:.2}% (band {:.0}%); worst significant collided-spectral |dev| = \
         {:.2}% (band {:.0}%).",
        overall_flux * 100.0,
        FLUX_BAND * 100.0,
        overall_spec * 100.0,
        SPECTRAL_BAND * 100.0,
    );
    if overall_flux <= FLUX_BAND && overall_spec <= SPECTRAL_BAND {
        println!(
            "IN BAND: the GPU multi-component evaporation sampler reproduces the CPU's \
             1/N-applicability (n,xn) mixture spectrum. (Before the fix, the GPU's \
             single-dominant-curve approximation biased Ba140's (n,2n) continuum by up to 47%.)"
        );
    } else {
        println!(
            "OUT OF BAND: the GPU (n,xn) outgoing spectrum diverges from the CPU -- the \
             multi-component evaporation sampler has regressed."
        );
    }

    // Parity regression for the multi-component evaporation sampler: the GPU
    // must reproduce the CPU's 1/N-applicability mixture spectrum. Bands match
    // the task's decision rule (<=2% flux, <=4% spectral).
    assert!(
        overall_flux <= FLUX_BAND,
        "GPU vs CPU total-flux deviation {:.2}% exceeds {:.0}% band",
        overall_flux * 100.0,
        FLUX_BAND * 100.0
    );
    assert!(
        overall_spec <= SPECTRAL_BAND,
        "GPU vs CPU significant collided-spectral deviation {:.2}% exceeds {:.0}% band",
        overall_spec * 100.0,
        SPECTRAL_BAND * 100.0
    );
}
