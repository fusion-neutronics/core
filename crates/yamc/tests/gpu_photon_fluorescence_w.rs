//! End-to-end proof that the endf-b8.1 auto-download + atomic-relaxation
//! cascade produce X-ray fluorescence on BOTH the CPU and GPU photon paths.
//!
//! A tungsten sphere is irradiated by an 90 keV photon point source -- above
//! the W K-edge (69.5 keV) -- so K-shell photoionisation drives the
//! characteristic K-alpha fluorescence (~58-59 keV) plus the L-line complex
//! (~8-10 keV). An energy-binned photon flux tally turns those characteristic
//! lines into spectral peaks.
//!
//! The W photon data is resolved from the endf-b8.1 library, which ships the
//! full atomic-relaxation tables (subshell binding energies + fluorescence /
//! Auger transitions). This is also the library the photon default now points
//! at (see `DEFAULT_PHOTON_LIBRARY` / `Material::read_nuclear_data_or_keyword`),
//! so `read_nuclear_data_or_keyword("endf-b8.1")` cleanly exercises the
//! auto-download for both the neutron and photon data in one call.
//!
//! Assertions:
//! - a clear K-alpha fluorescence peak (55-62 keV) sits well ABOVE the smooth
//!   continuum on BOTH backends -- if the relaxation tables were missing, the
//!   photoelectric branch would simply absorb and this peak would not exist;
//! - CPU and GPU agree within the usual photon band (25%) on the integrated
//!   flux AND on the K-alpha-bin flux.
//!
//! Self-skips without an f64 GPU adapter or when the endf-b8.1 data cannot be
//! resolved (offline + cold cache), so CI stays green.

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings};
use yamc_materials::Material;
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

/// 90 keV: above the W K-edge (69.5 keV) so K-shell photoionisation -- and
/// hence K-alpha fluorescence -- is energetically allowed.
const SOURCE_E: f64 = 90.0e3;
/// W K-alpha complex (Kalpha1 ~59.3 keV, Kalpha2 ~58.0 keV) lands here.
const KALPHA_LO: f64 = 55.0e3;
const KALPHA_HI: f64 = 62.0e3;

/// Log-spaced bin edges spanning 1 keV .. 100 keV (30 bins). Fine enough to
/// separate the W K-alpha line (~58-59 keV) and the L-line complex
/// (~8-10 keV) from the smooth scatter continuum.
fn bins() -> Vec<f64> {
    let (lo, hi, n) = (1.0e3_f64, 100.0e3_f64, 30usize);
    let (ln_lo, ln_hi) = (lo.ln(), hi.ln());
    (0..=n)
        .map(|i| (ln_lo + (ln_hi - ln_lo) * (i as f64) / (n as f64)).exp())
        .collect()
}

/// Build a W sphere photon model. Returns `None` (skip) if the endf-b8.1
/// data cannot be resolved (offline with a cold cache).
fn build() -> Option<(Model, Arc<Tally>, Vec<f64>, TransportSettings)> {
    let sphere = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            // ~2 cm of W (rho 19.3 g/cm3) is many mean-free-paths at 60-90 keV,
            // so most source photons photo-absorb and re-emit fluorescence
            // before escaping -- maximising the characteristic-line signal.
            radius: 2.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));

    // Natural tungsten by the stable isotopes endf-b8.1 publishes.
    let mut material = Material::new(
        HashMap::from([
            ("W182".into(), 0.2650),
            ("W183".into(), 0.1431),
            ("W184".into(), 0.3064),
            ("W186".into(), 0.2855),
        ]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");

    // Resolve neutron + photon data from endf-b8.1. The keyword form registers
    // each element's photon source too; the photon library now defaults to
    // endf-b8.1 (which carries the atomic-relaxation tables that drive
    // fluorescence). Auto-downloads W neutron + W photon Arrow tars if not
    // already cached.
    if let Err(e) = material.read_nuclear_data_or_keyword("endf-b8.1") {
        eprintln!("skipping -- could not resolve endf-b8.1 data (offline?): {e}");
        return None;
    }
    // Surface a cold-cache / offline failure as a skip rather than a panic.
    if material
        .init_photon_data(&material.photon_data_paths.clone())
        .is_err()
    {
        eprintln!("skipping -- could not load W photon data (offline?)");
        return None;
    }

    let cell = Cell::new(Some(1), region, Some("w".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Photon(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![SOURCE_E], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });

    let edges = bins();
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.filters.push(Filter::ParticleType(ParticleTypeFilter::new(
        yamc_particle::particle::ParticleType::Photon,
    )));
    t.filters
        .push(Filter::Energy(EnergyFilter::new(edges.clone())));
    t.scores = vec![Score::Flux(FluxScore)];
    t.initialize_batches(8);
    let t = Arc::new(t);

    let mut model = Model::new(geometry, vec![source], vec![t.clone()]);
    model.max_steps_per_particle = 5_000;
    model.transport_secondary_photons = true;
    let settings = TransportSettings {
        total_particles: Some(40_000 * 8),
        seed: 7,
        ..Default::default()
    };
    Some((model, t, edges, settings))
}

/// Sum the flux in bins whose centre falls in `[lo, hi)`.
fn band_flux(spectrum: &[f64], edges: &[f64], lo: f64, hi: f64) -> f64 {
    let mut s = 0.0;
    for (i, &f) in spectrum.iter().enumerate() {
        let centre = (edges[i] * edges[i + 1]).sqrt();
        if centre >= lo && centre < hi {
            s += f;
        }
    }
    s
}

#[test]
fn gpu_photon_fluorescence_w_matches_cpu() {
    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let Some((mut cpu_m, cpu_t, edges, settings)) = build() else {
        return;
    };
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU run");
    let cpu = cpu_t.get_mean();

    let Some((mut gpu_m, gpu_t, _, settings)) = build() else {
        return;
    };
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    let n_bins = cpu.len();

    // Print the spectrum so the fluorescence lines are visible in the log.
    eprintln!("\nW fluorescence spectrum (90 keV source, endf-b8.1, CPU vs GPU):");
    eprintln!("| bin | E_lo (keV) | E_hi (keV) | CPU flux | GPU flux | ratio |");
    eprintln!("|---|---|---|---|---|---|");
    for i in 0..n_bins {
        let (c, g) = (cpu[i], gpu[i]);
        let ratio = if c > 0.0 {
            format!("{:.3}", g / c)
        } else {
            "n/a".to_string()
        };
        eprintln!(
            "| {i} | {:.2} | {:.2} | {c:.4e} | {g:.4e} | {ratio} |",
            edges[i] / 1e3,
            edges[i + 1] / 1e3,
        );
    }

    // --- (a) K-alpha fluorescence peak well above the continuum, BOTH sides.
    // Compare the K-alpha band (55-62 keV) to a "continuum" reference taken
    // from the scatter region just below it (20-45 keV), where there is no
    // characteristic line. The peak must clearly exceed that baseline.
    let cont_lo = 20.0e3;
    let cont_hi = 45.0e3;
    for (label, spec) in [("CPU", &cpu), ("GPU", &gpu)] {
        let kalpha = band_flux(spec, &edges, KALPHA_LO, KALPHA_HI);
        let continuum = band_flux(spec, &edges, cont_lo, cont_hi);
        eprintln!(
            "{label}: K-alpha band (55-62 keV) flux = {kalpha:.4e}, \
             continuum (20-45 keV) flux = {continuum:.4e}"
        );
        assert!(
            kalpha > 0.0,
            "{label}: no flux in the W K-alpha band (55-62 keV) -- atomic \
             relaxation / fluorescence is missing"
        );
        // The characteristic line dominates the continuum at this baseline:
        // a comfortable margin (>1.5x) proves it is a peak, not noise.
        assert!(
            kalpha > 1.5 * continuum,
            "{label}: K-alpha band flux {kalpha:.4e} is not clearly above the \
             continuum {continuum:.4e} -- fluorescence peak absent"
        );
    }

    // --- (b) CPU and GPU agree within the usual photon band (~25%).
    let cpu_total: f64 = cpu.iter().sum();
    let gpu_total: f64 = gpu.iter().sum();
    assert!(cpu_total > 0.0 && gpu_total > 0.0, "empty spectrum");
    let total_ratio = gpu_total / cpu_total;
    eprintln!(
        "integrated photon flux: CPU = {cpu_total:.4e}, GPU = {gpu_total:.4e}, \
         ratio = {total_ratio:.3}"
    );
    assert!(
        (0.75..=1.25).contains(&total_ratio),
        "integrated GPU/CPU photon flux ratio {total_ratio:.3} outside [0.75, 1.25]"
    );

    let cpu_kalpha = band_flux(&cpu, &edges, KALPHA_LO, KALPHA_HI);
    let gpu_kalpha = band_flux(&gpu, &edges, KALPHA_LO, KALPHA_HI);
    let kalpha_ratio = gpu_kalpha / cpu_kalpha;
    eprintln!(
        "K-alpha-band flux: CPU = {cpu_kalpha:.4e}, GPU = {gpu_kalpha:.4e}, \
         ratio = {kalpha_ratio:.3}"
    );
    assert!(
        (0.75..=1.25).contains(&kalpha_ratio),
        "K-alpha-band GPU/CPU flux ratio {kalpha_ratio:.3} outside [0.75, 1.25] -- \
         fluorescence yield differs between backends"
    );
}

/// Regression for the GPU+LED fluorescence gate (yamc-verification issue #189).
/// Atomic relaxation is independent of the electron treatment, but the GPU
/// translate step used to skip the atomic-relaxation / Doppler packs unless TTB
/// was selected, so under LED (`ElectronTreatment::Local`) the GPU emitted no
/// fluorescence at all -- the W K-alpha peak vanished on the GPU while the CPU
/// kept it. This runs the same tungsten check under LED and requires the GPU
/// K-alpha peak to be present and to match the CPU. Before the fix the GPU
/// K-alpha band is ~0 (assertion fails); after, it matches the CPU.
#[test]
fn gpu_photon_fluorescence_w_led_matches_cpu() {
    use yamc::model::ElectronTreatment;

    if yamc_gpu::GpuContext::new().is_err() {
        eprintln!("skipping -- no GPU with f64 compute available");
        return;
    }

    let Some((mut cpu_m, cpu_t, edges, settings)) = build() else {
        return;
    };
    cpu_m.electron_treatment = ElectronTreatment::Local;
    cpu_m
        .simulate_transport(&TransportSettings {
            threads: Some(1),
            ..settings
        })
        .expect("CPU run");
    let cpu = cpu_t.get_mean();

    let Some((mut gpu_m, gpu_t, _, settings)) = build() else {
        return;
    };
    gpu_m.electron_treatment = ElectronTreatment::Local;
    yamc::gpu::run_on_gpu(&mut gpu_m, &settings).expect("GPU dispatch");
    let gpu = gpu_t.get_mean();

    let cpu_kalpha = band_flux(&cpu, &edges, KALPHA_LO, KALPHA_HI);
    let gpu_kalpha = band_flux(&gpu, &edges, KALPHA_LO, KALPHA_HI);
    let continuum = band_flux(&gpu, &edges, 20.0e3, 45.0e3);
    eprintln!(
        "LED W K-alpha band flux: CPU = {cpu_kalpha:.4e}, GPU = {gpu_kalpha:.4e} \
         (GPU continuum 20-45 keV = {continuum:.4e})"
    );

    // Core regression: under LED the GPU must still emit the K-alpha line (it
    // was zero before the atomic-relaxation pack gate was removed).
    assert!(
        gpu_kalpha > 0.0,
        "GPU+LED: no flux in the W K-alpha band (55-62 keV) -- atomic-relaxation \
         fluorescence dropped on the GPU under LED"
    );
    assert!(
        gpu_kalpha > 1.5 * continuum,
        "GPU+LED: K-alpha band {gpu_kalpha:.4e} not clearly above continuum \
         {continuum:.4e} -- fluorescence peak absent"
    );
    let ratio = gpu_kalpha / cpu_kalpha;
    eprintln!("LED K-alpha GPU/CPU ratio = {ratio:.3}");
    assert!(
        (0.75..=1.25).contains(&ratio),
        "GPU/CPU K-alpha ratio {ratio:.3} outside [0.75, 1.25] under LED"
    );
}
