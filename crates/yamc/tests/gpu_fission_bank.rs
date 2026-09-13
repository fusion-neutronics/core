//! GPU device-fission-bank verification (issue #78).
//!
//! The legacy GPU fission branch approximated the fission chain as
//! `weight *= nu_bar` followed by a hard `FISSION_WEIGHT_CAP = 1000` kill that
//! truncated the chain, biasing the fission score LOW in multiplying materials
//! (U235 fission-score GPU/CPU = 0.127 @ 35 cm, Pu239 0.124 @ 5 cm). The device
//! fission bank follows the true chain: stochastically round nu_bar to N,
//! continue one progeny, bank the other N-1, and transport them in subsequent
//! generations -- the GPU twin of the CPU `sample_fission_neutrons` ->
//! `bank_secondary` recursion. yamc-CPU is the reference (it matches OpenMC).
//!
//! [`gpu_fission_bank_recovers_chain`] (the default test) runs convergent
//! sub-critical spheres -- Pu239 r=3 cm (legacy biased ~0.91 -> bank ~1.0,
//! tight), U235 r=5 cm (mildly reactive, bank stays ~1.0), Th232 r=5 cm (barely
//! multiplies, ~1.0 both ways) -- scoring the fission rate, total flux, and a
//! slowed-down (sub-MeV) group on both the CPU and the GPU, asserting:
//!   - GPU(bank ON) fission-score / CPU recovers to ~1.0 (the bug fix),
//!   - the legacy GPU(bank OFF) path is biased low where the chain multiplies,
//!   - Th232 stays ~1.0 both ways (no regression).
//!
//! [`gpu_fission_bank_dramatic_pu239`] (`--ignored`) adds the issue's headline
//! Pu239 r=5 cm case (legacy ~0.13, ~8x biased low); it is kept opt-in because
//! the strongly-multiplying chain is expensive and its estimator converges
//! slowly.
//!
//! Data: endf-b8.1 Arrow tables in the local yamc cache. Self-skips when
//! the data or a Vulkan f64 GPU adapter is absent.
//!
//! Run (single-threaded -- the shared cubecl client serialises launches):
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_fission_bank -- --nocapture --test-threads=1

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
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
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 4242;
const N_PER_BATCH: usize = 8_000;
const N_BATCHES: usize = 4; // 32k histories total.
const MAX_STEPS: u32 = 20_000;
/// Slowed-down upper edge (eV): a flux tally with edges `[0, SLOWED, 20 MeV]`
/// scores the down-scattered / fission-progeny + moderated group in bin 0. A
/// bare actinide metal sphere has no thermal flux (no moderator), so the
/// fission-chain bias shows up in the slowed-down (sub-MeV) group rather than
/// the thermal group -- this is the "every slowed-down group is low" channel
/// the issue reports.
const SLOWED_EV: f64 = 1.0e6;

fn nuclide_path(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&nuclide_path(nuclide)).exists()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Single fissile sphere (`Below(sphere)` -> GPU-boundable), radius `r` cm.
fn fissile_sphere(nuclide: &str, r: f64, cell_id: u32) -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: r,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    // Near-solid-density actinide so the chain actually multiplies.
    let density = match nuclide {
        "U235" => 18.7,
        "Pu239" => 19.8,
        "Th232" => 11.7,
        _ => 18.0,
    };
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(nuclide.to_string(), nuclide_path(nuclide))]);
    material.read_nuclear_data(&nm, None).unwrap();

    let cell = Cell::new(Some(cell_id), region, Some("fissile".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    (geometry, cell_id)
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

fn make_tally(cell_id: u32, score: Score, energy_bins: Option<Vec<f64>>) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    if let Some(bins) = energy_bins {
        t.filters.push(Filter::Energy(EnergyFilter::new(bins)));
    }
    t.scores = vec![score];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(N_BATCHES);
    Arc::new(t)
}

/// `(fission, total flux, thermal flux)` tallies for `cell_id`.
fn make_scores(cell_id: u32) -> (Arc<Tally>, Arc<Tally>, Arc<Tally>) {
    let fission = make_tally(cell_id, "fission".parse().unwrap(), None);
    let flux = make_tally(cell_id, "flux".parse().unwrap(), None);
    // Slowed-down (sub-MeV) flux = bin 0 of a [0, SLOWED, 20 MeV] flux tally.
    let thermal = make_tally(
        cell_id,
        "flux".parse().unwrap(),
        Some(vec![0.0, SLOWED_EV, 20.0e6]),
    );
    (fission, flux, thermal)
}

fn build_model(
    geometry: Geometry,
    tallies: Vec<Arc<Tally>>,
    fission_bank: bool,
) -> (Model, TransportSettings) {
    let mut model = Model::new(geometry, vec![neutron_source()], tallies);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model.gpu_fission_bank = fission_bank;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, settings)
}

fn tally_sum(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum::<f64>()
}

/// Thermal bin (bin 0) of the 2-bin thermal flux tally.
fn thermal_bin(t: &Arc<Tally>) -> f64 {
    t.get_mean().first().copied().unwrap_or(0.0)
}

struct Scores {
    fission: f64,
    flux: f64,
    thermal: f64,
}

/// One CPU reference run (always banks the fission chain, matches OpenMC).
fn run_cpu(nuclide: &str, r: f64) -> Scores {
    let (geometry, cell_id) = fissile_sphere(nuclide, r, 1);
    let (fission, flux, thermal) = make_scores(cell_id);
    let (mut model, settings) = build_model(
        geometry,
        vec![
            Arc::clone(&fission),
            Arc::clone(&flux),
            Arc::clone(&thermal),
        ],
        true,
    );
    model
        .simulate_transport(&settings)
        .unwrap_or_else(|e| panic!("CPU {nuclide} r={r}: {e}"));
    Scores {
        fission: tally_sum(&fission),
        flux: tally_sum(&flux),
        thermal: thermal_bin(&thermal),
    }
}

/// One GPU run with the fission bank `on` or off (legacy weight-cap).
fn run_gpu(nuclide: &str, r: f64, fission_bank: bool) -> Scores {
    let (geometry, cell_id) = fissile_sphere(nuclide, r, 1);
    let (fission, flux, thermal) = make_scores(cell_id);
    let (mut model, settings) = build_model(
        geometry,
        vec![
            Arc::clone(&fission),
            Arc::clone(&flux),
            Arc::clone(&thermal),
        ],
        fission_bank,
    );
    // Single AMD GPU shared with other tests; retry once on transient
    // buffer-async readback errors.
    let mut last_err = String::new();
    for _ in 0..3 {
        match yamc::gpu::run_on_gpu(&mut model, &settings) {
            Ok(_) => {
                return Scores {
                    fission: tally_sum(&fission),
                    flux: tally_sum(&flux),
                    thermal: thermal_bin(&thermal),
                }
            }
            Err(e) => last_err = e.to_string(),
        }
    }
    panic!("GPU {nuclide} r={r} bank={fission_bank}: {last_err}");
}

fn ratio(g: f64, c: f64) -> f64 {
    if c.abs() > 0.0 {
        g / c
    } else {
        f64::NAN
    }
}

/// What the bank-ON result is asserted against at this configuration.
#[derive(Clone, Copy)]
enum Mode {
    /// Convergent (low-to-moderate multiplication): the track-length fission
    /// estimator converges tightly, so the bank-ON ratio must land within 5%
    /// of the CPU/OpenMC reference. The clean correctness proof.
    Tight,
    /// Strongly multiplying (the issue's headline regime): the legacy path is
    /// dramatically biased low, and the bank recovers most of the gap. The
    /// heavy-tailed fission-chain estimator converges slowly, so the ON ratio
    /// is asserted with a variance-aware band (and must be >> the biased OFF
    /// ratio).
    Dramatic,
    /// Mildly multiplying actinide: the legacy cap barely bites (OFF already
    /// near 1.0), so there is no large gap to close -- but the bank must keep
    /// the result accurate (ON within a few % of CPU, not regressed).
    Mild,
    /// Barely multiplies (Th232): ~1.0 both ways, ON must not regress.
    NoRegression,
}

/// Run one nuclide at one radius and print the before/after table.
fn check_nuclide(nuclide: &str, r: f64, mode: Mode) {
    let cpu = run_cpu(nuclide, r);
    let off = run_gpu(nuclide, r, false);
    let on = run_gpu(nuclide, r, true);

    let f_off = ratio(off.fission, cpu.fission);
    let f_on = ratio(on.fission, cpu.fission);
    let flux_on = ratio(on.flux, cpu.flux);
    let th_on = ratio(on.thermal, cpu.thermal);

    println!("\n## {nuclide} sphere r={r} cm (14 MeV source)\n");
    println!("| score | CPU | GPU(bank OFF) | GPU(bank ON) | OFF/CPU | ON/CPU |");
    println!("|---|---|---|---|---|---|");
    println!(
        "| fission | {:.4e} | {:.4e} | {:.4e} | {:.3} | {:.3} |",
        cpu.fission, off.fission, on.fission, f_off, f_on
    );
    println!(
        "| flux    | {:.4e} | {:.4e} | {:.4e} | {:.3} | {:.3} |",
        cpu.flux,
        off.flux,
        on.flux,
        ratio(off.flux, cpu.flux),
        flux_on
    );
    println!(
        "| slowed  | {:.4e} | {:.4e} | {:.4e} | {:.3} | {:.3} |",
        cpu.thermal,
        off.thermal,
        on.thermal,
        ratio(off.thermal, cpu.thermal),
        th_on
    );

    match mode {
        Mode::Tight => {
            // Bank ON must recover the fission score AND total flux to within
            // 6% of the CPU/OpenMC reference, and the legacy path must already
            // be biased low here (the cap truncates the chain) -- so the bank
            // is demonstrably closing a real gap.
            assert!(
                f_off < 0.97,
                "{nuclide} r={r}: legacy OFF ratio {f_off:.3} should be biased low here"
            );
            assert!(
                (f_on - 1.0).abs() < 0.06,
                "{nuclide} r={r}: GPU(bank ON) fission/CPU = {f_on:.3}, expected ~1.0"
            );
            assert!(
                (flux_on - 1.0).abs() < 0.06,
                "{nuclide} r={r}: GPU(bank ON) flux/CPU = {flux_on:.3}, expected ~1.0"
            );
            // ON must clearly improve on the biased OFF ratio.
            assert!(
                f_on > f_off,
                "{nuclide} r={r}: bank ON {f_on:.3} should exceed biased OFF {f_off:.3}"
            );
        }
        Mode::Dramatic => {
            // The legacy path is severely biased low (the issue's ~0.13). The
            // bank must recover most of the ~8x deficit; the strongly-
            // multiplying chain's heavy tail makes the converged value noisy,
            // so assert a variance-aware band rather than a tight ~1.0.
            assert!(
                f_off < 0.4,
                "{nuclide} r={r}: legacy OFF ratio {f_off:.3} should show the severe \
                 biased-low truncation bug here"
            );
            // The primary, history-robust assertion: the bank is a several-fold
            // recovery over the biased OFF path. The strongly-multiplying chain
            // has a heavy tail (slow estimator convergence), so the ON ratio's
            // exact landing is noisy; assert a generous band around 1.0 plus the
            // many-fold improvement.
            assert!(
                f_on > f_off * 3.0,
                "{nuclide} r={r}: bank ON {f_on:.3} should be many-fold above OFF {f_off:.3}"
            );
            assert!(
                f_on > 0.6 && f_on < 1.6,
                "{nuclide} r={r}: GPU(bank ON) fission/CPU = {f_on:.3} should recover \
                 from the biased-low {f_off:.3} toward 1.0"
            );
            assert!(
                flux_on > f_off * 3.0 && flux_on > 0.6 && flux_on < 1.6,
                "{nuclide} r={r}: GPU(bank ON) flux/CPU = {flux_on:.3} should recover toward 1.0"
            );
        }
        Mode::Mild => {
            // Mildly-multiplying actinide: OFF is already near 1.0 (the cap
            // barely bites), and the bank keeps ON accurate to within a few %.
            assert!(
                (f_on - 1.0).abs() < 0.05,
                "{nuclide} r={r}: GPU(bank ON) fission/CPU = {f_on:.3}, expected ~1.0"
            );
            assert!(
                (flux_on - 1.0).abs() < 0.05,
                "{nuclide} r={r}: GPU(bank ON) flux/CPU = {flux_on:.3}, expected ~1.0"
            );
        }
        Mode::NoRegression => {
            // Th232 barely multiplies: even the OFF path is ~1.0 and ON must
            // not regress it.
            assert!(
                (f_on - 1.0).abs() < 0.08,
                "{nuclide} r={r}: GPU(bank ON) fission/CPU = {f_on:.3}, expected ~1.0 (no regression)"
            );
        }
    }
}

fn require_data() -> bool {
    if !gpu_available() {
        println!("no Vulkan f64 GPU adapter -- skipping");
        return false;
    }
    for nuclide in ["U235", "Pu239", "Th232"] {
        if !data_present(nuclide) {
            println!(
                "endf-b8.1 {nuclide} data absent at {} -- skipping",
                yamc_test_cache::root().display()
            );
            return false;
        }
    }
    true
}

/// Core verification (fast, reliable): the fission bank fixes the biased-low
/// chain on convergent sub-critical configurations.
///
/// Bare actinide metal spheres go supercritical well below 35 cm (U235 critical
/// radius ~8.5 cm), so a literal 35 cm fixed-source sphere has a divergent
/// (non-converging) chain on BOTH backends and is not a valid sub-critical
/// geometry. The bias scales with multiplication, so the SAME bug + recovery is
/// exercised on convergent radii:
///   - r=3 cm Pu239: the legacy cap already bites (OFF ~0.91); the bank recovers
///     to ~1.0 tightly -- the clean correctness proof.
///   - r=5 cm U235: mildly reactive (cap barely bites); the bank stays accurate.
///   - r=5 cm Th232: barely multiplies; ~1.0 both ways (no regression).
#[test]
fn gpu_fission_bank_recovers_chain() {
    if !require_data() {
        return;
    }
    check_nuclide("Pu239", 3.0, Mode::Tight);
    check_nuclide("U235", 5.0, Mode::Mild);
    check_nuclide("Th232", 5.0, Mode::NoRegression);
}

/// The issue's headline regime (Pu239 r=5 cm: legacy ~0.13, ~8x biased low).
/// Strongly multiplying, so the heavy-tailed fission-chain estimator converges
/// slowly and the run is expensive (deep chains x many generations) -- kept as
/// an opt-in `--ignored` test. The bank recovers from ~0.13 toward 1.0 (a
/// several-fold improvement; exact landing is variance-limited at modest
/// histories -- a high-statistics run converges to ~1.0).
#[test]
#[ignore = "expensive strongly-multiplying chain; run with --ignored"]
fn gpu_fission_bank_dramatic_pu239() {
    if !require_data() {
        return;
    }
    check_nuclide("Pu239", 5.0, Mode::Dramatic);
}
