//! Issue #111 phase 2: the GPU must never lose an (n,xn) secondary.
//!
//! The neutron kernel transports a history's extra (n,xn) neutrons in the same
//! thread, on a stack of `PEND_SLOTS` thread-private slots. Before phase 2 the
//! slots were append-only, so the cap counted every secondary a history ever
//! queued rather than the ones outstanding, and past it the kernel reverted to
//! weight multiplication -- different physics from the CPU, which banks and
//! transports real neutrons. Phase 2 reclaims the slot on pop and, when the
//! stack is genuinely full, hands the secondary to the DEVICE PARTICLE BANK for
//! the host to drain in a later pass.
//!
//! The later pass is what these tests are about. A spilled secondary finishes
//! in a different launch from the rest of its history, so its contributions
//! have to fold back into the ORIGINATING history's variance sample or the mean
//! stays right while `std_dev` goes wrong. The non-fissile dispatch normally
//! runs `PerHistory` (one sample per thread, flushed when that thread's history
//! ends), which cannot express that; on detecting a spill it redoes the launch
//! under `PerSource` (one sample per source neutron, keyed by `bank_source_idx`
//! and therefore alive across launches) and drains into it.
//!
//! Finding a fixture that spills at all took some doing: with the slots
//! reclaimed, the depth needed is the emission tree's DFS depth, and a chain of
//! endothermic (n,xn) reactions runs out of energy against its own threshold
//! after a few levels (`matched_stream_diff::nxn_spill_depth_is_sufficient`
//! bounds the rate below one history in a thousand at 14 MeV and prints the
//! depth histogram per fixture). A THICK beryllium sphere at 19.9 MeV does it:
//! (n,2n) opens at 1.85 MeV and the extra ~6 MeV buys one more level of
//! multiplication than a 14 MeV source can.
//!
//! The coupled (neutron -> photon) and mixed-source passes used to refuse a
//! launch that spilled at all, because their bank drain was photon-only and a
//! banked neutron would have been filtered out and lost. They drain banked
//! neutrons now too (fusion-neutronics/core#20); the last test here runs the
//! same spilling fixture with secondary photons on and holds both backends to
//! the same agreement.
//!
//! Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test gpu_nxn_spill -- --nocapture --test-threads=1

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
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;

const SEED: u64 = 20260726;
/// 8 mean free paths of beryllium at 14 MeV, so a history multiplies deep into
/// the (n,2n) chain before it leaks.
const RADIUS: f64 = 40.0;
/// 2 mean free paths: deep enough to multiply, shallow enough that no history
/// ever exceeds the in-thread stack, so it exercises the ordinary path.
const THIN_RADIUS: f64 = 10.0;
const BE_DENSITY: f64 = 1.85;
/// Just below the top of the ENDF/B-8.1 evaluation. At 14 MeV fewer than one
/// history in a thousand needs more than the four in-thread slots; the extra
/// ~6 MeV here buys one more level of multiplication, which is what puts
/// histories over the edge.
const SOURCE_E: f64 = 19.9e6;
const MAX_STEPS: u32 = 10_000;
/// Enough histories that the ~1-in-4000 spill rate fires many times over, and
/// enough that the per-history std_dev comparison has something to say.
const N_HISTORIES: usize = 200_000;

fn data_path(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

fn data_present(nuclide: &str) -> bool {
    std::path::Path::new(&data_path(nuclide)).exists()
}

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

fn be9_sphere(radius: f64) -> (Geometry, u32) {
    be9_sphere_with_photons(radius, false)
}

/// The same sphere, optionally with beryllium photon data loaded so the
/// material can run coupled neutron -> photon transport.
fn be9_sphere_with_photons(radius: f64, coupled: bool) -> (Geometry, u32) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

    let mut material = Material::new(
        HashMap::from([("Be9".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(BE_DENSITY),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([("Be9".to_string(), data_path("Be9"))]);
    if coupled {
        let photon_paths = HashMap::from([("Be".to_string(), data_path("Be"))]);
        material
            .read_nuclear_data(&nm, Some(&photon_paths))
            .unwrap();
        material.init_photon_data(&photon_paths).unwrap();
    } else {
        material.read_nuclear_data(&nm, None).unwrap();
    }

    let cell = Cell::new(Some(1), region, Some("be".into()), Some(0));
    (
        Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap(),
        1,
    )
}

/// Cell-total neutron flux. Deliberately NOT energy-binned: the quantity under
/// test is how much neutron is in the problem (a lost secondary shows up
/// directly in the total), and a single bin keeps the per-history variance
/// comparison to one number per estimator.
fn flux_tally(cell_id: u32) -> Arc<Tally> {
    particle_flux_tally(cell_id, ParticleType::Neutron)
}

fn particle_flux_tally(cell_id: u32, particle: ParticleType) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(cell_id)));
    t.filters
        .push(Filter::ParticleType(ParticleTypeFilter::new(particle)));
    t.scores = vec!["flux".parse().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn build_model(total_particles: usize, energy_ev: f64, radius: f64) -> (Model, TransportSettings) {
    let (geometry, cell_id) = be9_sphere(radius);
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![flux_tally(cell_id)]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(total_particles),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, settings)
}

/// The spilling fixture with secondary photons switched on, so the dispatch
/// takes the coupled path: a neutron flux tally and a photon flux tally, both
/// particle-filtered so each counts one species on both backends.
fn build_coupled_model(
    total_particles: usize,
    energy_ev: f64,
    radius: f64,
) -> (Model, TransportSettings) {
    let (geometry, cell_id) = be9_sphere_with_photons(radius, true);
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let tallies = vec![
        particle_flux_tally(cell_id, ParticleType::Neutron),
        particle_flux_tally(cell_id, ParticleType::Photon),
    ];
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    model.transport_secondary_photons = true;
    model.photon_cutoff_energy = 1000.0;
    let settings = TransportSettings {
        total_particles: Some(total_particles),
        seed: SEED,
        threads: Some(0),
        ..Default::default()
    };
    (model, settings)
}

/// Run the same model on both backends and return `((cpu_mean, cpu_std_dev),
/// (gpu_mean, gpu_std_dev), spilled)` for the flux bin. `spilled` is how many
/// (n,xn) secondaries the GPU handed to the device bank, so a test can state
/// whether it exercised the spill path instead of assuming it did.
fn both_backends(n: usize, energy_ev: f64, radius: f64) -> ((f64, f64), (f64, f64), u64) {
    let (mut cpu_model, cpu_settings) = build_model(n, energy_ev, radius);
    cpu_model
        .simulate_transport(&cpu_settings)
        .expect("CPU reference run");
    let (mut gpu_model, gpu_settings) = build_model(n, energy_ev, radius);
    let spilled = run_gpu_retry(&mut gpu_model, &gpu_settings).expect("GPU run");
    (flux_stats(&cpu_model), flux_stats(&gpu_model), spilled)
}

/// `(mean, std_dev)` of the single flux bin.
fn flux_stats(model: &Model) -> (f64, f64) {
    tally_stats(model, 0)
}

/// `(mean, std_dev)` of the single bin of tally `index`.
fn tally_stats(model: &Model, index: usize) -> (f64, f64) {
    let t = &model.tallies[index];
    let mean = t.get_mean();
    let sd = t.get_std_dev();
    assert_eq!(mean.len(), 1, "expected a single-bin flux tally");
    (mean[0], sd[0])
}

/// The GPU intermittently surfaces a transient `BufferAsyncError` on this
/// shared adapter; retry a few times before calling it a failure.
fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) -> Result<u64, String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(r) => return Ok(r.n_spilled_secondaries),
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

/// The headline: on a model that actually spills, the GPU's flux MEAN and
/// per-history STD_DEV both agree with the CPU's.
///
/// The mean says no secondary was dropped (a dropped one is missing track
/// length, biased low). The std_dev is the part the spill machinery could get
/// wrong on its own: the spilling launch is redone under `PerSource`, so if the
/// per-source fold were broken the std_dev of the WHOLE launch -- all 200000
/// histories, not just the handful that spilled -- would move, which is what
/// makes this a discriminating check despite the spill being rare.
///
/// Tolerances are Monte-Carlo tolerances, not exactness: the two backends do
/// not run the same histories in the same order (the GPU banks and re-launches,
/// the CPU keeps everything in one in-history stack), so this is a statistical
/// comparison of two correct estimators, and the CPU is the reference.
#[test]
fn spilled_secondaries_keep_mean_and_std_dev() {
    if !data_present("Be9") || !gpu_available() {
        eprintln!("skipping gpu_nxn_spill: Be9 data or f64 GPU absent");
        return;
    }
    let ((cpu_mean, cpu_sd), (gpu_mean, gpu_sd), spilled) =
        both_backends(N_HISTORIES, SOURCE_E, RADIUS);
    eprintln!(
        "Be9 r={RADIUS} @ {:.1} MeV, {N_HISTORIES} histories, {spilled} spilled:\n  \
         mean    cpu {cpu_mean:.6e}  gpu {gpu_mean:.6e}  ratio {:.5}\n  \
         std_dev cpu {cpu_sd:.6e}  gpu {gpu_sd:.6e}  ratio {:.5}",
        SOURCE_E / 1e6,
        gpu_mean / cpu_mean,
        gpu_sd / cpu_sd,
    );

    // Without this the two comparisons below say nothing about the spill: they
    // would just be another CPU-vs-GPU parity check on a model that never left
    // the in-thread stack.
    assert!(
        spilled > 0,
        "no (n,xn) secondary spilled to the device bank, so this run never \
         exercised the drain. The fixture no longer reaches the stack depth \
         it was chosen for."
    );

    // A silently dropped secondary removes its whole sub-history's track
    // length, so the mean is the direct test that nothing was lost. 1% is far
    // wider than the statistical spread at this history count and far narrower
    // than the effect of losing secondaries.
    let mean_ratio = gpu_mean / cpu_mean;
    assert!(
        (mean_ratio - 1.0).abs() < 0.01,
        "GPU flux mean {gpu_mean:.6e} vs CPU {cpu_mean:.6e} (ratio {mean_ratio:.5}): \
         a spilled (n,xn) secondary was dropped or double-counted"
    );

    // The variance claim. A per-source fold that mis-attributed a spilled
    // secondary's contributions -- to the wrong source, or as a sample of its
    // own -- would show here even though the mean stayed right.
    let sd_ratio = gpu_sd / cpu_sd;
    assert!(
        (sd_ratio - 1.0).abs() < 0.05,
        "GPU flux std_dev {gpu_sd:.6e} vs CPU {cpu_sd:.6e} (ratio {sd_ratio:.5}): \
         the spilled secondaries' contributions are not folding into their \
         originating history's variance sample"
    );
}

/// Control: a thinner Be9 sphere at 14 MeV, which never spills, so the ordinary
/// (non-escalated) `PerHistory` path runs end to end. It pins that the
/// agreement above is not an artefact of loose tolerances, and that the
/// common-case path is unchanged.
#[test]
fn non_spilling_model_is_unaffected() {
    if !data_present("Be9") || !gpu_available() {
        eprintln!("skipping gpu_nxn_spill control: Be9 data or f64 GPU absent");
        return;
    }
    let n = 100_000;
    let ((cpu_mean, cpu_sd), (gpu_mean, gpu_sd), spilled) = both_backends(n, 14.06e6, THIN_RADIUS);
    eprintln!(
        "control Be9 r={THIN_RADIUS} @ 14.06 MeV, {n} histories, {spilled} spilled: \
         mean ratio {:.5}, std_dev ratio {:.5}",
        gpu_mean / cpu_mean,
        gpu_sd / cpu_sd,
    );
    assert_eq!(
        spilled, 0,
        "the control fixture spilled, so it is no longer a control"
    );
    assert!((gpu_mean / cpu_mean - 1.0).abs() < 0.01);
    assert!((gpu_sd / cpu_sd - 1.0).abs() < 0.05);
}

/// The coupled pass drains spilled (n,xn) secondaries too (core#20).
///
/// Same spilling fixture as `spilled_secondaries_keep_mean_and_std_dev`, with
/// secondary-photon transport on so the dispatch takes the coupled path. That
/// path used to refuse the moment a launch spilled, because its bank drain was
/// photon-only. It now relaunches the banked neutrons under their originating
/// source indices before the photon sub-pass, so the neutron flux has to agree
/// with the CPU exactly as on the neutron-only path, and the photons those
/// relaunched neutrons emit have to reach the photon tally: a drain that
/// transported the neutron but dropped its photons would pass the neutron
/// check and fail the photon one.
///
/// The photon flux from beryllium is small and its statistics are wide at this
/// history count, so it is held to a z-score against the two backends' own
/// standard deviations rather than a fixed ratio.
#[test]
fn spilled_secondaries_drain_on_the_coupled_pass() {
    if !data_present("Be9") || !data_present("Be") || !gpu_available() {
        eprintln!("skipping gpu_nxn_spill coupled: Be9 / Be data or f64 GPU absent");
        return;
    }
    let (mut cpu_model, cpu_settings) = build_coupled_model(N_HISTORIES, SOURCE_E, RADIUS);
    cpu_model
        .simulate_transport(&cpu_settings)
        .expect("CPU coupled reference run");
    let (mut gpu_model, gpu_settings) = build_coupled_model(N_HISTORIES, SOURCE_E, RADIUS);
    let spilled = run_gpu_retry(&mut gpu_model, &gpu_settings).expect("GPU coupled run");

    let (cpu_n_mean, cpu_n_sd) = tally_stats(&cpu_model, 0);
    let (gpu_n_mean, gpu_n_sd) = tally_stats(&gpu_model, 0);
    let (cpu_p_mean, cpu_p_sd) = tally_stats(&cpu_model, 1);
    let (gpu_p_mean, gpu_p_sd) = tally_stats(&gpu_model, 1);
    eprintln!(
        "coupled Be9 r={RADIUS} @ {:.1} MeV, {N_HISTORIES} histories, {spilled} spilled:\n  \
         neutron mean    cpu {cpu_n_mean:.6e}  gpu {gpu_n_mean:.6e}  ratio {:.5}\n  \
         neutron std_dev cpu {cpu_n_sd:.6e}  gpu {gpu_n_sd:.6e}  ratio {:.5}\n  \
         photon  mean    cpu {cpu_p_mean:.6e}  gpu {gpu_p_mean:.6e}  ratio {:.5}\n  \
         photon  std_dev cpu {cpu_p_sd:.6e}  gpu {gpu_p_sd:.6e}",
        SOURCE_E / 1e6,
        gpu_n_mean / cpu_n_mean,
        gpu_n_sd / cpu_n_sd,
        gpu_p_mean / cpu_p_mean,
    );

    assert!(
        spilled > 0,
        "no (n,xn) secondary spilled on the coupled pass, so this run never \
         exercised the coupled drain"
    );
    let mean_ratio = gpu_n_mean / cpu_n_mean;
    assert!(
        (mean_ratio - 1.0).abs() < 0.01,
        "coupled GPU neutron flux {gpu_n_mean:.6e} vs CPU {cpu_n_mean:.6e} (ratio \
         {mean_ratio:.5}): a spilled (n,xn) secondary was dropped or double-counted"
    );
    let sd_ratio = gpu_n_sd / cpu_n_sd;
    assert!(
        (sd_ratio - 1.0).abs() < 0.05,
        "coupled GPU neutron std_dev {gpu_n_sd:.6e} vs CPU {cpu_n_sd:.6e} (ratio \
         {sd_ratio:.5}): the relaunched secondaries are not folding into their \
         originating source's variance sample"
    );
    assert!(
        cpu_p_mean > 0.0 && gpu_p_mean > 0.0,
        "no secondary photons scored"
    );
    let z = (gpu_p_mean - cpu_p_mean).abs() / (cpu_p_sd * cpu_p_sd + gpu_p_sd * gpu_p_sd).sqrt();
    assert!(
        z < 4.0,
        "coupled GPU photon flux {gpu_p_mean:.6e} vs CPU {cpu_p_mean:.6e} is {z:.1} sigma \
         apart: photons emitted by relaunched (n,xn) secondaries are not reaching \
         the photon sub-pass"
    );
}
