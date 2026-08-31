//! GPU multi-nuclide moderation regression test (issue #74, Stages 1 + 2a + 2b).
//!
//! Before Stage 1 the GPU kernel used a single density-weighted *average*
//! target mass per material for elastic scattering. In a hydrogenous moderator
//! the light nucleus (large per-collision energy loss) is averaged against the
//! heavy one, so the kernel under-moderates and the thermal flux collapses to
//! roughly half the CPU value (the issue documents ~0.45x for an H2O sphere).
//!
//! Stage 1 selects WHICH nuclide is struck at every collision (proportional to
//! that nuclide's macroscopic total xs) and uses THAT nuclide's exact AWR for
//! the elastic kinematics, lifting the thermal ratio to ~0.66. Stage 2a then
//! samples the struck nuclide's OWN elastic CM angular table (not a material-
//! blended one), correcting the per-collision angular deflection and closing the
//! residual to ~0.96.
//!
//! Tests:
//!   1. a real H2O (H1 + O16) sphere using `~/.cache/yamc` data when present --
//!      the genuine #74 case;
//!   2. a 2:1 H2:C12 (CH2-like) surrogate from the shipped fixtures, used when
//!      the cache is unavailable; same physics (a light + heavy nuclide whose
//!      average AWR / blended angle badly mis-models the light nuclide);
//!   3. a single-nuclide Fe56 sphere that must stay in lock-step (Stages 1/2a/2b
//!      are no-ops with one nuclide -- no extra draw, slab == old material row).
//!   4. a natural-Fe (Fe54/56/57/58) sphere whose 14 MeV fast/inelastic spectrum
//!      was ~15.9% off pre-Stage-2b (the material-blended inelastic secondary
//!      distributions mis-modeled the per-isotope (n,n') down-scatter); Stage 2b
//!      samples the struck isotope's own inelastic distribution + reaction-type
//!      partials, dropping the per-bin GPU/CPU disagreement toward MC noise.
//!
//! Self-skips if the Arrow data or an f64 GPU adapter is absent. Retries the
//! GPU run a few times on transient `BufferAsyncError` (single shared adapter).

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
use yamc_tallies::score::{FluxScore, Score};
use yamc_tallies::tally::Tally;

const SEED: u64 = 20240618;
const N_PER_BATCH: usize = 20_000;
const N_BATCHES: usize = 4;
const MAX_STEPS: u32 = 20_000;
const RADIUS: f64 = 15.0;

fn gpu_available() -> bool {
    yamc_gpu::GpuContext::new().is_ok()
}

/// Energy-binned flux tally: bin 0 = thermal (< 0.5 eV), then a few decade bins
/// up to 20 MeV. `get_mean()[0]` is the thermal-bin flux.
fn energy_binned_tally(cell_id: u32) -> Arc<Tally> {
    let mut tally = Tally::new();
    tally
        .filters
        .push(Filter::Cell(CellFilter::from_id(cell_id)));
    tally.filters.push(Filter::Energy(EnergyFilter::new(vec![
        1.0e-5, 0.5, 1.0e2, 1.0e4, 1.0e6, 2.0e7,
    ])));
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.initialize_batches(N_BATCHES);
    Arc::new(tally)
}

/// Build a single-cell sphere with the given material composition + data, a
/// 14 MeV isotropic point source, and an energy-binned flux tally.
fn build_sphere(
    composition: HashMap<String, f64>,
    density_g_cc: f64,
    neutron_data: HashMap<String, String>,
) -> (Model, Arc<Tally>, TransportSettings) {
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

    let mut material = Material::new(composition, "atom", "g/cm3", Some(density_g_cc)).unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(&neutron_data, None).unwrap();

    finish_model(material, region)
}

/// Like [`build_sphere`] but tolerant of a failed nuclear-data load (the data
/// path may be absent on a given machine). Returns `None` if the data won't
/// load so the caller can self-skip.
fn try_build_sphere(
    composition: HashMap<String, f64>,
    density_g_cc: f64,
    neutron_data: HashMap<String, String>,
) -> Option<(Model, Arc<Tally>, TransportSettings)> {
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

    let mut material = Material::new(composition, "atom", "g/cm3", Some(density_g_cc)).unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    material.read_nuclear_data(&neutron_data, None).ok()?;

    Some(finish_model(material, region))
}

/// Shared tail of `build_sphere` / `try_build_sphere`: wrap the loaded material
/// in a sphere cell, a 14 MeV isotropic point source, and an energy-binned flux
/// tally.
fn finish_model(material: Material, region: Region) -> (Model, Arc<Tally>, TransportSettings) {
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();

    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });

    let tally = energy_binned_tally(1);
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(&tally)]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    let settings = TransportSettings {
        total_particles: Some(N_PER_BATCH * N_BATCHES),
        seed: SEED,
        threads: Some(1),
        ..Default::default()
    };
    (model, tally, settings)
}

/// Run on the GPU, retrying a few times on transient BufferAsyncError (the
/// shared single adapter can hiccup under contention).
fn run_gpu_retry(model: &mut Model, settings: &TransportSettings) {
    let mut last_err = String::new();
    for _ in 0..4 {
        match yamc::gpu::run_on_gpu(model, settings) {
            Ok(_) => return,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("BufferAsyncError") || msg.contains("Async") {
                    last_err = msg;
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    continue;
                }
                panic!("GPU run failed: {msg}");
            }
        }
    }
    panic!("GPU run failed after retries: {last_err}");
}

/// Sum a tally's per-bin mean (per-source-particle flux).
fn total_flux(t: &Arc<Tally>) -> f64 {
    t.get_mean().iter().sum::<f64>()
}

/// Thermal bin (< 0.5 eV) per-source-particle flux.
fn thermal_flux(t: &Arc<Tally>) -> f64 {
    t.get_mean()[0]
}

/// A 2:1 H2:C12 (CH2-like) moderator sphere: GPU thermal flux must recover
/// toward CPU. Pre-#74 the average-AWR kernel under-moderates and the thermal
/// ratio sits near ~0.45; per-collision nuclide selection + per-nuclide elastic
/// AWR (Stage 1) lifted it to ~0.66; per-nuclide elastic ANGULAR sampling
/// (Stage 2a) closes the residual to ~0.96. The Stage 1 gap was the still-
/// material-blended elastic CM angular table, which differs sharply between H2
/// (near-isotropic in CM) and C12 (forward-peaked at MeV) -- with the struck
/// nuclide's own table now sampled, the per-collision angular deflection (and
/// thus the lab-frame energy loss) is correct.
#[test]
fn gpu_multinuclide_moderator_thermal_flux_recovers() {
    if !std::path::Path::new("tests/H2.arrow").exists()
        || !std::path::Path::new("tests/C12.arrow").exists()
    {
        eprintln!("skipping: H2/C12 arrow data absent");
        return;
    }
    if !gpu_available() {
        eprintln!("skipping: no f64 GPU adapter");
        return;
    }

    let composition = HashMap::from([("H2".to_string(), 2.0), ("C12".to_string(), 1.0)]);
    let data = HashMap::from([
        ("H2".to_string(), "tests/H2.arrow".to_string()),
        ("C12".to_string(), "tests/C12.arrow".to_string()),
    ]);

    let (mut cpu_model, cpu_t, settings) = build_sphere(composition.clone(), 0.95, data.clone());
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu_thermal = thermal_flux(&cpu_t);
    let cpu_total = total_flux(&cpu_t);

    let (mut gpu_model, gpu_t, settings) = build_sphere(composition, 0.95, data);
    run_gpu_retry(&mut gpu_model, &settings);
    let gpu_thermal = thermal_flux(&gpu_t);
    let gpu_total = total_flux(&gpu_t);

    let thermal_ratio = gpu_thermal / cpu_thermal;
    let total_ratio = gpu_total / cpu_total;
    eprintln!(
        "H2:C12 moderator: CPU thermal={cpu_thermal:.5e} GPU thermal={gpu_thermal:.5e} \
         ratio={thermal_ratio:.3} (total ratio={total_ratio:.3})"
    );

    assert!(
        cpu_thermal > 0.0 && gpu_thermal > 0.0,
        "both backends must produce thermal flux (cpu {cpu_thermal}, gpu {gpu_thermal})"
    );
    // The #74 Stage 2a fix: with the struck nuclide's own elastic angular table
    // (on top of Stage 1's per-nuclide AWR), the thermal ratio recovers close to
    // 1.0 (~0.96 in practice). A floor of 0.85 fails hard if the per-nuclide
    // elastic angular re-key regresses (Stage 1 alone sat ~0.66, the average-AWR
    // baseline ~0.45); the upper bound catches over-moderation.
    assert!(
        (0.85..=1.12).contains(&thermal_ratio),
        "thermal-bin GPU/CPU ratio {thermal_ratio:.3} outside [0.85, 1.12] -- \
         per-nuclide elastic angular sampling (issue #74 Stage 2a) regressed; \
         Stage 1 (per-nuclide AWR only) sat ~0.66, the average baseline ~0.45"
    );
    // Total flux is dominated by fast transport and is far less sensitive to the
    // AWR fix; a coarse sanity bound.
    assert!(
        (0.85..=1.15).contains(&total_ratio),
        "total GPU/CPU flux ratio {total_ratio:.3} outside [0.85, 1.15]"
    );
}

/// Real H2O (H1 + O16) moderator sphere, drawing the actual ENDF/B-8.1 data
/// from `~/.cache/yamc` when present. This is the genuine #74 case (the H2:C12
/// surrogate above stands in when the cache is unavailable). Per-nuclide elastic
/// angular sampling (Stage 2a) roughly doubles the thermal recovery: with the
/// first-nuclide blended angle (Stage 1) the GPU/CPU thermal ratio sits ~0.43;
/// indexing the struck nuclide's own elastic table lifts it to ~0.81. The
/// remaining gap is the still-material-blended INELASTIC secondary distributions
/// (O16 inelastic is significant at 14 MeV) -- that is Stage 2b of issue #74.
#[test]
fn gpu_real_h2o_thermal_flux_recovers() {
    let h1 = yamc_test_cache::nuclide_path("H1");
    let o16 = yamc_test_cache::nuclide_path("O16");
    if !std::path::Path::new(&h1).exists() || !std::path::Path::new(&o16).exists() {
        eprintln!("skipping: H1/O16 cache data absent ({h1}, {o16})");
        return;
    }
    if !gpu_available() {
        eprintln!("skipping: no f64 GPU adapter");
        return;
    }

    // Water: 2 H : 1 O atom ratio, density 1.0 g/cc.
    let composition = HashMap::from([("H1".to_string(), 2.0), ("O16".to_string(), 1.0)]);
    let data = HashMap::from([("H1".to_string(), h1), ("O16".to_string(), o16)]);

    let Some((mut cpu_model, cpu_t, settings)) =
        try_build_sphere(composition.clone(), 1.0, data.clone())
    else {
        eprintln!("skipping: H1/O16 data present but failed to load at T=294");
        return;
    };
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu_thermal = thermal_flux(&cpu_t);
    let cpu_total = total_flux(&cpu_t);

    let (mut gpu_model, gpu_t, settings) = try_build_sphere(composition, 1.0, data).unwrap();
    run_gpu_retry(&mut gpu_model, &settings);
    let gpu_thermal = thermal_flux(&gpu_t);
    let gpu_total = total_flux(&gpu_t);

    let thermal_ratio = gpu_thermal / cpu_thermal;
    let total_ratio = gpu_total / cpu_total;
    eprintln!(
        "H2O (H1:O16): CPU thermal={cpu_thermal:.5e} GPU thermal={gpu_thermal:.5e} \
         ratio={thermal_ratio:.3} (total ratio={total_ratio:.3})"
    );

    assert!(
        cpu_thermal > 0.0 && gpu_thermal > 0.0,
        "both backends must produce thermal flux (cpu {cpu_thermal}, gpu {gpu_thermal})"
    );
    // Stage 2a recovery: per-nuclide elastic angular lifts the real H2O thermal
    // ratio from the Stage-1 blended-angle ~0.43 to ~0.81. A floor of 0.72 fails
    // hard if the per-nuclide elastic re-key regresses, while leaving the known
    // Stage 2b (inelastic-blend) gap; the upper bound catches over-moderation.
    assert!(
        (0.72..=1.10).contains(&thermal_ratio),
        "real-H2O thermal GPU/CPU ratio {thermal_ratio:.3} outside [0.72, 1.10] -- \
         per-nuclide elastic angular (issue #74 Stage 2a) regressed (blended-angle \
         Stage 1 sat ~0.43); residual above 0.81 is the inelastic blend (Stage 2b)"
    );
    assert!(
        (0.85..=1.15).contains(&total_ratio),
        "real-H2O total GPU/CPU flux ratio {total_ratio:.3} outside [0.85, 1.15]"
    );
}

/// Single-nuclide Fe56 sphere: Stage 1 is a no-op (one nuclide => no selection
/// draw), so the GPU must stay in lock-step with the CPU exactly as before #74.
/// This is the model-level confirmation of the single-nuclide gate (the
/// bit-for-bit kernel/twin check lives in the yamc-gpu `cpu_gpu_equivalence_*`
/// lib tests).
#[test]
fn gpu_single_nuclide_fe56_unchanged() {
    if !std::path::Path::new("tests/Fe56.arrow").exists() {
        eprintln!("skipping: Fe56 arrow data absent");
        return;
    }
    if !gpu_available() {
        eprintln!("skipping: no f64 GPU adapter");
        return;
    }

    let composition = HashMap::from([("Fe56".to_string(), 1.0)]);
    let data = HashMap::from([("Fe56".to_string(), "tests/Fe56.arrow".to_string())]);

    let (mut cpu_model, cpu_t, settings) = build_sphere(composition.clone(), 7.874, data.clone());
    cpu_model.simulate_transport(&settings).unwrap();
    let cpu_total = total_flux(&cpu_t);

    let (mut gpu_model, gpu_t, settings) = build_sphere(composition, 7.874, data);
    run_gpu_retry(&mut gpu_model, &settings);
    let gpu_total = total_flux(&gpu_t);

    let ratio = gpu_total / cpu_total;
    eprintln!(
        "Fe56 single-nuclide: CPU total={cpu_total:.5e} GPU total={gpu_total:.5e} ratio={ratio:.4}"
    );
    assert!(
        (0.97..=1.03).contains(&ratio),
        "single-nuclide Fe56 GPU/CPU flux ratio {ratio:.4} should be ~1 (Stage 1 is a no-op)"
    );
}

/// Largest relative per-bin disagreement between two flux spectra (over bins
/// where the CPU has meaningful counts). The pre-Stage-2b GPU mis-modeled the
/// per-isotope inelastic down-scatter, so the fast/intermediate bins drifted
/// ~15.9% from the CPU; Stage 2b should pull every bin into MC noise.
fn max_rel_bin_disagreement(cpu: &Arc<Tally>, gpu: &Arc<Tally>) -> f64 {
    let c = cpu.get_mean();
    let g = gpu.get_mean();
    let total: f64 = c.iter().sum();
    let mut worst = 0.0_f64;
    for (ci, gi) in c.iter().zip(g.iter()) {
        // Only bins carrying at least 1% of the total flux -- a tiny-count bin's
        // relative error is dominated by MC noise, not physics.
        if *ci > 0.01 * total {
            let rel = (gi - ci).abs() / ci;
            if rel > worst {
                worst = rel;
            }
        }
    }
    worst
}

/// Natural-Fe (Fe54/56/57/58 at natural abundance) sphere from `~/.cache/yamc`.
/// The genuine #74 Stage 2b case: with the material-blended inelastic secondary
/// distributions (Stages 1/2a) the 14 MeV fast/intermediate flux spectrum sat
/// ~15.9% off the CPU because the per-isotope (n,n') down-scatter was averaged
/// across isotopes. Stage 2b samples the STRUCK isotope's own inelastic
/// distribution (and splits the reaction type from its own partials), so the
/// largest per-bin GPU/CPU disagreement drops toward MC noise.
#[test]
fn gpu_natural_fe_inelastic_spectrum_recovers() {
    let isotopes = [
        ("Fe54", 0.05845, yamc_test_cache::nuclide_path("Fe54")),
        ("Fe56", 0.91754, yamc_test_cache::nuclide_path("Fe56")),
        ("Fe57", 0.02119, yamc_test_cache::nuclide_path("Fe57")),
        ("Fe58", 0.00282, yamc_test_cache::nuclide_path("Fe58")),
    ];
    if isotopes
        .iter()
        .any(|(_, _, p)| !std::path::Path::new(p).exists())
    {
        eprintln!("skipping: natural-Fe isotope cache data absent (need Fe54/56/57/58)");
        return;
    }
    if !gpu_available() {
        eprintln!("skipping: no f64 GPU adapter");
        return;
    }

    let composition: HashMap<String, f64> = isotopes
        .iter()
        .map(|(n, frac, _)| (n.to_string(), *frac))
        .collect();
    let data: HashMap<String, String> = isotopes
        .iter()
        .map(|(n, _, p)| (n.to_string(), p.clone()))
        .collect();

    let Some((mut cpu_model, cpu_t, settings)) =
        try_build_sphere(composition.clone(), 7.874, data.clone())
    else {
        eprintln!("skipping: natural-Fe data present but failed to load at T=294");
        return;
    };
    cpu_model.simulate_transport(&settings).unwrap();

    let (mut gpu_model, gpu_t, settings) = try_build_sphere(composition, 7.874, data).unwrap();
    run_gpu_retry(&mut gpu_model, &settings);

    let worst = max_rel_bin_disagreement(&cpu_t, &gpu_t);
    let total_ratio = total_flux(&gpu_t) / total_flux(&cpu_t);
    eprintln!(
        "natural Fe: max per-bin GPU/CPU disagreement={:.1}% (total ratio={total_ratio:.3})\n  CPU bins={:?}\n  GPU bins={:?}",
        worst * 100.0,
        cpu_t.get_mean(),
        gpu_t.get_mean()
    );

    // Pre-Stage-2b the worst significant bin sat ~15.9% off (the inelastic
    // blend). Stage 2b's per-isotope inelastic sampling should pull every
    // meaningful bin under ~8% (statistics on 80k histories); a hard ceiling of
    // 10% fails loudly if the per-nuclide inelastic re-key regresses.
    assert!(
        worst < 0.10,
        "natural-Fe max per-bin GPU/CPU disagreement {:.1}% exceeds 10% -- per-nuclide \
         inelastic distributions (issue #74 Stage 2b) regressed (the material-blended \
         baseline sat ~15.9%)",
        worst * 100.0
    );
    assert!(
        (0.92..=1.08).contains(&total_ratio),
        "natural-Fe total GPU/CPU flux ratio {total_ratio:.3} outside [0.92, 1.08]"
    );
}
