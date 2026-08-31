//! Issue #111: matched-stream LOCALIZATION harness.
//!
//! Generalizes the Fe56-only `matched_stream_diff` to any endf-b8.1 cache
//! nuclide, to pin down WHERE the production CPU (the OpenMC-validated path,
//! still on the legacy `FastRng` tabulated sampler for continuum / correlated /
//! Kalbach secondary energy) diverges from the GPU twin (the shared flat
//! sampler). The twin is bit-identical to the cubecl kernel, so a production-CPU
//! vs twin per-history diff localizes the un-unified `#111` secondary-production
//! path without needing a physical GPU.
//!
//! Unlike the earlier probe that produced the `#156` artifact, this records, at
//! the FIRST per-history divergence, the REAL reaction MT, the CPU
//! energy-distribution KIND (`TrackEvent::energy_dist` / `distribution`), and
//! the `energy_out` on BOTH sides -- so the localization is grounded in the
//! actual reaction/distribution, not a collapsed classifier code.
//!
//! Reads neutron data from the `endf-b8.1-<nuclide>.arrow` cache entry (the same
//! data the V&V harness uses), so it reproduces the exact divergence the V&V
//! sweep reports (F19 elastic/discrete -0.52%, U235 fission -0.53%).
//!
//! Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test matched_stream_localize -- --nocapture

#![cfg(all(feature = "gpu", not(target_os = "macos")))]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::gpu::translate_for_gpu;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc::track::{HistorySelection, TrackEventType};
use yamc_gpu::common::tallies::TalliesPack;
use yamc_gpu::neutron::transport::{
    run_multi_cell_transport_cpu, CollisionRecord, FissionBankInputs, PendDrain,
    SurvivalBiasingInputs,
};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const RADIUS: f64 = 5.0;
const MAX_STEPS: u32 = 5_000;
const SEED: u64 = 4242;
const FAST_ENERGY: f64 = 14.06e6;

fn cache_dir(nuclide: &str) -> String {
    yamc_test_cache::nuclide_path(nuclide)
}

/// Whether this machine can supply `nuclide` at the scope these tests need.
///
/// Reading is the check, not `is_dir`. Since #389 a cache directory is
/// routinely populated at activation scope, holding cross sections and none of
/// the transport sections; the directory exists either way, and the read below
/// is what tells them apart. CI fetches only the fixture list, so it skips
/// these outright; without this, any developer machine that has run a
/// transmutation panics in `nuclide_sphere` instead.
fn data_present(nuclide: &str) -> bool {
    let path = cache_dir(nuclide);
    if !std::path::Path::new(&path).is_dir() {
        return false;
    }
    yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow(
        std::path::Path::new(&path),
        &yamc_nuclide::LoadScope::full(),
    )
    .is_ok()
}

/// Single-nuclide sphere (vacuum boundary), neutron data from the endf-b8.1
/// cache. `Below(sphere)` keeps it a single bounded cell for the GPU AABB pass.
fn nuclide_sphere(nuclide: &str, density: f64) -> Geometry {
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
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

/// Monodirectional (+z) point source at the origin, single discrete energy:
/// zero RNG draws at birth, so the per-particle PCG stream starts fresh and
/// identical on both backends.
/// As [`build_model`], optionally with survival biasing enabled.
fn build_model_with(nuclide: &str, density: f64, energy_ev: f64, survival: bool) -> Model {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(nuclide_sphere(nuclide, density), vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    if survival {
        model.variance_reduction = vec![
            yamc::variance_reduction::VarianceReduction::SurvivalBiasing(
                yamc::variance_reduction::SurvivalBiasing::default(),
            ),
        ];
    }
    model
}

/// One collision, normalized across the two backends. `class` collapses the
/// ENDF MT so the absorbing collision (whose specific MT the twin reports as
/// 102) still aligns. `mt`/`dist` are the CPU-side raw reaction + energy-dist
/// kind, kept ONLY for divergence reporting (not used in equality).
#[derive(Clone)]
struct Col {
    e_in: f64,
    e_out: f64,
    class: u8,
    mt: i32,
    dist: String,
}

/// Reaction class: 0 elastic, 1 inelastic/scatter-with-multiplicity, 2 fission,
/// 3 absorption (capture / charged-particle). Mirrors `matched_stream_diff`.
fn class_of(mt: i32) -> u8 {
    match mt {
        2 => 0,
        18 => 2,
        16 | 17 => 1,
        51..=93 => 1,
        _ => 3,
    }
}

/// Map a twin `CollisionRecord.reaction` code back to a real ENDF MT.
/// `reaction_rec = 51 + slot`, so codes 51..=91 are literal MTs and codes from
/// 92 up index the extra `MT_SLOTS` entries (16/17/22/28/32/33/34/5/...).
/// Code 102 is the absorption literal (it also collides with slot 51 = MT25,
/// negligible at 14 MeV; treated as absorption here).
fn twin_code_to_mt(code: i32) -> i32 {
    match code {
        2 | 18 | 102 => code,
        51..=91 => code,
        92 => 16,
        93 => 17,
        94 => 22,
        95 => 28,
        96 => 32,
        97 => 33,
        98 => 34,
        99 => 5,
        100 => 23,
        101 => 24,
        103 => 37,
        104 => 41,
        105 => 44,
        106 => 45,
        other => other,
    }
}

fn col_eq(a: &Col, b: &Col) -> bool {
    a.e_in.to_bits() == b.e_in.to_bits()
        && a.e_out.to_bits() == b.e_out.to_bits()
        && a.class == b.class
}

/// Relative tolerance for the whole-history comparison, same value and same
/// reasoning as `matched_stream_diff`'s `HISTORY_REL_TOLERANCE`: a last-place
/// difference at one collision becomes the next collision's incoming energy, so
/// association noise compounds along a history. Still ~7 orders of magnitude
/// below any real divergence, because a different `xi`, level or law moves the
/// outgoing energy by a fraction of ITSELF, not by its trailing bits.
const HISTORY_REL_TOLERANCE: f64 = 1e-9;

/// Whether the two histories agree collision for collision to within
/// [`HISTORY_REL_TOLERANCE`]. This is what separates a real stream or sampler
/// divergence -- the backends sampled DIFFERENT physics, which moves an outgoing
/// energy by a fraction of ITSELF -- from the two arriving at the same closed
/// form a few ulp apart, which moves only its trailing bits. Read it next to the
/// strict count: a gap between them is that trailing-bit noise, and where it
/// comes from has to be chased case by case (for Am240 it was the twin's
/// effective target mass, a `(N * A) / N` round-trip that lands one ulp above the
/// `A` the CPU uses).
fn same_history_within_rounding(cpu: &[Col], twin: &[Col]) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= HISTORY_REL_TOLERANCE * x.abs().max(y.abs());
    cpu.len() == twin.len()
        && cpu
            .iter()
            .zip(twin)
            .all(|(a, b)| a.class == b.class && close(a.e_in, b.e_in) && close(a.e_out, b.e_out))
}

/// First collision index at which the two sequences differ, or `None` when
/// fully identical. A length mismatch with a matching prefix diverges at the
/// shorter length.
fn first_divergence(cpu: &[Col], twin: &[Col]) -> Option<usize> {
    let common = cpu.len().min(twin.len());
    for k in 0..common {
        if !col_eq(&cpu[k], &twin[k]) {
            return Some(k);
        }
    }
    if cpu.len() != twin.len() {
        Some(common)
    } else {
        None
    }
}

/// What [`run_case`] reports back, so callers can put light assertions on top of
/// the printed localization report.
struct CaseReport {
    /// Histories that collided at least once on BOTH backends.
    both: usize,
    /// Histories bit-identical collision for collision, end to end.
    identical: usize,
    /// Histories identical to within [`HISTORY_REL_TOLERANCE`]. The gap to
    /// `identical` is trailing-bit arithmetic, not divergent sampling.
    identical_within_rounding: usize,
    /// Histories run.
    n: usize,
    /// MT -> `(cpu_n, twin_n, mean gap %)` of the collision-0 outgoing-energy
    /// means, letting a caller pin one channel's sampler parity (e.g. Ar38 MT91,
    /// or MT18 for the fission chi).
    collision0_gaps: HashMap<i32, (usize, usize, f64)>,
}

/// Analog (survival-off) case, which is what most of these localizers want.
fn run_case(label: &str, nuclide: &str, density: f64, energy_ev: f64, n: usize) -> CaseReport {
    run_case_with(label, nuclide, density, energy_ev, n, false)
}

/// As [`run_case`], optionally with survival biasing on BOTH backends.
fn run_case_with(
    label: &str,
    nuclide: &str,
    density: f64,
    energy_ev: f64,
    n: usize,
    survival: bool,
) -> CaseReport {
    // --- Production CPU: per-history collision trace via track capture. ---
    let mut model = build_model_with(nuclide, density, energy_ev, survival);
    let settings = TransportSettings {
        total_particles: Some(n),
        seed: SEED,
        ..Default::default()
    };
    let inputs = translate_for_gpu(&model, n, settings.seed)
        .expect("translate single-nuclide model for GPU twin");
    for i in 0..n {
        assert_eq!(
            inputs.seeds[i],
            yamc_rng::history_seed(SEED, i as u64),
            "GPU per-particle seed must equal history_seed(base_seed, i) for batch 0"
        );
    }

    let storage = model
        .run_with_tracking(&settings, HistorySelection::first(n as u64))
        .expect("CPU tracked run");
    let mut cpu: Vec<Vec<Col>> = vec![Vec::new(); n];
    for track in &storage.tracks {
        if track.generation != 0 {
            continue;
        }
        for ev in &track.events {
            if ev.event_type == TrackEventType::Collision && ev.history < n {
                cpu[ev.history].push(Col {
                    e_in: ev.energy_in,
                    e_out: ev.energy_out,
                    class: class_of(ev.reaction_mt.unwrap_or(0)),
                    mt: ev.reaction_mt.unwrap_or(0),
                    dist: ev
                        .energy_dist
                        .clone()
                        .or_else(|| ev.distribution.clone())
                        .unwrap_or_default(),
                });
            }
        }
    }

    // --- GPU twin: same model, same seeds, per-collision trace. ---
    let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
    let (_twin_res, twin_traces) = run_multi_cell_transport_cpu(
        &inputs.seeds,
        &inputs.energies,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.coarse_log_energy_grid,
        &inputs.coarse_meta,
        &inputs.fine_log_energy_grid,
        &inputs.fine_meta,
        &inputs.xs_elastic_per_material,
        &inputs.xs_absorption_per_material,
        &inputs.xs_inelastic_per_material,
        &inputs.xs_fission_per_material,
        &inputs.nu_bar_per_material,
        &inputs.beta_delayed_per_material,
        &inputs.fission_a_per_material,
        &inputs.fission_b_per_material,
        &inputs.fission_eout_kind_per_material,
        &inputs.fission_eout_n_energies_per_material,
        &inputs.fission_eout_ae_offset,
        &inputs.fission_eout_energy_grid_per_material,
        &inputs.fission_eout_n_x_per_material,
        &inputs.fission_eout_x_offset,
        &inputs.fission_eout_x_per_material,
        &inputs.fission_eout_cdf_per_material,
        &inputs.fission_eout_p_per_material,
        &inputs.fission_eout_interp_per_material,
        &inputs.xs_inelastic_per_mt_sparse,
        &inputs.target_mass_per_material,
        &inputs.q_inelastic_per_mt,
        &inputs.yield_per_mt_sparse,
        &inputs.permt_meta,
        &inputs.angle_n_energies,
        &inputs.angle_ae_offset,
        &inputs.angle_energy_grid,
        &inputs.angle_n_mu,
        &inputs.angle_mu_offset,
        &inputs.angle_mu,
        &inputs.angle_cdf,
        &inputs.angle_pdf,
        &inputs.angle_interp,
        &inputs.eout_kind,
        &inputs.eout_n_energies,
        &inputs.eout_ae_offset,
        &inputs.eout_energy_grid,
        &inputs.eout_n_x,
        &inputs.eout_x_offset,
        &inputs.eout_x,
        &inputs.eout_cdf,
        &inputs.eout_histogram_interp,
        &inputs.eout_p,
        &inputs.eout_interp,
        &inputs.eout_n_discrete,
        &inputs.corr_n_energies,
        &inputs.corr_n_components,
        &inputs.corr_ae_offset,
        &inputs.corr_energy_grid,
        &inputs.corr_n_x,
        &inputs.corr_x_offset,
        &inputs.corr_x,
        &inputs.corr_cdf,
        &inputs.corr_p,
        &inputs.corr_interp,
        &inputs.corr_n_discrete,
        &inputs.corr_n_mu,
        &inputs.corr_mu_offset,
        &inputs.corr_mu,
        &inputs.corr_mu_cdf,
        &inputs.corr_mu_pdf,
        &inputs.corr_mu_interp,
        &inputs.scatter_in_cm_per_mt,
        &inputs.elastic_angle_n_energies,
        &inputs.elastic_angle_ae_offset,
        &inputs.elastic_angle_energy_grid,
        &inputs.elastic_angle_n_mu,
        &inputs.elastic_angle_mu_offset,
        &inputs.elastic_angle_mu,
        &inputs.elastic_angle_cdf,
        &inputs.elastic_angle_pdf,
        &inputs.elastic_angle_interp,
        &inputs.temperature_k_per_material,
        &inputs.km_n_energies,
        &inputs.km_ae_offset,
        &inputs.km_energy_grid,
        &inputs.km_interp,
        &inputs.km_n_discrete,
        &inputs.km_n_x,
        &inputs.km_x_offset,
        &inputs.km_x,
        &inputs.km_p,
        &inputs.km_c,
        &inputs.km_r,
        &inputs.km_a,
        &inputs.evap_n_energies,
        &inputs.evap_n_components,
        &inputs.evap_ae_offset,
        &inputs.evap_theta_offset,
        &inputs.evap_energy_grid,
        &inputs.evap_theta,
        &inputs.evap_u,
        &inputs.nbps_n_bodies,
        &inputs.nbps_total_mass,
        &inputs.maxwell_n_energies,
        &inputs.maxwell_ae_offset,
        &inputs.maxwell_energy_grid,
        &inputs.maxwell_theta,
        &inputs.maxwell_u,
        &inputs.watt_n_energies,
        &inputs.watt_ae_offset,
        &inputs.watt_energy_grid,
        &inputs.watt_a,
        &inputs.watt_b,
        &inputs.watt_u,
        &inputs.urr_meta,
        &inputs.urr_ae_offset,
        &inputs.urr_cdf_offset,
        &inputs.urr_energy_grid,
        &inputs.urr_cdf,
        &inputs.urr_xs,
        &inputs.urr_atom_density,
        &pack,
        &[],
        &if survival {
            let sb = yamc::variance_reduction::SurvivalBiasing::default();
            SurvivalBiasingInputs::on(sb.weight_cutoff, sb.weight_survive)
        } else {
            SurvivalBiasingInputs::off()
        },
        &inputs.nuclide_select,
        // Fission bank ON, which is what a fissile model runs on the GPU
        // (`Model::gpu_fission_bank` defaults true, and `gpu/dispatch.rs` routes
        // a model with fission cross sections to the per-source bank path). It
        // is also the only mode that models the same physics as the production
        // CPU: bank OFF multiplies the walk's weight by nu_bar and never draws
        // the multiplicity uniform, so a fission collision could not be
        // stream-identical to the CPU's analog banking however the draws were
        // ordered.
        &FissionBankInputs::on(),
        MAX_STEPS,
        400.0,
        true,
        // Drain the twin's in-thread (n,xn) queue LIFO, matching the
        // production CPU's `ParticleBank` stack, so the two backends emit a
        // history's collisions in the SAME sequence and the diff below stays a
        // measurement of the stream and the samplers rather than of the
        // scheduling. Since #111 phase 1 every secondary carries its own
        // identity-derived seed, so the order cannot change what a secondary
        // samples -- `secondary_stream_order.rs` proves that by running this
        // same twin FIFO and LIFO and demanding identical tallies. The kernel
        // itself stays FIFO.
        PendDrain::Lifo,
    );
    assert_eq!(twin_traces.len(), n, "twin must emit one trace per history");
    let twin: Vec<Vec<Col>> = twin_traces
        .iter()
        .map(|recs| {
            recs.iter()
                .map(|r: &CollisionRecord| Col {
                    e_in: r.energy_in,
                    e_out: r.energy_out,
                    class: class_of(r.reaction),
                    mt: r.reaction,
                    dist: String::new(),
                })
                .collect()
        })
        .collect();

    // --- Localize: first divergence per history, bucketed by CPU reaction. ---
    let mut both = 0usize;
    let mut fully_identical = 0usize;
    let mut identical_within_rounding = 0usize;
    let mut diverged = 0usize;
    // key: (cpu class, cpu mt, cpu dist) -> (count, sum |e_out gap| relative).
    let mut buckets: HashMap<(u8, i32, String), (usize, f64)> = HashMap::new();
    let mut examples: Vec<String> = Vec::new();
    let mut cpu_cols_total = 0usize;
    let mut twin_cols_total = 0usize;
    // Collision-0 outgoing-energy spectrum per real MT, accumulated separately
    // for CPU and twin. At collision 0 the incident energy is identical (the
    // source energy) on both paths, so a per-MT mean/std difference here is a
    // PURE secondary-energy sampler difference -- free of RNG-stream noise
    // (averaged out over many histories) and downstream feedback. key: real MT,
    // val: (count, sum_eout, sum_eout_sq).
    let mut cpu_spec: HashMap<i32, (usize, f64, f64)> = HashMap::new();
    let mut twin_spec: HashMap<i32, (usize, f64, f64)> = HashMap::new();
    let acc = |m: &mut HashMap<i32, (usize, f64, f64)>, mt: i32, e: f64| {
        let s = m.entry(mt).or_insert((0, 0.0, 0.0));
        s.0 += 1;
        s.1 += e;
        s.2 += e * e;
    };

    for i in 0..n {
        let c = &cpu[i];
        let t = &twin[i];
        cpu_cols_total += c.len();
        twin_cols_total += t.len();
        if let Some(c0) = c.first() {
            acc(&mut cpu_spec, c0.mt, c0.e_out);
        }
        if let Some(t0) = t.first() {
            acc(&mut twin_spec, twin_code_to_mt(t0.mt), t0.e_out);
        }
        if !c.is_empty() && !t.is_empty() {
            both += 1;
        }
        if same_history_within_rounding(c, t) {
            identical_within_rounding += 1;
        }
        match first_divergence(c, t) {
            None => fully_identical += 1,
            Some(k) => {
                diverged += 1;
                // Describe the divergence using the CPU collision at k (the
                // OpenMC-validated reference). If the CPU ran out first (length
                // mismatch), fall back to the twin collision.
                let (cls, mt, dist, c_eout, t_eout, e_in) = if k < c.len() && k < t.len() {
                    (
                        c[k].class,
                        c[k].mt,
                        c[k].dist.clone(),
                        c[k].e_out,
                        t[k].e_out,
                        c[k].e_in,
                    )
                } else if k < c.len() {
                    (
                        c[k].class,
                        c[k].mt,
                        c[k].dist.clone(),
                        c[k].e_out,
                        f64::NAN,
                        c[k].e_in,
                    )
                } else {
                    (
                        t[k].class,
                        t[k].mt,
                        "<twin>".to_string(),
                        f64::NAN,
                        t[k].e_out,
                        t[k].e_in,
                    )
                };
                let rel = if c_eout.is_finite() && t_eout.is_finite() && c_eout != 0.0 {
                    ((t_eout - c_eout) / c_eout).abs()
                } else {
                    0.0
                };
                let e = buckets.entry((cls, mt, dist.clone())).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += rel;
                if examples.len() < 25 {
                    examples.push(format!(
                        "  hist {i} @col {k}: cls={cls} cpu(mt={mt} dist='{dist}' e_in={e_in:.4e} e_out={c_eout:.4e}) twin(code={} e_out={t_eout:.4e})",
                        t.get(k).map(|x| x.mt).unwrap_or(-1)
                    ));
                }
            }
        }
    }

    eprintln!(
        "\n==== matched-stream LOCALIZE: {label} ({nuclide}, {energy_ev:.3e} eV, N={n}) ===="
    );
    eprintln!("histories with >=1 collision (both) : {both} / {n}");
    eprintln!(
        "fully bit-identical histories       : {fully_identical} / {n} ({:.2}%)",
        100.0 * fully_identical as f64 / n as f64
    );
    eprintln!(
        "identical within {HISTORY_REL_TOLERANCE:.0e}               : {identical_within_rounding} / {n} ({:.3}%)",
        100.0 * identical_within_rounding as f64 / n as f64
    );
    eprintln!(
        "histories that diverge              : {diverged} / {n} ({:.2}%)",
        100.0 * diverged as f64 / n as f64
    );
    eprintln!(
        "mean collisions / history           : cpu {:.2}, twin {:.2}",
        cpu_cols_total as f64 / n as f64,
        twin_cols_total as f64 / n as f64
    );
    eprintln!("\nfirst-divergence buckets (by CPU reaction at the divergence point):");
    let mut rows: Vec<_> = buckets.into_iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1 .0));
    for ((cls, mt, dist), (count, sum_rel)) in &rows {
        let mean_rel = if *count > 0 {
            sum_rel / *count as f64
        } else {
            0.0
        };
        eprintln!(
            "  cls={cls} mt={mt:<4} dist='{dist}'  -> {count} histories (mean |e_out gap| {:.2}%)",
            100.0 * mean_rel
        );
    }
    eprintln!("\nfirst {} divergence examples:", examples.len());
    for e in &examples {
        eprintln!("{e}");
    }

    // Collision-0 per-MT outgoing-energy spectrum: CPU vs twin mean (eV) at the
    // identical source incident energy. A mean gap > ~0.5% on a well-sampled MT
    // is the biased flat sampler for that reaction's law.
    eprintln!("\ncollision-0 outgoing-energy spectrum (CPU vs twin, same e_in):");
    eprintln!(
        "  {:<6} {:>8} {:>12} {:>8} {:>12}   {:>9}",
        "MT", "cpu_n", "cpu_mean", "twin_n", "twin_mean", "mean_gap"
    );
    let mut mts: Vec<i32> = cpu_spec.keys().chain(twin_spec.keys()).copied().collect();
    mts.sort_unstable();
    mts.dedup();
    let mean = |s: Option<&(usize, f64, f64)>| -> (usize, f64) {
        match s {
            Some(&(n, sum, _)) if n > 0 => (n, sum / n as f64),
            _ => (0, f64::NAN),
        }
    };
    let mut gaps: HashMap<i32, (usize, usize, f64)> = HashMap::new();
    for mt in mts {
        let (cn, cm) = mean(cpu_spec.get(&mt));
        let (tn, tm) = mean(twin_spec.get(&mt));
        // Only flag MTs with enough samples on both sides for a meaningful mean.
        let gap = if cm.is_finite() && tm.is_finite() && cm != 0.0 {
            (tm - cm) / cm * 100.0
        } else {
            f64::NAN
        };
        let flag = if cn >= 200 && tn >= 200 && gap.abs() > 0.5 {
            "  <-- SAMPLER GAP"
        } else {
            ""
        };
        eprintln!("  MT{mt:<4} {cn:>8} {cm:>12.4e} {tn:>8} {tm:>12.4e}   {gap:>+8.2}%{flag}");
        gaps.insert(mt, (cn, tn, gap));
    }
    CaseReport {
        both,
        identical: fully_identical,
        identical_within_rounding,
        n,
        collision0_gaps: gaps,
    }
}

/// F19 at 14 MeV: the FLiBe-relevant nuclide whose V&V residual (-0.52% across
/// elastic and discrete-level) is the cleanest non-fission case of the
/// un-unified continuum secondary-energy path. The asserts only guard the
/// already-unified majority; the divergence buckets are the localization output.
#[test]
fn localize_f19_14mev() {
    if !data_present("F19") {
        eprintln!("skipping localize_f19_14mev -- endf-b8.1-F19.arrow cache absent");
        return;
    }
    let n = 4000;
    let rep = run_case("F19 14 MeV", "F19", 1.7, FAST_ENERGY, n);
    // Non-triviality: a monodirectional 14 MeV beam in a 5 cm F19 sphere must
    // collide in a substantial fraction of histories, and the already-unified
    // channels (elastic / capture / discrete-level inelastic) must keep a large
    // bit-identical majority -- a regression here means a previously-unified
    // path desynced. The continuum/multi-body remainder is the open #111 work.
    assert!(
        rep.both > n / 4,
        "expected a substantial fraction of histories to collide ({} / {n})",
        rep.both
    );
    assert!(
        rep.identical > n / 2,
        "expected a bit-identical majority on the already-unified path ({} / {n})",
        rep.identical
    );
}

/// U235 at 14 MeV: the fissile residual of issue #154 (GPU/CPU flux 0.9978).
/// Fewer fissions than the Am240 cases (107 of 4000 histories at collision 0).
///
/// The end-to-end claim is the tolerant one (every history within
/// [`HISTORY_REL_TOLERANCE`], asserted at 100%); STRICT bit-identity is a floor,
/// not an equality, because the two backends' arithmetic ASSOCIATION differs on
/// some draws and whether that shows up in a given history depends on where the
/// stream lands. This case read a full 4000 / 4000 strict until the delayed-neutron
/// branch (issue #364) added one uniform per fission progeny; the same histories
/// still agree within 1e-9, one of them now off in the trailing bits of an MT 53
/// level draw. The floor is set well below the measured 3999 so an actual draw-order
/// break, which collapses strict agreement to a small fraction, still fails it.
#[test]
fn localize_u235_14mev() {
    if !data_present("U235") {
        eprintln!("skipping localize_u235_14mev -- endf-b8.1-U235.arrow cache absent");
        return;
    }
    let n = 4000;
    let rep = run_case("U235 14 MeV", "U235", 1.0, FAST_ENERGY, n);
    assert_identical_within_rounding(&rep);
    assert!(
        rep.identical * 100 >= n * 99,
        "U235 histories must be bit-identical to the twin end to end apart from \
         arithmetic association: {} / {n}",
        rep.identical
    );
}

/// The twin's effective target mass for a single-nuclide material must be that
/// nuclide's atomic weight ratio EXACTLY.
///
/// `extract_material_xs` reports a density-weighted average mass, and
/// `(N * A) / N` is not always `A`: for Am240 at 5 g/cm3 it lands one ulp above,
/// where W184, Fe56 and F19 happen to round-trip cleanly. The CPU's elastic
/// kinematics use `A` itself, so that one ulp put every non-free-gas elastic
/// scatter a few ulp off the CPU's -- 14% of Am240 histories at 14 MeV and 27% at
/// 180 keV, while W184 read 100% bit-identical (issue #111).
#[test]
fn twin_target_mass_is_the_nuclides_awr() {
    for (nuclide, density) in [("Am240", 5.0), ("W184", 19.3), ("Fe56", 7.87), ("F19", 1.7)] {
        if !data_present(nuclide) {
            continue;
        }
        let model = build_model_with(nuclide, density, FAST_ENERGY, false);
        let awr = model.geometry.materials()[0]
            .nuclide_data
            .get(nuclide)
            .expect("nuclide data loaded")
            .atomic_weight_ratio
            .expect("nuclide carries an atomic weight ratio");
        let inputs = translate_for_gpu(&model, 8, SEED).expect("translate single-nuclide model");
        let target_mass = inputs.target_mass_per_material[0];
        assert_eq!(
            target_mass.to_bits(),
            awr.to_bits(),
            "{nuclide}: twin target_mass {target_mass:.17e} is not the AWR {awr:.17e}"
        );
    }
}

/// Ar38 at 14 MeV: the V&V broomstick residual (reduced chi2 ~28, GPU flux
/// exactly zero across 8.2--11.6 MeV). Root cause: MT91 carries TWO
/// evaporation laws step-gated by applicability (u = 5.9 MeV below 11 MeV,
/// u = 2.2 MeV above) and the flat extractor's collapse grid missed the
/// 11 MeV window edge, so the twin/GPU kept u = 5.9 MeV at 14.06 MeV and
/// hard-truncated the outgoing spectrum at 8.16 MeV. The MT91 collision-0
/// mean-gap assertion guards that fix.
#[test]
fn parity_ar38_collision0_spectrum() {
    if !data_present("Ar38") {
        eprintln!("skipping parity_ar38_collision0_spectrum -- endf-b8.1-Ar38.arrow cache absent");
        return;
    }
    let n = 100_000;
    let rep = run_case("Ar38 collision-0 spectrum", "Ar38", 5.0, FAST_ENERGY, n);
    assert!(
        rep.both > n / 4,
        "expected substantial collisions ({} / {n})",
        rep.both
    );
    let (cpu_n, twin_n, gap) = rep
        .collision0_gaps
        .get(&91)
        .copied()
        .unwrap_or((0, 0, f64::NAN));
    assert!(
        cpu_n > 1000 && twin_n > 1000,
        "expected MT91 to be well-sampled at 14.06 MeV ({cpu_n} / {twin_n})"
    );
    // Pre-fix the truncated evaporation band gave -5.0%; MC noise at this N
    // is ~0.6% on the MT91 mean, so 2% cleanly separates the two.
    assert!(
        gap.abs() < 2.0,
        "Ar38 MT91 collision-0 outgoing-energy mean gap {gap:+.2}% (twin vs CPU)"
    );
}

/// Am240 at 14 MeV: the V&V sphere residual (reduced chi2 ~20, +/-8% flux
/// oscillation across 100--300 keV). Fission is ~45% of the collisions here, so
/// this is the fissile case for the #111 fission draw schedule.
///
/// The CPU used to round `nu_bar` into N BEFORE sampling the continuing
/// progeny's chi and to draw a fresh isotropic cosine for it, while the kernel
/// samples that chi first and reuses the reaction split's `xi3` as the cosine.
/// The chi therefore came off a different point of the shared stream on the two
/// backends, and the MT18 collision-0 outgoing-energy mean sat a percent or so
/// apart with no fixed sign (+4.3% on U235). That is the first assertion below.
///
/// The banked progeny then desynced the rest of the walk: the kernel spent a
/// variable-length Marsaglia azimuth (2 to 16 draws) and a drawn transport seed
/// on each one, where the CPU spends a single `TAU * xi` and derives the seed
/// from the emission ordinal. 78.30% of histories were bit-identical; the
/// second assertion is the end-to-end one.
#[test]
fn parity_am240_collision0_spectrum() {
    if !data_present("Am240") {
        eprintln!(
            "skipping parity_am240_collision0_spectrum -- endf-b8.1-Am240.arrow cache absent"
        );
        return;
    }
    let n = 100_000;
    let rep = run_case("Am240 collision-0 spectrum", "Am240", 5.0, FAST_ENERGY, n);
    assert!(
        rep.both > n / 4,
        "expected substantial collisions ({} / {n})",
        rep.both
    );
    assert_fission_chi_bit_identical(&rep);
    assert_identical_within_rounding(&rep);
}

/// The collision-0 MT18 outgoing energy must be EXACTLY equal on both backends.
/// At collision 0 the incident energy is the source energy on both sides, so a
/// fission there draws its chi from the same stream position off the same seed:
/// the two means are sums of the same values in the same order, hence bit-equal.
/// Any nonzero gap means the chi landed at a different stream position (the
/// pre-#111 order), or the two backends are reading different chi data.
fn assert_fission_chi_bit_identical(rep: &CaseReport) {
    let (cpu_n, twin_n, gap) = rep
        .collision0_gaps
        .get(&18)
        .copied()
        .unwrap_or((0, 0, f64::NAN));
    assert!(
        cpu_n > 1000 && twin_n > 1000,
        "expected fission to be well-sampled at collision 0 ({cpu_n} / {twin_n})"
    );
    assert_eq!(
        gap, 0.0,
        "MT18 collision-0 outgoing-energy mean gap {gap:+.3e}% (twin vs CPU): the fission chi \
         is not drawn at the same point of the shared stream, or the two backends disagree on \
         the chi data (#111 fission)"
    );
}

/// Every history must match collision for collision to within
/// [`HISTORY_REL_TOLERANCE`]. This is the end-to-end fission claim: the same
/// reactions, the same energies and the same NUMBER of collisions, banked fission
/// progeny included -- both backends transport those inside the history, the CPU
/// off its `ParticleBank` and the twin off its in-thread pending stack. The
/// remaining strict-bit gap is arithmetic association, which is why this is the
/// tolerant metric; `identical` is printed next to it.
fn assert_identical_within_rounding(rep: &CaseReport) {
    assert_eq!(
        rep.identical_within_rounding,
        rep.n,
        "{} / {} histories diverged beyond {HISTORY_REL_TOLERANCE:.0e} -- a fission draw \
         schedule or sampler difference, not association noise (#111 fission)",
        rep.n - rep.identical_within_rounding,
        rep.n
    );
}

/// Am240 at LOW incident energies: the V&V +/-8% flux oscillation sits at
/// 100--300 keV, right on the low-lying rotational levels (Q = 41..252 keV),
/// where the closed-form level kinematics and the per-MT angle tables are
/// most sensitive. Collision-0 at 400 keV / 200 keV exercises exactly those
/// channels (elastic + MT51..~57 + fission). Report-only localizer.
#[test]
fn parity_am240_low_energy_spectrum() {
    if !data_present("Am240") {
        eprintln!(
            "skipping parity_am240_low_energy_spectrum -- endf-b8.1-Am240.arrow cache absent"
        );
        return;
    }
    let n = 100_000;
    for &energy in &[250.0e3_f64, 180.0e3] {
        let label = format!("Am240 collision-0 spectrum @ {:.0} keV", energy / 1e3);
        let rep = run_case(&label, "Am240", 5.0, energy, n);
        assert!(
            rep.both > n / 4,
            "expected substantial collisions at {energy:.1e} eV ({} / {n})",
            rep.both
        );
        assert_fission_chi_bit_identical(&rep);
        assert_identical_within_rounding(&rep);
    }
}

/// Th232 at 14 MeV: the fissile case the shared fission path does NOT cover, so
/// the one place the assertions above deliberately do not apply.
///
/// Th232, Pa231 and Pa233 are the only three fissionable nuclides in endf-b8.1
/// whose prompt-fission spectrum is `CorrelatedAngleEnergy` (the other 85 are 74
/// ContinuousTabular + 11 Maxwell). `transport/fission.rs`'s `prompt_chi_dist`
/// only recognises the uncorrelated form, so the CPU gets `FissionChiFlat::None`
/// and falls back to the legacy `FastRng` sampler, off the shared stream, while
/// the GPU host packs the correlated table's E_out marginal into its fission
/// buffers and emits isotropically in lab. So the MT18 line below shows a real
/// gap where every other fissile nuclide's is bit-equal. Report-only; issue #356
/// carries the fix, which needs a decision on the emission angle first.
#[test]
fn localize_th232_correlated_chi() {
    if !data_present("Th232") {
        eprintln!("skipping localize_th232_correlated_chi -- endf-b8.1-Th232.arrow cache absent");
        return;
    }
    let n = 100_000;
    let rep = run_case("Th232 correlated chi", "Th232", 11.7, FAST_ENERGY, n);
    assert!(
        rep.both > n / 4,
        "expected substantial collisions ({} / {n})",
        rep.both
    );
}

/// High-statistics collision-0 spectrum parity for F19: drives many histories
/// at a dense sphere (so most have a first collision at exactly 14.06 MeV) and
/// compares the CPU vs twin per-MT outgoing-energy means. The MT(s) flagged
/// `SAMPLER GAP` are the biased flat-sampler law(s) behind the V&V residual.
/// Report-only localizer.
#[test]
fn parity_f19_collision0_spectrum() {
    if !data_present("F19") {
        eprintln!("skipping parity_f19_collision0_spectrum -- endf-b8.1-F19.arrow cache absent");
        return;
    }
    let n = 100_000;
    let rep = run_case("F19 collision-0 spectrum", "F19", 5.0, FAST_ENERGY, n);
    assert!(
        rep.both > n / 4,
        "expected substantial collisions ({} / {n})",
        rep.both
    );
}

/// W184 with the source inside its URR band (1e4 .. 1e5 eV), i.e. every
/// collision samples the probability table (issue #111 gap 1).
///
/// URR used to be the last whole branch of the collision loop off the shared
/// stream: `smooth_flight` bailed on `has_urr_in_range`, the band came from
/// `FastRng`, and the analog split was gated off. Against the twin that read
/// **1.23%** bit-identical, with 98.78% of histories diverging at collision 0
/// -- while the aggregate spectra agreed to -0.00%, the signature of a pure
/// stream desync rather than a physics gap.
///
/// With the band, the flight and the nuclide selection on the shared PCG
/// stream in the kernel's order, it is bit-identical end to end. Asserted
/// tightly precisely because that is what the unification buys: unlike the
/// 14 MeV case, nothing here is left to arithmetic association.
#[test]
fn localize_w184_urr_band() {
    if !data_present("W184") {
        eprintln!("skipping localize_w184_urr_band -- endf-b8.1-W184.arrow cache absent");
        return;
    }
    let n = 4000;
    let rep = run_case("W184 URR band (5e4 eV)", "W184", 19.3, 5.0e4, n);
    assert!(
        rep.both > n / 2,
        "expected most histories to collide in a 20 cm W184 sphere ({} / {n})",
        rep.both
    );
    assert_eq!(
        rep.identical, n,
        "URR histories must be bit-identical to the twin end to end: {} / {n}. \
         Is a URR collision back on the legacy FastRng stream (#111 gap 1)?",
        rep.identical
    );
}

/// Survival biasing on BOTH backends (issue #111 gap 2).
///
/// The CPU used to revert the flight, the nuclide selection and the reaction
/// split to `FastRng` whenever survival biasing was on, so a survival-biased
/// history ran a different draw schedule from the GPU entirely. All three are
/// on the shared stream now, and so is the weight-cutoff roulette draw.
///
/// B10 at thermal is the right fixture: a strong absorber with no fission, so
/// implicit capture fires at essentially every collision and the roulette
/// fires often, while the fissile scheme difference (#352) -- the one thing
/// still not unified under survival -- cannot intrude.
#[test]
fn localize_b10_survival_biasing() {
    if !data_present("B10") {
        eprintln!("skipping localize_b10_survival_biasing -- endf-b8.1-B10.arrow cache absent");
        return;
    }
    let n = 4000;
    let rep = run_case_with("B10 thermal, survival", "B10", 2.34, 0.025, n, true);
    let (both, fully_identical) = (rep.both, rep.identical);
    assert!(
        both > n / 2,
        "expected most histories to collide in a B10 absorber ({both} / {n})"
    );
    assert_eq!(
        fully_identical, n,
        "survival-biased histories must be bit-identical to the twin: \
         {fully_identical} / {n}. Is the flight, the nuclide selection, the \
         split or the roulette back on FastRng (#111 gap 2)?"
    );
}

/// U235 inside its OWN unresolved-resonance band (2250 .. 24999 eV), which is
/// where issue #154's fissile flux deficit comes from.
///
/// `localize_w184_urr_band` reads 100% bit-identical, so URR looked closed after
/// #351. It is not, and W184 structurally cannot show it: a nuclide's URR
/// probability tables optionally carry extra partial columns, recorded as the
/// `inelastic` / `absorption` indices in `urr.arrow`, and W184 has NEITHER
/// (both -1) while U235 has BOTH (inelastic 4, absorption 0). 159 of the 351
/// nuclides with URR tables in endf-b8.1 carry at least one, so the covered case
/// is the minority one.
///
/// It read ~49% bit-identical with `mean collisions / history` 13.36 against the
/// twin's 11.32, because the CPU lost ALL in-band capture: `urr_adjusted_reaction_xs`
/// subtracted fission from an absorption partial that already excluded it, which
/// clamps to zero whenever in-band fission exceeds capture, i.e. for every fissile
/// nuclide (#154). Fixing that takes this to ~82% strict / ~84% within 1e-9 and
/// closes the integral deficit; `urr_fissile_capture` is the direct guard.
///
/// The residual ~16% is a smaller in-band difference that is still open, so this
/// case stays report-only.
#[test]
fn localize_u235_urr_band() {
    if !data_present("U235") {
        eprintln!("skipping localize_u235_urr_band -- endf-b8.1-U235.arrow cache absent");
        return;
    }
    let n = 20_000;
    let rep = run_case("U235 URR band (1e4 eV)", "U235", 18.95, 1.0e4, n);
    assert!(
        rep.both > n / 2,
        "expected most histories to collide in a dense U235 sphere ({} / {n})",
        rep.both
    );
}

/// The issue-#154 model itself: a dense U235 sphere at 1 MeV, where the fission
/// chain actually runs (5.94 collisions per history against 0.08 for the thin
/// `localize_u235_14mev` fixture, which is why that one reads 100% and this one
/// does not).
///
/// Every other link in the CPU -> twin -> kernel -> dispatch chain is now exact
/// (`gpu_twin_kernel_fissile_parity`), so the ~0.3% of histories diverging here
/// ARE the -0.2% integral deficit: they slow down through the URR band and pick a
/// different reaction there. Report-only; see [`localize_u235_urr_band`].
#[test]
fn localize_u235_dense_chain() {
    if !data_present("U235") {
        eprintln!("skipping localize_u235_dense_chain -- endf-b8.1-U235.arrow cache absent");
        return;
    }
    let n = 100_000;
    let rep = run_case("U235 dense 1 MeV (chain)", "U235", 18.95, 1.0e6, n);
    assert!(
        rep.both > n / 2,
        "expected most histories to collide ({} / {n})",
        rep.both
    );
}
