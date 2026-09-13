//! Issue #40 (sub-step 4 of #111): matched-stream per-history CPU-vs-GPU
//! bit-identity diff harness.
//!
//! `#111` routed the production CPU collision path onto the GPU's shared
//! PCG-32 stream and draw schedule (free-flight `xi1`, nuclide-select `xi_n`,
//! reaction-type split `xi2`, elastic `xi3`+mu, fission chi, discrete inelastic
//! levels, continuum / tabulated inelastic). With a per-particle seed of
//! `history_seed(base_seed, i)` -- one shared definition, called by both
//! backends -- and a deterministic monodirectional source (every history born
//! at the same energy / position / direction, zero RNG draws at birth),
//! the production CPU and the GPU's exact CPU twin
//! (`run_multi_cell_transport_cpu`) should consume the *same* stream and so
//! produce *bit-identical* collisions, up to the first point where the two
//! draw schedules legitimately diverge.
//!
//! This harness measures exactly that. For N single-batch histories through a
//! single-nuclide Fe56 sphere it records, per history, the sequence of
//! collisions `(energy_in, energy_out, reaction_class)` from BOTH:
//!   * the production CPU, via `Model::run_with_tracking` (the real transport
//!     path, recording one `TrackEvent` per collision); and
//!   * the GPU twin, via `run_multi_cell_transport_cpu(.., capture_trace=true)`
//!     translated from the SAME model with the SAME seeds.
//!
//! It then diffs the two sequences and reports the fraction bit-identical and a
//! histogram of the first interaction where they diverge.
//!
//! ## Scope (what matches, what still diverges)
//! As of #136 the lab azimuth is unified: both the production CPU and the GPU
//! draw `phi = TAU * next_xi` (one PCG draw, replacing the GPU's old Marsaglia
//! rejection of 2..16 draws). So an elastic-only history now matches bit-for-bit
//! END-TO-END -- the whole per-collision energy sequence, not just the first
//! collision (before #136 the azimuth draw-count mismatch desynced the stream
//! after collision 0, capping the match at ~61%). At 14 MeV the continuum /
//! tabulated inelastic laws are on the shared flat samplers too, and the CPU's
//! per-MT constituent WALK now visits candidates in the GPU's `MT_SLOTS` order
//! (`yamc_nuclide::nuclide::INELASTIC_MT_SLOTS`), so the same `xi_mt` selects
//! the same MT on both sides. Both of those are asserted to be gone. So is the
//! (n,xn) / (n,n'x) arm (MT 5 / 16 / 17 / 22 / 28 / ...), which `#111` moved off
//! the legacy `scatter_other` FastRng sampler onto the same shared flat tables
//! and PCG stream, including the analog (n,xn) secondaries: those are sampled
//! independently on both backends, in the same draw order, and the CPU banks
//! them into the same history the twin queues them in, so a `(n,2n)` history
//! stays in lockstep past the multiplying collision. The overall 14 MeV rate
//! stays reported, not asserted.
//! The free-gas azimuth (#111 site D) is unified too: the GPU draws the lab
//! azimuth unconditionally (even when the free-gas vector CM transform already
//! set the direction, for warp coherence) and skips only the rotation, and the
//! production CPU now makes that same draw in the same place. The third case
//! below -- a 1 eV source in a deuterium moderator, an order of magnitude below
//! `400*kT`, so EVERY elastic collision is free-gas -- asserts it; the two Fe56
//! cases sit far above the free-gas threshold and never exercised it.
//!
//! ## Reading the reported rates
//! The rates below are measured on ONE stream, the one `SEED` selects, and are
//! sample statistics of it. Since #315 the base seed reaches the per-history
//! collision PCG (`history_seed(base_seed, i)`, the shared definition both
//! backends call), so the spread can be measured: over 12 base seeds the 14 MeV
//! whole-history rate is 93.4% +/- 0.5 pp (min 92.7%, max 94.3%) and the two
//! 100.000% cases stay at 100.000% (one seed of the 12 gave 3999/4000 on the
//! thermal case). A move of a few tenths of a point between two runs of this
//! harness with different `SEED`s is therefore expected; a stream or sampler
//! divergence is not subtle, it collapses the collision-0 rate.
//!
//! The twin is pure CPU and bit-identical to the cubecl kernel (the
//! `gpu_cpu_comparison_matrix` `cpu_gpu_equivalence_*` tests pin twin == kernel),
//! so no physical GPU is needed; the harness only requires the `gpu` feature
//! (for the translation + twin) and the `tests/Fe56.arrow` data.
//!
//! Run it:
//!   cargo test -p yamc --features gpu --release \
//!       --test matched_stream_diff -- --nocapture

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
    SurvivalBiasingInputs, PEND_SLOTS,
};
use yamc_materials::Material;
use yamc_rng::history_seed;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const MAX_STEPS: u32 = 5_000;
const SEED: u64 = 4242;
/// Below Fe56's first inelastic level (MT 51, ~0.847 MeV), so every collision
/// is elastic (MT 2) or capture (MT 102) -- the clean elastic-only sanity case.
/// Also well above `400*kT` (~10 eV at 294 K), so free-gas never fires.
const ELASTIC_ONLY_ENERGY: f64 = 0.5e6;
/// A fast source that reaches continuum inelastic / (n,2n): expected to diverge
/// earlier (reported, not asserted).
const FAST_ENERGY: f64 = 14.06e6;
/// Epithermal source, an order of magnitude BELOW `400*kT` (~10.1 eV at 294 K),
/// so every elastic collision takes the free-gas branch. This is the case that
/// exercises the free-gas azimuth draw schedule (#111 site D).
const THERMAL_ENERGY: f64 = 1.0;
/// How many `(cpu MT, twin MT)` mismatch pairs to print, most frequent first.
const MISMATCH_PAIRS_SHOWN: usize = 8;

/// A single-nuclide sphere fixture: which nuclide, its cached `.arrow` data,
/// the mass density, and the sphere radius (sized so a history collides a few
/// times before leaking).
struct SphereCase {
    nuclide: &'static str,
    data: &'static str,
    density_g_cm3: f64,
    radius: f64,
}

/// Fe56 at solid-iron density in a 5 cm sphere: ~1 cm mfp at MeV energies.
const FE56: SphereCase = SphereCase {
    nuclide: "Fe56",
    data: "tests/Fe56.arrow",
    density_g_cm3: 7.874,
    radius: 5.0,
};
/// Deuterium moderator: a light nuclide whose elastic mfp near 1 eV is ~10 cm,
/// so a 30 cm sphere gives a multi-collision moderating random walk with EVERY
/// elastic collision on the free-gas branch.
const H2: SphereCase = SphereCase {
    nuclide: "H2",
    data: "tests/H2.arrow",
    density_g_cm3: 1.0,
    radius: 30.0,
};

fn data_present(case: &SphereCase) -> bool {
    std::path::Path::new(case.data).exists()
}

/// Single-nuclide sphere (vacuum boundary), neutron data only. `Below(sphere)`
/// keeps it a single bounded cell, matching `gpu_cpu_comparison_matrix`.
fn sphere(case: &SphereCase) -> Geometry {
    let surface = Surface {
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
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(surface)));

    let mut material = Material::new(
        HashMap::from([(case.nuclide.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(case.density_g_cm3),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(case.nuclide.to_string(), case.data.to_string())]);
    material.read_nuclear_data(&nm, None).unwrap();

    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

/// Monodirectional (+z) point source at the origin, single discrete energy.
/// Point + Discrete + Monodirectional all sample deterministically with ZERO
/// RNG draws, so every history is born identical and the per-particle PCG
/// stream starts fresh at the first free flight on both backends.
fn build_model(case: &SphereCase, energy_ev: f64) -> Model {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::new_monodirectional(0.0, 0.0, 1.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![energy_ev], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(sphere(case), vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.gpu_max_steps_per_particle = MAX_STEPS;
    model.tracking_mode = TrackingMode::Surface;
    // Analog (no survival biasing), no secondary photons -- both default off.
    model
}

/// One collision, normalized across the two backends. `class` collapses the
/// ENDF MT to a reaction class so the absorbing collision (whose specific
/// capture / charged-particle MT the twin does not resolve) still aligns.
/// `mt` keeps the resolved ENDF MT (see [`twin_mt`]) so a divergence can be
/// attributed to the MT SELECTION versus the post-selection KINEMATICS; it is
/// diagnostic only and deliberately not part of [`col_eq`].
#[derive(Clone, Copy)]
struct Col {
    e_in: f64,
    e_out: f64,
    class: u8,
    mt: i32,
}

/// Reaction class: 0 elastic, 1 inelastic/scatter-with-multiplicity, 2 fission,
/// 3 absorption (capture / charged-particle).
fn class_of(mt: i32) -> u8 {
    match mt {
        2 => 0,
        18 => 2,
        16 | 17 => 1, // (n,2n), (n,3n)
        51..=93 => 1, // discrete + continuum inelastic; twin slots 41/42 -> 92/93
        _ => 3,       // 102 capture, 103/104/107 charged-particle, ...
    }
}

/// ENDF MT behind a twin `CollisionRecord::reaction`. The twin records the
/// inelastic branch as `51 + slot`, which is the MT itself only for the
/// leading MT 51..=91 run; slots 41.. carry MT 16 / 17 / 22 / ... Resolve
/// through the slot table so both backends' MTs are comparable.
fn twin_mt(reaction: i32) -> i32 {
    use yamc_gpu::neutron::xs::MT_SLOTS;
    let slot = reaction - 51;
    if (0..MT_SLOTS.len() as i32).contains(&slot) {
        MT_SLOTS[slot as usize]
    } else {
        reaction // elastic (2), fission (18), absorption (102)
    }
}

fn col_eq(a: &Col, b: &Col) -> bool {
    a.e_in.to_bits() == b.e_in.to_bits()
        && a.e_out.to_bits() == b.e_out.to_bits()
        && a.class == b.class
}

/// Ulp headroom for [`col_diff_is_physical`]. Generous enough to absorb the
/// last-place wobble of a handful of chained multiplies / divides, and still
/// ~14 orders of magnitude below any real sampler divergence (a different
/// `xi`, level or law moves the outgoing energy by a fraction of ITSELF, not by
/// its last bit).
const KINEMATICS_ULP_TOLERANCE: f64 = 8.0;

/// Whether two collisions differ by more than floating-point association can
/// explain: a different reaction class, a different incoming energy, or an
/// outgoing energy apart by more than [`KINEMATICS_ULP_TOLERANCE`] ulp.
///
/// The CPU and the GPU twin evaluate the SAME closed-form kinematics but do not
/// group the arithmetic identically, so a sample can land one ulp apart. That
/// is not a stream or sampler divergence and it is not what the assertions
/// below are guarding; a genuine divergence is many orders of magnitude larger.
fn col_diff_is_physical(a: &Col, b: &Col) -> bool {
    if a.class != b.class || a.e_in.to_bits() != b.e_in.to_bits() {
        return true;
    }
    let scale = a.e_out.abs().max(b.e_out.abs());
    (a.e_out - b.e_out).abs() > KINEMATICS_ULP_TOLERANCE * f64::EPSILON * scale
}

/// Relative tolerance for the WHOLE-HISTORY comparison
/// ([`CaseReport::fully_identical_within_rounding`]).
///
/// Looser than [`KINEMATICS_ULP_TOLERANCE`] because a last-place difference at
/// one collision becomes the NEXT collision's incoming energy, so the wobble
/// compounds along a history instead of being reset. Still ~7 orders of
/// magnitude below any real divergence: a different `xi`, level or law moves
/// the outgoing energy by a fraction of ITSELF, not by its trailing bits.
const HISTORY_REL_TOLERANCE: f64 = 1e-9;

/// Two collisions agreeing to within [`HISTORY_REL_TOLERANCE`] on both
/// energies, with the same reaction class. Used for the "same history up to
/// arithmetic association" metric.
fn col_close(a: &Col, b: &Col) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= HISTORY_REL_TOLERANCE * x.abs().max(y.abs());
    a.class == b.class && close(a.e_in, b.e_in) && close(a.e_out, b.e_out)
}

/// Whether the two histories agree collision for collision to within
/// [`HISTORY_REL_TOLERANCE`]. This is the metric that separates a real stream
/// or sampler divergence (the two backends sampled DIFFERENT physics) from the
/// CPU and the twin grouping the same closed-form kinematics differently and
/// landing a few ulp apart. `fully_identical` is the strict bit-equal form of
/// the same question, and the gap between the two is exactly that association
/// noise.
fn same_history_within_rounding(cpu: &[Col], twin: &[Col]) -> bool {
    cpu.len() == twin.len() && cpu.iter().zip(twin).all(|(a, b)| col_close(a, b))
}

/// First collision index at which the two sequences differ. `None` when they
/// are identical in full (same length, every collision bit-equal). A length
/// mismatch with a matching common prefix diverges at the shorter length.
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

struct CaseReport {
    n: usize,
    both_have_collision: usize,
    collision0_match: usize,
    fully_identical: usize,
    /// Histories that agree collision for collision to within
    /// [`HISTORY_REL_TOLERANCE`] (see [`same_history_within_rounding`]).
    /// `fully_identical` demands bit equality, so anything counted here but not
    /// there is the two backends' arithmetic association, not a divergence.
    fully_identical_within_rounding: usize,
    diverge_at_first_collision: usize,
    diverge_after_first_collision: usize,
    elastic0_total: usize,
    elastic0_match: usize,
    max_first_div_index: usize,
    cpu_collisions_total: usize,
    twin_collisions_total: usize,
    /// Collision-0 mismatches where the two backends picked DIFFERENT MTs.
    /// Before the walk-order unification this was the per-MT cumulative walk
    /// visiting reactions in a different sequence on each side; it is now 0 on
    /// a nuclide whose scatter MTs are all in the slot table.
    collision0_diff_mt: usize,
    /// Collision-0 mismatches on the SAME MT: the two backends chose one
    /// reaction and still produced different kinematics.
    collision0_same_mt: usize,
    /// Subset of `collision0_same_mt` on an MT the production CPU routes
    /// through the SHARED samplers (`scatter_inelastic_level` /
    /// `scatter_inelastic_shared` for MT 51..=91, `scatter_other_shared` for
    /// the (n,xn) / (n,n'x) slots). This is the metric #111 sub-step 3 and the
    /// (n,xn) arm move drove to the last-place-rounding floor. Reported; the
    /// assertion is on `collision0_same_mt_shared_physical`.
    collision0_same_mt_shared: usize,
    /// Subset of `collision0_same_mt_shared` that floating-point association
    /// cannot explain (see [`col_diff_is_physical`]): a different class, a
    /// different incoming energy, or an outgoing energy more than a few ulp
    /// away. This is the real "an arm is off the shared sampler" signal and it
    /// is asserted to be exactly zero.
    collision0_same_mt_shared_physical: usize,
    /// Subset of `collision0_same_mt` on an MT the production CPU can still
    /// only sample with the legacy `scatter_other` FastRng path: a channel
    /// outside the GPU's `MT_SLOTS` table (MT 11 / 29 / 30 / 35 / 36 / 42 /
    /// 152..200), which the GPU cannot select at all. Reported, not asserted;
    /// 0 on a fixture whose scatter MTs are all in the table.
    collision0_same_mt_legacy: usize,
    /// `(cpu_mt, twin_mt)` histogram of the collision-0 mismatches, so a
    /// residual can be attributed without re-instrumenting the harness.
    collision0_mismatch_pairs: HashMap<(i32, i32), usize>,
    /// For each history whose first divergence is at collision `k >= 1`, the MT
    /// of the LAST matching collision (`k - 1`) -- the reaction whose draw
    /// schedule desynchronised the stream. A residual concentrated on one MT
    /// points straight at the arm still out of step, so the next site can be
    /// found without re-instrumenting the harness.
    predivergence_mt: HashMap<i32, usize>,
}

/// True when the production CPU routes this inelastic constituent through the
/// SHARED PCG samplers: `scatter_inelastic_level` for the closed-form-Q
/// discrete levels, `scatter_inelastic_shared` for the continuum / tabulated
/// laws (the `50..=91 | 875..=890` match arm in `yamc::transport`'s analog
/// collision block), and `scatter_other_shared` for the (n,xn) / (n,n'x)
/// channels MT 5 / 16 / 17 / 22 / 28 / 32 / 33 / 34 / ... (its `_` arm).
///
/// The `_` arm covers any non-elastic scatter MT, but only the ones in the
/// GPU's slot table are comparable: a channel outside it (MT 11 / 29 / 30 /
/// 35 / 36 / 42 / 152..200) cannot be selected by the GPU at all, so a
/// collision on one is counted as legacy rather than asserted here.
fn cpu_uses_shared_sampler(mt: i32) -> bool {
    use yamc_gpu::neutron::xs::MT_SLOTS;
    matches!(mt, 50..=91 | 875..=890) || MT_SLOTS.contains(&mt)
}

/// Run the GPU CPU-twin over a translated model, with per-collision tracing on.
///
/// The twin driver's argument list mirrors the kernel's buffer bindings
/// one-for-one, so it is threaded here once and shared by the per-case diff and
/// the drain-order proof below rather than spelled out at each call.
fn run_twin(
    inputs: &yamc::gpu::GpuTransportInputs,
    pack: &TalliesPack,
    pend_drain: PendDrain,
) -> (
    yamc_gpu::neutron::transport::MultiCellResult,
    Vec<Vec<CollisionRecord>>,
) {
    run_multi_cell_transport_cpu(
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
        pack,
        &[],
        &SurvivalBiasingInputs::off(),
        &inputs.nuclide_select,
        &FissionBankInputs::off(),
        MAX_STEPS,
        400.0,
        true,
        pend_drain,
    )
}

fn run_case(label: &str, case: &SphereCase, energy_ev: f64, n: usize) -> CaseReport {
    // --- Production CPU: per-history collision trace via track capture. ---
    let mut model = build_model(case, energy_ev);
    let settings = TransportSettings {
        total_particles: Some(n),
        seed: SEED,
        ..Default::default()
    };
    let inputs = translate_for_gpu(&model, n, settings.seed)
        .expect("translate single-nuclide model for GPU twin");

    // Matched-stream contract self-check: single batch => the GPU seed buffer
    // holds exactly the shared per-history seed the CPU transport loop derives
    // for the same global index, so both backends start history `i` on the same
    // 64-bit PCG state.
    for i in 0..n {
        assert_eq!(
            inputs.seeds[i],
            history_seed(SEED, i as u64),
            "GPU per-particle seed must equal history_seed(base_seed, i) for batch 0"
        );
    }

    let storage = model
        .run_with_tracking(&settings, HistorySelection::first(n as u64))
        .expect("CPU tracked run");
    let mut cpu: Vec<Vec<Col>> = vec![Vec::new(); n];
    for track in &storage.tracks {
        // Primary (source) walk only; (n,2n) secondaries are a separate
        // generation that the single-walk twin handles via weight, not banking.
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
                });
            }
        }
    }

    // --- GPU twin: same model, same seeds, per-collision trace. ---
    let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
    let (_twin_res, twin_traces) = run_twin(
        &inputs,
        &pack,
        // Drain the twin's in-thread (n,xn) queue LIFO, matching the production
        // CPU's `ParticleBank` stack, so the two backends emit a history's
        // collisions in the SAME sequence and the diff below stays a
        // measurement of the stream and the samplers rather than of the
        // scheduling. Since #111 phase 1 every secondary carries its own
        // identity-derived seed the order cannot change what a secondary
        // samples, which `secondary_drain_order_is_unobservable` proves by
        // running this same twin both ways. The kernel itself stays FIFO.
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
                    mt: twin_mt(r.reaction),
                })
                .collect()
        })
        .collect();

    // --- Diff + metrics. ---
    let mut rep = CaseReport {
        n,
        both_have_collision: 0,
        collision0_match: 0,
        fully_identical: 0,
        fully_identical_within_rounding: 0,
        diverge_at_first_collision: 0,
        diverge_after_first_collision: 0,
        elastic0_total: 0,
        elastic0_match: 0,
        max_first_div_index: 0,
        cpu_collisions_total: 0,
        twin_collisions_total: 0,
        collision0_diff_mt: 0,
        collision0_same_mt: 0,
        collision0_same_mt_shared: 0,
        collision0_same_mt_shared_physical: 0,
        collision0_same_mt_legacy: 0,
        collision0_mismatch_pairs: HashMap::new(),
        predivergence_mt: HashMap::new(),
    };
    let mut div_examples: Vec<String> = Vec::new();
    for i in 0..n {
        let c = &cpu[i];
        let t = &twin[i];
        rep.cpu_collisions_total += c.len();
        rep.twin_collisions_total += t.len();

        // Self-check: a populated first collision must start at the source
        // energy on BOTH sides (catches a misaligned / empty trace harness bug).
        if let Some(c0) = c.first() {
            assert_eq!(
                c0.e_in.to_bits(),
                energy_ev.to_bits(),
                "CPU history {i} first collision energy_in must be the source energy"
            );
        }
        if let Some(t0) = t.first() {
            assert_eq!(
                t0.e_in.to_bits(),
                energy_ev.to_bits(),
                "twin history {i} first collision energy_in must be the source energy"
            );
        }

        if !c.is_empty() && !t.is_empty() {
            rep.both_have_collision += 1;
            if col_eq(&c[0], &t[0]) {
                rep.collision0_match += 1;
            } else {
                // Attribute the mismatch: a different MT means the per-MT
                // constituent walks disagreed BEFORE any kinematics ran; the
                // same MT means the kinematics themselves diverged, split by
                // whether the CPU sampled that MT on the shared samplers.
                *rep.collision0_mismatch_pairs
                    .entry((c[0].mt, t[0].mt))
                    .or_insert(0) += 1;
                if c[0].mt == t[0].mt {
                    rep.collision0_same_mt += 1;
                    if cpu_uses_shared_sampler(c[0].mt) {
                        rep.collision0_same_mt_shared += 1;
                        if col_diff_is_physical(&c[0], &t[0]) {
                            rep.collision0_same_mt_shared_physical += 1;
                        }
                    } else {
                        rep.collision0_same_mt_legacy += 1;
                    }
                } else {
                    rep.collision0_diff_mt += 1;
                }
                if div_examples.len() < 8 {
                    div_examples.push(format!(
                        "  hist {i}: cpu(mt={} e_in={:.6e} e_out={:.6e} cls={}) twin(mt={} e_in={:.6e} e_out={:.6e} cls={})",
                        c[0].mt, c[0].e_in, c[0].e_out, c[0].class,
                        t[0].mt, t[0].e_in, t[0].e_out, t[0].class
                    ));
                }
            }
            // Elastic-collision-0 subset: even at 14 MeV the elastic kinematics
            // are on the shared stream, so these must match bit-for-bit.
            if c[0].class == 0 {
                rep.elastic0_total += 1;
                if col_eq(&c[0], &t[0]) {
                    rep.elastic0_match += 1;
                }
            }
        }

        if same_history_within_rounding(c, t) {
            rep.fully_identical_within_rounding += 1;
        }
        match first_divergence(c, t) {
            None => rep.fully_identical += 1,
            Some(0) => rep.diverge_at_first_collision += 1,
            Some(k) => {
                rep.diverge_after_first_collision += 1;
                rep.max_first_div_index = rep.max_first_div_index.max(k);
                // Both sides agreed through collision k-1, so its reaction is
                // where the two draw schedules parted.
                *rep.predivergence_mt.entry(c[k - 1].mt).or_insert(0) += 1;
            }
        }
    }

    let pct = |num: usize, den: usize| -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    };
    eprintln!("\n==== matched-stream diff: {label} ({energy_ev:.3e} eV, N={n}) ====");
    eprintln!(
        "histories with >=1 collision (both)  : {} / {n}",
        rep.both_have_collision
    );
    eprintln!(
        "collision-0 bit-identical            : {} / {} ({:.3}%)   [HEADLINE: flight+select+reaction+kinematics on shared stream]",
        rep.collision0_match,
        rep.both_have_collision,
        pct(rep.collision0_match, rep.both_have_collision)
    );
    eprintln!(
        "  of which elastic collision-0       : {} / {} ({:.3}%)",
        rep.elastic0_match,
        rep.elastic0_total,
        pct(rep.elastic0_match, rep.elastic0_total)
    );
    eprintln!(
        "  collision-0 mismatch, SAME MT      : {}   [kinematics divergence]",
        rep.collision0_same_mt
    );
    eprintln!(
        "    of which shared-sampler MT       : {}   [MT 51..=91 + the (n,xn) slots -- #111 drives this to the last-place floor]",
        rep.collision0_same_mt_shared
    );
    eprintln!(
        "      beyond last-place rounding     : {}   [ASSERTED 0: a real sampler / stream divergence]",
        rep.collision0_same_mt_shared_physical
    );
    eprintln!(
        "    of which legacy FastRng arm      : {}   [scatter MTs outside MT_SLOTS, unselectable on GPU]",
        rep.collision0_same_mt_legacy
    );
    eprintln!(
        "  collision-0 mismatch, DIFFERENT MT : {}   [per-MT constituent walk order -- unified in #111]",
        rep.collision0_diff_mt
    );
    eprintln!(
        "fully bit-identical histories        : {} / {n} ({:.3}%)   [whole-history stream lockstep]",
        rep.fully_identical,
        pct(rep.fully_identical, rep.n)
    );
    eprintln!(
        "  same up to last-place rounding     : {} / {n} ({:.3}%)   [ASSERTED: real stream/sampler lockstep; the gap to the line above is CPU-vs-twin arithmetic association]",
        rep.fully_identical_within_rounding,
        pct(rep.fully_identical_within_rounding, rep.n)
    );
    eprintln!(
        "first divergence AT collision 0      : {} ({:.3}%)   [should be ~0 for a clean elastic-only case]",
        rep.diverge_at_first_collision,
        pct(rep.diverge_at_first_collision, rep.n)
    );
    eprintln!(
        "first divergence AFTER collision 0   : {} ({:.3}%)   [residual mid-history desync; deepest at collision {}]",
        rep.diverge_after_first_collision,
        pct(rep.diverge_after_first_collision, rep.n),
        rep.max_first_div_index
    );
    eprintln!(
        "mean collisions / history            : cpu {:.2}, twin {:.2}",
        rep.cpu_collisions_total as f64 / n as f64,
        rep.twin_collisions_total as f64 / n as f64
    );
    if !div_examples.is_empty() {
        eprintln!("first {} collision-0 divergences:", div_examples.len());
        for e in &div_examples {
            eprintln!("{e}");
        }
    }
    if !rep.predivergence_mt.is_empty() {
        let mut mts: Vec<_> = rep.predivergence_mt.iter().collect();
        mts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        eprintln!("MT of the last MATCHING collision before the first divergence, top {MISMATCH_PAIRS_SHOWN}:");
        for (mt, count) in mts.into_iter().take(MISMATCH_PAIRS_SHOWN) {
            eprintln!("  cpu MT {mt:>3} : {count}");
        }
    }
    if !rep.collision0_mismatch_pairs.is_empty() {
        let mut pairs: Vec<_> = rep.collision0_mismatch_pairs.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        eprintln!("collision-0 mismatch (cpu MT, twin MT) histogram, top {MISMATCH_PAIRS_SHOWN}:");
        for ((cpu_mt, twin_mt), count) in pairs.into_iter().take(MISMATCH_PAIRS_SHOWN) {
            eprintln!("  cpu MT {cpu_mt:>3} vs twin MT {twin_mt:>3} : {count}");
        }
    }
    rep
}

/// SANITY + headline: below Fe56's inelastic threshold every collision is
/// elastic or capture, all on the shared PCG stream. The first collision must
/// be bit-identical between the production CPU and the GPU twin for ~100% of
/// histories, and since the lab azimuth is unified (#136) the whole history is
/// too. 0.5 MeV is also far above the free-gas threshold on Fe56, so this case
/// never enters the free-gas branch and is insensitive to #111 site D -- the
/// thermal case below is what covers that.
#[test]
fn elastic_only_first_collision_is_bit_identical() {
    if !data_present(&FE56) {
        eprintln!("skipping matched_stream_diff -- tests/Fe56.arrow absent");
        return;
    }
    let n = 4000;
    let rep = run_case("Fe56 elastic-only", &FE56, ELASTIC_ONLY_ENERGY, n);

    // The sim must be non-trivial: a substantial fraction of histories collide
    // (a monodirectional beam in a 5 cm Fe sphere collides ~half the time).
    assert!(
        rep.both_have_collision > n / 4,
        "expected a substantial fraction of histories to collide ({} / {n})",
        rep.both_have_collision
    );
    // HEADLINE: first collision bit-identical for ~all histories. A failure
    // here is a genuine CPU-vs-GPU divergence in flight / reaction-split /
    // elastic kinematics on the shared stream (or a harness bug) -- not noise.
    let frac = rep.collision0_match as f64 / rep.both_have_collision as f64;
    assert!(
        frac >= 0.999,
        "elastic-only collision-0 bit-identity {:.4} < 0.999 ({} / {})",
        frac,
        rep.collision0_match,
        rep.both_have_collision
    );
    // With the lab azimuth unified (#136 / #111), the production CPU and the
    // GPU twin consume the IDENTICAL PCG stream for the whole elastic history,
    // so the per-collision energy sequences are bit-identical end-to-end (not
    // just at collision 0). Before #136 this was ~61% (the Marsaglia-vs-`TAU*xi`
    // azimuth draw-count mismatch desynced the stream after the first
    // collision); it is now ~100%.
    let full = rep.fully_identical as f64 / rep.n as f64;
    assert!(
        full >= 0.99,
        "elastic-only fully bit-identical {:.4} < 0.99 ({} / {}) -- azimuth unification (#136) regressed?",
        full,
        rep.fully_identical,
        rep.n
    );
    // The divergence (if any) must not be AT collision 0 (which would be a real
    // physics bug, not the few near-pole rotation-branch ulp flips allowed above).
    assert!(
        rep.diverge_at_first_collision <= rep.n / 1000,
        "too many first-collision divergences ({}) -- investigate a real CPU/GPU split",
        rep.diverge_at_first_collision
    );
}

/// REPORT (plus three hard invariants): a 14 MeV source reaches continuum
/// inelastic and `(n,2n)`.
///
/// Asserted: (1) elastic first-collisions still match exactly; (2) NO
/// collision-0 mismatch survives where both backends selected the same MT AND
/// the CPU sampled it on the shared samplers -- the discrete levels, the
/// continuum / tabulated laws (#111 sub-step 3) and the (n,xn) / (n,n'x)
/// channels (the `scatter_other_shared` move) all read the same flat tables off
/// the PCG stream; and (3) NO collision-0 mismatch survives from the two
/// backends selecting DIFFERENT MTs -- what the walk-order unification bought,
/// since the CPU now visits the non-elastic candidates in the GPU's `MT_SLOTS`
/// order, so a shared `xi_mt` picks the same reaction on both sides.
///
/// What is still reported and not asserted is the STRICT whole-history rate.
/// It is not a stream or sampler divergence: every history matches within
/// `HISTORY_REL_TOLERANCE` (4000 / 4000), so the gap is the production CPU and
/// the twin grouping the same closed-form kinematics differently and landing a
/// few ulp apart, which then compounds along the history. Drain order was the
/// earlier explanation and is no longer one: #322 gave every secondary an
/// identity-derived stream, and `secondary_drain_order_is_unobservable` proves
/// FIFO and LIFO transport the same emission tree identically.
#[test]
fn fast_source_divergence_report() {
    if !data_present(&FE56) {
        eprintln!("skipping matched_stream_diff (fast) -- tests/Fe56.arrow absent");
        return;
    }
    let n = 4000;
    let rep = run_case("Fe56 14 MeV", &FE56, FAST_ENERGY, n);
    assert!(
        rep.both_have_collision > 0,
        "expected some histories to collide at 14 MeV"
    );
    // Elastic collision-0 is on the shared stream even at 14 MeV: bit-identical.
    if rep.elastic0_total > 0 {
        let frac = rep.elastic0_match as f64 / rep.elastic0_total as f64;
        assert!(
            frac >= 0.999,
            "elastic collision-0 bit-identity at 14 MeV {:.4} < 0.999 ({} / {})",
            frac,
            rep.elastic0_match,
            rep.elastic0_total
        );
    }
    // Same-MT kinematics parity on every MT the CPU samples with a shared
    // sampler, to the last place.
    //
    // Recalibrated three times, every time by TIGHTENING or WIDENING what it
    // covers, never by loosening the bound:
    //   1. from `collision0_same_mt` (all MTs) to the shared-sampler subset
    //      when the walk order was unified (before that, a history whose CPU and
    //      GPU walks disagreed on the MT counted as a DIFFERENT-MT mismatch and
    //      never reached this bound);
    //   2. back over the (n,xn) / (n,n'x) MTs once `scatter_other_shared` put
    //      them on the same flat tables and PCG stream, bringing 528
    //      previously-excluded collisions (455 MT 16 + 73 MT 5, of 2661) under
    //      the assertion at 0;
    //   3. from `count <= n / 1000` on the raw bit-difference to `== 0` on the
    //      subset that last-place rounding cannot explain (#315). The two
    //      backends evaluate the same closed-form discrete-level kinematics with
    //      a different grouping of multiplies, so a sample can land ONE ulp
    //      apart; how many do is pure luck of the stream. Measured over 12 base
    //      seeds (#315 made the base seed reach the collision stream, so this
    //      can now be sampled at all) the raw count is 3.6 +/- 2.4 with a
    //      maximum of 8, exceeding the old `n / 1000 == 4` bound on 5 of the 12:
    //      that bound passed only because the frozen stream happened to give 1.
    //      The count
    //      of differences beyond `KINEMATICS_ULP_TOLERANCE` ulp is 0 on every
    //      one of those 12 seeds, and a genuine sampler or stream divergence is
    //      ~1e14 ulp, so `== 0` is a strictly stronger statement than the bound
    //      it replaces.
    assert_eq!(
        rep.collision0_same_mt_shared_physical,
        0,
        "collision-0 kinematics diverged beyond last-place rounding on {} SAME-MT \
         shared-sampler collisions (of {} raw same-MT shared mismatches, {} total \
         mismatches, {} on MTs the GPU cannot select) -- an inelastic or (n,xn) arm is \
         off the shared sampler, or an argument mismatch versus the GPU twin crept in",
        rep.collision0_same_mt_shared_physical,
        rep.collision0_same_mt_shared,
        rep.collision0_same_mt + rep.collision0_diff_mt,
        rep.collision0_same_mt_legacy
    );
    // Walk-order unification (#111): the CPU visits the non-elastic scatter
    // candidates in the GPU's `MT_SLOTS` order, so a shared `xi_mt` selects the
    // same MT on both backends. Fe56's scatter MTs (2, 5, 16, 51..=89, 91) are
    // all in the slot table, so the two candidate sets and normalizations match
    // exactly and there is no residual selection mismatch. Was 1141 / 2661
    // before the reorder.
    assert!(
        rep.collision0_diff_mt <= rep.n / 1000,
        "{} collision-0 mismatches selected DIFFERENT MTs -- the CPU's non-elastic \
         cumulative walk no longer visits candidates in `INELASTIC_MT_SLOTS` order, \
         or a scatter MT outside that table crept into the fixture",
        rep.collision0_diff_mt
    );
    // WHOLE-HISTORY lockstep, the metric the `fully bit-identical` line above
    // under-reports. That line demands bit equality, and at 14 MeV a continuum
    // / discrete-level sample can land a couple of ulp apart because the CPU
    // and the twin group the same closed-form kinematics differently; the
    // difference then becomes the next collision's incoming energy and the
    // history is written off from there. Every single one of the ~280 histories
    // it drops is that: to within `HISTORY_REL_TOLERANCE` all 4000 histories
    // agree collision for collision, so the two backends are on the same stream
    // end-to-end and no reaction arm is off the shared sampler.
    //
    // This is also why the `(n,xn)` entries in the pre-divergence histogram are
    // NOT an ordering artefact and did not move when issue #111 phase 1 gave
    // every secondary its own identity-derived stream: an Fe56 history holds at
    // most one secondary in the queue at a time (its (n,2n) threshold is ~11
    // MeV, so a walk multiplies once and then drops below it), which makes FIFO
    // and LIFO the same drain. `secondary_drain_order_is_unobservable` uses Be9
    // for the ordering claim precisely because Fe56 cannot exercise it.
    assert_eq!(
        rep.fully_identical_within_rounding,
        rep.n,
        "{} / {} histories diverged beyond last-place rounding -- a real stream or \
         sampler divergence, not the CPU-vs-twin arithmetic association the \
         bit-equal count absorbs",
        rep.n - rep.fully_identical_within_rounding,
        rep.n
    );
}

/// FREE-GAS case (#111 site D). A 1 eV source in a deuterium moderator sits an
/// order of magnitude below `400*kT` (~10.1 eV at 294 K), so EVERY elastic
/// collision takes the free-gas branch, where the vector CM transform writes the
/// outgoing direction directly. The GPU kernel and its twin still draw the lab
/// azimuth there (unconditionally, for warp coherence) and skip only the
/// rotation; the production CPU used to skip the DRAW too, so one fewer value
/// was consumed at every free-gas collision and the rest of the history
/// desynchronised. `scatter_elastic` now makes the same unconditional draw in
/// the same place, so a moderating random walk stays in lockstep end-to-end.
///
/// The other two cases cannot see this: 0.5 MeV and 14 MeV are both far above
/// the free-gas threshold on Fe56, so `did_run` is false there and the draw
/// schedule was already matched.
#[test]
fn thermal_free_gas_history_is_bit_identical() {
    if !data_present(&H2) {
        eprintln!("skipping matched_stream_diff (thermal) -- tests/H2.arrow absent");
        return;
    }
    let n = 4000;
    let rep = run_case("H2 moderator 1 eV (free-gas)", &H2, THERMAL_ENERGY, n);

    // Non-trivial: the walk must actually collide, repeatedly.
    assert!(
        rep.both_have_collision > n / 4,
        "expected a substantial fraction of histories to collide ({} / {n})",
        rep.both_have_collision
    );
    assert!(
        rep.cpu_collisions_total as f64 / n as f64 > 1.5,
        "expected a multi-collision moderating walk, got {:.2} collisions / history",
        rep.cpu_collisions_total as f64 / n as f64
    );
    // Collision 0 was already matched before site D (the divergent draw is made
    // at the END of the collision, so it only affects what follows).
    let frac = rep.collision0_match as f64 / rep.both_have_collision as f64;
    assert!(
        frac >= 0.999,
        "free-gas collision-0 bit-identity {:.4} < 0.999 ({} / {})",
        frac,
        rep.collision0_match,
        rep.both_have_collision
    );
    // HEADLINE for site D: whole free-gas histories are bit-identical. Before
    // the unconditional draw this was ~2% (essentially only the single-collision
    // histories); it is now ~100%.
    let full = rep.fully_identical as f64 / rep.n as f64;
    assert!(
        full >= 0.99,
        "free-gas fully bit-identical {:.4} < 0.99 ({} / {}) -- the CPU free-gas path \
         no longer consumes the lab-azimuth draw the GPU consumes (#111 site D)",
        full,
        rep.fully_identical,
        rep.n
    );
}

/// Beryllium: `(n,2n)` has a 1.85 MeV threshold and a large cross-section, so a
/// 14 MeV history multiplies REPEATEDLY and holds several secondaries in the
/// queue at once. That is what makes it, unlike Fe56 (whose (n,2n) threshold is
/// ~11 MeV, so a walk multiplies once and then drops below it), a fixture where
/// FIFO and LIFO actually transport the queue in different orders.
const BE9: SphereCase = SphereCase {
    nuclide: "Be9",
    data: "tests/Be9.arrow",
    density_g_cm3: 1.85,
    // ~2 mean free paths at 14 MeV. Deep enough that a walk often multiplies
    // twice and so holds more than one secondary in the queue at once (that is
    // what makes FIFO and LIFO visit the history differently: 7% of these
    // histories are genuinely reordered).
    radius: 10.0,
};

/// A thick Be9 sphere: ~8 mean free paths, so a 14 MeV history multiplies deep
/// into the (n,2n) chain before it leaks. The most demanding fixture available
/// for the kernel's pending-secondary stack depth.
const BE9_THICK: SphereCase = SphereCase {
    nuclide: "Be9",
    data: "tests/Be9.arrow",
    density_g_cm3: 1.85,
    radius: 40.0,
};

/// Lead: (n,2n) at ~7.4 MeV and (n,3n) just at the source energy, on a heavy
/// nucleus that barely slows the neutrons down, so the multiplying chain is
/// limited by reaction thresholds rather than by moderation.
const PB208: SphereCase = SphereCase {
    nuclide: "Pb208",
    data: "tests/Pb208.arrow",
    density_g_cm3: 11.35,
    radius: 30.0,
};

/// Tungsten: (n,2n) opens near 7.4 MeV with a cross-section above 2 b at
/// 14 MeV, on the heaviest nucleus in the fixture set, so the chain is again
/// threshold-limited rather than moderation-limited. About five mean free paths.
const W184: SphereCase = SphereCase {
    nuclide: "W184",
    data: "tests/W184.arrow",
    density_g_cm3: 19.3,
    radius: 15.0,
};

/// Chromium, standing in for the steel constituents: (n,2n) opens near 12 MeV,
/// so at 14 MeV only the first collision can multiply and the depth needed is
/// the smallest of the set. About five mean free paths.
const CR52: SphereCase = SphereCase {
    nuclide: "Cr52",
    data: "tests/Cr52.arrow",
    density_g_cm3: 7.19,
    radius: 15.0,
};

/// Issue #111 phase 2: how deep the kernel's thread-private (n,xn) stack has to
/// be, measured rather than assumed.
///
/// The kernel keeps [`PEND_SLOTS`] secondaries in registers and spills anything
/// deeper to the device particle bank for the host to drain in a later pass.
/// The spill is exact (the secondary carries its phase-1 identity seed either
/// way) but it costs an extra launch, so the depth is only worth having if
/// spilling is rare. This runs the twin -- whose stack is unbounded, and which
/// counts every push made while `PEND_SLOTS` were already outstanding, i.e.
/// exactly what the kernel would have spilled -- over the three most strongly
/// multiplying fixtures available and prints the rate.
///
/// It asserts only that the rate stays low enough for the spill to be a cold
/// path (below one in a thousand histories). The reason it is small is
/// structural, not tuning: the stack RECLAIMS slots on pop, so its depth is the
/// emission tree's DFS depth, and every (n,xn) is endothermic and divides what
/// is left of the incident energy between its products, so from 14 MeV the
/// chain runs out of energy against the reaction threshold after a few levels.
///
/// It also prints each fixture's histogram of peak pending depth per history,
/// because the rate alone cannot say whether a material that spills wants a
/// deeper in-thread stack or the bank drain: a tail that stops one slot past
/// `PEND_SLOTS` is a stack-depth question, a long tail is a drain question
/// (fusion-neutronics/core#20). Six fixtures at 14 MeV: beryllium thick and
/// thin, lead with (n,3n) open, iron and chromium for steel, and tungsten.
#[test]
fn nxn_spill_depth_is_sufficient() {
    let cases = [
        ("Be9 thick (8 mfp)", BE9_THICK),
        ("Be9 (2 mfp)", BE9),
        ("Pb208 (n,2n)+(n,3n)", PB208),
        ("Fe56", FE56),
        ("Cr52", CR52),
        ("W184", W184),
    ];
    let n = 4000;
    for (label, case) in cases {
        if !data_present(&case) {
            eprintln!("skipping {label} -- {} absent", case.data);
            continue;
        }
        let model = build_model(&case, FAST_ENERGY);
        let inputs = translate_for_gpu(&model, n, SEED).expect("translate for GPU twin");
        let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
        let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);
        let (res, _) = run_twin(&inputs, &pack, PendDrain::Lifo);
        let spilled = res.n_spilled_secondaries;
        let hist: Vec<String> = res
            .pend_depth_hist
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(depth, count)| format!("{depth}:{count}"))
            .collect();
        println!(
            "  {label:22}: deepest stack {} of {PEND_SLOTS} slots; {spilled} of {n} \
             histories would spill to the device bank ({:.4}%); peak depth histogram \
             [{}]",
            res.max_pend_depth,
            100.0 * spilled as f64 / n as f64,
            hist.join(" "),
        );
        assert_eq!(
            res.pend_depth_hist.iter().sum::<u64>(),
            n as u64,
            "{label}: the depth histogram does not account for every history"
        );
        assert!(
            spilled * 1000 <= n as u64,
            "{label}: {spilled} / {n} histories exceeded the {PEND_SLOTS}-deep \
             pending stack -- the spill is no longer a cold path, so either \
             deepen the stack or accept the extra drain launches"
        );
    }
}

/// One drain-order comparison. Returns `(compared, reordered, multiplying)`.
fn drain_order_case(case: &SphereCase, energy_ev: f64, n: usize) -> (usize, usize, usize) {
    let model = build_model(case, energy_ev);
    let inputs =
        translate_for_gpu(&model, n, SEED).expect("translate single-nuclide model for GPU twin");
    let edges = vec![(1.0e-5_f64).ln(), (20.0e6_f64).ln()];
    let pack = TalliesPack::flux_abs_pack((inputs.cell_aabbs.len() / 6) as u32, &edges);

    let (fifo_res, fifo_traces) = run_twin(&inputs, &pack, PendDrain::Fifo);
    let (lifo_res, lifo_traces) = run_twin(&inputs, &pack, PendDrain::Lifo);

    /// Order-insensitive key for one collision: the exact bits, so this stays a
    /// bit-identity claim and not an approximate one.
    fn key(r: &CollisionRecord) -> (u64, u64, i32) {
        (r.energy_in.to_bits(), r.energy_out.to_bits(), r.reaction)
    }

    let mut compared = 0usize;
    let mut reordered = 0usize;
    let mut multiplying = 0usize;
    for i in 0..n {
        let a: Vec<_> = fifo_traces[i].iter().map(key).collect();
        let b: Vec<_> = lifo_traces[i].iter().map(key).collect();
        // `reaction` is the twin's slot code; resolve it to the ENDF MT.
        if fifo_traces[i]
            .iter()
            .any(|r| matches!(twin_mt(r.reaction), 5 | 16 | 17 | 22 | 28 | 37))
        {
            multiplying += 1;
        }
        if a != b {
            // The two runs really did visit this history's walks in different
            // orders. These are the histories the test lives or dies on: a
            // fixture that produces none of them proves nothing.
            reordered += 1;
        }
        let mut sa = a;
        let mut sb = b;
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(
            sa, sb,
            "history {i} transported a different SET of collisions when the \
             pending-secondary queue was drained the other way -- a secondary's \
             stream still depends on when it was scheduled"
        );
        compared += 1;
    }

    // The tallies are the real end-to-end statement: every scored contribution
    // of every walk, summed in fixed point, identical bit for bit.
    assert_eq!(
        fifo_res.tally_outputs, lifo_res.tally_outputs,
        "the drain order changed the tallies"
    );

    println!(
        "  {:6} @ {:.2} MeV: {compared} / {n} histories identical under FIFO and LIFO; \
         {multiplying} had an (n,xn) collision, {reordered} were actually visited in a \
         different order",
        case.nuclide,
        energy_ev / 1.0e6,
    );
    (compared, reordered, multiplying)
}

/// Issue #111 phase 1, the proof that per-secondary identity seeding works: the
/// order the in-thread (n,xn) queue is drained in must not be observable.
///
/// Before phase 1 an in-history secondary simply CONTINUED whatever PCG state
/// the previous walk left behind, so which secondary went next decided what
/// each of them sampled. The CPU bank is a LIFO stack and the GPU's in-thread
/// queue is a FIFO, so the two backends made different physics out of the same
/// multiplying collision. Every secondary now carries a seed derived from its
/// place in the history's emission tree (`secondary_seed(parent seed, ordinal
/// within that parent)`), and the walk that pops it re-seeds from that, so the
/// schedule drops out of the answer.
///
/// This runs the same 4000 Fe56 histories at 14.06 MeV through the twin twice,
/// draining [`PendDrain::Fifo`] and then [`PendDrain::Lifo`], and demands:
///   * bit-identical tallies -- exact, because the accumulator is a fixed-point
///     integer sum, so a reordering cannot even round differently; and
///   * the same MULTISET of collisions per history (the sequence necessarily
///     differs, that is what "different order" means).
///
/// Nothing is excluded: since phase 2 the twin's queue is unbounded (the kernel
/// keeps the first `PEND_SLOTS` in registers and hands the rest to the device
/// bank, but either way every secondary is transported), so both drains see the
/// same SET and any difference between them would be a seeding failure.
#[test]
fn secondary_drain_order_is_unobservable() {
    if !data_present(&FE56) || !data_present(&BE9) {
        eprintln!("skipping drain-order proof -- tests/Fe56.arrow or tests/Be9.arrow absent");
        return;
    }
    let n = 4000;
    // Fe56 at 14.06 MeV is the production-relevant case: it is the fixture the
    // rest of this harness measures, and (n,2n) fires in a sixth of its
    // histories. It multiplies at most once per walk, though, so it produces
    // few genuine reorderings on its own.
    let (fe_compared, _fe_reordered, fe_multiplying) = drain_order_case(&FE56, FAST_ENERGY, n);
    // Be9 is the discriminating fixture: a low (n,2n) threshold and a large
    // cross-section mean a single walk multiplies several times and the queue
    // holds more than one secondary, so FIFO and LIFO really do transport them
    // in different orders.
    let (be_compared, be_reordered, be_multiplying) = drain_order_case(&BE9, FAST_ENERGY, n);

    assert!(
        fe_multiplying > n / 100,
        "Fe56 fixture queued (n,xn) secondaries in only {fe_multiplying} / {n} histories"
    );
    // Without this the assertions above are vacuous: the two drains would have
    // visited every history in the same order and compared a sequence with
    // itself.
    assert!(
        be_reordered > n / 100,
        "the Be9 fixture reordered only {be_reordered} / {n} histories, too few to \
         call the drain order exercised"
    );
    assert_eq!(
        fe_compared + be_compared,
        2 * n,
        "every history must be comparable: the twin's queue is unbounded, so \
         neither drain can drop a secondary"
    );
    let _ = be_multiplying;
}
