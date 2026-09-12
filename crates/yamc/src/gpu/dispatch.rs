//! Run a `Model` on the GPU.
//!
//! Wraps `translate_for_gpu` + `yamc_gpu::neutron::transport::run_multi_cell_transport`
//! so the Python `simulate_transport(compute='gpu')` path is a single call
//! into yamc's Rust core.
//!
//! ## Tally writeback
//!
//! The kernel takes a `TalliesPack` of N tallies and produces N parallel
//! output arrays. For each tally on the model, the dispatch builds one
//! pack entry -- its score kind (`Flux` / `Total` / `Absorption`), its
//! per-cell bin map (`CellFilter` and/or `MaterialFilter` → bin index,
//! `NOT_IN_TALLY` for cells neither filter covers), and its energy bin
//! edges (from `EnergyFilter`, or a single `[-∞, +∞]` bin if the tally
//! is cell-only). After the launch the dispatch walks every (tally,
//! spatial-bin, energy-bin) triple and stores the value into the
//! corresponding tally accumulator slot, then runs `accumulate_batch`
//! to fold it into the sum/sum² stream the Python layer reads.
//!
//! An `EnergyFunctionFilter` (`energy_function=` / `dose_coefficients=`)
//! rides along as a per-tally table rather than a bin dimension: the
//! kernel multiplies the score by the precomputed cubic spline
//! evaluated at the particle's energy, and drops the event outright
//! when that energy falls off the table.
//!
//! Models with no tally still run cleanly. Anything else (filters other
//! than `CellFilter` / `MaterialFilter` / `MeshFilter` / `EnergyFilter`
//! / `EnergyFunctionFilter` / `ParticleTypeFilter` /
//! `ParentNuclideFilter`, multi-score) is rejected up front with a
//! structured `GpuDispatchError`.
//!
//! ## macOS
//!
//! The kernel itself is `#[cfg(not(target_os = "macos"))]` in yamc-gpu
//! (cubecl's `spirv` feature can't build there without a Vulkan SDK).
//! On macOS `run_on_gpu` short-circuits with `GpuUnavailable` before
//! reaching any kernel call -- `GpuContext::new()` always returns
//! `NoF64Adapter` there.

// On macOS the GPU kernel modules (`yamc_gpu::neutron::transport` /
// `photon::transport`) are gated out, so the per-history dispatch entry points
// are stubbed and every kernel-side helper below (tally folding, launch
// chunking, bank draining, etc.) is compiled but unreachable. Allow that dead
// code on macOS only; non-macOS builds still lint dead code normally.
#![cfg_attr(target_os = "macos", allow(dead_code))]

use std::sync::Arc;
// Only the batch-free `LaunchLoop` (below) uses this, and it is compiled out on
// macOS (the GPU transport paths are stubbed there), so gate the import too.
#[cfg(not(target_os = "macos"))]
use std::time::Instant;

use yamc_gpu::common::tallies::TallyVarianceMode;
use yamc_gpu::common::tallies::{
    efunc_table_len, per_mt_fixed_point_scales, TalliesPack, DEFAULT_FIXED_POINT_SCALE,
    KERMA_FIXED_POINT_SCALE, MESH_CYLINDRICAL, MESH_NONE, MESH_RECT_ROWMAJOR, NOT_IN_TALLY,
    SCORE_FLUX, SCORE_PER_MT, SCORE_TOTAL,
};
use yamc_gpu::{GpuContext, GpuInitError, GpuSelectError};

use super::error::GpuTranslateError;
use super::translate::translate_for_gpu;
use crate::geometry::backend::GeometryKind;
use crate::geometry::Geometry;
use crate::model::Model;
use yamc_particle::particle::ParticleType;
use yamc_tallies::filter::Filter;
use yamc_tallies::mt::Mt;
use yamc_tallies::tally::{Score, Tally};

/// Errors `run_on_gpu` can return.
///
/// Wraps the lower-level `GpuInitError` (no usable GPU on this host)
/// and `GpuTranslateError` (model uses something the kernel can't
/// represent) so callers see one error type.
#[derive(Debug)]
pub enum GpuDispatchError {
    GpuUnavailable(GpuInitError),
    /// `compute='gpu <name>'` named an adapter not present on this host.
    /// Carries a message listing the valid adapter names. Distinct from
    /// `GpuUnavailable` so the Python layer can surface it as a `ValueError`
    /// (a user-input error) rather than a `RuntimeError`.
    AdapterNotFound(String),
    Translate(GpuTranslateError),
    /// Tally filters outside the supported set. Each tally must carry a
    /// spatial binner -- a `CellFilter`, a `MaterialFilter` and/or a
    /// `MeshFilter` -- with an optional `EnergyFilter`,
    /// `EnergyFunctionFilter` and/or `ParticleTypeFilter`, and nothing else.
    UnsupportedTallyFilters {
        tally_index: usize,
        found: Vec<&'static str>,
    },
    /// A `MeshFilter` in a configuration the GPU kernel does not yet score
    /// (issues #234, #279). Rectangular and cylindrical
    /// meshes are all supported; this now only rejects a mesh tally combined
    /// with the GPU fission bank, with an explicit message rather than a silent
    /// CPU-only fallback.
    UnsupportedMeshKind {
        tally_index: usize,
        reason: String,
    },
    /// Tally has more than one score, or a score the kernel can't
    /// estimate (only `Flux`, `ReactionRate(total)`, and
    /// `ReactionRate(absorption)` are supported).
    UnsupportedTallyScore {
        tally_index: usize,
        reason: String,
    },
    /// `model.variance_reduction` is non-empty. The kernel runs analog
    /// transport (fixed 1:1 particle map, no weight game); refusing
    /// keeps the requested technique honest instead of silently
    /// ignoring it.
    VarianceReductionUnsupported,
    /// Energy filter must define at least one bin (i.e. `bins.len() >=
    /// 2`). Lifted as of slice F: arbitrary bin edges (VITAMIN-J,
    /// EALF, custom) are passed straight through to the kernel.
    EnergyBinsTooFew {
        tally_index: usize,
    },
    /// Cell filter references a cell ID that doesn't exist in the
    /// model's geometry.
    CellNotInGeometry {
        tally_index: usize,
        cell_id: u32,
    },
    /// `Model::ensure_photon_data_for_gpu` failed -- a photon model's
    /// materials are missing photon data (`photon_data_paths` empty).
    PhotonDataPrep(String),
    /// `Model::ensure_neutron_temperatures_for_gpu` failed -- a material is
    /// labelled with a temperature its nuclide data cannot be widened to (#481).
    NeutronDataPrep(String),
    /// The device particle bank overflowed (`count > capacity`) -- the coupled
    /// secondary-photon bank or the #78 fission-progeny bank (one shared bank).
    /// The particle_bank contract forbids silent drops, so this is a hard error
    /// -- raise the per-batch bank capacity (`COUPLED_PHOTONS_PER_NEUTRON` /
    /// `FISSION_PROGENY_PER_NEUTRON`) and rerun.
    PhotonBankOverflow {
        overflow: u64,
        count: u64,
        capacity: usize,
    },
    /// A mixed neutron+photon primary source combined with D1S decay photons
    /// (`use_decay_photons`). Secondary-photon production
    /// (`transport_secondary_photons`) IS supported on the mixed path: the
    /// neutron share runs through the coupled kernel and its secondary bank is
    /// transported alongside the primary photons. Only the decay-photon
    /// coupling on top of a mixed source is unimplemented.
    MixedSourceWithSecondariesUnsupported,
    /// A history on a coupled (neutron -> photon) or mixed-source pass produced
    /// more simultaneous (n,xn) secondaries than a thread's in-thread stack
    /// holds, so the kernel handed them to the device particle bank (issue #111
    /// phase 2). The neutron-only paths drain banked neutrons; the coupled and
    /// mixed passes drain only banked PHOTONS, so continuing would silently drop
    /// a real neutron. Refused instead.
    ///
    /// This has never been observed: the stack is deep enough for every fixture
    /// measured (see `nxn_spill_depth_is_sufficient`, zero spills in 8e5
    /// histories of an 8 mean-free-path Be9 sphere at 14 MeV). It exists so a
    /// material that does exceed it fails loudly rather than quietly biasing the
    /// result low.
    CoupledNxnSpillUnsupported {
        spilled: u64,
    },
    /// A virtual-overlay tally (`response=`, i.e. `multiply_density == false`).
    /// The kernel scores the CELL material's macroscopic XS; the overlay regime
    /// scores a different material's response across the whole geometry, void
    /// included. The kernel has no machinery for it, and it used to look like an
    /// ordinary tally to the validator, so the GPU silently returned the plain
    /// cell-material score (issue #288: 11.8x off for a nuclide response).
    OverlayTallyUnsupported {
        tally_index: usize,
    },
    /// An uncapped run (`total_particles = None`) with no `max_runtime`. The
    /// GPU launch loop has no other stop condition, so it would launch forever.
    /// Rejected up front. The Python layer's own "at least one stop condition"
    /// guard fires first, so from Python this is unreachable.
    UncappedWithoutRuntime,
    /// `Model::convergence_targets` is non-empty. The GPU launch loop stops on
    /// `total_particles` and `max_runtime` only: it cannot evaluate a precision
    /// target between launches, because the per-history aggregate moments the
    /// targets are defined on are not read back from the device yet
    /// (fusion-neutronics/core#29). A run carrying targets plus a particle cap
    /// used to run silently to the cap, ignoring the precision the user asked
    /// to stop at (fusion-neutronics/core#23). Refused instead, whatever else
    /// is set, so the request is never quietly dropped.
    ConvergenceTargetsUnsupported {
        n_targets: usize,
    },
    /// Histories in a launch were still transporting when they hit
    /// `Model::gpu_max_steps_per_particle`. Their remaining track length was never
    /// scored, so every tally they touched is under-counted, by an amount
    /// nothing downstream can correct. This used to be a stderr warning gated
    /// on `verbose.summary`, so a `verbose=[]` run reported nothing and a
    /// truncated flux could pass for a converged one
    /// (fusion-neutronics/core#23). Now it fails the run at the first launch
    /// that truncates. The CPU never truncates (it runs every history to
    /// completion), so the remedy is a higher cap or the CPU.
    HistoriesTruncated {
        truncated: usize,
        launched: usize,
        max_steps: u32,
    },
    /// More than `Model::max_lost_particles` histories ended in no cell, i.e.
    /// the geometry does not cover the space particles reached (issue #289).
    /// The GPU twin of the CPU's `handle_lost_particle` abort: same cause, same
    /// remedy, checked at each launch boundary (so the reported count can
    /// overshoot the cap by up to one launch's losses, where the CPU stops at
    /// the exact particle). `records` carries the diagnostics, which
    /// `run_on_gpu_with_device` also copies onto `Model::lost_particles` so they
    /// stay inspectable after the failure.
    MaxLostParticlesExceeded {
        count: u64,
        max: usize,
        records: Vec<crate::util::lost_particle::LostParticle>,
    },
}

impl std::fmt::Display for GpuDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuUnavailable(e) => write!(f, "GPU unavailable: {e}"),
            Self::AdapterNotFound(msg) => write!(f, "{msg}"),
            Self::Translate(e) => write!(f, "{e}"),
            Self::UnsupportedTallyFilters { tally_index, found } => write!(
                f,
                "compute='gpu' tally {tally_index}: filters must be a spatial binner \
                 (a CellFilter, a MaterialFilter and/or a MeshFilter) with an optional \
                 EnergyFilter, EnergyFunctionFilter and/or ParticleTypeFilter(Neutron); \
                 got [{}]",
                found.join(", ")
            ),
            Self::UnsupportedMeshKind {
                tally_index,
                reason,
            } => write!(
                f,
                "compute='gpu' tally {tally_index}: mesh kind not supported on GPU: {reason}"
            ),
            Self::UnsupportedTallyScore {
                tally_index,
                reason,
            } => {
                write!(
                    f,
                    "compute='gpu' tally {tally_index}: score not supported: {reason}"
                )
            }
            Self::VarianceReductionUnsupported => write!(
                f,
                "compute='gpu' does not support variance_reduction (the kernel runs \
                 analog transport). Remove the variance_reduction entries or run on \
                 the CPU."
            ),
            Self::EnergyBinsTooFew { tally_index } => write!(
                f,
                "compute='gpu' tally {tally_index}: energy filter must define at least \
                 one bin (need at least 2 edge values, got fewer)"
            ),
            Self::CellNotInGeometry {
                tally_index,
                cell_id,
            } => write!(
                f,
                "compute='gpu' tally {tally_index}: CellFilter references cell_id={cell_id} \
                 which is not present in the model's geometry"
            ),
            Self::PhotonDataPrep(msg) => write!(f, "{msg}"),
            Self::NeutronDataPrep(msg) => write!(f, "{msg}"),
            Self::PhotonBankOverflow {
                overflow,
                count,
                capacity,
            } => write!(
                f,
                "compute='gpu' coupled neutron->photon: secondary-photon bank overflowed by \
                 {overflow} ({count} reservations, capacity {capacity}). Raise the per-batch \
                 photon bank capacity. Silent drops are forbidden by the particle_bank contract."
            ),
            Self::MixedSourceWithSecondariesUnsupported => write!(
                f,
                "compute='gpu' does not yet support a mixed neutron+photon source combined \
                 with D1S decay photons. Set use_decay_photons=False, or run on the CPU."
            ),
            Self::CoupledNxnSpillUnsupported { spilled } => write!(
                f,
                "compute='gpu' coupled/mixed pass: {spilled} (n,xn) secondaries needed more \
                 simultaneous in-thread slots than the neutron kernel holds and were handed to \
                 the device particle bank, which this pass drains for photons only. Refusing \
                 rather than dropping real neutrons. Run the neutron-only GPU path (which drains \
                 them) or run on the CPU."
            ),
            Self::OverlayTallyUnsupported { tally_index } => write!(
                f,
                "compute='gpu' tally {tally_index}: virtual-overlay tallies (response=) are \
                 not supported on the GPU -- the kernel scores the cell material's \
                 macroscopic cross section, so the requested response would be silently \
                 dropped. Run this tally with compute='cpu'."
            ),
            Self::UncappedWithoutRuntime => write!(
                f,
                "compute='gpu' needs total_particles or max_runtime to stop. Set one or both, \
                 or run on the CPU."
            ),
            Self::ConvergenceTargetsUnsupported { n_targets } => write!(
                f,
                "compute='gpu' cannot stop on convergence targets yet: the launch loop only \
                 checks total_particles and max_runtime between launches, so the {n_targets} \
                 target(s) on this model would be ignored and the run would go to the cap. \
                 Clear Model.convergence_targets to run this on the GPU, or run on the CPU."
            ),
            Self::HistoriesTruncated {
                truncated,
                launched,
                max_steps,
            } => write!(
                f,
                "{truncated} of {launched} GPU histories in one launch ({:.2}%) were still \
                 transporting when they hit gpu_max_steps_per_particle={max_steps}. Their remaining \
                 track length was not scored, so the tallies would be under-counted by an \
                 amount that cannot be corrected afterwards. The CPU runs every history to \
                 completion. Raise gpu_max_steps_per_particle (the default is 100000), or run on \
                 the CPU.",
                100.0 * *truncated as f64 / (*launched).max(1) as f64
            ),
            Self::MaxLostParticlesExceeded { count, max, .. } => write!(
                f,
                "Maximum lost particles exceeded ({count} > {max}). Most often this means the \
                 outermost surface of the geometry does not have `boundary='vacuum'` -- particles \
                 that escape the geometry have nowhere to terminate, so they are flagged as lost. \
                 Set `boundary='vacuum'` on the outer surface, fix any gaps between cells (the \
                 lost particle diagnostics above show where each particle was last located), or \
                 pass `max_lost_particles=` to `Model(...)` if leakage is expected and \
                 acceptable. (compute='gpu' counts losses per launch, so the count can exceed \
                 the cap by up to one launch.)"
            ),
        }
    }
}

impl std::error::Error for GpuDispatchError {}

impl From<GpuInitError> for GpuDispatchError {
    fn from(e: GpuInitError) -> Self {
        Self::GpuUnavailable(e)
    }
}

impl From<GpuSelectError> for GpuDispatchError {
    fn from(e: GpuSelectError) -> Self {
        match e {
            GpuSelectError::Unavailable => Self::GpuUnavailable(GpuInitError::NoF64Adapter),
            // A bad adapter name is user input, not "no GPU here" -- keep it a
            // distinct variant so the Python layer maps it to ValueError.
            GpuSelectError::AdapterNotFound(m) => Self::AdapterNotFound(m),
        }
    }
}

impl From<GpuTranslateError> for GpuDispatchError {
    fn from(e: GpuTranslateError) -> Self {
        Self::Translate(e)
    }
}

/// Aggregate result from a GPU run. Per-particle diagnostics surfaced
/// for completeness; tally outputs are written into each tally's
/// accumulator before this struct is returned (the user reads them
/// back through `SimulationResults`, not from here).
#[derive(Debug, Clone)]
pub struct GpuRunResult {
    pub n_particles: usize,
    pub n_cells: usize,
    pub alive: Vec<u32>,
    pub n_steps: Vec<u32>,
    pub final_energies: Vec<f64>,
    /// Particles the run lost to a geometry gap (issue #289). `lost_count` is
    /// every loss the kernels saw; `lost` carries the diagnostics kept for the
    /// first losses of each launch. Zero / empty for a sound geometry.
    /// `run_on_gpu_with_device` moves `lost` onto `Model::lost_particles`, so
    /// callers see them the same way as on the CPU path.
    pub lost_count: u64,
    pub lost: Vec<crate::util::lost_particle::LostParticle>,
    /// (n,xn) secondaries this run handed to the device particle bank because
    /// the producing thread's in-thread stack was full, and which were
    /// therefore transported in a later pass (issue #111 phase 2). Zero for
    /// every model measured so far; surfaced so a test can tell whether it
    /// exercised the spill path rather than assuming it did.
    pub n_spilled_secondaries: u64,
}

/// Lost-particle bookkeeping across a run's kernel launches (issue #289).
///
/// The kernels count every history that ended in no cell and keep diagnostics
/// for the first [`yamc_gpu::common::lost_particles::LOST_RECORD_CAPACITY`] of
/// them per launch. This folds those per-launch results into a run total,
/// converts the records into the same `LostParticle` values the CPU path
/// produces (resolving the cell index to its id / name), prints the same
/// diagnostic, and fails the run once `max_lost_particles` is exceeded.
///
/// Difference from the CPU, by construction: the CPU aborts at the exact
/// particle that crosses the cap, whereas the GPU can only check at a launch
/// boundary, so a launch that loses many particles reports a count past the
/// cap. The decision (run or refuse) is the same.
#[derive(Default)]
struct LostTracker {
    count: u64,
    records: Vec<crate::util::lost_particle::LostParticle>,
}

impl LostTracker {
    /// Fold one launch's device result in. `cells` resolves a record's cell
    /// index to the user-facing id / name; `ptype` is the species the launch
    /// transported (the kernels are single-species).
    #[cfg(not(target_os = "macos"))]
    fn absorb(
        &mut self,
        lost: &yamc_gpu::common::lost_particles::LostParticleResult,
        ptype: ParticleType,
        cells: &[crate::geometry::cell::Cell],
        max_lost: usize,
    ) -> Result<(), GpuDispatchError> {
        if lost.count == 0 {
            return Ok(());
        }
        self.count += lost.count;
        for r in &lost.records {
            // Keep (and print) a bounded number of records for the whole run:
            // the counter drives the abort, so extra records add spam, not
            // information. A run that aborts at the default cap of 10 still has
            // every record behind the decision.
            if self.records.len() >= yamc_gpu::common::lost_particles::LOST_RECORD_CAPACITY {
                break;
            }
            let cell = r.last_cell_index.and_then(|i| cells.get(i));
            let record = crate::util::lost_particle::LostParticle {
                particle_type: ptype,
                position: r.position,
                direction: r.direction,
                energy: r.energy,
                last_cell_index: r.last_cell_index,
                last_cell_id: cell.and_then(|c| c.cell_id),
                last_cell_name: cell.and_then(|c| c.name.clone()),
                // The kernels do not carry the crossed surface id through the
                // step, and the loss is detected at the next cell-find rather
                // than at the crossing itself.
                surface_id: None,
            };
            record.print_diagnostic();
            self.records.push(record);
        }
        if self.count > max_lost as u64 {
            return Err(GpuDispatchError::MaxLostParticlesExceeded {
                count: self.count,
                max: max_lost,
                records: self.records.clone(),
            });
        }
        Ok(())
    }
}

/// Warn when several MPI ranks will auto-select the same GPU.
///
/// Since issue #303 the launch chunks are partitioned across ranks, so the total
/// work is right; but every rank auto-selecting the same adapter means they
/// time-slice one device, which is SLOWER than running serially (measured on one
/// RADV adapter: 2M histories took 6.1 s at one rank and 12.6 s at two, for the
/// same total work). The speedup comes from one rank per device, which needs the
/// adapter pinned per rank via `compute='<adapter name>'` (see
/// `yamc.parallel.list_gpu_adapters()`) or by restricting each rank's visible
/// devices.
///
/// Silent when the caller already pinned an adapter, and for a single rank.
fn warn_if_ranks_share_one_device(device: Option<&str>, verbose: bool) {
    if device.is_some() {
        return;
    }
    let ctx = crate::mpi_context::MpiContext::init();
    if ctx.size() <= 1 || !ctx.is_root() || !verbose {
        return;
    }
    eprintln!(
        "WARNING: {} MPI ranks are each auto-selecting a GPU, so they will share \
         one device and time-slice it -- the histories are partitioned across \
         ranks, but sharing a device is slower than running on one rank. Pin one \
         adapter per rank with compute='<adapter name>' (see \
         yamc.parallel.list_gpu_adapters()), or restrict each rank's visible \
         devices.",
        ctx.size(),
    );
}

/// Fail the run when a launch truncated histories at the
/// `gpu_max_steps_per_particle` cap. A particle still `alive` at loop exit hit
/// the cap before it leaked or was absorbed, so its remaining track length
/// was never scored and every tally it touched is under-counted relative to
/// the CPU, which runs every history to completion and ignores the cap.
///
/// Checked after every launch rather than once at the end, so a run that
/// truncates fails at its first chunk instead of after the whole budget.
/// This was a `verbose.summary`-gated warning, which a `verbose=[]` run
/// silenced entirely while still returning the under-counted flux
/// (fusion-neutronics/core#23). A truncated flux is not a valid answer to
/// the question asked, so it is an error at every verbosity. The default
/// cap is high enough that this does not fire for normal physics; when it
/// does, the remedy is a higher cap or the CPU.
fn fail_if_truncated(alive: &[u32], max_steps: u32) -> Result<(), GpuDispatchError> {
    let truncated = alive.iter().filter(|&&a| a == 1).count();
    if truncated > 0 {
        return Err(GpuDispatchError::HistoriesTruncated {
            truncated,
            launched: alive.len(),
            max_steps,
        });
    }
    Ok(())
}

/// Warn when the model requests a non-Surface tracking mode. The GPU
/// neutron and photon kernels always surface-track and have no Woodcock
/// (delta) / Hybrid analogue, so a `tracking_mode = Woodcock | Hybrid`
/// request is not applied. The returned flux is still unbiased (surface
/// tracking and Woodcock/Hybrid are different estimators of the same
/// quantity), so this is a notice, not a hard error: rejecting would remove
/// a working, numerically-correct capability. Printed once per run at every
/// verbosity: it used to be gated on `verbose.summary`, so a `verbose=[]`
/// run dropped the request in silence (fusion-neutronics/core#23), and
/// `Verbose` governs progress output, not whether the user is told their
/// setting was ignored. Surface tracking (the default) is silent.
fn warn_if_tracking_mode_ignored(tracking_mode: crate::model::TrackingMode) {
    use crate::model::TrackingMode;
    if tracking_mode == TrackingMode::Surface {
        return;
    }
    eprintln!(
        "WARNING: GPU transport ignores tracking_mode={tracking_mode} and surface-tracks; \
         the flux is unbiased but the requested tracking method is not used. Set \
         tracking_mode=Surface to silence this, or run on CPU for true Woodcock/Hybrid \
         tracking."
    );
}

/// Run `model` on the GPU. Splits `settings.total_particles` into chunks
/// sized by the tallies' `particles_per_cache_write` (same derivation
/// the CPU path uses), then issues one kernel launch per chunk and
/// folds each realisation into the tally accumulators via
/// `accumulate_batch`. This matches the CPU path's variance estimator:
/// per-tally `standard_deviation` is computed across the realisations,
/// not across particles within a single launch.
///
/// Each batch gets a distinct RNG stream (see
/// `sample_initial_particles_for_batch`); identical `(settings.seed,
/// settings.total_particles, batch_idx)` gives identical results for
/// reproducibility (the batch size, and hence each batch's per-particle
/// seed band, derives from the total).
pub fn run_on_gpu(
    model: &mut Model,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    run_on_gpu_with_device(model, None, settings)
}

/// Per-source-neutron over-allocation for the device fission bank (#78). One
/// fission emits at most `ceil(nu_bar)` progeny and banks `N - 1` of them; a
/// fission history can chain several generations within one pass, but the bank
/// for THIS pass only ever holds the immediate progeny of the particles
/// transported in it. `nu_bar <= ~4.5` (fast Pu239) so `N - 1 <= 4` per
/// fission; this multiplier is generous head-room over the banked count per
/// transported neutron. Overflow is hard-errored after the launch.
const FISSION_PROGENY_PER_NEUTRON: usize = 8;

/// Generation cap for the device fission-bank drain loop (#78). A sub-critical
/// fixed-source fission chain converges geometrically: the banked population
/// shrinks by ~k_eff (< 1) each generation, so the per-source-neutron tally
/// contribution from generation g falls off as k_eff^g. For the most reactive
/// fissile spheres in scope (Pu239 / U235 at 14 MeV) a few dozen generations
/// capture the chain to far below statistical noise; the loop also exits early
/// the moment the bank empties. The cap only bounds a pathological
/// near-critical configuration (out of scope for fixed-source sub-critical
/// runs) so it never silently truncates a converging chain.
const MAX_FISSION_GENERATIONS: usize = 50;

/// Device-bank slots reserved per source neutron for (n,xn) secondaries that
/// overflow a thread's in-thread pending stack (issue #111 phase 2).
///
/// The measured need is zero: `nxn_spill_depth_is_sufficient` finds no history
/// in 8e5 needing more than the kernel's four stack slots, on the most strongly
/// multiplying fixtures available (an 8 mean-free-path Be9 sphere at 14 MeV,
/// and Pb208 where both (n,2n) and (n,3n) are open). One slot per source
/// neutron is therefore enormous head-room, and costs ~84 bytes per source
/// neutron of device bank -- an eighth of what a fissile run already allocates.
/// Overflow past it is a hard error, never a silent drop.
#[cfg(not(target_os = "macos"))]
const NXN_SPILL_SLOTS_PER_SOURCE: usize = 1;

/// Split-progeny seed for weight-w duplication (issue #236). Copy 0 keeps the
/// banked seed, so weight-1 progeny (the overwhelming majority) re-launch
/// bit-identically to before the fix; later copies get an independent
/// splitmix32-derived seed so each duplicated neutron transports on its own
/// stream.
fn split_progeny_seed(base_seed: u32, k: usize) -> u32 {
    if k == 0 {
        return base_seed;
    }
    let mut z = base_seed.wrapping_add((k as u32).wrapping_mul(0x9E37_79B9));
    z = (z ^ (z >> 16)).wrapping_mul(0x85EB_CA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
    z ^ (z >> 16)
}

/// Deterministic uniform in [0, 1) from a seed, for stochastic rounding of a
/// fractional banked weight (issue #236). The integer (n,2n)/(n,3n) yields make
/// the weight integral in practice, so this only fires for a rare
/// fractional-yield reaction; deriving from the banked seed keeps the result
/// reproducible (no global RNG state).
fn uniform_from_seed(seed: u32) -> f64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9);
    z = (z ^ (z >> 16)).wrapping_mul(0x85EB_CA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
    z ^= z >> 16;
    (z as f64) * (1.0 / 4_294_967_296.0)
}

/// Build the transport inputs for one fission-bank generation: clone the shared
/// geometry / cross-section inputs and replace the source particle SoA (seeds,
/// energies, positions, directions) with the first `count` banked neutrons.
/// The bank record layout is `bank_f64[8*i] = [E, px, py, pz, dx, dy, dz, w]`
/// and `bank_u32[4*i] = [ptype, cell, seed, gen]` (see
/// `yamc_gpu::common::particle_bank`). In the neutron-only dispatch every banked
/// record is a fission-progeny neutron, so no `ptype` filter is needed.
///
/// Weight is NOT a kernel input (every source particle starts at weight 1.0), so
/// a banked progeny whose weight was multiplied by an upstream (n,2n)/(n,3n)
/// (`weight *= yield` in the kernel) is instead re-launched as `round(w)`
/// unit-weight neutrons here (issue #236): the analog-equivalent of `w` real
/// neutrons, matching the CPU, which banks real (n,xn) neutrons rather than
/// weight-multiplying. `w == 1` yields exactly one copy with the original seed
/// (bit-identical to a non-multiplied chain); the integer (n,xn) yields make `w`
/// integral so the split is exact, with stochastic rounding covering any
/// fractional-yield case. Dropping the banked weight (the old behaviour) biased
/// the fixed-source fissile flux ~4% low for a 14 MeV source.
///
/// The returned SoA can therefore be LONGER than `count`; the caller sizes the
/// launch and the next generation's bank from the emitted length. Also returns
/// each emitted neutron's ORIGINATING source index (issue #233 Stage 2), read
/// from `bank_source_idx`, so the next generation launch keeps folding into the
/// right per-source variance sample (every copy inherits its progeny's source).
fn fission_source_inputs(
    base: &super::translate::GpuTransportInputs,
    bank_f64: &[f64],
    bank_u32: &[u32],
    bank_source_idx: &[u32],
    count: usize,
) -> (super::translate::GpuTransportInputs, Vec<u32>) {
    let mut out = base.clone();
    let mut seeds = Vec::with_capacity(count);
    let mut energies = Vec::with_capacity(count);
    let mut positions = Vec::with_capacity(count * 3);
    let mut directions = Vec::with_capacity(count * 3);
    let mut source_idx = Vec::with_capacity(count);
    for (i, &sidx) in bank_source_idx.iter().take(count).enumerate() {
        let f = i * 8;
        let u = i * 4;
        let base_seed = bank_u32[u + 2];
        // Unit-weight copies of this progeny (issue #236).
        let w = bank_f64[f + 7];
        let floor_w = w.floor();
        let mut n_copies = floor_w.max(0.0) as usize;
        let frac = w - floor_w;
        if frac > 0.0 && uniform_from_seed(base_seed) < frac {
            n_copies += 1;
        }
        for k in 0..n_copies {
            energies.push(bank_f64[f]);
            positions.push(bank_f64[f + 1]);
            positions.push(bank_f64[f + 2]);
            positions.push(bank_f64[f + 3]);
            directions.push(bank_f64[f + 4]);
            directions.push(bank_f64[f + 5]);
            directions.push(bank_f64[f + 6]);
            seeds.push(split_progeny_seed(base_seed, k));
            source_idx.push(sidx);
        }
    }
    out.seeds = seeds;
    out.energies = energies;
    out.positions = positions;
    out.directions = directions;
    (out, source_idx)
}

/// Like [`run_on_gpu`], but `device` optionally pins the GPU adapter by name
/// (the names are those from `yamc_gpu::list_vulkan_f64_adapters`). `None`
/// auto-selects (cubecl/wgpu's `HighPerformance` preference, which prefers a
/// discrete GPU). An unknown name returns `GpuDispatchError::AdapterNotFound`
/// rather than panicking.
///
/// Lost-particle diagnostics (issue #289) land on `model.lost_particles` here,
/// whether the run finished or aborted on `max_lost_particles`, so `compute='gpu'`
/// exposes them exactly like the CPU path does after its abort.
pub fn run_on_gpu_with_device(
    model: &mut Model,
    device: Option<&str>,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    let result = run_on_gpu_dispatch(model, device, settings);
    match result {
        Ok(mut run) => {
            model.lost_particles = std::mem::take(&mut run.lost);
            Ok(run)
        }
        Err(err) => {
            if let GpuDispatchError::MaxLostParticlesExceeded { records, .. } = &err {
                model.lost_particles = records.clone();
            }
            Err(err)
        }
    }
}

/// Body of [`run_on_gpu_with_device`]: validates the run, picks the kernel path
/// and drives it. Split out so the wrapper owns the one place that publishes
/// lost-particle diagnostics onto the model.
fn run_on_gpu_dispatch(
    model: &mut Model,
    device: Option<&str>,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    // GPU stop conditions (#230): a finite particle cap (`total_particles`)
    // and/or a wall-time budget (`max_runtime`), checked between launches.
    // Convergence targets are not one of them: the launch loop cannot evaluate
    // a precision target (fusion-neutronics/core#29), and a run that carries
    // targets alongside a cap used to go silently to the cap, so it is refused
    // first and in its own words (fusion-neutronics/core#23). `Some(0)` is an
    // error (0 is not "unlimited"); `None` + no `max_runtime` has no way to
    // stop, so it is rejected here rather than looping forever. Covers every
    // kernel path since they all route through this entry.
    if !model.convergence_targets.is_empty() {
        return Err(GpuDispatchError::ConvergenceTargetsUnsupported {
            n_targets: model.convergence_targets.len(),
        });
    }
    match (settings.total_particles, settings.max_runtime) {
        (Some(0), _) => {
            return Err(GpuDispatchError::Translate(
                super::error::GpuTranslateError::NoParticles,
            ))
        }
        (None, None) => return Err(GpuDispatchError::UncappedWithoutRuntime),
        _ => {}
    }
    // The neutron kernel supports survival biasing (implicit capture); any
    // other variance-reduction technique is rejected rather than silently
    // ignored. Iterate so an added variant (e.g. weight windows) fails fast
    // until its GPU path lands.
    for vr in &model.variance_reduction {
        match vr {
            crate::variance_reduction::VarianceReduction::SurvivalBiasing(_) => {}
            #[allow(unreachable_patterns)]
            _ => return Err(GpuDispatchError::VarianceReductionUnsupported),
        }
    }
    // The GPU kernels always surface-track. Unlike variance_reduction (a
    // hard reject), a non-Surface tracking_mode still yields an unbiased
    // flux, so warn-and-proceed rather than refuse -- a reject would drop a
    // working capability. Covers all kernel paths (neutron, photon, coupled)
    // since they share this entry. See `warn_if_tracking_mode_ignored`.
    warn_if_tracking_mode_ignored(model.tracking_mode);
    warn_if_ranks_share_one_device(device, model.verbose.summary);
    // Photon models need the same photon prep the CPU runs inline in
    // `run_internal` (`init_photon_data` + `init_bremsstrahlung`).
    // Without it the GPU's TTB / Doppler / relaxation tables come up
    // empty and the kernel silently transports without the
    // bremsstrahlung photon source the CPU emits -- the cause of issue
    // #415's photoelectric ~0.38x deficit. No-op for neutron-only
    // models; idempotent otherwise (`&mut Model` exists so every
    // caller gets prepared materials without a separate ensure step).
    // Same reasoning for the neutron side: the CPU gets its temperature widened
    // as a side effect of `calculate_macroscopic_xs` in `run_internal`, which
    // this path skips, so a material relabelled after its data was loaded would
    // reach `extract_material_xs` with reactions it never parsed in and fail
    // with `TemperatureNotLoaded` (#481). No-op when nothing needs widening.
    model
        .ensure_neutron_temperatures_for_gpu()
        .map_err(GpuDispatchError::NeutronDataPrep)?;
    model
        .ensure_photon_data_for_gpu()
        .map_err(GpuDispatchError::PhotonDataPrep)?;
    // Route to the photon kernel when every source emits photons.
    if !model.sources.is_empty()
        && model
            .sources
            .iter()
            .all(|s| s.particle_type() == ParticleType::Photon)
    {
        return run_on_gpu_photon(model, device, settings);
    }

    // Mixed primary source (>=1 neutron source AND >=1 photon source). Each
    // history is one source particle, sampled by strength (the CPU's
    // `sample_source`), so the run splits into a neutron-kernel pass over the
    // neutron-source share and a photon-kernel pass over the photon-source
    // share, folded per source particle. A neutron-only or photon-only model
    // never reaches here (handled above / below).
    if !model.sources.is_empty() {
        let has_neutron = model
            .sources
            .iter()
            .any(|s| s.particle_type() == ParticleType::Neutron);
        let has_photon = model
            .sources
            .iter()
            .any(|s| s.particle_type() == ParticleType::Photon);
        if has_neutron && has_photon {
            return run_on_gpu_mixed(model, device, settings);
        }
    }

    // Coupled neutron->photon production (S6). The model flag is the
    // opt-in -- same gate the CPU path uses (`has_photons()` is driven by
    // `transport_secondary_photons` for a neutron-source model). When set,
    // the neutron kernel emits secondary photons into a device bank that a
    // photon sub-pass then transports. The flag is false by default, so the
    // non-coupled neutron path below is reached unchanged.
    //
    // `tracking_mode` is not applied to transport here: the GPU neutron
    // and photon kernels always surface-track (the CPU's `tracking_mode !=
    // Surface` guard at `Model::build_photon_majorant` only gates the CPU's
    // Woodcock photon majorant, which has no GPU analogue). A non-Surface
    // request is not silent -- `warn_if_tracking_mode_ignored` above emits a
    // notice; the resulting flux is still unbiased. `ensure_photon_data_for_
    // gpu` above already validated the photon data is present.
    // D1S (`use_decay_photons`) also routes through the coupled path even if a
    // caller left `transport_secondary_photons` unset: the coupled dispatch
    // builds the decay tables and emits decay photons. Routing on either flag
    // (rather than asserting the Python-enforced implication) guarantees a D1S
    // model never falls through to the neutron-only path -- which would
    // silently drop every decay photon in a release build.
    if model.transport_secondary_photons || model.use_decay_photons {
        return run_on_gpu_coupled(model, device, settings);
    }

    let validated = validate_tallies(&model.tallies, ParticleType::Neutron)?;
    // Provisional size for the one-shot initial translate; the per-history loop
    // re-samples every launch chunk (total-independent), so it never affects
    // results or the RNG key (#230 task 1).
    let n_per_batch = INITIAL_TRANSLATE_SAMPLE;
    if settings.total_particles == Some(0) {
        return Err(GpuDispatchError::Translate(
            super::error::GpuTranslateError::NoParticles,
        ));
    }

    // Translate once; geometry/material/XS data is shared across launches.
    // Initial particle samples are overwritten per launch below.
    let mut inputs = translate_for_gpu(model, n_per_batch, settings.seed)?;

    let geometry = csg_geometry(model);
    let n_cells = inputs.cell_aabbs.len() / 6;

    // Distinct MT numbers across all SCORE_PER_MT tallies. The slot
    // each tally indexes into is `score_mts.binary_search(&mt)`.
    let score_mts = collect_score_mts(&validated);

    // Per-material macroscopic XS for the score-MT slots, on the same
    // log-energy grid the kernel uses. Empty when no tally needs per-MT
    // scoring; the kernel pads to a single zero slot in that case. When
    // the geometry has a void cell, `translate_for_gpu` appended one
    // synthetic void material slot, so pad a matching all-zero score row
    // (a void cell scores 0 for every reaction MT) to keep the kernel's
    // `n_materials = mat_f64_meta.len()/5` division consistent.
    let n_material_slots = inputs.target_mass_per_material.len();
    let xs_score_per_mt = build_xs_score_per_mt(
        &geometry.materials,
        &score_mts,
        &inputs.log_energy_grid,
        n_material_slots,
    )?;
    // Per-MT fixed-point scales sized from each channel's largest share
    // of the macroscopic total, so tiny high-threshold reaction-rate
    // tallies don't truncate to 0 (issue #150) while the integer
    // accumulator stays inside the total tally's envelope (issue #307).
    let sigma_t_score_grid = build_sigma_t_score_grid(
        &geometry.materials,
        &score_mts,
        &inputs.log_energy_grid,
        n_material_slots,
    )?;
    let per_mt_scales = per_mt_fixed_point_scales(
        &score_mts,
        &xs_score_per_mt,
        &sigma_t_score_grid,
        n_material_slots,
        inputs.log_energy_grid.len(),
    );

    // Build the multi-tally pack. With no tallies we hand the kernel a
    // dummy single-bin pack so the output buffer length stays > 0 (the
    // kernel rejects zero-length buffers); the dummy slot is never read
    // back into a Tally.
    let pack = if validated.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated, geometry, n_cells, &score_mts, &per_mt_scales)?
    };

    let max_steps = model.gpu_max_steps_per_particle;
    let survival = survival_inputs(model);
    let ctx = GpuContext::with_device(device)?;
    if model.verbose.summary {
        println!("GPU: {}", ctx.adapter_info());
    }

    // Batch-free per-history variance split (issue #233 Stages 1 + 2). Both
    // branches produce true per-history variance (batch-means is retired for the
    // neutron path); `total_particles` drops out of the RNG key in both.
    //
    // - A model that actually BANKS fission progeny (fission cross-section
    //   present AND the device fission bank is on) routes to the per-SOURCE path
    //   (`run_neutron_per_history_fissile`): a source neutron's contributions are
    //   spread across the source launch + every generation launch, so they are
    //   accumulated back into that source's per-history sample before squaring.
    // - Everything else routes to the non-fissile fast path
    //   (`run_neutron_per_history`): pure non-fissile, or fissile-with-bank-off
    //   (which multiplies weight in-thread via `weight *= nu_bar`, so each source
    //   thread is already a complete history -- Stage 1 handles it correctly).
    let has_fission_xs = inputs.xs_fission_per_material.iter().any(|&x| x > 0.0);
    // Mesh tallies use the per-source-direct variance path (issue #234), which
    // the non-fissile loop implements. Combining it with the GPU fission bank
    // (per-source accumulation across fission generations AND direct mesh
    // scoring) is not wired yet, so reject that combination explicitly rather
    // than silently mis-scoring.
    //
    // Making the bank work WITH meshes is the wanted fix and is tracked on #338;
    // the suggestion below is a way through in the meantime, not the answer.
    if has_fission_xs && model.gpu_fission_bank && pack.mesh_kind.iter().any(|&k| k != MESH_NONE) {
        if let Some(idx) = validated
            .iter()
            .position(|v| v.tally.get_mesh_filter().is_some())
        {
            return Err(GpuDispatchError::UnsupportedMeshKind {
                tally_index: validated[idx].index,
                reason: "mesh tallies combined with the GPU fission bank \
                         (gpu_fission_bank=True on a fissile model) are not yet supported; \
                         pass Model(gpu_fission_bank=False) to run this on the GPU without \
                         banking fission progeny on the device, or run the mesh tally on the CPU"
                    .to_string(),
            });
        }
    }
    if has_fission_xs && model.gpu_fission_bank {
        return run_neutron_per_history_fissile(
            model,
            settings,
            &validated,
            &mut inputs,
            &pack,
            &xs_score_per_mt,
            &survival,
            &ctx,
            max_steps,
            n_cells,
        );
    }
    run_neutron_per_history(
        model,
        settings,
        &validated,
        &mut inputs,
        &pack,
        &xs_score_per_mt,
        &survival,
        &ctx,
        max_steps,
        n_cells,
    )
}

/// Build the kernel's survival-biasing input buffer from the model's
/// variance-reduction config. Off when survival biasing is not requested
/// (gate flag 0.0 -> the kernel is byte-identical to the analog run).
fn survival_inputs(model: &Model) -> yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs {
    use yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs;
    match model.survival_biasing() {
        Some(sb) => SurvivalBiasingInputs::on(sb.weight_cutoff, sb.weight_survive),
        None => SurvivalBiasingInputs::off(),
    }
}

fn csg_geometry(model: &Model) -> &Geometry {
    // `translate_for_gpu` already validated geometry is CSG; this
    // unwrap matches that invariant. Mesh geometry is rejected
    // earlier in the call chain.
    match &model.geometry {
        GeometryKind::Csg(g) => g,
        #[cfg(feature = "mesh")]
        GeometryKind::Mesh(_) => unreachable!("mesh geometry rejected by translate_for_gpu"),
    }
}

/// Validated entry per (tally, score) pair. We resolve the score
/// discriminant once, when validating, so the pack-build and writeback
/// paths don't have to re-match the score enum.
///
/// A tally carrying several scores contributes one entry PER SCORE, all
/// consecutive and in score order. The kernel therefore never has to know
/// about multi-score tallies: every pack entry is a single-score tally of
/// exactly the shape it already handles. That works because `score` is the
/// OUTERMOST dimension of the CPU's 7D bin layout, with a stride equal to the
/// whole remaining block (see [`Tally::get_bin_index_7d`]), so a tally's bin
/// array is precisely its entries' blocks concatenated in score order.
#[derive(Debug)]
struct ValidatedTally<'a> {
    index: usize,
    tally: &'a Tally,
    /// Which of `tally.scores` this entry carries, and hence which block of
    /// the tally's bin array it accumulates into.
    score_index: usize,
    score_kind: u32,
    /// MT number for `SCORE_PER_MT` tallies. `None` for Flux / Total
    /// / Absorption / KERMA-shape tallies that route through the
    /// kernel's fast paths.
    score_mt: Option<i32>,
}

impl ValidatedTally<'_> {
    /// Bins in ONE score's block, i.e. what this entry accumulates: the CPU
    /// layout's stride for the score dimension. `Tally::num_bins()` counts
    /// every score, so a multi-score tally's array is `scores.len()` of these.
    fn block_bins(&self) -> usize {
        self.tally.num_bins() / self.tally.scores.len()
    }
}

/// Reject tally configurations the writeback can't handle. Each tally
/// must have either `[CellFilter]` or `[CellFilter, EnergyFilter]` in
/// either order, and exactly one supported score (Flux,
/// ReactionRate(total), ReactionRate(absorption), or any other
/// `ReactionRate(mt)` / named reaction -- the per-MT path handles them
/// uniformly).
///
/// `expected_particle` controls which `ParticleType(_)` filter is
/// accepted: the neutron dispatch passes `Neutron` (existing
/// behaviour), the photon dispatch passes `Photon`. The opposite
/// filter rejects with `UnsupportedTallyFilters` so the user gets a
/// clear error rather than a silent zero.
fn validate_tallies(
    tallies: &[Arc<Tally>],
    expected_particle: ParticleType,
) -> Result<Vec<ValidatedTally<'_>>, GpuDispatchError> {
    let mut out = Vec::with_capacity(tallies.len());
    for (idx, t) in tallies.iter().enumerate() {
        let t = t.as_ref();

        // Virtual-overlay tally (issue #288): `multiply_density == false` means
        // "score this response everywhere, decoupled from the cell material".
        // The kernel always folds the cell material's macroscopic XS, so an
        // overlay tally that reached it came back as the plain cell-material
        // score with no diagnostic. Reject instead of silently answering a
        // different question.
        if !t.multiply_density {
            return Err(GpuDispatchError::OverlayTallyUnsupported { tally_index: idx });
        }

        // Filters: a spatial binner is required -- a CellFilter, a
        // MaterialFilter and/or a MeshFilter (issues #234, #271). An optional
        // EnergyFilter, an optional EnergyFunctionFilter and an optional
        // `ParticleType` filter matching the pass are accepted (the last just
        // gates on something already true). Anything else -- a mismatched
        // particle filter, an unstructured mesh, etc. -- rejects.
        let mut have_cell = false;
        let mut have_material = false;
        let mut have_mesh = false;
        let mut have_energy = false;
        let mut found_names: Vec<&'static str> = Vec::new();
        let mut all_filters_supported = true;
        for filter in &t.filters {
            found_names.push(filter.type_name());
            match filter {
                Filter::Cell(_) => have_cell = true,
                Filter::Energy(_) => have_energy = true,
                // Material filter (issue #271). A cell's material is fixed for
                // the run, so the material bin is a function of the cell index
                // exactly like the cell bin: `build_tallies_pack` folds both
                // into the kernel's single spatial `cell_to_bin` dimension.
                Filter::Material(_) => have_material = true,
                // Energy-function filter (`energy_function=` /
                // `dose_coefficients=`, issue #271). Deliberately NOT counted
                // as a spatial binner below: it is a multiplicative weight on
                // the score plus an out-of-range gate, contributing no bins
                // (`Filter::num_bins() == 1`). The kernel evaluates the
                // precomputed cubic spline the CPU already solved, so both
                // backends interpolate identically.
                Filter::EnergyFunction(_) => {}
                // A `ParticleType` filter matching the pass is accepted -- the
                // kernel already transports only that species, so it just gates
                // on something already true.
                Filter::ParticleType(pf) if pf.particle_type == expected_particle => {}
                // Structured-mesh tally filter (issues #234, #279). Rectangular
                // meshes and cylindrical
                // meshes are all scored by the kernel's per-kind voxel walk.
                Filter::Mesh(_) => {
                    have_mesh = true;
                }
                // D1S `parent_nuclides` filter: only meaningful on the photon
                // pass (decay photons carry a parent-nuclide tag; neutrons do
                // not). The pack builder maps the filter's resolved ids into a
                // per-tally parent-bin dimension; the photon kernel bins each
                // photon by its banked parent id. On a neutron tally it would
                // never match, so restrict it to the photon pass.
                Filter::ParentNuclide(_) if expected_particle == ParticleType::Photon => {
                    // Accepted; handled in build_tallies_pack.
                }
                _ => all_filters_supported = false,
            }
        }
        if !all_filters_supported || !(have_cell || have_material || have_mesh) {
            return Err(GpuDispatchError::UnsupportedTallyFilters {
                tally_index: idx,
                found: found_names,
            });
        }

        // Estimator: both track-length (default) and collision are
        // supported. The pack builder sets a per-tally `is_collision`
        // flag from `t.estimator`; the kernel scores `weight × score_xs
        // / Σ_t` once per real collision for collision tallies and the
        // per-step `track_length × score_xs × weight` otherwise. No
        // estimator is rejected here.
        // `have_energy` is informational; SCORE_PER_MT and the
        // cell-only path both work, with or without an EnergyFilter.
        let _ = have_energy;

        // Score: exactly one. Flux / total use kernel fast paths (σ_t
        // is computed every step anyway); anything else is a
        // `ReactionRate(mt)` and routes through SCORE_PER_MT with a
        // per-tally slot index into the dispatch's xs_score_per_mt
        // buffer.
        //
        // Absorption (MT 27) deliberately does NOT use the kernel's
        // derived σ_a fast path: that derives σ_a = σ_t − σ_e − Σσ_inel
        // − σ_f, a small residual of large terms, which diverges from
        // the CPU (~1.3×) via catastrophic cancellation plus a
        // different inelastic-MT set (issue #415). Routing it through
        // SCORE_PER_MT(27) makes it read the same tabulated MT-27
        // reaction the CPU's macroscopic grid is built from, so it
        // matches like elastic / (n,γ) do. The kernel's SCORE_ABSORPTION
        // branch and the derived σ_a remain for the collision-physics
        // sampling (kill-vs-scatter-vs-fission), which is unaffected.
        if t.scores.is_empty() {
            return Err(GpuDispatchError::UnsupportedTallyScore {
                tally_index: idx,
                reason: "tally has no scores".to_string(),
            });
        }
        // Energy filter, if present, must give at least one bin. Checked once
        // per tally, before the per-score expansion below.
        if let Some(ef) = t.filters.iter().find_map(|f| match f {
            Filter::Energy(ef) => Some(ef),
            _ => None,
        }) {
            if ef.bins.len() < 2 {
                return Err(GpuDispatchError::EnergyBinsTooFew { tally_index: idx });
            }
        }
        // One entry per score, in score order, so the kernel only ever sees
        // single-score tallies (issue #271).
        for (score_index, score) in t.scores.iter().enumerate() {
            let (score_kind, score_mt) = classify_score(score, expected_particle, idx)?;
            out.push(ValidatedTally {
                index: idx,
                tally: t,
                score_index,
                score_kind,
                score_mt,
            });
        }
    }
    Ok(out)
}

/// Resolve one score to its kernel discriminant and, for the per-MT path, its
/// ENDF MT. Split out of `validate_tallies` so a multi-score tally classifies
/// each score through exactly the same rules a single-score tally always did.
fn classify_score(
    score: &Score,
    expected_particle: ParticleType,
    idx: usize,
) -> Result<(u32, Option<i32>), GpuDispatchError> {
    Ok(match score {
        Score::Flux(_) => (SCORE_FLUX, None),
        Score::ReactionRate(rr) if rr.mt == Mt::TOTAL => (SCORE_TOTAL, None),
        // Reaction-rate-by-MT is a neutron score. On the photon path a
        // specific MT is a category error -- a photon does not undergo a
        // neutron reaction -- so reject it rather than silently scoring
        // zero. `ReactionRate(TOTAL)` above stays allowed: it maps to the
        // photon total macroscopic XS via SCORE_TOTAL. For photons use
        // flux, heating, total, or the PhotonXS components.
        Score::ReactionRate(_) if expected_particle == ParticleType::Photon => {
            return Err(GpuDispatchError::UnsupportedTallyScore {
                tally_index: idx,
                reason: "reaction-rate-by-MT is a neutron score; for photons use \
                             flux, heating, total, or the PhotonXS components \
                             coherent/incoherent/photoelectric/pair"
                    .to_string(),
            })
        }
        Score::ReactionRate(rr) => (SCORE_PER_MT, Some(rr.mt.as_i32())),
        // KERMA-shape scores (heating, heating-local,
        // damage-energy) are macroscopic XS lookups by MT --
        // identical to per-MT reaction rates from the kernel's
        // perspective. Production scores (H1..He4) are the same
        // shape using the per-product MT (203..207).
        Score::Heating(_) => (SCORE_PER_MT, Some(Mt::HEATING.as_i32())),
        Score::HeatingLocal(_) => (SCORE_PER_MT, Some(Mt::HEATING_LOCAL.as_i32())),
        Score::DamageEnergy(_) => (SCORE_PER_MT, Some(Mt::DAMAGE_ENERGY.as_i32())),
        Score::Production(p) => (SCORE_PER_MT, Some(p.mt.as_i32())),
        // Photon component XS (coherent/incoherent/photoelectric/
        // pair). Routes through SCORE_PER_MT on the photon path;
        // the photon kernel switches on the MT to read from its
        // already-resident per-component XS arrays (no separate
        // xs_score_per_mt buffer needed -- see
        // `run_on_gpu_photon`). Rejected on the neutron path --
        // the neutron kernel has no machinery for photon MTs.
        Score::PhotonXS(p) if expected_particle == ParticleType::Photon => {
            (SCORE_PER_MT, Some(p.mt().as_i32()))
        }
        other => {
            return Err(GpuDispatchError::UnsupportedTallyScore {
                tally_index: idx,
                reason: format!(
                    "expected Flux, ReactionRate, Heating, HeatingLocal, \
                         DamageEnergy, or Production; got {:?}",
                    other
                ),
            })
        }
    })
}

/// Distinct score MTs across all `SCORE_PER_MT` tallies, sorted
/// ascending. The slot index for tally `t` is `score_mts.binary_search(&t.score_mt.unwrap()).unwrap()`.
/// Returned alongside the validated tallies so the dispatch can
/// build the matching `xs_score_per_mt` buffer once per launch.
fn collect_score_mts(validated: &[ValidatedTally<'_>]) -> Vec<i32> {
    let mut s: Vec<i32> = validated.iter().filter_map(|v| v.score_mt).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Translate the validated tallies into a flat `TalliesPack` for the
/// kernel. Builds the per-tally cell-bin map (every geom cell either
/// indexes into the tally's cell bins or is `NOT_IN_TALLY`), the
/// concatenated log-energy edges, and the output offsets so the
/// kernel can atomic-add into one flat buffer.
fn build_tallies_pack(
    validated: &[ValidatedTally<'_>],
    geometry: &Geometry,
    n_geom_cells: usize,
    score_mts: &[i32],
    per_mt_scales: &[f64],
) -> Result<TalliesPack, GpuDispatchError> {
    let n_tallies = validated.len();
    let mut score_kinds = Vec::with_capacity(n_tallies);
    let mut cell_to_bin = Vec::with_capacity(n_tallies * n_geom_cells);
    let mut n_cells_per_tally = Vec::with_capacity(n_tallies);
    let mut edges_offsets = Vec::with_capacity(n_tallies);
    let mut n_bins_per_tally = Vec::with_capacity(n_tallies);
    let mut log_edges: Vec<f64> = Vec::new();
    let mut out_offsets = Vec::with_capacity(n_tallies + 1);
    let mut score_data = Vec::with_capacity(n_tallies);
    let mut score_mt = Vec::with_capacity(n_tallies);
    let mut fixed_point_scales = Vec::with_capacity(n_tallies);
    let mut is_collision = Vec::with_capacity(n_tallies);
    let mut n_parent_per_tally = Vec::with_capacity(n_tallies);
    let mut parent_offsets = Vec::with_capacity(n_tallies + 1);
    let mut parent_ids: Vec<u32> = Vec::new();
    let mut n_mesh_per_tally = Vec::with_capacity(n_tallies);
    let mut mesh_kind = Vec::with_capacity(n_tallies);
    let mut mesh_params_offsets = Vec::with_capacity(n_tallies + 1);
    let mut mesh_params: Vec<f64> = Vec::new();
    let mut efunc_offsets = Vec::with_capacity(n_tallies + 1);
    let mut efunc_params: Vec<f64> = Vec::new();
    out_offsets.push(0u32);
    parent_offsets.push(0u32);
    mesh_params_offsets.push(0u32);
    efunc_offsets.push(0u32);

    for v in validated {
        score_kinds.push(v.score_kind);
        // Estimator flag: 1 = collision estimator (score
        // `weight × score_xs / Σ_t` once per real collision),
        // 0 = track-length (per-step `d × score_xs × weight`).
        is_collision.push(u32::from(
            v.tally.estimator == yamc_tallies::Estimator::Collision,
        ));
        // Per-tally auxiliary integer. SCORE_PER_MT carries the slot
        // index into `xs_score_per_mt`; everyone else gets 0 (ignored
        // by the kernel for non-PER_MT scores).
        let slot = v
            .score_mt
            .and_then(|mt| score_mts.binary_search(&mt).ok())
            .unwrap_or(0) as u32;
        score_data.push(slot);
        // Raw ENDF MT for the kernel's URR-window scoring fix. Only
        // SCORE_PER_MT tallies carry an MT (capture 102 / absorption 27
        // are the ones affected by URR self-shielding); other score
        // kinds get 0 and the kernel never substitutes.
        score_mt.push(v.score_mt.map(|mt| mt as u32).unwrap_or(0));

        // Fixed-point scale. KERMA-shape MTs (heating 301,
        // heating-local 901, damage-energy 444) carry per-step
        // contributions of order 1e6–1e7 eV in actinides, so the
        // default 2^30 scale would overflow the u64 atomic
        // accumulator; use scale 1.0 for those. Other SCORE_PER_MT
        // reaction-rate tallies get a per-MT scale sized from their
        // expected magnitude (issue #150): tiny high-threshold
        // channels (e.g. Fe56 MT111 `(n,2p)`) need a much larger
        // scale than `2^30` so their `~1e-11` per-collision
        // contribution survives the integer rounding instead of
        // truncating to 0. `per_mt_scales` is indexed by the same
        // slot as `score_mts`; when empty (no per-MT XS data, e.g.
        // the photon path) every per-MT tally falls back to the
        // default. Flux / total / absorption / elastic keep the
        // default scale.
        let scale = match v.score_mt {
            Some(301) | Some(901) | Some(444) => KERMA_FIXED_POINT_SCALE,
            Some(mt) => score_mts
                .binary_search(&mt)
                .ok()
                .and_then(|slot| per_mt_scales.get(slot).copied())
                .unwrap_or(DEFAULT_FIXED_POINT_SCALE),
            None => DEFAULT_FIXED_POINT_SCALE,
        };
        fixed_point_scales.push(scale);

        // Spatial-bin map. The kernel has one spatial dimension per tally
        // (`cell_to_bin`, indexed by geometry cell), and BOTH the CellFilter
        // and the MaterialFilter are constant per geometry cell -- a cell's
        // material does not change during the run -- so the two are folded into
        // it as the product bin `cell_bin * n_material_bins + material_bin`
        // (issue #271). That product is exactly the CPU's flat stride for the
        // pair: `get_bin_index_7d` contributes
        // `cell_bin * stride_material + material_bin * stride_nuclide` with
        // `stride_material = n_material_bins * stride_nuclide`, so the
        // writeback (`accumulate_kernel_tally`) divides the spatial bin back
        // out and the kernel needs no material machinery of its own.
        //
        // A cell outside either filter is `NOT_IN_TALLY`, matching the CPU,
        // where a missing cell bin or material bin skips the score outright.
        // A tally with neither filter (e.g. a bare `Tally(mesh=...)`) is
        // spatially unfiltered: every geometry cell maps to bin 0, so the mesh
        // (voxel) dimension is the only spatial binner, matching the CPU's
        // num_cell_bins == 1 (issue #234).
        let cell_filter = v.tally.get_cell_filter();
        let material_filter = v.tally.get_material_filter();
        if let Some(cf) = cell_filter {
            for &id in &cf.cell_ids {
                if !geometry.cells.iter().any(|c| c.cell_id == Some(id)) {
                    return Err(GpuDispatchError::CellNotInGeometry {
                        tally_index: v.index,
                        cell_id: id,
                    });
                }
            }
        }
        // A material id the geometry never uses is NOT an error here: the CPU
        // simply leaves that bin at zero, and diverging would make the same
        // model legal on one backend and not the other.
        let n_material_bins = material_filter.map(|mf| mf.num_bins()).unwrap_or(1) as u32;
        let n_cell_bins = cell_filter.map(|cf| cf.cell_ids.len()).unwrap_or(1) as u32;
        for cell in &geometry.cells {
            let cell_bin = match cell_filter {
                Some(cf) => cell.cell_id.and_then(|id| cf.get_bin(id)),
                None => Some(0),
            };
            let material_bin = match material_filter {
                Some(mf) => {
                    let material_id = geometry
                        .material_for(cell)
                        .and_then(|m| m.get_material_id());
                    mf.get_bin(material_id)
                }
                None => Some(0),
            };
            let bin = match (cell_bin, material_bin) {
                (Some(c), Some(m)) => c as u32 * n_material_bins + m as u32,
                _ => NOT_IN_TALLY,
            };
            cell_to_bin.push(bin);
        }
        let n_cells = n_cell_bins * n_material_bins;
        n_cells_per_tally.push(n_cells);

        // Energy bins. Cell-only tallies collapse to one bin spanning
        // [-∞, +∞] so the kernel always accumulates into bin 0.
        edges_offsets.push(log_edges.len() as u32);
        let bin_edges: Vec<f64> = match v.tally.filters.iter().find_map(|f| match f {
            Filter::Energy(ef) => Some(ef),
            _ => None,
        }) {
            Some(ef) => ef.bins.iter().map(|e| e.ln()).collect(),
            None => vec![f64::NEG_INFINITY, f64::INFINITY],
        };
        let n_bins = (bin_edges.len() - 1) as u32;
        log_edges.extend(bin_edges);
        n_bins_per_tally.push(n_bins);

        // D1S `parent_nuclides` dimension. A tally without a parent filter
        // collapses to n_parent == 1 (parent_bin 0), byte-identical to a run
        // without the parent dimension. A parent-filtered photon tally gets one
        // bin per resolved id, in filter order; the kernel maps a photon's
        // banked parent id to a bin by linear scan over these ids (matching the
        // CPU `ParentNuclideFilter::get_bin`). The ids must be resolved against
        // the same registry D1S precompute used (see
        // `prepare_decay_photon_data_for_gpu`); an unresolved filter yields zero
        // ids and contributes nothing.
        let n_ids_pushed = match v.tally.get_parent_nuclide_filter() {
            Some(pnf) => {
                let ids = pnf.resolved_ids();
                for id in ids {
                    parent_ids.push(id.get() as u32);
                }
                ids.len() as u32
            }
            None => 0,
        };
        // n_parent collapses to 1 when there is no filter (or an empty/
        // unresolved one), keeping the flat index byte-identical.
        let n_parent = n_ids_pushed.max(1);
        n_parent_per_tally.push(n_parent);
        let parent_prev = *parent_offsets.last().unwrap();
        parent_offsets.push(parent_prev + n_ids_pushed);

        // Mesh (voxel) dimension (issue #234). A tally without a MeshFilter
        // collapses to n_mesh == 1 (voxel_bin 0), byte-identical to a run
        // without mesh support. A mesh tally carries the mesh's storage size
        // (num_bins) and a packed geometry descriptor the kernel walks per
        // step. Mesh is the innermost flat dimension (stride 1).
        let n_mesh = match v.tally.get_mesh_filter() {
            Some(mf) => build_mesh_descriptor(mf, &mut mesh_kind, &mut mesh_params),
            None => {
                mesh_kind.push(MESH_NONE);
                1
            }
        };
        n_mesh_per_tally.push(n_mesh);
        mesh_params_offsets.push(mesh_params.len() as u32);

        // Energy-function table (`energy_function=` / `dose_coefficients=`,
        // issue #271). Note this adds NO factor to `out_offsets` below: the
        // filter reports `num_bins() == 1`, so it multiplies the value rather
        // than widening the output block. A tally without one leaves an empty
        // range, which is how the kernel detects its absence.
        //
        // The spline coefficients are the ones `EnergyFunctionFilter::new`
        // already solved on the host, so the kernel evaluates the same
        // polynomial rather than re-deriving (or approximating) the fit.
        if let Some(ef) = v.tally.get_energy_function_filter() {
            let energies = ef.energy();
            let coeffs = ef.spline_coeffs();
            debug_assert_eq!(coeffs.len(), energies.len() - 1);
            efunc_params.reserve(efunc_table_len(energies.len()));
            efunc_params.push(energies.len() as f64);
            efunc_params.extend_from_slice(energies);
            for c in coeffs {
                efunc_params.extend_from_slice(c);
            }
        }
        efunc_offsets.push(efunc_params.len() as u32);

        let prev = *out_offsets.last().unwrap();
        out_offsets.push(prev + n_cells * n_parent * n_bins * n_mesh);
    }

    Ok(TalliesPack {
        score_kinds,
        cell_to_bin,
        n_cells_per_tally,
        edges_offsets,
        n_bins_per_tally,
        log_edges,
        out_offsets,
        score_data,
        score_mt,
        fixed_point_scales,
        is_collision,
        n_parent_per_tally,
        parent_offsets,
        parent_ids,
        n_mesh_per_tally,
        mesh_kind,
        mesh_params_offsets,
        mesh_params,
        efunc_offsets,
        efunc_params,
    })
}

/// Pack a mesh tally's geometry descriptor into `mesh_params` (all `f64`) and
/// push its kind discriminant, returning the mesh's storage size (voxel bin
/// count). Mirrors the CPU `MeshFilter` layout the kernel walks per step
/// (issues #234, #279).
///
/// Rectangular and cylindrical meshes are both packed here and scored by the
/// kernel's matching per-kind voxel walk.
fn build_mesh_descriptor(
    mf: &yamc_tallies::MeshFilter,
    mesh_kind: &mut Vec<u32>,
    mesh_params: &mut Vec<f64>,
) -> u32 {
    match mf.kind() {
        yamc_tallies::MeshKind::Rectangular(m) => {
            mesh_kind.push(MESH_RECT_ROWMAJOR);
            let ll = m.lower_left();
            let ur = m.upper_right();
            let w = m.width();
            let shape = m.shape();
            // lower_left[3], upper_right[3], inv_width[3], width[3], shape[3]
            // (shape as f64). upper_right packed explicitly for bit-exact bounds.
            mesh_params.extend_from_slice(&ll);
            mesh_params.extend_from_slice(&ur);
            mesh_params.extend_from_slice(&[1.0 / w[0], 1.0 / w[1], 1.0 / w[2]]);
            mesh_params.extend_from_slice(&w);
            mesh_params.extend_from_slice(&[shape[0] as f64, shape[1] as f64, shape[2] as f64]);
            m.num_voxels() as u32
        }
        yamc_tallies::MeshKind::Cylindrical(m) => {
            mesh_kind.push(MESH_CYLINDRICAL);
            let o = m.origin();
            let r = m.r_grid();
            let phi = m.phi_grid();
            let z = m.z_grid();
            let shape = m.shape(); // [nr, nphi, nz]
            let full_phi = m.full_phi();
            // Header: origin[3], nr, nphi, nz, full_phi.
            mesh_params.extend_from_slice(&o);
            mesh_params.push(shape[0] as f64);
            mesh_params.push(shape[1] as f64);
            mesh_params.push(shape[2] as f64);
            mesh_params.push(if full_phi { 1.0 } else { 0.0 });
            // Grids: r, r^2, phi, z (r^2 recomputed to match the CPU's
            // precomputed r_grid_sq; IEEE multiply is deterministic).
            mesh_params.extend_from_slice(r);
            mesh_params.extend(r.iter().map(|x| x * x));
            mesh_params.extend_from_slice(phi);
            mesh_params.extend_from_slice(z);
            m.num_bins() as u32
        }
    }
}

/// Build the macroscopic per-MT score-XS buffer the kernel's
/// `SCORE_PER_MT` path indexes into. Layout is material-major:
/// `xs_score_per_mt[mat * n_score_mts * n_grid + slot * n_grid + ie]`.
/// Empty slice returned if no tally needs per-MT scoring.
fn build_xs_score_per_mt(
    materials: &[Arc<yamc_materials::material::Material>],
    score_mts: &[i32],
    log_energy_grid: &[f64],
    n_material_slots: usize,
) -> Result<Vec<f64>, GpuDispatchError> {
    if score_mts.is_empty() {
        return Ok(Vec::new());
    }
    // The kernel uses log-energy for indexing; the per-MT extractor
    // wants linear-energy. Convert once.
    let energy_grid: Vec<f64> = log_energy_grid.iter().map(|x| x.exp()).collect();

    let n_grid = energy_grid.len();
    let n_mts = score_mts.len();
    let mut out = Vec::with_capacity(materials.len() * n_mts * n_grid);

    for material in materials {
        let atoms_per_bcm = material
            .get_atoms_per_barn_cm()
            .unwrap_or_else(|e| panic!("{e}"));
        let mat_name = material
            .name
            .clone()
            .unwrap_or_else(|| format!("material_{:?}", material.material_id));
        let mut weighted: Vec<(&yamc_nuclide::Nuclide, f64)> = Vec::new();
        for (name, density) in &atoms_per_bcm {
            let Some(nuclide_arc) = material.nuclide_data.get(name) else {
                return Err(GpuDispatchError::Translate(
                    super::error::GpuTranslateError::NoLoadedTemperature {
                        nuclide: name.clone(),
                        material: mat_name.clone(),
                    },
                ));
            };
            weighted.push((nuclide_arc.as_ref(), *density));
        }
        let xs = yamc_gpu::extract_score_xs_per_mt(
            &weighted,
            material.temperature(),
            score_mts,
            &energy_grid,
        )
        .map_err(|e| {
            GpuDispatchError::Translate(super::error::GpuTranslateError::CellRegionUnsupported {
                cell_id: None,
                reason: format!("score-MT XS extraction for `{mat_name}`: {e}"),
            })
        })?;
        out.extend(xs);
    }

    // Pad an all-zero score row per synthetic void material slot (a
    // void cell scores 0 for every reaction MT). `n_material_slots`
    // is the material-row count the kernel sees (real materials +
    // any void slot); `materials` covers only the real ones.
    debug_assert!(n_material_slots >= materials.len());
    let n_void = n_material_slots - materials.len();
    out.extend(std::iter::repeat_n(0.0, n_void * n_mts * n_grid));

    debug_assert_eq!(out.len(), n_material_slots * n_mts * n_grid);
    Ok(out)
}

/// Macroscopic total cross section per material on the SAME grid and in
/// the same material order as `xs_score_per_mt`, laid out
/// `[material × n_grid]`. This is the anchor `per_mt_fixed_point_scales`
/// divides each per-MT curve by (issue #307).
///
/// Reuses `build_xs_score_per_mt` with the single score MT 1 (total), so
/// the totals come from exactly the same per-material nuclide + density
/// weighting and the same `total_xs_at` accounting the score curves do:
/// a per-MT curve and the total it is divided by are then guaranteed
/// consistent point for point. Returns an empty vec when there are no
/// per-MT tallies (nothing to scale).
fn build_sigma_t_score_grid(
    materials: &[Arc<yamc_materials::material::Material>],
    score_mts: &[i32],
    log_energy_grid: &[f64],
    n_material_slots: usize,
) -> Result<Vec<f64>, GpuDispatchError> {
    if score_mts.is_empty() {
        return Ok(Vec::new());
    }
    build_xs_score_per_mt(materials, &[MT_TOTAL], log_energy_grid, n_material_slots)
}

/// ENDF MT 1 -- the total cross section, used as the per-MT fixed-point
/// scale anchor (issue #307).
const MT_TOTAL: i32 = 1;

/// Add one launch's flat kernel tally output for tally `v` into a per-tally-bin
/// accumulator, mapping the kernel's `spatial -> parent -> energy -> mesh`
/// stride order (mesh innermost, issue #234) to the tally's 7D bin index.
///
/// The kernel's spatial dimension carries the cell and material bins folded
/// together as `cell_bin * n_material_bins + material_bin` (see
/// `build_tallies_pack`, issue #271), so it is divided back out here. A tally
/// without a `CellFilter` has one cell bin, without a `MaterialFilter` one
/// material bin and without a `MeshFilter` one mesh bin, so all three collapse
/// to the pre-mesh `cell -> parent -> energy` mapping (byte-identical). Does
/// NOT normalise: the batch-free per-history / per-source path accumulates the
/// raw per-history `sum` (and `sum_sq`) across launches and divides by the true
/// history count once, at finalize.
fn accumulate_kernel_tally(v: &ValidatedTally<'_>, out: &[f64], acc: &mut [f64]) {
    let n_cells = v
        .tally
        .get_cell_filter()
        .map(|cf| cf.cell_ids.len())
        .unwrap_or(1);
    let n_materials = v.tally.num_material_bins();
    let n_spatial = n_cells * n_materials;
    let n_parent = v.tally.num_parent_nuclide_bins().max(1);
    let n_mesh = v.tally.get_mesh_filter().map(|m| m.num_bins()).unwrap_or(1);
    // out block = n_spatial * n_parent * n_bins * n_mesh; recover n_bins (energy).
    let n_bins = out
        .len()
        .checked_div(n_spatial * n_parent * n_mesh)
        .unwrap_or(0);
    for spatial_bin in 0..n_spatial {
        let tally_cell_bin = spatial_bin / n_materials;
        let material_bin = spatial_bin % n_materials;
        for parent_bin in 0..n_parent {
            for energy_bin in 0..n_bins {
                for mesh_bin in 0..n_mesh {
                    let kernel_idx = ((spatial_bin * n_parent + parent_bin) * n_bins + energy_bin)
                        * n_mesh
                        + mesh_bin;
                    let tally_bin_idx = v
                        .tally
                        .get_bin_index_7d(
                            0,
                            tally_cell_bin,
                            material_bin,
                            0,
                            parent_bin,
                            energy_bin,
                            mesh_bin,
                        )
                        .expect("bin index in range -- bin counts validated up front");
                    acc[tally_bin_idx] += out[kernel_idx];
                }
            }
        }
    }
}

/// Finalize per-tally per-bin accumulators (physical `sum` + `sum_sq` over `n`
/// source histories) into the CPU's per-history Welford representation and
/// install them (issue #233): `mean = sum/n`, `m2 = (sum_sq - sum^2/n).max(0)`,
/// `n_histories = n`, `agg = ZERO`. Shared by the neutron and photon
/// per-history / per-source paths.
/// Globally-summed per-tally `(sum, sum_sq)` accumulators plus the pooled history
/// count, as returned by [`reduce_accumulators_across_ranks`].
type ReducedAccumulators = (Vec<Vec<f64>>, Vec<Vec<f64>>, u64);

/// Sum per-tally `(sum, sum_sq)` accumulators and the history count across MPI
/// ranks, returning `None` for a single-process run (nothing to do).
///
/// One collective pair carries every tally: the per-tally buffers are packed in
/// order with the history count appended, reduced to root and broadcast back, so
/// every rank ends up with the same global totals and the same numbers a serial
/// run would produce. Reduce-then-broadcast rather than each rank allreducing
/// keeps the result bit-identical across ranks.
fn reduce_accumulators_across_ranks(
    sum_acc: &[Vec<f64>],
    sumsq_acc: &[Vec<f64>],
    n: u64,
) -> Option<ReducedAccumulators> {
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    if mpi_ctx.size() <= 1 {
        return None;
    }
    let mut packed: Vec<f64> = Vec::with_capacity(
        sum_acc.iter().map(|v| v.len()).sum::<usize>()
            + sumsq_acc.iter().map(|v| v.len()).sum::<usize>()
            + 1,
    );
    for v in sum_acc {
        packed.extend_from_slice(v);
    }
    for v in sumsq_acc {
        packed.extend_from_slice(v);
    }
    packed.push(n as f64);

    mpi_ctx.reduce_sum_f64(&mut packed, 0);
    mpi_ctx.broadcast_f64(&mut packed, 0);

    let mut off = 0usize;
    let mut sums: Vec<Vec<f64>> = Vec::with_capacity(sum_acc.len());
    for v in sum_acc {
        sums.push(packed[off..off + v.len()].to_vec());
        off += v.len();
    }
    let mut sqs: Vec<Vec<f64>> = Vec::with_capacity(sumsq_acc.len());
    for v in sumsq_acc {
        sqs.push(packed[off..off + v.len()].to_vec());
        off += v.len();
    }
    Some((sums, sqs, packed[off] as u64))
}

/// Install per-bin `(sum, sum_sq)` accumulators onto the CPU tallies as Welford
/// stats: `mean = sum/n`, `m2 = (sum_sq - sum^2/n).max(0)`.
///
/// `sum_acc` / `sumsq_acc` are parallel to `validated`, i.e. one entry per
/// (tally, score). A tally's entries are consecutive and in score order, and
/// `score` is the outermost dimension of the CPU 7D layout with a stride of one
/// whole block, so concatenating a tally's blocks in that order IS its bin
/// array -- which is what lets the kernel stay single-score (issue #271).
fn install_grouped_stats(
    validated: &[ValidatedTally<'_>],
    sum_acc: &[Vec<f64>],
    sumsq_acc: &[Vec<f64>],
    n: u64,
) {
    let nf = n as f64;
    let mut t = 0usize;
    while t < validated.len() {
        let tally = validated[t].tally;
        let n_scores = tally.scores.len();
        debug_assert!(
            t + n_scores <= validated.len()
                && (0..n_scores).all(|k| {
                    validated[t + k].index == validated[t].index
                        && validated[t + k].score_index == k
                }),
            "a tally's score entries must be consecutive and in score order"
        );
        let mut mean: Vec<f64> = Vec::with_capacity(tally.num_bins());
        let mut m2: Vec<f64> = Vec::with_capacity(tally.num_bins());
        for k in 0..n_scores {
            mean.extend(
                sum_acc[t + k]
                    .iter()
                    .map(|&s| if n > 0 { s / nf } else { 0.0 }),
            );
            // m2 = Sum x^2 - (Sum x)^2 / n. Clamp against catastrophic-
            // cancellation round-off (a genuinely zero-variance bin can land at
            // a tiny negative).
            m2.extend(
                sum_acc[t + k]
                    .iter()
                    .zip(sumsq_acc[t + k].iter())
                    .map(|(&s, &sq)| {
                        if n > 0 {
                            (sq - s * s / nf).max(0.0)
                        } else {
                            0.0
                        }
                    }),
            );
        }
        tally.install_finalized(yamc_tallies::welford::WelfordTallyStats {
            mean,
            m2,
            n_histories: n,
            agg: yamc_tallies::welford::AggMoments::ZERO,
            score_pdf: yamc_tallies::welford::ScorePdf::default(),
        });
        t += n_scores;
    }
}

fn finalize_per_history_tallies(
    validated: &[ValidatedTally<'_>],
    sum_acc: &[Vec<f64>],
    sumsq_acc: &[Vec<f64>],
    n: u64,
) {
    // Under MPI each rank ran a DISJOINT subset of the launch chunks (see
    // `LaunchLoop::new_for_rank`), so these accumulators and the history count
    // are partial. Sum them across ranks before deriving mean / m2, else every
    // rank would normalise its own share by its own count and report a
    // single-rank answer -- which is what the GPU path did before issue #303,
    // duplicating the whole run on every rank instead.
    //
    // Per-bin sums and sums-of-squares are plain sums, so an elementwise
    // reduction is exact; there is no Welford fold to do here.
    let reduced = reduce_accumulators_across_ranks(sum_acc, sumsq_acc, n);
    let (sum_acc, sumsq_acc, n) = match reduced.as_ref() {
        Some((s, sq, rn)) => (&s[..], &sq[..], *rn),
        None => (sum_acc, sumsq_acc, n),
    };
    install_grouped_stats(validated, sum_acc, sumsq_acc, n);
}

/// Size of the one-shot initial `translate_*_for_gpu` particle sample. Every
/// launch chunk re-samples and OVERWRITES the initial particles via
/// `sample_initial_particles_for_chunk`, so this is provisional only: it is
/// total-INDEPENDENT and never affects results or the RNG key (#230 task 1 --
/// `total_particles` is not in the GPU RNG key). `>= 2` so the mixed-source
/// path's neutron/photon source-strength split is non-empty on both sides.
const INITIAL_TRANSLATE_SAMPLE: usize = 2;

/// GPU launch chunk size (histories / source neutrons per launch) for the
/// per-history variance paths (issue #233). Fixed and total-INDEPENDENT so
/// `total_particles` stays out of the RNG key. Capped at `mem_safe_max` (the
/// tally-shape memory bound -- a fine tally launches smaller chunks) and the
/// watchdog-safe 100k per-dispatch size. `YAMC_GPU_LAUNCH_CHUNK` overrides it
/// (still capped at `mem_safe_max`) for tuning and for exercising the
/// multi-chunk path in tests; the override is a fixed value, so results stay
/// total-independent.
fn launch_chunk_size(mem_safe_max: usize) -> usize {
    const FIXED_LAUNCH_CHUNK: usize = 100_000;
    let cap = mem_safe_max.max(1);
    match std::env::var("YAMC_GPU_LAUNCH_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(c) => c.max(1).min(cap),
        None => FIXED_LAUNCH_CHUNK.min(cap),
    }
}

/// `mem_safe_max` for a per-history SPILL-bounded path (non-fissile neutron /
/// photon source): how many histories one launch can hold given that each
/// carries `per_history_spill_cap` words of per-history variance state.
///
/// This is the whole ballgame for a fine tally (issue #237). The kernel is
/// memory-latency bound and has nothing but occupancy to hide that latency, so
/// throughput tracks the launch chunk almost linearly: on the issue's large
/// tally (101 cells x 500 energy) the same run measures 1,686 particles/s at a
/// 1,330-history chunk and 11,271 at 13,472, a 6.7x span for a 10x chunk. A
/// fine tally spends its per-history budget on spill and gets a small chunk, so
/// it runs the GPU at roughly one wavefront per CU.
///
/// Bound: the 64Mi-word spill budget divided by the capacity actually
/// ALLOCATED, which is the tight `per_history_spill_cap` (at most one flat bin
/// per tally per step). It used to be additionally clamped by a
/// `WATCHDOG_STEP_BUDGET / max_steps` term, on the theory that a bigger chunk
/// runs proportionally longer and could trip the driver watchdog. That clamp
/// was doing no useful work: it bounded only this one enlargement path, while
/// the very same model with a COARSER tally already launches the full
/// `FIXED_LAUNCH_CHUNK` of 100,000 histories at the same `max_steps` -- a
/// worst-case step count orders of magnitude past anything the clamp allowed
/// here. Since the value returned here is always `<= FIXED_LAUNCH_CHUNK`, no
/// launch this sizes can be longer than one the coarse-tally path already runs
/// routinely for the same model.
///
/// The remaining limit is the provisioning itself: `per_history_spill_cap` is a
/// worst case (a history that touched a distinct bin on every one of its
/// `max_steps` steps), and a real history touches ~100. So
/// `gpu_max_steps_per_particle` is now a throughput knob for a fine tally -- lower
/// it and the chunk grows -- and a model left at the default 100,000 gets
/// `spill_cap == total_out_len - K` and still lands on a small chunk. Cutting
/// that over-provisioning is the next step; it needs exact handling of a
/// history that overflows its spill, so it is not done here.
#[cfg(not(target_os = "macos"))]
fn spill_bounded_mem_safe_max(total_out_len: usize, max_steps: u32, n_tallies: usize) -> usize {
    use yamc_gpu::neutron::transport::per_history_spill_cap;
    let budget = 64usize * 1024 * 1024;
    let cap = per_history_spill_cap(total_out_len, max_steps, n_tallies);
    budget.checked_div(cap).map_or(usize::MAX, |c| c.max(1))
}

/// Batch-free per-history variance path for a NON-fissile neutron model (issue
/// #233 Stage 1). Replaces the batch-means estimator with a true per-history
/// `sum` + `sum_sq`, matching the CPU's per-history Welford:
///
/// - **Fixed launch chunk.** The chunk size is total-INDEPENDENT (the TDR-safe
///   launch size via `launch_chunk_size`, never `total_particles`-derived), so
///   the per-particle PCG seed -- keyed off the GLOBAL history index
///   `launch_idx * chunk + i` --
///   no longer depends on `total_particles` (removes it from the RNG key). A
///   fine tally that would need a large per-history spill shrinks the chunk to
///   bound the spill buffer; the chunk still depends only on the tally shape,
///   not the total.
/// - **Accumulate.** Each launch returns per-bin `sum` (Σ_h x_h) and `sum_sq`
///   (Σ_h x_h^2) in physical units; these fold additively (f64, host-side)
///   across launches, so the result is independent of how the histories are
///   chunked.
/// - **Finalize.** Per bin: `mean = sum / N`, `m2 = sum_sq − sum^2 / N`
///   (clamped `>= 0` against round-off), `n_histories = N` (the true particle
///   count). Installed into the tally's [`WelfordTallyStats`] exactly as the
///   CPU represents it. The tally-level reliability moments (`agg`) are ZERO for
///   now (variance-of-variance on GPU is a later enhancement).
///
/// Transports EXACTLY `settings.total_particles` histories (the last chunk may
/// be partial), unlike the batch-means path which dropped the sub-batch
/// remainder. When `total_particles` is `None` (uncapped) it instead launches
/// full-size chunks until `max_runtime` elapses.
///
/// Drives the outer launch/chunk loop for every batch-free GPU path, honouring
/// the same stop conditions as the CPU (#230): an optional particle cap
/// (`total_particles`) and/or an optional wall-time budget (`max_runtime`),
/// stopping at the first satisfied. The GPU can only stop between launches, so
/// a `max_runtime` overshoot of up to one chunk is expected (documented).
///
/// Usage mirrors the existing loops so a capped, budget-free run is
/// byte-identical to before (same chunk count, same `launch_idx * chunk` seed
/// band): `next()` at the top yields `(launch_idx, launch_n)` or `None` when
/// the particle cap is reached; `hit_time_budget()` at the bottom (after the
/// grand accumulators have folded this chunk) reports whether the wall-time
/// budget is spent. Callers with an inner sub-loop (fission generations, photon
/// bank drains) must let it complete and only consult `hit_time_budget()` at
/// the outer chunk boundary, so per-chunk accumulators are never left partial.
#[cfg(not(target_os = "macos"))]
struct LaunchLoop {
    cap: Option<usize>,
    budget_secs: Option<f64>,
    chunk: usize,
    start: Instant,
    idx: usize,
    /// Chunk-index stride: `1` single-process, `mpi_size` under MPI, so rank `r`
    /// takes global chunks `r, r + size, r + 2*size, ...` (issue #303).
    stride: usize,
}

#[cfg(not(target_os = "macos"))]
impl LaunchLoop {
    /// Rank-partitioned launch loop (issue #303). Single-process callers pass
    /// `rank = 0, size = 1`, which strides by one and is byte-identical to the
    /// pre-#303 loop. Rank `r` of `size` walks the
    /// global chunk indices `r, r + size, ...`, so the ranks cover the serial
    /// launch sequence exactly once between them.
    ///
    /// Partitioning BY CHUNK rather than within a chunk is what makes the result
    /// MPI-stable: `launch_idx * chunk` is the chunk's global seed band, so each
    /// rank replays the very same per-chunk RNG streams the serial run would, on
    /// a disjoint subset of chunks. Splitting a chunk between ranks instead would
    /// change the per-chunk key and give a different (though still valid) sample.
    fn new_for_rank(
        settings: &crate::model::TransportSettings,
        chunk: usize,
        rank: usize,
        size: usize,
    ) -> Self {
        let stride = size.max(1);
        LaunchLoop {
            cap: settings.total_particles,
            budget_secs: settings.max_runtime,
            chunk: chunk.max(1),
            start: Instant::now(),
            idx: rank.min(stride - 1),
            stride,
        }
    }

    /// Next chunk as `(launch_idx, launch_n)`, or `None` when a particle cap is
    /// reached. `launch_idx * chunk` is the chunk's global seed band, so the
    /// capped, budget-free sequence is identical to the old `0..n_launches`.
    fn next(&mut self) -> Option<(usize, usize)> {
        let launch_n = match self.cap {
            Some(total) => {
                let done = self.idx * self.chunk;
                if done >= total {
                    return None;
                }
                self.chunk.min(total - done)
            }
            None => self.chunk,
        };
        let idx = self.idx;
        self.idx += self.stride;
        Some((idx, launch_n))
    }

    /// Whether the wall-time budget is spent. Call at the outer chunk boundary
    /// after the chunk has fully folded into the grand accumulators.
    fn hit_time_budget(&self) -> bool {
        self.budget_secs
            .is_some_and(|b| self.start.elapsed().as_secs_f64() >= b)
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_neutron_per_history(
    model: &Model,
    settings: &crate::model::TransportSettings,
    validated: &[ValidatedTally<'_>],
    inputs: &mut super::translate::GpuTransportInputs,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    ctx: &GpuContext,
    max_steps: u32,
    n_cells: usize,
) -> Result<GpuRunResult, GpuDispatchError> {
    use yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs;

    // Fixed, total-independent launch chunk (issue #93's watchdog-safe size, NOT
    // derived from `total_particles`). A history that touches more distinct bins
    // than the register touched-list spills the overflow into a per-history
    // global buffer sized `chunk * spill_cap`. `spill_bounded_mem_safe_max` bounds
    // the chunk by that spill memory (using the SAME `per_history_spill_cap` the
    // kernel host allocates and indexes with) and by the driver watchdog. The
    // chunk depends only on the tally shape, so it stays total-independent.
    let total_out_len = pack.total_out_len() as usize;
    let mem_safe_max =
        spill_bounded_mem_safe_max(total_out_len, max_steps, pack.n_tallies() as usize);
    let chunk = launch_chunk_size(mem_safe_max);

    // No fission on this path (checked by the caller), so the device bank never
    // fires: keep it OFF (a size-1 bank), byte-identical transport.
    let fission_bank = FissionBankInputs::off();

    // A model carrying a mesh tally routes through the per-source direct path
    // (issue #234): the kernel accumulates each voxel crossing straight into
    // `src_acc` (no touched-list), and the host folds each source's grand total
    // as one variance sample -- exactly the fissile Stage-2 fold, but with
    // identity source indices (a non-fissile source has no progeny). This keeps
    // a track-length mesh tally O(1) per crossing. `flat_scales` unpacks the
    // fixed-point `src_acc` back to physical units.
    let has_mesh = pack.mesh_kind.iter().any(|&k| k != MESH_NONE);
    let flat_scales = flat_bin_scales(pack);

    // Per-tally per-bin accumulators (physical units), summed across launches.
    let mut sum_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sumsq_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();

    // Capped runs preallocate to the total; an uncapped (time-only) run grows
    // as it goes (`None` -> 0 hint).
    let mut alive_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut n_steps_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut final_energies_all: Vec<f64> =
        Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut last_n_cells = n_cells;
    let mut n_hist_total: u64 = 0;

    // Launch chunks until the particle cap is exhausted or `max_runtime`
    // elapses (checked between launches). A capped, budget-free run is
    // byte-identical to the old `0..n_launches` loop.
    // Lost-particle bookkeeping across this run's launches (issue #289).
    let mut lost = LostTracker::default();
    // (n,xn) secondaries handed to the device bank because a thread's in-thread
    // stack was full (issue #111 phase 2).
    let mut spilled_total: u64 = 0;
    // Partition the launch chunks across MPI ranks (issue #303): before this the
    // GPU path had no MPI awareness, so every rank transported the full
    // `total_particles` and rank 0 reported its own result -- n x the work, and
    // slower than serial because the ranks contend for one device.
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    let mut sched = LaunchLoop::new_for_rank(
        settings,
        chunk,
        mpi_ctx.rank() as usize,
        mpi_ctx.size() as usize,
    );
    while let Some((launch_idx, launch_n)) = sched.next() {
        // Re-sample this chunk's source particles with the global-index seed
        // band (total-independent). Overwrites the translate-provided batch-0
        // sample; the geometry / XS buffers are unchanged.
        let (seeds, energies, positions, directions) =
            super::translate::sample_initial_particles_for_chunk(
                model,
                launch_n,
                launch_idx,
                chunk,
                settings.seed,
            );
        inputs.seeds = seeds;
        inputs.energies = energies;
        inputs.positions = positions;
        inputs.directions = directions;

        // Per-history variance is only meaningful with real tallies; with none,
        // run the per-step path. A mesh model uses the per-source direct path.
        let variance = if validated.is_empty() {
            TallyVarianceMode::PerStep
        } else if has_mesh {
            TallyVarianceMode::PerSourceDirect {
                chunk_sources: launch_n as u32,
                total_bins: total_out_len as u32,
                source_idx: None,
            }
        } else {
            TallyVarianceMode::PerHistory
        };
        // Slots for (n,xn) secondaries that overflow a thread's in-thread stack
        // (issue #111 phase 2).
        let spill_capacity = launch_n.saturating_mul(NXN_SPILL_SLOTS_PER_SOURCE).max(1);

        let mut kernel = run_kernel_path(
            ctx,
            inputs,
            launch_n,
            pack,
            xs_score_per_mt,
            survival,
            &fission_bank,
            spill_capacity,
            max_steps,
            variance,
        )?;
        if kernel.bank_overflow > 0 {
            return Err(GpuDispatchError::PhotonBankOverflow {
                overflow: kernel.bank_overflow,
                count: kernel.bank_count,
                capacity: spill_capacity,
            });
        }

        // A history spilled part of itself to the bank, so it will finish in a
        // later launch. `PerHistory` flushes one variance sample per THREAD when
        // that thread's history ends, which would split such a history across
        // two samples: the mean would stay right and `std_dev` would not. Redo
        // the launch under `PerSource`, where the sample is keyed on the source
        // neutron and therefore survives across launches, and drain into the
        // same accumulator below. A launch is a pure function of its seeds, so
        // the redo reproduces it exactly and the first attempt is simply
        // discarded (nothing has been folded from it yet).
        //
        // This is a cold path: `nxn_spill_depth_is_sufficient` measures zero
        // spills in 8e5 histories of the most strongly multiplying fixture
        // available, so the common case pays only the `bank_count` read.
        //
        // A mesh model is already `PerSourceDirect`, which is per-source
        // accumulation, so it drains without redoing anything. A model with no
        // GPU tallies scores nothing at all, so whether a spilled secondary is
        // transported cannot change any output; it is left in the bank.
        let per_history = !validated.is_empty() && !has_mesh;
        let escalated = per_history && kernel.bank_count > 0;
        if escalated {
            kernel = run_kernel_path(
                ctx,
                inputs,
                launch_n,
                pack,
                xs_score_per_mt,
                survival,
                &fission_bank,
                spill_capacity,
                max_steps,
                per_source_variance(false, launch_n as u32, total_out_len as u32, None),
            )?;
            if kernel.bank_overflow > 0 {
                return Err(GpuDispatchError::PhotonBankOverflow {
                    overflow: kernel.bank_overflow,
                    count: kernel.bank_count,
                    capacity: spill_capacity,
                });
            }
        }
        last_n_cells = kernel.n_cells;
        // Counted from the launch actually kept, so an escalated chunk (whose
        // discarded first attempt spilled the same secondaries) counts once.
        spilled_total += kernel.n_spilled_secondaries;
        // Geometry gaps (issue #289): fold this launch's losses in and refuse to
        // continue past `max_lost_particles`, as the CPU does. Absorbed from the
        // launch actually kept, so an escalated chunk does not double-count.
        lost.absorb(
            &kernel.lost,
            ParticleType::Neutron,
            &csg_geometry(model).cells,
            model.max_lost_particles,
        )?;

        if has_mesh || escalated {
            // Fold each source's per-bin grand total as one variance sample
            // (`T -> sum += T`, `sum_sq += T^2`), unpacking the fixed-point
            // `src_acc` to physical units first (mirrors the fissile fold).
            let mut per_source_total = vec![0.0f64; launch_n * total_out_len];
            accumulate_src_acc(
                &kernel.src_acc,
                &flat_scales,
                total_out_len,
                &mut per_source_total,
            );
            // Any (n,xn) secondary that spilled finishes here, in a later
            // launch but under its own source's sample.
            spilled_total += drain_banked_neutrons(
                ctx,
                model,
                inputs,
                pack,
                xs_score_per_mt,
                survival,
                &fission_bank,
                max_steps,
                has_mesh,
                chunk,
                launch_n,
                total_out_len,
                &flat_scales,
                BankedNeutrons {
                    f64s: kernel.bank_f64,
                    u32s: kernel.bank_u32,
                    src: kernel.bank_source_idx,
                    count: kernel.bank_count as usize,
                },
                &mut lost,
                &mut per_source_total,
            )?;
            let mut sum_flat = vec![0.0f64; total_out_len];
            let mut sumsq_flat = vec![0.0f64; total_out_len];
            for s in 0..launch_n {
                let b0 = s * total_out_len;
                for b in 0..total_out_len {
                    let tv = per_source_total[b0 + b];
                    sum_flat[b] += tv;
                    sumsq_flat[b] += tv * tv;
                }
            }
            for (t, v) in validated.iter().enumerate() {
                let start = pack.out_offsets[t] as usize;
                let end = pack.out_offsets[t + 1] as usize;
                accumulate_kernel_tally(v, &sum_flat[start..end], &mut sum_acc[t]);
                accumulate_kernel_tally(v, &sumsq_flat[start..end], &mut sumsq_acc[t]);
            }
        } else {
            for (t, v) in validated.iter().enumerate() {
                accumulate_kernel_tally(v, &kernel.tally_outputs[t], &mut sum_acc[t]);
                accumulate_kernel_tally(v, &kernel.tally_sum_sq[t], &mut sumsq_acc[t]);
            }
        }

        fail_if_truncated(&kernel.alive, max_steps)?;

        alive_all.extend(kernel.alive);
        n_steps_all.extend(kernel.n_steps);
        final_energies_all.extend(kernel.final_energies);
        n_hist_total += launch_n as u64;

        if sched.hit_time_budget() {
            break;
        }
    }

    // Finalize per tally into the CPU's per-history Welford representation.
    finalize_per_history_tallies(validated, &sum_acc, &sumsq_acc, n_hist_total);
    Ok(GpuRunResult {
        n_particles: n_hist_total as usize,
        n_cells: last_n_cells,
        alive: alive_all,
        n_steps: n_steps_all,
        final_energies: final_energies_all,
        lost_count: lost.count,
        lost: lost.records,
        n_spilled_secondaries: spilled_total,
    })
}

/// Per-flat-`tally_out`-bin linear (sum) fixed-point scale, used to unpack the
/// per-source accumulator (`src_acc`) back to physical units. Flat bin `b`
/// belongs to the tally whose `out_offsets` range contains it (issue #233
/// Stage 2).
fn flat_bin_scales(pack: &TalliesPack) -> Vec<f64> {
    let total = pack.total_out_len() as usize;
    let mut scales = vec![1.0f64; total.max(1)];
    for t in 0..pack.fixed_point_scales.len() {
        let start = pack.out_offsets[t] as usize;
        let end = pack.out_offsets[t + 1] as usize;
        for s in scales.iter_mut().take(end).skip(start) {
            *s = pack.fixed_point_scales[t];
        }
    }
    scales
}

/// Fold ONE tally's per-source totals for a chunk into its running per-history
/// `sum` / `sum_sq` accumulators (issue #233 Stage 3). `per_source_total` is flat
/// `chunk_sources * row_stride` (physical, per-`(source, flat_bin)`); the tally
/// occupies kernel bins `[out_off .. out_off + n_kbins)`. Each source's total in
/// those bins is one variance sample; the sum and sum-of-squares over sources
/// fold into `sum_acc` / `sumsq_acc` (tally 7D order) via `accumulate_kernel_tally`.
#[allow(clippy::too_many_arguments)]
fn fold_per_source_tally(
    v: &ValidatedTally<'_>,
    per_source_total: &[f64],
    chunk_sources: usize,
    row_stride: usize,
    out_off: usize,
    n_kbins: usize,
    sum_acc: &mut [f64],
    sumsq_acc: &mut [f64],
) {
    let mut sum_flat = vec![0.0f64; n_kbins];
    let mut sumsq_flat = vec![0.0f64; n_kbins];
    for s in 0..chunk_sources {
        let base = s * row_stride + out_off;
        for b in 0..n_kbins {
            let t = per_source_total[base + b];
            sum_flat[b] += t;
            sumsq_flat[b] += t * t;
        }
    }
    accumulate_kernel_tally(v, &sum_flat, sum_acc);
    accumulate_kernel_tally(v, &sumsq_flat, sumsq_acc);
}

/// Like `fold_per_source_tally` but a source's sample is the SUM of its
/// neutron-pass and photon-pass totals -- an unfiltered "dual" (all-particle)
/// tally scored by both kernels. The neutron and photon contributions of one
/// source particle are added BEFORE squaring, so their correlation (shared
/// source) is captured exactly (issue #233 Stage 3, replacing the batch-means
/// per-batch summing).
#[allow(clippy::too_many_arguments)]
fn fold_per_source_dual(
    v: &ValidatedTally<'_>,
    pst_n: &[f64],
    stride_n: usize,
    off_n: usize,
    pst_p: &[f64],
    stride_p: usize,
    off_p: usize,
    chunk_sources: usize,
    n_kbins: usize,
    sum_acc: &mut [f64],
    sumsq_acc: &mut [f64],
) {
    let mut sum_flat = vec![0.0f64; n_kbins];
    let mut sumsq_flat = vec![0.0f64; n_kbins];
    for s in 0..chunk_sources {
        let bn = s * stride_n + off_n;
        let bp = s * stride_p + off_p;
        for b in 0..n_kbins {
            let t = pst_n[bn + b] + pst_p[bp + b];
            sum_flat[b] += t;
            sumsq_flat[b] += t * t;
        }
    }
    accumulate_kernel_tally(v, &sum_flat, sum_acc);
    accumulate_kernel_tally(v, &sumsq_flat, sumsq_acc);
}

/// Batch-free per-history variance for a FISSILE neutron model that banks
/// fission progeny (issue #233 Stage 2). A variance SAMPLE is one source
/// neutron plus ALL of its fission descendants (matching the CPU, which
/// transports progeny recursively inside the source particle's processing and
/// folds them into one per-history Welford sample). On GPU those descendants
/// are transported in separate GENERATION launches, so each launch scatters its
/// per-bin contributions into a per-SOURCE accumulator (`src_acc`, keyed by the
/// originating source index carried through the bank). We sum a chunk's
/// per-source totals across the source + all generation launches on the host,
/// then fold each source's grand total in as one sample (`sum += T`,
/// `sum_sq += T^2`), and finalize `mean = sum/N`, `m2 = sum_sq - sum^2/N` with
/// `N = the SOURCE-neutron count` (not source + progeny). Non-fissile Stage 1's
/// fast path is untouched.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_neutron_per_history_fissile(
    model: &Model,
    settings: &crate::model::TransportSettings,
    validated: &[ValidatedTally<'_>],
    inputs: &mut super::translate::GpuTransportInputs,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    ctx: &GpuContext,
    max_steps: u32,
    n_cells: usize,
) -> Result<GpuRunResult, GpuDispatchError> {
    use yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs;
    use yamc_gpu::neutron::transport::per_history_spill_cap;

    let total_out_len = pack.total_out_len() as usize;

    // Fixed, total-INDEPENDENT source chunk (drops total_particles from the RNG
    // key). The per-source accumulator is `chunk_sources * total_out_len`
    // fixed-point words; bound it (memory + u32 index) the way Stage 1 bounds the
    // spill, so a fine tally simply uses a smaller source chunk. The chunk
    // depends only on the tally shape, not `total_particles`. (The tighter
    // `spill_cap` is `<= total_out_len`, which dominates `per_thread_words`
    // below, so it does not change the chunk here; the src_acc footprint does.)
    let spill_cap = per_history_spill_cap(total_out_len, max_steps, pack.n_tallies() as usize);
    // Per source we hold `total_out_len` src_acc words; per thread `spill_cap`
    // spill words. Bound the larger footprint. 64Mi words ~= 512 MB and keeps
    // `chunk_sources * total_out_len` inside u32.
    let per_thread_words = total_out_len.max(spill_cap).max(1);
    let mem_safe_max = ((64usize * 1024 * 1024) / per_thread_words).max(1);
    let chunk = launch_chunk_size(mem_safe_max);

    // This path is only entered when `model.gpu_fission_bank` is on.
    let fission_bank = FissionBankInputs::on();
    let flat_scales = flat_bin_scales(pack);

    // Per-tally per-bin accumulators (physical), summed across source chunks.
    let mut sum_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sumsq_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();

    // Per-particle diagnostics: SOURCE neutrons only (ordered by global source
    // index, so a run of N and 2N share a bit-identical prefix). Progeny are
    // transported but their diagnostics are not surfaced.
    let mut alive_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut n_steps_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut final_energies_all: Vec<f64> =
        Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut last_n_cells = n_cells;
    let mut n_hist_total: u64 = 0;

    // One chunk = `chunk_sources` source neutrons plus their entire fission
    // cascade. `max_runtime` is checked only at this outer boundary (never
    // mid-cascade), so a source neutron's progeny are always fully counted.
    // Lost-particle bookkeeping across this run's launches (issue #289).
    let mut lost = LostTracker::default();
    // (n,xn) secondaries handed to the device bank alongside the fission
    // progeny (issue #111 phase 2); they drain together.
    let mut spilled_total: u64 = 0;
    // Partition the launch chunks across MPI ranks (issue #303): before this the
    // GPU path had no MPI awareness, so every rank transported the full
    // `total_particles` and rank 0 reported its own result -- n x the work, and
    // slower than serial because the ranks contend for one device.
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    let mut sched = LaunchLoop::new_for_rank(
        settings,
        chunk,
        mpi_ctx.rank() as usize,
        mpi_ctx.size() as usize,
    );
    while let Some((chunk_idx, chunk_sources)) = sched.next() {
        // Source launch: identity source_idx (`source_idx[i] = i`).
        let (seeds, energies, positions, directions) =
            super::translate::sample_initial_particles_for_chunk(
                model,
                chunk_sources,
                chunk_idx,
                chunk,
                settings.seed,
            );
        inputs.seeds = seeds;
        inputs.energies = energies;
        inputs.positions = positions;
        inputs.directions = directions;

        // Host-side per-(source, flat_bin) grand total (physical), accumulated
        // across the source + every generation launch of THIS chunk.
        let mut per_source_total = vec![0.0f64; chunk_sources * total_out_len];

        let bank_capacity = chunk_sources
            .saturating_mul(FISSION_PROGENY_PER_NEUTRON)
            .max(1);

        let kernel = run_kernel_path(
            ctx,
            inputs,
            chunk_sources,
            pack,
            xs_score_per_mt,
            survival,
            &fission_bank,
            bank_capacity,
            max_steps,
            TallyVarianceMode::PerSource {
                chunk_sources: chunk_sources as u32,
                total_bins: total_out_len as u32,
                source_idx: None,
            },
        )?;
        lost.absorb(
            &kernel.lost,
            ParticleType::Neutron,
            &csg_geometry(model).cells,
            model.max_lost_particles,
        )?;
        if kernel.bank_overflow > 0 {
            return Err(GpuDispatchError::PhotonBankOverflow {
                overflow: kernel.bank_overflow,
                count: kernel.bank_count,
                capacity: bank_capacity,
            });
        }
        last_n_cells = kernel.n_cells;
        spilled_total += kernel.n_spilled_secondaries;
        accumulate_src_acc(
            &kernel.src_acc,
            &flat_scales,
            total_out_len,
            &mut per_source_total,
        );
        fail_if_truncated(&kernel.alive, max_steps)?;
        alive_all.extend(kernel.alive);
        n_steps_all.extend(kernel.n_steps);
        final_energies_all.extend(kernel.final_energies);

        // Fission-generation loop: drain banked progeny, re-transport with their
        // INHERITED source index, folding into the SAME per-source accumulator
        // (do NOT reset it between generations). Each launch drains AT MOST
        // `chunk` progeny so the per-thread spill buffer (`n_drain * spill_cap`)
        // stays within the same memory bound as the source launch even if a
        // (super-critical) chain multiplies faster than it converges; any
        // leftover banked progeny carry forward, prepended to the next launch's
        // drain. For a sub-critical fixed-source model (the supported domain) the
        // bank shrinks each generation, so `n_drain < chunk` always and the whole
        // bank drains in one launch per generation -- byte-identical to draining
        // it directly, `leftover` stays 0, and the queue is inert.
        spilled_total += drain_banked_neutrons(
            ctx,
            model,
            inputs,
            pack,
            xs_score_per_mt,
            survival,
            &fission_bank,
            max_steps,
            false,
            chunk,
            chunk_sources,
            total_out_len,
            &flat_scales,
            BankedNeutrons {
                f64s: kernel.bank_f64,
                u32s: kernel.bank_u32,
                src: kernel.bank_source_idx,
                count: kernel.bank_count as usize,
            },
            &mut lost,
            &mut per_source_total,
        )?;

        // Fold this chunk's per-source grand totals into the tally accumulators:
        // one variance sample per source (`T -> sum += T`, `sum_sq += T^2`).
        let mut sum_flat = vec![0.0f64; total_out_len];
        let mut sumsq_flat = vec![0.0f64; total_out_len];
        for s in 0..chunk_sources {
            let base = s * total_out_len;
            for b in 0..total_out_len {
                let t = per_source_total[base + b];
                sum_flat[b] += t;
                sumsq_flat[b] += t * t;
            }
        }
        for (ti, v) in validated.iter().enumerate() {
            let start = pack.out_offsets[ti] as usize;
            let end = pack.out_offsets[ti + 1] as usize;
            accumulate_kernel_tally(v, &sum_flat[start..end], &mut sum_acc[ti]);
            accumulate_kernel_tally(v, &sumsq_flat[start..end], &mut sumsq_acc[ti]);
        }
        n_hist_total += chunk_sources as u64;

        if sched.hit_time_budget() {
            break;
        }
    }

    // Finalize into the CPU's per-history Welford representation. N is the
    // SOURCE-neutron count (each source is one variance sample), NOT source +
    // progeny -- so the reported n_histories / error bar / FOM are per source.
    //
    // Under MPI the accumulators cover only this rank's share of the launch
    // chunks, so reduce them first (issue #303), exactly as
    // `finalize_per_history_tallies` does for the other paths. This path derives
    // mean / m2 inline (its N is the source count), so it calls the shared
    // reduction directly rather than going through that helper.
    let reduced = reduce_accumulators_across_ranks(&sum_acc, &sumsq_acc, n_hist_total);
    let (sum_acc, sumsq_acc, n_hist_total) = match reduced {
        Some((s, sq, rn)) => (s, sq, rn),
        None => (sum_acc, sumsq_acc, n_hist_total),
    };
    let n = n_hist_total;
    install_grouped_stats(validated, &sum_acc, &sumsq_acc, n);
    Ok(GpuRunResult {
        n_particles: n_hist_total as usize,
        n_cells: last_n_cells,
        alive: alive_all,
        n_steps: n_steps_all,
        final_energies: final_energies_all,
        lost_count: lost.count,
        lost: lost.records,
        n_spilled_secondaries: spilled_total,
    })
}

/// One launch's banked neutrons, read back from the device particle bank:
/// fission progeny (#78) and/or (n,xn) secondaries that overflowed a thread's
/// pending stack (issue #111 phase 2). `src[i]` is the index of the SOURCE
/// neutron record `i` descends from.
#[cfg(not(target_os = "macos"))]
struct BankedNeutrons {
    f64s: Vec<f64>,
    u32s: Vec<u32>,
    src: Vec<u32>,
    count: usize,
}

/// Re-transport every neutron the kernel banked, generation by generation,
/// until the bank empties.
///
/// Two producers feed it: the fission chain (#78) and the (n,xn) spill (issue
/// #111 phase 2), a secondary produced while its thread already held
/// [`yamc_gpu::neutron::transport::PEND_SLOTS`] others. Both are just neutrons
/// with a position, a direction, an energy and their own transport seed, so
/// they drain the same way.
///
/// The variance bookkeeping is the reason this is not simply "launch them as
/// extra source particles": each banked neutron carries the index of the SOURCE
/// neutron it descends from, and every generation folds into
/// `per_source_total` under that index. A source neutron therefore contributes
/// ONE variance sample no matter how many launches its cascade took, which is
/// what keeps `std_dev` right when part of a history is transported in a later
/// pass than the rest of it.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn drain_banked_neutrons(
    ctx: &GpuContext,
    model: &Model,
    inputs: &super::translate::GpuTransportInputs,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    fission_bank: &yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs,
    max_steps: u32,
    has_mesh: bool,
    chunk: usize,
    chunk_sources: usize,
    total_out_len: usize,
    flat_scales: &[f64],
    bank: BankedNeutrons,
    lost: &mut LostTracker,
    per_source_total: &mut [f64],
) -> Result<u64, GpuDispatchError> {
    let mut spilled = 0u64;
    let mut pending_f64 = bank.f64s;
    let mut pending_u32 = bank.u32s;
    let mut pending_src = bank.src;
    let mut pending_count = bank.count;
    for _launch in 0..MAX_FISSION_GENERATIONS {
        if pending_count == 0 {
            break;
        }
        let avail = pending_f64.len() / 8;
        let n_drain = pending_count.min(avail).min(chunk);
        let (gen_inputs, gen_source_idx) =
            fission_source_inputs(inputs, &pending_f64, &pending_u32, &pending_src, n_drain);
        // Weight-w progeny are re-launched as `round(w)` unit-weight neutrons
        // (issue #236), so the emitted particle count can exceed `n_drain`;
        // size the launch and this generation's progeny bank from it.
        let n_launch = gen_inputs.seeds.len();
        let gen_cap = n_launch.saturating_mul(FISSION_PROGENY_PER_NEUTRON).max(1);
        let gen_kernel = run_kernel_path(
            ctx,
            &gen_inputs,
            n_launch,
            pack,
            xs_score_per_mt,
            survival,
            fission_bank,
            gen_cap,
            max_steps,
            per_source_variance(
                has_mesh,
                chunk_sources as u32,
                total_out_len as u32,
                Some(&gen_source_idx),
            ),
        )?;
        lost.absorb(
            &gen_kernel.lost,
            ParticleType::Neutron,
            &csg_geometry(model).cells,
            model.max_lost_particles,
        )?;
        if gen_kernel.bank_overflow > 0 {
            return Err(GpuDispatchError::PhotonBankOverflow {
                overflow: gen_kernel.bank_overflow,
                count: gen_kernel.bank_count,
                capacity: gen_cap,
            });
        }
        spilled += gen_kernel.n_spilled_secondaries;
        accumulate_src_acc(
            &gen_kernel.src_acc,
            flat_scales,
            total_out_len,
            per_source_total,
        );
        // Next pending = leftover of THIS bank (not drained this launch) then
        // the progeny just produced.
        let leftover = pending_count - n_drain;
        let new_n = (gen_kernel.bank_count as usize).min(gen_kernel.bank_f64.len() / 8);
        let mut next_f64 = Vec::with_capacity((leftover + new_n) * 8);
        let mut next_u32 = Vec::with_capacity((leftover + new_n) * 4);
        let mut next_src = Vec::with_capacity(leftover + new_n);
        next_f64.extend_from_slice(&pending_f64[n_drain * 8..pending_count * 8]);
        next_u32.extend_from_slice(&pending_u32[n_drain * 4..pending_count * 4]);
        next_src.extend_from_slice(&pending_src[n_drain..pending_count]);
        next_f64.extend_from_slice(&gen_kernel.bank_f64[..new_n * 8]);
        next_u32.extend_from_slice(&gen_kernel.bank_u32[..new_n * 4]);
        next_src.extend_from_slice(&gen_kernel.bank_source_idx[..new_n]);
        pending_f64 = next_f64;
        pending_u32 = next_u32;
        pending_src = next_src;
        pending_count = leftover + new_n;
    }
    Ok(spilled)
}

/// Unpack a per-source accumulator launch result (`src_acc`, fixed-point,
/// `chunk_sources * total_out_len` row-major) to physical units and ADD into the
/// host-side per-`(source, flat_bin)` running total (issue #233 Stage 2). Folds
/// each generation launch's contribution into the source it descends from.
fn accumulate_src_acc(
    src_acc: &[u64],
    flat_scales: &[f64],
    total_out_len: usize,
    per_source_total: &mut [f64],
) {
    if total_out_len == 0 {
        return;
    }
    for (i, &bits) in src_acc.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        let b = i % total_out_len;
        per_source_total[i] += (bits as i64 as f64) / flat_scales[b];
    }
}

/// Pick the per-source variance mode for a launch: the DIRECT variant (issue
/// #234) when the pass carries a mesh tally so the kernel fans each voxel
/// crossing straight into `src_acc`, else the touched-list `PerSource`. Both
/// share the same `src_acc` row layout, so callers fold identically.
fn per_source_variance(
    has_mesh: bool,
    chunk_sources: u32,
    total_bins: u32,
    source_idx: Option<&[u32]>,
) -> TallyVarianceMode<'_> {
    if has_mesh {
        TallyVarianceMode::PerSourceDirect {
            chunk_sources,
            total_bins,
            source_idx,
        }
    } else {
        TallyVarianceMode::PerSource {
            chunk_sources,
            total_bins,
            source_idx,
        }
    }
}

/// Photon counterpart to `run_on_gpu`. Internally similar shape:
/// translate the model into photon-side flat buffers, build the
/// same `TalliesPack`, loop over batches calling the photon kernel,
/// fold tally results back into the model.
///
/// Photon score support: `SCORE_FLUX`, `SCORE_TOTAL`, and
/// `SCORE_PER_MT` for the four photon-component MTs (502 coherent,
/// 504 incoherent, 522 photoelectric, 516 pair). The photon kernel
/// reads the MT number directly out of `pack.score_data` and
/// switches on it to read the already-resident per-component XS
/// arrays -- no separate `xs_score_per_mt` buffer (data-piggybacking
/// refinement of option A).
// macOS has no f64 GPU path: `yamc_gpu::neutron::transport` (the per-history
// kernel + `PERHIST_K`) is gated out, so the two neutron dispatch entry points
// are stubbed exactly like the photon/coupled/mixed ones below. macOS never
// reaches them at runtime (`GpuContext::with_device` returns `NoF64Adapter`
// first); the stubs only keep `run_on_gpu`'s call sites compiling.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_neutron_per_history(
    _model: &Model,
    _settings: &crate::model::TransportSettings,
    _validated: &[ValidatedTally<'_>],
    _inputs: &mut super::translate::GpuTransportInputs,
    _pack: &TalliesPack,
    _xs_score_per_mt: &[f64],
    _survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    _ctx: &GpuContext,
    _max_steps: u32,
    _n_cells: usize,
) -> Result<GpuRunResult, GpuDispatchError> {
    Err(GpuDispatchError::GpuUnavailable(
        yamc_gpu::GpuInitError::NoF64Adapter,
    ))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_neutron_per_history_fissile(
    _model: &Model,
    _settings: &crate::model::TransportSettings,
    _validated: &[ValidatedTally<'_>],
    _inputs: &mut super::translate::GpuTransportInputs,
    _pack: &TalliesPack,
    _xs_score_per_mt: &[f64],
    _survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    _ctx: &GpuContext,
    _max_steps: u32,
    _n_cells: usize,
) -> Result<GpuRunResult, GpuDispatchError> {
    Err(GpuDispatchError::GpuUnavailable(
        yamc_gpu::GpuInitError::NoF64Adapter,
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn run_on_gpu_photon(
    _model: &Model,
    _device: Option<&str>,
    _settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    // `yamc_gpu::photon::transport` is gated out on macOS (no f64 GPU path). The
    // neutron path mirrors this with `run_kernel_path`. Returning the
    // same error `GpuContext::new()` would on macOS keeps callers
    // platform-uniform.
    Err(GpuDispatchError::GpuUnavailable(
        yamc_gpu::GpuInitError::NoF64Adapter,
    ))
}

#[cfg(not(target_os = "macos"))]
pub(super) fn run_on_gpu_photon(
    model: &Model,
    device: Option<&str>,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    let validated = validate_tallies(&model.tallies, ParticleType::Photon)?;
    if settings.total_particles == Some(0) {
        return Err(GpuDispatchError::Translate(
            super::error::GpuTranslateError::NoParticles,
        ));
    }
    // Provisional sample size for translate; the per-history loop re-samples each
    // launch chunk (total-independent), so this value does not affect results.
    let seed_sample = INITIAL_TRANSLATE_SAMPLE;
    let mut inputs =
        super::translate_photon::translate_photon_for_gpu(model, seed_sample, settings.seed)?;
    let geometry = csg_geometry(model);
    let n_cells = inputs.cell_aabbs.len() / 6;

    let mut pack = if validated.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated, geometry, n_cells, &[], &[])?
    };
    // Refinement A: raw MT into score_data so the photon kernel switches on it.
    if !validated.is_empty() {
        for (i, v) in validated.iter().enumerate() {
            pack.score_data[i] = v.score_mt.map(|m| m as u32).unwrap_or(0);
        }
    }

    let max_steps = model.gpu_max_steps_per_particle;
    let ctx = GpuContext::with_device(device)?;
    if model.verbose.summary {
        println!("GPU: {}", ctx.adapter_info());
    }

    // Batch-free per-history variance (issue #233 Stage 3): each source photon IS
    // a history (its cascade is in-thread), so this is the Stage 1 mechanism on
    // the photon kernel. Fixed total-independent chunk; per-history sum + sum_sq;
    // finalize into WelfordTallyStats with N = source-photon count.
    let total_out_len = pack.total_out_len() as usize;
    // Watchdog-clamped memory bound (issue #233 Stage 4), same as the non-fissile
    // neutron path. `per_history_spill_cap` is recomputed identically in the
    // photon kernel host for the actual spill allocation + indexing.
    let mem_safe_max =
        spill_bounded_mem_safe_max(total_out_len, max_steps, pack.n_tallies() as usize);
    let chunk = launch_chunk_size(mem_safe_max);
    // A model carrying a mesh tally routes through the per-source direct path
    // (issue #234), exactly like the neutron host: the kernel fans each voxel
    // crossing straight into `src_acc` (no touched-list) and the host folds each
    // source photon's per-bin grand total as one variance sample. `flat_scales`
    // unpacks the fixed-point `src_acc` back to physical units.
    let has_mesh = pack.mesh_kind.iter().any(|&k| k != MESH_NONE);
    let flat_scales = flat_bin_scales(&pack);

    let mut sum_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sumsq_acc: Vec<Vec<f64>> = validated
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();

    let mut alive_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut n_steps_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut final_energies_all: Vec<f64> =
        Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut last_n_cells = n_cells;
    let mut n_hist_total: u64 = 0;

    // Launch chunks until the source-photon cap is exhausted or `max_runtime`
    // elapses (checked between launches).
    // Lost-particle bookkeeping across this run's launches (issue #289).
    let mut lost = LostTracker::default();
    // Partition the launch chunks across MPI ranks (issue #303): before this the
    // GPU path had no MPI awareness, so every rank transported the full
    // `total_particles` and rank 0 reported its own result -- n x the work, and
    // slower than serial because the ranks contend for one device.
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    let mut sched = LaunchLoop::new_for_rank(
        settings,
        chunk,
        mpi_ctx.rank() as usize,
        mpi_ctx.size() as usize,
    );
    while let Some((launch_idx, launch_n)) = sched.next() {
        // Per-history variance needs real tallies; with none, run the per-step
        // path. A mesh model uses the per-source direct path (chunk-sized here).
        let variance_mode = if validated.is_empty() {
            TallyVarianceMode::PerStep
        } else if has_mesh {
            TallyVarianceMode::PerSourceDirect {
                chunk_sources: launch_n as u32,
                total_bins: total_out_len as u32,
                source_idx: None,
            }
        } else {
            TallyVarianceMode::PerHistory
        };
        let (seeds, energies, positions, directions) =
            super::translate::sample_initial_particles_for_chunk(
                model,
                launch_n,
                launch_idx,
                chunk,
                settings.seed,
            );
        inputs.seeds = seeds;
        inputs.energies = energies;
        inputs.positions = positions;
        inputs.directions = directions;

        // Fresh-source photons carry weight 1.0 and no D1S parent.
        let weights = vec![1.0_f64; launch_n];
        let parent_ids = vec![0u32; launch_n];
        let result = run_photon_primary_pass(
            &ctx,
            &inputs,
            &weights,
            &parent_ids,
            &pack,
            max_steps,
            variance_mode,
        );
        lost.absorb(
            &result.lost,
            ParticleType::Photon,
            &csg_geometry(model).cells,
            model.max_lost_particles,
        )?;
        last_n_cells = result.n_cells;

        if has_mesh {
            // Fold each source photon's per-bin grand total as one variance
            // sample (unpacking fixed-point `src_acc` to physical units first),
            // exactly like the neutron mesh host.
            let mut per_source_total = vec![0.0f64; launch_n * total_out_len];
            accumulate_src_acc(
                &result.src_acc,
                &flat_scales,
                total_out_len,
                &mut per_source_total,
            );
            let mut sum_flat = vec![0.0f64; total_out_len];
            let mut sumsq_flat = vec![0.0f64; total_out_len];
            for s in 0..launch_n {
                let b0 = s * total_out_len;
                for b in 0..total_out_len {
                    let tv = per_source_total[b0 + b];
                    sum_flat[b] += tv;
                    sumsq_flat[b] += tv * tv;
                }
            }
            for (t, v) in validated.iter().enumerate() {
                let start = pack.out_offsets[t] as usize;
                let end = pack.out_offsets[t + 1] as usize;
                accumulate_kernel_tally(v, &sum_flat[start..end], &mut sum_acc[t]);
                accumulate_kernel_tally(v, &sumsq_flat[start..end], &mut sumsq_acc[t]);
            }
        } else {
            for (t, v) in validated.iter().enumerate() {
                accumulate_kernel_tally(v, &result.tally_outputs[t], &mut sum_acc[t]);
                accumulate_kernel_tally(v, &result.tally_sum_sq[t], &mut sumsq_acc[t]);
            }
        }

        fail_if_truncated(&result.alive, max_steps)?;

        alive_all.extend(result.alive);
        n_steps_all.extend(result.n_steps);
        final_energies_all.extend(result.final_energies);
        n_hist_total += launch_n as u64;

        if sched.hit_time_budget() {
            break;
        }
    }

    finalize_per_history_tallies(&validated, &sum_acc, &sumsq_acc, n_hist_total);
    Ok(GpuRunResult {
        n_particles: n_hist_total as usize,
        n_cells: last_n_cells,
        alive: alive_all,
        n_steps: n_steps_all,
        final_energies: final_energies_all,
        lost_count: lost.count,
        lost: lost.records,
        // Photon / coupled / mixed passes refuse an (n,xn) spill outright
        // (`CoupledNxnSpillUnsupported`), so reaching here means none happened.
        n_spilled_secondaries: 0,
    })
}

/// Per-neutron-history secondary-photon over-allocation for the coupled
/// device bank. The neutron kernel banks at most one photon per real
/// collision (it samples ONE photon product per collision; the per-material
/// photon yield `y_t < 1` for the structural materials this path targets),
/// so the banked count is bounded by the total collisions, itself bounded by
/// the per-history step count. A single high-energy (n,xn)/(n,gamma) history
/// in iron banks only a handful of photons in practice; this multiplier is a
/// generous head-room factor over that observed norm. `photon_bank.overflow`
/// is hard-errored after the launch, so an under-estimate fails loudly (with a
/// "raise the capacity" message) rather than silently dropping photons.
#[cfg(not(target_os = "macos"))]
const COUPLED_PHOTONS_PER_NEUTRON: usize = 8;

/// Coupled neutron->photon dispatch (S6). Runs the neutron kernel with photon
/// emission enabled, then drains the per-batch secondary-photon bank into the
/// photon kernel so the produced photons are transported too. Neutron tallies
/// are scored by the neutron kernel; photon tallies by the photon sub-pass;
/// both are merged back into the model at their original indices.
///
/// The non-coupled neutron path (`run_on_gpu_with_device` when
/// `transport_secondary_photons` is false) is untouched -- this is a separate
/// entry reached only when the flag is set.
#[cfg(target_os = "macos")]
fn run_on_gpu_coupled(
    _model: &mut Model,
    _device: Option<&str>,
    _settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    Err(GpuDispatchError::GpuUnavailable(
        yamc_gpu::GpuInitError::NoF64Adapter,
    ))
}

#[cfg(not(target_os = "macos"))]
fn run_on_gpu_coupled(
    model: &mut Model,
    device: Option<&str>,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    // Tally routing. A ParticleType filter picks the pass; a tally with NO
    // particle filter means ALL particles, so an unfiltered Flux / total
    // tally is the neutron + photon SUM (`dual`) -- scored by both passes and
    // summed per batch below. See `classify_coupled_tallies`. Dual tallies are
    // appended to BOTH pass lists (so each kernel scores them); particle-
    // filtered tallies go to exactly one pass. Each pass validates against its
    // own particle type so a mistargeted filter rejects rather than silently
    // zeroing.
    let (neutron_only, photon_only, dual) = classify_coupled_tallies(&model.tallies)?;
    // Boundaries into the PACK, which `validate_tallies` builds one entry per
    // (tally, score) -- so where the pass's own tallies end and the appended
    // dual ones begin is a score count, not a tally count (issue #271).
    let n_neutron_only: usize = neutron_only.iter().map(|t| t.scores.len()).sum();
    let n_photon_only: usize = photon_only.iter().map(|t| t.scores.len()).sum();
    let neutron_pass: Vec<Arc<Tally>> = neutron_only.iter().chain(dual.iter()).cloned().collect();
    let photon_pass: Vec<Arc<Tally>> = photon_only.iter().chain(dual.iter()).cloned().collect();
    let validated_n = validate_tallies(&neutron_pass, ParticleType::Neutron)?;
    let validated_p = validate_tallies(&photon_pass, ParticleType::Photon)?;

    if settings.total_particles == Some(0) {
        return Err(GpuDispatchError::Translate(
            super::error::GpuTranslateError::NoParticles,
        ));
    }
    // Provisional sample size for translate / photon-translate / bank sizing; the
    // per-source loop below re-samples each launch chunk (total-independent).
    let seed_sample = INITIAL_TRANSLATE_SAMPLE;

    // D1S decay-photon prep (only when use_decay_photons): builds one shared
    // NuclideRegistry, precomputes per-nuclide decay photon data interning the
    // chain emitters, re-Arcs the materials with sorted_nuclide_ids, and
    // resolves every tally's `ParentNuclideFilter` against the SAME registry.
    // Done BEFORE the immutable geometry/translate borrows below (it needs
    // `&mut model` and re-Arcs materials). When off, both are None and the path
    // is the prompt-coupled path unchanged. The Python API guarantees
    // use_decay_photons implies transport_secondary_photons, so this runs
    // inside the coupled dispatch.
    let decay_prep = if model.use_decay_photons {
        Some(
            model
                .prepare_decay_photon_data_for_gpu()
                .map_err(GpuDispatchError::PhotonDataPrep)?,
        )
    } else {
        None
    };

    // Neutron-side translation + per-MT score buffers (same as the
    // non-coupled neutron path).
    let mut inputs = translate_for_gpu(model, seed_sample, settings.seed)?;
    let geometry = csg_geometry(model);
    let n_cells = inputs.cell_aabbs.len() / 6;
    let score_mts = collect_score_mts(&validated_n);
    // Pad a void score row to match the synthetic void material slot
    // `translate_for_gpu` appends for void cells (see the non-coupled
    // path for the rationale).
    let n_material_slots = inputs.target_mass_per_material.len();
    let xs_score_per_mt = build_xs_score_per_mt(
        &geometry.materials,
        &score_mts,
        &inputs.log_energy_grid,
        n_material_slots,
    )?;
    let sigma_t_score_grid = build_sigma_t_score_grid(
        &geometry.materials,
        &score_mts,
        &inputs.log_energy_grid,
        n_material_slots,
    )?;
    let per_mt_scales = per_mt_fixed_point_scales(
        &score_mts,
        &xs_score_per_mt,
        &sigma_t_score_grid,
        n_material_slots,
        inputs.log_energy_grid.len(),
    );
    let pack_n = if validated_n.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated_n, geometry, n_cells, &score_mts, &per_mt_scales)?
    };

    // Coupled photon-production inputs (prompt) and D1S decay-photon inputs are
    // MUTUALLY EXCLUSIVE: a D1S run emits decay photons (decay gate 1, coupled
    // gate 0); a non-D1S coupled run emits prompt photons (coupled gate 1,
    // decay gate 0). Both are built in `geometry.materials` order (the same
    // order `cell_to_material` indexes into).
    let (coupled, decay) = if let Some((decay_data, registry)) = &decay_prep {
        // D1S path: build the decay tables with `n_material_slots` (>=
        // n_materials) so they include the synthetic void slot
        // `translate_for_gpu` appends for void cells, keeping `cell_to_material`
        // indexing in-bounds. The coupled-off / decay-off sizing uses the same
        // slot count so every photon-production buffer matches the neutron
        // buffers' material-row count.
        let decay = build_decay_photon_inputs(
            geometry,
            decay_data,
            registry,
            &inputs.log_energy_grid,
            n_material_slots,
        )?;
        let coupled = yamc_gpu::neutron::transport::CoupledPhotonInputs::coupled_off(
            n_material_slots,
            inputs.log_energy_grid.len(),
        );
        (coupled, decay)
    } else {
        // Prompt-coupled path: pass `n_material_slots` (>= n_materials) so the
        // production tables include the synthetic void slot `translate_for_gpu`
        // appends for void cells, keeping `cell_to_material` indexing in-bounds.
        let coupled =
            build_coupled_photon_inputs(geometry, &inputs.log_energy_grid, n_material_slots)?;
        let decay = yamc_gpu::neutron::transport::DecayPhotonInputs::decay_off(
            n_material_slots,
            inputs.log_energy_grid.len(),
        );
        (coupled, decay)
    };

    // Photon-side translation + pack for the bank-drain sub-pass. Built once;
    // the source SoA is filled per launch from the drained bank.
    let photon_inputs =
        super::translate_photon::translate_photon_for_gpu(model, seed_sample, settings.seed)?;
    let mut pack_p = if validated_p.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated_p, geometry, n_cells, &[], &[])?
    };
    // Photon kernel reads the raw MT out of `score_data` (refinement A), same
    // convention as `run_on_gpu_photon`.
    if !validated_p.is_empty() {
        for (i, v) in validated_p.iter().enumerate() {
            pack_p.score_data[i] = v.score_mt.map(|m| m as u32).unwrap_or(0);
        }
    }

    let max_steps = model.gpu_max_steps_per_particle;
    let survival = survival_inputs(model);
    let ctx = GpuContext::with_device(device)?;
    if model.verbose.summary {
        println!("GPU (coupled n->photon): {}", ctx.adapter_info());
    }

    // Batch-free per-SOURCE variance (issue #233 Stage 3). A source neutron's
    // per-history total spans the neutron launch AND the photon sub-pass that
    // drains its banked secondary/decay photons, so both scatter into the same
    // source-neutron row of a per-source accumulator (keyed by the source index
    // carried through the bank). Finalize per SOURCE NEUTRON: neutron-only tallies
    // from the neutron pass, photon-only from the photon pass, dual (unfiltered)
    // from their per-source SUM (added before squaring, capturing the
    // shared-source correlation the batch-means path handled by per-batch summing).
    let total_out_len_n = pack_n.total_out_len() as usize;
    let total_out_len_p = pack_p.total_out_len() as usize;
    let k = yamc_gpu::neutron::transport::PERHIST_K as usize;
    let per_thread_words = total_out_len_n
        .max(total_out_len_p)
        .max(total_out_len_n.saturating_sub(k))
        .max(total_out_len_p.saturating_sub(k))
        .max(1);
    let mem_safe_max = ((64usize * 1024 * 1024) / per_thread_words).max(1);
    let chunk = launch_chunk_size(mem_safe_max);

    // Per-final-tally accumulators (tally 7D bin order), summed across chunks.
    let mut sum_n: Vec<Vec<f64>> = validated_n[..n_neutron_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_n: Vec<Vec<f64>> = validated_n[..n_neutron_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sum_p: Vec<Vec<f64>> = validated_p[..n_photon_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_p: Vec<Vec<f64>> = validated_p[..n_photon_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    // Dual accumulators are parallel to the dual PACK entries (one per
    // (tally, score)), matching `sum_n` / `sum_p` above, so the shared
    // `finalize_per_history_tallies` sees the same layout on every pass.
    let mut sum_d: Vec<Vec<f64>> = validated_n[n_neutron_only..]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_d: Vec<Vec<f64>> = validated_n[n_neutron_only..]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let flat_scales_n = flat_bin_scales(&pack_n);
    let flat_scales_p = flat_bin_scales(&pack_p);
    // Issue #234: a pass carrying a mesh tally routes to the per-source DIRECT
    // mode (the kernel fans each voxel crossing straight into `src_acc`, no
    // touched-list). `PerSource` and `PerSourceDirect` share the same `src_acc`
    // row layout, so the per-source fold below is identical either way.
    let has_mesh_n = pack_n.mesh_kind.iter().any(|&k| k != MESH_NONE);
    let has_mesh_p = pack_p.mesh_kind.iter().any(|&k| k != MESH_NONE);

    let mut alive_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut n_steps_all: Vec<u32> = Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut final_energies_all: Vec<f64> =
        Vec::with_capacity(settings.total_particles.unwrap_or(0));
    let mut last_n_cells = n_cells;
    let mut n_hist_total: u64 = 0;

    // One chunk = a coupled neutron launch plus the full secondary/decay photon
    // drain for those source neutrons. `max_runtime` is checked only at this
    // outer boundary (never mid-drain), so each source row folds completely.
    // Lost-particle bookkeeping across this run's launches (issue #289).
    let mut lost = LostTracker::default();
    // Partition the launch chunks across MPI ranks (issue #303): before this the
    // GPU path had no MPI awareness, so every rank transported the full
    // `total_particles` and rank 0 reported its own result -- n x the work, and
    // slower than serial because the ranks contend for one device.
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    let mut sched = LaunchLoop::new_for_rank(
        settings,
        chunk,
        mpi_ctx.rank() as usize,
        mpi_ctx.size() as usize,
    );
    while let Some((chunk_idx, chunk_sources)) = sched.next() {
        let (seeds, energies, positions, directions) =
            super::translate::sample_initial_particles_for_chunk(
                model,
                chunk_sources,
                chunk_idx,
                chunk,
                settings.seed,
            );
        inputs.seeds = seeds;
        inputs.energies = energies;
        inputs.positions = positions;
        inputs.directions = directions;

        let bank_capacity = chunk_sources
            .saturating_mul(COUPLED_PHOTONS_PER_NEUTRON)
            .max(1);
        // Neutron launch (per-source): scores neutron_pass tallies into src_acc,
        // banks secondary/decay photons stamped with the source neutron index.
        // A mesh in the neutron pass switches it to the direct variant so the
        // kernel scores the voxel dimension (issue #234).
        let variance_n = per_source_variance(
            has_mesh_n,
            chunk_sources as u32,
            total_out_len_n as u32,
            None,
        );
        let kernel = run_coupled_kernel_path(
            &ctx,
            &inputs,
            &pack_n,
            &xs_score_per_mt,
            &survival,
            &coupled,
            &decay,
            bank_capacity,
            max_steps,
            variance_n,
        )?;
        last_n_cells = kernel.n_cells;
        if kernel.bank_overflow > 0 {
            return Err(GpuDispatchError::PhotonBankOverflow {
                overflow: kernel.bank_overflow,
                count: kernel.bank_count,
                capacity: bank_capacity,
            });
        }
        // The drain below is photon-only, so a banked NEUTRON would be filtered
        // out and silently lost (issue #111 phase 2).
        if kernel.n_spilled_secondaries > 0 {
            return Err(GpuDispatchError::CoupledNxnSpillUnsupported {
                spilled: kernel.n_spilled_secondaries,
            });
        }
        let mut pst_n = vec![0.0f64; chunk_sources * total_out_len_n];
        accumulate_src_acc(&kernel.src_acc, &flat_scales_n, total_out_len_n, &mut pst_n);
        fail_if_truncated(&kernel.alive, max_steps)?;
        alive_all.extend(kernel.alive);
        n_steps_all.extend(kernel.n_steps);
        final_energies_all.extend(kernel.final_energies);

        // Photon sub-pass (PerSource): drain the bank in sub-launches of at most
        // `chunk` photons so the per-thread spill stays bounded; each photon
        // carries its source neutron's index (from `bank_source_idx`), scattering
        // into `src_acc` keyed by the source neutron.
        let mut pst_p = vec![0.0f64; chunk_sources * total_out_len_p];
        let bank_count = kernel.bank_count as usize;
        let bank_avail = kernel.bank_f64.len() / 8;
        let mut drained = 0usize;
        while drained < bank_count {
            let n_this = (bank_count - drained).min(bank_avail - drained).min(chunk);
            if n_this == 0 {
                break;
            }
            let src_idx: Vec<u32> = kernel.bank_source_idx[drained..drained + n_this].to_vec();
            let variance_p = per_source_variance(
                has_mesh_p,
                chunk_sources as u32,
                total_out_len_p as u32,
                Some(&src_idx),
            );
            let pres = run_photon_subpass(
                &ctx,
                &photon_inputs,
                &kernel.bank_f64[drained * 8..],
                &kernel.bank_u32[drained * 4..],
                n_this,
                &pack_p,
                max_steps,
                variance_p,
            );
            lost.absorb(
                &pres.lost,
                ParticleType::Photon,
                &csg_geometry(model).cells,
                model.max_lost_particles,
            )?;
            accumulate_src_acc(&pres.src_acc, &flat_scales_p, total_out_len_p, &mut pst_p);
            drained += n_this;
        }

        // Fold this chunk's per-source totals into the tally accumulators.
        for (ti, v) in validated_n[..n_neutron_only].iter().enumerate() {
            let off = pack_n.out_offsets[ti] as usize;
            let nk = pack_n.out_offsets[ti + 1] as usize - off;
            fold_per_source_tally(
                v,
                &pst_n,
                chunk_sources,
                total_out_len_n,
                off,
                nk,
                &mut sum_n[ti],
                &mut sq_n[ti],
            );
        }
        for (ti, v) in validated_p[..n_photon_only].iter().enumerate() {
            let off = pack_p.out_offsets[ti] as usize;
            let nk = pack_p.out_offsets[ti + 1] as usize - off;
            fold_per_source_tally(
                v,
                &pst_p,
                chunk_sources,
                total_out_len_p,
                off,
                nk,
                &mut sum_p[ti],
                &mut sq_p[ti],
            );
        }
        for d in 0..(validated_n.len() - n_neutron_only) {
            let vi = &validated_n[n_neutron_only + d];
            let off_n = pack_n.out_offsets[n_neutron_only + d] as usize;
            let off_p = pack_p.out_offsets[n_photon_only + d] as usize;
            let nk = pack_n.out_offsets[n_neutron_only + d + 1] as usize - off_n;
            fold_per_source_dual(
                vi,
                &pst_n,
                total_out_len_n,
                off_n,
                &pst_p,
                total_out_len_p,
                off_p,
                chunk_sources,
                nk,
                &mut sum_d[d],
                &mut sq_d[d],
            );
        }
        n_hist_total += chunk_sources as u64;

        if sched.hit_time_budget() {
            break;
        }
    }

    // Finalize + install (N = source-neutron count). Dual tallies install through
    // their neutron-pass ValidatedTally entries (which wrap the same Tally).
    finalize_per_history_tallies(&validated_n[..n_neutron_only], &sum_n, &sq_n, n_hist_total);
    finalize_per_history_tallies(&validated_p[..n_photon_only], &sum_p, &sq_p, n_hist_total);
    finalize_per_history_tallies(&validated_n[n_neutron_only..], &sum_d, &sq_d, n_hist_total);
    Ok(GpuRunResult {
        n_particles: n_hist_total as usize,
        n_cells: last_n_cells,
        alive: alive_all,
        n_steps: n_steps_all,
        final_energies: final_energies_all,
        lost_count: lost.count,
        lost: lost.records,
        // Photon / coupled / mixed passes refuse an (n,xn) spill outright
        // (`CoupledNxnSpillUnsupported`), so reaching here means none happened.
        n_spilled_secondaries: 0,
    })
}

/// Mixed neutron+photon PRIMARY source (issue #58, E13b). macOS has no f64 GPU
/// path, so this mirrors the other dispatch entries by returning the same error
/// `GpuContext::new()` would.
#[cfg(target_os = "macos")]
fn run_on_gpu_mixed(
    _model: &mut Model,
    _device: Option<&str>,
    _settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    Err(GpuDispatchError::GpuUnavailable(
        yamc_gpu::GpuInitError::NoF64Adapter,
    ))
}

/// Transport a model whose source list mixes neutron-emitting and
/// photon-emitting sources (issue #58, E13b).
///
/// On the CPU each history is one source particle, drawn from the source list
/// in proportion to source strength (`Model::sample_source`), then transported
/// with its own particle physics; tallies are per-source-particle. The GPU has
/// separate neutron and photon kernels, so this splits the run: of every
/// `nb` source particles in a batch, `n_per_batch_n = nb * S_n/(S_n+S_p)` are
/// drawn from the neutron sources and transported by the neutron kernel, and
/// the remaining `n_per_batch_p` from the photon sources by the photon kernel.
/// Both per-type counts are <= `nb` <= the #93 watchdog cap, so neither
/// dispatch over-runs.
///
/// Normalisation is the key: each batch is one realisation of `batch_total =
/// nb` source particles, so EVERY tally's per-batch contribution is divided by
/// `batch_total` (not by the per-type count). A neutron-filtered tally then
/// reads the neutron pass alone (photons contribute zero to it), a
/// photon-filtered tally the photon pass alone, and an unfiltered (all-particle)
/// flux / total / heating tally the per-batch SUM of both passes -- the same
/// per-source-particle quantity the CPU reports. The batch-means Welford over
/// `n_batches` then yields per-source-particle means and a valid variance.
///
/// To match the CPU reference, the neutron share runs through the COUPLED kernel
/// so it emits secondary photons (capture / inelastic gammas) into a bank, and
/// the photon pass transports BOTH that secondary bank AND the primary photon
/// source. Any photon-source model produces neutron-induced gammas whenever
/// photon transport is on (a photon source requires it); the CPU auto-enables
/// `transport_secondary_photons` for such a model (`run_internal`), so the mixed
/// photon tally is primary + secondary, not primary alone. D1S decay photons on
/// top of a mixed source are not supported yet.
#[cfg(not(target_os = "macos"))]
fn run_on_gpu_mixed(
    model: &mut Model,
    device: Option<&str>,
    settings: &crate::model::TransportSettings,
) -> Result<GpuRunResult, GpuDispatchError> {
    if model.use_decay_photons {
        return Err(GpuDispatchError::MixedSourceWithSecondariesUnsupported);
    }

    // Two sub-models sharing the same (Arc-backed) geometry/materials, each
    // carrying only its own particle type's sources so the per-type translation
    // never sees a foreign source. The photon sub-model's seed is perturbed so
    // the two source samplers draw decorrelated ChaCha streams.
    let mut neutron_model = model.clone();
    neutron_model.sources = model
        .sources
        .iter()
        .filter(|s| s.particle_type() == ParticleType::Neutron)
        .cloned()
        .collect();
    let mut photon_model = model.clone();
    photon_model.sources = model
        .sources
        .iter()
        .filter(|s| s.particle_type() == ParticleType::Photon)
        .cloned()
        .collect();
    let neutron_seed = settings.seed;
    let photon_seed = settings.seed ^ 0x5DEE_CE66_D1F5_3A1B;

    let s_n: f64 = neutron_model.sources.iter().map(|s| s.strength()).sum();
    let s_p: f64 = photon_model.sources.iter().map(|s| s.strength()).sum();
    let s_total = s_n + s_p;
    if s_total <= 0.0 || settings.total_particles == Some(0) {
        return Err(GpuDispatchError::Translate(
            super::error::GpuTranslateError::NoParticles,
        ));
    }

    // Tally routing shared with the coupled path: ParticleType(Neutron) ->
    // neutron pass, ParticleType(Photon) -> photon pass, unfiltered
    // flux/total/heating -> dual (scored by both, summed per source particle).
    let (neutron_only, photon_only, dual) = classify_coupled_tallies(&model.tallies)?;
    // Boundaries into the PACK, which `validate_tallies` builds one entry per
    // (tally, score) -- so where the pass's own tallies end and the appended
    // dual ones begin is a score count, not a tally count (issue #271).
    let n_neutron_only: usize = neutron_only.iter().map(|t| t.scores.len()).sum();
    let n_photon_only: usize = photon_only.iter().map(|t| t.scores.len()).sum();
    let neutron_pass: Vec<Arc<Tally>> = neutron_only.iter().chain(dual.iter()).cloned().collect();
    let photon_pass: Vec<Arc<Tally>> = photon_only.iter().chain(dual.iter()).cloned().collect();
    let validated_n = validate_tallies(&neutron_pass, ParticleType::Neutron)?;
    let validated_p = validate_tallies(&photon_pass, ParticleType::Photon)?;

    // Provisional per-type sample sizes for translate; the per-source loop below
    // re-samples each launch chunk with a FIXED strength split (total-independent).
    let seed_sample = INITIAL_TRANSLATE_SAMPLE;
    let sample_n =
        (((seed_sample as f64) * (s_n / s_total)).round() as usize).clamp(1, seed_sample - 1);
    let sample_p = seed_sample - sample_n;

    // Neutron side: coupled-kernel inputs (secondary photons ON) + per-MT XS +
    // pack + survival. `geometry` is shared by both packs.
    let mut n_inputs = translate_for_gpu(&neutron_model, sample_n, neutron_seed)?;
    let geometry = csg_geometry(model);
    let n_cells = n_inputs.cell_aabbs.len() / 6;
    let score_mts = collect_score_mts(&validated_n);
    let n_material_slots = n_inputs.target_mass_per_material.len();
    let xs_score_per_mt = build_xs_score_per_mt(
        &geometry.materials,
        &score_mts,
        &n_inputs.log_energy_grid,
        n_material_slots,
    )?;
    let sigma_t_score_grid = build_sigma_t_score_grid(
        &geometry.materials,
        &score_mts,
        &n_inputs.log_energy_grid,
        n_material_slots,
    )?;
    let per_mt_scales = per_mt_fixed_point_scales(
        &score_mts,
        &xs_score_per_mt,
        &sigma_t_score_grid,
        n_material_slots,
        n_inputs.log_energy_grid.len(),
    );
    let pack_n = if validated_n.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated_n, geometry, n_cells, &score_mts, &per_mt_scales)?
    };
    let survival = survival_inputs(model);
    // Prompt secondary-photon production from the neutron share. No D1S here
    // (rejected above), so decay-off. The bank is sized per source neutron.
    let coupled =
        build_coupled_photon_inputs(geometry, &n_inputs.log_energy_grid, n_material_slots)?;
    let decay = yamc_gpu::neutron::transport::DecayPhotonInputs::decay_off(
        n_material_slots,
        n_inputs.log_energy_grid.len(),
    );
    // Photon side: one translation drives BOTH the secondary-bank sub-pass and
    // the primary photon-source sub-pass (same geometry / photon data). The
    // primary pass fills the source SoA per launch; the secondary pass reads the
    // neutron kernel's bank directly.
    let mut p_inputs =
        super::translate_photon::translate_photon_for_gpu(&photon_model, sample_p, photon_seed)?;
    let mut pack_p = if validated_p.is_empty() {
        TalliesPack::dummy_single_bin(n_cells as u32)
    } else {
        build_tallies_pack(&validated_p, geometry, n_cells, &[], &[])?
    };
    if !validated_p.is_empty() {
        for (i, v) in validated_p.iter().enumerate() {
            pack_p.score_data[i] = v.score_mt.map(|m| m as u32).unwrap_or(0);
        }
    }

    let max_steps = model.gpu_max_steps_per_particle;
    let ctx = GpuContext::with_device(device)?;
    if model.verbose.summary {
        println!("GPU (mixed neutron+photon source): {}", ctx.adapter_info());
    }

    // Batch-free per-SOURCE variance (issue #233 Stage 3). Each source particle
    // (neutron OR photon) is one variance sample in a UNIFIED index space:
    // `[0, chunk_n)` are the chunk's source neutrons, `[chunk_n, chunk_total)` its
    // source photons. Secondary photons from a source neutron inherit that
    // neutron's index (through the bank); primary source photons carry the offset
    // index. Neutron-only tallies fold from the neutron pass, photon-only from the
    // (secondary + primary) photon passes, dual from their per-source SUM. N =
    // total source particles.
    let total_out_len_n = pack_n.total_out_len() as usize;
    let total_out_len_p = pack_p.total_out_len() as usize;
    let k = yamc_gpu::neutron::transport::PERHIST_K as usize;
    let per_thread_words = total_out_len_n
        .max(total_out_len_p)
        .max(total_out_len_n.saturating_sub(k))
        .max(total_out_len_p.saturating_sub(k))
        .max(1);
    let mem_safe_max = ((64usize * 1024 * 1024) / per_thread_words).max(1);
    let chunk = launch_chunk_size(mem_safe_max).max(2);
    // Fixed per-type split of the source chunk (total-independent).
    let chunk_n = (((chunk as f64) * (s_n / s_total)).round() as usize).clamp(1, chunk - 1);
    let chunk_p = chunk - chunk_n;

    let mut sum_n: Vec<Vec<f64>> = validated_n[..n_neutron_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_n: Vec<Vec<f64>> = validated_n[..n_neutron_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sum_p: Vec<Vec<f64>> = validated_p[..n_photon_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_p: Vec<Vec<f64>> = validated_p[..n_photon_only]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    // Dual accumulators are parallel to the dual PACK entries (one per
    // (tally, score)), matching `sum_n` / `sum_p` above, so the shared
    // `finalize_per_history_tallies` sees the same layout on every pass.
    let mut sum_d: Vec<Vec<f64>> = validated_n[n_neutron_only..]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let mut sq_d: Vec<Vec<f64>> = validated_n[n_neutron_only..]
        .iter()
        .map(|v| vec![0.0f64; v.block_bins()])
        .collect();
    let flat_scales_n = flat_bin_scales(&pack_n);
    let flat_scales_p = flat_bin_scales(&pack_p);
    // Issue #234: a pass carrying a mesh tally routes to the per-source DIRECT
    // mode so the kernel scores the voxel dimension. Same `src_acc` layout as
    // `PerSource`, so the unified-sample fold below is unchanged.
    let has_mesh_n = pack_n.mesh_kind.iter().any(|&k| k != MESH_NONE);
    let has_mesh_p = pack_p.mesh_kind.iter().any(|&k| k != MESH_NONE);

    let mut alive_all: Vec<u32> = Vec::new();
    let mut n_steps_all: Vec<u32> = Vec::new();
    let mut final_energies_all: Vec<f64> = Vec::new();
    let mut last_n_cells = n_cells;
    let mut n_hist_total: u64 = 0;

    // One chunk = `this_total` unified source particles (neutron + photon
    // shares) plus the secondary-photon drain. `max_runtime` is checked only at
    // this outer boundary (never between the three sub-passes), so each unified
    // source sample's contributions fold completely.
    // Lost-particle bookkeeping across this run's launches (issue #289).
    let mut lost = LostTracker::default();
    // Partition the launch chunks across MPI ranks (issue #303): before this the
    // GPU path had no MPI awareness, so every rank transported the full
    // `total_particles` and rank 0 reported its own result -- n x the work, and
    // slower than serial because the ranks contend for one device.
    let mpi_ctx = crate::mpi_context::MpiContext::init();
    let mut sched = LaunchLoop::new_for_rank(
        settings,
        chunk,
        mpi_ctx.rank() as usize,
        mpi_ctx.size() as usize,
    );
    while let Some((chunk_idx, this_total)) = sched.next() {
        // Split this (possibly partial) chunk into its neutron / photon shares by
        // strength. Full chunks get exactly `chunk_n` / `chunk_p`
        // (`round(chunk * s_n/s_total) == chunk_n`); a partial last chunk splits
        // proportionally so the neutron:photon sample ratio stays correct (a plain
        // `chunk_n.min(this_total)` would make a small final chunk all-neutron).
        let this_n = if this_total == chunk {
            chunk_n
        } else {
            (((this_total as f64) * (s_n / s_total)).round() as usize).min(this_total)
        };
        let this_p = this_total - this_n;
        let chunk_total = this_total; // src_acc row count (unified samples)

        // --- Neutron pass (coupled): source neutrons are unified samples
        //     `[0, this_n)`; identity source_idx. Banks secondaries stamped with
        //     the neutron index. ---
        {
            let (seeds, energies, positions, directions) =
                super::translate::sample_initial_particles_for_chunk(
                    &neutron_model,
                    this_n,
                    chunk_idx,
                    chunk_n,
                    neutron_seed,
                );
            n_inputs.seeds = seeds;
            n_inputs.energies = energies;
            n_inputs.positions = positions;
            n_inputs.directions = directions;
        }
        let bank_capacity = this_n.saturating_mul(COUPLED_PHOTONS_PER_NEUTRON).max(1);
        let mut pst_n = vec![0.0f64; chunk_total * total_out_len_n];
        let mut pst_p = vec![0.0f64; chunk_total * total_out_len_p];
        let (mut bank_f64, mut bank_u32, mut bank_src, mut bank_count) =
            (Vec::new(), Vec::new(), Vec::new(), 0usize);
        if this_n > 0 {
            let kernel = run_coupled_kernel_path(
                &ctx,
                &n_inputs,
                &pack_n,
                &xs_score_per_mt,
                &survival,
                &coupled,
                &decay,
                bank_capacity,
                max_steps,
                per_source_variance(has_mesh_n, chunk_total as u32, total_out_len_n as u32, None),
            )?;
            last_n_cells = kernel.n_cells;
            if kernel.bank_overflow > 0 {
                return Err(GpuDispatchError::PhotonBankOverflow {
                    overflow: kernel.bank_overflow,
                    count: kernel.bank_count,
                    capacity: bank_capacity,
                });
            }
            // Photon-only drain below: a banked NEUTRON would be filtered out
            // and silently lost (issue #111 phase 2).
            if kernel.n_spilled_secondaries > 0 {
                return Err(GpuDispatchError::CoupledNxnSpillUnsupported {
                    spilled: kernel.n_spilled_secondaries,
                });
            }
            accumulate_src_acc(&kernel.src_acc, &flat_scales_n, total_out_len_n, &mut pst_n);
            fail_if_truncated(&kernel.alive, max_steps)?;
            alive_all.extend(kernel.alive);
            n_steps_all.extend(kernel.n_steps);
            final_energies_all.extend(kernel.final_energies);
            bank_f64 = kernel.bank_f64;
            bank_u32 = kernel.bank_u32;
            bank_src = kernel.bank_source_idx;
            bank_count = kernel.bank_count as usize;
        }

        // --- Photon pass (a): the neutron-induced secondary photons, folded into
        //     their SOURCE NEUTRON's unified sample (drained in <= chunk sub-launches). ---
        let bank_avail = bank_f64.len() / 8;
        let mut drained = 0usize;
        while drained < bank_count {
            let n_this = (bank_count - drained).min(bank_avail - drained).min(chunk);
            if n_this == 0 {
                break;
            }
            let src_idx: Vec<u32> = bank_src[drained..drained + n_this].to_vec();
            let pres = run_photon_subpass(
                &ctx,
                &p_inputs,
                &bank_f64[drained * 8..],
                &bank_u32[drained * 4..],
                n_this,
                &pack_p,
                max_steps,
                per_source_variance(
                    has_mesh_p,
                    chunk_total as u32,
                    total_out_len_p as u32,
                    Some(&src_idx),
                ),
            );
            lost.absorb(
                &pres.lost,
                ParticleType::Photon,
                &csg_geometry(model).cells,
                model.max_lost_particles,
            )?;
            accumulate_src_acc(&pres.src_acc, &flat_scales_p, total_out_len_p, &mut pst_p);
            drained += n_this;
        }

        // --- Photon pass (b): primary source photons are unified samples
        //     `[this_n, chunk_total)`; their source_idx is offset by `this_n`. ---
        if this_p > 0 {
            let (seeds, energies, positions, directions) =
                super::translate::sample_initial_particles_for_chunk(
                    &photon_model,
                    this_p,
                    chunk_idx,
                    chunk_p,
                    photon_seed,
                );
            // Offset the PCG seed band so it can't overlap the neutron pass's.
            p_inputs.seeds = seeds.iter().map(|s| s.wrapping_add(0x9E37_79B9)).collect();
            p_inputs.energies = energies;
            p_inputs.positions = positions;
            p_inputs.directions = directions;
            let weights = vec![1.0_f64; this_p];
            let parent_ids = vec![0u32; this_p];
            let primary_src_idx: Vec<u32> = (0..this_p as u32).map(|i| this_n as u32 + i).collect();
            let primary = run_photon_primary_pass(
                &ctx,
                &p_inputs,
                &weights,
                &parent_ids,
                &pack_p,
                max_steps,
                per_source_variance(
                    has_mesh_p,
                    chunk_total as u32,
                    total_out_len_p as u32,
                    Some(&primary_src_idx),
                ),
            );
            lost.absorb(
                &primary.lost,
                ParticleType::Photon,
                &csg_geometry(model).cells,
                model.max_lost_particles,
            )?;
            last_n_cells = primary.n_cells;
            accumulate_src_acc(
                &primary.src_acc,
                &flat_scales_p,
                total_out_len_p,
                &mut pst_p,
            );
            fail_if_truncated(&primary.alive, max_steps)?;
            alive_all.extend(primary.alive);
            n_steps_all.extend(primary.n_steps);
            final_energies_all.extend(primary.final_energies);
        }

        // Fold this chunk's per-source totals over the unified samples.
        for (ti, v) in validated_n[..n_neutron_only].iter().enumerate() {
            let off = pack_n.out_offsets[ti] as usize;
            let nk = pack_n.out_offsets[ti + 1] as usize - off;
            fold_per_source_tally(
                v,
                &pst_n,
                chunk_total,
                total_out_len_n,
                off,
                nk,
                &mut sum_n[ti],
                &mut sq_n[ti],
            );
        }
        for (ti, v) in validated_p[..n_photon_only].iter().enumerate() {
            let off = pack_p.out_offsets[ti] as usize;
            let nk = pack_p.out_offsets[ti + 1] as usize - off;
            fold_per_source_tally(
                v,
                &pst_p,
                chunk_total,
                total_out_len_p,
                off,
                nk,
                &mut sum_p[ti],
                &mut sq_p[ti],
            );
        }
        for d in 0..(validated_n.len() - n_neutron_only) {
            let vi = &validated_n[n_neutron_only + d];
            let off_n = pack_n.out_offsets[n_neutron_only + d] as usize;
            let off_p = pack_p.out_offsets[n_photon_only + d] as usize;
            let nk = pack_n.out_offsets[n_neutron_only + d + 1] as usize - off_n;
            fold_per_source_dual(
                vi,
                &pst_n,
                total_out_len_n,
                off_n,
                &pst_p,
                total_out_len_p,
                off_p,
                chunk_total,
                nk,
                &mut sum_d[d],
                &mut sq_d[d],
            );
        }
        n_hist_total += chunk_total as u64;

        if sched.hit_time_budget() {
            break;
        }
    }

    finalize_per_history_tallies(&validated_n[..n_neutron_only], &sum_n, &sq_n, n_hist_total);
    finalize_per_history_tallies(&validated_p[..n_photon_only], &sum_p, &sq_p, n_hist_total);
    finalize_per_history_tallies(&validated_n[n_neutron_only..], &sum_d, &sq_d, n_hist_total);
    Ok(GpuRunResult {
        n_particles: n_hist_total as usize,
        n_cells: last_n_cells,
        alive: alive_all,
        n_steps: n_steps_all,
        final_energies: final_energies_all,
        lost_count: lost.count,
        lost: lost.records,
        // Photon / coupled / mixed passes refuse an (n,xn) spill outright
        // (`CoupledNxnSpillUnsupported`), so reaching here means none happened.
        n_spilled_secondaries: 0,
    })
}

/// `(neutron-only tallies, photon-only tallies, dual tallies)` from
/// [`classify_coupled_tallies`]. Dual tallies are scored by both passes.
#[cfg(not(target_os = "macos"))]
type CoupledTallyGroups = (Vec<Arc<Tally>>, Vec<Arc<Tally>>, Vec<Arc<Tally>>);

/// Classify a coupled-run model's tallies into the neutron-pass set, the
/// photon-pass set, and the DUAL set (scored by BOTH passes and summed).
///
/// A tally's `ParticleType` filter selects which particles it scores. **A
/// tally with NO particle filter means ALL particles** -- in a coupled
/// neutron->photon run its flux / total-interaction rate is the SUM of the
/// neutron and photon contributions (the same total-particle quantity OpenMC
/// and the CPU report). Such unfiltered Flux / total tallies go in `dual` and
/// are scored by the neutron kernel AND the photon sub-pass, then summed per
/// batch.
///
/// Routing:
/// - `ParticleType(Photon)` filter -> photon pass only.
/// - `ParticleType(Neutron)` filter -> neutron pass only.
/// - no particle filter (ALL particles):
///   - Flux or total `ReactionRate` -> `dual` (neutron + photon, summed).
///   - a specific reaction MT / production / damage-energy -> neutron pass
///     only: photons contribute zero to a neutron reaction channel, so the
///     single-pass neutron score already IS the all-particle total.
///   - Heating / HeatingLocal -> `dual` (neutron + photon, summed): the
///     all-particle total is the neutron heating (neutron pass) plus the photon
///     heating (photon sub-pass), the same neutron + photon sum the CPU coupled
///     path reports. Both passes score the heating tally with the per-estimator
///     CPU convention (track-length KERMA, collision analog deposit), so the
///     per-batch sum matches the CPU.
///
/// Each returned `Tally` is the same `Arc` from the model. The caller appends
/// `dual` to BOTH the neutron and photon pass lists, so dual tallies validate
/// against each particle type and are scored by each kernel.
/// Does an unfiltered (all-particle) score need the photon pass added to the
/// neutron pass, or is the neutron pass already the all-particle answer?
///
/// Flux and total interact with both species; heating is the neutron + photon
/// sum. A specific reaction MT / production / damage-energy gets no photon
/// contribution, so the neutron-pass score already IS the all-particle total.
#[cfg(not(target_os = "macos"))]
fn score_needs_photon_pass(score: &Score) -> bool {
    match score {
        Score::Flux(_) => true,
        Score::ReactionRate(rr) => rr.mt == Mt::TOTAL,
        // Unfiltered (all-particle) heating is the neutron + photon sum: the
        // neutron pass scores neutron heating, the photon sub-pass scores photon
        // heating (both with the per-estimator CPU convention -- track-length
        // KERMA, collision analog deposit), and the dual fold sums them.
        Score::Heating(_) | Score::HeatingLocal(_) => true,
        _ => false,
    }
}

#[cfg(not(target_os = "macos"))]
fn classify_coupled_tallies(
    tallies: &[Arc<Tally>],
) -> Result<CoupledTallyGroups, GpuDispatchError> {
    let mut neutron = Vec::new();
    let mut photon = Vec::new();
    let mut dual = Vec::new();
    for (idx, t) in tallies.iter().enumerate() {
        let is_photon = t.filters.iter().any(
            |f| matches!(f, Filter::ParticleType(pf) if pf.particle_type == ParticleType::Photon),
        );
        let is_neutron = t.filters.iter().any(
            |f| matches!(f, Filter::ParticleType(pf) if pf.particle_type == ParticleType::Neutron),
        );
        if is_photon {
            photon.push(Arc::clone(t));
        } else if is_neutron {
            neutron.push(Arc::clone(t));
        } else {
            // Unfiltered = ALL particles. The scores decide whether the
            // all-particle total actually needs the photon contribution.
            //
            // Routing is per TALLY (a dual tally is appended to BOTH pass lists
            // and its two contributions summed), so every score on one unfiltered
            // tally has to want the same answer. When they disagree -- e.g.
            // `[flux, (n,gamma)]`, where flux needs the photon pass and the
            // capture rate does not -- there is no single correct routing, and
            // guessing would silently drop the photon half of the flux or score
            // a neutron MT on the photon pass. Reject with the two ways out
            // (issue #271). The CPU has no such constraint: it scores each
            // particle into whichever scores apply as it goes.
            let mut want = t.scores.iter().map(score_needs_photon_pass);
            let first = want.next().unwrap_or(false);
            if !want.all(|w| w == first) {
                let names: Vec<String> =
                    t.scores.iter().map(|s| format!("{:?}", s.kind())).collect();
                return Err(GpuDispatchError::UnsupportedTallyScore {
                    tally_index: idx,
                    reason: format!(
                        "on a coupled (neutron + photon) run, an all-particle tally's scores \
                         must agree on whether the photon pass contributes, but this tally \
                         mixes both kinds ({}). Add a `particle=` filter to say which species \
                         you mean, or split it into one tally per kind",
                        names.join(", ")
                    ),
                });
            }
            if first {
                dual.push(Arc::clone(t));
            } else {
                neutron.push(Arc::clone(t));
            }
        }
    }
    Ok((neutron, photon, dual))
}

/// Build one `GpuPhotonProductionXs` per material (in `geometry.materials`
/// order -- the same order `cell_to_material` indexes) and pack them into the
/// flat coupled inputs the neutron kernel reads at the photon-emission site.
/// Mirrors `build_xs_score_per_mt`'s per-material nuclide+density extraction.
#[cfg(not(target_os = "macos"))]
fn build_coupled_photon_inputs(
    geometry: &Geometry,
    log_energy_grid: &[f64],
    n_material_slots: usize,
) -> Result<yamc_gpu::neutron::transport::CoupledPhotonInputs, GpuDispatchError> {
    use yamc_gpu::neutron::transport::CoupledPhotonInputs;
    use yamc_gpu::neutron::xs::photon_production::{
        extract_photon_production_xs, GpuPhotonProductionXs,
    };

    let mut tables = Vec::with_capacity(n_material_slots);
    for material in &geometry.materials {
        let atoms_per_bcm = material
            .get_atoms_per_barn_cm()
            .map_err(|e| coupled_xs_error(material, e.to_string()))?;
        let mat_name = material
            .name
            .clone()
            .unwrap_or_else(|| format!("material_{:?}", material.material_id));
        let mut weighted: Vec<(&yamc_nuclide::Nuclide, f64)> = Vec::new();
        for (name, density) in &atoms_per_bcm {
            let Some(nuclide_arc) = material.nuclide_data.get(name) else {
                return Err(GpuDispatchError::Translate(
                    super::error::GpuTranslateError::NoLoadedTemperature {
                        nuclide: name.clone(),
                        material: mat_name.clone(),
                    },
                ));
            };
            weighted.push((nuclide_arc.as_ref(), *density));
        }
        let table =
            extract_photon_production_xs(&weighted, material.temperature(), log_energy_grid)
                .map_err(|e| coupled_xs_error(material, e.to_string()))?;
        tables.push(table);
    }
    // Append a non-emitting void slot per synthetic void material the
    // neutron translation added (`n_material_slots` is the material-row
    // count the kernel sees; `geometry.materials` covers only the real
    // ones). The neutron kernel never samples it -- emission is gated
    // behind `collide_first`, false in a void cell -- but the per-
    // material offset tables must still be in-bounds for the void index.
    for _ in geometry.materials.len()..n_material_slots {
        tables.push(GpuPhotonProductionXs::void(log_energy_grid.len()));
    }
    Ok(CoupledPhotonInputs::from_materials(&tables))
}

/// Linearly interpolate `src` (defined on the nuclide's `src_grid`, linear E)
/// onto `master_grid` (linear E). Mirrors the kernel's linear-E interpolation
/// and the CPU `interp_value` / `Material::lookup_xs` convention. Below/above
/// the source grid the endpoints are held (the source grids span the relevant
/// neutron energy range; D1S yields are zero in the thermal tail anyway).
#[cfg(not(target_os = "macos"))]
fn resample_linear_e(src: &[f64], src_grid: &[f64], master_grid: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; master_grid.len()];
    if src.is_empty() || src_grid.len() != src.len() {
        return out;
    }
    for (o, &e) in out.iter_mut().zip(master_grid.iter()) {
        // partition_point: first index with src_grid[i] > e.
        let hi = src_grid.partition_point(|&x| x <= e);
        if hi == 0 {
            *o = src[0];
        } else if hi >= src_grid.len() {
            *o = src[src.len() - 1];
        } else {
            let e_lo = src_grid[hi - 1];
            let e_hi = src_grid[hi];
            let denom = e_hi - e_lo;
            let frac = if denom > 0.0 { (e - e_lo) / denom } else { 0.0 };
            *o = src[hi - 1] + (src[hi] - src[hi - 1]) * frac;
        }
    }
    out
}

/// Build the per-material D1S decay-photon tables for the GPU neutron kernel,
/// in `geometry.materials` order (the same order `cell_to_material` indexes).
///
/// For each material, every nuclide with precomputed `DecayPhotonNuclideData`
/// contributes its channels: each channel's reaction XS and aggregate
/// photon-production XS are resampled onto the shared `log_energy_grid` and
/// scaled by the nuclide's atom density `N_n` (atoms/barn-cm), giving the
/// macroscopic weighted XS the kernel sums to `y_t`. Each channel carries its
/// parent-nuclide id (`target_id`, the chain emitter) and its discrete decay
/// spectrum as a cumulative-intensity CDF.
///
/// `decay_data` is indexed by `(NuclideId.get() - 1)`; `registry` maps a
/// material nuclide name to its id. The ids match the ones the tally's
/// `ParentNuclideFilter` was resolved against (both come from the registry
/// `prepare_decay_photon_data_for_gpu` built), so the GPU parent tag and the
/// filter bin agree.
#[cfg(not(target_os = "macos"))]
fn build_decay_photon_inputs(
    geometry: &Geometry,
    decay_data: &[Vec<yamc_physics::photon::decay_photon_production::DecayPhotonNuclideData>],
    registry: &yamc_nuclide::nuclide_registry::NuclideRegistry,
    log_energy_grid: &[f64],
    n_material_slots: usize,
) -> Result<yamc_gpu::neutron::transport::DecayPhotonInputs, GpuDispatchError> {
    use yamc_gpu::neutron::transport::{DecayPhotonInputs, MaterialDecayTable};

    let n_grid = log_energy_grid.len();
    let energy_grid: Vec<f64> = log_energy_grid.iter().map(|&le| le.exp()).collect();

    let mut tables = Vec::with_capacity(n_material_slots);
    for material in &geometry.materials {
        let atoms_per_bcm = material
            .get_atoms_per_barn_cm()
            .map_err(|e| coupled_xs_error(material, e))?;

        let mut table = MaterialDecayTable {
            photon_prod: vec![0.0_f64; n_grid],
            ..Default::default()
        };

        for (name, density) in &atoms_per_bcm {
            let Some(id) = registry.lookup(name) else {
                continue;
            };
            let slot = id.get() as usize - 1;
            let Some(temps) = decay_data.get(slot) else {
                continue;
            };
            if temps.is_empty() {
                continue;
            }
            let Some(nuclide_arc) = material.nuclide_data.get(name) else {
                continue;
            };
            let temp_idx = nuclide_arc
                .get_temp_idx(material.temperature())
                .unwrap_or(0)
                .min(temps.len().saturating_sub(1));
            let Some(decay_nuc) = temps.get(temp_idx) else {
                continue;
            };
            let Some(fast_grid) = nuclide_arc.fast_xs.get(temp_idx) else {
                continue;
            };
            let src_grid = &fast_grid.energy;

            for ch in &decay_nuc.channels {
                if ch.yield_constant <= 0.0 || ch.energies.is_empty() {
                    continue;
                }
                // Macroscopic weighted reaction xs on the master grid:
                // N_n * micro_rxn_xs(E) * yield_constant. The kernel sums these
                // rows to the aggregate photon_prod and selects channels by the
                // cumulative of the interpolated values -- matching the CPU
                // `sample_decay_photons` channel walk.
                let micro = resample_linear_e(&ch.xs, src_grid, &energy_grid);
                let mut row = vec![0.0_f64; n_grid];
                for (r, &m) in row.iter_mut().zip(micro.iter()) {
                    let v = m * density * ch.yield_constant;
                    *r = v;
                }
                for (g, &v) in row.iter().enumerate() {
                    table.photon_prod[g] += v;
                }

                // Cumulative intensity CDF (normalized to 1.0 at the last line).
                let total: f64 = ch.intensities.iter().sum();
                if total <= 0.0 {
                    continue;
                }
                let e_base = table.ch_energies.len() as u32;
                let mut cum = 0.0_f64;
                for (&e, &inten) in ch.energies.iter().zip(ch.intensities.iter()) {
                    cum += inten;
                    table.ch_energies.push(e);
                    table.ch_intensity_cdf.push(cum / total);
                }
                let e_count = ch.energies.len() as u32;

                table.ch_xs.extend_from_slice(&row);
                table.ch_parent_id.push(ch.target_id.get() as u32);
                table.ch_e_base.push(e_base);
                table.ch_e_count.push(e_count);
            }
        }

        tables.push(table);
    }

    // Append a non-emitting void slot per synthetic void material the neutron
    // translation added (`n_material_slots` is the material-row count the kernel
    // sees; `geometry.materials` covers only the real ones). The neutron kernel
    // never samples it -- decay emission is gated behind a collision, which a
    // `sigma_t = 0` void cell never has -- but the per-material offset tables
    // (`meta`) must still be in-bounds for the void index. Mirrors
    // `build_coupled_photon_inputs`.
    for _ in geometry.materials.len()..n_material_slots {
        tables.push(MaterialDecayTable::void(n_grid));
    }

    Ok(DecayPhotonInputs::from_materials(&tables, n_grid))
}

#[cfg(not(target_os = "macos"))]
fn coupled_xs_error(
    material: &yamc_materials::material::Material,
    reason: String,
) -> GpuDispatchError {
    let mat_name = material
        .name
        .clone()
        .unwrap_or_else(|| format!("material_{:?}", material.material_id));
    GpuDispatchError::Translate(super::error::GpuTranslateError::CellRegionUnsupported {
        cell_id: None,
        reason: format!("coupled photon-production XS extraction for `{mat_name}`: {reason}"),
    })
}

/// Output of the kernel-side launch -- small struct passed back to
/// `run_on_gpu` so it can both fold the per-tally values into the
/// model and return per-particle diagnostics.
///
/// On the non-coupled neutron path the bank fields are empty / zero
/// (`coupled_off` writes no records). On the coupled path
/// (`run_coupled_kernel_path`) they carry the drained secondary-photon bank.
struct KernelOutput {
    n_cells: usize,
    alive: Vec<u32>,
    n_steps: Vec<u32>,
    final_energies: Vec<f64>,
    tally_outputs: Vec<Vec<f64>>,
    // Per-tally per-bin sum-of-squares (batch-free per-history variance, issue
    // #233). Non-empty only for the `PerHistory` (Stage 1) mode; same
    // shape/order as `tally_outputs`. Empty on the per-step / per-source paths.
    tally_sum_sq: Vec<Vec<f64>>,
    // Per-source accumulator raw fixed-point words (issue #233 Stage 2,
    // `PerSource` mode): `chunk_sources * total_out_len`, this launch's
    // per-`(source, flat_bin)` sum. Empty otherwise.
    src_acc: Vec<u64>,
    // Originating source index of each banked fission progeny, by bank slot
    // (issue #233 Stage 2). Empty otherwise.
    bank_source_idx: Vec<u32>,
    // Device particle bank: coupled secondary photons (`run_on_gpu_coupled`)
    // and/or fission progeny (#78, the neutron path's generation loop). The
    // macOS `run_kernel_path` stub `unreachable!()`s before constructing a
    // `KernelOutput`, so these are never written on macOS; they are always
    // declared so the always-compiled `run_on_gpu_with_device` reads them
    // without a per-target cfg.
    bank_f64: Vec<f64>,
    bank_u32: Vec<u32>,
    bank_count: u64,
    bank_overflow: u64,
    // (n,xn) secondaries the kernel handed to the bank because a thread's
    // in-thread pending stack was full (issue #111 phase 2). Distinguished from
    // banked photons / fission progeny by the record's `gen` tag, so a coupled
    // pass can tell "the bank holds only photons" from "the bank holds a
    // neutron I am about to drop".
    n_spilled_secondaries: u64,
    // Histories that ended in no cell (issue #289). The caller folds this into
    // its `LostTracker`, which enforces `max_lost_particles`.
    #[cfg(not(target_os = "macos"))]
    lost: yamc_gpu::common::lost_particles::LostParticleResult,
}

/// Coupled-path kernel launch: the non-coupled launch with photon emission
/// enabled. Thin wrapper over [`run_kernel_path`] that passes the populated
/// [`yamc_gpu::neutron::transport::CoupledPhotonInputs`] + bank capacity
/// instead of the coupled-off defaults. Returns the same [`KernelOutput`],
/// whose bank fields now carry the produced secondary photons.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_coupled_kernel_path(
    ctx: &GpuContext,
    inputs: &super::translate::GpuTransportInputs,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    coupled: &yamc_gpu::neutron::transport::CoupledPhotonInputs,
    decay: &yamc_gpu::neutron::transport::DecayPhotonInputs,
    photon_bank_capacity: usize,
    max_steps: u32,
    // Tally-variance mode (issue #233 Stage 3): `PerSource` runs the neutron
    // kernel keyed by source neutron and stamps `bank_source_idx` on every banked
    // secondary/decay photon so the photon sub-pass folds into the same sample.
    variance: TallyVarianceMode<'_>,
) -> Result<KernelOutput, GpuDispatchError> {
    use yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs;
    run_kernel_path_impl(
        ctx,
        inputs,
        pack,
        xs_score_per_mt,
        survival,
        coupled,
        decay,
        // The coupled photon path does not (yet) also branch the fission chain;
        // keep the legacy fission terminator.
        &FissionBankInputs::off(),
        photon_bank_capacity,
        max_steps,
        variance,
    )
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_kernel_path(
    ctx: &GpuContext,
    inputs: &super::translate::GpuTransportInputs,
    _n_particles: usize,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    fission_bank: &yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs,
    fission_bank_capacity: usize,
    max_steps: u32,
    // Tally-variance mode (issue #233): `PerHistory` (non-fissile Stage 1),
    // `PerSource` (fissile Stage 2), or `PerStep`.
    variance: TallyVarianceMode<'_>,
) -> Result<KernelOutput, GpuDispatchError> {
    use yamc_gpu::neutron::transport::{CoupledPhotonInputs, DecayPhotonInputs};
    // Coupled-off + decay-off: byte-identical to a neutron-only run (no photon
    // RNG, no bank writes). The non-coupled neutron dispatch path goes here.
    // When `fission_bank` is on, the SHARED device bank instead collects the
    // fission progeny (capacity `fission_bank_capacity`); when off, a size-1
    // bank suffices.
    let coupled = CoupledPhotonInputs::coupled_off(
        inputs.target_mass_per_material.len(),
        inputs.log_energy_grid.len(),
    );
    let decay = DecayPhotonInputs::decay_off(
        inputs.target_mass_per_material.len(),
        inputs.log_energy_grid.len(),
    );
    run_kernel_path_impl(
        ctx,
        inputs,
        pack,
        xs_score_per_mt,
        survival,
        &coupled,
        &decay,
        fission_bank,
        fission_bank_capacity,
        max_steps,
        variance,
    )
}

/// Shared neutron-kernel launch. The single `coupled` + `photon_bank_capacity`
/// pair selects coupled-off (neutron-only, byte-identical to today) or
/// coupled-on (secondary photon emission). Both callers route through here so
/// the ~100-argument `run_multi_cell_transport` call lives in exactly one
/// place.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_kernel_path_impl(
    ctx: &GpuContext,
    inputs: &super::translate::GpuTransportInputs,
    pack: &TalliesPack,
    xs_score_per_mt: &[f64],
    survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    coupled: &yamc_gpu::neutron::transport::CoupledPhotonInputs,
    decay: &yamc_gpu::neutron::transport::DecayPhotonInputs,
    fission_bank: &yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs,
    photon_bank_capacity: usize,
    max_steps: u32,
    // Tally-variance mode (issue #233): forwarded to the kernel host, which
    // accumulates per-history sum/sum_sq (PerHistory) or the per-source
    // accumulator (PerSource), or the per-step path (PerStep).
    variance: TallyVarianceMode<'_>,
) -> Result<KernelOutput, GpuDispatchError> {
    use yamc_gpu::neutron::transport::run_multi_cell_transport;

    let result = run_multi_cell_transport(
        ctx,
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
        xs_score_per_mt,
        // Survival biasing (implicit capture). `off()` (gate flag 0.0) keeps
        // the neutron transport byte-identical to the analog kernel; the
        // survival path passes `on(..)` (gate flag 1.0).
        survival,
        // Coupled neutron->photon production. `coupled_off` (gate flag 0)
        // keeps neutron transport byte-identical to a neutron-only run; the
        // coupled path passes `from_materials(..)` (gate flag 1) to emit
        // secondary photons into the device bank.
        coupled,
        // D1S decay-photon production. `decay_off` (gate 0) keeps neutron
        // transport byte-identical; the D1S path passes `from_materials(..)`
        // (gate 1). Mutually exclusive with `coupled`.
        decay,
        // Per-collision nuclide selection (issue #74): per-(material, nuclide)
        // macroscopic totals + AWRs so multi-nuclide materials pick the struck
        // nuclide and use its exact elastic AWR (single-nuclide = no-op).
        &inputs.nuclide_select,
        // Device fission bank (#78). `off()` keeps the legacy `weight *= nu_bar`
        // + cap terminator (byte-identical); the fission-bank path passes
        // `on()` to branch the fission chain into the device bank.
        fission_bank,
        photon_bank_capacity,
        max_steps,
        inputs.free_gas_threshold,
        variance,
    );

    Ok(KernelOutput {
        n_cells: result.n_cells,
        alive: result.alive,
        n_steps: result.n_steps,
        final_energies: result.final_energies,
        tally_outputs: result.tally_outputs,
        tally_sum_sq: result.tally_sum_sq,
        src_acc: result.src_acc,
        bank_source_idx: result.bank_source_idx,
        bank_f64: result.photon_bank.bank_f64,
        bank_u32: result.photon_bank.bank_u32,
        bank_count: result.photon_bank.count,
        bank_overflow: result.photon_bank.overflow,
        n_spilled_secondaries: result.n_spilled_secondaries,
        lost: result.lost,
    })
}

/// Transport a batch of PRIMARY photons (a fresh photon source, not a drained
/// bank) and return the full kernel result. The source SoA
/// (`seeds`/`energies`/`positions`/`directions`) is read from `inputs`; the
/// caller supplies the per-photon `weights` (1.0 for an analog source) and
/// `parent_ids` (0 when there is no D1S parent). Shared by the photon-only
/// dispatch (`run_on_gpu_photon`) and the photon share of the mixed-source
/// dispatch (`run_on_gpu_mixed`) so the ~60-argument kernel call lives in one
/// place.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_photon_primary_pass(
    ctx: &GpuContext,
    inputs: &super::translate_photon::GpuPhotonTransportInputs,
    weights: &[f64],
    parent_ids: &[u32],
    pack: &TalliesPack,
    max_steps: u32,
    variance: TallyVarianceMode<'_>,
) -> yamc_gpu::photon::transport::PhotonMultiCellResult {
    use yamc_gpu::photon::transport::run_multi_cell_photon_transport;
    run_multi_cell_photon_transport(
        ctx,
        &inputs.seeds,
        &inputs.energies,
        weights,
        parent_ids,
        &inputs.positions,
        &inputs.directions,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.xs_total,
        &inputs.xs_coherent,
        &inputs.xs_incoherent,
        &inputs.xs_photoelectric,
        &inputs.xs_pair,
        &inputs.xs_heating,
        &inputs.rayleigh_x2,
        &inputs.rayleigh_cdf,
        &inputs.rayleigh_n_points,
        &inputs.ttb.e_grid_log,
        &inputs.ttb.electron_pdf,
        &inputs.ttb.electron_cdf,
        &inputs.ttb.electron_yield,
        &inputs.ttb.has_data,
        &inputs.doppler.pz_grid,
        &inputs.doppler.electron_pdf,
        &inputs.doppler.binding_energy,
        &inputs.doppler.profile_pdf,
        &inputs.doppler.profile_cdf,
        &inputs.doppler.n_shells,
        &inputs.doppler.has_data,
        &inputs.doppler.subshell_idx,
        &inputs.doppler.subshell_w0,
        &inputs.doppler.subshell_cnt,
        &inputs.iff.x,
        &inputs.iff.s,
        &inputs.iff.n_points,
        &inputs.iff.has_data,
        &inputs.atomic_relaxation.has_data,
        &inputs.atomic_relaxation.n_shells,
        &inputs.atomic_relaxation.binding_energy,
        &inputs.atomic_relaxation.pe_subshell_xs_log,
        &inputs.atomic_relaxation.n_trans,
        &inputs.atomic_relaxation.trans_primary,
        &inputs.atomic_relaxation.trans_secondary,
        &inputs.atomic_relaxation.trans_energy,
        &inputs.atomic_relaxation.trans_cum_prob,
        &inputs.ttb.positron_pdf,
        &inputs.ttb.positron_cdf,
        &inputs.ttb.positron_yield,
        &inputs.pair.has_data,
        &inputs.pair.r_z,
        &inputs.pair.a,
        &inputs.pair.c,
        &inputs.element_select.elem_macro_total,
        &inputs.element_select.mat_elem_meta,
        pack,
        max_steps,
        inputs.photon_cutoff_energy,
        variance,
    )
}

/// Photon sub-pass for the coupled path: drain `count` records from the
/// neutron kernel's secondary-photon bank and transport them through the
/// photon kernel, scoring into `pack` (the photon tallies). Returns the
/// per-tally outputs in `pack` order. The geometry / cross-section / TTB /
/// Doppler / IFF / relaxation / pair buffers come from the photon-side
/// translation (`translate_photon_for_gpu`), identical to the photon-source
/// path; only the source (the drained bank) differs.
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_photon_subpass(
    ctx: &GpuContext,
    inputs: &super::translate_photon::GpuPhotonTransportInputs,
    bank_f64: &[f64],
    bank_u32: &[u32],
    count: usize,
    pack: &TalliesPack,
    max_steps: u32,
    variance: TallyVarianceMode<'_>,
) -> yamc_gpu::photon::transport::PhotonMultiCellResult {
    use yamc_gpu::photon::from_bank::run_photon_transport_from_bank;

    run_photon_transport_from_bank(
        ctx,
        bank_f64,
        bank_u32,
        count,
        &inputs.cell_aabbs,
        &inputs.cell_to_material,
        &inputs.surface_types,
        &inputs.surface_params,
        &inputs.surface_boundaries,
        &inputs.region_program,
        &inputs.log_energy_grid,
        &inputs.xs_total,
        &inputs.xs_coherent,
        &inputs.xs_incoherent,
        &inputs.xs_photoelectric,
        &inputs.xs_pair,
        &inputs.xs_heating,
        &inputs.rayleigh_x2,
        &inputs.rayleigh_cdf,
        &inputs.rayleigh_n_points,
        &inputs.ttb.e_grid_log,
        &inputs.ttb.electron_pdf,
        &inputs.ttb.electron_cdf,
        &inputs.ttb.electron_yield,
        &inputs.ttb.has_data,
        &inputs.doppler.pz_grid,
        &inputs.doppler.electron_pdf,
        &inputs.doppler.binding_energy,
        &inputs.doppler.profile_pdf,
        &inputs.doppler.profile_cdf,
        &inputs.doppler.n_shells,
        &inputs.doppler.has_data,
        &inputs.doppler.subshell_idx,
        &inputs.doppler.subshell_w0,
        &inputs.doppler.subshell_cnt,
        &inputs.iff.x,
        &inputs.iff.s,
        &inputs.iff.n_points,
        &inputs.iff.has_data,
        &inputs.atomic_relaxation.has_data,
        &inputs.atomic_relaxation.n_shells,
        &inputs.atomic_relaxation.binding_energy,
        &inputs.atomic_relaxation.pe_subshell_xs_log,
        &inputs.atomic_relaxation.n_trans,
        &inputs.atomic_relaxation.trans_primary,
        &inputs.atomic_relaxation.trans_secondary,
        &inputs.atomic_relaxation.trans_energy,
        &inputs.atomic_relaxation.trans_cum_prob,
        &inputs.ttb.positron_pdf,
        &inputs.ttb.positron_cdf,
        &inputs.ttb.positron_yield,
        &inputs.pair.has_data,
        &inputs.pair.r_z,
        &inputs.pair.a,
        &inputs.pair.c,
        &inputs.element_select.elem_macro_total,
        &inputs.element_select.mat_elem_meta,
        pack,
        max_steps,
        inputs.photon_cutoff_energy,
        variance,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_kernel_path(
    _ctx: &GpuContext,
    _inputs: &super::translate::GpuTransportInputs,
    _n_particles: usize,
    _pack: &TalliesPack,
    _xs_score_per_mt: &[f64],
    _survival: &yamc_gpu::neutron::survival_biasing::SurvivalBiasingInputs,
    _fission_bank: &yamc_gpu::neutron::fission_bank_inputs::FissionBankInputs,
    _fission_bank_capacity: usize,
    _max_steps: u32,
    _variance: TallyVarianceMode<'_>,
) -> Result<KernelOutput, GpuDispatchError> {
    // Unreachable: `GpuContext::new()` above already returned
    // `Err(NoF64Adapter)` on macOS. Keep a typed return so the
    // public function signature stays platform-uniform.
    unreachable!("GpuContext::new() returns NoF64Adapter on macOS; this path is dead code")
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for the dispatch's pure-host validation and pack
    //! build paths. These exercise the multi-tally / cell-only /
    //! Total / Absorption support without spinning up a GPU context,
    //! so they run in CI even on machines without a Vulkan adapter.
    use super::*;
    use yamc_gpu::common::tallies::{SCORE_FLUX, SCORE_TOTAL};
    use yamc_tallies::filter::cell::CellFilter;
    use yamc_tallies::filter::energy::EnergyFilter;
    use yamc_tallies::filter::particle_type::ParticleTypeFilter;
    use yamc_tallies::tally::Tally;

    fn flux_tally_cell_energy(cell_ids: Vec<u32>, e_bins: Vec<f64>) -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids }),
            Filter::Energy(EnergyFilter::new(e_bins)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        Arc::new(t)
    }

    fn tally_with_score(cell_ids: Vec<u32>, score: Score) -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters = vec![Filter::Cell(CellFilter { cell_ids })];
        t.scores = vec![score];
        Arc::new(t)
    }

    #[test]
    fn validate_accepts_flux_with_cell_and_energy() {
        let t = flux_tally_cell_energy(vec![1], vec![1e-3, 1.0, 1e3]);
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
    }

    #[test]
    fn validate_accepts_cell_only_tally() {
        // Cell-only -- no EnergyFilter -- is valid; the kernel collapses
        // to a single bin spanning [-∞, +∞].
        let t = tally_with_score(vec![7], Score::Flux(yamc_tallies::score::FluxScore));
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn validate_accepts_total_score() {
        let t = tally_with_score(vec![1], "total".parse().unwrap());
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v[0].score_kind, SCORE_TOTAL);
    }

    #[test]
    fn validate_accepts_absorption_score() {
        // Absorption routes through SCORE_PER_MT(27) (tabulated MT 27),
        // NOT the derived SCORE_ABSORPTION fast path -- see #415.
        let t = tally_with_score(vec![1], "absorption".parse().unwrap());
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v[0].score_kind, SCORE_PER_MT);
        assert_eq!(v[0].score_mt, Some(Mt::ABSORPTION.as_i32()));
    }

    #[test]
    fn validate_rejects_unsupported_score() {
        // Photon coherent-scatter is neutron-out-of-scope for the kernel.
        let t = tally_with_score(vec![1], "coherent-scatter".parse().unwrap());
        let err = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap_err();
        match err {
            GpuDispatchError::UnsupportedTallyScore { tally_index, .. } => {
                assert_eq!(tally_index, 0);
            }
            other => panic!("expected UnsupportedTallyScore, got {other:?}"),
        }
    }

    #[test]
    fn run_on_gpu_accepts_survival_biasing() {
        // The neutron kernel supports survival biasing (implicit capture), so
        // it must NOT be rejected by the variance-reduction gate. The empty
        // fixture has no GPU / sources, so the run still errors -- just not
        // with `VarianceReductionUnsupported`. Checked before GPU init, so
        // this runs without hardware.
        let geometry = crate::geometry::Geometry::new(Vec::new(), Vec::new()).unwrap();
        let mut model = Model::new(geometry, Vec::new(), Vec::new());
        model.variance_reduction.push(
            crate::variance_reduction::VarianceReduction::SurvivalBiasing(Default::default()),
        );
        let err = run_on_gpu(&mut model, &crate::model::TransportSettings::default()).unwrap_err();
        assert!(
            !matches!(err, GpuDispatchError::VarianceReductionUnsupported),
            "survival biasing must be accepted by the VR gate, got {err:?}"
        );
    }

    #[test]
    fn validate_accepts_collision_estimator() {
        // Both estimators are supported: a collision-estimator tally
        // validates and the pack carries `is_collision == 1` so the
        // kernel scores it at collision sites (weight × score_xs / Σ_t).
        let mut t = Tally::new();
        t.filters = vec![Filter::Cell(CellFilter { cell_ids: vec![1] })];
        t.scores = vec!["flux".parse().unwrap()];
        t.estimator = yamc_tallies::Estimator::Collision;
        let t = Arc::new(t);
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(validated.len(), 1);

        // A single-cell geometry so the pack builder can map the filter.
        let sphere = Arc::new(crate::geo::Surface {
            surface_id: Some(1),
            kind: crate::geo::SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: crate::geo::BoundaryType::Vacuum,
            name: None,
        });
        let region =
            crate::geo::Region::new_from_halfspace(crate::geo::HalfspaceType::Below(sphere));
        let cell = crate::geometry::cell::Cell::new(Some(1), region, None, None);
        let geometry = crate::geometry::Geometry::new(vec![cell], Vec::new()).unwrap();
        let pack = build_tallies_pack(&validated, &geometry, 1, &[], &[]).unwrap();
        assert_eq!(pack.is_collision, vec![1]);
    }

    #[test]
    fn validate_track_length_pack_is_not_collision() {
        // The default estimator yields `is_collision == 0`.
        let t = tally_with_score(vec![1], "flux".parse().unwrap());
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let sphere = Arc::new(crate::geo::Surface {
            surface_id: Some(1),
            kind: crate::geo::SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: crate::geo::BoundaryType::Vacuum,
            name: None,
        });
        let region =
            crate::geo::Region::new_from_halfspace(crate::geo::HalfspaceType::Below(sphere));
        let cell = crate::geometry::cell::Cell::new(Some(1), region, None, None);
        let geometry = crate::geometry::Geometry::new(vec![cell], Vec::new()).unwrap();
        let pack = build_tallies_pack(&validated, &geometry, 1, &[], &[]).unwrap();
        assert_eq!(pack.is_collision, vec![0]);
    }

    #[test]
    fn validate_accepts_heating_and_production_scores() {
        // KERMA-shape and production scores route through SCORE_PER_MT
        // with the right MT (301 / 901 / 444 / 203..207).
        let h = tally_with_score(vec![1], "heating".parse().unwrap());
        let hl = tally_with_score(vec![1], "heating-local".parse().unwrap());
        let dmg = tally_with_score(vec![1], "damage-energy".parse().unwrap());
        let prod_h1 = tally_with_score(vec![1], "H1-production".parse().unwrap());
        let prod_he4 = tally_with_score(vec![1], "He4-production".parse().unwrap());
        let arr = vec![h, hl, dmg, prod_h1, prod_he4];
        let v = validate_tallies(&arr, ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 5);
        for entry in &v {
            assert_eq!(entry.score_kind, SCORE_PER_MT);
        }
        // Verify each MT mapping.
        assert_eq!(v[0].score_mt, Some(301)); // heating
        assert_eq!(v[1].score_mt, Some(901)); // heating-local
        assert_eq!(v[2].score_mt, Some(444)); // damage-energy
        assert_eq!(v[3].score_mt, Some(203)); // H1-production
        assert_eq!(v[4].score_mt, Some(207)); // He4-production
    }

    #[test]
    fn validate_accepts_per_mt_reaction_rate() {
        // ReactionRate(elastic) used to reject pre-#4; now it routes
        // to SCORE_PER_MT and validates clean.
        let t = tally_with_score(vec![1], "elastic".parse().unwrap());
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_PER_MT);
        // elastic = MT 2
        assert_eq!(v[0].score_mt, Some(2));
    }

    #[test]
    fn validate_collects_distinct_score_mts_in_sorted_order() {
        // Multiple per-MT tallies → score_mts contains each MT once,
        // sorted ascending. Slot index for tally `t` is the position
        // of its MT in the sorted vector.
        let a = tally_with_score(vec![1], "fission".parse().unwrap()); // MT 18
        let b = tally_with_score(vec![1], "elastic".parse().unwrap()); // MT 2
        let c = tally_with_score(vec![1], "elastic".parse().unwrap()); // MT 2 again -- dedup
        let d = tally_with_score(vec![1], "inelastic".parse().unwrap()); // MT 4
        let arr = vec![a, b, c, d];
        let v = validate_tallies(&arr, ParticleType::Neutron).unwrap();
        let score_mts = collect_score_mts(&v);
        assert_eq!(score_mts, vec![2, 4, 18]);
    }

    #[test]
    fn validate_rejects_missing_cell_filter() {
        let mut t = Tally::new();
        t.filters = vec![Filter::Energy(EnergyFilter::new(vec![1e-3, 1.0, 1e3]))];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let err = validate_tallies(std::slice::from_ref(&arc), ParticleType::Neutron).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::UnsupportedTallyFilters { .. }
        ));
    }

    // ---- MaterialFilter on the GPU (issue #271) -------------------------
    //
    // A cell's material is fixed for the run, so the material bin is folded
    // into the kernel's single spatial dimension alongside the cell bin as
    // `cell_bin * n_material_bins + material_bin`. These tests pin that
    // lowering and the writeback that divides it back out.

    /// Material with the given id and no nuclear data (the pack builder only
    /// reads `material_id`).
    fn bare_material(id: u32) -> Arc<yamc_materials::Material> {
        let mut m =
            yamc_materials::Material::new(std::collections::HashMap::new(), "atom", "sum", None)
                .unwrap();
        m.set_material_id(id);
        Arc::new(m)
    }

    /// `cell_specs` is `(cell_id, material_idx)`; every cell is a distinct
    /// unit sphere so the geometry validates.
    fn geometry_with_materials(
        cell_specs: &[(u32, Option<u32>)],
        materials: Vec<Arc<yamc_materials::Material>>,
    ) -> crate::geometry::Geometry {
        let cells = cell_specs
            .iter()
            .enumerate()
            .map(|(i, &(cell_id, material_idx))| {
                let sphere = Arc::new(crate::geo::Surface {
                    surface_id: Some(i + 1),
                    kind: crate::geo::SurfaceKind::Sphere {
                        x0: 10.0 * i as f64,
                        y0: 0.0,
                        z0: 0.0,
                        radius: 1.0,
                    },
                    boundary: crate::geo::BoundaryType::Vacuum,
                    name: None,
                });
                let region = crate::geo::Region::new_from_halfspace(
                    crate::geo::HalfspaceType::Below(sphere),
                );
                crate::geometry::cell::Cell::new(Some(cell_id), region, None, material_idx)
            })
            .collect();
        crate::geometry::Geometry::new(cells, materials).unwrap()
    }

    fn material_tally(material_ids: Vec<u32>, e_bins: Option<Vec<f64>>) -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters = vec![Filter::Material(yamc_tallies::MaterialFilter {
            material_ids,
        })];
        if let Some(bins) = e_bins {
            t.filters.push(Filter::Energy(EnergyFilter::new(bins)));
        }
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        Arc::new(t)
    }

    #[test]
    fn validate_accepts_material_filter_as_the_only_spatial_binner() {
        let t = material_tally(vec![1, 2], None);
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
    }

    #[test]
    fn validate_accepts_material_filter_on_the_photon_pass() {
        // Materials are species-agnostic, so a photon tally binned by material
        // is as valid as a neutron one.
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Material(yamc_tallies::MaterialFilter {
                material_ids: vec![1],
            }),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let t = Arc::new(t);
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Photon).unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn material_filter_maps_each_cell_to_its_material_bin() {
        // Three cells over two materials: cells 1 and 3 share material 10,
        // cell 2 holds material 20. A `materials=[10, 20]` tally has two bins
        // and every cell maps into one of them.
        let t = material_tally(vec![10, 20], None);
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let geometry = geometry_with_materials(
            &[(1, Some(0)), (2, Some(1)), (3, Some(0))],
            vec![bare_material(10), bare_material(20)],
        );
        let pack = build_tallies_pack(&validated, &geometry, 3, &[], &[]).unwrap();
        assert_eq!(pack.n_cells_per_tally, vec![2]);
        assert_eq!(pack.cell_to_bin, vec![0, 1, 0]);
        assert_eq!(pack.out_offsets, vec![0, 2]);
    }

    #[test]
    fn material_filter_excludes_unlisted_and_void_cells() {
        // Only material 20 is filtered on: the material-10 cell and the void
        // cell are both outside the tally, matching the CPU, where a missing
        // material bin skips the score.
        let t = material_tally(vec![20], None);
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let geometry = geometry_with_materials(
            &[(1, Some(0)), (2, Some(1)), (3, None)],
            vec![bare_material(10), bare_material(20)],
        );
        let pack = build_tallies_pack(&validated, &geometry, 3, &[], &[]).unwrap();
        assert_eq!(pack.n_cells_per_tally, vec![1]);
        assert_eq!(pack.cell_to_bin, vec![NOT_IN_TALLY, 0, NOT_IN_TALLY]);
    }

    #[test]
    fn material_id_absent_from_the_geometry_is_an_empty_bin_not_an_error() {
        // The CPU leaves such a bin at zero rather than raising, so the GPU
        // must too -- otherwise the same model would be legal on one backend
        // and rejected on the other.
        let t = material_tally(vec![10, 99], None);
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let geometry = geometry_with_materials(&[(1, Some(0))], vec![bare_material(10)]);
        let pack = build_tallies_pack(&validated, &geometry, 1, &[], &[]).unwrap();
        assert_eq!(
            pack.n_cells_per_tally,
            vec![2],
            "bin 1 (material 99) is kept"
        );
        assert_eq!(pack.cell_to_bin, vec![0]);
    }

    #[test]
    fn cell_and_material_filters_fold_into_product_bins() {
        // Stacking both filters gives `n_cell_bins * n_material_bins` spatial
        // bins with the material innermost -- the CPU's 7D stride order.
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter {
                cell_ids: vec![1, 2, 3],
            }),
            Filter::Material(yamc_tallies::MaterialFilter {
                material_ids: vec![10, 20],
            }),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let t = Arc::new(t);
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let geometry = geometry_with_materials(
            &[(1, Some(0)), (2, Some(1)), (3, Some(0))],
            vec![bare_material(10), bare_material(20)],
        );
        let pack = build_tallies_pack(&validated, &geometry, 3, &[], &[]).unwrap();
        assert_eq!(pack.n_cells_per_tally, vec![6]);
        // cell_bin * 2 + material_bin: (0,0) -> 0, (1,1) -> 3, (2,0) -> 4.
        assert_eq!(pack.cell_to_bin, vec![0, 3, 4]);
    }

    #[test]
    fn material_filter_writeback_lands_on_the_cpu_bin_layout() {
        // The kernel's flat block is `spatial -> parent -> energy -> mesh`;
        // the writeback must land each value on the same bin the CPU's
        // `get_bin_index_7d(score, cell, material, nuclide, parent, energy,
        // mesh)` would. With 2 materials x 3 energy bins that means material
        // is the OUTER dimension of the tally's 6 bins -- get the stride
        // backwards and the spectra swap between materials.
        let t = material_tally(vec![10, 20], Some(vec![1e-3, 1.0, 1e3, 1e6]));
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(t.num_bins(), 6);
        // Kernel order: material-major, energy-minor.
        let kernel_out = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut acc = vec![0.0; t.num_bins()];
        accumulate_kernel_tally(&validated[0], &kernel_out, &mut acc);
        for material_bin in 0..2 {
            for energy_bin in 0..3 {
                let expect = kernel_out[material_bin * 3 + energy_bin];
                let idx = t
                    .get_bin_index_7d(0, 0, material_bin, 0, 0, energy_bin, 0)
                    .unwrap();
                assert_eq!(
                    acc[idx], expect,
                    "material {material_bin} energy {energy_bin}"
                );
            }
        }
    }

    // ---- EnergyFunctionFilter on the GPU (issue #271) -------------------
    //
    // The filter is a multiplicative weight plus an out-of-range gate, not a
    // bin dimension, so it changes the packed table and nothing about the
    // output layout. These tests pin both halves of that.

    fn efunc_tally(energy: Vec<f64>, y: Vec<f64>) -> Arc<Tally> {
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::EnergyFunction(yamc_tallies::EnergyFunctionFilter::new(energy, y)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        Arc::new(t)
    }

    fn one_cell_geometry() -> crate::geometry::Geometry {
        let sphere = Arc::new(crate::geo::Surface {
            surface_id: Some(1),
            kind: crate::geo::SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: crate::geo::BoundaryType::Vacuum,
            name: None,
        });
        let region =
            crate::geo::Region::new_from_halfspace(crate::geo::HalfspaceType::Below(sphere));
        let cell = crate::geometry::cell::Cell::new(Some(1), region, None, None);
        crate::geometry::Geometry::new(vec![cell], Vec::new()).unwrap()
    }

    #[test]
    fn validate_accepts_energy_function_filter() {
        let t = efunc_tally(vec![1.0, 10.0, 100.0, 1000.0], vec![1.0, 2.0, 3.0, 4.0]);
        let v = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
    }

    #[test]
    fn energy_function_filter_alone_is_still_rejected() {
        // It weights, it does not bin. Without a CellFilter / MaterialFilter /
        // MeshFilter the tally has no spatial binner and must reject exactly as
        // a bare EnergyFilter does -- accepting it would silently score the
        // whole geometry into one bin.
        let mut t = Tally::new();
        t.filters = vec![Filter::EnergyFunction(
            yamc_tallies::EnergyFunctionFilter::new(
                vec![1.0, 10.0, 100.0, 1000.0],
                vec![1.0, 2.0, 3.0, 4.0],
            ),
        )];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let t = Arc::new(t);
        let err = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::UnsupportedTallyFilters { .. }
        ));
    }

    #[test]
    fn energy_function_table_packs_energies_then_spline_coeffs() {
        let energy = vec![1.0, 10.0, 100.0, 1000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let t = efunc_tally(energy.clone(), y.clone());
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let pack = build_tallies_pack(&validated, &one_cell_geometry(), 1, &[], &[]).unwrap();

        // `[n_points, energy[n], coeffs[4*(n-1)]]`.
        assert_eq!(pack.efunc_offsets, vec![0, efunc_table_len(4) as u32]);
        assert_eq!(pack.efunc_params[0], 4.0, "leading point-count word");
        assert_eq!(&pack.efunc_params[1..5], &energy[..]);
        let filter = t.get_energy_function_filter().unwrap();
        let flat: Vec<f64> = filter.spline_coeffs().iter().flatten().copied().collect();
        assert_eq!(&pack.efunc_params[5..], &flat[..]);

        // No bin dimension: one cell bin, one energy bin, nothing else.
        assert_eq!(pack.out_offsets, vec![0, 1]);
        assert_eq!(t.num_bins(), 1);
    }

    #[test]
    fn tally_without_energy_function_packs_an_empty_range() {
        // The empty range IS the "no filter" flag the kernel tests, so a plain
        // tally must leave the offsets equal rather than pushing a placeholder.
        let t = tally_with_score(vec![1], "flux".parse().unwrap());
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let pack = build_tallies_pack(&validated, &one_cell_geometry(), 1, &[], &[]).unwrap();
        assert_eq!(pack.efunc_offsets, vec![0, 0]);
        assert!(pack.efunc_params.is_empty());
    }

    #[test]
    fn packed_energy_function_evaluates_exactly_like_the_cpu_filter() {
        // The gate test the twin harness generalises: the packed descriptor
        // must reproduce `EnergyFunctionFilter::get_weight` bit for bit,
        // including the None-outside-range contract.
        let energy = vec![1.0, 10.0, 100.0, 1000.0];
        let y = vec![0.5, 2.0, 7.0, 3.0];
        let t = efunc_tally(energy.clone(), y.clone());
        let validated = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap();
        let pack = build_tallies_pack(&validated, &one_cell_geometry(), 1, &[], &[]).unwrap();
        let filter = t.get_energy_function_filter().unwrap();

        for &e in &[0.5, 1.0, 5.5, 10.0, 55.0, 100.0, 550.0, 1000.0, 1000.1] {
            let want = filter.get_weight(e);
            let got = yamc_gpu::common::tallies::energy_function_weight(&pack.efunc_params, 0, e);
            assert_eq!(got, want, "energy {e}");
        }
    }

    #[test]
    fn validate_accepts_neutron_particle_filter() {
        // The `particle="neutron"` kwarg in the Python Tally constructor
        // adds a `ParticleType(Neutron)` filter. The kernel is neutron-
        // only so it's fine to accept and ignore.
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Neutron)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let v = validate_tallies(std::slice::from_ref(&arc), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
    }

    #[test]
    fn validate_accepts_cell_energy_neutron_particle_filter() {
        // The verification suite's spectrum tally has all three:
        // [CellFilter, EnergyFilter, ParticleTypeFilter(Neutron)].
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::Energy(EnergyFilter::new(vec![1e-3, 1.0, 1e3])),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Neutron)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let v = validate_tallies(std::slice::from_ref(&arc), ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn validate_rejects_photon_particle_filter() {
        // Neutron-context: a `particle="photon"` filter must reject so
        // the user gets a clear error rather than a silent zero tally.
        // (The photon-context counterpart `validate_accepts_photon_*`
        // exercises the opposite.)
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let err = validate_tallies(std::slice::from_ref(&arc), ParticleType::Neutron).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::UnsupportedTallyFilters { .. }
        ));
    }

    #[test]
    fn validate_photon_accepts_photon_particle_filter() {
        // Photon-context: `particle="photon"` filter is accepted on the
        // photon dispatch path (mirror of the neutron acceptance test).
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let v = validate_tallies(std::slice::from_ref(&arc), ParticleType::Photon).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
    }

    #[test]
    fn validate_photon_rejects_neutron_particle_filter() {
        // Photon-context: a `particle="neutron"` filter must reject so
        // mistargeted tallies don't silently zero.
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Neutron)),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let err = validate_tallies(std::slice::from_ref(&arc), ParticleType::Photon).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::UnsupportedTallyFilters { .. }
        ));
    }

    #[test]
    fn validate_photon_accepts_photon_xs_scores() {
        // Photon-context: the 4 photon-component XS scores route
        // through SCORE_PER_MT with MTs 502/504/516/522. These are
        // what the broomstick verification notebook uses.
        let mk = |name: &str| {
            let mut t = Tally::new();
            t.filters = vec![
                Filter::Cell(CellFilter { cell_ids: vec![1] }),
                Filter::ParticleType(ParticleTypeFilter::new(ParticleType::Photon)),
            ];
            t.scores = vec![name.parse().unwrap()];
            Arc::new(t)
        };
        let arr = vec![
            mk("coherent-scatter"),
            mk("incoherent-scatter"),
            mk("photoelectric"),
            mk("pair-production"),
        ];
        let v = validate_tallies(&arr, ParticleType::Photon).unwrap();
        assert_eq!(v.len(), 4);
        for entry in &v {
            assert_eq!(entry.score_kind, SCORE_PER_MT);
        }
        assert_eq!(v[0].score_mt, Some(502));
        assert_eq!(v[1].score_mt, Some(504));
        assert_eq!(v[2].score_mt, Some(522));
        assert_eq!(v[3].score_mt, Some(516));
    }

    #[test]
    fn validate_neutron_rejects_photon_xs_scores() {
        // Neutron-context: photon-component scores must reject; the
        // neutron kernel has no photon-XS lookup. Same reject path as
        // the existing `validate_rejects_unsupported_score`.
        let t = tally_with_score(vec![1], "coherent-scatter".parse().unwrap());
        let err = validate_tallies(std::slice::from_ref(&t), ParticleType::Neutron).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::UnsupportedTallyScore { tally_index: 0, .. }
        ));
    }

    #[test]
    fn validate_accepts_multiple_tallies() {
        let a = flux_tally_cell_energy(vec![1], vec![1e-3, 1.0, 1e3]);
        let b = tally_with_score(vec![1], "total".parse().unwrap());
        let c = tally_with_score(vec![1], "absorption".parse().unwrap());
        let arr = vec![a, b, c];
        let v = validate_tallies(&arr, ParticleType::Neutron).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].score_kind, SCORE_FLUX);
        assert_eq!(v[1].score_kind, SCORE_TOTAL);
        // absorption now routes through SCORE_PER_MT(27) -- see #415.
        assert_eq!(v[2].score_kind, SCORE_PER_MT);
    }

    #[test]
    fn validate_rejects_too_few_energy_bins() {
        // EnergyFilter::new panics on < 2 bins so we can't reach this
        // branch through the public constructor. Hand-build the filter
        // via struct literal so the post-construction validity check
        // still has a path to exercise.
        let mut t = Tally::new();
        t.filters = vec![
            Filter::Cell(CellFilter { cell_ids: vec![1] }),
            Filter::Energy(EnergyFilter { bins: vec![1.0] }),
        ];
        t.scores = vec![Score::Flux(yamc_tallies::score::FluxScore)];
        let arc = Arc::new(t);
        let err = validate_tallies(std::slice::from_ref(&arc), ParticleType::Neutron).unwrap_err();
        assert!(matches!(
            err,
            GpuDispatchError::EnergyBinsTooFew { tally_index: 0 }
        ));
    }
}
