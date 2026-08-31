//! Multi-cell multi-step transport on the GPU.
//!
//! The composition of every previous kernel into one launch. Per
//! particle, per step:
//!
//! 1. **Find the current cell** via linear AABB scan (`cell_finding`).
//!    If outside all cells, mark the particle escaped and break.
//! 2. **XS lookup at the particle's energy** via log-energy binary
//!    search + linear interpolation (`xs_lookup`). All cells share
//!    the same single-material XS arrays in this first cut.
//! 3. **Sample free flight** `d_xs = -ln(ξ₁) / Σ_t` (Phase A).
//! 4. **Distance to nearest surface** `d_boundary` via the same
//!    sphere/plane scan as `boundary_distance`.
//! 5. **`d = min(d_xs, d_boundary)`** -- the actual transport step.
//! 6. **Score per-cell tally**: `track_length × Σ_a` into the
//!    accumulator slot for the current cell, fixed-point u64 atomic
//!    add (Phase D pattern).
//! 7. **Move the particle** by `d` along its direction.
//! 8. **If `d_xs` won**: collision. Sample MT (`collision_sampling`).
//!    Absorb (alive = 0) or elastic-scatter (energy + lab cosine +
//!    Marsaglia rejection + 3D direction rotation, same as
//!    `multi_step.rs`).
//! 9. **Else**: surface crossing. Position is on the surface; the
//!    next iteration's cell-finding will pick the new cell.
//!
//! The kernel terminates when the particle is absorbed, escapes the
//! geometry, or reaches `MAX_STEPS`.
//!
//! # Scope of this first cut
//!
//! - **Per-cell materials supported.** Each cell maps to a material
//!   index via `cell_to_material`; the material's XS arrays and
//!   `target_mass` are looked up from per-material flat buffers.
//!   The energy grid is shared across materials (typical in yamc:
//!   one master grid covers every nuclide's range).
//! - **Linear cell scan + linear surface scan.** O(N_cells +
//!   N_surfaces) per particle per step. Fine for tests with ≤16
//!   cells; production with thousands of cells needs BVH traversal
//!   and/or per-cell surface lists.
//! - **No surface sense / no exclude-the-just-crossed-surface trick.**
//!   The next step's cell-finding handles ambiguity at AABB
//!   boundaries (inclusive `>=`/`<=` gives deterministic lower-index
//!   selection). Standard MC techniques like `d + ε` nudges aren't
//!   used -- the test geometry doesn't need them.
//! - **No multi-bin energy tally.** One slot per cell, scoring
//!   `track_length × Σ_a`. Combining cell × energy is a flat-buffer
//!   indexing change.

use crate::common::tallies::TalliesPack;
use crate::neutron::xs::{MT_INELASTIC_COUNT, URR_META_COLS, URR_XS_COLS};
use crate::{GpuContext, WgpuRuntime};

mod cpu;
mod cpu_rayon;
mod decay_photon_emission;
mod dispatch;
mod host;
mod kernel;
mod photon_emission;
mod roulette;
mod shared;
mod urr_perturb;

pub(super) use kernel::multi_cell_transport_kernel;
pub use kernel::PERHIST_K;
// Shared #[cube] scoring helpers, reused by the photon kernel: mesh binning
// (issues #234, #279) and the energy-function evaluator (issue #271). The
// latter is species-agnostic -- it reads a tabulated curve at the particle's
// energy -- so both kernels call the same one rather than each carrying a copy
// of the spline evaluation.
pub(crate) use kernel::{
    cyl_mesh_bin_at_kernel, cyl_mesh_score_src_acc, energy_function_weight_kernel,
    mesh_rect_score_src_acc, rect_mesh_bin_at_kernel,
};

/// Exact per-history global-spill capacity for the batch-free variance path
/// (issue #233).
///
/// A history touches at most ONE flat tally bin per tally per transport step: it
/// is at a single cell and single energy each step, and the track-length and
/// same-step collision estimators score that same flat bin. So across at most
/// `max_steps` steps it touches at most `min(max_steps * n_tallies,
/// total_out_len)` distinct flat bins. The first `PERHIST_K` live in the
/// register touched-list; the global spill holds the remainder, so it needs at
/// most this many entries per history. This is a PROVEN upper bound (never
/// exceeded), so the spill never overflows and no bin is silently dropped from
/// the variance.
///
/// INVARIANT: assumes each estimator scores at most one distinct flat bin per
/// tally per step. This holds for the CSG track-length and collision estimators
/// (single cell + single energy per step); a track-length MESH tally crossing
/// multiple voxels in one step would violate it and need a larger bound.
///
/// `n_tallies` MUST be the PACK tally count ([`crate::common::tallies::TalliesPack::n_tallies`],
/// one entry per scored tally), not any coarser user-tally count. When
/// `max_steps` is effectively unbounded the product saturates and the bound
/// degrades gracefully to `total_out_len - PERHIST_K` (the previous cap).
pub fn per_history_spill_cap(total_out_len: usize, max_steps: u32, n_tallies: usize) -> usize {
    (max_steps as usize)
        .saturating_mul(n_tallies)
        .min(total_out_len)
        .saturating_sub(PERHIST_K as usize)
}

pub use crate::neutron::fission_bank_inputs::FissionBankInputs;
pub use crate::neutron::nuclide_select_inputs::{
    NuclideSelectInputs, NUC_PARTIAL_ABSORPTION, NUC_PARTIAL_COLS, NUC_PARTIAL_ELASTIC,
    NUC_PARTIAL_FISSION, NUC_PARTIAL_INELASTIC,
};
pub use crate::neutron::survival_biasing::SurvivalBiasingInputs;
// The `permt_meta` layout constants moved to the always-built
// `neutron::xs::constants` so the yamc translate layer can use them on the
// macOS stub build; re-exported here for the kernel-side users.
pub use crate::neutron::xs::{
    COL_PERMT_I_START, COL_PERMT_N_STORED, COL_PERMT_VALUE_OFFSET, PERMT_META_COLS,
};
pub use cpu::run_multi_cell_transport_cpu;
pub use cpu_rayon::run_multi_cell_transport_cpu_rayon;
pub use decay_photon_emission::{DecayPhotonInputs, MaterialDecayTable, DECAY_META_COLS};
pub use host::run_multi_cell_transport;
pub use photon_emission::{CoupledPhotonInputs, PhotonBankResult};
pub use shared::{CollisionRecord, PendDrain, PEND_SLOTS, PEND_SLOTS_U32};

/// Surface boundary discriminants. Match the yamc-geo
/// `BoundaryType` enum order so callers can `as u32` straight from
/// the public type. Transmission lets the particle continue (the
/// existing kernel behaviour); vacuum kills the particle on the
/// step where the surface is crossed.
pub const BOUNDARY_TRANSMISSION: u32 = 0;
pub const BOUNDARY_VACUUM: u32 = 1;

/// Maximum allowed particle weight before the kernel terminates the
/// tracer. Cap exists because the GPU transport doesn't have a
/// particle bank -- multi-neutron-out reactions (MT 16/17) and fission
/// scale weight by the slot's yield (`weight × ν̄`) instead of cloning
/// the particle. For non-fissionable nuclides the compounding is
/// bounded (n,2n thresholds slow neutrons below the (n,2n) threshold
/// after a few generations); for fissile actinides the chain has no
/// threshold, so weight can grow as ν̄^N for N fission events.
/// Without a cap, `weight × track_length × XS × 2^30` (the fixed-
/// point scale) overflows the i64 atomic-add buffer for histories
/// with ~15+ fission events, producing negative tally values.
///
/// `1000.0` allows up to ~6 fission generations at ν̄≈3 (3^6 = 729)
/// before terminating -- well clear of the i64 overflow threshold
/// while keeping the bias to a few percent on actinides at fast
/// energies. Histories above the cap are killed (alive=0); this is a
/// **biased-low** terminator (the truncated tail's expected
/// contribution is dropped). For fixed-source simulations with small
/// fissile content the bias is negligible; for systems where chain-
/// length matters (criticality, breeding) a particle bank would be
/// the principled fix.
pub const FISSION_WEIGHT_CAP: f64 = 1000.0;

/// Epsilon nudge applied along the direction after a transmission surface
/// crossing, so the point-in-region cell test (strict `<`/`>` on the
/// surface value) lands strictly inside the cell being entered rather than
/// exactly on the shared surface (where the point is in neither cell).
/// Matches the `eps = 1e-8` the CPU `Cell::distance_to_surface` /
/// `Region::is_exit_surface` use for the cell handoff.
pub const REGION_CROSS_EPS: f64 = 1e-8;

/// Storage-buffer descriptor binding count for the
/// `multi_cell_transport_kernel` `#[cube]` function. Each `&[T]`
/// parameter consumes one binding in a single descriptor set on
/// the cubecl-spirv backend, so the per-stage descriptor budget
/// the driver advertises (`max_storage_buffers_per_shader_stage`)
/// must be at least this large.
///
/// Driver behaviour past the limit is silent: bindings get aliased
/// or zeroed and the kernel reads garbage where it expects valid
/// data -- easy to mistake for a physics bug. We hit this exact
/// failure mode going from 78 to 91 bindings while adding
/// inelastic distributions; a 1 eV free-gas Pb208 sphere dropped
/// from GPU/CPU = 0.96 to 0.37 because the elastic-scatter path
/// was reading garbage from a shifted descriptor binding.
///
/// Updating this value: when the kernel signature gains or loses
/// `&[T]` arguments, edit this constant in lockstep -- the
/// `cargo test` suite has a sanity check that the constant equals
/// the actual count.
// NOTE: this counts the kernel's READ-ONLY `&[T]` slice parameters only, which
// is what the lockstep test in `tests.rs` checks; `&mut [T]` outputs (including
// the issue-#289 `lost_count` / `lost_f64` pair) are not included.
pub const KERNEL_STORAGE_BUFFER_COUNT: u32 = 181;

/// Compile-time cap on the number of secondary photons banked per neutron
/// collision (coupled neutron->photon production, S4b). cubecl needs a bounded
/// loop, so the per-collision emission loop runs at most this many iterations.
///
/// The physical yield `y_t = photon_prod / sigma_t` is typically well below 1
/// for the structural materials we care about (Fe56 at fast energies emits a
/// fraction of a photon per collision on average), so 16 is far above any
/// realistic per-collision photon count -- it exists only to bound the loop,
/// never to clip a physically-plausible draw. The photon COUNT sampler
/// (`sample_photon_count`) returns `floor(y_t) + Bernoulli(frac)`, which the
/// loop honours up to this cap.
pub const MAX_PHOTONS_PER_COLLISION: u32 = 16;

/// Column count for `mt_slot_u32_meta` -- the packed buffer that
/// holds one u32 per `(material, MT slot)` for every per-MT-slot
/// flag/length the kernel reads (eout discriminator, eout/corr/km/
/// evap n_energies, scatter-in-CM flag, n-body count). Packing
/// these into a single buffer instead of 7 separate `&[u32]`
/// parameters frees up 6 storage-buffer descriptor bindings, which
/// is what the kernel needs to keep adding distribution samplers
/// (Watt-inelastic, Tabulated equiprobable, URR) without tripping
/// the per-stage descriptor budget. Layout is row-major
/// `[n_mat × MT_INELASTIC_COUNT × MT_SLOT_U32_COLS]`, indexed as
/// `meta[(mat * MT_INELASTIC_COUNT + slot) * MT_SLOT_U32_COLS + col]`.
pub const MT_SLOT_U32_COLS: u32 = 11;
pub const COL_EOUT_KIND: u32 = 0;
pub const COL_EOUT_N_ENERGIES: u32 = 1;
pub const COL_CORR_N_ENERGIES: u32 = 2;
pub const COL_SCATTER_IN_CM: u32 = 3;
pub const COL_KM_N_ENERGIES: u32 = 4;
pub const COL_EVAP_N_ENERGIES: u32 = 5;
pub const COL_NBPS_N_BODIES: u32 = 6;
pub const COL_MAXWELL_N_ENERGIES: u32 = 7;
pub const COL_WATT_N_ENERGIES: u32 = 8;
/// Number of Evaporation sub-distribution components for this MT slot.
/// `>= 2` makes the kernel draw one uniform per collision to pick a
/// component (mirroring the CPU's per-collision applicability sampling);
/// `0`/`1` mean a single curve and no selector draw.
pub const COL_EVAP_N_COMPONENTS: u32 = 9;
/// Number of correlated angle-energy sub-distribution components for this
/// MT slot. A neutron product may carry several equally-weighted
/// `CorrelatedAngleEnergy` laws gated by applicability (e.g. F19 MT16 n,2n
/// has two with 0.5/0.5 weights). `>= 2` makes the kernel draw one uniform
/// per collision to pick a component (mirroring the CPU
/// `ReactionProduct::sample_distribution_index` for equal applicability);
/// `0`/`1` mean a single distribution and no selector draw. The `corr_*`
/// rows are component-major: component `c` occupies the `n_per_comp =
/// COL_CORR_N_ENERGIES / COL_CORR_N_COMPONENTS` rows starting at
/// `corr_ae_offset[slot] + c * n_per_comp`.
pub const COL_CORR_N_COMPONENTS: u32 = 10;

/// Column count for `mt_slot_f64_meta` -- packed `[n_mat ×
/// MT_INELASTIC_COUNT × MT_SLOT_F64_COLS]` buffer holding the
/// per-MT-slot f64 scalars the kernel reads (the Maxwell/Watt
/// restriction energies `maxwell_u` / `watt_u`, plus
/// `nbps_total_mass`). Evaporation's `evap_u` is incident-energy
/// dependent and lives in its own per-grid buffer, so it is not
/// packed here. Same packing motivation as
/// `mt_slot_u32_meta` -- collapses 3 separate `[n_mat ×
/// MT_INELASTIC_COUNT]` f64 bindings into one, freeing 2
/// storage-buffer descriptor bindings. Without this the kernel
/// hits the per-stage descriptor budget on some drivers (silent
/// aliasing past the limit, ratio 0.37 in the free-gas regression
/// test).
pub const MT_SLOT_F64_COLS: u32 = 4;
pub const COL_EVAP_U: u32 = 0;
pub const COL_MAXWELL_U: u32 = 1;
pub const COL_WATT_U: u32 = 2;
pub const COL_NBPS_TOTAL_MASS: u32 = 3;

/// Column count for `mat_f64_meta` -- packed `[n_mat × MAT_F64_COLS]`
/// buffer of per-material f64s the kernel reads in the elastic /
/// fission / free-gas branches. Same packing motivation as
/// `mt_slot_u32_meta`: collapses 4 separate `&[f64]` parameters
/// (target mass, temperature, Watt-spectrum a + b) into one upload,
/// freeing 3 storage-buffer descriptor bindings.
pub const MAT_F64_COLS: u32 = 5;
pub const COL_TARGET_MASS: u32 = 0;
pub const COL_TEMPERATURE_K: u32 = 1;
pub const COL_FISSION_A: u32 = 2;
pub const COL_FISSION_B: u32 = 3;
/// Atom density (atoms/barn-cm) of the dominant URR-bearing
/// nuclide in this material. Used to scale URR-perturbed micro
/// XS up to a macroscopic contribution; the existing aggregated
/// `xs_*_per_material` carries the smooth macro for the rest of
/// the material. Zero when no nuclide in the material has URR
/// data -- the kernel falls back to smooth XS in that case.
pub const COL_URR_ATOM_DENSITY: u32 = 4;

/// Column count for `coarse_meta` -- the packed per-material buffer that
/// describes each material's own COARSE energy grid (issue #212). Every
/// material carries its own coarse grid (its finest single per-nuclide grid),
/// concatenated tight into `coarse_log_energy_grid`; the per-MT inelastic
/// buffers (`xs_inelastic_per_mt` / `yield_per_mt`) are tight-CSR keyed per
/// slab off that same per-material grid. Sharing one coarse grid across
/// materials smeared non-first materials' inelastic thresholds (a ~2% flux
/// deficit below 1 MeV on multi-material problems); a full union grid closes it
/// but costs multi-GB. The per-material coarse grid closes it while keeping the
/// buffers bounded to one material's grid. Layout is row-major
/// `[n_mat × COARSE_META_COLS]`, indexed as
/// `coarse_meta[mat * COARSE_META_COLS + COL_*]`. For a single-material problem
/// the one row is `[0, coarse_len, 0]`, so the kernel indexes the coarse grid at
/// base 0 with the full length and the per-MT stride is the material's coarse
/// length, byte-identical to the old single shared coarse grid.
pub const COARSE_META_COLS: u32 = 3;
/// Base of this material's coarse grid inside the concatenated
/// `coarse_log_energy_grid` (the coarse bracket binary-search offset).
pub const COL_COARSE_GRID_OFFSET: u32 = 0;
/// This material's coarse grid length (the per-MT inelastic buffer stride and
/// the coarse-bracket clamp bound).
pub const COL_COARSE_N: u32 = 1;
/// Base into `permt_meta` ROWS for this material's FIRST slab (issue #212
/// sparse per-MT storage). Equals `nuc_off * MT_INELASTIC_COUNT` (the material's
/// first slab's global (slab, slot) row base). The kernel derives (slab `s`,
/// slot) 's permt row as `COL_COARSE_MT_BASE + (s - nuc_off) * MT_INELASTIC_COUNT
/// + slot`, which reduces to the global `s * MT_INELASTIC_COUNT + slot` (the same
/// ordering as `q_inelastic_per_mt`). The old dense per-MT XS / yield element
/// base this column used to hold is gone: the per-MT XS / yield are now sparse
/// (see `PERMT_META_COLS`), keyed off `permt_meta`, not off a per-material
///   element base.
pub const COL_COARSE_MT_BASE: u32 = 2;

/// Column count for `fine_meta` -- the packed per-material buffer that
/// describes each material's own FINE energy grid (issue #212). Where the
/// COARSE grid backs the memory-bounded per-MT inelastic buffers, the FINE grid
/// is resonance-critical: it backs the per-material aggregate macro XS
/// (`xs_elastic_per_material` etc.) and the per-(material, nuclide)
/// `nuc_macro_total` / `nuc_partial_xs` that drive Sigma_t, the per-collision
/// nuclide selection, and the #211 URR smooth baseline. Each material now
/// carries its OWN fine grid (its union grid), concatenated tight into
/// `fine_log_energy_grid`; the aggregate XS are tight-CSR keyed per material and
/// the nuc buffers per slab off that same per-material grid. Linear
/// interpolation along a material's own grid is invariant to which other
/// materials' points are present, so this reduces RAM (per-material-bounded fine
/// buffers, vs the global-union `n_slab × n_grid_global × 4` nuc_partial_xs)
/// while holding accuracy. The GLOBAL `log_energy_grid` is retained for the
/// grid-shared score / photon / decay lookups. Layout is row-major
/// `[n_mat × FINE_META_COLS]`, indexed as `fine_meta[mat * FINE_META_COLS +
/// COL_*]`. For a single-material problem the one row is `[0, fine_len, 0]`
/// (its union == the global union), so all indexing reduces to the old
/// `mat_idx × n_grid` / `slab × n_grid` and stays byte-identical.
pub const FINE_META_COLS: u32 = 3;
/// Base of this material's fine grid inside the concatenated
/// `fine_log_energy_grid`; ALSO the base into the per-material aggregate macro
/// XS buffers (`xs_elastic_per_material` etc.), since those are `[n_mat × fine_n]`
/// concatenated in the same per-material order and length as the grid.
pub const COL_FINE_GRID_OFFSET: u32 = 0;
/// This material's fine grid length (the aggregate-XS row length, the per-slab
/// `nuc_macro_total` / `nuc_partial_xs` stride, and the fine-bracket clamp bound).
pub const COL_FINE_N: u32 = 1;
/// Base into `nuc_macro_total` (element units) for this material's FIRST slab.
/// The kernel derives slab `s`'s row base as
/// `COL_FINE_NUC_BASE + (s - nuc_off) * COL_FINE_N`; `nuc_partial_xs` reuses the
/// same element index times `NUC_PARTIAL_COLS`. Every slab of a material shares
/// that material's fine length.
pub const COL_FINE_NUC_BASE: u32 = 2;

/// Per-particle + per-tally outputs from a multi-cell transport launch.
pub struct MultiCellResult {
    pub alive: Vec<u32>,
    pub n_steps: Vec<u32>,
    pub final_energies: Vec<f64>,
    /// Per-tally outputs in the same order as the input
    /// `TalliesPack`. Tally `t` has shape
    /// `[n_cells_per_tally[t] × n_bins_per_tally[t]]`, cell-major
    /// (`bin_cell * n_bins + bin_e`). When the run used per-history
    /// variance ([`tally_sum_sq`](Self::tally_sum_sq) non-empty) this
    /// is the per-bin `sum` (Σ_h x_h); otherwise it is the raw per-step
    /// accumulation.
    pub tally_outputs: Vec<Vec<f64>>,
    /// Per-tally per-bin sum-of-squares (Σ_h x_h², physical units), in
    /// the same shape/order as [`tally_outputs`](Self::tally_outputs).
    /// Non-empty only for a batch-free per-history-variance run (issue
    /// #233); empty for the per-step path and the CPU mirrors. Combined
    /// with `tally_outputs` (the sum) and the history count it gives the
    /// exact per-history variance `m2 = sum_sq − sum²/N`.
    pub tally_sum_sq: Vec<Vec<f64>>,
    /// Raw per-source accumulator (issue #233 Stage 2, `PerSource` mode only):
    /// flat `chunk_sources * total_out_len` fixed-point words, row-major by
    /// source index, holding this launch's per-`(source, flat_bin)` sum (per-tally
    /// scale). The dispatch accumulates it across the source + generation launches
    /// and, per source, adds the total into `sum` and its square into `sum_sq`.
    /// Empty for `PerStep` / `PerHistory` and the CPU mirrors.
    pub src_acc: Vec<u64>,
    /// Originating source index of each banked fission progeny (issue #233
    /// Stage 2), indexed by bank slot. The dispatch reads it alongside the bank
    /// to build the next generation launch's `source_idx`. Empty unless
    /// `PerSource`.
    pub bank_source_idx: Vec<u32>,
    /// Number of energy bins per tally -- copied from the input pack
    /// so callers can index `tally_outputs[t]` without re-threading
    /// the pack.
    pub n_bins_per_tally: Vec<u32>,
    pub n_cells: usize,
    /// Coupled neutron->photon production bank (slice S4b). Empty / zero
    /// counters when the run was coupled-OFF; otherwise the secondary photons
    /// emitted at collisions, ready for S5 to drain and transport.
    pub photon_bank: PhotonBankResult,
    /// Histories that ended in no cell, i.e. lost particles (issue #289).
    /// `count` is every loss this launch saw; `records` carries diagnostics for
    /// the first [`crate::common::lost_particles::LOST_RECORD_CAPACITY`] of
    /// them. The dispatch enforces `max_lost_particles` from `count`, matching
    /// the CPU's abort.
    pub lost: crate::common::lost_particles::LostParticleResult,
    /// (n,xn) secondaries produced while the thread already held
    /// [`shared::PEND_SLOTS`] of them (issue #111 phase 2). On the kernel these
    /// are appended to the device particle bank as `PTYPE_NEUTRON` records for
    /// the host to drain in a later pass; on the CPU mirrors they are
    /// transported in-thread and this is a pure diagnostic. Either way nothing
    /// is dropped, and a zero here means the in-thread stack held the whole
    /// history.
    pub n_spilled_secondaries: u64,
    /// Deepest the in-thread pending-secondary stack got across this launch's
    /// histories, i.e. how many of the [`shared::PEND_SLOTS`] register slots
    /// were actually used. CPU mirrors only (the kernel does not report it);
    /// zero from the GPU host path.
    pub max_pend_depth: u32,
}

impl MultiCellResult {
    /// Tally `t`'s output flattened over energy bins -- one value
    /// per cell-bin. Useful when the caller wants total counts and
    /// doesn't care about the energy resolution.
    pub fn tally_per_cell(&self, t: usize) -> Vec<f64> {
        let flat = &self.tally_outputs[t];
        let n_bins = self.n_bins_per_tally[t] as usize;
        let n_cell_bins = flat.len() / n_bins.max(1);
        sum_bins(flat, n_cell_bins, n_bins)
    }

    /// Convenience accessor for the kernel test fixtures, which
    /// build the standard 2-tally `flux_abs_pack`: tally 0 is
    /// per-cell flux, tally 1 is per-cell absorption.
    pub fn flux_per_cell_energy(&self) -> &[f64] {
        &self.tally_outputs[0]
    }

    /// Same fixture convention as `flux_per_cell_energy`.
    pub fn absorption_per_cell_energy(&self) -> &[f64] {
        &self.tally_outputs[1]
    }

    /// Per-cell flux summed over energy bins (test-fixture pack).
    pub fn flux_per_cell(&self) -> Vec<f64> {
        self.tally_per_cell(0)
    }

    /// Per-cell absorption summed over energy bins (test-fixture pack).
    pub fn absorption_per_cell(&self) -> Vec<f64> {
        self.tally_per_cell(1)
    }
}

fn sum_bins(flat: &[f64], n_cells: usize, n_bins: usize) -> Vec<f64> {
    (0..n_cells)
        .map(|c| {
            let off = c * n_bins;
            flat[off..off + n_bins].iter().sum()
        })
        .collect()
}

/// Slice the flat fixed-point `tally_out` buffer back into per-tally
/// f64 outputs. Each tally `t` gets a `Vec<f64>` of length
/// `n_cells_per_tally[t] * n_bins_per_tally[t]`. The fixed-point
/// scale is per-tally (`fixed_point_scales[t]`) so KERMA-shape
/// tallies that use a smaller scale to avoid u64 overflow are
/// unpacked at the same scale they were packed with.
pub(super) fn unpack_tally_outputs(bits: &[u64], tallies: &TalliesPack) -> Vec<Vec<f64>> {
    let n_tallies = tallies.n_tallies() as usize;
    let mut out = Vec::with_capacity(n_tallies);
    for t in 0..n_tallies {
        let start = tallies.out_offsets[t] as usize;
        let end = tallies.out_offsets[t + 1] as usize;
        let slice = &bits[start..end];
        let inv_scale = 1.0 / tallies.fixed_point_scales[t];
        let f: Vec<f64> = slice
            .iter()
            .map(|&b| (b as i64 as f64) * inv_scale)
            .collect();
        out.push(f);
    }
    out
}

/// Slice the second half of a per-history-variance `tally_out` buffer
/// (the sum-of-squares accumulator) back into per-tally f64 outputs
/// (issue #233). `bits` is the second half only (length `total_out_len`,
/// same layout as the sum half). Each tally `t` is unpacked at its
/// derived per-history sum-of-squares scale
/// [`sum_sq_fixed_point_scale`](crate::common::tallies::sum_sq_fixed_point_scale)`(fixed_point_scales[t])`,
/// the same scale the kernel packed with at history flush.
pub(super) fn unpack_tally_sum_sq(bits: &[u64], tallies: &TalliesPack) -> Vec<Vec<f64>> {
    let n_tallies = tallies.n_tallies() as usize;
    let mut out = Vec::with_capacity(n_tallies);
    for t in 0..n_tallies {
        let start = tallies.out_offsets[t] as usize;
        let end = tallies.out_offsets[t + 1] as usize;
        let slice = &bits[start..end];
        let inv_scale =
            1.0 / crate::common::tallies::sum_sq_fixed_point_scale(tallies.fixed_point_scales[t]);
        let f: Vec<f64> = slice
            .iter()
            .map(|&b| (b as i64 as f64) * inv_scale)
            .collect();
        out.push(f);
    }
    out
}

/// Binary search the log-space bin edges for the bin containing
/// `log_e`. Mirror of the GPU kernel's bin lookup. Bin `k` covers
/// `(edges[k], edges[k + 1]]` -- a value exactly AT an interior edge
/// scores into the LOWER bin, matching the CPU
/// `EnergyFilter::get_bin` convention `bins[i] < E <= bins[i+1]`.
/// Returns `None` for energies outside the filter range
/// (`log_e < edges[0]` or `log_e > edges[n_bins]`), so out-of-range
/// flux is dropped rather than piled into the first/last bin -- this
/// matches `EnergyFilter::get_bin`, which returns `None` outside range.
/// The first bin is lower-edge inclusive (`log_e == edges[0]` -> bin 0).
#[inline]
pub(super) fn log_edges_bin(edges: &[f64], log_e: f64) -> Option<usize> {
    let n_bins = edges.len() - 1;
    if log_e < edges[0] || log_e > edges[n_bins] {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = n_bins;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if log_e <= edges[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(lo)
}

/// Per-step multi-tally accumulator (mirror of the GPU kernel's
/// inner tally loop). Walks every tally, looks up the cell-bin
/// for `cell`, skips if `NOT_IN_TALLY`, binary-searches the
/// tally's `log_edges` slice for the energy bin, picks the score
/// factor (1.0 / σ_t / σ_a) by `score_kind`, and atomic-adds the
/// fixed-point bits into `tally_acc[out_offset + cell_bin * n_bins +
/// bin_e]`. Sequential here; the `_rayon` path operates on a
/// per-thread buffer that's reduced after the chunk completes.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_tallies(
    tallies: &TalliesPack,
    cell: usize,
    n_cells: usize,
    log_e: f64,
    // Linear energy, carried alongside `log_e` for the energy-function filter
    // (issue #271), whose spline is tabulated in linear E. Recovering it as
    // `exp(log_e)` would not be bit-identical to the CPU's own value.
    energy: f64,
    d: f64,
    weight: f64,
    sigma_t: f64,
    sigma_a: f64,
    sigma_f: f64,
    mat_idx: usize,
    n_grid: usize,
    idx_lo: usize,
    idx_hi: usize,
    frac: f64,
    xs_score_per_mt: &[f64],
    n_score_mts: usize,
    urr: UrrScore,
    // Segment endpoints and direction for mesh (voxel) tallies (issue #234):
    // `r0` is the step start, `r1 = r0 + d*dir` the end, `dir` the unit
    // direction. Ignored by non-mesh tallies.
    r0: [f64; 3],
    r1: [f64; 3],
    dir: [f64; 3],
    tally_acc: &mut [u64],
) {
    use crate::common::tallies::{rect_mesh_crossings, MESH_NONE, NOT_IN_TALLY};
    let n_tallies = tallies.n_tallies() as usize;
    for t in 0..n_tallies {
        // Collision-estimator tallies are scored at collision sites by
        // `accumulate_collision_tallies`, not per step -- skip them here
        // so they never get both contributions (mirrors the GPU kernel's
        // `is_collision[t] == 0` gate on the per-step block).
        if tallies.is_collision[t] == 1 {
            continue;
        }
        let cell_bin = tallies.cell_to_bin[t * n_cells + cell];
        if cell_bin == NOT_IN_TALLY {
            continue;
        }
        let n_bins = tallies.n_bins_per_tally[t] as usize;
        let edges_off = tallies.edges_offsets[t] as usize;
        let edges = &tallies.log_edges[edges_off..edges_off + n_bins + 1];
        let bin_e = match log_edges_bin(edges, log_e) {
            Some(b) => b,
            None => continue,
        };
        let score = tally_score_factor(
            tallies,
            t,
            sigma_t,
            sigma_a,
            sigma_f,
            mat_idx,
            n_grid,
            idx_lo,
            idx_hi,
            frac,
            xs_score_per_mt,
            n_score_mts,
            urr,
        );
        // Energy-function weighting (issue #271). Off the table means the CPU
        // drops the whole event, so this is a `continue`, not a zero score.
        let score = match apply_energy_function(tallies, t, energy, score) {
            Some(s) => s,
            None => continue,
        };
        let scale = tallies.fixed_point_scales[t];
        let out_off = tallies.out_offsets[t] as usize;
        let n_mesh = tallies.n_mesh_per_tally[t] as usize;
        // Cell x energy composite; mesh (voxel) is the innermost dimension
        // (stride 1), so the flat slot is `out_off + ce * n_mesh + voxel`.
        let ce = cell_bin as usize * n_bins + bin_e;
        if tallies.mesh_kind[t] == MESH_NONE {
            let contrib = d * score * weight;
            let scaled = crate::common::tallies::round_fixed_point_bits(contrib, scale) as i64;
            let idx = out_off + ce * n_mesh; // n_mesh == 1, voxel == 0
            tally_acc[idx] = tally_acc[idx].wrapping_add(scaled as u64);
        } else {
            // Track-length mesh tally: fan the step across every voxel it
            // crosses, weighting each by its in-voxel length fraction. Only
            // rectangular row-major meshes reach here (validate_tallies gate).
            let m0 = tallies.mesh_params_offsets[t] as usize;
            let m1 = tallies.mesh_params_offsets[t + 1] as usize;
            let params = &tallies.mesh_params[m0..m1];
            let base = out_off + ce * n_mesh;
            rect_mesh_crossings(params, r0, r1, dir, |voxel, lf| {
                let contrib = d * lf * score * weight;
                let scaled = crate::common::tallies::round_fixed_point_bits(contrib, scale) as i64;
                let idx = base + voxel as usize;
                tally_acc[idx] = tally_acc[idx].wrapping_add(scaled as u64);
            });
        }
    }
}

/// Multiply tally `t`'s energy-function weight into `score`, or `None` when
/// `energy` falls outside the table and the whole scoring event is dropped
/// (issue #271). A tally with no `EnergyFunctionFilter` has an empty offset
/// range and passes `score` straight through, which is the common case.
///
/// Twin of the kernel's `ef_in_range` block. `None` is deliberately distinct
/// from `Some(0.0)`: the CPU `return`s on an out-of-range energy, so the
/// contribution never reaches the accumulator at all.
#[inline]
fn apply_energy_function(tallies: &TalliesPack, t: usize, energy: f64, score: f64) -> Option<f64> {
    let lo = tallies.efunc_offsets[t] as usize;
    let hi = tallies.efunc_offsets[t + 1] as usize;
    if hi <= lo {
        return Some(score);
    }
    crate::common::tallies::energy_function_weight(&tallies.efunc_params, lo, energy)
        .map(|w| score * w)
}

/// Per-tally score-factor (1.0 / σ_t / σ_a+σ_f / per-MT σ), shared by
/// the per-step track-length accumulator and the per-collision
/// estimator. Mirrors the kernel's `score` selection by `score_kind`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn tally_score_factor(
    tallies: &TalliesPack,
    t: usize,
    sigma_t: f64,
    sigma_a: f64,
    sigma_f: f64,
    mat_idx: usize,
    n_grid: usize,
    idx_lo: usize,
    idx_hi: usize,
    frac: f64,
    xs_score_per_mt: &[f64],
    n_score_mts: usize,
    urr: UrrScore,
) -> f64 {
    use crate::common::tallies::{SCORE_ABSORPTION, SCORE_PER_MT, SCORE_TOTAL};
    match tallies.score_kinds[t] {
        SCORE_TOTAL => sigma_t,
        // ENDF MT 27 (neutron disappearance) = σ_a + σ_f. See the
        // matching comment in the cube kernel's SCORE_ABSORPTION
        // branch for why fission is added back here.
        SCORE_ABSORPTION => sigma_a + sigma_f,
        SCORE_PER_MT => {
            // URR self-shielding correlation: inside the URR window the smooth
            // `xs_score_per_mt` value is uncorrelated with the band that
            // perturbed the transport macro, which biases the score. Score each
            // URR-perturbed reaction with the macro the step actually used.
            // Twin of the kernel's `if urr_fired` substitution.
            if urr.fired {
                match tallies.score_mt[t] {
                    102 => return urr.macro_capture,
                    27 => return sigma_a + sigma_f,
                    2 => return urr.sigma_elastic,
                    18 => return sigma_f,
                    _ => {}
                }
            }
            let slot = tallies.score_data[t] as usize;
            let off = mat_idx * n_score_mts * n_grid + slot * n_grid;
            let xs_lo = xs_score_per_mt[off + idx_lo];
            let xs_hi = xs_score_per_mt[off + idx_hi];
            xs_lo + (xs_hi - xs_lo) * frac
        }
        _ => 1.0,
    }
}

/// URR-perturbed macros a step actually transported with, for the score
/// substitution above. `fired == false` (the [`Default`]) is a no-op.
#[derive(Clone, Copy, Default)]
pub(super) struct UrrScore {
    pub fired: bool,
    pub macro_capture: f64,
    pub sigma_elastic: f64,
}

/// Per-collision multi-tally accumulator (mirror of the GPU kernel's
/// collision-site scoring loop). Called once per real collision for
/// `is_collision[t] == 1` tallies: the collision-density flux
/// estimator scores `weight × score_xs / Σ_t` into the cell/energy
/// bin at the collision point. Track-length tallies are skipped here
/// (they accumulate per step in `accumulate_tallies`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_collision_tallies(
    tallies: &TalliesPack,
    cell: usize,
    n_cells: usize,
    log_e: f64,
    // Linear incident energy, for the energy-function filter (issue #271).
    energy: f64,
    weight: f64,
    sigma_t: f64,
    sigma_a: f64,
    sigma_f: f64,
    mat_idx: usize,
    n_grid: usize,
    idx_lo: usize,
    idx_hi: usize,
    frac: f64,
    xs_score_per_mt: &[f64],
    n_score_mts: usize,
    urr: UrrScore,
    // Collision point for mesh (voxel) tallies (issue #234): the voxel
    // containing this point receives the whole collision-estimator score.
    // Ignored by non-mesh tallies.
    pos: [f64; 3],
    tally_acc: &mut [u64],
) {
    use crate::common::tallies::{rect_mesh_bin_at, MESH_NONE, NOT_IN_TALLY};
    if sigma_t <= 0.0 {
        return;
    }
    let n_tallies = tallies.n_tallies() as usize;
    for t in 0..n_tallies {
        if tallies.is_collision[t] == 0 {
            continue;
        }
        let cell_bin = tallies.cell_to_bin[t * n_cells + cell];
        if cell_bin == NOT_IN_TALLY {
            continue;
        }
        let n_bins = tallies.n_bins_per_tally[t] as usize;
        let edges_off = tallies.edges_offsets[t] as usize;
        let edges = &tallies.log_edges[edges_off..edges_off + n_bins + 1];
        let bin_e = match log_edges_bin(edges, log_e) {
            Some(b) => b,
            None => continue,
        };
        // Mesh (voxel) bin at the collision point, matching the CPU
        // `score_collision`'s `get_bin(position)`. A collision outside the
        // mesh is not scored (the CPU drops it).
        let n_mesh = tallies.n_mesh_per_tally[t] as usize;
        let voxel = if tallies.mesh_kind[t] == MESH_NONE {
            0
        } else {
            let m0 = tallies.mesh_params_offsets[t] as usize;
            let m1 = tallies.mesh_params_offsets[t + 1] as usize;
            match rect_mesh_bin_at(&tallies.mesh_params[m0..m1], pos) {
                Some(v) => v as usize,
                None => continue,
            }
        };
        let score = tally_score_factor(
            tallies,
            t,
            sigma_t,
            sigma_a,
            sigma_f,
            mat_idx,
            n_grid,
            idx_lo,
            idx_hi,
            frac,
            xs_score_per_mt,
            n_score_mts,
            urr,
        );
        // Energy-function weighting (issue #271); off the table drops the event.
        let score = match apply_energy_function(tallies, t, energy, score) {
            Some(s) => s,
            None => continue,
        };
        // Collision-density estimator: weight × score_xs / Σ_t, once
        // per collision (vs the per-step track_length × score_xs ×
        // weight in `accumulate_tallies`).
        let contrib = score * weight / sigma_t;
        let scale = tallies.fixed_point_scales[t];
        let scaled = crate::common::tallies::round_fixed_point_bits(contrib, scale) as i64;
        let out_off = tallies.out_offsets[t] as usize;
        let idx = out_off + (cell_bin as usize * n_bins + bin_e) * n_mesh + voxel;
        tally_acc[idx] = tally_acc[idx].wrapping_add(scaled as u64);
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests;
