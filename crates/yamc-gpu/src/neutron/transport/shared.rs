//! Shared CPU mirror of the multi-cell transport kernel.
//!
//! Both [`super::cpu`] (sequential) and [`super::cpu_rayon`]
//! (rayon-parallel) call into [`transport_one_particle`] for the
//! per-particle physics. The two drivers differ only in how they
//! iterate particles and allocate/reduce the tally accumulator --
//! the algorithm itself (PCG-32 RNG sequence, linear cell scan,
//! XS lookup, Marsaglia rejection, fixed-point u64 tally
//! accumulation) lives here, in one place, so the
//! `cpu_gpu_equivalence_*` tests stay bit-for-bit green.
//!
//! The module is named `shared` rather than the more natural
//! `core` because cubecl's `#[cube]` proc-macro expands paths
//! like `core::hash::Hash` literally, and a sibling module named
//! `core` shadows the std-prelude path inside the same crate.

use super::dispatch::sample_inelastic_angle;
use super::urr_perturb;
use super::{
    accumulate_collision_tallies, accumulate_tallies, BOUNDARY_VACUUM, COARSE_META_COLS,
    COL_COARSE_GRID_OFFSET, COL_COARSE_MT_BASE, COL_COARSE_N, COL_FINE_GRID_OFFSET, COL_FINE_N,
    COL_FINE_NUC_BASE, COL_PERMT_I_START, COL_PERMT_N_STORED, COL_PERMT_VALUE_OFFSET,
    FINE_META_COLS, FISSION_WEIGHT_CAP, PERMT_META_COLS, REGION_CROSS_EPS,
};
use crate::common::geometry::bvh_cell_finding::bvh_find_cell_at_point;
use crate::common::geometry::cell_finding::CELL_NOT_FOUND;
use crate::common::geometry::surface_distance::{
    cone_smallest_positive_cpu, quadric_smallest_positive_cpu, torus_smallest_positive_cpu,
};
use crate::common::pcg32::pcg_out;
use crate::common::rng::{expand_seed, secondary_seed, PCG_INCR, PCG_MULT};
use crate::common::tallies::TalliesPack;
use crate::neutron::xs::{
    EOUT_KIND_CONTINUOUS_TABULAR, EOUT_KIND_EVAPORATION, EOUT_KIND_MAXWELL, MT_INELASTIC_COUNT,
};

/// Drain order of the in-thread (n,xn) pending-secondary queue.
///
/// The cubecl kernel is [`Lifo`](PendDrain::Lifo) and has no runtime choice: it
/// runs the same stack discipline as the production CPU bank
/// (`yamc_physics::util::bank::ParticleBank`) and OpenMC's per-particle
/// secondary bank. [`Fifo`](PendDrain::Fifo) exists on the CPU twin only, as a
/// verification instrument for issue #111. Since every queued secondary carries
/// its own identity-derived collision seed, the drain order cannot change a
/// history's physics, and yamc's
/// `matched_stream_diff::secondary_drain_order_is_unobservable` asserts exactly
/// that by running the twin both ways and demanding identical tallies.
///
/// Both orders drain the SAME set: the twin's queue is unbounded (issue #111
/// phase 2), so neither order can drop a secondary and the two runs differ only
/// in the sequence they are visited in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendDrain {
    /// Oldest queued secondary first.
    Fifo,
    /// Newest queued secondary first. What the kernel, the production CPU bank
    /// and OpenMC's per-particle secondary bank do.
    Lifo,
}

/// One secondary waiting to be transported inside its parent's history: an
/// (n,xn) neutron, or a banked fission progeny when the fission bank is on.
///
/// The kernel holds up to [`PEND_SLOTS`] of the (n,xn) ones in thread-private
/// registers and spills any further ones to the device particle bank; fission
/// progeny always go straight to that bank. The twin holds both kinds in a `Vec`
/// and transports every one in-thread (it is the mean-only reference, so where a
/// contribution is accumulated does not matter, only that it is accumulated),
/// which is safe because a secondary's stream is keyed on its emission ordinal
/// rather than on where it is transported (issue #322). `nxn` records which kind
/// it is, so the register-queue diagnostics below stay a measure of the kernel's
/// (n,xn) queue alone.
#[derive(Debug, Clone, Copy)]
struct PendEntry {
    /// Identity-derived collision seed (issue #111 phase 1).
    seed: u32,
    /// `true` for an (n,xn) secondary (occupies one of the kernel's register
    /// slots), `false` for a fission progeny (goes to the device bank).
    nxn: bool,
    energy: f64,
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    weight: f64,
}

/// Thread-private (n,xn) queue depth in the cubecl kernel. A walk holding more
/// than this many secondaries at once spills the extras to the device particle
/// bank, which the host drains in a later pass (issue #111 phase 2).
///
/// Four covers an (n,4n) plus a nested (n,2n) without spilling. Because the
/// queue is a stack whose slots are RECLAIMED on pop, the depth needed is the
/// emission tree's DFS depth, not its size: every (n,xn) is endothermic and
/// splits what is left of the incident energy between its products, so the
/// chain bottoms out against the reaction threshold after a handful of levels.
pub const PEND_SLOTS: usize = 4;

/// [`PEND_SLOTS`] as the `u32` the kernel's counter is typed as.
pub const PEND_SLOTS_U32: u32 = PEND_SLOTS as u32;

/// All immutable buffers and scalars the per-particle transport
/// function reads. Built once per launch by the driver and passed
/// by reference into [`transport_one_particle`]. The fields mirror
/// the slice arguments of the public `run_multi_cell_transport_cpu*`
/// functions one-for-one -- see those docs for the meaning of each
/// buffer.
#[allow(missing_docs)]
pub(super) struct TransportInputs<'a> {
    /// Drain order of the in-thread (n,xn) queue. `Fifo` on every production
    /// path (the kernel's order); see [`PendDrain`].
    pub pend_drain: PendDrain,
    // Particle inputs
    pub seeds: &'a [u32],
    pub energies_in: &'a [f64],
    pub positions_in: &'a [f64],
    pub directions_in: &'a [f64],
    // CSG geometry
    pub cell_aabbs: &'a [f64],
    pub cell_to_material: &'a [u32],
    pub surface_types: &'a [u32],
    pub surface_params: &'a [f64],
    pub surface_boundaries: &'a [u32],
    // Flat per-cell CSG region program (point-in-region cell finding).
    pub region_program: &'a [u32],
    // Pre-built BVH (built once outside the per-particle loop)
    pub bvh_aabbs: &'a [f64],
    pub bvh_meta: &'a [u32],
    pub bvh_prims: &'a [u32],
    pub bvh_unb: &'a [u32],
    // Energy grid + macro XS
    pub log_energy_grid: &'a [f64],
    /// Concatenated per-material COARSE grids backing the per-MT inelastic
    /// buffers (issue #212): each material owns its coarse grid, described by
    /// `coarse_meta`. Bit-twin of the kernel.
    pub coarse_log_energy_grid: &'a [f64],
    /// Packed `[n_materials × COARSE_META_COLS]` per-material coarse-grid
    /// descriptor (issue #212): base into `coarse_log_energy_grid`, coarse
    /// length, and per-MT-buffer first-slab base. Bit-twin of the kernel's
    /// `coarse_meta` binding.
    pub coarse_meta: &'a [u32],
    /// Concatenated per-material FINE grids backing the resonance-critical
    /// aggregate macro XS + the nuc buffers (issue #212); each material owns its
    /// fine grid, described by `fine_meta`. Distinct from the GLOBAL
    /// `log_energy_grid` (retained for the score lookup). Bit-twin of the kernel.
    pub fine_log_energy_grid: &'a [f64],
    /// Packed `[n_materials × FINE_META_COLS]` per-material fine-grid descriptor
    /// (issue #212): base into `fine_log_energy_grid` (== aggregate-XS row base),
    /// fine length, and first-slab element base into `nuc_macro_total`. Bit-twin
    /// of the kernel's `fine_meta` binding.
    pub fine_meta: &'a [u32],
    pub xs_elastic_per_material: &'a [f64],
    pub xs_absorption_per_material: &'a [f64],
    pub xs_inelastic_per_material: &'a [f64],
    pub xs_fission_per_material: &'a [f64],
    pub nu_bar_per_material: &'a [f64],
    /// Delayed-neutron fraction `beta(E) = nu_d(E) / nu_t(E)`, same flat
    /// `[n_materials x fine_n]` shape and indexing as `nu_bar_per_material`
    /// (issue #364). Bit-twin of the kernel's `beta_delayed_per_material`.
    pub beta_delayed_per_material: &'a [f64],
    pub target_mass_per_material: &'a [f64],
    pub temperature_k_per_material: &'a [f64],
    // Per-collision nuclide selection (issue #74). See `NuclideSelectInputs`.
    pub nuc_macro_total: &'a [f64],
    pub nuc_awr: &'a [f64],
    pub mat_nuclide_meta: &'a [u32],
    // Per-(slab, energy) reaction partials, density-weighted, packed
    // `[n_slab x n_grid x NUC_PARTIAL_COLS]` (#74 Stage 2b). See
    // `NuclideSelectInputs::nuc_partial_xs`.
    pub nuc_partial_xs: &'a [f64],
    // Fission outgoing energy
    pub fission_a_per_material: &'a [f64],
    pub fission_b_per_material: &'a [f64],
    // Two chi ROWS per material (issue #364): row `2*mat` is the prompt spectrum,
    // row `2*mat + 1` the delayed groups' folded spectrum. Every buffer below is
    // indexed by chi row, not by material.
    pub fission_eout_kind_per_material: &'a [u32],
    pub fission_eout_n_energies_per_material: &'a [u32],
    // Tight CSR (issue #104): `fission_eout_ae_offset` is the per-chi-row
    // global ae-row base; `fission_eout_x_offset` the per-ae-row global
    // (x, cdf)-point base.
    pub fission_eout_ae_offset: &'a [u32],
    pub fission_eout_energy_grid_per_material: &'a [f64],
    pub fission_eout_n_x_per_material: &'a [u32],
    pub fission_eout_x_offset: &'a [u32],
    pub fission_eout_x_per_material: &'a [f64],
    pub fission_eout_cdf_per_material: &'a [f64],
    // PDF alongside the CDF (same shape as `fission_eout_x`) plus the
    // per-ae-row interpolation code (0 histogram, 1 lin-lin) for the
    // interp-aware fission E_out inversion.
    pub fission_eout_p_per_material: &'a [f64],
    pub fission_eout_interp_per_material: &'a [u32],
    // Per-MT inelastic (SPARSE, issue #212). `permt_meta` is `[n_slab ×
    // MT_INELASTIC_COUNT × PERMT_META_COLS]`; the two value buffers hold each
    // (slab, MT slot)'s nonzero coarse-grid range tight, sharing the meta row.
    pub xs_inelastic_per_mt_sparse: &'a [f64],
    pub q_inelastic_per_mt: &'a [f64],
    pub yield_per_mt_sparse: &'a [f64],
    pub permt_meta: &'a [u32],
    pub scatter_in_cm_per_mt: &'a [u32],
    // Inelastic angle distribution (per (mat, MT slot))
    pub angle_n_energies: &'a [u32],
    pub angle_ae_offset: &'a [u32],
    pub angle_energy_grid: &'a [f64],
    pub angle_n_mu: &'a [u32],
    pub angle_mu_offset: &'a [u32],
    pub angle_mu: &'a [f64],
    pub angle_cdf: &'a [f64],
    pub angle_pdf: &'a [f64],
    pub angle_interp: &'a [u32],
    // Outgoing-energy (uncorrelated tabular). Tight CSR (issue #104):
    // `eout_ae_offset` is the per-(slab,MT) global ae-row base;
    // `eout_x_offset` the per-ae-row global x-point base.
    pub eout_kind: &'a [u32],
    pub eout_n_energies: &'a [u32],
    pub eout_ae_offset: &'a [u32],
    pub eout_energy_grid: &'a [f64],
    pub eout_n_x: &'a [u32],
    pub eout_x_offset: &'a [u32],
    pub eout_x: &'a [f64],
    pub eout_cdf: &'a [f64],
    pub eout_histogram_interp: &'a [u32],
    pub eout_p: &'a [f64],
    pub eout_interp: &'a [u32],
    pub eout_n_discrete: &'a [u32],
    // Correlated angle-energy
    // Correlated angle-energy (tight CSR, issue #104): `corr_ae_offset` is the
    // per-(slab,MT) global ae-row base; `corr_x_offset` the per-ae-row global
    // x-point base; `corr_mu_offset` the per-x-point global mu-point base.
    pub corr_n_energies: &'a [u32],
    /// Per-MT count of equally-weighted correlated components (issue #111).
    /// `>= 2` triggers a per-collision component pick in dispatch.
    pub corr_n_components: &'a [u32],
    pub corr_ae_offset: &'a [u32],
    pub corr_energy_grid: &'a [f64],
    pub corr_n_x: &'a [u32],
    pub corr_x_offset: &'a [u32],
    pub corr_x: &'a [f64],
    pub corr_cdf: &'a [f64],
    pub corr_p: &'a [f64],
    pub corr_interp: &'a [u32],
    pub corr_n_discrete: &'a [u32],
    pub corr_n_mu: &'a [u32],
    pub corr_mu_offset: &'a [u32],
    pub corr_mu: &'a [f64],
    pub corr_mu_cdf: &'a [f64],
    pub corr_mu_pdf: &'a [f64],
    pub corr_mu_interp: &'a [u32],
    // Elastic angle distribution
    pub elastic_angle_n_energies: &'a [u32],
    pub elastic_angle_ae_offset: &'a [u32],
    pub elastic_angle_energy_grid: &'a [f64],
    pub elastic_angle_n_mu: &'a [u32],
    pub elastic_angle_mu_offset: &'a [u32],
    pub elastic_angle_mu: &'a [f64],
    pub elastic_angle_cdf: &'a [f64],
    pub elastic_angle_pdf: &'a [f64],
    pub elastic_angle_interp: &'a [u32],
    // Kalbach-Mann
    pub km_n_energies: &'a [u32],
    /// Tight CSR (issue #104): `km_ae_offset` is the per-(slab,MT) global
    /// ae-row base; `km_x_offset` the per-ae-row global x-point base.
    pub km_ae_offset: &'a [u32],
    pub km_energy_grid: &'a [f64],
    pub km_interp: &'a [u32],
    pub km_n_discrete: &'a [u32],
    pub km_n_x: &'a [u32],
    pub km_x_offset: &'a [u32],
    pub km_x: &'a [f64],
    pub km_p: &'a [f64],
    pub km_c: &'a [f64],
    pub km_r: &'a [f64],
    pub km_a: &'a [f64],
    // Evaporation
    pub evap_n_energies: &'a [u32],
    pub evap_n_components: &'a [u32],
    /// Tight CSR (issue #104): `evap_ae_offset` is the per-(slab,MT) global
    /// E_in-row base into `evap_energy_grid` / `evap_u`; `evap_theta_offset`
    /// the per-(slab,MT) base into the component-major `evap_theta`.
    pub evap_ae_offset: &'a [u32],
    pub evap_theta_offset: &'a [u32],
    pub evap_energy_grid: &'a [f64],
    pub evap_theta: &'a [f64],
    pub evap_u: &'a [f64],
    // N-body phase-space
    pub nbps_n_bodies: &'a [u32],
    pub nbps_total_mass: &'a [f64],
    // Maxwell
    pub maxwell_n_energies: &'a [u32],
    /// Tight CSR (issue #104): per-(slab,MT) global E_in-row base into
    /// `maxwell_energy_grid` / `maxwell_theta`.
    pub maxwell_ae_offset: &'a [u32],
    pub maxwell_energy_grid: &'a [f64],
    pub maxwell_theta: &'a [f64],
    pub maxwell_u: &'a [f64],
    // Watt
    pub watt_n_energies: &'a [u32],
    /// Tight CSR (issue #104): per-(slab,MT) global E_in-row base into
    /// `watt_energy_grid` / `watt_a` / `watt_b`.
    pub watt_ae_offset: &'a [u32],
    pub watt_energy_grid: &'a [f64],
    pub watt_a: &'a [f64],
    pub watt_b: &'a [f64],
    pub watt_u: &'a [f64],
    // Tallies
    pub tallies: &'a TalliesPack,
    /// Padded SCORE_PER_MT lookup buffer -- the driver pads to a
    /// single zero slot when `xs_score_per_mt` was empty so the
    /// indexing math here stays valid.
    pub xs_score_per_mt_slice: &'a [f64],
    pub n_score_mts: usize,
    // Pre-cached derived sizes (computed once in the driver)
    pub n_cells: usize,
    pub n_surfaces: usize,
    pub n_grid: usize,
    pub max_steps: u32,
    /// Survival biasing (implicit capture + weight-cutoff roulette) enable
    /// flag, mirroring the kernel's `survival_params[0] != 0.0` gate. When
    /// `false` the reaction-selection block is byte-identical to the analog
    /// twin and no roulette draw fires.
    pub survival_enabled: bool,
    /// Weight-cutoff roulette trigger (kernel `survival_params[1]`). Only
    /// read when `survival_enabled`.
    pub weight_cutoff: f64,
    /// Roulette survivor weight (kernel `survival_params[2]`). Only read
    /// when `survival_enabled`.
    pub weight_survive: f64,
    /// Device-fission-bank gate, mirroring the kernel's
    /// `fission_bank_enabled[0] == 1`. When `false` the fission branch keeps
    /// the legacy `weight *= nu_bar` + `FISSION_WEIGHT_CAP` terminator; when
    /// `true` it stochastically rounds nu_bar to N, continues progeny 0 in the
    /// current walk, and advances the RNG for the N-1 progeny the kernel banks.
    /// The CPU mirror tracks one particle (it has no bank), so it advances the
    /// same RNG stream for the continuing walk but does not retain the banked
    /// secondaries.
    pub fission_bank_enabled: bool,
    /// Free-gas resonance/thermal cutoff multiplier (the model option of the
    /// same name, default `400.0`). The elastic branch treats the target as a
    /// free Maxwell-Boltzmann gas below `free_gas_threshold * kT` and as a cold
    /// target at or above it. Mirrors the kernel's `free_gas_threshold` buffer;
    /// both backends pass the same value (issue #102).
    pub free_gas_threshold: f64,
    /// Per-(material, nuclide) URR probability tables, bound exactly as the
    /// kernel binds them. The mirror perturbs the macroscopic partials with
    /// these the same way the kernel does, so a URR material is no longer
    /// invisible to the twin-vs-kernel equivalence tests (issue #342).
    pub urr_meta: &'a [u32],
    pub urr_ae_offset: &'a [u32],
    pub urr_cdf_offset: &'a [u32],
    pub urr_energy_grid: &'a [f64],
    pub urr_cdf: &'a [f64],
    pub urr_xs: &'a [f64],
    pub urr_atom_density: &'a [f64],
}

/// Validate every buffer-length invariant the per-particle loop
/// assumes. Pulled out of the drivers so neither has to repeat the
/// assertion block. Panics on the first violation -- the messages
/// match the historical ones in `cpu.rs` so existing test failures
/// stay searchable.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_transport_inputs(
    tallies: &TalliesPack,
    surface_types: &[u32],
    surface_boundaries: &[u32],
    cell_to_material: &[u32],
    n_cells: usize,
    n_materials: usize,
    // Per-(material, nuclide) slab count (#74 Stage 2b). Per-MT inelastic
    // pools are keyed by slab, not material; single-nuclide materials give
    // `n_slab == n_materials`.
    n_slab: usize,
    // Per-material coarse-grid descriptor (issue #212), packed
    // `[n_materials × COARSE_META_COLS]`, plus the per-material nuclide-selection
    // meta (stride 2) so the tight-CSR per-MT length can be summed over slabs.
    coarse_meta: &[u32],
    // Per-material FINE-grid descriptor (issue #212), packed
    // `[n_materials × FINE_META_COLS]`, so the tight-CSR aggregate-XS length can
    // be summed over materials (`sum_m fine_n[m]`).
    fine_meta: &[u32],
    mat_nuclide_meta: &[u32],
    xs_elastic_per_material: &[f64],
    xs_absorption_per_material: &[f64],
    xs_inelastic_per_material: &[f64],
    xs_fission_per_material: &[f64],
    nu_bar_per_material: &[f64],
    beta_delayed_per_material: &[f64],
    fission_a_per_material: &[f64],
    fission_b_per_material: &[f64],
    fission_eout_kind_per_material: &[u32],
    fission_eout_n_energies_per_material: &[u32],
    fission_eout_ae_offset: &[u32],
    fission_eout_energy_grid_per_material: &[f64],
    fission_eout_n_x_per_material: &[u32],
    fission_eout_x_offset: &[u32],
    fission_eout_x_per_material: &[f64],
    fission_eout_cdf_per_material: &[f64],
    fission_eout_p_per_material: &[f64],
    fission_eout_interp_per_material: &[u32],
    xs_inelastic_per_mt_sparse: &[f64],
    q_inelastic_per_mt: &[f64],
    yield_per_mt_sparse: &[f64],
    permt_meta: &[u32],
    angle_n_energies: &[u32],
    angle_ae_offset: &[u32],
    angle_energy_grid: &[f64],
    angle_n_mu: &[u32],
    angle_mu_offset: &[u32],
    angle_mu: &[f64],
    angle_cdf: &[f64],
    scatter_in_cm_per_mt: &[u32],
    eout_kind: &[u32],
    eout_n_energies: &[u32],
    eout_ae_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    eout_x_offset: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    eout_histogram_interp: &[u32],
    eout_p: &[f64],
    eout_interp: &[u32],
    eout_n_discrete: &[u32],
    corr_n_energies: &[u32],
    corr_n_components: &[u32],
    corr_ae_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    corr_x_offset: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    corr_p: &[f64],
    corr_interp: &[u32],
    corr_n_discrete: &[u32],
    corr_n_mu: &[u32],
    corr_mu_offset: &[u32],
    corr_mu: &[f64],
    corr_mu_cdf: &[f64],
    corr_mu_pdf: &[f64],
    corr_mu_interp: &[u32],
) {
    let n_tallies = tallies.n_tallies() as usize;
    assert!(n_tallies > 0, "tallies pack must have at least one tally");
    assert_eq!(cell_to_material.len(), n_cells);
    assert_eq!(
        surface_boundaries.len(),
        surface_types.len(),
        "surface_boundaries must match surface_types length"
    );
    // Aggregate macro XS ride the per-material FINE grid, tight CSR
    // `[sum_m fine_n[m]]` (issue #212). `n_grid` (global) no longer sizes them.
    assert_eq!(fine_meta.len(), n_materials * FINE_META_COLS as usize);
    let expected_fine_agg: usize = (0..n_materials)
        .map(|m| fine_meta[m * FINE_META_COLS as usize + COL_FINE_N as usize] as usize)
        .sum();
    assert_eq!(xs_elastic_per_material.len(), expected_fine_agg);
    assert_eq!(xs_absorption_per_material.len(), expected_fine_agg);
    assert_eq!(xs_inelastic_per_material.len(), expected_fine_agg);
    assert_eq!(xs_fission_per_material.len(), expected_fine_agg);
    assert_eq!(nu_bar_per_material.len(), expected_fine_agg);
    assert_eq!(
        beta_delayed_per_material.len(),
        expected_fine_agg,
        "beta_delayed_per_material must ride the per-material fine grid, like nu_bar"
    );
    assert_eq!(fission_a_per_material.len(), n_materials);
    assert_eq!(fission_b_per_material.len(), n_materials);
    // Variable-length tight layout (issue #104): per-material counts/bases are
    // [n_materials]; incident-energy rows are [total ae-rows]; (x, cdf) points
    // are [total x-points]. No fixed per-axis stride.
    let n_fission_eout_ae_rows = fission_eout_n_x_per_material.len();
    // TWO chi rows per material: prompt at `2*mat`, delayed at `2*mat + 1`
    // (issue #364).
    let n_chi_rows = 2 * n_materials;
    assert_eq!(fission_eout_kind_per_material.len(), n_chi_rows);
    assert_eq!(fission_eout_n_energies_per_material.len(), n_chi_rows);
    assert_eq!(fission_eout_ae_offset.len(), n_chi_rows);
    assert_eq!(
        fission_eout_energy_grid_per_material.len(),
        n_fission_eout_ae_rows
    );
    assert_eq!(fission_eout_x_offset.len(), n_fission_eout_ae_rows);
    assert_eq!(
        fission_eout_cdf_per_material.len(),
        fission_eout_x_per_material.len()
    );
    assert_eq!(
        fission_eout_p_per_material.len(),
        fission_eout_x_per_material.len()
    );
    assert_eq!(
        fission_eout_interp_per_material.len(),
        n_fission_eout_ae_rows
    );
    assert_eq!(coarse_meta.len(), n_materials * COARSE_META_COLS as usize);
    // `mat_nuclide_meta` is [n_materials × 2] ([nuc_off, nuc_count]); the per-
    // material nuclide counts must sum to n_slab (the total slab count the per-
    // slab buffers below are keyed on).
    assert_eq!(mat_nuclide_meta.len(), n_materials * 2);
    let total_slabs: usize = (0..n_materials)
        .map(|m| mat_nuclide_meta[m * 2 + 1] as usize)
        .sum();
    assert_eq!(total_slabs, n_slab);
    // SPARSE per-MT storage (issue #212): `permt_meta` is `[n_slab ×
    // MT_INELASTIC_COUNT × PERMT_META_COLS]`; the two value buffers are the tight
    // concatenation of every (slab, MT slot)'s stored range (equal length, xs and
    // yield share the range).
    assert_eq!(
        permt_meta.len(),
        n_slab * MT_INELASTIC_COUNT * PERMT_META_COLS as usize
    );
    let expected_sparse: usize = (0..permt_meta.len() / PERMT_META_COLS as usize)
        .map(|r| permt_meta[r * PERMT_META_COLS as usize + COL_PERMT_N_STORED as usize] as usize)
        .sum();
    assert_eq!(xs_inelastic_per_mt_sparse.len(), expected_sparse);
    assert_eq!(yield_per_mt_sparse.len(), expected_sparse);
    assert_eq!(q_inelastic_per_mt.len(), n_slab * MT_INELASTIC_COUNT);
    // Variable-length tight layout (issue #104): per-(slab,MT) counts/bases are
    // [n_slab × MT_INELASTIC_COUNT]; incident-energy rows are [total ae-rows];
    // (mu,cdf) points are [total mu-points]. No fixed per-axis stride.
    let n_inel_ae_rows = angle_n_mu.len();
    assert_eq!(angle_n_energies.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(angle_ae_offset.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(angle_energy_grid.len(), n_inel_ae_rows);
    assert_eq!(angle_mu_offset.len(), n_inel_ae_rows);
    assert_eq!(angle_cdf.len(), angle_mu.len());
    assert_eq!(scatter_in_cm_per_mt.len(), n_slab * MT_INELASTIC_COUNT);
    // Variable-length tight layout (issue #104): per-(slab,MT) scalars/bases are
    // [n_slab × MT_INELASTIC_COUNT]; incident-energy rows are [total ae-rows];
    // (x,cdf,p) points are [total x-points]. No fixed per-axis stride.
    let n_eout_ae_rows = eout_n_x.len();
    assert_eq!(eout_kind.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(eout_n_energies.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(eout_ae_offset.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(eout_energy_grid.len(), n_eout_ae_rows);
    assert_eq!(eout_x_offset.len(), n_eout_ae_rows);
    assert_eq!(eout_cdf.len(), eout_x.len());
    assert_eq!(eout_p.len(), eout_x.len());
    assert_eq!(eout_histogram_interp.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(eout_interp.len(), n_eout_ae_rows);
    assert_eq!(eout_n_discrete.len(), n_eout_ae_rows);
    // Variable-length tight layout (issue #104), three nesting levels: per
    // (slab,MT) counts/bases are [n_slab × MT_INELASTIC_COUNT]; ae-rows are
    // [total ae-rows]; E_out x-points are [total x-points]; mu points are
    // [total mu-points]. No fixed per-axis stride.
    let n_corr_ae_rows = corr_n_x.len();
    let n_corr_x_points = corr_n_mu.len();
    assert_eq!(corr_n_energies.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(corr_n_components.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(corr_ae_offset.len(), n_slab * MT_INELASTIC_COUNT);
    assert_eq!(corr_energy_grid.len(), n_corr_ae_rows);
    assert_eq!(corr_x_offset.len(), n_corr_ae_rows);
    assert_eq!(corr_x.len(), n_corr_x_points);
    assert_eq!(corr_cdf.len(), corr_x.len());
    assert_eq!(corr_p.len(), corr_x.len());
    assert_eq!(corr_interp.len(), n_corr_ae_rows);
    assert_eq!(corr_n_discrete.len(), n_corr_ae_rows);
    assert_eq!(corr_mu_offset.len(), n_corr_x_points);
    assert_eq!(corr_mu_cdf.len(), corr_mu.len());
    assert_eq!(corr_mu_pdf.len(), corr_mu.len());
    assert_eq!(corr_mu_interp.len(), n_corr_x_points);
    assert_eq!(tallies.cell_to_bin.len(), n_tallies * n_cells);
    assert_eq!(tallies.n_cells_per_tally.len(), n_tallies);
    assert_eq!(tallies.edges_offsets.len(), n_tallies);
    assert_eq!(tallies.n_bins_per_tally.len(), n_tallies);
    assert_eq!(tallies.out_offsets.len(), n_tallies + 1);
    for t in 0..n_tallies {
        let n_bins = tallies.n_bins_per_tally[t];
        assert!(n_bins >= 1, "tally {t} must have at least one energy bin");
        let off = tallies.edges_offsets[t] as usize;
        let end = off + n_bins as usize + 1;
        for (i, w) in tallies.log_edges[off..end].windows(2).enumerate() {
            assert!(
                w[1] > w[0],
                "tally {t} log_edges must be strictly increasing (bin {i})"
            );
        }
    }
}

/// Per-particle outcome returned by [`transport_one_particle`].
pub(super) struct ParticleOutcome {
    pub alive: u32,
    pub n_steps: u32,
    pub final_energy: f64,
    /// Set when the history ended in no cell, i.e. it was lost to a geometry
    /// gap (issue #289). Mirrors the kernel's `record_lost` so the twin reports
    /// the same losses the GPU does.
    pub lost: Option<crate::common::lost_particles::LostParticleRecord>,
    /// (n,xn) secondaries this history queued while [`PEND_SLOTS`] were already
    /// outstanding, i.e. the ones the cubecl kernel would have spilled to the
    /// device particle bank for the host to drain in a later pass (issue #111
    /// phase 2). The twin transports them in-thread, so this is a diagnostic
    /// only: it measures how often the kernel's spill path is reached.
    pub n_spilled: u32,
    /// Deepest the pending-secondary stack got during this history, i.e. how
    /// many of the kernel's [`PEND_SLOTS`] register slots were actually needed.
    /// Reported next to [`n_spilled`](Self::n_spilled) so the slot count can be
    /// justified by the headroom it leaves, not only by the absence of spills.
    pub max_pend_depth: u32,
}

/// One neutron collision recorded by the CPU twin, for the issue-#40
/// matched-stream per-history CPU-vs-GPU diff harness.
///
/// Emitted only when a trace sink (`Some(&mut Vec<_>)`) is passed to
/// [`transport_one_particle`]; the production / kernel paths pass `None` and
/// the recording branch is never taken, so the per-step stream stays
/// byte-identical. The fields mirror what the production CPU's track capture
/// records per collision (`TrackEvent::{energy_in, energy_out, reaction_mt}`),
/// so the harness can diff the two sequences interaction-by-interaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionRecord {
    /// Transport step index at which the collision occurred.
    pub step: u32,
    /// Neutron energy entering the collision (eV).
    pub energy_in: f64,
    /// Neutron energy leaving the collision (eV); equals `energy_in` for a
    /// terminal absorption (the energy is unchanged before the kill).
    pub energy_out: f64,
    /// ENDF-style reaction code: `2` elastic, `18` fission, `51 + slot`
    /// inelastic (MT 51..91), `102` absorption. The twin selects only the
    /// reaction *class* via the shared PCG stream and does not resolve the
    /// specific capture / charged-particle absorption MT, so all absorptions
    /// are reported as `102`.
    pub reaction: i32,
}

/// CPU twin of the kernel's `sample_fission_progeny_energy`: pick the prompt or
/// the delayed spectrum, then sample it (issue #364).
///
/// `beta` is the material's delayed fraction `nu_d(E) / nu_t(E)` at the incident
/// energy. ONE uniform, drawn only when `beta > 0.0`, so a material with no delayed
/// data keeps the previous draw schedule exactly.
fn fission_progeny_energy_cpu(
    inputs: &TransportInputs<'_>,
    mat_idx: usize,
    beta: f64,
    e_in: f64,
    state: &mut u64,
) -> f64 {
    let mut chi_row = 2 * mat_idx;
    if beta > 0.0 {
        let (xi_del, st_del) = crate::common::pcg32::draw_uniform_cpu(*state);
        *state = st_del;
        if xi_del < beta {
            chi_row = 2 * mat_idx + 1;
        }
    }
    fission_chi_cpu(inputs, chi_row, mat_idx, e_in, state)
}

/// CPU twin of the kernel's `sample_fission_chi`: sample one fission outgoing
/// energy from chi row `chi_row` at incident energy `e_in`, advancing `state`
/// exactly as the kernel does (so the continuing fission progeny's RNG stream
/// stays in lockstep). Dispatches on the per-row `fission_eout_kind`
/// (ContinuousTabular / Maxwell / Evaporation), falling back to the inline
/// Watt-rejection draw -- the exact logic that previously lived inline in the CPU
/// twin's fission branch.
///
/// Rows come in (prompt, delayed) pairs: material `m`'s prompt spectrum is row
/// `2m`, its delayed spectrum row `2m + 1` (issue #364). Callers go through
/// [`fission_progeny_energy_cpu`], which picks the row.
fn fission_chi_cpu(
    inputs: &TransportInputs<'_>,
    chi_row: usize,
    mat_idx: usize,
    e_in: f64,
    state: &mut u64,
) -> f64 {
    let fission_kind = inputs.fission_eout_kind_per_material[chi_row];
    let e_tab = if fission_kind == EOUT_KIND_CONTINUOUS_TABULAR {
        // Tight CSR (issue #104): the material's ae-rows start at `eg_off`;
        // (x, cdf) are read from the full arrays via the per-row global
        // `fission_eout_x_offset`. No per-axis stride.
        let n_e = inputs.fission_eout_n_energies_per_material[chi_row] as usize;
        let eg_off = inputs.fission_eout_ae_offset[chi_row] as usize;
        if n_e > 0 {
            yamc_physics::gpu::flat::fission_eout_continuous::sample_fission_eout_continuous(
                e_in,
                &inputs.fission_eout_energy_grid_per_material[eg_off..eg_off + n_e],
                &inputs.fission_eout_n_x_per_material[eg_off..eg_off + n_e],
                inputs.fission_eout_x_per_material,
                inputs.fission_eout_cdf_per_material,
                inputs.fission_eout_p_per_material,
                &inputs.fission_eout_x_offset[eg_off..eg_off + n_e],
                &inputs.fission_eout_interp_per_material[eg_off..eg_off + n_e],
                state,
            )
        } else {
            None
        }
    } else if fission_kind == EOUT_KIND_MAXWELL || fission_kind == EOUT_KIND_EVAPORATION {
        // Maxwell / Evaporation prompt-fission χ. Tight CSR (issue #104):
        // each E_in row carries one point, so θ at row i is
        // `fission_eout_x[fission_eout_x_offset[eg_off + i]]` and the scalar
        // `u` is in the material's row-0 single slot
        // `fission_eout_cdf[fission_eout_x_offset[eg_off]]`. Gather θ into a
        // contiguous slice for the shared CPU samplers, which advance the PCG
        // stream identically to the kernel's 32-iteration `*_rejection_draw`
        // loop. `None` on exhaustion falls through to the Watt branch --
        // matching the kernel's `sampled_from_table == 0` fall-through.
        let n_e = inputs.fission_eout_n_energies_per_material[chi_row] as usize;
        let eg_off = inputs.fission_eout_ae_offset[chi_row] as usize;
        if n_e > 0 {
            let grid = &inputs.fission_eout_energy_grid_per_material[eg_off..eg_off + n_e];
            let theta: Vec<f64> = (0..n_e)
                .map(|i| {
                    let x_off = inputs.fission_eout_x_offset[eg_off + i] as usize;
                    inputs.fission_eout_x_per_material[x_off]
                })
                .collect();
            let u_off = inputs.fission_eout_x_offset[eg_off] as usize;
            let u = inputs.fission_eout_cdf_per_material[u_off];
            if fission_kind == EOUT_KIND_MAXWELL {
                yamc_physics::gpu::flat::maxwell::sample_maxwell(e_in, grid, &theta, u, state)
            } else {
                yamc_physics::gpu::flat::evaporation::sample_evaporation(
                    e_in, grid, &theta, u, state,
                )
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Some(e) = e_tab {
        e
    } else {
        // The Watt fall-through's parameters are per MATERIAL, not per chi row --
        // the kernel is handed them from `mat_f64_meta` the same way.
        let watt_a = inputs.fission_a_per_material[mat_idx];
        let watt_b = inputs.fission_b_per_material[mat_idx];

        let s_xw1 = *state;
        let r_xw1 = pcg_out(s_xw1);
        *state = s_xw1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xw1 = (r_xw1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let s_xw2 = *state;
        let r_xw2 = pcg_out(s_xw2);
        *state = s_xw2.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xw2 = (r_xw2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let s_xw3 = *state;
        let r_xw3 = pcg_out(s_xw3);
        *state = s_xw3.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xw3 = (r_xw3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let cos_arg = std::f64::consts::FRAC_PI_2 * xw3;
        let c = cos_arg.cos();
        let w_max = -watt_a * (xw1.ln() + xw2.ln() * c * c);

        let s_xw4 = *state;
        let r_xw4 = pcg_out(s_xw4);
        *state = s_xw4.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xw4 = (r_xw4 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let a2b = watt_a * watt_a * watt_b;
        let u_sym = 2.0 * xw4 - 1.0;
        let mut e_out = w_max + 0.25 * a2b + u_sym * (a2b * w_max).sqrt();
        if e_out <= 0.0 {
            e_out = 1.0e-6;
        }
        e_out
    }
}

/// Transport particle `i` to completion, scoring tallies into
/// `tally_acc`. Each particle's history is independent: serial and
/// rayon-parallel callers produce bit-identical results.
///
/// Mirrors the cubecl kernel's algorithm exactly so the
/// `cpu_gpu_equivalence_*` tests stay green.
#[inline]
pub(super) fn transport_one_particle(
    inputs: &TransportInputs<'_>,
    i: usize,
    tally_acc: &mut [u64],
    // Issue-#40 verification trace sink. `None` on the production / kernel
    // paths (zero cost, byte-identical stream); `Some` only from the
    // traced CPU driver, which records one `CollisionRecord` per collision.
    mut trace: Option<&mut Vec<CollisionRecord>>,
) -> ParticleOutcome {
    let i3 = i * 3;
    let mut energy = inputs.energies_in[i];
    let mut px = inputs.positions_in[i3];
    let mut py = inputs.positions_in[i3 + 1];
    let mut pz = inputs.positions_in[i3 + 2];
    let mut dx = inputs.directions_in[i3];
    let mut dy = inputs.directions_in[i3 + 1];
    let mut dz = inputs.directions_in[i3 + 2];
    // Expand the 32-bit per-history seed into the 64-bit PCG state via
    // splitmix64 (issue #274), exactly as the kernel does in-kernel.
    // `walk_seed` tracks whose stream the thread is on: the source particle's,
    // then each popped secondary's own (issue #111).
    let mut walk_seed = inputs.seeds[i];
    let mut state = expand_seed(walk_seed);
    let mut walk_secondaries = 0u32;
    let mut alive = 1u32;
    let mut n_steps = 0u32;
    let mut weight = 1.0;
    // URR band held by this walk (issue #342): the base uniform is drawn once
    // per ENERGY, not per step, so an isotope keeps one resonance realisation
    // across boundary crossings and void excursions until a collision moves
    // the neutron off that energy. `None` is the CPU's `NO_URR`.
    let mut urr_held: Option<(f64, f64)> = None;

    // In-thread pending-secondary stack (issue #274): analog multiplicity queues
    // the extra secondaries and this walk transports them after the current
    // particle dies, keeping every secondary of the history in this one call.
    // Each entry carries its own identity-derived collision seed (issue #111
    // phase 1). Fission progeny join the same stack when the fission bank is on,
    // so the twin transports the population the production CPU's `ParticleBank`
    // does, in the same LIFO order.
    //
    // The kernel keeps [`PEND_SLOTS`] of the (n,xn) ones in registers and spills
    // deeper ones to the device bank for the host to drain; the twin is a `Vec`,
    // so it transports them all here. Same physics either way (same seeds, same
    // kinematics, and the twin has no per-history variance to get wrong), and it
    // makes `n_spilled` an exact count of what the kernel would have spilled:
    // with both draining LIFO, the depth at each push is the same. `n_pend_nxn`
    // is that depth -- the (n,xn) entries only, since a fission progeny never
    // occupies a register slot. `Vec::new` does not allocate, so a history that
    // never multiplies pays nothing.
    let mut pend: Vec<PendEntry> = Vec::new();
    let mut n_pend_nxn = 0usize;
    let mut n_spilled = 0u32;
    let mut max_pend_depth = 0u32;

    let n_grid = inputs.n_grid;
    let n_cells = inputs.n_cells;
    let n_surfaces = inputs.n_surfaces;
    // Cell of the previous step, for the lost-particle record (issue #289).
    let mut last_cell: Option<usize> = None;
    let mut lost: Option<crate::common::lost_particles::LostParticleRecord> = None;

    let mut step = 0u32;
    while (alive == 1 && step < inputs.max_steps) || !pend.is_empty() {
        // Current particle finished but secondaries are queued: pop the next one
        // and keep transporting inside the same history (issue #274), exact twin
        // of the kernel's top-of-loop pop -- plus the re-seed of the thread PCG
        // from the popped secondary's own stream (issue #111), which is what
        // makes the choice below immaterial.
        if alive == 0 || step >= inputs.max_steps {
            let e = match inputs.pend_drain {
                PendDrain::Fifo => pend.remove(0),
                PendDrain::Lifo => pend
                    .pop()
                    .expect("loop condition guarantees a pending entry"),
            };
            if e.nxn {
                // The kernel's register slot is reclaimed on pop.
                n_pend_nxn -= 1;
            }
            energy = e.energy;
            px = e.px;
            py = e.py;
            pz = e.pz;
            dx = e.dx;
            dy = e.dy;
            dz = e.dz;
            weight = e.weight;
            walk_seed = e.seed;
            state = expand_seed(walk_seed);
            walk_secondaries = 0;
            alive = 1;
            // A popped secondary is its own particle and holds no band yet,
            // matching `Particle::new`'s `NO_URR` (issue #342).
            urr_held = None;
            step = 0;
        }

        // 1. Cell-finding via stackless BVH traversal.
        let cell = bvh_find_cell_at_point(
            px,
            py,
            pz,
            inputs.bvh_aabbs,
            inputs.bvh_meta,
            inputs.bvh_prims,
            inputs.bvh_unb,
            inputs.cell_aabbs,
            inputs.region_program,
            inputs.surface_types,
            inputs.surface_params,
        );

        if cell == CELL_NOT_FOUND {
            // No cell covers this point: a lost particle, exactly as the
            // kernel's `record_lost` site treats it (issue #289).
            lost = Some(crate::common::lost_particles::LostParticleRecord {
                position: [px, py, pz],
                direction: [dx, dy, dz],
                energy,
                last_cell_index: last_cell,
            });
            alive = 0;
        } else {
            last_cell = Some(cell as usize);
            let mat_idx = inputs.cell_to_material[cell as usize] as usize;
            // `target_mass` is the density-weighted material average; a
            // multi-nuclide collision overrides it with the struck nuclide's
            // exact AWR (issue #74), mirrored from the kernel below.
            let mut target_mass = inputs.target_mass_per_material[mat_idx];

            // 2. XS lookup. Bracket search runs in log-E (the
            // grid is sorted in log just as it is in linear), but
            // the interpolation factor is computed in **linear E**
            // to mirror CPU's `lookup_xs_by_mt`. See the cube
            // kernel comment near the matching code.
            let log_e = energy.ln();
            // GLOBAL-grid bracket -- backs the grid-shared score lookup
            // (`xs_score_per_mt` stays on `log_energy_grid`). Bit-twin of the
            // kernel's global bracket.
            let lo = inputs.log_energy_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, n_grid - 1);
            let idx_lo = idx_hi - 1;
            let e_lo_lin = inputs.log_energy_grid[idx_lo].exp();
            let e_hi_lin = inputs.log_energy_grid[idx_hi].exp();
            let denom = e_hi_lin - e_lo_lin;
            let frac = if denom > 0.0 {
                (energy - e_lo_lin) / denom
            } else {
                0.0
            };
            // Per-material FINE-grid bracket (issue #212); bit-twin of the
            // kernel's fine bracket. This material owns its fine grid: the slice
            // `fine_log_energy_grid[fine_off .. fine_off + fine_n]` (stride-
            // FINE_META_COLS `fine_meta` cols GRID_OFFSET / N). `idx_lo_f` /
            // `idx_hi_f` stay RELATIVE to `fine_off`, which doubles as the
            // aggregate-XS row base; `fine_nuc_base` (col 2) is the first-slab
            // base into the nuc buffers. Single-material => base 0, full length,
            // so bit-identical to the old single shared fine grid.
            let fine_meta_off = mat_idx * FINE_META_COLS as usize;
            let fine_off = inputs.fine_meta[fine_meta_off + COL_FINE_GRID_OFFSET as usize] as usize;
            let fine_n = inputs.fine_meta[fine_meta_off + COL_FINE_N as usize] as usize;
            let fine_nuc_base =
                inputs.fine_meta[fine_meta_off + COL_FINE_NUC_BASE as usize] as usize;
            let fine_slice = &inputs.fine_log_energy_grid[fine_off..fine_off + fine_n];
            let lo_f = fine_slice.partition_point(|&x| x < log_e);
            let idx_hi_f = lo_f.clamp(1, fine_n - 1);
            let idx_lo_f = idx_hi_f - 1;
            let e_lo_f = fine_slice[idx_lo_f].exp();
            let e_hi_f = fine_slice[idx_hi_f].exp();
            let denom_f = e_hi_f - e_lo_f;
            let frac_f = if denom_f > 0.0 {
                (energy - e_lo_f) / denom_f
            } else {
                0.0
            };
            // Coarse-grid bracket for the per-MT inelastic / yield buffers
            // (issue #88 / #212); bit-twin of the kernel's coarse bracket. This
            // material owns its coarse grid: the slice
            // `coarse_log_energy_grid[coarse_base .. coarse_base + coarse_n]`
            // (stride-COARSE_META_COLS `coarse_meta` cols GRID_OFFSET / N).
            // `idx_lo_c` / `idx_hi_c` stay RELATIVE to `coarse_base` (they double
            // as the per-MT-buffer energy index); the base is added only when
            // reading the grid. Single-material => base 0, full length, so this
            // is bit-identical to the old single shared coarse grid.
            let coarse_meta_off = mat_idx * COARSE_META_COLS as usize;
            let coarse_base =
                inputs.coarse_meta[coarse_meta_off + COL_COARSE_GRID_OFFSET as usize] as usize;
            let coarse_n = inputs.coarse_meta[coarse_meta_off + COL_COARSE_N as usize] as usize;
            let coarse_slice = &inputs.coarse_log_energy_grid[coarse_base..coarse_base + coarse_n];
            let lo_c = coarse_slice.partition_point(|&x| x < log_e);
            let idx_hi_c = lo_c.clamp(1, coarse_n - 1);
            let idx_lo_c = idx_hi_c - 1;
            let e_lo_c = coarse_slice[idx_lo_c].exp();
            let e_hi_c = coarse_slice[idx_hi_c].exp();
            let denom_c = e_hi_c - e_lo_c;
            let frac_c = if denom_c > 0.0 {
                (energy - e_lo_c) / denom_c
            } else {
                0.0
            };
            // Aggregate macro XS ride the per-material FINE grid (issue #212):
            // row base `fine_off`, indices `idx_lo_f` / `idx_hi_f`, factor
            // `frac_f`. Single-material => `fine_off == mat_idx * n_grid`, so
            // byte-identical.
            let xs_e_lo = inputs.xs_elastic_per_material[fine_off + idx_lo_f];
            let xs_e_hi = inputs.xs_elastic_per_material[fine_off + idx_hi_f];
            let sigma_e = xs_e_lo + (xs_e_hi - xs_e_lo) * frac_f;
            let xs_a_lo = inputs.xs_absorption_per_material[fine_off + idx_lo_f];
            let xs_a_hi = inputs.xs_absorption_per_material[fine_off + idx_hi_f];
            let sigma_a = xs_a_lo + (xs_a_hi - xs_a_lo) * frac_f;
            let xs_i_lo = inputs.xs_inelastic_per_material[fine_off + idx_lo_f];
            let xs_i_hi = inputs.xs_inelastic_per_material[fine_off + idx_hi_f];
            let sigma_i = xs_i_lo + (xs_i_hi - xs_i_lo) * frac_f;
            let xs_f_lo = inputs.xs_fission_per_material[fine_off + idx_lo_f];
            let xs_f_hi = inputs.xs_fission_per_material[fine_off + idx_hi_f];
            let sigma_f = xs_f_lo + (xs_f_hi - xs_f_lo) * frac_f;
            let nu_lo = inputs.nu_bar_per_material[fine_off + idx_lo_f];
            let nu_hi = inputs.nu_bar_per_material[fine_off + idx_hi_f];
            let nu_bar = nu_lo + (nu_hi - nu_lo) * frac_f;
            // Delayed fraction on the same fine grid (issue #364).
            let beta_lo = inputs.beta_delayed_per_material[fine_off + idx_lo_f];
            let beta_hi = inputs.beta_delayed_per_material[fine_off + idx_hi_f];
            let beta_delayed = beta_lo + (beta_hi - beta_lo) * frac_f;

            // Per-(material, nuclide) URR probability-table perturbation
            // (issues #210, #342), twin of the kernel's block. The band base is
            // drawn once per ENERGY and held across steps, so the draw fires
            // only when this walk has no band or holds one for another energy;
            // a void or non-URR step leaves a held band intact.
            let urr_tables = urr_perturb::UrrTables {
                meta: inputs.urr_meta,
                ae_offset: inputs.urr_ae_offset,
                cdf_offset: inputs.urr_cdf_offset,
                energy_grid: inputs.urr_energy_grid,
                cdf: inputs.urr_cdf,
                xs: inputs.urr_xs,
                atom_density: inputs.urr_atom_density,
            };
            let urr_nuc_off = inputs.mat_nuclide_meta[mat_idx * 2] as usize;
            let urr_nuc_count = inputs.mat_nuclide_meta[mat_idx * 2 + 1] as usize;
            let mut urr_fired = false;
            let mut urr_macro_capture = 0.0;
            let (mut sigma_e, mut sigma_a, mut sigma_i, mut sigma_f) =
                (sigma_e, sigma_a, sigma_i, sigma_f);
            if urr_tables.any_in_range(urr_nuc_off, urr_nuc_count, energy) {
                let r_base = match urr_held {
                    Some((held_e, base)) if held_e == energy => base,
                    _ => {
                        let s = state;
                        let r = pcg_out(s);
                        state = s.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                        let base = (r as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                        urr_held = Some((energy, base));
                        base
                    }
                };
                let p = urr_perturb::perturb(
                    &urr_tables,
                    urr_perturb::Partials {
                        elastic: sigma_e,
                        absorption: sigma_a,
                        inelastic: sigma_i,
                        fission: sigma_f,
                    },
                    inputs.nuc_partial_xs,
                    |within| {
                        (
                            (fine_nuc_base + within * fine_n + idx_lo_f) * 4,
                            (fine_nuc_base + within * fine_n + idx_hi_f) * 4,
                        )
                    },
                    urr_nuc_off,
                    urr_nuc_count,
                    energy,
                    r_base,
                    frac_f,
                );
                sigma_e = p.partials.elastic;
                sigma_a = p.partials.absorption;
                sigma_i = p.partials.inelastic;
                sigma_f = p.partials.fission;
                urr_fired = p.fired;
                urr_macro_capture = p.macro_capture;
            }
            let urr_score = super::UrrScore {
                fired: urr_fired,
                macro_capture: urr_macro_capture,
                sigma_elastic: sigma_e,
            };
            let sigma_t = sigma_e + sigma_a + sigma_i + sigma_f;

            // 3. Free-flight sample.
            let s_xi1 = state;
            let r_xi1 = pcg_out(s_xi1);
            state = s_xi1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let xi1 = (r_xi1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let d_xs = -xi1.ln() / sigma_t;

            // 4. Boundary distance: scan surfaces. Stride-10 params
            // (wide enough for the general quadric's ten coefficients;
            // narrower kinds zero-pad the tail). All eight SurfaceKind
            // discriminants (stype 0..=7) are dispatched below.
            let mut d_boundary = 1e30_f64;
            let mut winner_surface = 0usize;
            for (s, &stype) in inputs.surface_types.iter().enumerate().take(n_surfaces) {
                let sb = s * 10;
                let p0 = inputs.surface_params[sb];
                let p1 = inputs.surface_params[sb + 1];
                let p2 = inputs.surface_params[sb + 2];
                let p3 = inputs.surface_params[sb + 3];
                let p4 = inputs.surface_params[sb + 4];
                let p5 = inputs.surface_params[sb + 5];
                let p6 = inputs.surface_params[sb + 6];
                let p7 = inputs.surface_params[sb + 7];
                let p8 = inputs.surface_params[sb + 8];
                let p9 = inputs.surface_params[sb + 9];
                let mut hit = 1e30_f64;
                if stype == 0 {
                    let qx = px - p0;
                    let qy = py - p1;
                    let qz = pz - p2;
                    let b = 2.0 * (qx * dx + qy * dy + qz * dz);
                    let cc = qx * qx + qy * qy + qz * qz - p3 * p3;
                    let disc = b * b - 4.0 * cc;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let t1 = (-b - sqrt_disc) * 0.5;
                        let t2 = (-b + sqrt_disc) * 0.5;
                        if t2 > 1e-12 {
                            hit = t2;
                        }
                        if t1 > 1e-12 {
                            hit = t1;
                        }
                    }
                }
                if stype == 1 {
                    let n_dot_d = p0 * dx + p1 * dy + p2 * dz;
                    if n_dot_d != 0.0 {
                        let n_dot_p = p0 * px + p1 * py + p2 * pz;
                        let t = (p3 - n_dot_p) / n_dot_d;
                        if t > 1e-12 {
                            hit = t;
                        }
                    }
                }
                if stype == 2 {
                    // cylinder: p0..p2 origin, p3 radius, p4..p6 axis
                    let qx = px - p0;
                    let qy = py - p1;
                    let qz = pz - p2;
                    let q_dot_a = qx * p4 + qy * p5 + qz * p6;
                    let mx = qx - q_dot_a * p4;
                    let my = qy - q_dot_a * p5;
                    let mz = qz - q_dot_a * p6;
                    let d_dot_a = dx * p4 + dy * p5 + dz * p6;
                    let dxp = dx - d_dot_a * p4;
                    let dyp = dy - d_dot_a * p5;
                    let dzp = dz - d_dot_a * p6;
                    let a_q = dxp * dxp + dyp * dyp + dzp * dzp;
                    let b_q = 2.0 * (dxp * mx + dyp * my + dzp * mz);
                    let c_q = mx * mx + my * my + mz * mz - p3 * p3;
                    if a_q > 0.0 {
                        let disc = b_q * b_q - 4.0 * a_q * c_q;
                        if disc >= 0.0 {
                            let sqrt_disc = disc.sqrt();
                            let inv_2a = 0.5 / a_q;
                            let t1 = (-b_q - sqrt_disc) * inv_2a;
                            let t2 = (-b_q + sqrt_disc) * inv_2a;
                            if t2 > 1e-12 {
                                hit = t2;
                            }
                            if t1 > 1e-12 {
                                hit = t1;
                            }
                        }
                    }
                }
                if stype == 3 {
                    // ZTorus: p0..p2 centre, p3 = a, p4 = b, p5 = c.
                    // Axial coord is z; transverse pair is (x, y).
                    hit = torus_smallest_positive_cpu(
                        px - p0,
                        py - p1,
                        pz - p2,
                        dx,
                        dy,
                        dz,
                        p3,
                        p4,
                        p5,
                    );
                }
                if stype == 4 {
                    // XTorus: axial coord is x; transverse pair is (y, z).
                    hit = torus_smallest_positive_cpu(
                        py - p1,
                        pz - p2,
                        px - p0,
                        dy,
                        dz,
                        dx,
                        p3,
                        p4,
                        p5,
                    );
                }
                if stype == 5 {
                    // YTorus: axial coord is y; transverse pair is (x, z).
                    hit = torus_smallest_positive_cpu(
                        px - p0,
                        pz - p2,
                        py - p1,
                        dx,
                        dz,
                        dy,
                        p3,
                        p4,
                        p5,
                    );
                }
                if stype == 6 {
                    // Quadric: p0..p9 = a, b, c, d, e, f, g, h, j, k.
                    hit = quadric_smallest_positive_cpu(
                        px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                    );
                }
                if stype == 7 {
                    // Cone: p0..p2 apex, p3..p5 unit axis, p6 = tan²θ.
                    hit = cone_smallest_positive_cpu(
                        px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6,
                    );
                }
                if hit < d_boundary {
                    d_boundary = hit;
                    winner_surface = s;
                }
            }

            // 5. Min.
            let collide_first = d_xs < d_boundary;
            let d = if collide_first { d_xs } else { d_boundary };

            // 6. Multi-tally accumulation. Mirror of the GPU
            // kernel: walk every tally, look up the cell-bin
            // (skip if not in this tally's CellFilter), find
            // the energy bin via binary search on this tally's
            // log_edges slice, multiply track length by the
            // score's xs factor and `weight`, and accumulate
            // into the slot.
            accumulate_tallies(
                inputs.tallies,
                cell as usize,
                n_cells,
                log_e,
                energy,
                d,
                weight,
                sigma_t,
                sigma_a,
                sigma_f,
                mat_idx,
                n_grid,
                idx_lo,
                idx_hi,
                frac,
                inputs.xs_score_per_mt_slice,
                inputs.n_score_mts,
                urr_score,
                // Mesh-tally segment: start (pre-move), end, unit direction.
                [px, py, pz],
                [px + dx * d, py + dy * d, pz + dz * d],
                [dx, dy, dz],
                tally_acc,
            );

            // 7. Move.
            px += dx * d;
            py += dy * d;
            pz += dz * d;

            // 8. Collide or cross.
            if collide_first {
                // Issue-#40 trace: energy entering this collision and the
                // reaction class chosen below (defaults to absorption; the
                // scatter / fission branches overwrite it). Only consumed by
                // the `trace.push` at the end of the collision block.
                let energy_in_rec = energy;
                let mut reaction_rec: i32 = 102;
                // Collision-estimator tallies: score weight × score_xs / Σ_t
                // once per real collision, at the collision point. Gated to
                // `is_collision[t] == 1` tallies; track-length tallies are
                // scored per step above.
                accumulate_collision_tallies(
                    inputs.tallies,
                    cell as usize,
                    n_cells,
                    log_e,
                    energy,
                    weight,
                    sigma_t,
                    sigma_a,
                    sigma_f,
                    mat_idx,
                    n_grid,
                    idx_lo,
                    idx_hi,
                    frac,
                    inputs.xs_score_per_mt_slice,
                    inputs.n_score_mts,
                    urr_score,
                    // Collision point (position already advanced by `d` above).
                    [px, py, pz],
                    tally_acc,
                );

                // Per-collision nuclide selection (issue #74). Exact twin of the
                // kernel block: in a multi-nuclide material, pick the struck
                // nuclide proportional to its macroscopic total at the collision
                // energy and override `target_mass` with that nuclide's AWR.
                // Gated on `count > 1` so single-nuclide materials draw NO random
                // and stay byte-identical to the pre-#74 stream.
                let nuc_off = inputs.mat_nuclide_meta[mat_idx * 2] as usize;
                let nuc_count = inputs.mat_nuclide_meta[mat_idx * 2 + 1] as usize;
                // Global slab of the struck nuclide (see kernel twin): defaults
                // to the material's first nuclide; the selection below picks
                // another when `count > 1`. Single-nuclide => `slab == nuc_off`,
                // the old `mat_idx` row, so behaviour is byte-identical.
                let mut slab = nuc_off;
                // Reaction-split partials (#74 Stage 2b): default to the
                // material aggregate (exact + byte-identical single-nuclide), a
                // multi-nuclide collision overrides them with the struck
                // nuclide's own partials. Mirrors the kernel exactly.
                let mut sigma_e_rx = sigma_e;
                let mut sigma_a_rx = sigma_a;
                let mut sigma_i_rx = sigma_i;
                let mut sigma_f_rx = sigma_f;
                if nuc_count > 1 {
                    // Per-material FINE grid (issue #212): slab `nuc_off + j`'s
                    // row base is `fine_nuc_base + j * fine_n`; interpolate in the
                    // FINE bracket. Single-material => `fine_nuc_base == 0`,
                    // `fine_n == n_grid`, byte-identical.
                    // A nuclide's share of the collision density is
                    // proportional to the cross section that actually governed
                    // the flight, so an in-range URR nuclide is weighted by its
                    // PERTURBED total, not the table average (issue #347).
                    // `urr_weight` reuses the band this walk holds, so flight,
                    // selection and split all ride one sample.
                    let urr_weight = |j: usize| -> f64 {
                        let row = fine_nuc_base + j * fine_n;
                        let smooth_lo = inputs.nuc_macro_total[row + idx_lo_f];
                        let smooth_hi = inputs.nuc_macro_total[row + idx_hi_f];
                        let smooth_total = smooth_lo + (smooth_hi - smooth_lo) * frac_f;
                        // Only a band held at THIS energy applies. Belt and
                        // braces: a band held from an earlier energy survives
                        // non-URR steps by design (#342), but it cannot corrupt
                        // a weight anyway, because `urr_held` is only set when
                        // `any_in_range` was true at that energy and
                        // `perturb_slab` re-checks the same range per slab -- so
                        // a stale band reaches only slabs that return `None`.
                        // Kept because it makes the invariant local, and the two
                        // range tests living in different functions is exactly
                        // the kind of coupling that rots.
                        let Some(r_base) =
                            urr_held.and_then(|(e, b)| if e == energy { Some(b) } else { None })
                        else {
                            return smooth_total;
                        };
                        let p_lo = (fine_nuc_base + j * fine_n + idx_lo_f) * 4;
                        let p_hi = (fine_nuc_base + j * fine_n + idx_hi_f) * 4;
                        let base =
                            urr_perturb::slab_baseline(inputs.nuc_partial_xs, p_lo, p_hi, frac_f);
                        match urr_perturb::perturb_slab(
                            &urr_tables,
                            nuc_off + j,
                            energy,
                            r_base,
                            base,
                        ) {
                            Some(p) => p.total(),
                            None => smooth_total,
                        }
                    };
                    let mut sigma_t_nuc = 0.0_f64;
                    for j in 0..nuc_count {
                        sigma_t_nuc += urr_weight(j);
                    }
                    let s_nuc = state;
                    let r_nuc = pcg_out(s_nuc);
                    state = s_nuc.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                    let xi_n = (r_nuc as f64 + 1.0) * (1.0 / 4_294_967_297.0) * sigma_t_nuc;
                    let mut accum = 0.0_f64;
                    let mut chosen = 0usize;
                    for k in 0..nuc_count {
                        accum += urr_weight(k);
                        if xi_n >= accum {
                            chosen += 1;
                        }
                    }
                    if chosen >= nuc_count {
                        chosen = nuc_count - 1;
                    }
                    slab = nuc_off + chosen;
                    target_mass = inputs.nuc_awr[slab];

                    // Selected nuclide's density-weighted partials, per-material
                    // FINE grid tight CSR `[sum_m nuc_count_m * fine_n_m x 4]`
                    // (cols e/a/i/f, issue #212): slab element base is
                    // `fine_nuc_base + (slab - nuc_off) * fine_n`, times 4.
                    // Interpolated in the FINE bracket. Twin of the kernel block.
                    let p_lo = (fine_nuc_base + (slab - nuc_off) * fine_n + idx_lo_f) * 4;
                    let p_hi = (fine_nuc_base + (slab - nuc_off) * fine_n + idx_hi_f) * 4;
                    let base =
                        urr_perturb::slab_baseline(inputs.nuc_partial_xs, p_lo, p_hi, frac_f);
                    // Split the struck nuclide on the SAME band its selection
                    // and the flight used (issue #347); smooth partials here
                    // would mis-weight capture against scatter, since the URR
                    // capture fraction moves strongly with the band.
                    let split = urr_held
                        .and_then(|(e, r_base)| if e == energy { Some(r_base) } else { None })
                        .and_then(|r_base| {
                            urr_perturb::perturb_slab(&urr_tables, slab, energy, r_base, base)
                        })
                        .unwrap_or(base);
                    sigma_e_rx = split.elastic;
                    sigma_a_rx = split.absorption;
                    sigma_i_rx = split.inelastic;
                    sigma_f_rx = split.fission;
                }

                let s_xi2 = state;
                let r_xi2 = pcg_out(s_xi2);
                state = s_xi2.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                let xi2 = (r_xi2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                // Survival biasing (implicit capture): exact twin of the
                // kernel branch. The reaction split runs against the SELECTED
                // nuclide's partials; single-nuclide => `*_rx == sigma_*`,
                // `sigma_t_rx == sigma_t`, byte-identical.
                let survival_on = inputs.survival_enabled;
                let sigma_t_rx = sigma_e_rx + sigma_a_rx + sigma_i_rx + sigma_f_rx;
                let sigma_sf = sigma_e_rx + sigma_i_rx + sigma_f_rx;
                let mut sel_denom = sigma_t_rx;
                if survival_on && sigma_sf > 0.0 {
                    sel_denom = sigma_sf;
                    weight *= sigma_sf / sigma_t_rx;
                }
                let p_elastic = sigma_e_rx / sel_denom;
                let p_scatter = (sigma_e_rx + sigma_i_rx) / sel_denom;
                let p_fission_or_scatter = sigma_sf / sel_denom;

                if xi2 >= p_fission_or_scatter {
                    alive = 0;
                } else {
                    let s_xi3 = state;
                    let r_xi3 = pcg_out(s_xi3);
                    state = s_xi3.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                    let xi3 = (r_xi3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

                    let mut mu_lab = 0.0_f64;
                    let mut skip_lab_rotation = false;
                    if xi2 < p_elastic {
                        reaction_rec = 2;
                        // Elastic: tabulated CM-frame mu sample
                        // (or isotropic fallback when no data).
                        // Index by the selected nuclide's global `slab`
                        // (#74 Stage 2a): per-(material, nuclide) elastic
                        // angular table. Single-nuclide => `slab == nuc_off`,
                        // the old `mat_idx` row (byte-identical).
                        let n_ae = inputs.elastic_angle_n_energies[slab] as usize;
                        // Tight CSR layout (issue #104): the slab's incident-
                        // energy rows start at this global ae-row base; the
                        // (mu, cdf, pdf) points are read from the full arrays
                        // via the per-row global `mu_offset`. No MAX_* stride.
                        let eg_off = inputs.elastic_angle_ae_offset[slab] as usize;
                        let mu_cm = yamc_physics::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm(
                            energy,
                            xi3,
                            &inputs.elastic_angle_energy_grid[eg_off..eg_off + n_ae],
                            &inputs.elastic_angle_n_mu[eg_off..eg_off + n_ae],
                            &inputs.elastic_angle_interp[eg_off..eg_off + n_ae],
                            inputs.elastic_angle_mu,
                            inputs.elastic_angle_cdf,
                            inputs.elastic_angle_pdf,
                            &inputs.elastic_angle_mu_offset[eg_off..eg_off + n_ae],
                            &mut state,
                        );
                        let temp_k = inputs.temperature_k_per_material[mat_idx];
                        let (new_dx, new_dy, new_dz, new_e, did_run) =
                            yamc_physics::gpu::flat::free_gas_elastic::sample_free_gas_elastic(
                                energy,
                                dx,
                                dy,
                                dz,
                                target_mass,
                                temp_k,
                                inputs.free_gas_threshold,
                                mu_cm,
                                &mut state,
                            );
                        if did_run {
                            dx = new_dx;
                            dy = new_dy;
                            dz = new_dz;
                            energy = new_e;
                            skip_lab_rotation = true;
                        } else {
                            let one_plus_a = target_mass + 1.0;
                            let denom = one_plus_a * one_plus_a;
                            let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
                            energy = energy * numer / denom;
                            mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();
                        }
                    } else if xi2 < p_scatter {
                        // Inelastic: per-MT sampling proportional
                        // to xs at energy. Mirrors the GPU kernel
                        // exactly so cumulative-sum walks produce
                        // bit-identical MT selections under the
                        // same RNG stream.
                        let s_xi_mt = state;
                        let r_xi_mt = pcg_out(s_xi_mt);
                        state = s_xi_mt.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                        let xi_mt = (r_xi_mt as f64 + 1.0) * (1.0 / 4_294_967_297.0);

                        // Per-MT inelastic buffers are keyed by the struck
                        // nuclide's slab (#74 Stage 2b); single-nuclide =>
                        // `slab == mat_idx`'s old row (byte-identical).
                        let mat_mt_off = slab * MT_INELASTIC_COUNT;
                        // SPARSE per-MT inelastic storage (issue #212), bit-twin of
                        // the kernel. Each (slab, MT slot) has a `permt_meta` row
                        // [value_offset, i_start, n_stored]; the slot's xs at coarse
                        // index `k` is the sparse value when `i_start <= k < i_start
                        // + n_stored`, else 0.0. The per-slab permt row base is
                        // `coarse_meta` col 2 (the material's first-slab row base)
                        // plus the slab's offset within the material -- reducing to
                        // the global `slab * MT_INELASTIC_COUNT`, the same ordering
                        // as `q_inelastic_per_mt`.
                        let coarse_mt_base = inputs.coarse_meta
                            [coarse_meta_off + COL_COARSE_MT_BASE as usize]
                            as usize;
                        let permt_row_base = coarse_mt_base + (slab - nuc_off) * MT_INELASTIC_COUNT;
                        // Walk against the SELECTED nuclide's inelastic xs. Reading
                        // 0 below threshold at both bracket endpoints matches the
                        // dense zeros exactly.
                        let target_cum = xi_mt * sigma_i_rx;
                        let mut cumulative = 0.0_f64;
                        let mut selected_slot = 0usize;
                        let mut found = false;
                        for slot in 0..MT_INELASTIC_COUNT {
                            let meta_off = (permt_row_base + slot) * PERMT_META_COLS as usize;
                            let value_offset = inputs.permt_meta
                                [meta_off + COL_PERMT_VALUE_OFFSET as usize]
                                as usize;
                            let i_start =
                                inputs.permt_meta[meta_off + COL_PERMT_I_START as usize] as usize;
                            let n_stored =
                                inputs.permt_meta[meta_off + COL_PERMT_N_STORED as usize] as usize;
                            let i_end = i_start + n_stored;
                            let mt_xs_lo = if idx_lo_c >= i_start && idx_lo_c < i_end {
                                inputs.xs_inelastic_per_mt_sparse
                                    [value_offset + (idx_lo_c - i_start)]
                            } else {
                                0.0
                            };
                            let mt_xs_hi = if idx_hi_c >= i_start && idx_hi_c < i_end {
                                inputs.xs_inelastic_per_mt_sparse
                                    [value_offset + (idx_hi_c - i_start)]
                            } else {
                                0.0
                            };
                            let mt_sigma = mt_xs_lo + (mt_xs_hi - mt_xs_lo) * frac_c;
                            cumulative += mt_sigma;
                            if !found && cumulative >= target_cum {
                                selected_slot = slot;
                                found = true;
                            }
                        }
                        if !found {
                            // Fallback to slot 40 (MT 91, continuum
                            // inelastic, single-neutron-out) on
                            // numerical drift -- don't pick the
                            // multi-neutron-out slots (41–42) here.
                            selected_slot = 40;
                        }
                        // Issue-#40 trace: slot k maps to MT 51+k for the
                        // discrete levels (slot 0 = MT 51) and MT 91 for the
                        // continuum (slot 40), matching the production CPU's
                        // recorded inelastic MT.
                        reaction_rec = 51 + selected_slot as i32;

                        // Per-MT energy-dependent yield ν(E_in), from the SAME
                        // sparse `permt_meta` row as the selected slot's xs. Outside
                        // the stored range the yield is 1.0 (the dense default),
                        // keeping a threshold bracket bit-identical.
                        let sel_meta_off =
                            (permt_row_base + selected_slot) * PERMT_META_COLS as usize;
                        let sel_value_offset = inputs.permt_meta
                            [sel_meta_off + COL_PERMT_VALUE_OFFSET as usize]
                            as usize;
                        let sel_i_start =
                            inputs.permt_meta[sel_meta_off + COL_PERMT_I_START as usize] as usize;
                        let sel_i_end = sel_i_start
                            + inputs.permt_meta[sel_meta_off + COL_PERMT_N_STORED as usize]
                                as usize;
                        let yield_lo = if idx_lo_c >= sel_i_start && idx_lo_c < sel_i_end {
                            inputs.yield_per_mt_sparse[sel_value_offset + (idx_lo_c - sel_i_start)]
                        } else {
                            1.0
                        };
                        let yield_hi = if idx_hi_c >= sel_i_start && idx_hi_c < sel_i_end {
                            inputs.yield_per_mt_sparse[sel_value_offset + (idx_hi_c - sel_i_start)]
                        } else {
                            1.0
                        };
                        let slot_yield = yield_lo + (yield_hi - yield_lo) * frac_c;
                        // Analog (n,xn) multiplicity (issue #274), exact twin
                        // of the kernel: an integral yield >= 2 keeps the
                        // weight unchanged and samples `yield - 1` extra
                        // independent secondaries below; fractional yields
                        // (and yield == 1, a multiply by exactly 1.0) keep
                        // the legacy weight multiplication.
                        let mut n_out = 1u32;
                        let yield_round = (slot_yield + 0.5) as u32;
                        if yield_round >= 2 && (slot_yield - yield_round as f64).abs() < 1e-10 {
                            n_out = yield_round;
                        } else {
                            weight *= slot_yield;
                            if weight > FISSION_WEIGHT_CAP {
                                alive = 0;
                            }
                        }

                        let q = inputs.q_inelastic_per_mt[mat_mt_off + selected_slot];
                        let abs_q = q.abs();
                        let threshold = (target_mass + 1.0) / target_mass * abs_q;
                        let mass_ratio = (target_mass / (target_mass + 1.0)).powi(2);
                        let e_in_inel = energy;
                        let e_cm_closed = mass_ratio * (e_in_inel - threshold);

                        // Sample the outgoing kinematics n_out times, exact
                        // twin of the kernel's k_out loop: iteration 0 is the
                        // continuing walk, iterations 1.. are the analog
                        // (n,xn) secondaries (fresh isotropic-fallback
                        // uniform, one azimuth draw, queued for in-thread
                        // transport; kinematics failures are dropped).
                        for k_out in 0..n_out {
                            let mut xi3_k = xi3;
                            if k_out > 0 {
                                let (xi_f, st_f) = crate::common::pcg32::draw_uniform_cpu(state);
                                state = st_f;
                                xi3_k = xi_f;
                            }
                            let (mu_s, e_lab_or_cm, alive_after) = sample_inelastic_angle(
                                e_in_inel,
                                target_mass,
                                e_cm_closed,
                                xi3_k,
                                slab,
                                selected_slot,
                                &mut state,
                                inputs.angle_n_energies,
                                inputs.angle_ae_offset,
                                inputs.angle_energy_grid,
                                inputs.angle_n_mu,
                                inputs.angle_mu_offset,
                                inputs.angle_mu,
                                inputs.angle_cdf,
                                inputs.angle_pdf,
                                inputs.angle_interp,
                                inputs.eout_kind,
                                inputs.eout_n_energies,
                                inputs.eout_ae_offset,
                                inputs.eout_energy_grid,
                                inputs.eout_n_x,
                                inputs.eout_x_offset,
                                inputs.eout_x,
                                inputs.eout_cdf,
                                inputs.eout_histogram_interp,
                                inputs.eout_p,
                                inputs.eout_interp,
                                inputs.eout_n_discrete,
                                inputs.corr_n_energies,
                                inputs.corr_n_components,
                                inputs.corr_ae_offset,
                                inputs.corr_energy_grid,
                                inputs.corr_n_x,
                                inputs.corr_x_offset,
                                inputs.corr_x,
                                inputs.corr_cdf,
                                inputs.corr_p,
                                inputs.corr_interp,
                                inputs.corr_n_discrete,
                                inputs.corr_n_mu,
                                inputs.corr_mu_offset,
                                inputs.corr_mu,
                                inputs.corr_mu_cdf,
                                inputs.corr_mu_pdf,
                                inputs.corr_mu_interp,
                                inputs.scatter_in_cm_per_mt,
                                inputs.km_n_energies,
                                inputs.km_ae_offset,
                                inputs.km_energy_grid,
                                inputs.km_interp,
                                inputs.km_n_discrete,
                                inputs.km_n_x,
                                inputs.km_x_offset,
                                inputs.km_x,
                                inputs.km_p,
                                inputs.km_c,
                                inputs.km_r,
                                inputs.km_a,
                                inputs.evap_n_energies,
                                inputs.evap_n_components,
                                inputs.evap_ae_offset,
                                inputs.evap_theta_offset,
                                inputs.evap_energy_grid,
                                inputs.evap_theta,
                                inputs.evap_u,
                                inputs.nbps_n_bodies,
                                inputs.nbps_total_mass,
                                inputs.maxwell_n_energies,
                                inputs.maxwell_ae_offset,
                                inputs.maxwell_energy_grid,
                                inputs.maxwell_theta,
                                inputs.maxwell_u,
                                inputs.watt_n_energies,
                                inputs.watt_ae_offset,
                                inputs.watt_energy_grid,
                                inputs.watt_a,
                                inputs.watt_b,
                                inputs.watt_u,
                                inputs.q_inelastic_per_mt[mat_mt_off + selected_slot],
                            );
                            if k_out == 0 {
                                if !alive_after {
                                    alive = 0;
                                } else {
                                    energy = e_lab_or_cm;
                                }
                                mu_lab = mu_s;
                            } else if alive_after {
                                // Lab azimuth for the extra secondary: one
                                // draw, TAU * xi, libm cos/sin (same ulp
                                // relationship to the kernel polyfill as the
                                // walk's shared rotation), applied around the
                                // INCIDENT direction.
                                let s_phx = state;
                                let r_phx = pcg_out(s_phx);
                                state = s_phx.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                                let xi_phx = (r_phx as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                                let phi_x = std::f64::consts::TAU * xi_phx;
                                let cos_phi_x = phi_x.cos();
                                let sin_phi_x = phi_x.sin();
                                let sin_th_sq_x = 1.0 - mu_s * mu_s;
                                let sin_th_x = if sin_th_sq_x > 0.0 {
                                    sin_th_sq_x.sqrt()
                                } else {
                                    0.0
                                };
                                let one_minus_w_sq_x = 1.0 - dz * dz;
                                let mut xdx = sin_th_x * cos_phi_x;
                                let mut xdy = sin_th_x * sin_phi_x;
                                let mut xdz = mu_s;
                                if dz < 0.0 {
                                    xdy = -xdy;
                                    xdz = -mu_s;
                                }
                                if one_minus_w_sq_x > 1e-14 {
                                    let sin_phi_w_x = one_minus_w_sq_x.sqrt();
                                    xdx = mu_s * dx
                                        + sin_th_x * (dx * dz * cos_phi_x - dy * sin_phi_x)
                                            / sin_phi_w_x;
                                    xdy = mu_s * dy
                                        + sin_th_x * (dy * dz * cos_phi_x + dx * sin_phi_x)
                                            / sin_phi_w_x;
                                    xdz = mu_s * dz - sin_th_x * sin_phi_w_x * cos_phi_x;
                                }
                                if n_pend_nxn >= PEND_SLOTS {
                                    // Past this depth the kernel spills to the
                                    // device bank; count it so tests can see
                                    // how often that path is reached.
                                    n_spilled += 1;
                                } else if n_pend_nxn as u32 + 1 > max_pend_depth {
                                    max_pend_depth = n_pend_nxn as u32 + 1;
                                }
                                n_pend_nxn += 1;
                                pend.push(PendEntry {
                                    // The secondary's own collision stream,
                                    // fixed by its place in the emission tree
                                    // (issue #111).
                                    seed: secondary_seed(walk_seed, walk_secondaries),
                                    nxn: true,
                                    energy: e_lab_or_cm,
                                    px,
                                    py,
                                    pz,
                                    dx: xdx,
                                    dy: xdy,
                                    dz: xdz,
                                    weight,
                                });
                                walk_secondaries += 1;
                            }
                        }
                    } else {
                        reaction_rec = 18;
                        // Fission. The continuing walk gets ONE chi-sampled
                        // outgoing energy (via `fission_chi_cpu`, the exact
                        // mirror of the kernel `sample_fission_chi`) and an
                        // isotropic-in-lab μ (`1 - 2·xi3`). Two modes, selected
                        // by `fission_bank_enabled` (kernel
                        // `fission_bank_enabled[0]`):
                        //   OFF: legacy `weight *= nu_bar` + `FISSION_WEIGHT_CAP`
                        //     terminator (byte-identical to the pre-bank twin).
                        //   ON: true fission-chain branching (#78). Stochastically
                        //     round nu_bar -> N, keep progeny 0 as the current walk
                        //     (weight UNCHANGED), and emit the other N-1 progeny.
                        //     The kernel appends them to the DEVICE bank for the
                        //     host to drain in a second pass; the twin queues them
                        //     on its in-thread `pend` stack and transports them
                        //     inside this history, exactly as it already does for
                        //     the (n,xn) secondaries the kernel spills. Both give
                        //     the same physics, because each progeny's collision
                        //     stream is keyed on its place in the emission tree
                        //     (issue #322), not on where it is transported -- and
                        //     it makes the twin's per-history collision sequence
                        //     comparable to the production CPU's, whose
                        //     `ParticleBank` transports fission progeny inside the
                        //     history too.
                        let e_incident_fis = energy;
                        energy = fission_progeny_energy_cpu(
                            inputs,
                            mat_idx,
                            beta_delayed,
                            e_incident_fis,
                            &mut state,
                        );
                        mu_lab = 1.0 - 2.0 * xi3;

                        if !inputs.fission_bank_enabled {
                            weight *= nu_bar;
                            if weight > FISSION_WEIGHT_CAP {
                                alive = 0;
                            }
                        } else {
                            // Stochastically round nu_bar -> N with one uniform
                            // (CPU `sample_fission_neutrons` rounding). Weight is
                            // left UNCHANGED (analog branching).
                            let (xi_nr, st_nr) = crate::common::pcg32::draw_uniform_cpu(state);
                            state = st_nr;
                            let n_floor = nu_bar as u32; // floor (nu_bar >= 0)
                            let frac_nr = nu_bar - (n_floor as f64);
                            let n_prog = if xi_nr < frac_nr {
                                n_floor + 1
                            } else {
                                n_floor
                            };
                            // N == 0: the fission emits no neutrons, so the
                            // continuing walk dies (matches CPU
                            // `sample_fission_neutrons` returning an empty batch
                            // and the kernel's `n_prog == 0` kill).
                            if n_prog == 0 {
                                alive = 0;
                            }
                            // Progeny 1..N: each gets an independent chi energy
                            // and an independent isotropic-in-lab direction (a
                            // uniform μ then a `TAU * xi` azimuth, ONE draw, as
                            // the kernel and the production CPU both do), built
                            // about the INCIDENT direction. Capped exactly as the
                            // kernel's comptime-bounded loop is.
                            let cap_prog = 8u32; // FISSION_BANK_PROGENY_CAP
                            let mut prog = 1u32;
                            while prog < cap_prog {
                                if prog < n_prog {
                                    let e_bank = fission_progeny_energy_cpu(
                                        inputs,
                                        mat_idx,
                                        beta_delayed,
                                        e_incident_fis,
                                        &mut state,
                                    );
                                    let (xi_mu, st_mu) =
                                        crate::common::pcg32::draw_uniform_cpu(state);
                                    state = st_mu;
                                    let mu_b = 1.0 - 2.0 * xi_mu;
                                    let s_phb = state;
                                    let r_phb = pcg_out(s_phb);
                                    state = s_phb.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                                    let xi_phb = (r_phb as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                                    let phi_b = std::f64::consts::TAU * xi_phb;
                                    let cos_phi_b = phi_b.cos();
                                    let sin_phi_b = phi_b.sin();
                                    let sin_th_sq_b = 1.0 - mu_b * mu_b;
                                    let sin_th_b = if sin_th_sq_b > 0.0 {
                                        sin_th_sq_b.sqrt()
                                    } else {
                                        0.0
                                    };
                                    let one_minus_w_sq_b = 1.0 - dz * dz;
                                    let mut bdx = sin_th_b * cos_phi_b;
                                    let mut bdy = sin_th_b * sin_phi_b;
                                    let mut bdz = mu_b;
                                    if dz < 0.0 {
                                        bdy = -bdy;
                                        bdz = -mu_b;
                                    }
                                    if one_minus_w_sq_b > 1e-14 {
                                        let sin_phi_w_b = one_minus_w_sq_b.sqrt();
                                        bdx = mu_b * dx
                                            + sin_th_b * (dx * dz * cos_phi_b - dy * sin_phi_b)
                                                / sin_phi_w_b;
                                        bdy = mu_b * dy
                                            + sin_th_b * (dy * dz * cos_phi_b + dx * sin_phi_b)
                                                / sin_phi_w_b;
                                        bdz = mu_b * dz - sin_th_b * sin_phi_w_b * cos_phi_b;
                                    }
                                    // Queue it for this history. `nxn: false`
                                    // keeps it out of `n_spilled` /
                                    // `max_pend_depth`: those track the kernel's
                                    // REGISTER queue for (n,xn) secondaries, and
                                    // a fission progeny never enters it (the
                                    // kernel always sends these to the device
                                    // bank).
                                    pend.push(PendEntry {
                                        seed: secondary_seed(walk_seed, walk_secondaries),
                                        nxn: false,
                                        energy: e_bank,
                                        px,
                                        py,
                                        pz,
                                        dx: bdx,
                                        dy: bdy,
                                        dz: bdz,
                                        weight,
                                    });
                                    walk_secondaries += 1;
                                }
                                prog += 1;
                            }
                        }
                    }

                    // Lab azimuth: phi = TAU * xi (ONE PCG draw), matching the
                    // production CPU (`scatter.rs` -> `rotate_direction_fast`)
                    // so the issue-#40 matched stream stays in lockstep past the
                    // first collision (#136 / #111). std libm cos/sin like the
                    // production CPU (the kernel uses the cos_f64/sin_f64
                    // polyfill via `azimuth_cos_sin_phi`; twin and kernel agree
                    // within ulps, as ln/exp already do -- see
                    // `assert_gpu_cpu_equiv`). Drawn unconditionally (even when
                    // free-gas already wrote the direction) so both kernel paths
                    // consume the same stream; `xi` is computed exactly as
                    // `next_xi` so the value bit-matches the production CPU.
                    let s_phi = state;
                    let r_phi = pcg_out(s_phi);
                    state = s_phi.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                    let xi_phi = (r_phi as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                    let phi = std::f64::consts::TAU * xi_phi;
                    let cos_phi = phi.cos();
                    let sin_phi = phi.sin();

                    // 3D rotation, mirrored from the GPU kernel.
                    // Skip when free-gas elastic already wrote
                    // the new direction directly.
                    if !skip_lab_rotation {
                        let sin_theta_sq = 1.0 - mu_lab * mu_lab;
                        let sin_theta = if sin_theta_sq > 0.0 {
                            sin_theta_sq.sqrt()
                        } else {
                            0.0
                        };
                        let one_minus_w_sq = 1.0 - dz * dz;

                        let mut new_u = sin_theta * cos_phi;
                        let mut new_v = sin_theta * sin_phi;
                        let mut new_w = mu_lab;
                        if dz < 0.0 {
                            new_v = -new_v;
                            new_w = -mu_lab;
                        }
                        if one_minus_w_sq > 1e-14 {
                            let sin_phi_w = one_minus_w_sq.sqrt();
                            new_u = mu_lab * dx
                                + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / sin_phi_w;
                            new_v = mu_lab * dy
                                + sin_theta * (dy * dz * cos_phi + dx * sin_phi) / sin_phi_w;
                            new_w = mu_lab * dz - sin_theta * sin_phi_w * cos_phi;
                        }
                        dx = new_u;
                        dy = new_v;
                        dz = new_w;
                    }
                }

                // Weight-cutoff Russian roulette: exact twin of the kernel
                // block. After the collision is fully processed, a still-alive
                // particle below `weight_cutoff` is rouletted -- survive with
                // probability `weight / weight_survive` (continue at
                // `weight_survive`) or die. One draw, only below the cutoff,
                // so an above-cutoff history draws nothing extra and a
                // survival-OFF run never enters this block (byte-identical).
                if survival_on && alive == 1 && weight < inputs.weight_cutoff {
                    let (xi, new_state) = crate::common::pcg32::draw_uniform_cpu(state);
                    state = new_state;
                    weight = super::roulette::weight_cutoff_roulette_cpu(
                        weight,
                        inputs.weight_survive,
                        xi,
                    );
                    if weight == 0.0 {
                        alive = 0;
                    }
                }

                // Issue-#40 trace: record this collision (energy in/out and
                // reaction class) once it is fully processed. Off (None) on
                // the production / kernel paths, so the stream is unchanged.
                if let Some(t) = trace.as_mut() {
                    t.push(CollisionRecord {
                        step,
                        energy_in: energy_in_rec,
                        energy_out: energy,
                        reaction: reaction_rec,
                    });
                }
            } else if inputs.surface_boundaries[winner_surface] == BOUNDARY_VACUUM {
                // Surface crossing into a vacuum boundary --
                // particle terminates at the surface.
                alive = 0;
            } else {
                // Transmission crossing: the step landed the particle
                // exactly ON the surface, where the point-in-region cell
                // test is in neither neighbouring cell. Nudge a hair past
                // the surface so the next cell-find lands strictly inside
                // the entered cell (mirrors the kernel + the CPU handoff
                // eps in `Cell::distance_to_surface`).
                px += dx * REGION_CROSS_EPS;
                py += dy * REGION_CROSS_EPS;
                pz += dz * REGION_CROSS_EPS;
            }
        }

        step += 1;
        n_steps = step;
    }

    ParticleOutcome {
        alive,
        n_steps,
        final_energy: energy,
        lost,
        n_spilled,
        max_pend_depth,
    }
}
