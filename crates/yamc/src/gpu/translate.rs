//! `Model` → `run_multi_cell_transport` flat-buffer inputs.
//!
//! See `crate::gpu` for the supported subset and the rejection rules.

use std::collections::HashMap;
use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use yamc_geo::{BoundaryType, HalfspaceType, RegionExpr, Surface, SurfaceKind};
use yamc_gpu::neutron::nuclide_select_inputs::{
    NuclideSelectInputs, NUC_PARTIAL_ABSORPTION, NUC_PARTIAL_COLS, NUC_PARTIAL_ELASTIC,
    NUC_PARTIAL_FISSION, NUC_PARTIAL_INELASTIC,
};
use yamc_gpu::neutron::xs::{
    extract_material_xs, extract_per_nuclide_elastic_angle, extract_per_nuclide_inelastic,
    extract_per_nuclide_macro_total_xs, union_energy_grid, GpuNuclideXs, PerNuclideElasticAngle,
    PerNuclideInelastic, PERMT_META_COLS, URR_META_COLS, URR_META_N_CDF, URR_META_N_ENERGIES,
};
use yamc_rng::history_seed;
use yamc_source::source::SourceSelector;

// Mirror of the kernel's `surface_types[i]` discriminants. The kernel
// re-exports these from `yamc_gpu::common::geometry::boundary_distance`, but
// that module is `#[cfg(not(target_os = "macos"))]`-gated on the
// stub-only macOS build. Define them inline so the translation
// compiles on macOS too -- values must stay in lock-step with the
// kernel's `boundary_distance.rs`.
const SURFACE_SPHERE: u32 = 0;
const SURFACE_PLANE: u32 = 1;
const SURFACE_CYLINDER: u32 = 2;
const SURFACE_ZTORUS: u32 = 3;
const SURFACE_XTORUS: u32 = 4;
const SURFACE_YTORUS: u32 = 5;
const SURFACE_QUADRIC: u32 = 6;
const SURFACE_CONE: u32 = 7;

// Region-program op encoding, in lock-step with the kernel's
// `yamc_gpu::common::geometry::region_eval` constants. Each op word packs
// `(opcode << REGION_OP_SHIFT) | surface_idx`. Defined inline (not imported)
// so the translation compiles on the macOS stub build, where `region_eval`
// is gated out, exactly like the `SURFACE_*` discriminants above.
const REGION_OP_ABOVE: u32 = 0;
const REGION_OP_BELOW: u32 = 1;
const REGION_OP_AND: u32 = 2;
const REGION_OP_OR: u32 = 3;
const REGION_OP_NOT: u32 = 4;
const REGION_OP_SHIFT: u32 = 28;

use super::error::GpuTranslateError;
use crate::geometry::backend::GeometryKind;
use crate::geometry::cell::Cell;
use crate::model::Model;
use yamc_materials::material::Material;

/// Stride of each surface's params slot in `surface_params`. Wide
/// enough for the general quadric's ten coefficients; the narrower
/// surfaces (sphere/plane 4, cylinder/cone 7, tori 6) zero-pad their
/// tail so the GPU indexing is a uniform `s * 10 + k`.
const SURFACE_PARAM_STRIDE: usize = 10;

/// Flat-buffer inputs ready to hand to
/// `yamc_gpu::neutron::transport::run_multi_cell_transport`. The dispatch layer
/// derives `tally_log_e_min` / `tally_log_e_max` / `n_energy_bins` /
/// `max_steps` separately from the model's tally configuration.
#[derive(Debug, Clone)]
pub struct GpuTransportInputs {
    /// Per-particle PCG seed.
    pub seeds: Vec<u32>,
    /// Per-particle initial energy (eV).
    pub energies: Vec<f64>,
    /// Per-particle initial position, stride 3.
    pub positions: Vec<f64>,
    /// Per-particle initial direction, stride 3 (unit vectors).
    pub directions: Vec<f64>,
    /// Cell AABBs, stride 6: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub cell_aabbs: Vec<f64>,
    /// Material index per cell.
    pub cell_to_material: Vec<u32>,
    /// Surface kind discriminant per surface (`SURFACE_*` constants).
    pub surface_types: Vec<u32>,
    /// Surface params per surface, stride `SURFACE_PARAM_STRIDE`.
    pub surface_params: Vec<f64>,
    /// Surface boundary discriminant per surface (matches the kernel's
    /// `BOUNDARY_TRANSMISSION = 0` / `BOUNDARY_VACUUM = 1` and yamc-geo's
    /// `BoundaryType` enum order).
    pub surface_boundaries: Vec<u32>,
    /// Flat per-cell CSG region program for point-in-region cell finding.
    /// The first `n_cells + 1` words are prefix offsets into the op stream
    /// that follows; each op word packs `(opcode << 28) | surface_idx`. See
    /// `yamc_gpu::common::geometry::region_eval` for the full layout. The
    /// kernel evaluates this so nested / overlapping cells resolve to the
    /// same cell the CPU picks, not just the first matching AABB.
    pub region_program: Vec<u32>,
    /// `ln(energy)` GLOBAL fine grid -- the union of every material's union grid
    /// (issue #212). Retained ONLY for the grid-shared score / photon / decay
    /// lookups (`xs_score_per_mt`, `photon_prod`, `decay_*`), which stay on this
    /// one grid. The resonance-critical aggregate macro XS + nuc buffers moved to
    /// the per-material `fine_log_energy_grid` below.
    pub log_energy_grid: Vec<f64>,
    /// `ln(energy)` FINE grids, one PER MATERIAL, concatenated tight (issue
    /// #212). Each material owns its fine grid (its union grid); `fine_meta`
    /// records each material's base + length here. Backs the per-material
    /// aggregate macro XS (`xs_elastic_per_material` etc.) and the
    /// per-(material, nuclide) `nuc_macro_total` / `nuc_partial_xs`. Linear
    /// interpolation along a material's own grid is invariant to the other
    /// materials' points, so this holds accuracy while bounding those buffers to
    /// one material's grid (RAM). A single-material problem has one grid at base
    /// 0 (its union == the global union), byte-identical to the old shared grid.
    pub fine_log_energy_grid: Vec<f64>,
    /// Packed `[n_materials × FINE_META_COLS]` per-material fine-grid descriptor
    /// (issue #212). Columns (see `COL_FINE_*`): base into `fine_log_energy_grid`
    /// (== per-material aggregate-XS row base), fine length, and first-slab
    /// element base into `nuc_macro_total`. Includes the synthetic void slot.
    pub fine_meta: Vec<u32>,
    /// `ln(energy)` COARSE grids, one PER MATERIAL, concatenated tight (issue
    /// #212). Each material owns its coarse grid (its finest single per-nuclide
    /// grid); `coarse_meta` records each material's base + length here. Backs the
    /// SPARSE per-MT inelastic buffers (`xs_inelastic_per_mt_sparse`,
    /// `yield_per_mt_sparse`, keyed by `permt_meta`) only.
    /// A single-material problem has one grid at base 0, byte-identical to the
    /// old single shared coarse grid.
    pub coarse_log_energy_grid: Vec<f64>,
    /// Packed `[n_materials × COARSE_META_COLS]` per-material coarse-grid
    /// descriptor (issue #212). Columns (see `COL_COARSE_*`): base into
    /// `coarse_log_energy_grid`, coarse length, and per-MT-buffer first-slab
    /// base. Includes the synthetic void material slot when present.
    pub coarse_meta: Vec<u32>,
    /// Macroscopic elastic XS per material, flat `[n_materials × n_grid]` on the
    /// FINE grid.
    pub xs_elastic_per_material: Vec<f64>,
    /// Macroscopic absorption (capture) XS per material, same shape.
    pub xs_absorption_per_material: Vec<f64>,
    /// Macroscopic inelastic XS per material, same shape -- aggregated
    /// across MT 51..=91 (discrete levels + continuum). Used to pick
    /// the inelastic *branch* in collision sampling.
    pub xs_inelastic_per_material: Vec<f64>,
    /// SPARSE per-MT inelastic xs values (issue #212). Concatenated tight in
    /// (slab, MT slot) order: each slot contributes only its nonzero
    /// (above-threshold) coarse-grid range, located by `permt_meta`. The dense
    /// `[n_slab × MT_INELASTIC_COUNT × coarse_n]` buffer was 90-99.9% zeros and
    /// overran the GPU's ~4 GB `maxStorageBufferRange` on a multi-material union
    /// grid; this shrinks it 10-1700x, f64-exact.
    pub xs_inelastic_per_mt_sparse: Vec<f64>,
    /// Packed `[n_slab × MT_INELASTIC_COUNT × PERMT_META_COLS]` per-(slab, MT
    /// slot) descriptor (issue #212): `[value_offset, i_start, n_stored]` (see
    /// `COL_PERMT_*`). `value_offset` is the base into
    /// `xs_inelastic_per_mt_sparse` / `yield_per_mt_sparse` (both share it),
    /// `i_start` the first nonzero coarse-grid index (relative to the material's
    /// coarse grid), `n_stored` the stored-point count (`0` = absent slot).
    /// Indexed `permt_meta[(slab * MT_INELASTIC_COUNT + slot) * PERMT_META_COLS +
    /// col]`, the same (slab, slot) ordering as `q_inelastic_per_mt`.
    pub permt_meta: Vec<u32>,
    /// Total fission macroscopic xs per material on the master grid,
    /// flat `[n_materials × n_grid]`. Drives the fission branch in the
    /// kernel's collision sampling.
    pub xs_fission_per_material: Vec<f64>,
    /// ν̄(E) per material on the master grid, flat `[n_materials ×
    /// n_grid]`. Used as the surviving-neutron weight multiplier when
    /// the kernel samples a fission collision.
    pub nu_bar_per_material: Vec<f64>,
    /// Delayed-neutron fraction `beta(E) = nu_d(E) / nu_t(E)`, same tight
    /// per-material fine-grid CSR shape as `nu_bar_per_material` (issue #364).
    pub beta_delayed_per_material: Vec<f64>,
    /// Per-material Watt-spectrum `a` parameter (eV) for the fission
    /// χ-spectrum sampler. Length `n_materials`.
    pub fission_a_per_material: Vec<f64>,
    /// Per-material Watt-spectrum `b` parameter (1/eV).
    pub fission_b_per_material: Vec<f64>,
    /// Per-material fission outgoing-energy sampler discriminant.
    /// `EOUT_KIND_CONTINUOUS_TABULAR` (1) selects the new tabulated
    /// sampler; `EOUT_KIND_WATT` (7) keeps the Watt-rejection fallback.
    /// Length `n_materials`.
    pub fission_eout_kind_per_material: Vec<u32>,
    /// Per-chi-row count of populated E_in points in
    /// `fission_eout_energy_grid`. Length `2 * n_materials`.
    pub fission_eout_n_energies_per_material: Vec<u32>,
    /// CSR base (issue #104): global ae-row where each chi row's
    /// incident-energy rows start in the tight
    /// `fission_eout_energy_grid_per_material` / `fission_eout_n_x_per_material`.
    /// Length `2 * n_materials`; the row's count is
    /// `fission_eout_n_energies_per_material[chi_row]`.
    pub fission_eout_ae_offset: Vec<u32>,
    /// Per-material fission incident-energy grid, tight CSR (issue #104):
    /// rows concatenated across materials, length `sum(n_energies)`.
    pub fission_eout_energy_grid_per_material: Vec<f64>,
    /// Per-material E_out-point count per E_in slice, tight CSR
    /// (issue #104): length `sum(n_energies)` (one entry per ae-row).
    pub fission_eout_n_x_per_material: Vec<u32>,
    /// CSR base (issue #104): index into the tight
    /// `fission_eout_x_per_material` / `fission_eout_cdf_per_material` where
    /// each ae-row's `(x, cdf)` points start. Length = total ae-rows
    /// (`sum(n_energies)`); row length is `fission_eout_n_x_per_material[ae]`.
    pub fission_eout_x_offset: Vec<u32>,
    /// Per-material tabulated E_out values, tight CSR (issue #104): rows
    /// concatenated back-to-back, length `sum(fission_eout_n_x)`.
    pub fission_eout_x_per_material: Vec<f64>,
    /// Per-material CDF for `fission_eout_x_per_material`, same shape.
    pub fission_eout_cdf_per_material: Vec<f64>,
    /// Per-material PDF for `fission_eout_x_per_material`, same shape,
    /// normalized like `fission_eout_cdf_per_material`. Zero-filled for
    /// rows without a usable PDF (sampler falls back to linear-in-c).
    pub fission_eout_p_per_material: Vec<f64>,
    /// Per-ae-row interpolation discriminant for the fission E_out
    /// inversion, tight CSR (issue #104): length `sum(n_energies)` (one
    /// entry per ae-row, like `fission_eout_n_x_per_material`). `0` =
    /// histogram, `1` = lin-lin.
    pub fission_eout_interp_per_material: Vec<u32>,
    /// Number-density-weighted average atomic-weight ratio per material.
    pub target_mass_per_material: Vec<f64>,
    /// Per-MT Q-values, flat layout `[n_materials × MT_INELASTIC_COUNT]`.
    /// Slot `k` is MT `MT_INELASTIC_FIRST + k`'s Q-value, used by the
    /// kernel's level-inelastic kinematics formula.
    pub q_inelastic_per_mt: Vec<f64>,
    /// SPARSE per-MT outgoing-neutron yield ν(E) values (issue #212).
    /// Concatenated tight in (slab, MT slot) order, parallel to
    /// `xs_inelastic_per_mt_sparse` and located by the SAME `permt_meta`
    /// row (identical `value_offset` / `i_start` / `n_stored`). Stored over each
    /// slot's nonzero-XS range; outside that range the kernel uses the dense
    /// default `1.0` (matching the old dense `yield_per_mt`, which was `1.0`
    /// wherever XS was zero), so the yield interpolation stays bit-identical even
    /// at a threshold bracket where one endpoint is below threshold.
    pub yield_per_mt_sparse: Vec<f64>,
    /// Per-MT tabulated angular distribution buffers, all flat
    /// across `[n_materials × MT_INELASTIC_COUNT × ...]`. See
    /// `yamc_gpu::neutron::xs::GpuNuclideXs` for the per-material
    /// shapes; this struct concatenates them across materials in
    /// `cell_to_material` order. Used by the kernel's slice-B
    /// inelastic angular sampling.
    pub angle_n_energies: Vec<u32>,
    /// CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    /// start in the tight `angle_energy_grid` / `n_mu` / `interp` arrays
    /// (issue #104). Length `n_slabs * MT_INELASTIC_COUNT`; slot count is
    /// `angle_n_energies[mat_slot]`.
    pub angle_ae_offset: Vec<u32>,
    pub angle_energy_grid: Vec<f64>,
    pub angle_n_mu: Vec<u32>,
    /// CSR base: index into the tight `angle_mu` / `cdf` / `pdf` arrays where
    /// ae-row's `(mu, cdf, pdf)` points start (issue #104). Length = total
    /// ae-rows; row length is `angle_n_mu[ae]`.
    pub angle_mu_offset: Vec<u32>,
    pub angle_mu: Vec<f64>,
    pub angle_cdf: Vec<f64>,
    /// Per-point PDF values matching `angle_mu` / `angle_cdf`,
    /// concatenated tight across all mu-points (no fixed per-axis stride).
    /// Phase-1a plumbing -- kernel doesn't consume yet; Phase-1b
    /// wires up the quadratic LinLin CDF inversion that mirrors
    /// `reaction_product::TabulatedAngleDistribution::sample`.
    pub angle_pdf: Vec<f64>,
    pub angle_interp: Vec<u32>,
    /// Per-MT outgoing-energy distribution buffers (slice C). See
    /// `yamc_gpu::neutron::xs::GpuNuclideXs` for the per-material
    /// shapes; this struct concatenates them across materials.
    /// `eout_kind` selects between the closed-form Q-value formula
    /// and CDF inversion sampling per (material × MT slot).
    pub eout_kind: Vec<u32>,
    pub eout_n_energies: Vec<u32>,
    /// CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    /// start in the tight `eout_energy_grid` / `n_x` / `interp` /
    /// `n_discrete` arrays (issue #104). Length `n_slabs *
    /// MT_INELASTIC_COUNT`; slot count is `eout_n_energies[mat_slot]`.
    pub eout_ae_offset: Vec<u32>,
    pub eout_histogram_interp: Vec<u32>,
    pub eout_energy_grid: Vec<f64>,
    pub eout_n_x: Vec<u32>,
    /// CSR base: index into the tight `eout_x` / `cdf` / `p` arrays where
    /// ae-row's `(x, cdf, p)` points start (issue #104). Length = total
    /// ae-rows; row length is `eout_n_x[ae]`.
    pub eout_x_offset: Vec<u32>,
    pub eout_x: Vec<f64>,
    pub eout_p: Vec<f64>,
    pub eout_cdf: Vec<f64>,
    pub eout_interp: Vec<u32>,
    pub eout_n_discrete: Vec<u32>,
    /// Per-MT correlated angle-energy buffers (slice D). Used when
    /// `eout_kind == EOUT_KIND_CORRELATED` for that slot. See
    /// `yamc_gpu::neutron::xs::GpuNuclideXs` for the per-material
    /// shapes; this struct concatenates them across materials.
    pub corr_n_energies: Vec<u32>,
    /// Per-MT count of equally-weighted correlated components (issue #111),
    /// length `n_slabs * MT_INELASTIC_COUNT`. `>= 2` triggers a per-collision
    /// component pick; component `c` occupies the `corr_n_energies /
    /// corr_n_components` rows at `corr_ae_offset[slot] + c * n_per_comp`.
    pub corr_n_components: Vec<u32>,
    /// CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    /// start in the tight `corr_energy_grid` / `corr_n_x` / `corr_interp` /
    /// `corr_n_discrete` arrays (issue #104). Length `n_slabs *
    /// MT_INELASTIC_COUNT`; slot count is `corr_n_energies[mat_slot]`.
    pub corr_ae_offset: Vec<u32>,
    pub corr_energy_grid: Vec<f64>,
    pub corr_n_x: Vec<u32>,
    /// CSR base: global x-point where ae-row's `(x, cdf, p)` / `n_mu` /
    /// `mu_interp` entries start (issue #104). Length = total ae-rows; row
    /// length is `corr_n_x[ae]`.
    pub corr_x_offset: Vec<u32>,
    pub corr_x: Vec<f64>,
    pub corr_cdf: Vec<f64>,
    pub corr_p: Vec<f64>,
    pub corr_interp: Vec<u32>,
    pub corr_n_discrete: Vec<u32>,
    pub corr_n_mu: Vec<u32>,
    /// CSR base: global mu-point where the x-point's `(mu, cdf, pdf)`
    /// sub-table starts (issue #104). Length = total x-points; indexed by the
    /// global x-point index (`corr_x_offset[ae] + j`).
    pub corr_mu_offset: Vec<u32>,
    pub corr_mu: Vec<f64>,
    pub corr_mu_cdf: Vec<f64>,
    pub corr_mu_pdf: Vec<f64>,
    pub corr_mu_interp: Vec<u32>,
    /// Per-MT scatter-frame flag, flat `[n_materials × MT_INELASTIC_COUNT]`.
    /// `1` = the angular table is in CM frame (kernel applies the
    /// two-body CM→lab conversion); `0` = lab frame.
    pub scatter_in_cm_per_mt: Vec<u32>,
    /// Temperature in Kelvin for each material, length `n_materials`.
    /// Used by the kernel's free-gas thermal scattering branch:
    /// when a neutron's energy drops below `free_gas_threshold · k_B · T`
    /// (and the target is heavier than the neutron) the kernel samples
    /// a target velocity from the CXS Maxwell distribution
    /// instead of treating the target as stationary.
    pub temperature_k_per_material: Vec<f64>,
    /// Free-gas resonance/thermal cutoff multiplier (`model.free_gas_threshold`,
    /// default `400.0`). The free-gas regime boundary is
    /// `free_gas_threshold · k_B · T`. Uploaded to the kernel as a 1-element
    /// f64 buffer; the CPU path and CPU twin pass the same value (issue #102).
    pub free_gas_threshold: f64,
    /// Elastic (MT 2) angular distribution, concatenated across
    /// materials. Same tight CSR layout as the per-MT inelastic angular
    /// buffers but for a single MT (no slot dimension): `n_energies` is
    /// per-material `[n_materials]`; the per-ae-row arrays
    /// (`energy_grid` / `n_mu` / `interp`) have length = total ae-rows;
    /// the per-mu-point arrays (`mu` / `cdf` / `pdf`) have length =
    /// total mu-points; the CSR base offsets (`ae_offset` / `mu_offset`)
    /// index into them.
    /// `n_energies[m] == 0` means no tabulated data -- kernel falls
    /// back to isotropic `mu_cm = 1 - 2·xi3` for that material.
    /// Per-slab incident-energy count, length `n_slabs`. 0 means "no
    /// tabulated data -> isotropic `mu_cm = 1 - 2·xi3`".
    pub elastic_angle_n_energies: Vec<u32>,
    /// CSR base: global ae-row index where slab `s`'s rows start in the
    /// tight `elastic_angle_energy_grid` / `n_mu` / `interp` arrays (issue
    /// #104). Length `n_slabs`; slab `s` spans `ae_offset[s] ..
    /// ae_offset[s] + n_energies[s]`.
    pub elastic_angle_ae_offset: Vec<u32>,
    pub elastic_angle_energy_grid: Vec<f64>,
    pub elastic_angle_n_mu: Vec<u32>,
    /// CSR base: index into the tight `elastic_angle_mu` / `cdf` / `pdf`
    /// arrays where ae-row `ae`'s `(mu, cdf, pdf)` points start (issue
    /// #104). Length = total ae-rows; row length is `elastic_angle_n_mu[ae]`.
    pub elastic_angle_mu_offset: Vec<u32>,
    pub elastic_angle_mu: Vec<f64>,
    pub elastic_angle_cdf: Vec<f64>,
    /// Per-point PDF for `elastic_angle_mu` / `elastic_angle_cdf`,
    /// same tight shape. Same purpose as `angle_pdf` but for elastic (MT 2).
    pub elastic_angle_pdf: Vec<f64>,
    pub elastic_angle_interp: Vec<u32>,
    /// Per-MT Kalbach-Mann tables, concatenated across materials.
    /// Same shape pattern as the per-MT corr buffers; populated
    /// by `KalbachSlot::from_reaction` per material × MT slot.
    /// The kernel routes to these buffers when
    /// `eout_kind[mat_slot] == EOUT_KIND_KALBACH_MANN`.
    pub km_n_energies: Vec<u32>,
    /// CSR base: global ae-row where (slab, MT) slot's incident-energy rows
    /// start in the tight `km_energy_grid` / `km_n_x` / `km_interp` /
    /// `km_n_discrete` arrays (issue #104). Length `n_slabs *
    /// MT_INELASTIC_COUNT`; slot count is `km_n_energies[mat_slot]`.
    pub km_ae_offset: Vec<u32>,
    pub km_energy_grid: Vec<f64>,
    pub km_interp: Vec<u32>,
    pub km_n_discrete: Vec<u32>,
    pub km_n_x: Vec<u32>,
    /// CSR base: index into the tight `km_x` / `km_p` / `km_c` / `km_r` /
    /// `km_a` arrays where ae-row's `(x, p, c, r, a)` points start (issue
    /// #104). Length = total ae-rows; row length is `km_n_x[ae]`.
    pub km_x_offset: Vec<u32>,
    pub km_x: Vec<f64>,
    pub km_p: Vec<f64>,
    pub km_c: Vec<f64>,
    pub km_r: Vec<f64>,
    pub km_a: Vec<f64>,
    /// Per-MT Evaporation E_out parameters, concatenated across
    /// materials. The kernel routes to these buffers when
    /// `eout_kind[mat_slot] == EOUT_KIND_EVAPORATION`.
    pub evap_n_energies: Vec<u32>,
    /// Per-MT Evaporation component count, concatenated across materials
    /// (`[n_materials × MT_INELASTIC_COUNT]`). `>= 2` selects the
    /// multi-component evaporation mixture sampler.
    pub evap_n_components: Vec<u32>,
    /// Tight CSR bases (issue #104). `evap_ae_offset` is the per-(slab,MT)
    /// global E_in-row base into `evap_energy_grid` / `evap_u` (slot row count
    /// is `evap_n_energies[mat_slot]`); `evap_theta_offset` is the per-(slab,MT)
    /// global base into the component-major `evap_theta`, where component `c`'s
    /// row starts at `evap_theta_offset[mat_slot] + c * evap_n_energies[mat_slot]`.
    pub evap_ae_offset: Vec<u32>,
    pub evap_theta_offset: Vec<u32>,
    pub evap_energy_grid: Vec<f64>,
    pub evap_theta: Vec<f64>,
    pub evap_u: Vec<f64>,
    /// Per-MT N-body phase-space buffers, concatenated across
    /// materials. Routed to when `eout_kind == EOUT_KIND_NBODY_PHASE_SPACE`.
    pub nbps_n_bodies: Vec<u32>,
    pub nbps_total_mass: Vec<f64>,
    /// Per-MT Maxwell E_out parameters, concatenated across
    /// materials. The kernel routes to these buffers when
    /// `eout_kind[mat_slot] == EOUT_KIND_MAXWELL`.
    pub maxwell_n_energies: Vec<u32>,
    /// Tight CSR base (issue #104): the per-(slab,MT) global E_in-row base into
    /// `maxwell_energy_grid` / `maxwell_theta`; slot row count is
    /// `maxwell_n_energies[mat_slot]`.
    pub maxwell_ae_offset: Vec<u32>,
    pub maxwell_energy_grid: Vec<f64>,
    pub maxwell_theta: Vec<f64>,
    pub maxwell_u: Vec<f64>,
    /// Per-MT Watt-inelastic E_out parameters, concatenated across
    /// materials. The kernel routes to these buffers when
    /// `eout_kind[mat_slot] == EOUT_KIND_WATT`. Both `a(E_in)` and
    /// `b(E_in)` are tabulated on a shared incident-energy grid;
    /// `u` is the per-slot restriction energy.
    pub watt_n_energies: Vec<u32>,
    /// Tight CSR base (issue #104): the per-(slab,MT) global E_in-row base into
    /// `watt_energy_grid` / `watt_a` / `watt_b`; slot row count is
    /// `watt_n_energies[mat_slot]`.
    pub watt_ae_offset: Vec<u32>,
    pub watt_energy_grid: Vec<f64>,
    pub watt_a: Vec<f64>,
    pub watt_b: Vec<f64>,
    pub watt_u: Vec<f64>,
    /// Per-(material, nuclide) URR (unresolved resonance region) probability
    /// table data, concatenated across the global slab index (issue #210: URR
    /// is applied to every in-range URR nuclide, so it is slab-keyed, one row
    /// per (material, nuclide), including the void slab). See
    /// `yamc_gpu::neutron::xs::URR_META_*` / `URR_XS_*` for the packings.
    /// Non-URR slabs have `urr_meta[.., PRESENT] = 0` and a zero-length table;
    /// the kernel keeps their smooth XS.
    pub urr_meta: Vec<u32>,
    /// Tight CSR bases (issue #104), one entry per slab.
    /// `urr_ae_offset[slab]` is the global base into `urr_energy_grid`
    /// (sum of prior slabs' `n_energies`); `urr_cdf_offset[slab]` is
    /// the global base into `urr_cdf` (sum of prior slabs'
    /// `n_energies × n_cdf`). The `urr_xs` base is derived from
    /// `urr_cdf_offset` (same cdf cell index × `URR_XS_COLS`), so no
    /// separate `urr_xs_offset` is needed.
    pub urr_ae_offset: Vec<u32>,
    pub urr_cdf_offset: Vec<u32>,
    pub urr_energy_grid: Vec<f64>,
    pub urr_cdf: Vec<f64>,
    pub urr_xs: Vec<f64>,
    /// Per-slab atom density (atoms/barn-cm), one entry per slab (0 for
    /// non-URR slabs). Scales a URR nuclide's perturbed micro XS to a
    /// macroscopic contribution (issue #210).
    pub urr_atom_density: Vec<f64>,
    /// Per-collision nuclide-selection inputs (issue #74, Stage 1): per-
    /// (material, nuclide) macroscopic totals + AWRs + slab offset table.
    /// Multi-nuclide materials select the struck nuclide per collision and use
    /// its exact AWR for elastic kinematics; single-nuclide materials are a
    /// no-op (no extra RNG draw).
    pub nuclide_select: NuclideSelectInputs,
}

/// Build flat-buffer inputs for the GPU multi-cell kernel from a yamc
/// `Model`. Returns a `GpuTranslateError` for any model feature the
/// kernel doesn't support; see `crate::gpu` for the rejection list.
pub fn translate_for_gpu(
    model: &Model,
    n_particles: usize,
    base_seed: u64,
) -> Result<GpuTransportInputs, GpuTranslateError> {
    if n_particles == 0 {
        return Err(GpuTranslateError::NoParticles);
    }
    if model.sources.is_empty() {
        return Err(GpuTranslateError::NoSources);
    }
    // Coupled secondary-photon production (`transport_secondary_photons`) is
    // supported as of S6: the neutron kernel emits photons into a device bank
    // that the coupled dispatch then transports through the photon kernel. A
    // photon SOURCE on this neutron-translate path is still rejected (those
    // models route to `run_on_gpu_photon` before reaching here); the explicit
    // guard below is defensive, matching the dispatch's all-photon routing.
    if model
        .sources
        .iter()
        .any(|s| s.particle_type() == yamc_particle::particle::ParticleType::Photon)
    {
        return Err(GpuTranslateError::PhotonSourceOnNeutronPath);
    }
    // D1S (`use_decay_photons`) is supported on GPU via the coupled dispatch:
    // the neutron kernel emits decay photons into the device bank (tagged by
    // parent radionuclide), the photon sub-pass transports them, and the
    // photon tally honours the `parent_nuclides` filter. No rejection here.
    for ps in &model.sources {
        let pt = ps.particle_type();
        if pt != yamc_particle::particle::ParticleType::Neutron {
            return Err(GpuTranslateError::NonNeutronSource {
                particle_type: format!("{pt:?}"),
            });
        }
    }

    #[cfg(feature = "mesh")]
    let geometry = match &model.geometry {
        GeometryKind::Csg(g) => g,
        GeometryKind::Mesh(_) => return Err(GpuTranslateError::MeshGeometryUnsupported),
    };
    #[cfg(not(feature = "mesh"))]
    let GeometryKind::Csg(geometry) = &model.geometry;
    if geometry.has_mesh_fills() {
        return Err(GpuTranslateError::MeshFillUnsupported);
    }

    let (cell_aabbs, cell_to_material, has_void) =
        translate_cells(&geometry.cells, geometry.materials.len())?;
    let (surface_types, surface_params, surface_boundaries) = translate_surfaces(&geometry.cells)?;
    let region_program = translate_region_program(&geometry.cells);
    let mut translated = translate_materials(&geometry.materials)?;
    // Void (material-less) cells map to a synthetic all-zero material
    // slot appended at index `n_materials`. Its zero XS rows make
    // `sigma_t = 0` in the kernel, so the particle streams to the next
    // surface with no collision (track-length flux still accrues;
    // reaction-rate / total / heating score 0). See `GpuNuclideXs::void`.
    if has_void {
        // Void material's coarse grid (issue #212): a minimal 2-point grid. The
        // void never collides (sigma_t = 0), but the kernel still computes the
        // coarse bracket for its cell every step, so the grid needs >= 2
        // monotonic points; its per-MT slab (never read) stays tiny. Reuse the
        // global fine grid's endpoints (strictly increasing log energies). Record
        // the packed `coarse_meta` row and append the grid BEFORE the void slab
        // so the per-MT base reflects the pre-void length.
        let void_coarse_log: Vec<f64> = if translated.log_energy_grid.len() >= 2 {
            let last = translated.log_energy_grid.len() - 1;
            vec![
                translated.log_energy_grid[0],
                translated.log_energy_grid[last],
            ]
        } else {
            vec![0.0, 1.0]
        };
        translated
            .coarse_meta
            .push(translated.coarse_log_energy_grid.len() as u32);
        translated.coarse_meta.push(void_coarse_log.len() as u32);
        // COL_COARSE_MT_BASE: the void slab's base into `permt_meta` ROWS (issue
        // #212). Equals `nuc_off * MT_INELASTIC_COUNT`, which is the current
        // permt_meta row count (all real slabs appended, void not yet).
        translated
            .coarse_meta
            .push((translated.permt_meta.len() / PERMT_META_COLS as usize) as u32);
        translated
            .coarse_log_energy_grid
            .extend_from_slice(&void_coarse_log);
        // Void material's fine grid (issue #212): the same minimal 2-point grid
        // rationale as the coarse grid above (never collides, but the kernel
        // still computes the fine bracket for its cell). Record the packed
        // `fine_meta` row (grid offset == aggregate-XS base, fine_n, nuc base ==
        // current concatenated nuc length) BEFORE appending the grid / rows.
        let void_fine_log: Vec<f64> = if translated.log_energy_grid.len() >= 2 {
            let last = translated.log_energy_grid.len() - 1;
            vec![
                translated.log_energy_grid[0],
                translated.log_energy_grid[last],
            ]
        } else {
            vec![0.0, 1.0]
        };
        let void_fine_n = void_fine_log.len();
        translated
            .fine_meta
            .push(translated.fine_log_energy_grid.len() as u32);
        translated.fine_meta.push(void_fine_n as u32);
        translated
            .fine_meta
            .push(translated.nuclide_select.nuc_macro_total.len() as u32);
        // `void_fine_log` values are already log energies (taken from the log
        // `log_energy_grid`), so append + hand to `void` as-is.
        translated
            .fine_log_energy_grid
            .extend_from_slice(&void_fine_log);
        let void_xs = GpuNuclideXs::void(void_fine_log, void_coarse_log);
        append_material_xs(&mut translated, &void_xs);
        // One empty per-MT slab for the void slot (its single nuclide never
        // collides), keeping the slab-major per-MT buffers aligned (#74 2b).
        append_void_per_mt_slab(&mut translated, &void_xs);
        // One empty elastic-angle slab for the void slot (its single nuclide
        // never collides), keeping the slab table aligned with `nuclide_select`.
        push_empty_elastic_angle_slab(&mut translated);
        // Mirror the void slot in the nuclide-selection slab table (single
        // nuclide => never drawn; void cells never collide anyway). Its nuc row
        // is `void_fine_n` wide to match the per-material FINE-grid CSR stride.
        translated
            .nuclide_select
            .push_single_nuclide_material(0.0, void_fine_n);
    }
    let (seeds, energies, positions, directions) =
        sample_initial_particles(model, n_particles, base_seed);

    Ok(GpuTransportInputs {
        seeds,
        energies,
        positions,
        directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        region_program,
        log_energy_grid: translated.log_energy_grid,
        coarse_log_energy_grid: translated.coarse_log_energy_grid,
        coarse_meta: translated.coarse_meta,
        fine_log_energy_grid: translated.fine_log_energy_grid,
        fine_meta: translated.fine_meta,
        xs_elastic_per_material: translated.xs_elastic_per_material,
        xs_absorption_per_material: translated.xs_absorption_per_material,
        xs_inelastic_per_material: translated.xs_inelastic_per_material,
        xs_inelastic_per_mt_sparse: translated.xs_inelastic_per_mt_sparse,
        permt_meta: translated.permt_meta,
        xs_fission_per_material: translated.xs_fission_per_material,
        nu_bar_per_material: translated.nu_bar_per_material,
        beta_delayed_per_material: translated.beta_delayed_per_material,
        fission_a_per_material: translated.fission_a_per_material,
        fission_b_per_material: translated.fission_b_per_material,
        fission_eout_kind_per_material: translated.fission_eout_kind_per_material,
        fission_eout_n_energies_per_material: translated.fission_eout_n_energies_per_material,
        fission_eout_ae_offset: translated.fission_eout_ae_offset,
        fission_eout_energy_grid_per_material: translated.fission_eout_energy_grid_per_material,
        fission_eout_n_x_per_material: translated.fission_eout_n_x_per_material,
        fission_eout_x_offset: translated.fission_eout_x_offset,
        fission_eout_x_per_material: translated.fission_eout_x_per_material,
        fission_eout_cdf_per_material: translated.fission_eout_cdf_per_material,
        fission_eout_p_per_material: translated.fission_eout_p_per_material,
        fission_eout_interp_per_material: translated.fission_eout_interp_per_material,
        target_mass_per_material: translated.target_mass_per_material,
        q_inelastic_per_mt: translated.q_inelastic_per_mt,
        yield_per_mt_sparse: translated.yield_per_mt_sparse,
        angle_n_energies: translated.angle_n_energies,
        angle_ae_offset: translated.angle_ae_offset,
        angle_energy_grid: translated.angle_energy_grid,
        angle_n_mu: translated.angle_n_mu,
        angle_mu_offset: translated.angle_mu_offset,
        angle_mu: translated.angle_mu,
        angle_cdf: translated.angle_cdf,
        angle_pdf: translated.angle_pdf,
        angle_interp: translated.angle_interp,
        eout_kind: translated.eout_kind,
        eout_n_energies: translated.eout_n_energies,
        eout_ae_offset: translated.eout_ae_offset,
        eout_histogram_interp: translated.eout_histogram_interp,
        eout_energy_grid: translated.eout_energy_grid,
        eout_n_x: translated.eout_n_x,
        eout_x_offset: translated.eout_x_offset,
        eout_x: translated.eout_x,
        eout_p: translated.eout_p,
        eout_cdf: translated.eout_cdf,
        eout_interp: translated.eout_interp,
        eout_n_discrete: translated.eout_n_discrete,
        corr_n_energies: translated.corr_n_energies,
        corr_n_components: translated.corr_n_components,
        corr_ae_offset: translated.corr_ae_offset,
        corr_energy_grid: translated.corr_energy_grid,
        corr_n_x: translated.corr_n_x,
        corr_x_offset: translated.corr_x_offset,
        corr_x: translated.corr_x,
        corr_cdf: translated.corr_cdf,
        corr_p: translated.corr_p,
        corr_interp: translated.corr_interp,
        corr_n_discrete: translated.corr_n_discrete,
        corr_n_mu: translated.corr_n_mu,
        corr_mu_offset: translated.corr_mu_offset,
        corr_mu: translated.corr_mu,
        corr_mu_cdf: translated.corr_mu_cdf,
        corr_mu_pdf: translated.corr_mu_pdf,
        corr_mu_interp: translated.corr_mu_interp,
        scatter_in_cm_per_mt: translated.scatter_in_cm_per_mt,
        elastic_angle_n_energies: translated.elastic_angle_n_energies,
        elastic_angle_ae_offset: translated.elastic_angle_ae_offset,
        elastic_angle_energy_grid: translated.elastic_angle_energy_grid,
        elastic_angle_n_mu: translated.elastic_angle_n_mu,
        elastic_angle_mu_offset: translated.elastic_angle_mu_offset,
        elastic_angle_mu: translated.elastic_angle_mu,
        elastic_angle_cdf: translated.elastic_angle_cdf,
        elastic_angle_pdf: translated.elastic_angle_pdf,
        elastic_angle_interp: translated.elastic_angle_interp,
        temperature_k_per_material: translated.temperature_k_per_material,
        free_gas_threshold: model.free_gas_threshold,
        km_n_energies: translated.km_n_energies,
        km_ae_offset: translated.km_ae_offset,
        km_energy_grid: translated.km_energy_grid,
        km_interp: translated.km_interp,
        km_n_discrete: translated.km_n_discrete,
        km_n_x: translated.km_n_x,
        km_x_offset: translated.km_x_offset,
        km_x: translated.km_x,
        km_p: translated.km_p,
        km_c: translated.km_c,
        km_r: translated.km_r,
        km_a: translated.km_a,
        evap_n_energies: translated.evap_n_energies,
        evap_n_components: translated.evap_n_components,
        evap_ae_offset: translated.evap_ae_offset,
        evap_theta_offset: translated.evap_theta_offset,
        evap_energy_grid: translated.evap_energy_grid,
        evap_theta: translated.evap_theta,
        evap_u: translated.evap_u,
        nbps_n_bodies: translated.nbps_n_bodies,
        nbps_total_mass: translated.nbps_total_mass,
        maxwell_n_energies: translated.maxwell_n_energies,
        maxwell_ae_offset: translated.maxwell_ae_offset,
        maxwell_energy_grid: translated.maxwell_energy_grid,
        maxwell_theta: translated.maxwell_theta,
        maxwell_u: translated.maxwell_u,
        watt_n_energies: translated.watt_n_energies,
        watt_ae_offset: translated.watt_ae_offset,
        watt_energy_grid: translated.watt_energy_grid,
        watt_a: translated.watt_a,
        watt_b: translated.watt_b,
        watt_u: translated.watt_u,
        urr_meta: translated.urr_meta,
        urr_ae_offset: translated.urr_ae_offset,
        urr_cdf_offset: translated.urr_cdf_offset,
        urr_energy_grid: translated.urr_energy_grid,
        urr_cdf: translated.urr_cdf,
        urr_xs: translated.urr_xs,
        urr_atom_density: translated.urr_atom_density,
        nuclide_select: translated.nuclide_select,
    })
}

/// Stride-6 AABB for every cell + material index per cell.
///
/// Finite void (material-less) cells map to a synthetic void material
/// slot at index `n_materials` (one past the real materials): the caller
/// appends a `GpuNuclideXs::void` slot when the returned `has_void` is
/// `true`, so the kernel's `cell_to_material[void] == n_materials` lookup
/// lands on an all-zero XS row and the particle streams through.
///
/// Cell ordering is preserved one-to-one: the GPU cell index is the
/// position in `cells`, which the kernel uses to index both
/// `cell_to_material` and the per-tally `cell_to_bin` map. A cell is
/// never dropped or reordered here.
///
/// **Implicit complement.** A cell whose region has an unbounded bounding
/// box is the geometry's implicit complement (the "rest of space" outside
/// every explicit cell, e.g. `outer.above` or `~region`). It is encoded as
/// an *empty* sentinel AABB (`lower = +inf`, `upper = -inf`): the kernel's
/// point-in-AABB test (`px >= min && px <= max`) can never be true for a
/// finite point, so the cell is never selected and any particle reaching it
/// is treated as having left the geometry (leakage / history terminates).
/// That matches a *vacuum* implicit complement on the CPU. It is only sound
/// when that cell is vacuum; an unbounded cell carrying a material is
/// rejected with a clear error rather than silently mis-simulated.
///
/// Returns `(aabbs, cell_to_material, has_void)`.
pub(crate) fn translate_cells(
    cells: &[Cell],
    n_materials: usize,
) -> Result<(Vec<f64>, Vec<u32>, bool), GpuTranslateError> {
    let mut aabbs = Vec::with_capacity(cells.len() * 6);
    let mut cell_to_material = Vec::with_capacity(cells.len());
    let void_slot = n_materials as u32;
    let mut has_void = false;

    for cell in cells {
        let bbox = cell.region.bounding_box();
        if !bbox.is_finite() {
            // Unbounded cell. Only a *vacuum* implicit complement (no
            // material) is supported on GPU -- emit an empty sentinel
            // AABB so the kernel never matches it and the particle
            // leaks. A material-filled unbounded cell can't be
            // represented by point-in-AABB cell finding, so reject it.
            if cell.material_idx.is_some() {
                return Err(GpuTranslateError::CellRegionUnsupported {
                    cell_id: cell.cell_id,
                    reason: "material-filled cell with unbounded extent (a material implicit \
                             complement has no AABB on GPU; only a vacuum implicit complement \
                             is supported)"
                        .to_string(),
                });
            }
            // Empty AABB sentinel: lower = +inf, upper = -inf. The
            // kernel routes non-finite AABBs through its unbounded
            // fallback list and tests `>= / <=` against these bounds,
            // which is always false -- so the cell is never found.
            aabbs.extend_from_slice(&[f64::INFINITY; 3]);
            aabbs.extend_from_slice(&[f64::NEG_INFINITY; 3]);
            // Placeholder material slot: never read, because the cell is
            // never matched. Keeps `cell_to_material` aligned with the
            // positional cell index.
            cell_to_material.push(0);
            continue;
        }
        aabbs.extend_from_slice(&bbox.lower_left);
        aabbs.extend_from_slice(&bbox.upper_right);

        match cell.material_idx {
            Some(idx) => cell_to_material.push(idx),
            None => {
                // Void cell: map to the synthetic void material slot.
                cell_to_material.push(void_slot);
                has_void = true;
            }
        }
    }

    Ok((aabbs, cell_to_material, has_void))
}

/// Walk every cell's region, dedupe surfaces by `Arc::as_ptr`, encode
/// each as `(type, [params; 7])`. Both `Transmission` and `Vacuum`
/// boundaries are supported by the kernel; `surface_boundaries[i]`
/// matches yamc-geo's `BoundaryType` enum order so the kernel sees
/// `0` for transmission and `1` for vacuum.
#[allow(clippy::type_complexity)]
pub(crate) fn translate_surfaces(
    cells: &[Cell],
) -> Result<(Vec<u32>, Vec<f64>, Vec<u32>), GpuTranslateError> {
    // Dedupe across cells via pointer identity; the same `Arc<Surface>`
    // is shared by every cell that references it.
    let mut seen: HashMap<*const Surface, ()> = HashMap::new();
    let mut surfaces_in_order: Vec<Arc<Surface>> = Vec::new();
    for cell in cells {
        for (surface_arc, _sense) in &cell.cached_surfaces {
            let ptr = Arc::as_ptr(surface_arc);
            if seen.insert(ptr, ()).is_none() {
                surfaces_in_order.push(Arc::clone(surface_arc));
            }
        }
    }

    let mut surface_types = Vec::with_capacity(surfaces_in_order.len());
    let mut surface_params = Vec::with_capacity(surfaces_in_order.len() * SURFACE_PARAM_STRIDE);
    let mut surface_boundaries = Vec::with_capacity(surfaces_in_order.len());

    for surface in &surfaces_in_order {
        let (kind, params) = encode_surface(&surface.kind);
        surface_types.push(kind);
        surface_params.extend_from_slice(&params);
        surface_boundaries.push(match surface.boundary {
            BoundaryType::Transmission => 0,
            BoundaryType::Vacuum => 1,
        });
    }

    Ok((surface_types, surface_params, surface_boundaries))
}

/// Build the flat per-cell CSG region program the kernel evaluates during
/// cell finding.
///
/// Mirrors `yamc_geo::region::FlatRegion`'s RPN compilation, but interns
/// surfaces against the *same* deduped index order [`translate_surfaces`]
/// produces (pointer identity, in cell-then-region order), so the `surf_idx`
/// in every op word indexes the `surface_types` / `surface_params` buffers.
///
/// Layout (see `yamc_gpu::common::geometry::region_eval`): the first
/// `n_cells + 1` words are prefix offsets into the op stream; cell `c`'s ops
/// span op-indices `[program[c], program[c + 1])`. The op stream follows the
/// header. Each op word is `(opcode << REGION_OP_SHIFT) | surface_idx`, with
/// `Above`/`Below` mapping to the same surface sense the CPU
/// `FlatRegion::contains` uses (`Above`: `evaluate > 0`, `Below`:
/// `evaluate < 0`) and `Intersection`/`Union`/`Complement` to `And`/`Or`/`Not`.
pub(crate) fn translate_region_program(cells: &[Cell]) -> Vec<u32> {
    // Same dedup as `translate_surfaces`: pointer identity, cell-then-region
    // order. This `ptr -> idx` map is the surface index every op references.
    let mut surface_idx: HashMap<*const Surface, u32> = HashMap::new();
    let mut next_idx = 0u32;
    for cell in cells {
        for (surface_arc, _sense) in &cell.cached_surfaces {
            let ptr = Arc::as_ptr(surface_arc);
            surface_idx.entry(ptr).or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });
        }
    }

    // Emit RPN for one region expression, appending op words. A direct
    // structural mirror of `FlatRegion::from_expr`: `Above`/`Below` push
    // their surface (same sense the CPU uses), and `Complement` emits an
    // explicit `Not` rather than pushing into the leaves -- this keeps the
    // exact boundary behaviour (`Not(Above)` at `evaluate == 0` is `true`,
    // which a De-Morgan'd `Below` would get wrong).
    fn emit(expr: &RegionExpr, surface_idx: &HashMap<*const Surface, u32>, ops: &mut Vec<u32>) {
        match expr {
            RegionExpr::Halfspace(hs) => {
                let (surf, above) = match hs {
                    HalfspaceType::Above(s) => (s, true),
                    HalfspaceType::Below(s) => (s, false),
                };
                let idx = surface_idx[&Arc::as_ptr(surf)];
                let op = if above {
                    REGION_OP_ABOVE
                } else {
                    REGION_OP_BELOW
                };
                ops.push((op << REGION_OP_SHIFT) | idx);
            }
            RegionExpr::Intersection(a, b) => {
                emit(a, surface_idx, ops);
                emit(b, surface_idx, ops);
                ops.push(REGION_OP_AND << REGION_OP_SHIFT);
            }
            RegionExpr::Union(a, b) => {
                emit(a, surface_idx, ops);
                emit(b, surface_idx, ops);
                ops.push(REGION_OP_OR << REGION_OP_SHIFT);
            }
            RegionExpr::Complement(inner) => {
                emit(inner, surface_idx, ops);
                ops.push(REGION_OP_NOT << REGION_OP_SHIFT);
            }
        }
    }

    let n_cells = cells.len();
    let mut ops: Vec<u32> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(n_cells + 1);
    for cell in cells {
        offsets.push(ops.len() as u32);
        emit(&cell.region.expr, &surface_idx, &mut ops);
    }
    offsets.push(ops.len() as u32);

    // Header (n_cells + 1 prefix offsets) followed by the op stream.
    let mut program = Vec::with_capacity(offsets.len() + ops.len());
    program.extend_from_slice(&offsets);
    program.extend_from_slice(&ops);
    program
}

/// Map a yamc-geo `SurfaceKind` to the kernel's `(type, [param; 10])`
/// encoding. The kernel reads a uniform stride-10 slot per surface and
/// dispatches on the type discriminant; unused tail slots are zero.
/// The match is exhaustive over every `SurfaceKind`, so a future
/// variant added without an encoding here is a compile error rather
/// than a runtime reject.
fn encode_surface(kind: &SurfaceKind) -> (u32, [f64; 10]) {
    match *kind {
        SurfaceKind::Sphere { x0, y0, z0, radius } => (
            SURFACE_SPHERE,
            [x0, y0, z0, radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        SurfaceKind::Plane { a, b, c, d } => {
            (SURFACE_PLANE, [a, b, c, d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        SurfaceKind::Cylinder {
            axis,
            origin,
            radius,
        } => (
            SURFACE_CYLINDER,
            [
                origin[0], origin[1], origin[2], radius, axis[0], axis[1], axis[2], 0.0, 0.0, 0.0,
            ],
        ),
        SurfaceKind::ZTorus {
            x0,
            y0,
            z0,
            a,
            b,
            c,
        } => (SURFACE_ZTORUS, [x0, y0, z0, a, b, c, 0.0, 0.0, 0.0, 0.0]),
        SurfaceKind::XTorus {
            x0,
            y0,
            z0,
            a,
            b,
            c,
        } => (SURFACE_XTORUS, [x0, y0, z0, a, b, c, 0.0, 0.0, 0.0, 0.0]),
        SurfaceKind::YTorus {
            x0,
            y0,
            z0,
            a,
            b,
            c,
        } => (SURFACE_YTORUS, [x0, y0, z0, a, b, c, 0.0, 0.0, 0.0, 0.0]),
        SurfaceKind::Quadric {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            j,
            k,
        } => (SURFACE_QUADRIC, [a, b, c, d, e, f, g, h, j, k]),
        SurfaceKind::Cone {
            apex,
            axis,
            tan2_theta,
        } => (
            SURFACE_CONE,
            [
                apex[0], apex[1], apex[2], axis[0], axis[1], axis[2], tan2_theta, 0.0, 0.0, 0.0,
            ],
        ),
    }
}

/// Aggregated XS + angle buffers across every material in the
/// geometry. Each `Vec<f64>` / `Vec<u32>` is the per-material
/// concatenation of the corresponding field on `GpuNuclideXs`, in
/// `materials` order.
#[derive(Default)]
struct TranslatedMaterials {
    log_energy_grid: Vec<f64>,
    coarse_log_energy_grid: Vec<f64>,
    /// Packed `[n_materials × COARSE_META_COLS]` per-material coarse-grid
    /// descriptor (issue #212), pushed one row per material (real then void).
    coarse_meta: Vec<u32>,
    /// Concatenated per-material FINE grids (issue #212).
    fine_log_energy_grid: Vec<f64>,
    /// Packed `[n_materials × FINE_META_COLS]` per-material fine-grid descriptor
    /// (issue #212), pushed one row per material (real then void).
    fine_meta: Vec<u32>,
    xs_elastic_per_material: Vec<f64>,
    xs_absorption_per_material: Vec<f64>,
    xs_inelastic_per_material: Vec<f64>,
    /// SPARSE per-MT inelastic xs values, concatenated tight in (slab, MT slot)
    /// order (issue #212). Located by `permt_meta`.
    xs_inelastic_per_mt_sparse: Vec<f64>,
    /// Packed `[n_slab × MT_INELASTIC_COUNT × PERMT_META_COLS]` per-(slab, MT
    /// slot) sparse descriptor `[value_offset, i_start, n_stored]` (issue #212).
    permt_meta: Vec<u32>,
    xs_fission_per_material: Vec<f64>,
    nu_bar_per_material: Vec<f64>,
    beta_delayed_per_material: Vec<f64>,
    fission_a_per_material: Vec<f64>,
    fission_b_per_material: Vec<f64>,
    fission_eout_kind_per_material: Vec<u32>,
    fission_eout_n_energies_per_material: Vec<u32>,
    fission_eout_ae_offset: Vec<u32>,
    fission_eout_energy_grid_per_material: Vec<f64>,
    fission_eout_n_x_per_material: Vec<u32>,
    fission_eout_x_offset: Vec<u32>,
    fission_eout_x_per_material: Vec<f64>,
    fission_eout_cdf_per_material: Vec<f64>,
    fission_eout_p_per_material: Vec<f64>,
    fission_eout_interp_per_material: Vec<u32>,
    target_mass_per_material: Vec<f64>,
    q_inelastic_per_mt: Vec<f64>,
    /// SPARSE per-MT yield values, parallel to `xs_inelastic_per_mt_sparse` and
    /// located by the same `permt_meta` row (issue #212).
    yield_per_mt_sparse: Vec<f64>,
    angle_n_energies: Vec<u32>,
    angle_ae_offset: Vec<u32>,
    angle_energy_grid: Vec<f64>,
    angle_n_mu: Vec<u32>,
    angle_mu_offset: Vec<u32>,
    angle_mu: Vec<f64>,
    angle_cdf: Vec<f64>,
    angle_pdf: Vec<f64>,
    angle_interp: Vec<u32>,
    eout_kind: Vec<u32>,
    eout_n_energies: Vec<u32>,
    eout_ae_offset: Vec<u32>,
    eout_histogram_interp: Vec<u32>,
    eout_energy_grid: Vec<f64>,
    eout_n_x: Vec<u32>,
    eout_x_offset: Vec<u32>,
    eout_x: Vec<f64>,
    eout_p: Vec<f64>,
    eout_cdf: Vec<f64>,
    eout_interp: Vec<u32>,
    eout_n_discrete: Vec<u32>,
    corr_n_energies: Vec<u32>,
    corr_n_components: Vec<u32>,
    corr_ae_offset: Vec<u32>,
    corr_energy_grid: Vec<f64>,
    corr_n_x: Vec<u32>,
    corr_x_offset: Vec<u32>,
    corr_x: Vec<f64>,
    corr_cdf: Vec<f64>,
    corr_p: Vec<f64>,
    corr_interp: Vec<u32>,
    corr_n_discrete: Vec<u32>,
    corr_n_mu: Vec<u32>,
    corr_mu_offset: Vec<u32>,
    corr_mu: Vec<f64>,
    corr_mu_cdf: Vec<f64>,
    corr_mu_pdf: Vec<f64>,
    corr_mu_interp: Vec<u32>,
    scatter_in_cm_per_mt: Vec<u32>,
    elastic_angle_n_energies: Vec<u32>,
    elastic_angle_ae_offset: Vec<u32>,
    elastic_angle_energy_grid: Vec<f64>,
    elastic_angle_n_mu: Vec<u32>,
    elastic_angle_mu_offset: Vec<u32>,
    elastic_angle_mu: Vec<f64>,
    elastic_angle_cdf: Vec<f64>,
    elastic_angle_pdf: Vec<f64>,
    elastic_angle_interp: Vec<u32>,
    temperature_k_per_material: Vec<f64>,
    km_n_energies: Vec<u32>,
    km_ae_offset: Vec<u32>,
    km_energy_grid: Vec<f64>,
    km_interp: Vec<u32>,
    km_n_discrete: Vec<u32>,
    km_n_x: Vec<u32>,
    km_x_offset: Vec<u32>,
    km_x: Vec<f64>,
    km_p: Vec<f64>,
    km_c: Vec<f64>,
    km_r: Vec<f64>,
    km_a: Vec<f64>,
    evap_n_energies: Vec<u32>,
    evap_n_components: Vec<u32>,
    evap_ae_offset: Vec<u32>,
    evap_theta_offset: Vec<u32>,
    evap_energy_grid: Vec<f64>,
    evap_theta: Vec<f64>,
    evap_u: Vec<f64>,
    nbps_n_bodies: Vec<u32>,
    nbps_total_mass: Vec<f64>,
    maxwell_n_energies: Vec<u32>,
    maxwell_ae_offset: Vec<u32>,
    maxwell_energy_grid: Vec<f64>,
    maxwell_theta: Vec<f64>,
    maxwell_u: Vec<f64>,
    watt_n_energies: Vec<u32>,
    watt_ae_offset: Vec<u32>,
    watt_energy_grid: Vec<f64>,
    watt_a: Vec<f64>,
    watt_b: Vec<f64>,
    watt_u: Vec<f64>,
    urr_meta: Vec<u32>,
    // Tight CSR bases (issue #104), one entry per slab (issue #210).
    urr_ae_offset: Vec<u32>,
    urr_cdf_offset: Vec<u32>,
    urr_energy_grid: Vec<f64>,
    urr_cdf: Vec<f64>,
    urr_xs: Vec<f64>,
    urr_atom_density: Vec<f64>,
    /// Per-collision nuclide-selection inputs (issue #74): per-(material,
    /// nuclide) macroscopic totals + AWRs + the slab offset table.
    nuclide_select: NuclideSelectInputs,
}

/// For each material, extract XS + per-MT angular data via yamc-gpu's
/// `extract_material_xs`. All materials share the first material's
/// energy grid (resampled onto it).
fn translate_materials(
    materials: &[Arc<Material>],
) -> Result<TranslatedMaterials, GpuTranslateError> {
    if materials.is_empty() {
        // The geometry has no materials at all -- a degenerate case.
        // Return empty arrays; the kernel tolerates n_materials=0 only
        // if no cell_to_material entry references one. The cell-side
        // validation already requires cell.material_idx, so if we get
        // here something is structurally off.
        return Ok(TranslatedMaterials {
            log_energy_grid: Vec::new(),
            coarse_log_energy_grid: Vec::new(),
            coarse_meta: Vec::new(),
            fine_log_energy_grid: Vec::new(),
            fine_meta: Vec::new(),
            xs_elastic_per_material: Vec::new(),
            xs_absorption_per_material: Vec::new(),
            xs_inelastic_per_material: Vec::new(),
            xs_inelastic_per_mt_sparse: Vec::new(),
            permt_meta: Vec::new(),
            xs_fission_per_material: Vec::new(),
            nu_bar_per_material: Vec::new(),
            beta_delayed_per_material: Vec::new(),
            fission_a_per_material: Vec::new(),
            fission_b_per_material: Vec::new(),
            fission_eout_kind_per_material: Vec::new(),
            fission_eout_n_energies_per_material: Vec::new(),
            fission_eout_ae_offset: Vec::new(),
            fission_eout_energy_grid_per_material: Vec::new(),
            fission_eout_n_x_per_material: Vec::new(),
            fission_eout_x_offset: Vec::new(),
            fission_eout_x_per_material: Vec::new(),
            fission_eout_cdf_per_material: Vec::new(),
            fission_eout_p_per_material: Vec::new(),
            fission_eout_interp_per_material: Vec::new(),
            target_mass_per_material: Vec::new(),
            q_inelastic_per_mt: Vec::new(),
            yield_per_mt_sparse: Vec::new(),
            angle_n_energies: Vec::new(),
            angle_ae_offset: Vec::new(),
            angle_energy_grid: Vec::new(),
            angle_n_mu: Vec::new(),
            angle_mu_offset: Vec::new(),
            angle_mu: Vec::new(),
            angle_cdf: Vec::new(),
            angle_pdf: Vec::new(),
            angle_interp: Vec::new(),
            eout_kind: Vec::new(),
            eout_n_energies: Vec::new(),
            eout_ae_offset: Vec::new(),
            eout_histogram_interp: Vec::new(),
            eout_energy_grid: Vec::new(),
            eout_n_x: Vec::new(),
            eout_x_offset: Vec::new(),
            eout_x: Vec::new(),
            eout_p: Vec::new(),
            eout_cdf: Vec::new(),
            eout_interp: Vec::new(),
            eout_n_discrete: Vec::new(),
            corr_n_energies: Vec::new(),
            corr_n_components: Vec::new(),
            corr_ae_offset: Vec::new(),
            corr_energy_grid: Vec::new(),
            corr_n_x: Vec::new(),
            corr_x_offset: Vec::new(),
            corr_x: Vec::new(),
            corr_cdf: Vec::new(),
            corr_p: Vec::new(),
            corr_interp: Vec::new(),
            corr_n_discrete: Vec::new(),
            corr_n_mu: Vec::new(),
            corr_mu_offset: Vec::new(),
            corr_mu: Vec::new(),
            corr_mu_cdf: Vec::new(),
            corr_mu_pdf: Vec::new(),
            corr_mu_interp: Vec::new(),
            scatter_in_cm_per_mt: Vec::new(),
            elastic_angle_n_energies: Vec::new(),
            elastic_angle_ae_offset: Vec::new(),
            elastic_angle_energy_grid: Vec::new(),
            elastic_angle_n_mu: Vec::new(),
            elastic_angle_mu_offset: Vec::new(),
            elastic_angle_mu: Vec::new(),
            elastic_angle_cdf: Vec::new(),
            elastic_angle_pdf: Vec::new(),
            elastic_angle_interp: Vec::new(),
            temperature_k_per_material: Vec::new(),
            km_n_energies: Vec::new(),
            km_ae_offset: Vec::new(),
            km_energy_grid: Vec::new(),
            km_interp: Vec::new(),
            km_n_discrete: Vec::new(),
            km_n_x: Vec::new(),
            km_x_offset: Vec::new(),
            km_x: Vec::new(),
            km_p: Vec::new(),
            km_c: Vec::new(),
            km_r: Vec::new(),
            km_a: Vec::new(),
            evap_n_energies: Vec::new(),
            evap_n_components: Vec::new(),
            evap_ae_offset: Vec::new(),
            evap_theta_offset: Vec::new(),
            evap_energy_grid: Vec::new(),
            evap_theta: Vec::new(),
            evap_u: Vec::new(),
            nbps_n_bodies: Vec::new(),
            nbps_total_mass: Vec::new(),
            maxwell_n_energies: Vec::new(),
            maxwell_ae_offset: Vec::new(),
            maxwell_energy_grid: Vec::new(),
            maxwell_theta: Vec::new(),
            maxwell_u: Vec::new(),
            watt_n_energies: Vec::new(),
            watt_ae_offset: Vec::new(),
            watt_energy_grid: Vec::new(),
            watt_a: Vec::new(),
            watt_b: Vec::new(),
            watt_u: Vec::new(),
            urr_meta: Vec::new(),
            urr_ae_offset: Vec::new(),
            urr_cdf_offset: Vec::new(),
            urr_energy_grid: Vec::new(),
            urr_cdf: Vec::new(),
            urr_xs: Vec::new(),
            urr_atom_density: Vec::new(),
            nuclide_select: NuclideSelectInputs::default(),
        });
    }

    let mut out = TranslatedMaterials {
        log_energy_grid: Vec::new(),
        coarse_log_energy_grid: Vec::new(),
        coarse_meta: Vec::new(),
        fine_log_energy_grid: Vec::new(),
        fine_meta: Vec::new(),
        xs_elastic_per_material: Vec::new(),
        xs_absorption_per_material: Vec::new(),
        xs_inelastic_per_material: Vec::new(),
        xs_inelastic_per_mt_sparse: Vec::new(),
        permt_meta: Vec::new(),
        xs_fission_per_material: Vec::new(),
        nu_bar_per_material: Vec::new(),
        beta_delayed_per_material: Vec::new(),
        fission_a_per_material: Vec::with_capacity(materials.len()),
        fission_b_per_material: Vec::with_capacity(materials.len()),
        fission_eout_kind_per_material: Vec::with_capacity(2 * materials.len()),
        fission_eout_n_energies_per_material: Vec::with_capacity(2 * materials.len()),
        fission_eout_ae_offset: Vec::with_capacity(2 * materials.len()),
        fission_eout_energy_grid_per_material: Vec::new(),
        fission_eout_n_x_per_material: Vec::new(),
        fission_eout_x_offset: Vec::new(),
        fission_eout_x_per_material: Vec::new(),
        fission_eout_cdf_per_material: Vec::new(),
        fission_eout_p_per_material: Vec::new(),
        fission_eout_interp_per_material: Vec::new(),
        target_mass_per_material: Vec::with_capacity(materials.len()),
        q_inelastic_per_mt: Vec::new(),
        yield_per_mt_sparse: Vec::new(),
        angle_n_energies: Vec::new(),
        angle_ae_offset: Vec::new(),
        angle_energy_grid: Vec::new(),
        angle_n_mu: Vec::new(),
        angle_mu_offset: Vec::new(),
        angle_mu: Vec::new(),
        angle_cdf: Vec::new(),
        angle_pdf: Vec::new(),
        angle_interp: Vec::new(),
        eout_kind: Vec::new(),
        eout_n_energies: Vec::new(),
        eout_ae_offset: Vec::new(),
        eout_histogram_interp: Vec::new(),
        eout_energy_grid: Vec::new(),
        eout_n_x: Vec::new(),
        eout_x_offset: Vec::new(),
        eout_x: Vec::new(),
        eout_p: Vec::new(),
        eout_cdf: Vec::new(),
        eout_interp: Vec::new(),
        eout_n_discrete: Vec::new(),
        corr_n_energies: Vec::new(),
        corr_n_components: Vec::new(),
        corr_ae_offset: Vec::new(),
        corr_energy_grid: Vec::new(),
        corr_n_x: Vec::new(),
        corr_x_offset: Vec::new(),
        corr_x: Vec::new(),
        corr_cdf: Vec::new(),
        corr_p: Vec::new(),
        corr_interp: Vec::new(),
        corr_n_discrete: Vec::new(),
        corr_n_mu: Vec::new(),
        corr_mu_offset: Vec::new(),
        corr_mu: Vec::new(),
        corr_mu_cdf: Vec::new(),
        corr_mu_pdf: Vec::new(),
        corr_mu_interp: Vec::new(),
        scatter_in_cm_per_mt: Vec::new(),
        elastic_angle_n_energies: Vec::with_capacity(materials.len()),
        elastic_angle_ae_offset: Vec::with_capacity(materials.len()),
        elastic_angle_energy_grid: Vec::new(),
        elastic_angle_n_mu: Vec::new(),
        elastic_angle_mu_offset: Vec::new(),
        elastic_angle_mu: Vec::new(),
        elastic_angle_cdf: Vec::new(),
        elastic_angle_pdf: Vec::new(),
        elastic_angle_interp: Vec::new(),
        temperature_k_per_material: Vec::with_capacity(materials.len()),
        km_n_energies: Vec::new(),
        km_ae_offset: Vec::new(),
        km_energy_grid: Vec::new(),
        km_interp: Vec::new(),
        km_n_discrete: Vec::new(),
        km_n_x: Vec::new(),
        km_x_offset: Vec::new(),
        km_x: Vec::new(),
        km_p: Vec::new(),
        km_c: Vec::new(),
        km_r: Vec::new(),
        km_a: Vec::new(),
        evap_n_energies: Vec::new(),
        evap_n_components: Vec::new(),
        evap_ae_offset: Vec::new(),
        evap_theta_offset: Vec::new(),
        evap_energy_grid: Vec::new(),
        evap_theta: Vec::new(),
        evap_u: Vec::new(),
        nbps_n_bodies: Vec::new(),
        nbps_total_mass: Vec::new(),
        maxwell_n_energies: Vec::new(),
        maxwell_ae_offset: Vec::new(),
        maxwell_energy_grid: Vec::new(),
        maxwell_theta: Vec::new(),
        maxwell_u: Vec::new(),
        watt_n_energies: Vec::new(),
        watt_ae_offset: Vec::new(),
        watt_energy_grid: Vec::new(),
        watt_a: Vec::new(),
        watt_b: Vec::new(),
        watt_u: Vec::new(),
        urr_meta: Vec::new(),
        urr_ae_offset: Vec::new(),
        urr_cdf_offset: Vec::new(),
        urr_energy_grid: Vec::new(),
        urr_cdf: Vec::new(),
        urr_xs: Vec::new(),
        urr_atom_density: Vec::new(),
        nuclide_select: NuclideSelectInputs::default(),
    };
    // FINE energy grid (issue #212). The GPU keeps every material's smooth macro
    // XS on ONE shared FINE grid, so that grid must be the UNION of every
    // material's union grid, NOT the first material's. Otherwise a material with
    // finer resonance structure than the first is resampled onto a coarser grid,
    // smearing its resonances; on a multi-material problem that biased the
    // epithermal flux by up to ~40% (PR #209 flagged it as "extra epithermal loss
    // for materials with finer grids"). The FINE grid (resonance-critical, backs
    // the per-nuclide totals / partials and hence Sigma_t) is the global union.
    //
    // The COARSE grid backing the memory-bounded 3D per-MT inelastic buffers is
    // NOT shared: each material carries its OWN coarse grid (its finest single
    // per-nuclide grid), concatenated tight below with a per-material
    // `coarse_meta` descriptor. Sharing one coarse grid smeared non-first
    // materials' inelastic thresholds (a ~2% flux deficit below 1 MeV); the full
    // union is exact but multi-GB. Per-material coarse grids close the gap while
    // keeping the buffers bounded to one material's grid. For a single-material
    // problem the global union equals that material's union, and its lone coarse
    // grid sits at base 0, so the whole path stays bit-identical.
    let mut global_fine: Vec<f64> = Vec::new();
    for material in materials {
        let atoms = match material.get_atoms_per_barn_cm() {
            Ok(a) => a,
            Err(_) => continue,
        };
        let mut weighted: Vec<(&yamc_nuclide::Nuclide, f64)> = Vec::new();
        for (name, density) in &atoms {
            if let Some(nuclide_arc) = material.nuclide_data.get(name) {
                weighted.push((nuclide_arc.as_ref(), *density));
            }
        }
        if weighted.is_empty() {
            continue;
        }
        if let Ok(u) = union_energy_grid(&weighted, material.temperature()) {
            global_fine.extend_from_slice(&u);
        }
    }
    global_fine.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    global_fine.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    if !global_fine.is_empty() {
        out.log_energy_grid = global_fine.iter().map(|e| e.ln()).collect();
    }
    // Per-collision nuclide-selection data (issue #74): for each material a
    // `(macro_total_rows, awr, partials)` triple, assembled into
    // `NuclideSelectInputs` after the loop once the master grid is fixed. Built
    // from the SAME master linear grid as the aggregate XS so the kernel's
    // `idx_lo/idx_hi/frac` align. `partials` is the packed per-nuclide
    // reaction-partial block (`[n_nuclides x n_grid x NUC_PARTIAL_COLS]`, #74
    // Stage 2b) driving the per-nuclide reaction-type split.
    let mut nuc_select_materials: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> =
        Vec::with_capacity(materials.len());
    // Running element base into the tight-CSR `nuc_macro_total` / `nuc_partial_xs`
    // for the NEXT material's first slab (issue #212). Advances by `nuc_count *
    // fine_n` per material; recorded in `fine_meta` col COL_FINE_NUC_BASE.
    let mut nuc_element_base: u32 = 0;

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
                return Err(GpuTranslateError::NoLoadedTemperature {
                    nuclide: name.clone(),
                    material: mat_name.clone(),
                });
            };
            weighted.push((nuclide_arc.as_ref(), *density));
        }

        // This material's OWN coarse grid (issue #212): its UNION grid (union of
        // its finest single per-nuclide grid, appended tight to the concatenated
        // coarse grid so no material's inelastic thresholds smear onto another's
        // grid (issue #212). Finest-single (not the material's union) keeps the
        // per-MT 3D buffers bounded (dual-grid, #88): the union would push the
        // concatenated per-MT buffer past the GPU's ~4 GB maxStorageBufferRange
        // on multi-material problems (silent bind failure -> zero flux), for only
        // a ~0.5% accuracy gain visible on adversarial multi-isotope materials.
        // Record the packed `coarse_meta` row (COL_COARSE_GRID_OFFSET / _N /
        // _MT_BASE) BEFORE appending: the grid base is the current concatenated
        // coarse length and the per-MT base is the current per-MT buffer length
        // (this material's first slab), which `append_per_nuclide_inelastic`
        // below fills in.
        let mat_coarse: Vec<f64> =
            union_energy_grid(&weighted, material.temperature()).map_err(|e| {
                GpuTranslateError::CellRegionUnsupported {
                    cell_id: None,
                    reason: format!("material `{mat_name}` coarse grid: {e}"),
                }
            })?;
        out.coarse_meta
            .push(out.coarse_log_energy_grid.len() as u32);
        out.coarse_meta.push(mat_coarse.len() as u32);
        // COL_COARSE_MT_BASE: this material's first-slab base into `permt_meta`
        // ROWS (issue #212). Equals `nuc_off * MT_INELASTIC_COUNT`, which is the
        // current permt_meta row count (prior slabs appended, this material's not
        // yet); `append_per_nuclide_inelastic` below fills the rows.
        out.coarse_meta
            .push((out.permt_meta.len() / PERMT_META_COLS as usize) as u32);
        out.coarse_log_energy_grid
            .extend(mat_coarse.iter().map(|e| e.ln()));

        // This material's OWN fine grid (issue #212): its UNION grid (union of
        // its nuclides' grids), appended tight to `fine_log_energy_grid` so no
        // material's resonance structure smears onto another's. Backs this
        // material's aggregate macro XS and per-(nuclide) totals / partials.
        // Record the packed `fine_meta` row (COL_FINE_GRID_OFFSET / _N /
        // _NUC_BASE) BEFORE appending: the grid base (== this material's
        // aggregate-XS row base) is the current concatenated fine length, and the
        // nuc base is the running per-slab element base into `nuc_macro_total`.
        // A single-material problem's union == the global union, so its lone fine
        // grid sits at base 0 and every index reduces to the old shared grid
        // (bit-identical).
        let mat_fine: Vec<f64> =
            union_energy_grid(&weighted, material.temperature()).map_err(|e| {
                GpuTranslateError::CellRegionUnsupported {
                    cell_id: None,
                    reason: format!("material `{mat_name}` fine grid: {e}"),
                }
            })?;
        out.fine_meta.push(out.fine_log_energy_grid.len() as u32);
        out.fine_meta.push(mat_fine.len() as u32);
        out.fine_meta.push(nuc_element_base);
        nuc_element_base += (weighted.len() * mat_fine.len()) as u32;
        out.fine_log_energy_grid
            .extend(mat_fine.iter().map(|e| e.ln()));

        let mat_xs = extract_material_xs(&weighted, material.temperature()).map_err(|e| {
            // Only the "missing reaction" error has structured info we
            // care about; other errors (no temp, no AWR, empty
            // material) get reported via display.
            match e {
                yamc_gpu::neutron::xs::NuclideXsError::MissingReaction { mt, .. } => {
                    GpuTranslateError::MissingReactionData {
                        material: mat_name.clone(),
                        mt,
                    }
                }
                yamc_gpu::neutron::xs::NuclideXsError::TemperatureNotLoaded(_) => {
                    GpuTranslateError::NoLoadedTemperature {
                        nuclide: "(any)".into(),
                        material: mat_name.clone(),
                    }
                }
                unslotted @ yamc_gpu::neutron::xs::NuclideXsError::UnslottedScatterMts {
                    ..
                } => GpuTranslateError::UnslottedScatterMts {
                    material: mat_name.clone(),
                    detail: unslotted.to_string(),
                },
                other => GpuTranslateError::CellRegionUnsupported {
                    cell_id: None,
                    reason: format!("material `{mat_name}`: {other}"),
                },
            }
        })?;

        // No resample (issue #212): `extract_material_xs` builds its aggregate
        // macro XS on `union_energy_grid(weighted)` == `mat_fine`, this material's
        // OWN fine grid, so `mat_xs.xs_*` already has length `mat_fine.len()`.
        // Appending it tight concatenates the per-material CSR the kernel indexes
        // via `fine_meta`. (Previously every material was resampled onto the
        // first material's grid, issue #208; per-material grids remove both the
        // resample and its smearing while cutting RAM.)
        debug_assert_eq!(
            mat_xs.log_energy_grid.len(),
            mat_fine.len(),
            "extract_material_xs must produce XS on the material's own union grid"
        );
        append_material_xs(&mut out, &mat_xs);

        // Per-collision nuclide-selection rows for this material, on THIS
        // material's OWN fine grid (`mat_fine`, linear, issue #212). Rows are
        // nuclide-major `[n_nuclides x fine_n]`; AWRs are per nuclide in the same
        // order as `weighted` (matching `extract_per_nuclide_macro_total_xs`).
        // The per-MT inelastic XS / Q / yield ride the per-material COARSE grid
        // (`mat_coarse`); the per-nuclide totals + four reaction partials ride the
        // per-material FINE grid, which is resonance-critical for the split.
        let per_nuc =
            extract_per_nuclide_macro_total_xs(&weighted, material.temperature(), &mat_fine)
                .map_err(|e| GpuTranslateError::CellRegionUnsupported {
                    cell_id: None,
                    reason: format!("material `{mat_name}` per-nuclide totals: {e}"),
                })?;
        let awr: Vec<f64> = weighted
            .iter()
            .map(|(nuc, _)| nuc.atomic_weight_ratio.unwrap_or(1.0))
            .collect();

        // Per-nuclide elastic (MT 2) angular pool, slab-major in the same
        // nuclide order as `weighted` (= the selection `chosen` order), so the
        // kernel indexes by the selected nuclide's global slab (#74 Stage 2a).
        let elastic_pool = extract_per_nuclide_elastic_angle(&weighted, material.temperature())
            .map_err(|e| GpuTranslateError::CellRegionUnsupported {
                cell_id: None,
                reason: format!("material `{mat_name}` per-nuclide elastic angle: {e}"),
            })?;
        append_elastic_angle_slabs(&mut out, &elastic_pool);

        // Per-(material, nuclide) inelastic distribution + reaction-partial pool
        // (#74 Stage 2b), slab-major in the same nuclide order. Appended to the
        // slab-keyed per-MT buffers, and the partials packed for the reaction
        // split. Built on THIS material's own fine (partials) / coarse (per-MT)
        // grids (issue #212).
        let inel_pool = extract_per_nuclide_inelastic(
            &weighted,
            material.temperature(),
            &mat_fine,
            &mat_coarse,
        )
        .map_err(|e| GpuTranslateError::CellRegionUnsupported {
            cell_id: None,
            reason: format!("material `{mat_name}` per-nuclide inelastic: {e}"),
        })?;
        append_per_nuclide_inelastic(&mut out, &inel_pool);
        let partials = pack_nuc_partials(&inel_pool, mat_fine.len());
        nuc_select_materials.push((per_nuc.macro_total_xs, awr, partials));
    }

    // Tight-CSR concatenation of the per-material nuc rows (issue #212): each
    // material's rows have length `nuc_count × fine_n[m]`, so the builder must
    // NOT assume a uniform grid; it derives each material's `fine_n` from its own
    // row block.
    out.nuclide_select =
        NuclideSelectInputs::from_materials_with_partials_per_material(&nuc_select_materials);

    Ok(out)
}

/// Append one material's extracted `GpuNuclideXs` onto the flat
/// per-material `TranslatedMaterials` buffers (called once per material
/// in grid order). Every output field is the concatenation of the
/// corresponding `GpuNuclideXs` field across materials.
///
/// The bulk of the fields share the same name on both sides, so the local
/// `ext!` macro lists them once as `out.f.extend(&x.f)`. The handful with
/// a different destination name, and the per-material scalars (`push`),
/// are spelled out explicitly below.
fn append_material_xs(out: &mut TranslatedMaterials, x: &GpuNuclideXs) {
    macro_rules! ext {
        ($($f:ident),* $(,)?) => {
            $( out.$f.extend(&x.$f); )*
        };
    }
    // Per-MATERIAL aggregate XS (leading dim n_materials) + per-(material,
    // nuclide) URR (leading dim n_slab, issue #210). The per-MT inelastic
    // distribution pools are NOT appended here: they are keyed per-(material,
    // nuclide) slab (#74 Stage 2b) and appended by `append_per_nuclide_inelastic`
    // from the per-nuclide pool. (For the void slot, `append_void_per_mt_slab`
    // appends one empty slab from this same aggregate.)
    //
    // Tight CSR bases (issue #104): the URR tables are keyed per SLAB now, so
    // push one `urr_ae_offset` / `urr_cdf_offset` base per nuclide slab, each
    // reflecting the concatenated length BEFORE that slab's rows are appended
    // (mirrors the `push_*_csr_offsets` helpers). The per-slab counts come from
    // this material's `urr_meta` rows. The `urr_xs` base reuses `urr_cdf_offset`
    // (xs is the cdf grid × URR_XS_COLS), so no separate offset is stored.
    let n_urr_slabs = x.urr_meta.len() / URR_META_COLS;
    let mut urr_eg_base = out.urr_energy_grid.len() as u32;
    let mut urr_cdf_base = out.urr_cdf.len() as u32;
    for s in 0..n_urr_slabs {
        out.urr_ae_offset.push(urr_eg_base);
        out.urr_cdf_offset.push(urr_cdf_base);
        let n_e = x.urr_meta[s * URR_META_COLS + URR_META_N_ENERGIES];
        let n_cdf = x.urr_meta[s * URR_META_COLS + URR_META_N_CDF];
        urr_eg_base += n_e;
        urr_cdf_base += n_e * n_cdf;
    }
    ext!(urr_meta, urr_energy_grid, urr_cdf, urr_xs, urr_atom_density,);

    // Renamed vector fields (destination name differs from the source).
    out.xs_elastic_per_material.extend(&x.xs_elastic);
    out.xs_absorption_per_material.extend(&x.xs_absorption);
    out.xs_inelastic_per_material.extend(&x.xs_inelastic);
    out.xs_fission_per_material.extend(&x.xs_fission);
    out.nu_bar_per_material.extend(&x.nu_bar);
    out.beta_delayed_per_material.extend(&x.beta_delayed);

    // Fission outgoing-energy chi, tight CSR (issue #104). Record this
    // material's bases BEFORE extending the tight data arrays, so they
    // reflect the prior length (mirrors `push_eout_csr_offsets`, but
    // per-MATERIAL: fission eout is appended once per material here, not
    // per nuclide). `fission_eout_ae_offset[mat]` is the material's first
    // ae-row in the concatenated `fission_eout_n_x` / `energy_grid`;
    // `fission_eout_x_offset` gets one entry per appended ae-row, the
    // global start of that row's `(x, cdf)` points.
    //
    // TWO chi rows per material (issue #364), pushed back to back: the prompt
    // spectrum at row `2*mat` and the delayed groups' folded spectrum at row
    // `2*mat + 1`. Appending the delayed spectrum as an extra row rather than a
    // parallel set of buffers keeps the kernel's storage-buffer count unchanged.
    push_fission_eout_row(
        out,
        &x.fission_eout_energy_grid,
        &x.fission_eout_n_x,
        &x.fission_eout_interp,
        &x.fission_eout_x,
        &x.fission_eout_cdf,
        &x.fission_eout_p,
        x.fission_eout_kind,
        x.fission_eout_n_energies,
    );
    push_fission_eout_row(
        out,
        &x.fission_eout_delayed_energy_grid,
        &x.fission_eout_delayed_n_x,
        &x.fission_eout_delayed_interp,
        &x.fission_eout_delayed_x,
        &x.fission_eout_delayed_cdf,
        &x.fission_eout_delayed_p,
        x.fission_eout_delayed_kind,
        x.fission_eout_delayed_n_energies,
    );

    // Per-material scalars (one element pushed per material).
    out.fission_a_per_material.push(x.fission_watt_a);
    out.fission_b_per_material.push(x.fission_watt_b);
    out.target_mass_per_material.push(x.target_mass);
    out.temperature_k_per_material.push(x.temperature_k);
}

/// Append ONE fission chi row onto the tight per-row CSR buffers (issue #104,
/// extended to two rows per material by issue #364). Records the row's bases
/// BEFORE extending the data arrays, so they reflect the prior length (mirrors
/// `push_eout_csr_offsets`): `fission_eout_ae_offset` gets the row's first ae-row
/// in the concatenated `fission_eout_n_x` / `energy_grid`, and
/// `fission_eout_x_offset` one entry per appended ae-row, the global start of that
/// ae-row's `(x, cdf)` points. An empty row (a material with no delayed data)
/// pushes a base with zero rows behind it, which the kernel never reads because
/// `beta == 0` there.
#[allow(clippy::too_many_arguments)]
fn push_fission_eout_row(
    out: &mut TranslatedMaterials,
    energy_grid: &[f64],
    n_x: &[u32],
    interp: &[u32],
    x_vals: &[f64],
    cdf: &[f64],
    p: &[f64],
    kind: u32,
    n_energies: u32,
) {
    out.fission_eout_ae_offset
        .push(out.fission_eout_n_x_per_material.len() as u32);
    let mut x_base = out.fission_eout_x_per_material.len() as u32;
    for &n in n_x {
        out.fission_eout_x_offset.push(x_base);
        x_base += n;
    }
    out.fission_eout_energy_grid_per_material
        .extend(energy_grid);
    out.fission_eout_n_x_per_material.extend(n_x);
    out.fission_eout_interp_per_material.extend(interp);
    out.fission_eout_x_per_material.extend(x_vals);
    out.fission_eout_cdf_per_material.extend(cdf);
    out.fission_eout_p_per_material.extend(p);
    out.fission_eout_kind_per_material.push(kind);
    out.fission_eout_n_energies_per_material.push(n_energies);
}

/// Append a material's per-(material, nuclide) inelastic distribution pool onto
/// the slab-major per-MT buffers (#74 Stage 2b). One block per nuclide,
/// concatenated in `mat_nuclide_meta` slab order so the kernel indexes by the
/// struck nuclide's global `slab`. A single-nuclide material appends exactly one
/// slab, byte-identical to the material-blended per-MT data the pre-Stage-2b
/// `append_material_xs` produced.
/// Build the per-MT inelastic angular CSR offsets (issue #104) for one slab's
/// worth of MT slots being appended: `angle_ae_offset` (per (slab,MT) slot, the
/// global ae-row start) and `angle_mu_offset` (per ae-row, the global mu-point
/// start), computed from the tight `n_energies` / `n_mu` counts. Must be called
/// BEFORE the tight data arrays are extended, so the bases reflect prior length.
/// Build the SPARSE per-MT `permt_meta` rows (issue #212) for one slab's worth
/// of MT slots being appended: one `[value_offset, i_start, n_stored]` row per
/// (slab, MT slot). `value_offset` is the global base into
/// `xs_inelastic_per_mt_sparse` / `yield_per_mt_sparse` (both share it), derived
/// from the current sparse-buffer length plus the running per-slot cumulative.
/// Must be called BEFORE the sparse value arrays are extended, so the bases
/// reflect the prior length (mirrors the `push_*_csr_offsets` helpers).
/// `i_start` / `n_stored` come one per (slab, MT slot) from the extracted pool.
fn push_permt_meta(out: &mut TranslatedMaterials, i_start: &[u32], n_stored: &[u32]) {
    let value_base = out.xs_inelastic_per_mt_sparse.len() as u32;
    let mut local = 0u32;
    for (&i0, &n) in i_start.iter().zip(n_stored) {
        out.permt_meta.push(value_base + local);
        out.permt_meta.push(i0);
        out.permt_meta.push(n);
        local += n;
    }
}

fn push_angle_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32], n_mu: &[u32]) {
    let ae_base = out.angle_n_mu.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.angle_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let mu_base = out.angle_mu.len() as u32;
    let mut local_mu = 0u32;
    for &n_m in n_mu {
        out.angle_mu_offset.push(mu_base + local_mu);
        local_mu += n_m;
    }
}

/// Build the per-MT outgoing-energy CSR offsets (issue #104) for one slab's
/// worth of MT slots being appended: `eout_ae_offset` (per (slab,MT) slot, the
/// global ae-row start) and `eout_x_offset` (per ae-row, the global x-point
/// start), computed from the tight `eout_n_energies` / `eout_n_x` counts. Must
/// be called BEFORE the tight data arrays are extended, so the bases reflect
/// the prior length. `n_energies` is one entry per MT slot (the slot's ae-row
/// count); `n_x` is one entry per appended ae-row (the row's x-point count).
fn push_eout_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32], n_x: &[u32]) {
    let ae_base = out.eout_n_x.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.eout_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let x_base = out.eout_x.len() as u32;
    let mut local_x = 0u32;
    for &n in n_x {
        out.eout_x_offset.push(x_base + local_x);
        local_x += n;
    }
}

/// Build the per-MT correlated angle-energy CSR offsets (issue #104) for one
/// slab's worth of MT slots being appended, three nesting levels:
/// `corr_ae_offset` (per (slab,MT) slot, the global ae-row start),
/// `corr_x_offset` (per ae-row, the global E_out x-point start) and
/// `corr_mu_offset` (per x-point, the global mu-point start), computed from the
/// tight `corr_n_energies` / `corr_n_x` / `corr_n_mu` counts. Must be called
/// BEFORE the tight data arrays are extended, so the bases reflect the prior
/// length. `n_energies` is one entry per MT slot (the slot's ae-row count);
/// `n_x` is one entry per appended ae-row (its E_out x-point count); `n_mu` is
/// one entry per appended x-point (its mu-point count).
fn push_corr_csr_offsets(
    out: &mut TranslatedMaterials,
    n_energies: &[u32],
    n_x: &[u32],
    n_mu: &[u32],
) {
    let ae_base = out.corr_n_x.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.corr_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let x_base = out.corr_x.len() as u32;
    let mut local_x = 0u32;
    for &n in n_x {
        out.corr_x_offset.push(x_base + local_x);
        local_x += n;
    }
    let mu_base = out.corr_mu.len() as u32;
    let mut local_mu = 0u32;
    for &n in n_mu {
        out.corr_mu_offset.push(mu_base + local_mu);
        local_mu += n;
    }
}

/// Build the per-MT Kalbach-Mann CSR offsets (issue #104) for one slab's worth
/// of MT slots being appended: `km_ae_offset` (per (slab,MT) slot, the global
/// ae-row start) and `km_x_offset` (per ae-row, the global x-point start),
/// computed from the tight `km_n_energies` / `km_n_x` counts. Kalbach-Mann's mu
/// is closed-form (no mu sub-table), so there are only two CSR levels. Must be
/// called BEFORE the tight data arrays are extended, so the bases reflect the
/// prior length. `n_energies` is one entry per MT slot (the slot's ae-row
/// count); `n_x` is one entry per appended ae-row (the row's x-point count).
fn push_km_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32], n_x: &[u32]) {
    let ae_base = out.km_n_x.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.km_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let x_base = out.km_x.len() as u32;
    let mut local_x = 0u32;
    for &n in n_x {
        out.km_x_offset.push(x_base + local_x);
        local_x += n;
    }
}

/// Build the per-MT Evaporation CSR offsets (issue #104) for one slab's worth of
/// MT slots being appended: `evap_ae_offset` (per (slab,MT) slot, the global
/// E_in-row start into the tight `evap_energy_grid` / `evap_u`) and
/// `evap_theta_offset` (per (slab,MT) slot, the global start into the
/// component-major `evap_theta`, where component `c`'s row begins at
/// `evap_theta_offset[slot] + c * n_energies[slot]`). Must be called BEFORE the
/// tight data arrays are extended, so the bases reflect the prior length.
/// `n_energies` / `n_components` are one entry per MT slot.
fn push_evap_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32], n_components: &[u32]) {
    let ae_base = out.evap_energy_grid.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.evap_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let theta_base = out.evap_theta.len() as u32;
    let mut local_theta = 0u32;
    for (&n_e, &n_c) in n_energies.iter().zip(n_components.iter()) {
        out.evap_theta_offset.push(theta_base + local_theta);
        local_theta += n_c * n_e;
    }
}

/// Build the per-MT Maxwell CSR offset (issue #104) for one slab's worth of MT
/// slots being appended: `maxwell_ae_offset` (per (slab,MT) slot, the global
/// E_in-row start into the tight `maxwell_energy_grid` / `maxwell_theta`),
/// computed from the tight `maxwell_n_energies` counts. Must be called BEFORE
/// the tight data arrays are extended, so the base reflects the prior length.
fn push_maxwell_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32]) {
    let ae_base = out.maxwell_energy_grid.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.maxwell_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
}

/// Build the per-MT Watt CSR offset (issue #104) for one slab's worth of MT
/// slots being appended: `watt_ae_offset` (per (slab,MT) slot, the global
/// E_in-row start into the tight `watt_energy_grid` / `watt_a` / `watt_b`),
/// computed from the tight `watt_n_energies` counts. Must be called BEFORE the
/// tight data arrays are extended, so the base reflects the prior length.
fn push_watt_csr_offsets(out: &mut TranslatedMaterials, n_energies: &[u32]) {
    let ae_base = out.watt_energy_grid.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in n_energies {
        out.watt_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
}

fn append_per_nuclide_inelastic(out: &mut TranslatedMaterials, p: &PerNuclideInelastic) {
    push_permt_meta(out, &p.permt_i_start, &p.permt_n_stored);
    out.xs_inelastic_per_mt_sparse
        .extend(&p.xs_inelastic_per_mt_sparse);
    out.yield_per_mt_sparse.extend(&p.yield_per_mt_sparse);
    push_angle_csr_offsets(out, &p.angle_n_energies, &p.angle_n_mu);
    push_eout_csr_offsets(out, &p.eout_n_energies, &p.eout_n_x);
    push_corr_csr_offsets(out, &p.corr_n_energies, &p.corr_n_x, &p.corr_n_mu);
    push_km_csr_offsets(out, &p.km_n_energies, &p.km_n_x);
    push_evap_csr_offsets(out, &p.evap_n_energies, &p.evap_n_components);
    push_maxwell_csr_offsets(out, &p.maxwell_n_energies);
    push_watt_csr_offsets(out, &p.watt_n_energies);
    macro_rules! ext {
        ($($f:ident),* $(,)?) => {
            $( out.$f.extend(&p.$f); )*
        };
    }
    ext!(
        q_inelastic_per_mt,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_histogram_interp,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_p,
        eout_cdf,
        eout_interp,
        eout_n_discrete,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        evap_n_energies,
        evap_n_components,
        evap_energy_grid,
        evap_theta,
        evap_u,
        nbps_n_bodies,
        nbps_total_mass,
        maxwell_n_energies,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
    );
    // Renamed: pool field `scatter_in_cm` -> out `scatter_in_cm_per_mt`.
    out.scatter_in_cm_per_mt.extend(&p.scatter_in_cm);
}

/// Pack a material's per-(nuclide, energy) reaction partials into the flat
/// `[n_nuclides x n_grid x NUC_PARTIAL_COLS]` block the kernel's reaction split
/// reads, column order elastic / absorption / inelastic / fission. The pool's
/// `sigma_*` are stored per-nuclide-then-per-energy (`slab * n_grid + i`); this
/// re-lays them column-interleaved per `(slab, energy)`.
fn pack_nuc_partials(p: &PerNuclideInelastic, n_grid: usize) -> Vec<f64> {
    let rows = p.n_nuclides * n_grid;
    let mut packed = vec![0.0_f64; rows * NUC_PARTIAL_COLS];
    for r in 0..rows {
        let off = r * NUC_PARTIAL_COLS;
        packed[off + NUC_PARTIAL_ELASTIC] = p.sigma_elastic[r];
        packed[off + NUC_PARTIAL_ABSORPTION] = p.sigma_absorption[r];
        packed[off + NUC_PARTIAL_INELASTIC] = p.sigma_inelastic[r];
        packed[off + NUC_PARTIAL_FISSION] = p.sigma_fission[r];
    }
    packed
}

/// Append one empty per-MT slab for the synthetic void material (#74 Stage 2b).
/// The void's single nuclide never collides, so the data is never read; this
/// just keeps the slab-major per-MT buffer lengths aligned with the slab table.
/// Reuses the void `GpuNuclideXs`'s all-empty per-MT fields (one slab's worth).
fn append_void_per_mt_slab(out: &mut TranslatedMaterials, x: &GpuNuclideXs) {
    // Void slab: one empty (n_stored == 0) `permt_meta` row per MT slot; no
    // sparse XS / yield values (issue #212). `q_inelastic_per_mt` on the void
    // `GpuNuclideXs` is one slab wide (MT_INELASTIC_COUNT), giving the slot count.
    let n_slots = x.q_inelastic_per_mt.len();
    let zeros = vec![0u32; n_slots];
    push_permt_meta(out, &zeros, &zeros);
    push_angle_csr_offsets(out, &x.angle_n_energies, &x.angle_n_mu);
    push_eout_csr_offsets(out, &x.eout_n_energies, &x.eout_n_x);
    push_corr_csr_offsets(out, &x.corr_n_energies, &x.corr_n_x, &x.corr_n_mu);
    push_km_csr_offsets(out, &x.km_n_energies, &x.km_n_x);
    push_evap_csr_offsets(out, &x.evap_n_energies, &x.evap_n_components);
    push_maxwell_csr_offsets(out, &x.maxwell_n_energies);
    push_watt_csr_offsets(out, &x.watt_n_energies);
    macro_rules! ext {
        ($($f:ident),* $(,)?) => {
            $( out.$f.extend(&x.$f); )*
        };
    }
    ext!(
        q_inelastic_per_mt,
        angle_n_energies,
        angle_energy_grid,
        angle_n_mu,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_histogram_interp,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_p,
        eout_cdf,
        eout_interp,
        eout_n_discrete,
        corr_n_energies,
        corr_n_components,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
        km_n_energies,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        evap_n_energies,
        evap_n_components,
        evap_energy_grid,
        evap_theta,
        evap_u,
        nbps_n_bodies,
        nbps_total_mass,
        maxwell_n_energies,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
    );
    out.scatter_in_cm_per_mt.extend(&x.scatter_in_cm);
}

/// Append a material's per-nuclide elastic-angle pool onto the slab-major
/// `out.elastic_angle_*` buffers (issue #74, Stage 2a). One slab row per
/// nuclide, concatenated in `mat_nuclide_meta` slab order so the kernel can
/// index by the selected nuclide's global `slab`.
fn append_elastic_angle_slabs(out: &mut TranslatedMaterials, pool: &PerNuclideElasticAngle) {
    // Build the global CSR bases for this material's slabs/rows (issue #104)
    // BEFORE extending the tight data arrays. `ae_offset[slab]` is the global
    // ae-row where the slab starts; `mu_offset[ae]` the global mu-point where
    // the row's (mu, cdf, pdf) start. Lengths come from `n_energies` / `n_mu`.
    let ae_base = out.elastic_angle_n_mu.len() as u32;
    let mut local_ae = 0u32;
    for &n_e in &pool.n_energies {
        out.elastic_angle_ae_offset.push(ae_base + local_ae);
        local_ae += n_e;
    }
    let mu_base = out.elastic_angle_mu.len() as u32;
    let mut local_mu = 0u32;
    for &n_m in &pool.n_mu {
        out.elastic_angle_mu_offset.push(mu_base + local_mu);
        local_mu += n_m;
    }
    out.elastic_angle_n_energies.extend(&pool.n_energies);
    out.elastic_angle_energy_grid.extend(&pool.energy_grid);
    out.elastic_angle_n_mu.extend(&pool.n_mu);
    out.elastic_angle_mu.extend(&pool.mu);
    out.elastic_angle_cdf.extend(&pool.cdf);
    out.elastic_angle_pdf.extend(&pool.pdf);
    out.elastic_angle_interp.extend(&pool.interp);
}

/// Append one empty elastic-angle slab (`n_energies == 0`, kernel falls back to
/// isotropic). Used for the synthetic void material slot, whose single nuclide
/// never collides. Tight layout: contributes no ae-rows / mu-points.
fn push_empty_elastic_angle_slab(out: &mut TranslatedMaterials) {
    out.elastic_angle_n_energies.push(0);
    // ae_offset base = current ae-row count; the slab spans an empty range.
    out.elastic_angle_ae_offset
        .push(out.elastic_angle_n_mu.len() as u32);
}

/// Sample `n_particles` initial states from the model's sources for
/// batch 0. Convenience wrapper around `sample_initial_particles_for_batch`.
fn sample_initial_particles(
    model: &Model,
    n_particles: usize,
    base_seed: u64,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    sample_initial_particles_for_batch(model, n_particles, 0, base_seed)
}

/// Sample initial states for batch `batch_idx`. Each batch gets its
/// own ChaCha8 stream (mixing `batch_idx` into the base seed) and its
/// own per-particle PCG seed band, so multi-batch GPU runs produce
/// independent realisations for the variance estimator. Determinism
/// is preserved: same `(base_seed, batch_idx, particle_idx)` always
/// gives the same particle stream.
///
/// The per-history PCG seed is
/// [`yamc_rng::history_seed`]`(base_seed, global_index)`, the
/// same call the CPU transport loop makes, so a history is on the same stream
/// on both backends (issue #40) and the base seed reaches the collision
/// physics (issue #315).
pub fn sample_initial_particles_for_batch(
    model: &Model,
    n_particles: usize,
    batch_idx: usize,
    base_seed: u64,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Mix batch index into the ChaCha seed via a large odd multiplier
    // (golden-ratio constant) so neighbouring batches get well-separated
    // streams. Wrapping is fine -- we just need different seeds.
    let batch_seed = base_seed.wrapping_add((batch_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = ChaCha8Rng::seed_from_u64(batch_seed);

    let mut seeds = Vec::with_capacity(n_particles);
    let mut energies = Vec::with_capacity(n_particles);
    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut directions = Vec::with_capacity(n_particles * 3);

    // Per-particle PCG seed band: offset by `batch_idx * n_particles` so
    // PCG streams don't overlap across batches. The seed itself comes from
    // the shared `history_seed`, the single definition the CPU transport
    // loop uses too.
    let seed_offset = (batch_idx as u64).wrapping_mul(n_particles as u64);
    let selector = SourceSelector::new(&model.sources);
    for i in 0..n_particles {
        let p = model.sample_source_with(&selector, &mut rng);
        let s = history_seed(base_seed, seed_offset.wrapping_add(i as u64));
        seeds.push(s);
        energies.push(p.energy);
        positions.extend_from_slice(&p.position);
        directions.extend_from_slice(&p.direction);
    }

    (seeds, energies, positions, directions)
}

/// Sample initial states for a FIXED-size launch chunk of the batch-free
/// per-history-variance neutron path (issue #233). Unlike
/// [`sample_initial_particles_for_batch`], the per-particle PCG seed band is
/// keyed off the GLOBAL history index `launch_idx * chunk_size + i` (via
/// `seed_offset = launch_idx * chunk_size`), NOT `launch_idx * n_particles`.
/// Because `chunk_size` is total-independent (the TDR-safe launch size, not
/// derived from `total_particles`), the seed for global history `h` is the
/// same regardless of how many histories the run requests -- so GPU results
/// are invariant to `total_particles` (removing it from the RNG key, #230
/// task 1). The last chunk of a run may transport fewer than `chunk_size`
/// histories (`n_particles < chunk_size`); the offset still uses the full
/// `chunk_size` stride so the global index stays contiguous.
///
/// The seed itself is `history_seed(base_seed, global_index)`, the shared
/// definition the CPU transport loop uses, so the two backends put the same
/// history on the same stream (the matched-stream equivalence contract).
pub fn sample_initial_particles_for_chunk(
    model: &Model,
    n_particles: usize,
    launch_idx: usize,
    chunk_size: usize,
    base_seed: u64,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Same per-chunk ChaCha decorrelation as the batch path: a fixed chunk
    // size makes the stream for chunk `launch_idx` total-independent.
    let chunk_seed =
        base_seed.wrapping_add((launch_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed);

    let mut seeds = Vec::with_capacity(n_particles);
    let mut energies = Vec::with_capacity(n_particles);
    let mut positions = Vec::with_capacity(n_particles * 3);
    let mut directions = Vec::with_capacity(n_particles * 3);

    // Global-history-index seed band: offset by `launch_idx * chunk_size`.
    let seed_offset = (launch_idx as u64).wrapping_mul(chunk_size as u64);
    let selector = SourceSelector::new(&model.sources);
    for i in 0..n_particles {
        let p = model.sample_source_with(&selector, &mut rng);
        let s = history_seed(base_seed, seed_offset.wrapping_add(i as u64));
        seeds.push(s);
        energies.push(p.energy);
        positions.extend_from_slice(&p.position);
        directions.extend_from_slice(&p.direction);
    }

    (seeds, energies, positions, directions)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    //! Validation-only tests. Exercising the full `translate_for_gpu`
    //! end-to-end requires real ENDF arrow data (loaded via
    //! `crate::Config::set_cross_section`) and is left to the dispatch
    //! PR (step 5), where it's natural to assert against the CPU path.
    //! The tests below build deliberately invalid `Model`s and check
    //! that the right rejection variant fires.

    use super::*;
    use crate::geometry::Geometry;
    use std::sync::Arc;

    use yamc_source::distribution::angular::AngularDistribution;
    use yamc_source::distribution::energy::Discrete;
    use yamc_source::source::{
        ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
    };
    use yamc_tallies::tally::Tally;

    #[test]
    fn append_material_xs_concatenates_every_field_kind() {
        // One field from each handling category that `append_material_xs`
        // still owns: an aligned vector (`urr_xs`, extended via the `ext!`
        // macro), a renamed vector (`xs_elastic` -> `xs_elastic_per_material`),
        // and two per-material scalars (`target_mass`, `fission_watt_a` ->
        // `fission_a_per_material`). (The per-MT inelastic distributions like
        // `angle_mu` are no longer appended here -- they moved to per-(material,
        // nuclide) slabs in #74 Stage 2b, appended by `append_per_nuclide_inelastic`.)
        //
        // URR is keyed per SLAB now (issue #210): each material carries one
        // `urr_meta` row per nuclide, and the per-slab CSR bases come from those
        // rows' `N_ENERGIES` / `N_CDF`. `a` has one URR slab (2 energy points, 2
        // cdf bands => 4 cdf cells), `b` one URR slab (1 energy point, 2 bands).
        let mut meta_a = vec![0u32; URR_META_COLS];
        meta_a[URR_META_N_ENERGIES] = 2;
        meta_a[URR_META_N_CDF] = 2;
        let mut meta_b = vec![0u32; URR_META_COLS];
        meta_b[URR_META_N_ENERGIES] = 1;
        meta_b[URR_META_N_CDF] = 2;
        let a = GpuNuclideXs {
            xs_elastic: vec![1.0, 2.0],
            urr_meta: meta_a,
            urr_energy_grid: vec![1.0, 2.0],
            urr_cdf: vec![0.1, 0.2, 0.3, 0.4],
            urr_xs: vec![0.5],
            urr_atom_density: vec![7.0],
            target_mass: 1.5,
            fission_watt_a: 2.0,
            ..Default::default()
        };
        let b = GpuNuclideXs {
            xs_elastic: vec![3.0],
            urr_meta: meta_b,
            urr_energy_grid: vec![3.0],
            urr_cdf: vec![0.5, 0.6],
            urr_xs: vec![0.1, 0.2, 0.3],
            urr_atom_density: vec![8.0],
            target_mass: 9.0,
            fission_watt_a: 4.0,
            ..Default::default()
        };

        let mut out = TranslatedMaterials::default();
        append_material_xs(&mut out, &a);
        append_material_xs(&mut out, &b);

        // Vectors concatenate in material (and within a material, slab) order;
        // their lengths sum.
        assert_eq!(out.xs_elastic_per_material, vec![1.0, 2.0, 3.0]);
        assert_eq!(out.urr_xs, vec![0.5, 0.1, 0.2, 0.3]);
        assert_eq!(out.urr_atom_density, vec![7.0, 8.0]);
        // Tight CSR bases (issue #104), now one per slab (issue #210): each
        // slab's base is the prior concatenated length. `a`'s slab starts at 0
        // with 2 energy points / 4 cdf cells, so `b`'s slab starts at 2 / 4.
        assert_eq!(out.urr_ae_offset, vec![0, 2]);
        assert_eq!(out.urr_cdf_offset, vec![0, 4]);
        // Scalars: exactly one element pushed per material.
        assert_eq!(out.target_mass_per_material, vec![1.5, 9.0]);
        assert_eq!(out.fission_a_per_material, vec![2.0, 4.0]);
    }

    #[test]
    fn encode_surface_covers_every_csg_kind() {
        // Each CSG surface kind must encode to its discriminant and a
        // stride-10 param slot (unused tail zero). No kind is rejected.
        let cases: Vec<(Surface, u32, [f64; 10])> = vec![
            (
                Surface::new_sphere(1.0, 2.0, 3.0, 4.0, None, None),
                SURFACE_SPHERE,
                [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_plane(1.0, 0.0, 0.0, 5.0, None, None),
                SURFACE_PLANE,
                [1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_cylinder([0.0, 0.0, 1.0], [1.0, 2.0, 3.0], 0.5, None, None),
                SURFACE_CYLINDER,
                [1.0, 2.0, 3.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_ztorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None),
                SURFACE_ZTORUS,
                [1.0, 2.0, 3.0, 4.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_xtorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None),
                SURFACE_XTORUS,
                [1.0, 2.0, 3.0, 4.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_ytorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None),
                SURFACE_YTORUS,
                [1.0, 2.0, 3.0, 4.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                Surface::new_quadric(
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, None, None,
                ),
                SURFACE_QUADRIC,
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ),
            (
                Surface::z_cone(1.0, 2.0, 3.0, 0.25, None, None),
                SURFACE_CONE,
                [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0],
            ),
        ];
        for (surface, want_type, want_params) in cases {
            let (kind, params) = encode_surface(&surface.kind);
            assert_eq!(kind, want_type, "type mismatch for {:?}", surface.kind);
            assert_eq!(
                params, want_params,
                "params mismatch for {:?}",
                surface.kind
            );
            assert_eq!(params.len(), SURFACE_PARAM_STRIDE);
        }
    }

    fn point_neutron_source() -> ParticleSource {
        ParticleSource::Neutron(Source {
            space: SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1.0e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        })
    }

    fn point_photon_source() -> ParticleSource {
        ParticleSource::Photon(Source {
            space: SourceSpatialDistribution::Point(
                yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
            ),
            angle: AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![1.0e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        })
    }

    fn empty_model(sources: Vec<ParticleSource>) -> Model {
        let geometry = Geometry::new(Vec::new(), Vec::new()).expect("empty geometry");
        let tallies: Vec<Arc<Tally>> = Vec::new();
        Model::new(geometry, sources, tallies)
    }

    #[test]
    fn rejects_zero_particles() {
        let model = empty_model(vec![point_neutron_source()]);
        assert_eq!(
            translate_for_gpu(&model, 0, 1).unwrap_err(),
            GpuTranslateError::NoParticles
        );
    }

    #[test]
    fn rejects_no_sources() {
        let model = empty_model(Vec::new());
        assert_eq!(
            translate_for_gpu(&model, 1, 1).unwrap_err(),
            GpuTranslateError::NoSources
        );
    }

    #[test]
    fn accepts_transport_secondary_photons() {
        // As of S6 coupled secondary-photon production is supported on the
        // neutron-translate path: the neutron kernel banks photons that the
        // coupled dispatch transports through the photon kernel. A neutron
        // source with the flag set must therefore translate cleanly (a photon
        // SOURCE is still rejected -- see `rejects_photon_source`).
        let mut model = empty_model(vec![point_neutron_source()]);
        model.transport_secondary_photons = true;
        let inputs = translate_for_gpu(&model, 1, 1).expect("coupled neutron translation succeeds");
        assert_eq!(inputs.seeds.len(), 1);
    }

    #[test]
    fn accepts_use_decay_photons() {
        // D1S is supported on GPU via the coupled dispatch (decay-photon
        // emission + parent-nuclide tally binning). The neutron-translate path
        // no longer rejects it; the full decay tables are built later in
        // `run_on_gpu_coupled`.
        let mut model = empty_model(vec![point_neutron_source()]);
        model.transport_secondary_photons = true;
        model.use_decay_photons = true;
        let inputs = translate_for_gpu(&model, 1, 1).expect("D1S neutron translation succeeds");
        assert_eq!(inputs.seeds.len(), 1);
    }

    #[test]
    fn rejects_photon_source() {
        let model = empty_model(vec![point_photon_source()]);
        let err = translate_for_gpu(&model, 1, 1).unwrap_err();
        // A photon source is rejected on GPU. It enables photon transport, so
        // the photon-source guard fires first (PhotonSourceOnNeutronPath);
        // NonNeutronSource is also an acceptable rejection of the same intent.
        assert!(
            matches!(
                err,
                GpuTranslateError::PhotonSourceOnNeutronPath
                    | GpuTranslateError::NonNeutronSource { .. }
            ),
            "expected a photon-source rejection, got {err:?}"
        );
    }

    #[test]
    fn empty_geometry_with_neutron_source_succeeds() {
        // No cells / no surfaces / no materials -- the kernel will see
        // empty buffers but the translation itself shouldn't error.
        // A degenerate but legal output.
        let model = empty_model(vec![point_neutron_source()]);
        let inputs = translate_for_gpu(&model, 4, 1).expect("translation succeeds");
        assert_eq!(inputs.seeds.len(), 4);
        assert_eq!(inputs.energies.len(), 4);
        assert_eq!(inputs.positions.len(), 12);
        assert_eq!(inputs.directions.len(), 12);
        assert!(inputs.cell_aabbs.is_empty());
        assert!(inputs.cell_to_material.is_empty());
        assert!(inputs.surface_types.is_empty());
        assert!(inputs.surface_params.is_empty());
        assert!(inputs.target_mass_per_material.is_empty());
        // Every initial direction should be a unit vector -- Isotropic
        // sampling is the contract.
        for chunk in inputs.directions.as_chunks::<3>().0 {
            let mag2 = chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2];
            assert!(
                (mag2 - 1.0).abs() < 1e-9,
                "direction not unit: |d|^2 = {mag2}"
            );
        }
    }

    use yamc_geo::region::{HalfspaceType, Region};

    fn unit_sphere() -> Arc<Surface> {
        Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 1.0, Some(1), None))
    }

    #[test]
    fn translate_cells_emits_empty_sentinel_for_vacuum_implicit_complement() {
        // A bounded material cell (inside the sphere) plus a void,
        // unbounded implicit-complement cell (outside the sphere). The
        // implicit complement must translate to an EMPTY sentinel AABB
        // (lower = +inf, upper = -inf) so the kernel never matches it,
        // and its material slot is a placeholder that is never read.
        let sphere = unit_sphere();
        let inside = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&sphere)));
        let outside = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&sphere)));
        let cells = vec![
            Cell::new(Some(1), inside, None, Some(0)),
            Cell::new(Some(2), outside, None, None), // void implicit complement
        ];

        let (aabbs, cell_to_material, _has_void) =
            translate_cells(&cells, 1).expect("vacuum IC must translate");
        assert_eq!(aabbs.len(), 12, "two cells, stride 6");
        assert_eq!(cell_to_material.len(), 2);

        // Cell 0: finite sphere AABB.
        assert_eq!(&aabbs[0..3], &[-1.0, -1.0, -1.0]);
        assert_eq!(&aabbs[3..6], &[1.0, 1.0, 1.0]);
        assert_eq!(cell_to_material[0], 0);

        // Cell 1 (implicit complement): empty sentinel -- lower = +inf,
        // upper = -inf. The kernel's `px >= min && px <= max` is then
        // false for every finite point, so the cell is never selected.
        assert!(aabbs[6..9].iter().all(|v| *v == f64::INFINITY));
        assert!(aabbs[9..12].iter().all(|v| *v == f64::NEG_INFINITY));
        // Placeholder material slot (never read); kept for index alignment.
        assert_eq!(cell_to_material[1], 0);
    }

    /// The flat region program the GPU evaluates must agree with the CPU's
    /// own `Region::contains` for every cell over a grid -- this pins the
    /// surface-index map and RPN encoding to the CPU semantics. Concentric
    /// shells are the case AABB-only finding cannot resolve.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn region_program_matches_cpu_contains_for_concentric_shells() {
        use yamc_gpu::common::geometry::region_eval::region_contains_cpu;

        let s1 = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 1.0, Some(1), None));
        let s2 = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 2.0, Some(2), None));
        let s3 = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 3.0, Some(3), None));

        // Core, two shells, all centred at origin (fully nested AABBs).
        let core = Region::new_from_halfspace(HalfspaceType::Below(Arc::clone(&s1)));
        let shell_a = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s1)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                Arc::clone(&s2),
            )));
        let shell_b = Region::new_from_halfspace(HalfspaceType::Above(Arc::clone(&s2)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                Arc::clone(&s3),
            )));
        let cells = vec![
            Cell::new(Some(1), core, None, Some(0)),
            Cell::new(Some(2), shell_a, None, Some(0)),
            Cell::new(Some(3), shell_b, None, Some(0)),
        ];

        let program = translate_region_program(&cells);
        let (surface_types, surface_params, _b) =
            translate_surfaces(&cells).expect("surfaces translate");
        let n_cells = cells.len() as u32;

        // Sweep a grid spanning all three shells and outside.
        let coords = [-3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5];
        for &x in &coords {
            for &y in &coords {
                for &z in &coords {
                    for (c, cell) in cells.iter().enumerate() {
                        let gpu = region_contains_cpu(
                            &program,
                            &surface_types,
                            &surface_params,
                            n_cells,
                            c as u32,
                            x,
                            y,
                            z,
                        );
                        let cpu = cell.region.contains((x, y, z));
                        assert_eq!(
                            gpu, cpu,
                            "cell {c} at ({x}, {y}, {z}): GPU region {gpu} != CPU {cpu}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn translate_cells_rejects_material_filled_unbounded_cell() {
        // A material-filled unbounded cell is a material implicit
        // complement, which point-in-AABB cell finding cannot transport
        // through. It must be rejected with a clear error rather than
        // silently turned into a leak sink.
        let sphere = unit_sphere();
        let outside = Region::new_from_halfspace(HalfspaceType::Above(sphere));
        let cells = vec![Cell::new(Some(7), outside, None, Some(0))];

        let err = translate_cells(&cells, 1).expect_err("material IC must be rejected");
        match err {
            GpuTranslateError::CellRegionUnsupported { cell_id, reason } => {
                assert_eq!(cell_id, Some(7));
                assert!(
                    reason.contains("material implicit complement"),
                    "reason should name the material implicit complement: {reason}"
                );
            }
            other => panic!("expected CellRegionUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn translate_cells_supports_finite_void_cell() {
        // A finite void cell (no material) is supported: it maps to the
        // synthetic void material slot at index `n_materials`, and
        // translate_cells reports `has_void = true`.
        let sphere = unit_sphere();
        let inside = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let cells = vec![Cell::new(Some(3), inside, None, None)];

        let (_aabbs, cell_to_material, has_void) =
            translate_cells(&cells, 0).expect("finite void cell is supported");
        assert!(has_void, "a material-less finite cell sets has_void");
        assert_eq!(
            cell_to_material[0], 0,
            "void cell maps to the void slot at index n_materials"
        );
    }

    /// Batch-free per-history variance (issue #233): the fixed-chunk seeding
    /// keys the per-particle PCG seed off the GLOBAL history index, so it is
    /// invariant to how many histories the run requests. Requesting N vs 2N
    /// particles from the same (launch_idx, chunk_size) yields the SAME
    /// per-particle stream on the shared prefix, and every seed is the shared
    /// `history_seed(base_seed, global_index)` (the matched-stream contract).
    #[test]
    fn fixed_chunk_seeding_is_total_independent() {
        let model = empty_model(vec![point_neutron_source()]);
        let chunk = 4000usize;
        let seed = 12345u64;

        // Chunk 0: seeds must be exactly the shared per-history seed for global
        // index `i` (matched-stream contract), and requesting fewer particles
        // must not change the shared prefix.
        let (s_full, e_full, p_full, d_full) =
            sample_initial_particles_for_chunk(&model, chunk, 0, chunk, seed);
        let half = chunk / 2;
        let (s_half, e_half, p_half, d_half) =
            sample_initial_particles_for_chunk(&model, half, 0, chunk, seed);
        for (i, &s) in s_full.iter().enumerate() {
            assert_eq!(
                s,
                history_seed(seed, i as u64),
                "chunk-0 seed[{i}] must equal history_seed(base, i)"
            );
        }
        assert_eq!(
            &s_full[..half],
            &s_half[..],
            "seed prefix total-independent"
        );
        assert_eq!(
            &e_full[..half],
            &e_half[..],
            "energy prefix total-independent"
        );
        assert_eq!(
            &p_full[..half * 3],
            &p_half[..],
            "position prefix total-independent"
        );
        assert_eq!(
            &d_full[..half * 3],
            &d_half[..],
            "direction prefix total-independent"
        );

        // Later chunk: the seed band is offset by the GLOBAL history index
        // `launch_idx * chunk_size + i`, independent of the requested count.
        let (s1_full, _, _, _) = sample_initial_particles_for_chunk(&model, chunk, 3, chunk, seed);
        let (s1_part, _, _, _) = sample_initial_particles_for_chunk(&model, half, 3, chunk, seed);
        for (i, &s) in s1_full.iter().enumerate() {
            let global = (3 * chunk + i) as u64;
            assert_eq!(
                s,
                history_seed(seed, global),
                "chunk-3 seed[{i}] keys off the global history index"
            );
        }
        assert_eq!(
            &s1_full[..half],
            &s1_part[..],
            "later-chunk seed prefix total-independent"
        );

        // Issue #315: the base seed reaches the per-history collision stream.
        // A different `TransportSettings::seed` must move every history's seed,
        // otherwise re-running with a new seed only re-samples the source.
        let (s_other, _, _, _) =
            sample_initial_particles_for_chunk(&model, chunk, 0, chunk, seed + 1);
        assert!(
            s_full.iter().zip(&s_other).all(|(a, b)| a != b),
            "a different base seed must change every per-history PCG seed"
        );
    }
}
