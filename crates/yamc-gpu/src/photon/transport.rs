//! GPU photon transport kernel.
//!
//! Mirrors the neutron `multi_cell_transport` kernel in shape -- same
//! BVH cell-finding, same surface-distance + tally infrastructure --
//! but with photon physics:
//!
//! - Macroscopic XS lookup on per-material `(total, coherent,
//!   incoherent, photoelectric, pair)` arrays, plus a per-material
//!   heating / KERMA array for `Score::Heating` / `Score::HeatingLocal`
//!   tallies (MT 301 / 901). No per-MT machinery.
//! - Reaction-type sampling among the four channels.
//! - Compton (incoherent) → Klein-Nishina via Kahn's rejection
//!   method, filtered against the bound-electron incoherent form
//!   factor `S(x, Z)`, with Doppler broadening of the scattered-photon
//!   energy from the per-shell `J(p_z)` Compton profiles -- matches
//!   the CPU `compton_scatter` + `compton_doppler` samplers.
//! - Rayleigh (coherent) → inverse-CDF sample on the per-material
//!   integrated coherent form factor `F(x², Z)`, then
//!   `μ = 1 − 2·x² / x²_max` with a Klein-Nishina-style angular
//!   acceptance `0.5·(1 + μ²)`.
//! - Photoelectric → full atomic-relaxation cascade: sample the
//!   absorbing subshell, then emit fluorescent X-rays / first-Auger
//!   electrons (the latter fed to TTB) -- matches the CPU
//!   `sample_photoelectric_subshell` + `atomic_relaxation` chain.
//! - Pair production → sample the e⁻/e⁺ energies and angles, run TTB
//!   bremsstrahlung on each, and emit two 511 keV annihilation
//!   photons.
//! - Heating / KERMA → all charged-particle kinetic energy is
//!   deposited locally (no electron transport), scored as
//!   `Σ_heating × track_length × weight` from the per-material heating
//!   array -- a table + arithmetic match to the CPU track-length
//!   photon-heating estimate `calculate_photon_xs().heating`
//!   (issue #356).
//!
//! Still out of scope (the CPU pipeline handles these): multi-hop
//! Auger-cascade TTB (only the first Auger electron is TTB'd here),
//! and explicit secondary-electron transport.
//!
//! ## Reproducibility
//!
//! Unlike the CPU and neutron-GPU paths, this kernel is **not**
//! bit-reproducible across runs: the secondary-particle cascade (pair
//! annihilation, fluorescence, positron/Auger TTB) makes tallies drift by
//! ~1-2% at high history counts (n ≳ 10k) under non-deterministic GPU
//! execution ordering. Use the CPU path (`compute="cpu"`) when exact
//! run-to-run reproducibility is required.

use crate::common::geometry::bvh_cell_finding::build_and_flatten_bvh;
use crate::common::geometry::region_eval::region_contains;
use crate::common::geometry::surface_distance::{
    cone_smallest_positive, quadric_smallest_positive, torus_smallest_positive,
};
use crate::common::polyfills::{cos_f64, exp_f64, ln_f64, sin_f64};
use crate::common::probes::atomic_relaxation::sample_relax_transition;
use crate::common::probes::compton_doppler::compton_doppler_sample;
use crate::common::probes::compton_scatter::compton_kahn_propose;
use crate::common::probes::incoherent_sf::incoherent_s_at;
use crate::common::probes::pe_subshell::sample_pe_subshell;
use crate::common::probes::photon_element_select::select_photon_element;
use crate::common::probes::photon_select::select_photon_reaction;
use crate::common::probes::rayleigh_scatter::rayleigh_propose;
use crate::common::tallies::{TalliesPack, TallyVarianceMode};
use crate::neutron::transport::{
    cyl_mesh_bin_at_kernel, cyl_mesh_score_src_acc, energy_function_weight_kernel,
    mesh_rect_score_src_acc, per_history_spill_cap, rect_mesh_bin_at_kernel, PERHIST_K,
};
use crate::photon::ttb_energy::ttb_photon_energy;
use crate::photon::xs::photon_xs::MAX_RAYLEIGH_FF;
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// `m_e · c²` in eV (CODATA 510 998.95 eV) -- used to convert photon energy
/// to the dimensionless `α = E / m_e c²` Klein-Nishina parameter and as the
/// energy of each e+ annihilation photon. Must equal the CPU constant
/// (`yamc_element::photon::MASS_ELECTRON_EV`) so the 511 keV annihilation line
/// lands in the same tally bin on both backends (issue #92).
pub const MASS_ELECTRON_EV: f64 = 0.510_998_950_00e6;

/// `h · c` in eV·angstrom -- used in the Rayleigh momentum-transfer
/// parameter `x = (m_e/hc) · α · sin(θ/2)`.
pub const PLANCK_C_EVA: f64 = 12_398.419_843_320_026;

/// Surface boundary discriminants (match the neutron kernel +
/// `yamc-geo`'s `BoundaryType` enum).
pub const PHOTON_BOUNDARY_TRANSMISSION: u32 = 0;
pub const PHOTON_BOUNDARY_VACUUM: u32 = 1;

/// Sanity-test target: number of read-only `&[...]` slice parameters in the
/// `multi_cell_photon_transport_kernel` signature (excludes the `&mut [...]`
/// outputs, matching the neutron `KERNEL_STORAGE_BUFFER_COUNT` convention).
/// Gates the launcher boundary against silent descriptor-aliasing: binding
/// more storage buffers than the adapter advertises aliases bindings and
/// produces garbage reads that look like physics bugs. Kept in lockstep with
/// the kernel signature by `photon_kernel_storage_buffer_count_matches_signature`.
pub const PHOTON_KERNEL_STORAGE_BUFFER_COUNT: u32 = 84;

/// `run_params` slot holding the model's `photon_cutoff_energy` in eV (issue
/// #286). One buffer carries the per-run scalars so adding another costs a slot
/// rather than a descriptor binding.
pub const PHOTON_PARAM_CUTOFF: usize = 0;
/// Length of the kernel's `run_params` buffer.
pub const PHOTON_RUN_PARAMS_LEN: usize = 1;

/// Per-thread cascade stack capacity. Holds secondaries (TTB brem
/// photons, fluorescent X-rays, e+ annihilation pair) that the
/// kernel produces before the primary dies. Bounded so the array
/// fits comfortably in thread-local storage / registers; sized for
/// the worst-case event we currently handle, which is a single
/// high-energy pair-production event in a high-Z medium (~2 × 511
/// keV annihilation photons + ~5-10 brem photons each from the e+
/// and e- TTB samplers + room for one extra cascade generation).
/// 16 covers this comfortably without the 60-byte-per-slot stack
/// blowing register pressure at workgroup size 128.
pub const PHOTON_CASCADE_STACK_CAP: u32 = 16;

/// Per-material atomic-relaxation shell stride. Must match
/// `yamc_gpu::MAX_AR_SHELLS`.
pub const AR_MAX_SHELLS: u32 = 16;
/// Per-(material, shell) atomic-transition stride. Must match
/// `yamc_gpu::MAX_AR_TRANS`.
pub const AR_MAX_TRANS: u32 = 32;

/// Per-material stride for the incoherent-form-factor `(x, S)`
/// table. Must match `yamc_gpu::incoherent_form_factor_xs::MAX_INCOHERENT_FF`.
pub const IFF_MAX_POINTS: u32 = 64;

/// Inverse fine-structure constant. Used by the Compton Doppler
/// sampler (same convention as CPU `yamc-element::photon`).
pub const FINE_STRUCTURE: f64 = 137.035_999_084;

/// Per-material shell-table stride for Doppler. Must match
/// `yamc_gpu::MAX_COMPTON_SHELLS` exactly.
pub const DOP_MAX_SHELLS: u32 = 32;
/// Per-(material, shell) pz-table stride. Must match
/// `yamc_gpu::MAX_COMPTON_PZ`.
pub const DOP_MAX_PZ: u32 = 64;
/// Per-(material, shell) relaxation-subshell stride. Must match
/// `yamc_gpu::MAX_COMPTON_RELAX` (the j = l +/- 1/2 doublet, so 2).
pub const DOP_MAX_RELAX: u32 = 2;

// `unused_assignments`: cubecl's `#[cube]` macro can't lower
// `let mut x: f64;` without an initializer (it needs the binding to
// have a concrete type in the SPIR-V code-gen pass), so the
// rotate-direction blocks initialise `new_dx / dy / dz` to `0.0`
// and then unconditionally overwrite. The dead store is intentional.
// `unused_variables`: `sigma_pair` exists for documentation parity
// with the other XS partials; the reaction-sampler picks "pair"
// only as the fall-through case so the value itself is never read.
#[allow(
    clippy::manual_clamp,
    clippy::assign_op_pattern,
    clippy::identity_op,
    clippy::too_many_arguments,
    unused_assignments,
    unused_variables
)]
#[cube(launch_unchecked)]
fn multi_cell_photon_transport_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    // Per-particle initial statistical weight (one f64 per source
    // particle, same indexing as `energies_in`). Banked coupled photons
    // carry the emitting neutron's weight, which is NOT always 1.0 (the
    // neutron kernel scales weight by fixed MT 16/17 (n,2n)/(n,3n) yields
    // and by nu_bar for fission). The kernel multiplies this weight into
    // every tally score site. Fresh-source photons pass all-1.0 so the
    // factor is a no-op (byte-identical to the pre-weight kernel).
    weights_in: &[f64],
    // Per-particle D1S parent-nuclide id (one u32 per source particle, same
    // indexing as `energies_in`). Read once into a per-thread register so every
    // secondary (fluorescence / Auger TTB / annihilation / brem) the photon
    // spawns inherits it -- they all score into the same parent-nuclide bin.
    // `0` (no parent) for prompt / fresh-source photons.
    parent_in: &[u32],
    positions_in: &[f64],
    directions_in: &[f64],
    cell_aabbs: &[f64],
    cell_to_material: &[u32],
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    bvh_prim_indices: &[u32],
    bvh_unbounded: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    surface_boundaries: &[u32],
    // Per-cell CSG region program (issue: photon cell-find must disambiguate
    // nested / overlapping AABBs with the exact region test, like the neutron
    // kernel; without it a photon far from the source is mis-assigned to an
    // inner cell whose bounding box still contains the point).
    region_program: &[u32],
    log_energy_grid: &[f64],
    // Per-material macroscopic photon XS, flat `[n_materials × n_grid]`.
    xs_total_per_material: &[f64],
    xs_coherent_per_material: &[f64],
    xs_incoherent_per_material: &[f64],
    xs_photoelectric_per_material: &[f64],
    xs_pair_per_material: &[f64],
    // Per-material macroscopic photon heating / KERMA XS, flat
    // `[n_materials × n_grid]`. Same density-weighted aggregation and
    // linear interpolation as the component XS above; the kernel scores
    // `Score::Heating` / `Score::HeatingLocal` photon tallies (MT 301 /
    // 901) from this array, mirroring the CPU track-length photon-heating
    // estimate `Σ_heating × track_length × weight` (issue #356).
    heating_xs_per_material: &[f64],
    // Per-material Rayleigh integrated coherent form factor.
    // `rayleigh_x2[mat * MAX_RAYLEIGH_FF + j]` is the `x²` axis (in
    // ENDF units -- `(m_e/hc)² · α² · sin²(θ/2)` scaled the same way
    // as the CPU `coherent_int_form_factor`).
    // `rayleigh_cdf[…]` is the integrated form-factor `F(x², Z)`.
    // `rayleigh_n_points[mat]` is the count of valid points.
    rayleigh_x2: &[f64],
    rayleigh_cdf: &[f64],
    rayleigh_n_points: &[u32],
    // Tally pack -- same layout as the neutron kernel. Photon
    // tallies exercise `SCORE_FLUX`, `SCORE_TOTAL`, and
    // `SCORE_PER_MT` for the four photon-component MTs (502
    // coherent, 504 incoherent, 522 photoelectric, 516 pair).
    // `tally_score_data[t]` carries the MT number directly for
    // SCORE_PER_MT tallies on the photon path -- the kernel
    // switches on the MT to read from the already-resident
    // per-component XS arrays (no separate xs_score_per_mt
    // buffer; data-piggybacking refinement of A).
    tally_score_kinds: &[u32],
    tally_cell_to_bin: &[u32],
    tally_n_cells: &[u32],
    tally_edges_offsets: &[u32],
    tally_n_bins: &[u32],
    tally_log_edges: &[f64],
    tally_out_offsets: &[u32],
    tally_score_data: &[u32],
    tally_fixed_point_scales: &[f64],
    // Per-tally estimator flag. `1` = collision estimator (score
    // `weight × score_xs / Σ_t` once per real collision), `0` =
    // track-length (the per-step `d × score_xs × weight` path). The
    // per-step block is gated on `== 0` and the collision
    // branch runs a parallel loop for `== 1`, so a tally never gets
    // both contributions. Length `n_tallies`.
    tally_is_collision: &[u32],
    // D1S `parent_nuclides` tally dimension. `tally_n_parent[t]` is the number
    // of parent-nuclide bins (1 = no parent filter, the parent dimension
    // collapses). `tally_parent_ids[tally_parent_offsets[t] ..
    // tally_parent_offsets[t+1]]` are the filter's resolved parent ids in bin
    // order; the kernel maps a photon's `parent_in` id to a bin by linear scan
    // (matching the CPU `ParentNuclideFilter::get_bin`). A photon whose parent
    // id is not in the range is NOT scored into that tally. Flat output index:
    // `out_off + (cell_bin * n_parent + parent_bin) * n_bins + energy_bin`.
    tally_n_parent: &[u32],
    tally_parent_offsets: &[u32],
    tally_parent_ids: &[u32],
    // Mesh (voxel) tally dimension (issue #234). `tally_n_mesh[t]` is the voxel
    // bin count (1 = no MeshFilter -> collapses, byte-identical). Mesh is the
    // INNERMOST flat dimension, so the full index is `out_off +
    // ((cell_bin * n_parent + parent_bin) * n_bins + e) * n_mesh + voxel`. Only
    // rectangular row-major meshes reach the kernel (validate_tallies gate); the
    // descriptor is packed in `tally_mesh_params[tally_mesh_params_offsets[t]..]`.
    // Only consumed in `mesh_direct` launches (mesh models route through the
    // per-source-direct variance path).
    tally_n_mesh: &[u32],
    tally_mesh_kind: &[u32],
    tally_mesh_params_offsets: &[u32],
    tally_mesh_params: &[f64],
    // Energy-function weighting (issue #271), the photon twin of the neutron
    // kernel's pair. `tally_efunc_offsets[t]..[t+1]` is tally `t`'s
    // `[n_points, energy[n], coeffs[4*(n-1)]]` table; an EMPTY range means no
    // `energy_function=` / `dose_coefficients=` filter. Multiplies the score
    // and drops the event when the photon energy is off the table.
    tally_efunc_offsets: &[u32],
    tally_efunc_params: &[f64],
    // Thick-target bremsstrahlung (TTB) tables. Populated when the
    // model has `electron_treatment == Ttb` and every material has
    // a `Bremsstrahlung` pack from CPU `init_bremsstrahlung`. When
    // `ttb_has_data[mat] == 0`, the TTB sampler is skipped for that
    // material (kernel reads the buffers but the per-material flag
    // gates whether the sample is consumed). All log-space (mirrors
    // CPU: `ttb_e_grid_log` is `ln(E)` for electron energy, both
    // PDF/CDF entries are values at those grid points). `n_ttb_e` is
    // the grid length (cached because `Array::len()` is not free in
    // cubecl-spirv).
    ttb_e_grid_log: &[f64],
    ttb_electron_pdf: &[f64],
    ttb_electron_cdf: &[f64],
    ttb_electron_yield: &[f64],
    ttb_has_data: &[u32],
    // Compton Doppler-broadening tables (yamc-element's
    // `compton_doppler` sampler ported to GPU). Per-material:
    // shell occupancy `dop_electron_pdf[m * MAX_SHELLS + s]`,
    // binding energies `dop_binding_energy[...]`, J(p_z) profile
    // and CDF on a shared pz grid. `dop_has_data[m]` gates whether
    // the kernel applies Doppler -- otherwise falls back to free
    // Klein-Nishina E_out.
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    // Compton-profile shell -> constituent atomic-relaxation subshells, letting
    // the incoherent branch relax the shell a Compton event ionizes (banking its
    // fluorescence) like the photoelectric branch. A Compton (n, l) shell maps to
    // up to DOP_MAX_RELAX (n, l, j) subshells; the kernel picks one weighted by
    // `dop_subshell_w0`. `dop_subshell_idx` is flat `[n_slab * DOP_MAX_SHELLS *
    // DOP_MAX_RELAX]` (u32::MAX pad); `dop_subshell_w0`/`dop_subshell_cnt` are flat
    // `[n_slab * DOP_MAX_SHELLS]` (cnt = 0 means no relaxation counterpart).
    dop_subshell_idx: &[u32],
    dop_subshell_w0: &[f64],
    dop_subshell_cnt: &[u32],
    // Incoherent (bound-electron Compton) form-factor S(x, Z) per
    // material. The kernel's Kahn loop rejects free-electron
    // Klein-Nishina samples with probability `1 - S(x)/S(x_max)`,
    // biasing the accepted (μ, α_out) distribution toward
    // larger-momentum-transfer events -- matches CPU's
    // `compton_scatter` rejection step. `iff_has_data[m] == 0`
    // → kernel skips rejection and uses pure free-electron Kahn.
    iff_x: &[f64],
    iff_s: &[f64],
    iff_n_points: &[u32],
    iff_has_data: &[u32],
    // Atomic-relaxation tables: per-subshell PE cross sections +
    // fluorescence / Auger transition CDFs. Used by the photoelectric
    // branch to sample which subshell absorbs the photon and emit
    // either a fluorescent X-ray (pushed onto the per-thread stack)
    // or an Auger electron (fed to TTB alongside the photoelectron).
    // `ar_has_data[m] == 0` → kernel falls back to K-shell guess with
    // no relaxation cascade.
    ar_has_data: &[u32],
    ar_n_shells: &[u32],
    ar_binding_energy: &[f64],
    ar_pe_subshell_xs_log: &[f64],
    ar_n_trans: &[u32],
    ar_trans_primary: &[u32],
    ar_trans_secondary: &[u32],
    ar_trans_energy: &[f64],
    ar_trans_cum_prob: &[f64],
    // Positron TTB tables (electron tables already present above).
    // Used by the pair-production branch to TTB the positron half
    // of the e+/e- pair before annihilation. Mirrors the electron
    // tables in shape: `[n_materials × n_ttb_e × n_ttb_e]` for the
    // PDF/CDF slabs and `[n_materials × n_ttb_e]` for the yield.
    ttb_positron_pdf: &[f64],
    ttb_positron_cdf: &[f64],
    ttb_positron_yield: &[f64],
    // Pair-production element constants (dominant-element only).
    // `pair_has_data[m] == 0` flags materials whose dominant Z is
    // out of the screening-radius table -- the kernel falls back to
    // killing the photon (slice-3a behavior). `r_z` is the
    // PENELOPE-2011 reduced screening radius; `a = Z / alpha`; `c`
    // is the precomputed Coulomb correction (Z-only, host-side).
    pair_has_data: &[u32],
    pair_r_z: &[f64],
    pair_a: &[f64],
    pair_c: &[f64],
    // Per-collision element-selection inputs (task #72). `elem_macro_total` is
    // the per-(element-slab, energy) macroscopic-total weight table, flat
    // `[n_slab x n_grid]`, element-major within a material and concatenated
    // material-major. `mat_elem_meta` is stride-2 `[offset, count]` per
    // material: the element-slab base and element count. At a collision the
    // kernel samples the interacting element by macro-XS contribution (mirroring
    // CPU `Material::sample_element`), computes its slab index `elem_off +
    // local`, and indexes that element's Rayleigh / Doppler / IFF / AR / pair
    // slab. `count == 1` -> the element is trivially `elem_off`; the selection
    // draw is SKIPPED so the RNG stream stays byte-identical to the #79
    // single-dominant-element path.
    elem_macro_total: &[f64],
    mat_elem_meta: &[u32],
    // Per-run scalars the kernel needs at runtime, packed into one binding so
    // future additions do not each cost a descriptor (issue #286). Slot layout:
    //   0 = `photon_cutoff_energy` (eV): photons at or below this are killed at
    //       the top of the step and never emitted as secondaries, mirroring the
    //       CPU's `photon_cutoff` threading through `transport/photon.rs`.
    run_params: &[f64],
    out_alive: &mut [u32],
    out_n_steps: &mut [u32],
    out_final_energy: &mut [f64],
    tally_out: &mut [Atomic<u64>],
    // Batch-free per-history / per-source variance (issue #233 Stage 3), mirroring
    // the neutron kernel. `spill_bin`/`spill_val` are the per-history overflow list
    // (a thread touching more than `PERHIST_K` distinct bins owns
    // `[ABSOLUTE_POS * spill_cap ..)`). `source_idx[i]` is the source PARTICLE this
    // photon descends from (identity for a photon source, the emitting neutron's
    // index for a drained secondary/decay photon), used to scatter this history's
    // per-bin sum into `src_acc[source_idx * total_bins + bin]` in `PerSource` mode.
    // All are size-1 dummies when off. The photon kernel never banks across a launch
    // (cascades are in-thread), so it needs no `bank_source_idx` output.
    spill_bin: &mut [u32],
    spill_val: &mut [f64],
    source_idx: &[u32],
    src_acc: &mut [Atomic<u64>],
    // Lost-particle diagnostics (issue #289), same contract as the neutron
    // kernel: `lost_count[0]` counts every history that ended in no cell and
    // `lost_f64` keeps the first `lost_f64.len() / LOST_F64_STRIDE` records.
    lost_count: &mut [Atomic<u64>],
    lost_f64: &mut [f64],
    #[comptime] max_steps: u32,
    // When true (PerHistory or PerSource), accumulate each history's per-bin total
    // in a thread-private touched-list (+ global spill) and flush at history end
    // instead of a per-step atomic add.
    #[comptime] per_history_var: bool,
    // Per-history spill capacity (`total_out_len - PERHIST_K`, clamped `>= 0`).
    #[comptime] spill_cap: u32,
    // PerSource: flush targets `src_acc` keyed by `source_idx` instead of the
    // sum/sum_sq halves of `tally_out`. Implies `per_history_var`.
    #[comptime] per_source_var: bool,
    // `src_acc` row stride (= `total_out_len`).
    #[comptime] total_bins: u32,
    // Mesh-tally variance path (issue #234). When true, each tally contribution
    // (each voxel crossing of a track-length mesh tally, or a single bin for a
    // non-mesh tally) is atomic-added DIRECTLY into `src_acc[source_idx*
    // total_bins + flat_idx]`, bypassing the touched-list (no O(distinct^2)
    // dedup for a many-voxel mesh step). Set together with the per-source host
    // layout; `spill_cap` is 0 so the history-end flush is a no-op. Mirrors the
    // neutron kernel's `mesh_direct`.
    #[comptime] mesh_direct: bool,
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    let i3 = ABSOLUTE_POS * 3;
    let mut energy = energies_in[ABSOLUTE_POS];
    let mut px = positions_in[i3];
    let mut py = positions_in[i3 + 1];
    let mut pz = positions_in[i3 + 2];
    let mut dx = directions_in[i3];
    let mut dy = directions_in[i3 + 1];
    let mut dz = directions_in[i3 + 2];
    let mut state = crate::common::pcg32::expand_seed(seeds[ABSOLUTE_POS]);

    // Per-particle statistical weight (mirrors the neutron kernel's
    // `weight`). Multiplied into every tally score. The photon kernel
    // never re-scales weight during transport (no VR splitting / no
    // multiplicity reactions here), so the whole TTB / fluorescence /
    // annihilation cascade spawned by this primary inherits this weight
    // -- the single register is correct for the entire thread. Fresh
    // sources pass 1.0, making the factor a no-op.
    let weight = weights_in[ABSOLUTE_POS];

    // D1S parent-nuclide id for this primary, read once. `0` for prompt /
    // fresh-source photons -- the parent dimension collapses.
    let parent_id = parent_in[ABSOLUTE_POS];
    // Parent-nuclide id of the particle CURRENTLY in transport, for D1S
    // `parent_nuclides` flux attribution. The primary photon (and its in-place
    // coherent/incoherent scatters) keeps `parent_id`; every transport-born
    // SECONDARY (fluorescence / Auger / annihilation / brem, drained from the
    // cascade stack below) is a fresh photon with NO radionuclide parent -- the
    // CPU/OpenMC reference does not attribute these to the source nuclide, so
    // they must not be binned into the parent's flux (issue #150: attributing
    // them over-predicted the D1S irradiation-phase photon flux by ~2% in the
    // thick sphere, where the secondary fraction is appreciable). Set to 0 when
    // a stack secondary is promoted to the transport variables.
    let mut current_parent = parent_id;

    let mut alive = 1u32;
    let mut n_steps = 0u32;

    let n_cells: u32 = (cell_aabbs.len() / 6) as u32;
    let n_surfaces: u32 = surface_types.len() as u32;
    let n_tallies: u32 = tally_score_kinds.len() as u32;
    let n_bvh_nodes: u32 = (bvh_aabbs.len() / 6) as u32;
    let n_bvh_unbounded: u32 = bvh_unbounded.len() as u32;
    let n_ttb_e: u32 = ttb_e_grid_log.len() as u32;

    // Batch-free per-history variance (issue #233 Stage 3): thread-private
    // touched-list (`th_bin`/`th_val`) + spill, mirroring the neutron kernel.
    // Declared UNCONDITIONALLY (a conditional `Array::new` trips a cubecl codegen
    // quirk); only read/written when `per_history_var`. `my_source_idx` is the
    // source particle this history descends from; `src_base` its `src_acc` row.
    let mut th_bin = Array::<u32>::new(PERHIST_K as usize);
    let mut th_val = Array::<f64>::new(PERHIST_K as usize);
    let mut th_count = 0u32;
    let mut spill_count = 0u32;
    let spill_base = ABSOLUTE_POS * spill_cap as usize;
    let mut my_source_idx = 0u32;
    if per_source_var || mesh_direct {
        my_source_idx = source_idx[ABSOLUTE_POS];
    }
    let src_base = my_source_idx as usize * total_bins as usize;

    // Per-thread bremsstrahlung secondary stack. When the primary
    // photon Compton-scatters, we sample N TTB photons from the
    // recoil electron's kinetic energy and push them here. After the
    // primary leaves the geometry (alive=0), the stack is drained:
    // pop the top, set it as the new primary, and continue the
    // transport loop. Capacity is fixed (slice-2 simplification);
    // overflow on push silently drops extras -- for the broomstick
    // verification setup, TTB yield per electron event is typically
    // < 1 so capacity 4 catches >99 % of events. Slice 5 will lift
    // this with recursive TTB + larger queues.
    //
    // `current_depth` and the parallel `stack_depth` array cap
    // cascade depth at `MAX_TTB_DEPTH` -- only photons whose depth is
    // strictly less than the cap get TTB-sampled on their next
    // Compton scatter. Without this, the cascade is unbounded and
    // GPU over-counts low-E flux dramatically (broomstick coherent
    // ~20× CPU).
    let mut stack_size = 0u32;
    let mut stack_e = Array::<f64>::new(16usize);
    let mut stack_dx = Array::<f64>::new(16usize);
    let mut stack_dy = Array::<f64>::new(16usize);
    let mut stack_dz = Array::<f64>::new(16usize);
    let mut stack_px = Array::<f64>::new(16usize);
    let mut stack_py = Array::<f64>::new(16usize);
    let mut stack_pz = Array::<f64>::new(16usize);
    let mut stack_depth = Array::<u32>::new(16usize);
    let mut current_depth = 0u32;

    // Atomic-relaxation vacancy stack, shared by the photoelectric and Compton
    // cascades below. Declared unconditionally (never inside an `if`) alongside
    // the emission stacks: a *conditional* `Array::new` is what the earlier
    // register-only cascade blamed for a cubecl-SPIRV/RADV codegen quirk. Size 8
    // >= the CPU `MAX_STACK_SIZE` (7), so the GPU follows the full primary +
    // secondary vacancy tree exactly like CPU `atomic_relaxation`.
    let mut ar_holes = Array::<u32>::new(8usize);

    // Model photon cutoff (issue #286). Read once: it is a per-run scalar, and
    // every cutoff site below uses this value rather than a literal, so
    // `photon_cutoff_energy` behaves the same on both backends.
    let photon_cutoff = run_params[PHOTON_PARAM_CUTOFF];

    // Cell of the previous step, for the lost-particle record (issue #289).
    // u32 sentinel `4_294_967_295` = "none yet" (a photon born outside every
    // cell); widened to f64 at the single record site.
    let mut last_cell = 4_294_967_295u32;

    let mut step = 0u32;
    let mut keep_going = 1u32;
    while step < max_steps && keep_going == 1u32 {
        // Photon cutoff: CPU kills photons whose energy drops below
        // `photon_cutoff_energy` (default 1 keV) during transport.
        // Without an equivalent on GPU, low-E TTB secondaries
        // accumulate flux that doesn't exist on the CPU side --
        // matches CPU `Model::run_internal`'s top-of-step kill.
        if alive == 1u32 && energy < photon_cutoff {
            alive = 0u32;
        }
        // If primary has died, try to promote a queued TTB photon
        // into the transport variables. When the stack is empty, this
        // thread is done -- set `keep_going = 0` so the while loop
        // exits cleanly (so we still get the output writes below).
        if alive == 0u32 {
            if stack_size > 0u32 {
                stack_size -= 1u32;
                energy = stack_e[stack_size as usize];
                dx = stack_dx[stack_size as usize];
                dy = stack_dy[stack_size as usize];
                dz = stack_dz[stack_size as usize];
                px = stack_px[stack_size as usize];
                py = stack_py[stack_size as usize];
                pz = stack_pz[stack_size as usize];
                current_depth = stack_depth[stack_size as usize];
                alive = 1u32;
                // Transport-born secondary: drop the D1S parent attribution so
                // its flux is not binned into the source radionuclide (#150).
                current_parent = 0u32;
            } else {
                keep_going = 0u32;
            }
        }
        // The transport body below only runs when we have a live
        // particle to push; otherwise we just spin step++ until the
        // outer while exits via `keep_going == 0`. Cubecl-spirv has
        // no `continue` so we gate the body on `keep_going`.
        if keep_going == 1u32 {
            // 1. BVH cell find. Mirrors the neutron kernel: the AABB is a cheap
            // pre-filter, then the exact CSG `region_contains` test disambiguates
            // nested / overlapping cells (an inner cell's bounding cube can still
            // contain a point that geometrically belongs to an outer cell).
            let n_cells: u32 = (cell_aabbs.len() / 6) as u32;
            let mut cell = 4_294_967_295u32;
            let mut bi = 0u32;
            while bi < n_bvh_nodes && cell == 4_294_967_295u32 {
                let ab: u32 = bi * 6u32;
                let min_x = bvh_aabbs[ab as usize];
                let min_y = bvh_aabbs[(ab + 1u32) as usize];
                let min_z = bvh_aabbs[(ab + 2u32) as usize];
                let max_x = bvh_aabbs[(ab + 3u32) as usize];
                let max_y = bvh_aabbs[(ab + 4u32) as usize];
                let max_z = bvh_aabbs[(ab + 5u32) as usize];
                let in_aabb = px >= min_x
                    && px <= max_x
                    && py >= min_y
                    && py <= max_y
                    && pz >= min_z
                    && pz <= max_z;
                let mb: u32 = bi * 3u32;
                let r_or_f = bvh_meta[mb as usize];
                let n_prims = bvh_meta[(mb + 1u32) as usize];
                let escape = bvh_meta[(mb + 2u32) as usize];
                if in_aabb {
                    if n_prims > 0u32 {
                        let mut p = 0u32;
                        while p < n_prims && cell == 4_294_967_295u32 {
                            let prim_idx = bvh_prim_indices[(r_or_f + p) as usize];
                            let cb: u32 = prim_idx * 6u32;
                            let cmin_x = cell_aabbs[cb as usize];
                            let cmin_y = cell_aabbs[(cb + 1u32) as usize];
                            let cmin_z = cell_aabbs[(cb + 2u32) as usize];
                            let cmax_x = cell_aabbs[(cb + 3u32) as usize];
                            let cmax_y = cell_aabbs[(cb + 4u32) as usize];
                            let cmax_z = cell_aabbs[(cb + 5u32) as usize];
                            if px >= cmin_x
                                && px <= cmax_x
                                && py >= cmin_y
                                && py <= cmax_y
                                && pz >= cmin_z
                                && pz <= cmax_z
                                && region_contains(
                                    region_program,
                                    surface_types,
                                    surface_params,
                                    n_cells,
                                    prim_idx,
                                    px,
                                    py,
                                    pz,
                                )
                            {
                                cell = prim_idx;
                            }
                            p += 1u32;
                        }
                    }
                    bi += 1u32;
                } else {
                    bi = escape;
                }
            }
            if cell == 4_294_967_295u32 {
                let mut u = 0u32;
                while u < n_bvh_unbounded && cell == 4_294_967_295u32 {
                    let prim_idx = bvh_unbounded[u as usize];
                    let cb: u32 = prim_idx * 6u32;
                    let cmin_x = cell_aabbs[cb as usize];
                    let cmin_y = cell_aabbs[(cb + 1u32) as usize];
                    let cmin_z = cell_aabbs[(cb + 2u32) as usize];
                    let cmax_x = cell_aabbs[(cb + 3u32) as usize];
                    let cmax_y = cell_aabbs[(cb + 4u32) as usize];
                    let cmax_z = cell_aabbs[(cb + 5u32) as usize];
                    if px >= cmin_x
                        && px <= cmax_x
                        && py >= cmin_y
                        && py <= cmax_y
                        && pz >= cmin_z
                        && pz <= cmax_z
                        && region_contains(
                            region_program,
                            surface_types,
                            surface_params,
                            n_cells,
                            prim_idx,
                            px,
                            py,
                            pz,
                        )
                    {
                        cell = prim_idx;
                    }
                    u += 1u32;
                }
            }

            if cell == 4_294_967_295u32 {
                // No cell covers this point: a lost photon, treated exactly as
                // the neutron kernel and the CPU treat it (issue #289). A leak
                // through a `boundary='vacuum'` surface never lands here (the
                // crossing block kills at the surface), so this only fires on a
                // genuine gap in the geometry.
                crate::common::lost_particles::record_lost(
                    lost_count,
                    lost_f64,
                    px,
                    py,
                    pz,
                    dx,
                    dy,
                    dz,
                    energy,
                    last_cell as f64,
                );
                alive = 0u32;
            } else {
                last_cell = cell;
                let mat_idx = cell_to_material[cell as usize];
                let n_grid: u32 = log_energy_grid.len() as u32;
                let mat_offset: u32 = mat_idx * n_grid;

                // 2. XS lookup at current energy.
                let log_e = ln_f64(energy);
                let mut lo = 0u32;
                let mut hi = n_grid;
                let mut iter = 0u32;
                while iter < 32u32 && lo < hi {
                    let mid: u32 = (lo + hi) / 2u32;
                    if log_energy_grid[mid as usize] < log_e {
                        lo = mid + 1u32;
                    } else {
                        hi = mid;
                    }
                    iter += 1u32;
                }
                let mut idx_hi = lo;
                if idx_hi < 1u32 {
                    idx_hi = 1u32;
                }
                if idx_hi >= n_grid {
                    idx_hi = n_grid - 1u32;
                }
                let idx_lo: u32 = idx_hi - 1u32;
                let x_lo = log_energy_grid[idx_lo as usize];
                let x_hi = log_energy_grid[idx_hi as usize];
                let frac = (log_e - x_lo) / (x_hi - x_lo);
                let xt_lo = xs_total_per_material[(mat_offset + idx_lo) as usize];
                let xt_hi = xs_total_per_material[(mat_offset + idx_hi) as usize];
                let sigma_t = xt_lo + (xt_hi - xt_lo) * frac;
                let xc_lo = xs_coherent_per_material[(mat_offset + idx_lo) as usize];
                let xc_hi = xs_coherent_per_material[(mat_offset + idx_hi) as usize];
                let sigma_coh = xc_lo + (xc_hi - xc_lo) * frac;
                let xi_lo = xs_incoherent_per_material[(mat_offset + idx_lo) as usize];
                let xi_hi = xs_incoherent_per_material[(mat_offset + idx_hi) as usize];
                let sigma_inc = xi_lo + (xi_hi - xi_lo) * frac;
                let xph_lo = xs_photoelectric_per_material[(mat_offset + idx_lo) as usize];
                let xph_hi = xs_photoelectric_per_material[(mat_offset + idx_hi) as usize];
                let sigma_photo = xph_lo + (xph_hi - xph_lo) * frac;
                let xp_lo = xs_pair_per_material[(mat_offset + idx_lo) as usize];
                let xp_hi = xs_pair_per_material[(mat_offset + idx_hi) as usize];
                let sigma_pair = xp_lo + (xp_hi - xp_lo) * frac;
                // Macroscopic photon heating / KERMA XS, interpolated the
                // SAME linear way as the component XS. The host array is
                // built from the density-weighted per-element
                // `ElementMicroXS.heating`, so this is an arithmetic match to
                // the CPU `calculate_photon_xs().heating`.
                let xh_lo = heating_xs_per_material[(mat_offset + idx_lo) as usize];
                let xh_hi = heating_xs_per_material[(mat_offset + idx_hi) as usize];
                let sigma_heating = xh_lo + (xh_hi - xh_lo) * frac;

                // 3. Free-flight sample.
                let d_xi1 = crate::common::pcg32::draw_uniform(state);
                state = d_xi1.state;
                let xi1 = d_xi1.xi;
                let mut d_xs = 1e30_f64;
                if sigma_t > 0.0 {
                    d_xs = -ln_f64(xi1) / sigma_t;
                }

                // 4. Distance to nearest surface. All eight surface kinds
                // (sphere, plane, cylinder, X/Y/Z tori, quadric, cone) --
                // same surface_types discriminants and stride-10 layout
                // as the neutron kernel.
                let mut d_boundary = 1e30_f64;
                let mut winner_surface = 0u32;
                let mut s = 0u32;
                while s < n_surfaces {
                    let stype = surface_types[s as usize];
                    let sb = (s * 10u32) as usize;
                    let p0 = surface_params[sb];
                    let p1 = surface_params[sb + 1];
                    let p2 = surface_params[sb + 2];
                    let p3 = surface_params[sb + 3];
                    let p4 = surface_params[sb + 4];
                    let p5 = surface_params[sb + 5];
                    let p6 = surface_params[sb + 6];
                    let p7 = surface_params[sb + 7];
                    let p8 = surface_params[sb + 8];
                    let p9 = surface_params[sb + 9];
                    let mut hit = 1e30_f64;
                    if stype == 0u32 {
                        // sphere
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
                    } else if stype == 1u32 {
                        // plane: a x + b y + c z = d (params p0..p3)
                        let denom = p0 * dx + p1 * dy + p2 * dz;
                        if denom.abs() > 1e-30 {
                            let t = (p3 - (p0 * px + p1 * py + p2 * pz)) / denom;
                            if t > 1e-12 {
                                hit = t;
                            }
                        }
                    } else if stype == 2u32 {
                        // cylinder: origin (p0,p1,p2), radius p3, axis (p4,p5,p6)
                        let qx = px - p0;
                        let qy = py - p1;
                        let qz = pz - p2;
                        let ax = p4;
                        let ay = p5;
                        let az = p6;
                        let dr_x = dx - (dx * ax + dy * ay + dz * az) * ax;
                        let dr_y = dy - (dx * ax + dy * ay + dz * az) * ay;
                        let dr_z = dz - (dx * ax + dy * ay + dz * az) * az;
                        let qr_x = qx - (qx * ax + qy * ay + qz * az) * ax;
                        let qr_y = qy - (qx * ax + qy * ay + qz * az) * ay;
                        let qr_z = qz - (qx * ax + qy * ay + qz * az) * az;
                        let a = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;
                        let bb = 2.0 * (qr_x * dr_x + qr_y * dr_y + qr_z * dr_z);
                        let c = qr_x * qr_x + qr_y * qr_y + qr_z * qr_z - p3 * p3;
                        if a > 1e-30 {
                            let disc = bb * bb - 4.0 * a * c;
                            if disc >= 0.0 {
                                let sqrt_disc = disc.sqrt();
                                let t1 = (-bb - sqrt_disc) / (2.0 * a);
                                let t2 = (-bb + sqrt_disc) / (2.0 * a);
                                if t2 > 1e-12 {
                                    hit = t2;
                                }
                                if t1 > 1e-12 {
                                    hit = t1;
                                }
                            }
                        }
                    } else if stype == 3u32 {
                        // ZTorus: axial coord is z; transverse pair is
                        // (x, y). Routing through the shared helper also
                        // picks up the c/b elliptical scaling the old
                        // inline form omitted.
                        hit = torus_smallest_positive(
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
                    } else if stype == 4u32 {
                        // XTorus: axial coord is x; transverse pair (y, z).
                        hit = torus_smallest_positive(
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
                    } else if stype == 5u32 {
                        // YTorus: axial coord is y; transverse pair (x, z).
                        hit = torus_smallest_positive(
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
                    } else if stype == 6u32 {
                        // Quadric: p0..p9 = a, b, c, d, e, f, g, h, j, k.
                        hit = quadric_smallest_positive(
                            px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                        );
                    } else if stype == 7u32 {
                        // Cone: p0..p2 apex, p3..p5 unit axis, p6 = tan²θ.
                        hit = cone_smallest_positive(
                            px, py, pz, dx, dy, dz, p0, p1, p2, p3, p4, p5, p6,
                        );
                    }
                    if hit < d_boundary {
                        d_boundary = hit;
                        winner_surface = s;
                    }
                    s += 1u32;
                }

                // 5. Min(d_xs, d_boundary).
                let collide_first = d_xs < d_boundary;
                let mut d = d_boundary;
                if collide_first {
                    d = d_xs;
                }

                // 6. Tally accumulation. Same pack layout as the neutron
                // kernel, but only `SCORE_FLUX` and `SCORE_TOTAL` are
                // meaningful for photons here.
                let mut t = 0u32;
                while t < n_tallies {
                    let cell_bin = tally_cell_to_bin[(t * n_cells + cell) as usize];
                    // Collision-estimator tallies (is_collision == 1) score at
                    // the collision site below, NOT per step -- skip them here
                    // so they never get both contributions.
                    let in_tally = cell_bin != 4_294_967_295u32;
                    let track_length_tally = tally_is_collision[t as usize] == 0u32;
                    // D1S parent-nuclide bin. n_parent == 1 -> bin 0, always
                    // scores (byte-identical to no parent dimension). n_parent
                    // > 1 -> scan the filter's resolved ids for this photon's
                    // parent; if not found, skip (parent_ok = false).
                    let n_parent = tally_n_parent[t as usize];
                    let mut parent_bin = 0u32;
                    let mut parent_ok = true;
                    if n_parent > 1u32 {
                        parent_ok = false;
                        let p_off = tally_parent_offsets[t as usize];
                        let mut pi = 0u32;
                        while pi < n_parent {
                            if !parent_ok
                                && tally_parent_ids[(p_off + pi) as usize] == current_parent
                            {
                                parent_bin = pi;
                                parent_ok = true;
                            }
                            pi += 1u32;
                        }
                    }
                    if in_tally && track_length_tally && parent_ok {
                        let n_bins = tally_n_bins[t as usize];
                        let edges_off = tally_edges_offsets[t as usize];
                        // Out-of-range guard: energies outside the filter
                        // range are dropped, not piled into the first/last
                        // bin (matches CPU `EnergyFilter::get_bin` -> None).
                        let lo_edge = tally_log_edges[edges_off as usize];
                        let hi_edge = tally_log_edges[(edges_off + n_bins) as usize];
                        let in_range = log_e >= lo_edge && log_e <= hi_edge;
                        let mut lo2 = 0u32;
                        let mut hi2 = n_bins;
                        let mut iter2 = 0u32;
                        while iter2 < 16u32 && lo2 + 1u32 < hi2 {
                            let mid: u32 = (lo2 + hi2) / 2u32;
                            let mid_edge = tally_log_edges[(edges_off + mid) as usize];
                            // `<=` so a particle exactly AT an interior edge
                            // scores into the LOWER bin -- matches the CPU
                            // `EnergyFilter::get_bin` convention
                            // `bins[i] < E <= bins[i+1]`. With `<`, a discrete
                            // source line sitting exactly on a group boundary
                            // (e.g. 1 MeV on CCFE-24) lands one bin high and
                            // the plotted spectrum peak shifts/scales wrongly.
                            if log_e <= mid_edge {
                                hi2 = mid;
                            } else {
                                lo2 = mid;
                            }
                            iter2 += 1u32;
                        }
                        let mut bin = lo2;
                        if bin >= n_bins {
                            bin = n_bins - 1u32;
                        }
                        let kind = tally_score_kinds[t as usize];
                        let mut score = 1.0; // SCORE_FLUX = 0
                        if kind == 1u32 {
                            score = sigma_t; // SCORE_TOTAL
                        } else if kind == 3u32 {
                            // SCORE_PER_MT -- photon path puts the MT
                            // number directly into score_data and the
                            // kernel switches on it to read the
                            // already-resident per-component XS. Unknown
                            // MTs yield zero so a tally accidentally
                            // wired to a neutron MT contributes nothing.
                            let mt = tally_score_data[t as usize];
                            let mut s_mt = 0.0;
                            if mt == 502u32 {
                                s_mt = sigma_coh;
                            } else if mt == 504u32 {
                                s_mt = sigma_inc;
                            } else if mt == 522u32 {
                                s_mt = sigma_photo;
                            } else if mt == 516u32 {
                                s_mt = sigma_pair;
                            } else if mt == 301u32 || mt == 901u32 {
                                // Heating (301) and heating-local (901):
                                // photons deposit all charged-particle KE
                                // locally (no electron transport), so the CPU
                                // scores BOTH from the same
                                // `calculate_photon_xs().heating` KERMA value.
                                s_mt = sigma_heating;
                            }
                            score = s_mt;
                        }
                        // Energy-function weighting (issue #271). Applied to
                        // `score` so the mesh DDA helpers, which take it by
                        // value, inherit it. Off the table drops the whole
                        // event (CPU `get_weight() == None` -> `return`), hence
                        // the AND into `in_range` rather than a zero score.
                        let ef_lo = tally_efunc_offsets[t as usize];
                        let ef_hi = tally_efunc_offsets[(t + 1u32) as usize];
                        let mut ef_in_range = true;
                        if ef_hi > ef_lo {
                            let n_ef = tally_efunc_params[ef_lo as usize] as u32;
                            let e_first = tally_efunc_params[(ef_lo + 1u32) as usize];
                            let e_last = tally_efunc_params[(ef_lo + n_ef) as usize];
                            if energy < e_first || energy > e_last {
                                ef_in_range = false;
                            } else {
                                score = score
                                    * energy_function_weight_kernel(
                                        tally_efunc_params,
                                        ef_lo,
                                        energy,
                                    );
                            }
                        }
                        let contrib = d * score * weight;
                        let scale = tally_fixed_point_scales[t as usize];
                        let scaled = (contrib * scale + 0.5) as i64;
                        let bits = u64::reinterpret(scaled);
                        let n_t_cells = tally_n_cells[t as usize];
                        let out_off = tally_out_offsets[t as usize];
                        // Stride order cell -> parent -> energy -> mesh (mesh
                        // innermost, matches the CPU 7D layout). n_parent == 1 and
                        // n_mesh == 1 collapse to the old index.
                        let cpe = (cell_bin * n_parent + parent_bin) * n_bins + bin;
                        let tally_idx = out_off + cpe;
                        let _ = n_t_cells;
                        if in_range && ef_in_range {
                            if mesh_direct {
                                // Issue #234: mesh models accumulate straight into
                                // the per-source accumulator (no touched-list). A
                                // mesh tally fans the track-length step across the
                                // voxels it crosses; a non-mesh tally writes its bin.
                                if tally_mesh_kind[t as usize] == 0u32 {
                                    src_acc[src_base + tally_idx as usize].fetch_add(bits);
                                } else {
                                    let n_mesh = tally_n_mesh[t as usize];
                                    let base = out_off + cpe * n_mesh;
                                    let mo = tally_mesh_params_offsets[t as usize];
                                    // Cylindrical (kind 3) vs rectangular (1/2)
                                    // voxel walk (issue #279).
                                    if tally_mesh_kind[t as usize] == 3u32 {
                                        cyl_mesh_score_src_acc(
                                            tally_mesh_params,
                                            mo,
                                            px,
                                            py,
                                            pz,
                                            dx,
                                            dy,
                                            dz,
                                            d,
                                            score,
                                            weight,
                                            base,
                                            src_base as u32,
                                            scale,
                                            src_acc,
                                        );
                                    } else {
                                        mesh_rect_score_src_acc(
                                            tally_mesh_params,
                                            mo,
                                            px,
                                            py,
                                            pz,
                                            dx,
                                            dy,
                                            dz,
                                            d,
                                            score,
                                            weight,
                                            base,
                                            src_base as u32,
                                            scale,
                                            src_acc,
                                        );
                                    }
                                }
                            } else if per_history_var {
                                // Batch-free per-history variance (issue #233): accumulate
                                // this history's per-bin PHYSICAL total in the touched-list
                                // (dedup on tally_idx), spilling past PERHIST_K. Flushed at
                                // history end.
                                let mut found = false;
                                let mut j = 0u32;
                                while j < th_count {
                                    if th_bin[j as usize] == tally_idx {
                                        th_val[j as usize] += contrib;
                                        found = true;
                                    }
                                    j += 1u32;
                                }
                                if !found {
                                    if th_count < PERHIST_K {
                                        th_bin[th_count as usize] = tally_idx;
                                        th_val[th_count as usize] = contrib;
                                        th_count += 1u32;
                                    } else {
                                        let mut sfound = false;
                                        let mut k = 0u32;
                                        while k < spill_count {
                                            if spill_bin[spill_base + k as usize] == tally_idx {
                                                spill_val[spill_base + k as usize] += contrib;
                                                sfound = true;
                                            }
                                            k += 1u32;
                                        }
                                        if !sfound {
                                            spill_bin[spill_base + spill_count as usize] =
                                                tally_idx;
                                            spill_val[spill_base + spill_count as usize] = contrib;
                                            spill_count += 1u32;
                                        }
                                    }
                                }
                            } else {
                                tally_out[tally_idx as usize].fetch_add(bits);
                            }
                        }
                    }
                    t += 1u32;
                }

                // 7. Move particle.
                px += dx * d;
                py += dy * d;
                pz += dz * d;

                if collide_first {
                    // 8a. Collision-estimator tallies. The collision-density
                    // flux estimator scores `weight × score_xs / Σ_t` once
                    // per real collision (vs the per-step
                    // `d × score_xs × weight` track-length path above). Same
                    // energy-bin / score-factor / atomic-add path -- only the
                    // effective distance changes from `d` to `1 / Σ_t`, and
                    // only `is_collision == 1` tallies fire. Σ_t > 0 holds
                    // (the distance was sampled against it).
                    let inv_sigma_t = 1.0 / sigma_t;
                    let mut tc = 0u32;
                    while tc < n_tallies {
                        let cell_bin_c = tally_cell_to_bin[(tc * n_cells + cell) as usize];
                        let in_tally_c = cell_bin_c != 4_294_967_295u32;
                        let collision_tally = tally_is_collision[tc as usize] == 1u32;
                        // D1S parent-nuclide bin (same scan as the track-length
                        // block above).
                        let n_parent_c = tally_n_parent[tc as usize];
                        let mut parent_bin_c = 0u32;
                        let mut parent_ok_c = true;
                        if n_parent_c > 1u32 {
                            parent_ok_c = false;
                            let p_off_c = tally_parent_offsets[tc as usize];
                            let mut pic = 0u32;
                            while pic < n_parent_c {
                                if !parent_ok_c
                                    && tally_parent_ids[(p_off_c + pic) as usize] == current_parent
                                {
                                    parent_bin_c = pic;
                                    parent_ok_c = true;
                                }
                                pic += 1u32;
                            }
                        }
                        if in_tally_c && collision_tally && parent_ok_c {
                            let n_bins_c = tally_n_bins[tc as usize];
                            let edges_off_c = tally_edges_offsets[tc as usize];
                            // Out-of-range guard (see per-step block above).
                            let lo_edge_c = tally_log_edges[edges_off_c as usize];
                            let hi_edge_c = tally_log_edges[(edges_off_c + n_bins_c) as usize];
                            let in_range_c = log_e >= lo_edge_c && log_e <= hi_edge_c;
                            let mut lo_c = 0u32;
                            let mut hi_c = n_bins_c;
                            let mut iter_c = 0u32;
                            while iter_c < 16u32 && lo_c + 1u32 < hi_c {
                                let mid_c: u32 = (lo_c + hi_c) / 2u32;
                                let mid_edge_c = tally_log_edges[(edges_off_c + mid_c) as usize];
                                if log_e <= mid_edge_c {
                                    hi_c = mid_c;
                                } else {
                                    lo_c = mid_c;
                                }
                                iter_c += 1u32;
                            }
                            let mut bin_c = lo_c;
                            if bin_c >= n_bins_c {
                                bin_c = n_bins_c - 1u32;
                            }
                            // Same score factor as the per-step block.
                            let kind_c = tally_score_kinds[tc as usize];
                            let mut score_c = 1.0; // SCORE_FLUX
                                                   // Photon heating / heating-local (MT 301 / 901) is
                                                   // NOT a collision-density (σ_heating/Σ_t) score on
                                                   // the collision estimator: the CPU instead deposits
                                                   // the ANALOG energy `E_in - E_out - banked_secondary
                                                   // _photon_energy` per real collision (see
                                                   // `score_photon_collision` in
                                                   // `crates/yamc/src/transport/mod.rs`). The KERMA
                                                   // table and the analog deposit differ by ~1% for Fe
                                                   // at 1.25 MeV, so scoring KERMA here ran the GPU
                                                   // collision heating ~0.96% high vs the CPU (task
                                                   // #68). Skip heating here; the analog deposit is
                                                   // scored in block 8c after the collision physics.
                            let mut is_heating_c = false;
                            if kind_c == 1u32 {
                                score_c = sigma_t; // SCORE_TOTAL
                            } else if kind_c == 3u32 {
                                let mt_c = tally_score_data[tc as usize];
                                let mut s_mt_c = 0.0;
                                if mt_c == 502u32 {
                                    s_mt_c = sigma_coh;
                                } else if mt_c == 504u32 {
                                    s_mt_c = sigma_inc;
                                } else if mt_c == 522u32 {
                                    s_mt_c = sigma_photo;
                                } else if mt_c == 516u32 {
                                    s_mt_c = sigma_pair;
                                } else if mt_c == 301u32 || mt_c == 901u32 {
                                    is_heating_c = true;
                                }
                                score_c = s_mt_c;
                            }
                            // Energy-function weighting (issue #271), same rule
                            // as the per-step block. Heating is excluded from
                            // this block entirely (`is_heating_c`) and picked up
                            // by the analog deposit in 8c, which applies the
                            // gate but not the weight -- see the note there.
                            let ef_lo_c = tally_efunc_offsets[tc as usize];
                            let ef_hi_c = tally_efunc_offsets[(tc + 1u32) as usize];
                            let mut ef_in_range_c = true;
                            if ef_hi_c > ef_lo_c {
                                let n_ef_c = tally_efunc_params[ef_lo_c as usize] as u32;
                                let e_first_c = tally_efunc_params[(ef_lo_c + 1u32) as usize];
                                let e_last_c = tally_efunc_params[(ef_lo_c + n_ef_c) as usize];
                                if energy < e_first_c || energy > e_last_c {
                                    ef_in_range_c = false;
                                } else {
                                    score_c = score_c
                                        * energy_function_weight_kernel(
                                            tally_efunc_params,
                                            ef_lo_c,
                                            energy,
                                        );
                                }
                            }
                            let contrib_c = score_c * weight * inv_sigma_t;
                            let scale_c = tally_fixed_point_scales[tc as usize];
                            let scaled_c = (contrib_c * scale_c + 0.5) as i64;
                            let bits_c = u64::reinterpret(scaled_c);
                            let out_off_c = tally_out_offsets[tc as usize];
                            let cpe_c = (cell_bin_c * n_parent_c + parent_bin_c) * n_bins_c + bin_c;
                            let tally_idx_c = out_off_c + cpe_c;
                            if in_range_c && !is_heating_c && ef_in_range_c {
                                if mesh_direct {
                                    // Issue #234: the collision-estimator score lands
                                    // in the single voxel holding the interaction
                                    // point (px,py,pz after the block-7 move). A
                                    // non-mesh tally writes its single composite bin.
                                    if tally_mesh_kind[tc as usize] == 0u32 {
                                        src_acc[src_base + tally_idx_c as usize].fetch_add(bits_c);
                                    } else {
                                        let n_mesh_c = tally_n_mesh[tc as usize];
                                        let mo_c = tally_mesh_params_offsets[tc as usize];
                                        // Cylindrical (kind 3) vs rectangular
                                        // (1/2) point binning (issue #279).
                                        let voxel_c = if tally_mesh_kind[tc as usize] == 3u32 {
                                            cyl_mesh_bin_at_kernel(
                                                tally_mesh_params,
                                                mo_c,
                                                px,
                                                py,
                                                pz,
                                            )
                                        } else {
                                            rect_mesh_bin_at_kernel(
                                                tally_mesh_params,
                                                mo_c,
                                                px,
                                                py,
                                                pz,
                                            )
                                        };
                                        if voxel_c != 4_294_967_295u32 {
                                            let base_c = out_off_c + cpe_c * n_mesh_c;
                                            src_acc[src_base + (base_c + voxel_c) as usize]
                                                .fetch_add(bits_c);
                                        }
                                    }
                                } else if per_history_var {
                                    let mut found = false;
                                    let mut j = 0u32;
                                    while j < th_count {
                                        if th_bin[j as usize] == tally_idx_c {
                                            th_val[j as usize] += contrib_c;
                                            found = true;
                                        }
                                        j += 1u32;
                                    }
                                    if !found {
                                        if th_count < PERHIST_K {
                                            th_bin[th_count as usize] = tally_idx_c;
                                            th_val[th_count as usize] = contrib_c;
                                            th_count += 1u32;
                                        } else {
                                            let mut sfound = false;
                                            let mut k = 0u32;
                                            while k < spill_count {
                                                if spill_bin[spill_base + k as usize] == tally_idx_c
                                                {
                                                    spill_val[spill_base + k as usize] += contrib_c;
                                                    sfound = true;
                                                }
                                                k += 1u32;
                                            }
                                            if !sfound {
                                                spill_bin[spill_base + spill_count as usize] =
                                                    tally_idx_c;
                                                spill_val[spill_base + spill_count as usize] =
                                                    contrib_c;
                                                spill_count += 1u32;
                                            }
                                        }
                                    }
                                } else {
                                    tally_out[tally_idx_c as usize].fetch_add(bits_c);
                                }
                            }
                        }
                        tc += 1u32;
                    }

                    // Analog photon-heating bookkeeping (task #68). The CPU
                    // collision estimator scores photon heating as the analog
                    // energy deposited at this real collision:
                    //   deposit = (E_in - E_out - banked_secondary_photon_E) · w
                    // (see `score_photon_collision`). Capture E_in and the
                    // current cascade-stack top NOW, before the physics below
                    // mutates `energy` and pushes secondaries. Every secondary
                    // this collision emits (fluorescence, Auger/recoil/pair TTB,
                    // 511 keV annihilation) is pushed onto `stack_e` between
                    // `heat_stack_start` and the post-physics `stack_size`, so
                    // summing that slice gives the banked secondary energy. A
                    // secondary dropped on stack overflow is (physically
                    // correctly) left as local deposit. Scored in block 8c.
                    let heat_e_in = energy;
                    let heat_stack_start = stack_size;

                    // 7b. Per-collision element selection (task #72). Mirror
                    // CPU `Material::sample_element`: pick the interacting
                    // element proportional to its macroscopic-total
                    // contribution at the collision energy, then run THAT
                    // element's coherent / incoherent / photoelectric / pair
                    // physics. The draw happens BEFORE the reaction draw to
                    // match the CPU draw order (`sample_element` then
                    // `handle_photon_collision`'s reaction cumulative).
                    // `elem_slab` indexes the per-element Rayleigh / Doppler /
                    // IFF / AR / pair slabs. When the material has a single
                    // element (`count == 1`) the draw is skipped -- `elem_slab`
                    // is trivially `elem_off`, no random is consumed, and the
                    // RNG stream stays byte-identical to the #79 path.
                    let elem_off = mat_elem_meta[(mat_idx * 2u32) as usize];
                    let elem_count = mat_elem_meta[(mat_idx * 2u32 + 1u32) as usize];
                    let mut elem_slab = elem_off;
                    if elem_count > 1u32 {
                        let sel = select_photon_element(
                            state,
                            elem_off,
                            elem_count,
                            n_grid,
                            idx_lo,
                            idx_hi,
                            frac,
                            elem_macro_total,
                        );
                        state = sel.state;
                        elem_slab = elem_off + sel.local;
                    }

                    // 8. Reaction-type sampling. Cumulative comparison
                    // against sigma_t.
                    let d_rxn = crate::common::pcg32::draw_uniform(state);
                    state = d_rxn.state;
                    let xi_rxn = d_rxn.xi;
                    let cutoff = xi_rxn * sigma_t;
                    // Cumulative interaction selection via the shared
                    // `select_photon_reaction` helper (single source of truth,
                    // unit-tested vs the CPU `handle_photon_collision`).
                    let rxn_kind =
                        select_photon_reaction(cutoff, sigma_coh, sigma_inc, sigma_photo);

                    if rxn_kind == 2u32 {
                        // Photoelectric absorption -- sample which
                        // subshell absorbs the photon (slice 3b);
                        // compute photoelectron KE = E - E_binding;
                        // emit TTB bremsstrahlung from the
                        // photoelectron; then run single-hop atomic
                        // relaxation to emit a fluorescent X-ray (if
                        // the sampled transition is radiative).
                        // Auger electrons (non-radiative transitions)
                        // are not yet TTB'd -- they're dropped, a
                        // known under-shoot for low-Z materials where
                        // Auger dominates. Mirrors CPU's
                        // `sample_photoelectric_subshell` +
                        // `atomic_relaxation` chain.

                        // ---- 1. Sample subshell. Two-pass: total →
                        // cumulative comparison. Matches CPU's
                        // `sample_photoelectric_subshell` semantics
                        // (skip below-threshold shells where
                        // `cross_sections[i_grid][s] == 0.0`, then
                        // cumulative compare on the per-element
                        // microscopic PE XS).
                        // Sample which subshell photo-absorbs via the shared
                        // `sample_pe_subshell` helper -- single source of truth,
                        // unit-tested against the CPU
                        // `sample_photoelectric_subshell`. Byte-identical (same
                        // one draw, same two-pass cumulative).
                        // Atomic-relaxation tables are keyed by the
                        // per-collision-selected element slab (task #72).
                        let mut sampled_shell = 0u32;
                        let ar_has = ar_has_data[elem_slab as usize];
                        let ar_ns = ar_n_shells[elem_slab as usize];
                        let mat_pe_off = elem_slab * n_grid;
                        if ar_has == 1u32 && ar_ns > 0u32 {
                            let s = sample_pe_subshell(
                                state,
                                ar_ns,
                                mat_pe_off,
                                idx_lo,
                                idx_hi,
                                frac,
                                ar_pe_subshell_xs_log,
                            );
                            state = s.state;
                            sampled_shell = s.shell;
                        }

                        // ---- 2. Binding energy for the sampled
                        // subshell. Mirror the CPU photoelectric path
                        // exactly: the photoelectron KE is `E - binding`
                        // where `binding = element.shells[i_shell]
                        // .binding_energy` (CPU `photon_photoelectric`,
                        // `yamc-element/src/photon.rs:326`). When the
                        // element has no atomic-relaxation data the CPU
                        // shells carry binding = 0, so the photoelectron
                        // keeps the full photon energy and TTB-radiates
                        // the whole brem tail. Falling back to Doppler's
                        // K-shell binding here (the old slice-3a guess)
                        // robbed the GPU photoelectron of most of its KE
                        // for AR-less downloaded data (e.g. fendl-3.2d
                        // Pb, K-binding ~88 keV), under-producing the
                        // low-energy bremsstrahlung continuum vs the CPU.
                        let e_b_k = if ar_has == 1u32 && ar_ns > 0u32 {
                            ar_binding_energy[(elem_slab * AR_MAX_SHELLS + sampled_shell) as usize]
                        } else {
                            energy * 0.0_f64
                        };

                        // ---- 3. Multi-hop atomic relaxation
                        // cascade. Mirrors CPU's `atomic_relaxation`:
                        // a per-thread hole stack tracks pending
                        // vacancies. For each, sample one transition;
                        // push the primary-electron's source vacancy
                        // onto the hole stack (always) plus the
                        // secondary vacancy for non-radiative
                        // transitions. Radiative -> push fluorescent
                        // photon onto the cascade stack. Non-radiative
                        // -> the Auger electron's KE feeds the inline
                        // TTB pass below; only the *first* Auger
                        // electron is TTB'd because we have a single
                        // inline TTB block per electron source -- the
                        // shallow cascade hops past that lose their
                        // electron's brem photons. The fluor chain is
                        // preserved in full.
                        let mut auger_ke = energy * 0.0_f64;
                        // Direction of the (first) Auger electron,
                        // sampled isotropically inside the cascade loop
                        // -- matches CPU's `atomic_relaxation` at
                        // `crates/yamc-element/src/photon.rs:815`. Only
                        // read when `got_auger == 1u32`, which gates the
                        // downstream Auger-TTB block, so the initial
                        // value below is never observed.
                        let mut auger_dx = dx * 0.0_f64;
                        let mut auger_dy = dy * 0.0_f64;
                        let mut auger_dz = dz * 0.0_f64 + 1.0_f64;
                        // Full atomic-relaxation cascade over the whole vacancy
                        // tree, mirroring CPU `atomic_relaxation`
                        // (crates/yamc-element/src/photon.rs:948). `ar_holes`
                        // (declared unconditionally with the emission stacks;
                        // size 8 >= CPU MAX_STACK_SIZE=7) is the pending-vacancy
                        // stack. Each pop samples one transition and pushes the
                        // primary vacancy always, plus the secondary (Auger)
                        // vacancy for a non-radiative transition -- so deep
                        // high-Z cascades and Auger-branch fluorescence are
                        // followed in full (the old 4-slot, primary-only variant
                        // truncated them, under-producing the soft fluorescence
                        // lines vs CPU). Radiative -> push fluorescent X-ray;
                        // a shell with no transitions emits a photon at its
                        // binding energy (CPU:972-977). The first Auger's KE
                        // feeds the inline TTB below; per-Auger TTB is a
                        // negligible sub-1 keV effect, deferred (see #37).
                        // `cascade_iter` bounds the loop trip count for SPIRV.
                        let mut got_auger = 0u32;
                        let mut n_holes = 0u32;
                        if ar_has == 1u32 {
                            ar_holes[0] = sampled_shell;
                            n_holes = 1u32;
                        }
                        let mut cascade_iter = 0u32;
                        while n_holes > 0u32 && cascade_iter < 64u32 {
                            cascade_iter += 1u32;
                            n_holes -= 1u32;
                            let i_hole = ar_holes[n_holes as usize];
                            if i_hole != 4_294_967_295u32 && i_hole < ar_ns {
                                let ns_idx = elem_slab * AR_MAX_SHELLS + i_hole;
                                let n_trans_s = ar_n_trans[ns_idx as usize];
                                // One isotropic direction per hole (matches CPU's
                                // single `isotropic_direction` call per loop pass).
                                let d_mu_iso = crate::common::pcg32::pcg_next(state);
                                let r_mu_iso = d_mu_iso.rand;
                                state = d_mu_iso.state;
                                let xi_mu_iso =
                                    (r_mu_iso as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                                let mu_iso = 2.0_f64 * xi_mu_iso - 1.0_f64;
                                let d_ph_iso = crate::common::pcg32::pcg_next(state);
                                let r_ph_iso = d_ph_iso.rand;
                                state = d_ph_iso.state;
                                let xi_ph_iso =
                                    (r_ph_iso as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                                let phi_iso = xi_ph_iso * std::f64::consts::TAU;
                                let sin_th_iso = (1.0_f64 - mu_iso * mu_iso).max(0.0_f64).sqrt();
                                let iso_dx = sin_th_iso * cos_f64(phi_iso);
                                let iso_dy = sin_th_iso * sin_f64(phi_iso);
                                let iso_dz = mu_iso;
                                if n_trans_s > 0u32 {
                                    let t_off = ns_idx * AR_MAX_TRANS;
                                    let rt = sample_relax_transition(
                                        state,
                                        t_off,
                                        n_trans_s,
                                        ar_trans_cum_prob,
                                        ar_trans_primary,
                                        ar_trans_secondary,
                                        ar_trans_energy,
                                    );
                                    state = rt.state;
                                    let primary = rt.primary;
                                    let secondary = rt.secondary;
                                    let e_trans = rt.energy;
                                    // Push primary vacancy (the electron that
                                    // filled the hole leaves a new vacancy).
                                    if primary != 4_294_967_295u32
                                        && primary < ar_ns
                                        && n_holes < 8u32
                                    {
                                        ar_holes[n_holes as usize] = primary;
                                        n_holes += 1u32;
                                    }
                                    if secondary == 4_294_967_295u32 {
                                        // Radiative: emit fluorescent X-ray.
                                        if e_trans > photon_cutoff
                                            && stack_size < PHOTON_CASCADE_STACK_CAP
                                        {
                                            stack_e[stack_size as usize] = e_trans;
                                            stack_dx[stack_size as usize] = iso_dx;
                                            stack_dy[stack_size as usize] = iso_dy;
                                            stack_dz[stack_size as usize] = iso_dz;
                                            stack_px[stack_size as usize] = px;
                                            stack_py[stack_size as usize] = py;
                                            stack_pz[stack_size as usize] = pz;
                                            stack_depth[stack_size as usize] = current_depth + 1u32;
                                            stack_size += 1u32;
                                        }
                                    } else {
                                        // Non-radiative: push the secondary
                                        // (Auger) vacancy so its own cascade is
                                        // followed too; TTB the first Auger only.
                                        if secondary < ar_ns && n_holes < 8u32 {
                                            ar_holes[n_holes as usize] = secondary;
                                            n_holes += 1u32;
                                        }
                                        if got_auger == 0u32 {
                                            auger_ke = e_trans;
                                            auger_dx = iso_dx;
                                            auger_dy = iso_dy;
                                            auger_dz = iso_dz;
                                            got_auger = 1u32;
                                        }
                                    }
                                } else {
                                    // No transition data: emit a photon at the
                                    // shell binding energy (CPU photon.rs:972-977).
                                    let be = ar_binding_energy[ns_idx as usize];
                                    if be > photon_cutoff && stack_size < PHOTON_CASCADE_STACK_CAP {
                                        stack_e[stack_size as usize] = be;
                                        stack_dx[stack_size as usize] = iso_dx;
                                        stack_dy[stack_size as usize] = iso_dy;
                                        stack_dz[stack_size as usize] = iso_dz;
                                        stack_px[stack_size as usize] = px;
                                        stack_py[stack_size as usize] = py;
                                        stack_pz[stack_size as usize] = pz;
                                        stack_depth[stack_size as usize] = current_depth + 1u32;
                                        stack_size += 1u32;
                                    }
                                }
                            }
                        }

                        // ---- 4a. TTB pass on the photoelectron.
                        let electron_ke = (energy - e_b_k).max(0.0_f64);
                        // Cap cascade at depth 3 -- primary + two
                        // generations of cascade photons trigger
                        // TTB. Bounds the cascade so per-thread
                        // stack (capacity 4) cannot overflow and
                        // low-E flux does not grow unboundedly.
                        if electron_ke > photon_cutoff
                            && ttb_has_data[mat_idx as usize] == 1u32
                            && n_ttb_e >= 2u32
                            && current_depth < 3u32
                        {
                            let log_ke = ln_f64(electron_ke);
                            // Binary search for largest j with
                            // ttb_e_grid_log[j] <= log_ke (matches the
                            // CPU's `lower_bound_index`).
                            let mut j_lo = 0u32;
                            let mut j_hi = n_ttb_e;
                            let mut j_iter = 0u32;
                            while j_iter < 16u32 && j_lo + 1u32 < j_hi {
                                let j_mid = (j_lo + j_hi) / 2u32;
                                if ttb_e_grid_log[j_mid as usize] <= log_ke {
                                    j_lo = j_mid;
                                } else {
                                    j_hi = j_mid;
                                }
                                j_iter += 1u32;
                            }
                            let mut j_idx = j_lo;
                            if j_idx >= n_ttb_e - 1u32 {
                                j_idx = n_ttb_e - 2u32;
                            }

                            let e_l_log = ttb_e_grid_log[j_idx as usize];
                            let e_r_log = ttb_e_grid_log[(j_idx + 1u32) as usize];
                            let denom_e = e_r_log - e_l_log;
                            let mut f_int = 0.0_f64;
                            if denom_e > 1e-30 {
                                f_int = (log_ke - e_l_log) / denom_e;
                            }
                            let yield_off = mat_idx * n_ttb_e;
                            let y_l = ttb_electron_yield[(yield_off + j_idx) as usize];
                            let y_r = ttb_electron_yield[(yield_off + j_idx + 1u32) as usize];
                            // y in log space; exponentiate after interp.
                            let y = exp_f64(y_l + (y_r - y_l) * f_int);

                            // Sample N = floor(y) + Bernoulli(frac).
                            let d_n = crate::common::pcg32::draw_uniform(state);
                            state = d_n.state;
                            let xi_n = d_n.xi;
                            let n_int = y as u32;
                            let frac_y = y - (n_int as f64);
                            let mut n_photons = n_int;
                            if xi_n < frac_y {
                                n_photons += 1u32;
                            }

                            // Choose CDF row (j or j+1) -- matches the
                            // CPU sampler's branch where it picks j+1
                            // with probability f_int (and always j+1
                            // when j == 0).
                            let d_ie = crate::common::pcg32::draw_uniform(state);
                            state = d_ie.state;
                            let xi_ie = d_ie.xi;

                            let slab_off = mat_idx * n_ttb_e * n_ttb_e;
                            let mut i_e = j_idx;
                            let mut c_max = 0.0_f64;
                            if xi_ie <= f_int || j_idx == 0u32 {
                                i_e = j_idx + 1u32;
                                let p_l = ttb_electron_pdf
                                    [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                let p_r =
                                    ttb_electron_pdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                                let c_l = ttb_electron_cdf
                                    [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                let a_int = ln_f64(p_r / p_l) / denom_e + 1.0;
                                let exp_term = exp_f64(a_int * (log_ke - e_l_log)) - 1.0;
                                c_max = c_l + exp_f64(e_l_log) * p_l / a_int * exp_term;
                            } else {
                                i_e = j_idx;
                                c_max = ttb_electron_cdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                            }

                            // Sample n_photons photon energies via
                            // inverse CDF, push to per-thread stack.
                            // Track cumulative emitted energy and cap
                            // each photon so total never exceeds the
                            // electron's KE -- matches CPU
                            // `thick_target_bremsstrahlung`'s energy-
                            // conservation cap and prevents the
                            // tail-of-spectrum over-emission that
                            // otherwise inflates downstream Compton
                            // and pair-cascade tallies.
                            let mut e_lost = electron_ke * 0.0_f64;
                            let mut k_photon = 0u32;
                            while k_photon < n_photons
                                && stack_size < PHOTON_CASCADE_STACK_CAP
                                && i_e >= 2u32
                            {
                                let d_c = crate::common::pcg32::draw_uniform(state);
                                state = d_c.state;
                                let xi_c = d_c.xi;
                                let c = xi_c * c_max;

                                // Binary search row [0..i_e] for the
                                // largest i_w with cdf[i_w] <= c.
                                let cdf_row_off = slab_off + i_e * n_ttb_e;
                                let mut iw_lo = 0u32;
                                let mut iw_hi = i_e;
                                let mut iw_iter = 0u32;
                                while iw_iter < 16u32 && iw_lo + 1u32 < iw_hi {
                                    let iw_mid = (iw_lo + iw_hi) / 2u32;
                                    if ttb_electron_cdf[(cdf_row_off + iw_mid) as usize] <= c {
                                        iw_lo = iw_mid;
                                    } else {
                                        iw_hi = iw_mid;
                                    }
                                    iw_iter += 1u32;
                                }
                                // No clamp on i_w. CPU's
                                // `lower_bound_index(cdf[..i_e], c)` returns
                                // i_w in [0, i_e-1]. When `i_w == i_e - 1`
                                // (the partial last-bin case where the
                                // photon energy lands between grid[i_e-1]
                                // and log_ke), the inverse-CDF formula
                                // pairs the real PDF at i_e-1 with the
                                // sentinel `pdf[i_e][i_e] = exp(-500)` --
                                // matches the c_max calculation above
                                // (very-negative `a` gives a steep
                                // exponential decay sample). Forcing
                                // i_w down to i_e-2 broke that branch
                                // and produced wildly inflated TTB
                                // contributions for the partial bin.
                                let i_w = iw_lo;

                                let w_l_log = ttb_e_grid_log[i_w as usize];
                                let w_r_log = ttb_e_grid_log[(i_w + 1u32) as usize];
                                let p_l_w = ttb_electron_pdf[(cdf_row_off + i_w) as usize];
                                let p_r_w = ttb_electron_pdf[(cdf_row_off + i_w + 1u32) as usize];
                                let c_l_w = ttb_electron_cdf[(cdf_row_off + i_w) as usize];
                                let mut w =
                                    ttb_photon_energy(c, w_l_log, w_r_log, p_l_w, p_r_w, c_l_w);

                                if w > photon_cutoff && stack_size < PHOTON_CASCADE_STACK_CAP {
                                    // Cap photon energy so cumulative
                                    // emission ≤ electron KE.
                                    if e_lost + w > electron_ke {
                                        w = electron_ke - e_lost;
                                    }
                                    if w > photon_cutoff {
                                        stack_e[stack_size as usize] = w;
                                        stack_dx[stack_size as usize] = dx;
                                        stack_dy[stack_size as usize] = dy;
                                        stack_dz[stack_size as usize] = dz;
                                        stack_px[stack_size as usize] = px;
                                        stack_py[stack_size as usize] = py;
                                        stack_pz[stack_size as usize] = pz;
                                        stack_depth[stack_size as usize] = current_depth + 1u32;
                                        stack_size += 1u32;
                                        e_lost += w;
                                    }
                                }
                                k_photon += 1u32;
                            }
                        }
                        // ----------- end TTB sampling -----------

                        // ---- 4b. TTB pass on the Auger electron
                        // (non-radiative atomic-relaxation product).
                        // Wrapped in `{}` to give the inner bindings
                        // a fresh scope so they shadow the
                        // photoelectron-TTB bindings cleanly. We
                        // duplicate the TTB block rather than wrap
                        // both passes in a `for` / `while` loop --
                        // empirically (on this kernel + cubecl
                        // version), wrapping the TTB body in a
                        // multi-iter loop doubled the kernel's
                        // track-length tally even when the second
                        // iteration's runtime if-guard skipped the
                        // body. Root cause is in cubecl/SPIR-V
                        // codegen of nested loops around mutating
                        // bodies; sidestepped by inline duplication.
                        {
                            let electron_ke = auger_ke;
                            if electron_ke > photon_cutoff
                                && ttb_has_data[mat_idx as usize] == 1u32
                                && n_ttb_e >= 2u32
                                && current_depth < 3u32
                            {
                                let log_ke = ln_f64(electron_ke);
                                let mut j_lo = 0u32;
                                let mut j_hi = n_ttb_e;
                                let mut j_iter = 0u32;
                                while j_iter < 16u32 && j_lo + 1u32 < j_hi {
                                    let j_mid = (j_lo + j_hi) / 2u32;
                                    if ttb_e_grid_log[j_mid as usize] <= log_ke {
                                        j_lo = j_mid;
                                    } else {
                                        j_hi = j_mid;
                                    }
                                    j_iter += 1u32;
                                }
                                let mut j_idx = j_lo;
                                if j_idx >= n_ttb_e - 1u32 {
                                    j_idx = n_ttb_e - 2u32;
                                }
                                let e_l_log = ttb_e_grid_log[j_idx as usize];
                                let e_r_log = ttb_e_grid_log[(j_idx + 1u32) as usize];
                                let denom_e = e_r_log - e_l_log;
                                let mut f_int = 0.0_f64;
                                if denom_e > 1e-30 {
                                    f_int = (log_ke - e_l_log) / denom_e;
                                }
                                let yield_off = mat_idx * n_ttb_e;
                                let y_l = ttb_electron_yield[(yield_off + j_idx) as usize];
                                let y_r = ttb_electron_yield[(yield_off + j_idx + 1u32) as usize];
                                let y = exp_f64(y_l + (y_r - y_l) * f_int);

                                let d_n = crate::common::pcg32::draw_uniform(state);
                                state = d_n.state;
                                let xi_n = d_n.xi;
                                let n_int = y as u32;
                                let frac_y = y - (n_int as f64);
                                let mut n_photons = n_int;
                                if xi_n < frac_y {
                                    n_photons += 1u32;
                                }

                                let d_ie = crate::common::pcg32::draw_uniform(state);
                                state = d_ie.state;
                                let xi_ie = d_ie.xi;

                                let slab_off = mat_idx * n_ttb_e * n_ttb_e;
                                let mut i_e = j_idx;
                                let mut c_max = 0.0_f64;
                                if xi_ie <= f_int || j_idx == 0u32 {
                                    i_e = j_idx + 1u32;
                                    let p_l = ttb_electron_pdf
                                        [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                    let p_r =
                                        ttb_electron_pdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                                    let c_l = ttb_electron_cdf
                                        [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                    let a_int = ln_f64(p_r / p_l) / denom_e + 1.0;
                                    let exp_term = exp_f64(a_int * (log_ke - e_l_log)) - 1.0;
                                    c_max = c_l + exp_f64(e_l_log) * p_l / a_int * exp_term;
                                } else {
                                    i_e = j_idx;
                                    c_max =
                                        ttb_electron_cdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                                }

                                let mut e_lost = electron_ke * 0.0_f64;
                                let mut k_photon = 0u32;
                                while k_photon < n_photons
                                    && stack_size < PHOTON_CASCADE_STACK_CAP
                                    && i_e >= 2u32
                                {
                                    let d_c = crate::common::pcg32::draw_uniform(state);
                                    state = d_c.state;
                                    let xi_c = d_c.xi;
                                    let c = xi_c * c_max;

                                    let cdf_row_off = slab_off + i_e * n_ttb_e;
                                    let mut iw_lo = 0u32;
                                    let mut iw_hi = i_e;
                                    let mut iw_iter = 0u32;
                                    while iw_iter < 16u32 && iw_lo + 1u32 < iw_hi {
                                        let iw_mid = (iw_lo + iw_hi) / 2u32;
                                        if ttb_electron_cdf[(cdf_row_off + iw_mid) as usize] <= c {
                                            iw_lo = iw_mid;
                                        } else {
                                            iw_hi = iw_mid;
                                        }
                                        iw_iter += 1u32;
                                    }
                                    let i_w = iw_lo;

                                    let w_l_log = ttb_e_grid_log[i_w as usize];
                                    let w_r_log = ttb_e_grid_log[(i_w + 1u32) as usize];
                                    let p_l_w = ttb_electron_pdf[(cdf_row_off + i_w) as usize];
                                    let p_r_w =
                                        ttb_electron_pdf[(cdf_row_off + i_w + 1u32) as usize];
                                    let c_l_w = ttb_electron_cdf[(cdf_row_off + i_w) as usize];
                                    let mut w =
                                        ttb_photon_energy(c, w_l_log, w_r_log, p_l_w, p_r_w, c_l_w);

                                    // Auger-TTB brem photons inherit
                                    // the Auger electron's (isotropic)
                                    // direction sampled in the multi-
                                    // hop loop above, not the parent
                                    // photon's direction -- matches CPU
                                    // `thick_target_bremsstrahlung`,
                                    // which is fed the electron's
                                    // direction by
                                    // `atomic_relaxation`.
                                    if w > photon_cutoff && stack_size < PHOTON_CASCADE_STACK_CAP {
                                        if e_lost + w > electron_ke {
                                            w = electron_ke - e_lost;
                                        }
                                        if w > photon_cutoff {
                                            stack_e[stack_size as usize] = w;
                                            stack_dx[stack_size as usize] = auger_dx;
                                            stack_dy[stack_size as usize] = auger_dy;
                                            stack_dz[stack_size as usize] = auger_dz;
                                            stack_px[stack_size as usize] = px;
                                            stack_py[stack_size as usize] = py;
                                            stack_pz[stack_size as usize] = pz;
                                            stack_depth[stack_size as usize] = current_depth + 1u32;
                                            stack_size += 1u32;
                                            e_lost += w;
                                        }
                                    }
                                    k_photon += 1u32;
                                }
                            }
                        }
                        // ----------- end photoelectric branch -----------

                        alive = 0u32;
                    } else if rxn_kind == 3u32 {
                        // Pair production (slice 4a). Sample the
                        // electron + positron energies / angles using
                        // PENELOPE-2011's compose-rejection
                        // formulation (mirrors CPU
                        // `PhotonInteraction::pair_production`). Push
                        // two 511 keV annihilation photons onto the
                        // cascade stack with isotropic + reverse
                        // directions. e+/e- TTB on the kinetic
                        // energies is a follow-up (slice 4b) -- it
                        // needs two more inline TTB blocks.
                        // Pair-production constants are keyed by the
                        // per-collision-selected element slab (task #72).
                        let pair_has = pair_has_data[elem_slab as usize];
                        let alpha_pp = energy / MASS_ELECTRON_EV;
                        // Pair-production threshold check: the
                        // reaction sampler can choose this branch
                        // only because `sigma_pair > 0` at the
                        // current energy, but if the dominant
                        // element's pair data is unusable we just
                        // absorb the photon -- same fallback as
                        // before this slice.
                        if pair_has == 1u32 && alpha_pp > 2.0_f64 {
                            let r_z = pair_r_z[elem_slab as usize];
                            let c_coul = pair_c[elem_slab as usize];
                            let a_born = pair_a[elem_slab as usize];

                            // Low-energy correction: closed-form
                            // polynomial in q and a from PENELOPE.
                            let q_lo = (2.0_f64 / alpha_pp).sqrt();
                            let q2 = q_lo * q_lo;
                            let q3 = q2 * q_lo;
                            let q4 = q2 * q2;
                            let a2 = a_born * a_born;
                            let f_corr = q_lo * (-0.1774_f64 - 12.10_f64 * a_born + 11.18_f64 * a2)
                                + q2 * (8.523_f64 + 73.26_f64 * a_born - 44.41_f64 * a2)
                                + q3 * (-13.52_f64 - 121.1_f64 * a_born + 96.41_f64 * a2)
                                + q4 * (8.946_f64 + 62.05_f64 * a_born - 63.41_f64 * a2);

                            // phi_i(1/2) bounding values.
                            let b_half = 2.0_f64 * r_z / alpha_pp;
                            // atan polyfill -- Taylor expansion of
                            // atan around 0, with the range-reduction
                            // identity atan(x) = π/2 sign(x) -
                            // atan(1/x) for |x| > 1. Only used twice
                            // per event (once for phi_i_max, once
                            // inside the rejection loop), so the
                            // ~20-term Horner cost is amortised. The
                            // existing `.cos()` / `.sin()` calls in
                            // this kernel work on the user's driver,
                            // but `.atan()` isn't validated -- the
                            // polyfill keeps the kernel portable.
                            // `b_half` is bounded below by 2 r_z /
                            // alpha_max; for our verification suite
                            // (alpha ≤ ~30 MeV / 0.511 MeV) and
                            // Z ≤ 98, b_half ∈ [~1.5, ~120], so
                            // 1/b_half ∈ [~0.008, ~0.67] -- well
                            // within the polyfill's |x| ≤ 1 regime.
                            let x_inv = 1.0_f64 / b_half;
                            // 12-term Horner Taylor for atan on
                            // |x| ≤ 1. Accuracy: ~1e-6 at x=1, much
                            // better near 0. Sufficient given MC
                            // noise.
                            let x2 = x_inv * x_inv;
                            let h11 = 1.0_f64 / 23.0_f64;
                            let h10 = -1.0_f64 / 21.0_f64 + x2 * h11;
                            let h9 = 1.0_f64 / 19.0_f64 + x2 * h10;
                            let h8 = -1.0_f64 / 17.0_f64 + x2 * h9;
                            let h7 = 1.0_f64 / 15.0_f64 + x2 * h8;
                            let h6 = -1.0_f64 / 13.0_f64 + x2 * h7;
                            let h5 = 1.0_f64 / 11.0_f64 + x2 * h6;
                            let h4 = -1.0_f64 / 9.0_f64 + x2 * h5;
                            let h3 = 1.0_f64 / 7.0_f64 + x2 * h4;
                            let h2 = -1.0_f64 / 5.0_f64 + x2 * h3;
                            let h1 = 1.0_f64 / 3.0_f64 + x2 * h2;
                            let atan_inv_b = x_inv * (1.0_f64 - x2 * h1);

                            let b2_half = b_half * b_half;
                            let t1m = 2.0_f64 * ln_f64(1.0_f64 + b2_half);
                            let t2m = b_half * atan_inv_b;
                            let t3m = b2_half
                                * (4.0_f64
                                    - 4.0_f64 * t2m
                                    - 3.0_f64 * ln_f64(1.0_f64 + 1.0_f64 / b2_half));
                            let t4_c = 4.0_f64 * ln_f64(r_z) - 4.0_f64 * c_coul + f_corr;
                            let phi1_max = 7.0_f64 / 3.0_f64 - t1m - 6.0_f64 * t2m - t3m + t4_c;
                            let phi2_max =
                                11.0_f64 / 6.0_f64 - t1m - 3.0_f64 * t2m + 0.5_f64 * t3m + t4_c;

                            let inv_alpha = 1.0_f64 / alpha_pp;
                            let half_minus_inv = 0.5_f64 - inv_alpha;
                            let u1 = 2.0_f64 / 3.0_f64 * half_minus_inv * half_minus_inv * phi1_max;
                            let u2 = phi2_max;
                            let mix = u1 / (u1 + u2);

                            // Bounded rejection loop (32 iters max).
                            // Average acceptance rate is high; in
                            // practice <10 iters suffice. If we
                            // exhaust the budget we keep the last
                            // trial -- accepts MC bias rather than
                            // hanging the kernel.
                            let mut e_pair = alpha_pp * 0.0_f64 + 0.5_f64;
                            let mut accepted = 0u32;
                            let mut pp_iter = 0u32;
                            while pp_iter < 32u32 && accepted == 0u32 {
                                // Two RNG draws per iteration.
                                let d_pp1 = crate::common::pcg32::pcg_next(state);
                                let r_pp1 = d_pp1.rand;
                                state = d_pp1.state;
                                let rn = (r_pp1 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

                                let d_pp2 = crate::common::pcg32::pcg_next(state);
                                let r_pp2 = d_pp2.rand;
                                state = d_pp2.state;
                                let xi_pi =
                                    (r_pp2 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

                                let mut e_trial = alpha_pp * 0.0_f64 + 0.5_f64;
                                let mut from_pi1 = 0u32;
                                if xi_pi < mix {
                                    from_pi1 = 1u32;
                                    // pi_1 inverse transform: e = 1/2 ±
                                    // (1/2 - 1/α) · cbrt(±(2 rn - 1))
                                    if rn >= 0.5_f64 {
                                        let arg = 2.0_f64 * rn - 1.0_f64;
                                        // cbrt(arg) for arg in [0, 1]
                                        // via exp(ln(arg)/3). Guard
                                        // arg == 0 by short-circuit.
                                        let cb = if arg > 0.0_f64 {
                                            exp_f64(ln_f64(arg) / 3.0_f64)
                                        } else {
                                            arg * 0.0_f64
                                        };
                                        e_trial = 0.5_f64 + half_minus_inv * cb;
                                    } else {
                                        let arg = 1.0_f64 - 2.0_f64 * rn;
                                        let cb = if arg > 0.0_f64 {
                                            exp_f64(ln_f64(arg) / 3.0_f64)
                                        } else {
                                            arg * 0.0_f64
                                        };
                                        e_trial = 0.5_f64 - half_minus_inv * cb;
                                    }
                                } else {
                                    // pi_2: uniform on [1/α, 1 - 1/α].
                                    e_trial = inv_alpha + half_minus_inv * 2.0_f64 * rn;
                                }

                                // Compute phi_i(e_trial).
                                let e_one_e = e_trial * (1.0_f64 - e_trial);
                                // Guard against pathological e_trial
                                // ~ 0 or ~ 1 -- would blow up b.
                                if e_one_e <= 0.0_f64 {
                                    pp_iter += 1u32;
                                } else {
                                    let b_e = r_z / (2.0_f64 * alpha_pp * e_one_e);
                                    let x_inv_e = 1.0_f64 / b_e;
                                    let x2_e = x_inv_e * x_inv_e;
                                    let h11e = 1.0_f64 / 23.0_f64;
                                    let h10e = -1.0_f64 / 21.0_f64 + x2_e * h11e;
                                    let h9e = 1.0_f64 / 19.0_f64 + x2_e * h10e;
                                    let h8e = -1.0_f64 / 17.0_f64 + x2_e * h9e;
                                    let h7e = 1.0_f64 / 15.0_f64 + x2_e * h8e;
                                    let h6e = -1.0_f64 / 13.0_f64 + x2_e * h7e;
                                    let h5e = 1.0_f64 / 11.0_f64 + x2_e * h6e;
                                    let h4e = -1.0_f64 / 9.0_f64 + x2_e * h5e;
                                    let h3e = 1.0_f64 / 7.0_f64 + x2_e * h4e;
                                    let h2e = -1.0_f64 / 5.0_f64 + x2_e * h3e;
                                    let h1e = 1.0_f64 / 3.0_f64 + x2_e * h2e;
                                    let atan_inv_be = x_inv_e * (1.0_f64 - x2_e * h1e);
                                    let b2_e = b_e * b_e;
                                    let t1e = 2.0_f64 * ln_f64(1.0_f64 + b2_e);
                                    let t2e = b_e * atan_inv_be;
                                    let t3e = b2_e
                                        * (4.0_f64
                                            - 4.0_f64 * t2e
                                            - 3.0_f64 * ln_f64(1.0_f64 + 1.0_f64 / b2_e));

                                    let d_acc = crate::common::pcg32::pcg_next(state);
                                    let r_acc = d_acc.rand;
                                    state = d_acc.state;
                                    let xi_acc =
                                        (r_acc as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

                                    if from_pi1 == 1u32 {
                                        let phi1 =
                                            7.0_f64 / 3.0_f64 - t1e - 6.0_f64 * t2e - t3e + t4_c;
                                        if xi_acc <= phi1 / phi1_max {
                                            e_pair = e_trial;
                                            accepted = 1u32;
                                        }
                                    } else {
                                        let phi2 = 11.0_f64 / 6.0_f64 - t1e - 3.0_f64 * t2e
                                            + 0.5_f64 * t3e
                                            + t4_c;
                                        if xi_acc <= phi2 / phi2_max {
                                            e_pair = e_trial;
                                            accepted = 1u32;
                                        }
                                    }
                                    pp_iter += 1u32;
                                }
                            }

                            // Kinetic energies of e- and e+.
                            let e_electron =
                                (alpha_pp * e_pair - 1.0_f64).max(0.0_f64) * MASS_ELECTRON_EV;
                            let e_positron = (alpha_pp * (1.0_f64 - e_pair) - 1.0_f64).max(0.0_f64)
                                * MASS_ELECTRON_EV;

                            // Sample electron / positron scattering
                            // mu via the CPU's inverse-transform
                            // `mu = (rn + β) / (rn β + 1)` with
                            // rn ∈ [-1, 1] and β the relativistic
                            // velocity ratio.
                            let beta_e_num =
                                (e_electron * (e_electron + 2.0_f64 * MASS_ELECTRON_EV)).sqrt();
                            let beta_e = beta_e_num / (e_electron + MASS_ELECTRON_EV);
                            let d_mue = crate::common::pcg32::pcg_next(state);
                            let r_mue = d_mue.rand;
                            state = d_mue.state;
                            let xi_mue = (r_mue as f64) * (1.0_f64 / 4_294_967_295.0_f64);
                            let rn_e = 2.0_f64 * xi_mue - 1.0_f64;
                            let mu_e = (rn_e + beta_e) / (rn_e * beta_e + 1.0_f64);

                            let beta_p_num =
                                (e_positron * (e_positron + 2.0_f64 * MASS_ELECTRON_EV)).sqrt();
                            let beta_p = beta_p_num / (e_positron + MASS_ELECTRON_EV);
                            let d_mup = crate::common::pcg32::pcg_next(state);
                            let r_mup = d_mup.rand;
                            state = d_mup.state;
                            let xi_mup = (r_mup as f64) * (1.0_f64 / 4_294_967_295.0_f64);
                            let rn_p = 2.0_f64 * xi_mup - 1.0_f64;
                            let mu_p = (rn_p + beta_p) / (rn_p * beta_p + 1.0_f64);

                            // Save the primary photon direction so
                            // both TTB blocks can rotate from it.
                            // After the second TTB the photon is
                            // killed (alive = 0), so no need to
                            // restore. The annihilation code below
                            // samples an isotropic direction
                            // independent of dx/dy/dz anyway.
                            let orig_dx = dx;
                            let orig_dy = dy;
                            let orig_dz = dz;

                            // ---- electron direction + TTB ----
                            {
                                // Sample azimuth and rotate primary
                                // direction by (mu_e, phi_e). Same
                                // formula as the Compton branch's
                                // rotation, with the polar-axis
                                // fallback.
                                let d_phe = crate::common::pcg32::pcg_next(state);
                                let r_phe = d_phe.rand;
                                state = d_phe.state;
                                let xi_phe =
                                    (r_phe as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                                let phi_e = xi_phe * std::f64::consts::TAU;
                                let cos_phi_e = cos_f64(phi_e);
                                let sin_phi_e = sin_f64(phi_e);
                                let sqrt_1_mue2 = (1.0_f64 - mu_e * mu_e).max(0.0_f64).sqrt();
                                let denom_dz_e = (1.0_f64 - orig_dz * orig_dz).max(0.0_f64).sqrt();
                                let mut nex = orig_dx * 0.0_f64;
                                let mut ney = orig_dy * 0.0_f64;
                                let mut nez = orig_dz * 0.0_f64;
                                if denom_dz_e > 1e-12_f64 {
                                    nex = mu_e * orig_dx
                                        + sqrt_1_mue2
                                            * (orig_dx * orig_dz * cos_phi_e - orig_dy * sin_phi_e)
                                            / denom_dz_e;
                                    ney = mu_e * orig_dy
                                        + sqrt_1_mue2
                                            * (orig_dy * orig_dz * cos_phi_e + orig_dx * sin_phi_e)
                                            / denom_dz_e;
                                    nez = mu_e * orig_dz - sqrt_1_mue2 * denom_dz_e * cos_phi_e;
                                } else {
                                    nex = sqrt_1_mue2 * cos_phi_e;
                                    ney = sqrt_1_mue2 * sin_phi_e;
                                    let mut sgn = 1.0_f64;
                                    if orig_dz < 0.0_f64 {
                                        sgn = 0.0_f64 - 1.0_f64;
                                    }
                                    nez = mu_e * sgn;
                                }
                                let nrm_e = (nex * nex + ney * ney + nez * nez).sqrt();
                                if nrm_e > 0.0_f64 {
                                    dx = nex / nrm_e;
                                    dy = ney / nrm_e;
                                    dz = nez / nrm_e;
                                }

                                // Inline electron TTB (mirrors PR
                                // #136 Auger-TTB pattern -- no `while`
                                // wrapper over electron sources to
                                // dodge cubecl loop-doubling).
                                let electron_ke = e_electron;
                                if electron_ke > photon_cutoff
                                    && ttb_has_data[mat_idx as usize] == 1u32
                                    && n_ttb_e >= 2u32
                                    && current_depth < 3u32
                                {
                                    let log_ke = ln_f64(electron_ke);
                                    let mut j_lo = 0u32;
                                    let mut j_hi = n_ttb_e;
                                    let mut j_iter = 0u32;
                                    while j_iter < 16u32 && j_lo + 1u32 < j_hi {
                                        let j_mid = (j_lo + j_hi) / 2u32;
                                        if ttb_e_grid_log[j_mid as usize] <= log_ke {
                                            j_lo = j_mid;
                                        } else {
                                            j_hi = j_mid;
                                        }
                                        j_iter += 1u32;
                                    }
                                    let mut j_idx = j_lo;
                                    if j_idx >= n_ttb_e - 1u32 {
                                        j_idx = n_ttb_e - 2u32;
                                    }
                                    let e_l_log = ttb_e_grid_log[j_idx as usize];
                                    let e_r_log = ttb_e_grid_log[(j_idx + 1u32) as usize];
                                    let denom_e = e_r_log - e_l_log;
                                    let mut f_int = 0.0_f64;
                                    if denom_e > 1e-30 {
                                        f_int = (log_ke - e_l_log) / denom_e;
                                    }
                                    let yield_off = mat_idx * n_ttb_e;
                                    let y_l = ttb_electron_yield[(yield_off + j_idx) as usize];
                                    let y_r =
                                        ttb_electron_yield[(yield_off + j_idx + 1u32) as usize];
                                    let y = exp_f64(y_l + (y_r - y_l) * f_int);

                                    let d_n = crate::common::pcg32::draw_uniform(state);
                                    state = d_n.state;
                                    let xi_n = d_n.xi;
                                    let n_int = y as u32;
                                    let frac_y = y - (n_int as f64);
                                    let mut n_photons = n_int;
                                    if xi_n < frac_y {
                                        n_photons += 1u32;
                                    }

                                    let d_ie = crate::common::pcg32::draw_uniform(state);
                                    state = d_ie.state;
                                    let xi_ie = d_ie.xi;

                                    let slab_off = mat_idx * n_ttb_e * n_ttb_e;
                                    let mut i_e = j_idx;
                                    let mut c_max = log_ke * 0.0_f64;
                                    if xi_ie <= f_int || j_idx == 0u32 {
                                        i_e = j_idx + 1u32;
                                        let p_l = ttb_electron_pdf
                                            [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                        let p_r = ttb_electron_pdf
                                            [(slab_off + i_e * n_ttb_e + i_e) as usize];
                                        let c_l = ttb_electron_cdf
                                            [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                        let a_int = ln_f64(p_r / p_l) / denom_e + 1.0;
                                        let exp_term = exp_f64(a_int * (log_ke - e_l_log)) - 1.0;
                                        c_max = c_l + exp_f64(e_l_log) * p_l / a_int * exp_term;
                                    } else {
                                        i_e = j_idx;
                                        c_max = ttb_electron_cdf
                                            [(slab_off + i_e * n_ttb_e + i_e) as usize];
                                    }

                                    let mut e_lost = electron_ke * 0.0_f64;
                                    let mut k_photon = 0u32;
                                    while k_photon < n_photons
                                        && stack_size < PHOTON_CASCADE_STACK_CAP
                                        && i_e >= 2u32
                                    {
                                        let d_c = crate::common::pcg32::draw_uniform(state);
                                        state = d_c.state;
                                        let xi_c = d_c.xi;
                                        let c = xi_c * c_max;

                                        let cdf_row_off = slab_off + i_e * n_ttb_e;
                                        let mut iw_lo = 0u32;
                                        let mut iw_hi = i_e;
                                        let mut iw_iter = 0u32;
                                        while iw_iter < 16u32 && iw_lo + 1u32 < iw_hi {
                                            let iw_mid = (iw_lo + iw_hi) / 2u32;
                                            if ttb_electron_cdf[(cdf_row_off + iw_mid) as usize]
                                                <= c
                                            {
                                                iw_lo = iw_mid;
                                            } else {
                                                iw_hi = iw_mid;
                                            }
                                            iw_iter += 1u32;
                                        }
                                        let i_w = iw_lo;

                                        let w_l_log = ttb_e_grid_log[i_w as usize];
                                        let w_r_log = ttb_e_grid_log[(i_w + 1u32) as usize];
                                        let p_l_w = ttb_electron_pdf[(cdf_row_off + i_w) as usize];
                                        let p_r_w =
                                            ttb_electron_pdf[(cdf_row_off + i_w + 1u32) as usize];
                                        let c_l_w = ttb_electron_cdf[(cdf_row_off + i_w) as usize];
                                        let mut w = ttb_photon_energy(
                                            c, w_l_log, w_r_log, p_l_w, p_r_w, c_l_w,
                                        );

                                        if w > photon_cutoff
                                            && stack_size < PHOTON_CASCADE_STACK_CAP
                                        {
                                            if e_lost + w > electron_ke {
                                                w = electron_ke - e_lost;
                                            }
                                            if w > photon_cutoff {
                                                stack_e[stack_size as usize] = w;
                                                stack_dx[stack_size as usize] = dx;
                                                stack_dy[stack_size as usize] = dy;
                                                stack_dz[stack_size as usize] = dz;
                                                stack_px[stack_size as usize] = px;
                                                stack_py[stack_size as usize] = py;
                                                stack_pz[stack_size as usize] = pz;
                                                stack_depth[stack_size as usize] =
                                                    current_depth + 1u32;
                                                stack_size += 1u32;
                                                e_lost += w;
                                            }
                                        }
                                        k_photon += 1u32;
                                    }
                                }
                            }
                            // ---- end electron TTB ----

                            // ---- positron direction + TTB ----
                            {
                                let d_php = crate::common::pcg32::pcg_next(state);
                                let r_php = d_php.rand;
                                state = d_php.state;
                                let xi_php =
                                    (r_php as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                                let phi_p = xi_php * std::f64::consts::TAU;
                                let cos_phi_p = cos_f64(phi_p);
                                let sin_phi_p = sin_f64(phi_p);
                                let sqrt_1_mup2 = (1.0_f64 - mu_p * mu_p).max(0.0_f64).sqrt();
                                let denom_dz_p = (1.0_f64 - orig_dz * orig_dz).max(0.0_f64).sqrt();
                                let mut npx_ = orig_dx * 0.0_f64;
                                let mut npy_ = orig_dy * 0.0_f64;
                                let mut npz_ = orig_dz * 0.0_f64;
                                if denom_dz_p > 1e-12_f64 {
                                    npx_ = mu_p * orig_dx
                                        + sqrt_1_mup2
                                            * (orig_dx * orig_dz * cos_phi_p - orig_dy * sin_phi_p)
                                            / denom_dz_p;
                                    npy_ = mu_p * orig_dy
                                        + sqrt_1_mup2
                                            * (orig_dy * orig_dz * cos_phi_p + orig_dx * sin_phi_p)
                                            / denom_dz_p;
                                    npz_ = mu_p * orig_dz - sqrt_1_mup2 * denom_dz_p * cos_phi_p;
                                } else {
                                    npx_ = sqrt_1_mup2 * cos_phi_p;
                                    npy_ = sqrt_1_mup2 * sin_phi_p;
                                    let mut sgn = 1.0_f64;
                                    if orig_dz < 0.0_f64 {
                                        sgn = 0.0_f64 - 1.0_f64;
                                    }
                                    npz_ = mu_p * sgn;
                                }
                                let nrm_p = (npx_ * npx_ + npy_ * npy_ + npz_ * npz_).sqrt();
                                if nrm_p > 0.0_f64 {
                                    dx = npx_ / nrm_p;
                                    dy = npy_ / nrm_p;
                                    dz = npz_ / nrm_p;
                                }

                                // Inline positron TTB -- same structure
                                // as the electron block above but
                                // reads the `ttb_positron_*` tables.
                                let electron_ke = e_positron;
                                if electron_ke > photon_cutoff
                                    && ttb_has_data[mat_idx as usize] == 1u32
                                    && n_ttb_e >= 2u32
                                    && current_depth < 3u32
                                {
                                    let log_ke = ln_f64(electron_ke);
                                    let mut j_lo = 0u32;
                                    let mut j_hi = n_ttb_e;
                                    let mut j_iter = 0u32;
                                    while j_iter < 16u32 && j_lo + 1u32 < j_hi {
                                        let j_mid = (j_lo + j_hi) / 2u32;
                                        if ttb_e_grid_log[j_mid as usize] <= log_ke {
                                            j_lo = j_mid;
                                        } else {
                                            j_hi = j_mid;
                                        }
                                        j_iter += 1u32;
                                    }
                                    let mut j_idx = j_lo;
                                    if j_idx >= n_ttb_e - 1u32 {
                                        j_idx = n_ttb_e - 2u32;
                                    }
                                    let e_l_log = ttb_e_grid_log[j_idx as usize];
                                    let e_r_log = ttb_e_grid_log[(j_idx + 1u32) as usize];
                                    let denom_e = e_r_log - e_l_log;
                                    let mut f_int = 0.0_f64;
                                    if denom_e > 1e-30 {
                                        f_int = (log_ke - e_l_log) / denom_e;
                                    }
                                    let yield_off = mat_idx * n_ttb_e;
                                    let y_l = ttb_positron_yield[(yield_off + j_idx) as usize];
                                    let y_r =
                                        ttb_positron_yield[(yield_off + j_idx + 1u32) as usize];
                                    let y = exp_f64(y_l + (y_r - y_l) * f_int);

                                    let d_n = crate::common::pcg32::draw_uniform(state);
                                    state = d_n.state;
                                    let xi_n = d_n.xi;
                                    let n_int = y as u32;
                                    let frac_y = y - (n_int as f64);
                                    let mut n_photons = n_int;
                                    if xi_n < frac_y {
                                        n_photons += 1u32;
                                    }

                                    let d_ie = crate::common::pcg32::draw_uniform(state);
                                    state = d_ie.state;
                                    let xi_ie = d_ie.xi;

                                    let slab_off = mat_idx * n_ttb_e * n_ttb_e;
                                    let mut i_e = j_idx;
                                    let mut c_max = log_ke * 0.0_f64;
                                    if xi_ie <= f_int || j_idx == 0u32 {
                                        i_e = j_idx + 1u32;
                                        let p_l = ttb_positron_pdf
                                            [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                        let p_r = ttb_positron_pdf
                                            [(slab_off + i_e * n_ttb_e + i_e) as usize];
                                        let c_l = ttb_positron_cdf
                                            [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                        let a_int = ln_f64(p_r / p_l) / denom_e + 1.0;
                                        let exp_term = exp_f64(a_int * (log_ke - e_l_log)) - 1.0;
                                        c_max = c_l + exp_f64(e_l_log) * p_l / a_int * exp_term;
                                    } else {
                                        i_e = j_idx;
                                        c_max = ttb_positron_cdf
                                            [(slab_off + i_e * n_ttb_e + i_e) as usize];
                                    }

                                    let mut e_lost = electron_ke * 0.0_f64;
                                    let mut k_photon = 0u32;
                                    while k_photon < n_photons
                                        && stack_size < PHOTON_CASCADE_STACK_CAP
                                        && i_e >= 2u32
                                    {
                                        let d_c = crate::common::pcg32::draw_uniform(state);
                                        state = d_c.state;
                                        let xi_c = d_c.xi;
                                        let c = xi_c * c_max;

                                        let cdf_row_off = slab_off + i_e * n_ttb_e;
                                        let mut iw_lo = 0u32;
                                        let mut iw_hi = i_e;
                                        let mut iw_iter = 0u32;
                                        while iw_iter < 16u32 && iw_lo + 1u32 < iw_hi {
                                            let iw_mid = (iw_lo + iw_hi) / 2u32;
                                            if ttb_positron_cdf[(cdf_row_off + iw_mid) as usize]
                                                <= c
                                            {
                                                iw_lo = iw_mid;
                                            } else {
                                                iw_hi = iw_mid;
                                            }
                                            iw_iter += 1u32;
                                        }
                                        let i_w = iw_lo;

                                        let w_l_log = ttb_e_grid_log[i_w as usize];
                                        let w_r_log = ttb_e_grid_log[(i_w + 1u32) as usize];
                                        let p_l_w = ttb_positron_pdf[(cdf_row_off + i_w) as usize];
                                        let p_r_w =
                                            ttb_positron_pdf[(cdf_row_off + i_w + 1u32) as usize];
                                        let c_l_w = ttb_positron_cdf[(cdf_row_off + i_w) as usize];
                                        let mut w = ttb_photon_energy(
                                            c, w_l_log, w_r_log, p_l_w, p_r_w, c_l_w,
                                        );

                                        if w > photon_cutoff
                                            && stack_size < PHOTON_CASCADE_STACK_CAP
                                        {
                                            if e_lost + w > electron_ke {
                                                w = electron_ke - e_lost;
                                            }
                                            if w > photon_cutoff {
                                                stack_e[stack_size as usize] = w;
                                                stack_dx[stack_size as usize] = dx;
                                                stack_dy[stack_size as usize] = dy;
                                                stack_dz[stack_size as usize] = dz;
                                                stack_px[stack_size as usize] = px;
                                                stack_py[stack_size as usize] = py;
                                                stack_pz[stack_size as usize] = pz;
                                                stack_depth[stack_size as usize] =
                                                    current_depth + 1u32;
                                                stack_size += 1u32;
                                                e_lost += w;
                                            }
                                        }
                                        k_photon += 1u32;
                                    }
                                }
                            }
                            // ---- end positron TTB ----
                        }

                        // Emit two 511 keV annihilation photons
                        // back-to-back. (Secondary emission like this is
                        // why the kernel is not bit-reproducible run-to-run
                        // -- see the "Reproducibility" note at the module
                        // head.) Direction is sampled isotropic
                        // -- the e+ slows to ~rest before annihilating
                        // (a CPU approximation we mirror here). The
                        // pair has space for two pushes if the stack
                        // has room; otherwise they're dropped, same
                        // soft cap as the rest of the cascade.
                        {
                            let d_th = crate::common::pcg32::pcg_next(state);
                            let r_th = d_th.rand;
                            state = d_th.state;
                            let xi_mu_ann =
                                (r_th as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                            let mu_ann = 2.0_f64 * xi_mu_ann - 1.0_f64;

                            let d_ph = crate::common::pcg32::pcg_next(state);
                            let r_ph = d_ph.rand;
                            state = d_ph.state;
                            let xi_ph_ann =
                                (r_ph as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                            let phi_ann = xi_ph_ann * std::f64::consts::TAU;
                            let sin_phi_ann = sin_f64(phi_ann);
                            let cos_phi_ann = cos_f64(phi_ann);
                            let sin_th_ann = (1.0_f64 - mu_ann * mu_ann).max(0.0_f64).sqrt();
                            let ann_dx = sin_th_ann * cos_phi_ann;
                            let ann_dy = sin_th_ann * sin_phi_ann;
                            let ann_dz = mu_ann;

                            // First annihilation photon.
                            if stack_size < PHOTON_CASCADE_STACK_CAP {
                                stack_e[stack_size as usize] = MASS_ELECTRON_EV;
                                stack_dx[stack_size as usize] = ann_dx;
                                stack_dy[stack_size as usize] = ann_dy;
                                stack_dz[stack_size as usize] = ann_dz;
                                stack_px[stack_size as usize] = px;
                                stack_py[stack_size as usize] = py;
                                stack_pz[stack_size as usize] = pz;
                                stack_depth[stack_size as usize] = current_depth + 1u32;
                                stack_size += 1u32;
                            }
                            // Second annihilation photon, opposite direction.
                            if stack_size < PHOTON_CASCADE_STACK_CAP {
                                stack_e[stack_size as usize] = MASS_ELECTRON_EV;
                                stack_dx[stack_size as usize] = 0.0_f64 - ann_dx;
                                stack_dy[stack_size as usize] = 0.0_f64 - ann_dy;
                                stack_dz[stack_size as usize] = 0.0_f64 - ann_dz;
                                stack_px[stack_size as usize] = px;
                                stack_py[stack_size as usize] = py;
                                stack_pz[stack_size as usize] = pz;
                                stack_depth[stack_size as usize] = current_depth + 1u32;
                                stack_size += 1u32;
                            }
                        }

                        alive = 0u32;
                    } else if rxn_kind == 1u32 {
                        // Compton (incoherent) -- Kahn's method for
                        // Klein-Nishina. Loop with up to 32 iterations.
                        // Save incident energy so we can compute the
                        // recoil-electron kinetic energy after sampling
                        // for the TTB photon-secondary loop below.
                        let e_in_compton = energy;
                        let alpha = energy / MASS_ELECTRON_EV;
                        let mut alpha_out = alpha;
                        let mut mu = 1.0_f64;
                        let mut accepted = 0u32;
                        let mut k_iter = 0u32;

                        // Pre-compute S(x_max, Z) once for this Compton
                        // event. x_max = (m_e/hc) · α corresponds to
                        // μ = -1 (back-scatter, max momentum transfer).
                        // The kernel evaluates the Tabulated1D linearly
                        // -- matches CPU `Tabulated1D::evaluate` for
                        // the lin-lin interpolation that ENDF/B incoherent
                        // form factors use.
                        // Incoherent form factor keyed by the per-collision-
                        // selected element slab (task #72).
                        let iff_n_pts = iff_n_points[elem_slab as usize];
                        let iff_has = iff_has_data[elem_slab as usize];
                        let iff_off = elem_slab * IFF_MAX_POINTS;
                        let x_max_iff = MASS_ELECTRON_EV / PLANCK_C_EVA * alpha;
                        // S(x_max) via the shared `incoherent_s_at` lookup
                        // (single source of truth, unit-tested against the CPU
                        // `incoherent_form_factor.evaluate`). The `else` arm is
                        // a runtime-1.0 sentinel; `iff_has == 0` makes the
                        // rejection below skip regardless.
                        let s_max_iff = if iff_has == 1u32 && iff_n_pts >= 2u32 {
                            incoherent_s_at(x_max_iff, iff_off, iff_n_pts, iff_x, iff_s)
                        } else {
                            x_max_iff * 0.0_f64 + 1.0_f64
                        };

                        while k_iter < 32u32 && accepted == 0u32 {
                            // One free-electron Klein-Nishina proposal via the
                            // shared `compton_kahn_propose` helper -- the
                            // single source of truth, unit-tested against the
                            // CPU `klein_nishina` in `compton_scatter.rs`. This
                            // is exactly one iteration of the old inlined loop
                            // (3 draws, one branch, one accept test), so the
                            // PCG stream and the shared Kahn+form-factor
                            // iteration cap are byte-identical to before.
                            let p = compton_kahn_propose(state, alpha);
                            state = p.state;
                            if p.accepted == 1u32 {
                                alpha_out = p.alpha_out;
                                mu = p.mu;
                                accepted = 1u32;
                            }
                            // Form-factor rejection -- when a Kahn
                            // sample was accepted above and the
                            // material has S(x, Z) tables loaded,
                            // multiply the acceptance with
                            // S(x_μ)/S(x_max). Implemented as a
                            // post-accept rejection: re-enter the
                            // loop (fresh Kahn sample) on reject.
                            if accepted == 1u32
                                && iff_has == 1u32
                                && iff_n_pts >= 2u32
                                && s_max_iff > 0.0_f64
                            {
                                // Momentum transfer at the sampled μ.
                                // mu can be slightly outside [-1, 1]
                                // due to numerical roundoff in the
                                // Kahn formulas; clamp via .max(0.0).
                                let one_minus_mu_iff = (1.0 - mu).max(0.0);
                                let x_iff = MASS_ELECTRON_EV / PLANCK_C_EVA
                                    * alpha
                                    * (0.5 * one_minus_mu_iff).sqrt();
                                // S(x) via the shared `incoherent_s_at` lookup.
                                let s_at_x =
                                    incoherent_s_at(x_iff, iff_off, iff_n_pts, iff_x, iff_s);
                                let d_xi_ff = crate::common::pcg32::draw_uniform(state);
                                state = d_xi_ff.state;
                                let xi_ff = d_xi_ff.xi;
                                if xi_ff >= s_at_x / s_max_iff {
                                    // Form factor rejects: continue
                                    // Kahn loop.
                                    accepted = 0u32;
                                }
                            }
                            k_iter += 1u32;
                        }
                        if accepted == 1u32 {
                            if mu < -1.0 {
                                mu = -1.0;
                            }
                            if mu > 1.0 {
                                mu = 1.0;
                            }
                            // Sample one phi shared between photon (rotated
                            // by phi+π) and the kinematic electron direction
                            // (rotated by phi). Matches CPU exactly
                            // (model.rs:466-497).
                            let d_phi = crate::common::pcg32::draw_uniform(state);
                            state = d_phi.state;
                            let xi_phi = d_phi.xi;
                            let phi = xi_phi * std::f64::consts::TAU;
                            let phi_photon = phi + std::f64::consts::PI;
                            let cos_phi = cos_f64(phi_photon);
                            let sin_phi = sin_f64(phi_photon);
                            let sqrt_1_mu2 = (1.0 - mu * mu).max(0.0).sqrt();
                            // Save pre-scatter direction so the Compton recoil
                            // TTB below can rotate it by (mu_electron, phi)
                            // for the brem photons (CPU model.rs:481-487
                            // / 558-560: brem inherits the electron's
                            // momentum direction, not the scattered photon's).
                            let dx_in_compton = dx;
                            let dy_in_compton = dy;
                            let dz_in_compton = dz;
                            // Kinematic mu_electron from energy-momentum
                            // conservation. alpha_out here is the KN value
                            // (pre-Doppler) -- matches CPU's mu_electron at
                            // model.rs:472-479. Stored for the recoil TTB
                            // direction rotation below.
                            let denom_kine = (alpha * alpha + alpha_out * alpha_out
                                - 2.0 * alpha * alpha_out * mu)
                                .max(0.0_f64)
                                .sqrt();
                            let mu_electron_compton = if denom_kine > 1e-30_f64 {
                                let cosp = (alpha - alpha_out * mu) / denom_kine;
                                cosp.max(0.0_f64 - 1.0_f64).min(1.0_f64)
                            } else {
                                alpha * 0.0_f64 + 1.0_f64
                            };
                            // Rotate photon direction by (mu, phi + π) --
                            // CPU uses the π offset so the photon and
                            // electron straddle the incoming axis (momentum
                            // conservation lives in the plane containing
                            // both).
                            let dz_abs_sq = dz * dz;
                            let denom_dz = (1.0 - dz_abs_sq).max(0.0).sqrt();
                            let mut new_dx = 0.0_f64;
                            let mut new_dy = 0.0_f64;
                            let mut new_dz = 0.0_f64;
                            if denom_dz > 1e-12 {
                                new_dx = mu * dx
                                    + sqrt_1_mu2 * (dx * dz * cos_phi - dy * sin_phi) / denom_dz;
                                new_dy = mu * dy
                                    + sqrt_1_mu2 * (dy * dz * cos_phi + dx * sin_phi) / denom_dz;
                                new_dz = mu * dz - sqrt_1_mu2 * denom_dz * cos_phi;
                            } else {
                                // Polar-axis fallback.
                                new_dx = sqrt_1_mu2 * cos_phi;
                                new_dy = sqrt_1_mu2 * sin_phi;
                                let mut sign_dz = 1.0_f64;
                                if dz < 0.0 {
                                    sign_dz = -1.0;
                                }
                                new_dz = mu * sign_dz;
                            }
                            // Renormalise to guard against drift.
                            let nrm = (new_dx * new_dx + new_dy * new_dy + new_dz * new_dz).sqrt();
                            if nrm > 0.0 {
                                dx = new_dx / nrm;
                                dy = new_dy / nrm;
                                dz = new_dz / nrm;
                            }
                            // Rotate incoming direction by
                            // (mu_electron, phi) to get the electron
                            // direction the brem cascade inherits.
                            let cos_phi_e = cos_f64(phi);
                            let sin_phi_e = sin_f64(phi);
                            let sqrt_1_me2 = (1.0 - mu_electron_compton * mu_electron_compton)
                                .max(0.0)
                                .sqrt();
                            let denom_dz_e = (1.0 - dz_in_compton * dz_in_compton).max(0.0).sqrt();
                            let mut e_dx_compton = dx_in_compton * 0.0_f64;
                            let mut e_dy_compton = dy_in_compton * 0.0_f64;
                            let mut e_dz_compton = dz_in_compton * 0.0_f64;
                            if denom_dz_e > 1e-12 {
                                e_dx_compton = mu_electron_compton * dx_in_compton
                                    + sqrt_1_me2
                                        * (dx_in_compton * dz_in_compton * cos_phi_e
                                            - dy_in_compton * sin_phi_e)
                                        / denom_dz_e;
                                e_dy_compton = mu_electron_compton * dy_in_compton
                                    + sqrt_1_me2
                                        * (dy_in_compton * dz_in_compton * cos_phi_e
                                            + dx_in_compton * sin_phi_e)
                                        / denom_dz_e;
                                e_dz_compton = mu_electron_compton * dz_in_compton
                                    - sqrt_1_me2 * denom_dz_e * cos_phi_e;
                            } else {
                                e_dx_compton = sqrt_1_me2 * cos_phi_e;
                                e_dy_compton = sqrt_1_me2 * sin_phi_e;
                                let mut sign_dz = 1.0_f64;
                                if dz_in_compton < 0.0_f64 {
                                    sign_dz = 0.0_f64 - 1.0_f64;
                                }
                                e_dz_compton = mu_electron_compton * sign_dz;
                            }
                            let nrm_e = (e_dx_compton * e_dx_compton
                                + e_dy_compton * e_dy_compton
                                + e_dz_compton * e_dz_compton)
                                .sqrt();
                            if nrm_e > 0.0_f64 {
                                e_dx_compton = e_dx_compton / nrm_e;
                                e_dy_compton = e_dy_compton / nrm_e;
                                e_dz_compton = e_dz_compton / nrm_e;
                            }

                            // ---------- Compton Doppler broadening ----------
                            // Shared `compton_doppler_sample` helper -- single
                            // source of truth, unit-tested vs the CPU
                            // `compton_doppler`. Byte-identical: same 3 draws,
                            // same kinematics; returns e_out_kn when skipped.
                            let e_out_kn = alpha_out * MASS_ELECTRON_EV;
                            // Doppler profiles keyed by the per-collision-
                            // selected element slab (task #72).
                            let ds = compton_doppler_sample(
                                state,
                                e_in_compton,
                                mu,
                                e_out_kn,
                                elem_slab,
                                dop_n_shells,
                                dop_has_data,
                                dop_pz_grid,
                                dop_electron_pdf,
                                dop_binding_energy,
                                dop_profile_pdf,
                                dop_profile_cdf,
                            );
                            state = ds.state;
                            let e_out_final = ds.e_out;

                            // Atomic relaxation of the Compton-ionized shell.
                            // The event ionizes the shell sampled from
                            // `dop_electron_pdf` (returned as `ds.shell`); the
                            // atom relaxes and emits fluorescence x-rays that
                            // must be banked + transported, exactly as the CPU
                            // `photon_incoherent` does
                            // (crates/yamc/src/transport/photon.rs). Without
                            // this the GPU low-energy photon spectrum was
                            // missing Compton-ionization fluorescence lines
                            // (issue #178). Mirrors the photoelectric branch's
                            // register-based hole0..hole3 cascade (NOT an
                            // Array -- avoids the RADV codegen quirk noted
                            // there) and reuses the SHARED cascade stack.
                            // Auger-electron TTB is a soft secondary and is
                            // deferred; the radiative x-rays are the
                            // spectrum-defining signal. No incident-energy
                            // guard: the CPU relaxes on every Compton event
                            // (its `energy + binding` argument defeats the
                            // relaxation guard), so the GPU does too.
                            {
                                let ar_has = ar_has_data[elem_slab as usize];
                                let ar_ns = ar_n_shells[elem_slab as usize];
                                if ds.shell < DOP_MAX_SHELLS && ar_has == 1u32 {
                                    // The ionized Compton (n, l) shell maps to one
                                    // or two (n, l, j) subshells; sample which one
                                    // took the vacancy, weighted by occupancy
                                    // (`dop_subshell_w0`), matching the CPU.
                                    let dop_base = (elem_slab * DOP_MAX_SHELLS + ds.shell) as usize;
                                    let relax_cnt = dop_subshell_cnt[dop_base];
                                    let mut i_relax = 4_294_967_295u32;
                                    if relax_cnt > 0u32 {
                                        let mut slot = 0u32;
                                        if relax_cnt >= 2u32 {
                                            let d_relax = crate::common::pcg32::pcg_next(state);
                                            state = d_relax.state;
                                            let xi_relax = (d_relax.rand as f64 + 1.0_f64)
                                                * (1.0_f64 / 4_294_967_297.0_f64);
                                            if xi_relax >= dop_subshell_w0[dop_base] {
                                                slot = 1u32;
                                            }
                                        }
                                        i_relax = dop_subshell_idx
                                            [(dop_base as u32 * DOP_MAX_RELAX + slot) as usize];
                                    }
                                    if i_relax != 4_294_967_295u32 && i_relax < ar_ns {
                                        // Full vacancy-tree cascade, mirroring CPU
                                        // `atomic_relaxation` and the photoelectric
                                        // branch: the `ar_holes` stack follows both
                                        // primary and secondary (Auger) vacancies,
                                        // emitting every fluorescent x-ray + the
                                        // no-transition binding-energy photon. The
                                        // Auger *electron's* TTB is deferred on the
                                        // Compton branch (soft secondary), but the
                                        // Auger vacancy's own fluorescence cascade
                                        // is followed. `cascade_iter` bounds the
                                        // loop trip count for SPIRV.
                                        ar_holes[0] = i_relax;
                                        let mut n_holes = 1u32;
                                        let mut cascade_iter = 0u32;
                                        while n_holes > 0u32 && cascade_iter < 64u32 {
                                            cascade_iter += 1u32;
                                            n_holes -= 1u32;
                                            let i_hole = ar_holes[n_holes as usize];
                                            if i_hole != 4_294_967_295u32 && i_hole < ar_ns {
                                                let ns_idx = elem_slab * AR_MAX_SHELLS + i_hole;
                                                let n_trans_s = ar_n_trans[ns_idx as usize];
                                                // One isotropic direction per hole.
                                                let d_mu_iso =
                                                    crate::common::pcg32::pcg_next(state);
                                                let r_mu_iso = d_mu_iso.rand;
                                                state = d_mu_iso.state;
                                                let xi_mu_iso = (r_mu_iso as f64 + 1.0_f64)
                                                    * (1.0_f64 / 4_294_967_297.0_f64);
                                                let mu_iso = 2.0_f64 * xi_mu_iso - 1.0_f64;
                                                let d_ph_iso =
                                                    crate::common::pcg32::pcg_next(state);
                                                let r_ph_iso = d_ph_iso.rand;
                                                state = d_ph_iso.state;
                                                let xi_ph_iso = (r_ph_iso as f64 + 1.0_f64)
                                                    * (1.0_f64 / 4_294_967_297.0_f64);
                                                let phi_iso = xi_ph_iso * std::f64::consts::TAU;
                                                let sin_th_iso =
                                                    (1.0_f64 - mu_iso * mu_iso).max(0.0_f64).sqrt();
                                                let iso_dx = sin_th_iso * cos_f64(phi_iso);
                                                let iso_dy = sin_th_iso * sin_f64(phi_iso);
                                                let iso_dz = mu_iso;
                                                if n_trans_s > 0u32 {
                                                    let t_off = ns_idx * AR_MAX_TRANS;
                                                    let rt = sample_relax_transition(
                                                        state,
                                                        t_off,
                                                        n_trans_s,
                                                        ar_trans_cum_prob,
                                                        ar_trans_primary,
                                                        ar_trans_secondary,
                                                        ar_trans_energy,
                                                    );
                                                    state = rt.state;
                                                    let primary = rt.primary;
                                                    let secondary = rt.secondary;
                                                    let e_trans = rt.energy;
                                                    if primary != 4_294_967_295u32
                                                        && primary < ar_ns
                                                        && n_holes < 8u32
                                                    {
                                                        ar_holes[n_holes as usize] = primary;
                                                        n_holes += 1u32;
                                                    }
                                                    if secondary == 4_294_967_295u32 {
                                                        // Radiative: emit fluorescent x-ray.
                                                        if e_trans > photon_cutoff
                                                            && stack_size < PHOTON_CASCADE_STACK_CAP
                                                        {
                                                            stack_e[stack_size as usize] = e_trans;
                                                            stack_dx[stack_size as usize] = iso_dx;
                                                            stack_dy[stack_size as usize] = iso_dy;
                                                            stack_dz[stack_size as usize] = iso_dz;
                                                            stack_px[stack_size as usize] = px;
                                                            stack_py[stack_size as usize] = py;
                                                            stack_pz[stack_size as usize] = pz;
                                                            stack_depth[stack_size as usize] =
                                                                current_depth + 1u32;
                                                            stack_size += 1u32;
                                                        }
                                                    } else if secondary < ar_ns && n_holes < 8u32 {
                                                        // Non-radiative: follow the
                                                        // Auger vacancy's cascade.
                                                        ar_holes[n_holes as usize] = secondary;
                                                        n_holes += 1u32;
                                                    }
                                                } else {
                                                    // No transitions: emit at binding energy.
                                                    let be = ar_binding_energy[ns_idx as usize];
                                                    if be > photon_cutoff
                                                        && stack_size < PHOTON_CASCADE_STACK_CAP
                                                    {
                                                        stack_e[stack_size as usize] = be;
                                                        stack_dx[stack_size as usize] = iso_dx;
                                                        stack_dy[stack_size as usize] = iso_dy;
                                                        stack_dz[stack_size as usize] = iso_dz;
                                                        stack_px[stack_size as usize] = px;
                                                        stack_py[stack_size as usize] = py;
                                                        stack_pz[stack_size as usize] = pz;
                                                        stack_depth[stack_size as usize] =
                                                            current_depth + 1u32;
                                                        stack_size += 1u32;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Commit the scattered photon energy.
                            // Without this assignment the photon
                            // continues at its pre-collision energy
                            // (analog Compton "scatter" with no
                            // energy loss). Symptom: σ_coh / σ_pe
                            // tallies, both of which rise sharply at
                            // lower photon energies, undershoot CPU
                            // by 3-10× because GPU photons never
                            // visit the low-E regime even though the
                            // Klein-Nishina + Doppler branch computed
                            // the correct E_out into `e_out_final`.
                            energy = e_out_final;

                            // -------- TTB sampling (Compton recoil) --------
                            // Convert the recoil electron's kinetic energy
                            // to bremsstrahlung photons, queued onto the
                            // per-thread stack. Inherits the scattered
                            // photon's direction + collision position
                            // (slice-2 simplification -- proper electron
                            // direction is a follow-up). Mirrors the
                            // logic in `yamc/src/bremsstrahlung.rs::thick_target_bremsstrahlung`.
                            let electron_ke = e_in_compton - energy;
                            // Cap cascade at depth 3 -- primary + two
                            // generations of cascade photons trigger
                            // TTB. Bounds the cascade so per-thread
                            // stack (capacity 4) cannot overflow and
                            // low-E flux does not grow unboundedly.
                            if electron_ke > photon_cutoff
                                && ttb_has_data[mat_idx as usize] == 1u32
                                && n_ttb_e >= 2u32
                                && current_depth < 3u32
                            {
                                let log_ke = ln_f64(electron_ke);
                                // Binary search for largest j with
                                // ttb_e_grid_log[j] <= log_ke (matches the
                                // CPU's `lower_bound_index`).
                                let mut j_lo = 0u32;
                                let mut j_hi = n_ttb_e;
                                let mut j_iter = 0u32;
                                while j_iter < 16u32 && j_lo + 1u32 < j_hi {
                                    let j_mid = (j_lo + j_hi) / 2u32;
                                    if ttb_e_grid_log[j_mid as usize] <= log_ke {
                                        j_lo = j_mid;
                                    } else {
                                        j_hi = j_mid;
                                    }
                                    j_iter += 1u32;
                                }
                                let mut j_idx = j_lo;
                                if j_idx >= n_ttb_e - 1u32 {
                                    j_idx = n_ttb_e - 2u32;
                                }

                                let e_l_log = ttb_e_grid_log[j_idx as usize];
                                let e_r_log = ttb_e_grid_log[(j_idx + 1u32) as usize];
                                let denom_e = e_r_log - e_l_log;
                                let mut f_int = 0.0_f64;
                                if denom_e > 1e-30 {
                                    f_int = (log_ke - e_l_log) / denom_e;
                                }
                                let yield_off = mat_idx * n_ttb_e;
                                let y_l = ttb_electron_yield[(yield_off + j_idx) as usize];
                                let y_r = ttb_electron_yield[(yield_off + j_idx + 1u32) as usize];
                                // y in log space; exponentiate after interp.
                                let y = exp_f64(y_l + (y_r - y_l) * f_int);

                                // Sample N = floor(y) + Bernoulli(frac).
                                let d_n = crate::common::pcg32::draw_uniform(state);
                                state = d_n.state;
                                let xi_n = d_n.xi;
                                let n_int = y as u32;
                                let frac_y = y - (n_int as f64);
                                let mut n_photons = n_int;
                                if xi_n < frac_y {
                                    n_photons += 1u32;
                                }

                                // Choose CDF row (j or j+1) -- matches the
                                // CPU sampler's branch where it picks j+1
                                // with probability f_int (and always j+1
                                // when j == 0).
                                let d_ie = crate::common::pcg32::draw_uniform(state);
                                state = d_ie.state;
                                let xi_ie = d_ie.xi;

                                let slab_off = mat_idx * n_ttb_e * n_ttb_e;
                                let mut i_e = j_idx;
                                let mut c_max = 0.0_f64;
                                if xi_ie <= f_int || j_idx == 0u32 {
                                    i_e = j_idx + 1u32;
                                    let p_l = ttb_electron_pdf
                                        [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                    let p_r =
                                        ttb_electron_pdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                                    let c_l = ttb_electron_cdf
                                        [(slab_off + i_e * n_ttb_e + (i_e - 1u32)) as usize];
                                    let a_int = ln_f64(p_r / p_l) / denom_e + 1.0;
                                    let exp_term = exp_f64(a_int * (log_ke - e_l_log)) - 1.0;
                                    c_max = c_l + exp_f64(e_l_log) * p_l / a_int * exp_term;
                                } else {
                                    i_e = j_idx;
                                    c_max =
                                        ttb_electron_cdf[(slab_off + i_e * n_ttb_e + i_e) as usize];
                                }

                                // Sample n_photons photon energies via
                                // inverse CDF, push to per-thread stack.
                                let mut e_lost = electron_ke * 0.0_f64;
                                let mut k_photon = 0u32;
                                while k_photon < n_photons
                                    && stack_size < PHOTON_CASCADE_STACK_CAP
                                    && i_e >= 2u32
                                {
                                    let d_c = crate::common::pcg32::draw_uniform(state);
                                    state = d_c.state;
                                    let xi_c = d_c.xi;
                                    let c = xi_c * c_max;

                                    // Binary search row [0..i_e] for the
                                    // largest i_w with cdf[i_w] <= c.
                                    let cdf_row_off = slab_off + i_e * n_ttb_e;
                                    let mut iw_lo = 0u32;
                                    let mut iw_hi = i_e;
                                    let mut iw_iter = 0u32;
                                    while iw_iter < 16u32 && iw_lo + 1u32 < iw_hi {
                                        let iw_mid = (iw_lo + iw_hi) / 2u32;
                                        if ttb_electron_cdf[(cdf_row_off + iw_mid) as usize] <= c {
                                            iw_lo = iw_mid;
                                        } else {
                                            iw_hi = iw_mid;
                                        }
                                        iw_iter += 1u32;
                                    }
                                    // No clamp on i_w. CPU's
                                    // `lower_bound_index(cdf[..i_e], c)` returns
                                    // i_w in [0, i_e-1]. When `i_w == i_e - 1`
                                    // (the partial last-bin case where the
                                    // photon energy lands between grid[i_e-1]
                                    // and log_ke), the inverse-CDF formula
                                    // pairs the real PDF at i_e-1 with the
                                    // sentinel `pdf[i_e][i_e] = exp(-500)` --
                                    // matches the c_max calculation above
                                    // (very-negative `a` gives a steep
                                    // exponential decay sample). Forcing
                                    // i_w down to i_e-2 broke that branch
                                    // and produced wildly inflated TTB
                                    // contributions for the partial bin.
                                    let i_w = iw_lo;

                                    let w_l_log = ttb_e_grid_log[i_w as usize];
                                    let w_r_log = ttb_e_grid_log[(i_w + 1u32) as usize];
                                    let p_l_w = ttb_electron_pdf[(cdf_row_off + i_w) as usize];
                                    let p_r_w =
                                        ttb_electron_pdf[(cdf_row_off + i_w + 1u32) as usize];
                                    let c_l_w = ttb_electron_cdf[(cdf_row_off + i_w) as usize];
                                    let mut w =
                                        ttb_photon_energy(c, w_l_log, w_r_log, p_l_w, p_r_w, c_l_w);

                                    if w > photon_cutoff && stack_size < PHOTON_CASCADE_STACK_CAP {
                                        if e_lost + w > electron_ke {
                                            w = electron_ke - e_lost;
                                        }
                                        if w > photon_cutoff {
                                            stack_e[stack_size as usize] = w;
                                            stack_dx[stack_size as usize] = e_dx_compton;
                                            stack_dy[stack_size as usize] = e_dy_compton;
                                            stack_dz[stack_size as usize] = e_dz_compton;
                                            stack_px[stack_size as usize] = px;
                                            stack_py[stack_size as usize] = py;
                                            stack_pz[stack_size as usize] = pz;
                                            stack_depth[stack_size as usize] = current_depth + 1u32;
                                            stack_size += 1u32;
                                            e_lost += w;
                                        }
                                    }
                                    k_photon += 1u32;
                                }
                            }
                            // ----------- end TTB sampling -----------
                        } else {
                            // Rejection sampler timed out -- terminate.
                            alive = 0u32;
                        }
                    } else {
                        // Rayleigh (coherent) -- inverse CDF on the
                        // integrated coherent form factor. Then
                        // μ = 1 − 2·x²/x²_max, with a Klein-Nishina-
                        // style angular acceptance 0.5·(1+μ²). Form factor
                        // keyed by the per-collision-selected element slab
                        // (task #72).
                        let n_ff = rayleigh_n_points[elem_slab as usize];
                        let max_ff_rt = 64u32;
                        let ff_off = elem_slab * max_ff_rt;
                        let mut accepted = 0u32;
                        let mut mu = 1.0_f64;
                        let mut k_iter = 0u32;
                        while k_iter < 32u32 && accepted == 0u32 && n_ff >= 2u32 {
                            // One Rayleigh angle proposal via the shared
                            // `rayleigh_propose` helper -- single source of
                            // truth, unit-tested against the CPU
                            // `rayleigh_scatter`. Exactly one iteration of the
                            // old inlined loop (2 draws), so byte-identical.
                            let p = rayleigh_propose(
                                state,
                                energy,
                                ff_off,
                                n_ff,
                                max_ff_rt,
                                rayleigh_x2,
                                rayleigh_cdf,
                            );
                            state = p.state;
                            if p.accepted == 1u32 {
                                mu = p.mu;
                                accepted = 1u32;
                            }
                            k_iter += 1u32;
                        }
                        if accepted == 1u32 {
                            // Rotate direction by (mu, phi) -- same as
                            // the Compton path.
                            let d_phi = crate::common::pcg32::draw_uniform(state);
                            state = d_phi.state;
                            let xi_phi = d_phi.xi;
                            let phi = xi_phi * std::f64::consts::TAU;
                            let cos_phi = cos_f64(phi);
                            let sin_phi = sin_f64(phi);
                            let sqrt_1_mu2 = (1.0 - mu * mu).max(0.0).sqrt();
                            let dz_abs_sq = dz * dz;
                            let denom_dz = (1.0 - dz_abs_sq).max(0.0).sqrt();
                            let mut new_dx = 0.0_f64;
                            let mut new_dy = 0.0_f64;
                            let mut new_dz = 0.0_f64;
                            if denom_dz > 1e-12 {
                                new_dx = mu * dx
                                    + sqrt_1_mu2 * (dx * dz * cos_phi - dy * sin_phi) / denom_dz;
                                new_dy = mu * dy
                                    + sqrt_1_mu2 * (dy * dz * cos_phi + dx * sin_phi) / denom_dz;
                                new_dz = mu * dz - sqrt_1_mu2 * denom_dz * cos_phi;
                            } else {
                                new_dx = sqrt_1_mu2 * cos_phi;
                                new_dy = sqrt_1_mu2 * sin_phi;
                                let mut sign_dz = 1.0_f64;
                                if dz < 0.0 {
                                    sign_dz = -1.0;
                                }
                                new_dz = mu * sign_dz;
                            }
                            let nrm = (new_dx * new_dx + new_dy * new_dy + new_dz * new_dz).sqrt();
                            if nrm > 0.0 {
                                dx = new_dx / nrm;
                                dy = new_dy / nrm;
                                dz = new_dz / nrm;
                            }
                        }
                        // Rayleigh: photon energy unchanged regardless
                        // of accept/reject (rejection-timeout case keeps
                        // the photon alive at its current direction --
                        // equivalent to a small bias toward forward-
                        // scattering, but exceedingly rare).
                    }

                    // 8c. Analog photon-heating deposit for collision-estimator
                    // tallies (task #68). Mirrors the CPU's post-collision
                    // analog dispatch (`score_photon_collision`):
                    //   deposit = (E_in - E_out - banked_secondary_photon_E) · w
                    // where E_out is the surviving photon's energy (0 if it was
                    // absorbed, i.e. alive == 0 after the physics) and the
                    // banked secondary energy is the sum of every secondary
                    // photon this collision pushed onto the cascade stack. This
                    // replaces the σ_heating/Σ_t KERMA score for heating MTs,
                    // which ran ~0.96% high vs the CPU collision estimator.
                    // Binned at the collision energy `heat_e_in` (the energy the
                    // free flight used) with the per-thread `weight`, exactly
                    // like the CPU. Same cell/parent/energy binning as 8a.
                    let mut banked_secondary_e = 0.0_f64;
                    let mut bs = heat_stack_start;
                    while bs < stack_size {
                        banked_secondary_e += stack_e[bs as usize];
                        bs += 1u32;
                    }
                    let mut e_out_heat = 0.0_f64;
                    if alive == 1u32 {
                        e_out_heat = energy;
                    }
                    let heat_deposit = (heat_e_in - e_out_heat - banked_secondary_e) * weight;
                    if heat_deposit != 0.0_f64 {
                        let log_e_heat = ln_f64(heat_e_in);
                        let mut th = 0u32;
                        while th < n_tallies {
                            let cell_bin_h = tally_cell_to_bin[(th * n_cells + cell) as usize];
                            let in_tally_h = cell_bin_h != 4_294_967_295u32;
                            let collision_tally_h = tally_is_collision[th as usize] == 1u32;
                            // Heating / heating-local (MT 301 / 901) only.
                            let mut is_heating_h = false;
                            if tally_score_kinds[th as usize] == 3u32 {
                                let mt_h = tally_score_data[th as usize];
                                if mt_h == 301u32 || mt_h == 901u32 {
                                    is_heating_h = true;
                                }
                            }
                            // D1S parent-nuclide bin (same scan as block 8a).
                            let n_parent_h = tally_n_parent[th as usize];
                            let mut parent_bin_h = 0u32;
                            let mut parent_ok_h = true;
                            if n_parent_h > 1u32 {
                                parent_ok_h = false;
                                let p_off_h = tally_parent_offsets[th as usize];
                                let mut pih = 0u32;
                                while pih < n_parent_h {
                                    if !parent_ok_h
                                        && tally_parent_ids[(p_off_h + pih) as usize] == parent_id
                                    {
                                        parent_bin_h = pih;
                                        parent_ok_h = true;
                                    }
                                    pih += 1u32;
                                }
                            }
                            if in_tally_h && collision_tally_h && is_heating_h && parent_ok_h {
                                let n_bins_h = tally_n_bins[th as usize];
                                let edges_off_h = tally_edges_offsets[th as usize];
                                let lo_edge_h = tally_log_edges[edges_off_h as usize];
                                let hi_edge_h = tally_log_edges[(edges_off_h + n_bins_h) as usize];
                                let in_range_h = log_e_heat >= lo_edge_h && log_e_heat <= hi_edge_h;
                                let mut lo_h = 0u32;
                                let mut hi_h = n_bins_h;
                                let mut iter_h = 0u32;
                                while iter_h < 16u32 && lo_h + 1u32 < hi_h {
                                    let mid_h: u32 = (lo_h + hi_h) / 2u32;
                                    let mid_edge_h =
                                        tally_log_edges[(edges_off_h + mid_h) as usize];
                                    if log_e_heat <= mid_edge_h {
                                        hi_h = mid_h;
                                    } else {
                                        lo_h = mid_h;
                                    }
                                    iter_h += 1u32;
                                }
                                let mut bin_h = lo_h;
                                if bin_h >= n_bins_h {
                                    bin_h = n_bins_h - 1u32;
                                }
                                // Energy function applied as both a gate and a
                                // weight (issues #378, #382), matching the CPU's
                                // `photon_heat_score_ev × ef_weight` arm in
                                // `Tally::score_collision`.
                                //
                                // The analog deposit takes the energy function
                                // but NOT weight/Sigma_t: it is already an eV
                                // value. Off the table drops the whole event
                                // (CPU `get_weight() == None` -> `return`),
                                // hence the AND into `in_range_h` rather than a
                                // zero score.
                                //
                                // `heat_e_in` is the pre-collision photon
                                // energy, which is the same E the CPU passes to
                                // `score_collision` and the same E the energy
                                // filter binned on.
                                let ef_lo_h = tally_efunc_offsets[th as usize];
                                let ef_hi_h = tally_efunc_offsets[(th + 1u32) as usize];
                                let mut ef_in_range_h = true;
                                let mut ef_weight_h = 1.0f64;
                                if ef_hi_h > ef_lo_h {
                                    let n_ef_h = tally_efunc_params[ef_lo_h as usize] as u32;
                                    let e_first_h = tally_efunc_params[(ef_lo_h + 1u32) as usize];
                                    let e_last_h = tally_efunc_params[(ef_lo_h + n_ef_h) as usize];
                                    if heat_e_in < e_first_h || heat_e_in > e_last_h {
                                        ef_in_range_h = false;
                                    } else {
                                        ef_weight_h = energy_function_weight_kernel(
                                            tally_efunc_params,
                                            ef_lo_h,
                                            heat_e_in,
                                        );
                                    }
                                }
                                // Weighted once, here, so every accumulation
                                // path below scores the same value: the
                                // fixed-point atomic path AND the per-history
                                // variance path, which accumulates the raw f64
                                // rather than `bits_h` and would otherwise drop
                                // the weighting (issue #382).
                                let heat_scored_h = heat_deposit * ef_weight_h;
                                let scale_h = tally_fixed_point_scales[th as usize];
                                let scaled_h = (heat_scored_h * scale_h + 0.5) as i64;
                                let bits_h = u64::reinterpret(scaled_h);
                                let out_off_h = tally_out_offsets[th as usize];
                                let cpe_h =
                                    (cell_bin_h * n_parent_h + parent_bin_h) * n_bins_h + bin_h;
                                let tally_idx_h = out_off_h + cpe_h;
                                if in_range_h && ef_in_range_h {
                                    if mesh_direct {
                                        // Issue #234: the analog heating deposit lands
                                        // in the voxel holding the collision point.
                                        if tally_mesh_kind[th as usize] == 0u32 {
                                            src_acc[src_base + tally_idx_h as usize]
                                                .fetch_add(bits_h);
                                        } else {
                                            let n_mesh_h = tally_n_mesh[th as usize];
                                            let mo_h = tally_mesh_params_offsets[th as usize];
                                            // Cylindrical (kind 3) vs rectangular
                                            // (1/2) point binning (issue #279).
                                            let voxel_h = if tally_mesh_kind[th as usize] == 3u32 {
                                                cyl_mesh_bin_at_kernel(
                                                    tally_mesh_params,
                                                    mo_h,
                                                    px,
                                                    py,
                                                    pz,
                                                )
                                            } else {
                                                rect_mesh_bin_at_kernel(
                                                    tally_mesh_params,
                                                    mo_h,
                                                    px,
                                                    py,
                                                    pz,
                                                )
                                            };
                                            if voxel_h != 4_294_967_295u32 {
                                                let base_h = out_off_h + cpe_h * n_mesh_h;
                                                src_acc[src_base + (base_h + voxel_h) as usize]
                                                    .fetch_add(bits_h);
                                            }
                                        }
                                    } else if per_history_var {
                                        let mut found = false;
                                        let mut j = 0u32;
                                        while j < th_count {
                                            if th_bin[j as usize] == tally_idx_h {
                                                th_val[j as usize] += heat_scored_h;
                                                found = true;
                                            }
                                            j += 1u32;
                                        }
                                        if !found {
                                            if th_count < PERHIST_K {
                                                th_bin[th_count as usize] = tally_idx_h;
                                                th_val[th_count as usize] = heat_scored_h;
                                                th_count += 1u32;
                                            } else {
                                                let mut sfound = false;
                                                let mut k = 0u32;
                                                while k < spill_count {
                                                    if spill_bin[spill_base + k as usize]
                                                        == tally_idx_h
                                                    {
                                                        spill_val[spill_base + k as usize] +=
                                                            heat_scored_h;
                                                        sfound = true;
                                                    }
                                                    k += 1u32;
                                                }
                                                if !sfound {
                                                    spill_bin[spill_base + spill_count as usize] =
                                                        tally_idx_h;
                                                    spill_val[spill_base + spill_count as usize] =
                                                        heat_scored_h;
                                                    spill_count += 1u32;
                                                }
                                            }
                                        }
                                    } else {
                                        tally_out[tally_idx_h as usize].fetch_add(bits_h);
                                    }
                                }
                            }
                            th += 1u32;
                        }
                    }
                } else {
                    // Surface crossing. Apply boundary semantics.
                    let boundary = surface_boundaries[winner_surface as usize];
                    if boundary == 1u32 {
                        // PHOTON_BOUNDARY_VACUUM
                        alive = 0u32;
                    }
                    // Nudge across so we don't re-find the same surface.
                    px += dx * 1e-9_f64;
                    py += dy * 1e-9_f64;
                    pz += dz * 1e-9_f64;
                }
            }
        } // closes `if keep_going == 1u32 {`

        step += 1u32;
        n_steps += 1u32;
    }

    // Batch-free per-history / per-source variance flush (issue #233 Stage 3),
    // mirroring the neutron kernel. PerHistory: per-bin `sum` (first half of
    // `tally_out`) + `sum_sq` (second half). PerSource: per-bin `sum` into this
    // history's source row of `src_acc` (`sum_sq` done host-side per source). The
    // owning tally `t` is recovered from the flat bin index via
    // `tally_out_offsets`. One atomic add per touched/spill bin.
    if per_history_var {
        let sq_off = (tally_out.len() / 2) as u32;
        let sq_anchor = crate::common::tallies::SUMSQ_SCALE_ANCHOR;
        let sq_ceiling = crate::common::tallies::FIXED_POINT_ACC_CEILING;
        let kerma_scale = crate::common::tallies::KERMA_FIXED_POINT_SCALE;
        let kerma_sumsq = crate::common::tallies::KERMA_SUMSQ_SCALE;
        // Touched-list (registers).
        let mut j = 0u32;
        while j < th_count {
            let idx = th_bin[j as usize];
            let x = th_val[j as usize];
            let mut t = 0u32;
            let mut tt = 0u32;
            while tt < n_tallies {
                if idx >= tally_out_offsets[tt as usize] {
                    t = tt;
                }
                tt += 1u32;
            }
            let s = tally_fixed_point_scales[t as usize];
            let sx = x * s;
            let sbits = if sx >= 0.0 {
                (sx + 0.5) as i64
            } else {
                -((-sx + 0.5) as i64)
            };
            if per_source_var {
                src_acc[src_base + idx as usize].fetch_add(u64::reinterpret(sbits));
            } else {
                tally_out[idx as usize].fetch_add(u64::reinterpret(sbits));
                let s_sq = if s <= kerma_scale {
                    kerma_sumsq
                } else {
                    let s2 = s * s / sq_anchor;
                    if s2 < sq_ceiling {
                        s2
                    } else {
                        sq_ceiling
                    }
                };
                let qx = x * x * s_sq;
                let qbits = (qx + 0.5) as i64;
                tally_out[(sq_off + idx) as usize].fetch_add(u64::reinterpret(qbits));
            }
            j += 1u32;
        }
        // Spill (per-history global overflow, exact for any distinct-bin count).
        let mut k = 0u32;
        while k < spill_count {
            let idx = spill_bin[spill_base + k as usize];
            let x = spill_val[spill_base + k as usize];
            let mut t = 0u32;
            let mut tt = 0u32;
            while tt < n_tallies {
                if idx >= tally_out_offsets[tt as usize] {
                    t = tt;
                }
                tt += 1u32;
            }
            let s = tally_fixed_point_scales[t as usize];
            let sx = x * s;
            let sbits = if sx >= 0.0 {
                (sx + 0.5) as i64
            } else {
                -((-sx + 0.5) as i64)
            };
            if per_source_var {
                src_acc[src_base + idx as usize].fetch_add(u64::reinterpret(sbits));
            } else {
                tally_out[idx as usize].fetch_add(u64::reinterpret(sbits));
                let s_sq = if s <= kerma_scale {
                    kerma_sumsq
                } else {
                    let s2 = s * s / sq_anchor;
                    if s2 < sq_ceiling {
                        s2
                    } else {
                        sq_ceiling
                    }
                };
                let qx = x * x * s_sq;
                let qbits = (qx + 0.5) as i64;
                tally_out[(sq_off + idx) as usize].fetch_add(u64::reinterpret(qbits));
            }
            k += 1u32;
        }
    }

    out_alive[ABSOLUTE_POS] = alive;
    out_n_steps[ABSOLUTE_POS] = n_steps;
    out_final_energy[ABSOLUTE_POS] = energy;
    let _ = exp_f64(0.0);
}

/// Aggregate result of a GPU photon transport launch -- same shape
/// as the neutron `MultiCellResult`.
#[derive(Debug, Clone)]
pub struct PhotonMultiCellResult {
    pub n_cells: usize,
    pub alive: Vec<u32>,
    pub n_steps: Vec<u32>,
    pub final_energies: Vec<f64>,
    /// Per-tally fixed-point-decoded outputs:
    /// `tally_outputs[t][bin_cell * n_bins + bin_e]` = value. In `PerHistory`
    /// mode this is the per-bin `sum`; in `PerSource` it is empty (the tally is
    /// reconstructed from `src_acc`).
    pub tally_outputs: Vec<Vec<f64>>,
    /// Per-tally per-bin sum-of-squares (issue #233 Stage 3), non-empty only in
    /// `PerHistory` mode. Same shape/order as `tally_outputs`.
    pub tally_sum_sq: Vec<Vec<f64>>,
    /// Raw per-source accumulator (issue #233 Stage 3, `PerSource` mode only):
    /// flat `chunk_sources * total_out_len` fixed-point words, this launch's
    /// per-`(source, flat_bin)` sum. Empty otherwise.
    pub src_acc: Vec<u64>,
    /// Photons that ended in no cell, i.e. lost particles (issue #289). The
    /// dispatch enforces `max_lost_particles` from `count`, matching the CPU.
    pub lost: crate::common::lost_particles::LostParticleResult,
}

fn unpack_photon_tally_outputs(bits: &[u64], tallies: &TalliesPack) -> Vec<Vec<f64>> {
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

/// Unpack the second (sum-of-squares) half of a per-history photon `tally_out`
/// buffer, at each tally's derived sum-of-squares scale (issue #233 Stage 3).
/// Mirror of the neutron `unpack_tally_sum_sq`.
fn unpack_photon_tally_sum_sq(bits: &[u64], tallies: &TalliesPack) -> Vec<Vec<f64>> {
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

/// Launch the GPU photon transport kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_cell_photon_transport(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    // Per-particle initial weight (length == `seeds.len()`). Fresh
    // sources pass all-1.0; banked coupled photons carry the emitting
    // neutron's weight. Applied at every tally score site.
    weights_in: &[f64],
    // Per-particle D1S parent-nuclide id (length == `seeds.len()`). `0` (no
    // parent) for prompt / fresh-source photons. Carried as a per-thread
    // register so every secondary the photon spawns inherits it; used by the
    // `parent_nuclides` tally binning. Fresh-source photons pass all-zero, so
    // the parent dimension collapses to bin 0 and scoring is byte-identical.
    parent_in: &[u32],
    positions_in: &[f64],
    directions_in: &[f64],
    cell_aabbs: &[f64],
    cell_to_material: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    surface_boundaries: &[u32],
    region_program: &[u32],
    log_energy_grid: &[f64],
    xs_total_per_material: &[f64],
    xs_coherent_per_material: &[f64],
    xs_incoherent_per_material: &[f64],
    xs_photoelectric_per_material: &[f64],
    xs_pair_per_material: &[f64],
    heating_xs_per_material: &[f64],
    rayleigh_x2: &[f64],
    rayleigh_cdf: &[f64],
    rayleigh_n_points: &[u32],
    ttb_e_grid_log: &[f64],
    ttb_electron_pdf: &[f64],
    ttb_electron_cdf: &[f64],
    ttb_electron_yield: &[f64],
    ttb_has_data: &[u32],
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    dop_subshell_idx: &[u32],
    dop_subshell_w0: &[f64],
    dop_subshell_cnt: &[u32],
    iff_x: &[f64],
    iff_s: &[f64],
    iff_n_points: &[u32],
    iff_has_data: &[u32],
    ar_has_data: &[u32],
    ar_n_shells: &[u32],
    ar_binding_energy: &[f64],
    ar_pe_subshell_xs_log: &[f64],
    ar_n_trans: &[u32],
    ar_trans_primary: &[u32],
    ar_trans_secondary: &[u32],
    ar_trans_energy: &[f64],
    ar_trans_cum_prob: &[f64],
    ttb_positron_pdf: &[f64],
    ttb_positron_cdf: &[f64],
    ttb_positron_yield: &[f64],
    pair_has_data: &[u32],
    pair_r_z: &[f64],
    pair_a: &[f64],
    pair_c: &[f64],
    // Per-collision element-selection inputs (task #72). See the kernel
    // signature docs: `elem_macro_total` is the per-(element-slab, energy)
    // macro-total weight table; `mat_elem_meta` is stride-2 `[offset, count]`
    // per material.
    elem_macro_total: &[f64],
    mat_elem_meta: &[u32],
    tallies: &TalliesPack,
    max_steps: u32,
    // `Model::photon_cutoff_energy` in eV (issue #286). Photons at or below it
    // are killed at the top of the step and never emitted as secondaries, the
    // same threshold the CPU applies. The GPU used to hardcode the 1 keV
    // default, so a non-default cutoff silently did nothing here.
    photon_cutoff_energy: f64,
    // Tally variance mode (issue #233 Stage 3): `PerStep` is the byte-identical
    // pre-#233 path; `PerHistory` flushes per-history sum + sum_sq into a doubled
    // `tally_out`; `PerSource` scatters per-history sums into `src_acc` keyed by
    // `source_idx` (for coupled/mixed/D1S secondary/decay photons).
    variance: TallyVarianceMode,
) -> PhotonMultiCellResult {
    assert!(
        ctx.max_storage_buffers_per_stage >= PHOTON_KERNEL_STORAGE_BUFFER_COUNT,
        "GPU adapter supports {} storage buffers per stage, but \
         multi_cell_photon_transport_kernel needs {}. Bindings past \
         the limit are silently aliased -- the kernel would produce \
         numerically-plausible but wrong tally results. Adapter: \
         {:?}",
        ctx.max_storage_buffers_per_stage,
        PHOTON_KERNEL_STORAGE_BUFFER_COUNT,
        ctx.adapter_info(),
    );

    let n = seeds.len();
    assert_eq!(
        weights_in.len(),
        n,
        "weights_in must have one entry per source particle"
    );
    let n_cells = cell_aabbs.len() / 6;
    let n_grid = log_energy_grid.len();
    let n_materials = xs_total_per_material.len() / n_grid.max(1);
    assert_eq!(
        cell_to_material.len(),
        n_cells,
        "cell_to_material must have one entry per cell"
    );
    assert_eq!(
        surface_boundaries.len(),
        surface_types.len(),
        "surface_boundaries length must match surface_types"
    );
    assert_eq!(
        xs_coherent_per_material.len(),
        n_materials * n_grid,
        "xs_coherent_per_material must be [n_materials × n_grid]"
    );
    assert_eq!(
        heating_xs_per_material.len(),
        n_materials * n_grid,
        "heating_xs_per_material must be [n_materials × n_grid]"
    );
    // Rayleigh / form-factor / relaxation / pair packs are now keyed by the
    // per-collision-selected element slab (task #72), so their leading
    // dimension is the total element count `n_slab`, not `n_materials`.
    let n_slab = rayleigh_n_points.len();
    assert_eq!(rayleigh_x2.len(), n_slab * MAX_RAYLEIGH_FF);
    assert_eq!(rayleigh_cdf.len(), n_slab * MAX_RAYLEIGH_FF);
    assert_eq!(
        mat_elem_meta.len(),
        n_materials * 2,
        "mat_elem_meta must be stride-2 [offset, count] per material"
    );
    assert_eq!(
        elem_macro_total.len(),
        (n_slab.max(1)) * n_grid,
        "elem_macro_total must be [n_slab × n_grid]"
    );

    let (bvh_aabbs, bvh_meta, bvh_prim_indices, bvh_unbounded) = build_and_flatten_bvh(cell_aabbs);
    let client = ctx.client();
    // Start each launch from a fully-idle, cleaned client - the shared cubecl
    // client is a process-global singleton, so back-to-back launches otherwise
    // accumulate state that intermittently corrupts results. See the neutron
    // launcher and project_gpu_photon_kernel_race notes.
    let _ = pollster::block_on(client.sync());
    client.memory_cleanup();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_h = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let weights_h = client.create_from_slice(bytemuck::cast_slice(weights_in));
    let parent_h = client.create_from_slice(bytemuck::cast_slice(parent_in));
    let positions_h = client.create_from_slice(bytemuck::cast_slice(positions_in));
    let dirs_h = client.create_from_slice(bytemuck::cast_slice(directions_in));
    let aabbs_h = client.create_from_slice(bytemuck::cast_slice(cell_aabbs));
    let cell_mat_h = client.create_from_slice(bytemuck::cast_slice(cell_to_material));
    let bvh_aabbs_data: &[f64] = if bvh_aabbs.is_empty() {
        &[0.0; 6]
    } else {
        &bvh_aabbs
    };
    let bvh_meta_data: &[u32] = if bvh_meta.is_empty() {
        &[0u32; 3]
    } else {
        &bvh_meta
    };
    let bvh_prim_data: &[u32] = if bvh_prim_indices.is_empty() {
        &[0u32]
    } else {
        &bvh_prim_indices
    };
    let bvh_unb_data: &[u32] = if bvh_unbounded.is_empty() {
        &[0u32]
    } else {
        &bvh_unbounded
    };
    let bvh_aabbs_h = client.create_from_slice(bytemuck::cast_slice(bvh_aabbs_data));
    let bvh_meta_h = client.create_from_slice(bytemuck::cast_slice(bvh_meta_data));
    let bvh_prim_h = client.create_from_slice(bytemuck::cast_slice(bvh_prim_data));
    let bvh_unb_h = client.create_from_slice(bytemuck::cast_slice(bvh_unb_data));
    let stypes_h = client.create_from_slice(bytemuck::cast_slice(surface_types));
    let sparams_h = client.create_from_slice(bytemuck::cast_slice(surface_params));
    let sboundaries_h = client.create_from_slice(bytemuck::cast_slice(surface_boundaries));
    // Per-cell CSG region program. Padded to one dummy word when empty (cubecl
    // rejects zero-length buffers); the kernel reads it only when n_cells > 0.
    let region_program_data: &[u32] = if region_program.is_empty() {
        &[0u32]
    } else {
        region_program
    };
    let region_program_h = client.create_from_slice(bytemuck::cast_slice(region_program_data));
    let log_grid_h = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_t_h = client.create_from_slice(bytemuck::cast_slice(xs_total_per_material));
    let xs_c_h = client.create_from_slice(bytemuck::cast_slice(xs_coherent_per_material));
    let xs_i_h = client.create_from_slice(bytemuck::cast_slice(xs_incoherent_per_material));
    let xs_ph_h = client.create_from_slice(bytemuck::cast_slice(xs_photoelectric_per_material));
    let xs_p_h = client.create_from_slice(bytemuck::cast_slice(xs_pair_per_material));
    let xs_h_h = client.create_from_slice(bytemuck::cast_slice(heating_xs_per_material));
    let ray_x2_h = client.create_from_slice(bytemuck::cast_slice(rayleigh_x2));
    let ray_cdf_h = client.create_from_slice(bytemuck::cast_slice(rayleigh_cdf));
    let ray_np_h = client.create_from_slice(bytemuck::cast_slice(rayleigh_n_points));
    let tally_score_kinds_h = client.create_from_slice(bytemuck::cast_slice(&tallies.score_kinds));
    let tally_cell_to_bin_h = client.create_from_slice(bytemuck::cast_slice(&tallies.cell_to_bin));
    let tally_n_cells_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.n_cells_per_tally));
    let tally_edges_offsets_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.edges_offsets));
    let tally_n_bins_h = client.create_from_slice(bytemuck::cast_slice(&tallies.n_bins_per_tally));
    let tally_log_edges_h = client.create_from_slice(bytemuck::cast_slice(&tallies.log_edges));
    let tally_out_offsets_h = client.create_from_slice(bytemuck::cast_slice(&tallies.out_offsets));
    let tally_score_data_h = client.create_from_slice(bytemuck::cast_slice(&tallies.score_data));
    let tally_fixed_point_scales_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.fixed_point_scales));
    let tally_is_collision_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.is_collision));
    let tally_n_parent_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.n_parent_per_tally));
    let tally_parent_offsets_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.parent_offsets));
    // `parent_ids` is empty when no tally has a `parent_nuclides` filter;
    // cubecl rejects zero-length bindings, so pad to a single dummy id (never
    // read because every such tally has n_parent == 1).
    let parent_ids_data: &[u32] = if tallies.parent_ids.is_empty() {
        &[0u32]
    } else {
        &tallies.parent_ids
    };
    let tally_parent_ids_h = client.create_from_slice(bytemuck::cast_slice(parent_ids_data));
    // Mesh (voxel) tally buffers (issue #234). `mesh_params` is padded to a
    // single dummy slot when no tally carries a mesh (cubecl rejects zero-length
    // buffers; the kernel only reads it when `mesh_direct` and the tally's kind
    // is non-`MESH_NONE`).
    let tally_n_mesh_h = client.create_from_slice(bytemuck::cast_slice(&tallies.n_mesh_per_tally));
    let tally_mesh_kind_h = client.create_from_slice(bytemuck::cast_slice(&tallies.mesh_kind));
    let tally_mesh_params_offsets_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.mesh_params_offsets));
    let mesh_params_padded: Vec<f64> = if tallies.mesh_params.is_empty() {
        vec![0.0]
    } else {
        tallies.mesh_params.clone()
    };
    let tally_mesh_params_h = client.create_from_slice(bytemuck::cast_slice(&mesh_params_padded));
    // Energy-function tally buffers (issue #271), padded like the mesh params:
    // with no `energy_function=` tally the params buffer would be zero-length,
    // which cubecl rejects. Offsets stay all-equal so every range is empty.
    let tally_efunc_offsets_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.efunc_offsets));
    let efunc_params_padded: Vec<f64> = if tallies.efunc_params.is_empty() {
        vec![0.0]
    } else {
        tallies.efunc_params.clone()
    };
    let tally_efunc_params_h = client.create_from_slice(bytemuck::cast_slice(&efunc_params_padded));
    let ttb_e_grid_log_h = client.create_from_slice(bytemuck::cast_slice(ttb_e_grid_log));
    let ttb_electron_pdf_h = client.create_from_slice(bytemuck::cast_slice(ttb_electron_pdf));
    let ttb_electron_cdf_h = client.create_from_slice(bytemuck::cast_slice(ttb_electron_cdf));
    let ttb_electron_yield_h = client.create_from_slice(bytemuck::cast_slice(ttb_electron_yield));
    let ttb_has_data_h = client.create_from_slice(bytemuck::cast_slice(ttb_has_data));
    let dop_pz_grid_h = client.create_from_slice(bytemuck::cast_slice(dop_pz_grid));
    let dop_electron_pdf_h = client.create_from_slice(bytemuck::cast_slice(dop_electron_pdf));
    let dop_binding_energy_h = client.create_from_slice(bytemuck::cast_slice(dop_binding_energy));
    let dop_profile_pdf_h = client.create_from_slice(bytemuck::cast_slice(dop_profile_pdf));
    let dop_profile_cdf_h = client.create_from_slice(bytemuck::cast_slice(dop_profile_cdf));
    let dop_n_shells_h = client.create_from_slice(bytemuck::cast_slice(dop_n_shells));
    let dop_has_data_h = client.create_from_slice(bytemuck::cast_slice(dop_has_data));
    let dop_subshell_idx_h = client.create_from_slice(bytemuck::cast_slice(dop_subshell_idx));
    let dop_subshell_w0_h = client.create_from_slice(bytemuck::cast_slice(dop_subshell_w0));
    let dop_subshell_cnt_h = client.create_from_slice(bytemuck::cast_slice(dop_subshell_cnt));
    let iff_x_h = client.create_from_slice(bytemuck::cast_slice(iff_x));
    let iff_s_h = client.create_from_slice(bytemuck::cast_slice(iff_s));
    let iff_n_points_h = client.create_from_slice(bytemuck::cast_slice(iff_n_points));
    let iff_has_data_h = client.create_from_slice(bytemuck::cast_slice(iff_has_data));
    let ar_has_data_h = client.create_from_slice(bytemuck::cast_slice(ar_has_data));
    let ar_n_shells_h = client.create_from_slice(bytemuck::cast_slice(ar_n_shells));
    let ar_binding_energy_h = client.create_from_slice(bytemuck::cast_slice(ar_binding_energy));
    let ar_pe_subshell_xs_log_h =
        client.create_from_slice(bytemuck::cast_slice(ar_pe_subshell_xs_log));
    let ar_n_trans_h = client.create_from_slice(bytemuck::cast_slice(ar_n_trans));
    let ar_trans_primary_h = client.create_from_slice(bytemuck::cast_slice(ar_trans_primary));
    let ar_trans_secondary_h = client.create_from_slice(bytemuck::cast_slice(ar_trans_secondary));
    let ar_trans_energy_h = client.create_from_slice(bytemuck::cast_slice(ar_trans_energy));
    let ar_trans_cum_prob_h = client.create_from_slice(bytemuck::cast_slice(ar_trans_cum_prob));
    let ttb_positron_pdf_h = client.create_from_slice(bytemuck::cast_slice(ttb_positron_pdf));
    let ttb_positron_cdf_h = client.create_from_slice(bytemuck::cast_slice(ttb_positron_cdf));
    let ttb_positron_yield_h = client.create_from_slice(bytemuck::cast_slice(ttb_positron_yield));
    let pair_has_data_h = client.create_from_slice(bytemuck::cast_slice(pair_has_data));
    let pair_r_z_h = client.create_from_slice(bytemuck::cast_slice(pair_r_z));
    let pair_a_h = client.create_from_slice(bytemuck::cast_slice(pair_a));
    let pair_c_h = client.create_from_slice(bytemuck::cast_slice(pair_c));
    let elem_macro_total_h = client.create_from_slice(bytemuck::cast_slice(elem_macro_total));
    let mat_elem_meta_h = client.create_from_slice(bytemuck::cast_slice(mat_elem_meta));

    // Per-run scalars (issue #286): slot 0 = the model photon cutoff.
    let run_params_v = vec![photon_cutoff_energy; PHOTON_RUN_PARAMS_LEN];
    let run_params_h = client.create_from_slice(bytemuck::cast_slice(&run_params_v));

    let out_alive_h = client.empty(std::mem::size_of_val(seeds));
    let out_steps_h = client.empty(std::mem::size_of_val(seeds));
    let out_e_h = client.empty(std::mem::size_of_val(energies_in));
    let total_out_len = tallies.total_out_len() as usize;
    // Batch-free variance buffers (issue #233 Stage 3), mirroring the neutron
    // host. PerHistory doubles `tally_out` (2nd half = sum_sq); PerSource writes
    // `src_acc` instead (size-1 `tally_out` dummy). Both per-history modes share
    // the thread-private touched-list + spill.
    let per_history = variance.per_history();
    let per_source = variance.per_source();
    // Issue #234 mesh path: direct-to-`src_acc` scoring (no touched-list). Shares
    // the `src_acc` read-back and per-source finalize with `PerSource`.
    let mesh_direct = variance.mesh_direct();
    let uses_src_acc = variance.uses_src_acc();
    // Same proven per-history bound as the neutron host: at most one flat bin per
    // tally per step over `max_steps` steps (issue #233 Stage 4). The mesh path
    // accumulates straight into `src_acc` (no touched-list), so it needs no spill.
    let spill_cap = if mesh_direct {
        0
    } else {
        per_history_spill_cap(total_out_len, max_steps, tallies.n_tallies() as usize)
    };
    let alloc_out_len = match variance {
        TallyVarianceMode::PerHistory => total_out_len * 2,
        // PerSource and PerSourceDirect (issue #234 mesh) both write `src_acc`
        // instead of `tally_out`, so the `tally_out` dummy is size 1.
        TallyVarianceMode::PerSource { .. } | TallyVarianceMode::PerSourceDirect { .. } => 1,
        TallyVarianceMode::PerStep => total_out_len,
    };
    let initial_tally: Vec<u64> = vec![0u64; alloc_out_len.max(1)];
    let tally_out_h = client.create_from_slice(bytemuck::cast_slice(&initial_tally));
    let spill_len = if per_history {
        n.saturating_mul(spill_cap).max(1)
    } else {
        1
    };
    // Uninitialized: the kernel writes every spill slot before reading it
    // (guarded by `spill_count`), so no zero-init / upload is needed (issue #233
    // Stage 4). Mirrors the neutron host.
    let spill_bin_h = client.empty(spill_len * std::mem::size_of::<u32>());
    let spill_val_h = client.empty(spill_len * std::mem::size_of::<f64>());
    let total_bins = variance.total_bins() as usize;
    let src_acc_len = match variance {
        TallyVarianceMode::PerSource { chunk_sources, .. }
        | TallyVarianceMode::PerSourceDirect { chunk_sources, .. } => {
            (chunk_sources as usize).saturating_mul(total_bins).max(1)
        }
        _ => 1,
    };
    let src_acc_h = client.create_from_slice(bytemuck::cast_slice(&vec![0u64; src_acc_len]));
    // Lost-particle diagnostics (issue #289): counter + capped record buffer.
    let lost_count_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));
    let lost_records_z = vec![
        0.0_f64;
        crate::common::lost_particles::LOST_RECORD_CAPACITY
            * crate::common::lost_particles::LOST_F64_STRIDE
    ];
    let lost_f64_h = client.create_from_slice(bytemuck::cast_slice(&lost_records_z));
    let source_idx_vec: Vec<u32> = match variance {
        TallyVarianceMode::PerSource {
            source_idx: Some(si),
            ..
        }
        | TallyVarianceMode::PerSourceDirect {
            source_idx: Some(si),
            ..
        } => si.to_vec(),
        TallyVarianceMode::PerSource {
            source_idx: None, ..
        }
        | TallyVarianceMode::PerSourceDirect {
            source_idx: None, ..
        } => (0..n as u32).collect(),
        _ => vec![0u32; 1],
    };
    let source_idx_len = source_idx_vec.len();
    let source_idx_h = client.create_from_slice(bytemuck::cast_slice(&source_idx_vec));

    // WORKGROUP_SIZE = 128 keeps `groups = n/128` under the Vulkan
    // per-dimension limit (65535) up to ~8.4M particles per launch --
    // headroom for verification-suite batch sizes (5M is typical).
    // Bumping to 128 from 64 was a one-line fix; if we need more
    // headroom later, split into chunks via repeated launches.
    const WORKGROUP_SIZE: u32 = 128;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        multi_cell_photon_transport_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(energies_h, n),
            BufferArg::from_raw_parts(weights_h, n),
            BufferArg::from_raw_parts(parent_h, n),
            BufferArg::from_raw_parts(positions_h, positions_in.len()),
            BufferArg::from_raw_parts(dirs_h, directions_in.len()),
            BufferArg::from_raw_parts(aabbs_h, cell_aabbs.len()),
            BufferArg::from_raw_parts(cell_mat_h, cell_to_material.len()),
            BufferArg::from_raw_parts(bvh_aabbs_h, bvh_aabbs_data.len()),
            BufferArg::from_raw_parts(bvh_meta_h, bvh_meta_data.len()),
            BufferArg::from_raw_parts(bvh_prim_h, bvh_prim_data.len()),
            BufferArg::from_raw_parts(bvh_unb_h, bvh_unb_data.len()),
            BufferArg::from_raw_parts(stypes_h, surface_types.len()),
            BufferArg::from_raw_parts(sparams_h, surface_params.len()),
            BufferArg::from_raw_parts(sboundaries_h, surface_boundaries.len()),
            BufferArg::from_raw_parts(region_program_h, region_program_data.len()),
            BufferArg::from_raw_parts(log_grid_h, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_t_h, xs_total_per_material.len()),
            BufferArg::from_raw_parts(xs_c_h, xs_coherent_per_material.len()),
            BufferArg::from_raw_parts(xs_i_h, xs_incoherent_per_material.len()),
            BufferArg::from_raw_parts(xs_ph_h, xs_photoelectric_per_material.len()),
            BufferArg::from_raw_parts(xs_p_h, xs_pair_per_material.len()),
            BufferArg::from_raw_parts(xs_h_h, heating_xs_per_material.len()),
            BufferArg::from_raw_parts(ray_x2_h, rayleigh_x2.len()),
            BufferArg::from_raw_parts(ray_cdf_h, rayleigh_cdf.len()),
            BufferArg::from_raw_parts(ray_np_h, rayleigh_n_points.len()),
            BufferArg::from_raw_parts(tally_score_kinds_h, tallies.score_kinds.len()),
            BufferArg::from_raw_parts(tally_cell_to_bin_h, tallies.cell_to_bin.len()),
            BufferArg::from_raw_parts(tally_n_cells_h, tallies.n_cells_per_tally.len()),
            BufferArg::from_raw_parts(tally_edges_offsets_h, tallies.edges_offsets.len()),
            BufferArg::from_raw_parts(tally_n_bins_h, tallies.n_bins_per_tally.len()),
            BufferArg::from_raw_parts(tally_log_edges_h, tallies.log_edges.len()),
            BufferArg::from_raw_parts(tally_out_offsets_h, tallies.out_offsets.len()),
            BufferArg::from_raw_parts(tally_score_data_h, tallies.score_data.len()),
            BufferArg::from_raw_parts(tally_fixed_point_scales_h, tallies.fixed_point_scales.len()),
            BufferArg::from_raw_parts(tally_is_collision_h, tallies.is_collision.len()),
            BufferArg::from_raw_parts(tally_n_parent_h, tallies.n_parent_per_tally.len()),
            BufferArg::from_raw_parts(tally_parent_offsets_h, tallies.parent_offsets.len()),
            BufferArg::from_raw_parts(tally_parent_ids_h, parent_ids_data.len()),
            BufferArg::from_raw_parts(tally_n_mesh_h, tallies.n_mesh_per_tally.len()),
            BufferArg::from_raw_parts(tally_mesh_kind_h, tallies.mesh_kind.len()),
            BufferArg::from_raw_parts(
                tally_mesh_params_offsets_h,
                tallies.mesh_params_offsets.len(),
            ),
            BufferArg::from_raw_parts(tally_mesh_params_h, mesh_params_padded.len()),
            BufferArg::from_raw_parts(tally_efunc_offsets_h, tallies.efunc_offsets.len()),
            BufferArg::from_raw_parts(tally_efunc_params_h, efunc_params_padded.len()),
            BufferArg::from_raw_parts(ttb_e_grid_log_h, ttb_e_grid_log.len()),
            BufferArg::from_raw_parts(ttb_electron_pdf_h, ttb_electron_pdf.len()),
            BufferArg::from_raw_parts(ttb_electron_cdf_h, ttb_electron_cdf.len()),
            BufferArg::from_raw_parts(ttb_electron_yield_h, ttb_electron_yield.len()),
            BufferArg::from_raw_parts(ttb_has_data_h, ttb_has_data.len()),
            BufferArg::from_raw_parts(dop_pz_grid_h, dop_pz_grid.len()),
            BufferArg::from_raw_parts(dop_electron_pdf_h, dop_electron_pdf.len()),
            BufferArg::from_raw_parts(dop_binding_energy_h, dop_binding_energy.len()),
            BufferArg::from_raw_parts(dop_profile_pdf_h, dop_profile_pdf.len()),
            BufferArg::from_raw_parts(dop_profile_cdf_h, dop_profile_cdf.len()),
            BufferArg::from_raw_parts(dop_n_shells_h, dop_n_shells.len()),
            BufferArg::from_raw_parts(dop_has_data_h, dop_has_data.len()),
            BufferArg::from_raw_parts(dop_subshell_idx_h, dop_subshell_idx.len()),
            BufferArg::from_raw_parts(dop_subshell_w0_h, dop_subshell_w0.len()),
            BufferArg::from_raw_parts(dop_subshell_cnt_h, dop_subshell_cnt.len()),
            BufferArg::from_raw_parts(iff_x_h, iff_x.len()),
            BufferArg::from_raw_parts(iff_s_h, iff_s.len()),
            BufferArg::from_raw_parts(iff_n_points_h, iff_n_points.len()),
            BufferArg::from_raw_parts(iff_has_data_h, iff_has_data.len()),
            BufferArg::from_raw_parts(ar_has_data_h, ar_has_data.len()),
            BufferArg::from_raw_parts(ar_n_shells_h, ar_n_shells.len()),
            BufferArg::from_raw_parts(ar_binding_energy_h, ar_binding_energy.len()),
            BufferArg::from_raw_parts(ar_pe_subshell_xs_log_h, ar_pe_subshell_xs_log.len()),
            BufferArg::from_raw_parts(ar_n_trans_h, ar_n_trans.len()),
            BufferArg::from_raw_parts(ar_trans_primary_h, ar_trans_primary.len()),
            BufferArg::from_raw_parts(ar_trans_secondary_h, ar_trans_secondary.len()),
            BufferArg::from_raw_parts(ar_trans_energy_h, ar_trans_energy.len()),
            BufferArg::from_raw_parts(ar_trans_cum_prob_h, ar_trans_cum_prob.len()),
            BufferArg::from_raw_parts(ttb_positron_pdf_h, ttb_positron_pdf.len()),
            BufferArg::from_raw_parts(ttb_positron_cdf_h, ttb_positron_cdf.len()),
            BufferArg::from_raw_parts(ttb_positron_yield_h, ttb_positron_yield.len()),
            BufferArg::from_raw_parts(pair_has_data_h, pair_has_data.len()),
            BufferArg::from_raw_parts(pair_r_z_h, pair_r_z.len()),
            BufferArg::from_raw_parts(pair_a_h, pair_a.len()),
            BufferArg::from_raw_parts(pair_c_h, pair_c.len()),
            BufferArg::from_raw_parts(elem_macro_total_h, elem_macro_total.len()),
            BufferArg::from_raw_parts(mat_elem_meta_h, mat_elem_meta.len()),
            BufferArg::from_raw_parts(run_params_h, run_params_v.len()),
            BufferArg::from_raw_parts(out_alive_h.clone(), n),
            BufferArg::from_raw_parts(out_steps_h.clone(), n),
            BufferArg::from_raw_parts(out_e_h.clone(), n),
            BufferArg::from_raw_parts(tally_out_h.clone(), alloc_out_len.max(1)),
            BufferArg::from_raw_parts(spill_bin_h.clone(), spill_len),
            BufferArg::from_raw_parts(spill_val_h.clone(), spill_len),
            BufferArg::from_raw_parts(source_idx_h.clone(), source_idx_len),
            BufferArg::from_raw_parts(src_acc_h.clone(), src_acc_len),
            BufferArg::from_raw_parts(lost_count_h.clone(), 1),
            BufferArg::from_raw_parts(lost_f64_h.clone(), lost_records_z.len()),
            max_steps,
            per_history,
            spill_cap as u32,
            per_source,
            total_bins as u32,
            mesh_direct,
        );
    }

    let alive: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_alive_h).unwrap()).to_vec();
    let n_steps: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_steps_h).unwrap()).to_vec();
    let final_energies: Vec<f64> =
        bytemuck::cast_slice(&client.read_one(out_e_h).unwrap()).to_vec();
    // Tally results (issue #233 Stage 3). PerStep/PerHistory read `tally_out`
    // (first half = sum, second half = sum_sq for PerHistory); PerSource reads
    // the per-source `src_acc` (the dispatch reconstructs the tally from it).
    let (tally_outputs, tally_sum_sq, src_acc) = if uses_src_acc {
        let src_acc_bits: Vec<u64> =
            bytemuck::cast_slice(&client.read_one(src_acc_h).unwrap()).to_vec();
        (Vec::new(), Vec::new(), src_acc_bits)
    } else {
        let tally_bits: Vec<u64> =
            bytemuck::cast_slice(&client.read_one(tally_out_h).unwrap()).to_vec();
        let sum_len = total_out_len.max(1);
        let outputs = unpack_photon_tally_outputs(&tally_bits[..sum_len], tallies);
        let sum_sq = if per_history {
            unpack_photon_tally_sum_sq(&tally_bits[total_out_len..total_out_len * 2], tallies)
        } else {
            Vec::new()
        };
        (outputs, sum_sq, Vec::new())
    };

    let lost = {
        let count = bytemuck::cast_slice::<u8, u64>(&client.read_one(lost_count_h).unwrap())[0];
        let records =
            bytemuck::cast_slice::<u8, f64>(&client.read_one(lost_f64_h).unwrap()).to_vec();
        crate::common::lost_particles::LostParticleResult::from_device(count, &records)
    };

    PhotonMultiCellResult {
        n_cells,
        alive,
        n_steps,
        final_energies,
        tally_outputs,
        tally_sum_sq,
        src_acc,
        lost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: `PHOTON_KERNEL_STORAGE_BUFFER_COUNT` must match the actual number
    /// of read-only `&[...]` slice parameters in the
    /// `multi_cell_photon_transport_kernel` `#[cube]` signature. If it drifts,
    /// the descriptor-budget guard at the launcher boundary under-checks and the
    /// kernel can bind more storage buffers than the adapter advertises --
    /// silently aliasing bindings into garbage reads that look like physics bugs.
    /// Mirrors the neutron `kernel_storage_buffer_count_matches_signature`; we
    /// grep the source because the `#[cube]` macro hides the signature from
    /// introspection. Immutable `&[` params are the read storage buffers; the
    /// `&mut [` outputs do not contain `&[` and are excluded by convention.
    #[test]
    fn photon_kernel_storage_buffer_count_matches_signature() {
        let src = include_str!("transport.rs");
        let mut in_kernel = false;
        let mut count = 0u32;
        for line in src.lines() {
            if line.trim_start().starts_with("#[cube(launch_unchecked)]") {
                in_kernel = true;
                continue;
            }
            if in_kernel {
                if line.starts_with(") {") || line.starts_with(")\u{a0}{") {
                    break;
                }
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") || trimmed.starts_with("///") {
                    continue;
                }
                if line.contains("&[") {
                    count += 1;
                }
            }
        }
        assert_eq!(
            count, PHOTON_KERNEL_STORAGE_BUFFER_COUNT,
            "found {count} `&[...]` slice parameters in the photon kernel \
             signature, but PHOTON_KERNEL_STORAGE_BUFFER_COUNT is set to \
             {PHOTON_KERNEL_STORAGE_BUFFER_COUNT}. Update the constant in \
             lockstep with kernel-signature changes -- it gates the GPU \
             descriptor-budget guard at the launcher boundary."
        );
    }
}
