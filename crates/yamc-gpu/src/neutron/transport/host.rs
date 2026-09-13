//! GPU host driver: pack the input buffers, launch the cubecl
//! kernel, read back the result. The kernel itself lives in the
//! parent `mod.rs` (the `#[cube]` macro generates the
//! `multi_cell_transport_kernel` launcher there); this submodule
//! holds the host-side dispatch logic and its small helper
//! `sum_bins`.

use cubecl::prelude::*;

// Glob-import the parent so layout constants (`COL_*`, `MT_SLOT_*`,
// `MAT_F64_COLS`, `KERNEL_STORAGE_BUFFER_COUNT`, `MAX_*`), the
// cubecl-generated `multi_cell_transport_kernel` launcher, and the
// shared `unpack_tally_outputs` helper are in scope. The full
// explicit list would be ~40 imports; the glob matches the close
// coupling between the host driver and the kernel it launches.
use super::*;
use crate::common::geometry::bvh_cell_finding::build_and_flatten_bvh;
use crate::common::tallies::TallyVarianceMode;

/// Run the multi-cell multi-step transport kernel with per-cell
/// materials.
///
/// `cell_to_material[i]` is the material index for cell `i`.
/// `xs_elastic_per_material` and `xs_absorption_per_material` are
/// flat buffers of length `n_materials × log_energy_grid.len()`,
/// laid out material-major. `target_mass_per_material` has one entry
/// per material.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_cell_transport(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    positions_in: &[f64],
    directions_in: &[f64],
    cell_aabbs: &[f64],
    cell_to_material: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    surface_boundaries: &[u32],
    region_program: &[u32],
    log_energy_grid: &[f64],
    coarse_log_energy_grid: &[f64],
    // Packed `[n_materials × COARSE_META_COLS]` per-material coarse-grid
    // descriptor (issue #212): base into `coarse_log_energy_grid`, coarse
    // length, and per-MT-buffer first-slab base for each material. Drives the
    // kernel's per-material coarse bracket + per-MT stride.
    coarse_meta: &[u32],
    // Concatenated per-material FINE grids backing the resonance-critical
    // aggregate macro XS + the nuc buffers (issue #212); each material owns its
    // fine grid, described by `fine_meta`. Distinct from the GLOBAL
    // `log_energy_grid` (retained for score / photon / decay).
    fine_log_energy_grid: &[f64],
    // Packed `[n_materials × FINE_META_COLS]` per-material fine-grid descriptor
    // (issue #212): base into `fine_log_energy_grid` (== aggregate-XS row base),
    // fine length, and first-slab element base into `nuc_macro_total`. Drives the
    // kernel's per-material fine bracket + aggregate-XS / nuc-buffer strides.
    fine_meta: &[u32],
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
    target_mass_per_material: &[f64],
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
    angle_pdf: &[f64],
    angle_interp: &[u32],
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
    scatter_in_cm_per_mt: &[u32],
    elastic_angle_n_energies: &[u32],
    elastic_angle_ae_offset: &[u32],
    elastic_angle_energy_grid: &[f64],
    elastic_angle_n_mu: &[u32],
    elastic_angle_mu_offset: &[u32],
    elastic_angle_mu: &[f64],
    elastic_angle_cdf: &[f64],
    elastic_angle_pdf: &[f64],
    elastic_angle_interp: &[u32],
    temperature_k_per_material: &[f64],
    km_n_energies: &[u32],
    km_ae_offset: &[u32],
    km_energy_grid: &[f64],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_n_x: &[u32],
    km_x_offset: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    evap_n_energies: &[u32],
    evap_n_components: &[u32],
    evap_ae_offset: &[u32],
    evap_theta_offset: &[u32],
    evap_energy_grid: &[f64],
    evap_theta: &[f64],
    evap_u: &[f64],
    nbps_n_bodies: &[u32],
    nbps_total_mass: &[f64],
    maxwell_n_energies: &[u32],
    maxwell_ae_offset: &[u32],
    maxwell_energy_grid: &[f64],
    maxwell_theta: &[f64],
    maxwell_u: &[f64],
    watt_n_energies: &[u32],
    watt_ae_offset: &[u32],
    watt_energy_grid: &[f64],
    watt_a: &[f64],
    watt_b: &[f64],
    watt_u: &[f64],
    urr_meta: &[u32],
    urr_ae_offset: &[u32],
    urr_cdf_offset: &[u32],
    urr_energy_grid: &[f64],
    urr_cdf: &[f64],
    urr_xs: &[f64],
    urr_atom_density: &[f64],
    tallies: &TalliesPack,
    xs_score_per_mt: &[f64],
    // Survival biasing (implicit capture). Pass `SurvivalBiasingInputs::off()`
    // (gate flag 0.0) to keep the neutron transport byte-identical to the
    // analog kernel; pass `SurvivalBiasingInputs::on(..)` (gate flag 1.0) to
    // enable implicit capture (never terminate on capture, discount weight by
    // the scatter+fission fraction). The buffer is always length 3.
    survival: &SurvivalBiasingInputs,
    // Coupled neutron->photon production (S4b). Pass
    // `CoupledPhotonInputs::coupled_off(n_materials, n_grid)` to keep the
    // neutron path byte-identical to a neutron-only run; pass
    // `CoupledPhotonInputs::from_materials(..)` (gate flag 1) to emit secondary
    // photons into the device bank. `photon_bank_capacity` slots are allocated
    // for the bank (ignored / size-1 when coupled-off).
    coupled: &CoupledPhotonInputs,
    // D1S decay-photon production. Pass `DecayPhotonInputs::decay_off(..)` (gate
    // 0) to keep the neutron path byte-identical; pass
    // `DecayPhotonInputs::from_materials(..)` (gate 1) to emit decay photons
    // into the device bank. Mutually exclusive with `coupled` (at most one gate
    // is 1).
    decay: &DecayPhotonInputs,
    // Per-collision nuclide selection (issue #74, Stage 1). Pass
    // `NuclideSelectInputs::single_nuclide(target_mass_per_material, n_grid)` to
    // keep the kernel byte-identical to pre-#74 behaviour (every material has
    // one nuclide, no selection draw); pass `NuclideSelectInputs::from_materials`
    // to enable per-collision selection + per-nuclide elastic AWR for
    // multi-nuclide materials.
    nuclide_select: &NuclideSelectInputs,
    // Device fission bank (issue #78). Pass `FissionBankInputs::off()` (gate 0)
    // to keep the fission branch on the legacy `weight *= nu_bar` + cap
    // terminator (byte-identical to the pre-bank kernel); pass
    // `FissionBankInputs::on()` (gate 1) to branch the fission chain into the
    // device bank. When on, `photon_bank_capacity` slots back the SHARED bank
    // that also serves coupled photons.
    fission_bank: &FissionBankInputs,
    photon_bank_capacity: usize,
    max_steps: u32,
    // Free-gas resonance/thermal cutoff multiplier (model option, default
    // 400.0); the regime boundary is `free_gas_threshold * kT`. Uploaded as a
    // 1-element f64 buffer (the `survival_params` pattern) and read by the
    // kernel as `free_gas_threshold[0]`; the CPU twin reads the same value from
    // `TransportInputs`. At the default 400.0 both paths are byte-identical to
    // the previous hardcode (issue #102).
    free_gas_threshold: f64,
    // Tally variance accumulation mode (issue #233). `PerStep` is the
    // byte-identical pre-#233 path; `PerHistory` (Stage 1, non-fissile) flushes
    // per-history sum + sum_sq into a doubled `tally_out`; `PerSource` (Stage 2,
    // fissile) flushes per-history sums into the per-source `src_acc` for
    // cross-launch fission-progeny grouping.
    variance: TallyVarianceMode,
) -> MultiCellResult {
    // Serialize multi-cell GPU dispatches during this crate's own test runs.
    // All `GpuContext::new()` calls share one cached device, so several heavy
    // multi-cell launches issued concurrently by parallel test threads can
    // contend on it and read back empty results (mean steps = 0). One large
    // dispatch at a time keeps the suite reliable on GPU hosts. This guard is
    // compiled only under `cfg(test)` of this crate, so it has zero effect on
    // production runs or on downstream crates that call this function.
    #[cfg(test)]
    let _gpu_dispatch_guard = {
        static GPU_DISPATCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GPU_DISPATCH_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };

    let n = seeds.len();
    let n_cells = cell_aabbs.len() / 6;
    assert_eq!(
        cell_to_material.len(),
        n_cells,
        "cell_to_material must have one entry per cell"
    );
    assert_eq!(
        surface_boundaries.len(),
        surface_types.len(),
        "surface_boundaries must have one entry per surface (matches surface_types length)"
    );
    // GPU descriptor-budget guard. The kernel binds
    // `KERNEL_STORAGE_BUFFER_COUNT` storage buffers in a single
    // descriptor set; if the host's adapter limit is below that,
    // bindings past the limit get silently aliased or zeroed --
    // tallies look fine but report wrong values. Panic loudly
    // here instead: the cost of a misleading-success failure
    // mode is much higher than a clear startup error.
    assert!(
        ctx.max_storage_buffers_per_stage >= KERNEL_STORAGE_BUFFER_COUNT,
        "GPU adapter supports {} storage buffers per stage, but \
         multi_cell_transport_kernel needs {}. Bindings past the \
         limit are silently aliased -- the kernel would produce \
         numerically-plausible but wrong tally results. Adapter: \
         {:?}",
        ctx.max_storage_buffers_per_stage,
        KERNEL_STORAGE_BUFFER_COUNT,
        ctx.adapter_info(),
    );
    let n_materials = target_mass_per_material.len();
    // Per-material COARSE grids (issue #212): each material owns its coarse grid
    // (its finest single per-nuclide grid), concatenated tight into
    // `coarse_log_energy_grid` and described per material by `coarse_meta`
    // (stride COARSE_META_COLS). The per-MT inelastic buffers ride each
    // material's own coarse grid, so their total length is the sum over slabs of
    // `MT_INELASTIC_COUNT × coarse_n[material]`. Single-material problems have
    // `coarse_meta == [0, coarse_len, 0]` and one coarse grid, so this reduces to
    // the old `n_slab × MT_INELASTIC_COUNT × coarse_len` and stays bit-identical.
    assert_eq!(
        coarse_meta.len(),
        n_materials * COARSE_META_COLS as usize,
        "coarse_meta must be flat [n_materials × COARSE_META_COLS]"
    );
    // Per-material FINE grids (issue #212): each material owns its fine grid (its
    // union grid), concatenated tight into `fine_log_energy_grid` and described
    // per material by `fine_meta` (stride FINE_META_COLS). The aggregate macro XS
    // are tight CSR keyed per material (total `sum_m fine_n[m]` ==
    // `fine_log_energy_grid.len()`), and the nuc buffers per slab (total
    // `sum_m nuc_count[m] × fine_n[m]`). Single-material problems have
    // `fine_meta == [0, fine_len, 0]`, reducing to `n_materials × n_grid` /
    // `n_slab × n_grid` (bit-identical).
    assert_eq!(
        fine_meta.len(),
        n_materials * FINE_META_COLS as usize,
        "fine_meta must be flat [n_materials × FINE_META_COLS]"
    );
    let expected_fine_agg: usize = fine_log_energy_grid.len();
    let expected_nuc: usize = (0..n_materials)
        .map(|m| {
            let nuc_count = nuclide_select.mat_nuclide_meta[m * 2 + 1] as usize;
            let fine_n = fine_meta[m * FINE_META_COLS as usize + COL_FINE_N as usize] as usize;
            nuc_count * fine_n
        })
        .sum();
    // Per-MT inelastic pools + the elastic-angle pool are keyed per-(material,
    // nuclide) SLAB (#74 Stages 2a / 2b), so their leading dimension is `n_slab`
    // (= `nuc_awr.len()`), not `n_materials`. Single-nuclide materials give one
    // slab each, so for those `n_slab == n_materials`.
    let n_slab = nuclide_select.nuc_awr.len();
    assert_eq!(
        nuclide_select.nuc_partial_xs.len(),
        expected_nuc * NUC_PARTIAL_COLS,
        "nuc_partial_xs must be tight CSR: sum_m nuc_count[m] × fine_n[m] × NUC_PARTIAL_COLS"
    );
    assert_eq!(
        nuclide_select.nuc_macro_total.len(),
        expected_nuc,
        "nuc_macro_total must be tight CSR: sum_m nuc_count[m] × fine_n[m]"
    );
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
    // Variable-length tight layout (issue #104): per-chi-row counts/bases are
    // [2 * n_materials] (prompt at row 2m, delayed at row 2m + 1, issue #364);
    // incident-energy rows are [total ae-rows]; (x, cdf) points are [total
    // x-points]. No fixed per-axis stride.
    let n_fission_eout_ae_rows = fission_eout_n_x_per_material.len();
    let n_chi_rows = 2 * n_materials;
    assert_eq!(fission_eout_kind_per_material.len(), n_chi_rows);
    assert_eq!(fission_eout_n_energies_per_material.len(), n_chi_rows);
    assert_eq!(
        fission_eout_ae_offset.len(),
        n_chi_rows,
        "fission_eout_ae_offset must be flat [2 * n_materials]"
    );
    assert_eq!(
        fission_eout_energy_grid_per_material.len(),
        n_fission_eout_ae_rows,
        "fission_eout_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        fission_eout_x_offset.len(),
        n_fission_eout_ae_rows,
        "fission_eout_x_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        fission_eout_cdf_per_material.len(),
        fission_eout_x_per_material.len(),
        "fission_eout_cdf must match fission_eout_x [total x-points]"
    );
    assert_eq!(
        fission_eout_p_per_material.len(),
        fission_eout_x_per_material.len(),
        "fission_eout_p must match fission_eout_x [total x-points]"
    );
    assert_eq!(
        fission_eout_interp_per_material.len(),
        n_fission_eout_ae_rows,
        "fission_eout_interp must be flat [total ae-rows]"
    );
    // SPARSE per-MT storage (issue #212). `permt_meta` is flat `[n_slab ×
    // MT_INELASTIC_COUNT × PERMT_META_COLS]`; the two value buffers are the tight
    // concatenation of every (slab, MT slot)'s stored range, so their length is
    // the sum of `n_stored` over all permt rows, and they must be equal (xs and
    // yield share the same range). `q_inelastic_per_mt` stays the flat per-(slab,
    // MT slot) scalar.
    assert_eq!(
        permt_meta.len(),
        n_slab * MT_INELASTIC_COUNT * PERMT_META_COLS as usize,
        "permt_meta must be flat [n_slab × MT_INELASTIC_COUNT × PERMT_META_COLS]"
    );
    let expected_sparse: usize = (0..permt_meta.len() / PERMT_META_COLS as usize)
        .map(|r| permt_meta[r * PERMT_META_COLS as usize + COL_PERMT_N_STORED as usize] as usize)
        .sum();
    assert_eq!(
        xs_inelastic_per_mt_sparse.len(),
        expected_sparse,
        "xs_inelastic_per_mt_sparse must be tight CSR: sum over permt rows of n_stored"
    );
    assert_eq!(
        yield_per_mt_sparse.len(),
        expected_sparse,
        "yield_per_mt_sparse must match xs_inelastic_per_mt_sparse (shared per-slot range)"
    );
    assert_eq!(
        q_inelastic_per_mt.len(),
        n_slab * MT_INELASTIC_COUNT,
        "q_inelastic_per_mt must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    let n_tallies = tallies.n_tallies() as usize;
    assert!(n_tallies > 0, "tallies pack must have at least one tally");
    assert_eq!(tallies.cell_to_bin.len(), n_tallies * n_cells);
    assert_eq!(tallies.n_cells_per_tally.len(), n_tallies);
    assert_eq!(tallies.edges_offsets.len(), n_tallies);
    assert_eq!(tallies.n_bins_per_tally.len(), n_tallies);
    assert_eq!(tallies.out_offsets.len(), n_tallies + 1);
    // Variable-length tight layout (issue #104): per-(slab,MT) counts/bases are
    // [n_slab × MT_INELASTIC_COUNT]; incident-energy rows are [total ae-rows];
    // (mu,cdf,pdf) points are [total mu-points]. No fixed per-axis stride.
    let n_inel_ae_rows = angle_n_mu.len();
    assert_eq!(
        angle_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "angle_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        angle_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "angle_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        angle_energy_grid.len(),
        n_inel_ae_rows,
        "angle_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        angle_mu_offset.len(),
        n_inel_ae_rows,
        "angle_mu_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        angle_mu.len(),
        angle_cdf.len(),
        "angle_cdf must match angle_mu [total mu-points]"
    );
    assert_eq!(
        angle_pdf.len(),
        angle_mu.len(),
        "angle_pdf must match angle_mu [total mu-points]"
    );
    assert_eq!(
        angle_interp.len(),
        n_inel_ae_rows,
        "angle_interp must be flat [total ae-rows]"
    );
    // Variable-length tight layout (issue #104): per-(slab,MT) scalars/bases are
    // [n_slab × MT_INELASTIC_COUNT]; incident-energy rows are [total ae-rows];
    // (x,cdf,p) points are [total x-points]. No fixed per-axis stride.
    let n_eout_ae_rows = eout_n_x.len();
    assert_eq!(
        eout_kind.len(),
        n_slab * MT_INELASTIC_COUNT,
        "eout_kind must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        eout_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "eout_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        eout_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "eout_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        eout_energy_grid.len(),
        n_eout_ae_rows,
        "eout_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        eout_x_offset.len(),
        n_eout_ae_rows,
        "eout_x_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        eout_cdf.len(),
        eout_x.len(),
        "eout_cdf must match eout_x [total x-points]"
    );
    assert_eq!(
        eout_histogram_interp.len(),
        n_slab * MT_INELASTIC_COUNT,
        "eout_histogram_interp must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        eout_p.len(),
        eout_x.len(),
        "eout_p must match eout_x [total x-points]"
    );
    assert_eq!(
        eout_interp.len(),
        n_eout_ae_rows,
        "eout_interp must be flat [total ae-rows]"
    );
    assert_eq!(
        eout_n_discrete.len(),
        n_eout_ae_rows,
        "eout_n_discrete must be flat [total ae-rows]"
    );
    // Variable-length tight layout (issue #104), three nesting levels: per
    // (slab,MT) slot counts/bases are [n_slab × MT_INELASTIC_COUNT]; the
    // incident-energy ae-rows are [total ae-rows]; the E_out x-points are
    // [total x-points]; the mu points are [total mu-points]. No fixed
    // per-axis stride.
    let n_corr_ae_rows = corr_n_x.len();
    let n_corr_x_points = corr_n_mu.len();
    assert_eq!(
        corr_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "corr_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        corr_n_components.len(),
        n_slab * MT_INELASTIC_COUNT,
        "corr_n_components must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        corr_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "corr_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        corr_energy_grid.len(),
        n_corr_ae_rows,
        "corr_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        corr_x_offset.len(),
        n_corr_ae_rows,
        "corr_x_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        corr_x.len(),
        n_corr_x_points,
        "corr_x must be flat [total x-points]"
    );
    assert_eq!(
        corr_cdf.len(),
        corr_x.len(),
        "corr_cdf must match corr_x [total x-points]"
    );
    assert_eq!(
        corr_p.len(),
        corr_x.len(),
        "corr_p must match corr_x [total x-points]"
    );
    assert_eq!(
        corr_interp.len(),
        n_corr_ae_rows,
        "corr_interp must be flat [total ae-rows]"
    );
    assert_eq!(
        corr_n_discrete.len(),
        n_corr_ae_rows,
        "corr_n_discrete must be flat [total ae-rows]"
    );
    assert_eq!(
        corr_mu_offset.len(),
        n_corr_x_points,
        "corr_mu_offset must be flat [total x-points]"
    );
    assert_eq!(
        corr_mu_cdf.len(),
        corr_mu.len(),
        "corr_mu_cdf must match corr_mu [total mu-points]"
    );
    assert_eq!(
        corr_mu_pdf.len(),
        corr_mu.len(),
        "corr_mu_pdf must match corr_mu [total mu-points]"
    );
    assert_eq!(
        corr_mu_interp.len(),
        n_corr_x_points,
        "corr_mu_interp must be flat [total x-points]"
    );
    assert_eq!(
        scatter_in_cm_per_mt.len(),
        n_slab * MT_INELASTIC_COUNT,
        "scatter_in_cm_per_mt must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    // Elastic angular tables are keyed per-(material, nuclide) slab (#74
    // Stage 2a), same `n_slab` leading dimension as the per-MT inelastic pools.
    // Variable-length tight layout (issue #104): per-slab counts/bases are
    // `[n_slab]`; the incident-energy rows (`energy_grid`/`n_mu`/`interp`/
    // `mu_offset`) are `[total ae-rows]`; the (mu,cdf,pdf) points are
    // `[total mu-points]`. No fixed MAX_* stride.
    let n_ae_rows = elastic_angle_n_mu.len();
    assert_eq!(
        elastic_angle_n_energies.len(),
        n_slab,
        "elastic_angle_n_energies must be flat [n_slab]"
    );
    assert_eq!(
        elastic_angle_ae_offset.len(),
        n_slab,
        "elastic_angle_ae_offset must be flat [n_slab]"
    );
    assert_eq!(
        elastic_angle_energy_grid.len(),
        n_ae_rows,
        "elastic_angle_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        elastic_angle_interp.len(),
        n_ae_rows,
        "elastic_angle_interp must be flat [total ae-rows]"
    );
    assert_eq!(
        elastic_angle_mu_offset.len(),
        n_ae_rows,
        "elastic_angle_mu_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        elastic_angle_cdf.len(),
        elastic_angle_mu.len(),
        "elastic_angle_cdf must match elastic_angle_mu [total mu-points]"
    );
    assert_eq!(
        elastic_angle_pdf.len(),
        elastic_angle_mu.len(),
        "elastic_angle_pdf must match elastic_angle_mu [total mu-points]"
    );
    assert_eq!(
        temperature_k_per_material.len(),
        n_materials,
        "temperature_k_per_material must be flat [n_materials]"
    );
    // Variable-length tight layout (issue #104): per-(slab,MT) scalars/bases
    // are [n_slab × MT_INELASTIC_COUNT]; incident-energy rows are [total
    // ae-rows]; (x,p,c,r,a) points are [total x-points]. No fixed per-axis
    // stride.
    let n_km_ae_rows = km_n_x.len();
    assert_eq!(
        km_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "km_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        km_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "km_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        km_energy_grid.len(),
        n_km_ae_rows,
        "km_energy_grid must be flat [total ae-rows]"
    );
    assert_eq!(
        km_interp.len(),
        n_km_ae_rows,
        "km_interp must be flat [total ae-rows]"
    );
    assert_eq!(
        km_n_discrete.len(),
        n_km_ae_rows,
        "km_n_discrete must be flat [total ae-rows]"
    );
    assert_eq!(
        km_x_offset.len(),
        n_km_ae_rows,
        "km_x_offset must be flat [total ae-rows]"
    );
    assert_eq!(
        km_p.len(),
        km_x.len(),
        "km_p must match km_x [total x-points]"
    );
    assert_eq!(
        km_c.len(),
        km_x.len(),
        "km_c must match km_x [total x-points]"
    );
    assert_eq!(
        km_r.len(),
        km_x.len(),
        "km_r must match km_x [total x-points]"
    );
    assert_eq!(
        km_a.len(),
        km_x.len(),
        "km_a must match km_x [total x-points]"
    );
    // Tight variable-length layout (issue #104): per (slab,MT) slot
    // counts/bases are [n_slab × MT_INELASTIC_COUNT]; the E_in rows are
    // [total rows] (no fixed per-axis stride). `evap_theta` is
    // component-major: [total component-rows].
    assert_eq!(
        evap_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "evap_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        evap_n_components.len(),
        n_slab * MT_INELASTIC_COUNT,
        "evap_n_components must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        evap_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "evap_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        evap_theta_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "evap_theta_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        evap_u.len(),
        evap_energy_grid.len(),
        "evap_u must match evap_energy_grid [total E_in rows]"
    );
    assert_eq!(
        nbps_n_bodies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "nbps_n_bodies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        nbps_total_mass.len(),
        n_slab * MT_INELASTIC_COUNT,
        "nbps_total_mass must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        maxwell_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "maxwell_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        maxwell_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "maxwell_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        maxwell_theta.len(),
        maxwell_energy_grid.len(),
        "maxwell_theta must match maxwell_energy_grid [total E_in rows]"
    );
    assert_eq!(
        maxwell_u.len(),
        n_slab * MT_INELASTIC_COUNT,
        "maxwell_u must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        watt_n_energies.len(),
        n_slab * MT_INELASTIC_COUNT,
        "watt_n_energies must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        watt_ae_offset.len(),
        n_slab * MT_INELASTIC_COUNT,
        "watt_ae_offset must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    assert_eq!(
        watt_a.len(),
        watt_energy_grid.len(),
        "watt_a must match watt_energy_grid [total E_in rows]"
    );
    assert_eq!(
        watt_b.len(),
        watt_energy_grid.len(),
        "watt_b must match watt_energy_grid [total E_in rows]"
    );
    assert_eq!(
        watt_u.len(),
        n_slab * MT_INELASTIC_COUNT,
        "watt_u must be flat [n_slab × MT_INELASTIC_COUNT]"
    );
    // URR is keyed per-(material, nuclide) SLAB (issue #210): one `urr_meta`
    // row and one `urr_atom_density` / CSR-base entry per global slab.
    assert_eq!(
        urr_meta.len(),
        n_slab * URR_META_COLS,
        "urr_meta must be flat [n_slab × URR_META_COLS]"
    );
    // Tight CSR (issue #104): `urr_ae_offset` / `urr_cdf_offset` carry one
    // per-slab base each; `urr_energy_grid` / `urr_cdf` / `urr_xs` are
    // concatenated tight (no MAX_URR_* padding). `urr_xs` is the cdf grid ×
    // URR_XS_COLS, so its length is exactly `urr_cdf.len() * URR_XS_COLS`.
    assert_eq!(
        urr_ae_offset.len(),
        n_slab,
        "urr_ae_offset must have one entry per slab"
    );
    assert_eq!(
        urr_cdf_offset.len(),
        n_slab,
        "urr_cdf_offset must have one entry per slab"
    );
    assert_eq!(
        urr_xs.len(),
        urr_cdf.len() * URR_XS_COLS,
        "urr_xs must be flat [urr_cdf cells × URR_XS_COLS]"
    );
    assert_eq!(
        urr_atom_density.len(),
        n_slab,
        "urr_atom_density must have one entry per slab"
    );
    for (t, &n_bins) in tallies.n_bins_per_tally.iter().enumerate() {
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

    let (bvh_aabbs, bvh_meta, bvh_prim_indices, bvh_unbounded) = build_and_flatten_bvh(cell_aabbs);

    let client = ctx.client();
    // EXPERIMENT: start each launch from a fully-idle, cleaned client. The
    // shared cubecl client is a process-global singleton, so back-to-back
    // launches (test suite, or repeated model.simulate_transport('gpu'))
    // otherwise accumulate state that intermittently corrupts results.
    let _ = pollster::block_on(client.sync());
    client.memory_cleanup();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_h = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let positions_h = client.create_from_slice(bytemuck::cast_slice(positions_in));
    let dirs_h = client.create_from_slice(bytemuck::cast_slice(directions_in));
    let aabbs_h = client.create_from_slice(bytemuck::cast_slice(cell_aabbs));
    let cell_mat_h = client.create_from_slice(bytemuck::cast_slice(cell_to_material));
    // cubecl rejects zero-sized buffers; fall back to single-slot
    // padding when there are no nodes / unbounded prims. Kernel
    // guards on `n_bvh_nodes > 0` and `n_bvh_unbounded > 0` so the
    // padding is never read.
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
    // cubecl rejects zero-sized buffers; a geometry with no cells has an
    // empty program. Pad a single sentinel u32 -- the kernel only reads
    // the program when `n_cells > 0`, so the padding is never touched.
    let region_program_data: &[u32] = if region_program.is_empty() {
        &[0u32]
    } else {
        region_program
    };
    let region_program_h = client.create_from_slice(bytemuck::cast_slice(region_program_data));
    let log_grid_h = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let coarse_log_grid_h = client.create_from_slice(bytemuck::cast_slice(coarse_log_energy_grid));
    let coarse_meta_h = client.create_from_slice(bytemuck::cast_slice(coarse_meta));
    let fine_log_grid_h = client.create_from_slice(bytemuck::cast_slice(fine_log_energy_grid));
    let fine_meta_h = client.create_from_slice(bytemuck::cast_slice(fine_meta));
    let xs_e_h = client.create_from_slice(bytemuck::cast_slice(xs_elastic_per_material));
    let xs_a_h = client.create_from_slice(bytemuck::cast_slice(xs_absorption_per_material));
    let xs_i_h = client.create_from_slice(bytemuck::cast_slice(xs_inelastic_per_material));
    let xs_i_mt_h = client.create_from_slice(bytemuck::cast_slice(xs_inelastic_per_mt_sparse));
    let permt_meta_h = client.create_from_slice(bytemuck::cast_slice(permt_meta));
    let xs_f_h = client.create_from_slice(bytemuck::cast_slice(xs_fission_per_material));
    let nu_h = client.create_from_slice(bytemuck::cast_slice(nu_bar_per_material));
    let beta_h = client.create_from_slice(bytemuck::cast_slice(beta_delayed_per_material));
    let fission_eout_kind_h =
        client.create_from_slice(bytemuck::cast_slice(fission_eout_kind_per_material));
    let fission_eout_nae_h =
        client.create_from_slice(bytemuck::cast_slice(fission_eout_n_energies_per_material));
    let fission_eout_ae_off_h =
        client.create_from_slice(bytemuck::cast_slice(fission_eout_ae_offset));
    // Tight CSR (issue #104): for a model with no fissile material every
    // fission-eout data array is empty (zero ae-rows, zero points). wgpu
    // cannot allocate a zero-length storage buffer, so pad a single sentinel
    // element -- the kernel only reads these when
    // `fission_eout_n_energies > 0`, so the padding is never touched.
    let fission_eout_eg_pad: Vec<f64>;
    let fission_eout_eg_slice: &[f64] = if fission_eout_energy_grid_per_material.is_empty() {
        fission_eout_eg_pad = vec![0.0];
        &fission_eout_eg_pad
    } else {
        fission_eout_energy_grid_per_material
    };
    let fission_eout_n_x_pad: Vec<u32>;
    let fission_eout_n_x_slice: &[u32] = if fission_eout_n_x_per_material.is_empty() {
        fission_eout_n_x_pad = vec![0u32];
        &fission_eout_n_x_pad
    } else {
        fission_eout_n_x_per_material
    };
    let fission_eout_x_off_pad: Vec<u32>;
    let fission_eout_x_off_slice: &[u32] = if fission_eout_x_offset.is_empty() {
        fission_eout_x_off_pad = vec![0u32];
        &fission_eout_x_off_pad
    } else {
        fission_eout_x_offset
    };
    let fission_eout_x_pad: Vec<f64>;
    let fission_eout_x_slice: &[f64] = if fission_eout_x_per_material.is_empty() {
        fission_eout_x_pad = vec![0.0];
        &fission_eout_x_pad
    } else {
        fission_eout_x_per_material
    };
    let fission_eout_cdf_pad: Vec<f64>;
    let fission_eout_cdf_slice: &[f64] = if fission_eout_cdf_per_material.is_empty() {
        fission_eout_cdf_pad = vec![0.0];
        &fission_eout_cdf_pad
    } else {
        fission_eout_cdf_per_material
    };
    let fission_eout_p_pad: Vec<f64>;
    let fission_eout_p_slice: &[f64] = if fission_eout_p_per_material.is_empty() {
        fission_eout_p_pad = vec![0.0];
        &fission_eout_p_pad
    } else {
        fission_eout_p_per_material
    };
    let fission_eout_interp_pad: Vec<u32>;
    let fission_eout_interp_slice: &[u32] = if fission_eout_interp_per_material.is_empty() {
        fission_eout_interp_pad = vec![0u32];
        &fission_eout_interp_pad
    } else {
        fission_eout_interp_per_material
    };
    let fission_eout_eg_h = client.create_from_slice(bytemuck::cast_slice(fission_eout_eg_slice));
    let fission_eout_n_x_h = client.create_from_slice(bytemuck::cast_slice(fission_eout_n_x_slice));
    let fission_eout_x_off_h =
        client.create_from_slice(bytemuck::cast_slice(fission_eout_x_off_slice));
    let fission_eout_x_h = client.create_from_slice(bytemuck::cast_slice(fission_eout_x_slice));
    let fission_eout_cdf_h = client.create_from_slice(bytemuck::cast_slice(fission_eout_cdf_slice));
    let fission_eout_p_h = client.create_from_slice(bytemuck::cast_slice(fission_eout_p_slice));
    let fission_eout_interp_h =
        client.create_from_slice(bytemuck::cast_slice(fission_eout_interp_slice));
    // Pack the per-material f64 scalars (target mass, temperature,
    // Watt a, Watt b) into a single `[n_mat × MAT_F64_COLS]` upload. Same
    // packing motivation as `mt_slot_u32_meta`: collapses what would otherwise
    // be separate scalar bindings into one. The URR-atom-density column
    // (`COL_URR_ATOM_DENSITY`) is left dead (issue #210 moved URR atom density
    // to its own per-slab `urr_atom_density` binding); the stride stays 5 so
    // the kernel's hardcoded `mat_f64_meta.len() / 5` material count and the
    // other columns' offsets are untouched.
    let mat_f64_meta = pack_mat_f64_meta(
        target_mass_per_material,
        temperature_k_per_material,
        fission_a_per_material,
        fission_b_per_material,
    );
    let mat_f64_meta_h = client.create_from_slice(bytemuck::cast_slice(&mat_f64_meta));
    let q_inel_mt_h = client.create_from_slice(bytemuck::cast_slice(q_inelastic_per_mt));
    let yield_mt_h = client.create_from_slice(bytemuck::cast_slice(yield_per_mt_sparse));
    let ang_n_e_h = client.create_from_slice(bytemuck::cast_slice(angle_n_energies));
    let ang_ae_off_h = client.create_from_slice(bytemuck::cast_slice(angle_ae_offset));
    let ang_eg_h = client.create_from_slice(bytemuck::cast_slice(angle_energy_grid));
    let ang_n_mu_h = client.create_from_slice(bytemuck::cast_slice(angle_n_mu));
    let ang_mu_off_h = client.create_from_slice(bytemuck::cast_slice(angle_mu_offset));
    let ang_mu_h = client.create_from_slice(bytemuck::cast_slice(angle_mu));
    let ang_cdf_h = client.create_from_slice(bytemuck::cast_slice(angle_cdf));
    let ang_pdf_h = client.create_from_slice(bytemuck::cast_slice(angle_pdf));
    let ang_interp_h = client.create_from_slice(bytemuck::cast_slice(angle_interp));
    // Pack the 7 per-MT-slot u32 metadata buffers (eout_kind,
    // eout_n_energies, corr_n_energies, scatter_in_cm_per_mt,
    // km_n_energies, evap_n_energies, nbps_n_bodies) into a single
    // `[n_mat × MT_INELASTIC_COUNT × MT_SLOT_U32_COLS]` upload. The
    // kernel reads from this buffer at column offsets matching the
    // `COL_*` constants. Saves 6 storage-buffer descriptor bindings,
    // head-room the kernel needs for follow-on distribution samplers
    // (Watt-inelastic, Tabulated, URR) without tripping the
    // per-stage descriptor budget.
    let mt_slot_u32_meta = pack_mt_slot_u32_meta(
        eout_kind,
        eout_n_energies,
        corr_n_energies,
        corr_n_components,
        scatter_in_cm_per_mt,
        km_n_energies,
        evap_n_energies,
        evap_n_components,
        nbps_n_bodies,
        maxwell_n_energies,
        watt_n_energies,
    );
    let mt_slot_meta_h = client.create_from_slice(bytemuck::cast_slice(&mt_slot_u32_meta));
    let eout_ae_off_h = client.create_from_slice(bytemuck::cast_slice(eout_ae_offset));
    let eout_eg_h = client.create_from_slice(bytemuck::cast_slice(eout_energy_grid));
    let eout_n_x_h = client.create_from_slice(bytemuck::cast_slice(eout_n_x));
    let eout_x_off_h = client.create_from_slice(bytemuck::cast_slice(eout_x_offset));
    let eout_x_h = client.create_from_slice(bytemuck::cast_slice(eout_x));
    let eout_cdf_h = client.create_from_slice(bytemuck::cast_slice(eout_cdf));
    let eout_hist_h = client.create_from_slice(bytemuck::cast_slice(eout_histogram_interp));
    let eout_p_h = client.create_from_slice(bytemuck::cast_slice(eout_p));
    let eout_interp_h = client.create_from_slice(bytemuck::cast_slice(eout_interp));
    let eout_n_discrete_h = client.create_from_slice(bytemuck::cast_slice(eout_n_discrete));
    let corr_ae_off_h = client.create_from_slice(bytemuck::cast_slice(corr_ae_offset));
    let corr_eg_h = client.create_from_slice(bytemuck::cast_slice(corr_energy_grid));
    let corr_n_x_h = client.create_from_slice(bytemuck::cast_slice(corr_n_x));
    let corr_x_off_h = client.create_from_slice(bytemuck::cast_slice(corr_x_offset));
    let corr_x_h = client.create_from_slice(bytemuck::cast_slice(corr_x));
    let corr_cdf_h = client.create_from_slice(bytemuck::cast_slice(corr_cdf));
    let corr_p_h = client.create_from_slice(bytemuck::cast_slice(corr_p));
    let corr_interp_h = client.create_from_slice(bytemuck::cast_slice(corr_interp));
    let corr_n_discrete_h = client.create_from_slice(bytemuck::cast_slice(corr_n_discrete));
    let corr_n_mu_h = client.create_from_slice(bytemuck::cast_slice(corr_n_mu));
    let corr_mu_off_h = client.create_from_slice(bytemuck::cast_slice(corr_mu_offset));
    let corr_mu_h = client.create_from_slice(bytemuck::cast_slice(corr_mu));
    let corr_mu_cdf_h = client.create_from_slice(bytemuck::cast_slice(corr_mu_cdf));
    let corr_mu_pdf_h = client.create_from_slice(bytemuck::cast_slice(corr_mu_pdf));
    let corr_mu_interp_h = client.create_from_slice(bytemuck::cast_slice(corr_mu_interp));
    let el_ae_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_n_energies));
    let el_ae_off_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_ae_offset));
    let el_eg_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_energy_grid));
    let el_n_mu_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_n_mu));
    let el_mu_off_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_mu_offset));
    let el_mu_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_mu));
    let el_cdf_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_cdf));
    let el_pdf_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_pdf));
    let el_interp_h = client.create_from_slice(bytemuck::cast_slice(elastic_angle_interp));
    let km_ae_off_h = client.create_from_slice(bytemuck::cast_slice(km_ae_offset));
    let km_eg_h = client.create_from_slice(bytemuck::cast_slice(km_energy_grid));
    let km_interp_h = client.create_from_slice(bytemuck::cast_slice(km_interp));
    let km_n_disc_h = client.create_from_slice(bytemuck::cast_slice(km_n_discrete));
    let km_n_x_h = client.create_from_slice(bytemuck::cast_slice(km_n_x));
    let km_x_off_h = client.create_from_slice(bytemuck::cast_slice(km_x_offset));
    let km_x_h = client.create_from_slice(bytemuck::cast_slice(km_x));
    let km_p_h = client.create_from_slice(bytemuck::cast_slice(km_p));
    let km_c_h = client.create_from_slice(bytemuck::cast_slice(km_c));
    let km_r_h = client.create_from_slice(bytemuck::cast_slice(km_r));
    let km_a_h = client.create_from_slice(bytemuck::cast_slice(km_a));
    let evap_ae_off_h = client.create_from_slice(bytemuck::cast_slice(evap_ae_offset));
    let evap_theta_off_h = client.create_from_slice(bytemuck::cast_slice(evap_theta_offset));
    let evap_eg_h = client.create_from_slice(bytemuck::cast_slice(evap_energy_grid));
    let evap_theta_h = client.create_from_slice(bytemuck::cast_slice(evap_theta));
    let evap_u_h = client.create_from_slice(bytemuck::cast_slice(evap_u));
    let maxwell_ae_off_h = client.create_from_slice(bytemuck::cast_slice(maxwell_ae_offset));
    let maxwell_eg_h = client.create_from_slice(bytemuck::cast_slice(maxwell_energy_grid));
    let maxwell_theta_h = client.create_from_slice(bytemuck::cast_slice(maxwell_theta));
    let watt_ae_off_h = client.create_from_slice(bytemuck::cast_slice(watt_ae_offset));
    let watt_eg_h = client.create_from_slice(bytemuck::cast_slice(watt_energy_grid));
    // Pack the 3 per-MT-slot f64 scalars (maxwell_u, watt_u,
    // nbps_total_mass) into a single packed buffer of
    // stride MT_SLOT_F64_COLS. (evap_u is incident-energy dependent
    // and lives in its own per-grid buffer, created above.)
    // Saves 3 storage-buffer descriptor
    // bindings -- without this consolidation the kernel hits the
    // per-stage descriptor budget on some drivers (silent aliasing
    // past the limit corrupts adjacent buffers; caught by the
    // free-gas thermal regression test).
    let mt_slot_f64_meta = pack_mt_slot_f64_meta(maxwell_u, watt_u, nbps_total_mass);
    let mt_slot_f64_h = client.create_from_slice(bytemuck::cast_slice(&mt_slot_f64_meta));
    // Interleave Watt's `a` and `b` parameters into a single
    // stride-2 buffer to keep one fewer storage-buffer descriptor
    // binding. Element-parallel to the tight `watt_energy_grid` (issue
    // #104): `watt_ab[(watt_ae_offset[mat_slot] + i) * 2 + 0]` is a,
    // `+ 1` is b. Same ordering otherwise.
    let watt_ab = pack_watt_ab_interleaved(watt_a, watt_b);
    let watt_ab_h = client.create_from_slice(bytemuck::cast_slice(&watt_ab));
    let urr_meta_h = client.create_from_slice(bytemuck::cast_slice(urr_meta));
    let urr_ae_off_h = client.create_from_slice(bytemuck::cast_slice(urr_ae_offset));
    let urr_cdf_off_h = client.create_from_slice(bytemuck::cast_slice(urr_cdf_offset));
    let urr_energy_grid_h = client.create_from_slice(bytemuck::cast_slice(urr_energy_grid));
    let urr_cdf_h = client.create_from_slice(bytemuck::cast_slice(urr_cdf));
    let urr_xs_h = client.create_from_slice(bytemuck::cast_slice(urr_xs));
    let urr_atom_density_h = client.create_from_slice(bytemuck::cast_slice(urr_atom_density));
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
    let tally_score_mt_h = client.create_from_slice(bytemuck::cast_slice(&tallies.score_mt));
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
    // Energy-function tally buffers (issue #271). Padded the same way: with no
    // `energy_function=` / `dose_coefficients=` tally the params buffer would be
    // zero-length, which cubecl rejects. The offsets stay all-equal so every
    // tally's range is empty and the kernel never reads the pad.
    let tally_efunc_offsets_h =
        client.create_from_slice(bytemuck::cast_slice(&tallies.efunc_offsets));
    let efunc_params_padded: Vec<f64> = if tallies.efunc_params.is_empty() {
        vec![0.0]
    } else {
        tallies.efunc_params.clone()
    };
    let tally_efunc_params_h = client.create_from_slice(bytemuck::cast_slice(&efunc_params_padded));
    // Pad an all-zero slot when the model has no per-MT score tallies
    // -- cubecl rejects zero-length buffers and the kernel only reads
    // this when SCORE_PER_MT fires.
    let n_materials_count = target_mass_per_material.len();
    let n_grid_count = log_energy_grid.len();
    let xs_score_pad: Vec<f64>;
    let xs_score_slice: &[f64] = if xs_score_per_mt.is_empty() {
        xs_score_pad = vec![0.0; n_materials_count.max(1) * n_grid_count.max(1)];
        &xs_score_pad
    } else {
        xs_score_per_mt
    };
    let xs_score_h = client.create_from_slice(bytemuck::cast_slice(xs_score_slice));

    // Survival biasing (implicit capture): a single length-3 f64 buffer
    // `[enable, weight_cutoff, weight_survive]`. The gate flag (slot 0) is
    // 0.0 when off, keeping the kernel byte-identical to the analog run.
    let survival_h = client.create_from_slice(bytemuck::cast_slice(&survival.params));

    // Free-gas resonance/thermal cutoff multiplier (issue #102): a single-
    // element f64 buffer mirroring `survival_h`. The kernel reads
    // `free_gas_threshold[0]` for its free-gas regime boundary. A buffer (not a
    // comptime arg) because cubecl comptime values must be `Hash` and `f64` is
    // not. At the default 400.0 this is byte-identical to the old hardcode.
    let free_gas_threshold_slice = [free_gas_threshold];
    let free_gas_threshold_h =
        client.create_from_slice(bytemuck::cast_slice(&free_gas_threshold_slice));

    // ----------------------- coupled neutron->photon (S4b) -----------------
    // Upload the (possibly coupled-off) photon-production tables and allocate
    // the device particle bank. When coupled-off the gate flag is 0 so the
    // kernel never reads the tables nor writes the bank; a size-1 bank keeps
    // cubecl happy (it rejects zero-length buffers).
    // Three producers append here: coupled/decay secondary photons, fission
    // progeny (#78), and any (n,xn) secondary that overflows a thread's
    // pending stack (issue #111 phase 2). The last one is live on EVERY neutron
    // run, so the requested capacity is always honoured -- collapsing to a
    // size-1 bank when the first two are off would turn a rare spill into a
    // hard overflow error. A caller whose path cannot use banked neutrons still
    // asks for 1 and pays nothing.
    let bank_capacity = photon_bank_capacity.max(1);
    let coupled_enabled_h =
        client.create_from_slice(bytemuck::cast_slice(&coupled.coupled_enabled));
    let fission_bank_enabled_h =
        client.create_from_slice(bytemuck::cast_slice(&fission_bank.enabled));
    let ph_prod_h = client.create_from_slice(bytemuck::cast_slice(&coupled.photon_prod));
    let ph_rxn_xs_h = client.create_from_slice(bytemuck::cast_slice(&coupled.rxn_xs));
    let ph_prod_rxn_idx_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_rxn_idx));
    let ph_prod_yield_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_yield_grid));
    let ph_prod_kind_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_eout_kind));
    let ph_prod_line_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_line_energy));
    let ph_prod_pflag_h =
        client.create_from_slice(bytemuck::cast_slice(&coupled.prod_primary_flag));
    let ph_prod_awr_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_awr));
    let ph_prod_slot_h = client.create_from_slice(bytemuck::cast_slice(&coupled.prod_dist_slot));
    // Tight CSR (issue #104): for a coupled-OFF / no-photon-production model the
    // per-row (`pa_energy_grid` / `pa_n_mu` / `pa_interp` / `pa_mu_offset`) and
    // per-point (`pa_mu` / `pa_cdf` / `pa_pdf`) angle arrays are empty. wgpu
    // cannot allocate a zero-length storage buffer, so pad a single sentinel
    // element -- the kernel only reads these when `pa_n_energies > 0` (and
    // only when emission is gated on), so the padding is never touched. The
    // per-product `pa_n_energies` / `pa_ae_offset` always carry >= 1 product.
    let ph_pa_ne_h = client.create_from_slice(bytemuck::cast_slice(&coupled.pa_n_energies));
    let ph_pa_aeoff_h = client.create_from_slice(bytemuck::cast_slice(&coupled.pa_ae_offset));
    let pa_mu_off_pad: Vec<u32>;
    let pa_mu_off_slice: &[u32] = if coupled.pa_mu_offset.is_empty() {
        pa_mu_off_pad = vec![0u32];
        &pa_mu_off_pad
    } else {
        &coupled.pa_mu_offset
    };
    let pa_eg_pad: Vec<f64>;
    let pa_eg_slice: &[f64] = if coupled.pa_energy_grid.is_empty() {
        pa_eg_pad = vec![0.0];
        &pa_eg_pad
    } else {
        &coupled.pa_energy_grid
    };
    let pa_nmu_pad: Vec<u32>;
    let pa_nmu_slice: &[u32] = if coupled.pa_n_mu.is_empty() {
        pa_nmu_pad = vec![0u32];
        &pa_nmu_pad
    } else {
        &coupled.pa_n_mu
    };
    let pa_mu_pad: Vec<f64>;
    let pa_mu_slice: &[f64] = if coupled.pa_mu.is_empty() {
        pa_mu_pad = vec![0.0];
        &pa_mu_pad
    } else {
        &coupled.pa_mu
    };
    let pa_cdf_pad: Vec<f64>;
    let pa_cdf_slice: &[f64] = if coupled.pa_cdf.is_empty() {
        pa_cdf_pad = vec![0.0];
        &pa_cdf_pad
    } else {
        &coupled.pa_cdf
    };
    let pa_pdf_pad: Vec<f64>;
    let pa_pdf_slice: &[f64] = if coupled.pa_pdf.is_empty() {
        pa_pdf_pad = vec![0.0];
        &pa_pdf_pad
    } else {
        &coupled.pa_pdf
    };
    let pa_interp_pad: Vec<u32>;
    let pa_interp_slice: &[u32] = if coupled.pa_interp.is_empty() {
        pa_interp_pad = vec![0u32];
        &pa_interp_pad
    } else {
        &coupled.pa_interp
    };
    let ph_pa_muoff_h = client.create_from_slice(bytemuck::cast_slice(pa_mu_off_slice));
    let ph_pa_eg_h = client.create_from_slice(bytemuck::cast_slice(pa_eg_slice));
    let ph_pa_nmu_h = client.create_from_slice(bytemuck::cast_slice(pa_nmu_slice));
    let ph_pa_mu_h = client.create_from_slice(bytemuck::cast_slice(pa_mu_slice));
    let ph_pa_cdf_h = client.create_from_slice(bytemuck::cast_slice(pa_cdf_slice));
    let ph_pa_pdf_h = client.create_from_slice(bytemuck::cast_slice(pa_pdf_slice));
    let ph_pa_interp_h = client.create_from_slice(bytemuck::cast_slice(pa_interp_slice));
    let ph_ct_aeoff_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_ae_offset));
    let ph_ct_xoff_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_x_offset));
    let ph_ct_eg_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_energy_grid));
    let ph_ct_nx_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_n_x));
    let ph_ct_x_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_x));
    let ph_ct_cdf_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_cdf));
    let ph_ct_p_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_p));
    let ph_ct_interp_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_interp));
    let ph_ct_ndisc_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_n_discrete));
    let ph_ct_neout_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_n_eout));
    let ph_ct_hist_h = client.create_from_slice(bytemuck::cast_slice(&coupled.ct_hist));
    let ph_np_h = client.create_from_slice(bytemuck::cast_slice(&coupled.n_product_per_material));
    let ph_prod_base_h =
        client.create_from_slice(bytemuck::cast_slice(&coupled.prod_base_per_material));
    let ph_pp_base_h =
        client.create_from_slice(bytemuck::cast_slice(&coupled.pp_base_per_material));
    // D1S decay-photon buffers (decay_off keeps them size-1 / gate 0).
    let decay_enabled_h = client.create_from_slice(bytemuck::cast_slice(&decay.decay_enabled));
    let decay_pp_h = client.create_from_slice(bytemuck::cast_slice(&decay.photon_prod));
    let decay_ch_xs_h = client.create_from_slice(bytemuck::cast_slice(&decay.ch_xs));
    let decay_ch_pid_h = client.create_from_slice(bytemuck::cast_slice(&decay.ch_parent_id));
    let decay_ch_emeta_h = client.create_from_slice(bytemuck::cast_slice(&decay.ch_e_meta));
    let decay_ch_e_h = client.create_from_slice(bytemuck::cast_slice(&decay.ch_energies));
    let decay_ch_cdf_h = client.create_from_slice(bytemuck::cast_slice(&decay.ch_intensity_cdf));
    let decay_meta_h = client.create_from_slice(bytemuck::cast_slice(&decay.meta));
    // Per-collision nuclide-selection buffers (issue #74).
    let nuc_macro_h =
        client.create_from_slice(bytemuck::cast_slice(&nuclide_select.nuc_macro_total));
    let nuc_awr_h = client.create_from_slice(bytemuck::cast_slice(&nuclide_select.nuc_awr));
    let nuc_meta_h =
        client.create_from_slice(bytemuck::cast_slice(&nuclide_select.mat_nuclide_meta));
    let nuc_partial_h =
        client.create_from_slice(bytemuck::cast_slice(&nuclide_select.nuc_partial_xs));
    let bank_f64_z = vec![0.0_f64; bank_capacity * crate::common::particle_bank::BANK_F64_STRIDE];
    let bank_u32_z = vec![0u32; bank_capacity * crate::common::particle_bank::BANK_U32_STRIDE];
    let bank_f64_h = client.create_from_slice(bytemuck::cast_slice(&bank_f64_z));
    let bank_u32_h = client.create_from_slice(bytemuck::cast_slice(&bank_u32_z));
    // Per-source index of each banked fission progeny (issue #233 Stage 2), one
    // u32 per bank slot (parallel to `bank_u32`), so we never change the shared
    // bank stride. Written by the kernel at each fission reservation.
    let bank_source_idx_z = vec![0u32; bank_capacity];
    let bank_source_idx_h = client.create_from_slice(bytemuck::cast_slice(&bank_source_idx_z));
    let bank_count_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));
    let bank_overflow_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));
    // Lost-particle diagnostics (issue #289): one u64 atomic counter plus a
    // small stride-packed record buffer. Both are zero-initialised per launch,
    // and the dispatch accumulates the counts across launches.
    let lost_count_h = client.create_from_slice(bytemuck::cast_slice(&[0u64]));
    let lost_records_z = vec![
        0.0_f64;
        crate::common::lost_particles::LOST_RECORD_CAPACITY
            * crate::common::lost_particles::LOST_F64_STRIDE
    ];
    let lost_f64_h = client.create_from_slice(bytemuck::cast_slice(&lost_records_z));

    let out_alive_h = client.empty(std::mem::size_of_val(seeds));
    let out_steps_h = client.empty(std::mem::size_of_val(seeds));
    let out_e_h = client.empty(std::mem::size_of_val(energies_in));
    let total_out_len = tallies.total_out_len() as usize;
    // Tally-variance buffers (issue #233). `PerHistory` (Stage 1) doubles
    // `tally_out` (2nd half = sum_sq); `PerSource` (Stage 2) writes the
    // per-source `src_acc` instead, so `tally_out` is an unused size-1 dummy.
    // Both per-history modes share the thread-private touched-list + spill.
    // `spill_cap` is the PROVEN worst-case distinct bins a single history can
    // touch beyond the register list (at most one flat bin per tally per step,
    // over `max_steps` steps; `0` -> size-1 dummy). This is the value the kernel
    // indexes `spill_base = ABSOLUTE_POS * spill_cap` with, and the dispatch
    // chunk-sizer uses the same `per_history_spill_cap` so the two never drift.
    let per_history = variance.per_history();
    let per_source = variance.per_source();
    // Issue #234: the mesh path accumulates straight into `src_acc` (no
    // touched-list), so it allocates NO per-history spill (`spill_cap == 0`).
    let mesh_direct = variance.mesh_direct();
    let spill_cap = if mesh_direct {
        0
    } else {
        per_history_spill_cap(total_out_len, max_steps, tallies.n_tallies() as usize)
    };
    let alloc_out_len = match variance {
        TallyVarianceMode::PerHistory => total_out_len * 2,
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
    // Uninitialized device memory: the kernel writes every spill slot (each read
    // guarded by the per-history `spill_count`) before reading it, so the spill
    // needs no zero-init and no host->device upload (unlike the accumulating
    // `tally_out` / `src_acc`). Removes an ~n*spill_cap-word host zero + copy per
    // launch, which dominated large-tally throughput (issue #233 Stage 4).
    let spill_bin_h = client.empty(spill_len * std::mem::size_of::<u32>());
    let spill_val_h = client.empty(spill_len * std::mem::size_of::<f64>());
    // Per-source accumulator (Stage 2): `chunk_sources * total_bins` fixed-point
    // words scatter-written by `source_idx`. Size-1 dummy off the fissile path.
    let total_bins = variance.total_bins() as usize;
    let src_acc_len = match variance {
        TallyVarianceMode::PerSource { chunk_sources, .. }
        | TallyVarianceMode::PerSourceDirect { chunk_sources, .. } => {
            (chunk_sources as usize).saturating_mul(total_bins).max(1)
        }
        _ => 1,
    };
    let src_acc_h = client.create_from_slice(bytemuck::cast_slice(&vec![0u64; src_acc_len]));
    // Per-particle source index (Stage 2): a fission-generation launch uploads
    // the drained progeny's banked indices; a source launch uses identity
    // `[0..n)`; off the fissile path a size-1 dummy the kernel never reads.
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

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        multi_cell_transport_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(energies_h, n),
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
            BufferArg::from_raw_parts(coarse_log_grid_h, coarse_log_energy_grid.len()),
            BufferArg::from_raw_parts(coarse_meta_h, coarse_meta.len()),
            BufferArg::from_raw_parts(fine_log_grid_h, fine_log_energy_grid.len()),
            BufferArg::from_raw_parts(fine_meta_h, fine_meta.len()),
            BufferArg::from_raw_parts(xs_e_h, xs_elastic_per_material.len()),
            BufferArg::from_raw_parts(xs_a_h, xs_absorption_per_material.len()),
            BufferArg::from_raw_parts(xs_i_h, xs_inelastic_per_material.len()),
            BufferArg::from_raw_parts(xs_f_h, xs_fission_per_material.len()),
            BufferArg::from_raw_parts(nu_h, nu_bar_per_material.len()),
            BufferArg::from_raw_parts(beta_h, beta_delayed_per_material.len()),
            BufferArg::from_raw_parts(fission_eout_kind_h, fission_eout_kind_per_material.len()),
            BufferArg::from_raw_parts(
                fission_eout_nae_h,
                fission_eout_n_energies_per_material.len(),
            ),
            BufferArg::from_raw_parts(fission_eout_ae_off_h, fission_eout_ae_offset.len()),
            // Lengths reflect the padded slices (>= 1) so they match the
            // allocated buffers when the tight arrays are empty (issue #104).
            BufferArg::from_raw_parts(fission_eout_eg_h, fission_eout_eg_slice.len()),
            BufferArg::from_raw_parts(fission_eout_n_x_h, fission_eout_n_x_slice.len()),
            BufferArg::from_raw_parts(fission_eout_x_off_h, fission_eout_x_off_slice.len()),
            BufferArg::from_raw_parts(fission_eout_x_h, fission_eout_x_slice.len()),
            BufferArg::from_raw_parts(fission_eout_cdf_h, fission_eout_cdf_slice.len()),
            BufferArg::from_raw_parts(fission_eout_p_h, fission_eout_p_slice.len()),
            BufferArg::from_raw_parts(fission_eout_interp_h, fission_eout_interp_slice.len()),
            BufferArg::from_raw_parts(mat_f64_meta_h, mat_f64_meta.len()),
            BufferArg::from_raw_parts(xs_i_mt_h, xs_inelastic_per_mt_sparse.len()),
            BufferArg::from_raw_parts(q_inel_mt_h, q_inelastic_per_mt.len()),
            BufferArg::from_raw_parts(yield_mt_h, yield_per_mt_sparse.len()),
            BufferArg::from_raw_parts(permt_meta_h, permt_meta.len()),
            BufferArg::from_raw_parts(ang_n_e_h, angle_n_energies.len()),
            BufferArg::from_raw_parts(ang_ae_off_h, angle_ae_offset.len()),
            BufferArg::from_raw_parts(ang_eg_h, angle_energy_grid.len()),
            BufferArg::from_raw_parts(ang_n_mu_h, angle_n_mu.len()),
            BufferArg::from_raw_parts(ang_mu_off_h, angle_mu_offset.len()),
            BufferArg::from_raw_parts(ang_mu_h, angle_mu.len()),
            BufferArg::from_raw_parts(ang_cdf_h, angle_cdf.len()),
            BufferArg::from_raw_parts(ang_pdf_h, angle_pdf.len()),
            BufferArg::from_raw_parts(ang_interp_h, angle_interp.len()),
            BufferArg::from_raw_parts(mt_slot_meta_h, mt_slot_u32_meta.len()),
            BufferArg::from_raw_parts(eout_ae_off_h, eout_ae_offset.len()),
            BufferArg::from_raw_parts(eout_eg_h, eout_energy_grid.len()),
            BufferArg::from_raw_parts(eout_n_x_h, eout_n_x.len()),
            BufferArg::from_raw_parts(eout_x_off_h, eout_x_offset.len()),
            BufferArg::from_raw_parts(eout_x_h, eout_x.len()),
            BufferArg::from_raw_parts(eout_cdf_h, eout_cdf.len()),
            BufferArg::from_raw_parts(eout_hist_h, eout_histogram_interp.len()),
            BufferArg::from_raw_parts(eout_p_h, eout_p.len()),
            BufferArg::from_raw_parts(eout_interp_h, eout_interp.len()),
            BufferArg::from_raw_parts(eout_n_discrete_h, eout_n_discrete.len()),
            BufferArg::from_raw_parts(corr_ae_off_h, corr_ae_offset.len()),
            BufferArg::from_raw_parts(corr_eg_h, corr_energy_grid.len()),
            BufferArg::from_raw_parts(corr_n_x_h, corr_n_x.len()),
            BufferArg::from_raw_parts(corr_x_off_h, corr_x_offset.len()),
            BufferArg::from_raw_parts(corr_x_h, corr_x.len()),
            BufferArg::from_raw_parts(corr_cdf_h, corr_cdf.len()),
            BufferArg::from_raw_parts(corr_p_h, corr_p.len()),
            BufferArg::from_raw_parts(corr_interp_h, corr_interp.len()),
            BufferArg::from_raw_parts(corr_n_discrete_h, corr_n_discrete.len()),
            BufferArg::from_raw_parts(corr_n_mu_h, corr_n_mu.len()),
            BufferArg::from_raw_parts(corr_mu_off_h, corr_mu_offset.len()),
            BufferArg::from_raw_parts(corr_mu_h, corr_mu.len()),
            BufferArg::from_raw_parts(corr_mu_cdf_h, corr_mu_cdf.len()),
            BufferArg::from_raw_parts(corr_mu_pdf_h, corr_mu_pdf.len()),
            BufferArg::from_raw_parts(corr_mu_interp_h, corr_mu_interp.len()),
            BufferArg::from_raw_parts(el_ae_h, elastic_angle_n_energies.len()),
            BufferArg::from_raw_parts(el_ae_off_h, elastic_angle_ae_offset.len()),
            BufferArg::from_raw_parts(el_eg_h, elastic_angle_energy_grid.len()),
            BufferArg::from_raw_parts(el_n_mu_h, elastic_angle_n_mu.len()),
            BufferArg::from_raw_parts(el_mu_off_h, elastic_angle_mu_offset.len()),
            BufferArg::from_raw_parts(el_mu_h, elastic_angle_mu.len()),
            BufferArg::from_raw_parts(el_cdf_h, elastic_angle_cdf.len()),
            BufferArg::from_raw_parts(el_pdf_h, elastic_angle_pdf.len()),
            BufferArg::from_raw_parts(el_interp_h, elastic_angle_interp.len()),
            BufferArg::from_raw_parts(km_ae_off_h, km_ae_offset.len()),
            BufferArg::from_raw_parts(km_eg_h, km_energy_grid.len()),
            BufferArg::from_raw_parts(km_interp_h, km_interp.len()),
            BufferArg::from_raw_parts(km_n_disc_h, km_n_discrete.len()),
            BufferArg::from_raw_parts(km_n_x_h, km_n_x.len()),
            BufferArg::from_raw_parts(km_x_off_h, km_x_offset.len()),
            BufferArg::from_raw_parts(km_x_h, km_x.len()),
            BufferArg::from_raw_parts(km_p_h, km_p.len()),
            BufferArg::from_raw_parts(km_c_h, km_c.len()),
            BufferArg::from_raw_parts(km_r_h, km_r.len()),
            BufferArg::from_raw_parts(km_a_h, km_a.len()),
            BufferArg::from_raw_parts(evap_ae_off_h, evap_ae_offset.len()),
            BufferArg::from_raw_parts(evap_theta_off_h, evap_theta_offset.len()),
            BufferArg::from_raw_parts(evap_eg_h, evap_energy_grid.len()),
            BufferArg::from_raw_parts(evap_theta_h, evap_theta.len()),
            BufferArg::from_raw_parts(evap_u_h, evap_u.len()),
            BufferArg::from_raw_parts(maxwell_ae_off_h, maxwell_ae_offset.len()),
            BufferArg::from_raw_parts(maxwell_eg_h, maxwell_energy_grid.len()),
            BufferArg::from_raw_parts(maxwell_theta_h, maxwell_theta.len()),
            BufferArg::from_raw_parts(watt_ae_off_h, watt_ae_offset.len()),
            BufferArg::from_raw_parts(watt_eg_h, watt_energy_grid.len()),
            BufferArg::from_raw_parts(watt_ab_h, watt_ab.len()),
            BufferArg::from_raw_parts(mt_slot_f64_h, mt_slot_f64_meta.len()),
            BufferArg::from_raw_parts(urr_meta_h, urr_meta.len()),
            BufferArg::from_raw_parts(urr_ae_off_h, urr_ae_offset.len()),
            BufferArg::from_raw_parts(urr_cdf_off_h, urr_cdf_offset.len()),
            BufferArg::from_raw_parts(urr_energy_grid_h, urr_energy_grid.len()),
            BufferArg::from_raw_parts(urr_cdf_h, urr_cdf.len()),
            BufferArg::from_raw_parts(urr_xs_h, urr_xs.len()),
            BufferArg::from_raw_parts(urr_atom_density_h, urr_atom_density.len()),
            BufferArg::from_raw_parts(tally_score_kinds_h, tallies.score_kinds.len()),
            BufferArg::from_raw_parts(tally_cell_to_bin_h, tallies.cell_to_bin.len()),
            BufferArg::from_raw_parts(tally_n_cells_h, tallies.n_cells_per_tally.len()),
            BufferArg::from_raw_parts(tally_edges_offsets_h, tallies.edges_offsets.len()),
            BufferArg::from_raw_parts(tally_n_bins_h, tallies.n_bins_per_tally.len()),
            BufferArg::from_raw_parts(tally_log_edges_h, tallies.log_edges.len()),
            BufferArg::from_raw_parts(tally_out_offsets_h, tallies.out_offsets.len()),
            BufferArg::from_raw_parts(tally_score_data_h, tallies.score_data.len()),
            BufferArg::from_raw_parts(xs_score_h, xs_score_slice.len()),
            BufferArg::from_raw_parts(tally_fixed_point_scales_h, tallies.fixed_point_scales.len()),
            BufferArg::from_raw_parts(tally_is_collision_h, tallies.is_collision.len()),
            BufferArg::from_raw_parts(tally_score_mt_h, tallies.score_mt.len()),
            // ---- mesh (voxel) tally binning (issue #234) ----
            BufferArg::from_raw_parts(tally_n_mesh_h, tallies.n_mesh_per_tally.len()),
            BufferArg::from_raw_parts(tally_mesh_kind_h, tallies.mesh_kind.len()),
            BufferArg::from_raw_parts(
                tally_mesh_params_offsets_h,
                tallies.mesh_params_offsets.len(),
            ),
            BufferArg::from_raw_parts(tally_mesh_params_h, mesh_params_padded.len()),
            // ---- energy-function weighting (issue #271) ----
            BufferArg::from_raw_parts(tally_efunc_offsets_h, tallies.efunc_offsets.len()),
            BufferArg::from_raw_parts(tally_efunc_params_h, efunc_params_padded.len()),
            // ---- survival biasing (implicit capture) ----
            BufferArg::from_raw_parts(survival_h, survival.params.len()),
            // ---- free-gas threshold (issue #102) ----
            BufferArg::from_raw_parts(free_gas_threshold_h, free_gas_threshold_slice.len()),
            // ---- coupled neutron->photon (S4b) ----
            BufferArg::from_raw_parts(coupled_enabled_h, coupled.coupled_enabled.len()),
            BufferArg::from_raw_parts(ph_prod_h, coupled.photon_prod.len()),
            BufferArg::from_raw_parts(ph_rxn_xs_h, coupled.rxn_xs.len()),
            BufferArg::from_raw_parts(ph_prod_rxn_idx_h, coupled.prod_rxn_idx.len()),
            BufferArg::from_raw_parts(ph_prod_yield_h, coupled.prod_yield_grid.len()),
            BufferArg::from_raw_parts(ph_prod_kind_h, coupled.prod_eout_kind.len()),
            BufferArg::from_raw_parts(ph_prod_line_h, coupled.prod_line_energy.len()),
            BufferArg::from_raw_parts(ph_prod_pflag_h, coupled.prod_primary_flag.len()),
            BufferArg::from_raw_parts(ph_prod_awr_h, coupled.prod_awr.len()),
            BufferArg::from_raw_parts(ph_prod_slot_h, coupled.prod_dist_slot.len()),
            BufferArg::from_raw_parts(ph_pa_ne_h, coupled.pa_n_energies.len()),
            BufferArg::from_raw_parts(ph_pa_aeoff_h, coupled.pa_ae_offset.len()),
            BufferArg::from_raw_parts(ph_pa_muoff_h, pa_mu_off_slice.len()),
            BufferArg::from_raw_parts(ph_pa_eg_h, pa_eg_slice.len()),
            BufferArg::from_raw_parts(ph_pa_nmu_h, pa_nmu_slice.len()),
            BufferArg::from_raw_parts(ph_pa_mu_h, pa_mu_slice.len()),
            BufferArg::from_raw_parts(ph_pa_cdf_h, pa_cdf_slice.len()),
            BufferArg::from_raw_parts(ph_pa_pdf_h, pa_pdf_slice.len()),
            BufferArg::from_raw_parts(ph_pa_interp_h, pa_interp_slice.len()),
            BufferArg::from_raw_parts(ph_ct_aeoff_h, coupled.ct_ae_offset.len()),
            BufferArg::from_raw_parts(ph_ct_xoff_h, coupled.ct_x_offset.len()),
            BufferArg::from_raw_parts(ph_ct_eg_h, coupled.ct_energy_grid.len()),
            BufferArg::from_raw_parts(ph_ct_nx_h, coupled.ct_n_x.len()),
            BufferArg::from_raw_parts(ph_ct_x_h, coupled.ct_x.len()),
            BufferArg::from_raw_parts(ph_ct_cdf_h, coupled.ct_cdf.len()),
            BufferArg::from_raw_parts(ph_ct_p_h, coupled.ct_p.len()),
            BufferArg::from_raw_parts(ph_ct_interp_h, coupled.ct_interp.len()),
            BufferArg::from_raw_parts(ph_ct_ndisc_h, coupled.ct_n_discrete.len()),
            BufferArg::from_raw_parts(ph_ct_neout_h, coupled.ct_n_eout.len()),
            BufferArg::from_raw_parts(ph_ct_hist_h, coupled.ct_hist.len()),
            BufferArg::from_raw_parts(ph_np_h, coupled.n_product_per_material.len()),
            BufferArg::from_raw_parts(ph_prod_base_h, coupled.prod_base_per_material.len()),
            BufferArg::from_raw_parts(ph_pp_base_h, coupled.pp_base_per_material.len()),
            BufferArg::from_raw_parts(decay_enabled_h, decay.decay_enabled.len()),
            BufferArg::from_raw_parts(decay_pp_h, decay.photon_prod.len()),
            BufferArg::from_raw_parts(decay_ch_xs_h, decay.ch_xs.len()),
            BufferArg::from_raw_parts(decay_ch_pid_h, decay.ch_parent_id.len()),
            BufferArg::from_raw_parts(decay_ch_emeta_h, decay.ch_e_meta.len()),
            BufferArg::from_raw_parts(decay_ch_e_h, decay.ch_energies.len()),
            BufferArg::from_raw_parts(decay_ch_cdf_h, decay.ch_intensity_cdf.len()),
            BufferArg::from_raw_parts(decay_meta_h, decay.meta.len()),
            BufferArg::from_raw_parts(nuc_macro_h, nuclide_select.nuc_macro_total.len()),
            BufferArg::from_raw_parts(nuc_awr_h, nuclide_select.nuc_awr.len()),
            BufferArg::from_raw_parts(nuc_meta_h, nuclide_select.mat_nuclide_meta.len()),
            BufferArg::from_raw_parts(nuc_partial_h, nuclide_select.nuc_partial_xs.len()),
            BufferArg::from_raw_parts(fission_bank_enabled_h, fission_bank.enabled.len()),
            BufferArg::from_raw_parts(
                bank_f64_h.clone(),
                bank_capacity * crate::common::particle_bank::BANK_F64_STRIDE,
            ),
            BufferArg::from_raw_parts(
                bank_u32_h.clone(),
                bank_capacity * crate::common::particle_bank::BANK_U32_STRIDE,
            ),
            BufferArg::from_raw_parts(bank_count_h.clone(), 1),
            BufferArg::from_raw_parts(bank_overflow_h.clone(), 1),
            BufferArg::from_raw_parts(out_alive_h.clone(), n),
            BufferArg::from_raw_parts(out_steps_h.clone(), n),
            BufferArg::from_raw_parts(out_e_h.clone(), n),
            BufferArg::from_raw_parts(tally_out_h.clone(), alloc_out_len.max(1)),
            BufferArg::from_raw_parts(spill_bin_h.clone(), spill_len),
            BufferArg::from_raw_parts(spill_val_h.clone(), spill_len),
            BufferArg::from_raw_parts(source_idx_h.clone(), source_idx_len),
            BufferArg::from_raw_parts(src_acc_h.clone(), src_acc_len),
            BufferArg::from_raw_parts(bank_source_idx_h.clone(), bank_capacity),
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
    // Tally results (issue #233). `PerStep` / `PerHistory` read `tally_out`
    // (first `total_out_len` words = sum; second half = sum_sq for PerHistory).
    // `PerSource` (fissile) and `PerSourceDirect` (mesh, issue #234) write no
    // `tally_out`; the dispatch reconstructs the tally from the per-source
    // accumulator (`src_acc`) instead.
    let uses_src_acc = variance.uses_src_acc();
    let (tally_outputs, tally_sum_sq, src_acc) = if uses_src_acc {
        let src_acc_bits: Vec<u64> =
            bytemuck::cast_slice(&client.read_one(src_acc_h).unwrap()).to_vec();
        (Vec::new(), Vec::new(), src_acc_bits)
    } else {
        let tally_out_bits: Vec<u64> =
            bytemuck::cast_slice(&client.read_one(tally_out_h).unwrap()).to_vec();
        let sum_len = total_out_len.max(1);
        let outputs = unpack_tally_outputs(&tally_out_bits[..sum_len], tallies);
        let sum_sq = if per_history {
            unpack_tally_sum_sq(&tally_out_bits[total_out_len..total_out_len * 2], tallies)
        } else {
            Vec::new()
        };
        (outputs, sum_sq, Vec::new())
    };
    // Read back the device particle bank: coupled photons (S4b), fission
    // progeny (#78) and (n,xn) spills (issue #111 phase 2) share it. Overflow
    // > 0 is a hard error per the particle_bank contract -- never a silent drop.
    //
    // The RECORDS are only fetched when something was actually banked. Since
    // phase 2 every neutron run allocates a real bank (an (n,xn) spill can
    // happen on any of them), and copying a `bank_capacity`-sized pair of
    // arrays back per launch would cost more than the spill it is there to
    // catch: on a neutron-only run the bank is normally untouched, so this is
    // the difference between two 8-byte counter reads and ~10 MB of transfer.
    let photon_bank = {
        let count = bytemuck::cast_slice::<u8, u64>(&client.read_one(bank_count_h).unwrap())[0];
        let overflow =
            bytemuck::cast_slice::<u8, u64>(&client.read_one(bank_overflow_h).unwrap())[0];
        assert_eq!(
            overflow, 0,
            "device particle bank overflowed by {overflow} (capacity {bank_capacity}, \
             {count} reservations) -- raise the bank capacity. Silent drop is \
             forbidden by the particle_bank contract."
        );
        let (bank_f64, bank_u32) = if count == 0 {
            (Vec::new(), Vec::new())
        } else {
            (
                bytemuck::cast_slice::<u8, f64>(&client.read_one(bank_f64_h).unwrap()).to_vec(),
                bytemuck::cast_slice::<u8, u32>(&client.read_one(bank_u32_h).unwrap()).to_vec(),
            )
        };
        PhotonBankResult {
            count,
            overflow,
            bank_f64,
            bank_u32,
        }
    };

    // Only meaningful, and only fetched, alongside actual banked records.
    let bank_source_idx = if uses_src_acc && photon_bank.count > 0 {
        bytemuck::cast_slice::<u8, u32>(&client.read_one(bank_source_idx_h).unwrap()).to_vec()
    } else {
        Vec::new()
    };

    let lost = {
        let count = bytemuck::cast_slice::<u8, u64>(&client.read_one(lost_count_h).unwrap())[0];
        let records =
            bytemuck::cast_slice::<u8, f64>(&client.read_one(lost_f64_h).unwrap()).to_vec();
        crate::common::lost_particles::LostParticleResult::from_device(count, &records)
    };

    // (n,xn) secondaries that overflowed the kernel's thread-private pending
    // stack and were handed to the bank (issue #111 phase 2). Counted from the
    // `gen` provenance tag rather than a second device counter; the scan is
    // over the banked records only, which is empty on a run that banks nothing.
    //
    // The `ptype` test is load-bearing, not defensive: `gen` only means
    // "provenance" on a banked NEUTRON. On a banked PHOTON the D1S emission site
    // reuses the same slot for the parent-nuclide id, so a decay photon whose
    // parent id happens to be `BANK_GEN_NXN_SPILL` (2, i.e. the third entry in
    // the D1S parent registry) is indistinguishable from a spilled neutron
    // without it. Reading `gen` unqualified reported thousands of phantom spills
    // on every D1S run and made the coupled dispatch refuse to run at all.
    let n_spilled_secondaries =
        crate::common::particle_bank::count_nxn_spills(&photon_bank.bank_u32, photon_bank.count);

    MultiCellResult {
        alive,
        n_steps,
        final_energies,
        tally_outputs,
        tally_sum_sq,
        src_acc,
        bank_source_idx,
        n_bins_per_tally: tallies.n_bins_per_tally.clone(),
        n_cells,
        photon_bank,
        lost,
        n_spilled_secondaries,
        // CPU-mirror diagnostics only; the kernel does not report stack depth.
        max_pend_depth: 0,
        pend_depth_hist: Vec::new(),
    }
}

/// Pack the per-material f64 scalars (target mass, temperature, Watt `a`,
/// Watt `b`) into one `[n_mat × MAT_F64_COLS]` buffer at the kernel's `COL_*`
/// offsets, collapsing the scalar bindings into one. The
/// `COL_URR_ATOM_DENSITY` column is intentionally left at zero: URR atom
/// density moved to its own per-slab `urr_atom_density` binding (issue #210),
/// but the stride stays `MAT_F64_COLS = 5` so the kernel's hardcoded
/// `mat_f64_meta.len() / 5` material-count derivation and the other columns'
/// offsets are unchanged.
fn pack_mat_f64_meta(
    target_mass: &[f64],
    temperature_k: &[f64],
    fission_a: &[f64],
    fission_b: &[f64],
) -> Vec<f64> {
    let cols = MAT_F64_COLS as usize;
    let n = target_mass.len();
    let mut meta = vec![0.0f64; n * cols];
    for m in 0..n {
        let off = m * cols;
        meta[off + COL_TARGET_MASS as usize] = target_mass[m];
        meta[off + COL_TEMPERATURE_K as usize] = temperature_k[m];
        meta[off + COL_FISSION_A as usize] = fission_a[m];
        meta[off + COL_FISSION_B as usize] = fission_b[m];
        // COL_URR_ATOM_DENSITY intentionally left 0 (dead slot, issue #210).
    }
    meta
}

/// Pack the per-MT-slot u32 metadata arrays into one
/// `[n_slots × MT_SLOT_U32_COLS]` buffer at the kernel's `COL_*` offsets.
/// Collapses the scalar bindings into one, keeping the kernel under the
/// per-stage storage-buffer descriptor budget.
#[allow(clippy::too_many_arguments)]
fn pack_mt_slot_u32_meta(
    eout_kind: &[u32],
    eout_n_energies: &[u32],
    corr_n_energies: &[u32],
    corr_n_components: &[u32],
    scatter_in_cm: &[u32],
    km_n_energies: &[u32],
    evap_n_energies: &[u32],
    evap_n_components: &[u32],
    nbps_n_bodies: &[u32],
    maxwell_n_energies: &[u32],
    watt_n_energies: &[u32],
) -> Vec<u32> {
    let cols = MT_SLOT_U32_COLS as usize;
    let n = eout_kind.len();
    let mut meta = vec![0u32; n * cols];
    for slot in 0..n {
        let off = slot * cols;
        meta[off + COL_EOUT_KIND as usize] = eout_kind[slot];
        meta[off + COL_EOUT_N_ENERGIES as usize] = eout_n_energies[slot];
        meta[off + COL_CORR_N_ENERGIES as usize] = corr_n_energies[slot];
        meta[off + COL_CORR_N_COMPONENTS as usize] = corr_n_components[slot];
        meta[off + COL_SCATTER_IN_CM as usize] = scatter_in_cm[slot];
        meta[off + COL_KM_N_ENERGIES as usize] = km_n_energies[slot];
        meta[off + COL_EVAP_N_ENERGIES as usize] = evap_n_energies[slot];
        meta[off + COL_EVAP_N_COMPONENTS as usize] = evap_n_components[slot];
        meta[off + COL_NBPS_N_BODIES as usize] = nbps_n_bodies[slot];
        meta[off + COL_MAXWELL_N_ENERGIES as usize] = maxwell_n_energies[slot];
        meta[off + COL_WATT_N_ENERGIES as usize] = watt_n_energies[slot];
    }
    meta
}

/// Pack the per-MT-slot f64 scalars (maxwell_u, watt_u, nbps_total_mass)
/// into one `[n_slots × MT_SLOT_F64_COLS]` buffer. Evaporation's
/// restriction energy `u` is NOT packed here -- it is incident-energy
/// dependent (multi-law applicability) and lives in its own per-grid
/// `evap_u` storage buffer; `COL_EVAP_U` stays zero.
fn pack_mt_slot_f64_meta(maxwell_u: &[f64], watt_u: &[f64], nbps_total_mass: &[f64]) -> Vec<f64> {
    let cols = MT_SLOT_F64_COLS as usize;
    let n = nbps_total_mass.len();
    let mut meta = vec![0.0_f64; n * cols];
    for slot in 0..n {
        let off = slot * cols;
        meta[off + COL_MAXWELL_U as usize] = maxwell_u[slot];
        meta[off + COL_WATT_U as usize] = watt_u[slot];
        meta[off + COL_NBPS_TOTAL_MASS as usize] = nbps_total_mass[slot];
    }
    meta
}

/// Interleave Watt's `a` and `b` parameters into one stride-2 buffer
/// (`ab[i*2] = a[i]`, `ab[i*2 + 1] = b[i]`) to save a descriptor binding.
fn pack_watt_ab_interleaved(watt_a: &[f64], watt_b: &[f64]) -> Vec<f64> {
    let n = watt_a.len();
    let mut ab = vec![0.0_f64; n * 2];
    for i in 0..n {
        ab[i * 2] = watt_a[i];
        ab[i * 2 + 1] = watt_b[i];
    }
    ab
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_watt_ab_interleaves_pairs() {
        let ab = pack_watt_ab_interleaved(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
        assert_eq!(ab, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn pack_mat_f64_meta_places_each_scalar_at_its_column() {
        // Two materials; verify stride and that each scalar lands at its
        // COL_* offset (robust to the actual column numbering).
        let meta = pack_mat_f64_meta(&[1.0, 2.0], &[300.0, 600.0], &[0.1, 0.2], &[0.3, 0.4]);
        let cols = MAT_F64_COLS as usize;
        assert_eq!(meta.len(), 2 * cols);
        assert_eq!(meta[COL_TARGET_MASS as usize], 1.0);
        assert_eq!(meta[COL_TEMPERATURE_K as usize], 300.0);
        assert_eq!(meta[COL_FISSION_A as usize], 0.1);
        assert_eq!(meta[COL_FISSION_B as usize], 0.3);
        // The URR-atom-density column is a dead slot now (issue #210): it must
        // stay zero so the kernel never reads a stale per-material value.
        assert_eq!(meta[COL_URR_ATOM_DENSITY as usize], 0.0);
        assert_eq!(meta[cols + COL_TARGET_MASS as usize], 2.0);
        assert_eq!(meta[cols + COL_URR_ATOM_DENSITY as usize], 0.0);
    }

    #[test]
    fn pack_mt_slot_u32_meta_strides_and_places_columns() {
        let z = [0u32, 0u32];
        let meta = pack_mt_slot_u32_meta(&[7, 8], &z, &z, &z, &z, &z, &z, &z, &z, &z, &[3, 4]);
        let cols = MT_SLOT_U32_COLS as usize;
        assert_eq!(meta.len(), 2 * cols);
        assert_eq!(meta[COL_EOUT_KIND as usize], 7);
        assert_eq!(meta[COL_WATT_N_ENERGIES as usize], 3);
        assert_eq!(meta[cols + COL_EOUT_KIND as usize], 8);
        assert_eq!(meta[cols + COL_WATT_N_ENERGIES as usize], 4);
    }
}
