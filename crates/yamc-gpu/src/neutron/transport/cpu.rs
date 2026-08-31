//! Sequential CPU mirror of the GPU kernel.
//!
//! Thin driver around [`super::core::transport_one_particle`] -- one
//! particle at a time, no parallelism. The rayon-parallel variant
//! lives in [`super::cpu_rayon`] and calls the same per-particle
//! function so both stay bit-for-bit aligned with the cubecl
//! kernel.

use super::shared::{
    transport_one_particle, validate_transport_inputs, CollisionRecord, PendDrain, TransportInputs,
};
use super::{unpack_tally_outputs, MultiCellResult, NuclideSelectInputs};
use crate::common::geometry::bvh_cell_finding::build_and_flatten_bvh;
use crate::common::tallies::TalliesPack;

/// CPU equivalent of `run_multi_cell_transport`. Mirrors the GPU
/// kernel's algorithm exactly: same PCG-32, same linear cell scan,
/// same XS lookup, same boundary scan, same Marsaglia rejection,
/// same fixed-point u64 tally accumulation. Sequential per-particle.
///
/// `pend_drain` selects the in-thread (n,xn) queue's drain order. Production
/// callers pass `PendDrain::Fifo`, the kernel's order; the alternative exists
/// only so the issue-#111 order-independence test can prove that the per-
/// secondary identity seeding makes the order unobservable.
///
/// `capture_trace` drives the issue-#40 matched-stream diff harness: when
/// `true`, the returned `Vec<Vec<CollisionRecord>>` holds one per-particle
/// collision trace (energy in/out + reaction class per collision) for
/// diffing against the production CPU's track capture; when `false` it is an
/// empty `Vec` and the per-step recording branch is never entered, so the
/// stream and timing are byte-identical to the untraced kernel/twin.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_cell_transport_cpu(
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
    // Concatenated per-material coarse grids backing the per-MT inelastic
    // buffers (issue #212); each material owns its coarse grid, described by
    // `coarse_meta`. Mirrors the GPU launch so the CPU twin indexes the per-MT
    // buffers identically.
    coarse_log_energy_grid: &[f64],
    // Packed `[n_materials × COARSE_META_COLS]` per-material coarse-grid
    // descriptor (issue #212). Mirrors the GPU launch's `coarse_meta` binding.
    coarse_meta: &[u32],
    // Concatenated per-material FINE grids backing the resonance-critical
    // aggregate macro XS + nuc buffers (issue #212); each material owns its fine
    // grid, described by `fine_meta`. Mirrors the GPU launch.
    fine_log_energy_grid: &[f64],
    // Packed `[n_materials × FINE_META_COLS]` per-material fine-grid descriptor
    // (issue #212). Mirrors the GPU launch's `fine_meta` binding.
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
    survival: &super::SurvivalBiasingInputs,
    nuclide_select: &NuclideSelectInputs,
    fission_bank: &super::FissionBankInputs,
    max_steps: u32,
    // Free-gas resonance/thermal cutoff multiplier (model option, default
    // 400.0); the regime boundary is `free_gas_threshold * kT`. Mirrors the
    // kernel's `free_gas_threshold` buffer so the twin stays bit-equivalent
    // (issue #102).
    free_gas_threshold: f64,
    capture_trace: bool,
    // Drain order of the in-thread (n,xn) queue. Production and the kernel are
    // `PendDrain::Fifo`; `Lifo` is the issue-#111 verification instrument (see
    // `PendDrain`), which the order-independence test and the matched-stream
    // harness use.
    pend_drain: PendDrain,
) -> (MultiCellResult, Vec<Vec<CollisionRecord>>) {
    let n = seeds.len();
    let n_cells = cell_aabbs.len() / 6;
    let n_surfaces = surface_types.len();
    let n_grid = log_energy_grid.len();
    let n_materials = target_mass_per_material.len();

    // SCORE_PER_MT lookup data: pad to a single zero slot when the
    // model has no per-MT tallies so the indexing math stays valid.
    let xs_score_pad: Vec<f64>;
    let xs_score_per_mt_slice: &[f64] = if xs_score_per_mt.is_empty() {
        xs_score_pad = vec![0.0; n_materials.max(1) * n_grid.max(1)];
        &xs_score_pad
    } else {
        xs_score_per_mt
    };
    let n_score_mts: usize = if xs_score_per_mt_slice.is_empty() {
        0
    } else {
        xs_score_per_mt_slice.len() / (n_materials.max(1) * n_grid.max(1))
    };

    // Per-MT inelastic pools are keyed per-(material, nuclide) slab (#74
    // Stages 2a / 2b); single-nuclide materials give `n_slab == n_materials`.
    let n_slab = nuclide_select.nuc_awr.len();
    validate_transport_inputs(
        tallies,
        surface_types,
        surface_boundaries,
        cell_to_material,
        n_cells,
        n_materials,
        n_slab,
        coarse_meta,
        fine_meta,
        &nuclide_select.mat_nuclide_meta,
        xs_elastic_per_material,
        xs_absorption_per_material,
        xs_inelastic_per_material,
        xs_fission_per_material,
        nu_bar_per_material,
        beta_delayed_per_material,
        fission_a_per_material,
        fission_b_per_material,
        fission_eout_kind_per_material,
        fission_eout_n_energies_per_material,
        fission_eout_ae_offset,
        fission_eout_energy_grid_per_material,
        fission_eout_n_x_per_material,
        fission_eout_x_offset,
        fission_eout_x_per_material,
        fission_eout_cdf_per_material,
        fission_eout_p_per_material,
        fission_eout_interp_per_material,
        xs_inelastic_per_mt_sparse,
        q_inelastic_per_mt,
        yield_per_mt_sparse,
        permt_meta,
        angle_n_energies,
        angle_ae_offset,
        angle_energy_grid,
        angle_n_mu,
        angle_mu_offset,
        angle_mu,
        angle_cdf,
        scatter_in_cm_per_mt,
        eout_kind,
        eout_n_energies,
        eout_ae_offset,
        eout_energy_grid,
        eout_n_x,
        eout_x_offset,
        eout_x,
        eout_cdf,
        eout_histogram_interp,
        eout_p,
        eout_interp,
        eout_n_discrete,
        corr_n_energies,
        corr_n_components,
        corr_ae_offset,
        corr_energy_grid,
        corr_n_x,
        corr_x_offset,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu_offset,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
    );

    let total_out_len = tallies.total_out_len() as usize;
    let mut tally_acc: Vec<u64> = vec![0u64; total_out_len];

    let (bvh_aabbs, bvh_meta, bvh_prims, bvh_unb) = build_and_flatten_bvh(cell_aabbs);

    let inputs = TransportInputs {
        pend_drain,
        seeds,
        energies_in,
        positions_in,
        directions_in,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        region_program,
        bvh_aabbs: &bvh_aabbs,
        bvh_meta: &bvh_meta,
        bvh_prims: &bvh_prims,
        bvh_unb: &bvh_unb,
        log_energy_grid,
        // Per-material coarse grids (issue #212): the per-MT inelastic buffers
        // ride each material's own coarse grid, matching the GPU launch.
        coarse_log_energy_grid,
        coarse_meta,
        // Per-material fine grids (issue #212): the aggregate macro XS + nuc
        // buffers ride each material's own fine grid, matching the GPU launch.
        fine_log_energy_grid,
        fine_meta,
        xs_elastic_per_material,
        xs_absorption_per_material,
        xs_inelastic_per_material,
        xs_fission_per_material,
        nu_bar_per_material,
        beta_delayed_per_material,
        target_mass_per_material,
        temperature_k_per_material,
        nuc_macro_total: &nuclide_select.nuc_macro_total,
        nuc_awr: &nuclide_select.nuc_awr,
        mat_nuclide_meta: &nuclide_select.mat_nuclide_meta,
        nuc_partial_xs: &nuclide_select.nuc_partial_xs,
        fission_a_per_material,
        fission_b_per_material,
        fission_eout_kind_per_material,
        fission_eout_n_energies_per_material,
        fission_eout_ae_offset,
        fission_eout_energy_grid_per_material,
        fission_eout_n_x_per_material,
        fission_eout_x_offset,
        fission_eout_x_per_material,
        fission_eout_cdf_per_material,
        fission_eout_p_per_material,
        fission_eout_interp_per_material,
        xs_inelastic_per_mt_sparse,
        q_inelastic_per_mt,
        yield_per_mt_sparse,
        permt_meta,
        scatter_in_cm_per_mt,
        angle_n_energies,
        angle_ae_offset,
        angle_energy_grid,
        angle_n_mu,
        angle_mu_offset,
        angle_mu,
        angle_cdf,
        angle_pdf,
        angle_interp,
        eout_kind,
        eout_n_energies,
        eout_ae_offset,
        eout_energy_grid,
        eout_n_x,
        eout_x_offset,
        eout_x,
        eout_cdf,
        eout_histogram_interp,
        eout_p,
        eout_interp,
        eout_n_discrete,
        corr_n_energies,
        corr_n_components,
        corr_ae_offset,
        corr_energy_grid,
        corr_n_x,
        corr_x_offset,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        corr_n_mu,
        corr_mu_offset,
        corr_mu,
        corr_mu_cdf,
        corr_mu_pdf,
        corr_mu_interp,
        elastic_angle_n_energies,
        elastic_angle_ae_offset,
        elastic_angle_energy_grid,
        elastic_angle_n_mu,
        elastic_angle_mu_offset,
        elastic_angle_mu,
        elastic_angle_cdf,
        elastic_angle_pdf,
        elastic_angle_interp,
        km_n_energies,
        km_ae_offset,
        km_energy_grid,
        km_interp,
        km_n_discrete,
        km_n_x,
        km_x_offset,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        evap_n_energies,
        evap_n_components,
        evap_ae_offset,
        evap_theta_offset,
        evap_energy_grid,
        evap_theta,
        evap_u,
        nbps_n_bodies,
        nbps_total_mass,
        maxwell_n_energies,
        maxwell_ae_offset,
        maxwell_energy_grid,
        maxwell_theta,
        maxwell_u,
        watt_n_energies,
        watt_ae_offset,
        watt_energy_grid,
        watt_a,
        watt_b,
        watt_u,
        tallies,
        xs_score_per_mt_slice,
        n_score_mts,
        n_cells,
        n_surfaces,
        n_grid,
        max_steps,
        survival_enabled: survival.enabled(),
        weight_cutoff: survival.weight_cutoff(),
        weight_survive: survival.weight_survive(),
        fission_bank_enabled: fission_bank.is_on(),
        free_gas_threshold,
        urr_meta,
        urr_ae_offset,
        urr_cdf_offset,
        urr_energy_grid,
        urr_cdf,
        urr_xs,
        urr_atom_density,
    };

    let mut alive_out = Vec::with_capacity(n);
    let mut n_steps_out = Vec::with_capacity(n);
    let mut final_energies = Vec::with_capacity(n);
    let mut traces: Vec<Vec<CollisionRecord>> = if capture_trace {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    // Lost-particle diagnostics (issue #289), mirroring the kernel's counter +
    // capped record list so the mirror reports the same losses.
    let mut lost = crate::common::lost_particles::LostParticleResult::default();
    let mut n_spilled_secondaries = 0u64;
    let mut max_pend_depth = 0u32;
    for i in 0..n {
        let mut rec: Vec<CollisionRecord> = Vec::new();
        let outcome = transport_one_particle(
            &inputs,
            i,
            &mut tally_acc,
            if capture_trace { Some(&mut rec) } else { None },
        );
        alive_out.push(outcome.alive);
        n_steps_out.push(outcome.n_steps);
        final_energies.push(outcome.final_energy);
        n_spilled_secondaries += outcome.n_spilled as u64;
        max_pend_depth = max_pend_depth.max(outcome.max_pend_depth);
        if let Some(record) = outcome.lost {
            lost.count += 1;
            if lost.records.len() < crate::common::lost_particles::LOST_RECORD_CAPACITY {
                lost.records.push(record);
            }
        }
        if capture_trace {
            traces.push(rec);
        }
    }

    let tally_outputs = unpack_tally_outputs(&tally_acc, tallies);

    (
        MultiCellResult {
            alive: alive_out,
            n_steps: n_steps_out,
            final_energies,
            tally_outputs,
            // The CPU mirror is the per-step (mean-only) reference; it emits no
            // per-history sum-of-squares or per-source accumulator (issue #233
            // batch-free variance is GPU-only).
            tally_sum_sq: Vec::new(),
            src_acc: Vec::new(),
            bank_source_idx: Vec::new(),
            n_bins_per_tally: tallies.n_bins_per_tally.clone(),
            n_cells,
            // The CPU mirror does not emit coupled photons (S4b is GPU-only);
            // the bank is empty.
            photon_bank: super::PhotonBankResult::default(),
            lost,
            n_spilled_secondaries,
            max_pend_depth,
        },
        traces,
    )
}
