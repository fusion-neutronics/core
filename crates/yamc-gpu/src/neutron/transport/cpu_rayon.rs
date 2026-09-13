//! Rayon-parallel CPU equivalent of `run_multi_cell_transport`.
//!
//! Thin driver around [`super::core::transport_one_particle`].
//! Each particle's history is fully independent so the parallelism
//! is trivial; the only catch is per-thread tally accumulation
//! followed by a reduction step at the end.

use super::shared::{
    transport_one_particle, validate_transport_inputs, PendDrain, TransportInputs,
};
use super::{unpack_tally_outputs, MultiCellResult, NuclideSelectInputs};
use crate::common::geometry::bvh_cell_finding::build_and_flatten_bvh;
use crate::common::tallies::TalliesPack;

/// Rayon-parallel CPU equivalent of `run_multi_cell_transport`.
/// Each particle's history is fully independent so the parallelism
/// is trivial; the only catch is per-thread tally accumulation
/// followed by a reduction step at the end.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_cell_transport_cpu_rayon(
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
) -> MultiCellResult {
    use rayon::prelude::*;

    let n = seeds.len();
    let n_cells = cell_aabbs.len() / 6;
    let n_surfaces = surface_types.len();
    let n_grid = log_energy_grid.len();
    let n_materials = target_mass_per_material.len();

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

    let (bvh_aabbs, bvh_meta, bvh_prims, bvh_unb) = build_and_flatten_bvh(cell_aabbs);

    let inputs = TransportInputs {
        // The kernel's order; the CPU-twin-only `Lifo` is a verification
        // instrument reached through `run_multi_cell_transport_cpu` (issue #111).
        pend_drain: PendDrain::Fifo,
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

    /// Per-chunk outcome. `tally_acc` is allocated once per chunk
    /// (not per particle); chunks are sized so allocation is
    /// amortised across thousands of particles.
    struct ChunkResult {
        chunk_start: usize,
        alive: Vec<u32>,
        n_steps: Vec<u32>,
        final_energies: Vec<f64>,
        tally_acc: Vec<u64>,
        /// Lost-particle records from this chunk (issue #289), concatenated in
        /// chunk order below so the mirror's diagnostics are deterministic.
        lost: Vec<crate::common::lost_particles::LostParticleRecord>,
        /// (n,xn) secondaries this chunk queued past the kernel's in-thread
        /// stack depth (issue #111 phase 2), summed below.
        n_spilled: u64,
        /// Deepest pending-secondary stack seen in this chunk.
        max_pend_depth: u32,
        /// Histories per peak pending depth in this chunk, merged below.
        pend_depth_hist: Vec<u64>,
    }

    /// Per-thread chunk size. 4096 keeps the per-thread allocation
    /// footprint small (n_cells × n_energy_bins × 8 B × 2) while
    /// giving rayon enough work-units to balance across cores.
    const CHUNK: usize = 4096;

    let chunk_starts: Vec<usize> = (0..n).step_by(CHUNK).collect();
    let mut chunks: Vec<ChunkResult> = chunk_starts
        .into_par_iter()
        .map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK).min(n);
            let chunk_len = chunk_end - chunk_start;
            let mut alive_chunk: Vec<u32> = Vec::with_capacity(chunk_len);
            let mut n_steps_chunk: Vec<u32> = Vec::with_capacity(chunk_len);
            let mut energies_chunk: Vec<f64> = Vec::with_capacity(chunk_len);
            let mut tally_acc: Vec<u64> = vec![0u64; total_out_len];
            let mut lost_chunk: Vec<crate::common::lost_particles::LostParticleRecord> = Vec::new();
            let mut n_spilled_chunk = 0u64;
            let mut max_depth_chunk = 0u32;
            let mut hist_chunk: Vec<u64> = Vec::new();
            for i in chunk_start..chunk_end {
                let outcome = transport_one_particle(&inputs, i, &mut tally_acc, None);
                alive_chunk.push(outcome.alive);
                n_steps_chunk.push(outcome.n_steps);
                energies_chunk.push(outcome.final_energy);
                n_spilled_chunk += outcome.n_spilled as u64;
                max_depth_chunk = max_depth_chunk.max(outcome.max_pend_depth);
                let d = outcome.max_pend_depth as usize;
                if hist_chunk.len() <= d {
                    hist_chunk.resize(d + 1, 0);
                }
                hist_chunk[d] += 1;
                if let Some(record) = outcome.lost {
                    lost_chunk.push(record);
                }
            }
            ChunkResult {
                chunk_start,
                alive: alive_chunk,
                n_steps: n_steps_chunk,
                final_energies: energies_chunk,
                tally_acc,
                lost: lost_chunk,
                n_spilled: n_spilled_chunk,
                max_pend_depth: max_depth_chunk,
                pend_depth_hist: hist_chunk,
            }
        })
        .collect();

    // Sort chunks back into the original particle order before
    // concatenating. Tally summation is order-independent (integer
    // addition) but the per-particle Vecs are not.
    chunks.sort_by_key(|c| c.chunk_start);
    let mut alive_out: Vec<u32> = Vec::with_capacity(n);
    let mut n_steps_out: Vec<u32> = Vec::with_capacity(n);
    let mut final_energies: Vec<f64> = Vec::with_capacity(n);
    let mut total_tally_acc: Vec<u64> = vec![0u64; total_out_len];
    let mut lost = crate::common::lost_particles::LostParticleResult::default();
    let mut n_spilled_secondaries = 0u64;
    let mut max_pend_depth = 0u32;
    let mut pend_depth_hist: Vec<u64> = Vec::new();
    for c in chunks {
        n_spilled_secondaries += c.n_spilled;
        max_pend_depth = max_pend_depth.max(c.max_pend_depth);
        if pend_depth_hist.len() < c.pend_depth_hist.len() {
            pend_depth_hist.resize(c.pend_depth_hist.len(), 0);
        }
        for (slot, count) in pend_depth_hist.iter_mut().zip(c.pend_depth_hist.iter()) {
            *slot += count;
        }
        alive_out.extend(c.alive);
        n_steps_out.extend(c.n_steps);
        final_energies.extend(c.final_energies);
        for (slot, contrib) in total_tally_acc.iter_mut().zip(c.tally_acc.iter()) {
            *slot = slot.wrapping_add(*contrib);
        }
        lost.count += c.lost.len() as u64;
        for record in c.lost {
            if lost.records.len() < crate::common::lost_particles::LOST_RECORD_CAPACITY {
                lost.records.push(record);
            }
        }
    }
    let tally_outputs = unpack_tally_outputs(&total_tally_acc, tallies);

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
        // The CPU mirror does not emit coupled photons (S4b is GPU-only); the
        // bank is empty.
        photon_bank: super::PhotonBankResult::default(),
        lost,
        n_spilled_secondaries,
        max_pend_depth,
        pend_depth_hist,
    }
}
