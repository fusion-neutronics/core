//! Host-side adapter that drains a populated device particle bank into
//! the photon transport kernel's structure-of-arrays source.
//!
//! The coupled neutron->photon pipeline is: a neutron kernel emits
//! secondary photons into the device [`particle_bank`], then this drain
//! turns those banked records into the SoA source arrays the photon
//! transport kernel consumes. Banked coupled photons carry the EMITTING
//! NEUTRON's weight (the neutron kernel scales weight by fixed MT 16/17
//! (n,2n)/(n,3n) yields and by nu_bar for fission), so the drain
//! propagates that per-record weight straight into the photon source --
//! [`crate::photon::transport::run_multi_cell_photon_transport`] then
//! applies it at every tally score site.
//!
//! # Performance note
//!
//! This drain reads the bank back to the host, unpacks it into Vecs, and
//! the launcher re-uploads those Vecs to the device. That host round-trip
//! is acceptable for correctness (slice S5) but is an obvious target for a
//! later on-device drain (compact-in-place / indirect dispatch off the
//! bank counter) -- a performance concern deferred to a later slice.
//!
//! [`particle_bank`]: crate::common::particle_bank

use crate::common::particle_bank::{BANK_F64_STRIDE, BANK_U32_STRIDE, PTYPE_PHOTON};
use crate::common::tallies::TalliesPack;
use crate::photon::transport::{run_multi_cell_photon_transport, PhotonMultiCellResult};
use crate::GpuContext;

/// The photon kernel's structure-of-arrays source, unpacked from a bank.
/// Field lengths are all the photon record count `m`, except `positions`
/// and `directions` which are stride-3 (`3 * m`).
#[derive(Debug, Clone, Default)]
pub struct PhotonSource {
    /// Per-photon PCG seed (one u32 each), from the bank's `seed` field.
    pub seeds: Vec<u32>,
    /// Per-photon energy in eV (one f64 each).
    pub energies: Vec<f64>,
    /// Per-photon position, stride 3 (`[px, py, pz]` per photon).
    pub positions: Vec<f64>,
    /// Per-photon direction unit vector, stride 3 (`[dx, dy, dz]`).
    pub directions: Vec<f64>,
    /// Per-photon statistical weight (one f64 each), from the bank's
    /// `weight` field -- the emitting neutron's weight.
    pub weights: Vec<f64>,
    /// Per-photon D1S parent-nuclide id (one u32 each), from the bank's `gen`
    /// slot. `0` (no D1S parent) for prompt secondary photons. Fed to the
    /// photon kernel so `parent_nuclides` tally binning can attribute the
    /// photon (and every secondary it spawns) to its parent radionuclide.
    pub parent_nuclides: Vec<u32>,
    /// Per-photon originating source-particle index (issue #233 Stage 3), one
    /// u32 each, present only when the drain is given the bank's per-slot
    /// `bank_source_idx`. COMPACTED in lockstep with the `PTYPE_PHOTON` filter,
    /// so `source_indices[k]` is the source index of the `k`th drained photon
    /// (the same `k` the photon kernel reads as `source_idx[ABSOLUTE_POS]`). A
    /// raw per-slot slice would desync from this compacted SoA the moment the
    /// shared bank also holds a non-photon record (VR splitting / (n,2n)
    /// multiplication), silently misattributing per-source variance.
    pub source_indices: Vec<u32>,
}

/// Unpack the photon records of a populated bank into the photon kernel's
/// SoA source arrays.
///
/// For each slot `s` in `0..min(count, capacity)` (capacity =
/// `bank_f64.len() / BANK_F64_STRIDE`) whose `ptype == PTYPE_PHOTON`, emit
/// one source photon:
/// - `energy   = bank_f64[s*8]`
/// - `position = bank_f64[s*8+1 .. s*8+4]`
/// - `direction= bank_f64[s*8+4 .. s*8+7]`
/// - `weight   = bank_f64[s*8+7]`
/// - `seed     = bank_u32[s*4+2]`
///
/// The bank's `cell` field is intentionally ignored: the photon kernel
/// re-derives the starting cell from the position via BVH, exactly as for
/// a fresh source photon. Filtering on `PTYPE_PHOTON` keeps the drain
/// correct if the bank later also holds neutrons (variance-reduction
/// splitting / secondary-neutron multiplication).
///
/// `count` is clamped to `capacity` so an over-filled bank (overflow > 0,
/// which the caller must treat as a hard error before draining) never
/// reads past the slots that were actually written.
///
/// When `bank_source_idx` is `Some` (issue #233 Stage 3 per-source variance),
/// the originating source-particle index of each KEPT (photon) slot is pushed
/// into `PhotonSource::source_indices` in the same order, so it stays aligned
/// with the compacted SoA even when the filter drops non-photon slots. It is
/// indexed by raw slot `s` (same indexing as `bank_f64`/`bank_u32`), so the
/// caller passes the per-slot slice for the drained range.
pub fn drain_bank_to_photon_source(
    bank_f64: &[f64],
    bank_u32: &[u32],
    count: usize,
    bank_source_idx: Option<&[u32]>,
) -> PhotonSource {
    let capacity = bank_f64.len() / BANK_F64_STRIDE;
    debug_assert_eq!(
        bank_u32.len() / BANK_U32_STRIDE,
        capacity,
        "bank_f64 / bank_u32 capacities differ"
    );
    let n = count.min(capacity);

    let mut src = PhotonSource::default();
    for s in 0..n {
        if bank_u32[s * BANK_U32_STRIDE] != PTYPE_PHOTON {
            continue;
        }
        let f = s * BANK_F64_STRIDE;
        src.energies.push(bank_f64[f]);
        src.positions.push(bank_f64[f + 1]);
        src.positions.push(bank_f64[f + 2]);
        src.positions.push(bank_f64[f + 3]);
        src.directions.push(bank_f64[f + 4]);
        src.directions.push(bank_f64[f + 5]);
        src.directions.push(bank_f64[f + 6]);
        src.weights.push(bank_f64[f + 7]);
        src.seeds.push(bank_u32[s * BANK_U32_STRIDE + 2]);
        // gen slot (u32 idx 3) carries the D1S parent-nuclide id (0 = none).
        src.parent_nuclides.push(bank_u32[s * BANK_U32_STRIDE + 3]);
        if let Some(bsi) = bank_source_idx {
            debug_assert!(s < bsi.len(), "bank_source_idx shorter than drained count");
            src.source_indices.push(bsi[s]);
        }
    }
    src
}

/// Drain a populated bank's photon records and transport them with the
/// photon kernel -- the coupled-pipeline entry point.
///
/// This is a thin convenience over [`drain_bank_to_photon_source`] +
/// [`run_multi_cell_photon_transport`]: it unpacks the bank on the host
/// and feeds the resulting SoA source straight into the launcher (which
/// re-uploads it). The geometry / cross-section / TTB / Doppler / IFF /
/// atomic-relaxation / pair arguments are identical to
/// [`run_multi_cell_photon_transport`] and forwarded verbatim.
///
/// See the module-level note: the host round-trip is an S5 correctness
/// simplification; an on-device drain is a later performance concern.
#[allow(clippy::too_many_arguments)]
pub fn run_photon_transport_from_bank(
    ctx: &GpuContext,
    bank_f64: &[f64],
    bank_u32: &[u32],
    count: usize,
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
    // Per-collision element-selection inputs (task #72), forwarded verbatim.
    elem_macro_total: &[f64],
    mat_elem_meta: &[u32],
    tallies: &TalliesPack,
    max_steps: u32,
    // `Model::photon_cutoff_energy` in eV (issue #286), forwarded to the kernel
    // so drained secondaries obey the same cutoff as primaries.
    photon_cutoff_energy: f64,
    // Tally variance mode (issue #233 Stage 3). For the coupled `PerSource` case
    // the caller builds `source_idx` from the neutron kernel's `bank_source_idx`
    // (the drained secondaries' originating source-neutron indices), one entry
    // PER RAW BANK SLOT of the drained range. The drain filters non-photon slots,
    // so we compact that per-slot index in lockstep here and forward the
    // compacted slice (aligned with the drained SoA the kernel reads) rather than
    // the raw one, which would desync if the shared bank ever holds a non-photon.
    variance: crate::common::tallies::TallyVarianceMode,
) -> PhotonMultiCellResult {
    use crate::common::tallies::TallyVarianceMode;
    let raw_source_idx = match variance {
        TallyVarianceMode::PerSource { source_idx, .. }
        | TallyVarianceMode::PerSourceDirect { source_idx, .. } => source_idx,
        _ => None,
    };
    let src = drain_bank_to_photon_source(bank_f64, bank_u32, count, raw_source_idx);
    // Re-point `source_idx` at the drained-and-compacted indices. `PerSourceDirect`
    // (issue #234 mesh) folds per originating source neutron exactly like
    // `PerSource`; both remap in lockstep with the drained SoA.
    let variance = match variance {
        TallyVarianceMode::PerSource {
            chunk_sources,
            total_bins,
            source_idx: Some(_),
        } => TallyVarianceMode::PerSource {
            chunk_sources,
            total_bins,
            source_idx: Some(&src.source_indices),
        },
        TallyVarianceMode::PerSourceDirect {
            chunk_sources,
            total_bins,
            source_idx: Some(_),
        } => TallyVarianceMode::PerSourceDirect {
            chunk_sources,
            total_bins,
            source_idx: Some(&src.source_indices),
        },
        other => other,
    };
    run_multi_cell_photon_transport(
        ctx,
        &src.seeds,
        &src.energies,
        &src.weights,
        &src.parent_nuclides,
        &src.positions,
        &src.directions,
        cell_aabbs,
        cell_to_material,
        surface_types,
        surface_params,
        surface_boundaries,
        region_program,
        log_energy_grid,
        xs_total_per_material,
        xs_coherent_per_material,
        xs_incoherent_per_material,
        xs_photoelectric_per_material,
        xs_pair_per_material,
        heating_xs_per_material,
        rayleigh_x2,
        rayleigh_cdf,
        rayleigh_n_points,
        ttb_e_grid_log,
        ttb_electron_pdf,
        ttb_electron_cdf,
        ttb_electron_yield,
        ttb_has_data,
        dop_pz_grid,
        dop_electron_pdf,
        dop_binding_energy,
        dop_profile_pdf,
        dop_profile_cdf,
        dop_n_shells,
        dop_has_data,
        dop_subshell_idx,
        dop_subshell_w0,
        dop_subshell_cnt,
        iff_x,
        iff_s,
        iff_n_points,
        iff_has_data,
        ar_has_data,
        ar_n_shells,
        ar_binding_energy,
        ar_pe_subshell_xs_log,
        ar_n_trans,
        ar_trans_primary,
        ar_trans_secondary,
        ar_trans_energy,
        ar_trans_cum_prob,
        ttb_positron_pdf,
        ttb_positron_cdf,
        ttb_positron_yield,
        pair_has_data,
        pair_r_z,
        pair_a,
        pair_c,
        elem_macro_total,
        mat_elem_meta,
        tallies,
        max_steps,
        photon_cutoff_energy,
        variance,
    )
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::particle_bank::{PTYPE_NEUTRON, PTYPE_PHOTON};

    /// Hand-pack one bank record into the stride-packed arrays.
    #[allow(clippy::too_many_arguments)]
    fn pack_record(
        bank_f64: &mut [f64],
        bank_u32: &mut [u32],
        slot: usize,
        ptype: u32,
        energy: f64,
        pos: [f64; 3],
        dir: [f64; 3],
        weight: f64,
        seed: u32,
    ) {
        let f = slot * BANK_F64_STRIDE;
        bank_f64[f] = energy;
        bank_f64[f + 1] = pos[0];
        bank_f64[f + 2] = pos[1];
        bank_f64[f + 3] = pos[2];
        bank_f64[f + 4] = dir[0];
        bank_f64[f + 5] = dir[1];
        bank_f64[f + 6] = dir[2];
        bank_f64[f + 7] = weight;
        let u = slot * BANK_U32_STRIDE;
        bank_u32[u] = ptype;
        bank_u32[u + 1] = 99; // cell -- must be IGNORED by the drain
        bank_u32[u + 2] = seed;
        bank_u32[u + 3] = 0; // gen
    }

    /// A bank with a neutron record interleaved drains only the photon
    /// records (filter on `PTYPE_PHOTON`), preserving field values, and
    /// `count` clamps to capacity so an over-stated count never reads
    /// past the written slots.
    #[test]
    fn drain_filters_ptype_and_clamps_count() {
        let capacity = 4usize;
        let mut bf = vec![0.0_f64; capacity * BANK_F64_STRIDE];
        let mut bu = vec![0u32; capacity * BANK_U32_STRIDE];

        // slot 0: photon, slot 1: neutron (must be skipped), slot 2: photon
        pack_record(
            &mut bf,
            &mut bu,
            0,
            PTYPE_PHOTON,
            1.0e6,
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            2.0,
            111,
        );
        pack_record(
            &mut bf,
            &mut bu,
            1,
            PTYPE_NEUTRON,
            14.0e6,
            [9.0, 9.0, 9.0],
            [0.0, 1.0, 0.0],
            1.0,
            222,
        );
        pack_record(
            &mut bf,
            &mut bu,
            2,
            PTYPE_PHOTON,
            5.0e5,
            [-1.0, -2.0, -3.0],
            [0.0, 0.0, 1.0],
            0.5,
            333,
        );

        // count overstates capacity -- must clamp to 4 and never read slot 4+.
        let src = drain_bank_to_photon_source(&bf, &bu, 100, None);

        // Only the two photon records survive (neutron at slot 1 dropped).
        assert_eq!(src.seeds.len(), 2, "only photon records drained");
        assert_eq!(src.energies, vec![1.0e6, 5.0e5]);
        assert_eq!(src.weights, vec![2.0, 0.5]);
        assert_eq!(src.seeds, vec![111, 333]);
        assert_eq!(src.positions, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        assert_eq!(src.directions, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(src.positions.len(), 3 * 2);
        assert_eq!(src.directions.len(), 3 * 2);
        assert!(
            src.source_indices.is_empty(),
            "no source_idx requested -> none compacted"
        );

        // A count smaller than the photon population truncates the drain.
        let src2 = drain_bank_to_photon_source(&bf, &bu, 1, None);
        assert_eq!(src2.seeds, vec![111], "count=1 only reads slot 0");
    }

    /// With a non-photon slot interleaved, the per-slot `source_idx` is
    /// COMPACTED in lockstep with the `PTYPE_PHOTON` filter, so
    /// `source_indices[k]` is the source index of the `k`th drained photon (the
    /// raw slot's index), NOT of raw slot `k`. This is the Stage-3 latent-trap
    /// guard: a raw per-slot slice would attribute photon 1 (raw slot 2) to the
    /// dropped neutron slot's source.
    #[test]
    fn drain_compacts_source_idx_across_filtered_slots() {
        let capacity = 4usize;
        let mut bf = vec![0.0_f64; capacity * BANK_F64_STRIDE];
        let mut bu = vec![0u32; capacity * BANK_U32_STRIDE];
        // slot 0: photon (src 10), slot 1: neutron (src 11, dropped), slot 2:
        // photon (src 12).
        pack_record(
            &mut bf,
            &mut bu,
            0,
            PTYPE_PHOTON,
            1.0e6,
            [0.0; 3],
            [1.0, 0.0, 0.0],
            1.0,
            1,
        );
        pack_record(
            &mut bf,
            &mut bu,
            1,
            PTYPE_NEUTRON,
            14e6,
            [0.0; 3],
            [1.0, 0.0, 0.0],
            1.0,
            2,
        );
        pack_record(
            &mut bf,
            &mut bu,
            2,
            PTYPE_PHOTON,
            5.0e5,
            [0.0; 3],
            [1.0, 0.0, 0.0],
            1.0,
            3,
        );
        let bank_src = [10u32, 11, 12, 0];

        let src = drain_bank_to_photon_source(&bf, &bu, 3, Some(&bank_src));
        assert_eq!(src.seeds.len(), 2, "two photons drained");
        // The neutron's source index (11) must NOT appear; the two photons keep
        // their own raw-slot source indices 10 and 12.
        assert_eq!(
            src.source_indices,
            vec![10, 12],
            "source_idx compacted with the photon filter (neutron slot's 11 dropped)"
        );
    }
}
