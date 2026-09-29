//! Host-side packing for D1S (Direct-1-Step) decay-photon production.
//!
//! D1S replaces the prompt secondary-photon yields of a coupled run with
//! *decay* photons: at every neutron collision the kernel emits one photon
//! whose energy is drawn from the decay photon spectrum (lines or a continuum)
//! of an activation/transmutation product and whose weight is scaled by the
//! decay photon-production yield. Each emitted photon is tagged with the
//! *parent radionuclide* (the chain emitter), so a `parent_nuclides` tally
//! filter can bin the photon flux per radionuclide for the host-side
//! time-correction-factor (TCF) post-processing.
//!
//! # Material-aggregate layout (mirrors the prompt coupled path)
//!
//! The GPU neutron kernel works at the MATERIAL macroscopic level (its
//! `sigma_t` is the material macro total). The CPU's per-nuclide expected
//! per-collision photon yield, summed over the collision-nuclide sampling
//! probability, equals `macro_decay_photon_prod / sigma_t` -- so the kernel uses
//! one material-aggregate `y_t = photon_prod[mat, i_grid] / sigma_t`, emits a
//! SINGLE photon of weight `w * y_t` (implicit capture, matching the CPU
//! `sample_decay_photons` estimator), then selects a channel proportional to
//! `channel_xs[i_grid]` across ALL the material's channels. Each channel's
//! `xs` row is the macroscopic `N_n * micro_rxn_xs * yield_constant`, so the
//! aggregate `photon_prod[g] == sum over channels of ch_xs[ch_row + g]`. The
//! selected channel supplies the photon energy and the parent-nuclide id
//! stamped on the photon.
//!
//! # Channel spectra
//!
//! A channel holds either discrete lines or one continuum, and the high bits
//! of its packed count say which ([`DECAY_SPECTRUM_SHIFT`]):
//!
//! - lines: `count` energies, each beside its cumulative intensity
//!   (normalized to 1.0 at the last line);
//! - a histogram or linear-linear continuum of `count` points: a span of
//!   `2 * count` entries. The first `count` hold the energies beside the
//!   running integral at each point (0.0 at the first, 1.0 at the last); the
//!   next `count` hold the density divided by the integral, in the CDF array,
//!   beside 0.0 padding in the energy array. The kernel picks the interval by
//!   the running integral and inverts it exactly inside, as the CPU's
//!   `sample_continuum_energy` does, so a continuum's photons are drawn from
//!   its density instead of being dropped or read as lines.
//!
//! This reproduces the CPU joint (parent, energy) distribution in expectation;
//! it is statistically (not bit-) equal, the same acceptance class as the
//! prompt coupled path.
//!
//! # Packing
//!
//! Per-channel data is concatenated material-major and addressed by the per-
//! material base/count tables. To keep the kernel's storage-buffer descriptor
//! budget small (the neutron kernel already runs near the per-stage limit), the
//! per-material scalar metadata is packed into ONE `u32` meta buffer indexed
//! `[mat * DECAY_META_COLS + col]`, and the per-channel discrete-energy table
//! base/count is interleaved into the per-channel `u32` arrays.
//!
//! # Decay-off default
//!
//! [`DecayPhotonInputs::decay_off`] builds a minimal size-1 dummy table with the
//! gate flag `0`. The kernel's decay-emission block is gated on
//! `decay_enabled[0] == 1`, so a decay-off launch draws ZERO decay RNG and
//! writes ZERO decay bank records -- byte-identical to a run without D1S.

/// Columns in the packed per-material `u32` metadata buffer.
/// `[pp_base, ch_base, ch_count]`.
pub const DECAY_META_COLS: usize = 3;

/// A channel of discrete lines.
pub const DECAY_SPECTRUM_LINES: u32 = 0;
/// A channel holding a histogram (ENDF INT=1) continuum.
pub const DECAY_SPECTRUM_HISTOGRAM: u32 = 1;
/// A channel holding a linear-linear (ENDF INT=2) continuum.
pub const DECAY_SPECTRUM_LINEAR_LINEAR: u32 = 2;
/// The spectrum kind sits in the packed count's bits from here up; the bits
/// below are the count.
pub const DECAY_SPECTRUM_SHIFT: u32 = 30;
/// The count bits of a packed count.
pub const DECAY_COUNT_MASK: u32 = (1u32 << DECAY_SPECTRUM_SHIFT) - 1;

/// Per-material decay-photon table for ONE material, built on the host from the
/// per-nuclide `DecayPhotonNuclideData` aggregated by atom density. Concatenated
/// into [`DecayPhotonInputs`] by [`DecayPhotonInputs::from_materials`].
#[derive(Debug, Clone, Default)]
pub struct MaterialDecayTable {
    /// Aggregate decay photon-production macro XS on the shared grid (length
    /// `n_grid`). `photon_prod[g] == sum over this material's channels of
    /// `ch_xs[ch][g]`.
    pub photon_prod: Vec<f64>,
    /// Per-channel macroscopic weighted reaction XS rows (`n_grid` each),
    /// concatenated channel-major. Channel `c` occupies `ch_xs[c*n_grid ..
    /// (c+1)*n_grid]`.
    pub ch_xs: Vec<f64>,
    /// Per-channel parent-nuclide id (the chain emitter's `NuclideId.get()`,
    /// widened to u32). One per channel.
    pub ch_parent_id: Vec<u32>,
    /// Per-channel spectrum energies [eV], concatenated channel-major, laid
    /// out as the module docs describe. Channel `c` starts at `ch_e_base[c]`.
    pub ch_energies: Vec<f64>,
    /// Per-channel cumulative intensity CDF, and for a continuum its
    /// normalized densities, concatenated channel-major, parallel to
    /// `ch_energies`.
    pub ch_intensity_cdf: Vec<f64>,
    /// Per-channel base offset into `ch_energies` / `ch_intensity_cdf`.
    pub ch_e_base: Vec<u32>,
    /// Per-channel line count, or a continuum's point count.
    pub ch_e_count: Vec<u32>,
    /// Per-channel spectrum kind, one of the `DECAY_SPECTRUM_*` constants.
    pub ch_e_kind: Vec<u32>,
}

/// The entries a channel occupies in `ch_energies` / `ch_intensity_cdf`.
pub fn decay_spectrum_span(kind: u32, count: u32) -> usize {
    if kind == DECAY_SPECTRUM_LINES {
        count as usize
    } else {
        2 * count as usize
    }
}

impl MaterialDecayTable {
    /// Number of channels in this material's table.
    pub fn n_channels(&self) -> usize {
        self.ch_parent_id.len()
    }

    /// Build a synthetic non-emitting "void material" decay table for a
    /// material-less (void) cell on the D1S decay-photon path.
    ///
    /// It carries a single zero aggregate `photon_prod` row of length `n_grid`
    /// and zero channels, mirroring the prompt-coupled
    /// `GpuPhotonProductionXs::void`. A void cell maps to this slot via
    /// `cell_to_material`; because the neutron kernel's decay-emission block is
    /// gated behind a collision (never reached in a `sigma_t = 0` void cell),
    /// the slot is never sampled -- it exists only to keep the per-material
    /// offset tables (`meta`) in-bounds for the void index.
    pub fn void(n_grid: usize) -> Self {
        MaterialDecayTable {
            photon_prod: vec![0.0; n_grid],
            ..Default::default()
        }
    }
}

/// Flat, material-major D1S decay-photon buffers + the packed per-material
/// metadata, ready to upload to the neutron transport kernel's decay-emission
/// site. See the module docs for the concatenation scheme.
#[derive(Debug, Clone)]
pub struct DecayPhotonInputs {
    /// Runtime gate (1 element). `1` enables decay-photon emission, `0`
    /// disables it (the kernel then does no decay RNG and no decay bank writes).
    pub decay_enabled: Vec<u32>,
    /// Aggregate decay photon-production macro XS, concatenated material-major;
    /// material `m`'s `n_grid` row starts at meta col `pp_base`.
    pub photon_prod: Vec<f64>,
    /// Per-channel weighted reaction XS rows (`n_grid` each), concatenated
    /// material-major then channel-major within a material.
    pub ch_xs: Vec<f64>,
    /// Per-channel parent-nuclide id (widened `NuclideId.get()`), concatenated.
    pub ch_parent_id: Vec<u32>,
    /// GLOBAL per-channel `[base, count | kind << DECAY_SPECTRUM_SHIFT]` into
    /// `ch_energies` / `ch_intensity_cdf`, interleaved (stride 2) and
    /// concatenated. Channel `c` (global) reads `ch_e_meta[c*2]` /
    /// `ch_e_meta[c*2 + 1]`.
    pub ch_e_meta: Vec<u32>,
    /// Per-channel spectrum energies [eV], concatenated.
    pub ch_energies: Vec<f64>,
    /// Per-channel cumulative intensity CDF and continuum densities,
    /// concatenated, parallel to `ch_energies`.
    pub ch_intensity_cdf: Vec<f64>,
    /// Packed per-material metadata, `[mat * DECAY_META_COLS + col]`:
    /// col 0 = `pp_base` (aggregate-row offset into `photon_prod`, element
    /// units), col 1 = `ch_base` (GLOBAL channel index of the material's first
    /// channel), col 2 = `ch_count` (number of channels).
    pub meta: Vec<u32>,
}

impl DecayPhotonInputs {
    /// Decay-OFF default: a single size-1 dummy table with the gate flag `0`.
    /// The kernel reads none of it (the emission block is gated on
    /// `decay_enabled[0] == 1`); it exists only so the buffers are non-empty
    /// (cubecl rejects zero-length bindings). Every material maps to the same
    /// dummy sub-table at base 0 with zero channels.
    pub fn decay_off(n_materials: usize, n_grid: usize) -> Self {
        let n_mat = n_materials.max(1);
        DecayPhotonInputs {
            decay_enabled: vec![0u32],
            photon_prod: vec![0.0; n_grid.max(1)],
            ch_xs: vec![0.0; n_grid.max(1)],
            ch_parent_id: vec![0u32],
            ch_e_meta: vec![0u32, 0u32],
            ch_energies: vec![0.0],
            ch_intensity_cdf: vec![0.0],
            // pp_base / ch_base / ch_count all 0 for every material: the gate is
            // off so the kernel never reads them.
            meta: vec![0u32; n_mat * DECAY_META_COLS],
        }
    }

    /// Concatenate per-material [`MaterialDecayTable`]s into the flat kernel
    /// buffers and set the gate flag to `1` (decay ON). Every table must share
    /// the same `n_grid` (the shared log-energy grid the kernel uses). The
    /// per-channel `[base, count | kind]` spectrum metadata is rewritten GLOBAL
    /// as each channel's spectrum is concatenated.
    pub fn from_materials(tables: &[MaterialDecayTable], n_grid: usize) -> Self {
        let n_mat = tables.len().max(1);
        let mut out = DecayPhotonInputs {
            decay_enabled: vec![1u32],
            photon_prod: Vec::new(),
            ch_xs: Vec::new(),
            ch_parent_id: Vec::new(),
            ch_e_meta: Vec::new(),
            ch_energies: Vec::new(),
            ch_intensity_cdf: Vec::new(),
            meta: vec![0u32; n_mat * DECAY_META_COLS],
        };

        for (m, t) in tables.iter().enumerate() {
            // The aggregate photon_prod row for this material must be exactly
            // n_grid (decay_off uses n_grid; from_materials must match).
            debug_assert!(
                t.photon_prod.len() == n_grid || t.n_channels() == 0,
                "material decay photon_prod row length must equal n_grid"
            );

            let pp_base = out.photon_prod.len() as u32;
            let ch_base = out.ch_parent_id.len() as u32;
            let ch_count = t.n_channels() as u32;

            out.meta[m * DECAY_META_COLS] = pp_base;
            out.meta[m * DECAY_META_COLS + 1] = ch_base;
            out.meta[m * DECAY_META_COLS + 2] = ch_count;

            if t.photon_prod.is_empty() {
                out.photon_prod.extend(std::iter::repeat_n(0.0, n_grid));
            } else {
                out.photon_prod.extend_from_slice(&t.photon_prod);
            }
            out.ch_xs.extend_from_slice(&t.ch_xs);
            out.ch_parent_id.extend_from_slice(&t.ch_parent_id);

            for c in 0..t.n_channels() {
                let global_base = out.ch_energies.len() as u32;
                let count = t.ch_e_count[c];
                let kind = t.ch_e_kind[c];
                assert!(
                    count <= DECAY_COUNT_MASK,
                    "decay photon channel with {count} points overflows the packed count"
                );
                out.ch_e_meta.push(global_base);
                out.ch_e_meta.push(count | (kind << DECAY_SPECTRUM_SHIFT));
                let local_base = t.ch_e_base[c] as usize;
                let local_end = local_base + decay_spectrum_span(kind, count);
                out.ch_energies
                    .extend_from_slice(&t.ch_energies[local_base..local_end]);
                out.ch_intensity_cdf
                    .extend_from_slice(&t.ch_intensity_cdf[local_base..local_end]);
            }
        }

        // cubecl rejects zero-length bindings: pad empty arrays to size 1.
        if out.photon_prod.is_empty() {
            out.photon_prod.push(0.0);
        }
        if out.ch_xs.is_empty() {
            out.ch_xs.push(0.0);
        }
        if out.ch_parent_id.is_empty() {
            out.ch_parent_id.push(0u32);
        }
        if out.ch_e_meta.is_empty() {
            out.ch_e_meta.push(0u32);
            out.ch_e_meta.push(0u32);
        }
        if out.ch_energies.is_empty() {
            out.ch_energies.push(0.0);
        }
        if out.ch_intensity_cdf.is_empty() {
            out.ch_intensity_cdf.push(0.0);
        }

        out
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_off_has_gate_zero_and_nonempty_buffers() {
        let d = DecayPhotonInputs::decay_off(3, 5);
        assert_eq!(d.decay_enabled, vec![0]);
        assert_eq!(d.photon_prod.len(), 5);
        assert_eq!(d.meta.len(), 3 * DECAY_META_COLS);
        assert!(d.meta.iter().all(|&x| x == 0));
        assert!(!d.ch_parent_id.is_empty());
        assert!(!d.ch_energies.is_empty());
    }

    #[test]
    fn from_materials_concatenates_and_rewrites_global_bases() {
        let n_grid = 4;
        // Material 0: one channel, 2 lines.
        let mat0 = MaterialDecayTable {
            photon_prod: vec![1.0, 2.0, 3.0, 4.0],
            ch_xs: vec![1.0, 2.0, 3.0, 4.0],
            ch_parent_id: vec![7],
            ch_energies: vec![100.0, 200.0],
            ch_intensity_cdf: vec![0.4, 1.0],
            ch_e_base: vec![0],
            ch_e_count: vec![2],
            ch_e_kind: vec![DECAY_SPECTRUM_LINES],
        };
        // Material 1: two channels, 1 + 3 lines.
        let mat1 = MaterialDecayTable {
            photon_prod: vec![5.0, 6.0, 7.0, 8.0],
            ch_xs: vec![2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0],
            ch_parent_id: vec![11, 13],
            ch_energies: vec![300.0, 400.0, 500.0, 600.0],
            ch_intensity_cdf: vec![1.0, 0.3, 0.6, 1.0],
            ch_e_base: vec![0, 1],
            ch_e_count: vec![1, 3],
            ch_e_kind: vec![DECAY_SPECTRUM_LINES; 2],
        };
        let d = DecayPhotonInputs::from_materials(&[mat0, mat1], n_grid);

        assert_eq!(d.decay_enabled, vec![1]);
        // Material 0 meta: pp_base 0, ch_base 0, ch_count 1.
        assert_eq!(&d.meta[0..3], &[0, 0, 1]);
        // Material 1 meta: pp_base 4, ch_base 1, ch_count 2.
        assert_eq!(&d.meta[3..6], &[4, 1, 2]);

        assert_eq!(d.photon_prod.len(), 8);
        assert_eq!(d.ch_parent_id, vec![7, 11, 13]);

        // ch_e_meta interleaved [base, count] per GLOBAL channel:
        // ch0: base 0 count 2; ch1: base 2 count 1; ch2: base 3 count 3.
        assert_eq!(d.ch_e_meta, vec![0, 2, 2, 1, 3, 3]);
        assert_eq!(
            d.ch_energies,
            vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        );
        assert_eq!(d.ch_intensity_cdf, vec![0.4, 1.0, 1.0, 0.3, 0.6, 1.0]);
    }

    /// A continuum channel carries twice its point count, and its kind rides
    /// in the packed count so the kernel can tell it from lines.
    #[test]
    fn a_continuum_channel_packs_its_kind_and_both_halves() {
        let table = MaterialDecayTable {
            photon_prod: vec![2.0],
            ch_xs: vec![1.0, 1.0],
            ch_parent_id: vec![3, 5],
            ch_energies: vec![100.0, 1.0e3, 2.0e3, 3.0e3, 0.0, 0.0, 0.0],
            ch_intensity_cdf: vec![1.0, 0.0, 0.4, 1.0, 4.0e-4, 4.0e-4, 8.0e-4],
            ch_e_base: vec![0, 1],
            ch_e_count: vec![1, 3],
            ch_e_kind: vec![DECAY_SPECTRUM_LINES, DECAY_SPECTRUM_LINEAR_LINEAR],
        };
        let d = DecayPhotonInputs::from_materials(&[table], 1);
        assert_eq!(
            d.ch_e_meta,
            vec![
                0,
                1,
                1,
                3 | (DECAY_SPECTRUM_LINEAR_LINEAR << DECAY_SPECTRUM_SHIFT)
            ]
        );
        assert_eq!(d.ch_e_meta[3] & DECAY_COUNT_MASK, 3);
        assert_eq!(d.ch_energies.len(), 7);
        assert_eq!(d.ch_intensity_cdf[4..], [4.0e-4, 4.0e-4, 8.0e-4]);
    }
}
