//! Per-collision nuclide-selection inputs for the neutron transport kernel
//! (issue #74, Stage 1).
//!
//! At a collision in a multi-nuclide material the transport must pick *which*
//! nuclide is struck, proportional to that nuclide's macroscopic total cross
//! section at the collision energy, and then use THAT nuclide's atomic-weight
//! ratio (AWR) for the elastic kinematics. Without this the kernel uses a single
//! density-weighted *average* target mass per material, which under-moderates:
//! an H2O sphere recovers only ~0.45x the CPU thermal flux because hydrogen's
//! large per-collision energy loss is averaged away against oxygen.
//!
//! This struct bundles the three buffers the kernel needs into a single launch
//! parameter (the [`super::CoupledPhotonInputs`] idiom), so adding the feature
//! costs one public API argument and a fixed handful of kernel storage-buffer
//! bindings:
//!
//! - [`Self::nuc_macro_total`] -- per-(material, nuclide) macroscopic total xs
//!   on the shared grid, flat `[n_slab x n_grid]` where
//!   `n_slab = sum_mat n_nuclides(mat)`. This is the per-collision selection
//!   weight table (cumulative-sum walk against the collision-energy interpolant)
//!   and is built by [`crate::extract_per_nuclide_macro_total_xs`] per material,
//!   concatenated material-major.
//! - [`Self::nuc_awr`] -- per-(material, nuclide) AWR, flat `[n_slab]`.
//! - [`Self::mat_nuclide_meta`] -- stride-2 `[offset, count]` per material:
//!   `mat_nuclide_meta[mat*2]` is the slab base of the material's first nuclide
//!   row, `mat_nuclide_meta[mat*2 + 1]` is its nuclide count.
//!
//! # Single-nuclide bit-identity
//!
//! When a material has exactly one nuclide (`count == 1`) the kernel and the CPU
//! twin SKIP the selection draw entirely -- the nuclide is trivially the only
//! one, so no extra random is consumed and the RNG stream is byte-identical to
//! pre-Stage-1 behaviour. [`Self::single_nuclide`] builds the degenerate
//! one-nuclide-per-material layout used by every fixture / caller that does not
//! yet thread real per-nuclide data; with it, Stage 1 is a no-op and the kernel
//! reproduces today's results bit-for-bit.

/// Number of per-(slab, energy) reaction-partial columns packed into
/// [`NuclideSelectInputs::nuc_partial_xs`]: elastic / absorption / inelastic /
/// fission, in that column order (issue #74, Stage 2b). The kernel reads them
/// after selecting the struck nuclide to split the reaction type from THAT
/// nuclide's own partials, mirroring CPU `Nuclide::sample_reaction_type`.
pub const NUC_PARTIAL_COLS: usize = 4;
pub const NUC_PARTIAL_ELASTIC: usize = 0;
pub const NUC_PARTIAL_ABSORPTION: usize = 1;
pub const NUC_PARTIAL_INELASTIC: usize = 2;
pub const NUC_PARTIAL_FISSION: usize = 3;

/// Columns of [`NuclideSelectInputs::chi_slab_meta`], the per-slab fission chi
/// row table (fusion-neutronics/core#34 entry 1). For slab `s` the row
/// `chi_slab_meta[s * CHI_SLAB_META_COLS ..]` holds: the first of its prompt chi
/// rows, its fission channel count, its delayed chi row, and its element base
/// into [`NuclideSelectInputs::fission_channel_xs`].
pub const CHI_SLAB_META_COLS: usize = 4;
pub const CHI_SLAB_PROMPT_ROW: usize = 0;
pub const CHI_SLAB_N_CHANNELS: usize = 1;
pub const CHI_SLAB_DELAYED_ROW: usize = 2;
pub const CHI_SLAB_CHANNEL_XS_BASE: usize = 3;

/// Packed per-collision nuclide-selection inputs (see module docs). Built once
/// per launch and passed by reference into the host launcher and the CPU twin.
#[derive(Debug, Clone, Default)]
pub struct NuclideSelectInputs {
    /// Per-(material, nuclide) macroscopic total xs on the shared grid, flat
    /// `[n_slab x n_grid]`, nuclide-major within a material and concatenated
    /// material-major. Row `slab` starts at `slab * n_grid`.
    pub nuc_macro_total: Vec<f64>,
    /// Per-(material, nuclide) AWR (neutron-mass units), flat `[n_slab]`.
    pub nuc_awr: Vec<f64>,
    /// Stride-2 `[offset, count]` per material: slab base + nuclide count.
    pub mat_nuclide_meta: Vec<u32>,
    /// Per-(slab, energy) reaction partials, DENSITY-WEIGHTED (macroscopic),
    /// packed `[n_slab x n_grid x NUC_PARTIAL_COLS]`: the entry for slab `s`,
    /// grid point `i`, column `c` is `nuc_partial_xs[(s * n_grid + i) *
    /// NUC_PARTIAL_COLS + c]`. Columns are elastic / absorption / inelastic /
    /// fission. After selecting nuclide `s`, the kernel splits the reaction
    /// type four-way from these (mirroring CPU `Nuclide::sample_reaction_type`,
    /// whose scattering = elastic + inelastic). Packing the four partials into
    /// one buffer adds a single storage-buffer binding (#74 Stage 2b).
    pub nuc_partial_xs: Vec<f64>,
    /// Per-slab fission chi row table, packed `[n_slab x CHI_SLAB_META_COLS]`
    /// (fusion-neutronics/core#34 entry 1). The `fission_eout_*` chi buffers are
    /// indexed by ROW, and this table says which rows belong to which struck
    /// nuclide: `CHI_SLAB_PROMPT_ROW` is the first of the slab's prompt rows, one
    /// per fission channel (the non-redundant MTs among 18 / 19 / 20 / 21 / 38,
    /// in the CPU fast grid's `fission_mt_numbers` order); `CHI_SLAB_N_CHANNELS`
    /// is how many; `CHI_SLAB_DELAYED_ROW` is the slab's delayed-spectrum row;
    /// `CHI_SLAB_CHANNEL_XS_BASE` is its element base into `fission_channel_xs`.
    /// A fission in a slab with more than one channel draws one uniform and walks
    /// the channels' cross sections at the collision energy to pick the prompt
    /// row, mirroring CPU `FastXSGrid::sample_fission_reaction`; a single-channel
    /// slab takes no draw and uses the base row, so every evaluation with MT 18
    /// alone keeps its RNG stream byte for byte.
    pub chi_slab_meta: Vec<u32>,
    /// Per-channel microscopic fission cross sections for the slabs that carry
    /// more than one channel, tight CSR on the owning material's FINE grid: the
    /// value for slab `s`, channel `c`, fine index `i` sits at element
    /// `base_s + c * fine_n + i`. Single-channel slabs contribute nothing; one
    /// zero pads an all-single model, since cubecl rejects an empty buffer.
    pub fission_channel_xs: Vec<f64>,
}

/// The two-rows-per-slab chi table: slab `s` reads its prompt spectrum at row
/// `2s` and its delayed spectrum at row `2s + 1`, one channel each. This is the
/// layout every fixture that hands the chi buffers over per material uses
/// (slab == material there), and the layout the translator starts from before
/// [`NuclideSelectInputs::set_fission_chi`] replaces it with the real one.
fn two_rows_per_slab(n_slab: usize) -> Vec<u32> {
    let mut meta = Vec::with_capacity(n_slab.max(1) * CHI_SLAB_META_COLS);
    for s in 0..n_slab.max(1) {
        meta.push(2 * s as u32);
        meta.push(1);
        meta.push(2 * s as u32 + 1);
        meta.push(0);
    }
    meta
}

impl NuclideSelectInputs {
    /// Degenerate single-nuclide-per-material layout: one slab row per material,
    /// `count == 1`, so the kernel / CPU twin never draws the selection random
    /// and stays byte-identical to pre-Stage-1 behaviour. `target_mass_per_material`
    /// supplies each material's (single) AWR; the macro-total row is a single
    /// `n_grid`-wide block of zeros (never read when `count == 1`).
    pub fn single_nuclide(target_mass_per_material: &[f64], n_grid: usize) -> Self {
        let n_mat = target_mass_per_material.len();
        let mut mat_nuclide_meta = Vec::with_capacity(n_mat * 2);
        for slab in 0..n_mat {
            mat_nuclide_meta.push(slab as u32); // offset = one row per material
            mat_nuclide_meta.push(1u32); // count = 1
        }
        Self {
            nuc_macro_total: vec![0.0; n_mat.max(1) * n_grid.max(1)],
            nuc_awr: target_mass_per_material.to_vec(),
            mat_nuclide_meta,
            // Single-nuclide materials never trigger the per-nuclide reaction
            // split (the kernel uses the material-aggregate partials when
            // count == 1), so these rows are never read. Sized to one row per
            // material to keep the host length assertion happy.
            nuc_partial_xs: vec![0.0; n_mat.max(1) * n_grid.max(1) * NUC_PARTIAL_COLS],
            chi_slab_meta: two_rows_per_slab(n_mat),
            fission_channel_xs: vec![0.0],
        }
    }

    /// Replace the chi row table and the per-channel fission cross sections
    /// (fusion-neutronics/core#34 entry 1) once the caller has laid the chi rows
    /// out per (slab, channel). `chi_slab_meta` must hold one
    /// `CHI_SLAB_META_COLS` row per slab; an empty `fission_channel_xs` is padded
    /// to one element.
    pub fn set_fission_chi(&mut self, chi_slab_meta: Vec<u32>, fission_channel_xs: Vec<f64>) {
        assert_eq!(
            chi_slab_meta.len(),
            self.nuc_awr.len() * CHI_SLAB_META_COLS,
            "chi_slab_meta must be [n_slab x CHI_SLAB_META_COLS]"
        );
        self.chi_slab_meta = chi_slab_meta;
        self.fission_channel_xs = if fission_channel_xs.is_empty() {
            vec![0.0]
        } else {
            fission_channel_xs
        };
    }

    /// Build from per-material `(macro_total_rows, awr)` where `macro_total_rows`
    /// is that material's `[n_nuclides x n_grid]` flat block (nuclide-major, as
    /// produced by [`crate::extract_per_nuclide_macro_total_xs`]) and `awr` is its
    /// per-nuclide AWR. Materials are concatenated in the given order, building
    /// the slab offset table as it goes.
    ///
    /// Every material must have at least one nuclide. `n_grid` is the shared grid
    /// length (columns per row).
    pub fn from_materials(materials: &[(Vec<f64>, Vec<f64>)], n_grid: usize) -> Self {
        Self::from_materials_with_partials(
            &materials
                .iter()
                .map(|(rows, awr)| (rows.clone(), awr.clone(), Vec::new()))
                .collect::<Vec<_>>(),
            n_grid,
        )
    }

    /// Like [`Self::from_materials`] but each material also supplies its
    /// per-(nuclide, energy) reaction partials packed `[n_nuclides x n_grid x
    /// NUC_PARTIAL_COLS]` (column order elastic / absorption / inelastic /
    /// fission, density-weighted), as produced by
    /// [`crate::extract_per_nuclide_inelastic`] and packed by the translator.
    /// An empty `partials` block falls back to a zero block of the right size
    /// (single-nuclide materials never read it).
    pub fn from_materials_with_partials(
        materials: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
        n_grid: usize,
    ) -> Self {
        let mut nuc_macro_total = Vec::new();
        let mut nuc_awr = Vec::new();
        let mut nuc_partial_xs = Vec::new();
        let mut mat_nuclide_meta = Vec::with_capacity(materials.len() * 2);
        let mut slab_base: u32 = 0;
        for (rows, awr, partials) in materials {
            let count = awr.len();
            debug_assert_eq!(
                rows.len(),
                count * n_grid,
                "macro-total block must be [n_nuclides x n_grid]"
            );
            mat_nuclide_meta.push(slab_base);
            mat_nuclide_meta.push(count as u32);
            nuc_macro_total.extend_from_slice(rows);
            nuc_awr.extend_from_slice(awr);
            if partials.is_empty() {
                nuc_partial_xs.extend(std::iter::repeat_n(0.0, count * n_grid * NUC_PARTIAL_COLS));
            } else {
                debug_assert_eq!(
                    partials.len(),
                    count * n_grid * NUC_PARTIAL_COLS,
                    "partials block must be [n_nuclides x n_grid x NUC_PARTIAL_COLS]"
                );
                nuc_partial_xs.extend_from_slice(partials);
            }
            slab_base += count as u32;
        }
        // cubecl rejects zero-length buffers; pad an empty model to one slot.
        if nuc_macro_total.is_empty() {
            nuc_macro_total = vec![0.0; n_grid.max(1)];
        }
        if nuc_awr.is_empty() {
            nuc_awr = vec![1.0];
        }
        if mat_nuclide_meta.is_empty() {
            mat_nuclide_meta = vec![0, 1];
        }
        if nuc_partial_xs.is_empty() {
            nuc_partial_xs = vec![0.0; n_grid.max(1) * NUC_PARTIAL_COLS];
        }
        let chi_slab_meta = two_rows_per_slab(nuc_awr.len());
        Self {
            nuc_macro_total,
            nuc_awr,
            mat_nuclide_meta,
            nuc_partial_xs,
            chi_slab_meta,
            fission_channel_xs: vec![0.0],
        }
    }

    /// Like [`Self::from_materials_with_partials`] but each material may ride its
    /// OWN fine grid (issue #212): the row/partial blocks are concatenated tight
    /// without assuming a uniform `n_grid`, so material `m`'s block is
    /// `nuc_count[m] × fine_n[m]` (and `× NUC_PARTIAL_COLS` for the partials).
    /// Each material's `fine_n` is derived from its own row block
    /// (`rows.len() / count`); the caller records the per-material element bases
    /// (`nuc_macro_total` base) in `fine_meta` separately. Materials are
    /// concatenated in the given order, building the slab offset table as it
    /// goes; every material must have at least one nuclide.
    pub fn from_materials_with_partials_per_material(
        materials: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
    ) -> Self {
        let mut nuc_macro_total = Vec::new();
        let mut nuc_awr = Vec::new();
        let mut nuc_partial_xs = Vec::new();
        let mut mat_nuclide_meta = Vec::with_capacity(materials.len() * 2);
        let mut slab_base: u32 = 0;
        for (rows, awr, partials) in materials {
            let count = awr.len();
            debug_assert!(count > 0, "every material must have at least one nuclide");
            debug_assert_eq!(
                rows.len() % count,
                0,
                "macro-total block must be [n_nuclides x fine_n]"
            );
            let fine_n = rows.len() / count;
            mat_nuclide_meta.push(slab_base);
            mat_nuclide_meta.push(count as u32);
            nuc_macro_total.extend_from_slice(rows);
            nuc_awr.extend_from_slice(awr);
            if partials.is_empty() {
                nuc_partial_xs.extend(std::iter::repeat_n(0.0, count * fine_n * NUC_PARTIAL_COLS));
            } else {
                debug_assert_eq!(
                    partials.len(),
                    count * fine_n * NUC_PARTIAL_COLS,
                    "partials block must be [n_nuclides x fine_n x NUC_PARTIAL_COLS]"
                );
                nuc_partial_xs.extend_from_slice(partials);
            }
            slab_base += count as u32;
        }
        // cubecl rejects zero-length buffers; pad an empty model to one slot.
        if nuc_macro_total.is_empty() {
            nuc_macro_total = vec![0.0; 1];
        }
        if nuc_awr.is_empty() {
            nuc_awr = vec![1.0];
        }
        if mat_nuclide_meta.is_empty() {
            mat_nuclide_meta = vec![0, 1];
        }
        if nuc_partial_xs.is_empty() {
            nuc_partial_xs = vec![0.0; NUC_PARTIAL_COLS];
        }
        let chi_slab_meta = two_rows_per_slab(nuc_awr.len());
        Self {
            nuc_macro_total,
            nuc_awr,
            mat_nuclide_meta,
            nuc_partial_xs,
            chi_slab_meta,
            fission_channel_xs: vec![0.0],
        }
    }

    /// Append one more single-nuclide material (count 1, so it never triggers a
    /// selection draw). Used for the synthetic void material slot appended after
    /// the real materials. `n_grid` must match the existing rows. `awr` is the
    /// (never-read) target mass for the slot; `prompt_row` and `delayed_row` are
    /// the slot's (never-read) chi rows, which the caller has already pushed onto
    /// the chi buffers so the row table stays aligned with them.
    pub fn push_single_nuclide_material(
        &mut self,
        awr: f64,
        n_grid: usize,
        prompt_row: u32,
        delayed_row: u32,
    ) {
        let slab_base = self.nuc_awr.len() as u32;
        self.mat_nuclide_meta.push(slab_base);
        self.mat_nuclide_meta.push(1);
        self.chi_slab_meta
            .extend_from_slice(&[prompt_row, 1, delayed_row, 0]);
        self.nuc_macro_total
            .extend(std::iter::repeat_n(0.0, n_grid));
        self.nuc_awr.push(awr);
        // Single-nuclide => the per-nuclide partial row is never read; append a
        // zero block to keep `nuc_partial_xs` length == n_slab x n_grid x cols.
        self.nuc_partial_xs
            .extend(std::iter::repeat_n(0.0, n_grid * NUC_PARTIAL_COLS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture layout: every slab reads its prompt spectrum at row `2s` and
    /// its delayed spectrum at row `2s + 1`, one channel each, so no channel
    /// draw is ever taken.
    #[test]
    fn builders_default_to_two_rows_per_slab() {
        let inputs = NuclideSelectInputs::from_materials(
            &[(vec![0.0; 6], vec![1.0, 2.0]), (vec![0.0; 3], vec![3.0])],
            3,
        );
        assert_eq!(inputs.nuc_awr.len(), 3);
        assert_eq!(
            inputs.chi_slab_meta,
            vec![0, 1, 1, 0, 2, 1, 3, 0, 4, 1, 5, 0]
        );
        assert_eq!(inputs.fission_channel_xs, vec![0.0]);
        let single = NuclideSelectInputs::single_nuclide(&[1.0, 1.0], 4);
        assert_eq!(single.chi_slab_meta, vec![0, 1, 1, 0, 2, 1, 3, 0]);
    }

    /// The translator replaces the table wholesale once the rows are laid out
    /// per (slab, channel); the void slot appended afterwards carries its own
    /// two rows.
    #[test]
    fn set_fission_chi_replaces_the_table_and_the_void_slot_appends() {
        let mut inputs = NuclideSelectInputs::from_materials(&[(vec![0.0; 4], vec![1.0])], 4);
        inputs.set_fission_chi(vec![0, 4, 4, 0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(inputs.chi_slab_meta, vec![0, 4, 4, 0]);
        assert_eq!(inputs.fission_channel_xs.len(), 4);
        inputs.push_single_nuclide_material(0.0, 2, 5, 6);
        assert_eq!(inputs.chi_slab_meta, vec![0, 4, 4, 0, 5, 1, 6, 0]);
        assert_eq!(inputs.nuc_awr.len(), 2);
        let mut padded = NuclideSelectInputs::from_materials(&[(vec![0.0; 4], vec![1.0])], 4);
        padded.set_fission_chi(vec![0, 1, 1, 0], Vec::new());
        assert_eq!(padded.fission_channel_xs, vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "chi_slab_meta must be [n_slab x CHI_SLAB_META_COLS]")]
    fn set_fission_chi_rejects_a_table_of_the_wrong_size() {
        let mut inputs = NuclideSelectInputs::from_materials(&[(vec![0.0; 4], vec![1.0])], 4);
        inputs.set_fission_chi(vec![0, 1, 1], Vec::new());
    }
}
