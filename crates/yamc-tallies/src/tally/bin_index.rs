use super::*;

impl Tally {
    /// Get the flat index for 7D indexing:
    /// `score → cell → material → nuclide → parent_nuclide → energy → mesh`
    ///
    /// Dimension order is outermost (slowest-varying) to innermost. Mesh stays the
    /// innermost dimension so that each nuclide's spatial data forms a contiguous
    /// block, which is optimal for D1S time-correction-factor application
    /// (`block *= scalar`).
    ///
    /// When a filter is absent or single-bin, its `*_bin` argument must be 0 and
    /// the corresponding dimension collapses to 1, preserving the pre-existing
    /// layout for the common case.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn get_bin_index_7d(
        &self,
        score_index: usize,
        cell_bin: usize,
        material_bin: usize,
        nuclide_bin: usize,
        parent_bin: usize,
        energy_bin: usize,
        mesh_bin: usize,
    ) -> Option<usize> {
        let num_cell_bins = self.num_cell_bins();
        let num_material_bins = self.num_material_bins();
        let num_nuclide_bins = self.num_nuclide_bins();
        let num_parent_bins = self.num_parent_nuclide_bins();
        let num_energy_bins = self.num_energy_bins();
        let num_mesh_bins = self.num_mesh_bins();

        if score_index >= self.scores.len()
            || cell_bin >= num_cell_bins
            || material_bin >= num_material_bins
            || nuclide_bin >= num_nuclide_bins
            || parent_bin >= num_parent_bins
            || energy_bin >= num_energy_bins
            || mesh_bin >= num_mesh_bins
        {
            return None;
        }

        let stride_mesh = num_mesh_bins;
        let stride_energy = num_energy_bins * stride_mesh;
        let stride_parent = num_parent_bins * stride_energy;
        let stride_nuclide = num_nuclide_bins * stride_parent;
        let stride_material = num_material_bins * stride_nuclide;
        let stride_cell = num_cell_bins * stride_material;

        Some(
            score_index * stride_cell
                + cell_bin * stride_material
                + material_bin * stride_nuclide
                + nuclide_bin * stride_parent
                + parent_bin * stride_energy
                + energy_bin * stride_mesh
                + mesh_bin,
        )
    }
}
