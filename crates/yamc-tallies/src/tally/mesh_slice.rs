use super::*;

impl Tally {
    /// Extract a 2D slice of mesh tally data for a given basis plane.
    ///
    /// Returns a 2D grid where values_2d[row][col] contains the tally value
    /// at that voxel in the slice.
    ///
    /// Basis planes:
    /// - "xy": fixed z, x horizontal (cols), y vertical (rows)
    /// - "xz": fixed y, x horizontal (cols), z vertical (rows)
    /// - "yz": fixed x, y horizontal (cols), z vertical (rows)
    ///
    /// # Arguments
    /// * `basis` - The slice plane: "xy", "xz", or "yz" str
    /// * `slice_coord` - Position along the fixed axis (if None, uses mesh center) float
    /// * `score_index` - Which score to extract (0-indexed) int
    /// * `energy_index` - Which energy bin (None = sum over all energy bins) int
    /// * `value_type` - "mean", "standard_deviation", or "relative_error" str
    ///
    /// # Returns
    /// Ok(values_2d) or Err with message
    pub fn extract_mesh_slice(
        &self,
        basis: &str,
        slice_coord: Option<f64>,
        score_index: usize,
        energy_index: Option<usize>,
        value_type: &str,
    ) -> Result<Vec<Vec<f64>>, String> {
        let mesh_filter = self.get_mesh_filter().ok_or("Tally has no MeshFilter")?;
        let mesh = mesh_filter
            .rectangular_mesh()
            .ok_or("extract_mesh_slice requires a rectangular mesh")?;

        let ll = mesh.lower_left();
        let ur = mesh.upper_right();
        let dim = mesh.shape();
        let width = mesh.width();

        // Determine fixed axis and (h_size, v_size) for this basis.
        let (fixed_axis, h_size, _v_size) = Self::slice_geometry(basis, dim)?;

        // Determine which bin along the fixed axis the coordinate falls in.
        let fixed_dim = dim[fixed_axis];
        let fixed_ll = ll[fixed_axis];
        let fixed_ur = ur[fixed_axis];
        let fixed_width = width[fixed_axis];

        let coord = slice_coord.unwrap_or((fixed_ll + fixed_ur) / 2.0);

        if coord < fixed_ll || coord >= fixed_ur {
            return Err(format!(
                "slice_coord {coord:.4} is outside mesh bounds [{fixed_ll:.4}, {fixed_ur:.4})"
            ));
        }

        let fixed_index = ((coord - fixed_ll) / fixed_width).floor() as usize;
        let fixed_index = fixed_index.min(fixed_dim - 1);

        // Reuse the multi-slice extractor (single slice) and reshape the
        // flat row-major buffer into the `Vec<Vec<f64>>` grid this
        // function's callers expect.
        let slices =
            self.extract_mesh_slices(basis, &[fixed_index], score_index, energy_index, value_type)?;
        let (_, flat) = slices
            .into_iter()
            .next()
            .ok_or("extract_mesh_slice produced no slice")?;

        let values_2d = flat
            .chunks(h_size)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        Ok(values_2d)
    }

    /// Resolve the slice geometry for a basis plane: the fixed axis index
    /// (0=x, 1=y, 2=z) and the `(h_size, v_size)` of the resulting 2D grid.
    fn slice_geometry(basis: &str, dim: [usize; 3]) -> Result<(usize, usize, usize), String> {
        let [nx, ny, nz] = dim;
        match basis {
            "xy" => Ok((2, nx, ny)),
            "xz" => Ok((1, nx, nz)),
            "yz" => Ok((0, ny, nz)),
            _ => Err(format!(
                "Invalid basis '{basis}', must be 'xy', 'xz', or 'yz'"
            )),
        }
    }

    /// Extract multiple 2D slices from a mesh tally for the interactive viewer.
    ///
    /// Returns `Vec<(bin_index, flat_2d_row_major)>` where each slice is a flat
    /// `Vec<f64>` of length `h_size * v_size` in row-major order (v slow, h fast).
    pub fn extract_mesh_slices(
        &self,
        basis: &str,
        slice_indices: &[usize],
        score_index: usize,
        energy_index: Option<usize>,
        value_type: &str,
    ) -> Result<Vec<(usize, Vec<f64>)>, String> {
        let mesh_filter = self.get_mesh_filter().ok_or("Tally has no MeshFilter")?;
        let mesh = mesh_filter
            .rectangular_mesh()
            .ok_or("extract_mesh_slices requires a rectangular mesh")?;
        let dim = mesh.shape();

        let (fixed_axis, h_size, v_size) = Self::slice_geometry(basis, dim)?;

        let fixed_dim = dim[fixed_axis];

        // Validate score_index
        if score_index >= self.scores.len() {
            return Err(format!(
                "score_index {} out of range (tally has {} scores)",
                score_index,
                self.scores.len()
            ));
        }

        let data = match value_type {
            "mean" => self.get_mean(),
            "standard_deviation" => self.get_std_dev(),
            "relative_error" => self.get_rel_error(),
            _ => {
                return Err(format!(
                    "Invalid value_type '{value_type}', must be 'mean', 'standard_deviation', or 'relative_error'"
                ))
            }
        };

        let num_energy_bins = self.num_energy_bins();
        let num_mesh_bins = mesh_filter.num_bins();

        let mut result = Vec::with_capacity(slice_indices.len());

        for &fixed_index in slice_indices {
            if fixed_index >= fixed_dim {
                return Err(format!(
                    "slice index {} out of range for {} axis (dimension {})",
                    fixed_index,
                    match basis {
                        "xy" => "z",
                        "xz" => "y",
                        "yz" => "x",
                        _ => "?",
                    },
                    fixed_dim
                ));
            }

            let mut flat = vec![0.0f64; h_size * v_size];

            for v in 0..v_size {
                for h in 0..h_size {
                    let (ix, iy, iz) = match basis {
                        "xy" => (h, v, fixed_index),
                        "xz" => (h, fixed_index, v),
                        "yz" => (fixed_index, h, v),
                        _ => unreachable!(),
                    };

                    // Share the mesh's own index mapping rather than
                    // open-coding `(iz * ny + iy) * nx + ix` here.
                    let mesh_bin = mesh.get_bin_from_indices([ix, iy, iz]);

                    let val = if let Some(e_idx) = energy_index {
                        if e_idx >= num_energy_bins {
                            return Err(format!(
                                "energy_index {e_idx} out of range (tally has {num_energy_bins} energy bins)"
                            ));
                        }
                        let bin_idx = score_index * (num_energy_bins * num_mesh_bins)
                            + e_idx * num_mesh_bins
                            + mesh_bin;
                        data.get(bin_idx).copied().unwrap_or(0.0)
                    } else {
                        let mut sum = 0.0;
                        for e in 0..num_energy_bins {
                            let bin_idx = score_index * (num_energy_bins * num_mesh_bins)
                                + e * num_mesh_bins
                                + mesh_bin;
                            sum += data.get(bin_idx).copied().unwrap_or(0.0);
                        }
                        sum
                    };

                    flat[v * h_size + h] = val;
                }
            }

            result.push((fixed_index, flat));
        }

        Ok(result)
    }
}
