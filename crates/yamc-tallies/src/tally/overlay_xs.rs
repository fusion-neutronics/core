use super::*;

impl OverlayXsData {
    /// Look up microscopic neutron cross section for a nuclide/MT at given energy.
    /// Uses O(1) log-grid lookup with linear interpolation. Returns barns.
    pub fn lookup_neutron(&self, nuclide: &str, mt: i32, energy: f64) -> f64 {
        let xs_vec = match self.micro_xs.get(nuclide).and_then(|m| m.get(&mt)) {
            Some(v) => v,
            None => return 0.0,
        };
        if self.energy_grid.len() < 2 || xs_vec.len() != self.energy_grid.len() {
            return 0.0;
        }
        let n = self.energy_grid.len();

        // Boundary cases
        if energy <= self.energy_grid[0] {
            return xs_vec[0];
        }
        if energy >= self.energy_grid[n - 1] {
            return xs_vec[n - 1];
        }

        // O(1) log-grid lookup
        let log_e = energy.ln();
        let bin = ((log_e - self.log_e_min) * self.inv_log_delta) as usize;
        let bin = bin.min(OVERLAY_N_LOG_BINS - 1);

        let i_low = self.log_grid_index[bin] as usize;
        let i_high = (self.log_grid_index[bin + 1] as usize + 1).min(n);

        // Short linear search in narrow range
        let mut i_grid = i_low;
        while i_grid < i_high - 1 && self.energy_grid[i_grid + 1] <= energy {
            i_grid += 1;
        }
        let i_grid = i_grid.min(n - 2);

        // Linear interpolation
        let e0 = self.energy_grid[i_grid];
        let e1 = self.energy_grid[i_grid + 1];
        let f = (energy - e0) / (e1 - e0);
        xs_vec[i_grid] + f * (xs_vec[i_grid + 1] - xs_vec[i_grid])
    }

    /// Look up microscopic photon cross sections for a nuclide at given energy.
    /// Returns the full ElementMicroXS struct (coherent, incoherent, photoelectric,
    /// pair production, heating) in barns.
    pub fn lookup_photon(
        &self,
        nuclide: &str,
        energy: f64,
    ) -> Option<yamc_element::photon::ElementMicroXS> {
        self.photon_elements
            .get(nuclide)
            .map(|elem| elem.calculate_xs(energy))
    }

    /// Look up microscopic photon heating KERMA cross section for a nuclide.
    /// Returns eV·barns (energy deposited per unit fluence).
    pub fn lookup_photon_heating(&self, nuclide: &str, energy: f64) -> f64 {
        match self.photon_elements.get(nuclide) {
            Some(elem) => elem.calculate_xs(energy).heating,
            None => 0.0,
        }
    }

    /// Per-nuclide number density (atoms / barn-cm) to weight the microscopic
    /// XS by. Returns `1.0` for any nuclide without a stored density, so a
    /// unit-density nuclide overlay (empty `densities`) behaves exactly as
    /// before - the historical microscopic overlay scored bare barns.
    pub fn density(&self, nuclide: &str) -> f64 {
        self.densities.get(nuclide).copied().unwrap_or(1.0)
    }

    /// The overlay's own nuclide list (stable order). Used by the combined
    /// material-response path, where `Tally::nuclides` is just `[Total]`.
    pub fn nuclide_names(&self) -> &[String] {
        &self.nuclide_names
    }

    /// Whether this overlay collapses to a single combined macroscopic bin
    /// (material response) vs one bin per nuclide (unit-density overlay).
    pub fn combine(&self) -> bool {
        self.combine
    }
}
