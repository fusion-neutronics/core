/// Energy filter for tallies - filters events based on particle energy
/// Energy bins are defined by bin edges [E0, E1, E2, ..., En]
/// This creates bins: [E0, E1), [E1, E2), ..., [En-1, En]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EnergyFilter {
    /// Energy bin boundaries in eV, must be in ascending order
    /// For n boundaries, creates n-1 bins
    pub bins: Vec<f64>,
}

impl EnergyFilter {
    /// Create a new EnergyFilter with the given bin boundaries
    ///
    /// # Arguments
    /// * `bins` - Energy bin boundaries in eV, must be in ascending order
    ///
    /// # Returns
    /// A new EnergyFilter
    ///
    /// # Panics
    /// Panics if bins are not in ascending order or if there are fewer than 2 bins
    pub fn new(bins: Vec<f64>) -> Self {
        if bins.len() < 2 {
            panic!("EnergyFilter requires at least 2 bin boundaries (to create at least 1 bin)");
        }

        // Verify bins are in ascending order
        for i in 1..bins.len() {
            if bins[i] <= bins[i - 1] {
                panic!("Energy bins must be in strictly ascending order");
            }
        }

        Self { bins }
    }

    /// Create an EnergyFilter from a named group structure
    ///
    /// # Arguments
    /// * `name` - Name of the group structure (e.g., "VITAMIN-J-175")
    ///
    /// # Returns
    /// * `Ok(EnergyFilter)` - Filter with boundaries from the named structure
    /// * `Err(String)` - Error if structure name is not recognized
    ///
    /// # Example
    /// ```
    /// use yamc_tallies::filter::energy::EnergyFilter;
    ///
    /// let filter = EnergyFilter::from_group_structure("VITAMIN-J-175").unwrap();
    /// assert_eq!(filter.num_bins(), 175);
    /// ```
    pub fn from_group_structure(name: &str) -> Result<Self, String> {
        use yamc_nuclide::group_structures::get_group_structure;

        let boundaries = get_group_structure(name)?;
        Ok(Self::new(boundaries.to_vec()))
    }

    /// Get the bin index for a given energy
    ///
    /// # Arguments
    /// * `energy` - The particle energy in eV
    ///
    /// # Returns
    /// `Some(bin_index)` if the energy falls within the filter range, `None` otherwise
    ///
    /// # Bin Convention
    /// Uses the following convention:
    /// - First bin: [E0, E1] (includes both boundaries)
    /// - Other bins: (Ei-1, Ei] (excludes lower, includes upper)
    /// - E < E0 or E > En is outside
    pub fn get_bin(&self, energy: f64) -> Option<usize> {
        // Energy must be >= first bin and <= last bin
        if energy < self.bins[0] || energy > *self.bins.last().unwrap() {
            return None;
        }

        // Binary search for the bin
        // We want to find i such that bins[i] < energy <= bins[i+1]
        // Exception: first bin includes energy == bins[0]
        let result = self.bins.binary_search_by(|&bin| {
            if bin < energy {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        match result {
            Ok(i) => Some(i),
            Err(i) => {
                if i > 0 && i < self.bins.len() {
                    Some(i - 1)
                } else if i == 0 {
                    // Handle E == bins[0] case: should go to bin 0
                    Some(0)
                } else {
                    None
                }
            }
        }
    }

    /// Check if this filter matches a given energy (returns true if energy is in range)
    ///
    /// # Arguments
    /// * `energy` - The particle energy in eV to check
    ///
    /// # Returns
    /// `true` if the energy falls within any bin of this filter
    pub fn matches(&self, energy: f64) -> bool {
        self.get_bin(energy).is_some()
    }

    /// Get the number of energy bins
    pub fn num_bins(&self) -> usize {
        self.bins.len() - 1
    }

    /// Base-10 log width of each energy bin: `log10(E_high / E_low)`.
    ///
    /// Useful for plotting lethargy-normalized flux spectra.
    pub fn lethargy_bin_width(&self) -> Vec<f64> {
        (0..self.num_bins())
            .map(|i| (self.bins[i + 1] / self.bins[i]).log10())
            .collect()
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lethargy_bin_width() {
        // 3 bins: [1, 10), [10, 100), [100, 1000] → widths all log10(10) = 1.0
        let f = EnergyFilter::new(vec![1.0, 10.0, 100.0, 1000.0]);
        let w = f.lethargy_bin_width();
        assert_eq!(w.len(), 3);
        for &wi in &w {
            assert!((wi - 1.0).abs() < 1e-12);
        }

        // Non-uniform: [1, 100, 1000] → [2.0, 1.0]
        let f2 = EnergyFilter::new(vec![1.0, 100.0, 1000.0]);
        let w2 = f2.lethargy_bin_width();
        assert!((w2[0] - 2.0).abs() < 1e-12);
        assert!((w2[1] - 1.0).abs() < 1e-12);
    }
}
