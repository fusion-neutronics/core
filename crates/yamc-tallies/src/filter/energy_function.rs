//! Energy function filter for tallies with cubic spline interpolation.
//!
//! This filter multiplies tally scores by an energy-dependent function,
//! typically used for dose calculations with ICRP dose coefficients.
//! Uses cubic spline interpolation as recommended by ICRP.

/// Filter that multiplies tally scores by an energy-dependent function.
///
/// This filter is used to convert flux tallies to dose tallies by
/// multiplying the flux by energy-dependent dose conversion coefficients.
///
/// Uses natural cubic spline interpolation as recommended by ICRP for dose coefficients.
///
/// Unlike EnergyFilter which bins by energy, this filter always has
/// exactly 1 bin and applies a multiplicative weight based on the
/// particle's incident energy.
///
/// # Example
/// ```
/// use yamc_tallies::EnergyFunctionFilter;
///
/// let energy = vec![1.0, 10.0, 100.0, 1000.0];
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let filter = EnergyFunctionFilter::new(energy, y);
///
/// // Get weight at 50.0 eV
/// let weight = filter.get_weight(50.0);
/// assert!(weight.is_some());
/// ```
/// Serialization goes via [`EnergyFunctionFilterSerde`] -- only the
/// user-supplied energy / y / units survive on disk. Spline
/// coefficients are recomputed by `EnergyFunctionFilter::new` on load.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(into = "EnergyFunctionFilterSerde", from = "EnergyFunctionFilterSerde")]
pub struct EnergyFunctionFilter {
    /// Energy grid in eV (must be monotonically increasing)
    energy: Vec<f64>,
    /// Interpolant values (same length as energy)
    y: Vec<f64>,
    /// Spline coefficients for cubic interpolation (computed on construction)
    /// For each interval i, coefficients are [a_i, b_i, c_i, d_i] where
    /// y(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    spline_coeffs: Vec<[f64; 4]>,
    /// Optional user-supplied units string (e.g., "pSv·cm²" for dose coefficients)
    pub units: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EnergyFunctionFilterSerde {
    pub energy: Vec<f64>,
    pub y: Vec<f64>,
    pub units: Option<String>,
}

impl From<EnergyFunctionFilter> for EnergyFunctionFilterSerde {
    fn from(f: EnergyFunctionFilter) -> Self {
        Self {
            energy: f.energy,
            y: f.y,
            units: f.units,
        }
    }
}

impl From<EnergyFunctionFilterSerde> for EnergyFunctionFilter {
    fn from(s: EnergyFunctionFilterSerde) -> Self {
        let mut f = EnergyFunctionFilter::new(s.energy, s.y);
        f.units = s.units;
        f
    }
}

impl EnergyFunctionFilter {
    /// Create a new EnergyFunctionFilter with cubic spline interpolation
    ///
    /// # Arguments
    /// * `energy` - Energy grid in eV (must be monotonically increasing, >= 4 points)
    /// * `y` - Function values at each energy point
    ///
    /// # Panics
    /// Panics if energy is not monotonically increasing, lengths don't match,
    /// or fewer than 4 data points (required for cubic spline)
    pub fn new(energy: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(
            energy.len(),
            y.len(),
            "Energy and y arrays must have the same length"
        );
        assert!(
            energy.len() >= 4,
            "EnergyFunctionFilter requires at least 4 data points for cubic interpolation"
        );

        // Verify monotonically increasing
        for i in 1..energy.len() {
            assert!(
                energy[i] > energy[i - 1],
                "Energy grid must be monotonically increasing"
            );
        }

        // Compute natural cubic spline coefficients
        let spline_coeffs = compute_cubic_spline_coefficients(&energy, &y);

        Self {
            energy,
            y,
            spline_coeffs,
            units: None,
        }
    }

    /// Create a new EnergyFunctionFilter with user-supplied units
    ///
    /// # Arguments
    /// * `energy` - Energy grid in eV (must be monotonically increasing, >= 4 points)
    /// * `y` - Function values at each energy point
    /// * `units` - Physical units string (e.g., "pSv·cm²")
    pub fn with_units(energy: Vec<f64>, y: Vec<f64>, units: &str) -> Self {
        let mut filter = Self::new(energy, y);
        filter.units = Some(units.to_string());
        filter
    }

    /// Get the weight (interpolated y value) for a given energy
    ///
    /// Returns None if energy is outside the grid range
    #[inline]
    pub fn get_weight(&self, energy: f64) -> Option<f64> {
        if energy < self.energy[0] || energy > *self.energy.last().unwrap() {
            return None;
        }

        // Binary search for the interval
        let idx = match self
            .energy
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap())
        {
            Ok(i) => i.min(self.energy.len() - 2),
            Err(i) => (i.saturating_sub(1)).min(self.energy.len() - 2),
        };

        // Cubic spline evaluation: y = a + b*dx + c*dx^2 + d*dx^3
        let dx = energy - self.energy[idx];
        let [a, b, c, d] = self.spline_coeffs[idx];
        let weight = a + dx * (b + dx * (c + dx * d));

        Some(weight)
    }

    /// This filter always has exactly 1 bin
    #[inline]
    pub fn num_bins(&self) -> usize {
        1
    }

    /// Get the energy grid
    pub fn energy(&self) -> &[f64] {
        &self.energy
    }

    /// Get the y values
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Get the precomputed natural-cubic-spline coefficients, one
    /// `[a, b, c, d]` per interval (so `energy().len() - 1` entries).
    ///
    /// Exposed so a backend that cannot solve the spline itself can evaluate
    /// the SAME polynomial the CPU does. The GPU kernel ships these straight
    /// through and evaluates `a + dx*(b + dx*(c + dx*d))` on linear energy,
    /// which is what makes GPU and CPU agree bit for bit rather than merely
    /// closely (issue #271).
    pub fn spline_coeffs(&self) -> &[[f64; 4]] {
        &self.spline_coeffs
    }
}

/// Compute natural cubic spline coefficients using Thomas algorithm
/// Returns coefficients [a, b, c, d] for each interval where:
/// y(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
fn compute_cubic_spline_coefficients(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let mut h = vec![0.0; n - 1];
    let mut alpha = vec![0.0; n - 1];
    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];

    // Step 1: Compute h_i = x_{i+1} - x_i
    for i in 0..n - 1 {
        h[i] = x[i + 1] - x[i];
    }

    // Step 2: Compute alpha_i
    for i in 1..n - 1 {
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Step 3-4: Solve tridiagonal system (Thomas algorithm)
    for i in 1..n - 1 {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Step 5: Back substitution
    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    // Collect coefficients [a, b, c, d] for each interval
    let mut coeffs = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        coeffs.push([y[i], b[i], c[i], d[i]]);
    }

    coeffs
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_function_filter_creation() {
        let energy = vec![1.0, 10.0, 100.0, 1000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy, y);

        assert_eq!(filter.num_bins(), 1);
        assert_eq!(filter.energy().len(), 4);
        assert_eq!(filter.y().len(), 4);
    }

    #[test]
    fn test_cubic_interpolation_at_data_points() {
        let energy = vec![1.0, 10.0, 100.0, 1000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy.clone(), y.clone());

        // At data points, cubic spline should return exact values
        for (i, &e) in energy.iter().enumerate() {
            let weight = filter.get_weight(e).unwrap();
            assert!(
                (weight - y[i]).abs() < 1e-10,
                "At energy {}, expected {}, got {}",
                e,
                y[i],
                weight
            );
        }
    }

    #[test]
    fn test_cubic_interpolation_smooth() {
        // Create data from a smooth function (y = x^0.5)
        let energy = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // sqrt values
        let filter = EnergyFunctionFilter::new(energy, y);

        // Test at midpoint - cubic spline should give smooth result
        let weight = filter.get_weight(6.25).unwrap(); // sqrt(6.25) = 2.5
                                                       // Allow some deviation since cubic spline won't be exact for sqrt
        assert!(
            (weight - 2.5).abs() < 0.3,
            "Interpolated value at 6.25 should be close to 2.5, got {}",
            weight
        );
    }

    #[test]
    fn test_outside_range() {
        let energy = vec![10.0, 100.0, 1000.0, 10000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy, y);

        assert_eq!(filter.get_weight(5.0), None);
        assert_eq!(filter.get_weight(50000.0), None);
    }

    #[test]
    fn test_at_boundaries() {
        let energy = vec![10.0, 100.0, 1000.0, 10000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy, y);

        // At exact boundaries
        assert!(filter.get_weight(10.0).is_some());
        assert!(filter.get_weight(10000.0).is_some());
    }

    #[test]
    #[should_panic(expected = "monotonically increasing")]
    fn test_non_monotonic_energy_panics() {
        let energy = vec![10.0, 5.0, 100.0, 1000.0]; // Not monotonic
        let y = vec![1.0, 2.0, 3.0, 4.0];
        EnergyFunctionFilter::new(energy, y);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths_panics() {
        let energy = vec![10.0, 100.0, 1000.0, 10000.0];
        let y = vec![1.0, 2.0, 3.0]; // One less
        EnergyFunctionFilter::new(energy, y);
    }

    #[test]
    #[should_panic(expected = "at least 4")]
    fn test_too_few_points_panics() {
        let energy = vec![10.0, 100.0, 1000.0]; // Only 3 points
        let y = vec![1.0, 2.0, 3.0];
        EnergyFunctionFilter::new(energy, y);
    }
}
