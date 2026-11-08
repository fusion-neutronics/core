//! Energy function filter for tallies.
//!
//! This filter multiplies tally scores by an energy-dependent function,
//! typically used for dose calculations with ICRP dose coefficients. The
//! function is tabulated and interpolated by one of three rules, see
//! [`Interpolation`].

/// How an [`EnergyFunctionFilter`] interpolates between its tabulated points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Interpolation {
    /// Natural cubic spline in linear energy. Needs at least four points.
    #[default]
    Cubic,
    /// Linear in both energy and value.
    Linear,
    /// Linear in log energy and log value, the convention for tabulated
    /// coefficients such as ICRP dose coefficients and NIST attenuation data,
    /// and how contact dose reads them. An interval with a zero or negative
    /// value falls back to linear-linear, since it has no logarithm, and an
    /// interval whose two energies share a logarithm (an absorption edge, one
    /// ulp wide) is a step. Energies must be positive.
    LogLog,
}

impl Interpolation {
    /// Parse the user-facing name: `"cubic"`, `"linear"` or `"log-log"`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cubic" => Some(Self::Cubic),
            "linear" => Some(Self::Linear),
            "log-log" => Some(Self::LogLog),
            _ => None,
        }
    }

    /// The user-facing name, the inverse of [`Interpolation::from_name`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Cubic => "cubic",
            Self::Linear => "linear",
            Self::LogLog => "log-log",
        }
    }

    /// The fewest tabulated points the rule can interpolate.
    pub fn min_points(self) -> usize {
        match self {
            Self::Cubic => 4,
            Self::Linear | Self::LogLog => 2,
        }
    }

    /// The table-mode word a backend reads: [`EFUNC_MODE_POLYNOMIAL`] when each
    /// interval is a polynomial in linear energy (cubic and linear), or
    /// [`EFUNC_MODE_INTERVAL_KIND`] when each interval carries its own kind
    /// (log-log).
    pub fn mode_word(self) -> f64 {
        match self {
            Self::Cubic | Self::Linear => EFUNC_MODE_POLYNOMIAL,
            Self::LogLog => EFUNC_MODE_INTERVAL_KIND,
        }
    }
}

/// Table mode: interval `i`'s four words `[a, b, c, d]` give
/// `a + dx*(b + dx*(c + dx*d))` with `dx = energy - energy[i]`.
pub const EFUNC_MODE_POLYNOMIAL: f64 = 0.0;
/// Table mode: interval `i`'s four words are `[w0, w1, w2, kind]`, read by kind.
pub const EFUNC_MODE_INTERVAL_KIND: f64 = 1.0;
/// Interval kind: `exp(w0 + w1*(ln(energy) - w2))`, with `w0 = ln y_i`,
/// `w1` the log-log slope and `w2 = ln energy_i`.
pub const EFUNC_KIND_LOG_LOG: f64 = 0.0;
/// Interval kind: `w0 + w1*(energy - w2)`, with `w0 = y_i`, `w1` the linear
/// slope and `w2 = energy_i`.
pub const EFUNC_KIND_LINEAR: f64 = 1.0;
/// Interval kind: `w0` at or below `w2`, else `w1`. An absorption edge.
pub const EFUNC_KIND_STEP: f64 = 2.0;

/// Filter that multiplies tally scores by an energy-dependent function.
///
/// This filter is used to convert flux tallies to dose tallies by
/// multiplying the flux by energy-dependent dose conversion coefficients.
///
/// Unlike EnergyFilter which bins by energy, this filter always has
/// exactly 1 bin and applies a multiplicative weight based on the
/// particle's incident energy.
///
/// # Example
/// ```
/// use yamc_tallies::{EnergyFunctionFilter, Interpolation};
///
/// let energy = vec![1.0, 10.0, 100.0, 1000.0];
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let filter = EnergyFunctionFilter::new(energy, y, Interpolation::LogLog);
///
/// // Get weight at 50.0 eV
/// let weight = filter.get_weight(50.0);
/// assert!(weight.is_some());
/// ```
/// Serialization goes via [`EnergyFunctionFilterSerde`]: only the
/// user-supplied energy, y, units and interpolation survive on disk. The
/// interval coefficients are recomputed by `EnergyFunctionFilter::new` on load,
/// and a filter saved before the interpolation was recorded loads as cubic,
/// which is what it was.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(into = "EnergyFunctionFilterSerde", from = "EnergyFunctionFilterSerde")]
pub struct EnergyFunctionFilter {
    /// Energy grid in eV (must be monotonically increasing)
    energy: Vec<f64>,
    /// Interpolant values (same length as energy)
    y: Vec<f64>,
    /// How the values are interpolated.
    interpolation: Interpolation,
    /// Four words per interval, read according to `interpolation.mode_word()`
    /// (see [`EFUNC_MODE_POLYNOMIAL`] and [`EFUNC_MODE_INTERVAL_KIND`]).
    interval_coeffs: Vec<[f64; 4]>,
    /// Optional user-supplied units string (e.g., "pSv·cm²" for dose coefficients)
    pub units: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EnergyFunctionFilterSerde {
    pub energy: Vec<f64>,
    pub y: Vec<f64>,
    pub units: Option<String>,
    #[serde(default)]
    pub interpolation: Interpolation,
}

impl From<EnergyFunctionFilter> for EnergyFunctionFilterSerde {
    fn from(f: EnergyFunctionFilter) -> Self {
        Self {
            energy: f.energy,
            y: f.y,
            units: f.units,
            interpolation: f.interpolation,
        }
    }
}

impl From<EnergyFunctionFilterSerde> for EnergyFunctionFilter {
    fn from(s: EnergyFunctionFilterSerde) -> Self {
        let mut f = EnergyFunctionFilter::new(s.energy, s.y, s.interpolation);
        f.units = s.units;
        f
    }
}

impl EnergyFunctionFilter {
    /// Create a new EnergyFunctionFilter.
    ///
    /// # Arguments
    /// * `energy` - Energy grid in eV (must be monotonically increasing, and
    ///   positive for log-log)
    /// * `y` - Function values at each energy point
    /// * `interpolation` - How to interpolate between the points
    ///
    /// # Panics
    /// Panics if energy is not monotonically increasing, lengths don't match,
    /// there are fewer points than the interpolation needs (4 for cubic, 2
    /// otherwise), or a log-log grid has an energy that is not positive.
    pub fn new(energy: Vec<f64>, y: Vec<f64>, interpolation: Interpolation) -> Self {
        assert_eq!(
            energy.len(),
            y.len(),
            "Energy and y arrays must have the same length"
        );
        let min = interpolation.min_points();
        assert!(
            energy.len() >= min,
            "EnergyFunctionFilter requires at least {min} data points for {} interpolation",
            interpolation.name()
        );

        // Verify monotonically increasing
        for i in 1..energy.len() {
            assert!(
                energy[i] > energy[i - 1],
                "Energy grid must be monotonically increasing"
            );
        }
        if interpolation == Interpolation::LogLog {
            assert!(
                energy[0] > 0.0,
                "log-log interpolation needs positive energies"
            );
        }

        let interval_coeffs = match interpolation {
            Interpolation::Cubic => compute_cubic_spline_coefficients(&energy, &y),
            Interpolation::Linear => linear_coefficients(&energy, &y),
            Interpolation::LogLog => log_log_coefficients(&energy, &y),
        };

        Self {
            energy,
            y,
            interpolation,
            interval_coeffs,
            units: None,
        }
    }

    /// Create a new EnergyFunctionFilter with user-supplied units
    ///
    /// # Arguments
    /// * `energy` - Energy grid in eV (see [`EnergyFunctionFilter::new`])
    /// * `y` - Function values at each energy point
    /// * `interpolation` - How to interpolate between the points
    /// * `units` - Physical units string (e.g., "pSv·cm²")
    pub fn with_units(
        energy: Vec<f64>,
        y: Vec<f64>,
        interpolation: Interpolation,
        units: &str,
    ) -> Self {
        let mut filter = Self::new(energy, y, interpolation);
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

        Some(evaluate_interval(
            self.interpolation.mode_word(),
            self.energy[idx],
            self.interval_coeffs[idx],
            energy,
        ))
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

    /// How the values are interpolated.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// The four words per interval (so `energy().len() - 1` entries), read
    /// according to `interpolation().mode_word()`.
    ///
    /// Exposed so a backend can evaluate exactly the rule the CPU does
    /// (see [`evaluate_interval`]) rather than re-deriving the fit: the GPU
    /// kernel ships these straight through.
    pub fn interval_coeffs(&self) -> &[[f64; 4]] {
        &self.interval_coeffs
    }
}

/// Evaluate one interval of a packed energy-function table at `energy`.
///
/// `mode` is the table's mode word, `x_i` the interval's lower energy and `w`
/// its four words. The GPU twin and kernel in `yamc-gpu` implement this same
/// rule.
#[inline]
pub fn evaluate_interval(mode: f64, x_i: f64, w: [f64; 4], energy: f64) -> f64 {
    if mode == EFUNC_MODE_POLYNOMIAL {
        let dx = energy - x_i;
        let [a, b, c, d] = w;
        return a + dx * (b + dx * (c + dx * d));
    }
    let [w0, w1, w2, kind] = w;
    if kind == EFUNC_KIND_LOG_LOG {
        (w0 + w1 * (energy.ln() - w2)).exp()
    } else if kind == EFUNC_KIND_LINEAR {
        w0 + w1 * (energy - w2)
    } else if energy <= w2 {
        w0
    } else {
        w1
    }
}

/// Linear interpolation as a polynomial table: `[y_i, slope_i, 0, 0]`.
fn linear_coefficients(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    (0..x.len() - 1)
        .map(|i| [y[i], (y[i + 1] - y[i]) / (x[i + 1] - x[i]), 0.0, 0.0])
        .collect()
}

/// Log-log interpolation as an interval-kind table. The arithmetic is the one
/// `CoefficientTable::interpolate` uses for contact dose, so the two agree.
fn log_log_coefficients(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    (0..x.len() - 1)
        .map(|i| {
            let (x0, x1, y0, y1) = (x[i], x[i + 1], y[i], y[i + 1]);
            if y0 <= 0.0 || y1 <= 0.0 {
                // No logarithm: linear-linear over this interval.
                return [y0, (y1 - y0) / (x1 - x0), x0, EFUNC_KIND_LINEAR];
            }
            let log_span = x1.ln() - x0.ln();
            if log_span <= 0.0 {
                // An absorption edge one ulp wide: the jump happens here.
                return [y0, y1, x0, EFUNC_KIND_STEP];
            }
            let slope = (y1.ln() - y0.ln()) / log_span;
            [y0.ln(), slope, x0.ln(), EFUNC_KIND_LOG_LOG]
        })
        .collect()
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
        let filter = EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);

        assert_eq!(filter.num_bins(), 1);
        assert_eq!(filter.energy().len(), 4);
        assert_eq!(filter.y().len(), 4);
    }

    #[test]
    fn test_cubic_interpolation_at_data_points() {
        let energy = vec![1.0, 10.0, 100.0, 1000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy.clone(), y.clone(), Interpolation::Cubic);

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
        let filter = EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);

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
        let filter = EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);

        assert_eq!(filter.get_weight(5.0), None);
        assert_eq!(filter.get_weight(50000.0), None);
    }

    #[test]
    fn test_at_boundaries() {
        let energy = vec![10.0, 100.0, 1000.0, 10000.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let filter = EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);

        // At exact boundaries
        assert!(filter.get_weight(10.0).is_some());
        assert!(filter.get_weight(10000.0).is_some());
    }

    #[test]
    #[should_panic(expected = "monotonically increasing")]
    fn test_non_monotonic_energy_panics() {
        let energy = vec![10.0, 5.0, 100.0, 1000.0]; // Not monotonic
        let y = vec![1.0, 2.0, 3.0, 4.0];
        EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths_panics() {
        let energy = vec![10.0, 100.0, 1000.0, 10000.0];
        let y = vec![1.0, 2.0, 3.0]; // One less
        EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);
    }

    #[test]
    #[should_panic(expected = "at least 4")]
    fn test_too_few_points_panics() {
        let energy = vec![10.0, 100.0, 1000.0]; // Only 3 points
        let y = vec![1.0, 2.0, 3.0];
        EnergyFunctionFilter::new(energy, y, Interpolation::Cubic);
    }

    #[test]
    fn linear_interpolation_is_exact_on_a_straight_line() {
        let filter =
            EnergyFunctionFilter::new(vec![1.0, 3.0], vec![2.0, 6.0], Interpolation::Linear);
        assert_eq!(filter.get_weight(2.0), Some(4.0));
        assert_eq!(filter.get_weight(1.0), Some(2.0));
        assert_eq!(filter.get_weight(3.0), Some(6.0));
    }

    #[test]
    fn log_log_interpolation_is_exact_on_a_power_law() {
        // y = 5 E^0.7 is a straight line in log-log, so any point between
        // knots comes back to within rounding.
        let energy = vec![1.0, 10.0, 1e3, 1e6];
        let y: Vec<f64> = energy.iter().map(|e: &f64| 5.0 * e.powf(0.7)).collect();
        let filter = EnergyFunctionFilter::new(energy, y, Interpolation::LogLog);
        for e in [1.0, 2.5, 47.0, 9.9e5, 1e6] {
            let want = 5.0 * f64::powf(e, 0.7);
            let got = filter.get_weight(e).unwrap();
            assert!((got - want).abs() <= 1e-12 * want, "{e}: {got} vs {want}");
        }
    }

    #[test]
    fn log_log_needs_only_two_points() {
        let filter =
            EnergyFunctionFilter::new(vec![1.0, 100.0], vec![1.0, 100.0], Interpolation::LogLog);
        assert!((filter.get_weight(10.0).unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn log_log_falls_back_to_linear_where_a_value_is_zero() {
        let filter = EnergyFunctionFilter::new(
            vec![1.0, 2.0, 4.0],
            vec![0.0, 2.0, 8.0],
            Interpolation::LogLog,
        );
        assert_eq!(filter.get_weight(1.5), Some(1.0));
        assert!((filter.get_weight(2.828_427_124_746_19).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn log_log_steps_across_an_absorption_edge() {
        // An edge tabulated twice, one ulp apart, as photon coefficient tables
        // carry them: the value jumps there rather than ringing.
        let edge: f64 = 8.979e3;
        let below = f64::from_bits(edge.to_bits() - 1);
        let filter = EnergyFunctionFilter::new(
            vec![1e3, below, edge, 1e5],
            vec![100.0, 10.0, 80.0, 1.0],
            Interpolation::LogLog,
        );
        let just_below = filter.get_weight(below).unwrap();
        let just_above = filter.get_weight(edge).unwrap();
        assert!((just_below - 10.0).abs() < 1e-9, "{just_below}");
        assert!((just_above - 80.0).abs() < 1e-9, "{just_above}");
        // No ringing either side of the edge.
        for e in [2e3, 5e3, 8e3, 1e4, 5e4] {
            let w = filter.get_weight(e).unwrap();
            assert!((1.0..=100.0).contains(&w), "{e}: {w}");
        }
    }

    #[test]
    #[should_panic(expected = "positive energies")]
    fn log_log_refuses_a_zero_energy() {
        EnergyFunctionFilter::new(vec![0.0, 1.0], vec![1.0, 2.0], Interpolation::LogLog);
    }

    #[test]
    fn interpolation_names_round_trip() {
        for i in [
            Interpolation::Cubic,
            Interpolation::Linear,
            Interpolation::LogLog,
        ] {
            assert_eq!(Interpolation::from_name(i.name()), Some(i));
        }
        assert_eq!(Interpolation::from_name("spline"), None);
    }

    #[test]
    fn a_filter_saved_without_an_interpolation_loads_as_cubic() {
        let json = r#"{"energy":[1.0,10.0,100.0,1000.0],"y":[1.0,2.0,3.0,4.0],"units":null}"#;
        let filter: EnergyFunctionFilter = serde_json::from_str(json).unwrap();
        assert_eq!(filter.interpolation(), Interpolation::Cubic);
    }

    #[test]
    fn the_interpolation_survives_a_round_trip() {
        let filter =
            EnergyFunctionFilter::new(vec![1.0, 10.0], vec![1.0, 2.0], Interpolation::LogLog);
        let json = serde_json::to_string(&filter).unwrap();
        let back: EnergyFunctionFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(back, filter);
    }
}
