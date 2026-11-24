//! Tabulated functions: the TAB1 and TAB2 types of the ENDF-6 format.

use crate::data::EV_PER_MEV;

/// A one-dimensional tabulated function, mirroring the format's TAB1 type.
///
/// The `(x, y)` pairs are interpolated according to `interpolation`, one scheme
/// per region, where `breakpoints` holds the one-based index of each region's
/// last point.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tabulated1D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// One-based index of the last point in each interpolation region.
    pub breakpoints: Vec<i32>,
    /// ENDF interpolation scheme per region: 1 histogram, 2 linear-linear,
    /// 3 linear-log, 4 log-linear, 5 log-log.
    pub interpolation: Vec<i32>,
    /// Where this function starts on the nuclide's own energy grid.
    ///
    /// A processed cross section begins at its threshold rather than at the
    /// bottom of the grid, and summing several of them needs to know where
    /// each one began. `None` for anything not read from an ACE table.
    pub threshold_idx: Option<usize>,
}

impl Tabulated1D {
    /// A function with a single linear-linear region, the format's default.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len() as i32;
        Tabulated1D {
            x,
            y,
            breakpoints: vec![n],
            interpolation: vec![2],
            threshold_idx: None,
        }
    }

    pub fn with_regions(
        x: Vec<f64>,
        y: Vec<f64>,
        breakpoints: Vec<i32>,
        interpolation: Vec<i32>,
    ) -> Self {
        if breakpoints.is_empty() || interpolation.is_empty() {
            return Tabulated1D::new(x, y);
        }
        Tabulated1D {
            x,
            y,
            breakpoints,
            interpolation,
            threshold_idx: None,
        }
    }

    pub fn n_pairs(&self) -> usize {
        self.x.len()
    }

    pub fn n_regions(&self) -> usize {
        self.breakpoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Evaluate the function at `x`, clamping to the tabulated range.
    pub fn eval(&self, x: f64) -> f64 {
        if self.x.is_empty() {
            return 0.0;
        }
        if x <= self.x[0] {
            return self.y[0];
        }
        let last = self.x.len() - 1;
        if x >= self.x[last] {
            return self.y[last];
        }

        // Index of the bin containing x: the count of points at or below it,
        // less one.
        let idx = self.x.partition_point(|&v| v <= x) - 1;

        // The scheme for the region this bin falls in. Regions are ordered, so
        // the first breakpoint past the bin wins; if none is, the last scheme
        // applies.
        let mut scheme = *self.interpolation.last().unwrap_or(&2);
        for (b, p) in self.breakpoints.iter().zip(&self.interpolation) {
            if (idx as i64) < (*b as i64) - 1 {
                scheme = *p;
                break;
            }
        }

        let (xi, xi1) = (self.x[idx], self.x[idx + 1]);
        let (yi, yi1) = (self.y[idx], self.y[idx + 1]);

        match scheme {
            1 => yi,
            2 => yi + (x - xi) / (xi1 - xi) * (yi1 - yi),
            3 => yi + (x / xi).ln() / (xi1 / xi).ln() * (yi1 - yi),
            4 => yi * ((x - xi) / (xi1 - xi) * (yi1 / yi).ln()).exp(),
            5 => yi * ((x / xi).ln() / (xi1 / xi).ln() * (yi1 / yi).ln()).exp(),
            _ => yi + (x - xi) / (xi1 - xi) * (yi1 - yi),
        }
    }

    /// Partial integrals from the start of the range to each tabulated point.
    ///
    /// The returned vector is the same length as `x`, beginning at zero.
    pub fn integral(&self) -> Vec<f64> {
        let n = self.x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut partial = vec![0.0; n.saturating_sub(1)];

        let mut i_low = 0usize;
        for (k, &b) in self.breakpoints.iter().enumerate() {
            let i_high = ((b as i64 - 1).max(0) as usize).min(n.saturating_sub(1));
            let scheme = *self.interpolation.get(k).unwrap_or(&2);
            // The bin index reads x and y at both i and i+1, so it is the index
            // itself that is wanted here, not an iterator over one of them.
            #[allow(clippy::needless_range_loop)]
            for i in i_low..i_high {
                let (x0, x1) = (self.x[i], self.x[i + 1]);
                let (y0, y1) = (self.y[i], self.y[i + 1]);
                partial[i] = match scheme {
                    1 => y0 * (x1 - x0),
                    2 => {
                        let m = (y1 - y0) / (x1 - x0);
                        (y0 - m * x0) * (x1 - x0) + m * (x1 * x1 - x0 * x0) / 2.0
                    }
                    3 => {
                        let logx = (x1 / x0).ln();
                        let m = (y1 - y0) / logx;
                        y0 + m * (x1 * (logx - 1.0) + x0)
                    }
                    4 => {
                        let m = (y1 / y0).ln() / (x1 - x0);
                        y0 / m * ((m * (x1 - x0)).exp() - 1.0)
                    }
                    5 => {
                        let m = (y1 / y0).ln() / (x1 / x0).ln();
                        y0 / ((m + 1.0) * x0.powf(m)) * (x1.powf(m + 1.0) - x0.powf(m + 1.0))
                    }
                    _ => 0.0,
                };
            }
            i_low = i_high;
        }

        let mut out = Vec::with_capacity(n);
        out.push(0.0);
        let mut running = 0.0;
        for v in partial {
            running += v;
            out.push(running);
        }
        out
    }

    /// Read a tabulated function from an ACE table's XSS array.
    ///
    /// `idx` is the zero-based offset of the record within `xss`, which begins
    /// with the number of interpolation regions. `convert_units` multiplies the
    /// abscissa by [`EV_PER_MEV`], which is wanted whenever it is an energy.
    ///
    /// The number of values consumed is
    /// `2 + 2 * n_regions + 2 * n_pairs`; [`Self::ace_len`] computes it for a
    /// caller that has to step past the record.
    pub fn from_ace(xss: &[f64], idx: usize, convert_units: bool) -> Tabulated1D {
        let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);

        let n_regions = at(idx) as usize;
        let n_pairs = at(idx + 1 + 2 * n_regions) as usize;

        let mut idx = idx + 1;
        let (breakpoints, interpolation) = if n_regions > 0 {
            (
                (0..n_regions).map(|i| at(idx + i) as i32).collect(),
                (0..n_regions)
                    .map(|i| at(idx + n_regions + i) as i32)
                    .collect(),
            )
        } else {
            // Zero regions means one linear-linear region over the whole table.
            (vec![n_pairs as i32], vec![2])
        };

        idx += 2 * n_regions + 1;
        let mut x: Vec<f64> = (0..n_pairs).map(|i| at(idx + i)).collect();
        let y: Vec<f64> = (0..n_pairs).map(|i| at(idx + n_pairs + i)).collect();

        if convert_units {
            for v in &mut x {
                *v *= EV_PER_MEV;
            }
        }

        Tabulated1D {
            x,
            y,
            breakpoints,
            interpolation,
            threshold_idx: None,
        }
    }

    /// Number of XSS values the record at `idx` occupies.
    pub fn ace_len(xss: &[f64], idx: usize) -> usize {
        let at = |i: usize| xss.get(i).copied().unwrap_or(0.0);
        let n_regions = at(idx) as usize;
        let n_pairs = at(idx + 1 + 2 * n_regions) as usize;
        2 + 2 * n_regions + 2 * n_pairs
    }
}

/// A Legendre series, `sum_l c_l P_l(x)`.
///
/// The angular distributions of MF=4 are given this way. The coefficients are
/// the series' own, not those of the equivalent power polynomial, so this is a
/// different type from [`Polynomial`] rather than a conversion of it.
///
/// Evaluation and integration follow `numpy.polynomial.legendre` step for
/// step: Clenshaw recurrence for the value, the three-term relation for the
/// antiderivative. That is because the Python reader uses it and the two have to agree
/// to the last bit.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Legendre {
    /// Coefficients in ascending order of degree.
    pub coefficients: Vec<f64>,
}

impl Legendre {
    /// A series with the given coefficients, kept exactly as passed.
    ///
    /// Trailing zeros are not dropped, matching `numpy.polynomial`, whose
    /// constructor takes `trim=False`. It matters: the Clenshaw recurrence
    /// takes one step per coefficient, so a trailing zero moves the last bits
    /// of every value the series produces.
    pub fn new(coefficients: Vec<f64>) -> Self {
        Legendre { coefficients }
    }

    /// Evaluate the series at `x` by Clenshaw recurrence, as `legval` does.
    ///
    /// The two ratios are formed before they multiply `c1`, which is how
    /// NumPy 2 spells the recurrence. Associating the other way, as NumPy 1
    /// did, changes the last bit or two of the result.
    pub fn eval(&self, x: f64) -> f64 {
        legval(x, &self.coefficients)
    }

    /// The antiderivative that vanishes at zero, as `legint` gives it.
    pub fn integ(&self) -> Legendre {
        let c = &self.coefficients;
        let n = c.len();
        if n == 0 {
            return Legendre::new(Vec::new());
        }
        if n == 1 && c[0] == 0.0 {
            return Legendre::new(vec![0.0]);
        }

        let mut tmp = vec![0.0; n + 1];
        tmp[1] = c[0];
        if n > 1 {
            tmp[2] = c[1] / 3.0;
        }
        for j in 2..n {
            let d = c[j] / (2.0 * j as f64 + 1.0);
            tmp[j + 1] = d;
            tmp[j - 1] -= d;
        }
        // Shift so the antiderivative vanishes at the lower bound of zero.
        // `tmp[0]` is still zero at this point, so subtracting the value there
        // is the whole of the adjustment. `legint` evaluates the untrimmed
        // coefficients here and trims only what it returns.
        tmp[0] -= legval(0.0, &tmp);
        Legendre::new(tmp)
    }
}

/// `numpy.polynomial.legendre.legval`: a Legendre series at one point.
///
/// A free function because [`Legendre::integ`] has to evaluate coefficients
/// that have not been trimmed yet, exactly as `legint` does.
fn legval(x: f64, c: &[f64]) -> f64 {
    match c.len() {
        0 => 0.0,
        1 => c[0],
        2 => c[0] + c[1] * x,
        len => {
            let mut nd = len;
            let mut c0 = c[len - 2];
            let mut c1 = c[len - 1];
            for i in 3..=len {
                let tmp = c0;
                nd -= 1;
                let nd_f = nd as f64;
                c0 = c[len - i] - c1 * ((nd_f - 1.0) / nd_f);
                c1 = tmp + c1 * x * ((2.0 * nd_f - 1.0) / nd_f);
            }
            c0 + c1 * x
        }
    }
}

/// Interpolation metadata for a two-dimensional function, the format's TAB2.
///
/// TAB2 carries no values of its own: it describes how to interpolate between
/// the subrecords that follow it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tabulated2D {
    pub breakpoints: Vec<i32>,
    pub interpolation: Vec<i32>,
}

/// A polynomial in ascending powers, used where the format gives coefficients
/// rather than a table (nu-bar in MF=1, for instance).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Polynomial { coefficients }
    }

    pub fn eval(&self, x: f64) -> f64 {
        // Horner, from the highest power down.
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * x + c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_linear_matches_the_docstring_example() {
        // >>> f = Tabulated1D([0, 10], [4, 5])
        // [4.0, 4.25, 4.5, 4.75, 5.0]
        let f = Tabulated1D::new(vec![0.0, 10.0], vec![4.0, 5.0]);
        let got: Vec<f64> = [0.0, 2.5, 5.0, 7.5, 10.0]
            .iter()
            .map(|&x| f.eval(x))
            .collect();
        for (a, b) in got.iter().zip([4.0, 4.25, 4.5, 4.75, 5.0]) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn clamps_outside_the_tabulated_range() {
        let f = Tabulated1D::new(vec![1.0, 2.0], vec![10.0, 20.0]);
        assert_eq!(f.eval(-5.0), 10.0);
        assert_eq!(f.eval(99.0), 20.0);
    }

    #[test]
    fn histogram_holds_the_left_value() {
        let f =
            Tabulated1D::with_regions(vec![0.0, 1.0, 2.0], vec![3.0, 7.0, 9.0], vec![3], vec![1]);
        assert_eq!(f.eval(0.5), 3.0);
        assert_eq!(f.eval(1.5), 7.0);
    }

    #[test]
    fn integral_of_a_constant_is_the_width() {
        let f = Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![5.0, 5.0, 5.0]);
        let it = f.integral();
        assert_eq!(it.len(), 3);
        assert!((it[0] - 0.0).abs() < 1e-12);
        assert!((it[1] - 5.0).abs() < 1e-12);
        assert!((it[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn legendre_evaluates_the_series() {
        // P_2(x) = (3x^2 - 1)/2, so P_2(0.5) = -0.125 exactly.
        let l = Legendre::new(vec![0.0, 0.0, 1.0]);
        assert_eq!(l.eval(0.5), -0.125);
        assert_eq!(Legendre::new(vec![3.0]).eval(7.0), 3.0);
        assert_eq!(Legendre::new(vec![1.0, 2.0]).eval(0.25), 1.5);
    }

    #[test]
    fn legendre_integrates_as_numpy_does() {
        // The antiderivative of P_2 is (P_3 - P_1)/5, and numpy gives the
        // coefficients and the value at one to the last bit.
        let a = Legendre::new(vec![0.0, 0.0, 1.0]).integ();
        assert_eq!(a.coefficients, vec![0.0, -0.2, 0.0, 0.2]);
        assert_eq!(a.eval(1.0), -5.551115123125783e-17);
    }

    #[test]
    fn legendre_keeps_trailing_zeros() {
        // numpy's constructor does not trim, and the recurrence takes a step
        // per coefficient, so the length is part of the answer.
        let l = Legendre::new(vec![1.0, 0.0, 0.0]);
        assert_eq!(l.coefficients.len(), 3);
        assert_eq!(l.integ().coefficients, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn reads_a_tabulated_function_from_an_ace_array() {
        // One region, three pairs, energies in MeV.
        let xss = vec![
            1.0, // n_regions
            3.0, // breakpoint
            5.0, // log-log
            3.0, // n_pairs
            1.0, 2.0, 3.0, // x, in MeV
            10.0, 20.0, 30.0, // y
        ];
        let f = Tabulated1D::from_ace(&xss, 0, true);
        assert_eq!(f.breakpoints, vec![3]);
        assert_eq!(f.interpolation, vec![5]);
        assert_eq!(f.x, vec![1.0e6, 2.0e6, 3.0e6]);
        assert_eq!(f.y, vec![10.0, 20.0, 30.0]);
        assert_eq!(Tabulated1D::ace_len(&xss, 0), xss.len());

        // Left in MeV when the abscissa is not an energy.
        assert_eq!(Tabulated1D::from_ace(&xss, 0, false).x, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn zero_regions_in_an_ace_array_means_one_linear_region() {
        let xss = vec![0.0, 2.0, 1.0, 2.0, 7.0, 8.0];
        let f = Tabulated1D::from_ace(&xss, 0, false);
        assert_eq!(f.breakpoints, vec![2]);
        assert_eq!(f.interpolation, vec![2]);
        assert_eq!(f.x, vec![1.0, 2.0]);
        assert_eq!(f.y, vec![7.0, 8.0]);
        assert_eq!(Tabulated1D::ace_len(&xss, 0), xss.len());
    }

    #[test]
    fn polynomial_evaluates_in_ascending_powers() {
        // 1 + 2x + 3x^2 at x = 2 -> 17
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.eval(2.0), 17.0);
    }
}
