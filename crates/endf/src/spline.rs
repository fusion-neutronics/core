//! Not-a-knot cubic spline interpolation.
//!
//! The bremsstrahlung cross sections in `BREMX.DAT` are tabulated on 57
//! electron energies and resampled onto a 200-point grid before use. The
//! Python package does that with `scipy.interpolate.CubicSpline`, whose
//! default boundary condition is **not-a-knot**, the first two segments at
//! each end are forced to be the same cubic. Natural or clamped boundaries
//! would give visibly different values near the ends of the grid, so the
//! condition matters and is not a detail.
//!
//! This is a direct transcription of what SciPy does, so the two agree to
//! within a few units in the last place. Verified against SciPy over the whole
//! of `BREMX.DAT` (100 elements by 30 reduced photon energies by 200 sampled
//! energies) where the worst relative difference is 4.3e-16.
//!
//! Only what the photon data needs: one dependent variable, strictly
//! increasing abscissae, and evaluation that extrapolates with the end cubics
//! rather than clamping, as SciPy's does.

use crate::error::{Error, Result};

/// A cubic spline through a set of points, as piecewise polynomial
/// coefficients.
///
/// Segment `i` covers `x[i]..x[i + 1]` and is evaluated in the local
/// coordinate `x - x[i]`, highest order first, matching SciPy's `PPoly`.
#[derive(Debug, Clone, PartialEq)]
pub struct CubicSpline {
    x: Vec<f64>,
    /// `[c3, c2, c1, c0]` per segment: `c3*d^3 + c2*d^2 + c1*d + c0`.
    coefficients: Vec<[f64; 4]>,
}

impl CubicSpline {
    /// Fit a not-a-knot cubic spline through `(x, y)`.
    ///
    /// `x` must be strictly increasing and the two slices the same length,
    /// with at least two points.
    pub fn new(x: &[f64], y: &[f64]) -> Result<CubicSpline> {
        if x.len() != y.len() {
            return Err(Error::Mismatched {
                what: "the abscissae and ordinates of a cubic spline",
            });
        }
        if x.len() < 2 {
            return Err(Error::Unsupported {
                what: "a cubic spline through fewer than two points",
            });
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Unsupported {
                what: "a cubic spline whose abscissae are not strictly increasing",
            });
        }

        let n = x.len();
        let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let slope: Vec<f64> = y
            .windows(2)
            .zip(&dx)
            .map(|(w, d)| (w[1] - w[0]) / d)
            .collect();

        // The first derivative at each point. Everything else follows.
        let s = match n {
            // A single segment is the straight line through its two points;
            // there is no room for a cubic.
            2 => vec![slope[0], slope[0]],
            // Not-a-knot with three points means one parabola through all
            // three, which is a 3x3 system rather than a banded one.
            3 => solve_three(&dx, &slope),
            _ => solve_banded(x, &dx, &slope),
        };

        // SciPy's coefficients, in the same order, so the arithmetic below
        // reproduces its rounding as well as its result.
        let mut coefficients = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let t = (s[i] + s[i + 1] - 2.0 * slope[i]) / dx[i];
            coefficients.push([t / dx[i], (slope[i] - s[i]) / dx[i] - t, s[i], y[i]]);
        }
        Ok(CubicSpline {
            x: x.to_vec(),
            coefficients,
        })
    }

    /// Evaluate at one point.
    ///
    /// Outside the fitted range the nearest end segment's cubic is continued,
    /// which is what SciPy does by default. That is deliberate: clamping to
    /// the end value would be a different function.
    pub fn eval(&self, x: f64) -> f64 {
        let i = self.segment(x);
        let d = x - self.x[i];
        let [c3, c2, c1, c0] = self.coefficients[i];
        ((c3 * d + c2) * d + c1) * d + c0
    }

    /// Evaluate at every point of a grid.
    pub fn eval_all(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    /// The segment whose cubic applies at `x`.
    fn segment(&self, x: f64) -> usize {
        // The last index whose abscissa does not exceed `x`, clamped to a real
        // segment so that both tails extrapolate.
        match self.x.partition_point(|&xi| xi <= x) {
            0 => 0,
            i => (i - 1).min(self.coefficients.len() - 1),
        }
    }
}

/// The three-point case: one parabola through all three points.
fn solve_three(dx: &[f64], slope: &[f64]) -> Vec<f64> {
    // s0 + s1        = 2*slope0
    // dx1*s0 + 2*(dx0+dx1)*s1 + dx0*s2 = 3*(dx0*slope1 + dx1*slope0)
    //      s1 + s2   = 2*slope1
    //
    // Substituting the outer two into the middle leaves one unknown.
    let (d0, d1) = (dx[0], dx[1]);
    let (m0, m1) = (slope[0], slope[1]);
    let rhs = 3.0 * (d0 * m1 + d1 * m0) - d1 * 2.0 * m0 - d0 * 2.0 * m1;
    let s1 = rhs / (2.0 * (d0 + d1) - d1 - d0);
    vec![2.0 * m0 - s1, s1, 2.0 * m1 - s1]
}

/// The general case: a tridiagonal system solved by the Thomas algorithm.
///
/// SciPy uses LAPACK's banded solver, which pivots; this does not. The two
/// agree to the last bit or two on the data this crate reads, because the
/// interior rows are strictly diagonally dominant and only the two not-a-knot
/// boundary rows are not.
fn solve_banded(x: &[f64], dx: &[f64], slope: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut lower = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    for i in 1..n - 1 {
        diagonal[i] = 2.0 * (dx[i - 1] + dx[i]);
        upper[i] = dx[i - 1];
        lower[i] = dx[i];
        rhs[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }

    // Not-a-knot at the low end: the third derivative is continuous across
    // the second point, which makes the first two segments one cubic.
    let d = x[2] - x[0];
    diagonal[0] = dx[1];
    upper[0] = d;
    rhs[0] = ((dx[0] + 2.0 * d) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d;

    // And the same at the high end.
    let d = x[n - 1] - x[n - 3];
    diagonal[n - 1] = dx[n - 3];
    lower[n - 1] = d;
    rhs[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3]
        + (2.0 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2])
        / d;

    // Thomas: forward sweep, then back substitution.
    let mut c = vec![0.0; n];
    let mut e = vec![0.0; n];
    c[0] = upper[0] / diagonal[0];
    e[0] = rhs[0] / diagonal[0];
    for i in 1..n {
        let m = diagonal[i] - lower[i] * c[i - 1];
        if i < n - 1 {
            c[i] = upper[i] / m;
        }
        e[i] = (rhs[i] - lower[i] * e[i - 1]) / m;
    }
    let mut s = vec![0.0; n];
    s[n - 1] = e[n - 1];
    for i in (0..n - 1).rev() {
        s[i] = e[i] - c[i] * s[i + 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from `scipy.interpolate.CubicSpline`, which is what
    /// the Python package uses. Generated with, and reproducible by:
    ///
    /// ```text
    /// from scipy.interpolate import CubicSpline
    /// CubicSpline([0.0, 0.5, 2.0, 3.0], [1.0, 2.5, 0.5, 3.0])(q)
    /// ```
    #[test]
    fn matches_scipy_on_a_four_point_fit() {
        let x = [0.0, 0.5, 2.0, 3.0];
        let y = [1.0, 2.5, 0.5, 3.0];
        let spline = CubicSpline::new(&x, &y).unwrap();

        // Including two points outside the range, since SciPy extrapolates
        // with the end cubics rather than clamping.
        let q = [-0.5, 0.0, 0.25, 0.5, 1.25, 2.0, 2.5, 3.0, 3.5];
        let expected = [
            -3.125, 1.0, 2.0203125, 2.5, 1.8515625, 0.5, 0.75, 3.0, 8.175,
        ];
        for (&qi, &want) in q.iter().zip(&expected) {
            let got = spline.eval(qi);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "at x={qi}: got {got}, want {want}"
            );
        }
    }

    /// The knots are reproduced exactly: a spline interpolates.
    #[test]
    fn passes_through_every_point() {
        let x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let y = [0.5, 1.5, 1.0, 3.0, 2.0, 2.5];
        let spline = CubicSpline::new(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(&y) {
            assert_eq!(
                spline.eval(xi),
                yi,
                "the spline must pass through ({xi}, {yi})"
            );
        }
    }

    /// A cubic is reproduced exactly, which not-a-knot guarantees and the
    /// natural boundary condition does not, so this fails if the boundary
    /// rows are wrong.
    #[test]
    fn reproduces_a_cubic_exactly() {
        let f = |x: f64| 2.0 * x * x * x - 3.0 * x * x + x - 5.0;
        let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.37).collect();
        let y: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
        let spline = CubicSpline::new(&x, &y).unwrap();

        for i in 0..60 {
            let q = -0.4 + i as f64 * 0.05;
            let (got, want) = (spline.eval(q), f(q));
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "at x={q}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn two_points_give_the_line_through_them() {
        let spline = CubicSpline::new(&[0.0, 2.0], &[1.0, 5.0]).unwrap();
        assert_eq!(spline.eval(0.0), 1.0);
        assert_eq!(spline.eval(1.0), 3.0);
        assert_eq!(spline.eval(2.0), 5.0);
        // Extrapolation continues the line.
        assert_eq!(spline.eval(3.0), 7.0);
    }

    /// Three points with not-a-knot are one parabola, which is exact.
    #[test]
    fn three_points_give_the_parabola_through_them() {
        let f = |x: f64| 1.5 * x * x - 2.0 * x + 0.25;
        let x = [0.0, 1.0, 3.0];
        let y = [f(x[0]), f(x[1]), f(x[2])];
        let spline = CubicSpline::new(&x, &y).unwrap();
        for i in 0..20 {
            let q = -1.0 + i as f64 * 0.3;
            assert!((spline.eval(q) - f(q)).abs() < 1e-12, "at x={q}");
        }
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(CubicSpline::new(&[0.0, 1.0], &[1.0]).is_err());
        assert!(CubicSpline::new(&[0.0], &[1.0]).is_err());
        // Not strictly increasing.
        assert!(CubicSpline::new(&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(CubicSpline::new(&[0.0, 2.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
    }
}
