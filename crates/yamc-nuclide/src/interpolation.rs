//! Shared interpolation helpers used by `yamc-materials`, `yamc`, and
//! downstream consumers of evaluated nuclear data.

/// Linear interpolation on a linear scale.
///
/// Given arrays of `x` and `y` values, find the `y` value at `x_new`.
/// `x` is expected to be sorted ascending. If `x_new` is outside the
/// range of `x`, the first or last `y` is returned (no extrapolation).
/// Empty `x` returns `NaN`.
#[inline]
pub fn interpolate_linear(x: &[f64], y: &[f64], x_new: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    if x.len() == 1 {
        return y[0];
    }
    if x_new <= x[0] {
        return y[0];
    }
    if x_new >= x[x.len() - 1] {
        return y[y.len() - 1];
    }

    let mut low = 0usize;
    let mut high = x.len() - 1;
    while high - low > 1 {
        let mid = (low + high) >> 1;
        if x[mid] <= x_new {
            low = mid;
        } else {
            high = mid;
        }
    }
    let idx = low;
    let x1 = x[idx];
    let x2 = x[idx + 1];
    let y1 = y[idx];
    let y2 = y[idx + 1];
    y1 + (x_new - x1) * (y2 - y1) / (x2 - x1)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_empty_array() {
        let result = interpolate_linear(&[], &[], 1.0);
        assert!(result.is_nan());
    }

    #[test]
    fn test_linear_single_point() {
        let x = [5.0];
        let y = [42.0];
        assert_eq!(interpolate_linear(&x, &y, 0.0), 42.0);
        assert_eq!(interpolate_linear(&x, &y, 5.0), 42.0);
        assert_eq!(interpolate_linear(&x, &y, 100.0), 42.0);
    }

    #[test]
    fn test_linear_below_range() {
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(interpolate_linear(&x, &y, 0.0), 10.0);
        assert_eq!(interpolate_linear(&x, &y, -5.0), 10.0);
    }

    #[test]
    fn test_linear_above_range() {
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(interpolate_linear(&x, &y, 5.0), 30.0);
        assert_eq!(interpolate_linear(&x, &y, 100.0), 30.0);
    }

    #[test]
    fn test_linear_at_exact_bounds() {
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(interpolate_linear(&x, &y, 1.0), 10.0);
        assert_eq!(interpolate_linear(&x, &y, 3.0), 30.0);
    }

    #[test]
    fn test_linear_midpoint() {
        let x = [0.0, 10.0];
        let y = [0.0, 100.0];
        assert!((interpolate_linear(&x, &y, 5.0) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_quarter_point() {
        let x = [0.0, 10.0];
        let y = [0.0, 100.0];
        assert!((interpolate_linear(&x, &y, 2.5) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_multi_segment() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 10.0, 20.0, 30.0, 40.0];
        assert!((interpolate_linear(&x, &y, 0.5) - 5.0).abs() < 1e-12);
        assert!((interpolate_linear(&x, &y, 3.5) - 35.0).abs() < 1e-12);
        assert!((interpolate_linear(&x, &y, 2.0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_non_uniform_spacing() {
        let x = [0.0, 1.0, 10.0];
        let y = [0.0, 1.0, 10.0];
        assert!((interpolate_linear(&x, &y, 5.5) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn test_linear_two_points() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        assert!((interpolate_linear(&x, &y, 0.5) - 0.5).abs() < 1e-12);
    }
}
