//! Utility functions for yamc

#[doc(hidden)]
pub use yamc_nuclide::interpolate_linear;

#[doc(hidden)]
/// Log-log interpolation (hidden from docs)
///
/// Given arrays of x and y values, interpolate on a log-log scale to find the y value at x_new.
/// If x_new is outside the range of x, returns the first or last y value.
/// All x and y values must be positive.
pub fn interpolate_log_log(x: &[f64], y: &[f64], x_new: f64) -> f64 {
    // Edge cases / validation
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

    // Binary search for interval
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
    let log_x1 = x1.ln();
    let log_x2 = x2.ln();
    let log_y1 = y1.ln();
    let log_y2 = y2.ln();
    let log_x_new = x_new.ln();
    let log_y_new = log_y1 + (log_x_new - log_x1) * (log_y2 - log_y1) / (log_x2 - log_x1);
    log_y_new.exp()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- interpolate_log_log tests ----

    #[test]
    fn test_log_log_empty_array() {
        let result = interpolate_log_log(&[], &[], 1.0);
        assert!(result.is_nan());
    }

    #[test]
    fn test_log_log_single_point() {
        let x = [5.0];
        let y = [42.0];
        assert_eq!(interpolate_log_log(&x, &y, 0.1), 42.0);
        assert_eq!(interpolate_log_log(&x, &y, 5.0), 42.0);
        assert_eq!(interpolate_log_log(&x, &y, 100.0), 42.0);
    }

    #[test]
    fn test_log_log_below_range() {
        let x = [1.0, 10.0, 100.0];
        let y = [1.0, 10.0, 100.0];
        assert_eq!(interpolate_log_log(&x, &y, 0.5), 1.0);
    }

    #[test]
    fn test_log_log_above_range() {
        let x = [1.0, 10.0, 100.0];
        let y = [1.0, 10.0, 100.0];
        assert_eq!(interpolate_log_log(&x, &y, 200.0), 100.0);
    }

    #[test]
    fn test_log_log_at_bounds() {
        let x = [1.0, 10.0, 100.0];
        let y = [1.0, 10.0, 100.0];
        assert_eq!(interpolate_log_log(&x, &y, 1.0), 1.0);
        assert_eq!(interpolate_log_log(&x, &y, 100.0), 100.0);
    }

    #[test]
    fn test_log_log_power_law() {
        // y = x^2 is a perfect straight line on a log-log scale
        let x = [1.0, 10.0];
        let y = [1.0, 100.0]; // y = x^2
                              // At x = 3.0, y should be 9.0
        let result = interpolate_log_log(&x, &y, 3.0);
        assert!((result - 9.0).abs() < 1e-10, "Expected 9.0, got {result}");
    }

    #[test]
    fn test_log_log_sqrt_relation() {
        // y = sqrt(x) => y = x^0.5, also linear on log-log
        let x = [1.0, 100.0];
        let y = [1.0, 10.0]; // y = x^0.5
        let result = interpolate_log_log(&x, &y, 25.0);
        assert!((result - 5.0).abs() < 1e-10, "Expected 5.0, got {result}");
    }

    #[test]
    fn test_log_log_multi_segment() {
        // y = x^2 across three segments
        let x = [1.0, 10.0, 100.0, 1000.0];
        let y = [1.0, 100.0, 10000.0, 1000000.0];
        // At x = 50, y should be 2500
        let result = interpolate_log_log(&x, &y, 50.0);
        assert!(
            (result - 2500.0).abs() < 1e-6,
            "Expected 2500.0, got {result}"
        );
    }

    #[test]
    fn test_log_log_two_points() {
        let x = [1.0, 100.0];
        let y = [1.0, 100.0]; // y = x (slope 1 on log-log)
        let result = interpolate_log_log(&x, &y, 10.0);
        assert!((result - 10.0).abs() < 1e-10, "Expected 10.0, got {result}");
    }
}
