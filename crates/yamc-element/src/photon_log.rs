//! Log-transform utilities for photon cross section data.
//!
//! Used by the `photon_arrow` module.

/// Threshold for log transform: values below exp(-499) are stored as -900.0
pub(crate) const LOG_THRESHOLD: f64 = -499.0;
pub(crate) const LOG_ZERO: f64 = -900.0;

/// Apply log transform to a cross section value.
/// If sigma > exp(-499), return ln(sigma); otherwise return -900.0.
#[inline]
pub(crate) fn safe_log(value: f64) -> f64 {
    if value > LOG_THRESHOLD.exp() {
        value.ln()
    } else {
        LOG_ZERO
    }
}

/// Apply log transform to an entire vector of cross section values.
pub(crate) fn log_transform_xs(xs: &[f64]) -> Vec<f64> {
    xs.iter().map(|&v| safe_log(v)).collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_log() {
        // Normal positive value
        let v = 1.0;
        assert!((safe_log(v) - 0.0).abs() < 1e-12);

        // Large value
        let v = 100.0;
        assert!((safe_log(v) - 100.0_f64.ln()).abs() < 1e-12);

        // Very small value -> -900
        let v = 0.0;
        assert_eq!(safe_log(v), LOG_ZERO);

        // Negative value -> -900
        let v = -1.0;
        assert_eq!(safe_log(v), LOG_ZERO);
    }

    #[test]
    fn test_log_transform_xs() {
        let xs = vec![1.0, 10.0, 0.0, 100.0];
        let result = log_transform_xs(&xs);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[1] - 10.0_f64.ln()).abs() < 1e-12);
        assert_eq!(result[2], LOG_ZERO);
        assert!((result[3] - 100.0_f64.ln()).abs() < 1e-12);
    }
}
