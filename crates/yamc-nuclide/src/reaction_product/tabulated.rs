//! Tabular probability distributions: `Tabulated` (x/PDF/CDF with histogram or
//! lin-lin interpolation) and the `Tabulated1D` interpolation table, plus the
//! shared trapezoidal `cumulative_from_pdf` helper they build on.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Helper: compute cumulative distribution from PDF (lin-lin, normalized)
fn cumulative_from_pdf(x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(x.len());
    let mut sum = 0.0;
    c.push(0.0);
    for i in 1..x.len() {
        // Trapezoidal rule for lin-lin
        let dx = x[i] - x[i - 1];
        let avg = 0.5 * (p[i] + p[i - 1]);
        sum += avg * dx;
        c.push(sum);
    }
    // Normalize
    if sum > 0.0 {
        for v in &mut c {
            *v /= sum;
        }
    }
    c
}

// ============================================================================
// SHARED TYPES
// ============================================================================

/// Interpolation type for tabular distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TabulatedInterp {
    #[default]
    Histogram,
    LinLin,
}

/// Tabulated probability distribution with x, p (PDF), and c (CDF) values
/// Matches the Tabular distribution structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tabulated {
    pub x: Vec<f64>, // x values (e.g., mu for angle, energy for energy)
    pub p: Vec<f64>, // PDF values
    #[serde(default)]
    pub c: Vec<f64>, // CDF values (from the Arrow data)
    #[serde(default)]
    pub interp: TabulatedInterp, // Interpolation type
}

impl Tabulated {
    /// Normalize the PDF and CDF by the integral (last CDF value).
    /// Ensures the PDF integrates to 1 and the CDF ends at 1.
    /// Must be called after loading from the Arrow data to ensure proper sampling.
    pub fn normalize(&mut self) {
        if self.c.is_empty() || self.c.len() < 2 {
            return;
        }

        let integral = *self.c.last().unwrap();
        if integral <= 0.0 || (integral - 1.0).abs() < 1e-14 {
            // Already normalized or invalid
            return;
        }

        // Normalize both PDF and CDF
        for p_val in &mut self.p {
            *p_val /= integral;
        }
        for c_val in &mut self.c {
            *c_val /= integral;
        }
    }

    /// Sample from the tabular distribution using the algorithm
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        if self.x.is_empty() || self.p.is_empty() {
            return 0.0;
        }

        // Single point distribution - always return that point
        if self.x.len() == 1 {
            return self.x[0];
        }

        // Use CDF directly if available, otherwise compute from PDF
        let cdf = if !self.c.is_empty() && self.c.len() == self.x.len() {
            &self.c
        } else {
            // Fallback: compute CDF from PDF (less accurate)
            return self.sample_from_pdf(rng);
        };

        // Sample value of CDF
        let c_sample = rng.random::<f64>();

        // Find first CDF bin which is above the sampled value
        let n = cdf.len();
        let mut i = 0;
        for j in 0..n - 1 {
            if c_sample <= cdf[j + 1] {
                i = j;
                break;
            }
            i = j + 1;
        }

        // Ensure we don't go out of bounds
        if i >= n - 1 {
            i = n - 2;
        }
        let c_i = cdf[i];

        // Determine bounding PDF values
        let x_i = self.x[i];
        let p_i = self.p[i];

        match self.interp {
            TabulatedInterp::Histogram => {
                // Histogram interpolation
                if p_i > 0.0 {
                    x_i + (c_sample - c_i) / p_i
                } else {
                    x_i
                }
            }
            TabulatedInterp::LinLin => {
                // Linear-linear interpolation
                let x_i1 = self.x[i + 1];
                let p_i1 = self.p[i + 1];

                let m = (p_i1 - p_i) / (x_i1 - x_i);
                if m == 0.0 {
                    x_i + (c_sample - c_i) / p_i
                } else {
                    x_i + ((p_i * p_i + 2.0 * m * (c_sample - c_i)).max(0.0).sqrt() - p_i) / m
                }
            }
        }
    }

    /// Fallback sampling from PDF when CDF not available
    fn sample_from_pdf<R: Rng>(&self, rng: &mut R) -> f64 {
        // Single point distribution - always return that point
        if self.x.len() == 1 {
            return self.x[0];
        }

        // Compute CDF from PDF using trapezoidal integration
        let cdf_p = cumulative_from_pdf(&self.x, &self.p);

        let c_sample = rng.random::<f64>();
        let n = cdf_p.len();

        let mut i = 0;
        for j in 0..n - 1 {
            if c_sample <= cdf_p[j + 1] {
                i = j;
                break;
            }
            i = j + 1;
        }

        if i >= n - 1 {
            i = n - 2;
        }
        let c_i = cdf_p[i];

        let x_i = self.x[i];
        let p_i = self.p[i];

        // Use lin-lin interpolation by default for computed CDF
        let x_i1 = self.x.get(i + 1).copied().unwrap_or(x_i);
        let p_i1 = self.p.get(i + 1).copied().unwrap_or(p_i);

        let m = if (x_i1 - x_i).abs() > 1e-30 {
            (p_i1 - p_i) / (x_i1 - x_i)
        } else {
            0.0
        };

        if m == 0.0 {
            if p_i > 0.0 {
                x_i + (c_sample - c_i) / p_i
            } else {
                x_i
            }
        } else {
            x_i + ((p_i * p_i + 2.0 * m * (c_sample - c_i)).max(0.0).sqrt() - p_i) / m
        }
    }

    pub fn to_cdf(&self) -> Self {
        if self.p.is_empty() || self.x.is_empty() {
            return self.clone();
        }

        // Use proper trapezoidal integration to convert PDF to CDF
        // This accounts for non-uniform spacing in x values
        let cdf_p = cumulative_from_pdf(&self.x, &self.p);

        // Return a CDF where p contains the CDF values (for test compatibility)
        // and c also contains the CDF values (for sampling)
        Tabulated {
            x: self.x.clone(),
            p: cdf_p.clone(),
            c: cdf_p,
            interp: self.interp,
        }
    }
}

/// 1D tabulated function
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tabulated1D {
    #[serde(rename = "Tabulated1D")]
    Tabulated1D {
        x: Vec<f64>,
        y: Vec<f64>,
        breakpoints: Vec<i32>,
        interpolation: Vec<i32>,
    },
}

impl Tabulated1D {
    pub fn evaluate(&self, energy: f64) -> f64 {
        match self {
            Tabulated1D::Tabulated1D {
                x,
                y,
                breakpoints,
                interpolation,
            } => {
                if x.is_empty() || y.is_empty() {
                    return 0.0;
                }

                if energy <= x[0] {
                    return y[0];
                }
                if energy >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }

                let i = x
                    .binary_search_by(|&val| val.partial_cmp(&energy).unwrap())
                    .unwrap_or_else(|i| i.saturating_sub(1));

                if i >= x.len() - 1 {
                    return y[y.len() - 1];
                }

                // Determine interpolation scheme from breakpoints
                let mut interp = interpolation.first().copied().unwrap_or(2);
                if !breakpoints.is_empty() && !interpolation.is_empty() {
                    let n_regions = breakpoints.len().min(interpolation.len());
                    for j in 0..n_regions {
                        let bp = breakpoints[j];
                        let bp0 = if bp > 0 { (bp - 1) as usize } else { 0 };
                        if i < bp0 {
                            interp = interpolation[j];
                            break;
                        }
                    }
                }

                let x0 = x[i];
                let x1 = x[i + 1];
                let y0 = y[i];
                let y1 = y[i + 1];

                match interp {
                    1 => y0, // histogram
                    3 => {
                        // lin-log
                        if x0 > 0.0 && x1 > 0.0 {
                            let r = (energy / x0).ln() / (x1 / x0).ln();
                            y0 + r * (y1 - y0)
                        } else {
                            let r = (energy - x0) / (x1 - x0);
                            y0 + r * (y1 - y0)
                        }
                    }
                    4 => {
                        // log-lin
                        if y0 > 0.0 && y1 > 0.0 {
                            let r = (energy - x0) / (x1 - x0);
                            y0 * (r * (y1 / y0).ln()).exp()
                        } else {
                            let r = (energy - x0) / (x1 - x0);
                            y0 + r * (y1 - y0)
                        }
                    }
                    5 => {
                        // log-log
                        if x0 > 0.0 && x1 > 0.0 && y0 > 0.0 && y1 > 0.0 {
                            let r = (energy / x0).ln() / (x1 / x0).ln();
                            y0 * (r * (y1 / y0).ln()).exp()
                        } else {
                            let r = (energy - x0) / (x1 - x0);
                            y0 + r * (y1 - y0)
                        }
                    }
                    _ => {
                        // lin-lin (default)
                        let r = (energy - x0) / (x1 - x0);
                        y0 + r * (y1 - y0)
                    }
                }
            }
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_from_pdf_is_normalized_and_monotonic() {
        // Flat PDF over [0, 4] -> CDF rises linearly to 1.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let p = [1.0, 1.0, 1.0, 1.0, 1.0];
        let c = cumulative_from_pdf(&x, &p);
        assert_eq!(c.len(), x.len());
        assert_eq!(c[0], 0.0);
        assert!((c.last().unwrap() - 1.0).abs() < 1e-12, "CDF must end at 1");
        // Equal-width flat bins -> evenly spaced cumulative mass.
        for (i, ci) in c.iter().enumerate() {
            assert!((ci - i as f64 / 4.0).abs() < 1e-12);
        }
        // Monotonic non-decreasing.
        assert!(c.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn cumulative_from_pdf_handles_zero_integral() {
        // All-zero PDF -> the un-normalized (all-zero) CDF is returned as-is.
        let x = [0.0, 1.0, 2.0];
        let p = [0.0, 0.0, 0.0];
        let c = cumulative_from_pdf(&x, &p);
        assert_eq!(c, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn tabulated_single_point_samples_that_point() {
        let t = Tabulated {
            x: vec![2.5],
            p: vec![1.0],
            c: vec![],
            interp: TabulatedInterp::Histogram,
        };
        let mut rng = rand::rng();
        assert_eq!(t.sample(&mut rng), 2.5);
    }
}
