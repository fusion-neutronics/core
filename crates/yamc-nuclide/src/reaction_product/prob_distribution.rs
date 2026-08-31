//! Tabulated outgoing-energy probability distribution (with discrete-line support).

use super::TabulatedInterp;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Tabulated probability distribution for outgoing energy
/// Includes CDF values directly from the Arrow data for accurate sampling
/// Supports n_discrete for distributions with discrete energy lines at the beginning
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TabulatedProbability {
    #[serde(rename = "Tabulated")]
    Tabulated {
        x: Vec<f64>, // Outgoing energy values
        p: Vec<f64>, // PDF values
        #[serde(default)]
        c: Vec<f64>, // CDF values (from the Arrow data for accuracy)
        #[serde(default)]
        interp: TabulatedInterp, // Per-bin interpolation (histogram or lin-lin)
        /// Number of discrete energy lines at the beginning of the distribution.
        /// The first n_discrete entries are discrete lines (delta functions).
        /// Discrete lines precede the continuous portion in the CDF
        #[serde(default)]
        n_discrete: usize,
    },
}

impl TabulatedProbability {
    /// Normalize the PDF and CDF by the integral (last CDF value).
    /// Ensures the PDF integrates to 1 and the CDF ends at 1.
    /// Must be called after loading from the Arrow data to ensure proper sampling.
    pub fn normalize(&mut self) {
        match self {
            TabulatedProbability::Tabulated { p, c, .. } => {
                if c.is_empty() || c.len() < 2 {
                    return;
                }

                let integral = *c.last().unwrap();
                if integral <= 0.0 || (integral - 1.0).abs() < 1e-14 {
                    // Already normalized or invalid
                    return;
                }

                // Normalize both PDF and CDF
                for p_val in p.iter_mut() {
                    *p_val /= integral;
                }
                for c_val in c.iter_mut() {
                    *c_val /= integral;
                }
            }
        }
    }

    /// Sample from the tabulated distribution, handling discrete lines properly.
    /// Implements ContinuousTabular sampling with discrete and continuous portions
    ///
    /// The distribution may have n_discrete discrete energy lines at the beginning,
    /// followed by a continuous distribution. The sampling algorithm:
    /// 1. Search the discrete portion first (indices 0 to n_discrete-1)
    /// 2. If not selected, search the continuous portion
    /// 3. For discrete lines, return the exact energy (no interpolation)
    /// 4. For continuous portion, interpolate based on interp type
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            TabulatedProbability::Tabulated {
                x,
                p,
                c,
                interp,
                n_discrete,
            } => {
                if x.is_empty() || c.is_empty() {
                    return 0.0;
                }

                // Single point - return it directly
                if x.len() == 1 {
                    return x[0];
                }

                let n = x.len();
                let r1 = rng.random::<f64>();

                // Search for the bin using CDF inversion
                let mut k = 0;
                let mut c_k = c[0];
                let end = n - 1;

                // Discrete portion: search through first n_discrete entries
                for j in 0..*n_discrete {
                    k = j;
                    c_k = c[k];
                    if r1 < c_k {
                        // Selected a discrete line - return exact energy (no interpolation)
                        return x[k];
                    }
                }

                // Continuous portion: search through rest of distribution
                for j in *n_discrete..end {
                    k = j;
                    let c_k1 = c[k + 1];
                    if r1 < c_k1 {
                        break;
                    }
                    k = j + 1;
                    c_k = c_k1;
                }

                // Ensure bounds
                if k >= n - 1 {
                    k = n - 2;
                }

                let x_k = x[k];
                let p_k = p[k];

                // Interpolate within the continuous bin
                // Only interpolate if k >= n_discrete (i.e., in continuous portion)
                match interp {
                    TabulatedInterp::Histogram => {
                        // Histogram interpolation
                        // Only interpolate if p_k > 0 and in continuous portion
                        if p_k > 0.0 && k >= *n_discrete {
                            x_k + (r1 - c_k) / p_k
                        } else {
                            x_k
                        }
                    }
                    TabulatedInterp::LinLin => {
                        // Linear-linear interpolation
                        if k + 1 >= n {
                            return x_k;
                        }
                        let x_k1 = x[k + 1];
                        let p_k1 = p[k + 1];

                        if x_k == x_k1 {
                            return x_k;
                        }

                        let frac = (p_k1 - p_k) / (x_k1 - x_k);
                        if frac == 0.0 {
                            x_k + (r1 - c_k) / p_k
                        } else {
                            x_k + ((p_k * p_k + 2.0 * frac * (r1 - c_k)).max(0.0).sqrt() - p_k)
                                / frac
                        }
                    }
                }
            }
        }
    }

    /// Get all x (energy) values
    pub fn get_x_values(&self) -> &[f64] {
        match self {
            TabulatedProbability::Tabulated { x, .. } => x,
        }
    }

    /// Get the number of discrete lines at the beginning of the distribution
    pub fn get_n_discrete(&self) -> usize {
        match self {
            TabulatedProbability::Tabulated { n_discrete, .. } => *n_discrete,
        }
    }

    /// Get the first continuous energy value (for interpolation bounds)
    /// This is x[n_discrete], not x[0], when discrete lines are present
    pub fn get_first_continuous_energy(&self) -> f64 {
        match self {
            TabulatedProbability::Tabulated { x, n_discrete, .. } => {
                if *n_discrete < x.len() {
                    x[*n_discrete]
                } else if !x.is_empty() {
                    x[0]
                } else {
                    0.0
                }
            }
        }
    }

    /// Get the last energy value (max E_out)
    pub fn get_last_energy(&self) -> f64 {
        match self {
            TabulatedProbability::Tabulated { x, .. } => x.last().copied().unwrap_or(0.0),
        }
    }

    /// Sample from the distribution, returning (energy, is_from_discrete_line)
    /// This allows the caller to skip incident energy interpolation for discrete lines.
    /// The k >= n_discrete check determines whether to interpolate or return exact energy.
    pub fn sample_with_discrete_info<R: Rng>(&self, rng: &mut R) -> (f64, bool) {
        match self {
            TabulatedProbability::Tabulated {
                x,
                p,
                c,
                interp,
                n_discrete,
            } => {
                if x.is_empty() || c.is_empty() {
                    return (0.0, false);
                }

                // Single point - return it directly
                if x.len() == 1 {
                    return (x[0], *n_discrete > 0);
                }

                let n = x.len();
                let r1 = rng.random::<f64>();

                // Search for the bin using CDF inversion
                let mut k = 0;
                let mut c_k = c[0];
                let end = n - 1;

                // Discrete portion: search through first n_discrete entries
                for j in 0..*n_discrete {
                    k = j;
                    c_k = c[k];
                    if r1 < c_k {
                        // Selected a discrete line - return exact energy (no interpolation)
                        return (x[k], true); // is_discrete = true
                    }
                }

                // Continuous portion: search through rest of distribution
                for j in *n_discrete..end {
                    k = j;
                    let c_k1 = c[k + 1];
                    if r1 < c_k1 {
                        break;
                    }
                    k = j + 1;
                    c_k = c_k1;
                }

                // Ensure bounds
                if k >= n - 1 {
                    k = n - 2;
                }

                let x_k = x[k];
                let p_k = p[k];
                let is_discrete = k < *n_discrete;

                // Interpolate within the continuous bin
                let e_out = match interp {
                    TabulatedInterp::Histogram => {
                        if p_k > 0.0 && k >= *n_discrete {
                            x_k + (r1 - c_k) / p_k
                        } else {
                            x_k
                        }
                    }
                    TabulatedInterp::LinLin => {
                        if k + 1 >= n || k < *n_discrete {
                            x_k
                        } else {
                            let x_k1 = x[k + 1];
                            let p_k1 = p[k + 1];

                            if x_k == x_k1 {
                                x_k
                            } else {
                                let frac = (p_k1 - p_k) / (x_k1 - x_k);
                                if frac == 0.0 {
                                    x_k + (r1 - c_k) / p_k
                                } else {
                                    x_k + ((p_k * p_k + 2.0 * frac * (r1 - c_k)).max(0.0).sqrt()
                                        - p_k)
                                        / frac
                                }
                            }
                        }
                    }
                };

                (e_out, is_discrete)
            }
        }
    }
}
