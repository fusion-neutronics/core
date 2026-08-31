//! Kalbach-Mann correlated angle-energy sampling.
//!
//! Based on:
//! - Kalbach, C. (1988). "Systematics of continuum angular distributions:
//!   Extensions to higher energies", Phys. Rev. C 37, 2350-2370.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Interpolation {
    Histogram,
    LinLin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalbachMann {
    pub energy: Vec<f64>, // incident energy grid
    pub distributions: Vec<KMTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMTable {
    pub interpolation: Interpolation,
    pub n_discrete: usize,
    pub e_out: Vec<f64>,
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    pub r: Vec<f64>,
    pub a: Vec<f64>,
}

impl KMTable {
    /// Normalize the PDF and CDF by the integral (last CDF value).
    pub fn normalize(&mut self) {
        if self.c.is_empty() || self.c.len() < 2 {
            return;
        }
        let integral = *self.c.last().unwrap();
        if integral <= 0.0 || (integral - 1.0).abs() < 1e-14 {
            return;
        }
        for p_val in &mut self.p {
            *p_val /= integral;
        }
        for c_val in &mut self.c {
            *c_val /= integral;
        }
    }
}

impl KalbachMann {
    /// Sample outgoing energy and angle from Kalbach-Mann distribution.
    pub fn sample<R: Rng>(&self, e_in: f64, rng: &mut R) -> (f64, f64) {
        let n_energy = self.energy.len();
        if n_energy < 2 {
            return (e_in, 2.0 * rng.random::<f64>() - 1.0);
        }

        // Find energy bin and calculate interpolation factor
        let (i, r_interp) = if e_in < self.energy[0] {
            (0, 0.0)
        } else if e_in > self.energy[n_energy - 1] {
            (n_energy - 2, 1.0)
        } else {
            let i = lower_bound_index(&self.energy, e_in);
            let r = (e_in - self.energy[i]) / (self.energy[i + 1] - self.energy[i]);
            (i, r)
        };

        // Stochastically select between bin i and i+1
        let l = if r_interp > rng.random::<f64>() {
            i + 1
        } else {
            i
        };

        // Get distributions for interpolation
        let dist_i = &self.distributions[i];
        let dist_i1 = &self.distributions[i + 1];

        // Get E_out bounds for interpolation
        // E_i_1 = min continuous E_out for bin i, E_i_K = max E_out for bin i
        let n_discrete_i = dist_i.n_discrete;
        let e_i_1 = dist_i.e_out.get(n_discrete_i).copied().unwrap_or(0.0);
        let e_i_k = *dist_i.e_out.last().unwrap_or(&0.0);

        let n_discrete_i1 = dist_i1.n_discrete;
        let e_i1_1 = dist_i1.e_out.get(n_discrete_i1).copied().unwrap_or(0.0);
        let e_i1_k = *dist_i1.e_out.last().unwrap_or(&0.0);

        // Interpolated E_out bounds
        let e_1 = e_i_1 + r_interp * (e_i1_1 - e_i_1);
        let e_k = e_i_k + r_interp * (e_i1_k - e_i_k);

        let dist = &self.distributions[l];
        let n_energy_out = dist.e_out.len();
        let n_discrete = dist.n_discrete;

        // Sample outgoing energy bin from CDF
        let r1 = rng.random::<f64>();
        let mut k = 0;
        let mut c_k = dist.c[0];
        let mut end = n_energy_out - 2;

        // Discrete portion
        for j in 0..n_discrete {
            k = j;
            c_k = dist.c[k];
            if r1 < c_k {
                end = j;
                break;
            }
        }

        // Continuous portion
        let mut c_k1;
        for j in n_discrete..end {
            k = j;
            c_k1 = dist.c[k + 1];
            if r1 < c_k1 {
                break;
            }
            k = j + 1;
            c_k = c_k1;
        }

        let e_l_k = dist.e_out[k];
        let p_l_k = dist.p[k];

        // Sample E_out and determine Kalbach-Mann r,a parameters
        // CRITICAL: r,a must be determined from the PRE-interpolated E_out value
        let (mut e_out, km_r, km_a) = match dist.interpolation {
            Interpolation::Histogram => {
                // Histogram interpolation for energy
                let e_out = if p_l_k > 0.0 && k >= n_discrete {
                    e_l_k + (r1 - c_k) / p_l_k
                } else {
                    e_l_k
                };
                // For histogram: use r[k], a[k] directly (NO interpolation)
                (e_out, dist.r[k], dist.a[k])
            }
            Interpolation::LinLin => {
                // Linear-linear interpolation for energy
                let e_l_k1 = dist.e_out[k + 1];
                let p_l_k1 = dist.p[k + 1];
                let frac = (p_l_k1 - p_l_k) / (e_l_k1 - e_l_k);
                let e_out = if frac == 0.0 {
                    e_l_k + (r1 - c_k) / p_l_k
                } else {
                    e_l_k
                        + ((p_l_k * p_l_k + 2.0 * frac * (r1 - c_k)).max(0.0).sqrt() - p_l_k) / frac
                };
                // For lin-lin: interpolate r,a using PRE-interpolated E_out
                let f = (e_out - e_l_k) / (e_l_k1 - e_l_k);
                let km_r = dist.r[k] + f * (dist.r[k + 1] - dist.r[k]);
                let km_a = dist.a[k] + f * (dist.a[k + 1] - dist.a[k]);
                (e_out, km_r, km_a)
            }
        };

        // Now interpolate E_out between incident energy bins
        // This happens AFTER r,a are determined
        if k >= n_discrete {
            let (e_l_1, e_l_k_max) = if l == i {
                (e_i_1, e_i_k)
            } else {
                (e_i1_1, e_i1_k)
            };
            if e_l_k_max > e_l_1 {
                e_out = e_1 + (e_out - e_l_1) * (e_k - e_1) / (e_l_k_max - e_l_1);
            }
        }

        // Sample angle using Kalbach-Mann parameters
        // if prn > r -> compound (arcsinh), else -> precompound (CDF inversion)
        let mu = if rng.random::<f64>() > km_r {
            // Compound nucleus: sample from sinh distribution using arcsinh
            // T = uniform(-1,1) * sinh(a), mu = arcsinh(T) / a
            if km_a.abs() < 1e-6 {
                2.0 * rng.random::<f64>() - 1.0
            } else {
                let t = (2.0 * rng.random::<f64>() - 1.0) * km_a.sinh();
                (t + (t * t + 1.0).sqrt()).ln() / km_a
            }
        } else {
            // Precompound: CDF inversion of exp(a*mu) distribution
            // mu = ln(r*exp(a) + (1-r)*exp(-a)) / a
            if km_a.abs() < 1e-6 {
                2.0 * rng.random::<f64>() - 1.0
            } else {
                let r = rng.random::<f64>();
                (r * km_a.exp() + (1.0 - r) * (-km_a).exp()).ln() / km_a
            }
        };

        (e_out, mu)
    }
}

fn lower_bound_index(grid: &[f64], value: f64) -> usize {
    if grid.len() < 2 {
        return 0;
    }

    // Binary search: find first index where grid[idx] >= value
    let mut left = 0usize;
    let mut right = grid.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if grid[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Convert to lower-bound index for interpolation
    if left == 0 {
        0
    } else {
        let mut idx = left - 1;
        if idx >= grid.len() - 1 {
            idx = grid.len() - 2;
        }
        idx
    }
}

// Tests should be rewritten to use the new KalbachMann struct and its sample method.
