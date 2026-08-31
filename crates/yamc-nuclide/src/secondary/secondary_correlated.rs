use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Interpolation type for tabular distributions
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Histogram,
    LinLin,
}

/// Tabular distribution for outgoing energy or angle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tabular {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    pub interpolation: Interpolation,
    pub n_discrete: usize, // Number of discrete lines (for outgoing energy)
}

impl Tabular {
    /// Normalize the PDF and CDF by the integral (last CDF value).
    /// Ensures the PDF integrates to 1 and the CDF ends at 1.
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

    /// Sample from the tabular distribution (ACE-style)
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let r = rng.random::<f64>();
        let n = self.x.len();
        let mut k = 0;
        let mut c_k = self.c[0];
        let mut end = n - 2;
        // Discrete portion
        for j in 0..self.n_discrete {
            k = j;
            c_k = self.c[k];
            if r < c_k {
                end = j;
                break;
            }
        }
        // Continuous portion
        for j in self.n_discrete..end {
            k = j;
            let c_k1 = self.c[k + 1];
            if r < c_k1 {
                break;
            }
            k = j + 1;
            c_k = c_k1;
        }
        let x_k = self.x[k];
        let p_k = self.p[k];
        if self.interpolation == Interpolation::Histogram {
            if p_k > 0.0 && k >= self.n_discrete {
                x_k + (r - c_k) / p_k
            } else {
                x_k
            }
        } else {
            // LinLin
            let x_k1 = self.x[k + 1];
            let p_k1 = self.p[k + 1];
            let frac = (p_k1 - p_k) / (x_k1 - x_k);
            if frac == 0.0 {
                x_k + (r - c_k) / p_k
            } else {
                x_k + ((p_k * p_k + 2.0 * frac * (r - c_k)).max(0.0).sqrt() - p_k) / frac
            }
        }
    }
}

/// Correlated angle-energy distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedAngleEnergy {
    pub energy: Vec<f64>, // incident energy grid
    pub distributions: Vec<CorrTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrTable {
    pub interpolation: Interpolation,
    pub n_discrete: usize,
    pub e_out: Vec<f64>,
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    pub angle: Vec<Tabular>, // angular distributions for each outgoing energy
}

impl CorrTable {
    /// Normalize the PDF and CDF by the integral (last CDF value).
    /// Also normalizes all angle distributions.
    pub fn normalize(&mut self) {
        if self.c.is_empty() || self.c.len() < 2 {
            return;
        }
        let integral = *self.c.last().unwrap();
        if integral > 0.0 && (integral - 1.0).abs() >= 1e-14 {
            for p_val in &mut self.p {
                *p_val /= integral;
            }
            for c_val in &mut self.c {
                *c_val /= integral;
            }
        }
        // Normalize all angle distributions
        for ang in &mut self.angle {
            ang.normalize();
        }
    }
}

impl CorrelatedAngleEnergy {
    /// Sample outgoing energy and angle given incident energy
    pub fn sample<R: Rng>(&self, e_in: f64, rng: &mut R) -> (f64, f64) {
        let n_energy_in = self.energy.len();
        if n_energy_in < 2 {
            return (e_in, 2.0 * rng.random::<f64>() - 1.0);
        }

        // Find energy bin and calculate interpolation factor
        let (i, r) = if e_in < self.energy[0] {
            (0, 0.0)
        } else if e_in > self.energy[n_energy_in - 1] {
            (n_energy_in - 2, 1.0)
        } else {
            let i = lower_bound_index(&self.energy, e_in);
            let r = if i + 1 < n_energy_in {
                (e_in - self.energy[i]) / (self.energy[i + 1] - self.energy[i])
            } else {
                0.0
            };
            (i, r)
        };

        // Interpolation for energy E_1 and E_K
        let dist_i = &self.distributions[i];
        let n_energy_out_i = dist_i.e_out.len();
        let n_discrete_i = dist_i.n_discrete;
        let e_i_1 = if n_discrete_i < n_energy_out_i {
            dist_i.e_out[n_discrete_i]
        } else {
            dist_i.e_out[0]
        };
        let e_i_k = dist_i.e_out[n_energy_out_i - 1];

        let dist_i1 = &self.distributions[i + 1];
        let n_energy_out_i1 = dist_i1.e_out.len();
        let n_discrete_i1 = dist_i1.n_discrete;
        let e_i1_1 = if n_discrete_i1 < n_energy_out_i1 {
            dist_i1.e_out[n_discrete_i1]
        } else {
            dist_i1.e_out[0]
        };
        let e_i1_k = dist_i1.e_out[n_energy_out_i1 - 1];

        let e_1 = e_i_1 + r * (e_i1_1 - e_i_1);
        let e_k = e_i_k + r * (e_i1_k - e_i_k);

        // Sample between the ith and [i+1]th bin (stochastic interpolation)
        let l = if r > rng.random::<f64>() { i + 1 } else { i };
        let dist = &self.distributions[l];
        let n_energy_out = dist.e_out.len();
        let n_discrete = dist.n_discrete;

        // Sample outgoing energy bin using CDF
        let r1 = rng.random::<f64>();
        let mut c_k = dist.c[0];
        let mut k = 0;
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
        let mut c_k1 = 0.0;
        for j in n_discrete..end {
            k = j;
            c_k1 = dist.c[k + 1];
            if r1 < c_k1 {
                break;
            }
            k = j + 1;
            c_k = c_k1;
        }

        // Sample outgoing energy from bin k
        let e_l_k = dist.e_out[k];
        let p_l_k = dist.p[k];
        let mut e_out = if dist.interpolation == Interpolation::Histogram {
            // Histogram interpolation
            if p_l_k > 0.0 && k >= n_discrete {
                e_l_k + (r1 - c_k) / p_l_k
            } else {
                e_l_k
            }
        } else {
            // Linear-linear interpolation
            if k + 1 < n_energy_out {
                let e_l_k1 = dist.e_out[k + 1];
                let p_l_k1 = dist.p[k + 1];
                let frac = (p_l_k1 - p_l_k) / (e_l_k1 - e_l_k);
                if frac == 0.0 {
                    e_l_k + (r1 - c_k) / p_l_k
                } else {
                    e_l_k
                        + ((p_l_k * p_l_k + 2.0 * frac * (r1 - c_k)).max(0.0).sqrt() - p_l_k) / frac
                }
            } else {
                e_l_k
            }
        };

        // Now interpolate between incident energy bins i and i+1
        if k >= n_discrete {
            let (e_ref_1, e_ref_k) = if l == i {
                (e_i_1, e_i_k)
            } else {
                (e_i1_1, e_i1_k)
            };
            if (e_ref_k - e_ref_1).abs() > 1e-30 {
                e_out = e_1 + (e_out - e_ref_1) * (e_k - e_1) / (e_ref_k - e_ref_1);
            }
        }

        // Find correlated angular distribution for closest outgoing energy bin
        // For discrete lines or histogram interpolation: always use angle[k]
        // For lin-lin continuous: use closer of k or k+1 based on CDF position
        let mu = if !dist.angle.is_empty() {
            let angle_idx = if k < n_discrete
                || dist.interpolation == Interpolation::Histogram
                || (c_k1 > 0.0 && r1 - c_k < c_k1 - r1)
            {
                k.min(dist.angle.len() - 1)
            } else {
                (k + 1).min(dist.angle.len() - 1)
            };
            dist.angle[angle_idx].sample(rng)
        } else {
            // No angular distribution, sample isotropic
            2.0 * rng.random::<f64>() - 1.0
        };

        (e_out, mu)
    }
}

/// Find the lower bound index for interpolation (binary search)
fn lower_bound_index(grid: &[f64], value: f64) -> usize {
    if grid.len() < 2 {
        return 0;
    }

    // Binary search: find first index where grid[idx] >= value (std::lower_bound)
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
