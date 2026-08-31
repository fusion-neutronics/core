//! Energy-dependent angular (scattering-cosine) distribution.

use super::{Tabulated, TabulatedInterp};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Per-bracket interpolation flag: histogram (`mu = x_j + (xi - c_j)/p_j`).
/// Mirrors `yamc_physics::gpu::flat::elastic_mu_cm::ANGLE_INTERP_HISTOGRAM`.
pub const ANGLE_INTERP_HISTOGRAM: u32 = 0;
/// Per-bracket interpolation flag: linear-linear (quadratic CDF inversion).
/// Mirrors `yamc_physics::gpu::flat::elastic_mu_cm::ANGLE_INTERP_LINLIN`.
pub const ANGLE_INTERP_LINLIN: u32 = 1;

/// Angular distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleDistribution {
    pub energy: Vec<f64>,
    pub mu: Vec<Tabulated>,
}

/// Flat (slice-of-`f64`) form of an [`AngleDistribution`], laid out for the
/// shared GPU/CPU elastic-cosine sampler
/// `yamc_physics::gpu::flat::elastic_mu_cm::sample_elastic_mu_cm` (issue #111).
///
/// The CPU transport and the GPU kernel both sample from the *same* tabulated
/// data through that one function; building this flat form on the CPU (cached
/// per nuclide, see [`ElasticFlatCache`]) lets the production elastic scatter
/// call the identical sampler the GPU twin uses, so the two paths cannot drift.
/// Full resolution -- no stride-subsampling (the CPU is the reference); the GPU
/// host extraction applies its own fixed-cap subsampling (issue #104).
#[derive(Debug, Default, Clone)]
pub struct ElasticAngleFlat {
    /// Incident-energy grid, length `n_ae`.
    pub energy_grid: Vec<f64>,
    /// Number of (mu, cdf) points at each incident energy, length `n_ae`.
    pub n_mu: Vec<u32>,
    /// Per-incident-energy interpolation flag, length `n_ae`.
    pub interp: Vec<u32>,
    /// Variable-length mu / cdf / pdf tables: row `i` occupies
    /// `mu_offset[i] .. mu_offset[i] + n_mu[i]`, packed back-to-back
    /// with no padding.
    pub mu: Vec<f64>,
    pub cdf: Vec<f64>,
    pub pdf: Vec<f64>,
    /// Start index of each incident-energy row in `mu`/`cdf`/`pdf`,
    /// length `n_ae` (CSR-style; row length is `n_mu[i]`).
    pub mu_offset: Vec<u32>,
}

impl ElasticAngleFlat {
    /// An empty table; the sampler falls back to isotropic (`1 - 2*xi`).
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Trapezoidal CDF of `p` over `x` (fallback when the Arrow data carried no
/// pre-computed CDF). Mirrors the GPU host extraction's `trapezoidal_cdf`.
fn trapezoidal_cdf(x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        out[i] = out[i - 1] + 0.5 * (p[i] + p[i - 1]) * dx;
    }
    out
}

impl AngleDistribution {
    /// Flatten into the [`ElasticAngleFlat`] layout consumed by the shared
    /// elastic-cosine sampler. The CDF is renormalized to end at 1.0 and the
    /// PDF scaled by the same factor, byte-for-byte matching the GPU host
    /// extraction so both backends sample the identical distribution.
    pub fn to_elastic_flat(&self) -> ElasticAngleFlat {
        let n_ae = self.energy.len();
        if n_ae == 0 || self.mu.len() != n_ae {
            return ElasticAngleFlat::empty();
        }
        // CSR row offsets: row `i` occupies `n_mu[i]` points packed
        // back-to-back (no fixed stride / padding).
        let mut mu_offset = vec![0u32; n_ae];
        let mut total = 0usize;
        for (i, tab) in self.mu.iter().enumerate() {
            mu_offset[i] = total as u32;
            total += tab.x.len();
        }
        if total == 0 {
            return ElasticAngleFlat::empty();
        }
        let mut out = ElasticAngleFlat {
            energy_grid: self.energy.clone(),
            n_mu: vec![0u32; n_ae],
            interp: vec![ANGLE_INTERP_HISTOGRAM; n_ae],
            mu: vec![0.0; total],
            cdf: vec![0.0; total],
            pdf: vec![0.0; total],
            mu_offset,
        };
        for (i, tab) in self.mu.iter().enumerate() {
            let m = tab.x.len();
            if m == 0 {
                continue;
            }
            out.n_mu[i] = m as u32;
            out.interp[i] = match tab.interp {
                TabulatedInterp::Histogram => ANGLE_INTERP_HISTOGRAM,
                TabulatedInterp::LinLin => ANGLE_INTERP_LINLIN,
            };
            let cdf_owned;
            let cdf_slice: &[f64] = if tab.c.len() == m {
                &tab.c
            } else {
                cdf_owned = trapezoidal_cdf(&tab.x, &tab.p);
                &cdf_owned
            };
            let cdf_max = cdf_slice.last().copied().unwrap_or(0.0);
            let pdf_scale = if cdf_max > 0.0 { 1.0 / cdf_max } else { 1.0 };
            let off = out.mu_offset[i] as usize;
            #[allow(clippy::needless_range_loop)] // parallel offset writes into mu/cdf/pdf
            for j in 0..m {
                out.mu[off + j] = tab.x[j];
                out.cdf[off + j] = if cdf_max > 0.0 {
                    cdf_slice[j] / cdf_max
                } else {
                    j as f64 / (m - 1).max(1) as f64
                };
                out.pdf[off + j] = tab.p.get(j).copied().unwrap_or(0.0) * pdf_scale;
            }
        }
        out
    }
}

/// Per-nuclide cache of the flattened elastic angular table, built lazily on
/// the first elastic collision and shared (read-only) across transport
/// threads. `clone()` resets the cache (it is rebuilt on demand), and it is
/// skipped by serde, so adding it to [`crate::nuclide::Nuclide`] keeps that
/// type's `Clone`/`Serialize`/`Deserialize` derives intact.
#[derive(Debug, Default)]
pub struct ElasticFlatCache(std::sync::OnceLock<ElasticAngleFlat>);

impl Clone for ElasticFlatCache {
    fn clone(&self) -> Self {
        Self(std::sync::OnceLock::new())
    }
}

impl ElasticFlatCache {
    /// Return the cached flat table, building it from `angle` (or an empty
    /// table when `angle` is `None`) on first access. A nuclide's elastic
    /// angular distribution is fixed, so the first build is authoritative.
    pub fn get_or_build(&self, angle: Option<&AngleDistribution>) -> &ElasticAngleFlat {
        self.0.get_or_init(|| match angle {
            Some(a) => a.to_elastic_flat(),
            None => ElasticAngleFlat::empty(),
        })
    }
}

/// Per-nuclide cache of flattened DISCRETE-inelastic-level angular tables,
/// keyed by MT (issue #111 sub-step 3). Unlike the single elastic distribution,
/// a nuclide has many discrete levels (MT 51-90), so this maps `mt -> flat`,
/// each built lazily on the first collision of that level and shared read-only
/// (`Arc`) across transport threads. The flat reuses [`ElasticAngleFlat`]
/// because the GPU kernel samples the discrete-level cosine with the very same
/// `elastic_mu_cm` sampler, so routing the CPU through it keeps the two
/// backends bit-identical. Reads take a shared lock (lock-free contention after
/// each level's first build); `clone()` resets the cache; skipped by serde.
#[derive(Debug, Default)]
pub struct InelasticAngleFlatCache(
    std::sync::RwLock<std::collections::HashMap<i32, std::sync::Arc<ElasticAngleFlat>>>,
);

impl Clone for InelasticAngleFlatCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl InelasticAngleFlatCache {
    /// Return the cached flat table for level `mt`, building it from `angle`
    /// (an empty/isotropic table when `None`) on first access.
    pub fn get_or_build(
        &self,
        mt: i32,
        angle: Option<&AngleDistribution>,
    ) -> std::sync::Arc<ElasticAngleFlat> {
        if let Some(flat) = self.0.read().unwrap_or_else(|p| p.into_inner()).get(&mt) {
            return flat.clone();
        }
        let mut w = self.0.write().unwrap_or_else(|p| p.into_inner());
        w.entry(mt)
            .or_insert_with(|| {
                std::sync::Arc::new(match angle {
                    Some(a) => a.to_elastic_flat(),
                    None => ElasticAngleFlat::empty(),
                })
            })
            .clone()
    }
}

impl AngleDistribution {
    /// Sample scattering cosine from angular distribution
    /// Follows the AngleDistribution::sample() approach with stochastic interpolation
    pub fn sample<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> f64 {
        if self.energy.is_empty() || self.mu.is_empty() {
            // No distribution - return isotropic
            return 2.0 * rng.random::<f64>() - 1.0;
        }

        let n = self.energy.len();

        // Find energy bin and calculate interpolation factor
        let (i, r) = if incoming_energy < self.energy[0] {
            (0, 0.0)
        } else if incoming_energy > self.energy[n - 1] {
            (n - 2, 1.0)
        } else {
            let idx = self.find_energy_index(incoming_energy);
            let interp = if idx + 1 < n && self.energy[idx + 1] > self.energy[idx] {
                (incoming_energy - self.energy[idx]) / (self.energy[idx + 1] - self.energy[idx])
            } else {
                0.0
            };
            (idx, interp)
        };

        // Stochastic interpolation: sample between the ith and (i+1)th bin
        let bin = if r > rng.random::<f64>() { i + 1 } else { i };

        // Ensure bin is within bounds
        let bin = bin.min(self.mu.len() - 1);

        // Sample from the selected distribution
        // The distribution uses the CDF directly if available (from the Arrow data)
        let mu_sample = self.mu[bin].sample(rng);

        // Make sure mu is in range [-1, 1]
        let mu_clamped = if mu_sample.abs() > 1.0 {
            1.0_f64.copysign(mu_sample)
        } else {
            mu_sample
        };

        #[cfg(feature = "debug_sampling")]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static MU_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
            let count = MU_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
            if count < 100 || count.is_multiple_of(10000) {
                eprintln!(
                    "[DEBUG_SAMPLING] mu_cm: E_in={:.4e} eV, bin={}, mu={:+.6}",
                    incoming_energy, bin, mu_clamped
                );
            }
        }

        mu_clamped
    }

    fn find_energy_index(&self, energy: f64) -> usize {
        // Find lower bound index for interpolation
        // Returns index i such that self.energy[i] <= energy < self.energy[i+1]

        if self.energy.is_empty() {
            return 0;
        }

        // Special case: if energy equals first grid point, return 0
        // return 0`)
        if energy == self.energy[0] {
            return 0;
        }

        // Find first index where self.energy[i] >= energy (like std::lower_bound)
        let mut left = 0;
        let mut right = self.energy.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.energy[mid] < energy {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // left now points to first element >= energy; subtract 1 to get the lower bound
        if left > 0 {
            let idx = left - 1;
            // Clamp to [0, len-2] for valid interpolation range
            idx.min(self.energy.len().saturating_sub(2))
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_elastic_flat_layout_and_renormalization() {
        // Two incident energies; the second mu-row is unnormalized (CDF ends at
        // 1.075) so we can check the renormalization to 1.0.
        let mut c1 = vec![0.0, 1.0];
        let row0 = Tabulated {
            x: vec![-1.0, 1.0],
            p: vec![0.5, 0.5],
            c: std::mem::take(&mut c1),
            interp: TabulatedInterp::Histogram,
        };
        let row1 = Tabulated {
            x: vec![-1.0, 0.0, 1.0],
            p: vec![0.2, 0.4, 1.6],
            c: vec![0.0, 0.3, 1.075],
            interp: TabulatedInterp::LinLin,
        };
        let angle = AngleDistribution {
            energy: vec![1.0e3, 1.0e7],
            mu: vec![row0, row1],
        };
        let flat = angle.to_elastic_flat();

        assert_eq!(flat.energy_grid, vec![1.0e3, 1.0e7]);
        assert_eq!(flat.n_mu, vec![2, 3]);
        assert_eq!(
            flat.interp,
            vec![ANGLE_INTERP_HISTOGRAM, ANGLE_INTERP_LINLIN]
        );
        assert_eq!(flat.mu_offset, vec![0, 2]); // tight CSR: row1 starts after row0's 2 points

        // Row 0 (already normalized): CDF ends at 1.0, mu copied verbatim.
        assert_eq!(flat.mu[0], -1.0);
        assert_eq!(flat.mu[1], 1.0);
        assert!((flat.cdf[1] - 1.0).abs() < 1e-12);

        // Row 1: CDF renormalized so the last point is 1.0 (1.075 -> 1.0),
        // PDF scaled by the same 1/1.075 factor.
        let off = flat.mu_offset[1] as usize;
        assert!((flat.cdf[off + 2] - 1.0).abs() < 1e-12);
        assert!((flat.cdf[off + 1] - 0.3 / 1.075).abs() < 1e-12);
        assert!((flat.pdf[off + 2] - 1.6 / 1.075).abs() < 1e-12);
    }

    #[test]
    fn empty_angle_distribution_flattens_to_empty() {
        let empty = AngleDistribution {
            energy: vec![],
            mu: vec![],
        };
        let flat = empty.to_elastic_flat();
        assert!(flat.mu_offset.is_empty());
        assert!(flat.energy_grid.is_empty());

        // The cache builds an empty table from `None`.
        let cache = ElasticFlatCache::default();
        assert!(cache.get_or_build(None).mu_offset.is_empty());
    }
}
