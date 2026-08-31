//! URR (Unresolved Resonance Range) Probability Tables
//!
//! Implements probability table sampling for the unresolved resonance range,
//! as described in the ENDF-6 Formats Manual (BNL-90365-2009 Rev.2, Section 2.3).
//! See also: Levitt, Nucl. Sci. Eng. 49, 450-457, 1972.

use serde::{Deserialize, Serialize};

/// Derive an independent per-nuclide URR probability-table random in `[0, 1)`
/// from a per-collision base seed and the nuclide's `ZA` identifier
/// (`Z * 1000 + A`).
///
/// In the unresolved resonance range each nuclide's cross section is an
/// independent draw from its own probability table, because different
/// isotopes' resonance structures are statistically uncorrelated. A single
/// shared random across a material's nuclides forces their sampled cross
/// sections to move together, inflating the variance of the macroscopic
/// `Sigma_t` and over-transmitting neutrons through multi-isotope materials
/// (issue #204).
///
/// This mirrors OpenMC's `future_prn(nuclide_index, urr_seed)`: one base seed
/// per collision (redrawn only when the energy changes, so the band is stable
/// across a flight), decorrelated per nuclide by mixing in `za`. Because the
/// key is the nuclide's intrinsic `ZA`, the flight and the follow-up reaction
/// sample on the *struck* nuclide derive the identical random, keeping the
/// within-collision self-shielding correlation that distance and reaction
/// sampling rely on.
///
/// A mono-isotopic material collapses to a single hashed draw, statistically
/// equivalent to using the base seed directly.
#[inline]
pub fn urr_nuclide_random(base_seed: f64, za: u32) -> f64 {
    // SplitMix64 finalizer over (base entropy XOR za*golden). Bijective, so
    // distinct (base, za) map to distinct outputs; strong avalanche makes
    // adjacent za (e.g. 74182 / 74183 / 74184) decorrelate.
    let mut z = base_seed
        .to_bits()
        .wrapping_add((za as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits -> uniform [0, 1).
    (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Interpolation type for URR probability tables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UrrInterpolation {
    /// Linear-linear interpolation
    #[default]
    LinLin,
    /// Log-log interpolation (most common for URR)
    LogLog,
}

/// Cross-section set for a single probability table entry
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UrrXsSet {
    pub total: f64,
    pub elastic: f64,
    pub fission: f64,
    pub n_gamma: f64, // capture
    pub heating: f64,
}

/// URR probability table data for a single temperature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrrData {
    /// Interpolation type (lin-lin or log-log)
    pub interp: UrrInterpolation,

    /// Inelastic competition flag
    /// -1 = compute from (total - elastic - capture - fission)
    /// >0 = MT number of reaction to use for inelastic
    pub inelastic_flag: i32,

    /// Other absorption flag (0 or 1)
    pub absorption_flag: i32,

    /// If true, table values are relative factors to multiply by smooth XS
    pub multiply_smooth: bool,

    /// Energy grid points where URR tables are defined
    pub energy: Vec<f64>,

    /// Cumulative probability distribution values
    /// Shape: [n_energy, n_cdf]
    /// Stored as Vec<Vec<f64>> where outer vec is energy, inner is CDF values
    pub cdf_values: Vec<Vec<f64>>,

    /// Cross-section values corresponding to CDF points
    /// Shape: [n_energy, n_cdf]
    pub xs_values: Vec<Vec<UrrXsSet>>,
}

impl UrrData {
    /// Create empty URR data
    pub fn new() -> Self {
        Self {
            interp: UrrInterpolation::LinLin,
            inelastic_flag: -1,
            absorption_flag: 0,
            multiply_smooth: false,
            energy: Vec::new(),
            cdf_values: Vec::new(),
            xs_values: Vec::new(),
        }
    }

    /// Check if energy is within URR bounds
    #[inline]
    pub fn energy_in_bounds(&self, e: f64) -> bool {
        if self.energy.is_empty() {
            return false;
        }
        e > self.energy[0] && e < *self.energy.last().unwrap()
    }

    /// Get number of energy grid points
    #[inline]
    pub fn n_energy(&self) -> usize {
        self.energy.len()
    }

    /// Get number of CDF values per energy point
    #[inline]
    pub fn n_cdf(&self) -> usize {
        if self.cdf_values.is_empty() {
            0
        } else {
            self.cdf_values[0].len()
        }
    }

    /// Largest `elastic + n_gamma + fission` value across the
    /// probability-table CDF at the bracketing energy grid points
    /// around `energy`. Used to construct Woodcock majorants -- the
    /// returned value is the worst-case URR-sampled per-nuclide
    /// microscopic Σ_t (or factor, if `multiply_smooth` is true)
    /// for this nuclide at this energy. Iterating both bracketing
    /// energy points is intentional and slightly over-bounds the
    /// interpolated max -- safe for a majorant.
    ///
    /// Returns 0 if `energy` is outside the URR range or the table
    /// is empty.
    pub fn max_total_in_table(&self, energy: f64) -> f64 {
        if self.energy.is_empty() || !self.energy_in_bounds(energy) {
            return 0.0;
        }
        let i_energy = self.find_energy_index(energy);
        let i_next = (i_energy + 1).min(self.energy.len() - 1);
        let mut max_total = 0.0_f64;
        for i_e in [i_energy, i_next] {
            for xs_set in &self.xs_values[i_e] {
                // Per `sample()`, the URR total is recomputed from
                // partials, not read from the `.total` field. We
                // mirror that: max of `elastic + n_gamma + fission`,
                // each clamped non-negative. Inelastic is added by
                // the caller from smooth XS (it's not modified by
                // URR).
                let t = xs_set.elastic.max(0.0) + xs_set.n_gamma.max(0.0) + xs_set.fission.max(0.0);
                if t > max_total {
                    max_total = t;
                }
            }
        }
        max_total
    }

    /// Find the energy grid index below the given energy
    /// Returns index i such that energy[i] <= e < energy[i+1]
    pub fn find_energy_index(&self, e: f64) -> usize {
        if self.energy.is_empty() {
            return 0;
        }

        // Binary search for upper bound (first element > e)
        let mut left = 0;
        let mut right = self.energy.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.energy[mid] <= e {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Return index of element just at or below e
        // left now points to first element > e, so left-1 is <= e
        if left > 0 {
            left - 1
        } else {
            0
        }
    }

    /// Find the probability-table band for a random number at a given
    /// incident-energy index.
    ///
    /// `cdf[k]` is the cumulative probability up to and including band `k`, so the
    /// band containing `r` is the FIRST entry strictly greater than `r`:
    /// `r < cdf[0]` selects band 0, and `r` in `[cdf[k-1], cdf[k])` selects band
    /// `k`. That is what OpenMC computes -- `nuclide.cpp` does
    /// `upper_bound_index(cdf, r) + 1`, and its `upper_bound_index` is
    /// `std::upper_bound(..) - first - 1`, so the `+ 1` cancels back to a plain
    /// upper bound (`search.h`).
    ///
    /// The description this replaces claimed a genuine `+1` band offset ("for
    /// `r < cdf[0]`, use band 1, not band 0"), which is neither what the code
    /// below does nor what OpenMC does. Verified against both while measuring
    /// issue #364; the code was right and only the comment was wrong.
    fn find_cdf_index(&self, i_energy: usize, r: f64) -> usize {
        let cdf = &self.cdf_values[i_energy];

        // Binary search for upper bound (first element > r). After the loop,
        // `left` is the index of the first element strictly greater than `r`,
        // so `r ∈ [cdf[left-1], cdf[left])` and the band that contains `r` is
        // `left`.
        //
        // For `r < cdf[0]`, `left == 0` selects band 0. For `r` near 1.0 the
        // clamp guards the unlikely `left == cdf.len()` case.
        let mut left = 0;
        let mut right = cdf.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if cdf[mid] <= r {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left.min(cdf.len().saturating_sub(1))
    }

    /// Sample cross-sections from the probability tables
    ///
    /// # Arguments
    /// * `e` - Particle energy (eV)
    /// * `r` - Random number [0, 1)
    /// * `smooth_elastic` - Smooth elastic XS (used if multiply_smooth is true)
    /// * `smooth_absorption` - Smooth absorption XS (capture + fission + other_abs)
    /// * `smooth_fission` - Smooth fission XS
    /// * `smooth_inelastic` - Smooth inelastic XS (used when inelastic_flag > 0)
    /// * `smooth_ngamma` - Smooth n_gamma (MT=102) XS, used for multiply_smooth=false to
    ///   correctly separate n_gamma from other absorption. If None, falls back to
    ///   using (smooth_absorption - smooth_fission) which may include other_abs.
    ///
    /// # Returns
    /// Tuple of (urr_total, elastic, capture, fission, smooth_total_for_ratio)
    /// - urr_total: Total XS computed as elastic + inelastic + capture + fission
    /// - smooth_total_for_ratio: The smooth total that matches what's included in urr_total,
    ///   for proper ratio calculation. When inelastic_flag <= 0, this excludes inelastic.
    ///   For multiply_smooth=false, this ONLY includes reactions that URR modifies
    ///   (elastic + ngamma + fission), NOT other absorption like (n,p), (n,alpha).
    ///
    /// Inelastic handling:
    /// - If inelastic_flag > 0: use smooth_inelastic
    /// - If inelastic_flag <= 0: inelastic = 0
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        e: f64,
        r: f64,
        smooth_elastic: f64,
        smooth_absorption: f64,
        smooth_fission: f64,
        smooth_inelastic: f64,
        smooth_ngamma: Option<f64>,
    ) -> (f64, f64, f64, f64, f64) {
        // Skip if no URR data or outside energy range
        if self.energy.is_empty() || !self.energy_in_bounds(e) {
            let smooth_capture = smooth_absorption - smooth_fission;
            let smooth_total = smooth_elastic + smooth_inelastic + smooth_capture + smooth_fission;
            return (
                smooth_total,
                smooth_elastic,
                smooth_capture,
                smooth_fission,
                smooth_total,
            );
        }

        // Find energy grid index
        let i_energy = self.find_energy_index(e);
        let i_energy_next = (i_energy + 1).min(self.energy.len() - 1);

        // Calculate interpolation factor
        let f = match self.interp {
            UrrInterpolation::LinLin => {
                if i_energy_next > i_energy {
                    (e - self.energy[i_energy])
                        / (self.energy[i_energy_next] - self.energy[i_energy])
                } else {
                    0.0
                }
            }
            UrrInterpolation::LogLog => {
                if i_energy_next > i_energy && self.energy[i_energy] > 0.0 {
                    (e / self.energy[i_energy]).ln()
                        / (self.energy[i_energy_next] / self.energy[i_energy]).ln()
                } else {
                    0.0
                }
            }
        };

        // Find CDF indices at both energy points
        let i_low = self.find_cdf_index(i_energy, r);
        let i_up = self.find_cdf_index(i_energy_next, r);

        // Get cross-section sets
        let xs_low = &self.xs_values[i_energy][i_low];
        let xs_up = &self.xs_values[i_energy_next][i_up];

        // Interpolate cross-sections from the URR table
        // Note: Some URR tables (e.g., Cd106) have negative values in extreme probability bands
        // which are physically impossible. We clamp all values to be non-negative.
        let (_table_total, elastic, capture, fission) = match self.interp {
            UrrInterpolation::LinLin => {
                let total = ((1.0 - f) * xs_low.total + f * xs_up.total).max(0.0);
                let elastic = ((1.0 - f) * xs_low.elastic + f * xs_up.elastic).max(0.0);
                let capture = ((1.0 - f) * xs_low.n_gamma + f * xs_up.n_gamma).max(0.0);
                let fission = ((1.0 - f) * xs_low.fission + f * xs_up.fission).max(0.0);
                (total, elastic, capture, fission)
            }
            UrrInterpolation::LogLog => {
                let total = if xs_low.total > 0.0 && xs_up.total > 0.0 {
                    ((1.0 - f) * xs_low.total.ln() + f * xs_up.total.ln()).exp()
                } else {
                    ((1.0 - f) * xs_low.total + f * xs_up.total).max(0.0)
                };

                let elastic = if xs_low.elastic > 0.0 && xs_up.elastic > 0.0 {
                    ((1.0 - f) * xs_low.elastic.ln() + f * xs_up.elastic.ln()).exp()
                } else {
                    ((1.0 - f) * xs_low.elastic + f * xs_up.elastic).max(0.0)
                };

                let capture = if xs_low.n_gamma > 0.0 && xs_up.n_gamma > 0.0 {
                    ((1.0 - f) * xs_low.n_gamma.ln() + f * xs_up.n_gamma.ln()).exp()
                } else {
                    ((1.0 - f) * xs_low.n_gamma + f * xs_up.n_gamma).max(0.0)
                };

                let fission = if xs_low.fission > 0.0 && xs_up.fission > 0.0 {
                    ((1.0 - f) * xs_low.fission.ln() + f * xs_up.fission.ln()).exp()
                } else {
                    ((1.0 - f) * xs_low.fission + f * xs_up.fission).max(0.0)
                };

                (total, elastic, capture, fission)
            }
        };

        // Determine inelastic cross-section based on inelastic_flag
        // If inelastic_flag > 0: use smooth_inelastic (the MT specified by the flag)
        // If inelastic_flag <= 0 (C_NONE): inelastic = 0
        let inelastic = if self.inelastic_flag > 0 {
            smooth_inelastic
        } else {
            0.0
        };

        // Apply smooth cross-section multiplication if required
        if self.multiply_smooth {
            // When multiply_smooth=true, table values are FACTORS to multiply by smooth
            let elastic_final = elastic * smooth_elastic;
            let capture_final = capture * (smooth_absorption - smooth_fission);
            let fission_final = fission * smooth_fission;
            // IMPORTANT: Compute total as sum of partials, NOT from table's total column
            // The total XS is calculated as a sum of partials instead of the
            // table-provided value
            let total_final = elastic_final + inelastic + capture_final + fission_final;

            // For multiply_smooth=true, smooth_total_for_ratio includes all of smooth_absorption
            // because the URR factors scale the entire absorption (capture + fission).
            let smooth_total_for_ratio = if self.inelastic_flag > 0 {
                smooth_elastic + smooth_inelastic + smooth_absorption
            } else {
                smooth_elastic + smooth_absorption
            };

            (
                total_final,
                elastic_final,
                capture_final,
                fission_final,
                smooth_total_for_ratio,
            )
        } else {
            // When multiply_smooth=false, table values are ABSOLUTE cross-sections.
            //
            // When multiply_smooth=false:
            // 1. Use elastic, capture, fission directly from URR table (no multiplication)
            // 2. Total is recomputed as sum of partials (NOT using table's total column)
            // 3. micro.absorption = capture + fission (no other_abs added)
            //
            // IMPORTANT: When absorption_flag=0, the URR table's capture column only contains
            // n_gamma. For the ratio calculation, smooth_total_for_ratio must also only include
            // n_gamma (not full absorption with other_abs), otherwise the ratio will be wrong.

            // Recompute total from partials (sum of partials instead of table-provided value)
            let total_computed = elastic + inelastic + capture + fission;

            // For ratio calculation, smooth_total_for_ratio must match total_computed's components.
            // When absorption_flag=0 (URR capture = n_gamma only), we must use smooth n_gamma,
            // NOT the full smooth_absorption (which includes other_abs like n,p, n,alpha).
            let smooth_ngamma_val = if self.absorption_flag == 0 {
                // Use smooth n_gamma if available, else fall back to (absorption - fission)
                // which is an approximation that includes other_abs
                smooth_ngamma.unwrap_or(smooth_absorption - smooth_fission)
            } else {
                // When absorption_flag != 0, URR capture includes all absorption
                smooth_absorption - smooth_fission
            };

            let smooth_total_for_ratio = if self.inelastic_flag > 0 {
                smooth_elastic + smooth_inelastic + smooth_ngamma_val + smooth_fission
            } else {
                smooth_elastic + smooth_ngamma_val + smooth_fission
            };

            (
                total_computed,
                elastic,
                capture,
                fission,
                smooth_total_for_ratio,
            )
        }
    }
}

impl Default for UrrData {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `urr_nuclide_random` must be a deterministic function of (base, za) in
    /// `[0, 1)`, and produce statistically independent streams for the distinct
    /// ZAs of a real multi-isotope material (issue #204). Correlated per-nuclide
    /// draws are exactly the bug this decorrelation fixes.
    #[test]
    fn test_urr_nuclide_random_decorrelated_and_deterministic() {
        // Natural W isotopes.
        let zas: [u32; 5] = [74180, 74182, 74183, 74184, 74186];

        // Deterministic + in range.
        let base = 0.123_456_789;
        for &za in &zas {
            let a = urr_nuclide_random(base, za);
            let b = urr_nuclide_random(base, za);
            assert_eq!(a, b, "must be deterministic in (base, za)");
            assert!((0.0..1.0).contains(&a), "must be a uniform in [0,1): {a}");
        }
        // Distinct za -> distinct band random for the same base.
        for i in 0..zas.len() {
            for j in (i + 1)..zas.len() {
                assert_ne!(
                    urr_nuclide_random(base, zas[i]),
                    urr_nuclide_random(base, zas[j]),
                    "distinct ZAs must not collide onto the same random"
                );
            }
        }

        // Pairwise correlation ~0 over many base seeds, and each marginal ~U(0,1).
        let m = 200_000usize;
        let n = zas.len();
        let mut sum = vec![0.0f64; n];
        let mut sumsq = vec![0.0f64; n];
        let mut cross = vec![vec![0.0f64; n]; n];
        // simple SplitMix64 base-seed generator (Math.random-free, deterministic)
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next_base = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        };
        for _ in 0..m {
            let base = next_base();
            let r: Vec<f64> = zas.iter().map(|&za| urr_nuclide_random(base, za)).collect();
            for i in 0..n {
                sum[i] += r[i];
                sumsq[i] += r[i] * r[i];
                for j in 0..n {
                    cross[i][j] += r[i] * r[j];
                }
            }
        }
        let mf = m as f64;
        for i in 0..n {
            let mi = sum[i] / mf;
            let vi = sumsq[i] / mf - mi * mi;
            assert!((mi - 0.5).abs() < 0.01, "marginal mean off: {mi}");
            assert!((vi - 1.0 / 12.0).abs() < 0.005, "marginal var off: {vi}");
            for j in (i + 1)..n {
                let mj = sum[j] / mf;
                let vj = sumsq[j] / mf - mj * mj;
                let corr = (cross[i][j] / mf - mi * mj) / (vi.sqrt() * vj.sqrt());
                assert!(
                    corr.abs() < 0.02,
                    "ZAs {} and {} correlated: corr={corr}",
                    zas[i],
                    zas[j]
                );
            }
        }
    }

    #[test]
    fn test_energy_in_bounds() {
        let mut urr = UrrData::new();
        urr.energy = vec![1000.0, 5000.0, 10000.0];

        assert!(!urr.energy_in_bounds(500.0)); // Below range
        assert!(urr.energy_in_bounds(2000.0)); // In range
        assert!(urr.energy_in_bounds(7000.0)); // In range
        assert!(!urr.energy_in_bounds(10000.0)); // At upper bound (exclusive)
        assert!(!urr.energy_in_bounds(15000.0)); // Above range
    }

    #[test]
    fn test_find_energy_index() {
        let mut urr = UrrData::new();
        urr.energy = vec![1000.0, 5000.0, 10000.0, 20000.0];

        assert_eq!(urr.find_energy_index(500.0), 0);
        assert_eq!(urr.find_energy_index(1000.0), 0);
        assert_eq!(urr.find_energy_index(3000.0), 0);
        assert_eq!(urr.find_energy_index(5000.0), 1);
        assert_eq!(urr.find_energy_index(7000.0), 1);
        assert_eq!(urr.find_energy_index(15000.0), 2);
    }

    /// Regression test: `find_cdf_index` must return the band that
    /// CONTAINS `r` -- i.e. `r ∈ [cdf[i-1], cdf[i])` selects band `i`.
    /// An off-by-one (returning band `i + 1`) systematically biases the
    /// URR sample to higher cross sections, inflating reaction rates by
    /// ~10–20% on isotopes with URR tables (Fe55, Fe58 in ENDF/B-VIII.1).
    /// Cross-checked against a reference code's URR sampling
    /// (`upper_bound_index(...) + 1`).
    #[test]
    fn test_find_cdf_index_matches_reference_band_convention() {
        let mut urr = UrrData::new();
        urr.energy = vec![1.0e5];
        urr.cdf_values = vec![vec![0.10, 0.30, 0.60, 1.00]];
        urr.xs_values = vec![vec![
            UrrXsSet {
                total: 0.0,
                elastic: 0.0,
                fission: 0.0,
                n_gamma: 0.0,
                heating: 0.0,
            };
            4
        ]];

        // r before any cdf entry → band 0
        assert_eq!(urr.find_cdf_index(0, 0.05), 0);
        // r at lower edge of band 0 → band 0
        assert_eq!(urr.find_cdf_index(0, 0.0), 0);
        // r ∈ [0.10, 0.30) → band 1
        assert_eq!(urr.find_cdf_index(0, 0.10), 1);
        assert_eq!(urr.find_cdf_index(0, 0.20), 1);
        // r ∈ [0.30, 0.60) → band 2
        assert_eq!(urr.find_cdf_index(0, 0.30), 2);
        assert_eq!(urr.find_cdf_index(0, 0.45), 2);
        // r ∈ [0.60, 1.00) → band 3
        assert_eq!(urr.find_cdf_index(0, 0.60), 3);
        assert_eq!(urr.find_cdf_index(0, 0.99), 3);
        // r at/above the last cdf clamp to last band (defensive -- r should be < 1).
        assert_eq!(urr.find_cdf_index(0, 1.0), 3);
    }

    /// Regression test: probability-table sampling must preserve the
    /// table's mean cross-section. Off-by-one in the band lookup biased
    /// the empirical mean upward by ~20–30% on representative URR tables;
    /// here we use a contrived 4-band table whose CDF-weighted means are
    /// exactly known and check the stratified sample reproduces them.
    #[test]
    fn test_find_cdf_index_preserves_table_mean() {
        let mut urr = UrrData::new();
        urr.energy = vec![1.0e5];
        urr.cdf_values = vec![vec![0.25, 0.50, 0.75, 1.00]]; // four equal-prob bands
        urr.xs_values = vec![vec![
            UrrXsSet {
                total: 1.0,
                elastic: 0.5,
                fission: 0.0,
                n_gamma: 0.1,
                heating: 0.0,
            },
            UrrXsSet {
                total: 2.0,
                elastic: 1.0,
                fission: 0.0,
                n_gamma: 0.2,
                heating: 0.0,
            },
            UrrXsSet {
                total: 3.0,
                elastic: 1.5,
                fission: 0.0,
                n_gamma: 0.3,
                heating: 0.0,
            },
            UrrXsSet {
                total: 4.0,
                elastic: 2.0,
                fission: 0.0,
                n_gamma: 0.4,
                heating: 0.0,
            },
        ]];

        // Stratified sample: one r in each band's interior.
        let rs = [0.125, 0.375, 0.625, 0.875];
        let mean_total: f64 = rs
            .iter()
            .map(|&r| urr.xs_values[0][urr.find_cdf_index(0, r)].total)
            .sum::<f64>()
            / 4.0;
        let mean_elastic: f64 = rs
            .iter()
            .map(|&r| urr.xs_values[0][urr.find_cdf_index(0, r)].elastic)
            .sum::<f64>()
            / 4.0;
        let mean_capture: f64 = rs
            .iter()
            .map(|&r| urr.xs_values[0][urr.find_cdf_index(0, r)].n_gamma)
            .sum::<f64>()
            / 4.0;

        // Expected = simple arithmetic mean of band values for equal-prob bands.
        assert!(
            (mean_total - 2.5).abs() < 1e-12,
            "URR total bias: got {mean_total}, expected 2.5"
        );
        assert!(
            (mean_elastic - 1.25).abs() < 1e-12,
            "URR elastic bias: got {mean_elastic}, expected 1.25"
        );
        assert!(
            (mean_capture - 0.25).abs() < 1e-12,
            "URR capture bias: got {mean_capture}, expected 0.25"
        );
    }
}
