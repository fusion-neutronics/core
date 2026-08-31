//! Sample perturbed reaction rates from the folded covariance.
//!
//! One factorization per nuclide up front, then one cheap matrix-vector product
//! per replica. The perturbation is multiplicative and relative:
//!
//! ```text
//! L  = V √Λ        from the eigendecomposition of the relative covariance
//! z  ~ N(0, I)
//! δ  = L z
//! R' = R · (1 + δ)      floored at zero
//! ```
//!
//! # Why Jacobi, and not a linear algebra dependency
//!
//! The matrices are one per nuclide over that nuclide's activation channels, so
//! they are single digits to low tens on a side. At that size the cyclic Jacobi
//! rotation is fast, needs no dependency, is bit-reproducible across platforms
//! because it is pure arithmetic in a fixed order, and gives the eigenvectors
//! that the negative-eigenvalue clipping below needs anyway. A Cholesky would
//! be the obvious choice if the matrices were positive definite, and the whole
//! point is that they are not.
//!
//! # Clipping is reported, not silent
//!
//! MF=33 matrices are frequently not positive semi-definite as evaluated, and a
//! covariance that is not PSD has no square root, so the negative eigenvalues
//! have to go. Setting them to zero is the standard repair and it is what
//! happens here, but how much was thrown away is recorded in [`Clipping`]: a
//! matrix that needed a large repair is one whose sampled spread no longer means
//! what the evaluation said.
//!
//! # Seeds
//!
//! `history_seed(base, replica)` gives the replica its stream and
//! `secondary_seed(replica_seed, hash(nuclide))` gives each nuclide its own
//! within it. Both are pure functions of their arguments, so a rate depends on
//! `(seed, replica, nuclide)` and on nothing else: not on how many replicas were
//! run, not on the order they ran in, and not on which other nuclides were in
//! the material.

use std::collections::BTreeMap;

use yani::ReactionRates;

use crate::covariance_fold::RateCovariance;

/// How much repair the covariance matrices needed to be sampled from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Clipping {
    /// Matrices that had at least one negative eigenvalue.
    pub matrices_clipped: usize,
    /// The largest `|λ_min| / λ_max` over every matrix.
    ///
    /// Near zero is round-off and means nothing. Order one means the evaluation
    /// stated a covariance that is not a covariance, and the sampled spread is
    /// this code's repair rather than the evaluator's number.
    pub worst_relative_clip: f64,
}

/// How often a sampled rate went negative and was floored.
///
/// A negative cross section is unphysical, so the sample is truncated. Frequent
/// truncation means the Gaussian is being used past the point where it
/// describes the quantity, and the caller needs that reported rather than
/// absorbed: truncation biases the mean upward.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Truncations {
    pub floored: usize,
    pub sampled: usize,
}

/// One nuclide's factorized covariance.
struct Factor {
    kinds: Vec<String>,
    /// Row-major `n × n`, `L = V √Λ`.
    l: Vec<f64>,
    /// Per-channel relative standard deviation, `sqrt(sum_j L[i][j]^2)`.
    ///
    /// The marginal each row of `L` produces. Kept because the lognormal
    /// transform needs the marginal, not the joint, and recomputing it per
    /// replica would repeat the factorization's own work every draw.
    sigma: Vec<f64>,
}

/// The per-nuclide factorizations, built once and reused for every replica.
pub struct Sampler {
    factors: BTreeMap<String, Factor>,
    pub clipping: Clipping,
}

impl Sampler {
    /// Factorize every nuclide's relative covariance.
    pub fn new(covariance: &BTreeMap<String, RateCovariance>) -> Self {
        let mut factors = BTreeMap::new();
        let mut clipping = Clipping::default();

        // One eigendecomposition per nuclide, each reading only its own matrix.
        // The results land in a `BTreeMap` keyed by name, and the clipping
        // report is a count and a maximum, so nothing here can depend on the
        // order they finish in (issue #576, finding 5c).
        let entries: Vec<(&String, &RateCovariance)> = covariance.iter().collect();
        let one =
            |&(name, cov): &(&String, &RateCovariance)| -> Option<(String, Factor, Clipping)> {
                let mut clipping = Clipping::default();
                let n = cov.n();
                if n == 0 {
                    return None;
                }
                let (values, vectors) = jacobi_eigen(&cov.relative, n);

                let max = values.iter().cloned().fold(0.0_f64, f64::max);
                let min = values.iter().cloned().fold(0.0_f64, f64::min);
                if min < 0.0 {
                    clipping.matrices_clipped += 1;
                    if max > 0.0 {
                        clipping.worst_relative_clip = clipping.worst_relative_clip.max(-min / max);
                    }
                }

                // L = V √Λ, with negative eigenvalues clipped to zero. Column j of
                // V is eigenvector j, so L[i][j] = V[i][j] · √λ_j.
                let mut l = vec![0.0; n * n];
                for j in 0..n {
                    let s = values[j].max(0.0).sqrt();
                    if s == 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        l[i * n + j] = vectors[i * n + j] * s;
                    }
                }

                // The marginal each row produces, which is what the lognormal
                // transform is matched against. Computed here rather than per
                // replica: it is a property of the factorization, not of a draw.
                let sigma: Vec<f64> = (0..n)
                    .map(|i| {
                        l[i * n..(i + 1) * n]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();

                Some((
                    name.clone(),
                    Factor {
                        kinds: cov.kinds.clone(),
                        l,
                        sigma,
                    },
                    clipping,
                ))
            };

        let factorized: Vec<Option<(String, Factor, Clipping)>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                entries.par_iter().map(one).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                entries.iter().map(one).collect()
            }
        };
        for (name, factor, local) in factorized.into_iter().flatten() {
            clipping.matrices_clipped += local.matrices_clipped;
            clipping.worst_relative_clip =
                clipping.worst_relative_clip.max(local.worst_relative_clip);
            factors.insert(name, factor);
        }

        Sampler { factors, clipping }
    }

    /// Whether anything was factorized at all.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// The relative perturbations one replica applies, per nuclide and kind.
    ///
    /// Separated from [`Sampler::perturb`] so a test can look at the deviates
    /// themselves rather than inferring them from perturbed rates.
    pub fn deviates(&self, base_seed: u64, replica: u64) -> BTreeMap<String, Vec<f64>> {
        let replica_seed = yamc_rng::history_seed(base_seed, replica);
        let mut out = BTreeMap::new();
        for (name, factor) in &self.factors {
            let n = factor.kinds.len();
            // Keyed on the NAME, not on a position, so a nuclide's stream does
            // not move when a different nuclide joins or leaves the material.
            let seed = yamc_rng::secondary_seed(replica_seed, name_ordinal(name));
            let mut state = yamc_rng::expand_seed(seed);
            let z = standard_normals(&mut state, n);

            // delta = L z, one row of the factor at a time.
            let delta: Vec<f64> = factor
                .l
                .chunks_exact(n)
                .map(|row| row.iter().zip(&z).map(|(l, z)| l * z).sum())
                .collect();
            out.insert(name.clone(), delta);
        }
        out
    }

    /// Apply one replica's perturbation to a set of unit-flux rates.
    ///
    /// Rates for nuclides or kinds with no covariance are passed through
    /// unchanged. That is deliberate and is what the coverage report is for: an
    /// unperturbed rate contributes no uncertainty, and the reason has to be
    /// visible rather than inferred from a suspiciously small sigma.
    pub fn perturb(
        &self,
        rates: &ReactionRates,
        base_seed: u64,
        replica: u64,
    ) -> (ReactionRates, Truncations) {
        let deviates = self.deviates(base_seed, replica);
        let mut truncations = Truncations::default();
        let mut out = rates.clone();

        for (name, factor) in &self.factors {
            let (Some(delta), Some(nuclide_rates)) = (deviates.get(name), out.get_mut(name)) else {
                continue;
            };
            for (i, kind) in factor.kinds.iter().enumerate() {
                let Some(rate) = nuclide_rates.get_mut(kind) else {
                    continue;
                };
                truncations.sampled += 1;
                let scaled = *rate * lognormal_factor(delta[i], factor.sigma[i]);
                if scaled < 0.0 {
                    // Unreachable: the factor is an exponential. Kept as the
                    // assertion it now is -- a nonzero count here is a bug in
                    // this function, not a property of the data.
                    truncations.floored += 1;
                    *rate = 0.0;
                } else {
                    *rate = scaled;
                }
            }
        }

        (out, truncations)
    }
}

/// A stable 32-bit hash of a nuclide name.
///
/// FNV-1a, spelled out rather than taken from `DefaultHasher`, whose output is
/// explicitly not stable across releases. The seed contract promises the same
/// answer on every platform and every build, and a hash that may change is not
/// compatible with that.
fn name_ordinal(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// `n` standard normal deviates, by Box-Muller over the PCG uniform stream.
///
/// `next_xi` returns a value in (0, 1], so the logarithm is always defined and
/// no rejection loop is needed. Deviates are produced in pairs and the spare of
/// an odd request is discarded, which keeps the stream position a function of
/// `n` alone.
pub(crate) fn standard_normals(state: &mut u64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = yamc_rng::next_xi(state);
        let u2 = yamc_rng::next_xi(state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

/// Eigen-decompose a symmetric matrix by cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j` of
/// the row-major `n × n` result.
///
/// The sweep order is fixed and the iteration count is bounded, so this is a
/// pure function of its input: two runs on the same matrix give the same last
/// bit, which is what the reproducibility contract needs. Convergence is
/// quadratic and these matrices are tiny, so the bound is never the thing that
/// stops it in practice.
fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    // Symmetrize on the way in. The fold fills both halves with the same value,
    // so this is a no-op on real input; it costs nothing and means a caller
    // cannot hand in something the rotation would silently misread.
    for i in 0..n {
        for j in (i + 1)..n {
            let m = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = m;
            a[j * n + i] = m;
        }
    }

    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Enough sweeps for quadratic convergence at any size this sees, with a
    // hard stop so a pathological input cannot spin.
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off <= f64::EPSILON * f64::EPSILON {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // The rotation that zeroes (p, q). `theta` can overflow only if
                // apq is denormal beside a huge diagonal, in which case the
                // rotation is the identity anyway.
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                if !t.is_finite() {
                    continue;
                }
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let values = (0..n).map(|i| a[i * n + i]).collect();
    (values, v)
}

/// A positive multiplier with mean 1 and variance `sigma^2`.
///
/// `deviate` is one component of `L z`, so it is normal with standard
/// deviation `sigma` and carries this channel's correlations with the others.
/// Feeding it through the lognormal marginal
///
/// ```text
/// s^2 = ln(1 + sigma^2)
/// f   = exp(s * (deviate / sigma) - s^2 / 2)
/// ```
///
/// keeps those two moments exactly and cannot go negative, which
/// `R * (1 + deviate)` could and did: on the FNS decay-heat benchmark that
/// form floored 8.3% of all sampled rates at zero, 13% on iron, and every
/// truncation biases the mean upward.
///
/// For a small `sigma` the two agree to second order -- `exp(s z - s^2/2)`
/// is `1 + sigma z + O(sigma^2)` -- so a well known cross section samples as
/// it did before, and only the channels that were being pushed past where a
/// Gaussian describes them move.
///
/// What this does NOT preserve exactly is the linear correlation between
/// channels. The deviates keep their Gaussian dependence and each is then
/// transformed monotonically, which is a Gaussian copula: rank correlation is
/// preserved exactly, Pearson correlation shifts by a factor that goes to one
/// as sigma goes to zero. The alternative was a truncated normal, which
/// preserves neither the mean nor positivity without an accept-reject loop
/// whose cost depends on the data.
fn lognormal_factor(deviate: f64, sigma: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 || !deviate.is_finite() {
        return 1.0;
    }
    let s_squared = (1.0 + sigma * sigma).ln();
    let s = s_squared.sqrt();
    (s * (deviate / sigma) - 0.5 * s_squared).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cov(kinds: &[&str], relative: Vec<f64>) -> RateCovariance {
        RateCovariance {
            kinds: kinds.iter().map(|s| s.to_string()).collect(),
            relative,
        }
    }

    #[test]
    fn jacobi_recovers_a_known_spectrum() {
        // diag(4, 1) rotated by 45 degrees: [[2.5, 1.5], [1.5, 2.5]].
        let (values, _) = jacobi_eigen(&[2.5, 1.5, 1.5, 2.5], 2);
        let mut v = values;
        v.sort_by(f64::total_cmp);
        assert!((v[0] - 1.0).abs() < 1e-12, "{v:?}");
        assert!((v[1] - 4.0).abs() < 1e-12, "{v:?}");
    }

    /// `L Lᵀ` must reproduce the matrix it was factorized from.
    #[test]
    fn the_factor_reproduces_a_positive_definite_matrix() {
        let c = cov(&["(n,gamma)", "(n,p)"], vec![0.04, 0.012, 0.012, 0.09]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c.clone())]));
        assert_eq!(s.clipping.matrices_clipped, 0);

        let f = &s.factors["Fe56"];
        for i in 0..2 {
            for j in 0..2 {
                let got: f64 = (0..2).map(|k| f.l[i * 2 + k] * f.l[j * 2 + k]).sum();
                assert!((got - c.get(i, j)).abs() < 1e-12, "({i},{j}) {got}");
            }
        }
    }

    /// A matrix that is not a covariance is repaired, and the repair is
    /// reported rather than absorbed.
    #[test]
    fn a_non_psd_matrix_is_clipped_and_counted() {
        // Correlation above one: not positive semi-definite.
        let c = cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]);
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));
        assert_eq!(s.clipping.matrices_clipped, 1);
        assert!(
            s.clipping.worst_relative_clip > 0.5,
            "a large repair must read as large: {}",
            s.clipping.worst_relative_clip
        );
    }

    /// The same (seed, replica, nuclide) gives the same deviates, and a
    /// different replica gives different ones.
    #[test]
    fn deviates_are_a_pure_function_of_seed_and_replica() {
        let c = cov(&["a", "b"], vec![0.04, 0.0, 0.0, 0.09]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));

        assert_eq!(s.deviates(42, 7), s.deviates(42, 7));
        assert_ne!(s.deviates(42, 7), s.deviates(42, 8));
        assert_ne!(s.deviates(1, 7), s.deviates(42, 7));
    }

    /// A nuclide's stream does not move when another nuclide is added.
    ///
    /// Keying the sub-seed on the name rather than on a position is what buys
    /// this, and without it every rate in a material would change when an
    /// unrelated nuclide entered the chain.
    #[test]
    fn one_nuclides_stream_does_not_depend_on_the_others() {
        let a = cov(&["a"], vec![0.04]);
        let b = cov(&["b"], vec![0.09]);
        let alone = Sampler::new(&BTreeMap::from([("Fe56".to_string(), a.clone())]));
        let together = Sampler::new(&BTreeMap::from([
            ("Co59".to_string(), b),
            ("Fe56".to_string(), a),
        ]));
        assert_eq!(
            alone.deviates(5, 3)["Fe56"],
            together.deviates(5, 3)["Fe56"]
        );
    }

    /// Over many replicas the sampled relative spread matches the diagonal it
    /// was built from.
    #[test]
    fn the_sampled_spread_matches_the_stated_uncertainty() {
        // 20% and 30% relative, uncorrelated.
        let c = cov(&["a", "b"], vec![0.04, 0.0, 0.0, 0.09]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));

        let n = 20_000;
        let mut sums = [0.0; 2];
        let mut squares = [0.0; 2];
        for k in 0..n {
            let d = &s.deviates(11, k)["Fe56"];
            for i in 0..2 {
                sums[i] += d[i];
                squares[i] += d[i] * d[i];
            }
        }
        let nf = n as f64;
        for (i, want) in [0.2_f64, 0.3].into_iter().enumerate() {
            let mean = sums[i] / nf;
            let sd = (squares[i] / nf - mean * mean).sqrt();
            assert!(mean.abs() < 0.02, "mean {mean} for channel {i}");
            assert!(
                (sd - want).abs() < 0.02 * want.max(1.0) + 0.01,
                "sd {sd} != {want} for channel {i}"
            );
        }
    }

    /// A correlated pair comes out correlated.
    #[test]
    fn correlations_survive_the_factorization() {
        // 20% each, correlation +0.8.
        let c = cov(&["a", "b"], vec![0.04, 0.032, 0.032, 0.04]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));

        let n = 20_000;
        let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..n {
            let d = &s.deviates(3, k)["Fe56"];
            sa += d[0];
            sb += d[1];
            saa += d[0] * d[0];
            sbb += d[1] * d[1];
            sab += d[0] * d[1];
        }
        let nf = n as f64;
        let (ma, mb) = (sa / nf, sb / nf);
        let cov_ab = sab / nf - ma * mb;
        let rho = cov_ab / ((saa / nf - ma * ma).sqrt() * (sbb / nf - mb * mb).sqrt());
        assert!((rho - 0.8).abs() < 0.03, "rho {rho}");
    }

    #[test]
    fn nothing_is_floored_any_more_because_nothing_goes_negative() {
        // The same case that used to floor: a relative variance of 9, so a
        // relative sigma of 3. A Gaussian multiplier `1 + delta` is negative
        // about a third of the time at that width; an exponential one never
        // is.
        let c = cov(&["(n,gamma)"], vec![9.0]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));
        let rates: ReactionRates = std::collections::HashMap::from([(
            "Fe56".to_string(),
            std::collections::HashMap::from([("(n,gamma)".to_string(), 1.0e-8)]),
        )]);

        let mut floored = 0;
        for k in 0..2000 {
            let (out, t) = s.perturb(&rates, 9, k);
            floored += t.floored;
            assert!(out["Fe56"]["(n,gamma)"] > 0.0, "a rate must stay positive");
        }
        assert_eq!(floored, 0, "the lognormal draw cannot be floored");
    }

    #[test]
    fn the_ensemble_mean_is_the_nominal_rate() {
        // The property the old form lost. Truncating at zero threw away the
        // negative tail and kept its probability mass at the boundary, which
        // pushed the mean up; that is what `rates_floored` was reporting.
        // A relative variance of 0.25, so a relative sigma of 0.5: `cov` takes
        // the covariance matrix, not the standard deviations.
        let c = cov(&["(n,gamma)"], vec![0.25]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));
        let nominal = 1.0e-8;
        let rates: ReactionRates = std::collections::HashMap::from([(
            "Fe56".to_string(),
            std::collections::HashMap::from([("(n,gamma)".to_string(), nominal)]),
        )]);

        let n = 20_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for k in 0..n {
            let v = s.perturb(&rates, 11, k).0["Fe56"]["(n,gamma)"];
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;

        // Both moments, not just the mean: matching the mean by shrinking the
        // spread would be no use to an uncertainty.
        assert!(
            (mean / nominal - 1.0).abs() < 0.02,
            "mean {mean:e} against nominal {nominal:e}"
        );
        let relative_sigma = var.sqrt() / nominal;
        assert!(
            (relative_sigma - 0.5).abs() < 0.03,
            "sampled relative sigma {relative_sigma} against the stated 0.5"
        );
    }

    #[test]
    fn a_small_uncertainty_samples_as_it_did_before() {
        // exp(s z - s^2/2) is 1 + sigma z to second order, so a well known
        // cross section must not move just because the distribution changed.
        for sigma in [0.001, 0.01, 0.05] {
            for z in [-2.0, -0.5, 0.5, 2.0] {
                let deviate = sigma * z;
                let lognormal = super::lognormal_factor(deviate, sigma);
                let linear = 1.0 + deviate;
                assert!(
                    (lognormal - linear).abs() < 2.0 * sigma * sigma * (1.0 + z * z),
                    "sigma {sigma} z {z}: {lognormal} vs {linear}"
                );
            }
        }
    }

    /// A nuclide with no covariance is passed through untouched.
    #[test]
    fn rates_without_covariance_are_unchanged() {
        let s = Sampler::new(&BTreeMap::new());
        assert!(s.is_empty());
        let rates: ReactionRates = std::collections::HashMap::from([(
            "Fe56".to_string(),
            std::collections::HashMap::from([("(n,gamma)".to_string(), 2.5)]),
        )]);
        let (out, t) = s.perturb(&rates, 1, 0);
        assert_eq!(out, rates);
        assert_eq!(t, Truncations::default());
    }
}
