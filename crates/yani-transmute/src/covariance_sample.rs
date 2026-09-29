//! Sample perturbed reaction rates from the folded covariance.
//!
//! One factorization per nuclide up front, then one cheap matrix-vector product
//! per replica. The perturbation is multiplicative and relative:
//!
//! ```text
//! L   = V √Λ        from the eigendecomposition of the relative covariance
//! z   ~ N(0, I)
//! δ   = L z
//! σ_i = sqrt(Σ_j L_ij²)
//! s_i = sqrt(ln(1 + σ_i²))
//! R'  = R · exp(s_i δ_i / σ_i - s_i² / 2)
//! ```
//!
//! Each channel's multiplier is lognormal with mean 1 and variance `σ_i²`, the
//! evaluation's own two moments after any PSD repair, so a sampled rate is
//! never negative and nothing is floored. [`lognormal_factor`] says what that
//! keeps and what it does not.
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
//! happens here. It is not neutral: with `C = C+ + C-` split by the sign of
//! the eigenvalues, the matrix sampled is `C+ = C - C-`, and `-C-` is positive
//! semi-definite, so every diagonal of `C+` is at least the evaluated one.
//! Clipping only ever adds variance, and it can give a spread to a channel the
//! evaluation states as exactly zero. What it did is recorded per nuclide and
//! spectrum in [`Repair`], with the evaluated sigma `sqrt(C_ii)` next to the
//! sampled `sqrt(C+_ii)` for every channel, because a small `|λ_min| / λ_max`
//! does not mean a small change to the sigma that matters.
//!
//! # Seeds
//!
//! `history_seed(base, replica)` gives the replica its stream and
//! `secondary_seed(replica_seed, hash(nuclide))` gives each nuclide its own
//! within it. Both are pure functions of their arguments, so a rate depends on
//! `(seed, replica, nuclide)` and on nothing else: not on how many replicas were
//! run, not on the order they ran in, and not on which other nuclides were in
//! the material.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use yani::ReactionRates;

use crate::covariance_fold::RateCovariance;

/// How far below zero the smallest eigenvalue of the correlation matrix may
/// come back, per channel with a positive stated variance, before the
/// covariance counts as repaired: the threshold is `-m * REPAIR_TOLERANCE`
/// with `m` the number of such channels, the side of the correlation matrix.
///
/// The test is on `R = D^-1/2 C D^-1/2` over the channels with a positive
/// stated variance, not on `C` itself. `R` is congruent to that block of `C`,
/// so by Sylvester's law one is PSD exactly when the other is, and a channel
/// stated at zero or below is judged without an eigenvalue at all: a negative
/// variance is a repair, and so is a zero one with any covariance to another
/// channel. What scaling buys is that round-off is on each channel's own
/// scale. The Jacobi round-off on `C` is absolute, about `eps · λ_max`, and a
/// small channel can carry an O(1) share of a null-space eigenvector, so on
/// `C` a rank-one matrix whose channels span a few orders in sigma already
/// reads as needing clipping. On `R` every diagonal is one and `λ_max <= m`,
/// so the round-off is about `m · eps`, orders below `m · 1e-12`, whatever the
/// spread. The spread itself is real: TENDL-2017 has channels at a relative
/// sigma up to 5.6e8, a variance near 3e17, and a threshold against the
/// `λ_max` of `C` would sit far above a real repair among the ordinary
/// channels beside one.
///
/// Below the threshold the matrix is PSD to within round-off and the
/// eigenvalues that come back negative are still clipped, since they have no
/// square root either. What that clipping adds to a channel is round-off of
/// the decomposition of `C`, about `eps · λ_max`, and it is not counted as a
/// repair. It is not hidden either: the rate-weighted headline reads every
/// channel's sampled sigma off the factorization, so a matrix below the
/// threshold shows its round-off there as it is.
pub const REPAIR_TOLERANCE: f64 = 1.0e-12;

/// One channel of a repaired matrix, its sigma as evaluated and as sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSigma {
    pub kind: String,
    /// `C_ii`, the folded relative variance exactly as the evaluation gives
    /// it. A matrix that is not PSD can fold to a negative diagonal, and that
    /// is kept as it is rather than read as a zero.
    pub evaluated_variance: f64,
    /// `sqrt(C+_ii)`, the relative sigma the lognormal is matched to after the
    /// repair. Never below the evaluated sigma beyond round-off.
    pub sampled: f64,
}

impl ChannelSigma {
    /// `sqrt(C_ii)`, the folded relative sigma the evaluation states, or
    /// `None` when the stated variance is negative and so has no sigma.
    pub fn evaluated_sigma(&self) -> Option<f64> {
        (self.evaluated_variance >= 0.0).then(|| self.evaluated_variance.sqrt())
    }

    /// `sampled / evaluated - 1`: how far the repair widened this channel.
    /// Infinite when the repair gave a spread to a channel whose stated
    /// variance is zero or negative.
    pub fn inflation(&self) -> f64 {
        match self.evaluated_sigma() {
            Some(e) if e > 0.0 => self.sampled / e - 1.0,
            _ if self.sampled > 0.0 => f64::INFINITY,
            _ => 0.0,
        }
    }
}

/// One nuclide's covariance on one spectrum that had to be repaired to be
/// sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct Repair {
    pub nuclide: String,
    /// Index of the spectrum the covariance was folded against, in the order
    /// the schedule names them.
    pub spectrum: usize,
    /// The most negative eigenvalue of the folded relative covariance.
    pub lambda_min: f64,
    /// The largest eigenvalue.
    pub lambda_max: f64,
    /// `sum |λ_neg| / trace(C)`: the variance the clipping added, as a share
    /// of the variance the evaluation states. Infinite when the stated trace
    /// is not positive.
    pub clipped_fraction: f64,
    /// Every channel of the matrix, in the fold's kind order.
    pub channels: Vec<ChannelSigma>,
}

/// The eigenvalue summary of a matrix whose repair counts as one.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Eigen {
    lambda_min: f64,
    lambda_max: f64,
    clipped_fraction: f64,
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
    /// Per-channel `C_ii` as folded, before any repair, negative or not.
    evaluated_variance: Vec<f64>,
    /// Set when the matrix is not PSD past [`REPAIR_TOLERANCE`].
    repair: Option<Eigen>,
}

/// The per-nuclide factorizations, built once and reused for every replica.
pub struct Sampler {
    factors: BTreeMap<String, Factor>,
}

impl Sampler {
    /// Factorize every nuclide's relative covariance.
    pub fn new(covariance: &BTreeMap<String, RateCovariance>) -> Self {
        // One eigendecomposition per nuclide, each reading only its own matrix.
        // The results land in a `BTreeMap` keyed by name, so nothing here can
        // depend on the order they finish in (issue #576, finding 5c).
        let entries: Vec<(&String, &RateCovariance)> = covariance.iter().collect();
        let one = |&(name, cov): &(&String, &RateCovariance)| -> Option<(String, Factor)> {
            let n = cov.n();
            if n == 0 {
                return None;
            }
            let (values, vectors) = jacobi_eigen(&cov.relative, n);

            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let repair = needs_repair(cov).then(|| {
                let trace: f64 = (0..n).map(|i| cov.get(i, i)).sum();
                let clipped: f64 = values.iter().filter(|v| **v < 0.0).map(|v| -v).sum();
                Eigen {
                    lambda_min: min,
                    lambda_max: max,
                    clipped_fraction: if trace > 0.0 {
                        clipped / trace
                    } else {
                        f64::INFINITY
                    },
                }
            });

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
                    evaluated_variance: (0..n).map(|i| cov.get(i, i)).collect(),
                    repair,
                },
            ))
        };

        let factorized: Vec<Option<(String, Factor)>> = {
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
        Sampler {
            factors: factorized.into_iter().flatten().collect(),
        }
    }

    /// Every matrix this sampler had to repair, tagged with `spectrum`.
    pub fn repairs(&self, spectrum: usize) -> Vec<Repair> {
        self.factors
            .iter()
            .filter_map(|(name, f)| {
                let r = f.repair?;
                Some(Repair {
                    nuclide: name.clone(),
                    spectrum,
                    lambda_min: r.lambda_min,
                    lambda_max: r.lambda_max,
                    clipped_fraction: r.clipped_fraction,
                    channels: f
                        .kinds
                        .iter()
                        .zip(f.evaluated_variance.iter().zip(&f.sigma))
                        .map(|(kind, (v, s))| ChannelSigma {
                            kind: kind.clone(),
                            evaluated_variance: *v,
                            sampled: *s,
                        })
                        .collect(),
                })
            })
            .collect()
    }

    /// Every channel with a factor, as `(nuclide, kind, evaluated, sampled)`
    /// relative sigmas, the evaluated one zero where the stated variance is
    /// negative (it states no spread a weighted mean could use).
    ///
    /// The sampled sigma is the one the lognormal is matched to, on every
    /// matrix: where no repair was needed it differs from the evaluated one
    /// by the decomposition's round-off, and that is reported as it is rather
    /// than replaced by the evaluated value.
    pub(crate) fn channel_sigmas(&self) -> impl Iterator<Item = (&str, &str, f64, f64)> {
        self.factors.iter().flat_map(|(name, f)| {
            f.kinds.iter().enumerate().map(move |(i, kind)| {
                let evaluated = f.evaluated_variance[i].max(0.0).sqrt();
                (name.as_str(), kind.as_str(), evaluated, f.sigma[i])
            })
        })
    }

    /// Whether anything was factorized at all.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// The nuclides carrying a factor, with each one's reaction kinds and its
    /// row-major factor `L` (`L L^T` the relative covariance of those kinds'
    /// rates), for first-order attribution.
    pub(crate) fn factors(&self) -> impl Iterator<Item = (&String, &[String], &[f64])> {
        self.factors
            .iter()
            .map(|(name, f)| (name, f.kinds.as_slice(), f.l.as_slice()))
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

    /// Apply one replica's perturbation to a set of unit-flux rates, and count
    /// the rates it drew.
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
    ) -> (ReactionRates, usize) {
        let deviates = self.deviates(base_seed, replica);
        let mut sampled = 0;
        let mut out = rates.clone();

        for (name, factor) in &self.factors {
            let (Some(delta), Some(nuclide_rates)) = (deviates.get(name), out.get_mut(name)) else {
                continue;
            };
            for (i, kind) in factor.kinds.iter().enumerate() {
                let Some(rate) = nuclide_rates.get_mut(kind) else {
                    continue;
                };
                sampled += 1;
                *rate *= lognormal_factor(delta[i], factor.sigma[i]);
            }
        }

        (out, sampled)
    }
}

/// The repairs and the widest sigmas over every spectrum of a run, for the
/// nuclides the material can populate.
///
/// All of it is read off the factorization, so it is exact with respect to the
/// folded covariance: nothing here is a modelling choice, it says where one
/// was made. The fold covers every chain nuclide with data, and from almost
/// any composition the chain's closure saturates on one large component, so
/// a pure W182 material folds Xe135 and Mo100. Those are left out here by
/// `populated`, the nuclides `yani::populated_nuclides` bounds at or above the
/// solver's own [`crate::DENSITY_FLOOR`] over the nominal schedule. The bound
/// never under-estimates the nominal solve, so there a nuclide outside it
/// cannot move any density by as much as the floor the solver drops a
/// nuclide at. It is a bound on the nominal rates only: a replica draws each
/// rate from a lognormal, and on a wide channel that can put a rate orders
/// above nominal, so a replica can populate a nuclide the nominal bound
/// leaves out, and the repairs and wide channels of those nuclides are
/// named separately rather than dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SigmaReport {
    /// One entry per (populated nuclide, spectrum) whose matrix needed a
    /// repair, every channel of it whether or not a draw can move the result.
    /// A repair of a nuclide outside `populated` is not listed here; its
    /// nuclide is named in `repaired_outside_bound` instead.
    pub repairs: Vec<Repair>,
    /// The nuclides outside `populated` whose matrix needed a repair on some
    /// spectrum. Not a gap on the nominal bound, but named so that leaving
    /// their records out is visible, since a replica can populate them.
    pub repaired_outside_bound: BTreeSet<String>,
    /// The nuclides of `repairs` with at least one repaired channel a draw
    /// can move: a positive rate on a spectrum the schedule irradiates with.
    /// A repair only on an unirradiated spectrum, or only on channels with no
    /// rate, widens nothing the ensemble sees, so it is recorded but is not
    /// a gap.
    pub repaired: BTreeSet<String>,
    /// The largest [`ChannelSigma::inflation`] over the repaired channels
    /// that can move the result: a populated nuclide with a positive rate on
    /// a spectrum the schedule irradiates with. Zero with no such repair, and
    /// infinite when the repair gave a spread to a channel whose stated
    /// variance is zero or negative.
    pub worst_sigma_inflation: f64,
    /// `sum_c w_c (sampled_c / evaluated_c - 1)` and `sum_c w_c` over every
    /// sampled channel `c` of every spectrum with a positive evaluated sigma,
    /// `w_c` the channel's unit-flux rate times the spectrum's fluence in the
    /// schedule times the parent's initial density: the reactions the
    /// schedule puts through the channel, at the starting composition. The
    /// weight is on each channel's own inflation, not on its sigma, so a wide
    /// channel with a small rate cannot drown out a repair on the channels
    /// that carry the reactions. Kept as sums because sums merge across
    /// spectra.
    weighted_inflation: f64,
    weight: f64,
    /// Whether a weighted channel with no evaluated sigma was sampled with a
    /// spread, which makes the weighted inflation infinite.
    weighted_spread_from_nothing: bool,
    /// Sampled channels of populated nuclides with a positive rate on a
    /// spectrum the schedule irradiates with, whose folded relative sigma
    /// `sqrt(C_ii)`, as evaluated and before any repair, is at least one,
    /// keyed by (nuclide, kind), the largest over the spectra. Past this
    /// width the answer depends on the distribution chosen to carry the two
    /// moments, not only on the moments the evaluation states. A repair's own
    /// widening is in `repairs`.
    pub sigma_at_least_one: BTreeMap<(String, String), f64>,
    /// The subset at ten or more.
    pub sigma_at_least_ten: BTreeMap<(String, String), f64>,
    /// The same as `sigma_at_least_one` for the nuclides outside `populated`:
    /// sampled channels with a positive rate on a spectrum the schedule
    /// irradiates with, evaluated at a relative sigma of one or more. Not a
    /// gap on the nominal bound, but a replica's draw on exactly such a
    /// channel can sit orders above nominal and populate the nuclide, so they
    /// are named. The ten-or-more subset reads off the values.
    pub sigma_at_least_one_outside_bound: BTreeMap<(String, String), f64>,
}

impl SigmaReport {
    /// Fold in one spectrum's sampler, with the unit-flux `rates` it perturbs,
    /// the `fluence` the schedule gives that spectrum (the sum of flux times
    /// duration over its irradiation steps) and the material's initial atom
    /// densities for the weighting, and the `populated` nuclides the report
    /// is restricted to.
    ///
    /// Only channels of a populated nuclide with a positive rate on a spectrum
    /// with a positive fluence are counted: [`Sampler::perturb`] draws for
    /// every channel with a rate, but a multiplier on a zero rate moves
    /// nothing.
    pub fn add(
        &mut self,
        spectrum: usize,
        sampler: &Sampler,
        rates: &ReactionRates,
        fluence: f64,
        densities: &HashMap<String, f64>,
        populated: &HashSet<String>,
    ) {
        let drawn = |nuclide: &str, kind: &str| {
            fluence > 0.0
                && rates
                    .get(nuclide)
                    .and_then(|k| k.get(kind))
                    .is_some_and(|r| *r > 0.0)
        };
        let moves = |nuclide: &str, kind: &str| populated.contains(nuclide) && drawn(nuclide, kind);
        for repair in sampler.repairs(spectrum) {
            if !populated.contains(&repair.nuclide) {
                self.repaired_outside_bound.insert(repair.nuclide);
                continue;
            }
            for c in repair
                .channels
                .iter()
                .filter(|c| moves(&repair.nuclide, &c.kind))
            {
                self.repaired.insert(repair.nuclide.clone());
                self.worst_sigma_inflation = self.worst_sigma_inflation.max(c.inflation());
            }
            self.repairs.push(repair);
        }
        for (nuclide, kind, evaluated, sampled) in sampler.channel_sigmas() {
            if !drawn(nuclide, kind) {
                continue;
            }
            let key = || (nuclide.to_string(), kind.to_string());
            if !populated.contains(nuclide) {
                if evaluated >= 1.0 {
                    let e = self
                        .sigma_at_least_one_outside_bound
                        .entry(key())
                        .or_insert(0.0);
                    *e = e.max(evaluated);
                }
                continue;
            }
            let w = rates[nuclide][kind] * fluence * densities.get(nuclide).copied().unwrap_or(0.0);
            if w > 0.0 && w.is_finite() {
                if evaluated > 0.0 {
                    self.weighted_inflation += w * (sampled / evaluated - 1.0);
                    self.weight += w;
                } else if sampled > 0.0 {
                    self.weighted_spread_from_nothing = true;
                }
            }
            for (threshold, set) in [
                (1.0, &mut self.sigma_at_least_one),
                (10.0, &mut self.sigma_at_least_ten),
            ] {
                if evaluated >= threshold {
                    let e = set.entry(key()).or_insert(0.0);
                    *e = e.max(evaluated);
                }
            }
        }
    }

    /// The weighted mean of `sampled / evaluated - 1` over every sampled
    /// channel of a populated nuclide, each weighted by the reactions the
    /// schedule puts through it at the initial composition (see
    /// `weighted_inflation`).
    ///
    /// The headline beside [`SigmaReport::worst_sigma_inflation`]: a repair on
    /// a channel that carries no reactions costs nothing here, and one on the
    /// dominant channel costs its full share, however wide the channels
    /// beside it are. It covers the reactions on the initial composition
    /// only: a nuclide the material starts without has no initial density and
    /// so no weight, however much of the inventory passes through it, and its
    /// repairs are in `repairs` and `worst_sigma_inflation` instead. Infinite
    /// when a weighted channel with no evaluated sigma was sampled with a
    /// spread, and `None` when no weighted channel has an evaluated sigma.
    pub fn rate_weighted_sigma_inflation(&self) -> Option<f64> {
        if self.weighted_spread_from_nothing {
            Some(f64::INFINITY)
        } else if self.weight > 0.0 {
            Some(self.weighted_inflation / self.weight)
        } else {
            None
        }
    }
}

/// Whether `cov` is not PSD past [`REPAIR_TOLERANCE`], judged on its
/// correlation matrix so that round-off is relative to each channel.
fn needs_repair(cov: &RateCovariance) -> bool {
    let n = cov.n();
    let positive: Vec<usize> = (0..n).filter(|&i| cov.get(i, i) > 0.0).collect();
    // A PSD matrix with a zero diagonal has a zero row there, so any
    // covariance from such a channel, or a negative variance, is a repair,
    // and an exact one to state.
    let stated_without_variance = (0..n)
        .filter(|&i| cov.get(i, i) <= 0.0)
        .any(|i| cov.get(i, i) < 0.0 || (0..n).any(|j| j != i && cov.get(i, j) != 0.0));
    if stated_without_variance {
        return true;
    }
    let m = positive.len();
    if m < 2 {
        return false;
    }
    let scale: Vec<f64> = positive.iter().map(|&i| cov.get(i, i).sqrt()).collect();
    let mut r = vec![0.0; m * m];
    for (a, &i) in positive.iter().enumerate() {
        for (b, &j) in positive.iter().enumerate() {
            r[a * m + b] = if a == b {
                1.0
            } else {
                cov.get(i, j) / scale[a] / scale[b]
            };
        }
    }
    let (values, _) = jacobi_eigen(&r, m);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    min < -(m as f64) * REPAIR_TOLERANCE
}

/// A stable 32-bit hash of a nuclide name.
///
/// FNV-1a, spelled out rather than taken from `DefaultHasher`, whose output is
/// explicitly not stable across releases. The seed contract promises the same
/// answer on every platform and every build, and a hash that may change is not
/// compatible with that.
pub(crate) fn name_ordinal(name: &str) -> u32 {
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

    /// A populated set wide enough for every nuclide these tests name.
    fn everyone() -> HashSet<String> {
        ["X", "Y"].iter().map(|s| s.to_string()).collect()
    }

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
        assert!(s.repairs(0).is_empty());

        let f = &s.factors["Fe56"];
        for i in 0..2 {
            for j in 0..2 {
                let got: f64 = (0..2).map(|k| f.l[i * 2 + k] * f.l[j * 2 + k]).sum();
                assert!((got - c.get(i, j)).abs() < 1e-12, "({i},{j}) {got}");
            }
        }
    }

    /// A matrix that is not a covariance is repaired, and the repair is
    /// reported rather than absorbed, with what it did to each sigma.
    #[test]
    fn a_non_psd_matrix_is_clipped_and_recorded() {
        // Correlation of five: eigenvalues 0.06 and -0.04. Clipping keeps
        // 0.06 along (1, 1)/sqrt(2), so each diagonal becomes 0.03 against
        // the 0.01 evaluated, and the variance added is 0.04 on a trace of 0.02.
        let c = cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]);
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));
        let repairs = s.repairs(3);
        assert_eq!(repairs.len(), 1);
        let r = &repairs[0];
        assert_eq!((r.nuclide.as_str(), r.spectrum), ("X", 3));
        assert!((r.lambda_min + 0.04).abs() < 1e-12, "{}", r.lambda_min);
        assert!((r.lambda_max - 0.06).abs() < 1e-12, "{}", r.lambda_max);
        assert!(
            (r.clipped_fraction - 2.0).abs() < 1e-9,
            "{}",
            r.clipped_fraction
        );
        for ch in &r.channels {
            assert_eq!(ch.evaluated_variance, 0.01);
            assert!(
                (ch.evaluated_sigma().unwrap() - 0.1).abs() < 1e-12,
                "{ch:?}"
            );
            assert!((ch.sampled - 0.03_f64.sqrt()).abs() < 1e-12, "{ch:?}");
        }
    }

    /// Clipping only adds variance, and gives a spread to a channel the
    /// evaluation states as exactly zero.
    #[test]
    fn clipping_can_give_sigma_to_a_zero_diagonal() {
        let c = cov(&["a", "b"], vec![0.04, 0.01, 0.01, 0.0]);
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));
        let r = &s.repairs(0)[0];
        assert_eq!(r.channels[1].evaluated_sigma(), Some(0.0));
        assert!(r.channels[1].sampled > 0.0, "{:?}", r.channels[1]);
        assert!(r.channels[0].sampled >= r.channels[0].evaluated_sigma().unwrap());

        let mut report = SigmaReport::default();
        report.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("b", 1.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(report.worst_sigma_inflation, f64::INFINITY);
    }

    /// Unit-flux rates for one nuclide.
    fn unit_rates(nuclide: &str, kinds: &[(&str, f64)]) -> ReactionRates {
        HashMap::from([(
            nuclide.to_string(),
            kinds.iter().map(|(k, r)| (k.to_string(), *r)).collect(),
        )])
    }

    /// A negative folded diagonal is kept as the evaluation gives it, not
    /// read as a sigma of zero, and a spread the repair gives it is infinite
    /// inflation.
    #[test]
    fn a_negative_diagonal_is_kept_as_stated() {
        let c = cov(&["a", "b"], vec![0.04, 0.01, 0.01, -0.001]);
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));
        let ch = &s.repairs(0)[0].channels[1];
        assert_eq!(ch.evaluated_variance, -0.001);
        assert_eq!(ch.evaluated_sigma(), None);
        assert!(ch.sampled > 0.0, "{ch:?}");
        assert_eq!(ch.inflation(), f64::INFINITY);
    }

    /// A repaired channel with no rate, or on a spectrum the schedule never
    /// irradiates with, cannot move the result, so it stays in the record but
    /// not in the headline, and a repair with no such channel is not a gap.
    #[test]
    fn the_worst_inflation_counts_only_channels_a_draw_can_move() {
        let c = cov(&["a", "b"], vec![0.04, 0.01, 0.01, 0.0]);
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));

        let mut no_rate = SigmaReport::default();
        no_rate.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("b", 0.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(no_rate.repairs.len(), 1);
        assert_eq!(no_rate.repaired, BTreeSet::from(["X".to_string()]));
        assert!(no_rate.worst_sigma_inflation.is_finite());
        assert!(no_rate.worst_sigma_inflation > 0.0);

        let mut no_fluence = SigmaReport::default();
        no_fluence.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("b", 1.0)]),
            0.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(no_fluence.repairs.len(), 1);
        assert!(no_fluence.repaired.is_empty());
        assert_eq!(no_fluence.worst_sigma_inflation, 0.0);
    }

    /// A nuclide repaired on two spectra is one repaired nuclide with a
    /// record per spectrum.
    #[test]
    fn a_repair_on_two_spectra_is_one_repaired_nuclide() {
        let s = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]),
        )]));
        let r = unit_rates("X", &[("a", 1.0), ("b", 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &r, 1.0, &Default::default(), &everyone());
        report.add(1, &s, &r, 1.0, &Default::default(), &everyone());
        assert_eq!(report.repairs.len(), 2);
        assert_eq!(report.repaired, BTreeSet::from(["X".to_string()]));
    }

    /// Each spectrum is weighted by the fluence the schedule gives it, so a
    /// short, weak spectrum barely moves the headline.
    #[test]
    fn the_rate_weighted_inflation_weighs_spectra_by_fluence() {
        let repaired = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]),
        )]));
        let clean = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(&["a", "b"], vec![0.01, 0.0, 0.0, 0.01]),
        )]));
        let r = unit_rates("X", &[("a", 1.0), ("b", 1.0)]);
        let densities = HashMap::from([("X".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &clean, &r, 1.0e3, &densities, &everyone());
        report.add(1, &repaired, &r, 1.0, &densities, &everyone());

        // 2e3 channel-weights at no inflation against 2 at sqrt(3) - 1.
        let want = 2.0 * (3.0_f64.sqrt() - 1.0) / 2.002e3;
        let got = report.rate_weighted_sigma_inflation().unwrap();
        assert!((got - want).abs() < 1e-12, "{got} against {want}");
    }

    /// Only a correlation eigenvalue below `-n * REPAIR_TOLERANCE` counts,
    /// so a PSD matrix that comes back a few ulps negative is not a repair.
    #[test]
    fn the_repair_threshold_is_on_the_correlation_matrix() {
        // [[0.5, b], [b, 0.5]] is the correlation [[1, 2b], [2b, 1]] scaled,
        // with smallest eigenvalue 1 - 2b, against a threshold of -2e-12.
        // The n = 3 matrix is the correlation with every off-diagonal at r,
        // smallest eigenvalue 1 - r, against -3e-12. Each pair brackets its
        // own threshold within a factor of 1.5, so between them they pin both
        // the tolerance and the factor of n.
        let repaired = |c: RateCovariance| {
            !Sampler::new(&BTreeMap::from([("X".to_string(), c)]))
                .repairs(0)
                .is_empty()
        };
        let two = |lambda_min: f64| {
            let b = 0.5 - lambda_min / 2.0;
            cov(&["a", "b"], vec![0.5, b, b, 0.5])
        };
        let three = |lambda_min: f64| {
            let r = 1.0 - lambda_min;
            cov(&["a", "b", "c"], vec![1.0, r, r, r, 1.0, r, r, r, 1.0])
        };
        assert!(!repaired(two(-1.5e-12)));
        assert!(repaired(two(-3.0e-12)));
        assert!(!repaired(three(-2.5e-12)));
        assert!(repaired(three(-4.5e-12)));

        // Rank one, so singular as evaluated: PSD, whatever the round-off,
        // and still so when the channels span twenty orders in variance. The
        // last carries a small channel in the null space, where the round-off
        // of a decomposition of C itself, about eps * lambda_max, is 5e-11 of
        // that channel's variance.
        for v in [
            vec![0.1, 0.2, 0.3],
            vec![3.0e8, 0.1, 0.002],
            vec![-0.527, -0.01145, -1.6e-5, -0.2358],
        ] {
            let n = v.len();
            let kinds: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            let kinds: Vec<&str> = kinds.iter().map(String::as_str).collect();
            let rank_one = cov(&kinds, (0..n * n).map(|k| v[k / n] * v[k % n]).collect());
            assert!(
                Sampler::new(&BTreeMap::from([("X".to_string(), rank_one)]))
                    .repairs(0)
                    .is_empty(),
                "{v:?}"
            );
        }
    }

    /// A channel stated at zero variance with no covariance to any other is a
    /// PSD zero row, not a repair, and the rest of the matrix is judged
    /// without it.
    #[test]
    fn a_zero_channel_with_no_covariance_is_not_a_repair() {
        let c = cov(
            &["a", "zero", "b"],
            vec![
                0.04, 0.0, 0.01, //
                0.0, 0.0, 0.0, //
                0.01, 0.0, 0.09,
            ],
        );
        assert!(Sampler::new(&BTreeMap::from([("X".to_string(), c)]))
            .repairs(0)
            .is_empty());
    }

    /// A huge variance on one channel cannot hide a real repair on the
    /// others: the test is on the correlation matrix, not against the largest
    /// eigenvalue of the covariance, which here is 1e17 against a clipped
    /// -0.04.
    #[test]
    fn a_huge_channel_does_not_hide_a_repair_beside_it() {
        let c = cov(
            &["wide", "a", "b"],
            vec![
                1.0e17, 0.0, 0.0, //
                0.0, 0.01, 0.05, //
                0.0, 0.05, 0.01,
            ],
        );
        let s = Sampler::new(&BTreeMap::from([("X".to_string(), c)]));
        let repairs = s.repairs(0);
        assert_eq!(repairs.len(), 1);
        let r = &repairs[0];
        assert!((r.lambda_min + 0.04).abs() < 1e-12, "{}", r.lambda_min);
        assert_eq!(r.lambda_max, 1.0e17);
        assert_eq!(r.channels[0].sampled, 1.0e17_f64.sqrt());
        for ch in &r.channels[1..] {
            assert!((ch.sampled - 0.03_f64.sqrt()).abs() < 1e-12, "{ch:?}");
        }

        let mut report = SigmaReport::default();
        report.add(
            0,
            &s,
            &unit_rates("X", &[("wide", 1.0), ("a", 1.0), ("b", 1.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        let want = 0.03_f64.sqrt() / 0.1 - 1.0;
        assert!(
            (report.worst_sigma_inflation - want).abs() < 1e-12,
            "{}",
            report.worst_sigma_inflation
        );
    }

    /// A nuclide the material cannot populate has no repair record, no weight
    /// and no place in the wide-sigma lists, and is named instead as a repair
    /// outside the bound with its wide channels beside it.
    #[test]
    fn a_nuclide_the_material_cannot_populate_is_named_outside_the_bound() {
        let s = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(&["a", "b"], vec![4.0, 50.0, 50.0, 400.0]),
        )]));
        let rates = unit_rates("X", &[("a", 1.0), ("b", 1.0)]);
        let densities = HashMap::from([("X".to_string(), 1.0)]);

        let mut reported = SigmaReport::default();
        reported.add(0, &s, &rates, 1.0, &densities, &everyone());
        assert_eq!(reported.repairs.len(), 1);
        assert_eq!(reported.sigma_at_least_ten.len(), 1);

        // Its wide channels are named outside the bound, since a replica's
        // draw on such a channel can populate the nuclide.
        let mut unpopulated = SigmaReport::default();
        unpopulated.add(0, &s, &rates, 1.0, &densities, &Default::default());
        assert_eq!(
            unpopulated,
            SigmaReport {
                repaired_outside_bound: BTreeSet::from(["X".to_string()]),
                sigma_at_least_one_outside_bound: BTreeMap::from([
                    (("X".to_string(), "a".to_string()), 2.0),
                    (("X".to_string(), "b".to_string()), 20.0),
                ]),
                ..Default::default()
            }
        );
        assert!(reported.sigma_at_least_one_outside_bound.is_empty());
        assert!(reported.repaired_outside_bound.is_empty());
    }

    /// A wide channel with a small rate beside a repair on the channels that
    /// carry the reactions does not drown the repair out: the weight is on
    /// each channel's inflation, not on its sigma.
    #[test]
    fn a_wide_channel_does_not_mask_a_repair_on_the_dominant_ones() {
        let s = Sampler::new(&BTreeMap::from([
            (
                "X".to_string(),
                cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]),
            ),
            ("Y".to_string(), cov(&["a"], vec![1.0e6])),
        ]));
        let rates: ReactionRates = HashMap::from([
            (
                "X".to_string(),
                HashMap::from([("a".to_string(), 1.0), ("b".to_string(), 1.0)]),
            ),
            ("Y".to_string(), HashMap::from([("a".to_string(), 0.01)])),
        ]);
        let densities = HashMap::from([("X".to_string(), 1.0), ("Y".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());

        // 99.5% of the weight on channels widened by sqrt(3) - 1.
        let want = 2.0 * (3.0_f64.sqrt() - 1.0) / 2.01;
        let got = report.rate_weighted_sigma_inflation().unwrap();
        assert!((got - want).abs() < 1e-12, "{got} against {want}");
        assert!(got > 0.7);
    }

    /// A weighted channel the evaluation gives no sigma but the repair gives
    /// a spread makes the weighted headline infinite rather than ignored.
    #[test]
    fn a_spread_from_nothing_makes_the_weighted_inflation_infinite() {
        let s = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.0]),
        )]));
        let rates = unit_rates("X", &[("a", 1.0), ("b", 1.0)]);
        let densities = HashMap::from([("X".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());
        assert!(report.repairs[0].channels[1].sampled > 0.0);
        assert_eq!(report.rate_weighted_sigma_inflation(), Some(f64::INFINITY));
    }

    /// The rate-weighted headline weighs each channel by its rate times its
    /// parent's density, over every sampled channel.
    #[test]
    fn the_rate_weighted_inflation_counts_every_sampled_channel() {
        let s = Sampler::new(&BTreeMap::from([
            (
                "X".to_string(),
                cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]),
            ),
            ("Y".to_string(), cov(&["a"], vec![0.04])),
        ]));
        let rates: ReactionRates = HashMap::from([
            (
                "X".to_string(),
                HashMap::from([("a".to_string(), 1.0), ("b".to_string(), 1.0)]),
            ),
            ("Y".to_string(), HashMap::from([("a".to_string(), 2.0)])),
        ]);
        let densities = HashMap::from([("X".to_string(), 1.0), ("Y".to_string(), 0.5)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());

        // X: two channels at 0.1 evaluated, sqrt(0.03) sampled, weight 1 each.
        // Y: one channel at 0.2 and not widened, weight 2 * 0.5.
        let want = 2.0 * (3.0_f64.sqrt() - 1.0) / 3.0;
        let got = report.rate_weighted_sigma_inflation().unwrap();
        assert!((got - want).abs() < 1e-12, "{got} against {want}");
        assert!((report.worst_sigma_inflation - (0.03_f64.sqrt() / 0.1 - 1.0)).abs() < 1e-12);
        assert_eq!(report.repairs.len(), 1);

        // No repair anywhere reads as exactly zero, not as round-off.
        let mut clean = SigmaReport::default();
        let y = Sampler::new(&BTreeMap::from([(
            "Y".to_string(),
            cov(&["a"], vec![0.04]),
        )]));
        clean.add(0, &y, &rates, 1.0, &densities, &everyone());
        assert_eq!(clean.rate_weighted_sigma_inflation(), Some(0.0));
        assert_eq!(clean.worst_sigma_inflation, 0.0);
    }

    /// Channels evaluated at a relative sigma of one or ten and above are named,
    /// and a channel with no rate or a zero one, which a draw cannot move, is
    /// not.
    #[test]
    fn wide_sigmas_are_named() {
        let s = Sampler::new(&BTreeMap::from([(
            "X".to_string(),
            cov(
                &["a", "b", "c", "d"],
                vec![
                    4.0, 0.0, 0.0, 0.0, //
                    0.0, 400.0, 0.0, 0.0, //
                    0.0, 0.0, 0.25, 0.0, //
                    0.0, 0.0, 0.0, 900.0,
                ],
            ),
        )]));
        let rates: ReactionRates = HashMap::from([(
            "X".to_string(),
            HashMap::from([
                ("a".to_string(), 1.0),
                ("b".to_string(), 1.0),
                ("c".to_string(), 1.0),
                ("d".to_string(), 0.0),
            ]),
        )]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &Default::default(), &everyone());
        let key = |k: &str| ("X".to_string(), k.to_string());
        assert_eq!(
            report
                .sigma_at_least_one
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec![key("a"), key("b")]
        );
        assert_eq!(
            report
                .sigma_at_least_ten
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec![key("b")]
        );
        assert!((report.sigma_at_least_ten[&key("b")] - 20.0).abs() < 1e-12);

        // Evaluated at 0.9 and sampled at sqrt(1.405) after the repair: the
        // list is of what the evaluation states, and the widening is in the
        // repair record instead.
        let y = Sampler::new(&BTreeMap::from([(
            "Y".to_string(),
            cov(&["a", "b"], vec![0.81, 2.0, 2.0, 0.81]),
        )]));
        let mut repaired = SigmaReport::default();
        repaired.add(
            0,
            &y,
            &unit_rates("Y", &[("a", 1.0), ("b", 1.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert!(repaired.repairs[0].channels[0].sampled > 1.0);
        assert!(repaired.sigma_at_least_one.is_empty());
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
    fn a_wide_sigma_never_draws_a_negative_rate() {
        // The case that used to floor: a relative variance of 9, so a relative
        // sigma of 3. A Gaussian multiplier `1 + delta` is negative about a
        // third of the time at that width; an exponential one never is, which
        // is why there is no floor left to count.
        let c = cov(&["(n,gamma)"], vec![9.0]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 1.0e-8)]),
        )]);

        let mut sampled = 0;
        for k in 0..2000 {
            let (out, n) = s.perturb(&rates, 9, k);
            sampled += n;
            assert!(out["Fe56"]["(n,gamma)"] > 0.0, "a rate must stay positive");
        }
        assert_eq!(sampled, 2000, "one draw per channel per replica");
    }

    #[test]
    fn the_ensemble_mean_is_the_nominal_rate() {
        // The property the old form lost. Truncating at zero threw away the
        // negative tail and kept its probability mass at the boundary, which
        // pushed the mean up.
        // A relative variance of 0.25, so a relative sigma of 0.5: `cov` takes
        // the covariance matrix, not the standard deviations.
        let c = cov(&["(n,gamma)"], vec![0.25]);
        let s = Sampler::new(&BTreeMap::from([("Fe56".to_string(), c)]));
        let nominal = 1.0e-8;
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), nominal)]),
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
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 2.5)]),
        )]);
        let (out, sampled) = s.perturb(&rates, 1, 0);
        assert_eq!(out, rates);
        assert_eq!(sampled, 0);
    }
}
