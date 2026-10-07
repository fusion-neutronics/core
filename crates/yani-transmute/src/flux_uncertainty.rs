//! Propagate the uncertainty on a supplied flux spectrum to the inventory.
//!
//! `Material.transmute` takes the flux as exact. It usually is not: it arrives
//! from a Monte Carlo tally with a standard deviation per bin, and that error
//! goes straight into every reaction rate and so into the whole inventory.
//!
//! # It costs one dot product per replica
//!
//! The rate is linear in the flux,
//!
//! ```text
//! R_i = 1e-24 * sum_g  sigma_{i,g} * phi_g
//! ```
//!
//! and the collapse in [`crate::multigroup`] already computes each
//! `sigma_{i,g} * phi_g` on its way to that sum. Keeping the terms instead of
//! discarding them makes a perturbed rate
//!
//! ```text
//! R_i' = 1e-24 * sum_g  sigma_{i,g} * phi_g * (1 + delta_g)
//! ```
//!
//! which is exact rather than linearized, because the relationship IS linear.
//! Nothing is re-collapsed and no cross section is touched twice.
//!
//! # Why one draw is shared by every nuclide
//!
//! Flux bins are shared. Perturbing bin `g` moves every reaction in every
//! nuclide that sees that bin, together. That is the opposite of the MF=33
//! covariance in [`crate::covariance_fold`], which never crosses evaluations
//! and so factorizes per nuclide.
//!
//! So the draw is per REPLICA, not per nuclide, and the correlation across the
//! whole chain comes out for free. Expressing it as a covariance matrix instead
//! would need one dense block over every edge in the chain, which is the thing
//! the per-nuclide design exists to avoid.
//!
//! # Absent is not zero
//!
//! A spectrum from a published reference set (the FISPACT reference input
//! spectra, say) carries no stated error at all, and that has to stay
//! distinguishable from a spectrum measured to be exact. A caller who supplies
//! no sigma gets no flux contribution and a report saying so.

use std::collections::HashMap;

use yani::ReactionRates;

/// Per-group contributions to each reaction rate, in 1/s.
///
/// `terms[nuclide][kind][g]` is `1e-24 * sigma_g * phi_g` for group `g`, so the
/// nominal rate is the sum over `g` and a perturbed one is the sum weighted by
/// `(1 + delta_g)`.
///
/// Built only when flux uncertainty is asked for. It is the same size as the
/// rate map times the group count, which for a 709-group structure and a few
/// hundred activation channels is tens of MB, and there is no reason to pay
/// that on the default path.
pub type PerGroupRates = HashMap<String, HashMap<String, Vec<f64>>>;

/// What the flux uncertainty could and could not be applied to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FluxCoverage {
    /// Spectra that carried a per-bin sigma.
    pub spectra_with_sigma: usize,
    /// Spectra that did not, so contributed no flux uncertainty.
    pub spectra_without_sigma: usize,
    pub bins_sampled: usize,
    /// Spectra, by index, whose stated covariance is not a lognormal's, with
    /// how far the sampled one is from it (see
    /// [`crate::covariance_sample::LognormalLimit`]). A per-bin standard
    /// deviation always is one.
    pub lognormal_not_carried:
        std::collections::BTreeMap<usize, crate::covariance_sample::LognormalLimit>,
}

impl FluxCoverage {
    pub fn has_gaps(&self) -> bool {
        self.spectra_without_sigma > 0
    }
}

/// Turn an absolute per-bin standard deviation into the relative form the
/// perturbation multiplies by.
///
/// A bin with zero flux has no relative error to state, and one with a zero
/// sigma is a bin the caller says is exact; both come out as zero, which
/// perturbs nothing. Returns `None` when the lengths disagree, which is a
/// caller error rather than something to paper over: a sigma vector that does
/// not line up with the flux is not a sigma for that flux.
pub fn relative_std_dev(flux: &[f64], std_dev: &[f64]) -> Option<Vec<f64>> {
    if flux.len() != std_dev.len() {
        return None;
    }
    Some(
        flux.iter()
            .zip(std_dev)
            .map(|(f, s)| if *f > 0.0 { (s / f).abs() } else { 0.0 })
            .collect(),
    )
}

/// A spectrum's stated error, relative to its own values.
///
/// A Monte Carlo spectrum's bins are scored by the same histories and are
/// correlated. A per-bin standard deviation treats them as independent, which
/// understates the error of anything summed over a band whose bins move
/// together, and a band is exactly what a reaction rate sums over. The
/// covariance form keeps the correlations.
#[derive(Clone, Debug, PartialEq)]
pub enum FluxError {
    /// Per-bin relative standard deviation, bins independent.
    RelativeStdDev(Vec<f64>),
    /// Full relative covariance, `C_ij / (phi_i phi_j)`, carried as the
    /// lognormal that matches it.
    RelativeCovariance(RelativeFluxCovariance),
}

impl FluxError {
    /// Number of bins it describes.
    pub fn len(&self) -> usize {
        match self {
            FluxError::RelativeStdDev(v) => v.len(),
            FluxError::RelativeCovariance(c) => c.n,
        }
    }

    /// Whether it describes no bins.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An exact identity for keying a shared collapse.
    pub(crate) fn key_bits(&self) -> Vec<u64> {
        match self {
            FluxError::RelativeStdDev(v) => std::iter::once(0u64)
                .chain(v.iter().map(|x| x.to_bits()))
                .collect(),
            FluxError::RelativeCovariance(c) => std::iter::once(1u64)
                .chain(c.log_factor.iter().map(|x| x.to_bits()))
                .chain(c.half_log_variance.iter().map(|x| x.to_bits()))
                .collect(),
        }
    }
}

/// A spectrum's relative covariance, as the lognormal that carries it.
///
/// Each bin's flux is drawn as a factor `exp(y_i - Σ_N,ii / 2)` with
/// `y ~ N(0, Σ_N)` and `Σ_N = ln(1 + C)` elementwise, the transform the cross
/// sections use: every factor has mean one, `Cov(m_i, m_j) = C_ij` wherever a
/// lognormal can carry it, and no sampled flux can go negative.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativeFluxCovariance {
    n: usize,
    /// Factor of `Σ_N`, row-major `n x n`, with `L L^T = Σ_N`. Lower
    /// triangular where `Σ_N` is positive semi-definite.
    log_factor: Vec<f64>,
    /// `Σ_N,ii / 2` of the matrix sampled.
    half_log_variance: Vec<f64>,
    /// How far the sampled covariance is from the stated one, where the
    /// stated one is not a lognormal's.
    lognormal_limit: Option<crate::covariance_sample::LognormalLimit>,
}

/// Cholesky factor of a symmetric `n x n` matrix, row-major, skipping a zero
/// pivot so a singular matrix factors, or the bin whose remaining variance is
/// negative past round-off.
fn cholesky(matrix: &[f64], n: usize) -> Result<Vec<f64>, usize> {
    let scale = (0..n).map(|i| matrix[i * n + i]).fold(0.0_f64, f64::max);
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = matrix[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d < -1.0e-9 * scale {
            return Err(j);
        }
        if d <= 1.0e-12 * scale {
            continue;
        }
        let pivot = d.sqrt();
        l[j * n + j] = pivot;
        for i in (j + 1)..n {
            let mut v = matrix[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = v / pivot;
        }
    }
    Ok(l)
}

impl RelativeFluxCovariance {
    /// From the absolute covariance of the spectrum's own values, `flux`.
    ///
    /// Checked rather than trusted, since a covariance that is not one would
    /// sample nonsense without complaint: it must be square with one row per
    /// bin, symmetric, and positive semi-definite. A bin with no flux has no
    /// relative error to state and is taken as exact, as in
    /// [`relative_std_dev`].
    pub fn from_absolute(flux: &[f64], covariance: &[Vec<f64>]) -> Result<Self, String> {
        let n = flux.len();
        if covariance.len() != n || covariance.iter().any(|row| row.len() != n) {
            return Err(format!(
                "the flux covariance must be {n} x {n}, one row and column per spectrum bin"
            ));
        }
        let scale = (0..n)
            .map(|i| covariance[i][i].abs())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        for (i, row) in covariance.iter().enumerate() {
            if !row[i].is_finite() || row[i] < 0.0 {
                return Err(format!(
                    "flux covariance diagonal entry {i} is {}; a variance is finite and \
                     non-negative",
                    row[i]
                ));
            }
            for (j, &a) in row.iter().enumerate().take(i) {
                let b = covariance[j][i];
                if !a.is_finite() || (a - b).abs() > 1.0e-9 * scale {
                    return Err(format!(
                        "the flux covariance is not symmetric: entry ({i}, {j}) is {a} and \
                         ({j}, {i}) is {b}"
                    ));
                }
            }
        }
        // Relative covariance, symmetrized.
        let rel = |i: usize, j: usize| -> f64 {
            if flux[i] > 0.0 && flux[j] > 0.0 {
                0.5 * (covariance[i][j] + covariance[j][i]) / (flux[i] * flux[j])
            } else {
                0.0
            }
        };
        let relative: Vec<f64> = (0..n * n).map(|ij| rel(ij / n, ij % n)).collect();
        if let Err(j) = cholesky(&relative, n) {
            return Err(format!(
                "the flux covariance is not positive semi-definite: bin {j} has a \
                 negative remaining variance"
            ));
        }
        // The lognormal carrying it. Where `Σ_N` factors as it stands, the
        // stated covariance is carried exactly and the factor is triangular,
        // so a diagonal covariance draws bin for bin as the per-bin standard
        // deviation does. Where it does not, or an entry has no logarithm,
        // the nearest lognormal is sampled and how far it is recorded.
        let (log, substituted) = crate::covariance_sample::log_covariance(&relative, n);
        let (log_factor, sampled_log) = match (substituted, cholesky(&log, n)) {
            (false, Ok(l)) => (l, None),
            _ => {
                let (l, _, sampled, _) = crate::covariance_sample::clipped_factor(&log, n, false);
                (l, Some(sampled))
            }
        };
        let (half_log_variance, lognormal_limit) = match sampled_log {
            None => ((0..n).map(|i| 0.5 * log[i * n + i]).collect(), None),
            Some(s) => {
                let sampled: Vec<f64> = s.iter().map(|v| v.exp_m1()).collect();
                (
                    (0..n).map(|i| 0.5 * s[i * n + i]).collect(),
                    crate::covariance_sample::lognormal_limit(&relative, &sampled, n),
                )
            }
        };
        Ok(RelativeFluxCovariance {
            n,
            log_factor,
            half_log_variance,
            lognormal_limit,
        })
    }
}

/// One replica's per-bin flux perturbations, `delta_g`.
///
/// Drawn from the spectrum's own relative error, once per replica and shared
/// across every nuclide: independently per bin for a standard deviation, and
/// through the factor for a covariance, so correlated bins move together.
/// Each bin's flux is scaled by a lognormal factor with mean one, so
/// `delta_g` is that factor minus one: never below `-1`, so a sampled flux
/// cannot go negative even where a bin's relative error exceeds 100%, normal
/// in the tail of a tally, and with no floor to bias its mean.
pub fn flux_deviates(
    error: &FluxError,
    base_seed: u64,
    replica: u64,
    spectrum_index: usize,
    coverage: &mut FluxCoverage,
) -> Vec<f64> {
    // A stream of its own, keyed on the spectrum so a schedule carrying a DD
    // and a DT campaign perturbs them independently, and so adding a spectrum
    // does not shift the one before it.
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    let seed = yamc_rng::secondary_seed(replica_seed, FLUX_STREAM ^ spectrum_index as u32);
    let mut state = yamc_rng::expand_seed(seed);

    let n = error.len();
    let z = crate::covariance_sample::standard_normals(&mut state, n);
    coverage.bins_sampled += n;
    match error {
        FluxError::RelativeStdDev(relative) => relative
            .iter()
            .zip(&z)
            .map(|(sigma, z)| {
                if *sigma > 0.0 {
                    crate::covariance_sample::lognormal_multiplier(*z, *sigma) - 1.0
                } else {
                    0.0
                }
            })
            .collect(),
        FluxError::RelativeCovariance(c) => {
            if let Some(limit) = c.lognormal_limit {
                coverage.lognormal_not_carried.insert(spectrum_index, limit);
            }
            (0..n)
                .map(|i| {
                    let y: f64 = c.log_factor[i * n..(i + 1) * n]
                        .iter()
                        .zip(&z)
                        .map(|(l, z)| l * z)
                        .sum();
                    (y - c.half_log_variance[i]).exp_m1()
                })
                .collect()
        }
    }
}

/// Keeps the flux stream clear of the per-nuclide cross-section streams, which
/// are keyed on a hash of the nuclide name.
pub(crate) const FLUX_STREAM: u32 = 0xF10D_5EED;

/// Apply one replica's flux perturbation to a set of unit-flux rates.
///
/// The perturbation is applied as a factor on the rate, not as a replacement
/// for it: the terms say how the rate is distributed over the bins, and the
/// rate itself stays the one the collapse produced. So an unperturbed replica
/// reproduces the nominal rate bit for bit rather than to rounding, and a
/// weighting the terms do not describe cannot be silently substituted for the
/// one that drove the nominal run.
///
/// Rates with no per-group terms are passed through unchanged, which is what a
/// nuclide whose cross sections were never collapsed against this spectrum
/// looks like, and so is a channel whose rate comes from the branching overlay
/// rather than from a group average.
pub fn perturb_rates(
    rates: &ReactionRates,
    per_group: &PerGroupRates,
    delta: &[f64],
) -> ReactionRates {
    let mut out = rates.clone();
    for (nuclide, kinds) in per_group {
        let Some(nuclide_rates) = out.get_mut(nuclide) else {
            continue;
        };
        for (kind, terms) in kinds {
            let Some(rate) = nuclide_rates.get_mut(kind) else {
                continue;
            };
            let nominal: f64 = terms.iter().sum();
            if nominal <= 0.0 {
                continue;
            }
            let perturbed: f64 = terms
                .iter()
                .zip(delta)
                .map(|(term, d)| term * (1.0 + d))
                .sum();
            *rate *= perturbed / nominal;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates(v: f64) -> ReactionRates {
        HashMap::from([("Li6".to_string(), HashMap::from([("(n,t)".to_string(), v)]))])
    }

    fn per_group(terms: Vec<f64>) -> PerGroupRates {
        HashMap::from([(
            "Li6".to_string(),
            HashMap::from([("(n,t)".to_string(), terms)]),
        )])
    }

    #[test]
    fn a_relative_sigma_is_the_absolute_one_over_the_flux() {
        let r = relative_std_dev(&[100.0, 50.0], &[5.0, 10.0]).expect("same length");
        assert_eq!(r, vec![0.05, 0.2]);
    }

    /// A bin with no flux has no relative error to state.
    #[test]
    fn a_zero_flux_bin_gets_no_relative_error() {
        let r = relative_std_dev(&[0.0, 50.0], &[5.0, 10.0]).expect("same length");
        assert_eq!(r, vec![0.0, 0.2]);
    }

    /// A sigma that does not line up with the flux is not a sigma for it.
    #[test]
    fn a_length_mismatch_is_refused_rather_than_truncated() {
        assert_eq!(relative_std_dev(&[1.0, 2.0], &[0.1]), None);
    }

    /// Zero perturbation reproduces the nominal rate exactly.
    #[test]
    fn an_unperturbed_replica_reproduces_the_nominal_rate() {
        let terms = vec![0.25, 0.5, 0.25];
        let nominal: f64 = terms.iter().sum();
        let out = perturb_rates(&rates(nominal), &per_group(terms), &[0.0, 0.0, 0.0]);
        assert_eq!(out["Li6"]["(n,t)"], nominal);
    }

    /// And reproduces it even when the terms describe a different weighting.
    ///
    /// The terms are a shape, not a substitute for the rate. Assigning their
    /// sum instead of scaling by it is how a shielded rate used to be replaced
    /// by a dilute one at zero perturbation.
    #[test]
    fn an_unperturbed_replica_does_not_replace_the_rate_with_the_terms() {
        // A rate 30% below what the terms sum to, which is roughly what
        // self-shielding does to a resonance absorber's capture.
        let shielded = 0.7;
        let out = perturb_rates(
            &rates(shielded),
            &per_group(vec![0.25, 0.5, 0.25]),
            &[0.0, 0.0, 0.0],
        );
        assert_eq!(out["Li6"]["(n,t)"], shielded);
    }

    /// A term set summing to nothing has no shape to perturb along.
    #[test]
    fn a_channel_with_no_rate_in_any_bin_is_left_alone() {
        let out = perturb_rates(
            &rates(1.0),
            &per_group(vec![0.0, 0.0, 0.0]),
            &[0.5, 0.5, 0.5],
        );
        assert_eq!(out["Li6"]["(n,t)"], 1.0);
    }

    /// The rate is linear in the flux, so a uniform perturbation scales it.
    #[test]
    fn a_uniform_perturbation_scales_the_rate() {
        let terms = vec![0.25, 0.5, 0.25];
        let out = perturb_rates(&rates(1.0), &per_group(terms), &[0.1, 0.1, 0.1]);
        assert!((out["Li6"]["(n,t)"] - 1.1).abs() < 1e-12);
    }

    /// Perturbing one bin moves the rate by that bin's share, not by the whole.
    #[test]
    fn one_bin_moves_the_rate_by_its_own_share() {
        // The middle bin carries half the rate, so +20% on it is +10% overall.
        let out = perturb_rates(
            &rates(1.0),
            &per_group(vec![0.25, 0.5, 0.25]),
            &[0.0, 0.2, 0.0],
        );
        assert!((out["Li6"]["(n,t)"] - 1.1).abs() < 1e-12);
    }

    #[test]
    fn deviates_are_a_pure_function_of_seed_replica_and_spectrum() {
        let mut c = FluxCoverage::default();
        let rel = FluxError::RelativeStdDev(vec![0.05, 0.10, 0.20]);
        let a = flux_deviates(&rel, 42, 7, 0, &mut c);
        let b = flux_deviates(&rel, 42, 7, 0, &mut c);
        assert_eq!(a, b, "same inputs, same answer");
        assert_ne!(a, flux_deviates(&rel, 42, 8, 0, &mut c), "replica matters");
        assert_ne!(a, flux_deviates(&rel, 43, 7, 0, &mut c), "seed matters");
        // A DD and a DT campaign in one schedule must move independently.
        assert_ne!(a, flux_deviates(&rel, 42, 7, 1, &mut c), "spectrum matters");
    }

    /// A bin the caller says is exact is not perturbed.
    #[test]
    fn a_zero_sigma_bin_does_not_move() {
        let mut c = FluxCoverage::default();
        let d = flux_deviates(&FluxError::RelativeStdDev(vec![0.0, 0.1]), 1, 0, 0, &mut c);
        assert_eq!(d[0], 0.0);
    }

    /// A relative error above 100%, normal in the tail of a tally, still
    /// never samples a negative flux, and with no floor the factor keeps its
    /// mean of one.
    #[test]
    fn a_wide_flux_error_never_goes_negative_and_keeps_its_mean() {
        let mut c = FluxCoverage::default();
        let draws = 20_000;
        let mut sum = 0.0;
        for k in 0..draws {
            // 300% relative.
            let d = flux_deviates(&FluxError::RelativeStdDev(vec![3.0]), 9, k, 0, &mut c);
            assert!(d[0] > -1.0, "a flux bin cannot go negative");
            sum += 1.0 + d[0];
        }
        let mean = sum / draws as f64;
        // The factor's sigma is 3, so the mean's standard error is ~0.02.
        assert!((mean - 1.0).abs() < 0.15, "mean factor {mean}");
        assert_eq!(c.bins_sampled, draws as usize);
        assert!(c.lognormal_not_carried.is_empty());
    }

    /// Two fully correlated bins with different sigmas are not a lognormal's:
    /// the nearest one is sampled, and how far it is gets recorded.
    #[test]
    fn a_covariance_no_lognormal_carries_is_recorded() {
        let flux = [1.0, 1.0];
        let (a, b) = (0.5, 2.0);
        let cov = vec![vec![a * a, a * b], vec![a * b, b * b]];
        let rel = RelativeFluxCovariance::from_absolute(&flux, &cov).unwrap();
        let limit = rel.lognormal_limit.expect("not a lognormal's");
        assert!(limit.cells > 0);
        assert!(limit.largest_correlation_change > 0.0 || limit.largest_sigma_change > 0.0);
        let mut c = FluxCoverage::default();
        let d = flux_deviates(&FluxError::RelativeCovariance(rel), 1, 0, 3, &mut c);
        assert!(d.iter().all(|d| *d > -1.0));
        assert_eq!(c.lognormal_not_carried.get(&3), Some(&limit));
    }

    /// An ordinary covariance is carried exactly, so nothing is recorded.
    #[test]
    fn an_ordinary_covariance_records_nothing() {
        let flux = [1.0, 2.0];
        let cov = vec![vec![0.01, 0.005], vec![0.005, 0.04]];
        let rel = RelativeFluxCovariance::from_absolute(&flux, &cov).unwrap();
        assert_eq!(rel.lognormal_limit, None);
    }

    /// Over many replicas the spread matches the sigma it was built from.
    #[test]
    fn the_sampled_spread_matches_the_stated_flux_error() {
        let mut c = FluxCoverage::default();
        let stated = vec![0.05, 0.10];
        let rel = FluxError::RelativeStdDev(stated.clone());
        let n = 20_000;
        let (mut s, mut sq) = ([0.0; 2], [0.0; 2]);
        for k in 0..n {
            let d = flux_deviates(&rel, 3, k, 0, &mut c);
            for i in 0..2 {
                s[i] += d[i];
                sq[i] += d[i] * d[i];
            }
        }
        let nf = n as f64;
        for (i, want) in stated.iter().enumerate() {
            let mean = s[i] / nf;
            let sd = (sq[i] / nf - mean * mean).sqrt();
            assert!(mean.abs() < 0.01, "mean {mean} for bin {i}");
            assert!((sd - want).abs() < 0.01, "sd {sd} != {want} for bin {i}");
        }
    }

    /// A covariance with no off-diagonal terms is the per-bin standard
    /// deviation it contains, draw for draw.
    #[test]
    fn a_diagonal_covariance_draws_as_the_standard_deviation_does() {
        let flux = [10.0, 20.0, 40.0];
        let sd = [0.5, 3.0, 2.0];
        let cov: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| if i == j { sd[i] * sd[i] } else { 0.0 })
                    .collect()
            })
            .collect();
        let as_cov = FluxError::RelativeCovariance(
            RelativeFluxCovariance::from_absolute(&flux, &cov).unwrap(),
        );
        let as_sd = FluxError::RelativeStdDev(relative_std_dev(&flux, &sd).unwrap());
        let mut c = FluxCoverage::default();
        for k in 0..20 {
            let a = flux_deviates(&as_cov, 5, k, 0, &mut c);
            let b = flux_deviates(&as_sd, 5, k, 0, &mut c);
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-12, "{x} vs {y}");
            }
        }
    }

    /// Fully correlated bins move together, which independent standard
    /// deviations cannot express, and the draws reproduce the correlation.
    #[test]
    fn correlated_bins_move_together() {
        let flux = [1.0, 2.0];
        // 10% on each, correlation 0.8.
        let (s0, s1) = (0.1, 0.2);
        let cov = vec![vec![s0 * s0, 0.8 * s0 * s1], vec![0.8 * s0 * s1, s1 * s1]];
        let e = FluxError::RelativeCovariance(
            RelativeFluxCovariance::from_absolute(&flux, &cov).unwrap(),
        );
        let mut c = FluxCoverage::default();
        let draws: Vec<Vec<f64>> = (0..20000)
            .map(|k| flux_deviates(&e, 1, k, 0, &mut c))
            .collect();
        let n = draws.len() as f64;
        let m = |i: usize| draws.iter().map(|d| d[i]).sum::<f64>() / n;
        let v = |i: usize, j: usize| {
            let (mi, mj) = (m(i), m(j));
            draws.iter().map(|d| (d[i] - mi) * (d[j] - mj)).sum::<f64>() / (n - 1.0)
        };
        let corr = v(0, 1) / (v(0, 0) * v(1, 1)).sqrt();
        assert!((v(0, 0).sqrt() / 0.1 - 1.0).abs() < 0.03);
        assert!((v(1, 1).sqrt() / 0.1 - 1.0).abs() < 0.03);
        assert!((corr - 0.8).abs() < 0.02, "correlation {corr}");
    }

    #[test]
    fn a_covariance_that_is_not_one_is_refused() {
        let flux = [1.0, 1.0];
        let asymmetric = vec![vec![1.0, 0.5], vec![0.2, 1.0]];
        assert!(RelativeFluxCovariance::from_absolute(&flux, &asymmetric)
            .unwrap_err()
            .contains("not symmetric"));
        let indefinite = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(RelativeFluxCovariance::from_absolute(&flux, &indefinite)
            .unwrap_err()
            .contains("not positive semi-definite"));
        let wrong_shape = vec![vec![1.0]];
        assert!(RelativeFluxCovariance::from_absolute(&flux, &wrong_shape)
            .unwrap_err()
            .contains("2 x 2"));
        // Singular but semi-definite (fully correlated) is a covariance.
        let singular = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(RelativeFluxCovariance::from_absolute(&flux, &singular).is_ok());
    }
}
