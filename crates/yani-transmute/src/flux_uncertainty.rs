//! Propagate the uncertainty on a supplied flux spectrum to the inventory.
//!
//! `Material.transmute` takes the flux as exact. It usually is not: it arrives
//! from a Monte Carlo tally with a standard deviation per bin, and that error
//! goes straight into every reaction rate and so into the whole inventory
//! (issue #559).
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
    /// Bins whose sampled flux went negative and was floored at zero.
    pub bins_floored: usize,
    pub bins_sampled: usize,
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

/// One replica's per-bin flux perturbations, `delta_g`.
///
/// Drawn from the spectrum's own relative sigma, once per replica and shared
/// across every nuclide. Floored at `-1` so a sampled flux cannot go negative:
/// a negative flux is not a physical state, and a bin whose relative error
/// exceeds 100% -- normal in the tail of a tally -- would otherwise produce one.
pub fn flux_deviates(
    relative: &[f64],
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

    let z = crate::covariance_sample::standard_normals(&mut state, relative.len());
    relative
        .iter()
        .zip(z)
        .map(|(sigma, z)| {
            coverage.bins_sampled += 1;
            let d = sigma * z;
            if d < -1.0 {
                coverage.bins_floored += 1;
                -1.0
            } else {
                d
            }
        })
        .collect()
}

/// Keeps the flux stream clear of the per-nuclide cross-section streams, which
/// are keyed on a hash of the nuclide name.
const FLUX_STREAM: u32 = 0xF10D_5EED;

/// Apply one replica's flux perturbation to a set of unit-flux rates.
///
/// Rates with no per-group terms are passed through unchanged, which is what a
/// nuclide whose cross sections were never collapsed against this spectrum
/// looks like.
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
            // Sum the same terms the nominal rate is the sum of, weighted. With
            // every delta zero this reproduces the nominal rate exactly, which
            // is what keeps an unperturbed replica honest.
            *rate = terms
                .iter()
                .zip(delta)
                .map(|(term, d)| term * (1.0 + d))
                .sum();
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
        let rel = vec![0.05, 0.10, 0.20];
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
        let d = flux_deviates(&[0.0, 0.1], 1, 0, 0, &mut c);
        assert_eq!(d[0], 0.0);
    }

    /// A relative error above 100% would otherwise sample a negative flux.
    #[test]
    fn a_negative_flux_is_floored_and_counted() {
        let mut c = FluxCoverage::default();
        let mut floored_any = false;
        for k in 0..400 {
            // 300% relative: a large share of draws fall below -1.
            let d = flux_deviates(&[3.0], 9, k, 0, &mut c);
            assert!(d[0] >= -1.0, "a flux bin cannot go negative");
            floored_any |= d[0] == -1.0;
        }
        assert!(floored_any, "some draws must have been floored");
        assert!(c.bins_floored > 0 && c.bins_sampled == 400);
    }

    /// Over many replicas the spread matches the sigma it was built from.
    #[test]
    fn the_sampled_spread_matches_the_stated_flux_error() {
        let mut c = FluxCoverage::default();
        let rel = vec![0.05, 0.10];
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
        for (i, want) in rel.iter().enumerate() {
            let mean = s[i] / nf;
            let sd = (sq[i] / nf - mean * mean).sqrt();
            assert!(mean.abs() < 0.01, "mean {mean} for bin {i}");
            assert!((sd - want).abs() < 0.01, "sd {sd} != {want} for bin {i}");
        }
    }
}
