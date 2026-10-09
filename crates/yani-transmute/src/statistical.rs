//! Statistical uncertainty on transport-tallied reaction rates.
//!
//! A reaction rate from transport is a Monte Carlo estimate, and the rates of
//! one material are estimated from the same histories, so they are correlated:
//! [`TransmutationTallies::get_reaction_rate_covariance`](crate::TransmutationTallies::get_reaction_rate_covariance)
//! measures that covariance history by history. Here it is sampled: each
//! replica draws the whole rate vector from `N(mean, Sigma)`, which by the
//! central limit theorem is what the estimate's sampling distribution is, and
//! the draw keeps the correlations the covariance carries.
//!
//! The draw uses a stream of its own, keyed on the replica, so the statistical
//! axis is independent of the nuclear-data ones and the total spread over all
//! sources is their quadrature sum.
//!
//! The stream is also keyed on the material. Different materials' tallies are
//! separate estimates, so their statistical errors are drawn independently: on
//! a shared stream two materials would move up and down together replica by
//! replica, and a per-replica sum over materials would overstate the spread.
//! The nuclear-data streams are keyed on the nuclide alone and stay shared
//! across materials, since one evaluation's error is common to all of them.

use std::collections::HashMap;

use yani::ReactionRates;

use crate::history_statistics::{RateCovariance, RateLabel};
use crate::transmutation_tallies::PartialRates;

/// Keeps the statistical stream clear of every other replica stream.
pub(crate) const STATISTICAL_STREAM: u32 = 0x57A7_1571;

/// Relative size below which a Cholesky pivot is treated as zero. A rate
/// covariance is positive semi-definite, not definite: a rate equal to a sum of
/// others, or one no history scored, leaves a zero pivot that has no direction
/// to sample.
const PIVOT_TOLERANCE: f64 = 1.0e-12;

/// A material's tallied rates, ready to be sampled with their covariance.
#[derive(Debug, Clone)]
pub struct StatisticalRates {
    labels: Vec<RateLabel>,
    means: Vec<f64>,
    /// Lower-triangular factor, row-major, with `L L^T = Sigma`.
    factor: Vec<f64>,
}

impl StatisticalRates {
    /// Factorize a rate covariance. The rates are what the tally reports, so
    /// normalize the covariance the way the solve's rates are normalized.
    pub fn new(covariance: &RateCovariance) -> Self {
        let m = covariance.len();
        let mut l = vec![0.0; m * m];
        let scale = (0..m)
            .map(|i| covariance.covariance(i, i))
            .fold(0.0_f64, f64::max);
        for j in 0..m {
            let mut d = covariance.covariance(j, j);
            for k in 0..j {
                d -= l[j * m + k] * l[j * m + k];
            }
            if d <= PIVOT_TOLERANCE * scale || d <= 0.0 {
                // No independent direction left: this rate is fixed by the
                // ones before it, or has no spread at all.
                continue;
            }
            let pivot = d.sqrt();
            l[j * m + j] = pivot;
            for i in (j + 1)..m {
                let mut v = covariance.covariance(i, j);
                for k in 0..j {
                    v -= l[i * m + k] * l[j * m + k];
                }
                l[i * m + j] = v / pivot;
            }
        }
        StatisticalRates {
            labels: covariance.labels.clone(),
            means: covariance.rates.clone(),
            factor: l,
        }
    }

    /// Rates carried, the totals and the partials together.
    pub fn len(&self) -> usize {
        self.means.len()
    }

    /// Whether there is nothing to sample.
    pub fn is_empty(&self) -> bool {
        self.means.is_empty()
    }

    /// One replica's rates: totals as [`ReactionRates`], partials as
    /// [`PartialRates`], and how many draws came out negative and were floored
    /// at zero. A negative rate is not a physical state; with a statistical
    /// error of a few percent it is vanishingly rare, and counting it says when
    /// it is not.
    ///
    /// `material_id` keys the draw to the material whose tally this is, so
    /// two materials' draws are independent. It is the material's id rather
    /// than its position in a call, so the draw does not depend on the order
    /// materials are visited in.
    pub fn sample(
        &self,
        base_seed: u64,
        material_id: u32,
        replica: u64,
    ) -> (ReactionRates, PartialRates, usize) {
        let m = self.len();
        let replica_seed = yamc_rng::history_seed(base_seed, replica);
        // A second level under the statistical stream, rather than the id
        // folded into the stream tag, so the tag stays clear of every other
        // replica stream whatever the id.
        let stream = yamc_rng::secondary_seed(replica_seed, STATISTICAL_STREAM);
        let mut state = yamc_rng::expand_seed(yamc_rng::secondary_seed(stream, material_id));
        let z = crate::covariance_sample::standard_normals(&mut state, m);
        let mut totals: ReactionRates = HashMap::new();
        let mut partials: PartialRates = HashMap::new();
        let mut floored = 0;
        for i in 0..m {
            let row = &self.factor[i * m..i * m + i + 1];
            let delta: f64 = row.iter().zip(&z).map(|(l, z)| l * z).sum();
            let mut value = self.means[i] + delta;
            if value < 0.0 {
                floored += 1;
                value = 0.0;
            }
            let label = &self.labels[i];
            match &label.target {
                None => {
                    totals
                        .entry(label.nuclide.clone())
                        .or_default()
                        .insert(label.kind.clone(), value);
                }
                Some(target) => partials
                    .entry(label.nuclide.clone())
                    .or_default()
                    .entry(label.kind.clone())
                    .or_default()
                    .push((target.clone(), value)),
            }
        }
        (totals, partials, floored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(nuclide: &str, kind: &str, target: Option<&str>) -> RateLabel {
        RateLabel {
            nuclide: nuclide.to_string(),
            kind: kind.to_string(),
            target: target.map(str::to_string),
        }
    }

    /// Packed upper triangle of a symmetric matrix.
    fn packed(m: &[&[f64]]) -> Vec<f64> {
        let n = m.len();
        (0..n).flat_map(|i| (i..n).map(move |j| m[i][j])).collect()
    }

    /// The draws reproduce the means, variances and correlation they were
    /// factorized from, and a rate that is the sum of two others (a singular
    /// covariance) is sampled rather than refused.
    #[test]
    fn draws_carry_the_covariance() {
        let means = vec![10.0, 5.0, 15.0];
        // Rate 2 is rate 0 plus rate 1, so the matrix has rank 2.
        let (a, b, c) = (4.0, 1.0, 1.2);
        let cov = packed(&[
            &[a, c, a + c],
            &[c, b, c + b],
            &[a + c, c + b, a + b + 2.0 * c],
        ]);
        let rc = RateCovariance::from_parts(
            vec![
                label("Fe56", "(n,p)", None),
                label("Fe56", "(n,a)", None),
                label("Co59", "(n,gamma)", Some("Co60_m1")),
            ],
            means.clone(),
            1000,
            cov,
        );
        let s = StatisticalRates::new(&rc);
        let n = 20000;
        let mut xs: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
        for r in 0..n as u64 {
            let (totals, partials, floored) = s.sample(3, 0, r);
            assert_eq!(floored, 0);
            xs[0].push(totals["Fe56"]["(n,p)"]);
            xs[1].push(totals["Fe56"]["(n,a)"]);
            xs[2].push(partials["Co59"]["(n,gamma)"][0].1);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let cov_of = |u: &[f64], v: &[f64]| {
            let (mu, mv) = (mean(u), mean(v));
            u.iter()
                .zip(v)
                .map(|(x, y)| (x - mu) * (y - mv))
                .sum::<f64>()
                / (u.len() - 1) as f64
        };
        for (i, want) in means.iter().enumerate() {
            assert!((mean(&xs[i]) - want).abs() < 0.05, "mean {i}");
        }
        assert!((cov_of(&xs[0], &xs[0]) / a - 1.0).abs() < 0.04);
        assert!((cov_of(&xs[1], &xs[1]) / b - 1.0).abs() < 0.04);
        assert!((cov_of(&xs[0], &xs[1]) / c - 1.0).abs() < 0.08);
        // The dependent rate follows the other two exactly.
        for ((sum, a), b) in xs[2].iter().zip(&xs[0]).zip(&xs[1]) {
            assert!((sum - a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn the_draw_is_a_function_of_seed_and_replica() {
        let rc = RateCovariance::from_parts(
            vec![label("Fe56", "(n,p)", None)],
            vec![2.0],
            100,
            vec![0.25],
        );
        let s = StatisticalRates::new(&rc);
        let one = s.sample(9, 0, 4).0["Fe56"]["(n,p)"];
        assert_eq!(
            one.to_bits(),
            s.sample(9, 0, 4).0["Fe56"]["(n,p)"].to_bits()
        );
        assert_ne!(
            one.to_bits(),
            s.sample(9, 0, 5).0["Fe56"]["(n,p)"].to_bits()
        );
    }

    /// Two materials' tallies are separate estimates, so their draws are
    /// independent: across replicas the deviates of two materials with the
    /// same seed are uncorrelated, while each material's own draw reproduces
    /// under the same seed.
    #[test]
    fn materials_draw_independently_and_reproducibly() {
        let a = StatisticalRates::new(&RateCovariance::from_parts(
            vec![label("Fe56", "(n,p)", None)],
            vec![2.0],
            100,
            vec![0.25],
        ));
        let b = StatisticalRates::new(&RateCovariance::from_parts(
            vec![label("Fe56", "(n,p)", None)],
            vec![7.0],
            100,
            vec![0.81],
        ));
        let n = 20000;
        let draw = |s: &StatisticalRates, id: u32| -> Vec<f64> {
            (0..n as u64)
                .map(|r| s.sample(11, id, r).0["Fe56"]["(n,p)"])
                .collect()
        };
        let (xa, xb) = (draw(&a, 1), draw(&b, 2));
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mb) = (mean(&xa), mean(&xb));
        let cov: f64 = xa.iter().zip(&xb).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let var = |v: &[f64], m: f64| v.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
        let corr = cov / (var(&xa, ma) * var(&xb, mb)).sqrt();
        // Five sigma of a zero correlation at this sample size.
        assert!(
            corr.abs() < 5.0 / (n as f64).sqrt(),
            "materials 1 and 2 are correlated: {corr}"
        );
        // The same seed reproduces each material bit for bit.
        assert_eq!(
            xa.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            draw(&a, 1).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            xb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            draw(&b, 2).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // The id is what separates them: the same tally under the same id
        // draws the same deviates.
        let first = a.sample(11, 1, 0).0["Fe56"]["(n,p)"];
        assert_eq!(
            first.to_bits(),
            a.sample(11, 1, 0).0["Fe56"]["(n,p)"].to_bits()
        );
        assert_ne!(
            first.to_bits(),
            a.sample(11, 2, 0).0["Fe56"]["(n,p)"].to_bits()
        );
    }
}
