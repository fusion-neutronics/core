//! Decay branching ratio uncertainty.
//!
//! MT=457 gives each decay mode a ratio and a sigma, and no covariance between
//! the modes. The ratios of a parent sum to a fixed total `T` (one, after the
//! converter has put any evaluated residual on the largest mode), so the modes
//! cannot be independent: every replica must keep `T`, and every ratio must
//! stay in `[0, T]`. A draw here is built in two parts, the covariance of the
//! modes and the shape of each mode.
//!
//! The covariance keeps `T`, so its rows sum to zero. It is fixed by the
//! stated sigmas for two modes, and for three, and is not for four or more.
//! Here it is the zero-sum covariance that reproduces every stated sigma and
//! has the largest entropy as a Gaussian, so it assumes no correlation the sum
//! rule does not force:
//!
//! - two modes: perfectly anticorrelated at the one sigma. An evaluator writes
//!   one number for a complementary pair (Bi212 0.3594 and 0.6406, both
//!   0.0006), or states it on one mode only and leaves the other as its
//!   complement (JENDL-5.0 Bi212 gives alpha 0.0006 and beta- none);
//! - three modes: the unique one with the three variances;
//! - more: the maximum-entropy one. Where one mode's sigma is the quadrature
//!   sum of the others' (JENDL-5.0 writes Li11 and Cs114 that way) it is the
//!   evaluator's own model, the minor modes independent and the dominant one
//!   their complement.
//!
//! A zero-sum covariance can realise sigmas only if none exceeds the sum of
//! the others (the deviations are the sides of a closed polygon, for any
//! distribution). Where the largest does (two modes with different sigmas, or
//! ENDF/B-VIII.1 Hf182m's 0.076 beside 0.020 and 0.0033), the others are
//! realised exactly, all moving together, and the largest takes the sum of
//! theirs, the most the sum rule allows. The parent is sampled and reported
//! in `sigma_adjusted`. A mode with no stated sigma has no variance to match.
//! A single one is the complement of the others, which are then independent
//! (the maximum-entropy reading), when it is the largest mode, as an
//! evaluator writes a dominant mode as one minus the rest (ENDF/B-VIII.1 P28),
//! or when only one other mode states a sigma. A small unstated mode beside
//! two or more stated ones (JENDL-5.0 I112's alpha, 1.2e-5) is held instead:
//! the stated modes can trade between themselves, and as their complement it
//! would take a sigma a hundred times its ratio. Two or more unstated modes
//! are held too, and the stated modes share what is left.
//!
//! The shape is bounded. Every moving mode but one is drawn from its own
//! logit-normal on `[0, T]`, `T / (1 + exp(-(mu + sigma u)))` for a standard
//! normal `u`, with `mu` and `sigma` fitted by quadrature so that its mean is
//! the evaluated ratio and its standard deviation the sigma above, both to
//! rounding. The latent normals are correlated (a Gaussian copula), with each
//! pair's correlation solved, through the Hermite (Mehler) series of the
//! pair's covariance, so that the modes themselves carry the covariance above
//! (NORTA). The remaining mode, the absorber, takes what is left of the
//! total, so its mean is exact and its sigma is the covariance's wherever the
//! copula carries the covariance. Of the moving modes at least five sigmas
//! from zero, the absorber is the first, most sigmas first, whose sigma comes
//! out to 1e-6, or else the closest. A copula can fall short in two ways,
//! both reported in `sigma_adjusted` beyond 0.1%: drawn modes of different
//! shape cannot be perfectly correlated, as where the polygon closes flat
//! (under a percent), and two small skewed modes cannot be strongly
//! anticorrelated, since non-negative ratios covary by no less than minus the
//! product of their means (JENDL-5.0 B15's 0.0032 +- 0.0112 and 0.0075 +-
//! 0.0075 would need 2.3 times that, so its 0.9893 is drawn at 0.0120 rather
//! than 0.0083). Why this shape:
//!
//! - a logit-normal reaches every mean and sigma a ratio in `[0, T]` can
//!   have, `sigma^2 < b (T - b)`, and never leaves `[0, T]`, so a small branch
//!   with a large relative sigma (an actinide's spontaneous fission, a
//!   beta-delayed two-neutron mode) is sampled rather than held;
//! - it tends to the Gaussian as the sigma shrinks, so a mode many sigmas from
//!   zero is drawn as the zero-sum Gaussian would draw it;
//! - one minus a logit-normal is a logit-normal, so a pair is drawn the same
//!   whichever mode is the complement, and two modes never leave the simplex.
//!   It is the one-mode marginal of Aitchison's logistic-normal, the usual
//!   model of a composition.
//!
//! Against the alternatives, each fitted to the same means and sigmas and
//! run over every sampled parent of ENDF/B-VIII.1, JEFF-4.0 and JENDL-5.0:
//! drawing each mode independently and renormalising (SANDY's
//! `BranchingRatio.normalize`, after Gaussian or lognormal draws) imposes the
//! sum rule after the draw and misses a stated sigma by more than 5% on 40%
//! to 60% of parents, mostly by shrinking the dominant mode's; a Dirichlet has
//! one concentration for all modes, so is exact for two modes and wrong by
//! more than 20% on nearly every parent of three or more; a generalised
//! (Connor-Mosimann) Dirichlet depends on the order of the modes and leaves
//! the last mode's sigma where it falls (off by more than 5% on 6, 4 and 53
//! parents); lognormal marginals are unbounded above, so two wide modes leave
//! the simplex; Beta marginals reach what the logit-normal reaches but have
//! no quantile cheap enough to evaluate per replica. The logit-normal copula
//! realises every drawn mode's mean and sigma to rounding, and every
//! complement's to 0.1% on all but 3, 0 and 13 parents, those where the
//! stated sigmas need a correlation non-negative ratios cannot carry.
//!
//! A stated sigma no ratio with that mean can have, `sigma^2 >= b (T - b)`
//! (the Bhatia-Davis bound; JENDL-5.0 has ratios near 1e-16 beside sigmas of
//! 0.04, and Bi205 a sigma of 675), describes no distribution, and the mode is
//! held at its ratio. For a complement the check uses the sigma it would take.
//! The rest share what is left, so a parent can be sampled with some of its
//! modes held (`partly_held`), and is held whole (`too_wide`) when no mode
//! can move. With three or more moving modes the drawn ones can together
//! exceed the total, leaving the complement short; with the absorber five
//! sigmas clear this is a few draws in a million over the three libraries.
//! Such a draw is pulled back towards the nominal ratios until every ratio is
//! non-negative, keeping the total, and counted in `floored`.
//!
//! A spontaneous-fission mode is a mode like any other here: it is a share of
//! the parent's decays, whether or not the chain models a product for it.
//! Modes that make the same daughter are separate rows with separate sigmas,
//! and the daughter takes their sum.
//!
//! First order is linear in the ratios: sensitivities are taken along
//! `directions`, a factor of the covariance the draw realises, so a parent's
//! first-order variance is `J C J^T`, the delta method on the covariance the
//! replicas actually have. The shape (the skew of a small branch) enters only
//! beyond first order. The linearity check maps each replica's drawn
//! deviation back onto the directions, so its residual measures how far the
//! response is from linear in the ratios, as before.
//!
//! Only the inventory moves. A parent's per-decay emission (its lines and
//! decay energy) is the decay scheme's and stays nominal, since no correlation
//! between a ratio and an absolute line intensity is published, and the report
//! lists it under `not_perturbed`.

use std::collections::{BTreeSet, HashMap};

use yani::ChainNuclide;

use crate::covariance_sample::{eigen, name_ordinal, standard_normals};

/// Keeps the decay-branching streams clear of every other per-nuclide stream.
pub(crate) const DECAY_BRANCHING_STREAM: u32 = 0xDB2A_0C5E;

/// Two stated sigmas closer than this, relatively, are the same number.
const SAME_SIGMA: f64 = 1.0e-9;

/// The largest sigma within this, relatively, of the sum of the others closes
/// the polygon flat: every other mode moves together. Short of it the
/// maximum-entropy covariance is nearly singular and slow to find, and the
/// flat one differs from it by less than this in any sigma.
const FLAT: f64 = 1.0e-6;

/// A latent normal is integrated over `[-U_MAX, U_MAX]`; the weight beyond is
/// under 1e-57.
const U_MAX: f64 = 16.0;

/// The trapezoid step over a latent normal, at most. The rule is spectrally
/// accurate for a smooth integrand of Gaussian decay, its error near
/// `exp(-2 pi^2 / h^2)`, so this is ample for a mode that is nearly Gaussian.
const STEP: f64 = 0.1;

/// Trapezoid steps per unit of `1 / sigma` of a latent normal. The logistic's
/// nearest poles are `pi / sigma` off the real line, which bounds the rule's
/// error by about `exp(-2 pi^2 STEPS_PER_TURN)`: a wide mode is integrated
/// more finely.
const STEPS_PER_TURN: f64 = 4.0;

/// The finest trapezoid step, reached only by a sigma close to the
/// Bhatia-Davis bound.
const FINEST_STEP: f64 = 2.0e-4;

/// Terms of the Hermite series for the covariance of two drawn modes.
const HERMITE_TERMS: usize = 60;

/// How many sigmas from zero a row must stand to be preferred as the
/// absorber for its sigma's fidelity rather than for its distance.
const ABSORBER_SIGMAS: f64 = 5.0;

/// An absorber whose sigma the copula reproduces to this, relatively, is
/// taken without trying another.
const ABSORBER_TOLERANCE: f64 = 1.0e-6;

/// A row drawn at a sigma further than this, relatively, from its stated
/// one is reported.
const ADJUSTED: f64 = 1.0e-3;

/// An eigenvalue of a correlation below this, relative to the largest, is
/// dropped: the latent normals then need fewer independent ones, as where
/// the polygon closes flat and every drawn mode moves together, and the
/// realised covariance fewer directions.
const RANK_TOLERANCE: f64 = 1.0e-10;

/// One mode drawn from its own marginal, a logit-normal on `[0, moving]`:
/// `moving / (1 + exp(-(mu + sigma u)))` for its latent standard normal `u`,
/// with the mode's ratio as its mean and its sigma as its standard deviation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Marginal {
    /// The mode's row of `decays`.
    pub(crate) row: usize,
    pub(crate) mu: f64,
    pub(crate) sigma: f64,
}

impl Marginal {
    /// The ratio at latent normal `u`, on `[0, moving]`.
    fn value(&self, moving: f64, u: f64) -> f64 {
        moving * logistic(self.mu + self.sigma * u)
    }
}

/// One parent whose decay branching is sampled.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Split {
    pub(crate) parent: String,
    /// The nominal ratios, row by row of `decays`.
    pub(crate) ratios: Vec<f64>,
    /// Their sum, which every replica keeps.
    pub(crate) total: f64,
    /// The sum of the moving rows, which every drawn row stays within.
    pub(crate) moving: f64,
    /// The sigma the draw aims at on each row: the stated one, or the sum
    /// rule's where that cannot be, and the complement's for an unstated
    /// complement; zero on a held row.
    pub(crate) sigmas: Vec<f64>,
    /// The row that takes `total` minus every other row: of the moving rows,
    /// the one most sigmas from zero.
    pub(crate) absorber: usize,
    /// Every other moving row, each from its own logit-normal.
    pub(crate) marginals: Vec<Marginal>,
    /// One row per marginal: its latent normal per unit of each standard
    /// normal. Every row has unit length, so each latent normal is standard.
    pub(crate) latent: Vec<Vec<f64>>,
    /// A factor of the covariance the draw realises, one column per
    /// direction: every row's deviation per unit of it, zero on a held row,
    /// summing to zero. First-order sensitivities are taken along these.
    pub(crate) directions: Vec<Vec<f64>>,
    /// One row per direction: a deviation's coordinate along it, read from
    /// the drawn rows, so the directions times the coordinates give back the
    /// deviation (to the variance a dropped eigenvalue carried).
    pub(crate) coordinates: Vec<Vec<f64>>,
}

impl Split {
    /// How many standard normals one draw takes.
    pub(crate) fn width(&self) -> usize {
        self.latent.first().map_or(0, Vec::len)
    }

    /// The ratios for the standard normals `z`, [`width`](Self::width) of
    /// them, and the factor the deviation was scaled by to keep the absorber
    /// non-negative (1 unless the drawn rows exceeded the total).
    pub(crate) fn ratios_for(&self, z: &[f64]) -> (Vec<f64>, f64) {
        debug_assert_eq!(z.len(), self.width());
        let mut deviation = vec![0.0; self.ratios.len()];
        for (m, l) in self.marginals.iter().zip(&self.latent) {
            let u = l.iter().zip(z).fold(0.0, |acc, (a, z)| acc + a * z);
            deviation[m.row] = m.value(self.moving, u) - self.ratios[m.row];
        }
        let d = -deviation.iter().sum::<f64>();
        deviation[self.absorber] = d;
        let b = self.ratios[self.absorber];
        let scale = if b + d < 0.0 { b / -d } else { 1.0 };
        (self.place(&deviation, scale), scale)
    }

    /// The nominal ratios moved by `scale` times `deviation`, the absorber
    /// taking what the other rows leave of the total.
    fn place(&self, deviation: &[f64], scale: f64) -> Vec<f64> {
        let mut out: Vec<f64> = self
            .ratios
            .iter()
            .zip(deviation)
            .map(|(b, d)| {
                if scale < 1.0 {
                    (b + scale * d).max(0.0)
                } else {
                    b + d
                }
            })
            .collect();
        let rest = out
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.absorber)
            .fold(0.0, |acc, (_, r)| acc + r);
        out[self.absorber] = (self.total - rest).max(0.0);
        out
    }

    /// A replica's ratios as coordinates along the directions, for the
    /// linearity check.
    pub(crate) fn coordinates_of(&self, ratios: &[f64]) -> Vec<f64> {
        self.coordinates
            .iter()
            .map(|row| {
                row.iter()
                    .zip(ratios.iter().zip(&self.ratios))
                    .fold(0.0, |acc, (y, (r, b))| acc + y * (r - b))
            })
            .collect()
    }

    /// The step, in units of direction `k`, that moves no row by more than
    /// `relative` of its nominal ratio.
    pub(crate) fn step(&self, k: usize, relative: f64) -> f64 {
        relative
            * self
                .ratios
                .iter()
                .zip(&self.directions[k])
                .filter(|(_, a)| **a != 0.0)
                .map(|(b, a)| b / a.abs())
                .fold(f64::INFINITY, f64::min)
    }

    /// The ratios moved `step` along direction `k`, for a sensitivity.
    pub(crate) fn shifted(&self, k: usize, step: f64) -> Vec<f64> {
        let deviation: Vec<f64> = self.directions[k].iter().map(|a| a * step).collect();
        self.place(&deviation, 1.0)
    }
}

/// The multi-mode parents of a chain: the sampled ones and the reasons the
/// rest are not, or not wholly.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Candidates {
    /// Sampled, sorted by parent.
    pub(crate) splits: Vec<Split>,
    /// Held at nominal: no mode states a sigma, or only one does beside two
    /// or more that do not, so no mode can take up its change.
    pub(crate) without: BTreeSet<String>,
    /// Held at nominal: every mode that could move states a sigma no ratio
    /// with its mean can have.
    pub(crate) too_wide: BTreeSet<String>,
    /// Sampled, with one or more modes held at their ratios.
    pub(crate) partly_held: BTreeSet<String>,
    /// Sampled, with the largest sigma above the sum of the others and
    /// realised as that sum.
    pub(crate) sigma_adjusted: BTreeSet<String>,
}

/// Sort the unstable parents of `chain` with more than one decay mode.
///
/// A single mode is not listed anywhere: its ratio is the whole of the
/// parent's decays whatever sigma the tape writes beside it.
pub(crate) fn candidates(chain: &HashMap<String, ChainNuclide>) -> Candidates {
    let mut out = Candidates::default();
    for (name, cn) in chain {
        if cn.half_life.is_none_or(|t| t <= 0.0) || cn.decays.len() < 2 {
            continue;
        }
        let ratios: Vec<f64> = cn.decays.iter().map(|d| d.branching).collect();
        // A stored 0.0 is MT=457's "not stated", as a null is.
        let sigmas: Vec<Option<f64>> = cn
            .decays
            .iter()
            .map(|d| d.branching_uncertainty.filter(|s| *s > 0.0))
            .collect();
        match split(name, &ratios, &sigmas) {
            Sorted::Without => {
                out.without.insert(name.clone());
            }
            Sorted::TooWide => {
                out.too_wide.insert(name.clone());
            }
            Sorted::Sampled {
                split,
                partly_held,
                sigma_adjusted,
            } => {
                if partly_held {
                    out.partly_held.insert(name.clone());
                }
                if sigma_adjusted {
                    out.sigma_adjusted.insert(name.clone());
                }
                out.splits.push(*split);
            }
        }
    }
    out.splits.sort_by(|a, b| a.parent.cmp(&b.parent));
    out
}

/// Where one parent lands.
#[derive(Debug)]
enum Sorted {
    Without,
    TooWide,
    Sampled {
        split: Box<Split>,
        partly_held: bool,
        sigma_adjusted: bool,
    },
}

/// Whether a ratio `b` in `[0, t]` can have standard deviation `s`: the
/// Bhatia-Davis bound, `s^2 < b (t - b)`.
fn possible(b: f64, s: f64, t: f64) -> bool {
    s * s < b * (t - b)
}

/// One parent's sampling, from its ratios and stated sigmas, row by row.
fn split(parent: &str, ratios: &[f64], sigmas: &[Option<f64>]) -> Sorted {
    let n = ratios.len();
    let total = ratios.iter().fold(0.0, |acc, b| acc + b);
    let unstated: Vec<usize> = (0..n).filter(|&i| sigmas[i].is_none()).collect();
    // Two or more unstated modes have nothing to move by, and a lone one is
    // the complement of the rest.
    let mut held: Vec<bool> = (0..n)
        .map(|i| unstated.len() > 1 && sigmas[i].is_none())
        .collect();
    let stated = n - unstated.len();
    if stated + usize::from(unstated.len() == 1) < 2 {
        return Sorted::Without;
    }
    // Hold every mode whose sigma no ratio with its mean can have. Holding
    // one shrinks what the rest share, so repeat until none is; a mode once
    // held stays held, so this ends.
    let largest = ratios.iter().copied().fold(0.0_f64, f64::max);
    let (free, complement) = loop {
        let moving = total
            - (0..n)
                .filter(|&i| held[i])
                .fold(0.0, |acc, i| acc + ratios[i]);
        let free: Vec<(usize, f64)> = (0..n)
            .filter(|&i| !held[i])
            .filter_map(|i| sigmas[i].map(|s| (i, s)))
            .filter(|&(i, s)| possible(ratios[i], s, moving))
            .collect();
        // A lone unstated mode is the complement when it is the largest, as
        // an evaluator writes a dominant mode, or when no two stated modes
        // can trade between themselves.
        let complement = match unstated[..] {
            [c] if !held[c] && !free.is_empty() && (free.len() < 2 || ratios[c] >= largest) => {
                let s = free.iter().map(|(_, s)| s * s).sum::<f64>().sqrt();
                possible(ratios[c], s, moving).then_some(c)
            }
            _ => None,
        };
        let next: Vec<bool> = (0..n)
            .map(|i| held[i] || !(free.iter().any(|&(j, _)| j == i) || complement == Some(i)))
            .collect();
        if next == held || free.len() + usize::from(complement.is_some()) < 2 {
            break (free, complement);
        }
        held = next;
    };
    let movable = free.len() + usize::from(complement.is_some());
    if movable < 2 {
        return Sorted::TooWide;
    }
    let partly_held = movable < n;
    let column = |entries: &[(usize, f64)], pivot: usize| -> Vec<f64> {
        let mut a = vec![0.0; n];
        for &(i, v) in entries {
            a[i] = v;
        }
        a[pivot] = -entries.iter().fold(0.0, |acc, (_, v)| acc + v);
        a
    };
    // The covariance to reproduce, as a factor: one column per independent
    // normal of the zero-sum Gaussian with it.
    let target: Vec<Vec<f64>> = if let Some(c) = complement {
        // The stated modes independent, the complement their sum rule.
        free.iter().map(|&(i, s)| column(&[(i, s)], c)).collect()
    } else {
        // The largest sigma is the pivot; on a tie, the last row.
        let largest = free.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
        let k = free
            .iter()
            .rposition(|(_, s)| *s >= largest * (1.0 - SAME_SIGMA))
            .expect("a free mode holds the largest sigma");
        let (pivot, s_k) = free[k];
        let others: Vec<(usize, f64)> = free
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, m)| *m)
            .collect();
        let sum_others = others.iter().fold(0.0, |acc, (_, s)| acc + s);
        if s_k >= sum_others * (1.0 - FLAT) {
            // The polygon closes flat: every other mode moves together.
            vec![column(&others, pivot)]
        } else {
            let s: Vec<f64> = others.iter().map(|(_, s)| *s).collect();
            let w: Vec<f64> = s.iter().map(|s| s / s_k).collect();
            let l = max_entropy_factor(&w);
            let m = others.len();
            (0..m)
                .map(|col| {
                    let entries: Vec<(usize, f64)> = others
                        .iter()
                        .enumerate()
                        .map(|(row, &(i, si))| (i, si * l[row * m + col]))
                        .collect();
                    column(&entries, pivot)
                })
                .collect()
        }
    };
    let moving_rows: Vec<usize> = (0..n)
        .filter(|&i| free.iter().any(|&(j, _)| j == i) || complement == Some(i))
        .collect();
    let moving = moving_rows.iter().fold(0.0, |acc, &i| acc + ratios[i]);
    let covariance =
        |i: usize, j: usize| -> f64 { target.iter().fold(0.0, |acc, a| acc + a[i] * a[j]) };
    // Every moving row's own logit-normal, its variance and its Hermite
    // series, whichever row turns out to absorb.
    let fitted: Vec<Fitted> = moving_rows
        .iter()
        .map(|&row| {
            let (mu, sigma) = fit_logit_normal(
                ratios[row] / moving,
                covariance(row, row) / (moving * moving),
            );
            let marginal = Marginal { row, mu, sigma };
            let (_, variance) = marginal_moments(&marginal, moving);
            let series = hermite_coefficients(&marginal, moving);
            Fitted {
                marginal,
                variance,
                series,
            }
        })
        .collect();
    // The absorber: of the moving rows at least ABSORBER_SIGMAS from zero (so
    // the complement is seldom pushed below it), most sigmas first, the first
    // whose sigma the copula reproduces. Where none does exactly, the one that
    // comes closest, and where no row is that far from zero, the farthest. A
    // copula falls short where the drawn modes' shapes cannot carry their
    // correlation: two modes of different shape cannot be perfectly
    // correlated (the polygon closed flat), and two small skewed ones cannot
    // be strongly anticorrelated while staying non-negative.
    let z = |k: usize| ratios[moving_rows[k]] / covariance(moving_rows[k], moving_rows[k]).sqrt();
    let mut order: Vec<usize> = (0..moving_rows.len()).collect();
    order.sort_by(|&a, &b| z(b).total_cmp(&z(a)).then(b.cmp(&a)));
    let mut best: Option<Choice> = None;
    for (rank, k) in order.into_iter().enumerate() {
        if rank > 0 && z(k) < ABSORBER_SIGMAS {
            break;
        }
        let drawn: Vec<&Fitted> = (0..moving_rows.len())
            .filter(|&a| a != k)
            .map(|a| &fitted[a])
            .collect();
        let f = drawn.len();
        let mut correlation = vec![0.0; f * f];
        for a in 0..f {
            correlation[a * f + a] = 1.0;
            for b in (a + 1)..f {
                let p = latent_correlation(
                    &drawn[a].series,
                    &drawn[b].series,
                    covariance(drawn[a].marginal.row, drawn[b].marginal.row),
                );
                correlation[a * f + b] = p;
                correlation[b * f + a] = p;
            }
        }
        let latent = unit_factor(&correlation, f);
        let realised = drawn_covariance(&drawn, &latent);
        // The absorber's variance is the drawn rows' sum's.
        let variance = realised.iter().sum::<f64>();
        let want = covariance(moving_rows[k], moving_rows[k]);
        let error = ((variance.max(0.0) / want).sqrt() - 1.0).abs();
        if best.as_ref().is_none_or(|b| error < b.error) {
            best = Some(Choice {
                error,
                absorber: k,
                latent,
                realised,
            });
        }
        if error <= ABSORBER_TOLERANCE {
            break;
        }
    }
    let Choice {
        absorber: k,
        latent,
        realised,
        ..
    } = best.expect("two or more rows move");
    let absorber = moving_rows[k];
    let marginals: Vec<Marginal> = fitted
        .into_iter()
        .enumerate()
        .filter(|(a, _)| *a != k)
        .map(|(_, f)| f.marginal)
        .collect();
    // The directions factor the covariance the draw realises, the drawn rows'
    // through the eigenvectors of their correlation (so a sigma six decades
    // below another keeps its share) and the absorber minus their sum, so
    // first order sees the spread the replicas have.
    let f = marginals.len();
    let sd: Vec<f64> = (0..f).map(|a| realised[a * f + a].sqrt()).collect();
    let correlation: Vec<f64> = (0..f * f)
        .map(|ab| realised[ab] / (sd[ab / f] * sd[ab % f]))
        .collect();
    let (values, vectors) = eigen(&correlation, f);
    let top = values.iter().copied().fold(0.0_f64, f64::max);
    let kept: Vec<usize> = (0..f)
        .filter(|&j| values[j] > RANK_TOLERANCE * top)
        .collect();
    let directions: Vec<Vec<f64>> = kept
        .iter()
        .map(|&j| {
            let entries: Vec<(usize, f64)> = marginals
                .iter()
                .enumerate()
                .map(|(a, m)| (m.row, sd[a] * vectors[a * f + j] * values[j].sqrt()))
                .collect();
            column(&entries, absorber)
        })
        .collect();
    // A row drawn at a sigma other than its stated one, or for a complement
    // than the one the sum rule gives it.
    let sigma_adjusted = moving_rows.iter().any(|&i| {
        let drawn = directions.iter().map(|a| a[i] * a[i]).sum::<f64>().sqrt();
        let stated = sigmas[i].unwrap_or_else(|| covariance(i, i).sqrt());
        (drawn / stated - 1.0).abs() > ADJUSTED
    });
    // A deviation's coordinates: the drawn rows scaled by their sigmas and
    // projected onto each kept eigenvector.
    let coordinates: Vec<Vec<f64>> = kept
        .iter()
        .map(|&j| {
            let mut y = vec![0.0; n];
            for (a, m) in marginals.iter().enumerate() {
                y[m.row] = vectors[a * f + j] / (sd[a] * values[j].sqrt());
            }
            y
        })
        .collect();
    Sorted::Sampled {
        split: Box::new(Split {
            parent: parent.to_string(),
            ratios: ratios.to_vec(),
            total,
            moving,
            sigmas: (0..n).map(|i| covariance(i, i).sqrt()).collect(),
            absorber,
            marginals,
            latent,
            directions,
            coordinates,
        }),
        partly_held,
        sigma_adjusted,
    }
}

/// One candidate absorber: its index among the moving rows, how far its
/// realised sigma is from the one it should take, relatively, and the drawn
/// rows' latent factor and realised covariance.
struct Choice {
    error: f64,
    absorber: usize,
    latent: Vec<Vec<f64>>,
    realised: Vec<f64>,
}

/// A moving row's logit-normal, with its variance and its Hermite series.
struct Fitted {
    marginal: Marginal,
    variance: f64,
    series: Vec<f64>,
}

/// The covariance of the drawn rows, row-major `f x f`: their own variances,
/// and between two the Hermite series at their latent correlation.
fn drawn_covariance(drawn: &[&Fitted], latent: &[Vec<f64>]) -> Vec<f64> {
    let f = drawn.len();
    let mut out = vec![0.0; f * f];
    for a in 0..f {
        for b in 0..f {
            out[a * f + b] = if a == b {
                drawn[a].variance
            } else {
                let p = latent[a]
                    .iter()
                    .zip(&latent[b])
                    .fold(0.0, |acc, (x, y)| acc + x * y);
                mehler(&drawn[a].series, &drawn[b].series, p)
            };
        }
    }
    out
}

/// `1 / (1 + exp(-y))`, without overflow either way.
fn logistic(y: f64) -> f64 {
    if y >= 0.0 {
        1.0 / (1.0 + (-y).exp())
    } else {
        let e = y.exp();
        e / (1.0 + e)
    }
}

/// Trapezoid nodes and weights over a standard normal, fine enough for a
/// logit-normal of latent scale `sigma`.
fn quadrature(sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let h = STEP.min(1.0 / (STEPS_PER_TURN * sigma)).max(FINEST_STEP);
    let k = (U_MAX / h).ceil() as i64;
    let norm = h / std::f64::consts::TAU.sqrt();
    (-k..=k)
        .map(|j| {
            let u = j as f64 * h;
            (u, norm * (-0.5 * u * u).exp())
        })
        .unzip()
}

/// `(mu, sigma)` of the logit-normal on `[0, 1]` with mean `m` and variance
/// `v`, which needs `0 < v < m (1 - m)`.
///
/// At a fixed `sigma` the mean rises with `mu`, found by Newton safeguarded
/// by bisection. The variance at that mean rises with `sigma`, found by a
/// secant, again kept inside a bracket.
pub(crate) fn fit_logit_normal(m: f64, v: f64) -> (f64, f64) {
    let at = |sigma: f64| -> (f64, f64) {
        let (u, w) = quadrature(sigma);
        let (mut lo, mut hi) = (-1.0e3, 1.0e3);
        let mut mu = (m / (1.0 - m)).ln();
        for _ in 0..200 {
            let (mut f, mut d) = (-m, 0.0);
            for (u, w) in u.iter().zip(&w) {
                let x = logistic(mu + sigma * u);
                f += w * x;
                d += w * x * (1.0 - x);
            }
            if f == 0.0 {
                break;
            }
            if f > 0.0 {
                hi = mu;
            } else {
                lo = mu;
            }
            let mut next = mu - f / d;
            if d.is_nan() || d <= 0.0 || next <= lo || next >= hi {
                next = 0.5 * (lo + hi);
            }
            let done = (next - mu).abs() <= 1.0e-14 * (1.0 + mu.abs());
            mu = next;
            if done {
                break;
            }
        }
        let mean = u
            .iter()
            .zip(&w)
            .fold(0.0, |acc, (u, w)| acc + w * logistic(mu + sigma * u));
        let variance = u.iter().zip(&w).fold(0.0, |acc, (u, w)| {
            let d = logistic(mu + sigma * u) - mean;
            acc + w * d * d
        });
        (mu, variance)
    };
    // Gaussian to first order, or lognormal for a mode near either end;
    // the smaller is the closer start.
    let near_end = m.min(1.0 - m);
    let mut sigma = (v.sqrt() / (m * (1.0 - m))).min((v / (near_end * near_end)).ln_1p().sqrt());
    // The variance against sigma, both logarithmic, is a line of slope 2
    // while the mode is nearly Gaussian and flattens towards the bound: a
    // secant on it, kept inside the bracket by bisection.
    let (mut lo, mut hi) = (0.0, f64::INFINITY);
    let mut previous: Option<(f64, f64)> = None;
    for _ in 0..200 {
        let (_, variance) = at(sigma);
        let gap = (variance / v).ln();
        if gap.abs() <= 1.0e-14 {
            break;
        }
        if gap > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        let slope = match previous {
            Some((s0, g0)) if s0 != sigma && g0 != gap => (gap - g0) / (sigma / s0).ln(),
            _ => 2.0,
        };
        previous = Some((sigma, gap));
        let mut next = sigma * (-gap / slope).exp();
        if next.is_nan() || slope <= 0.0 || next <= lo || next >= hi {
            next = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                2.0 * sigma
            };
        }
        if (next / sigma - 1.0).abs() <= 1.0e-15 {
            break;
        }
        sigma = next;
    }
    (at(sigma).0, sigma)
}

/// A drawn mode's mean and variance on `[0, moving]`, by quadrature.
fn marginal_moments(m: &Marginal, moving: f64) -> (f64, f64) {
    let (u, w) = quadrature(m.sigma);
    let mean = u
        .iter()
        .zip(&w)
        .fold(0.0, |acc, (u, w)| acc + w * m.value(moving, *u));
    let variance = u.iter().zip(&w).fold(0.0, |acc, (u, w)| {
        let d = m.value(moving, *u) - mean;
        acc + w * d * d
    });
    (mean, variance)
}

/// The Hermite coefficients of a drawn mode's deviation from its mean as a
/// function of its latent normal, `E[(x - mean) He_k(u)] / sqrt(k!)` for `k`
/// from 1. Two modes whose latent normals correlate at `p` then covary by
/// `sum_k a_k b_k p^k` (Mehler's formula).
fn hermite_coefficients(m: &Marginal, moving: f64) -> Vec<f64> {
    let (u, w) = quadrature(m.sigma);
    let x: Vec<f64> = u.iter().map(|u| m.value(moving, *u)).collect();
    let mean = x.iter().zip(&w).fold(0.0, |acc, (x, w)| acc + w * x);
    let mut a = vec![0.0; HERMITE_TERMS];
    for ((u, w), x) in u.iter().zip(&w).zip(&x) {
        let d = w * (x - mean);
        // The normalised Hermite polynomials, by their recurrence.
        let (mut previous, mut h) = (1.0, *u);
        for (k, a) in a.iter_mut().enumerate() {
            *a += d * h;
            let next = (u * h - ((k + 1) as f64).sqrt() * previous) / ((k + 2) as f64).sqrt();
            previous = h;
            h = next;
        }
    }
    a
}

/// The covariance of two drawn modes, from their Hermite coefficients, when
/// their latent normals correlate at `p`.
fn mehler(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter()
        .zip(b)
        .rev()
        .fold(0.0, |acc, (x, y)| (acc + x * y) * p)
}

/// The latent correlation that gives two drawn modes the covariance
/// `target`, or the nearest of -1 and 1 where no correlation does (two modes
/// of different shape cannot be perfectly correlated). The covariance rises
/// with the latent correlation, so bisection finds it.
fn latent_correlation(a: &[f64], b: &[f64], target: f64) -> f64 {
    if target == 0.0 {
        return 0.0;
    }
    if target >= mehler(a, b, 1.0) {
        return 1.0;
    }
    if target <= mehler(a, b, -1.0) {
        return -1.0;
    }
    let (mut lo, mut hi) = (-1.0_f64, 1.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mehler(a, b, mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A factor `G` of the `f x f` latent correlation, `G G^T` its nearest
/// positive semidefinite matrix, one row per drawn mode with unit length so
/// each latent normal stays standard and each mode keeps its mean and sigma.
/// Columns whose eigenvalue is negligible are dropped.
fn unit_factor(correlation: &[f64], f: usize) -> Vec<Vec<f64>> {
    let (values, vectors) = eigen(correlation, f);
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    let kept: Vec<usize> = (0..f)
        .filter(|&j| values[j] > RANK_TOLERANCE * largest)
        .collect();
    (0..f)
        .map(|i| {
            let row: Vec<f64> = kept
                .iter()
                .map(|&j| vectors[i * f + j] * values[j].sqrt())
                .collect();
            let norm = row.iter().map(|g| g * g).sum::<f64>().sqrt();
            row.into_iter().map(|g| g / norm).collect()
        })
        .collect()
}

/// Cholesky factor, row-major `m x m`, of the correlation matrix `R` of `m`
/// modes beside an absorbing one, where `w[i]` is mode `i`'s sigma over the
/// absorber's: unit diagonal, `w^T R w = 1` (the absorber, minus their sum,
/// keeps its own sigma), and the largest determinant.
///
/// The maximiser is `R = (diag(lambda) + mu w w^T)^-1`; `(lambda, mu)` minimise
/// the convex dual `-log det(diag(lambda) + mu w w^T) + sum(lambda) + mu`,
/// found by damped Newton from `R = I`, where the step is never so long that
/// the matrix stops being positive definite. Needs `sum(w) > 1`, that the
/// absorber's sigma is under the sum of the others'.
fn max_entropy_factor(w: &[f64]) -> Vec<f64> {
    let m = w.len();
    let p = m + 1;
    let mut lambda = vec![1.0; m];
    let mut mu = 0.0;
    let invert = |lambda: &[f64], mu: f64| -> Option<Vec<f64>> {
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                a[i * m + j] = mu * w[i] * w[j];
            }
            a[i * m + i] += lambda[i];
        }
        let l = cholesky(&a, m)?;
        Some(inverse_from_cholesky(&l, m))
    };
    let mut r = invert(&lambda, mu).expect("the identity is positive definite");
    for _ in 0..1000 {
        let rw: Vec<f64> = (0..m)
            .map(|i| (0..m).map(|j| r[i * m + j] * w[j]).sum())
            .collect();
        let wrw: f64 = w.iter().zip(&rw).map(|(a, b)| a * b).sum();
        let mut gradient: Vec<f64> = (0..m).map(|i| 1.0 - r[i * m + i]).collect();
        gradient.push(1.0 - wrw);
        let mut hessian = vec![0.0; p * p];
        for i in 0..m {
            for j in 0..m {
                hessian[i * p + j] = r[i * m + j] * r[i * m + j];
            }
            hessian[i * p + m] = rw[i] * rw[i];
            hessian[m * p + i] = rw[i] * rw[i];
        }
        hessian[m * p + m] = wrw * wrw;
        let Some(h) = cholesky(&hessian, p) else {
            break;
        };
        let newton = solve_cholesky(&h, p, &gradient);
        let decrement2: f64 = newton.iter().zip(&gradient).map(|(a, b)| a * b).sum();
        if decrement2 < 1.0e-24 {
            break;
        }
        // Self-concordance keeps a step of 1 / (1 + decrement) inside the
        // domain; a full one is taken once quadratic convergence has begun.
        let decrement = decrement2.sqrt();
        let t = if decrement > 0.25 {
            1.0 / (1.0 + decrement)
        } else {
            1.0
        };
        let next: Vec<f64> = (0..m).map(|i| lambda[i] - t * newton[i]).collect();
        let next_mu = mu - t * newton[m];
        match invert(&next, next_mu) {
            Some(next_r) => {
                lambda = next;
                mu = next_mu;
                r = next_r;
            }
            None => break,
        }
    }
    cholesky(&r, m).expect("a correlation matrix found positive definite")
}

/// Lower Cholesky factor of a symmetric positive-definite `n x n` matrix,
/// row-major, or `None` where a pivot is not positive.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let d = a[j * n + j] - (0..j).map(|k| l[j * n + k] * l[j * n + k]).sum::<f64>();
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let pivot = d.sqrt();
        l[j * n + j] = pivot;
        for i in (j + 1)..n {
            let v = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = v / pivot;
        }
    }
    Some(l)
}

/// `x` with `L L^T x = b`, for the lower Cholesky factor `l`.
fn solve_cholesky(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i * n + k] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - ((i + 1)..n).map(|k| l[k * n + i] * x[k]).sum::<f64>()) / l[i * n + i];
    }
    x
}

/// The inverse of `L L^T`, for the lower Cholesky factor `l`.
fn inverse_from_cholesky(l: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        for (i, v) in solve_cholesky(l, n, &e).into_iter().enumerate() {
            out[i * n + j] = v;
        }
    }
    out
}

/// One replica's standard normals for a parent, [`Split::width`] of them, on a
/// stream keyed on the parent's name, so a parent splits the same way in every
/// spectrum and every material of a run.
pub(crate) fn normals(split: &Split, base_seed: u64, replica: u64) -> Vec<f64> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    let seed = yamc_rng::secondary_seed(
        replica_seed,
        name_ordinal(&split.parent) ^ DECAY_BRANCHING_STREAM,
    );
    let mut state = yamc_rng::expand_seed(seed);
    standard_normals(&mut state, split.width())
}

/// One replica's ratios for every sampled parent, keyed by parent, one entry
/// per row of its `decays`. A draw pulled back onto the simplex is counted in
/// `floored`.
pub(crate) fn sample(
    splits: &[Split],
    base_seed: u64,
    replica: u64,
    floored: &mut usize,
) -> HashMap<String, Vec<f64>> {
    splits
        .iter()
        .map(|t| {
            let (ratios, scale) = t.ratios_for(&normals(t, base_seed, replica));
            if scale < 1.0 {
                *floored += 1;
            }
            (t.parent.clone(), ratios)
        })
        .collect()
}

/// Give a chain nuclide's decay modes the ratios in `ratios`, row by row.
///
/// Nothing else stored on the nuclide depends on a ratio: the matrix reads
/// `branching` for the target and the light particles alike, and the loss
/// stays the decay constant.
pub(crate) fn set_decay_branchings(cn: &mut ChainNuclide, ratios: &[f64]) {
    debug_assert_eq!(cn.decays.len(), ratios.len(), "{}", cn.name);
    for (d, r) in cn.decays.iter_mut().zip(ratios) {
        d.branching = *r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(kind: &str, target: Option<&str>, b: f64, s: Option<f64>) -> yani::ChainReaction {
        yani::ChainReaction {
            kind: kind.to_string(),
            target: target.map(str::to_string),
            branching: b,
            q_value: None,
            branching_uncertainty: s,
            evaluated_branching: None,
            multiplicity: None,
        }
    }

    fn parent(name: &str, decays: Vec<yani::ChainReaction>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: Some(3600.0),
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: [None; 3],
            reactions: Vec::new(),
            decays,
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    fn chain_of(nuclides: Vec<ChainNuclide>) -> HashMap<String, ChainNuclide> {
        nuclides.into_iter().map(|c| (c.name.clone(), c)).collect()
    }

    /// Four modes, none dominant in sigma, none a quadrature sum.
    const FOUR: [(f64, Option<f64>); 4] = [
        (0.85, Some(0.012)),
        (0.004, Some(0.0006)),
        (0.13, Some(0.013)),
        (0.016, Some(0.003)),
    ];

    /// The split of a parent with these `(ratio, sigma)` rows.
    fn split_of(rows: &[(f64, Option<f64>)]) -> Sorted {
        let ratios: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let sigmas: Vec<Option<f64>> = rows.iter().map(|r| r.1).collect();
        split("P", &ratios, &sigmas)
    }

    fn sampled(rows: &[(f64, Option<f64>)]) -> Split {
        match split_of(rows) {
            Sorted::Sampled { split, .. } => *split,
            other => panic!("{rows:?} is not sampled: {other:?}"),
        }
    }

    /// Mean and standard deviation of every row over `n` replicas, and the
    /// largest departure of a replica's total from the nominal one.
    fn moments(t: &Split, n: u64) -> (Vec<f64>, Vec<f64>, f64) {
        let rows = t.ratios.len();
        let (mut sum, mut sum2) = (vec![0.0; rows], vec![0.0; rows]);
        let mut worst: f64 = 0.0;
        let mut floored = 0;
        for replica in 0..n {
            let r = &sample(std::slice::from_ref(t), 11, replica, &mut floored)["P"];
            assert!(r.iter().all(|x| *x >= 0.0), "{r:?}");
            worst = worst.max((r.iter().sum::<f64>() - t.total).abs());
            for (i, x) in r.iter().enumerate() {
                sum[i] += x;
                sum2[i] += x * x;
            }
        }
        assert_eq!(floored, 0);
        let n = n as f64;
        let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
        let sd = sum2
            .iter()
            .zip(&mean)
            .map(|(s2, m)| ((s2 / n - m * m) * n / (n - 1.0)).max(0.0).sqrt())
            .collect();
        (mean, sd, worst)
    }

    /// Each row's realised sigma, from the directions, which factor the
    /// covariance the draw realises.
    fn realised(t: &Split) -> Vec<f64> {
        (0..t.ratios.len())
            .map(|i| t.directions.iter().map(|a| a[i] * a[i]).sum::<f64>().sqrt())
            .collect()
    }

    fn correlation(t: &Split, i: usize, j: usize) -> f64 {
        let s = realised(t);
        t.directions.iter().map(|a| a[i] * a[j]).sum::<f64>() / (s[i] * s[j])
    }

    /// Every row's mean and standard deviation under the draw, before any
    /// pull-back: the drawn rows by quadrature, the absorber from their
    /// covariances by the Hermite series at the latent correlations.
    fn exact_moments(t: &Split) -> (Vec<f64>, Vec<f64>) {
        let mut mean = t.ratios.clone();
        let mut sd = vec![0.0; t.ratios.len()];
        let moments: Vec<(f64, f64)> = t
            .marginals
            .iter()
            .map(|m| marginal_moments(m, t.moving))
            .collect();
        let series: Vec<Vec<f64>> = t
            .marginals
            .iter()
            .map(|m| hermite_coefficients(m, t.moving))
            .collect();
        let mut absorber_variance = 0.0;
        for (a, m) in t.marginals.iter().enumerate() {
            mean[m.row] = moments[a].0;
            sd[m.row] = moments[a].1.sqrt();
            for b in 0..t.marginals.len() {
                absorber_variance += if a == b {
                    moments[a].1
                } else {
                    let p: f64 = t.latent[a]
                        .iter()
                        .zip(&t.latent[b])
                        .map(|(x, y)| x * y)
                        .sum();
                    mehler(&series[a], &series[b], p)
                };
            }
        }
        let rest: f64 = mean
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != t.absorber)
            .map(|(_, m)| m)
            .sum();
        mean[t.absorber] = t.total - rest;
        sd[t.absorber] = absorber_variance.sqrt();
        (mean, sd)
    }

    #[test]
    fn each_parent_lands_in_its_category() {
        let chain = chain_of(vec![
            // Equal sigmas: sampled on the first row.
            parent(
                "Equal",
                vec![
                    mode("beta-", Some("A"), 0.6406, Some(6e-4)),
                    mode("alpha", Some("B"), 0.3594, Some(6e-4)),
                ],
            ),
            // Only the minor mode states one, with the dominant one written as
            // 0.0: sampled on the stated row.
            parent(
                "OneSided",
                vec![
                    mode("ec/beta+", Some("A"), 0.9963, Some(0.0)),
                    mode("p", Some("B"), 0.0037, Some(5e-4)),
                ],
            ),
            // Sampled at the smaller sigma, the most a pair allows.
            parent(
                "Unequal",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.01)),
                    mode("alpha", Some("B"), 0.5, Some(0.02)),
                ],
            ),
            // 0.0037 +- 0.0015 is 2.5 sigmas from zero, which a logit-normal
            // draws.
            parent(
                "Near",
                vec![
                    mode("ec/beta+", Some("A"), 0.9963, None),
                    mode("p", Some("B"), 0.0037, Some(0.0015)),
                ],
            ),
            // JENDL-5.0 Fm248: the sf complement would take 0.05 on 0.00105,
            // more than any ratio with that mean can have.
            parent(
                "Wide",
                vec![
                    mode("alpha", Some("A"), 0.998949525, Some(0.05)),
                    mode("sf", None, 0.001050475, None),
                ],
            ),
            // JENDL-5.0 Hf180m: a converter residual of 4e-19 stating 0.0008
            // is held, the other two move.
            parent(
                "Residual",
                vec![
                    mode("beta-", Some("A"), 4.33638e-19, Some(0.0008)),
                    mode("IT", Some("B"), 0.99690031, Some(0.001)),
                    mode("beta-", Some("C"), 0.00309969, Some(0.0008)),
                ],
            ),
            // One stated mode, two unstated: none can take up its change.
            parent(
                "OneOfThree",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.01)),
                    mode("beta-,n", Some("B"), 0.3, None),
                    mode("beta-,2n", Some("C"), 0.2, None),
                ],
            ),
            // Three modes, the alpha 3.3 sigmas from zero: all move.
            parent(
                "Np236",
                vec![
                    mode("alpha", Some("A"), 0.002, Some(0.0006)),
                    mode("ec/beta+", Some("B"), 0.863, Some(0.008)),
                    mode("beta-", Some("C"), 0.135, Some(0.008)),
                ],
            ),
            parent(
                "Three",
                vec![
                    mode("beta-", Some("A"), 0.86149, Some(0.006)),
                    mode("beta-,n", Some("B"), 0.138, Some(0.006)),
                    mode("beta-,2n", Some("C"), 0.00051, Some(7e-5)),
                ],
            ),
            parent(
                "Unstated",
                vec![
                    mode("beta-", Some("A"), 0.5, Some(0.0)),
                    mode("sf", None, 0.5, None),
                ],
            ),
            // A single mode's ratio is the whole: nothing to list.
            parent("Single", vec![mode("beta-", Some("A"), 1.0, Some(0.1))]),
        ]);
        let c = candidates(&chain);
        let sampled: Vec<(&str, usize)> = c
            .splits
            .iter()
            .map(|t| (t.parent.as_str(), t.absorber))
            .collect();
        // Each absorber is the moving row most sigmas from zero.
        assert_eq!(
            sampled,
            vec![
                ("Equal", 0),
                ("Near", 0),
                ("Np236", 1),
                ("OneSided", 0),
                ("Residual", 1),
                ("Three", 0),
                ("Unequal", 1)
            ]
        );
        let set = |xs: &[&str]| -> BTreeSet<String> { xs.iter().map(|s| s.to_string()).collect() };
        // With the residual held, the IT's 0.001 exceeds the other's 0.0008.
        assert_eq!(c.sigma_adjusted, set(&["Residual", "Unequal"]));
        assert_eq!(c.partly_held, set(&["Residual"]));
        assert_eq!(c.too_wide, set(&["Wide"]));
        assert_eq!(c.without, set(&["OneOfThree", "Unstated"]));
    }

    /// A logit-normal is fitted to any mean and variance a ratio can have:
    /// nearly Gaussian, small and skewed, and close to the Bhatia-Davis bound.
    #[test]
    fn a_logit_normal_takes_the_mean_and_sigma_it_is_fitted_to() {
        for (m, cv) in [
            (0.3594, 0.0017),
            (0.5, 0.2),
            (0.45, 0.38),
            (0.002, 0.3),
            (3.4e-7, 23.5),
            (1.0e-6, 400.0),
            (0.999, 0.03),
            (0.5, 0.95),
        ] {
            let v = (cv * m) * (cv * m);
            let (mu, sigma) = fit_logit_normal(m, v);
            let marginal = Marginal { row: 0, mu, sigma };
            let (got_m, got_v) = marginal_moments(&marginal, 1.0);
            assert!((got_m / m - 1.0).abs() < 1e-9, "{m} {cv}: mean {got_m}");
            assert!(
                (got_v.sqrt() / v.sqrt() - 1.0).abs() < 1e-9,
                "{m} {cv}: sigma {} against {}",
                got_v.sqrt(),
                v.sqrt()
            );
        }
    }

    /// Two wide modes (ENDF/B-VIII.1 Cu79, 0.45 and 0.55 +- 0.17, 2.6
    /// sigmas from zero) are drawn as a logit-normal pair: never off the
    /// simplex, each at its mean and sigma, and skewed away from the nearer
    /// end, where the Gaussian would cross zero once in two hundred draws.
    #[test]
    fn two_wide_modes_stay_on_the_simplex() {
        let t = sampled(&[(0.45, Some(0.17)), (0.55, Some(0.17))]);
        assert_eq!(t.absorber, 1);
        let (mean, sd, worst) = moments(&t, 40_000);
        assert!(worst < 4.0 * f64::EPSILON);
        for (i, b) in [0.45, 0.55].iter().enumerate() {
            assert!((mean[i] - b).abs() < 4.0 * 0.17 / 40_000f64.sqrt());
            assert!((sd[i] / 0.17 - 1.0).abs() < 0.02, "{i}: {}", sd[i]);
        }
        let mut floored = 0;
        let skew: f64 = (0..40_000)
            .map(|r| {
                let x = sample(std::slice::from_ref(&t), 11, r, &mut floored)["P"][0];
                ((x - 0.45) / 0.17).powi(3)
            })
            .sum::<f64>()
            / 40_000.0;
        assert_eq!(floored, 0);
        assert!(skew > 0.05, "skewness {skew}");
    }

    /// Small branches with large relative sigmas, beside a dominant mode
    /// (ENDF/B-VIII.1 Np236, and JENDL-5.0 Xe113 with an alpha of 1e-4 +-
    /// 1e-4): every mode keeps its mean and sigma, the small ones exactly
    /// from their fit and the absorber from the latent correlations.
    #[test]
    fn small_branches_keep_their_mean_and_sigma() {
        for rows in [
            vec![
                (0.002, Some(0.0006)),
                (0.863, Some(0.008)),
                (0.135, Some(0.008)),
            ],
            vec![
                (0.92990701, Some(0.04)),
                (0.069993, Some(0.04)),
                (9.999e-05, Some(0.0001)),
            ],
        ] {
            let t = sampled(&rows);
            let (mean, sd) = exact_moments(&t);
            let want = &t.sigmas;
            for i in 0..rows.len() {
                assert!((mean[i] / rows[i].0 - 1.0).abs() < 1e-9, "{rows:?} {i}");
                assert!(
                    (sd[i] / want[i] - 1.0).abs() < 1e-6,
                    "{rows:?} {i}: {} against {}",
                    sd[i],
                    want[i]
                );
            }
            let (mc_mean, mc_sd, worst) = moments(&t, 40_000);
            assert!(worst < 4.0 * f64::EPSILON);
            for i in 0..rows.len() {
                assert!((mc_mean[i] - rows[i].0).abs() < 5.0 * want[i] / 200.0);
                assert!((mc_sd[i] / want[i] - 1.0).abs() < 0.05, "{i}");
            }
        }
    }

    #[test]
    fn a_draw_keeps_the_total_and_is_reproducible() {
        let t = sampled(&[(0.6406, Some(6e-4)), (0.3594, Some(6e-4))]);
        let mut floored = 0;
        let a = sample(std::slice::from_ref(&t), 3, 17, &mut floored);
        let b = sample(std::slice::from_ref(&t), 3, 17, &mut floored);
        assert_eq!(a, b);
        let r = &a["P"];
        assert!((r[0] + r[1] - 1.0).abs() <= f64::EPSILON, "{r:?}");
        assert_ne!(r[0], 0.6406, "the draw moved the ratio");
        assert_eq!(floored, 0);
    }

    /// Three modes: the one covariance with the stated variances and rows
    /// summing to zero, realised over many draws, every replica on the
    /// simplex with the total kept.
    #[test]
    fn three_modes_realise_every_sigma_and_keep_the_total() {
        // ENDF/B-VIII.1 Rb98, and Pd111m whose largest sigma squared exceeds
        // the sum of the others' squares (so two modes correlate positively).
        for rows in [
            vec![
                (0.86149, Some(0.006)),
                (0.138, Some(0.006)),
                (0.00051, Some(7e-5)),
            ],
            vec![
                (0.0789123, Some(0.009315774)),
                (0.1910876, Some(0.02626949)),
                (0.7300001, Some(0.03)),
            ],
        ] {
            let t = sampled(&rows);
            let s: Vec<f64> = rows.iter().map(|r| r.1.unwrap()).collect();
            for (got, want) in realised(&t).iter().zip(&s) {
                assert!((got / want - 1.0).abs() < 1e-6, "{got} against {want}");
            }
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        let k = 3 - i - j;
                        let want = (s[k] * s[k] - s[i] * s[i] - s[j] * s[j]) / (2.0 * s[i] * s[j]);
                        assert!((correlation(&t, i, j) - want).abs() < 1e-6);
                    }
                }
            }
            let (mean, sd, worst) = moments(&t, 20_000);
            assert!(worst < 4.0 * f64::EPSILON, "total moved by {worst:e}");
            for i in 0..3 {
                // 20 000 replicas: the mean within 4 standard errors, a
                // sigma within 3%.
                assert!((mean[i] - rows[i].0).abs() < 4.0 * s[i] / 20_000f64.sqrt());
                assert!(
                    (sd[i] / s[i] - 1.0).abs() < 0.03,
                    "{i}: {} against {}",
                    sd[i],
                    s[i]
                );
            }
        }
    }

    /// Four modes whose largest sigma is the quadrature sum of the others'
    /// (JENDL-5.0 Li11): the maximum-entropy split is the evaluator's model,
    /// the minor modes independent and the dominant one their complement.
    #[test]
    fn a_quadrature_sum_reads_as_independent_minor_modes() {
        let rows = [
            (0.073, Some(0.01374773)),
            (0.866, Some(0.013)),
            (0.042, Some(0.004)),
            (0.019, Some(0.002)),
        ];
        let t = sampled(&rows);
        // The 0.866 is the most sigmas from zero, and absorbs.
        assert_eq!(t.absorber, 1);
        for (got, (_, want)) in realised(&t).iter().zip(&rows) {
            assert!((got / want.unwrap() - 1.0).abs() < 1e-6, "{got}");
        }
        for (i, j) in [(1, 2), (1, 3), (2, 3)] {
            assert!(correlation(&t, i, j).abs() < 1e-6);
        }
        let (_, sd, worst) = moments(&t, 20_000);
        assert!(worst < 4.0 * f64::EPSILON);
        for (got, (_, want)) in sd.iter().zip(&rows) {
            assert!((got / want.unwrap() - 1.0).abs() < 0.03);
        }
    }

    /// Four modes with no structure to the sigmas: every one realised, by
    /// the largest-determinant covariance.
    #[test]
    fn four_modes_realise_every_sigma() {
        let rows = FOUR;
        let t = sampled(&rows);
        for (got, (_, want)) in realised(&t).iter().zip(&rows) {
            assert!((got / want.unwrap() - 1.0).abs() < 1e-9, "{got}");
        }
        let (mean, sd, worst) = moments(&t, 20_000);
        assert!(worst < 4.0 * f64::EPSILON);
        for i in 0..4 {
            let s = rows[i].1.unwrap();
            assert!((mean[i] - rows[i].0).abs() < 4.0 * s / 20_000f64.sqrt());
            assert!((sd[i] / s - 1.0).abs() < 0.03);
        }
    }

    /// Close to flat, and across six decades of sigma, the covariance is
    /// still found and every sigma realised.
    #[test]
    fn hard_polygons_still_realise_every_sigma() {
        for rows in [
            vec![
                (0.5, Some(0.01)),
                (0.3, Some(0.006)),
                (0.2, Some(0.0040001)),
            ],
            vec![
                (0.9, Some(1e-3)),
                (0.0999, Some(1e-3)),
                (1e-4, Some(2e-9)),
                (1e-9, Some(1e-10)),
            ],
        ] {
            let t = sampled(&rows);
            for (got, (_, want)) in realised(&t).iter().zip(&rows) {
                assert!((got / want.unwrap() - 1.0).abs() < 1e-6, "{rows:?}: {got}");
            }
        }
    }

    /// A largest sigma beyond the sum of the others' (ENDF/B-VIII.1 Hf182m)
    /// is realised as that sum, the others exactly.
    #[test]
    fn a_sigma_the_sum_rule_cannot_hold_is_reduced() {
        let rows = [
            (0.4301644, Some(0.07606925)),
            (0.03533465, Some(0.003304615)),
            (0.46000005, Some(0.02)),
            (0.0745009, Some(0.01880015)),
        ];
        let Sorted::Sampled {
            split: t,
            sigma_adjusted,
            partly_held,
        } = split_of(&rows)
        else {
            panic!("sampled");
        };
        assert!(sigma_adjusted && !partly_held);
        let s = &t.sigmas;
        for (i, want) in [(1, 0.003304615), (2, 0.02), (3, 0.01880015)] {
            assert!((s[i] / want - 1.0).abs() < 1e-12);
        }
        assert!((s[0] / (0.003304615 + 0.02 + 0.01880015) - 1.0).abs() < 1e-12);
        // The modes move on one latent normal, perfectly rank correlated;
        // their different shapes leave the realised sigmas within a percent.
        assert_eq!(t.width(), 1);
        for (got, want) in realised(&t).iter().zip(s) {
            assert!((got / want - 1.0).abs() < 0.01, "{got} against {want}");
        }
        assert!(correlation(&t, 1, 2) > 0.99);
    }

    /// One unstated mode is the complement of the others, which are then
    /// independent (ENDF/B-VIII.1 P28); the held rows never move.
    #[test]
    fn an_unstated_mode_is_the_complement_and_held_rows_stay() {
        let t = sampled(&[(0.98, None), (0.012, Some(0.002)), (0.008, Some(0.001))]);
        assert_eq!(t.absorber, 0);
        let s = realised(&t);
        assert!((s[0] - (0.002f64.hypot(0.001))).abs() < 1e-15);
        assert!(correlation(&t, 1, 2).abs() < 1e-15);

        // Two unstated modes are held; the stated pair shares the rest.
        let rows = [
            (0.6, Some(0.006)),
            (0.35, Some(0.006)),
            (0.03, None),
            (0.02, None),
        ];
        let t = sampled(&rows);
        let mut floored = 0;
        for replica in 0..256 {
            let r = &sample(std::slice::from_ref(&t), 1, replica, &mut floored)["P"];
            assert_eq!(r[2].to_bits(), 0.03f64.to_bits());
            assert_eq!(r[3].to_bits(), 0.02f64.to_bits());
            assert!((r[0] + r[1] - 0.95).abs() < 1e-15);
        }
    }

    /// A draw past the simplex is pulled back onto it, total kept, and
    /// counted.
    #[test]
    fn a_draw_off_the_simplex_is_pulled_back() {
        let t = sampled(&[(0.8, Some(0.01)), (0.12, Some(0.01)), (0.08, Some(0.01))]);
        assert_eq!(t.width(), 2);
        let mut pulled = 0;
        for z in [[50.0, 50.0], [50.0, -50.0], [-50.0, 50.0], [-50.0, -50.0]] {
            let (r, scale) = t.ratios_for(&z);
            assert!(r.iter().all(|x| *x >= 0.0), "{r:?}");
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            if scale < 1.0 {
                pulled += 1;
                assert!(r.iter().any(|x| *x < 1e-15), "{r:?}");
            }
        }
        assert!(pulled > 0);
    }

    /// A sensitivity step along any direction moves no row by more than the
    /// relative step, and keeps the total.
    #[test]
    fn a_shift_stays_within_its_step() {
        let rows = FOUR;
        let t = sampled(&rows);
        for k in 0..t.directions.len() {
            let step = t.step(k, 1e-3);
            let r = t.shifted(k, step);
            let mut tight = 0.0_f64;
            for (x, (b, _)) in r.iter().zip(&rows) {
                tight = tight.max((x / b - 1.0).abs());
            }
            assert!(tight <= 1e-3 * (1.0 + 1e-6), "{tight}");
            assert!(tight >= 1e-3 * (1.0 - 1e-6), "{tight}");
            assert!((r.iter().sum::<f64>() - t.total).abs() < 1e-15);
        }
    }

    /// Every published library's parents, sorted, and every sampled split
    /// checked against the sigmas it was built from. Exactly, before any
    /// pull-back: every drawn mode at its ratio and its covariance sigma (a
    /// free mode's own stated one), the absorber at its ratio and at most its
    /// own, a held mode not moving. Then over `YANI_DECAY_BRANCHING_DRAWS`
    /// draws per parent (default 4000), pull-backs included: the worst mean
    /// and sigma error, and how many draws were pulled back.
    ///
    /// Opt-in, since it reads whole local libraries. Set
    /// `YANI_DECAY_BRANCHING_DATA` to a directory holding the published
    /// `transmutation-<library>.arrow/decay/`, then run `cargo test --release
    /// -p yani-transmute --lib real_libraries -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads whole local libraries; set YANI_DECAY_BRANCHING_DATA"]
    fn real_libraries_realise_their_sigmas() {
        let root = std::path::PathBuf::from(
            std::env::var_os("YANI_DECAY_BRANCHING_DATA").expect("YANI_DECAY_BRANCHING_DATA"),
        );
        let draws: u64 = std::env::var("YANI_DECAY_BRANCHING_DRAWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4000);
        for library in ["endf-b8.1", "jeff-4.0", "jendl-5.0"] {
            let decay = root
                .join(format!("transmutation-{library}.arrow"))
                .join("decay");
            if !decay.is_dir() {
                println!("{library}: not present");
                continue;
            }
            let chain = yani::load_chain_parts(&decay.to_string_lossy(), None, None, &[])
                .expect("chain loads")
                .chain;
            let started = std::time::Instant::now();
            let c = candidates(&chain);
            let setup = started.elapsed();
            let mut by_modes: std::collections::BTreeMap<usize, usize> = Default::default();
            // Parents with a moving mode under five sigmas from zero, which
            // a Gaussian could not draw untruncated.
            let mut near_zero = 0;
            let mut skewed_modes = 0;
            // Each worst error with the parent it is on.
            let worse = |w: &mut (f64, String), e: f64, p: &str| {
                if e > w.0 {
                    *w = (e, p.to_string());
                }
            };
            let mut worst_mean = (0.0, String::new());
            let mut worst_sd = (0.0, String::new());
            let mut worst_absorber = (0.0, String::new());
            let mut mc_mean = (0.0, String::new());
            let mut mc_sd = (0.0, String::new());
            let (mut pulled, mut pulled_parents) = (0, 0);
            let mut unresolved = 0;
            for t in &c.splits {
                *by_modes.entry(t.ratios.len()).or_default() += 1;
                let cn = &chain[&t.parent];
                let want = &t.sigmas;
                let (mean, sd) = exact_moments(t);
                let wide = (0..t.ratios.len())
                    .filter(|&i| want[i] > 0.0 && t.ratios[i] < 5.0 * want[i])
                    .count();
                skewed_modes += wide;
                near_zero += usize::from(wide > 0);
                for (i, d) in cn.decays.iter().enumerate() {
                    let stated = d.branching_uncertainty.filter(|s| *s > 0.0);
                    if want[i] == 0.0 {
                        assert_eq!(mean[i], t.ratios[i], "{library} {}", t.parent);
                        assert_eq!(sd[i], 0.0, "{library} {}", t.parent);
                        continue;
                    }
                    // A sigma under rounding of its ratio (JENDL-5.0 Pd115m's
                    // 1.2e-17 on 0.92) cannot be resolved in a double.
                    if want[i] < 1e-8 * t.ratios[i] {
                        unresolved += 1;
                        continue;
                    }
                    let m = (mean[i] - t.ratios[i]).abs() / want[i];
                    let s = (sd[i] / want[i] - 1.0).abs();
                    if i == t.absorber {
                        worse(&mut worst_absorber, s, &t.parent);
                        if s > 1e-4 {
                            println!(
                                "  {} absorber sigma {:.4e} against {:.4e}: ratios {:?}",
                                t.parent, sd[i], want[i], t.ratios
                            );
                        }
                    } else {
                        worse(&mut worst_mean, m, &t.parent);
                        worse(&mut worst_sd, s, &t.parent);
                    }
                    if let Some(stated) = stated {
                        assert!(want[i] <= stated * (1.0 + 1e-6), "{library} {}", t.parent);
                    }
                }
                // Over many draws, pull-backs included.
                let rows = t.ratios.len();
                let (mut sum, mut sum2) = (vec![0.0; rows], vec![0.0; rows]);
                let mut floored = 0;
                for replica in 0..draws {
                    let r = &sample(std::slice::from_ref(t), 7, replica, &mut floored)[&t.parent];
                    assert!(r.iter().all(|x| *x >= 0.0), "{library} {}", t.parent);
                    for (i, x) in r.iter().enumerate() {
                        let d = x - t.ratios[i];
                        sum[i] += d;
                        sum2[i] += d * d;
                    }
                }
                pulled += floored;
                pulled_parents += usize::from(floored > 0);
                if floored > 0 {
                    println!(
                        "  pulled back {:.2}% of {}'s draws: ratios {:?}, sigmas {:?}",
                        100.0 * floored as f64 / draws as f64,
                        t.parent,
                        t.ratios,
                        want
                    );
                }
                let n = draws as f64;
                for i in 0..rows {
                    if want[i] < 1e-8 * t.ratios[i] {
                        continue;
                    }
                    let m = sum[i] / n;
                    let sd = ((sum2[i] / n - m * m) * n / (n - 1.0)).max(0.0).sqrt();
                    worse(&mut mc_mean, m.abs() / want[i] * n.sqrt(), &t.parent);
                    worse(&mut mc_sd, (sd / want[i] - 1.0).abs(), &t.parent);
                }
            }
            println!(
                "{library}: sampled {} (by modes {by_modes:?}; {near_zero} with {skewed_modes} \
                 modes under five sigmas from zero), partly held {}, sigma adjusted {}, too \
                 wide {}, no sigma {}; set up in {setup:.2?}",
                c.splits.len(),
                c.partly_held.len(),
                c.sigma_adjusted.len(),
                c.too_wide.len(),
                c.without.len(),
            );
            println!(
                "  exact: worst drawn-mode mean error {:.1e} sigma ({}), sigma error {:.1e} \
                 ({}), absorber sigma error {:.1e} ({}); {unresolved} sigmas under rounding",
                worst_mean.0,
                worst_mean.1,
                worst_sd.0,
                worst_sd.1,
                worst_absorber.0,
                worst_absorber.1
            );
            println!(
                "  {draws} draws each: worst mean error {:.2} standard errors ({}), worst sigma \
                 error {:.3} ({}), {pulled} draws pulled back over {pulled_parents} parents",
                mc_mean.0, mc_mean.1, mc_sd.0, mc_sd.1
            );
            println!("  too wide: {:?}", c.too_wide);
            assert!(worst_mean.0 < 1e-6 && worst_sd.0 < 1e-6, "{library}");
            for name in [
                "Am242_m1", "Np236", "As86", "Pa230", "Cf252", "Xe113", "Ag124",
            ] {
                if let Some(t) = c.splits.iter().find(|t| t.parent == name) {
                    let (_, sd) = exact_moments(t);
                    println!(
                        "  {name}: ratios {:?}, sigmas {:?}, drawn {:?}",
                        t.ratios,
                        realised(t),
                        sd
                    );
                }
            }
        }
    }

    #[test]
    fn the_stream_does_not_collide_with_any_other() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
        let chain = yani::parse_chain_arrow(&path).expect("parse chain");
        // Stream 0 is the cross sections, keyed on the bare name ordinal.
        let tags = [
            0,
            crate::uncertainty::HALF_LIFE_STREAM,
            crate::uncertainty::DECAY_ENERGY_STREAM,
            crate::uncertainty::DECAY_PHOTON_STREAM,
            DECAY_BRANCHING_STREAM,
            crate::resonance_sampling::RESONANCE_PARAMETER_STREAM,
        ];
        let mut keys = std::collections::HashSet::new();
        for name in chain.keys() {
            for t in tags {
                let k = name_ordinal(name) ^ t;
                assert!(keys.insert(k), "{name} collides under tag {t:#x}");
                assert_ne!(
                    k,
                    crate::statistical::STATISTICAL_STREAM,
                    "{name} hits the statistical stream"
                );
                for i in 0..64u32 {
                    assert_ne!(
                        k,
                        crate::flux_uncertainty::FLUX_STREAM ^ i,
                        "{name} hits flux stream {i}"
                    );
                }
            }
        }
    }
}
