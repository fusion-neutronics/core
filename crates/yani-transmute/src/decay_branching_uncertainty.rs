//! Decay branching ratio uncertainty.
//!
//! MT=457 gives each decay mode a ratio and a sigma, and no covariance between
//! the modes. The ratios of a parent sum to a fixed total `T` (one, after the
//! converter has put any evaluated residual on the largest mode), so the modes
//! cannot be independent: every replica must keep `T`, which makes the
//! deviations a zero-sum Gaussian. Its covariance is fixed by the stated
//! sigmas for two modes, and for three, and is not for four or more. Here it
//! is the zero-sum covariance that reproduces every stated sigma and has the
//! largest entropy, so it assumes no correlation the sum rule does not force:
//!
//! - two modes: one degree of freedom, `r1 = b1 + sigma z`, `r2 = T - r1`. An
//!   evaluator writes one number for a complementary pair (Bi212 0.3594 and
//!   0.6406, both 0.0006), or states it on one mode only and leaves the other
//!   as its complement (JENDL-5.0 Bi212 gives alpha 0.0006 and beta- none);
//! - three modes: the covariance is the unique one with the three variances
//!   and rows summing to zero;
//! - more: the maximum-entropy one. Where one mode's sigma is the quadrature
//!   sum of the others' (JENDL-5.0 writes Li11 and Cs114 that way) it is the
//!   evaluator's own model, the minor modes independent and the dominant one
//!   their complement.
//!
//! A zero-sum Gaussian can realise sigmas only if none exceeds the sum of the
//! others (the deviations are the sides of a closed polygon). Where the
//! largest does (two modes with different sigmas, or ENDF/B-VIII.1 Hf182m's
//! 0.076 beside 0.020 and 0.0033), the others are realised exactly, all moving together,
//! and the largest takes the sum of theirs, the most the sum rule allows. The
//! parent is sampled and reported in `sigma_reduced`.
//!
//! A mode with no stated sigma has no variance to match. A single one is the
//! complement of the others, which are then independent (the maximum-entropy
//! reading, and how an evaluator writes a dominant mode as one minus the rest:
//! ENDF/B-VIII.1 P28). Two or more are held at their ratios, and the stated
//! modes share what is left.
//!
//! A mode under [`MIN_SIGMAS`] sigmas from zero (its stated one, or for a
//! complement the one it would take) is held at its ratio too: a Gaussian that
//! wide must be truncated, which the evaluation does not describe. The rest
//! share what is left, so a parent can be sampled with some of its modes held
//! (`partly_held`), and is held whole (`too_wide`) when no mode can move. With
//! every free mode at least five sigmas clear, the chance that a draw leaves
//! the simplex is a few in ten million; such a draw is pulled back towards the
//! nominal ratios until every ratio is non-negative, keeping the total, and
//! counted in `floored`.
//!
//! A spontaneous-fission mode is a mode like any other here: it is a share of
//! the parent's decays, whether or not the chain models a product for it.
//! Modes that make the same daughter are separate rows with separate sigmas,
//! and the daughter takes their sum.
//!
//! Only the inventory moves. A parent's per-decay emission (its lines and
//! decay energy) is the decay scheme's and stays nominal, since no correlation
//! between a ratio and an absolute line intensity is published, and the report
//! lists it under `not_perturbed`.

use std::collections::{BTreeSet, HashMap};

use yani::ChainNuclide;

use crate::covariance_sample::{name_ordinal, standard_normals};

/// Keeps the decay-branching streams clear of every other per-nuclide stream.
pub(crate) const DECAY_BRANCHING_STREAM: u32 = 0xDB2A_0C5E;

/// How many sigmas a free mode's ratio must stand from zero to be sampled.
pub(crate) const MIN_SIGMAS: f64 = 5.0;

/// Two stated sigmas closer than this, relatively, are the same number.
const SAME_SIGMA: f64 = 1.0e-9;

/// The largest sigma within this, relatively, of the sum of the others closes
/// the polygon flat: every other mode moves together. Short of it the
/// maximum-entropy covariance is nearly singular and slow to find, and the
/// flat one differs from it by less than this in any sigma.
const FLAT: f64 = 1.0e-6;

/// One parent whose decay branching is sampled.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Split {
    pub(crate) parent: String,
    /// The nominal ratios, row by row of `decays`.
    pub(crate) ratios: Vec<f64>,
    /// Their sum, which every replica keeps.
    pub(crate) total: f64,
    /// The row that takes `total` minus every other row.
    pub(crate) absorber: usize,
    /// One column per standard normal: every row's deviation per unit of it,
    /// zero on a held row, minus the sum of the others on the absorber.
    pub(crate) directions: Vec<Vec<f64>>,
}

impl Split {
    /// The ratios for the standard normals `z`, one per direction, and the
    /// factor the deviation was scaled by to keep every ratio non-negative
    /// (1 unless the draw left the simplex).
    pub(crate) fn ratios_for(&self, z: &[f64]) -> (Vec<f64>, f64) {
        debug_assert_eq!(z.len(), self.directions.len());
        let deviation: Vec<f64> = (0..self.ratios.len())
            .map(|i| {
                self.directions
                    .iter()
                    .zip(z)
                    .fold(0.0, |acc, (a, z)| acc + a[i] * z)
            })
            .collect();
        // The largest share of the deviation every row can take.
        let scale = self
            .ratios
            .iter()
            .zip(&deviation)
            .filter(|(_, d)| **d < 0.0)
            .map(|(b, d)| b / -d)
            .fold(1.0_f64, f64::min);
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

    /// The step, in units of direction `k`'s normal, that moves no row by more
    /// than `relative` of its nominal ratio.
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
    /// Held at nominal: every mode that could move is under [`MIN_SIGMAS`]
    /// sigmas from zero.
    pub(crate) too_wide: BTreeSet<String>,
    /// Sampled, with one or more modes held at their ratios.
    pub(crate) partly_held: BTreeSet<String>,
    /// Sampled, with the largest sigma above the sum of the others and
    /// realised as that sum.
    pub(crate) sigma_reduced: BTreeSet<String>,
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
                sigma_reduced,
            } => {
                if partly_held {
                    out.partly_held.insert(name.clone());
                }
                if sigma_reduced {
                    out.sigma_reduced.insert(name.clone());
                }
                out.splits.push(split);
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
        split: Split,
        partly_held: bool,
        sigma_reduced: bool,
    },
}

/// One parent's sampling, from its ratios and stated sigmas, row by row.
fn split(parent: &str, ratios: &[f64], sigmas: &[Option<f64>]) -> Sorted {
    let n = ratios.len();
    let total = ratios.iter().fold(0.0, |acc, b| acc + b);
    let unstated: Vec<usize> = (0..n).filter(|&i| sigmas[i].is_none()).collect();
    if unstated.len() == n {
        return Sorted::Without;
    }
    // Stated modes the Gaussian keeps clear of zero.
    let free: Vec<(usize, f64)> = (0..n)
        .filter_map(|i| sigmas[i].map(|s| (i, s)))
        .filter(|&(i, s)| ratios[i] >= MIN_SIGMAS * s)
        .collect();
    let too_close = free.len() + unstated.len() < n;
    // A lone unstated mode is the complement, if it too stays clear of zero.
    let complement = match unstated[..] {
        [c] if !free.is_empty() => {
            let s = free.iter().map(|(_, s)| s * s).sum::<f64>().sqrt();
            (ratios[c] >= MIN_SIGMAS * s).then_some(c)
        }
        _ => None,
    };
    let movable = free.len() + usize::from(complement.is_some());
    if movable < 2 {
        // Nothing to trade with. Held for want of sigmas unless a stated one
        // was set aside for being too wide.
        let wide = too_close || (unstated.len() == 1 && !free.is_empty());
        return if wide {
            Sorted::TooWide
        } else {
            Sorted::Without
        };
    }
    let partly_held = movable < n;
    let column = |entries: &[(usize, f64)], absorber: usize| -> Vec<f64> {
        let mut a = vec![0.0; n];
        for &(i, v) in entries {
            a[i] = v;
        }
        a[absorber] = -entries.iter().fold(0.0, |acc, (_, v)| acc + v);
        a
    };
    let mut sigma_reduced = false;
    let (absorber, directions) = if let Some(c) = complement {
        // The stated modes independent, the complement their sum rule.
        let directions = free.iter().map(|&(i, s)| column(&[(i, s)], c)).collect();
        (c, directions)
    } else {
        // The largest sigma absorbs; on a tie, the last row, so a pair
        // stating one number draws on its first row.
        let largest = free.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
        let k = free
            .iter()
            .rposition(|(_, s)| *s >= largest * (1.0 - SAME_SIGMA))
            .expect("a free mode holds the largest sigma");
        let (absorber, s_k) = free[k];
        let others: Vec<(usize, f64)> = free
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, m)| *m)
            .collect();
        let sum_others = others.iter().fold(0.0, |acc, (_, s)| acc + s);
        if s_k >= sum_others * (1.0 - FLAT) {
            // The polygon closes flat: every other mode moves together.
            sigma_reduced = s_k > sum_others * (1.0 + SAME_SIGMA);
            (absorber, vec![column(&others, absorber)])
        } else {
            let s: Vec<f64> = others.iter().map(|(_, s)| *s).collect();
            let w: Vec<f64> = s.iter().map(|s| s / s_k).collect();
            let l = max_entropy_factor(&w);
            let m = others.len();
            let directions = (0..m)
                .map(|col| {
                    let entries: Vec<(usize, f64)> = others
                        .iter()
                        .enumerate()
                        .map(|(row, &(i, si))| (i, si * l[row * m + col]))
                        .collect();
                    column(&entries, absorber)
                })
                .collect();
            (absorber, directions)
        }
    };
    Sorted::Sampled {
        split: Split {
            parent: parent.to_string(),
            ratios: ratios.to_vec(),
            total,
            absorber,
            directions,
        },
        partly_held,
        sigma_reduced,
    }
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

/// One replica's standard normals for a parent: one per direction, on a
/// stream keyed on the parent's name, so a parent splits the same way in every
/// spectrum and every material of a run.
pub(crate) fn normals(split: &Split, base_seed: u64, replica: u64) -> Vec<f64> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    let seed = yamc_rng::secondary_seed(
        replica_seed,
        name_ordinal(&split.parent) ^ DECAY_BRANCHING_STREAM,
    );
    let mut state = yamc_rng::expand_seed(seed);
    standard_normals(&mut state, split.directions.len())
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
            Sorted::Sampled { split, .. } => split,
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

    /// Each row's realised sigma, from the directions.
    fn realised(t: &Split) -> Vec<f64> {
        (0..t.ratios.len())
            .map(|i| t.directions.iter().map(|a| a[i] * a[i]).sum::<f64>().sqrt())
            .collect()
    }

    fn correlation(t: &Split, i: usize, j: usize) -> f64 {
        let s = realised(t);
        t.directions.iter().map(|a| a[i] * a[j]).sum::<f64>() / (s[i] * s[j])
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
            // 0.0037 / 0.0015 is 2.5 sigmas.
            parent(
                "Wide",
                vec![
                    mode("ec/beta+", Some("A"), 0.9963, None),
                    mode("p", Some("B"), 0.0037, Some(0.0015)),
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
            // Three modes: the alpha at 3.3 sigmas is held, the rest move.
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
        assert_eq!(
            sampled,
            vec![
                ("Equal", 1),
                ("Np236", 2),
                ("OneSided", 0),
                ("Three", 1),
                ("Unequal", 1)
            ]
        );
        let set = |xs: &[&str]| -> BTreeSet<String> { xs.iter().map(|s| s.to_string()).collect() };
        assert_eq!(c.sigma_reduced, set(&["Unequal"]));
        assert_eq!(c.partly_held, set(&["Np236"]));
        assert_eq!(c.too_wide, set(&["Wide"]));
        assert_eq!(c.without, set(&["OneOfThree", "Unstated"]));
    }

    /// A pair draws one normal on its stream and moves its rows by it exactly
    /// as the two-mode sampling always has: bit for bit.
    #[test]
    fn a_pair_draws_as_before() {
        for (rows, drawn, other) in [
            (vec![(0.6406, Some(6e-4)), (0.3594, Some(6e-4))], 0, 1),
            (vec![(0.8914, Some(0.0)), (0.1086, Some(0.0013))], 1, 0),
            (vec![(0.8914, None), (0.1086, Some(0.0013))], 1, 0),
        ] {
            let sigmas: Vec<Option<f64>> = rows.iter().map(|r| r.1.filter(|s| *s > 0.0)).collect();
            let ratios: Vec<f64> = rows.iter().map(|r| r.0).collect();
            let Sorted::Sampled { split: t, .. } = split("Bi212", &ratios, &sigmas) else {
                panic!("{rows:?} is sampled");
            };
            let sigma = sigmas[drawn].unwrap();
            let total = ratios[drawn] + ratios[other];
            for replica in 0..64 {
                let mut floored = 0;
                let got = &sample(std::slice::from_ref(&t), 3, replica, &mut floored)["Bi212"];
                // The two-mode rule: one deviate on the parent's stream.
                let replica_seed = yamc_rng::history_seed(3, replica);
                let seed = yamc_rng::secondary_seed(
                    replica_seed,
                    name_ordinal("Bi212") ^ DECAY_BRANCHING_STREAM,
                );
                let z = standard_normals(&mut yamc_rng::expand_seed(seed), 1)[0];
                let r = ratios[drawn] + sigma * z;
                assert_eq!(got[drawn].to_bits(), r.to_bits());
                assert_eq!(got[other].to_bits(), (total - r).to_bits());
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
                assert!((got / want - 1.0).abs() < 1e-9, "{got} against {want}");
            }
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        let k = 3 - i - j;
                        let want = (s[k] * s[k] - s[i] * s[i] - s[j] * s[j]) / (2.0 * s[i] * s[j]);
                        assert!((correlation(&t, i, j) - want).abs() < 1e-9);
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
        // Li11's 0.01374773 on 0.073 is the larger, and absorbs.
        assert_eq!(t.absorber, 0);
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
            sigma_reduced,
            partly_held,
        } = split_of(&rows)
        else {
            panic!("sampled");
        };
        // 0.0745 is 4.0 sigmas from zero: held.
        assert!(sigma_reduced && partly_held);
        assert_eq!(t.absorber, 0);
        let s = realised(&t);
        assert_eq!(s[3], 0.0);
        assert!((s[1] / 0.003304615 - 1.0).abs() < 1e-12);
        assert!((s[2] / 0.02 - 1.0).abs() < 1e-12);
        assert!((s[0] / (0.003304615 + 0.02) - 1.0).abs() < 1e-12);
        assert!((correlation(&t, 1, 2) - 1.0).abs() < 1e-12);
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
        assert_eq!(t.directions.len(), 2);
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
    /// checked against the sigmas it was built from: a free mode realises its
    /// own, an absorber at most its own, and a held mode does not move.
    ///
    /// Opt-in, since it reads whole local libraries. Set
    /// `YANI_DECAY_BRANCHING_DATA` to a directory holding the published
    /// `transmutation-<library>.arrow/decay/`, then run `cargo test -p
    /// yani-transmute --lib real_libraries -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads whole local libraries; set YANI_DECAY_BRANCHING_DATA"]
    fn real_libraries_realise_their_sigmas() {
        let root = std::path::PathBuf::from(
            std::env::var_os("YANI_DECAY_BRANCHING_DATA").expect("YANI_DECAY_BRANCHING_DATA"),
        );
        for library in ["endf-b8.1", "jeff-4.0", "jendl-5.0"] {
            let decay = root
                .join(format!("transmutation-{library}.arrow"))
                .join("decay");
            if !decay.is_dir() {
                println!("{library}: not present");
                continue;
            }
            let chain = yani::load_chain_parts(&decay.to_string_lossy(), None, None, None)
                .expect("chain loads")
                .chain;
            let c = candidates(&chain);
            let mut by_modes: std::collections::BTreeMap<usize, usize> = Default::default();
            let mut entropy = 0;
            for t in &c.splits {
                *by_modes.entry(t.ratios.len()).or_default() += 1;
                let cn = &chain[&t.parent];
                let s = realised(t);
                // Two or more normals and a stated sigma on the absorber: not
                // flat, not a complement, so the maximum-entropy covariance.
                if t.directions.len() > 1
                    && cn.decays[t.absorber]
                        .branching_uncertainty
                        .is_some_and(|s| s > 0.0)
                {
                    entropy += 1;
                }
                for (i, d) in cn.decays.iter().enumerate() {
                    let stated = d.branching_uncertainty.filter(|s| *s > 0.0);
                    match stated {
                        _ if s[i] == 0.0 => {}
                        Some(want) if i == t.absorber => assert!(
                            s[i] <= want * (1.0 + 1e-6),
                            "{library} {}: absorber {} against {want}",
                            t.parent,
                            s[i]
                        ),
                        Some(want) => assert!(
                            (s[i] / want - 1.0).abs() < 1e-6,
                            "{library} {} row {i}: {} against {want}",
                            t.parent,
                            s[i]
                        ),
                        None => assert_eq!(i, t.absorber, "{library} {}", t.parent),
                    }
                    assert!(t.ratios[i] >= MIN_SIGMAS * s[i] * (1.0 - 1e-6));
                }
            }
            println!(
                "{library}: sampled {} (by modes {by_modes:?}, {entropy} with a \
                 maximum-entropy covariance over three or more moving modes), \
                 partly held {}, sigma reduced {}, too wide {}, no sigma {}",
                c.splits.len(),
                c.partly_held.len(),
                c.sigma_reduced.len(),
                c.too_wide.len(),
                c.without.len(),
            );
            for name in [
                "K40", "Bi212", "Am242_m1", "Np236", "Cf252", "Es254_m1", "Rb98", "In130",
            ] {
                if let Some(t) = c.splits.iter().find(|t| t.parent == name) {
                    println!(
                        "  {name}: ratios {:?}, realised sigmas {:?}",
                        t.ratios,
                        realised(t)
                    );
                }
            }
            assert!(
                by_modes.range(3..).next().is_some(),
                "{library}: no parent of three or more modes sampled"
            );
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
