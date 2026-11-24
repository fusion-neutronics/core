//! Stopping a run once every tally bin passes its statistical checks.
//!
//! A run with no stop condition (no particle cap, wall-time budget or
//! convergence target) goes on until every scored bin of every tally passes
//! the checks of a [`ConvergenceLevel`]. Each bin is judged as if it were a
//! single-bin tally: from its own moments over all histories, its own
//! convergence history and, at the strict level, its own large-score tail.
//! [`BinConvergence`] keeps that per-bin history between checkpoints.

use crate::result::{ConvergencePoint, StatisticalChecks};
use crate::welford::{hill_tail_slope, merge_top, AggMoments, WelfordWorkerState};

/// How many statistical checks every bin must pass. Each level includes
/// every check of the looser ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvergenceLevel {
    /// Relative error below 0.10, the mean stable, the relative error
    /// decreasing and falling as `1/sqrt(N)`, and the figure of merit
    /// stable with no trend. Uses only each bin's mean and variance.
    Loose,
    /// Loose plus the variance of the variance below 0.10, decreasing and
    /// falling as `1/N`. Keeps each bin's third and fourth moments.
    #[default]
    Standard,
    /// Standard plus a large-score tail slope above 3. Keeps each bin's
    /// largest scores (about 1.6 KiB per scored bin per worker thread).
    Strict,
}

impl ConvergenceLevel {
    /// Every level, loosest first.
    pub const ALL: [ConvergenceLevel; 3] = [Self::Loose, Self::Standard, Self::Strict];

    pub fn name(self) -> &'static str {
        match self {
            Self::Loose => "loose",
            Self::Standard => "standard",
            Self::Strict => "strict",
        }
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|l| l.name() == name)
            .ok_or_else(|| {
                format!(
                    "unknown convergence level {name:?}: use \"loose\", \"standard\" (the \
                     default) or \"strict\", each checking more than the one before"
                )
            })
    }

    /// Whether bins keep their third and fourth moments.
    pub fn needs_higher_moments(self) -> bool {
        self != Self::Loose
    }

    /// Whether bins keep their largest scores.
    pub fn needs_tail(self) -> bool {
        self == Self::Strict
    }

    /// The checks of this level, each `Some(true)` when it passed.
    pub fn checks(self, c: &StatisticalChecks) -> Vec<Option<bool>> {
        let mut v = vec![
            c.relative_error_ok,
            c.mean_stable,
            c.relative_error_decreasing,
            c.relative_error_rate_ok,
            c.figure_of_merit_stable,
            c.figure_of_merit_no_trend,
        ];
        if self.needs_higher_moments() {
            v.extend([
                c.variance_of_variance_ok,
                c.variance_of_variance_decreasing,
                c.variance_of_variance_rate_ok,
            ]);
        }
        if self.needs_tail() {
            v.push(c.tail_slope_ok);
        }
        v
    }

    /// Number of checks in this level.
    pub fn n_checks(self) -> usize {
        self.checks(&StatisticalChecks::default()).len()
    }
}

/// Most history points kept per bin. When full, every other older point is
/// dropped, so the history keeps spanning the whole run in log N.
const MAX_POINTS: usize = 64;

/// The required relative error and variance of the variance, used to
/// project how many histories a bin still needs.
const THRESHOLD: f64 = 0.10;

/// One tally's per-bin convergence history: point-major columns of each
/// scored bin's mean, relative error and variance of the variance.
#[derive(Debug, Default)]
struct TallyHistory {
    n_bins: usize,
    mean: Vec<f64>,
    rel: Vec<f32>,
    vov: Vec<f32>,
}

/// The state of a run at a checkpoint, as judged by [`BinConvergence`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvergenceStatus {
    /// Every scored bin of every tally passed every check of the level.
    pub converged: bool,
    /// The fewest of the level's checks any scored bin passes.
    pub passing: usize,
    /// Projected histories until the relative error (and, above loose, the
    /// variance of the variance) of every scored bin reaches 0.10; `None`
    /// while some tally has no scored bin.
    pub needed: Option<u64>,
    /// The bin projected to need the most histories: tally index, bin
    /// index and its relative error.
    pub worst: Option<(usize, usize, f64)>,
}

/// Per-bin convergence history of every tally of a run, and the decision at
/// each checkpoint. See the module docs.
#[derive(Debug)]
pub struct BinConvergence {
    pub level: ConvergenceLevel,
    /// (histories, transport seconds) of each kept point.
    points: Vec<(u64, f64)>,
    tallies: Vec<TallyHistory>,
    /// Per-bin moments over all histories, rebuilt at each checkpoint.
    scratch: Vec<AggMoments>,
    tops: Vec<Vec<f64>>,
    history: Vec<ConvergencePoint>,
}

impl BinConvergence {
    pub fn new(level: ConvergenceLevel) -> Self {
        Self {
            level,
            points: Vec::new(),
            tallies: Vec::new(),
            scratch: Vec::new(),
            tops: Vec::new(),
            history: Vec::with_capacity(MAX_POINTS + 1),
        }
    }

    /// Record a checkpoint after `n` histories and `elapsed` transport
    /// seconds from the workers' per-bin state, and judge every scored bin.
    pub fn checkpoint<'a>(
        &mut self,
        workers: impl Iterator<Item = &'a WelfordWorkerState> + Clone,
        n: u64,
        elapsed: f64,
    ) -> ConvergenceStatus {
        let Some(first) = workers.clone().next() else {
            return ConvergenceStatus::default();
        };
        if self.tallies.is_empty() {
            self.tallies = first
                .tallies
                .iter()
                .map(|t| TallyHistory {
                    n_bins: t.welford.len(),
                    ..Default::default()
                })
                .collect();
        }
        self.points.push((n, elapsed));
        let n_points = self.points.len();
        // Bonferroni family: every bin of every tally is tested together.
        let family: usize = self.tallies.iter().map(|t| t.n_bins).sum();
        let level = self.level;
        let mut status = ConvergenceStatus {
            converged: true,
            passing: level.n_checks(),
            needed: Some(0),
            worst: None,
        };
        let mut worst_needed = 0u64;
        for (ti, hist) in self.tallies.iter_mut().enumerate() {
            let nb = hist.n_bins;
            // Fold every worker's touched-history moments per bin, then the
            // untouched histories as zeros, giving each bin's moments over
            // all `n` histories.
            self.scratch.clear();
            self.scratch.resize(nb, AggMoments::ZERO);
            if level.needs_tail() {
                self.tops.iter_mut().for_each(Vec::clear);
                self.tops.resize(nb, Vec::new());
            }
            for w in workers.clone() {
                let t = &w.tallies[ti];
                for (b, slot) in self.scratch.iter_mut().enumerate() {
                    if t.welford[b].n_touched > 0 {
                        slot.combine(&t.bin_moments(b));
                    }
                }
                if let Some(top) = t.top.as_ref() {
                    for (dst, src) in self.tops.iter_mut().zip(top) {
                        merge_top(dst, src);
                    }
                }
            }
            let mut any_scored = false;
            for (b, m) in self.scratch.iter_mut().enumerate() {
                let touched = m.n;
                if touched > 0 && touched < n {
                    m.combine(&AggMoments {
                        n: n - touched,
                        ..AggMoments::ZERO
                    });
                }
                let rel = rel_err(m);
                hist.mean.push(m.mean);
                hist.rel.push(rel as f32);
                hist.vov.push(m.variance_of_variance() as f32);
                if touched == 0 || m.mean == 0.0 {
                    continue;
                }
                any_scored = true;
                // This bin's history, as a single-bin tally would keep it.
                self.history.clear();
                for (p, &(pn, pt)) in self.points.iter().enumerate() {
                    let r = hist.rel[p * nb + b] as f64;
                    self.history.push(ConvergencePoint {
                        n_histories: pn,
                        mean: hist.mean[p * nb + b],
                        relative_error: r,
                        variance_of_variance: hist.vov[p * nb + b] as f64,
                        figure_of_merit: if r > 0.0 && pt > 0.0 {
                            1.0 / (r * r * pt)
                        } else {
                            0.0
                        },
                        tail_slope: 0.0,
                        elapsed_secs: pt,
                    });
                }
                let slope = if level.needs_tail() {
                    hill_tail_slope(&self.tops[b])
                } else {
                    0.0
                };
                let checks = StatisticalChecks::aggregate(m, slope, &self.history, family);
                let results = level.checks(&checks);
                let passed = results.iter().filter(|v| **v == Some(true)).count();
                status.converged &= passed == results.len();
                status.passing = status.passing.min(passed);
                let vov = if level.needs_higher_moments() {
                    m.variance_of_variance()
                } else {
                    0.0
                };
                let need = histories_to_converge(n, rel, vov).unwrap_or(n);
                if status.worst.is_none() || need > worst_needed {
                    worst_needed = need;
                    status.worst = Some((ti, b, rel));
                }
            }
            if !any_scored {
                status.converged = false;
                status.needed = None;
            }
        }
        if let Some(needed) = status.needed.as_mut() {
            *needed = worst_needed;
        }
        if n_points > MAX_POINTS {
            self.thin();
        }
        status
    }

    /// Drop every other point of the older half, keeping the newest half
    /// whole: the trend checks read the last half of the history.
    fn thin(&mut self) {
        let n = self.points.len();
        let keep: Vec<usize> = (0..n).filter(|&p| p >= n / 2 || p % 2 == 0).collect();
        self.points = keep.iter().map(|&p| self.points[p]).collect();
        for h in &mut self.tallies {
            let nb = h.n_bins;
            let pick = |col: &[f32]| -> Vec<f32> {
                keep.iter()
                    .flat_map(|&p| col[p * nb..(p + 1) * nb].iter().copied())
                    .collect()
            };
            h.rel = pick(&h.rel);
            h.vov = pick(&h.vov);
            h.mean = keep
                .iter()
                .flat_map(|&p| h.mean[p * nb..(p + 1) * nb].iter().copied())
                .collect();
        }
    }
}

fn rel_err(m: &AggMoments) -> f64 {
    if m.n < 2 || m.mean == 0.0 {
        return 0.0;
    }
    let n = m.n as f64;
    (m.m2 / ((n - 1.0) * n)).max(0.0).sqrt() / m.mean.abs()
}

/// Histories at which a relative error `rel` (`∝ N^-1/2`) and a variance of
/// the variance `vov` (`∝ N^-1`), seen at `n` histories, both reach 0.10.
/// `None` before anything has scored.
pub fn histories_to_converge(n: u64, rel: f64, vov: f64) -> Option<u64> {
    (rel > 0.0).then(|| {
        let n = n as f64;
        (n * (rel / THRESHOLD).powi(2)).max(n * vov / THRESHOLD) as u64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::welford::WelfordWorkerState;

    #[test]
    fn levels_nest_and_parse() {
        let counts: Vec<usize> = ConvergenceLevel::ALL.iter().map(|l| l.n_checks()).collect();
        assert_eq!(counts, [6, 9, 10]);
        for l in ConvergenceLevel::ALL {
            assert_eq!(ConvergenceLevel::from_name(l.name()), Ok(l));
        }
        assert_eq!(ConvergenceLevel::default(), ConvergenceLevel::Standard);
        assert!(ConvergenceLevel::from_name("medium")
            .unwrap_err()
            .contains("\"loose\", \"standard\""));
    }

    #[test]
    fn histories_to_converge_projects_both_rates() {
        assert_eq!(histories_to_converge(1000, 0.2, 0.05), Some(4000));
        assert_eq!(histories_to_converge(1000, 0.05, 0.5), Some(5000));
        assert_eq!(histories_to_converge(1000, 0.0, 0.0), None);
    }

    /// Per-bin M3/M4 kept over touched histories, folded with the untouched
    /// zeros, equal the moments of the full zero-padded sample.
    #[test]
    fn bin_moments_match_the_zero_padded_sample() {
        let mut w = WelfordWorkerState::new(&[2]).with_convergence(true, true);
        let scores = [3.0, 0.0, 1.5, 0.0, 0.0, 7.0, 2.0];
        let mut direct = AggMoments::ZERO;
        for &x in &scores {
            if x != 0.0 {
                w.add_contribution(0, 1, x);
            }
            w.finish_history();
            direct.update(x);
        }
        let mut m = w.tallies[0].bin_moments(1);
        m.combine(&AggMoments {
            n: scores.len() as u64 - m.n,
            ..AggMoments::ZERO
        });
        for (a, b) in [
            (m.mean, direct.mean),
            (m.m2, direct.m2),
            (m.m3, direct.m3),
            (m.m4, direct.m4),
        ] {
            assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
        }
        assert_eq!(
            w.tallies[0].top.as_ref().unwrap()[1],
            vec![1.5, 2.0, 3.0, 7.0]
        );
    }

    /// A bin with an exponential score converges at every level; a tally
    /// with a bin never scored waits only on its scored bins.
    #[test]
    fn a_well_behaved_bin_converges_at_every_level() {
        for level in ConvergenceLevel::ALL {
            let mut w = WelfordWorkerState::new(&[3])
                .with_convergence(level.needs_higher_moments(), level.needs_tail());
            let mut monitor = BinConvergence::new(level);
            let mut rng = 0x2545_f491_4f6c_dd1du64;
            let mut status = ConvergenceStatus::default();
            let mut done = 0u64;
            let mut streak = 0;
            for _ in 0..200 {
                let size = (done / 4).clamp(100, 100_000);
                for _ in 0..size {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
                    w.add_contribution(0, 0, -(1.0 - u).ln());
                    if u < 0.5 {
                        w.add_contribution(0, 1, u);
                    }
                    w.finish_history();
                }
                done += size;
                status = monitor.checkpoint(std::iter::once(&w), done, done as f64 * 1e-6);
                streak = if status.converged { streak + 1 } else { 0 };
                if streak >= 2 {
                    break;
                }
            }
            assert!(status.converged, "{level:?} never converged: {status:?}");
            assert_eq!(status.passing, level.n_checks());
            assert!(monitor.points.len() <= MAX_POINTS);
        }
    }

    /// Judged on the same histories and timings, a stricter level never
    /// stops before a looser one: each level only adds checks. (Separate
    /// runs can stop in any order, as each run's own wall time enters the
    /// figure-of-merit checks.)
    #[test]
    fn stricter_levels_never_stop_first_on_the_same_run() {
        let mut w = WelfordWorkerState::new(&[2]).with_convergence(true, true);
        let mut monitors = ConvergenceLevel::ALL.map(BinConvergence::new);
        let mut first: [Option<u64>; 3] = [None; 3];
        let mut streak = [0u32; 3];
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut done = 0u64;
        for _ in 0..200 {
            let size = (done / 4).clamp(100, 100_000);
            for _ in 0..size {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
                // A Pareto score (tail slope ~4) for bin 0, uniform for bin 1.
                w.add_contribution(0, 0, (1.0 - u).powf(-1.0 / 3.0));
                w.add_contribution(0, 1, u);
                w.finish_history();
            }
            done += size;
            for (i, m) in monitors.iter_mut().enumerate() {
                let s = m.checkpoint(std::iter::once(&w), done, done as f64 * 1e-6);
                streak[i] = if s.converged { streak[i] + 1 } else { 0 };
                if streak[i] >= 2 && first[i].is_none() {
                    first[i] = Some(done);
                }
            }
            if first.iter().all(Option::is_some) {
                break;
            }
        }
        let first = first.map(|f| f.expect("every level converges"));
        assert!(first[0] <= first[1] && first[1] <= first[2], "{first:?}");
    }

    /// A bin whose score keeps drifting never converges.
    #[test]
    fn a_drifting_bin_does_not_converge() {
        let mut w = WelfordWorkerState::new(&[1]).with_convergence(true, false);
        let mut monitor = BinConvergence::new(ConvergenceLevel::Standard);
        let mut done = 0u64;
        for chunk in 0..40 {
            let size = (done / 4).clamp(100, 100_000);
            for h in 0..size {
                let x = (done + h) as f64 * 1e-3 + (h % 7) as f64;
                w.add_contribution(0, 0, x);
                w.finish_history();
            }
            done += size;
            let status = monitor.checkpoint(std::iter::once(&w), done, done as f64 * 1e-6);
            assert!(!status.converged || chunk < 4, "drift passed at {done}");
        }
    }
}
