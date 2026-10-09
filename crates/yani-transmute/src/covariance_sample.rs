//! Sample nuclear cross sections from their covariance, and read every
//! spectrum's reaction rates off each draw.
//!
//! A replica draws each nuclide's cross sections once, as a field over its
//! covariance cells (see [`CellField`](crate::covariance_fold::CellField)),
//! and every spectrum, step and channel reads its rate off that one draw. So
//! one evaluation is one uncertainty everywhere it is used: a nuclide under
//! two spectra moves in both as one perturbed cross section moves it, and a
//! spectrum added to a schedule does not change what the others draw.
//!
//! ```text
//! relative cells   Σ_N = ln(1 + C),  y = L_N z,  m_k = exp(y_k - Σ_N,kk / 2)
//! absolute cells   a = L_A z'                      (barns)
//! short-range      ΔB_j ~ N(0, F_k ΔE_k w_j)       (independent pieces)
//! rate             R'_i = R_i + Σ_k p_ik (m_k - 1) + Σ_k q_ik a_k + Σ_j w_ij ΔB_j
//! ```
//!
//! with `p`, `q` and `w` the spectrum's partial rates, partial fluxes and
//! flux densities on the cells. A rate is linear in the cross section, so the
//! last line is exact, not a linearization. The relative multipliers are a
//! multivariate lognormal with mean one and the evaluation's own covariance
//! (Žerovnik et al., NIM A 727 (2013) 33), so every perturbed cross section
//! is positive, and so is every rate that only adds reactions.
//!
//! # The eigensolver
//!
//! Every factorization is an eigendecomposition, by Householder
//! tridiagonalization and implicit QL ([`symmetric_eigen`]): bit-reproducible
//! across platforms because it is pure arithmetic in a fixed order, O(n³)
//! once on fields that run to hundreds of cells, and it gives the
//! eigenvectors that set the round-off negatives of a PSD matrix to zero. A
//! Cholesky would be the obvious choice if the matrices were positive
//! definite, and they are only semi-definite at best.
//!
//! # Repair keeps every evaluated sigma
//!
//! MF=33 matrices are frequently not positive semi-definite as evaluated, and
//! a covariance that is not PSD has no square root, so it has to be repaired
//! to be sampled. Clipping its negative eigenvalues would only ever add
//! variance, so the repair is instead on the correlation matrix of the cells
//! with a positive stated variance: it is replaced by the nearest correlation
//! matrix (Higham; see [`crate::nearest_correlation`]) and rescaled by the
//! evaluated sigmas,
//!
//! ```text
//! R = D^-1/2 C D^-1/2,   R' = nearest correlation matrix to R,   C' = D^1/2 R' D^1/2
//! ```
//!
//! so every cell is sampled at its evaluated sigma exactly and only
//! correlations move, by as little as any PSD matrix allows. A cell stated
//! with a negative variance, or a zero one and a covariance to another cell,
//! has no correlation to keep and is held at nominal. The repair is on the
//! evaluation's own `C`, before the lognormal transform; where `ln(1 + C')`
//! is still not PSD, its own correlation matrix is repaired the same way,
//! keeping every log-space variance and so again every sigma (see
//! [`LognormalLimit`]). The relative and absolute cells are repaired apart,
//! as they are drawn apart. What each repair did is recorded per nuclide in
//! [`FieldRepair`] and per spectrum in [`Repair`], with each channel's
//! evaluated sigma next to its sampled one: a channel that folds several
//! cells reads their correlations, so its sigma can still move, either way.
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
/// scale. The eigensolver's round-off on `C` is absolute, about `eps · λ_max`, and a
/// small channel can carry an O(1) share of a null-space eigenvector, so on
/// `C` a rank-one matrix whose channels span a few orders in sigma already
/// reads as needing repair. On `R` every diagonal is one, so the trace is
/// `m`, and the round-off is about `eps` times the largest `|λ|` of `R`. That
/// is at most `m` on a PSD `R` and near it close to the PSD boundary, where
/// the test decides anything, so the round-off is about `m · eps`, orders
/// below `m · 1e-12`, whatever the spread. (A far-from-PSD `R` can have
/// `λ_max > m`, but its negative eigenvalue is then far past the threshold.) The spread itself is real: TENDL-2017 has channels at a relative
/// sigma up to 5.6e8, a variance near 3e17, and a threshold against the
/// `λ_max` of `C` would sit far above a real repair among the ordinary
/// channels beside one.
///
/// Below the threshold the matrix is PSD to within round-off and the
/// eigenvalues that come back negative are set to zero in the factor, since
/// they have no square root either. What that adds to a channel is round-off
/// of the decomposition of `C`, about `eps · λ_max`, and it is not counted as
/// a repair. It is not hidden either: the rate-weighted headline reads every
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
    /// The relative sigma the channel is sampled at after the repair. Every
    /// cell keeps its evaluated sigma, but a channel that folds several cells
    /// reads their correlations, which the repair moves, so this can sit
    /// either side of the evaluated sigma.
    pub sampled: f64,
}

impl ChannelSigma {
    /// `sqrt(C_ii)`, the folded relative sigma the evaluation states, or
    /// `None` when the stated variance is negative and so has no sigma.
    pub fn evaluated_sigma(&self) -> Option<f64> {
        (self.evaluated_variance >= 0.0).then(|| self.evaluated_variance.sqrt())
    }

    /// `sampled / evaluated - 1`: how far the repair moved this channel's
    /// sigma, negative where it narrowed. Infinite when the repair gave a
    /// spread to a channel whose stated variance is zero or negative.
    pub fn change(&self) -> f64 {
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
    /// What the repair of the nuclide's cell covariance did, the same under
    /// every spectrum.
    pub field: FieldRepair,
    /// Every channel the spectrum reads, in the fold's kind order.
    pub channels: Vec<ChannelSigma>,
}

/// Where a relative covariance is not a lognormal's, and how far the sampled
/// one is from it.
///
/// The multipliers are drawn as `exp(y - diag(Σ_N)/2)` with `Σ_N = ln(1 + C)`
/// elementwise, which carries `C` exactly only where `Σ_N` is positive
/// semi-definite and every `1 + C_kl` is positive. Two fully correlated cells
/// with different sigmas, or an anticorrelation with `1 + C_kl <= 0`, are not
/// a lognormal's, and the nearest one is sampled instead: where `Σ_N` is not
/// PSD its correlation matrix is replaced by the nearest correlation matrix,
/// which keeps every `Σ_N,kk` and so every sigma, and moves correlations
/// only. That is a property of the distribution rather than a defect of the
/// data, so it is reported here and not as a repair, and it does not count as
/// a gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LognormalLimit {
    /// Cells (or flux bins) whose sampled sigma, or whose correlation with
    /// another cell, differs from the stated one by more than
    /// [`LOGNORMAL_LIMIT_TOLERANCE`].
    pub cells: usize,
    /// The largest `|sampled sigma / stated sigma - 1|` over the cells.
    pub largest_sigma_change: f64,
    /// The largest change in a correlation coefficient,
    /// `|C'_kl - C_kl| / sqrt(C_kk C_ll)`, over the pairs of cells.
    pub largest_correlation_change: f64,
    /// The log-space repair, where `Σ_N` was not PSD past
    /// [`REPAIR_TOLERANCE`]: its eigenvalue and correlation changes are of the
    /// correlation matrix of `Σ_N`, not of `C`. `None` where the sampled
    /// covariance differs only by round-off of a PSD `Σ_N`, or by entries
    /// with no logarithm.
    pub log_space_repair: Option<FieldRepair>,
}

/// The relative change below which a sampled sigma or correlation counts as
/// the stated one: well above the round-off of the transform, well below
/// anything that moves an answer.
pub const LOGNORMAL_LIMIT_TOLERANCE: f64 = 1.0e-9;

/// Compare the relative covariance a lognormal draw carries, `sampled`, with
/// the one it was matched to, `stated`, both row-major `n x n`. `None` when
/// they agree to [`LOGNORMAL_LIMIT_TOLERANCE`] everywhere.
pub(crate) fn lognormal_limit(stated: &[f64], sampled: &[f64], n: usize) -> Option<LognormalLimit> {
    let mut affected = vec![false; n];
    let mut largest_sigma_change = 0.0_f64;
    let mut largest_correlation_change = 0.0_f64;
    for k in 0..n {
        let s = stated[k * n + k];
        if s > 0.0 {
            let change = (sampled[k * n + k].max(0.0) / s).sqrt() - 1.0;
            if change.abs() > LOGNORMAL_LIMIT_TOLERANCE {
                affected[k] = true;
            }
            largest_sigma_change = largest_sigma_change.max(change.abs());
        }
    }
    for k in 0..n {
        for l in (k + 1)..n {
            let norm = (stated[k * n + k] * stated[l * n + l]).sqrt();
            if norm <= 0.0 || norm.is_nan() {
                continue;
            }
            let change = (sampled[k * n + l] - stated[k * n + l]).abs() / norm;
            if change > LOGNORMAL_LIMIT_TOLERANCE {
                affected[k] = true;
                affected[l] = true;
            }
            largest_correlation_change = largest_correlation_change.max(change);
        }
    }
    let cells = affected.iter().filter(|a| **a).count();
    (cells > 0).then_some(LognormalLimit {
        cells,
        largest_sigma_change,
        largest_correlation_change,
        log_space_repair: None,
    })
}

/// `Σ_N = ln(1 + C)` elementwise for a relative covariance `C`, row-major
/// `n x n`, and whether an entry had no logarithm. A diagonal at or below
/// `-1` is not a variance and maps to zero; an off-diagonal with
/// `1 + C_kl <= 0`, which no lognormal can carry, maps to the most negative
/// covariance its two diagonals allow.
pub(crate) fn log_covariance(relative: &[f64], n: usize) -> (Vec<f64>, bool) {
    let mut log = vec![0.0; n * n];
    let mut substituted = false;
    for k in 0..n {
        let c = relative[k * n + k];
        log[k * n + k] = if c > -1.0 { c.ln_1p() } else { 0.0 };
    }
    for k in 0..n {
        for l in 0..n {
            if k != l {
                let c = relative[k * n + l];
                log[k * n + l] = if c > -1.0 {
                    c.ln_1p()
                } else {
                    substituted = true;
                    -(log[k * n + k].max(0.0) * log[l * n + l].max(0.0)).sqrt()
                };
            }
        }
    }
    (log, substituted)
}

/// What the repair of one nuclide's evaluated cell covariance did (see
/// [`Repair`] for the per-spectrum record), over its relative and absolute
/// cells together.
///
/// `R` is the correlation matrix of the cells with a positive stated
/// variance and `R'` the nearest correlation matrix to it that was sampled.
/// Every such cell keeps its evaluated variance exactly, so the correlations
/// are all that moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldRepair {
    /// The most negative eigenvalue of `R` before the repair (one where fewer
    /// than two cells have a positive variance).
    pub lambda_min: f64,
    /// The largest `|R'_kl - R_kl|`: the most any correlation moved.
    pub largest_correlation_change: f64,
    /// `||R' - R||_F` over both triangles.
    pub correlation_frobenius_change: f64,
    /// Cells in the coupled blocks that were repaired.
    pub cells: usize,
    /// Cells stated with a negative variance, or a zero one and a covariance
    /// to another cell. A PSD matrix has neither, and neither has a
    /// correlation to keep, so each is held at nominal: its variance and its
    /// covariances are sampled as zero.
    pub held_cells: usize,
    /// Whether every repaired block met
    /// [`crate::nearest_correlation::NEAREST_CORRELATION_TOLERANCE`]. A block
    /// that did not is still repaired to a valid correlation matrix, only not
    /// the nearest one.
    pub converged: bool,
}

impl FieldRepair {
    /// The two repairs as one, of the block-diagonal matrix they make.
    fn merge(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (Some(a), Some(b)) => Some(FieldRepair {
                lambda_min: a.lambda_min.min(b.lambda_min),
                largest_correlation_change: a
                    .largest_correlation_change
                    .max(b.largest_correlation_change),
                correlation_frobenius_change: a
                    .correlation_frobenius_change
                    .hypot(b.correlation_frobenius_change),
                cells: a.cells + b.cells,
                held_cells: a.held_cells + b.held_cells,
                converged: a.converged && b.converged,
            }),
            (a, b) => a.or(b),
        }
    }
}

/// `covariance` (row-major `n × n`) repaired to be sampled, keeping every
/// positive variance it states, and what the repair did, or `None` where it
/// is PSD past [`REPAIR_TOLERANCE`].
///
/// A PSD matrix with a zero diagonal has a zero row there, so a cell stated
/// with a negative variance, or a zero one and any covariance to another
/// cell, needs a repair, and an exact one to state: it is held (its row and
/// column zero). The cells with a positive variance are judged and repaired
/// on their correlation matrix by
/// [`crate::nearest_correlation::repaired_with_minimum`], so that round-off
/// is relative to each cell (see [`REPAIR_TOLERANCE`]), and rescaled by
/// their evaluated sigmas. A block of cells the threshold passes is kept bit
/// for bit.
fn repaired_covariance(covariance: &[f64], n: usize) -> Option<(Vec<f64>, FieldRepair)> {
    let c = |i: usize, j: usize| covariance[i * n + j];
    let held_cells = (0..n)
        .filter(|&i| c(i, i) < 0.0 || (c(i, i) == 0.0 && (0..n).any(|j| j != i && c(i, j) != 0.0)))
        .count();
    let positive: Vec<usize> = (0..n).filter(|&i| c(i, i) > 0.0).collect();
    let m = positive.len();
    let sigma: Vec<f64> = positive.iter().map(|&i| c(i, i).sqrt()).collect();
    let mut correlation = vec![0.0; m * m];
    for (a, &i) in positive.iter().enumerate() {
        for (b, &j) in positive.iter().enumerate() {
            correlation[a * m + b] = if a == b {
                1.0
            } else {
                c(i, j) / sigma[a] / sigma[b]
            };
        }
    }
    let (lambda_min, near) = crate::nearest_correlation::repaired_with_minimum(&correlation, m);
    if near.is_none() && held_cells == 0 {
        return None;
    }
    let mut out = vec![0.0; n * n];
    for (a, &i) in positive.iter().enumerate() {
        out[i * n + i] = c(i, i);
        for (b, &j) in positive.iter().enumerate().skip(a + 1) {
            // The upper triangle of the repaired correlation, mirrored, so
            // the result is symmetric to the bit.
            let v = match &near {
                Some((r, _)) => r[a * m + b] * sigma[a] * sigma[b],
                None => c(i, j),
            };
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    let record = match near {
        Some((_, r)) => FieldRepair {
            lambda_min,
            largest_correlation_change: r.max_change,
            correlation_frobenius_change: r.frobenius_change,
            cells: r.parameters,
            held_cells,
            converged: r.converged,
        },
        None => FieldRepair {
            lambda_min,
            largest_correlation_change: 0.0,
            correlation_frobenius_change: 0.0,
            cells: 0,
            held_cells,
            converged: true,
        },
    };
    Some((out, record))
}

/// Keeps the absolute-cell streams clear of every other per-nuclide stream.
const ABSOLUTE_STREAM: u32 = 0xAB50_1C0B;
/// Keeps the short-range streams clear of every other per-nuclide stream.
const SHORT_RANGE_STREAM: u32 = 0x5807_7A9E;

/// One nuclide's covariance, factorized: everything that depends only on the
/// evaluation, so one is shared by every material, spectrum and run that
/// reads the same covariance (see [`factorized`]).
struct Factorized {
    /// Row-major `n × r` factor of the log-space covariance of the relative
    /// cells, `Σ_N = ln(1 + C)` after any repair, with the directions whose
    /// variance is below round-off dropped, so a draw costs `n r`.
    log_factor: Vec<f64>,
    log_rank: usize,
    /// `Σ_N,kk / 2` per relative cell, after repair, so `exp(y - half)` has
    /// mean one.
    half_log_variance: Vec<f64>,
    /// `exp(Σ_N) - 1`: the relative covariance the multipliers actually have.
    relative_sampled: Vec<f64>,
    /// Row-major factor of `relative_sampled`, the linear map first-order
    /// attribution reads. Built on first use, since only attribution needs it.
    relative_factor: std::sync::OnceLock<Vec<f64>>,
    /// Row-major `m × r` factor of the absolute cells' covariance, in barns.
    absolute_factor: Vec<f64>,
    absolute_rank: usize,
    /// The absolute cells' sampled covariance, in barn^2.
    absolute_sampled: Vec<f64>,
    repair: Option<FieldRepair>,
    /// Where the relative cells are not a lognormal's (see [`LognormalLimit`]).
    lognormal_limit: Option<LognormalLimit>,
    /// The inputs, so a cache hit is checked exactly rather than trusted to a
    /// hash.
    relative_input: Vec<f64>,
    absolute_input: Vec<f64>,
    /// Draws already made, keyed by `(nuclide ordinal, seed, replica)`: a draw
    /// depends on nothing else, so every material reading this nuclide in a
    /// run shares it rather than redoing the matrix-vector products.
    draws: std::sync::Mutex<HashMap<(u32, u64, u64), GaussianDraw>>,
}

/// One replica's relative multipliers minus one and absolute shifts.
type GaussianDraw = std::sync::Arc<(Vec<f64>, Vec<f64>)>;

/// How many draws one factorization keeps before it starts over.
const DRAW_CACHE: usize = 1 << 13;

/// How many factorizations the process keeps before it starts over.
const FACTORIZED_CACHE: usize = 256;

/// One nuclide's field as one run reads it: its shared factorization, and the
/// short-range pieces this run's spectra cut.
struct Field {
    core: std::sync::Arc<Factorized>,
    short: Vec<crate::covariance_fold::ShortRange>,
    /// Per short-range block and interval, the pieces drawn per replica: the
    /// union of every spectrum's cuts there, so each spectrum's piece is a
    /// whole number of them.
    short_pieces: Vec<Vec<Vec<f64>>>,
}

impl Field {
    fn n_relative(&self) -> usize {
        self.core.half_log_variance.len()
    }

    fn n_absolute(&self) -> usize {
        (self.core.absolute_sampled.len() as f64).sqrt() as usize
    }

    fn n_short_pieces(&self) -> usize {
        self.short_pieces
            .iter()
            .flatten()
            .map(|p| p.len().saturating_sub(1))
            .sum()
    }
}

/// One spectrum's reading of one nuclide's field.
struct View {
    kinds: Vec<String>,
    /// Full rate per channel.
    rates: Vec<f64>,
    /// Row-major `kinds × relative cells` partial rates.
    relative: Vec<f64>,
    /// Row-major `kinds × absolute cells` partial fluxes times 1e-24.
    absolute: Vec<f64>,
    /// Per channel, `(block, interval, weight per drawn piece)`: the term
    /// weights carried onto the union pieces, summed over the channel's
    /// terms, so a channel's short-range shift is one dot product per
    /// interval.
    short: Vec<Vec<(usize, usize, Vec<f64>)>>,
    /// Per channel `C_ii` as folded, before any repair, or `None` for a
    /// channel the fold did not give a variance.
    evaluated_variance: Vec<f64>,
    /// Per channel, the relative variance of the sampled rate.
    sampled_variance: Vec<f64>,
}

/// The cross-section uncertainty of one schedule: every nuclide's field,
/// factorized once, and how each spectrum reads it.
///
/// A replica draws each nuclide's cross sections once, as a field over its
/// covariance cells, and every spectrum's rates are read off that one draw.
/// So a nuclide irradiated under two spectra moves in both exactly as one
/// perturbed evaluation would move it, and a spectrum added to the schedule
/// does not change what the others draw.
pub struct Sampler {
    fields: BTreeMap<String, Field>,
    views: Vec<BTreeMap<String, View>>,
}

/// One nuclide's principal modes: a basis for the linear part of any
/// response to its draws, largest share of variance first, relative cells
/// and absolute cells apart (they are drawn independently).
///
/// With `p` a draw's relative multipliers minus one (or absolute shifts), `C`
/// the covariance the draws actually have (after the lognormal transform and
/// any repair, not the evaluation's before them) and `W = diag(1 / √C_jj)`,
/// the modes are the eigenpairs `(λ_k, e_k)` of the correlation matrix
/// `W C W`. A draw's coordinate on mode `k` is `ξ_k = e_kᵀ W p`, and the
/// coordinates have mean zero, variance `λ_k` and no correlation, exactly. A
/// response's derivative along mode `k` is `D_k = sᵀ W⁻¹ e_k`, `s` its
/// gradient in `p`; over every mode `Σ_k D_k ξ_k = sᵀ p`. Keeping only the
/// leading modes makes `Σ_k D_k ξ_k` a linear predictor whose variance,
/// `Σ_k λ_k D_k²`, is known exactly; what the dropped modes would have
/// carried only makes the predictor a poorer control variate, never a
/// biased one.
///
/// Correlation rather than covariance, because the covariance's leading
/// modes are its widest cells (a threshold channel at hundreds of percent),
/// not the ones a response feels, and correlation needs no guess at what a
/// response feels: every cell counts alike, and the modes a correlated block
/// collapses to are kept whatever their width.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Modes {
    /// `λ_k` per relative mode.
    pub relative_variance: Vec<f64>,
    /// Row-major `relative cells × relative modes`, `W e_k` in column `k`:
    /// a draw's coordinate is this column dotted with its multipliers minus
    /// one.
    pub relative_projection: Vec<f64>,
    /// Row-major `relative cells × relative modes`, `W⁻¹ e_k` in column `k`:
    /// a derivative along the mode is this column dotted with the gradient.
    pub relative_loading: Vec<f64>,
    /// `λ_k` per absolute mode.
    pub absolute_variance: Vec<f64>,
    pub absolute_projection: Vec<f64>,
    pub absolute_loading: Vec<f64>,
}

impl Modes {
    pub fn n_relative(&self) -> usize {
        self.relative_variance.len()
    }

    pub fn n_absolute(&self) -> usize {
        self.absolute_variance.len()
    }

    /// Modes in all, relative first.
    pub fn len(&self) -> usize {
        self.n_relative() + self.n_absolute()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The leading eigenpairs of the correlation matrix of the `n × n`
/// covariance `matrix`, until they hold `kept` of its trace: `(variances,
/// projection W e, loading W⁻¹ e)`, each row-major `n × k`, with
/// `W = diag(1 / √C_jj)`. A cell of no variance is in no mode.
fn principal(matrix: &[f64], n: usize, kept: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let v = matrix[i * n + i];
            if v > 0.0 {
                1.0 / v.sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let mut correlation = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            correlation[i * n + j] = w[i] * matrix[i * n + j] * w[j];
        }
    }
    let (values, vectors) = leading_eigen(&correlation, n);
    let mut order: Vec<usize> = (0..n).filter(|&j| values[j] > 0.0).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    let total: f64 = order.iter().map(|&j| values[j]).sum();
    let mut chosen = Vec::new();
    let mut held = 0.0;
    for &j in &order {
        if held >= kept * total {
            break;
        }
        held += values[j];
        chosen.push(j);
    }
    let k = chosen.len();
    let (mut projection, mut loading) = (vec![0.0; n * k], vec![0.0; n * k]);
    for i in 0..n {
        if w[i] == 0.0 {
            continue;
        }
        for (c, &j) in chosen.iter().enumerate() {
            projection[i * k + c] = w[i] * vectors[i * n + j];
            loading[i * k + c] = vectors[i * n + j] / w[i];
        }
    }
    (
        chosen.iter().map(|&j| values[j]).collect(),
        projection,
        loading,
    )
}

/// One replica's draw of every nuclide's field.
pub struct Draw {
    nuclides: BTreeMap<String, NuclideDraw>,
}

impl Draw {
    /// `m_k - 1` per relative cell of `nuclide`'s field, in the field's cell
    /// order, or `None` for a nuclide this draw has no field for.
    pub fn relative(&self, nuclide: &str) -> Option<&[f64]> {
        self.nuclides.get(nuclide).map(|d| d.relative.as_slice())
    }

    /// The shift in barns per absolute cell of `nuclide`'s field.
    pub fn absolute(&self, nuclide: &str) -> Option<&[f64]> {
        self.nuclides.get(nuclide).map(|d| d.absolute.as_slice())
    }
}

struct NuclideDraw {
    /// `m_k - 1` per relative cell.
    relative: Vec<f64>,
    /// `a_k` per absolute cell, in barns.
    absolute: Vec<f64>,
    /// Per short-range block and interval, the noise integrated over each
    /// drawn piece, in barn-eV.
    short: Vec<Vec<Vec<f64>>>,
}

/// The factor `L` of a symmetric matrix with its negative eigenvalues set
/// to zero, `L Lᵀ`, and whether that did anything: any eigenvalue below
/// `-1e-12` of the largest. Where it did not, `L Lᵀ` is the matrix itself,
/// exactly, and is not recomputed.
///
/// Only for a matrix that is PSD to round-off, or one whose remaining
/// negative part is reported where it is made: a repair of the evaluation is
/// [`repaired_covariance`]'s, before this.
pub(crate) fn clipped_factor(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, bool) {
    if n == 0 {
        return (Vec::new(), Vec::new(), false);
    }
    let (values, vectors) = eigen(matrix, n);
    let lambda_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let lambda_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let clipped = lambda_min < -1.0e-12 * lambda_max.abs().max(lambda_min.abs());
    // L = V √Λ: column j of V is eigenvector j, so L[i][j] = V[i][j] · √λ_j.
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
    let product = if clipped {
        let mut product = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v: f64 = (0..n).map(|k| l[i * n + k] * l[j * n + k]).sum();
                product[i * n + j] = v;
                product[j * n + i] = v;
            }
        }
        product
    } else {
        matrix.to_vec()
    };
    (l, product, clipped)
}

/// Drop the columns of a row-major `n × n` factor whose variance is below
/// round-off of the total, returning the `n × r` factor and `r`.
///
/// Column `j` of `L = V √Λ` carries variance `λ_j`, so a column below
/// `1e-15` of the trace moves no variance anything downstream can resolve,
/// and dropping it makes every draw cost `n r` instead of `n²`. MF=33 cell
/// covariances are far from full rank: ENDF/B-VIII.1 Fe56 keeps about 400 of
/// 650 directions.
fn truncated(l: Vec<f64>, n: usize) -> (Vec<f64>, usize) {
    if n == 0 {
        return (l, 0);
    }
    let weight: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| l[i * n + j] * l[i * n + j]).sum())
        .collect();
    let total: f64 = weight.iter().sum();
    let kept: Vec<usize> = (0..n).filter(|&j| weight[j] > 1.0e-15 * total).collect();
    let r = kept.len();
    let mut out = vec![0.0; n * r];
    for i in 0..n {
        for (c, &j) in kept.iter().enumerate() {
            out[i * r + c] = l[i * n + j];
        }
    }
    (out, r)
}

impl Factorized {
    /// Factorize one nuclide's cells.
    ///
    /// The relative cells are sampled as a multivariate lognormal matched to
    /// the evaluation's covariance: `m = exp(y - diag(Σ_N)/2)` with
    /// `y ~ N(0, Σ_N)` and `Σ_N = ln(1 + C)` elementwise, which gives every
    /// multiplier mean one and `Cov(m_k, m_l) = C_kl` exactly wherever a
    /// lognormal can carry it (Žerovnik et al., NIM A 727 (2013) 33). Every
    /// multiplier is positive, so every perturbed cross section is.
    ///
    /// Not every covariance is a lognormal's: two fully correlated cells with
    /// different sigmas cannot be, and an anticorrelation with
    /// `1 + C_kl <= 0` cannot be at all. So `Σ_N` can come out indefinite
    /// from a perfectly good evaluation, and where it does past
    /// [`REPAIR_TOLERANCE`] its correlation matrix is replaced by the nearest
    /// correlation matrix, keeping every `Σ_N,kk` and so every cell's sigma.
    /// That is a property of the distribution, not a defect of the data, so
    /// it is not reported as a repair but in [`LognormalLimit`], beside how
    /// far the sampled covariance is from `C`. It shrinks as `σ⁴`, so it only
    /// shows on wide channels, which the report names separately. A repair is
    /// the evaluation's own matrix `C` not being PSD, decided and made on `C`
    /// before the transform ([`repaired_covariance`]).
    fn new(relative: &[f64], absolute: &[f64]) -> Self {
        let n = (relative.len() as f64).sqrt() as usize;
        let (relative_repair, repaired) = match repaired_covariance(relative, n) {
            Some((repaired, record)) => (Some(record), repaired),
            None => (None, relative.to_vec()),
        };
        let (log, substituted) = log_covariance(&repaired, n);
        let (log, log_space_repair) = match repaired_covariance(&log, n) {
            Some((log, record)) => (log, Some(record)),
            None => (log, None),
        };
        let (log_factor, log_sampled, log_clipped) = clipped_factor(&log, n);
        let half_log_variance = (0..n).map(|k| 0.5 * log_sampled[k * n + k]).collect();
        let (log_factor, log_rank) = truncated(log_factor, n);
        // Where the lognormal carries the covariance, what is sampled is the
        // (repaired) evaluation itself, exactly, and is reported as that
        // rather than through a round trip through `ln` and `exp` that costs an
        // ulp. Only where `Σ_N` had to be repaired or clipped, or an entry had
        // no logarithm, is the sampled one different, and then it is recorded
        // how much.
        let (relative_sampled, lognormal_limit): (Vec<f64>, Option<LognormalLimit>) =
            if log_clipped || substituted || log_space_repair.is_some() {
                let sampled: Vec<f64> = log_sampled.iter().map(|v| v.exp_m1()).collect();
                let limit = lognormal_limit(&repaired, &sampled, n).map(|l| LognormalLimit {
                    log_space_repair,
                    ..l
                });
                (sampled, limit)
            } else {
                (repaired, None)
            };

        let m = (absolute.len() as f64).sqrt() as usize;
        let (absolute_repair, absolute_sampled) = match repaired_covariance(absolute, m) {
            Some((repaired, record)) => (Some(record), repaired),
            None => (None, absolute.to_vec()),
        };
        let (absolute_factor, _, _) = clipped_factor(&absolute_sampled, m);
        let (absolute_factor, absolute_rank) = truncated(absolute_factor, m);

        Factorized {
            log_factor,
            log_rank,
            half_log_variance,
            relative_sampled,
            relative_factor: std::sync::OnceLock::new(),
            absolute_factor,
            absolute_rank,
            absolute_sampled,
            repair: FieldRepair::merge(relative_repair, absolute_repair),
            lognormal_limit,
            relative_input: relative.to_vec(),
            absolute_input: absolute.to_vec(),
            draws: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// The relative multipliers minus one and the absolute shifts of one
    /// replica's draw for the nuclide with `ordinal`, made once and shared.
    fn gaussian_draw(&self, ordinal: u32, base_seed: u64, replica: u64) -> GaussianDraw {
        let key = (ordinal, base_seed, replica);
        if let Some(d) = self.draws.lock().expect("draw cache").get(&key) {
            return d.clone();
        }
        let replica_seed = yamc_rng::history_seed(base_seed, replica);
        let n_relative = self.half_log_variance.len();
        let n_absolute = (self.absolute_sampled.len() as f64).sqrt() as usize;
        let gaussian = |stream: u32, factor: &[f64], rank: usize, n: usize| -> Vec<f64> {
            if rank == 0 {
                return vec![0.0; n];
            }
            let seed = yamc_rng::secondary_seed(replica_seed, ordinal ^ stream);
            let mut state = yamc_rng::expand_seed(seed);
            let z = standard_normals(&mut state, rank);
            factor
                .chunks_exact(rank)
                .map(|row| row.iter().zip(&z).map(|(l, z)| l * z).sum())
                .collect()
        };
        let relative = gaussian(0, &self.log_factor, self.log_rank, n_relative)
            .into_iter()
            .zip(&self.half_log_variance)
            .map(|(y, half)| (y - half).exp_m1())
            .collect();
        let absolute = gaussian(
            ABSOLUTE_STREAM,
            &self.absolute_factor,
            self.absolute_rank,
            n_absolute,
        );
        let drawn = std::sync::Arc::new((relative, absolute));
        let mut cache = self.draws.lock().expect("draw cache");
        if cache.len() >= DRAW_CACHE {
            cache.clear();
        }
        cache.insert(key, drawn.clone());
        drawn
    }
}

/// The factorization of `relative` and `absolute`, from the process-wide
/// cache when the same covariance was factorized before.
///
/// The factorization depends on the covariance alone, so every material of a
/// many-material run, every spectrum and every later run reading the same
/// evaluation shares one. Keyed on a hash of the matrices and confirmed
/// against them exactly.
fn factorized(relative: &[f64], absolute: &[f64]) -> std::sync::Arc<Factorized> {
    type Cache = HashMap<u64, Vec<std::sync::Arc<Factorized>>>;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Cache>> = std::sync::OnceLock::new();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in relative.iter().chain([f64::NAN].iter()).chain(absolute) {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let cache = CACHE.get_or_init(Default::default);
    let same = |f: &std::sync::Arc<Factorized>| {
        f.relative_input.len() == relative.len()
            && f.absolute_input.len() == absolute.len()
            && f.relative_input
                .iter()
                .zip(relative)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && f.absolute_input
                .iter()
                .zip(absolute)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    if let Some(f) = cache
        .lock()
        .expect("factorization cache")
        .get(&h)
        .and_then(|bucket| bucket.iter().find(|f| same(f)).cloned())
    {
        return f;
    }
    // Factorized outside the lock, so nuclides factorize in parallel. Two
    // threads racing on one covariance both compute it, and get the same
    // answer either way.
    let built = std::sync::Arc::new(Factorized::new(relative, absolute));
    let mut map = cache.lock().expect("factorization cache");
    if map.values().map(Vec::len).sum::<usize>() >= FACTORIZED_CACHE {
        map.clear();
    }
    let bucket = map.entry(h).or_default();
    if let Some(f) = bucket.iter().find(|f| same(f)) {
        return f.clone();
    }
    bucket.push(built.clone());
    built
}

impl Field {
    fn new(cells: &crate::covariance_fold::CellField) -> Self {
        let core = factorized(&cells.relative, &cells.absolute);
        let short_pieces = cells
            .short
            .iter()
            .enumerate()
            .map(|(b, block)| {
                (0..block.variance.len())
                    .map(|k| {
                        let mut edges: Vec<f64> = cells
                            .projections
                            .iter()
                            .flatten()
                            .flat_map(|p| p.short.iter().flatten())
                            .filter(|t| t.block == b && t.interval == k)
                            .flat_map(|t| t.cuts.iter().copied())
                            .collect();
                        edges.sort_by(f64::total_cmp);
                        edges.dedup();
                        edges
                    })
                    .collect()
            })
            .collect();

        Field {
            core,
            short: cells.short.clone(),
            short_pieces,
        }
    }

    /// One replica's draw of this field.
    fn draw(&self, name: &str, base_seed: u64, replica: u64) -> NuclideDraw {
        let replica_seed = yamc_rng::history_seed(base_seed, replica);
        // Keyed on the NAME, not on a position, so a nuclide's stream does not
        // move when a different nuclide joins or leaves the material.
        let ordinal = name_ordinal(name);
        let gaussian = self.core.gaussian_draw(ordinal, base_seed, replica);
        let (relative, absolute) = (gaussian.0.clone(), gaussian.1.clone());

        // Short-range noise: independent increments over each drawn piece,
        // `ΔB_j ~ N(0, F_k ΔE_k w_j)`, one stream per block and interval keyed
        // on the block's identity in the evaluation.
        let short_seed = yamc_rng::secondary_seed(replica_seed, ordinal ^ SHORT_RANGE_STREAM);
        let short = self
            .short
            .iter()
            .zip(&self.short_pieces)
            .map(|(block, intervals)| {
                intervals
                    .iter()
                    .enumerate()
                    .map(|(k, edges)| {
                        if edges.len() < 2 {
                            return Vec::new();
                        }
                        let (mt, sub, idx) = block.key;
                        let key = name_ordinal(&format!("{mt}/{sub}/{idx}/{k}"));
                        let mut state =
                            yamc_rng::expand_seed(yamc_rng::secondary_seed(short_seed, key));
                        let z = standard_normals(&mut state, edges.len() - 1);
                        let width = block.edges[k + 1] - block.edges[k];
                        edges
                            .windows(2)
                            .zip(z)
                            .map(|(w, z)| {
                                z * (block.variance[k] * width * (w[1] - w[0])).max(0.0).sqrt()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        NuclideDraw {
            relative,
            absolute,
            short,
        }
    }
}

impl View {
    fn new(
        projection: &crate::covariance_fold::Projection,
        field: &Field,
        folded: Option<&RateCovariance>,
    ) -> Self {
        let n = projection.kinds.len();
        let (nr, na) = (field.n_relative(), field.n_absolute());
        // Each term's weight per spectrum piece, carried onto the drawn
        // pieces it is made of and summed over the channel's terms.
        let short: Vec<Vec<(usize, usize, Vec<f64>)>> = projection
            .short
            .iter()
            .map(|terms| {
                let mut out: Vec<(usize, usize, Vec<f64>)> = Vec::new();
                for t in terms {
                    let pieces = &field.short_pieces[t.block][t.interval];
                    let slot = match out
                        .iter()
                        .position(|(b, k, _)| *b == t.block && *k == t.interval)
                    {
                        Some(i) => i,
                        None => {
                            out.push((
                                t.block,
                                t.interval,
                                vec![0.0; pieces.len().saturating_sub(1)],
                            ));
                            out.len() - 1
                        }
                    };
                    for (j, w) in pieces.windows(2).enumerate() {
                        // The term's piece holding drawn piece `j`, if any.
                        if let Some(p) =
                            t.cuts.windows(2).position(|c| c[0] <= w[0] && w[1] <= c[1])
                        {
                            out[slot].2[j] += t.weight[p];
                        }
                    }
                }
                out
            })
            .collect();

        let evaluated_variance = projection
            .kinds
            .iter()
            .map(|kind| {
                folded
                    .and_then(|f| f.kinds.iter().position(|k| k == kind).map(|i| f.get(i, i)))
                    .unwrap_or(0.0)
            })
            .collect();
        let sampled_variance = (0..n)
            .map(|i| {
                let r = projection.rates[i];
                if r == 0.0 {
                    return 0.0;
                }
                let p = &projection.relative[i * nr..(i + 1) * nr];
                let q = &projection.absolute[i * na..(i + 1) * na];
                let quadratic = |x: &[f64], m: &[f64], n: usize| -> f64 {
                    (0..n)
                        .filter(|&a| x[a] != 0.0)
                        .map(|a| x[a] * (0..n).map(|b| m[a * n + b] * x[b]).sum::<f64>())
                        .sum()
                };
                let short_var: f64 = short[i]
                    .iter()
                    .map(|(b, k, w)| {
                        let block = &field.short[*b];
                        let width = block.edges[*k + 1] - block.edges[*k];
                        let pieces = &field.short_pieces[*b][*k];
                        w.iter()
                            .zip(pieces.windows(2))
                            .map(|(w, e)| block.variance[*k] * width * (e[1] - e[0]) * w * w)
                            .sum::<f64>()
                    })
                    .sum();
                (quadratic(p, &field.core.relative_sampled, nr)
                    + quadratic(q, &field.core.absolute_sampled, na)
                    + short_var)
                    / (r * r)
            })
            .collect();
        View {
            kinds: projection.kinds.clone(),
            rates: projection.rates.clone(),
            relative: projection.relative.clone(),
            absolute: projection.absolute.clone(),
            short,
            evaluated_variance,
            sampled_variance,
        }
    }

    /// Channel `i`'s rate relative to nominal under `draw`, before flooring.
    fn ratio(&self, i: usize, draw: &NuclideDraw) -> f64 {
        let r = self.rates[i];
        if r == 0.0 {
            return 1.0;
        }
        let (nr, na) = (draw.relative.len(), draw.absolute.len());
        let mut shift: f64 = self.relative[i * nr..(i + 1) * nr]
            .iter()
            .zip(&draw.relative)
            .map(|(p, d)| p * d)
            .sum();
        shift += self.absolute[i * na..(i + 1) * na]
            .iter()
            .zip(&draw.absolute)
            .map(|(q, a)| q * a)
            .sum::<f64>();
        for (b, k, w) in &self.short[i] {
            shift += w
                .iter()
                .zip(&draw.short[*b][*k])
                .map(|(w, db)| w * db)
                .sum::<f64>();
        }
        1.0 + shift / r
    }
}

impl Sampler {
    /// Factorize every nuclide's field and build each spectrum's reading of
    /// it.
    ///
    /// `fields` from [`cell_fields`](crate::covariance_fold::cell_fields), and
    /// `folds[a]` spectrum `a`'s [`fold_rate_covariance`](crate::covariance_fold::fold_rate_covariance),
    /// which supplies each channel's evaluated variance for the report.
    pub fn new(
        fields: &BTreeMap<String, crate::covariance_fold::CellField>,
        folds: &[BTreeMap<String, RateCovariance>],
    ) -> Self {
        let entries: Vec<(&String, &crate::covariance_fold::CellField)> = fields.iter().collect();
        let one = |&(name, cells): &(&String, &crate::covariance_fold::CellField)| -> (String, Field, Vec<Option<View>>) {
            let field = Field::new(cells);
            let views = cells
                .projections
                .iter()
                .enumerate()
                .map(|(a, p)| {
                    p.as_ref()
                        .map(|p| View::new(p, &field, folds.get(a).and_then(|f| f.get(name))))
                })
                .collect();
            (name.clone(), field, views)
        };
        let built: Vec<(String, Field, Vec<Option<View>>)> = {
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
        let n_spectra = folds.len().max(
            fields
                .values()
                .map(|f| f.projections.len())
                .max()
                .unwrap_or(0),
        );
        let mut sampler = Sampler {
            fields: BTreeMap::new(),
            views: (0..n_spectra).map(|_| BTreeMap::new()).collect(),
        };
        for (name, field, views) in built {
            for (a, view) in views.into_iter().enumerate() {
                if let Some(view) = view {
                    sampler.views[a].insert(name.clone(), view);
                }
            }
            sampler.fields.insert(name, field);
        }
        sampler
    }

    /// A sampler over collapsed rates directly, for tests: each channel of
    /// each nuclide one relative cell with a partial rate equal to its rate,
    /// read the same way by each of `spectra` spectra. A rate is then exactly
    /// its cell's multiplier, so this is the multivariate lognormal with
    /// covariance `C`.
    #[cfg(test)]
    pub(crate) fn from_rate_covariance(
        covariance: &BTreeMap<String, RateCovariance>,
        spectra: usize,
    ) -> Self {
        use crate::covariance_fold::{Cell, CellField, Projection};
        let fields = covariance
            .iter()
            .filter(|(_, c)| c.n() > 0)
            .map(|(name, c)| {
                let n = c.n();
                let mut identity = vec![0.0; n * n];
                for i in 0..n {
                    identity[i * n + i] = 1.0;
                }
                (
                    name.clone(),
                    CellField {
                        relative_cells: (0..n)
                            .map(|i| Cell {
                                mt: i as i32,
                                lo: 0.0,
                                hi: 1.0,
                            })
                            .collect(),
                        relative: c.relative.clone(),
                        absolute_cells: Vec::new(),
                        absolute: Vec::new(),
                        short: Vec::new(),
                        projections: vec![
                            Some(Projection {
                                kinds: c.kinds.clone(),
                                rates: vec![1.0; n],
                                relative: identity,
                                absolute: Vec::new(),
                                short: vec![Vec::new(); n],
                            });
                            spectra
                        ],
                    },
                )
            })
            .collect();
        Self::new(&fields, &vec![covariance.clone(); spectra])
    }

    /// Every matrix this sampler had to repair, tagged with `spectrum`.
    ///
    /// A nuclide's field is factorized once for every spectrum, so a repair
    /// of it is listed under each spectrum that reads it, with that
    /// spectrum's channels and the same [`FieldRepair`].
    pub fn repairs(&self, spectrum: usize) -> Vec<Repair> {
        let Some(views) = self.views.get(spectrum) else {
            return Vec::new();
        };
        views
            .iter()
            .filter_map(|(name, v)| {
                let r = self.fields[name].core.repair?;
                Some(Repair {
                    nuclide: name.clone(),
                    spectrum,
                    field: r,
                    channels: v
                        .kinds
                        .iter()
                        .zip(v.evaluated_variance.iter().zip(&v.sampled_variance))
                        .map(|(kind, (e, s))| ChannelSigma {
                            kind: kind.clone(),
                            evaluated_variance: *e,
                            sampled: s.max(0.0).sqrt(),
                        })
                        .collect(),
                })
            })
            .collect()
    }

    /// Every nuclide whose evaluated covariance had to be repaired to be
    /// sampled, with what the repair did, independent of any spectrum.
    pub fn field_repairs(&self) -> BTreeMap<String, FieldRepair> {
        self.fields
            .iter()
            .filter_map(|(name, f)| Some((name.clone(), f.core.repair?)))
            .collect()
    }

    /// Every nuclide whose relative cells are not a lognormal's, with how far
    /// the sampled covariance is from the stated one, independent of any
    /// spectrum.
    pub fn lognormal_limits(&self) -> BTreeMap<String, LognormalLimit> {
        self.fields
            .iter()
            .filter_map(|(name, f)| Some((name.clone(), f.core.lognormal_limit?)))
            .collect()
    }

    /// Every channel `spectrum` reads, as `(nuclide, kind, evaluated,
    /// sampled)` relative sigmas, the evaluated one zero where the stated
    /// variance is negative (it states no spread a weighted mean could use).
    ///
    /// The sampled sigma is the sampled rate's own, read off the factorized
    /// field: where no repair was needed it differs from the evaluated one
    /// by the decomposition's round-off, and that is reported as it is.
    pub fn channel_sigmas(&self, spectrum: usize) -> impl Iterator<Item = (&str, &str, f64, f64)> {
        self.views
            .get(spectrum)
            .into_iter()
            .flat_map(|views| views.iter())
            .flat_map(|(name, v)| {
                v.kinds.iter().enumerate().map(move |(i, kind)| {
                    (
                        name.as_str(),
                        kind.as_str(),
                        v.evaluated_variance[i].max(0.0).sqrt(),
                        v.sampled_variance[i].max(0.0).sqrt(),
                    )
                })
            })
    }

    /// Whether anything was factorized at all.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Under `spectrum`, the nuclides it reads, with each one's reaction
    /// kinds, the row-major linear map `J` from the nuclide's independent
    /// deviates to its channels' relative rate shifts (`J Jᵀ` the sampled
    /// relative covariance of those rates), and its column count, for
    /// first-order attribution.
    ///
    /// The columns are the nuclide's field, the same for every spectrum, so
    /// contributions over spectra add column by column.
    pub(crate) fn factors(&self, spectrum: usize) -> Vec<(&String, &[String], Vec<f64>, usize)> {
        let Some(views) = self.views.get(spectrum) else {
            return Vec::new();
        };
        views
            .iter()
            .map(|(name, v)| {
                let f = &self.fields[name];
                let (nr, na, ns) = (f.n_relative(), f.n_absolute(), f.n_short_pieces());
                let cols = nr + na + ns;
                let n = v.kinds.len();
                let mut j = vec![0.0; n * cols];
                // Where each block and interval's pieces start among the
                // short-range columns.
                let mut short_offset: Vec<Vec<usize>> = Vec::new();
                let mut at = nr + na;
                for intervals in &f.short_pieces {
                    let mut row = Vec::new();
                    for edges in intervals {
                        row.push(at);
                        at += edges.len().saturating_sub(1);
                    }
                    short_offset.push(row);
                }
                for i in 0..n {
                    let r = v.rates[i];
                    if r == 0.0 {
                        continue;
                    }
                    let out = &mut j[i * cols..(i + 1) * cols];
                    let p = &v.relative[i * nr..(i + 1) * nr];
                    let relative_factor = f
                        .core
                        .relative_factor
                        .get_or_init(|| clipped_factor(&f.core.relative_sampled, nr).0);
                    for (c, o) in out[..nr].iter_mut().enumerate() {
                        *o = (0..nr)
                            .map(|k| p[k] * relative_factor[k * nr + c])
                            .sum::<f64>()
                            / r;
                    }
                    let q = &v.absolute[i * na..(i + 1) * na];
                    for (c, o) in out[nr..nr + na].iter_mut().enumerate() {
                        *o = (0..na)
                            .map(|k| q[k] * f.core.absolute_factor[k * na + c])
                            .sum::<f64>()
                            / r;
                    }
                    for (b, k, w) in &v.short[i] {
                        let block = &f.short[*b];
                        let width = block.edges[*k + 1] - block.edges[*k];
                        let pieces = &f.short_pieces[*b][*k];
                        for (p, (w, e)) in w.iter().zip(pieces.windows(2)).enumerate() {
                            out[short_offset[*b][*k] + p] += w
                                * (block.variance[*k] * width * (e[1] - e[0])).max(0.0).sqrt()
                                / r;
                        }
                    }
                }
                (name, v.kinds.as_slice(), j, cols)
            })
            .collect()
    }

    /// The covariance `nuclide`'s draws actually have, row-major: of its
    /// relative cells' multipliers (`exp(Σ_N) - 1`, after the lognormal
    /// transform and any repair) and of its absolute cells' shifts, in barn².
    /// `None` for a nuclide without a field.
    pub fn sampled_covariance(&self, nuclide: &str) -> Option<(&[f64], &[f64])> {
        let f = self.fields.get(nuclide)?;
        Some((&f.core.relative_sampled, &f.core.absolute_sampled))
    }

    /// `nuclide`'s principal modes holding `kept` (in `(0, 1]`) of the
    /// correlation of its relative cells and of its absolute cells, or `None`
    /// for a nuclide without a field. See [`Modes`].
    pub fn modes(&self, nuclide: &str, kept: f64) -> Option<Modes> {
        let f = self.fields.get(nuclide)?;
        let (relative_variance, relative_projection, relative_loading) =
            principal(&f.core.relative_sampled, f.n_relative(), kept);
        let (absolute_variance, absolute_projection, absolute_loading) =
            principal(&f.core.absolute_sampled, f.n_absolute(), kept);
        Some(Modes {
            relative_variance,
            relative_projection,
            relative_loading,
            absolute_variance,
            absolute_projection,
            absolute_loading,
        })
    }

    /// One replica's draw of every nuclide's field.
    pub fn draw(&self, base_seed: u64, replica: u64) -> Draw {
        Draw {
            nuclides: self
                .fields
                .iter()
                .map(|(name, f)| (name.clone(), f.draw(name, base_seed, replica)))
                .collect(),
        }
    }

    /// The relative cell multipliers minus one that one replica draws, per
    /// nuclide, for a test to look at the draw itself.
    pub fn deviates(&self, base_seed: u64, replica: u64) -> BTreeMap<String, Vec<f64>> {
        self.draw(base_seed, replica)
            .nuclides
            .into_iter()
            .map(|(name, d)| (name, d.relative))
            .collect()
    }

    /// Apply one replica's draw to `spectrum`'s unit-flux rates, and count the
    /// rates it drew and the ones it floored at zero.
    ///
    /// Rates for nuclides or kinds with no covariance are passed through
    /// unchanged. That is deliberate and is what the coverage report is for:
    /// an unperturbed rate contributes no uncertainty, and the reason has to
    /// be visible rather than inferred from a suspiciously small sigma.
    ///
    /// A rate of a channel whose terms all have positive coefficients reads
    /// only positive cross sections off relative cells and cannot go
    /// negative. One that subtracts reactions (an NC derivation), or reads an
    /// absolute or short-range shift, can, and is floored at zero and counted.
    pub fn perturb_with(
        &self,
        draw: &Draw,
        spectrum: usize,
        rates: &ReactionRates,
    ) -> (ReactionRates, usize, usize) {
        let mut out = rates.clone();
        let (mut sampled, mut floored) = (0, 0);
        let Some(views) = self.views.get(spectrum) else {
            return (out, 0, 0);
        };
        for (name, view) in views {
            let (Some(d), Some(nuclide_rates)) = (draw.nuclides.get(name), out.get_mut(name))
            else {
                continue;
            };
            for (i, kind) in view.kinds.iter().enumerate() {
                let Some(rate) = nuclide_rates.get_mut(kind) else {
                    continue;
                };
                sampled += 1;
                let ratio = view.ratio(i, d);
                if ratio < 0.0 {
                    floored += 1;
                }
                *rate *= ratio.max(0.0);
            }
        }
        (out, sampled, floored)
    }

    /// [`Sampler::perturb_with`] on a fresh draw, without the floor count.
    pub fn perturb(
        &self,
        spectrum: usize,
        rates: &ReactionRates,
        base_seed: u64,
        replica: u64,
    ) -> (ReactionRates, usize) {
        let (out, n, _) = self.perturb_with(&self.draw(base_seed, replica), spectrum, rates);
        (out, n)
    }
}

/// The repairs and the widest sigmas over every spectrum of a run, for the
/// nuclides the material can populate.
///
/// Every sampled sigma is read off the factorization that is sampled, so it
/// is what was drawn, to that decomposition's round-off:
/// nothing here is a modelling choice, it says where one was made. The fold covers every chain nuclide with data, and from almost
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
    /// spectrum, with a channel a draw can move there: a positive rate on a
    /// spectrum the schedule irradiates with. Their records are left out, but
    /// the bound holds at nominal rates only and a replica's draw can
    /// populate them, so these count as a gap (see `Info::has_gaps`).
    pub repaired_outside_bound: BTreeSet<String>,
    /// The nuclides of `repairs` with at least one repaired channel a draw
    /// can move: a positive rate on a spectrum the schedule irradiates with.
    /// A repair only on an unirradiated spectrum, or only on channels with no
    /// rate, moves nothing the ensemble sees, so it is recorded but is not
    /// a gap.
    pub repaired: BTreeSet<String>,
    /// The largest `|`[`ChannelSigma::change`]`|` over the repaired channels
    /// that can move the result: a populated nuclide with a positive rate on
    /// a spectrum the schedule irradiates with. Zero with no such repair, and
    /// infinite when the repair gave a spread to a channel whose stated
    /// variance is zero or negative.
    pub worst_sigma_change: f64,
    /// `sum_c w_c |sampled_c / evaluated_c - 1|` and `sum_c w_c` over every
    /// sampled channel `c` of every spectrum with a positive evaluated sigma,
    /// `w_c` the channel's unit-flux rate times the spectrum's fluence in the
    /// schedule times the parent's initial density: the reactions the
    /// schedule puts through the channel, at the starting composition. The
    /// weight is on each channel's own change, not on its sigma, so a wide
    /// channel with a small rate cannot drown out a repair on the channels
    /// that carry the reactions, and the change is absolute, since a repair
    /// can move a channel either way and two channels moved opposite ways do
    /// not cancel. Kept as sums because sums merge across spectra.
    weighted_change: f64,
    weight: f64,
    /// Whether a weighted channel with no evaluated sigma was sampled with a
    /// spread, which makes the weighted change infinite.
    weighted_spread_from_nothing: bool,
    /// Sampled channels of populated nuclides with a positive rate on a
    /// spectrum the schedule irradiates with, whose folded relative sigma
    /// `sqrt(C_ii)`, as evaluated and before any repair, is at least one,
    /// keyed by (nuclide, kind), the largest over the spectra. Past this
    /// width the answer depends on the distribution chosen to carry the two
    /// moments, not only on the moments the evaluation states. What a repair
    /// did to a channel is in `repairs`.
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
                // Named only when a draw can reach it: a multiplier on a zero
                // rate, or on an unirradiated spectrum, moves nothing in any
                // replica, bound or not.
                if repair
                    .channels
                    .iter()
                    .any(|c| drawn(&repair.nuclide, &c.kind))
                {
                    self.repaired_outside_bound.insert(repair.nuclide);
                }
                continue;
            }
            for c in repair
                .channels
                .iter()
                .filter(|c| moves(&repair.nuclide, &c.kind))
            {
                self.repaired.insert(repair.nuclide.clone());
                self.worst_sigma_change = self.worst_sigma_change.max(c.change().abs());
            }
            self.repairs.push(repair);
        }
        for (nuclide, kind, evaluated, sampled) in sampler.channel_sigmas(spectrum) {
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
                    self.weighted_change += w * (sampled / evaluated - 1.0).abs();
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

    /// The weighted mean of `|sampled / evaluated - 1|` over every sampled
    /// channel of a populated nuclide, each weighted by the reactions the
    /// schedule puts through it at the initial composition (see
    /// `weighted_change`).
    ///
    /// The headline beside [`SigmaReport::worst_sigma_change`]: a repair on
    /// a channel that carries no reactions costs nothing here, and one on the
    /// dominant channel costs its full share, however wide the channels
    /// beside it are. It covers the reactions on the initial composition
    /// only: a nuclide the material starts without has no initial density and
    /// so no weight, however much of the inventory passes through it, and its
    /// repairs are in `repairs` and `worst_sigma_change` instead. Infinite
    /// when a weighted channel with no evaluated sigma was sampled with a
    /// spread, and `None` when no weighted channel has an evaluated sigma.
    pub fn rate_weighted_sigma_change(&self) -> Option<f64> {
        if self.weighted_spread_from_nothing {
            Some(f64::INFINITY)
        } else if self.weight > 0.0 {
            Some(self.weighted_change / self.weight)
        } else {
            None
        }
    }
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

/// Eigen-decompose a symmetric matrix, block by block.
///
/// Cells no covariance couples are independent, so the matrix is split into
/// its connected blocks and each is decomposed on its own: exact, and on a
/// nuclide's field, where reactions are coupled only through the evaluation's
/// cross blocks, a large saving. A block that is small, or whose positive
/// variances span more than [`DYNAMIC_RANGE`], goes to [`jacobi_eigen`], which
/// keeps a small eigenvalue beside a huge one to high relative accuracy
/// (TENDL-2017 has channels at a variance near 3e17 beside ordinary ones);
/// the rest to [`symmetric_eigen`].
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j`
/// of the row-major `n × n` result, blocks in order of their first index, so
/// the result is a pure function of the input.
pub(crate) fn eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    blockwise(matrix, n, true)
}

/// [`eigen`]'s eigenvalues alone, which skips accumulating the
/// transformations and so costs a fraction as much.
pub(crate) fn eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    blockwise(matrix, n, false).0
}

/// [`eigen`], with or without the eigenvectors (empty when not asked for).
fn blockwise(matrix: &[f64], n: usize, want_vectors: bool) -> (Vec<f64>, Vec<f64>) {
    blocks_eigen(matrix, n, want_vectors, true)
}

/// [`eigen`] for a caller that wants only the leading eigenpairs: every
/// block past [`SMALL_BLOCK`] goes to QL whatever its dynamic range, since
/// QL's error, relative to the largest eigenvalue, is what the leading ones
/// need. Far faster on a large field (O16's 2839 cells) than Jacobi.
fn leading_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    blocks_eigen(matrix, n, true, false)
}

/// The block decomposition behind [`eigen`] and [`leading_eigen`]; with
/// `small_accurately`, a block of wide dynamic range goes to Jacobi.
fn blocks_eigen(
    matrix: &[f64],
    n: usize,
    want_vectors: bool,
    small_accurately: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::with_capacity(n);
    let mut vectors = if want_vectors {
        vec![0.0; n * n]
    } else {
        Vec::new()
    };
    for members in &coupled_blocks(matrix, n) {
        let m = members.len();
        let mut sub = vec![0.0; m * m];
        for (a, &i) in members.iter().enumerate() {
            for (b, &j) in members.iter().enumerate() {
                sub[a * m + b] = matrix[i * n + j];
            }
        }
        let positive = members
            .iter()
            .map(|&i| matrix[i * n + i])
            .filter(|v| *v > 0.0);
        let (lo, hi) = positive.fold((f64::INFINITY, 0.0_f64), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
        let (sub_values, sub_vectors) =
            if m <= SMALL_BLOCK || (small_accurately && lo > 0.0 && hi / lo > DYNAMIC_RANGE) {
                jacobi_eigen(&sub, m)
            } else {
                tridiagonal_ql(&sub, m, want_vectors)
            };
        for c in 0..m {
            let col = values.len();
            values.push(sub_values[c]);
            if want_vectors {
                for (a, &i) in members.iter().enumerate() {
                    vectors[i * n + col] = sub_vectors[a * m + c];
                }
            }
        }
    }
    (values, vectors)
}

/// The connected blocks of a symmetric `n × n` matrix: indices joined by a
/// nonzero off-diagonal, each block in ascending order and the blocks in
/// order of their first index.
pub(crate) fn coupled_blocks(matrix: &[f64], n: usize) -> Vec<Vec<usize>> {
    // Union-find over the nonzero couplings.
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if matrix[i * n + j] != 0.0 || matrix[j * n + i] != 0.0 {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let r = root(&mut parent, i);
        blocks.entry(r).or_default().push(i);
    }
    blocks.into_values().collect()
}

/// Blocks at most this size go to [`jacobi_eigen`]: there its cost is nothing
/// and its accuracy is the best available.
const SMALL_BLOCK: usize = 24;

/// A block whose largest positive variance exceeds its smallest by more than
/// this goes to [`jacobi_eigen`], since QL's error is relative to the largest
/// eigenvalue.
const DYNAMIC_RANGE: f64 = 1.0e8;

/// Eigen-decompose a small symmetric matrix by cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j` of
/// the row-major `n × n` result.
///
/// The sweep order is fixed and the iteration count is bounded, so this is a
/// pure function of its input: two runs on the same matrix give the same last
/// bit, which is what the reproducibility contract needs. Convergence is
/// quadratic and the blocks [`eigen`] hands it are small or nearly diagonal,
/// so the bound is never the thing that stops it in practice. Unlike QL it
/// keeps a small eigenvalue beside a huge one to high relative accuracy,
/// which is why [`eigen`] uses it on blocks whose variances span many orders.
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

/// Eigen-decompose a symmetric matrix: Householder reduction to tridiagonal
/// form, then the implicit QL algorithm with shifts (EISPACK `tred2` and
/// `tql2`, in the form JAMA gives them).
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j` of
/// the row-major `n × n` result.
///
/// Pure arithmetic in a fixed order with no data-dependent parallelism, so it
/// is a pure function of its input: two runs on the same matrix give the same
/// last bit, on every platform, which is what the reproducibility contract
/// needs. It is O(n³) once, where cyclic Jacobi is O(n³) per sweep, and that
/// matters here: a nuclide's field has a cell per covariance interval of
/// every reaction it reaches, hundreds on an evaluation like ENDF/B-VIII.1
/// Fe56.
// Index loops kept as EISPACK writes them, so the port can be checked line
// by line against the reference.
#[cfg(test)]
fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    tridiagonal_ql(matrix, n, true)
}

/// [`symmetric_eigen`], optionally without the eigenvectors: the reduction
/// still uses the matrix as its workspace, but the transformations are not
/// accumulated and the QL rotations are not applied to them.
#[allow(clippy::needless_range_loop)]
fn tridiagonal_ql(matrix: &[f64], n: usize, want_vectors: bool) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // Symmetrize on the way in. The fold fills both halves with the same
    // value, so this is a no-op on real input; it costs nothing and means a
    // caller cannot hand in something the reduction would silently misread.
    let mut v = matrix.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let m = 0.5 * (v[i * n + j] + v[j * n + i]);
            v[i * n + j] = m;
            v[j * n + i] = m;
        }
    }
    // The reduction runs on the transpose (the input is symmetric, so that
    // is the same matrix): its inner loops walk columns of `V`, which are
    // then contiguous rows, and the accumulated `V` comes out transposed,
    // which is the layout the QL rotations below want.
    let at = |i: usize, j: usize| j * n + i;
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];

    // tred2: Householder reduction to tridiagonal form, accumulating the
    // transformations in `v`.
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for x in e.iter_mut().take(i) {
                *x = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[at(j, i)] = f;
                g = e[j] + v[at(j, j)] * f;
                for k in (j + 1)..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    if !want_vectors {
        // The tridiagonal's diagonal is on the diagonal of the workspace.
        for j in 0..n {
            d[j] = v[at(j, j)];
        }
    }
    for i in 0..(n - 1) {
        if !want_vectors {
            break;
        }
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    if want_vectors {
        for j in 0..n {
            d[j] = v[at(n - 1, j)];
            v[at(n - 1, j)] = 0.0;
        }
        v[at(n - 1, n - 1)] = 1.0;
    }
    e[0] = 0.0;

    // tql2: the implicit QL algorithm on the tridiagonal matrix. The
    // rotations combine two eigenvector COLUMNS, which in the transposed `V`
    // are two contiguous rows; it is transposed back after.
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1: f64 = 0.0;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        // `e[n - 1]` is zero, so `m` stops at `n - 1` at the latest.
        let m = m.min(n - 1);
        if m > l {
            // Convergence is cubic; the cap only keeps a pathological input
            // from spinning.
            for _ in 0..64 {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in d.iter_mut().skip(l + 2) {
                    *x -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    if want_vectors {
                        let (lo, hi) = v.split_at_mut((i + 1) * n);
                        let (col_i, col_next) = (&mut lo[i * n..], &mut hi[..n]);
                        for (a, b) in col_i.iter_mut().zip(col_next.iter_mut()) {
                            let h = *b;
                            *b = s * *a + c * h;
                            *a = c * *a - s * h;
                        }
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    if !want_vectors {
        return (d, Vec::new());
    }
    for i in 0..n {
        for j in (i + 1)..n {
            v.swap(at(i, j), at(j, i));
        }
    }
    (d, v)
}

/// A positive multiplier with mean 1 and variance `sigma^2`, from a standard
/// normal `z`, for a caller that draws one independent quantity at a time.
///
/// The half-life and decay-energy sources are that caller. Their evaluations
/// state an expected value and a standard deviation and nothing else (ENDF-102
/// section 29.1), and each nuclide is drawn on its own, so there is no
/// correlation for the transform to distort and this reproduces exactly what
/// the evaluation states. `sigma` must be positive and finite.
pub(crate) fn lognormal_multiplier(z: f64, sigma: f64) -> f64 {
    debug_assert!(
        sigma.is_finite() && sigma > 0.0,
        "lognormal sigma {sigma} is not positive and finite"
    );
    let s_squared = (1.0 + sigma * sigma).ln();
    let s = s_squared.sqrt();
    (s * z - 0.5 * s_squared).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lognormal_limit_is_none_where_the_covariance_is_carried() {
        let c = [0.01, 0.002, 0.002, 0.04];
        assert_eq!(lognormal_limit(&c, &c, 2), None);
    }

    #[test]
    fn a_lognormal_limit_counts_the_cells_and_the_largest_changes() {
        let stated = [0.04, 0.02, 0.02, 0.09];
        // Second sigma 0.3 -> 0.33, correlation 0.02/(0.2*0.3) -> 0.01/(0.06).
        let sampled = [0.04, 0.01, 0.01, 0.1089];
        let l = lognormal_limit(&stated, &sampled, 2).unwrap();
        assert_eq!(l.cells, 2);
        assert!((l.largest_sigma_change - 0.1).abs() < 1e-12);
        assert!((l.largest_correlation_change - 0.01 / 0.06).abs() < 1e-12);
    }

    #[test]
    fn an_anticorrelation_past_minus_one_is_a_lognormal_limit() {
        // 1 + C_01 <= 0: no lognormal carries it, even though C is PSD.
        let n = 2;
        let c = [4.0, -1.5, -1.5, 4.0];
        let (_, substituted) = log_covariance(&c, n);
        assert!(substituted);
        let f = Factorized::new(&c, &[]);
        assert!(f.lognormal_limit.is_some());
    }

    /// A populated set wide enough for every nuclide these tests name.
    fn everyone() -> HashSet<String> {
        ["X", "Y"].iter().map(|s| s.to_string()).collect()
    }

    /// A rate-level sampler under one spectrum.
    fn rate_sampler(covariance: &BTreeMap<String, RateCovariance>) -> Sampler {
        Sampler::from_rate_covariance(covariance, 1)
    }

    fn cov(kinds: &[&str], relative: Vec<f64>) -> RateCovariance {
        RateCovariance {
            kinds: kinds.iter().map(|s| s.to_string()).collect(),
            relative,
        }
    }

    #[test]
    fn the_eigensolver_recovers_a_known_spectrum() {
        // diag(4, 1) rotated by 45 degrees: [[2.5, 1.5], [1.5, 2.5]].
        let (values, _) = symmetric_eigen(&[2.5, 1.5, 1.5, 2.5], 2);
        let mut v = values;
        v.sort_by(f64::total_cmp);
        assert!((v[0] - 1.0).abs() < 1e-12, "{v:?}");
        assert!((v[1] - 4.0).abs() < 1e-12, "{v:?}");
    }

    /// `V Λ Vᵀ` reproduces a dense indefinite matrix and `V` is orthonormal,
    /// at a size where the tridiagonal path does real work.
    #[test]
    fn the_eigensolver_reconstructs_a_dense_matrix() {
        let n = 60;
        let mut state = yamc_rng::expand_seed(42);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let x = yamc_rng::next_xi(&mut state) - 0.5;
                a[i * n + j] = x;
                a[j * n + i] = x;
            }
        }
        let (values, v) = symmetric_eigen(&a, n);
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..n)
                    .map(|k| v[i * n + k] * values[k] * v[j * n + k])
                    .sum();
                assert!((rebuilt - a[i * n + j]).abs() < 1e-12, "({i},{j})");
                let dot: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "VᵀV ({i},{j}) = {dot}");
            }
        }
    }

    /// The eigenvalues-only path gives the same eigenvalues as the full one.
    #[test]
    fn eigenvalues_alone_match_the_full_decomposition() {
        let n = 50;
        let mut state = yamc_rng::expand_seed(7);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let x = yamc_rng::next_xi(&mut state) - 0.5;
                a[i * n + j] = x;
                a[j * n + i] = x;
            }
        }
        let mut full = symmetric_eigen(&a, n).0;
        let mut alone = tridiagonal_ql(&a, n, false).0;
        full.sort_by(f64::total_cmp);
        alone.sort_by(f64::total_cmp);
        for (x, y) in full.iter().zip(&alone) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    /// `L Lᵀ` must reproduce the matrix it was factorized from.
    #[test]
    fn the_factor_reproduces_a_positive_definite_matrix() {
        let c = cov(&["(n,gamma)", "(n,p)"], vec![0.04, 0.012, 0.012, 0.09]);
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c.clone())]));
        assert!(s.repairs(0).is_empty());

        let factors = s.factors(0);
        let (_, _, l, cols) = &factors[0];
        for i in 0..2 {
            for j in 0..2 {
                let got: f64 = (0..*cols).map(|k| l[i * cols + k] * l[j * cols + k]).sum();
                assert!((got - c.get(i, j)).abs() < 1e-12, "({i},{j}) {got}");
            }
        }
    }

    /// One nuclide's name, cell covariance and channels, each a kind and its
    /// partial rates on the cells.
    type FoldedNuclide<'a> = (&'a str, Vec<f64>, &'a [(&'a str, &'a [f64])]);

    /// A sampler over one spectrum (or `spectra` alike) of nuclides each
    /// with relative cells of covariance `cells` (row-major), read by
    /// channels that fold the cells with the given partial rates, a channel's
    /// rate their sum. Each channel's evaluated variance is the fold of
    /// `cells` as stated, by the same arithmetic the sampler reads its
    /// sampled one with, so where nothing is repaired the two agree to the
    /// bit.
    fn folded_sampler(nuclides: &[FoldedNuclide], spectra: usize) -> Sampler {
        use crate::covariance_fold::{Cell, CellField, Projection};
        let mut fields = BTreeMap::new();
        let mut fold = BTreeMap::new();
        for (name, cells, channels) in nuclides {
            let n = (cells.len() as f64).sqrt() as usize;
            let kinds: Vec<String> = channels.iter().map(|(k, _)| k.to_string()).collect();
            let rates: Vec<f64> = channels.iter().map(|(_, p)| p.iter().sum()).collect();
            let partial: Vec<f64> = channels
                .iter()
                .flat_map(|(_, p)| p.iter().copied())
                .collect();
            let k = channels.len();
            let mut folded = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    let (p, q) = (channels[i].1, channels[j].1);
                    let v: f64 = (0..n)
                        .filter(|&a| p[a] != 0.0)
                        .map(|a| p[a] * (0..n).map(|b| cells[a * n + b] * q[b]).sum::<f64>())
                        .sum();
                    folded[i * k + j] = v / (rates[i] * rates[j]);
                }
            }
            fold.insert(
                name.to_string(),
                RateCovariance {
                    kinds: kinds.clone(),
                    relative: folded,
                },
            );
            fields.insert(
                name.to_string(),
                CellField {
                    relative_cells: (0..n)
                        .map(|i| Cell {
                            mt: i as i32,
                            lo: 0.0,
                            hi: 1.0,
                        })
                        .collect(),
                    relative: cells.clone(),
                    absolute_cells: Vec::new(),
                    absolute: Vec::new(),
                    short: Vec::new(),
                    projections: vec![
                        Some(Projection {
                            kinds,
                            rates,
                            relative: partial,
                            absolute: Vec::new(),
                            short: vec![Vec::new(); k],
                        });
                        spectra
                    ],
                },
            );
        }
        Sampler::new(&fields, &vec![fold; spectra])
    }

    /// Two cells at a sigma of 0.1 with a stated correlation of 1.5, which
    /// the repair takes to 1, read by a channel of one cell and a channel of
    /// both. The one-cell channel keeps its sigma; the two-cell one reads the
    /// correlation, so it is sampled at 0.1 against the `sqrt(0.0125)` its
    /// fold states.
    fn over_correlated() -> (Vec<f64>, &'static [(&'static str, &'static [f64])]) {
        (
            vec![0.01, 0.015, 0.015, 0.01],
            &[("a", &[1.0, 0.0]), ("sum", &[1.0, 1.0])],
        )
    }

    /// `over_correlated`'s two-cell channel: sampled over evaluated sigma,
    /// minus one.
    fn over_correlated_change() -> f64 {
        0.1 / 0.0125_f64.sqrt() - 1.0
    }

    /// A matrix that is not a covariance is repaired to the nearest
    /// correlation matrix, keeping every sigma, and the repair is reported
    /// rather than absorbed.
    #[test]
    fn a_non_psd_matrix_is_repaired_keeping_every_sigma() {
        // A correlation of five: R has eigenvalues 6 and -4, and the nearest
        // correlation matrix is the correlation of one.
        let c = cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]);
        let s = Sampler::from_rate_covariance(&BTreeMap::from([("X".to_string(), c)]), 4);
        let repairs = s.repairs(3);
        assert_eq!(repairs.len(), 1);
        let r = &repairs[0];
        assert_eq!((r.nuclide.as_str(), r.spectrum), ("X", 3));
        let f = r.field;
        assert!((f.lambda_min + 4.0).abs() < 1e-12, "{f:?}");
        assert!((f.largest_correlation_change - 4.0).abs() < 1e-9, "{f:?}");
        assert!(
            (f.correlation_frobenius_change - 32.0_f64.sqrt()).abs() < 1e-9,
            "{f:?}"
        );
        assert_eq!((f.cells, f.held_cells, f.converged), (2, 0, true));
        assert_eq!(s.field_repairs()["X"], f);
        for ch in &r.channels {
            assert_eq!(ch.evaluated_variance, 0.01);
            assert!((ch.sampled - 0.1).abs() < 1e-15, "{ch:?}");
            assert!(ch.change().abs() < 1e-14, "{ch:?}");
        }
        let (relative, _) = s.sampled_covariance("X").unwrap();
        assert!((relative[1] - 0.01).abs() < 1e-11, "{relative:?}");
    }

    /// A matrix just past the threshold, with sigmas over three orders, is
    /// repaired to a PSD matrix with every variance as stated, whose
    /// correlation is the one Higham's alternating projections converge to.
    #[test]
    fn a_just_indefinite_matrix_keeps_its_sigmas_and_is_the_nearest() {
        let sigma = [0.001, 0.02, 0.3];
        let rho = [[1.0, 0.725, 0.725], [0.725, 1.0, 0.04], [0.725, 0.04, 1.0]];
        let n = 3;
        let stated_correlation: Vec<f64> = (0..n * n).map(|k| rho[k / n][k % n]).collect();
        let c: Vec<f64> = (0..n * n)
            .map(|k| rho[k / n][k % n] * sigma[k / n] * sigma[k % n])
            .collect();
        let f = Factorized::new(&c, &[]);
        let repair = f.repair.expect("repaired");
        // 1 + b/2 - sqrt(b^2/4 + 2a^2) for a = 0.725, b = 0.04: just below
        // zero.
        let lambda = 1.02 - (0.0004_f64 + 2.0 * 0.725 * 0.725).sqrt();
        assert!(lambda < 0.0 && lambda > -1e-2, "{lambda}");
        assert!((repair.lambda_min - lambda).abs() < 1e-12, "{repair:?}");
        assert!(repair.converged && repair.cells == 3, "{repair:?}");
        let repaired = repaired_covariance(&c, n).expect("repaired").0;
        let mut correlation = vec![0.0; n * n];
        for i in 0..n {
            assert_eq!(repaired[i * n + i], c[i * n + i]);
            for j in 0..n {
                assert_eq!(repaired[i * n + j], repaired[j * n + i]);
                correlation[i * n + j] = repaired[i * n + j] / (sigma[i] * sigma[j]);
            }
        }
        let min = eigenvalues(&correlation, n)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        assert!(min >= -1e-14, "{min}");
        let reference = crate::nearest_correlation::tests::dykstra(&stated_correlation, n);
        for (a, b) in correlation.iter().zip(&reference) {
            assert!(
                (a - b).abs() < 1e-8,
                "{correlation:?} against {reference:?}"
            );
        }
        let max = correlation
            .iter()
            .zip(&stated_correlation)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            (repair.largest_correlation_change - max).abs() < 1e-12,
            "{repair:?}"
        );
        // And the multipliers carry the stated sigmas.
        for (i, s) in sigma.iter().enumerate() {
            let sampled = f.relative_sampled[i * n + i].sqrt();
            assert!((sampled / s - 1.0).abs() < 1e-12, "{i} {sampled}");
        }
    }

    /// A covariance that needs no repair is sampled exactly as before the
    /// nearest-correlation repair existed: these are the bits the clipping
    /// sampler drew for the same seeds, relative and absolute. The bits were
    /// taken on x86-64 Linux; other targets may round the last place
    /// differently in `ln`, `exp` and fused arithmetic, so each value is held
    /// to within a few units in the last place rather than bit for bit.
    #[test]
    fn a_valid_matrix_draws_as_it_always_did() {
        // Within `ulps` units in the last place, for finite values of one sign.
        fn assert_ulps(got: &[u64], want: &[u64], ulps: u64, what: &str) {
            assert_eq!(got.len(), want.len(), "{what}");
            for (g, w) in got.iter().zip(want) {
                let (gv, wv) = (f64::from_bits(*g), f64::from_bits(*w));
                assert!(
                    gv.signum() == wv.signum() && g.abs_diff(*w) <= ulps,
                    "{what}: got {gv:e} ({g}), want {wv:e} ({w})"
                );
            }
        }
        let c = cov(
            &["a", "b", "c"],
            vec![
                0.04, 0.012, -0.004, //
                0.012, 0.09, 0.006, //
                -0.004, 0.006, 0.0025,
            ],
        );
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c)]));
        assert!(s.repairs(0).is_empty());
        let before: [[u64; 3]; 3] = [
            [
                13816687105146680565,
                13809921563573214875,
                13807147318081454187,
            ],
            [
                13799776928416527474,
                4596996472187675269,
                4591291726209650103,
            ],
            [
                13821431275358218956,
                13820582890229615446,
                4569554506801662790,
            ],
        ];
        for (r, want) in before.iter().enumerate() {
            let got: Vec<u64> = s.deviates(11, r as u64)["X"]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_ulps(&got, want, 4, &format!("replica {r}"));
        }
        let sampled: Vec<u64> = s
            .sampled_covariance("X")
            .unwrap()
            .0
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_ulps(
            &sampled,
            &[
                4585925428558828667,
                4578071150808694522,
                13794633745026886140,
                4578071150808694522,
                4591149604126578442,
                4573567551181324026,
                13794633745026886140,
                4573567551181324026,
                4567911030049346683,
            ],
            4,
            "sampled covariance",
        );

        let f = Factorized::new(&[0.01, 0.002, 0.002, 0.04], &[4.0, -1.0, -1.0, 9.0]);
        assert_eq!(f.repair, None);
        let g = f.gaussian_draw(7, 11, 2);
        let bits = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_ulps(
            &bits(&g.0),
            &[4580321971225683108, 13814906861411971440],
            4,
            "gaussian draw",
        );
        assert_ulps(
            &bits(&g.1),
            &[4613374713056438380, 4607834391228464327],
            4,
            "gaussian draw multipliers",
        );
    }

    /// Absolute cells are repaired as the relative ones are, on their own
    /// correlation matrix, keeping every variance in barn².
    #[test]
    fn absolute_cells_are_repaired_keeping_their_variances() {
        let absolute = [4.0, 9.0, 9.0, 16.0];
        let f = Factorized::new(&[], &absolute);
        let repair = f.repair.expect("repaired");
        // Correlation 9 / 8: eigenvalues 2.125 and -0.125.
        assert!((repair.lambda_min + 0.125).abs() < 1e-12, "{repair:?}");
        assert!((repair.largest_correlation_change - 0.125).abs() < 1e-9);
        assert_eq!(f.absolute_sampled[0], 4.0);
        assert_eq!(f.absolute_sampled[3], 16.0);
        assert!(
            (f.absolute_sampled[1] - 8.0).abs() < 1e-8,
            "{:?}",
            f.absolute_sampled
        );
    }

    /// Two fully correlated cells of different sigmas are a PSD `C` whose
    /// `ln(1 + C)` is not PSD. That is repaired in log space, keeping every
    /// log variance and so every sigma, and reported as a lognormal limit,
    /// not as a repair of the data.
    #[test]
    fn an_indefinite_log_covariance_keeps_every_sigma() {
        let (a, b) = (0.5_f64, 2.0_f64);
        let c = [a * a, a * b, a * b, b * b];
        let f = Factorized::new(&c, &[]);
        assert_eq!(f.repair, None);
        let limit = f.lognormal_limit.expect("not a lognormal's");
        let log = limit.log_space_repair.expect("repaired in log space");
        assert!(log.lambda_min < 0.0 && log.converged, "{log:?}");
        assert_eq!(log.held_cells, 0);
        assert!(limit.largest_sigma_change < 1e-12, "{limit:?}");
        assert!(limit.largest_correlation_change > 0.0, "{limit:?}");
        for (i, s) in [a, b].iter().enumerate() {
            let sampled = f.relative_sampled[i * 2 + i].sqrt();
            assert!((sampled / s - 1.0).abs() < 1e-12, "{sampled}");
        }
    }

    /// A cell stated at zero variance with a covariance to another has no
    /// correlation to keep: it is held, and every other cell keeps its sigma.
    #[test]
    fn a_zero_variance_cell_with_a_covariance_is_held() {
        let c = cov(&["a", "b"], vec![0.04, 0.01, 0.01, 0.0]);
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c)]));
        let r = &s.repairs(0)[0];
        assert_eq!((r.field.held_cells, r.field.cells), (1, 0));
        assert_eq!(r.channels[1].evaluated_sigma(), Some(0.0));
        assert_eq!(r.channels[1].sampled, 0.0);
        assert_eq!(r.channels[0].sampled, 0.2);

        let mut report = SigmaReport::default();
        report.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("b", 1.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(report.worst_sigma_change, 0.0);
        assert_eq!(report.repaired, BTreeSet::from(["X".to_string()]));
    }

    /// Unit-flux rates for one nuclide.
    fn unit_rates(nuclide: &str, kinds: &[(&str, f64)]) -> ReactionRates {
        HashMap::from([(
            nuclide.to_string(),
            kinds.iter().map(|(k, r)| (k.to_string(), *r)).collect(),
        )])
    }

    /// A negative folded diagonal is kept as the evaluation gives it, not
    /// read as a sigma of zero, and its cell, which has no sigma to keep, is
    /// held.
    #[test]
    fn a_negative_diagonal_is_kept_as_stated() {
        let c = cov(&["a", "b"], vec![0.04, 0.01, 0.01, -0.001]);
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c)]));
        let r = &s.repairs(0)[0];
        assert_eq!(r.field.held_cells, 1);
        let ch = &r.channels[1];
        assert_eq!(ch.evaluated_variance, -0.001);
        assert_eq!(ch.evaluated_sigma(), None);
        assert_eq!(ch.sampled, 0.0);
        assert_eq!(ch.change(), 0.0);
    }

    /// A channel that folds repaired cells reads their moved correlation, so
    /// its sigma can narrow, and the change is reported as it is.
    #[test]
    fn a_channel_over_repaired_cells_reads_the_moved_correlation() {
        let (cells, channels) = over_correlated();
        let s = folded_sampler(&[("X", cells, channels)], 1);
        let r = &s.repairs(0)[0];
        assert_eq!(r.channels[0].kind, "a");
        assert!(r.channels[0].change().abs() < 1e-12, "{:?}", r.channels[0]);
        let change = r.channels[1].change();
        assert!(
            (change - over_correlated_change()).abs() < 1e-9,
            "{change} against {}",
            over_correlated_change()
        );
        assert!(change < 0.0);
    }

    /// A repaired channel with no rate, or on a spectrum the schedule never
    /// irradiates with, cannot move the result, so it stays in the record but
    /// not in the headline, and a repair with no such channel is not a gap.
    #[test]
    fn the_worst_change_counts_only_channels_a_draw_can_move() {
        let (cells, channels) = over_correlated();
        let s = folded_sampler(&[("X", cells, channels)], 1);

        let mut no_rate = SigmaReport::default();
        no_rate.add(
            0,
            &s,
            &unit_rates("X", &[("a", 0.0), ("sum", 1.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(no_rate.repairs.len(), 1);
        assert_eq!(no_rate.repaired, BTreeSet::from(["X".to_string()]));
        assert!(
            (no_rate.worst_sigma_change - over_correlated_change().abs()).abs() < 1e-9,
            "{}",
            no_rate.worst_sigma_change
        );

        // Only the channel the repair did not move has a rate.
        let mut unmoved = SigmaReport::default();
        unmoved.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("sum", 0.0)]),
            1.0,
            &Default::default(),
            &everyone(),
        );
        assert!(unmoved.worst_sigma_change < 1e-12);

        let mut no_fluence = SigmaReport::default();
        no_fluence.add(
            0,
            &s,
            &unit_rates("X", &[("a", 1.0), ("sum", 1.0)]),
            0.0,
            &Default::default(),
            &everyone(),
        );
        assert_eq!(no_fluence.repairs.len(), 1);
        assert!(no_fluence.repaired.is_empty());
        assert_eq!(no_fluence.worst_sigma_change, 0.0);
    }

    /// A nuclide repaired on two spectra is one repaired nuclide with a
    /// record per spectrum.
    #[test]
    fn a_repair_on_two_spectra_is_one_repaired_nuclide() {
        let s = Sampler::from_rate_covariance(
            &BTreeMap::from([(
                "X".to_string(),
                cov(&["a", "b"], vec![0.01, 0.05, 0.05, 0.01]),
            )]),
            2,
        );
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
    fn the_rate_weighted_change_weighs_spectra_by_fluence() {
        let (cells, channels) = over_correlated();
        let repaired = folded_sampler(&[("X", cells, channels)], 2);
        let clean = folded_sampler(&[("X", vec![0.01, 0.0, 0.0, 0.01], channels)], 1);
        let r = unit_rates("X", &[("a", 1.0), ("sum", 1.0)]);
        let densities = HashMap::from([("X".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &clean, &r, 1.0e3, &densities, &everyone());
        report.add(1, &repaired, &r, 1.0, &densities, &everyone());

        // 2e3 channel-weights at no change and one at no change, against one
        // at the two-cell channel's.
        let want = over_correlated_change().abs() / 2.002e3;
        let got = report.rate_weighted_sigma_change().unwrap();
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
            !rate_sampler(&BTreeMap::from([("X".to_string(), c)]))
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
                rate_sampler(&BTreeMap::from([("X".to_string(), rank_one)]))
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
        assert!(rate_sampler(&BTreeMap::from([("X".to_string(), c)]))
            .repairs(0)
            .is_empty());
    }

    /// A huge variance on one channel cannot hide a real repair on the
    /// others: the test is on the correlation matrix, not against the largest
    /// eigenvalue of the covariance, and the repair is of the coupled pair
    /// alone, which leaves the wide channel exactly as stated.
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
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c)]));
        let repairs = s.repairs(0);
        assert_eq!(repairs.len(), 1);
        let r = &repairs[0];
        assert!((r.field.lambda_min + 4.0).abs() < 1e-12, "{:?}", r.field);
        assert_eq!(r.field.cells, 2);
        assert_eq!(r.channels[0].sampled, 1.0e17_f64.sqrt());
        for ch in &r.channels[1..] {
            assert!((ch.sampled - 0.1).abs() < 1e-15, "{ch:?}");
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
        assert!(
            report.worst_sigma_change < 1e-14,
            "{}",
            report.worst_sigma_change
        );
    }

    /// Coupled to the repaired pair, the wide channel is in the repaired
    /// block, and still keeps its sigma: the repair is on the correlation
    /// matrix, where it is one more unit diagonal.
    #[test]
    fn a_huge_channel_coupled_to_a_repair_keeps_its_sigma() {
        let w = 1.0e17_f64;
        let x = 0.5 * w.sqrt() * 0.1;
        let c = cov(
            &["wide", "a", "b"],
            vec![
                w, x, 0.0, //
                x, 0.01, 0.05, //
                0.0, 0.05, 0.01,
            ],
        );
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c)]));
        let repairs = s.repairs(0);
        assert_eq!(repairs.len(), 1);
        let r = &repairs[0];
        // R = [[1, 0.5, 0], [0.5, 1, 5], [0, 5, 1]], eigenvalues 1 and
        // 1 ± sqrt(25.25).
        let want = 1.0 - 25.25_f64.sqrt();
        assert!((r.field.lambda_min - want).abs() < 1e-12, "{:?}", r.field);
        assert_eq!(r.field.cells, 3);
        for (ch, sigma) in r.channels.iter().zip([w.sqrt(), 0.1, 0.1]) {
            assert!((ch.sampled / sigma - 1.0).abs() < 1e-9, "{ch:?}");
        }
    }

    /// A nuclide the material cannot populate has no repair record, no weight
    /// and no place in the wide-sigma lists, and is named instead as a repair
    /// outside the bound with its wide channels beside it, when a draw can
    /// move one of its channels.
    #[test]
    fn a_nuclide_the_material_cannot_populate_is_named_outside_the_bound() {
        let s = rate_sampler(&BTreeMap::from([(
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

        // With no rate on any of its channels no replica's draw moves it, so
        // outside the bound it is not named at all, and not a gap.
        let mut no_rate = SigmaReport::default();
        no_rate.add(
            0,
            &s,
            &unit_rates("X", &[("a", 0.0), ("b", 0.0)]),
            1.0,
            &densities,
            &Default::default(),
        );
        assert_eq!(no_rate, SigmaReport::default());
    }

    /// A wide channel with a small rate beside a repair on the channels that
    /// carry the reactions does not drown the repair out: the weight is on
    /// each channel's change, not on its sigma.
    #[test]
    fn a_wide_channel_does_not_mask_a_repair_on_the_dominant_ones() {
        let (cells, _) = over_correlated();
        let s = folded_sampler(
            &[
                ("X", cells, &[("sum", &[1.0, 1.0])]),
                ("Y", vec![1.0e6], &[("a", &[1.0])]),
            ],
            1,
        );
        let rates: ReactionRates = HashMap::from([
            ("X".to_string(), HashMap::from([("sum".to_string(), 1.0)])),
            ("Y".to_string(), HashMap::from([("a".to_string(), 0.01)])),
        ]);
        let densities = HashMap::from([("X".to_string(), 1.0), ("Y".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());

        // 99% of the weight on the channel the repair moved.
        let want = over_correlated_change().abs() / 1.01;
        let got = report.rate_weighted_sigma_change().unwrap();
        assert!((got - want).abs() < 1e-12, "{got} against {want}");
        assert!(got > 0.1);
    }

    /// A weighted channel the evaluation gives no sigma but the repair gives
    /// a spread makes the weighted headline infinite rather than ignored.
    #[test]
    fn a_spread_from_nothing_makes_the_weighted_change_infinite() {
        // Sigmas 0.1 and 0.2 at a stated correlation of -1.5: the two-cell
        // channel folds to a negative variance, and the repaired correlation
        // of -1 leaves it a spread.
        let s = folded_sampler(
            &[("X", vec![0.01, -0.03, -0.03, 0.04], &[("sum", &[1.0, 1.0])])],
            1,
        );
        let rates = unit_rates("X", &[("sum", 1.0)]);
        let densities = HashMap::from([("X".to_string(), 1.0)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());
        let ch = &report.repairs[0].channels[0];
        assert_eq!(ch.evaluated_sigma(), None);
        assert!(ch.sampled > 0.0, "{ch:?}");
        assert_eq!(report.worst_sigma_change, f64::INFINITY);
        assert_eq!(report.rate_weighted_sigma_change(), Some(f64::INFINITY));
    }

    /// The rate-weighted headline weighs each channel by its rate times its
    /// parent's density, over every sampled channel.
    #[test]
    fn the_rate_weighted_change_counts_every_sampled_channel() {
        let (cells, channels) = over_correlated();
        let s = folded_sampler(
            &[("X", cells, channels), ("Y", vec![0.04], &[("a", &[1.0])])],
            1,
        );
        let rates: ReactionRates = HashMap::from([
            (
                "X".to_string(),
                HashMap::from([("a".to_string(), 1.0), ("sum".to_string(), 1.0)]),
            ),
            ("Y".to_string(), HashMap::from([("a".to_string(), 2.0)])),
        ]);
        let densities = HashMap::from([("X".to_string(), 1.0), ("Y".to_string(), 0.5)]);
        let mut report = SigmaReport::default();
        report.add(0, &s, &rates, 1.0, &densities, &everyone());

        // X: a one-cell channel unmoved and the two-cell one moved, weight 1
        // each. Y: one channel unmoved, weight 2 * 0.5.
        let want = over_correlated_change().abs() / 3.0;
        let got = report.rate_weighted_sigma_change().unwrap();
        assert!((got - want).abs() < 1e-12, "{got} against {want}");
        assert!((report.worst_sigma_change - over_correlated_change().abs()).abs() < 1e-9);
        assert_eq!(report.repairs.len(), 1);

        // No repair anywhere reads as exactly zero, not as round-off.
        let mut clean = SigmaReport::default();
        let y = rate_sampler(&BTreeMap::from([(
            "Y".to_string(),
            cov(&["a"], vec![0.04]),
        )]));
        clean.add(0, &y, &rates, 1.0, &densities, &everyone());
        assert_eq!(clean.rate_weighted_sigma_change(), Some(0.0));
        assert_eq!(clean.worst_sigma_change, 0.0);
    }

    /// Channels evaluated at a relative sigma of one or ten and above are named,
    /// and a channel with no rate or a zero one, which a draw cannot move, is
    /// not.
    #[test]
    fn wide_sigmas_are_named() {
        let s = rate_sampler(&BTreeMap::from([(
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

        // Evaluated at 0.9 and repaired, keeping the sigma: the list is of
        // what the evaluation states, and the repair is in its own record.
        let y = rate_sampler(&BTreeMap::from([(
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
        let sampled = repaired.repairs[0].channels[0].sampled;
        assert!((sampled - 0.9).abs() < 1e-12, "{sampled}");
        assert!(repaired.sigma_at_least_one.is_empty());
    }

    /// The same (seed, replica, nuclide) gives the same deviates, and a
    /// different replica gives different ones.
    #[test]
    fn deviates_are_a_pure_function_of_seed_and_replica() {
        let c = cov(&["a", "b"], vec![0.04, 0.0, 0.0, 0.09]);
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c)]));

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
        let alone = rate_sampler(&BTreeMap::from([("Fe56".to_string(), a.clone())]));
        let together = rate_sampler(&BTreeMap::from([
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
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c)]));

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
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c)]));

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
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c)]));
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 1.0e-8)]),
        )]);

        let mut sampled = 0;
        for k in 0..2000 {
            let (out, n) = s.perturb(0, &rates, 9, k);
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
        let s = rate_sampler(&BTreeMap::from([("Fe56".to_string(), c)]));
        let nominal = 1.0e-8;
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), nominal)]),
        )]);

        let n = 20_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for k in 0..n {
            let v = s.perturb(0, &rates, 11, k).0["Fe56"]["(n,gamma)"];
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
                let lognormal = super::lognormal_multiplier(z, sigma);
                let linear = 1.0 + sigma * z;
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
        let s = rate_sampler(&BTreeMap::new());
        assert!(s.is_empty());
        let rates: ReactionRates = HashMap::from([(
            "Fe56".to_string(),
            HashMap::from([("(n,gamma)".to_string(), 2.5)]),
        )]);
        let (out, sampled) = s.perturb(0, &rates, 1, 0);
        assert_eq!(out, rates);
        assert_eq!(sampled, 0);
    }

    /// Small enough that a draw almost never takes a rate below zero, where
    /// the floor would cut the sampled covariance short of the stated one.
    const MIXED_SHORT_VARIANCE: f64 = 1.0e-4;

    /// A field with every kind of cell, read by two spectra: two relative
    /// cells, one absolute, and one short-range interval `[0, 4]` that the
    /// two spectra cut differently. Unit rates, so a shift is a relative one.
    fn mixed_field() -> crate::covariance_fold::CellField {
        use crate::covariance_fold::{Cell, CellField, Projection, ShortRange, ShortTerm};
        let projection = |p: [f64; 2], q: f64, cuts: Vec<f64>, weight: Vec<f64>| {
            Some(Projection {
                kinds: vec!["x".to_string()],
                rates: vec![1.0],
                relative: p.to_vec(),
                absolute: vec![q],
                short: vec![vec![ShortTerm {
                    block: 0,
                    interval: 0,
                    cuts,
                    weight,
                }]],
            })
        };
        CellField {
            relative_cells: vec![
                Cell {
                    mt: 1,
                    lo: 0.0,
                    hi: 1.0,
                },
                Cell {
                    mt: 1,
                    lo: 1.0,
                    hi: 2.0,
                },
            ],
            relative: vec![0.04, 0.02, 0.02, 0.09],
            absolute_cells: vec![Cell {
                mt: 2,
                lo: 0.0,
                hi: 1.0,
            }],
            absolute: vec![0.25],
            short: vec![ShortRange {
                key: (1, 0, 0),
                mt: 1,
                edges: vec![0.0, 4.0],
                variance: vec![MIXED_SHORT_VARIANCE],
            }],
            projections: vec![
                projection([1.0, 0.5], 0.2, vec![0.0, 2.0, 4.0], vec![1.0, 3.0]),
                projection([0.3, 1.0], 0.4, vec![0.0, 1.0, 4.0], vec![2.0, 1.0]),
            ],
        }
    }

    /// The analytic covariance of the two spectra's shifts: `pᵀ C p'` over
    /// the relative cells, `qᵀ C q'` over the absolute one, and
    /// `F ΔE ∫ ψ ψ' dE` over the short-range interval.
    fn mixed_covariance(a: usize, b: usize) -> f64 {
        let p = [[1.0, 0.5], [0.3, 1.0]];
        let q = [0.2, 0.4];
        let c = [[0.04, 0.02], [0.02, 0.09]];
        let relative: f64 = (0..2)
            .flat_map(|k| (0..2).map(move |l| (k, l)))
            .map(|(k, l)| p[a][k] * c[k][l] * p[b][l])
            .sum();
        // Piecewise weights on the union [0, 1, 2, 4] of both cuts.
        let psi = [[1.0, 1.0, 3.0], [2.0, 1.0, 1.0]];
        let widths = [1.0, 1.0, 2.0];
        let short: f64 = (0..3)
            .map(|j| MIXED_SHORT_VARIANCE * 4.0 * widths[j] * psi[a][j] * psi[b][j])
            .sum();
        relative + q[a] * 0.25 * q[b] + short
    }

    /// Each spectrum's sampled variance is its analytic one.
    #[test]
    fn a_mixed_field_samples_its_stated_variance_under_each_spectrum() {
        let field = mixed_field();
        let s = Sampler::new(
            &BTreeMap::from([("X".to_string(), field)]),
            &[BTreeMap::new(), BTreeMap::new()],
        );
        for a in 0..2 {
            let (_, _, _, sampled) = s.channel_sigmas(a).next().expect("one channel");
            let want = mixed_covariance(a, a).sqrt();
            assert!(
                (sampled - want).abs() < 1e-12,
                "spectrum {a}: {sampled} vs {want}"
            );
        }
    }

    /// Two spectra reading one draw covary as the field says, including the
    /// short-range term through the energies both weight, and the first-order
    /// rows reproduce the same covariance over the shared columns.
    #[test]
    fn two_spectra_reading_one_field_covary_as_the_field_says() {
        let s = Sampler::new(
            &BTreeMap::from([("X".to_string(), mixed_field())]),
            &[BTreeMap::new(), BTreeMap::new()],
        );
        let rates: ReactionRates =
            HashMap::from([("X".to_string(), HashMap::from([("x".to_string(), 1.0)]))]);
        let n = 40_000;
        let (mut sab, mut saa, mut sbb, mut ma, mut mb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for r in 0..n {
            let draw = s.draw(13, r);
            let a = s.perturb_with(&draw, 0, &rates).0["X"]["x"] - 1.0;
            let b = s.perturb_with(&draw, 1, &rates).0["X"]["x"] - 1.0;
            sab += a * b;
            saa += a * a;
            sbb += b * b;
            ma += a;
            mb += b;
        }
        let nf = n as f64;
        let cov = sab / nf - (ma / nf) * (mb / nf);
        let want = mixed_covariance(0, 1);
        // The standard error of a sample covariance is about
        // sqrt((var_a var_b + cov^2) / n).
        let se = ((saa / nf) * (sbb / nf) + want * want).sqrt() / nf.sqrt();
        assert!(
            (cov - want).abs() < 4.0 * se,
            "cross covariance {cov} vs {want} (se {se})"
        );

        let fa = s.factors(0);
        let fb = s.factors(1);
        let (_, _, ja, cols) = &fa[0];
        let (_, _, jb, cols_b) = &fb[0];
        assert_eq!(cols, cols_b, "both spectra read the same columns");
        let linear: f64 = (0..*cols).map(|k| ja[k] * jb[k]).sum();
        assert!(
            (linear - want).abs() < 1e-12,
            "first-order {linear} vs {want}"
        );
    }

    /// The relative cells are lognormal with the evaluation's covariance:
    /// every multiplier positive, mean one, and the stated covariance.
    #[test]
    fn relative_cells_are_lognormal_with_the_stated_covariance() {
        let c = cov(&["a", "b"], vec![0.25, 0.15, 0.15, 0.36]);
        let s = rate_sampler(&BTreeMap::from([("X".to_string(), c.clone())]));
        let n = 40_000;
        let mut sum = [0.0; 2];
        let mut cross = [[0.0; 2]; 2];
        for r in 0..n {
            let d = &s.deviates(3, r)["X"];
            for k in 0..2 {
                assert!(d[k] > -1.0, "multiplier {} not positive", 1.0 + d[k]);
                sum[k] += d[k];
                for l in 0..2 {
                    cross[k][l] += d[k] * d[l];
                }
            }
        }
        let nf = n as f64;
        for k in 0..2 {
            assert!((sum[k] / nf).abs() < 0.01, "mean shift {}", sum[k] / nf);
            for l in 0..2 {
                let got = cross[k][l] / nf - (sum[k] / nf) * (sum[l] / nf);
                assert!(
                    (got - c.get(k, l)).abs() < 0.06 * c.get(k, k).max(c.get(l, l)),
                    "({k},{l}): {got} vs {}",
                    c.get(k, l)
                );
            }
        }
    }
}
