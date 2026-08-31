//! Per-history Welford with per-worker accumulator + lazy zero-fold
//! at read time. Welford of the variance-algorithm migration; see
//! auto-memory `project-variance-welford-decision`.
//!
//! # Data flow
//!
//! For each rayon worker, per simulation:
//!
//! 1. [`WelfordWorkerState::new`] allocates a per-worker (n_touched,
//!    mean, M2) array per tally plus an empty sparse scratch hash map.
//! 2. During a history, scoring writes contributions to
//!    [`WelfordTallyWorker::add_contribution`] which sums them into
//!    `scratch_map: FxHashMap<bin_idx, value>` -- only the bins
//!    actually touched in the current history are kept resident.
//! 3. [`WelfordWorkerState::finish_history`] is called at the end of
//!    each source particle's history (after all secondaries die).
//!    Iterates the sparse map, applies one Welford update per touched
//!    bin to the worker's `welford` state, clears the map.
//! 4. After the rayon `par_iter().fold(...)` produces N_workers
//!    [`WelfordWorkerState`]s, [`WelfordWorkerState::combine`] is
//!    invoked pairwise in the `.reduce()` step. Per-bin Chen
//!    pairwise combine; per-tally n_histories sums.
//! 5. The final combined state is converted to global
//!    [`WelfordGlobalStats`] via [`WelfordWorkerState::finalize`] --
//!    this is where lazy zero-fold happens: for each bin with
//!    `n_touched < n_histories`, fold in `(n_histories - n_touched)`
//!    implicit zero samples via closed-form Chen.
//!
//! # Correctness
//!
//! Lazy zero-fold at finalize is mathematically equivalent to eager
//! fold-zeros at history-end + Chen pairwise combine. The "fold K
//! zero samples" closed form is:
//!
//! ```text
//! new_n    = K
//! new_mean = mean * n_touched / K
//! new_M2   = M2 + mean² * n_touched * (K - n_touched) / K
//! ```
//!
//! ...applied per bin at finalize, where `K = n_histories`.
//!
//! # Memory
//!
//! Per-worker peak: `Σ_tallies (welford_bytes + scratch_map_bytes)`.
//! Welford is dense -- `num_bins × 24 bytes` (the AoS `WelfordBin`).
//! The scratch map grows to ~the number of bins touched in the
//! currently-in-progress history (typical: a few hundred entries
//! ≲ 8 KB per tally). For a 1M-bin mesh tally on 32 workers this
//! is ~768 MB of welford + negligible scratch; the previous dense
//! scratch implementation added another ~288 MB on top.

/// Welford state per bin: count of histories that touched the bin,
/// running mean, running sum of squared deviations. AoS layout (vs
/// three separate Vecs) reduces cache lines touched per bin update
/// at history-end fold from 3 to 1 (each `WelfordBin` is 24 bytes;
/// alignment-padded to 24; 2.6 bins per 64-byte cache line).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WelfordBin {
    pub n_touched: u32,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordBin {
    pub const ZERO: WelfordBin = WelfordBin {
        n_touched: 0,
        mean: 0.0,
        m2: 0.0,
    };
}

/// Running central moments (count, mean, M2, M3, M4) of a single scalar
/// series, updated one sample at a time with the numerically-stable
/// higher-order recurrence. Tracked per tally for the per-history *total*
/// score, so tally-level variance-of-the-variance, skewness and kurtosis
/// can be derived without any per-bin cost. `M2`/`M3`/`M4` are the summed
/// central powers (not divided by n), matching the per-bin `m2` convention.
#[derive(Debug, Clone, Copy, Default)]
pub struct AggMoments {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
}

impl AggMoments {
    pub const ZERO: AggMoments = AggMoments {
        n: 0,
        mean: 0.0,
        m2: 0.0,
        m3: 0.0,
        m4: 0.0,
    };

    /// Fold one sample in (higher-order Welford). Order matters: M4 reads the
    /// old M2/M3, M3 reads the old M2, then M2 last.
    #[inline]
    pub fn update(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// Closed-form pairwise merge of another set into this one (parallel
    /// higher-order combine). Exact in real arithmetic for any split.
    pub fn combine(&mut self, b: &AggMoments) {
        if b.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *b;
            return;
        }
        let na = self.n as f64;
        let nb = b.n as f64;
        let n = na + nb;
        let delta = b.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;
        let m4 = self.m4
            + b.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * b.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * b.m3 - nb * self.m3) / n;
        let m3 = self.m3
            + b.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * b.m2 - nb * self.m2) / n;
        let m2 = self.m2 + b.m2 + delta2 * na * nb / n;
        self.mean += delta * nb / n;
        self.m2 = m2;
        self.m3 = m3;
        self.m4 = m4;
        self.n += b.n;
    }

    /// Relative variance of the variance: `M4 / M2² − 1/n`. A sensitive
    /// reliability metric -- large values flag that the estimated error is
    /// itself unreliable. 0 when undefined (n < 2 or no spread).
    pub fn variance_of_variance(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        (self.m4 / (self.m2 * self.m2) - 1.0 / n).max(0.0)
    }

    /// Population skewness `√n · M3 / M2^{3/2}`. 0 when undefined.
    pub fn skewness(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        n.sqrt() * self.m3 / self.m2.powf(1.5)
    }

    /// Population excess kurtosis `n · M4 / M2² − 3`. 0 when undefined.
    pub fn excess_kurtosis(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }
}

/// Lowest magnitude (decimal exponent) tracked by the empirical score PDF.
pub const PDF_LOG_MIN: f64 = -30.0;
/// Highest magnitude (decimal exponent) tracked by the empirical score PDF.
pub const PDF_LOG_MAX: f64 = 30.0;
/// Log-spaced bins per decade of magnitude.
pub const PDF_BINS_PER_DECADE: usize = 4;
/// Total fixed bin count of the empirical score PDF.
pub const PDF_NBINS: usize = ((PDF_LOG_MAX - PDF_LOG_MIN) as usize) * PDF_BINS_PER_DECADE;

/// Empirical probability density of the per-history *total* score, in raw
/// form: a fixed log-spaced histogram of `|total score|`. The bin layout is
/// fixed (constants above) so independent runs combine by summing counts.
/// Bin `i` spans `|x|` in `[10^(LOG_MIN + i/BPD), 10^(LOG_MIN + (i+1)/BPD))`.
/// Number of largest history scores retained for the Hill tail-index fit.
pub const PDF_TAIL_K: usize = 50;

#[derive(Debug, Clone, Default)]
pub struct ScorePdf {
    /// Per-bin counts (length [`PDF_NBINS`]); empty when no sampling
    /// occurred (e.g. the GPU path installs no per-history state).
    pub counts: Vec<u64>,
    /// Histories whose total score was exactly zero (or non-finite).
    pub zero: u64,
    /// The [`PDF_TAIL_K`] largest positive history scores seen, kept sorted
    /// ascending. Used for a Hill tail-index estimate of the PDF tail slope
    /// (far more robust than a histogram fit). Not exposed; a few hundred
    /// bytes per tally.
    pub top: Vec<f64>,
}

impl ScorePdf {
    pub fn new() -> Self {
        Self {
            counts: vec![0; PDF_NBINS],
            zero: 0,
            top: Vec::with_capacity(PDF_TAIL_K + 1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Record one per-history total score into the histogram and the
    /// top-K tail buffer.
    #[inline]
    pub fn record(&mut self, x: f64) {
        let a = x.abs();
        if a == 0.0 || !a.is_finite() {
            self.zero += 1;
            return;
        }
        let pos = (a.log10() - PDF_LOG_MIN) * PDF_BINS_PER_DECADE as f64;
        let idx = if pos < 0.0 {
            0
        } else {
            (pos as usize).min(PDF_NBINS - 1)
        };
        self.counts[idx] += 1;
        self.push_top(a);
    }

    /// Insert `a` into the ascending top-K buffer (smallest first).
    #[inline]
    fn push_top(&mut self, a: f64) {
        if self.top.len() < PDF_TAIL_K {
            let pos = self.top.partition_point(|&v| v < a);
            self.top.insert(pos, a);
        } else if a > self.top[0] {
            self.top.remove(0);
            let pos = self.top.partition_point(|&v| v < a);
            self.top.insert(pos, a);
        }
    }

    /// Sum another histogram (same fixed layout) into this one and merge the
    /// tail buffers.
    pub fn combine(&mut self, other: &ScorePdf) {
        if other.counts.is_empty() {
            return;
        }
        if self.counts.is_empty() {
            self.counts = other.counts.clone();
            self.zero += other.zero;
            self.top = other.top.clone();
            return;
        }
        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a += *b;
        }
        self.zero += other.zero;
        // Merge the two ascending top-K buffers, keep the K largest.
        let mut merged = self.top.clone();
        merged.extend_from_slice(&other.top);
        merged.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = merged.len();
        if n > PDF_TAIL_K {
            merged.drain(0..n - PDF_TAIL_K);
        }
        self.top = merged;
    }

    /// Lower magnitude edge of bin `i`.
    pub fn bin_lower_edge(i: usize) -> f64 {
        10f64.powf(PDF_LOG_MIN + i as f64 / PDF_BINS_PER_DECADE as f64)
    }

    /// Slope `s` of the large-score PDF tail (`f(x) ∝ x^{-s}`), estimated
    /// with the Hill tail-index estimator over the [`PDF_TAIL_K`] largest
    /// history scores. The tally's variance is well-defined only when
    /// `s > 3`; a light/bounded tail returns a large slope (capped at 10).
    /// Returns 0.0 ("not estimable") when there are too few distinct large
    /// samples to fit.
    pub fn tail_slope(&self) -> f64 {
        let t = &self.top;
        if t.len() < 5 {
            return 0.0;
        }
        // Threshold = smallest of the retained top-K (ascending buffer).
        let thresh = t[0];
        if thresh <= 0.0 {
            return 0.0;
        }
        // Hill estimator of the CDF tail index: alpha = (k-1) / Σ ln(x_i / u).
        let mut sum = 0.0;
        let mut count = 0.0;
        for &x in &t[1..] {
            if x > 0.0 {
                sum += (x / thresh).ln();
                count += 1.0;
            }
        }
        if count < 1.0 || sum <= 0.0 {
            // All retained values nearly equal -> extremely light tail.
            return 10.0;
        }
        let alpha = count / sum;
        // PDF tail exponent f(x) ∝ x^{-(alpha+1)}.
        let s = alpha + 1.0;
        if !s.is_finite() || s <= 0.0 {
            return 0.0;
        }
        s.min(10.0)
    }
}

/// Per-tally state for one rayon worker: Welford accumulator + per-
/// history sparse scratch.
///
/// Scratch is a hash map (`FxHashMap<bin_idx, summed_value>`) rather
/// than three dense Vecs (`scratch`, `scratch_touched`, `touched_bins`)
/// because on a 1M-bin tally the dense form occupies ~9 MB per worker
/// and every per-event `add_contribution` is a random touch into that
/// 9 MB -- guaranteed L3 miss when the welford state (32 MB / worker)
/// also lives nearby. The sparse map holds only the ~100–500 bins
/// actually touched in the current history (≲ 8 KB), so per-event
/// reads/writes stay in L1.
#[derive(Debug)]
pub struct WelfordTallyWorker {
    /// Per-bin (n_touched, mean, M2) Welford state. Dense -- must allow
    /// O(1) random access at fold time and is what `combine` /
    /// `finalize` consume.
    pub welford: Vec<WelfordBin>,
    /// Per-history sparse scratch: `bin_idx -> sum_of_contributions`
    /// for bins touched in the currently-in-progress history. Cleared
    /// at the end of each history. FxHash is used because keys are
    /// already-uniform u32 indices -- no need to pay SipHash's overhead.
    pub scratch_map: rustc_hash::FxHashMap<u32, f64>,
    /// Moments of this tally's per-history *total* score (sum over all
    /// bins). Updated once per history from the scratch walk -- a handful
    /// of scalars, no per-bin cost. Drives the tally-level reliability
    /// metrics.
    pub agg: AggMoments,
    /// Empirical PDF of the per-history total score (raw histogram).
    pub score_pdf: ScorePdf,
}

impl WelfordTallyWorker {
    pub fn new(num_bins: usize) -> Self {
        Self {
            welford: vec![WelfordBin::ZERO; num_bins],
            // Capacity hint: enough to avoid most rehashes during a
            // single history. 256 entries × 12 B ≈ 3 KB; cheap, well
            // within L1.
            scratch_map: rustc_hash::FxHashMap::with_capacity_and_hasher(256, Default::default()),
            agg: AggMoments::ZERO,
            score_pdf: ScorePdf::new(),
        }
    }
}

/// Per-rayon-worker accumulator state. Lives in the worker's fold
/// closure, gets combined pairwise in `.reduce()` after the fold
/// completes.
#[derive(Debug)]
pub struct WelfordWorkerState {
    /// One per tally (indexed by tally position in the model's
    /// `tallies` list). Order is preserved from the input.
    pub tallies: Vec<WelfordTallyWorker>,
    /// Total source histories processed by this worker. Increments
    /// once per `finish_history`. After combine, this is the sum
    /// across all workers folded in.
    pub n_histories: u64,
}

impl WelfordWorkerState {
    /// Allocate a fresh per-worker state, sized by per-tally bin counts.
    pub fn new(tally_num_bins: &[usize]) -> Self {
        Self {
            tallies: tally_num_bins
                .iter()
                .copied()
                .map(WelfordTallyWorker::new)
                .collect(),
            n_histories: 0,
        }
    }

    /// Add a per-step contribution to a bin in the currently-in-progress
    /// history. Hot path -- kept inline. `scratch_map.entry(...)` is one
    /// hash lookup against the L1-resident sparse-touched-bins map; on
    /// first touch in this history it inserts a `0.0` slot, on
    /// subsequent touches it returns the existing slot.
    ///
    /// Signed contributions that cancel back to zero mid-history are
    /// fine: the slot stays in `scratch_map` until `finish_history`
    /// drains it, so the bin still contributes one Welford update at
    /// history-end (with `value = 0.0`), exactly as the dense-array
    /// implementation did before.
    #[inline]
    pub fn add_contribution(&mut self, tally_idx: usize, bin_idx: usize, value: f64) {
        let t = &mut self.tallies[tally_idx];
        *t.scratch_map.entry(bin_idx as u32).or_insert(0.0) += value;
    }

    /// Called when a source history finishes (after all secondaries
    /// die). Walks the sparse `scratch_map` and applies the per-bin
    /// Welford update for each touched bin, then clears the map.
    ///
    /// Iteration order is whatever the hash map gives us -- Welford
    /// updates are independent across slots, so order doesn't change
    /// the result.
    pub fn finish_history(&mut self) {
        self.n_histories += 1;
        for t in self.tallies.iter_mut() {
            // Sum the per-history total score while doing the per-bin Welford
            // walk we already do -- so the tally-level moments / PDF cost one
            // add per touched bin plus a constant per history, never touching
            // the per-event hot path. Untouched tallies record a zero sample.
            let mut total = 0.0;
            for (&bin_u32, &x) in t.scratch_map.iter() {
                let w = &mut t.welford[bin_u32 as usize];
                w.n_touched += 1;
                let n = w.n_touched as f64;
                let delta = x - w.mean;
                w.mean += delta / n;
                let delta2 = x - w.mean;
                w.m2 += delta * delta2;
                total += x;
            }
            t.scratch_map.clear();
            t.agg.update(total);
            t.score_pdf.record(total);
        }
    }

    /// Chen pairwise combine of two worker states. The result has
    /// `n_histories = a.n_histories + b.n_histories`; per-bin Welford
    /// state is combined treating each side's `n_touched[bin]` as the
    /// local sample count (Chen handles unequal n per bin correctly).
    /// Used in the rayon `.reduce()` step.
    pub fn combine(mut self, other: WelfordWorkerState) -> WelfordWorkerState {
        debug_assert_eq!(
            self.tallies.len(),
            other.tallies.len(),
            "worker state shapes must match"
        );
        for (a, b) in self.tallies.iter_mut().zip(other.tallies) {
            debug_assert_eq!(
                a.welford.len(),
                b.welford.len(),
                "per-tally bin counts must match"
            );
            for (wa, wb) in a.welford.iter_mut().zip(b.welford.iter()) {
                let n_a = wa.n_touched as f64;
                let n_b = wb.n_touched as f64;
                let n_ab = n_a + n_b;
                if n_ab == 0.0 {
                    continue;
                }
                let delta = wb.mean - wa.mean;
                wa.mean += delta * n_b / n_ab;
                wa.m2 += wb.m2 + delta * delta * n_a * n_b / n_ab;
                wa.n_touched += wb.n_touched;
            }
            a.agg.combine(&b.agg);
            a.score_pdf.combine(&b.score_pdf);
        }
        self.n_histories += other.n_histories;
        self
    }

    /// Apply lazy zero-fold to produce the global Welford state.
    /// For each bin with `n_touched < n_histories`, Chen-combine in
    /// `(n_histories - n_touched, 0, 0)` to bring the bin's effective
    /// sample count up to `n_histories`. Returns the global stats.
    pub fn finalize(self) -> WelfordGlobalStats {
        let n_hist = self.n_histories;
        let n_hist_f = n_hist as f64;
        let per_tally: Vec<WelfordTallyStats> = self
            .tallies
            .into_iter()
            .map(|t| {
                let mut mean: Vec<f64> = t.welford.iter().map(|w| w.mean).collect();
                let mut m2: Vec<f64> = t.welford.iter().map(|w| w.m2).collect();
                let agg = t.agg;
                let score_pdf = t.score_pdf;
                if n_hist == 0 {
                    return WelfordTallyStats {
                        mean,
                        m2,
                        n_histories: 0,
                        agg,
                        score_pdf,
                    };
                }
                for i in 0..mean.len() {
                    let n_touched = t.welford[i].n_touched as f64;
                    if n_touched == 0.0 {
                        // Bin never touched; mean and m2 stay 0; new_n = K.
                        continue;
                    }
                    if n_touched as u64 == n_hist {
                        // No zero-fold needed; bin was touched every history.
                        continue;
                    }
                    let k_zeros = n_hist_f - n_touched;
                    let mean_orig = mean[i];
                    // Chen combine of (n_touched, mean, m2) with (k_zeros, 0, 0).
                    mean[i] = mean_orig * n_touched / n_hist_f;
                    m2[i] += mean_orig * mean_orig * n_touched * k_zeros / n_hist_f;
                }
                WelfordTallyStats {
                    mean,
                    m2,
                    n_histories: n_hist,
                    agg,
                    score_pdf,
                }
            })
            .collect();
        WelfordGlobalStats { per_tally }
    }
}

/// Final global Welford stats per tally, after lazy zero-fold.
/// Consumed by the tally's `get_mean` / `get_std_dev` / etc. when
/// `welford` is enabled.
#[derive(Debug, Clone)]
pub struct WelfordGlobalStats {
    pub per_tally: Vec<WelfordTallyStats>,
}

#[derive(Debug, Clone)]
pub struct WelfordTallyStats {
    /// Per-bin Welford mean. After finalize, this is the global
    /// per-history score-mean for the bin, treating untouched
    /// histories as contributing zero.
    pub mean: Vec<f64>,
    /// Per-bin Welford sum of squared deviations.
    pub m2: Vec<f64>,
    /// Total source histories processed (uniform across all bins).
    pub n_histories: u64,
    /// Moments of the per-history total score (sum over bins). Drives the
    /// tally-level variance-of-the-variance / skewness / kurtosis. `ZERO`
    /// when no per-history sampling occurred (e.g. GPU runs).
    pub agg: AggMoments,
    /// Raw empirical PDF of the per-history total score. Empty when no
    /// per-history sampling occurred.
    pub score_pdf: ScorePdf,
}

impl WelfordTallyStats {
    /// Chen pairwise combine with another finalized per-tally stats.
    ///
    /// After [`WelfordWorkerState::finalize`]'s lazy zero-fold every
    /// bin's effective sample count equals `n_histories` on each side,
    /// so the per-bin merge uses the scalar history counts. The result
    /// is the statistics of the pooled histories "as if one run":
    /// exact in real arithmetic, deterministic for a given operand
    /// order. Used for cross-run [`combine_results`](crate::combine)
    /// merging and for the MPI rank reduction.
    pub fn combine(&mut self, other: &WelfordTallyStats) -> Result<(), String> {
        if self.mean.len() != other.mean.len() {
            return Err(format!(
                "cannot combine Welford stats with different bin counts ({} vs {})",
                self.mean.len(),
                other.mean.len()
            ));
        }
        if other.n_histories == 0 {
            return Ok(());
        }
        if self.n_histories == 0 {
            self.mean.clone_from(&other.mean);
            self.m2.clone_from(&other.m2);
            self.n_histories = other.n_histories;
            self.agg = other.agg;
            self.score_pdf = other.score_pdf.clone();
            return Ok(());
        }
        let n_a = self.n_histories as f64;
        let n_b = other.n_histories as f64;
        let n_ab = n_a + n_b;
        for i in 0..self.mean.len() {
            let delta = other.mean[i] - self.mean[i];
            self.mean[i] += delta * n_b / n_ab;
            self.m2[i] += other.m2[i] + delta * delta * n_a * n_b / n_ab;
        }
        self.agg.combine(&other.agg);
        self.score_pdf.combine(&other.score_pdf);
        self.n_histories += other.n_histories;
        Ok(())
    }

    /// Standard error of the mean per bin: `sqrt(m2 / ((n-1) * n))`.
    /// Returns zeros if `n_histories <= 1`.
    pub fn std_err(&self) -> Vec<f64> {
        if self.n_histories <= 1 {
            return vec![0.0; self.mean.len()];
        }
        let n = self.n_histories as f64;
        self.m2
            .iter()
            .map(|&m| (m / ((n - 1.0) * n)).max(0.0).sqrt())
            .collect()
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// One worker, fixed per-history per-bin scores, all bins touched
    /// every history. After finalize, mean and std_err should match
    /// a direct hand calculation.
    #[test]
    fn single_worker_all_bins_touched() {
        let mut w = WelfordWorkerState::new(&[3]);
        // 3 histories, 3 bins. history i, bin j: value = (i+1) * (j+1)
        for i in 0..3 {
            for j in 0..3 {
                w.add_contribution(0, j, (i + 1) as f64 * (j + 1) as f64);
            }
            w.finish_history();
        }
        let global = w.finalize();
        let t = &global.per_tally[0];
        assert_eq!(t.n_histories, 3);
        // Mean per bin = average of (1*(j+1), 2*(j+1), 3*(j+1)) = 2*(j+1)
        for j in 0..3 {
            let expected = 2.0 * (j + 1) as f64;
            assert!((t.mean[j] - expected).abs() < 1e-12);
        }
    }

    /// Single worker, sparse pattern: only bin 0 is touched, and
    /// only in 2 out of 3 histories. After finalize the bin's mean
    /// should be scaled down by the "fold-in" of the third (zero) history.
    #[test]
    fn lazy_zero_fold_handles_untouched_history() {
        let mut w = WelfordWorkerState::new(&[2]);
        // history 0: bin 0 = 10
        w.add_contribution(0, 0, 10.0);
        w.finish_history();
        // history 1: bin 0 = 14
        w.add_contribution(0, 0, 14.0);
        w.finish_history();
        // history 2: nothing scored (zero sample for both bins)
        w.finish_history();

        let global = w.finalize();
        let t = &global.per_tally[0];
        assert_eq!(t.n_histories, 3);
        // True mean over 3 histories: (10 + 14 + 0) / 3 = 8.0
        assert!((t.mean[0] - 8.0).abs() < 1e-12);
        assert!((t.mean[1] - 0.0).abs() < 1e-12);
    }

    /// Mid-history contributions that cancel to zero (e.g. signed
    /// photon-heating scores) must not double-record the bin in
    /// `touched_bins`. Regression test for the previous
    /// `scratch[bin] == 0.0` first-touch sentinel which broke when an
    /// accumulated value happened to be zero between contributions.
    #[test]
    fn cancelling_contributions_do_not_double_count() {
        let mut w = WelfordWorkerState::new(&[1]);
        // bin 0: +5, then -5 (sum = 0), then +3 (sum = 3). The sparse
        // scratch must hold exactly one entry for bin 0, regardless of
        // whether intermediate sums cancel to zero.
        w.add_contribution(0, 0, 5.0);
        w.add_contribution(0, 0, -5.0);
        w.add_contribution(0, 0, 3.0);
        assert_eq!(
            w.tallies[0].scratch_map.len(),
            1,
            "cancelling contributions caused duplicate scratch_map entry"
        );
        w.finish_history();
        // n_touched should be 1 (one history saw this bin), not 2.
        let stats = w.finalize();
        assert_eq!(stats.per_tally[0].n_histories, 1);
        // Mean = 3.0 / 1 = 3.0.
        assert!((stats.per_tally[0].mean[0] - 3.0).abs() < 1e-12);
    }

    /// Real two-pass variance guard at the estimator level: push known
    /// per-history samples (with one sparse bin so the lazy zero-fold
    /// runs) through `finalize`, then check the finalized `mean` and
    /// `std_err()` against a genuinely independent two-pass computation
    /// over the full per-bin sample vectors (pass 1: mean = Σx/N;
    /// pass 2: M2 = Σ(x−mean)²). This does not reuse the online `m2`,
    /// so it actually validates the estimator rather than confirming
    /// floating-point associativity.
    #[test]
    fn finalize_std_err_matches_two_pass() {
        // bin 0 touched every history; bin 1 sparse (untouched in
        // histories 1 and 3 -> implicit zeros).
        let histories: [&[(usize, f64)]; 5] = [
            &[(0, 2.0), (1, 7.0)],
            &[(0, 4.0)],
            &[(0, 1.0), (1, 3.0)],
            &[(0, 3.0)],
            &[(0, 5.0), (1, 5.0)],
        ];
        let n = histories.len();
        let num_bins = 2;

        // Full per-bin sample vectors (one per history; zero where untouched).
        let mut samples = vec![vec![0.0_f64; n]; num_bins];
        for (h, contribs) in histories.iter().enumerate() {
            for &(bin, v) in contribs.iter() {
                samples[bin][h] += v;
            }
        }

        let mut w = WelfordWorkerState::new(&[num_bins]);
        for contribs in histories.iter() {
            for &(bin, v) in contribs.iter() {
                w.add_contribution(0, bin, v);
            }
            w.finish_history();
        }
        let t = &w.finalize().per_tally[0];
        let std_err = t.std_err();

        let nf = n as f64;
        for bin in 0..num_bins {
            let s = &samples[bin];
            let mean_ref = s.iter().sum::<f64>() / nf;
            let m2_ref: f64 = s.iter().map(|&x| (x - mean_ref) * (x - mean_ref)).sum();
            let std_err_ref = (m2_ref / ((nf - 1.0) * nf)).sqrt();
            assert!(
                (t.mean[bin] - mean_ref).abs() < 1e-12,
                "bin {bin} mean: {} vs two-pass {mean_ref}",
                t.mean[bin]
            );
            assert!(
                (std_err[bin] - std_err_ref).abs()
                    < 1e-12 * std_err_ref.abs().max(std_err[bin].abs()).max(1e-18),
                "bin {bin} std_err: {} vs two-pass {std_err_ref}",
                std_err[bin]
            );
        }
    }

    /// Two workers, each processes half the histories, then combined
    /// and finalized. Result should match a single-worker run of the
    /// same data.
    #[test]
    fn two_worker_combine_matches_single_worker() {
        let make = |histories: &[(usize, f64)]| {
            let mut w = WelfordWorkerState::new(&[1]);
            for &(_, x) in histories {
                w.add_contribution(0, 0, x);
                w.finish_history();
            }
            w
        };
        // Single-worker reference
        let single = make(&[(0, 1.0), (0, 2.0), (0, 3.0), (0, 4.0), (0, 5.0)]).finalize();
        // Two workers: split 3+2
        let a = make(&[(0, 1.0), (0, 2.0), (0, 3.0)]);
        let b = make(&[(0, 4.0), (0, 5.0)]);
        let combined = a.combine(b).finalize();

        let ts = &single.per_tally[0];
        let tc = &combined.per_tally[0];
        assert_eq!(tc.n_histories, ts.n_histories);
        assert!((tc.mean[0] - ts.mean[0]).abs() < 1e-12);
        assert!((tc.m2[0] - ts.m2[0]).abs() < 1e-12);
    }

    /// Higher-order Welford moments match a direct two-pass computation.
    #[test]
    fn agg_moments_match_two_pass() {
        let xs = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0];
        let mut a = AggMoments::ZERO;
        for &x in &xs {
            a.update(x);
        }
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let m2: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum();
        let m3: f64 = xs.iter().map(|&x| (x - mean).powi(3)).sum();
        let m4: f64 = xs.iter().map(|&x| (x - mean).powi(4)).sum();
        assert_eq!(a.n, 7);
        assert!((a.mean - mean).abs() < 1e-9);
        assert!((a.m2 - m2).abs() < 1e-6 * m2.abs().max(1.0));
        assert!((a.m3 - m3).abs() < 1e-6 * m3.abs().max(1.0));
        assert!((a.m4 - m4).abs() < 1e-6 * m4.abs().max(1.0));
    }

    /// Parallel combine of two `AggMoments` matches a single accumulation.
    #[test]
    fn agg_combine_matches_single() {
        let xs = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        let mut single = AggMoments::ZERO;
        for &x in &xs {
            single.update(x);
        }
        let mut a = AggMoments::ZERO;
        for &x in &xs[..3] {
            a.update(x);
        }
        let mut b = AggMoments::ZERO;
        for &x in &xs[3..] {
            b.update(x);
        }
        a.combine(&b);
        assert_eq!(a.n, single.n);
        assert!((a.mean - single.mean).abs() < 1e-9);
        assert!((a.m2 - single.m2).abs() < 1e-6 * single.m2.abs());
        assert!((a.m3 - single.m3).abs() < 1e-6 * single.m3.abs().max(1.0));
        assert!((a.m4 - single.m4).abs() < 1e-6 * single.m4.abs());
    }

    /// The empirical-PDF histogram records magnitudes and zeros, and two
    /// histograms with the same layout sum on combine.
    #[test]
    fn score_pdf_records_and_combines() {
        let mut p = ScorePdf::new();
        p.record(1.0);
        p.record(10.0);
        p.record(0.0);
        assert_eq!(p.zero, 1);
        assert_eq!(p.counts.iter().sum::<u64>(), 2);
        // 1.0 → log10 0 → bin (0 - (-30))*4 = 120; 10.0 → bin 124.
        assert_eq!(p.counts[120], 1);
        assert_eq!(p.counts[124], 1);
        let mut q = ScorePdf::new();
        q.record(1.0);
        p.combine(&q);
        assert_eq!(p.counts.iter().sum::<u64>(), 3);
        assert_eq!(p.counts[120], 2);
    }

    /// A light/bounded tail (clustered large scores) yields a high slope.
    #[test]
    fn tail_slope_light_tail_is_high() {
        let mut p = ScorePdf::new();
        for _ in 0..200 {
            p.record(5.0);
        }
        assert!(p.tail_slope() >= 9.0, "got {}", p.tail_slope());
    }

    /// A heavy (power-law) tail yields a low slope (< 3, variance ill-defined).
    #[test]
    fn tail_slope_heavy_tail_is_low() {
        let mut p = ScorePdf::new();
        for i in 0..30 {
            p.record(2f64.powi(i));
        }
        let s = p.tail_slope();
        assert!(s > 0.0 && s < 3.0, "got {s}");
    }
}
