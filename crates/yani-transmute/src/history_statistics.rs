//! Per-history covariance of the transmutation tally's flux moments.
//!
//! [`TransmutationTallies`](crate::TransmutationTallies) accumulates raw sums
//! and nothing else, so a reaction rate comes out of transport with no
//! statistical uncertainty on it. A variance per rate would not be enough even
//! if it existed: parent and daughter rates are estimated from the same
//! histories and move together, and it is that correlation that dominates the
//! inventory variance. The object needed is a covariance.
//!
//! # What the covariance is on
//!
//! Not on the rates. Every rate the tally reports is a linear function of the
//! flux moments, so the covariance goes on the moment vector
//!
//! ```text
//! s = (s0_0 .. s0_{B-1}, s1_0 .. s1_{B-1}, y_0 .. y_{Y-1})
//! ```
//!
//! and any rate's variance follows as `a^T Sigma_s a` for the fold weights `a`
//! of that rate. That scales with the grid rather than with the nuclide count,
//! and covers nuclides the tally never scored.
//!
//! The moments here are taken on the fixed base spectrum grid, not on the full
//! union grid the means use. The union grid grows with every branch-curve
//! breakpoint and its square is the memory cost, so the covariance is coarsened
//! to the base grid while the means stay exactly where they are. Each union bin
//! lies inside one base bin (the base edges are part of the union grid), and
//! `s1` is taken relative to the base bin's own lower edge, so the coarse
//! moments are exact sums of the fine ones rather than an approximation of
//! them. What coarsening gives up is resolution in the covariance, not
//! accuracy in any mean.
//!
//! # Why raw product sums, not Welford co-moments
//!
//! A history touches a few hundred of the ~1300 entries of `s`, and the update
//! has to be sparse to be affordable. A Welford co-moment update cannot be:
//! every entry the history did NOT touch still deviates from its running mean,
//! so each history would cost the full `dim^2`. Raw sums
//! `S_i = sum_h x_hi` and `P_ij = sum_h x_hi x_hj` only change where the history
//! scored, the untouched histories contribute exact zeros with no fold, and the
//! sums add across threads and ranks. The covariance is recovered once, at
//! extraction, as `(P_ij - S_i S_j / n) / (n - 1)`.
//!
//! The cost of that form is cancellation when a bin's per-history score barely
//! varies, which in Monte Carlo transport it does not: a history's score in any
//! one bin ranges from zero to many mean free paths. This is the estimator
//! OpenMC uses for every tally.
//!
//! # Threads
//!
//! Each rayon worker owns a scratch vector for the history it is running.
//! [`HistoryStatistics::finish_history`] folds it into the shared sums, which
//! are split into row stripes, each behind its own lock, so concurrent workers
//! fold into different stripes rather than queueing on one. Each worker starts
//! at a different stripe for the same reason.
//!
//! None of this exists unless it is asked for. The default path allocates
//! nothing, and the raw sums the means are built from are not touched, so the
//! means are bit-identical either way.

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Row stripes each co-moment block is split into, so workers folding
/// histories at the same time mostly take different locks.
const STRIPES: usize = 64;

/// One contiguous run of rows of a material's co-moment block.
struct Stripe {
    /// `S_i` for the stripe's rows.
    sum: Vec<f64>,
    /// `P_ij` for `j >= i`, row by row, each row starting at its diagonal.
    prod: Vec<f64>,
}

/// One material's raw sums, split into stripes.
struct Block {
    dim: usize,
    rows_per_stripe: usize,
    stripes: Box<[Mutex<Stripe>]>,
}

/// Offset of row `r`'s diagonal in a packed upper triangle of size `dim`.
#[inline]
fn row_start(r: usize, dim: usize) -> usize {
    r * dim - r * r.saturating_sub(1) / 2
}

impl Block {
    fn new(dim: usize) -> Self {
        let rows_per_stripe = dim.div_ceil(STRIPES).max(1);
        let stripes = (0..dim.div_ceil(rows_per_stripe))
            .map(|s| {
                let row_lo = s * rows_per_stripe;
                let row_hi = (row_lo + rows_per_stripe).min(dim);
                let len = row_start(row_hi, dim) - row_start(row_lo, dim);
                Mutex::new(Stripe {
                    sum: vec![0.0; row_hi - row_lo],
                    prod: vec![0.0; len],
                })
            })
            .collect();
        Block {
            dim,
            rows_per_stripe,
            stripes,
        }
    }

    /// Fold one history's sparse score vector, sorted by index, into the sums.
    fn fold(&self, entries: &[(u32, f64)], first_stripe: usize) {
        let n = self.stripes.len();
        for k in 0..n {
            let s = (first_stripe + k) % n;
            let lo = s * self.rows_per_stripe;
            let hi = (lo + self.rows_per_stripe).min(self.dim);
            let a0 = entries.partition_point(|e| (e.0 as usize) < lo);
            let a1 = entries.partition_point(|e| (e.0 as usize) < hi);
            if a0 == a1 {
                continue;
            }
            let mut stripe = self.stripes[s].lock().unwrap_or_else(|p| p.into_inner());
            let base = row_start(lo, self.dim);
            for a in a0..a1 {
                let (i, vi) = (entries[a].0 as usize, entries[a].1);
                stripe.sum[i - lo] += vi;
                let row = row_start(i, self.dim) - base;
                for &(j, vj) in &entries[a..] {
                    stripe.prod[row + (j as usize - i)] += vi * vj;
                }
            }
        }
    }

    fn reset(&self) {
        for stripe in self.stripes.iter() {
            let mut st = stripe.lock().unwrap_or_else(|p| p.into_inner());
            st.sum.fill(0.0);
            st.prod.fill(0.0);
        }
    }

    /// `S` in row order and `P` as one packed upper triangle.
    fn gather(&self) -> (Vec<f64>, Vec<f64>) {
        let mut sum = Vec::with_capacity(self.dim);
        let mut prod = Vec::with_capacity(row_start(self.dim, self.dim));
        for stripe in self.stripes.iter() {
            let st = stripe.lock().unwrap_or_else(|p| p.into_inner());
            sum.extend_from_slice(&st.sum);
            prod.extend_from_slice(&st.prod);
        }
        (sum, prod)
    }

    fn pack(&self, out: &mut Vec<f64>) {
        let (sum, prod) = self.gather();
        out.extend_from_slice(&sum);
        out.extend_from_slice(&prod);
    }

    /// Overwrite the sums from `data[off..]`, returning the offset past them.
    fn unpack(&self, data: &[f64], mut off: usize) -> usize {
        for stripe in self.stripes.iter() {
            let mut st = stripe.lock().unwrap_or_else(|p| p.into_inner());
            let n = st.sum.len();
            st.sum.copy_from_slice(&data[off..off + n]);
            off += n;
        }
        for stripe in self.stripes.iter() {
            let mut st = stripe.lock().unwrap_or_else(|p| p.into_inner());
            let n = st.prod.len();
            st.prod.copy_from_slice(&data[off..off + n]);
            off += n;
        }
        off
    }
}

/// One worker's score vector for one material, for the history in progress.
struct Scratch {
    values: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<u32>,
    /// Reused buffer for the sorted `(index, value)` list handed to the fold.
    entries: Vec<(u32, f64)>,
}

impl Scratch {
    fn new(dim: usize) -> Self {
        Scratch {
            values: vec![0.0; dim],
            seen: vec![false; dim],
            touched: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[inline]
    fn add(&mut self, i: usize, v: f64) {
        if v == 0.0 {
            return;
        }
        if !self.seen[i] {
            self.seen[i] = true;
            self.touched.push(i as u32);
        }
        self.values[i] += v;
    }

    /// Move the history's vector into `entries`, sorted, and clear the scratch.
    fn drain(&mut self) {
        self.touched.sort_unstable();
        self.entries.clear();
        for &i in &self.touched {
            let i_us = i as usize;
            self.entries.push((i, self.values[i_us]));
            self.values[i_us] = 0.0;
            self.seen[i_us] = false;
        }
        self.touched.clear();
    }
}

/// The rayon worker this call is running on. Transport runs one history
/// entirely on one worker, which is what makes the worker's scratch that
/// history's scratch; `model.rs` picks its Welford slot the same way.
#[inline]
fn worker_index() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_thread_index().unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

/// Per-history co-moment accumulation for every transmutable material of one
/// tally. Built only by
/// [`TransmutationTallies::with_history_statistics`](crate::TransmutationTallies::with_history_statistics).
pub(crate) struct HistoryStatistics {
    /// Base-grid edges [eV]; bin `c` is `[grid[c], grid[c+1])`, the last runs
    /// to infinity.
    grid: Vec<f64>,
    /// Union-grid bin -> the base-grid bin containing it.
    base_bin: Vec<u32>,
    /// Material id -> its position in `blocks` and in each worker's scratch.
    slot_of: std::collections::HashMap<u32, usize>,
    blocks: Vec<Block>,
    /// One scratch set per rayon worker, sized by
    /// [`HistoryStatistics::prepare_workers`] before transport starts.
    workers: OnceLock<Box<[Mutex<Vec<Scratch>>]>>,
}

/// A worker's scratch for one material, held for the length of one `score`.
pub(crate) struct History<'a> {
    stats: &'a HistoryStatistics,
    scratch: MutexGuard<'a, Vec<Scratch>>,
    slot: usize,
}

impl History<'_> {
    /// A segment of weighted length `tl` at `energy`, landing in union-grid
    /// bin `union_bin`.
    #[inline]
    pub(crate) fn add_moments(&mut self, union_bin: usize, energy: f64, tl: f64) {
        let c = self.stats.base_bin[union_bin] as usize;
        let n_bins = self.stats.grid.len();
        let edge = self.stats.grid[c];
        let s = &mut self.scratch[self.slot];
        s.add(c, tl);
        s.add(n_bins + c, (energy - edge) * tl);
    }

    /// A contribution to MF=9 yield channel `k`.
    #[inline]
    pub(crate) fn add_yield(&mut self, k: usize, value: f64) {
        let n_bins = self.stats.grid.len();
        self.scratch[self.slot].add(2 * n_bins + k, value);
    }
}

impl HistoryStatistics {
    /// `materials` is `(material id, number of MF=9 yield channels)`.
    pub(crate) fn new(base_grid: Vec<f64>, union_grid: &[f64], materials: &[(u32, usize)]) -> Self {
        let base_bin = union_grid
            .iter()
            .map(|&e| (base_grid.partition_point(|&b| b <= e).max(1) - 1) as u32)
            .collect();
        let n_bins = base_grid.len();
        let mut slot_of = std::collections::HashMap::new();
        let mut blocks = Vec::with_capacity(materials.len());
        for (slot, &(id, n_yields)) in materials.iter().enumerate() {
            slot_of.insert(id, slot);
            blocks.push(Block::new(2 * n_bins + n_yields));
        }
        HistoryStatistics {
            grid: base_grid,
            base_bin,
            slot_of,
            blocks,
            workers: OnceLock::new(),
        }
    }

    /// Allocate one scratch set per worker. Must run single-threaded before
    /// any scoring; a later call for no more workers than the first is a
    /// no-op, so a tally reused across transmutation steps keeps its scratch.
    pub(crate) fn prepare_workers(&self, n_workers: usize) -> Result<(), String> {
        let n_workers = n_workers.max(1);
        let slots = self.workers.get_or_init(|| {
            (0..n_workers)
                .map(|_| Mutex::new(self.blocks.iter().map(|b| Scratch::new(b.dim)).collect()))
                .collect()
        });
        if slots.len() < n_workers {
            return Err(format!(
                "transmutation history statistics were sized for {} worker threads \
                 and this run uses {n_workers}; build a new tally for a run with \
                 more threads",
                slots.len()
            ));
        }
        Ok(())
    }

    fn worker_scratch(&self) -> MutexGuard<'_, Vec<Scratch>> {
        let slots = self.workers.get().expect(
            "TransmutationTallies::prepare_history_workers must run before scoring \
             with history statistics on",
        );
        let idx = worker_index();
        let Some(slot) = slots.get(idx) else {
            panic!(
                "transmutation history statistics were sized for {} worker threads \
                 and were scored from worker {idx}",
                slots.len()
            );
        };
        slot.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// This worker's scratch for `material_id`, or `None` for a material the
    /// tally does not carry.
    #[inline]
    pub(crate) fn history(&self, material_id: u32) -> Option<History<'_>> {
        let slot = *self.slot_of.get(&material_id)?;
        Some(History {
            stats: self,
            scratch: self.worker_scratch(),
            slot,
        })
    }

    /// Close the history running on this worker: fold its score vector into
    /// the shared sums and clear the scratch for the next one.
    pub(crate) fn finish_history(&self) {
        let first_stripe = worker_index();
        let mut scratch = self.worker_scratch();
        for (slot, s) in scratch.iter_mut().enumerate() {
            if s.touched.is_empty() {
                continue;
            }
            s.drain();
            self.blocks[slot].fold(&s.entries, first_stripe);
        }
    }

    pub(crate) fn reset(&self) {
        for block in &self.blocks {
            block.reset();
        }
        if let Some(workers) = self.workers.get() {
            for w in workers.iter() {
                let mut scratch = w.lock().unwrap_or_else(|p| p.into_inner());
                for s in scratch.iter_mut() {
                    s.drain();
                }
            }
        }
    }

    pub(crate) fn pack(&self, material_id: u32, out: &mut Vec<f64>) {
        if let Some(&slot) = self.slot_of.get(&material_id) {
            self.blocks[slot].pack(out);
        }
    }

    pub(crate) fn unpack(&self, material_id: u32, data: &[f64], off: usize) -> usize {
        match self.slot_of.get(&material_id) {
            Some(&slot) => self.blocks[slot].unpack(data, off),
            None => off,
        }
    }

    /// The moment statistics of `material_id` over `n_histories` histories.
    pub(crate) fn extract(
        &self,
        material_id: u32,
        n_histories: u64,
        yield_channels: Vec<YieldChannelLabel>,
    ) -> Option<MomentCovariance> {
        let block = &self.blocks[*self.slot_of.get(&material_id)?];
        let (sum, prod) = block.gather();
        let dim = block.dim;
        let n = n_histories as f64;
        let mean: Vec<f64> = if n_histories == 0 {
            vec![0.0; dim]
        } else {
            sum.iter().map(|s| s / n).collect()
        };
        let mut covariance = vec![0.0; prod.len()];
        if n_histories >= 2 {
            for i in 0..dim {
                let row = row_start(i, dim);
                for j in i..dim {
                    let k = row + (j - i);
                    covariance[k] = (prod[k] - sum[i] * sum[j] / n) / (n - 1.0);
                }
            }
        }
        Some(MomentCovariance {
            grid: self.grid.clone(),
            yield_channels,
            n_histories,
            mean,
            covariance,
        })
    }
}

/// An MF=9 yield channel's place in the moment vector: `(parent, reaction
/// kind, final-state target)`.
pub type YieldChannelLabel = (String, String, String);

/// The per-history mean and covariance of one material's flux-moment vector.
///
/// The vector is laid out as `(s0_c, s1_c, y_k)`: the track length in base-grid
/// bin `c`, the first moment `sum((E - grid[c]) * TL)` in that bin, and the
/// MF=9 yield channels in the order of
/// [`yield_channels`](Self::yield_channels). Everything is per source particle
/// and unnormalized, in the units the tally scores (cm, eV cm, b cm), so a
/// quantity folded from it converts to a rate by the same
/// `source_rate / (volume * 1e24)` the rates use, and its variance by that
/// factor squared.
#[derive(Debug, Clone)]
pub struct MomentCovariance {
    /// Base-grid edges [eV]. Bin `c` is `[grid[c], grid[c+1])`; the last bin
    /// runs to infinity.
    pub grid: Vec<f64>,
    /// The MF=9 yield channels at the end of the vector, in order.
    pub yield_channels: Vec<YieldChannelLabel>,
    /// Source histories the statistics cover, histories that scored nothing
    /// in this material included.
    pub n_histories: u64,
    /// Per-history mean of each entry, i.e. its per-source-particle tally.
    pub mean: Vec<f64>,
    /// Sample covariance of the per-history vector, packed upper triangle.
    covariance: Vec<f64>,
}

impl MomentCovariance {
    /// Length of the moment vector.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Base-grid bins, `B`.
    pub fn n_bins(&self) -> usize {
        self.grid.len()
    }

    /// Index of `s0` for bin `c`.
    pub fn s0_index(&self, c: usize) -> usize {
        c
    }

    /// Index of `s1` for bin `c`.
    pub fn s1_index(&self, c: usize) -> usize {
        self.n_bins() + c
    }

    /// Index of MF=9 yield channel `k`.
    pub fn yield_index(&self, k: usize) -> usize {
        2 * self.n_bins() + k
    }

    /// Sample covariance of entries `i` and `j` of a single history's vector.
    /// Zero with fewer than two histories.
    pub fn covariance(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        self.covariance[row_start(i, self.dim()) + (j - i)]
    }

    /// Covariance of the estimated means of entries `i` and `j`, i.e. of what
    /// the tally reports: the per-history covariance over `n_histories`.
    pub fn covariance_of_mean(&self, i: usize, j: usize) -> f64 {
        if self.n_histories == 0 {
            return 0.0;
        }
        self.covariance(i, j) / self.n_histories as f64
    }

    /// Standard error of the estimated mean of entry `i`.
    pub fn std_err(&self, i: usize) -> f64 {
        self.covariance_of_mean(i, i).max(0.0).sqrt()
    }

    /// Mean and variance of the estimate of `weights . s`.
    ///
    /// Any quantity folded linearly from the moments, a reaction rate among
    /// them, has its statistical variance given by this quadratic form, with
    /// the cross terms that make correlated rates correlated.
    pub fn linear_combination(&self, weights: &[f64]) -> (f64, f64) {
        assert_eq!(
            weights.len(),
            self.dim(),
            "weights must have one entry per moment"
        );
        let mean = weights.iter().zip(&self.mean).map(|(w, m)| w * m).sum();
        let dim = self.dim();
        let mut var = 0.0;
        for i in 0..dim {
            let wi = weights[i];
            if wi == 0.0 {
                continue;
            }
            let row = row_start(i, dim);
            var += wi * wi * self.covariance[row];
            for (k, &wj) in weights[i + 1..].iter().enumerate() {
                if wj != 0.0 {
                    var += 2.0 * wi * wj * self.covariance[row + 1 + k];
                }
            }
        }
        let var = if self.n_histories == 0 {
            0.0
        } else {
            var / self.n_histories as f64
        };
        (mean, var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_rows_are_contiguous() {
        let dim = 7;
        let mut expected = 0;
        for r in 0..dim {
            assert_eq!(row_start(r, dim), expected);
            expected += dim - r;
        }
        assert_eq!(row_start(dim, dim), dim * (dim + 1) / 2);
    }

    /// The stripes must tile the triangle exactly, whatever the dimension,
    /// including dimensions below and not divisible by the stripe count.
    #[test]
    fn stripes_tile_the_triangle() {
        for dim in [1, 2, 5, 63, 64, 65, 130, 1257] {
            let block = Block::new(dim);
            let (sum, prod) = block.gather();
            assert_eq!(sum.len(), dim, "dim {dim}");
            assert_eq!(prod.len(), dim * (dim + 1) / 2, "dim {dim}");
        }
    }

    /// Folding sparse histories must give the same sums as the dense outer
    /// product, for every stripe the history crosses and whichever stripe the
    /// fold starts at.
    #[test]
    fn sparse_fold_matches_dense_outer_product() {
        let dim = 150;
        let histories: Vec<Vec<(u32, f64)>> = vec![
            vec![(0, 1.5), (3, -2.0), (70, 0.25), (149, 4.0)],
            vec![(2, 3.0)],
            vec![(1, 1.0), (2, 2.0), (3, 3.0), (100, 5.0), (148, -1.0)],
        ];
        let block = Block::new(dim);
        let mut dense_s = vec![0.0; dim];
        let mut dense_p = vec![0.0; dim * dim];
        for (h, entries) in histories.iter().enumerate() {
            block.fold(entries, h * 17);
            for &(i, vi) in entries {
                dense_s[i as usize] += vi;
                for &(j, vj) in entries {
                    dense_p[i as usize * dim + j as usize] += vi * vj;
                }
            }
        }
        let (sum, prod) = block.gather();
        assert_eq!(sum, dense_s);
        for i in 0..dim {
            for j in i..dim {
                assert_eq!(
                    prod[row_start(i, dim) + (j - i)],
                    dense_p[i * dim + j],
                    "P[{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn pack_unpack_round_trips() {
        let block = Block::new(80);
        block.fold(&[(1, 2.0), (40, 3.0), (79, 5.0)], 3);
        let mut packed = Vec::new();
        block.pack(&mut packed);
        let other = Block::new(80);
        assert_eq!(other.unpack(&packed, 0), packed.len());
        assert_eq!(other.gather(), block.gather());
    }
}
