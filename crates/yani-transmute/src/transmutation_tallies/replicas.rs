//! Per-replica transmutation tallies, scored under the transport's
//! nuclear-data replica weights.
//!
//! A transport run with correlated replica weights (`yamc`'s
//! `transport::replica`) carries, per history, one likelihood ratio per
//! replica: the ratio of the probability of its path under replica `k`'s
//! cross sections to the nominal one. A track-length score of replica `k` is
//! the nominal score times that ratio integrated along the segment,
//! `r_k φ(ΔΣ_k d)`, with the score's own cross section taken as replica `k`'s.
//! This module keeps those per-replica sums for everything the transmutation
//! reads from its tally, so replica `k`'s reaction rates are the rates
//! transport would have tallied under replica `k`'s data: the flux they are
//! folded with answers the replica's cross sections (a stronger resonance
//! deepens its own self-shielding dip, a perturbed scatterer moves the
//! spectrum), which a fold of perturbed cross sections against the nominal
//! flux cannot see.
//!
//! # What is kept per replica
//!
//! Per material, one row of `R` sums (one per replica, contiguous) for each
//! of:
//!
//! - the total track length (the scalar flux);
//! - every scored `sum(sigma * TL)` of a nuclide and MT, with `sigma` the
//!   replica's cross section, `sigma + Δsigma_k`, from
//!   [`ReplicaChanges::add_changes`];
//! - every state of a branching list scored at the collision energy, its
//!   production evaluated against the replica's total;
//! - every fission-yield interpolation weight, against the replica's fission
//!   cross section;
//! - with [`TransmutationTallies::with_isomer_intervals`], every MF=40 state's
//!   production on each of its covariance intervals, the replica's production
//!   of the state binned as the nominal one is;
//! - the zeroth flux moment in each union-grid bin, and the first moment too
//!   where the tally folds `(n,n')` partial curves from them.
//!
//! The nominal sums are untouched and are scored exactly as without replicas,
//! so the nominal rates are bit-identical with this on or off.
//!
//! # Two halves, for the Monte Carlo noise
//!
//! The sums are kept twice, once over the even-numbered source histories and
//! once over the odd ones (by global history index, so the split does not
//! depend on threads, chunks or ranks), with the histories each half saw.
//! The full tally is their sum. Each replica's difference from the nominal
//! carries Monte Carlo noise that correlated sampling reduces but does not
//! remove, and the two halves' noise is independent, so a transmutation
//! solved once more from the even half alone measures it: see
//! [`ReplicaTallied::even`].
//!
//! # Adding a quantity
//!
//! A new nominal accumulator gets its per-replica copy by adding a segment to
//! [`Layout`], scoring it in [`ReplicaSums::score`] beside the nominal score
//! with the segment's weights `w_k = TL r_k φ_k` (and the replica's cross
//! section where the score carries one), and reading it back in
//! [`TransmutationTallies::replica_tallied`]. Rows are replica-contiguous, so
//! a score is one fused multiply-add over `R` per row it touches.
//!
//! # Cost
//!
//! Each worker thread keeps a dense copy of every material's sums, folded
//! into the shared ones at chunk boundaries: `2 x (1 + rates + yields +
//! fission weights + moments) x R` doubles per material per worker, the 2
//! for the two halves. A segment is scored into its history's half only. Scoring costs `R`
//! multiply-adds per scored row per track segment, plus the cross-section
//! changes, which [`ReplicaChanges`] reads once per nuclide per segment.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use super::{
    fission_yield_weights_of_sums, rates_of_sums, MaterialTransmutationData, TransmutationTallies,
};
use crate::branching_rule::Bound;
use yamc_materials::material::Material;
use yani::{fission_yield_interp_weights, FissionYieldWeights, ReactionRates};

/// Each replica's change to the cross sections a transmutation tally scores,
/// from the transport that carries the replica weights.
///
/// The transport is what draws the replicas, so it is what knows them; the
/// tally asks it once per nuclide per track segment.
pub trait ReplicaChanges: Sync {
    /// Add, for nuclide `nuclide` of material `material_id` (its position in
    /// the tally's [`TransmutationTallies::layout`]) at `energy` [eV], every
    /// replica's change `Δsigma_k` [barns] to each of the tally's MTs for the
    /// material, into `out`, laid out MT-major: `out[m * R + k]` for MT
    /// `mts[m]` and replica `k`. Return whether anything was added; `out`
    /// arrives zeroed.
    fn add_changes(&self, material_id: u32, nuclide: usize, energy: f64, out: &mut [f64]) -> bool;
}

/// One material's transmutation tally as replica `k` of the transport's
/// data saw it, per source particle, for every replica: the per-replica
/// counterpart of the nominal getters of [`TransmutationTallies`], at the
/// same normalization (unit source rate, per unit volume).
#[derive(Debug, Clone, Default)]
pub struct ReplicaTallied {
    /// The seed the transport drew the replicas from.
    pub seed: u64,
    /// Per replica, the reaction rates per atom [1/s at unit source rate],
    /// as [`TransmutationTallies::get_reaction_rates`] gives the nominal.
    pub rates: Vec<ReactionRates>,
    /// Per replica, the isomeric partials, as
    /// [`TransmutationTallies::get_partial_rates`] gives the nominal.
    pub partials: Vec<super::PartialRates>,
    /// Per replica, the fission-yield spectrum weights, as
    /// [`TransmutationTallies::get_fission_yield_weights`] gives the nominal.
    pub fy_weights: Vec<FissionYieldWeights>,
    /// Per replica, each MF=40 state's production on its covariance
    /// intervals, as [`TransmutationTallies::get_isomer_productions`] gives
    /// the nominal; empty when the tally did not score them.
    pub isomer: Vec<crate::IsomerIntervalProductions>,
    /// Per replica, the flux on the union grid, as
    /// [`TransmutationTallies::flux_spectrum`] gives the nominal: what a fold
    /// of a quantity the transport does not perturb (an isomer production
    /// curve, say) against replica `k`'s own spectrum reads.
    pub spectra: Vec<crate::MultigroupSpectrum>,
    /// How the transport took each nuclide's resolved resonance range: its
    /// parameters sampled per replica, or the first-order MF=32 rows, and
    /// why. Filled by the transport driver; empty from the tally alone.
    pub resonance_parameters:
        std::collections::BTreeMap<String, crate::resonance_rates::ResonanceMethod>,
    /// The same tally over the even-numbered source histories alone, at the
    /// same normalization (per source particle of that half), or `None` when
    /// either half saw no history or the transport did not mark its histories
    /// ([`TransmutationTallies::begin_history`]).
    ///
    /// Replica `k`'s full tally is `x_k = (n_e x_k^e + n_o x_k^o) / n`, and
    /// its Monte Carlo noise about what transport would give under its data
    /// is the two halves' noise, independent of each other. Over the
    /// replicas, the spread of `x_k^e - x_k` then has expectation
    /// `(n_o / n_e)` times the full tally's replica-to-replica noise, while
    /// every data draw, and the nominal's own error, cancels from it. A
    /// transmutation solved from each replica's even half beside its full
    /// tally carries that through the Bateman solve, nonlinearity and all.
    pub even: Option<Box<ReplicaTallied>>,
    /// Source histories in the even and in the odd half.
    pub halves: [u64; 2],
}

impl ReplicaTallied {
    /// Replicas carried.
    pub fn replicas(&self) -> usize {
        self.rates.len()
    }

    /// The factor that turns the replica spread of the difference between
    /// an output solved from the even half and from the full tally into the
    /// full tally's Monte Carlo noise variance, `n_e / n_o`, or `None`
    /// without [`Self::even`].
    pub fn noise_scale(&self) -> Option<f64> {
        let [e, o] = self.halves;
        (self.even.is_some() && e > 0 && o > 0).then(|| e as f64 / o as f64)
    }
}

/// Where each quantity's rows start in a material's per-replica sums.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// The first scored rate's row; the flux is row 0.
    rates: usize,
    yields: usize,
    fission: usize,
    /// The MF=40 interval productions, one row per state interval.
    isomer: usize,
    /// The zeroth union-grid moments, one row per bin.
    s0: usize,
    /// The first moments, or `None` when no curve is folded from them.
    s1: Option<usize>,
    /// Rows in all.
    rows: usize,
}

impl Layout {
    fn of(mat: &MaterialTransmutationData, n_bins: usize, first_moments: bool) -> Self {
        let rates = 1;
        let yields = rates + mat.nuclide_names.len() * mat.mt_numbers.len();
        let fission = yields + mat.yield_channels.len();
        let isomer = fission + mat.fy_batch.len();
        let s0 = isomer + mat.isomer.as_ref().map_or(0, |t| t.batch.len());
        let (s1, rows) = if first_moments {
            (Some(s0 + n_bins), s0 + 2 * n_bins)
        } else {
            (None, s0 + n_bins)
        };
        Layout {
            rates,
            yields,
            fission,
            isomer,
            s0,
            s1,
            rows,
        }
    }
}

/// One worker's sums for the chunk in progress, and its scratch.
struct Worker {
    /// Per material slot, the even half's `rows x R`, row-major, then the
    /// odd half's.
    sums: Vec<Vec<f64>>,
    /// Per material slot, whether this chunk scored it.
    touched: Vec<bool>,
    /// The half the history in progress on this worker belongs to.
    half: usize,
    /// Histories begun per half this chunk.
    histories: [u64; 2],
    /// `TL r_k φ_k` of the segment being scored.
    weights: Vec<f64>,
    /// The changes [`ReplicaChanges::add_changes`] writes, `n_mts x R`.
    changes: Vec<f64>,
    /// A branching list's point values.
    values: Vec<f64>,
}

/// The per-replica sums of a [`TransmutationTallies`], when it was built
/// [`with_replicas`](TransmutationTallies::with_replicas).
pub(crate) struct ReplicaSums {
    replicas: usize,
    seed: u64,
    slot_of: HashMap<u32, usize>,
    layouts: Vec<Layout>,
    /// Per material slot, the sums over every finished chunk, the even
    /// half's `rows x R` then the odd half's.
    sums: Vec<Mutex<Vec<f64>>>,
    /// Histories per half over every finished chunk.
    histories: Mutex<[u64; 2]>,
    workers: OnceLock<Box<[Mutex<Worker>]>>,
}

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

/// `row[k] += w[k] * x` over the replicas.
#[inline]
fn add_scaled(row: &mut [f64], w: &[f64], x: f64) {
    for (o, w) in row.iter_mut().zip(w) {
        *o += w * x;
    }
}

impl ReplicaSums {
    pub(crate) fn new(tallies: &TransmutationTallies, replicas: usize, seed: u64) -> Self {
        let mut ids: Vec<u32> = tallies.materials.keys().copied().collect();
        ids.sort_unstable();
        let first_moments = !tallies.moment_curves.is_empty();
        let n_bins = tallies.union_grid.len();
        let layouts: Vec<Layout> = ids
            .iter()
            .map(|id| Layout::of(&tallies.materials[id], n_bins, first_moments))
            .collect();
        ReplicaSums {
            replicas,
            seed,
            slot_of: ids.iter().enumerate().map(|(s, id)| (*id, s)).collect(),
            sums: layouts
                .iter()
                .map(|l| Mutex::new(vec![0.0; 2 * l.rows * replicas]))
                .collect(),
            histories: Mutex::new([0; 2]),
            layouts,
            workers: OnceLock::new(),
        }
    }

    /// Doubles the sums take for `workers` worker threads, the shared copy
    /// included.
    pub(crate) fn doubles(&self, workers: usize) -> usize {
        let per: usize = self
            .layouts
            .iter()
            .map(|l| 2 * l.rows * self.replicas)
            .sum();
        per * (workers + 1)
    }

    pub(crate) fn prepare_workers(&self, n_workers: usize) -> Result<(), String> {
        let n_workers = n_workers.max(1);
        let max_mts = self.layouts.iter().map(|l| l.yields - l.rates).max();
        let slots = self.workers.get_or_init(|| {
            (0..n_workers)
                .map(|_| {
                    Mutex::new(Worker {
                        sums: self
                            .layouts
                            .iter()
                            .map(|l| vec![0.0; 2 * l.rows * self.replicas])
                            .collect(),
                        touched: vec![false; self.layouts.len()],
                        half: 0,
                        histories: [0; 2],
                        weights: vec![0.0; self.replicas],
                        changes: vec![0.0; max_mts.unwrap_or(0) * self.replicas],
                        values: Vec::new(),
                    })
                })
                .collect()
        });
        if slots.len() < n_workers {
            return Err(format!(
                "transmutation replica tallies were sized for {} worker threads and this \
                 run uses {n_workers}; build a new tally for a run with more threads",
                slots.len()
            ));
        }
        Ok(())
    }

    /// Mark the source history with global index `index` as the one now
    /// running on the calling worker: its segments go to its half, by the
    /// index's parity.
    pub(crate) fn begin_history(&self, index: u64) {
        let Some(workers) = self.workers.get() else {
            return;
        };
        let mut w = workers[worker_index()]
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let half = (index & 1) as usize;
        w.half = half;
        w.histories[half] += 1;
    }

    /// Score one track segment of weighted length `track_length` at `energy`
    /// in `material_id` for every replica, `factors[k]` the segment's
    /// `r_k φ(ΔΣ_k d)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score(
        &self,
        tallies: &TransmutationTallies,
        material_id: u32,
        energy: f64,
        track_length: f64,
        material: &Material,
        factors: &[f64],
        changes: &dyn ReplicaChanges,
    ) {
        let (Some(&slot), Some(mat)) = (
            self.slot_of.get(&material_id),
            tallies.materials.get(&material_id),
        ) else {
            return;
        };
        let r = self.replicas;
        let layout = self.layouts[slot];
        let workers = self.workers.get().expect(
            "TransmutationTallies::prepare_workers must run before scoring with replicas on",
        );
        let mut guard = workers[worker_index()]
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let Worker {
            sums,
            touched,
            half,
            weights,
            changes: delta,
            values,
            ..
        } = &mut *guard;
        touched[slot] = true;
        let n = layout.rows * r;
        let buf = &mut sums[slot][*half * n..(*half + 1) * n];
        let row = |i: usize| i * r..(i + 1) * r;
        for (w, f) in weights.iter_mut().zip(factors) {
            *w = track_length * f;
        }
        add_scaled(&mut buf[row(0)], weights, 1.0);

        let grid = &tallies.union_grid;
        if !grid.is_empty() && energy >= grid[0] {
            let g = grid.partition_point(|&e| e <= energy) - 1;
            add_scaled(&mut buf[row(layout.s0 + g)], weights, 1.0);
            if let Some(s1) = layout.s1 {
                add_scaled(&mut buf[row(s1 + g)], weights, energy - grid[g]);
            }
        }

        let temperature = material.temperature();
        let n_mts = mat.mt_numbers.len();
        let delta = &mut delta[..n_mts * r];
        for (nuc_idx, name) in mat.nuclide_names.iter().enumerate() {
            let Some(reactions) = material
                .nuclide_data
                .get(name)
                .and_then(|nd| nd.reactions_for_temp(temperature))
            else {
                continue;
            };
            delta.fill(0.0);
            let moved = changes.add_changes(material_id, nuc_idx, energy, delta);
            // `sigma_k = max(sigma + Δsigma_k, 0)` into the change's slot, so
            // the lists below read each replica's total from it.
            for (mt_idx, &mt) in mat.mt_numbers.iter().enumerate() {
                let sigma = reactions
                    .get(&mt)
                    .and_then(|x| x.cross_section_at(energy))
                    .unwrap_or(0.0);
                let d = &mut delta[mt_idx * r..(mt_idx + 1) * r];
                if sigma <= 0.0 {
                    // Nothing scored nominally, nothing for a replica to move.
                    d.fill(0.0);
                    continue;
                }
                for x in d.iter_mut() {
                    *x = (sigma + *x).max(0.0);
                }
                let out = &mut buf[row(layout.rates + nuc_idx * n_mts + mt_idx)];
                for ((o, w), x) in out.iter_mut().zip(weights.iter()).zip(d.iter()) {
                    *o += w * x;
                }
                if Some(mt_idx) == mat.fy_mt_idx {
                    if let Some(ch) = &mat.fy_channels[nuc_idx] {
                        if let Some(hats) =
                            fission_yield_interp_weights(&ch.energies, &ch.laws, energy)
                        {
                            for (h, hw) in hats {
                                if hw == 0.0 {
                                    continue;
                                }
                                let out = &mut buf[row(layout.fission + ch.offset + h)];
                                for ((o, w), x) in out.iter_mut().zip(weights.iter()).zip(d.iter())
                                {
                                    *o += hw * w * x;
                                }
                            }
                        }
                    }
                }
            }

            // The lists of this nuclide, each state's production as the rule
            // gives it against replica `k`'s total: a share moves with the
            // total, and an absolute partial is clipped and extended against
            // the replica's total as it is against the nominal.
            for list in &mat.lists[nuc_idx] {
                let Some(total) = reactions.get(&list.mt) else {
                    continue;
                };
                let t = total.cross_section_at(energy).unwrap_or(0.0);
                if t <= 0.0 && !list.rule.is_absolute() {
                    continue;
                }
                let bound = Bound {
                    rule: &list.rule,
                    total: Some(total.as_ref()),
                    tail: &list.tails,
                };
                let n = list.rule.curves.len();
                values.resize(n, 0.0);
                let absolute = list.rule.is_absolute();
                let own_mt = mat.mt_numbers.iter().position(|&m| m == list.mt);
                let produce = |values: &mut [f64], t_k: f64| -> f64 {
                    let point = bound.point(energy, t_k, values);
                    if absolute {
                        1.0
                    } else {
                        point.factor
                    }
                };
                match own_mt.filter(|_| moved) {
                    Some(m) => {
                        for k in 0..r {
                            let t_k = delta[m * r + k];
                            let factor = produce(values, t_k);
                            for (c, channel) in list.channels.iter().enumerate() {
                                let Some(ch) = channel else {
                                    continue;
                                };
                                let p = values[c] * factor;
                                if p > 0.0 {
                                    buf[(layout.yields + ch) * r + k] += weights[k] * p;
                                    for iv in list.intervals.iter().filter(|iv| iv.curve == c) {
                                        if let Some((at, share)) =
                                            iv.locate(energy, &list.rule.curves[c])
                                        {
                                            buf[(layout.isomer + at) * r + k] +=
                                                weights[k] * p * share;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        let factor = produce(values, t);
                        for (c, channel) in list.channels.iter().enumerate() {
                            let Some(ch) = channel else {
                                continue;
                            };
                            let p = values[c] * factor;
                            if p > 0.0 {
                                add_scaled(&mut buf[row(layout.yields + ch)], weights, p);
                                for iv in list.intervals.iter().filter(|iv| iv.curve == c) {
                                    if let Some((at, share)) =
                                        iv.locate(energy, &list.rule.curves[c])
                                    {
                                        add_scaled(
                                            &mut buf[row(layout.isomer + at)],
                                            weights,
                                            p * share,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Fold every worker's chunk sums into the shared ones. Single-threaded,
    /// at a chunk boundary.
    pub(crate) fn accumulate(&self) {
        let Some(workers) = self.workers.get() else {
            return;
        };
        let mut histories = self.histories.lock().unwrap_or_else(|p| p.into_inner());
        for w in workers.iter() {
            let mut w = w.lock().unwrap_or_else(|p| p.into_inner());
            let Worker {
                sums,
                touched,
                histories: begun,
                ..
            } = &mut *w;
            for (total, b) in histories.iter_mut().zip(begun.iter_mut()) {
                *total += *b;
                *b = 0;
            }
            for (slot, (local, hit)) in sums.iter_mut().zip(touched.iter_mut()).enumerate() {
                if !*hit {
                    continue;
                }
                let mut shared = self.sums[slot].lock().unwrap_or_else(|p| p.into_inner());
                for (s, l) in shared.iter_mut().zip(local.iter_mut()) {
                    *s += *l;
                    *l = 0.0;
                }
                *hit = false;
            }
        }
    }

    /// Append the shared sums, slot by slot, for a reduction across ranks.
    /// They are sums over each rank's histories of the same replicas' scores
    /// (every rank draws the replicas from the same seed), so they add.
    pub(crate) fn pack(&self, out: &mut Vec<f64>) {
        for s in &self.sums {
            out.extend_from_slice(&s.lock().unwrap_or_else(|p| p.into_inner()));
        }
        // Each rank counts the histories it ran, so the counts add too.
        let histories = self.histories.lock().unwrap_or_else(|p| p.into_inner());
        out.extend(histories.iter().map(|&n| n as f64));
    }

    /// Read back what [`Self::pack`] wrote from `packed` at `off`; the offset
    /// past it.
    pub(crate) fn unpack(&self, packed: &[f64], mut off: usize) -> usize {
        for s in &self.sums {
            let mut s = s.lock().unwrap_or_else(|p| p.into_inner());
            let n = s.len();
            s.copy_from_slice(&packed[off..off + n]);
            off += n;
        }
        let mut histories = self.histories.lock().unwrap_or_else(|p| p.into_inner());
        for h in histories.iter_mut() {
            *h = packed[off].round() as u64;
            off += 1;
        }
        off
    }

    pub(crate) fn reset(&self) {
        for s in &self.sums {
            s.lock().unwrap_or_else(|p| p.into_inner()).fill(0.0);
        }
        *self.histories.lock().unwrap_or_else(|p| p.into_inner()) = [0; 2];
        if let Some(workers) = self.workers.get() {
            for w in workers.iter() {
                let mut w = w.lock().unwrap_or_else(|p| p.into_inner());
                let Worker {
                    sums,
                    touched,
                    histories,
                    ..
                } = &mut *w;
                for (local, hit) in sums.iter_mut().zip(touched.iter_mut()) {
                    local.fill(0.0);
                    *hit = false;
                }
                *histories = [0; 2];
            }
        }
    }
}

impl TransmutationTallies {
    /// Also keep, for each of `replicas` nuclear-data replicas drawn from
    /// `seed`, a copy of every sum the transmutation reads, scored under the
    /// transport's replica weights (see the [module docs](self)).
    ///
    /// Off unless asked for. A transport that scores this tally must pass
    /// each segment's replica factors to
    /// [`score_with_replicas`](Self::score_with_replicas), and must call
    /// [`prepare_workers`](Self::prepare_workers) before it starts.
    pub fn with_replicas(mut self, replicas: usize, seed: u64) -> Self {
        self.replicas = Some(ReplicaSums::new(&self, replicas, seed));
        self
    }

    /// The replicas [`with_replicas`](Self::with_replicas) keeps, or `None`.
    pub fn replicas(&self) -> Option<usize> {
        self.replicas.as_ref().map(|r| r.replicas)
    }

    /// The seed [`with_replicas`](Self::with_replicas) was given, or `None`.
    pub fn replica_seed(&self) -> Option<u64> {
        self.replicas.as_ref().map(|r| r.seed)
    }

    /// Doubles the replica sums take on `workers` threads, or zero without
    /// them, so a driver can refuse a run that would not fit before it starts.
    pub fn replica_doubles(&self, workers: usize) -> usize {
        self.replicas.as_ref().map_or(0, |r| r.doubles(workers))
    }

    /// The material ids this tally scores, ascending.
    pub fn material_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.materials.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// The nuclides and MTs this tally scores for `material_id`, in the
    /// order [`ReplicaChanges::add_changes`] indexes them, or `None` for a
    /// material it does not score.
    pub fn layout(&self, material_id: u32) -> Option<(&[String], &[i32])> {
        let m = self.materials.get(&material_id)?;
        Some((&m.nuclide_names, &m.mt_numbers))
    }

    /// Mark the source history with global index `index` as the one starting
    /// on the calling worker thread, so with replicas on its segments are
    /// kept in its half (even or odd index) of the per-replica sums. A no-op
    /// without replicas. A transport that carries replica weights calls it
    /// once per source history, before scoring any of it; without it the
    /// halves are not formed and [`ReplicaTallied::even`] is `None`.
    #[inline]
    pub fn begin_history(&self, index: u64) {
        if let Some(r) = &self.replicas {
            r.begin_history(index);
        }
    }

    /// [`score`](Self::score), and with replicas on, every replica's score
    /// of the same segment: `factors[k]` is the segment's `r_k φ(ΔΣ_k d)`
    /// (the particle's ratio integrated along the segment, per unit length),
    /// and `changes` gives each replica's cross sections.
    #[inline]
    pub fn score_with_replicas(
        &self,
        material_id: u32,
        energy: f64,
        track_length: f64,
        material: &Material,
        factors: &[f64],
        changes: &dyn ReplicaChanges,
    ) {
        self.score(material_id, energy, track_length, material);
        if let Some(r) = &self.replicas {
            r.score(
                self,
                material_id,
                energy,
                track_length,
                material,
                factors,
                changes,
            );
        }
    }

    /// Every replica's view of `material_id`'s tally, per source particle and
    /// unit volume, or `None` without replicas, for a material the tally
    /// does not score, before any particle was scored, or for a volume that
    /// is not positive. With both halves populated, and every history marked
    /// with [`begin_history`](Self::begin_history), it carries the even
    /// half's view as [`ReplicaTallied::even`].
    pub fn replica_tallied(&self, material_id: u32, volume: f64) -> Option<ReplicaTallied> {
        let sums = self.replicas.as_ref()?;
        let slot = *sums.slot_of.get(&material_id)?;
        let mat = self.materials.get(&material_id)?;
        let total_particles = mat.total_particles.load(Ordering::Relaxed);
        if total_particles == 0 || volume <= 0.0 {
            return None;
        }
        let layout = sums.layouts[slot];
        let halves = *sums.histories.lock().unwrap_or_else(|p| p.into_inner());
        let shared = sums.sums[slot].lock().unwrap_or_else(|p| p.into_inner());
        let n = layout.rows * sums.replicas;
        let (even, odd) = shared.split_at(n);
        let full: Vec<f64> = even.iter().zip(odd).map(|(a, b)| a + b).collect();
        let mut out = self.tallied_of(sums, mat, layout, &full, total_particles as f64, volume);
        out.halves = halves;
        // The halves are usable only when they account for every history
        // the tally normalizes by: a transport that did not mark its
        // histories leaves them all in the even half, uncounted.
        if halves[0] > 0 && halves[1] > 0 && halves[0] + halves[1] == total_particles as u64 {
            out.even = Some(Box::new(self.tallied_of(
                sums,
                mat,
                layout,
                even,
                halves[0] as f64,
                volume,
            )));
        }
        Some(out)
    }

    /// One set of replica sums, `rows x R`, over `histories` source
    /// histories, as [`ReplicaTallied`].
    fn tallied_of(
        &self,
        sums: &ReplicaSums,
        mat: &MaterialTransmutationData,
        layout: Layout,
        shared: &[f64],
        histories: f64,
        volume: f64,
    ) -> ReplicaTallied {
        let r = sums.replicas;
        let column = |first: usize, len: usize, k: usize| -> Vec<f64> {
            (first..first + len).map(|i| shared[i * r + k]).collect()
        };
        let n_rates = layout.yields - layout.rates;
        let n_bins = self.union_grid.len();
        let inv_total = 1.0 / histories;
        let scale = 1.0 / (histories * volume * 1.0e24);
        let per_source = 1.0 / (histories * volume);
        let mut boundaries = self.union_grid.clone();
        if let Some(&last) = boundaries.last() {
            boundaries.push(if last > 0.0 { last * 10.0 } else { 1.0 });
        }
        let mut out = ReplicaTallied {
            seed: sums.seed,
            ..Default::default()
        };
        for k in 0..r {
            let rates = column(layout.rates, n_rates, k);
            out.rates
                .push(rates_of_sums(mat, &rates, inv_total, volume, 1.0));
            let s0 = column(layout.s0, n_bins, k);
            let s1 = match layout.s1 {
                Some(s1) => column(s1, n_bins, k),
                None => Vec::new(),
            };
            let yields = column(layout.yields, mat.yield_channels.len(), k);
            out.partials
                .push(self.partial_rates_of_sums(mat, &s0, &s1, &yields, scale));
            let fission = column(layout.fission, mat.fy_batch.len(), k);
            out.fy_weights
                .push(fission_yield_weights_of_sums(mat, &fission));
            if let (Some(covariance), Some(tally)) = (&self.isomer, &mat.isomer) {
                let sums = column(layout.isomer, tally.batch.len(), k);
                out.isomer.push(
                    self.isomer_productions_of_sums(covariance, tally, &sums, &s0, &s1, scale),
                );
            }
            out.spectra.push(crate::MultigroupSpectrum {
                boundaries: boundaries.clone(),
                masses: s0.iter().map(|v| v * per_source).collect(),
                flux_error: None,
            });
        }
        out
    }
}
