/// Flux-weighted reaction rate tallying for coupled transport-transmutation.
///
/// Scores σ_nuclide_MT(E) * track_length during transport for each nuclide/reaction
/// pair in transmutable materials. This gives the exact flux-weighted reaction rate
/// without any group approximation. Isomeric MF=10 final-state partials from
/// the branching overlay are folded exactly too (issue #218): the tally
/// accumulates two flux moments per bin of the union grid of every curve's
/// breakpoints, which reconstructs `sum(sigma_partial(E_i) * TL_i)` exactly
/// for any piecewise-linear curve, for every chain parent (including products
/// that build up during a step). MF=9 yields, which weight the parent's
/// transport cross section, are scored directly at the collision energy for
/// the material's own nuclides. So is an isomer-only MF=10 partial on one of
/// them above its last breakpoint: it follows the transport total there, at
/// the share it ends on, which the moments cannot fold, so its fold stops at
/// that breakpoint and the rest is a yield channel (`build_tail_channels`).
///
/// That grid also always carries a base log-spaced spectrum, whether or not an
/// overlay is configured, so the same two moments answer a second question:
/// what rate would a nuclide this tally does NOT carry have seen? Folding them
/// against its cross sections with a per-bin maximum bounds that rate without
/// scoring it, which is what deciding whether to carry it needs (issue #404).
///
/// Rate formula: rate_ij [1/s] = mean(σ_ij · TL) * 1e-24 * source_rate / volume
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use super::history_statistics::{
    fold_covariance, HistoryCovariance, HistoryStatistics, RateCovariance, RateLabel,
};
use super::material_transmute::{curve_interp, remainder_state};
use super::{mt_to_reaction_type, reaction_type_to_mt};
use yamc_materials::material::Material;
use yani::{
    fission_yield_interp_weights, BranchQuantity, BranchTable, ChainNuclide, FissionYieldWeights,
    ReactionRates,
};

/// MT for total fission, whose rate distribution weights the yield fold.
const MT_FISSION: i32 = 18;

/// Per-final-state isomeric partial rates for a material:
/// `parent -> reaction kind -> [(final-state target, rate [1/s])]`.
pub type PartialRates = HashMap<String, HashMap<String, Vec<(String, f64)>>>;

/// The collective operations [`TransmutationTallies::reduce_across_ranks`]
/// needs from a communicator.
///
/// WHAT is packed, and in which order, belongs with the accumulators; the
/// communicator does not. MPI lives in `yamc` behind its own cargo feature, so
/// the reduction takes the communicator through this trait instead of naming
/// it. `yamc::mpi_context::MpiContext` is the only implementor, and its
/// non-MPI build reports `size() == 1`, which short-circuits the reduction.
pub trait CollectiveOps {
    /// Number of ranks in the communicator (1 when MPI is absent).
    fn size(&self) -> i32;
    /// Element-wise sum of `local` across all ranks, left on `root_rank`.
    fn reduce_sum_f64(&self, local: &mut [f64], root_rank: i32);
    /// Overwrite `data` on every rank with `root_rank`'s copy.
    fn broadcast_f64(&self, data: &mut [f64], root_rank: i32);
}

/// A channel scored directly at the collision energy (issue #218): its curve
/// times the parent's transport cross section, `c(E) * sigma_MT(E)`. The curve
/// is an MF=9 yield `y_s(E)`, or, with `tail`, the share of that cross section
/// an isomer-only MF=10 partial ends on, from its last breakpoint up (see
/// [`build_tail_channels`]). Only built for the material's own nuclides (the
/// lookup needs their loaded cross sections; parents outside the material keep
/// their base split for a yield, as the multigroup fold also did, and a flat
/// tail for a partial).
struct YieldChannel {
    /// Chain nuclide the curve belongs to.
    parent: String,
    /// Reaction kind, e.g. `(n,gamma)`.
    kind: String,
    /// GNDS final-state name, e.g. `Ag110` (ground) or `Ag110_m1`.
    target: String,
    /// Transport MT whose cross section weights the yield.
    mt: i32,
    /// Curve grid (incident energy [eV], ascending) and values.
    energy: Vec<f64>,
    values: Vec<f64>,
    /// Whether this is the part of an isomer-only MF=10 partial above its last
    /// breakpoint (see [`build_tail_channels`]), which that partial's rate
    /// carries, rather than an MF=9 yield of its own.
    tail: bool,
}

/// How one material scores the part of a moment curve above its last
/// breakpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Tail {
    /// Folded from the moments with the rest of the curve, held flat (the
    /// `curve_interp` convention).
    Flat,
    /// Scored directly by the yield channel at this index (see
    /// [`build_tail_channels`]), or by nothing when there is no share of the
    /// total to hold. The fold stops at the last breakpoint either way.
    Scored(Option<usize>),
}

/// A fissionable nuclide of this material whose tabulated fission-yield
/// energies are folded against the continuous-energy fission-rate distribution
/// (issue #379).
///
/// The linear interpolation hats of the tabulated energies are scored with the
/// same `sigma_MT(E) * TL` the fission total uses, so the fold sees exactly the
/// flux the rate sees. At most two hats are non-zero at any energy, so this
/// costs two atomic adds per fissionable nuclide per track segment, and nothing
/// at all in a material with no fissionable nuclide.
struct FissionYieldChannel {
    /// Tabulated incident energies [eV], ascending. Positionally identical to
    /// the nuclide's `FissionYieldSet::yields`, which is what the resulting
    /// weights index into.
    energies: Vec<f64>,
    /// Offset of this channel's first accumulator slot; it owns
    /// `energies.len()` consecutive slots.
    offset: usize,
}

/// An MF=10 partial cross-section curve folded exactly from the union-grid
/// flux moments. Piecewise linear between breakpoints, zero below the first
/// point, flat above the last (the `curve_interp` conventions), except where a
/// material scores the part above the last directly (see
/// [`build_tail_channels`]).
struct MomentCurve {
    parent: String,
    kind: String,
    target: String,
    energy: Vec<f64>,
    values: Vec<f64>,
    /// Whether the curve's list names only metastable states and leaves the
    /// ground state the remainder (see `remainder_state`), which makes it a
    /// share of the transport total rather than of its fellow partials.
    isomer_only: bool,
}

/// Per-material accumulation data for transmutation tallying.
struct MaterialTransmutationData {
    /// Nuclide names (ordered, defines row index)
    nuclide_names: Vec<String>,
    /// MT numbers tracked (ordered, defines column index)
    mt_numbers: Vec<i32>,
    /// Accumulator: [nuclide_idx * n_mts + mt_idx] -> AtomicU64
    /// Stores running sum of σ(E)*TL for current batch (f64 as bits)
    batch_accum: Vec<AtomicU64>,
    /// Running sum of raw `Sum(σ·TL)` over all CPU chunks (normalized once, at
    /// extraction). Protected by Mutex for safe single-threaded access at chunk
    /// boundaries.
    sum_means: Mutex<Vec<f64>>,
    /// Total source particles accumulated across all CPU chunks -- the single
    /// normalization denominator. The CPU chunk count is statistics-neutral and
    /// must not affect results (issue #128).
    total_particles: AtomicUsize,
    /// Flux accumulator (total track length for this batch, f64 as bits)
    flux_batch_accum: AtomicU64,
    /// Sum of flux batch means.
    /// Protected by Mutex for safe single-threaded access at batch boundaries.
    flux_sum_means: Mutex<f64>,
    /// Zeroth flux moment per union-grid bin for this chunk: `sum(TL)` of the
    /// segments whose energy falls in the bin (f64 as bits). Always present:
    /// the grid carries a base spectrum even with no branching overlay, so a
    /// rate can be folded for a nuclide this tally does not carry.
    moment_s0_batch: Vec<AtomicU64>,
    /// First flux moment per union-grid bin for this chunk, taken relative to
    /// the bin's lower edge: `sum((E - grid[g]) * TL)`. The relative form
    /// avoids the catastrophic cancellation `sum(E*TL) - edge*sum(TL)` would
    /// suffer at high energies.
    moment_s1_batch: Vec<AtomicU64>,
    /// Running per-bin moment sums across all CPU chunks.
    moment_s0: Mutex<Vec<f64>>,
    moment_s1: Mutex<Vec<f64>>,
    /// MF=9 yield channels scored at the collision energy, then the parts of
    /// isomer-only MF=10 partials above their last breakpoint (see
    /// [`build_tail_channels`]); empty when no branching overlay is configured.
    yield_channels: Vec<YieldChannel>,
    /// Per moment curve, how this material scores its part above the last
    /// breakpoint.
    tails: Vec<Tail>,
    /// Per-channel accumulator for the current chunk (f64 as bits), parallel to
    /// `yield_channels`.
    yield_batch: Vec<AtomicU64>,
    /// Running per-channel sums across all CPU chunks.
    yield_sums: Mutex<Vec<f64>>,
    /// Fission-yield fold channels, indexed by the same position as
    /// `nuclide_names`; `None` for a nuclide with no tabulated yields.
    fy_channels: Vec<Option<FissionYieldChannel>>,
    /// Position of `MT_FISSION` in `mt_numbers`, so the hot path can spot the
    /// fission cross section it already looked up. `None` when no nuclide in
    /// this material fissions.
    fy_mt_idx: Option<usize>,
    /// Per-tabulated-energy accumulator for the current chunk (f64 as bits),
    /// laid out by `FissionYieldChannel::offset`.
    fy_batch: Vec<AtomicU64>,
    /// Running per-tabulated-energy sums across all CPU chunks.
    fy_sums: Mutex<Vec<f64>>,
}

/// Thread-safe transmutation tally accumulator.
///
/// Scores microscopic σ(E) * track_length during transport for each
/// nuclide/reaction pair in transmutable materials.
///
/// The hot path (`score`) uses lock-free AtomicU64 accumulators.
/// Batch accumulation and rate extraction acquire Mutexes but are
/// only called single-threaded at batch boundaries.
pub struct TransmutationTallies {
    /// Per-material accumulation data
    materials: HashMap<u32, MaterialTransmutationData>,
    /// Union grid [eV] of a base log-spaced spectrum grid and every moment
    /// curve's breakpoints, sorted and deduplicated. Between consecutive edges
    /// every curve is linear, so two flux moments per bin reconstruct each
    /// curve's `sum(sigma(E_i)*TL_i)` exactly; the extra base edges only
    /// subdivide those segments further, which the fold handles unchanged. The
    /// last bin extends to infinity (curves are flat above their last
    /// breakpoint); the first edge is zero, so nothing goes unscored.
    union_grid: Vec<f64>,
    /// MF=10 partial curves for every chain parent in the branch table (not
    /// just the material's nuclides: their fold needs no per-nuclide transport
    /// data, so products that build up during a step are covered too).
    moment_curves: Vec<MomentCurve>,
    /// Per-history covariance of the flux moments, when asked for with
    /// [`TransmutationTallies::with_history_statistics`]. `None` on the default
    /// path, which then allocates nothing for it and scores exactly as before.
    statistics: Option<HistoryStatistics>,
}

/// A malformed curve (empty or mismatched grids) would panic in `curve_interp`
/// or the moment fold; drop it at construction.
fn well_formed(c: &yani::BranchCurve) -> bool {
    !c.energy.is_empty() && c.energy.len() == c.values.len()
}

/// Collect the MF=10 moment curves from the branching overlay for every chain
/// parent, mirroring the curve selection of the multigroup fold: for `(n,n')`
/// only partials with a real state change; for other kinds MF=10 wins over
/// MF=9 when both exist.
fn build_moment_curves(
    chain: &HashMap<String, ChainNuclide>,
    branch: &BranchTable,
) -> Vec<MomentCurve> {
    let mut curves_out = Vec::new();
    // Sort parents and kinds so curve order (and thus extraction-time
    // summation order) is deterministic across runs.
    let mut parents: Vec<&String> = branch.keys().filter(|p| chain.contains_key(*p)).collect();
    parents.sort();
    for parent in parents {
        let kinds = &branch[parent];
        let mut kind_names: Vec<&String> = kinds.keys().collect();
        kind_names.sort();
        for kind in kind_names {
            let first = curves_out.len();
            for c in &kinds[kind] {
                if !well_formed(c) || c.quantity != BranchQuantity::CrossSection {
                    continue;
                }
                if kind == "(n,n')" && &c.target == parent {
                    continue; // ground self-loop is a no-op
                }
                curves_out.push(MomentCurve {
                    parent: parent.clone(),
                    kind: kind.clone(),
                    target: c.target.clone(),
                    energy: c.energy.clone(),
                    values: c.values.clone(),
                    isomer_only: false,
                });
            }
            // Decided over the curves kept, which are the targets
            // `apply_coupled_branching` will see for this kind.
            let listed = &mut curves_out[first..];
            let isomer_only = kind != "(n,n')"
                && remainder_state(
                    &chain[parent],
                    kind,
                    listed.iter().map(|c| c.target.as_str()),
                )
                .is_some();
            for c in listed {
                c.isomer_only = isomer_only;
            }
        }
    }
    curves_out
}

/// Sorted, deduplicated union of every moment curve's breakpoints.
fn build_union_grid(curves: &[MomentCurve]) -> Vec<f64> {
    let mut grid: Vec<f64> = base_spectrum_grid();
    grid.extend(curves.iter().flat_map(|c| c.energy.iter().copied()));
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup();
    grid
}

/// Bins per decade of the base grid the flux spectrum is always tallied on.
///
/// Fine enough that a threshold cross section's variation across one bin is
/// small, coarse enough to cost one 10-level binary search per track segment.
const SPECTRUM_BINS_PER_DECADE: usize = 50;

/// The base energy grid [eV] merged into the moment grid, so a flux spectrum
/// exists whether or not a branching overlay does.
///
/// Log spaced from 1e-5 eV to 30 MeV, prefixed with zero: energies below the
/// grid's first edge score nothing, and a segment that scores nothing would
/// silently drop flux out of any rate folded from these moments.
///
/// A spectrum is not a group approximation to anything the solver uses. The
/// reaction rates the burnup matrix is built from are still the exact
/// continuous-energy `sum(sigma(E_i) * TL_i)` accumulated per nuclide and MT.
/// These bins exist so a driver can ask what rate a nuclide the tally does not
/// carry WOULD have seen, which is what deciding whether to carry it needs
/// (issue #404).
fn base_spectrum_grid() -> Vec<f64> {
    const MIN_EV: f64 = 1.0e-5;
    const MAX_EV: f64 = 3.0e7;
    let decades = (MAX_EV / MIN_EV).log10();
    let n = (decades * SPECTRUM_BINS_PER_DECADE as f64).ceil() as usize;
    let mut grid = Vec::with_capacity(n + 2);
    grid.push(0.0);
    let step = decades / n as f64;
    for i in 0..=n {
        grid.push(MIN_EV * 10f64.powf(step * i as f64));
    }
    grid
}

/// The largest value a piecewise-linear curve takes in each grid bin, weighted
/// by that bin's track length and summed.
///
/// An upper bound on the continuous-energy `sum(sigma(E_i) * TL_i)` the tally
/// would accumulate for the same curve, never an under-estimate: every segment
/// lands in exactly one bin, and no segment can score more than its bin's peak.
/// Bin `g` covers `[grid[g], grid[g+1])` and the last bin runs to infinity.
///
/// The peak is taken over the breakpoints ENCLOSING the bin rather than over
/// the bin itself. The curve is linear between breakpoints, so the enclosing
/// pair bounds every value inside, and it costs no interpolation and makes no
/// assumption about what the curve does off the ends of its grid: below the
/// first point a curve is zero and a cross section is either zero (threshold)
/// or its first value, and above the last point both are flat, all of which
/// this already covers.
///
/// One merged walk over the bins and the curve's grid, so O(bins + points).
fn fold_bin_maxima(energy: &[f64], values: &[f64], grid: &[f64], s0: &[f64]) -> f64 {
    let n_points = energy.len().min(values.len());
    if n_points == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut idx = 0usize;
    for (g, &tl) in s0.iter().enumerate() {
        let lo = grid[g];
        let hi = grid.get(g + 1).copied().unwrap_or(f64::INFINITY);
        while idx < n_points && energy[idx] < lo {
            idx += 1;
        }
        if tl <= 0.0 {
            continue;
        }
        let mut peak = if idx > 0 { values[idx - 1] } else { 0.0 };
        let mut j = idx;
        while j < n_points && energy[j] < hi {
            peak = peak.max(values[j]);
            j += 1;
        }
        if j < n_points {
            peak = peak.max(values[j]);
        }
        if peak > 0.0 {
            total += peak * tl;
        }
    }
    total
}

/// Collect the MF=9 yield channels for one material's nuclides: kinds with no
/// MF=10 partials whose MT resolves (the yield weights `sigma_MT(E)` at
/// scoring time, so the parent's cross sections must be loaded).
fn build_yield_channels(nuclide_names: &[String], branch: &BranchTable) -> Vec<YieldChannel> {
    let mut channels = Vec::new();
    for name in nuclide_names {
        let Some(kinds) = branch.get(name) else {
            continue;
        };
        let mut kind_names: Vec<&String> = kinds.keys().collect();
        kind_names.sort();
        for kind in kind_names {
            let curves = &kinds[kind];
            if kind == "(n,n')"
                || curves
                    .iter()
                    .any(|c| c.quantity == BranchQuantity::CrossSection)
            {
                continue; // MF=10 partials are folded from the moments
            }
            let Some(mt) = reaction_type_to_mt(kind) else {
                continue;
            };
            for c in curves {
                if well_formed(c) && c.quantity == BranchQuantity::Yield {
                    channels.push(YieldChannel {
                        parent: name.clone(),
                        kind: kind.clone(),
                        target: c.target.clone(),
                        mt,
                        energy: c.energy.clone(),
                        values: c.values.clone(),
                        tail: false,
                    });
                }
            }
        }
    }
    channels
}

/// For one material, append to `channels` the part of each isomer-only MF=10
/// partial above its last breakpoint, scored at the collision energy as a
/// share of the transport total, and return per moment curve how this material
/// scores that part.
///
/// Such a partial is a share of its reaction's transport total, and past its
/// last breakpoint it follows that total at the share it ends on, as the
/// multigroup fold's `along_the_total` has it. Held flat instead, a partial
/// that stops at 20 MeV was a share of a total that goes on changing, and
/// could make the isomer all of the reaction above it. The moments cannot fold
/// the total, which is not one of their curves, so the part above is a yield
/// channel, `sigma_s(E_l) / sigma_MT(E_l)` times `sigma_MT(E)` from the last
/// breakpoint `E_l` on, and the curve's own fold stops at `E_l`. With no total
/// at `E_l` there is no share to hold, and the fold stops with no channel.
///
/// Only for the material's own nuclides: they are the only parents with a
/// tallied total to take a share of. Any other parent's curve keeps its flat
/// tail, and `apply_coupled_branching` leaves its split alone.
fn build_tail_channels(
    curves: &[MomentCurve],
    nuclide_names: &[String],
    material: &Material,
    channels: &mut Vec<YieldChannel>,
) -> Vec<Tail> {
    let mut tails = vec![Tail::Flat; curves.len()];
    for (c, tail) in curves.iter().zip(&mut tails) {
        if !c.isomer_only || !nuclide_names.contains(&c.parent) {
            continue;
        }
        let Some(mt) = reaction_type_to_mt(&c.kind) else {
            continue;
        };
        let Some(reaction) = material
            .nuclide_data
            .get(&c.parent)
            .and_then(|nd| nd.reactions_for_temp(material.temperature()))
            .and_then(|reactions| reactions.get(&mt))
        else {
            continue;
        };
        *tail = Tail::Scored(None);
        let e_last = c.energy[c.energy.len() - 1];
        let share = match reaction.cross_section_at(e_last) {
            Some(total) if total > 0.0 => c.values[c.values.len() - 1] / total,
            _ => continue,
        };
        if share > 0.0 {
            *tail = Tail::Scored(Some(channels.len()));
            channels.push(YieldChannel {
                parent: c.parent.clone(),
                kind: c.kind.clone(),
                target: c.target.clone(),
                mt,
                // One point, which `curve_interp` reads as zero below `E_l`
                // and the share from `E_l` on, `E_l` included: a collision
                // there lands in the union bin starting at `E_l`, which the
                // fold stops short of. A step `[E_l, E_l] -> [0, share]` reads
                // zero at `E_l`, and a 20 MeV source, where every ENDF/B-VIII.1
                // and JEFF-4.0 isomer-only partial ends, then made no isomer
                // from its uncollided flux.
                energy: vec![e_last],
                values: vec![share],
                tail: true,
            });
        }
    }
    tails
}

/// Exactly reconstruct `sum(sigma(E_i) * TL_i)` for a piecewise-linear curve
/// from the union-grid flux moments: within a bin every curve is linear, so
/// the sum is `sigma(edge) * S0 + slope * S1` with `S1` relative to the edge.
/// `suffix_s0[g]` is `sum(s0[g..])`, used for the flat tail above the curve's
/// last breakpoint in O(1). Without `flat_tail` the sum stops at that
/// breakpoint, for a curve whose part above is scored directly (see
/// [`build_tail_channels`]).
///
/// At a duplicated breakpoint (a step discontinuity) a segment landing exactly
/// on the jump energy folds with the right-limit value; `curve_interp` picks
/// an arbitrary side there too (std's binary_search tie), so either choice is
/// within the data's own ambiguity.
fn fold_curve_from_moments(
    energy: &[f64],
    values: &[f64],
    grid: &[f64],
    s0: &[f64],
    s1: &[f64],
    suffix_s0: &[f64],
    flat_tail: bool,
) -> f64 {
    let mut total = 0.0;
    // Linear interior segments. The union grid contains every breakpoint, so
    // each bin inside [e0, e1) lies entirely within this linear segment.
    for k in 0..energy.len() - 1 {
        let (e0, e1) = (energy[k], energy[k + 1]);
        if e1 <= e0 {
            continue;
        }
        let slope = (values[k + 1] - values[k]) / (e1 - e0);
        let g0 = grid.partition_point(|&e| e < e0);
        let g1 = grid.partition_point(|&e| e < e1);
        for g in g0..g1 {
            let sigma_at_edge = values[k] + slope * (grid[g] - e0);
            total += sigma_at_edge * s0[g] + slope * s1[g];
        }
    }
    // Flat tail above the last breakpoint (curve_interp convention).
    let v_last = values[values.len() - 1];
    if flat_tail && v_last != 0.0 {
        let g0 = grid.partition_point(|&e| e < energy[energy.len() - 1]);
        if g0 < suffix_s0.len() {
            total += v_last * suffix_s0[g0];
        }
    }
    total
}

impl TransmutationTallies {
    /// Create a new TransmutationTallies for all transmutable materials.
    ///
    /// # Arguments
    /// * `transmutable_cells` - Map of material_id -> Vec<cell_index>
    /// * `materials` - Map of material_id -> &Material (for nuclide data)
    /// * `chain` - Transmutation chain (nuclide name -> ChainNuclide)
    /// * `branch` - Isomeric-branching overlay (empty when not configured);
    ///   its per-final-state curves are scored directly at the collision
    ///   energy (issue #218)
    /// * `carried` - Per material, the nuclides worth scoring. A material with
    ///   no entry scores every nuclide it has cross sections for, which is what
    ///   a driver that has not worked out a product bound wants. A material
    ///   with an entry scores only those, which is where the pruning saves its
    ///   `n_nuclides x n_MTs x n_segments` (issue #404).
    ///
    ///   This is a filter rather than something the driver expresses by
    ///   trimming `Material::nuclide_data`, because that map is transport's as
    ///   well as ours: trimming it moves the flux.
    pub fn new(
        transmutable_cells: &HashMap<u32, Vec<usize>>,
        materials: &HashMap<u32, &Material>,
        chain: &HashMap<String, ChainNuclide>,
        branch: &BranchTable,
        carried: &HashMap<u32, std::collections::HashSet<String>>,
    ) -> Self {
        let mut mat_data = HashMap::new();
        let moment_curves = build_moment_curves(chain, branch);
        let union_grid = build_union_grid(&moment_curves);
        let n_moment_bins = union_grid.len();

        for &mat_id in transmutable_cells.keys() {
            let material = match materials.get(&mat_id) {
                Some(m) => *m,
                None => continue,
            };

            // Find nuclides that are in both the material and the chain, minus
            // any the caller's product bound has ruled out.
            let keep = carried.get(&mat_id);
            let mut nuclide_names: Vec<String> = material
                .nuclide_data
                .keys()
                .filter(|name| chain.contains_key(*name))
                .filter(|name| keep.is_none_or(|k| k.contains(*name)))
                .cloned()
                .collect();
            nuclide_names.sort();

            if nuclide_names.is_empty() {
                continue;
            }

            // Collect all unique MTs needed for these nuclides from the chain
            let mut mt_set = std::collections::HashSet::new();
            for name in &nuclide_names {
                if let Some(chain_nuclide) = chain.get(name) {
                    for reaction in &chain_nuclide.reactions {
                        if let Some(mt) = reaction_type_to_mt(&reaction.kind) {
                            mt_set.insert(mt);
                        }
                    }
                }
            }

            let mut mt_numbers: Vec<i32> = mt_set.into_iter().collect();
            mt_numbers.sort();

            if mt_numbers.is_empty() {
                continue;
            }

            let n_bins = nuclide_names.len() * mt_numbers.len();
            let batch_accum: Vec<AtomicU64> = (0..n_bins).map(|_| AtomicU64::new(0)).collect();

            let mut yield_channels = build_yield_channels(&nuclide_names, branch);
            let tails = build_tail_channels(
                &moment_curves,
                &nuclide_names,
                material,
                &mut yield_channels,
            );
            let n_channels = yield_channels.len();

            // Fission-yield fold channels (issue #379). Only meaningful when
            // the material actually tracks fission, so the whole apparatus
            // stays empty for the fusion materials that never fission.
            let fy_mt_idx = mt_numbers.iter().position(|&mt| mt == MT_FISSION);
            let mut n_fy_slots = 0usize;
            let fy_channels: Vec<Option<FissionYieldChannel>> = nuclide_names
                .iter()
                .map(|name| {
                    fy_mt_idx?;
                    let energies = chain
                        .get(name)?
                        .fission_yields
                        .as_ref()
                        .filter(|s| !s.yields.is_empty())?
                        .energies();
                    let offset = n_fy_slots;
                    n_fy_slots += energies.len();
                    Some(FissionYieldChannel { energies, offset })
                })
                .collect();

            mat_data.insert(
                mat_id,
                MaterialTransmutationData {
                    nuclide_names,
                    mt_numbers,
                    batch_accum,
                    sum_means: Mutex::new(vec![0.0; n_bins]),
                    total_particles: AtomicUsize::new(0),
                    flux_batch_accum: AtomicU64::new(0),
                    flux_sum_means: Mutex::new(0.0),
                    moment_s0_batch: (0..n_moment_bins).map(|_| AtomicU64::new(0)).collect(),
                    moment_s1_batch: (0..n_moment_bins).map(|_| AtomicU64::new(0)).collect(),
                    moment_s0: Mutex::new(vec![0.0; n_moment_bins]),
                    moment_s1: Mutex::new(vec![0.0; n_moment_bins]),
                    yield_channels,
                    tails,
                    yield_batch: (0..n_channels).map(|_| AtomicU64::new(0)).collect(),
                    yield_sums: Mutex::new(vec![0.0; n_channels]),
                    fy_channels,
                    fy_mt_idx,
                    fy_batch: (0..n_fy_slots).map(|_| AtomicU64::new(0)).collect(),
                    fy_sums: Mutex::new(vec![0.0; n_fy_slots]),
                },
            );
        }

        TransmutationTallies {
            materials: mat_data,
            union_grid,
            moment_curves,
            statistics: None,
        }
    }

    /// Also accumulate the per-history covariance of each material's tally
    /// (see [`HistoryCovariance`]), so every rate it reports carries a
    /// statistical uncertainty and a correlation with every other.
    ///
    /// Off unless asked for. It costs `(626 + Y + R)^2 / 2` doubles per
    /// transmutable material, for `Y` yield channels and `R` scored
    /// nuclide-MT pairs (about 2 MB with no rates, 5 MB at `R = 500`), plus
    /// a sparse outer product per history. None of it touches the sums the
    /// means are built from, so the means are bit-identical with it on or off.
    ///
    /// A run that scores this tally must call
    /// [`prepare_history_workers`](Self::prepare_history_workers) before
    /// transport and [`finish_history`](Self::finish_history) after each
    /// source history.
    pub fn with_history_statistics(mut self) -> Self {
        let mut ids: Vec<u32> = self.materials.keys().copied().collect();
        ids.sort_unstable();
        let materials: Vec<(u32, usize, usize)> = ids
            .iter()
            .map(|id| {
                let m = &self.materials[id];
                (
                    *id,
                    m.yield_channels.len(),
                    m.nuclide_names.len() * m.mt_numbers.len(),
                )
            })
            .collect();
        self.statistics = Some(HistoryStatistics::new(
            base_spectrum_grid(),
            &self.union_grid,
            &materials,
        ));
        self
    }

    /// Whether [`with_history_statistics`](Self::with_history_statistics) is on.
    pub fn has_history_statistics(&self) -> bool {
        self.statistics.is_some()
    }

    /// Size the per-worker history scratch for a transport run on `n_workers`
    /// rayon threads. Call single-threaded before transport. A no-op when
    /// history statistics are off, and when repeated for no more workers than
    /// the first call, so a tally reused across transmutation steps is fine;
    /// more workers than the first call is an error.
    pub fn prepare_history_workers(&self, n_workers: usize) -> Result<(), String> {
        match &self.statistics {
            Some(stats) => stats.prepare_workers(n_workers),
            None => Ok(()),
        }
    }

    /// Close the source history running on the calling worker thread, after
    /// every particle it produced has finished. A no-op when history
    /// statistics are off.
    #[inline]
    pub fn finish_history(&self) {
        if let Some(stats) = &self.statistics {
            stats.finish_history();
        }
    }

    /// The per-history mean and covariance of `material_id`'s tally vector,
    /// over every source particle accumulated so far. `None` when history
    /// statistics are off or the material is not tallied.
    pub fn history_covariance(&self, material_id: u32) -> Option<HistoryCovariance> {
        let stats = self.statistics.as_ref()?;
        let mat_data = self.materials.get(&material_id)?;
        let n = mat_data.total_particles.load(Ordering::Relaxed) as u64;
        let labels = mat_data
            .yield_channels
            .iter()
            .map(|c| (c.parent.clone(), c.kind.clone(), c.target.clone()))
            .collect();
        let rates = mat_data
            .nuclide_names
            .iter()
            .flat_map(|name| {
                mat_data
                    .mt_numbers
                    .iter()
                    .map(move |&mt| (name.clone(), mt))
            })
            .collect();
        stats.extract(material_id, n, labels, rates)
    }

    /// The statistical covariance of every rate this material's tally
    /// reports, the totals of [`get_reaction_rates`](Self::get_reaction_rates)
    /// and the partials of [`get_partial_rates`](Self::get_partial_rates),
    /// normalized the same way. `None` when history statistics are off or the
    /// material is not tallied.
    ///
    /// The totals and the MF=9 yields are entries of the history vector
    /// themselves, so their covariances are exact. The MF=10 partials are
    /// folded from the spectrum, like their means, and get their covariance as
    /// `a_i^T Sigma a_j / n` (see [`HistoryCovariance`]). Their fold weights
    /// come from the tally itself: in each base-grid bin, the curve averaged
    /// over the flux the union-grid moments say that bin actually saw, then
    /// scaled so the fold reproduces the partial exactly. So the fold decides
    /// only how a partial's fluctuation is spread over the spectrum, never the
    /// partial itself. What it cannot see is a fluctuation in the shape of the
    /// spectrum inside one base bin (50 per decade), which matters only where
    /// the curve changes a lot across one bin; the branching curves are smooth.
    /// An isomer-only partial's part above its last breakpoint is an entry of
    /// its own (see `build_tail_channels`), so its row is the fold's weights
    /// up to that breakpoint plus that entry, exactly.
    pub fn get_reaction_rate_covariance(
        &self,
        material_id: u32,
        volume: f64,
        source_rate: f64,
    ) -> Option<RateCovariance> {
        let moments = self.history_covariance(material_id)?;
        let mat_data = self.materials.get(&material_id)?;
        let n = moments.n_histories;
        if n == 0 || volume <= 0.0 {
            return Some(RateCovariance::from_parts(
                Vec::new(),
                Vec::new(),
                n,
                Vec::new(),
            ));
        }
        // Rates are raw sums times `to_rate`. The weights act on the vector's
        // MEANS, which are already per source particle, so they carry
        // `per_mean` instead.
        let to_rate = source_rate / (n as f64 * volume * 1.0e24);
        let per_mean = source_rate / (volume * 1.0e24);

        // Fine-grid flux per bin and its flux-weighted energy, and which base
        // bin each fine bin sits in.
        let base = &moments.grid;
        let s0 = mat_data
            .moment_s0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let s1 = mat_data
            .moment_s1
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let centroid: Vec<f64> = self
            .union_grid
            .iter()
            .zip(s0.iter().zip(&s1))
            .map(|(&edge, (&m0, &m1))| if m0 > 0.0 { edge + m1 / m0 } else { edge })
            .collect();
        let base_of: Vec<usize> = self
            .union_grid
            .iter()
            .map(|&e| base.partition_point(|&b| b <= e).max(1) - 1)
            .collect();
        let mut base_s0 = vec![0.0; base.len()];
        for (g, &c) in base_of.iter().enumerate() {
            base_s0[c] += s0[g];
        }

        // One weight row from a cross section evaluated at the fine centroids,
        // scaled so it folds to `raw_sum`, the tally's own `sum(sigma * TL)`.
        let row_for = |sigma: &dyn Fn(f64) -> f64, raw_sum: f64| -> Vec<(usize, f64)> {
            let mut per_base = vec![0.0; base.len()];
            for (g, &m0) in s0.iter().enumerate() {
                if m0 > 0.0 {
                    per_base[base_of[g]] += sigma(centroid[g]) * m0;
                }
            }
            // Folding per_base / base_s0 against the base s0 sums gives back
            // sum(per_base), so that is the normalization.
            let folded: f64 = per_base.iter().sum();
            if folded <= 0.0 {
                return Vec::new();
            }
            let k = raw_sum / folded * per_mean;
            per_base
                .iter()
                .enumerate()
                .filter(|(c, v)| **v > 0.0 && base_s0[*c] > 0.0)
                .map(|(c, v)| (moments.s0_index(c), v / base_s0[c] * k))
                .collect()
        };

        let mut labels = Vec::new();
        let mut rates = Vec::new();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();

        // Reaction totals, scored directly: an exact entry each. Only those
        // `get_reaction_rates` reports, which is every pair that scored.
        let sums = mat_data
            .sum_means
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let n_mts = mat_data.mt_numbers.len();
        for (nuc_idx, name) in mat_data.nuclide_names.iter().enumerate() {
            for (mt_idx, &mt) in mat_data.mt_numbers.iter().enumerate() {
                let j = nuc_idx * n_mts + mt_idx;
                let (Some(kind), true) = (mt_to_reaction_type(mt), sums[j] > 0.0) else {
                    continue;
                };
                labels.push(RateLabel {
                    nuclide: name.clone(),
                    kind: kind.to_string(),
                    target: None,
                });
                rates.push(sums[j] * to_rate);
                rows.push(vec![(moments.rate_index(j), per_mean)]);
            }
        }

        // MF=10 partials, folded from the moments for the mean as well.
        let ysums = mat_data
            .yield_sums
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let mut suffix_s0 = vec![0.0; s0.len() + 1];
        for g in (0..s0.len()).rev() {
            suffix_s0[g] = suffix_s0[g + 1] + s0[g];
        }
        for (c, &tail) in self.moment_curves.iter().zip(&mat_data.tails) {
            let flat_tail = tail == Tail::Flat;
            let exact = fold_curve_from_moments(
                &c.energy,
                &c.values,
                &self.union_grid,
                &s0,
                &s1,
                &suffix_s0,
                flat_tail,
            );
            labels.push(RateLabel {
                nuclide: c.parent.clone(),
                kind: c.kind.clone(),
                target: Some(c.target.clone()),
            });
            // A tail scored as a yield channel is that channel's, added below.
            let e_last = c.energy[c.energy.len() - 1];
            let sigma = |e: f64| {
                if !flat_tail && e >= e_last {
                    0.0
                } else {
                    curve_interp(&c.energy, &c.values, e)
                }
            };
            let mut row = if exact > 0.0 {
                row_for(&sigma, exact)
            } else {
                Vec::new()
            };
            match tail {
                Tail::Scored(Some(k)) => {
                    rates.push((exact + ysums[k]) * to_rate);
                    row.push((moments.yield_index(k), per_mean));
                }
                _ => rates.push(exact * to_rate),
            }
            rows.push(row);
        }

        // MF=9 yields: scored directly, so the weight is exact.
        for (k, (ch, &sum)) in mat_data.yield_channels.iter().zip(&ysums).enumerate() {
            if ch.tail {
                continue; // its partial's, above
            }
            labels.push(RateLabel {
                nuclide: ch.parent.clone(),
                kind: ch.kind.clone(),
                target: Some(ch.target.clone()),
            });
            rates.push(sum * to_rate);
            rows.push(vec![(moments.yield_index(k), per_mean)]);
        }

        let covariance = fold_covariance(&moments, &rows);
        Some(RateCovariance::from_parts(labels, rates, n, covariance))
    }

    /// The flux shape this material's tally saw, on the union grid, as a
    /// multigroup spectrum: each bin's track length. `None` when the material
    /// is not tallied or nothing was scored.
    ///
    /// For folding MF=33 covariance against the transport spectrum, which is
    /// relative and so needs only the shape. The union grid's last bin runs to
    /// infinity; it is closed at ten times its lower edge, which for the base
    /// grid is 300 MeV and holds no flux in any fixed-source problem this code
    /// runs.
    pub fn flux_spectrum(&self, material_id: u32) -> Option<crate::MultigroupSpectrum> {
        let mat_data = self.materials.get(&material_id)?;
        let s0 = mat_data
            .moment_s0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if s0.iter().all(|v| *v <= 0.0) {
            return None;
        }
        let mut boundaries = self.union_grid.clone();
        let last = *boundaries.last()?;
        boundaries.push(if last > 0.0 { last * 10.0 } else { 1.0 });
        Some(crate::MultigroupSpectrum {
            boundaries,
            masses: s0,
            flux_error: None,
        })
    }

    /// Score a track segment in a transmutable material.
    ///
    /// For each nuclide/MT pair tracked in this material, looks up σ(E)
    /// and atomically accumulates σ * track_length.
    ///
    /// Called from parallel transport threads. Uses only lock-free atomics.
    ///
    /// # Arguments
    /// * `material_id` - ID of the material being traversed
    /// * `energy` - Neutron energy [eV]
    /// * `track_length` - Track segment length [cm]
    /// * `material` - Reference to the material (for nuclide data access)
    #[inline]
    pub fn score(&self, material_id: u32, energy: f64, track_length: f64, material: &Material) {
        let mat_data = match self.materials.get(&material_id) {
            Some(d) => d,
            None => return,
        };

        let temperature = material.temperature();
        let n_mts = mat_data.mt_numbers.len();
        let mut history = self
            .statistics
            .as_ref()
            .and_then(|s| s.history(material_id));

        // Score flux (total track length)
        atomic_add_f64(&mat_data.flux_batch_accum, track_length);

        // Two-moment flux tally on the branch-curve union grid (issue #218):
        // one binary search and two adds reconstruct sum(sigma(E_i)*TL_i)
        // exactly for every piecewise-linear MF=10 partial at extraction time,
        // however many curves the overlay carries. Below the first edge every
        // curve is zero; the last bin extends to infinity (flat tails).
        let grid = &self.union_grid;
        if !grid.is_empty() && energy >= grid[0] {
            let g = grid.partition_point(|&e| e <= energy) - 1;
            atomic_add_f64(&mat_data.moment_s0_batch[g], track_length);
            atomic_add_f64(
                &mat_data.moment_s1_batch[g],
                (energy - grid[g]) * track_length,
            );
            if let Some(h) = &mut history {
                h.add_track_length(g, track_length);
            }
        }

        // Score σ*TL for each nuclide/MT pair
        for (nuc_idx, nuclide_name) in mat_data.nuclide_names.iter().enumerate() {
            let nuclide_data = match material.nuclide_data.get(nuclide_name) {
                Some(nd) => nd,
                None => continue,
            };

            let reactions = match nuclide_data.reactions_for_temp(temperature) {
                Some(r) => r,
                None => continue,
            };

            for (mt_idx, &mt) in mat_data.mt_numbers.iter().enumerate() {
                if let Some(reaction) = reactions.get(&mt) {
                    if let Some(xs) = reaction.cross_section_at(energy) {
                        if xs > 0.0 {
                            let bin_idx = nuc_idx * n_mts + mt_idx;
                            atomic_add_f64(&mat_data.batch_accum[bin_idx], xs * track_length);
                            if let Some(h) = &mut history {
                                h.add_rate(bin_idx, xs * track_length);
                            }

                            // Split this segment's fission rate across the
                            // nuclide's tabulated yield energies (issue #379),
                            // reusing the cross section already in hand. Two
                            // hats are non-zero at most, so two adds.
                            if Some(mt_idx) == mat_data.fy_mt_idx {
                                if let Some(ch) = &mat_data.fy_channels[nuc_idx] {
                                    if let Some(hats) =
                                        fission_yield_interp_weights(&ch.energies, energy)
                                    {
                                        for (k, w) in hats {
                                            if w != 0.0 {
                                                atomic_add_f64(
                                                    &mat_data.fy_batch[ch.offset + k],
                                                    w * xs * track_length,
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
        }

        // Direct continuous-energy scoring of the MF=9 yield channels, and of
        // the isomer-only partials' tails, at the collision energy (issue
        // #218): the yield weighted by the parent's transport cross section,
        // using the identical σ(E)·TL machinery as the totals above so a
        // reaction's final-state partials and its total see the same flux.
        for (ch_idx, ch) in mat_data.yield_channels.iter().enumerate() {
            let Some(nd) = material.nuclide_data.get(&ch.parent) else {
                continue;
            };
            let Some(reactions) = nd.reactions_for_temp(temperature) else {
                continue;
            };
            let Some(reaction) = reactions.get(&ch.mt) else {
                continue;
            };
            let contrib = match reaction.cross_section_at(energy) {
                Some(xs) if xs > 0.0 => curve_interp(&ch.energy, &ch.values, energy) * xs,
                _ => 0.0,
            };
            if contrib > 0.0 {
                atomic_add_f64(&mat_data.yield_batch[ch_idx], contrib * track_length);
                if let Some(h) = &mut history {
                    h.add_yield(ch_idx, contrib * track_length);
                }
            }
        }
    }

    /// Fold the current CPU chunk's raw sums into the running totals.
    ///
    /// Adds this chunk's raw `Sum(σ·TL)` (and raw flux `Sum(TL)`) into the
    /// running sums and advances the source-particle count by
    /// `chunk_particles`, then resets the per-chunk accumulators. Normalization
    /// by the total source-particle count happens once, at extraction time
    /// (`get_reaction_rates` / `get_flux`), so the number of CPU chunks the
    /// transport was split into does not affect results (issue #128).
    ///
    /// Must be called single-threaded at chunk boundaries.
    pub fn accumulate_batch(&self, chunk_particles: usize) {
        for mat_data in self.materials.values() {
            let mut sum_means = mat_data
                .sum_means
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            for (i, sum_mean) in sum_means.iter_mut().enumerate() {
                let raw = mat_data.batch_accum[i].load(Ordering::Relaxed);
                *sum_mean += f64::from_bits(raw);
                // Reset for the next chunk
                mat_data.batch_accum[i].store(0, Ordering::Relaxed);
            }

            // Accumulate flux (raw total track length)
            let flux_raw = mat_data.flux_batch_accum.load(Ordering::Relaxed);
            let mut flux_sum_means = mat_data
                .flux_sum_means
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *flux_sum_means += f64::from_bits(flux_raw);
            mat_data.flux_batch_accum.store(0, Ordering::Relaxed);

            // Accumulate the union-grid flux moments
            {
                let mut s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
                let mut s1 = mat_data.moment_s1.lock().unwrap_or_else(|p| p.into_inner());
                for g in 0..s0.len() {
                    s0[g] += f64::from_bits(mat_data.moment_s0_batch[g].load(Ordering::Relaxed));
                    s1[g] += f64::from_bits(mat_data.moment_s1_batch[g].load(Ordering::Relaxed));
                    mat_data.moment_s0_batch[g].store(0, Ordering::Relaxed);
                    mat_data.moment_s1_batch[g].store(0, Ordering::Relaxed);
                }
            }

            // Accumulate per-channel yields, the MF=9 ones and the tails
            {
                let mut psums = mat_data
                    .yield_sums
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                for (i, ps) in psums.iter_mut().enumerate() {
                    let raw = mat_data.yield_batch[i].load(Ordering::Relaxed);
                    *ps += f64::from_bits(raw);
                    mat_data.yield_batch[i].store(0, Ordering::Relaxed);
                }
            }

            // Accumulate the per-tabulated-energy fission-rate shares
            {
                let mut fsums = mat_data.fy_sums.lock().unwrap_or_else(|p| p.into_inner());
                for (i, fs) in fsums.iter_mut().enumerate() {
                    let raw = mat_data.fy_batch[i].load(Ordering::Relaxed);
                    *fs += f64::from_bits(raw);
                    mat_data.fy_batch[i].store(0, Ordering::Relaxed);
                }
            }

            mat_data
                .total_particles
                .fetch_add(chunk_particles, Ordering::Relaxed);
        }
    }

    /// Get reaction rates for a specific material.
    ///
    /// Computes: rate_ij [1/s] = mean(σ_ij · TL) * 1e-24 * source_rate / volume
    ///
    /// # Arguments
    /// * `material_id` - ID of the material
    /// * `volume` - Material volume [cm³]
    /// * `source_rate` - Total source rate [n/s]
    ///
    /// # Returns
    /// ReactionRates: nuclide_name -> reaction_type -> rate [1/s]
    pub fn get_reaction_rates(
        &self,
        material_id: u32,
        volume: f64,
        source_rate: f64,
    ) -> ReactionRates {
        let mut rates: ReactionRates = HashMap::new();

        let mat_data = match self.materials.get(&material_id) {
            Some(d) => d,
            None => return rates,
        };

        let total_particles = mat_data.total_particles.load(Ordering::Relaxed);
        let sum_means = mat_data
            .sum_means
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if total_particles == 0 || volume <= 0.0 {
            return rates;
        }

        #[cfg(feature = "debug_transmutation")]
        eprintln!(
            "[DEBUG transmutation] Material {}: volume={:.6e} cm³, source_rate={:.6e} n/s, total_particles={}",
            material_id, volume, source_rate, total_particles
        );

        let n_mts = mat_data.mt_numbers.len();
        let inv_total = 1.0 / total_particles as f64;

        for (nuc_idx, nuclide_name) in mat_data.nuclide_names.iter().enumerate() {
            let mut nuclide_rates: HashMap<String, f64> = HashMap::new();

            for (mt_idx, &mt) in mat_data.mt_numbers.iter().enumerate() {
                let bin_idx = nuc_idx * n_mts + mt_idx;
                let mean_sigma_tl = sum_means[bin_idx] * inv_total;

                if mean_sigma_tl > 0.0 {
                    // rate = (σ×TL / volume_b_cm) × source_rate
                    // where σ×TL is in [b·cm] per source particle
                    // volume_b_cm = volume × 1e24 [b-cm]
                    // Result: [(b·cm) / (b·cm)] × [src/s] = [1/s] per atom
                    let volume_b_cm = volume * 1.0e24;
                    let rate = (mean_sigma_tl / volume_b_cm) * source_rate;

                    #[cfg(feature = "debug_transmutation")]
                    if let Some(rx_name) = mt_to_reaction_type(mt) {
                        eprintln!(
                            "[DEBUG transmutation]   {} MT={} {}: mean_sigma_tl={:.6e} b·cm, rate={:.6e} /s/atom",
                            nuclide_name, mt, rx_name, mean_sigma_tl, rate
                        );
                    }

                    if let Some(rx_name) = mt_to_reaction_type(mt) {
                        nuclide_rates.insert(rx_name.to_string(), rate);
                    }
                }
            }

            if !nuclide_rates.is_empty() {
                rates.insert(nuclide_name.clone(), nuclide_rates);
            }
        }

        rates
    }

    /// Upper-bound per-atom reaction rates [1/s] for every chain nuclide this
    /// material carries cross sections for, folded from the tallied flux
    /// spectrum rather than scored per collision.
    ///
    /// `get_reaction_rates` can only answer for the nuclides this tally
    /// registered, which is the set whose cost the pruning exists to cut. This
    /// answers for any nuclide whose cross sections are loaded, at the price of
    /// a bin maximum in place of an exact fold, so a driver can rate a
    /// candidate product without first paying to score it (issue #404).
    ///
    /// Every rate is an upper bound on what scoring the same flux would give
    /// (see [`fold_bin_maxima`]), so feeding these to
    /// `yani::populated_nuclides` keeps that function's answer an upper bound
    /// too. Rates are keyed by the chain's own reaction-kind spelling, which is
    /// what the bound looks up.
    ///
    /// `(n,n')` has no transport MT, exactly as in the multigroup collapse, so
    /// its rate comes from the branching overlay's MF=10 partials folded the
    /// same way. Every other kind's partials are splits of a total that is
    /// already counted, so they are not added again.
    ///
    /// Normalization matches `get_reaction_rates` exactly:
    /// `mean(sigma·TL) / (volume · 1e24) · source_rate`.
    pub fn bounding_reaction_rates(
        &self,
        material_id: u32,
        material: &Material,
        chain: &HashMap<String, ChainNuclide>,
        volume: f64,
        source_rate: f64,
    ) -> ReactionRates {
        let mut rates: ReactionRates = HashMap::new();

        let Some(mat_data) = self.materials.get(&material_id) else {
            return rates;
        };
        let total_particles = mat_data.total_particles.load(Ordering::Relaxed);
        if total_particles == 0 || volume <= 0.0 || source_rate <= 0.0 {
            return rates;
        }
        let s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
        let scale = source_rate / (total_particles as f64 * volume * 1.0e24);
        let temperature = material.temperature();

        for (name, nuclide_data) in &material.nuclide_data {
            let Some(chain_nuclide) = chain.get(name) else {
                continue;
            };
            let Some(reactions) = nuclide_data.reactions_for_temp(temperature) else {
                continue;
            };
            let mut per_kind: HashMap<String, f64> = HashMap::new();
            // Every kind the chain names for this parent, plus fission when it
            // carries yields but no explicit fission channel: the bound asks
            // for a "fission" rate wherever there are yields to spread.
            let kinds = chain_nuclide.reactions.iter().map(|rx| rx.kind.as_str());
            let fission_kind = chain_nuclide.fission_yields.is_some().then_some("fission");
            for kind in kinds.chain(fission_kind) {
                if per_kind.contains_key(kind) {
                    continue;
                }
                let Some(mt) = reaction_type_to_mt(kind) else {
                    continue;
                };
                let Some(reaction) = reactions.get(&mt) else {
                    continue;
                };
                let folded = fold_bin_maxima(
                    &reaction.energy,
                    &reaction.cross_section,
                    &self.union_grid,
                    &s0,
                );
                if folded > 0.0 {
                    per_kind.insert(kind.to_string(), folded * scale);
                }
            }
            if !per_kind.is_empty() {
                rates.insert(name.clone(), per_kind);
            }
        }

        // Kinds with no transport MT are carried entirely by the overlay.
        for curve in &self.moment_curves {
            if reaction_type_to_mt(&curve.kind).is_some() {
                continue;
            }
            let folded = fold_bin_maxima(&curve.energy, &curve.values, &self.union_grid, &s0);
            if folded > 0.0 {
                *rates
                    .entry(curve.parent.clone())
                    .or_default()
                    .entry(curve.kind.clone())
                    .or_insert(0.0) += folded * scale;
            }
        }

        rates
    }

    /// Spectrum weights folding each fissionable nuclide's tabulated fission
    /// yields against the flux this material actually saw (issue #379).
    ///
    /// Entry `k` of a nuclide's vector is the share of its fission rate that
    /// the linear interpolation assigns to tabulated point `k`, normalized to
    /// sum to one. Feeding these to the matrix builder is the commuted form of
    /// interpolating the yield vector at every collision energy: exact, since
    /// the interpolation is linear and so commutes with the sum over segments.
    ///
    /// Normalization is by the nuclide's own accumulated fission rate, so no
    /// particle count or volume enters and the result is independent of how the
    /// transport was chunked. Nuclides whose fission rate is zero are omitted:
    /// there is no fold to define, and the matrix builder only demands weights
    /// where the rate is non-zero.
    pub fn get_fission_yield_weights(&self, material_id: u32) -> FissionYieldWeights {
        let mut out: FissionYieldWeights = HashMap::new();

        let Some(mat_data) = self.materials.get(&material_id) else {
            return out;
        };
        if mat_data.fy_batch.is_empty() {
            return out;
        }
        let sums = mat_data.fy_sums.lock().unwrap_or_else(|p| p.into_inner());

        for (nuc_idx, name) in mat_data.nuclide_names.iter().enumerate() {
            let Some(ch) = &mat_data.fy_channels[nuc_idx] else {
                continue;
            };
            let slots = &sums[ch.offset..ch.offset + ch.energies.len()];
            let total: f64 = slots.iter().sum();
            if total <= 0.0 {
                continue;
            }
            out.insert(name.clone(), slots.iter().map(|s| s / total).collect());
        }

        out
    }

    /// Reset all accumulators for a new transmutation step.
    ///
    /// Clears sum_means, total_particles, and batch accumulators.
    /// Called between transmutation steps.
    pub fn reset(&self) {
        for mat_data in self.materials.values() {
            let mut sum_means = mat_data
                .sum_means
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            for (sum_mean, accum) in sum_means.iter_mut().zip(mat_data.batch_accum.iter()) {
                *sum_mean = 0.0;
                accum.store(0, Ordering::Relaxed);
            }

            mat_data.total_particles.store(0, Ordering::Relaxed);
            *mat_data
                .flux_sum_means
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = 0.0;
            mat_data.flux_batch_accum.store(0, Ordering::Relaxed);

            {
                let mut s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
                let mut s1 = mat_data.moment_s1.lock().unwrap_or_else(|p| p.into_inner());
                for g in 0..s0.len() {
                    s0[g] = 0.0;
                    s1[g] = 0.0;
                    mat_data.moment_s0_batch[g].store(0, Ordering::Relaxed);
                    mat_data.moment_s1_batch[g].store(0, Ordering::Relaxed);
                }
            }

            let mut psums = mat_data
                .yield_sums
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (ps, accum) in psums.iter_mut().zip(mat_data.yield_batch.iter()) {
                *ps = 0.0;
                accum.store(0, Ordering::Relaxed);
            }

            let mut fsums = mat_data
                .fy_sums
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (fs, accum) in fsums.iter_mut().zip(mat_data.fy_batch.iter()) {
                *fs = 0.0;
                accum.store(0, Ordering::Relaxed);
            }
        }
        if let Some(stats) = &self.statistics {
            stats.reset();
        }
    }

    /// Sum every rank's accumulators into a global total, on every rank
    /// (issue #287).
    ///
    /// Under MPI each rank transports its own share of the histories, so these
    /// accumulators (`sum_means`, the flux sum, the moment sums and the yield
    /// sums) are rank-local, while `total_particles` counts the GLOBAL per-chunk
    /// particle count on every rank (`Model::run_internal` passes the global
    /// `chunk_size`, matching how the ordinary tallies normalise). Dividing a
    /// rank-local numerator by the global denominator is what made
    /// `simulate_transmutation` report inventories low by exactly `1/n_ranks`.
    ///
    /// The reduction is an allreduce (reduce to root, then broadcast), not a
    /// reduce-to-root: every rank runs the same Bateman solve, so leaving
    /// non-root ranks with partial rates would just move the wrong answer
    /// somewhere less visible.
    ///
    /// One collective pair carries every material: the per-material buffers are
    /// packed in ascending material-id order, which also keeps the collective
    /// call order identical on every rank (a `HashMap` iteration order would
    /// not).
    pub fn reduce_across_ranks<C: CollectiveOps>(&self, mpi_ctx: &C) {
        if mpi_ctx.size() <= 1 {
            return;
        }
        let mut mat_ids: Vec<u32> = self.materials.keys().copied().collect();
        mat_ids.sort_unstable();

        // Pack: per material, sum_means ++ [flux_sum_means] ++ moment_s0 ++
        // moment_s1 ++ yield_sums ++ fy_sums, then the history statistics'
        // raw sums when they are on. Those are sums of per-history products,
        // so they add across ranks like everything else here.
        let mut packed: Vec<f64> = Vec::new();
        for id in &mat_ids {
            let mat_data = &self.materials[id];
            packed.extend_from_slice(
                &mat_data
                    .sum_means
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            );
            packed.push(
                *mat_data
                    .flux_sum_means
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()),
            );
            packed.extend_from_slice(
                &mat_data
                    .moment_s0
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            );
            packed.extend_from_slice(
                &mat_data
                    .moment_s1
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            );
            packed.extend_from_slice(
                &mat_data
                    .yield_sums
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            );
            packed.extend_from_slice(
                &mat_data
                    .fy_sums
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            );
            if let Some(stats) = &self.statistics {
                stats.pack(*id, &mut packed);
            }
        }

        mpi_ctx.reduce_sum_f64(&mut packed, 0);
        mpi_ctx.broadcast_f64(&mut packed, 0);

        // Unpack in the same order.
        let mut off = 0usize;
        for id in &mat_ids {
            let mat_data = &self.materials[id];
            {
                let mut sums = mat_data.sum_means.lock().unwrap_or_else(|p| p.into_inner());
                let n = sums.len();
                sums.copy_from_slice(&packed[off..off + n]);
                off += n;
            }
            {
                let mut flux = mat_data
                    .flux_sum_means
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                *flux = packed[off];
                off += 1;
            }
            {
                let mut s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
                let n = s0.len();
                s0.copy_from_slice(&packed[off..off + n]);
                off += n;
            }
            {
                let mut s1 = mat_data.moment_s1.lock().unwrap_or_else(|p| p.into_inner());
                let n = s1.len();
                s1.copy_from_slice(&packed[off..off + n]);
                off += n;
            }
            {
                let mut y = mat_data
                    .yield_sums
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                let n = y.len();
                y.copy_from_slice(&packed[off..off + n]);
                off += n;
            }
            {
                let mut f = mat_data.fy_sums.lock().unwrap_or_else(|p| p.into_inner());
                let n = f.len();
                f.copy_from_slice(&packed[off..off + n]);
                off += n;
            }
            if let Some(stats) = &self.statistics {
                off = stats.unpack(*id, &packed, off);
            }
        }
        debug_assert_eq!(off, packed.len(), "pack / unpack layout must agree");
    }

    /// Get the mean flux (track length per source particle) for a material.
    ///
    /// Returns mean TL / volume * source_rate = scalar flux [n/cm²/s]
    pub fn get_flux(&self, material_id: u32, volume: f64, source_rate: f64) -> f64 {
        let mat_data = match self.materials.get(&material_id) {
            Some(d) => d,
            None => return 0.0,
        };

        let total_particles = mat_data.total_particles.load(Ordering::Relaxed);
        let flux_sum_means = *mat_data
            .flux_sum_means
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if total_particles == 0 || volume <= 0.0 {
            return 0.0;
        }

        let mean_tl = flux_sum_means / total_particles as f64;
        let flux = mean_tl * source_rate / volume;

        #[cfg(feature = "debug_transmutation")]
        eprintln!(
            "[DEBUG transmutation] Material {} flux: mean_TL={:.6e} cm, scalar_flux={:.6e} n/cm²/s",
            material_id, mean_tl, flux
        );

        flux
    }

    /// Per-final-state isomeric partial rates for a material, normalized
    /// exactly like `get_reaction_rates` (`mean(partial·TL) / (volume·1e24) ·
    /// source_rate`, [1/s] per atom), so the `(n,n')` rate injected from these
    /// partials is consistent with the tallied totals. MF=10 partials are
    /// folded exactly from the union-grid flux moments (every chain parent);
    /// MF=9 yields come from the directly-scored channels (material nuclides).
    /// An isomer-only MF=10 partial on one of the material's own nuclides is
    /// its fold up to its last breakpoint plus the directly-scored share of
    /// the total above it (see `build_tail_channels`), one entry.
    /// Zero-rate targets are kept (a zero fraction is information: the flux
    /// never reached that state's threshold); kinds whose targets are all zero
    /// are kept too and resolved by the caller, which keeps the base split, or
    /// for isomers listed alone over a tallied total gives them nothing and the
    /// ground state the reaction. Returns an empty map when the material is
    /// unknown, nothing was scored, or no branching overlay is configured.
    pub fn get_partial_rates(
        &self,
        material_id: u32,
        volume: f64,
        source_rate: f64,
    ) -> PartialRates {
        let mut out: PartialRates = HashMap::new();

        let Some(mat_data) = self.materials.get(&material_id) else {
            return out;
        };
        let total_particles = mat_data.total_particles.load(Ordering::Relaxed);
        if total_particles == 0 || volume <= 0.0 {
            return out;
        }
        let scale = source_rate / (total_particles as f64 * volume * 1.0e24);

        let ysums = mat_data
            .yield_sums
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !self.moment_curves.is_empty() {
            let s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
            let s1 = mat_data.moment_s1.lock().unwrap_or_else(|p| p.into_inner());
            // Suffix sums of S0 make each curve's flat tail O(1).
            let mut suffix_s0 = vec![0.0; s0.len() + 1];
            for g in (0..s0.len()).rev() {
                suffix_s0[g] = suffix_s0[g + 1] + s0[g];
            }
            for (c, &tail) in self.moment_curves.iter().zip(&mat_data.tails) {
                let fold = |flat_tail: bool| {
                    fold_curve_from_moments(
                        &c.energy,
                        &c.values,
                        &self.union_grid,
                        &s0,
                        &s1,
                        &suffix_s0,
                        flat_tail,
                    )
                };
                let sum = match tail {
                    Tail::Flat => fold(true),
                    Tail::Scored(k) => fold(false) + k.map_or(0.0, |k| ysums[k]),
                };
                out.entry(c.parent.clone())
                    .or_default()
                    .entry(c.kind.clone())
                    .or_default()
                    .push((c.target.clone(), sum * scale));
            }
        }

        for (ch, &sum) in mat_data.yield_channels.iter().zip(ysums.iter()) {
            if ch.tail {
                continue; // its partial's, above
            }
            out.entry(ch.parent.clone())
                .or_default()
                .entry(ch.kind.clone())
                .or_default()
                .push((ch.target.clone(), sum * scale));
        }

        out
    }
}

/// Atomically add an f64 value to an AtomicU64 (stored as bits).
/// Uses compare-exchange loop for lock-free accumulation.
#[inline]
fn atomic_add_f64(atomic: &AtomicU64, value: f64) {
    let mut current = atomic.load(Ordering::Relaxed);
    loop {
        let current_f64 = f64::from_bits(current);
        let new_f64 = current_f64 + value;
        match atomic.compare_exchange_weak(
            current,
            new_f64.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mt_to_reaction_type() {
        assert_eq!(mt_to_reaction_type(102), Some("(n,gamma)"));
        // "fission" is the chain spelling, so it is the canonical rate key.
        assert_eq!(mt_to_reaction_type(18), Some("fission"));
        assert_eq!(mt_to_reaction_type(16), Some("(n,2n)"));
        assert_eq!(mt_to_reaction_type(999), None);
    }

    #[test]
    fn test_reaction_type_to_mt() {
        assert_eq!(reaction_type_to_mt("(n,gamma)"), Some(102));
        assert_eq!(reaction_type_to_mt("(n,fission)"), Some(18));
        assert_eq!(reaction_type_to_mt("unknown"), None);
    }

    #[test]
    fn test_atomic_add_f64() {
        let atomic = AtomicU64::new(0.0f64.to_bits());
        atomic_add_f64(&atomic, 1.5);
        atomic_add_f64(&atomic, 2.5);
        let result = f64::from_bits(atomic.load(Ordering::Relaxed));
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_transmutation_tallies() {
        let transmutable_cells: HashMap<u32, Vec<usize>> = HashMap::new();
        let materials: HashMap<u32, &Material> = HashMap::new();
        let chain: HashMap<String, ChainNuclide> = HashMap::new();

        let tallies = TransmutationTallies::new(
            &transmutable_cells,
            &materials,
            &chain,
            &BranchTable::new(),
            &HashMap::new(),
        );
        let rates = tallies.get_reaction_rates(1, 100.0, 1e10);
        assert!(rates.is_empty());
        assert!(tallies.get_partial_rates(1, 100.0, 1e10).is_empty());
    }

    /// Build a minimal one-material/one-MT tally by hand. The `score` hot path
    /// needs a full Material + chain, so this injects known sums directly into
    /// the accumulators, or drives `score` with synthetic curves.
    fn one_bin_tally() -> TransmutationTallies {
        one_bin_tally_with_branch(&BranchTable::new())
    }

    fn one_bin_tally_with_branch(branch: &BranchTable) -> TransmutationTallies {
        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
        for parent in branch.keys() {
            chain.insert(
                parent.clone(),
                ChainNuclide {
                    name: parent.clone(),
                    half_life: None,
                    decay_energy: 0.0,
                    reactions: vec![],
                    decays: vec![],
                    fission_yields: None,
                    sources: Vec::new(),
                    half_life_uncertainty: None,
                    decay_energy_uncertainty: None,
                    decay_energy_components: Default::default(),
                },
            );
        }
        let moment_curves = build_moment_curves(&chain, branch);
        let union_grid = build_union_grid(&moment_curves);
        let n_moment_bins = union_grid.len();
        let mut materials = HashMap::new();
        materials.insert(
            7u32,
            MaterialTransmutationData {
                nuclide_names: vec!["U238".to_string()],
                mt_numbers: vec![102], // (n,gamma)
                batch_accum: vec![AtomicU64::new(0)],
                sum_means: Mutex::new(vec![0.0]),
                total_particles: AtomicUsize::new(0),
                flux_batch_accum: AtomicU64::new(0),
                flux_sum_means: Mutex::new(0.0),
                moment_s0_batch: (0..n_moment_bins).map(|_| AtomicU64::new(0)).collect(),
                moment_s1_batch: (0..n_moment_bins).map(|_| AtomicU64::new(0)).collect(),
                moment_s0: Mutex::new(vec![0.0; n_moment_bins]),
                moment_s1: Mutex::new(vec![0.0; n_moment_bins]),
                yield_channels: Vec::new(),
                tails: vec![Tail::Flat; moment_curves.len()],
                yield_batch: Vec::new(),
                yield_sums: Mutex::new(Vec::new()),
                // The fixture tracks only (n,gamma), so nothing fissions.
                fy_channels: vec![None],
                fy_mt_idx: None,
                fy_batch: Vec::new(),
                fy_sums: Mutex::new(Vec::new()),
            },
        );
        TransmutationTallies {
            materials,
            union_grid,
            moment_curves,
            statistics: None,
        }
    }

    /// The CPU transport chunk count is statistics-neutral: splitting the same
    /// total `Sum(σ·TL)` over more `accumulate_batch` calls must not change the
    /// extracted reaction rate or flux (regression for issue #128, where the
    /// tally divided by the chunk count and came out N_chunks times low).
    #[test]
    fn reaction_rate_invariant_to_chunk_count() {
        let total_sigma_tl = 5.0_f64; // b·cm summed over ALL source particles
        let total_tl = 800.0_f64; // cm summed over ALL source particles
        let n_particles = 1000usize;
        let volume = 100.0_f64; // cm³
        let source_rate = 1.0e18_f64; // n/s

        let expected_rate = (total_sigma_tl / n_particles as f64 / (volume * 1.0e24)) * source_rate;
        let expected_flux = (total_tl / n_particles as f64) * source_rate / volume;

        // Drive the same totals through `chunks` equal accumulate_batch calls.
        let run = |chunks: usize| -> (f64, f64) {
            assert_eq!(n_particles % chunks, 0, "test requires even chunking");
            let t = one_bin_tally();
            let per_chunk_particles = n_particles / chunks;
            let per_chunk_sigma_tl = total_sigma_tl / chunks as f64;
            let per_chunk_tl = total_tl / chunks as f64;
            for _ in 0..chunks {
                let mat = t.materials.get(&7).unwrap();
                atomic_add_f64(&mat.batch_accum[0], per_chunk_sigma_tl);
                atomic_add_f64(&mat.flux_batch_accum, per_chunk_tl);
                t.accumulate_batch(per_chunk_particles);
            }
            let rate = t.get_reaction_rates(7, volume, source_rate)["U238"]["(n,gamma)"];
            let flux = t.get_flux(7, volume, source_rate);
            (rate, flux)
        };

        let (rate_1, flux_1) = run(1); // one chunk of 1000
        let (rate_10, flux_10) = run(10); // ten chunks of 100

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        assert!(
            rel(rate_1, expected_rate) < 1e-9,
            "1 chunk: {rate_1:e} vs {expected_rate:e}"
        );
        assert!(
            rel(rate_10, expected_rate) < 1e-9,
            "10 chunks: {rate_10:e} vs {expected_rate:e}"
        );
        assert!(
            rel(rate_1, rate_10) < 1e-12,
            "rate varies with chunk count: {rate_1:e} vs {rate_10:e}"
        );
        assert!(
            rel(flux_1, expected_flux) < 1e-9,
            "1 chunk flux: {flux_1:e} vs {expected_flux:e}"
        );
        assert!(
            rel(flux_10, expected_flux) < 1e-9,
            "10 chunks flux: {flux_10:e} vs {expected_flux:e}"
        );
        assert!(
            rel(flux_1, flux_10) < 1e-12,
            "flux varies with chunk count: {flux_1:e} vs {flux_10:e}"
        );
    }

    /// A branch table with two U238 (n,2n) partials whose curves exercise the
    /// full piecewise-linear machinery: a ramp with a threshold and a flat
    /// curve, plus a second parent so union-grid points interleave.
    fn moment_branch() -> BranchTable {
        let mut branch = BranchTable::new();
        branch.entry("U238".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![
                yani::BranchCurve {
                    target: "U237".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e6, 3.0e6, 5.0e6],
                    values: vec![0.0, 2.0, 1.0],
                    states: Default::default(),
                    normalisation: None,
                },
                yani::BranchCurve {
                    target: "U237_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e5, 3.0e6],
                    values: vec![0.5, 0.5],
                    states: Default::default(),
                    normalisation: None,
                },
            ],
        );
        // Second parent whose breakpoints interleave with U238's, so the
        // union grid subdivides U238's segments.
        branch.entry("Am241".to_string()).or_default().insert(
            "(n,gamma)".to_string(),
            vec![yani::BranchCurve {
                target: "Am242_m1".to_string(),
                quantity: BranchQuantity::CrossSection,
                energy: vec![5.0e5, 2.0e6, 4.0e6],
                values: vec![0.1, 0.3, 0.2],
                states: Default::default(),
                normalisation: None,
            }],
        );
        branch
    }

    /// The moment fold must reconstruct `sum(curve_interp(E_i) * TL_i)`
    /// exactly (to float roundoff) for arbitrary piecewise-linear curves and
    /// arbitrary segment energies: below threshold, mid-segment, exactly on
    /// breakpoints, between other curves' breakpoints, and in the flat tail
    /// above the last breakpoint.
    #[test]
    fn moment_fold_matches_brute_force() {
        let branch = moment_branch();
        let t = one_bin_tally_with_branch(&branch);
        let material = Material::new(
            HashMap::from([("U238".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        // (energy [eV], track length [cm]) segments covering every regime.
        let segments = [
            (5.0e4, 3.0), // below every threshold
            (1.0e5, 2.0), // exactly on the flat curve's first point
            (7.5e5, 4.0), // inside flat curve, below ramp threshold
            (1.0e6, 5.0), // exactly on the ramp threshold
            (1.7e6, 1.5), // mid ramp-up, mid Am segment
            (2.0e6, 2.5), // exactly on an Am breakpoint (foreign to U238)
            (2.9e6, 0.5), // just below a shared breakpoint
            (3.0e6, 1.0), // exactly on the ramp peak / flat curve end
            (4.5e6, 2.0), // ramp-down segment, above Am's last point
            (5.0e6, 3.5), // exactly on the ramp's last point
            (1.4e7, 7.0), // flat tails for everything
        ];
        // Split scoring over two chunks: the moment accumulators must be
        // chunk-count invariant like every other accumulator (issue #128).
        for &(e, tl) in &segments[..5] {
            t.score(7, e, tl, &material);
        }
        t.accumulate_batch(2);
        for &(e, tl) in &segments[5..] {
            t.score(7, e, tl, &material);
        }
        t.accumulate_batch(2);
        let n_particles = 4usize;

        let (volume, source_rate) = (2.0_f64, 1.0e12_f64);
        let partials = t.get_partial_rates(7, volume, source_rate);
        let scale = source_rate / (n_particles as f64 * volume * 1.0e24);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        let get = |parent: &str, kind: &str, target: &str| {
            partials[parent][kind]
                .iter()
                .find(|(t, _)| t == target)
                .map(|(_, r)| *r)
                .unwrap()
        };
        for (parent, kind, target, energy, values) in [
            (
                "U238",
                "(n,2n)",
                "U237",
                vec![1.0e6, 3.0e6, 5.0e6],
                vec![0.0, 2.0, 1.0],
            ),
            (
                "U238",
                "(n,2n)",
                "U237_m1",
                vec![1.0e5, 3.0e6],
                vec![0.5, 0.5],
            ),
            (
                "Am241",
                "(n,gamma)",
                "Am242_m1",
                vec![5.0e5, 2.0e6, 4.0e6],
                vec![0.1, 0.3, 0.2],
            ),
        ] {
            let brute: f64 = segments
                .iter()
                .map(|&(e, tl)| curve_interp(&energy, &values, e) * tl)
                .sum();
            let got = get(parent, kind, target);
            assert!(
                rel(got, brute * scale) < 1e-12,
                "{parent} {kind} -> {target}: moment fold {got:e} vs brute {:e}",
                brute * scale
            );
        }
    }

    /// Moment curves cover every chain parent in the branch table, not just
    /// the material's nuclides; MF=9-only kinds become per-material yield
    /// channels; malformed curves and (n,n') ground self-loops are dropped.
    #[test]
    fn curve_selection_policy() {
        let mut branch = BranchTable::new();
        branch.entry("Ag109".to_string()).or_default().insert(
            "(n,n')".to_string(),
            vec![
                yani::BranchCurve {
                    target: "Ag109".to_string(), // ground self-loop: dropped
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0, 1.0],
                    states: Default::default(),
                    normalisation: None,
                },
                yani::BranchCurve {
                    target: "Ag109_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0, 1.0],
                    states: Default::default(),
                    normalisation: None,
                },
            ],
        );
        branch.entry("Ag109".to_string()).or_default().insert(
            "(n,gamma)".to_string(),
            vec![
                yani::BranchCurve {
                    target: "Ag110".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: vec![1.0, 2.0],
                    values: vec![0.9, 0.9],
                    states: Default::default(),
                    normalisation: None,
                },
                yani::BranchCurve {
                    target: "Ag110_m1".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: vec![1.0, 2.0],
                    values: vec![0.1, 0.1],
                    states: Default::default(),
                    normalisation: None,
                },
            ],
        );
        // A kind not in REACTION_MT_MAP with only yields: dropped entirely.
        branch.entry("Ag109".to_string()).or_default().insert(
            "(n,unmapped)".to_string(),
            vec![yani::BranchCurve {
                target: "X".to_string(),
                quantity: BranchQuantity::Yield,
                energy: vec![1.0, 2.0],
                values: vec![1.0, 1.0],
                states: Default::default(),
                normalisation: None,
            }],
        );
        // Malformed curves must be dropped, well-formed sibling kept.
        branch.entry("Ag107".to_string()).or_default().insert(
            "(n,2n)".to_string(),
            vec![
                yani::BranchCurve {
                    target: "Ag106".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![],
                    values: vec![],
                    states: Default::default(),
                    normalisation: None,
                },
                yani::BranchCurve {
                    target: "Ag106_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0], // mismatched lengths
                    states: Default::default(),
                    normalisation: None,
                },
                yani::BranchCurve {
                    target: "Ag106_m2".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0, 1.0],
                    states: Default::default(),
                    normalisation: None,
                },
            ],
        );

        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
        for parent in ["Ag109", "Ag107"] {
            chain.insert(
                parent.to_string(),
                ChainNuclide {
                    name: parent.to_string(),
                    half_life: None,
                    decay_energy: 0.0,
                    reactions: vec![],
                    decays: vec![],
                    fission_yields: None,
                    sources: Vec::new(),
                    half_life_uncertainty: None,
                    decay_energy_uncertainty: None,
                    decay_energy_components: Default::default(),
                },
            );
        }

        let curves = build_moment_curves(&chain, &branch);
        let mut summary: Vec<(String, String, String)> = curves
            .iter()
            .map(|c| (c.parent.clone(), c.kind.clone(), c.target.clone()))
            .collect();
        summary.sort();
        assert_eq!(
            summary,
            vec![
                (
                    "Ag107".to_string(),
                    "(n,2n)".to_string(),
                    "Ag106_m2".to_string()
                ),
                (
                    "Ag109".to_string(),
                    "(n,n')".to_string(),
                    "Ag109_m1".to_string()
                ),
            ]
        );

        // Yield channels only exist for the material's own nuclides.
        let channels = build_yield_channels(&["Ag109".to_string()], &branch);
        let mut ysummary: Vec<(String, String, i32)> = channels
            .iter()
            .map(|c| (c.kind.clone(), c.target.clone(), c.mt))
            .collect();
        ysummary.sort();
        assert_eq!(
            ysummary,
            vec![
                ("(n,gamma)".to_string(), "Ag110".to_string(), 102),
                ("(n,gamma)".to_string(), "Ag110_m1".to_string(), 102),
            ]
        );
    }

    /// The bin-maximum fold must bound the continuous-energy sum it stands in
    /// for, and must be exact where the curve is flat across the bins the flux
    /// occupies (issue #404). A rate that could come out LOW is the one failure
    /// this cannot have: it would drop a nuclide the solve goes on to populate.
    #[test]
    fn bin_maxima_bound_the_continuous_energy_fold() {
        let grid = base_spectrum_grid();
        // Segments spread over the whole range, including the extremes.
        let segments = [
            (1.0e-3, 2.0),
            (1.0, 1.0),
            (5.5e3, 4.0),
            (1.0e6, 3.0),
            (2.7e6, 1.5),
            (1.4e7, 6.0),
            (2.9e7, 0.5),
        ];
        let mut s0 = vec![0.0; grid.len()];
        for &(e, tl) in &segments {
            let g = grid.partition_point(|&x| x <= e) - 1;
            s0[g] += tl;
        }

        for (label, energy, values) in [
            // Flat: the maximum IS the value, so the fold is exact.
            ("flat", vec![1.0e-5, 3.0e7], vec![2.5, 2.5]),
            // Steep threshold ramp, curve breakpoints far from the bin edges.
            ("ramp", vec![1.0e6, 1.0e7, 3.0e7], vec![0.0, 4.0, 1.0]),
            // Narrow spike between two bins, the case a bin average would miss.
            (
                "spike",
                vec![1.0e-5, 5.4e3, 5.5e3, 5.6e3, 3.0e7],
                vec![0.1, 0.1, 900.0, 0.1, 0.1],
            ),
        ] {
            let folded = fold_bin_maxima(&energy, &values, &grid, &s0);
            let exact: f64 = segments
                .iter()
                .map(|&(e, tl)| curve_interp(&energy, &values, e) * tl)
                .sum();
            assert!(
                folded >= exact * (1.0 - 1e-12),
                "{label}: fold {folded:e} must not undercut the exact sum {exact:e}"
            );
            if label == "flat" {
                let rel = (folded - exact).abs() / exact;
                assert!(rel < 1e-12, "flat curve must fold exactly, off by {rel:e}");
            }
        }
    }

    /// Every energy a transport segment can carry must land in a bin. An
    /// unscored segment is flux the fold never sees, which would make the
    /// bound an under-estimate for reasons nothing downstream could detect.
    #[test]
    fn spectrum_grid_bins_every_energy() {
        let grid = base_spectrum_grid();
        assert_eq!(grid[0], 0.0, "the first edge must admit any energy");
        for e in [0.0, 1.0e-11, 1.0e-5, 1.0, 1.4e7, 3.0e7, 1.0e9] {
            let g = grid.partition_point(|&x| x <= e).saturating_sub(1);
            assert!(g < grid.len(), "energy {e:e} fell outside the grid");
            assert!(grid[g] <= e, "energy {e:e} landed in a bin above it");
        }
        // Strictly ascending, so `partition_point` gives one bin per energy.
        assert!(grid.windows(2).all(|w| w[0] < w[1]));
    }

    // ---- Per-history statistics (issue #140, item 1) ----

    /// Deterministic pseudo-random histories: `(energy [eV], track length)`
    /// segments, log-uniform in energy from 1e-3 eV to 20 MeV so they cross
    /// many base bins and both halves of the moment branch's union grid.
    /// Every seventh history scores nothing, so the implicit zeros count.
    fn synthetic_histories(n: usize) -> Vec<Vec<(f64, f64)>> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|h| {
                if h % 7 == 3 {
                    return Vec::new();
                }
                let n_seg = 1 + (next() * 6.0) as usize;
                (0..n_seg)
                    .map(|_| {
                        let e = 10f64.powf(-3.0 + next() * (7.301 + 3.0));
                        (e, 0.1 + 3.0 * next())
                    })
                    .collect()
            })
            .collect()
    }

    fn bare_material() -> Material {
        Material::new(
            HashMap::from([("U238".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap()
    }

    fn run_histories(t: &TransmutationTallies, histories: &[Vec<(f64, f64)>], material: &Material) {
        for h in histories {
            for &(e, tl) in h {
                t.score(7, e, tl, material);
            }
            t.finish_history();
        }
    }

    /// Each history's vector, computed directly from its segments: the
    /// reference the accumulator has to reproduce. The bare material scores
    /// no rates and there are no yield channels, so only the spectrum part is
    /// non-zero; the fixture's one (U238, MT 102) rate entry stays zero.
    fn reference_vectors(histories: &[Vec<(f64, f64)>]) -> Vec<Vec<f64>> {
        let grid = base_spectrum_grid();
        histories
            .iter()
            .map(|h| {
                let mut x = vec![0.0; grid.len() + 1];
                for &(e, tl) in h {
                    x[grid.partition_point(|&g| g <= e) - 1] += tl;
                }
                x
            })
            .collect()
    }

    fn stats_tally(branch: &BranchTable, workers: usize) -> TransmutationTallies {
        let t = one_bin_tally_with_branch(branch).with_history_statistics();
        t.prepare_history_workers(workers).unwrap();
        t
    }

    /// The accumulated mean and covariance must equal a two-pass computation
    /// over the full per-history vectors, zeros of the empty histories
    /// included. The moment branch makes the union grid finer than the base
    /// grid, so this also checks the fine-to-coarse bin map.
    #[test]
    fn history_covariance_matches_two_pass() {
        let histories = synthetic_histories(400);
        let t = stats_tally(&moment_branch(), 1);
        let material = bare_material();
        run_histories(&t, &histories, &material);
        t.accumulate_batch(histories.len());

        let cov = t.history_covariance(7).expect("statistics are on");
        let xs = reference_vectors(&histories);
        let n = xs.len() as f64;
        let dim = xs[0].len();
        assert_eq!(cov.dim(), dim, "base bins, no yields, one rate entry");
        assert_eq!(cov.rate_channels, vec![("U238".to_string(), 102)]);
        assert_eq!(cov.n_histories, histories.len() as u64);

        let mean: Vec<f64> = (0..dim)
            .map(|i| xs.iter().map(|x| x[i]).sum::<f64>() / n)
            .collect();
        let touched: Vec<usize> = (0..dim).filter(|&i| mean[i] != 0.0).collect();
        assert!(touched.len() > 50, "histories should spread over the grid");
        for &i in &touched {
            let rel = (cov.mean[i] - mean[i]).abs() / mean[i].abs();
            assert!(rel < 1e-12, "mean[{i}] off by {rel:e}");
            for &j in &touched {
                let two_pass: f64 = xs
                    .iter()
                    .map(|x| (x[i] - mean[i]) * (x[j] - mean[j]))
                    .sum::<f64>()
                    / (n - 1.0);
                let scale = (cov.covariance(i, i) * cov.covariance(j, j)).sqrt();
                let err = (cov.covariance(i, j) - two_pass).abs();
                assert!(
                    err <= 1e-9 * scale.max(1e-300),
                    "cov[{i}][{j}] = {:e}, two-pass {two_pass:e}",
                    cov.covariance(i, j)
                );
            }
        }
        // A bin nothing reached has no mean and no covariance.
        let untouched = (0..dim).find(|i| mean[*i] == 0.0).unwrap();
        assert_eq!(cov.mean[untouched], 0.0);
        assert_eq!(cov.covariance(untouched, touched[0]), 0.0);

        // The quadratic form is the variance of the folded total, which here
        // is the total track length per history.
        let mut w = vec![0.0; dim];
        w[..cov.n_bins()].fill(1.0);
        let (total, var) = cov.linear_combination(&w);
        let totals: Vec<f64> = xs.iter().map(|x| x[..cov.n_bins()].iter().sum()).collect();
        let t_mean = totals.iter().sum::<f64>() / n;
        let t_var = totals.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / (n - 1.0) / n;
        assert!((total - t_mean).abs() < 1e-12 * t_mean);
        assert!((var - t_var).abs() < 1e-9 * t_var, "{var:e} vs {t_var:e}");
    }

    /// The coarse track lengths are sums of the fine ones, not approximations
    /// of them: summing the union-grid means over each base bin must give the
    /// statistics' means.
    #[test]
    fn coarse_track_lengths_are_sums_of_the_fine_ones() {
        let histories = synthetic_histories(200);
        let t = stats_tally(&moment_branch(), 1);
        run_histories(&t, &histories, &bare_material());
        t.accumulate_batch(histories.len());

        let cov = t.history_covariance(7).unwrap();
        let base = base_spectrum_grid();
        let mat = &t.materials[&7];
        let s0 = mat.moment_s0.lock().unwrap();
        let n = histories.len() as f64;
        let mut coarse = vec![0.0; base.len()];
        for (g, &edge) in t.union_grid.iter().enumerate() {
            coarse[base.partition_point(|&b| b <= edge) - 1] += s0[g];
        }
        for (c, fine) in coarse.iter().enumerate() {
            let expected = fine / n;
            assert!(
                (cov.mean[c] - expected).abs() <= 1e-10 * expected.abs().max(1e-300),
                "bin {c}: {} vs {expected:e}",
                cov.mean[c]
            );
        }
    }

    /// Off by default, and turning it on must not move a single mean by one
    /// bit: the statistics sit beside the sums, never in them.
    #[test]
    fn history_statistics_leave_the_means_bit_identical() {
        let histories = synthetic_histories(150);
        let material = bare_material();
        let off = one_bin_tally_with_branch(&moment_branch());
        assert!(!off.has_history_statistics());
        let on = stats_tally(&moment_branch(), 1);
        for t in [&off, &on] {
            run_histories(t, &histories, &material);
            t.accumulate_batch(histories.len());
        }
        assert!(off.history_covariance(7).is_none());

        let bits = |t: &TransmutationTallies| {
            let mut v: Vec<u64> = Vec::new();
            let mut partial: Vec<(String, String, String, f64)> = t
                .get_partial_rates(7, 2.0, 1.0e12)
                .into_iter()
                .flat_map(|(p, kinds)| {
                    kinds.into_iter().flat_map(move |(k, targets)| {
                        let p = p.clone();
                        targets
                            .into_iter()
                            .map(move |(tg, r)| (p.clone(), k.clone(), tg, r))
                    })
                })
                .collect();
            partial.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
            v.extend(partial.iter().map(|x| x.3.to_bits()));
            v.push(t.get_flux(7, 2.0, 1.0e12).to_bits());
            let mat = &t.materials[&7];
            v.extend(mat.moment_s0.lock().unwrap().iter().map(|x| x.to_bits()));
            v.extend(mat.moment_s1.lock().unwrap().iter().map(|x| x.to_bits()));
            v
        };
        assert_eq!(bits(&off), bits(&on));
    }

    /// Histories spread over worker threads must give the statistics one
    /// worker gives, whichever thread ran which history.
    #[test]
    fn history_statistics_are_thread_invariant() {
        use rayon::prelude::*;
        let histories = synthetic_histories(600);
        let material = bare_material();

        let serial = stats_tally(&moment_branch(), 1);
        run_histories(&serial, &histories, &material);
        serial.accumulate_batch(histories.len());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = stats_tally(&moment_branch(), 4);
        pool.install(|| {
            histories.par_iter().for_each(|h| {
                for &(e, tl) in h {
                    parallel.score(7, e, tl, &material);
                }
                parallel.finish_history();
            })
        });
        parallel.accumulate_batch(histories.len());

        let a = serial.history_covariance(7).unwrap();
        let b = parallel.history_covariance(7).unwrap();
        for i in 0..a.dim() {
            assert!((a.mean[i] - b.mean[i]).abs() <= 1e-12 * a.mean[i].abs());
            for j in i..a.dim() {
                let scale = (a.covariance(i, i) * a.covariance(j, j)).sqrt();
                assert!(
                    (a.covariance(i, j) - b.covariance(i, j)).abs() <= 1e-9 * scale.max(1e-300),
                    "cov[{i}][{j}]: {:e} vs {:e}",
                    a.covariance(i, j),
                    b.covariance(i, j)
                );
            }
        }
    }

    /// A stand-in for MPI across two ranks: `reduce_sum_f64` adds the other
    /// rank's packed buffer, which is what the allreduce leaves on root.
    struct TwoRanks {
        other: Mutex<Vec<f64>>,
        record: bool,
    }

    impl CollectiveOps for TwoRanks {
        fn size(&self) -> i32 {
            2
        }
        fn reduce_sum_f64(&self, local: &mut [f64], _root: i32) {
            let mut other = self.other.lock().unwrap();
            if self.record {
                *other = local.to_vec();
            } else {
                for (l, o) in local.iter_mut().zip(other.iter()) {
                    *l += o;
                }
            }
        }
        fn broadcast_f64(&self, _data: &mut [f64], _root: i32) {}
    }

    /// Histories split across two ranks and reduced must give the statistics
    /// of one rank running them all. Each rank counts the global particle
    /// total, as `Model::run_internal` passes it.
    #[test]
    fn history_statistics_reduce_across_ranks() {
        let histories = synthetic_histories(300);
        let material = bare_material();
        let (first, second) = histories.split_at(137);

        let whole = stats_tally(&moment_branch(), 1);
        run_histories(&whole, &histories, &material);
        whole.accumulate_batch(histories.len());

        let rank0 = stats_tally(&moment_branch(), 1);
        let rank1 = stats_tally(&moment_branch(), 1);
        run_histories(&rank0, first, &material);
        run_histories(&rank1, second, &material);
        rank0.accumulate_batch(histories.len());
        rank1.accumulate_batch(histories.len());

        let ops = TwoRanks {
            other: Mutex::new(Vec::new()),
            record: true,
        };
        rank1.reduce_across_ranks(&ops);
        let ops = TwoRanks {
            other: ops.other,
            record: false,
        };
        rank0.reduce_across_ranks(&ops);

        let a = whole.history_covariance(7).unwrap();
        let b = rank0.history_covariance(7).unwrap();
        assert_eq!(a.n_histories, b.n_histories);
        for i in 0..a.dim() {
            assert!((a.mean[i] - b.mean[i]).abs() <= 1e-12 * a.mean[i].abs());
            for j in i..a.dim() {
                let scale = (a.covariance(i, i) * a.covariance(j, j)).sqrt();
                assert!(
                    (a.covariance(i, j) - b.covariance(i, j)).abs() <= 1e-9 * scale.max(1e-300)
                );
            }
        }
    }

    /// `reset` starts a new step from nothing, and a later run on more
    /// threads than the scratch was sized for is refused rather than
    /// scored into a slot that does not exist.
    #[test]
    fn history_statistics_reset_and_worker_sizing() {
        let histories = synthetic_histories(50);
        let t = stats_tally(&BranchTable::new(), 2);
        run_histories(&t, &histories, &bare_material());
        t.accumulate_batch(histories.len());
        assert!(t
            .history_covariance(7)
            .unwrap()
            .mean
            .iter()
            .any(|m| *m != 0.0));

        t.reset();
        let cleared = t.history_covariance(7).unwrap();
        assert_eq!(cleared.n_histories, 0);
        assert!(cleared.mean.iter().all(|m| *m == 0.0));

        assert!(t.prepare_history_workers(2).is_ok());
        assert!(t.prepare_history_workers(1).is_ok());
        let err = t.prepare_history_workers(3).unwrap_err();
        assert!(err.contains("sized for 2"), "{err}");
    }

    /// The folded rate covariance against the covariance of the rates
    /// themselves, accumulated history by history. The MF=10 partials of the
    /// moment branch are exact continuous-energy curves, so each history's
    /// own contribution to every partial rate can be computed directly and the
    /// two-pass covariance of those is the reference. The fold only sees the
    /// spectrum per base bin, so agreement is to the size of a cross section's
    /// change across one bin, not to rounding.
    #[test]
    fn rate_covariance_matches_the_per_history_rates() {
        let branch = moment_branch();
        let histories = synthetic_histories(4000);
        let t = stats_tally(&branch, 1);
        let material = bare_material();
        run_histories(&t, &histories, &material);
        t.accumulate_batch(histories.len());

        let (volume, source_rate) = (2.0, 1.0e12);
        let rc = t
            .get_reaction_rate_covariance(7, volume, source_rate)
            .expect("statistics are on");
        let per_mean = source_rate / (volume * 1.0e24);
        let n = histories.len() as f64;

        // The rates are the ones the accessor reports.
        let partials = t.get_partial_rates(7, volume, source_rate);
        for (i, l) in rc.labels.iter().enumerate() {
            let target = l.target.as_deref().unwrap();
            let expected = partials[&l.nuclide][&l.kind]
                .iter()
                .find(|(tg, _)| tg == target)
                .unwrap()
                .1;
            assert!(
                (rc.rates[i] - expected).abs() <= 1e-12 * expected.abs(),
                "{l:?}: {} vs {expected}",
                rc.rates[i]
            );
        }

        // Each history's own score in every partial.
        let curves: Vec<(&[f64], &[f64])> = t
            .moment_curves
            .iter()
            .map(|c| (c.energy.as_slice(), c.values.as_slice()))
            .collect();
        let x: Vec<Vec<f64>> = histories
            .iter()
            .map(|h| {
                curves
                    .iter()
                    .map(|(e, v)| h.iter().map(|&(en, tl)| curve_interp(e, v, en) * tl).sum())
                    .collect()
            })
            .collect();
        let m = curves.len();
        let mean: Vec<f64> = (0..m)
            .map(|i| x.iter().map(|r| r[i]).sum::<f64>() / n)
            .collect();
        let reference = |i: usize, j: usize| {
            x.iter()
                .map(|r| (r[i] - mean[i]) * (r[j] - mean[j]))
                .sum::<f64>()
                / (n - 1.0)
                / n
                * per_mean
                * per_mean
        };
        for i in 0..m {
            for j in i..m {
                let r = reference(i, j);
                let scale = (reference(i, i) * reference(j, j)).sqrt();
                let err = (rc.covariance(i, j) - r).abs() / scale;
                assert!(
                    err < 0.01,
                    "cov[{i}][{j}]: {:e} vs {r:e}",
                    rc.covariance(i, j)
                );
            }
        }
    }

    /// Off by default here too: no statistics, no covariance.
    #[test]
    fn rate_covariance_needs_history_statistics() {
        let t = one_bin_tally_with_branch(&moment_branch());
        assert!(t.get_reaction_rate_covariance(7, 2.0, 1.0e12).is_none());
    }

    /// In115 carrying one transport reaction, (n,2n), at 294 K.
    fn indium_n2n(energy: Vec<f64>, cross_section: Vec<f64>) -> Material {
        use std::sync::Arc;
        let reaction = yamc_nuclide::reaction::Reaction {
            cross_section: cross_section.into(),
            threshold_idx: 1,
            energy: energy.into(),
            mt_number: 16,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };
        let temperature = "294".to_string();
        let nuclide = yamc_nuclide::nuclide::Nuclide {
            name: Some("In115".to_string()),
            element: None,
            atomic_symbol: Some("In".to_string()),
            atomic_number: Some(49),
            neutron_number: Some(66),
            mass_number: Some(115),
            atomic_weight_ratio: Some(113.9),
            library: None,
            energy: None,
            reactions: vec![HashMap::from([(16, Arc::new(reaction))])],
            fissionable: false,
            available_temperatures: vec![temperature.clone()],
            loaded_temperatures: vec![temperature.clone()],
            data_path: None,
            fission_nu: None,
            fast_xs: vec![],
            urr_data: vec![],
            urr_present: false,
            fission_photon_release: None,
            covariance: None,
            elastic_flat_cache: Default::default(),
            fission_chi_flat_cache: Default::default(),
            delayed_neutron_cache: Default::default(),
            inelastic_angle_flat_cache: Default::default(),
            load_scope: Default::default(),
        };
        let mut m = Material::new(
            HashMap::from([("In115".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        m.set_temperature(&temperature);
        m.nuclide_data
            .insert("In115".to_string(), Arc::new(nuclide));
        m
    }

    /// The In115 (n,2n) chain with the ground at 1.0 and the isomer grafted at
    /// 0.0, and an overlay listing In114_m1 alone, from 1 b at 10 MeV to 1.6 b
    /// at 20 MeV, or with `ground` In114 too, from 1 b to 0.4 b, a full list.
    fn isomer_only_n2n(ground: bool) -> (HashMap<String, ChainNuclide>, BranchTable) {
        let edge = |target: &str, branching: f64| yani::ChainReaction {
            kind: "(n,2n)".to_string(),
            target: Some(target.to_string()),
            branching,
            q_value: None,
        };
        let mut chain: HashMap<String, ChainNuclide> = HashMap::new();
        chain.insert(
            "In115".to_string(),
            ChainNuclide {
                name: "In115".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![edge("In114", 1.0), edge("In114_m1", 0.0)],
                decays: vec![],
                fission_yields: None,
                sources: Vec::new(),
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
                decay_energy_components: Default::default(),
            },
        );
        let curve = |target: &str, values: Vec<f64>| yani::BranchCurve {
            target: target.to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: vec![1.0e7, 2.0e7],
            values,
        };
        let mut curves = vec![curve("In114_m1", vec![1.0, 1.6])];
        if ground {
            curves.push(curve("In114", vec![1.0, 0.4]));
        }
        let mut branch = BranchTable::new();
        branch
            .entry("In115".to_string())
            .or_default()
            .insert("(n,2n)".to_string(), curves);
        (chain, branch)
    }

    fn one_material_tally(
        material: &Material,
        chain: &HashMap<String, ChainNuclide>,
        branch: &BranchTable,
    ) -> TransmutationTallies {
        TransmutationTallies::new(
            &HashMap::from([(7u32, vec![0usize])]),
            &HashMap::from([(7u32, material)]),
            chain,
            branch,
            &HashMap::new(),
        )
    }

    /// An isomer-only MF=10 partial on a material nuclide is its moment fold up
    /// to its last breakpoint plus, above it, the scored share of the transport
    /// total it ends on, and the split is that over the tallied total. Here the
    /// partial ends at 20 MeV on 1.6 b, 0.8 of a total that falls from 2 b to
    /// nothing at 30 MeV. The two parts are reported as one partial, with one
    /// covariance row: the fold's weights stopping at 20 MeV, and the tail's
    /// own history entry.
    #[test]
    fn an_isomer_only_partial_scores_its_tail_as_a_share_of_the_total() {
        let (xs_e, xs) = (vec![1.0e7, 2.0e7, 3.0e7], vec![2.0, 2.0, 0.0]);
        let material = indium_n2n(xs_e.clone(), xs.clone());
        let (chain, branch) = isomer_only_n2n(false);
        let (p_e, p) = (vec![1.0e7, 2.0e7], vec![1.0, 1.6]);
        let t = one_material_tally(&material, &chain, &branch).with_history_statistics();
        t.prepare_history_workers(1).unwrap();

        // Histories spread over 10 to 35 MeV: below the partial's last point,
        // above it, and above the total's own grid, where it is flat at zero.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let histories: Vec<Vec<(f64, f64)>> = (0..3000)
            .map(|_| {
                (0..1 + (next() * 4.0) as usize)
                    .map(|_| (1.0e7 + 2.5e7 * next(), 0.1 + next()))
                    .collect()
            })
            .collect();
        for h in &histories {
            for &(e, tl) in h {
                t.score(7, e, tl, &material);
            }
            t.finish_history();
        }
        t.accumulate_batch(histories.len());

        let (volume, source_rate) = (2.0, 1.0e12);
        let n = histories.len() as f64;
        let scale = source_rate / (n * volume * 1.0e24);
        let sigma = |e: f64| curve_interp(&xs_e, &xs, e);
        let below = |e: f64| {
            if e < 2.0e7 {
                curve_interp(&p_e, &p, e)
            } else {
                0.0
            }
        };
        let above = |e: f64| if e >= 2.0e7 { 0.8 * sigma(e) } else { 0.0 };
        let per_history = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
            histories
                .iter()
                .map(|h| h.iter().map(|&(e, tl)| f(e) * tl).sum())
                .collect()
        };
        let (x_below, x_above, x_total) = (
            per_history(&below),
            per_history(&above),
            per_history(&sigma),
        );
        let x_partial: Vec<f64> = x_below.iter().zip(&x_above).map(|(b, a)| b + a).collect();
        let sum = |x: &[f64]| x.iter().sum::<f64>() * scale;

        let partials = t.get_partial_rates(7, volume, source_rate);
        let parts = &partials["In115"]["(n,2n)"];
        assert_eq!(parts.len(), 1, "the fold and the tail, as one: {parts:?}");
        assert_eq!(parts[0].0, "In114_m1");
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        let want = sum(&x_below) + sum(&x_above);
        assert!(rel(parts[0].1, want) < 1e-12, "{parts:?} vs {want}");

        let mut rates = t.get_reaction_rates(7, volume, source_rate);
        let total = rates["In115"]["(n,2n)"];
        assert!(rel(total, sum(&x_total)) < 1e-12);
        let folded =
            crate::apply_coupled_branching(&std::sync::Arc::new(chain), &partials, &mut rates);
        let share = folded["In115"]
            .reactions
            .iter()
            .find(|r| r.target.as_deref() == Some("In114_m1"))
            .unwrap()
            .branching;
        let want = sum(&x_partial) / sum(&x_total);
        assert!(rel(share, want) < 1e-12, "{share} vs {want}");

        // One row under the label, with the rate the accessor reports and the
        // partial's own variance, to within what one base bin hides: 20 MeV is
        // not a base-grid edge, so the fold's step to zero there sits inside a
        // bin. The tail is a history entry of its own, so its variance is
        // exact.
        let rc = t
            .get_reaction_rate_covariance(7, volume, source_rate)
            .expect("statistics are on");
        let at: Vec<usize> = (0..rc.len())
            .filter(|&i| rc.labels[i].target.as_deref() == Some("In114_m1"))
            .collect();
        assert_eq!(at.len(), 1);
        assert!(rel(rc.rates[at[0]], parts[0].1) < 1e-12);
        let per_mean = source_rate / (volume * 1.0e24);
        let variance = |x: &[f64]| {
            let mean = x.iter().sum::<f64>() / n;
            x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0) / n
                * per_mean
                * per_mean
        };
        let (got, want) = (rc.covariance(at[0], at[0]), variance(&x_partial));
        assert!(rel(got, want) < 0.05, "variance: {got:e} vs {want:e}");
        let moments = t.history_covariance(7).expect("statistics are on");
        let k = moments
            .yield_channels
            .iter()
            .position(|(_, _, target)| target == "In114_m1")
            .expect("the tail's entry");
        let i = moments.yield_index(k);
        let (got, want) = (
            moments.covariance_of_mean(i, i) * per_mean * per_mean,
            variance(&x_above),
        );
        assert!(rel(got, want) < 1e-12, "tail variance: {got:e} vs {want:e}");
    }

    /// A collision at exactly the partial's last breakpoint, 20 MeV, where
    /// every ENDF/B-VIII.1 and JEFF-4.0 isomer-only partial ends, is the
    /// tail's, at the share the partial ends on, 1.6 b of 2 b. It used to fall
    /// between the fold, which stops short of it, and the tail, which read zero
    /// there, so a 20 MeV source made no isomer from its uncollided flux. A
    /// full list, the ground listed too, gives the same 0.8 there, and the
    /// split is continuous across the breakpoint.
    #[test]
    fn an_isomer_only_partial_counts_a_collision_at_its_last_breakpoint() {
        let material = indium_n2n(vec![1.0e7, 2.0e7, 3.0e7], vec![2.0, 2.0, 0.0]);
        let isomer_share = |ground: bool, e: f64| {
            let (chain, branch) = isomer_only_n2n(ground);
            let t = one_material_tally(&material, &chain, &branch);
            t.score(7, e, 1.0, &material);
            t.accumulate_batch(1);
            let partials = t.get_partial_rates(7, 1.0, 1.0);
            let mut rates = t.get_reaction_rates(7, 1.0, 1.0);
            let folded =
                crate::apply_coupled_branching(&std::sync::Arc::new(chain), &partials, &mut rates);
            folded["In115"]
                .reactions
                .iter()
                .find(|r| r.target.as_deref() == Some("In114_m1"))
                .unwrap()
                .branching
        };
        for e in [2.0e7_f64.next_down(), 2.0e7, 2.0e7_f64.next_up()] {
            let got = isomer_share(false, e);
            assert!((got - 0.8).abs() < 1e-12, "at {e:e} eV: {got}");
        }
        let full = isomer_share(true, 2.0e7);
        assert!((full - 0.8).abs() < 1e-12, "full list: {full}");
    }
}
