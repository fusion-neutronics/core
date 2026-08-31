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
/// the material's own nuclides.
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

use super::material_transmute::curve_interp;
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

/// An MF=9 yield channel scored directly at the collision energy (issue #218):
/// the yield times the parent's transport cross section, `y_s(E) * sigma_MT(E)`.
/// Only built for the material's own nuclides (the lookup needs their loaded
/// cross sections; parents outside the material keep their base split, as the
/// multigroup fold also did).
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
/// point, flat above the last (the `curve_interp` conventions).
struct MomentCurve {
    parent: String,
    kind: String,
    target: String,
    energy: Vec<f64>,
    values: Vec<f64>,
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
    /// MF=9 yield channels scored at the collision energy; empty when no
    /// branching overlay is configured.
    yield_channels: Vec<YieldChannel>,
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
                });
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
                    });
                }
            }
        }
    }
    channels
}

/// Exactly reconstruct `sum(sigma(E_i) * TL_i)` for a piecewise-linear curve
/// from the union-grid flux moments: within a bin every curve is linear, so
/// the sum is `sigma(edge) * S0 + slope * S1` with `S1` relative to the edge.
/// `suffix_s0[g]` is `sum(s0[g..])`, used for the flat tail above the curve's
/// last breakpoint in O(1).
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
    if v_last != 0.0 {
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

            let yield_channels = build_yield_channels(&nuclide_names, branch);
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
        }
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

        // Direct continuous-energy scoring of the MF=9 yield channels at the
        // collision energy (issue #218): the yield weighted by the parent's
        // transport cross section, using the identical σ(E)·TL machinery as
        // the totals above so a reaction's final-state partials and its total
        // see the same flux.
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

            // Accumulate per-channel MF=9 yields
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
        // moment_s1 ++ yield_sums ++ fy_sums.
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
    /// Zero-rate targets are kept (a zero fraction is information: the flux
    /// never reached that state's threshold); kinds whose targets are all zero
    /// are kept too and resolved by the caller (base split preserved). Returns
    /// an empty map when the material is unknown, nothing was scored, or no
    /// branching overlay is configured.
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

        if !self.moment_curves.is_empty() {
            let s0 = mat_data.moment_s0.lock().unwrap_or_else(|p| p.into_inner());
            let s1 = mat_data.moment_s1.lock().unwrap_or_else(|p| p.into_inner());
            // Suffix sums of S0 make each curve's flat tail O(1).
            let mut suffix_s0 = vec![0.0; s0.len() + 1];
            for g in (0..s0.len()).rev() {
                suffix_s0[g] = suffix_s0[g + 1] + s0[g];
            }
            for c in &self.moment_curves {
                let sum = fold_curve_from_moments(
                    &c.energy,
                    &c.values,
                    &self.union_grid,
                    &s0,
                    &s1,
                    &suffix_s0,
                );
                out.entry(c.parent.clone())
                    .or_default()
                    .entry(c.kind.clone())
                    .or_default()
                    .push((c.target.clone(), sum * scale));
            }
        }

        let ysums = mat_data
            .yield_sums
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for (ch, &sum) in mat_data.yield_channels.iter().zip(ysums.iter()) {
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
                },
                yani::BranchCurve {
                    target: "U237_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0e5, 3.0e6],
                    values: vec![0.5, 0.5],
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
                },
                yani::BranchCurve {
                    target: "Ag109_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0, 1.0],
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
                },
                yani::BranchCurve {
                    target: "Ag110_m1".to_string(),
                    quantity: BranchQuantity::Yield,
                    energy: vec![1.0, 2.0],
                    values: vec![0.1, 0.1],
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
                },
                yani::BranchCurve {
                    target: "Ag106_m1".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0], // mismatched lengths
                },
                yani::BranchCurve {
                    target: "Ag106_m2".to_string(),
                    quantity: BranchQuantity::CrossSection,
                    energy: vec![1.0, 2.0],
                    values: vec![1.0, 1.0],
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
}
