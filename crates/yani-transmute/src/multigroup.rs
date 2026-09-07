/// Multigroup flux collapse and reaction rate computation.
///
/// Computes effective one-group cross sections and reaction rates from a
/// user-provided multigroup flux spectrum, enabling standalone transmutation
/// without re-running transport at each timestep.
use std::collections::HashMap;
use std::error::Error;

use yamc_materials::material::Material;
use yamc_nuclide::group_structures::get_group_structure;
use yamc_nuclide::reaction::Reaction;
use yani::{ChainNuclide, FissionYieldSet, FissionYieldWeights, ReactionRates};

use super::reaction_type_to_mt;
use crate::self_shielding::{flux_shape, mixture_total, FluxShape, Shielding, ShieldingInfo};

/// MT for total fission, whose rate distribution weights the yield fold.
const MT_FISSION: i32 = 18;

/// Specification of energy group boundaries.
pub enum EnergyGroups {
    /// A named group structure (e.g., "VITAMIN-J-175")
    Named(String),
    /// Explicit group boundaries in eV (ascending order, len = n_groups + 1)
    Explicit(Vec<f64>),
}

impl EnergyGroups {
    /// Resolve to concrete energy group boundaries in eV.
    pub fn resolve(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        match self {
            EnergyGroups::Named(name) => {
                let boundaries = get_group_structure(name)?;
                Ok(boundaries.to_vec())
            }
            EnergyGroups::Explicit(boundaries) => {
                if boundaries.len() < 2 {
                    return Err("Energy group boundaries must have at least 2 values".into());
                }
                // Verify ascending order
                for w in boundaries.windows(2) {
                    if w[1] <= w[0] {
                        return Err(format!(
                            "Energy group boundaries must be in ascending order, \
                             found {} followed by {}",
                            w[0], w[1]
                        )
                        .into());
                    }
                }
                Ok(boundaries.clone())
            }
        }
    }
}

/// The interior evaluation points of one group: `{e in grid : e_lo < e < e_hi}`.
///
/// The reaction's cross section is linear between its own grid points, so a
/// group integral that evaluates at all of them is exact for the cross section
/// alone. Which points those are used to be found by scanning the whole grid
/// per group and sorting what came back; the grid ascends -- `cross_section_at`
/// already binary-searches it, and the Arrow reader enforces it at load -- so
/// the two predicates that scan filtered on are a pair of bisections and the
/// answer is a contiguous sub-slice, already sorted and needing no `Vec`.
///
/// That is `O(grid_len)` per group replaced by `O(log grid_len)`. On CCFE-709
/// against an 18,000-point grid it is 1.3e7 comparisons per reaction against
/// 1.4e4, and it removes the per-group allocation and sort with them
/// (issue #576, finding 1a).
///
/// **Runs of equal energies are kept**, where the old `Vec::dedup` collapsed
/// them. ENDF spells a step discontinuity as a repeated energy, and a repeated
/// energy makes a zero-width trapezoid segment, which is bit-neutral at every
/// call site: [`group_averaged_xs`] adds it as exactly `+0.0` to a finite
/// accumulator, and every other caller already skips it with a `de <= 0.0`
/// guard. The non-degenerate segments, their end points, the values at them and
/// the order they are summed in are all unchanged, which is what makes this
/// bit-identical rather than merely equivalent.
fn interior(grid: &[f64], e_lo: f64, e_hi: f64) -> &[f64] {
    debug_assert!(
        grid.windows(2).all(|w| w[0] <= w[1]),
        "an evaluation grid must ascend"
    );
    interior_from(grid, e_lo, e_hi).1
}

/// As [`interior`], and where in `grid` the slice starts.
///
/// The index is what lets [`walk_group`] read the cross section straight out of
/// `reaction.cross_section` instead of bisecting `reaction.energy` for a
/// position it already has (issue #576, finding 1b).
fn interior_from(grid: &[f64], e_lo: f64, e_hi: f64) -> (usize, &[f64]) {
    let start = grid.partition_point(|&e| e <= e_lo);
    // `max(start)` only matters for a NaN edge, where both bisections answer 0
    // and the empty point set is what the scan produced too.
    let end = grid.partition_point(|&e| e < e_hi).max(start);
    (start, &grid[start..end])
}

/// The evaluation points of one group *after* its lower edge: every grid point
/// strictly inside, in order, then `e_hi`.
///
/// `e_lo` is left out because every caller already holds it and wants to
/// evaluate there once before the loop rather than re-evaluate it as the left
/// end of the first segment.
///
/// Shared with [`crate::self_shielding`], which integrates over the same points
/// so that a shielded average and a dilute one differ only by the weight.
pub(crate) fn group_points_after(
    grid: &[f64],
    e_lo: f64,
    e_hi: f64,
) -> impl Iterator<Item = f64> + '_ {
    interior(grid, e_lo, e_hi)
        .iter()
        .copied()
        .chain(std::iter::once(e_hi))
}

/// As [`group_points_after`], over two grids at once: their interiors merged in
/// non-decreasing order, then `e_hi`.
///
/// For an integrand that bends on two grids -- the cross section and the
/// fission-yield hats, or a branching curve and the cross section -- where the
/// evaluation points are the union of both. Merged rather than concatenated and
/// sorted, and duplicates are kept rather than deduplicated: between two
/// consecutive distinct values there is exactly one segment of non-zero width,
/// which is what a dedup left, and the zero-width ones either add `+0.0` or hit
/// a `de <= 0.0` guard.
pub(crate) fn group_points_after_merged<'a>(
    first: &'a [f64],
    second: &'a [f64],
    e_lo: f64,
    e_hi: f64,
) -> Merged<'a> {
    Merged {
        first: interior(first, e_lo, e_hi),
        second: interior(second, e_lo, e_hi),
        i: 0,
        j: 0,
        e_hi,
        done: false,
    }
}

/// The iterator [`group_points_after_merged`] returns.
pub(crate) struct Merged<'a> {
    first: &'a [f64],
    second: &'a [f64],
    i: usize,
    j: usize,
    e_hi: f64,
    done: bool,
}

impl Iterator for Merged<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.done {
            return None;
        }
        Some(match (self.first.get(self.i), self.second.get(self.j)) {
            (Some(&a), Some(&b)) if a <= b => {
                self.i += 1;
                a
            }
            (Some(_), Some(&b)) => {
                self.j += 1;
                b
            }
            (Some(&a), None) => {
                self.i += 1;
                a
            }
            (None, Some(&b)) => {
                self.j += 1;
                b
            }
            (None, None) => {
                self.done = true;
                self.e_hi
            }
        })
    }
}

/// Everything one group's trapezoid walk can produce, from one walk.
///
/// Three quantities are wanted over the same points of the same group, and each
/// used to walk them itself (issue #576, finding 2):
///
/// * the dilute group average, always;
/// * the shielded one, when a lump geometry was given;
/// * the self-shielding indicator, on the dilute path, which is #564's report
///   that a dilute run may be over-predicting.
///
/// The pairs are kept as separate accumulators rather than derived from each
/// other because they are not the same sum: the indicator divides by the summed
/// segment widths and the group average divides by `e_hi - e_lo`, and
/// `sum(de) != e_hi - e_lo` in floating point. Fusing the walks does not change
/// any accumulator's own sequence of terms, which is what keeps this
/// bit-identical to three separate walks.
#[derive(Default)]
pub(crate) struct GroupTerms {
    /// `∫ σ dE`, over the whole point set including zero-width segments.
    integral: f64,
    /// `∫ σ φ dE` and `∫ φ dE` under a shielded flux shape.
    shielded_num: f64,
    shielded_den: f64,
    /// `∫ σ w dE`, `∫ w dE`, `∫ σ dE` and `∫ dE` with `w = 1 / (1 + N σ)`, over
    /// the non-degenerate segments only.
    bound_num: f64,
    bound_den: f64,
    bound_plain: f64,
    bound_width: f64,
}

impl GroupTerms {
    /// The dilute group average, `σ_g`.
    pub(crate) fn dilute(&self, e_lo: f64, e_hi: f64) -> f64 {
        self.integral / (e_hi - e_lo)
    }

    /// The group average under the shielded flux shape, or 0.0 where the shape
    /// integrates to nothing.
    pub(crate) fn shielded(&self) -> f64 {
        if self.shielded_den > 0.0 {
            self.shielded_num / self.shielded_den
        } else {
            0.0
        }
    }

    /// This group's contributions to the self-shielding indicator, weighted by
    /// `phi`, or `None` where the group is degenerate.
    pub(crate) fn bound_contribution(&self, phi: f64) -> Option<(f64, f64)> {
        if self.bound_den > 0.0 && self.bound_width > 0.0 {
            Some((
                self.bound_num / self.bound_den * phi,
                self.bound_plain / self.bound_width * phi,
            ))
        } else {
            None
        }
    }
}

/// Walk one group's evaluation points once, accumulating whatever was asked for.
///
/// `shape` turns on the shielded average; `bound_density` (atoms/barn-cm, and
/// only meaningful when positive) turns on the self-shielding indicator.
pub(crate) fn walk_group(
    reaction: &Reaction,
    e_lo: f64,
    e_hi: f64,
    shape: Option<&FluxShape>,
    bound_density: Option<f64>,
) -> GroupTerms {
    let mut terms = GroupTerms::default();
    if e_lo >= e_hi {
        return terms;
    }

    // Each point is evaluated once and carried into the next segment.
    let xs_at = |e: f64| reaction.cross_section_at(e).unwrap_or(0.0);
    let phi_at = |e: f64| shape.map_or(0.0, |s| s.at(e));
    let mut prev_e = e_lo;
    let mut prev_xs = xs_at(e_lo);
    let mut prev_phi = phi_at(e_lo);

    // An interior point IS a grid point, and its position is already known --
    // `interior_from` returned it. `cross_section_at` would bisect
    // `reaction.energy` to find that same position again, once per point per
    // group, so the cross section is read by index instead (issue #576,
    // finding 1b).
    //
    // The carve-out is a run of equal energies, which is how ENDF spells a step
    // discontinuity. `slice::binary_search_by` documents that "if there are
    // multiple matches, then any one of the matches could be returned", so
    // which of a run's cross sections `cross_section_at` hands back is an
    // implementation detail -- and indexing would be betting on it. Those
    // points keep going through `cross_section_at`, which is what makes this
    // bit-identical rather than merely equal on well-behaved data.
    let grid: &[f64] = &reaction.energy;
    let values: &[f64] = &reaction.cross_section;
    // The Arrow reader enforces this at load (a file where the two disagree
    // fails to load rather than panicking on the first lookup, issue #507), so
    // it holds for anything that came from data. Checked anyway, because
    // `group_averaged_xs` is reachable with a hand-built `Reaction` and the
    // index below should not be what discovers it.
    let lengths_agree = grid.len() == values.len();

    // Above the evaluation's last energy point there is no cross section, and
    // the walk stops there. `cross_section_at` holds its last value above the
    // grid, which is right for a lookup at a single energy just past the end
    // but not for an integral: a CCFE-709 group reaching 1 GeV against an
    // evaluation that ends at 200 MeV would carry the 200 MeV cross section
    // across the other 800 MeV. The group keeps its full width in the dilute
    // average, and the tail keeps its flux in the shielded denominator, so the
    // averages say what they should: no cross section over that part of the
    // group. A group that lies entirely above the grid has nothing to walk.
    let top = grid.last().copied().unwrap_or(e_lo);
    let e_end = e_hi.min(top);
    if e_end <= e_lo {
        if shape.is_some() {
            terms.shielded_den += 0.5 * (prev_phi + phi_at(e_hi)) * (e_hi - e_lo);
        }
        if bound_density.is_some() {
            terms.bound_den += e_hi - e_lo;
            terms.bound_width += e_hi - e_lo;
        }
        return terms;
    }
    let (start, inner) = interior_from(grid, e_lo, e_end);

    let points = inner
        .iter()
        .enumerate()
        .map(|(k, &e)| (Some(k), e))
        // The upper edge is not a grid point in general, so it is interpolated.
        .chain(std::iter::once((None, e_end)));
    for (offset, e) in points {
        let xs = match offset.filter(|_| lengths_agree) {
            Some(k) => {
                let i = start + k;
                let alone =
                    (i == 0 || grid[i - 1] != e) && (i + 1 == grid.len() || grid[i + 1] != e);
                if alone {
                    values[i]
                } else {
                    xs_at(e)
                }
            }
            None => xs_at(e),
        };
        let phi = phi_at(e);
        let de = e - prev_e;
        let mean = 0.5 * (prev_xs + xs);

        // No `de <= 0.0` guard here, deliberately: `group_averaged_xs` never
        // had one, so a zero-width segment adds exactly `+0.0` and the sum is
        // the one it produced.
        terms.integral += mean * de;

        if de > 0.0 {
            if shape.is_some() {
                let phi_mean = 0.5 * (prev_phi + phi);
                terms.shielded_num += mean * phi_mean * de;
                terms.shielded_den += phi_mean * de;
            }
            if let Some(density) = bound_density {
                // This cross section as its own absorber, against a unit
                // background: an indicator of how much a nuclide's resonances
                // could suppress it, not a correction. See
                // `self_shielding::strongest_possible_suppression`.
                let weight = 1.0 / (1.0 + density * mean);
                terms.bound_num += mean * weight * de;
                terms.bound_den += weight * de;
                terms.bound_plain += mean * de;
                terms.bound_width += de;
            }
        }

        prev_e = e;
        prev_xs = xs;
        prev_phi = phi;
    }

    // The part of the group above the evaluation: zero cross section, so it
    // adds to the denominators alone.
    if e_end < e_hi {
        let de = e_hi - e_end;
        if shape.is_some() {
            terms.shielded_den += 0.5 * (phi_at(e_end) + phi_at(e_hi)) * de;
        }
        if bound_density.is_some() {
            terms.bound_den += de;
            terms.bound_width += de;
        }
    }

    terms
}

/// The share of a spectrum's flux above `top`, with the flux flat inside each
/// group, which is what the fold assumes.
pub(crate) fn fraction_above(multigroup_flux: &[f64], group_boundaries: &[f64], top: f64) -> f64 {
    let total: f64 = multigroup_flux.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut above = 0.0;
    for (g, &phi) in multigroup_flux.iter().enumerate() {
        let (lo, hi) = (group_boundaries[g], group_boundaries[g + 1]);
        if phi <= 0.0 || hi <= top {
            continue;
        }
        above += phi * (hi - lo.max(top)) / (hi - lo);
    }
    above / total
}

/// The spectrum's flux above the last energy each nuclide's evaluation
/// reaches: `(nuclide, top energy [eV], fraction of the flux)`, for the
/// nuclides where that fraction is not zero, in name order.
///
/// No cross section exists there and [`walk_group`] folds it as zero, which is
/// the honest value for one group's tail but misstates every rate once a
/// material part of the spectrum is involved. TENDL evaluations end at 200
/// MeV and ENDF/B's at 20 MeV, and CCFE-709 runs to 1 GeV, so a spectrum with
/// flux in its top groups can reach this with either library.
pub fn spectrum_above_evaluation(
    material: &Material,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
) -> Vec<(String, f64, f64)> {
    let mut names: Vec<&String> = material.nuclide_data.keys().collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let nuclide_data = &material.nuclide_data[name];
        let temperature = if material.temperature().is_empty() {
            crate::default_temperature(nuclide_data)
        } else {
            Some(material.temperature().to_string())
        };
        let Some(temperature) = temperature else {
            continue;
        };
        let Some(reactions) = nuclide_data.reactions_for_temp(&temperature) else {
            continue;
        };
        let top = reactions
            .values()
            .filter_map(|r| r.energy.last().copied())
            .fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            continue;
        }
        let fraction = fraction_above(multigroup_flux, group_boundaries, top);
        if fraction > 0.0 {
            out.push((name.clone(), top, fraction));
        }
    }
    out
}

/// The most of a spectrum that may lie above a nuclide's evaluation before a
/// transmutation refuses to run on it.
pub const ABOVE_EVALUATION_TOLERANCE: f64 = 1.0e-3;

/// Compute group-averaged cross section for a single energy group via trapezoidal integration.
///
/// σ_g = ∫σ(E)dE / (E_hi - E_lo)
///
/// Evaluation points are: [e_lo, grid points in (e_lo, e_hi), e_hi],
/// using the reaction's `cross_section_at()` for interpolation at boundaries.
pub(crate) fn group_averaged_xs(reaction: &Reaction, e_lo: f64, e_hi: f64) -> f64 {
    if e_lo >= e_hi {
        return 0.0;
    }
    walk_group(reaction, e_lo, e_hi, None, None).dilute(e_lo, e_hi)
}

/// Split the group-averaged fission cross section across a nuclide's tabulated
/// yield energies, adding each share into `out`.
///
/// Entry `k` accumulates `∫ σ_f(E) · φ_k(E) dE / (E_hi - E_lo)`, where `φ_k` is
/// the linear interpolation hat function of tabulated point `k` (clamped flat
/// outside the tabulated range).
///
/// The hats are a partition of unity at every evaluation point and the
/// trapezoid rule is linear in its integrand, so the entries sum to the group
/// average of `σ_f` over this same point set. Weighting them by the group
/// fluxes and normalizing therefore gives coefficients that sum to one, and
/// folding those against the yield vectors is the commuted form of
/// interpolating the yields group by group.
///
/// The point set refines [`group_averaged_xs`]'s by the tabulated energies,
/// where the hats bend, so the totals agree exactly for a cross section that
/// is linear across them and otherwise differ by the refinement, which is the
/// more accurate of the two. Normalization uses this function's own total, so
/// the weights are a partition of unity either way.
fn add_group_fission_xs_by_yield_point(
    reaction: &Reaction,
    fy_set: &FissionYieldSet,
    fy_energies: &[f64],
    e_lo: f64,
    e_hi: f64,
    scale: f64,
    out: &mut [f64],
) {
    if e_lo >= e_hi || fy_set.yields.is_empty() {
        return;
    }

    /// Cross section at a point and the yield hats it falls between.
    type PointTerm = (f64, Option<[(usize, f64); 2]>);
    let term = |e: f64| -> PointTerm {
        (
            reaction.cross_section_at(e).unwrap_or(0.0),
            fy_set.interp_weights(e),
        )
    };

    // Two interior slices merged rather than two grids concatenated and sorted.
    // `fy_energies` are the tabulated yield energies, where the hats bend, so
    // they are breakpoints of the integrand just as the cross-section grid
    // points are; `FissionYieldSet::new` sorts them, so both slices ascend and
    // the merge is a walk. It keeps duplicates, within a slice and across the
    // two, which the `de <= 0.0` skip below discards exactly as `Vec::dedup`
    // used to -- so the surviving segments, and the order they are accumulated
    // in, are unchanged.
    let width = e_hi - e_lo;
    let mut prev_e = e_lo;
    let mut prev = term(e_lo);
    for next_e in group_points_after_merged(&reaction.energy, fy_energies, e_lo, e_hi) {
        let next = term(next_e);
        let de = next_e - prev_e;
        if de > 0.0 {
            for (xs, hats) in [&prev, &next] {
                if *xs == 0.0 {
                    continue;
                }
                let Some(hats) = hats else { continue };
                for (k, w) in hats {
                    if *w != 0.0 {
                        out[*k] += scale * 0.5 * xs * w * de / width;
                    }
                }
            }
        }
        prev_e = next_e;
        prev = next;
    }
}

/// Compute multigroup reaction rates for a material.
///
/// For each nuclide in the material that appears in the chain, collapses
/// pointwise cross sections with the multigroup flux to obtain effective
/// one-group rates.
///
/// # Arguments
/// * `material` - Material with loaded nuclear data
/// * `chain` - Transmutation chain
/// * `multigroup_flux` - Flux per group [n/cm²/s], length = n_groups
/// * `group_boundaries` - Energy group boundaries in eV (ascending), length = n_groups + 1
/// * `source_rate` - Scalar multiplier for the flux (1.0 = full, 0.0 = decay only)
///
/// # Returns
/// The one-group `ReactionRates` (nuclide_name -> reaction_type -> rate [1/s])
/// and, for every fissionable nuclide that carries yields, the
/// [`FissionYieldWeights`] folding its tabulated yield vectors against this
/// spectrum (issue #379). The weights are a normalized shape, so the per-step
/// `scale_rates` leaves them correct.
pub fn compute_multigroup_reaction_rates(
    material: &Material,
    chain: &HashMap<String, ChainNuclide>,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
    source_rate: f64,
) -> (ReactionRates, FissionYieldWeights) {
    let (rates, weights, _) = compute_multigroup_reaction_rates_shielded(
        material,
        chain,
        multigroup_flux,
        group_boundaries,
        source_rate,
        None,
    );
    (rates, weights)
}

/// As [`compute_multigroup_reaction_rates`], optionally weighting each group
/// average by the flux inside a lump rather than the caller's spectrum.
///
/// `shielding` of `None` is the default path: no flux shape is built, nothing
/// is weighted, and the result is bit-identical to a build without the
/// [`crate::self_shielding`] module. The returned [`ShieldingInfo`] says which
/// nuclides were corrected and how strongly, so a shielded run is never
/// mistaken for an unshielded one.
pub fn compute_multigroup_reaction_rates_shielded(
    material: &Material,
    chain: &HashMap<String, ChainNuclide>,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
    source_rate: f64,
    shielding: Option<&Shielding>,
) -> (ReactionRates, FissionYieldWeights, ShieldingInfo) {
    let mut rates: ReactionRates = HashMap::new();
    let mut fy_weights: FissionYieldWeights = HashMap::new();

    let mut info = ShieldingInfo::default();
    if let Some(s) = shielding {
        info.method = Some("slowing-down".to_string());
        info.chord_cm = Some(s.chord_cm);
    }

    // If source_rate is zero, return empty rates (decay only)
    if source_rate == 0.0 {
        return (rates, fy_weights, info);
    }

    // Total scalar flux = sum of group fluxes
    let Some(setup) = CollapseSetup::new(material, multigroup_flux, shielding) else {
        return (rates, fy_weights, info);
    };

    // One nuclide's collapse is independent of every other's: it reads the
    // material, the spectrum and its own chain entry, and writes only its own
    // rates and its own yield weights. So the loop is a map, and only its
    // driver line is `#[cfg]`-ed (issue #576, finding 5b).
    //
    // `entries` is a snapshot of `chain.iter()`, and the merge below walks it
    // in the same order, so even `info.shielded` -- a `Vec` built by pushing --
    // comes out in the order the sequential loop produced rather than merely
    // holding the same names.
    let entries: Vec<(&String, &ChainNuclide)> = chain.iter().collect();
    let context = setup.context(
        material,
        multigroup_flux,
        group_boundaries,
        source_rate,
        shielding,
    );
    let collapsed: Vec<Option<NuclideCollapse>> = {
        let one = |&(name, chain_nuclide): &(&String, &ChainNuclide)| {
            context.collapse_nuclide(name, chain_nuclide)
        };
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

    for ((name, _), collapsed) in entries.iter().zip(collapsed) {
        let Some(collapsed) = collapsed else { continue };
        // Keyed by a name that appears once, so these three are order-free and
        // the merge order is only about `shielded` below.
        if !collapsed.rates.is_empty() {
            rates.insert((*name).clone(), collapsed.rates);
        }
        if let Some(weights) = collapsed.fy_weights {
            fy_weights.insert((*name).clone(), weights);
        }
        if let Some(reason) = collapsed.not_shielded {
            info.not_shielded.insert((*name).clone(), reason);
        }
        if let Some(bound) = collapsed.would_shield {
            info.would_shield.insert((*name).clone(), bound);
        }
        if collapsed.shielded {
            info.shielded.push((*name).clone());
        }
        // A `min` over a set, so its order does not reach the answer. The
        // `dilute > 0.0` guard inside the walk is what keeps a NaN out of it.
        if let Some(factor) = collapsed.strongest_factor {
            info.strongest_factor =
                Some(info.strongest_factor.map_or(factor, |f: f64| f.min(factor)));
        }
    }

    (rates, fy_weights, info)
}

/// Everything one nuclide's collapse produces, held rather than written
/// straight into the shared maps.
///
/// Every `info.*` mutation the loop used to make in place is a field here, so
/// the work is a pure function of its inputs and the merge is the only thing
/// that touches shared state (issue #576, finding 5b).
struct NuclideCollapse {
    /// Reaction kind -> one-group rate [1/s].
    rates: HashMap<String, f64>,
    /// The normalized fission-yield spectrum weights, when this spectrum drives
    /// any fission at all.
    fy_weights: Option<Vec<f64>>,
    /// Why this nuclide was not shielded, when a shielded run could not be.
    not_shielded: Option<String>,
    /// Whether a flux shape was built for it.
    shielded: bool,
    /// The self-shielding indicator, when it is low enough to report.
    would_shield: Option<f64>,
    /// The strongest suppression any of its groups saw, on the shielded path.
    strongest_factor: Option<f64>,
}

/// Everything derived from the material and the spectrum that every nuclide's
/// collapse shares, gathered once.
///
/// A struct rather than four locals so that [`reaction_rate_spectrum`], which
/// walks one channel long after the solve, builds its context the same way
/// [`compute_multigroup_reaction_rates_shielded`] does. Two copies of this
/// gather would be two chances for an energy-resolved rate and the collapsed
/// rate it is supposed to sum to to disagree.
struct CollapseSetup<'a> {
    densities: HashMap<String, f64>,
    mixture: Option<crate::self_shielding::MixtureTotal<'a>>,
    active: Vec<usize>,
    total_flux: f64,
}

impl<'a> CollapseSetup<'a> {
    /// `None` when the spectrum carries no flux, which drives nothing.
    fn new(
        material: &'a Material,
        multigroup_flux: &[f64],
        shielding: Option<&Shielding>,
    ) -> Option<Self> {
        let total_flux: f64 = multigroup_flux.iter().sum();
        if total_flux <= 0.0 {
            return None;
        }

        // The flux depression is a property of the whole material, so the
        // mixture total is gathered once and every nuclide's shape is built
        // against it.
        let densities = material.get_atoms_per_barn_cm().unwrap_or_default();
        let temperature_for = |name: &str| -> Option<String> {
            let nd = material.nuclide_data.get(name)?;
            if material.temperature().is_empty() {
                crate::default_temperature(nd)
            } else {
                Some(material.temperature().to_string())
            }
        };
        let mixture = shielding.map(|_| mixture_total(material, &densities, &temperature_for));

        // The groups worth walking, decided once for the whole spectrum rather
        // than per nuclide per reaction. A group with no flux contributes
        // `sigma_g * 0.0` to the collapse and `scale = 0.0` to the yield fold,
        // both of which are exactly `+0.0`, so skipping it is bit-neutral --
        // and a monoenergetic 14 MeV source in CCFE-709 leaves 1 group of 709
        // (issue #576, finding 2).
        //
        // Only on the dilute path. `info.strongest_factor` mins over every
        // group a shielded run walks, zero-flux ones included, so skipping them
        // there would change a reported number rather than only the time taken.
        let n_groups = multigroup_flux.len();
        let active: Vec<usize> = if shielding.is_none() {
            (0..n_groups)
                .filter(|&g| multigroup_flux[g] != 0.0)
                .collect()
        } else {
            (0..n_groups).collect()
        };

        Some(Self {
            densities,
            mixture,
            active,
            total_flux,
        })
    }

    /// The per-nuclide collapse context this setup supports.
    fn context<'b>(
        &'b self,
        material: &'b Material,
        multigroup_flux: &'b [f64],
        group_boundaries: &'b [f64],
        source_rate: f64,
        shielding: Option<&'b Shielding>,
    ) -> Collapse<'b>
    where
        'a: 'b,
    {
        Collapse {
            material,
            multigroup_flux,
            group_boundaries,
            source_rate,
            shielding,
            densities: &self.densities,
            mixture: self.mixture.as_ref(),
            active: &self.active,
            total_flux: self.total_flux,
        }
    }
}

/// What one channel's walk accumulates besides the rate itself.
///
/// Returned rather than written in place so the walk is a pure function of its
/// inputs and both of its callers can ignore what they do not want.
#[derive(Default)]
struct ChannelWalk {
    /// The strongest per-group suppression the shielded average applied.
    strongest_factor: Option<f64>,
    /// Numerator and denominator of the dilute run's self-shielding indicator.
    bound_shielded: f64,
    bound_dilute: f64,
}

/// What one nuclide's collapse needs from outside itself. All of it read-only,
/// which is what makes the map safe to run in parallel.
struct Collapse<'a> {
    material: &'a Material,
    multigroup_flux: &'a [f64],
    group_boundaries: &'a [f64],
    source_rate: f64,
    shielding: Option<&'a Shielding>,
    densities: &'a HashMap<String, f64>,
    mixture: Option<&'a crate::self_shielding::MixtureTotal<'a>>,
    active: &'a [usize],
    total_flux: f64,
}

impl Collapse<'_> {
    /// The reactions loaded for one nuclide at the temperature in use.
    ///
    /// Shared so that every consumer of the collapse resolves the temperature
    /// the same way; a second copy of this is a second chance to read a
    /// different evaluation than the one the rates came from.
    fn reactions_for(&self, nuclide_name: &str) -> Option<&HashMap<i32, std::sync::Arc<Reaction>>> {
        let nuclide_data = self.material.nuclide_data.get(nuclide_name)?;
        let temperature = if self.material.temperature().is_empty() {
            crate::default_temperature(nuclide_data)?
        } else {
            self.material.temperature().to_string()
        };
        nuclide_data.reactions_for_temp(&temperature)
    }

    /// The flux shape one nuclide's group averages are taken under, with what
    /// a report should say about it: `(shape, why not, whether it was)`.
    ///
    /// One flux shape per nuclide: the mixture sets the depression, the
    /// nuclide's own elastic scattering fills its resonance dips. `None` on the
    /// dilute path, and on the shielded path for a nuclide whose name or data
    /// will not support one.
    fn shape_for(
        &self,
        nuclide_name: &str,
        reactions: &HashMap<i32, std::sync::Arc<Reaction>>,
    ) -> (Option<FluxShape>, Option<String>, bool) {
        match (self.shielding, self.mixture) {
            (Some(request), Some(mix)) => match mass_number_of(nuclide_name) {
                Some(mass_number) => {
                    let elastic = reactions
                        .get(&crate::self_shielding::MT_ELASTIC)
                        .map(|r| r.as_ref());
                    let why_not = elastic.is_none().then(|| "no MT=2 elastic".to_string());
                    let density = self.densities.get(nuclide_name).copied().unwrap_or(0.0);
                    (
                        Some(flux_shape(mix, elastic, density, mass_number, request)),
                        why_not,
                        true,
                    )
                }
                None => (
                    None,
                    Some("mass number not parsable from the name".to_string()),
                    false,
                ),
            },
            _ => (None, None, false),
        }
    }

    /// Walk one channel over every active group, handing each group's average
    /// to `on_group` as `(g, sigma_g, phi_g)`.
    ///
    /// This is the one definition of what `sigma_g` means for a group, so the
    /// collapsed rate and any energy-resolved view of it cannot drift apart:
    /// [`Collapse::collapse_nuclide`] sums `sigma_g * phi_g` out of it and
    /// [`reaction_rate_spectrum`] keeps the terms.
    ///
    /// `bound_density` (atoms/barn-cm, and only meaningful when positive) turns
    /// on the dilute run's self-shielding indicator. A caller that does not
    /// want it passes `None` and the walk does less work.
    fn walk_channel(
        &self,
        reaction: &Reaction,
        shape: Option<&FluxShape>,
        bound_density: Option<f64>,
        mut on_group: impl FnMut(usize, f64, f64),
    ) -> ChannelWalk {
        let mut walk = ChannelWalk::default();
        for &g in self.active {
            let e_lo = self.group_boundaries[g];
            let e_hi = self.group_boundaries[g + 1];
            let phi = self.multigroup_flux[g];
            // One walk of this group's points, whichever of the three
            // integrals over them are wanted.
            let terms = walk_group(
                reaction,
                e_lo,
                e_hi,
                shape,
                // `strongest_possible_suppression` skips a group with no
                // flux in it, so the indicator must not accumulate one here.
                bound_density.filter(|_| phi > 0.0),
            );
            let sigma_g = if shape.is_some() {
                let shielded = terms.shielded();
                let dilute = terms.dilute(e_lo, e_hi);
                if dilute > 0.0 {
                    let factor = shielded / dilute;
                    walk.strongest_factor =
                        Some(walk.strongest_factor.map_or(factor, |f: f64| f.min(factor)));
                }
                shielded
            } else {
                terms.dilute(e_lo, e_hi)
            };
            if let Some((s, d)) = terms.bound_contribution(phi) {
                walk.bound_shielded += s;
                walk.bound_dilute += d;
            }
            on_group(g, sigma_g, phi);
        }
        walk
    }

    /// Collapse one nuclide, or `None` when it has nothing to contribute.
    fn collapse_nuclide(
        &self,
        nuclide_name: &str,
        chain_nuclide: &ChainNuclide,
    ) -> Option<NuclideCollapse> {
        if chain_nuclide.reactions.is_empty() {
            return None;
        }

        let reactions = self.reactions_for(nuclide_name)?;

        let mut out = NuclideCollapse {
            rates: HashMap::new(),
            fy_weights: None,
            not_shielded: None,
            shielded: false,
            would_shield: None,
            strongest_factor: None,
        };

        let (shape, not_shielded, shielded) = self.shape_for(nuclide_name, reactions);
        out.not_shielded = not_shielded;
        out.shielded = shielded;

        for &rx_type in &kinds_of(chain_nuclide) {
            let mt = match reaction_type_to_mt(rx_type) {
                Some(mt) => mt,
                None => continue,
            };

            let reaction = match reactions.get(&mt) {
                Some(r) => r,
                None => continue,
            };

            // The fission-rate distribution R_g = sigma_f,g * phi_g is the
            // summand of this very loop, so the yield fold rides along on it
            // with no extra cross-section work (issue #379).
            let fold_yields = fy_set_for(chain_nuclide).filter(|_| mt == MT_FISSION);
            let mut yield_shares = vec![0.0; fold_yields.map_or(0, |s| s.yields.len())];
            // Once per reaction rather than once per group: this is a property
            // of the nuclide's yields, and rebuilding it 709 times per fissile
            // reaction was 709 allocations for the same handful of energies.
            let fy_energies: Vec<f64> = fold_yields.map_or_else(Vec::new, |s| s.energies());

            // A dilute run says what it did not correct for (#564). The
            // indicator uses only this reaction, which is already loaded, so it
            // costs no extra data and no geometry -- but it used to walk every
            // group a second time, with two more `cross_section_at` per window,
            // to get it. It rides along on the walk below instead.
            //
            // `strongest_possible_suppression` answers 1.0 immediately for a
            // non-positive density, and 1.0 is never reported, so a nuclide the
            // material does not contain accumulates nothing here either.
            let bound_density = self
                .shielding
                .is_none()
                .then(|| self.densities.get(nuclide_name).copied().unwrap_or(0.0))
                .filter(|&d| d > 0.0);

            // Collapse: σ_eff = Σ(σ_g × φ_g) / Σ(φ_g)
            let mut sigma_phi_sum = 0.0;
            let walk = self.walk_channel(
                reaction,
                shape.as_ref(),
                bound_density,
                |g, sigma_g, phi| {
                    sigma_phi_sum += sigma_g * phi;
                    if let Some(fy_set) = fold_yields {
                        add_group_fission_xs_by_yield_point(
                            reaction,
                            fy_set,
                            &fy_energies,
                            self.group_boundaries[g],
                            self.group_boundaries[g + 1],
                            phi,
                            &mut yield_shares,
                        );
                    }
                },
            );
            if let Some(factor) = walk.strongest_factor {
                out.strongest_factor =
                    Some(out.strongest_factor.map_or(factor, |f: f64| f.min(factor)));
            }

            if bound_density.is_some() {
                let bound = if walk.bound_dilute > 0.0 {
                    (walk.bound_shielded / walk.bound_dilute).min(1.0)
                } else {
                    1.0
                };
                if bound < 0.99 {
                    out.would_shield = Some(match out.would_shield {
                        Some(worst) if worst <= bound => worst,
                        _ => bound,
                    });
                }
            }

            // Normalize the shares into a partition of unity. A zero total
            // means this spectrum drives no fission at all, so there is no
            // fold to define and the nuclide is left out; the matrix builder
            // only demands weights where the fission rate is non-zero.
            let shares_total: f64 = yield_shares.iter().sum();
            if shares_total > 0.0 {
                for c in &mut yield_shares {
                    *c /= shares_total;
                }
                out.fy_weights = Some(yield_shares);
            }

            let sigma_eff = sigma_phi_sum / self.total_flux;

            if sigma_eff > 0.0 {
                // rate = σ_eff [barn] × 1e-24 [cm²/barn] × total_flux [n/cm²/s] × source_rate
                let rate = sigma_eff * 1.0e-24 * self.total_flux * self.source_rate;
                out.rates.insert(rx_type.to_string(), rate);
            }
        }

        Some(out)
    }
}

/// The reaction kinds the chain drives on one nuclide, sorted and deduped.
fn kinds_of(chain_nuclide: &ChainNuclide) -> Vec<&str> {
    let mut kinds: Vec<&str> = chain_nuclide
        .reactions
        .iter()
        .map(|r| r.kind.as_str())
        .collect();
    kinds.sort();
    kinds.dedup();
    kinds
}

/// The nuclide's fission yields, if it has any tabulated.
fn fy_set_for(chain_nuclide: &ChainNuclide) -> Option<&FissionYieldSet> {
    chain_nuclide
        .fission_yields
        .as_deref()
        .filter(|s| !s.yields.is_empty())
}

/// The per-group contributions to one channel's collapsed reaction rate.
///
/// `out[g]` is `1e-24 * sigma_g * phi_g * source_rate` in 1/s, so summing over
/// the groups gives the rate
/// [`compute_multigroup_reaction_rates_shielded`] reports for the same
/// channel, and each entry is that group's share of it. Both come from
/// [`Collapse::walk_channel`], self-shielding included, so the two cannot
/// disagree about what `sigma_g` means; they agree to floating-point rounding
/// rather than bit-exactly, because the collapse divides the sum by the total
/// flux and multiplies it back.
///
/// # Why one channel, and nothing stored
///
/// This is the question a one-group rate cannot answer: a rate of 57 mb
/// against a spectrum that is 89% fast and 0.7% below 100 keV says nothing
/// about which of those two the rate came from, and the answer decides whether
/// a disagreement belongs to resonance processing or to the fast cross section
/// (yani#27).
///
/// Keeping the breakdown for every channel would be the rate map times the
/// group count, tens of MB on a 709-group structure, which is why
/// [`per_group_reaction_rates`] is built only when flux uncertainty asks for it
/// (issue #559). A diagnostic asks about one channel at a time, and walking a
/// single reaction over 709 groups is fast enough to do on demand, so nothing
/// is stored and the default path pays nothing.
///
/// # Returns
///
/// `None` when the material holds no data for `nuclide`, when `kind` names no
/// MT this build collapses, when the reaction is absent from the data, or when
/// the spectrum carries no flux. `(n,n')` is the notable absence: it has no
/// transport total, so its rate comes from the branching overlay's MF=10
/// partials rather than from a group average, and there is no `sigma_g` here
/// to report.
pub fn reaction_rate_spectrum(
    material: &Material,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
    source_rate: f64,
    shielding: Option<&Shielding>,
    nuclide: &str,
    kind: &str,
) -> Option<Vec<f64>> {
    // A pulse at zero flux drives nothing, exactly as
    // `compute_multigroup_reaction_rates_shielded` treats it, and a vector of
    // zeros would read as a rate resolved rather than as no rate at all.
    if source_rate == 0.0 {
        return None;
    }
    let mt = reaction_type_to_mt(kind)?;
    let setup = CollapseSetup::new(material, multigroup_flux, shielding)?;
    let context = setup.context(
        material,
        multigroup_flux,
        group_boundaries,
        source_rate,
        shielding,
    );
    let reactions = context.reactions_for(nuclide)?;
    let reaction = reactions.get(&mt)?;
    // The shielded average needs the nuclide's flux shape; the self-shielding
    // indicator does not belong to a rate, so its density is left off and the
    // walk skips that accumulation.
    let (shape, _, _) = context.shape_for(nuclide, reactions);

    // A zero-flux group is not walked on the dilute path, and its term is
    // exactly zero, so the vector is sized to the whole structure and only the
    // active groups are written into it.
    let mut out = vec![0.0; multigroup_flux.len()];
    context.walk_channel(reaction, shape.as_ref(), None, |g, sigma_g, phi| {
        out[g] = sigma_g * phi * 1.0e-24 * source_rate;
    });
    Some(out)
}

/// The per-group contributions to each reaction rate, for flux uncertainty.
///
/// `out[nuclide][kind][g]` is that group's share of the rate, so the nominal
/// rate is the sum over `g`. Every group average comes from
/// [`Collapse::walk_channel`], the same walk
/// [`compute_multigroup_reaction_rates_shielded`] collapses with and under the
/// same `shielding`, so the terms sum to the rate that run produced rather than
/// to a differently weighted one.
///
/// Separate rather than an extra return value because it is only ever wanted
/// when flux uncertainty is requested: it is the rate map times the group
/// count, which on a 709-group structure is tens of MB, and the default path
/// should not pay that (issue #559). [`reaction_rate_spectrum`] is the
/// one-channel form, for asking rather than for perturbing.
pub fn per_group_reaction_rates(
    material: &Material,
    chain: &HashMap<String, ChainNuclide>,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
    shielding: Option<&Shielding>,
) -> crate::flux_uncertainty::PerGroupRates {
    let mut out: crate::flux_uncertainty::PerGroupRates = HashMap::new();
    let n_groups = multigroup_flux.len();
    let Some(setup) = CollapseSetup::new(material, multigroup_flux, shielding) else {
        return out;
    };
    // Unit source rate: these decompose the unit-flux rates the replicas
    // perturb, and the step's magnitude is applied to both alike afterwards.
    let context = setup.context(material, multigroup_flux, group_boundaries, 1.0, shielding);

    // Per nuclide and independent, exactly as the collapse it mirrors is, and
    // merged into a map keyed by a name that appears once -- so the order the
    // results arrive in cannot reach the answer (issue #576, finding 5c).
    let entries: Vec<(&String, &ChainNuclide)> = chain.iter().collect();
    let one = |&(nuclide_name, chain_nuclide): &(&String, &ChainNuclide)| {
        if chain_nuclide.reactions.is_empty() {
            return None;
        }
        let reactions = context.reactions_for(nuclide_name)?;
        let (shape, _, _) = context.shape_for(nuclide_name, reactions);

        let mut per_kind: HashMap<String, Vec<f64>> = HashMap::new();
        for kind in kinds_of(chain_nuclide) {
            let Some(mt) = reaction_type_to_mt(kind) else {
                continue;
            };
            let Some(reaction) = reactions.get(&mt) else {
                continue;
            };
            // A group the walk skips carries no flux, so its term is exactly
            // zero and the vector is sized to the whole structure regardless:
            // the perturbation indexes it by the caller's own bin.
            let mut terms = vec![0.0; n_groups];
            context.walk_channel(reaction, shape.as_ref(), None, |g, sigma_g, phi| {
                // The same 1e-24 barn-to-cm^2 factor the rate carries, so these
                // sum to the rate rather than to something proportional to it.
                terms[g] = sigma_g * phi * 1.0e-24;
            });
            if terms.iter().any(|t| *t > 0.0) {
                per_kind.insert(kind.to_string(), terms);
            }
        }
        (!per_kind.is_empty()).then_some(per_kind)
    };

    let per_nuclide: Vec<Option<HashMap<String, Vec<f64>>>> = {
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
    for ((name, _), per_kind) in entries.iter().zip(per_nuclide) {
        if let Some(per_kind) = per_kind {
            out.insert((*name).clone(), per_kind);
        }
    }
    out
}

/// Scale all reaction rates by a multiplier.
///
/// Returns a new `ReactionRates` with each rate multiplied by `factor`.
pub fn scale_rates(rates: &ReactionRates, factor: f64) -> ReactionRates {
    rates
        .iter()
        .map(|(nuc, rxns)| {
            let scaled: HashMap<String, f64> = rxns
                .iter()
                .map(|(rx, &rate)| (rx.clone(), rate * factor))
                .collect();
            (nuc.clone(), scaled)
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a Reaction with a flat (constant) cross section.
    fn flat_reaction(energy_range: (f64, f64), xs_value: f64, mt: i32) -> Reaction {
        Reaction {
            cross_section: vec![xs_value, xs_value].into(),
            threshold_idx: 0,
            energy: vec![energy_range.0, energy_range.1].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// Helper: build a Reaction with a linear ramp cross section.
    fn ramp_reaction(e0: f64, e1: f64, xs0: f64, xs1: f64, mt: i32) -> Reaction {
        Reaction {
            cross_section: vec![xs0, xs1].into(),
            threshold_idx: 0,
            energy: vec![e0, e1].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    #[test]
    fn test_group_averaged_xs_flat() {
        // Flat XS of 10.0 barns across entire range
        let rxn = flat_reaction((1.0, 1e6), 10.0, 102);

        // Group that spans the whole range
        let avg = group_averaged_xs(&rxn, 1.0, 1e6);
        assert!(
            (avg - 10.0).abs() < 1e-10,
            "Flat XS should average to 10.0, got {avg}"
        );

        // Subgroup should also be 10.0
        let avg2 = group_averaged_xs(&rxn, 100.0, 1000.0);
        assert!(
            (avg2 - 10.0).abs() < 1e-10,
            "Flat XS subgroup should be 10.0, got {avg2}"
        );
    }

    /// Above the evaluation's last point the cross section is not the last
    /// value carried on, it is nothing: the group average over the part of a
    /// group past the grid is zero, and a group entirely past it is zero.
    #[test]
    fn the_average_stops_at_the_evaluations_last_point() {
        let rxn = flat_reaction((1e6, 1e7), 2.0, 16);
        // Half of this group is above the grid.
        let avg = group_averaged_xs(&rxn, 5e6, 1.5e7);
        assert!((avg - 1.0).abs() < 1e-12, "got {avg}");
        // All of it.
        assert_eq!(group_averaged_xs(&rxn, 2e7, 3e7), 0.0);
        // None of it: unchanged.
        assert!((group_averaged_xs(&rxn, 2e6, 4e6) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_flux_above_an_evaluation_is_counted_flat_within_a_group() {
        let flux = [1.0, 2.0, 4.0];
        let edges = [0.0, 1e7, 2e7, 4e7];
        // Nothing above 4e7.
        assert_eq!(fraction_above(&flux, &edges, 4e7), 0.0);
        // The top group only, whole: 4 of 7.
        assert!((fraction_above(&flux, &edges, 2e7) - 4.0 / 7.0).abs() < 1e-12);
        // Half of the top group: 2 of 7.
        assert!((fraction_above(&flux, &edges, 3e7) - 2.0 / 7.0).abs() < 1e-12);
        // Everything.
        assert!((fraction_above(&flux, &edges, 0.0) - 1.0).abs() < 1e-12);
        // No flux at all.
        assert_eq!(fraction_above(&[0.0, 0.0], &[0.0, 1.0, 2.0], 0.5), 0.0);
    }

    #[test]
    fn test_group_averaged_xs_ramp() {
        // Linear ramp from 0 to 100 barns over 0 to 1e6 eV
        let rxn = ramp_reaction(0.0, 1e6, 0.0, 100.0, 102);

        // Average of a linear function over full range = (0 + 100) / 2 = 50
        let avg = group_averaged_xs(&rxn, 0.0, 1e6);
        assert!(
            (avg - 50.0).abs() < 1e-6,
            "Ramp XS average should be 50.0, got {avg}"
        );
    }

    #[test]
    fn test_group_averaged_xs_threshold() {
        // Threshold reaction: zero below 1e5 eV, rises to 1.0 barn at 1e6 eV
        let rxn = Reaction {
            cross_section: vec![0.0, 1.0].into(),
            threshold_idx: 1, // nonzero = threshold
            energy: vec![1e5, 1e6].into(),
            mt_number: 16,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        };

        // Group entirely below threshold
        let avg_below = group_averaged_xs(&rxn, 1.0, 1e4);
        assert!(
            avg_below.abs() < 1e-10,
            "Below threshold should be 0, got {avg_below}"
        );

        // Group spanning threshold
        let avg_span = group_averaged_xs(&rxn, 1e4, 1e6);
        assert!(
            avg_span > 0.0,
            "Spanning threshold should give non-zero average"
        );
    }

    #[test]
    fn test_group_averaged_xs_zero_width() {
        let rxn = flat_reaction((1.0, 1e6), 10.0, 102);
        let avg = group_averaged_xs(&rxn, 100.0, 100.0);
        assert!(avg.abs() < 1e-10, "Zero-width group should return 0");
    }

    // --- the point set, now that it is a bisection rather than a scan (#576) ---
    //
    // `interior` finds the same points a full scan used to, but by two
    // `partition_point` calls, and it keeps runs of equal energies where
    // `Vec::dedup` collapsed them. These pin the cases where the two could come
    // apart.

    /// A reaction on an arbitrary grid, so a test can put a boundary exactly on
    /// a grid point or repeat an energy.
    fn on_grid(energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
        Reaction {
            cross_section: cross_section.into(),
            threshold_idx: 0,
            energy: energy.into(),
            mt_number: 102,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// What the scan-and-sort point set used to be, kept as the reference the
    /// bisection is compared against.
    fn scanned_points(grid: &[f64], e_lo: f64, e_hi: f64) -> Vec<f64> {
        let mut points = vec![e_lo];
        points.extend(grid.iter().copied().filter(|&e| e > e_lo && e < e_hi));
        points.push(e_hi);
        points.sort_by(f64::total_cmp);
        points.dedup();
        points
    }

    /// The trapezoid rule over the old point set, to the bit.
    fn scanned_group_averaged_xs(reaction: &Reaction, e_lo: f64, e_hi: f64) -> f64 {
        if e_lo >= e_hi {
            return 0.0;
        }
        // The walk stops at the evaluation's last point and counts nothing
        // above it, so the reference integrates to there and divides by the
        // whole width.
        let e_end = e_hi.min(reaction.energy.last().copied().unwrap_or(e_lo));
        if e_end <= e_lo {
            return 0.0;
        }
        let points = scanned_points(&reaction.energy, e_lo, e_end);
        let xs: Vec<f64> = points
            .iter()
            .map(|&e| reaction.cross_section_at(e).unwrap_or(0.0))
            .collect();
        let mut integral = 0.0;
        for i in 0..points.len() - 1 {
            integral += 0.5 * (xs[i] + xs[i + 1]) * (points[i + 1] - points[i]);
        }
        integral / (e_hi - e_lo)
    }

    #[test]
    fn the_bisected_point_set_is_the_scanned_one() {
        // Repeated energies, a threshold, and a grid that starts above zero,
        // so boundaries can land below, on, inside and above it.
        let grid = vec![1.0, 5.0, 5.0, 9.0, 12.0, 12.0, 12.0, 20.0];
        for (e_lo, e_hi) in [
            (0.0, 25.0),   // spans everything
            (1.0, 20.0),   // both boundaries exactly on a grid point
            (1.0, 9.0),    // e_lo on a point
            (5.0, 20.0),   // e_lo on a repeated point
            (0.0, 5.0),    // e_hi on a repeated point
            (2.0, 4.0),    // no interior points at all
            (0.0, 0.5),    // entirely below the grid
            (25.0, 30.0),  // entirely above the grid
            (12.0, 12.0),  // zero width, on a repeat
            (11.9, 12.01), // straddles a repeat
        ] {
            let bisected: Vec<f64> = std::iter::once(e_lo)
                .chain(group_points_after(&grid, e_lo, e_hi))
                .collect();
            let mut collapsed = bisected.clone();
            collapsed.dedup();
            assert_eq!(
                collapsed,
                scanned_points(&grid, e_lo, e_hi),
                "group ({e_lo}, {e_hi})"
            );
        }
    }

    #[test]
    fn a_duplicated_energy_averages_bit_identically() {
        // ENDF spells a step discontinuity as a repeated energy carrying two
        // different cross sections. The old point set dropped the repeat; the
        // new one keeps it as a zero-width segment. And a point inside such a
        // run is the carve-out of finding 1b: `binary_search_by` documents that
        // any of the matches could be returned, so which cross section
        // `cross_section_at` hands back there is an implementation detail and
        // the walk must not index around it. The result must not move either
        // way.
        //
        // Two grids: a plain pair, and a run of three, so the middle of a run
        // is covered as well as its ends.
        for rxn in [
            on_grid(vec![1.0, 5.0, 5.0, 9.0], vec![1.0, 2.0, 7.0, 8.0]),
            on_grid(vec![1.0, 5.0, 5.0, 5.0, 9.0], vec![1.0, 2.0, 4.0, 7.0, 8.0]),
        ] {
            for (e_lo, e_hi) in [
                (0.0, 10.0),
                (1.0, 9.0),
                (5.0, 9.0),
                (1.0, 5.0),
                (4.9, 5.1),
                (2.0, 3.0),
            ] {
                assert_eq!(
                    group_averaged_xs(&rxn, e_lo, e_hi).to_bits(),
                    scanned_group_averaged_xs(&rxn, e_lo, e_hi).to_bits(),
                    "group ({e_lo}, {e_hi}) moved"
                );
            }
        }
    }

    #[test]
    fn a_group_below_the_grid_reads_the_threshold_flag() {
        let mut rxn = on_grid(vec![10.0, 20.0], vec![3.0, 4.0]);

        // A threshold reaction is zero below its first grid point.
        rxn.threshold_idx = 1;
        assert_eq!(group_averaged_xs(&rxn, 1.0, 5.0), 0.0);

        // A non-threshold one (capture, say) is flat-clamped to its first value.
        rxn.threshold_idx = 0;
        assert_eq!(group_averaged_xs(&rxn, 1.0, 5.0), 3.0);

        // Above the last grid point there is no cross section either way:
        // `cross_section_at` holds its last value up there for a single
        // lookup, but an integral over energies the evaluation never reached
        // counts nothing.
        assert_eq!(group_averaged_xs(&rxn, 30.0, 40.0), 0.0);
        rxn.threshold_idx = 1;
        assert_eq!(group_averaged_xs(&rxn, 30.0, 40.0), 0.0);
        // A group straddling the end integrates to the end and no further:
        // 4.0 over half the group.
        assert_eq!(
            group_averaged_xs(&rxn, 15.0, 25.0),
            (0.5 * (3.5 + 4.0) * 5.0) / 10.0
        );
    }

    #[test]
    fn an_empty_grid_is_zero_rather_than_a_panic() {
        // `cross_section_at` answers `None` on an empty grid, which
        // `unwrap_or(0.0)` turns into a zero integrand rather than an index
        // out of bounds.
        let rxn = on_grid(vec![], vec![]);
        assert_eq!(group_averaged_xs(&rxn, 1.0, 100.0), 0.0);
    }

    #[test]
    fn a_boundary_on_a_grid_point_is_not_counted_twice() {
        // A ramp whose average is exactly known, cut at a grid point: the two
        // halves must add back to the whole, which they cannot if a shared
        // boundary appears as both an interior point and an edge.
        let rxn = on_grid(vec![0.0, 10.0, 20.0], vec![0.0, 10.0, 20.0]);
        let whole = group_averaged_xs(&rxn, 0.0, 20.0) * 20.0;
        let lower = group_averaged_xs(&rxn, 0.0, 10.0) * 10.0;
        let upper = group_averaged_xs(&rxn, 10.0, 20.0) * 10.0;
        assert!(
            (whole - (lower + upper)).abs() < 1e-12 * whole,
            "{whole} != {lower} + {upper}"
        );
        assert!(
            (whole - 200.0).abs() < 1e-12,
            "ramp integral is 200, got {whole}"
        );
    }

    #[test]
    fn test_scale_rates() {
        let mut rates: ReactionRates = HashMap::new();
        let mut nuc_rates = HashMap::new();
        nuc_rates.insert("(n,gamma)".to_string(), 1.0e-6);
        nuc_rates.insert("(n,2n)".to_string(), 2.0e-8);
        rates.insert("Fe56".to_string(), nuc_rates);

        let scaled = scale_rates(&rates, 0.5);
        assert!((scaled["Fe56"]["(n,gamma)"] - 0.5e-6).abs() < 1e-20);
        assert!((scaled["Fe56"]["(n,2n)"] - 1.0e-8).abs() < 1e-20);

        // Scale by zero
        let zeroed = scale_rates(&rates, 0.0);
        assert!(zeroed["Fe56"]["(n,gamma)"].abs() < 1e-30);
    }

    // --- fission-yield spectrum fold (issue #379) ---

    fn fy_set(energies: &[f64]) -> FissionYieldSet {
        FissionYieldSet::new(
            energies
                .iter()
                .map(|&energy| yani::FissionYield {
                    energy,
                    products: vec![],
                })
                .collect(),
        )
    }

    #[test]
    fn yield_point_shares_sum_to_the_group_average() {
        // The hats are a partition of unity at every evaluation point and the
        // trapezoid rule is linear in its integrand, so splitting the group
        // average across the tabulated energies must not create or destroy any
        // of it. This is what makes the normalized weights a partition of unity.
        let rxn = ramp_reaction(0.0, 2.0e7, 0.0, 5.0, MT_FISSION);
        let set = fy_set(&[0.0253, 5.0e5, 1.4e7]);

        for (e_lo, e_hi) in [
            (0.0, 2.0e7),   // spans every tabulated point
            (1.0e6, 2.0e6), // strictly between two points
            (1.4e7, 2.0e7), // above the last point (flat clamp)
            (0.0, 0.0253),  // below the first point (flat clamp)
            (4.0e5, 6.0e5), // straddles a single point
        ] {
            let mut out = vec![0.0; 3];
            add_group_fission_xs_by_yield_point(
                &rxn,
                &set,
                &set.energies(),
                e_lo,
                e_hi,
                1.0,
                &mut out,
            );
            let split: f64 = out.iter().sum();
            let whole = group_averaged_xs(&rxn, e_lo, e_hi);
            assert!(
                (split - whole).abs() <= 1e-12 * whole.abs().max(1.0),
                "group ({e_lo}, {e_hi}): shares sum to {split}, group average is {whole}"
            );
        }
    }

    #[test]
    fn a_group_inside_one_tabulated_point_puts_everything_there() {
        // A group wholly below the first tabulated energy sees the flat clamp,
        // so all of its fission rate belongs to that first point.
        let rxn = flat_reaction((0.0, 2.0e7), 3.0, MT_FISSION);
        let set = fy_set(&[1.0e6, 1.4e7]);

        let mut out = vec![0.0; 2];
        add_group_fission_xs_by_yield_point(&rxn, &set, &set.energies(), 1.0, 100.0, 1.0, &mut out);
        assert!(out[1].abs() < 1e-30, "nothing belongs to the fast point");
        assert!(out[0] > 0.0, "everything belongs to the thermal point");
    }

    #[test]
    fn a_14_mev_group_selects_the_top_tabulated_point() {
        // The fusion case: a spectrum sitting on the highest tabulated energy
        // must weight that yield vector alone, not the thermal one that
        // `yields.first()` used to pick.
        let rxn = flat_reaction((0.0, 2.0e7), 2.0, MT_FISSION);
        let set = fy_set(&[0.0253, 5.0e5, 1.4e7]);

        let mut out = vec![0.0; 3];
        add_group_fission_xs_by_yield_point(
            &rxn,
            &set,
            &set.energies(),
            1.4e7,
            1.5e7,
            1.0,
            &mut out,
        );
        let total: f64 = out.iter().sum();
        assert!(
            (out[2] / total - 1.0).abs() < 1e-12,
            "expected all weight on 14 MeV, got {out:?}"
        );
    }

    #[test]
    fn shares_scale_linearly_with_the_group_flux() {
        // `scale` carries phi_g, so doubling the group flux doubles its
        // contribution and nothing else changes.
        let rxn = ramp_reaction(0.0, 2.0e7, 1.0, 4.0, MT_FISSION);
        let set = fy_set(&[0.0253, 1.4e7]);

        let mut once = vec![0.0; 2];
        let mut twice = vec![0.0; 2];
        add_group_fission_xs_by_yield_point(
            &rxn,
            &set,
            &set.energies(),
            0.0,
            2.0e7,
            1.0,
            &mut once,
        );
        add_group_fission_xs_by_yield_point(
            &rxn,
            &set,
            &set.energies(),
            0.0,
            2.0e7,
            2.0,
            &mut twice,
        );
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((2.0 * a - b).abs() < 1e-15 * b.abs().max(1.0));
        }
    }

    /// The fold over the old concatenate-sort-dedup point set, to the bit.
    fn scanned_fold(
        reaction: &Reaction,
        fy_set: &FissionYieldSet,
        e_lo: f64,
        e_hi: f64,
        scale: f64,
        out: &mut [f64],
    ) {
        if e_lo >= e_hi || fy_set.yields.is_empty() {
            return;
        }
        let fy_energies = fy_set.energies();
        let mut points = vec![e_lo];
        for grid in [&reaction.energy[..], &fy_energies[..]] {
            points.extend(grid.iter().copied().filter(|&e| e > e_lo && e < e_hi));
        }
        points.push(e_hi);
        points.sort_by(f64::total_cmp);
        points.dedup();

        /// Cross section at a point and the yield hats it falls between,
        /// as `add_group_fission_xs_by_yield_point` spells it.
        type PointTerm = (f64, Option<[(usize, f64); 2]>);
        let per_point: Vec<PointTerm> = points
            .iter()
            .map(|&e| {
                (
                    reaction.cross_section_at(e).unwrap_or(0.0),
                    fy_set.interp_weights(e),
                )
            })
            .collect();
        let width = e_hi - e_lo;
        for i in 0..points.len() - 1 {
            let de = points[i + 1] - points[i];
            if de <= 0.0 {
                continue;
            }
            for (xs, hats) in [&per_point[i], &per_point[i + 1]] {
                if *xs == 0.0 {
                    continue;
                }
                let Some(hats) = hats else { continue };
                for (k, w) in hats {
                    if *w != 0.0 {
                        out[*k] += scale * 0.5 * xs * w * de / width;
                    }
                }
            }
        }
    }

    #[test]
    fn the_merged_fold_is_bit_identical_to_the_sorted_one() {
        // The fold merges two interior slices rather than concatenating and
        // sorting two grids. A yield energy equal to a cross-section grid point
        // is the case where the merge has to emit both and let the zero-width
        // segment fall away; duplicated yield energies are the case where it
        // has to do that within one slice; a repeated cross-section energy is
        // ENDF's step discontinuity. Compared against the old algorithm rather
        // than against a property, because bit-identity IS the requirement.
        let rxn = on_grid(
            vec![0.0, 1.0e6, 1.0e6, 1.4e7, 2.0e7],
            vec![0.0, 2.0, 3.0, 4.0, 5.0],
        );
        for energies in [
            vec![1.0e6, 1.4e7],         // both exactly on a grid point
            vec![0.0253, 1.0e6, 1.4e7], // one off-grid, two on
            vec![1.0e6, 1.0e6, 1.4e7],  // a duplicated yield energy
            vec![5.0e5, 5.0e5, 5.0e5],  // all of them duplicated
            vec![3.0e7],                // entirely above the group
        ] {
            let set = fy_set(&energies);
            let fy_energies = set.energies();
            for (e_lo, e_hi) in [
                (0.0, 2.0e7),
                (1.0e6, 1.4e7),
                (9.0e5, 1.1e6),
                (1.4e7, 2.0e7),
                (2.0e6, 3.0e6),
                (1.0e6, 1.0e6),
            ] {
                let mut merged = vec![0.0; set.yields.len()];
                add_group_fission_xs_by_yield_point(
                    &rxn,
                    &set,
                    &fy_energies,
                    e_lo,
                    e_hi,
                    1.0,
                    &mut merged,
                );
                let mut sorted = vec![0.0; set.yields.len()];
                scanned_fold(&rxn, &set, e_lo, e_hi, 1.0, &mut sorted);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&merged),
                    bits(&sorted),
                    "{energies:?} over ({e_lo}, {e_hi})"
                );
            }
        }
    }

    #[test]
    fn the_fold_still_adds_up_when_a_yield_energy_sits_on_a_grid_point() {
        // Additivity, on a cross section that is continuous: the hats are a
        // partition of unity at every evaluation point, so splitting the group
        // average across the tabulated energies must not create or destroy any
        // of it. A yield energy exactly on a grid point and duplicated yield
        // energies are the two cases the merge could get wrong.
        //
        // Continuous on purpose. Where the cross section jumps -- an energy
        // repeated with two values -- the fold's finer point set legitimately
        // disagrees with `group_averaged_xs`'s coarser one, because
        // `cross_section_at` returns the far side of the jump at the shared
        // energy and refining the interval either side of it changes the
        // trapezoid. That is documented on this function and predates #576;
        // `the_merged_fold_is_bit_identical_to_the_sorted_one` is what covers
        // the discontinuous case.
        let rxn = on_grid(vec![0.0, 1.0e6, 1.4e7, 2.0e7], vec![0.0, 2.0, 4.0, 5.0]);
        for energies in [
            vec![1.0e6, 1.4e7],         // both exactly on a grid point
            vec![0.0253, 1.0e6, 1.4e7], // one off-grid, two on
            vec![1.0e6, 1.0e6, 1.4e7],  // a duplicated yield energy
            vec![5.0e5, 5.0e5, 5.0e5],  // all of them duplicated
        ] {
            let set = fy_set(&energies);
            let fy_energies = set.energies();
            for (e_lo, e_hi) in [
                (0.0, 2.0e7),
                (1.0e6, 1.4e7),
                (9.0e5, 1.1e6),
                (1.4e7, 2.0e7),
                (2.0e6, 3.0e6),
            ] {
                let mut out = vec![0.0; set.yields.len()];
                add_group_fission_xs_by_yield_point(
                    &rxn,
                    &set,
                    &fy_energies,
                    e_lo,
                    e_hi,
                    1.0,
                    &mut out,
                );
                let split: f64 = out.iter().sum();
                let whole = group_averaged_xs(&rxn, e_lo, e_hi);
                assert!(
                    (split - whole).abs() <= 1e-12 * whole.abs().max(1.0),
                    "{energies:?} over ({e_lo}, {e_hi}): shares sum to {split}, \
                     group average is {whole}"
                );
            }
        }
    }

    #[test]
    fn test_energy_groups_explicit_validation() {
        // Valid
        let groups = EnergyGroups::Explicit(vec![1.0, 100.0, 1e6]);
        assert!(groups.resolve().is_ok());

        // Too few
        let bad = EnergyGroups::Explicit(vec![1.0]);
        assert!(bad.resolve().is_err());

        // Not ascending
        let bad2 = EnergyGroups::Explicit(vec![1e6, 100.0, 1.0]);
        assert!(bad2.resolve().is_err());
    }
}

/// The mass number in a nuclide name, `"Ta180_m1"` -> 180.
///
/// `endf::zam` is the crate's own reader for this and already gets the
/// isomeric suffix right, so it is used rather than another hand-rolled split:
/// stripping trailing digits from `Ta180_m1` leaves the state index, not the
/// mass, which is the bug this avoids inheriting.
fn mass_number_of(name: &str) -> Option<f64> {
    endf::data::zam(name).ok().map(|(_z, a, _m)| a as f64)
}

#[cfg(test)]
mod shielding_tests {
    use super::mass_number_of;

    #[test]
    fn a_mass_number_survives_an_isomeric_suffix() {
        assert_eq!(mass_number_of("Fe56"), Some(56.0));
        assert_eq!(mass_number_of("Ta180_m1"), Some(180.0));
        assert_eq!(mass_number_of("U235"), Some(235.0));
        assert_eq!(mass_number_of("nonsense"), None);
        // The suffix must not be read as the mass.
        assert_eq!(mass_number_of("Ag110_m1"), Some(110.0));
    }
}
