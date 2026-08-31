//! Fold MF=33 covariance against the flux to get the covariance of the
//! collapsed reaction rates.
//!
//! # Why this needs no group structure
//!
//! A reaction rate is linear in the cross section, and MF=33 states a
//! covariance that is constant on each cell of the evaluation's own energy
//! grid. Put those together and the covariance of two collapsed rates is
//!
//! ```text
//! Cov(R_i, R_j) = Σ_{k,l} C_ij[k,l] · r_i[k] · r_j[l]
//! ```
//!
//! where `r_i[k]` is reaction `i`'s rate restricted to covariance interval `k`.
//! That is exact, not an approximation, and it never projects the covariance
//! onto the user's group structure: the integration grid IS the covariance
//! grid. It is the same `A Σ Aᵀ` shape the transport side uses, with `A` the
//! partial-rate vector.
//!
//! Blocks are therefore never summed onto a union grid either. Each one folds
//! on its own grid and the contributions add, because the sum over blocks is
//! outside the contraction.
//!
//! # What the rate actually is
//!
//! [`compute_multigroup_reaction_rates`](crate::compute_multigroup_reaction_rates)
//! computes `σ_eff = Σ_g σ_g φ_g / Σ_g φ_g` and then
//! `rate = σ_eff · 1e-24 · Σ_g φ_g`, so the total flux cancels and
//!
//! ```text
//! rate = 1e-24 · Σ_g σ_g φ_g = 1e-24 · ∫ σ(E) ψ(E) dE,   ψ(E) = φ_g / ΔE_g
//! ```
//!
//! with `ψ` the piecewise-constant flux density. Restricting that integral to a
//! covariance interval is exactly what a partial rate is, and it is computed
//! with `multigroup::group_averaged_xs` over each
//! overlap rather than a second integrator.
//!
//! # Where the uncertainty is diluted, and why that is right
//!
//! A covariance grid need not span the whole flux range. Rate that comes from
//! outside it is rate the evaluation states no uncertainty for, so it enters
//! `R_i` (the denominator) and not the sum (the numerator), and the relative
//! uncertainty comes out smaller than the covariance grid alone would suggest.
//! That is the honest answer rather than a bug, but it is also invisible, which
//! is why [`Coverage`] records the fraction of each rate the covariance grid
//! actually covered.

use std::collections::{BTreeMap, BTreeSet};

use yamc_materials::Material;
use yamc_nuclide::covariance::expand::{expand_ni, ExpandedBlock, Scale, Unsupported};
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yamc_nuclide::reaction::Reaction;
use yani::reactions::reaction_type_to_mt;
use yani::{ChainNuclide, ReactionRates};

use crate::multigroup::group_averaged_xs;

/// Barns to cm^2, the factor the rate collapse already carries.
const BARN_TO_CM2: f64 = 1.0e-24;

/// The relative covariance of one nuclide's collapsed reaction rates.
///
/// Indexed by reaction KIND rather than by MT, because that is what
/// [`ReactionRates`] is keyed by and what the sampler has to perturb. `kinds`
/// is sorted, so the matrix layout is reproducible across runs.
#[derive(Debug, Clone, PartialEq)]
pub struct RateCovariance {
    /// Reaction kinds, sorted, indexing both axes.
    pub kinds: Vec<String>,
    /// Row-major `kinds.len()^2` relative covariance:
    /// `Cov(R_i, R_j) / (R_i · R_j)`.
    pub relative: Vec<f64>,
}

impl RateCovariance {
    pub fn n(&self) -> usize {
        self.kinds.len()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.relative[i * self.n() + j]
    }

    /// The relative standard deviation of each rate: the square root of the
    /// diagonal.
    pub fn relative_std_devs(&self) -> Vec<f64> {
        (0..self.n())
            .map(|i| self.get(i, i).max(0.0).sqrt())
            .collect()
    }
}

/// What the fold could and could not use.
///
/// Every field here exists so that a gap is reportable. A nuclide with no
/// published covariance and a nuclide whose covariance is genuinely zero must
/// not look the same downstream, and neither must a block that was skipped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Coverage {
    /// Nuclides that contributed at least one usable block.
    pub covered: BTreeSet<String>,
    /// Nuclides in the chain that had rates but no covariance data at all.
    pub without_data: BTreeSet<String>,
    /// Blocks correlating with another evaluation (`mat1 != 0`), not consumed.
    pub skipped_cross_material: usize,
    /// NC blocks (a covariance derived from other reactions), not consumed.
    pub skipped_nc: usize,
    /// Blocks whose `lb` layout is not implemented, counted per `lb`.
    pub unsupported_layouts: BTreeMap<i64, usize>,
    /// Blocks whose arrays did not match their own declared sizes.
    pub malformed: usize,
    /// Per (nuclide, kind), the fraction of the rate the covariance grid spans.
    ///
    /// Below one means part of the rate comes from energies the evaluation
    /// gives no covariance for, so the relative uncertainty is diluted by
    /// exactly that much.
    pub rate_fraction_covered: BTreeMap<(String, String), f64>,
}

impl Coverage {
    /// Fold one nuclide's report into this one.
    ///
    /// Every field is order-free: the sets union, the counters sum, and
    /// `rate_fraction_covered` is keyed by (nuclide, kind) and takes the
    /// smaller claim. That is what lets the fold below run per nuclide in
    /// parallel and merge afterwards (issue #576, finding 5c).
    pub fn absorb(&mut self, other: Coverage) {
        self.covered.extend(other.covered);
        self.without_data.extend(other.without_data);
        self.skipped_cross_material += other.skipped_cross_material;
        self.skipped_nc += other.skipped_nc;
        self.malformed += other.malformed;
        for (lb, n) in other.unsupported_layouts {
            *self.unsupported_layouts.entry(lb).or_insert(0) += n;
        }
        for (key, fraction) in other.rate_fraction_covered {
            self.rate_fraction_covered
                .entry(key)
                .and_modify(|f| *f = f.min(fraction))
                .or_insert(fraction);
        }
    }

    /// Whether anything at all was skipped or missing.
    pub fn has_gaps(&self) -> bool {
        !self.without_data.is_empty()
            || self.skipped_cross_material > 0
            || self.skipped_nc > 0
            || !self.unsupported_layouts.is_empty()
            || self.malformed > 0
    }
}

/// The flux density `ψ(E) = φ_g / ΔE_g`, as the group structure and the fluxes.
///
/// Kept as the two vectors rather than as a function, because every integral
/// below is over the overlap of an arbitrary interval with the groups, and the
/// overlap is what the trapezoid integrator needs.
struct FluxDensity<'a> {
    boundaries: &'a [f64],
    flux: &'a [f64],
}

impl FluxDensity<'_> {
    /// The groups that can overlap `[a, b]`, as a range over `flux`.
    ///
    /// A covariance block's energy grid is coarse -- a handful of intervals --
    /// while the flux grid is the user's, up to 1968 groups, so both integrals
    /// below used to walk every group to find the two or three that overlap.
    /// The boundaries ascend (`transmute_material_shielded` refuses a spectrum
    /// where they do not), so the ends bisect. Bit-identical: the groups this
    /// leaves out are exactly the ones the `lo >= hi` guard skipped, and the
    /// ones it keeps are summed in the same order (issue #576, finding 8).
    fn overlapping(&self, a: f64, b: f64) -> std::ops::Range<usize> {
        // Group `g` spans `[boundaries[g], boundaries[g + 1]]`, so it can
        // overlap when `boundaries[g + 1] > a` and `boundaries[g] < b`.
        let start = self
            .boundaries
            .partition_point(|&e| e <= a)
            .saturating_sub(1);
        let end = self
            .boundaries
            .partition_point(|&e| e < b)
            .min(self.flux.len());
        start.min(end)..end
    }

    /// `∫_a^b σ(E) ψ(E) dE`, in barn n/cm^2/s.
    ///
    /// Group by group, because `ψ` is constant within a group and the cross
    /// section is not: on each overlap the integral is the group-averaged cross
    /// section over that overlap times its width times the group's own density.
    fn integrate_xs(&self, reaction: &Reaction, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let mut total = 0.0;
        for g in self.overlapping(a, b) {
            let (glo, ghi) = (self.boundaries[g], self.boundaries[g + 1]);
            let lo = a.max(glo);
            let hi = b.min(ghi);
            if lo >= hi || ghi <= glo {
                continue;
            }
            let density = self.flux[g] / (ghi - glo);
            total += group_averaged_xs(reaction, lo, hi) * (hi - lo) * density;
        }
        total
    }

    /// `∫_a^b ψ(E) dE`, in n/cm^2/s. Needed only by absolute (`lb = 0`) blocks,
    /// whose covariance multiplies the flux rather than the rate.
    fn integrate_flux(&self, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let mut total = 0.0;
        for g in self.overlapping(a, b) {
            let (glo, ghi) = (self.boundaries[g], self.boundaries[g + 1]);
            let lo = a.max(glo);
            let hi = b.min(ghi);
            if lo >= hi || ghi <= glo {
                continue;
            }
            total += self.flux[g] * (hi - lo) / (ghi - glo);
        }
        total
    }
}

/// Reaction `i`'s partial rates over one block's grid, and their total.
///
/// The total is NOT the reaction's full rate: it is only the part the grid
/// spans, which is what the coverage fraction is measured against.
struct Partials {
    /// Per interval, in 1/s.
    per_interval: Vec<f64>,
}

impl Partials {
    fn total(&self) -> f64 {
        self.per_interval.iter().sum()
    }
}

/// The partial rates of `reaction` over the intervals of `grid`.
fn partial_rates(flux: &FluxDensity, reaction: &Reaction, grid: &[f64], scale: Scale) -> Partials {
    let n = grid.len().saturating_sub(1);
    let mut per_interval = Vec::with_capacity(n);
    for k in 0..n {
        let v = match scale {
            // A relative covariance multiplies the rate, so the weight is the
            // partial RATE.
            Scale::Relative => BARN_TO_CM2 * flux.integrate_xs(reaction, grid[k], grid[k + 1]),
            // An absolute covariance is already in barns squared, so the weight
            // is the partial FLUX and the cross section must not appear twice.
            Scale::Absolute => BARN_TO_CM2 * flux.integrate_flux(grid[k], grid[k + 1]),
        };
        per_interval.push(v);
    }
    Partials { per_interval }
}

/// Contract `rᵢᵀ C rⱼ`.
fn contract(block: &ExpandedBlock, row: &Partials, col: &Partials) -> f64 {
    let mut total = 0.0;
    for i in 0..block.n_rows().min(row.per_interval.len()) {
        let ri = row.per_interval[i];
        if ri == 0.0 {
            continue;
        }
        for j in 0..block.n_cols().min(col.per_interval.len()) {
            total += block.get(i, j) * ri * col.per_interval[j];
        }
    }
    total
}

/// The activation MTs of a chain nuclide, paired with the kind that names them.
///
/// Sorted by kind so the matrix layout is reproducible: the same material and
/// chain must give the same index for the same reaction on every run, which is
/// what makes a seeded perturbation reproducible at all.
fn kinds_and_mts(chain_nuclide: &ChainNuclide) -> Vec<(String, i32)> {
    let mut out: Vec<(String, i32)> = chain_nuclide
        .reactions
        .iter()
        .filter_map(|r| reaction_type_to_mt(&r.kind).map(|mt| (r.kind.clone(), mt)))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Fold one nuclide's covariance blocks against the flux.
///
/// Returns `None` when the nuclide contributed nothing: no covariance data, or
/// none of it usable. The caller records which of those it was.
#[allow(clippy::too_many_arguments)]
fn fold_nuclide(
    flux: &FluxDensity,
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
    kinds: &[(String, i32)],
    rates: &BTreeMap<String, f64>,
    nuclide: &str,
    coverage: &mut Coverage,
) -> Option<RateCovariance> {
    let n = kinds.len();
    if n == 0 {
        return None;
    }
    let index: BTreeMap<i32, usize> = kinds
        .iter()
        .enumerate()
        .map(|(i, (_, mt))| (*mt, i))
        .collect();

    // Absolute covariance, in (1/s)^2, before relativizing.
    let mut absolute = vec![0.0; n * n];
    let mut used = 0;
    // Per MT, the largest span of rate any block covered. Blocks may overlap,
    // so this is a bound on coverage rather than a sum over blocks.
    let mut covered_rate: BTreeMap<i32, f64> = BTreeMap::new();

    for block in blocks {
        if block.is_cross_material() {
            coverage.skipped_cross_material += 1;
            continue;
        }
        let ni = match &block.data {
            CovarianceData::Ni(ni) => ni,
            CovarianceData::Nc(_) => {
                coverage.skipped_nc += 1;
                continue;
            }
        };

        let (row_mt, col_mt) = (block.mt, block.partner_mt());
        let (Some(&i), Some(&j)) = (index.get(&row_mt), index.get(&col_mt)) else {
            // A covariance for a reaction this chain does not drive. Not a gap:
            // there is no rate for it to be the uncertainty of.
            continue;
        };
        let (Some(row_rx), Some(col_rx)) = (reactions.get(&row_mt), reactions.get(&col_mt)) else {
            continue;
        };

        let expanded = match expand_ni(ni) {
            Ok(e) => e,
            Err(Unsupported::Layout(lb)) => {
                *coverage.unsupported_layouts.entry(lb).or_insert(0) += 1;
                continue;
            }
            Err(Unsupported::Malformed) => {
                coverage.malformed += 1;
                continue;
            }
        };
        if expanded.is_empty() {
            continue;
        }

        let row = partial_rates(flux, row_rx, &expanded.row_energies, expanded.scale);
        let col = partial_rates(flux, col_rx, &expanded.col_energies, expanded.scale);
        let contribution = contract(&expanded, &row, &col);

        absolute[i * n + j] += contribution;
        if i != j {
            // The tape stores each cross-reaction pair once, with MT1 >= MT,
            // and the (MT1, MT) block IS this one's transpose. Filling both
            // halves here is what that means; the file never carries the
            // partner separately for it to be added twice.
            absolute[j * n + i] += contribution;
        }

        if expanded.scale == Scale::Relative {
            let e = covered_rate.entry(row_mt).or_insert(0.0);
            *e = e.max(row.total().abs());
            let e = covered_rate.entry(col_mt).or_insert(0.0);
            *e = e.max(col.total().abs());
        }
        used += 1;
    }

    if used == 0 {
        return None;
    }

    // Relativize with the FULL rates, so rate from outside every covariance
    // grid dilutes the uncertainty exactly as much as it should.
    let mut relative = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (ri, rj) = (rates.get(&kinds[i].0), rates.get(&kinds[j].0));
            match (ri, rj) {
                (Some(&ri), Some(&rj)) if ri != 0.0 && rj != 0.0 => {
                    relative[i * n + j] = absolute[i * n + j] / (ri * rj);
                }
                _ => {}
            }
        }
    }

    for (kind, mt) in kinds {
        if let (Some(&covered), Some(&full)) = (covered_rate.get(mt), rates.get(kind)) {
            if full != 0.0 {
                coverage.rate_fraction_covered.insert(
                    (nuclide.to_string(), kind.clone()),
                    (covered / full).min(1.0),
                );
            }
        }
    }

    coverage.covered.insert(nuclide.to_string());
    Some(RateCovariance {
        kinds: kinds.iter().map(|(k, _)| k.clone()).collect(),
        relative,
    })
}

/// Fold every transmutable nuclide's covariance against one spectrum.
///
/// `rates` must be the unit-flux rates from
/// [`compute_multigroup_reaction_rates`](crate::compute_multigroup_reaction_rates)
/// for the SAME spectrum, because the relativization divides by them. Relative
/// covariance is invariant under the per-step `scale_rates`, which is why this
/// is computed once per distinct spectrum rather than once per step.
pub fn fold_rate_covariance(
    material: &Material,
    chain: &std::collections::HashMap<String, ChainNuclide>,
    rates: &ReactionRates,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
) -> (BTreeMap<String, RateCovariance>, Coverage) {
    let mut out = BTreeMap::new();
    let mut coverage = Coverage::default();

    if multigroup_flux.is_empty() || group_boundaries.len() != multigroup_flux.len() + 1 {
        return (out, coverage);
    }
    let flux = FluxDensity {
        boundaries: group_boundaries,
        flux: multigroup_flux,
    };

    // Sorted, so the fold visits nuclides in the same order every run.
    let mut names: Vec<&String> = rates.keys().collect();
    names.sort();

    // One nuclide's fold reads only its own covariance blocks, its own rates
    // and the shared flux, so the nuclides are independent. Each keeps its own
    // `Coverage` and the driver merges them in `names` order below; every field
    // of `Coverage` is order-free anyway, but merging in a fixed order costs
    // nothing and leaves nothing to argue about (issue #576, finding 5c).
    let one = |name: &&String| -> (Option<RateCovariance>, Coverage) {
        let mut coverage = Coverage::default();
        let Some(chain_nuclide) = chain.get(*name) else {
            return (None, coverage);
        };
        let Some(nuclide_data) = material.nuclide_data.get(*name) else {
            return (None, coverage);
        };
        let Some(blocks) = nuclide_data.covariance.as_ref() else {
            coverage.without_data.insert((*name).clone());
            return (None, coverage);
        };

        let temperature = if material.temperature().is_empty() {
            match crate::default_temperature(nuclide_data) {
                Some(t) => t,
                None => return (None, coverage),
            }
        } else {
            material.temperature().to_string()
        };
        let Some(by_mt) = nuclide_data.reactions_for_temp(&temperature) else {
            return (None, coverage);
        };

        let kinds = kinds_and_mts(chain_nuclide);
        let reactions: BTreeMap<i32, &Reaction> = kinds
            .iter()
            .filter_map(|(_, mt)| by_mt.get(mt).map(|r| (*mt, r.as_ref())))
            .collect();
        let nuclide_rates: BTreeMap<String, f64> = rates
            .get(*name)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();

        let folded = fold_nuclide(
            &flux,
            blocks,
            &reactions,
            &kinds,
            &nuclide_rates,
            name,
            &mut coverage,
        );
        if folded.is_none() {
            coverage.without_data.insert((*name).clone());
        }
        (folded, coverage)
    };

    let folded: Vec<(Option<RateCovariance>, Coverage)> = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            names.par_iter().map(one).collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            names.iter().map(one).collect()
        }
    };
    for (name, (cov, local)) in names.iter().zip(folded) {
        coverage.absorb(local);
        if let Some(cov) = cov {
            out.insert((*name).clone(), cov);
        }
    }

    (out, coverage)
}

#[cfg(test)]
mod integration_range_tests {
    use super::*;

    /// A reaction with a known non-trivial shape, so a dropped group shows.
    fn ramp(e0: f64, e1: f64, xs0: f64, xs1: f64) -> Reaction {
        Reaction {
            cross_section: vec![xs0, xs1].into(),
            threshold_idx: 0,
            energy: vec![e0, e1].into(),
            mt_number: 102,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// The scan the bisection replaced, kept as the reference it is compared to.
    fn scanned_xs(density: &FluxDensity<'_>, reaction: &Reaction, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let mut total = 0.0;
        for g in 0..density.flux.len() {
            let (glo, ghi) = (density.boundaries[g], density.boundaries[g + 1]);
            let (lo, hi) = (a.max(glo), b.min(ghi));
            if lo >= hi || ghi <= glo {
                continue;
            }
            total +=
                group_averaged_xs(reaction, lo, hi) * (hi - lo) * (density.flux[g] / (ghi - glo));
        }
        total
    }

    fn scanned_flux(density: &FluxDensity<'_>, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let mut total = 0.0;
        for g in 0..density.flux.len() {
            let (glo, ghi) = (density.boundaries[g], density.boundaries[g + 1]);
            let (lo, hi) = (a.max(glo), b.min(ghi));
            if lo >= hi || ghi <= glo {
                continue;
            }
            total += density.flux[g] * (hi - lo) / (ghi - glo);
        }
        total
    }

    #[test]
    fn bisecting_the_group_range_changes_no_bits() {
        // A grid with uneven widths, so an off-by-one at either end moves the
        // answer rather than adding a group that happens to contribute the
        // same amount.
        let boundaries: Vec<f64> = vec![1.0e-5, 0.625, 10.0, 1.0e3, 1.0e5, 1.0e6, 1.4e7, 2.0e7];
        let flux: Vec<f64> = vec![1.0, 3.0, 7.0, 2.0, 11.0, 5.0, 13.0];
        let density = FluxDensity {
            boundaries: &boundaries,
            flux: &flux,
        };
        let reaction = ramp(0.0, 2.0e7, 0.5, 9.0);

        for (a, b) in [
            (1.0e-5, 2.0e7), // the whole grid, both ends exactly on a boundary
            (0.625, 1.0e6),  // both ends on interior boundaries
            (5.0, 500.0),    // strictly inside two groups
            (0.0, 1.0e-5),   // entirely below the grid
            (2.0e7, 3.0e7),  // entirely at and above the top
            (1.0e5, 1.0e5),  // zero width, on a boundary
            (9.9, 10.1),     // straddles one boundary
            (1.0e-6, 0.7),   // starts below the grid and ends inside
            (1.3e7, 2.5e7),  // starts inside and ends above
        ] {
            assert_eq!(
                density.integrate_xs(&reaction, a, b).to_bits(),
                scanned_xs(&density, &reaction, a, b).to_bits(),
                "integrate_xs over ({a}, {b})"
            );
            assert_eq!(
                density.integrate_flux(a, b).to_bits(),
                scanned_flux(&density, a, b).to_bits(),
                "integrate_flux over ({a}, {b})"
            );
        }
    }
}
