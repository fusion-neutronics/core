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
//! is why [`Coverage`] records the fraction of each rate that carries a stated
//! uncertainty.
//!
//! Spanning an energy is not the same as stating an uncertainty there. A grid
//! can run across the whole range with a variance of zero on some intervals,
//! and rate from those intervals dilutes exactly as rate from outside the grid
//! does. ENDF/B-VIII.1 W186 `(n,gamma)` is the case: its one self-covariance
//! block starts with a single interval, 1e-5 eV to 10 keV, whose variance is
//! zero, because the evaluation puts the resonance-range uncertainty in MF=32.
//! Nearly all of a capture rate comes from there, so the coverage counts only
//! rate from energies whose stated variance is nonzero.

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
    /// Per (nuclide, kind), the share of the dilute rate that comes from
    /// energies where the evaluation states a nonzero variance for that
    /// reaction.
    ///
    /// Exactly: the dilute rate integrated over the energies where the diagonal
    /// of the reaction's own covariance, summed over every self-covariance
    /// block the fold consumed, is nonzero, divided by the dilute rate over the
    /// whole flux range. Rate from an interval a grid spans with a variance of
    /// zero counts as uncovered, the same as rate from outside every grid,
    /// because neither carries a stated uncertainty. Below one, part of the
    /// dilute rate enters the relative covariance's denominator and not its
    /// numerator; how much that lowers the relative sigma also depends on how
    /// the stated variance is spread over the covered part, so the share is
    /// not itself a dilution factor.
    ///
    /// Both integrals are the fold's own, the dilute cross section against the
    /// flux the partial rates are weighted with, and the numerator adds a
    /// subset of the denominator's terms in the same order. So the share lies
    /// in [0, 1] without a clamp, and it does not depend on how the rate the
    /// covariance is divided by was computed. It describes the dilute rate
    /// only: on a self-shielded or a tallied rate, the share of THAT rate
    /// coming from covered energies is not computed (the fold has only the
    /// dilute cross section to split a rate by energy with), and the dilution
    /// the fold then applies differs from this share; where the partials add
    /// up to more than that rate, [`Coverage::partials_above_rate`] lists it.
    /// On a dilute collapse under the flat-within-group weight, the denominator
    /// is that rate to a few parts in 1e15. Under the `1/E` weight it is only
    /// where no covariance edge cuts a group, for the reason given on
    /// [`Coverage::partials_above_rate`].
    ///
    /// Every channel a consumed block names has an entry, unless its dilute
    /// rate over the flux range is zero (a threshold above the spectrum's top
    /// edge), which has no share to report. So a channel whose blocks state no
    /// variance anywhere reads zero rather than being absent.
    pub rate_fraction_covered: BTreeMap<(String, String), f64>,
    /// Per (nuclide, kind), where the partial rates the covariance was weighted
    /// with add up to more than the rate it was divided by, their ratio to it.
    ///
    /// The sum is every partial rate of one relative block's grid, zero
    /// variance intervals included, since the fold weights with all of them;
    /// the largest such sum over the channel's blocks is the one compared. A
    /// part of a rate cannot exceed the rate, and this is past rounding, so an
    /// entry means the numerator and the denominator of the relative
    /// covariance were computed two different ways: the channel's relative
    /// sigma is overstated. The partials are the dilute cross section against
    /// a flux that is flat within each group, so any rate computed otherwise,
    /// a self-shielded or a tallied one among them, can land here. Reported
    /// rather than clamped away, since the clamp is what used to hide it.
    ///
    /// Under the `1/E` within-group weight (`Weighting::OneOverE`) a dilute
    /// run can land here too. A partial over the part of a group a covariance
    /// edge cuts off is that part's own lethargy average times its share of
    /// the group's energy width, and those parts do not add up to the group's
    /// lethargy average times its flux, which is what the collapse gives. The
    /// error is the partials' and can be large: Fe56 `(n,p)` on three groups
    /// with its 4.3 MeV edge inside the fast one sums to about ten times its
    /// rate. Where no covariance edge falls inside a group the two agree.
    ///
    /// Absolute (`lb = 0`) blocks weight with partial fluxes rather than
    /// partial rates, which have no rate to compare with, so they are not
    /// checked. Absence from this map therefore says the relative blocks are
    /// consistent, not that an absolute one is.
    pub partials_above_rate: BTreeMap<(String, String), f64>,
    /// The production this spectrum drove, each channel's weighted by its
    /// share in [`Coverage::rate_fraction_covered`], and the production it
    /// drove in total. Both are per barn-cm per second, and both are weighted
    /// by the parent's own density, so a channel on a trace isotope counts for
    /// what it actually made.
    ///
    /// Each channel's production is the rate this run used, which may be
    /// self-shielded or tallied, while its share is of the dilute rate. On a
    /// dilute collapse the covered sum is exactly the production from energies
    /// where a covariance states a nonzero variance. Otherwise it is that
    /// production only if the rate kept the dilute rate's distribution in
    /// energy, which shielding does not: it depresses the resonance range,
    /// where capture blocks often state zero. The covered share of a shielded
    /// or tallied production is not computed.
    ///
    /// Kept as two sums rather than as their ratio because sums merge and a
    /// ratio does not: the fold runs per nuclide and a schedule can name more
    /// than one spectrum, and each part adds its own share without having to
    /// know how many parts there are. [`Coverage::rate_fraction_total`]
    /// divides them.
    pub covered_production: f64,
    pub total_production: f64,
}

impl Coverage {
    /// The production-weighted mean of the per-channel dilute shares, weighted
    /// by the rate this run used and by parent density.
    ///
    /// On a dilute run that is the share of the production driven from
    /// energies where a covariance states a nonzero variance. On a
    /// self-shielded or tallied run it is not, for the reason given on
    /// [`Coverage::covered_production`].
    ///
    /// The number to read before any sigma from this fold. Counting nuclides
    /// with MF=33 answers a different and much weaker question: an evaluation
    /// can carry covariance for every isotope in the material and state none of
    /// it for the channel that makes the product of interest, which leaves the
    /// count reading as full coverage while the ensemble perturbs almost
    /// nothing. Tungsten is the case that shows it, where two major libraries
    /// state covariance for all five natural isotopes, give it for `(n,3n)` and
    /// `(n,gamma)`, and give none for the `(n,2n)` that makes 98% of the decay
    /// heat.
    ///
    /// `None` when this run drove no production at all, which is a decay-only
    /// schedule and has no fraction to report rather than a fraction of zero.
    pub fn rate_fraction_total(&self) -> Option<f64> {
        (self.total_production > 0.0)
            .then(|| (self.covered_production / self.total_production).clamp(0.0, 1.0))
    }

    /// Fold one nuclide's report into this one.
    ///
    /// Every field is order-free: the sets union, the counters sum,
    /// `rate_fraction_covered` is keyed by (nuclide, kind) and takes the
    /// smaller claim, and `partials_above_rate` takes the larger excess. That
    /// is what lets the fold below run per nuclide in parallel and merge
    /// afterwards (issue #576, finding 5c).
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
        for (key, ratio) in other.partials_above_rate {
            self.partials_above_rate
                .entry(key)
                .and_modify(|r| *r = r.max(ratio))
                .or_insert(ratio);
        }
        self.covered_production += other.covered_production;
        self.total_production += other.total_production;
    }

    /// Whether anything at all was skipped, missing or inconsistent.
    pub fn has_gaps(&self) -> bool {
        !self.without_data.is_empty()
            || self.skipped_cross_material > 0
            || self.skipped_nc > 0
            || !self.unsupported_layouts.is_empty()
            || self.malformed > 0
            || !self.partials_above_rate.is_empty()
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

/// Reaction `i`'s partial rates over one block's grid.
struct Partials {
    /// Per interval, in 1/s.
    per_interval: Vec<f64>,
}

impl Partials {
    /// Every interval's partial, zero variance or not, since the fold weights
    /// the block with all of them.
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

/// How far above one a channel's summed partial rates over its rate may sit
/// and still be rounding.
///
/// On a dilute collapse under the flat-within-group weight the two add the
/// same terms grouped differently, and on CCFE-709 they agree to a few parts
/// in 1e15. Under the `1/E` weight they need not, see
/// `Coverage::partials_above_rate`. This sits six orders of magnitude above
/// that, and an excess below it would move a relative sigma by less than a
/// part in a billion.
const PARTIALS_ROUNDING: f64 = 1.0e-9;

/// The interval of `grid` that holds all of `[a, b]`, if one does.
fn interval_holding(grid: &[f64], a: f64, b: f64) -> Option<usize> {
    let k = grid.partition_point(|&e| e <= a).checked_sub(1)?;
    (k + 1 < grid.len() && b <= grid[k + 1]).then_some(k)
}

/// Where one self-covariance block states a variance: its matrix on the
/// diagonal `E = E'`, over the energies where that is nonzero.
struct Diagonal {
    scale: Scale,
    /// `(lo, hi, variance)`, ascending and disjoint, with the zeros left out.
    pieces: Vec<(f64, f64, f64)>,
}

impl Diagonal {
    /// Read from the matrix rather than from the layout, so every `lb` is read
    /// alike. An energy sits in one row interval and one column interval at
    /// once, which on the square layouts is the same index and on the
    /// rectangular ones (`lb` 3 and 6) is wherever the two grids overlap.
    fn of(block: &ExpandedBlock) -> Self {
        let mut edges: Vec<f64> = block
            .row_energies
            .iter()
            .chain(&block.col_energies)
            .copied()
            .collect();
        edges.sort_by(f64::total_cmp);
        edges.dedup();
        let pieces = edges
            .windows(2)
            .filter_map(|w| {
                let i = interval_holding(&block.row_energies, w[0], w[1])?;
                let j = interval_holding(&block.col_energies, w[0], w[1])?;
                let variance = block.get(i, j);
                (variance != 0.0).then_some((w[0], w[1], variance))
            })
            .collect();
        Self {
            scale: block.scale,
            pieces,
        }
    }

    /// The variance stated over `[a, b]`, which lies inside one piece or
    /// outside all of them because `a` and `b` are adjacent edges of a grid
    /// that includes every piece's own.
    fn over(&self, a: f64, b: f64) -> f64 {
        match self.pieces.partition_point(|p| p.0 <= a).checked_sub(1) {
            Some(k) if b <= self.pieces[k].1 => self.pieces[k].2,
            _ => 0.0,
        }
    }
}

/// The share of `reaction`'s dilute rate over the flux range that comes from
/// energies where its own blocks state a nonzero variance, or `None` when that
/// rate is zero.
///
/// Walked on the union of the blocks' edges and the flux range's two ends, so
/// every cell lies inside one piece of each block or outside it, and a cell
/// counts when the variances stated over it sum to something nonzero. Summed
/// rather than tested one by one because the blocks of one subsection add, so
/// two whose tape values cancel exactly, bit for bit, state no variance there.
/// No tolerance is applied: values that cancel only to rounding count as
/// stated, since calling them zero would be a judgement the tape does not
/// make. Relative and absolute blocks are summed apart, having different
/// units.
///
/// The covered sum adds a subset of the total's terms in the same order, and
/// rounded addition of terms that are not negative is monotone, so the share
/// cannot exceed one. A block that states a variance on every interval across
/// the whole flux range therefore reads exactly one.
fn stated_variance_share(
    flux: &FluxDensity,
    reaction: &Reaction,
    diagonals: &[Diagonal],
) -> Option<f64> {
    let (&lo, &hi) = (flux.boundaries.first()?, flux.boundaries.last()?);
    let mut edges: Vec<f64> = diagonals
        .iter()
        .flat_map(|d| d.pieces.iter().flat_map(|&(lo, hi, _)| [lo, hi]))
        .chain([lo, hi])
        .collect();
    edges.sort_by(f64::total_cmp);
    edges.dedup();
    let (mut covered, mut total) = (0.0, 0.0);
    for w in edges.windows(2) {
        let (a, b) = (w[0], w[1]);
        let rate = flux.integrate_xs(reaction, a, b);
        total += rate;
        let (mut relative, mut absolute) = (0.0, 0.0);
        for d in diagonals {
            match d.scale {
                Scale::Relative => relative += d.over(a, b),
                Scale::Absolute => absolute += d.over(a, b),
            }
        }
        if relative != 0.0 || absolute != 0.0 {
            covered += rate;
        }
    }
    (total != 0.0).then(|| covered / total)
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
    // Per MT named by a consumed block, the variance each of its own blocks
    // states. Kept block by block rather than summed onto one grid, because
    // the blocks need not share a grid; `stated_variance_share` walks them
    // together.
    let mut variances: BTreeMap<i32, Vec<Diagonal>> = BTreeMap::new();
    // Per MT, the largest sum of partial rates any relative block weighted it
    // with, zero variance intervals included. The consistency check against
    // the rate, and not the share: it is what the covariance's numerator was
    // built from, whatever the block states.
    let mut weighted_rate: BTreeMap<i32, f64> = BTreeMap::new();

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

        // Only a reaction's own blocks state its variance. A cross block
        // correlates two reactions and gives neither a variance, so it names
        // both channels and covers neither.
        variances.entry(col_mt).or_default();
        let own = variances.entry(row_mt).or_default();
        if row_mt == col_mt {
            own.push(Diagonal::of(&expanded));
        }
        if expanded.scale == Scale::Relative {
            for (mt, partials) in [(row_mt, &row), (col_mt, &col)] {
                let sum = partials.total().abs();
                let e = weighted_rate.entry(mt).or_insert(0.0);
                *e = e.max(sum);
            }
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
        let (Some(diagonals), Some(reaction)) = (variances.get(mt), reactions.get(mt)) else {
            continue;
        };
        let key = (nuclide.to_string(), kind.clone());
        if let Some(share) = stated_variance_share(flux, reaction, diagonals) {
            coverage.rate_fraction_covered.insert(key.clone(), share);
        }
        if let (Some(&weighted), Some(&full)) = (weighted_rate.get(mt), rates.get(kind)) {
            if full != 0.0 && weighted / full > 1.0 + PARTIALS_ROUNDING {
                coverage.partials_above_rate.insert(key, weighted / full);
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

    // How much of the production this spectrum drove is covered, weighted by
    // rate and by the parent's own density. The share is of the dilute rate
    // whatever `rates` is, see `Coverage::covered_production`. Done here
    // rather than per nuclide because it is a property of the material: the
    // per-nuclide fold knows its own rates but not how many atoms of it there
    // are, and a channel on a 0.1%-abundance isotope must not count the same
    // as one on the bulk.
    //
    // Weighted by rate rather than counted per channel for the same reason the
    // per-channel fraction exists: a channel with no covariance costs nothing
    // if nothing went through it.
    let densities = material.get_atoms_per_barn_cm().unwrap_or_default();
    for (nuclide, kinds) in rates {
        let density = densities.get(nuclide).copied().unwrap_or(0.0);
        if density <= 0.0 {
            continue;
        }
        for (kind, rate) in kinds {
            let production = density * rate;
            coverage.total_production += production;
            let fraction = coverage
                .rate_fraction_covered
                .get(&(nuclide.clone(), kind.clone()))
                .copied()
                .unwrap_or(0.0);
            coverage.covered_production += production * fraction;
        }
    }

    (out, coverage)
}

#[cfg(test)]
mod coverage_total_tests {
    use super::*;

    /// Two nuclides, one covered and one not, weighted by what each produced.
    ///
    /// The point of the rate weighting: counting nuclides would call this half
    /// covered, and half the production went through the channel with nothing
    /// stated about it.
    #[test]
    fn the_total_is_weighted_by_production_and_not_by_nuclide_count() {
        let mut a = Coverage {
            covered_production: 3.0,
            total_production: 4.0,
            ..Default::default()
        };
        let b = Coverage {
            covered_production: 0.0,
            total_production: 4.0,
            ..Default::default()
        };
        a.absorb(b);
        // Sums merge; the ratio is taken once at the end over both.
        assert_eq!(a.covered_production, 3.0);
        assert_eq!(a.total_production, 8.0);
        assert_eq!(a.rate_fraction_total(), Some(0.375));
    }

    /// A decay-only schedule drove no production, so there is no share to
    /// report. Zero would be the wrong answer: it reads as "nothing is
    /// covered", which is a statement about the data rather than about there
    /// being nothing to cover.
    #[test]
    fn no_production_reports_no_fraction_rather_than_zero() {
        assert_eq!(Coverage::default().rate_fraction_total(), None);
    }

    /// Floating point can put the ratio a hair over one when every channel is
    /// fully covered; a coverage of 100.0000001% is not a thing to print.
    #[test]
    fn full_coverage_cannot_exceed_one() {
        let c = Coverage {
            covered_production: 1.0 + f64::EPSILON,
            total_production: 1.0,
            ..Default::default()
        };
        assert_eq!(c.rate_fraction_total(), Some(1.0));
    }
}

#[cfg(test)]
mod stated_variance_tests {
    use super::*;
    use endf::mf::covariance::NiSubsection;

    /// Uneven groups with no boundary at 10 keV, so the covariance edge there
    /// falls inside a group and the overlap is exercised rather than avoided.
    const BOUNDARIES: [f64; 8] = [1.0e-5, 0.625, 10.0, 1.0e3, 1.0e5, 1.0e6, 1.4e7, 2.0e7];
    const FLUX: [f64; 7] = [1.0, 3.0, 7.0, 2.0, 11.0, 5.0, 13.0];

    fn flux() -> FluxDensity<'static> {
        FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
        }
    }

    /// A falling cross section, so where the rate comes from matters.
    fn reaction(mt: i32) -> Reaction {
        Reaction {
            cross_section: vec![100.0, 1.0, 0.1].into(),
            threshold_idx: 0,
            energy: vec![1.0e-5, 1.0e4, 2.0e7].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    fn block(mt: i32, mt1: i32, ni: NiSubsection) -> CovarianceBlock {
        CovarianceBlock {
            mt,
            subsection_idx: 0,
            block_idx: 0,
            mat1: 0,
            mt1,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            data: CovarianceData::Ni(ni),
        }
    }

    /// `lb = 5`, `ls = 1`: the upper triangle, row by row.
    fn lb5(energies: &[f64], upper: &[f64]) -> NiSubsection {
        NiSubsection {
            lb: 5,
            ls: 1,
            ne: energies.len() as i64,
            ek: energies.to_vec(),
            fkk: upper.to_vec(),
            ..Default::default()
        }
    }

    /// `lb = 1` or `lb = 8`: one variance per interval.
    fn diagonal(lb: i64, energies: &[f64], variances: &[f64]) -> NiSubsection {
        NiSubsection {
            lb,
            ek: energies.to_vec(),
            fk: variances.to_vec(),
            ..Default::default()
        }
    }

    /// The rate the collapse would give over `[a, b]`.
    fn rate(a: f64, b: f64) -> f64 {
        BARN_TO_CM2 * flux().integrate_xs(&reaction(102), a, b)
    }

    /// Fold `blocks` for `(n,gamma)` and `(n,2n)` against the rates the
    /// collapse would give, returning the report.
    fn fold(blocks: &[CovarianceBlock], scale_rate: f64) -> Coverage {
        let (capture, n2n) = (reaction(102), reaction(16));
        let reactions: BTreeMap<i32, &Reaction> = BTreeMap::from([(102, &capture), (16, &n2n)]);
        let kinds = vec![("(n,2n)".to_string(), 16), ("(n,gamma)".to_string(), 102)];
        let full = rate(1.0e-5, 2.0e7) * scale_rate;
        let rates = BTreeMap::from([
            ("(n,2n)".to_string(), full),
            ("(n,gamma)".to_string(), full),
        ]);
        let mut coverage = Coverage::default();
        fold_nuclide(
            &flux(),
            blocks,
            &reactions,
            &kinds,
            &rates,
            "W186",
            &mut coverage,
        )
        .expect("the blocks are usable");
        coverage
    }

    fn fraction(coverage: &Coverage, kind: &str) -> f64 {
        coverage.rate_fraction_covered[&("W186".to_string(), kind.to_string())]
    }

    /// ENDF/B-VIII.1 W186 `(n,gamma)` in miniature: one block over the whole
    /// range whose first interval, the resonance range up to 10 keV, has a
    /// variance of zero. Spanning that interval states nothing about it, so
    /// the rate from there is uncovered, where it used to read as covered.
    #[test]
    fn a_zero_variance_resonance_range_reads_as_uncovered() {
        let grid = [1.0e-5, 1.0e4, 2.0e7];
        let c = fold(&[block(102, 102, lb5(&grid, &[0.0, 0.0, 0.01]))], 1.0);

        let expected = rate(1.0e4, 2.0e7) / rate(1.0e-5, 2.0e7);
        let got = fraction(&c, "(n,gamma)");
        assert!(
            (got - expected).abs() <= 1.0e-12 * expected,
            "only the rate above 10 keV carries a variance: {got} against {expected}"
        );
        assert!(got < 0.5, "most of this rate is below 10 keV: {got}");
        assert!(c.partials_above_rate.is_empty());

        // Against half the rate, as a shielded collapse would give, the
        // partials the fold weighted with sum to twice it. The zero variance
        // interval is among them, so the check must see it even though the
        // share leaves it out, and the share itself does not move.
        let c = fold(&[block(102, 102, lb5(&grid, &[0.0, 0.0, 0.01]))], 0.5);
        let ratio = c.partials_above_rate[&("W186".to_string(), "(n,gamma)".to_string())];
        assert!((ratio - 2.0).abs() < 1.0e-12, "{ratio}");
        assert_eq!(fraction(&c, "(n,gamma)").to_bits(), got.to_bits());
    }

    /// A block that is zero everywhere is still the evaluation's statement
    /// about the channel, so it reads as zero coverage rather than as no entry.
    #[test]
    fn an_all_zero_block_reads_zero_rather_than_absent() {
        let grid = [1.0e-5, 1.0e4, 2.0e7];
        let c = fold(&[block(102, 102, lb5(&grid, &[0.0, 0.0, 0.0]))], 1.0);
        assert_eq!(fraction(&c, "(n,gamma)"), 0.0);
    }

    /// A block with a variance on every interval across the whole flux range
    /// reads exactly one, where the old share, its partials over the rate,
    /// read one to rounding. So a channel whose covariance is stated
    /// everywhere does not move.
    #[test]
    fn a_block_with_variance_everywhere_reads_one() {
        let grid = [1.0e-5, 0.5, 1.0e4, 3.0e6, 2.0e7];
        let upper = [
            0.04, 0.01, 0.0, 0.0, //
            0.02, 0.005, 0.0, //
            0.03, 0.01, //
            0.05,
        ];
        let c = fold(&[block(102, 102, lb5(&grid, &upper))], 1.0);

        let partials = partial_rates(&flux(), &reaction(102), &grid, Scale::Relative);
        let before = partials.total() / rate(1.0e-5, 2.0e7);
        assert_eq!(fraction(&c, "(n,gamma)"), 1.0);
        assert!((before - 1.0).abs() < 1.0e-15, "{before}");
        assert!(
            c.partials_above_rate.is_empty(),
            "a dilute fold is consistent"
        );
    }

    /// A block stated on every interval of a grid that spans part of the
    /// range reads what the old share did, its partials over the rate, to
    /// rounding: the two differ only in how the same integral is grouped.
    #[test]
    fn a_block_with_variance_on_part_of_the_range_is_unchanged() {
        let grid = [1.0, 1.0e3, 1.0e6];
        let c = fold(&[block(102, 102, lb5(&grid, &[0.01, 0.002, 0.03]))], 1.0);

        let partials = partial_rates(&flux(), &reaction(102), &grid, Scale::Relative);
        let before = partials.total() / rate(1.0e-5, 2.0e7);
        let got = fraction(&c, "(n,gamma)");
        assert!(before < 1.0);
        assert!(
            (got - before).abs() <= 1.0e-14 * before,
            "{got} against {before}"
        );
    }

    /// A threshold above the spectrum's top edge drives no dilute rate, so
    /// there is no share to report, even though its block states a variance.
    #[test]
    fn a_channel_with_no_dilute_rate_has_no_share() {
        let above = Reaction {
            cross_section: vec![0.0, 0.0, 1.0].into(),
            energy: vec![1.0e-5, 2.5e7, 3.0e7].into(),
            ..reaction(16)
        };
        let d = Diagonal::of(&ExpandedBlock {
            row_energies: vec![2.5e7, 3.0e7],
            col_energies: vec![2.5e7, 3.0e7],
            values: vec![0.01],
            scale: Scale::Relative,
        });
        assert_eq!(stated_variance_share(&flux(), &above, &[d]), None);
    }

    /// A cross block correlates two reactions and states a variance for
    /// neither, so the channel it alone names is not covered by it.
    #[test]
    fn a_cross_block_names_a_channel_without_covering_it() {
        let grid = [1.0e-5, 1.0e4, 2.0e7];
        let cross = NiSubsection {
            lb: 6,
            er: vec![1.0e6, 2.0e7],
            ec: grid.to_vec(),
            fkl: vec![0.0, 0.001],
            ..Default::default()
        };
        let c = fold(
            &[
                block(102, 102, lb5(&grid, &[0.01, 0.0, 0.01])),
                block(16, 102, cross),
            ],
            1.0,
        );
        assert_eq!(fraction(&c, "(n,2n)"), 0.0);
        assert!((fraction(&c, "(n,gamma)") - 1.0).abs() < 1.0e-12);
    }

    /// Blocks of one subsection add, so coverage is where their summed
    /// variance is nonzero: the union of what each states, less anywhere two
    /// cancel exactly.
    #[test]
    fn blocks_on_different_grids_cover_their_union() {
        let overlapping = [
            block(102, 102, diagonal(1, &[1.0e-5, 1.0, 1.0e3], &[0.0, 0.01])),
            block(102, 102, diagonal(8, &[1.0e2, 1.0e5], &[0.02])),
        ];
        let c = fold(&overlapping, 1.0);
        let expected = rate(1.0, 1.0e5) / rate(1.0e-5, 2.0e7);
        let got = fraction(&c, "(n,gamma)");
        assert!(
            (got - expected).abs() <= 1.0e-12 * expected,
            "{got} against {expected}"
        );

        // A negative variance cancelling the first block above 1 keV. Not
        // physical, but it is what summing the blocks means. The two tape
        // values cancel bit for bit, which is the only cancellation counted.
        let cancelling = [
            block(102, 102, diagonal(1, &[1.0, 1.0e3, 1.0e5], &[0.01, 0.02])),
            block(102, 102, diagonal(8, &[1.0e3, 1.0e5], &[-0.02])),
        ];
        let c = fold(&cancelling, 1.0);
        let expected = rate(1.0, 1.0e3) / rate(1.0e-5, 2.0e7);
        let got = fraction(&c, "(n,gamma)");
        assert!(
            (got - expected).abs() <= 1.0e-12 * expected,
            "{got} against {expected}"
        );
    }

    /// A rectangular block on a reaction with itself states its variance where
    /// its row and column grids overlap, read off the matrix there.
    #[test]
    fn a_rectangular_diagonal_is_read_where_the_grids_overlap() {
        let expanded = ExpandedBlock {
            row_energies: vec![1.0, 10.0, 100.0],
            col_energies: vec![5.0, 50.0],
            values: vec![0.3, 0.7],
            scale: Scale::Relative,
        };
        let d = Diagonal::of(&expanded);
        assert_eq!(d.pieces, vec![(5.0, 10.0, 0.3), (10.0, 50.0, 0.7)]);
        assert_eq!(d.over(10.0, 50.0), 0.7);
        assert_eq!(d.over(50.0, 100.0), 0.0);
        assert_eq!(d.over(1.0, 5.0), 0.0);
    }

    /// Partials above the rate are an inconsistency between the numerator and
    /// the denominator, not a coverage of more than all of it. The excess is
    /// reported, where it used to be clamped silently, and the share, measured
    /// against the fold's own dilute integral, reads what it would anyway.
    #[test]
    fn partials_above_the_rate_are_reported_not_clamped_away() {
        let grid = [1.0e-5, 1.0e4, 2.0e7];
        // Half the rate the partials sum to, as a shielded collapse would give
        // against dilute partials.
        let c = fold(&[block(102, 102, lb5(&grid, &[0.01, 0.0, 0.01]))], 0.5);
        let key = ("W186".to_string(), "(n,gamma)".to_string());
        assert_eq!(fraction(&c, "(n,gamma)"), 1.0);
        let ratio = c.partials_above_rate[&key];
        assert!((ratio - 2.0).abs() < 1.0e-12, "{ratio}");
        assert!(
            c.has_gaps(),
            "an overstated sigma is something to know about"
        );

        // Across spectra the larger excess is the one kept.
        let mut merged = Coverage::default();
        merged.absorb(c.clone());
        merged.absorb(Coverage {
            partials_above_rate: BTreeMap::from([(key.clone(), 1.5)]),
            ..Default::default()
        });
        assert_eq!(merged.partials_above_rate[&key], ratio);
    }
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
