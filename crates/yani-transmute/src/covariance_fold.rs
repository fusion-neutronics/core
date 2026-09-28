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
//! A channel whose covariance the tape derives from other reactions (an NC
//! block with LTY=0) is folded the same way, through the reactions it names:
//! its rate is a sum of theirs over the block's energy range, so `r_i` becomes
//! a sum of their partial rates, each against their own blocks. See [`Term`].
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
//! A self-shielded collapse takes each group average under the nuclide's flux
//! shape inside the lump instead, `σ_g = ∫_g σ s dE / ∫_g s dE`, so there `ψ`
//! is `φ_g · s(E) / ∫_g s dE` and the partials are taken under that same shape
//! with the same walk. Otherwise the fold would weight the covariance with
//! dilute partials and divide by the shielded rate, which overstates the
//! relative sigma by about one over the shielding factor wherever a covariance
//! interval carries both: ENDF/B-VIII.1 Au197 `(n,gamma)` in a 0.1 mm foil
//! under `1/E` on VITAMIN-J-175 read 4.61% where its shielded rate's sigma is
//! 1.70%. The shape is the nominal one, as in the collapse: a perturbed cross
//! section does not change the flux dip it sits in.
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

use endf::mf::covariance::NiSubsection;
use yamc_materials::Material;
use yamc_nuclide::covariance::expand::{expand_ni, ExpandedBlock, Scale, Unsupported};
use yamc_nuclide::covariance::{CovarianceBlock, CovarianceData};
use yamc_nuclide::reaction::Reaction;
use yani::reactions::reaction_type_to_mt;
use yani::{ChainNuclide, ReactionRates};

use crate::multigroup::{group_averaged_xs, walk_group, CollapseShapes, GroupTerms};
use crate::self_shielding::{FluxShape, Shielding};

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
    /// Per nuclide, its blocks correlating one of its reactions with a
    /// reaction of ANOTHER evaluation, not consumed.
    ///
    /// Only genuine ones: a `mat1` naming this evaluation's own MAT is this
    /// evaluation (ENDF-102 33.3.1) and is folded. Only relevant ones: a block
    /// is counted only on a reaction the fold reaches (a channel, or one a
    /// channel is derived from), since elsewhere there is no rate for it to
    /// be the uncertainty of. Per nuclide rather than a sum,
    /// so a run with several spectra counts each block once.
    pub skipped_cross_material: BTreeMap<String, usize>,
    /// Per nuclide, its blocks correlating one of its reactions with a
    /// quantity of this evaluation that is not a cross section (`xmf1` other
    /// than 0 or 3, or a final state in `xlfs1`), not consumed. Counted like
    /// [`Coverage::skipped_cross_material`]. No library yani reads has one.
    pub skipped_other_file: BTreeMap<String, usize>,
    /// Per (nuclide, kind, kind), where a cross-reaction pair is stored in both
    /// orientations and the two copies are not each other's transpose, the
    /// largest difference between them relative to the largest entry.
    ///
    /// The kinds are ordered as their MTs, and the copy in the lower MT's
    /// section is the one folded. A reaction the fold reaches only through an
    /// NC derivation, such as O16 MT 600, is named `MT600`. JEFF-4.0 Be9
    /// stores 130 of its pairs both ways; folding both would count each
    /// covariance twice.
    pub mirrored_disagree: BTreeMap<(String, String, String), f64>,
    /// Per nuclide, its NC blocks (a covariance derived from other reactions)
    /// that could not be derived, not consumed.
    ///
    /// An LTY=0 block is derived (see [`Term`]) unless it names a reaction the
    /// evaluation has no cross section for, or one whose own derivation leads
    /// back to it, or it sits in a cross-reaction subsection, which ENDF-102
    /// 33.3.2 a.3 does not allow, or its list of reactions is empty or does
    /// not match its coefficients, or its own `[E1, E2]` is empty. LTY 1 to 4
    /// derive from ratios to another evaluation's standard and are not
    /// derived; no library yani reads has one. Counted like
    /// [`Coverage::skipped_cross_material`]: only blocks on a reaction the
    /// fold reaches over part of their range, and once per nuclide. A block
    /// met circularly on one path and derived on another was consumed and is
    /// not counted.
    pub skipped_nc: BTreeMap<String, usize>,
    /// Blocks whose `lb` layout is not implemented, counted per `lb`.
    ///
    /// This and the block counter below add over a run's spectra, so a run
    /// with several spectra counts a block once per spectrum.
    pub unsupported_layouts: BTreeMap<i64, usize>,
    /// Blocks whose arrays did not match their own declared sizes.
    pub malformed: usize,
    /// Per (nuclide, kind), the share of the fold's own rate that comes from
    /// energies where the evaluation states a nonzero variance for that
    /// reaction.
    ///
    /// The fold's own rate is the cross section against the flux the partial
    /// rates are weighted with: dilute on a dilute run, and under the
    /// nuclide's shielded flux shape on a self-shielded one, where it is the
    /// shielded rate the run used. Exactly: that rate integrated over the
    /// energies where the diagonal of the reaction's own covariance, summed
    /// over every self-covariance block the fold consumed, is nonzero, divided
    /// by that rate over the whole flux range. On a derived channel the
    /// blocks are also those of the reactions it is derived from, each over
    /// the NC block's energy range only, and the energy counts where any of
    /// them states a variance. Where the cross sections do not satisfy the
    /// derivation `σ_MT = Σ c_i σ_MTi` past rounding and the named reactions
    /// add up to less, the rate they miss, `c (σ_MT - Σ c_i σ_MTi)` for a
    /// derivation reached with coefficient `c`, is taken off each energy it
    /// falls in, down to none of it, and [`Coverage::partials_below_rate`]
    /// lists the channel as well: ENDF/B-VIII.1 O16 `(n,d)` above 20 MeV,
    /// where MT 660 to 669 carry about 7% of the rate at 25 MeV and no
    /// covariance.
    /// Rate from an interval a grid
    /// spans with a variance of zero counts as uncovered, the same as rate
    /// from outside every grid, because neither carries a stated uncertainty.
    /// Below one, part of the rate enters the relative covariance's
    /// denominator and not its numerator; how much that lowers the relative
    /// sigma also depends on how the stated variance is spread over the
    /// covered part, so the share is not itself a dilution factor.
    ///
    /// The numerator adds, in the same order, the denominator's terms or
    /// smaller ones that are not negative, so the share lies in [0, 1], and it does not
    /// depend on how the rate the covariance is divided by was computed. On a
    /// tallied rate, whose within-bin weighting the fold does not have, it is
    /// the share of the dilute rate over the tally spectrum and not of the
    /// tallied one, and the dilution the fold then applies differs from it;
    /// where the partials add up to more than that rate,
    /// [`Coverage::partials_above_rate`] lists it. On a dilute collapse under
    /// the flat-within-group weight, and on a shielded one, the denominator is
    /// the collapsed rate to rounding. Under the `1/E` weight it is only where
    /// no covariance edge cuts a group, for the reason given on
    /// [`Coverage::partials_above_rate`].
    ///
    /// Every channel a consumed block names has an entry, unless its rate over
    /// the flux range is zero (a threshold above the spectrum's top edge),
    /// which has no share to report. So a channel whose blocks state no
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
    /// sigma is overstated. A derived channel's partials are the named
    /// reactions', so for it the partials of its own reaction's blocks are
    /// compared as above and, for every NC block it was expanded through,
    /// the named reactions' rate over the block's range, `Σ c_i R_MTi`, with
    /// the rate of the reaction it derives over the same range; a ratio past
    /// rounding either way lands in this map or the one below, the larger
    /// excess or the larger shortfall kept. The partials are taken under the
    /// within-group flux the collapse used, flat or shielded, so a rate
    /// computed some other way, a tallied one among them, can land here.
    /// Reported rather than clamped away, since the clamp is what used to
    /// hide it.
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
    /// checked. The check from below is
    /// [`Coverage::partials_below_rate`], and it reaches only some blocks, so
    /// absence from both maps is not a statement that every block is
    /// consistent.
    pub partials_above_rate: BTreeMap<(String, String), f64>,
    /// Per (nuclide, kind), where a relative block's grid spans the whole
    /// flux range and its partial rates add up to less than the rate the
    /// covariance was divided by, their ratio to it.
    ///
    /// Such a grid leaves no energy outside it, so its partials are the whole
    /// rate integrated interval by interval and must equal it; the smallest
    /// sum over the channel's spanning blocks is the one compared. A shortfall
    /// past rounding is the same inconsistency as an excess, the other way:
    /// the channel's relative sigma is understated. The `1/E` within-group
    /// weight gives one for a reaction falling with energy, such as a `1/v`
    /// capture, whenever a covariance edge cuts a group, for the reason given
    /// on [`Coverage::partials_above_rate`].
    ///
    /// A grid that stops short of either end of the flux range cannot be
    /// checked from below: rate from outside it is rate with no stated
    /// uncertainty, which legitimately leaves its partials short of the rate
    /// (see [`Coverage::rate_fraction_covered`]). Absolute blocks are not
    /// checked, as above. A derived channel whose named reactions add up to
    /// less than the reaction they derive is listed here too, with the
    /// smallest ratio over its NC blocks (see
    /// [`Coverage::partials_above_rate`]): ENDF/B-VIII.1 O16 `(n,d)` above
    /// 20 MeV, where MT 104 holds 660 to 669 and its NC block names 650 to
    /// 659.
    pub partials_below_rate: BTreeMap<(String, String), f64>,
    /// The production this spectrum drove, each channel's weighted by its
    /// share in [`Coverage::rate_fraction_covered`], and the production it
    /// drove in total. Both are per barn-cm per second, and both are weighted
    /// by the parent's own density, so a channel on a trace isotope counts for
    /// what it actually made.
    ///
    /// Each channel's production is the rate this run used, and its share is
    /// of the fold's own rate. On a dilute or a self-shielded collapse the two
    /// are one rate, and the covered sum is exactly the production from
    /// energies where a covariance states a nonzero variance. On a tallied
    /// rate the share is of the dilute rate over the tally spectrum, so the
    /// covered sum is that production only if the tallied rate kept the
    /// dilute rate's distribution in energy, which self-shielding in the
    /// transport does not: it depresses the resonance range, where capture
    /// blocks often state zero. The covered share of a tallied production is
    /// not computed.
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
    /// The production-weighted mean of the per-channel shares, weighted by the
    /// rate this run used and by parent density.
    ///
    /// On a dilute or a self-shielded spectrum run that is the share of the
    /// production driven from energies where a covariance states a nonzero
    /// variance. On a tallied run it is not, for the reason given on
    /// [`Coverage::covered_production`], and the run reports no total there
    /// rather than this figure (`Info::rate_fraction_covered_total`).
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
    ///
    /// In [0, 1] without a clamp: each channel adds `production * share` to
    /// one sum and `production` to the other, in the same order, with the
    /// share at most one. Rounding is monotone, so the covered sum cannot pass
    /// the total. A ratio outside [0, 1] is a bug upstream (a negative rate,
    /// or a share above one), and clamping it would hide that, so it panics
    /// in every build rather than return a share that is not one.
    pub fn rate_fraction_total(&self) -> Option<f64> {
        (self.total_production > 0.0).then(|| {
            let fraction = self.covered_production / self.total_production;
            assert!(
                (0.0..=1.0).contains(&fraction),
                "covered production {} against a total of {} is not a share",
                self.covered_production,
                self.total_production
            );
            fraction
        })
    }

    /// Fold one nuclide's report into this one.
    ///
    /// Every field is order-free: the sets union, the counters sum, the
    /// per-nuclide counts and mismatches take the larger,
    /// `rate_fraction_covered` is keyed by (nuclide, kind) and takes the
    /// smaller claim, `partials_above_rate` takes the larger excess and
    /// `partials_below_rate` the larger shortfall, the smaller ratio. That
    /// is what lets the fold below run per nuclide in parallel and merge
    /// afterwards (issue #576, finding 5c).
    pub fn absorb(&mut self, other: Coverage) {
        self.covered.extend(other.covered);
        self.without_data.extend(other.without_data);
        // The same nuclide gives the same counts and the same mismatch on
        // every spectrum, since they depend only on its data and its chain,
        // so a repeat is taken once rather than added.
        for (nuclide, n) in other.skipped_cross_material {
            let e = self.skipped_cross_material.entry(nuclide).or_insert(0);
            *e = (*e).max(n);
        }
        for (nuclide, n) in other.skipped_other_file {
            let e = self.skipped_other_file.entry(nuclide).or_insert(0);
            *e = (*e).max(n);
        }
        for (key, d) in other.mirrored_disagree {
            let e = self.mirrored_disagree.entry(key).or_insert(0.0);
            *e = e.max(d);
        }
        for (nuclide, n) in other.skipped_nc {
            let e = self.skipped_nc.entry(nuclide).or_insert(0);
            *e = (*e).max(n);
        }
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
        for (key, ratio) in other.partials_below_rate {
            self.partials_below_rate
                .entry(key)
                .and_modify(|r| *r = r.min(ratio))
                .or_insert(ratio);
        }
        self.covered_production += other.covered_production;
        self.total_production += other.total_production;
    }

    /// Whether anything at all was skipped, missing or inconsistent.
    pub fn has_gaps(&self) -> bool {
        !self.without_data.is_empty()
            || !self.skipped_cross_material.is_empty()
            || !self.skipped_other_file.is_empty()
            || !self.mirrored_disagree.is_empty()
            || !self.skipped_nc.is_empty()
            || !self.unsupported_layouts.is_empty()
            || self.malformed > 0
            || !self.partials_above_rate.is_empty()
            || !self.partials_below_rate.is_empty()
    }
}

/// The flux density the collapse weighted one nuclide's rates with, as the
/// group structure, the fluxes and, on a shielded run, that nuclide's shape.
///
/// Dilute, `ψ(E) = φ_g / ΔE_g`. Shielded, `ψ(E) = φ_g · s(E) / ∫_g s dE` with
/// `s` the nuclide's flux shape inside the lump, which is what the collapse's
/// `σ_g = ∫_g σ s dE / ∫_g s dE` times `φ_g` integrates against.
///
/// Kept as the vectors rather than as a function, because every integral
/// below is over the overlap of an arbitrary interval with the groups, and the
/// overlap is what the trapezoid integrator needs.
struct FluxDensity<'a> {
    boundaries: &'a [f64],
    flux: &'a [f64],
    shape: Option<&'a FluxShape>,
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

    /// `∫ σ(E) ψ(E) dE` over each interval of the ascending `grid`, in barn
    /// n/cm^2/s.
    fn xs_over(&self, reaction: &Reaction, grid: &[f64]) -> Vec<f64> {
        match self.shape {
            None => grid
                .windows(2)
                .map(|w| self.integrate_xs(reaction, w[0], w[1]))
                .collect(),
            Some(shape) => self.apportion(
                reaction,
                shape,
                grid,
                |whole, phi| whole.shielded() * phi,
                GroupTerms::shielded_integral,
            ),
        }
    }

    /// `∫ ψ(E) dE` over each interval of the ascending `grid`, in n/cm^2/s.
    ///
    /// `reaction` matters only on a shielded run, where the shape is integrated
    /// on the same points the collapse integrated it on for that reaction.
    fn flux_over(&self, reaction: &Reaction, grid: &[f64]) -> Vec<f64> {
        match self.shape {
            None => grid
                .windows(2)
                .map(|w| self.integrate_flux(w[0], w[1]))
                .collect(),
            Some(shape) => self.apportion(
                reaction,
                shape,
                grid,
                |_, phi| phi,
                GroupTerms::shielded_weight,
            ),
        }
    }

    /// Split each group's term of the shielded collapse across the intervals
    /// of `grid` it overlaps.
    ///
    /// `whole` is the group's term as the collapse has it, from its walk of
    /// the whole group and its flux, and `weight` the integral a part of the
    /// group carries of it. A group that no edge of `grid` cuts goes whole to
    /// the interval holding it, so there the partial IS the collapse's term,
    /// bit for bit. A cut group is split in proportion to its parts' own
    /// shielded integrals, normalized by their sum rather than by the whole
    /// group's: inserting an edge adds a point to the trapezoid, and the shape
    /// there is interpolated in log energy on its own grid rather than along
    /// the segment the edge splits, so the parts' integrals do not add up to
    /// the whole group's exactly. Normalized by their own sum, the parts
    /// partition the collapse's term, and the partials over a grid spanning
    /// the flux range sum to the shielded rate to rounding, which is what the
    /// relative covariance divides them by.
    fn apportion(
        &self,
        reaction: &Reaction,
        shape: &FluxShape,
        grid: &[f64],
        whole: impl Fn(&GroupTerms, f64) -> f64,
        weight: impl Fn(&GroupTerms) -> f64,
    ) -> Vec<f64> {
        let n = grid.len().saturating_sub(1);
        let mut out = vec![0.0; n];
        if n == 0 {
            return out;
        }
        let mut cuts: Vec<f64> = Vec::new();
        let mut parts: Vec<f64> = Vec::new();
        for g in self.overlapping(grid[0], grid[n]) {
            let (glo, ghi) = (self.boundaries[g], self.boundaries[g + 1]);
            if ghi <= glo {
                continue;
            }
            let term = whole(
                &walk_group(reaction, glo, ghi, Some(shape), None),
                self.flux[g],
            );
            if term == 0.0 {
                continue;
            }
            let first = grid.partition_point(|&e| e <= glo);
            let last = grid.partition_point(|&e| e < ghi);
            if first >= last {
                if let Some(k) = interval_holding(grid, glo, ghi) {
                    out[k] += term;
                }
                continue;
            }
            cuts.clear();
            cuts.push(glo);
            cuts.extend_from_slice(&grid[first..last]);
            cuts.push(ghi);
            parts.clear();
            parts.extend(
                cuts.windows(2)
                    .map(|w| weight(&walk_group(reaction, w[0], w[1], Some(shape), None))),
            );
            let sum: f64 = parts.iter().sum();
            if sum <= 0.0 {
                continue;
            }
            for (w, part) in cuts.windows(2).zip(&parts) {
                if let Some(k) = interval_holding(grid, w[0], w[1]) {
                    out[k] += term * part / sum;
                }
            }
        }
        out
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
    let integrals = match scale {
        // A relative covariance multiplies the rate, so the weight is the
        // partial RATE.
        Scale::Relative => flux.xs_over(reaction, grid),
        // An absolute covariance is already in barns squared, so the weight is
        // the partial FLUX and the cross section must not appear twice.
        Scale::Absolute => flux.flux_over(reaction, grid),
    };
    Partials {
        per_interval: integrals.into_iter().map(|v| BARN_TO_CM2 * v).collect(),
    }
}

/// The partial rates of `reaction` over the intervals of `grid`, counting only
/// the part of each interval inside `range`.
///
/// A range that holds the whole grid is [`partial_rates`] itself, bit for bit,
/// which is every term but a derived one. Otherwise the grid is cut at the
/// range's ends, the partials are taken on the cut grid, and each interval
/// keeps the sum of its pieces inside the range. Taken on the cut grid rather
/// than scaled, because the cross section is not flat across an interval.
fn partial_rates_within(
    flux: &FluxDensity,
    reaction: &Reaction,
    grid: &[f64],
    scale: Scale,
    range: (f64, f64),
) -> Partials {
    let (Some(&first), Some(&last)) = (grid.first(), grid.last()) else {
        return Partials {
            per_interval: Vec::new(),
        };
    };
    if range.0 <= first && range.1 >= last {
        return partial_rates(flux, reaction, grid, scale);
    }
    let mut cut: Vec<f64> = grid
        .iter()
        .copied()
        .chain(
            [range.0, range.1]
                .into_iter()
                .filter(|e| *e > first && *e < last),
        )
        .collect();
    cut.sort_by(f64::total_cmp);
    cut.dedup();
    let pieces = partial_rates(flux, reaction, &cut, scale);
    let mut per_interval = vec![0.0; grid.len() - 1];
    for (w, part) in cut.windows(2).zip(pieces.per_interval) {
        if w[0] >= range.0 && w[1] <= range.1 {
            if let Some(k) = interval_holding(grid, w[0], w[1]) {
                per_interval[k] += part;
            }
        }
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

/// How far from one a channel's summed partial rates over its rate may sit
/// and still be rounding, above it or, for a grid spanning the whole flux
/// range, below it.
///
/// On a dilute collapse under the flat-within-group weight the two add the
/// same terms grouped differently, and on CCFE-709 they agree to a few parts
/// in 1e15. On a shielded one each group's term is split across the intervals
/// and the parts add back to it, so the same holds. Under the `1/E` weight
/// they need not, see
/// `Coverage::partials_above_rate`. This sits six orders of magnitude above
/// that, and an excess below it would move a relative sigma by less than a
/// part in a billion.
///
/// It separates rounding from an inconsistency, not a large inconsistency
/// from a small one: any excess past rounding means the covariance's
/// numerator and denominator were computed two different ways, and every
/// such channel is listed with its ratio for the reader to weigh.
const PARTIALS_ROUNDING: f64 = 1.0e-9;

/// How far the rate of the reactions an NC block names may sit from the rate
/// of the one it derives and still be rounding, relative to the larger of
/// that rate and `Σ |c_i| R_MTi` (see [`derivation_mismatch`]).
///
/// Wider than [`PARTIALS_ROUNDING`], because the two sides are different
/// tape values rather than one integral grouped two ways: an ENDF float
/// carries six or seven significant digits, so a redundant cross section
/// written as the rounded sum of its parts can sit up to about 5e-6 from the
/// sum of the rounded parts. This is twice that, as [`MIRROR_ROUNDING`] is.
/// ENDF/B-VIII.1, FENDL-3.2d, JEFF-4.0 and TENDL-2017 O16 MT 103 agree with
/// 600 to 603 to 4e-8 at 14 MeV, while 650 to 659 fall 5% short of MT 104
/// from 20 to 150 MeV.
const DERIVATION_ROUNDING: f64 = 1.0e-5;

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

    /// The pieces inside `range`, cut at its ends: what a derived term states
    /// for the channel it derives, since it applies only over its NC block's
    /// energy range.
    fn within(&self, range: (f64, f64)) -> Self {
        Self {
            scale: self.scale,
            pieces: self
                .pieces
                .iter()
                .map(|&(lo, hi, v)| (lo.max(range.0), hi.min(range.1), v))
                .filter(|&(lo, hi, _)| lo < hi)
                .collect(),
        }
    }
}

/// The share of `reaction`'s rate over the flux range, as the fold integrates
/// it, that comes from energies where its own blocks state a nonzero variance,
/// or `None` when that rate is zero.
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
/// Where a derivation in `short` names reactions that add up to less than the
/// one it derives, the rate they miss is taken off each cell it falls in, down
/// to none of the cell: it is rate the covariance states nothing for.
///
/// The covered sum adds, in the same order, the total's terms or smaller ones
/// that are not negative, and rounded addition of such terms is monotone, so
/// the share cannot exceed one. A block that states a variance on every
/// interval across the whole flux range therefore reads exactly one.
fn stated_variance_share(
    flux: &FluxDensity,
    reaction: &Reaction,
    diagonals: &[Diagonal],
    reactions: &BTreeMap<i32, &Reaction>,
    short: &[&Derivation],
) -> Option<f64> {
    let (&lo, &hi) = (flux.boundaries.first()?, flux.boundaries.last()?);
    let mut edges: Vec<f64> = diagonals
        .iter()
        .flat_map(|d| d.pieces.iter().flat_map(|&(lo, hi, _)| [lo, hi]))
        .chain(short.iter().flat_map(|d| [d.range.0, d.range.1]))
        .chain([lo, hi])
        .collect();
    edges.sort_by(f64::total_cmp);
    edges.dedup();
    // Per cell, the channel's rate no named reaction carries: each short
    // derivation's `coefficient (R_MT - Σ c_i R_MTi)` over its range. The
    // gaps add, since a nested derivation's is the shortfall inside a
    // reaction its parent names, so their sum is the channel's rate less
    // that of the reactions the covariance is actually stated for.
    let mut missing = vec![0.0; edges.len() - 1];
    for d in short {
        let rate = |mt: i32| flux.xs_over(reactions[&mt], &edges);
        let mut gap = rate(d.mt);
        for &(c, mt) in &d.named {
            for (g, r) in gap.iter_mut().zip(rate(mt)) {
                *g -= c * r;
            }
        }
        for (k, w) in edges.windows(2).enumerate() {
            if d.range.0 <= w[0] && w[1] <= d.range.1 {
                missing[k] += d.coefficient * gap[k];
            }
        }
    }
    let (mut covered, mut total) = (0.0, 0.0);
    for (k, (w, rate)) in edges
        .windows(2)
        .zip(flux.xs_over(reaction, &edges))
        .enumerate()
    {
        let (a, b) = (w[0], w[1]);
        total += rate;
        let (mut relative, mut absolute) = (0.0, 0.0);
        for d in diagonals {
            match d.scale {
                Scale::Relative => relative += d.over(a, b),
                Scale::Absolute => absolute += d.over(a, b),
            }
        }
        if relative != 0.0 || absolute != 0.0 {
            // Clamped to the cell's rate, so one cell where the named
            // reactions exceed it cannot make up for another's shortfall.
            covered += rate - missing[k].max(0.0).min(rate);
        }
    }
    (total != 0.0).then(|| covered / total)
}

/// How far apart the two stored orientations of one pair may be, relative to
/// the pair's largest entry, and still be the same numbers.
///
/// An ENDF float carries six or seven significant digits, so a copy written
/// by a code that rounds each orientation on its own can differ from the
/// transpose of the other by up to about 5e-6 of an entry, and so of the
/// largest. This is twice that. The 130 pairs JEFF-4.0 Be9 stores both ways
/// agree exactly.
const MIRROR_ROUNDING: f64 = 1.0e-5;

/// The cross-reaction pairs, as `(lower MT, higher MT)`, that this
/// evaluation's explicit blocks give in both orientations, between two
/// reactions the fold reaches.
///
/// ENDF-102 puts a pair in the section of its lower MT, with MT1 above MT, but
/// nothing forbids the transpose in the other section as well, and JEFF-4.0
/// Be9 has 130 such pairs. The fold fills both halves from one block, so
/// keeping both copies would count the pair twice.
fn mirrored_pairs(blocks: &[CovarianceBlock], reached: &BTreeSet<i32>) -> BTreeSet<(i32, i32)> {
    let oriented: BTreeSet<(i32, i32)> = blocks
        .iter()
        .filter(|b| b.is_same_evaluation() && matches!(b.data, CovarianceData::Ni(_)))
        .map(|b| (b.mt, b.partner_mt()))
        .filter(|(a, b)| a != b && reached.contains(a) && reached.contains(b))
        .collect();
    oriented
        .iter()
        .filter(|(a, b)| a < b && oriented.contains(&(*b, *a)))
        .copied()
        .collect()
}

/// This evaluation's explicit (`row`, `col`) blocks.
fn orientation(
    blocks: &[CovarianceBlock],
    row: i32,
    col: i32,
) -> impl Iterator<Item = &NiSubsection> {
    blocks.iter().filter_map(move |blk| match &blk.data {
        CovarianceData::Ni(ni)
            if blk.is_same_evaluation() && blk.mt == row && blk.partner_mt() == col =>
        {
            Some(ni)
        }
        _ => None,
    })
}

/// Whether every one of this evaluation's explicit (`row`, `col`) blocks
/// expands, so that copy of a mirrored pair can stand for the pair.
fn orientation_expands(blocks: &[CovarianceBlock], row: i32, col: i32) -> bool {
    orientation(blocks, row, col).all(|ni| expand_ni(ni).is_ok())
}

/// Count the (`row`, `col`) blocks that do not expand into the coverage, as
/// the fold does for the blocks it reaches.
fn count_unexpandable(blocks: &[CovarianceBlock], row: i32, col: i32, coverage: &mut Coverage) {
    for ni in orientation(blocks, row, col) {
        match expand_ni(ni) {
            Ok(_) => {}
            Err(Unsupported::Layout(lb)) => {
                *coverage.unsupported_layouts.entry(lb).or_insert(0) += 1;
            }
            Err(Unsupported::Malformed) => coverage.malformed += 1,
        }
    }
}

/// The largest difference between the (`a`, `b`) blocks and the transpose of
/// the (`b`, `a`) ones, relative to the largest entry of either, with `a < b`,
/// the worse of the relative and the absolute blocks.
///
/// Each orientation's blocks are summed, since blocks of one pair add, and the
/// sums are compared cell by cell on the union of every grid either uses, one
/// scale at a time. Called only when every block of both orientations
/// expands ([`orientation_expands`]).
fn mirror_mismatch(blocks: &[CovarianceBlock], a: i32, b: i32) -> f64 {
    let expanded = |row: i32, col: i32| -> Vec<ExpandedBlock> {
        orientation(blocks, row, col)
            .filter_map(|ni| expand_ni(ni).ok())
            .filter(|e| !e.is_empty())
            .collect()
    };
    let (forward, backward) = (expanded(a, b), expanded(b, a));
    let edges = |grids: &mut dyn Iterator<Item = &Vec<f64>>| -> Vec<f64> {
        let mut e: Vec<f64> = grids.flatten().copied().collect();
        e.sort_by(f64::total_cmp);
        e.dedup();
        e
    };
    // Energies of `a` are the forward rows and the backward columns.
    let a_edges = edges(
        &mut forward
            .iter()
            .map(|e| &e.row_energies)
            .chain(backward.iter().map(|e| &e.col_energies)),
    );
    let b_edges = edges(
        &mut forward
            .iter()
            .map(|e| &e.col_energies)
            .chain(backward.iter().map(|e| &e.row_energies)),
    );
    let at = |e: &ExpandedBlock, x: &[f64], y: &[f64]| -> f64 {
        match (
            interval_holding(&e.row_energies, x[0], x[1]),
            interval_holding(&e.col_energies, y[0], y[1]),
        ) {
            (Some(i), Some(j)) => e.get(i, j),
            _ => 0.0,
        }
    };
    let mut mismatch = 0.0_f64;
    for scale in [Scale::Relative, Scale::Absolute] {
        let (mut largest, mut worst) = (0.0_f64, 0.0_f64);
        for x in a_edges.windows(2) {
            for y in b_edges.windows(2) {
                let f: f64 = forward
                    .iter()
                    .filter(|e| e.scale == scale)
                    .map(|e| at(e, x, y))
                    .sum();
                let g: f64 = backward
                    .iter()
                    .filter(|e| e.scale == scale)
                    .map(|e| at(e, y, x))
                    .sum();
                largest = largest.max(f.abs()).max(g.abs());
                worst = worst.max((f - g).abs());
            }
        }
        if largest > 0.0 {
            mismatch = mismatch.max(worst / largest);
        }
    }
    mismatch
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

/// The whole energy axis, the range of every term that is not derived.
const EVERYWHERE: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);

/// One term of a channel's cross section as the covariance sees it:
/// `coefficient` times reaction `mt`'s cross section over `range`.
///
/// A channel is its own reaction everywhere, and each NC block with LTY=0 on
/// it adds a term per reaction it names: ENDF-102 33.2.2.1 states that over
/// `[E1, E2]` the cross section is `σ_MT = Σ_i c_i σ_MTi`, and that the
/// covariance is to be derived as if that held. A perturbation of the channel's
/// rate is then the sum of its terms' perturbations, so
///
/// ```text
/// Cov(R_a, R_b) = Σ_{t ∈ a, u ∈ b} c_t c_u Σ_{blocks (mt_t, mt_u)} r_tᵀ C r_u
/// ```
///
/// with `r_t` reaction `mt_t`'s partial rates restricted to `range_t`. That is
/// the sandwich ENDF-102 33.3.2 a.3 leaves to the processing code, exactly: no
/// covariance is invented, the numbers are the named reactions' own NI blocks
/// and the cross blocks between them. The own term stays alongside the derived
/// ones because the blocks of a subsection add (33.2.1); 33.3.3 item 3 has its
/// NI blocks state zero over `[E1, E2]`, so the two do not overlap on a tape
/// that follows it. ENDF/B-VIII.1 O16 `(n,p)` is `600 + 601 + 602 + 603` and
/// states its covariance no other way.
///
/// A named reaction may be derived in its turn, as ENDF/B-VIII.1 O16 MT 4 is
/// from MT 1, 103, 104, 105 and 107, so terms expand until every one is a
/// reaction whose blocks are explicit, each restricted to the overlap of the
/// ranges it was reached through.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Term {
    /// Index of the channel in `kinds`.
    channel: usize,
    coefficient: f64,
    mt: i32,
    range: (f64, f64),
}

impl Term {
    /// Whether this is the channel's own reaction everywhere, rather than a
    /// term an NC block derives.
    fn is_own(&self, kinds: &[(String, i32)]) -> bool {
        self.mt == kinds[self.channel].1 && self.range == EVERYWHERE && self.coefficient == 1.0
    }
}

/// An NC block's identity within the evaluation, so one reached from several
/// channels counts once.
fn block_key(block: &CovarianceBlock) -> (i32, i32, i32) {
    (block.mt, block.subsection_idx, block.block_idx)
}

/// The reactions an LTY=0 block names, as `(c_i, MT_i)`, or `None` when it
/// cannot be derived: an LTY other than 0, a cross-reaction subsection, an
/// empty or malformed list, an `XMTI` that is not an MT, a reaction with no
/// cross section here, or one on the path `through` that led here, the
/// block's own MT included, which would make the derivation circular.
fn derivation(
    block: &CovarianceBlock,
    reactions: &BTreeMap<i32, &Reaction>,
    through: &[i32],
) -> Option<Vec<(f64, i32)>> {
    let CovarianceData::Nc(nc) = &block.data else {
        return None;
    };
    if nc.lty != 0
        || block.partner_mt() != block.mt
        || nc.ci.is_empty()
        || nc.ci.len() != nc.xmti.len()
    {
        return None;
    }
    nc.ci
        .iter()
        .zip(&nc.xmti)
        .map(|(&c, &x)| {
            let mt = x as i32;
            let named = f64::from(mt) == x
                && reactions.contains_key(&mt)
                && !through.contains(&mt)
                && mt != block.mt;
            named.then_some((c, mt))
        })
        .collect()
}

/// One NC block as the fold applied it: over `range`, reaction `mt` of
/// `channel` is `Σ c_i σ_MTi` for the `(c_i, MT_i)` in `named`, and enters the
/// channel scaled by `coefficient`, the product of the coefficients on the
/// path that reached it.
#[derive(Debug, Clone, PartialEq)]
struct Derivation {
    channel: usize,
    mt: i32,
    coefficient: f64,
    range: (f64, f64),
    named: Vec<(f64, i32)>,
}

/// Every channel's terms, and what the NC blocks on the way did.
#[derive(Debug, Default)]
struct Expansion {
    terms: Vec<Term>,
    /// Every derivation a term was expanded through, for the check that the
    /// reactions it names add up to the one it derives.
    derivations: Vec<Derivation>,
    /// NC blocks derived on at least one path.
    derived: BTreeSet<(i32, i32, i32)>,
    /// NC blocks that could not be derived on some path.
    underived: BTreeSet<(i32, i32, i32)>,
}

impl Expansion {
    /// The NC blocks no path derived. A block circular on one path and
    /// derived on another was consumed, so it is not among them.
    fn skipped(&self) -> usize {
        self.underived.difference(&self.derived).count()
    }
}

/// Expand reaction `mt`, scaled by `coefficient` over `range`, into `out`.
///
/// `through` is the path of MTs that led here.
#[allow(clippy::too_many_arguments)]
fn expand_terms(
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
    channel: usize,
    mt: i32,
    coefficient: f64,
    range: (f64, f64),
    through: &mut Vec<i32>,
    out: &mut Expansion,
) {
    out.terms.push(Term {
        channel,
        coefficient,
        mt,
        range,
    });
    through.push(mt);
    for block in blocks {
        let CovarianceData::Nc(nc) = &block.data else {
            continue;
        };
        // Another evaluation's or another file's NC block is counted with
        // those, not here.
        if block.mt != mt || !block.is_same_evaluation() {
            continue;
        }
        // A block whose own range is empty derives nothing anywhere. One
        // whose range misses the range this term was reached over applies
        // elsewhere, and is judged on the paths that reach it there.
        if nc.e1 >= nc.e2 {
            out.underived.insert(block_key(block));
            continue;
        }
        let within = (range.0.max(nc.e1), range.1.min(nc.e2));
        if within.0 >= within.1 {
            continue;
        }
        let Some(named) = derivation(block, reactions, through) else {
            out.underived.insert(block_key(block));
            continue;
        };
        out.derived.insert(block_key(block));
        out.derivations.push(Derivation {
            channel,
            mt,
            coefficient,
            range: within,
            named: named.clone(),
        });
        for (c, named_mt) in named {
            expand_terms(
                blocks,
                reactions,
                channel,
                named_mt,
                coefficient * c,
                within,
                through,
                out,
            );
        }
    }
    through.pop();
}

/// Every channel's terms, and the NC blocks on the way.
fn channel_terms(
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
    kinds: &[(String, i32)],
) -> Expansion {
    let mut out = Expansion::default();
    for (channel, (_, mt)) in kinds.iter().enumerate() {
        // A channel with no cross section has no rate for a covariance to be
        // the uncertainty of.
        if reactions.contains_key(mt) {
            expand_terms(
                blocks,
                reactions,
                channel,
                *mt,
                1.0,
                EVERYWHERE,
                &mut Vec::new(),
                &mut out,
            );
        }
    }
    out
}

/// Every MT the fold can reach on `chain_nuclide`: its channels' own, and
/// those the LTY=0 NC blocks on them name, transitively.
///
/// An activation load reads only the MTs it is asked for, and the chain names
/// none of the partials a derivation weights with, such as O16 MT 600 to 603,
/// so without them [`derivation`] finds no cross section and the channel folds
/// to nothing. Read off the blocks alone, before any cross section, so the
/// loader can be asked for them. A named MT the evaluation does not publish is
/// asked for and not found, and the fold reports its block in
/// [`Coverage::skipped_nc`].
pub(crate) fn reachable_mts(
    chain_nuclide: &ChainNuclide,
    blocks: &[CovarianceBlock],
) -> BTreeSet<i32> {
    let mut reached: BTreeSet<i32> = kinds_and_mts(chain_nuclide)
        .into_iter()
        .map(|(_, mt)| mt)
        .collect();
    let mut pending: Vec<i32> = reached.iter().copied().collect();
    while let Some(mt) = pending.pop() {
        for block in blocks {
            let CovarianceData::Nc(nc) = &block.data else {
                continue;
            };
            if block.mt != mt
                || !block.is_same_evaluation()
                || nc.lty != 0
                || block.partner_mt() != block.mt
            {
                continue;
            }
            for &x in &nc.xmti {
                let named = x as i32;
                if f64::from(named) == x && reached.insert(named) {
                    pending.push(named);
                }
            }
        }
    }
    reached
}

/// The named reactions' rate over a derivation's range, over the rate of the
/// reaction it derives, when the two differ by more than rounding.
///
/// ENDF-102 33.2.2.1 derives the covariance as if `σ_MT = Σ c_i σ_MTi` held
/// over `[E1, E2]`, and the fold builds the numerator from the right side and
/// divides by the rate of the left. So a ratio away from one means the
/// library's cross sections do not satisfy the derivation its covariance
/// states: ENDF/B-VIII.1 O16 MT 104 names 650 to 659, and above 20 MeV its
/// cross section also holds 660 to 669, which have no covariance.
///
/// Rounding acts on each named rate, so the difference is judged against
/// `Σ |c_i| R_MTi` rather than against the derived rate: with cancelling
/// coefficients, as in TENDL-2017 H2 `(n,2n)` = `σ_1 - σ_2 - σ_102`, a derived
/// rate of millibarns is the difference of barns, each rounded to its own
/// seven digits.
fn derivation_mismatch(
    flux: &FluxDensity,
    reactions: &BTreeMap<i32, &Reaction>,
    d: &Derivation,
) -> Option<f64> {
    let (&lo, &hi) = (flux.boundaries.first()?, flux.boundaries.last()?);
    let edges = [d.range.0.max(lo), d.range.1.min(hi)];
    if edges[0] >= edges[1] {
        return None;
    }
    let rate = |mt: i32| flux.xs_over(reactions[&mt], &edges)[0];
    let derived = rate(d.mt);
    let (mut named, mut magnitude) = (0.0, 0.0);
    for &(c, mt) in &d.named {
        let r = rate(mt);
        named += c * r;
        magnitude += (c * r).abs();
    }
    ((named - derived).abs() > DERIVATION_ROUNDING * magnitude.max(derived.abs()))
        .then(|| named / derived)
}

/// Fold one nuclide's covariance blocks against the flux.
///
/// `reactions` holds every cross section the nuclide has, not only the
/// channels', since a derived channel is weighted with the cross sections of
/// the reactions it is derived from.
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
    let expansion = channel_terms(blocks, reactions, kinds);
    let skipped = expansion.skipped();
    if skipped > 0 {
        coverage.skipped_nc.insert(nuclide.to_string(), skipped);
    }
    let mut by_mt: BTreeMap<i32, Vec<&Term>> = BTreeMap::new();
    for term in &expansion.terms {
        by_mt.entry(term.mt).or_default().push(term);
    }
    let reached: BTreeSet<i32> = by_mt.keys().copied().collect();
    // A channel's own name, or `MT<n>` for a reaction the fold reaches only
    // through a derivation.
    let name = |mt: i32| -> String {
        kinds
            .iter()
            .find(|(_, k)| *k == mt)
            .map(|(kind, _)| kind.clone())
            .unwrap_or_else(|| format!("MT{mt}"))
    };

    // Absolute covariance, in (1/s)^2, before relativizing.
    let mut absolute = vec![0.0; n * n];
    let mut used = 0;
    // Per channel named by a consumed block, the variance each of its terms'
    // own blocks states. Kept block by block rather than summed onto one grid,
    // because the blocks need not share a grid; `stated_variance_share` walks
    // them together.
    let mut variances: BTreeMap<usize, Vec<Diagonal>> = BTreeMap::new();
    // Per channel, the largest sum of partial rates any relative block of its
    // own reaction weighted it with, zero variance intervals included. The
    // consistency check against the rate, and not the share: it is what the
    // covariance's numerator was built from, whatever the block states. A
    // derived term's partials are another reaction's and are not compared.
    let mut weighted_rate: BTreeMap<usize, f64> = BTreeMap::new();
    // Per channel, the smallest such sum over the relative blocks of its own
    // reaction whose grid spans the whole flux range, the only ones whose
    // partials must add up to the rate rather than to at most it.
    let mut spanning_rate: BTreeMap<usize, f64> = BTreeMap::new();
    let spans = |grid: &[f64]| match (grid.first(), grid.last()) {
        (Some(&a), Some(&b)) => {
            a <= flux.boundaries[0] && b >= flux.boundaries[flux.boundaries.len() - 1]
        }
        _ => false,
    };

    // A pair stored in both orientations is folded from one copy only: the
    // lower MT's section, unless a block of it does not expand and the other
    // copy's all do. Only two copies that both expand have numbers to
    // compare. A copy left unused that does not expand is a layout gap
    // whichever copy it is: its numbers were never read, so nothing checks
    // them against the copy that was folded. Only pairs between two reactions
    // the fold reaches are listed, so a pair it would skip reports nothing.
    let mirrored = mirrored_pairs(blocks, &reached);
    let mut skipped_copy: BTreeSet<(i32, i32)> = BTreeSet::new();
    for &(a, b) in &mirrored {
        let (lower_expands, higher_expands) = (
            orientation_expands(blocks, a, b),
            orientation_expands(blocks, b, a),
        );
        let (unused, unused_expands) = if lower_expands || !higher_expands {
            ((b, a), higher_expands)
        } else {
            ((a, b), lower_expands)
        };
        skipped_copy.insert(unused);
        if lower_expands && higher_expands {
            let mismatch = mirror_mismatch(blocks, a, b);
            if mismatch > MIRROR_ROUNDING {
                coverage
                    .mirrored_disagree
                    .insert((nuclide.to_string(), name(a), name(b)), mismatch);
            }
        } else if !unused_expands {
            // The fold skips this copy before it expands it, so its failures
            // are counted here. The folded copy's, if neither expands, are
            // counted by the fold as it reaches them.
            count_unexpandable(blocks, unused.0, unused.1, coverage);
        }
    }

    for block in blocks {
        // Whether the fold reaches the block's own reaction. The partner's is
        // checked below for this evaluation's blocks; another evaluation's
        // partner is not this nuclide's to check.
        let reaches_row = reached.contains(&block.mt);
        if block.is_cross_material() {
            if reaches_row {
                *coverage
                    .skipped_cross_material
                    .entry(nuclide.to_string())
                    .or_insert(0) += 1;
            }
            continue;
        }
        if !block.names_cross_section() {
            if reaches_row {
                *coverage
                    .skipped_other_file
                    .entry(nuclide.to_string())
                    .or_insert(0) += 1;
            }
            continue;
        }
        let ni = match &block.data {
            CovarianceData::Ni(ni) => ni,
            // Consumed by `channel_terms`, as the terms it derives.
            CovarianceData::Nc(_) => continue,
        };

        let (row_mt, col_mt) = (block.mt, block.partner_mt());
        let (Some(row_terms), Some(col_terms)) = (by_mt.get(&row_mt), by_mt.get(&col_mt)) else {
            // A covariance for a reaction the fold does not reach. Not a gap:
            // there is no rate for it to be the uncertainty of.
            continue;
        };
        // Every reached MT has a cross section: `channel_terms` checks.
        let (row_rx, col_rx) = (reactions[&row_mt], reactions[&col_mt]);
        if skipped_copy.contains(&(row_mt, col_mt)) {
            continue;
        }

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

        // One set of partials per distinct range, since a reaction reached
        // through several derivations is usually reached over the same one.
        let partials = |rx: &Reaction, grid: &[f64], ts: &[&Term]| -> Vec<((f64, f64), Partials)> {
            let mut out: Vec<((f64, f64), Partials)> = Vec::new();
            for t in ts {
                if !out.iter().any(|(r, _)| *r == t.range) {
                    let p = partial_rates_within(flux, rx, grid, expanded.scale, t.range);
                    out.push((t.range, p));
                }
            }
            out
        };
        let rows = partials(row_rx, &expanded.row_energies, row_terms);
        let cols = partials(col_rx, &expanded.col_energies, col_terms);
        let of = |ps: &'_ [((f64, f64), Partials)], range: (f64, f64)| -> usize {
            ps.iter()
                .position(|(r, _)| *r == range)
                .expect("computed above")
        };

        for t in row_terms {
            let row = &rows[of(&rows, t.range)].1;
            for u in col_terms {
                let col = &cols[of(&cols, u.range)].1;
                let contribution = t.coefficient * u.coefficient * contract(&expanded, row, col);
                let (i, j) = (t.channel, u.channel);
                absolute[i * n + j] += contribution;
                if row_mt != col_mt {
                    // `rᵀ C r'` is a number, so it is both Cov(R_mt, R_mt1)
                    // and its transpose whichever section the block sits in: a
                    // block with MT1 below MT (ENDF/B-VIII.1 Np237 has 22)
                    // fills the same two cells as one written the other way
                    // round. A pair the tape also stores the other way is
                    // folded from one copy only (`mirrored_pairs`). A
                    // self-covariance block needs no mirror, since every
                    // ordered pair of terms visits it. Two terms of one
                    // channel correlated by a cross block land on the same
                    // cell twice, which is the `2 c_t c_u Cov` of a sum.
                    absolute[j * n + i] += contribution;
                }
            }
        }

        // Only a reaction's own blocks state its variance, and a term's over
        // its range only. A cross block correlates two reactions and gives
        // neither a variance, so it names the channels it reaches and covers
        // none.
        for t in col_terms {
            variances.entry(t.channel).or_default();
        }
        let diagonal = (row_mt == col_mt).then(|| Diagonal::of(&expanded));
        for t in row_terms {
            let own = variances.entry(t.channel).or_default();
            if let Some(d) = &diagonal {
                if t.coefficient != 0.0 {
                    own.push(d.within(t.range));
                }
            }
        }
        if expanded.scale == Scale::Relative {
            for (ts, ps, grid) in [
                (row_terms, &rows, &expanded.row_energies),
                (col_terms, &cols, &expanded.col_energies),
            ] {
                for t in ts.iter().filter(|t| t.is_own(kinds)) {
                    let sum = ps[of(ps, t.range)].1.total().abs();
                    let e = weighted_rate.entry(t.channel).or_insert(0.0);
                    *e = e.max(sum);
                    if spans(grid) {
                        let e = spanning_rate.entry(t.channel).or_insert(f64::INFINITY);
                        *e = e.min(sum);
                    }
                }
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

    for (channel, (kind, mt)) in kinds.iter().enumerate() {
        let (Some(diagonals), Some(reaction)) = (variances.get(&channel), reactions.get(mt)) else {
            continue;
        };
        let key = (nuclide.to_string(), kind.clone());
        // A derived channel's numerator is built from the reactions its NC
        // blocks name, so its check is that they add up to the reaction they
        // derive, over each block's range, nested derivations included.
        let mismatched: Vec<(&Derivation, f64)> = expansion
            .derivations
            .iter()
            .filter(|d| d.channel == channel)
            .filter_map(|d| Some((d, derivation_mismatch(flux, reactions, d)?)))
            .collect();
        // Rate a derivation's named reactions do not carry has no stated
        // variance, so it is taken out of the covered share.
        let short: Vec<&Derivation> = mismatched
            .iter()
            .filter(|(_, ratio)| *ratio < 1.0)
            .map(|(d, _)| *d)
            .collect();
        if let Some(share) = stated_variance_share(flux, reaction, diagonals, reactions, &short) {
            coverage.rate_fraction_covered.insert(key.clone(), share);
        }
        let mut ratios = Vec::new();
        if let (Some(&weighted), Some(&full)) = (weighted_rate.get(&channel), rates.get(kind)) {
            if full != 0.0 && weighted / full > 1.0 + PARTIALS_ROUNDING {
                ratios.push(weighted / full);
            }
        }
        if let (Some(&spanning), Some(&full)) = (spanning_rate.get(&channel), rates.get(kind)) {
            if full != 0.0 && spanning / full < 1.0 - PARTIALS_ROUNDING {
                ratios.push(spanning / full);
            }
        }
        ratios.extend(mismatched.iter().map(|(_, ratio)| *ratio));
        for ratio in ratios {
            if ratio > 1.0 {
                let e = coverage
                    .partials_above_rate
                    .entry(key.clone())
                    .or_insert(ratio);
                *e = e.max(ratio);
            } else {
                let e = coverage
                    .partials_below_rate
                    .entry(key.clone())
                    .or_insert(ratio);
                *e = e.min(ratio);
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
/// [`compute_multigroup_reaction_rates_shielded`](crate::multigroup::compute_multigroup_reaction_rates_shielded)
/// for the SAME spectrum and the SAME `shielding`, because the relativization
/// divides by them. Relative covariance is invariant under the per-step
/// `scale_rates`, which is why this is computed once per distinct spectrum
/// rather than once per step.
///
/// With `shielding`, each nuclide's partial rates are taken under the flux
/// shape its collapse used, so they sum to the shielded rate they are divided
/// by. The shape is the nominal one: a perturbed cross section does not
/// deepen or fill its own flux dip here.
pub fn fold_rate_covariance(
    material: &Material,
    chain: &std::collections::HashMap<String, ChainNuclide>,
    rates: &ReactionRates,
    multigroup_flux: &[f64],
    group_boundaries: &[f64],
    shielding: Option<&Shielding>,
) -> (BTreeMap<String, RateCovariance>, Coverage) {
    let mut out = BTreeMap::new();
    let mut coverage = Coverage::default();

    if multigroup_flux.is_empty() || group_boundaries.len() != multigroup_flux.len() + 1 {
        return (out, coverage);
    }
    let shapes = CollapseShapes::new(material, multigroup_flux, group_boundaries, shielding);

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
        // Every cross section, not only the channels': a derived channel is
        // weighted with those of the reactions it is derived from.
        let reactions: BTreeMap<i32, &Reaction> =
            by_mt.iter().map(|(mt, r)| (*mt, r.as_ref())).collect();
        let nuclide_rates: BTreeMap<String, f64> = rates
            .get(*name)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();

        // The nuclide's own shape, as its collapse built it; `None` wherever
        // the collapse averaged dilute, so the two agree nuclide by nuclide.
        let shape = shapes.as_ref().and_then(|s| s.shape_for(name));
        let flux = FluxDensity {
            boundaries: group_boundaries,
            flux: multigroup_flux,
            shape: shape.as_ref(),
        };
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
    // rate and by the parent's own density. The share is of the fold's own
    // rate whatever `rates` is, see `Coverage::covered_production`. Done here
    // rather than per nuclide because it is a property of the material: the
    // per-nuclide fold knows its own rates but not how many atoms of it there
    // are, and a channel on a 0.1%-abundance isotope must not count the same
    // as one on the bulk.
    //
    // Weighted by rate rather than counted per channel for the same reason the
    // per-channel fraction exists: a channel with no covariance costs nothing
    // if nothing went through it. Summed in `names` order and in sorted kind
    // order, as the fold above is, so the totals are the same bits every run
    // rather than depending on the hash maps' iteration order.
    let densities = material.get_atoms_per_barn_cm().unwrap_or_default();
    for nuclide in names {
        let density = densities.get(nuclide).copied().unwrap_or(0.0);
        if density <= 0.0 {
            continue;
        }
        let mut kinds: Vec<(&String, &f64)> = rates[nuclide].iter().collect();
        kinds.sort_by(|a, b| a.0.cmp(b.0));
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

    /// Full coverage reads exactly one with no clamp to make it so: a share of
    /// 1.0 adds each production to both sums unchanged and in the same order,
    /// on productions whose sum rounds (0.1 + 0.2 is not 0.3).
    #[test]
    fn full_coverage_reads_exactly_one() {
        let mut c = Coverage::default();
        for production in [0.1, 0.2, 0.7, 1.0e-17, 3.0e5] {
            c.total_production += production;
            c.covered_production += production * 1.0;
        }
        assert_eq!(c.rate_fraction_total(), Some(1.0));
    }
}

#[cfg(test)]
mod stated_variance_tests {
    use super::*;

    /// Uneven groups with no boundary at 10 keV, so the covariance edge there
    /// falls inside a group and the overlap is exercised rather than avoided.
    const BOUNDARIES: [f64; 8] = [1.0e-5, 0.625, 10.0, 1.0e3, 1.0e5, 1.0e6, 1.4e7, 2.0e7];
    const FLUX: [f64; 7] = [1.0, 3.0, 7.0, 2.0, 11.0, 5.0, 13.0];

    fn flux() -> FluxDensity<'static> {
        FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
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
            mat: 0,
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
        assert!(c.partials_below_rate.is_empty());

        // Against half the rate, as a rate weighted some other way than the
        // partials could give, the partials the fold weighted with sum to
        // twice it. The zero variance interval is among them, so the check
        // must see it even though the share leaves it out, and the share
        // itself does not move.
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
        // Short of the rate, rightly: the rest of it is outside the grid.
        assert!(c.partials_below_rate.is_empty());
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
        assert_eq!(
            stated_variance_share(&flux(), &above, &[d], &BTreeMap::new(), &[]),
            None
        );
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
    /// against the fold's own integral, reads what it would anyway.
    #[test]
    fn partials_above_the_rate_are_reported_not_clamped_away() {
        let grid = [1.0e-5, 1.0e4, 2.0e7];
        // Half the rate the partials sum to, as a tallied rate could give
        // against partials weighted flat within each group.
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

    /// A grid spanning the whole flux range leaves no rate outside it, so its
    /// partials short of the rate are the same inconsistency as partials above
    /// it, understating the sigma instead. A grid that stops short cannot be
    /// checked that way, because rate from outside it is rightly left out.
    #[test]
    fn partials_below_the_rate_are_reported_where_the_grid_spans_it() {
        let key = ("W186".to_string(), "(n,gamma)".to_string());
        // Twice the rate the partials sum to.
        let spanning = [1.0e-5, 1.0e4, 2.0e7];
        let c = fold(&[block(102, 102, lb5(&spanning, &[0.01, 0.0, 0.01]))], 2.0);
        let ratio = c.partials_below_rate[&key];
        assert!((ratio - 0.5).abs() < 1.0e-12, "{ratio}");
        assert!(c.partials_above_rate.is_empty());
        assert!(
            c.has_gaps(),
            "an understated sigma is something to know about"
        );
        assert_eq!(fraction(&c, "(n,gamma)"), 1.0);

        let short = [1.0, 1.0e3, 1.0e6];
        let c = fold(&[block(102, 102, lb5(&short, &[0.01, 0.002, 0.03]))], 2.0);
        assert!(c.partials_below_rate.is_empty());

        // Across spectra the larger shortfall is the one kept.
        let mut merged = Coverage::default();
        merged.absorb(Coverage {
            partials_below_rate: BTreeMap::from([(key.clone(), 0.8)]),
            ..Default::default()
        });
        merged.absorb(Coverage {
            partials_below_rate: BTreeMap::from([(key.clone(), ratio)]),
            ..Default::default()
        });
        assert_eq!(merged.partials_below_rate[&key], ratio);
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
            shape: None,
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

#[cfg(test)]
mod shielded_split_tests {
    use super::*;

    /// Three groups, the middle one cut at 1 keV by the covariance grid below,
    /// so the split of a shielded group term between two intervals is what
    /// the partials depend on.
    const BOUNDARIES: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
    const FLUX: [f64; 3] = [2.0, 3.0, 5.0];
    const GRID: [f64; 5] = [1.0e-5, 0.625, 1.0e3, 1.0e5, 2.0e7];
    const CUT: f64 = 1.0e3;

    /// A dip between 100 eV and 10 keV, deepest at 1 keV, so the shielded
    /// split of the middle group differs from both the dilute split and a
    /// split by width.
    fn shape() -> FluxShape {
        FluxShape::from_points(
            vec![2.0e7, 1.0e4, 1.0e3, 1.0e2, 1.0e-5],
            vec![1.0, 1.0, 0.2, 1.0, 1.0],
        )
    }

    /// A cross section with points on both sides of the cut, so both parts of
    /// the middle group see the dip.
    fn capture() -> Reaction {
        Reaction {
            cross_section: vec![100.0, 10.0, 30.0, 50.0, 20.0, 1.0, 0.1].into(),
            threshold_idx: 0,
            energy: vec![1.0e-5, 1.0, 50.0, 500.0, 5.0e3, 5.0e4, 2.0e7].into(),
            mt_number: 102,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// The partials over `GRID` built from independent walks: each group's
    /// term as the collapse has it, and the cut group's term shared between
    /// its two parts in proportion to `weight` of each part's own walk.
    fn expected(
        whole: impl Fn(&GroupTerms, f64) -> f64,
        weight: impl Fn(&GroupTerms) -> f64,
    ) -> [f64; 4] {
        let (shape, xs) = (shape(), capture());
        let walk = |a: f64, b: f64| walk_group(&xs, a, b, Some(&shape), None);
        let group = |g: usize| whole(&walk(BOUNDARIES[g], BOUNDARIES[g + 1]), FLUX[g]);
        let below = weight(&walk(BOUNDARIES[1], CUT));
        let above = weight(&walk(CUT, BOUNDARIES[2]));
        let middle = group(1);
        [
            group(0),
            middle * below / (below + above),
            middle * above / (below + above),
            group(2),
        ]
        .map(|v| BARN_TO_CM2 * v)
    }

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1.0e-14 * w.abs(),
                "interval {k}: {g} against {w}"
            );
        }
    }

    /// A relative block weights with the partial rate: the cut group's
    /// shielded rate term, shared by its parts' shielded `∫ σ φ dE`.
    #[test]
    fn a_cut_group_splits_its_shielded_rate_by_its_parts_shielded_integrals() {
        let shape = shape();
        let flux = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: Some(&shape),
        };
        let got = partial_rates(&flux, &capture(), &GRID, Scale::Relative).per_interval;
        let want = expected(|t, phi| t.shielded() * phi, GroupTerms::shielded_integral);
        close(&got, &want);

        // The dip moves the split away from the dilute one, so the weighting
        // above is what the check sees.
        let dilute = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
        };
        let d = partial_rates(&dilute, &capture(), &GRID, Scale::Relative).per_interval;
        let (shielded_share, dilute_share) = (got[1] / (got[1] + got[2]), d[1] / (d[1] + d[2]));
        assert!(
            (shielded_share / dilute_share - 1.0).abs() > 0.05,
            "{shielded_share} against {dilute_share}"
        );
    }

    /// An absolute block weights with the partial flux: the cut group's flux,
    /// shared by its parts' shielded `∫ φ dE`.
    #[test]
    fn a_cut_group_splits_its_flux_by_its_parts_shielded_weights() {
        let shape = shape();
        let flux = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: Some(&shape),
        };
        let got = partial_rates(&flux, &capture(), &GRID, Scale::Absolute).per_interval;
        let want = expected(|_, phi| phi, GroupTerms::shielded_weight);
        close(&got, &want);
        // Split by the shape rather than by width.
        let by_width = (CUT - BOUNDARIES[1]) / (BOUNDARIES[2] - BOUNDARIES[1]);
        let share = got[1] / (got[1] + got[2]);
        assert!(
            (share / by_width - 1.0).abs() > 0.05,
            "{share} against {by_width}"
        );
    }

    /// Uncorrelated variances on the two sides of the cut give
    /// `(v_below r_below² + v_above r_above²) / R²` with the independent split
    /// above, so a variance on either side counts in proportion to that side's
    /// shielded share and not to where the whole group's term sits.
    #[test]
    fn uncorrelated_variances_either_side_of_a_cut_weight_with_the_shielded_split() {
        let shape = shape();
        let flux = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: Some(&shape),
        };
        let r = expected(|t, phi| t.shielded() * phi, GroupTerms::shielded_integral);
        let rate: f64 = r.iter().sum();
        let variances = [0.0, 0.04, 0.01, 0.0];
        let block = CovarianceBlock {
            mt: 102,
            subsection_idx: 0,
            block_idx: 0,
            mat1: 0,
            mt1: 102,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat: 0,
            data: CovarianceData::Ni(NiSubsection {
                lb: 1,
                ek: GRID.to_vec(),
                fk: variances.to_vec(),
                ..Default::default()
            }),
        };
        let xs = capture();
        let reactions = BTreeMap::from([(102, &xs)]);
        let kinds = vec![("(n,gamma)".to_string(), 102)];
        let rates = BTreeMap::from([("(n,gamma)".to_string(), rate)]);
        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &flux,
            &[block],
            &reactions,
            &kinds,
            &rates,
            "Au197",
            &mut coverage,
        )
        .expect("the block is usable");

        let want: f64 = variances
            .iter()
            .zip(&r)
            .map(|(v, r)| v * r * r)
            .sum::<f64>()
            / (rate * rate);
        let got = folded.get(0, 0);
        assert!((got / want - 1.0).abs() < 1.0e-12, "{got} against {want}");
        assert!(coverage.partials_above_rate.is_empty());
    }
}

#[cfg(test)]
mod same_evaluation_tests {
    //! Blocks ENDF-102 33.3.1 lets a tape write for this evaluation in more
    //! than one way: `mat1` as 0 or as the own MAT, `xmf1` as 0 or 3, a pair
    //! in the lower or the higher MT's section, or in both.

    use super::*;

    /// FENDL-3.2d Pt194's MAT.
    const MAT: i32 = 7837;
    const BOUNDARIES: [f64; 4] = [1.0e-5, 1.0, 1.0e6, 2.0e7];
    const FLUX: [f64; 3] = [1.0e10, 1.0e11, 1.0e12];
    const GRID: [f64; 3] = [1.0e-5, 1.0e4, 2.0e7];
    const KINDS: [(&str, i32); 3] = [("(n,2n)", 16), ("(n,gamma)", 102), ("(n,p)", 103)];

    fn flux() -> FluxDensity<'static> {
        FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
        }
    }

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

    fn block(mt: i32, mat1: i32, mt1: i32, ni: NiSubsection) -> CovarianceBlock {
        CovarianceBlock {
            mt,
            subsection_idx: 0,
            block_idx: 0,
            mat1,
            mt1,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat: MAT,
            data: CovarianceData::Ni(ni),
        }
    }

    /// `lb = 5`, `ls = 1` on `GRID`: 0.01 and 0.04 on the diagonal, 0.005 off.
    fn own(mt: i32) -> CovarianceBlock {
        let ni = NiSubsection {
            lb: 5,
            ls: 1,
            ne: GRID.len() as i64,
            ek: GRID.to_vec(),
            fkk: vec![0.01, 0.005, 0.04],
            ..Default::default()
        };
        block(mt, 0, mt, ni)
    }

    /// `lb = 6`: rows on `er`, columns on `ec`, `fkl` row-major.
    fn lb6(er: &[f64], ec: &[f64], fkl: &[f64]) -> NiSubsection {
        NiSubsection {
            lb: 6,
            ner: er.len() as i64,
            nec: ec.len() as i64,
            er: er.to_vec(),
            ec: ec.to_vec(),
            fkl: fkl.to_vec(),
            ..Default::default()
        }
    }

    /// A cross block of `mt` with `mt1`, rows on `GRID` and columns on a
    /// grid of their own, so a transposed copy has to swap them.
    fn cross(mt: i32, mat1: i32, mt1: i32) -> CovarianceBlock {
        block(
            mt,
            mat1,
            mt1,
            lb6(
                &GRID,
                &[1.0e-5, 1.0e2, 1.0e5, 2.0e7],
                &[0.002, -0.001, 0.003, 0.001, -0.004, 0.006],
            ),
        )
    }

    /// The same numbers as `cross(mt1, mat1, mt)`, written from `mt1`'s side.
    fn transposed(mt: i32, mat1: i32, mt1: i32) -> CovarianceBlock {
        let CovarianceData::Ni(ni) = cross(mt1, mat1, mt).data else {
            unreachable!()
        };
        let (rows, cols) = (ni.er.len() - 1, ni.ec.len() - 1);
        let fkl = (0..cols)
            .flat_map(|j| (0..rows).map(move |i| (i, j)))
            .map(|(i, j)| ni.fkl[i * cols + j])
            .collect::<Vec<_>>();
        block(mt, mat1, mt1, lb6(&ni.ec, &ni.er, &fkl))
    }

    fn fold(blocks: &[CovarianceBlock]) -> (Option<RateCovariance>, Coverage) {
        fold_driving(blocks, &KINDS.map(|(_, mt)| mt))
    }

    /// [`fold`] for a nuclide whose data has a reaction only for `driven`, of
    /// the chain's `KINDS`.
    fn fold_driving(
        blocks: &[CovarianceBlock],
        driven: &[i32],
    ) -> (Option<RateCovariance>, Coverage) {
        let rxs: Vec<Reaction> = KINDS.iter().map(|(_, mt)| reaction(*mt)).collect();
        let reactions: BTreeMap<i32, &Reaction> = KINDS
            .iter()
            .zip(&rxs)
            .map(|((_, mt), r)| (*mt, r))
            .filter(|(mt, _)| driven.contains(mt))
            .collect();
        let kinds: Vec<(String, i32)> = KINDS.iter().map(|(k, mt)| (k.to_string(), *mt)).collect();
        let full = BARN_TO_CM2 * flux().integrate_xs(&rxs[0], 1.0e-5, 2.0e7);
        let rates: BTreeMap<String, f64> = kinds.iter().map(|(k, _)| (k.clone(), full)).collect();
        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &flux(),
            blocks,
            &reactions,
            &kinds,
            &rates,
            "Pt194",
            &mut coverage,
        );
        (folded, coverage)
    }

    fn relative(blocks: &[CovarianceBlock]) -> Vec<f64> {
        fold(blocks).0.expect("the blocks are usable").relative
    }

    /// Index of Cov((n,2n), (n,gamma)) in the row-major matrix.
    const N2N_CAPTURE: usize = 1;

    /// FENDL-3.2d writes Pt194's (n,2n) x (n,gamma) block with `mat1` equal to
    /// its own MAT. That is this evaluation, so the correlation is folded, bit
    /// for bit as if `mat1` were 0, and nothing is counted as skipped.
    #[test]
    fn a_mat1_naming_the_own_mat_is_folded_like_zero() {
        let zero = relative(&[own(16), own(102), cross(16, 0, 102)]);
        let (named, coverage) = fold(&[own(16), own(102), cross(16, MAT, 102)]);
        let named = named.expect("usable").relative;
        assert_ne!(zero[N2N_CAPTURE], 0.0, "the cross block correlates the two");
        assert_eq!(named, zero);
        assert!(coverage.skipped_cross_material.is_empty());
        assert!(!coverage.has_gaps());
    }

    /// Without the own MAT, as in a file written before the column, the same
    /// block cannot be shown to be this evaluation's: it is left out and
    /// counted once for the nuclide.
    #[test]
    fn without_the_own_mat_the_block_is_skipped_and_counted() {
        let mut blocks = vec![own(16), own(102), cross(16, MAT, 102)];
        for b in &mut blocks {
            b.mat = 0;
        }
        let (folded, coverage) = fold(&blocks);
        assert_eq!(folded.expect("usable").relative[N2N_CAPTURE], 0.0);
        assert_eq!(
            coverage.skipped_cross_material,
            BTreeMap::from([("Pt194".to_string(), 1)])
        );
    }

    #[test]
    fn xmf1_three_is_a_cross_section_like_zero() {
        let zero = relative(&[own(16), own(102), cross(16, MAT, 102)]);
        let mut three = cross(16, MAT, 102);
        three.xmf1 = 3.0;
        assert_eq!(relative(&[own(16), own(102), three]), zero);
    }

    /// A partner in another file, MF=10 here, is not a cross section: not
    /// folded, and counted apart from another evaluation's blocks.
    #[test]
    fn another_file_is_skipped_and_counted_apart() {
        let mut other = cross(16, MAT, 102);
        other.xmf1 = 10.0;
        let (folded, coverage) = fold(&[own(16), own(102), other]);
        assert_eq!(folded.expect("usable").relative[N2N_CAPTURE], 0.0);
        assert_eq!(
            coverage.skipped_other_file,
            BTreeMap::from([("Pt194".to_string(), 1)])
        );
        assert!(coverage.skipped_cross_material.is_empty());
        assert!(coverage.has_gaps());
    }

    /// ENDF/B-VIII.1 Np237 writes some pairs only from the higher MT's
    /// section, (n,3n) with (n,2n) and capture with fission among them.
    /// `rᵀ C r'` is a number, so the block fills the same two cells as the
    /// same numbers written from the lower MT's side.
    #[test]
    fn a_pair_written_from_the_higher_mt_folds_like_its_transpose() {
        let lower = relative(&[own(16), own(102), cross(16, MAT, 102)]);
        let higher = relative(&[own(16), own(102), transposed(102, MAT, 16)]);
        for (i, (h, l)) in higher.iter().zip(&lower).enumerate() {
            assert!(
                (h - l).abs() <= 1.0e-14 * l.abs(),
                "cell {i}: {h} against {l}"
            );
        }
        let n = KINDS.len();
        assert_eq!(higher[N2N_CAPTURE], higher[n], "both halves are filled");
    }

    /// JEFF-4.0 Be9 stores 130 pairs both ways. Folding both copies would
    /// count each covariance twice, so the lower MT's copy is the one used,
    /// and the copies agreeing is checked rather than assumed.
    #[test]
    fn a_pair_stored_both_ways_is_folded_once() {
        let once = relative(&[own(16), own(102), cross(16, MAT, 102)]);
        let (both, coverage) = fold(&[
            own(16),
            own(102),
            cross(16, MAT, 102),
            transposed(102, MAT, 16),
        ]);
        assert_eq!(both.expect("usable").relative, once);
        assert!(
            coverage.mirrored_disagree.is_empty(),
            "{:?}",
            coverage.mirrored_disagree
        );
    }

    /// Copies that disagree are reported with how far apart they are, and
    /// the lower MT's copy is still the one folded.
    #[test]
    fn copies_that_disagree_are_reported() {
        let once = relative(&[own(16), own(102), cross(16, MAT, 102)]);
        let mut upper = transposed(102, MAT, 16);
        let CovarianceData::Ni(ni) = &mut upper.data else {
            unreachable!()
        };
        // The largest entry is 0.006; move one of them by 0.0006.
        ni.fkl[0] += 0.0006;
        let (both, coverage) = fold(&[own(16), own(102), cross(16, MAT, 102), upper]);
        assert_eq!(both.expect("usable").relative, once);
        let key = (
            "Pt194".to_string(),
            "(n,2n)".to_string(),
            "(n,gamma)".to_string(),
        );
        let mismatch = coverage.mirrored_disagree[&key];
        assert!((mismatch - 0.1).abs() < 1.0e-12, "{mismatch}");
        assert!(coverage.has_gaps());
    }

    /// A pair the nuclide has no rate for on one side is not folded, so two
    /// copies of it that disagree are not reported either.
    #[test]
    fn copies_of_a_pair_without_a_rate_are_not_compared() {
        let mut upper = transposed(102, MAT, 16);
        let CovarianceData::Ni(ni) = &mut upper.data else {
            unreachable!()
        };
        ni.fkl[0] += 0.0006;
        let (_, coverage) =
            fold_driving(&[own(16), own(102), cross(16, MAT, 102), upper], &[16, 103]);
        assert!(coverage.mirrored_disagree.is_empty());
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// `block` with its layout changed to one the fold does not expand.
    fn unexpandable(mut block: CovarianceBlock) -> CovarianceBlock {
        let CovarianceData::Ni(ni) = &mut block.data else {
            unreachable!()
        };
        ni.lb = 9;
        block
    }

    /// A higher-MT copy that does not expand has no numbers to compare. It
    /// is a layout gap, not a disagreement, and the lower copy is folded.
    #[test]
    fn an_unexpandable_higher_copy_is_a_layout_gap() {
        let once = relative(&[own(16), own(102), cross(16, MAT, 102)]);
        let (both, coverage) = fold(&[
            own(16),
            own(102),
            cross(16, MAT, 102),
            unexpandable(transposed(102, MAT, 16)),
        ]);
        assert_eq!(both.expect("usable").relative, once);
        assert!(coverage.mirrored_disagree.is_empty());
        assert_eq!(coverage.unsupported_layouts, BTreeMap::from([(9, 1)]));
    }

    /// When the lower copy does not expand, the higher one stands for the
    /// pair, and the lower copy is a layout gap, as the higher one is when
    /// the roles are swapped: nothing checked it against the folded copy.
    #[test]
    fn an_unexpandable_lower_copy_gives_way_to_the_higher() {
        let higher = relative(&[own(16), own(102), transposed(102, MAT, 16)]);
        let (both, coverage) = fold(&[
            own(16),
            own(102),
            unexpandable(cross(16, MAT, 102)),
            transposed(102, MAT, 16),
        ]);
        assert_eq!(both.expect("usable").relative, higher);
        assert!(coverage.mirrored_disagree.is_empty());
        assert_eq!(coverage.unsupported_layouts, BTreeMap::from([(9, 1)]));
    }

    /// When neither copy expands, both are counted as layout gaps.
    #[test]
    fn two_unexpandable_copies_are_both_counted() {
        let (_, coverage) = fold(&[
            own(16),
            own(102),
            unexpandable(cross(16, MAT, 102)),
            unexpandable(transposed(102, MAT, 16)),
        ]);
        assert!(coverage.mirrored_disagree.is_empty());
        assert_eq!(coverage.unsupported_layouts, BTreeMap::from([(9, 2)]));
    }

    /// A committed trimmed tape, converted and read back as a run reads it.
    fn blocks_from(compressed: &[u8], name: &str) -> Vec<CovarianceBlock> {
        let tmp = tempfile::tempdir().expect("temp dir");
        let mut raw = Vec::new();
        lzma_rs::xz_decompress(&mut &compressed[..], &mut raw).expect("fixture decompresses");
        let text = String::from_utf8(raw).expect("ENDF is text");
        let material = endf::Material::from_str(&text).expect("evaluation parses");
        assert!(yamc_convert::covariance::write_covariance(&material, tmp.path()).expect("writes"));
        yamc_nuclide::arrow::covariance_arrow::read_covariance(tmp.path(), name)
            .expect("reads")
            .expect("the file is there")
    }

    /// Fold `blocks` with every MT they name standing in as a channel, all
    /// with the falling cross section above.
    fn fold_all(blocks: &[CovarianceBlock]) -> (RateCovariance, Coverage) {
        let mts: BTreeSet<i32> = blocks.iter().flat_map(|b| [b.mt, b.partner_mt()]).collect();
        let rxs: Vec<Reaction> = mts.iter().map(|&mt| reaction(mt)).collect();
        let reactions: BTreeMap<i32, &Reaction> = mts.iter().copied().zip(&rxs).collect();
        let kinds: Vec<(String, i32)> = mts.iter().map(|mt| (format!("mt{mt:03}"), *mt)).collect();
        let full = BARN_TO_CM2 * flux().integrate_xs(&rxs[0], 1.0e-5, 2.0e7);
        let rates: BTreeMap<String, f64> = kinds.iter().map(|(k, _)| (k.clone(), full)).collect();
        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &flux(),
            blocks,
            &reactions,
            &kinds,
            &rates,
            "X",
            &mut coverage,
        )
        .expect("usable");
        (folded, coverage)
    }

    /// JEFF-4.0 Be9 as it is on the tape: its cross-reaction blocks name its
    /// own MAT, 130 pairs are stored in both orientations, and the two copies
    /// of each are exactly each other's transpose. Folding the tape gives what
    /// folding it with the higher-MT copies deleted gives.
    #[test]
    fn jeff_be9_stores_130_pairs_both_ways_and_folds_each_once() {
        const BE9: &[u8] = include_bytes!("../../endf/fixtures/n-004_Be_009_jeff-4.0_mf33.endf.xz");
        let blocks = blocks_from(BE9, "Be9");
        assert!(blocks.iter().all(|b| b.is_same_evaluation()));

        let every: BTreeSet<i32> = blocks.iter().flat_map(|b| [b.mt, b.partner_mt()]).collect();
        let mirrored = mirrored_pairs(&blocks, &every);
        assert_eq!(mirrored.len(), 130);
        for &(a, b) in &mirrored {
            assert_eq!(mirror_mismatch(&blocks, a, b), 0.0, "MT {a} x {b}");
        }

        let (tape, coverage) = fold_all(&blocks);
        assert!(coverage.mirrored_disagree.is_empty());
        assert!(coverage.skipped_cross_material.is_empty());
        let lower_only: Vec<CovarianceBlock> = blocks
            .iter()
            .filter(|b| !(b.partner_mt() < b.mt && mirrored.contains(&(b.partner_mt(), b.mt))))
            .cloned()
            .collect();
        assert_eq!(lower_only.len(), blocks.len() - 130);
        let (once, _) = fold_all(&lower_only);
        assert_eq!(tape, once);
        assert!(tape.relative.iter().any(|&v| v != 0.0));
    }

    /// ENDF/B-VIII.1 Np237's MT 16, 17, 18 and 102 sections: every block is
    /// this evaluation's, several are written from the higher MT's section,
    /// and none is stored twice, so each is folded as it stands. The tape
    /// folds to what it folds to with each of those blocks rewritten as its
    /// transpose in the lower MT's section.
    #[test]
    fn endfb_np237_pairs_from_the_higher_mt_are_folded() {
        const NP237: &[u8] = include_bytes!("../../endf/fixtures/n-093_Np_237_mf33.endf.xz");
        let blocks = blocks_from(NP237, "Np237");
        assert!(blocks.iter().all(|b| b.is_same_evaluation()));
        let higher = blocks
            .iter()
            .filter(|b| b.partner_mt() < b.mt && b.mat1 == b.mat)
            .count();
        assert!(higher > 0, "the fixture writes pairs from the higher MT");

        let (folded, coverage) = fold_all(&blocks);
        assert!(coverage.skipped_cross_material.is_empty());
        assert!(coverage.mirrored_disagree.is_empty());
        let lower: Vec<CovarianceBlock> = blocks
            .iter()
            .map(|b| {
                if b.partner_mt() >= b.mt {
                    return b.clone();
                }
                let CovarianceData::Ni(ni) = &b.data else {
                    panic!("MT {} x {} is not explicit", b.mt, b.mt1);
                };
                let e = expand_ni(ni).expect("the block expands");
                assert_eq!(e.scale, Scale::Relative);
                let (rows, cols) = (e.n_rows(), e.n_cols());
                let fkl = (0..cols)
                    .flat_map(|j| (0..rows).map(move |i| (i, j)))
                    .map(|(i, j)| e.get(i, j))
                    .collect();
                CovarianceBlock {
                    mt: b.partner_mt(),
                    mt1: b.mt,
                    data: CovarianceData::Ni(NiSubsection {
                        lb: 6,
                        ner: e.col_energies.len() as i64,
                        nec: e.row_energies.len() as i64,
                        er: e.col_energies.clone(),
                        ec: e.row_energies.clone(),
                        fkl,
                        ..Default::default()
                    }),
                    ..b.clone()
                }
            })
            .collect();
        let (rewritten, _) = fold_all(&lower);
        assert_eq!(folded.kinds, rewritten.kinds);
        // The same products summed in another order, so equal to rounding.
        let largest = folded.relative.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (k, (x, y)) in folded.relative.iter().zip(&rewritten.relative).enumerate() {
            assert!(
                (x - y).abs() <= 1.0e-12 * largest,
                "entry {k}: {x} against {y}"
            );
        }
        // Fission with capture and (n,2n) with (n,3n), pairs the tape writes
        // only from the higher MT's section.
        let pair = |mt: i32, mt1: i32| blocks.iter().any(|b| b.mt == mt && b.partner_mt() == mt1);
        let at = |mt: i32| {
            folded
                .kinds
                .iter()
                .position(|k| *k == format!("mt{mt:03}"))
                .unwrap()
        };
        for (lower, higher) in [(18, 102), (16, 17)] {
            assert!(pair(higher, lower) && !pair(lower, higher));
            assert_ne!(
                folded.get(at(lower), at(higher)),
                0.0,
                "MT {lower} x {higher}"
            );
        }
    }

    /// Only blocks on a channel the chain drives count, and a nuclide counts
    /// its blocks once however many spectra fold it.
    #[test]
    fn only_relevant_other_evaluation_blocks_count_once_per_nuclide() {
        // MT 2, elastic, is not a transmutation channel.
        let blocks = [own(16), cross(16, 9228, 18), cross(2, 9228, 18)];
        let (_, first) = fold(&blocks);
        let (_, second) = fold(&blocks);
        let mut merged = first.clone();
        merged.absorb(second);
        let expected = BTreeMap::from([("Pt194".to_string(), 1)]);
        assert_eq!(first.skipped_cross_material, expected);
        assert_eq!(merged.skipped_cross_material, expected);
    }
}

#[cfg(test)]
mod nc_derived_tests {
    //! NC blocks with LTY=0: a channel whose covariance is stated only as a
    //! sum of other reactions', as ENDF/B-VIII.1 O16 `(n,p)` is from its
    //! partials 600 to 603. Every expected value is the sandwich written out
    //! by hand with `integrate_xs` over each interval, not the fold's own
    //! partials.

    use super::*;
    use endf::mf::covariance::NcSubsection;

    const BOUNDARIES: [f64; 5] = [1.0e-5, 1.0, 1.0e5, 5.0e6, 2.0e7];
    const FLUX: [f64; 4] = [1.0e10, 3.0e11, 2.0e12, 5.0e12];
    const GRID: [f64; 3] = [1.0e-5, 1.0e4, 2.0e7];
    const NP: i32 = 103;
    const CAPTURE: i32 = 102;

    fn flux() -> FluxDensity<'static> {
        FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
        }
    }

    /// A cross section of its own shape per MT, so a weighting with the wrong
    /// reaction's shows.
    fn reaction(mt: i32) -> Reaction {
        let k = f64::from(mt % 100 + 1);
        Reaction {
            cross_section: vec![k, 0.5 * k, 0.1 * k * k].into(),
            threshold_idx: 0,
            energy: vec![1.0e-5, 1.0e5, 2.0e7].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![],
            scatter_in_cm: false,
            redundant: false,
        }
    }

    /// The reactions whose cross section is a combination of others', as
    /// `(MT, [(c_i, MT_i)])`: what an NC block derives has to hold in the
    /// cross sections too, or the fold reports that it does not.
    type Sums<'a> = &'a [(i32, &'a [(f64, i32)])];

    /// Reaction `mt`, summed from its terms where `sums` names it. Every
    /// reaction shares one energy grid, so the sum is exact pointwise and
    /// under the trapezoid.
    fn reaction_in(mt: i32, sums: Sums) -> Reaction {
        let Some((_, terms)) = sums.iter().find(|(m, _)| *m == mt) else {
            return reaction(mt);
        };
        let mut out = reaction(mt);
        let parts: Vec<Reaction> = terms.iter().map(|&(_, m)| reaction_in(m, sums)).collect();
        out.cross_section = (0..out.energy.len())
            .map(|k| {
                terms
                    .iter()
                    .zip(&parts)
                    .map(|(&(c, _), r)| c * r.cross_section[k])
                    .sum()
            })
            .collect::<Vec<f64>>()
            .into();
        out
    }

    fn with(mt: i32, mt1: i32, data: CovarianceData) -> CovarianceBlock {
        CovarianceBlock {
            mt,
            subsection_idx: 0,
            block_idx: 0,
            mat1: 0,
            mt1,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat: 825,
            data,
        }
    }

    /// `lb = 5`, `ls = 1` on `GRID`, upper triangle `[a, b, c]`.
    fn own(mt: i32, upper: [f64; 3]) -> CovarianceBlock {
        let ni = NiSubsection {
            lb: 5,
            ls: 1,
            ne: GRID.len() as i64,
            ek: GRID.to_vec(),
            fkk: upper.to_vec(),
            ..Default::default()
        };
        with(mt, mt, CovarianceData::Ni(ni))
    }

    /// `lb = 5`, `ls = 0` on `GRID`: the full 2x2 relative covariance of `mt`
    /// with `mt1`, row-major.
    fn cross(mt: i32, mt1: i32, full: [f64; 4]) -> CovarianceBlock {
        let ni = NiSubsection {
            lb: 5,
            ls: 0,
            ne: GRID.len() as i64,
            ek: GRID.to_vec(),
            fkk: full.to_vec(),
            ..Default::default()
        };
        with(mt, mt1, CovarianceData::Ni(ni))
    }

    /// An LTY=0 block: over `[e1, e2]`, `σ_mt = Σ c_i σ_mt_i`.
    fn nc(mt: i32, e1: f64, e2: f64, terms: &[(f64, i32)]) -> CovarianceBlock {
        let data = NcSubsection {
            lty: 0,
            e1,
            e2,
            nci: terms.len() as i64,
            ci: terms.iter().map(|t| t.0).collect(),
            xmti: terms.iter().map(|t| f64::from(t.1)).collect(),
            ..Default::default()
        };
        with(mt, mt, CovarianceData::Nc(data))
    }

    /// Reaction `mt`'s partial rate over interval `k` of `GRID`, only its part
    /// inside `[lo, hi]`.
    fn partial(mt: i32, sums: Sums, k: usize, lo: f64, hi: f64) -> f64 {
        let (a, b) = (GRID[k].max(lo), GRID[k + 1].min(hi));
        BARN_TO_CM2 * flux().integrate_xs(&reaction_in(mt, sums), a, b)
    }

    /// `rᵀ C r'` with `C` full and row-major on `GRID`.
    fn sandwich(row: [f64; 2], c: [f64; 4], col: [f64; 2]) -> f64 {
        (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| row[i] * c[i * 2 + j] * col[j])
            .sum()
    }

    fn partials(mt: i32, sums: Sums, lo: f64, hi: f64) -> [f64; 2] {
        [partial(mt, sums, 0, lo, hi), partial(mt, sums, 1, lo, hi)]
    }

    /// The full symmetric matrix of `own`'s upper triangle.
    fn symmetric(u: [f64; 3]) -> [f64; 4] {
        [u[0], u[1], u[1], u[2]]
    }

    /// The rate the collapse gives `mt` over the whole flux range.
    fn rate(mt: i32, sums: Sums) -> f64 {
        BARN_TO_CM2 * flux().integrate_xs(&reaction_in(mt, sums), BOUNDARIES[0], BOUNDARIES[4])
    }

    /// Fold `blocks` with `channels` driven and every MT in `present` given a
    /// cross section, the ones in `sums` summed from others'.
    fn fold(
        blocks: &[CovarianceBlock],
        channels: &[(&str, i32)],
        present: &[i32],
        sums: Sums,
    ) -> (Option<RateCovariance>, Coverage) {
        let rxs: Vec<Reaction> = present.iter().map(|&mt| reaction_in(mt, sums)).collect();
        let reactions: BTreeMap<i32, &Reaction> = present.iter().copied().zip(&rxs).collect();
        let kinds: Vec<(String, i32)> = channels
            .iter()
            .map(|(k, mt)| (k.to_string(), *mt))
            .collect();
        let rates: BTreeMap<String, f64> = kinds
            .iter()
            .map(|(k, mt)| (k.clone(), rate(*mt, sums)))
            .collect();
        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &flux(),
            blocks,
            &reactions,
            &kinds,
            &rates,
            "O16",
            &mut coverage,
        );
        (folded, coverage)
    }

    fn close(got: f64, want: f64) {
        assert!(want != 0.0);
        assert!(
            ((got - want) / want).abs() < 1.0e-12,
            "got {got:e}, want {want:e}"
        );
    }

    const C600: [f64; 3] = [0.04, 0.01, 0.09];
    const C601: [f64; 3] = [0.01, -0.002, 0.0225];
    const C600_601: [f64; 4] = [0.003, 0.001, -0.004, 0.02];

    /// `(n,p) = 600 + 601` everywhere, with the two partials correlated: the
    /// variance is `r600ᵀ C00 r600 + r601ᵀ C11 r601 + 2 r600ᵀ C01 r601` over
    /// the `(n,p)` rate squared, and the NC block is not reported as skipped.
    #[test]
    fn a_sum_of_correlated_partials_is_the_hand_sandwich() {
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &[(1.0, 600), (1.0, 601)]),
            own(600, C600),
            cross(600, 601, C600_601),
            own(601, C601),
        ];
        let sums: Sums = &[(NP, &[(1.0, 600), (1.0, 601)])];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        let folded = folded.expect("derived");
        let (r0, r1) = (
            partials(600, sums, 0.0, f64::INFINITY),
            partials(601, sums, 0.0, f64::INFINITY),
        );
        let absolute = sandwich(r0, symmetric(C600), r0)
            + sandwich(r1, symmetric(C601), r1)
            + 2.0 * sandwich(r0, C600_601, r1);
        close(folded.get(0, 0), absolute / rate(NP, sums).powi(2));
        assert!(coverage.skipped_nc.is_empty());
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// The derivation applies over `[E1, E2]` only, here cutting the upper
    /// interval at 3 MeV, and each `c_i` enters with its sign: `c = -1` flips
    /// the cross term.
    #[test]
    fn the_range_and_the_signs_are_the_blocks_own() {
        const E2: f64 = 3.0e6;
        let terms = [(1.0, 601), (-1.0, 600)];
        let blocks = [
            nc(NP, 1.0e-5, E2, &terms),
            own(600, C600),
            cross(600, 601, C600_601),
            own(601, C601),
        ];
        let sums: Sums = &[(NP, &terms)];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        let (r0, r1) = (partials(600, sums, 0.0, E2), partials(601, sums, 0.0, E2));
        let absolute = sandwich(r0, symmetric(C600), r0) + sandwich(r1, symmetric(C601), r1)
            - 2.0 * sandwich(r0, C600_601, r1);
        close(
            folded.expect("derived").get(0, 0),
            absolute / rate(NP, sums).powi(2),
        );
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// A derived channel is correlated with another channel through the cross
    /// blocks of the reactions it is derived from, and its own NI blocks add
    /// to the derived part, since the blocks of a subsection add.
    #[test]
    fn a_derived_channel_correlates_through_its_partials_cross_blocks() {
        const C102: [f64; 3] = [0.0025, 0.0, 0.01];
        const C102_600: [f64; 4] = [0.001, 0.0, 0.002, 0.005];
        // The own block states variance only above 10 keV and the NC block
        // derives below it, as ENDF-102 33.3.3 item 3 has it.
        const OWN_NP: [f64; 3] = [0.0, 0.0, 0.0064];
        let blocks = [
            own(CAPTURE, C102),
            cross(CAPTURE, 600, C102_600),
            nc(NP, 1.0e-5, 1.0e4, &[(2.0, 600)]),
            own(NP, OWN_NP),
            own(600, C600),
        ];
        let channels = [("(n,gamma)", CAPTURE), ("(n,p)", NP)];
        let sums: Sums = &[(NP, &[(2.0, 600)])];
        let (folded, coverage) = fold(&blocks, &channels, &[CAPTURE, NP, 600], sums);
        let folded = folded.expect("folded");
        let all = |mt| partials(mt, sums, 0.0, f64::INFINITY);
        let r600 = partials(600, sums, 0.0, 1.0e4);
        let (r102, r103) = (all(CAPTURE), all(NP));
        let (gamma, np) = (rate(CAPTURE, sums), rate(NP, sums));

        let covariance = 2.0 * sandwich(r102, C102_600, r600);
        close(folded.get(0, 1), covariance / (gamma * np));
        close(folded.get(1, 0), covariance / (gamma * np));
        let variance =
            sandwich(r103, symmetric(OWN_NP), r103) + 4.0 * sandwich(r600, symmetric(C600), r600);
        close(folded.get(1, 1), variance / np.powi(2));
        close(
            folded.get(0, 0),
            sandwich(r102, symmetric(C102), r102) / gamma.powi(2),
        );
        // Covered where either the own block or the derivation states a
        // variance, which is the whole range here.
        let share = coverage.rate_fraction_covered[&("O16".to_string(), "(n,p)".to_string())];
        assert_eq!(share, 1.0);
    }

    /// A derivation from a reaction that is itself derived expands to the
    /// explicit ones, over the overlap of the two ranges, with the
    /// coefficients multiplied: ENDF/B-VIII.1 O16 MT 4 names MT 103, which is
    /// `600 + ... + 603`.
    #[test]
    fn a_derivation_from_a_derived_reaction_expands_to_the_explicit_ones() {
        const E2: f64 = 3.0e6;
        let blocks = [
            nc(4, 1.0e-5, 2.0e7, &[(3.0, NP)]),
            nc(NP, 1.0e-5, E2, &[(2.0, 600)]),
            own(600, C600),
        ];
        let sums: Sums = &[(4, &[(3.0, NP)]), (NP, &[(2.0, 600)])];
        let (folded, coverage) = fold(&blocks, &[("(n,inelastic)", 4)], &[4, NP, 600], sums);
        let r = partials(600, sums, 0.0, E2);
        let absolute = 36.0 * sandwich(r, symmetric(C600), r);
        close(
            folded.expect("derived").get(0, 0),
            absolute / rate(4, sums).powi(2),
        );
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// What cannot be derived is counted once per nuclide and folds nothing:
    /// a named reaction with no cross section (the whole block is skipped,
    /// not the part that could be summed), an LTY other than 0, and two
    /// reactions each derived from the other. An NC block on a reaction the
    /// fold does not reach is not counted.
    #[test]
    fn what_cannot_be_derived_is_counted_and_folds_nothing() {
        let mut ratio = nc(NP, 1.0e-5, 2.0e7, &[]);
        if let CovarianceData::Nc(data) = &mut ratio.data {
            data.lty = 1;
        }
        let mut missing = nc(NP, 1.0e-5, 2.0e7, &[(1.0, 600), (1.0, 601)]);
        missing.block_idx = 1;
        let unreached = nc(2, 1.0e-5, 2.0e7, &[(1.0, 600)]);
        let blocks = [ratio, missing, unreached, own(600, C600), own(601, C601)];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600], &[]);
        assert!(folded.is_none(), "601 has no cross section");
        assert_eq!(
            coverage.skipped_nc,
            BTreeMap::from([("O16".to_string(), 2)])
        );

        let first = nc(NP, 1.0e-5, 2.0e7, &[(1.0, 600)]);
        let back = nc(600, 1.0e-5, 2.0e7, &[(1.0, NP)]);
        let sums: Sums = &[(NP, &[(1.0, 600)])];
        let (folded, coverage) = fold(
            &[first, back, own(600, C600)],
            &[("(n,p)", NP)],
            &[NP, 600],
            sums,
        );
        // 103 derives from 600, whose block naming 103 is circular and is
        // skipped; 600's own block still folds through the first derivation.
        let r = partials(600, sums, 0.0, f64::INFINITY);
        close(
            folded.expect("derived").get(0, 0),
            sandwich(r, symmetric(C600), r) / rate(NP, sums).powi(2),
        );
        assert_eq!(
            coverage.skipped_nc,
            BTreeMap::from([("O16".to_string(), 1)])
        );
    }

    /// A derivation whose named reactions do not add up to the channel's
    /// cross section over its range is reported with their ratio, below one
    /// where the channel holds rate the block does not name (ENDF/B-VIII.1
    /// O16 MT 104 above 20 MeV) and above one the other way. Over the range
    /// only: the rate outside it is the own blocks' to state.
    #[test]
    fn a_derivation_the_cross_sections_do_not_satisfy_is_reported() {
        const E2: f64 = 3.0e6;
        let terms = [(1.0, 600), (1.0, 601)];
        let blocks = [nc(NP, 1.0e-5, E2, &terms), own(600, C600), own(601, C601)];
        let over =
            |mt: i32, sums: Sums| flux().integrate_xs(&reaction_in(mt, sums), BOUNDARIES[0], E2);
        let key = ("O16".to_string(), "(n,p)".to_string());

        // The channel's own cross section, larger than the sum it is said to be.
        let (_, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], &[]);
        let want = (over(600, &[]) + over(601, &[])) / over(NP, &[]);
        assert!(want < 1.0);
        close(coverage.partials_below_rate[&key], want);
        assert!(coverage.partials_above_rate.is_empty());

        let sums: Sums = &[(NP, &[(0.5, 600), (0.5, 601)])];
        let (_, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        close(coverage.partials_above_rate[&key], 2.0);
        assert!(coverage.partials_below_rate.is_empty());
    }

    /// An LTY=0 block with an empty range of its own, or naming nothing,
    /// derives nothing and is counted, not consumed in silence.
    #[test]
    fn an_empty_range_or_an_empty_list_is_counted() {
        let backwards = nc(NP, 2.0e7, 1.0e-5, &[(1.0, 600)]);
        let mut empty = nc(NP, 1.0e-5, 2.0e7, &[]);
        empty.block_idx = 1;
        let (folded, coverage) = fold(
            &[backwards, empty, own(600, C600)],
            &[("(n,p)", NP)],
            &[NP, 600],
            &[],
        );
        assert!(folded.is_none());
        assert_eq!(
            coverage.skipped_nc,
            BTreeMap::from([("O16".to_string(), 2)])
        );
    }

    /// A block is skipped only if no path derives it. Two channels each
    /// derived from the other derive each block on one path and meet it
    /// circularly on the other, and a block whose range misses the range it
    /// was reached over does not apply there and is not counted.
    #[test]
    fn a_block_derived_on_some_path_is_not_counted_as_skipped() {
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &[(1.0, 600)]),
            nc(600, 1.0e-5, 2.0e7, &[(1.0, NP)]),
            own(600, C600),
        ];
        let sums: Sums = &[(NP, &[(1.0, 600)])];
        let channels = [("(n,p)", NP), ("(n,p0)", 600)];
        let (folded, coverage) = fold(&blocks, &channels, &[NP, 600], sums);
        assert!(folded.is_some());
        assert!(coverage.skipped_nc.is_empty(), "{:?}", coverage.skipped_nc);

        let mut ratio = nc(NP, 1.0e5, 2.0e7, &[(1.0, 600)]);
        if let CovarianceData::Nc(data) = &mut ratio.data {
            data.lty = 1;
        }
        ratio.block_idx = 1;
        let blocks = [nc(4, 1.0e-5, 1.0e4, &[(3.0, NP)]), ratio, own(NP, C600)];
        let sums: Sums = &[(4, &[(3.0, NP)])];
        let (folded, coverage) = fold(&blocks, &[("(n,inelastic)", 4)], &[4, NP], sums);
        assert!(folded.is_some());
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// A range that holds the whole grid gives the undivided partials, bit
    /// for bit, so a nuclide without NC blocks folds as it always has.
    #[test]
    fn a_range_over_the_whole_grid_changes_no_bits() {
        let rx = reaction(600);
        let whole = partial_rates(&flux(), &rx, &GRID, Scale::Relative).per_interval;
        for range in [EVERYWHERE, (1.0e-5, 2.0e7), (0.0, 1.0e8)] {
            let within = partial_rates_within(&flux(), &rx, &GRID, Scale::Relative, range);
            assert_eq!(within.per_interval, whole);
        }
    }
}
