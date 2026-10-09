//! Fold MF=33 covariance against the flux to get the covariance of the
//! collapsed reaction rates.
//!
//! # Why the covariance is never regridded
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
//! outside the contraction. (An `lb = 4` block is written on the union of its
//! own two tables, which is still that one block's grid.)
//!
//! The one term that depends on the flux groups is `lb = 8`. ENDF-102 states it
//! as an absolute short-range variance whose effect on an average over `ΔEj`
//! scales as `ΔEk/ΔEj`, so the rate's variance depends on how the flux varies
//! inside each tape interval. The fold cuts the interval at the group
//! boundaries, where the flux density is constant, which makes that term exact
//! too; see [`Scale::ShortRange`].
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

use std::borrow::Cow;
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
    /// be the uncertainty of. Relevant on this nuclide's side only: a block
    /// is counted whether or not the evaluation `mat1` names is in the run,
    /// since a MAT is known only from a covariance block and a partner in the
    /// run with no covariance of its own would have none to match. Per
    /// nuclide rather than a sum, so a run with several spectra counts each
    /// block once.
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
    /// Blocks not consumed because they break ENDF-102's rules for their
    /// layout: arrays that do not match their own declared sizes, an LB=0 to 2
    /// block carrying a second table, an LB=3 or 4 block without one or whose
    /// two tables share no energy range, or an LB=8 variance stated between
    /// two different reactions.
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
    /// them states a variance. A reaction reached on several paths whose
    /// coefficients cancel there, to an exact zero, states none for the
    /// channel. Where the cross sections do not satisfy the derivation
    /// `σ_MT = Σ c_i σ_MTi` past rounding and the named reactions add up to
    /// less, the rate they miss, `c (σ_MT - Σ c_i σ_MTi)` for a derivation
    /// reached with coefficient `c`, is taken off each energy it falls in,
    /// down to none of it, and [`Coverage::partials_below_rate`] lists the
    /// channel as well: ENDF/B-VIII.1 O16 `(n,d)` above 20 MeV, where MT 660
    /// to 669 carry about 7% of the rate at 25 MeV and no covariance.
    ///
    /// Rate from an interval a grid spans with a variance of zero counts as
    /// uncovered, the same as rate from outside every grid, because neither
    /// carries a stated uncertainty. Below one, part of the rate enters the
    /// relative covariance's denominator and not its numerator; how much that
    /// lowers the relative sigma also depends on how the stated variance is
    /// spread over the covered part, so the share is not itself a dilution
    /// factor.
    ///
    /// The numerator adds, in the same order, the denominator's terms or
    /// smaller ones that are not negative, so the share lies in [0, 1] as long
    /// as no cross section or flux in it is negative, which
    /// [`Coverage::rate_fraction_total`] checks, and it does not depend on how
    /// the rate the covariance is divided by was computed. On a tallied rate,
    /// whose within-bin weighting the fold does not have, it is the share of
    /// the dilute rate over the tally spectrum and not of the tallied one, and
    /// the dilution the fold then applies differs from it; where the partials
    /// add up to more than that rate, [`Coverage::partials_above_rate`] lists
    /// it. On a dilute collapse under the flat-within-group weight, and on a
    /// shielded one, the denominator is the collapsed rate to rounding. Under
    /// the `1/E` weight it is only where no covariance edge cuts a group, for
    /// the reason given on
    /// [`Coverage::partials_above_rate`].
    ///
    /// Every channel a consumed block names has an entry, unless its rate over
    /// the flux range is zero (a threshold above the spectrum's top edge),
    /// which has no share to report. So a channel whose blocks state no
    /// variance anywhere reads zero rather than being absent.
    ///
    /// Every consumed self-covariance block counts, whatever its `lb`: a
    /// relative (LB=1 to 6), an absolute (LB=0) or a short-range (LB=8) block
    /// covers the energies where its own diagonal is nonzero. An LB=8 block
    /// states a variance `Fk` on each of its intervals, so an interval with a
    /// nonzero `Fk` counts as covered even when no other block states one
    /// there. An LB=8 block between two reactions is malformed and not
    /// consumed, so it names neither channel.
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
    /// sigma is overstated. A derived channel's partials are also the named
    /// reactions', over each NC block's range. Each such term's partials on a
    /// relative block are compared with its reaction's rate over that range,
    /// as the fold integrates it, and the ratio is carried over to the rate
    /// the covariance is divided by with the fold's own rate for the channel
    /// over that rate, which for the channel's own reaction is the comparison
    /// above. For every NC block the channel was expanded through, the named
    /// reactions' rate over the block's range, `Σ c_i R_MTi`, is compared
    /// with the rate of the reaction it derives over the same range (with
    /// the channel's own rate where that is zero). A ratio past rounding
    /// either way lands in this map or the one below, the larger excess or
    /// the larger shortfall kept. The partials are taken under the
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
    /// Absolute (`lb = 0`) and short-range (`lb = 8`) blocks weight with
    /// partial fluxes rather than partial rates, which have no rate to compare
    /// with, so they are not checked. The check from below is
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
    /// (see [`Coverage::rate_fraction_covered`]). Absolute and short-range
    /// blocks are not checked, as above. A derived channel whose named reactions add up to
    /// less than the reaction they derive is listed here too, with the
    /// smallest ratio over its NC blocks (see
    /// [`Coverage::partials_above_rate`]): ENDF/B-VIII.1 O16 `(n,d)` above
    /// 20 MeV, where MT 104 holds 660 to 669 and its NC block names 650 to
    /// 659.
    pub partials_below_rate: BTreeMap<(String, String), f64>,
    /// Per (nuclide, kind), where a derived channel's terms name two
    /// reactions that enter it with opposite signs over a common energy
    /// range, each with a self-covariance block stating a variance there, and
    /// the evaluation states no block between them, those pairs by kind,
    /// lower MT first.
    ///
    /// The fold reads the absent block as a zero covariance, which ENDF-102
    /// 33.3.2 a.1 lets a tape leave unstated, and with opposing signs that
    /// reading decides the channel's sigma: the two variances add where a positive correlation
    /// would cancel them. FENDL-3.2d and TENDL-2017 H2 `(n,2n)` is
    /// `σ_1 - σ_2 - σ_102` from 3.339 MeV with self blocks on MT 1, 2 and 102
    /// only, and folds to about 2400% near threshold and 22% at 14 MeV on
    /// TENDL-2017, while `σ_1` holds `σ_2` by construction. Reported so the
    /// sigma is read as the tape's literal statement and not as a measured
    /// one. Counted as a gap.
    pub derived_opposing_uncorrelated: BTreeMap<(String, String), BTreeSet<(String, String)>>,
    /// Per (nuclide, lumped MT), a lumped reaction whose covariance the fold
    /// could not give to any reaction, with its components by kind (`MT<n>`
    /// for one that is not a channel).
    ///
    /// ENDF-102 33.2.3 states a lumped reaction's covariance, MT 851-870, for
    /// the SUM of its components and none for each of them. Two uses of it are
    /// exact, and the fold makes both. A lump with one component is that
    /// component, and its blocks are read as the component's
    /// ([`single_component_lumps`]). A lump an LTY=0 block names is folded
    /// through that derivation, with its cross section built as the sum of
    /// its components' ([`lump_cross_sections`]): ENDF/B-VIII.1 and TENDL-2017
    /// U235 and U238 state MT 4 only as MT 51 plus MT 851, the sum of MT 52 to
    /// 91. Anywhere else, giving the sum's covariance to one component, or
    /// splitting it among several, would need an assumption the evaluation
    /// does not make, so the lump is listed here and not folded.
    /// ENDF/B-VIII.1, FENDL-3.2d and JEFF-4.0 W180 to W186 give `(n,2n)` only
    /// as MT 852, the sum of MT 16 and 41, and no derivation of a channel
    /// names it.
    ///
    /// Only lumps with a block of their own, and only where the fold reaches
    /// a component: a channel or a reaction one is derived from, or, for a
    /// lump the fold does not reach itself, a level of one (MT 51-91 of MT 4,
    /// 600-649 of MT 103, and so on to 875-891 of MT 16), since a channel's
    /// rate holds its levels'. So W186 MT 855, the sum of MT 600 and 649, is
    /// listed wherever `(n,p)` is a channel. It is not listed where the sum
    /// has a diagonal block of its own, as that states the channel's rate
    /// variance whole and loses nothing to a lump of part of it. A channel
    /// whose covariance is lumped this way may have none of its own, so it
    /// can also appear with no stated variance at all. Counted as a gap.
    ///
    /// A single-component lump whose component is a level, as Li7 MT 852 is
    /// MT 51 alone, is not listed where only the level's sum is reached. It
    /// is not a sum at all but that level's own covariance, and a reaction's
    /// own covariance the fold does not reach is not a gap anywhere in the
    /// fold, as on a tape writing it under MT 51 itself. A lump of several
    /// levels is listed because its covariance exists only for their sum,
    /// part of a reached channel's rate, and no reaction the fold reads can
    /// carry it.
    pub lumped_covariance_not_assignable: BTreeMap<(String, i32), BTreeSet<String>>,
    /// The production this spectrum drove, each channel's weighted by its
    /// share in [`Coverage::rate_fraction_covered`], and the production it
    /// drove in total. Both are per barn-cm per second, and both are weighted
    /// by the parent's own density, so a channel on a trace isotope counts for
    /// what it actually made.
    ///
    /// Each channel's production is the rate this run used, and its share is
    /// of the fold's own rate. On a dilute or a self-shielded collapse the two
    /// are one rate, and the covered sum is exactly the production from
    /// energies where a covariance states a nonzero variance, provided the
    /// flat within-group weight is in force (under `1/E` the share itself is
    /// off where an edge cuts a group, see [`Coverage::rate_fraction_covered`])
    /// and the fold weights each channel by the reaction whose rate it drove.
    /// The branching fold on the spectrum path breaks the last: it replaces
    /// the `(n,n')` rate of a nuclide with a metastable by the sum of its MF=10
    /// partials to the metastables, while the fold still weights and shares
    /// that channel by MT 4; with a relative MT 4 block the channel lands in
    /// [`Coverage::partials_above_rate`]. On a tallied rate the share is of the
    /// dilute rate over the tally spectrum, so the covered sum is that
    /// production only if the tallied rate kept the dilute rate's distribution
    /// in energy, which self-shielding in the transport does not: it depresses
    /// the resonance range, where capture blocks often state zero. The covered
    /// share of such a production is not computed, and the run reports no
    /// total (`Info::rate_fraction_covered_total`).
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
    /// On a dilute or a self-shielded spectrum run under the flat within-group
    /// weight whose partials agree with every rate, that is the share of the
    /// production driven from energies where a covariance states a nonzero
    /// variance. On a tallied run, where a channel is listed in
    /// [`Coverage::partials_above_rate`] or [`Coverage::partials_below_rate`],
    /// or under `1/E`, it is not, for the reasons given on
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
    /// In [0, 1] without a clamp as long as no cross section or production is
    /// negative: each channel adds `production * share` to one sum and
    /// `production` to the other, in the same order, with the share at most
    /// one. Rounding is monotone, so the covered sum cannot pass the total.
    /// Nothing upstream checks the sign, so a per-channel share or a total
    /// outside [0, 1] (a negative tabulated cross section in a covered cell, a
    /// negative production) is returned as an error naming it. Clamping would
    /// hide it, and a panic would abort a run that otherwise fails with a
    /// message.
    pub fn rate_fraction_total(&self) -> Result<Option<f64>, String> {
        let share = 0.0..=1.0;
        if let Some(((nuclide, kind), s)) = self
            .rate_fraction_covered
            .iter()
            .find(|(_, s)| !share.contains(*s))
        {
            return Err(format!(
                "{nuclide} {kind}: the share of its rate with a stated variance \
                 came out {s}, outside [0, 1]: a cross section or flux it integrates is \
                 negative"
            ));
        }
        if self.total_production <= 0.0 {
            return Ok(None);
        }
        let fraction = self.covered_production / self.total_production;
        if !share.contains(&fraction) {
            return Err(format!(
                "covered production {} against a total of {} is not a share: a \
                 production entering the covariance total is negative",
                self.covered_production, self.total_production
            ));
        }
        Ok(Some(fraction))
    }

    /// Fold one nuclide's report into this one.
    ///
    /// Every field is order-free: the sets union, the counters sum, the
    /// per-nuclide counts and mismatches take the larger,
    /// `rate_fraction_covered` is keyed by (nuclide, kind) and takes the
    /// smaller claim, `partials_above_rate` takes the larger excess and
    /// `partials_below_rate` the larger shortfall, the smaller ratio, and
    /// `derived_opposing_uncorrelated` and `lumped_covariance_not_assignable`
    /// union theirs. That is what lets the fold below run per nuclide in
    /// parallel and merge afterwards.
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
        for (key, pairs) in other.derived_opposing_uncorrelated {
            self.derived_opposing_uncorrelated
                .entry(key)
                .or_default()
                .extend(pairs);
        }
        for (key, components) in other.lumped_covariance_not_assignable {
            self.lumped_covariance_not_assignable
                .entry(key)
                .or_default()
                .extend(components);
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
            || !self.derived_opposing_uncorrelated.is_empty()
            || !self.lumped_covariance_not_assignable.is_empty()
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
    /// ones it keeps are summed in the same order.
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
            // The pieces' walks visit every point of the whole group's walk,
            // so on a nonnegative cross section a nonzero term leaves a
            // nonzero sum of pieces. A zero one is a bug here, and any way
            // of completing the partition (by width, say) would be the dilute
            // assumption this split exists to avoid, so it stops the run
            // rather than hand the partials a guess. A negative sum, possible
            // only on a cross section that goes negative, still normalizes to
            // a partition of the term and needs no special case.
            assert!(
                sum != 0.0,
                "the shielded pieces of group [{glo}, {ghi}] eV sum to zero \
                 against a collapse term of {term}"
            );
            for (w, part) in cuts.windows(2).zip(&parts) {
                if let Some(k) = interval_holding(grid, w[0], w[1]) {
                    out[k] += term * part / sum;
                }
            }
        }
        out
    }

    /// `∫_a^b ψ(E)² dE`, in (n/cm^2/s)^2 per eV, with `ψ` flat within each
    /// group. Needed only by short-range (`lb = 8`) blocks; see
    /// [`partial_rates`] and [`FluxDensity::density_squared`].
    fn integrate_density_squared(&self, a: f64, b: f64) -> f64 {
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
            total += density * density * (hi - lo);
        }
        total
    }

    /// `∫_a^b ψ(E)² dE` for a short-range block's interval, in (n/cm^2/s)^2
    /// per eV.
    ///
    /// Dilute, this is [`FluxDensity::integrate_density_squared`]. Shielded,
    /// the interval is cut at the flux-group boundaries and each piece carries
    /// its own shielded flux, split from the collapse's term by
    /// [`FluxDensity::flux_over`] exactly as an absolute block's is, with the
    /// density flat at that flux over the piece. That is the dilute rule with
    /// the shielded flux in place of the dilute one, and the same value when
    /// the shape is flat. Where the shape varies inside a piece the true
    /// `∫ ψ² dE` is larger than the flat one (Cauchy-Schwarz), so there the
    /// short-range variance is a lower bound.
    fn density_squared(&self, reaction: &Reaction, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        if self.shape.is_none() {
            return self.integrate_density_squared(a, b);
        }
        let mut cuts = vec![a];
        cuts.extend(self.boundaries.iter().copied().filter(|&e| e > a && e < b));
        cuts.push(b);
        self.flux_over(reaction, &cuts)
            .into_iter()
            .zip(cuts.windows(2))
            .map(|(phi, w)| phi * phi / (w[1] - w[0]))
            .sum()
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
        // A short-range variance `Fk` on `ΔEk` is `Fk·ΔEk/ΔEj` for the
        // average over any `ΔEj` inside it, with nothing correlating two
        // such intervals (ENDF-102 section 33.2.2.2). Cut `ΔEk` at the
        // flux-group boundaries, where `ψ` is constant, and the partial
        // rate over each piece `j` is that average times `ψ_j·ΔEj`, so
        // the rate's variance is `Fk·ΔEk·Σ_j ψ_j²·ΔEj`. The block is
        // diagonal, so the weight is the square root of what multiplies
        // `Fk`. That is exact for any flux grid, and a flat flux over the
        // whole interval gives `Fk·Φk²`, the absolute diagonal. On a
        // shielded run each piece's `ψ_j` is its shielded flux over its
        // width; see [`FluxDensity::density_squared`].
        Scale::ShortRange => grid
            .windows(2)
            .map(|w| ((w[1] - w[0]) * flux.density_squared(reaction, w[0], w[1])).sqrt())
            .collect(),
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
    // A short-range weight is `sqrt(ΔEk·∫ ψ² dE)` with `ΔEk` the block's own
    // interval, not the part of it in range, and square roots of pieces do
    // not add: the integral is restricted, the width is not.
    if scale == Scale::ShortRange {
        let per_interval = grid
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0].max(range.0), w[1].min(range.1));
                if a >= b {
                    return 0.0;
                }
                BARN_TO_CM2 * ((w[1] - w[0]) * flux.density_squared(reaction, a, b)).sqrt()
            })
            .collect();
        return Partials { per_interval };
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
}

/// The edges of the cells over which a channel's leaves and their stated
/// variances are constant, and finer than which the flux does not resolve a
/// rate: the flux group `boundaries` cut at every edge of the terms' and
/// derivations' ranges and of the own blocks of the reactions the terms name.
fn cell_edges(
    boundaries: &[f64],
    terms: &[&Term],
    derivations: &[&Derivation],
    diagonals: &BTreeMap<i32, Vec<Diagonal>>,
) -> Vec<f64> {
    let (Some(&lo), Some(&hi)) = (boundaries.first(), boundaries.last()) else {
        return Vec::new();
    };
    let mts: BTreeSet<i32> = terms.iter().map(|t| t.mt).collect();
    let mut edges: Vec<f64> = mts
        .iter()
        .filter_map(|mt| diagonals.get(mt))
        .flatten()
        .flat_map(|d| d.pieces.iter().flat_map(|&(lo, hi, _)| [lo, hi]))
        .chain(terms.iter().flat_map(|t| [t.range.0, t.range.1]))
        .chain(derivations.iter().flat_map(|d| [d.range.0, d.range.1]))
        .filter(|&e| lo < e && e < hi)
        .chain(boundaries.iter().copied())
        .collect();
    edges.sort_by(f64::total_cmp);
    edges.dedup();
    edges
}

/// Whether cell `w` lies inside `range`.
fn inside(range: (f64, f64), w: &[f64]) -> bool {
    range.0 <= w[0] && w[1] <= range.1
}

/// The net coefficient reaction `mt` enters a channel with over cell `w`:
/// the terms' less the derivations', since a derivation replaces the term it
/// expands by the reactions it names. Zero where `mt` is expanded, or not
/// reached.
fn leaf_coefficient(terms: &[&Term], derivations: &[&Derivation], mt: i32, w: &[f64]) -> f64 {
    terms
        .iter()
        .filter(|t| t.mt == mt && inside(t.range, w))
        .map(|t| t.coefficient)
        .sum::<f64>()
        - derivations
            .iter()
            .filter(|d| d.mt == mt && inside(d.range, w))
            .map(|d| d.coefficient)
            .sum::<f64>()
}

/// Whether reaction `mt`'s own blocks state a variance over cell `w`,
/// summed over them since the blocks of one subsection add, so two whose
/// tape values cancel exactly, bit for bit, state none. No tolerance is
/// applied: values that cancel only to rounding count as stated, since
/// calling them zero would be a judgement the tape does not make. Relative
/// and absolute blocks are summed apart, having different units. Short-range
/// (`lb = 8`) blocks are summed apart from both: their `Fk` is in barns
/// squared like an absolute block's, but it is the variance of the average
/// over the block's own interval and scales with the width of any narrower
/// one, so a cancellation against an absolute value at the tape's width would
/// not be one at any other.
fn states_variance(diagonals: &BTreeMap<i32, Vec<Diagonal>>, mt: i32, w: &[f64]) -> bool {
    let (mut relative, mut absolute, mut short_range) = (0.0, 0.0, 0.0);
    for d in diagonals.get(&mt).into_iter().flatten() {
        match d.scale {
            Scale::Relative => relative += d.over(w[0], w[1]),
            Scale::Absolute => absolute += d.over(w[0], w[1]),
            Scale::ShortRange => short_range += d.over(w[0], w[1]),
        }
    }
    relative != 0.0 || absolute != 0.0 || short_range != 0.0
}

/// The share of a channel's rate over the flux range, as the fold integrates
/// it, that comes from energies where the covariance states a variance for
/// it, or `None` when that rate is zero.
///
/// The channel's rate is carried, cell by cell, by its leaves: the reactions
/// among `terms` with a nonzero [`leaf_coefficient`] there. Outside every NC
/// range that is the channel's own reaction; inside one, the reactions the
/// block names. A leaf states a variance where [`states_variance`] says so.
/// An expanded reaction's own blocks cover nothing where it is expanded,
/// since ENDF-102 33.3.3 item 3 has them state zero over the NC range.
///
/// A cell counts when at least one leaf states a variance over it, less the
/// rate of the leaves that state none there, as `|c R|` so that one entering
/// with a negative coefficient is taken off too, and less the rate any
/// derivation misses over that cell, where its named reactions add up to
/// less than the one it derives by more than rounding: rate the covariance
/// states nothing for. Judged cell by cell, so a shortfall in one cell is
/// not made up by an excess in another within the same NC range. Down to
/// none of the cell, and clamped to it. A cell whose leaves all state a
/// variance, with no derivation short over it, counts whole.
///
/// The covered sum adds, in the same order, the total's terms or smaller ones
/// that are not negative, and rounded addition of such terms is monotone, so
/// the share cannot exceed one. A reaction whose blocks state a variance on
/// every interval across the whole flux range therefore reads exactly one.
fn stated_variance_share(
    flux: &FluxDensity,
    reaction: &Reaction,
    terms: &[&Term],
    derivations: &[&Derivation],
    diagonals: &BTreeMap<i32, Vec<Diagonal>>,
    reactions: &BTreeMap<i32, &Reaction>,
) -> Option<f64> {
    let edges = cell_edges(flux.boundaries, terms, derivations, diagonals);
    if edges.is_empty() {
        return None;
    }
    let mts: BTreeSet<i32> = terms.iter().map(|t| t.mt).collect();
    let rates: BTreeMap<i32, Vec<f64>> = mts
        .iter()
        .map(|&mt| (mt, flux.xs_over(reactions[&mt], &edges)))
        .collect();
    // Per cell, the channel's rate no named reaction carries: each
    // derivation's `coefficient (R_MT - Σ c_i R_MTi)` where that is a
    // shortfall past rounding, judged against `Σ |c_i R_MTi|` for the reason
    // given on [`derivation_mismatch`]. Every term of a derivation is itself
    // a term, so its rates are among `rates`. The gaps add, since a nested
    // derivation's is the shortfall inside a reaction its parent names.
    let mut missing = vec![0.0; edges.len() - 1];
    for d in derivations {
        for (k, w) in edges.windows(2).enumerate() {
            if !inside(d.range, w) {
                continue;
            }
            let derived = rates[&d.mt][k];
            let (mut named, mut magnitude) = (0.0, 0.0);
            for &(c, mt) in &d.named {
                let r = c * rates[&mt][k];
                named += r;
                magnitude += r.abs();
            }
            let gap = derived - named;
            if gap.abs() > DERIVATION_ROUNDING * magnitude.max(derived.abs()) {
                missing[k] += (d.coefficient * gap).max(0.0);
            }
        }
    }
    let (mut covered, mut total) = (0.0, 0.0);
    for (k, (w, rate)) in edges
        .windows(2)
        .zip(flux.xs_over(reaction, &edges))
        .enumerate()
    {
        total += rate;
        let (mut stating, mut unstated) = (false, 0.0);
        for &mt in &mts {
            let leaf = leaf_coefficient(terms, derivations, mt, w);
            if leaf == 0.0 {
                continue;
            }
            if states_variance(diagonals, mt, w) {
                stating = true;
            } else {
                unstated += (leaf * rates[&mt][k]).abs();
            }
        }
        if stating {
            covered += rate - (missing[k] + unstated).min(rate);
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

/// Every MT the fold can reach on `chain_nuclide`: its channels' own, those
/// the LTY=0 NC blocks on them name, transitively, and the components of a
/// lumped reaction among them.
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
    // As the fold reads them, so an NC block naming a single-component lump
    // asks for the component.
    let blocks = single_component_lumps(blocks);
    let blocks = blocks.as_ref();
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
    // A lumped reaction has no cross section of its own, and the fold builds
    // one from its components' ([`lump_cross_sections`]). Those are HEAD
    // records with no blocks, so nothing above asks for them.
    for (mtl, components) in lumped_reactions(blocks) {
        if reached.contains(&mtl) {
            reached.extend(components);
        }
    }
    reached
}

/// The lumped reactions an evaluation defines, each MTL with its components,
/// read off the components' own sections ([`CovarianceBlock::lumped_into`]).
fn lumped_reactions(blocks: &[CovarianceBlock]) -> BTreeMap<i32, BTreeSet<i32>> {
    let mut out: BTreeMap<i32, BTreeSet<i32>> = BTreeMap::new();
    for block in blocks {
        if let Some(mtl) = block.lumped_into() {
            out.entry(mtl).or_default().insert(block.mt);
        }
    }
    out
}

/// The blocks, with every lumped reaction that has a single component read
/// as that component.
///
/// ENDF-102 33.2.3 defines a lumped reaction as the sum of its components, so
/// with one component the two are one cross section and the lump's blocks
/// are the component's covariance, exactly: its own blocks, its cross blocks
/// with other reactions, and the LTY=0 NC blocks naming it. The fold has a
/// cross section for the component and none for MT 851-870, which the format
/// keeps out of Files 3 to 6, so the lump is renamed rather than left
/// unreached. The component's own section is its HEAD alone, so nothing is
/// stated for it twice, and that section's row is dropped here, having been
/// read. Only this evaluation's references are renamed: an `mt1` beside
/// another material's `mat1` is that material's MT.
///
/// Borrowed unchanged when no lump has a single component. On the local
/// ENDF/B-VIII.1, FENDL-3.2d, JEFF-4.0, JENDL-5.0, TENDL-2017 and TENDL-2025
/// tapes that is every evaluation but Li7, which gives MT 51 as MT 852 and
/// MT 56 as MT 854 in all four libraries that lump at all.
pub(crate) fn single_component_lumps(blocks: &[CovarianceBlock]) -> Cow<'_, [CovarianceBlock]> {
    let alias: BTreeMap<i32, i32> = lumped_reactions(blocks)
        .into_iter()
        .filter(|(_, components)| components.len() == 1)
        .filter_map(|(mtl, components)| components.first().map(|&c| (mtl, c)))
        .collect();
    if alias.is_empty() {
        return Cow::Borrowed(blocks);
    }
    let read = |mt: i32| alias.get(&mt).copied().unwrap_or(mt);
    Cow::Owned(
        blocks
            .iter()
            .filter(|b| !b.lumped_into().is_some_and(|mtl| alias.contains_key(&mtl)))
            .map(|b| {
                let mut b = b.clone();
                b.mt = read(b.mt);
                if b.is_same_evaluation() {
                    // `mt1 == 0` means `mt` itself, and stays so.
                    b.mt1 = read(b.mt1);
                    if let CovarianceData::Nc(nc) = &mut b.data {
                        if nc.lty == 0 {
                            for x in &mut nc.xmti {
                                let mt = *x as i32;
                                if f64::from(mt) == *x {
                                    *x = f64::from(read(mt));
                                }
                            }
                        }
                    }
                }
                b
            })
            .collect(),
    )
}

/// The cross section of each lumped reaction an LTY=0 block names, built as
/// the sum of its components' where that sum is exact on one grid.
///
/// ENDF-102 33.2.3 keeps a lumped reaction out of Files 3 to 6: "the net cross
/// section ... must be constructed at the processing stage by summing over
/// the reaction components", and a reaction may be derived from lumped ones.
/// ENDF/B-VIII.1 and TENDL-2017 U235 and U238 state MT 4 only as MT 51 plus
/// MT 851, the sum of MT 52 to 91. With the sum built, that derivation folds
/// the lump's own blocks exactly, and nothing is given to a component. Only
/// lumps with several components, since one with a single component is read
/// as it ([`single_component_lumps`]), and only where every component has a
/// cross section here.
///
/// The components are summed point by point, which is exact only when they
/// share one grid: each component's energies are the tail of the one reaching
/// lowest, as a nuclide's reactions are on its own grid past their threshold
/// index, and each one starting higher is zero at its first point, since below
/// it the component is zero and the sum interpolates from there. Otherwise no
/// sum is built, the derivation finds no cross section, and its block is
/// counted in [`Coverage::skipped_nc`].
fn lump_cross_sections(
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
) -> Vec<Reaction> {
    let named: BTreeSet<i32> = blocks
        .iter()
        .filter(|b| b.is_same_evaluation())
        .filter_map(|b| match &b.data {
            CovarianceData::Nc(nc) if nc.lty == 0 => Some(&nc.xmti),
            _ => None,
        })
        .flatten()
        .filter(|&&x| f64::from(x as i32) == x)
        .map(|&x| x as i32)
        .collect();
    lumped_reactions(blocks)
        .into_iter()
        .filter(|(mtl, components)| {
            components.len() > 1 && named.contains(mtl) && !reactions.contains_key(mtl)
        })
        .filter_map(|(mtl, components)| {
            let parts: Vec<&Reaction> = components
                .iter()
                .map(|c| reactions.get(c).copied())
                .collect::<Option<_>>()?;
            summed(mtl, &parts)
        })
        .collect()
}

/// Reaction `mt`, the sum of `parts`, when [`lump_cross_sections`]'s grid
/// condition holds.
fn summed(mt: i32, parts: &[&Reaction]) -> Option<Reaction> {
    let base = parts.iter().min_by_key(|r| r.threshold_idx)?;
    let grid: &[f64] = &base.energy;
    let mut sum = vec![0.0; grid.len()];
    for part in parts {
        let (energy, values): (&[f64], &[f64]) = (&part.energy, &part.cross_section);
        let offset = part.threshold_idx - base.threshold_idx;
        let tail = grid.get(offset..)?;
        let continuous = offset == 0 || values.first() == Some(&0.0);
        if energy.is_empty() || energy != tail || values.len() != energy.len() || !continuous {
            return None;
        }
        for (s, v) in sum[offset..].iter_mut().zip(values) {
            *s += v;
        }
    }
    Some(Reaction {
        cross_section: sum.into(),
        threshold_idx: base.threshold_idx,
        energy: grid.to_vec().into(),
        mt_number: mt,
        q_value: 0.0,
        products: vec![].into(),
        scatter_in_cm: false,
        redundant: true,
    })
}

/// The redundant reaction whose cross section holds level `mt`, per ENDF-102
/// Appendix B.1: MT 4 holds 51 to 91, MT 103 to 107 the levels of each emitted
/// particle, and MT 16 its levels 875 to 891.
fn level_sum(mt: i32) -> Option<i32> {
    match mt {
        51..=91 => Some(4),
        600..=649 => Some(103),
        650..=699 => Some(104),
        700..=749 => Some(105),
        750..=799 => Some(106),
        800..=849 => Some(107),
        875..=891 => Some(16),
        _ => None,
    }
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
///
/// Where the derived rate over the range is zero and the named ones are not,
/// the ratio is the channel's instead: its rate `channel_rate` plus what the
/// named reactions add through the derivation's `coefficient`, over
/// `channel_rate`.
fn derivation_mismatch(
    flux: &FluxDensity,
    reactions: &BTreeMap<i32, &Reaction>,
    d: &Derivation,
    channel_rate: f64,
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
    if (named - derived).abs() <= DERIVATION_ROUNDING * magnitude.max(derived.abs()) {
        return None;
    }
    if derived != 0.0 {
        return Some(named / derived);
    }
    // Named reactions carry rate over a range where the one they derive has
    // none, so there is no ratio to it. What they add to the channel's
    // partials is `coefficient Σ c_i R_MTi`, which is compared with the
    // channel's own rate instead; a channel with no rate has no covariance
    // to be wrong about.
    (channel_rate != 0.0).then(|| 1.0 + d.coefficient * named / channel_rate)
}

/// The pairs of reactions, lower MT first, that enter one channel with
/// opposite signs over a common cell of the flux group `boundaries`, each as a leaf there
/// (see [`leaf_coefficient`]) whose own blocks state a variance over it, and
/// with no consumed block between them.
///
/// ENDF-102 33.3.2 a.1 lets a tape leave a zero covariance between two cross
/// sections unstated, so an absent block reads as zero, and the fold reads
/// it so. Where the two enter with the same sign that is the usual
/// omission. Where they oppose it decides the answer: the two variances add
/// where a positive correlation would cancel them, and a derivation such as
/// FENDL-3.2d and TENDL-2017 H2 `(n,2n)` = `σ_1 - σ_2 - σ_102`, whose named
/// reactions contain the one derived, cannot have its parts uncorrelated in
/// fact. Such a sigma is the tape's literal statement, so it is reported
/// rather than altered.
///
/// Judged on the leaves, cell by cell, because a pair matters only where
/// both enter: a channel's own term runs over the whole axis, but where an
/// NC block derives it, it is replaced by the reactions the block names and
/// its own blocks state zero (33.3.3 item 3).
fn opposing_uncorrelated(
    terms: &[&Term],
    derivations: &[&Derivation],
    diagonals: &BTreeMap<i32, Vec<Diagonal>>,
    correlated: &BTreeSet<(i32, i32)>,
    boundaries: &[f64],
) -> BTreeSet<(i32, i32)> {
    let mts: BTreeSet<i32> = terms.iter().map(|t| t.mt).collect();
    let mut out = BTreeSet::new();
    for w in cell_edges(boundaries, terms, derivations, diagonals).windows(2) {
        // In MT order, so each pair comes lower MT first.
        let live: Vec<(i32, f64)> = mts
            .iter()
            .filter_map(|&mt| {
                let c = leaf_coefficient(terms, derivations, mt, w);
                (c != 0.0 && states_variance(diagonals, mt, w)).then_some((mt, c))
            })
            .collect();
        for (k, &(a, ca)) in live.iter().enumerate() {
            for &(b, cb) in &live[k + 1..] {
                if ca * cb < 0.0 && !correlated.contains(&(a, b)) {
                    out.insert((a, b));
                }
            }
        }
    }
    out
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
    let blocks = single_component_lumps(blocks);
    let blocks = blocks.as_ref();
    let sums = lump_cross_sections(blocks, reactions);
    let mut reactions = reactions.clone();
    reactions.extend(sums.iter().map(|r| (r.mt_number, r)));
    let reactions = &reactions;
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

    // A lumped reaction with several components states the covariance of
    // their sum and of none of them. Reported whether or not anything else
    // folds, since the channel it would have covered may have nothing else.
    for (mtl, components) in lumped_reactions(blocks) {
        if components.len() < 2 {
            continue;
        }
        let stated = blocks.iter().any(|b| {
            b.lumped_into().is_none()
                && (b.mt == mtl || (b.is_same_evaluation() && b.partner_mt() == mtl))
        });
        // A lump a derivation reaches is folded whole, as its components'
        // sum, and so covers the levels it holds. A component reached in its
        // own right still has no covariance of its own. A level sum whose own
        // section states its covariance explicitly loses nothing to a lump of
        // part of it. A derivation on the sum does not count: it may be the
        // very one that names the lump and could not be folded.
        let states_own = |mt: i32| {
            blocks
                .iter()
                .any(|b| b.is_diagonal() && b.mt == mt && matches!(b.data, CovarianceData::Ni(_)))
        };
        let reaches = components.iter().any(|&c| reached.contains(&c))
            || (!reached.contains(&mtl)
                && components.iter().any(|&c| {
                    level_sum(c).is_some_and(|s| reached.contains(&s) && !states_own(s))
                }));
        if stated && reaches {
            coverage.lumped_covariance_not_assignable.insert(
                (nuclide.to_string(), mtl),
                components.iter().map(|&c| name(c)).collect(),
            );
        }
    }

    // Absolute covariance, in (1/s)^2, before relativizing.
    let mut absolute = vec![0.0; n * n];
    let mut used = 0;
    // The channels a consumed block names, and per reached reaction the
    // variance each of its own blocks states. Kept block by block rather than
    // summed onto one grid, because the blocks need not share a grid;
    // `stated_variance_share` walks them together.
    let mut named: BTreeSet<usize> = BTreeSet::new();
    let mut diagonals: BTreeMap<i32, Vec<Diagonal>> = BTreeMap::new();
    // Per channel, the largest sum of partial rates any relative block of its
    // own reaction weighted it with, zero variance intervals included. The
    // consistency check against the rate, and not the share: it is what the
    // covariance's numerator was built from, whatever the block states.
    let mut weighted_rate: BTreeMap<usize, f64> = BTreeMap::new();
    // Per channel, the smallest such sum over the relative blocks of its own
    // reaction whose grid spans the whole flux range, the only ones whose
    // partials must add up to the rate rather than to at most it.
    let mut spanning_rate: BTreeMap<usize, f64> = BTreeMap::new();
    // The same two for a derived term, whose partials are another reaction's
    // over its range: as a ratio to that reaction's rate there, as the fold
    // integrates it, since that is the part of the channel's rate they stand
    // for. The ratio of the channel's own rate to the one it is divided by
    // is applied below.
    let mut derived_weighted: BTreeMap<usize, f64> = BTreeMap::new();
    let mut derived_spanning: BTreeMap<usize, f64> = BTreeMap::new();
    // The pairs of distinct reactions a consumed cross block correlates, lower
    // MT first, for the check that opposing terms of a derivation had one.
    let mut correlated: BTreeSet<(i32, i32)> = BTreeSet::new();
    let (flux_lo, flux_hi) = (
        flux.boundaries[0],
        flux.boundaries[flux.boundaries.len() - 1],
    );
    let spans = |grid: &[f64], range: (f64, f64)| match (grid.first(), grid.last()) {
        (Some(&a), Some(&b)) => a <= range.0.max(flux_lo) && b >= range.1.min(flux_hi),
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
        // checked below for this evaluation's blocks. Another evaluation's
        // partner is not: its MAT is known only if it has covariance blocks of
        // its own, so a partner in the run without any could not be told from
        // one that is absent.
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
            // A lumped reaction's component, with no covariance of its own.
            CovarianceData::Lumped => continue,
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
        // ENDF-102 defines `lb = 8` as a variance, so it belongs only on a
        // self-covariance. Its weights are square roots with no sign or
        // cross-section factor, and folding one between two reactions would
        // add `Σ Fk·w²` to their covariance with nothing to justify it.
        if expanded.scale == Scale::ShortRange && row_mt != col_mt {
            coverage.malformed += 1;
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

        // A short-range variance states nothing correlating two parts of one
        // interval (ENDF-102 section 33.2.2.2), so two terms of the reaction
        // over different ranges covary only over the part of each interval
        // both ranges hold: `Fk·ΔEk·∫ ψ² dE` over that overlap. Square-root
        // weights taken per range and multiplied would give the geometric
        // mean of the two ranges' integrals instead, so each pair of terms is
        // contracted against the weights of its overlap. A block that is not
        // a self-covariance was refused above.
        if expanded.scale == Scale::ShortRange {
            for t in row_terms {
                for u in col_terms {
                    let overlap = (t.range.0.max(u.range.0), t.range.1.min(u.range.1));
                    if overlap.0 >= overlap.1 {
                        continue;
                    }
                    let w = partial_rates_within(
                        flux,
                        row_rx,
                        &expanded.row_energies,
                        expanded.scale,
                        overlap,
                    );
                    absolute[t.channel * n + u.channel] +=
                        t.coefficient * u.coefficient * contract(&expanded, &w, &w);
                }
            }
        } else {
            for t in row_terms {
                let row = &rows[of(&rows, t.range)].1;
                for u in col_terms {
                    let col = &cols[of(&cols, u.range)].1;
                    let contribution =
                        t.coefficient * u.coefficient * contract(&expanded, row, col);
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
        }

        // Only a reaction's own blocks state its variance. A cross block
        // correlates two reactions and gives neither a variance, so it names
        // the channels it reaches and covers none.
        named.extend(row_terms.iter().chain(col_terms).map(|t| t.channel));
        if row_mt == col_mt {
            diagonals
                .entry(row_mt)
                .or_default()
                .push(Diagonal::of(&expanded));
        }
        if expanded.scale == Scale::Relative {
            for (ts, ps, grid) in [
                (row_terms, &rows, &expanded.row_energies),
                (col_terms, &cols, &expanded.col_energies),
            ] {
                for t in ts.iter() {
                    let sum = ps[of(ps, t.range)].1.total().abs();
                    let (weighted, spanning, sum) = if t.is_own(kinds) {
                        (&mut weighted_rate, &mut spanning_rate, sum)
                    } else {
                        let within = [t.range.0.max(flux_lo), t.range.1.min(flux_hi)];
                        if t.coefficient == 0.0 || within[0] >= within[1] {
                            continue;
                        }
                        let rate = BARN_TO_CM2 * flux.xs_over(reactions[&t.mt], &within)[0];
                        if rate == 0.0 {
                            continue;
                        }
                        (&mut derived_weighted, &mut derived_spanning, sum / rate)
                    };
                    let e = weighted.entry(t.channel).or_insert(0.0);
                    *e = e.max(sum);
                    if spans(grid, t.range) {
                        let e = spanning.entry(t.channel).or_insert(f64::INFINITY);
                        *e = e.min(sum);
                    }
                }
            }
        }
        if row_mt != col_mt {
            correlated.insert((row_mt.min(col_mt), row_mt.max(col_mt)));
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
        let Some(reaction) = reactions.get(mt).filter(|_| named.contains(&channel)) else {
            continue;
        };
        let key = (nuclide.to_string(), kind.clone());
        let own_rate = flux.xs_over(reaction, &[flux_lo, flux_hi])[0];
        // A derived channel's numerator is built from the reactions its NC
        // blocks name, so its check is that they add up to the reaction they
        // derive, over each block's range, nested derivations included.
        let mismatched: Vec<(&Derivation, f64)> = expansion
            .derivations
            .iter()
            .filter(|d| d.channel == channel)
            .filter_map(|d| Some((d, derivation_mismatch(flux, reactions, d, own_rate)?)))
            .collect();
        let terms: Vec<&Term> = expansion
            .terms
            .iter()
            .filter(|t| t.channel == channel)
            .collect();
        let derivations: Vec<&Derivation> = expansion
            .derivations
            .iter()
            .filter(|d| d.channel == channel)
            .collect();
        let opposing = opposing_uncorrelated(
            &terms,
            &derivations,
            &diagonals,
            &correlated,
            flux.boundaries,
        );
        if !opposing.is_empty() {
            coverage.derived_opposing_uncorrelated.insert(
                key.clone(),
                opposing
                    .into_iter()
                    .map(|(a, b)| (name(a), name(b)))
                    .collect(),
            );
        }
        if let Some(share) =
            stated_variance_share(flux, reaction, &terms, &derivations, &diagonals, reactions)
        {
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
        // A derived term's partials against its own reaction's rate, carried
        // over to the rate the covariance is divided by through the fold's
        // own rate for the channel. On an own term the same product is the
        // comparison above, so a tallied rate reads alike on both.
        if let Some(&full) = rates
            .get(kind)
            .filter(|&&full| full != 0.0 && own_rate != 0.0)
        {
            let to_full = BARN_TO_CM2 * own_rate / full;
            if let Some(&weighted) = derived_weighted.get(&channel) {
                if weighted * to_full > 1.0 + PARTIALS_ROUNDING {
                    ratios.push(weighted * to_full);
                }
            }
            if let Some(&spanning) = derived_spanning.get(&channel) {
                if spanning * to_full < 1.0 - PARTIALS_ROUNDING {
                    ratios.push(spanning * to_full);
                }
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
    // nothing and leaves nothing to argue about.
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

/// One spectrum of a schedule, as [`cell_fields`] reads it: the chain it was
/// collapsed with (which fixes each nuclide's channels and their order), its
/// unit-flux rates and its group fluxes.
pub struct FoldSpectrum<'a> {
    pub chain: &'a std::collections::HashMap<String, ChainNuclide>,
    pub rates: &'a ReactionRates,
    pub multigroup_flux: &'a [f64],
    pub group_boundaries: &'a [f64],
}

/// One cell of a nuclide's cross-section field: reaction `mt` over
/// `[lo, hi]` eV, an interval of the union of every covariance grid that
/// reaction appears on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    pub mt: i32,
    pub lo: f64,
    pub hi: f64,
}

/// One short-range (`lb = 8`) self-covariance block: a variance `F_k` on
/// each interval `[edges[k], edges[k + 1]]`, with nothing correlating two
/// averages inside one interval (ENDF-102 section 33.2.2.2).
#[derive(Debug, Clone, PartialEq)]
pub struct ShortRange {
    /// The block's identity within the evaluation, `(mt, subsection,
    /// block)`, which keys its noise so it does not move when other blocks
    /// come or go.
    pub key: (i32, i32, i32),
    pub mt: i32,
    pub edges: Vec<f64>,
    /// `F_k`, in barn^2.
    pub variance: Vec<f64>,
}

/// One channel's view of one short-range interval under one spectrum: the
/// interval's part inside the term's range and the flux range, cut at the
/// spectrum's group boundaries, and per piece the weight that turns the
/// noise's integral over the piece into a rate.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortTerm {
    /// Index into [`CellField::short`].
    pub block: usize,
    pub interval: usize,
    /// Ascending piece edges, in eV.
    pub cuts: Vec<f64>,
    /// Per piece, `c · 1e-24 · ψ_j`: the term's coefficient times the flux
    /// density flat over the piece, in cm^2 n/cm^2/s/eV per barn-eV.
    pub weight: Vec<f64>,
}

/// How one spectrum's rates read a nuclide's cross-section field.
///
/// A perturbed cross section `σ'(E)` changes channel `i`'s rate by exactly
///
/// ```text
/// R'_i - R_i = Σ_k p_ik (m_k - 1) + Σ_k q_ik a_k + Σ_terms Σ_j w_j ΔB_j
/// ```
///
/// with `m_k` the multiplier on relative cell `k`, `a_k` the shift in barns
/// on absolute cell `k`, and `ΔB_j` the short-range noise integrated over
/// piece `j`. Nothing in that is linearized: a reaction rate is linear in the
/// cross section, and every perturbation here is constant on its cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    /// The spectrum's channels for this nuclide, in the fold's order.
    pub kinds: Vec<String>,
    /// The full rate of each channel, the denominator the fold divides by.
    pub rates: Vec<f64>,
    /// Row-major `kinds × relative_cells`: `p_ik`, the term-weighted partial
    /// rate of channel `i` over cell `k`, in 1/s.
    pub relative: Vec<f64>,
    /// Row-major `kinds × absolute_cells`: `q_ik`, the term-weighted partial
    /// flux times 1e-24, in cm^2/barn n/cm^2/s.
    pub absolute: Vec<f64>,
    /// Per channel, its short-range terms.
    pub short: Vec<Vec<ShortTerm>>,
}

/// One nuclide's cross-section uncertainty as a field over its covariance
/// cells, independent of any spectrum, with how each spectrum reads it.
///
/// The fold contracts each block against one spectrum's partials. This keeps
/// the blocks uncontracted instead, every block refined onto the union of its
/// reactions' grids, which is exact because MF=33 covariance is constant on
/// each interval of its own grid. A draw of the field is then one draw of the
/// cross sections themselves, and every spectrum, step and material reads
/// its rates off the same draw. Within a spectrum the rates' covariance is
/// [`fold_rate_covariance`]'s, and between two spectra it is the same
/// contraction with each side's own partials.
///
/// The cells are the union over every reaction the full chain's channels
/// reach through this evaluation, not only those one spectrum's pruned chain
/// reaches, so the field, and with it a replica's draw, is the same whichever
/// spectra and materials read it. Blocks are consumed under exactly
/// [`fold_nuclide`]'s rules.
#[derive(Debug, Clone, PartialEq)]
pub struct CellField {
    pub relative_cells: Vec<Cell>,
    /// Row-major relative covariance over `relative_cells`.
    pub relative: Vec<f64>,
    pub absolute_cells: Vec<Cell>,
    /// Row-major absolute covariance over `absolute_cells`, in barn^2.
    pub absolute: Vec<f64>,
    pub short: Vec<ShortRange>,
    /// Per spectrum, in order, or `None` where that spectrum has no rates or
    /// no channels for this nuclide.
    pub projections: Vec<Option<Projection>>,
}

/// Every nuclide's [`CellField`], for the spectra of one schedule.
///
/// `chain` is the full chain, whose channels fix the cells; each spectrum's
/// own (possibly pruned) chain fixes only which channels it reads. A nuclide
/// with no covariance data, or none the channels reach, is left out, as the
/// fold leaves it out.
pub fn cell_fields(
    material: &Material,
    chain: &std::collections::HashMap<String, ChainNuclide>,
    spectra: &[FoldSpectrum],
    shielding: Option<&Shielding>,
) -> BTreeMap<String, CellField> {
    let shapes: Vec<Option<CollapseShapes>> = spectra
        .iter()
        .map(|s| {
            let valid = !s.multigroup_flux.is_empty()
                && s.group_boundaries.len() == s.multigroup_flux.len() + 1;
            if valid {
                CollapseShapes::new(material, s.multigroup_flux, s.group_boundaries, shielding)
            } else {
                None
            }
        })
        .collect();
    let names: BTreeSet<&String> = spectra.iter().flat_map(|s| s.rates.keys()).collect();
    let names: Vec<&String> = names.into_iter().collect();

    let one = |name: &&String| -> Option<(String, CellField)> {
        let full = chain.get(*name)?;
        let nuclide_data = material.nuclide_data.get(*name)?;
        let blocks = nuclide_data.covariance.as_ref()?;
        let temperature = if material.temperature().is_empty() {
            crate::default_temperature(nuclide_data)?
        } else {
            material.temperature().to_string()
        };
        let by_mt = nuclide_data.reactions_for_temp(&temperature)?;
        let reactions: BTreeMap<i32, &Reaction> =
            by_mt.iter().map(|(mt, r)| (*mt, r.as_ref())).collect();
        let field = nuclide_field(
            name,
            &kinds_and_mts(full),
            blocks,
            &reactions,
            spectra,
            &shapes,
        )?;
        Some(((*name).clone(), field))
    };

    let fields: Vec<Option<(String, CellField)>> = {
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
    fields.into_iter().flatten().collect()
}

/// Where one transport reaction's perturbation comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Read {
    /// The evaluation states covariance for this reaction itself, directly or
    /// through an NC derivation from other reactions
    /// ([`TransportField::derived`]).
    Own,
    /// The reaction has no covariance of its own, but the summed reaction it
    /// is a component of does, and it takes that one's perturbation: MT 51 to
    /// 91 from MT 4, 600 to 649 from MT 103, and so on. The components then
    /// move together, fully correlated, which is how a covariance stated only
    /// on the sum is applied to the reactions transport samples (SANDY does
    /// the same). This adds no uncertainty the evaluation does not state: the
    /// sum moves exactly as its covariance says.
    Parent(i32),
    /// No covariance transport can apply reaches it; it is held at nominal.
    Nominal,
}

/// One term an NC derivation adds to a transport reaction's perturbation.
///
/// Over `range`, an LTY=0 block states `σ_MT = Σ c_i σ_MTi` (ENDF-102
/// 33.2.2.1), so a draw that moves each `σ_MTi` by `δσ_MTi` moves `σ_MT` by
/// `Σ c_i δσ_MTi`. This is one such `c_i δσ_MTi`, expanded as the fold
/// expands it ([`channel_terms`]): a named reaction derived in its turn
/// contributes its own named reactions, with the coefficients multiplied and
/// the ranges intersected. ENDF/B-VIII.1 Pb208 elastic above 1.5 MeV is
/// `σ_1 - σ_4 - σ_16 - σ_102`, and has no cells of its own there.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedTerm {
    /// The product of the coefficients on the path that reached `mt`.
    pub coefficient: f64,
    /// The reaction whose cells the term reads.
    pub mt: i32,
    /// The reactions whose cross sections add up to `mt`'s: `mt` itself, or
    /// the components of a lumped reaction, which has none of its own
    /// (ENDF-102 33.2.3).
    pub cross_sections: Vec<i32>,
    /// `[lo, hi)` in eV, where the derivation holds.
    pub range: (f64, f64),
}

/// One nuclide's cross-section field for transport, and how each reaction
/// transport samples reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct TransportField {
    /// Every non-redundant reaction the nuclide's transport data holds, the
    /// partials that make up its total, and where each one's perturbation
    /// comes from.
    pub reads: BTreeMap<i32, Read>,
    /// Per [`Read::Own`] reaction whose covariance an NC block derives, the
    /// terms the derivation adds to what its own cells give it. A reaction
    /// absent here reads its own cells only. Only terms whose reaction has
    /// cells over their range are kept, since the others move nothing.
    pub derived: BTreeMap<i32, Vec<DerivedTerm>>,
    /// The field over the covariance cells those reactions reach, or `None`
    /// when none does. It has no spectrum projections: transport reads it at
    /// each event's energy rather than through a group spectrum.
    pub field: Option<CellField>,
}

/// Every nuclide's [`TransportField`] for one material, and the nuclides with
/// no covariance data at all.
///
/// Transport perturbs the partial reactions it samples, the non-redundant
/// ones, and rebuilds every total from them, so the field is built over
/// exactly those. A redundant reaction's own covariance (MT 1, MT 4, MT 103)
/// is used only through [`Read::Parent`], for components that state none,
/// and through [`TransportField::derived`], for partials an NC block derives
/// from it.
/// The material's nuclides must be loaded with covariance (see
/// `Material::ensure_covariance_loaded`); a nuclide whose data carries none
/// is named in the returned set rather than reported as exact.
pub fn transport_fields(
    material: &Material,
) -> (BTreeMap<String, TransportField>, BTreeSet<String>) {
    let mut names: Vec<&String> = material.nuclides.keys().collect();
    names.sort();
    let mut without_data = BTreeSet::new();
    let mut out = BTreeMap::new();
    for name in names {
        let Some(nuclide_data) = material.nuclide_data.get(name) else {
            continue;
        };
        let temperature = if material.temperature().is_empty() {
            match crate::default_temperature(nuclide_data) {
                Some(t) => t,
                None => continue,
            }
        } else {
            material.temperature().to_string()
        };
        let Some(by_mt) = nuclide_data.reactions_for_temp(&temperature) else {
            continue;
        };
        let partials: Vec<i32> = by_mt
            .iter()
            .filter(|(_, r)| !r.redundant)
            .map(|(mt, _)| *mt)
            .collect::<BTreeSet<i32>>()
            .into_iter()
            .collect();
        let Some(blocks) = nuclide_data.covariance.as_ref() else {
            without_data.insert(name.clone());
            continue;
        };
        let reactions: BTreeMap<i32, &Reaction> =
            by_mt.iter().map(|(mt, r)| (*mt, r.as_ref())).collect();
        out.insert(
            name.clone(),
            transport_field(name, &partials, blocks, &reactions),
        );
    }
    (out, without_data)
}

/// Whether `field` has a relative or absolute cell of reaction `mt` that
/// overlaps `range`: something a transport draw moves `mt` by there.
///
/// Short-range (`lb = 8`) blocks do not count. Transport holds their noise at
/// nominal, as it averages away along a track, so a reaction they alone
/// cover is not perturbed by any replica.
fn moves(field: &Option<CellField>, mt: i32, range: (f64, f64)) -> bool {
    field.as_ref().is_some_and(|f| {
        f.relative_cells
            .iter()
            .chain(&f.absolute_cells)
            .any(|c| c.mt == mt && c.lo < range.1 && c.hi > range.0)
    })
}

/// [`transport_fields`] for one nuclide.
fn transport_field(
    name: &str,
    partials: &[i32],
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
) -> TransportField {
    let lumped = single_component_lumps(blocks);
    let blocks_ref = lumped.as_ref();
    let sums = lump_cross_sections(blocks_ref, reactions);
    let mut all = reactions.clone();
    all.extend(sums.iter().map(|r| (r.mt_number, r)));
    let components = lumped_reactions(blocks_ref);

    // The reactions some usable block states a covariance for.
    let stated: BTreeSet<i32> = blocks_ref
        .iter()
        .filter(|b| !b.is_cross_material() && b.names_cross_section())
        .filter(|b| match &b.data {
            CovarianceData::Ni(ni) => expand_ni(ni).is_ok_and(|e| !e.is_empty()),
            _ => false,
        })
        .flat_map(|b| [b.mt, b.partner_mt()])
        .collect();
    // A reaction's terms: itself everywhere, and what the NC blocks on it
    // derive it from, exactly as the fold expands a channel.
    let terms_of = |mt: i32| channel_terms(blocks_ref, &all, &[(String::new(), mt)]).terms;
    let reaches = |mt: i32| terms_of(mt).iter().any(|t| stated.contains(&t.mt));

    let mut reads = BTreeMap::new();
    for &mt in partials {
        let read = if reaches(mt) {
            Read::Own
        } else {
            match level_sum(mt) {
                Some(sum) if all.contains_key(&sum) && reaches(sum) => Read::Parent(sum),
                _ => Read::Nominal,
            }
        };
        reads.insert(mt, read);
    }
    let build = |reads: &BTreeMap<i32, Read>| {
        let mut reach: Vec<(String, i32)> = reads
            .iter()
            .filter_map(|(mt, r)| match r {
                Read::Own => Some((format!("MT{mt}"), *mt)),
                Read::Parent(sum) => Some((format!("MT{mt}"), *sum)),
                Read::Nominal => None,
            })
            .collect();
        // A parent read by several components is one set of cells.
        reach.sort_by_key(|(_, mt)| *mt);
        reach.dedup_by_key(|(_, mt)| *mt);
        if reach.is_empty() {
            None
        } else {
            nuclide_field(name, &reach, blocks, reactions, &[], &[])
        }
    };
    let mut field = build(&reads);

    // A read is kept only if a draw moves it, so nothing is reported as
    // perturbed that every replica holds at nominal. A sum lends its
    // perturbation only if it has cells of its own: one whose covariance is
    // an NC derivation from its components (ENDF/B-VIII.1 Cr52 MT 4 from its
    // levels) states nothing beyond what those components state, so a
    // component stating nothing takes nothing from it; nor does a sum whose
    // only block is all zeros, or only short-range. A reaction read as its
    // own needs cells of its own, or of a reaction it is derived from over
    // the range of the derivation. Dropping a read can drop the cells of a
    // reaction only it reached, so this repeats until nothing changes.
    loop {
        let demoted: Vec<i32> = reads
            .iter()
            .filter_map(|(mt, r)| {
                let kept = match r {
                    Read::Own => terms_of(*mt).iter().any(|t| moves(&field, t.mt, t.range)),
                    Read::Parent(sum) => moves(&field, *sum, EVERYWHERE),
                    Read::Nominal => true,
                };
                (!kept).then_some(*mt)
            })
            .collect();
        if demoted.is_empty() {
            break;
        }
        for mt in demoted {
            reads.insert(mt, Read::Nominal);
        }
        field = build(&reads);
    }

    let derived = reads
        .iter()
        .filter(|(_, r)| **r == Read::Own)
        .filter_map(|(&mt, _)| {
            let kinds = [(String::new(), mt)];
            let terms: Vec<DerivedTerm> = terms_of(mt)
                .into_iter()
                .filter(|t| !t.is_own(&kinds))
                .filter(|t| moves(&field, t.mt, t.range))
                .map(|t| DerivedTerm {
                    coefficient: t.coefficient,
                    mt: t.mt,
                    cross_sections: match components.get(&t.mt) {
                        Some(c) if !reactions.contains_key(&t.mt) => c.iter().copied().collect(),
                        _ => vec![t.mt],
                    },
                    range: t.range,
                })
                .collect();
            (!terms.is_empty()).then_some((mt, terms))
        })
        .collect();
    TransportField {
        reads,
        derived,
        field,
    }
}

/// [`cell_fields`] for one nuclide.
#[allow(clippy::too_many_arguments)]
///
/// `reach` names the channels whose reactions fix the cells, as `(kind, MT)`:
/// the full chain's for activation, the transport reactions for transport.
fn nuclide_field(
    name: &str,
    reach: &[(String, i32)],
    blocks: &[CovarianceBlock],
    reactions: &BTreeMap<i32, &Reaction>,
    spectra: &[FoldSpectrum],
    shapes: &[Option<CollapseShapes>],
) -> Option<CellField> {
    let blocks = single_component_lumps(blocks);
    let blocks = blocks.as_ref();
    let sums = lump_cross_sections(blocks, reactions);
    let mut reactions = reactions.clone();
    reactions.extend(sums.iter().map(|r| (r.mt_number, r)));
    let reactions = &reactions;

    // The cells come from `reach`, not from any spectrum's channels, so they do
    // not depend on which spectra read them.
    let reached: BTreeSet<i32> = channel_terms(blocks, reactions, reach)
        .terms
        .iter()
        .map(|t| t.mt)
        .collect();
    if reached.is_empty() {
        return None;
    }
    let mut skipped_copy: BTreeSet<(i32, i32)> = BTreeSet::new();
    for (lo, hi) in mirrored_pairs(blocks, &reached) {
        if orientation_expands(blocks, lo, hi) || !orientation_expands(blocks, hi, lo) {
            skipped_copy.insert((hi, lo));
        } else {
            skipped_copy.insert((lo, hi));
        }
    }

    // The blocks the fold consumes, by its rules.
    let mut consumed: Vec<(&CovarianceBlock, ExpandedBlock)> = Vec::new();
    for block in blocks {
        if block.is_cross_material() || !block.names_cross_section() {
            continue;
        }
        let CovarianceData::Ni(ni) = &block.data else {
            continue;
        };
        let (row_mt, col_mt) = (block.mt, block.partner_mt());
        if !reached.contains(&row_mt)
            || !reached.contains(&col_mt)
            || skipped_copy.contains(&(row_mt, col_mt))
        {
            continue;
        }
        let Ok(expanded) = expand_ni(ni) else {
            continue;
        };
        if expanded.is_empty() || (expanded.scale == Scale::ShortRange && row_mt != col_mt) {
            continue;
        }
        consumed.push((block, expanded));
    }
    if consumed.is_empty() {
        return None;
    }

    // Per (reaction, scale), the union of every grid it appears on.
    let mut edges: BTreeMap<(i32, bool), Vec<f64>> = BTreeMap::new();
    let mut short: Vec<ShortRange> = Vec::new();
    for (block, expanded) in &consumed {
        let absolute = match expanded.scale {
            Scale::Relative => false,
            Scale::Absolute => true,
            Scale::ShortRange => {
                short.push(ShortRange {
                    key: block_key(block),
                    mt: block.mt,
                    edges: expanded.row_energies.clone(),
                    variance: (0..expanded.n_rows()).map(|k| expanded.get(k, k)).collect(),
                });
                continue;
            }
        };
        let (row_mt, col_mt) = (block.mt, block.partner_mt());
        edges
            .entry((row_mt, absolute))
            .or_default()
            .extend_from_slice(&expanded.row_energies);
        edges
            .entry((col_mt, absolute))
            .or_default()
            .extend_from_slice(&expanded.col_energies);
    }
    let mut grids: BTreeMap<(i32, bool), Vec<f64>> = BTreeMap::new();
    for (key, mut e) in edges {
        e.sort_by(f64::total_cmp);
        e.dedup();
        if e.len() >= 2 {
            grids.insert(key, e);
        }
    }

    // Cells per scale, in (mt, energy) order, and each grid's offset.
    let mut cells: [Vec<Cell>; 2] = [Vec::new(), Vec::new()];
    let mut offset: BTreeMap<(i32, bool), usize> = BTreeMap::new();
    for (&(mt, absolute), grid) in &grids {
        let list = &mut cells[usize::from(absolute)];
        offset.insert((mt, absolute), list.len());
        list.extend(grid.windows(2).map(|w| Cell {
            mt,
            lo: w[0],
            hi: w[1],
        }));
    }

    // Each block refined onto the cells: a cell inside interval `i` of the
    // block's row grid and one inside interval `j` of its column grid covary
    // by the block's (i, j). A block between two reactions fills both
    // orientations, as the fold's `rᵀ C r'` is both covariances.
    let mut covariance: [Vec<f64>; 2] = [
        vec![0.0; cells[0].len() * cells[0].len()],
        vec![0.0; cells[1].len() * cells[1].len()],
    ];
    for (block, expanded) in &consumed {
        let absolute = match expanded.scale {
            Scale::Relative => false,
            Scale::Absolute => true,
            Scale::ShortRange => continue,
        };
        let s = usize::from(absolute);
        let n = cells[s].len();
        let (row_mt, col_mt) = (block.mt, block.partner_mt());
        let (row_grid, col_grid) = (&grids[&(row_mt, absolute)], &grids[&(col_mt, absolute)]);
        let (row_off, col_off) = (offset[&(row_mt, absolute)], offset[&(col_mt, absolute)]);
        let row_of: Vec<Option<usize>> = row_grid
            .windows(2)
            .map(|w| interval_holding(&expanded.row_energies, w[0], w[1]))
            .collect();
        let col_of: Vec<Option<usize>> = col_grid
            .windows(2)
            .map(|w| interval_holding(&expanded.col_energies, w[0], w[1]))
            .collect();
        for (k, i) in row_of.iter().enumerate() {
            let Some(i) = *i else { continue };
            if i >= expanded.n_rows() {
                continue;
            }
            for (l, j) in col_of.iter().enumerate() {
                let Some(j) = *j else { continue };
                if j >= expanded.n_cols() {
                    continue;
                }
                let v = expanded.get(i, j);
                if v == 0.0 {
                    continue;
                }
                let (a, b) = (row_off + k, col_off + l);
                covariance[s][a * n + b] += v;
                if row_mt != col_mt {
                    covariance[s][b * n + a] += v;
                }
            }
        }
    }

    // A cell no block states anything on carries no uncertainty, and leaving
    // it in would only widen the factorization. Two adjacent cells of one
    // reaction whose rows are identical are one random variable (correlation
    // one, equal variance): the union grid split them only because some
    // other block has an edge there that no block of theirs distinguishes.
    // They are merged, which is exact, since their partials then multiply the
    // same draw; on an evaluation whose channels derive from many partials,
    // such as ENDF/B-VIII.1 O16, it shrinks the factorization several-fold.
    let mut groups: [Vec<Vec<usize>>; 2] = [Vec::new(), Vec::new()];
    for s in 0..2 {
        let n = cells[s].len();
        let row = |a: usize| &covariance[s][a * n..(a + 1) * n];
        for a in (0..n).filter(|&a| row(a).iter().any(|v| *v != 0.0)) {
            let joins = groups[s].last().is_some_and(|g: &Vec<usize>| {
                let prev = *g.last().expect("a group is never empty");
                cells[s][prev].mt == cells[s][a].mt
                    && cells[s][prev].hi == cells[s][a].lo
                    && row(g[0]) == row(a)
            });
            if joins {
                groups[s].last_mut().expect("checked").push(a);
            } else {
                groups[s].push(vec![a]);
            }
        }
    }
    let index: [BTreeMap<usize, usize>; 2] = [0, 1].map(|s| {
        groups[s]
            .iter()
            .enumerate()
            .flat_map(|(new, g)| g.iter().map(move |&old| (old, new)))
            .collect()
    });
    let compact = |s: usize| -> (Vec<Cell>, Vec<f64>) {
        let n = cells[s].len();
        let m = groups[s].len();
        let mut c = vec![0.0; m * m];
        for (a, ga) in groups[s].iter().enumerate() {
            for (b, gb) in groups[s].iter().enumerate() {
                c[a * m + b] = covariance[s][ga[0] * n + gb[0]];
            }
        }
        let merged = groups[s]
            .iter()
            .map(|g| Cell {
                mt: cells[s][g[0]].mt,
                lo: cells[s][g[0]].lo,
                hi: cells[s][*g.last().expect("a group is never empty")].hi,
            })
            .collect();
        (merged, c)
    };
    // A nuclide whose blocks state nothing but zeros keeps a field with no
    // cells, as the fold keeps its zero matrix: it was covered, and every
    // replica reads it at nominal, which is not the same as having no data.
    let (relative_cells, relative) = compact(0);
    let (absolute_cells, absolute) = compact(1);

    let projections = spectra
        .iter()
        .zip(shapes)
        .map(|(spectrum, shapes)| {
            let kinds = kinds_and_mts(spectrum.chain.get(name)?);
            let nuclide_rates = spectrum.rates.get(name)?;
            if kinds.is_empty()
                || spectrum.multigroup_flux.is_empty()
                || spectrum.group_boundaries.len() != spectrum.multigroup_flux.len() + 1
            {
                return None;
            }
            let shape = shapes.as_ref().and_then(|s| s.shape_for(name));
            let flux = FluxDensity {
                boundaries: spectrum.group_boundaries,
                flux: spectrum.multigroup_flux,
                shape: shape.as_ref(),
            };
            let (flux_lo, flux_hi) = (
                spectrum.group_boundaries[0],
                spectrum.group_boundaries[spectrum.group_boundaries.len() - 1],
            );
            let n = kinds.len();
            let (nr, na) = (relative_cells.len(), absolute_cells.len());
            let mut projection = Projection {
                kinds: kinds.iter().map(|(k, _)| k.clone()).collect(),
                rates: kinds
                    .iter()
                    .map(|(k, _)| nuclide_rates.get(k).copied().unwrap_or(0.0))
                    .collect(),
                relative: vec![0.0; n * nr],
                absolute: vec![0.0; n * na],
                short: vec![Vec::new(); n],
            };
            for t in channel_terms(blocks, reactions, &kinds).terms {
                let rx = reactions[&t.mt];
                for (s, scale, out, width) in [
                    (0usize, Scale::Relative, &mut projection.relative, nr),
                    (1usize, Scale::Absolute, &mut projection.absolute, na),
                ] {
                    let absolute = s == 1;
                    let Some(grid) = grids.get(&(t.mt, absolute)) else {
                        continue;
                    };
                    let base = offset[&(t.mt, absolute)];
                    let partials = partial_rates_within(&flux, rx, grid, scale, t.range);
                    for (k, p) in partials.per_interval.iter().enumerate() {
                        if let Some(&cell) = index[s].get(&(base + k)) {
                            out[t.channel * width + cell] += t.coefficient * p;
                        }
                    }
                }
                for (b, block) in short.iter().enumerate() {
                    if block.mt != t.mt {
                        continue;
                    }
                    for (k, w) in block.edges.windows(2).enumerate() {
                        if k >= block.variance.len() || block.variance[k] == 0.0 {
                            continue;
                        }
                        let lo = w[0].max(t.range.0).max(flux_lo);
                        let hi = w[1].min(t.range.1).min(flux_hi);
                        if lo >= hi {
                            continue;
                        }
                        let mut cuts = vec![lo];
                        cuts.extend(
                            spectrum
                                .group_boundaries
                                .iter()
                                .copied()
                                .filter(|&e| e > lo && e < hi),
                        );
                        cuts.push(hi);
                        let weight = flux
                            .flux_over(rx, &cuts)
                            .iter()
                            .zip(cuts.windows(2))
                            .map(|(phi, c)| t.coefficient * BARN_TO_CM2 * phi / (c[1] - c[0]))
                            .collect();
                        projection.short[t.channel].push(ShortTerm {
                            block: b,
                            interval: k,
                            cuts,
                            weight,
                        });
                    }
                }
            }
            Some(projection)
        })
        .collect();

    Some(CellField {
        relative_cells,
        relative,
        absolute_cells,
        absolute,
        short,
        projections,
    })
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
        assert_eq!(a.rate_fraction_total(), Ok(Some(0.375)));
    }

    /// A decay-only schedule drove no production, so there is no share to
    /// report. Zero would be the wrong answer: it reads as "nothing is
    /// covered", which is a statement about the data rather than about there
    /// being nothing to cover.
    #[test]
    fn no_production_reports_no_fraction_rather_than_zero() {
        assert_eq!(Coverage::default().rate_fraction_total(), Ok(None));
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
        assert_eq!(c.rate_fraction_total(), Ok(Some(1.0)));
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
            products: vec![].into(),
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

    /// `lb = 1` or `lb = 8`: one variance per interval. Written as the tape
    /// writes it, `np` (E, F) pairs with the last `F` a zero past the top
    /// edge, which is what an `lb = 1` block is checked against.
    fn diagonal(lb: i64, energies: &[f64], variances: &[f64]) -> NiSubsection {
        assert_eq!(variances.len() + 1, energies.len());
        NiSubsection {
            lb,
            np: energies.len() as i64,
            ek: energies.to_vec(),
            fk: variances.iter().copied().chain([0.0]).collect(),
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
        let own = Term {
            channel: 0,
            coefficient: 1.0,
            mt: 16,
            range: EVERYWHERE,
        };
        assert_eq!(
            stated_variance_share(
                &flux(),
                &above,
                &[&own],
                &[],
                &BTreeMap::from([(16, vec![d])]),
                &BTreeMap::from([(16, &above)]),
            ),
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
    /// cancel exactly. The short-range block covers where its `Fk` is
    /// nonzero like any other.
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
        // Both are relative: a short-range value is summed apart, so the same
        // -0.02 stated as LB=8 would not cancel it.
        let cancelling = [
            block(102, 102, diagonal(1, &[1.0, 1.0e3, 1.0e5], &[0.01, 0.02])),
            block(102, 102, diagonal(1, &[1.0e3, 1.0e5], &[-0.02])),
        ];
        let c = fold(&cancelling, 1.0);
        let expected = rate(1.0, 1.0e3) / rate(1.0e-5, 2.0e7);
        let got = fraction(&c, "(n,gamma)");
        assert!(
            (got - expected).abs() <= 1.0e-12 * expected,
            "{got} against {expected}"
        );

        let apart = [
            block(102, 102, diagonal(1, &[1.0, 1.0e3, 1.0e5], &[0.01, 0.02])),
            block(102, 102, diagonal(8, &[1.0e3, 1.0e5], &[-0.02])),
        ];
        let c = fold(&apart, 1.0);
        let expected = rate(1.0, 1.0e5) / rate(1.0e-5, 2.0e7);
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
            products: vec![].into(),
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

    /// An `lb = 8` variance `Fk` on `[0, 10]`, folded against a flux that is
    /// not flat inside it: 1 n/cm^2/s in each of `[0, 2]` and `[2, 10]`. ENDF-102
    /// gives each group's average `Fk·10/2` and `Fk·10/8`, uncorrelated, so
    /// the rate variance is `Fk·10/2·1² + Fk·10/8·1² = 6.25·Fk` (times the
    /// barn factor twice). The absolute diagonal would say `Fk·2² = 4·Fk`.
    #[test]
    fn a_short_range_variance_scales_with_the_width_of_each_flux_group() {
        let boundaries = vec![0.0, 2.0, 10.0];
        let flux = vec![1.0, 1.0];
        let density = FluxDensity {
            boundaries: &boundaries,
            flux: &flux,
            shape: None,
        };
        let fk = 0.3;
        let block = ExpandedBlock {
            row_energies: vec![0.0, 10.0],
            col_energies: vec![0.0, 10.0],
            values: vec![fk],
            scale: Scale::ShortRange,
        };
        let reaction = ramp(0.0, 10.0, 1.0, 1.0);
        let w = partial_rates(&density, &reaction, &block.row_energies, block.scale);
        let got = contract(&block, &w, &w) / (BARN_TO_CM2 * BARN_TO_CM2);
        assert!((got - 6.25 * fk).abs() <= 1e-12, "{got}");
    }

    /// An `lb = 8` block between two reactions is refused rather than folded
    /// with square-root weights, while the same block on a self-covariance
    /// folds. The refused one names no channel in `rate_fraction_covered`,
    /// and the folded one covers its channel wherever its `Fk` is nonzero,
    /// here the whole flux range.
    #[test]
    fn a_short_range_block_between_two_reactions_is_malformed() {
        use endf::mf::covariance::NiSubsection;

        let boundaries = vec![0.0, 10.0];
        let flux = vec![1.0];
        let density = FluxDensity {
            boundaries: &boundaries,
            flux: &flux,
            shape: None,
        };
        let capture = ramp(0.0, 10.0, 1.0, 1.0);
        let mut proton = ramp(0.0, 10.0, 1.0, 1.0);
        proton.mt_number = 103;
        let reactions = BTreeMap::from([(102, &capture), (103, &proton)]);
        let kinds = vec![("(n,gamma)".to_string(), 102), ("(n,p)".to_string(), 103)];
        let full = BARN_TO_CM2 * density.integrate_xs(&capture, 0.0, 10.0);
        let rates = BTreeMap::from([("(n,gamma)".to_string(), full), ("(n,p)".to_string(), full)]);
        let block = |mt1: i32| CovarianceBlock {
            mt: 102,
            subsection_idx: 0,
            block_idx: 0,
            mat: 0,
            mat1: 0,
            mt1,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            data: CovarianceData::Ni(NiSubsection {
                lb: 8,
                np: 2,
                nt: 4,
                ek: vec![0.0, 10.0],
                fk: vec![0.01, 0.0],
                ..NiSubsection::default()
            }),
        };

        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &density,
            &[block(103)],
            &reactions,
            &kinds,
            &rates,
            "X1",
            &mut coverage,
        );
        assert!(folded.is_none());
        assert_eq!(coverage.malformed, 1);
        assert!(coverage.rate_fraction_covered.is_empty());

        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &density,
            &[block(102)],
            &reactions,
            &kinds,
            &rates,
            "X1",
            &mut coverage,
        );
        let folded = folded.expect("a self-covariance LB=8 block folds");
        assert_eq!(coverage.malformed, 0);
        assert!(folded.relative[0] > 0.0, "{:?}", folded.relative);
        assert_eq!(
            coverage.rate_fraction_covered,
            BTreeMap::from([(("X1".to_string(), "(n,gamma)".to_string()), 1.0)])
        );
        assert!(coverage.partials_above_rate.is_empty());
        assert!(coverage.partials_below_rate.is_empty());
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
            products: vec![].into(),
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

    /// A short-range block on a shielded run weights each flux-group piece of
    /// its interval with that piece's shielded flux, split from the collapse's
    /// term as an absolute block's is. Over `[1 eV, 20 MeV]` the pieces are
    /// the middle group's part above 1 eV, shared by the parts' shielded
    /// `∫ φ dE`, and the whole top group, which passes through as its flux.
    /// A flat shape gives the dilute weight back.
    #[test]
    fn a_short_range_block_weights_with_the_shielded_flux_of_each_piece() {
        let (shape, xs) = (shape(), capture());
        let flux = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: Some(&shape),
        };
        let grid = [1.0, 2.0e7];
        let got = partial_rates(&flux, &xs, &grid, Scale::ShortRange).per_interval;
        let walk = |a: f64, b: f64| walk_group(&xs, a, b, Some(&shape), None).shielded_weight();
        let (below, above) = (walk(BOUNDARIES[1], 1.0), walk(1.0, BOUNDARIES[2]));
        let middle = FLUX[1] * above / (below + above);
        let squared =
            middle * middle / (BOUNDARIES[2] - 1.0) + FLUX[2] * FLUX[2] / (2.0e7 - BOUNDARIES[2]);
        let want = BARN_TO_CM2 * ((2.0e7 - 1.0) * squared).sqrt();
        close(&got, &[want]);

        let flat = FluxShape::from_points(vec![2.0e7, 1.0e-5], vec![1.0, 1.0]);
        let shielded = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: Some(&flat),
        };
        let dilute = FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
        };
        close(
            &partial_rates(&shielded, &xs, &grid, Scale::ShortRange).per_interval,
            &partial_rates(&dilute, &xs, &grid, Scale::ShortRange).per_interval,
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
                np: GRID.len() as i64,
                ek: GRID.to_vec(),
                fk: variances.iter().copied().chain([0.0]).collect(),
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
            products: vec![].into(),
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
            products: vec![].into(),
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
        fold_against(blocks, channels, present, &rxs, 1.0)
    }

    /// [`fold`] with the cross sections `rxs`, one per MT in `present`, and
    /// the rates it divides by scaled by `rate_scale`, as a rate computed
    /// some other way would be.
    fn fold_against(
        blocks: &[CovarianceBlock],
        channels: &[(&str, i32)],
        present: &[i32],
        rxs: &[Reaction],
        rate_scale: f64,
    ) -> (Option<RateCovariance>, Coverage) {
        let reactions: BTreeMap<i32, &Reaction> = present.iter().copied().zip(rxs).collect();
        let kinds: Vec<(String, i32)> = channels
            .iter()
            .map(|(k, mt)| (k.to_string(), *mt))
            .collect();
        let rates: BTreeMap<String, f64> = kinds
            .iter()
            .map(|(k, mt)| {
                let full =
                    BARN_TO_CM2 * flux().integrate_xs(reactions[mt], BOUNDARIES[0], BOUNDARIES[4]);
                (k.clone(), rate_scale * full)
            })
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

    /// Opposing terms with no block between them fold as uncorrelated, which
    /// is the tape's statement, and the channel is reported as resting on
    /// that: the variances add, with nothing to cancel them.
    #[test]
    fn opposing_terms_without_a_cross_block_are_reported() {
        let terms = [(1.0, 601), (-1.0, 600)];
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &terms),
            own(600, C600),
            own(601, C601),
        ];
        let sums: Sums = &[(NP, &terms)];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        let (r0, r1) = (
            partials(600, sums, 0.0, f64::INFINITY),
            partials(601, sums, 0.0, f64::INFINITY),
        );
        let absolute = sandwich(r0, symmetric(C600), r0) + sandwich(r1, symmetric(C601), r1);
        close(
            folded.expect("derived").get(0, 0),
            absolute / rate(NP, sums).powi(2),
        );
        let key = ("O16".to_string(), "(n,p)".to_string());
        assert_eq!(
            coverage.derived_opposing_uncorrelated[&key],
            [("MT600".to_string(), "MT601".to_string())].into()
        );
        assert!(coverage.has_gaps());
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

    /// A derived channel divided by a rate computed some other way, such as
    /// a tallied one, is reported like an own channel would be: the named
    /// reactions' partials over the derived rate, times the fold's own rate
    /// over the one divided by. A half rate gives twice, a double one half,
    /// the latter since `GRID` spans the flux range.
    #[test]
    fn a_derived_channel_on_a_rate_computed_another_way_is_reported() {
        let terms = [(1.0, 600), (1.0, 601)];
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &terms),
            own(600, C600),
            own(601, C601),
        ];
        let sums: Sums = &[(NP, &terms)];
        let present = [NP, 600, 601];
        let rxs: Vec<Reaction> = present.iter().map(|&mt| reaction_in(mt, sums)).collect();
        let key = ("O16".to_string(), "(n,p)".to_string());
        let channels = [("(n,p)", NP)];

        let (_, coverage) = fold_against(&blocks, &channels, &present, &rxs, 0.5);
        assert!((coverage.partials_above_rate[&key] - 2.0).abs() < 1.0e-12);
        assert!(coverage.partials_below_rate.is_empty());

        let (_, coverage) = fold_against(&blocks, &channels, &present, &rxs, 2.0);
        assert!((coverage.partials_below_rate[&key] - 0.5).abs() < 1.0e-12);
        assert!(coverage.partials_above_rate.is_empty());
    }

    /// A reaction reached with coefficients that cancel adds nothing to the
    /// channel's covariance, so its variance covers nothing: here 601 enters
    /// with `+1` and `-1`, and only 600's upper interval states a variance.
    #[test]
    fn cancelling_terms_cover_nothing() {
        let terms = [(1.0, 600), (1.0, 601), (-1.0, 601)];
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &terms),
            own(600, [0.0, 0.0, 0.09]),
            own(601, C601),
        ];
        let sums: Sums = &[(NP, &terms)];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        let r = partials(600, sums, 0.0, f64::INFINITY);
        close(
            folded.expect("derived").get(0, 0),
            sandwich(r, symmetric([0.0, 0.0, 0.09]), r) / rate(NP, sums).powi(2),
        );
        let share = coverage.rate_fraction_covered[&("O16".to_string(), "(n,p)".to_string())];
        close(
            share,
            partial(NP, sums, 1, 0.0, f64::INFINITY) / rate(NP, sums),
        );
    }

    /// Where one named partial states a variance over a cell and another
    /// carries rate there and states none, only the stating one's rate is
    /// covered, as ENDF/B-VIII.1 O16 `(n,a)` = 800 + ... + 803 is from 20.5
    /// to 30 MeV, where 800's grid has ended. Here 601 states a variance on
    /// the lower interval only.
    #[test]
    fn a_partial_stating_nothing_over_a_cell_is_not_covered_there() {
        let terms = [(1.0, 600), (1.0, 601)];
        let blocks = [
            nc(NP, 1.0e-5, 2.0e7, &terms),
            own(600, C600),
            own(601, [0.01, 0.0, 0.0]),
        ];
        let sums: Sums = &[(NP, &terms)];
        let (_, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        let share = coverage.rate_fraction_covered[&("O16".to_string(), "(n,p)".to_string())];
        let upper_601 = partial(601, sums, 1, 0.0, f64::INFINITY);
        close(share, 1.0 - upper_601 / rate(NP, sums));
        assert!(share < 1.0);
    }

    /// A shortfall in one cell is not made up by an excess in another: here
    /// `(n,p)` is said to be 600 everywhere, its cross section is above
    /// 600's below 100 keV and below it above, and the heavy fast flux puts
    /// the named rate above the derived one over the whole range. Only the
    /// part of each cell's rate that 600 carries is covered.
    #[test]
    fn a_shortfall_in_one_cell_is_not_covered_by_an_excess_in_another() {
        let blocks = [nc(NP, 1.0e-5, 2.0e7, &[(1.0, 600)]), own(600, C600)];
        let mut np = reaction(NP);
        np.cross_section = vec![2.0, 0.5, 0.05].into();
        let present = [NP, 600];
        let rxs = [np, reaction(600)];
        let (_, coverage) = fold_against(&blocks, &[("(n,p)", NP)], &present, &rxs, 1.0);
        let key = ("O16".to_string(), "(n,p)".to_string());
        assert!(coverage.partials_above_rate[&key] > 1.0);
        let cells = [1.0e-5, 1.0, 1.0e4, 1.0e5, 5.0e6, 2.0e7];
        let (mut carried, mut total) = (0.0, 0.0);
        for w in cells.windows(2) {
            let derived = flux().integrate_xs(&rxs[0], w[0], w[1]);
            let named = flux().integrate_xs(&rxs[1], w[0], w[1]);
            carried += derived.min(named);
            total += derived;
        }
        let share = coverage.rate_fraction_covered[&key];
        close(share, carried / total);
        assert!(share < 1.0);
    }

    /// A channel's own blocks below an NC range and a derivation above it
    /// with a `c = -1` partial never enter together, so the absent block
    /// between the channel and that partial decides nothing and is not
    /// reported.
    #[test]
    fn an_own_block_outside_the_derivation_is_not_opposed_to_its_terms() {
        const E1: f64 = 1.0e4;
        let terms = [(1.0, 601), (-1.0, 600)];
        let blocks = [
            nc(NP, E1, 2.0e7, &terms),
            own(NP, [0.0025, 0.0, 0.0]),
            own(600, [0.0, 0.0, 0.04]),
            cross(600, 601, [0.0, 0.0, 0.0, 0.001]),
            own(601, [0.0, 0.0, 0.0225]),
        ];
        let sums: Sums = &[(NP, &terms)];
        let (folded, coverage) = fold(&blocks, &[("(n,p)", NP)], &[NP, 600, 601], sums);
        assert!(folded.is_some());
        assert!(
            coverage.derived_opposing_uncorrelated.is_empty(),
            "{coverage:?}"
        );
        assert!(!coverage.has_gaps(), "{coverage:?}");

        // Without the cross block the two partials do oppose over the range.
        let (_, coverage) = fold(
            &[
                blocks[0].clone(),
                blocks[1].clone(),
                blocks[2].clone(),
                blocks[4].clone(),
            ],
            &[("(n,p)", NP)],
            &[NP, 600, 601],
            sums,
        );
        assert_eq!(
            coverage.derived_opposing_uncorrelated[&("O16".to_string(), "(n,p)".to_string())],
            [("MT600".to_string(), "MT601".to_string())].into()
        );
    }

    /// Named reactions with rate over a range where the reaction they derive
    /// has none have no ratio to it, and are reported against the channel's
    /// whole rate instead of as an infinite ratio: `(n,p)` is zero up to
    /// 100 keV and the block says it is 600 there.
    #[test]
    fn a_derivation_of_a_reaction_with_no_rate_there_is_finite() {
        const E2: f64 = 1.0e4;
        let blocks = [nc(NP, 1.0e-5, E2, &[(1.0, 600)]), own(600, C600)];
        let mut np = reaction(NP);
        np.cross_section = vec![0.0, 0.0, 5.0].into();
        let present = [NP, 600];
        let rxs = [np, reaction(600)];
        let (_, coverage) = fold_against(&blocks, &[("(n,p)", NP)], &present, &rxs, 1.0);
        let key = ("O16".to_string(), "(n,p)".to_string());
        let whole = flux().integrate_xs(&rxs[0], BOUNDARIES[0], BOUNDARIES[4]);
        let named = flux().integrate_xs(&rxs[1], BOUNDARIES[0], E2);
        let ratio = coverage.partials_above_rate[&key];
        assert!(ratio.is_finite());
        close(ratio, 1.0 + named / whole);
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

    /// A short-range weight restricted to a range keeps the block's own
    /// interval width and restricts only `∫ ψ² dE`, so the squared weights of
    /// two ranges that split an interval add up to the whole interval's: the
    /// variance of its two parts, with nothing correlating them.
    #[test]
    fn a_short_range_weight_within_a_range_restricts_only_the_flux_integral() {
        let rx = reaction(600);
        let cut = 1.0e6;
        let whole = partial_rates(&flux(), &rx, &GRID, Scale::ShortRange).per_interval;
        let below = partial_rates_within(&flux(), &rx, &GRID, Scale::ShortRange, (0.0, cut));
        let above = partial_rates_within(&flux(), &rx, &GRID, Scale::ShortRange, (cut, 1.0e8));
        let width = GRID[2] - GRID[1];
        let want = BARN_TO_CM2 * (width * flux().integrate_density_squared(GRID[1], cut)).sqrt();
        assert!((below.per_interval[1] / want - 1.0).abs() < 1.0e-12);
        assert_eq!(below.per_interval[0], whole[0]);
        assert_eq!(above.per_interval[0], 0.0);
        let parts = below.per_interval[1].powi(2) + above.per_interval[1].powi(2);
        assert!((parts / whole[1].powi(2) - 1.0).abs() < 1.0e-12, "{parts}");
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

#[cfg(test)]
mod lumped_tests {
    //! Lumped reactions (ENDF-102 33.2.3): MT 851-870 state the covariance of
    //! the sum of their components, which are named only on the components'
    //! own HEAD records. One component is that component; several are
    //! reported and not folded.

    use super::*;
    use endf::mf::covariance::NcSubsection;

    const BOUNDARIES: [f64; 4] = [1.0e-5, 1.0, 1.0e6, 2.0e7];
    const FLUX: [f64; 3] = [1.0e10, 1.0e11, 1.0e12];
    const GRID: [f64; 3] = [1.0e-5, 1.0e4, 2.0e7];
    const N2N: i32 = 16;
    const CAPTURE: i32 = 102;
    const NP: i32 = 103;

    fn flux() -> FluxDensity<'static> {
        FluxDensity {
            boundaries: &BOUNDARIES,
            flux: &FLUX,
            shape: None,
        }
    }

    fn reaction(mt: i32) -> Reaction {
        let k = f64::from(mt % 100 + 1);
        Reaction {
            cross_section: vec![k, 0.5 * k, 0.1 * k * k].into(),
            threshold_idx: 0,
            energy: vec![1.0e-5, 1.0e5, 2.0e7].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![].into(),
            scatter_in_cm: false,
            redundant: false,
        }
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
            mat: 7443,
            data,
        }
    }

    /// Reaction `mt`'s section when it is a component of `mtl`: its HEAD
    /// and nothing else, as `covariance.arrow` reads it back.
    fn component(mt: i32, mtl: i32) -> CovarianceBlock {
        CovarianceBlock {
            mat1: 0,
            mt1: 0,
            mtl,
            ..with(mt, 0, CovarianceData::Lumped)
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

    /// `lb = 5`, `ls = 0` on `GRID`: `mt` with `mt1`, row-major.
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

    /// An LTY=0 block over the whole axis, `σ_mt = Σ c_i σ_mt_i`.
    fn nc(mt: i32, terms: &[(f64, i32)]) -> CovarianceBlock {
        let data = NcSubsection {
            lty: 0,
            e1: 1.0e-5,
            e2: 2.0e7,
            nci: terms.len() as i64,
            ci: terms.iter().map(|t| t.0).collect(),
            xmti: terms.iter().map(|t| f64::from(t.1)).collect(),
            ..Default::default()
        };
        with(mt, mt, CovarianceData::Nc(data))
    }

    fn kind(mt: i32) -> &'static str {
        match mt {
            N2N => "(n,2n)",
            CAPTURE => "(n,gamma)",
            NP => "(n,p)",
            _ => unreachable!("not a channel here"),
        }
    }

    /// Fold `blocks` for W186 with `channels` driven and a cross section for
    /// every MT in `present`.
    fn fold(
        blocks: &[CovarianceBlock],
        channels: &[i32],
        present: &[i32],
    ) -> (Option<RateCovariance>, Coverage) {
        let rxs: Vec<Reaction> = present.iter().map(|&mt| reaction(mt)).collect();
        fold_with("W186", blocks, channels, &rxs)
    }

    /// [`fold`] for `nuclide`, with the cross sections given.
    fn fold_with(
        nuclide: &str,
        blocks: &[CovarianceBlock],
        channels: &[i32],
        rxs: &[Reaction],
    ) -> (Option<RateCovariance>, Coverage) {
        let reactions: BTreeMap<i32, &Reaction> = rxs.iter().map(|r| (r.mt_number, r)).collect();
        let kinds: Vec<(String, i32)> = channels
            .iter()
            .map(|&mt| (kind(mt).to_string(), mt))
            .collect();
        let rates: BTreeMap<String, f64> = kinds
            .iter()
            .map(|(k, mt)| {
                let full =
                    BARN_TO_CM2 * flux().integrate_xs(reactions[mt], BOUNDARIES[0], BOUNDARIES[3]);
                (k.clone(), full)
            })
            .collect();
        let mut coverage = Coverage::default();
        let folded = fold_nuclide(
            &flux(),
            blocks,
            &reactions,
            &kinds,
            &rates,
            nuclide,
            &mut coverage,
        );
        (folded, coverage)
    }

    const U: [f64; 3] = [0.01, 0.004, 0.0225];
    const V: [f64; 3] = [0.04, -0.01, 0.09];
    const X: [f64; 4] = [0.003, 0.001, -0.002, 0.005];

    /// A lump of one reaction is that reaction, so its own and its cross
    /// blocks fold exactly, bit for bit, as the same numbers written on the
    /// component would.
    #[test]
    fn a_single_component_lump_folds_as_its_component() {
        let lumped = [
            component(N2N, 851),
            own(CAPTURE, V),
            cross(CAPTURE, 851, X),
            own(851, U),
        ];
        let direct = [own(CAPTURE, V), cross(CAPTURE, N2N, X), own(N2N, U)];
        let (a, coverage) = fold(&lumped, &[N2N, CAPTURE], &[N2N, CAPTURE]);
        let (b, direct_coverage) = fold(&direct, &[N2N, CAPTURE], &[N2N, CAPTURE]);
        let a = a.expect("the lump is the component's covariance");
        assert_eq!(Some(a.clone()), b);
        assert!(a.get(0, 1) != 0.0, "the cross block is folded");
        assert_eq!(coverage, direct_coverage);
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// An NC block naming a single-component lump derives from the
    /// component, which has the cross section the lump lacks.
    #[test]
    fn a_derivation_naming_a_single_component_lump_reads_the_component() {
        let lumped = [nc(NP, &[(1.0, 851)]), component(600, 851), own(851, U)];
        let direct = [nc(NP, &[(1.0, 600)]), own(600, U)];
        let (a, coverage) = fold(&lumped, &[NP], &[NP, 600]);
        let (b, _) = fold(&direct, &[NP], &[NP, 600]);
        assert!(a.is_some());
        assert_eq!(a, b);
        assert!(coverage.skipped_nc.is_empty(), "{coverage:?}");
    }

    /// Only this evaluation's MTs are renamed: another material's `mt1`
    /// that happens to equal the lump's is that material's reaction.
    #[test]
    fn another_materials_mt1_is_not_renamed() {
        let mut other = cross(CAPTURE, 851, X);
        other.mat1 = 9228;
        let blocks = [component(N2N, 851), other];
        let renamed = single_component_lumps(&blocks);
        assert_eq!(renamed.len(), 1, "the component's HEAD is read and dropped");
        assert_eq!(renamed[0].mt1, 851);
    }

    /// W180 to W186 in ENDF/B-VIII.1, FENDL-3.2d and JEFF-4.0 give `(n,2n)`
    /// only as MT 852 = MT 16 + MT 41. Handing the sum's covariance to 16
    /// would be an assumption, so nothing folds and the lump is reported.
    #[test]
    fn a_lump_of_several_components_is_reported_and_not_folded() {
        let blocks = [component(N2N, 852), component(41, 852), own(852, U)];
        let (folded, coverage) = fold(&blocks, &[N2N], &[N2N, 41]);
        assert!(folded.is_none());
        let want: BTreeSet<String> = ["(n,2n)", "MT41"].map(String::from).into();
        assert_eq!(
            coverage.lumped_covariance_not_assignable,
            BTreeMap::from([(("W186".to_string(), 852), want)])
        );
        assert!(coverage.has_gaps());
    }

    /// W186 MT 855 is MT 600 + MT 649, the levels `(n,p)` holds, so it is
    /// the channel's covariance lost even though no component is a channel.
    #[test]
    fn a_lump_of_a_channels_levels_is_reported() {
        let blocks = [component(600, 855), component(649, 855), own(855, U)];
        let (_, coverage) = fold(&blocks, &[NP], &[NP]);
        let want: BTreeSet<String> = ["MT600", "MT649"].map(String::from).into();
        assert_eq!(
            coverage.lumped_covariance_not_assignable,
            BTreeMap::from([(("W186".to_string(), 855), want)])
        );
    }

    /// A lump of `(n,p)`'s levels is not lost where MT 103 states its own
    /// covariance: the channel's rate variance is that section's, whole.
    #[test]
    fn a_lump_of_levels_is_not_reported_where_their_sum_states_its_own() {
        let blocks = [
            own(NP, U),
            component(600, 855),
            component(649, 855),
            own(855, U),
        ];
        let (folded, coverage) = fold(&blocks, &[NP], &[NP]);
        assert!(folded.is_some());
        assert!(coverage.lumped_covariance_not_assignable.is_empty());
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// U235 MT 4 is only a derivation, MT 51 + MT 851. Where the lump's sum
    /// cannot be built, that derivation states nothing, so it does not hide
    /// the lump of `(n,n')`'s levels the way an explicit section would.
    #[test]
    fn a_derivation_on_the_level_sum_does_not_hide_an_unfolded_lump() {
        let blocks = [
            nc(NP, &[(1.0, 855)]),
            component(600, 855),
            component(649, 855),
            own(855, U),
        ];
        let (_, coverage) = fold(&blocks, &[NP], &[NP]);
        let want: BTreeSet<String> = ["MT600", "MT649"].map(String::from).into();
        assert_eq!(
            coverage.lumped_covariance_not_assignable,
            BTreeMap::from([(("W186".to_string(), 855), want)])
        );
        assert!(coverage.has_gaps());
    }

    /// A lump no channel reaches has no rate to be the uncertainty of, and
    /// one with no block of its own states nothing to lose.
    #[test]
    fn an_unreached_or_empty_lump_is_not_reported() {
        let blocks = [component(N2N, 852), component(41, 852), own(852, U)];
        let (_, coverage) = fold(&blocks, &[CAPTURE], &[CAPTURE]);
        assert!(coverage.lumped_covariance_not_assignable.is_empty());
        let blocks = [component(N2N, 852), component(41, 852), own(CAPTURE, V)];
        let (_, coverage) = fold(&blocks, &[N2N, CAPTURE], &[N2N, CAPTURE]);
        assert!(coverage.lumped_covariance_not_assignable.is_empty());
    }

    /// ENDF/B-VIII.1 Li7's MF=33, through `covariance.arrow` as a run reads it.
    fn li7() -> Vec<CovarianceBlock> {
        let tmp = tempfile::tempdir().expect("temp dir");
        let mut raw = Vec::new();
        lzma_rs::xz_decompress(
            &mut &include_bytes!("../../endf/fixtures/n-003_Li_007_mf33.endf.xz")[..],
            &mut raw,
        )
        .expect("fixture decompresses");
        let text = String::from_utf8(raw).expect("ENDF is text");
        let material = endf::Material::from_str(&text).expect("evaluation parses");
        assert!(yamc_convert::covariance::write_covariance(&material, tmp.path()).expect("writes"));
        yamc_nuclide::arrow::covariance_arrow::read_covariance(tmp.path(), "Li7")
            .expect("reads")
            .expect("the file is there")
    }

    /// ENDF/B-VIII.1 Li7 lumps MT 51 alone as MT 852 and MT 56 alone as MT
    /// 854. Those two read as their components, blocks, cross blocks and
    /// all, and the seven lumps of several components are left as written.
    #[test]
    fn endfb_li7_single_component_lumps_read_as_their_components() {
        let blocks = li7();
        let renamed = single_component_lumps(&blocks);
        let names = |bs: &[CovarianceBlock], mt: i32| {
            bs.iter()
                .filter(|b| b.lumped_into().is_none() && (b.mt == mt || b.partner_mt() == mt))
                .count()
        };
        for (mtl, component) in [(852, 51), (854, 56)] {
            let stated = names(&blocks, mtl);
            assert!(stated > 0, "MT {mtl} has blocks");
            assert_eq!(
                names(&renamed, mtl),
                0,
                "MT {mtl} is read as MT {component}"
            );
            assert_eq!(names(&renamed, component), stated);
            assert!(renamed.iter().all(|b| b.lumped_into() != Some(mtl)));
        }
        // Every other lump keeps its own MT, and its components their HEADs.
        for mtl in [851, 853, 855, 856, 857, 858, 859] {
            assert_eq!(names(&renamed, mtl), names(&blocks, mtl), "MT {mtl}");
        }
        assert_eq!(
            renamed.iter().filter(|b| b.lumped_into().is_some()).count(),
            blocks.iter().filter(|b| b.lumped_into().is_some()).count() - 2
        );
        assert_eq!(renamed.len(), blocks.len() - 2);
    }

    /// ENDF/B-VIII.1 Li7's MT 2 section names MT 851, the lump of MT 16 and
    /// MT 24, which has no cross section of its own. On the cached evaluation
    /// the two share the nuclide's grid, so the sum is built, point for point
    /// the components' sum, and the derivation is not skipped for want of it.
    /// Self-skips when the fixture is not cached.
    #[test]
    fn endfb_li7_n2n_lump_sum_is_built_on_the_cached_evaluation() {
        let Some(dir) = yamc_test_cache::nuclide("Li7") else {
            eprintln!("skipping: the Li7 fixture is not cached");
            return;
        };
        let mut m = yamc_materials::Material::new(
            std::collections::HashMap::from([("Li7".to_string(), 1.0)]),
            "atom",
            "sum",
            None,
        )
        .expect("material");
        m.set_temperature("294");
        m.read_nuclear_data(
            &std::collections::HashMap::from([("Li7".to_string(), dir)]),
            None,
        )
        .expect("Li7 loads");
        let by_mt = m.nuclide_data["Li7"]
            .reactions_for_temp("294")
            .expect("294 K");
        let reactions: BTreeMap<i32, &Reaction> =
            by_mt.iter().map(|(mt, r)| (*mt, r.as_ref())).collect();
        assert!(!reactions.contains_key(&851), "a lump has no MF=3 section");
        let sums = lump_cross_sections(&li7(), &reactions);
        let sum = sums
            .iter()
            .find(|r| r.mt_number == 851)
            .expect("MT 851 is built from MT 16 and MT 24");
        for (&e, &x) in sum.energy.iter().zip(sum.cross_section.iter()) {
            let parts: f64 = [16, 24]
                .iter()
                .map(|mt| reactions[mt].cross_section_at(e).unwrap_or(0.0))
                .sum();
            assert!(
                (x - parts).abs() <= 1e-12 * parts.abs(),
                "{e} eV: {x} vs {parts}"
            );
        }
    }

    /// `mt` on the shared grid `GRID` from index `threshold`, as a nuclide's
    /// reactions sit on its own grid.
    fn on_grid(mt: i32, threshold: usize, xs: &[f64]) -> Reaction {
        Reaction {
            cross_section: xs.to_vec().into(),
            threshold_idx: threshold,
            energy: GRID[threshold..].to_vec().into(),
            ..reaction(mt)
        }
    }

    /// ENDF/B-VIII.1 and TENDL-2017 U235 and U238 state MT 4 only as MT 51
    /// plus the lumped MT 851, the sum of MT 52 to 91; here `(n,p)` is MT 600
    /// plus a lump of MT 601 and 602. The lump's cross section is its
    /// components' sum, so the derivation folds the lump's own and cross
    /// blocks exactly, as the same numbers on a reaction with that cross
    /// section would, and nothing is listed as unassignable.
    #[test]
    fn a_derivation_naming_a_lump_of_several_folds_its_sum() {
        let (a, b) = ([0.0, 2.0, 1.0], [0.0, 3.0]);
        let first = on_grid(600, 0, &[4.0, 0.5, 0.25]);
        let lump = on_grid(851, 0, &[a[0], a[1] + b[0], a[2] + b[1]]);
        let total = on_grid(NP, 0, &[4.0, 0.5 + a[1] + b[0], 0.25 + a[2] + b[1]]);
        let derived = nc(NP, &[(1.0, 600), (1.0, 851)]);
        let lumped = [
            derived.clone(),
            component(601, 851),
            component(602, 851),
            own(600, V),
            cross(600, 851, X),
            own(851, U),
        ];
        let direct = [derived, own(600, V), cross(600, 851, X), own(851, U)];
        let (got, coverage) = fold_with(
            "U235",
            &lumped,
            &[NP],
            &[
                total.clone(),
                first.clone(),
                on_grid(601, 0, &a),
                on_grid(602, 1, &b),
            ],
        );
        let (want, _) = fold_with("U235", &direct, &[NP], &[total, first, lump]);
        let got = got.expect("the lump folds through the derivation");
        assert_eq!(Some(got.clone()), want);
        assert!(got.get(0, 0) > 0.0);
        assert!(coverage.lumped_covariance_not_assignable.is_empty());
        assert!(coverage.skipped_nc.is_empty(), "{coverage:?}");
        assert!(!coverage.has_gaps(), "{coverage:?}");
    }

    /// The components are summed point by point only where that is the sum:
    /// one grid, each component its tail, and any starting above the lowest
    /// zero at its first point, as it is below it.
    #[test]
    fn a_lump_is_summed_only_on_one_grid() {
        let low = on_grid(601, 0, &[1.0, 2.0, 3.0]);
        let sum = summed(851, &[&low, &on_grid(602, 1, &[0.0, 5.0])]).expect("one grid");
        assert_eq!(&sum.cross_section[..], &[1.0, 2.0, 8.0]);
        assert_eq!(sum.threshold_idx, 0);
        assert_eq!(&sum.energy[..], &GRID);
        // A step at the component's first point is not linear from the one
        // below it.
        assert!(summed(851, &[&low, &on_grid(602, 1, &[1.0, 5.0])]).is_none());
        // Nor is a component on a grid of its own.
        let own_grid = Reaction {
            energy: vec![2.0e-5, 1.0e4, 2.0e7].into(),
            ..low.clone()
        };
        assert!(summed(851, &[&low, &own_grid]).is_none());
    }

    /// Without its components' cross sections a lump has none, so the
    /// derivation through it is skipped and counted, and the load is asked
    /// for them.
    #[test]
    fn a_lump_without_its_components_cross_sections_is_skipped() {
        let blocks = [
            nc(NP, &[(1.0, 600), (1.0, 851)]),
            component(601, 851),
            component(602, 851),
            own(851, U),
        ];
        let (_, coverage) = fold(&blocks, &[NP], &[NP, 600, 601]);
        assert_eq!(coverage.skipped_nc.get("W186"), Some(&1));
        let chain = yani::ChainNuclide {
            name: "W186".to_string(),
            half_life: None,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: vec![yani::ChainReaction {
                kind: "(n,p)".to_string(),
                target: Some("Ta186".to_string()),
                branching: 1.0,
                q_value: Some(0.0),
                branching_uncertainty: None,
                evaluated_branching: None,
            }],
            decays: Vec::new(),
            fission_yields: None,
            sources: Vec::new(),
        };
        assert_eq!(
            reachable_mts(&chain, &blocks),
            BTreeSet::from([NP, 600, 601, 602, 851])
        );
    }

    /// ENDF/B-VIII.1 Li7 MT 851 is MT 16 plus MT 24, and no derivation of
    /// `(n,2n)` names it, so a run driving `(n,2n)` lists it from the
    /// fixture's own rows.
    #[test]
    fn endfb_li7_lists_its_n2n_lump() {
        let rxs = [reaction(N2N), reaction(24)];
        let (_, coverage) = fold_with("Li7", &li7(), &[N2N], &rxs);
        let want: BTreeSet<String> = ["(n,2n)", "MT24"].map(String::from).into();
        assert_eq!(
            coverage.lumped_covariance_not_assignable,
            BTreeMap::from([(("Li7".to_string(), 851), want)])
        );
        assert!(coverage.has_gaps());
    }

    /// The fold's report merges across spectra like the rest.
    #[test]
    fn the_report_unions_across_spectra() {
        let key = ("W186".to_string(), 852);
        let mut a = Coverage::default();
        a.lumped_covariance_not_assignable
            .insert(key.clone(), ["(n,2n)".to_string()].into());
        let mut b = Coverage::default();
        b.lumped_covariance_not_assignable
            .insert(key.clone(), ["MT41".to_string()].into());
        a.absorb(b);
        assert_eq!(a.lumped_covariance_not_assignable[&key].len(), 2);
    }
}

#[cfg(test)]
mod transport_field_tests {
    use super::*;

    fn reaction(mt: i32, redundant: bool) -> Reaction {
        Reaction {
            cross_section: vec![1.0, 1.0].into(),
            threshold_idx: 0,
            energy: vec![1.0e-5, 2.0e7].into(),
            mt_number: mt,
            q_value: 0.0,
            products: vec![].into(),
            scatter_in_cm: false,
            redundant,
        }
    }

    fn self_block(mt: i32, variance: f64) -> CovarianceBlock {
        CovarianceBlock {
            mt,
            subsection_idx: 0,
            block_idx: 0,
            mat1: 0,
            mt1: mt,
            xmf1: 0.0,
            xlfs1: 0.0,
            mtl: 0,
            mat: 0,
            data: CovarianceData::Ni(NiSubsection {
                lb: 1,
                np: 3,
                ek: vec![1.0e-5, 1.0e6, 2.0e7],
                fk: vec![variance, variance, 0.0],
                ..Default::default()
            }),
        }
    }

    fn field(partials: &[i32], blocks: &[CovarianceBlock]) -> TransportField {
        let owned: Vec<Reaction> = partials
            .iter()
            .map(|&mt| reaction(mt, false))
            .chain([reaction(4, true)])
            .collect();
        let reactions: BTreeMap<i32, &Reaction> = owned.iter().map(|r| (r.mt_number, r)).collect();
        transport_field("X", partials, blocks, &reactions)
    }

    /// Elastic states its own covariance, the levels take the total
    /// inelastic's, and capture with none is held at nominal.
    #[test]
    fn partials_read_their_own_their_sums_or_nothing() {
        let t = field(
            &[2, 51, 52, 102],
            &[self_block(2, 0.01), self_block(4, 0.04)],
        );
        assert_eq!(t.reads[&2], Read::Own);
        assert_eq!(t.reads[&51], Read::Parent(4));
        assert_eq!(t.reads[&52], Read::Parent(4));
        assert_eq!(t.reads[&102], Read::Nominal);
        assert!(
            !t.reads.contains_key(&4),
            "a redundant sum is not sampled itself"
        );
        // Elastic's two cells and the inelastic sum's two, the levels sharing
        // the sum's.
        let f = t.field.expect("covered reactions make a field");
        let mts: BTreeSet<i32> = f.relative_cells.iter().map(|c| c.mt).collect();
        assert_eq!(mts, BTreeSet::from([2, 4]));
        assert_eq!(f.relative_cells.len(), 4);
    }

    /// A level that states its own covariance reads that, not the sum's.
    #[test]
    fn a_level_with_its_own_covariance_reads_it() {
        let t = field(&[51, 52], &[self_block(51, 0.09), self_block(4, 0.04)]);
        assert_eq!(t.reads[&51], Read::Own);
        assert_eq!(t.reads[&52], Read::Parent(4));
    }

    /// With no covariance anywhere, every partial is held and there is no
    /// field.
    #[test]
    fn no_covariance_holds_every_partial() {
        let t = field(&[2, 102], &[]);
        assert!(t.reads.values().all(|r| *r == Read::Nominal));
        assert!(t.field.is_none());
    }

    /// `self_block` with the variance on its first interval only, `[1e-5,
    /// 1e6]`, and `lb` chosen.
    fn low_block(mt: i32, lb: i64, variance: f64) -> CovarianceBlock {
        let mut b = self_block(mt, variance);
        if let CovarianceData::Ni(ni) = &mut b.data {
            ni.lb = lb;
            ni.fk = vec![variance, 0.0, 0.0];
        }
        b
    }

    /// An LTY=0 block on `mt` over `[e1, e2]`, `σ_mt = Σ c_i σ_mt_i`.
    fn nc(mt: i32, e1: f64, e2: f64, terms: &[(f64, i32)]) -> CovarianceBlock {
        CovarianceBlock {
            data: CovarianceData::Nc(endf::mf::covariance::NcSubsection {
                lty: 0,
                e1,
                e2,
                nci: terms.len() as i64,
                ci: terms.iter().map(|t| t.0).collect(),
                xmti: terms.iter().map(|t| f64::from(t.1)).collect(),
                ..Default::default()
            }),
            ..self_block(mt, 0.0)
        }
    }

    /// [`field`] with the total held too, as a redundant reaction.
    fn field_with_total(partials: &[i32], blocks: &[CovarianceBlock]) -> TransportField {
        let owned: Vec<Reaction> = partials
            .iter()
            .map(|&mt| reaction(mt, false))
            .chain([reaction(1, true), reaction(4, true)])
            .collect();
        let reactions: BTreeMap<i32, &Reaction> = owned.iter().map(|r| (r.mt_number, r)).collect();
        transport_field("X", partials, blocks, &reactions)
    }

    /// Elastic with no cells of its own, derived above 1 MeV from the total
    /// less the other partials as ENDF/B-VIII.1 Pb208 derives it above
    /// 1.5 MeV, reads the named reactions' cells over that range. A named
    /// reaction with no cells, here `(n,2n)`, adds nothing.
    #[test]
    fn a_partial_an_nc_block_derives_reads_the_named_reactions() {
        let t = field_with_total(
            &[2, 16, 51, 102],
            &[
                self_block(1, 0.01),
                self_block(4, 0.04),
                self_block(102, 0.09),
                nc(
                    2,
                    1.0e6,
                    2.0e7,
                    &[(1.0, 1), (-1.0, 4), (-1.0, 16), (-1.0, 102)],
                ),
            ],
        );
        assert_eq!(t.reads[&2], Read::Own);
        assert_eq!(t.reads[&16], Read::Nominal);
        assert_eq!(t.reads[&51], Read::Parent(4));
        assert_eq!(t.reads[&102], Read::Own);
        let range = (1.0e6, 2.0e7);
        let term = |coefficient: f64, mt: i32| DerivedTerm {
            coefficient,
            mt,
            cross_sections: vec![mt],
            range,
        };
        assert_eq!(
            t.derived,
            BTreeMap::from([(2, vec![term(1.0, 1), term(-1.0, 4), term(-1.0, 102)])])
        );
        let f = t.field.expect("a field");
        assert!(f.relative_cells.iter().any(|c| c.mt == 1));
        assert!(!f.relative_cells.iter().any(|c| c.mt == 2));
    }

    /// A derivation is kept only where it reaches cells: one naming a
    /// reaction whose only variance lies below the derivation's range moves
    /// nothing, so the partial is held at nominal rather than reported as
    /// perturbed with no draw ever moving it.
    #[test]
    fn a_derivation_that_reaches_no_cells_over_its_range_is_held() {
        let t = field_with_total(
            &[2],
            &[low_block(1, 1, 0.01), nc(2, 1.0e6, 2.0e7, &[(1.0, 1)])],
        );
        assert_eq!(t.reads[&2], Read::Nominal);
        assert!(t.derived.is_empty());

        // Over a range the named reaction's cells do cover, it reads them.
        let t = field_with_total(
            &[2],
            &[low_block(1, 1, 0.01), nc(2, 1.0e-5, 2.0e7, &[(1.0, 1)])],
        );
        assert_eq!(t.reads[&2], Read::Own);
        assert_eq!(t.derived[&2][0].mt, 1);
    }

    /// Transport holds short-range (`lb = 8`) noise at nominal, so a sum or
    /// a reaction that only such a block covers perturbs nothing and is
    /// reported held, not perturbed.
    #[test]
    fn short_range_blocks_alone_perturb_nothing_in_transport() {
        let t = field(
            &[2, 51, 102],
            &[low_block(4, 8, 0.04), low_block(2, 8, 0.01)],
        );
        assert_eq!(t.reads[&2], Read::Nominal);
        assert_eq!(t.reads[&51], Read::Nominal);
        assert_eq!(t.reads[&102], Read::Nominal);
        assert!(t.field.is_none());

        // Beside a block transport applies, the read stands.
        let t = field(&[51], &[low_block(4, 8, 0.04), self_block(4, 0.04)]);
        assert_eq!(t.reads[&51], Read::Parent(4));
    }
}
