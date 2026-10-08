//! Nuclear-data uncertainty on the inventory, by resampling and re-solving.
//!
//! The Bateman solve is deterministic and every expensive input is a reaction
//! rate, so the uncertainty is propagated by perturbing the rates and running
//! the same solve again. That is exact to all orders in the matrix exponential:
//! nothing is linearized, and the solver is not touched at all, only its input.
//!
//! What can be perturbed is [`Source::IMPLEMENTED`], each source documented on
//! its variant. [`Info`] says per run what was perturbed, what carried no
//! stated uncertainty, and what was held at nominal, rather than leaving it to
//! be inferred from a small sigma.
//!
//! # Cost
//!
//! One CRAM solve per replica per step, on top of the nominal run. The cross
//! sections are loaded once, the covariance is folded once per distinct
//! spectrum, and the factorization is done once, so a replica is the solve and
//! nothing else.
//!
//! # Nothing happens unless asked
//!
//! With no [`DataUncertainty`] the covariance is never read from disk, no
//! matrix is folded or factorized, and the step loop runs exactly once. The
//! means are bit-identical to a build without any of this.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use yamc_materials::Material;

use crate::covariance_fold::Coverage;
use crate::covariance_sample::{lognormal_multiplier, Repair, SigmaReport};

/// One input to the Bateman matrix that can be perturbed.
///
/// Named rather than boolean-per-field so the set can grow without every
/// caller's signature changing, and so asking for one that does not exist yet
/// is an ERROR rather than a silent no-op. That matters for the workflow this
/// exists to support: adding sources one at a time and watching the inventory
/// sigma grow. A typo, or a source that has not landed, must not look like a
/// source that contributed nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    /// Activation cross sections, from ENDF MF=33 covariance.
    CrossSections,
    /// The supplied flux spectrum, from a per-bin standard deviation the caller
    /// provides.
    ///
    /// Unlike the others this needs no nuclear data: the flux is the caller's
    /// own input. It contributes only where a sigma was actually given, which
    /// a spectrum from a published reference set does not have.
    FluxSpectrum,
    /// Half-lives, from the decay sublibrary's own standard deviation on each
    /// (the uncertainty on the MT=457 `T1/2`), each nuclide drawn
    /// independently as a lognormal with the stated mean and sigma.
    ///
    /// A replica's half-lives are used everywhere that replica's decay
    /// constants appear: in the solve and in the activity, decay heat and dose
    /// evaluated from its inventory. Using the nominal value in either place
    /// would break the cancellation that makes a saturated activity nearly
    /// insensitive to its own half-life (`N ~ R / lambda`, so `A ~ R`), and
    /// inflate its uncertainty.
    HalfLife,
    /// Decay branching ratios, from the MT=457 sigma on each mode, for the
    /// parents whose sum rule fixes the joint distribution: exactly two modes,
    /// one sigma between them (both state the same one, or one states it and
    /// the other is its complement), and the smaller ratio at least five
    /// sigmas from zero. A replica draws one deviate per parent, moves one
    /// mode by `sigma z` and the other by `-sigma z`, so the pair's total is
    /// kept and the two are perfectly anticorrelated.
    ///
    /// Any other multi-mode parent stays at nominal and [`Info`] names it by
    /// why. It moves the inventory only: a parent's own per-decay emission is
    /// the decay scheme's.
    DecayBranching,
    /// The Monte Carlo statistical uncertainty of transport-tallied reaction
    /// rates, from their per-history covariance.
    ///
    /// Applies to `Model.simulate_transmutation`, whose rates come from
    /// transport; a spectrum run's rates are a deterministic collapse with no
    /// sampling error, so it has nothing to perturb there.
    Statistical,
    /// Mean decay energies, from the decay data's sigma on each
    /// recoverable-heat component (beta, gamma, alpha), or on the total where
    /// the data carries no split, each drawn as a lognormal with the stated
    /// mean and sigma.
    ///
    /// A decay energy does not enter the Bateman matrix, so the inventory is
    /// untouched: it moves the decay heat evaluated from each replica, and
    /// nothing else.
    DecayEnergy,
}

impl Source {
    /// Every source this build can perturb. [`Info::not_perturbed`] lists what
    /// a run held at nominal.
    pub const IMPLEMENTED: &'static [Source] = &[
        Source::CrossSections,
        Source::FluxSpectrum,
        Source::HalfLife,
        Source::DecayBranching,
        Source::Statistical,
        Source::DecayEnergy,
    ];

    /// The name used in the API and in the coverage report.
    pub fn name(self) -> &'static str {
        match self {
            Source::CrossSections => "cross_sections",
            Source::FluxSpectrum => "flux_spectrum",
            Source::HalfLife => "half_life",
            Source::DecayBranching => "decay_branching",
            Source::Statistical => "statistical",
            Source::DecayEnergy => "decay_energy",
        }
    }

    /// Parse a name, naming what IS available when it is not one of them.
    pub fn parse(name: &str) -> Result<Source, String> {
        Source::IMPLEMENTED
            .iter()
            .copied()
            .find(|s| s.name() == name)
            .ok_or_else(|| {
                let have: Vec<&str> = Source::IMPLEMENTED.iter().map(|s| s.name()).collect();
                format!(
                    "unknown uncertainty source {name:?}; this build can perturb {have:?}. \
                     Fission yields carry published uncertainties that are not read yet, \
                     and reaction branching has none in ENDF-6 to read."
                )
            })
    }
}

/// Ask for nuclear-data uncertainty on a transmutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataUncertainty {
    /// Base seed. The perturbation of a given nuclide in a given replica is a
    /// pure function of `(seed, replica, nuclide)`, so a rerun with the same
    /// seed gives the same answer whatever else changed about the run.
    pub seed: u64,
    /// Fixed replica count, or `None` to run until the sigma estimate settles.
    ///
    /// `None` is the recommended setting and the one with no accuracy knob in
    /// it: the driver adds replicas until the reported sigmas stop moving by
    /// more than an internal tolerance. A number here is for reproducing a
    /// specific run or for bounding cost, not for trading accuracy.
    pub samples: Option<usize>,
    /// Which inputs to perturb. Empty means every implemented source.
    ///
    /// Restricting this is how a run isolates one contribution, which is what
    /// makes "add a source, watch the sigma grow" a measurement rather than a
    /// guess. Sources are independent, so the total sigma must not DECREASE as
    /// the set grows; that is a property worth testing.
    pub sources: Vec<Source>,
    /// Also say where the uncertainty comes from ([`Attribution`]).
    ///
    /// Off by default because it costs further solves: one ensemble per
    /// source, and one deterministic solve per contributor. It changes no
    /// number the run otherwise reports.
    pub attribution: bool,
}

impl Default for DataUncertainty {
    fn default() -> Self {
        Self {
            seed: 1,
            samples: None,
            sources: Source::IMPLEMENTED.to_vec(),
            attribution: false,
        }
    }
}

impl DataUncertainty {
    /// Whether `source` is switched on for this request.
    ///
    /// An empty set means every implemented source, so a default-constructed
    /// request perturbs everything this build can.
    pub fn wants(&self, source: Source) -> bool {
        self.sources.is_empty() || self.sources.contains(&source)
    }
}

/// Replicas are added in blocks, and convergence is judged between blocks.
pub(crate) const BLOCK: usize = 64;
/// Below this the sigma estimate is too noisy to judge, whatever it says.
pub(crate) const MIN_SAMPLES: usize = 128;
/// A ceiling, so a pathological problem cannot run forever.
pub(crate) const MAX_SAMPLES: usize = 1024;
/// Relative movement in a nuclide's sigma between blocks that counts as settled.
pub(crate) const TOLERANCE: f64 = 0.02;
/// Nuclides below this share of the largest final density are not tracked for
/// convergence: their sigma is noise on a number nobody reads.
pub(crate) const SIGNIFICANCE: f64 = 1.0e-6;

/// What was perturbed, and what could not be.
///
/// The point of this type is that a missing uncertainty and a genuinely zero
/// one must never look the same. A nuclide in `no_covariance_data` contributed
/// nothing to the spread because the evaluation says nothing about it, which is
/// a different statement from "this channel is well known".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Info {
    /// Replicas actually run.
    pub samples: usize,
    /// Whether the run stopped because the sigmas settled rather than on the cap.
    pub converged: bool,
    /// Nuclides whose cross sections were perturbed.
    pub perturbed: BTreeSet<String>,
    /// Nuclides with rates but no usable covariance, so no stated uncertainty.
    pub no_covariance_data: BTreeSet<String>,
    /// Per nuclide, blocks correlating one of its reactions with a reaction of
    /// another evaluation, not consumed. A `mat1` naming the nuclide's own MAT
    /// is its own evaluation and is folded, and a block is counted only on a
    /// reaction the fold reaches (a channel, or one a channel is derived
    /// from). The partner is not checked: a block is counted whether or not
    /// the evaluation `mat1` names is in the run.
    pub skipped_cross_material: BTreeMap<String, usize>,
    /// Per nuclide, blocks correlating one of its reactions with a quantity
    /// that is not a cross section (`xmf1` other than 0 or 3), not consumed.
    /// Counted like `skipped_cross_material`.
    pub skipped_other_file: BTreeMap<String, usize>,
    /// Per (nuclide, kind, kind), where a pair stored in both orientations
    /// has copies that are not each other's transpose, the largest difference
    /// relative to the largest entry. The lower MT's copy is the one used.
    pub mirrored_disagree: BTreeMap<(String, String, String), f64>,
    /// Per nuclide, NC blocks (covariance derived from other reactions) that
    /// could not be derived, not consumed. LTY=0 blocks are derived from the
    /// reactions they name; what is left is an LTY other than 0, a block in
    /// a cross-reaction subsection, one whose list of reactions is empty or
    /// does not match its coefficients, one whose own energy range is empty,
    /// one naming a reaction with no cross section, and one met only
    /// circularly.
    pub skipped_nc: BTreeMap<String, usize>,
    /// Blocks whose `lb` layout is not implemented, counted per `lb`.
    pub unsupported_layouts: BTreeMap<i64, usize>,
    /// Blocks not consumed because they break ENDF-102's rules for their
    /// layout: arrays that disagree with their declared sizes, an LB=0 to 2
    /// block carrying a second table, an LB=3 or 4 block without one or whose
    /// tables share no energy range, or an LB=8 block between two reactions.
    pub malformed_blocks: usize,
    /// Per (nuclide, reaction kind), the share of the rate over the flux range
    /// that comes from energies where the evaluation states a nonzero variance
    /// for that reaction. The rate is the fold's own: dilute on a dilute run,
    /// and the shielded rate the run used on a self-shielded one. Below one
    /// means part of the rate carries no stated uncertainty, so it dilutes the
    /// relative sigma. An interval a covariance grid spans with a variance of
    /// zero counts as uncovered: it states no uncertainty either. In [0, 1]
    /// whatever rate the covariance is divided by. On a tallied run it is a
    /// share of the dilute rate over the tally spectrum, and the covered share
    /// of the tallied rate is not computed. See [`Info::partials_above_rate`]
    /// for when that rate disagrees with the partials. Exact under the flat
    /// within-group weight; under `Weighting::OneOverE` a group a covariance
    /// edge cuts is split by energy width rather than lethargy on a nuclide
    /// the collapse took dilute, so the share is off there. No entry for a
    /// channel whose rate over the flux range is zero. Every consumed
    /// self-covariance block counts where its own diagonal is nonzero,
    /// relative (LB=1 to 6), absolute (LB=0) and short-range (LB=8) alike.
    pub rate_fraction_covered: BTreeMap<(String, String), f64>,
    /// Per (nuclide, reaction kind), where the partial rates a relative
    /// covariance block was weighted with, zero variance intervals included,
    /// add up to more than the rate it was divided by, their ratio to it. Each
    /// is a channel whose relative sigma is overstated, because the two were
    /// computed different ways: a tallied rate against partials weighted flat
    /// within each bin, a grafted `(n,n')` whose rate is the metastables'
    /// MF=10 production while its partials are MT 4's, or the `1/E`
    /// within-group weight with a covariance edge inside a group. Its
    /// `rate_fraction_covered` is of the fold's own rate, not of the listed
    /// one, and under the `1/E` weight it is off as well wherever an edge cuts
    /// a group. On a derived channel the partials of the reactions each NC
    /// block names are checked the same way, and the check is also that they
    /// add up to the reaction it derives over the block's range, a sum above
    /// it landing here.
    pub partials_above_rate: BTreeMap<(String, String), f64>,
    /// Per (nuclide, reaction kind), where a relative block's grid spans the
    /// whole flux range and its partial rates add up to less than the rate it
    /// was divided by, their ratio to it: a channel whose relative sigma is
    /// understated. A grid that stops short of the flux range cannot be
    /// checked this way, since rate from outside it rightly leaves its
    /// partials short. A derived channel whose NC block names reactions that
    /// add up to less than the one it derives lands here too: ENDF/B-VIII.1
    /// O16 `(n,d)` above 20 MeV, where its cross section holds MT 660 to 669
    /// and the block names 650 to 659.
    pub partials_below_rate: BTreeMap<(String, String), f64>,
    /// Per (nuclide, reaction kind), where a channel derived through an NC
    /// block names two reactions with opposite signs, each with a variance
    /// block of its own, and the evaluation states no covariance between
    /// them, those pairs by kind. The absent block is read as zero, since
    /// ENDF-102 33.3.2 a.1 lets a tape leave a zero covariance unstated, and
    /// with opposing signs that reading sets the sigma: FENDL-3.2d and
    /// TENDL-2017 H2 `(n,2n)` = `σ_1 - σ_2 - σ_102` folds to about 22% at 14
    /// MeV and thousands of percent near threshold. The tape's literal
    /// statement, so reported rather than altered, and counted as a gap.
    pub derived_opposing_uncorrelated: BTreeMap<(String, String), BTreeSet<(String, String)>>,
    /// Per (nuclide, lumped MT), a lumped reaction (MT 851-870) with several
    /// components, whose covariance ENDF-102 33.2.3 states for their sum and
    /// for none of them, with the components by kind (`MT<n>` for one that
    /// is not a channel). A lump with a single component is that component
    /// and is folded as its covariance, and a lump an LTY=0 block names is
    /// folded through that derivation, with its cross section the sum of its
    /// components' (ENDF/B-VIII.1 and TENDL-2017 U235 and U238 MT 4 is MT 51
    /// plus MT 851). Listed only where neither holds, since giving the sum's
    /// covariance to a component would be an assumption, for a lump with
    /// blocks of its own where the fold reaches a component, or the redundant
    /// reaction holding one as a level: ENDF/B-VIII.1, FENDL-3.2d and
    /// JEFF-4.0 W180 to W186 give `(n,2n)` only as MT 852, the sum of MT 16
    /// and 41. Counted as a gap.
    pub lumped_covariance_not_assignable: BTreeMap<(String, i32), BTreeSet<String>>,
    /// Mean of the per-channel shares in [`Info::rate_fraction_covered`],
    /// weighted by the production each channel drove (the rate this run used
    /// times parent density): the share of the production driven from
    /// energies where a covariance states a nonzero variance.
    ///
    /// `None` for a decay-only schedule, which drove no production, and on a
    /// transport run, where that share is not computed. There the per-channel
    /// shares are of the dilute rate over the tally spectrum, and
    /// self-shielding in the transport moves rate out of the resonance range,
    /// where capture blocks often state zero, so weighting them by the tallied
    /// production would give a figure that is not the share its name claims.
    /// `None` under `Weighting::OneOverE`, whose shares are not exact where a
    /// covariance edge cuts a group (see [`Info::rate_fraction_covered`]).
    /// `None` too where a channel is listed in [`Info::partials_above_rate`] or
    /// [`Info::partials_below_rate`]: its rate is not the one its partials and
    /// share are of. The `(n,n')` of a nuclide with a metastable is the known
    /// case, its rate replaced by the MF=10 partials to the metastables while
    /// its covariance and share are MT 4's.
    ///
    /// The number to read before any sigma here, and not the same question as
    /// how many nuclides carry MF=33: an evaluation can state covariance for
    /// every isotope in the material and none of it for the channel making the
    /// product of interest, which leaves the count reading as full coverage
    /// while the ensemble perturbs almost nothing.
    pub rate_fraction_covered_total: Option<f64>,
    /// Nuclides the material can populate whose folded covariance was not
    /// positive semi-definite on at least one spectrum, past round-off, and
    /// had its negative eigenvalues clipped to be sampled, with a channel of
    /// that matrix a draw can move (a positive rate on a spectrum the
    /// schedule irradiates with).
    ///
    /// Populated means `yani::populated_nuclides` bounds the nuclide at or
    /// above [`crate::DENSITY_FLOOR`] over the schedule at nominal rates. The
    /// fold covers every chain nuclide with data, which from almost any
    /// composition is the chain's whole closure, and in the nominal solve a
    /// nuclide the bound leaves out cannot move any density by as much as
    /// that floor. A replica's lognormal rates on a wide channel can sit
    /// orders above nominal, so the bound does not hold for every replica.
    /// Past round-off means the smallest eigenvalue of the correlation matrix
    /// is below `-m * 1e-12`, `m` the number of channels with a positive
    /// stated variance (see `covariance_sample::REPAIR_TOLERANCE`), or a
    /// channel is stated with a negative variance, or with a zero one and a
    /// covariance to another channel.
    ///
    /// A gap: clipping only ever adds variance, so each of these was sampled
    /// wider than its evaluation states. How much is in `covariance_repairs`,
    /// which also keeps the repairs of populated nuclides no draw can move.
    pub covariance_repaired: BTreeSet<String>,
    /// One record per repaired (populated nuclide, spectrum): the eigenvalues,
    /// the share of the stated variance the clipping added, and every
    /// channel's evaluated sigma beside the sigma it was sampled at. A repair
    /// of a nuclide outside the populated bound has no record here; its
    /// nuclide is in `covariance_repaired_outside_bound`.
    pub covariance_repairs: Vec<Repair>,
    /// Nuclides outside the populated bound whose covariance needed a repair
    /// on some spectrum, with a channel a draw can move there (a positive rate
    /// on a spectrum the schedule irradiates with). A gap: the bound holds at
    /// nominal rates only, a replica's lognormal draw can populate them, and
    /// whether one did is not something the stepper's end-of-step densities
    /// can settle, since the solve applies every reachable nuclide's rates
    /// within a step. Their per-channel records are not kept.
    pub covariance_repaired_outside_bound: BTreeSet<String>,
    /// The largest `sampled / evaluated - 1` over the repaired channels a draw
    /// can move: a populated nuclide, present at the start or produced, with a
    /// positive rate on a spectrum the schedule irradiates with. Zero
    /// with no such repair; infinite when a repair gave a spread to a channel
    /// whose stated variance is zero or negative.
    pub worst_sigma_inflation: f64,
    /// The weighted mean of `sampled / evaluated - 1` over every sampled
    /// channel of a populated nuclide, each weighted by its unit-flux rate
    /// times its spectrum's fluence in the schedule times its parent's
    /// initial density, so a repair on a channel nothing went through reads
    /// as nothing and one on the channels that carry the reactions reads in
    /// full, however wide the channels beside them. The weight is the initial
    /// composition's, so this covers first-generation reactions only: a
    /// nuclide the material starts without carries no weight, and its repairs
    /// are in `worst_sigma_inflation` and `covariance_repairs`. On a matrix
    /// that needed no repair a channel's sampled sigma differs from the
    /// evaluated one by the decomposition's round-off, which shows here as it
    /// is. Infinite when a weighted channel with no evaluated sigma was
    /// sampled with a spread; `None` when no weighted channel has an
    /// evaluated sigma.
    pub rate_weighted_sigma_inflation: Option<f64>,
    /// Sampled channels of populated nuclides with a positive rate on a
    /// spectrum the schedule irradiates with, keyed by (nuclide, kind), whose
    /// folded relative sigma `sqrt(C_ii)`, as evaluated and before any repair,
    /// was at least one, with the largest over the spectra.
    ///
    /// At that width the spread depends on the distribution used to carry the
    /// evaluation's two moments (here a lognormal), not on the evaluation
    /// alone, so a sigma these dominate is partly this code's choice.
    pub sigma_at_least_one: BTreeMap<(String, String), f64>,
    /// The subset at ten or more.
    pub sigma_at_least_ten: BTreeMap<(String, String), f64>,
    /// The same as `sigma_at_least_one` for the nuclides outside the
    /// populated bound: sampled channels with a positive rate on a spectrum
    /// the schedule irradiates with, evaluated at one or more. Not a gap on
    /// the nominal bound, but named because a replica's draw on exactly such
    /// a channel can sit orders above nominal and populate the nuclide. The
    /// ten-or-more subset reads off the values.
    pub sigma_at_least_one_outside_bound: BTreeMap<(String, String), f64>,
    /// Cross-section rate draws made, one per perturbed channel per spectrum
    /// per replica.
    ///
    /// Each is read off one draw of the nuclide's cross sections, whose
    /// relative cells are lognormal multipliers, so a channel reading only
    /// those with positive coefficients cannot go negative.
    pub rates_sampled: usize,
    /// Cross-section rate draws that came out negative and were floored at
    /// zero: a channel that subtracts reactions (an NC derivation), or reads
    /// an absolute or short-range shift, which is additive.
    pub rates_floored: usize,
    /// Spectra that carried a per-bin flux sigma, and those that did not.
    ///
    /// A spectrum taken from a published reference set has no stated error, so
    /// it contributes nothing and that has to be visible rather than read as a
    /// flux known exactly.
    pub spectra_with_flux_sigma: usize,
    pub spectra_without_flux_sigma: usize,
    /// Flux bins drawn, one per bin per spectrum with a stated error per
    /// replica. Each is a lognormal factor with mean one, so none can go
    /// negative and none is floored.
    pub flux_bins_sampled: usize,
    /// Spectra, by index, whose stated flux covariance is not a lognormal's,
    /// with how far the sampled covariance is from it. Not a gap: see
    /// [`crate::covariance_sample::LognormalLimit`].
    pub flux_lognormal_not_carried: BTreeMap<usize, crate::covariance_sample::LognormalLimit>,
    /// Nuclides whose evaluated relative covariance is not a lognormal's,
    /// with how far the sampled covariance is from it. Not a gap: it is a
    /// property of the distribution that carries the evaluation's two
    /// moments, not a defect of the data, and the effect on each channel is
    /// also in its sampled sigma beside the evaluated one.
    pub lognormal_not_carried: BTreeMap<String, crate::covariance_sample::LognormalLimit>,
    /// Unstable nuclides the material can reach whose half-life was perturbed.
    pub half_lives_perturbed: BTreeSet<String>,
    /// Unstable nuclides the material can reach whose evaluation states no
    /// half-life uncertainty, so their half-life was held at nominal.
    ///
    /// Not a claim that the half-life is exact: the evaluation said nothing.
    pub no_half_life_uncertainty: BTreeSet<String>,
    /// Reachable unstable nuclides whose evaluation states a half-life sigma
    /// no draw can carry (not finite, or not finite relative to the
    /// half-life), so their half-life was held at nominal.
    pub half_life_uncertainty_not_carried: BTreeSet<String>,
    /// Half-life draws made, one per perturbed nuclide per replica.
    ///
    /// Each is a lognormal matched to the evaluation's mean and sigma, so none
    /// can go non-positive and there is no floor to count.
    pub half_lives_sampled: usize,
    /// Reachable parents whose decay branching was perturbed: two modes whose
    /// sum rule fixes how the stated sigma is shared.
    pub decay_branchings_perturbed: BTreeSet<String>,
    /// Reachable parents with several decay modes and no stated sigma on any,
    /// held at nominal. Not a claim that the split is exact.
    pub no_decay_branching_uncertainty: BTreeSet<String>,
    /// Reachable parents with three or more modes and a stated sigma, held at
    /// nominal: MT=457 gives no covariance between the modes, and with more
    /// than one degree of freedom the sum rule does not supply it.
    pub decay_branchings_three_or_more_modes: BTreeSet<String>,
    /// Reachable two-mode parents whose modes state different sigmas, held at
    /// nominal: a sum rule leaves room for one.
    pub decay_branchings_unequal_sigmas: BTreeSet<String>,
    /// Reachable two-mode parents whose smaller ratio is under five sigmas,
    /// held at nominal: a Gaussian that wide would have to be truncated at
    /// zero, which the evaluation does not describe.
    pub decay_branchings_too_wide: BTreeSet<String>,
    /// Branching draws that fell outside `[0, T]` and were clamped, and draws
    /// made, one per perturbed parent per replica.
    pub decay_branchings_floored: usize,
    pub decay_branchings_sampled: usize,
    /// Reachable unstable nuclides whose decay energy was perturbed.
    pub decay_energies_perturbed: BTreeSet<String>,
    /// Reachable unstable nuclides with a decay energy but no stated sigma on
    /// it, held at nominal. Not a claim that it is exact.
    pub no_decay_energy_uncertainty: BTreeSet<String>,
    /// Reachable unstable nuclides with a decay-energy sigma, on the total or
    /// on a component, that no draw can carry: stated on an energy of zero,
    /// or not finite. That energy was held at nominal; any other component of
    /// the same nuclide with a usable sigma was still drawn.
    pub decay_energy_uncertainty_not_carried: BTreeSet<String>,
    /// Tallied rates sampled statistically, the totals and partials together;
    /// zero off the transport path or with the source off.
    pub statistical_rates: usize,
    /// Statistically drawn rates that came out negative and were floored.
    pub statistical_floored: usize,
    pub statistical_sampled: usize,
    /// Inputs this run held at their nominal values, for the record.
    ///
    /// Every input the answer depends on and no source here samples, whether
    /// the data carries an uncertainty for it or not. MF=33 blocks that were
    /// present but could not be used are counted in `skipped_cross_material`,
    /// `skipped_other_file`, `skipped_nc`, `unsupported_layouts` and
    /// `malformed_blocks`, and lumped reactions that could not be given to a
    /// reaction in `lumped_covariance_not_assignable`. A pair stored in both
    /// orientations is used once, and a second copy that disagrees with the
    /// first is reported in `mirrored_disagree`. A block on a reaction the
    /// fold does not reach (neither a channel nor one a channel's NC
    /// derivation names) is neither listed nor counted: the chain has no rate
    /// for it to be the uncertainty of. A partial-level section such as
    /// MT=600-849 is reached, and its blocks folded and counted, when an LTY=0
    /// NC block names it.
    pub not_perturbed: Vec<String>,
    /// Which sources this run perturbed, by name.
    pub sources: Vec<String>,
}

impl Info {
    /// `collapsed_flat` is whether the rates the fold was divided by are the
    /// collapse's, dilute or self-shielded, under the flat within-group
    /// weight, whose shares the fold computed under the same flux. Only then,
    /// and with no channel's partials disagreeing with its rate, is the
    /// production total the covered share; a tallied rate is not. An error
    /// when a share is not one, see [`Coverage::rate_fraction_total`].
    pub(crate) fn from_fold(
        coverage: &Coverage,
        sigmas: &SigmaReport,
        collapsed_flat: bool,
    ) -> Result<Self, String> {
        let total = coverage.rate_fraction_total()?;
        Ok(Self {
            perturbed: coverage.covered.clone(),
            no_covariance_data: coverage.without_data.clone(),
            skipped_cross_material: coverage.skipped_cross_material.clone(),
            skipped_other_file: coverage.skipped_other_file.clone(),
            mirrored_disagree: coverage.mirrored_disagree.clone(),
            skipped_nc: coverage.skipped_nc.clone(),
            unsupported_layouts: coverage.unsupported_layouts.clone(),
            malformed_blocks: coverage.malformed,
            rate_fraction_covered: coverage.rate_fraction_covered.clone(),
            partials_above_rate: coverage.partials_above_rate.clone(),
            partials_below_rate: coverage.partials_below_rate.clone(),
            derived_opposing_uncorrelated: coverage.derived_opposing_uncorrelated.clone(),
            lumped_covariance_not_assignable: coverage.lumped_covariance_not_assignable.clone(),
            rate_fraction_covered_total: total.filter(|_| {
                collapsed_flat
                    && coverage.partials_above_rate.is_empty()
                    && coverage.partials_below_rate.is_empty()
            }),
            // Distinct nuclides, not a count over spectra: one evaluation
            // folded against three spectra is one evaluation that needed it.
            covariance_repaired: sigmas.repaired.clone(),
            covariance_repairs: sigmas.repairs.clone(),
            covariance_repaired_outside_bound: sigmas.repaired_outside_bound.clone(),
            worst_sigma_inflation: sigmas.worst_sigma_inflation,
            rate_weighted_sigma_inflation: sigmas.rate_weighted_sigma_inflation(),
            sigma_at_least_one: sigmas.sigma_at_least_one.clone(),
            sigma_at_least_ten: sigmas.sigma_at_least_ten.clone(),
            sigma_at_least_one_outside_bound: sigmas.sigma_at_least_one_outside_bound.clone(),
            not_perturbed: [
                "fission yield",
                "isomeric branching (MF=9/MF=10)",
                "covariance with another evaluation (MAT1 naming another material)",
                "covariance with a quantity that is not a cross section (MF=33 XMF1 not 0 or 3)",
                "NC-derived covariance that cannot be derived (MF=33 NC LTY 1-4, or LTY=0 in skipped_nc)",
                "lumped-reaction covariance of several components no derivation names (MF=33 MT=851-870, in lumped_covariance_not_assignable)",
                "resonance-parameter covariance (MF=32)",
                "decay photon line energy and intensity (MF=8 MT=457)",
                "decay photon continuum normalisation and shape (MF=8 MT=457 continuum and its covariance)",
                "photon attenuation coefficient (XCOM)",
                "air energy-absorption coefficient (NIST SRD 126)",
                "fluence-to-dose coefficient (ICRP-116)",
                "contact-dose build-up factor",
                "material composition",
                "material density",
                "natural isotopic abundance",
                "atomic mass (AME2020)",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ..Default::default()
        })
    }

    pub(crate) fn add_flux_coverage(&mut self, c: &crate::flux_uncertainty::FluxCoverage) {
        self.spectra_with_flux_sigma = c.spectra_with_sigma;
        self.spectra_without_flux_sigma = c.spectra_without_sigma;
        self.flux_bins_sampled = c.bins_sampled;
        self.flux_lognormal_not_carried = c.lognormal_not_carried.clone();
    }

    /// Whether anything was left out, or is inconsistent, that a reader should
    /// know about.
    pub fn has_gaps(&self) -> bool {
        !self.no_covariance_data.is_empty()
            || !self.skipped_cross_material.is_empty()
            || !self.skipped_other_file.is_empty()
            || !self.mirrored_disagree.is_empty()
            || !self.skipped_nc.is_empty()
            || !self.unsupported_layouts.is_empty()
            || self.malformed_blocks > 0
            || !self.partials_above_rate.is_empty()
            || !self.partials_below_rate.is_empty()
            || !self.derived_opposing_uncorrelated.is_empty()
            || !self.lumped_covariance_not_assignable.is_empty()
            || !self.covariance_repaired.is_empty()
            || !self.covariance_repaired_outside_bound.is_empty()
            || self.spectra_without_flux_sigma > 0
            || !self.no_half_life_uncertainty.is_empty()
            || !self.no_decay_branching_uncertainty.is_empty()
            || !self.decay_branchings_three_or_more_modes.is_empty()
            || !self.decay_branchings_unequal_sigmas.is_empty()
            || !self.decay_branchings_too_wide.is_empty()
            || !self.no_decay_energy_uncertainty.is_empty()
            || !self.half_life_uncertainty_not_carried.is_empty()
            || !self.decay_energy_uncertainty_not_carried.is_empty()
    }
}

/// Running mean and second moment for one nuclide at one step.
///
/// Welford rather than a two-pass sum: for a run whose evaluations carry no
/// covariance every replica is the same inventory, and Welford returns a
/// standard deviation of exactly zero there, where `sum / n` would leave the
/// mean a rounding off the sample and report a sigma a few ulps wide. On a log
/// axis that reads as a real one. Shared with `derived`, so a density sigma and
/// a decay-heat sigma are the same statistic and not two spellings of it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Moments {
    pub(crate) n: u64,
    pub(crate) mean: f64,
    m2: f64,
}

impl Moments {
    pub(crate) fn update(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// The sample standard deviation, or zero below two samples.
    pub(crate) fn std_dev(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        (self.m2 / (self.n - 1) as f64).max(0.0).sqrt()
    }
}

/// Accumulates the ensemble of perturbed inventories.
///
/// One [`Moments`] per (step, nuclide), plus the per-replica densities so a
/// derived quantity can be evaluated per sample rather than from the mean. That
/// distinction is load-bearing for anything summed over nuclides: evaluating
/// from the mean throws away the inter-nuclide correlations the resampling
/// exists to capture.
#[derive(Debug, Default, Clone)]
pub struct Ensemble {
    /// `[step][nuclide]`, one entry per schedule step.
    moments: Vec<HashMap<String, Moments>>,
    /// `[replica][step][nuclide]`.
    samples: Vec<Vec<HashMap<String, f64>>>,
    /// `[replica]`: the half-lives [s] that replica was solved with, for the
    /// nuclides whose half-life was perturbed. Empty maps when half-lives were
    /// not perturbed.
    half_lives: Vec<HashMap<String, f64>>,
    /// Where the uncertainty comes from, when it was asked for.
    pub attribution: Option<Attribution>,
    /// The seed decay energies are drawn from, per replica and nuclide, when
    /// they were perturbed. They do not change the inventory, so they are
    /// drawn where decay heat is evaluated rather than stored.
    pub decay_energy_seed: Option<u64>,
}

/// Where an inventory's uncertainty comes from.
///
/// Two levels, which answer different questions:
///
/// - `by_source` is exact: the ensemble re-run with each source alone, so a
///   nuclide's statistical and nuclear-data variances are measured the same
///   way the total is. Sources are independent, so they sum to the total up to
///   interaction and sampling noise, which is the unattributed residual.
/// - `contributors` is first order: within the cross sections, the
///   half-lives and the decay branchings, one deterministic solve per nuclide
///   (and per reaction) gives its sensitivity, and its variance is that
///   squared against its own stated uncertainty. It says which evaluation to
///   look at, not the total, which is always the resampled one.
#[derive(Debug, Clone, Default)]
pub struct Attribution {
    /// Source name -> `[step][nuclide]` variance, from that source alone.
    pub by_source: BTreeMap<String, Vec<HashMap<String, f64>>>,
    /// First-order contributions, the largest-reaching first.
    pub contributors: Vec<Contributor>,
}

/// One first-order contribution to the inventory variance.
#[derive(Debug, Clone)]
pub struct Contributor {
    /// The source it belongs to, e.g. `cross_sections`.
    pub source: String,
    /// The nuclide whose data it is.
    pub nuclide: String,
    /// The reaction, for one cross-section channel alone; `None` for the
    /// nuclide's whole evaluation (every channel with its correlations) or for
    /// a half-life. A `decay_branching` contributor is a two-mode parent's
    /// one degree of freedom, so `None` too.
    pub reaction: Option<String>,
    /// `[step][nuclide]` variance it contributes, only where it is non-zero.
    pub variance: Vec<HashMap<String, f64>>,
}

impl Ensemble {
    pub(crate) fn new(n_steps: usize) -> Self {
        Self {
            moments: vec![HashMap::new(); n_steps],
            samples: Vec::new(),
            half_lives: Vec::new(),
            attribution: None,
            decay_energy_seed: None,
        }
    }

    /// Record one replica's per-step inventories, solved with nominal
    /// half-lives.
    #[cfg(test)]
    pub(crate) fn push(&mut self, per_step: Vec<HashMap<String, f64>>) {
        self.push_with_half_lives(per_step, HashMap::new());
    }

    /// Record one replica's per-step inventories and the perturbed half-lives
    /// it was solved with, so a quantity derived from its inventory uses them
    /// too.
    pub(crate) fn push_with_half_lives(
        &mut self,
        per_step: Vec<HashMap<String, f64>>,
        half_lives: HashMap<String, f64>,
    ) {
        self.half_lives.push(half_lives);
        for (step, densities) in per_step.iter().enumerate() {
            let Some(slot) = self.moments.get_mut(step) else {
                continue;
            };
            for (name, &value) in densities {
                slot.entry(name.clone()).or_default().update(value);
            }
        }
        self.samples.push(per_step);
    }

    /// A nuclide absent from a replica is a zero in that replica, not a gap.
    ///
    /// The stepper drops anything at or below its density floor, so a nuclide
    /// present in one replica and absent from another differs by less than the
    /// floor. Folding those in as zeros keeps every moment over the same number
    /// of samples, without which the variance would be computed over a
    /// different subset per nuclide.
    pub(crate) fn fold_absences(&mut self) {
        let replicas = self.samples.len() as u64;
        for slot in &mut self.moments {
            for m in slot.values_mut() {
                while m.n < replicas {
                    m.update(0.0);
                }
            }
        }
    }

    pub fn replicas(&self) -> usize {
        self.samples.len()
    }

    /// Standard deviation of every nuclide's density at `step`.
    pub fn std_dev_at(&self, step: usize) -> HashMap<String, f64> {
        self.moments
            .get(step)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.std_dev())).collect())
            .unwrap_or_default()
    }

    /// One nuclide's standard deviation across every step.
    pub fn std_dev_evolution(&self, nuclide: &str) -> Vec<f64> {
        self.moments
            .iter()
            .map(|m| m.get(nuclide).map_or(0.0, Moments::std_dev))
            .collect()
    }

    /// One nuclide's density in every replica at `step`, for a derived quantity
    /// that has to be evaluated per sample.
    pub fn samples_at(&self, step: usize, nuclide: &str) -> Vec<f64> {
        self.samples
            .iter()
            .map(|r| {
                r.get(step)
                    .and_then(|s| s.get(nuclide))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// The perturbed half-lives [s] each replica was solved with, by replica.
    ///
    /// Empty maps when half-lives were not perturbed. A nuclide absent from a
    /// replica's map kept its nominal half-life there.
    pub fn half_lives(&self) -> &[HashMap<String, f64>] {
        &self.half_lives
    }

    /// Every replica's full inventory at `step`.
    ///
    /// The shape a derived quantity wants: `activity`, `decay_heat` and the
    /// decay-photon spectrum are all functions of a whole inventory, and
    /// evaluating them once per entry here is what keeps their uncertainty
    /// correlated across nuclides.
    pub fn inventories_at(&self, step: usize) -> Vec<&HashMap<String, f64>> {
        self.samples.iter().filter_map(|r| r.get(step)).collect()
    }

    /// The nuclides worth judging convergence on at the final step.
    ///
    /// Everything within [`SIGNIFICANCE`] of the largest density. A trace many
    /// decades down carries no significant figures anyway, and letting its
    /// sigma decide when to stop would run the cap every time.
    fn significant(&self) -> Vec<String> {
        let Some(last) = self.moments.last() else {
            return Vec::new();
        };
        let max = last.values().map(|m| m.mean.abs()).fold(0.0_f64, f64::max);
        if max <= 0.0 {
            return Vec::new();
        }
        let mut out: Vec<String> = last
            .iter()
            .filter(|(_, m)| m.mean.abs() >= max * SIGNIFICANCE)
            .map(|(k, _)| k.clone())
            .collect();
        out.sort();
        out
    }

    /// The final-step sigmas of the significant nuclides, for comparison
    /// against the previous block.
    pub(crate) fn convergence_probe(&self) -> BTreeMap<String, f64> {
        let Some(last) = self.moments.last() else {
            return BTreeMap::new();
        };
        self.significant()
            .into_iter()
            .filter_map(|k| last.get(&k).map(|m| (k, m.std_dev())))
            .collect()
    }
}

/// Whether the sigmas have stopped moving between two blocks.
///
/// Compared on relative movement per nuclide, and judged on the WORST of them
/// rather than an average: an average lets a single unconverged nuclide hide
/// behind a hundred settled ones. Nuclides whose sigma is zero in both probes
/// are skipped, since a relative change is undefined there and a genuinely
/// deterministic nuclide (a stable one, or one no perturbed channel feeds) is
/// converged the moment it is seen.
pub(crate) fn settled(previous: &BTreeMap<String, f64>, current: &BTreeMap<String, f64>) -> bool {
    if previous.is_empty() || current.is_empty() {
        return false;
    }
    let mut compared = 0;
    for (name, &now) in current {
        let Some(&before) = previous.get(name) else {
            return false;
        };
        let scale = now.abs().max(before.abs());
        if scale == 0.0 {
            continue;
        }
        compared += 1;
        if (now - before).abs() / scale > TOLERANCE {
            return false;
        }
    }
    compared > 0
}

/// The per-step nuclide densities of one replica.
pub(crate) fn densities_of(materials: &[Material]) -> Vec<HashMap<String, f64>> {
    materials.iter().map(|m| m.nuclides.clone()).collect()
}

/// Keeps the half-life streams clear of the per-nuclide cross-section streams,
/// which are keyed on the same name hash without it.
pub(crate) const HALF_LIFE_STREAM: u32 = 0x4A1F_11FE;

/// Whether an evaluation states `sigma`: a stored 0.0 is "not stated", the
/// same as a null, never an exact value.
fn stated(sigma: Option<f64>) -> bool {
    sigma.is_some_and(|s| s > 0.0)
}

/// `sigma` when a lognormal draw of `value` can carry it: stated, on a
/// positive value, with a relative sigma that is positive and whose square
/// is finite.
///
/// A mean of zero for a quantity that cannot go negative is zero in every
/// draw, and an infinite sigma has no lognormal, so either is reported as
/// not carried rather than drawn as a NaN or silently held. The square is
/// checked, not just the ratio, because the lognormal is built from
/// `ln(1 + relative^2)` and a ratio above about 1e154 overflows there into
/// the same NaN.
fn carried(value: f64, sigma: Option<f64>) -> Option<f64> {
    let sigma = sigma.filter(|s| *s > 0.0 && s.is_finite())?;
    let relative = sigma / value;
    (value > 0.0 && relative > 0.0 && (relative * relative).is_finite()).then_some(sigma)
}

/// `(name, half-life, sigma)` for one nuclide whose half-life is drawn.
pub(crate) type HalfLifeSigma = (String, f64, f64);

/// The unstable nuclides of `chain` split three ways: those with a half-life
/// sigma a draw can carry, as `(name, half-life, sigma)`; those that state
/// none; and those that state one no draw can carry.
pub(crate) fn half_life_candidates(
    chain: &HashMap<String, yani::ChainNuclide>,
) -> (Vec<HalfLifeSigma>, BTreeSet<String>, BTreeSet<String>) {
    let mut with = Vec::new();
    let mut without = BTreeSet::new();
    let mut not_carried = BTreeSet::new();
    for (name, cn) in chain {
        let Some(t) = cn.half_life.filter(|t| *t > 0.0) else {
            continue;
        };
        if let Some(sigma) = carried(t, cn.half_life_uncertainty) {
            with.push((name.clone(), t, sigma));
        } else if stated(cn.half_life_uncertainty) {
            not_carried.insert(name.clone());
        } else {
            without.insert(name.clone());
        }
    }
    with.sort_by(|a, b| a.0.cmp(&b.0));
    (with, without, not_carried)
}

/// One replica's half-lives, from a stream keyed on `(seed, replica, nuclide)`.
///
/// Each is `T` times a lognormal multiplier with mean 1 and variance
/// `(sigma / T)^2`, so the draws have the evaluation's mean and standard
/// deviation exactly and are always positive. `T + sigma z` floored at zero
/// did neither once the sigma was a large share of `T`: JENDL-5.0 states 120
/// half-lives with a sigma above half of the value and 31 above all of it,
/// and every floored draw pulled the mean up.
///
/// Keyed on the name, like the cross sections, so a nuclide draws the same
/// half-life in every spectrum and every material of a run: one evaluation is
/// uncertain in one way wherever it is used.
pub(crate) fn sample_half_lives(
    candidates: &[(String, f64, f64)],
    base_seed: u64,
    replica: u64,
) -> HashMap<String, f64> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    candidates
        .iter()
        .map(|(name, t, sigma)| {
            let seed = yamc_rng::secondary_seed(
                replica_seed,
                crate::covariance_sample::name_ordinal(name) ^ HALF_LIFE_STREAM,
            );
            let mut state = yamc_rng::expand_seed(seed);
            let z = crate::covariance_sample::standard_normals(&mut state, 1)[0];
            (name.clone(), t * lognormal_multiplier(z, sigma / t))
        })
        .collect()
}

/// Keeps the decay-energy streams clear of every other per-nuclide stream.
pub(crate) const DECAY_ENERGY_STREAM: u32 = 0xDEC4_E6E1;

/// One replica's decay energy for one nuclide, and its components, drawn
/// from each component's own sigma, or from the total's where the data gives
/// no split. `None` when no sigma can be carried (see
/// [`has_decay_energy_sigma`]).
///
/// Each drawn energy is its nominal times [`lognormal_multiplier`], for the
/// reason [`sample_half_lives`] gives: the evaluation's mean and sigma exactly,
/// and never negative, where a Gaussian floored at zero raised the mean.
pub(crate) fn sample_decay_energy(
    cn: &yani::ChainNuclide,
    base_seed: u64,
    replica: u64,
) -> Option<(f64, [Option<yani::DecayEnergyComponent>; 3])> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    let seed = yamc_rng::secondary_seed(
        replica_seed,
        crate::covariance_sample::name_ordinal(&cn.name) ^ DECAY_ENERGY_STREAM,
    );
    if !has_decay_energy_sigma(cn) {
        return None;
    }
    let mut state = yamc_rng::expand_seed(seed);
    if components_state_sigma(cn) {
        let z = crate::covariance_sample::standard_normals(&mut state, 3);
        let mut parts = cn.decay_energy_components;
        let mut total = 0.0;
        for (part, z) in parts.iter_mut().zip(z) {
            if let Some(p) = part {
                if let Some(sigma) = carried(p.energy, p.uncertainty) {
                    p.energy *= lognormal_multiplier(z, sigma / p.energy);
                }
                total += p.energy;
            }
        }
        return Some((total, parts));
    }
    let sigma = carried(cn.decay_energy, cn.decay_energy_uncertainty)?;
    let z = crate::covariance_sample::standard_normals(&mut state, 1)[0];
    Some((
        cn.decay_energy * lognormal_multiplier(z, sigma / cn.decay_energy),
        cn.decay_energy_components,
    ))
}

/// Whether any component states a sigma, which makes the components, not the
/// total, what is drawn: the total's sigma is theirs combined.
fn components_state_sigma(cn: &yani::ChainNuclide) -> bool {
    cn.decay_energy_components
        .iter()
        .any(|c| c.is_some_and(|c| stated(c.uncertainty)))
}

/// Whether a nuclide's decay energy carries a sigma a draw can sample from.
pub(crate) fn has_decay_energy_sigma(cn: &yani::ChainNuclide) -> bool {
    if components_state_sigma(cn) {
        cn.decay_energy_components
            .iter()
            .any(|c| c.is_some_and(|c| carried(c.energy, c.uncertainty).is_some()))
    } else {
        carried(cn.decay_energy, cn.decay_energy_uncertainty).is_some()
    }
}

/// Whether a nuclide states a decay-energy sigma that no draw can carry, on
/// the total or a component: one on an energy of zero, or one not finite.
/// ENDF/B-VIII.1 has none of either, but the rule is to report such a sigma,
/// not absorb it.
pub(crate) fn has_decay_energy_sigma_not_carried(cn: &yani::ChainNuclide) -> bool {
    let unusable =
        |energy: f64, sigma: Option<f64>| stated(sigma) && carried(energy, sigma).is_none();
    if components_state_sigma(cn) {
        cn.decay_energy_components
            .iter()
            .any(|c| c.is_some_and(|c| unusable(c.energy, c.uncertainty)))
    } else {
        unusable(cn.decay_energy, cn.decay_energy_uncertainty)
    }
}

/// `chain` with the half-lives of `sampled` substituted.
pub(crate) fn with_half_lives(
    chain: &HashMap<String, yani::ChainNuclide>,
    sampled: &HashMap<String, f64>,
) -> HashMap<String, yani::ChainNuclide> {
    let mut out = chain.clone();
    for (name, t) in sampled {
        if let Some(cn) = out.get_mut(name) {
            set_half_life(cn, *t);
        }
    }
    out
}

/// Give a chain nuclide a different half-life, and everything stored in the
/// chain as a function of it.
///
/// Decay-source intensities are stored per atom per second, which is the
/// emission probability per decay times the decay constant (Co60's 1332 keV
/// line is `0.9998 * ln2 / T`), and a continuum's density per eV is the same
/// product. The per-decay probability is the decay scheme's and does not
/// change with the half-life, so the stored intensity scales as
/// `T_nominal / T`. Leaving it would evaluate a replica's photon
/// emission as `N_k lambda y` instead of `N_k lambda_k y`: at saturation
/// `N_k ~ 1 / lambda_k`, so the photon rate would inherit the half-life's
/// whole spread, which is the inconsistent-lambda inflation the per-replica
/// half-lives exist to prevent. Each line's intensity sigma is stored in the
/// same units and scales with it, so its ratio to the intensity stays the
/// tape's.
pub(crate) fn set_half_life(cn: &mut yani::ChainNuclide, half_life: f64) {
    if let Some(nominal) = cn.half_life.filter(|t| *t > 0.0 && half_life > 0.0) {
        let scale = nominal / half_life;
        for source in &mut cn.sources {
            source.scale_rates(scale);
        }
    }
    cn.half_life = Some(half_life);
}

/// The standard error of the sample standard deviation of `values`, or
/// `None` below four values, where the kurtosis it needs is not estimated.
///
/// `Var(s^2) = s^4 (2/(n-1) + kappa/n)` with `kappa` the sample excess
/// kurtosis, so a heavy-tailed output (a lognormal one, say) reads as less
/// settled than a Gaussian one at the same count, as it is. Then
/// `SE(s) = sqrt(Var(s^2)) / (2 s)` to first order. Identical values give
/// exactly zero.
pub fn std_dev_standard_error(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 4 {
        return None;
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let m2 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nf;
    if m2 <= 0.0 {
        return Some(0.0);
    }
    let m4 = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / nf;
    let kurtosis = m4 / (m2 * m2) - 3.0;
    let s2 = m2 * nf / (nf - 1.0);
    let var_s2 = (s2 * s2 * (2.0 / (nf - 1.0) + kurtosis / nf)).max(0.0);
    Some(var_s2.sqrt() / (2.0 * s2.sqrt()))
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_standard_error_needs_four_values_and_is_zero_without_spread() {
        assert_eq!(std_dev_standard_error(&[1.0, 2.0, 3.0]), None);
        assert_eq!(std_dev_standard_error(&[2.0; 8]), Some(0.0));
    }

    #[test]
    fn a_gaussian_standard_error_is_sigma_over_sqrt_2n() {
        // Box-Muller on a fixed stream: kurtosis near zero, so SE(s) is about
        // s / sqrt(2(n-1)).
        let mut state = yamc_rng::expand_seed(7);
        let z = crate::covariance_sample::standard_normals(&mut state, 20_000);
        let se = std_dev_standard_error(&z).unwrap();
        let want = 1.0 / (2.0 * 19_999.0_f64).sqrt();
        assert!((se / want - 1.0).abs() < 0.05, "{se} against {want}");
    }

    #[test]
    fn a_heavy_tail_reads_as_less_settled() {
        let mut state = yamc_rng::expand_seed(7);
        let z = crate::covariance_sample::standard_normals(&mut state, 4_000);
        let lognormal: Vec<f64> = z.iter().map(|z| (0.8 * z).exp()).collect();
        let s = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let relative = |v: &[f64]| std_dev_standard_error(v).unwrap() / s(v);
        assert!(relative(&lognormal) > 1.5 * relative(&z));
    }
    use super::*;

    /// Every run states the decay photon sources it holds at nominal, lines
    /// and continua both, so a reader does not take their silence for zero.
    #[test]
    fn the_report_names_the_photon_sources_held_at_nominal() {
        let info = Info::from_fold(&Coverage::default(), &SigmaReport::default(), true)
            .expect("no shares to check");
        for source in [
            "decay photon line energy and intensity (MF=8 MT=457)",
            "decay photon continuum normalisation and shape (MF=8 MT=457 continuum and its covariance)",
        ] {
            assert!(
                info.not_perturbed.iter().any(|s| s == source),
                "{source:?} missing from {:?}",
                info.not_perturbed
            );
        }
    }

    /// A new half-life rescales a continuum's density exactly as it rescales a
    /// line, so both keep their per-decay yield.
    #[test]
    fn a_half_life_rescales_lines_and_continua_alike() {
        let lambda = std::f64::consts::LN_2 / 318.0;
        let source = |distribution| yani::DecaySource {
            particle: "photon".to_string(),
            radiation: None,
            uncertainty: None,
            distribution,
        };
        let mut cn = yani::ChainNuclide {
            name: "Sm158".to_string(),
            half_life: Some(318.0),
            half_life_uncertainty: Some(1.8),
            decay_energy: 1.0e6,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: vec![],
            decays: vec![],
            fission_yields: None,
            sources: vec![
                source(yani::DecaySourceDistribution::Discrete {
                    energies: vec![2.0e5],
                    intensities: vec![0.5 * lambda],
                }),
                source(yani::DecaySourceDistribution::Tabular {
                    energies: vec![1.0e4, 1.0e6],
                    intensities: vec![3.0e-6 * lambda, 0.0],
                    interpolation: Some(yani::Interpolation::Histogram),
                }),
            ],
        };
        set_half_life(&mut cn, 636.0);
        let lambda_k = std::f64::consts::LN_2 / 636.0;
        let per_decay: Vec<f64> = cn
            .sources
            .iter()
            .map(|s| s.distribution.emission_rate().unwrap() / lambda_k)
            .collect();
        assert!((per_decay[0] - 0.5).abs() < 1e-14, "{per_decay:?}");
        assert!(
            (per_decay[1] - 3.0e-6 * 9.9e5).abs() < 1e-14,
            "{per_decay:?}"
        );
    }

    /// A line's intensity sigma is a rate like the intensity, so a new
    /// half-life moves both and leaves the tape's dRI/RI. The normalisation
    /// and the energy sigmas are per decay and stay.
    #[test]
    fn a_half_life_rescales_the_intensity_sigmas_with_the_lines() {
        let lambda = std::f64::consts::LN_2 / 318.0;
        let stated = yani::DecaySourceUncertainty {
            normalization: Some(0.27),
            normalization_uncertainty: Some(0.01),
            intensity_uncertainties: Some(vec![0.27 * 0.02 * lambda]),
            energy_uncertainties: Some(vec![30.0]),
            covariance: None,
        };
        let mut cn = yani::ChainNuclide {
            name: "W187".to_string(),
            half_life: Some(318.0),
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: vec![],
            decays: vec![],
            fission_yields: None,
            sources: vec![yani::DecaySource {
                particle: "photon".to_string(),
                radiation: Some("gamma".to_string()),
                distribution: yani::DecaySourceDistribution::Discrete {
                    energies: vec![6.858e5],
                    intensities: vec![0.27 * 0.5 * lambda],
                },
                uncertainty: Some(std::sync::Arc::new(stated.clone())),
            }],
        };
        let shared = cn.sources[0].uncertainty.clone().unwrap();
        set_half_life(&mut cn, 636.0);
        let scaled = cn.sources[0].uncertainty.as_deref().unwrap();
        let lambda_k = std::f64::consts::LN_2 / 636.0;
        let sigma = scaled.intensity_uncertainties.as_ref().unwrap()[0];
        assert!((sigma / lambda_k - 0.27 * 0.02).abs() < 1e-15, "{sigma}");
        assert_eq!(scaled.normalization, stated.normalization);
        assert_eq!(
            scaled.normalization_uncertainty,
            stated.normalization_uncertainty
        );
        assert_eq!(scaled.energy_uncertainties, stated.energy_uncertainties);
        // The nominal chain the replica was cloned from keeps its own sigmas.
        assert_eq!(*shared, stated);
    }

    /// The production total weights the fold's shares by the production the
    /// run drove, which is the covered share only when that production is the
    /// rate the shares are of: the collapse's, dilute or shielded. A tallied
    /// run reports no total rather than a figure that is not the share its
    /// name claims, and keeps the per-channel shares, which are.
    #[test]
    fn only_a_collapsed_run_reports_a_production_total() {
        let coverage = Coverage {
            rate_fraction_covered: BTreeMap::from([(("W186".into(), "(n,gamma)".into()), 0.25)]),
            covered_production: 1.0,
            total_production: 4.0,
            ..Default::default()
        };
        let sigmas = SigmaReport::default();
        let collapsed = Info::from_fold(&coverage, &sigmas, true).expect("shares are shares");
        assert_eq!(collapsed.rate_fraction_covered_total, Some(0.25));
        let tallied = Info::from_fold(&coverage, &sigmas, false).expect("shares are shares");
        assert_eq!(tallied.rate_fraction_covered_total, None);
        assert_eq!(
            tallied.rate_fraction_covered,
            coverage.rate_fraction_covered
        );
    }

    /// A lumped reaction the fold could not give to any reaction reaches the
    /// report as it is, and is a gap by itself.
    #[test]
    fn an_unassignable_lump_is_carried_into_the_report_as_a_gap() {
        let lump = (
            ("W186".to_string(), 852),
            BTreeSet::from(["(n,2n)".to_string(), "(n,2np)".to_string()]),
        );
        let coverage = Coverage {
            lumped_covariance_not_assignable: BTreeMap::from([lump]),
            ..Default::default()
        };
        let info =
            Info::from_fold(&coverage, &SigmaReport::default(), true).expect("shares are shares");
        assert_eq!(
            info.lumped_covariance_not_assignable,
            coverage.lumped_covariance_not_assignable
        );
        assert!(info.has_gaps());
        assert!(
            !Info::from_fold(&Coverage::default(), &SigmaReport::default(), true)
                .expect("shares are shares")
                .has_gaps()
        );
    }

    /// A repair a draw can move is a gap: the sampled spread is wider than
    /// the evaluation's. One on two spectra is one repaired nuclide.
    #[test]
    fn a_repair_counts_once_per_nuclide_and_is_a_gap() {
        let repair = |spectrum| Repair {
            nuclide: "W182".to_string(),
            spectrum,
            lambda_min: -1.0e-4,
            lambda_max: 1.0e-2,
            clipped_fraction: 0.01,
            channels: Vec::new(),
        };
        let mut sigmas = SigmaReport::default();
        sigmas.repairs = vec![repair(0), repair(1)];
        sigmas.repaired = BTreeSet::from(["W182".to_string()]);
        let info =
            Info::from_fold(&Coverage::default(), &sigmas, true).expect("no shares to check");
        assert_eq!(
            info.covariance_repaired,
            BTreeSet::from(["W182".to_string()])
        );
        assert_eq!(info.covariance_repairs.len(), 2);
        assert!(info.has_gaps());

        // Outside the populated bound a repair is named, and is a gap too:
        // the bound is nominal and a replica's draw can populate the nuclide.
        let mut outside = SigmaReport::default();
        outside.repaired_outside_bound = BTreeSet::from(["Xe135".to_string()]);
        let info =
            Info::from_fold(&Coverage::default(), &outside, true).expect("no shares to check");
        assert_eq!(
            info.covariance_repaired_outside_bound,
            BTreeSet::from(["Xe135".to_string()])
        );
        assert!(info.has_gaps());
        assert!(
            !Info::from_fold(&Coverage::default(), &SigmaReport::default(), true)
                .expect("no shares to check")
                .has_gaps()
        );
    }

    /// A dilute run whose partials disagree with a channel's rate has a
    /// production that is not the rate its share is of, as a grafted `(n,n')`
    /// weighted by MT 4 has. The total is withheld there too, and the
    /// per-channel shares and the listed ratio stay.
    #[test]
    fn a_dilute_run_with_partials_off_the_rate_reports_no_total() {
        let key = ("Rh103".to_string(), "(n,n')".to_string());
        for (above, below) in [
            (BTreeMap::from([(key.clone(), 4.0)]), BTreeMap::new()),
            (BTreeMap::new(), BTreeMap::from([(key.clone(), 0.5)])),
        ] {
            let coverage = Coverage {
                rate_fraction_covered: BTreeMap::from([(key.clone(), 0.25)]),
                partials_above_rate: above,
                partials_below_rate: below,
                covered_production: 1.0,
                total_production: 4.0,
                ..Default::default()
            };
            let info = Info::from_fold(&coverage, &SigmaReport::default(), true)
                .expect("shares are shares");
            assert_eq!(info.rate_fraction_covered_total, None);
            assert_eq!(info.rate_fraction_covered, coverage.rate_fraction_covered);
        }
    }

    /// A share outside [0, 1] needs a negative cross section, flux or
    /// production, which nothing upstream rules out. It fails the run with a
    /// message naming it rather than panicking or being clamped into range.
    #[test]
    fn a_share_outside_zero_to_one_is_an_error() {
        let channel = Coverage {
            rate_fraction_covered: BTreeMap::from([(("W186".into(), "(n,gamma)".into()), 1.5)]),
            ..Default::default()
        };
        let err = Info::from_fold(&channel, &SigmaReport::default(), true).unwrap_err();
        assert!(err.contains("W186 (n,gamma)"), "{err}");
        let total = Coverage {
            covered_production: 5.0,
            total_production: 4.0,
            ..Default::default()
        };
        let err = Info::from_fold(&total, &SigmaReport::default(), true).unwrap_err();
        assert!(err.contains("not a share"), "{err}");
    }

    /// Every input held at nominal whatever the run was, named so a reader
    /// does not have to know the code to see what the sigma leaves out.
    #[test]
    fn the_report_names_every_input_held_at_nominal() {
        let info = Info::from_fold(&Coverage::default(), &SigmaReport::default(), true)
            .expect("no shares to check");
        for held in [
            "fission yield",
            "isomeric branching (MF=9/MF=10)",
            "covariance with another evaluation (MAT1 naming another material)",
            "covariance with a quantity that is not a cross section (MF=33 XMF1 not 0 or 3)",
            "NC-derived covariance that cannot be derived (MF=33 NC LTY 1-4, or LTY=0 in skipped_nc)",
            "lumped-reaction covariance of several components no derivation names (MF=33 MT=851-870, in lumped_covariance_not_assignable)",
            "resonance-parameter covariance (MF=32)",
            "decay photon line energy and intensity (MF=8 MT=457)",
            "photon attenuation coefficient (XCOM)",
            "air energy-absorption coefficient (NIST SRD 126)",
            "fluence-to-dose coefficient (ICRP-116)",
            "contact-dose build-up factor",
            "material composition",
            "material density",
            "natural isotopic abundance",
            "atomic mass (AME2020)",
        ] {
            assert!(
                info.not_perturbed.iter().any(|s| s == held),
                "{held:?} missing from {:?}",
                info.not_perturbed
            );
        }
        // These depend on the run and the sources asked for, so the fold
        // alone must not claim them.
        for conditional in [
            "self-shielding correction",
            "flux response to perturbed cross sections (one transport)",
            "tallied-rate statistics",
            "flux spectrum",
            "flux spectrum (spectra without a sigma only)",
            "activation cross section (MF=33)",
            "half-life",
            "decay branching ratio",
            "decay energy",
        ] {
            assert!(!info.not_perturbed.iter().any(|s| s == conditional));
        }
    }

    #[test]
    fn moments_match_a_hand_computed_standard_deviation() {
        let mut m = Moments::default();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            m.update(x);
        }
        assert_eq!(m.mean, 5.0);
        // Sample (n-1) standard deviation of that set is sqrt(32/7).
        assert!((m.std_dev() - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_single_sample_has_no_spread() {
        let mut m = Moments::default();
        m.update(3.0);
        assert_eq!(m.std_dev(), 0.0);
    }

    /// A nuclide that appears in only some replicas is zero in the others, so
    /// its variance is computed over every replica rather than over the subset
    /// that happened to carry it.
    #[test]
    fn absences_fold_in_as_zeros() {
        let mut e = Ensemble::new(1);
        e.push(vec![HashMap::from([("Mn56".to_string(), 2.0)])]);
        e.push(vec![HashMap::new()]);
        e.push(vec![HashMap::from([("Mn56".to_string(), 4.0)])]);
        e.fold_absences();

        let sd = e.std_dev_at(0);
        // Over {2, 0, 4}: mean 2, sample sd = 2.
        assert!((sd["Mn56"] - 2.0).abs() < 1e-12, "{sd:?}");
        assert_eq!(e.samples_at(0, "Mn56"), vec![2.0, 0.0, 4.0]);
    }

    #[test]
    fn settling_is_judged_on_the_worst_nuclide_not_the_average() {
        let before = BTreeMap::from([("a".into(), 1.0), ("b".into(), 1.0)]);
        // "a" is unchanged, "b" moved 50%. The average movement is 25%, under
        // any sane tolerance; the worst is not.
        let after = BTreeMap::from([("a".into(), 1.0), ("b".into(), 1.5)]);
        assert!(!settled(&before, &after));

        let close = BTreeMap::from([("a".into(), 1.0), ("b".into(), 1.005)]);
        assert!(settled(&before, &close));
    }

    #[test]
    fn a_nuclide_that_appears_late_is_not_treated_as_settled() {
        let before = BTreeMap::from([("a".into(), 1.0)]);
        let after = BTreeMap::from([("a".into(), 1.0), ("b".into(), 0.5)]);
        assert!(!settled(&before, &after), "a new nuclide has not settled");
    }

    /// A nuclide whose every sigma is the 0.0 MT=457 writes for "not stated",
    /// which the chain files store as 0.0 rather than as null.
    fn stated_as_zero() -> yani::ChainNuclide {
        yani::ChainNuclide {
            name: "In116_m1".to_string(),
            half_life: Some(3257.4),
            half_life_uncertainty: Some(0.0),
            decay_energy: 2.8e6,
            decay_energy_uncertainty: Some(0.0),
            decay_energy_components: [
                Some(yani::DecayEnergyComponent {
                    energy: 2.8e6,
                    uncertainty: Some(0.0),
                }),
                None,
                None,
            ],
            reactions: Vec::new(),
            decays: Vec::new(),
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    #[test]
    fn a_stored_zero_sigma_is_not_stated_rather_than_exact() {
        // Every reader of these sigmas has to take 0.0 as "not stated", the
        // way it takes a null: a zero sampled as a sigma would report the
        // nuclide as known exactly, and it would not be listed among the
        // inputs that carry no uncertainty.
        let cn = stated_as_zero();
        let chain = HashMap::from([(cn.name.clone(), cn.clone())]);
        let (with, without, not_carried) = half_life_candidates(&chain);
        assert!(
            with.is_empty(),
            "a 0.0 half-life sigma was sampled: {with:?}"
        );
        assert!(
            without.contains("In116_m1"),
            "and it was not reported unstated"
        );
        assert!(not_carried.is_empty());

        assert!(!has_decay_energy_sigma(&cn));
        assert_eq!(sample_decay_energy(&cn, 7, 0), None);
    }

    #[test]
    fn only_significant_nuclides_drive_convergence() {
        let mut e = Ensemble::new(1);
        for v in [1.0, 1.1, 0.9] {
            e.push(vec![HashMap::from([
                ("big".to_string(), v),
                ("trace".to_string(), v * 1e-12),
            ])]);
        }
        let probe = e.convergence_probe();
        assert!(probe.contains_key("big"));
        assert!(
            !probe.contains_key("trace"),
            "a trace 12 decades down must not decide when to stop"
        );
    }

    /// Assert `draws` have mean `mean` and standard deviation `sigma`, to five
    /// standard errors of a lognormal with those two moments.
    ///
    /// The tolerance is the lognormal's own: the variance of a sample mean is
    /// `sigma^2 / n` and of a sample variance `sigma^4 (w^4 + 2w^3 + 3w^2 - 4)
    /// / n`, with `w = 1 + (sigma / mean)^2`. At a relative sigma of 2 that
    /// second one is 946 / n, which is why the wide case needs a million draws.
    fn assert_moments(draws: &[f64], mean: f64, sigma: f64) {
        let n = draws.len() as f64;
        let m = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        let w = 1.0 + (sigma / mean).powi(2);
        let mean_tol = 5.0 * sigma / n.sqrt();
        let var_tol = 5.0 * ((w.powi(4) + 2.0 * w.powi(3) + 3.0 * w * w - 4.0) / n).sqrt();
        assert!(
            (m - mean).abs() < mean_tol,
            "mean {m} against {mean}, tolerance {mean_tol}"
        );
        assert!(
            (var / (sigma * sigma) - 1.0).abs() < var_tol,
            "variance {var} against {}, relative tolerance {var_tol}",
            sigma * sigma
        );
        assert!(draws.iter().all(|x| *x > 0.0), "a draw went non-positive");
    }

    const RELATIVE_SIGMAS: [f64; 3] = [0.01, 0.5, 2.0];
    const DRAWS: u64 = 1_000_000;

    #[test]
    fn half_life_draws_carry_the_stated_mean_and_sigma() {
        // A Gaussian floored at zero fails this at 0.5 and badly at 2: at a
        // relative sigma of 2 its mean is 1.40 T, not T.
        for rel in RELATIVE_SIGMAS {
            let t = 3600.0;
            let candidates = vec![("X".to_string(), t, rel * t)];
            let draws: Vec<f64> = (0..DRAWS)
                .map(|r| sample_half_lives(&candidates, 5, r)["X"])
                .collect();
            assert_moments(&draws, t, rel * t);
        }
    }

    #[test]
    fn decay_energy_draws_carry_the_stated_mean_and_sigma() {
        let mut cn = stated_as_zero();
        // The total, where the data gives no split.
        cn.decay_energy_components = [None; 3];
        for rel in RELATIVE_SIGMAS {
            cn.decay_energy_uncertainty = Some(rel * cn.decay_energy);
            let draws: Vec<f64> = (0..DRAWS)
                .map(|r| sample_decay_energy(&cn, 5, r).unwrap().0)
                .collect();
            assert_moments(&draws, cn.decay_energy, rel * cn.decay_energy);
        }

        // The components, each from its own sigma, one of each width at once.
        let energies = [1.5e6, 0.8e6, 5.0e6];
        for (c, (e, rel)) in cn
            .decay_energy_components
            .iter_mut()
            .zip(energies.iter().zip(RELATIVE_SIGMAS))
        {
            *c = Some(yani::DecayEnergyComponent {
                energy: *e,
                uncertainty: Some(rel * e),
            });
        }
        let drawn: Vec<_> = (0..DRAWS)
            .map(|r| sample_decay_energy(&cn, 5, r).unwrap())
            .collect();
        for (i, (e, rel)) in energies.iter().zip(RELATIVE_SIGMAS).enumerate() {
            let draws: Vec<f64> = drawn.iter().map(|d| d.1[i].unwrap().energy).collect();
            assert_moments(&draws, *e, rel * e);
        }
        for (total, parts) in &drawn {
            let sum: f64 = parts.iter().flatten().map(|p| p.energy).sum();
            assert_eq!(*total, sum, "the total is its components' sum");
        }
    }

    /// At a small relative sigma the lognormal is the Gaussian it replaced to
    /// first order, `T exp(s z - s^2/2) = T + sigma z + T rel^2 (z^2 - 1) / 2 +
    /// O(rel^3)`, so a well known half-life samples as it did before. Drawn
    /// from the same stream and the same `z`, so the seed contract is kept.
    #[test]
    fn a_small_sigma_draws_what_the_gaussian_did() {
        let (t, rel) = (3600.0, 0.01);
        let candidates = vec![("X".to_string(), t, rel * t)];
        for replica in 0..10_000 {
            let replica_seed = yamc_rng::history_seed(5, replica);
            let seed = yamc_rng::secondary_seed(
                replica_seed,
                crate::covariance_sample::name_ordinal("X") ^ HALF_LIFE_STREAM,
            );
            let z =
                crate::covariance_sample::standard_normals(&mut yamc_rng::expand_seed(seed), 1)[0];
            let gaussian = t + rel * t * z;
            let drawn = sample_half_lives(&candidates, 5, replica)["X"];
            let bound = t * rel * rel * (1.0 + z * z);
            assert!(
                (drawn - gaussian).abs() < bound,
                "replica {replica}: {drawn} against {gaussian} with z {z}"
            );
        }

        let mut cn = stated_as_zero();
        cn.decay_energy_components = [None; 3];
        cn.decay_energy_uncertainty = Some(rel * cn.decay_energy);
        for replica in 0..10_000 {
            let replica_seed = yamc_rng::history_seed(5, replica);
            let seed = yamc_rng::secondary_seed(
                replica_seed,
                crate::covariance_sample::name_ordinal(&cn.name) ^ DECAY_ENERGY_STREAM,
            );
            let z =
                crate::covariance_sample::standard_normals(&mut yamc_rng::expand_seed(seed), 1)[0];
            let gaussian = cn.decay_energy * (1.0 + rel * z);
            let drawn = sample_decay_energy(&cn, 5, replica).unwrap().0;
            let bound = cn.decay_energy * rel * rel * (1.0 + z * z);
            assert!(
                (drawn - gaussian).abs() < bound,
                "replica {replica}: {drawn} against {gaussian} with z {z}"
            );
        }
    }

    /// A non-negative energy with a mean of zero is zero in every draw, so a
    /// sigma stated on one cannot be carried and the energy stays at zero.
    #[test]
    fn a_zero_energy_stays_zero() {
        let mut cn = stated_as_zero();
        cn.decay_energy_components = [
            Some(yani::DecayEnergyComponent {
                energy: 0.0,
                uncertainty: Some(1.0e3),
            }),
            None,
            None,
        ];
        // A zero energy is zero in every draw, so its sigma cannot be carried:
        // nothing is drawn, and the nuclide is reported rather than listed as
        // perturbed.
        assert_eq!(sample_decay_energy(&cn, 5, 0), None);
        assert!(!has_decay_energy_sigma(&cn));
        assert!(has_decay_energy_sigma_not_carried(&cn));
    }

    /// A finite relative sigma whose square overflows would make the
    /// lognormal NaN, so it is reported like an infinite one.
    #[test]
    fn a_relative_sigma_whose_square_overflows_is_not_carried() {
        assert_eq!(carried(1.0, Some(1.0e155)), None);
        assert_eq!(carried(1.0e-10, Some(1.0e150)), None);
        assert_eq!(carried(1.0, Some(1.0e150)), Some(1.0e150));
    }

    #[test]
    fn a_sigma_no_draw_can_carry_is_reported_not_drawn() {
        let mut cn = stated_as_zero();
        cn.half_life = Some(10.0);
        cn.half_life_uncertainty = Some(f64::INFINITY);
        cn.decay_energy = 1.0e6;
        cn.decay_energy_uncertainty = Some(f64::INFINITY);
        let chain = HashMap::from([(cn.name.clone(), cn.clone())]);
        let (with, without, not_carried) = half_life_candidates(&chain);
        assert!(
            with.is_empty() && without.is_empty(),
            "{with:?} {without:?}"
        );
        assert!(not_carried.contains("In116_m1"));
        assert_eq!(sample_decay_energy(&cn, 5, 0), None);
        assert!(has_decay_energy_sigma_not_carried(&cn));

        // One component carried and one not: the nuclide is drawn, and still
        // reported for the component whose sigma is lost.
        cn.decay_energy_components = [
            Some(yani::DecayEnergyComponent {
                energy: 0.0,
                uncertainty: Some(1.0e3),
            }),
            Some(yani::DecayEnergyComponent {
                energy: 1.0e6,
                uncertainty: Some(1.0e5),
            }),
            None,
        ];
        assert!(has_decay_energy_sigma(&cn));
        assert!(has_decay_energy_sigma_not_carried(&cn));
        let (_, parts) = sample_decay_energy(&cn, 5, 0).unwrap();
        assert_eq!(parts[0].unwrap().energy, 0.0);
        assert_ne!(parts[1].unwrap().energy, 1.0e6);
    }
}
