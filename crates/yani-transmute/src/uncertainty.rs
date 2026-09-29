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
use crate::covariance_sample::Clipping;

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
    /// Activation cross sections, from ENDF MF=33 covariance (issue #514).
    CrossSections,
    /// The supplied flux spectrum, from a per-bin standard deviation the caller
    /// provides (issue #559).
    ///
    /// Unlike the others this needs no nuclear data: the flux is the caller's
    /// own input. It contributes only where a sigma was actually given, which
    /// a spectrum from a published reference set does not have.
    FluxSpectrum,
    /// Half-lives, from the decay sublibrary's own standard deviation on each
    /// (the uncertainty on the MT=457 `T1/2`).
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
    /// rates, from their per-history covariance (issue #140, item 1).
    ///
    /// Applies to `Model.simulate_transmutation`, whose rates come from
    /// transport; a spectrum run's rates are a deterministic collapse with no
    /// sampling error, so it has nothing to perturb there.
    Statistical,
    /// Mean decay energies, from the decay data's sigma on each
    /// recoverable-heat component (beta, gamma, alpha), or on the total where
    /// the data carries no split.
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
                     Fission yields carry published uncertainties that are not read yet \
                     (issue #140), and reaction branching has none in ENDF-6 to read."
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
    /// Blocks correlating with another evaluation (`mat1 != 0`), not consumed.
    pub skipped_cross_material: usize,
    /// NC blocks (covariance derived from other reactions), not consumed.
    pub skipped_nc: usize,
    /// Blocks whose `lb` layout is not implemented, counted per `lb`.
    pub unsupported_layouts: BTreeMap<i64, usize>,
    /// Blocks not consumed because they break ENDF-102's rules for their
    /// layout: arrays that disagree with their declared sizes, an LB=0 to 2
    /// block carrying a second table, an LB=3 or 4 block without one or whose
    /// tables share no energy range, or an LB=8 block between two reactions.
    pub malformed_blocks: usize,
    /// Per (nuclide, reaction kind), the share of the dilute rate over the
    /// flux range that comes from energies where the evaluation states a
    /// nonzero variance for that reaction. Below one means part of the dilute
    /// rate carries no stated uncertainty, so it dilutes the relative sigma.
    /// An interval a covariance grid spans with a variance of zero counts as
    /// uncovered: it states no uncertainty either. In [0, 1] whatever rate the
    /// covariance is divided by, and a share of the dilute rate only: on a
    /// self-shielded or tallied run the covered share of the rate actually
    /// used is not computed. See [`Info::partials_above_rate`] for when that
    /// rate disagrees with the partials. No entry for a channel whose dilute
    /// rate over the flux range is zero. Every consumed self-covariance block
    /// counts where its own diagonal is nonzero, relative (LB=1 to 6),
    /// absolute (LB=0) and short-range (LB=8) alike.
    pub rate_fraction_covered: BTreeMap<(String, String), f64>,
    /// Per (nuclide, reaction kind), where the partial rates a relative
    /// covariance block was weighted with, zero variance intervals included,
    /// add up to more than the rate it was divided by, their ratio to it. Each
    /// is a channel whose relative sigma is overstated, because the two were
    /// computed different ways; a self-shielded rate against dilute partials
    /// is one, and a tallied rate is another. Its `rate_fraction_covered` is
    /// unaffected. Until the fold weights a shielded rate with shielded
    /// partials (#166 item 4), a self-shielded run lists most channels a
    /// relative block names, many only a few parts in 1e7 over.
    pub partials_above_rate: BTreeMap<(String, String), f64>,
    /// Per (nuclide, reaction kind), where a relative block's grid spans the
    /// whole flux range and its partial rates add up to less than the rate it
    /// was divided by, their ratio to it: a channel whose relative sigma is
    /// understated. A grid that stops short of the flux range cannot be
    /// checked this way, since rate from outside it rightly leaves its
    /// partials short.
    pub partials_below_rate: BTreeMap<(String, String), f64>,
    /// Mean of the per-channel shares in [`Info::rate_fraction_covered`],
    /// weighted by the production each channel drove (the rate this run used
    /// times parent density): the share of the production driven from
    /// energies where a covariance states a nonzero variance.
    ///
    /// `None` for a decay-only schedule, which drove no production, and on a
    /// self-shielded or tallied run, where that share is not computed. The
    /// per-channel shares are of the dilute rate, and shielding moves rate
    /// out of the resonance range, where capture blocks often state zero, so
    /// weighting them by the shielded or tallied production would give a
    /// figure that is not the share its name claims.
    ///
    /// The number to read before any sigma here, and not the same question as
    /// how many nuclides carry MF=33: an evaluation can state covariance for
    /// every isotope in the material and none of it for the channel making the
    /// product of interest, which leaves the count reading as full coverage
    /// while the ensemble perturbs almost nothing.
    pub rate_fraction_covered_total: Option<f64>,
    /// Covariance matrices that were not positive semi-definite as evaluated,
    /// and the worst repair that had to be made.
    pub matrices_clipped: usize,
    pub worst_relative_clip: f64,
    /// Cross-section rate draws made, one per perturbed channel per spectrum
    /// per replica.
    ///
    /// Each draw is a lognormal multiplier matched to the channel's mean and
    /// variance, so none can go negative and there is no floor to count.
    pub rates_sampled: usize,
    /// Spectra that carried a per-bin flux sigma, and those that did not.
    ///
    /// A spectrum taken from a published reference set has no stated error, so
    /// it contributes nothing and that has to be visible rather than read as a
    /// flux known exactly (issue #559).
    pub spectra_with_flux_sigma: usize,
    pub spectra_without_flux_sigma: usize,
    /// Sampled flux bins that went negative and were floored at zero.
    pub flux_bins_floored: usize,
    pub flux_bins_sampled: usize,
    /// Unstable nuclides the material can reach whose half-life was perturbed.
    pub half_lives_perturbed: BTreeSet<String>,
    /// Unstable nuclides the material can reach whose evaluation states no
    /// half-life uncertainty, so their half-life was held at nominal.
    ///
    /// Not a claim that the half-life is exact: the evaluation said nothing.
    pub no_half_life_uncertainty: BTreeSet<String>,
    /// Sampled half-lives that came out non-positive and were floored.
    ///
    /// Only possible where the stated sigma is a large share of the half-life
    /// itself; a count well above zero says the Gaussian is being used past
    /// where it describes the evaluation.
    pub half_lives_floored: usize,
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
    /// `skipped_nc`, `unsupported_layouts` and `malformed_blocks`. A block for
    /// a reaction the chain does not drive (a partial-level section such as
    /// MT=600-849) is neither listed nor counted: the chain has no rate for it
    /// to be the uncertainty of.
    pub not_perturbed: Vec<String>,
    /// Which sources this run perturbed, by name.
    pub sources: Vec<String>,
}

impl Info {
    /// `dilute` is whether the rates the fold was divided by are the dilute
    /// collapse, the only case in which the production total is the covered
    /// share.
    pub(crate) fn from_fold(coverage: &Coverage, clipping: &Clipping, dilute: bool) -> Self {
        Self {
            perturbed: coverage.covered.clone(),
            no_covariance_data: coverage.without_data.clone(),
            skipped_cross_material: coverage.skipped_cross_material,
            skipped_nc: coverage.skipped_nc,
            unsupported_layouts: coverage.unsupported_layouts.clone(),
            malformed_blocks: coverage.malformed,
            rate_fraction_covered: coverage.rate_fraction_covered.clone(),
            partials_above_rate: coverage.partials_above_rate.clone(),
            partials_below_rate: coverage.partials_below_rate.clone(),
            rate_fraction_covered_total: coverage.rate_fraction_total().filter(|_| dilute),
            matrices_clipped: clipping.matrices_clipped,
            worst_relative_clip: clipping.worst_relative_clip,
            not_perturbed: [
                "fission yield",
                "isomeric branching (MF=9/MF=10)",
                "cross-material covariance (MAT1 != 0)",
                "NC-derived covariance (MF=33 NC)",
                "lumped-reaction covariance (MF=33 MT=851-870)",
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
        }
    }

    pub(crate) fn add_flux_coverage(&mut self, c: &crate::flux_uncertainty::FluxCoverage) {
        self.spectra_with_flux_sigma = c.spectra_with_sigma;
        self.spectra_without_flux_sigma = c.spectra_without_sigma;
        self.flux_bins_floored = c.bins_floored;
        self.flux_bins_sampled = c.bins_sampled;
    }

    /// Whether anything was left out, or is inconsistent, that a reader should
    /// know about.
    pub fn has_gaps(&self) -> bool {
        !self.no_covariance_data.is_empty()
            || self.skipped_cross_material > 0
            || self.skipped_nc > 0
            || !self.unsupported_layouts.is_empty()
            || self.malformed_blocks > 0
            || !self.partials_above_rate.is_empty()
            || !self.partials_below_rate.is_empty()
            || self.spectra_without_flux_sigma > 0
            || !self.no_half_life_uncertainty.is_empty()
            || !self.no_decay_branching_uncertainty.is_empty()
            || !self.decay_branchings_three_or_more_modes.is_empty()
            || !self.decay_branchings_unequal_sigmas.is_empty()
            || !self.decay_branchings_too_wide.is_empty()
            || !self.no_decay_energy_uncertainty.is_empty()
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

/// Where an inventory's uncertainty comes from (issue #140, item 4).
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

/// The unstable nuclides of `chain` split into those with a stated half-life
/// sigma, as `(name, half-life, sigma)`, and those without.
pub(crate) fn half_life_candidates(
    chain: &HashMap<String, yani::ChainNuclide>,
) -> (Vec<(String, f64, f64)>, BTreeSet<String>) {
    let mut with = Vec::new();
    let mut without = BTreeSet::new();
    for (name, cn) in chain {
        let Some(t) = cn.half_life.filter(|t| *t > 0.0) else {
            continue;
        };
        match cn.half_life_uncertainty.filter(|s| *s > 0.0) {
            Some(sigma) => with.push((name.clone(), t, sigma)),
            None => {
                without.insert(name.clone());
            }
        }
    }
    with.sort_by(|a, b| a.0.cmp(&b.0));
    (with, without)
}

/// One replica's half-lives: `T + sigma z` per nuclide, from a stream keyed on
/// `(seed, replica, nuclide)`.
///
/// Keyed on the name, like the cross sections, so a nuclide draws the same
/// half-life in every spectrum and every material of a run: one evaluation is
/// uncertain in one way wherever it is used. A draw at or below zero is
/// floored at a millionth of the nominal and counted.
pub(crate) fn sample_half_lives(
    candidates: &[(String, f64, f64)],
    base_seed: u64,
    replica: u64,
    floored: &mut usize,
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
            let mut sampled = t + sigma * z;
            if sampled <= 0.0 {
                *floored += 1;
                sampled = t * 1.0e-6;
            }
            (name.clone(), sampled)
        })
        .collect()
}

/// Keeps the decay-energy streams clear of every other per-nuclide stream.
pub(crate) const DECAY_ENERGY_STREAM: u32 = 0xDEC4_E6E1;

/// One replica's decay energy for one nuclide, and its components, drawn
/// from each component's own sigma, or from the total's where the data gives
/// no split. `None` when there is nothing to perturb. A draw below zero is
/// floored at zero.
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
    let mut state = yamc_rng::expand_seed(seed);
    let with_sigma = cn
        .decay_energy_components
        .iter()
        .any(|c| c.is_some_and(|c| c.uncertainty.is_some_and(|s| s > 0.0)));
    if with_sigma {
        let z = crate::covariance_sample::standard_normals(&mut state, 3);
        let mut parts = cn.decay_energy_components;
        let mut total = 0.0;
        for (part, z) in parts.iter_mut().zip(z) {
            if let Some(p) = part {
                if let Some(sigma) = p.uncertainty.filter(|s| *s > 0.0) {
                    p.energy = (p.energy + sigma * z).max(0.0);
                }
                total += p.energy;
            }
        }
        return Some((total, parts));
    }
    let sigma = cn.decay_energy_uncertainty.filter(|s| *s > 0.0)?;
    let z = crate::covariance_sample::standard_normals(&mut state, 1)[0];
    Some((
        (cn.decay_energy + sigma * z).max(0.0),
        cn.decay_energy_components,
    ))
}

/// Whether a nuclide's decay energy carries a sigma to sample from.
pub(crate) fn has_decay_energy_sigma(cn: &yani::ChainNuclide) -> bool {
    cn.decay_energy_components
        .iter()
        .any(|c| c.is_some_and(|c| c.uncertainty.is_some_and(|s| s > 0.0)))
        || cn.decay_energy_uncertainty.is_some_and(|s| s > 0.0)
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
/// half-lives exist to prevent.
pub(crate) fn set_half_life(cn: &mut yani::ChainNuclide, half_life: f64) {
    if let Some(nominal) = cn.half_life.filter(|t| *t > 0.0 && half_life > 0.0) {
        let scale = nominal / half_life;
        for source in &mut cn.sources {
            for i in source.distribution.intensities_mut().iter_mut() {
                *i *= scale;
            }
        }
    }
    cn.half_life = Some(half_life);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every run states the decay photon sources it holds at nominal, lines
    /// and continua both, so a reader does not take their silence for zero.
    #[test]
    fn the_report_names_the_photon_sources_held_at_nominal() {
        let info = Info::from_fold(&Coverage::default(), &Clipping::default(), true);
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

    /// The production total weights dilute shares by the production the run
    /// drove, which is the covered share only when that production is the
    /// dilute one. A shielded or tallied run reports no total rather than a
    /// figure that is not the share its name claims, and keeps the
    /// per-channel dilute shares, which are.
    #[test]
    fn only_a_dilute_run_reports_a_production_total() {
        let coverage = Coverage {
            rate_fraction_covered: BTreeMap::from([(("W186".into(), "(n,gamma)".into()), 0.25)]),
            covered_production: 1.0,
            total_production: 4.0,
            ..Default::default()
        };
        let clipping = Clipping::default();
        let dilute = Info::from_fold(&coverage, &clipping, true);
        assert_eq!(dilute.rate_fraction_covered_total, Some(0.25));
        let other = Info::from_fold(&coverage, &clipping, false);
        assert_eq!(other.rate_fraction_covered_total, None);
        assert_eq!(other.rate_fraction_covered, coverage.rate_fraction_covered);
    }

    /// Every input held at nominal whatever the run was, named so a reader
    /// does not have to know the code to see what the sigma leaves out.
    #[test]
    fn the_report_names_every_input_held_at_nominal() {
        let info = Info::from_fold(&Coverage::default(), &Clipping::default(), true);
        for held in [
            "fission yield",
            "isomeric branching (MF=9/MF=10)",
            "cross-material covariance (MAT1 != 0)",
            "NC-derived covariance (MF=33 NC)",
            "lumped-reaction covariance (MF=33 MT=851-870)",
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
        let (with, without) = half_life_candidates(&chain);
        assert!(
            with.is_empty(),
            "a 0.0 half-life sigma was sampled: {with:?}"
        );
        assert!(
            without.contains("In116_m1"),
            "and it was not reported unstated"
        );

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
}
