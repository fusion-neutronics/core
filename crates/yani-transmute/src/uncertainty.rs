//! Nuclear-data uncertainty on the inventory, by resampling and re-solving.
//!
//! The Bateman solve is deterministic and every expensive input is a reaction
//! rate, so the uncertainty is propagated by perturbing the rates and running
//! the same solve again. That is exact to all orders in the matrix exponential:
//! nothing is linearized, and the solver is not touched at all, only its input.
//!
//! What is perturbed is the **activation cross sections**, through the MF=33
//! covariance folded against this material's own spectrum. Half-lives, decay
//! branching ratios, fission yields and the isomeric-branching overlay are held
//! at their nominal values; they carry their own uncertainties and are out of
//! scope here (issue #515). [`Info`] says so per run rather than leaving it to
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
use crate::covariance_sample::{Clipping, Truncations};

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
}

impl Source {
    /// Every source this build can actually perturb.
    ///
    /// Four more inputs feed the matrix and are held at nominal: half-life,
    /// decay branching, fission yields and the isomeric-branching overlay. The
    /// first three have published uncertainties this build does not read yet
    /// (issue #515); the fourth has none in ENDF-6 at all, so it could only
    /// ever carry an assumed value.
    pub const IMPLEMENTED: &'static [Source] = &[Source::CrossSections, Source::FluxSpectrum];

    /// The name used in the API and in the coverage report.
    pub fn name(self) -> &'static str {
        match self {
            Source::CrossSections => "cross_sections",
            Source::FluxSpectrum => "flux_spectrum",
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
                     Half-life, decay branching and fission yields carry published \
                     uncertainties that are not read yet (issue #515), and reaction \
                     branching has none in ENDF-6 to read."
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
}

impl Default for DataUncertainty {
    fn default() -> Self {
        Self {
            seed: 1,
            samples: None,
            sources: Source::IMPLEMENTED.to_vec(),
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
    /// Blocks whose arrays disagreed with their own declared sizes.
    pub malformed_blocks: usize,
    /// Per (nuclide, reaction kind), the share of the rate the covariance grid
    /// spans. Below one means part of the rate carries no stated uncertainty
    /// and the sigma is diluted accordingly.
    pub rate_fraction_covered: BTreeMap<(String, String), f64>,
    /// Share of the production this run drove that carries a stated covariance,
    /// weighted by rate and by parent density, or `None` for a decay-only
    /// schedule that drove none.
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
    /// Sampled rates that went negative and were floored at zero.
    ///
    /// A large share means the Gaussian is being used past where it describes
    /// the cross section, and the truncation biases the mean upward.
    pub rates_floored: usize,
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
    /// Sources deliberately NOT perturbed, for the record.
    pub not_perturbed: Vec<String>,
    /// Which sources this run perturbed, by name.
    pub sources: Vec<String>,
}

impl Info {
    pub(crate) fn from_fold(coverage: &Coverage, clipping: &Clipping) -> Self {
        Self {
            perturbed: coverage.covered.clone(),
            no_covariance_data: coverage.without_data.clone(),
            skipped_cross_material: coverage.skipped_cross_material,
            skipped_nc: coverage.skipped_nc,
            unsupported_layouts: coverage.unsupported_layouts.clone(),
            malformed_blocks: coverage.malformed,
            rate_fraction_covered: coverage.rate_fraction_covered.clone(),
            rate_fraction_covered_total: coverage.rate_fraction_total(),
            matrices_clipped: clipping.matrices_clipped,
            worst_relative_clip: clipping.worst_relative_clip,
            not_perturbed: [
                "half-life",
                "decay branching ratio",
                "fission yield",
                "isomeric branching (MF=9/MF=10)",
                "cross-material covariance (MAT1 != 0)",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ..Default::default()
        }
    }

    pub(crate) fn add_truncations(&mut self, t: &Truncations) {
        self.rates_floored += t.floored;
        self.rates_sampled += t.sampled;
    }

    pub(crate) fn add_flux_coverage(&mut self, c: &crate::flux_uncertainty::FluxCoverage) {
        self.spectra_with_flux_sigma = c.spectra_with_sigma;
        self.spectra_without_flux_sigma = c.spectra_without_sigma;
        self.flux_bins_floored = c.bins_floored;
        self.flux_bins_sampled = c.bins_sampled;
    }

    /// Whether anything was left out that a reader should know about.
    pub fn has_gaps(&self) -> bool {
        !self.no_covariance_data.is_empty()
            || self.skipped_cross_material > 0
            || self.skipped_nc > 0
            || !self.unsupported_layouts.is_empty()
            || self.malformed_blocks > 0
            || self.spectra_without_flux_sigma > 0
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
}

impl Ensemble {
    pub(crate) fn new(n_steps: usize) -> Self {
        Self {
            moments: vec![HashMap::new(); n_steps],
            samples: Vec::new(),
        }
    }

    /// Record one replica's per-step inventories.
    pub(crate) fn push(&mut self, per_step: Vec<HashMap<String, f64>>) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
