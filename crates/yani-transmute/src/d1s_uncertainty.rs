//! Half-life and decay photon normalisation uncertainty on a D1S shutdown
//! dose rate.
//!
//! A D1S tally scores, per emitting radionuclide, the dose its decay photons
//! give per decay, and the time-correction factor (TCF) turns that into a dose
//! rate: the emitter's Bateman activity over the schedule. Only the TCF reads a
//! half-life. The in-line photon yield is the per-decay probability (the
//! chain's per-atom intensity over the decay constant it was stored with), a
//! property of the decay scheme, so the tally needs no re-transport.
//!
//! The rule of one set of decay constants per replica used
//! everywhere, is kept by drawing each replica's half-lives from the same
//! `(seed, replica, nuclide)` streams a transmutation uses and computing that
//! replica's TCFs, for every campaign of the schedule, from them. Drawing a
//! campaign's constants independently of another's would treat one evaluation
//! as several.
//!
//! The decay photon data enter the tally itself, as the yield and spectrum of
//! each emitter's photons. Of their uncertainty, the spectrum normalisation
//! (dFD, dFC) scales an emitter's whole spectrum, so it scales that emitter's
//! tally and is applied to it after the fact, from the same streams a
//! transmutation draws it from. The line intensities (dRI) and energies (dER)
//! change the spectrum's shape, which a per-emitter tally cannot follow
//! without a tally resolved by line, so they are held at nominal and listed
//! under `not_perturbed`. For ENDF/B-VIII.1 that leaves almost nothing drawn:
//! it writes dFD = 0 and folds the normalisation into the dRI, which
//! `decay_photon_spectra_folded` names.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use yani::ChainNuclide;

use crate::uncertainty::{
    folded_photon_spectra, half_life_candidates, has_decay_photon_normalisation_sigma,
    photon_normalisation_scale, sample_half_lives, with_half_lives, DataUncertainty,
    PhotonCorrelation, Source, BLOCK, MAX_SAMPLES, MIN_SAMPLES, TOLERANCE,
};

/// What a D1S dose holds at nominal of the decay photon data when the
/// `decay_photon_lines` source is on.
pub const D1S_LINES_NOT_PERTURBED: &str =
    "decay photon line relative intensity (dRI) and line energy (dER): they change an \
     emitter's spectrum shape, which a per-emitter tally cannot follow without line-resolved \
     tallies";

/// What a D1S dose holds at nominal of the decay photon data when the
/// `decay_photon_lines` source is off.
pub const D1S_PHOTONS_NOT_PERTURBED: &str =
    "decay photon spectrum normalisation, line intensity and line energy";

/// One replica's TCFs: per campaign, emitter -> TCF per schedule index.
pub type ReplicaTcfs = Vec<HashMap<String, Vec<f64>>>;

/// The TCFs of every replica, and what went into them.
#[derive(Debug, Clone, Default)]
pub struct TcfEnsemble {
    /// `[replica][campaign]` -> emitter -> TCF per schedule index, as
    /// [`yani_decay::time_correction_factors`] returns them, times the
    /// emitter's drawn photon normalisation scale when the
    /// `decay_photon_lines` source is on. An emitter's spectra draw their
    /// normalisations independently here, the lower end of the range their
    /// unstated correlation leaves.
    pub replicas: Vec<ReplicaTcfs>,
    /// The same replicas with each emitter's spectra (gamma and x-ray) drawing
    /// one normalisation deviate between them, the upper end. Equal to
    /// `replicas` when no normalisation is drawn. Every non-negative
    /// correlation between the spectra gives a spread between the two;
    /// negative ones are not considered, since what the spectra share is a
    /// common scheme normalisation and conversion coefficients, which move
    /// them the same way.
    pub replicas_correlated: Vec<ReplicaTcfs>,
    /// Half-lives sampled: the emitters and every nuclide decaying into one.
    pub half_lives_perturbed: BTreeSet<String>,
    /// The same, for nuclides whose evaluation states no half-life sigma, so
    /// held at nominal. Not a claim that they are exact.
    pub no_half_life_uncertainty: BTreeSet<String>,
    /// The same, for nuclides whose stated half-life sigma no draw can carry
    /// (not finite, or not finite relative to the half-life), so held at
    /// nominal.
    pub half_life_uncertainty_not_carried: BTreeSet<String>,
    /// Emitters whose photon spectrum normalisation (FD or FC) carries a sigma
    /// that was drawn.
    pub decay_photon_normalisations_perturbed: BTreeSet<String>,
    /// Emitters with a photon spectrum written the ENDF/B way, and the
    /// radiation of each such spectrum: FD = 1 with no sigma and the
    /// normalisation's sigma folded into the dRI, which the tally cannot draw.
    pub decay_photon_spectra_folded: BTreeMap<String, Vec<String>>,
    /// Whether the TCF spreads settled rather than hitting the cap.
    pub converged: bool,
    /// The sources that applied, by name: `half_life`, which acts on a TCF,
    /// and `decay_photon_lines`, whose normalisation scales an emitter's tally.
    pub sources: Vec<String>,
    /// What feeds a TCF and is held at nominal here, for the record.
    ///
    /// Decay branching moves a TCF too, through the share of a parent's
    /// decays that feeds an emitter, but a TCF ensemble does not draw it, so
    /// a D1S dose carries none of the spread a transmutation reports for it.
    pub not_perturbed: Vec<String>,
}

/// The emitters and every nuclide that decays into one of them, so the only
/// half-lives a TCF can depend on.
fn feeding(chain: &HashMap<String, ChainNuclide>, emitters: &[String]) -> HashSet<String> {
    let mut parents: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, cn) in chain {
        for d in &cn.decays {
            if let Some(t) = &d.target {
                parents.entry(t.as_str()).or_default().push(name.as_str());
            }
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack: Vec<&str> = emitters.iter().map(|s| s.as_str()).collect();
    while let Some(n) = stack.pop() {
        if !seen.insert(n.to_string()) {
            continue;
        }
        if let Some(ps) = parents.get(n) {
            stack.extend(ps.iter().copied());
        }
    }
    seen
}

/// Resample the half-lives behind a schedule's TCFs, and the photon
/// normalisation that scales each emitter's tally.
///
/// `source_rates` holds one rate history per campaign of the schedule, as
/// `time_correct_tally` splits it. Replicas are added in blocks until every
/// emitter's final-step TCF spread settles, as the transmutation driver does,
/// or `request.samples` fixes the count.
///
/// With `decay_photon_lines` on, each replica's TCFs are multiplied by the
/// emitter's normalisation scale, at both ends of the range of the unstated
/// correlation between its spectra (see [`TcfEnsemble::replicas_correlated`]).
/// The scale is weighted over an emitter's spectra by their share of its
/// photon energy, since the tally does not resolve spectra; in the libraries
/// at most one spectrum of an emitter states a normalisation sigma.
pub fn time_correction_factor_ensemble(
    emitters: &[String],
    timesteps: &[f64],
    source_rates: &[Vec<f64>],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    request: &DataUncertainty,
) -> Result<TcfEnsemble, String> {
    let mut out = TcfEnsemble {
        not_perturbed: vec!["decay branching ratio".to_string()],
        ..Default::default()
    };
    let want_half_life = request.wants(Source::HalfLife);
    let want_photons = request.wants(Source::DecayPhotonLines);
    if !want_half_life {
        out.not_perturbed.insert(0, "half-life".to_string());
    }
    if want_photons {
        out.not_perturbed.push(D1S_LINES_NOT_PERTURBED.to_string());
    } else {
        out.not_perturbed
            .push(D1S_PHOTONS_NOT_PERTURBED.to_string());
    }
    out.sources = [Source::HalfLife, Source::DecayPhotonLines]
        .into_iter()
        .filter(|s| request.wants(*s))
        .map(|s| s.name().to_string())
        .collect();

    let mut candidates: Vec<(String, f64, f64)> = Vec::new();
    if want_half_life {
        let relevant = feeding(chain, emitters);
        let (all, without, not_carried) = half_life_candidates(chain);
        candidates = all
            .into_iter()
            .filter(|(n, _, _)| relevant.contains(n))
            .collect();
        out.half_lives_perturbed = candidates.iter().map(|(n, _, _)| n.clone()).collect();
        out.no_half_life_uncertainty = without
            .into_iter()
            .filter(|n| relevant.contains(n))
            .collect();
        out.half_life_uncertainty_not_carried = not_carried
            .into_iter()
            .filter(|n| relevant.contains(n))
            .collect();
    }
    let mut scaled: Vec<(&String, &ChainNuclide)> = Vec::new();
    if want_photons {
        for name in emitters {
            let Some(cn) = chain.get(name) else { continue };
            if has_decay_photon_normalisation_sigma(cn) {
                out.decay_photon_normalisations_perturbed
                    .insert(name.clone());
                scaled.push((name, cn));
            }
            let folded = folded_photon_spectra(cn);
            if !folded.is_empty() {
                out.decay_photon_spectra_folded.insert(name.clone(), folded);
            }
        }
    }
    if candidates.is_empty() && scaled.is_empty() {
        out.converged = true;
        return Ok(out);
    }

    let tcfs = |chain: &Arc<HashMap<String, ChainNuclide>>| -> Result<ReplicaTcfs, String> {
        source_rates
            .iter()
            .map(|rates| yani_decay::time_correction_factors(emitters, timesteps, rates, chain))
            .collect()
    };
    let nominal = if candidates.is_empty() {
        Some(tcfs(chain)?)
    } else {
        None
    };
    let one = |replica: u64| -> Result<(ReplicaTcfs, ReplicaTcfs), String> {
        let independent = match &nominal {
            Some(nominal) => nominal.clone(),
            None => {
                let sampled = sample_half_lives(&candidates, request.seed, replica);
                tcfs(&Arc::new(with_half_lives(chain, &sampled)))?
            }
        };
        let mut correlated = independent.clone();
        let mut independent = independent;
        for (name, cn) in &scaled {
            for (ends, correlation) in [
                (&mut independent, PhotonCorrelation::Independent),
                (&mut correlated, PhotonCorrelation::Correlated),
            ] {
                let scale = photon_normalisation_scale(cn, request.seed, replica, correlation);
                for campaign in ends.iter_mut() {
                    if let Some(tcf) = campaign.get_mut(*name) {
                        tcf.iter_mut().for_each(|t| *t *= scale);
                    }
                }
            }
        }
        Ok((independent, correlated))
    };

    // The final TCF of every (campaign, emitter), the quantity judged settled.
    let spread = |replicas: &[ReplicaTcfs]| -> HashMap<(usize, String), f64> {
        let mut out = HashMap::new();
        let n = replicas.len() as f64;
        if n < 2.0 {
            return out;
        }
        for (c, campaign) in replicas[0].iter().enumerate() {
            for name in campaign.keys() {
                let values: Vec<f64> = replicas
                    .iter()
                    .filter_map(|r| r[c].get(name).and_then(|v| v.last()).copied())
                    .collect();
                let mean = values.iter().sum::<f64>() / n;
                if mean == 0.0 {
                    continue;
                }
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                out.insert((c, name.clone()), var.sqrt() / mean.abs());
            }
        }
        out
    };

    let cap = request.samples.unwrap_or(MAX_SAMPLES);
    let mut previous: HashMap<(usize, String), f64> = HashMap::new();
    let mut replica = 0usize;
    while replica < cap {
        let end = (replica + BLOCK).min(cap);
        let block: Vec<u64> = (replica as u64..end as u64).collect();
        let results: Vec<Result<(ReplicaTcfs, ReplicaTcfs), String>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                block.par_iter().map(|&r| one(r)).collect()
            }
            #[cfg(target_arch = "wasm32")]
            {
                block.iter().map(|&r| one(r)).collect()
            }
        };
        for r in results {
            let (independent, correlated) = r?;
            out.replicas.push(independent);
            out.replicas_correlated.push(correlated);
        }
        replica = end;
        if request.samples.is_some() {
            continue;
        }
        let now = spread(&out.replicas);
        if replica >= MIN_SAMPLES
            && now.iter().all(|(k, s)| {
                previous
                    .get(k)
                    .is_some_and(|p| *s == 0.0 || ((s - p) / s).abs() <= TOLERANCE)
            })
        {
            out.converged = true;
            break;
        }
        previous = now;
    }
    if request.samples.is_some() {
        out.converged = true;
    }
    Ok(out)
}
