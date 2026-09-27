//! Half-life uncertainty on a D1S shutdown dose rate (issue #140, item 5).
//!
//! A D1S tally scores, per emitting radionuclide, the dose its decay photons
//! give per decay, and the time-correction factor (TCF) turns that into a dose
//! rate: the emitter's Bateman activity over the schedule. Only the TCF reads a
//! half-life. The in-line photon yield is the per-decay probability (the
//! chain's per-atom intensity over the decay constant it was stored with), a
//! property of the decay scheme, so the tally needs no re-transport.
//!
//! The rule #140 records, one set of decay constants per replica used
//! everywhere, is kept by drawing each replica's half-lives from the same
//! `(seed, replica, nuclide)` streams a transmutation uses and computing that
//! replica's TCFs, for every campaign of the schedule, from them. Drawing a
//! campaign's constants independently of another's would treat one evaluation
//! as several.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use yani::ChainNuclide;

use crate::uncertainty::{
    half_life_candidates, sample_half_lives, with_half_lives, DataUncertainty, Source, BLOCK,
    MAX_SAMPLES, MIN_SAMPLES, TOLERANCE,
};

/// One replica's TCFs: per campaign, emitter -> TCF per schedule index.
pub type ReplicaTcfs = Vec<HashMap<String, Vec<f64>>>;

/// The TCFs of every replica, and what went into them.
#[derive(Debug, Clone, Default)]
pub struct TcfEnsemble {
    /// `[replica][campaign]` -> emitter -> TCF per schedule index, as
    /// [`yani_decay::time_correction_factors`] returns them.
    pub replicas: Vec<ReplicaTcfs>,
    /// Half-lives sampled: the emitters and every nuclide decaying into one.
    pub half_lives_perturbed: BTreeSet<String>,
    /// The same, for nuclides whose evaluation states no half-life sigma, so
    /// held at nominal. Not a claim that they are exact.
    pub no_half_life_uncertainty: BTreeSet<String>,
    /// Whether the TCF spreads settled rather than hitting the cap.
    pub converged: bool,
    /// The sources that applied, by name. Only `half_life` acts on a TCF.
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

/// Resample the half-lives behind a schedule's TCFs.
///
/// `source_rates` holds one rate history per campaign of the schedule, as
/// `time_correct_tally` splits it. Replicas are added in blocks until every
/// emitter's final-step TCF spread settles, as the transmutation driver does,
/// or `request.samples` fixes the count.
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
    if !request.wants(Source::HalfLife) {
        out.converged = true;
        return Ok(out);
    }
    out.sources = vec![Source::HalfLife.name().to_string()];

    let relevant = feeding(chain, emitters);
    let (all, without) = half_life_candidates(chain);
    let candidates: Vec<(String, f64, f64)> = all
        .into_iter()
        .filter(|(n, _, _)| relevant.contains(n))
        .collect();
    out.half_lives_perturbed = candidates.iter().map(|(n, _, _)| n.clone()).collect();
    out.no_half_life_uncertainty = without
        .into_iter()
        .filter(|n| relevant.contains(n))
        .collect();
    if candidates.is_empty() {
        out.converged = true;
        return Ok(out);
    }

    let one = |replica: u64| -> Result<ReplicaTcfs, String> {
        let mut floored = 0;
        let sampled = sample_half_lives(&candidates, request.seed, replica, &mut floored);
        let chain_k = Arc::new(with_half_lives(chain, &sampled));
        source_rates
            .iter()
            .map(|rates| yani_decay::time_correction_factors(emitters, timesteps, rates, &chain_k))
            .collect()
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
        let results: Vec<Result<ReplicaTcfs, String>> = {
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
            out.replicas.push(r?);
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
