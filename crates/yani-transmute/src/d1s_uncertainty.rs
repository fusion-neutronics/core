//! Half-life uncertainty on a D1S shutdown dose rate.
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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use yani::ChainNuclide;

use crate::uncertainty::{
    check_convergence, half_life_candidates, misses, sample_half_lives, with_half_lives,
    worst_first, DataUncertainty, Output, Source, Unconverged, BLOCK, MAX_SAMPLES, MIN_SAMPLES,
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
    /// The same, for nuclides whose stated half-life sigma no draw can carry
    /// (not finite, or not finite relative to the half-life), so held at
    /// nominal.
    pub half_life_uncertainty_not_carried: BTreeSet<String>,
    /// Whether every TCF's sigma reached the convergence target, so
    /// `unconverged` is empty. With a fixed sample count, whether it did at
    /// that count.
    pub converged: bool,
    /// The convergence target judged against, `SE(sigma) / sigma`.
    pub convergence: f64,
    /// Whether the run stopped on the replica cap with some TCF's sigma still
    /// short of the target. Never set on a fixed sample count.
    pub hit_cap: bool,
    /// Every (campaign, emitter, schedule index) TCF whose sigma did not
    /// reach the target, with the `SE(sigma) / sigma` it did reach, worst
    /// first.
    pub unconverged: Vec<Unconverged>,
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
/// `time_correct_tally` splits it. Replicas are added in blocks, as the
/// transmutation driver adds them, until the sigma of every emitter's TCF in
/// every campaign at every schedule index has `SE(sigma) / sigma` below
/// `request.convergence`, or `request.samples` fixes the count.
pub fn time_correction_factor_ensemble(
    emitters: &[String],
    timesteps: &[f64],
    source_rates: &[Vec<f64>],
    chain: &Arc<HashMap<String, ChainNuclide>>,
    request: &DataUncertainty,
) -> Result<TcfEnsemble, String> {
    check_convergence(request.convergence)?;
    let mut out = TcfEnsemble {
        not_perturbed: vec!["decay branching ratio".to_string()],
        convergence: request.convergence,
        ..Default::default()
    };
    if !request.wants(Source::HalfLife) {
        out.not_perturbed.insert(0, "half-life".to_string());
        out.converged = true;
        return Ok(out);
    }
    out.sources = vec![Source::HalfLife.name().to_string()];

    let relevant = feeding(chain, emitters);
    let (all, without, not_carried) = half_life_candidates(chain);
    let candidates: Vec<(String, f64, f64)> = all
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
    if candidates.is_empty() {
        out.converged = true;
        return Ok(out);
    }

    let one = |replica: u64| -> Result<ReplicaTcfs, String> {
        let sampled = sample_half_lives(&candidates, request.seed, replica);
        let chain_k = Arc::new(with_half_lives(chain, &sampled));
        source_rates
            .iter()
            .map(|rates| yani_decay::time_correction_factors(emitters, timesteps, rates, &chain_k))
            .collect()
    };

    // Every (campaign, emitter, schedule index) TCF whose sigma misses the
    // target. A TCF with no spread (the pre-irradiation zero) or a negligible
    // one (a saturated emitter's) has nothing worth converging.
    let unconverged = |replicas: &[ReplicaTcfs]| -> Vec<Unconverged> {
        let mut out = Vec::new();
        let Some(first) = replicas.first() else {
            return out;
        };
        let mut values = Vec::with_capacity(replicas.len());
        for (c, campaign) in first.iter().enumerate() {
            for (name, tcf) in campaign {
                for step in 0..tcf.len() {
                    values.clear();
                    values.extend(replicas.iter().map(|r| {
                        r[c].get(name)
                            .and_then(|v| v.get(step))
                            .copied()
                            .unwrap_or(0.0)
                    }));
                    if let Some(reached) = misses(&values, request.convergence) {
                        out.push(Unconverged {
                            output: Output::TimeCorrectionFactor {
                                campaign: c,
                                emitter: name.clone(),
                            },
                            step,
                            relative_standard_error: reached,
                        });
                    }
                }
            }
        }
        worst_first(&mut out);
        out
    };

    let cap = request.samples.unwrap_or(MAX_SAMPLES);
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
        if replica >= MIN_SAMPLES && unconverged(&out.replicas).is_empty() {
            break;
        }
    }
    out.unconverged = unconverged(&out.replicas);
    out.converged = out.unconverged.is_empty();
    out.hit_cap = request.samples.is_none() && !out.converged;
    Ok(out)
}
