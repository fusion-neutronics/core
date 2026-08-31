//! D1S (Direct 1-Step) post-processing logic.
//!
//! Provides functions for shutdown dose rate calculations:
//! - Radionuclide discovery from chain reactions
//! - Time correction factor (TCF) computation
//! - Applying TCFs to tally results

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use yani::ChainNuclide;

use crate::decay_chain;

/// Find photon-emitting radionuclides reachable from a set of nuclides via
/// neutron reactions followed by radioactive decay.
///
/// For each input nuclide, follows its reactions to the produced target, then
/// walks the **full decay chain** from that target, collecting every
/// photon-emitting descendant (including the target itself). This captures
/// cases where the directly-produced nuclide is a pure beta emitter whose
/// daughter emits the gammas (e.g. Cs137 -> Ba137m, Mo99 -> Tc99m).
///
/// Returns a sorted `Vec` of unique emitter names - these are the parent-nuclide
/// tally bins, and the keys [`time_correction_factors`] is expected to scale.
pub fn get_radionuclides_from_chain(
    nuclide_names: &[String],
    chain: &Arc<HashMap<String, ChainNuclide>>,
) -> Vec<String> {
    let mut emitters = HashSet::new();

    for nuc_name in nuclide_names {
        let chain_nuclide = match chain.get(nuc_name.as_str()) {
            Some(cn) => cn,
            None => continue,
        };

        for reaction in &chain_nuclide.reactions {
            let target_name = match &reaction.target {
                Some(t) => t.as_str(),
                None => continue,
            };
            // Walk the decay chain from the produced target, collecting every
            // photon-emitting nuclide along it.
            for path in decay_chain::descendant_paths(chain, target_name) {
                emitters.insert(path.emitter);
            }
        }
    }

    let mut result: Vec<String> = emitters.into_iter().collect();
    result.sort();
    result
}

/// Compute time correction factors (TCFs) for decay-photon post-processing.
///
/// Each TCF is the Bateman **activity** of an emitter over the
/// irradiation/cooling schedule, given production into its chain root at the
/// `source_rates`. For an emitter that is itself directly produced (a length-1
/// chain) this reduces exactly to the classic single-nuclide recurrence
/// `h[i+1] = S*(1-exp(-λ·dt)) + h[i]*exp(-λ·dt)`; for a daughter emitter it
/// accounts for the parent's buildup/decay feeding the daughter (see
/// [`crate::decay_chain`]).
///
/// Returns `HashMap<String, Vec<f64>>` with `timesteps.len() + 1` values per
/// emitter (index 0 = 0.0, the pre-irradiation baseline).
pub fn time_correction_factors(
    nuclides: &[String],
    timesteps: &[f64],
    source_rates: &[f64],
    chain: &Arc<HashMap<String, ChainNuclide>>,
) -> Result<HashMap<String, Vec<f64>>, String> {
    if timesteps.len() != source_rates.len() {
        return Err(format!(
            "timesteps ({}) and source_rates ({}) must have the same length",
            timesteps.len(),
            source_rates.len()
        ));
    }

    // Map each photon-emitting nuclide to the decay path that produces it.
    let emitter_paths = decay_chain::build_emitter_paths(chain);

    let mut tcf = HashMap::new();
    for nuc in nuclides {
        // Prefer the chain path (daughter emitters); otherwise fall back to a
        // length-1 self-path for a directly-produced unstable emitter.
        let (lambdas, branchings) = match emitter_paths.get(nuc.as_str()) {
            Some(path) => (path.lambdas.clone(), path.branchings.clone()),
            None => {
                let half_life = chain
                    .get(nuc.as_str())
                    .and_then(|c| c.half_life)
                    .ok_or_else(|| {
                        format!("Nuclide '{}' not found in chain file or is stable", nuc)
                    })?;
                (vec![std::f64::consts::LN_2 / half_life], Vec::new())
            }
        };

        let h = decay_chain::evolve_chain_activity(&lambdas, &branchings, timesteps, source_rates);
        tcf.insert(nuc.clone(), h);
    }

    Ok(tcf)
}

/// Corrected `(means, std_devs)` with one flat row per requested timestep.
pub type CorrectedSteps = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Apply time correction factors to decay-photon tally results, for the
/// requested timesteps.
///
/// Bin layout (slowest to fastest): score -> parent -> energy*mesh
///
/// `steps` lists which TCF timestep indices to evaluate (e.g. `[0]` for the
/// baseline, the last index for end-of-cooling, or any subset). Returns one
/// corrected `(mean, std_dev)` row per entry in `steps`, in the same order.
/// Each row is a flat array:
/// - `sum_nuclides == true`:  length `n_scores * n_em_bins` (summed over the
///   parent dimension, uncertainties propagated in quadrature).
/// - `sum_nuclides == false`: length `n_parent_bins * n_scores * n_em_bins`,
///   ordered parent -> score -> em.
///
/// The input `mean`/`std_dev` arrays are validated and the bin layout parsed
/// once, then reused across all requested steps -- so callers get many cooling
/// steps without re-marshalling the (potentially large) tally arrays per step.
/// A `step` beyond a nuclide's TCF length simply contributes nothing for that
/// nuclide.
pub fn apply_time_correction(
    mean: &[f64],
    std_dev: &[f64],
    nuclides: &[String],
    n_scores: usize,
    tcf: &HashMap<String, Vec<f64>>,
    steps: &[usize],
    sum_nuclides: bool,
) -> Result<CorrectedSteps, String> {
    let n_parent_bins = nuclides.len();
    let n_total_bins = mean.len();

    if n_total_bins != std_dev.len() {
        return Err("mean and std_dev must have the same length".to_string());
    }

    if n_scores == 0 {
        return Err("n_scores must be greater than 0".to_string());
    }

    if n_parent_bins == 0 {
        // No D1S parent nuclides (a material with no gamma-emitting activation
        // products, e.g. pure Li6/Li7): the decay-photon source is empty, so
        // every corrected step is identically zero. The tally collapses the
        // absent parent dimension to a single bin, so n_total_bins ==
        // n_scores * n_em_bins. Return zeros shaped like a normal result rather
        // than erroring, matching OpenMC's D1S.
        if !n_total_bins.is_multiple_of(n_scores) {
            return Err(format!(
                "Total bins ({}) is not evenly divisible by scores ({})",
                n_total_bins, n_scores
            ));
        }
        let n_em_bins = n_total_bins / n_scores;
        // sum_nuclides: one summed row per step; otherwise no parent slices.
        let row_len = if sum_nuclides {
            n_scores * n_em_bins
        } else {
            0
        };
        let zeros: Vec<Vec<f64>> = (0..steps.len()).map(|_| vec![0.0; row_len]).collect();
        return Ok((zeros.clone(), zeros));
    }

    if !n_total_bins.is_multiple_of(n_scores * n_parent_bins) {
        return Err(format!(
            "Total bins ({}) is not evenly divisible by scores ({}) * parent bins ({})",
            n_total_bins, n_scores, n_parent_bins
        ));
    }

    let n_em_bins = n_total_bins / (n_scores * n_parent_bins);

    let mut out_means = Vec::with_capacity(steps.len());
    let mut out_stds = Vec::with_capacity(steps.len());
    for &step in steps {
        let (m, s) = apply_one_step(
            mean,
            std_dev,
            nuclides,
            n_scores,
            n_parent_bins,
            n_em_bins,
            tcf,
            step,
            sum_nuclides,
        );
        out_means.push(m);
        out_stds.push(s);
    }
    Ok((out_means, out_stds))
}

/// Apply the time correction factor for a single timestep `step`.
///
/// A nuclide whose TCF array is shorter than `step` simply contributes nothing
/// to that step (its inventory is undefined there), so this never fails.
#[allow(clippy::too_many_arguments)]
fn apply_one_step(
    mean: &[f64],
    std_dev: &[f64],
    nuclides: &[String],
    n_scores: usize,
    n_parent_bins: usize,
    n_em_bins: usize,
    tcf: &HashMap<String, Vec<f64>>,
    step: usize,
    sum_nuclides: bool,
) -> (Vec<f64>, Vec<f64>) {
    if sum_nuclides {
        let out_len = n_scores * n_em_bins;
        let mut total_mean = vec![0.0f64; out_len];
        let mut total_std_sq = vec![0.0f64; out_len];

        for (p_idx, nuc_name) in nuclides.iter().enumerate() {
            let tcf_arr = match tcf.get(nuc_name.as_str()) {
                Some(arr) => arr,
                None => continue,
            };

            let factor = match tcf_arr.get(step) {
                Some(&f) => f,
                None => continue,
            };

            for s_idx in 0..n_scores {
                let start = s_idx * (n_parent_bins * n_em_bins) + p_idx * n_em_bins;
                let out_start = s_idx * n_em_bins;

                for k in 0..n_em_bins {
                    let val = mean[start + k] * factor;
                    let sd = std_dev[start + k] * factor;
                    total_mean[out_start + k] += val;
                    total_std_sq[out_start + k] += sd * sd;
                }
            }
        }

        let total_std: Vec<f64> = total_std_sq.iter().map(|&v| v.sqrt()).collect();
        (total_mean, total_std)
    } else {
        let out_len = n_parent_bins * n_scores * n_em_bins;
        let mut corrected_mean = vec![0.0f64; out_len];
        let mut corrected_std = vec![0.0f64; out_len];

        for (p_idx, nuc_name) in nuclides.iter().enumerate() {
            let tcf_arr = match tcf.get(nuc_name.as_str()) {
                Some(arr) => arr,
                None => continue,
            };

            let factor = match tcf_arr.get(step) {
                Some(&f) => f,
                None => continue,
            };

            for s_idx in 0..n_scores {
                let in_start = s_idx * (n_parent_bins * n_em_bins) + p_idx * n_em_bins;
                let out_start = p_idx * (n_scores * n_em_bins) + s_idx * n_em_bins;

                for k in 0..n_em_bins {
                    corrected_mean[out_start + k] = mean[in_start + k] * factor;
                    corrected_std[out_start + k] = std_dev[in_start + k] * factor;
                }
            }
        }

        (corrected_mean, corrected_std)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yani::{ChainReaction, DecaySource, DecaySourceDistribution};

    fn make_test_chain() -> Arc<HashMap<String, ChainNuclide>> {
        let mut chain = HashMap::new();

        // Fe56: stable, has (n,gamma) -> Mn56
        chain.insert(
            "Fe56".to_string(),
            ChainNuclide {
                name: "Fe56".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Mn56".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Mn56: unstable, has photon source
        chain.insert(
            "Mn56".to_string(),
            ChainNuclide {
                name: "Mn56".to_string(),
                half_life: Some(9282.6),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![846764.0],
                        intensities: vec![7.381e-5],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Co59: stable, has (n,gamma) -> Co60
        chain.insert(
            "Co59".to_string(),
            ChainNuclide {
                name: "Co59".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("Co60".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Co60: unstable, has photon source
        chain.insert(
            "Co60".to_string(),
            ChainNuclide {
                name: "Co60".to_string(),
                half_life: Some(1.6632e8),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![1173228.0, 1332492.0],
                        intensities: vec![4.161e-9, 4.167e-9],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        // Ni60: stable, no photon source (should NOT appear in results)
        chain.insert(
            "Ni60".to_string(),
            ChainNuclide {
                name: "Ni60".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );

        Arc::new(chain)
    }

    // --- get_radionuclides_from_chain tests ---

    #[test]
    fn test_radionuclides_basic() {
        let chain = make_test_chain();
        let result = get_radionuclides_from_chain(&["Fe56".to_string()], &chain);
        assert_eq!(result, vec!["Mn56"]);
    }

    #[test]
    fn test_radionuclides_multiple_parents() {
        let chain = make_test_chain();
        let result =
            get_radionuclides_from_chain(&["Fe56".to_string(), "Co59".to_string()], &chain);
        assert_eq!(result, vec!["Co60", "Mn56"]); // sorted
    }

    #[test]
    fn test_radionuclides_unknown_nuclide() {
        let chain = make_test_chain();
        let result = get_radionuclides_from_chain(&["U235".to_string()], &chain);
        assert!(result.is_empty());
    }

    #[test]
    fn test_radionuclides_stable_target_excluded() {
        let chain = make_test_chain();
        let result =
            get_radionuclides_from_chain(&["Fe56".to_string(), "Co59".to_string()], &chain);
        assert!(!result.contains(&"Ni60".to_string()));
    }

    // --- time_correction_factors tests ---

    #[test]
    fn test_tcf_single_step() {
        let chain = make_test_chain();
        let dt = 3600.0;
        let src = 1e14;
        let hl = 9282.6;
        let decay_const = std::f64::consts::LN_2 / hl;
        let expected = src * (1.0 - (-decay_const * dt).exp());

        let tcf = time_correction_factors(&["Mn56".to_string()], &[dt], &[src], &chain).unwrap();

        assert_eq!(tcf["Mn56"].len(), 2); // N+1
        assert!((tcf["Mn56"][0] - 0.0).abs() < 1e-15);
        assert!((tcf["Mn56"][1] - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn test_tcf_cooling_step() {
        let chain = make_test_chain();
        let hl = 9282.6;
        let decay_const = std::f64::consts::LN_2 / hl;
        let dt_irr = 3600.0;
        let dt_cool = 3600.0;
        let src = 1e14;

        let tcf = time_correction_factors(
            &["Mn56".to_string()],
            &[dt_irr, dt_cool],
            &[src, 0.0],
            &chain,
        )
        .unwrap();

        let h_irr = src * (1.0 - (-decay_const * dt_irr).exp());
        let h_cool = h_irr * (-decay_const * dt_cool).exp();

        assert_eq!(tcf["Mn56"].len(), 3);
        assert!((tcf["Mn56"][2] - h_cool).abs() / h_cool < 1e-12);
    }

    #[test]
    fn test_tcf_multi_step() {
        let chain = make_test_chain();
        let hl = 9282.6;
        let decay_const = std::f64::consts::LN_2 / hl;

        let timesteps = [3600.0, 3600.0, 3600.0, 3600.0];
        let source_rates = [1e14, 2e14, 1e14, 0.0];

        let tcf = time_correction_factors(&["Mn56".to_string()], &timesteps, &source_rates, &chain)
            .unwrap();

        // Manually compute recurrence
        let mut expected = vec![0.0];
        let mut h_prev = 0.0;
        for (&dt, &src) in timesteps.iter().zip(source_rates.iter()) {
            let exp_term = (-decay_const * dt).exp();
            let h = if src != 0.0 {
                src * (1.0 - exp_term) + h_prev * exp_term
            } else {
                h_prev * exp_term
            };
            expected.push(h);
            h_prev = h;
        }

        assert_eq!(tcf["Mn56"].len(), 5);
        for (a, b) in tcf["Mn56"].iter().zip(expected.iter()) {
            if *b != 0.0 {
                assert!((a - b).abs() / b.abs() < 1e-12);
            } else {
                assert!(a.abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_tcf_length_mismatch() {
        let chain = make_test_chain();
        let result =
            time_correction_factors(&["Mn56".to_string()], &[3600.0, 7200.0], &[1e14], &chain);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("same length"));
    }

    #[test]
    fn test_tcf_unknown_nuclide() {
        let chain = make_test_chain();
        let result = time_correction_factors(&["Pu239".to_string()], &[3600.0], &[1e14], &chain);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    // --- apply_time_correction tests ---

    #[test]
    fn test_apply_tcf_sum_nuclides() {
        // 1 score, 2 parent nuclides, 3 energy*mesh bins
        let mean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let std_dev = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let nuclides = vec!["Mn56".to_string(), "Co60".to_string()];
        let mut tcf = HashMap::new();
        tcf.insert("Mn56".to_string(), vec![0.0, 2.0]);
        tcf.insert("Co60".to_string(), vec![0.0, 3.0]);

        let (cm, cs) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[0, 1], true).unwrap();
        // One row per requested step (here 2: baseline + one step).
        assert_eq!(cm.len(), 2);
        assert_eq!(cs.len(), 2);
        // Step 0 is the pre-irradiation baseline (factor 0 -> all zeros).
        assert!(cm[0].iter().all(|&v| v == 0.0));

        // Step 1: expected_mean = [1*2+4*3, 2*2+5*3, 3*2+6*3] = [14, 19, 24]
        let cm = &cm[1];
        let cs = &cs[1];
        assert_eq!(cm.len(), 3);
        assert!((cm[0] - 14.0).abs() < 1e-10);
        assert!((cm[1] - 19.0).abs() < 1e-10);
        assert!((cm[2] - 24.0).abs() < 1e-10);

        // expected_std = sqrt((0.1*2)^2 + (0.4*3)^2) etc.
        let exp_s0 = ((0.1 * 2.0_f64).powi(2) + (0.4 * 3.0_f64).powi(2)).sqrt();
        assert!((cs[0] - exp_s0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_tcf_no_parent_nuclides_is_zero() {
        // A material with no gamma-emitting activation products (e.g. Li6/Li7)
        // has an empty parent-nuclide set. The tally collapses the parent
        // dimension to one bin, so the mean holds n_scores * n_em_bins entries;
        // the corrected result must be all zeros of that shape, not an error.
        let mean = vec![0.0, 0.0, 0.0]; // 1 score, collapsed parent (1), 3 em bins
        let std_dev = vec![0.0, 0.0, 0.0];
        let nuclides: Vec<String> = vec![];
        let tcf = HashMap::new();

        let (cm, cs) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[0, 1], true).unwrap();
        assert_eq!(cm.len(), 2);
        assert_eq!(cs.len(), 2);
        for row in cm.iter().chain(cs.iter()) {
            assert_eq!(row.len(), 3);
            assert!(row.iter().all(|&v| v == 0.0));
        }

        // Non-sum form has no parent slices, so each row is empty.
        let (per_nuc, _) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[1], false).unwrap();
        assert_eq!(per_nuc.len(), 1);
        assert!(per_nuc[0].is_empty());
    }

    #[test]
    fn test_apply_tcf_no_sum() {
        let mean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let std_dev = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let nuclides = vec!["Mn56".to_string(), "Co60".to_string()];
        let mut tcf = HashMap::new();
        tcf.insert("Mn56".to_string(), vec![0.0, 2.0]);
        tcf.insert("Co60".to_string(), vec![0.0, 3.0]);

        // Request only the single non-baseline step.
        let (cm, _cs) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[1], false).unwrap();
        assert_eq!(cm.len(), 1);
        let cm = &cm[0];

        // Output: parent -> score -> em, flattened
        // Mn56 bins: [1*2, 2*2, 3*2] = [2, 4, 6]
        // Co60 bins: [4*3, 5*3, 6*3] = [12, 15, 18]
        assert_eq!(cm.len(), 6);
        assert!((cm[0] - 2.0).abs() < 1e-10);
        assert!((cm[1] - 4.0).abs() < 1e-10);
        assert!((cm[2] - 6.0).abs() < 1e-10);
        assert!((cm[3] - 12.0).abs() < 1e-10);
        assert!((cm[4] - 15.0).abs() < 1e-10);
        assert!((cm[5] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_tcf_multiple_steps() {
        // One call returns one row per requested step, in the given order.
        let mean = vec![1.0, 2.0];
        let std_dev = vec![0.1, 0.2];
        let nuclides = vec!["Mn56".to_string()];
        let mut tcf = HashMap::new();
        tcf.insert("Mn56".to_string(), vec![0.0, 5.0, 10.0]);

        let (cm, _) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[0, 1, 2], true).unwrap();

        // 3 steps -> 3 rows, each scaled by the matching TCF factor.
        assert_eq!(cm.len(), 3);
        assert!((cm[0][0] - 0.0).abs() < 1e-10); // step 0: factor 0.0
        assert!((cm[0][1] - 0.0).abs() < 1e-10);
        assert!((cm[1][0] - 5.0).abs() < 1e-10); // step 1: factor 5.0
        assert!((cm[1][1] - 10.0).abs() < 1e-10);
        assert!((cm[2][0] - 10.0).abs() < 1e-10); // step 2: factor 10.0
        assert!((cm[2][1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_tcf_step_subset_order_preserved() {
        // A subset in a custom order returns rows in that order.
        let mean = vec![1.0, 2.0];
        let std_dev = vec![0.1, 0.2];
        let nuclides = vec!["Mn56".to_string()];
        let mut tcf = HashMap::new();
        tcf.insert("Mn56".to_string(), vec![0.0, 5.0, 10.0]);

        let (cm, _) =
            apply_time_correction(&mean, &std_dev, &nuclides, 1, &tcf, &[2, 0], true).unwrap();
        assert_eq!(cm.len(), 2);
        assert!((cm[0][0] - 10.0).abs() < 1e-10); // step 2 first
        assert!((cm[1][0] - 0.0).abs() < 1e-10); // step 0 second
    }

    // --- decay-chain (daughter emitter) tests ---

    /// X --(n,g)--> A (beta, no photons) --> B (photon emitter).
    fn beta_parent_chain() -> Arc<HashMap<String, ChainNuclide>> {
        use yani::{ChainReaction, DecaySource, DecaySourceDistribution};
        let mut c = HashMap::new();
        c.insert(
            "X".to_string(),
            ChainNuclide {
                name: "X".to_string(),
                half_life: None,
                decay_energy: 0.0,
                reactions: vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("A".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                decays: vec![],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        c.insert(
            "A".to_string(),
            ChainNuclide {
                name: "A".to_string(),
                half_life: Some(3600.0),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                fission_yields: None,
                sources: vec![],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        c.insert(
            "B".to_string(),
            ChainNuclide {
                name: "B".to_string(),
                half_life: Some(600.0),
                decay_energy: 0.0,
                reactions: vec![],
                decays: vec![],
                fission_yields: None,
                sources: vec![DecaySource {
                    particle: "photon".to_string(),
                    distribution: DecaySourceDistribution::Discrete {
                        energies: vec![1.0e6],
                        intensities: vec![1.0e-3],
                    },
                }],
                half_life_uncertainty: None,
                decay_energy_uncertainty: None,
            },
        );
        Arc::new(c)
    }

    #[test]
    fn test_radionuclides_follow_decay_chain() {
        // The directly-produced nuclide A emits nothing; discovery must still
        // surface its photon-emitting daughter B. (Single-generation code
        // returned [] here.)
        let chain = beta_parent_chain();
        let result = get_radionuclides_from_chain(&["X".to_string()], &chain);
        assert_eq!(result, vec!["B"]);
    }

    #[test]
    fn test_tcf_daughter_builds_up_and_cools() {
        // TCF for the daughter emitter B must be nonzero (it builds up via A's
        // decay during irradiation) and then decay away over long cooling.
        // The cooling step is several parent (A) half-lives so the eventual
        // decay dominates the brief transient-equilibrium catch-up that can
        // make a fast daughter's activity peak just after production stops.
        let chain = beta_parent_chain();
        let timesteps = [3600.0, 3600.0, 14400.0];
        let source_rates = [1.0e12, 1.0e12, 0.0]; // two irradiation steps, one long cool
        let tcf =
            time_correction_factors(&["B".to_string()], &timesteps, &source_rates, &chain).unwrap();
        let h = &tcf["B"];
        assert_eq!(h.len(), 4);
        assert_eq!(h[0], 0.0);
        assert!(h[1] > 0.0, "daughter activity should build up: {h:?}");
        assert!(
            h[2] > h[1],
            "activity should keep building under irradiation"
        );
        assert!(
            h[3] < h[2],
            "activity should decay over long cooling: {h:?}"
        );
    }
}
