//! Radioactive decay-chain enumeration and Bateman activity evolution.
//!
//! The decay-photon (D1S-derived) shutdown-dose method emits, at a neutron
//! collision, the decay photons of the nuclide produced by the reaction. A
//! produced nuclide `A` may be a pure beta emitter whose *daughter* `B` is the
//! gamma emitter (e.g. Cs137 -> Ba137m, Mo99 -> Tc99m). To capture those, we
//! walk the decay chain from each produced nuclide ("root") to every
//! photon-emitting descendant ("emitter"), and the time-correction factor for
//! an emitter is its Bateman *activity* given production into the root.
//!
//! The activity is evolved one irradiation/cooling step at a time with a small
//! matrix exponential of the linear-chain transition matrix (augmented with a
//! constant production term). For a length-1 path this reduces exactly to the
//! classic single-nuclide recurrence
//! `h[i+1] = S*(1-exp(-λ·dt)) + h[i]*exp(-λ·dt)`.

use std::collections::{HashMap, HashSet};

use yani::ChainNuclide;

/// Maximum decay-chain depth walked from a root (guards pathological data).
const MAX_CHAIN_DEPTH: usize = 24;

/// A linear decay path from a production root to a photon-emitting nuclide.
///
/// `lambdas[0]` is the root's decay constant and `lambdas.last()` the emitter's;
/// `branchings[i]` is the branching fraction from node `i` to node `i+1`
/// (so `branchings.len() == lambdas.len() - 1`).
#[derive(Clone, Debug)]
pub struct DecayChainPath {
    /// Name of the photon-emitting nuclide at the end of the path.
    pub emitter: String,
    /// Decay constants (s^-1) for each node, root first.
    pub lambdas: Vec<f64>,
    /// Branching fraction between successive nodes (length = lambdas.len() - 1).
    pub branchings: Vec<f64>,
}

impl DecayChainPath {
    /// Product of branchings along the whole path (1.0 for a length-1 path).
    pub fn path_branching(&self) -> f64 {
        self.branchings.iter().product()
    }
}

fn decay_constant(half_life: Option<f64>) -> Option<f64> {
    match half_life {
        Some(t) if t > 0.0 => Some(std::f64::consts::LN_2 / t),
        _ => None,
    }
}

fn emits_photons(nuc: &ChainNuclide) -> bool {
    nuc.sources.iter().any(|s| s.particle == "photon")
}

/// Enumerate every photon-emitting nuclide reachable from `root` by radioactive
/// decay (including `root` itself), returning the linear path to each.
///
/// `root` must be an unstable nuclide present in `chain`; otherwise an empty
/// `Vec` is returned. Cyclic / re-visited nuclides are skipped, and depth is
/// capped at [`MAX_CHAIN_DEPTH`].
pub fn descendant_paths(chain: &HashMap<String, ChainNuclide>, root: &str) -> Vec<DecayChainPath> {
    let mut out = Vec::new();
    let root_nuc = match chain.get(root) {
        Some(n) => n,
        None => return out,
    };
    let root_lambda = match decay_constant(root_nuc.half_life) {
        Some(l) => l,
        None => return out,
    };
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(root.to_string());
    walk(
        chain,
        root,
        root_nuc,
        &mut vec![root_lambda],
        &mut Vec::new(),
        &mut visited,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn walk(
    chain: &HashMap<String, ChainNuclide>,
    node_name: &str,
    node: &ChainNuclide,
    lambdas: &mut Vec<f64>,
    branchings: &mut Vec<f64>,
    visited: &mut HashSet<String>,
    out: &mut Vec<DecayChainPath>,
) {
    if emits_photons(node) {
        out.push(DecayChainPath {
            emitter: node_name.to_string(),
            lambdas: lambdas.clone(),
            branchings: branchings.clone(),
        });
    }
    if lambdas.len() > MAX_CHAIN_DEPTH {
        return;
    }
    for decay in &node.decays {
        let target_name = match &decay.target {
            Some(t) => t,
            None => continue,
        };
        if visited.contains(target_name) {
            continue;
        }
        let child = match chain.get(target_name) {
            Some(c) => c,
            None => continue,
        };
        // Only unstable children continue the chain (stable nuclides have no
        // decay activity, hence no decay photons).
        let child_lambda = match decay_constant(child.half_life) {
            Some(l) => l,
            None => continue,
        };
        visited.insert(target_name.clone());
        lambdas.push(child_lambda);
        branchings.push(decay.branching);
        walk(chain, target_name, child, lambdas, branchings, visited, out);
        lambdas.pop();
        branchings.pop();
        visited.remove(target_name);
    }
}

/// Build a map from each photon-emitting nuclide to the decay path that
/// produces it, rooted at an unstable neutron-reaction target in `chain`.
///
/// When an emitter is reachable from several roots (rare), the shortest path
/// wins (ties broken by sorted root name) - this is the single-production-root
/// approximation noted in the module docs.
pub fn build_emitter_paths(
    chain: &HashMap<String, ChainNuclide>,
) -> HashMap<String, DecayChainPath> {
    // Roots = unstable nuclides that appear as a neutron-reaction target.
    let mut roots: Vec<String> = HashSet::<String>::from_iter(
        chain
            .values()
            .flat_map(|n| n.reactions.iter())
            .filter_map(|r| r.target.clone())
            .filter(|t| chain.get(t).and_then(|c| c.half_life).is_some()),
    )
    .into_iter()
    .collect();
    roots.sort();

    let mut map: HashMap<String, DecayChainPath> = HashMap::new();
    for root in &roots {
        for path in descendant_paths(chain, root) {
            match map.get(&path.emitter) {
                Some(existing) if existing.lambdas.len() <= path.lambdas.len() => {}
                _ => {
                    map.insert(path.emitter.clone(), path);
                }
            }
        }
    }
    map
}

/// Evolve the Bateman *activity* of the emitter (last node) across an
/// irradiation/cooling schedule, returning `timesteps.len() + 1` values with
/// index 0 = 0.0 (pre-irradiation baseline) and index `i` = activity at the end
/// of step `i-1`.
///
/// `source_rates[i]` is the production rate into the root during step `i`
/// (0.0 for a cooling step). Production into descendants happens only by decay.
pub fn evolve_chain_activity(
    lambdas: &[f64],
    branchings: &[f64],
    timesteps: &[f64],
    source_rates: &[f64],
) -> Vec<f64> {
    let m = lambdas.len();
    debug_assert_eq!(branchings.len(), m.saturating_sub(1));

    // Pure-decay transition matrix D (s^-1): dN/dt = D N + b, with constant
    // production b = [src, 0, ..] into the root over the step.
    //   D[k][k]     = -λ_k
    //   D[k][k-1]   =  b_{k-1} λ_{k-1}
    // Per step: N(dt) = E·N(0) + D^{-1}(E - I)·b, with E = exp(D·dt). Keeping
    // the source out of the matrix avoids the huge src/λ dynamic range that
    // wrecks scaling-and-squaring; D is lower-triangular so D^{-1}·r is a cheap
    // forward substitution.
    let mut d = vec![vec![0.0f64; m]; m];
    for k in 0..m {
        d[k][k] = -lambdas[k];
    }
    for k in 1..m {
        d[k][k - 1] = branchings[k - 1] * lambdas[k - 1];
    }

    let mut state = vec![0.0f64; m];
    let mut h = vec![0.0f64; timesteps.len() + 1];
    for (i, (&dt, &src)) in timesteps.iter().zip(source_rates.iter()).enumerate() {
        let e = expm(&d, dt);
        // homogeneous part E·N0
        let mut next = matvec(&e, &state);
        if src != 0.0 {
            // particular part: solve D·x = (E - I)·b, b = [src,0,..].
            // (E - I)·b = src·(first column of E - e0).
            let mut r = vec![0.0f64; m];
            for (k, rk) in r.iter_mut().enumerate() {
                let ek0 = e[k][0] - if k == 0 { 1.0 } else { 0.0 };
                *rk = src * ek0;
            }
            // Forward substitution on lower-triangular D (diagonal -λ_k).
            let mut x = vec![0.0f64; m];
            for k in 0..m {
                let sub = if k > 0 {
                    branchings[k - 1] * lambdas[k - 1] * x[k - 1]
                } else {
                    0.0
                };
                x[k] = (r[k] - sub) / (-lambdas[k]);
            }
            for k in 0..m {
                next[k] += x[k];
            }
        }
        state = next;
        h[i + 1] = lambdas[m - 1] * state[m - 1];
    }
    h
}

// =============================================================================
// Small dense matrix exponential (scaling-and-squaring + Taylor series).
// Matrices here are tiny (<= ~6x6) with non-positive real eigenvalues, so this
// is robust and cheap.
// =============================================================================

/// Compute `exp(a * dt)` for a small square matrix `a`.
fn expm(a: &[Vec<f64>], dt: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    // m = a * dt
    let mut m = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a[i][j] * dt;
        }
    }

    // Scale so that ||m||_inf / 2^s <= 0.5, then square s times at the end.
    let norm = inf_norm(&m);
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 2f64.powi(s as i32);
    if scale != 1.0 {
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v /= scale;
            }
        }
    }

    // Taylor: exp(m) = I + m + m^2/2! + ...
    let mut result = identity(n);
    let mut term = identity(n);
    for k in 1..=18u32 {
        term = matmul(&term, &m);
        let inv = 1.0 / (k as f64);
        for row in term.iter_mut() {
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        add_inplace(&mut result, &term);
    }

    for _ in 0..s {
        result = matmul(&result, &result);
    }
    result
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0f64; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[i][j] * x[j];
        }
        y[i] = acc;
    }
    y
}

fn add_inplace(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for (ra, rb) in a.iter_mut().zip(b.iter()) {
        for (va, vb) in ra.iter_mut().zip(rb.iter()) {
            *va += vb;
        }
    }
}

fn inf_norm(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use yani::{ChainNuclide, ChainReaction, DecaySource, DecaySourceDistribution};

    fn photon_source() -> Vec<DecaySource> {
        vec![DecaySource {
            particle: "photon".to_string(),
            distribution: DecaySourceDistribution::Discrete {
                energies: vec![1.0e6],
                intensities: vec![1.0],
            },
        }]
    }

    fn nuc(
        name: &str,
        half_life: Option<f64>,
        reactions: Vec<ChainReaction>,
        decays: Vec<ChainReaction>,
        sources: Vec<DecaySource>,
    ) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            decay_energy: 0.0,
            reactions,
            decays,
            fission_yields: None,
            sources,
            half_life_uncertainty: None,
            decay_energy_uncertainty: None,
        }
    }

    /// X --(n,g)--> A (beta, no photons) --decays--> B (photons).
    fn beta_parent_chain() -> HashMap<String, ChainNuclide> {
        let mut c = HashMap::new();
        c.insert(
            "X".to_string(),
            nuc(
                "X",
                None,
                vec![ChainReaction {
                    kind: "(n,gamma)".to_string(),
                    target: Some("A".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![],
                vec![],
            ),
        );
        c.insert(
            "A".to_string(),
            nuc(
                "A",
                Some(3600.0),
                vec![],
                vec![ChainReaction {
                    kind: "beta-".to_string(),
                    target: Some("B".to_string()),
                    branching: 1.0,
                    q_value: None,
                }],
                vec![], // no photons on the parent
            ),
        );
        c.insert(
            "B".to_string(),
            nuc("B", Some(600.0), vec![], vec![], photon_source()),
        );
        c
    }

    #[test]
    fn descendant_paths_reaches_grandchild_emitter() {
        let chain = beta_parent_chain();
        let paths = descendant_paths(&chain, "A");
        // A has no photons; only B is an emitter, via A -> B.
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].emitter, "B");
        assert_eq!(paths[0].lambdas.len(), 2);
        assert_eq!(paths[0].branchings, vec![1.0]);
    }

    #[test]
    fn build_emitter_paths_keys_by_emitter() {
        let chain = beta_parent_chain();
        let map = build_emitter_paths(&chain);
        // Root A is discovered as a reaction target; emitter is B.
        assert!(map.contains_key("B"));
        assert_eq!(map["B"].lambdas.len(), 2);
    }

    #[test]
    fn single_node_matches_scalar_recurrence() {
        // Length-1 path must reproduce h[i+1] = S(1-e^{-λdt}) + h[i]e^{-λdt}.
        let lambda = std::f64::consts::LN_2 / 9282.6; // Mn56-like
        let timesteps = [3600.0, 3600.0, 3600.0, 3600.0];
        let source_rates = [1e14, 2e14, 1e14, 0.0];
        let h = evolve_chain_activity(&[lambda], &[], &timesteps, &source_rates);

        let mut expected = vec![0.0];
        let mut prev = 0.0;
        for (&dt, &s) in timesteps.iter().zip(source_rates.iter()) {
            let e = (-lambda * dt).exp();
            let next = if s != 0.0 {
                s * (1.0 - e) + prev * e
            } else {
                prev * e
            };
            expected.push(next);
            prev = next;
        }
        assert_eq!(h.len(), expected.len());
        for (a, b) in h.iter().zip(expected.iter()) {
            if *b != 0.0 {
                assert!((a - b).abs() / b.abs() < 1e-9, "{a} vs {b}");
            } else {
                assert!(a.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn short_lived_daughter_reaches_secular_equilibrium() {
        // Parent A (long) produced at rate S until it saturates; the short-lived
        // daughter B then sits in secular equilibrium with activity ~= S (and
        // ~= the parent's). (A rising parent leaves the daughter lagging by
        // ~(λ_a/λ_b)·S, so we irradiate long enough to saturate A first.)
        let lambda_a = std::f64::consts::LN_2 / 1.0e6; // long
        let lambda_b = std::f64::consts::LN_2 / 10.0; // short
        let s = 1.0e12;
        let timesteps = vec![5.0e5; 30]; // 15 parent half-lives -> saturated
        let source_rates = vec![s; 30];

        let daughter =
            *evolve_chain_activity(&[lambda_a, lambda_b], &[1.0], &timesteps, &source_rates)
                .last()
                .unwrap();
        let parent = *evolve_chain_activity(&[lambda_a], &[], &timesteps, &source_rates)
            .last()
            .unwrap();

        assert!((parent / s - 1.0).abs() < 1e-3, "parent {parent} vs S {s}");
        assert!(
            (daughter - parent).abs() / parent < 1e-4,
            "daughter {daughter} vs parent {parent}"
        );
    }

    #[test]
    fn cooling_decays_emitter_activity() {
        // After irradiation, a cooling step reduces the emitter activity.
        let lambda = std::f64::consts::LN_2 / 600.0;
        let timesteps = [3600.0, 600.0];
        let source_rates = [1e12, 0.0];
        let h = evolve_chain_activity(&[lambda], &[], &timesteps, &source_rates);
        assert!(h[2] < h[1]);
        // One half-life of cooling halves the activity.
        assert!((h[2] / h[1] - 0.5).abs() < 1e-6);
    }
}
