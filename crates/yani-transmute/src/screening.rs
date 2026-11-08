//! Which products of a transmutation need their cross sections loaded.
//!
//! The chain's reachable closure is a graph answer, and from almost any seed it
//! saturates on one large strongly connected component: on ENDF/B-VIII.1, Fe56,
//! Al27 and everything from Be9 up reach the same 465 nuclides with neutron
//! data, and loading them all is hundreds of MB. An iron sphere under a fusion
//! spectrum populates a few dozen.
//!
//! [`screen_products`] finds the ones that matter by solving. It loads the
//! composition, solves the schedule, and loads every product with neutron
//! reactions that, at any state of the schedule, reached `threshold` as
//! either
//!
//! * its atom fraction, or
//! * its share of the activity, the decay heat or the decay photon-line power;
//!
//! then it solves again with those products reacting, and repeats until a solve
//! populates nothing new above the threshold. A product left out is still
//! produced, still decays and is still reported: only its own neutron-induced
//! reactions are dropped.
//!
//! # What dropping a product can change
//!
//! Two things, both bounded by what the final solve says about the product:
//!
//! * Its own inventory comes out high, since the reactions that would remove
//!   it are missing. The screening tests the inventory with them missing, so
//!   the true one is lower still, and the error in its atom fraction and in
//!   its share of each decay quantity is below `threshold`. This is why the
//!   decay quantities are tested as well as the atom fraction: Na22 in an
//!   irradiated Li4SiO4 has a 28 kb thermal (n,p) cross section, burns out
//!   twelvefold, and a month after shutdown carries nearly all of a contact
//!   dose that is by then twelve orders below its peak. At an atom fraction of
//!   1e-15 it is far below any atom-fraction threshold worth having.
//! * What its reactions would have sent onwards. An atom reacts at most once
//!   per reaction, so that is at most its own atoms times the share of them
//!   that react, `min(1, sigma phi T)`, with a few light ejectiles per
//!   reaction on top. Summed over the products left out it is bounded by
//!   [`ProductScreening::dropped_fraction`], which is reported.
//!
//! At the default of 1e-12, against the whole closure under a 14 MeV fusion
//! spectrum (Fe56, Al27, a W/Eurofer/SS316 mix and Li4SiO4, a year at 1e14
//! n/cm2/s and cooling to a hundred years), no atom fraction moved by more than
//! 4e-13 and no activity, decay heat or contact dose at any step by more than
//! 3e-8 of itself.
//!
//! A threshold of 0 loads every product the solve populates at all, which is
//! exact: a product the solve never populates cannot react.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use yani::ChainNuclide;

/// The default threshold for loading a product's cross sections, as an atom
/// fraction and as a share of each decay quantity. See the module
/// documentation for what it bounds.
pub const DEFAULT_PRODUCT_THRESHOLD: f64 = 1e-12;

/// Which products a transmutation loaded cross sections for, and which it
/// left out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProductScreening {
    /// The threshold the screening applied, to atom fractions and to shares of
    /// the activity, decay heat and decay photon-line power.
    pub threshold: f64,
    /// Screening solves run. One means the composition's own reactions
    /// populated nothing new above the threshold.
    pub rounds: usize,
    /// Nuclides with neutron reactions in the chain's reachable closure of the
    /// composition: what loading the whole closure would have loaded.
    pub closure: usize,
    /// Nuclides whose cross sections were asked for: the composition's
    /// reactive nuclides and every product that crossed the threshold.
    pub requested: BTreeSet<String>,
    /// Of `requested`, the ones the library had no cross sections for. They
    /// still decay and are still produced, and do not react.
    pub unavailable: BTreeSet<String>,
    /// Reactive products the final solve populated below the threshold, each
    /// with its largest atom fraction over the schedule. Their neutron
    /// reactions were left out.
    pub below_threshold: BTreeMap<String, f64>,
}

impl ProductScreening {
    /// Nuclides whose cross sections were loaded and drive the solve.
    pub fn loaded(&self) -> BTreeSet<String> {
        self.requested
            .difference(&self.unavailable)
            .cloned()
            .collect()
    }

    /// The summed largest atom fraction of every product left out: an upper
    /// bound on the share of the material's atoms those products could have
    /// sent onwards by reacting, before multiplying by the share of them that
    /// reacts.
    pub fn dropped_fraction(&self) -> f64 {
        self.below_threshold.values().sum()
    }
}

/// Refuse a threshold that is not a fraction.
pub fn check_threshold(threshold: f64) -> Result<(), String> {
    if threshold.is_finite() && (0.0..1.0).contains(&threshold) {
        Ok(())
    } else {
        Err(format!(
            "product_threshold must be an atom fraction in [0, 1), got {threshold}; \
             0 loads every product the solve populates"
        ))
    }
}

/// Grow the set of nuclides whose cross sections are loaded until a solve
/// populates no unloaded reactive product above `threshold`, as an atom
/// fraction or a share of the activity, decay heat or decay photon-line power.
///
/// `round(to_load, loaded)` loads the nuclides in `to_load`, then solves the
/// schedule with the reactions of the nuclides in `loaded` (which includes
/// `to_load`) and nothing else. It returns the names of `to_load` the library
/// had data for, and the atom densities of every state the solve passed
/// through, the initial one included. The fraction tested is each nuclide's
/// density over the state's total, and the shares are taken with the chain's
/// half-lives, mean decay energies and photon lines.
///
/// Terminates: every round but the last asks for at least one nuclide not
/// asked for before, and there are finitely many in the chain.
pub fn screen_products<R>(
    chain: &HashMap<String, ChainNuclide>,
    composition: &[String],
    threshold: f64,
    mut round: R,
) -> Result<ProductScreening, Box<dyn std::error::Error>>
where
    R: FnMut(
        &BTreeSet<String>,
        &BTreeSet<String>,
    )
        -> Result<(BTreeSet<String>, Vec<HashMap<String, f64>>), Box<dyn std::error::Error>>,
{
    check_threshold(threshold)?;
    let reactive = |name: &str| chain.get(name).is_some_and(|cn| !cn.reactions.is_empty());
    let seeds: Vec<&str> = composition.iter().map(String::as_str).collect();
    let closure = yani::reachable_nuclides(chain, &seeds)
        .iter()
        .filter(|n| reactive(n))
        .count();

    let mut requested: BTreeSet<String> = composition
        .iter()
        .filter(|n| reactive(n))
        .cloned()
        .collect();
    let mut loaded: BTreeSet<String> = BTreeSet::new();
    let mut unavailable: BTreeSet<String> = BTreeSet::new();
    let mut to_load = requested.clone();
    let mut rounds = 0;
    loop {
        loaded.extend(to_load.iter().cloned());
        let (found, states) = round(&to_load, &loaded)?;
        for name in &to_load {
            if !found.contains(name) {
                loaded.remove(name);
                unavailable.insert(name.clone());
            }
        }
        rounds += 1;

        let peaks = peak_fractions(chain, &states);
        let mut next = BTreeSet::new();
        let mut below = BTreeMap::new();
        for (name, &(fraction, share)) in &peaks {
            if fraction <= 0.0 || requested.contains(name) || !reactive(name) {
                continue;
            }
            if fraction >= threshold || share >= threshold {
                next.insert(name.clone());
            } else {
                below.insert(name.clone(), fraction);
            }
        }
        if next.is_empty() {
            return Ok(ProductScreening {
                threshold,
                rounds,
                closure,
                requested,
                unavailable,
                below_threshold: below,
            });
        }
        requested.extend(next.iter().cloned());
        to_load = next;
    }
}

/// Each nuclide's largest atom fraction over `states`, and its largest share
/// of the activity, the decay heat or the decay photon-line power of any of
/// them.
fn peak_fractions(
    chain: &HashMap<String, ChainNuclide>,
    states: &[HashMap<String, f64>],
) -> HashMap<String, (f64, f64)> {
    // Per atom: activity [1/s], decay heat [eV/s], photon-line power [eV/s].
    let mut weights: HashMap<&str, [f64; 3]> = HashMap::new();
    let mut weight = |name: &str| -> [f64; 3] {
        if let Some(w) = weights.get(name) {
            return *w;
        }

        match chain.get_key_value(name) {
            Some((key, cn)) => {
                let lambda = match cn.half_life {
                    Some(t) if t > 0.0 => std::f64::consts::LN_2 / t,
                    _ => 0.0,
                };
                let photons: f64 = cn.photon_lines().iter().map(|(e, r)| e * r).sum();
                let w = [lambda, lambda * cn.decay_energy, photons];
                weights.insert(key.as_str(), w);
                w
            }
            None => [0.0; 3],
        }
    };
    let mut peaks: HashMap<String, (f64, f64)> = HashMap::new();
    for state in states {
        // Summed in name order. A fraction is compared against the threshold,
        // so a last-bit difference in a total could move a product across it,
        // and under MPI every rank screens for itself and must load the same
        // products: their tallies are reduced into one layout.
        let mut names: Vec<&String> = state.keys().collect();
        names.sort_unstable();
        let mut total = 0.0;
        let mut totals = [0.0; 3];
        for name in names {
            let n = state[name];
            if n > 0.0 {
                total += n;
                let w = weight(name);
                for q in 0..3 {
                    totals[q] += w[q] * n;
                }
            }
        }
        if total <= 0.0 {
            continue;
        }
        for (name, &n) in state {
            if n <= 0.0 {
                continue;
            }
            let w = weight(name);
            let share = (0..3)
                .filter(|&q| totals[q] > 0.0)
                .map(|q| w[q] * n / totals[q])
                .fold(0.0, f64::max);
            let peak = peaks.entry(name.clone()).or_insert((0.0, 0.0));
            peak.0 = peak.0.max(n / total);
            peak.1 = peak.1.max(share);
        }
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reactive(name: &str, target: &str) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life: None,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_evaluated: true,
            excitation_energy: None,
            level_index: None,
            decay_energy_components: Default::default(),
            reactions: vec![yani::ChainReaction {
                kind: "(n,gamma)".to_string(),
                target: Some(target.to_string()),
                branching: 1.0,
                q_value: None,
                branching_uncertainty: None,
                evaluated_branching: None,
                multiplicity: None,
            }],
            decays: vec![],
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    /// A -> B -> C -> D -> E by capture, every rung reactive.
    fn ladder() -> HashMap<String, ChainNuclide> {
        ["A", "B", "C", "D", "E"]
            .windows(2)
            .map(|w| (w[0].to_string(), reactive(w[0], w[1])))
            .collect()
    }

    /// A solve where each loaded nuclide feeds the next rung at 1e-6 of
    /// itself, and an unloaded one feeds nothing.
    fn solve(loaded: &BTreeSet<String>) -> Vec<HashMap<String, f64>> {
        let order = ["A", "B", "C", "D"];
        let mut state = HashMap::from([("A".to_string(), 1.0)]);
        for w in order.windows(2) {
            let parent = state.get(w[0]).copied().unwrap_or(0.0);
            if loaded.contains(w[0]) && parent > 0.0 {
                state.insert(w[1].to_string(), parent * 1e-6);
            }
        }
        vec![HashMap::from([("A".to_string(), 1.0)]), state]
    }

    fn run(threshold: f64) -> ProductScreening {
        screen_products(
            &ladder(),
            &["A".to_string()],
            threshold,
            |to_load, loaded| Ok((to_load.clone(), solve(loaded))),
        )
        .unwrap()
    }

    #[test]
    fn grows_rung_by_rung_and_stops_at_the_threshold() {
        let s = run(1e-9);
        assert_eq!(s.loaded(), BTreeSet::from(["A".into(), "B".into()]));
        assert_eq!(s.rounds, 2);
        let below: Vec<&String> = s.below_threshold.keys().collect();
        assert_eq!(below, vec!["C"]);
        assert!((s.below_threshold["C"] / 1e-12 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_product_carrying_a_decay_quantity_is_loaded_however_few_its_atoms() {
        // C is the only radioactive rung, so it carries all of the activity
        // at an atom fraction of 1e-12, under a 1e-9 threshold.
        let mut chain = ladder();
        chain.get_mut("C").unwrap().half_life = Some(3600.0);
        let s = screen_products(&chain, &["A".to_string()], 1e-9, |to_load, loaded| {
            Ok((to_load.clone(), solve(loaded)))
        })
        .unwrap();
        assert!(s.loaded().contains("C"));
        assert_eq!(s.rounds, 3);
        // D is fed only through C and is stable, so it is left out.
        assert_eq!(s.below_threshold.keys().collect::<Vec<_>>(), vec!["D"]);
    }

    #[test]
    fn zero_loads_everything_the_solve_populates() {
        let s = run(0.0);
        assert_eq!(s.loaded().len(), 4);
        assert!(s.below_threshold.is_empty());
        assert_eq!(s.rounds, 4);
        assert_eq!(s.closure, 4);
    }

    #[test]
    fn a_product_the_library_lacks_is_reported_and_not_retried() {
        let mut calls = 0;
        let s = screen_products(&ladder(), &["A".to_string()], 0.0, |to_load, loaded| {
            calls += 1;
            let found: BTreeSet<String> = to_load.iter().filter(|n| *n != "B").cloned().collect();
            let mut usable = loaded.clone();
            usable.remove("B");
            Ok((found, solve(&usable)))
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(s.unavailable, BTreeSet::from(["B".to_string()]));
        assert_eq!(s.loaded(), BTreeSet::from(["A".to_string()]));
    }

    #[test]
    fn a_threshold_outside_zero_to_one_is_refused() {
        for bad in [-1e-12, 1.0, f64::NAN, f64::INFINITY] {
            assert!(check_threshold(bad).is_err(), "{bad}");
        }
    }
}
