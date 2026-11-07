//! What a solve's answer rests on that the nuclear data does not give, per
//! material: products whose decay the decay library has no record for, and
//! fission whose products the library gives no yields for.
//!
//! Nothing here fills a gap. A product with no decay record is never booked
//! onto another nuclide, and a nuclide's fission products are never another
//! nuclide's. The chain makes each such product a sink (see
//! `yani::chain_arrow::close_over_products`): it neither decays nor reacts,
//! so the solve counts exactly how many atoms of it are produced. Those atoms
//! are kept out of the inventory, since nothing says they are still there,
//! and recorded here with what they bound.
//!
//! The bound: under any production history `R(s)` (atoms per second), a
//! nuclide of decay constant `l` has activity
//! `A(t) = l * integral R(s) exp(-l (t - s)) ds`, which is never more than
//! `max R(s)` over `s <= t`, whatever `l` is. So a product whose decay is
//! unknown has activity at most its largest production rate so far. Here the
//! production rate of each step is its average over the step (the atoms the
//! step produced over its length); the bound is exact for a production rate
//! constant within each step.
//!
//! The gaps are the nuclear data library's, named as such: the record is
//! what an evaluator would need to see to fill them.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use yani::{ChainNuclide, EdgeRates};

/// cm^2 per barn: atoms per barn-cm times this, times a volume in cm^3, is
/// atoms.
const BARN_PER_CM_SQ: f64 = 1e24;

/// One product the decay data has no record for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FateUnknown {
    /// Atoms per barn-cm produced by the end of each state, indexed as the
    /// material's inventories (0 is the initial composition).
    pub atoms_per_barn_cm: Vec<f64>,
    /// The average production rate over each step, in atoms per barn-cm per
    /// second, indexed as the steps.
    pub production_rate: Vec<f64>,
    /// The bound on its activity at each state, in decays per barn-cm per
    /// second: the largest step production rate up to that state (0 for the
    /// initial composition).
    pub activity_bound: Vec<f64>,
    /// What produces it in the chain: `"<parent> <reaction>"`,
    /// `"<parent> decay"` or `"<parent> fission"`, sorted.
    pub produced_by: Vec<String>,
}

/// One fissioning nuclide the library has no fission yields for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FissionWithoutYields {
    /// Its fission rate in each step, in fissions per barn-cm per second, at
    /// the step's starting inventory.
    pub fission_rate: Vec<f64>,
}

/// Everything one material's answer rests on that the data does not give.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DataGaps {
    /// The decay data the run read, as it was configured: the library whose
    /// records these products are missing from.
    pub decay_library: Option<String>,
    /// The material's volume in cm^3, to turn the per barn-cm figures into
    /// totals, when it has one.
    pub volume: Option<f64>,
    pub fate_unknown: BTreeMap<String, FateUnknown>,
    pub fission_without_yields: BTreeMap<String, FissionWithoutYields>,
    /// The share of the material's whole fission rate in each step whose
    /// products are unknown.
    pub unknown_fission_share: Vec<f64>,
}

impl DataGaps {
    /// Whether the answer rests on nothing the data lacks.
    pub fn is_empty(&self) -> bool {
        self.fate_unknown
            .values()
            .all(|f| f.atoms_per_barn_cm.iter().all(|&a| a == 0.0))
            && self.fission_without_yields.is_empty()
    }

    /// The products of unknown fate the run made any of, largest final
    /// activity bound first.
    pub fn produced(&self) -> Vec<(&str, &FateUnknown)> {
        let mut out: Vec<(&str, &FateUnknown)> = self
            .fate_unknown
            .iter()
            .filter(|(_, f)| f.atoms_per_barn_cm.iter().any(|&a| a > 0.0))
            .map(|(n, f)| (n.as_str(), f))
            .collect();
        let last = |f: &FateUnknown| f.activity_bound.last().copied().unwrap_or(0.0);
        out.sort_by(|a, b| last(b.1).total_cmp(&last(a.1)).then(a.0.cmp(b.0)));
        out
    }

    /// A figure per barn-cm as a total over the material, when it has a
    /// volume.
    pub fn total(&self, per_barn_cm: f64) -> Option<f64> {
        self.volume.map(|v| per_barn_cm * BARN_PER_CM_SQ * v)
    }
}

/// What the chain says about its gaps, worked out once per solve.
#[derive(Debug, Clone, Default)]
pub struct ChainGaps {
    /// Every nuclide the decay data has no record for, with what produces it.
    pub fate_unknown: BTreeMap<String, Vec<String>>,
    /// Every nuclide with a fission reaction and no fission yields.
    pub without_yields: BTreeSet<String>,
}

impl ChainGaps {
    pub fn of(chain: &HashMap<String, ChainNuclide>) -> ChainGaps {
        let mut gaps = ChainGaps::default();
        for (name, nuc) in chain {
            if !nuc.decay_evaluated {
                gaps.fate_unknown.entry(name.clone()).or_default();
            }
        }
        for (name, nuc) in chain {
            for rx in &nuc.reactions {
                if let Some(t) = &rx.target
                    && let Some(by) = gaps.fate_unknown.get_mut(t)
                {
                    by.push(format!("{name} {}", rx.kind));
                }
            }
            for d in &nuc.decays {
                if let Some(t) = &d.target
                    && let Some(by) = gaps.fate_unknown.get_mut(t)
                {
                    by.push(format!("{name} decay"));
                }
            }
            if let Some(set) = &nuc.fission_yields {
                let products: BTreeSet<&str> = set
                    .yields
                    .iter()
                    .flat_map(|y| y.products.iter().map(|(p, _)| p.as_str()))
                    .collect();
                for p in products {
                    if let Some(by) = gaps.fate_unknown.get_mut(p) {
                        by.push(format!("{name} fission"));
                    }
                }
            }
            let fissions = nuc.reactions.iter().any(|r| r.kind.contains("fission"));
            if fissions && nuc.fission_yields.is_none() {
                gaps.without_yields.insert(name.clone());
            }
        }
        for by in gaps.fate_unknown.values_mut() {
            by.sort();
            by.dedup();
        }
        gaps
    }
}

impl DataGaps {
    /// Take the products of unknown fate out of `densities` (atoms per
    /// barn-cm, one state's inventory) and record them as state `state`,
    /// which follows a step of length `dt` (0.0 for the initial composition).
    pub(crate) fn record_state(
        &mut self,
        chain: &ChainGaps,
        densities: &mut HashMap<String, f64>,
        dt: f64,
    ) {
        for (name, by) in &chain.fate_unknown {
            let atoms = densities.remove(name).unwrap_or(0.0);
            let entry = self
                .fate_unknown
                .entry(name.clone())
                .or_insert_with(|| FateUnknown {
                    produced_by: by.clone(),
                    ..Default::default()
                });
            if let Some(&before) = entry.atoms_per_barn_cm.last() {
                let rate = if dt > 0.0 {
                    ((atoms - before) / dt).max(0.0)
                } else {
                    0.0
                };
                entry.production_rate.push(rate);
                let bound = entry
                    .activity_bound
                    .last()
                    .copied()
                    .unwrap_or(0.0)
                    .max(rate);
                entry.activity_bound.push(bound);
            } else {
                entry.activity_bound.push(0.0);
            }
            entry.atoms_per_barn_cm.push(atoms);
        }
    }

    /// Record a step's fission with unknown products, from the edge rates it
    /// was solved with and the inventory it started from.
    pub(crate) fn record_fission(
        &mut self,
        chain: &ChainGaps,
        rates: &EdgeRates,
        start: &HashMap<String, f64>,
    ) {
        let mut total = 0.0;
        let mut unknown = 0.0;
        let mut by_parent: BTreeMap<&str, f64> = BTreeMap::new();
        for (parent, kinds) in rates {
            let n = start.get(parent).copied().unwrap_or(0.0);
            for (kind, edges) in kinds {
                if !kind.contains("fission") {
                    continue;
                }
                let rate: f64 = edges.iter().map(|(_, r)| r).sum::<f64>() * n;
                total += rate;
                if chain.without_yields.contains(parent) {
                    unknown += rate;
                    *by_parent.entry(parent.as_str()).or_default() += rate;
                }
            }
        }
        let steps = self.unknown_fission_share.len();
        for (parent, rate) in by_parent {
            let entry = self
                .fission_without_yields
                .entry(parent.to_string())
                .or_default();
            entry.fission_rate.resize(steps, 0.0);
            entry.fission_rate.push(rate);
        }
        for entry in self.fission_without_yields.values_mut() {
            entry.fission_rate.resize(steps + 1, 0.0);
        }
        self.unknown_fission_share
            .push(if total > 0.0 { unknown / total } else { 0.0 });
    }
}

/// How many names a summary line shows before it counts the rest.
const SHOWN: usize = 3;

impl crate::results::TransmutationResults {
    /// One message saying how much of this result rests on data the libraries
    /// do not give, or `None` when it rests on none: per kind of gap, how many
    /// and how large against the result, and where the detail is. Every
    /// material is covered; the largest share across them is the one quoted.
    pub fn data_gaps_summary(&self) -> Option<String> {
        let chain = self.chain.as_ref()?;
        let mut lines = Vec::new();

        // Products of unknown fate: their activity bound against the
        // material's whole activity, at the last state.
        let mut nuclides: BTreeSet<&str> = BTreeSet::new();
        let mut worst: Option<(f64, u32)> = None;
        let mut largest: Vec<(f64, &str)> = Vec::new();
        let mut library = None;
        for (id, gaps) in &self.data_gaps {
            library = library.or(gaps.decay_library.as_deref());
            let produced = gaps.produced();
            if produced.is_empty() {
                continue;
            }
            let bound: f64 = produced
                .iter()
                .map(|(_, f)| f.activity_bound.last().copied().unwrap_or(0.0))
                .sum();
            let activity = self
                .materials
                .get(id)
                .and_then(|states| states.last())
                .and_then(|m| m.get_atoms_per_barn_cm().ok())
                .map(|densities| {
                    densities
                        .iter()
                        .filter_map(|(n, d)| {
                            let t = chain.get(n)?.half_life?;
                            (t > 0.0).then(|| std::f64::consts::LN_2 / t * d)
                        })
                        .sum::<f64>()
                })
                .unwrap_or(0.0);
            let share = if activity + bound > 0.0 {
                bound / (activity + bound)
            } else {
                1.0
            };
            if worst.is_none_or(|(w, _)| share > w) {
                worst = Some((share, *id));
            }
            for (name, f) in produced {
                nuclides.insert(name);
                largest.push((f.activity_bound.last().copied().unwrap_or(0.0), name));
            }
        }
        if let Some((share, id)) = worst {
            largest.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
            largest.dedup_by(|a, b| a.1 == b.1);
            let names: Vec<&str> = largest.iter().take(SHOWN).map(|(_, n)| *n).collect();
            let more = nuclides.len().saturating_sub(names.len());
            lines.push(format!(
                "  decay data   {} has no decay record for {} nuclide{} this run produces \
                 ({}{}); their fate is unknown, so they are counted as produced and kept \
                 out of the inventory. Their activity is at most {} of the total at the \
                 end (material {id}), and their daughters and emissions are not in the \
                 results",
                library.unwrap_or("the decay data"),
                nuclides.len(),
                if nuclides.len() == 1 { "" } else { "s" },
                names.join(", "),
                if more > 0 {
                    format!(" and {more} more")
                } else {
                    String::new()
                },
                percent(share),
            ));
        }

        // Fission with no yields: the largest share of any step's fission
        // rate, over every material.
        let mut parents: BTreeSet<&str> = BTreeSet::new();
        let mut share = 0.0f64;
        for gaps in self.data_gaps.values() {
            parents.extend(gaps.fission_without_yields.keys().map(String::as_str));
            share = gaps
                .unknown_fission_share
                .iter()
                .copied()
                .fold(share, f64::max);
        }
        if share > 0.0 {
            let names: Vec<&str> = parents.iter().take(SHOWN).copied().collect();
            let more = parents.len().saturating_sub(names.len());
            lines.push(format!(
                "  fission      {} of the fission rate is in nuclides with no fission yields \
                 ({}{}); the products of that fission are not in the results",
                percent(share),
                names.join(", "),
                if more > 0 {
                    format!(" and {more} more")
                } else {
                    String::new()
                },
            ));
        }

        if lines.is_empty() {
            return None;
        }
        Some(format!(
            "this result relies on incomplete nuclear data; every gap is listed, per \
             material, in results.data_gaps\n{}",
            lines.join("\n")
        ))
    }
}

/// A share as a percentage, with as many figures as a small one needs.
fn percent(share: f64) -> String {
    let p = share * 100.0;
    if p >= 1.0 {
        format!("{p:.1}%")
    } else if p > 0.0 {
        format!("{p:.2e}%")
    } else {
        "0%".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_activity_bound_is_the_largest_production_rate_so_far() {
        let chain = ChainGaps {
            fate_unknown: BTreeMap::from([("Lu185".to_string(), vec!["Hf186 (n,np)".to_string()])]),
            ..Default::default()
        };
        let mut gaps = DataGaps::default();
        let state = |a: f64| HashMap::from([("Lu185".to_string(), a), ("W186".to_string(), 1.0)]);
        let mut s0 = state(0.0);
        gaps.record_state(&chain, &mut s0, 0.0);
        assert!(!s0.contains_key("Lu185"), "taken out of the inventory");
        gaps.record_state(&chain, &mut state(10.0), 10.0); // 1 per s
        gaps.record_state(&chain, &mut state(40.0), 10.0); // 3 per s
        gaps.record_state(&chain, &mut state(45.0), 10.0); // 0.5 per s, cooling
        let lu = &gaps.fate_unknown["Lu185"];
        assert_eq!(lu.production_rate, [1.0, 3.0, 0.5]);
        assert_eq!(lu.activity_bound, [0.0, 1.0, 3.0, 3.0]);
        assert_eq!(gaps.produced()[0].0, "Lu185");
        assert!(!gaps.is_empty());
    }

    #[test]
    fn fission_without_yields_is_a_share_of_the_whole() {
        let chain = ChainGaps {
            without_yields: BTreeSet::from(["U238".to_string()]),
            ..Default::default()
        };
        let rates: EdgeRates = HashMap::from([
            (
                "U235".to_string(),
                HashMap::from([("fission".to_string(), vec![(None, 3.0)])]),
            ),
            (
                "U238".to_string(),
                HashMap::from([("fission".to_string(), vec![(None, 1.0)])]),
            ),
        ]);
        let start = HashMap::from([("U235".to_string(), 1.0), ("U238".to_string(), 1.0)]);
        let mut gaps = DataGaps::default();
        gaps.record_fission(&chain, &rates, &start);
        assert_eq!(gaps.unknown_fission_share, [0.25]);
        assert_eq!(gaps.fission_without_yields["U238"].fission_rate, [1.0]);
    }
}
