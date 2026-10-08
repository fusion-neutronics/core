//! Fission yield uncertainty, from the DY the evaluation states on each
//! independent yield.
//!
//! The solver does not read the tape's yields. It reads `products`, which the
//! converter derived from them: a product with no decay data is replaced by a
//! stand-in (`endf::chain::replace_missing`) and products landing on the same
//! name are summed. A summed yield has no DY of its own, since that would need
//! the correlation of its parts, so the draw is made on the tape's own
//! products, one lognormal per stated DY, and the draws are summed onto the
//! solver's products the way the converter summed the nominal yields.
//!
//! That needs the mapping the converter used, which the chain does not store.
//! It is recomputed with the converter's own rule
//! (`endf::chain::replace_missing_in`), walking each missing product towards
//! stability to the first of the solver's products it reaches, and checked:
//! the nominal tape yields summed through it must give `products` back. A
//! parent where they do not is held at nominal and named, rather than drawn
//! through a mapping that is not the one its yields were built with.
//!
//! No evaluation publishes a correlation between yields (ENDF/B-VIII.1,
//! JEFF-4.0 and JENDL-5.0 carry only MF=8 MT=454/459), so every product is
//! drawn independently, and nothing is renormalised: the yields of a draw do
//! not sum to the nominal total, as they need not under the data's own
//! statement. Cumulative yields (MT=459) are not solved with and are not
//! drawn.
//!
//! An alias, an actinide that borrows another's yields, shares that
//! evaluation's draw, since it is the same evaluation. The stream is keyed on
//! the yields themselves, so the same evaluation draws the same numbers
//! whichever of its users a material reaches.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use yani::{ChainNuclide, FissionYield, FissionYieldSet};

use crate::covariance_sample::{lognormal_multiplier, standard_normals};

/// Keeps the fission-yield streams clear of every other stream.
pub(crate) const FISSION_YIELD_STREAM: u32 = 0xF155_10A1;

/// Two sums closer than this, relatively, are the same yield: the converter
/// summed in tape order, as this does, so they agree to the last bit unless
/// the mapping differs.
const SAME_YIELD: f64 = 1.0e-9;

/// One tape product with a DY a draw can carry, and where it lands.
#[derive(Debug, Clone, PartialEq)]
struct Term {
    /// Its position in the evaluation's deviates.
    deviate: usize,
    /// The index in `products` of the solver's product it was summed onto.
    product: usize,
    /// Its nominal yield.
    yield_: f64,
    /// Its relative sigma, DY / Y.
    relative: f64,
}

/// One evaluation's yields, ready to draw.
#[derive(Debug, Clone)]
struct Evaluation {
    /// The stream key, a hash of the yields.
    key: u32,
    /// The nominal yields every draw starts from.
    nominal: Arc<FissionYieldSet>,
    /// Per tabulated energy, the terms a draw moves.
    terms: Vec<Vec<Term>>,
    /// One per tape product at every energy with independent yields, sigma
    /// or not, so a product draws the same deviate whatever its neighbours
    /// state.
    deviates: usize,
    /// The parents that read these yields.
    parents: Vec<String>,
}

/// The reachable fissioning parents split by what can be drawn for them.
#[derive(Debug, Clone, Default)]
pub(crate) struct Candidates {
    evaluations: Vec<Evaluation>,
    /// Parents whose yields carry at least one DY a draw can carry.
    pub(crate) perturbed: BTreeSet<String>,
    /// Parents whose yields state no DY, or carry no evaluated yields at all
    /// (a chain without `evaluated_yields.arrow`), held at nominal.
    pub(crate) without: BTreeSet<String>,
    /// Parents with a DY no draw can carry (on a zero yield, or not finite);
    /// that yield is held, any other is still drawn.
    pub(crate) not_carried: BTreeSet<String>,
    /// Parents whose tape yields, summed through the converter's rule, do not
    /// give the solver's products back, held at nominal.
    pub(crate) mapping_mismatch: BTreeSet<String>,
}

impl Candidates {
    pub(crate) fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }
}

/// A stable 32-bit hash of an evaluation's independent yields: FNV-1a over
/// each energy and its tape products and yields, so an evaluation keys the
/// same stream in every chain that carries it.
fn evaluation_key(set: &FissionYieldSet) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    };
    for point in &set.yields {
        eat(&point.energy.to_bits().to_le_bytes());
        if let Some(ev) = &point.independent {
            for (name, y) in ev.products.iter().zip(&ev.yields) {
                eat(name.as_bytes());
                eat(&y.to_bits().to_le_bytes());
            }
        }
    }
    h
}

/// The isotopes of each element in `library`, as `(name, stable, half-life)`
/// in ascending name order, which is what `replace_missing_in` searches for
/// the direction of stability. A chain stores no half-life for a stable
/// nuclide, which is where the converter left it out.
struct Isotopes<'a> {
    by_symbol: HashMap<&'a str, Vec<(&'a str, bool, f64)>>,
}

impl<'a> Isotopes<'a> {
    fn new(library: &'a HashMap<String, ChainNuclide>) -> Self {
        let mut by_symbol: HashMap<&str, Vec<(&str, bool, f64)>> = HashMap::new();
        for (name, cn) in library {
            by_symbol.entry(symbol(name)).or_default().push((
                name.as_str(),
                cn.half_life.is_none(),
                cn.half_life.unwrap_or(0.0),
            ));
        }
        for isotopes in by_symbol.values_mut() {
            isotopes.sort_by(|a, b| a.0.cmp(b.0));
        }
        Self { by_symbol }
    }

    /// The name the converter gave a tape `product` among the solver's
    /// `listed` products: itself where it is listed, otherwise the first
    /// listed nuclide on its walk towards stability. `None` for a product
    /// with no stand-in (a bare neutron), which the converter drops.
    ///
    /// Stopping at the first LISTED nuclide, rather than the first one the
    /// loaded decay library knows, is what lets this work when the yields were
    /// converted against another decay library than the one loaded: the
    /// converter's stand-in received a yield, so it is listed, and every
    /// nuclide its walk passed before it was one the converter's library
    /// lacked, so none of them is. With the same library on both sides the
    /// two rules agree. Only the direction of the walk is read from the loaded
    /// library, and a parent where that differs fails [`maps_back`].
    fn name(&self, product: &str, listed: &std::collections::HashSet<&str>) -> Option<String> {
        if listed.contains(product) {
            return Some(product.to_string());
        }
        let isotopes = self
            .by_symbol
            .get(symbol(product))
            .map_or(&[][..], Vec::as_slice);
        endf::chain::replace_missing_in(
            product,
            |name| listed.contains(name),
            isotopes.iter().copied(),
        )
    }
}

/// The element symbol a nuclide name starts with.
fn symbol(name: &str) -> &str {
    let end = name
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(name.len());
    &name[..end]
}

/// What one parent's yields allow: the evaluation to draw, or why not.
enum Plan {
    Draw {
        terms: Vec<Vec<Term>>,
        deviates: usize,
        not_carried: bool,
    },
    Without,
    Mismatch,
}

fn plan(set: &FissionYieldSet, isotopes: &Isotopes) -> Plan {
    let mut terms = Vec::with_capacity(set.yields.len());
    let mut deviates = 0usize;
    let mut any_evaluated = false;
    let mut not_carried = false;
    for point in &set.yields {
        let Some(ev) = &point.independent else {
            terms.push(Vec::new());
            continue;
        };
        any_evaluated = true;
        let listed: std::collections::HashSet<&str> = point
            .products
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let names: Vec<Option<String>> = ev
            .products
            .iter()
            .map(|product| isotopes.name(product, &listed))
            .collect();
        if !maps_back(point, ev, &names) {
            return Plan::Mismatch;
        }
        let index: HashMap<&str, usize> = point
            .products
            .iter()
            .enumerate()
            .map(|(j, (name, _))| (name.as_str(), j))
            .collect();
        let mut here = Vec::new();
        for (i, (name, y)) in names.iter().zip(&ev.yields).enumerate() {
            let sigma = ev.uncertainties.get(i).copied().flatten();
            let Some(name) = name else {
                continue;
            };
            match crate::uncertainty::carried(*y, sigma) {
                Some(sigma) => here.push(Term {
                    deviate: deviates + i,
                    product: index[name.as_str()],
                    yield_: *y,
                    relative: sigma / y,
                }),
                None if crate::uncertainty::stated(sigma) => not_carried = true,
                None => {}
            }
        }
        deviates += ev.products.len();
        terms.push(here);
    }
    if !any_evaluated || terms.iter().all(Vec::is_empty) {
        return Plan::Without;
    }
    Plan::Draw {
        terms,
        deviates,
        not_carried,
    }
}

/// Whether the tape's yields, summed onto `names` in tape order as the
/// converter summed them, give this energy's `products` back: every product
/// they land on at the value listed, and every listed product with a yield
/// accounted for.
fn maps_back(point: &FissionYield, ev: &yani::EvaluatedYields, names: &[Option<String>]) -> bool {
    let mut sums: HashMap<&str, f64> = HashMap::new();
    for (name, y) in names.iter().zip(&ev.yields) {
        if let Some(name) = name {
            *sums.entry(name.as_str()).or_insert(0.0) += y;
        }
    }
    let listed: HashMap<&str, f64> = point
        .products
        .iter()
        .map(|(name, y)| (name.as_str(), *y))
        .collect();
    let same = |a: f64, b: f64| (a - b).abs() <= SAME_YIELD * a.abs().max(b.abs());
    sums.iter()
        .all(|(name, sum)| listed.get(name).is_some_and(|y| same(*sum, *y)))
        && listed
            .iter()
            .all(|(name, y)| *y == 0.0 || sums.contains_key(name))
}

/// Sort the fissioning parents of `reachable` by what can be drawn for them,
/// naming products with the converter's rule over `library`, the whole chain
/// the run loaded: the rule searches every isotope of an element, so a pruned
/// chain could send a product another way.
pub(crate) fn candidates(
    reachable: &HashMap<String, ChainNuclide>,
    library: &HashMap<String, ChainNuclide>,
) -> Candidates {
    /// What one set allows, decided once and shared by every parent reading
    /// it: an alias shares its owner's `Arc`, and so its verdict.
    #[derive(Clone, Copy)]
    enum Verdict {
        Draw {
            evaluation: usize,
            not_carried: bool,
        },
        Without,
        Mismatch,
    }

    let isotopes = Isotopes::new(library);
    let mut out = Candidates::default();
    let mut parents: Vec<(&String, &Arc<FissionYieldSet>)> = reachable
        .iter()
        .filter_map(|(name, cn)| Some((name, cn.fission_yields.as_ref()?)))
        .collect();
    parents.sort_by(|a, b| a.0.cmp(b.0));
    let mut verdicts: HashMap<*const FissionYieldSet, Verdict> = HashMap::new();
    for (parent, set) in parents {
        let verdict =
            *verdicts
                .entry(Arc::as_ptr(set))
                .or_insert_with(|| match plan(set, &isotopes) {
                    Plan::Draw {
                        terms,
                        deviates,
                        not_carried,
                    } => {
                        out.evaluations.push(Evaluation {
                            key: evaluation_key(set),
                            nominal: Arc::clone(set),
                            terms,
                            deviates,
                            parents: Vec::new(),
                        });
                        Verdict::Draw {
                            evaluation: out.evaluations.len() - 1,
                            not_carried,
                        }
                    }
                    Plan::Without => Verdict::Without,
                    Plan::Mismatch => Verdict::Mismatch,
                });
        match verdict {
            Verdict::Draw {
                evaluation,
                not_carried,
            } => {
                out.evaluations[evaluation].parents.push(parent.clone());
                out.perturbed.insert(parent.clone());
                if not_carried {
                    out.not_carried.insert(parent.clone());
                }
            }
            Verdict::Without => {
                out.without.insert(parent.clone());
            }
            Verdict::Mismatch => {
                out.mapping_mismatch.insert(parent.clone());
            }
        }
    }
    out
}

/// One replica's fission yields: for every parent drawn, the set it is solved
/// with, shared between the parents of one evaluation.
///
/// Each drawn set keeps the nominal energies, so the spectrum weights folded
/// for the nominal yields still apply, and carries `products` only: the
/// evaluated yields it was drawn from are not read by a solve, and copying
/// them per replica would cost more than the draw.
pub(crate) fn sample(
    candidates: &Candidates,
    base_seed: u64,
    replica: u64,
) -> HashMap<String, Arc<FissionYieldSet>> {
    let replica_seed = yamc_rng::history_seed(base_seed, replica);
    let mut out = HashMap::new();
    for e in &candidates.evaluations {
        let seed = yamc_rng::secondary_seed(replica_seed, e.key ^ FISSION_YIELD_STREAM);
        let z = standard_normals(&mut yamc_rng::expand_seed(seed), e.deviates);
        let yields = e
            .nominal
            .yields
            .iter()
            .zip(&e.terms)
            .map(|(point, terms)| {
                let mut products = point.products.clone();
                for t in terms {
                    products[t.product].1 +=
                        t.yield_ * (lognormal_multiplier(z[t.deviate], t.relative) - 1.0);
                }
                FissionYield {
                    energy: point.energy,
                    products,
                    independent: None,
                    cumulative: None,
                }
            })
            .collect();
        let drawn = Arc::new(FissionYieldSet::new(yields));
        for parent in &e.parents {
            out.insert(parent.clone(), Arc::clone(&drawn));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nuclide(name: &str, half_life: Option<f64>) -> ChainNuclide {
        ChainNuclide {
            name: name.to_string(),
            half_life,
            half_life_uncertainty: None,
            decay_energy: 0.0,
            decay_energy_uncertainty: None,
            decay_energy_components: Default::default(),
            reactions: Vec::new(),
            decays: Vec::new(),
            fission_yields: None,
            sources: Vec::new(),
        }
    }

    /// One energy of U235 yields from the tape: Kr85, Kr85_m1 (which the
    /// library lacks, so it lands on Kr85) and Xe135, with `dy` the relative
    /// DY on each. `products` is what the converter derives from them.
    fn point(dy: [f64; 3], kr85: f64) -> FissionYield {
        let tape = [("Kr85", 0.002), ("Kr85_m1", 0.011), ("Xe135", 0.003)];
        FissionYield {
            energy: 0.0253,
            products: vec![("Kr85".to_string(), kr85), ("Xe135".to_string(), 0.003)],
            independent: Some(yani::EvaluatedYields {
                products: tape.iter().map(|(n, _)| n.to_string()).collect(),
                yields: tape.iter().map(|(_, y)| *y).collect(),
                uncertainties: tape.iter().zip(dy).map(|((_, y), d)| Some(y * d)).collect(),
                interpolation: None,
            }),
            cumulative: None,
        }
    }

    fn library(set: FissionYieldSet) -> HashMap<String, ChainNuclide> {
        let mut u235 = nuclide("U235", Some(2.2e16));
        u235.fission_yields = Some(Arc::new(set));
        HashMap::from([
            ("U235".to_string(), u235),
            ("Kr85".to_string(), nuclide("Kr85", Some(3.4e8))),
            ("Rb85".to_string(), nuclide("Rb85", None)),
            ("Xe135".to_string(), nuclide("Xe135", Some(3.3e4))),
            ("Cs135".to_string(), nuclide("Cs135", Some(7.3e13))),
        ])
    }

    fn kr85(set: &FissionYieldSet) -> f64 {
        set.yields[0].products[0].1
    }

    /// The tape's products are drawn and summed onto the solver's: Kr85 moves
    /// as its two tape parents in quadrature, centred on the nominal sum.
    #[test]
    fn draws_land_on_the_products_the_converter_summed_them_onto() {
        let chain = library(FissionYieldSet::new(vec![point([0.1, 0.2, 0.05], 0.013)]));
        let c = candidates(&chain, &chain);
        assert_eq!(c.perturbed, BTreeSet::from(["U235".to_string()]));
        assert!(c.mapping_mismatch.is_empty() && c.without.is_empty());

        let n = 20_000;
        let draws: Vec<f64> = (0..n).map(|r| kr85(&sample(&c, 4, r)["U235"])).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let sd = (draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        let want = ((0.002_f64 * 0.1).powi(2) + (0.011_f64 * 0.2).powi(2)).sqrt();
        assert!((mean / 0.013 - 1.0).abs() < 0.003, "mean {mean}");
        assert!((sd / want - 1.0).abs() < 0.03, "sigma {sd} against {want}");

        let drawn = sample(&c, 4, 0);
        let set = &drawn["U235"];
        assert_eq!(set.yields[0].energy, 0.0253);
        assert!(set.yields[0].independent.is_none());
        assert_ne!(set.yields[0].products[1].1, 0.003, "Xe135 moved too");
        // The nominal set is untouched.
        assert_eq!(kr85(chain["U235"].fission_yields.as_ref().unwrap()), 0.013);
    }

    /// Products that do not sum back to the solver's are a mapping this rule
    /// did not make, and the parent is held, never drawn through it.
    #[test]
    fn a_mapping_that_does_not_give_the_products_back_is_held() {
        let chain = library(FissionYieldSet::new(vec![point([0.1, 0.2, 0.05], 0.012)]));
        let c = candidates(&chain, &chain);
        assert!(c.is_empty());
        assert_eq!(c.mapping_mismatch, BTreeSet::from(["U235".to_string()]));
    }

    /// No DY, or no evaluated yields at all, is "no uncertainty", reported.
    #[test]
    fn yields_with_no_stated_dy_are_listed_without() {
        let chain = library(FissionYieldSet::new(vec![point([0.0, 0.0, 0.0], 0.013)]));
        let c = candidates(&chain, &chain);
        assert!(c.is_empty());
        assert_eq!(c.without, BTreeSet::from(["U235".to_string()]));

        let mut bare = point([0.1, 0.1, 0.1], 0.013);
        bare.independent = None;
        let chain = library(FissionYieldSet::new(vec![bare]));
        assert_eq!(
            candidates(&chain, &chain).without,
            BTreeSet::from(["U235".to_string()])
        );
    }

    /// A DY on a zero yield cannot be drawn: that yield is held and the parent
    /// named, while its other yields are still drawn.
    #[test]
    fn a_dy_on_a_zero_yield_is_named_and_the_rest_still_drawn() {
        let mut p = point([0.1, 0.2, 0.05], 0.013);
        let ev = p.independent.as_mut().unwrap();
        ev.products.push("Cs135".to_string());
        ev.yields.push(0.0);
        ev.uncertainties.push(Some(1.0e-5));
        p.products.push(("Cs135".to_string(), 0.0));
        let chain = library(FissionYieldSet::new(vec![p]));
        let c = candidates(&chain, &chain);
        assert_eq!(c.perturbed, BTreeSet::from(["U235".to_string()]));
        assert_eq!(c.not_carried, BTreeSet::from(["U235".to_string()]));
        assert_eq!(sample(&c, 1, 0)["U235"].yields[0].products[2].1, 0.0);
    }

    /// An actinide borrowing another's yields shares the evaluation and so
    /// its draw, in every replica.
    #[test]
    fn an_alias_shares_its_owners_draw() {
        let mut chain = library(FissionYieldSet::new(vec![point([0.1, 0.2, 0.05], 0.013)]));
        let mut u238 = nuclide("U238", Some(1.4e17));
        u238.fission_yields = chain["U235"].fission_yields.clone();
        chain.insert("U238".to_string(), u238);
        let c = candidates(&chain, &chain);
        assert_eq!(
            c.perturbed,
            BTreeSet::from(["U235".to_string(), "U238".to_string()])
        );
        for replica in 0..8 {
            let drawn = sample(&c, 2, replica);
            assert!(Arc::ptr_eq(&drawn["U235"], &drawn["U238"]));
        }
    }

    /// A product draws the same deviate whatever its neighbours state.
    #[test]
    fn a_yield_draws_the_same_deviate_whatever_its_neighbours_state() {
        let a = library(FissionYieldSet::new(vec![point([0.0, 0.0, 0.05], 0.013)]));
        let b = library(FissionYieldSet::new(vec![point([0.1, 0.2, 0.05], 0.013)]));
        // Different yields would key a different stream; the DY is not part of
        // the key, so these two share one.
        let (ca, cb) = (candidates(&a, &a), candidates(&b, &b));
        for replica in 0..8 {
            let xe = |c: &Candidates| sample(c, 3, replica)["U235"].yields[0].products[1].1;
            assert_eq!(xe(&ca), xe(&cb));
        }
    }

    /// Yields converted against another decay library than the one loaded
    /// still map: here the loaded library has Kr85_m1, which the converter's
    /// did not, so the converter summed it onto Kr85, and the walk stops at
    /// the first product the converter listed rather than at the first
    /// nuclide the loaded library knows.
    #[test]
    fn yields_converted_against_another_decay_library_still_map() {
        let mut chain = library(FissionYieldSet::new(vec![point([0.1, 0.2, 0.05], 0.013)]));
        chain.insert("Kr85_m1".to_string(), nuclide("Kr85_m1", Some(1.6e4)));
        chain.remove("Xe135");
        let c = candidates(&chain, &chain);
        assert!(c.mapping_mismatch.is_empty(), "{c:?}");
        assert_eq!(c.perturbed, BTreeSet::from(["U235".to_string()]));
    }
}
