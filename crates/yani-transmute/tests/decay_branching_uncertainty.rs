//! Decay branching ratio uncertainty, on the two-mode parents whose sum rule
//! fixes it.
//!
//! Each parent here decays for one of its half-lives into stable daughters, so
//! a daughter holds `r N0 / 2` exactly and moves one for one with its ratio.
//! Bi212 states the same sigma on both of its modes and K40 states one only
//! on its minor mode, so each daughter should spread by `sigma N0 / 2`, the
//! two in opposite directions, with their sum the same in every replica. Every
//! other kind of multi-mode parent is held at nominal and must be named in the
//! report by why.
//!
//! Needs no nuclear data: nothing is irradiated.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::d1s_uncertainty::time_correction_factor_ensemble;
use yani_transmute::uncertainty::{DataUncertainty, Info, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const N0: f64 = 1.0e-3;
const HALF_LIFE: f64 = 3600.0;
const SAMPLES: usize = 512;
const BI212_SIGMA: f64 = 0.02;
const K40_SIGMA: f64 = 0.013;

fn mode(kind: &str, target: Option<&str>, b: f64, sigma: Option<f64>) -> yani::ChainReaction {
    yani::ChainReaction {
        kind: kind.to_string(),
        target: target.map(str::to_string),
        branching: b,
        q_value: None,
        branching_uncertainty: sigma,
        evaluated_branching: None,
    }
}

fn nuclide(name: &str, decays: Vec<yani::ChainReaction>) -> yani::ChainNuclide {
    yani::ChainNuclide {
        name: name.to_string(),
        half_life: (!decays.is_empty()).then_some(HALF_LIFE),
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: [None; 3],
        reactions: Vec::new(),
        decays,
        fission_yields: None,
        sources: Vec::new(),
    }
}

/// The parents, each with a shape of MT=457 data a real library has, and
/// their stable daughters. MT=457's 0.0 for "not stated" appears as `Some(0.0)`.
fn parents() -> Vec<yani::ChainNuclide> {
    vec![
        nuclide(
            "Bi212",
            vec![
                mode("beta-", Some("Po212"), 0.6406, Some(BI212_SIGMA)),
                mode("alpha", Some("Tl208"), 0.3594, Some(BI212_SIGMA)),
            ],
        ),
        nuclide(
            "K40",
            vec![
                mode("beta-", Some("Ca40"), 0.8928, Some(0.0)),
                mode("ec/beta+", Some("Ar40"), 0.1072, Some(K40_SIGMA)),
            ],
        ),
        nuclide(
            "Cu64",
            vec![
                mode("beta-", Some("Zn64"), 0.385, Some(0.01)),
                mode("ec/beta+", Some("Ni64"), 0.615, Some(0.02)),
            ],
        ),
        nuclide(
            "Ca50",
            vec![
                mode("beta-", Some("Sc50"), 0.9965, None),
                mode("beta-,n", Some("Sc49"), 0.0035, Some(0.0035)),
            ],
        ),
        nuclide(
            "Np236",
            vec![
                mode("ec/beta+", Some("U236"), 0.87, Some(0.01)),
                mode("beta-", Some("Pu236"), 0.125, Some(0.01)),
                mode("alpha", Some("Pa232"), 0.005, None),
            ],
        ),
        // A spontaneous-fission mode is a mode: the chain reader drops its
        // target, and it still takes a share of the decays.
        nuclide(
            "Cf252",
            vec![
                mode("alpha", Some("Cm248"), 0.96908, Some(0.0)),
                mode("sf", None, 0.03092, None),
            ],
        ),
    ]
}

fn chain_with(parents: Vec<yani::ChainNuclide>) -> Arc<HashMap<String, yani::ChainNuclide>> {
    let mut chain: HashMap<String, yani::ChainNuclide> = HashMap::new();
    for p in parents {
        for d in &p.decays {
            if let Some(t) = &d.target {
                chain.insert(t.clone(), nuclide(t, Vec::new()));
            }
        }
        chain.insert(p.name.clone(), p);
    }
    Arc::new(chain)
}

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    chain_with(parents())
}

/// Every parent at `N0`, cooled for one half-life.
fn cool(
    chain: Arc<HashMap<String, yani::ChainNuclide>>,
    request: Option<&DataUncertainty>,
) -> yani_transmute::TransmutationResults {
    let names: Vec<String> = parents().into_iter().map(|p| p.name).collect();
    let mut material = Material::new(
        names.iter().map(|n| (n.clone(), 1.0)).collect(),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    for n in &names {
        material.nuclides.insert(n.clone(), N0);
    }
    material.set_temperature("294");
    // Never folded: a decay-only schedule reads no cross section.
    let spectrum = MultigroupSpectrum {
        boundaries: vec![1.0e-5, 2.0e7],
        masses: vec![1.0],
        flux_error: None,
    };
    let steps = [TransmuteStep {
        dt: HALF_LIFE,
        irradiation: None,
    }];
    transmute_material(
        &mut material,
        &[spectrum],
        &steps,
        chain,
        &Default::default(),
        Default::default(),
        request,
    )
    .expect("transmute")
}

fn request(sources: Vec<Source>, attribution: bool) -> DataUncertainty {
    DataUncertainty {
        seed: 5,
        samples: Some(SAMPLES),
        sources,
        attribution,
        ..Default::default()
    }
}

fn info_of(results: &yani_transmute::TransmutationResults) -> &Info {
    results.uncertainty_info.get(&0).expect("uncertainty asked")
}

fn replicas(results: &yani_transmute::TransmutationResults, nuclide: &str) -> Vec<f64> {
    results
        .uncertainty_inventories(0, 1)
        .expect("uncertainty asked")
        .iter()
        .map(|inv| inv.get(nuclide).copied().unwrap_or(0.0))
        .collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn std_dev(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let cov = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (a.len() - 1) as f64;
    cov / (std_dev(a) * std_dev(b))
}

/// Both of a pair's daughters spread by the stated sigma, in opposite
/// directions, and their sum is the same in every replica.
fn assert_pair_realises(
    results: &yani_transmute::TransmutationResults,
    first: &str,
    second: &str,
    sigma: f64,
) {
    let a = replicas(results, first);
    let b = replicas(results, second);
    assert_eq!(a.len(), SAMPLES);
    let want = sigma * N0 / 2.0;
    // Sampling error on a sigma from 512 replicas is about 3%.
    for (name, x) in [(first, &a), (second, &b)] {
        let got = std_dev(x);
        assert!(
            (got / want - 1.0).abs() < 0.12,
            "{name} spreads by {got:e}, the stated sigma gives {want:e}"
        );
    }
    let r = correlation(&a, &b);
    assert!(r < -1.0 + 1e-9, "{first} and {second} correlate at {r}");
    let nominal = results.get_material(0, 1).unwrap().nuclides[first]
        + results.get_material(0, 1).unwrap().nuclides[second];
    for (x, y) in a.iter().zip(&b) {
        assert!(
            ((x + y) / nominal - 1.0).abs() < 1e-12,
            "the pair's total moved: {} against {nominal}",
            x + y
        );
    }
}

#[test]
fn equal_sigmas_are_realised_on_both_modes() {
    let results = cool(chain(), Some(&request(vec![Source::DecayBranching], false)));
    assert_pair_realises(&results, "Po212", "Tl208", BI212_SIGMA);
}

#[test]
fn a_sigma_on_one_mode_moves_its_complement_too() {
    let results = cool(chain(), Some(&request(vec![Source::DecayBranching], false)));
    assert_pair_realises(&results, "Ar40", "Ca40", K40_SIGMA);
}

#[test]
fn every_parent_held_at_nominal_is_named_by_why() {
    let results = cool(chain(), Some(&request(vec![Source::DecayBranching], false)));
    let info = info_of(&results);
    let names = |xs: &[&str]| -> std::collections::BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    };
    assert_eq!(info.decay_branchings_perturbed, names(&["Bi212", "K40"]));
    assert_eq!(info.decay_branchings_unequal_sigmas, names(&["Cu64"]));
    assert_eq!(info.decay_branchings_too_wide, names(&["Ca50"]));
    assert_eq!(info.decay_branchings_three_or_more_modes, names(&["Np236"]));
    assert_eq!(info.no_decay_branching_uncertainty, names(&["Cf252"]));
    assert!(info.has_gaps());
    assert_eq!(info.decay_branchings_sampled, 2 * SAMPLES);
    assert_eq!(info.decay_branchings_floored, 0);
    assert!(info.sources.contains(&"decay_branching".to_string()));
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s == "decay branching ratio"));
    assert!(
        info.not_perturbed
            .iter()
            .any(|s| s.starts_with("decay emission per branch")),
        "the nominal per-branch emission must be stated: {:?}",
        info.not_perturbed
    );

    // What is held at nominal does not move at all.
    for (daughter, parent) in [
        ("Zn64", "Cu64"),
        ("Sc49", "Ca50"),
        ("U236", "Np236"),
        ("Cm248", "Cf252"),
    ] {
        let x = replicas(&results, daughter);
        assert!(
            x.iter().all(|v| v.to_bits() == x[0].to_bits()),
            "{parent}'s split moved"
        );
    }
}

#[test]
fn asking_leaves_the_nominal_inventory_bit_identical() {
    let plain = cool(chain(), None);
    let asked = cool(chain(), Some(&request(vec![Source::DecayBranching], false)));
    let a = &plain.get_material(0, 1).unwrap().nuclides;
    let b = &asked.get_material(0, 1).unwrap().nuclides;
    assert_eq!(a.len(), b.len());
    for (name, x) in a {
        assert_eq!(x.to_bits(), b[name].to_bits(), "{name}");
    }
}

/// With the source off the stated branching sigmas are not read: the run is
/// the one a chain with none of them gives with the source on, replica for
/// replica.
#[test]
fn switched_off_nothing_is_read() {
    let with_sigma = |mut chain: HashMap<String, yani::ChainNuclide>| {
        let bi = chain.get_mut("Bi212").unwrap();
        bi.half_life_uncertainty = Some(0.05 * HALF_LIFE);
        Arc::new(chain)
    };
    let stated = with_sigma((*chain()).clone());
    let mut bare = parents();
    for p in &mut bare {
        for d in &mut p.decays {
            d.branching_uncertainty = None;
        }
    }
    let unstated = with_sigma((*chain_with(bare)).clone());

    let off = cool(stated, Some(&request(vec![Source::HalfLife], false)));
    let on = cool(
        unstated,
        Some(&request(
            vec![Source::HalfLife, Source::DecayBranching],
            false,
        )),
    );
    let info = info_of(&off);
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "decay branching ratio"));
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("decay emission per branch")));
    assert!(info.decay_branchings_perturbed.is_empty());
    assert!(info.no_decay_branching_uncertainty.is_empty());
    assert_eq!(info.decay_branchings_sampled, 0);

    let a = off.uncertainty_inventories(0, 1).unwrap();
    let b = on.uncertainty_inventories(0, 1).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.len(), y.len());
        for (name, v) in x.iter() {
            assert_eq!(v.to_bits(), y[name].to_bits(), "{name}");
        }
    }
}

/// A daughter fed by one branch alone is linear in the ratio, so the first
/// order contribution is the whole variance: `(sigma N0 / 2)^2`.
#[test]
fn the_contributor_is_the_parent_with_its_whole_variance() {
    let results = cool(chain(), Some(&request(vec![Source::DecayBranching], true)));
    let attribution = results.uncertainty[&0]
        .attribution
        .as_ref()
        .expect("attribution asked");
    let bi = attribution
        .contributors
        .iter()
        .find(|c| c.source == "decay_branching" && c.nuclide == "Bi212")
        .expect("Bi212 contributes");
    assert_eq!(bi.reaction, None);
    let want = (BI212_SIGMA * N0 / 2.0).powi(2);
    for daughter in ["Po212", "Tl208"] {
        let got = bi.variance[0][daughter];
        assert!(
            (got / want - 1.0).abs() < 1e-6,
            "{daughter}: {got:e} against {want:e}"
        );
    }
    assert!(
        !attribution
            .contributors
            .iter()
            .any(|c| c.source == "decay_branching" && c.nuclide == "Cu64"),
        "a parent held at nominal contributes nothing"
    );
}

/// D1S draws no branching, and says so.
#[test]
fn a_d1s_time_correction_lists_the_branching_as_not_perturbed() {
    let emitters = vec!["Bi212".to_string()];
    let ensemble = time_correction_factor_ensemble(
        &emitters,
        &[HALF_LIFE],
        &[vec![1.0e10]],
        &chain(),
        &request(Source::IMPLEMENTED.to_vec(), false),
    )
    .expect("tcf");
    assert_eq!(
        ensemble.not_perturbed,
        vec!["decay branching ratio".to_string()]
    );
}

/// A D1S request without the half-life source says the half-lives were held
/// at nominal too, as a transmutation report does.
#[test]
fn a_d1s_time_correction_without_half_lives_lists_them_as_not_perturbed() {
    let emitters = vec!["Bi212".to_string()];
    let ensemble = time_correction_factor_ensemble(
        &emitters,
        &[HALF_LIFE],
        &[vec![1.0e10]],
        &chain(),
        &request(vec![Source::DecayBranching], false),
    )
    .expect("tcf");
    assert!(ensemble.sources.is_empty());
    assert_eq!(
        ensemble.not_perturbed,
        vec!["half-life".to_string(), "decay branching ratio".to_string()]
    );
}
