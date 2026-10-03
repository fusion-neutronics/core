//! A branching overlay that lists only a reaction's isomer, the ground state
//! being the remainder, end to end through `transmute_material`.
//!
//! ENDF/B-VIII.1 gives In115 (n,gamma) as one MF=9 yield to In116_m1 (a flat
//! 0.79) and In115 (n,2n) as one MF=10 partial to In114_m1. The base chain
//! carries each ground state at 1.0 and the isomer is grafted at 0.0. The fold
//! used to normalize over the listed states and then re-partition only the
//! mass they carried, which for a grafted isomer is none, so In115 capture
//! never made In116_m1, the standard activation-foil product.
//!
//! The synthetic case always runs and checks the split exactly, in what
//! `get_isomeric_branching` reports and in the inventory. The ENDF/B-VIII.1
//! cases read the In115 and Mo92 fixtures and the chain fixture, which
//! `scripts/fetch_test_fixtures.py` fetches, and self-skip on a machine that
//! has not run it. Their branching rows are
//! `tests/fixtures/endf-b8.1-branching-In115-Mo92`, the In115 and Mo92 rows of
//! ENDF/B-VIII.1's `branching.arrow` as the converter writes it now, with the
//! facts saying each list gives its isomer alone: the published subsection
//! predates those facts, and the rule refuses a list without them.

use std::collections::HashMap;
use std::sync::Arc;

use yamc_materials::Material;
use yamc_nuclide::nuclide::Nuclide;
use yamc_nuclide::reaction::Reaction;
use yani::{BranchCurve, BranchQuantity, BranchState, BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

fn reaction(mt: i32, threshold_idx: usize, energy: Vec<f64>, cross_section: Vec<f64>) -> Reaction {
    Reaction {
        cross_section: cross_section.into(),
        threshold_idx,
        energy: energy.into(),
        mt_number: mt,
        q_value: 0.0,
        products: vec![],
        scatter_in_cm: false,
        redundant: false,
    }
}

/// The (n,2n) cross section: zero at a 10 MeV threshold, 2 b at 20 MeV.
fn n2n() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0e7, 1.5e7, 2.0e7], vec![0.0, 1.6, 2.0])
}

/// In115 with a 1/v-like capture cross section and a threshold (n,2n), at
/// 294 K, and nothing else.
fn indium() -> Material {
    let temperature = "294".to_string();
    let capture_energy: Vec<f64> = (0..=24)
        .map(|i| 10f64.powf(-5.0 + 12.0 * i as f64 / 24.0))
        .collect();
    let capture_xs: Vec<f64> = capture_energy.iter().map(|e| 30.0 / e.sqrt()).collect();
    let (e16, xs16) = n2n();
    let reactions: HashMap<i32, Arc<Reaction>> = HashMap::from([
        (102, Arc::new(reaction(102, 0, capture_energy, capture_xs))),
        (16, Arc::new(reaction(16, 1, e16, xs16))),
    ]);
    let nuclide = Nuclide {
        name: Some("In115".to_string()),
        element: None,
        atomic_symbol: Some("In".to_string()),
        atomic_number: Some(49),
        neutron_number: Some(66),
        mass_number: Some(115),
        atomic_weight_ratio: Some(113.9),
        library: None,
        energy: None,
        reactions: vec![reactions],
        fissionable: false,
        available_temperatures: vec![temperature.clone()],
        loaded_temperatures: vec![temperature.clone()],
        data_path: None,
        data_source: None,
        fission_nu: None,
        fast_xs: vec![],
        urr_data: vec![],
        urr_present: false,
        fission_photon_release: None,
        covariance: None,
        elastic_flat_cache: Default::default(),
        fission_chi_flat_cache: Default::default(),
        delayed_neutron_cache: Default::default(),
        inelastic_angle_flat_cache: Default::default(),
        load_scope: Default::default(),
    };
    let mut m = Material::new(
        HashMap::from([("In115".to_string(), 3.8e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("indium");
    m.set_temperature(&temperature);
    m.nuclide_data
        .insert("In115".to_string(), Arc::new(nuclide));
    m
}

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: vec![],
        fission_yields: None,
        sources: Vec::new(),
    }
}

/// The chain as the ENDF/B-VIII.1 overlay leaves it: grounds at 1.0 from the
/// base, isomers grafted at 0.0. The products are stable so the inventory
/// holds the split exactly.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let edge = |kind: &str, target: &str, branching: f64| ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching,
        branching_uncertainty: None,
        evaluated_branching: None,
        q_value: Some(0.0),
    };
    let mut map = HashMap::new();
    map.insert(
        "In115".to_string(),
        nuclide(
            "In115",
            vec![
                edge("(n,gamma)", "In116", 1.0),
                edge("(n,gamma)", "In116_m1", 0.0),
                edge("(n,2n)", "In114", 1.0),
                edge("(n,2n)", "In114_m1", 0.0),
            ],
        ),
    );
    for name in ["In116", "In116_m1", "In114", "In114_m1"] {
        map.insert(name.to_string(), nuclide(name, vec![]));
    }
    Arc::new(map)
}

/// The converter's facts for an isomer its list gives alone.
fn isomer_only(mt: i32) -> Arc<[BranchState]> {
    Arc::from(vec![BranchState {
        mt,
        lfs: 1,
        lmf: Some(10),
        list_complete: false,
        level_route: "energy".to_string(),
        level_energy: 0.0,
        level_energy_difference: Some(0.0),
        mf3_cross_section: None,
    }])
}

/// Only the isomers, as ENDF/B-VIII.1 lists them: a flat 0.79 capture yield,
/// and an (n,2n) partial at 0.9 of the cross section on its own grid.
fn branch() -> BranchTable {
    let (e16, xs16) = n2n();
    let mut branch = BranchTable::new();
    let kinds = branch.curves_mut().entry("In115".to_string()).or_default();
    kinds.insert(
        "(n,gamma)".to_string(),
        vec![BranchCurve {
            target: "In116_m1".to_string(),
            quantity: BranchQuantity::Yield,
            energy: vec![1.0e-5, 2.0e7],
            values: vec![0.79, 0.79],
            states: isomer_only(102),
            normalisation: None,
        }],
    );
    kinds.insert(
        "(n,2n)".to_string(),
        vec![BranchCurve {
            target: "In114_m1".to_string(),
            quantity: BranchQuantity::CrossSection,
            energy: e16,
            values: xs16.iter().map(|x| 0.9 * x).collect(),
            states: isomer_only(16),
            normalisation: None,
        }],
    );
    branch
}

/// Thermal and 14 MeV flux in one spectrum, so both reactions fire.
fn spectrum() -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![1.0e-5, 0.1, 1.0, 1.0e6, 1.35e7, 1.45e7, 2.0e7],
        masses: vec![0.3, 0.2, 0.1, 0.0, 0.4, 0.0],
        flux_error: None,
    }
}

fn split_of(split: &[(String, f64)], target: &str) -> f64 {
    split
        .iter()
        .find(|(t, _)| t == target)
        .unwrap_or_else(|| panic!("{target} missing from {split:?}"))
        .1
}

#[test]
fn an_isomer_only_list_splits_off_the_ground_state() {
    let mut material = indium();
    let steps = [TransmuteStep {
        dt: 3600.0,
        irradiation: Some((0, 1.0e12)),
    }];
    let results = transmute_material(
        &mut material,
        &[spectrum()],
        &steps,
        chain(),
        &branch(),
        Default::default(),
        None,
    )
    .expect("transmute");

    let channels = results.get_isomeric_branching(0, 0).expect("step 0");
    let channel = |reaction: &str| {
        channels
            .iter()
            .find(|c| c.parent == "In115" && c.reaction == reaction)
            .unwrap_or_else(|| panic!("In115 {reaction} not reported: {channels:?}"))
    };
    let capture = channel("(n,gamma)");
    assert_eq!(capture.split[0].0, "In116_m1", "the isomer is most of it");
    assert!((split_of(&capture.split, "In116_m1") - 0.79).abs() < 1e-12);
    assert!((split_of(&capture.split, "In116") - 0.21).abs() < 1e-12);
    let n2n = channel("(n,2n)");
    assert!((split_of(&n2n.split, "In114_m1") - 0.9).abs() < 1e-12);
    assert!((split_of(&n2n.split, "In114") - 0.1).abs() < 1e-12);

    // The inventory holds the same split, the products being stable.
    let after = &results.get_final_material(0).expect("final").nuclides;
    let ratio = |m: &str, g: &str| after[m] / after[g];
    assert!(
        (ratio("In116_m1", "In116") - 0.79 / 0.21).abs() < 1e-9 * 0.79 / 0.21,
        "{after:?}"
    );
    assert!(
        (ratio("In114_m1", "In114") - 9.0).abs() < 1e-9 * 9.0,
        "{after:?}"
    );
}

/// ENDF/B-VIII.1's chain fixture with the fixture branching rows, and a
/// one-nuclide material of `nuclide` loaded at the scope a transmute loads, or
/// `None` where this machine lacks a fixture.
fn endf_b8_1_case(nuclide: &str) -> Option<(yani::LoadedChain, Material)> {
    let Some(path) = yamc_test_cache::nuclide(nuclide) else {
        eprintln!("skipping -- {nuclide} fixture absent");
        return None;
    };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let chain_fixture = manifest.join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    let dirs = [
        chain_fixture.join("decay"),
        chain_fixture.join("reactions"),
        chain_fixture.join("fission_yields"),
        manifest.join("tests/fixtures/endf-b8.1-branching-In115-Mo92"),
    ];
    if let Some(missing) = dirs.iter().find(|d| !d.is_dir()) {
        eprintln!("skipping -- {} absent", missing.display());
        return None;
    }
    let dir = |i: usize| dirs[i].to_string_lossy().into_owned();
    let loaded = yani::load_chain_parts(&dir(0), Some(&dir(1)), Some(&dir(2)), Some(&dir(3)))
        .expect("chain");
    let mut material = Material::new(
        HashMap::from([(nuclide.to_string(), 3.8e-2)]),
        "atom",
        "sum",
        None,
    )
    .expect("material");
    material.set_temperature("294");
    // At the scope a transmute loads, which is also the only scope a cache
    // entry fetched for one carries.
    let scope = yamc_nuclide::load_scope::LoadScope::activation(yani_transmute::activation_mts(
        &loaded.chain,
        &loaded.branch,
    ));
    let data = yamc_nuclide::nuclide::get_or_load_nuclide(
        nuclide,
        &HashMap::from([(nuclide.to_string(), path)]),
        &scope,
    )
    .expect("load");
    material.nuclide_data.insert(nuclide.to_string(), data);
    Some((loaded, material))
}

/// Each spectrum as one step, and the share `target` takes of `reaction` on
/// `parent` in each.
fn shares(
    loaded: &yani::LoadedChain,
    material: &mut Material,
    spectra: &[MultigroupSpectrum],
    parent: &str,
    reaction: &str,
    target: &str,
) -> Vec<f64> {
    let steps: Vec<TransmuteStep> = (0..spectra.len())
        .map(|i| TransmuteStep {
            dt: 60.0,
            irradiation: Some((i, 1.0e10)),
        })
        .collect();
    let results = transmute_material(
        material,
        spectra,
        &steps,
        Arc::clone(&loaded.chain),
        &loaded.branch,
        loaded.parts,
        None,
    )
    .expect("transmute");
    (0..spectra.len())
        .map(|step| {
            let channels = results.get_isomeric_branching(0, step).expect("step");
            let channel = channels
                .iter()
                .find(|c| c.parent == parent && c.reaction == reaction)
                .unwrap_or_else(|| panic!("{parent} {reaction} not reported at step {step}"));
            let report = results.get_branching_report(0, step).expect("report");
            let rule = report
                .channels
                .iter()
                .find(|c| c.parent == parent && c.reaction == reaction)
                .expect("the rule's report");
            assert!(!rule.complete, "{rule:?}");
            split_of(&channel.split, target)
        })
        .collect()
}

fn group(masses: [f64; 3]) -> MultigroupSpectrum {
    MultigroupSpectrum {
        boundaries: vec![1.0e-5, 0.1, 1.35e7, 1.45e7],
        masses: masses.to_vec(),
        flux_error: None,
    }
}

/// ENDF/B-VIII.1 itself: In115 capture makes In116_m1 at the evaluation's flat
/// 0.79 (it was never made at all), and 14 MeV (n,2n) makes mostly In114_m1,
/// whose partial is 1.33 of the 1.46 b total at 14 MeV.
#[test]
fn endf_b8_1_in115_makes_its_isomers() {
    let Some((loaded, mut material)) = endf_b8_1_case("In115") else {
        return;
    };
    let thermal = group([1.0, 0.0, 0.0]);
    let dt = group([0.0, 0.0, 1.0]);
    let capture = shares(
        &loaded,
        &mut material,
        std::slice::from_ref(&thermal),
        "In115",
        "(n,gamma)",
        "In116_m1",
    )[0];
    assert!((capture - 0.79).abs() < 1e-9, "In116_m1 share {capture}");
    let n2n = shares(
        &loaded,
        &mut material,
        std::slice::from_ref(&dt),
        "In115",
        "(n,2n)",
        "In114_m1",
    )[0];
    assert!((0.88..0.94).contains(&n2n), "In114_m1 share {n2n}");
}

/// ENDF/B-VIII.1 Mo92 (n,p) lists Nb92_m1 alone, an MF=10 partial, which made
/// no Nb92_m1 before the isomer-only rule. Over 13.5-14.5 MeV it is 0.571 of
/// the reaction: the partial is 0.59 of MF=3 at 13.5 MeV and 0.55 at 14.5.
#[test]
fn endf_b8_1_mo92_makes_its_isomer() {
    let Some((loaded, mut material)) = endf_b8_1_case("Mo92") else {
        return;
    };
    let share = shares(
        &loaded,
        &mut material,
        &[group([0.0, 0.0, 1.0])],
        "Mo92",
        "(n,p)",
        "Nb92_m1",
    )[0];
    assert!((share - 0.571).abs() < 1e-3, "Nb92_m1 share {share}");
}
