//! MT=5's `(n,X)` rows: read into the chain and the branching table, written
//! back by a re-export, refused where the matrix would solve with them
//! unfolded, and refused where a file says something the converter never
//! writes.

use std::collections::HashMap;
use std::sync::Arc;

use yani::reactions::ANYTHING;
use yani::{
    build_matrix_triplets, export_chain_parts, parse_chain_parts, BranchCurve, BranchQuantity,
    BranchState, ChainNuclide, ChainParts, ChainReaction, FissionYieldWeights, ReactionRates,
};

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: Vec::new(),
        fission_yields: None,
        sources: Vec::new(),
    }
}

/// A product's curve as the reader builds it from a row.
fn curve(target: &str, share: bool, lfs: i32, values: &[f64]) -> BranchCurve {
    BranchCurve {
        target: target.to_string(),
        quantity: if share {
            BranchQuantity::Yield
        } else {
            BranchQuantity::Multiplicity
        },
        energy: vec![1.0e7, 2.0e7],
        values: values.to_vec(),
        states: Arc::from(vec![BranchState {
            mt: 5,
            lfs,
            lmf: None,
            list_complete: true,
            level_route: if lfs == 0 { "ground" } else { "level_index" }.to_string(),
            level_energy: 0.0,
            level_energy_difference: (lfs == 0).then_some(0.0),
            mf3_cross_section: None,
        }]),
        normalisation: None,
    }
}

fn row(kind: &str, target: Option<&str>, branching: f64, c: Option<BranchCurve>) -> ChainReaction {
    ChainReaction {
        kind: kind.to_string(),
        target: target.map(str::to_string),
        branching,
        q_value: Some(-1.0e6),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: c.map(Arc::new),
    }
}

/// Fe54 with an `(n,X)` residual pair (Mn52 and its isomer), one light
/// particle, and Fe58 with the row that says no residual is given.
fn chain() -> HashMap<String, ChainNuclide> {
    let fe54 = vec![
        row("(n,p)", Some("Mn54"), 1.0, None),
        row(
            ANYTHING,
            Some("Mn52"),
            f64::NAN,
            Some(curve("Mn52", true, 0, &[0.6, 0.5])),
        ),
        row(
            ANYTHING,
            Some("Mn52_m1"),
            f64::NAN,
            Some(curve("Mn52_m1", true, 1, &[0.4, 0.5])),
        ),
        row(
            ANYTHING,
            Some("He4"),
            f64::NAN,
            Some(curve("He4", false, 0, &[0.2, 1.4])),
        ),
    ];
    let fe58 = vec![
        row(
            ANYTHING,
            Some("H1"),
            f64::NAN,
            Some(curve("H1", false, 0, &[0.1, 0.3])),
        ),
        row(ANYTHING, None, 0.0, None),
    ];
    [
        nuclide("Fe54", fe54),
        nuclide("Fe58", fe58),
        nuclide("Mn54", Vec::new()),
        nuclide("Mn52", Vec::new()),
        nuclide("Mn52_m1", Vec::new()),
        nuclide("He4", Vec::new()),
        nuclide("H1", Vec::new()),
    ]
    .into_iter()
    .map(|n| (n.name.clone(), n))
    .collect()
}

fn round_trip(written: &HashMap<String, ChainNuclide>, tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("yani-anything-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    export_chain_parts(written, &dir, Some("test")).expect("export");
    dir
}

fn anything_rows(n: &ChainNuclide) -> Vec<(Option<&str>, Option<&BranchCurve>)> {
    n.reactions
        .iter()
        .filter(|r| r.kind == ANYTHING)
        .map(|r| (r.target.as_deref(), r.multiplicity.as_deref()))
        .collect()
}

/// Every row comes back with its curve, the curves are in the branching table
/// under `(n,X)` whatever branching subsection is loaded (here none), and a
/// second export writes the same file as the first.
#[test]
fn anything_rows_survive_a_round_trip_and_reach_the_branching_table() {
    let written = chain();
    let dir = round_trip(&written, "trip");
    let (read, branch) =
        parse_chain_parts(&dir.join("decay"), Some(&dir.join("reactions")), None, None)
            .expect("read");
    for parent in ["Fe54", "Fe58"] {
        let a = anything_rows(&written[parent]);
        let b = anything_rows(&read[parent]);
        assert_eq!(a.len(), b.len(), "{parent}");
        for ((ta, ca), (tb, cb)) in a.iter().zip(&b) {
            assert_eq!(ta, tb);
            match (ca, cb) {
                (Some(ca), Some(cb)) => {
                    assert_eq!(ca.quantity, cb.quantity, "{parent} {ta:?}");
                    assert_eq!(ca.energy, cb.energy);
                    assert_eq!(ca.values, cb.values);
                    assert_eq!(ca.states[0], cb.states[0]);
                }
                (None, None) => {}
                _ => panic!("{parent} {ta:?}: a curve was lost or invented"),
            }
        }
        let curves = &branch.curves()[parent][ANYTHING];
        assert_eq!(curves.len(), b.iter().filter(|(_, c)| c.is_some()).count());
    }
    // The unfolded split is NaN, the row with no target zero.
    let fe58 = &read["Fe58"].reactions;
    assert!(fe58
        .iter()
        .any(|r| r.target.is_none() && r.branching == 0.0));
    assert!(fe58
        .iter()
        .filter(|r| r.target.is_some())
        .all(|r| r.branching.is_nan()));

    let again = round_trip(&read, "again");
    let bytes = |d: &std::path::Path| std::fs::read(d.join("reactions/reactions.arrow")).unwrap();
    assert_eq!(bytes(&dir), bytes(&again), "a re-export changes nothing");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&again);
}

/// A rate reaching an `(n,X)` split no fold has set is refused rather than
/// solved into NaN; once the split is set, the residuals take the rate as
/// shares and a light particle its multiplicity, while the parent is removed
/// once.
#[test]
fn an_unfolded_split_is_refused_and_a_folded_one_builds_the_matrix() {
    let mut chain = chain();
    let names: Vec<String> = {
        let mut n: Vec<String> = chain.keys().cloned().collect();
        n.sort();
        n
    };
    let rates: ReactionRates = HashMap::from([(
        "Fe54".to_string(),
        HashMap::from([(ANYTHING.to_string(), 2.0)]),
    )]);
    let build = |chain: &HashMap<String, ChainNuclide>| {
        build_matrix_triplets(
            chain,
            &names,
            &rates,
            &FissionYieldWeights::new(),
            ChainParts::default(),
        )
    };
    let err = build(&chain).unwrap_err();
    assert!(
        err.contains("Fe54") && err.contains("not been folded"),
        "{err}"
    );

    for r in chain.get_mut("Fe54").unwrap().reactions.iter_mut() {
        match r.target.as_deref() {
            Some("Mn52") => r.branching = 0.55,
            Some("Mn52_m1") => r.branching = 0.45,
            Some("He4") => r.branching = 1.3,
            _ => {}
        }
    }
    let (triplets, _) = build(&chain).expect("folded");
    let at = |row: &str, col: &str| -> f64 {
        let (r, c) = (
            names.iter().position(|n| n == row).unwrap(),
            names.iter().position(|n| n == col).unwrap(),
        );
        triplets
            .iter()
            .filter(|t| t.0 == r && t.1 == c)
            .map(|t| t.2)
            .sum()
    };
    assert_eq!(at("Fe54", "Fe54"), -2.0);
    assert!((at("Mn52", "Fe54") - 1.1).abs() < 1e-15);
    assert!((at("Mn52_m1", "Fe54") - 0.9).abs() < 1e-15);
    assert!((at("He4", "Fe54") - 2.6).abs() < 1e-15);
}

/// A multiplicity on a row of another kind is not something the converter
/// writes, and is refused.
#[test]
fn a_multiplicity_on_another_kind_is_refused() {
    let mut chain = chain();
    chain.get_mut("Fe54").unwrap().reactions[0].multiplicity =
        Some(Arc::new(curve("Mn54", true, 0, &[1.0, 1.0])));
    let dir = round_trip(&chain, "bad");
    let err = parse_chain_parts(&dir.join("decay"), Some(&dir.join("reactions")), None, None)
        .unwrap_err()
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(err.contains("only (n,X) rows"), "{err}");
}
