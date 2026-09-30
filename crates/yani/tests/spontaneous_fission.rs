//! A spontaneous-fission mode is read as removing its parent and making
//! nothing.
//!
//! The decay files write an `sf` mode's target as the chain build's daughter
//! rule gives it, and fission moves neither Z nor A, so ENDF/B-VIII.1's Cf252
//! carries `sf -> Cf252` at 0.03092. The reader keeps the ratio, which is
//! evaluated data, and drops the target, so nothing downstream of it sees a
//! self-edge. These pin that on both layouts, and that a chain with no such mode
//! reads, and builds its matrix, exactly as before.

use std::collections::HashMap;
use std::path::Path;

use yani::{
    build_matrix_triplets, export_chain_parts, parse_chain_arrow, parse_chain_parts, ChainNuclide,
    ChainParts, ChainReaction, FissionYieldWeights, ReactionRates,
};

/// The flat-layout ENDF/B-VIII.1 chain committed with this crate, and the
/// split-layout one from the downloaded fixtures when they have been fetched.
fn libraries() -> Vec<(&'static str, HashMap<String, ChainNuclide>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = vec![(
        "flat",
        parse_chain_arrow(root.join("tests/transmutation-endf-b8.1-sfr.arrow"))
            .expect("parse the committed chain"),
    )];
    let split = root.join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    if split.join("decay/nuclides.arrow").exists() {
        let (chain, _branch) = parse_chain_parts(
            &split.join("decay"),
            Some(&split.join("reactions")),
            Some(&split.join("fission_yields")),
            None,
        )
        .expect("parse the published chain");
        out.push(("split", chain));
    } else {
        eprintln!("split layout skipped: run scripts/fetch_test_fixtures.py");
    }
    out
}

fn modes(nuclide: &ChainNuclide) -> Vec<(&str, Option<&str>, f64)> {
    nuclide
        .decays
        .iter()
        .map(|d| (d.kind.as_str(), d.target.as_deref(), d.branching))
        .collect()
}

fn involves_fission(kind: &str) -> bool {
    kind.split(',').any(|mode| mode == "sf")
}

#[test]
fn cf252_keeps_its_fission_ratio_and_loses_the_self_edge() {
    for (layout, chain) in libraries() {
        assert_eq!(
            modes(&chain["Cf252"]),
            vec![("alpha", Some("Cm248"), 0.96908), ("sf", None, 0.03092)],
            "{layout} layout"
        );
    }
}

#[test]
fn no_decay_mode_in_the_library_is_a_self_edge_or_names_a_fission_product() {
    for (layout, chain) in libraries() {
        let mut fission_modes = 0;
        for (name, nuclide) in &chain {
            for d in &nuclide.decays {
                assert_ne!(
                    d.target.as_deref(),
                    Some(name.as_str()),
                    "{layout}: {name} {} names its own parent",
                    d.kind
                );
                if involves_fission(&d.kind) {
                    fission_modes += 1;
                    assert_eq!(d.target, None, "{layout}: {name} {}", d.kind);
                }
            }
        }
        // 125 `sf` and 3 `ec/beta+,sf`, so the loop above was not vacuous.
        assert_eq!(fission_modes, 128, "{layout} layout");
    }
}

fn nuclide(name: &str, half_life: Option<f64>, decays: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions: Vec::new(),
        decays,
        fission_yields: None,
        sources: Vec::new(),
    }
}

fn mode(kind: &str, target: &str, branching: f64) -> ChainReaction {
    ChainReaction {
        kind: kind.to_string(),
        target: Some(target.to_string()),
        branching,
        q_value: None,
        branching_uncertainty: None,
    }
}

#[test]
fn a_chain_without_such_a_mode_builds_the_same_matrix_after_a_round_trip() {
    // Bi212 branches without fission, and sits in the same chain as a Cf252
    // written the way the published files write it.
    let written: HashMap<String, ChainNuclide> = [
        nuclide(
            "Bi212",
            Some(3633.0),
            vec![
                mode("alpha", "Tl208", 0.3594),
                mode("beta-", "Po212", 0.6406),
            ],
        ),
        nuclide("Po212", Some(2.94e-7), vec![mode("alpha", "Pb208", 1.0)]),
        nuclide("Tl208", Some(183.18), vec![mode("beta-", "Pb208", 1.0)]),
        nuclide("Pb208", None, Vec::new()),
        nuclide("He4", None, Vec::new()),
        nuclide(
            "Cf252",
            Some(8.3468070e7),
            vec![
                mode("alpha", "Cm248", 0.96908),
                mode("sf", "Cf252", 0.03092),
            ],
        ),
        nuclide("Cm248", None, Vec::new()),
    ]
    .into_iter()
    .map(|n| (n.name.clone(), n))
    .collect();

    let dir = std::env::temp_dir().join(format!("yani-sf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    export_chain_parts(&written, &dir, Some("test")).expect("export");
    let (read, _branch) = parse_chain_parts(&dir.join("decay"), None, None, None).expect("read");
    let _ = std::fs::remove_dir_all(&dir);

    for name in ["Bi212", "Po212", "Tl208"] {
        assert_eq!(modes(&read[name]), modes(&written[name]), "{name}");
    }
    let mut names: Vec<String> = ["Bi212", "He4", "Pb208", "Po212", "Tl208"]
        .map(String::from)
        .to_vec();
    names.sort();
    let matrix = |chain: &HashMap<String, ChainNuclide>| {
        build_matrix_triplets(
            chain,
            &names,
            &ReactionRates::new(),
            &FissionYieldWeights::new(),
            ChainParts::default(),
        )
        .expect("matrix")
    };
    let (before, after) = (matrix(&written), matrix(&read));
    assert_eq!(before.1, after.1);
    let bits = |t: &[(usize, usize, f64)]| -> Vec<(usize, usize, u64)> {
        t.iter().map(|&(r, c, v)| (r, c, v.to_bits())).collect()
    };
    assert_eq!(bits(&before.0), bits(&after.0), "Bi212's matrix moved");

    assert_eq!(
        modes(&read["Cf252"]),
        vec![("alpha", Some("Cm248"), 0.96908), ("sf", None, 0.03092)]
    );
}
