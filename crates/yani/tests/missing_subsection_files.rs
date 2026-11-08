//! A reactions or fission-yields folder without its primary file is refused.
//!
//! The reader used to treat a missing `reactions.arrow` or
//! `fission_yields.arrow` as an absent section, so a source pointing at the
//! wrong folder loaded as a chain with no reactions (or no yields) that still
//! counted the part as loaded, and every rate through it produced nothing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use yani::{export_chain_parts, parse_chain_parts, ChainNuclide, ChainReaction};

/// A two-nuclide chain with one reaction, exported to a fresh folder.
fn exported(tag: &str) -> PathBuf {
    let nuclide = |name: &str, reactions: Vec<ChainReaction>| ChainNuclide {
        name: name.to_string(),
        half_life: None,
        half_life_uncertainty: None,
        decay_energy: 0.0,
        decay_energy_uncertainty: None,
        decay_energy_components: Default::default(),
        reactions,
        decays: Vec::new(),
        fission_yields: None,
        sources: Default::default(),
    };
    let capture = ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("Fe57".to_string()),
        branching: 1.0,
        q_value: Some(7.6e6),
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    let chain: HashMap<String, ChainNuclide> =
        [nuclide("Fe56", vec![capture]), nuclide("Fe57", Vec::new())]
            .into_iter()
            .map(|n| (n.name.clone(), n))
            .collect();
    let dir = std::env::temp_dir().join(format!("yani-missing-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    export_chain_parts(&chain, &dir, Some("test")).expect("export");
    dir
}

fn refusal(decay: &Path, reactions: Option<&Path>, fission_yields: Option<&Path>) -> String {
    parse_chain_parts(decay, reactions, fission_yields, &[])
        .map(|_| ())
        .expect_err("a folder without its primary file must be refused")
        .to_string()
}

#[test]
fn a_reactions_folder_without_reactions_arrow_is_refused() {
    let dir = exported("reactions");
    let reactions = dir.join("reactions");
    let (chain, _) = parse_chain_parts(&dir.join("decay"), Some(&reactions), None, &[])
        .expect("the exported chain reads");
    assert_eq!(chain["Fe56"].reactions.len(), 1);

    std::fs::remove_file(reactions.join("reactions.arrow")).unwrap();
    let err = refusal(&dir.join("decay"), Some(&reactions), None);
    assert!(err.contains("has no reactions.arrow"), "{err}");
    assert!(err.contains("transmutation_reactions = False"), "{err}");
    // Leaving the part out on purpose still works.
    assert!(parse_chain_parts(&dir.join("decay"), None, None, &[]).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_fission_yields_folder_without_fission_yields_arrow_is_refused() {
    let dir = exported("yields");
    let yields = dir.join("fission_yields");
    assert!(parse_chain_parts(&dir.join("decay"), None, Some(&yields), &[]).is_ok());

    std::fs::remove_file(yields.join("fission_yields.arrow")).unwrap();
    let err = refusal(&dir.join("decay"), None, Some(&yields));
    assert!(err.contains("has no fission_yields.arrow"), "{err}");
    assert!(
        err.contains("transmutation_fission_yields = False"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A source naming a folder that is not there says so.
#[test]
fn a_missing_folder_is_named_as_missing() {
    let dir = exported("absent");
    let err = refusal(&dir.join("decay"), Some(&dir.join("no-such-folder")), None);
    assert!(err.contains("(the directory does not exist)"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
