//! The bytes entry point must load exactly what the path one does.
//!
//! `parse_chain_parts` is now a loader in front of `parse_chain_parts_from_bytes`,
//! which is what a host with no filesystem calls. If the two ever disagree, a
//! browser build silently transmutes against a different chain from the wheel.

use std::path::{Path, PathBuf};

use yani::{parse_chain_parts, parse_chain_parts_from_bytes, ChainNuclide, ChainSections};

/// The split-layout chain from the downloaded test fixtures.
///
/// `python3 scripts/fetch_test_fixtures.py` populates this; skip rather than
/// fail when it is absent, matching how the rest of the suite treats fixtures.
fn fixture() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yamc/tests/transmutation-endf-b8.1-sfr.arrow");
    dir.join("decay/nuclides.arrow").exists().then_some(dir)
}

/// Load the same seven files a filesystem caller would, as bytes.
fn sections(root: &Path) -> ChainSections {
    let mut parts = ChainSections::default();
    for (subsection, dir, file) in [
        ("decay", "decay", "nuclides.arrow"),
        ("decay", "decay", "decay_modes.arrow"),
        ("decay", "decay", "sources.arrow"),
        ("reactions", "reactions", "reactions.arrow"),
        ("fission_yields", "fission_yields", "fission_yields.arrow"),
        ("fission_yields", "fission_yields", "aliases.arrow"),
        ("branching", "branching", "branching.arrow"),
    ] {
        let path = root.join(dir).join(file);
        if path.exists() {
            parts
                .insert(subsection, file, std::fs::read(&path).unwrap())
                .unwrap();
        }
    }
    parts
}

/// Everything about a nuclide that a transmutation depends on, in a form two
/// chains can be compared by. `ChainNuclide` carries no `PartialEq`.
fn fingerprint(n: &ChainNuclide) -> String {
    let mut reactions: Vec<String> = n
        .reactions
        .iter()
        .map(|r| format!("{}->{:?}@{}", r.kind, r.target, r.branching))
        .collect();
    reactions.sort();
    let mut decays: Vec<String> = n
        .decays
        .iter()
        .map(|d| format!("{}->{:?}@{}", d.kind, d.target, d.branching))
        .collect();
    decays.sort();
    format!(
        "{} hl={:?} q={} rx=[{}] dk=[{}] fy={} src={}",
        n.name,
        n.half_life,
        n.decay_energy,
        reactions.join(","),
        decays.join(","),
        n.fission_yields.as_ref().map_or(0, |set| set.yields.len()),
        n.sources.len(),
    )
}

#[test]
fn bytes_and_paths_load_the_same_chain() {
    let Some(root) = fixture() else {
        eprintln!("skipping: run scripts/fetch_test_fixtures.py first");
        return;
    };

    let (from_paths, branch_from_paths) = parse_chain_parts(
        &root.join("decay"),
        Some(&root.join("reactions")),
        Some(&root.join("fission_yields")),
        Some(&root.join("branching")),
    )
    .expect("path load");

    let (from_bytes, branch_from_bytes) =
        parse_chain_parts_from_bytes(&sections(&root)).expect("bytes load");

    assert!(!from_paths.is_empty(), "fixture chain should not be empty");
    assert_eq!(
        from_paths.len(),
        from_bytes.len(),
        "different nuclide counts",
    );

    let mut names: Vec<&String> = from_paths.keys().collect();
    names.sort();
    for name in names {
        let want = &from_paths[name];
        let got = from_bytes.get(name).unwrap_or_else(|| {
            panic!("{name} present via paths but absent via bytes");
        });
        assert_eq!(fingerprint(got), fingerprint(want), "{name} differs");
    }

    assert_eq!(
        branch_from_bytes.len(),
        branch_from_paths.len(),
        "branch tables differ in size",
    );
}

#[test]
fn a_chain_without_decay_nuclides_is_rejected() {
    // The guard that keeps a missing required section from loading as an empty
    // chain, which would transmute to nothing rather than complaining.
    let err = parse_chain_parts_from_bytes(&ChainSections::default())
        .expect_err("an empty chain must not load");
    assert!(
        err.to_string().contains("decay/nuclides.arrow"),
        "error should name the missing section, got: {err}",
    );
}

#[test]
fn an_unknown_subsection_is_rejected() {
    // Dropping the bytes instead would load a partial chain and silently omit
    // whatever the caller thought it was supplying.
    let mut parts = ChainSections::default();
    let err = parts
        .insert("decays", "nuclides.arrow", Vec::new())
        .expect_err("'decays' is not a subsection name");
    assert!(err.to_string().contains("decays"), "got: {err}");
}
