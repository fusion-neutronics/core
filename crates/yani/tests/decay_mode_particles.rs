//! Every decay mode a published chain spells yields the light particles its
//! parts name.
//!
//! The decay reader names a mode by joining one part per RTYP digit
//! (`crates/endf/src/decay.rs`), so two protons are `ec/beta+,p,p`. A table
//! keyed by a different spelling (`ec/beta+,2p`) silently emitted nothing.

use std::path::PathBuf;

use yani::matrix::decay_particle_products;
use yani::parse_chain_arrow;

#[test]
fn every_chain_decay_mode_yields_the_particles_its_parts_name() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/transmutation-endf-b8.1-sfr.arrow");
    if !path.is_dir() {
        eprintln!("skipping: chain fixture not fetched");
        return;
    }
    let chain = parse_chain_arrow(&path).expect("parse the test chain");
    let known: Vec<&str> = endf::decay::DECAY_MODES.iter().map(|(n, _)| *n).collect();
    let mut modes: Vec<&str> = chain
        .values()
        .flat_map(|n| n.decays.iter().map(|d| d.kind.as_str()))
        .collect();
    modes.sort_unstable();
    modes.dedup();
    assert!(!modes.is_empty());
    for mode in modes {
        let parts: Vec<&str> = mode.split(',').collect();
        for part in &parts {
            assert!(
                known.contains(part),
                "mode {mode:?} has a part {part:?} the decay reader does not name"
            );
        }
        let count = |name: &str| parts.iter().filter(|p| **p == name).count() as f64;
        let products = decay_particle_products(mode);
        let got = |nuclide: &str| {
            products
                .iter()
                .find(|(n, _)| *n == nuclide)
                .map_or(0.0, |(_, c)| *c)
        };
        assert_eq!(got("H1"), count("p"), "{mode}");
        assert_eq!(got("He4"), count("alpha"), "{mode}");
    }
}
