//! A chain written before the uncertainty columns existed must still load.
//!
//! `half_life_uncertainty` and `decay_energy_uncertainty` are nullable and
//! appended last precisely so that older files keep working, and the committed
//! `transmutation-endf-b8.1-sfr.arrow` fixture is one: it carries `name`,
//! `half_life` and `decay_energy` and nothing else. That makes it the only
//! check in the workspace that the promise holds against a real file rather
//! than one a test just wrote.
//!
//! What it must yield is `None`, not `0.0`. Those are different claims -- the
//! file does not state an uncertainty, which is not the same as stating that
//! there is none -- and a reader that defaulted to zero would report the
//! oldest data in the repository as the most precisely known.

use std::path::PathBuf;

use yani::parse_chain_arrow;

#[test]
fn a_chain_without_the_uncertainty_columns_loads_and_states_nothing() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/transmutation-endf-b8.1-sfr.arrow");
    let chain = parse_chain_arrow(&path).expect("a chain predating the columns still parses");
    assert!(!chain.is_empty(), "the fixture should hold nuclides");

    let decaying: Vec<_> = chain.values().filter(|n| n.half_life.is_some()).collect();
    assert!(
        !decaying.is_empty(),
        "the fixture should hold unstable nuclides, or this proves nothing"
    );

    for nuclide in chain.values() {
        assert_eq!(
            nuclide.half_life_uncertainty, None,
            "{}: a column the file does not have must read as unstated, not zero",
            nuclide.name
        );
        assert_eq!(
            nuclide.decay_energy_uncertainty, None,
            "{}: a column the file does not have must read as unstated, not zero",
            nuclide.name
        );
    }
}
