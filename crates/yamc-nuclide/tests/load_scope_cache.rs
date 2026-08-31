//! Issue #389: the global nuclide cache must not hand a narrow load to a caller
//! that needs a wide one, and must not thrash between two narrow callers.
//!
//! One test, deliberately. The cache is process-global and keyed by
//! `{name}@{path}`, so two tests in this binary racing on the same fixture
//! would make the assertions order-dependent.

use std::collections::HashMap;
use std::sync::Arc;
use yamc_nuclide::nuclide::get_or_load_nuclide;
use yamc_nuclide::{LoadScope, Nuclide};

fn path_map() -> Option<HashMap<String, String>> {
    let fe56 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../yamc/tests/Fe56.arrow");
    if !fe56.is_dir() {
        return None;
    }
    Some(HashMap::from([(
        "Fe56".to_string(),
        fe56.to_string_lossy().to_string(),
    )]))
}

fn activation(mts: &[i32]) -> LoadScope {
    LoadScope::activation(mts.iter().copied().collect())
}

fn mts_of(n: &Nuclide) -> Vec<i32> {
    let mut v: Vec<i32> = n.reactions[0].keys().copied().collect();
    v.sort_unstable();
    v
}

#[test]
fn cache_widens_rather_than_thrashing_or_leaking_a_narrow_load() {
    let Some(map) = path_map() else {
        eprintln!("skip: Fe56.arrow not present");
        return;
    };

    // Hold every Arc alive: the cache stores Weak refs, so dropping one would
    // turn the next lookup into an ordinary miss and prove nothing.
    let first: Arc<Nuclide> =
        get_or_load_nuclide("Fe56", &map, &activation(&[102])).expect("activation load");
    assert_eq!(mts_of(&first), vec![102]);
    assert!(first.fast_xs.is_empty());

    // Same scope: must be the very same Arc, not a reload.
    let again = get_or_load_nuclide("Fe56", &map, &activation(&[102])).expect("cached load");
    assert!(
        Arc::ptr_eq(&first, &again),
        "an identical scope should hit the cache"
    );

    // A different MT set is not covered, so this reloads. It must come back
    // holding the UNION, otherwise the two callers evict each other forever.
    let widened =
        get_or_load_nuclide("Fe56", &map, &activation(&[103])).expect("second activation");
    assert!(!Arc::ptr_eq(&first, &widened), "a new MT should reload");
    assert_eq!(
        mts_of(&widened),
        vec![102, 103],
        "the reload should carry the union of both requests"
    );

    // And now the first caller's scope is still satisfied without a reload.
    let revisit = get_or_load_nuclide("Fe56", &map, &activation(&[102])).expect("revisit");
    assert!(
        Arc::ptr_eq(&widened, &revisit),
        "the widened entry should serve the original narrower request"
    );

    // Transport asks for everything. An XsOnly entry can never satisfy that:
    // it has no products, no distributions and no fast_xs.
    let full = get_or_load_nuclide("Fe56", &map, &LoadScope::full()).expect("full load");
    assert!(!Arc::ptr_eq(&widened, &full), "Full must not reuse XsOnly");
    assert!(
        !full.fast_xs.is_empty(),
        "a full load must build the fast_xs accelerator"
    );
    assert!(
        full.reactions[0].len() > 2,
        "a full load must carry every MT, not just the two asked for earlier"
    );

    // Having widened all the way to Full, the narrow requests are covered too.
    let after = get_or_load_nuclide("Fe56", &map, &activation(&[102])).expect("narrow after full");
    assert!(
        Arc::ptr_eq(&full, &after),
        "a full entry should serve any activation request"
    );
}
