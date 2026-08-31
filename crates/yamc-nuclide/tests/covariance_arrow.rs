//! Reading `covariance.arrow`, and reading a directory that has none.
//!
//! The absence case is the one with a promise attached. Every `{Nuclide}.arrow/`
//! published before this section existed has no such file, and most evaluations
//! have no MF=33 at all, so a missing file has to load exactly as it does today
//! rather than becoming an error or a zero. The schema says so; this is what
//! holds it to it.

use std::collections::HashSet;
use std::path::PathBuf;

use yamc_nuclide::arrow::covariance_arrow::read_covariance;
use yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow;
use yamc_nuclide::LoadScope;

/// A cached Fe56, if this machine has the fixtures.
fn fe56() -> Option<PathBuf> {
    yamc_test_cache::nuclide("Fe56").map(PathBuf::from)
}

fn skip(what: &str) {
    eprintln!("skipping: {what} (nuclear-data fixtures missing)");
}

/// A published directory has no `covariance.arrow`, and must load unchanged.
///
/// Two halves, and both matter. The read returns `None` rather than erroring,
/// and a nuclide loaded WITH `covariance` set still loads: asking for something
/// the data does not have is not a failure, it is an absent optional section.
#[test]
fn a_directory_without_covariance_still_loads() {
    let Some(dir) = fe56() else {
        return skip("a_directory_without_covariance_still_loads");
    };
    assert!(
        !dir.join("covariance.arrow").exists(),
        "this test is about the published data, which carries no covariance"
    );

    assert!(
        read_covariance(&dir, "Fe56")
            .expect("absence is not an error")
            .is_none(),
        "a missing file reads as no covariance"
    );

    let scope = LoadScope::activation(HashSet::from([102])).with_covariance(true);
    let nuclide = read_nuclide_from_arrow(&dir, &scope).expect("Fe56 loads");
    assert!(
        nuclide.covariance.is_none(),
        "asked for, not present, so None -- not an error and not an empty vector"
    );
    assert!(
        nuclide.load_scope.covariance,
        "the scope records what was asked for, so the cache can tell this entry \
         apart from one loaded without covariance"
    );
}

/// The same directory read WITHOUT asking for covariance is byte-for-byte the
/// load it always was.
#[test]
fn not_asking_for_covariance_reads_nothing() {
    let Some(dir) = fe56() else {
        return skip("not_asking_for_covariance_reads_nothing");
    };
    let scope = LoadScope::activation(HashSet::from([102]));
    assert!(!scope.covariance, "off by default");

    let nuclide = read_nuclide_from_arrow(&dir, &scope).expect("Fe56 loads");
    assert!(nuclide.covariance.is_none());
    assert!(!nuclide.load_scope.covariance);
}

/// A cached entry loaded without covariance does not satisfy a request for it.
///
/// The failure this prevents is silent: the global cache hands back an entry it
/// already holds when the scope covers the request, and without this axis a
/// nuclide loaded for an ordinary transmute would be reused for an uncertainty
/// run and report no covariance at all.
#[test]
fn a_load_without_covariance_does_not_cover_one_with_it() {
    let plain = LoadScope::activation(HashSet::from([102]));
    let with = LoadScope::activation(HashSet::from([102])).with_covariance(true);

    assert!(!plain.covers(&with), "narrow must not claim to cover wide");
    assert!(with.covers(&plain), "wide covers narrow");
    assert!(
        plain.union(&with).covariance,
        "the union carries covariance, so a reload serves both callers"
    );
}
