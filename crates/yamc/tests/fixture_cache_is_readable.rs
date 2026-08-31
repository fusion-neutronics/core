//! The fixtures a job downloaded must be the fixtures its tests can read.
//!
//! Issue #544: 35 test files resolved the cache through `$HOME` alone, which is
//! not a Windows variable, so on `windows-latest` every one of them resolved
//! under a literal `/home/jon`, found nothing, and took its "data absent" skip
//! path. They passed while reading nothing, and the whole suite finished in
//! 0.00s. `scripts/fetch_test_fixtures.py` had downloaded the data to
//! `C:\\Users\\runneradmin\\.cache\\yamc` and nothing ever opened it.
//!
//! The resolution is one rule now (`yamc_nuclide::url_cache::cache_root`, which
//! `yamc_test_cache` wraps), so that exact failure cannot recur. What can recur
//! is the SHAPE of it: the fetch script, `actions/cache`'s `path:` and this
//! crate each decide where the cache is, and a job whose three answers drift
//! apart reports green over a suite of skips.
//!
//! So `YAMC_REQUIRE_FIXTURES=1` turns absence into a failure. CI sets it, right
//! after the fetch step that is supposed to have produced them; nothing else
//! does, so a developer with a partial cache still gets skips.

/// One nuclide and one element from `scripts/fetch_test_fixtures.py`, chosen to
/// cover both halves of the cache layout rather than to be exhaustive: if the
/// root resolves at all, it resolves for every entry under it.
const REQUIRED_NUCLIDE: &str = "Fe56";
const REQUIRED_ELEMENT: &str = "Fe";

fn required() -> bool {
    std::env::var("YAMC_REQUIRE_FIXTURES").as_deref() == Ok("1")
}

#[test]
fn the_resolved_cache_holds_what_the_fetch_script_downloads() {
    let root = yamc_test_cache::root();
    // Printed either way. When this does skip, the path it skipped on is the
    // first thing anyone reading the log needs.
    eprintln!("fixture cache root: {}", root.display());

    let missing: Vec<&str> = [REQUIRED_NUCLIDE, REQUIRED_ELEMENT]
        .into_iter()
        .filter(|name| !yamc_test_cache::have(name))
        .collect();

    if missing.is_empty() {
        return;
    }

    assert!(
        !required(),
        "YAMC_REQUIRE_FIXTURES=1, so the fixtures scripts/fetch_test_fixtures.py \
         downloads must be readable at the cache this build resolves, but {missing:?} \
         are absent from {}. Either the fetch step did not run, or it wrote \
         somewhere else: it uses YAMC_CACHE_DIR then Path.home(), and this uses \
         YAMC_CACHE_DIR then USERPROFILE on Windows and HOME elsewhere.",
        root.display()
    );
    eprintln!("skip -- {missing:?} absent; run scripts/fetch_test_fixtures.py");
}

/// Present is not the same as readable. A directory can exist and hold a
/// half-written or narrower-scope entry, which is the other way a suite reports
/// green over nothing. Under the switch that is a failure; without it, a skip,
/// because a developer machine that has run a transmutation legitimately holds
/// entries this cannot read.
#[test]
fn a_required_fixture_actually_loads() {
    let Some(path) = yamc_test_cache::nuclide(REQUIRED_NUCLIDE) else {
        assert!(
            !required(),
            "YAMC_REQUIRE_FIXTURES=1 but {REQUIRED_NUCLIDE} is absent from {}",
            yamc_test_cache::root().display()
        );
        eprintln!("skip -- no {REQUIRED_NUCLIDE} fixture");
        return;
    };

    // A read failure is a skip too, unless the switch is on. An entry can be
    // present and still not be this: an activation load caches a nuclide at
    // cross-sections-only scope, and when it names specific MTs it writes only
    // `subset/reactions.arrow` with the canonical name deliberately absent. An
    // interrupted fetch leaves the same shape. Panicking on that would fail
    // every developer who has ever run a transmutation, which is exactly the
    // hazard crates/yamc/tests/matched_stream_localize.rs already documents.
    let nuclide = match yamc_nuclide::nuclide_arrow::read_nuclide_from_arrow(
        std::path::Path::new(&path),
        &yamc_nuclide::LoadScope::full(),
    ) {
        Ok(nuclide) => nuclide,
        Err(e) => {
            assert!(
                !required(),
                "YAMC_REQUIRE_FIXTURES=1, so the fixtures the fetch step downloaded must be \
                 readable, but {path} is not: {e}"
            );
            eprintln!("skip -- {path} is cached at a narrower scope or incomplete ({e})");
            return;
        }
    };

    assert_eq!(nuclide.name.as_deref(), Some(REQUIRED_NUCLIDE));
    assert!(
        !nuclide.loaded_temperatures.is_empty(),
        "{path} loaded no temperatures"
    );
    assert!(
        nuclide.reaction_mts().is_some_and(|mts| mts.contains(&2)),
        "{path} carries no elastic scattering"
    );
}
