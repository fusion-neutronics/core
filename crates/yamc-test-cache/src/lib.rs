//! Where the test suites find the nuclear-data fixtures.
//!
//! Thirty-five test files used to resolve this themselves, with
//!
//! ```ignore
//! let home = std::env::var("HOME").unwrap_or_else(|_| "/home/jon".to_string());
//! format!("{home}/.cache/yamc/endf-b8.1-{n}.arrow")
//! ```
//!
//! which is wrong twice over (issue #544). `HOME` is not a Windows variable, so
//! on `windows-latest` every one of those paths resolved under a literal
//! `/home/jon`, nothing loaded, and the tests took their "data absent" skip
//! path and passed while reading nothing. `scripts/fetch_test_fixtures.py`
//! *does* run on that runner, so the fixtures were downloaded and then never
//! opened.
//!
//! The fallback is the other half. A hardcoded developer path as a silent
//! default turns "this machine has no cache" into "this machine has a cache
//! somewhere else", which is why the failure was quiet enough to survive.
//! [`root`] panics instead: a test that cannot work out where the fixtures
//! would be has learned nothing about whether they are there.

use std::path::PathBuf;

/// Library whose fixtures the suites read. `scripts/fetch_test_fixtures.py`
/// downloads this one, and the cache entry names carry it.
pub const LIBRARY: &str = "endf-b8.1";

/// Root of the fixture cache, as [`yamc_nuclide::url_cache::cache_root`]
/// resolves it: `YAMC_CACHE_DIR` when set, else `<home>/.cache/yamc`.
///
/// Panics when neither resolves. That is a broken environment rather than an
/// empty cache, and the two must not look alike: skipping on it is what let
/// #544 hide for as long as it did.
pub fn root() -> PathBuf {
    yamc_nuclide::url_cache::cache_root().expect(
        "no nuclear-data cache location: set YAMC_CACHE_DIR, or HOME (USERPROFILE on Windows)",
    )
}

/// Path a fixture for `nuclide` would occupy, whether or not it is there.
///
/// Returned as a `String` because that is what the load entry points take.
pub fn nuclide_path(nuclide: &str) -> String {
    yamc_nuclide::url_cache::cached_entry_path(LIBRARY, nuclide)
        .expect(
            "no nuclear-data cache location: set YAMC_CACHE_DIR, or HOME (USERPROFILE on Windows)",
        )
        .to_string_lossy()
        .into_owned()
}

/// The fixture for `nuclide`, or `None` when this machine does not carry it.
///
/// Absence is the only thing this reports. Where the cache *is* has already
/// been resolved by then, so a `None` here means the fixture was not fetched,
/// not that the lookup went somewhere wrong.
///
/// A cache entry is a DIRECTORY of sections, so that is what is checked. A
/// regular file at the same path is not one, and admitting it would hand the
/// loader something it fails on at the first section read, turning a skip into
/// a panic.
pub fn nuclide(nuclide: &str) -> Option<String> {
    let path = nuclide_path(nuclide);
    std::path::Path::new(&path).is_dir().then_some(path)
}

/// Whether the fixture for `nuclide` is cached, as a directory of sections.
pub fn have(nuclide: &str) -> bool {
    std::path::Path::new(&nuclide_path(nuclide)).is_dir()
}

/// The photoatomic fixture for `element`, or `None` when it is not cached.
///
/// Same layout and the same naming rule as a nuclide: an element is cached as
/// `<library>-Fe.arrow`. Named separately because the two are different
/// fixtures with different sections inside, and a caller asking for one and
/// getting the other would find out at the first section read.
pub fn element(element: &str) -> Option<String> {
    nuclide(element)
}

/// Root of the raw ENDF/B-VIII.1 evaluation tree the NJOY-backed tests read.
///
/// `YAMC_ENDF_DIR` when set, else `<home>/nuclear_data/endfb-viii.1-endf`. A
/// different thing from the fixture cache and deliberately kept apart from it:
/// these are the multi-gigabyte source evaluations, no CI job fetches them, and
/// the tests that want them need NJOY as well. They skip everywhere but a
/// machine that has both.
///
/// Here rather than spelled out per test because the home half is the same rule
/// (issue #544): `std::env::var("HOME")` resolves to nothing on Windows, and
/// four sites in `yamc-convert` built a path under an empty string when it did.
pub fn endf_evaluations() -> PathBuf {
    if let Some(dir) = std::env::var_os("YAMC_ENDF_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    yamc_nuclide::url_cache::home_dir()
        .expect("no home directory: set YAMC_ENDF_DIR, or HOME (USERPROFILE on Windows)")
        .join("nuclear_data")
        .join("endfb-viii.1-endf")
}
