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

/// Whether this is a CI runner rather than someone's machine.
///
/// For the tests that produce no assertion, only output for a person to read:
/// the CPU-versus-GPU comparison matrix and the photon spectrum diagnostics.
/// Those cost real time (the whole CPU side of every mode, in the matrix's
/// case) and yield nothing a runner can check, so they skip here and run at
/// home.
///
/// They used to carry `#[ignore]`, which got the default backwards. `#[ignore]`
/// means a developer sees them silently not run and has to know to pass
/// `-- --ignored`, which is exactly the special step nobody should need in
/// order to run everything locally. Skipping on CI instead means the local
/// command is just `cargo test`.
///
/// This is NOT how a test decides it has no GPU. That question is answered by
/// asking for one (`GpuContext::new().is_err()`), which is right on a CI runner
/// and on a laptop alike, and would still be right if a GPU runner ever
/// appeared. Environment detection beats environment guessing, and this
/// function exists only for the one question the environment cannot answer:
/// whether a human is going to read the output.
///
/// Both variables, because `GITHUB_ACTIONS` is specific and `CI` is set by
/// essentially every runner. Matching more than GitHub is deliberate.
///
/// Presence is not enough, which is the trap: `CI=false` and `CI=0` are things
/// people set ON PURPOSE to force the non-CI path, and an `is_some()` check
/// reads them as "yes, CI" and does the opposite of what was asked. So the
/// value is inspected, the way `YAMC_REQUIRE_FIXTURES` inspects its own.
pub fn on_ci() -> bool {
    let set = |name: &str| {
        std::env::var(name)
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v.is_empty() || v == "false" || v == "0")
            })
            .unwrap_or(false)
    };
    set("CI") || set("GITHUB_ACTIONS")
}

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
///
/// A directory stamped with a `format_version` this build does not read is
/// rejected for the same reason, and it is not hypothetical: after the
/// 2026-09-18 republish moved the libraries to format 2, any cache entry the
/// fetch script does not refresh is left at format 1, the loader refuses it
/// with "Unsupported Arrow format version: 1 (this build reads 2)", and a test
/// written to skip on an absent nuclide panics instead. Three files did on a
/// developer box carrying a pre-republish U235. Being unable to read a fixture
/// is what the caller means by "this machine cannot run that case", however the
/// directory came to be unreadable.
///
/// Absence is still the only thing reported to CI as a skip, because
/// `YAMC_REQUIRE_FIXTURES=1` turns the `None` into a failure there
/// (`crates/yamc/tests/fixture_cache_is_readable.rs`), so a stale fixture is
/// loud on a runner and quiet on a laptop.
pub fn nuclide(nuclide: &str) -> Option<String> {
    let path = nuclide_path(nuclide);
    let dir = std::path::Path::new(&path);
    (dir.is_dir() && format_version_is_readable(dir)).then_some(path)
}

/// Whether the NUCLIDE cache directory at `dir` carries a `format_version` the
/// nuclide loader reads. Elements are not checked this way, see [`element`].
///
/// True when there is no `version.json` at all, matching the loader: it only
/// applies the check when the marker is present, so a directory without one is
/// not rejected here either. An unreadable or malformed marker is treated the
/// same way, since the loader will not reject it on that basis and this is not
/// the place to invent a stricter rule.
pub fn format_version_is_readable(dir: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("version.json")) else {
        return true;
    };
    let Ok(marker) = serde_json::from_str::<serde_json::Value>(&text) else {
        return true;
    };
    let Some(found) = marker.get("format_version").and_then(|v| v.as_i64()) else {
        return true;
    };
    found == yamc_nuclide::arrow::nuclide_arrow::FORMAT_VERSION
}

/// Whether the fixture for `nuclide` is cached, as a directory of sections.
pub fn have(nuclide: &str) -> bool {
    std::path::Path::new(&nuclide_path(nuclide)).is_dir()
}

/// The fixture for `nuclide` when it carries the TRANSPORT sections, or `None`.
///
/// What a test that samples a secondary needs, and a stricter question than
/// [`nuclide`]. Since #389 a cache directory is routinely left at activation
/// scope, holding `reactions.arrow` and none of the products or distributions:
/// the directory is there, `nuclide` says yes, and the loader does not object
/// either, because `narrow_to_present_sections` narrows a `Full` request to
/// what is on disk rather than failing it. The first `(n,2n)` then panics with
/// "Missing product distributions", which is how a windows CI runner whose
/// restored fixture cache carried an activation-scope Ar38 failed
/// `parity_ar38_collision0_spectrum` while every other runner skipped it.
///
/// So the load is the check, and what it loaded is the answer: a nuclide whose
/// own `load_scope` came back without the transport sections is reported
/// absent, which is what the caller means by "this machine cannot run that
/// case".
pub fn transport_nuclide(name: &str) -> Option<String> {
    let path = nuclide(name)?;
    transport_ready(std::path::Path::new(&path)).then_some(path)
}

/// Whether the cache directory at `path` loads WITH its transport sections.
///
/// The path-taking half of [`transport_nuclide`], so the behaviour can be
/// tested on a directory built for the purpose rather than on whatever this
/// machine's cache happens to hold.
pub fn transport_ready(path: &std::path::Path) -> bool {
    yamc_nuclide::arrow::nuclide_arrow::read_nuclide_from_arrow(
        path,
        &yamc_nuclide::LoadScope::full(),
    )
    .is_ok_and(|n| n.load_scope.wants_transport_sections())
}

/// The photoatomic fixture for `element`, or `None` when it is not cached.
///
/// Same layout and the same naming rule as a nuclide: an element is cached as
/// `<library>-Fe.arrow`. Named separately because the two are different
/// fixtures with different sections inside, and a caller asking for one and
/// getting the other would find out at the first section read.
///
/// Presence only, deliberately NOT the `format_version` gate [`nuclide`]
/// applies. That gate mirrors the nuclide loader, and the element loader has
/// no counterpart: format 2 moved the neutron energy grids into `energy.arrow`
/// and split `reactions.arrow` per (MT, temperature), and a photon directory
/// (`element.arrow`, `subshells.arrow`, `compton.arrow`, `bremsstrahlung.arrow`)
/// carries none of that. The converter stamps every directory it writes, so a
/// pre-republish element is stamped 1 and reads exactly as one stamped 2 does;
/// routing it through the nuclide check reported the W the photon Z sweep had
/// just run on as absent.
pub fn element(element: &str) -> Option<String> {
    let path = nuclide_path(element);
    std::path::Path::new(&path).is_dir().then_some(path)
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
