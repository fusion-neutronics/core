//! Which build of core this extension is, as the data it writes records it.
//!
//! `build.rs` sets `YANI_BUILD_COMMIT` and `YANI_BUILD_DIRTY` at compile time
//! (exposed as `__commit__` and `__commit_dirty__`); every converter and
//! `generate_library` stamp what they write with them, so a folder can be
//! traced to the commit that produced it even when the wheel's version is the
//! `0.0.0.dev0` of a build from source.

use nuclear_data_schema::BuildIdentity;
use pyo3::prelude::*;

/// The core commit this extension was compiled from, `None` outside a git
/// checkout of core.
pub const COMMIT: Option<&str> = option_env!("YANI_BUILD_COMMIT");

/// Whether a tracked file differed from [`COMMIT`] when this was compiled,
/// `None` when the commit is unknown.
pub fn dirty() -> Option<bool> {
    option_env!("YANI_BUILD_DIRTY").map(|dirty| dirty == "true")
}

/// `"yani-python <version>"`, with the installed yani-core wheel's version,
/// which the release tag sets. The crate's own `CARGO_PKG_VERSION` is not
/// bumped, so it is only the fallback for an interpreter that never saw a
/// wheel.
pub fn package(py: Python<'_>) -> String {
    let version = py
        .import("importlib.metadata")
        .and_then(|m| m.call_method1("version", ("yani-core",)))
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string());
    format!("yani-python {version}")
}

/// The identity of this build, borrowing `package` from [`package`].
pub fn identity(package: &str) -> BuildIdentity<'_> {
    BuildIdentity {
        package,
        commit: COMMIT,
        dirty: dirty(),
    }
}

/// The `converter` stamp the converters write into every folder.
pub fn converter(py: Python<'_>) -> String {
    identity(&package(py)).stamp()
}
