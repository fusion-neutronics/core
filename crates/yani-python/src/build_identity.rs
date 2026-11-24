//! Which build of core this extension is, as the data it writes records it.
//!
//! `build.rs` sets `YANI_BUILD_COMMIT` and `YANI_BUILD_DIRTY` at compile time
//! (exposed as `__commit__` and `__commit_dirty__`); every converter and
//! `generate_library` stamp what they write with them, so a folder can be
//! traced to the commit that produced it even when the wheel's version is the
//! `0.0.0.dev0` of a build from source.
//!
//! This crate is compiled into both wheels, so the stamp names the wheel doing
//! the writing: `register_classes` records it once, when the extension loads.

use std::sync::OnceLock;

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

/// The import name of the wheel this extension is (`"yamc"` or `"yani"`), set
/// by `register_classes`. Each wheel compiles its own copy of this crate, so
/// each extension holds its own value.
static WHEEL: OnceLock<String> = OnceLock::new();

/// Record which wheel this extension is. Called once, from `register_classes`.
pub fn set_wheel(package: &str) {
    let _ = WHEEL.set(package.to_string());
}

/// `"<wheel>-python <version>"`, such as `"yamc-python 0.24.0"`, with the
/// version of the installed `<wheel>-core` distribution, which the release tag
/// sets. `"unknown"` stands in for the version when the interpreter has no
/// metadata for it, rather than the crate's own `CARGO_PKG_VERSION`, which is
/// never bumped and would name a release that did not write the data.
pub fn package(py: Python<'_>) -> String {
    let wheel = WHEEL.get().map(String::as_str).unwrap_or("yani");
    let version = py
        .import("importlib.metadata")
        .and_then(|m| m.call_method1("version", (format!("{wheel}-core"),)))
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| "unknown".to_string());
    format!("{wheel}-python {version}")
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
