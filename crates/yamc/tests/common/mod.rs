//! Shared test helpers for the integration tests.
//!
//! Each integration-test file in `tests/` is its own crate, so anything
//! used from more than one test belongs here. Bench code under
//! `benches/` cannot share this module (separate compilation unit), so
//! the same builders are duplicated there.

// Each test file uses only part of this module; the rest is dead there.
#[allow(dead_code)]
pub mod cell_geometries;
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
pub mod gpu_twin;
