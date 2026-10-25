//! Storage abstraction, caching, and the nuclide-load entry point.
//!
//! `backend` is the pluggable `Storage` trait (native `std::fs` by default; a
//! browser host can swap in OPFS-backed reads via `set_storage`); `in_memory_storage`
//! is the RAM-backed impl; `release` is the published release layout (pointer,
//! manifest, supported formats, sha256 checks); `local_release` reads release
//! folders on local disk; `url_cache` resolves data keywords and caches downloads;
//! manifest, supported formats, sha256 checks); `url_cache` resolves data keywords and caches downloads;
//! `fetch` downloads one file pinned by its size and sha256 (a local library build's sources);
//! `nuclide_loader` is the public load-a-nuclide entry point. The `backend` trait
//! surface is glob-re-exported here so `crate::storage::{Storage, set_storage, ...}`
//! and `yamc_nuclide::storage::*` keep resolving unchanged.

mod backend;
pub use backend::*;

// Native only: it streams over reqwest, which the emscripten build does not
// link, and nothing in a browser builds a library.
#[cfg(all(feature = "download", not(target_os = "emscripten")))]
pub mod fetch;
pub mod in_memory_storage;
mod local_release;
pub mod nuclide_loader;
pub mod release;
#[cfg(feature = "download")]
mod release_cache;
pub mod url_cache;
