//! Storage abstraction, caching, and the nuclide-load entry point.
//!
//! `backend` is the pluggable `Storage` trait (native `std::fs` by default; a
//! browser host can swap in OPFS-backed reads via `set_storage`); `in_memory_storage`
//! is the RAM-backed impl; `url_cache` resolves data keywords and caches downloads;
//! `nuclide_loader` is the public load-a-nuclide entry point. The `backend` trait
//! surface is glob-re-exported here so `crate::storage::{Storage, set_storage, ...}`
//! and `yamc_nuclide::storage::*` keep resolving unchanged.

mod backend;
pub use backend::*;

pub mod in_memory_storage;
pub mod nuclide_loader;
pub mod url_cache;
