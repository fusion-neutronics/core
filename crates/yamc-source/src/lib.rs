//! Particle source specifications and the spatial / energy / angular
//! distribution samplers they compose from.
//!
//! Extracted from `yamc` so downstream tools (a slim WASM model viewer,
//! GPU front-ends, custom source generators) can pull just source
//! sampling without dragging in the transport engine, transmutation, MPI,
//! or tally machinery. yamc itself depends on this crate and re-exports
//! its types under their original paths.

pub mod distribution;
pub mod source;
pub mod tokamak;

pub use source::Source;
