//! Evaluated photon nuclear data: `Element`, photon interaction
//! cross-sections (Compton, photoelectric, pair production, …),
//! atomic transition / electron-shell data, and the Arrow loader
//! that reads them off disk.
//!
//! Extracted from `yamc` so downstream tools -- the GPU integration
//! (photon transport kernel), cross-section plotters, data
//! converters -- can pull just photon data without dragging in the
//! transport engine. yamc itself depends on this crate and
//! re-exports its types under their original paths.
//!
//! Companion crate: `yamc-nuclide` (neutron-side data, on which
//! this crate depends -- `Tabulated1D` lives there and is the basic
//! interpolation primitive used by both neutron and photon XS).
//!
//! ## What's NOT here
//!
//! Photon *transport-coupling* code (bremsstrahlung sampling,
//! photon production from neutron reactions, decay-photon
//! production, D1S) lives in `yamc-physics` because those functions
//! need `ParticleBank` and the transport engine. This crate is data +
//! data-side helpers only. [`bremsstrahlung`] is the dividing line:
//! the TTB tables are built here from the per-element Arrow data,
//! and sampled from there.

pub mod bremsstrahlung;
pub mod element;
pub mod photon;
#[cfg(feature = "arrow")]
pub mod photon_arrow;
pub mod photon_log;

pub use element::Element;
