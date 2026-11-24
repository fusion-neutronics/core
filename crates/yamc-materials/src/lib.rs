//! Material composition and macroscopic cross-section data for yamc.
//!
//! Extracted from `yamc` so downstream tools (model builders, transmutation
//! drivers, the tallies crate) can pull just material data without
//! dragging in the transport engine. yamc itself depends on this crate
//! and re-exports `Material` under its original path.

pub mod collections;
pub mod majorant;
pub mod material;

pub use majorant::{GlobalMajorant, GlobalPhotonMajorant, LocalMajorant, Majorant};
pub use material::{
    DensityUnits, FractionType, MacroPhotonXS, Material, MaterialFastXS, UrrMacroXs,
};

pub(crate) use yamc_nuclide::interpolate_linear;
