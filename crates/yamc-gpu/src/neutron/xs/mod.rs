//! Extract GPU-ready cross-section arrays from yamc's `Nuclide`.
//!
//! Bridges the rich CPU-side nuclide data structure (per-temperature
//! reactions, threshold-truncated grids, MT enums, …) into the flat
//! arrays the cubecl transport kernel takes. Split by concern:
//!
//! - [`constants`] -- MT slot tables and per-distribution dimension caps.
//! - [`types`] -- the `GpuNuclideXs` container and `NuclideXsError`.
//! - [`distributions`] -- flat per-MT buffer builders (angle, eout,
//!   correlated, Kalbach-Mann, evaporation, n-body, Maxwell, Watt, URR).
//!   The per-law single-reaction extraction they call lives in
//!   `yamc_physics::gpu::flat::eout_extract` (issue #111) and is
//!   re-exported from `distributions` at its old paths.
//! - [`extract`] -- the public `extract_*` entry points that assemble the
//!   above into `GpuNuclideXs` / per-MT score buffers.
//! - [`photon_production`] -- host-only secondary photon-production
//!   product-selection table (`GpuPhotonProductionXs`), packing the per-product
//!   CPU Pass-1 photon weights for the coupled neutron->photon GPU walk.
//!
//! Pure CPU data prep -- always built, no cubecl dependency.

pub mod constants;
pub mod distributions;
pub mod extract;
pub mod photon_production;
pub mod types;

pub use constants::*;
pub use extract::*;
pub use photon_production::*;
pub use types::*;

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests;
