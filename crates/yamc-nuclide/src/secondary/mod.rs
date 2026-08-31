//! Secondary-particle angle/energy distribution samplers.
//!
//! Correlated (Kalbach-Mann and fully tabular) and uncorrelated angle-energy
//! distributions, plus the pure Maxwell/Watt fission-spectrum samplers they
//! build on. Consumed by [`crate::reaction_product`] (which dispatches to
//! them) and the Arrow loader. Re-exported at the crate root so the historic
//! `yamc_nuclide::secondary_*` / `yamc_nuclide::sampling` paths stay stable.

pub mod sampling;
pub mod secondary_correlated;
pub mod secondary_kalbach;
pub mod secondary_uncorrelated;
