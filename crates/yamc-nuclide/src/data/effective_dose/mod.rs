//! Effective dose conversion coefficients from ICRP publications.
//!
//! Provides fluence-to-effective-dose conversion factors for neutrons and
//! photons based on ICRP Publication 74 and 116.

mod dose;

pub use dose::{dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle};
