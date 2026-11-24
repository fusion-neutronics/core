//! Dose conversion coefficients from ICRP publications.
//!
//! Provides fluence-to-effective-dose conversion factors for neutrons and
//! photons based on ICRP Publication 74 and 116, and fluence-to-ambient-dose-
//! equivalent H*(10) factors based on ICRP Publication 74.

mod dose;

pub use dose::{ambient_dose_coefficients, dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle};
