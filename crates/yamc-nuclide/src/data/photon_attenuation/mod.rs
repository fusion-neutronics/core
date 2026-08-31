//! Photon attenuation and energy-absorption coefficients.
//!
//! Two tabulations, both read with log-log interpolation: the mass attenuation
//! coefficient mu/rho of each element, which turns a material composition into
//! the linear attenuation coefficient the photons of a contact-dose estimate
//! are self-shielded by, and the mass energy-absorption coefficient mu_en/rho
//! of air, which turns the photons that get out into an absorbed dose.

mod coefficients;

pub use coefficients::{
    mass_attenuation_coefficient, mass_energy_absorption_air, CoefficientTable,
};
