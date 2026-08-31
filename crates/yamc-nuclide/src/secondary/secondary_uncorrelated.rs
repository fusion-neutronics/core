// Uncorrelated angle-energy distributions

// use serde::{Deserialize, Serialize};
use crate::reaction_product::{AngleDistribution, EnergyDistribution};
use rand::Rng;

/// Sample from uncorrelated angle-energy distribution
/// Implements UncorrelatedAngleEnergy sampling
///
/// The angle and energy are sampled independently:
/// - If angle distribution exists, sample mu from it
/// - Otherwise use isotropic scattering
/// - If energy distribution exists, sample E_out from it
/// - If energy distribution is missing, panic
pub fn sample_uncorrelated<R: Rng>(
    incoming_energy: f64,
    angle: &AngleDistribution,
    energy: &Option<EnergyDistribution>,
    rng: &mut R,
) -> (f64, f64) {
    // Sample cosine of scattering angle
    let mu = angle.sample(incoming_energy, rng);

    // Sample outgoing energy if distribution exists
    let Some(energy_dist) = energy.as_ref() else {
        panic!("Missing energy distribution for uncorrelated angle-energy sampling");
    };
    let e_out = energy_dist.sample(incoming_energy, rng);

    (e_out, mu)
}

/// Sample from uncorrelated angle-energy for elastic scattering
/// Applies two-body kinematics to compute outgoing energy from angle
pub fn sample_elastic_uncorrelated<R: Rng>(
    incoming_energy: f64,
    angle: &AngleDistribution,
    awr: f64, // atomic weight ratio (target mass / neutron mass)
    rng: &mut R,
) -> (f64, f64) {
    // Sample cosine of scattering angle in center-of-mass frame
    let mu_cm = angle.sample(incoming_energy, rng);

    // Two-body elastic kinematics (target at rest approximation)
    // In CM frame: E_out/E_in = (1 + alpha + (1 - alpha)*mu_cm) / 2
    // where alpha = ((A-1)/(A+1))^2
    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    let e_out = incoming_energy * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;

    // Convert mu from CM to lab frame
    // mu_lab = (1 + A*mu_cm) / sqrt(1 + A^2 + 2*A*mu_cm)
    let denom = (1.0 + awr * awr + 2.0 * awr * mu_cm).sqrt();
    let mu_lab = (1.0 + awr * mu_cm) / denom;

    (e_out, mu_lab)
}
