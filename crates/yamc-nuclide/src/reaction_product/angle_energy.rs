//! Combined angle-energy distributions (uncorrelated, Kalbach-Mann, correlated, evaporation).

use super::*;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Angle-energy distribution types for secondary particle emission
/// Variants correspond to different ENDF representations:
/// - UncorrelatedAngleEnergy: independent angle and energy sampling
/// - KalbachMann: Kalbach-Mann systematics (Kalbach, Phys. Rev. C 37, 2350, 1988)
/// - CorrelatedAngleEnergy: fully correlated angle-energy tables
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AngleEnergyDistribution {
    UncorrelatedAngleEnergy {
        angle: AngleDistribution,
        energy: Option<EnergyDistribution>,
    },
    KalbachMann {
        #[serde(flatten)]
        kalbach: crate::secondary_kalbach::KalbachMann,
    },
    CorrelatedAngleEnergy {
        #[serde(flatten)]
        correlated: crate::secondary_correlated::CorrelatedAngleEnergy,
    },
    /// Evaporation energy distribution (ENDF File 5, Law 9)
    #[serde(rename = "Evaporation")]
    Evaporation {
        theta: Option<Tabulated1D>, // nuclear temperature parameter as function of E
        u: f64,                     // restriction energy (threshold)
    },
    /// N-body phase space distribution (ENDF File 6, Law 6)
    /// Used for reactions like (n,2n) where multiple particles share the available energy
    #[serde(rename = "NBodyPhaseSpace")]
    NBodyPhaseSpace {
        n_bodies: i32,   // Number of particles in final state (e.g., 3 for n,2n: 2n + residual)
        total_mass: f64, // Total mass ratio of all products (sum of AWRs)
        awr: f64,        // Atomic weight ratio of target
        q_value: f64,    // Q-value of reaction in eV
    },
}

impl AngleEnergyDistribution {
    /// Sample outgoing energy and scattering cosine
    /// Delegates to the appropriate secondary distribution sampler
    pub fn sample<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> (f64, f64) {
        match self {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { angle, energy } => {
                // Energy distribution must exist when sample() is called
                // For elastic scattering, use sample_elastic_mu_cm() instead
                crate::secondary_uncorrelated::sample_uncorrelated(
                    incoming_energy,
                    angle,
                    energy,
                    rng,
                )
            }
            AngleEnergyDistribution::KalbachMann { kalbach } => {
                kalbach.sample(incoming_energy, rng)
            }
            AngleEnergyDistribution::CorrelatedAngleEnergy { correlated } => {
                correlated.sample(incoming_energy, rng)
            }
            AngleEnergyDistribution::Evaporation { theta, u } => {
                // Sample outgoing energy from evaporation spectrum
                // p(E) ~ E * exp(-E/theta), 0 < E_out < E_in - u
                // Rejection method with two random numbers
                let e_in = incoming_energy;
                let theta_val = if let Some(tab) = theta {
                    tab.evaluate(e_in)
                } else {
                    1.0 // fallback, should not happen
                };
                let y = (e_in - *u) / theta_val;
                let v = 1.0 - (-y).exp();
                let e_out = loop {
                    let xi1 = rng.random::<f64>();
                    let xi2 = rng.random::<f64>();
                    let x = -((1.0 - v * xi1) * (1.0 - v * xi2)).ln();
                    if x <= y {
                        break x * theta_val;
                    }
                };
                // Isotropic emission
                let mu = 2.0 * rng.random::<f64>() - 1.0;
                (e_out, mu)
            }
            AngleEnergyDistribution::NBodyPhaseSpace {
                n_bodies,
                total_mass,
                awr,
                q_value,
            } => {
                // N-body phase space sampling
                // Angle is isotropic for N-body phase space
                let mu = 2.0 * rng.random::<f64>() - 1.0;

                // Calculate maximum available energy for this particle
                // E_max = (Ap - 1)/Ap * (A/(A+1) * E_in + Q)
                let ap = *total_mass;
                let a = *awr;
                let q = *q_value;
                let e_max = (ap - 1.0) / ap * (a / (a + 1.0) * incoming_energy + q);

                // Sample from N-body phase space distribution
                // x and y are Maxwellian-like samples
                let x = sample_maxwell(rng);
                let y = match *n_bodies {
                    3 => sample_maxwell(rng),
                    4 => {
                        let r1: f64 = rng.random();
                        let r2: f64 = rng.random();
                        let r3: f64 = rng.random();
                        -(r1 * r2 * r3).ln()
                    }
                    5 => {
                        let r1: f64 = rng.random();
                        let r2: f64 = rng.random();
                        let r3: f64 = rng.random();
                        let r4: f64 = rng.random();
                        let r5: f64 = rng.random();
                        let r6: f64 = rng.random();
                        -(r1 * r2 * r3 * r4).ln()
                            - r5.ln() * (std::f64::consts::FRAC_PI_2 * r6).cos().powi(2)
                    }
                    _ => panic!("N-body phase space with >5 bodies."),
                };

                // Energy fraction
                let v = x / (x + y);
                let e_out = e_max * v;

                (e_out, mu)
            }
        }
    }

    /// Get the name of this angle-energy distribution type
    pub fn distribution_name(&self) -> &'static str {
        match self {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { .. } => "UncorrelatedAngleEnergy",
            AngleEnergyDistribution::KalbachMann { .. } => "KalbachMann",
            AngleEnergyDistribution::CorrelatedAngleEnergy { .. } => "CorrelatedAngleEnergy",
            AngleEnergyDistribution::Evaporation { .. } => "Evaporation",
            AngleEnergyDistribution::NBodyPhaseSpace { .. } => "NBodyPhaseSpace",
        }
    }

    /// Get the energy distribution name if using UncorrelatedAngleEnergy
    pub fn energy_distribution_name(&self) -> Option<&'static str> {
        match self {
            AngleEnergyDistribution::UncorrelatedAngleEnergy { energy, .. } => {
                energy.as_ref().map(|e| e.distribution_name())
            }
            _ => None,
        }
    }
}

/// Sample from the unit-temperature Maxwell distribution (`p(x) ~ x·exp(-x)`).
///
/// Delegates to the shared [`crate::sampling::sample_maxwell_spectrum`] with
/// `theta = 1`, which is bit-identical: `-1·(ln r1 + ln r2·cos²(π/2·r3))`
/// equals `-ln r1 - ln r2·cos²(π/2·r3)` (IEEE negation is exact and rounding
/// is symmetric; `powi(2)` is `c·c`).
fn sample_maxwell<R: Rng>(rng: &mut R) -> f64 {
    crate::sampling::sample_maxwell_spectrum(1.0, rng)
}
