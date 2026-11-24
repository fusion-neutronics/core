//! Reaction-product types and their secondary-particle distributions.
//!
//! The distribution kinds live in submodules (tabulated / prob_distribution /
//! angle / energy / angle_energy / yield_mode); this module re-exports them all so
//! the historic flat `crate::reaction_product::<Type>` paths stay stable, and owns
//! the top-level [`ReactionProduct`] container that ties a distribution to a yield.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

pub mod angle;
pub mod angle_energy;
pub mod energy;
pub mod prob_distribution;
pub mod tabulated;
pub mod yield_mode;

pub use angle::{AngleDistribution, ElasticAngleFlat, ElasticFlatCache, InelasticAngleFlatCache};
pub use angle_energy::AngleEnergyDistribution;
pub use energy::{
    weighted_energy_mixture, EnergyDistribution, FissionChiFlat, FissionChiFlatCache,
};
pub use prob_distribution::TabulatedProbability;
pub use tabulated::{Tabulated, Tabulated1D, TabulatedInterp};
pub use yield_mode::Yield;

// Re-export ParticleType (canonical definition lives in this crate's
// `particle_type` module).
pub use crate::particle_type::ParticleType;

/// Reaction product with distributions
/// Corresponds to the ReactionProduct class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionProduct {
    pub particle: ParticleType,
    pub emission_mode: String,
    pub decay_rate: f64,
    pub applicability: Vec<Tabulated1D>,
    pub distribution: Vec<AngleEnergyDistribution>,
    #[serde(rename = "yield", default)]
    pub product_yield: Option<Yield>,
}

impl ReactionProduct {
    /// Sample an outgoing particle from this product
    pub fn sample<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> (f64, f64) {
        if self.distribution.is_empty() {
            return (incoming_energy, 2.0 * rng.random::<f64>() - 1.0);
        }

        let distribution_index = if self.distribution.len() == 1 {
            0
        } else {
            self.sample_distribution_index(incoming_energy, rng)
        };

        self.distribution[distribution_index].sample(incoming_energy, rng)
    }

    /// Sample which distribution to use based on applicability
    /// - Uses cumulative probability WITHOUT normalization
    /// - Applicability functions are assumed to sum to 1.0
    fn sample_distribution_index<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> usize {
        if self.applicability.is_empty() || self.applicability.len() != self.distribution.len() {
            return 0;
        }

        let c = rng.random::<f64>();
        let mut prob = 0.0;

        // Walk cumulative applicability until we exceed the sampled value
        for (i, app) in self.applicability.iter().enumerate() {
            prob += app.evaluate(incoming_energy);
            if c <= prob {
                return i;
            }
        }

        // If we get here (due to floating point issues), return the last distribution
        self.distribution.len() - 1
    }

    pub fn sample_multiple<R: Rng>(&self, incoming_energy: f64, rng: &mut R) -> Vec<(f64, f64)> {
        self.distribution
            .iter()
            .map(|dist| dist.sample(incoming_energy, rng))
            .collect()
    }

    pub fn is_particle_type(&self, particle_type: &ParticleType) -> bool {
        self.particle == *particle_type
    }

    pub fn is_prompt(&self) -> bool {
        self.emission_mode == "prompt"
    }

    pub fn is_delayed(&self) -> bool {
        self.emission_mode == "delayed"
    }

    pub fn get_decay_rate(&self) -> f64 {
        self.decay_rate
    }
}

// =====================
//        TESTS
// =====================
