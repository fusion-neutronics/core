use crate::distribution::energy::{Discrete, Histogram, Normal, Uniform};
use crate::distribution::spatial::{CylindricalRing, Point};
use serde::{Deserialize, Serialize};
use yamc_particle::ParticleType;

/// Energy distribution for sources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceEnergyDistribution {
    /// Discrete energy distribution with specific energies and probabilities
    Discrete(Discrete),
    /// Histogram (piecewise-constant) energy distribution over energy bins
    Histogram(Histogram),
    /// Uniform energy distribution with lower and upper bounds
    Uniform(Uniform),
    /// Normal (Gaussian) energy distribution with mean and standard deviation
    Normal(Normal),
}

/// Spatial distribution for sources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceSpatialDistribution {
    /// Point spatial distribution - fixed position in space
    Point(Point),
    /// Cylindrical independent distribution
    CylindricalRing(Box<CylindricalRing>),
}

impl SourceEnergyDistribution {
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            SourceEnergyDistribution::Discrete(d) => d.sample(rng),
            SourceEnergyDistribution::Histogram(h) => h.sample(rng),
            SourceEnergyDistribution::Uniform(u) => u.sample(rng),
            SourceEnergyDistribution::Normal(n) => n.sample(rng),
        }
    }
}

impl SourceSpatialDistribution {
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> [f64; 3] {
        match self {
            SourceSpatialDistribution::Point(p) => p.sample(rng),
            SourceSpatialDistribution::CylindricalRing(ref c) => c.sample(rng),
        }
    }
}

/// Source configuration: spatial, angular, and energy distributions plus strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub space: SourceSpatialDistribution,
    pub angle: crate::distribution::angular::AngularDistribution,
    pub energy: SourceEnergyDistribution,
    pub strength: f64,
}

impl Default for Source {
    fn default() -> Self {
        Self::new()
    }
}

impl Source {
    /// Create a source with the default neutron birth energy (14.06 MeV,
    /// D-T fusion). This default is only physically meaningful for neutrons;
    /// see [`Source::with_default_energy`] for the per-particle-type contract.
    pub fn new() -> Self {
        Self {
            space: SourceSpatialDistribution::Point(Point::default()),
            angle: crate::distribution::angular::AngularDistribution::Isotropic,
            energy: SourceEnergyDistribution::Discrete(
                Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
            ),
            strength: 1.0,
        }
    }

    /// Build a source for `particle_type`, applying a particle-appropriate
    /// energy default when `energy` is `None`.
    ///
    /// - Neutrons default to 14.06 MeV (D-T fusion birth energy) when no
    ///   energy is supplied, matching [`Source::new`].
    /// - Photons have no single canonical birth energy (decay gammas,
    ///   bremsstrahlung and characteristic X-rays span keV to MeV), so an
    ///   explicit energy is **required**: omitting it returns an error rather
    ///   than silently emitting a 14 MeV gamma.
    ///
    /// Spatial, angular and strength fields take the same defaults as
    /// [`Source::new`] and can be overridden after construction.
    pub fn with_default_energy(
        particle_type: ParticleType,
        energy: Option<SourceEnergyDistribution>,
    ) -> Result<Self, String> {
        let energy = match (particle_type, energy) {
            (_, Some(e)) => e,
            (ParticleType::Neutron, None) => {
                SourceEnergyDistribution::Discrete(Discrete::new(vec![14.06e6], vec![1.0]).unwrap())
            }
            (ParticleType::Photon, None) => {
                return Err(
                    "PhotonSource requires an explicit energy: photons have no default \
                     birth energy (pass e.g. energy=1e6 or yamc.Discrete([1e6], [1.0]))"
                        .to_string(),
                );
            }
        };
        Ok(Self {
            space: SourceSpatialDistribution::Point(Point::default()),
            angle: crate::distribution::angular::AngularDistribution::Isotropic,
            energy,
            strength: 1.0,
        })
    }
}

/// Cumulative strengths of a source list, so picking a source is a binary
/// search rather than a walk.
///
/// A parametric plasma source is thousands of ring sources (one per mesh
/// voxel per reaction), and a linear walk over them on every history costs
/// as much as transporting it. Build one of these once per run and hand it
/// to the sampler.
#[derive(Debug, Clone)]
pub struct SourceSelector {
    cumulative: Vec<f64>,
}

impl SourceSelector {
    /// Accumulate the strengths of `sources`, in list order.
    pub fn new(sources: &[ParticleSource]) -> Self {
        let mut cumulative = Vec::with_capacity(sources.len());
        let mut running = 0.0;
        for source in sources {
            running += source.strength();
            cumulative.push(running);
        }
        Self { cumulative }
    }

    /// Total strength of the source list.
    pub fn total(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or(0.0)
    }

    /// Index of the first source whose cumulative strength reaches
    /// `threshold`, which the caller draws uniformly from `[0, total)`. A
    /// threshold past the total (only reachable by rounding) selects the last
    /// source.
    pub fn index_for(&self, threshold: f64) -> usize {
        let index = self.cumulative.partition_point(|&value| value < threshold);
        index.min(self.cumulative.len().saturating_sub(1))
    }
}

/// A particle source tagged with its particle type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParticleSource {
    Neutron(Source),
    Photon(Source),
}

impl ParticleSource {
    pub fn particle_type(&self) -> ParticleType {
        match self {
            ParticleSource::Neutron(_) => ParticleType::Neutron,
            ParticleSource::Photon(_) => ParticleType::Photon,
        }
    }

    pub fn source(&self) -> &Source {
        match self {
            ParticleSource::Neutron(s) | ParticleSource::Photon(s) => s,
        }
    }

    pub fn source_mut(&mut self) -> &mut Source {
        match self {
            ParticleSource::Neutron(s) | ParticleSource::Photon(s) => s,
        }
    }

    pub fn strength(&self) -> f64 {
        self.source().strength
    }

    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> yamc_particle::Particle {
        let src = self.source();
        let sampled_space = src.space.sample(rng);
        let sampled_angle = src.angle.sample(rng);
        let sampled_energy = src.energy.sample(rng);
        let mut particle =
            yamc_particle::Particle::new(sampled_space, sampled_angle, sampled_energy);
        particle.particle_type = self.particle_type();
        particle
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn discrete(energy: f64) -> SourceEnergyDistribution {
        SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap())
    }

    #[test]
    fn neutron_default_energy_is_dt_birth_energy() {
        // Omitting the energy for a neutron keeps the 14.06 MeV D-T default.
        let src = Source::with_default_energy(ParticleType::Neutron, None).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(src.energy.sample(&mut rng), 14.06e6);
        // ... and matches the unchanged `Source::new` default.
        assert_eq!(src.energy, Source::new().energy);
    }

    #[test]
    fn photon_without_energy_is_rejected() {
        // A photon has no canonical birth energy: fail fast rather than
        // silently emitting the 14 MeV D-T neutron energy as a gamma.
        let err = Source::with_default_energy(ParticleType::Photon, None).unwrap_err();
        assert!(
            err.contains("PhotonSource requires an explicit energy"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn photon_with_explicit_energy_is_honoured() {
        let src = Source::with_default_energy(ParticleType::Photon, Some(discrete(1.0e6))).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(src.energy.sample(&mut rng), 1.0e6);
        // The 14.06 MeV neutron default must never leak into photon sources.
        assert_ne!(src.energy.sample(&mut rng), 14.06e6);
    }

    #[test]
    fn explicit_energy_overrides_neutron_default() {
        let src =
            Source::with_default_energy(ParticleType::Neutron, Some(discrete(2.0e6))).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(src.energy.sample(&mut rng), 2.0e6);
    }
}
