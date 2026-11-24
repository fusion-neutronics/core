use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rand::{rng, RngExt};
use yamc_nuclide::reaction_product::{
    AngleDistribution, AngleEnergyDistribution, ParticleType, ReactionProduct, Tabulated,
};

/// A product emitted by a nuclear reaction: particle type, emission mode,
/// decay rate, and the angle/energy distributions used to sample it.
#[gen_stub_pyclass]
#[pyclass(name = "ReactionProduct", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyReactionProduct {
    pub inner: ReactionProduct,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReactionProduct {
    fn __repr__(&self) -> String {
        let particle = match self.inner.particle {
            ParticleType::Neutron => "neutron",
            ParticleType::Photon => "photon",
        };
        format!(
            "ReactionProduct(particle='{}', emission_mode='{}', distributions={})",
            particle,
            self.inner.emission_mode,
            self.inner.distribution.len()
        )
    }

    #[new]
    fn new(particle: String, emission_mode: String, decay_rate: f64) -> Self {
        let particle_type = match particle.as_str() {
            "neutron" => ParticleType::Neutron,
            "photon" => ParticleType::Photon,
            _ => ParticleType::Neutron, // Default fallback
        };

        Self {
            inner: ReactionProduct {
                particle: particle_type,
                emission_mode,
                decay_rate,
                applicability: vec![],
                distribution: vec![],
                product_yield: None,
            },
        }
    }

    /// Sample an outgoing particle from this product
    /// Returns tuple of (outgoing_energy, mu_cosine)
    fn sample(&self, incoming_energy: f64) -> (f64, f64) {
        let mut rng = rng();
        self.inner.sample(incoming_energy, &mut rng)
    }

    /// Sample multiple outgoing particles
    /// Returns list of (outgoing_energy, mu_cosine) tuples
    fn sample_multiple(&self, incoming_energy: f64) -> Vec<(f64, f64)> {
        let mut rng = rng();
        self.inner.sample_multiple(incoming_energy, &mut rng)
    }

    /// Check if this product represents a specific particle type
    fn is_particle_type(&self, particle_name: &str) -> bool {
        let particle_type = match particle_name {
            "neutron" => ParticleType::Neutron,
            "photon" => ParticleType::Photon,
            _ => return false,
        };
        self.inner.is_particle_type(&particle_type)
    }

    /// Check if this is a prompt emission
    fn is_prompt(&self) -> bool {
        self.inner.is_prompt()
    }

    /// Check if this is a delayed emission
    fn is_delayed(&self) -> bool {
        self.inner.is_delayed()
    }

    /// Get the decay rate for delayed neutron precursors
    fn get_decay_rate(&self) -> f64 {
        self.inner.get_decay_rate()
    }

    /// Get particle type as string
    #[getter]
    fn particle(&self) -> String {
        match self.inner.particle {
            ParticleType::Neutron => "neutron".to_string(),
            ParticleType::Photon => "photon".to_string(),
        }
    }

    /// Get emission mode
    #[getter]
    fn emission_mode(&self) -> String {
        self.inner.emission_mode.clone()
    }

    /// Get number of distributions
    #[getter]
    fn num_distributions(&self) -> usize {
        self.inner.distribution.len()
    }

    /// Get distribution types as strings
    #[getter]
    fn distribution_types(&self) -> Vec<String> {
        self.inner
            .distribution
            .iter()
            .map(|dist| {
                match dist {
                yamc_nuclide::reaction_product::AngleEnergyDistribution::UncorrelatedAngleEnergy {
                    ..
                } => "UncorrelatedAngleEnergy".to_string(),
                yamc_nuclide::reaction_product::AngleEnergyDistribution::KalbachMann { .. } => {
                    "KalbachMann".to_string()
                }
                yamc_nuclide::reaction_product::AngleEnergyDistribution::CorrelatedAngleEnergy {
                    ..
                } => "CorrelatedAngleEnergy".to_string(),
                yamc_nuclide::reaction_product::AngleEnergyDistribution::Evaporation { .. } => {
                    "Evaporation".to_string()
                }
                yamc_nuclide::reaction_product::AngleEnergyDistribution::NBodyPhaseSpace { .. } => {
                    "NBodyPhaseSpace".to_string()
                }
            }
            })
            .collect()
    }

    /// Get angle distribution if available (for UncorrelatedAngleEnergy)
    /// Returns None if no angle distribution exists
    #[getter]
    fn angle_distribution(&self) -> Option<PyAngleDistribution> {
        if let Some(dist) = self.inner.distribution.first() {
            match dist {
                yamc_nuclide::reaction_product::AngleEnergyDistribution::UncorrelatedAngleEnergy {
                    angle,
                    ..
                } => Some(PyAngleDistribution {
                    inner: angle.clone(),
                }),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl PyReactionProduct {
    /// Create PyReactionProduct from a Rust ReactionProduct
    pub fn from_reaction_product(
        product: ReactionProduct,
        _py: pyo3::Python,
    ) -> pyo3::PyResult<Self> {
        Ok(PyReactionProduct { inner: product })
    }
}

/// Energy-dependent secondary-angle distribution; `sample(energy)` draws a
/// scattering cosine (mu) for a given incoming energy.
#[gen_stub_pyclass]
#[pyclass(name = "AngleDistribution", from_py_object)]
#[derive(Clone)]
pub struct PyAngleDistribution {
    pub inner: AngleDistribution,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAngleDistribution {
    fn __repr__(&self) -> String {
        format!(
            "AngleDistribution(energy_points={})",
            self.inner.energy.len()
        )
    }

    /// Sample scattering cosine (mu) for a given incoming energy
    fn sample(&self, incoming_energy: f64) -> f64 {
        let mut rng = rng();
        self.inner.sample(incoming_energy, &mut rng)
    }

    /// Get energy grid
    #[getter]
    fn energy(&self) -> Vec<f64> {
        self.inner.energy.clone()
    }

    /// Get number of energy points
    #[getter]
    fn num_energy_points(&self) -> usize {
        self.inner.energy.len()
    }
}

/// A tabulated probability distribution (`values` with their `probabilities`,
/// internally carrying a precomputed CDF) used for inverse-transform sampling.
#[gen_stub_pyclass]
#[pyclass(name = "Tabulated", from_py_object)]
#[derive(Clone)]
pub struct PyTabulated {
    pub inner: Tabulated,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTabulated {
    fn __repr__(&self) -> String {
        format!("Tabulated(points={})", self.inner.x.len())
    }

    #[new]
    fn new(values: Vec<f64>, probabilities: Vec<f64>) -> Self {
        // Check if the probabilities already look like a CDF (start near 0, end
        // near 1, monotonically increasing).
        let p = probabilities;
        let looks_like_cdf = !p.is_empty()
            && p[0].abs() < 0.01
            && (p[p.len() - 1] - 1.0).abs() < 0.01
            && p.windows(2).all(|w| w[1] >= w[0]);

        Self {
            inner: Tabulated {
                x: values,
                p: p.clone(),
                c: if looks_like_cdf { p } else { vec![] },
                interp: yamc_nuclide::reaction_product::TabulatedInterp::LinLin,
            },
        }
    }

    /// Sample from the tabulated distribution
    fn sample(&self) -> f64 {
        let mut rng = rng();
        self.inner.sample(&mut rng)
    }

    /// The tabulated x values (abscissae).
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    /// The probability density at each x value.
    #[getter]
    fn probabilities(&self) -> Vec<f64> {
        self.inner.p.clone()
    }
}

/// Sample a scattering cosine (mu in [-1, 1]) for isotropic scattering.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn sample_scatter_cosine() -> f64 {
    let mut rng = rng();
    2.0 * rng.random::<f64>() - 1.0
}

/// Test function to create a simple reaction product for testing
#[gen_stub_pyfunction]
#[pyfunction]
pub fn create_test_reaction_product() -> PyReactionProduct {
    // Create a test product with inelastic-like scattering (has energy distribution)
    // sample() requires an energy distribution

    // Create truly isotropic angular distribution (uniform in mu)
    let mu_dist = Tabulated {
        x: vec![-1.0, 1.0],
        p: vec![0.5, 0.5], // PDF for uniform distribution
        c: vec![0.0, 1.0], // CDF for uniform distribution
        interp: yamc_nuclide::reaction_product::TabulatedInterp::LinLin,
    };

    let angle_dist = AngleDistribution {
        energy: vec![1e5, 1e6, 1e7], // 0.1, 1, 10 MeV
        mu: vec![mu_dist.clone(), mu_dist.clone(), mu_dist.clone()],
    };

    // Create a simple LevelInelastic energy distribution
    // E_out = mass_ratio * (E_in - threshold)
    // With threshold=0 and mass_ratio=0.99, output is ~99% of input
    let energy_dist = yamc_nuclide::reaction_product::EnergyDistribution::LevelInelastic {
        threshold: 0.0,
        mass_ratio: 0.99, // Slight energy loss
    };

    // Create uncorrelated angle-energy distribution (with energy dist)
    let angle_energy_dist = AngleEnergyDistribution::UncorrelatedAngleEnergy {
        angle: angle_dist,
        energy: Some(energy_dist),
    };

    let product = ReactionProduct {
        particle: ParticleType::Neutron,
        emission_mode: "prompt".to_string(),
        decay_rate: 0.0,
        applicability: vec![],
        distribution: vec![angle_energy_dist],
        product_yield: None,
    };

    PyReactionProduct { inner: product }
}
