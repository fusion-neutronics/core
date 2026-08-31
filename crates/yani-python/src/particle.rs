use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_particle::particle::{Particle, ParticleType};

/// A single transport particle: position, direction, energy, and alive/id state.
#[gen_stub_pyclass]
#[pyclass(name = "Particle", from_py_object)]
#[derive(Clone)]
pub struct PyParticle {
    pub inner: Particle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyParticle {
    #[new]
    pub fn new(
        position: [f64; 3],
        direction: [f64; 3],
        energy: f64,
        alive: Option<bool>,
        id: Option<u32>,
    ) -> Self {
        PyParticle {
            inner: Particle {
                particle_type: ParticleType::Neutron,
                position,
                last_position: position, // Initialize to same as position
                direction,
                energy,
                weight: 1.0, // Default weight
                alive: alive.unwrap_or(true),
                id: id.unwrap_or(0),
                current_cell_index: yamc_particle::particle::NO_CELL, // Will be set during transport
                previous_cell_index: yamc_particle::particle::NO_CELL, // Set on surface crossings for neighbor acceleration
                urr_random: yamc_particle::particle::NO_URR, // Set per-collision when in URR range
                urr_energy: -1.0, // Invalid energy to force initial sampling
                last_surface_id: yamc_particle::particle::NO_SURFACE, // Set on surface crossings for lost particle diagnostics
                parent_nuclide: None, // Only set for D1S decay photons
                #[cfg(feature = "debug_history")]
                history: Vec::new(),
            },
        }
    }

    pub fn __repr__(&self) -> String {
        let ptype = match self.inner.particle_type {
            ParticleType::Neutron => "neutron",
            ParticleType::Photon => "photon",
        };
        format!(
            "Particle(type={}, energy={:.6e}, position={:?})",
            ptype, self.inner.energy, self.inner.position,
        )
    }

    /// Particle position in 3D space (x, y, z).
    ///
    /// Returns:
    ///     Array of 3 floats [x, y, z]
    #[getter]
    pub fn position(&self) -> [f64; 3] {
        self.inner.position
    }

    /// Particle direction unit vector (u, v, w).
    ///
    /// Returns:
    ///     Array of 3 floats [u, v, w]
    #[getter]
    pub fn direction(&self) -> [f64; 3] {
        self.inner.direction
    }

    /// Particle kinetic energy in eV.
    ///
    /// Returns:
    ///     Float energy in eV
    #[getter]
    pub fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// Whether the particle is still alive (not absorbed or leaked).
    ///
    /// Returns:
    ///     Boolean indicating if particle is alive
    #[getter]
    pub fn alive(&self) -> bool {
        self.inner.alive
    }

    /// Unique identifier for this particle.
    ///
    /// Returns:
    ///     Integer particle ID
    #[getter]
    pub fn id(&self) -> u32 {
        self.inner.id
    }

    /// Particle type ("neutron" or "photon").
    ///
    /// Returns:
    ///     str: The particle type
    #[getter]
    pub fn particle_type(&self) -> String {
        match self.inner.particle_type {
            ParticleType::Neutron => "neutron".to_string(),
            ParticleType::Photon => "photon".to_string(),
        }
    }
}
