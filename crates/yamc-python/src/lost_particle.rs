use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc::util::lost_particle::LostParticle;
use yamc_particle::particle::ParticleType;

/// A particle that was lost during transport due to a geometry gap.
///
/// Contains diagnostic information about the particle's state at the
/// time it was lost, including position, direction, energy, and the
/// last known cell and surface.
///
/// Examples:
///     >>> model.simulate_transport()
///     >>> for lp in model.lost_particles:
///     ...     print(f"Lost {lp.particle_type} at {lp.position}, E={lp.energy} eV")
///     ...     print(f"  Last cell: {lp.last_cell_name} (ID: {lp.last_cell_id})")
///     ...     print(f"  Surface crossed: {lp.surface_id}")
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "LostParticle", from_py_object)]
#[derive(Clone)]
pub struct PyLostParticle {
    pub inner: LostParticle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLostParticle {
    /// Particle type ("neutron" or "photon").
    #[getter]
    fn particle_type(&self) -> String {
        match self.inner.particle_type {
            ParticleType::Neutron => "neutron".to_string(),
            ParticleType::Photon => "photon".to_string(),
        }
    }

    /// Position where the particle was lost [x, y, z] in cm.
    #[getter]
    fn position(&self) -> [f64; 3] {
        self.inner.position
    }

    /// Direction the particle was traveling [u, v, w].
    #[getter]
    fn direction(&self) -> [f64; 3] {
        self.inner.direction
    }

    /// Energy of the particle in eV.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// Cell ID of the last cell the particle was in before getting lost.
    #[getter]
    fn last_cell_id(&self) -> Option<u32> {
        self.inner.last_cell_id
    }

    /// Name of the last cell the particle was in before getting lost.
    #[getter]
    fn last_cell_name(&self) -> Option<String> {
        self.inner.last_cell_name.clone()
    }

    /// Surface ID of the surface that was crossed when the particle got lost.
    #[getter]
    fn surface_id(&self) -> Option<usize> {
        self.inner.surface_id
    }

    fn __repr__(&self) -> String {
        format!(
            "LostParticle(type={}, pos=[{:.6e}, {:.6e}, {:.6e}], E={:.6e} eV)",
            match self.inner.particle_type {
                ParticleType::Neutron => "neutron",
                ParticleType::Photon => "photon",
            },
            self.inner.position[0],
            self.inner.position[1],
            self.inner.position[2],
            self.inner.energy
        )
    }
}

impl From<LostParticle> for PyLostParticle {
    fn from(lp: LostParticle) -> Self {
        PyLostParticle { inner: lp }
    }
}
