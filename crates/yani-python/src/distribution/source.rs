use crate::distribution::PyCylindricalRing;
use crate::distribution::{PyDiscrete, PyHistogram, PyNormal, PyUniform};
use crate::distribution::{PyIsotropic, PyMonodirectional};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

// Helper conversions between Python objects and the yamc-source types.
// These were previously `FromPyObject` / `IntoPyObject` impls, but the
// orphan rule blocks impls of foreign traits on foreign types once those
// types live in `yamc-source`. Plain functions keep the same behaviour.

pub(crate) fn extract_angular_distribution(ob: &Bound<'_, PyAny>) -> PyResult<AngularDistribution> {
    if let Ok(mono) = ob.extract::<PyMonodirectional>() {
        Ok(mono.inner.clone())
    } else if let Ok(iso) = ob.extract::<PyIsotropic>() {
        Ok(iso.inner.clone())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected Isotropic or Monodirectional",
        ))
    }
}

pub(crate) fn angular_distribution_into_py(
    py: Python<'_>,
    d: AngularDistribution,
) -> PyResult<Py<PyAny>> {
    match d {
        AngularDistribution::Isotropic => {
            let obj = Py::new(py, PyIsotropic { inner: d })?;
            Ok(obj.into_any())
        }
        AngularDistribution::Monodirectional { .. } => {
            let obj = Py::new(py, PyMonodirectional { inner: d })?;
            Ok(obj.into_any())
        }
    }
}

pub(crate) fn extract_source_energy_distribution(
    ob: &Bound<'_, PyAny>,
) -> PyResult<SourceEnergyDistribution> {
    if let Ok(discrete) = ob.extract::<PyDiscrete>() {
        Ok(SourceEnergyDistribution::Discrete(discrete.inner.clone()))
    } else if let Ok(histogram) = ob.extract::<PyHistogram>() {
        Ok(SourceEnergyDistribution::Histogram(histogram.inner.clone()))
    } else if let Ok(uniform) = ob.extract::<PyUniform>() {
        Ok(SourceEnergyDistribution::Uniform(uniform.inner.clone()))
    } else if let Ok(normal) = ob.extract::<PyNormal>() {
        Ok(SourceEnergyDistribution::Normal(normal.inner.clone()))
    } else if let Ok(energy_val) = ob.extract::<f64>() {
        match yamc_source::distribution::energy::Discrete::new(vec![energy_val], vec![1.0]) {
            Ok(discrete) => Ok(SourceEnergyDistribution::Discrete(discrete)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid energy value: {}",
                e
            ))),
        }
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Energy must be a Discrete, Histogram, Uniform, or Normal distribution or a float (e.g., yamc.Histogram([0.0, 1e6, 20e6], [0.3, 0.7]) or yamc.Discrete([14.06e6], [1.0]) or 14.06e6)"
        ))
    }
}

pub(crate) fn source_energy_distribution_into_py(
    py: Python<'_>,
    d: SourceEnergyDistribution,
) -> PyResult<Py<PyAny>> {
    match d {
        SourceEnergyDistribution::Discrete(d) => {
            let obj = Py::new(py, PyDiscrete { inner: d })?;
            Ok(obj.into_any())
        }
        SourceEnergyDistribution::Histogram(h) => {
            let obj = Py::new(py, PyHistogram { inner: h })?;
            Ok(obj.into_any())
        }
        SourceEnergyDistribution::Uniform(u) => {
            let obj = Py::new(py, PyUniform { inner: u })?;
            Ok(obj.into_any())
        }
        SourceEnergyDistribution::Normal(n) => {
            let obj = Py::new(py, PyNormal { inner: n })?;
            Ok(obj.into_any())
        }
    }
}

pub(crate) fn extract_source_spatial_distribution(
    ob: &Bound<'_, PyAny>,
) -> PyResult<SourceSpatialDistribution> {
    if let Ok(cyl) = ob.extract::<PyCylindricalRing>() {
        Ok(SourceSpatialDistribution::CylindricalRing(Box::new(
            cyl.inner.clone(),
        )))
    } else if let Ok(xyz) = ob.extract::<[f64; 3]>() {
        Ok(SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new(xyz),
        ))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "position must be a tuple/list of 3 floats (x, y, z), or a CylindricalRing distribution",
        ))
    }
}

pub(crate) fn source_spatial_distribution_into_py(
    py: Python<'_>,
    d: SourceSpatialDistribution,
) -> PyResult<Py<PyAny>> {
    match d {
        SourceSpatialDistribution::Point(p) => {
            // A fixed point is returned as a plain (x, y, z) list, matching the
            // tuple form accepted by `position=`.
            let pos = p.position().to_vec();
            Ok(pos.into_pyobject(py)?.into_any().unbind())
        }
        SourceSpatialDistribution::CylindricalRing(c) => {
            let obj = Py::new(py, PyCylindricalRing { inner: *c })?;
            Ok(obj.into_any())
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers to avoid duplicating code in NeutronSource / PhotonSource
// ---------------------------------------------------------------------------

fn source_repr(name: &str, src: &Source, strength: f64) -> String {
    let position_str = match &src.space {
        SourceSpatialDistribution::Point(p) => {
            let pos = p.position();
            format!("({}, {}, {})", pos[0], pos[1], pos[2])
        }
        SourceSpatialDistribution::CylindricalRing(ref c) => {
            let o = c.origin;
            format!("CylindricalRing(origin=[{}, {}, {}])", o[0], o[1], o[2])
        }
    };
    let energy_str = match &src.energy {
        SourceEnergyDistribution::Discrete(d) => {
            format!(
                "Discrete(energies={:?}, probabilities={:?})",
                d.energies(),
                d.probabilities()
            )
        }
        SourceEnergyDistribution::Histogram(h) => {
            format!(
                "Histogram(boundaries={:?}, probabilities={:?})",
                h.boundaries(),
                h.probabilities()
            )
        }
        SourceEnergyDistribution::Uniform(u) => {
            format!("Uniform(low={}, high={})", u.a(), u.b())
        }
        SourceEnergyDistribution::Normal(n) => {
            format!("Normal(mean={}, std_dev={})", n.mean_val(), n.std_dev())
        }
    };
    let strength_str = if (strength - 1.0).abs() > f64::EPSILON {
        format!(", strength={}", strength)
    } else {
        String::new()
    };
    format!(
        "{}(position={}, energy={}{})",
        name, position_str, energy_str, strength_str
    )
}

fn build_source(
    particle_type: yamc_particle::ParticleType,
    position: Option<SourceSpatialDistribution>,
    energy: Option<SourceEnergyDistribution>,
    direction: Option<AngularDistribution>,
    strength: Option<f64>,
) -> PyResult<Source> {
    // Apply the per-particle-type energy default (or fail-fast for photons,
    // which have no canonical default birth energy) in yamc-source.
    let mut src = Source::with_default_energy(particle_type, energy)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    if let Some(val) = position {
        src.space = val;
    }
    if let Some(val) = direction {
        src.angle = val;
    }
    if let Some(s) = strength {
        src.strength = s;
    }
    Ok(src)
}

// ---------------------------------------------------------------------------
// NeutronSource / PhotonSource
//
// The two pyclasses expose an identical method set; they differ only in the
// Python class name, the `ParticleSource` variant their constructor builds,
// the `__repr__` label, and a couple of docstring examples. The macro below
// generates both from a single template so the shared logic lives in one
// place.
// ---------------------------------------------------------------------------

macro_rules! particle_source_pyclass {
    (
        $rust_name:ident,
        py_name = $py_name:literal,
        variant = $variant:ident,
        struct_doc = $struct_doc:literal,
        new_doc = $new_doc:literal,
    ) => {
        #[doc = $struct_doc]
        #[gen_stub_pyclass]
        #[pyclass(name = $py_name, from_py_object)]
        #[derive(Clone)]
        pub struct $rust_name {
            pub inner: ParticleSource,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $rust_name {
            #[doc = $new_doc]
            #[new]
            #[pyo3(signature = (*, position=None, energy=None, direction=None, strength=None))]
            pub fn new(
                position: Option<&Bound<'_, PyAny>>,
                energy: Option<&Bound<'_, PyAny>>,
                direction: Option<&Bound<'_, PyAny>>,
                strength: Option<f64>,
            ) -> PyResult<Self> {
                let position = position
                    .map(extract_source_spatial_distribution)
                    .transpose()?;
                let energy = energy.map(extract_source_energy_distribution).transpose()?;
                let direction = direction.map(extract_angular_distribution).transpose()?;
                Ok($rust_name {
                    inner: ParticleSource::$variant(build_source(
                        yamc_particle::ParticleType::$variant,
                        position,
                        energy,
                        direction,
                        strength,
                    )?),
                })
            }

            /// Sample a particle from the source.
            pub fn sample(&self) -> crate::particle::PyParticle {
                let mut rng = rand::rng();
                crate::particle::PyParticle {
                    inner: self.inner.sample(&mut rng),
                }
            }

            /// Sample ``n`` particles at once, returning their birth positions
            /// and energies.
            ///
            /// The compiled batch form of calling :meth:`sample` in a loop:
            /// used by the VTK-HDF source export to avoid one Python<->Rust
            /// crossing per sample.
            ///
            /// Args:
            ///     n: Number of particles to sample.
            ///
            /// Returns:
            ///     A ``(positions, energies)`` tuple where ``positions`` is a
            ///     list of ``[x, y, z]`` (length ``n``) and ``energies`` is a
            ///     list of floats in eV (length ``n``).
            #[pyo3(signature = (n))]
            pub fn sample_n(&self, n: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
                let mut rng = rand::rng();
                let mut positions = Vec::with_capacity(n);
                let mut energies = Vec::with_capacity(n);
                for _ in 0..n {
                    let p = self.inner.sample(&mut rng);
                    positions.push(p.position);
                    energies.push(p.energy);
                }
                (positions, energies)
            }

            /// Spatial distribution describing where particles are sampled.
            /// Returns an ``(x, y, z)`` list for a fixed point, or a
            /// ``CylindricalRing`` distribution.
            #[getter]
            pub fn position(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                source_spatial_distribution_into_py(py, self.inner.source().space.clone())
            }

            #[setter(position)]
            pub fn set_position(&mut self, position: &Bound<'_, PyAny>) -> PyResult<()> {
                self.inner.source_mut().space = extract_source_spatial_distribution(position)?;
                Ok(())
            }

            /// Energy distribution describing source particle energies in eV.
            /// Returns a `Discrete`, `Uniform`, or `Normal` distribution.
            #[getter]
            pub fn energy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                source_energy_distribution_into_py(py, self.inner.source().energy.clone())
            }

            #[setter(energy)]
            pub fn set_energy(&mut self, energy: &Bound<'_, PyAny>) -> PyResult<()> {
                self.inner.source_mut().energy = extract_source_energy_distribution(energy)?;
                Ok(())
            }

            /// Angular distribution describing source particle directions.
            /// Returns an `Isotropic` or `Monodirectional` distribution.
            #[getter]
            pub fn direction(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                angular_distribution_into_py(py, self.inner.source().angle.clone())
            }

            #[setter(direction)]
            pub fn set_direction(&mut self, direction: &Bound<'_, PyAny>) -> PyResult<()> {
                self.inner.source_mut().angle = extract_angular_distribution(direction)?;
                Ok(())
            }

            /// Relative strength of this source for multi-source weighting.
            /// Default is 1.0; only meaningful when multiple sources are passed to a Model.
            #[getter]
            pub fn strength(&self) -> f64 {
                self.inner.strength()
            }

            #[setter]
            pub fn set_strength(&mut self, value: f64) {
                self.inner.source_mut().strength = value;
            }

            pub fn __repr__(&self) -> String {
                source_repr($py_name, self.inner.source(), self.inner.strength())
            }
        }
    };
}

particle_source_pyclass! {
    PyNeutronSource,
    py_name = "NeutronSource",
    variant = Neutron,
    struct_doc = " A neutron particle source with spatial, energy, and angular distributions.",
    new_doc = " Create a neutron source.\n\n Args:\n     position: Birth location. Pass a tuple/list ``(x, y, z)`` in cm for a fixed\n            point (recommended), or a spatial distribution object\n            (yamc.Point, yamc.CylindricalRing) for finer control.\n            Defaults to the origin (0, 0, 0).\n     energy: Birth energy. Pass a float in eV for a single energy (recommended),\n            or an energy distribution (yamc.Discrete, yamc.Uniform, yamc.Normal)\n            for a spectrum. Defaults to 14.06 MeV (D-T fusion).\n     direction: Angular distribution (e.g., yamc.Isotropic()).\n            Defaults to Isotropic.\n     strength: Source strength for multi-source weighting. Defaults to 1.0.\n\n Returns:\n     NeutronSource: A neutron source.",
}

particle_source_pyclass! {
    PyPhotonSource,
    py_name = "PhotonSource",
    variant = Photon,
    struct_doc = " A photon particle source with spatial, energy, and angular distributions.",
    new_doc = " Create a photon source.\n\n Args:\n     position: Birth location. Pass a tuple/list ``(x, y, z)`` in cm for a fixed\n            point (recommended), or a spatial distribution object\n            (yamc.Point, yamc.CylindricalRing) for finer control.\n            Defaults to the origin (0, 0, 0).\n     energy: Birth energy. Pass a float in eV for a single energy (recommended),\n            or an energy distribution (yamc.Discrete, yamc.Uniform, yamc.Normal)\n            for a spectrum. Required: photons have no default birth energy, so\n            omitting this raises a ValueError.\n     direction: Angular distribution (e.g., yamc.Isotropic()).\n            Defaults to Isotropic.\n     strength: Source strength for multi-source weighting. Defaults to 1.0.\n\n Returns:\n     PhotonSource: A photon source.",
}
