//! A decay photon continuum, as `Material.decay_photon_continua` returns it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yani_decay::PhotonContinuum;

/// One nuclide's decay photon continuum within a material's inventory.
///
/// ENDF gives part of some decay spectra as a density over energy rather than
/// as lines: the spontaneous-fission photons of an actinide, or the whole
/// photon emission of a nuclide far from stability, whose lines were never
/// measured. The values are a rate per eV, so they are not line rates and
/// cannot be added to ``decay_photon_spectrum()``'s. Their rate is the
/// integral, which ``emission_rate`` gives. Both follow the ``per`` argument
/// of ``decay_photon_continua()``: for the whole material, or per cm³ or per g.
#[gen_stub_pyclass]
#[pyclass(name = "PhotonContinuum", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPhotonContinuum {
    inner: PhotonContinuum,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPhotonContinuum {
    /// The emitting nuclide.
    #[getter]
    fn nuclide(&self) -> String {
        self.inner.nuclide.clone()
    }

    /// Tabulated energies [eV], ascending.
    #[getter]
    fn energies(&self) -> Vec<f64> {
        self.inner.energies.clone()
    }

    /// The emission-rate density at each energy: photons/s/eV for the whole
    /// material, or per cm³ or per g following the ``per`` argument the
    /// continuum was requested with.
    #[getter]
    fn rates(&self) -> Vec<f64> {
        self.inner.rates.clone()
    }

    /// How ``rates`` is read between the energies: the ENDF law by name
    /// (``"histogram"``, ``"linear-linear"``, ...), or None where the data
    /// states no law, which data written before the law was stored does.
    #[getter]
    fn interpolation(&self) -> Option<&'static str> {
        self.inner.interpolation.map(|law| law.name())
    }

    /// The emission rate over the whole continuum, its integral read under its
    /// law: photons/s for the whole material, or per cm³ or per g following
    /// the ``per`` argument the continuum was requested with.
    ///
    /// Raises:
    ///     ValueError: If the law is not stated, or is one this build does not
    ///         integrate, or if the energy and rate lists are unpaired, the
    ///         energies are not finite or descend, or the rates are negative
    ///         or not finite. The integral is then unknown, and no number is
    ///         returned in its place.
    #[getter]
    fn emission_rate(&self) -> PyResult<f64> {
        self.inner.emission_rate().map_err(|why| {
            PyValueError::new_err(format!(
                "the decay photon continuum of {} {why}",
                self.inner.nuclide
            ))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PhotonContinuum(nuclide={:?}, points={}, interpolation={}, emission_rate={})",
            self.inner.nuclide,
            self.inner.energies.len(),
            self.inner
                .interpolation
                .map_or("None".to_string(), |law| format!("{:?}", law.name())),
            self.inner
                .emission_rate()
                .map_or("None".to_string(), |rate| format!("{rate:.4e}")),
        )
    }
}

impl From<PhotonContinuum> for PyPhotonContinuum {
    fn from(inner: PhotonContinuum) -> Self {
        Self { inner }
    }
}
