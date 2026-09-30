//! Python bindings for dose coefficient functions.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yamc_nuclide::data::effective_dose::{
    ambient_dose_coefficients as rust_ambient_dose_coefficients,
    dose_coefficients as rust_dose_coefficients, DoseDataSource, DoseGeometry, DoseParticle,
};
use yamc_nuclide::data::photon_attenuation::{
    mass_attenuation_coefficient as rust_mass_attenuation_coefficient,
    mass_energy_absorption_air as rust_mass_energy_absorption_air, CoefficientTable,
};

/// Fluence-to-dose conversion coefficients (effective dose or ambient dose
/// equivalent H*(10)), returned by :func:`yamc.data.dose_coefficients`.
///
/// Carries the energy grid, the coefficients, and their units, and can
/// repackage itself for a tally via :meth:`as_energy_function`.
#[gen_stub_pyclass]
#[pyclass(name = "DoseCoefficients", from_py_object)]
#[derive(Clone)]
pub struct PyDoseCoefficients {
    energy: Vec<f64>,
    coefficients: Vec<f64>,
    units: String,
    particle: String,
    dose_quantity: String,
    /// `None` for ambient dose equivalent, which has no irradiation geometry.
    geometry: Option<String>,
    data_source: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDoseCoefficients {
    /// Incident energies in eV.
    #[getter]
    fn energy(&self) -> Vec<f64> {
        self.energy.clone()
    }

    /// Fluence-to-dose coefficients, one per energy.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    /// Units of the coefficients (``"pSv cm2"``).
    #[getter]
    fn units(&self) -> String {
        self.units.clone()
    }

    /// Repackage as the ``(energy, coefficients, units)`` tuple accepted by
    /// ``Tally(energy_function=...)``.
    fn as_energy_function(&self) -> (Vec<f64>, Vec<f64>, String) {
        (
            self.energy.clone(),
            self.coefficients.clone(),
            self.units.clone(),
        )
    }

    fn __repr__(&self) -> String {
        let quantity = match &self.geometry {
            Some(geometry) => geometry.clone(),
            None => self.dose_quantity.clone(),
        };
        format!(
            "DoseCoefficients({}, {}, {}, {} points, {})",
            self.particle,
            quantity,
            self.data_source,
            self.energy.len(),
            self.units
        )
    }
}

/// Return fluence-to-dose conversion coefficients.
///
/// ``dose_quantity='effective'`` (the default) returns fluence-to-effective-dose
/// coefficients from ICRP Publication 74 or 116 for an irradiation geometry.
///
/// ``dose_quantity='ambient'`` returns fluence-to-ambient-dose-equivalent
/// H*(10) coefficients from ICRP Publication 74 (Table A.21 for photons, Table
/// A.42 for neutrons), the ICRU-57 definition most regulations still reference.
/// H*(10) is defined at 10 mm depth in the ICRU sphere in an aligned and
/// expanded field, so it takes no ``geometry``, and ICRP-116 does not tabulate
/// it. It is designed to over-estimate effective dose for area monitoring, but
/// for high-energy neutrons it under-reads it. The tables cover 1 meV to 20 MeV
/// for neutrons and 10 keV to 10 MeV for photons, and are not extrapolated: a
/// tally folded with them scores nothing for particles outside that range.
///
/// Args:
///     particle (str): 'neutron' or 'photon'.
///     geometry (str, optional): Irradiation geometry for effective dose. One of:
///         'AP' (Anterior-Posterior), 'PA' (Posterior-Anterior),
///         'LLAT' (Left Lateral), 'RLAT' (Right Lateral),
///         'ROT' (Rotational), 'ISO' (Isotropic). Defaults to 'AP'. Must not be
///         given for ``dose_quantity='ambient'``.
///     data_source (str, optional): 'icrp74' or 'icrp116'. Defaults to
///         'icrp116' for effective dose and 'icrp74' for ambient dose
///         equivalent, which only ICRP-74 tabulates.
///     dose_quantity (str): 'effective' (default) for effective dose or
///         'ambient' for ambient dose equivalent H*(10).
///
/// Returns:
///     DoseCoefficients: an object with ``.energy`` (eV), ``.coefficients``
///     (pSv cm2), ``.units``, and ``.as_energy_function()``.
///
/// Examples:
///     import yamc
///
///     # Get ICRP-116 AP geometry dose coefficients
///     dc = yamc.data.dose_coefficients('neutron', 'AP')
///     dc.energy          # incident energies in eV
///     dc.coefficients    # fluence-to-dose coefficients
///     dc.units           # 'pSv cm2'
///
///     # ICRP-74 ambient dose equivalent H*(10)
///     h10 = yamc.data.dose_coefficients('neutron', dose_quantity='ambient')
///
///     # Fold them into a flux tally via the energy_function= argument
///     dose_tally = yamc.Tally(scores=['flux'], energy_function=dc.as_energy_function())
///
///     # Or let the tally fetch the same coefficients itself:
///     dose_tally = yamc.Tally(scores=['flux'], dose_coefficients=('neutron', 'AP'))
///
/// References:
///     - ICRP Publication 74: https://doi.org/10.1016/S0146-6453(96)90010-X
///     - ICRP Publication 116: https://doi.org/10.1016/j.icrp.2011.10.001
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (particle, geometry=None, data_source=None, dose_quantity="effective"))]
pub fn dose_coefficients(
    particle: &str,
    geometry: Option<&str>,
    data_source: Option<&str>,
    dose_quantity: &str,
) -> PyResult<PyDoseCoefficients> {
    // Parse particle
    let particle_kind = match particle {
        "neutron" => DoseParticle::Neutron,
        "photon" => DoseParticle::Photon,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Particle '{particle}' not supported. Must be 'neutron' or 'photon'."
            )))
        }
    };

    let ambient = match dose_quantity {
        "effective" => false,
        "ambient" => true,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "dose_quantity must be 'effective' or 'ambient', got '{other}'"
            )))
        }
    };

    // Parse data source, defaulting to the newest publication that
    // tabulates the requested quantity.
    let data_source = data_source.unwrap_or(if ambient { "icrp74" } else { "icrp116" });
    let source = match data_source.to_lowercase().as_str() {
        "icrp74" => DoseDataSource::ICRP74,
        "icrp116" => DoseDataSource::ICRP116,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid data_source '{data_source}'. Must be 'icrp74' or 'icrp116'"
            )))
        }
    };

    if ambient {
        if let Some(geometry) = geometry {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "geometry='{geometry}' was given with dose_quantity='ambient', but ambient \
                 dose equivalent H*(10) has no irradiation geometry: it is defined in an \
                 aligned and expanded field in the ICRU sphere. Omit geometry."
            )));
        }
        let (energy, coefficients) = rust_ambient_dose_coefficients(particle_kind, source)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "data_source='{data_source}' does not tabulate ambient dose equivalent \
                     H*(10). Use data_source='icrp74'."
                ))
            })?;
        return Ok(PyDoseCoefficients {
            energy,
            coefficients,
            units: "pSv cm2".to_string(),
            particle: particle.to_string(),
            dose_quantity: dose_quantity.to_string(),
            geometry: None,
            data_source: data_source.to_lowercase(),
        });
    }

    // Parse geometry
    let geometry = geometry.unwrap_or("AP");
    let geom = match geometry.to_uppercase().as_str() {
        "AP" => DoseGeometry::AP,
        "PA" => DoseGeometry::PA,
        "LLAT" => DoseGeometry::LLAT,
        "RLAT" => DoseGeometry::RLAT,
        "ROT" => DoseGeometry::ROT,
        "ISO" => DoseGeometry::ISO,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid geometry '{geometry}'. Must be one of: AP, PA, LLAT, RLAT, ROT, ISO"
            )))
        }
    };

    let (energy, coefficients) = rust_dose_coefficients(particle_kind, geom, source);
    Ok(PyDoseCoefficients {
        energy,
        coefficients,
        units: "pSv cm2".to_string(),
        particle: particle.to_string(),
        dose_quantity: dose_quantity.to_string(),
        geometry: Some(geometry.to_uppercase()),
        data_source: data_source.to_lowercase(),
    })
}

/// A photon coefficient tabulated against energy, returned by
/// :func:`yamc.data.mass_attenuation_coefficient` and
/// :func:`yamc.data.mass_energy_absorption_coefficient`.
///
/// Carries the energy grid and the coefficients, and evaluates itself between
/// the tabulated points with the log-log interpolation NIST publishes these for.
#[gen_stub_pyclass]
#[pyclass(name = "PhotonCoefficients", from_py_object)]
#[derive(Clone)]
pub struct PyPhotonCoefficients {
    table: CoefficientTable,
    units: String,
    quantity: String,
    material: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPhotonCoefficients {
    /// Tabulated energies in eV. An absorption edge appears as two energies one
    /// ulp apart, carrying the value below and above the edge.
    #[getter]
    fn energy(&self) -> Vec<f64> {
        self.table.energy().to_vec()
    }

    /// The coefficient at each tabulated energy, in cm²/g.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.table.value().to_vec()
    }

    /// Units of the coefficients (``"cm2/g"``).
    #[getter]
    fn units(&self) -> String {
        self.units.clone()
    }

    /// The coefficient at ``energy`` (eV), log-log interpolated.
    ///
    /// Energies outside the tabulated range return the nearest end value rather
    /// than extrapolating: the tabulation stops where the data does.
    fn interpolate(&self, energy: f64) -> f64 {
        self.table.interpolate(energy)
    }

    /// Repackage as the ``(energy, coefficients, units)`` tuple accepted by
    /// ``Tally(energy_function=...)``.
    fn as_energy_function(&self) -> (Vec<f64>, Vec<f64>, String) {
        (self.energy(), self.coefficients(), self.units.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "PhotonCoefficients({}, {}, {} points, {})",
            self.quantity,
            self.material,
            self.table.energy().len(),
            self.units
        )
    }
}

/// Return the photon mass attenuation coefficient mu/rho of an element.
///
/// Total attenuation with coherent scattering, from the NIST XCOM database,
/// tabulated from 1 keV to 20 MeV for Z = 1 to 100. Dividing a material's dose
/// or fluence by ``sum(density_i * mu_over_rho_i(E))`` is what a self-shielding
/// estimate like :meth:`Material.contact_dose` does with it.
///
/// Args:
///     element (str | int): Element symbol (``'Fe'``) or atomic number (``26``).
///
/// Returns:
///     PhotonCoefficients: mu/rho in cm²/g against energy in eV.
///
/// Examples:
///     import yamc
///
///     iron = yamc.data.mass_attenuation_coefficient('Fe')
///     iron.interpolate(1.0e6)     # 0.05995 cm2/g at 1 MeV
///     iron.energy, iron.coefficients
///
/// References:
///     - NIST Standard Reference Database 8 (XCOM): <https://doi.org/10.18434/T48G6X>
#[gen_stub_pyfunction]
#[pyfunction]
pub fn mass_attenuation_coefficient(element: &Bound<'_, PyAny>) -> PyResult<PyPhotonCoefficients> {
    let (z, label) = if let Ok(z) = element.extract::<u32>() {
        (z, z.to_string())
    } else {
        let symbol: String = element.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "element must be an element symbol (e.g. 'Fe') or an atomic number (e.g. 26)",
            )
        })?;
        let z = endf::data::atomic_number(&symbol).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "'{symbol}' is not a recognized element symbol"
            ))
        })?;
        (z, symbol)
    };

    let table = rust_mass_attenuation_coefficient(z).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "No photon mass attenuation data for Z={z}; the tabulation covers Z = 1 to 100"
        ))
    })?;
    Ok(PyPhotonCoefficients {
        table: table.clone(),
        units: "cm2/g".to_string(),
        quantity: "mu/rho".to_string(),
        material: label,
    })
}

/// Return the mass energy-absorption coefficient mu_en/rho of a material.
///
/// The fraction of incident photon energy actually absorbed per unit mass, less
/// what scattered photons carry away, from NIST SRD 126. This is the response an
/// absorbed dose in air folds against, and it is a different quantity from the
/// attenuation coefficient: attenuation counts photons removed from the beam,
/// this counts energy deposited.
///
/// Args:
///     material (str): Currently only ``'air'`` (dry, near sea level), tabulated
///         from 1 keV to 20 MeV.
///
/// Returns:
///     PhotonCoefficients: mu_en/rho in cm²/g against energy in eV.
///
/// Examples:
///     import yamc
///
///     air = yamc.data.mass_energy_absorption_coefficient('air')
///     air.interpolate(1.0e6)      # 0.02789 cm2/g at 1 MeV
///
/// References:
///     - NIST Standard Reference Database 126: <https://doi.org/10.18434/T4D01F>
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (material="air"))]
pub fn mass_energy_absorption_coefficient(material: &str) -> PyResult<PyPhotonCoefficients> {
    if material != "air" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "No mass energy-absorption data for '{material}'. Only 'air' is tabulated."
        )));
    }
    Ok(PyPhotonCoefficients {
        table: rust_mass_energy_absorption_air().clone(),
        units: "cm2/g".to_string(),
        quantity: "mu_en/rho".to_string(),
        material: material.to_string(),
    })
}
