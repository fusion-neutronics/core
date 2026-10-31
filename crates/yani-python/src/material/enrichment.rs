//! The enrichment specification a `Material` composition value can carry.
//!
//! This was a Python `@dataclass` in each wheel's `__init__.py`, read back in
//! `PyMaterial::new` by `getattr`. Two copies of one type meant `yamc.Enriched`
//! and `yani.Enriched` were unrelated classes, and the stub had to name one of
//! them, so the yani stub imported yamc: type-checking a yani-only install
//! needed a package that install does not have.
//!
//! One `#[pyclass]` in the shared bindings crate is one type, and the values
//! are checked when they are built rather than several frames later inside the
//! composition expansion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use yamc_nuclide::composition::validate_fraction_type;

/// Enrichment specification for use in Material composition dicts.
///
/// Build one with ``enriched()`` rather than by hand; the constructor is the
/// same either way.
#[gen_stub_pyclass]
// `from_py_object` is what lets `PyMaterial::new` extract one straight out of
// the composition dict, the same opt-in `PyMaterial` itself carries.
#[pyclass(name = "Enriched", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyEnriched {
    fraction: f64,
    target: String,
    percent: f64,
    fraction_type: String,
}

impl PyEnriched {
    /// The tuple `expand_composition_entry` takes, in its argument order.
    pub fn as_entry(&self) -> (f64, f64, &str, &str) {
        (
            self.fraction,
            self.percent,
            self.target.as_str(),
            self.fraction_type.as_str(),
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEnriched {
    /// Create an enrichment specification.
    ///
    /// Args:
    ///     fraction (float): Fraction of this component in the material.
    ///     target (str): Target nuclide for enrichment (e.g. "Li6").
    ///     percent (float): Enrichment percentage of the target isotope.
    ///     fraction_type (str): "atom" for atom percent or "mass" for weight percent.
    ///
    /// Returns:
    ///     Enriched: An enrichment specification.
    #[new]
    #[pyo3(signature = (fraction, *, target, percent, fraction_type = "atom".to_string()))]
    fn new(fraction: f64, target: String, percent: f64, fraction_type: String) -> PyResult<Self> {
        // The same bounds `expand_element_enriched` applies, checked here so a
        // bad value is rejected where it is written rather than wherever the
        // object is eventually used. An enrichment that is never passed to a
        // Material used to be silently accepted.
        validate_fraction_type(&fraction_type).map_err(PyValueError::new_err)?;
        if !fraction.is_finite() || fraction <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "fraction must be a positive, finite number, got {fraction}"
            )));
        }
        if !percent.is_finite() || percent <= 0.0 || percent > 100.0 {
            return Err(PyValueError::new_err(format!(
                "percent must be between 0 (exclusive) and 100 (inclusive), got {percent}"
            )));
        }
        if target.is_empty() {
            return Err(PyValueError::new_err("target must be a nuclide name"));
        }
        Ok(Self {
            fraction,
            target,
            percent,
            fraction_type,
        })
    }

    /// Fraction of this component in the material.
    #[getter]
    fn fraction(&self) -> f64 {
        self.fraction
    }

    /// Target nuclide for enrichment (e.g. "Li6").
    #[getter]
    fn target(&self) -> &str {
        &self.target
    }

    /// Enrichment percentage of the target isotope.
    #[getter]
    fn percent(&self) -> f64 {
        self.percent
    }

    /// "atom" for atom percent or "mass" for weight percent.
    #[getter]
    fn fraction_type(&self) -> &str {
        &self.fraction_type
    }

    fn __repr__(&self) -> String {
        format!(
            "Enriched(fraction={}, target='{}', percent={}, fraction_type='{}')",
            self.fraction, self.target, self.percent, self.fraction_type
        )
    }
}

/// Create an enrichment specification for a Material composition entry.
///
/// Args:
///     fraction (float): Fraction of this component in the material.
///     target (str): Target nuclide for enrichment (e.g. "Li6").
///     percent (float): Enrichment percentage of the target isotope.
///     fraction_type (str): "atom" for atom percent or "mass" for weight percent.
///
/// Returns:
///     Enriched: An enrichment specification.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (fraction, *, target, percent, fraction_type = "atom".to_string()))]
pub fn enriched(
    fraction: f64,
    target: String,
    percent: f64,
    fraction_type: String,
) -> PyResult<PyEnriched> {
    PyEnriched::new(fraction, target, percent, fraction_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(fraction: f64, percent: f64, ftype: &str) -> PyResult<PyEnriched> {
        PyEnriched::new(fraction, "Li6".to_string(), percent, ftype.to_string())
    }

    #[test]
    fn accepts_a_valid_specification() {
        let e = build(0.5, 60.0, "atom").expect("valid");
        assert_eq!(e.as_entry(), (0.5, 60.0, "Li6", "atom"));
    }

    #[test]
    fn percent_of_exactly_100_is_allowed() {
        assert!(build(1.0, 100.0, "atom").is_ok());
    }

    #[test]
    fn rejects_out_of_range_percent() {
        for bad in [0.0, -1.0, 100.1, f64::NAN] {
            assert!(build(1.0, bad, "atom").is_err(), "percent {bad} accepted");
        }
    }

    #[test]
    fn rejects_non_positive_fraction() {
        for bad in [0.0, -0.5, f64::INFINITY] {
            assert!(build(bad, 60.0, "atom").is_err(), "fraction {bad} accepted");
        }
    }

    #[test]
    fn rejects_an_unknown_fraction_type() {
        assert!(build(1.0, 60.0, "weight").is_err());
        assert!(build(1.0, 60.0, "mass").is_ok());
    }

    #[test]
    fn rejects_an_empty_target() {
        assert!(PyEnriched::new(1.0, String::new(), 60.0, "atom".to_string()).is_err());
    }
}
