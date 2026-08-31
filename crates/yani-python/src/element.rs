// Python bindings for the element module
// Implement PyO3 wrappers here if needed

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use yamc_element::element::Element;

#[gen_stub_pyclass]
#[pyclass(name = "Element")]
/// Chemical element container providing isotope helper methods.
///
/// Args:
///     name (str): Element symbol (e.g. "Fe", "U", "H").
pub struct PyElement {
    pub inner: Element,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyElement {
    #[new]
    #[pyo3(text_signature = "(name)")]
    /// Create a new Element.
    ///
    /// Args:
    ///     name (str): Element symbol (e.g. "Fe", "U", "H").
    fn new(name: String) -> Self {
        Self {
            inner: Element::new(name),
        }
    }

    fn __repr__(&self) -> String {
        format!("Element(name={})", self.inner.name)
    }

    /// Element symbol.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Return list of isotope (nuclide) identifiers for this element.
    ///
    /// Returns:
    ///     List[str]: Isotope names sorted by mass number (e.g. ["Fe54", "Fe56"]).
    #[pyo3(text_signature = "(self)")]
    fn get_nuclides(&self) -> Vec<String> {
        self.inner.get_nuclides()
    }
}
