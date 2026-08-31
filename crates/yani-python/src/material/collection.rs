//! Python bindings for the curated material collections.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yamc_materials::collections::{pnnl, CollectionEntry};

use crate::material::PyMaterial;

/// A read-only, name-keyed table of published material compositions.
///
/// Indexing returns an ordinary :class:`Material`, freshly built each time,
/// so it can be renamed and mutated without affecting the collection or any
/// other lookup of the same name.
///
/// Examples:
///     >>> import yamc
///     >>> steel = yamc.materials.pnnl["Steel, Stainless 304"]
///     >>> steel.name = "firstwall_material"
///     >>> yamc.materials.pnnl.search("concrete")[:2]
///     ['Concrete, Barite (Type BA)', 'Concrete, Barytes-Limonite']
#[gen_stub_pyclass]
#[pyclass(name = "MaterialCollection", frozen)]
pub struct PyMaterialCollection {
    key: &'static str,
    citation: &'static str,
    url: &'static str,
}

impl PyMaterialCollection {
    pub fn pnnl() -> Self {
        PyMaterialCollection {
            key: "pnnl",
            citation: "PNNL-15870, Rev. 2",
            url: "https://www.pnnl.gov/main/publications/external/technical_reports/\
                  PNNL-15870Rev2.pdf",
        }
    }

    fn entries(&self) -> &'static [CollectionEntry] {
        match self.key {
            "pnnl" => pnnl::entries(),
            other => unreachable!("unknown collection {other}"),
        }
    }

    fn lookup(&self, name: &str) -> PyResult<&'static CollectionEntry> {
        pnnl::entry(name).ok_or_else(|| PyKeyError::new_err(unknown(name)))
    }
}

fn unknown(name: &str) -> String {
    // Reuse the core's suggestion logic so Rust and Python fail identically.
    match pnnl::material(name) {
        Err(msg) => msg,
        Ok(_) => unreachable!("lookup succeeded on the error path"),
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMaterialCollection {
    /// Build the named material.
    ///
    /// Args:
    ///     name (str): The source document's name, verbatim.
    ///
    /// Returns:
    ///     Material: A new material carrying the collection's composition,
    ///     density and name.
    ///
    /// Raises:
    ///     KeyError: If no entry has that name. The message suggests near
    ///         matches.
    fn __getitem__(&self, name: &str) -> PyResult<PyMaterial> {
        let entry = self.lookup(name)?;
        Ok(PyMaterial {
            internal: entry.to_material().map_err(PyValueError::new_err)?,
        })
    }

    /// Number of materials in the collection.
    fn __len__(&self) -> usize {
        self.entries().len()
    }

    /// Whether a name is in the collection (exact match).
    fn __contains__(&self, name: &str) -> bool {
        pnnl::entry(name).is_some()
    }

    /// Iterate over the material names, in source-document order.
    fn __iter__(&self) -> NameIter {
        NameIter {
            names: self.entries().iter().map(|e| e.name).collect(),
            pos: 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<MaterialCollection '{}': {} materials from {}>",
            self.key,
            self.entries().len(),
            self.citation
        )
    }

    /// Every material name, in source-document order.
    ///
    /// Returns:
    ///     list[str]: All names in the collection.
    fn names(&self) -> Vec<&'static str> {
        self.entries().iter().map(|e| e.name).collect()
    }

    /// Names containing ``query``, compared case-insensitively.
    ///
    /// Args:
    ///     query (str): Substring to look for, e.g. ``"concrete"``.
    ///
    /// Returns:
    ///     list[str]: Matching names, ready to pass straight back to ``[]``.
    fn search(&self, query: &str) -> Vec<&'static str> {
        pnnl::search(query)
    }

    /// Build the named material, overriding fields the collection does not set.
    ///
    /// Equivalent to ``collection[key]`` followed by attribute assignment,
    /// except for ``density``, which :class:`Material` does not allow to be
    /// reassigned after construction. Use it for a material used at other
    /// than its reference density, e.g. a concrete at a different porosity.
    ///
    /// Args:
    ///     key (str): The source document's name, verbatim.
    ///     name (Optional[str]): Rename the material, e.g. to match a mesh
    ///         geometry's material name.
    ///     density (Optional[float]): Mass density in g/cm3, replacing the
    ///         collection's reference density.
    ///     temperature (Optional[float]): Temperature in Kelvin.
    ///     volume (Optional[float]): Volume in cm³.
    ///     transmutable (bool): Mark for depletion by
    ///         ``Model.simulate_transmutation``.
    ///     id (Optional[int]): Material ID.
    ///
    /// Returns:
    ///     Material: The material with the overrides applied.
    #[pyo3(
        signature = (key, *, name=None, density=None, temperature=None, volume=None, transmutable=false, id=None),
        text_signature = "(key, *, name=None, density=None, temperature=None, volume=None, transmutable=False, id=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn material(
        &self,
        key: &str,
        name: Option<String>,
        density: Option<f64>,
        temperature: Option<f64>,
        volume: Option<f64>,
        transmutable: bool,
        id: Option<u32>,
    ) -> PyResult<PyMaterial> {
        let entry = self.lookup(key)?;
        let mut internal = match density {
            // Rebuild at the requested density rather than mutating after the
            // fact, so the per-nuclide atom densities are consistent with it.
            Some(d) => {
                if !(d.is_finite() && d > 0.0) {
                    return Err(PyValueError::new_err(format!(
                        "density must be a positive, finite number, got {d}"
                    )));
                }
                let mut e = entry.clone();
                e.density = d;
                e.to_material().map_err(PyValueError::new_err)?
            }
            None => entry.to_material().map_err(PyValueError::new_err)?,
        };
        if let Some(n) = name {
            internal.name = Some(n);
        }
        internal.material_id = id;
        internal.transmutable = transmutable;
        if let Some(v) = volume {
            internal.volume(Some(v)).map_err(PyValueError::new_err)?;
        }
        if let Some(t) = temperature {
            if !t.is_finite() {
                return Err(PyValueError::new_err("temperature must be a finite number"));
            }
            internal.set_temperature(format!("{t}"));
        }
        Ok(PyMaterial { internal })
    }

    /// The source document's own record for a material, for citation.
    ///
    /// Args:
    ///     name (str): The source document's name, verbatim.
    ///
    /// Returns:
    ///     dict: ``name``, ``number`` (the entry number in the source
    ///     document), ``density`` (g/cm3), ``formula`` (or None),
    ///     ``composition`` (element symbol or nuclide name -> atom
    ///     fraction), ``citation`` and ``url``.
    fn entry<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let e = self.lookup(name)?;
        let d = PyDict::new(py);
        d.set_item("name", e.name)?;
        d.set_item("number", e.number)?;
        d.set_item("density", e.density)?;
        d.set_item("formula", e.formula)?;
        let comp = PyDict::new(py);
        for (k, v) in &e.composition {
            comp.set_item(*k, *v)?;
        }
        d.set_item("composition", comp)?;
        d.set_item("citation", self.citation)?;
        d.set_item("url", self.url)?;
        Ok(d)
    }

    /// Short citation for the source document, e.g. ``"PNNL-15870, Rev. 2"``.
    #[getter]
    fn citation(&self) -> &'static str {
        self.citation
    }

    /// URL of the source document.
    #[getter]
    fn url(&self) -> &'static str {
        self.url
    }
}

/// Iterator over a collection's material names.
#[gen_stub_pyclass]
#[pyclass]
pub struct NameIter {
    names: Vec<&'static str>,
    pos: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl NameIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<&'static str> {
        let out = slf.names.get(slf.pos).copied();
        slf.pos += 1;
        out
    }
}

/// Names of the available material collections.
///
/// Returns:
///     list[str]: Collection names, each an attribute of ``yamc.materials``.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn collections() -> Vec<&'static str> {
    vec!["pnnl"]
}
