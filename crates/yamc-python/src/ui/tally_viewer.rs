//! Python-facing wrapper for the interactive mesh-tally viewer.
//!
//! The pure-Rust core (templating, the ~1k-line JS body, all the
//! tally-specific structs) moved to `yamc_plot::tally_html` in the
//! share-tally-viewer refactor so the geometry viewer and the tally
//! viewer can share infrastructure (escape_js, parse_hex_color, and
//! eventually surface-hover tooltips).
//!
//! This file now keeps only what talks to Python:
//! - `PyInteractiveTallyPlot` (the `_repr_html_` / `save()` pyclass)
//!
//! Re-exports the moved items so existing callers inside yamc-python
//! (`tally.rs`, etc.) can keep importing through this module.

pub use yamc_plot::{
    build_interactive_tally_html, EmbeddedSlice, ExtraValueSlices, InteractiveTallyParams, MeshMeta,
};

use pyo3::prelude::*;
use pyo3::types::PyDictMethods;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Result object for interactive tally plots.
///
/// Renders inline in Jupyter notebooks via ``_repr_html_()``.
/// Use :meth:`save` to write the result to disk.
///
/// Examples:
///     >>> plot = tally.interactive_plot(geometry=geo, basis="xy")
///     >>> plot.save("tally.html")   # interactive HTML
#[gen_stub_pyclass]
#[pyclass(name = "InteractiveTallyPlot", module = "yamc._core", unsendable)]
pub struct PyInteractiveTallyPlot {
    html: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInteractiveTallyPlot {
    /// Jupyter rich display -- renders the interactive viewer inline.
    fn _repr_html_(&self) -> &str {
        &self.html
    }

    /// Rich display as an *isolated* output. The viewer is a full standalone
    /// HTML document, so we flag `text/html` as `isolated` -- Jupyter and
    /// myst-nb then sandbox it in an iframe instead of inlining the nested
    /// document (which would be invalid HTML). Takes precedence over
    /// `_repr_html_` where supported.
    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_<'py>(
        &self,
        py: pyo3::prelude::Python<'py>,
        include: Option<&pyo3::prelude::Bound<'py, pyo3::types::PyAny>>,
        exclude: Option<&pyo3::prelude::Bound<'py, pyo3::types::PyAny>>,
    ) -> pyo3::prelude::PyResult<(
        pyo3::prelude::Bound<'py, pyo3::types::PyDict>,
        pyo3::prelude::Bound<'py, pyo3::types::PyDict>,
    )> {
        let _ = (include, exclude);
        let data = pyo3::types::PyDict::new(py);
        data.set_item("text/html", &self.html)?;
        let html_meta = pyo3::types::PyDict::new(py);
        html_meta.set_item("isolated", true)?;
        let metadata = pyo3::types::PyDict::new(py);
        metadata.set_item("text/html", html_meta)?;
        Ok((data, metadata))
    }

    fn __str__(&self) -> &str {
        &self.html
    }

    fn __repr__(&self) -> String {
        format!("InteractiveTallyPlot({} bytes HTML)", self.html.len())
    }

    /// The raw HTML string.
    #[getter]
    fn html(&self) -> &str {
        &self.html
    }

    /// Save the plot to a file.
    ///
    /// Args:
    ///     filename: Output file path (`.html`).
    fn save(&self, filename: &str) -> pyo3::PyResult<()> {
        std::fs::write(filename, &self.html)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

impl PyInteractiveTallyPlot {
    pub fn new(html: String) -> Self {
        Self { html }
    }
}
