//! Python-facing wrapper for the shared interactive viewer.
//!
//! The pure-Rust core (templating, slice rasterizer, PNG renderer, all
//! the JS string literals) lives in the `yamc-plot` crate so the
//! browser-side editor can reuse the same code without pulling pyo3.
//! This module keeps only the things that talk to Python:
//!
//! - `parse_contour_kwargs` / `parse_source_kwargs` -- extract a
//!   `Bound<PyDict>` into the pure-Rust [`ContourParams`] / [`SourceParams`].
//! - `PyInteractivePlot` -- the `_repr_html_` / `save()` pyclass.
//!
//! Everything else is re-exported from `yamc_plot` so callers inside
//! `yamc-python` (e.g. `interactive_tally_viewer.rs`) keep working
//! through `use crate::ui::{escape_js, ...}`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDictMethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

pub use yamc_plot::{
    build_interactive_html, draw_source_overlay, escape_js, parse_hex_color, render_grid_rgba,
    ContourParams, InteractiveViewParams, PresampledGrid, SourceOverlay, SourceParams,
    DISCRETE_COLORS,
};

// Local helper since `render_png_rgba` is pub but our pyo3 wrapper benefits
// from the same module path. Re-export so the existing call sites compile.
pub use yamc_plot::viewer_html::render_png_rgba;

/// Extract contour_kwargs from a Python dict.
pub fn parse_contour_kwargs(
    kwargs: Option<&pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
) -> pyo3::PyResult<ContourParams> {
    let mut p = ContourParams::default();
    if let Some(d) = kwargs {
        if let Some(v) = d.get_item("colors")? {
            p.colors = v.extract::<String>()?;
        }
        if let Some(v) = d.get_item("linewidths")? {
            p.linewidths = v.extract::<usize>()?;
        }
    }
    Ok(p)
}

/// Extract source_kwargs from a Python dict.
pub fn parse_source_kwargs(
    kwargs: Option<&pyo3::prelude::Bound<'_, pyo3::types::PyDict>>,
) -> pyo3::PyResult<SourceParams> {
    let mut p = SourceParams::default();
    if let Some(d) = kwargs {
        if let Some(v) = d.get_item("color")? {
            p.color = v.extract::<String>()?;
        }
        if let Some(v) = d.get_item("size")? {
            p.size = v.extract::<usize>()?;
        }
    }
    Ok(p)
}

/// Result of ``.plot()`` -- renders inline in Jupyter and
/// supports saving to ``.html`` or ``.png``.
///
/// In a Jupyter notebook the plot renders automatically via
/// ``_repr_html_()``.  Use :meth:`save` to write the result to disk.
///
/// Examples:
///     >>> plot = geometry.plot(basis="xy")
///     >>> plot.to_html("geometry.html")   # interactive HTML
///     >>> plot.to_png("geometry.png")     # static PNG snapshot
#[gen_stub_pyclass]
#[pyclass(name = "InteractivePlot", module = "yamc._core", unsendable)]
pub struct PyInteractivePlot {
    html: String,
    /// Pre-rendered PNG RGBA data (with axes, border, ticks).
    png_rgba: Vec<u8>,
    png_w: u32,
    png_h: u32,
    /// Plot grid dimensions (without margins).
    ph: usize,
    pv: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInteractivePlot {
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
        format!(
            "InteractivePlot({}x{} px, {} bytes HTML)",
            self.ph,
            self.pv,
            self.html.len()
        )
    }

    /// The raw HTML string.
    #[getter]
    fn html(&self) -> &str {
        &self.html
    }

    /// Save the plot as a self-contained interactive HTML page, or return
    /// its bytes when ``path`` is None.
    ///
    /// Args:
    ///     path (str | os.PathLike | None): Where to write.
    ///
    /// Returns:
    ///     bytes | None: The page when ``path`` is None.
    #[pyo3(signature = (path=None))]
    fn to_html(
        &self,
        py: Python<'_>,
        path: Option<std::path::PathBuf>,
    ) -> PyResult<Option<Py<pyo3::types::PyBytes>>> {
        yani_python::file_io::write_or_return(py, path, self.html.as_bytes())
    }

    /// Save the plot as a static PNG image with axes, border and tick
    /// marks, or return its bytes when ``path`` is None.
    ///
    /// Args:
    ///     path (str | os.PathLike | None): Where to write.
    ///
    /// Returns:
    ///     bytes | None: The image when ``path`` is None.
    #[pyo3(signature = (path=None))]
    fn to_png(
        &self,
        py: Python<'_>,
        path: Option<std::path::PathBuf>,
    ) -> PyResult<Option<Py<pyo3::types::PyBytes>>> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, self.png_w, self.png_h);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            writer
                .write_image_data(&self.png_rgba)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        yani_python::file_io::write_or_return(py, path, &bytes)
    }
}

impl PyInteractivePlot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        html: String,
        presampled: &PresampledGrid,
        color_by: &str,
        outline: Option<&str>,
        colors: Option<&HashMap<i32, String>>,
        origin: (f64, f64, f64),
        width: (f64, f64),
        basis: &str,
        axis_units: &str,
        font_size: usize,
        outline_color: &str,
        outline_thickness: usize,
        source: Option<&SourceOverlay>,
    ) -> PyResult<Self> {
        let mut grid_rgba = render_grid_rgba(
            presampled,
            color_by,
            outline,
            colors,
            outline_color,
            outline_thickness,
        );
        if let Some(src) = source {
            draw_source_overlay(&mut grid_rgba, presampled.ph, presampled.pv, src);
        }
        let (png_rgba, png_w, png_h) = render_png_rgba(
            &grid_rgba,
            presampled.ph,
            presampled.pv,
            origin,
            width,
            basis,
            axis_units,
            font_size,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyInteractivePlot {
            html,
            png_rgba,
            png_w,
            png_h,
            ph: presampled.ph,
            pv: presampled.pv,
        })
    }
}
