//! Shared interactive-plot HTML/JS templating used by both:
//!
//! 1. `yamc-python::PyInteractivePlot` -- the Python `model.plot()` /
//!    `geometry.plot()` / `cell.plot()` family. Calls
//!    [`build_interactive_html`] and ships the result through Jupyter's
//!    `_repr_html_` (or [`PyInteractivePlot::save`] writes to disk).
//! 2. `yamc::wasm::WasmSimulation` -- the browser-side editor exposes
//!    wasm-bindgen entries that return raw slice samples; the JS host
//!    mounts the viewer code (extracted from this crate) and refreshes
//!    via those samples on each Apply.
//!
//! The pyo3-using bits (`PyInteractivePlot`, the `parse_*_kwargs`
//! helpers) stay in `yamc-python` because they need PyDict extraction
//! and Python-side error types. Everything else -- the templating, the
//! PNG/RGBA rasterizer, the bitmap-font axis labels, the JS source --
//! lives here so both consumers compile against the same source of truth.

pub mod surfaces;
pub mod tally_html;
pub mod viewer_html;

pub use surfaces::{build_surface_table, collect_region_surfaces, SurfaceTableEntry};
pub use tally_html::{
    build_interactive_tally_html, EmbeddedSlice, ExtraValueSlices, InteractiveTallyParams, MeshMeta,
};
pub use viewer_html::{
    build_interactive_html, draw_source_overlay, escape_js, parse_hex_color, render_grid_rgba,
    render_png_rgba, ContourParams, InteractiveViewParams, PresampledGrid, SourceOverlay,
    SourceParams, DISCRETE_COLORS,
};
