//! `WasmSimulation` -- the wasm-bindgen entry point for running the
//! transport loop from JavaScript.
//!
//! ## Flow
//!
//! 1. `new WasmSimulation()` -- installs an in-memory `Storage` backend
//!    as the process-global `yamc_nuclide` storage.
//! 2. `sim.load_model_json(json)` -- deserializes a `Model` from a JSON
//!    string (the one produced by Python's `model.save(path)` or
//!    `model.export(html)`). The model fully describes geometry,
//!    materials, source, and tallies; the wasm side just executes it.
//! 3. `sim.add_file("/<Nuclide>.arrow/<file>", bytes)` once per option-D
//!    section object the JS host fetched under `<Nuclide>.arrow/`
//!    (nuclide.arrow, reactions.arrow, version.json, ...). No tar: each
//!    section is fetched and registered directly. The lz4-compressed
//!    sections decode transparently in the arrow reader. The model's
//!    materials reference these paths as `/<Nuclide>.arrow`.
//! 4. `sim.simulate_transport(particles, batches, seed)` -- runs the
//!    loaded model. Returns JSON: `{status, tritium_mean, tritium_std,
//!    particles, batches, seed}` on success, `{status: "error",
//!    message}` on failure.
//!
//! `particles` / `batches` / `seed` passed to `simulate_transport`
//! override whatever the model JSON had; the engineer's exported model
//! supplies the geometry and tally definitions, while the recipient
//! controls run size.
//!
//! ## Single-instance assumption
//!
//! `yamc_nuclide::storage::set_storage` swaps a process-global backend,
//! so constructing a second `WasmSimulation` would wipe the first one's
//! nuclide data. JS hosts should keep one instance.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use wasm_bindgen::prelude::*;

use yamc_nuclide::in_memory_storage::InMemoryStorage;
use yamc_nuclide::storage;

use crate::geometry::backend::GeometryKind;
use crate::model::Model;

/// Browser-facing handle. Owns the in-memory `Storage` backend plus the
/// currently-loaded model (if any).
#[wasm_bindgen]
pub struct WasmSimulation {
    /// Cheaply cloneable; the process-global `Storage` holds a twin
    /// pointing at the same backing store.
    storage: InMemoryStorage,
    /// `None` until [`WasmSimulation::load_model_json`] is called.
    model: Arc<RwLock<Option<Model>>>,
}

#[wasm_bindgen]
impl WasmSimulation {
    /// Construct a new simulation. Installs a fresh in-memory `Storage`
    /// backend as the process-global `yamc_nuclide` storage.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let storage = InMemoryStorage::new();
        storage::set_storage(Box::new(storage.clone()));
        WasmSimulation {
            storage,
            model: Arc::new(RwLock::new(None)),
        }
    }

    // --- File / storage API ---

    /// Add a single virtual file at `path`. Called once per option-D
    /// section object under a nuclide's `.arrow/` prefix before
    /// `simulate_transport`.
    #[wasm_bindgen]
    pub fn add_file(&self, path: String, bytes: Vec<u8>) {
        self.storage.add_file(path, bytes);
    }

    /// Number of files currently held in the in-memory backend.
    #[wasm_bindgen]
    pub fn file_count(&self) -> usize {
        self.storage.len()
    }

    // --- Model API ---

    /// Load a model from its JSON representation (as produced by Python's
    /// `model.save(path)` / `model.export(html)`). Replaces any previously
    /// loaded model. Returns an error string on parse failure.
    #[wasm_bindgen]
    pub fn load_model_json(&self, json: String) -> Result<(), String> {
        let model: Model =
            serde_json::from_str(&json).map_err(|e| format!("model JSON parse failed: {e}"))?;
        *self.model.write().unwrap_or_else(|p| p.into_inner()) = Some(model);
        Ok(())
    }

    /// Diagnostic: list of nuclide names referenced by the loaded model's
    /// materials. JS uses this to know which `<Nuclide>.arrow/` section
    /// sets to fetch. Returns a comma-joined string; empty if no model is
    /// loaded.
    #[wasm_bindgen]
    pub fn model_required_nuclides(&self) -> String {
        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let Some(model) = model.as_ref() else {
            return String::new();
        };
        model.required_nuclides().join(",")
    }

    /// Element symbols the loaded model needs *photon* data for, comma-
    /// joined. Empty when the model has no photons in flight (no photon
    /// source and no secondary-photon production), or no model loaded.
    /// Photon data is per-element (`Fe`, not `Fe56`); the JS host fetches
    /// the option-D section objects under `endf-b8.1/photon/<El>.arrow/`
    /// for each and registers the files under `/<El>.arrow/`, mirroring the
    /// neutron convention.
    #[wasm_bindgen]
    pub fn model_required_elements(&self) -> String {
        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let Some(model) = model.as_ref() else {
            return String::new();
        };
        model.required_elements().join(",")
    }

    // --- Geometry plotting API (used by the in-browser viewer) ---
    //
    // These four entries let the JS host drive the same interactive plot
    // the Python `model.plot()` flow produces, refreshed from the *currently
    // loaded* model. The viewer JS expects raw slice data + bounding box +
    // CsgGeometry JSON; this returns exactly those shapes so the shared
    // `yamc-plot::VIEWER_JS` can mount once and refresh on every Apply.

    /// Sample a 2D slice through the loaded model's geometry.
    ///
    /// `params_json` accepts:
    /// ```json
    /// { "origin":  [ox, oy, oz],
    ///   "width":   [wh, wv],
    ///   "pixels":  [ph, pv],
    ///   "basis":   "xy" | "xz" | "yz" }
    /// ```
    ///
    /// Returns an interleaved `[cell_id, mat_id, cell_id, mat_id, ...]`
    /// flat `Vec<i32>` of length `ph * pv * 2`, rows top-to-bottom. `-1`
    /// for void cells / material-less cells. Matches the wire format the
    /// inline `JsCsgPlotter.sampleGrid(...)` produces, so the same
    /// viewer JS consumes both.
    #[wasm_bindgen(js_name = sampleSlice)]
    pub fn sample_slice(&self, params_json: String) -> Result<Vec<i32>, String> {
        #[derive(serde::Deserialize)]
        struct Params {
            origin: [f64; 3],
            width: [f64; 2],
            pixels: [usize; 2],
            basis: String,
        }
        let p: Params = serde_json::from_str(&params_json)
            .map_err(|e| format!("sampleSlice params JSON: {e}"))?;

        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded -- call load_model_json first".to_string())?;
        let geom = match &model.geometry {
            GeometryKind::Csg(g) => g,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => {
                return Err("mesh-geometry plotting is not yet supported in the wasm path".into())
            }
        };

        let (cells, mats) = geom.sample_slice(
            (p.origin[0], p.origin[1], p.origin[2]),
            (p.width[0], p.width[1]),
            (p.pixels[0], p.pixels[1]),
            &p.basis,
        );

        // Flatten to [cell, mat, cell, mat, ...] row-by-row. Rows come back
        // bottom-to-top from sample_slice; the viewer JS expects top-to-bottom,
        // so reverse the outer iteration (matches `to_presampled_grid` in
        // yamc-python).
        let ph = p.pixels[0];
        let pv = p.pixels[1];
        let mut out = Vec::with_capacity(ph * pv * 2);
        for row in (0..pv).rev() {
            let crow = &cells[row];
            let mrow = &mats[row];
            for col in 0..ph {
                out.push(crow[col]);
                out.push(mrow[col]);
            }
        }
        Ok(out)
    }

    /// Bounding box of the loaded geometry as `[cx, cy, cz, wx, wy, wz]`
    /// (center + widths in cm). Used by the JS viewer to seed initial view
    /// origin/width and to scale wheel-zoom step size.
    #[wasm_bindgen(js_name = boundingBox)]
    pub fn bounding_box(&self) -> Result<Vec<f64>, String> {
        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let geom = match &model.geometry {
            GeometryKind::Csg(g) => g,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => return Err("mesh boundingBox not supported".into()),
        };
        let bb = geom.bounding_box();
        let cx = 0.5 * (bb.lower_left[0] + bb.upper_right[0]);
        let cy = 0.5 * (bb.lower_left[1] + bb.upper_right[1]);
        let cz = 0.5 * (bb.lower_left[2] + bb.upper_right[2]);
        let wx = bb.upper_right[0] - bb.lower_left[0];
        let wy = bb.upper_right[1] - bb.lower_left[1];
        let wz = bb.upper_right[2] - bb.lower_left[2];
        Ok(vec![cx, cy, cz, wx, wy, wz])
    }

    /// CsgGeometry JSON for the *visualization* -- same shape Python's
    /// `model.plot()` embeds as `GEOMETRY_JSON`. Lets the JS viewer build
    /// its cell/material name maps + legend from the currently loaded
    /// model. Errors if no model is loaded or the model is mesh-backed.
    #[wasm_bindgen(js_name = geometryJson)]
    pub fn geometry_json(&self) -> Result<String, String> {
        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let geom = match &model.geometry {
            GeometryKind::Csg(g) => g,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => return Err("mesh geometryJson not supported".into()),
        };
        let csg = crate::geometry::conversion::geometry_to_csg(geom);
        serde_json::to_string(&csg).map_err(|e| format!("geometry_to_csg serialize: {e}"))
    }

    /// Build the full interactive-viewer HTML for the currently loaded
    /// model -- same shape Python's `model.plot()._repr_html_()` produces.
    /// The editor mounts this in an iframe; on every Apply it re-calls
    /// `plotHtml` and rewrites `iframe.srcdoc` so the viewer reflects the
    /// post-edit geometry/source.
    ///
    /// `params_json` accepts an optional initial view spec -- same keys as
    /// `sampleSlice` (`origin`, `width`, `pixels`, `basis`). Empty / "{}"
    /// auto-fits to the model's bounding box.
    #[wasm_bindgen(js_name = plotHtml)]
    pub fn plot_html(&self, params_json: String) -> Result<String, String> {
        #[derive(serde::Deserialize, Default)]
        struct Params {
            origin: Option<[f64; 3]>,
            width: Option<[f64; 2]>,
            pixels: Option<[usize; 2]>,
            basis: Option<String>,
        }
        let p: Params = if params_json.trim().is_empty() {
            Params::default()
        } else {
            serde_json::from_str(&params_json).map_err(|e| format!("plotHtml params: {e}"))?
        };

        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let geom = match &model.geometry {
            GeometryKind::Csg(g) => g,
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => return Err("mesh plotHtml not supported".into()),
        };
        let csg = crate::geometry::conversion::geometry_to_csg(geom);
        let geometry_json =
            serde_json::to_string(&csg).map_err(|e| format!("geometry_to_csg serialize: {e}"))?;

        // Auto-fit view if caller passed nothing. Matches the defaults
        // Python's `model.plot()` uses (see yamc-python::id_map_helper::
        // parse_sample_slice_params): center on bbox.center, width on
        // bbox.width along the chosen basis, 40 000 total pixels with
        // aspect ratio derived from width.
        let bb = geom.bounding_box();
        let bb_width = bb.width();
        let wx = bb_width[0].max(1.0);
        let wy = bb_width[1].max(1.0);
        let wz = bb_width[2].max(1.0);
        let basis = p.basis.unwrap_or_else(|| "xy".to_string());
        let origin = p.origin.unwrap_or_else(|| bb.center());
        let (def_h, def_v) = match basis.as_str() {
            "xz" => (wx, wz),
            "yz" => (wy, wz),
            _ => (wx, wy),
        };
        let width = p.width.unwrap_or([def_h, def_v]);
        let pixels = p.pixels.unwrap_or_else(|| {
            // 40 000 default total, split by aspect ratio (same as Python).
            let aspect = width[0] / width[1];
            let pv = (40_000.0 / aspect).sqrt();
            let ph = (40_000.0 / pv) as usize;
            [ph, pv as usize]
        });
        let total_pixels = pixels[0] * pixels[1];

        // Build cell + material name maps the same way Python does:
        // iterate the CsgGeometry's cells (NOT geom.materials) so an
        // unreferenced material doesn't sneak into the legend, and the
        // material_name carries through cell_to_geo_cell's lookup logic.
        let mut cell_names = std::collections::HashMap::new();
        let mut material_names = std::collections::HashMap::new();
        for cell in &csg.cells {
            if let Some(ref n) = cell.name {
                cell_names.insert(cell.cell_id, n.clone());
            }
            if cell.material_id != -1 {
                if let Some(ref name) = cell.material_name {
                    material_names.insert(cell.material_id, name.clone());
                }
            }
        }

        // Optional source-distribution JSON -- viewer.js draws the source
        // overlay when this is non-null. Shape must match what the Python
        // `model.plot()` path emits (see yamc-python::model::plot):
        //   {"sources": [{"strength": ..., "spatial": ..., "energy": ...}, ...]}
        // Anything else and the viewer's inline sampler returns no points,
        // which is why "I don't see the source" was the first symptom.
        let source_dist_json = if !model.sources.is_empty() {
            let entries: Vec<serde_json::Value> = model
                .sources
                .iter()
                .map(|ps| {
                    let src = ps.source();
                    serde_json::json!({
                        "strength": src.strength,
                        "spatial": &src.space,
                        "energy": &src.energy,
                    })
                })
                .collect();
            serde_json::to_string(&serde_json::json!({ "sources": entries })).ok()
        } else {
            None
        };

        let params = yamc_plot::InteractiveViewParams {
            origin: (origin[0], origin[1], origin[2]),
            width: (width[0], width[1]),
            total_pixels,
            basis: basis.clone(),
            // Match Python `model.plot()`'s default (color_by="cell").
            // For the editor's two-cell demos "material" looks identical,
            // but a single-material multi-cell model is only visually
            // distinct under "cell".
            color_by: "cell".into(),
            outline: Some("cell".into()),
            axis_units: "cm".into(),
            bbox_widths: [wx, wy, wz],
            outline_color: "#000000".into(),
            outline_thickness: 1,
            // Editor specifically wants the source visible by default --
            // Python's `model.plot()` defaults to None (hidden), but a
            // recipient opening an exported HTML needs to see the source
            // to understand the geometry. The checkbox is still toggleable.
            n_samples: if model.sources.is_empty() {
                None
            } else {
                Some(1000)
            },
            plane_tolerance: 1.0,
            source_color: "#ff0000".into(),
            source_size: 3,
        };

        let surface_table = yamc_plot::build_surface_table(&csg);

        // Pre-sample the initial view so the viewer can paint the
        // colormap immediately instead of running its JS per-pixel
        // cell-finder on first render. The grid is base64-i32-pairs
        // (`grid.data.len() = ph * pv * 2` ints), so for the default
        // 200×200 view it's ~320 KB inlined in the HTML. Big save in
        // time-to-first-paint for multi-cell / many-surface models.
        let (cells, mats) = geom.sample_slice(
            (origin[0], origin[1], origin[2]),
            (width[0], width[1]),
            (pixels[0], pixels[1]),
            &basis,
        );
        let mut presampled_data = Vec::with_capacity(pixels[0] * pixels[1] * 2);
        // sample_slice rows are bottom-to-top; viewer expects top-to-bottom.
        for row in (0..pixels[1]).rev() {
            for col in 0..pixels[0] {
                presampled_data.push(cells[row][col]);
                presampled_data.push(mats[row][col]);
            }
        }
        let presampled = yamc_plot::PresampledGrid {
            data: presampled_data,
            ph: pixels[0],
            pv: pixels[1],
        };

        Ok(yamc_plot::build_interactive_html(
            &geometry_json,
            "csg",
            "",
            &params,
            &cell_names,
            &material_names,
            None,
            source_dist_json.as_deref(),
            Some(&presampled),
            // The wasm path rejects mesh-geometry models and cannot load a
            // filled model at all (fills do not round-trip from JSON), so there
            // is never a fill to render here (issue #291).
            false,
            Some(&surface_table),
        ))
    }

    /// Build an interactive mesh-tally plot HTML from the currently
    /// loaded model's *current* tally state. Used by the editor's
    /// Simulate handler to refresh the tally-plot iframe after each
    /// in-browser run.
    ///
    /// Args (all in `params_json`):
    /// - `tally_index` (usize): which tally to plot (default 0)
    /// - `basis` ("xy" | "xz" | "yz", default "xz")
    /// - `score_index` (usize, default 0)
    /// - `outline` ("cell" | "material" | "none", default "cell")
    ///
    /// Returns the full self-contained interactive tally viewer HTML --
    /// same shape Python's `tally.plot()._repr_html_()` produces.
    /// Errors if the tally has no MeshFilter or no model is loaded.
    #[wasm_bindgen(js_name = tallyPlotHtml)]
    pub fn tally_plot_html(&self, params_json: String) -> Result<String, String> {
        #[derive(serde::Deserialize, Default)]
        struct Params {
            #[serde(default)]
            tally_index: usize,
            #[serde(default)]
            basis: Option<String>,
            #[serde(default)]
            score_index: usize,
            #[serde(default)]
            outline: Option<String>,
        }
        let p: Params = if params_json.trim().is_empty() {
            Params::default()
        } else {
            serde_json::from_str(&params_json).map_err(|e| format!("tallyPlotHtml params: {e}"))?
        };
        let basis = p.basis.unwrap_or_else(|| "xz".to_string());
        let outline = p.outline.unwrap_or_else(|| "cell".to_string());

        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let tally = model
            .tallies
            .get(p.tally_index)
            .ok_or_else(|| format!("tally_index {} out of range", p.tally_index))?;
        let mesh_filter = tally
            .get_mesh_filter()
            .ok_or_else(|| format!("tally {} has no MeshFilter", p.tally_index))?;
        let mesh = match mesh_filter.kind() {
            yamc_tallies::MeshKind::Rectangular(m) => m,
            yamc_tallies::MeshKind::Cylindrical(_) => {
                return Err(format!(
                    "tally {} uses a cylindrical mesh, which tallyPlotHtml does not support",
                    p.tally_index
                ));
            }
        };
        let ll = mesh.lower_left();
        let ur = mesh.upper_right();
        let dim = mesh.shape();
        let width = mesh.width();

        let fixed_axis = match basis.as_str() {
            "xy" => 2,
            "xz" => 1,
            "yz" => 0,
            other => return Err(format!("invalid basis '{other}'")),
        };
        let fixed_dim = dim[fixed_axis];
        // Embed every slice along the perpendicular axis -- same default
        // Python's tally.plot(slices="all") uses.
        let slice_indices: Vec<usize> = (0..fixed_dim).collect();

        let mean_slices = tally
            .extract_mesh_slices(&basis, &slice_indices, p.score_index, None, "mean")
            .map_err(|e| format!("extract_mesh_slices(mean): {e}"))?;
        let std_slices = tally
            .extract_mesh_slices(
                &basis,
                &slice_indices,
                p.score_index,
                None,
                "standard_deviation",
            )
            .map_err(|e| format!("extract_mesh_slices(std): {e}"))?;
        let rel_slices = tally
            .extract_mesh_slices(
                &basis,
                &slice_indices,
                p.score_index,
                None,
                "relative_error",
            )
            .map_err(|e| format!("extract_mesh_slices(rel): {e}"))?;

        let embedded_slices: Vec<yamc_plot::EmbeddedSlice> = mean_slices
            .into_iter()
            .map(|(bin_index, data)| yamc_plot::EmbeddedSlice {
                basis: basis.clone(),
                bin_index,
                data,
            })
            .collect();
        let extra_values = vec![
            yamc_plot::ExtraValueSlices {
                value_name: "standard_deviation".into(),
                slices: std_slices
                    .into_iter()
                    .map(|(b, d)| yamc_plot::EmbeddedSlice {
                        basis: basis.clone(),
                        bin_index: b,
                        data: d,
                    })
                    .collect(),
            },
            yamc_plot::ExtraValueSlices {
                value_name: "relative_error".into(),
                slices: rel_slices
                    .into_iter()
                    .map(|(b, d)| yamc_plot::EmbeddedSlice {
                        basis: basis.clone(),
                        bin_index: b,
                        data: d,
                    })
                    .collect(),
            },
        ];

        let mesh_meta = yamc_plot::MeshMeta {
            lower_left: ll,
            upper_right: ur,
            shape: dim,
            width,
        };

        // Geometry overlay -- same plumbing plotHtml uses.
        let (geometry_json, cell_names, material_names, surface_table) = match &model.geometry {
            GeometryKind::Csg(g) => {
                let csg = crate::geometry::conversion::geometry_to_csg(g);
                let mut cn = std::collections::HashMap::new();
                let mut mn = std::collections::HashMap::new();
                for cell in &csg.cells {
                    if let Some(ref n) = cell.name {
                        cn.insert(cell.cell_id, n.clone());
                    }
                    if cell.material_id != -1 {
                        if let Some(ref name) = cell.material_name {
                            mn.insert(cell.material_id, name.clone());
                        }
                    }
                }
                let table = yamc_plot::build_surface_table(&csg);
                (
                    Some(serde_json::to_string(&csg).unwrap_or_default()),
                    cn,
                    mn,
                    Some(table),
                )
            }
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => {
                return Err("mesh-geometry tally overlay not supported in wasm yet".into())
            }
        };

        let initial_slice_index = slice_indices[0].min(fixed_dim - 1);
        let tally_params = yamc_plot::InteractiveTallyParams {
            initial_basis: basis.clone(),
            initial_slice_index,
            colorscale: "Viridis".into(),
            log_scale: true,
            outline: if outline == "none" {
                None
            } else {
                Some(outline)
            },
            outline_color: "#000000".into(),
            outline_thickness: 1,
            axis_units: "cm".into(),
            outline_pixels: 40_000,
            title: tally.name.clone().unwrap_or_else(|| "tally".to_string()),
            colorbar_title: "value".into(),
            scaling_factor: 1.0,
            font_size: 18,
            show_colorbar: true,
            display_value: "mean".into(),
        };

        Ok(yamc_plot::build_interactive_tally_html(
            geometry_json.as_deref(),
            Some("csg"),
            &mesh_meta,
            &embedded_slices,
            &extra_values,
            &tally_params,
            &cell_names,
            &material_names,
            None,
            surface_table.as_deref(),
        ))
    }

    /// Sample `n` source-particle positions using the loaded model's source
    /// distribution. Returns a flat `[x0,y0,z0, x1,y1,z1, ...]` of length
    /// `n * 3`. Drives the JS viewer's source-overlay dot cloud -- using the
    /// real Rust sampler avoids the drift the inline JS sampler warns about.
    #[wasm_bindgen(js_name = sampleSourcePoints)]
    pub fn sample_source_points(&self, n: u32, seed: u64) -> Result<Vec<f64>, String> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let model = self.model.read().unwrap_or_else(|p| p.into_inner());
        let model = model
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n as usize * 3);
        let selector = yamc_source::source::SourceSelector::new(&model.sources);
        for _ in 0..n {
            let particle = model.sample_source_with(&selector, &mut rng);
            out.push(particle.position[0]);
            out.push(particle.position[1]);
            out.push(particle.position[2]);
        }
        Ok(out)
    }

    /// Run the loaded model. `particles`, `batches`, `seed` override the
    /// model's defaults. Returns a JSON result string with per-tally,
    /// per-score breakdown:
    ///
    /// ```json
    /// {
    ///   "status": "ok",
    ///   "particles": ..., "batches": ..., "seed": ...,
    ///   "tallies": [
    ///     {
    ///       "name": "tritium",
    ///       "scores": [{ "name": "105", "mean": ..., "std": ..., "fom": ... }, ...]
    ///     },
    ///     ...
    ///   ]
    /// }
    /// ```
    ///
    /// On failure: `{"status":"error","message":...}`.
    #[wasm_bindgen]
    pub fn simulate_transport(&self, particles: u32, batches: u32, seed: u64) -> String {
        match self.run(particles as usize, batches as usize, seed) {
            Ok(tallies) => {
                // Hand-build JSON to keep this allocation-light. The Tally /
                // Score names are well-formed strings; we escape via serde_json
                // for safety against odd characters.
                let mut parts = String::new();
                parts.push_str(&format!(
                    r#"{{"status":"ok","particles":{particles},"batches":{batches},"seed":{seed},"tallies":["#
                ));
                for (ti, t) in tallies.iter().enumerate() {
                    if ti > 0 {
                        parts.push(',');
                    }
                    let name_json = serde_json::to_string(&t.name).unwrap();
                    parts.push_str(&format!(r#"{{"name":{name_json},"scores":["#));
                    for (si, s) in t.scores.iter().enumerate() {
                        if si > 0 {
                            parts.push(',');
                        }
                        let score_name_json = serde_json::to_string(&s.name).unwrap();
                        parts.push_str(&format!(
                            r#"{{"name":{},"mean":{:.6e},"std":{:.6e},"fom":{:.6e}}}"#,
                            score_name_json, s.mean, s.std, s.fom
                        ));
                    }
                    parts.push_str("]}");
                }
                parts.push_str("]}");
                parts
            }
            Err(e) => {
                let msg = serde_json::to_string(&e).unwrap_or_else(|_| "\"<unprintable>\"".into());
                format!(r#"{{"status":"error","message":{msg}}}"#)
            }
        }
    }
}

/// Per-score result row carried out of [`WasmSimulation::run`].
struct ScoreResult {
    name: String,
    mean: f64,
    std: f64,
    /// Figure of merit `1 / (rel_err² × elapsed_secs)`; 0 when the score has
    /// no signal or the run was too fast to time.
    fom: f64,
}

/// Per-tally result, with one [`ScoreResult`] per score in the tally.
struct TallyResult {
    name: Option<String>,
    scores: Vec<ScoreResult>,
}

impl WasmSimulation {
    /// Internal helper -- not exported to JS.
    ///
    /// Pulls the loaded model, clones it (so each run starts with fresh
    /// tally state), loads nuclear data from the in-memory `Storage` for
    /// each material, applies the JS-supplied `particles` / `batches` /
    /// `seed`, runs `simulate_transport`, then returns per-tally and
    /// per-score means + stds.
    fn run(&self, particles: usize, batches: usize, seed: u64) -> Result<Vec<TallyResult>, String> {
        let mut model = {
            let guard = self.model.read().unwrap_or_else(|p| p.into_inner());
            guard
                .as_ref()
                .ok_or_else(|| "no model loaded -- call load_model_json first".to_string())?
                .clone()
        };

        // Run parameters come from the caller, not the loaded model JSON.
        let settings = crate::model::TransportSettings {
            total_particles: Some(particles * batches),
            seed,
            // Single-threaded: rayon thread pools aren't available on wasm.
            threads: Some(1),
            max_runtime: None,
        };

        // Captured before the &mut borrow of `geometry` below. Photon data
        // is needed whenever photons will be in flight -- a photon source
        // counts even with transport_secondary_photons off.
        let photon_on = model.has_photons();

        // For each material in the geometry, re-load nuclide data from
        // the in-memory storage. The path convention is `/<Name>.arrow`
        // matching what the JS host registered via `add_file`.
        match &mut model.geometry {
            GeometryKind::Csg(geom) => {
                let mut new_materials = Vec::with_capacity(geom.materials.len());
                for mat_arc in &geom.materials {
                    let mut mat = (**mat_arc).clone();
                    let nuclide_map: HashMap<String, String> = mat
                        .nuclides
                        .keys()
                        .map(|name| (name.clone(), format!("/{name}.arrow")))
                        .collect();
                    // Photon transport needs per-element photon data. Mirror
                    // the neutron convention: files live at /<El>.arrow/<file>,
                    // so point each element at that virtual directory.
                    let photon_map: Option<HashMap<String, String>> = if photon_on {
                        let mut m = HashMap::new();
                        for nuc in mat.nuclides.keys() {
                            let el: String =
                                nuc.chars().take_while(|c| c.is_alphabetic()).collect();
                            m.entry(el.clone())
                                .or_insert_with(|| format!("/{el}.arrow"));
                        }
                        Some(m)
                    } else {
                        None
                    };
                    mat.read_nuclear_data(&nuclide_map, photon_map.as_ref())
                        .map_err(|e| format!("read_nuclear_data: {e}"))?;
                    new_materials.push(Arc::new(mat));
                }
                geom.materials = new_materials;
            }
            #[cfg(feature = "mesh")]
            GeometryKind::Mesh(_) => {
                return Err("mesh-geometry models are not yet supported in the wasm path".into());
            }
        }

        // Tallies need their batch storage initialised against the
        // (possibly overridden) batch count.
        for tally in &model.tallies {
            tally.initialize_batches_shared(batches);
        }

        // Wall-clock around transport, for the figure of merit. The shared
        // `Timer` is a 0.0 stub under wasm32, so measure the elapsed time
        // directly here. On wasm32 use the JS millisecond clock; on a
        // native target (the `--features wasm-test` integration test) use
        // `std::time::Instant` -- calling `js_sys::Date::now()` natively
        // panics ("cannot call wasm-bindgen imported functions on non-wasm
        // targets").
        #[cfg(target_arch = "wasm32")]
        let t0 = js_sys::Date::now();
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();

        model.simulate_transport(&settings)?;

        #[cfg(target_arch = "wasm32")]
        let elapsed_secs = (js_sys::Date::now() - t0) / 1000.0;
        #[cfg(not(target_arch = "wasm32"))]
        let elapsed_secs = t0.elapsed().as_secs_f64();

        let mut results = Vec::with_capacity(model.tallies.len());
        for tally in &model.tallies {
            let means = tally.get_mean();
            let stds = tally.get_std_dev();
            // `means.len()` == `stds.len()` == `tally.scores.len()` by
            // construction of the accumulator.
            let scores = tally
                .scores
                .iter()
                .zip(means.iter().copied().chain(std::iter::repeat(0.0)))
                .zip(stds.iter().copied().chain(std::iter::repeat(0.0)))
                .map(|((score, mean), std)| {
                    let fom = if mean > 0.0 && std > 0.0 && elapsed_secs > 0.0 {
                        let re = std / mean;
                        1.0 / (re * re * elapsed_secs)
                    } else {
                        0.0
                    };
                    ScoreResult {
                        name: score.name(),
                        mean,
                        std,
                        fom,
                    }
                })
                .take(tally.scores.len())
                .collect();
            results.push(TallyResult {
                name: tally.name.clone(),
                scores,
            });
        }
        Ok(results)
    }
}

impl Default for WasmSimulation {
    fn default() -> Self {
        Self::new()
    }
}
