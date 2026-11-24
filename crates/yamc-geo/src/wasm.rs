use wasm_bindgen::prelude::*;

use crate::geometry::CsgGeometry;
use crate::mesh::GeoMesh;
use crate::plot::PlotParams;

enum PlotterKind {
    Csg(CsgGeometry),
    Mesh(GeoMesh),
}

/// WASM-accessible geometry plotter.
///
/// Loads CSG or mesh geometry from JSON and samples 2D grids for
/// interactive visualization in the browser.
#[wasm_bindgen]
pub struct WasmPlotter {
    inner: PlotterKind,
}

#[wasm_bindgen]
impl WasmPlotter {
    /// Create a new WasmPlotter from JSON geometry data.
    ///
    /// - `json`: serialized CsgGeometry or GeoMesh
    /// - `kind`: `"csg"` or `"mesh"`
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str, kind: &str) -> Result<WasmPlotter, JsValue> {
        console_error_panic_hook::set_once();

        let inner = match kind {
            "csg" => {
                let geo: CsgGeometry = serde_json::from_str(json)
                    .map_err(|e| JsValue::from_str(&format!("Failed to parse CSG JSON: {e}")))?;
                PlotterKind::Csg(geo)
            }
            "mesh" => {
                let geo: GeoMesh = serde_json::from_str(json)
                    .map_err(|e| JsValue::from_str(&format!("Failed to parse mesh JSON: {e}")))?;
                PlotterKind::Mesh(geo)
            }
            _ => return Err(JsValue::from_str("kind must be 'csg' or 'mesh'")),
        };

        Ok(WasmPlotter { inner })
    }

    /// Sample a 2D grid and return interleaved [cell_id, mat_id, cell_id, mat_id, ...]
    ///
    /// Returns a flat i32 array of length `pixels_h * pixels_v * 2`.
    // The arguments are flat scalars because this is a wasm-bindgen boundary:
    // a Rust struct cannot cross it without a JS-side counterpart, so the
    // grouping happens on the far side and again in PlotParams just below.
    // Collapsing them to satisfy the lint would change the JS API.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = sampleGrid)]
    pub fn sample_grid(
        &self,
        basis: &str,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        width_h: f64,
        width_v: f64,
        pixels_h: u32,
        pixels_v: u32,
    ) -> Result<Vec<i32>, JsValue> {
        let params = PlotParams {
            origin: (origin_x, origin_y, origin_z),
            width: (width_h, width_v),
            pixels: (pixels_h as usize, pixels_v as usize),
            basis: basis.to_string(),
        };

        let grid = match &self.inner {
            PlotterKind::Csg(geo) => geo
                .sample_grid(&params)
                .map_err(|e| JsValue::from_str(&e))?,
            PlotterKind::Mesh(geo) => geo
                .sample_grid(&params)
                .map_err(|e| JsValue::from_str(&e))?,
        };

        // Interleave cell_ids and material_ids into a flat array
        let ph = pixels_h as usize;
        let pv = pixels_v as usize;
        let mut result = Vec::with_capacity(ph * pv * 2);
        for row_idx in 0..pv {
            for col_idx in 0..ph {
                result.push(grid.cell_ids[row_idx][col_idx]);
                result.push(grid.material_ids[row_idx][col_idx]);
            }
        }

        Ok(result)
    }

    /// Get the bounding box as [cx, cy, cz, wx, wy, wz]
    #[wasm_bindgen(js_name = boundingBox)]
    pub fn bounding_box(&self) -> Vec<f64> {
        let bb = match &self.inner {
            PlotterKind::Csg(geo) => geo.bounding_box(),
            PlotterKind::Mesh(geo) => geo.bounding_box(),
        };
        let center = bb.center();
        let width = bb.width();
        vec![
            center[0], center[1], center[2], width[0], width[1], width[2],
        ]
    }
}
