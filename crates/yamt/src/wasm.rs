//! WASM entry point for mesh geometry point-in-volume queries.
//!
//! Accepts the same GeoMesh JSON that the interactive viewer produces
//! (volumes with pre-expanded triangle coordinates), builds a BVH for
//! each volume, and exposes `find_volume` for fast pixel sampling.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::Deserialize;

#[cfg(feature = "wasm")]
use crate::accel::bvh::Bvh;

#[cfg(feature = "wasm")]
use crate::query::intersect;

#[cfg(feature = "wasm")]
use crate::query::point_in_volume;

// ---------------------------------------------------------------------------
// JSON-compatible input types (match yamc-geo GeoMesh serialization)
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
#[derive(Deserialize)]
struct GeoMeshJson {
    volumes: Vec<GeoMeshVolumeJson>,
    global_min: [f64; 3],
    global_max: [f64; 3],
}

#[cfg(feature = "wasm")]
#[derive(Deserialize)]
struct GeoMeshVolumeJson {
    cell_id: i32,
    material_id: i32,
    triangles: Vec<[[f64; 3]; 3]>,
}

// ---------------------------------------------------------------------------
// Per-volume BVH data (mirrors SurfaceBvhData from geometry.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
struct VolumeBvhData {
    bvh: Bvh,
    v0: Vec<[f64; 3]>,
    edge1: Vec<[f64; 3]>,
    edge2: Vec<[f64; 3]>,
    cell_id: i32,
    material_id: i32,
    aabb: [f64; 6], // [min_x, min_y, min_z, max_x, max_y, max_z]
}

// ---------------------------------------------------------------------------
// WASM-exported plotter
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmMeshPlotter {
    volumes: Vec<VolumeBvhData>,
    global_min: [f64; 3],
    global_max: [f64; 3],
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmMeshPlotter {
    /// Construct from GeoMesh JSON. Builds a BVH for each volume.
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<WasmMeshPlotter, JsValue> {
        let geo: GeoMeshJson =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut volumes = Vec::with_capacity(geo.volumes.len());
        for vol in &geo.volumes {
            let n = vol.triangles.len();
            let mut aabbs = Vec::with_capacity(n);
            let mut v0s = Vec::with_capacity(n);
            let mut edge1s = Vec::with_capacity(n);
            let mut edge2s = Vec::with_capacity(n);

            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];

            for tri in &vol.triangles {
                let va = tri[0];
                let vb = tri[1];
                let vc = tri[2];

                v0s.push(va);
                edge1s.push(intersect::sub(vb, va));
                edge2s.push(intersect::sub(vc, va));

                let tri_min = [
                    va[0].min(vb[0]).min(vc[0]),
                    va[1].min(vb[1]).min(vc[1]),
                    va[2].min(vb[2]).min(vc[2]),
                ];
                let tri_max = [
                    va[0].max(vb[0]).max(vc[0]),
                    va[1].max(vb[1]).max(vc[1]),
                    va[2].max(vb[2]).max(vc[2]),
                ];
                aabbs.push([
                    tri_min[0], tri_min[1], tri_min[2], tri_max[0], tri_max[1], tri_max[2],
                ]);

                for d in 0..3 {
                    if tri_min[d] < min[d] {
                        min[d] = tri_min[d];
                    }
                    if tri_max[d] > max[d] {
                        max[d] = tri_max[d];
                    }
                }
            }

            let bvh = Bvh::build(&aabbs);
            volumes.push(VolumeBvhData {
                bvh,
                v0: v0s,
                edge1: edge1s,
                edge2: edge2s,
                cell_id: vol.cell_id,
                material_id: vol.material_id,
                aabb: [min[0], min[1], min[2], max[0], max[1], max[2]],
            });
        }

        Ok(WasmMeshPlotter {
            volumes,
            global_min: geo.global_min,
            global_max: geo.global_max,
        })
    }

    /// Sample a grid of points and return interleaved [cell_id, material_id, ...].
    ///
    /// `basis`: "xy", "xz", or "yz"
    /// Returns an Int32Array of length `ph * pv * 2`.
    #[wasm_bindgen(js_name = sampleGrid)]
    #[allow(clippy::too_many_arguments)]
    pub fn sample_grid(
        &self,
        basis: &str,
        ox: f64,
        oy: f64,
        oz: f64,
        wh: f64,
        wv: f64,
        ph: u32,
        pv: u32,
    ) -> Vec<i32> {
        let n = (ph as usize) * (pv as usize) * 2;
        let mut data = vec![-1i32; n];
        let half_h = wh / 2.0;
        let half_v = wv / 2.0;

        for i in 0..pv {
            let v = if pv > 1 {
                half_v - wv * (i as f64) / ((pv - 1) as f64)
            } else {
                0.0
            };
            for j in 0..ph {
                let h = if ph > 1 {
                    -half_h + wh * (j as f64) / ((ph - 1) as f64)
                } else {
                    0.0
                };

                let (px, py, pz) = match basis {
                    "xz" => (ox + h, oy, oz + v),
                    "yz" => (ox, oy + h, oz + v),
                    _ => (ox + h, oy + v, oz), // "xy"
                };
                let point = [px, py, pz];

                let idx = ((i as usize) * (ph as usize) + (j as usize)) * 2;

                for vol in &self.volumes {
                    // Quick AABB rejection
                    if px < vol.aabb[0]
                        || px > vol.aabb[3]
                        || py < vol.aabb[1]
                        || py > vol.aabb[4]
                        || pz < vol.aabb[2]
                        || pz > vol.aabb[5]
                    {
                        continue;
                    }

                    if point_in_volume::point_in_volume(
                        &vol.bvh, &vol.v0, &vol.edge1, &vol.edge2, point,
                    ) {
                        data[idx] = vol.cell_id;
                        data[idx + 1] = vol.material_id;
                        break;
                    }
                }
            }
        }

        data
    }

    /// Return the global bounding box as [cx, cy, cz, wx, wy, wz].
    #[wasm_bindgen(js_name = boundingBox)]
    pub fn bounding_box(&self) -> Vec<f64> {
        let mn = self.global_min;
        let mx = self.global_max;
        vec![
            (mn[0] + mx[0]) / 2.0,
            (mn[1] + mx[1]) / 2.0,
            (mn[2] + mx[2]) / 2.0,
            mx[0] - mn[0],
            mx[1] - mn[1],
            mx[2] - mn[2],
        ]
    }
}
