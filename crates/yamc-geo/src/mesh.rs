use serde::{Deserialize, Serialize};

use crate::bounding_box::BoundingBox;
use crate::plot::{PlotGrid, PlotParams, PlotSample};

/// A single mesh volume with pre-expanded triangle vertex coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoMeshVolume {
    pub cell_id: i32,
    pub material_id: i32,
    pub material_name: Option<String>,
    /// Pre-expanded triangle vertex coordinates: each entry is [[x0,y0,z0],[x1,y1,z1],[x2,y2,z2]]
    pub triangles: Vec<[[f64; 3]; 3]>,
}

/// Lightweight mesh geometry for visualization -- serializable, no BVH.
///
/// Uses brute-force ray-casting for point-in-volume queries.
/// Suitable for small meshes; large meshes should use the full yamt backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoMesh {
    pub volumes: Vec<GeoMeshVolume>,
    pub global_min: [f64; 3],
    pub global_max: [f64; 3],
}

impl GeoMesh {
    /// Compute the bounding box
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox::new(self.global_min, self.global_max)
    }

    /// Test if a point is inside a volume using ray-casting (count triangle crossings).
    ///
    /// Fires a ray in a slightly perturbed direction (mostly +X) and counts how
    /// many triangles it crosses. Odd count = inside, even count = outside.
    /// The perturbation avoids hitting shared edges/vertices of adjacent triangles.
    pub fn point_in_volume(vol: &GeoMeshVolume, point: [f64; 3]) -> bool {
        // Slightly perturbed direction to avoid edge/vertex coincidence
        let dir = [1.0, 1.000000013e-5, 1.000000017e-6];
        let mut crossings = 0u32;
        for tri in &vol.triangles {
            if ray_intersects_triangle_dir(point, dir, tri) {
                crossings += 1;
            }
        }
        crossings % 2 == 1
    }

    /// Find the first volume containing the point, or None.
    pub fn find_volume(&self, point: [f64; 3]) -> Option<usize> {
        // Quick AABB check
        for (p, (mn, mx)) in point
            .iter()
            .zip(self.global_min.iter().zip(self.global_max.iter()))
        {
            if p < mn || p > mx {
                return None;
            }
        }
        for (idx, vol) in self.volumes.iter().enumerate() {
            if Self::point_in_volume(vol, point) {
                return Some(idx);
            }
        }
        None
    }

    /// Sample a 2D plot grid of cell/material IDs
    pub fn sample_grid(&self, params: &PlotParams) -> Result<PlotGrid, String> {
        crate::plot::sample_plot_grid(params, |point| {
            let p = [point.0, point.1, point.2];
            if let Some(idx) = self.find_volume(p) {
                let vol = &self.volumes[idx];
                PlotSample {
                    cell_id: vol.cell_id,
                    material_id: vol.material_id,
                    hover_text: String::new(),
                }
            } else {
                PlotSample {
                    cell_id: -1,
                    material_id: -1,
                    hover_text: String::new(),
                }
            }
        })
    }
}

/// Moller-Trumbore ray-triangle intersection test with arbitrary direction.
/// Returns true if the ray (origin + t * dir) intersects the triangle at t > 0.
fn ray_intersects_triangle_dir(origin: [f64; 3], dir: [f64; 3], tri: &[[f64; 3]; 3]) -> bool {
    let v0 = tri[0];
    let v1 = tri[1];
    let v2 = tri[2];

    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    // h = cross(dir, edge2)
    let h = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];
    let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    if a.abs() < 1e-14 {
        return false; // Ray is parallel to triangle
    }

    let f = 1.0 / a;
    let s = [origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    // q = cross(s, edge1)
    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    // t = f * dot(edge2, q)
    let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    t > 1e-14
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_intersects_triangle_hit() {
        let tri = [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]];
        // Ray from (-1, 0, 0) in +X direction should hit the triangle at x=0
        assert!(ray_intersects_triangle_dir(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            &tri
        ));
    }

    #[test]
    fn test_ray_intersects_triangle_miss() {
        let tri = [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]];
        // Ray from (-1, 5, 0) in +X direction should miss
        assert!(!ray_intersects_triangle_dir(
            [-1.0, 5.0, 0.0],
            [1.0, 0.0, 0.0],
            &tri
        ));
    }

    #[test]
    fn test_point_in_unit_cube_volume() {
        // Build a cube [0,1]^3 from 12 triangles (2 per face)
        let triangles = cube_triangles(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let vol = GeoMeshVolume {
            cell_id: 1,
            material_id: 1,
            material_name: Some("test".to_string()),
            triangles,
        };
        assert!(GeoMesh::point_in_volume(&vol, [0.5, 0.5, 0.5]));
        assert!(!GeoMesh::point_in_volume(&vol, [2.0, 0.5, 0.5]));
    }

    /// Helper: build 12 triangles for an axis-aligned box.
    fn cube_triangles(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> Vec<[[f64; 3]; 3]> {
        let v = [
            [x0, y0, z0], // 0
            [x1, y0, z0], // 1
            [x1, y1, z0], // 2
            [x0, y1, z0], // 3
            [x0, y0, z1], // 4
            [x1, y0, z1], // 5
            [x1, y1, z1], // 6
            [x0, y1, z1], // 7
        ];
        vec![
            // -Z face
            [v[0], v[2], v[1]],
            [v[0], v[3], v[2]],
            // +Z face
            [v[4], v[5], v[6]],
            [v[4], v[6], v[7]],
            // -Y face
            [v[0], v[1], v[5]],
            [v[0], v[5], v[4]],
            // +Y face
            [v[3], v[6], v[2]],
            [v[3], v[7], v[6]],
            // -X face
            [v[0], v[4], v[7]],
            [v[0], v[7], v[3]],
            // +X face
            [v[1], v[2], v[6]],
            [v[1], v[6], v[5]],
        ]
    }
}
