use std::collections::HashSet;
use std::ops::Range;

use crate::size_field::SizeField;

/// Input for constrained Delaunay triangulation.
pub struct CDTInput {
    /// Boundary vertices in order.
    pub vertices: Vec<[f64; 2]>,
    /// Constraint edges as pairs of vertex indices.
    pub constraints: Vec<[usize; 2]>,
    /// Which constraints form boundary loops (outer boundary + holes).
    /// First loop is outer boundary (CCW), subsequent loops are holes (CW).
    pub boundary_loops: Vec<Range<usize>>,
    /// Target maximum edge length for refinement. None = coarse (no refinement).
    pub max_edge_length: Option<f64>,
    /// Minimum angle threshold in degrees for refinement quality.
    pub min_angle: Option<f64>,
    /// Optional curvature-adaptive size field.  When present, the refinement
    /// measures edges in 3D (via the metric tensor) and uses spatially-varying
    /// target sizes instead of the uniform `max_edge_length`.
    pub size_field: Option<SizeField>,
    /// Whether this face has periodic seams that require Steiner point
    /// exclusion near the UV domain boundary.  When true, Ruppert refinement
    /// avoids inserting circumcenters near u_min/u_max/v_min/v_max so that
    /// seam stitching produces a watertight mesh.
    pub periodic_seams: bool,
}

/// Output of constrained Delaunay triangulation.
pub struct CDTOutput {
    /// All vertices: original input vertices followed by any Steiner points.
    pub vertices: Vec<[f64; 2]>,
    /// Number of original (input) vertices. First `num_original_vertices` entries
    /// in `vertices` correspond to input vertices at their original indices.
    pub num_original_vertices: usize,
    /// Triangle connectivity. Each triple indexes into `vertices`.
    /// All triangles are oriented counter-clockwise (positive area).
    pub triangles: Vec<[usize; 3]>,
}

impl CDTOutput {
    /// Identify original boundary vertices that lie on the UV domain seam.
    ///
    /// A vertex is "on the seam" if it's an original boundary vertex at the
    /// UV domain edge (u ~ u_min/u_max or v ~ v_min/v_max).  Seam edges
    /// (where both endpoints are seam vertices) must not be swapped or split
    /// because that would break periodic surface stitching.
    pub fn seam_vertices(&self, sf: &SizeField) -> HashSet<usize> {
        let eps_u = (sf.u_max - sf.u_min) * 1e-6;
        let eps_v = (sf.v_max - sf.v_min) * 1e-6;
        let mut seam = HashSet::new();
        for i in 0..self.num_original_vertices {
            let u = self.vertices[i][0];
            let v = self.vertices[i][1];
            if (u - sf.u_min).abs() < eps_u
                || (u - sf.u_max).abs() < eps_u
                || (v - sf.v_min).abs() < eps_v
                || (v - sf.v_max).abs() < eps_v
            {
                seam.insert(i);
            }
        }
        seam
    }

    /// Remove duplicate triangles and non-manifold excess triangles.
    ///
    /// After edge collapse, the mesh may contain:
    /// - Exact duplicate triangles (same sorted vertex triple)
    /// - Non-manifold edges (3+ triangles sharing the same edge)
    ///
    /// This method removes duplicates (keeping the first occurrence) and
    /// for each non-manifold edge, keeps only the 2 triangles that form
    /// the best quality pair, removing excess triangles.
    pub fn sanitize(&mut self) {
        use std::collections::HashMap;

        // Pass 1: Remove exact duplicate triangles
        let mut seen: HashSet<[usize; 3]> = HashSet::new();
        self.triangles.retain(|tri| {
            let mut key = *tri;
            key.sort();
            seen.insert(key)
        });

        // Pass 2: Remove non-manifold excess triangles (iterative)
        for _ in 0..5 {
            let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for (ti, tri) in self.triangles.iter().enumerate() {
                for k in 0..3 {
                    let a = tri[k];
                    let b = tri[(k + 1) % 3];
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    edge_tris.entry((lo, hi)).or_default().push(ti);
                }
            }

            let mut remove: HashSet<usize> = HashSet::new();
            for tis in edge_tris.values() {
                if tis.len() <= 2 {
                    continue;
                }
                // Keep the 2 triangles with the largest area, remove the rest
                let mut scored: Vec<(usize, f64)> = tis
                    .iter()
                    .map(|&ti| {
                        let t = &self.triangles[ti];
                        let a = self.vertices[t[0]];
                        let b = self.vertices[t[1]];
                        let c = self.vertices[t[2]];
                        let area =
                            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
                        (ti, area)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(ti, _) in &scored[2..] {
                    remove.insert(ti);
                }
            }

            if remove.is_empty() {
                break;
            }

            let remove_vec: Vec<bool> = (0..self.triangles.len())
                .map(|i| remove.contains(&i))
                .collect();
            let mut idx = 0;
            self.triangles.retain(|_| {
                let keep = !remove_vec[idx];
                idx += 1;
                keep
            });
        }
    }
}
