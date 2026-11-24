use std::collections::{HashMap, HashSet};

use spade::{handles::FixedVertexHandle, ConstrainedDelaunayTriangulation, Point2, Triangulation};

use super::types::{CDTInput, CDTOutput};

/// Constrained Delaunay triangulation using spade.
#[allow(clippy::upper_case_acronyms)]
pub struct CDT {
    tri: ConstrainedDelaunayTriangulation<Point2<f64>>,
    /// Mapping from our vertex indices to spade vertex handles.
    handles: Vec<FixedVertexHandle>,
    /// The vertices refinement added by splitting boundary segments, in the
    /// order it added them (see [`CDT::split_constraint`]).
    split_handles: Vec<FixedVertexHandle>,
    /// How many vertices the triangulation held once built, before any
    /// refinement: they come first in the output.
    num_built: usize,
}

impl CDT {
    /// Create a new CDT from input.
    pub fn new(input: &CDTInput) -> Self {
        let tri = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

        CDT {
            tri,
            handles: Vec::with_capacity(input.vertices.len()),
            split_handles: Vec::new(),
            num_built: 0,
        }
    }

    /// Build the constrained Delaunay triangulation.
    pub fn build(&mut self, input: &CDTInput) {
        // spade rejects subnormal coordinates (0 < |x| < MIN_ALLOWED_VALUE).
        // Dirty STEP pcurves produce them (GEOUNED's FWTBM1.step: a plane
        // face with v = 3.2e-45) - such values are physically zero, so clamp
        // them instead of letting insert() fail. The Err fallback below
        // reuses the NEAREST existing vertex (or the origin when the
        // triangulation is empty), which collapses the boundary loop and
        // silently emits a zero-triangle face.
        let sane = |x: f64| {
            if x != 0.0 && x.abs() < spade::MIN_ALLOWED_VALUE {
                0.0
            } else {
                x
            }
        };
        // Insert all vertices
        for v in &input.vertices {
            let point = Point2::new(sane(v[0]), sane(v[1]));
            let handle = match self.tri.insert(point) {
                Ok(h) => h,
                Err(_) => {
                    // Invalid coordinate (NaN, subnormal, etc.) - find the
                    // nearest already-inserted vertex and reuse its handle.
                    self.nearest_handle(v).unwrap_or_else(|| {
                        // Triangulation is empty; clamp to zero as last resort.
                        self.tri
                            .insert(Point2::new(0.0, 0.0))
                            .expect("inserting origin should not fail")
                    })
                }
            };
            self.handles.push(handle);
        }

        // Add constraint edges. `add_constraint` PANICS when the new
        // constraint intersects an existing one - which real (dirty) CAD
        // produces: self-touching UV boundaries on degenerate faces (observed
        // on GEOUNED's FWTBM1.step / RJ24.stp), which once crashed the whole
        // meshing run through PyO3. The fast path adds the constraint directly
        // when it doesn't conflict; otherwise (see the `else` below) it is
        // SPLIT at the crossing rather than skipped, so the boundary loop stays
        // enforced and the face stays watertight.
        let mut split = 0usize;
        for c in &input.constraints {
            if self
                .tri
                .can_add_constraint(self.handles[c[0]], self.handles[c[1]])
            {
                self.tri
                    .add_constraint(self.handles[c[0]], self.handles[c[1]]);
            } else {
                // Self-touching / intersecting boundary (dirty CAD, e.g.
                // GEOUNED RJ24.stp / Triangle.stp): rather than SKIP the
                // constraint - which leaves the boundary unenforced and opens
                // edges - split it at its intersection(s) with the existing
                // constraints, inserting a vertex at each crossing so the full
                // boundary loop is still enforced and the face stays
                // watertight. The split vertices are appended after the input
                // vertices, so `num_original_vertices` (= input count) still
                // correctly marks the boundary verts and they evaluate onto
                // the surface like Steiner points.
                self.tri
                    .add_constraint_and_split(self.handles[c[0]], self.handles[c[1]], |p| p);
                split += 1;
            }
        }
        self.num_built = self.tri.num_vertices();
        if split > 0 {
            eprintln!(
                "    [cdt] split {split}/{} self-intersecting constraint edges (self-touching face boundary)",
                input.constraints.len()
            );
        }
    }

    /// Convert to output, removing exterior triangles and holes.
    ///
    /// The vertices come in the triangulation's own order, except that the
    /// points refinement added on the boundary ([`CDT::split_constraint`])
    /// are moved up to follow the vertices it was built from, ahead of the
    /// Steiner points, so that no later pass mistakes them for interior
    /// points to move or collapse. With no such points the order is the
    /// triangulation's.
    pub fn into_output(self, input: &CDTInput) -> CDTOutput {
        let split: HashSet<FixedVertexHandle> = self.split_handles.iter().copied().collect();
        let built = self.tri.vertices().take(self.num_built).map(|v| v.fix());
        let rest = self
            .tri
            .vertices()
            .skip(self.num_built)
            .map(|v| v.fix())
            .filter(|h| !split.contains(h));
        let order: Vec<FixedVertexHandle> = built
            .chain(self.split_handles.iter().copied())
            .chain(rest)
            .collect();
        let vertices: Vec<[f64; 2]> = order
            .iter()
            .map(|&h| {
                let p = self.tri.vertex(h).position();
                [p.x, p.y]
            })
            .collect();

        // Build a mapping from FixedVertexHandle index to our vertex index
        let handle_to_idx: std::collections::HashMap<FixedVertexHandle, usize> =
            order.iter().enumerate().map(|(i, &h)| (h, i)).collect();

        // Collect all inner triangles
        let all_triangles: Vec<[usize; 3]> = self
            .tri
            .inner_faces()
            .map(|f| {
                let vs = f.vertices();
                [
                    handle_to_idx[&vs[0].fix()],
                    handle_to_idx[&vs[1].fix()],
                    handle_to_idx[&vs[2].fix()],
                ]
            })
            .collect();

        // Filter out triangles outside boundaries and inside holes
        let kept = classify_triangles(&vertices, &all_triangles, input);

        // Ensure CCW orientation
        let mut triangles = Vec::with_capacity(kept.len());
        for tri in kept {
            let a = vertices[tri[0]];
            let b = vertices[tri[1]];
            let c = vertices[tri[2]];
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cross > 0.0 {
                triangles.push(tri);
            } else if cross < 0.0 {
                triangles.push([tri[0], tri[2], tri[1]]);
            }
            // Skip degenerate (zero-area) triangles
        }

        CDTOutput {
            vertices,
            num_original_vertices: input.vertices.len(),
            triangles,
            num_boundary_splits: self.split_handles.len(),
            split_requests: Vec::new(),
        }
    }

    /// Find the nearest already-inserted vertex handle to the given point.
    fn nearest_handle(&self, v: &[f64; 2]) -> Option<FixedVertexHandle> {
        if self.handles.is_empty() {
            return None;
        }
        let mut best = self.handles[0];
        let mut best_dist = f64::MAX;
        for &h in &self.handles {
            let p = self.tri.vertex(h).position();
            let dx = p.x - v[0];
            let dy = p.y - v[1];
            let d = dx * dx + dy * dy;
            if d < best_dist {
                best_dist = d;
                best = h;
            }
        }
        Some(best)
    }

    /// The handle of input vertex `i`.
    pub fn handle(&self, i: usize) -> Option<FixedVertexHandle> {
        self.handles.get(i).copied()
    }

    /// The UV position of vertex `h`.
    pub fn position(&self, h: FixedVertexHandle) -> [f64; 2] {
        let p = self.tri.vertex(h).position();
        [p.x, p.y]
    }

    /// The number of vertices in the triangulation.
    pub fn num_vertices(&self) -> usize {
        self.tri.num_vertices()
    }

    /// Split the boundary segment from `a` to `b` at its midpoint: the
    /// constraint is replaced by the two halves through a new vertex there,
    /// which is kept on the boundary from then on. Returns the new vertex,
    /// or `None`, changing nothing, when `a` and `b` are no longer joined by
    /// a constraint edge or the halves cannot be added.
    pub fn split_constraint(
        &mut self,
        a: FixedVertexHandle,
        b: FixedVertexHandle,
    ) -> Option<FixedVertexHandle> {
        let edge = self
            .tri
            .get_edge_from_neighbors(a, b)?
            .fix()
            .as_undirected();
        if !self.tri.is_constraint_edge(edge) {
            return None;
        }
        let (pa, pb) = (self.position(a), self.position(b));
        let mid = Point2::new((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5);
        self.tri.remove_constraint_edge(edge);
        let m = match self.tri.insert(mid) {
            Ok(m) if m != a && m != b => m,
            _ => {
                self.tri.add_constraint(a, b);
                return None;
            }
        };
        for (p, q) in [(a, m), (m, b)] {
            if self.tri.can_add_constraint(p, q) {
                self.tri.add_constraint(p, q);
            } else {
                self.tri.add_constraint_and_split(p, q, |x| x);
            }
        }
        self.split_handles.push(m);
        Some(m)
    }

    /// Insert a Steiner point and return its index in the output vertex list.
    pub fn insert_steiner(&mut self, point: [f64; 2]) -> Option<FixedVertexHandle> {
        self.tri.insert(Point2::new(point[0], point[1])).ok()
    }

    /// Get current triangles as vertex index triples (for refinement checking).
    pub fn get_triangle_positions(&self) -> Vec<([f64; 2], [f64; 2], [f64; 2])> {
        self.tri
            .inner_faces()
            .map(|f| {
                let vs = f.vertices();
                let a = vs[0].position();
                let b = vs[1].position();
                let c = vs[2].position();
                ([a.x, a.y], [b.x, b.y], [c.x, c.y])
            })
            .collect()
    }

    /// Return the set of `FixedVertexHandle`s that correspond to the original
    /// input (boundary / constraint) vertices.  These must not be moved during
    /// smoothing.
    pub fn boundary_handles(&self) -> HashSet<FixedVertexHandle> {
        self.handles
            .iter()
            .chain(&self.split_handles)
            .copied()
            .collect()
    }

    /// Build an adjacency map: for every vertex, the set of neighbouring
    /// vertex handles (connected by a triangle edge).
    pub fn vertex_adjacency(&self) -> HashMap<FixedVertexHandle, Vec<FixedVertexHandle>> {
        let mut adj: HashMap<FixedVertexHandle, Vec<FixedVertexHandle>> = HashMap::new();
        for face in self.tri.inner_faces() {
            let vs = face.vertices();
            let h = [vs[0].fix(), vs[1].fix(), vs[2].fix()];
            for i in 0..3 {
                let j = (i + 1) % 3;
                adj.entry(h[i]).or_default().push(h[j]);
                adj.entry(h[j]).or_default().push(h[i]);
            }
        }
        // Deduplicate neighbours (an edge shared by two faces adds duplicates).
        for neighbours in adj.values_mut() {
            neighbours.sort_by_key(|h| h.index());
            neighbours.dedup();
        }
        adj
    }

    /// Get the position of a vertex by its handle.
    pub fn vertex_position(&self, handle: FixedVertexHandle) -> [f64; 2] {
        let p = self.tri.vertex(handle).position();
        [p.x, p.y]
    }

    /// Move a vertex to a new position.
    ///
    /// This directly mutates the stored `Point2` via spade's
    /// `vertex_data_mut`.  The Delaunay property is invalidated but the
    /// triangle connectivity is preserved, which is exactly what Laplacian
    /// smoothing needs.
    pub fn set_vertex_position(&mut self, handle: FixedVertexHandle, pos: [f64; 2]) {
        let pt = self.tri.vertex_data_mut(handle);
        *pt = Point2::new(pos[0], pos[1]);
    }

    /// Return all fixed vertex handles in the triangulation.
    #[allow(dead_code)]
    pub fn all_vertex_handles(&self) -> Vec<FixedVertexHandle> {
        self.tri.fixed_vertices().collect()
    }
}

/// Classify which triangles are inside the domain (outside holes).
/// Uses point-in-polygon test on triangle centroids against boundary loops.
fn classify_triangles(
    vertices: &[[f64; 2]],
    triangles: &[[usize; 3]],
    input: &CDTInput,
) -> Vec<[usize; 3]> {
    if input.boundary_loops.is_empty() {
        return triangles.to_vec();
    }

    // Build boundary polygons from the constraint edges in each loop
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    for loop_range in &input.boundary_loops {
        let mut polygon = Vec::new();
        for ci in loop_range.clone() {
            if ci < input.constraints.len() {
                polygon.push(input.vertices[input.constraints[ci][0]]);
            }
        }
        if !polygon.is_empty() {
            loops.push(polygon);
        }
    }

    triangles
        .iter()
        .filter(|tri| {
            let a = vertices[tri[0]];
            let b = vertices[tri[1]];
            let c = vertices[tri[2]];
            let cx = (a[0] + b[0] + c[0]) / 3.0;
            let cy = (a[1] + b[1] + c[1]) / 3.0;
            let centroid = [cx, cy];

            // Must be inside the outer boundary (first loop)
            if !loops.is_empty() && !point_in_polygon(centroid, &loops[0]) {
                return false;
            }

            // Must not be inside any hole (subsequent loops)
            for hole in loops.iter().skip(1) {
                if point_in_polygon(centroid, hole) {
                    return false;
                }
            }

            true
        })
        .copied()
        .collect()
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let yi = polygon[i][1];
        let yj = polygon[j][1];

        if (yi > point[1]) != (yj > point[1]) {
            let xi = polygon[i][0];
            let xj = polygon[j][0];
            let x_intersect = xi + (point[1] - yi) / (yj - yi) * (xj - xi);
            if point[0] < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }

    inside
}
