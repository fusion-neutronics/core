//! Bidirectional Data Structure (BDS) half-edge mesh.
//!
//! A half-edge mesh data structure optimised for the split / collapse / swap /
//! smooth operations used in MeshAdapt-style surface meshing.  Inspired by
//! gmsh's `BDS.h` / `BDS.cpp` (Shephard & Beall) but implemented idiomatically
//! in Rust using arena-style index vectors instead of raw pointers.
//!
//! Every directed edge is represented by a [`BDSHalfEdge`].  Each interior
//! edge has a twin half-edge in the opposite direction; boundary edges have
//! `twin == NONE`.  Half-edges are linked into face loops via `next` / `prev`.

use smallvec::SmallVec;
use std::collections::HashMap;

/// Sentinel value used in place of `Option<usize>` for half-edge links that
/// are absent (e.g. a boundary edge with no twin, or a boundary half-edge
/// with no face).
pub const NONE: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A vertex in the BDS mesh.
#[derive(Debug, Clone)]
pub struct BDSVertex {
    /// Unique vertex identifier (index into `BDSMesh::vertices`).
    pub id: usize,
    /// UV parameter coordinate u.
    pub u: f64,
    /// UV parameter coordinate v.
    pub v: f64,
    /// Cached 3D position x (from UV evaluation on the surface).
    pub x: f64,
    /// Cached 3D position y.
    pub y: f64,
    /// Cached 3D position z.
    pub z: f64,
    /// Local target edge length for size-field adaptation.
    pub target_h: f64,
    /// Whether this vertex lies on a mesh boundary.
    pub on_boundary: bool,
    /// One outgoing half-edge from this vertex (for traversal).
    /// `NONE` if the vertex is isolated.
    pub he: usize,
}

/// A half-edge in the BDS mesh.
#[derive(Debug, Clone)]
pub struct BDSHalfEdge {
    /// Unique half-edge identifier (index into `BDSMesh::half_edges`).
    pub id: usize,
    /// Origin vertex id.
    pub origin: usize,
    /// Face id this half-edge borders (`NONE` if boundary).
    pub face: usize,
    /// Twin (opposite) half-edge id (`NONE` if boundary).
    pub twin: usize,
    /// Next half-edge in the face loop.
    pub next: usize,
    /// Previous half-edge in the face loop.
    pub prev: usize,
}

/// A triangular face in the BDS mesh.
#[derive(Debug, Clone)]
pub struct BDSFace {
    /// Unique face identifier (index into `BDSMesh::faces`).
    pub id: usize,
    /// One half-edge belonging to this face.
    pub he: usize,
    /// Soft-deletion flag (for lazy cleanup during topology edits).
    pub deleted: bool,
}

// ---------------------------------------------------------------------------
// The mesh
// ---------------------------------------------------------------------------

/// The BDS half-edge mesh.
#[derive(Debug, Clone)]
pub struct BDSMesh {
    pub vertices: Vec<BDSVertex>,
    pub half_edges: Vec<BDSHalfEdge>,
    pub faces: Vec<BDSFace>,
}

impl BDSMesh {
    /// Build a BDS mesh from flat arrays of UV vertices and triangles.
    ///
    /// `vertices_uv` contains `[u, v]` pairs.  `triangles` contains triples
    /// of vertex indices (into `vertices_uv`) defining counter-clockwise
    /// oriented triangles.
    ///
    /// # Panics
    ///
    /// Panics if any triangle index is out of bounds.
    pub fn from_triangles(vertices_uv: &[[f64; 2]], triangles: &[[usize; 3]]) -> Self {
        let nv = vertices_uv.len();
        let nf = triangles.len();

        // --- Vertices ---
        let mut vertices: Vec<BDSVertex> = vertices_uv
            .iter()
            .enumerate()
            .map(|(i, uv)| BDSVertex {
                id: i,
                u: uv[0],
                v: uv[1],
                x: uv[0],
                y: uv[1],
                z: 0.0,
                target_h: 1.0,
                on_boundary: false,
                he: NONE,
            })
            .collect();

        // We build half-edges in two passes:
        //   Pass 1: create 3 half-edges per triangle, link next/prev within
        //           each face, and record them in an edge map keyed by
        //           (origin, destination).
        //   Pass 2: pair up twins using the edge map and detect boundary
        //           vertices.

        // Map (origin, dest) -> half-edge id.  Used to find twins.
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::with_capacity(nf * 3);

        let mut half_edges: Vec<BDSHalfEdge> = Vec::with_capacity(nf * 3);
        let mut faces: Vec<BDSFace> = Vec::with_capacity(nf);

        for (fi, tri) in triangles.iter().enumerate() {
            assert!(
                tri[0] < nv && tri[1] < nv && tri[2] < nv,
                "triangle vertex index out of bounds"
            );

            let base = half_edges.len(); // index of first half-edge for this face

            // Create 3 half-edges for this triangle.
            for (k, &origin) in tri.iter().enumerate().take(3) {
                let he_id = base + k;
                half_edges.push(BDSHalfEdge {
                    id: he_id,
                    origin,
                    face: fi,
                    twin: NONE,
                    next: base + (k + 1) % 3,
                    prev: base + (k + 2) % 3,
                });
            }

            // Record in edge map and set vertex outgoing half-edge.
            for k in 0..3 {
                let origin = tri[k];
                let dest = tri[(k + 1) % 3];
                let he_id = base + k;
                edge_map.insert((origin, dest), he_id);

                // Give each vertex an outgoing half-edge (overwrite is fine;
                // we just need *some* valid outgoing half-edge).
                if vertices[origin].he == NONE {
                    vertices[origin].he = he_id;
                }
            }

            faces.push(BDSFace {
                id: fi,
                he: base,
                deleted: false,
            });
        }

        // --- Twin pairing ---
        for he_id in 0..half_edges.len() {
            if half_edges[he_id].twin != NONE {
                continue; // already paired
            }
            let origin = half_edges[he_id].origin;
            let next_he = half_edges[he_id].next;
            let dest = half_edges[next_he].origin;

            if let Some(&twin_id) = edge_map.get(&(dest, origin)) {
                half_edges[he_id].twin = twin_id;
                half_edges[twin_id].twin = he_id;
            }
        }

        // --- Boundary detection ---
        // A vertex is on the boundary if any of its outgoing half-edges has
        // no twin.  We also prefer storing a boundary outgoing half-edge so
        // that traversal from a boundary vertex can discover the boundary.
        for he in &half_edges {
            if he.twin == NONE {
                vertices[he.origin].on_boundary = true;
                // Prefer a boundary half-edge as the vertex's outgoing he.
                vertices[he.origin].he = he.id;
                // The destination vertex is also on the boundary.
                let dest = half_edges[he.next].origin;
                vertices[dest].on_boundary = true;
            }
        }

        BDSMesh {
            vertices,
            half_edges,
            faces,
        }
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Destination vertex of a half-edge.
    #[inline]
    pub fn he_dest(&self, he: usize) -> usize {
        self.half_edges[self.half_edges[he].next].origin
    }

    /// Iterate over all outgoing half-edges from vertex `v`, returning their
    /// indices.  Works for both interior and boundary vertices.
    pub fn vertex_half_edges(&self, v: usize) -> SmallVec<[usize; 8]> {
        let start = self.vertices[v].he;
        if start == NONE {
            return SmallVec::new();
        }

        let mut result = SmallVec::new();
        let mut current = start;

        // For interior vertices we walk around using prev->twin.
        // For boundary vertices we handle twin == NONE.
        loop {
            result.push(current);

            // Move to the next outgoing half-edge: go to prev of current,
            // then to its twin.
            let prev_he = self.half_edges[current].prev;
            let twin_of_prev = self.half_edges[prev_he].twin;

            if twin_of_prev == NONE {
                // Hit a boundary going clockwise.
                break;
            }

            current = twin_of_prev;
            if current == start {
                return result; // full loop for interior vertex
            }
        }

        // Vertex is on the boundary.  We collected half-edges going
        // "clockwise" from start.  Now walk "counter-clockwise" from start
        // using twin->next to collect the other direction.
        let twin_of_start = self.half_edges[start].twin;
        if twin_of_start != NONE {
            let mut current = self.half_edges[twin_of_start].next;
            loop {
                result.push(current);
                let twin = self.half_edges[current].twin;
                if twin == NONE {
                    break;
                }
                current = self.half_edges[twin].next;
                if current == start {
                    break;
                }
            }
        }

        result
    }

    /// Get all vertices adjacent to vertex `v` (the 1-ring neighbourhood).
    ///
    /// For boundary vertices, this includes both neighbours reachable via
    /// outgoing half-edges and the neighbour reached via the incoming boundary
    /// half-edge (whose reverse doesn't exist).
    pub fn vertex_neighbors(&self, v: usize) -> SmallVec<[usize; 8]> {
        let start = self.vertices[v].he;
        if start == NONE {
            return SmallVec::new();
        }

        let mut result = SmallVec::new();
        let mut current = start;

        // Walk clockwise via prev->twin, collecting dest of each outgoing he.
        loop {
            result.push(self.he_dest(current));

            let prev_he = self.half_edges[current].prev;
            let twin_of_prev = self.half_edges[prev_he].twin;

            if twin_of_prev == NONE {
                // Hit a boundary.  The prev half-edge points INTO v; its
                // origin is a neighbour we haven't recorded yet via an
                // outgoing half-edge.
                let incoming_origin = self.half_edges[prev_he].origin;
                if !result.contains(&incoming_origin) {
                    result.push(incoming_origin);
                }
                break;
            }

            current = twin_of_prev;
            if current == start {
                return result; // full loop for interior vertex
            }
        }

        // Vertex is on the boundary.  Walk counter-clockwise from start
        // using twin->next.
        let twin_of_start = self.half_edges[start].twin;
        if twin_of_start != NONE {
            let mut current = self.half_edges[twin_of_start].next;
            loop {
                result.push(self.he_dest(current));

                let twin = self.half_edges[current].twin;
                if twin == NONE {
                    break;
                }
                current = self.half_edges[twin].next;
                if current == start {
                    break;
                }
            }
        }

        result
    }

    /// Get all (non-deleted) faces adjacent to vertex `v`.
    pub fn vertex_faces(&self, v: usize) -> SmallVec<[usize; 8]> {
        let mut result = SmallVec::new();
        for &he_id in &self.vertex_half_edges(v) {
            let face = self.half_edges[he_id].face;
            if face != NONE && !self.faces[face].deleted && !result.contains(&face) {
                result.push(face);
            }
        }
        result
    }

    /// Get the two faces sharing the edge represented by half-edge `he`.
    ///
    /// Returns `(face_of_he, Some(face_of_twin))` for interior edges, or
    /// `(face_of_he, None)` for boundary edges.
    pub fn edge_faces(&self, he: usize) -> (usize, Option<usize>) {
        let f1 = self.half_edges[he].face;
        let twin = self.half_edges[he].twin;
        if twin == NONE {
            (f1, None)
        } else {
            (f1, Some(self.half_edges[twin].face))
        }
    }

    /// Validate the entire half-edge mesh for internal consistency.
    ///
    /// Checks:
    /// - next/prev form closed loops of length 3 per face
    /// - twin symmetry: `twin(twin(he)) == he`
    /// - face references are consistent
    /// - every vertex's `he` points to a valid half-edge originating from it
    /// - every non-deleted face's half-edges reference it
    ///
    /// Returns `true` if valid.
    pub fn validate(&self) -> bool {
        for he in &self.half_edges {
            // next/prev consistency
            if self.half_edges[he.next].prev != he.id {
                return false;
            }
            if self.half_edges[he.prev].next != he.id {
                return false;
            }

            // Face loop must be a triangle (length 3).
            {
                let mut cur = he.next;
                let mut count = 1;
                while cur != he.id {
                    cur = self.half_edges[cur].next;
                    count += 1;
                    if count > 3 {
                        return false;
                    }
                }
                if count != 3 {
                    return false;
                }
            }

            // Twin symmetry.
            if he.twin != NONE {
                let twin = &self.half_edges[he.twin];
                if twin.twin != he.id {
                    return false;
                }
                // Twin must go in the opposite direction.
                if twin.origin != self.he_dest(he.id) {
                    return false;
                }
                if self.he_dest(he.twin) != he.origin {
                    return false;
                }
            }

            // Half-edge origin must be a valid vertex.
            if he.origin >= self.vertices.len() {
                return false;
            }

            // Face reference consistency.
            if he.face != NONE && he.face >= self.faces.len() {
                return false;
            }
        }

        // Check each vertex.
        for v in &self.vertices {
            if v.he != NONE {
                if v.he >= self.half_edges.len() {
                    return false;
                }
                if self.half_edges[v.he].origin != v.id {
                    return false;
                }
            }
        }

        // Check each non-deleted face.
        for f in &self.faces {
            if f.deleted {
                continue;
            }
            if f.he >= self.half_edges.len() {
                return false;
            }
            // All three half-edges of the face must reference this face.
            let mut cur = f.he;
            for _ in 0..3 {
                if self.half_edges[cur].face != f.id {
                    return false;
                }
                cur = self.half_edges[cur].next;
            }
            if cur != f.he {
                return false;
            }
        }

        true
    }

    /// Count the number of live (non-deleted) faces.
    pub fn num_live_faces(&self) -> usize {
        self.faces.iter().filter(|f| !f.deleted).count()
    }

    /// Export the mesh back to flat UV-vertex and triangle-index arrays.
    ///
    /// Deleted faces are skipped.  Vertex indices are compacted so the
    /// returned vertex array contains only vertices referenced by at least one
    /// live face.
    pub fn to_triangles(&self) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        // Collect all vertex ids used by live faces.
        let mut used = vec![false; self.vertices.len()];
        let mut tri_raw: Vec<[usize; 3]> = Vec::with_capacity(self.num_live_faces());

        for f in &self.faces {
            if f.deleted {
                continue;
            }
            let he0 = f.he;
            let he1 = self.half_edges[he0].next;
            let he2 = self.half_edges[he1].next;
            let v0 = self.half_edges[he0].origin;
            let v1 = self.half_edges[he1].origin;
            let v2 = self.half_edges[he2].origin;
            used[v0] = true;
            used[v1] = true;
            used[v2] = true;
            tri_raw.push([v0, v1, v2]);
        }

        // Build compacted vertex array and old-to-new index map.
        let mut old_to_new = vec![0usize; self.vertices.len()];
        let mut verts: Vec<[f64; 2]> = Vec::new();
        for (i, v) in self.vertices.iter().enumerate() {
            if used[i] {
                old_to_new[i] = verts.len();
                verts.push([v.u, v.v]);
            }
        }

        let tris: Vec<[usize; 3]> = tri_raw
            .iter()
            .map(|t| [old_to_new[t[0]], old_to_new[t[1]], old_to_new[t[2]]])
            .collect();

        (verts, tris)
    }

    /// Get the three vertex ids of a face.
    pub fn face_vertices(&self, f: usize) -> [usize; 3] {
        let he0 = self.faces[f].he;
        let he1 = self.half_edges[he0].next;
        let he2 = self.half_edges[he1].next;
        [
            self.half_edges[he0].origin,
            self.half_edges[he1].origin,
            self.half_edges[he2].origin,
        ]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles sharing an edge:
    ///
    /// ```text
    ///   2---3
    ///   |\ |
    ///   | \|
    ///   0---1
    /// ```
    ///
    /// Triangles: (0,1,2), (1,3,2)
    fn two_triangle_mesh() -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let verts = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tris = vec![[0, 1, 2], [1, 3, 2]];
        (verts, tris)
    }

    /// Diamond (4 triangles around a central vertex):
    ///
    /// ```text
    ///       2
    ///      /|\
    ///     / | \
    ///    3--4--1
    ///     \ | /
    ///      \|/
    ///       0
    /// ```
    ///
    /// Triangles: (4,0,1), (4,1,2), (4,2,3), (4,3,0)
    fn diamond_mesh() -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, -1.0], // 0: bottom
            [1.0, 0.0],  // 1: right
            [0.0, 1.0],  // 2: top
            [-1.0, 0.0], // 3: left
            [0.0, 0.0],  // 4: center
        ];
        let tris = vec![[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]];
        (verts, tris)
    }

    /// Generate an NxN grid of quads, each split into 2 triangles.
    /// Returns (n+1)*(n+1) vertices and 2*n*n triangles.
    fn grid_mesh(n: usize) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let mut verts = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            for i in 0..=n {
                verts.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        let mut tris = Vec::with_capacity(2 * n * n);
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;
                tris.push([v00, v10, v11]);
                tris.push([v00, v11, v01]);
            }
        }
        (verts, tris)
    }

    // ---------------------------------------------------------------
    // Basic construction tests
    // ---------------------------------------------------------------

    #[test]
    fn two_triangles_basic() {
        let (v, t) = two_triangle_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);
        // 2 triangles * 3 half-edges = 6
        assert_eq!(mesh.half_edges.len(), 6);
        assert!(mesh.validate());
    }

    #[test]
    fn two_triangles_half_edge_connectivity() {
        let (v, t) = two_triangle_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        let mut interior_count = 0;
        let mut boundary_count = 0;
        for he in &mesh.half_edges {
            if he.twin == NONE {
                boundary_count += 1;
            } else {
                interior_count += 1;
            }
        }
        // 5 edges total: 1 interior (2 half-edges with twins) + 4 boundary
        assert_eq!(
            interior_count, 2,
            "should have 2 interior half-edges (1 shared edge)"
        );
        assert_eq!(boundary_count, 4, "should have 4 boundary half-edges");
    }

    #[test]
    fn two_triangles_boundary_vertices() {
        let (v, t) = two_triangle_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        // In a 2-triangle quad, all 4 vertices are on the boundary.
        for vi in 0..4 {
            assert!(
                mesh.vertices[vi].on_boundary,
                "vertex {} should be on boundary",
                vi
            );
        }
    }

    // ---------------------------------------------------------------
    // Diamond mesh
    // ---------------------------------------------------------------

    #[test]
    fn diamond_basic() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        assert_eq!(mesh.vertices.len(), 5);
        assert_eq!(mesh.faces.len(), 4);
        assert_eq!(mesh.half_edges.len(), 12);
        assert!(mesh.validate());
    }

    #[test]
    fn diamond_vertex_neighbors() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        // Centre vertex 4 should be adjacent to all 4 outer vertices.
        let mut nbrs = mesh.vertex_neighbors(4);
        nbrs.sort();
        assert_eq!(nbrs.as_slice(), &[0, 1, 2, 3]);

        // Outer vertex 0 should be adjacent to 1, 3, and centre 4.
        let mut nbrs0 = mesh.vertex_neighbors(0);
        nbrs0.sort();
        assert_eq!(nbrs0.as_slice(), &[1, 3, 4]);
    }

    #[test]
    fn diamond_vertex_faces() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        // Centre vertex 4 touches all 4 faces.
        let faces4 = mesh.vertex_faces(4);
        assert_eq!(faces4.len(), 4);

        // Outer vertex 0 touches 2 faces: (4,0,1) and (4,3,0).
        let faces0 = mesh.vertex_faces(0);
        assert_eq!(faces0.len(), 2);
    }

    #[test]
    fn diamond_edge_faces() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        let mut found_interior = false;
        let mut found_boundary = false;
        for he in &mesh.half_edges {
            let (f1, f2) = mesh.edge_faces(he.id);
            if let Some(f2_val) = f2 {
                found_interior = true;
                assert!(f1 != NONE);
                assert!(f2_val != f1, "faces should differ");
            } else {
                found_boundary = true;
                assert!(f1 != NONE);
            }
        }
        assert!(found_interior, "diamond should have interior edges");
        assert!(found_boundary, "diamond should have boundary edges");
    }

    #[test]
    fn diamond_boundary() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        // Centre vertex is interior.
        assert!(
            !mesh.vertices[4].on_boundary,
            "centre vertex should be interior"
        );

        // Outer vertices are on boundary.
        for vi in 0..4 {
            assert!(
                mesh.vertices[vi].on_boundary,
                "outer vertex {} should be on boundary",
                vi
            );
        }
    }

    // ---------------------------------------------------------------
    // Grid mesh
    // ---------------------------------------------------------------

    #[test]
    fn grid_10x10_validate() {
        let (v, t) = grid_mesh(10);
        let mesh = BDSMesh::from_triangles(&v, &t);

        assert_eq!(mesh.vertices.len(), 11 * 11);
        assert_eq!(mesh.faces.len(), 2 * 10 * 10);
        assert!(mesh.validate());
    }

    #[test]
    fn grid_10x10_boundary_count() {
        let (v, t) = grid_mesh(10);
        let mesh = BDSMesh::from_triangles(&v, &t);

        let boundary_count = mesh.vertices.iter().filter(|v| v.on_boundary).count();
        // Boundary of an 11x11 grid: 4*10 = 40 vertices on the perimeter.
        assert_eq!(boundary_count, 40);
    }

    #[test]
    fn grid_10x10_interior_vertex_neighbors() {
        let (v, t) = grid_mesh(10);
        let mesh = BDSMesh::from_triangles(&v, &t);

        // An interior vertex should have 6 neighbors in this triangulation
        // pattern (each quad splits into 2 triangles along the diagonal).
        // Pick vertex at grid position (5, 5) = index 5*11 + 5 = 60.
        let vi = 5 * 11 + 5;
        assert!(!mesh.vertices[vi].on_boundary);
        let nbrs = mesh.vertex_neighbors(vi);
        assert_eq!(
            nbrs.len(),
            6,
            "interior grid vertex should have 6 neighbors"
        );
    }

    #[test]
    fn grid_5x5_all_halfedges_valid() {
        let (v, t) = grid_mesh(5);
        let mesh = BDSMesh::from_triangles(&v, &t);

        assert!(mesh.validate());

        // Check twin symmetry exhaustively.
        for he in &mesh.half_edges {
            if he.twin != NONE {
                let twin = &mesh.half_edges[he.twin];
                assert_eq!(twin.twin, he.id, "twin symmetry broken");
                assert_eq!(twin.origin, mesh.he_dest(he.id));
                assert_eq!(mesh.he_dest(he.twin), he.origin);
            }
        }
    }

    #[test]
    fn grid_20x20_validate() {
        let (v, t) = grid_mesh(20);
        let mesh = BDSMesh::from_triangles(&v, &t);
        assert_eq!(mesh.vertices.len(), 21 * 21);
        assert_eq!(mesh.faces.len(), 2 * 20 * 20);
        assert!(mesh.validate());
    }

    // ---------------------------------------------------------------
    // Boundary detection
    // ---------------------------------------------------------------

    #[test]
    fn single_triangle_all_boundary() {
        let verts = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let tris = vec![[0, 1, 2]];
        let mesh = BDSMesh::from_triangles(&verts, &tris);

        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.half_edges.len(), 3);
        assert!(mesh.validate());

        // All edges are boundary (no twins).
        for he in &mesh.half_edges {
            assert_eq!(
                he.twin, NONE,
                "single triangle should have all boundary edges"
            );
        }
        // All vertices are boundary.
        for v in &mesh.vertices {
            assert!(v.on_boundary);
        }
    }

    #[test]
    fn boundary_edges_have_no_twin() {
        let (v, t) = grid_mesh(3);
        let mesh = BDSMesh::from_triangles(&v, &t);

        let boundary_he_count = mesh.half_edges.iter().filter(|he| he.twin == NONE).count();
        // A 3x3 grid has 4*3 = 12 boundary edges.
        assert_eq!(boundary_he_count, 12);
    }

    // ---------------------------------------------------------------
    // vertex_neighbors correctness
    // ---------------------------------------------------------------

    #[test]
    fn two_triangles_vertex_neighbors() {
        let (v, t) = two_triangle_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        // vertex 0 is connected to 1 and 2
        let mut n0 = mesh.vertex_neighbors(0);
        n0.sort();
        assert_eq!(n0.as_slice(), &[1, 2]);

        // vertex 1 is connected to 0, 2, 3
        let mut n1 = mesh.vertex_neighbors(1);
        n1.sort();
        assert_eq!(n1.as_slice(), &[0, 2, 3]);

        // vertex 2 is connected to 0, 1, 3
        let mut n2 = mesh.vertex_neighbors(2);
        n2.sort();
        assert_eq!(n2.as_slice(), &[0, 1, 3]);

        // vertex 3 is connected to 1, 2
        let mut n3 = mesh.vertex_neighbors(3);
        n3.sort();
        assert_eq!(n3.as_slice(), &[1, 2]);
    }

    // ---------------------------------------------------------------
    // Roundtrip: from_triangles -> to_triangles -> from_triangles
    // ---------------------------------------------------------------

    #[test]
    fn roundtrip_two_triangles() {
        let (v, t) = two_triangle_mesh();
        let mesh1 = BDSMesh::from_triangles(&v, &t);
        let (v2, t2) = mesh1.to_triangles();
        let mesh2 = BDSMesh::from_triangles(&v2, &t2);

        assert!(mesh2.validate());
        assert_eq!(mesh2.num_live_faces(), mesh1.num_live_faces());
        assert_eq!(mesh2.vertices.len(), mesh1.vertices.len());
    }

    #[test]
    fn roundtrip_diamond() {
        let (v, t) = diamond_mesh();
        let mesh1 = BDSMesh::from_triangles(&v, &t);
        let (v2, t2) = mesh1.to_triangles();
        let mesh2 = BDSMesh::from_triangles(&v2, &t2);

        assert!(mesh2.validate());
        assert_eq!(mesh2.num_live_faces(), mesh1.num_live_faces());
        assert_eq!(mesh2.vertices.len(), mesh1.vertices.len());
    }

    #[test]
    fn roundtrip_grid() {
        let (v, t) = grid_mesh(8);
        let mesh1 = BDSMesh::from_triangles(&v, &t);
        let (v2, t2) = mesh1.to_triangles();
        let mesh2 = BDSMesh::from_triangles(&v2, &t2);

        assert!(mesh2.validate());
        assert_eq!(mesh2.num_live_faces(), mesh1.num_live_faces());
        assert_eq!(mesh2.vertices.len(), mesh1.vertices.len());

        // Check that UV coordinates survived the roundtrip.
        for i in 0..v2.len() {
            let orig = &mesh1.vertices[i];
            let rt = &mesh2.vertices[i];
            assert!(
                (orig.u - rt.u).abs() < 1e-15 && (orig.v - rt.v).abs() < 1e-15,
                "UV mismatch at vertex {}",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // num_live_faces
    // ---------------------------------------------------------------

    #[test]
    fn num_live_faces_with_deletion() {
        let (v, t) = diamond_mesh();
        let mut mesh = BDSMesh::from_triangles(&v, &t);
        assert_eq!(mesh.num_live_faces(), 4);

        mesh.faces[1].deleted = true;
        assert_eq!(mesh.num_live_faces(), 3);

        mesh.faces[0].deleted = true;
        mesh.faces[2].deleted = true;
        assert_eq!(mesh.num_live_faces(), 1);
    }

    // ---------------------------------------------------------------
    // face_vertices
    // ---------------------------------------------------------------

    #[test]
    fn face_vertices_match_input() {
        let (v, t) = diamond_mesh();
        let mesh = BDSMesh::from_triangles(&v, &t);

        for (fi, tri) in t.iter().enumerate() {
            let fv = mesh.face_vertices(fi);
            let mut fv_sorted = fv.to_vec();
            fv_sorted.sort();
            let mut tri_sorted = tri.to_vec();
            tri_sorted.sort();
            assert_eq!(fv_sorted, tri_sorted, "face {} vertex set mismatch", fi);
        }
    }

    // ---------------------------------------------------------------
    // Edge consistency on larger mesh
    // ---------------------------------------------------------------

    #[test]
    fn grid_interior_edge_has_two_faces() {
        let (v, t) = grid_mesh(4);
        let mesh = BDSMesh::from_triangles(&v, &t);

        for he in &mesh.half_edges {
            if he.twin != NONE {
                let (f1, f2) = mesh.edge_faces(he.id);
                assert!(f1 != NONE);
                assert!(f2.is_some());
                assert_ne!(f1, f2.unwrap());
            }
        }
    }

    // ---------------------------------------------------------------
    // Edge counting / Euler formula
    // ---------------------------------------------------------------

    #[test]
    fn edge_count_euler() {
        // For a disk triangulation: V - E + F = 1.
        let (v, t) = grid_mesh(6);
        let mesh = BDSMesh::from_triangles(&v, &t);

        let nv = mesh.vertices.len();
        let nf = mesh.num_live_faces();

        let total_he = mesh.half_edges.len();
        let boundary_he = mesh.half_edges.iter().filter(|he| he.twin == NONE).count();
        let interior_he = total_he - boundary_he;
        let ne = interior_he / 2 + boundary_he;

        let euler = nv as isize - ne as isize + nf as isize;
        assert_eq!(euler, 1, "Euler formula should hold for a disk mesh");
    }

    // ---------------------------------------------------------------
    // Stress test: large grid
    // ---------------------------------------------------------------

    #[test]
    fn grid_50x50_validate() {
        let (v, t) = grid_mesh(50);
        let mesh = BDSMesh::from_triangles(&v, &t);
        assert_eq!(mesh.vertices.len(), 51 * 51);
        assert_eq!(mesh.faces.len(), 2 * 50 * 50);
        assert!(mesh.validate());
    }
}
