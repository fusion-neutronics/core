//! Local mesh operations for tet mesh quality improvement.
//!
//! Iteratively applies four local operations to bring all tet edges
//! close to `target_edge_length`:
//!
//! 1. **Edge split**: split edges > split_threshold (default 1.4h)
//! 2. **Edge swap**: 2-to-3 flips to improve quality
//!
//! Each operation is LOCAL and preserves mesh validity (no overlaps, no gaps).
use super::dethash::{HashMap, HashSet};
use super::optimize;
use super::predicates3d::{self, dist_sq_3d, orient_3d};

// ── Adjacency data structure ─────────────────────────────────────────────

/// Tet mesh with adjacency for local operations.
pub struct TetMesh {
    pub vertices: Vec<[f64; 3]>,
    pub tets: Vec<[usize; 4]>,
    pub n_boundary: usize,
    /// Vertices that must be treated as ON THE SURFACE, beyond the leading
    /// `n_boundary` block. The cut-cell clip (#136) introduces vertices that lie
    /// exactly on boundary triangles but land in the interior index range, and
    /// without this a swap would happily dismantle a clipped boundary face and
    /// undo the conformity the clip just established - measured: the clip leaves
    /// the fill exact to 6.7e-15 before this pass and 9.8e-3 after it.
    pinned: Vec<bool>,
    /// vertex → list of tet indices containing that vertex
    vert_tets: Vec<Vec<usize>>,
    /// sorted face → tet indices sharing that face (max 2)
    face_tets: HashMap<[usize; 3], Vec<usize>>,
    /// Bitmap of dead (tombstone) tets - avoids retain+reindex.
    is_dead: Vec<bool>,
}

fn sorted_face(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut f = [a, b, c];
    f.sort();
    f
}

fn tet_faces(t: &[usize; 4]) -> [[usize; 3]; 4] {
    [
        sorted_face(t[1], t[2], t[3]),
        sorted_face(t[0], t[2], t[3]),
        sorted_face(t[0], t[1], t[3]),
        sorted_face(t[0], t[1], t[2]),
    ]
}

fn sorted_edge(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

impl TetMesh {
    /// Build from raw vertices and tets, with an explicit on-surface set.
    /// `None` keeps the historical rule (the leading `n_boundary` vertices).
    pub fn new_pinned(
        vertices: Vec<[f64; 3]>,
        tets: Vec<[usize; 4]>,
        n_boundary: usize,
        pinned: Option<&[bool]>,
    ) -> Self {
        let n = tets.len();
        let mut pin = vec![false; vertices.len()];
        for (i, p) in pin.iter_mut().enumerate() {
            *p = i < n_boundary || pinned.map(|s| s.get(i) == Some(&true)).unwrap_or(false);
        }
        let mut mesh = TetMesh {
            vertices,
            tets,
            n_boundary,
            pinned: pin,
            vert_tets: Vec::new(),
            face_tets: HashMap::default(),
            is_dead: vec![false; n],
        };
        mesh.rebuild_adjacency();
        mesh
    }

    fn rebuild_adjacency(&mut self) {
        self.vert_tets = vec![Vec::new(); self.vertices.len()];
        self.face_tets.clear();
        self.face_tets.reserve(self.tets.len() * 2);

        for (ti, t) in self.tets.iter().enumerate() {
            if self.is_dead.get(ti).copied().unwrap_or(false) {
                continue;
            }
            for &v in t {
                if v < self.vert_tets.len() {
                    self.vert_tets[v].push(ti);
                }
            }
            for f in &tet_faces(t) {
                self.face_tets.entry(*f).or_default().push(ti);
            }
        }
    }

    /// Extend vert_tets if new vertices were added.
    fn ensure_vert_tets(&mut self) {
        while self.vert_tets.len() < self.vertices.len() {
            self.vert_tets.push(Vec::new());
        }
    }

    /// Check if a vertex is on the boundary (first n_boundary vertices).
    fn is_boundary_vertex(&self, v: usize) -> bool {
        match self.pinned.get(v) {
            Some(&p) => p,
            None => v < self.n_boundary,
        }
    }

    /// Get all tets sharing an edge.
    fn tets_around_edge(&self, a: usize, b: usize) -> Vec<usize> {
        let mut result = Vec::new();
        if a < self.vert_tets.len() {
            for &ti in &self.vert_tets[a] {
                if !self.is_dead.get(ti).copied().unwrap_or(true)
                    && self.tets[ti].contains(&b)
                    && !result.contains(&ti)
                {
                    result.push(ti);
                }
            }
        }
        result
    }

    /// Check if replacing a tet would create a negative-volume tet.
    fn has_positive_volume(&self, t: &[usize; 4]) -> bool {
        let vol = orient_3d(
            self.vertices[t[0]],
            self.vertices[t[1]],
            self.vertices[t[2]],
            self.vertices[t[3]],
        );
        vol > 1e-14
    }

    /// Mark a tet as dead (tombstone).
    fn kill_tet(&mut self, ti: usize) {
        self.is_dead[ti] = true;
    }

    /// Append a new live tet, registering it in the vertex→tet adjacency.
    ///
    /// The registration is LOAD-BEARING (issue #60): without it, every tet
    /// created during a pass is invisible to `tets_around_edge`, so later
    /// splits in the same pass operate on INCOMPLETE rings - the missed
    /// children keep the original edge alive, the next pass re-collects it
    /// and splits it again, pushing an exact-duplicate midpoint vertex. That
    /// minted thousands of combinatorially-distinct coincident vertices
    /// (58% of the Cuboid@1.25 output!) plus partial-ring T-junction
    /// structure throughout the improved mesh.
    fn push_tet(&mut self, t: [usize; 4]) -> usize {
        let ti = self.tets.len();
        self.tets.push(t);
        self.is_dead.push(false);
        self.ensure_vert_tets();
        for &v in &t {
            if v < self.vert_tets.len() {
                self.vert_tets[v].push(ti);
            }
        }
        ti
    }

    /// Compact dead tets and rebuild adjacency.
    fn compact_and_rebuild(&mut self) {
        if self.is_dead.iter().any(|&d| d) {
            let mut write = 0;
            for read in 0..self.tets.len() {
                if !self.is_dead[read] {
                    self.tets[write] = self.tets[read];
                    write += 1;
                }
            }
            self.tets.truncate(write);
            self.is_dead.clear();
            self.is_dead.resize(write, false);
        }
        self.rebuild_adjacency();
    }
}

// ── Edge Split ──────────────────────────────────────────────────────────

/// Split all edges longer than `max_len`. Returns number of splits performed.
fn split_long_edges(mesh: &mut TetMesh, max_len: f64) -> usize {
    let max_len_sq = max_len * max_len;
    let mut splits = 0;

    // Collect edges to split - use Vec<bool> dedup instead of HashSet for speed.
    // Edge key: (min, max) mapped to a flat index isn't feasible for sparse vertex IDs,
    // so we use HashSet but with pre-allocated capacity.
    let mut edges_to_split: Vec<(usize, usize, f64)> = Vec::new();
    let est_edges = mesh.tets.len() * 2; // rough estimate of unique edges
    let mut seen_edges: HashSet<(usize, usize)> =
        HashSet::with_capacity_and_hasher(est_edges, Default::default());

    for (ti, t) in mesh.tets.iter().enumerate() {
        if mesh.is_dead.get(ti).copied().unwrap_or(false) {
            continue;
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let e = sorted_edge(t[i], t[j]);
                if seen_edges.insert(e) {
                    let d = dist_sq_3d(mesh.vertices[e.0], mesh.vertices[e.1]);
                    if d > max_len_sq {
                        edges_to_split.push((e.0, e.1, d));
                    }
                }
            }
        }
    }

    // Sort longest first (using cached distance). DETERMINISM: the edges are
    // gathered via a HashSet (randomised insertion order), so equal-length edges
    // would otherwise be split in a per-run-random order - which changes the
    // refined tet count run-to-run. Break ties by the (sorted) vertex pair so
    // identical input → identical refinement.
    edges_to_split.sort_unstable_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap()
            .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
    });

    for (va, vb, _) in edges_to_split {
        let pa = mesh.vertices[va];
        let pb = mesh.vertices[vb];
        // Always split at the LINEAR midpoint. Splitting the ring of tets
        // around edge (va,vb) at the linear midpoint is exactly
        // volume-preserving. Projecting the midpoint onto the boundary (the
        // former behaviour) was unsound: two boundary vertices are frequently
        // joined by an *interior chord* (not a surface edge), so projecting
        // its midpoint "onto the boundary" pulled interior points outward onto
        // the surface, bulging the surrounding tets OUTSIDE the triangulation
        // - inflating the volume (e.g. EllipticCylinder +52%) and pushing the
        // tet boundary off the DAGMC surface it must coincide with. The input
        // boundary triangulation is authoritative (it is the DAGMC surface),
        // so improvement must never move points off it.
        let mid = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];

        // Find all tets sharing this edge
        let ring = mesh.tets_around_edge(va, vb);
        if ring.is_empty() {
            continue;
        }

        // Add the midpoint vertex
        let mid_idx = mesh.vertices.len();
        // A midpoint between two on-surface vertices lies on that (planar)
        // boundary face, so it is on the surface too and must be pinned; if the
        // edge happened to be an interior chord instead, pinning it merely
        // forgoes a few swaps, which is the safe direction.
        let mid_pinned = mesh.is_boundary_vertex(va) && mesh.is_boundary_vertex(vb);
        mesh.vertices.push(mid);
        mesh.pinned.push(mid_pinned);
        mesh.ensure_vert_tets();

        // Each tet [va, vb, c, d] in the ring splits into two:
        // [va, mid, c, d] and [mid, vb, c, d]
        let mut new_tets: Vec<[usize; 4]> = Vec::new();
        let mut old_tet_indices: Vec<usize> = Vec::new();

        let mut valid = true;
        for &ti in &ring {
            let t = mesh.tets[ti];
            // Find the two vertices that are not va or vb
            let mut c = usize::MAX;
            let mut d = usize::MAX;
            for &v in &t {
                if v != va && v != vb {
                    if c == usize::MAX {
                        c = v;
                    } else {
                        d = v;
                    }
                }
            }
            if c == usize::MAX || d == usize::MAX {
                valid = false;
                break;
            }

            let mut t1 = [va, mid_idx, c, d];
            let mut t2 = [mid_idx, vb, c, d];

            // Fix orientation of split tets
            let v1 = orient_3d(
                mesh.vertices[t1[0]],
                mesh.vertices[t1[1]],
                mesh.vertices[t1[2]],
                mesh.vertices[t1[3]],
            );
            if v1 < 0.0 {
                t1.swap(2, 3);
            } else if v1.abs() < 1e-15 {
                valid = false;
                break;
            }

            let v2 = orient_3d(
                mesh.vertices[t2[0]],
                mesh.vertices[t2[1]],
                mesh.vertices[t2[2]],
                mesh.vertices[t2[3]],
            );
            if v2 < 0.0 {
                t2.swap(2, 3);
            } else if v2.abs() < 1e-15 {
                valid = false;
                break;
            }

            new_tets.push(t1);
            new_tets.push(t2);
            old_tet_indices.push(ti);
        }

        if !valid || new_tets.is_empty() {
            mesh.vertices.pop(); // revert midpoint
            continue;
        }

        // Mark old tets as dead, append new ones
        for &ti in &old_tet_indices {
            mesh.kill_tet(ti);
        }
        for nt in new_tets {
            mesh.push_tet(nt);
        }

        splits += 1;
    }

    if splits > 0 {
        mesh.compact_and_rebuild();
    }

    splits
}

// ── Edge Swap (2-to-3 flips) ───────────────────────────────────────────

/// Try to improve quality by swapping edges.
/// A 2-to-3 flip replaces 2 tets sharing a face with 3 tets sharing an edge.
/// Returns number of swaps performed.
fn swap_edges(mesh: &mut TetMesh, _target_h: f64) -> usize {
    let mut swaps = 0;

    // Iterate face_tets directly, collecting swaps to apply. DETERMINISM:
    // HashMap key iteration order is randomised per run, and the order faces are
    // swapped in changes the resulting mesh; sort by the (sorted) face vertices
    // so identical input → identical swap sequence.
    let mut faces: Vec<[usize; 3]> = mesh.face_tets.keys().copied().collect();
    faces.sort_unstable();

    for face in faces {
        let adj = match mesh.face_tets.get(&face) {
            Some(v) if v.len() == 2 => (v[0], v[1]),
            _ => continue,
        };
        let (ti, tj) = adj;
        if mesh.is_dead.get(ti).copied().unwrap_or(true)
            || mesh.is_dead.get(tj).copied().unwrap_or(true)
        {
            continue;
        }
        // STALENESS GUARD (issue #60): earlier swaps REUSE tet slots in
        // place, so a `face_tets` entry can point at a slot that no longer
        // contains this face - the apex lookup below would then operate on an
        // unrelated tet and the "flip" would commit overlapping garbage.
        if !face.iter().all(|v| mesh.tets[ti].contains(v))
            || !face.iter().all(|v| mesh.tets[tj].contains(v))
        {
            continue;
        }

        // Get the two apices (vertices not on the shared face)
        let apex_i = mesh.tets[ti].iter().find(|&&v| !face.contains(&v)).copied();
        let apex_j = mesh.tets[tj].iter().find(|&&v| !face.contains(&v)).copied();
        let (Some(ai), Some(aj)) = (apex_i, apex_j) else {
            continue;
        };

        // Don't swap boundary faces
        if mesh.is_boundary_vertex(face[0])
            && mesh.is_boundary_vertex(face[1])
            && mesh.is_boundary_vertex(face[2])
        {
            continue;
        }

        // Quality of current config
        let q_before = {
            let qi = optimize::tet_quality(
                mesh.vertices[mesh.tets[ti][0]],
                mesh.vertices[mesh.tets[ti][1]],
                mesh.vertices[mesh.tets[ti][2]],
                mesh.vertices[mesh.tets[ti][3]],
            );
            let qj = optimize::tet_quality(
                mesh.vertices[mesh.tets[tj][0]],
                mesh.vertices[mesh.tets[tj][1]],
                mesh.vertices[mesh.tets[tj][2]],
                mesh.vertices[mesh.tets[tj][3]],
            );
            qi.min(qj)
        };

        // 2-to-3 flip: create 3 tets sharing edge (ai, aj)
        let (a, b, c) = (face[0], face[1], face[2]);
        let new_tets = [[a, b, ai, aj], [b, c, ai, aj], [c, a, ai, aj]];

        // Check all 3 new tets have positive volume and better quality
        let mut valid = true;
        let mut q_after = f64::MAX;
        for nt in &new_tets {
            if !mesh.has_positive_volume(nt) {
                valid = false;
                break;
            }
            let q = optimize::tet_quality(
                mesh.vertices[nt[0]],
                mesh.vertices[nt[1]],
                mesh.vertices[nt[2]],
                mesh.vertices[nt[3]],
            );
            q_after = q_after.min(q);
        }

        if !valid {
            continue;
        }

        // Only swap if quality improves by at least 5%
        if q_after <= q_before * 1.05 {
            continue;
        }

        // Apply the swap: reuse ti and tj slots, push third
        mesh.tets[ti] = new_tets[0];
        mesh.tets[tj] = new_tets[1];
        mesh.push_tet(new_tets[2]);
        swaps += 1;
    }

    if swaps > 0 {
        mesh.compact_and_rebuild();
    }
    swaps
}

// ── Main improvement loop ───────────────────────────────────────────────

/// Improve a tet mesh to bring all edges close to target_edge_length.
///
/// Iterates: split → swap until convergence.
// One more argument than clippy's default taste allows. The alternative is an
// options struct for three call sites, which would obscure rather than clarify;
// `predicates3d` takes the same allow for the same reason.
#[allow(clippy::too_many_arguments)]
pub fn improve_mesh(
    vertices: &mut Vec<[f64; 3]>,
    tets: &mut Vec<[usize; 4]>,
    n_boundary: usize,
    boundary_vertices: &[[f64; 3]],
    boundary_triangles: &[[usize; 3]],
    target_edge_length: f64,
    max_passes: usize,
    pinned: Option<&[bool]>,
) {
    let split_threshold = target_edge_length * 1.4;
    let _collapse_threshold = target_edge_length * 0.4;
    // Boundary stays fixed: edge splits use the linear midpoint, so the input
    // surface triangulation (the authoritative DAGMC surface) is never moved.
    let _ = (boundary_vertices, boundary_triangles);

    let mut mesh = TetMesh::new_pinned(vertices.clone(), tets.clone(), n_boundary, pinned);

    for pass in 0..max_passes {
        let splits = split_long_edges(&mut mesh, split_threshold);
        let swaps = swap_edges(&mut mesh, target_edge_length);
        eprintln!(
            "    [improve] pass {}/{}: {} splits, {} swaps, {} tets",
            pass + 1,
            max_passes,
            splits,
            swaps,
            mesh.tets.len()
        );

        if splits == 0 && swaps == 0 {
            break;
        }
        // Early exit when work is negligible relative to mesh size (<0.01% of tets modified).
        // Late passes on large meshes do tiny work but still rebuild adjacency.
        if (splits + swaps) * 10_000 < mesh.tets.len() {
            eprintln!("    [improve] converged (< 0.01% tets modified)");
            break;
        }
    }

    // Smoothing disabled - causes volume loss near boundary

    // Fix orientation after all operations
    for t in &mut mesh.tets {
        let vol = predicates3d::tet_volume(
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
            mesh.vertices[t[3]],
        );
        if vol < 0.0 {
            t.swap(2, 3);
        }
    }

    // Remove degenerate tets. Threshold 1e-12 ABSOLUTE - the quality bar the
    // zoo asserts (and far below any real tet: the smallest legitimate
    // elements are ~(0.1·target)³). Flat pancakes in the 1e-15..1e-12 band
    // survive every flip/swap (their removal configurations are themselves
    // degenerate - the issue-#47/#60 singleton class), so dropping is the
    // only local resolution; the cost is a zero-volume void whose 4 faces
    // show up as cosmetic non-manifold edges in the tet-boundary complex
    // (volume and tallies unaffected - the void carries no measure).
    mesh.tets.retain(|t| {
        predicates3d::tet_volume(
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
            mesh.vertices[t[3]],
        )
        .abs()
            > 1e-12
    });

    *vertices = mesh.vertices;
    *tets = mesh.tets;
}

// ── Tests ───────────────────────────────────────────────────────────────

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_reduces_max_edge() {
        // Use Delaunay to get a proper mesh with correct orientation
        use super::super::delaunay3d::Delaunay3D;

        let mut pts = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [0.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ];
        let dt = Delaunay3D::new(&pts);
        let mut tets = dt.extract_tets();
        let n_boundary = 8;

        // Fix orientation
        for t in &mut tets {
            let vol = predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
            if vol < 0.0 {
                t.swap(2, 3);
            }
        }

        let mut max_before = 0.0_f64;
        for t in &tets {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let d = dist_sq_3d(pts[t[i]], pts[t[j]]).sqrt();
                    max_before = max_before.max(d);
                }
            }
        }

        let tets_before = tets.len();
        let bv = pts[..n_boundary].to_vec();
        let bt: Vec<[usize; 3]> = vec![]; // no boundary tris for this test
        improve_mesh(&mut pts, &mut tets, n_boundary, &bv, &bt, 0.8, 10, None);

        let mut max_after = 0.0_f64;
        for t in &tets {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let d = dist_sq_3d(pts[t[i]], pts[t[j]]).sqrt();
                    max_after = max_after.max(d);
                }
            }
        }

        assert!(
            tets.len() >= tets_before,
            "Should have at least as many tets"
        );
        // Max edge should be bounded by split threshold (1.4 * 0.8 = 1.12)
        assert!(
            max_after < 1.2,
            "Max edge {max_after:.2} should be < 1.2 after improvement"
        );

        for t in &tets {
            let vol = predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
            assert!(vol > 0.0, "Tet has non-positive volume: {vol}");
        }
    }
}
