//! Pure-geometry utility functions that don't depend on OCC.
//!
//! These are helpers ported from the Python `cad` package for
//! performance and to keep geometry logic in Rust.

use std::collections::{HashMap, HashSet, VecDeque};

/// Remap triangle vertex indices, filtering degenerate triangles.
///
/// A triangle is degenerate if any two of its remapped vertices are equal.
/// When `dedup` is true, also removes duplicate triangles (same sorted
/// vertex set) - needed for UV-seam faces on spheres and tori.
pub fn remap_tris(
    tris: &[[usize; 3]],
    remap: &[usize],
    dedup: bool,
    flip: bool,
) -> Vec<[usize; 3]> {
    let mut seen: Option<HashSet<[usize; 3]>> = if dedup { Some(HashSet::new()) } else { None };
    let mut result = Vec::with_capacity(tris.len());

    for t in tris {
        let (a, b, c) = if flip {
            (remap[t[0]], remap[t[2]], remap[t[1]])
        } else {
            (remap[t[0]], remap[t[1]], remap[t[2]])
        };

        // Skip degenerate triangles.
        if a == b || b == c || a == c {
            continue;
        }

        if let Some(ref mut set) = seen {
            let mut key = [a, b, c];
            key.sort();
            if !set.insert(key) {
                continue;
            }
        }

        result.push([a, b, c]);
    }

    result
}

/// Remap many triangle buckets against ONE shared remap table.
///
/// Batched form of [`remap_tris`]: the per-bucket Python call converted
/// the full vertex-length remap list at the PyO3 boundary once per face
/// (O(faces x vertices) marshalling - 1.9 s of a 9.8 s SCDR surface
/// run). One crossing remaps every face bucket.
pub fn remap_tris_many(
    buckets: &[Vec<[usize; 3]>],
    remap: &[usize],
    dedup: bool,
    flip: bool,
) -> Vec<Vec<[usize; 3]>> {
    buckets
        .iter()
        .map(|tris| remap_tris(tris, remap, dedup, flip))
        .collect()
}

/// Compute the bounding-box diagonal of a list of 3D vertices.
pub fn bbox_diagonal(vertices: &[[f64; 3]]) -> f64 {
    if vertices.is_empty() {
        return 0.0;
    }
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for v in vertices {
        for i in 0..3 {
            if v[i] < min[i] {
                min[i] = v[i];
            }
            if v[i] > max[i] {
                max[i] = v[i];
            }
        }
    }
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Collect UV points from boundary edges, deduplicating shared endpoints.
///
/// Each edge is a list of UV points. If `reversed[i]` is true the points
/// are iterated in reverse order. Adjacent edges that share an endpoint
/// (within 1e-12) have the duplicate removed. The closing vertex is also
/// removed if it matches the first.
pub fn collect_wire_uv(edges: &[Vec<[f64; 2]>], reversed: &[bool]) -> Vec<[f64; 2]> {
    let mut wire: Vec<[f64; 2]> = Vec::new();

    for (pts, &rev) in edges.iter().zip(reversed.iter()) {
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if rev {
            Box::new(pts.iter().rev())
        } else {
            Box::new(pts.iter())
        };

        for (i, uv) in iter.enumerate() {
            if i == 0 && !wire.is_empty() {
                let last = wire.last().unwrap();
                if (last[0] - uv[0]).abs() < 1e-12 && (last[1] - uv[1]).abs() < 1e-12 {
                    continue;
                }
            }
            wire.push(*uv);
        }
    }

    // Remove closing duplicate.
    if wire.len() >= 2 {
        let first = wire[0];
        let last = *wire.last().unwrap();
        if (first[0] - last[0]).abs() < 1e-12 && (first[1] - last[1]).abs() < 1e-12 {
            wire.pop();
        }
    }

    wire
}

// ---------------------------------------------------------------------------
//  Vertex merging
// ---------------------------------------------------------------------------

/// Compute a grid key for a vertex.  Matches the Python floor-towards-negative-infinity logic.
#[inline]
fn grid_key(v: &[f64; 3], inv_cell: f64) -> (i64, i64, i64) {
    #[inline]
    fn coord(x: f64, inv: f64) -> i64 {
        let f = x * inv;
        if x >= 0.0 {
            f as i64
        } else {
            (f as i64) - 1
        }
    }
    (
        coord(v[0], inv_cell),
        coord(v[1], inv_cell),
        coord(v[2], inv_cell),
    )
}

/// Auto-compute merge tolerance from a bounding-box diagonal.
fn auto_tol(vertices: &[[f64; 3]]) -> f64 {
    let diag = bbox_diagonal(vertices);
    f64::max(diag * 1e-8, 1e-10)
}

/// Merge duplicate 3D vertices using a spatial hash grid.
///
/// Returns `(new_vertices, remap)` where `remap[old_index]` gives the
/// new index in the compacted vertex list.
///
/// If `tol` is `None`, the tolerance is `max(bbox_diagonal * 1e-8, 1e-10)`.
pub fn merge_vertices_with_remap(
    vertices: &[[f64; 3]],
    tol: Option<f64>,
) -> (Vec<[f64; 3]>, Vec<usize>) {
    let n = vertices.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let tol = tol.unwrap_or_else(|| auto_tol(vertices));
    let cell = tol * 2.0;
    let inv_cell = if cell > 0.0 { 1.0 / cell } else { 1.0 };
    let tol_sq = tol * tol;

    // canonical[i] = the first vertex index that i merges into
    let mut canonical: Vec<usize> = (0..n).collect();
    // spatial hash: grid_key -> list of vertex indices (non-merged representatives)
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

    for i in 0..n {
        let v = &vertices[i];
        let gk = grid_key(v, inv_cell);
        let mut merged = false;

        'outer: for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let nk = (gk.0 + dx, gk.1 + dy, gk.2 + dz);
                    if let Some(bucket) = grid.get(&nk) {
                        for &j in bucket {
                            let vj = &vertices[j];
                            let d = (v[0] - vj[0]).powi(2)
                                + (v[1] - vj[1]).powi(2)
                                + (v[2] - vj[2]).powi(2);
                            if d < tol_sq {
                                canonical[i] = canonical[j];
                                merged = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if !merged {
            grid.entry(gk).or_default().push(i);
        }
    }

    // Compact: assign new contiguous indices
    let mut old_to_new = vec![0usize; n];
    let mut new_verts: Vec<[f64; 3]> = Vec::new();
    // Map from canonical root -> new index
    let mut root_to_new: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let root = canonical[i];
        let new_idx = match root_to_new.get(&root) {
            Some(&idx) => idx,
            None => {
                let idx = new_verts.len();
                new_verts.push(vertices[root]);
                root_to_new.insert(root, idx);
                idx
            }
        };
        old_to_new[i] = new_idx;
    }

    (new_verts, old_to_new)
}

/// Merge vertices from DIFFERENT faces only (preserves within-face topology).
///
/// `face_ranges` is a list of `(start, end)` index pairs, one per face.
/// Two vertices are only merged if they belong to different faces.
///
/// Returns `(new_vertices, remap)`.
pub fn merge_vertices_between_faces(
    vertices: &[[f64; 3]],
    face_ranges: &[(usize, usize)],
    tol: Option<f64>,
) -> (Vec<[f64; 3]>, Vec<usize>) {
    let n = vertices.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let tol = tol.unwrap_or_else(|| auto_tol(vertices));
    let cell = tol * 2.0;
    let inv_cell = if cell > 0.0 { 1.0 / cell } else { 1.0 };
    let tol_sq = tol * tol;

    // Build face membership: vert_face[i] = which face_range index owns vertex i
    // Use usize::MAX as sentinel for "no face".
    let mut vert_face = vec![usize::MAX; n];
    for (fi, &(s, e)) in face_ranges.iter().enumerate() {
        for vf in vert_face.iter_mut().take(e).skip(s) {
            *vf = fi;
        }
    }

    let mut canonical: Vec<usize> = (0..n).collect();
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

    for i in 0..n {
        let v = &vertices[i];
        let gk = grid_key(v, inv_cell);
        let mut merged = false;

        'outer: for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let nk = (gk.0 + dx, gk.1 + dy, gk.2 + dz);
                    if let Some(bucket) = grid.get(&nk) {
                        for &j in bucket {
                            // Only merge across different faces
                            if vert_face[i] == vert_face[j] {
                                continue;
                            }
                            let vj = &vertices[j];
                            let d = (v[0] - vj[0]).powi(2)
                                + (v[1] - vj[1]).powi(2)
                                + (v[2] - vj[2]).powi(2);
                            if d < tol_sq {
                                canonical[i] = canonical[j];
                                merged = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if !merged {
            grid.entry(gk).or_default().push(i);
        }
    }

    // Compact
    let mut old_to_new = vec![0usize; n];
    let mut new_verts: Vec<[f64; 3]> = Vec::new();
    let mut root_to_new: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let root = canonical[i];
        let new_idx = match root_to_new.get(&root) {
            Some(&idx) => idx,
            None => {
                let idx = new_verts.len();
                new_verts.push(vertices[root]);
                root_to_new.insert(root, idx);
                idx
            }
        };
        old_to_new[i] = new_idx;
    }

    (new_verts, old_to_new)
}

/// Merge duplicate 3D vertices and remap triangle indices.
///
/// Thin wrapper around [`merge_vertices_with_remap`] + [`remap_tris`].
/// Removes degenerate triangles but does not deduplicate.
pub fn merge_vertices(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    tol: Option<f64>,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (new_verts, remap) = merge_vertices_with_remap(vertices, tol);
    let new_tris = remap_tris(triangles, &remap, false, false);
    (new_verts, new_tris)
}

// ---------------------------------------------------------------------------
//  Normal fixing via BFS
// ---------------------------------------------------------------------------

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns `true` if the ray from `origin` in `dir` intersects the triangle
/// `(a, b, c)` at `t > eps`.
#[inline]
fn ray_hits_triangle(
    origin: &[f64; 3],
    dir: &[f64; 3],
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> bool {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let h = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];
    let det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if det.abs() < 1e-14 {
        return false;
    }
    let f = 1.0 / det;
    let s = [origin[0] - a[0], origin[1] - a[1], origin[2] - a[2]];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    t > 1e-14
}

/// Signed volume contribution of one triangle (with respect to the origin).
#[inline]
fn signed_volume_tri(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    (a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0]))
        / 6.0
}

/// Undirected edge `(min, max)` → the triangles on it, each with the directed
/// half-edge `(u, v)` it traverses. Two triangles agreeing on direction across a
/// shared edge means one of them is wound the wrong way.
type EdgeIncidence = std::collections::HashMap<(usize, usize), Vec<(usize, (usize, usize))>>;

/// Signed volume enclosed by a triangulated surface, via the divergence
/// theorem: V = (1/6) Σ a·(b×c) over all triangles.
///
/// The per-face surface meshes that feed the volume mesher are each oriented by
/// their own surface normal, so the concatenated boundary triangles are NOT
/// guaranteed to wind consistently (some point outward, some inward). A naive
/// `Σ a·(b×c)` would then partially cancel and underestimate (or zero out) the
/// volume. To be winding-robust we first propagate a CONSISTENT orientation
/// across the closed manifold (BFS over shared edges: neighbours that traverse
/// a shared edge in the same direction are flipped), then sum. The result is
/// `|V|`, independent of whether the consistent orientation came out inward or
/// outward. This is exact (no tolerance) and is the reference the carved-volume
/// gate is compared against.
pub fn surface_enclosed_volume(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    use std::collections::HashMap;

    let (verts, tris) = (vertices, triangles);

    let n = tris.len();
    if n == 0 {
        return 0.0;
    }

    // Map each undirected edge → the (at most two, for a manifold) triangles
    // incident to it, recording the directed half-edge (u, v) per triangle so
    // we can tell whether two neighbours agree or disagree on orientation.
    // Key: sorted (min, max). Value: list of (tri_idx, directed (u, v)).
    let mut edge_map: EdgeIncidence = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for &(u, v) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = (u.min(v), u.max(v));
            edge_map.entry(key).or_default().push((ti, (u, v)));
        }
    }

    // BFS orientation propagation. `flip[ti]` = should triangle ti be reversed
    // to match the seed component's consistent orientation. `seen` marks
    // visited triangles. Disconnected components each seed independently.
    let mut flip = vec![false; n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    for seed in 0..n {
        if seen[seed] {
            continue;
        }
        seen[seed] = true;
        stack.push(seed);
        while let Some(ti) = stack.pop() {
            let t = tris[ti];
            // Directed half-edges of ti AS CURRENTLY ORIENTED (accounting for
            // its own flip state).
            let dir_edges = if flip[ti] {
                [(t[1], t[0]), (t[2], t[1]), (t[0], t[2])]
            } else {
                [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
            };
            for &(u, v) in &dir_edges {
                let key = (u.min(v), u.max(v));
                if let Some(incident) = edge_map.get(&key) {
                    for &(nbr, (nu, nv)) in incident {
                        if nbr == ti || seen[nbr] {
                            continue;
                        }
                        // Neighbour's directed half-edge on this shared edge,
                        // accounting for its current (unflipped) orientation.
                        // For a CONSISTENT orientation across a shared edge the
                        // two triangles must traverse it in OPPOSITE directions.
                        // If they traverse it the SAME way, the neighbour must be
                        // flipped.
                        let nbr_same_dir = (nu, nv) == (u, v);
                        flip[nbr] = nbr_same_dir;
                        seen[nbr] = true;
                        stack.push(nbr);
                    }
                }
            }
        }
    }

    // ── NESTED COMPONENTS (issue #37, hollow solids) ──
    // A hollow solid's surface has several connected components (outer shell +
    // sealed cavities, e.g. the coil casing around its winding pack). Each
    // component's |signed volume| measures the region IT encloses, but the
    // BFS orients components independently, so a naive |Σ| can come out as
    // V_outer + V_cavity instead of the material volume V_outer − V_cavity
    // (observed: casing 2.667e9 vs true 7.41e8). Compute per-component
    // volumes, then sign each by its NESTING DEPTH: a component contained in
    // an odd number of other components is a cavity boundary → subtract.
    // Containment is decided by strict AABB inclusion - exact for the
    // per-solid meshing domain (cavities lie strictly inside the outer shell;
    // disjoint solids are meshed separately).
    //
    // First: component id per triangle (re-run the same BFS grouping).
    let mut comp = vec![usize::MAX; n];
    let mut ncomp = 0usize;
    {
        let mut stack: Vec<usize> = Vec::new();
        for seed in 0..n {
            if comp[seed] != usize::MAX {
                continue;
            }
            comp[seed] = ncomp;
            stack.push(seed);
            while let Some(ti) = stack.pop() {
                let t = tris[ti];
                for &(u, v) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let key = (u.min(v), u.max(v));
                    if let Some(incident) = edge_map.get(&key) {
                        for &(nbr, _) in incident {
                            if comp[nbr] == usize::MAX {
                                comp[nbr] = ncomp;
                                stack.push(nbr);
                            }
                        }
                    }
                }
            }
            ncomp += 1;
        }
    }

    // Per-component signed volume (consistent within the component thanks to
    // the flip propagation) and AABB.
    let mut vol_c = vec![0.0f64; ncomp];
    let mut bb_min = vec![[f64::INFINITY; 3]; ncomp];
    let mut bb_max = vec![[f64::NEG_INFINITY; 3]; ncomp];
    for (ti, t) in tris.iter().enumerate() {
        let (i0, i1, i2) = if flip[ti] {
            (t[0], t[2], t[1])
        } else {
            (t[0], t[1], t[2])
        };
        let a = verts[i0];
        let b = verts[i1];
        let c = verts[i2];
        let cross = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        vol_c[comp[ti]] += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
        for &vi in &[i0, i1, i2] {
            for k in 0..3 {
                bb_min[comp[ti]][k] = bb_min[comp[ti]][k].min(verts[vi][k]);
                bb_max[comp[ti]][k] = bb_max[comp[ti]][k].max(verts[vi][k]);
            }
        }
    }

    let mut total = 0.0f64;
    for c in 0..ncomp {
        // Nesting depth: number of OTHER components whose AABB strictly
        // contains this component's AABB.
        let depth = (0..ncomp)
            .filter(|&d| {
                d != c && (0..3).all(|k| bb_min[d][k] < bb_min[c][k] && bb_max[d][k] > bb_max[c][k])
            })
            .count();
        let sign = if depth % 2 == 1 { -1.0 } else { 1.0 };
        total += sign * (vol_c[c] / 6.0).abs();
    }
    total.abs()
}

/// Total volume of a tetrahedral mesh: the sum of the absolute volume of every
/// tetrahedron. Used to verify that an interior tetrahedralisation fills the
/// same region its boundary surface encloses.
pub fn tet_mesh_volume(vertices: &[[f64; 3]], tetrahedra: &[[usize; 4]]) -> f64 {
    tetrahedra
        .iter()
        .map(|t| {
            let a = vertices[t[0]];
            let b = vertices[t[1]];
            let c = vertices[t[2]];
            let d = vertices[t[3]];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            // ab · (ac × ad) / 6
            let cross = [
                ac[1] * ad[2] - ac[2] * ad[1],
                ac[2] * ad[0] - ac[0] * ad[2],
                ac[0] * ad[1] - ac[1] * ad[0],
            ];
            (ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]).abs() / 6.0
        })
        .sum()
}

/// Fix triangle winding using BFS edge propagation + signed volume.
///
/// For each solid: make all normals consistent via BFS across shared edges,
/// then check the signed volume and flip if negative.  Multi-component
/// solids (hollow volumes) use a ray-cast containment test to identify holes
/// and ensure their normals point inward.
///
/// The input and output use `HashMap<usize, HashMap<usize, Vec<[usize; 3]>>>`
/// keyed by `solid_id -> face_id -> triangles`.
pub fn fix_normals_bfs(
    vertices: &[[f64; 3]],
    triangles_by_solid_by_face: &HashMap<usize, HashMap<usize, Vec<[usize; 3]>>>,
) -> HashMap<usize, HashMap<usize, Vec<[usize; 3]>>> {
    let mut result: HashMap<usize, HashMap<usize, Vec<[usize; 3]>>> = HashMap::new();

    for (&solid_id, faces) in triangles_by_solid_by_face {
        // Flatten all triangles, recording which face-range each belongs to
        let mut all_tris: Vec<[usize; 3]> = Vec::new();
        let mut face_ranges: Vec<(usize, usize, usize)> = Vec::new(); // (face_id, start, end)

        // Sort face ids for deterministic iteration
        let mut face_ids: Vec<usize> = faces.keys().copied().collect();
        face_ids.sort();

        for &fid in &face_ids {
            let face_tris = &faces[&fid];
            let start = all_tris.len();
            all_tris.extend_from_slice(face_tris);
            face_ranges.push((fid, start, all_tris.len()));
        }

        let n = all_tris.len();
        if n == 0 {
            result.insert(solid_id, HashMap::new());
            continue;
        }

        // Build edge -> [(tri_idx, v0, v1)] adjacency
        #[allow(clippy::type_complexity)]
        let mut edge_adj: HashMap<(usize, usize), Vec<(usize, usize, usize)>> = HashMap::new();
        for (ti, tri) in all_tris.iter().enumerate() {
            for k in 0..3 {
                let v0 = tri[k];
                let v1 = tri[(k + 1) % 3];
                let key = (v0.min(v1), v0.max(v1));
                edge_adj.entry(key).or_default().push((ti, v0, v1));
            }
        }

        // BFS per connected component
        let mut flip = vec![false; n];
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for seed in 0..n {
            if visited[seed] {
                continue;
            }
            visited[seed] = true;
            let mut comp = vec![seed];
            let mut queue = VecDeque::new();
            queue.push_back(seed);

            while let Some(ti) = queue.pop_front() {
                let tri = all_tris[ti];
                for k in 0..3 {
                    // Get the effective edge direction after potential flip
                    let (v0, v1) = if flip[ti] {
                        let flipped = [tri[0], tri[2], tri[1]];
                        (flipped[k], flipped[(k + 1) % 3])
                    } else {
                        (tri[k], tri[(k + 1) % 3])
                    };
                    let key = (v0.min(v1), v0.max(v1));
                    if let Some(neighbors) = edge_adj.get(&key) {
                        for &(tj, u0, u1) in neighbors {
                            if visited[tj] {
                                continue;
                            }
                            visited[tj] = true;
                            // If shared edge has same direction, neighbor needs flipping
                            if u0 == v0 && u1 == v1 {
                                flip[tj] = true;
                            }
                            comp.push(tj);
                            queue.push_back(tj);
                        }
                    }
                }
            }

            components.push(comp);
        }

        // Apply flips
        for ti in 0..n {
            if flip[ti] {
                let t = all_tris[ti];
                all_tris[ti] = [t[0], t[2], t[1]];
            }
        }

        // Compute signed volume per component
        let comp_vols: Vec<f64> = components
            .iter()
            .map(|comp| {
                comp.iter()
                    .map(|&ti| {
                        let tri = &all_tris[ti];
                        signed_volume_tri(&vertices[tri[0]], &vertices[tri[1]], &vertices[tri[2]])
                    })
                    .sum()
            })
            .collect();

        if components.len() == 1 {
            // Single component: just check sign
            if comp_vols[0] < 0.0 {
                for &ti in &components[0] {
                    let t = all_tris[ti];
                    all_tris[ti] = [t[0], t[2], t[1]];
                }
            }
        } else {
            // Multiple components: determine holes via ray-cast
            let ray_dir = [1.0, 0.00013, 0.00017];
            let mut is_hole = vec![false; components.len()];

            for ci in 0..components.len() {
                let test_pt = vertices[all_tris[components[ci][0]][0]];
                for (cj, comp_cj) in components.iter().enumerate() {
                    if ci == cj {
                        continue;
                    }
                    let mut crossings = 0usize;
                    for &ti in comp_cj {
                        let tri = &all_tris[ti];
                        if ray_hits_triangle(
                            &test_pt,
                            &ray_dir,
                            &vertices[tri[0]],
                            &vertices[tri[1]],
                            &vertices[tri[2]],
                        ) {
                            crossings += 1;
                        }
                    }
                    if crossings % 2 == 1 {
                        is_hole[ci] = true;
                        break;
                    }
                }
            }

            for (ci, (comp, &cv)) in components.iter().zip(comp_vols.iter()).enumerate() {
                if is_hole[ci] {
                    // Hole: normals should give negative volume
                    if cv > 0.0 {
                        for &ti in comp {
                            let t = all_tris[ti];
                            all_tris[ti] = [t[0], t[2], t[1]];
                        }
                    }
                } else {
                    // Outer: normals should give positive volume
                    if cv < 0.0 {
                        for &ti in comp {
                            let t = all_tris[ti];
                            all_tris[ti] = [t[0], t[2], t[1]];
                        }
                    }
                }
            }
        }

        // Write back into per-face structure
        let mut solid_faces: HashMap<usize, Vec<[usize; 3]>> = HashMap::new();
        for &(fid, start, end) in &face_ranges {
            solid_faces.insert(fid, all_tris[start..end].to_vec());
        }
        result.insert(solid_id, solid_faces);
    }

    result
}

/// Fill boundary holes in a triangle mesh using fan triangulation.
///
/// Finds boundary edges (edges with exactly 1 adjacent triangle),
/// traces directed boundary loops following triangle winding, and
/// fills each loop with fan triangles from the first vertex.
///
/// Returns the original triangles plus any new hole-filling triangles.
pub fn fill_boundary_holes(_vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut result: Vec<[usize; 3]> = triangles.to_vec();

    // 1. Count edge occurrences (sorted pair)
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let a = t[k];
            let b = t[(k + 1) % 3];
            let key = (a.min(b), a.max(b));
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // 2. Boundary edges = edges with count == 1
    let boundary_edges: Vec<(usize, usize)> = edge_count
        .iter()
        .filter(|&(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .collect();

    if boundary_edges.is_empty() {
        return result;
    }

    // 3. For each boundary edge, find the directed boundary direction.
    //    If a triangle has edge a→b, the boundary direction is b→a (opposite).
    let mut directed: HashMap<usize, Vec<usize>> = HashMap::new();
    let boundary_set: HashSet<(usize, usize)> = boundary_edges.iter().copied().collect();

    for t in triangles {
        for k in 0..3 {
            let a = t[k];
            let b = t[(k + 1) % 3];
            let sorted_key = (a.min(b), a.max(b));
            if boundary_set.contains(&sorted_key) {
                // Triangle has edge a→b, so boundary direction is b→a
                directed.entry(b).or_default().push(a);
            }
        }
    }

    // 4. Trace loops: follow directed chains
    let mut used_edges: HashSet<(usize, usize)> = HashSet::new();

    // Sort keys for deterministic iteration
    let mut start_vertices: Vec<usize> = directed.keys().copied().collect();
    start_vertices.sort();

    // 5. Build set of existing triangle keys for deduplication
    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    for t in triangles {
        let mut key = *t;
        key.sort();
        seen.insert(key);
    }

    for start_v in &start_vertices {
        if let Some(destinations) = directed.get(start_v) {
            for &first_nxt in destinations {
                let edge_key = (*start_v, first_nxt);
                if used_edges.contains(&edge_key) {
                    continue;
                }
                let mut loop_verts = vec![*start_v, first_nxt];
                used_edges.insert(edge_key);
                let mut current = first_nxt;

                loop {
                    // Find an unused next vertex
                    let mut nxt = None;
                    if let Some(candidates) = directed.get(&current) {
                        for &candidate in candidates {
                            let ek = (current, candidate);
                            if !used_edges.contains(&ek) {
                                nxt = Some(candidate);
                                break;
                            }
                        }
                    }
                    match nxt {
                        Some(n) if n == *start_v => break,
                        Some(n) => {
                            used_edges.insert((current, n));
                            loop_verts.push(n);
                            current = n;
                        }
                        None => break,
                    }
                }

                // Fan-triangulate if we have a closed loop of 3+ vertices
                if loop_verts.len() >= 3 {
                    for i in 1..loop_verts.len() - 1 {
                        let tri = [loop_verts[0], loop_verts[i], loop_verts[i + 1]];
                        let mut key = tri;
                        key.sort();
                        if !seen.contains(&key) {
                            seen.insert(key);
                            result.push(tri);
                        }
                    }
                }
            }
        }
    }

    result
}

/// Close open edges by greedily adding triangles.
///
/// For each open edge (shared by exactly 1 triangle), finds the best
/// vertex to form a closing triangle on the opposite side.  The best
/// vertex is one that:
/// 1. Is connected to both endpoints via other open edges (forms a
///    boundary chain), OR
/// 2. Is the nearest vertex that produces a valid (non-overlapping) triangle
///
/// This is more robust than `fill_boundary_holes` for complex boundary
/// patterns (non-simple loops, branching vertices) that arise from
/// periodic seam stitching.
pub fn close_open_edges(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut result: Vec<[usize; 3]> = triangles.to_vec();

    for _ in 0..10 {
        // Build edge adjacency
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (ti, t) in result.iter().enumerate() {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let key = (a.min(b), a.max(b));
                *edge_count.entry(key).or_insert(0) += 1;
                edge_tris.entry(key).or_default().push(ti);
            }
        }

        // Find open edges
        let open_edges: Vec<(usize, usize)> = edge_count
            .iter()
            .filter(|&(_, &c)| c == 1)
            .map(|(&e, _)| e)
            .collect();

        if open_edges.is_empty() {
            break;
        }

        // Build adjacency on open edges: for each vertex, which other
        // vertices is it connected to via an open edge?
        let mut open_adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in &open_edges {
            open_adj.entry(a).or_default().push(b);
            open_adj.entry(b).or_default().push(a);
        }

        // Existing triangle keys for dedup
        let mut seen: HashSet<[usize; 3]> = HashSet::new();
        for t in &result {
            let mut key = *t;
            key.sort();
            seen.insert(key);
        }

        let mut added = false;

        // For each open edge, try to find a closing triangle
        for &(ea, eb) in &open_edges {
            // Find vertices that are open-edge neighbors of BOTH ea and eb
            let nbrs_a: HashSet<usize> = open_adj
                .get(&ea)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();
            let nbrs_b: HashSet<usize> = open_adj
                .get(&eb)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();

            let common: Vec<usize> = nbrs_a
                .intersection(&nbrs_b)
                .copied()
                .filter(|&v| v != ea && v != eb)
                .collect();

            if common.is_empty() {
                continue;
            }

            // Pick the common vertex closest to the edge midpoint
            let mid = [
                (vertices[ea][0] + vertices[eb][0]) * 0.5,
                (vertices[ea][1] + vertices[eb][1]) * 0.5,
                (vertices[ea][2] + vertices[eb][2]) * 0.5,
            ];
            let mut best_v = common[0];
            let mut best_d = f64::MAX;
            for &cv in &common {
                let dx = vertices[cv][0] - mid[0];
                let dy = vertices[cv][1] - mid[1];
                let dz = vertices[cv][2] - mid[2];
                let d = dx * dx + dy * dy + dz * dz;
                if d < best_d {
                    best_d = d;
                    best_v = cv;
                }
            }

            let mut key = [ea, eb, best_v];
            key.sort();
            if seen.contains(&key) {
                continue;
            }

            // Check that this triangle doesn't create a non-manifold edge
            let edges_to_check = [
                (ea.min(best_v), ea.max(best_v)),
                (eb.min(best_v), eb.max(best_v)),
            ];
            let mut creates_nm = false;
            for &ek in &edges_to_check {
                if let Some(&c) = edge_count.get(&ek) {
                    if c >= 2 {
                        creates_nm = true;
                        break;
                    }
                }
            }
            if creates_nm {
                continue;
            }

            seen.insert(key);
            // Use consistent winding - compute normal dot with edge normal
            let tri = [ea, eb, best_v];
            result.push(tri);
            added = true;
        }

        if !added {
            break;
        }
    }

    // Final dedup + non-manifold cleanup
    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    result.retain(|t| {
        let mut key = *t;
        key.sort();
        seen.insert(key)
    });

    result
}

/// Merge only boundary vertices (indices 0..n_boundary) by 3D proximity.
///
/// Interior vertices (index >= n_boundary) are never merged, preserving
/// CDT topology.  Returns `(new_vertices, remap)` where `remap[old_idx]`
/// gives the new index.
pub fn merge_boundary_only_3d(
    vertices: &[[f64; 3]],
    n_boundary: usize,
) -> (Vec<[f64; 3]>, Vec<usize>) {
    let n = vertices.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n_boundary < 2 {
        return (vertices.to_vec(), (0..n).collect());
    }

    // Auto-compute tolerance from boundary vertex extent
    let mut max_span: f64 = 0.0;
    for k in 0..3 {
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for v in vertices.iter().take(n_boundary.min(n)) {
            lo = lo.min(v[k]);
            hi = hi.max(v[k]);
        }
        max_span = max_span.max(hi - lo);
    }
    let tol_sq = (max_span * 1e-6) * (max_span * 1e-6);

    let mut new_verts: Vec<[f64; 3]> = Vec::new();
    let mut remap = vec![0usize; n];

    // Merge boundary vertices
    for (i, &vi) in vertices.iter().enumerate().take(n_boundary.min(n)) {
        let mut merged = false;
        for (j, nv) in new_verts.iter().enumerate() {
            let dx = vi[0] - nv[0];
            let dy = vi[1] - nv[1];
            let dz = vi[2] - nv[2];
            if dx * dx + dy * dy + dz * dz < tol_sq {
                remap[i] = j;
                merged = true;
                break;
            }
        }
        if !merged {
            remap[i] = new_verts.len();
            new_verts.push(vi);
        }
    }

    // Interior vertices appended without merging
    for i in n_boundary.min(n)..n {
        remap[i] = new_verts.len();
        new_verts.push(vertices[i]);
    }

    (new_verts, remap)
}

/// Combined mesh cleanup: dedup + non-manifold removal + hole fill + edge close.
///
/// Iteratively:
/// 1. Removes duplicate triangles (same sorted vertex triple)
/// 2. Removes excess triangles on non-manifold edges (keeps 2 with largest area)
/// 3. Fills boundary holes via loop tracing
/// 4. Closes remaining open edges via greedy bridging
pub fn cleanup_non_manifold(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
    use std::collections::BTreeMap;
    let mut tris = triangles.to_vec();

    for _ in 0..5 {
        let mut changed = false;

        // Step 1: Remove duplicate triangles. Use BTreeSet semantics via
        // a sorted Vec to keep order deterministic across runs (HashSet
        // iteration order depends on a per-process random seed).
        let before = tris.len();
        let mut seen: std::collections::BTreeSet<[usize; 3]> = std::collections::BTreeSet::new();
        tris.retain(|t| {
            let mut key = *t;
            key.sort();
            seen.insert(key)
        });
        if tris.len() < before {
            changed = true;
        }

        // Step 2: Remove non-manifold excess triangles.
        //
        // Determinism: BTreeMap iterates in sorted key order, and we
        // break triangle-area ties on `(area, ti)` so the same input
        // mesh always picks the same triangles to remove. Without this
        // tie-break, two runs of the same input can produce different
        // cleanup results when multiple triangles have equal area
        // (common at U-periodic seams where BRepMesh emits mirrored
        // duplicate strips).
        let mut edge_tris: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (ti, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_tris.entry((lo, hi)).or_default().push(ti);
            }
        }

        let mut remove: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for tis in edge_tris.values() {
            if tis.len() <= 2 {
                continue;
            }
            // Keep 2 triangles with largest area, remove rest. Sort by
            // (-area, ti) so larger area wins; ties broken by lower ti.
            let mut scored: Vec<(usize, f64)> = tis
                .iter()
                .map(|&ti| {
                    let t = &tris[ti];
                    if t[0] >= vertices.len() || t[1] >= vertices.len() || t[2] >= vertices.len() {
                        return (ti, 0.0);
                    }
                    let a = vertices[t[0]];
                    let b = vertices[t[1]];
                    let c = vertices[t[2]];
                    let dx1 = b[0] - a[0];
                    let dy1 = b[1] - a[1];
                    let dz1 = b[2] - a[2];
                    let dx2 = c[0] - a[0];
                    let dy2 = c[1] - a[1];
                    let dz2 = c[2] - a[2];
                    let nx = dy1 * dz2 - dz1 * dy2;
                    let ny = dz1 * dx2 - dx1 * dz2;
                    let nz = dx1 * dy2 - dy1 * dx2;
                    (ti, nx * nx + ny * ny + nz * nz)
                })
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            for &(ti, _) in &scored[2..] {
                remove.insert(ti);
            }
        }

        if !remove.is_empty() {
            let remove_flags: Vec<bool> = (0..tris.len()).map(|i| remove.contains(&i)).collect();
            let mut idx = 0;
            tris.retain(|_| {
                let keep = !remove_flags[idx];
                idx += 1;
                keep
            });
            changed = true;
        }

        if !changed {
            break;
        }

        // Step 3: Fill holes
        tris = fill_boundary_holes(vertices, &tris);
    }

    // Step 4: Close remaining open edges
    tris = close_open_edges(vertices, &tris);

    tris
}

/// Repair T-junctions in a triangle mesh by splitting parent triangles.
///
/// A T-junction occurs when a vertex C sits ON the interior of an edge
/// (A, B) of some triangle (A, B, X). The mesh ends up with three
/// edges (A, B), (A, C), (C, B) where (A, C) and (C, B) belong to the
/// "through-vertex" triangles on the OTHER side, but (A, B, X) bridges
/// across as if C didn't exist. The result is a non-manifold edge:
/// (A, C) and (C, B) each appear in 2 triangles (one through-vertex,
/// one bridging - because (A, B, X) shares (A, C) by collinearity).
///
/// The repair: for every (A, B) with C collinear and between A and B,
/// replace (A, B, X) with (A, C, X) and (C, B, X). C becomes part of
/// the edge instead of a stranded vertex.
///
/// `eps_rel`: tolerance for "C is on segment AB", as a fraction of the
/// segment length. Default 1e-3 - generous enough to catch numerical
/// drift, tight enough to ignore genuinely-distinct nearby vertices.
///
/// Iterates up to 5 times to handle cascading splits.
///
/// Returns the repaired triangle list.
pub fn repair_t_junctions(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    eps_rel: f64,
) -> Vec<[usize; 3]> {
    repair_t_junctions_tracked(vertices, triangles, eps_rel).0
}

/// Like [`repair_t_junctions`] but also returns, per output triangle, the
/// index of the INPUT triangle it descends from. Callers that bucket
/// triangles (e.g. by face id) can then re-bucket exactly instead of
/// guessing by centroid - the heuristic mis-assigned rim sub-triangles of a
/// split to the adjacent (shared) face's bucket, which is blanked at
/// emission, silently LOSING them (issue #33).
pub fn repair_t_junctions_tracked(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    eps_rel: f64,
) -> (Vec<[usize; 3]>, Vec<usize>) {
    use std::collections::BTreeMap;
    let mut tris: Vec<[usize; 3]> = triangles.to_vec();
    let mut parent: Vec<usize> = (0..tris.len()).collect();

    for _pass in 0..5 {
        // Build edge → triangle map (deterministic via BTreeMap).
        let mut edge_tris: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_tris.entry((lo, hi)).or_default().push(ti);
            }
        }

        // Build vertex → set of incident edges (for collinearity lookup).
        let mut vert_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); vertices.len()];
        for &(a, b) in edge_tris.keys() {
            vert_edges[a].push((a, b));
            vert_edges[b].push((a, b));
        }

        // Find T-junctions: vertex C strictly between A and B for some edge (A, B).
        // Restrict candidates: C must share at least one incident edge with A or B
        // (otherwise it can't be a T-junction in this mesh).
        let mut splits: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (&(a, b), tis) in &edge_tris {
            // Only repair edges with abnormal sharing - manifold edges (count 2)
            // are fine; we look for non-manifold or open edges that might have
            // a collinear vertex creating the issue.
            if tis.len() == 2 {
                continue;
            }
            let va = vertices[a];
            let vb = vertices[b];
            let ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
            let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
            if ab_len_sq < 1e-30 {
                continue;
            }
            let ab_len = ab_len_sq.sqrt();
            let eps_abs = ab_len * eps_rel;
            // Candidate Cs: vertices that share an incident edge with A or B,
            // excluding A and B themselves.
            let mut candidates: std::collections::BTreeSet<usize> =
                std::collections::BTreeSet::new();
            for &(ea, eb) in &vert_edges[a] {
                if ea != a {
                    candidates.insert(ea);
                }
                if eb != a {
                    candidates.insert(eb);
                }
            }
            for &(ea, eb) in &vert_edges[b] {
                if ea != b {
                    candidates.insert(ea);
                }
                if eb != b {
                    candidates.insert(eb);
                }
            }
            candidates.remove(&a);
            candidates.remove(&b);

            for &c in &candidates {
                let vc = vertices[c];
                let ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
                let dot = ac[0] * ab[0] + ac[1] * ab[1] + ac[2] * ab[2];
                let t = dot / ab_len_sq;
                if !(t > 1e-6 && t < 1.0 - 1e-6) {
                    continue; // C must be STRICTLY between A and B
                }
                // Perpendicular distance from C to line AB.
                let proj = [va[0] + t * ab[0], va[1] + t * ab[1], va[2] + t * ab[2]];
                let perp = [vc[0] - proj[0], vc[1] - proj[1], vc[2] - proj[2]];
                let perp_dist = (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt();
                if perp_dist <= eps_abs {
                    splits.insert((a, b), c);
                    break;
                }
            }
        }

        if splits.is_empty() {
            break;
        }

        // Apply splits: for each triangle (A, B, X) where edge (A, B) is to be
        // split through C, replace with (A, C, X) and (C, B, X).
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 2);
        let mut new_parent: Vec<usize> = Vec::with_capacity(tris.len() * 2);
        for (ti, t) in tris.iter().enumerate() {
            // Find a split edge in this triangle.
            let mut split = None;
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if let Some(&c) = splits.get(&(lo, hi)) {
                    // Preserve winding: A, B in original order, C inserted.
                    split = Some((k, c));
                    break;
                }
            }
            if let Some((k, c)) = split {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let x = t[(k + 2) % 3];
                if c != a && c != b && c != x {
                    new_tris.push([a, c, x]);
                    new_tris.push([c, b, x]);
                    new_parent.push(parent[ti]);
                    new_parent.push(parent[ti]);
                } else {
                    new_tris.push(*t);
                    new_parent.push(parent[ti]);
                }
            } else {
                new_tris.push(*t);
                new_parent.push(parent[ti]);
            }
        }
        tris = new_tris;
        parent = new_parent;
    }

    (tris, parent)
}

/// Add a 2-ring pole fan between a pole vertex and a boundary row.
///
/// Creates 2 intermediate concentric rings at 1/3 and 2/3 of the distance
/// from pole to boundary, with fan + strip triangulation.
///
/// `boundary_row`: sorted (u_coordinate, vertex_index) pairs on the boundary
/// `u0`, `u1`: UV domain u-range
/// `v_pole`: v-coordinate of the pole
/// `v_boundary`: v-coordinate of the boundary row
/// `pole_at_top`: if true, the pole is at v_max (affects winding)
#[allow(clippy::too_many_arguments)]
pub fn add_multi_ring_pole_fan(
    verts_uv: &mut Vec<[f64; 2]>,
    tris: &mut Vec<[usize; 3]>,
    boundary_row: &[(f64, usize)], // sorted (u, vertex_index)
    u0: f64,
    u1: f64,
    v_pole: f64,
    v_boundary: f64,
    pole_at_top: bool,
) {
    let v_range = v_boundary - v_pole;

    if boundary_row.len() < 2 {
        // Simple fan fallback
        let pole_idx = verts_uv.len();
        verts_uv.push([(u0 + u1) / 2.0, v_pole]);
        for j in 0..boundary_row.len().saturating_sub(1) {
            let (_, idx_a) = boundary_row[j];
            let (_, idx_b) = boundary_row[j + 1];
            if pole_at_top {
                tris.push([pole_idx, idx_a, idx_b]);
            } else {
                tris.push([pole_idx, idx_b, idx_a]);
            }
        }
        return;
    }

    let u_coords: Vec<f64> = boundary_row.iter().map(|&(u, _)| u).collect();
    let n = u_coords.len();

    // Ring 1 at 1/3 distance from pole
    let v_ring1 = v_pole + 0.33 * v_range;
    let ring1_start = verts_uv.len();
    for &u in &u_coords {
        verts_uv.push([u, v_ring1]);
    }

    // Ring 2 at 2/3 distance from pole
    let v_ring2 = v_pole + 0.67 * v_range;
    let ring2_start = verts_uv.len();
    for &u in &u_coords {
        verts_uv.push([u, v_ring2]);
    }

    // Pole vertex
    let pole_idx = verts_uv.len();
    verts_uv.push([(u0 + u1) / 2.0, v_pole]);

    // Pole → ring1 (fan)
    for j in 0..n - 1 {
        let r1a = ring1_start + j;
        let r1b = ring1_start + j + 1;
        if pole_at_top {
            tris.push([pole_idx, r1a, r1b]);
        } else {
            tris.push([pole_idx, r1b, r1a]);
        }
    }

    // Ring1 → ring2 (strip)
    for j in 0..n - 1 {
        let r1a = ring1_start + j;
        let r1b = ring1_start + j + 1;
        let r2a = ring2_start + j;
        let r2b = ring2_start + j + 1;
        if pole_at_top {
            tris.push([r1a, r2a, r2b]);
            tris.push([r1a, r2b, r1b]);
        } else {
            tris.push([r1a, r2b, r2a]);
            tris.push([r1a, r1b, r2b]);
        }
    }

    // Ring2 → boundary (strip)
    for j in 0..n - 1 {
        let r2a = ring2_start + j;
        let r2b = ring2_start + j + 1;
        let ba = boundary_row[j].1;
        let bb = boundary_row[j + 1].1;
        if pole_at_top {
            tris.push([r2a, ba, bb]);
            tris.push([r2a, bb, r2b]);
        } else {
            tris.push([r2a, bb, ba]);
            tris.push([r2a, r2b, bb]);
        }
    }
}

/// Refine a triangle mesh so no edge exceeds `max_edge_length`.
///
/// Iteratively splits long edges by inserting midpoints. Each triangle
/// with a split edge becomes 2, 3, or 4 triangles depending on how many
/// of its edges are split.
///
/// Returns `(new_vertices, new_triangles)`.
pub fn refine_surface_to_edge_length(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    max_edge_length: f64,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut verts: Vec<[f64; 3]> = vertices.to_vec();
    let mut tris: Vec<[usize; 3]> = triangles.to_vec();
    let max_len_sq = max_edge_length * max_edge_length;

    for _ in 0..10 {
        // Find edges that exceed max length and compute midpoints
        let mut splits: HashMap<(usize, usize), (usize, [f64; 3])> = HashMap::new();
        {
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for t in &tris {
                for i in 0..3 {
                    let a = t[i];
                    let b = t[(i + 1) % 3];
                    let e = (a.min(b), a.max(b));
                    if !seen.insert(e) {
                        continue;
                    }
                    let va = verts[a];
                    let vb = verts[b];
                    let d2 =
                        (va[0] - vb[0]).powi(2) + (va[1] - vb[1]).powi(2) + (va[2] - vb[2]).powi(2);
                    if d2 > max_len_sq {
                        let mid = [
                            (va[0] + vb[0]) * 0.5,
                            (va[1] + vb[1]) * 0.5,
                            (va[2] + vb[2]) * 0.5,
                        ];
                        let mid_idx = verts.len() + splits.len();
                        splits.insert(e, (mid_idx, mid));
                    }
                }
            }
        }

        if splits.is_empty() {
            break;
        }

        // Add midpoint vertices
        let mut mids: Vec<(usize, [f64; 3])> = splits.values().copied().collect();
        mids.sort_by_key(|(idx, _)| *idx);
        for (_, mid) in &mids {
            verts.push(*mid);
        }

        // Split triangles
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 2);
        for t in tris.iter() {
            let mut split_edges: Vec<(usize, usize)> = Vec::new();
            for i in 0..3 {
                let a = t[i];
                let b = t[(i + 1) % 3];
                let e = (a.min(b), a.max(b));
                if let Some(&(mid_idx, _)) = splits.get(&e) {
                    split_edges.push((i, mid_idx));
                }
            }

            match split_edges.len() {
                0 => new_tris.push(*t),
                1 => {
                    let (ei, m) = split_edges[0];
                    let a = t[ei];
                    let b = t[(ei + 1) % 3];
                    let c = t[(ei + 2) % 3];
                    new_tris.push([a, m, c]);
                    new_tris.push([m, b, c]);
                }
                2 => {
                    let (ei0, m0) = split_edges[0];
                    let (ei1, m1) = split_edges[1];
                    let (a, b, c) = (t[0], t[1], t[2]);
                    match (ei0, ei1) {
                        (0, 1) => {
                            new_tris.push([a, m0, c]);
                            new_tris.push([m0, b, m1]);
                            new_tris.push([m0, m1, c]);
                        }
                        (0, 2) => {
                            new_tris.push([a, m0, m1]);
                            new_tris.push([m0, b, c]);
                            new_tris.push([m0, c, m1]);
                        }
                        (1, 2) => {
                            new_tris.push([a, b, m0]);
                            new_tris.push([a, m0, m1]);
                            new_tris.push([m0, c, m1]);
                        }
                        _ => new_tris.push(*t),
                    }
                }
                _ => {
                    // All 3 edges split
                    let mut m = [0usize; 3];
                    for &(ei, mi) in &split_edges {
                        m[ei] = mi;
                    }
                    let (a, b, c) = (t[0], t[1], t[2]);
                    new_tris.push([a, m[0], m[2]]);
                    new_tris.push([m[0], b, m[1]]);
                    new_tris.push([m[2], m[1], c]);
                    new_tris.push([m[0], m[1], m[2]]);
                }
            }
        }
        tris = new_tris;
    }

    (verts, tris)
}

/// Refine a surface mesh, optionally skipping flagged boundary edges and
/// propagating per-triangle face labels through splits.
///
/// Same bisection algorithm as [`refine_surface_to_edge_length`], with two
/// additions:
/// - `boundary_edges` is a set of `(min, max)` vertex-index tuples that
///   must NOT be split (e.g. edges shared with adjacent faces in a
///   per-face refinement context).
/// - `face_ids` is one label per input triangle; each sub-triangle from
///   a 1-to-2 / 1-to-3 / 1-to-4 split inherits the parent's label.
///
/// Returns `(new_vertices, new_triangles, new_face_ids)`.
pub fn refine_surface_to_edge_length_boundary_aware(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    face_ids: &[usize],
    max_edge_length: f64,
    boundary_edges: &HashSet<(usize, usize)>,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<usize>) {
    assert_eq!(
        triangles.len(),
        face_ids.len(),
        "face_ids length must match triangles length"
    );
    let mut verts: Vec<[f64; 3]> = vertices.to_vec();
    let mut tris: Vec<[usize; 3]> = triangles.to_vec();
    let mut fids: Vec<usize> = face_ids.to_vec();
    let max_len_sq = max_edge_length * max_edge_length;

    for _ in 0..10 {
        // Find edges that exceed max length and aren't in the boundary set.
        let mut splits: HashMap<(usize, usize), (usize, [f64; 3])> = HashMap::new();
        {
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for t in &tris {
                for i in 0..3 {
                    let a = t[i];
                    let b = t[(i + 1) % 3];
                    let e = (a.min(b), a.max(b));
                    if !seen.insert(e) {
                        continue;
                    }
                    if boundary_edges.contains(&e) {
                        continue;
                    }
                    let va = verts[a];
                    let vb = verts[b];
                    let d2 =
                        (va[0] - vb[0]).powi(2) + (va[1] - vb[1]).powi(2) + (va[2] - vb[2]).powi(2);
                    if d2 > max_len_sq {
                        let mid = [
                            (va[0] + vb[0]) * 0.5,
                            (va[1] + vb[1]) * 0.5,
                            (va[2] + vb[2]) * 0.5,
                        ];
                        let mid_idx = verts.len() + splits.len();
                        splits.insert(e, (mid_idx, mid));
                    }
                }
            }
        }

        if splits.is_empty() {
            break;
        }

        // Add midpoint vertices in index order.
        let mut mids: Vec<(usize, [f64; 3])> = splits.values().copied().collect();
        mids.sort_by_key(|(idx, _)| *idx);
        for (_, mid) in &mids {
            verts.push(*mid);
        }

        // Split triangles, threading face_id through every sub-triangle.
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 2);
        let mut new_fids: Vec<usize> = Vec::with_capacity(tris.len() * 2);
        for (ti, t) in tris.iter().enumerate() {
            let fid = fids[ti];
            let mut split_edges: Vec<(usize, usize)> = Vec::new();
            for i in 0..3 {
                let a = t[i];
                let b = t[(i + 1) % 3];
                let e = (a.min(b), a.max(b));
                if let Some(&(mid_idx, _)) = splits.get(&e) {
                    split_edges.push((i, mid_idx));
                }
            }

            match split_edges.len() {
                0 => {
                    new_tris.push(*t);
                    new_fids.push(fid);
                }
                1 => {
                    let (ei, m) = split_edges[0];
                    let a = t[ei];
                    let b = t[(ei + 1) % 3];
                    let c = t[(ei + 2) % 3];
                    new_tris.push([a, m, c]);
                    new_tris.push([m, b, c]);
                    new_fids.push(fid);
                    new_fids.push(fid);
                }
                2 => {
                    let (ei0, m0) = split_edges[0];
                    let (ei1, m1) = split_edges[1];
                    let (a, b, c) = (t[0], t[1], t[2]);
                    match (ei0, ei1) {
                        (0, 1) => {
                            new_tris.push([a, m0, c]);
                            new_tris.push([m0, b, m1]);
                            new_tris.push([m0, m1, c]);
                        }
                        (0, 2) => {
                            new_tris.push([a, m0, m1]);
                            new_tris.push([m0, b, c]);
                            new_tris.push([m0, c, m1]);
                        }
                        (1, 2) => {
                            new_tris.push([a, b, m0]);
                            new_tris.push([a, m0, m1]);
                            new_tris.push([m0, c, m1]);
                        }
                        _ => new_tris.push(*t),
                    }
                    new_fids.push(fid);
                    new_fids.push(fid);
                    new_fids.push(fid);
                }
                _ => {
                    // All 3 edges split.
                    let mut m = [0usize; 3];
                    for &(ei, mi) in &split_edges {
                        m[ei] = mi;
                    }
                    let (a, b, c) = (t[0], t[1], t[2]);
                    new_tris.push([a, m[0], m[2]]);
                    new_tris.push([m[0], b, m[1]]);
                    new_tris.push([m[2], m[1], c]);
                    new_tris.push([m[0], m[1], m[2]]);
                    for _ in 0..4 {
                        new_fids.push(fid);
                    }
                }
            }
        }
        tris = new_tris;
        fids = new_fids;
    }

    (verts, tris, fids)
}

/// Stitch non-manifold seams caused by independently-meshed face boundary overlap.
///
/// For each non-manifold edge (a,b) with 4 triangles (2 from each face), keeps
/// the 2 tris from one face (the pair that share a non-nm edge, i.e., were
/// neighbors inside the face) and removes the other pair. This preserves a
/// manifold surface without creating holes.
///
/// The face to keep is chosen by largest total area of the pair.
/// Fix non-manifold edges by inserting Steiner points.
///
/// For each non-manifold edge (a,b) with 4 tris from 2 overlapping faces:
/// - Face A has tris `[a,b,c_a1]` and `[a,c_a2,b]` with third vertices c_a1, c_a2
/// - Face B has tris `[a,b,c_b1]` and `[a,c_b2,b]` with third vertices c_b1, c_b2
/// - c_a1 ≈ c_b1 and c_a2 ≈ c_b2 (near-duplicates, ~3mm apart)
///
/// The fix: insert c_b1 into face A's triangles (and c_a1 into face B's)
/// by splitting the triangle that contains the point. After splitting, both
/// faces have identical boundary vertices, and the duplicate tris dedup away.
pub fn stitch_non_manifold_seams(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> Vec<[usize; 3]> {
    use std::collections::{HashMap, HashSet};

    let mut tris = triangles.to_vec();

    for _pass in 0..3 {
        // Build edge→tri map
        let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = t[k].min(t[(k + 1) % 3]);
                let b = t[k].max(t[(k + 1) % 3]);
                edge_tris.entry((a, b)).or_default().push(ti);
            }
        }

        let nm_edges: Vec<(usize, usize)> = edge_tris
            .iter()
            .filter(|(_, tis)| tis.len() > 2)
            .map(|(e, _)| *e)
            .collect();

        if nm_edges.is_empty() {
            break;
        }

        // Collect all Steiner point insertions needed:
        // For each nm edge, find near-duplicate third-vertex pairs and
        // determine which triangle each needs to be inserted into.
        // tri_splits: tri_index → list of Steiner vertex indices to insert
        let mut tri_splits: HashMap<usize, Vec<usize>> = HashMap::new();

        for &(a, b) in &nm_edges {
            let tis = match edge_tris.get(&(a, b)) {
                Some(v) => v,
                None => continue,
            };
            if tis.len() != 4 {
                continue;
            }

            // Get third vertices for each tri
            let mut thirds: Vec<(usize, usize)> = Vec::new(); // (tri_idx, third_vertex)
            for &ti in tis {
                let t = &tris[ti];
                for &v in t {
                    if v != a && v != b {
                        thirds.push((ti, v));
                        break;
                    }
                }
            }

            if thirds.len() != 4 {
                continue;
            }

            // Find the 2 closest third-vertex pairs (these are the near-duplicates)
            let mut pairs: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let (ti_a, c_a) = thirds[i];
                    let (ti_b, c_b) = thirds[j];
                    if c_a == c_b {
                        continue;
                    }
                    let pa = vertices[c_a];
                    let pb = vertices[c_b];
                    let d_sq =
                        (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
                    pairs.push((d_sq, ti_a, ti_b, c_a, c_b));
                }
            }
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // For the 2 closest pairs: insert each vertex into the other's triangle
            for &(d_sq, ti_a, ti_b, c_a, c_b) in pairs.iter().take(2) {
                // Only process if vertices are reasonably close
                let edge_len_sq = {
                    let pa = vertices[a];
                    let pb = vertices[b];
                    (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)
                };
                if d_sq > edge_len_sq {
                    continue;
                }

                // Insert c_b as Steiner point into tri_a's face.
                // Find which triangle of face A (near ti_a) contains point c_b.
                // The containing tri is likely ti_a itself or a neighbor.
                // For simplicity, search all tris that share vertex a or b.
                let candidate_tris: Vec<usize> = {
                    let mut cands = HashSet::new();
                    // Tris sharing vertex a
                    for (&e, tis_list) in &edge_tris {
                        if (e.0 == a || e.1 == a || e.0 == b || e.1 == b)
                            && tis_list.contains(&ti_a)
                        {
                            for &t in tis_list {
                                cands.insert(t);
                            }
                        }
                    }
                    // Also add direct neighbors of ti_a
                    for k in 0..3 {
                        let ea = tris[ti_a][k].min(tris[ti_a][(k + 1) % 3]);
                        let eb = tris[ti_a][k].max(tris[ti_a][(k + 1) % 3]);
                        if let Some(adj) = edge_tris.get(&(ea, eb)) {
                            for &t in adj {
                                cands.insert(t);
                            }
                        }
                    }
                    cands.into_iter().collect()
                };

                // Find which candidate tri contains point c_b
                let p = vertices[c_b];
                for &cti in &candidate_tris {
                    if cti == ti_b {
                        continue;
                    } // skip the tri that already has c_b
                    let t = &tris[cti];
                    if t.contains(&c_b) {
                        continue;
                    } // already has this vertex
                    if point_in_triangle_3d(p, vertices[t[0]], vertices[t[1]], vertices[t[2]]) {
                        tri_splits.entry(cti).or_default().push(c_b);
                        break;
                    }
                }

                // Same in reverse: insert c_a into tri_b's face
                let candidate_tris_b: Vec<usize> = {
                    let mut cands = HashSet::new();
                    for k in 0..3 {
                        let ea = tris[ti_b][k].min(tris[ti_b][(k + 1) % 3]);
                        let eb = tris[ti_b][k].max(tris[ti_b][(k + 1) % 3]);
                        if let Some(adj) = edge_tris.get(&(ea, eb)) {
                            for &t in adj {
                                cands.insert(t);
                            }
                        }
                    }
                    cands.into_iter().collect()
                };

                let p_a = vertices[c_a];
                for &cti in &candidate_tris_b {
                    if cti == ti_a {
                        continue;
                    }
                    let t = &tris[cti];
                    if t.contains(&c_a) {
                        continue;
                    }
                    if point_in_triangle_3d(p_a, vertices[t[0]], vertices[t[1]], vertices[t[2]]) {
                        tri_splits.entry(cti).or_default().push(c_a);
                        break;
                    }
                }
            }
        }

        if tri_splits.is_empty() {
            break;
        }

        // Apply splits: for each tri that needs Steiner points, replace it
        // with sub-triangles. A tri [p,q,r] split by point s becomes
        // [p,q,s], [q,r,s], [r,p,s] (3 sub-tris).
        let mut new_tris: Vec<[usize; 3]> = Vec::new();
        let mut removed: HashSet<usize> = HashSet::new();

        for (&ti, steiner_pts) in &tri_splits {
            let [p, q, r] = tris[ti];
            removed.insert(ti);
            // For each Steiner point, split the current triangle.
            // If multiple Steiner points, split sequentially.
            let mut current_tris = vec![[p, q, r]];
            for &s in steiner_pts {
                let mut next_tris = Vec::new();
                for &[a, b, c] in &current_tris {
                    if point_in_triangle_3d(vertices[s], vertices[a], vertices[b], vertices[c]) {
                        // Split [a,b,c] into 3 sub-tris around s
                        next_tris.push([a, b, s]);
                        next_tris.push([b, c, s]);
                        next_tris.push([c, a, s]);
                    } else {
                        next_tris.push([a, b, c]);
                    }
                }
                current_tris = next_tris;
            }
            new_tris.extend_from_slice(&current_tris);
        }

        // Rebuild tri list: keep unsplit tris + add split results
        let mut result: Vec<[usize; 3]> = Vec::with_capacity(tris.len() + new_tris.len());
        for (i, t) in tris.iter().enumerate() {
            if !removed.contains(&i) {
                result.push(*t);
            }
        }
        result.extend_from_slice(&new_tris);

        // Dedup
        let mut seen: HashSet<[usize; 3]> = HashSet::new();
        result.retain(|t| {
            let mut key = *t;
            key.sort();
            seen.insert(key)
        });

        // Remove degenerate tris
        result.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);

        tris = result;
    }

    tris
}

/// Test if point p is inside triangle (a,b,c) in 3D using barycentric coordinates.
fn point_in_triangle_3d(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> bool {
    // Compute vectors
    let v0 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v2 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];

    let dot00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    let dot01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    let dot02 = v0[0] * v2[0] + v0[1] * v2[1] + v0[2] * v2[2];
    let dot11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let dot12 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    if !inv_denom.is_finite() {
        return false;
    }
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Use a generous tolerance since the point is from an overlapping face
    // and may be slightly off the triangle plane
    let eps = -0.01;
    u >= eps && v >= eps && (u + v) <= 1.0 - eps
}

/// Run [`repair_t_junctions_tracked`] to its fixpoint (up to 12 outer
/// iterations of the 5-pass inner repair) with cumulative parent
/// tracking - the Python callers looped this across the PyO3 boundary,
/// re-marshalling the full vertex array every iteration.
pub fn repair_t_junctions_to_fixpoint(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    eps_rel: f64,
) -> (Vec<[usize; 3]>, Vec<usize>) {
    let mut tris: Vec<[usize; 3]> = triangles.to_vec();
    let mut parents: Vec<usize> = (0..tris.len()).collect();
    for _ in 0..12 {
        let (nt, par) = repair_t_junctions_tracked(vertices, &tris, eps_rel);
        if nt.len() == tris.len() {
            break;
        }
        parents = par.iter().map(|&pi| parents[pi]).collect();
        tris = nt;
    }
    (tris, parents)
}

/// Per-solid, per-bucket triangle lists: `[solid][bucket][triangle]`.
pub type SolidBuckets = Vec<Vec<Vec<[usize; 3]>>>;

/// Fused post-meshing finalization in ONE crossing:
/// vertex merge → per-bucket remap+dedup → per-solid T-junction repair
/// to fixpoint → exact parent re-bucketing.
///
/// Mirrors the previous Python orchestration exactly (global
/// `merge_vertices_with_remap`, `remap_tris(dedup=true)` per face
/// bucket, then per solid up to 12 outer iterations of
/// `repair_t_junctions_tracked` with `eps = max(1e-6, merge_tol)`,
/// re-bucketing each output triangle to its parent's face bucket).
/// That orchestration cost one PyO3 crossing per repair iteration per
/// solid (159 crossings on a 153-solid model) and re-marshalled the
/// full vertex array every time. Solids repair in parallel here.
///
/// Boundary census of a triangle mesh in one sweep.
///
/// Returns `(boundary_vertex_indices, n_boundary_edges, n_total_edges)`.
/// Replaces the per-face Python edge-count loops in `_optimize_3d_mesh`
/// (2.5M list appends / 2.7M `max()` calls per 153-solid run).
pub fn boundary_census(triangles: &[[usize; 3]]) -> (Vec<usize>, usize, usize) {
    let mut edge_count: HashMap<(usize, usize), u32> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let a = t[k];
            let b = t[(k + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let mut bset: HashSet<usize> = HashSet::new();
    let mut n_bdy_edges = 0usize;
    for (&(a, b), &c) in &edge_count {
        if c == 1 {
            n_bdy_edges += 1;
            bset.insert(a);
            bset.insert(b);
        }
    }
    let mut bverts: Vec<usize> = bset.into_iter().collect();
    bverts.sort_unstable();
    (bverts, n_bdy_edges, edge_count.len())
}

/// Bilinear interpolation of a `grid_n x grid_n` scalar field over a UV
/// rectangle, evaluated at many UV points in one crossing. Clamps to the
/// grid like the Python `_interp_target_h` it replaces.
#[allow(clippy::too_many_arguments)]
pub fn interp_grid_bilinear(
    uvs: &[[f64; 2]],
    h_grid: &[f64],
    grid_n: usize,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
) -> Vec<f64> {
    let gmax = (grid_n - 1) as f64;
    uvs.iter()
        .map(|&[u, v]| {
            let mut fu = (u - u0) / (u1 - u0) * gmax;
            let mut fv = (v - v0) / (v1 - v0) * gmax;
            fu = fu.clamp(0.0, gmax);
            fv = fv.clamp(0.0, gmax);
            let iu = fu as usize;
            let iv = fv as usize;
            let iu1 = (iu + 1).min(grid_n - 1);
            let iv1 = (iv + 1).min(grid_n - 1);
            let su = fu - iu as f64;
            let sv = fv - iv as f64;
            let v00 = h_grid[iv * grid_n + iu];
            let v10 = h_grid[iv * grid_n + iu1];
            let v01 = h_grid[iv1 * grid_n + iu];
            let v11 = h_grid[iv1 * grid_n + iu1];
            (1.0 - su) * (1.0 - sv) * v00
                + su * (1.0 - sv) * v10
                + (1.0 - su) * sv * v01
                + su * sv * v11
        })
        .collect()
}

/// Fill missing per-vertex UVs (`NaN` markers) by averaging the UVs of
/// vertices sharing a triangle - the same neighbour-average rule as the
/// Python loop in `_optimize_3d_mesh`. Returns the filled UV list and
/// the indices that were filled.
pub fn fill_missing_uvs(
    triangles: &[[usize; 3]],
    uvs: &[[f64; 2]],
    n_verts: usize,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        if i < uvs.len() {
            out.push(uvs[i]);
        } else {
            out.push([f64::NAN, f64::NAN]);
        }
    }
    // vertex -> triangles adjacency, only needed for missing verts
    let missing: Vec<usize> = (0..n_verts)
        .filter(|&i| out[i][0].is_nan() || out[i][1].is_nan())
        .collect();
    if missing.is_empty() {
        return (out, missing);
    }
    let miss_set: HashSet<usize> = missing.iter().copied().collect();
    let mut sums: HashMap<usize, (f64, f64, usize)> = HashMap::new();
    for t in triangles {
        for &v in t {
            if miss_set.contains(&v) {
                for &w in t {
                    if w < uvs.len() && !uvs[w][0].is_nan() && !uvs[w][1].is_nan() {
                        let e = sums.entry(v).or_insert((0.0, 0.0, 0));
                        e.0 += uvs[w][0];
                        e.1 += uvs[w][1];
                        e.2 += 1;
                    }
                }
            }
        }
    }
    let mut filled: Vec<usize> = Vec::new();
    for &v in &missing {
        if let Some(&(su, sv, n)) = sums.get(&v) {
            if n > 0 {
                out[v] = [su / n as f64, sv / n as f64];
                filled.push(v);
            }
        }
    }
    (out, filled)
}

/// Returns `(merged_vertices, per_solid_buckets)` where
/// `per_solid_buckets[s][k]` holds the triangles of solid `s`'s k-th
/// input bucket (same order as `solids[s]`).
pub fn finalize_surface_buckets(
    vertices: &[[f64; 3]],
    buckets: &[Vec<[usize; 3]>],
    solids: &[Vec<usize>],
    merge_tol: Option<f64>,
) -> (Vec<[f64; 3]>, SolidBuckets) {
    use rayon::prelude::*;

    let (merged_verts, remap) = merge_vertices_with_remap(vertices, merge_tol);

    // Remap + dedup every bucket once (shared buckets stay shared here;
    // each solid below works on its own gathered copy).
    let remapped: Vec<Vec<[usize; 3]>> = buckets
        .iter()
        .map(|b| remap_tris(b, &remap, true, false))
        .collect();

    let t_eps = merge_tol
        .unwrap_or(0.0)
        .max(if merge_tol.is_none() {
            // Match the auto tolerance the merge itself used.
            f64::max(bbox_diagonal(vertices) * 1e-8, 1e-10)
        } else {
            0.0
        })
        .max(1e-6);

    let per_solid: Vec<Vec<Vec<[usize; 3]>>> = solids
        .par_iter()
        .map(|bucket_ids| {
            // Gather this solid's triangles with their bucket of origin.
            let mut solid_tris: Vec<[usize; 3]> = Vec::new();
            let mut tri_bucket: Vec<usize> = Vec::new();
            for (k, &bi) in bucket_ids.iter().enumerate() {
                for t in &remapped[bi] {
                    solid_tris.push(*t);
                    tri_bucket.push(k);
                }
            }
            let n_in = solid_tris.len();
            if n_in < 4 {
                // Too small to repair - emit the remapped buckets as-is.
                return bucket_ids.iter().map(|&bi| remapped[bi].clone()).collect();
            }
            // Fixpoint: the inner repair caps at 5 passes; conforming a
            // coarse rim to a finely refined shared face can need ~15
            // cascading splits per edge (issue #33).
            let mut tris = solid_tris.clone();
            let mut parents: Vec<usize> = (0..tris.len()).collect();
            for _ in 0..12 {
                let (nt, par) = repair_t_junctions_tracked(&merged_verts, &tris, t_eps);
                if nt.len() == tris.len() {
                    break;
                }
                parents = par.iter().map(|&pi| parents[pi]).collect();
                tris = nt;
            }
            let mut out: Vec<Vec<[usize; 3]>> = vec![Vec::new(); bucket_ids.len()];
            if tris.len() == n_in {
                // No T-junctions - keep the gathered buckets.
                for (t, &k) in solid_tris.iter().zip(tri_bucket.iter()) {
                    out[k].push(*t);
                }
            } else {
                for (t, &pi) in tris.iter().zip(parents.iter()) {
                    out[tri_bucket[pi]].push(*t);
                }
            }
            out
        })
        .collect();

    (merged_verts, per_solid)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_surface_buckets_merges_and_repairs() {
        // Square split into 2 tris (bucket 0) + a neighbour fan (bucket 1)
        // whose vertex 4 sits exactly mid-edge of the square's bottom edge
        // (a T-junction), + vertex 5 a near-duplicate of vertex 1 that the
        // merge must weld. The solid needs >= 4 triangles or the repair is
        // skipped (mirrors the Python orchestration). After fusion:
        // 7 verts -> 6, and the square's bottom triangle splits at the
        // T-vertex (bucket 0 grows 2 -> 3).
        let verts = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [2.0, 2.0, 0.0],  // 2
            [0.0, 2.0, 0.0],  // 3
            [1.0, 0.0, 0.0],  // 4 - on edge (0,1)
            [2.0, 0.0, 1e-9], // 5 - duplicate of 1 within tol
            [1.0, -1.0, 0.0], // 6 - off-axis apex
        ];
        let buckets = vec![
            vec![[0usize, 1, 2], [2, 3, 0]],
            vec![[4usize, 5, 6], [0, 4, 6]],
        ];
        let solids = vec![vec![0usize, 1]];
        let (mv, per_solid) = finalize_surface_buckets(&verts, &buckets, &solids, Some(1e-6));
        assert_eq!(mv.len(), 6, "vertex 5 must weld into vertex 1");
        assert_eq!(per_solid.len(), 1);
        assert_eq!(per_solid[0].len(), 2, "bucket alignment preserved");
        // The square's bottom tri (0,1,2) must split at the T-vertex 4.
        assert_eq!(per_solid[0][0].len(), 3, "T-junction split adds one tri");
        assert_eq!(per_solid[0][1].len(), 2, "neighbour bucket unchanged");
    }

    #[test]
    fn remap_tris_basic() {
        let tris = vec![[0, 1, 2], [2, 3, 0]];
        let remap = vec![0, 1, 2, 3];
        let out = remap_tris(&tris, &remap, false, false);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn remap_tris_degenerate() {
        // Remap makes two vertices the same → degenerate.
        let tris = vec![[0, 1, 2]];
        let remap = vec![0, 0, 2]; // vertex 1 maps to 0
        let out = remap_tris(&tris, &remap, false, false);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn remap_tris_dedup() {
        let tris = vec![[0, 1, 2], [2, 0, 1]]; // same triangle, different winding
        let remap = vec![0, 1, 2];
        let out = remap_tris(&tris, &remap, true, false);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn remap_tris_flip() {
        let tris = vec![[0, 1, 2]];
        let remap = vec![0, 1, 2];
        let out = remap_tris(&tris, &remap, false, true);
        assert_eq!(out, vec![[0, 2, 1]]);
    }

    #[test]
    fn refine_boundary_aware_skips_flagged_edges() {
        // Closed tetrahedron: 4 verts, 4 tris.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
            [5.0, 5.0, 10.0],
        ];
        let tris = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let fids = vec![0, 0, 0, 0];

        // Flag every edge as boundary - refine should be a no-op.
        let mut all_edges: HashSet<(usize, usize)> = HashSet::new();
        for t in &tris {
            for i in 0..3 {
                let a = t[i];
                let b = t[(i + 1) % 3];
                all_edges.insert((a.min(b), a.max(b)));
            }
        }
        let (v2, t2, f2) =
            refine_surface_to_edge_length_boundary_aware(&verts, &tris, &fids, 1.0, &all_edges);
        assert_eq!(v2.len(), verts.len());
        assert_eq!(t2.len(), tris.len());
        assert_eq!(f2, fids);
    }

    #[test]
    fn refine_boundary_aware_splits_when_not_flagged() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
            [5.0, 5.0, 10.0],
        ];
        let tris = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let fids = vec![7, 9, 11, 13];
        let bdy: HashSet<(usize, usize)> = HashSet::new();

        let (v2, t2, f2) =
            refine_surface_to_edge_length_boundary_aware(&verts, &tris, &fids, 1.0, &bdy);
        assert!(v2.len() > verts.len(), "no vertices added");
        assert!(t2.len() > tris.len(), "no triangles added");
        assert_eq!(t2.len(), f2.len(), "face_ids must match triangles");

        // Every output face_id must be one of the input values (no fabrications).
        let valid: HashSet<usize> = fids.iter().copied().collect();
        for fid in &f2 {
            assert!(
                valid.contains(fid),
                "fabricated face_id {} not in input set",
                fid
            );
        }
    }

    #[test]
    fn repair_t_junctions_basic() {
        // Two triangles meeting at a T-junction:
        //   Triangle 1: (A, B, X) - bridges across with C on the AB line
        //   Triangle 2: (A, C, Y) and Triangle 3: (C, B, Y) - use C
        //
        //   B
        //   |\
        //   C-Y
        //   |/
        //   A    (X off to one side, Y off to the other)
        //
        // Triangle 1 should be split into (A, C, X) and (C, B, X).
        let verts = vec![
            [0.0, 0.0, 0.0],  // A = 0
            [10.0, 0.0, 0.0], // B = 1
            [5.0, 0.0, 0.0],  // C = 2 (midpoint of AB)
            [5.0, 5.0, 0.0],  // X = 3
            [5.0, -5.0, 0.0], // Y = 4
        ];
        let tris = vec![
            [0, 1, 3], // (A, B, X) - bridging triangle
            [0, 2, 4], // (A, C, Y)
            [2, 1, 4], // (C, B, Y)
        ];
        let out = repair_t_junctions(&verts, &tris, 1e-3);
        // Bridging triangle should be split into 2.
        assert_eq!(out.len(), 4, "expected 4 tris (3 original + 1 from split)");
        let sorted: std::collections::BTreeSet<[usize; 3]> = out
            .iter()
            .map(|t| {
                let mut k = *t;
                k.sort();
                k
            })
            .collect();
        // After repair: (A, C, X), (C, B, X), (A, C, Y), (C, B, Y) - 4 triangles
        // each with C in them.
        assert!(sorted.contains(&[0, 2, 3]), "missing (A, C, X) = {{0,2,3}}");
        assert!(sorted.contains(&[1, 2, 3]), "missing (C, B, X) = {{1,2,3}}");
    }

    #[test]
    fn repair_t_junctions_no_op_on_manifold() {
        // Two tetrahedron faces sharing an edge - no T-junction, no change.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [5.0, -5.0, 0.0],
        ];
        let tris = vec![[0, 1, 2], [0, 1, 3]];
        let out = repair_t_junctions(&verts, &tris, 1e-3);
        assert_eq!(out, tris, "should be no-op on manifold mesh");
    }

    #[test]
    fn repair_t_junctions_off_line_vertex_not_split() {
        // C is near AB but not on it - must NOT trigger a split.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 1.0, 0.0], // C at perpendicular distance 1.0 from AB
            [5.0, 5.0, 0.0],
            [5.0, -5.0, 0.0],
        ];
        let tris = vec![[0, 1, 3], [0, 2, 4], [2, 1, 4]];
        // eps_rel * len(AB) = 1e-3 * 10 = 0.01 - C at dist 1.0 is far outside.
        let out = repair_t_junctions(&verts, &tris, 1e-3);
        assert_eq!(out.len(), 3, "off-line vertex must not trigger split");
    }

    #[test]
    fn refine_boundary_aware_preserves_face_id_per_parent() {
        // Build a tetrahedron where each of the 4 faces has a unique fid.
        // After refinement every sub-triangle must keep ITS parent's fid -
        // verify by checking that no sub-tri has a fid from another face.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
            [5.0, 5.0, 10.0],
        ];
        let tris = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let fids = vec![100, 200, 300, 400];
        let bdy: HashSet<(usize, usize)> = HashSet::new();

        let (_v2, t2, f2) =
            refine_surface_to_edge_length_boundary_aware(&verts, &tris, &fids, 1.0, &bdy);
        // Count tris per face_id - distribution should be non-zero for all
        // input ids (each face starts with 1 tri, splits into many).
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &f in &f2 {
            *counts.entry(f).or_insert(0) += 1;
        }
        for &expected in &fids {
            assert!(
                counts.get(&expected).copied().unwrap_or(0) >= 1,
                "no tris with face_id {} after refine",
                expected
            );
        }
        assert_eq!(t2.len(), f2.len());
    }

    #[test]
    fn bbox_diagonal_empty() {
        assert_eq!(bbox_diagonal(&[]), 0.0);
    }

    #[test]
    fn bbox_diagonal_cube() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let d = bbox_diagonal(&verts);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn collect_wire_uv_basic() {
        let edges = vec![
            vec![[0.0, 0.0], [1.0, 0.0]],
            vec![[1.0, 0.0], [1.0, 1.0]],
            vec![[1.0, 1.0], [0.0, 1.0]],
            vec![[0.0, 1.0], [0.0, 0.0]],
        ];
        let rev = vec![false; 4];
        let wire = collect_wire_uv(&edges, &rev);
        // 4 edges with shared endpoints → 4 unique vertices (closing dedup)
        assert_eq!(wire.len(), 4);
    }

    #[test]
    fn collect_wire_uv_reversed() {
        let edges = vec![
            vec![[0.0, 0.0], [1.0, 0.0]],
            vec![[1.0, 1.0], [1.0, 0.0]], // stored reversed
        ];
        let rev = vec![false, true]; // second edge reversed
        let wire = collect_wire_uv(&edges, &rev);
        assert_eq!(wire.len(), 3); // [0,0], [1,0], [1,1]
        assert_eq!(wire[0], [0.0, 0.0]);
        assert_eq!(wire[1], [1.0, 0.0]);
        assert_eq!(wire[2], [1.0, 1.0]);
    }

    // --- merge_vertices_with_remap tests ---

    #[test]
    fn merge_vertices_empty() {
        let (verts, remap) = merge_vertices_with_remap(&[], None);
        assert!(verts.is_empty());
        assert!(remap.is_empty());
    }

    #[test]
    fn merge_vertices_no_duplicates() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (new_verts, remap) = merge_vertices_with_remap(&verts, Some(1e-6));
        assert_eq!(new_verts.len(), 3);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    #[test]
    fn merge_vertices_with_duplicates() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1e-12], // near-duplicate of vertex 0
            [1.0, 1e-12, 0.0], // near-duplicate of vertex 1
        ];
        let (new_verts, remap) = merge_vertices_with_remap(&verts, Some(1e-6));
        assert_eq!(new_verts.len(), 2);
        assert_eq!(remap[0], remap[2]); // verts 0 and 2 merged
        assert_eq!(remap[1], remap[3]); // verts 1 and 3 merged
        assert_ne!(remap[0], remap[1]); // different groups
    }

    #[test]
    fn merge_vertices_auto_tolerance() {
        // Two points far apart and two close together
        let verts = vec![
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 0.0, 1e-10], // very close to 0
        ];
        let (new_verts, remap) = merge_vertices_with_remap(&verts, None);
        // bbox diagonal ~ 100, tol ~ 1e-6, so 1e-10 should merge
        assert_eq!(new_verts.len(), 2);
        assert_eq!(remap[0], remap[2]);
    }

    // --- merge_vertices_between_faces tests ---

    #[test]
    fn merge_between_faces_same_face_no_merge() {
        // Two near-duplicate verts in the SAME face should NOT merge
        let verts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1e-12], [1.0, 0.0, 0.0]];
        let face_ranges = vec![(0, 2)]; // both verts 0,1 in face 0; vert 2 in no face
        let (new_verts, remap) = merge_vertices_between_faces(&verts, &face_ranges, Some(1e-6));
        // Verts 0 and 1 are in the same face, so should not merge
        assert_eq!(new_verts.len(), 3);
        assert_ne!(remap[0], remap[1]);
    }

    #[test]
    fn merge_between_faces_different_faces_merge() {
        // Two near-duplicate verts in DIFFERENT faces should merge
        let verts = vec![
            [0.0, 0.0, 0.0],   // face 0
            [1.0, 0.0, 0.0],   // face 0
            [0.0, 0.0, 1e-12], // face 1 -- near-duplicate of vert 0
            [0.0, 1.0, 0.0],   // face 1
        ];
        let face_ranges = vec![(0, 2), (2, 4)];
        let (new_verts, remap) = merge_vertices_between_faces(&verts, &face_ranges, Some(1e-6));
        assert_eq!(new_verts.len(), 3); // vert 0 and 2 merge
        assert_eq!(remap[0], remap[2]);
        assert_ne!(remap[0], remap[1]);
    }

    // --- merge_vertices (wrapper) tests ---

    #[test]
    fn merge_vertices_wrapper() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1e-12], // duplicate of vert 0
        ];
        let tris = vec![[0, 1, 2], [3, 1, 2]]; // second tri has duplicate vert
        let (new_verts, new_tris) = merge_vertices(&verts, &tris, Some(1e-6));
        assert_eq!(new_verts.len(), 3);
        // Both triangles remap to the same triangle after merging, one is degenerate-free
        // Actually [0,1,2] and [0,1,2] - the second becomes identical (not degenerate)
        // but remap_tris with dedup=false keeps both
        assert_eq!(new_tris.len(), 2);
        assert_eq!(new_tris[0], new_tris[1]);
    }

    // --- fix_normals_bfs tests ---

    /// Build a unit cube mesh (12 triangles, outward normals) for testing.
    fn unit_cube() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let v = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ];
        // Outward-facing triangles (CCW when viewed from outside)
        let t = vec![
            // bottom (z=0) - normal -z
            [0, 2, 1],
            [0, 3, 2],
            // top (z=1) - normal +z
            [4, 5, 6],
            [4, 6, 7],
            // front (y=0) - normal -y
            [0, 1, 5],
            [0, 5, 4],
            // back (y=1) - normal +y
            [2, 3, 7],
            [2, 7, 6],
            // left (x=0) - normal -x
            [0, 4, 7],
            [0, 7, 3],
            // right (x=1) - normal +x
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, t)
    }

    fn signed_volume(verts: &[[f64; 3]], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| signed_volume_tri(&verts[t[0]], &verts[t[1]], &verts[t[2]]))
            .sum()
    }

    #[test]
    fn surface_enclosed_volume_unit_cube() {
        let (v, t) = unit_cube();
        let vol = surface_enclosed_volume(&v, &t);
        assert!(
            (vol - 1.0).abs() < 1e-12,
            "unit cube enclosed volume should be 1.0, got {vol}"
        );
    }

    #[test]
    fn surface_enclosed_volume_orientation_independent() {
        let (v, t) = unit_cube();
        // Reverse every triangle's winding; the magnitude must be unchanged.
        let flipped: Vec<[usize; 3]> = t.iter().map(|tri| [tri[0], tri[2], tri[1]]).collect();
        let vol = surface_enclosed_volume(&v, &flipped);
        assert!(
            (vol - 1.0).abs() < 1e-12,
            "reversed winding should give the same magnitude, got {vol}"
        );
    }

    #[test]
    fn surface_enclosed_volume_mixed_winding() {
        // The bug this function used to have. Per-face surface meshes are each
        // oriented by their own surface normal, so a concatenated boundary is
        // not consistently wound. Under the old naive `|sum a.(b x c)|` the
        // contributions partially cancel: TruncatedCone read 145.4 against a
        // true 406.17. Flip half the cube's triangles and the volume must not
        // move.
        let (v, t) = unit_cube();
        let mixed: Vec<[usize; 3]> = t
            .iter()
            .enumerate()
            .map(|(i, tri)| {
                if i % 2 == 0 {
                    *tri
                } else {
                    [tri[0], tri[2], tri[1]]
                }
            })
            .collect();
        let vol = surface_enclosed_volume(&v, &mixed);
        assert!(
            (vol - 1.0).abs() < 1e-12,
            "inconsistently wound cube must still enclose 1.0, got {vol}"
        );
    }

    #[test]
    fn surface_enclosed_volume_hollow_subtracts_cavity() {
        // A hollow solid's surface has several components: an outer shell plus
        // sealed cavities. The material volume is outer MINUS cavity, not their
        // sum -- the naive version added them, so a coil casing came out at
        // 2.667e9 against a true 7.41e8.
        let (outer_v, outer_t) = unit_cube();
        // Inner cube of side 0.5 centred inside the unit cube.
        let inner_v: Vec<[f64; 3]> = outer_v
            .iter()
            .map(|p| [0.25 + p[0] * 0.5, 0.25 + p[1] * 0.5, 0.25 + p[2] * 0.5])
            .collect();
        let off = outer_v.len();
        let mut v = outer_v.clone();
        v.extend_from_slice(&inner_v);
        let mut tris = outer_t.clone();
        tris.extend(outer_t.iter().map(|f| [f[0] + off, f[1] + off, f[2] + off]));

        let vol = surface_enclosed_volume(&v, &tris);
        let expected = 1.0 - 0.5f64.powi(3);
        assert!(
            (vol - expected).abs() < 1e-12,
            "hollow cube should be {expected} (outer minus cavity), got {vol}"
        );
    }

    #[test]
    fn tet_mesh_volume_single_tet() {
        // Corner tetrahedron of the unit cube has volume 1/6.
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let t = vec![[0usize, 1, 2, 3]];
        let vol = tet_mesh_volume(&v, &t);
        assert!(
            (vol - 1.0 / 6.0).abs() < 1e-12,
            "unit corner tet volume should be 1/6, got {vol}"
        );
    }

    #[test]
    fn fix_normals_correct_cube() {
        let (verts, tris) = unit_cube();
        // Already correct normals - signed volume should be positive
        let vol = signed_volume(&verts, &tris);
        assert!(
            vol > 0.0,
            "Cube should have positive signed volume, got {vol}"
        );

        // Put into the nested HashMap structure
        let mut faces: HashMap<usize, Vec<[usize; 3]>> = HashMap::new();
        faces.insert(0, tris);
        let mut input: HashMap<usize, HashMap<usize, Vec<[usize; 3]>>> = HashMap::new();
        input.insert(1, faces);

        let output = fix_normals_bfs(&verts, &input);
        let out_tris: Vec<[usize; 3]> = output[&1]
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        let out_vol = signed_volume(&verts, &out_tris);
        assert!(
            out_vol > 0.0,
            "Fixed cube should still have positive volume, got {out_vol}"
        );
    }

    #[test]
    fn fix_normals_inverted_cube() {
        let (verts, tris) = unit_cube();
        // Flip all triangles to get inward normals
        let flipped: Vec<[usize; 3]> = tris.iter().map(|t| [t[0], t[2], t[1]]).collect();
        let vol = signed_volume(&verts, &flipped);
        assert!(vol < 0.0, "Flipped cube should have negative signed volume");

        let mut faces: HashMap<usize, Vec<[usize; 3]>> = HashMap::new();
        faces.insert(0, flipped);
        let mut input: HashMap<usize, HashMap<usize, Vec<[usize; 3]>>> = HashMap::new();
        input.insert(1, faces);

        let output = fix_normals_bfs(&verts, &input);
        let out_tris: Vec<[usize; 3]> = output[&1]
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        let out_vol = signed_volume(&verts, &out_tris);
        assert!(
            out_vol > 0.0,
            "Fixed inverted cube should have positive volume, got {out_vol}"
        );
    }

    #[test]
    fn fix_normals_preserves_face_structure() {
        let (verts, tris) = unit_cube();
        // Split into 6 faces (2 tris each)
        let mut faces: HashMap<usize, Vec<[usize; 3]>> = HashMap::new();
        for (i, chunk) in tris.chunks(2).enumerate() {
            faces.insert(i, chunk.to_vec());
        }
        let mut input: HashMap<usize, HashMap<usize, Vec<[usize; 3]>>> = HashMap::new();
        input.insert(1, faces);

        let output = fix_normals_bfs(&verts, &input);
        // Should still have 6 faces with 2 tris each
        assert_eq!(output[&1].len(), 6);
        for face_tris in output[&1].values() {
            assert_eq!(face_tris.len(), 2);
        }
    }

    #[test]
    fn ray_hits_triangle_basic() {
        // Triangle in the XY plane at z=1
        let a = [0.0, 0.0, 1.0];
        let b = [2.0, 0.0, 1.0];
        let c = [0.0, 2.0, 1.0];

        // Ray from origin in +z direction
        assert!(ray_hits_triangle(
            &[0.1, 0.1, 0.0],
            &[0.0, 0.0, 1.0],
            &a,
            &b,
            &c
        ));
        // Ray from origin in -z direction
        assert!(!ray_hits_triangle(
            &[0.1, 0.1, 0.0],
            &[0.0, 0.0, -1.0],
            &a,
            &b,
            &c
        ));
        // Ray missing the triangle
        assert!(!ray_hits_triangle(
            &[5.0, 5.0, 0.0],
            &[0.0, 0.0, 1.0],
            &a,
            &b,
            &c
        ));
    }

    // --- fill_boundary_holes tests ---

    #[test]
    fn fill_boundary_holes_no_holes() {
        // A closed tetrahedron has no boundary edges - all edges shared by 2 tris
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let tris = vec![
            [0, 2, 1], // bottom
            [0, 1, 3], // front
            [1, 2, 3], // right
            [0, 3, 2], // left
        ];
        let out = fill_boundary_holes(&verts, &tris);
        assert_eq!(out.len(), 4, "No holes: output should equal input");
        assert_eq!(out, tris);
    }

    #[test]
    fn fill_boundary_holes_single_triangle_hole() {
        // Tetrahedron with one face removed → one triangular hole
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let tris = vec![
            [0, 2, 1], // bottom
            [1, 2, 3], // right
            [0, 3, 2], // left
                       // Missing: [0, 1, 3] - the front face is the hole
        ];
        let out = fill_boundary_holes(&verts, &tris);
        assert_eq!(
            out.len(),
            4,
            "Should fill the triangular hole with 1 triangle"
        );
        // The new triangle should cover vertices 0, 1, 3
        let new_tri = out[3];
        let mut sorted_new = new_tri;
        sorted_new.sort();
        assert_eq!(sorted_new, [0, 1, 3]);
    }

    #[test]
    fn fill_boundary_holes_quad_hole() {
        // A mesh with a 4-vertex hole: two triangles forming a square base,
        // plus two side triangles, leaving a quad hole on the top.
        //
        //  Vertices:
        //  0 = (0,0,0), 1 = (1,0,0), 2 = (1,1,0), 3 = (0,1,0), 4 = (0.5,0.5,-1)
        //
        //  Bottom pyramid: 4 side faces, open top (quad 0-1-2-3)
        let verts = vec![
            [0.0, 0.0, 0.0],  // 0
            [1.0, 0.0, 0.0],  // 1
            [1.0, 1.0, 0.0],  // 2
            [0.0, 1.0, 0.0],  // 3
            [0.5, 0.5, -1.0], // 4 (apex below)
        ];
        let tris = vec![
            [0, 4, 1], // side
            [1, 4, 2], // side
            [2, 4, 3], // side
            [3, 4, 0], // side
                       // Missing top face (quad 0-1-2-3) → boundary loop of 4 vertices
        ];
        let out = fill_boundary_holes(&verts, &tris);
        // Should add 2 fan triangles to fill the quad hole
        assert_eq!(out.len(), 6, "Should fill quad hole with 2 fan triangles");
        // Verify the new triangles cover the quad vertices
        let new_tris = &out[4..];
        let mut all_verts_in_new: HashSet<usize> = HashSet::new();
        for t in new_tris {
            for &v in t {
                all_verts_in_new.insert(v);
            }
        }
        // All 4 top vertices should appear
        assert!(all_verts_in_new.contains(&0));
        assert!(all_verts_in_new.contains(&1));
        assert!(all_verts_in_new.contains(&2));
        assert!(all_verts_in_new.contains(&3));
        // Apex should NOT appear in hole-filling triangles
        assert!(!all_verts_in_new.contains(&4));
    }
}
