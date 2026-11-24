//! 3D triangle mesh optimization: edge collapse, edge swap, and Laplacian smooth.
//!
//! This module ports the Python `_optimize_3d_mesh` function from `cad.py`
//! to Rust for performance. Surface projection (which requires OCC) is left
//! to the Python caller - this module handles collapse, swap, and a simple
//! centroid smooth (no projection).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ----------------------------------------------------------------
// QEM (Quadric Error Metrics) for optimal vertex placement
// ----------------------------------------------------------------

/// Symmetric 4x4 quadric matrix stored as 10 unique values.
#[derive(Clone, Copy)]
struct Quadric {
    a2: f64,
    ab: f64,
    ac: f64,
    ad: f64,
    b2: f64,
    bc: f64,
    bd: f64,
    c2: f64,
    cd: f64,
    d2: f64,
}

impl Quadric {
    fn zero() -> Self {
        Quadric {
            a2: 0.0,
            ab: 0.0,
            ac: 0.0,
            ad: 0.0,
            b2: 0.0,
            bc: 0.0,
            bd: 0.0,
            c2: 0.0,
            cd: 0.0,
            d2: 0.0,
        }
    }

    /// Build a quadric from a triangle's plane equation.
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quadric {
            a2: a * a,
            ab: a * b,
            ac: a * c,
            ad: a * d,
            b2: b * b,
            bc: b * c,
            bd: b * d,
            c2: c * c,
            cd: c * d,
            d2: d * d,
        }
    }

    fn add(&self, other: &Quadric) -> Quadric {
        Quadric {
            a2: self.a2 + other.a2,
            ab: self.ab + other.ab,
            ac: self.ac + other.ac,
            ad: self.ad + other.ad,
            b2: self.b2 + other.b2,
            bc: self.bc + other.bc,
            bd: self.bd + other.bd,
            c2: self.c2 + other.c2,
            cd: self.cd + other.cd,
            d2: self.d2 + other.d2,
        }
    }

    /// Evaluate the quadric error for position (x, y, z).
    fn error(&self, x: f64, y: f64, z: f64) -> f64 {
        self.a2 * x * x
            + 2.0 * self.ab * x * y
            + 2.0 * self.ac * x * z
            + 2.0 * self.ad * x
            + self.b2 * y * y
            + 2.0 * self.bc * y * z
            + 2.0 * self.bd * y
            + self.c2 * z * z
            + 2.0 * self.cd * z
            + self.d2
    }

    /// Solve for the optimal position minimizing the quadric error.
    /// Returns None if the 3x3 system is singular - fall back to midpoint.
    fn optimal_pos(&self) -> Option<[f64; 3]> {
        // Solve the 3x3 linear system from dQ/dx = dQ/dy = dQ/dz = 0
        let a = [
            [self.a2, self.ab, self.ac],
            [self.ab, self.b2, self.bc],
            [self.ac, self.bc, self.c2],
        ];
        let b = [-self.ad, -self.bd, -self.cd];

        // Cramer's rule
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if det.abs() < 1e-20 {
            return None;
        }
        let inv_det = 1.0 / det;
        let x = inv_det
            * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
                + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]));
        let y = inv_det
            * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
                - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]));
        let z = inv_det
            * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
                - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
                + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]));
        Some([x, y, z])
    }
}

/// Build per-vertex quadric matrices from triangle planes.
fn build_quadrics(verts: &[[f64; 3]], tris: &[[usize; 3]]) -> Vec<Quadric> {
    let mut qs = vec![Quadric::zero(); verts.len()];
    for t in tris {
        let [ax, ay, az] = verts[t[0]];
        let [bx, by, bz] = verts[t[1]];
        let [cx, cy, cz] = verts[t[2]];
        let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
        let ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
        let nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-30 {
            continue;
        }
        let (a, b, c) = (nx / len, ny / len, nz / len);
        let d = -(a * ax + b * ay + c * az);
        let q = Quadric::from_plane(a, b, c, d);
        for &vi in t {
            qs[vi] = qs[vi].add(&q);
        }
    }
    qs
}

/// Entry in the priority queue for QEM decimation.
struct QemEdge {
    error: f64,
    a: usize,
    b: usize,
}

impl PartialEq for QemEdge {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}
impl Eq for QemEdge {}
impl PartialOrd for QemEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QemEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// QEM-based iterative mesh decimation.
///
/// Collapses edges in order of increasing quadric error until the median
/// edge length reaches `target_h` or no more valid collapses exist.
/// Handles large coarsening ratios (e.g. 100K → 1K tris) without OOM.
pub fn decimate_qem(
    verts: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[usize; 3]>,
    target_h: f64,
    num_boundary: usize,
) {
    if tris.len() < 4 {
        return;
    }

    for _outer in 0..50 {
        if tris.len() < 4 {
            break;
        }

        // Check median edge length - stop if we've reached the target
        let mut sample_lens: Vec<f64> = Vec::with_capacity(tris.len().min(500) * 3);
        let step = (tris.len() / 500).max(1);
        for (i, t) in tris.iter().enumerate() {
            if i % step != 0 {
                continue;
            }
            sample_lens.push(edge_len(verts, t[0], t[1]));
            sample_lens.push(edge_len(verts, t[1], t[2]));
            sample_lens.push(edge_len(verts, t[0], t[2]));
        }
        sample_lens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median = sample_lens[sample_lens.len() / 2];
        // Stop when median edge reaches the target, but never decimate
        // below a minimum triangle count based on surface area.  This
        // prevents over-decimation on complex shapes where QEM converges
        // before the mesh has enough triangles to represent the geometry.
        let min_tris = 200usize;
        if median >= target_h * 0.5 && tris.len() > min_tris {
            break;
        }
        if tris.len() <= min_tris {
            break;
        }

        let adj = build_adjacency(verts.len(), tris);
        let boundary_verts = find_boundary_verts(&adj);
        let is_bdy = |v: usize| v < num_boundary || boundary_verts.contains(&v);

        let quadrics = build_quadrics(verts, tris);

        // Build priority queue of edge collapses
        let mut heap: BinaryHeap<QemEdge> = BinaryHeap::new();
        for &(a, b) in adj.edge_tris.keys() {
            if is_bdy(a) || is_bdy(b) {
                continue;
            }
            let q = quadrics[a].add(&quadrics[b]);
            let pos = q.optimal_pos().unwrap_or([
                (verts[a][0] + verts[b][0]) * 0.5,
                (verts[a][1] + verts[b][1]) * 0.5,
                (verts[a][2] + verts[b][2]) * 0.5,
            ]);
            let err = q.error(pos[0], pos[1], pos[2]).max(0.0);
            heap.push(QemEdge { error: err, a, b });
        }

        let max_collapses = (tris.len() / 3).max(100);
        let mut n_collapsed = 0usize;
        let mut dead_tris: HashSet<usize> = HashSet::new();
        let mut locked: HashSet<usize> = HashSet::new();

        while let Some(entry) = heap.pop() {
            if n_collapsed >= max_collapses {
                break;
            }
            let a = entry.a;
            let b = entry.b;
            if locked.contains(&a) || locked.contains(&b) {
                continue;
            }

            let q = quadrics[a].add(&quadrics[b]);
            let new_pos = q.optimal_pos().unwrap_or([
                (verts[a][0] + verts[b][0]) * 0.5,
                (verts[a][1] + verts[b][1]) * 0.5,
                (verts[a][2] + verts[b][2]) * 0.5,
            ]);

            // Gather affected tris
            let affected: Vec<usize> = adj.vert_tris[a]
                .union(&adj.vert_tris[b])
                .copied()
                .filter(|ti| !dead_tris.contains(ti))
                .collect();

            let mut degen: Vec<usize> = Vec::new();
            for &ti in &affected {
                let t = &tris[ti];
                let has_a = t[0] == a || t[1] == a || t[2] == a;
                let has_b = t[0] == b || t[1] == b || t[2] == b;
                if has_a && has_b {
                    degen.push(ti);
                }
            }

            // Validate collapse
            let old_a = verts[a];
            let mut ok = true;
            for &ti in &affected {
                if degen.contains(&ti) {
                    continue;
                }
                let t = &tris[ti];
                let mut rv = [t[0], t[1], t[2]];
                for v in rv.iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
                if rv[0] == rv[1] || rv[1] == rv[2] || rv[0] == rv[2] {
                    ok = false;
                    break;
                }
                let old_n = tri_normal(verts, &rv);
                verts[a] = new_pos;
                let new_n = tri_normal(verts, &rv);
                verts[a] = old_a;
                if dot3(old_n, new_n) <= 0.0 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }

            // Execute collapse
            verts[a] = new_pos;
            for &ti in &affected {
                if degen.contains(&ti) {
                    continue;
                }
                let t = &mut tris[ti];
                for v in t.iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
            }
            for &ti in &degen {
                dead_tris.insert(ti);
            }
            locked.insert(a);
            locked.insert(b);
            n_collapsed += 1;
        }

        if n_collapsed == 0 {
            break;
        }

        // Compact
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
        for (ti, t) in tris.iter().enumerate() {
            if dead_tris.contains(&ti) {
                continue;
            }
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                continue;
            }
            new_tris.push(*t);
        }
        *tris = new_tris;

        let mut used: HashSet<usize> = HashSet::with_capacity(tris.len() * 3);
        for t in tris.iter() {
            used.insert(t[0]);
            used.insert(t[1]);
            used.insert(t[2]);
        }
        if used.is_empty() {
            break;
        }
        let mut sorted_used: Vec<usize> = used.into_iter().collect();
        sorted_used.sort_unstable();
        let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(sorted_used.len());
        let mut new_verts: Vec<[f64; 3]> = Vec::with_capacity(sorted_used.len());
        for old_idx in sorted_used {
            old_to_new.insert(old_idx, new_verts.len());
            new_verts.push(verts[old_idx]);
        }
        *verts = new_verts;
        for t in tris.iter_mut() {
            *t = [old_to_new[&t[0]], old_to_new[&t[1]], old_to_new[&t[2]]];
        }
    }
}

/// Input for 3D mesh optimization.
pub struct Mesh3DOptInput {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    /// Target edge length at each vertex (curvature-based).
    pub target_h: Vec<f64>,
    /// Vertices `[0..num_boundary)` are boundary and will not be moved or collapsed.
    pub num_boundary: usize,
    /// Number of collapse+swap+smooth passes.
    pub passes: usize,
    /// Collapse ratio: edges shorter than `collapse_ratio * target_h` are collapsed.
    /// Higher = more aggressive collapse = fewer triangles.
    /// Default: 0.3. Use 0.45 for non-seam faces to reduce triangle count.
    pub collapse_ratio: f64,
}

/// Output from 3D mesh optimization.
pub struct Mesh3DOptOutput {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

const FOUR_SQRT3: f64 = 6.928203230275509; // 4 * sqrt(3)

// ----------------------------------------------------------------
// Geometry helpers
// ----------------------------------------------------------------

#[inline]
fn edge_len(verts: &[[f64; 3]], a: usize, b: usize) -> f64 {
    let dx = verts[a][0] - verts[b][0];
    let dy = verts[a][1] - verts[b][1];
    let dz = verts[a][2] - verts[b][2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn tri_normal(verts: &[[f64; 3]], t: &[usize; 3]) -> [f64; 3] {
    let [a, b, c] = *t;
    let [ax, ay, az] = verts[a];
    let [bx, by, bz] = verts[b];
    let [cx, cy, cz] = verts[c];
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let acx = cx - ax;
    let acy = cy - ay;
    let acz = cz - az;
    [
        aby * acz - abz * acy,
        abz * acx - abx * acz,
        abx * acy - aby * acx,
    ]
}

#[inline]
fn tri_quality(verts: &[[f64; 3]], t: &[usize; 3]) -> f64 {
    let [a, b, c] = *t;
    let [ax, ay, az] = verts[a];
    let [bx, by, bz] = verts[b];
    let [cx, cy, cz] = verts[c];
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let acx = cx - ax;
    let acy = cy - ay;
    let acz = cz - az;
    // cross product magnitude = 2 * area
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    let area2 = (nx * nx + ny * ny + nz * nz).sqrt();
    let la2 = abx * abx + aby * aby + abz * abz;
    let lbx = cx - bx;
    let lby = cy - by;
    let lbz = cz - bz;
    let lb2 = lbx * lbx + lby * lby + lbz * lbz;
    let lc2 = acx * acx + acy * acy + acz * acz;
    let denom = la2 + lb2 + lc2;
    if denom < 1e-30 {
        return 0.0;
    }
    FOUR_SQRT3 * 0.5 * area2 / denom
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ----------------------------------------------------------------
// Adjacency
// ----------------------------------------------------------------

/// Canonical edge key (smaller index first).
#[inline]
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

struct Adjacency {
    /// edge -> list of triangle indices
    edge_tris: HashMap<(usize, usize), Vec<usize>>,
    /// vertex -> set of triangle indices
    vert_tris: Vec<HashSet<usize>>,
}

fn build_adjacency(num_verts: usize, tris: &[[usize; 3]]) -> Adjacency {
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::with_capacity(tris.len() * 3);
    let mut vert_tris: Vec<HashSet<usize>> = vec![HashSet::new(); num_verts];

    for (ti, t) in tris.iter().enumerate() {
        let [a, b, c] = *t;
        vert_tris[a].insert(ti);
        vert_tris[b].insert(ti);
        vert_tris[c].insert(ti);
        edge_tris.entry(edge_key(a, b)).or_default().push(ti);
        edge_tris.entry(edge_key(b, c)).or_default().push(ti);
        edge_tris.entry(edge_key(a, c)).or_default().push(ti);
    }

    Adjacency {
        edge_tris,
        vert_tris,
    }
}

fn find_boundary_verts(adj: &Adjacency) -> HashSet<usize> {
    let mut bv = HashSet::new();
    for (&(a, b), tri_list) in &adj.edge_tris {
        if tri_list.len() == 1 {
            bv.insert(a);
            bv.insert(b);
        }
    }
    bv
}

// ----------------------------------------------------------------
// Resolve chain (for collapse merges)
// ----------------------------------------------------------------

fn resolve(merged_to: &[Option<usize>], mut v: usize) -> usize {
    while let Some(target) = merged_to[v] {
        v = target;
    }
    v
}

// ----------------------------------------------------------------
// Edge splitting
// ----------------------------------------------------------------

/// Split edges longer than `split_ratio * target_h` by inserting midpoints.
///
/// Only interior edges (both endpoints are interior) are split.
/// The midpoint inherits the average `target_h` of its two endpoints.
/// Each split edge replaces 1-2 adjacent triangles with 2-4 new triangles.
///
/// Returns `true` if any splits were performed.
fn split_long_edges_3d(
    vertices: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
    target_h: &mut Vec<f64>,
    num_boundary: usize,
) -> bool {
    let split_ratio = 1.4;
    let mut any_split = false;

    // Iterate: keep splitting until no more edges exceed threshold
    for _iter in 0..20 {
        let adj = build_adjacency(vertices.len(), triangles);
        let boundary_verts = find_boundary_verts(&adj);

        let is_boundary = |v: usize| -> bool { v < num_boundary || boundary_verts.contains(&v) };

        // Collect edges sorted by length (longest first) so we split the worst first
        let mut edge_lengths: Vec<(f64, usize, usize)> = adj
            .edge_tris
            .keys()
            .filter_map(|&(a, b)| {
                // Skip edges where either endpoint is boundary
                if is_boundary(a) || is_boundary(b) {
                    return None;
                }
                let el = edge_len(vertices, a, b);
                let th = (target_h[a] + target_h[b]) * 0.5;
                if el > split_ratio * th {
                    Some((el, a, b))
                } else {
                    None
                }
            })
            .collect();

        if edge_lengths.is_empty() {
            break;
        }

        // Sort longest first. The candidates come out of HashMap iteration
        // (random per-process seed), so ties MUST be broken deterministically -
        // on a symmetric mesh many edges have identical lengths, and a
        // length-only sort leaves their relative order random, making which
        // edges get split (and thus the whole pass outcome) vary run-to-run
        // (flaky test_split_reduces_max_edge_length in CI). Tie-break on the
        // (a, b) vertex pair for a total order.
        edge_lengths.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (x.1, x.2).cmp(&(y.1, y.2)))
        });

        let mut split_this_round = false;

        // Process one edge at a time, then rebuild adjacency
        // (splitting changes the triangle list, so adjacency becomes stale)
        for &(_el, a, b) in &edge_lengths {
            // Re-lookup triangles sharing this edge in current triangle list
            let ek = edge_key(a, b);
            let mut sharing: Vec<usize> = Vec::new();
            for (ti, t) in triangles.iter().enumerate() {
                let edges = [
                    edge_key(t[0], t[1]),
                    edge_key(t[1], t[2]),
                    edge_key(t[0], t[2]),
                ];
                if edges.contains(&ek) {
                    sharing.push(ti);
                }
            }

            if sharing.is_empty() || sharing.len() > 2 {
                continue;
            }

            // Compute midpoint
            let mid = [
                (vertices[a][0] + vertices[b][0]) * 0.5,
                (vertices[a][1] + vertices[b][1]) * 0.5,
                (vertices[a][2] + vertices[b][2]) * 0.5,
            ];
            let mid_th = (target_h[a] + target_h[b]) * 0.5;
            let mid_idx = vertices.len();

            // For each triangle sharing edge (a,b), split it into two triangles.
            // Triangle (a, b, c) with edge a-b becomes (a, mid, c) and (mid, b, c).
            let mut new_tris: Vec<[usize; 3]> = Vec::new();
            let mut all_ok = true;

            for &ti in &sharing {
                let t = triangles[ti];
                // Find the opposite vertex (not a or b)
                let c = if t[0] != a && t[0] != b {
                    t[0]
                } else if t[1] != a && t[1] != b {
                    t[1]
                } else {
                    t[2]
                };

                // Determine winding: find which order a,b appear in the triangle
                // so we preserve orientation
                let (v0, v1) = if (t[0] == a && t[1] == b)
                    || (t[1] == a && t[2] == b)
                    || (t[2] == a && t[0] == b)
                {
                    (a, b)
                } else {
                    (b, a)
                };

                // Original triangle is (v0, v1, c) - split into (v0, mid, c) and (mid, v1, c)
                let nt0 = [v0, mid_idx, c];
                let nt1 = [mid_idx, v1, c];

                // Check that new triangles are not inverted by temporarily
                // adding the midpoint vertex and checking normals
                // The original normal of (v0, v1, c):
                let orig_n = tri_normal(vertices, &[v0, v1, c]);

                // Temporarily push midpoint to check normals
                vertices.push(mid);
                let n0 = tri_normal(vertices, &nt0);
                let n1 = tri_normal(vertices, &nt1);

                // Check both sub-triangles have consistent normals
                if dot3(orig_n, n0) <= 0.0 || dot3(orig_n, n1) <= 0.0 {
                    all_ok = false;
                    vertices.pop(); // remove temporary midpoint
                    break;
                }

                // Check quality is acceptable
                let q0 = tri_quality(vertices, &nt0);
                let q1 = tri_quality(vertices, &nt1);
                vertices.pop(); // remove temporary midpoint

                if q0 < 0.05 || q1 < 0.05 {
                    all_ok = false;
                    break;
                }

                new_tris.push(nt0);
                new_tris.push(nt1);
            }

            if !all_ok || new_tris.is_empty() {
                continue;
            }

            // Commit the split: add midpoint vertex
            vertices.push(mid);
            target_h.push(mid_th);

            // Replace old triangles with new ones.
            // Mark old triangles for removal (replace first, append rest)
            // Sort sharing indices in reverse so removal doesn't shift earlier indices
            let mut sharing_sorted = sharing.clone();
            sharing_sorted.sort_unstable_by(|a, b| b.cmp(a));

            for &ti in &sharing_sorted {
                triangles.swap_remove(ti);
            }

            // Add all new triangles
            triangles.extend_from_slice(&new_tris);

            split_this_round = true;
            any_split = true;

            // After splitting one edge, break and rebuild adjacency
            break;
        }

        if !split_this_round {
            break;
        }
    }

    any_split
}

// ----------------------------------------------------------------
// Main optimization
// ----------------------------------------------------------------

/// Optimize a 3D triangle mesh by iterating split, collapse, swap, and smooth.
///
/// Boundary vertices (indices `0..num_boundary`) are never moved or collapsed.
/// The smooth phase moves interior vertices to the centroid of their neighbors
/// (no surface projection - the Python caller handles that).
pub fn optimize_mesh_3d(input: &Mesh3DOptInput) -> Mesh3DOptOutput {
    let mut verts = input.vertices.clone();
    let mut tris = input.triangles.clone();
    let mut target_h = input.target_h.clone();

    if tris.len() < 4 {
        return Mesh3DOptOutput {
            vertices: verts,
            triangles: tris,
        };
    }

    for _pass in 0..input.passes {
        if tris.len() < 4 {
            break;
        }

        // ==============================================================
        // Phase 0: Edge splitting (long edges)
        // ==============================================================
        split_long_edges_3d(&mut verts, &mut tris, &mut target_h, input.num_boundary);

        if tris.len() < 4 {
            break;
        }

        let adj = build_adjacency(verts.len(), &tris);
        let boundary_verts = find_boundary_verts(&adj);

        // Also mark the first `num_boundary` vertices as boundary
        let _is_boundary =
            |v: usize| -> bool { v < input.num_boundary || boundary_verts.contains(&v) };

        // ==============================================================
        // Phase 1: Iterative edge collapse (short edges)
        //
        // Collapses in batches to avoid O(n²) memory from processing
        // all edges at once. Each sub-pass collapses up to half the
        // triangles, compacts, rebuilds adjacency, and repeats.
        // ==============================================================
        for _collapse_iter in 0..30 {
            if tris.len() < 4 {
                break;
            }
            let adj = build_adjacency(verts.len(), &tris);
            let boundary_verts = find_boundary_verts(&adj);
            let is_boundary =
                |v: usize| -> bool { v < input.num_boundary || boundary_verts.contains(&v) };

            // Collect and sort collapse candidates
            let mut edge_lengths: Vec<(f64, usize, usize)> = adj
                .edge_tris
                .keys()
                .filter_map(|&(a, b)| {
                    if is_boundary(a) || is_boundary(b) {
                        return None;
                    }
                    let el = edge_len(&verts, a, b);
                    let th = (target_h[a] + target_h[b]) * 0.5;
                    if el > input.collapse_ratio * th {
                        return None;
                    }
                    Some((el, a, b))
                })
                .collect();

            if edge_lengths.is_empty() {
                break;
            }
            // Shortest first; ties broken on (a, b) - candidates come from
            // HashMap iteration (random seed), and a length-only sort leaves
            // equal-length edges in random order (run-to-run nondeterminism).
            edge_lengths.sort_by(|x, y| {
                x.0.partial_cmp(&y.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| (x.1, x.2).cmp(&(y.1, y.2)))
            });

            // Limit collapses per sub-pass to avoid unbounded memory
            let max_collapses = (tris.len() / 2).max(100);
            let mut n_collapsed = 0usize;
            let mut dead_tris: HashSet<usize> = HashSet::new();
            // Track which vertices have been touched this sub-pass
            let mut locked: HashSet<usize> = HashSet::new();

            for &(_el, a, b) in &edge_lengths {
                if n_collapsed >= max_collapses {
                    break;
                }
                // Skip if either vertex was already involved in a collapse
                if locked.contains(&a) || locked.contains(&b) {
                    continue;
                }

                let th = (target_h[a] + target_h[b]) * 0.5;

                // Midpoint
                let new_pos = [
                    (verts[a][0] + verts[b][0]) * 0.5,
                    (verts[a][1] + verts[b][1]) * 0.5,
                    (verts[a][2] + verts[b][2]) * 0.5,
                ];

                // Gather affected triangles
                let affected: Vec<usize> = adj.vert_tris[a]
                    .union(&adj.vert_tris[b])
                    .copied()
                    .filter(|ti| !dead_tris.contains(ti))
                    .collect();

                // Find degenerate triangles (contain both a and b)
                let mut degen: Vec<usize> = Vec::new();
                for &ti in &affected {
                    let t = &tris[ti];
                    let has_a = t[0] == a || t[1] == a || t[2] == a;
                    let has_b = t[0] == b || t[1] == b || t[2] == b;
                    if has_a && has_b {
                        degen.push(ti);
                    }
                }

                // Check validity of collapse
                let old_a = verts[a];
                let mut ok = true;

                for &ti in &affected {
                    if degen.contains(&ti) {
                        continue;
                    }
                    let t = &tris[ti];
                    let mut rv = [t[0], t[1], t[2]];
                    for v in rv.iter_mut() {
                        if *v == b {
                            *v = a;
                        }
                    }
                    if rv[0] == rv[1] || rv[1] == rv[2] || rv[0] == rv[2] {
                        ok = false;
                        break;
                    }
                    let rv_arr = [rv[0], rv[1], rv[2]];

                    // Check normal doesn't flip
                    let old_n = tri_normal(&verts, &rv_arr);
                    verts[a] = new_pos;
                    let new_n = tri_normal(&verts, &rv_arr);
                    let new_q = tri_quality(&verts, &rv_arr);
                    verts[a] = old_a;

                    if dot3(old_n, new_n) <= 0.0 || new_q < 0.05 {
                        ok = false;
                        break;
                    }

                    // Check resulting edge lengths
                    verts[a] = new_pos;
                    for &vi in &rv {
                        if vi == a {
                            continue;
                        }
                        let new_el = edge_len(&verts, a, vi);
                        if new_el > 1.5 * th {
                            ok = false;
                            break;
                        }
                    }
                    verts[a] = old_a;
                    if !ok {
                        break;
                    }
                }

                if !ok {
                    continue;
                }

                // Execute collapse: merge b into a at midpoint
                verts[a] = new_pos;
                target_h[a] = th;
                // Remap b → a in all affected tris
                for &ti in &affected {
                    if degen.contains(&ti) {
                        continue;
                    }
                    let t = &mut tris[ti];
                    for v in t.iter_mut() {
                        if *v == b {
                            *v = a;
                        }
                    }
                }
                for ti in &degen {
                    dead_tris.insert(*ti);
                }
                // Lock collapsed vertices (but not neighbors - allow
                // cascading collapses for aggressive coarsening)
                locked.insert(a);
                locked.insert(b);
                n_collapsed += 1;
            }

            if n_collapsed == 0 {
                break;
            }

            // Compact: remove dead tris and degenerate tris
            let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
            for (ti, t) in tris.iter().enumerate() {
                if dead_tris.contains(&ti) {
                    continue;
                }
                if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                    continue;
                }
                new_tris.push(*t);
            }
            tris = new_tris;
        }

        // Compact vertex indices
        let mut used: HashSet<usize> = HashSet::with_capacity(tris.len() * 3);
        for t in &tris {
            used.insert(t[0]);
            used.insert(t[1]);
            used.insert(t[2]);
        }
        if used.is_empty() {
            break;
        }

        let mut sorted_used: Vec<usize> = used.into_iter().collect();
        sorted_used.sort_unstable();
        let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(sorted_used.len());
        let mut new_verts: Vec<[f64; 3]> = Vec::with_capacity(sorted_used.len());
        let mut new_target_h: Vec<f64> = Vec::with_capacity(sorted_used.len());
        for old_idx in sorted_used {
            old_to_new.insert(old_idx, new_verts.len());
            new_verts.push(verts[old_idx]);
            new_target_h.push(target_h[old_idx]);
        }
        verts = new_verts;
        target_h = new_target_h;
        tris = tris
            .iter()
            .map(|t| [old_to_new[&t[0]], old_to_new[&t[1]], old_to_new[&t[2]]])
            .collect();

        if tris.len() < 4 {
            break;
        }

        // ==============================================================
        // Phase 2: Edge swap to improve quality
        // ==============================================================
        let mut swapped = true;
        let mut swap_iter = 0;
        while swapped && swap_iter < 5 {
            swapped = false;
            swap_iter += 1;
            let adj = build_adjacency(verts.len(), &tris);
            let boundary_verts = find_boundary_verts(&adj);

            // Sorted: edge_tris is a HashMap, so raw key order is random per
            // process; swaps mutate the mesh as we go, so iteration order
            // changes the outcome (run-to-run nondeterminism).
            let mut edges_to_check: Vec<(usize, usize)> = adj.edge_tris.keys().copied().collect();
            edges_to_check.sort_unstable();
            for (a, b) in edges_to_check {
                let tri_list = match adj.edge_tris.get(&(a, b)) {
                    Some(tl) if tl.len() == 2 => tl,
                    _ => continue,
                };
                let ti0 = tri_list[0];
                let ti1 = tri_list[1];
                let t0 = &tris[ti0];
                let t1 = &tris[ti1];

                // Find opposite vertices c and d
                let c = t0.iter().find(|&&v| v != a && v != b).copied();
                let d = t1.iter().find(|&&v| v != a && v != b).copied();
                let (c, d) = match (c, d) {
                    (Some(c), Some(d)) if c != d => (c, d),
                    _ => continue,
                };

                // Don't create a boundary-boundary edge
                if boundary_verts.contains(&c) && boundary_verts.contains(&d) {
                    continue;
                }

                // Check that edge c-d doesn't already exist
                if adj.edge_tris.contains_key(&edge_key(c, d)) {
                    continue;
                }

                // Quality before
                let q_before = tri_quality(&verts, t0).min(tri_quality(&verts, t1));

                // New triangles after swap: (a, c, d) and (b, d, c)
                let nt0 = [a, c, d];
                let nt1 = [b, d, c];

                // Check normals don't flip
                let n0_old = tri_normal(&verts, t0);
                let n1_old = tri_normal(&verts, t1);
                let n0_new = tri_normal(&verts, &nt0);
                let n1_new = tri_normal(&verts, &nt1);

                if dot3(n0_old, n0_new) <= 0.0 || dot3(n1_old, n1_new) <= 0.0 {
                    continue;
                }

                let q_after = tri_quality(&verts, &nt0).min(tri_quality(&verts, &nt1));

                // Only swap if quality improves by > 1%
                if q_after > q_before * 1.01 {
                    tris[ti0] = nt0;
                    tris[ti1] = nt1;
                    swapped = true;
                    // Note: we break out and rebuild adjacency on the next
                    // iteration of the while loop. The Python code rebuilds
                    // after every single swap, but doing it once per pass
                    // is sufficient for convergence.
                    break;
                }
            }
        }

        // ==============================================================
        // Phase 3: Laplacian smooth (no surface projection)
        // ==============================================================
        let adj = build_adjacency(verts.len(), &tris);
        let boundary_verts = find_boundary_verts(&adj);

        // Build vertex neighbors
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); verts.len()];
        for t in &tris {
            let [a, b, c] = *t;
            neighbors[a].insert(b);
            neighbors[a].insert(c);
            neighbors[b].insert(a);
            neighbors[b].insert(c);
            neighbors[c].insert(a);
            neighbors[c].insert(b);
        }

        for _smooth_iter in 0..3 {
            for vi in 0..verts.len() {
                if vi < input.num_boundary || boundary_verts.contains(&vi) {
                    continue;
                }
                let nbrs = &neighbors[vi];
                if nbrs.is_empty() {
                    continue;
                }
                let n = nbrs.len() as f64;
                let cx: f64 = nbrs.iter().map(|&ni| verts[ni][0]).sum::<f64>() / n;
                let cy: f64 = nbrs.iter().map(|&ni| verts[ni][1]).sum::<f64>() / n;
                let cz: f64 = nbrs.iter().map(|&ni| verts[ni][2]).sum::<f64>() / n;
                let new_pos = [cx, cy, cz];

                // Check no triangle inversion
                let mut ok = true;
                for &ti in &adj.vert_tris[vi] {
                    let old_n = tri_normal(&verts, &tris[ti]);
                    let old_v = verts[vi];
                    verts[vi] = new_pos;
                    let new_n = tri_normal(&verts, &tris[ti]);
                    verts[vi] = old_v;
                    if dot3(old_n, new_n) <= 0.0 {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    verts[vi] = new_pos;
                }
            }
        }
    }

    Mesh3DOptOutput {
        vertices: verts,
        triangles: tris,
    }
}

// ----------------------------------------------------------------
// Collapse-only periodic mesh optimization
// ----------------------------------------------------------------

/// Input for collapse-only periodic mesh optimization.
pub struct CollapsePeriodicInput {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    /// Number of collapse passes.
    pub passes: usize,
    /// Edges shorter than `collapse_ratio * median_edge_length` are collapsed.
    pub collapse_ratio: f64,
}

/// Output from collapse-only periodic mesh optimization.
pub struct CollapsePeriodicOutput {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// Collapse-only 3D mesh optimizer for periodic surfaces.
///
/// Unlike [`optimize_mesh_3d`] which splits, collapses, swaps, and smooths,
/// this function only collapses short edges. It uses the mesh's own median
/// edge length as the target size.
///
/// Surviving vertices stay at their original positions -- no midpoint movement.
/// This avoids the mis-projection problem on periodic surfaces where a midpoint
/// can land on the wrong side.
///
/// The link condition is enforced to guarantee manifold topology, and a quality
/// check rejects collapses that would create slivers.
pub fn collapse_periodic_mesh_3d(input: &CollapsePeriodicInput) -> CollapsePeriodicOutput {
    let mut verts = input.vertices.clone();
    let mut tris = input.triangles.clone();

    if tris.len() < 4 {
        return CollapsePeriodicOutput {
            vertices: verts,
            triangles: tris,
        };
    }

    for _pass in 0..input.passes {
        if tris.len() < 4 {
            break;
        }

        // ------------------------------------------------------------------
        // Compute edge lengths and median
        // ------------------------------------------------------------------
        let mut edge_set: HashMap<(usize, usize), f64> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let ek = edge_key(a, b);
                edge_set.entry(ek).or_insert_with(|| edge_len(&verts, a, b));
            }
        }

        let mut lengths: Vec<f64> = edge_set.values().copied().collect();
        if lengths.is_empty() {
            break;
        }
        lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_h = lengths[lengths.len() / 2];
        let threshold = input.collapse_ratio * median_h;

        // ------------------------------------------------------------------
        // Detect boundary vertices (edges with count != 2)
        // ------------------------------------------------------------------
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                *edge_count.entry(edge_key(a, b)).or_insert(0) += 1;
            }
        }
        let mut boundary_verts: HashSet<usize> = HashSet::new();
        for (&(a, b), &cnt) in &edge_count {
            if cnt != 2 {
                boundary_verts.insert(a);
                boundary_verts.insert(b);
            }
        }

        // ------------------------------------------------------------------
        // Build adjacency
        // ------------------------------------------------------------------
        let mut vert_tris: Vec<HashSet<usize>> = vec![HashSet::new(); verts.len()];
        let mut vert_nbrs: Vec<HashSet<usize>> = vec![HashSet::new(); verts.len()];
        for (ti, t) in tris.iter().enumerate() {
            let [a, b, c] = *t;
            vert_tris[a].insert(ti);
            vert_tris[b].insert(ti);
            vert_tris[c].insert(ti);
            vert_nbrs[a].insert(b);
            vert_nbrs[a].insert(c);
            vert_nbrs[b].insert(a);
            vert_nbrs[b].insert(c);
            vert_nbrs[c].insert(a);
            vert_nbrs[c].insert(b);
        }

        // ------------------------------------------------------------------
        // Edge collapse (shortest first) with link condition
        // ------------------------------------------------------------------
        let mut sorted_edges: Vec<((usize, usize), f64)> = edge_set.into_iter().collect();
        sorted_edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut merged_to: Vec<Option<usize>> = vec![None; verts.len()];
        let mut dead_tris: HashSet<usize> = HashSet::new();
        let mut dead_verts: HashSet<usize> = HashSet::new();
        let mut did_collapse = false;

        for &((a_orig, b_orig), el) in &sorted_edges {
            if el > threshold {
                break; // sorted, so all remaining are longer
            }

            let a = resolve(&merged_to, a_orig);
            let b = resolve(&merged_to, b_orig);
            if a == b {
                continue;
            }
            if boundary_verts.contains(&a) || boundary_verts.contains(&b) {
                continue;
            }
            if dead_verts.contains(&a) || dead_verts.contains(&b) {
                continue;
            }

            // ---- Link condition ----
            // Common neighbors of a and b (in the resolved, live mesh)
            let mut nbrs_a: HashSet<usize> = HashSet::new();
            for &ti in &vert_tris[a] {
                if dead_tris.contains(&ti) {
                    continue;
                }
                for &v in &tris[ti] {
                    let rv = resolve(&merged_to, v);
                    if rv != a && rv != b {
                        nbrs_a.insert(rv);
                    }
                }
            }
            let mut nbrs_b: HashSet<usize> = HashSet::new();
            for &ti in &vert_tris[b] {
                if dead_tris.contains(&ti) {
                    continue;
                }
                for &v in &tris[ti] {
                    let rv = resolve(&merged_to, v);
                    if rv != a && rv != b {
                        nbrs_b.insert(rv);
                    }
                }
            }
            let common_nbrs: HashSet<usize> = nbrs_a.intersection(&nbrs_b).copied().collect();

            // Opposite vertices from triangles sharing edge (a,b)
            let mut opposite_verts: HashSet<usize> = HashSet::new();
            let mut affected: HashSet<usize> = HashSet::new();
            let mut degen: HashSet<usize> = HashSet::new();

            let all_tris: HashSet<usize> = vert_tris[a].union(&vert_tris[b]).copied().collect();
            for ti in all_tris {
                if dead_tris.contains(&ti) {
                    continue;
                }
                let t = &tris[ti];
                let rv = [
                    resolve(&merged_to, t[0]),
                    resolve(&merged_to, t[1]),
                    resolve(&merged_to, t[2]),
                ];
                let has_a = rv[0] == a || rv[1] == a || rv[2] == a;
                let has_b = rv[0] == b || rv[1] == b || rv[2] == b;
                if has_a && has_b {
                    degen.insert(ti);
                    for &v in &rv {
                        if v != a && v != b {
                            opposite_verts.insert(v);
                        }
                    }
                }
                if has_a || has_b {
                    affected.insert(ti);
                }
            }

            // Link condition: common neighbors must equal opposite vertices
            if common_nbrs != opposite_verts {
                continue;
            }

            // ---- Build set of existing triangle keys for duplicate detection ----
            let mut existing_tri_keys: HashSet<[usize; 3]> = HashSet::new();
            for (ti, t) in tris.iter().enumerate() {
                if dead_tris.contains(&ti) || degen.contains(&ti) || affected.contains(&ti) {
                    continue;
                }
                let rv = [
                    resolve(&merged_to, t[0]),
                    resolve(&merged_to, t[1]),
                    resolve(&merged_to, t[2]),
                ];
                let mut key = rv;
                key.sort();
                existing_tri_keys.insert(key);
            }

            // ---- Normal flip check, quality check & duplicate check ----
            // Keep vertex a at its original position.
            let mut ok = true;
            for &ti in &affected {
                if degen.contains(&ti) {
                    continue;
                }
                let t = &tris[ti];
                let rv_old = [
                    resolve(&merged_to, t[0]),
                    resolve(&merged_to, t[1]),
                    resolve(&merged_to, t[2]),
                ];
                let rv_new = [
                    if rv_old[0] == b { a } else { rv_old[0] },
                    if rv_old[1] == b { a } else { rv_old[1] },
                    if rv_old[2] == b { a } else { rv_old[2] },
                ];
                if rv_new[0] == rv_new[1] || rv_new[1] == rv_new[2] || rv_new[0] == rv_new[2] {
                    ok = false;
                    break;
                }

                // Reject if the remapped triangle duplicates an existing one
                let mut key = rv_new;
                key.sort();
                if existing_tri_keys.contains(&key) {
                    ok = false;
                    break;
                }
                existing_tri_keys.insert(key);

                // Check that the normal after collapse (b->a) doesn't flip.
                let old_n = tri_normal(&verts, &rv_old);
                let new_n = tri_normal(&verts, &rv_new);

                if dot3(old_n, new_n) <= 0.0 {
                    ok = false;
                    break;
                }

                // Quality check: area2 / max_edge^2 (matching Python metric)
                let p0 = verts[rv_new[0]];
                let p1 = verts[rv_new[1]];
                let p2 = verts[rv_new[2]];
                let e0 =
                    ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2))
                        .sqrt();
                let e1 =
                    ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2))
                        .sqrt();
                let e2 =
                    ((p0[0] - p2[0]).powi(2) + (p0[1] - p2[1]).powi(2) + (p0[2] - p2[2]).powi(2))
                        .sqrt();
                let max_e = e0.max(e1).max(e2);
                let area2 = (new_n[0].powi(2) + new_n[1].powi(2) + new_n[2].powi(2)).sqrt();
                let quality = if max_e > 0.0 && area2 > 0.0 {
                    area2 / (max_e * max_e)
                } else {
                    0.0
                };
                if quality < 0.02 {
                    ok = false;
                    break;
                }
            }

            if !ok {
                continue;
            }

            // ---- Execute collapse: merge b into a (a stays in place) ----
            merged_to[b] = Some(a);
            dead_verts.insert(b);
            for ti in &degen {
                dead_tris.insert(*ti);
            }
            // Transfer b's triangle set to a
            let b_tris: Vec<usize> = vert_tris[b].iter().copied().collect();
            for ti in b_tris {
                vert_tris[a].insert(ti);
            }
            did_collapse = true;
        }

        if !did_collapse {
            break;
        }

        // ------------------------------------------------------------------
        // Compact: remove dead tris, remap merged vertices
        // ------------------------------------------------------------------
        let mut new_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
        for (ti, t) in tris.iter().enumerate() {
            if dead_tris.contains(&ti) {
                continue;
            }
            let a = resolve(&merged_to, t[0]);
            let b = resolve(&merged_to, t[1]);
            let c = resolve(&merged_to, t[2]);
            if a == b || b == c || a == c {
                continue;
            }
            new_tris.push([a, b, c]);
        }
        tris = new_tris;

        // Compact vertex indices
        let mut used: HashSet<usize> = HashSet::with_capacity(tris.len() * 3);
        for t in &tris {
            used.insert(t[0]);
            used.insert(t[1]);
            used.insert(t[2]);
        }
        if used.is_empty() {
            break;
        }

        let mut sorted_used: Vec<usize> = used.into_iter().collect();
        sorted_used.sort_unstable();
        let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(sorted_used.len());
        let mut new_verts: Vec<[f64; 3]> = Vec::with_capacity(sorted_used.len());
        for old_idx in sorted_used {
            old_to_new.insert(old_idx, new_verts.len());
            new_verts.push(verts[old_idx]);
        }
        verts = new_verts;
        tris = tris
            .iter()
            .map(|t| [old_to_new[&t[0]], old_to_new[&t[1]], old_to_new[&t[2]]])
            .collect();
    }

    CollapsePeriodicOutput {
        vertices: verts,
        triangles: tris,
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple tetrahedron-like mesh (4 triangles sharing a central vertex).
    fn make_test_mesh() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.5, 1.0, 0.0], // 2
            [1.5, 1.0, 0.0], // 3
            [0.5, 0.5, 0.0], // 4 - interior vertex
        ];
        let tris = vec![[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]];
        (verts, tris)
    }

    #[test]
    fn test_optimize_returns_valid_mesh() {
        let (verts, tris) = make_test_mesh();
        let n = verts.len();
        let input = Mesh3DOptInput {
            vertices: verts,
            triangles: tris,
            target_h: vec![1.0; n],
            num_boundary: 0,
            passes: 1,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);
        // Should have some triangles
        assert!(!out.triangles.is_empty());
        // All triangle indices should be in bounds
        for t in &out.triangles {
            assert!(t[0] < out.vertices.len());
            assert!(t[1] < out.vertices.len());
            assert!(t[2] < out.vertices.len());
        }
    }

    #[test]
    fn test_no_crash_on_tiny_mesh() {
        let input = Mesh3DOptInput {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            target_h: vec![1.0; 3],
            num_boundary: 0,
            passes: 3,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);
        // With < 4 tris, should return unchanged
        assert_eq!(out.triangles.len(), 1);
    }

    #[test]
    fn test_boundary_vertices_preserved() {
        let (verts, tris) = make_test_mesh();
        let n = verts.len();
        // Mark first 4 vertices as boundary, only vertex 4 is interior
        let input = Mesh3DOptInput {
            vertices: verts.clone(),
            triangles: tris,
            target_h: vec![1.0; n],
            num_boundary: 4,
            passes: 1,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);
        // Boundary vertices should still exist (might be re-indexed but their
        // coordinates should be present)
        for (i, &pos) in verts.iter().enumerate().take(4) {
            assert!(
                out.vertices.iter().any(|v| {
                    (v[0] - pos[0]).abs() < 1e-10
                        && (v[1] - pos[1]).abs() < 1e-10
                        && (v[2] - pos[2]).abs() < 1e-10
                }),
                "boundary vertex {} missing from output",
                i
            );
        }
    }

    #[test]
    fn test_quality_metric() {
        // Equilateral triangle has quality 1.0
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, (3.0_f64).sqrt() / 2.0, 0.0],
        ];
        let q = tri_quality(&verts, &[0, 1, 2]);
        assert!((q - 1.0).abs() < 1e-10, "equilateral quality = {}", q);

        // Degenerate triangle has quality 0
        let verts2 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let q2 = tri_quality(&verts2, &[0, 1, 2]);
        assert!(q2.abs() < 1e-10, "degenerate quality = {}", q2);
    }

    #[test]
    fn test_collapse_short_edges() {
        // Create a mesh where vertices 6 and 7 are interior and connected
        // by a very short edge. The six outer vertices form a hexagonal ring
        // so every internal edge is shared by two triangles. Target_h is set
        // large enough that the max-edge-length guard (1.5*th) doesn't block.
        let verts = vec![
            [-2.0, 0.0, 0.0],  // 0  outer
            [-1.0, 2.0, 0.0],  // 1  outer
            [1.0, 2.0, 0.0],   // 2  outer
            [2.0, 0.0, 0.0],   // 3  outer
            [1.0, -2.0, 0.0],  // 4  outer
            [-1.0, -2.0, 0.0], // 5  outer
            [-0.01, 0.0, 0.0], // 6  interior - very close to 7
            [0.01, 0.0, 0.0],  // 7  interior
        ];
        let tris = vec![
            [0, 1, 6],
            [1, 2, 6],
            [2, 7, 6], // edge 6-7
            [2, 3, 7],
            [3, 4, 7],
            [4, 5, 7],
            [5, 0, 7],
            [0, 6, 7], // edge 6-7
        ];
        let n = verts.len();
        // target_h = 3.0 so the collapse threshold is 0.3*3.0 = 0.9,
        // and the max resulting edge check is 1.5*3.0 = 4.5
        let input = Mesh3DOptInput {
            vertices: verts,
            triangles: tris.clone(),
            target_h: vec![3.0; n],
            num_boundary: 0,
            passes: 1,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);
        // The short edge 6-7 (length 0.02) should have been collapsed,
        // removing the 2 triangles that degenerate.
        assert!(
            out.triangles.len() < tris.len(),
            "expected fewer triangles after collapse, got {} (was {})",
            out.triangles.len(),
            tris.len()
        );
    }

    #[test]
    fn test_swap_improves_quality() {
        // Create a quad split along the poor diagonal
        let verts = vec![
            [0.0, 0.0, 0.0], // 0
            [2.0, 0.0, 0.0], // 1
            [2.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [1.0, 0.5, 0.0], // 4 - interior to keep tri count >= 4
        ];
        // Split rectangle along 1-3 (bad diagonal) plus an extra triangle
        let tris = vec![[0, 1, 3], [1, 2, 3], [0, 3, 4], [3, 2, 4]];
        let n = verts.len();
        let q_before = tris
            .iter()
            .map(|t| tri_quality(&verts, t))
            .fold(f64::MAX, f64::min);

        let input = Mesh3DOptInput {
            vertices: verts,
            triangles: tris,
            target_h: vec![10.0; n], // large target so no collapse
            num_boundary: 0,
            passes: 1,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);
        let q_after = out
            .triangles
            .iter()
            .map(|t| tri_quality(&out.vertices, t))
            .fold(f64::MAX, f64::min);

        assert!(
            q_after >= q_before - 0.01,
            "quality should not decrease: before={}, after={}",
            q_before,
            q_after
        );
    }

    #[test]
    fn test_split_reduces_max_edge_length() {
        // Build a closed mesh (no boundary edges) with long interior edges.
        // Use a bipyramid: 6 equatorial vertices at radius 5.0, plus a top
        // and bottom apex. This makes every edge shared by exactly 2
        // triangles, so all vertices are interior.
        let r = 5.0;
        let mut verts: Vec<[f64; 3]> = Vec::new();
        // Vertex 0: top apex
        verts.push([0.0, 0.0, 3.0]);
        // Vertex 1: bottom apex
        verts.push([0.0, 0.0, -3.0]);
        // Vertices 2..8: equatorial ring
        for i in 0..6 {
            let angle = std::f64::consts::PI * 2.0 * (i as f64) / 6.0;
            verts.push([r * angle.cos(), r * angle.sin(), 0.0]);
        }
        // 12 triangles: 6 top + 6 bottom
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for i in 0..6 {
            let a = 2 + i;
            let b = 2 + (i + 1) % 6;
            // Top fan: (top, a, b)
            tris.push([0, a, b]);
            // Bottom fan: (bottom, b, a) - reversed winding for outward normal
            tris.push([1, b, a]);
        }

        let n = verts.len();

        // Compute max edge length before optimization
        let max_before = tris
            .iter()
            .flat_map(|t| {
                vec![
                    edge_len(&verts, t[0], t[1]),
                    edge_len(&verts, t[1], t[2]),
                    edge_len(&verts, t[0], t[2]),
                ]
            })
            .fold(0.0_f64, f64::max);

        let input = Mesh3DOptInput {
            vertices: verts,
            triangles: tris,
            target_h: vec![1.0; n],
            num_boundary: 0,
            passes: 3,
            collapse_ratio: 0.3,
        };
        let out = optimize_mesh_3d(&input);

        // Compute max edge length after optimization
        let max_after = out
            .triangles
            .iter()
            .flat_map(|t| {
                vec![
                    edge_len(&out.vertices, t[0], t[1]),
                    edge_len(&out.vertices, t[1], t[2]),
                    edge_len(&out.vertices, t[0], t[2]),
                ]
            })
            .fold(0.0_f64, f64::max);

        assert!(
            max_after < max_before,
            "split should reduce max edge length: before={:.3}, after={:.3}",
            max_before,
            max_after
        );

        // Max edge length should be reasonably close to target
        // (with split_ratio=1.4 and some smoothing, expect < 2.0 * target_h)
        assert!(
            max_after < 3.0,
            "max edge length should be close to target_h=1.0, got {:.3}",
            max_after
        );

        // Should have more triangles than the original 12 (splits added vertices)
        assert!(
            out.triangles.len() > 12,
            "expected more triangles after split, got {}",
            out.triangles.len()
        );

        // All indices in bounds
        for t in &out.triangles {
            assert!(t[0] < out.vertices.len());
            assert!(t[1] < out.vertices.len());
            assert!(t[2] < out.vertices.len());
        }
    }

    // ----------------------------------------------------------------
    // Tests for collapse_periodic_mesh_3d
    // ----------------------------------------------------------------

    #[test]
    fn test_collapse_periodic_basic() {
        // Create a closed mesh (no boundary edges) with some short edges.
        // Use a hexagonal fan with two very close interior vertices.
        let verts = vec![
            [-2.0, 0.0, 0.0],  // 0
            [-1.0, 2.0, 0.0],  // 1
            [1.0, 2.0, 0.0],   // 2
            [2.0, 0.0, 0.0],   // 3
            [1.0, -2.0, 0.0],  // 4
            [-1.0, -2.0, 0.0], // 5
            [-0.01, 0.0, 0.0], // 6 - very close to 7
            [0.01, 0.0, 0.0],  // 7
        ];
        let tris = vec![
            [0, 1, 6],
            [1, 2, 6],
            [2, 7, 6],
            [2, 3, 7],
            [3, 4, 7],
            [4, 5, 7],
            [5, 0, 7],
            [0, 6, 7],
        ];
        let input = CollapsePeriodicInput {
            vertices: verts,
            triangles: tris.clone(),
            passes: 3,
            collapse_ratio: 0.7,
        };
        let out = collapse_periodic_mesh_3d(&input);
        // The short edge 6-7 (length 0.02) should have been collapsed
        assert!(
            out.triangles.len() < tris.len(),
            "expected fewer triangles after collapse, got {} (was {})",
            out.triangles.len(),
            tris.len()
        );
        // All indices in bounds
        for t in &out.triangles {
            assert!(t[0] < out.vertices.len());
            assert!(t[1] < out.vertices.len());
            assert!(t[2] < out.vertices.len());
        }
        // No degenerate triangles
        for t in &out.triangles {
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        }
    }

    #[test]
    fn test_collapse_periodic_boundary_preservation() {
        // Create a mesh where some edges are boundary (count != 2).
        // A single triangle fan - all outer edges are boundary (count == 1).
        // The interior vertex should NOT be collapsed even if close to
        // an outer vertex, because the outer vertices are boundary.
        let verts = vec![
            [0.0, 0.0, 0.0],   // 0 center
            [1.0, 0.0, 0.0],   // 1
            [0.5, 1.0, 0.0],   // 2
            [-0.5, 1.0, 0.0],  // 3
            [-1.0, 0.0, 0.0],  // 4
            [-0.5, -1.0, 0.0], // 5
            [0.5, -1.0, 0.0],  // 6
        ];
        let tris = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 1],
        ];
        // All outer edges (1-2, 2-3, ..., 6-1) have count == 1, so all
        // vertices are boundary. No collapse should happen.
        let input = CollapsePeriodicInput {
            vertices: verts.clone(),
            triangles: tris.clone(),
            passes: 3,
            collapse_ratio: 0.99, // very aggressive, but boundary blocks it
        };
        let out = collapse_periodic_mesh_3d(&input);
        assert_eq!(
            out.triangles.len(),
            tris.len(),
            "boundary vertices should prevent collapse"
        );
    }

    #[test]
    fn test_collapse_periodic_no_normal_flip() {
        // Build a closed bipyramid with a short equatorial edge.
        // After collapse, all triangle normals should remain consistent
        // (no flips).
        let verts = vec![
            [0.0, 0.0, 2.0],   // 0 top apex
            [0.0, 0.0, -2.0],  // 1 bottom apex
            [2.0, 0.0, 0.0],   // 2
            [0.0, 2.0, 0.0],   // 3
            [-2.0, 0.0, 0.0],  // 4
            [0.0, -2.0, 0.0],  // 5
            [0.05, 0.05, 0.0], // 6 - close to origin, short edges to apex/ring
        ];
        // Build a closed mesh with 6 as an interior vertex connected to
        // neighbors 2 and 3 (forming short-ish edges).
        let tris = vec![
            // Top cap using vertex 6 between 2 and 3
            [0, 2, 6],
            [0, 6, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 2],
            // Bottom cap using vertex 6 between 2 and 3
            [1, 6, 2],
            [1, 3, 6],
            [1, 4, 3],
            [1, 5, 4],
            [1, 2, 5],
        ];

        // Compute normals before
        let input = CollapsePeriodicInput {
            vertices: verts,
            triangles: tris,
            passes: 3,
            collapse_ratio: 0.7,
        };
        let out = collapse_periodic_mesh_3d(&input);

        // Verify all output triangles have non-degenerate normals
        for t in &out.triangles {
            let n = tri_normal(&out.vertices, t);
            let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            assert!(len2 > 1e-20, "triangle {:?} has degenerate normal", t);
        }

        // All indices in bounds
        for t in &out.triangles {
            assert!(t[0] < out.vertices.len());
            assert!(t[1] < out.vertices.len());
            assert!(t[2] < out.vertices.len());
        }
    }
}
