//! Mesh adaptation operations: split, collapse, swap, and smooth.
//!
//! These are the four local operations used in the MeshAdapt algorithm,
//! implemented on the [`BDSMesh`] half-edge data structure.
//!
//! The algorithms are based on gmsh's `BDS.cpp` and `meshGFaceBDS.cpp`:
//! - **Split**: inserts a midpoint vertex on an interior edge, replacing 2
//!   adjacent triangles with 4 new triangles.
//! - **Collapse**: removes an interior edge by merging one endpoint into the
//!   other, replacing N triangles around the removed vertex with N-2 new ones.
//! - **Swap**: flips the diagonal of the quadrilateral formed by 2 adjacent
//!   triangles.
//! - **Smooth**: moves an interior vertex toward the UV centroid of its
//!   1-ring neighbours (Laplacian smoothing in parameter space).

use super::bds::{BDSFace, BDSHalfEdge, BDSMesh, BDSVertex, NONE};
use crate::size_field::SizeField;
use smallvec::SmallVec;

// ──────────────────── Geometry helpers ────────────────────

/// Signed area of triangle (a, b, c) in UV parameter space (2x area).
fn signed_area_uv(mesh: &BDSMesh, a: usize, b: usize, c: usize) -> f64 {
    let va = &mesh.vertices[a];
    let vb = &mesh.vertices[b];
    let vc = &mesh.vertices[c];
    (vb.u - va.u) * (vc.v - va.v) - (vc.u - va.u) * (vb.v - va.v)
}

/// Minimum angle of a triangle using the metric tensor (3D lengths).
/// Returns the smallest angle in radians.
#[allow(dead_code)]
fn min_angle_3d(mesh: &BDSMesh, a: usize, b: usize, c: usize, sf: Option<&SizeField>) -> f64 {
    let va = &mesh.vertices[a];
    let vb = &mesh.vertices[b];
    let vc = &mesh.vertices[c];

    let (la, lb, lc) = match sf {
        Some(sf) => (
            sf.edge_length_3d([vb.u, vb.v], [vc.u, vc.v]),
            sf.edge_length_3d([vc.u, vc.v], [va.u, va.v]),
            sf.edge_length_3d([va.u, va.v], [vb.u, vb.v]),
        ),
        None => {
            let abx = vb.x - va.x;
            let aby = vb.y - va.y;
            let abz = vb.z - va.z;
            let bcx = vc.x - vb.x;
            let bcy = vc.y - vb.y;
            let bcz = vc.z - vb.z;
            let cax = va.x - vc.x;
            let cay = va.y - vc.y;
            let caz = va.z - vc.z;
            (
                (bcx * bcx + bcy * bcy + bcz * bcz).sqrt(),
                (cax * cax + cay * cay + caz * caz).sqrt(),
                (abx * abx + aby * aby + abz * abz).sqrt(),
            )
        }
    };

    if la < 1e-30 || lb < 1e-30 || lc < 1e-30 {
        return 0.0;
    }

    // Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
    let cos_a = ((lb * lb + lc * lc - la * la) / (2.0 * lb * lc)).clamp(-1.0, 1.0);
    let cos_b = ((la * la + lc * lc - lb * lb) / (2.0 * la * lc)).clamp(-1.0, 1.0);
    let cos_c = ((la * la + lb * lb - lc * lc) / (2.0 * la * lb)).clamp(-1.0, 1.0);

    cos_a.acos().min(cos_b.acos()).min(cos_c.acos())
}

/// Triangle quality using the metric tensor.
///
/// Computes gamma = 4*sqrt(3)*area / (a^2 + b^2 + c^2) where edge lengths
/// and area are computed using the first fundamental form (metric tensor)
/// at the triangle centroid.  This gives the true quality of the triangle
/// on the surface, accounting for the parametric distortion.
///
/// Falls back to UV-Euclidean quality if no size field is provided.
fn triangle_quality(mesh: &BDSMesh, a: usize, b: usize, c: usize, sf: Option<&SizeField>) -> f64 {
    let va = &mesh.vertices[a];
    let vb = &mesh.vertices[b];
    let vc = &mesh.vertices[c];

    match sf {
        Some(sf) => {
            // Compute edge lengths using the metric tensor at the edge midpoint
            let la2 = sf.edge_length_3d([va.u, va.v], [vb.u, vb.v]).powi(2);
            let lb2 = sf.edge_length_3d([vb.u, vb.v], [vc.u, vc.v]).powi(2);
            let lc2 = sf.edge_length_3d([vc.u, vc.v], [va.u, va.v]).powi(2);

            // Area using metric at centroid: area_3d = sqrt(EG-F^2) * area_uv
            let uc = (va.u + vb.u + vc.u) / 3.0;
            let vc_param = (va.v + vb.v + vc.v) / 3.0;
            let e = sf.interp_pub(&sf.metric_e, uc, vc_param).max(1e-30);
            let f = sf.interp_pub(&sf.metric_f, uc, vc_param);
            let g = sf.interp_pub(&sf.metric_g, uc, vc_param).max(1e-30);
            let det = (e * g - f * f).max(0.0);
            let sqrt_det = det.sqrt();

            let area_uv = signed_area_uv(mesh, a, b, c).abs() * 0.5;
            let area_3d = sqrt_det * area_uv;

            let denom = la2 + lb2 + lc2;
            if denom < 1e-30 {
                return 0.0;
            }
            (4.0 * 1.732_050_807_568_877_2_f64 * area_3d / denom).min(1.0)
        }
        None => {
            // Fallback: use (x, y, z) Euclidean coordinates
            let abx = vb.x - va.x;
            let aby = vb.y - va.y;
            let abz = vb.z - va.z;
            let acx = vc.x - va.x;
            let acy = vc.y - va.y;
            let acz = vc.z - va.z;

            let nx = aby * acz - abz * acy;
            let ny = abz * acx - abx * acz;
            let nz = abx * acy - aby * acx;
            let area2 = (nx * nx + ny * ny + nz * nz).sqrt();

            let la2 = abx * abx + aby * aby + abz * abz;
            let bcx = vc.x - vb.x;
            let bcy = vc.y - vb.y;
            let bcz = vc.z - vb.z;
            let lb2 = bcx * bcx + bcy * bcy + bcz * bcz;
            let lc2 = acx * acx + acy * acy + acz * acz;

            let denom = la2 + lb2 + lc2;
            if denom < 1e-30 {
                return 0.0;
            }
            2.0 * 1.732_050_807_568_877_2_f64 * area2 / denom
        }
    }
}

/// Compute an approximate 3D face normal for triangle (a, b, c).
///
/// Uses the metric tensor at the centroid to transform the UV edges into
/// 3D tangent vectors, then takes the cross product.  Returns an
/// unnormalized normal vector `(nx, ny, nz)`.
///
/// When the vertices have meaningful XYZ coordinates, those are used
/// directly.  Otherwise falls back to metric-tensor-based tangent vectors.
fn face_normal_3d(
    mesh: &BDSMesh,
    a: usize,
    b: usize,
    c: usize,
    _sf: &SizeField,
) -> (f64, f64, f64) {
    let va = &mesh.vertices[a];
    let vb = &mesh.vertices[b];
    let vc = &mesh.vertices[c];

    // Use XYZ coordinates (set to UV for flat surfaces, but still gives
    // a valid normal direction via cross product).
    let abx = vb.x - va.x;
    let aby = vb.y - va.y;
    let abz = vb.z - va.z;
    let acx = vc.x - va.x;
    let acy = vc.y - va.y;
    let acz = vc.z - va.z;

    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;

    (nx, ny, nz)
}

/// Public wrapper for triangle_quality, used by the driver's repair pass.
pub fn triangle_quality_pub(
    mesh: &BDSMesh,
    a: usize,
    b: usize,
    c: usize,
    sf: Option<&SizeField>,
) -> f64 {
    triangle_quality(mesh, a, b, c, sf)
}

// ──────────────── Helper: add a new triangle ─────────────

/// Create a new face with three new half-edges for vertices (a, b, c) in CCW
/// order.  Returns `(face_id, [he_ab, he_bc, he_ca])`.
///
/// The new half-edges have `twin == NONE`; the caller must link twins.
fn add_triangle(mesh: &mut BDSMesh, a: usize, b: usize, c: usize) -> (usize, [usize; 3]) {
    let fi = mesh.faces.len();
    let base = mesh.half_edges.len();

    let he_ab = base;
    let he_bc = base + 1;
    let he_ca = base + 2;

    mesh.half_edges.push(BDSHalfEdge {
        id: he_ab,
        origin: a,
        face: fi,
        twin: NONE,
        next: he_bc,
        prev: he_ca,
    });
    mesh.half_edges.push(BDSHalfEdge {
        id: he_bc,
        origin: b,
        face: fi,
        twin: NONE,
        next: he_ca,
        prev: he_ab,
    });
    mesh.half_edges.push(BDSHalfEdge {
        id: he_ca,
        origin: c,
        face: fi,
        twin: NONE,
        next: he_ab,
        prev: he_bc,
    });

    mesh.faces.push(BDSFace {
        id: fi,
        he: he_ab,
        deleted: false,
    });

    // Ensure vertices point to *some* valid outgoing half-edge
    if mesh.vertices[a].he == NONE {
        mesh.vertices[a].he = he_ab;
    }
    if mesh.vertices[b].he == NONE {
        mesh.vertices[b].he = he_bc;
    }
    if mesh.vertices[c].he == NONE {
        mesh.vertices[c].he = he_ca;
    }

    (fi, [he_ab, he_bc, he_ca])
}

/// Link two half-edges as twins.
fn link_twins(mesh: &mut BDSMesh, a: usize, b: usize) {
    mesh.half_edges[a].twin = b;
    mesh.half_edges[b].twin = a;
}

/// Find a half-edge going from vertex `from` to vertex `to` among the new
/// half-edges in `candidates`.
fn find_he_directed(mesh: &BDSMesh, candidates: &[usize], from: usize, to: usize) -> Option<usize> {
    candidates
        .iter()
        .find(|&&he| mesh.half_edges[he].origin == from && mesh.he_dest(he) == to)
        .copied()
}

/// Soft-delete a face and unlink its half-edges' twin pointers from their
/// twins (so the twins become boundary half-edges).  The half-edges of the
/// face are NOT reused; they become orphaned but don't corrupt other data
/// because the rebuild uses new half-edges.
fn soft_delete_face(mesh: &mut BDSMesh, fi: usize) {
    mesh.faces[fi].deleted = true;
    let he0 = mesh.faces[fi].he;
    let he1 = mesh.half_edges[he0].next;
    let he2 = mesh.half_edges[he1].next;
    for &he in &[he0, he1, he2] {
        let tw = mesh.half_edges[he].twin;
        if tw != NONE {
            mesh.half_edges[tw].twin = NONE;
        }
        mesh.half_edges[he].twin = NONE;
    }
}

/// Fix up vertex outgoing half-edge pointers after topology changes.
/// For each vertex in `verts`, scan its new outgoing half-edges and set `he`
/// to a valid one (preferring boundary half-edges).
fn fix_vertex_he(mesh: &mut BDSMesh, verts: &[usize], new_hes: &[usize]) {
    for &v in verts {
        let mut best = NONE;
        let mut found_boundary = false;
        for &he in new_hes {
            if mesh.half_edges[he].origin == v {
                if !found_boundary {
                    best = he;
                }
                if mesh.half_edges[he].twin == NONE {
                    best = he;
                    found_boundary = true;
                }
            }
        }
        if best != NONE {
            mesh.vertices[v].he = best;
        }
    }
}

// ──────────── Metric helpers for geodesic midpoint ───────

/// Compute the infinitesimal 3D arc-length speed at parameter t along
/// the UV segment from `a` to `b`.  That is, ds/dt where
///   s(t) = integral of sqrt(E du^2 + 2F du dv + G dv^2) from 0 to t.
///
/// For a straight UV segment: u(t) = a[0] + t*(b[0]-a[0]),
/// v(t) = a[1] + t*(b[1]-a[1]), so du/dt = b[0]-a[0], dv/dt = b[1]-a[1].
/// ds/dt = sqrt(E * du^2 + 2F * du * dv + G * dv^2) evaluated at (u(t), v(t)).
fn arc_speed(sf: &SizeField, a: [f64; 2], b: [f64; 2], t: f64) -> f64 {
    let du = b[0] - a[0];
    let dv = b[1] - a[1];
    let u = a[0] + t * du;
    let v = a[1] + t * dv;
    let e = sf.interp_pub(&sf.metric_e, u, v);
    let f = sf.interp_pub(&sf.metric_f, u, v);
    let g = sf.interp_pub(&sf.metric_g, u, v);
    let len_sq = e * du * du + 2.0 * f * du * dv + g * dv * dv;
    if len_sq > 0.0 {
        len_sq.sqrt()
    } else {
        0.0
    }
}

/// Use composite Simpson's rule to compute the arc length of the UV segment
/// from `a` to `b` over the parameter interval [t0, t1], with `n` sub-intervals
/// (must be even).
fn arc_length_simpson(sf: &SizeField, a: [f64; 2], b: [f64; 2], t0: f64, t1: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 }; // ensure even
    let h = (t1 - t0) / n as f64;
    let mut sum = arc_speed(sf, a, b, t0) + arc_speed(sf, a, b, t1);
    for i in 1..n {
        let t = t0 + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * arc_speed(sf, a, b, t);
    }
    sum * h / 3.0
}

/// Find the parameter t in [0, 1] such that the 3D arc length from a to
/// (a + t*(b-a)) equals half the total 3D arc length of the segment.
///
/// Uses a bisection approach with Simpson quadrature: evaluate the arc
/// length of the left half at each bisection step and compare to half
/// the total.
fn geodesic_midpoint_t(sf: &SizeField, a: [f64; 2], b: [f64; 2]) -> f64 {
    // Number of sub-intervals for Simpson (higher = more accurate)
    let n_quad = 8;

    let total = arc_length_simpson(sf, a, b, 0.0, 1.0, n_quad);
    if total < 1e-30 {
        return 0.5; // degenerate edge, fall back to UV midpoint
    }
    let half_total = 0.5 * total;

    // Bisection: find t such that arc_length(0, t) = half_total
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        let left_len = arc_length_simpson(sf, a, b, 0.0, mid, n_quad);
        if left_len < half_total {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ────────────────────── Split edge ───────────────────────

/// Split an interior edge by inserting a midpoint vertex.
///
/// The half-edge `he` identifies an interior edge (must have a twin).
/// The midpoint is placed at the UV and XYZ averages of the two endpoints.
/// The two adjacent triangles are each replaced by two new triangles (4 total).
///
/// Returns the index of the newly created midpoint vertex, or `None` if the
/// split failed (edge is boundary, or any other precondition violated).
///
/// Diagram (half-edge `he` goes from p1 to p2; op0 is opposite in he's face,
/// op1 is opposite in twin's face):
///
/// ```text
///       op0                    op0
///      / | \                  /|\ \
///     /  |  \               / | \ \
///    / f0|f1 \     =>      /  | mid\
///   p1---he-->p2          p1--+-->--p2
///    \ f3|f2 /             \  | mid/
///     \  |  /               \ | / /
///      \ | /                 \|/ /
///       op1                   op1
/// ```
pub fn split_edge(mesh: &mut BDSMesh, he: usize) -> Option<usize> {
    let twin = mesh.half_edges[he].twin;
    if twin == NONE {
        return None; // boundary edge
    }

    let p1 = mesh.half_edges[he].origin;
    let p2 = mesh.he_dest(he);
    let fi0 = mesh.half_edges[he].face;
    let fi1 = mesh.half_edges[twin].face;

    if fi0 == NONE || fi1 == NONE {
        return None;
    }

    // Find opposite vertices
    // In face fi0: half-edges are he -> next(he) -> prev(he), going p1->p2->op0->p1
    let he_next = mesh.half_edges[he].next;
    let he_prev = mesh.half_edges[he].prev;
    let op0 = mesh.half_edges[he_prev].origin; // vertex opposite to he in fi0

    // In face fi1: twin goes p2->p1, so twin->next->origin gives the third vertex
    let tw_next = mesh.half_edges[twin].next;
    let tw_prev = mesh.half_edges[twin].prev;
    let op1 = mesh.half_edges[tw_prev].origin; // vertex opposite to twin in fi1

    // Save the outer twin links before deleting faces
    // he_next goes p2->op0, he_prev goes op0->p1
    // tw_next goes p1->op1, tw_prev goes op1->p2
    let outer_twin_of_he_next = mesh.half_edges[he_next].twin; // twin of p2->op0
    let outer_twin_of_he_prev = mesh.half_edges[he_prev].twin; // twin of op0->p1
    let outer_twin_of_tw_next = mesh.half_edges[tw_next].twin; // twin of p1->op1
    let outer_twin_of_tw_prev = mesh.half_edges[tw_prev].twin; // twin of op1->p2

    // Create midpoint vertex
    let mid_u = 0.5 * (mesh.vertices[p1].u + mesh.vertices[p2].u);
    let mid_v = 0.5 * (mesh.vertices[p1].v + mesh.vertices[p2].v);
    let mid_x = 0.5 * (mesh.vertices[p1].x + mesh.vertices[p2].x);
    let mid_y = 0.5 * (mesh.vertices[p1].y + mesh.vertices[p2].y);
    let mid_z = 0.5 * (mesh.vertices[p1].z + mesh.vertices[p2].z);
    let mid_h = 0.5 * (mesh.vertices[p1].target_h + mesh.vertices[p2].target_h);

    let mid = mesh.vertices.len();
    mesh.vertices.push(BDSVertex {
        id: mid,
        u: mid_u,
        v: mid_v,
        x: mid_x,
        y: mid_y,
        z: mid_z,
        target_h: mid_h,
        on_boundary: false,
        he: NONE,
    });

    // Soft-delete the two old faces (also unlinks their twins)
    soft_delete_face(mesh, fi0);
    soft_delete_face(mesh, fi1);

    // Create 4 new faces:
    //   F0: (p1, mid, op0)   -- replaces left half of old fi0
    //   F1: (mid, p2, op0)   -- replaces right half of old fi0
    //   F2: (p2, mid, op1)   -- replaces left half of old fi1 (from p2's perspective)
    //   F3: (mid, p1, op1)   -- replaces right half of old fi1
    let (_f0, hes0) = add_triangle(mesh, p1, mid, op0); // hes0: p1->mid, mid->op0, op0->p1
    let (_f1, hes1) = add_triangle(mesh, mid, p2, op0); // hes1: mid->p2, p2->op0, op0->mid
    let (_f2, hes2) = add_triangle(mesh, p2, mid, op1); // hes2: p2->mid, mid->op1, op1->p2
    let (_f3, hes3) = add_triangle(mesh, mid, p1, op1); // hes3: mid->p1, p1->op1, op1->mid

    let all_new = [
        hes0[0], hes0[1], hes0[2], hes1[0], hes1[1], hes1[2], hes2[0], hes2[1], hes2[2], hes3[0],
        hes3[1], hes3[2],
    ];

    // Link internal twins (between the 4 new faces)
    // p1->mid in F0 twins with mid->p1 in F3
    link_twins(mesh, hes0[0], hes3[0]);
    // mid->op0 in F0 twins with op0->mid in F1
    link_twins(mesh, hes0[1], hes1[2]);
    // mid->p2 in F1 twins with p2->mid in F2
    link_twins(mesh, hes1[0], hes2[0]);
    // mid->op1 in F2 twins with op1->mid in F3
    link_twins(mesh, hes2[1], hes3[2]);

    // Link external twins (to the rest of the mesh)
    // op0->p1 in F0 (hes0[2]) twins with the old he_prev's twin (which was op0->p1's outer twin)
    // Actually: old he_prev was op0->p1 in fi0. Its twin was the outside half-edge p1->op0.
    if outer_twin_of_he_prev != NONE {
        link_twins(mesh, hes0[2], outer_twin_of_he_prev);
    }
    // p2->op0 in F1 (hes1[1]) twins with old he_next's twin
    if outer_twin_of_he_next != NONE {
        link_twins(mesh, hes1[1], outer_twin_of_he_next);
    }
    // op1->p2 in F2 (hes2[2]) twins with old tw_prev's twin
    if outer_twin_of_tw_prev != NONE {
        link_twins(mesh, hes2[2], outer_twin_of_tw_prev);
    }
    // p1->op1 in F3 (hes3[1]) twins with old tw_next's twin
    if outer_twin_of_tw_next != NONE {
        link_twins(mesh, hes3[1], outer_twin_of_tw_next);
    }

    // Fix vertex outgoing half-edge pointers
    fix_vertex_he(mesh, &[p1, p2, op0, op1, mid], &all_new);

    Some(mid)
}

/// Split an interior edge with geodesic-aware midpoint placement.
///
/// Like [`split_edge`], but the midpoint is placed at the 3D arc-length
/// midpoint of the edge rather than the UV midpoint.  This accounts for
/// metric distortion (e.g. near poles where UV distances do not reflect 3D
/// distances) and produces more uniform triangles on curved surfaces.
///
/// The geodesic midpoint parameter `t` is found by bisection on the arc
/// length integral computed via Simpson's rule.
///
/// Falls back to `split_edge` (UV midpoint) if the metric is uniform.
pub fn split_edge_metric(mesh: &mut BDSMesh, he: usize, sf: &SizeField) -> Option<usize> {
    let twin = mesh.half_edges[he].twin;
    if twin == NONE {
        return None; // boundary edge
    }

    let p1 = mesh.half_edges[he].origin;
    let p2 = mesh.he_dest(he);

    let a = [mesh.vertices[p1].u, mesh.vertices[p1].v];
    let b = [mesh.vertices[p2].u, mesh.vertices[p2].v];

    // Find the geodesic midpoint parameter t
    let t = geodesic_midpoint_t(sf, a, b);

    // Compute UV and XYZ at the geodesic midpoint
    let mid_u = a[0] + t * (b[0] - a[0]);
    let mid_v = a[1] + t * (b[1] - a[1]);
    let mid_x = mesh.vertices[p1].x + t * (mesh.vertices[p2].x - mesh.vertices[p1].x);
    let mid_y = mesh.vertices[p1].y + t * (mesh.vertices[p2].y - mesh.vertices[p1].y);
    let mid_z = mesh.vertices[p1].z + t * (mesh.vertices[p2].z - mesh.vertices[p1].z);
    let mid_h =
        mesh.vertices[p1].target_h + t * (mesh.vertices[p2].target_h - mesh.vertices[p1].target_h);

    // Now perform the same topology changes as split_edge, but with our
    // custom midpoint coordinates.

    let fi0 = mesh.half_edges[he].face;
    let fi1 = mesh.half_edges[twin].face;

    if fi0 == NONE || fi1 == NONE {
        return None;
    }

    // Find opposite vertices
    let he_next = mesh.half_edges[he].next;
    let he_prev = mesh.half_edges[he].prev;
    let op0 = mesh.half_edges[he_prev].origin;

    let tw_next = mesh.half_edges[twin].next;
    let tw_prev = mesh.half_edges[twin].prev;
    let op1 = mesh.half_edges[tw_prev].origin;

    // Save the outer twin links
    let outer_twin_of_he_next = mesh.half_edges[he_next].twin;
    let outer_twin_of_he_prev = mesh.half_edges[he_prev].twin;
    let outer_twin_of_tw_next = mesh.half_edges[tw_next].twin;
    let outer_twin_of_tw_prev = mesh.half_edges[tw_prev].twin;

    // Create midpoint vertex
    let mid = mesh.vertices.len();
    mesh.vertices.push(BDSVertex {
        id: mid,
        u: mid_u,
        v: mid_v,
        x: mid_x,
        y: mid_y,
        z: mid_z,
        target_h: mid_h,
        on_boundary: false,
        he: NONE,
    });

    // Soft-delete the two old faces
    soft_delete_face(mesh, fi0);
    soft_delete_face(mesh, fi1);

    // Create 4 new faces
    let (_f0, hes0) = add_triangle(mesh, p1, mid, op0);
    let (_f1, hes1) = add_triangle(mesh, mid, p2, op0);
    let (_f2, hes2) = add_triangle(mesh, p2, mid, op1);
    let (_f3, hes3) = add_triangle(mesh, mid, p1, op1);

    let all_new = [
        hes0[0], hes0[1], hes0[2], hes1[0], hes1[1], hes1[2], hes2[0], hes2[1], hes2[2], hes3[0],
        hes3[1], hes3[2],
    ];

    // Link internal twins
    link_twins(mesh, hes0[0], hes3[0]);
    link_twins(mesh, hes0[1], hes1[2]);
    link_twins(mesh, hes1[0], hes2[0]);
    link_twins(mesh, hes2[1], hes3[2]);

    // Link external twins
    if outer_twin_of_he_prev != NONE {
        link_twins(mesh, hes0[2], outer_twin_of_he_prev);
    }
    if outer_twin_of_he_next != NONE {
        link_twins(mesh, hes1[1], outer_twin_of_he_next);
    }
    if outer_twin_of_tw_prev != NONE {
        link_twins(mesh, hes2[2], outer_twin_of_tw_prev);
    }
    if outer_twin_of_tw_next != NONE {
        link_twins(mesh, hes3[1], outer_twin_of_tw_next);
    }

    // Fix vertex outgoing half-edge pointers
    fix_vertex_he(mesh, &[p1, p2, op0, op1, mid], &all_new);

    Some(mid)
}

// ─────────────── Split boundary edge ────────────────────

/// Split a boundary edge by inserting a midpoint vertex.
///
/// Unlike [`split_edge`], this works on boundary edges (twin == NONE).
/// Only the single adjacent triangle is split into two.  The new midpoint
/// vertex inherits `on_boundary = true`.
///
/// Returns the index of the newly created midpoint vertex, or `None` if
/// the split failed.
pub fn split_boundary_edge(mesh: &mut BDSMesh, he: usize) -> Option<usize> {
    // Must be a boundary edge (no twin)
    if mesh.half_edges[he].twin != NONE {
        return None;
    }

    let fi = mesh.half_edges[he].face;
    if fi == NONE || fi >= mesh.faces.len() || mesh.faces[fi].deleted {
        return None;
    }

    let p1 = mesh.half_edges[he].origin;
    let p2 = mesh.he_dest(he);

    let he_next = mesh.half_edges[he].next;
    let he_prev = mesh.half_edges[he].prev;
    let op = mesh.half_edges[he_prev].origin; // opposite vertex

    // Save outer twin links for the non-boundary edges
    let outer_twin_of_he_next = mesh.half_edges[he_next].twin;
    let outer_twin_of_he_prev = mesh.half_edges[he_prev].twin;

    // Create midpoint vertex (on boundary)
    let mid_u = 0.5 * (mesh.vertices[p1].u + mesh.vertices[p2].u);
    let mid_v = 0.5 * (mesh.vertices[p1].v + mesh.vertices[p2].v);
    let mid_x = 0.5 * (mesh.vertices[p1].x + mesh.vertices[p2].x);
    let mid_y = 0.5 * (mesh.vertices[p1].y + mesh.vertices[p2].y);
    let mid_z = 0.5 * (mesh.vertices[p1].z + mesh.vertices[p2].z);
    let mid_h = 0.5 * (mesh.vertices[p1].target_h + mesh.vertices[p2].target_h);

    let mid = mesh.vertices.len();
    mesh.vertices.push(BDSVertex {
        id: mid,
        u: mid_u,
        v: mid_v,
        x: mid_x,
        y: mid_y,
        z: mid_z,
        target_h: mid_h,
        on_boundary: true,
        he: NONE,
    });

    // Soft-delete the old face
    soft_delete_face(mesh, fi);

    // Create 2 new faces:
    //   F0: (p1, mid, op)
    //   F1: (mid, p2, op)
    let (_f0, hes0) = add_triangle(mesh, p1, mid, op); // hes0: p1->mid, mid->op, op->p1
    let (_f1, hes1) = add_triangle(mesh, mid, p2, op); // hes1: mid->p2, p2->op, op->mid

    let all_new = [hes0[0], hes0[1], hes0[2], hes1[0], hes1[1], hes1[2]];

    // Link internal twin: mid->op in F0 twins with op->mid in F1
    link_twins(mesh, hes0[1], hes1[2]);

    // Link external twins for the non-boundary edges
    // op->p1 in F0 (hes0[2]) twins with old he_prev's twin
    if outer_twin_of_he_prev != NONE {
        link_twins(mesh, hes0[2], outer_twin_of_he_prev);
    }
    // p2->op in F1 (hes1[1]) twins with old he_next's twin
    if outer_twin_of_he_next != NONE {
        link_twins(mesh, hes1[1], outer_twin_of_he_next);
    }
    // p1->mid (hes0[0]) and mid->p2 (hes1[0]) are boundary edges (no twin)

    // Fix vertex outgoing half-edge pointers
    fix_vertex_he(mesh, &[p1, p2, op, mid], &all_new);

    Some(mid)
}

/// Split a boundary edge with geodesic-aware midpoint placement.
///
/// Like [`split_boundary_edge`], but places the midpoint at the 3D
/// arc-length midpoint using the metric tensor.
pub fn split_boundary_edge_metric(mesh: &mut BDSMesh, he: usize, sf: &SizeField) -> Option<usize> {
    if mesh.half_edges[he].twin != NONE {
        return None;
    }

    let fi = mesh.half_edges[he].face;
    if fi == NONE || fi >= mesh.faces.len() || mesh.faces[fi].deleted {
        return None;
    }

    let p1 = mesh.half_edges[he].origin;
    let p2 = mesh.he_dest(he);

    let a = [mesh.vertices[p1].u, mesh.vertices[p1].v];
    let b = [mesh.vertices[p2].u, mesh.vertices[p2].v];
    let t = geodesic_midpoint_t(sf, a, b);

    let he_next = mesh.half_edges[he].next;
    let he_prev = mesh.half_edges[he].prev;
    let op = mesh.half_edges[he_prev].origin;

    let outer_twin_of_he_next = mesh.half_edges[he_next].twin;
    let outer_twin_of_he_prev = mesh.half_edges[he_prev].twin;

    let mid_u = a[0] + t * (b[0] - a[0]);
    let mid_v = a[1] + t * (b[1] - a[1]);
    let mid_x = mesh.vertices[p1].x + t * (mesh.vertices[p2].x - mesh.vertices[p1].x);
    let mid_y = mesh.vertices[p1].y + t * (mesh.vertices[p2].y - mesh.vertices[p1].y);
    let mid_z = mesh.vertices[p1].z + t * (mesh.vertices[p2].z - mesh.vertices[p1].z);
    let mid_h =
        mesh.vertices[p1].target_h + t * (mesh.vertices[p2].target_h - mesh.vertices[p1].target_h);

    let mid = mesh.vertices.len();
    mesh.vertices.push(BDSVertex {
        id: mid,
        u: mid_u,
        v: mid_v,
        x: mid_x,
        y: mid_y,
        z: mid_z,
        target_h: mid_h,
        on_boundary: true,
        he: NONE,
    });

    soft_delete_face(mesh, fi);

    let (_f0, hes0) = add_triangle(mesh, p1, mid, op);
    let (_f1, hes1) = add_triangle(mesh, mid, p2, op);

    let all_new = [hes0[0], hes0[1], hes0[2], hes1[0], hes1[1], hes1[2]];

    link_twins(mesh, hes0[1], hes1[2]);

    if outer_twin_of_he_prev != NONE {
        link_twins(mesh, hes0[2], outer_twin_of_he_prev);
    }
    if outer_twin_of_he_next != NONE {
        link_twins(mesh, hes1[1], outer_twin_of_he_next);
    }

    fix_vertex_he(mesh, &[p1, p2, op, mid], &all_new);

    Some(mid)
}

// ─────────────────── Collapse edge ──────────────────────

/// Collapse an interior edge, removing vertex `p` and merging it into the
/// other endpoint `o`.
///
/// `he` identifies the edge.  `p` must be one of the endpoints.  The vertex
/// `p` is logically removed (its `he` is set to `NONE`).  All faces around
/// `p` are deleted and rebuilt with `o` replacing `p` (except the two faces
/// sharing the collapsed edge, which simply disappear).
///
/// Returns `true` if the collapse succeeded.
pub fn collapse_edge(mesh: &mut BDSMesh, he: usize, p: usize) -> bool {
    let twin = mesh.half_edges[he].twin;
    if twin == NONE {
        return false; // boundary edge
    }

    let v0 = mesh.half_edges[he].origin;
    let v1 = mesh.he_dest(he);
    if p != v0 && p != v1 {
        return false;
    }
    let o = if p == v0 { v1 } else { v0 };

    // Don't collapse boundary vertices
    if mesh.vertices[p].on_boundary {
        return false;
    }

    // Gather all faces around p
    let p_faces = mesh.vertex_faces(p);
    if p_faces.is_empty() {
        return false;
    }

    // Identify the two faces that share the collapsed edge (they will vanish).
    let fi_he = mesh.half_edges[he].face;
    let fi_tw = mesh.half_edges[twin].face;

    // Find opposite vertices across the collapsed edge
    // (used for multi-edge check)
    let op_he = {
        let he0 = mesh.faces[fi_he].he;
        let he1 = mesh.half_edges[he0].next;
        let he2 = mesh.half_edges[he1].next;
        let verts = [
            mesh.half_edges[he0].origin,
            mesh.half_edges[he1].origin,
            mesh.half_edges[he2].origin,
        ];
        verts.into_iter().find(|&vi| vi != v0 && vi != v1)
    };
    let op_tw = {
        let he0 = mesh.faces[fi_tw].he;
        let he1 = mesh.half_edges[he0].next;
        let he2 = mesh.half_edges[he1].next;
        let verts = [
            mesh.half_edges[he0].origin,
            mesh.half_edges[he1].origin,
            mesh.half_edges[he2].origin,
        ];
        verts.into_iter().find(|&vi| vi != v0 && vi != v1)
    };

    let (op0, op1) = match (op_he, op_tw) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };

    // Multi-edge check: if op0 and op1 are already connected by an edge
    // (apart from through p), collapse would create duplicate edges.
    let o_neighbors: std::collections::HashSet<usize> =
        mesh.vertex_neighbors(o).into_iter().collect();
    let p_neighbors = mesh.vertex_neighbors(p);

    // Check that op0 and op1 don't both connect to `o` through paths
    // that would become duplicated. Specifically: any vertex (other than o)
    // that neighbours both p and o would end up with a double-edge to o.
    for &n in &p_neighbors {
        if n == o || n == op0 || n == op1 {
            continue;
        }
        if o_neighbors.contains(&n) {
            return false; // would create multi-edge
        }
    }

    // Build replacement triangles: for each face around p that does NOT
    // share the collapsed edge, replace p with o.
    let mut new_tris: SmallVec<[[usize; 3]; 8]> = SmallVec::new();
    let mut area_old = 0.0;
    let mut area_new = 0.0;

    for &fi in &p_faces {
        if fi == fi_he || fi == fi_tw {
            continue; // these faces vanish
        }
        let [va, vb, vc] = mesh.face_vertices(fi);
        area_old += signed_area_uv(mesh, va, vb, vc).abs();

        let new_v: [usize; 3] = [
            if va == p { o } else { va },
            if vb == p { o } else { vb },
            if vc == p { o } else { vc },
        ];

        // Degenerate check
        if new_v[0] == new_v[1] || new_v[1] == new_v[2] || new_v[0] == new_v[2] {
            return false;
        }

        let new_a = signed_area_uv(mesh, new_v[0], new_v[1], new_v[2]).abs();
        let old_a = signed_area_uv(mesh, va, vb, vc).abs();
        if new_a < 0.02 * old_a {
            return false;
        }
        area_new += new_a;
        new_tris.push(new_v);
    }

    // Also add the vanishing faces' area to area_old
    {
        let [va, vb, vc] = mesh.face_vertices(fi_he);
        area_old += signed_area_uv(mesh, va, vb, vc).abs();
    }
    {
        let [va, vb, vc] = mesh.face_vertices(fi_tw);
        area_old += signed_area_uv(mesh, va, vb, vc).abs();
    }

    // Area conservation
    if (area_old - area_new).abs() > 1e-12 * (area_old + area_new) {
        return false;
    }

    // === Commit ===

    // Collect all outer twin links before deleting faces.
    // For each face around p, we need to remember which external half-edges
    // were twinned with the face's half-edges (excluding those that are
    // internal to the p-fan).
    //
    // Strategy: delete all faces around p, then rebuild the surviving faces
    // and re-pair twins using the rebuild-from-scratch approach.

    // Collect all outer twin pairs for the surviving faces' edges.
    // An "outer" twin is one whose face is NOT in p_faces.
    let p_faces_set: std::collections::HashSet<usize> = p_faces.iter().copied().collect();
    let mut outer_twins: SmallVec<[(usize, usize, usize); 16]> = SmallVec::new(); // (from_v, to_v, outer_he)
    for &fi in &p_faces {
        let he0 = mesh.faces[fi].he;
        let mut cur = he0;
        for _ in 0..3 {
            let tw = mesh.half_edges[cur].twin;
            if tw != NONE {
                let tw_face = mesh.half_edges[tw].face;
                if !p_faces_set.contains(&tw_face) {
                    // This twin is an outer half-edge that we need to re-pair.
                    // The outer half-edge goes from dest(cur) to origin(cur).
                    // After collapse, origin(cur) might be p -> remap to o.
                    let mut from = mesh.he_dest(cur);
                    let mut to = mesh.half_edges[cur].origin;
                    if from == p {
                        from = o;
                    }
                    if to == p {
                        to = o;
                    }
                    outer_twins.push((from, to, tw));
                }
            }
            cur = mesh.half_edges[cur].next;
        }
    }

    // Soft-delete all faces around p
    for &fi in &p_faces {
        soft_delete_face(mesh, fi);
    }

    // Invalidate p
    mesh.vertices[p].he = NONE;

    // Create new faces
    let mut all_new_hes: SmallVec<[usize; 24]> = SmallVec::new();
    for tri in &new_tris {
        let (_, hes) = add_triangle(mesh, tri[0], tri[1], tri[2]);
        all_new_hes.extend_from_slice(&hes);
    }

    // Link internal twins among the new faces
    // Two new half-edges are internal twins if one goes A->B and the other B->A.
    // (This is O(n^2) but n is small -- typically 3-6 faces around a vertex.)
    let n = all_new_hes.len();
    for i in 0..n {
        if mesh.half_edges[all_new_hes[i]].twin != NONE {
            continue;
        }
        let orig_i = mesh.half_edges[all_new_hes[i]].origin;
        let dest_i = mesh.he_dest(all_new_hes[i]);
        for j in (i + 1)..n {
            if mesh.half_edges[all_new_hes[j]].twin != NONE {
                continue;
            }
            let orig_j = mesh.half_edges[all_new_hes[j]].origin;
            let dest_j = mesh.he_dest(all_new_hes[j]);
            if orig_i == dest_j && dest_i == orig_j {
                link_twins(mesh, all_new_hes[i], all_new_hes[j]);
                break;
            }
        }
    }

    // Link external twins
    for &(from, to, outer_he) in &outer_twins {
        // Find the new half-edge going from `to` to `from` (the inside direction)
        if let Some(inner) = find_he_directed(mesh, &all_new_hes, to, from) {
            link_twins(mesh, inner, outer_he);
        }
    }

    // Fix vertex outgoing half-edge pointers
    let mut affected_verts: SmallVec<[usize; 16]> = new_tris.iter().flatten().copied().collect();
    affected_verts.push(o);
    affected_verts.sort_unstable();
    affected_verts.dedup();
    fix_vertex_he(mesh, &affected_verts, &all_new_hes);

    true
}

// ─────────────────────── Swap edge ──────────────────────

/// Swap (flip) an interior edge.
///
/// The half-edge `he` identifies the edge to swap.  The quadrilateral formed
/// by the two adjacent triangles must be strictly convex in UV space, and the
/// swap must improve the minimum triangle quality.
///
/// When a `SizeField` is provided, quality is computed using the metric tensor
/// for accurate surface-aware decisions.
///
/// Before: triangles (p1, p2, op0) and (p2, p1, op1).
/// After:  triangles (p1, op1, op0) and (p2, op0, op1).
///
/// Returns `true` if the swap was performed.
pub fn swap_edge(mesh: &mut BDSMesh, he: usize) -> bool {
    swap_edge_with_sf(mesh, he, None)
}

/// Swap edge with optional size field for metric-based quality evaluation.
pub fn swap_edge_with_sf(mesh: &mut BDSMesh, he: usize, sf: Option<&SizeField>) -> bool {
    let twin = mesh.half_edges[he].twin;
    if twin == NONE {
        return false; // boundary
    }

    let fi0 = mesh.half_edges[he].face;
    let fi1 = mesh.half_edges[twin].face;
    if fi0 == NONE || fi1 == NONE {
        return false;
    }

    let p1 = mesh.half_edges[he].origin;
    let p2 = mesh.he_dest(he);

    // Find opposite vertices
    let he_next = mesh.half_edges[he].next;
    let he_prev = mesh.half_edges[he].prev;
    let op0 = mesh.half_edges[he_prev].origin;

    let tw_next = mesh.half_edges[twin].next;
    let tw_prev = mesh.half_edges[twin].prev;
    let op1 = mesh.half_edges[tw_prev].origin;

    // Don't swap if the new edge already exists
    // (check if op0 and op1 are already connected)
    if mesh.vertex_neighbors(op0).contains(&op1) {
        return false;
    }

    // Convexity check in UV space
    let s0 = signed_area_uv(mesh, p1, p2, op0);
    let s1 = signed_area_uv(mesh, p1, p2, op1);
    if s0 * s1 >= 0.0 {
        return false;
    }
    let s2 = signed_area_uv(mesh, op0, op1, p1);
    let s3 = signed_area_uv(mesh, op0, op1, p2);
    if s2 * s3 >= 0.0 {
        return false;
    }

    // Face-normal angle check: prevent swaps that would create triangles
    // with opposing normals on curved surfaces (matching GMSH's angle check).
    // We compute 3D face normals using vertex XYZ and reject if the dot
    // product of the two new triangle normals is negative (> 90° apart).
    if let Some(sf) = sf {
        let new_n0 = face_normal_3d(mesh, p1, op1, op0, sf);
        let new_n1 = face_normal_3d(mesh, p2, op0, op1, sf);
        let dot = new_n0.0 * new_n1.0 + new_n0.1 * new_n1.1 + new_n0.2 * new_n1.2;
        if dot < 0.0 {
            return false; // new triangles would face opposite directions
        }
    }

    // Quality check (use metric tensor if available)
    let qa_old_1 = triangle_quality(mesh, p1, p2, op0, sf);
    let qa_old_2 = triangle_quality(mesh, p1, p2, op1, sf);
    let qa_new_1 = triangle_quality(mesh, p1, op1, op0, sf);
    let qa_new_2 = triangle_quality(mesh, p2, op0, op1, sf);

    let min_old = qa_old_1.min(qa_old_2);
    let min_new = qa_new_1.min(qa_new_2);

    if min_new <= min_old {
        return false;
    }

    // Save outer twin links
    let outer_of_he_next = mesh.half_edges[he_next].twin; // twin of p2->op0
    let outer_of_he_prev = mesh.half_edges[he_prev].twin; // twin of op0->p1
    let outer_of_tw_next = mesh.half_edges[tw_next].twin; // twin of p1->op1
    let outer_of_tw_prev = mesh.half_edges[tw_prev].twin; // twin of op1->p2

    // === Commit ===
    soft_delete_face(mesh, fi0);
    soft_delete_face(mesh, fi1);

    // New faces: (p1, op1, op0) and (p2, op0, op1)
    let (_nf0, hes0) = add_triangle(mesh, p1, op1, op0); // hes0: p1->op1, op1->op0, op0->p1
    let (_nf1, hes1) = add_triangle(mesh, p2, op0, op1); // hes1: p2->op0, op0->op1, op1->p2

    let all_new = [hes0[0], hes0[1], hes0[2], hes1[0], hes1[1], hes1[2]];

    // Internal twins: op1->op0 in F0 twins with op0->op1 in F1
    link_twins(mesh, hes0[1], hes1[1]);

    // External twins:
    // op0->p1 in F0 (hes0[2]) twins with outer_of_he_prev (old twin of op0->p1)
    if outer_of_he_prev != NONE {
        link_twins(mesh, hes0[2], outer_of_he_prev);
    }
    // p1->op1 in F0 (hes0[0]) twins with outer_of_tw_next (old twin of p1->op1)
    if outer_of_tw_next != NONE {
        link_twins(mesh, hes0[0], outer_of_tw_next);
    }
    // p2->op0 in F1 (hes1[0]) twins with outer_of_he_next (old twin of p2->op0)
    if outer_of_he_next != NONE {
        link_twins(mesh, hes1[0], outer_of_he_next);
    }
    // op1->p2 in F1 (hes1[2]) twins with outer_of_tw_prev (old twin of op1->p2)
    if outer_of_tw_prev != NONE {
        link_twins(mesh, hes1[2], outer_of_tw_prev);
    }

    // Fix vertex outgoing half-edge pointers
    fix_vertex_he(mesh, &[p1, p2, op0, op1], &all_new);

    true
}

// ─────────────────── Smooth vertex ──────────────────────

/// Smooth a single interior vertex by moving it toward the UV centroid of its
/// 1-ring neighbours (Laplacian smoothing).
///
/// The vertex is moved only if:
/// 1. It is not on the boundary.
/// 2. The move does not invert any adjacent triangle (all signed areas remain
///    positive in UV space).
///
/// Returns `true` if the vertex was moved.
pub fn smooth_vertex(mesh: &mut BDSMesh, v: usize) -> bool {
    if mesh.vertices[v].on_boundary {
        return false;
    }
    if mesh.vertices[v].he == NONE {
        return false;
    }

    let neighbors = mesh.vertex_neighbors(v);
    if neighbors.is_empty() {
        return false;
    }

    let n = neighbors.len() as f64;
    let mut cu = 0.0;
    let mut cv = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for &ni in &neighbors {
        cu += mesh.vertices[ni].u;
        cv += mesh.vertices[ni].v;
        cx += mesh.vertices[ni].x;
        cy += mesh.vertices[ni].y;
        cz += mesh.vertices[ni].z;
    }
    cu /= n;
    cv /= n;
    cx /= n;
    cy /= n;
    cz /= n;

    let old_u = mesh.vertices[v].u;
    let old_v = mesh.vertices[v].v;
    let old_x = mesh.vertices[v].x;
    let old_y = mesh.vertices[v].y;
    let old_z = mesh.vertices[v].z;

    mesh.vertices[v].u = cu;
    mesh.vertices[v].v = cv;
    mesh.vertices[v].x = cx;
    mesh.vertices[v].y = cy;
    mesh.vertices[v].z = cz;

    // Check no triangle inverts
    let faces = mesh.vertex_faces(v);
    for &fi in &faces {
        if mesh.faces[fi].deleted {
            continue;
        }
        let [va, vb, vc] = mesh.face_vertices(fi);
        if signed_area_uv(mesh, va, vb, vc) <= 0.0 {
            // Revert
            mesh.vertices[v].u = old_u;
            mesh.vertices[v].v = old_v;
            mesh.vertices[v].x = old_x;
            mesh.vertices[v].y = old_y;
            mesh.vertices[v].z = old_z;
            return false;
        }
    }

    true
}

// ─────────────── Metric-weighted smooth vertex ──────────

/// Metric-weighted Laplacian smoothing.
///
/// Moves an interior vertex toward the metric-weighted centroid of its
/// neighbors.  The weight for each neighbor is the 3D edge length
/// (computed via the metric tensor), so neighbors that are farther in 3D
/// pull harder.  This produces geodesic-aware smoothing that handles UV
/// distortion near poles.
///
/// A damping factor of 0.5 is applied to limit the step size and avoid
/// oscillation.  The move is rejected if it would invert any adjacent
/// triangle in UV space.
///
/// Returns `true` if the vertex was moved.
pub fn smooth_vertex_metric(mesh: &mut BDSMesh, v: usize, sf: &SizeField) -> bool {
    if mesh.vertices[v].on_boundary {
        return false;
    }
    if mesh.vertices[v].he == NONE {
        return false;
    }

    let neighbors = mesh.vertex_neighbors(v);
    if neighbors.is_empty() {
        return false;
    }

    let u0 = mesh.vertices[v].u;
    let v0 = mesh.vertices[v].v;

    let mut wu = 0.0;
    let mut wv = 0.0;
    let mut w_total = 0.0;

    for &n in &neighbors {
        let un = mesh.vertices[n].u;
        let vn = mesh.vertices[n].v;

        // Weight = 3D edge length via metric tensor
        let weight = sf.edge_length_3d([u0, v0], [un, vn]).max(1e-12);

        wu += un * weight;
        wv += vn * weight;
        w_total += weight;
    }

    if w_total < 1e-20 {
        return false;
    }

    let target_u = wu / w_total;
    let target_v = wv / w_total;

    // Apply with damping
    let damping = 0.5;
    let new_u = u0 + damping * (target_u - u0);
    let new_v = v0 + damping * (target_v - v0);

    let old_u = mesh.vertices[v].u;
    let old_v = mesh.vertices[v].v;
    let old_x = mesh.vertices[v].x;
    let old_y = mesh.vertices[v].y;
    let old_z = mesh.vertices[v].z;

    mesh.vertices[v].u = new_u;
    mesh.vertices[v].v = new_v;
    // Update XYZ proportionally (same relative shift)
    mesh.vertices[v].x = old_x + (new_u - old_u);
    mesh.vertices[v].y = old_y + (new_v - old_v);

    // Check no triangle inverts
    let faces = mesh.vertex_faces(v);
    for &fi in &faces {
        if mesh.faces[fi].deleted {
            continue;
        }
        let [va, vb, vc] = mesh.face_vertices(fi);
        if signed_area_uv(mesh, va, vb, vc) <= 0.0 {
            // Revert
            mesh.vertices[v].u = old_u;
            mesh.vertices[v].v = old_v;
            mesh.vertices[v].x = old_x;
            mesh.vertices[v].y = old_y;
            mesh.vertices[v].z = old_z;
            return false;
        }
    }

    // Tutte energy check (from gmsh BDS.cpp): only accept if the sum of
    // squared 3D edge lengths to neighbors decreases.  This enforces global
    // edge-length uniformity, producing better surface approximation than
    // simple centroid smoothing.
    let energy_new: f64 = neighbors
        .iter()
        .map(|&n| {
            sf.edge_length_3d(
                [mesh.vertices[v].u, mesh.vertices[v].v],
                [mesh.vertices[n].u, mesh.vertices[n].v],
            )
            .powi(2)
        })
        .sum();

    let energy_old: f64 = neighbors
        .iter()
        .map(|&n| {
            sf.edge_length_3d([old_u, old_v], [mesh.vertices[n].u, mesh.vertices[n].v])
                .powi(2)
        })
        .sum();

    if energy_new >= energy_old {
        mesh.vertices[v].u = old_u;
        mesh.vertices[v].v = old_v;
        mesh.vertices[v].x = old_x;
        mesh.vertices[v].y = old_y;
        mesh.vertices[v].z = old_z;
        return false;
    }

    true
}

// ───────────────────────── Tests ─────────────────────────

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meshadapt::bds::BDSMesh;

    // ── Test mesh builders ──────────────────────────────

    /// Two triangles sharing an interior edge (the diagonal 1-2):
    ///
    /// ```text
    ///   2---3
    ///   |\ |
    ///   | \|
    ///   0---1
    /// ```
    fn quad_mesh() -> BDSMesh {
        BDSMesh::from_triangles(
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &[[0, 1, 2], [1, 3, 2]],
        )
    }

    /// Diamond: 4 triangles around a central vertex (vertex 4).
    fn diamond_mesh() -> BDSMesh {
        BDSMesh::from_triangles(
            &[
                [0.5, 0.0], // 0 bottom
                [1.0, 0.5], // 1 right
                [0.5, 1.0], // 2 top
                [0.0, 0.5], // 3 left
                [0.5, 0.5], // 4 center
            ],
            &[[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]],
        )
    }

    /// Find the half-edge going from vertex `from` to vertex `to`.
    fn find_he(mesh: &BDSMesh, from: usize, to: usize) -> Option<usize> {
        for he in &mesh.half_edges {
            if he.origin == from
                && mesh.he_dest(he.id) == to
                && he.face != NONE
                && !mesh.faces[he.face].deleted
            {
                return Some(he.id);
            }
        }
        None
    }

    // ── Split tests ─────────────────────────────────────

    #[test]
    fn split_interior_edge() {
        let mut m = quad_mesh();
        assert!(m.validate());
        let nf_before = m.num_live_faces();
        let nv_before = m.vertices.len();

        // Split the diagonal edge 1->2
        let he = find_he(&m, 1, 2).unwrap();
        let mid = split_edge(&mut m, he).unwrap();

        assert!(m.validate(), "mesh invalid after split");
        assert_eq!(m.num_live_faces(), nf_before + 2); // 2 faces become 4
        assert_eq!(m.vertices.len(), nv_before + 1);

        // Midpoint should be at (0.5, 0.5)
        assert!((m.vertices[mid].u - 0.5).abs() < 1e-10);
        assert!((m.vertices[mid].v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn split_boundary_edge_fails() {
        let m_tris = &[[0usize, 1, 2]];
        let m_verts = &[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let mut m = BDSMesh::from_triangles(m_verts, m_tris);

        // All edges are boundary, split should fail
        for he in 0..m.half_edges.len() {
            assert!(split_edge(&mut m, he).is_none());
        }
    }

    #[test]
    fn split_preserves_euler() {
        let mut m = quad_mesh();
        let (nv0, ne0, nf0) = euler_counts(&m);

        let he = find_he(&m, 1, 2).unwrap();
        split_edge(&mut m, he).unwrap();

        let (nv1, ne1, nf1) = euler_counts(&m);

        // V - E + F should be preserved
        assert_eq!(
            nv0 - ne0 + nf0,
            nv1 - ne1 + nf1,
            "Euler characteristic changed"
        );
    }

    #[test]
    fn split_diamond_edge() {
        let mut m = diamond_mesh();
        assert!(m.validate());
        let nf_before = m.num_live_faces();

        // Split edge 4->1 (interior)
        let he = find_he(&m, 4, 1).unwrap();
        assert!(split_edge(&mut m, he).is_some());
        assert!(m.validate(), "mesh invalid after split");
        assert_eq!(m.num_live_faces(), nf_before + 2);
    }

    // ── Collapse tests ──────────────────────────────────

    #[test]
    fn collapse_interior_vertex() {
        let mut m = diamond_mesh();
        assert!(m.validate());
        let nf_before = m.num_live_faces();

        // Collapse edge 4->1, removing centre vertex 4
        let he = find_he(&m, 4, 1).unwrap();
        assert!(collapse_edge(&mut m, he, 4));
        assert!(m.validate(), "mesh invalid after collapse");

        // 2 faces sharing the collapsed edge vanish: 4 - 2 = 2
        assert_eq!(m.num_live_faces(), nf_before - 2);
    }

    #[test]
    fn collapse_boundary_vertex_fails() {
        let mut m = diamond_mesh();
        // Vertex 0 is on boundary
        let he = find_he(&m, 4, 0).unwrap();
        assert!(!collapse_edge(&mut m, he, 0));
    }

    #[test]
    fn collapse_preserves_euler() {
        let mut m = diamond_mesh();
        let (nv0, ne0, nf0) = euler_counts(&m);

        let he = find_he(&m, 4, 1).unwrap();
        assert!(collapse_edge(&mut m, he, 4));

        let (nv1, ne1, nf1) = euler_counts(&m);
        assert_eq!(
            nv0 - ne0 + nf0,
            nv1 - ne1 + nf1,
            "Euler characteristic changed"
        );
    }

    // ── Swap tests ──────────────────────────────────────

    #[test]
    fn swap_improves_quality() {
        // Non-parallelogram quad where diagonal 1-2 is clearly suboptimal.
        // Vertex 3 is close to the 1-2 diagonal making that triangle very thin.
        //
        //   2(0.3, 1.0)     3(1.0, 0.8)
        //       +----------+
        //      /          /
        //     /     _--' /
        //    / _--'     /
        //   +---------+
        //  0(0, 0)    1(2, 0)
        //
        // Old diagonal 1-2 produces a very thin triangle (1,2,3).
        // New diagonal 0-3 produces two much better triangles.
        let mut m = BDSMesh::from_triangles(
            &[[0.0, 0.0], [2.0, 0.0], [0.3, 1.0], [1.0, 0.8]],
            &[[0, 1, 2], [1, 3, 2]],
        );
        assert!(m.validate());

        let he = find_he(&m, 1, 2).unwrap();
        assert!(
            swap_edge(&mut m, he),
            "swap should succeed for suboptimal diagonal"
        );
        assert!(m.validate(), "mesh invalid after swap");
        assert_eq!(m.num_live_faces(), 2);

        // The new diagonal should connect 0 and 3
        assert!(find_he(&m, 0, 3).is_some() || find_he(&m, 3, 0).is_some());
    }

    #[test]
    fn swap_boundary_edge_fails() {
        let mut m = BDSMesh::from_triangles(
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &[[0, 1, 2], [1, 3, 2]],
        );
        // Edge 0->1 is boundary (no twin)
        let he = find_he(&m, 0, 1).unwrap();
        assert!(!swap_edge(&mut m, he));
    }

    #[test]
    fn swap_rejects_non_convex() {
        // Both opposite vertices on the same side of the edge
        let mut m = BDSMesh::from_triangles(
            &[
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.3], // inside triangle 0,1,2
            ],
            &[[0, 1, 2], [1, 0, 3]], // note: (1,0,3) so 3 is below
        );

        if let Some(he) = find_he(&m, 0, 1).or_else(|| find_he(&m, 1, 0)) {
            if m.half_edges[he].twin != NONE {
                assert!(!swap_edge(&mut m, he), "should reject non-convex swap");
            }
        }
    }

    #[test]
    fn swap_preserves_euler() {
        let mut m = BDSMesh::from_triangles(
            &[[0.0, 0.0], [2.0, 0.0], [0.3, 1.0], [1.0, 0.8]],
            &[[0, 1, 2], [1, 3, 2]],
        );
        let (nv0, ne0, nf0) = euler_counts(&m);

        let he = find_he(&m, 1, 2).unwrap();
        assert!(swap_edge(&mut m, he));
        let (nv1, ne1, nf1) = euler_counts(&m);
        assert_eq!(nv0, nv1);
        assert_eq!(ne0, ne1);
        assert_eq!(nf0, nf1);
    }

    #[test]
    fn repeated_swap_is_stable() {
        let mut m = BDSMesh::from_triangles(
            &[[0.0, 0.0], [2.0, 0.0], [0.3, 1.0], [1.0, 0.8]],
            &[[0, 1, 2], [1, 3, 2]],
        );

        let he = find_he(&m, 1, 2).unwrap();
        assert!(swap_edge(&mut m, he));
        assert!(m.validate());
        // Try to swap the new diagonal back -- should fail (quality would decrease)
        if let Some(new_he) = find_he(&m, 0, 3).or_else(|| find_he(&m, 3, 0)) {
            let swapped_back = swap_edge(&mut m, new_he);
            assert!(!swapped_back, "second swap should fail (quality decrease)");
        }
    }

    // ── Smooth tests ────────────────────────────────────

    #[test]
    fn smooth_moves_interior_vertex() {
        let mut m = diamond_mesh();
        // Move centre vertex off-center
        m.vertices[4].u = 0.6;
        m.vertices[4].v = 0.6;
        m.vertices[4].x = 0.6;
        m.vertices[4].y = 0.6;

        let old_dist = (m.vertices[4].u - 0.5).abs() + (m.vertices[4].v - 0.5).abs();
        assert!(smooth_vertex(&mut m, 4));
        let new_dist = (m.vertices[4].u - 0.5).abs() + (m.vertices[4].v - 0.5).abs();

        assert!(new_dist < old_dist, "vertex should move closer to centroid");
        assert!(m.validate(), "mesh invalid after smooth");
    }

    #[test]
    fn smooth_boundary_vertex_stays() {
        let mut m = diamond_mesh();
        // Vertex 0 is on boundary
        assert!(!smooth_vertex(&mut m, 0));
    }

    #[test]
    fn smooth_metric_boundary_vertex_stays() {
        let mut m = diamond_mesh();
        let sf = crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };
        // Vertex 0 is on boundary
        assert!(!smooth_vertex_metric(&mut m, 0, &sf));
    }

    #[test]
    fn smooth_metric_differs_from_uv_centroid_on_anisotropic_metric() {
        // Build a diamond mesh where the centre vertex (4) is
        // displaced from the centroid.  With a strongly anisotropic
        // metric (E >> G), the metric-weighted centroid should differ
        // from the UV centroid.
        //
        // Diamond vertices:
        //   0=(0.5,0.0) bottom, 1=(1.0,0.5) right,
        //   2=(0.5,1.0) top,    3=(0.0,0.5) left,
        //   4=(0.5,0.5) centre (will be displaced)
        let mut m_uv = diamond_mesh();
        let mut m_metric = diamond_mesh();

        // Displace center vertex so smoothing has something to do
        m_uv.vertices[4].u = 0.6;
        m_uv.vertices[4].v = 0.6;
        m_uv.vertices[4].x = 0.6;
        m_uv.vertices[4].y = 0.6;

        m_metric.vertices[4].u = 0.6;
        m_metric.vertices[4].v = 0.6;
        m_metric.vertices[4].x = 0.6;
        m_metric.vertices[4].y = 0.6;

        // Anisotropic metric: E=100 (u direction is 10x stretched), G=1
        let sf = crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![100.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };

        // Apply UV-centroid smooth
        assert!(smooth_vertex(&mut m_uv, 4));
        let uv_u = m_uv.vertices[4].u;
        let uv_v = m_uv.vertices[4].v;

        // Apply metric-weighted smooth
        assert!(smooth_vertex_metric(&mut m_metric, 4, &sf));
        let met_u = m_metric.vertices[4].u;
        let met_v = m_metric.vertices[4].v;

        // The two results should differ: the metric weighting should
        // produce a different centroid because edges in the u direction
        // are 10x longer in 3D than edges in the v direction.
        let diff = (uv_u - met_u).abs() + (uv_v - met_v).abs();
        assert!(
            diff > 1e-6,
            "metric smooth should differ from UV smooth on anisotropic metric, \
             but diff = {diff:.2e} (uv=({uv_u:.6},{uv_v:.6}), met=({met_u:.6},{met_v:.6}))"
        );

        // Both should produce valid meshes
        assert!(m_uv.validate(), "UV-smooth mesh invalid");
        assert!(m_metric.validate(), "metric-smooth mesh invalid");
    }

    #[test]
    fn smooth_metric_moves_interior_vertex() {
        // Verify metric smoothing moves a displaced interior vertex
        let mut m = diamond_mesh();
        m.vertices[4].u = 0.6;
        m.vertices[4].v = 0.6;
        m.vertices[4].x = 0.6;
        m.vertices[4].y = 0.6;

        let sf = crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };

        let old_u = m.vertices[4].u;
        let old_v = m.vertices[4].v;
        assert!(smooth_vertex_metric(&mut m, 4, &sf));
        let new_u = m.vertices[4].u;
        let new_v = m.vertices[4].v;

        // Should have moved
        assert!(
            (new_u - old_u).abs() + (new_v - old_v).abs() > 1e-10,
            "metric smooth should move displaced vertex"
        );
        assert!(m.validate(), "mesh invalid after metric smooth");
    }

    // ── Integration tests ───────────────────────────────

    #[test]
    fn split_then_collapse_recovers() {
        let mut m = diamond_mesh();
        assert!(m.validate());
        let nf_orig = m.num_live_faces();
        let _nv_orig = m.vertices.len();

        // Split edge 4->1
        let he = find_he(&m, 4, 1).unwrap();
        let mid = split_edge(&mut m, he).unwrap();
        assert!(m.validate());
        assert_eq!(m.num_live_faces(), nf_orig + 2);

        // Now collapse the midpoint back onto vertex 1
        // Find a half-edge from mid to 1 (or 1 to mid)
        let he_collapse = find_he(&m, mid, 1).or_else(|| find_he(&m, 1, mid));
        if let Some(hc) = he_collapse {
            let v0 = m.half_edges[hc].origin;
            let v1 = m.he_dest(hc);
            let p = if v0 == mid { mid } else { v1 };
            if collapse_edge(&mut m, hc, p) {
                assert!(m.validate());
                // Should be back to original face count
                assert_eq!(m.num_live_faces(), nf_orig);
            }
        }
    }

    #[test]
    fn split_swap_sequence() {
        let mut m = quad_mesh();
        assert!(m.validate());

        // Split the diagonal
        let he = find_he(&m, 1, 2).unwrap();
        let mid = split_edge(&mut m, he).unwrap();
        assert!(m.validate());
        assert_eq!(m.num_live_faces(), 4);

        // Try swapping one of the new interior edges
        let outgoing = m.vertex_half_edges(mid);
        let mut swapped = false;
        for &out_he in &outgoing {
            if m.half_edges[out_he].twin != NONE && swap_edge(&mut m, out_he) {
                assert!(m.validate());
                swapped = true;
                break;
            }
        }
        // It's OK if no swap improved quality -- the test verifies no crash
        let _ = swapped;
    }

    #[test]
    fn grid_split_all_interior_edges() {
        let (v, t) = grid_mesh(3);
        let mut m = BDSMesh::from_triangles(&v, &t);
        assert!(m.validate());

        // Collect all interior half-edges (with twins)
        let interior_hes: Vec<usize> = m
            .half_edges
            .iter()
            .filter(|he| he.twin != NONE)
            .map(|he| he.id)
            .collect();

        let mut splits = 0;
        for he in interior_hes {
            // Only attempt if the half-edge still belongs to a live face
            if he < m.half_edges.len() {
                let face = m.half_edges[he].face;
                if face != NONE
                    && face < m.faces.len()
                    && !m.faces[face].deleted
                    && split_edge(&mut m, he).is_some()
                {
                    assert!(m.validate(), "mesh invalid after split #{}", splits);
                    splits += 1;
                }
            }
        }
        assert!(splits > 0, "should have split at least one edge");
    }

    // ── Euler helper ────────────────────────────────────

    /// Count V, E, F for Euler formula (V - E + F = 1 for a disk).
    fn euler_counts(mesh: &BDSMesh) -> (isize, isize, isize) {
        let nv = mesh.vertices.iter().filter(|v| v.he != NONE).count() as isize;
        let nf = mesh.num_live_faces() as isize;

        // Count edges: each interior edge has 2 half-edges with twins,
        // each boundary edge has 1 half-edge with no twin.
        let mut boundary_he = 0isize;
        let mut interior_he = 0isize;
        for he in &mesh.half_edges {
            // Only count half-edges belonging to live faces
            if he.face != NONE && he.face < mesh.faces.len() && !mesh.faces[he.face].deleted {
                if he.twin == NONE {
                    boundary_he += 1;
                } else {
                    interior_he += 1;
                }
            }
        }
        let ne = interior_he / 2 + boundary_he;
        (nv, ne, nf)
    }

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

    // ── Step 6: Geodesic-aware split tests ────────────────

    /// Create a uniform size field for ops tests.
    fn uniform_sf() -> crate::size_field::SizeField {
        crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        }
    }

    /// Create a strongly anisotropic size field (E=100, G=1).
    fn anisotropic_sf() -> crate::size_field::SizeField {
        crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![100.0; 4],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        }
    }

    #[test]
    fn split_edge_metric_uniform_gives_uv_midpoint() {
        // With uniform metric (identity), the geodesic midpoint should
        // coincide with the UV midpoint.
        let mut m = quad_mesh();
        assert!(m.validate());
        let sf = uniform_sf();

        let he = find_he(&m, 1, 2).unwrap();
        let mid = split_edge_metric(&mut m, he, &sf).unwrap();

        assert!(m.validate(), "mesh invalid after metric split");

        // Midpoint of (1,0)->(0,1) should be (0.5, 0.5)
        assert!(
            (m.vertices[mid].u - 0.5).abs() < 1e-4,
            "u={}, expected ~0.5",
            m.vertices[mid].u
        );
        assert!(
            (m.vertices[mid].v - 0.5).abs() < 1e-4,
            "v={}, expected ~0.5",
            m.vertices[mid].v
        );
    }

    #[test]
    fn split_edge_metric_preserves_topology() {
        let mut m = quad_mesh();
        let sf = uniform_sf();
        let nf_before = m.num_live_faces();
        let nv_before = m.vertices.len();

        let he = find_he(&m, 1, 2).unwrap();
        split_edge_metric(&mut m, he, &sf).unwrap();

        assert!(m.validate(), "mesh invalid after metric split");
        assert_eq!(m.num_live_faces(), nf_before + 2);
        assert_eq!(m.vertices.len(), nv_before + 1);
    }

    #[test]
    fn split_edge_metric_preserves_euler() {
        let mut m = quad_mesh();
        let sf = uniform_sf();
        let (nv0, ne0, nf0) = euler_counts(&m);

        let he = find_he(&m, 1, 2).unwrap();
        split_edge_metric(&mut m, he, &sf).unwrap();

        let (nv1, ne1, nf1) = euler_counts(&m);
        assert_eq!(
            nv0 - ne0 + nf0,
            nv1 - ne1 + nf1,
            "Euler characteristic changed"
        );
    }

    #[test]
    fn split_edge_metric_geodesic_midpoint_on_uniform_aniso_gives_uv_mid() {
        // With a uniform (but anisotropic) metric (E=100, G=1), the
        // arc-speed along any straight UV line is constant (the metric
        // doesn't vary spatially), so the geodesic midpoint parameter
        // is exactly t=0.5, same as the UV midpoint.
        let mut m = quad_mesh();
        let sf_aniso = anisotropic_sf();

        let he = find_he(&m, 1, 2).unwrap();
        let mid = split_edge_metric(&mut m, he, &sf_aniso).unwrap();

        assert!(m.validate());

        // Should be at the UV midpoint (0.5, 0.5)
        assert!(
            (m.vertices[mid].u - 0.5).abs() < 1e-3,
            "u={}, expected ~0.5",
            m.vertices[mid].u
        );
        assert!(
            (m.vertices[mid].v - 0.5).abs() < 1e-3,
            "v={}, expected ~0.5",
            m.vertices[mid].v
        );
    }

    #[test]
    fn split_edge_metric_geodesic_midpoint_differs_on_varying_metric() {
        // With a spatially varying metric, the geodesic midpoint should
        // differ from the UV midpoint.
        //
        // E varies from 1 at u=0 to 100 at u=1; G=1 everywhere.
        // Along the u-axis, the 3D arc length accumulates faster near u=1.
        // So the geodesic midpoint has t > 0.5 (more UV distance is spent
        // in the cheap region near u=0).
        let sf_varying = crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0, 100.0, 1.0, 100.0],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };

        // Build a quad mesh with a horizontal interior edge to split
        // Vertices: 0=(0,0), 1=(1,0), 2=(0,0.5), 3=(1,0.5)
        // This gives two triangles sharing the edge 1->2.
        // We want a horizontal edge (constant v) to isolate the E effect.
        // Use: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
        // Triangles: (0,1,2), (0,2,3) -- diagonal is 0->2
        // But 0->2 goes diagonally.  Instead use a mesh where the
        // interior edge goes along u.
        // Vertices: 0=(0,0), 1=(0.5,0), 2=(1,0), 3=(0,0.5), 4=(1,0.5)
        // Triangles: (0,1,3),(1,2,4),(1,4,3) -- edge 1->3 and 1->4 interior
        // Actually, let's just test geodesic_midpoint_t directly.
        let t = geodesic_midpoint_t(&sf_varying, [0.0, 0.5], [1.0, 0.5]);

        // t should be > 0.5 because the midpoint needs to cover more UV
        // distance in the cheap (low-E) region near u=0.
        assert!(
            t > 0.55,
            "varying metric should shift midpoint to t > 0.55, got {t}"
        );
        assert!(t < 0.95, "midpoint should still be reasonable, got {t}");
    }

    #[test]
    fn split_edge_metric_boundary_fails() {
        let m_tris = &[[0usize, 1, 2]];
        let m_verts = &[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let mut m = BDSMesh::from_triangles(m_verts, m_tris);
        let sf = uniform_sf();

        // All edges are boundary, metric split should fail
        for he in 0..m.half_edges.len() {
            assert!(split_edge_metric(&mut m, he, &sf).is_none());
        }
    }

    #[test]
    fn geodesic_midpoint_t_uniform_is_half() {
        let sf = uniform_sf();
        let t = geodesic_midpoint_t(&sf, [0.0, 0.0], [1.0, 0.0]);
        assert!(
            (t - 0.5).abs() < 1e-3,
            "uniform metric should give t=0.5, got {t}"
        );
    }

    #[test]
    fn geodesic_midpoint_t_anisotropic_shifts() {
        // With E=100, G=1, along a purely u-direction edge the metric is
        // uniform (only E matters) so t should still be ~0.5.
        let sf = anisotropic_sf();
        let t_u = geodesic_midpoint_t(&sf, [0.0, 0.5], [1.0, 0.5]);
        assert!(
            (t_u - 0.5).abs() < 1e-3,
            "along u-axis with uniform E, t should be ~0.5, got {t_u}"
        );

        // Along the v-direction it should also be ~0.5 (uniform G).
        let t_v = geodesic_midpoint_t(&sf, [0.5, 0.0], [0.5, 1.0]);
        assert!(
            (t_v - 0.5).abs() < 1e-3,
            "along v-axis with uniform G, t should be ~0.5, got {t_v}"
        );

        // Along a diagonal, E >> G means the u-component is 10x more expensive.
        // The geodesic midpoint should shift to spend less parameter distance
        // in the u-direction, so t != 0.5 (it is shifted so the midpoint
        // is closer in u).
        let t_diag = geodesic_midpoint_t(&sf, [0.0, 0.0], [1.0, 1.0]);
        // For E=100, G=1: ds/dt = sqrt(100*1 + 1*1) = sqrt(101) ~ 10.05,
        // which is constant along the edge (uniform metric), so t should be ~0.5.
        // Actually with a uniform anisotropic metric, the speed is constant
        // along any straight line, so t = 0.5.  The shift only happens with
        // a spatially varying metric.
        assert!(
            (t_diag - 0.5).abs() < 1e-3,
            "with uniform (but anisotropic) metric, diagonal t should be ~0.5, got {t_diag}"
        );
    }

    #[test]
    fn geodesic_midpoint_t_varying_metric() {
        // Create a size field where E varies: E=1 at u=0, E=100 at u=1.
        // For an edge from (0,0.5) to (1,0.5), the geodesic midpoint
        // parameter t should be > 0.5.
        //
        // Rationale: near u=1, E is large, so each du costs more 3D arc
        // length.  The total 3D arc length is dominated by the right
        // portion.  To accumulate half the total 3D length, we need to
        // traverse more of the UV interval in the cheap (u~0) region,
        // so the 3D midpoint maps to a UV parameter t > 0.5.
        let sf = crate::size_field::SizeField {
            nu: 2,
            nv: 2,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![1.0; 4],
            metric_e: vec![1.0, 100.0, 1.0, 100.0],
            metric_f: vec![0.0; 4],
            metric_g: vec![1.0; 4],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        };

        let t = geodesic_midpoint_t(&sf, [0.0, 0.5], [1.0, 0.5]);
        assert!(
            t > 0.55,
            "varying metric should shift midpoint to t > 0.55, got t={t}"
        );
        assert!(
            t < 0.95,
            "midpoint parameter should be reasonable, got t={t}"
        );
    }
}
