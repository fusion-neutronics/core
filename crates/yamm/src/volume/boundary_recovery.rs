#![allow(clippy::needless_range_loop)]
use super::delaunay3d::{self, Delaunay3D, Tet, INFINITE};
/// Boundary recovery for Delaunay tetrahedralizations.
///
/// After Bowyer-Watson insertion of all surface vertices, the resulting
/// Delaunay tet mesh may not contain all boundary edges and faces from
/// the input surface mesh. This module recovers missing boundary edges
/// via local flip operations (2-to-3 and 3-to-2 flips).
///
/// Reference: Si, H. "TetGen, a Delaunay-Based Quality Tetrahedral Mesh
/// Generator", ACM TOMS 2015. Section 4: Boundary Recovery.
use super::dethash::{HashMap, HashSet};
use super::predicates3d::{in_sphere, orient_3d, orient_3d_sos};

// ── Recovery instrumentation (issue #37) ──
// Counts how often the O(#tets) crossing-scan FALLBACKS actually fire during
// segment recovery (the primary path is `finddirection`, O(degree)). Reported
// under YAMM_CARVE_DBG; the counter increments are unconditional but negligible.
// These confirmed the fallbacks fire ~0× on the casing - i.e. they are NOT the
// O(#tets) bottleneck (that was `lawson_restore`, since fixed). Kept for the
// ongoing near-tangent / perf work.
static FALLBACK_FSCP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FALLBACK_FSCP_INCL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FINDDIR_BOUNDARY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// High-water mark of marches consumed by a SINGLE top-level flip recovery.
/// Diagnostic only - it is what `FLIP_RECOVER_BUDGET` is calibrated against, so
/// the budget can be set from measurement rather than taste.
static FLIP_BUDGET_HIGH_WATER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Check if a tet contains the INFINITE (hull) vertex.
fn is_hull_tet(t: &Tet) -> bool {
    t.verts.contains(&INFINITE)
}

// ── Per-flip volume audit (issue #37, YAMM_FLIPVOL) ──
// Every flip replaces a set of tets with another set covering the SAME region,
// so the summed FINITE tet volume must be unchanged. A nonzero Δ identifies the
// primitive (and whether hull tets were involved) that leaks cover - the
// segment-pass +0.44 overlap the volume gate rejects. Diagnostic, env-gated.
fn flipvol_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("YAMM_FLIPVOL").is_ok())
}
/// (summed |finite volume|, any-hull-involved) over live tets in `idxs`.
fn flipvol_of(tets: &Delaunay3D, idxs: &[usize]) -> (f64, bool) {
    let mut s = 0.0;
    let mut hull = false;
    for &i in idxs {
        if !tets.is_live(i) {
            continue;
        }
        if is_hull_tet(&tets.tets[i]) {
            hull = true;
            continue;
        }
        let v = tets.tets[i].verts;
        s += orient_3d(
            tets.vertices[v[0]],
            tets.vertices[v[1]],
            tets.vertices[v[2]],
            tets.vertices[v[3]],
        )
        .abs()
            / 6.0;
    }
    (s, hull)
}
fn flipvol_report(prim: &str, old: (f64, bool), tets: &Delaunay3D, new_idxs: &[usize]) {
    let new = flipvol_of(tets, new_idxs);
    let d = new.0 - old.0;
    if d.abs() > 1e-12 {
        eprintln!(
            "    [FLIPVOL] {prim} Δ={d:+.9} hull_old={} hull_new={}",
            old.1, new.1
        );
    }
}

/// Summed |finite volume| of CANDIDATE tets given as vertex quadruples
/// (pre-commit - nothing allocated yet). Hull candidates (INFINITE) skipped.
fn flipvol_of_candidates(tets: &Delaunay3D, cands: &[[usize; 4]]) -> f64 {
    let mut s = 0.0;
    for v in cands {
        if v.contains(&INFINITE) {
            continue;
        }
        s += orient_3d(
            tets.vertices[v[0]],
            tets.vertices[v[1]],
            tets.vertices[v[2]],
            tets.vertices[v[3]],
        )
        .abs()
            / 6.0;
    }
    s
}

/// TRANSACTIONAL VOLUME GUARD (issue #37): a flip replaces a set of tets with
/// another set that must tile the SAME region, so the summed finite volume is
/// invariant. The orientation/convexity predicates in the primitives are
/// necessary but NOT sufficient on reflex / near-degenerate configurations -
/// flips were committing with overlaps (flip_ring4, Δ>0) and gaps
/// (flip_3_to_2, Δ<0) that accumulated to exactly the carve overshoot the
/// volume gate rejects. Volume equality is the ground-truth tiling condition;
/// a flip that fails it is geometrically invalid and must be a no-op (the
/// caller falls through to other strategies / Steiner refinement, which are
/// exact). Tolerance is relative (exact-arithmetic-clean flips match to ~1
/// ulp; real leaks are many orders larger).
fn flip_volume_ok(old: f64, new: f64) -> bool {
    (new - old).abs() <= (old.max(new) * 1e-9).max(1e-12)
}

/// An active geometric constraint that flips must not destroy.
///
/// During boundary recovery we flip the mesh to introduce a wanted edge or
/// face. A flip is only *eligible* if the new element it creates does not
/// cross the constraint in its interior - otherwise it would push the mesh
/// further from the boundary it is trying to recover (or worse, sever a
/// boundary feature that is already present).
///
/// Vertex indices are indices into `Delaunay3D::vertices`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constraint {
    /// No constraint - every (otherwise valid) flip is eligible.
    None,
    /// A boundary edge (u, v) that must not be crossed in its interior.
    Edge(usize, usize),
    /// A boundary face {a, b, c} that must not be crossed in its interior.
    Face([usize; 3]),
}

/// Exact test: do the open segments seg1=(verts[s1[0]], verts[s1[1]]) and
/// seg2=(verts[s2[0]], verts[s2[1]]) cross in their interior?
///
/// Returns true only if the two segments are coplanar, properly intersect
/// (not merely touch at a shared endpoint), and the crossing point lies
/// strictly inside *both* segments. Endpoints shared between the segments are
/// not crossings. Uses exact `orient_3d` throughout.
fn segments_cross_interior(verts: &[[f64; 3]], s1: [usize; 2], s2: [usize; 2]) -> bool {
    let (ia, ib) = (s1[0], s1[1]);
    let (ic, id) = (s2[0], s2[1]);
    // Shared endpoints never count as an interior crossing.
    if ia == ic || ia == id || ib == ic || ib == id {
        return false;
    }
    let a = verts[ia];
    let b = verts[ib];
    let c = verts[ic];
    let d = verts[id];
    // Must be coplanar for the segments to cross at a point.
    if orient_3d(a, b, c, d) != 0.0 {
        return false;
    }
    // In the common plane, (c,d) crosses (a,b) iff c and d straddle line ab
    // AND a and b straddle line cd. We test straddling with orient_3d using
    // an auxiliary off-plane apex; the sign of orient_3d(a, b, x, apex)
    // reports which side of line ab the point x lies on within the plane.
    let n = {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ]
    };
    let apex = [a[0] + n[0], a[1] + n[1], a[2] + n[2]];
    let s_c = orient_3d(a, b, c, apex);
    let s_d = orient_3d(a, b, d, apex);
    // c and d must be on strictly opposite sides of line ab.
    if s_c == 0.0 || s_d == 0.0 || (s_c > 0.0) == (s_d > 0.0) {
        return false;
    }
    let s_a = orient_3d(c, d, a, apex);
    let s_b = orient_3d(c, d, b, apex);
    if s_a == 0.0 || s_b == 0.0 || (s_a > 0.0) == (s_b > 0.0) {
        return false;
    }
    true
}

/// Does the edge (eu, ev) cross the active `constraint` in its interior?
///
/// Used as the "new element" test for a 2-3 flip (the new shared edge).
fn edge_crosses_constraint(
    tets: &Delaunay3D,
    eu: usize,
    ev: usize,
    constraint: &Constraint,
) -> bool {
    match *constraint {
        Constraint::None => false,
        Constraint::Edge(cu, cv) => segments_cross_interior(&tets.vertices, [eu, ev], [cu, cv]),
        Constraint::Face([ca, cb, cc]) => {
            // The new edge crosses the constraint face if the (open) segment
            // pierces the (open) triangle interior. Skip if the edge shares a
            // vertex with the face (touching is allowed).
            if eu == ca || eu == cb || eu == cc || ev == ca || ev == cb || ev == cc {
                return false;
            }
            segment_intersects_triangle(
                tets.vertices[eu],
                tets.vertices[ev],
                tets.vertices[ca],
                tets.vertices[cb],
                tets.vertices[cc],
            )
        }
    }
}

/// Does the face (fa, fb, fc) cross the active `constraint` in its interior?
///
/// Used as the "new element" test for a 3-2 flip (the new shared face). A
/// triangle crosses a constraint *edge* if that edge pierces the triangle
/// interior; it crosses a constraint *edge segment* if one of the triangle's
/// own edges crosses the constraint edge. A triangle crossing a constraint
/// *face* is handled by checking edge-vs-face for each triangle edge.
fn face_crosses_constraint(
    tets: &Delaunay3D,
    fa: usize,
    fb: usize,
    fc: usize,
    constraint: &Constraint,
) -> bool {
    match *constraint {
        Constraint::None => false,
        Constraint::Edge(cu, cv) => {
            // 1) The constraint edge pierces the new triangle's interior.
            if cu != fa
                && cu != fb
                && cu != fc
                && cv != fa
                && cv != fb
                && cv != fc
                && segment_intersects_triangle(
                    tets.vertices[cu],
                    tets.vertices[cv],
                    tets.vertices[fa],
                    tets.vertices[fb],
                    tets.vertices[fc],
                )
            {
                return true;
            }
            // 2) One of the triangle's edges crosses the constraint edge in
            //    its interior (coplanar diagonal-type crossing).
            for &(eu, ev) in &[(fa, fb), (fb, fc), (fc, fa)] {
                if segments_cross_interior(&tets.vertices, [eu, ev], [cu, cv]) {
                    return true;
                }
            }
            false
        }
        Constraint::Face([ca, cb, cc]) => {
            // The new triangle crosses the constraint face if any of its edges
            // pierces the constraint-face interior.
            for &(eu, ev) in &[(fa, fb), (fb, fc), (fc, fa)] {
                if eu == ca || eu == cb || eu == cc || ev == ca || ev == cb || ev == cc {
                    continue;
                }
                if segment_intersects_triangle(
                    tets.vertices[eu],
                    tets.vertices[ev],
                    tets.vertices[ca],
                    tets.vertices[cb],
                    tets.vertices[cc],
                ) {
                    return true;
                }
            }
            false
        }
    }
}

/// Eligibility guard for a candidate flip.
///
/// `new_edge` is `Some((u, v))` for a 2-3 flip (the new shared edge);
/// `new_face` is `Some([a, b, c])` for a 3-2 / 2-2 flip (the new shared
/// face). Returns `false` if the new element crosses the active constraint in
/// its interior, `true` otherwise. `Constraint::None` is always eligible.
fn check_flip_eligibility(
    tets: &Delaunay3D,
    new_edge: Option<(usize, usize)>,
    new_face: Option<[usize; 3]>,
    constraint: &Constraint,
) -> bool {
    if let Constraint::None = constraint {
        return true;
    }
    if let Some((u, v)) = new_edge {
        if edge_crosses_constraint(tets, u, v, constraint) {
            return false;
        }
    }
    if let Some([a, b, c]) = new_face {
        if face_crosses_constraint(tets, a, b, c, constraint) {
            return false;
        }
    }
    true
}

/// Check if edge [p0, p1] crosses triangle [a, b, c].
/// Returns true if the intersection point lies strictly inside the triangle.
fn edge_crosses_triangle(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    p0: [f64; 3],
    p1: [f64; 3],
) -> bool {
    // Moller-Trumbore ray-triangle intersection
    let dir = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let edge2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];

    let h = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];
    let det = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
    if det.abs() < 1e-14 {
        return false;
    } // parallel

    let f = 1.0 / det;
    let s = [p0[0] - a[0], p0[1] - a[1], p0[2] - a[2]];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(-1e-10..=1.0 + 1e-10).contains(&u) {
        return false;
    }

    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if v < -1e-10 || u + v > 1.0 + 1e-10 {
        return false;
    }

    let t_param = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    // Intersection must be strictly between endpoints (not at endpoints)
    t_param > 1e-6 && t_param < 1.0 - 1e-6
}

/// Validate tet mesh adjacency and orientation (debug builds only).
///
/// Checks:
/// - No tet has negative volume
/// - Adjacency is symmetric (if tet A says neighbor on face F is tet B,
///   then B says its neighbor on the corresponding face is A)
///
/// Issues are logged to stderr rather than panicking, since flip-based
/// boundary recovery can leave transient adjacency inconsistencies that
/// do not affect correctness of the final mesh.
#[cfg(debug_assertions)]
fn debug_validate_tets(tets: &Delaunay3D) {
    let mut neg_vol_count = 0usize;
    let mut adj_broken_count = 0usize;

    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        // Check positive volume
        let v = &tet.verts;
        let vol = orient_3d(
            tets.vertices[v[0]],
            tets.vertices[v[1]],
            tets.vertices[v[2]],
            tets.vertices[v[3]],
        );
        if vol < -1e-30 {
            neg_vol_count += 1;
        }
        // Check adjacency symmetry
        for fi in 0..4 {
            let neighbor = tet.adj[fi];
            if neighbor == usize::MAX || !tets.is_live(neighbor) {
                continue;
            }
            let ntet = &tets.tets[neighbor];
            let found = ntet.adj.contains(&i);
            if !found {
                adj_broken_count += 1;
            }
        }
    }

    if neg_vol_count > 0 || adj_broken_count > 0 {
        eprintln!(
            "cad-to-dagmc-mesher: debug_validate_tets: {neg_vol_count} negative-volume tet(s), \
             {adj_broken_count} broken adjacency link(s)"
        );
    }
}

/// Recover boundary edges in the Delaunay tetrahedralization.
///
/// For each edge (a, b) in the input boundary that is NOT already present
/// as a tet edge, perform local flips to recover it.
///
/// Returns `(recovered, failed, failed_edges)`.
pub fn recover_edges(
    tets: &mut Delaunay3D,
    boundary_edges: &[(usize, usize)],
) -> (usize, usize, Vec<[usize; 2]>) {
    let mut recovered = 0;
    let mut failed = 0;
    let mut failed_edges = Vec::new();

    for &(a, b) in boundary_edges {
        if edge_exists_in_tets(tets, a, b) {
            continue; // already present
        }
        if recover_edge_by_flips(tets, a, b) {
            recovered += 1;
        } else {
            failed += 1;
            failed_edges.push([a, b]);
        }
    }

    #[cfg(debug_assertions)]
    debug_validate_tets(tets);

    (recovered, failed, failed_edges)
}

/// Check whether edge (a, b) exists in any live tet.
///
/// O(degree) via the incidence index when active (a tet has edge (a,b) iff it
/// contains both, so only tets incident to `a` can qualify); falls back to the
/// O(#tets) scan when the index is not built (standalone/unit-test use).
pub fn edge_exists_in_tets(tets: &Delaunay3D, a: usize, b: usize) -> bool {
    if tets.index_active() {
        for &ti in tets.incident_tets(a) {
            let ti = ti as usize;
            if tets.is_live(ti) && tets.tets[ti].verts.contains(&b) {
                return true;
            }
        }
        return false;
    }
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) {
            continue;
        }
        let v = &tet.verts;
        let has_a = v[0] == a || v[1] == a || v[2] == a || v[3] == a;
        let has_b = v[0] == b || v[1] == b || v[2] == b || v[3] == b;
        if has_a && has_b {
            return true;
        }
    }
    false
}

/// Find all live FINITE tets that contain vertex `v`. O(degree) via the index.
fn tets_incident_to_vertex(tets: &Delaunay3D, v: usize) -> Vec<usize> {
    let mut result = Vec::new();
    if tets.index_active() {
        for &ti in tets.incident_tets(v) {
            let ti = ti as usize;
            if tets.is_live(ti) && !is_hull_tet(&tets.tets[ti]) {
                result.push(ti);
            }
        }
        return result;
    }
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        if tet.verts.contains(&v) {
            result.push(i);
        }
    }
    result
}

/// Find all live FINITE tets that share edge (a, b). O(degree) via the index.
fn tets_sharing_edge(tets: &Delaunay3D, a: usize, b: usize) -> Vec<usize> {
    let mut result = Vec::new();
    if tets.index_active() {
        for &ti in tets.incident_tets(a) {
            let ti = ti as usize;
            if tets.is_live(ti) && !is_hull_tet(&tets.tets[ti]) && tets.tets[ti].verts.contains(&b)
            {
                result.push(ti);
            }
        }
        return result;
    }
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        let v = &tet.verts;
        let has_a = v[0] == a || v[1] == a || v[2] == a || v[3] == a;
        let has_b = v[0] == b || v[1] == b || v[2] == b || v[3] == b;
        if has_a && has_b {
            result.push(i);
        }
    }
    result
}

/// Determine whether the line segment from point `a` to point `b`
/// intersects the interior of triangle (p, q, r).
///
/// This is the core geometric test for finding blocking faces.
/// Returns true if segment AB passes through the open interior of triangle PQR.
fn segment_intersects_triangle(
    a: [f64; 3],
    b: [f64; 3],
    p: [f64; 3],
    q: [f64; 3],
    r: [f64; 3],
) -> bool {
    // The segment AB intersects triangle PQR if:
    // 1) A and B are on opposite sides of plane(PQR)
    // 2) The intersection point lies inside the triangle

    let o_a = orient_3d(p, q, r, a);
    let o_b = orient_3d(p, q, r, b);

    // Both on same side or on the plane => no proper crossing
    if o_a * o_b >= 0.0 {
        return false;
    }

    // A and B are on opposite sides. Now check if the crossing point
    // is inside triangle PQR. This is equivalent to checking that
    // the edge AB is on the same side of each edge of the triangle
    // when projected along the triangle normal.
    //
    // Equivalently, we can use orient_3d tests:
    // The intersection is inside PQR iff:
    //   orient_3d(a, b, p, q) * orient_3d(a, b, p, r) <= 0  (not sufficient alone)
    //
    // Better approach: parametric intersection. But we can use the
    // signed volumes approach from Devillers & Guigue:
    // The segment AB crosses triangle PQR iff all three "side" tests
    // have the same sign:
    let o1 = orient_3d(a, b, p, q);
    let o2 = orient_3d(a, b, q, r);
    let o3 = orient_3d(a, b, r, p);

    // All same sign (all positive or all negative) means intersection
    // inside the triangle. Zero means on an edge (we treat as not
    // blocking to avoid ambiguity).
    if o1 == 0.0 || o2 == 0.0 || o3 == 0.0 {
        return false;
    }

    (o1 > 0.0 && o2 > 0.0 && o3 > 0.0) || (o1 < 0.0 && o2 < 0.0 && o3 < 0.0)
}

/// Find tets whose faces are crossed by the segment (a, b).
///
/// Returns the list of tet indices that have at least one face whose
/// interior is properly intersected by the segment from vertex `a` to
/// vertex `b`. These are the tets forming the "crossing cavity" that
/// must be flipped to recover edge (a, b).
pub fn find_tets_crossing_edge(tets: &Delaunay3D, a: usize, b: usize) -> Vec<usize> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let mut crossing = Vec::new();

    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        // Skip tets that contain a or b as a vertex
        if tet.verts.contains(&a) || tet.verts.contains(&b) {
            continue;
        }

        // Check each of the 4 faces of this tet
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(tet.verts, fi);
            let p = tets.vertices[face[0]];
            let q = tets.vertices[face[1]];
            let r = tets.vertices[face[2]];
            if segment_intersects_triangle(pa, pb, p, q, r) {
                crossing.push(i);
                break; // only need to add the tet once
            }
        }
    }

    crossing
}

/// Find the point where segment (a, b) FIRST crosses a finite tet face,
/// measured from `a`. Used by conforming segment recovery: inserting a Steiner
/// point here removes that crossing (it lies on the crossed face AND on the
/// segment). Returns `None` if the segment crosses no finite face interior
/// (e.g. it already runs along existing edges/faces). The returned point lies
/// strictly between `a` and `b` (parameter in (eps, 1-eps)).
fn first_segment_crossing_point(tets: &Delaunay3D, a: usize, b: usize) -> Option<[f64; 3]> {
    FALLBACK_FSCP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    first_segment_crossing_point_impl(tets, a, b, true, 1e-9)
}

/// Fraction of the segment length within which a crossing counts as GRAZING
/// an endpoint (rejected by the Steiner filter: splitting essentially at an
/// endpoint spawns a sliver whose faces are grazed again - the historical
/// non-terminating cascade). Shared by the rejection filter and the
/// past-the-graze rescan below so they can never disagree.
const GRAZE_T_BAND: f64 = 1e-2;

/// First crossing of segment (a, b) PAST the endpoint-grazing band: the
/// minimum-parameter face crossing with `t ∈ (GRAZE_T_BAND, 1−GRAZE_T_BAND)`,
/// endpoint-incident tets included. The curved-wall long-chord class (issue
/// #57) enters the mesh at a shallow angle, so its FIRST crossing grazes an
/// endpoint and gets rejected - but the chord crosses many more faces, and
/// splitting at the first non-grazing one is exactly as conformal and
/// productive (it removes that crossing). O(#tets); only runs after the
/// normal pick was rejected, i.e. on the previously-unproductive branch.
fn first_segment_crossing_point_past_graze(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
) -> Option<[f64; 3]> {
    first_segment_crossing_point_impl(tets, a, b, false, GRAZE_T_BAND)
}

/// NEAR-COPLANAR BLOCKING EDGE (issue #57 layer 4, the irreducible class):
/// the segment lies within fp dust of a mesh face's plane and crosses that
/// face IN-PLANE - so the elementary obstruction is a mesh EDGE (u, v)
/// crossing the open segment at near-zero separation, exactly the #31
/// coplanar `AcrossEdge` with `orient_3d(a,b,u,v) ≈ 1e-16` instead of an
/// exact zero. The march cannot see it (no transversal crossing exists) and
/// every flip is rejected by the volume guards (near-zero children).
///
/// The convergent resolution mirrors the 2-D constrained insertion realized
/// through Steiner topology instead of flips: split the blocking edge AT the
/// crossing point, placed exactly ON the segment (surface geometry is
/// preserved; the point is within dust of (u, v), so `insert_steiner_local`
/// classifies it on-edge and `split_edge_on_constraint` performs the
/// topological split). Each such split consumes one crossing edge; the
/// sub-segments then close by flips - finitely many crossing edges, so this
/// terminates like its 2-D counterpart.
///
/// Returns the crossing point with the SMALLEST parameter along (a, b) among
/// mesh edges that cross the open segment with: both endpoints off the
/// segment's vertices, not a protected surface segment (two surface segments
/// at dust distance is a surface self-touch - out of scope), crossing
/// parameters interior on both (the near-vertex cases belong to the
/// reference-point / voe machinery), and inter-segment distance within the
/// codebase's established dust standard (`dist² ≤ |ab|²·1e-24`, the
/// vertex-on-edge tolerance). O(#tets); failure-path only.
/// Mesh edge passing exactly THROUGH a segment endpoint (issue #57 layer 4b):
/// endpoint `e ∈ {a, b}` lies ON the open edge (u, v) within the codebase's
/// dust standard (`dist² ≤ |uv|²·1e-24`, the vertex-on-edge tolerance), with
/// the projection strictly interior. This is exactly the voe class - but the
/// march never reports it here (the surrounding configuration is the
/// near-tangent plane where `finddirection` dead-ends), so it needs a direct
/// scan. The caller resolves it topologically via `split_edge_at_vertex`
/// (no Steiner point, exactly the #55/#59 machinery - including the
/// sliver-cluster surgery for rings already containing the vertex).
/// Protected (surface) edges are skipped. O(#tets); failure-path only.
fn edge_through_endpoint(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
    psegs: &HashSet<(usize, usize)>,
) -> Option<(usize, usize, usize)> {
    let mut seen: HashSet<(usize, usize)> = HashSet::default();
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        let v = tet.verts;
        for &[ei, ej] in &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] {
            let (u, w) = (v[ei], v[ej]);
            if u == a || u == b || w == a || w == b {
                continue;
            }
            let key = (u.min(w), u.max(w));
            if !seen.insert(key) || psegs.contains(&key) {
                continue;
            }
            let pu = tets.vertices[u];
            let pw = tets.vertices[w];
            let d = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if l2 <= 0.0 {
                continue;
            }
            for &e in &[a, b] {
                let pe = tets.vertices[e];
                let r = [pe[0] - pu[0], pe[1] - pu[1], pe[2] - pu[2]];
                let s = (r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / l2;
                if !(1e-6..=1.0 - 1e-6).contains(&s) {
                    continue;
                }
                let proj = [pu[0] + s * d[0], pu[1] + s * d[1], pu[2] + s * d[2]];
                let dist2 = (pe[0] - proj[0]).powi(2)
                    + (pe[1] - proj[1]).powi(2)
                    + (pe[2] - proj[2]).powi(2);
                if dist2 <= l2 * 1e-24 {
                    return Some((u, w, e));
                }
            }
        }
    }
    None
}

fn nearest_inplane_edge_crossing(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
    psegs: &HashSet<(usize, usize)>,
) -> Option<[f64; 3]> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let d1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let l1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
    if l1 <= 0.0 {
        return None;
    }
    let mut seen: HashSet<(usize, usize)> = HashSet::default();
    let mut best: Option<(f64, [f64; 3])> = None; // (t, point)
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        let v = tet.verts;
        for &[ei, ej] in &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] {
            let (u, w) = (v[ei], v[ej]);
            if u == a || u == b || w == a || w == b {
                continue;
            }
            let key = (u.min(w), u.max(w));
            if !seen.insert(key) || psegs.contains(&key) {
                continue;
            }
            let pu = tets.vertices[u];
            let pw = tets.vertices[w];
            let d2 = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
            let l2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
            if l2 <= 0.0 {
                continue;
            }
            // Closest approach of the two infinite lines: solve
            //   [ l1   -d12 ] [t]   [ r·d1 ]
            //   [ d12  -l2  ] [s] = [ r·d2 ]   with r = pu - pa.
            let d12 = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
            let denom = l1 * l2 - d12 * d12;
            // Near-parallel lines never "cross"; skip (relative test).
            if denom <= l1 * l2 * 1e-12 {
                continue;
            }
            let r = [pu[0] - pa[0], pu[1] - pa[1], pu[2] - pa[2]];
            let rd1 = r[0] * d1[0] + r[1] * d1[1] + r[2] * d1[2];
            let rd2 = r[0] * d2[0] + r[1] * d2[1] + r[2] * d2[2];
            let t = (rd1 * l2 - rd2 * d12) / denom;
            let s = (rd1 * d12 - rd2 * l1) / denom;
            // Interior on both: near-endpoint crossings belong to the
            // grazing / reference-point / voe layers.
            if !(GRAZE_T_BAND..=1.0 - GRAZE_T_BAND).contains(&t)
                || !(GRAZE_T_BAND..=1.0 - GRAZE_T_BAND).contains(&s)
            {
                continue;
            }
            if best.map(|(bt, _)| t >= bt).unwrap_or(false) {
                continue; // already have an earlier crossing
            }
            let p1 = [pa[0] + t * d1[0], pa[1] + t * d1[1], pa[2] + t * d1[2]];
            let p2 = [pu[0] + s * d2[0], pu[1] + s * d2[1], pu[2] + s * d2[2]];
            let dist2 = (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2);
            if dist2 > l1 * 1e-24 {
                continue; // not an in-plane (dust-level) crossing
            }
            best = Some((t, p1));
        }
    }
    best.map(|(_, p)| p)
}

/// TetGen-style REFERENCE-POINT split position for a segment with no usable
/// face crossing (issue #57's near-tangent class: the blocking face is almost
/// parallel to the segment, so the plane-crossing parameter is ill-defined
/// or outside the segment). The obstruction is then governed by the mesh
/// vertex closest to the OPEN segment: split at that vertex's projection.
/// This keys each split to a concrete blocking feature (TetGen's `refpt`),
/// unlike blind midpoint halving which never converges here. Returns the
/// projection point for the nearest vertex with projection parameter inside
/// the grazing band and a strictly-positive (not-on-segment) distance; the
/// exact on-segment cases belong to the vertex-on-edge machinery. O(#verts);
/// failure-path only.
fn reference_point_on_segment(tets: &Delaunay3D, a: usize, b: usize) -> Option<[f64; 3]> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if l2 <= 0.0 {
        return None;
    }
    let mut best: Option<(f64, f64)> = None; // (dist2, t)
    for (vi, pv) in tets.vertices.iter().enumerate() {
        if vi == a || vi == b {
            continue;
        }
        let ap = [pv[0] - pa[0], pv[1] - pa[1], pv[2] - pa[2]];
        let t = (ap[0] * d[0] + ap[1] * d[1] + ap[2] * d[2]) / l2;
        if !(GRAZE_T_BAND..=1.0 - GRAZE_T_BAND).contains(&t) {
            continue;
        }
        let proj = [pa[0] + t * d[0], pa[1] + t * d[1], pa[2] + t * d[2]];
        let dist2 =
            (pv[0] - proj[0]).powi(2) + (pv[1] - proj[1]).powi(2) + (pv[2] - proj[2]).powi(2);
        // A vertex effectively ON the segment is the voe/coplanar machinery's
        // case; splitting at its projection would mint a near-duplicate.
        if dist2 <= l2 * 1e-24 {
            continue;
        }
        if best.map(|(bd, _)| dist2 < bd).unwrap_or(true) {
            best = Some((dist2, t));
        }
    }
    best.map(|(_, t)| [pa[0] + t * d[0], pa[1] + t * d[1], pa[2] + t * d[2]])
}

/// Like [`first_segment_crossing_point`] but INCLUDES endpoint-incident tets
/// (does NOT skip tets containing `a` or `b`). The first crossing along the
/// march from `a` can lie on a face of an endpoint-incident tet - exactly the
/// case the skip-endpoint scan (and `finddirection`, when the chord exits the
/// hull at a reflex vertex) cannot see. Faces touching BOTH endpoints are still
/// skipped (a face through `a` or `b` is not a true interior obstruction unless
/// the opposite endpoint is off it, which the strict-straddle test enforces).
fn first_segment_crossing_point_incl_endpoints(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
) -> Option<[f64; 3]> {
    FALLBACK_FSCP_INCL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    first_segment_crossing_point_impl(tets, a, b, false, 1e-9)
}

/// `t_band`: crossings with parameter within `t_band` of either endpoint are
/// skipped (1e-9 = strictly-interior only; GRAZE_T_BAND = also skip the
/// endpoint-grazing zone).
fn first_segment_crossing_point_impl(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
    skip_endpoint_tets: bool,
    t_band: f64,
) -> Option<[f64; 3]> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let dir = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];

    let mut best_t = f64::INFINITY;
    let mut best_pt: Option<[f64; 3]> = None;

    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) || is_hull_tet(tet) {
            continue;
        }
        if skip_endpoint_tets && (tet.verts.contains(&a) || tet.verts.contains(&b)) {
            continue;
        }
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(tet.verts, fi);
            // A face containing `a` or `b` cannot be a true interior crossing of
            // the open segment (the strict-straddle test below would reject it
            // anyway, but skip early to avoid spurious near-endpoint hits).
            if face.contains(&a) || face.contains(&b) {
                continue;
            }
            let p = tets.vertices[face[0]];
            let q = tets.vertices[face[1]];
            let r = tets.vertices[face[2]];
            if !segment_intersects_triangle(pa, pb, p, q, r) {
                continue;
            }
            // Parametric intersection of segment with the triangle's plane.
            let n = {
                let pq = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                let pr = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
                [
                    pq[1] * pr[2] - pq[2] * pr[1],
                    pq[2] * pr[0] - pq[0] * pr[2],
                    pq[0] * pr[1] - pq[1] * pr[0],
                ]
            };
            let denom = dir[0] * n[0] + dir[1] * n[1] + dir[2] * n[2];
            if denom.abs() < 1e-30 {
                continue; // segment parallel to face plane
            }
            let ap = [p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]];
            let t = (ap[0] * n[0] + ap[1] * n[1] + ap[2] * n[2]) / denom;
            // Keep crossings outside the excluded endpoint bands only.
            if t <= t_band || t >= 1.0 - t_band {
                continue;
            }
            if t < best_t {
                best_t = t;
                best_pt = Some([pa[0] + t * dir[0], pa[1] + t * dir[1], pa[2] + t * dir[2]]);
            }
        }
    }
    best_pt
}

/// Perform a 2-to-3 flip: replace two tets sharing a face with three
/// tets sharing an edge.
///
/// Given tets T1 = (a, b, c, d) and T2 = (a, b, c, e) sharing face
/// (a, b, c), create three new tets:
///   (a, b, d, e), (b, c, d, e), (a, c, e, d)
///
/// The new tets all share edge (d, e).
///
/// This is **transactional**: all candidate new tets are validated for
/// positive orientation (orient_3d != 0 after orientation fixing) and the
/// new shared edge (d, e) is checked against the active `constraint` BEFORE
/// any old tet is freed. On any rejection the function is a true no-op - the
/// mesh is left byte-for-byte unchanged and `None` is returned.
///
/// Returns the indices of the 3 new tets, or `None` if the flip is not
/// geometrically valid or is constraint-ineligible.
pub fn flip_2_to_3(
    tets: &mut Delaunay3D,
    t1: usize,
    t2: usize,
    constraint: &Constraint,
) -> Option<[usize; 3]> {
    if is_hull_tet(&tets.tets[t1]) || is_hull_tet(&tets.tets[t2]) {
        return None;
    }
    let (fi1, fi2) = delaunay3d::shared_face_indices(&tets.tets[t1], &tets.tets[t2])?;

    let shared_face = delaunay3d::opposite_face(tets.tets[t1].verts, fi1);
    let fa = shared_face[0];
    let fb = shared_face[1];
    let fc = shared_face[2];

    // The apex of t1 opposite the shared face
    let d = tets.tets[t1].verts[fi1];
    // The apex of t2 opposite the shared face
    let e = tets.tets[t2].verts[fi2];

    // ── VALIDATION PHASE (no mutation) ──
    // TOPOLOGICAL VALIDITY (TetGen's `getedge` guard): the new shared edge
    // (d, e) must NOT already exist. If it does, the three new tets would
    // coincide with / overlap existing tets (a non-simplicial result), so the
    // flip is invalid - reject as a no-op. Without this guard a 2-3 flip on a
    // (near-)coplanar pair can silently introduce overlapping tets.
    if edge_exists_in_tets(tets, d, e) {
        return None;
    }

    // GEOMETRIC CONVEXITY: a 2-3 flip is only valid (non-overlapping) when the
    // union of the two tets is a convex bipyramid - i.e. the new edge (d, e)
    // pierces the INTERIOR of the shared triangle (fa, fb, fc). Equivalently
    // the three signed volumes orient(fa,fb,d,e), orient(fb,fc,d,e),
    // orient(fc,fa,d,e) must all share one sign. If they differ (a reflex/
    // non-convex pair), the three candidate tets would overlap even though each
    // is individually positive after orientation-fixing. Reject as a no-op.
    {
        let vd = tets.vertices[d];
        let ve = tets.vertices[e];
        let s1 = orient_3d(tets.vertices[fa], tets.vertices[fb], vd, ve);
        let s2 = orient_3d(tets.vertices[fb], tets.vertices[fc], vd, ve);
        let s3 = orient_3d(tets.vertices[fc], tets.vertices[fa], vd, ve);
        if s1 == 0.0 || s2 == 0.0 || s3 == 0.0 {
            return None; // (d,e) grazes a face edge → degenerate
        }
        let all_pos = s1 > 0.0 && s2 > 0.0 && s3 > 0.0;
        let all_neg = s1 < 0.0 && s2 < 0.0 && s3 < 0.0;
        if !(all_pos || all_neg) {
            return None; // non-convex pair → 2-3 flip would overlap
        }
    }

    // Build the 3 candidate new tets: each uses one edge of the face + both
    // apices. All three share the new edge (d, e).
    let candidate_verts: [[usize; 4]; 3] = [[fa, fb, d, e], [fb, fc, d, e], [fc, fa, d, e]];

    // 1) Every candidate must be non-degenerate (orient_3d != 0). Compute the
    //    correctly-oriented vertex list for each without touching the mesh.
    let mut oriented = [[0usize; 4]; 3];
    for (k, nv) in candidate_verts.iter().enumerate() {
        let mut v = *nv;
        let o = orient_3d(
            tets.vertices[v[0]],
            tets.vertices[v[1]],
            tets.vertices[v[2]],
            tets.vertices[v[3]],
        );
        if o < 0.0 {
            v.swap(0, 1);
        } else if o == 0.0 {
            return None; // degenerate candidate → reject, nothing changed
        }
        oriented[k] = v;
    }

    // 2) Constraint guard: the new shared edge (d, e) must not cross the
    //    active constraint in its interior.
    if !check_flip_eligibility(tets, Some((d, e)), None, constraint) {
        return None;
    }

    // 3) Gather external adjacency (neighbor tet + the shared face) for every
    //    non-shared face of t1 and t2, while the old tets are still live.
    let mut ext_links: Vec<(usize, [usize; 3])> = Vec::with_capacity(6);
    for fi in 0..4 {
        if fi == fi1 {
            continue;
        }
        let neighbor = tets.tets[t1].adj[fi];
        let face = delaunay3d::opposite_face(tets.tets[t1].verts, fi);
        ext_links.push((neighbor, face));
    }
    for fi in 0..4 {
        if fi == fi2 {
            continue;
        }
        let neighbor = tets.tets[t2].adj[fi];
        let face = delaunay3d::opposite_face(tets.tets[t2].verts, fi);
        ext_links.push((neighbor, face));
    }

    // TRANSACTIONAL VOLUME GUARD (see flip_volume_ok): the candidate tets must
    // tile the same finite volume as the tets they replace - reject otherwise.
    {
        let (vo, _) = flipvol_of(tets, &[t1, t2]);
        let vn = flipvol_of_candidates(tets, &oriented);
        if !flip_volume_ok(vo, vn) {
            return None;
        }
    }
    // ── COMMIT PHASE (everything validated; now mutate) ──
    let _fv = if flipvol_enabled() {
        Some(flipvol_of(tets, &[t1, t2]))
    } else {
        None
    };
    tets.free_tet(t1);
    tets.free_tet(t2);

    let mut new_indices = [0usize; 3];
    for (k, v) in oriented.iter().enumerate() {
        new_indices[k] = tets.alloc_tet(Tet {
            verts: *v,
            adj: [usize::MAX; 4],
        });
    }
    if let Some(o) = _fv {
        flipvol_report("flip_2_to_3", o, tets, &new_indices);
    }

    // Inter-new-tet adjacency: each pair shares a face (d, e, one face vertex).
    for i in 0..3 {
        for j in (i + 1)..3 {
            let ni = new_indices[i];
            let nj = new_indices[j];
            if let Some((si, sj)) = delaunay3d::shared_face_indices(&tets.tets[ni], &tets.tets[nj])
            {
                tets.tets[ni].adj[si] = nj;
                tets.tets[nj].adj[sj] = ni;
            }
        }
    }

    // External adjacency: link each new tet that owns an external face to the
    // corresponding outside neighbor (and fix the neighbor's back-pointer).
    for (neighbor, face) in &ext_links {
        if *neighbor == usize::MAX || !tets.is_live(*neighbor) {
            continue;
        }
        for &ni in &new_indices {
            let mut linked = false;
            for fi in 0..4 {
                let nf = delaunay3d::opposite_face(tets.tets[ni].verts, fi);
                if delaunay3d::faces_match(&nf, face) {
                    tets.tets[ni].adj[fi] = *neighbor;
                    for nfi in 0..4 {
                        let neighbor_face =
                            delaunay3d::opposite_face(tets.tets[*neighbor].verts, nfi);
                        if delaunay3d::faces_match(&neighbor_face, face) {
                            tets.tets[*neighbor].adj[nfi] = ni;
                            break;
                        }
                    }
                    linked = true;
                    break;
                }
            }
            if linked {
                break;
            }
        }
    }

    Some(new_indices)
}

/// Perform a 3-to-2 flip: replace three tets sharing an edge with two
/// tets sharing a face.
///
/// Given three tets that all share edge (p, q), and whose "ring" vertices
/// around that edge are (a, b, c), create two new tets:
///   (a, b, c, p) and (a, b, c, q)  [with correct orientation]
///
/// This is the inverse of a 2-to-3 flip.
///
/// `tet_indices`: the 3 tet indices sharing edge (p, q).
/// `p`, `q`: the shared edge vertices.
///
/// This is **transactional**: the ring is validated (exactly 3 ring vertices,
/// p and q on opposite sides of plane(a,b,c)) and the new shared face
/// (a, b, c) is checked against the active `constraint` BEFORE any old tet is
/// freed. On any rejection the function is a true no-op - the mesh is left
/// byte-for-byte unchanged and `None` is returned.
///
/// Returns the indices of the 2 new tets, or `None` if the flip is invalid or
/// constraint-ineligible.
pub fn flip_3_to_2(
    tets: &mut Delaunay3D,
    tet_indices: [usize; 3],
    p: usize,
    q: usize,
    constraint: &Constraint,
) -> Option<[usize; 2]> {
    if tet_indices.iter().any(|&ti| is_hull_tet(&tets.tets[ti])) {
        return None;
    }
    // Find the ring vertices: the 3 vertices other than p and q among the 3 tets.
    // Each tet has 4 vertices, two of which are p and q. The other two
    // include ring vertices shared with neighboring tets in the ring.
    let mut ring_verts: Vec<usize> = Vec::new();
    for &ti in &tet_indices {
        for &v in &tets.tets[ti].verts {
            if v != p && v != q && !ring_verts.contains(&v) {
                ring_verts.push(v);
            }
        }
    }

    if ring_verts.len() != 3 {
        return None; // Not a valid 3-tet ring around edge (p, q)
    }

    let a = ring_verts[0];
    let b = ring_verts[1];
    let c = ring_verts[2];

    // ── VALIDATION PHASE (no mutation) ──
    // Check that the two new tets would have positive orientation.
    let va = tets.vertices[a];
    let vb = tets.vertices[b];
    let vc = tets.vertices[c];
    let vp = tets.vertices[p];
    let vq = tets.vertices[q];

    let o1 = orient_3d(va, vb, vc, vp);
    let o2 = orient_3d(va, vb, vc, vq);

    // For a valid flip, p and q must be on STRICTLY opposite sides of
    // plane(a,b,c). Coplanar (o==0) would yield a degenerate new tet.
    if o1 == 0.0 || o2 == 0.0 || (o1 > 0.0) == (o2 > 0.0) {
        return None; // Not a valid configuration for 3-to-2 flip
    }

    // TOPOLOGICAL VALIDITY: the new shared face (a, b, c) must NOT already exist
    // elsewhere in the mesh. If it does, the two new tets would coincide with /
    // overlap an existing tet (a non-simplicial result), so reject as a no-op.
    if face_exists_in_tets(tets, a, b, c) {
        return None;
    }

    // Constraint guard: the new shared face (a, b, c) must not cross the
    // active constraint in its interior.
    if !check_flip_eligibility(tets, None, Some([a, b, c]), constraint) {
        return None;
    }

    // Collect external adjacency before deleting old tets.
    // Each old tet has 4 faces. Two of those faces are shared with other
    // tets in the ring (they contain both p and q plus one ring vertex).
    // The remaining 2 faces are external.
    //
    // External faces of the ring:
    // - 3 faces containing p (one per ring-edge: ab, bc, ca)
    // - 3 faces containing q (one per ring-edge: ab, bc, ca)
    // Total 6 external faces. The 2 new tets each have 4 faces:
    // new1 = (a,b,c,p): faces opp a=(b,c,p), opp b=(a,c,p), opp c=(a,b,p), opp p=(a,b,c)
    // new2 = (a,b,c,q): faces opp a=(b,c,q), opp b=(a,c,q), opp c=(a,b,q), opp q=(a,b,c)
    // Faces (a,b,c) in new1 and new2 are the shared face between the two new tets.
    // The other 3 faces of each new tet are external.

    let old_tets_set: HashSet<usize> = tet_indices.iter().copied().collect();

    // For each old tet, find external neighbors (neighbors not in the ring).
    let mut ext_links: Vec<(usize, [usize; 3])> = Vec::new();
    for &ti in &tet_indices {
        for fi in 0..4 {
            let neighbor = tets.tets[ti].adj[fi];
            if neighbor == usize::MAX || !old_tets_set.contains(&neighbor) {
                // External face
                let face = delaunay3d::opposite_face(tets.tets[ti].verts, fi);
                ext_links.push((neighbor, face));
            }
        }
    }

    // ── COMMIT PHASE (everything validated; now mutate) ──
    // Compute oriented vertex lists for the new tets (orient checked != 0 above).
    let mut v1 = [a, b, c, p];
    if o1 < 0.0 {
        v1.swap(0, 1);
    }
    let mut v2 = [a, b, c, q];
    if o2 < 0.0 {
        v2.swap(0, 1);
    }

    // TRANSACTIONAL VOLUME GUARD (see flip_volume_ok): the candidate tets must
    // tile the same finite volume as the tets they replace - reject otherwise.
    {
        let (vo, _) = flipvol_of(tets, &tet_indices);
        let vn = flipvol_of_candidates(tets, &[v1, v2]);
        if !flip_volume_ok(vo, vn) {
            return None;
        }
    }
    // Delete old tets
    let _fv = if flipvol_enabled() {
        Some(flipvol_of(tets, &tet_indices))
    } else {
        None
    };
    for &ti in &tet_indices {
        tets.free_tet(ti);
    }

    let n1 = tets.alloc_tet(Tet {
        verts: v1,
        adj: [usize::MAX; 4],
    });
    let n2 = tets.alloc_tet(Tet {
        verts: v2,
        adj: [usize::MAX; 4],
    });
    if let Some(o) = _fv {
        flipvol_report("flip_3_to_2", o, tets, &[n1, n2]);
    }

    // Link the two new tets to each other via face (a, b, c)
    if let Some((f1, f2)) = delaunay3d::shared_face_indices(&tets.tets[n1], &tets.tets[n2]) {
        tets.tets[n1].adj[f1] = n2;
        tets.tets[n2].adj[f2] = n1;
    }

    // Restore external adjacency
    let new_tets = [n1, n2];
    for (neighbor, face) in &ext_links {
        if *neighbor == usize::MAX {
            // Find which new tet has this face and leave adj as MAX
            continue;
        }
        if !tets.is_live(*neighbor) {
            continue;
        }
        for &ni in &new_tets {
            for fi in 0..4 {
                let nf = delaunay3d::opposite_face(tets.tets[ni].verts, fi);
                if delaunay3d::faces_match(&nf, face) {
                    tets.tets[ni].adj[fi] = *neighbor;
                    // Update neighbor's adj to point to new tet
                    for nfi in 0..4 {
                        if tets.tets[*neighbor].adj[nfi] == usize::MAX {
                            continue;
                        }
                        // The neighbor used to point to one of the old tets.
                        // Check if neighbor's face at nfi matches.
                        if old_tets_set.contains(&tets.tets[*neighbor].adj[nfi]) {
                            let nface = delaunay3d::opposite_face(tets.tets[*neighbor].verts, nfi);
                            if delaunay3d::faces_match(&nface, face) {
                                tets.tets[*neighbor].adj[nfi] = ni;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    Some([n1, n2])
}

/// Perform a 2-to-2 flip: swap the shared diagonal of a coplanar quad.
///
/// Given two tets T1 and T2 sharing triangular face F = (f0, f1, f2), with
/// apices d (of T1) and e (of T2), the five-vertex region {f0,f1,f2,d,e} is a
/// pyramid over a planar quadrilateral base whenever exactly one edge
/// (f_i, f_j) of the shared face is coplanar with both apices d and e (i.e.
/// `orient_3d(f_i, f_j, d, e) == 0`). In that case the base quad is split on
/// the diagonal (f_i, f_j); swapping it to the diagonal (d, e) replaces the two
/// tets with
///   (f_k, f_i, d, e) and (f_k, f_j, d, e)
/// (where f_k is the third shared-face vertex), which now share face
/// (f_k, d, e). This is the degenerate limit of a 2-3 flip in which the third
/// candidate tet (f_i, f_j, d, e) is flat - exactly the operation a cube's
/// boundary quad needs when the Delaunay picked the opposite diagonal.
///
/// This is **transactional** and guarded by `constraint`: all new tets are
/// validated for non-zero orientation and the new edge (d, e) + new face
/// (f_k, d, e) are checked against the active constraint BEFORE any old tet is
/// freed. On any rejection it is a true no-op returning `None`.
///
/// Returns the indices of the 2 new tets, or `None` if the configuration is
/// not a coplanar-quad diagonal swap or the flip is constraint-ineligible.
pub fn flip_2_to_2(
    tets: &mut Delaunay3D,
    t1: usize,
    t2: usize,
    constraint: &Constraint,
) -> Option<[usize; 2]> {
    if is_hull_tet(&tets.tets[t1]) || is_hull_tet(&tets.tets[t2]) {
        return None;
    }
    let (fi1, fi2) = delaunay3d::shared_face_indices(&tets.tets[t1], &tets.tets[t2])?;

    let shared_face = delaunay3d::opposite_face(tets.tets[t1].verts, fi1);
    let f = [shared_face[0], shared_face[1], shared_face[2]];
    let d = tets.tets[t1].verts[fi1]; // apex of T1 opposite F
    let e = tets.tets[t2].verts[fi2]; // apex of T2 opposite F

    let vd = tets.vertices[d];
    let ve = tets.vertices[e];

    // ── VALIDATION PHASE (no mutation) ──
    // Identify the coplanar base diagonal: the unique shared-face edge (f_i,f_j)
    // for which f_i, f_j, d, e are coplanar. The remaining vertex is f_k.
    let edge_pairs = [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)];
    let mut chosen: Option<(usize, usize, usize)> = None; // (f_i, f_j, f_k)
    for &(i, j, k) in &edge_pairs {
        let o = orient_3d(tets.vertices[f[i]], tets.vertices[f[j]], vd, ve);
        if o == 0.0 {
            if chosen.is_some() {
                // More than one coplanar edge ⇒ the whole region is degenerate;
                // not a clean 2-2 diagonal swap.
                return None;
            }
            chosen = Some((f[i], f[j], f[k]));
        }
    }
    let (f_i, f_j, f_k) = chosen?;

    // TOPOLOGICAL VALIDITY (TetGen's `getedge` guard): the new diagonal (d, e)
    // must NOT already exist as a mesh edge. If it does, the two new tets we are
    // about to create would coincide with / overlap an existing tet (a
    // non-simplicial result), so reject as a no-op. Without this guard the swap
    // can silently introduce overlapping tets (carve then overshoots).
    if edge_exists_in_tets(tets, d, e) {
        return None;
    }

    // Candidate new tets share the new face (f_k, d, e) and the new edge (d, e).
    let candidate_verts: [[usize; 4]; 2] = [[f_k, f_i, d, e], [f_k, f_j, d, e]];
    let mut oriented = [[0usize; 4]; 2];
    for (idx, nv) in candidate_verts.iter().enumerate() {
        let mut v = *nv;
        let o = orient_3d(
            tets.vertices[v[0]],
            tets.vertices[v[1]],
            tets.vertices[v[2]],
            tets.vertices[v[3]],
        );
        if o < 0.0 {
            v.swap(0, 1);
        } else if o == 0.0 {
            return None; // degenerate candidate → reject
        }
        oriented[idx] = v;
    }

    // Constraint guard: protect the active constraint against the new interior
    // edge (d, e) and the new interior face (f_k, d, e).
    if !check_flip_eligibility(tets, Some((d, e)), Some([f_k, d, e]), constraint) {
        return None;
    }

    // Gather external adjacency for the 4 non-shared faces of T1 and T2.
    let mut ext_links: Vec<(usize, [usize; 3])> = Vec::with_capacity(4);
    for fi in 0..4 {
        if fi == fi1 {
            continue;
        }
        ext_links.push((
            tets.tets[t1].adj[fi],
            delaunay3d::opposite_face(tets.tets[t1].verts, fi),
        ));
    }
    for fi in 0..4 {
        if fi == fi2 {
            continue;
        }
        ext_links.push((
            tets.tets[t2].adj[fi],
            delaunay3d::opposite_face(tets.tets[t2].verts, fi),
        ));
    }

    // TRANSACTIONAL VOLUME GUARD (see flip_volume_ok): the candidate tets must
    // tile the same finite volume as the tets they replace - reject otherwise.
    {
        let (vo, _) = flipvol_of(tets, &[t1, t2]);
        let vn = flipvol_of_candidates(tets, &oriented);
        if !flip_volume_ok(vo, vn) {
            return None;
        }
    }
    // ── COMMIT PHASE ──
    let _fv = if flipvol_enabled() {
        Some(flipvol_of(tets, &[t1, t2]))
    } else {
        None
    };
    tets.free_tet(t1);
    tets.free_tet(t2);

    let n1 = tets.alloc_tet(Tet {
        verts: oriented[0],
        adj: [usize::MAX; 4],
    });
    let n2 = tets.alloc_tet(Tet {
        verts: oriented[1],
        adj: [usize::MAX; 4],
    });
    if let Some(o) = _fv {
        flipvol_report("flip_2_to_2", o, tets, &[n1, n2]);
    }

    // Link the two new tets via their shared face (f_k, d, e).
    if let Some((s1, s2)) = delaunay3d::shared_face_indices(&tets.tets[n1], &tets.tets[n2]) {
        tets.tets[n1].adj[s1] = n2;
        tets.tets[n2].adj[s2] = n1;
    }

    // Restore external adjacency.
    let new_tets = [n1, n2];
    for (neighbor, face) in &ext_links {
        if *neighbor == usize::MAX || !tets.is_live(*neighbor) {
            continue;
        }
        for &ni in &new_tets {
            let mut linked = false;
            for fi in 0..4 {
                let nf = delaunay3d::opposite_face(tets.tets[ni].verts, fi);
                if delaunay3d::faces_match(&nf, face) {
                    tets.tets[ni].adj[fi] = *neighbor;
                    for nfi in 0..4 {
                        let neighbor_face =
                            delaunay3d::opposite_face(tets.tets[*neighbor].verts, nfi);
                        if delaunay3d::faces_match(&neighbor_face, face) {
                            tets.tets[*neighbor].adj[nfi] = ni;
                            break;
                        }
                    }
                    linked = true;
                    break;
                }
            }
            if linked {
                break;
            }
        }
    }

    Some([n1, n2])
}

/// The two vertices of tet `ti` other than the edge (p, q). Returns `None` if
/// the tet does not actually contain both p and q.
fn tet_ring_apices(tets: &Delaunay3D, ti: usize, p: usize, q: usize) -> Option<[usize; 2]> {
    let v = tets.tets[ti].verts;
    if !v.contains(&p) || !v.contains(&q) {
        return None;
    }
    let mut others = [usize::MAX; 2];
    let mut k = 0;
    for &x in &v {
        if x != p && x != q {
            if k < 2 {
                others[k] = x;
            }
            k += 1;
        }
    }
    if k != 2 {
        return None;
    }
    Some(others)
}

/// Walk the cyclic ring of tets around interior edge (p, q), returning them in
/// cyclic order together with each tet's "leading" apex (the ring vertex it
/// does NOT share with the previous tet). Hull tets (INFINITE apex) are
/// included - this is what makes the boundary-quad ring (2 interior + 2 hull
/// tets) tractable. Returns `None` if the ring is not a single closed cycle
/// (e.g. the edge is on the boundary of the live region in a way that does not
/// close), in which case the caller must not flip it.
///
/// The walk steps across faces (p, q, apex): from tet T with apices [x, y], the
/// face (p, q, x) leads to the neighbor on the other side, etc.
fn edge_ring_cycle(tets: &Delaunay3D, p: usize, q: usize) -> Option<Vec<usize>> {
    // Find a starting tet containing edge (p, q). O(degree) via the index when
    // active (only tets incident to `p` can hold the edge); full scan otherwise.
    let start = if tets.index_active() {
        tets.incident_tets(p)
            .iter()
            .map(|&ti| ti as usize)
            .find(|&i| tets.is_live(i) && tets.tets[i].verts.contains(&q))?
    } else {
        (0..tets.tets.len()).find(|&i| {
            tets.is_live(i) && {
                let v = tets.tets[i].verts;
                v.contains(&p) && v.contains(&q)
            }
        })?
    };

    let mut ring = vec![start];
    let mut cur = start;
    // The apex we just "entered through" - start by entering through one of the
    // start tet's two ring apices arbitrarily; we track the apex we exit on.
    let apices = tet_ring_apices(tets, start, p, q)?;
    let mut enter_apex = apices[0];

    loop {
        let v = tets.tets[cur].verts;
        let cur_apices = tet_ring_apices(tets, cur, p, q)?;
        // The exit apex is the OTHER ring vertex (the face (p,q,exit) leads on).
        let exit_apex = if cur_apices[0] == enter_apex {
            cur_apices[1]
        } else {
            cur_apices[0]
        };
        // The face we cross is (p, q, exit_apex). Find its index in `cur`.
        let mut next = usize::MAX;
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(v, fi);
            if face.contains(&p) && face.contains(&q) && face.contains(&exit_apex) {
                next = tets.tets[cur].adj[fi];
                break;
            }
        }
        if next == usize::MAX || !tets.is_live(next) {
            return None; // open ring - not flippable here
        }
        if next == start {
            return Some(ring); // closed the cycle
        }
        if ring.contains(&next) || ring.len() > tets.tets.len() {
            return None; // non-manifold / runaway
        }
        ring.push(next);
        // We entered `next` through face (p,q,exit_apex), so enter_apex = exit_apex.
        enter_apex = exit_apex;
        cur = next;
    }
}

/// Hull-aware edge-star reducer (TetGen's `flipnm`, specialized to the cases
/// boundary recovery needs).
///
/// Given an interior edge (p, q) whose star is a single closed ring of tets -
/// possibly including hull tets (the boundary-quad case: 2 interior + 2 hull
/// tets around the wrong diagonal) - reduce the ring so the edge (p, q) is
/// removed and replaced by the opposite diagonal of the link.
///
/// HULL HANDLING (the orientation rule for INFINITE): a hull tet
/// `[x, y, z, INFINITE]` represents exterior space; its finite face `[x, y, z]`
/// is oriented (by `Delaunay3D::oriented_hull_tet`) so interior points are on
/// the negative side. We never call `orient_3d` with INFINITE. Instead, when a
/// ring apex is INFINITE we (a) reject any candidate finite tet that would be
/// degenerate/inverted using only finite vertices, and (b) reconstruct hull
/// tets symbolically with the fixed convention. This mirrors TetGen's
/// `dummypoint` reconfiguration in `flip23`/`flip32`.
///
/// Currently implements the n == 4 reduction (the 4-to-4 flip, which subsumes
/// the 2-interior + 2-hull boundary-quad ring) and the n == 3 reduction (a
/// 3-to-2 flip). Both are transactional: validated fully before any tet is
/// freed, guarded by `constraint`, and a rejected flip is a true no-op.
///
/// Returns the indices of the new tets on success, or `None` (mesh unchanged)
/// otherwise.
pub fn flip_nm(
    tets: &mut Delaunay3D,
    p: usize,
    q: usize,
    constraint: &Constraint,
) -> Option<Vec<usize>> {
    let ring = edge_ring_cycle(tets, p, q)?;
    match ring.len() {
        3 => {
            // All three must be finite for a plain 3-to-2; if a hull tet is in
            // the ring it is handled by the n==4 path (a hull 3-ring around an
            // interior edge is not a configuration we flip here).
            if ring.iter().any(|&ti| is_hull_tet(&tets.tets[ti])) {
                return None;
            }
            let idx = [ring[0], ring[1], ring[2]];
            flip_3_to_2(tets, idx, p, q, constraint).map(|r| r.to_vec())
        }
        4 => flip_ring4(tets, p, q, &ring, constraint),
        _ => None,
    }
}

/// 4-to-4 edge flip (hull-aware): replace the 4-tet ring around edge (p, q)
/// with the 4-tet ring around the opposite link diagonal. Handles the case
/// where two of the ring tets are hull tets (the boundary-quad fix).
///
/// `ring` is the cyclic order of the four tets around (p, q) as produced by
/// [`edge_ring_cycle`]. The four ring apices, in cyclic order, are
/// `[w0, w1, w2, w3]` (one or two of them may be INFINITE for hull tets). The
/// new edge is the opposite diagonal `(w0, w2)` (or `(w1, w3)`), chosen so it
/// is a *finite* edge - we never create the edge through INFINITE.
fn flip_ring4(
    tets: &mut Delaunay3D,
    p: usize,
    q: usize,
    ring: &[usize],
    constraint: &Constraint,
) -> Option<Vec<usize>> {
    debug_assert_eq!(ring.len(), 4);

    // Recover the cyclic apex sequence [w0, w1, w2, w3] where ring[k] has ring
    // apices {w_k, w_{k+1}} (mod 4). Reconstruct it from adjacency: ring[k] and
    // ring[k+1] share a face (p, q, w) - that shared w is the apex between them.
    let mut w = [usize::MAX; 4];
    for k in 0..4 {
        let a = ring[k];
        let b = ring[(k + 1) % 4];
        let va = tets.tets[a].verts;
        // shared apex = the ring apex of `a` that is also a vertex of `b`.
        let aps = tet_ring_apices(tets, a, p, q)?;
        let shared = if tets.tets[b].verts.contains(&aps[0]) {
            aps[0]
        } else if tets.tets[b].verts.contains(&aps[1]) {
            aps[1]
        } else {
            return None;
        };
        // The "leading" apex of ring[k] is the one NOT shared with ring[k+1];
        // store it at w[k] so that ring[k] = (p, q, w[k], w[k+1]).
        let leading = if aps[0] == shared { aps[1] } else { aps[0] };
        let _ = va;
        w[k] = leading;
        // w[(k+1)%4] will be set as the shared apex by the next iteration's
        // `leading` of ring[k+1]; consistency is checked below.
    }
    // Validate the cyclic structure: ring[k] must contain exactly {w[k], w[k+1]}.
    for k in 0..4 {
        let aps = tet_ring_apices(tets, ring[k], p, q)?;
        let mut s = [aps[0], aps[1]];
        s.sort();
        let mut t = [w[k], w[(k + 1) % 4]];
        t.sort();
        if s != t {
            return None;
        }
    }

    // Choose the new diagonal: opposite pairs are (w0, w2) and (w1, w3). Pick
    // the pair with NO INFINITE (we must not create an edge through INFINITE).
    let diag_a = [w[0], w[2]];
    let diag_b = [w[1], w[3]];
    let (d0, d1, other0, other1) = if !diag_a.contains(&INFINITE) {
        (w[0], w[2], w[1], w[3])
    } else if !diag_b.contains(&INFINITE) {
        (w[1], w[3], w[0], w[2])
    } else {
        return None; // both diagonals hit INFINITE - cannot flip
    };
    if d0 == d1 {
        return None;
    }

    // TOPOLOGICAL VALIDITY (TetGen's `getedge` guard): the new edge (d0, d1)
    // must NOT already exist in the mesh. If it does, the four tets we are about
    // to create would coincide with / overlap existing tets (a non-simplicial
    // result), so the flip is invalid - reject as a no-op. This is exactly the
    // case where an interior point already split the link quad's far side.
    if edge_exists_in_tets(tets, d0, d1) {
        return None;
    }

    // The four new tets are the ring around the new edge (d0, d1):
    //   (p, q replaced) - actually the new ring around (d0,d1) has apices, in
    //   The link 4-cycle is [d0, other0, d1, other1] (i.e. w0, w1, w2, w3 with
    //   the new diagonal joining w0=d0 and w2=d1). Flipping to that diagonal
    //   splits the link quad into two triangles, (d0, other0, d1) and
    //   (d0, d1, other1); each triangle forms a bipyramid with the old edge
    //   endpoints p, q, giving the four new tets:
    //     (d0, other0, d1, p)   (d0, other0, d1, q)
    //     (d0, d1, other1, p)   (d0, d1, other1, q)
    // `other0`/`other1` and `p`/`q` may be finite; `d0`,`d1` are finite by
    // construction. A hull tet arises for each new tet whose `other*` is
    // INFINITE (exactly two of them when one link apex is INFINITE).
    let new_finite: [[usize; 4]; 4] = [
        [d0, other0, d1, p],
        [d0, other0, d1, q],
        [d0, d1, other1, p],
        [d0, d1, other1, q],
    ];

    // ── VALIDATION PHASE (no mutation) ──
    // Build oriented vertex lists. Finite tets: orient via orient_3d (reject
    // degenerate). Hull tets: reconstruct symbolically, and validate the finite
    // remainder is non-degenerate.
    let mut oriented: Vec<[usize; 4]> = Vec::with_capacity(4);
    for nv in &new_finite {
        let inf_count = nv.iter().filter(|&&x| x == INFINITE).count();
        if inf_count == 0 {
            let mut v = *nv;
            let o = orient_3d(
                tets.vertices[v[0]],
                tets.vertices[v[1]],
                tets.vertices[v[2]],
                tets.vertices[v[3]],
            );
            if o < 0.0 {
                v.swap(0, 1);
            } else if o == 0.0 {
                return None; // degenerate finite candidate → reject
            }
            oriented.push(v);
        } else if inf_count == 1 {
            // Hull tet: the three finite vertices form the boundary face. They
            // must be non-collinear (else the hull face is degenerate). Use the
            // interior-seed convention to orient.
            let finite: Vec<usize> = nv.iter().copied().filter(|&x| x != INFINITE).collect();
            let (a, b, c) = (finite[0], finite[1], finite[2]);
            // Reject collinear finite face (degenerate hull tet). orient with
            // the interior seed will be 0 iff the seed is coplanar - instead
            // test triangle area via cross product magnitude.
            let pa = tets.vertices[a];
            let pb = tets.vertices[b];
            let pc = tets.vertices[c];
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let nx = ab[1] * ac[2] - ab[2] * ac[1];
            let ny = ab[2] * ac[0] - ab[0] * ac[2];
            let nz = ab[0] * ac[1] - ab[1] * ac[0];
            if nx * nx + ny * ny + nz * nz == 0.0 {
                return None; // degenerate hull face
            }
            oriented.push(tets.oriented_hull_tet(a, b, c));
        } else {
            return None; // a candidate tet with two INFINITE - impossible here
        }
    }

    // Constraint guard: the new shared edge (d0, d1) must not pierce the active
    // constraint. (The new faces all contain d0,d1 plus a ring apex; the
    // critical new interior element is the diagonal edge.)
    if !check_flip_eligibility(tets, Some((d0, d1)), None, constraint) {
        return None;
    }

    // Gather external adjacency: every face of every ring tet that does NOT
    // contain edge (p, q) is an outer face (it is either shared with a tet
    // outside the ring, or - for hull tets - a hull face (p|q + INFINITE)).
    let ring_set: HashSet<usize> = ring.iter().copied().collect();
    let mut ext_links: Vec<(usize, [usize; 3])> = Vec::with_capacity(8);
    for &ti in ring {
        let v = tets.tets[ti].verts;
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(v, fi);
            if face.contains(&p) && face.contains(&q) {
                continue; // inner face (shared within the ring)
            }
            let nb = tets.tets[ti].adj[fi];
            if nb != usize::MAX && ring_set.contains(&nb) {
                continue; // safety: still an inner link
            }
            ext_links.push((nb, face));
        }
    }

    // TRANSACTIONAL VOLUME GUARD (see flip_volume_ok): the candidate tets must
    // tile the same finite volume as the tets they replace - reject otherwise.
    {
        let (vo, _) = flipvol_of(tets, ring);
        let vn = flipvol_of_candidates(tets, &oriented);
        if !flip_volume_ok(vo, vn) {
            return None;
        }
    }
    // ── COMMIT PHASE ──
    let _fv = if flipvol_enabled() {
        Some(flipvol_of(tets, ring))
    } else {
        None
    };
    for &ti in ring {
        tets.free_tet(ti);
    }
    let mut new_indices: Vec<usize> = Vec::with_capacity(4);
    for v in &oriented {
        new_indices.push(tets.alloc_tet(Tet {
            verts: *v,
            adj: [usize::MAX; 4],
        }));
    }
    if let Some(o) = _fv {
        flipvol_report("flip_ring4", o, tets, &new_indices);
    }

    // Inter-new-tet adjacency.
    for i in 0..new_indices.len() {
        for j in (i + 1)..new_indices.len() {
            let ni = new_indices[i];
            let nj = new_indices[j];
            if let Some((si, sj)) = delaunay3d::shared_face_indices(&tets.tets[ni], &tets.tets[nj])
            {
                tets.tets[ni].adj[si] = nj;
                tets.tets[nj].adj[sj] = ni;
            }
        }
    }

    // External adjacency: relink each outer face to its neighbor.
    relink_external(tets, &new_indices, &ext_links);

    Some(new_indices)
}

/// The cyclic apex sequence `[w_0 .. w_{n-1}]` of the ring of edge (p, q):
/// `ring[k]` is the tet whose two ring apices are `{w_k, w_{k+1 mod n}}`. Hull
/// tets contribute an `INFINITE` apex. Returns `None` if the ring is not a clean
/// cyclic chain (caller must not flip it). This is the n-generalisation of the
/// reconstruction inside [`flip_ring4`].
fn ring_apex_cycle(tets: &Delaunay3D, p: usize, q: usize, ring: &[usize]) -> Option<Vec<usize>> {
    let n = ring.len();
    if n < 3 {
        return None;
    }
    let mut w = vec![usize::MAX; n];
    for k in 0..n {
        let ta = ring[k];
        let tb = ring[(k + 1) % n];
        let aps = tet_ring_apices(tets, ta, p, q)?;
        // shared apex = the ring apex of `ta` that is also a vertex of `tb`.
        let shared = if tets.tets[tb].verts.contains(&aps[0]) {
            aps[0]
        } else if tets.tets[tb].verts.contains(&aps[1]) {
            aps[1]
        } else {
            return None;
        };
        // The "leading" apex of ring[k] is the one NOT shared with ring[k+1].
        w[k] = if aps[0] == shared { aps[1] } else { aps[0] };
    }
    // Validate: ring[k] must carry exactly {w[k], w[k+1]}.
    for k in 0..n {
        let aps = tet_ring_apices(tets, ring[k], p, q)?;
        let mut s = [aps[0], aps[1]];
        s.sort_unstable();
        let mut t = [w[k], w[(k + 1) % n]];
        t.sort_unstable();
        if s != t {
            return None;
        }
    }
    // Validate: the link polygon must be SIMPLE, i.e. every finite apex occurs
    // exactly once. A repeated apex means the cycle is PINCHED (a figure-eight
    // through that vertex), which is not a polygon, so the bipyramid rebuild in
    // `flip_ring_general` is not defined on it: its polygon DP would form a
    // "triangle" with a repeated vertex, a degenerate simplex that no
    // orientation predicate can sign, and `orient_3d_sos` would (correctly)
    // panic on it rather than invent a sign.
    //
    // INFINITE is exempt: the hull pair legitimately contributes it twice, and
    // the DP never puts two INFINITE apices in one fan triangle.
    //
    // The pinch is not created here. It originates in the coplanar-region
    // rebuild, which replays a 2-D flip log as a stack of zero-volume tets, one
    // per flip. Sequential constrained recovery flips a diagonal out and later
    // flips it back, and that inverse pair replays as two COINCIDENT tets with
    // identical vertex sets, leaving in-plane faces owned by four tets instead
    // of two. That is tracked separately as a redesign of the sandwich
    // construction; rejecting the ring here is what keeps the corruption away
    // from a predicate that cannot sign it.
    //
    // Ported from fusion-energy/cad-to-dagmc-mesher#158 (its issue #157), where
    // this aborted a two-solid assembly sharing a curved interface.
    for k in 0..n {
        if w[k] != INFINITE && w[..k].contains(&w[k]) {
            return None; // pinched link polygon: not a clean cyclic chain
        }
    }
    Some(w)
}

/// SoS-resolved orientation of the lifted tet `(w_i, w_j, w_k, apex)` where the
/// link triangle `(w_i, w_j, w_k)` is part of the link-polygon retriangulation
/// of edge (p, q) and `apex` is p or q. Returns the SIGNED volume sign with the
/// coplanar (zero-volume) degeneracy resolved DETERMINISTICALLY by global vertex
/// index via [`orient_3d_sos`] - so flat-face slivers get a consistent virtual
/// orientation and the in-plane diagonal choice is well-defined. INFINITE inputs
/// are never passed here (hull triangles are validated by finite-area instead).
fn lifted_orient_sos(tets: &Delaunay3D, wi: usize, wj: usize, wk: usize, apex: usize) -> f64 {
    orient_3d_sos(
        tets.vertices[wi],
        wi,
        tets.vertices[wj],
        wj,
        tets.vertices[wk],
        wk,
        tets.vertices[apex],
        apex,
    )
}

/// Hull-aware GENERAL edge remover for a ring of size n >= 4 - the
/// n-generalisation of [`flip_ring4`]/[`flip_nm`], realised as a single cavity
/// rebuild (Shewchuk's edge removal). The ring tets of edge (p, q) fill the
/// bipyramid over the link polygon `W = [w_0 .. w_{n-1}]` (cyclic apices; one or
/// two may be INFINITE for the hull pair). Removing (p, q) means re-triangulating
/// the link polygon into n-2 triangles and lifting each link triangle
/// `(w_i, w_j, w_k)` to the two tets `(w_i, w_j, w_k, p)` and `(.., q)`.
///
/// A pure 2-3 ear-clip reducer STALLS on a reflex link polygon (LShaped's flat
/// reflex faces: most ears are non-convex, so no single 2-3 flip is valid). We
/// instead pick the link triangulation directly by a polygon DP that MAXIMISES
/// the minimum lifted-tet volume, rejecting any fan whose triangle lifts to an
/// inverted/overlapping tet. COPLANAR (zero-volume) link triangles - the flat-
/// face slivers - are resolved with [`orient_3d_sos`] so the consistent virtual
/// orientation makes the in-plane diagonal swap well-defined and deterministic.
/// The hull pair is handled symbolically (a link triangle through INFINITE lifts
/// to two hull tets via [`Delaunay3D::oriented_hull_tet`]).
///
/// PROTECTED: rejects the rebuild if it would destroy an already-recovered
/// boundary segment/face (`protect`), and guards the new interior diagonals
/// against crossing the active constraint. TRANSACTIONAL: the whole rebuild is
/// validated before any old tet is freed; on any rejection the mesh is left
/// byte-for-byte unchanged and `false` is returned.
///
/// DETERMINISM: the DP is over the fixed cyclic order and SoS ties break by
/// global index, so identical input → identical rebuild.
fn flip_ring_general(
    tets: &mut Delaunay3D,
    p: usize,
    q: usize,
    protect: &ProtectedBoundary,
) -> bool {
    if !edge_exists_in_tets(tets, p, q) {
        return true;
    }
    if protect.is_protected_edge(p, q) {
        return false; // (p, q) is itself a recovered boundary segment
    }
    let ring = match edge_ring_cycle(tets, p, q) {
        Some(r) => r,
        None => return false,
    };
    let n = ring.len();
    if n < 4 {
        // Let the existing n==3 path handle it (with protection).
        if ring_flip_destroys_protected(tets, p, q, protect) {
            return false;
        }
        return flip_nm(tets, p, q, &protect.active).is_some() && !edge_exists_in_tets(tets, p, q);
    }
    // Bound the polygon DP (O(n^3)) - the flat fans are small (n <= ~12).
    if n > 24 {
        return false;
    }
    let w = match ring_apex_cycle(tets, p, q, &ring) {
        Some(w) => w,
        None => return false,
    };
    // At most one INFINITE link vertex can sit in a single fan triangle; two
    // INFINITE link vertices are allowed only if they are ADJACENT in the cycle
    // (the hull pair shares the hull face (p,q,INF) - handled by never putting
    // both in one triangle). If more than two are INFINITE the configuration is
    // not a clean hull fan; bail.
    if w.iter().filter(|&&x| x == INFINITE).count() > 2 {
        return false;
    }

    // No new interior diagonal of the link polygon may already exist as a mesh
    // edge (that would make the rebuilt tets overlap existing ones), and none may
    // cross the active constraint. We test candidate diagonals inside the DP.
    //
    // The validity of a fan triangle (w_i, w_j, w_k): both lifted tets
    // (w_i,w_j,w_k,p) and (..,q) must be NON-INVERTED and CONSISTENTLY oriented
    // (p and q on opposite sides of the link-triangle plane - they are, since
    // the bipyramid is convex across each link triangle). With SoS the coplanar
    // case gets a definite sign. A triangle through INFINITE lifts to two hull
    // tets; we only require its finite part to have non-zero area.
    let tri_min_vol = |wi: usize, wj: usize, wk: usize| -> Option<f64> {
        let inf = (wi == INFINITE) as u8 + (wj == INFINITE) as u8 + (wk == INFINITE) as u8;
        if inf >= 2 {
            return None; // a fan triangle may contain at most one INFINITE
        }
        if inf == 1 {
            // Hull triangle (w_i, w_j, w_k) through INFINITE lifts to the two hull
            // tets (f0, f1, p, INF) and (f0, f1, q, INF) where {f0, f1} are the two
            // FINITE link verts. Validity = both finite hull FACES (f0, f1, p) and
            // (f0, f1, q) have non-zero AREA (a real boundary triangle). These
            // faces lie in the flat-face plane, so their tet volume against any
            // in-plane point is zero - measure triangle area directly (NOT tet
            // volume).
            let fin: Vec<usize> = [wi, wj, wk]
                .into_iter()
                .filter(|&x| x != INFINITE)
                .collect();
            let (f0, f1) = (fin[0], fin[1]);
            let tri_area2 = |x: usize, y: usize, z: usize| -> f64 {
                let a = tets.vertices[x];
                let b = tets.vertices[y];
                let c = tets.vertices[z];
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let nx = ab[1] * ac[2] - ab[2] * ac[1];
                let ny = ab[2] * ac[0] - ab[0] * ac[2];
                let nz = ab[0] * ac[1] - ab[1] * ac[0];
                nx * nx + ny * ny + nz * nz
            };
            let ap = tri_area2(f0, f1, p);
            let aq = tri_area2(f0, f1, q);
            if ap == 0.0 || aq == 0.0 {
                return None; // a degenerate (collinear) hull face
            }
            // A valid hull triangle does not constrain interior quality, so it is
            // "free" in the max-min objective (does not drag the min down). The
            // INFINITE link vertex MUST live in some triangle, so penalising hull
            // triangles would wrongly make the DP avoid them.
            return Some(f64::INFINITY);
        }
        // Finite fan triangle: both lifts must be valid + opposite-signed.
        let sp = lifted_orient_sos(tets, wi, wj, wk, p);
        let sq = lifted_orient_sos(tets, wi, wj, wk, q);
        if sp == 0.0 || sq == 0.0 {
            return None; // SoS could not be reached (should not happen)
        }
        if (sp > 0.0) == (sq > 0.0) {
            return None; // p, q on same side ⇒ inverted lift
        }
        Some((sp.abs()).min(sq.abs()))
    };

    // ── Polygon DP over the cyclic link order [w_0 .. w_{n-1}] ──
    // dp[i][j] = best achievable min-volume triangulating the chain w_i..w_j
    // (the polygon edge w_i-w_j closing it), with split[i][j] the chosen apex k.
    const NEG: f64 = f64::NEG_INFINITY;
    let mut dp = vec![vec![NEG; n]; n];
    let mut split = vec![vec![usize::MAX; n]; n];
    for i in 0..n {
        if i + 1 < n {
            dp[i][i + 1] = f64::INFINITY; // adjacent: empty chain (no triangle)
        }
    }
    for len in 2..n {
        for i in 0..(n - len) {
            let j = i + len;
            let mut best = NEG;
            let mut best_k = usize::MAX;
            for k in (i + 1)..j {
                if dp[i][k] == NEG || dp[k][j] == NEG {
                    continue;
                }
                let tv = match tri_min_vol(w[i], w[k], w[j]) {
                    Some(v) => v,
                    None => continue,
                };
                // New interior diagonals introduced by this triangle: (w_i,w_k)
                // if k != i+1, and (w_k,w_j) if j != k+1. They must not already
                // exist as mesh edges nor cross the active constraint.
                let mut ok = true;
                for (di, dj, adj_along) in [(w[i], w[k], k == i + 1), (w[k], w[j], j == k + 1)] {
                    if adj_along {
                        continue; // a polygon edge, not a new diagonal
                    }
                    if di == INFINITE || dj == INFINITE {
                        ok = false; // cannot create an edge through INFINITE
                        break;
                    }
                    if edge_exists_in_tets(tets, di, dj)
                        || edge_crosses_constraint(tets, di, dj, &protect.active)
                    {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let cand = dp[i][k].min(dp[k][j]).min(tv);
                if cand > best {
                    best = cand;
                    best_k = k;
                }
            }
            dp[i][j] = best;
            split[i][j] = best_k;
        }
    }
    if dp[0][n - 1] == NEG {
        return false; // no valid link triangulation exists - fall through
    }

    // Reconstruct the chosen link triangles.
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(n - 2);
    let mut stack = vec![(0usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let k = split[i][j];
        if k == usize::MAX {
            return false; // inconsistent (shouldn't happen given dp != NEG)
        }
        tris.push([w[i], w[k], w[j]]);
        stack.push((i, k));
        stack.push((k, j));
    }

    // Build the new tets (two per link triangle): finite triangles lift to two
    // finite tets; a triangle through INFINITE lifts to two hull tets.
    let mut oriented: Vec<[usize; 4]> = Vec::with_capacity(2 * tris.len());
    for tri in &tris {
        let inf = tri.iter().filter(|&&x| x == INFINITE).count();
        if inf == 1 {
            let fin: Vec<usize> = tri.iter().copied().filter(|&x| x != INFINITE).collect();
            let (f0, f1) = (fin[0], fin[1]);
            // Two hull tets: finite faces (f0, f1, p) and (f0, f1, q).
            oriented.push(tets.oriented_hull_tet(f0, f1, p));
            oriented.push(tets.oriented_hull_tet(f0, f1, q));
        } else {
            for apex in [p, q] {
                let mut v = [tri[0], tri[1], tri[2], apex];
                let o = orient_3d(
                    tets.vertices[v[0]],
                    tets.vertices[v[1]],
                    tets.vertices[v[2]],
                    tets.vertices[v[3]],
                );
                if o < 0.0 {
                    v.swap(0, 1);
                } else if o == 0.0 {
                    // Coplanar flat sliver: orient by SoS (consistent virtual
                    // orientation). A zero-volume tet is acceptable here - the
                    // ring already contained such flat slivers; we only re-route
                    // the in-plane diagonal. Orient so SoS sign is positive.
                    let s = lifted_orient_sos(tets, tri[0], tri[1], tri[2], apex);
                    if s < 0.0 {
                        v.swap(0, 1);
                    }
                }
                oriented.push(v);
            }
        }
    }

    // PROTECTION: the rebuild removes edge (p, q) and every interior face of the
    // old ring that contains (p, q). Reject if any such face is a protected
    // boundary triangle, or (p, q) is a protected segment (checked above).
    if ring_flip_destroys_protected(tets, p, q, protect) {
        return false;
    }

    // Gather external adjacency: every face of every ring tet NOT containing edge
    // (p, q) is an outer face of the cavity (shared with a tet outside the ring,
    // or a hull face). These are exactly the faces the rebuilt tets must re-link
    // to. (The faces containing (p, q) are interior to the cavity and disappear.)
    let ring_set: HashSet<usize> = ring.iter().copied().collect();
    let mut ext_links: Vec<(usize, [usize; 3])> = Vec::with_capacity(2 * n);
    for &ti in &ring {
        let v = tets.tets[ti].verts;
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(v, fi);
            if face.contains(&p) && face.contains(&q) {
                continue; // interior cavity face
            }
            let nb = tets.tets[ti].adj[fi];
            if nb != usize::MAX && ring_set.contains(&nb) {
                continue;
            }
            ext_links.push((nb, face));
        }
    }

    // TRANSACTIONAL VOLUME GUARD (see flip_volume_ok): the candidate tets must
    // tile the same finite volume as the tets they replace - reject otherwise.
    {
        let (vo, _) = flipvol_of(tets, &ring);
        let vn = flipvol_of_candidates(tets, &oriented);
        if !flip_volume_ok(vo, vn) {
            return false;
        }
    }
    // ── COMMIT PHASE (validated; now mutate) ──
    let _fv = if flipvol_enabled() {
        Some(flipvol_of(tets, &ring))
    } else {
        None
    };
    for &ti in &ring {
        tets.free_tet(ti);
    }
    let mut new_indices: Vec<usize> = Vec::with_capacity(oriented.len());
    for v in &oriented {
        new_indices.push(tets.alloc_tet(Tet {
            verts: *v,
            adj: [usize::MAX; 4],
        }));
    }
    if let Some(o) = _fv {
        flipvol_report("flip_ring_general", o, tets, &new_indices);
    }
    // Inter-new-tet adjacency.
    for i in 0..new_indices.len() {
        for j in (i + 1)..new_indices.len() {
            let ni = new_indices[i];
            let nj = new_indices[j];
            if let Some((si, sj)) = delaunay3d::shared_face_indices(&tets.tets[ni], &tets.tets[nj])
            {
                tets.tets[ni].adj[si] = nj;
                tets.tets[nj].adj[sj] = ni;
            }
        }
    }
    // External adjacency.
    relink_external(tets, &new_indices, &ext_links);

    !edge_exists_in_tets(tets, p, q)
}

/// Relink a set of freshly-created tets to their external neighbors across the
/// given (neighbor, face) links, fixing the neighbor's back-pointer too. Faces
/// may be hull faces (contain INFINITE); `faces_match` sorts and compares
/// indices so INFINITE is matched like any other index.
fn relink_external(
    tets: &mut Delaunay3D,
    new_indices: &[usize],
    ext_links: &[(usize, [usize; 3])],
) {
    for (neighbor, face) in ext_links {
        // Find the new tet that owns this face and set its adj slot.
        for &ni in new_indices {
            let mut linked = false;
            for fi in 0..4 {
                let nf = delaunay3d::opposite_face(tets.tets[ni].verts, fi);
                if delaunay3d::faces_match(&nf, face) {
                    tets.tets[ni].adj[fi] = *neighbor;
                    if *neighbor != usize::MAX && tets.is_live(*neighbor) {
                        for nfi in 0..4 {
                            let neighbor_face =
                                delaunay3d::opposite_face(tets.tets[*neighbor].verts, nfi);
                            if delaunay3d::faces_match(&neighbor_face, face) {
                                tets.tets[*neighbor].adj[nfi] = ni;
                                break;
                            }
                        }
                    }
                    linked = true;
                    break;
                }
            }
            if linked {
                break;
            }
        }
    }
}

/// Recover a missing boundary face (a, b, c) by flipping the WRONG diagonal
/// edge of its coplanar quad across the full ring of tets around that edge -
/// including hull tets.
///
/// The quad is {a, b, c, x} for some 4th corner `x`. The wanted face splits the
/// quad on the wanted diagonal (d0, d1) - two of {a, b, c}. The OTHER (wrong)
/// diagonal connects the third face vertex `w` to the 4th corner `x`, and is
/// the edge currently present in the mesh. That wrong-diagonal edge (w, x) is
/// shared by a ring of tets; on a cube boundary the ring is 2 interior + 2 hull
/// tets. Flipping (w, x) → (d0, d1) with the hull-aware [`flip_nm`] makes both
/// the interior and the hull adopt the wanted diagonal, so the carve no longer
/// leaks. Returns true iff (a, b, c) exists afterwards. A failed flip is a
/// no-op (transactional), so this never regresses the mesh.
fn recover_face_by_ring_flip(tets: &mut Delaunay3D, a: usize, b: usize, c: usize) -> bool {
    let face = [a, b, c];
    // Deterministically ordered candidate wrong-diagonal edges (w, x):
    //   - w is one of the three wanted-face vertices (the "third" vertex, not on
    //     the wanted diagonal),
    //   - x is the 4th quad corner: a neighbor of w connected by an edge whose
    //     4-tet ring's opposite diagonal is exactly the other two face vertices.
    // We try each (w → wanted diagonal = the other two face verts).
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for &w in &face {
        let diag: Vec<usize> = face.iter().copied().filter(|&v| v != w).collect();
        let (d0, d1) = (diag[0], diag[1]);
        // The wrong diagonal starts at w and goes to some 4th corner x. Find x
        // by scanning edges (w, x) that currently exist with a 4-tet ring whose
        // opposite link diagonal is (d0, d1).
        for x in 0..tets.vertices.len() {
            if x == a || x == b || x == c {
                continue;
            }
            if !edge_exists_in_tets(tets, w, x) {
                continue;
            }
            // Quick geometric filter: the wanted diagonal (d0,d1) and the wrong
            // diagonal (w,x) must cross in their interior (they are the two
            // diagonals of the same planar quad).
            if !segments_cross_interior(&tets.vertices, [d0, d1], [w, x]) {
                continue;
            }
            candidates.push((w.min(x), w.max(x)));
        }
        // Dedup per-w handled by the global sort+dedup below.
        let _ = (d0, d1);
    }
    candidates.sort();
    candidates.dedup();

    for (p, q) in candidates {
        if !edge_exists_in_tets(tets, p, q) {
            continue;
        }
        // The flip must not pierce the wanted boundary face; guard with it.
        if flip_nm(tets, p, q, &Constraint::Face(face)).is_some()
            && face_exists_in_tets(tets, a, b, c)
        {
            return true;
        }
    }
    false
}

/// Recover a missing boundary face (a, b, c) when it is blocked by the WRONG
/// diagonal of a coplanar quad - the classic cube-face failure.
///
/// The two tets that carry the wrong diagonal share an interior face
/// F = (g, h, m): the off-plane apex `m` plus one endpoint of the wrong base
/// diagonal (one of g, h). Their apices opposite F are `d` and `e`, the two
/// vertices of the coplanar base that are NOT on the wrong diagonal. A 2-2
/// diagonal swap of these two tets exchanges the wrong base diagonal for the
/// new diagonal (d, e), and the new tet (f_k, f_i, d, e) carries the wanted
/// boundary triangle.
///
/// The wanted face (a, b, c) is exactly that new boundary triangle when:
///   * both apices d and e belong to {a, b, c} (they become the new diagonal,
///     which is an edge of the wanted triangle), and
///   * the third wanted vertex is a shared-face vertex (a wrong-diagonal
///     endpoint), and
///   * all of {a, b, c} lie in the coplanar base plane.
///
/// [`flip_2_to_2`] re-validates the coplanar-quad condition exactly and only
/// commits if it holds.
///
/// Returns true iff the face exists after the swap. Boundary vertices are
/// never moved. Determinism: candidate pairs are tried in an order keyed on
/// the (sorted) new diagonal then the sorted shared-face vertices, so the
/// outcome is independent of the order tets occupy in the arena (and hence of
/// the input point ordering).
///
/// `use_ring_flip`: when true, first try the hull-aware wrong-diagonal EDGE
/// flip ([`recover_face_by_ring_flip`]). This is required by the CONFORMING
/// carve path so the hull adopts the wanted diagonal too (otherwise the carve
/// flood leaks). The LEGACY filter path passes `false`: it discards exterior
/// tets by an inside-test rather than carving, so it does not need the hull
/// kept consistent - and applying the edge flip there changes which tets the
/// downstream overlap-removal/filter sees, perturbing volumes on curved models.
fn recover_face_by_diagonal_swap(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    c: usize,
    use_ring_flip: bool,
) -> bool {
    if face_exists_in_tets(tets, a, b, c) {
        return true;
    }

    // First try the HULL-AWARE ring flip: when the boundary quad's wrong
    // diagonal is shared by a 4-tet ring spanning the interior AND the hull
    // (the cube-face case - 2 interior + 2 hull tets), a plain 2-2 swap only
    // fixes the interior pair and leaves the hull on the old diagonal (the
    // carve then leaks). Flipping the wrong-diagonal EDGE with `flip_nm` makes
    // both the interior and the hull adopt the wanted diagonal.
    if use_ring_flip && recover_face_by_ring_flip(tets, a, b, c) {
        return true;
    }

    let face = [a, b, c];

    // Gather candidate (t1, t2) pairs together with a deterministic sort key:
    // (sorted new diagonal (d,e), sorted shared face F).
    let mut candidates: Vec<([usize; 2], [usize; 3], usize, usize)> = Vec::new();

    // Every qualifying t1 contains apex d ∈ {a,b,c}, and its partner t2
    // contains apex e ∈ {a,b,c} - BOTH pair members lie in the vertex stars of
    // the wanted face, so it suffices to iterate the star union (O(degree) via
    // the index; full scan when inactive). The `t2 <= t1` skip still
    // deduplicates pairs because both members are iterated.
    let star_t1: Vec<usize> = if tets.index_active() {
        let mut v: Vec<usize> = [a, b, c]
            .iter()
            .flat_map(|&t| tets.incident_tets(t).iter().map(|&x| x as usize))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    } else {
        (0..tets.tets.len()).collect()
    };
    for t1 in star_t1 {
        if !tets.is_live(t1) || is_hull_tet(&tets.tets[t1]) {
            continue;
        }
        let v1 = tets.tets[t1].verts;
        for fi1 in 0..4 {
            let t2 = tets.tets[t1].adj[fi1];
            if t2 == usize::MAX || t2 <= t1 || !tets.is_live(t2) || is_hull_tet(&tets.tets[t2]) {
                continue; // each unordered pair considered once (t1 < t2)
            }
            let f = delaunay3d::opposite_face(v1, fi1);
            // Apex of T1 opposite the shared face F.
            let d = v1[fi1];
            // Apex of T2 opposite F.
            let fi2 = match delaunay3d::shared_face_indices(&tets.tets[t1], &tets.tets[t2]) {
                Some((_, fj)) => fj,
                None => continue,
            };
            let e = tets.tets[t2].verts[fi2];

            // Both apices must be wanted-face vertices (they become the new
            // diagonal = an edge of the wanted triangle).
            if !face.contains(&d) || !face.contains(&e) || d == e {
                continue;
            }
            // The third wanted vertex must be a shared-face vertex (a
            // wrong-diagonal endpoint). flip_2_to_2 re-checks coplanarity of
            // the base quad exactly and only commits if it holds.
            let w = *face.iter().find(|&&x| x != d && x != e).unwrap();
            if !f.contains(&w) {
                continue;
            }

            let mut diag = [d, e];
            diag.sort();
            let mut sf = f;
            sf.sort();
            candidates.push((diag, sf, t1, t2));
        }
    }

    if candidates.is_empty() {
        return false;
    }

    // Deterministic ordering independent of arena layout / input ordering.
    candidates.sort();
    candidates.dedup();

    // Guard the swap with the wanted face as the active constraint so the new
    // interior elements cannot pierce a boundary face. (The new edge/face only
    // touch (a,b,c) at shared vertices, so an eligible swap is never blocked
    // by its own target.)
    for (_diag, _sf, t1, t2) in candidates {
        if !tets.is_live(t1) || !tets.is_live(t2) {
            continue;
        }
        if flip_2_to_2(tets, t1, t2, &Constraint::Face(face)).is_some()
            && face_exists_in_tets(tets, a, b, c)
        {
            return true;
        }
    }

    false
}

/// General flip-based boundary FACET recovery (blueprint step 2b).
///
/// The coplanar-quad recovery ([`recover_face_by_diagonal_swap`] /
/// [`recover_face_by_ring_flip`]) only handles the case where the wanted facet
/// and the obstructing edge are the two diagonals of a single planar quad. On
/// CURVED surfaces (Cone, Capsule, Hemisphere, TruncatedCone), thin/annular
/// walls (Pipe, ThinWalledCylinder) and non-convex solids (LShaped) a missing
/// boundary facet (a, b, c) is instead pierced by an *arbitrary* interior mesh
/// edge (d, e) - that piercing edge is exactly why the facet is absent.
///
/// This recovers the facet by repeatedly removing such an obstruction: find an
/// interior edge (d, e) that pierces the interior of triangle (a, b, c) and
/// reduce its star with the hull-aware [`flip_nm`], guarded by
/// `Constraint::Face([a, b, c])` so no flip may itself cross the wanted facet
/// (which would push the mesh further from the boundary). Repeat until the
/// facet is present or no piercing edge can be flipped.
///
/// Boundary vertices are never moved and no points are added. Every flip is
/// transactional, so a failed attempt is a true no-op - this never regresses
/// the mesh. Returns true iff (a, b, c) exists afterwards. A `false` return
/// means a genuine Schoenhardt/Steiner case (every piercing edge is
/// unflippable) that a later phase must handle.
///
/// The facet's three EDGES must be present first. The conforming path runs a
/// global edge-recovery pass (`recover_edges`) before face recovery; a facet
/// whose edge is still missing afterwards is on a region edge recovery could not
/// resolve, so we bail cheaply rather than re-running the expensive edge
/// recovery per facet.
fn recover_face_by_edge_flips(tets: &mut Delaunay3D, a: usize, b: usize, c: usize) -> bool {
    if face_exists_in_tets(tets, a, b, c) {
        return true;
    }

    // The facet's three EDGES must be present before it can close. The global
    // `recover_edges` pass (run before face recovery in `mesh_volume_conforming`)
    // already attempted every boundary edge; a facet whose edge is still missing
    // is on a region the edge recovery could not resolve, so the facet is not
    // flip-recoverable here either. Bail cheaply rather than re-running the
    // (expensive) edge recovery per facet - that would scan the whole mesh
    // hundreds of times on thin-walled models that ultimately fall back anyway.
    if !edge_exists_in_tets(tets, a, b)
        || !edge_exists_in_tets(tets, b, c)
        || !edge_exists_in_tets(tets, c, a)
    {
        return false;
    }

    let face = [a, b, c];
    // Bound the work: each successful flip removes one obstruction, but a flip
    // can also introduce a new (closer) piercing edge, so allow a generous cap.
    let max_iters = 200usize;

    for _ in 0..max_iters {
        if face_exists_in_tets(tets, a, b, c) {
            return true;
        }

        // Collect every interior edge that pierces the interior of (a, b, c).
        // LOCALIZED (issue #30 perf): a piercing edge belongs to a tet whose
        // closure intersects the closed triangle, and that set (the triangle
        // "pipe") is face-connected and reaches the triangle's rim - so a BFS
        // from the vertex stars of {a, b, c}, admitting only tets whose AABB
        // overlaps the (padded) triangle AABB, visits a superset of it in
        // O(local) instead of the previous full O(#tets) scan PER ITERATION
        // (the dominant cost of the facet pass on fine meshes). Falls back to
        // the full scan if the index is inactive or the local region exceeds a
        // sanity cap. Deterministic output (sorted, deduped) as before.
        let mut piercing: Vec<[usize; 2]> = Vec::new();
        let pa = tets.vertices[a];
        let pb = tets.vertices[b];
        let pc = tets.vertices[c];
        let mut lo = [0.0f64; 3];
        let mut hi = [0.0f64; 3];
        for k in 0..3 {
            lo[k] = pa[k].min(pb[k]).min(pc[k]);
            hi[k] = pa[k].max(pb[k]).max(pc[k]);
            let pad = (hi[k] - lo[k]).max(1e-12) * 1e-6;
            lo[k] -= pad;
            hi[k] += pad;
        }
        let tet_overlaps = |tets: &Delaunay3D, i: usize| -> bool {
            let v = tets.tets[i].verts;
            let mut tlo = [f64::INFINITY; 3];
            let mut thi = [f64::NEG_INFINITY; 3];
            for &x in &v {
                if x == INFINITE {
                    return true; // hull tets span outward - treat as overlapping
                }
                for k in 0..3 {
                    tlo[k] = tlo[k].min(tets.vertices[x][k]);
                    thi[k] = thi[k].max(tets.vertices[x][k]);
                }
            }
            (0..3).all(|k| thi[k] >= lo[k] && tlo[k] <= hi[k])
        };
        const LOCAL_CAP: usize = 50_000;
        let mut candidates: Vec<usize> = Vec::new();
        if tets.index_active() {
            let mut stack: Vec<usize> = Vec::new();
            let mut seen: HashSet<usize> = HashSet::default();
            for &t in &[a, b, c] {
                for &ti in tets.incident_tets(t) {
                    let ti = ti as usize;
                    if tets.is_live(ti) && seen.insert(ti) {
                        stack.push(ti);
                    }
                }
            }
            while let Some(ti) = stack.pop() {
                candidates.push(ti);
                if candidates.len() > LOCAL_CAP {
                    break;
                }
                for fi in 0..4 {
                    let nb = tets.tets[ti].adj[fi];
                    if nb != usize::MAX
                        && tets.is_live(nb)
                        && !seen.contains(&nb)
                        && tet_overlaps(tets, nb)
                    {
                        seen.insert(nb);
                        stack.push(nb);
                    }
                }
            }
        }
        if !tets.index_active() || candidates.len() > LOCAL_CAP {
            candidates = (0..tets.tets.len()).collect();
        }
        for i in candidates {
            if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
                continue;
            }
            let v = tets.tets[i].verts;
            for ei in 0..6 {
                let (d, e) = tet_edge(v, ei);
                if d == INFINITE || e == INFINITE {
                    continue;
                }
                // The piercing edge must not share a vertex with the facet
                // (an edge incident to the facet cannot pierce its interior).
                if d == a || d == b || d == c || e == a || e == b || e == c {
                    continue;
                }
                if segment_intersects_triangle(
                    tets.vertices[d],
                    tets.vertices[e],
                    tets.vertices[a],
                    tets.vertices[b],
                    tets.vertices[c],
                ) {
                    piercing.push([d.min(e), d.max(e)]);
                }
            }
        }
        if piercing.is_empty() {
            // No interior edge pierces the facet, yet it is still absent. Nothing
            // local can be flipped to introduce it here.
            return false;
        }
        piercing.sort();
        piercing.dedup();

        // Try to reduce the star of each piercing edge. The Face constraint
        // forbids any flip that would itself cross the wanted facet.
        let mut made_progress = false;
        for [d, e] in piercing {
            if !edge_exists_in_tets(tets, d, e) {
                continue; // an earlier flip this pass already removed it
            }
            if flip_nm(tets, d, e, &Constraint::Face(face)).is_some() {
                made_progress = true;
                break; // re-scan: the mesh (and the set of piercing edges) changed
            }
        }

        if !made_progress {
            // Every piercing edge is unflippable (a genuine Schoenhardt/Steiner
            // case) - leave the mesh untouched for a later phase / legacy gate.
            return false;
        }
    }

    face_exists_in_tets(tets, a, b, c)
}

/// The obstruction the ray a→b first meets while marching out of vertex `a`
/// (TetGen's `finddirection` result). Each variant carries enough information
/// to either remove the obstruction by a flip or place a Steiner point on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Obstruction {
    /// The ray reached `b`: edge (a, b) is present (or the next hop lands on b).
    Recovered,
    /// a→b passes through the INTERIOR of the face (the face OPPOSITE `a` in the
    /// current `a`-incident tet `ti`). `face` are that face's three vertices.
    /// This is the obstruction shared with the tet on the far side of the ray.
    AcrossFace { ti: usize, face: [usize; 3] },
    /// a→b passes through the interior of an edge `(u, v)` of the exit face - an
    /// interior mesh edge crossing the segment.
    AcrossEdge { u: usize, v: usize },
    /// a→b passes through an existing vertex `v` lying ON the open segment (a, b):
    /// split the segment at v (free - no Steiner point needed).
    AcrossVert { v: usize },
    /// The march reached the convex-hull boundary (a hull tet) or could not make
    /// progress (degenerate / non-manifold local configuration). Treated as an
    /// obstruction the local flips cannot clear from this vertex.
    Boundary,
}

/// Find the first obstruction the ray `a`→`b` meets as it leaves vertex `a`
/// (TetGen's `finddirection`, realized as a direct scan of `a`'s tet star).
///
/// Crucially - and unlike [`find_tets_crossing_edge`] /
/// [`first_segment_crossing_point`], which SKIP endpoint-incident tets - this
/// examines the tets touching `a`. For a SHORT missing segment the obstruction
/// is the face shared between an `a`-incident tet and a `b`-incident tet; both
/// were invisible to the old skip-endpoint scans (the dominant "none" class
/// identified in the diagnosis).
///
/// The ray a→b enters exactly one finite `a`-incident tet through the solid
/// angle at `a` subtended by that tet's face opposite `a`. We find it by the
/// TetGen orientation logic, specialized: for a tet `(a, p, q, r)` oriented so
/// `orient_3d(a, p, q, r) > 0`, the three faces through `a` -
/// `(a,p,q)` [opp r], `(a,q,r)` [opp p], `(a,r,p)` [opp q] - each have their
/// opposite vertex on the strictly-positive side, so `b` is on the INTERIOR
/// side of a through-face (toward the exit face `(p,q,r)`) iff
/// `orient_3d(face..., b) > 0`. The ray enters this tet iff `b` is interior to
/// all three (each side value `>= 0`); the exact-zero pattern then distinguishes
/// `AcrossFace` (no zero), `AcrossEdge` (one zero) and `AcrossVert` (two zeros).
/// Walking the whole finite star (rather than a hull-crossing march) keeps the
/// routine robust against reflex/non-convex vertices and never dead-ends in a
/// hull tet. Direct interior face crossings win over edge/vertex grazes.
fn finddirection(tets: &Delaunay3D, a: usize, b: usize) -> Obstruction {
    // A direct hit (edge already present) short-circuits.
    if edge_exists_in_tets(tets, a, b) {
        return Obstruction::Recovered;
    }

    let pa = tets.vertices[a];
    let pb = tets.vertices[b];

    // Scan the FINITE star of `a` directly (rather than a hull-walking march):
    // the ray a→b enters exactly one finite `a`-incident tet through the solid
    // angle at `a` subtended by that tet's opposite face. Examining the whole
    // star - endpoint-incident tets included - is the whole point: the
    // obstruction face of a SHORT segment is shared between an `a`-incident tet
    // and a `b`-incident tet, which the old skip-endpoint scans missed. The
    // star is small, so this is cheap, and it never dead-ends on a hull tet.
    //
    // For a finite tet (a, p, q, r) oriented so orient_3d(a,p,q,r) > 0, the exit
    // face (opposite a) is (p, q, r) and the three faces through `a` -
    // (a,p,q) [opp r], (a,q,r) [opp p], (a,r,p) [opp q] - each have their
    // OPPOSITE vertex on the strictly-positive side. So `b` is on the INTERIOR
    // side of a through-face (toward the exit face) iff orient_3d(face..., b) > 0.
    // The ray enters THIS tet iff `b` is interior to all three (>= 0); a zero
    // means the ray grazes an edge/vertex of the exit face. We prefer a strict
    // interior hit; if only edge/vertex (zero) hits exist we take the best of
    // those.
    let mut best_edge: Option<(usize, usize)> = None;
    let mut best_vert: Option<usize> = None;

    // Only `a`-incident tets matter; iterate the index's incidence list when
    // active (O(degree)), else the whole array (full scan). The `v.contains(&a)`
    // guard below stays as the source of truth either way.
    let candidates: Vec<usize> = if tets.index_active() {
        tets.incident_tets(a).iter().map(|&x| x as usize).collect()
    } else {
        (0..tets.tets.len()).collect()
    };
    for ti in candidates {
        if !tets.is_live(ti) || is_hull_tet(&tets.tets[ti]) {
            continue;
        }
        let v = tets.tets[ti].verts;
        if !v.contains(&a) {
            continue;
        }
        // The other three vertices.
        let mut o3 = [usize::MAX; 3];
        let mut k = 0;
        for &x in &v {
            if x != a {
                if k < 3 {
                    o3[k] = x;
                }
                k += 1;
            }
        }
        if k != 3 {
            continue;
        }
        // A direct hit on b in this tet → edge a-b is an edge of it.
        if o3.contains(&b) {
            return Obstruction::Recovered;
        }
        // Orient (p, q, r) so (a, p, q, r) is positively oriented.
        let (p, mut q, mut r) = (o3[0], o3[1], o3[2]);
        let oa = orient_3d(pa, tets.vertices[p], tets.vertices[q], tets.vertices[r]);
        if oa < 0.0 {
            std::mem::swap(&mut q, &mut r);
        } else if oa == 0.0 {
            continue; // degenerate tet - skip
        }
        let pp = tets.vertices[p];
        let pq = tets.vertices[q];
        let pr = tets.vertices[r];

        // Side of `b` w.r.t. each through-face; positive ⇒ interior (toward exit).
        let sr = orient_3d(pa, pp, pq, pb); // face (a,p,q), opposite r
        let sp = orient_3d(pa, pq, pr, pb); // face (a,q,r), opposite p
        let sq = orient_3d(pa, pr, pp, pb); // face (a,r,p), opposite q

        // The ray enters this tet's solid angle at `a` iff none is strictly
        // negative (b is interior to / on all three through-faces).
        if sr < 0.0 || sp < 0.0 || sq < 0.0 {
            continue;
        }

        let zr = sr == 0.0;
        let zp = sp == 0.0;
        let zq = sq == 0.0;

        // Two zeros ⇒ a→b collinear with edge a→(shared vertex of the two zero
        // planes). Record as a candidate ACROSSVERT (only used if no interior or
        // edge hit is found).
        if zr && zp {
            if best_vert.is_none() {
                if let Obstruction::AcrossVert { v } = acrossvert_or_edge(tets, a, b, q) {
                    best_vert = Some(v);
                }
            }
            continue;
        }
        if zr && zq {
            if best_vert.is_none() {
                if let Obstruction::AcrossVert { v } = acrossvert_or_edge(tets, a, b, p) {
                    best_vert = Some(v);
                }
            }
            continue;
        }
        if zp && zq {
            if best_vert.is_none() {
                if let Obstruction::AcrossVert { v } = acrossvert_or_edge(tets, a, b, r) {
                    best_vert = Some(v);
                }
            }
            continue;
        }
        // One zero ⇒ a→b crosses an EDGE of the exit face (the edge not
        // containing the zero through-face's opposite vertex). Record as a
        // candidate ACROSSEDGE.
        if zr || zp || zq {
            let (u, w) = if zr {
                (p, q)
            } else if zp {
                (q, r)
            } else {
                (r, p)
            };
            if best_edge.is_none() {
                best_edge = Some((u, w));
            }
            continue;
        }
        // No zero ⇒ a→b pierces the INTERIOR of the exit face - the cleanest
        // obstruction; return immediately.
        return Obstruction::AcrossFace {
            ti,
            face: [p, q, r],
        };
    }

    // No strict interior face crossing. Prefer an edge crossing, then a vertex.
    if let Some((u, w)) = best_edge {
        return Obstruction::AcrossEdge { u, v: w };
    }
    if let Some(v) = best_vert {
        return Obstruction::AcrossVert { v };
    }
    Obstruction::Boundary
}

/// Helper for the ACROSSVERT case: vertex `v` lies on the line a→b past `a`. It
/// is only a usable split point if it lies STRICTLY inside the open segment
/// (a, b); otherwise the collinear vertex is beyond `b` (or behind `a`) and the
/// real obstruction is elsewhere - return `Boundary` so the caller falls back.
fn acrossvert_or_edge(tets: &Delaunay3D, a: usize, b: usize, v: usize) -> Obstruction {
    if v == a || v == b {
        return Obstruction::Boundary;
    }
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let pv = tets.vertices[v];
    let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let av = [pv[0] - pa[0], pv[1] - pa[1], pv[2] - pa[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if len2 == 0.0 {
        return Obstruction::Boundary;
    }
    let t = (av[0] * ab[0] + av[1] * ab[1] + av[2] * ab[2]) / len2;
    if t > 1e-9 && t < 1.0 - 1e-9 {
        Obstruction::AcrossVert { v }
    } else {
        Obstruction::Boundary
    }
}

/// The point where segment (a, b) crosses the obstruction face `face`
/// (the first exit face returned by [`finddirection`], which - unlike
/// [`first_segment_crossing_point`] - is found even when the crossed face is
/// shared with an endpoint-incident tet). Returns the crossing strictly between
/// `a` and `b`, or `None` if the segment is (numerically) parallel to / grazes
/// the face.
fn segment_face_crossing_point(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
    face: [usize; 3],
) -> Option<[f64; 3]> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let dir = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let p = tets.vertices[face[0]];
    let q = tets.vertices[face[1]];
    let r = tets.vertices[face[2]];
    let n = {
        let pq = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let pr = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        [
            pq[1] * pr[2] - pq[2] * pr[1],
            pq[2] * pr[0] - pq[0] * pr[2],
            pq[0] * pr[1] - pq[1] * pr[0],
        ]
    };
    let denom = dir[0] * n[0] + dir[1] * n[1] + dir[2] * n[2];
    if denom.abs() < 1e-30 {
        return None;
    }
    let ap = [p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]];
    let t = (ap[0] * n[0] + ap[1] * n[1] + ap[2] * n[2]) / denom;
    if t <= 1e-9 || t >= 1.0 - 1e-9 {
        return None;
    }
    Some([pa[0] + t * dir[0], pa[1] + t * dir[1], pa[2] + t * dir[2]])
}

/// The point where segment (a, b) crosses the (coplanar) edge (u, v) - the
/// ACROSSEDGE obstruction from [`finddirection`]. The two segments are coplanar
/// (the edge lies on the exit-face plane the ray pierces), so they meet at a
/// point; return it as the parameter along (a, b), strictly interior, or `None`
/// if numerically degenerate.
fn edge_segment_crossing_point(
    tets: &Delaunay3D,
    a: usize,
    b: usize,
    u: usize,
    w: usize,
) -> Option<[f64; 3]> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let pu = tets.vertices[u];
    let pw = tets.vertices[w];
    let d1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let d2 = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
    let r = [pu[0] - pa[0], pu[1] - pa[1], pu[2] - pa[2]];
    // Solve for t on (a,b): minimize via the common-perpendicular formula.
    // t = det([r, d2, d1×d2]) / |d1×d2|^2
    let n = [
        d1[1] * d2[2] - d1[2] * d2[1],
        d1[2] * d2[0] - d1[0] * d2[2],
        d1[0] * d2[1] - d1[1] * d2[0],
    ];
    let n2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if n2 < 1e-30 {
        if std::env::var("YAMM_ESC_DBG").is_ok() {
            eprintln!("    [esc DBG] ({a},{b})x({u},{w}): PARALLEL n2={n2:e}");
        }
        return None; // parallel
    }
    // t = ((r × d2) · n) / n2
    let rxd2 = [
        r[1] * d2[2] - r[2] * d2[1],
        r[2] * d2[0] - r[0] * d2[2],
        r[0] * d2[1] - r[1] * d2[0],
    ];
    let t = (rxd2[0] * n[0] + rxd2[1] * n[1] + rxd2[2] * n[2]) / n2;
    if t <= 1e-9 || t >= 1.0 - 1e-9 {
        if std::env::var("YAMM_ESC_DBG").is_ok() {
            // How far is the segment endpoint (t~0 -> a, t~1 -> b) from the
            // LINE (u,w)? Distinguishes "edge passes through the endpoint
            // vertex" (true vertex-on-edge degeneracy) from a near-miss.
            let pe = if t > 0.5 { pb } else { pa };
            let l2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
            let pu_pe = [pe[0] - pu[0], pe[1] - pu[1], pe[2] - pu[2]];
            let s_on = (pu_pe[0] * d2[0] + pu_pe[1] * d2[1] + pu_pe[2] * d2[2]) / l2;
            let proj = [
                pu[0] + s_on * d2[0],
                pu[1] + s_on * d2[1],
                pu[2] + s_on * d2[2],
            ];
            let dist =
                ((pe[0] - proj[0]).powi(2) + (pe[1] - proj[1]).powi(2) + (pe[2] - proj[2]).powi(2))
                    .sqrt();
            eprintln!(
                "    [esc DBG] ({a},{b})x({u},{w}): T-RANGE t={t:e} endpoint_dist_to_line={dist:e} s_on_edge={s_on:.6}"
            );
        }
        return None;
    }
    Some([pa[0] + t * d1[0], pa[1] + t * d1[1], pa[2] + t * d1[2]])
}

/// Attempt to recover edge (a, b) by performing local flips.
///
/// Strategy: iteratively find faces that block the edge and flip them away.
/// For each blocking face, try a 2-to-3 flip to remove it. Then look for
/// configurations where a 3-to-2 flip can create the desired edge.
///
/// Returns true if the edge was successfully recovered.
pub fn recover_edge_by_flips(tets: &mut Delaunay3D, a: usize, b: usize) -> bool {
    // Unprotected entry point (tests / external callers): no other boundary
    // elements are protected - identical to the historical behaviour.
    let empty_s = HashSet::default();
    let empty_f = HashSet::default();
    recover_edge_by_flips_protected(tets, a, b, &empty_s, &empty_f)
}

/// Like [`recover_edge_by_flips`] but **protects all already-recovered boundary
/// elements** (`psegs`/`pfaces`, the refined surface's segments/faces): no flip
/// is allowed to destroy another boundary segment or face while recovering
/// `(a, b)`. Without this guard, recovering one segment freely flips away a
/// neighbouring already-recovered segment (issue #37: the dominant cause of the
/// residual - a recovered edge destroyed by the next segment's recovery, never
/// re-recovered). A segment whose only obstruction IS a protected boundary
/// element correctly falls through to Steiner refinement instead of clobbering
/// it. Mirrors the guard `lawson_restore` already applies.
pub fn recover_edge_by_flips_protected(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    psegs: &HashSet<(usize, usize)>,
    pfaces: &HashSet<[usize; 3]>,
) -> bool {
    let mut budget = FLIP_RECOVER_BUDGET;
    let r = recover_edge_by_flips_budgeted(tets, a, b, psegs, pfaces, &mut budget);
    FLIP_BUDGET_HIGH_WATER.fetch_max(
        FLIP_RECOVER_BUDGET - budget,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

/// MARCH budget for ONE top-level flip recovery (see the per-iteration charge in
/// `recover_edge_by_flips_budgeted`).
///
/// The `AcrossVert` arm recurses on BOTH sub-segments, so a segment carrying a
/// chain of `d` collinear vertices costs O(2^d) invocations - effectively
/// unbounded, and a stack-overflow risk as well as a time one. Measured: a
/// `BlanketModule` tet boundary refined to target leaves ~32 collinear vertices
/// along a formerly 33×-target edge, and recovery then wedges for over 15
/// minutes inside a SINGLE segment. None of the segment loop's own caps can fire
/// there, because control never returns to the loop to be checked - which is why
/// this looked like "the carve grinds" rather than "the carve hangs".
///
/// A budget shared across the whole recursion bounds the tree without capping
/// legitimate depth (a long collinear chain is a normal thing to recover). It is
/// a march count, not a wall clock, so the same input behaves identically on a
/// fast or a slow CPU.
///
/// CALIBRATED against `flip_marches_max` (reported on the `[seg DBG] budget:`
/// line) over every zoo solid whose carve succeeds - i.e. the 15 that fail with
/// `YAMM_NO_CONFORMING=1`:
///
///   Circulartorus 5   Nestedtorus 5/7/7/7   Oktavian_2vol 6
///   SphereWithMultipleHoles 7   LShaped 8   EllipticCylinder 9   Ellipsoid 13
///
/// So a healthy recovery needs at most 13 marches and 256 is ~20x headroom. The
/// first version of this bound guessed 20_000 - 1500x the measured need - and
/// was still slow enough to leave the wedge in place.
const FLIP_RECOVER_BUDGET: u32 = 256;

/// March/ring-clear rounds per coplanar-ring recovery attempt. Also the weight
/// that attempt carries against the segment pass's work bound, since each round
/// can cost O(#tets).
const COPL_MAX_ITERS: usize = 32;

fn recover_edge_by_flips_budgeted(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    psegs: &HashSet<(usize, usize)>,
    pfaces: &HashSet<[usize; 3]>,
    budget: &mut u32,
) -> bool {
    if a == b {
        return false;
    }
    // Maximum iterations to prevent infinite loops. Each iteration makes
    // progress (one flip removes one obstruction, or an AcrossVert split
    // reduces the segment); a non-converging configuration bails to the gate.
    let max_iters = 500;

    for _ in 0..max_iters {
        // Charged PER ITERATION, not per invocation. Charging per invocation (the
        // first version of this bound) still permitted BUDGET × max_iters marches
        // - up to 1e7 - and each march is a walk over a mesh that can be 150k
        // tets, which is exactly the multi-minute wedge the bound was meant to
        // stop. Shared across the whole AcrossVert recursion, so it caps total
        // marches per top-level recovery however the tree is shaped.
        if *budget == 0 {
            return false; // exhausted - caller falls back to Steiner / the gate
        }
        *budget -= 1;
        if edge_exists_in_tets(tets, a, b) {
            return true;
        }
        // The active constraint is the target edge (its interior must not be
        // crossed); the protected SETS forbid destroying any other recovered
        // boundary element. Rebuilt per iteration (cheap - borrows the sets).
        let protect = ProtectedBoundary {
            active: Constraint::Edge(a, b),
            segments: psegs,
            faces: pfaces,
        };

        // ── finddirection-driven recovery ──
        // March from `a` toward `b`. The march examines `a`-incident tets, so it
        // SEES the obstruction even when it is the face shared with a
        // `b`-incident tet (the dominant "none" class the old skip-endpoint
        // scans missed). Try the symmetric march from `b` if `a`'s march hits a
        // boundary/degenerate dead end.
        let obs = match finddirection(tets, a, b) {
            Obstruction::Boundary => finddirection(tets, b, a),
            other => other,
        };

        match obs {
            Obstruction::Recovered => return true,
            Obstruction::AcrossVert { v } => {
                // The collinear vertex `v` lies ON segment (a, b). Recover the
                // two sub-segments instead - no Steiner point needed.
                let left = edge_exists_in_tets(tets, a, v)
                    || recover_edge_by_flips_budgeted(tets, a, v, psegs, pfaces, budget);
                let right = edge_exists_in_tets(tets, v, b)
                    || recover_edge_by_flips_budgeted(tets, v, b, psegs, pfaces, budget);
                // The original edge (a, b) cannot be a single tet edge through an
                // interior vertex `v`; success here means BOTH halves exist.
                return left && right;
            }
            Obstruction::AcrossEdge { u, v } => {
                // An interior mesh edge (u, v) crosses the segment in its
                // interior. Reduce its star with the hull-aware flip_nm, guarded
                // by the segment we are recovering so no flip pierces it - and
                // refuse the reduction if it would destroy a protected boundary
                // element (else recovering this segment clobbers a neighbour).
                if !ring_flip_destroys_protected(tets, u, v, &protect)
                    && flip_nm(tets, u, v, &protect.active).is_some()
                {
                    continue; // re-march from a
                }
                return false;
            }
            Obstruction::AcrossFace { ti, face } => {
                // The segment pierces the interior of `face`, shared between `ti`
                // (incident to a) and its neighbor across that face. A 2-3 flip
                // of that pair removes the obstructing face.
                let mut fi_shared = usize::MAX;
                for fi in 0..4 {
                    if delaunay3d::faces_match(
                        &delaunay3d::opposite_face(tets.tets[ti].verts, fi),
                        &face,
                    ) {
                        fi_shared = fi;
                        break;
                    }
                }
                if fi_shared != usize::MAX {
                    let neighbor = tets.tets[ti].adj[fi_shared];
                    if neighbor != usize::MAX
                        && tets.is_live(neighbor)
                        && !is_hull_tet(&tets.tets[neighbor])
                        && !flip23_destroys_protected(tets, ti, neighbor, &protect)
                        && flip_2_to_3(tets, ti, neighbor, &protect.active).is_some()
                    {
                        continue; // re-march from a
                    }
                }
                // The pair is not 2-3 flippable (non-convex / reflex). The
                // blocking element is then a reflex EDGE of the obstruction face;
                // try to reduce the star of each of its three edges.
                let mut flipped = false;
                for &(u, v) in &[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                    if !ring_flip_destroys_protected(tets, u, v, &protect)
                        && flip_nm(tets, u, v, &protect.active).is_some()
                    {
                        flipped = true;
                        break;
                    }
                }
                if flipped {
                    continue; // re-march from a
                }
                // Nothing local could clear this obstruction by flips.
                return false;
            }
            Obstruction::Boundary => {
                // Neither march found a flippable obstruction. Fall back to the
                // legacy direct-creation heuristics (2-3 / 3-2 near the edge) -
                // passing `protect` so this arm cannot destroy a recovered
                // neighbour either (the same #37 guard as the arms above).
                if try_create_edge_by_flip(tets, a, b, &protect) {
                    return true;
                }
                return false;
            }
        }
    }

    // Check one final time
    edge_exists_in_tets(tets, a, b)
}

/// Try to create edge (a, b) via a 3-to-2 flip.
///
/// Look for three tets sharing an edge that, when flipped to two tets
/// sharing a face, would introduce edge (a, b).
///
/// A 3-to-2 flip replaces tets around edge (p, q) with two tets sharing
/// face (r1, r2, r3). The new tets are (r1, r2, r3, p) and (r1, r2, r3, q).
/// Edge (a, b) appears if {a, b} are each in different tets -- e.g. if
/// a is a ring vertex and b = p, or a and b are both ring vertices.
///
/// The most direct way: if a is a ring vertex and b is p or q (or vice versa),
/// then edge (a, b) already exists in the old configuration (since a is a vert
/// of a tet containing p and q). So 3-to-2 flip creates edge (a, b) only if
/// a and b are both ring vertices (both in {r1, r2, r3}).
///
/// PROTECTED: `protect` carries the already-recovered boundary segments/faces;
/// neither the 2-3 nor the 3-2 flip below may destroy one (the same guard the
/// rest of segment recovery applies). Without it, this fallback arm re-opens the
/// #37 neighbour-clobber that the rest of the path closes. Pass an all-empty
/// `ProtectedBoundary` to recover the historical (unprotected) behaviour.
fn try_create_edge_by_flip(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    protect: &ProtectedBoundary,
) -> bool {
    // Strategy 1: Direct 2-to-3 flip.
    // If there is a face shared between a tet containing `a` (but not `b`)
    // and a tet containing `b` (but not `a`), then a 2-to-3 flip on that
    // face creates edge (a, b) directly. This is the simplest case.
    {
        let a_tets = tets_incident_to_vertex(tets, a);
        for &ta in &a_tets {
            if !tets.is_live(ta) || tets.tets[ta].verts.contains(&b) {
                continue;
            }
            for fi in 0..4 {
                let neighbor = tets.tets[ta].adj[fi];
                if neighbor == usize::MAX || !tets.is_live(neighbor) {
                    continue;
                }
                // Check if neighbor contains b but not a
                if tets.tets[neighbor].verts.contains(&b) && !tets.tets[neighbor].verts.contains(&a)
                {
                    // The shared face separates a from b.
                    // A 2-to-3 flip replaces these 2 tets with 3 tets
                    // that all share edge (apex_of_ta, apex_of_neighbor) = (a_apex, b_apex).
                    // Wait -- a_apex is vertex `a` (the vertex of ta opposite the shared face)
                    // only if `a` is opposite face fi. Let's check.
                    let apex_ta = tets.tets[ta].verts[fi];
                    if apex_ta == a {
                        // Yes, `a` is the apex of ta opposite the shared face.
                        // Find the apex of neighbor opposite the shared face.
                        // The shared face from neighbor's perspective:
                        if let Some((_, fj)) =
                            delaunay3d::shared_face_indices(&tets.tets[ta], &tets.tets[neighbor])
                        {
                            let apex_nb = tets.tets[neighbor].verts[fj];
                            if apex_nb == b {
                                // Perfect: flipping creates edge (a, b) - but not
                                // if it would destroy a protected boundary face.
                                if !flip23_destroys_protected(tets, ta, neighbor, protect)
                                    && flip_2_to_3(tets, ta, neighbor, &Constraint::None).is_some()
                                    && edge_exists_in_tets(tets, a, b)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 2: 3-to-2 flip to create the edge.
    // Find all tets incident to vertex `a`
    let a_tets = tets_incident_to_vertex(tets, a);

    for &ti in &a_tets {
        if !tets.is_live(ti) {
            continue;
        }
        // For each edge of this tet that does NOT contain `a`,
        // check if 3 tets share that edge, and if a 3-to-2 flip
        // around it would produce a face containing both a and b.
        let v = tets.tets[ti].verts;
        for ei in 0..6 {
            let (e0, e1) = tet_edge(v, ei);
            if e0 == a || e1 == a {
                continue;
            }
            // Edge (e0, e1) does not contain a.
            // A 3-to-2 flip around (e0, e1) would replace the ring around
            // that edge with 2 tets sharing a face. The ring vertices are
            // the non-{e0, e1} vertices. For this tet, that includes `a`.
            // If `b` is also a ring vertex, the flip creates edge (a, b).
            let sharing = tets_sharing_edge(tets, e0, e1);
            if sharing.len() != 3 {
                continue;
            }
            // Collect ring vertices
            let mut ring: Vec<usize> = Vec::new();
            for &si in &sharing {
                for &sv in &tets.tets[si].verts {
                    if sv != e0 && sv != e1 && !ring.contains(&sv) {
                        ring.push(sv);
                    }
                }
            }
            if ring.len() == 3 && ring.contains(&a) && ring.contains(&b) {
                // A 3-to-2 flip here would create face (ring[0], ring[1], ring[2])
                // which contains both a and b, creating edge (a, b). It REMOVES
                // edge (e0, e1) and its ring faces, so skip it if that would
                // destroy a protected boundary segment/face.
                let indices = [sharing[0], sharing[1], sharing[2]];
                if !ring_flip_destroys_protected(tets, e0, e1, protect)
                    && flip_3_to_2(tets, indices, e0, e1, &Constraint::None).is_some()
                    && edge_exists_in_tets(tets, a, b)
                {
                    return true;
                }
            }
        }
    }

    false
}

/// Return the i-th edge (0..6) of a tet as a pair of vertex indices.
fn tet_edge(verts: [usize; 4], idx: usize) -> (usize, usize) {
    match idx {
        0 => (verts[0], verts[1]),
        1 => (verts[0], verts[2]),
        2 => (verts[0], verts[3]),
        3 => (verts[1], verts[2]),
        4 => (verts[1], verts[3]),
        5 => (verts[2], verts[3]),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Face Recovery
// ---------------------------------------------------------------------------

/// Check if a triangle (a, b, c) exists as a face of any live tet.
///
/// A tet has 4 faces; we check whether any live tet has a face whose three
/// vertex indices are exactly the set {a, b, c} (in any order).
pub fn face_exists_in_tets(tets: &Delaunay3D, a: usize, b: usize, c: usize) -> bool {
    // A tet has face {a,b,c} iff its vertex set contains all three (the 4th
    // vertex is the apex). So only tets incident to `a` can qualify → O(degree)
    // via the index; fall back to the full scan when the index is inactive.
    if tets.index_active() {
        for &ti in tets.incident_tets(a) {
            let ti = ti as usize;
            if tets.is_live(ti) {
                let v = &tets.tets[ti].verts;
                if v.contains(&b) && v.contains(&c) {
                    return true;
                }
            }
        }
        return false;
    }
    for (i, tet) in tets.tets.iter().enumerate() {
        if !tets.is_live(i) {
            continue;
        }
        // Check each of the 4 faces of this tet
        for fi in 0..4 {
            let face = delaunay3d::opposite_face(tet.verts, fi);
            let mut sorted = [face[0], face[1], face[2]];
            sorted.sort();
            let mut target = [a, b, c];
            target.sort();
            if sorted == target {
                return true;
            }
        }
    }
    false
}

/// Recover a missing boundary face (a, b, c) by flipping blocking internal
/// faces.
///
/// After edge recovery, all three edges (a,b), (b,c), (a,c) are present in
/// the tet mesh. The face (a,b,c) may still be missing because internal tet
/// faces "block" it. This function uses both 3-to-2 and 2-to-3 flips to
/// recover the face.
///
/// Returns true if the face was successfully recovered.
pub fn recover_face_by_flips(tets: &mut Delaunay3D, a: usize, b: usize, c: usize) -> bool {
    let max_iters = 500;

    for _ in 0..max_iters {
        if face_exists_in_tets(tets, a, b, c) {
            return true;
        }

        // Strategy 1: Try a 3-to-2 flip on an internal edge whose ring
        // vertices are exactly {a, b, c}. This directly produces the face.
        if try_recover_face_by_3to2(tets, a, b, c) && face_exists_in_tets(tets, a, b, c) {
            return true;
        }

        // Strategy 2: Ring-based 2-to-3 flips around each edge of the
        // target face to open space for vertex c.
        let mut made_progress = false;
        for &(e0, e1, target) in &[(a, b, c), (a, c, b), (b, c, a)] {
            if made_progress {
                break;
            }

            let ring = tets_sharing_edge(tets, e0, e1);
            if ring.is_empty() {
                continue;
            }

            // If target is already a ring vertex, the face exists.
            let target_in_ring = ring.iter().any(|&ti| tets.tets[ti].verts.contains(&target));
            if target_in_ring {
                return true;
            }

            let pe0 = tets.vertices[e0];
            let pe1 = tets.vertices[e1];
            let pt = tets.vertices[target];

            for &ti in &ring {
                if !tets.is_live(ti) || is_hull_tet(&tets.tets[ti]) {
                    continue;
                }
                let v = tets.tets[ti].verts;
                for fi in 0..4 {
                    let apex = v[fi];
                    if apex == e0 || apex == e1 {
                        continue;
                    }
                    let face = delaunay3d::opposite_face(v, fi);
                    if !face.contains(&e0) || !face.contains(&e1) {
                        continue;
                    }
                    let d = face.iter().find(|&&x| x != e0 && x != e1).copied().unwrap();
                    let pd = tets.vertices[d];
                    let p_apex = tets.vertices[apex];

                    let o_t = orient_3d(pe0, pe1, pd, pt);
                    let o_apex = orient_3d(pe0, pe1, pd, p_apex);

                    if o_t == 0.0 || o_apex == 0.0 {
                        continue;
                    }
                    if o_t * o_apex >= 0.0 {
                        continue; // same side, not blocking
                    }

                    let neighbor = tets.tets[ti].adj[fi];
                    if neighbor == usize::MAX || !tets.is_live(neighbor) {
                        continue;
                    }

                    if flip_2_to_3(tets, ti, neighbor, &Constraint::None).is_some() {
                        made_progress = true;
                        break;
                    }
                }
                if made_progress {
                    break;
                }
            }
        }

        if !made_progress {
            return false;
        }
    }

    face_exists_in_tets(tets, a, b, c)
}

/// Try to recover face (a, b, c) by finding an internal edge whose removal
/// via a 3-to-2 flip would produce the desired face.
///
/// Looks for edges (p, q) shared by exactly 3 tets whose ring vertices
/// include {a, b, c}. A 3-to-2 flip would replace those 3 tets with 2 tets
/// sharing face (a, b, c).
fn try_recover_face_by_3to2(tets: &mut Delaunay3D, a: usize, b: usize, c: usize) -> bool {
    // Look at tets incident to vertex a. For each edge of such a tet that
    // does NOT include any of {a, b, c}, check if 3 tets share that edge
    // and the ring vertices are exactly {a, b, c}.
    let a_tets = tets_incident_to_vertex(tets, a);

    for &ti in &a_tets {
        if !tets.is_live(ti) {
            continue;
        }
        let v = tets.tets[ti].verts;

        for ei in 0..6 {
            let (e0, e1) = tet_edge(v, ei);

            // The edge must NOT be one of the face edges -- we want an
            // internal edge that crosses through the face.
            if (e0 == a || e0 == b || e0 == c) && (e1 == a || e1 == b || e1 == c) {
                continue;
            }

            let sharing = tets_sharing_edge(tets, e0, e1);
            if sharing.len() != 3 {
                continue;
            }

            // Collect ring vertices
            let mut ring: Vec<usize> = Vec::new();
            for &si in &sharing {
                for &sv in &tets.tets[si].verts {
                    if sv != e0 && sv != e1 && !ring.contains(&sv) {
                        ring.push(sv);
                    }
                }
            }

            if ring.len() == 3 && ring.contains(&a) && ring.contains(&b) && ring.contains(&c) {
                let indices = [sharing[0], sharing[1], sharing[2]];
                if flip_3_to_2(tets, indices, e0, e1, &Constraint::None).is_some() {
                    return true;
                }
            }
        }
    }

    false
}
/// Re-triangulate a face-recovery cavity with a local CONSTRAINED Delaunay.
///
/// Builds ONE local DT from the target face vertices [a,b,c] + every cavity
/// vertex, then FORCES the required faces (all cavity boundary faces + the
/// target) with the standard flip arsenal - a required face (often an adjacent
/// surface-facet "wall") is generally NOT Delaunay among the local vertex
/// subset, so an unconstrained DT would never contain it and the caller's
/// enlargement loop would spin forever on the same missing wall. The local DT
/// is tiny (tens of points), so even the full-scan recovery fallbacks are
/// trivially cheap here.
///
/// Flood-fills from a tet owning the target face, stopping at the cavity
/// BOUNDARY faces only - the target itself is INTERNAL, the flood crosses it
/// to fill both sides (no above/below split: side-classifying rim faces that
/// straddle the target plane leaks the flood; whole-cavity treatment avoids
/// that entirely). Kept tets are then CLIPPED to the OLD cavity volume
/// (centroid-inside test): the local DT fills the convex hull of its points,
/// which can exceed a non-convex cavity, and keeping a hull-pocket tet would
/// overlap mesh OUTSIDE the cavity (carve overshoot, gate reject).
///
/// Returns `Ok(tets)` with global-indexed tets, or `Err(missing_face)` if a
/// required face could not be forced into the local DT.
fn delaunize_cavity(
    all_vertices: &[[f64; 3]],
    face_abc: [usize; 3],
    cavity_verts: &[usize],
    boundary_faces: &[[usize; 3]],
    old_cavity: &[[usize; 4]],
) -> std::result::Result<Vec<[usize; 4]>, [usize; 3]> {
    let [a, b, c] = face_abc;

    if cavity_verts.is_empty() {
        return Err(face_abc);
    }

    // Build vertex mapping: local index ↔ global index
    let mut local_to_global: Vec<usize> = vec![a, b, c];
    let mut global_to_local: HashMap<usize, usize> = HashMap::default();
    global_to_local.insert(a, 0);
    global_to_local.insert(b, 1);
    global_to_local.insert(c, 2);
    for &gi in cavity_verts {
        if let std::collections::hash_map::Entry::Vacant(e) = global_to_local.entry(gi) {
            let li = local_to_global.len();
            e.insert(li);
            local_to_global.push(gi);
        }
    }
    // Also include all boundary face vertices (handles on-plane vertices)
    for bf in boundary_faces {
        for &vi in bf {
            if let std::collections::hash_map::Entry::Vacant(e) = global_to_local.entry(vi) {
                let li = local_to_global.len();
                e.insert(li);
                local_to_global.push(vi);
            }
        }
    }

    let local_points: Vec<[f64; 3]> = local_to_global.iter().map(|&gi| all_vertices[gi]).collect();
    if local_points.len() < 4 {
        return Err(face_abc);
    }

    // Build the local Delaunay triangulation…
    let mut dt = Delaunay3D::new(&local_points);

    // …and FORCE the required faces into it (see doc comment): edges first,
    // then the face, a few rounds to absorb recover-X-breaks-Y interactions.
    let req_local: Vec<[usize; 3]> = {
        let mut v: Vec<[usize; 3]> = boundary_faces
            .iter()
            .map(|f| {
                [
                    global_to_local[&f[0]],
                    global_to_local[&f[1]],
                    global_to_local[&f[2]],
                ]
            })
            .collect();
        v.push([0, 1, 2]); // face_abc is locals 0,1,2 by construction
        v
    };
    for _round in 0..4 {
        let mut all_present = true;
        for rf in &req_local {
            if face_exists_in_tets(&dt, rf[0], rf[1], rf[2]) {
                continue;
            }
            all_present = false;
            for &(u, v) in &[(rf[0], rf[1]), (rf[1], rf[2]), (rf[2], rf[0])] {
                if !edge_exists_in_tets(&dt, u, v) {
                    recover_edge_by_flips(&mut dt, u, v);
                }
            }
            let _ = recover_face_by_diagonal_swap(&mut dt, rf[0], rf[1], rf[2], true)
                || recover_face_by_edge_flips(&mut dt, rf[0], rf[1], rf[2]);
        }
        if all_present {
            break;
        }
    }

    let local_tets = dt.extract_tets();
    if local_tets.is_empty() {
        return Err(face_abc);
    }

    // Map tets back to global indices
    let global_tets: Vec<[usize; 4]> = local_tets
        .iter()
        .map(|t| {
            [
                local_to_global[t[0]],
                local_to_global[t[1]],
                local_to_global[t[2]],
                local_to_global[t[3]],
            ]
        })
        .collect();

    // Every required face must now exist in the local DT.
    let boundary_sorted: HashSet<[usize; 3]> = boundary_faces
        .iter()
        .map(|f| {
            let mut s = *f;
            s.sort();
            s
        })
        .collect();
    let mut required: Vec<[usize; 3]> = boundary_faces.to_vec();
    required.push(face_abc);
    for rf in &required {
        let mut sorted = *rf;
        sorted.sort();
        let found = global_tets.iter().any(|t| {
            (0..4).any(|fi| {
                let mut f = delaunay3d::opposite_face(*t, fi);
                f.sort();
                f == sorted
            })
        });
        if !found {
            return Err(*rf);
        }
    }

    // Build local adjacency among DT tets
    let n = global_tets.len();
    let mut adj: Vec<[usize; 4]> = vec![[usize::MAX; 4]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            for fi in 0..4 {
                let mut fa = delaunay3d::opposite_face(global_tets[i], fi);
                fa.sort();
                for fj in 0..4 {
                    let mut fb = delaunay3d::opposite_face(global_tets[j], fj);
                    fb.sort();
                    if fa == fb {
                        adj[i][fi] = j;
                        adj[j][fj] = i;
                    }
                }
            }
        }
    }

    // Centroid-in-old-cavity clip helpers.
    let inside_old = |g: [f64; 3]| -> bool {
        old_cavity.iter().any(|t| {
            let (p0, p1, p2, p3) = (
                all_vertices[t[0]],
                all_vertices[t[1]],
                all_vertices[t[2]],
                all_vertices[t[3]],
            );
            let o = orient_3d(p0, p1, p2, p3);
            if o == 0.0 {
                return false;
            }
            let s = o.signum();
            orient_3d(g, p1, p2, p3) * s >= 0.0
                && orient_3d(p0, g, p2, p3) * s >= 0.0
                && orient_3d(p0, p1, g, p3) * s >= 0.0
                && orient_3d(p0, p1, p2, g) * s >= 0.0
        })
    };
    let centroid = |t: &[usize; 4]| -> [f64; 3] {
        [
            (all_vertices[t[0]][0]
                + all_vertices[t[1]][0]
                + all_vertices[t[2]][0]
                + all_vertices[t[3]][0])
                / 4.0,
            (all_vertices[t[0]][1]
                + all_vertices[t[1]][1]
                + all_vertices[t[2]][1]
                + all_vertices[t[3]][1])
                / 4.0,
            (all_vertices[t[0]][2]
                + all_vertices[t[1]][2]
                + all_vertices[t[2]][2]
                + all_vertices[t[3]][2])
                / 4.0,
        ]
    };

    // Seed: a tet owning the target face whose centroid lies inside the cavity.
    let mut target_sorted = face_abc;
    target_sorted.sort();
    let mut seed = usize::MAX;
    for (ti, t) in global_tets.iter().enumerate() {
        let owns = (0..4).any(|fi| {
            let mut f = delaunay3d::opposite_face(*t, fi);
            f.sort();
            f == target_sorted
        });
        if owns && inside_old(centroid(t)) {
            seed = ti;
            break;
        }
    }
    if seed == usize::MAX {
        return Err(face_abc);
    }

    // Flood from the seed, stopping at cavity BOUNDARY faces only; the target
    // face is internal - the flood crosses it to cover both sides.
    let mut keep: HashSet<usize> = HashSet::default();
    let mut stack = vec![seed];
    while let Some(ti) = stack.pop() {
        if !keep.insert(ti) {
            continue;
        }
        for (fi, &ni) in adj[ti].iter().enumerate().take(4) {
            let mut f = delaunay3d::opposite_face(global_tets[ti], fi);
            f.sort();
            if boundary_sorted.contains(&f) {
                continue;
            }
            if ni != usize::MAX {
                stack.push(ni);
            }
        }
    }

    // Collect, CLIP to the old cavity, and orient.
    let mut result: Vec<[usize; 4]> = keep
        .iter()
        .map(|&ti| global_tets[ti])
        .filter(|t| inside_old(centroid(t)))
        .collect();
    for t in &mut result {
        let o = orient_3d(
            all_vertices[t[0]],
            all_vertices[t[1]],
            all_vertices[t[2]],
            all_vertices[t[3]],
        );
        if o < 0.0 {
            t.swap(0, 1);
        }
    }

    Ok(result)
}

/// Collect the LOCAL candidate tets whose closure could interact with the open
/// triangle [a, b, c]: BFS from the vertex stars of {a, b, c} (O(degree) via
/// the incidence index), admitting only tets whose AABB overlaps the padded
/// triangle AABB. Any tet an edge of which crosses the triangle interior - and
/// any tet of the triangle "pipe" - has an overlapping AABB and is
/// face-connected to the stars through such tets, so the BFS visits a superset
/// of them in O(local). Falls back to the full tet range when the index is
/// inactive or the local region exceeds the sanity cap (correctness preserved;
/// issue #47 - the per-call full scan made legacy `recover_faces` grind for
/// ~40 min on a 306k-face boundary).
fn collect_tri_local_candidates(tets: &Delaunay3D, a: usize, b: usize, c: usize) -> Vec<usize> {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let pc = tets.vertices[c];
    let mut lo = [0.0f64; 3];
    let mut hi = [0.0f64; 3];
    for k in 0..3 {
        lo[k] = pa[k].min(pb[k]).min(pc[k]);
        hi[k] = pa[k].max(pb[k]).max(pc[k]);
        let pad = (hi[k] - lo[k]).max(1e-12) * 1e-6;
        lo[k] -= pad;
        hi[k] += pad;
    }
    let tet_overlaps = |i: usize| -> bool {
        let v = tets.tets[i].verts;
        let mut tlo = [f64::INFINITY; 3];
        let mut thi = [f64::NEG_INFINITY; 3];
        for &x in &v {
            if x == INFINITE {
                return true; // hull tets span outward - treat as overlapping
            }
            for k in 0..3 {
                tlo[k] = tlo[k].min(tets.vertices[x][k]);
                thi[k] = thi[k].max(tets.vertices[x][k]);
            }
        }
        (0..3).all(|k| thi[k] >= lo[k] && tlo[k] <= hi[k])
    };
    const LOCAL_CAP: usize = 50_000;
    let mut candidates: Vec<usize> = Vec::new();
    if tets.index_active() {
        let mut stack: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::default();
        for &t in &[a, b, c] {
            for &ti in tets.incident_tets(t) {
                let ti = ti as usize;
                if tets.is_live(ti) && seen.insert(ti) {
                    stack.push(ti);
                }
            }
        }
        while let Some(ti) = stack.pop() {
            candidates.push(ti);
            if candidates.len() > LOCAL_CAP {
                break;
            }
            for fi in 0..4 {
                let nb = tets.tets[ti].adj[fi];
                if nb != usize::MAX && tets.is_live(nb) && !seen.contains(&nb) && tet_overlaps(nb) {
                    seen.insert(nb);
                    stack.push(nb);
                }
            }
        }
    }
    if !tets.index_active() || candidates.len() > LOCAL_CAP {
        candidates = (0..tets.tets.len()).collect();
    }
    candidates
}

/// Recover a missing face via cavity re-triangulation with iterative
/// cavity enlargement (TetGen-style delaunizecavity).
///
/// 1. Find the cavity: the tets the open triangle [a,b,c] passes through -
///    edge-piercing tets, plus (for an UNOBSTRUCTED missing facet, issue #37)
///    the edge-wedge / vertex-cone tets of the triangle "pipe".
/// 2. Separate surrounding vertices into above/below the face plane
/// 3. Build a local Delaunay on each side
/// 4. If a boundary face is missing from the local DT, enlarge the cavity
///    by adding the external neighbor of that face and retry
/// 5. Once both sides succeed, commit the new tets
///
/// `walls` are the OTHER surface facets (sorted triples, target excluded): the
/// cavity must never swallow one whole (both supports) nor grow across one -
/// they bound the cavity, become required boundary faces of the side
/// delaunizations, and are therefore PRESERVED. Without this, recovering one
/// facet destroys its neighbours (the carve then leaks elsewhere). Pass an
/// empty set for the historical unconstrained behaviour (legacy callers).
fn recover_face_by_cavity(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    c: usize,
    walls: &HashSet<[usize; 3]>,
) -> bool {
    let pa = tets.vertices[a];
    let pb = tets.vertices[b];
    let pc = tets.vertices[c];
    // Target centroid: the proxy direction "into the triangle interior" used by
    // the wedge / cone membership tests below.
    let pg = [
        (pa[0] + pb[0] + pc[0]) / 3.0,
        (pa[1] + pb[1] + pc[1]) / 3.0,
        (pa[2] + pb[2] + pc[2]) / 3.0,
    ];
    // The TARGET is never a wall, even if the caller's set includes it (lets
    // callers pass their full surface-facet list without excluding the target).
    let target_sorted_wall = {
        let mut s = [a, b, c];
        s.sort();
        s
    };
    let is_wall = move |f: [usize; 3]| -> bool {
        let mut s = f;
        s.sort();
        s != target_sorted_wall && walls.contains(&s)
    };

    // Find initial cavity: tets whose edges cross the face [a,b,c].
    // Include tets incident to face vertices (they may share a crossing edge
    // like the ring of tets around an edge that pierces the face).
    // Only skip tets that already contain face [a,b,c] as a tet face.
    // O(local) candidate collection (issue #47): a crossing tet's AABB overlaps
    // the triangle AABB and the crossing set is face-connected to the vertex
    // stars, so the star-BFS superset suffices; full-scan fallback inside.
    let mut cavity_set: HashSet<usize> = HashSet::default();
    for i in collect_tri_local_candidates(tets, a, b, c) {
        if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
            continue;
        }
        let v = tets.tets[i].verts;
        // Skip tets that already contain face [a,b,c]
        if v.contains(&a) && v.contains(&b) && v.contains(&c) {
            continue;
        }

        // Check non-face vertices for above/below classification
        let mut has_above = false;
        let mut has_below = false;
        for &vi in &v {
            if vi == INFINITE || vi == a || vi == b || vi == c {
                continue;
            }
            let o = orient_3d(pa, pb, pc, tets.vertices[vi]);
            if o > 1e-14 {
                has_above = true;
            } else if o < -1e-14 {
                has_below = true;
            }
        }
        if !has_above || !has_below {
            continue;
        }

        // Check if any non-face-vertex edge crosses the triangle [a,b,c]
        let edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        let any_crosses = edges.iter().any(|&[ei, ej]| {
            let vi = v[ei];
            let vj = v[ej];
            if vi == INFINITE || vj == INFINITE {
                return false;
            }
            // Skip edges that touch a face vertex (they start/end on the face)
            if vi == a || vi == b || vi == c {
                return false;
            }
            if vj == a || vj == b || vj == c {
                return false;
            }
            let oi = orient_3d(pa, pb, pc, tets.vertices[vi]);
            let oj = orient_3d(pa, pb, pc, tets.vertices[vj]);
            if oi * oj >= 0.0 {
                return false;
            }
            edge_crosses_triangle(pa, pb, pc, tets.vertices[vi], tets.vertices[vj])
        });
        if any_crosses {
            cavity_set.insert(i);
        }
    }

    // Triangle-"pipe" seed (issue #37, the UNOBSTRUCTED missing facet): the open
    // triangle's interior also passes through tets that touch it only at its
    // boundary - through the WEDGE of tets at each target edge and the CONE of
    // tets at each target vertex. Collect the tets whose wedge/cone strictly
    // contains the direction toward the target centroid `pg`. This is the exact
    // pipe: for an adjacent surface facet sharing a target edge, only the ONE
    // support on the target's side qualifies, so the neighbour facet stays on
    // the cavity BOUNDARY (required → preserved) rather than being swallowed.
    // Under-inclusion (the triangle's angular sector spans more tets than the
    // single centroid direction) is completed by the wall-respecting enlargement
    // loop below. O(degree) via the incidence index.
    {
        let tri = [a, b, c];
        let mut cand: Vec<usize> = Vec::new();
        for &t in &tri {
            if tets.index_active() {
                cand.extend(tets.incident_tets(t).iter().map(|&x| x as usize));
            }
        }
        if !tets.index_active() {
            cand = (0..tets.tets.len()).collect();
        }
        cand.sort_unstable();
        cand.dedup();
        for i in cand {
            if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) || cavity_set.contains(&i) {
                continue;
            }
            let v = tets.tets[i].verts;
            let shared: Vec<usize> = tri.iter().copied().filter(|t| v.contains(t)).collect();
            let others: Vec<usize> = v
                .iter()
                .copied()
                .filter(|x| !tri.contains(x) && *x != INFINITE)
                .collect();
            match shared.len() {
                3 => continue, // already owns the face (impossible: it's missing)
                2 => {
                    // Edge-wedge test: tet (u, v, x, y) around target edge (u, v).
                    // The triangle enters this tet iff `pg` lies strictly inside
                    // the wedge spanned by faces (u,v,x) and (u,v,y): pg on the
                    // y-side of (u,v,x) AND on the x-side of (u,v,y).
                    if others.len() != 2 {
                        continue;
                    }
                    let (u, w) = (shared[0], shared[1]);
                    let (x, y) = (others[0], others[1]);
                    let (puu, pww) = (tets.vertices[u], tets.vertices[w]);
                    let (px, py) = (tets.vertices[x], tets.vertices[y]);
                    let sx_y = orient_3d(puu, pww, px, py);
                    let sx_g = orient_3d(puu, pww, px, pg);
                    let sy_x = orient_3d(puu, pww, py, px);
                    let sy_g = orient_3d(puu, pww, py, pg);
                    if sx_y != 0.0 && sy_x != 0.0 && sx_g * sx_y > 0.0 && sy_g * sy_x > 0.0 {
                        cavity_set.insert(i);
                    }
                }
                1 => {
                    // Vertex-cone test: tet (t, p, q, r) at target vertex t. The
                    // triangle enters iff `pg` is strictly inside the solid angle
                    // at t: on the interior side of all three faces through t.
                    if others.len() != 3 {
                        continue;
                    }
                    let t = shared[0];
                    let (p, q, r) = (others[0], others[1], others[2]);
                    let pt = tets.vertices[t];
                    let (pp, pq, pr) = (tets.vertices[p], tets.vertices[q], tets.vertices[r]);
                    let f1 = orient_3d(pt, pp, pq, pr) * orient_3d(pt, pp, pq, pg);
                    let f2 = orient_3d(pt, pq, pr, pp) * orient_3d(pt, pq, pr, pg);
                    let f3 = orient_3d(pt, pr, pp, pq) * orient_3d(pt, pr, pp, pg);
                    if f1 > 0.0 && f2 > 0.0 && f3 > 0.0 {
                        cavity_set.insert(i);
                    }
                }
                _ => {}
            }
        }
    }

    // Safety pass: the cavity must never contain BOTH supports of a wall (an
    // adjacent surface facet) - that would make the wall internal and the side
    // delaunization (which only splits along the TARGET's plane) would destroy
    // it. Keep the support whose fourth vertex's wedge faces the target centroid
    // and evict the other (the far side of the wall).
    {
        let in_cavity: Vec<usize> = cavity_set.iter().copied().collect();
        for &i in &in_cavity {
            if !cavity_set.contains(&i) {
                continue;
            }
            let v = tets.tets[i].verts;
            for fi in 0..4 {
                let f = delaunay3d::opposite_face(v, fi);
                if f.contains(&INFINITE) || !is_wall(f) {
                    continue;
                }
                let nb = tets.tets[i].adj[fi];
                if nb == usize::MAX || !cavity_set.contains(&nb) {
                    continue; // wall already on the cavity boundary - fine
                }
                // Both supports inside: evict the one on the far side of the
                // wall from the target centroid.
                let (f0, f1v, f2v) = (f[0], f[1], f[2]);
                let (q0, q1, q2) = (tets.vertices[f0], tets.vertices[f1v], tets.vertices[f2v]);
                let og = orient_3d(q0, q1, q2, pg);
                let apex_i = v[fi];
                let oi = if apex_i == INFINITE {
                    0.0
                } else {
                    orient_3d(q0, q1, q2, tets.vertices[apex_i])
                };
                if og != 0.0 && oi != 0.0 && og * oi > 0.0 {
                    cavity_set.remove(&nb); // `i` is on the target side - keep it
                } else {
                    cavity_set.remove(&i);
                }
            }
        }
    }

    let cdbg = std::env::var("YAMM_CAVITY_DBG").is_ok();
    if cavity_set.is_empty() {
        if cdbg {
            eprintln!("    [CAVITY] ({a},{b},{c}) empty seed - bail");
        }
        return false;
    }
    if cdbg {
        eprintln!(
            "    [CAVITY] ({a},{b},{c}) seed={} walls={}",
            cavity_set.len(),
            walls.len()
        );
    }

    // Limit cavity size to prevent runaway enlargement on complex models
    let max_cavity_size = (tets.tets.len() / 4).max(20);
    let max_enlargements = 50;

    for _enl in 0..max_enlargements {
        if cavity_set.len() > max_cavity_size {
            if cdbg {
                eprintln!("    [CAVITY] too big {} - bail", cavity_set.len());
            }
            return false;
        }

        // DETERMINISM: `cavity_set` is a HashSet and Rust randomises HashSet
        // iteration order per process, so every use of that order below leaks
        // into the result: the vertex list, boundary-face list and tet list
        // handed to `delaunize_cavity`, the summation order of the volume
        // check (which moves its last bits across a tolerance), and the order
        // tets are freed (which changes the indices `alloc_tet` hands out
        // next). Sort once and drive all of them off that -- the same fix the
        // segment queue in `recover_segments_with_steiner` already applies for
        // the same reason.
        let mut cavity_sorted: Vec<usize> = cavity_set.iter().copied().collect();
        cavity_sorted.sort_unstable();

        // Collect every cavity vertex (both sides AND on-plane - the whole-
        // cavity delaunization needs them all) and the cavity boundary faces.
        let mut cav_verts: Vec<usize> = Vec::new();
        let mut bfaces: Vec<(usize, [usize; 3])> = Vec::new();

        for &ti in &cavity_sorted {
            for &vi in &tets.tets[ti].verts {
                if vi == INFINITE || vi == a || vi == b || vi == c {
                    continue;
                }
                if !cav_verts.contains(&vi) {
                    cav_verts.push(vi);
                }
            }
        }

        for &ti in &cavity_sorted {
            let tet = &tets.tets[ti];
            for fi in 0..4 {
                let nb = tet.adj[fi];
                if nb != usize::MAX && cavity_set.contains(&nb) {
                    continue;
                }
                let face = delaunay3d::opposite_face(tet.verts, fi);
                if face.contains(&INFINITE) {
                    continue;
                }
                bfaces.push((nb, face));
            }
        }

        let bf: Vec<[usize; 3]> = bfaces.iter().map(|(_, f)| *f).collect();
        let cavity_tet_verts: Vec<[usize; 4]> = cavity_sorted
            .iter()
            .map(|&ti| tets.tets[ti].verts)
            .collect();
        let cavity_result = delaunize_cavity(
            &tets.vertices,
            [a, b, c],
            &cav_verts,
            &bf,
            &cavity_tet_verts,
        );

        match cavity_result {
            Ok(new_tets) => {
                // Volume preservation check: compare old cavity volume to new
                let old_vol: f64 = cavity_sorted
                    .iter()
                    .map(|&ti| {
                        let v = tets.tets[ti].verts;
                        orient_3d(
                            tets.vertices[v[0]],
                            tets.vertices[v[1]],
                            tets.vertices[v[2]],
                            tets.vertices[v[3]],
                        )
                        .abs()
                            / 6.0
                    })
                    .sum();

                let new_vol: f64 = new_tets
                    .iter()
                    .map(|t| {
                        orient_3d(
                            tets.vertices[t[0]],
                            tets.vertices[t[1]],
                            tets.vertices[t[2]],
                            tets.vertices[t[3]],
                        )
                        .abs()
                            / 6.0
                    })
                    .sum();

                // Reject unless the replacement EXACTLY fills the cavity: any
                // volume LOSS means a gap; any GAIN means the new tets spill
                // outside the cavity and overlap existing mesh (carve
                // overshoot). Tolerance scales with the cavity volume.
                let vtol = (old_vol * 1e-9).max(1e-14);
                if old_vol > 1e-15 && (new_vol - old_vol).abs() > vtol {
                    if cdbg {
                        eprintln!("    [CAVITY] volume reject old={old_vol:.6} new={new_vol:.6}");
                    }
                    return false;
                }

                // Re-triangulation succeeded - swap out the cavity tets.
                for &ti in &cavity_sorted {
                    tets.free_tet(ti);
                }

                let mut new_indices: Vec<usize> = Vec::new();
                for tv in new_tets.iter() {
                    let ti = tets.alloc_tet(Tet {
                        verts: *tv,
                        adj: [usize::MAX; 4],
                    });
                    new_indices.push(ti);
                }

                // Inter-new-tet adjacency
                for i in 0..new_indices.len() {
                    for j in (i + 1)..new_indices.len() {
                        let ni = new_indices[i];
                        let nj = new_indices[j];
                        if let Some((fi, fj)) =
                            delaunay3d::shared_face_indices(&tets.tets[ni], &tets.tets[nj])
                        {
                            tets.tets[ni].adj[fi] = nj;
                            tets.tets[nj].adj[fj] = ni;
                        }
                    }
                }

                // External adjacency: connect new tets to neighbors outside the cavity
                for &(nb, ref bf) in bfaces.iter() {
                    if nb == usize::MAX || !tets.is_live(nb) {
                        continue;
                    }
                    for &ni in &new_indices {
                        for fi in 0..4 {
                            let nf = delaunay3d::opposite_face(tets.tets[ni].verts, fi);
                            if delaunay3d::faces_match(&nf, bf) {
                                tets.tets[ni].adj[fi] = nb;
                                // Update neighbor to point back to the new tet
                                for nfi in 0..4 {
                                    let nbf = delaunay3d::opposite_face(tets.tets[nb].verts, nfi);
                                    if delaunay3d::faces_match(&nbf, bf) {
                                        tets.tets[nb].adj[nfi] = ni;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }

                return face_exists_in_tets(tets, a, b, c);
            }
            Err(missing) => {
                // A required face is missing from the local DT.
                // Enlarge the cavity by adding the external neighbor of that face
                // - but NEVER across a wall (another surface facet): walls bound
                // the cavity so they stay required-and-preserved. If the missing
                // face IS a wall, this facet can't be cavity-recovered without
                // destroying a neighbour - give up (no-op; gate falls back).
                let mut sorted_missing = missing;
                sorted_missing.sort();
                let missing_is_wall = is_wall(sorted_missing);
                if cdbg {
                    eprintln!(
                        "    [CAVITY] enl={_enl} cav={} missing-required={sorted_missing:?} wall={missing_is_wall}",
                        cavity_set.len(),
                    );
                }

                // Absorbing tet `i` must never make a wall INTERNAL to the
                // cavity (its other support already inside): the side
                // delaunization only preserves walls on the cavity BOUNDARY.
                // (The seed's both-supports safety pass covers the seed; this
                // covers every enlargement.)
                let internalizes_wall =
                    |tets: &Delaunay3D, cavity_set: &HashSet<usize>, i: usize| -> bool {
                        (0..4).any(|fi| {
                            let f = delaunay3d::opposite_face(tets.tets[i].verts, fi);
                            if f.contains(&INFINITE) || !is_wall(f) {
                                return false;
                            }
                            let nb = tets.tets[i].adj[fi];
                            nb != usize::MAX && cavity_set.contains(&nb)
                        })
                    };

                let mut enlarged = false;
                // Absorb the neighbour across the missing face - FORBIDDEN when
                // that face is a wall (another surface facet): walls bound the
                // cavity. A missing wall is instead addressed by the fallback
                // enlargement below (absorb elsewhere → the side's local DT
                // changes and may then reproduce the wall).
                if !missing_is_wall {
                    for &(nb, ref bf) in bfaces.iter() {
                        let mut sbf = *bf;
                        sbf.sort();
                        if sbf == sorted_missing
                            && nb != usize::MAX
                            && tets.is_live(nb)
                            && !is_hull_tet(&tets.tets[nb])
                            && !cavity_set.contains(&nb)
                            && !internalizes_wall(tets, &cavity_set, nb)
                        {
                            cavity_set.insert(nb);
                            enlarged = true;
                            break;
                        }
                    }
                }

                if !enlarged {
                    // Directed fallback: absorb exactly ONE tet bordering the
                    // cavity across a NON-wall face, preferring one that shares a
                    // vertex with the missing required face (it is the local DT
                    // around that face we need to change). Absorbing one at a
                    // time keeps the cavity minimal - bulk absorption blows it
                    // into a non-convex blob the side delaunization cannot fill
                    // (volume-loss reject).
                    let mut pick: Option<usize> = None; // (preferred)
                    let mut pick_any: Option<usize> = None;
                    'search: for &v in &[a, b, c] {
                        for (i, tet) in tets.tets.iter().enumerate() {
                            if !tets.is_live(i) || is_hull_tet(tet) {
                                continue;
                            }
                            if !tet.verts.contains(&v) {
                                continue;
                            }
                            if cavity_set.contains(&i) {
                                continue;
                            }
                            let adjacent_via_nonwall = (0..4).any(|fi| {
                                let nb = tet.adj[fi];
                                nb != usize::MAX
                                    && cavity_set.contains(&nb)
                                    && !is_wall(delaunay3d::opposite_face(tet.verts, fi))
                            });
                            if !adjacent_via_nonwall || internalizes_wall(tets, &cavity_set, i) {
                                continue;
                            }
                            if pick_any.is_none() {
                                pick_any = Some(i);
                            }
                            if sorted_missing.iter().any(|m| tet.verts.contains(m)) {
                                pick = Some(i);
                                break 'search;
                            }
                        }
                    }
                    if let Some(i) = pick.or(pick_any) {
                        cavity_set.insert(i);
                        enlarged = true;
                    }
                    if !enlarged {
                        return false;
                    }
                }
            }
        }
    }

    false
}

/// Recover all missing boundary faces.
///
/// For each boundary triangle (a, b, c), checks if it already exists as a
/// face in the tetrahedralization. If not, attempts to recover it via local
/// flip operations.
///
/// Returns `(recovered, failed)` -- the number of faces that were
/// successfully recovered and the number that could not be recovered.
pub fn recover_faces(
    tets: &mut Delaunay3D,
    boundary_faces: &[[usize; 3]],
) -> (usize, usize, Vec<[usize; 3]>) {
    let mut recovered = 0;
    let mut failed = 0;
    let mut failed_faces = Vec::new();

    // Fast count of missing faces using a HashSet lookup.
    // If too many are missing (>10% and >20), skip recovery - curved
    // boundaries where most faces aren't in the Delaunay.
    {
        let free_list = &tets.free_list;
        let mut existing: HashSet<[usize; 3]> = HashSet::default();
        for (i, tet) in tets.tets.iter().enumerate() {
            if free_list.contains(&i) {
                continue;
            }
            for fi in 0..4 {
                let mut f = delaunay3d::opposite_face(tet.verts, fi);
                f.sort();
                existing.insert(f);
            }
        }
        let missing_count = boundary_faces
            .iter()
            .filter(|face| {
                let mut s = **face;
                s.sort();
                !existing.contains(&s)
            })
            .count();
        if std::env::var("YAMM_TIME_DBG").is_ok() {
            eprintln!(
                "    [time] legacy recover_faces: {} of {} boundary faces missing",
                missing_count,
                boundary_faces.len()
            );
        }
        if missing_count > 20 && missing_count > boundary_faces.len() / 10 {
            return (
                0,
                missing_count,
                boundary_faces
                    .iter()
                    .filter(|face| {
                        let mut s = **face;
                        s.sort();
                        !existing.contains(&s)
                    })
                    .copied()
                    .collect(),
            );
        }
    }

    // All boundary faces are WALLS for the cavity recovery (the target itself is
    // exempted inside): recovering one face must not destroy another.
    let walls: HashSet<[usize; 3]> = boundary_faces
        .iter()
        .map(|f| {
            let mut s = *f;
            s.sort();
            s
        })
        .collect();
    for face in boundary_faces {
        let (a, b, c) = (face[0], face[1], face[2]);
        if face_exists_in_tets(tets, a, b, c) {
            continue;
        }
        // Cheap, exact coplanar-quad diagonal swap first (the cube case);
        // fall back to full cavity re-triangulation only if that does not apply.
        // Legacy filter path: no hull-aware ring flip (see use_ring_flip docs).
        if recover_face_by_diagonal_swap(tets, a, b, c, false)
            || recover_face_by_cavity(tets, a, b, c, &walls)
        {
            recovered += 1;
        } else {
            failed += 1;
            failed_faces.push([a, b, c]);
        }
    }

    #[cfg(debug_assertions)]
    debug_validate_tets(tets);

    (recovered, failed, failed_faces)
}

/// Like [`recover_faces`] but never gives up early: it attempts cavity
/// recovery for *every* missing boundary face, even when most are missing
/// (normal for curved boundaries). The conforming volume path needs full
/// boundary conformity so the interior can be carved by flood-fill without
/// leaks; the early-bail in [`recover_faces`] is for the legacy filter path.
///
/// Boundary vertices are never moved and no points are added here, so the
/// input surface triangulation is preserved exactly. Returns
/// `(recovered, failed, failed_faces)`.
pub fn recover_faces_no_bail(
    tets: &mut Delaunay3D,
    boundary_faces: &[[usize; 3]],
) -> (usize, usize, Vec<[usize; 3]>) {
    let mut recovered = 0;
    let mut failed = 0;
    let mut failed_faces = Vec::new();
    // All boundary faces are WALLS for the cavity recovery (the target itself is
    // exempted inside): recovering one face must not destroy another.
    let walls: HashSet<[usize; 3]> = boundary_faces
        .iter()
        .map(|f| {
            let mut s = *f;
            s.sort();
            s
        })
        .collect();
    for face in boundary_faces {
        let (a, b, c) = (face[0], face[1], face[2]);
        if face_exists_in_tets(tets, a, b, c) {
            continue;
        }
        // Coplanar-quad diagonal swap first (the cube case), then the general
        // piercing-edge flip recovery (curved / thin / non-convex facets), then
        // cavity re-triangulation as a last resort.
        // Conforming carve path: enable the hull-aware ring flip so the hull
        // adopts the wanted diagonal too (otherwise the carve flood leaks).
        if recover_face_by_diagonal_swap(tets, a, b, c, true)
            || recover_face_by_edge_flips(tets, a, b, c)
            || recover_face_by_cavity(tets, a, b, c, &walls)
        {
            recovered += 1;
        } else {
            failed += 1;
            failed_faces.push([a, b, c]);
        }
    }
    #[cfg(debug_assertions)]
    debug_validate_tets(tets);
    (recovered, failed, failed_faces)
}

// ---------------------------------------------------------------------------
// Phase 3: Conforming Delaunay refinement by Steiner points
// ---------------------------------------------------------------------------
//
// Flip-only recovery plateaus on curved / thin / non-convex solids: a boundary
// SEGMENT can be crossed by a large edge-star (ring size > 4) that `flip_nm`
// cannot reduce, and a FACET cannot close while one of its edges is missing.
//
// KEY INSIGHT - splitting a boundary segment at a point ON it, or a boundary
// facet at a point IN its plane, produces a FINER TRIANGULATION OF THE SAME
// GEOMETRIC SURFACE. The enclosed volume is unchanged. We therefore recover by
// conforming refinement (TetGen's segment/facet recovery): place Steiner points
// on the boundary until every (refined) segment and facet is present, tracking
// the evolving refined boundary-triangle set `cur_faces` so the carve flood is
// blocked by the actual present faces. The volume gate references the ORIGINAL
// input triangles (the true target volume), so correctness is guaranteed.
//
// Both passes are GUARANTEED TO TERMINATE: each Steiner split strictly shrinks
// a segment / facet, and the total length / area is bounded.

// ---------------------------------------------------------------------------
// Hull-aware Lawson flip restoration (TetGen's lawsonflip3d after insertion)
// ---------------------------------------------------------------------------
//
// The Steiner LOCAL SPLITS (split_edge_on_constraint 1→2, split_face_on_constraint
// 1→3, split_tet_1_to_4) are conformal and hull-aware but do NOT restore the
// empty-circumsphere (Delaunay) property. After a split the sub-segments /
// sub-facets through the new vertex are usually still NOT mesh edges/faces, so
// the recovery cannot close them and the refinement recurses into slivers
// without converging.
//
// `lawson_restore` re-establishes local Delaunay-ness around the freshly
// inserted vertex by Lawson flips (standard incremental-Delaunay; TetGen's
// lawsonflip3d). After it runs, the boundary sub-elements through the new vertex
// become real Delaunay edges/faces and refinement converges.

/// The set of boundary elements a Lawson flip must never destroy: the segment /
/// facet currently being recovered (`active`), plus every boundary segment and
/// triangle already recovered so far. A flip is rejected (kept as a no-op) if it
/// would remove a protected edge or face from the mesh.
pub struct ProtectedBoundary<'a> {
    /// The constraint being recovered right now (its interior must not be
    /// crossed by any new flip element). Reuses the existing eligibility guard.
    pub active: Constraint,
    /// Already-recovered boundary segments (sorted (min,max) vertex pairs).
    pub segments: &'a HashSet<(usize, usize)>,
    /// Already-recovered boundary triangles (sorted vertex triples).
    pub faces: &'a HashSet<[usize; 3]>,
}

impl ProtectedBoundary<'_> {
    #[inline]
    fn is_protected_edge(&self, u: usize, v: usize) -> bool {
        if u == INFINITE || v == INFINITE {
            return false;
        }
        self.segments.contains(&(u.min(v), u.max(v)))
    }
    #[inline]
    fn is_protected_face(&self, f: [usize; 3]) -> bool {
        if f.contains(&INFINITE) {
            return false;
        }
        let mut s = f;
        s.sort();
        self.faces.contains(&s)
    }
}

/// Would a 2-3 flip of tets (t1, t2) - which removes their shared face and
/// creates the new edge (d, e) - destroy a protected boundary FACE? The only
/// face it removes is the shared triangle; the three faces it creates contain
/// that triangle's edges plus d or e (never removing other faces). So the flip
/// is forbidden iff the shared face itself is protected.
fn flip23_destroys_protected(
    tets: &Delaunay3D,
    t1: usize,
    t2: usize,
    protect: &ProtectedBoundary,
) -> bool {
    if let Some((fi1, _)) = delaunay3d::shared_face_indices(&tets.tets[t1], &tets.tets[t2]) {
        let f = delaunay3d::opposite_face(tets.tets[t1].verts, fi1);
        protect.is_protected_face(f)
    } else {
        false
    }
}

/// Would reducing the star of edge (p, q) (a 3-2 / 4-4 ring flip) destroy a
/// protected boundary element? Such a flip REMOVES the edge (p, q) and the
/// interior faces of its ring that contain (p, q). It is forbidden iff (p, q)
/// itself is a protected segment, OR any ring face (p, q, w) is a protected
/// boundary triangle.
fn ring_flip_destroys_protected(
    tets: &Delaunay3D,
    p: usize,
    q: usize,
    protect: &ProtectedBoundary,
) -> bool {
    if protect.is_protected_edge(p, q) {
        return true;
    }
    // Every face (p, q, w) currently in the mesh would be replaced. Only tets
    // incident to `p` can hold edge (p, q) → O(degree) via the index when active
    // (was a full O(#tets) scan; this guard runs per flip in recover_edge_by_flips).
    let check = |i: usize| -> bool {
        if !tets.is_live(i) {
            return false;
        }
        let v = tets.tets[i].verts;
        if !v.contains(&p) || !v.contains(&q) {
            return false;
        }
        (0..4).any(|fi| {
            let f = delaunay3d::opposite_face(v, fi);
            f.contains(&p) && f.contains(&q) && protect.is_protected_face(f)
        })
    };
    if tets.index_active() {
        tets.incident_tets(p).iter().any(|&x| check(x as usize))
    } else {
        (0..tets.tets.len()).any(check)
    }
}

/// The candidate tets to examine when looking for tets incident to `v`: the
/// vertex→tet index's incidence list (O(degree)) when active, else the whole
/// array (full scan - only the unit tests that skip `build_vert_tets` hit this).
/// Returned as an owned `Vec` so the caller may freely mutate the mesh (flips)
/// inside the loop without holding the index's borrow.
fn incident_candidates(tets: &Delaunay3D, v: usize) -> Vec<usize> {
    if tets.index_active() {
        let mut c: Vec<usize> = tets.incident_tets(v).iter().map(|&x| x as usize).collect();
        // Process in ASCENDING tet-index order, identical to the original
        // `0..tets.tets.len()` scan. The Lawson seed order determines the flip
        // sequence, and that sequence is order-sensitive on degenerate/cospherical
        // configurations (the thin-wall Pipe/TWC recovery). Matching the old order
        // keeps the result byte-for-byte identical - this is a pure perf change.
        c.sort_unstable();
        c
    } else {
        (0..tets.tets.len()).collect()
    }
}

/// True iff live finite tet `i` equals `{new_vertex} ∪ linkface` - i.e. it is the
/// tet whose face opposite `new_vertex` is the (sorted) `linkface`.
fn tet_owns_link_face(
    tets: &Delaunay3D,
    i: usize,
    new_vertex: usize,
    linkface: [usize; 3],
) -> bool {
    if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
        return false;
    }
    let v = tets.tets[i].verts;
    if !v.contains(&new_vertex) {
        return false;
    }
    let mut others = [0usize; 3];
    let mut k = 0;
    for &x in &v {
        if x == new_vertex {
            continue;
        }
        if k < 3 {
            others[k] = x;
        }
        k += 1;
    }
    if k != 3 {
        return false;
    }
    others.sort();
    others == linkface
}

/// Re-establish local Delaunay-ness around the freshly-inserted vertex
/// `new_vertex` by Lawson flips (TetGen's `lawsonflip3d`), so the boundary
/// sub-elements through `new_vertex` become real Delaunay edges/faces and
/// conforming refinement converges.
///
/// HULL-AWARE: a hull tet (apex INFINITE) is never used as the tet that owns the
/// tested face (TetGen skips `ishulltet(fliptets[0])`); and the in_sphere test
/// against a hull neighbor (apex INFINITE) is treated as "outside" (the finite
/// point cannot be inside a hull tet's degenerate circumsphere here - the carve
/// handles the hull boundary), so a hull link face is never flipped by this
/// pass. The actual flips reuse the existing transactional, hull-aware
/// [`flip_2_to_3`] (convex case) and [`flip_nm`] (3-2 / 4-4 reductions), each of
/// which independently re-validates orientation and topology and is a true no-op
/// on rejection.
///
/// PROTECTED: every candidate flip is guarded by `protect` - it is skipped if it
/// would (a) create an element crossing the active constraint's interior
/// (`check_flip_eligibility`, already enforced inside the flip primitives via
/// `protect.active`), or (b) destroy an already-recovered boundary segment or
/// triangle. By conforming-Delaunay theory, once enough Steiner points are added
/// the boundary elements ARE Delaunay (so they are never flip candidates); the
/// guard only prevents transient destruction.
///
/// The flip stack and total flip count are bounded so a pathological
/// configuration bails (leaving the mesh valid) rather than looping.
pub fn lawson_restore(
    tets: &mut Delaunay3D,
    new_vertex: usize,
    protect: &ProtectedBoundary,
) -> usize {
    use std::collections::VecDeque;

    // A face to test, identified by the FINITE tet on the new_vertex side and
    // the (sorted) vertices of the face OPPOSITE new_vertex in that tet. We
    // re-locate the tet by its vertex set at pop time (indices are stable but
    // the tet may have been freed by an earlier flip).
    let mut stack: VecDeque<[usize; 3]> = VecDeque::new();
    // Seed the flip stack from `new_vertex`'s link faces (the face opposite
    // `new_vertex` in each incident finite tet). O(degree) via the vertex index;
    // the incidence list is re-read on each re-seed so it reflects the flips made
    // in the prior round.
    let seed = |tets: &Delaunay3D, stack: &mut VecDeque<[usize; 3]>| {
        for i in incident_candidates(tets, new_vertex) {
            if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
                continue;
            }
            let v = tets.tets[i].verts;
            if !v.contains(&new_vertex) {
                continue;
            }
            // The link face is the face OPPOSITE new_vertex.
            for fi in 0..4 {
                if v[fi] == new_vertex {
                    let mut f = delaunay3d::opposite_face(v, fi);
                    f.sort();
                    stack.push_back(f);
                    break;
                }
            }
        }
    };
    seed(tets, &mut stack);

    // Bound the work: a healthy insertion settles in O(link size) flips. A
    // pathological one bails (the mesh stays valid; the gate arbitrates).
    let max_flips = (tets.tets.len()).clamp(256, 20_000);
    let mut flips = 0usize;
    let mut iters = 0usize;
    let max_iters = max_flips.saturating_mul(8).max(4096);

    // Outer round loop (TetGen's `while (flippool->items != 0l)`): a non-Delaunay
    // face that is unflippable RIGHT NOW (its reflex edge has a ring this flip
    // layer can't reduce yet) often becomes flippable after a neighboring flip.
    // After the stack drains, if this round made progress, re-seed from ALL of
    // new_vertex's current link faces and try again. Bounded so a genuinely
    // unflippable residue (Schoenhardt-like) terminates instead of looping.
    let max_rounds = 64usize;
    let mut round = 0usize;
    let mut round_flips = usize::MAX; // force at least one round

    while round_flips > 0 && round < max_rounds {
        round += 1;
        let flips_before_round = flips;

        while let Some(linkface) = stack.pop_front() {
            iters += 1;
            if iters > max_iters || flips > max_flips {
                return flips; // perf bound - mesh left valid
            }

            // Locate the finite tet T that contains new_vertex AND has `linkface`
            // as the face opposite new_vertex (i.e. T = {new_vertex} ∪ linkface).
            // O(degree) via the vertex index (was a full O(#tets) scan per popped
            // link face - the dominant recovery cost on large meshes, issue #37).
            // The match is unique (a face has one tet on new_vertex's side), so no
            // ordering is needed and the index slice is iterated alloc-free.
            let t = if tets.index_active() {
                tets.incident_tets(new_vertex)
                    .iter()
                    .map(|&x| x as usize)
                    .find(|&i| tet_owns_link_face(tets, i, new_vertex, linkface))
            } else {
                (0..tets.tets.len()).find(|&i| tet_owns_link_face(tets, i, new_vertex, linkface))
            }
            .unwrap_or(usize::MAX);
            if t == usize::MAX {
                continue; // face no longer present (already flipped) - skip
            }

            // The neighbor N across `linkface` (the face opposite new_vertex).
            let vt = tets.tets[t].verts;
            let mut fi_link = usize::MAX;
            for fi in 0..4 {
                if vt[fi] == new_vertex {
                    fi_link = fi;
                    break;
                }
            }
            if fi_link == usize::MAX {
                continue;
            }
            let n = tets.tets[t].adj[fi_link];
            if n == usize::MAX || !tets.is_live(n) {
                continue;
            }
            if is_hull_tet(&tets.tets[n]) {
                // Hull neighbor: the link face is on the convex hull. A finite point
                // is "inside a hull tet's circumsphere" iff it is OUTSIDE the hull
                // face - but new_vertex was inserted conformally on/inside the
                // domain, so we never flip a hull link face here. (TetGen handles
                // hull slivers separately; the carve handles the hull boundary.)
                continue;
            }

            // p = the vertex of N opposite the shared face (the apex of N).
            let (_, fi_n) = match delaunay3d::shared_face_indices(&tets.tets[t], &tets.tets[n]) {
                Some(x) => x,
                None => continue,
            };
            let p = tets.tets[n].verts[fi_n];
            if p == INFINITE {
                continue;
            }

            // Delaunay test: is p strictly inside T's circumsphere?
            // T = {new_vertex} ∪ linkface, positively oriented.
            let mut tv = vt;
            let o = orient_3d(
                tets.vertices[tv[0]],
                tets.vertices[tv[1]],
                tets.vertices[tv[2]],
                tets.vertices[tv[3]],
            );
            if o == 0.0 {
                continue; // degenerate (should not happen for a live finite tet)
            }
            if o < 0.0 {
                tv.swap(0, 1);
            }
            let sign = in_sphere(
                tets.vertices[tv[0]],
                tets.vertices[tv[1]],
                tets.vertices[tv[2]],
                tets.vertices[tv[3]],
                tets.vertices[p],
            );
            if sign <= 0.0 {
                continue; // locally Delaunay - nothing to do
            }

            // ── Non-Delaunay: decide which flip restores it ──
            // The shared face is `linkface` = (a, b, c); d = new_vertex (apex of T),
            // e = p (apex of N). TetGen's convexity test: a 2-3 flip is valid iff the
            // new edge (d, e) pierces the interior of the shared triangle, i.e. the
            // three orient(face_edge, d, e) share one sign. Otherwise the offending
            // shared-face edge (a', b') is reflex/flat and must be reduced by a
            // 3-2 / 4-4 ring flip on that edge.
            let d = new_vertex;
            let e = p;
            let f0 = linkface[0];
            let f1 = linkface[1];
            let f2 = linkface[2];
            let vd = tets.vertices[d];
            let ve = tets.vertices[e];
            let s01 = orient_3d(tets.vertices[f0], tets.vertices[f1], vd, ve);
            let s12 = orient_3d(tets.vertices[f1], tets.vertices[f2], vd, ve);
            let s20 = orient_3d(tets.vertices[f2], tets.vertices[f0], vd, ve);

            let all_same =
                (s01 > 0.0 && s12 > 0.0 && s20 > 0.0) || (s01 < 0.0 && s12 < 0.0 && s20 < 0.0);

            let mut did_flip = false;
            if all_same && s01 != 0.0 && s12 != 0.0 && s20 != 0.0 {
                // Convex bipyramid → 2-3 flip (removes the shared face, makes (d,e)).
                if !flip23_destroys_protected(tets, t, n, protect) {
                    if let Some(new_idx) = flip_2_to_3(tets, t, n, &protect.active) {
                        // Push the newly-created link faces incident to new_vertex.
                        push_new_link_faces(tets, &new_idx, new_vertex, &mut stack);
                        did_flip = true;
                        flips += 1;
                    }
                }
            } else {
                // Reflex/flat edge: pick the shared-face edge (a', b') whose
                // orient(a', b', d, e) <= 0 and reduce its star with the hull-aware
                // flip_nm (3-2 when the ring has 3 tets, 4-4 when 4).
                let edge_opt = if s01 <= 0.0 {
                    Some((f0, f1))
                } else if s12 <= 0.0 {
                    Some((f1, f2))
                } else if s20 <= 0.0 {
                    Some((f2, f0))
                } else {
                    None
                };
                if let Some((pp, qq)) = edge_opt {
                    if !ring_flip_destroys_protected(tets, pp, qq, protect) {
                        if let Some(new_idx) = flip_nm(tets, pp, qq, &protect.active) {
                            push_new_link_faces(tets, &new_idx, new_vertex, &mut stack);
                            did_flip = true;
                            flips += 1;
                        }
                    }
                }
            }
            let _ = did_flip; // a no-op flip simply leaves the face un-flipped
        }

        round_flips = flips - flips_before_round;
        // If progress was made, re-seed from all current link faces for another
        // pass (the residual non-Delaunay faces may now be flippable).
        if round_flips > 0 {
            seed(tets, &mut stack);
        }
    }

    // Diagnostic: count link faces of new_vertex still non-Delaunay (a perfect
    // restoration leaves zero). Helps distinguish "lawson incomplete" from
    // "strategy can't make this segment Delaunay".
    if std::env::var("YAMM_LAWSON_DBG").is_ok() {
        let mut nd = 0usize;
        for i in incident_candidates(tets, new_vertex) {
            if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
                continue;
            }
            let v = tets.tets[i].verts;
            if !v.contains(&new_vertex) {
                continue;
            }
            let mut fi_link = usize::MAX;
            for fi in 0..4 {
                if v[fi] == new_vertex {
                    fi_link = fi;
                    break;
                }
            }
            if fi_link == usize::MAX {
                continue;
            }
            let n = tets.tets[i].adj[fi_link];
            if n == usize::MAX || !tets.is_live(n) || is_hull_tet(&tets.tets[n]) {
                continue;
            }
            let (_, fn_) = match delaunay3d::shared_face_indices(&tets.tets[i], &tets.tets[n]) {
                Some(x) => x,
                None => continue,
            };
            let p = tets.tets[n].verts[fn_];
            if p == INFINITE {
                continue;
            }
            let mut tv = v;
            let o = orient_3d(
                tets.vertices[tv[0]],
                tets.vertices[tv[1]],
                tets.vertices[tv[2]],
                tets.vertices[tv[3]],
            );
            if o < 0.0 {
                tv.swap(0, 1);
            }
            if in_sphere(
                tets.vertices[tv[0]],
                tets.vertices[tv[1]],
                tets.vertices[tv[2]],
                tets.vertices[tv[3]],
                tets.vertices[p],
            ) > 0.0
            {
                nd += 1;
            }
        }
        if nd > 0 {
            eprintln!("    [lawson DBG] m={new_vertex}: {nd} link faces still non-Delaunay after {flips} flips");
        }
    }

    flips
}

/// After a flip, push every NEW tet's face opposite `new_vertex` (its link face)
/// onto the stack so it is re-tested for local Delaunay-ness.
fn push_new_link_faces(
    tets: &Delaunay3D,
    new_tets: &[usize],
    new_vertex: usize,
    stack: &mut std::collections::VecDeque<[usize; 3]>,
) {
    for &ni in new_tets {
        if !tets.is_live(ni) || is_hull_tet(&tets.tets[ni]) {
            continue;
        }
        let v = tets.tets[ni].verts;
        if !v.contains(&new_vertex) {
            continue;
        }
        for fi in 0..4 {
            if v[fi] == new_vertex {
                let mut f = delaunay3d::opposite_face(v, fi);
                f.sort();
                stack.push_back(f);
                break;
            }
        }
    }
}

/// Build the protected boundary sets (segments + sorted triangles) from the
/// current refined boundary-triangle set `cur_faces`. Every triangle's three
/// edges become protected segments; every triangle (sorted) becomes a protected
/// face. Used to guard Lawson flips against destroying recovered boundary
/// elements.
/// Also returns an edge -> incident-surface-faces index, built in the same pass.
///
/// The index exists because the coplanar-REGION recovery needs the surface faces
/// incident to one segment, and it used to get them by scanning all of
/// `cur_faces` - O(#faces) per segment, and its early "crease / not coplanar"
/// bails do NOT memoize, so a crease segment re-paid that scan every visit. On a
/// target-sized BlanketModule boundary that is 72210 segments x 48140 faces ~
/// 3.5e9, which is the wedge that made the carve look like it hung. Returning the
/// index from HERE rather than building it at the call sites means it cannot go
/// stale: every place that rebuilds the protected sets rebuilds the index with
/// them, and the compiler enforces it.
type ProtectedSets = (
    HashSet<(usize, usize)>,
    HashSet<[usize; 3]>,
    HashMap<(usize, usize), Vec<[usize; 3]>>,
);

fn protected_sets_from_faces(cur_faces: &[[usize; 3]]) -> ProtectedSets {
    let mut segs: HashSet<(usize, usize)> = HashSet::default();
    let mut faces: HashSet<[usize; 3]> = HashSet::default();
    let mut inc: HashMap<(usize, usize), Vec<[usize; 3]>> = HashMap::default();
    for f in cur_faces {
        for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            segs.insert((u.min(v), u.max(v)));
            inc.entry((u.min(v), u.max(v))).or_default().push(*f);
        }
        let mut s = *f;
        s.sort();
        faces.insert(s);
    }
    (segs, faces, inc)
}

/// The clamp TetGen applies to a segment split parameter to avoid creating a
/// near-degenerate sub-segment when the exact midpoint is geometrically
/// problematic: the split point is restricted to `[CLAMP, 1-CLAMP]` of the
/// segment.
const STEINER_SEG_CLAMP: f64 = 0.2;

/// Point at parameter `t` along segment (pa, pb): pa + t*(pb-pa).
fn lerp(pa: [f64; 3], pb: [f64; 3], t: f64) -> [f64; 3] {
    [
        pa[0] + t * (pb[0] - pa[0]),
        pa[1] + t * (pb[1] - pa[1]),
        pa[2] + t * (pb[2] - pa[2]),
    ]
}

/// Recover a missing boundary segment (a, b) that is blocked by a COPLANAR
/// wrong-diagonal edge - the flat-reflex-face case (LShaped). After the normal
/// flip recovery has failed, the segment's obstruction (from `finddirection`)
/// is an `AcrossEdge {u, v}` where (u, v) is COPLANAR with (a, b)
/// (`orient_3d(a, b, u, v) == 0`) and the two segments cross in their interior -
/// i.e. they are the two diagonals of a flat-face quad and (u, v) is the wrong
/// one. The non-degenerate flips reject the larger (n > 4) wrong-diagonal rings
/// of these flat fans, so use the general hull-aware ring reducer
/// [`flip_ring_general`] to clear (u, v); the segment driver then re-marches and
/// the wanted edge (a, b) closes.
///
/// PROTECTED + DETERMINISTIC: the reducer is guarded by `Constraint::Edge(a, b)`
/// (no sub-flip may cross the wanted segment) and by a `ProtectedBoundary` built
/// from the current refined `cur_faces` (no sub-flip may destroy an already-
/// recovered boundary segment or face). The protected sets are rebuilt before
/// each ring clearance (the mesh changes between iterations). Returns `true` iff
/// (a, b) is present afterwards. A failed reduction is a transactional no-op, so
/// this never regresses the mesh; the caller falls through to the Steiner path.
///
/// SCOPE: fires ONLY when the obstruction is COPLANAR with the wanted segment
/// (a flat-face region). Curved thin walls (Pipe / ThinWalledCylinder) have no
/// such coplanar-diagonal obstruction, so this path is never entered for them.
fn recover_segment_coplanar_ring(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    psegs: &HashSet<(usize, usize)>,
    pfaces: &HashSet<[usize; 3]>,
) -> bool {
    // Bound the march/clear loop: each successful ring clearance removes one
    // crossing edge, so the segment closes in a few iterations; a non-converging
    // configuration bails (transactional no-op) rather than looping.
    let max_iters = COPL_MAX_ITERS;
    for _ in 0..max_iters {
        if edge_exists_in_tets(tets, a, b) {
            return true;
        }
        // Locate the obstruction, marching from a, then b.
        let obs = match finddirection(tets, a, b) {
            Obstruction::Boundary => finddirection(tets, b, a),
            other => other,
        };
        let (u, v) = match obs {
            Obstruction::AcrossEdge { u, v } => (u, v),
            _ => return false, // not an edge obstruction - not our case
        };
        // The crossing edge must be COPLANAR with the wanted segment and cross
        // it in the interior (the two diagonals of a flat-face quad). This is the
        // exact signature of the LShaped flat-reflex-face residual; bailing here
        // keeps curved/thin-wall models out of this path entirely.
        if orient_3d(
            tets.vertices[a],
            tets.vertices[b],
            tets.vertices[u],
            tets.vertices[v],
        ) != 0.0
        {
            return false;
        }
        if !segments_cross_interior(&tets.vertices, [a, b], [u, v]) {
            return false;
        }
        // The protected boundary is the caller's, which it already maintains
        // incrementally (rebuilt only when `cur_faces` actually changes).
        //
        // This used to call `protected_sets_from_faces(cur_faces)` HERE, inside
        // the 32-iteration loop, in a function called once per segment. The
        // borrow was immutable, so all 32 rebuilds produced identical sets - pure
        // waste, and O(#faces) each. It was the dominant cost of the whole
        // conforming carve on planar-dominated geometry, because the coplanar
        // guard above (orient_3d == 0) is exactly what a flat face satisfies:
        // BlanketModule with a target-sized boundary (48140 faces, 72210
        // segments) spent >15 minutes here and never returned to the segment
        // loop, so none of that loop's caps could fire. Measured with this path
        // disabled (YAMM_NO_COPL_REGION=1) the same carve finishes in 15.4s.
        let protect = ProtectedBoundary {
            active: Constraint::Edge(a, b),
            segments: psegs,
            faces: pfaces,
        };
        if !flip_ring_general(tets, u, v, &protect) {
            return false; // could not clear this crossing - fall through
        }
        // Cleared (u, v); re-march for the next crossing (or success).
    }
    edge_exists_in_tets(tets, a, b)
}

/// Recover ALL boundary segments (flips, then Steiner refinement), maintaining
/// the evolving refined boundary-triangle set `cur_faces` so that when a segment
/// is split its two incident facets are split with it (the new vertex sits on
/// their shared edge). Returns `true` iff every (refined) segment is present in
/// the tetrahedralization afterwards.
///
/// `cur_faces` starts as the input boundary triangles and is refined in place.
/// Every Steiner point lies on the original surface (on a boundary segment), so
/// the geometry - and hence the enclosed volume - is preserved exactly.
///
/// GUARANTEED TO TERMINATE: a Steiner split replaces a segment by two strictly
/// shorter sub-segments; we cap the total number of splits to a generous
/// multiple of the segment count and bail (mesh left for the gate to fall back)
/// if exceeded - a perf timeout, never a correctness compromise.
pub fn recover_segments_with_steiner(
    tets: &mut Delaunay3D,
    cur_faces: &mut Vec<[usize; 3]>,
) -> bool {
    use std::collections::VecDeque;

    // Build the initial segment queue from cur_faces' edges (deduplicated).
    let mut seg_set: HashSet<(usize, usize)> = HashSet::default();
    for f in cur_faces.iter() {
        for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            seg_set.insert((u.min(v), u.max(v)));
        }
    }
    // Each queue entry carries a recursion DEPTH (number of Steiner splits on
    // the chain that produced it). A bounded depth makes the pass terminate even
    // on degenerate thin/reflex features: a healthy segment closes within a few
    // splits, whereas a sliver cascade blows past the cap and we bail to the
    // gate (fallback) rather than loop forever.
    if std::env::var("YAMM_SEG_DBG").is_ok() {
        let mut n_missing = 0usize;
        let mut n_vertex_on = 0usize;
        let mut n_face_cross = 0usize;
        let mut n_edge_cross = 0usize;
        let mut n_none = 0usize;
        for &(a, b) in &seg_set {
            if edge_exists_in_tets(tets, a, b) {
                continue;
            }
            n_missing += 1;
            // Classify by what `finddirection` (the recovery driver) actually
            // sees - marching from `a`, and from `b` if `a` dead-ends. The
            // columns map: vertex_on=AcrossVert, face_cross=AcrossFace,
            // edge_cross=AcrossEdge, none=Boundary (unmarchable / hull).
            let obs = match finddirection(tets, a, b) {
                Obstruction::Boundary => finddirection(tets, b, a),
                other => other,
            };
            match obs {
                Obstruction::AcrossVert { .. } => n_vertex_on += 1,
                Obstruction::AcrossFace { .. } => n_face_cross += 1,
                Obstruction::AcrossEdge { .. } => n_edge_cross += 1,
                // Recovered should be impossible (we filtered present edges) but
                // bucket it as resolvable; Boundary is the unmarchable residue.
                Obstruction::Recovered => {}
                Obstruction::Boundary => n_none += 1,
            }
        }
        eprintln!(
            "    [seg DBG] segments={} missing={} | vertex_on={} face_cross={} edge_cross={} none={}",
            seg_set.len(), n_missing, n_vertex_on, n_face_cross, n_edge_cross, n_none
        );
        // INPUT QUALITY: length percentiles of all vs missing segments, and the
        // minimum angle over the boundary triangles. A boundary triangulation
        // carrying slivers / near-zero-length edges cannot be made recoverable by
        // refinement - refinement inherits the degeneracy - so this separates "the
        // recovery is weak" from "the input boundary is degenerate".
        let seg_len = |a: usize, b: usize| -> f64 {
            let (pa, pb) = (tets.vertices[a], tets.vertices[b]);
            ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2) + (pb[2] - pa[2]).powi(2)).sqrt()
        };
        let mut all_l: Vec<f64> = Vec::new();
        let mut miss_l: Vec<f64> = Vec::new();
        for f in cur_faces.iter() {
            for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let l = seg_len(u, v);
                all_l.push(l);
                if !edge_exists_in_tets(tets, u, v) {
                    miss_l.push(l);
                }
            }
        }
        let pct = |v: &mut Vec<f64>| -> String {
            if v.is_empty() {
                return "n/a".to_string();
            }
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let q = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
            format!(
                "min={:.3e} p1={:.3e} p50={:.3e} max={:.3e}",
                v[0],
                q(0.01),
                q(0.50),
                v[v.len() - 1]
            )
        };
        // Smallest angle of each boundary triangle (degrees).
        let mut min_ang: Vec<f64> = Vec::new();
        for f in cur_faces.iter() {
            let (l0, l1, l2) = (
                seg_len(f[1], f[2]),
                seg_len(f[0], f[2]),
                seg_len(f[0], f[1]),
            );
            // Law of cosines: angle opposite side `o`, between sides `x` and `y`.
            let ang = |o: f64, x: f64, y: f64| -> f64 {
                if x <= 0.0 || y <= 0.0 {
                    return 0.0;
                }
                ((x * x + y * y - o * o) / (2.0 * x * y))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            };
            let a = ang(l0, l1, l2).min(ang(l1, l0, l2)).min(ang(l2, l0, l1));
            min_ang.push(a);
        }
        eprintln!("    [seg DBG] input quality: all-seg {}", pct(&mut all_l));
        eprintln!("    [seg DBG] input quality: missing  {}", pct(&mut miss_l));
        eprintln!(
            "    [seg DBG] input quality: tri min-angle(deg) {} | tris={} under-1deg={} under-5deg={}",
            pct(&mut min_ang.clone()),
            cur_faces.len(),
            min_ang.iter().filter(|&&a| a < 1.0).count(),
            min_ang.iter().filter(|&&a| a < 5.0).count()
        );
    }
    // DETERMINISM: HashSet iteration order is randomised per run (Rust's
    // RandomState), and the order segments are recovered in changes which flips
    // / Steiner insertions happen - i.e. it perturbs the final mesh, and made
    // Pipe/ThinWalledCylinder conform only intermittently. Sort the initial
    // queue by (min,max) vertex pair so identical input → identical mesh.
    let mut seg_vec: Vec<(usize, usize)> = seg_set.into_iter().collect();
    seg_vec.sort_unstable();
    let mut queue: VecDeque<(usize, usize, u32)> =
        seg_vec.into_iter().map(|(a, b)| (a, b, 0u32)).collect();

    // Budget: a healthy model recovers all segments by flips alone (zero
    // Steiner splits) or with a handful. A model whose curved/thin/reflex
    // features prevent flip recovery induces a sliver cascade; each Steiner
    // split here costs O(tets) (crossing scan + edge-existence), so on a large
    // mesh an unbounded cascade is O(n²) and would hang. Cap the split count
    // both relatively AND absolutely so a non-converging model bails FAST to the
    // gate (clean fallback) instead of grinding. (Correctness is unaffected: the
    // gate validates the carved volume exactly regardless.)
    // Cap the split count both relatively AND absolutely so a non-converging
    // model bails FAST to the gate (clean fallback) instead of grinding. (Lawson
    // restoration after each split improves sub-segment closure, but a model
    // whose missing segments are blocked by collinear vertices/edges - where the
    // crossing-point scheme finds no usable crossing - still cannot converge by
    // local refinement and must fall back. Correctness is unaffected: the gate
    // validates the carved volume exactly regardless.)
    // Work bounds are SCALE-RELATIVE (proportional to the boundary size) and
    // MACHINE-INDEPENDENT - no wall clock, so the same input conforms/bails
    // identically on a fast or slow CPU, and a larger geometry gets
    // proportionally more budget. A conformable boundary recovers almost
    // entirely by flips, needing ~zero Steiner splits and ZERO "unproductive"
    // events (Pipe: 0). A boundary that cannot conform by local refinement
    // (fine curved surfaces) accrues unproductive events (the crossing-point
    // scheme finds no usable crossing → midpoint/grazed/no-cross cascade); we
    // bail once those exceed a fraction of the boundary size - fast for failure,
    // never triggered by success.
    let n_seg0 = queue.len();
    // Split budget. Historically a TIGHT cap ((n_seg0/64).clamp(32, 96)) served
    // as the primary "not converging" detector - at the time fine curved
    // surfaces COULD NOT converge (the residual was a boundary-protection leak,
    // since fixed) so bailing fast was right. Now they DO converge (the
    // EllipticCylinder proxy drains ~180 splits to FULL conformity), so the
    // tight cap was the last thing forcing convex curved models onto the
    // fallback. Non-convergence is instead detected by `max_unproductive`
    // below - a converging recovery accrues ZERO unproductive events (measured:
    // proxy AND casing both 0), a non-converging one accrues them steadily. So
    // the split budget is only a generous scale-relative backstop now.
    // The budget must scale with the INITIALLY MISSING count, not the total
    // segment count: NestedCylinder has 5112 segments of which 1367 are
    // missing - n_seg0/8 = 639 splits could never cover the work even with
    // every split productive (issue #47 cluster B). O(missing × degree) to
    // count (the incidence index is active here).
    let n_missing0 = queue
        .iter()
        .filter(|&&(a, b, _)| !edge_exists_in_tets(tets, a, b))
        .count();
    let max_splits = std::env::var("YAMM_MAX_SPLITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(((n_seg0 / 8).max(4 * n_missing0)).clamp(256, 8192));
    // No-progress bail (replaces a fragile, machine-/size-dependent wall clock).
    // A conformable boundary recovers almost entirely by FLIPS; the Steiner
    // "unproductive" events - where the segment crosses no usable face and we
    // fall back to a midpoint / grazing / no-crossing split - are the signature
    // of a boundary that local refinement CANNOT converge (fine curved or
    // non-convex surfaces). Conformable models accrue ~zero of these (Pipe: 0),
    // so a SMALL total cap separates success from grind. It is machine-
    // independent (a count, not a time) and size-robust (the count tracks
    // non-convergence, not geometry size - a large conformable model still has
    // ~zero); a tiny size-proportional term gives big inputs slight headroom.
    // Each unproductive attempt is O(tets), so the small cap also bounds wasted
    // work. (A consecutive-streak test does NOT work here: the cascade
    // interleaves the occasional productive real-crossing split, resetting any
    // streak while never converging.)
    let max_unproductive = std::env::var("YAMM_MAX_UNPRODUCTIVE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or((n_seg0 / 256).clamp(32, 96));
    let max_seg_depth: u32 = std::env::var("YAMM_MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(16);
    // ── WORK BOUND (issue #136) ──
    // The two dominant costs of this pass are both O(#tets) PER OPERATION.
    // Measured on AnnularSector (656 splits, 1.4s, YAMM_SEG_PROF):
    //
    //   insert=0.82s (52%)  fallback(O#tets)=0.61s (39%)
    //   flips=0.03  crossing=0.00  split_faces=0.00  protected=0.12  lawson=0.04
    //
    // `insert_steiner_local`'s point location is a WALK bounded by 2·#tets, and
    // the crossing rescans (`first_segment_crossing_point` and friends) are full
    // O(#tets) sweeps. So the pass costs roughly (splits + probes) × #tets - and
    // nothing bounded that product. The existing caps count splits alone, which
    // is mesh-size-blind: on a fine boundary BlanketModule (48140 boundary
    // triangles, 90550 tets) grinds for over 15 minutes and then FAILS, after
    // which the legacy mesh is kept anyway. The entire grind is waste.
    //
    // Every carve that SUCCEEDS is cheap. Measured over 14 of the 15 zoo solids
    // that DEPEND on the carve (they fail with YAMM_NO_CONFORMING=1): splits
    // 0..377 and ZERO unproductive events in every single one. The worst product
    // is SphereWithMultipleHoles at 377 splits × 17065 tets ≈ 6.4e6 tet-visits;
    // next are Ellipsoid 3.4e6 and EllipticCylinder 4.0e6. The default budget
    // below is ~15× the worst success, so it cannot cost a case that would have
    // converged - it only truncates grinds that were going to fail.
    //
    // Counted in tet-visits rather than seconds, so the same input conforms or
    // bails identically on a fast or a slow CPU (the same reason the other caps
    // avoid a wall clock).
    //
    // CALIBRATION. The default is 11x the most expensive carve measured that
    // SUCCEEDS, so it cannot cost a case that would have converged:
    //
    //   LShaped                 4.5e5   exact
    //   Ellipsoid               1.6e8   exact
    //   EllipticCylinder        2.5e8   exact
    //   SphereWithMultipleHoles 3.5e8   exact  <- worst success
    //
    // An earlier 1e8 default was measured to REGRESS EllipticCylinder (1.7e-13 ->
    // 9.5e-5) by bailing on a healthy carve, which is what this headroom is for.
    let max_work: u64 = std::env::var("YAMM_MAX_WORK")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(4_000_000_000);
    let tet_scale = tets.tets.len().max(1) as u64;
    // Iterations that reach the expensive region (past flip/vertex-on-edge
    // recovery, which are O(degree) and cheap). Each one may run an O(#tets)
    // crossing sweep, so this is the probe count in the cost model above.
    let mut n_probe: u64 = 0;
    let mut n_iter: u64 = 0;
    let heartbeat: u64 = std::env::var("YAMM_HEARTBEAT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(5000);
    // Divergence bail threshold (see the bail site in the loop below). Overridable
    // so the "is this real divergence or transient sub-segment churn?" question is
    // answerable by measurement instead of a rebuild.
    let max_queue = std::env::var("YAMM_MAX_QUEUE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| n_seg0.max(512));
    let mut splits = 0usize;
    let dbg = std::env::var("YAMM_CARVE_DBG").is_ok();
    // Convergence-trace interval (splits between `[seg TRACE]` lines).
    let trace_every = std::env::var("YAMM_TRACE_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(500);
    let mut n_flip_ok = 0usize;
    let mut n_already = 0usize;
    let mut n_midpoint = 0usize;
    let mut n_grazed = 0usize;
    let mut n_no_cross = 0usize;
    // No-crossing fallback accounting (which layer actually resolves the
    // near-tangent class, and whether its insertion lands).
    let mut n_ec_found = 0usize;
    let mut n_ec_none = 0usize;
    let mut n_ec_ins_ok = 0usize;
    let mut n_rp_found = 0usize;
    let mut n_rp_ins_ok = 0usize;
    // Coplanar-region recovery (issue #31): kill switch + failure memo (verts
    // of regions whose analysis bailed, so later segments of the same broken
    // region skip the region BFS).
    let no_copl = std::env::var("YAMM_NO_COPL_REGION").is_ok();
    let mut copl_failed: HashSet<usize> = HashSet::default();
    let mut n_copl_ok = 0usize;
    // ── Volume-invariant probe (issue #37, YAMM_VOLCHECK): every operation in
    // this pass must preserve the total finite tet volume (the mesh partitions
    // the hull). A nonzero Δ after an op pinpoints an overlapping/gapping
    // commit. Diagnostic only: O(#tets) per probe.
    let volck = std::env::var("YAMM_VOLCHECK").is_ok();
    let total_vol = |tets: &Delaunay3D| -> f64 {
        let mut s = 0.0;
        for i in 0..tets.tets.len() {
            if !tets.is_live(i) || is_hull_tet(&tets.tets[i]) {
                continue;
            }
            let v = tets.tets[i].verts;
            s += orient_3d(
                tets.vertices[v[0]],
                tets.vertices[v[1]],
                tets.vertices[v[2]],
                tets.vertices[v[3]],
            )
            .abs()
                / 6.0;
        }
        s
    };
    let mut v_prev = if volck { total_vol(tets) } else { 0.0 };
    macro_rules! volprobe {
        ($label:expr, $a:expr, $b:expr) => {
            if volck {
                let v = total_vol(tets);
                if (v - v_prev).abs() > 1e-9 {
                    eprintln!(
                        "    [VOLCHECK] Δ={:+.6} at {} seg({},{})",
                        v - v_prev,
                        $label,
                        $a,
                        $b
                    );
                }
                v_prev = v;
            }
        };
    }

    // ── Per-phase timers (issue #37, gated by YAMM_SEG_PROF) ──
    // Identified `lawson_restore` as ~71% of recovery (since converted to
    // O(degree)); now the lens for the next per-split costs (insert, protected).
    let prof = std::env::var("YAMM_SEG_PROF").is_ok();
    let mut t_flips = 0.0f64;
    let mut t_crossing = 0.0f64;
    let mut t_fallback = 0.0f64;
    let mut t_insert = 0.0f64;
    let mut t_split_faces = 0.0f64;
    let mut t_protected = 0.0f64;
    let mut t_lawson = 0.0f64;
    let mut t_children = 0.0f64;
    macro_rules! timed {
        ($acc:ident, $body:expr) => {{
            if prof {
                let _t = std::time::Instant::now();
                let r = $body;
                $acc += _t.elapsed().as_secs_f64();
                r
            } else {
                $body
            }
        }};
    }
    // Prints the per-phase profile + O(#tets) fallback-fire counts. Called at
    // EVERY exit (incl. the early `return false` bails) so a max_splits/depth/
    // no-progress bail still reports where the time went.
    macro_rules! dump_prof {
        () => {{
            if dbg {
                use std::sync::atomic::Ordering::Relaxed;
                eprintln!(
                    "    [seg DBG] O(#tets) fallback fires: first_seg_crossing={} incl_endpoints={} | finddir_boundary_in_steiner={} | coplanar_region_ok={}",
                    FALLBACK_FSCP.load(Relaxed),
                    FALLBACK_FSCP_INCL.load(Relaxed),
                    FINDDIR_BOUNDARY.load(Relaxed),
                    n_copl_ok,
                );
                eprintln!(
                    "    [seg DBG] budget: splits={splits}/{max_splits} queue_cap={max_queue} unproductive={}/{max_unproductive} depth_cap={max_seg_depth} work={}/{max_work} (probes={n_probe} tets={tet_scale}) flip_marches_max={}/{FLIP_RECOVER_BUDGET} copl_region_max={}",
                    n_midpoint + n_grazed + n_no_cross,
                    (splits as u64 + n_probe) * tet_scale,
                    FLIP_BUDGET_HIGH_WATER.load(std::sync::atomic::Ordering::Relaxed),
                    super::coplanar::REGION_HIGH_WATER.load(std::sync::atomic::Ordering::Relaxed)
                );
                eprintln!(
                    "    [seg DBG] no-cross fallbacks: inplane_edge found={n_ec_found} (insert ok={n_ec_ins_ok}) none={n_ec_none} | refpt found={n_rp_found} (insert ok={n_rp_ins_ok}) | midpoint={n_midpoint} flip_ok={n_flip_ok} already={n_already} grazed={n_grazed} no_cross={n_no_cross}"
                );
            }
            if prof {
                eprintln!(
                    "    [seg PROF] splits={splits} flips={t_flips:.2}s crossing(finddir)={t_crossing:.2}s fallback(O#tets)={t_fallback:.2}s insert={t_insert:.2}s split_faces={t_split_faces:.2}s protected_sets={t_protected:.2}s lawson={t_lawson:.2}s children={t_children:.2}s"
                );
            }
        }};
    }

    // Protected boundary sets (all refined surface segments/faces). Threaded into
    // every flip-based recovery so recovering one segment cannot destroy an
    // already-recovered neighbour (issue #37). Rebuilt only when `cur_faces`
    // changes (a Steiner / AcrossVert split) - O(#faces) per split, not per call.
    let (mut psegs, mut pfaces, mut pinc) = protected_sets_from_faces(cur_faces);

    while let Some((a, b, depth)) = queue.pop_front() {
        n_iter += 1;
        // Progress heartbeat: distinguishes "looping fast but not converging"
        // from "wedged inside one segment's recovery". The split-keyed
        // [seg TRACE] cannot, because a pass making zero splits prints nothing.
        if dbg && n_iter.is_multiple_of(heartbeat) {
            eprintln!(
                "    [seg BEAT] iter={n_iter} queue={} splits={splits} probes={n_probe} flip_ok={n_flip_ok} already={n_already} copl_ok={n_copl_ok} seg=({a},{b})",
                queue.len()
            );
        }
        if n_midpoint + n_grazed + n_no_cross > max_unproductive {
            if dbg {
                eprintln!(
                    "    [seg DBG] no-progress bail: {} unproductive > cap {} (splits={splits})",
                    n_midpoint + n_grazed + n_no_cross,
                    max_unproductive
                );
            }
            dump_prof!();
            return false; // not converging - clean fallback (gate-validated)
        }
        // DIVERGENCE bail (issue #57): a converging recovery's queue only
        // shrinks (modulo transient sub-segment churn); a queue that has
        // OUTGROWN the entire initial segment count means every split breeds
        // more missing sub-segments than it closes - the measured signature
        // of small-input-angle (tangential surface intersection) divergence,
        // where refinement provably cannot terminate. Bail early instead of
        // grinding to the split budget.
        if (splits as u64 + n_probe) * tet_scale > max_work {
            if dbg {
                eprintln!(
                    "    [seg DBG] work bail: {} tet-visits > cap {max_work} (splits={splits} probes={n_probe} tets={tet_scale})",
                    (splits as u64 + n_probe) * tet_scale
                );
            }
            dump_prof!();
            return false; // grinding, and a grind here always ends in fallback
        }
        if queue.len() > max_queue {
            if dbg {
                eprintln!(
                    "    [seg DBG] divergence bail: queue {} > cap {} (initial segments {}, splits={splits})",
                    queue.len(),
                    max_queue,
                    n_seg0
                );
            }
            dump_prof!();
            return false;
        }
        if edge_exists_in_tets(tets, a, b) {
            n_already += 1;
            continue;
        }
        if depth >= max_seg_depth {
            if dbg {
                eprintln!("    [seg DBG] segment ({a},{b}) exceeded depth {max_seg_depth} - bail");
            }
            dump_prof!();
            return false; // sliver cascade - fall back rather than loop
        }
        // Try flip-based recovery first (cheap when it works).
        let fr = timed!(
            t_flips,
            recover_edge_by_flips_protected(tets, a, b, &psegs, &pfaces)
        );
        volprobe!("flip-recover", a, b);
        if fr {
            n_flip_ok += 1;
            continue;
        }
        // COPLANAR FLAT-FACE recovery (LShaped reflex faces): the segment (a, b)
        // is blocked by a wrong-diagonal edge (u, v) that is COPLANAR with it
        // (the two diagonals of a flat-face quad) and whose ring is larger than
        // the n==3/4 `flip_nm` can reduce. Clear it with the general hull-aware
        // ring reducer, protecting every already-recovered boundary element
        // (built from the refined `cur_faces`) so the reduction never destroys a
        // recovered segment/face. Runs AFTER the normal flip recovery and BEFORE
        // any Steiner split. This fires only for coplanar-quad obstructions, so
        // curved thin walls (Pipe/TWC - no such region) are unaffected.
        // Weighted against the work bound by its own iteration cap: one attempt
        // runs up to COPL_MAX_ITERS march/ring-clear rounds, each of which can be
        // O(#tets), so charging it as a single probe under-counts it ~32x. This
        // path is where a planar-dominated boundary spends nearly all of its time
        // (see recover_segment_coplanar_ring), so the bound has to see its real
        // cost or it cannot bound anything.
        n_probe += COPL_MAX_ITERS as u64;
        let cr = timed!(
            t_flips,
            recover_segment_coplanar_ring(tets, a, b, &psegs, &pfaces)
        );
        volprobe!("coplanar", a, b);
        if cr {
            n_flip_ok += 1;
            continue;
        }
        // An existing vertex `v` may lie ON the open segment (a, b) (a T-junction
        // in the refined surface, or a vertex the chord passes through). No
        // Steiner point is needed: split the segment at v and recover the two
        // halves. `finddirection` reports this as ACROSSVERT, marching from a and
        // then b. This must run BEFORE the crossing-point Steiner path, which
        // would otherwise find no usable crossing (the obstruction is a vertex,
        // not a face/edge) and degenerate into non-converging midpoint halving.
        let av = match finddirection(tets, a, b) {
            Obstruction::AcrossVert { v } => Some(v),
            Obstruction::Boundary => match finddirection(tets, b, a) {
                Obstruction::AcrossVert { v } => Some(v),
                _ => None,
            },
            _ => None,
        };
        if let Some(v) = av {
            // Subdivide the incident facets at the existing vertex v (it lies on
            // edge (a,b), so each triangle (a,b,w) → (a,v,w),(v,b,w)).
            let opp_w = split_incident_faces_on_edge(cur_faces, a, b, v);
            // cur_faces changed → refresh the protected sets.
            let (ns, nf, ni) = protected_sets_from_faces(cur_faces);
            psegs = ns;
            pfaces = nf;
            pinc = ni;
            // Recover the two halves AND every new cross-edge (v, w).
            for w in std::iter::once(a)
                .chain(std::iter::once(b))
                .chain(opp_w.iter().copied())
            {
                let (u, w) = (v, w);
                if !edge_exists_in_tets(tets, u, w)
                    && !recover_edge_by_flips_protected(tets, u, w, &psegs, &pfaces)
                {
                    queue.push_back((u.min(w), u.max(w), depth + 1));
                }
            }
            volprobe!("acrossvert", a, b);
            continue;
        }
        // COPLANAR-REGION recovery (issue #31, the endgame): a segment in the
        // interior of a FLAT surface region whose obstruction survived all the
        // flip machinery is the in-plane pillow class - zero-volume tets
        // bridging two triangulations of the same planar region, unfixable by
        // any 3-D local operation (every predicate is exactly zero). Replace
        // the whole flat sandwich by a stack realizing a 2-D constrained
        // re-triangulation of the maximal coplanar region (see `coplanar`).
        // Exactly volume-preserving, transactional, and side-tets-untouched;
        // fires only when ALL surface faces at (a, b) are exactly coplanar, so
        // curved thin walls (Pipe/TWC) never enter. Runs BEFORE the Steiner
        // path, whose in-plane crossing points are the known-degenerate
        // midpoint cascade.
        if !no_copl {
            match super::coplanar::recover_segment_coplanar_region(
                tets,
                a,
                b,
                &pinc,
                &psegs,
                &mut copl_failed,
            ) {
                super::coplanar::CoplanarOutcome::Recovered => {
                    n_copl_ok += 1;
                    volprobe!("coplanar-region", a, b);
                    continue;
                }
                super::coplanar::CoplanarOutcome::SplitSegment { seg: (u, v), at: w } => {
                    // A region vertex lies exactly ON surface segment (u, v)
                    // (often (a, b) itself): split it there - the in-plane
                    // analogue of the AcrossVert resolution above - and
                    // re-attempt (a, b) once the blocker is gone.
                    let opp_w = split_incident_faces_on_edge(cur_faces, u, v, w);
                    let (ns, nf, ni) = protected_sets_from_faces(cur_faces);
                    psegs = ns;
                    pfaces = nf;
                    pinc = ni;
                    for x in std::iter::once(u)
                        .chain(std::iter::once(v))
                        .chain(opp_w.iter().copied())
                    {
                        if !edge_exists_in_tets(tets, w, x)
                            && !recover_edge_by_flips_protected(tets, w, x, &psegs, &pfaces)
                        {
                            queue.push_back((w.min(x), w.max(x), depth + 1));
                        }
                    }
                    if (u.min(v), u.max(v)) != (a.min(b), a.max(b)) {
                        queue.push_back((a.min(b), a.max(b), depth));
                    }
                    volprobe!("coplanar-split", a, b);
                    continue;
                }
                super::coplanar::CoplanarOutcome::Failed => {}
            }
        }
        // Steiner: insert a point lying ON the missing segment, at the location
        // where the segment FIRST crosses a tet face (the entry crossing). That
        // point lies on the original surface (the segment is straight) AND on a
        // mesh face/edge, so the corresponding LOCAL split is conformal and -
        // crucially - REMOVES that crossing. This is the convergent move: each
        // insertion eliminates one face-crossing of the segment, so the segment
        // becomes recoverable after finitely many splits (TetGen's
        // `splitsegment` reference-point scheme). A pure midpoint split does NOT
        // remove crossings and degenerates into endless halving.
        if splits >= max_splits {
            if dbg {
                let dx = tets.vertices[a][0] - tets.vertices[b][0];
                let dy = tets.vertices[a][1] - tets.vertices[b][1];
                let dz = tets.vertices[a][2] - tets.vertices[b][2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                eprintln!(
                    "    [seg DBG] max_splits={max_splits} hit; seg ({a},{b}) len={len:.5} queue~{} flip_ok={n_flip_ok} already={n_already} midpoint={n_midpoint} grazed={n_grazed} no_cross={n_no_cross} splits={splits}",
                    queue.len()
                );
            }
            dump_prof!();
            return false; // perf bail - gate falls back to legacy
        }
        let pa = tets.vertices[a];
        let pb = tets.vertices[b];
        let seg_len2 = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
        // Find the Steiner crossing point via `finddirection` FIRST: it returns
        // the obstruction face even when that face is shared with an
        // endpoint-incident tet (the dominant "none" class), where the legacy
        // `first_segment_crossing_point` (which skips endpoint tets) finds
        // nothing. Fall back to the legacy scan only if the march hits a
        // boundary/degenerate dead end.
        let mut ff_obs = timed!(t_crossing, finddirection(tets, a, b));
        if matches!(ff_obs, Obstruction::Boundary) {
            FINDDIR_BOUNDARY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // The march from `a` exits the hull (reflex chord / hull-adjacent
            // start). The obstruction is still discoverable from the OTHER
            // end: re-march b -> a. The crossing point is a point on the
            // segment either way; the vertex-on-edge resolution below is
            // endpoint-symmetric (issues #31/#47 flat-cap class - half its
            // members are only marchable from one side).
            let rev = finddirection(tets, b, a);
            if !matches!(rev, Obstruction::Boundary) {
                ff_obs = rev;
            }
        }
        let ff_pt = match ff_obs {
            Obstruction::AcrossFace { face, .. } => segment_face_crossing_point(tets, a, b, face),
            Obstruction::AcrossEdge { u, v } => edge_segment_crossing_point(tets, a, b, u, v)
                .or_else(|| edge_segment_crossing_point(tets, b, a, u, v)),
            // Recovered / AcrossVert are handled by the flip path above; Boundary
            // falls back to the legacy scan.
            _ => None,
        };
        // VERTEX-ON-EDGE degeneracy (issues #31/#47, the flat-cap class): the
        // march reports a crossing edge but the crossing parameter is exactly
        // at a segment ENDPOINT - the endpoint vertex lies ON the open edge
        // (u, v) (a surface Steiner vertex on an in-plane cap edge; measured
        // distances 0..4.4e-16). No usable Steiner point exists, so this
        // previously degenerated into the unproductive midpoint cascade.
        // Resolve TOPOLOGICALLY: split the edge AT the existing vertex
        // ({a,b,x,y} ring -> {a,w,x,y}+{w,b,x,y}; exactly volume-preserving),
        // unless (u, v) is itself a protected boundary segment. Re-queue and
        // continue - a productive step that consumes no split budget.
        if ff_pt.is_none() {
            if let Obstruction::AcrossEdge { u, v } = ff_obs {
                let key = (u.min(v), u.max(v));
                if dbg && psegs.contains(&key) {
                    eprintln!("    [voe DBG] ({a},{b}): obstruction edge ({u},{v}) is PROTECTED");
                }
                if !psegs.contains(&key) {
                    let pu = tets.vertices[u];
                    let pv = tets.vertices[v];
                    let d2v = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
                    let l2 = d2v[0] * d2v[0] + d2v[1] * d2v[1] + d2v[2] * d2v[2];
                    let mut done = false;
                    if l2 > 0.0 {
                        for &e in &[a, b] {
                            let pe = tets.vertices[e];
                            let pu_pe = [pe[0] - pu[0], pe[1] - pu[1], pe[2] - pu[2]];
                            let t_on =
                                (pu_pe[0] * d2v[0] + pu_pe[1] * d2v[1] + pu_pe[2] * d2v[2]) / l2;
                            if !(t_on > 1e-6 && t_on < 1.0 - 1e-6) {
                                continue;
                            }
                            let proj = [
                                pu[0] + t_on * d2v[0],
                                pu[1] + t_on * d2v[1],
                                pu[2] + t_on * d2v[2],
                            ];
                            let dist2 = (pe[0] - proj[0]).powi(2)
                                + (pe[1] - proj[1]).powi(2)
                                + (pe[2] - proj[2]).powi(2);
                            // On the line within fp noise, relative to the
                            // edge's own scale.
                            if dist2 <= l2 * 1e-24 {
                                let ok = tets.split_edge_at_vertex(u, v, e);
                                if dbg {
                                    eprintln!(
                                        "    [voe DBG] ({a},{b}): endpoint {e} ON edge ({u},{v}) t={t_on:.4} -> split {}",
                                        if ok { "OK" } else { "REJECTED" }
                                    );
                                }
                                if ok {
                                    done = true;
                                    break;
                                }
                            } else if dbg && dist2 <= l2 * 1e-12 {
                                eprintln!(
                                    "    [voe DBG] ({a},{b}): endpoint {e} NEAR edge ({u},{v}) dist2={dist2:e} (tol {:e})",
                                    l2 * 1e-24
                                );
                            }
                        }
                    }
                    if done {
                        queue.push_back((a.min(b), a.max(b), depth));
                        volprobe!("vertex-on-edge", a, b);
                        continue;
                    }
                }
            }
        }
        // Past flips and the vertex-on-edge resolution: everything below can run
        // an O(#tets) sweep, so this is where the work bound's probe count ticks.
        n_probe += 1;
        let mut cp = ff_pt;
        if cp.is_none() {
            cp = timed!(t_fallback, first_segment_crossing_point(tets, a, b));
        }
        if cp.is_none() {
            // Last resort (reflex chord exits the hull at an endpoint, so neither
            // finddirection nor the skip-endpoint scan sees the obstruction):
            // scan ALL finite faces, endpoint-incident tets included, for the
            // first interior crossing along the ray. Splitting there carves the
            // chord back into the meshed region.
            cp = timed!(
                t_fallback,
                first_segment_crossing_point_incl_endpoints(tets, a, b)
            );
        }
        let endpoint_clearance2 = |p: &[f64; 3], band: f64| -> bool {
            let da2 = (p[0] - pa[0]).powi(2) + (p[1] - pa[1]).powi(2) + (p[2] - pa[2]).powi(2);
            let db2 = (p[0] - pb[0]).powi(2) + (p[1] - pb[1]).powi(2) + (p[2] - pb[2]).powi(2);
            let band2 = seg_len2 * band * band;
            da2 > band2 && db2 > band2
        };
        // Prefer a crossing clear of both endpoints (> GRAZE_T_BAND of the
        // segment length): splitting near an endpoint makes a micro sub-
        // segment and poor-quality children.
        let mut crossing_pt = cp.filter(|p| endpoint_clearance2(p, GRAZE_T_BAND));
        if crossing_pt.is_none() {
            // The nearest crossing grazes an endpoint. If the chord crosses
            // more faces further along, take the first one PAST the band.
            crossing_pt = timed!(
                t_fallback,
                first_segment_crossing_point_past_graze(tets, a, b)
            );
        }
        if crossing_pt.is_none() {
            // Every crossing grazes an endpoint - the issue-#57 class: the
            // chord runs almost entirely inside one or two large tets, with
            // an endpoint nearly coplanar with the blocking face (which is
            // also why the flip arsenal rejected it: the 2-3 flip's child is
            // near-degenerate). Splitting AT the grazing crossing is
            // conformal, removes the crossing, and CONVERGES here (the
            // sub-segments then recover by flips) - the historical blanket
            // rejection turned this whole class into the midpoint cascade
            // and a guaranteed bail. Keep only a micro guard: a crossing
            // within 1e-6 of an endpoint would make a near-duplicate vertex;
            // leave those to the fallback paths.
            crossing_pt = cp.filter(|p| endpoint_clearance2(p, 1e-6));
        }
        if crossing_pt.is_none() {
            // Layer 4b (issue #57): a mesh edge passes exactly THROUGH one of
            // the segment endpoints - the voe class, but invisible to the
            // march in the near-tangent neighbourhood. Resolve topologically
            // (no Steiner point, no budget) and re-attempt the segment.
            if let Some((u, v, e)) = timed!(t_fallback, edge_through_endpoint(tets, a, b, &psegs)) {
                if tets.split_edge_at_vertex(u, v, e) {
                    if dbg {
                        eprintln!(
                            "    [seg DBG] edge-through-endpoint: split ({u},{v}) at {e} for ({a},{b})"
                        );
                    }
                    queue.push_back((a.min(b), a.max(b), depth));
                    volprobe!("edge-through-endpoint", a, b);
                    continue;
                }
            }
        }
        // NOTE (issue #57 post-mortem): a "dust-coplanar covering face"
        // exemption from the no-progress cap was tried here - the theory
        // being that midpoint splits on such faces are convergent
        // conforming-Delaunay refinement. The convergence trace refuted it on
        // the only models that exercise it (tangential surface intersections:
        // the queue GROWS super-linearly while segment lengths collapse to
        // 1e-5 - the small-input-angle divergence), so the exemption only
        // extended doomed grinds. The cap stays unconditional.
        if crossing_pt.is_none() {
            // Count UNCONDITIONALLY: this feeds the no-progress bail, and the
            // historical dbg-gated increment made production runs grind for
            // minutes through doomed reference/midpoint cascades that a
            // YAMM_CARVE_DBG run cut off in seconds. Only the grazed/no-cross
            // CLASSIFICATION (an extra O(#tets) scan) stays dbg-only.
            let grazed =
                dbg && (ff_pt.is_some() || first_segment_crossing_point(tets, a, b).is_some());
            if grazed {
                n_grazed += 1;
            } else {
                n_no_cross += 1;
            }
            // Sample the first few unproductive segments (issue #47 cluster B,
            // the hull-chord class): what does the march actually see?
            if dbg && n_grazed + n_no_cross <= 8 {
                let obs_ab = finddirection(tets, a, b);
                let obs_ba = finddirection(tets, b, a);
                eprintln!(
                    "    [seg DBG] UNPRODUCTIVE ({a},{b}) len={:.4} grazed={grazed} depth={depth} obs(a->b)={obs_ab:?} obs(b->a)={obs_ba:?} pa={pa:?} pb={pb:?}",
                    seg_len2.sqrt()
                );
                // Nearest off-segment vertices (the issue-#57 blocker probe):
                // distance to the OPEN segment, projection within the band.
                let mut near: Vec<(f64, usize, f64)> = Vec::new();
                let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                for (vi, pv) in tets.vertices.iter().enumerate() {
                    if vi == a || vi == b {
                        continue;
                    }
                    let ap = [pv[0] - pa[0], pv[1] - pa[1], pv[2] - pa[2]];
                    let t = (ap[0] * d[0] + ap[1] * d[1] + ap[2] * d[2]) / seg_len2;
                    if !(GRAZE_T_BAND..=1.0 - GRAZE_T_BAND).contains(&t) {
                        continue;
                    }
                    let proj = [pa[0] + t * d[0], pa[1] + t * d[1], pa[2] + t * d[2]];
                    let dist2 = (pv[0] - proj[0]).powi(2)
                        + (pv[1] - proj[1]).powi(2)
                        + (pv[2] - proj[2]).powi(2);
                    near.push((dist2, vi, t));
                }
                near.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                near.truncate(3);
                for (d2, vi, t) in near {
                    eprintln!(
                        "    [seg DBG]   near-vert {vi} dist={:.3e} t={t:.3} (rel {:.3e})",
                        d2.sqrt(),
                        d2.sqrt() / seg_len2.sqrt()
                    );
                }
                // Nearest edge-crossing candidates regardless of the layer-4
                // filters (to see WHY nearest_inplane_edge_crossing skipped).
                {
                    let d1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let l1 = seg_len2;
                    let mut seen: HashSet<(usize, usize)> = HashSet::default();
                    let mut cands: Vec<(f64, f64, f64, usize, usize, bool)> = Vec::new();
                    for (i, tet) in tets.tets.iter().enumerate() {
                        if !tets.is_live(i) || is_hull_tet(tet) {
                            continue;
                        }
                        let v = tet.verts;
                        for &[ei, ej] in &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] {
                            let (u, w) = (v[ei], v[ej]);
                            if u == a || u == b || w == a || w == b {
                                continue;
                            }
                            let key = (u.min(w), u.max(w));
                            if !seen.insert(key) {
                                continue;
                            }
                            let pu = tets.vertices[u];
                            let pw = tets.vertices[w];
                            let d2v = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
                            let l2 = d2v[0] * d2v[0] + d2v[1] * d2v[1] + d2v[2] * d2v[2];
                            if l2 <= 0.0 {
                                continue;
                            }
                            let d12 = d1[0] * d2v[0] + d1[1] * d2v[1] + d1[2] * d2v[2];
                            let denom = l1 * l2 - d12 * d12;
                            if denom <= l1 * l2 * 1e-12 {
                                continue;
                            }
                            let rr = [pu[0] - pa[0], pu[1] - pa[1], pu[2] - pa[2]];
                            let rd1 = rr[0] * d1[0] + rr[1] * d1[1] + rr[2] * d1[2];
                            let rd2 = rr[0] * d2v[0] + rr[1] * d2v[1] + rr[2] * d2v[2];
                            let t = (rd1 * l2 - rd2 * d12) / denom;
                            let s = (rd1 * d12 - rd2 * l1) / denom;
                            if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&s) {
                                continue;
                            }
                            let p1 = [pa[0] + t * d1[0], pa[1] + t * d1[1], pa[2] + t * d1[2]];
                            let p2 = [pu[0] + s * d2v[0], pu[1] + s * d2v[1], pu[2] + s * d2v[2]];
                            let dist2 = (p1[0] - p2[0]).powi(2)
                                + (p1[1] - p2[1]).powi(2)
                                + (p1[2] - p2[2]).powi(2);
                            cands.push((dist2, t, s, u, w, psegs.contains(&key)));
                        }
                    }
                    cands.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                    cands.truncate(3);
                    for (d2, t, s, u, w, prot) in cands {
                        eprintln!(
                            "    [seg DBG]   xedge ({u},{w}) dist2={d2:.3e} (thresh {:.3e}) t={t:.3} s={s:.3} protected={prot}",
                            l1 * 1e-24
                        );
                    }
                }
                // Segment walk (issue #57 layer 4): sample points along the
                // open segment and report which tet contains each (full
                // scan, exact orientations) - ground truth for what the
                // marching/crossing machinery should have seen.
                if std::env::var("YAMM_SEG_WALK").is_ok() {
                    let nsamp = 16usize;
                    let mut chain: Vec<String> = Vec::new();
                    for k in 1..nsamp {
                        let t = k as f64 / nsamp as f64;
                        let p = lerp(pa, pb, t);
                        let mut found = String::from("NONE");
                        for ci in 0..tets.tets.len() {
                            if !tets.is_live(ci) || is_hull_tet(&tets.tets[ci]) {
                                continue;
                            }
                            let v = tets.tets[ci].verts;
                            let (qa, qb, qc, qd) = (
                                tets.vertices[v[0]],
                                tets.vertices[v[1]],
                                tets.vertices[v[2]],
                                tets.vertices[v[3]],
                            );
                            let v6 = orient_3d(qa, qb, qc, qd);
                            if v6 <= 0.0 {
                                continue;
                            }
                            let oc = [
                                orient_3d(p, qb, qc, qd),
                                orient_3d(qa, p, qc, qd),
                                orient_3d(qa, qb, p, qd),
                                orient_3d(qa, qb, qc, p),
                            ];
                            if oc.iter().all(|&x| x >= 0.0) {
                                let nz = oc.iter().filter(|&&x| x == 0.0).count();
                                found = format!("{ci}(z{nz},v{v6:.1e})");
                                break;
                            }
                        }
                        chain.push(format!("t={:.2}:{found}", t));
                    }
                    eprintln!("    [seg WALK] ({a},{b}): {}", chain.join(" "));
                }
                // Obstructing-face anatomy (issue #57 layer 4): geometry of
                // the blocking face and its two supports.
                for obs in [&obs_ab, &obs_ba] {
                    let Obstruction::AcrossFace { ti, face } = obs else {
                        continue;
                    };
                    let (p0, p1, p2) = (
                        tets.vertices[face[0]],
                        tets.vertices[face[1]],
                        tets.vertices[face[2]],
                    );
                    let oa = orient_3d(p0, p1, p2, pa);
                    let ob = orient_3d(p0, p1, p2, pb);
                    let e01 = ((p1[0] - p0[0]).powi(2)
                        + (p1[1] - p0[1]).powi(2)
                        + (p1[2] - p0[2]).powi(2))
                    .sqrt();
                    let e12 = ((p2[0] - p1[0]).powi(2)
                        + (p2[1] - p1[1]).powi(2)
                        + (p2[2] - p1[2]).powi(2))
                    .sqrt();
                    let e20 = ((p0[0] - p2[0]).powi(2)
                        + (p0[1] - p2[1]).powi(2)
                        + (p0[2] - p2[2]).powi(2))
                    .sqrt();
                    let volt = {
                        let v = tets.tets[*ti].verts;
                        orient_3d(
                            tets.vertices[v[0]],
                            tets.vertices[v[1]],
                            tets.vertices[v[2]],
                            tets.vertices[v[3]],
                        )
                    };
                    // The neighbour across the face.
                    let mut voln = f64::NAN;
                    let mut apexes = (usize::MAX, usize::MAX);
                    for fi in 0..4 {
                        if delaunay3d::faces_match(
                            &delaunay3d::opposite_face(tets.tets[*ti].verts, fi),
                            face,
                        ) {
                            apexes.0 = tets.tets[*ti].verts[fi];
                            let nb = tets.tets[*ti].adj[fi];
                            if nb != usize::MAX && tets.is_live(nb) {
                                let v = tets.tets[nb].verts;
                                if !v.contains(&INFINITE) {
                                    voln = orient_3d(
                                        tets.vertices[v[0]],
                                        tets.vertices[v[1]],
                                        tets.vertices[v[2]],
                                        tets.vertices[v[3]],
                                    );
                                    apexes.1 = *v
                                        .iter()
                                        .find(|&&x| !face.contains(&x))
                                        .unwrap_or(&usize::MAX);
                                }
                            }
                            break;
                        }
                    }
                    eprintln!(
                        "    [seg DBG]   obs-face {face:?} edges=({e01:.3},{e12:.3},{e20:.3}) oa={oa:+.3e} ob={ob:+.3e} vol(ti)={volt:+.3e} vol(nb)={voln:+.3e} apexes={apexes:?}"
                    );
                }
            }
        }
        let m_opt = match crossing_pt {
            Some(p) => timed!(t_insert, tets.insert_steiner_local(p)),
            None => {
                // No usable interior crossing at all - the near-tangent class
                // (issue #57). Most specific first: a mesh edge crossing the
                // segment in-plane at dust distance (layer 4 - split the
                // blocking edge at the crossing point, exactly the convergent
                // 2-D constrained-insertion move). Then the TetGen-style
                // vertex reference point. Clamped midpoints only as the last
                // resort; the depth/unproductive caps still bound everything.
                let ec = timed!(
                    t_fallback,
                    nearest_inplane_edge_crossing(tets, a, b, &psegs)
                );
                let mut got = match ec {
                    Some(p) => timed!(t_insert, tets.insert_steiner_local(p)),
                    None => None,
                };
                if ec.is_some() {
                    n_ec_found += 1;
                    if got.is_some() {
                        n_ec_ins_ok += 1;
                    }
                }
                if dbg && n_ec_found + n_ec_none <= 8 {
                    eprintln!(
                        "    [seg DBG] no-cross fallback ({a},{b}): inplane_edge={} insert={got:?}",
                        if ec.is_some() { "FOUND" } else { "none" }
                    );
                }
                if ec.is_none() {
                    n_ec_none += 1;
                }
                if got.is_none() {
                    let rp = timed!(t_fallback, reference_point_on_segment(tets, a, b));
                    got = match rp {
                        Some(p) => timed!(t_insert, tets.insert_steiner_local(p)),
                        None => None,
                    };
                    if rp.is_some() {
                        n_rp_found += 1;
                        if got.is_some() {
                            n_rp_ins_ok += 1;
                        }
                    }
                }
                if got.is_none() {
                    n_midpoint += 1;
                    for &t in &[0.5_f64, STEINER_SEG_CLAMP, 1.0 - STEINER_SEG_CLAMP] {
                        if let Some(m) =
                            timed!(t_insert, tets.insert_steiner_local(lerp(pa, pb, t)))
                        {
                            got = Some(m);
                            break;
                        }
                    }
                }
                got
            }
        };
        let Some(m) = m_opt else {
            if std::env::var("YAMM_CARVE_DBG").is_ok() {
                eprintln!(
                    "    [seg DBG] insert failed for segment ({a},{b}) after {splits} splits"
                );
            }
            dump_prof!();
            return false;
        };
        if m == a || m == b {
            // insert_steiner_local resolved the point to an EXISTING vertex
            // (3-zeros classification) that happens to be a segment endpoint -
            // nothing to split here. Count as unproductive and re-queue; the
            // depth/unproductive caps bound re-detection.
            n_midpoint += 1;
            queue.push_back((a.min(b), a.max(b), depth + 1));
            continue;
        }
        splits += 1;
        // Convergence trace (issue #57): queue trend vs split count - a
        // draining queue means the cascade converges and only needs budget; a
        // growing one means refinement is breeding sub-segments faster than
        // it closes them (the small-input-angle signature).
        if dbg && splits.is_multiple_of(trace_every) {
            let missing = queue
                .iter()
                .filter(|&&(u, w, _)| !edge_exists_in_tets(tets, u, w))
                .count();
            eprintln!(
                "    [seg TRACE] splits={splits} queue={} missing-in-queue={missing} seg=({a},{b}) len={:.2e} depth={depth}",
                queue.len(),
                seg_len2.sqrt()
            );
        }
        volprobe!("insert", a, b);

        // Refine cur_faces: every triangle carrying edge (a,b) is split by
        // connecting the new vertex m to the triangle's opposite vertex.
        let opp_w = timed!(
            t_split_faces,
            split_incident_faces_on_edge(cur_faces, a, b, m)
        );
        // cur_faces changed → refresh the protected sets (reused by both the
        // Lawson restoration below and the child recoveries that follow).
        {
            let (ns, nf, ni) = timed!(t_protected, protected_sets_from_faces(cur_faces));
            psegs = ns;
            pfaces = nf;
            pinc = ni;
        }

        // Re-establish local Delaunay-ness around the new vertex `m` by Lawson
        // flips (TetGen's lawsonflip3d). Without this the sub-segments/sub-facets
        // through `m` are usually NOT mesh edges/faces, recovery cannot close
        // them, and the split recurses without converging. Protect the boundary:
        // the active constraint is the segment line (a,b) that `m` lies on, plus
        // every already-recovered boundary segment/face, so no flip destroys a
        // recovered boundary element.
        {
            // The active constraint is `None`: Delaunay restoration around `m`
            // (which lies ON segment line (a,b)) NEEDS flips whose new elements
            // cross that line - forbidding them (an Edge/Face "do-not-cross"
            // guard) would stall the restoration and prevent the sub-segments
            // from becoming Delaunay. Boundary integrity is instead protected by
            // the segment/face SET membership checks below (a flip is rejected
            // only if it would DESTROY an already-present recovered element),
            // mirroring TetGen's issubseg/issubface guards in lawsonflip3d.
            let protect = ProtectedBoundary {
                active: Constraint::None,
                segments: &psegs,
                faces: &pfaces,
            };
            timed!(t_lawson, lawson_restore(tets, m, &protect));
        }
        volprobe!("lawson", a, b);

        // Eagerly recover the two split halves (a, m)/(m, b) AND every new
        // cross-edge (m, w) created by subdividing the incident facets - all are
        // edges of the refined surface and must be present for the facet pass to
        // close. (a, m) ends at the first crossing so it should close immediately;
        // re-queue (with incremented depth) any that still resist. Omitting the
        // (m, w) cross-edges left a residual class of un-recovered sub-segments
        // (issue #37) - they were never re-queued at all.
        for w in std::iter::once(a)
            .chain(std::iter::once(b))
            .chain(opp_w.iter().copied())
        {
            let (u, w) = (m, w);
            if !edge_exists_in_tets(tets, u, w)
                && !timed!(
                    t_children,
                    recover_edge_by_flips_protected(tets, u, w, &psegs, &pfaces)
                )
            {
                queue.push_back((u.min(w), u.max(w), depth + 1));
            }
        }
        volprobe!("children", a, b);
    }

    // Final check: every refined segment present.
    let mut ok = true;
    let mut check_set: HashSet<(usize, usize)> = HashSet::default();
    for f in cur_faces.iter() {
        for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            check_set.insert((u.min(v), u.max(v)));
        }
    }
    let mut n_missing = 0usize;
    for (a, b) in check_set {
        if !edge_exists_in_tets(tets, a, b) {
            ok = false;
            n_missing += 1;
        }
    }
    dump_prof!();
    if !ok && std::env::var("YAMM_CARVE_DBG").is_ok() {
        eprintln!(
            "    [seg DBG] after {splits} splits: {n_missing} refined segments still missing (max_splits={max_splits})"
        );
    }
    ok
}

/// In `cur_faces`, replace every triangle that contains edge (a, b) with its two
/// children obtained by splitting that edge at the new vertex `m` (which lies on
/// the edge): triangle (a, b, w) → (a, m, w) and (m, b, w). Triangles not
/// carrying (a, b) are untouched. The replacement preserves the surface exactly
/// (m is on segment a-b).
///
/// Returns the DISTINCT opposite vertices `w` of the split facets. Each split
/// introduces a NEW cross-edge `(m, w)` - a real edge of the refined surface
/// that must also be recovered. The caller queues those (the loop previously
/// re-queued only the two halves `(a, m)`/`(m, b)`, leaving the `(m, w)` cross-
/// edges silently un-recovered → a residual class of missing sub-segments, #37).
fn split_incident_faces_on_edge(
    cur_faces: &mut Vec<[usize; 3]>,
    a: usize,
    b: usize,
    m: usize,
) -> Vec<usize> {
    let mut out: Vec<[usize; 3]> = Vec::with_capacity(cur_faces.len() + 4);
    let mut opp: Vec<usize> = Vec::new();
    for &f in cur_faces.iter() {
        let has_a = f.contains(&a);
        let has_b = f.contains(&b);
        if has_a && has_b {
            // The third (opposite) vertex. A face with no third vertex is
            // already-degenerate debris - drop it rather than crash.
            let Some(&w) = f.iter().find(|&&x| x != a && x != b) else {
                debug_assert!(false, "degenerate surface face {f:?}");
                continue;
            };
            if w == m {
                // The face IS (a, b, m) with m on segment (a, b): a zero-area
                // sliver. Its split children (a,m,m)/(m,b,m) are nothing -
                // the face DISSOLVES: its two short edges (a,m),(m,b) are
                // exactly the split halves carried by the neighbouring
                // faces' children, so the surface stays edge-balanced.
                // (Arises when an existing vertex resolves the split point,
                // issue #57's on-vertex Steiner class.)
                continue;
            }
            out.push([a, m, w]);
            out.push([m, b, w]);
            if !opp.contains(&w) {
                opp.push(w);
            }
        } else {
            out.push(f);
        }
    }
    *cur_faces = out;
    opp
}

/// Recover ALL boundary facets (flips, then Steiner refinement), maintaining the
/// evolving refined boundary-triangle set `cur_faces`. Call AFTER
/// [`recover_segments_with_steiner`] so every facet's edges are already present.
/// Returns `true` iff every (refined) triangle in `cur_faces` is present as a
/// tet face afterwards.
///
/// For a facet still missing after flip recovery, place a Steiner point IN the
/// facet (its barycenter - guaranteed strictly interior, lies in the facet
/// plane so the surface is preserved) via `split_face_on_constraint`, replace
/// the triangle by its three children (a,b,m),(b,c,m),(c,a,m), and re-queue.
/// The three new edges (a,m),(b,m),(c,m) are created by the split, so the child
/// facets' edges are immediately present.
///
/// GUARANTEED TO TERMINATE: a Steiner split replaces a facet by three strictly
/// smaller facets; capped generously and bails (perf, never correctness).
pub fn recover_facets_with_steiner(tets: &mut Delaunay3D, cur_faces: &mut Vec<[usize; 3]>) -> bool {
    use std::collections::VecDeque;

    let mut queue: VecDeque<[usize; 3]> = cur_faces.iter().copied().collect();
    // The set of "final" (present) faces accumulates; we rebuild cur_faces from
    // the queue's terminal triangles as we go.
    let mut final_faces: Vec<[usize; 3]> = Vec::with_capacity(cur_faces.len());

    // Bound the Steiner facet splits relatively AND absolutely (same O(n²)
    // concern as segments: each split + recovery scan is O(tets)). A clean model
    // needs zero; a stubborn one bails fast to the gate.
    // Scale-relative, machine-independent bound (no wall clock): a stubborn
    // facet is split at most a number of times proportional to the boundary
    // size; success needs ~zero splits.
    let max_splits = (queue.len() / 4).clamp(64, 800);
    let mut splits = 0usize;
    let mut full_conform = true;

    while let Some(f) = queue.pop_front() {
        let (a, b, c) = (f[0], f[1], f[2]);
        if face_exists_in_tets(tets, a, b, c) {
            final_faces.push(f);
            continue;
        }
        // Flip-based recovery (diagonal swap / ring flip / piercing-edge flips).
        if (recover_face_by_diagonal_swap(tets, a, b, c, true)
            || recover_face_by_edge_flips(tets, a, b, c))
            && face_exists_in_tets(tets, a, b, c)
        {
            final_faces.push(f);
            continue;
        }
        // Constrained cavity re-triangulation (TetGen delaunizecavity) for a
        // facet flips cannot close - in particular the UNOBSTRUCTED missing
        // facet (all edges present, nothing piercing its interior; the local
        // fan straddles the plane but never closes the base - issue #37). All
        // OTHER surface facets in flight are passed as WALLS: the cavity never
        // swallows or grows across one, so they become required boundary faces
        // of the re-triangulation and are preserved. Uses existing vertices only
        // (no Steiner points → no slivers); on failure it is a no-op and we fall
        // through. Only runs for the rare flip-resistant facet, so the O(#faces)
        // wall-set build is paid per stuck facet, not per facet.
        {
            let mut walls: HashSet<[usize; 3]> = HashSet::default();
            let mut tgt = f;
            tgt.sort();
            for g in final_faces.iter().chain(queue.iter()) {
                let mut s = *g;
                s.sort();
                if s != tgt {
                    walls.insert(s);
                }
            }
            if recover_face_by_cavity(tets, a, b, c, &walls) && face_exists_in_tets(tets, a, b, c) {
                final_faces.push(f);
                continue;
            }
        }
        // Steiner: split the facet at its barycenter (strictly interior, in
        // plane). Requires all three edges present (guaranteed post-segment).
        // If an edge is missing or the budget is exhausted, keep this facet (as
        // its un-split parent) in the result and mark non-conforming - the carve
        // will leak there and the volume gate will arbitrate.
        if !edge_exists_in_tets(tets, a, b)
            || !edge_exists_in_tets(tets, b, c)
            || !edge_exists_in_tets(tets, c, a)
            || splits >= max_splits
        {
            if std::env::var("YAMM_CARVE_DBG").is_ok() {
                eprintln!(
                    "    [facet DBG] punt ({a},{b},{c}): edges=({},{},{}) splits={splits}/{max_splits}",
                    edge_exists_in_tets(tets, a, b),
                    edge_exists_in_tets(tets, b, c),
                    edge_exists_in_tets(tets, c, a),
                );
            }
            final_faces.push(f);
            full_conform = false;
            continue;
        }
        let bary = [
            (tets.vertices[a][0] + tets.vertices[b][0] + tets.vertices[c][0]) / 3.0,
            (tets.vertices[a][1] + tets.vertices[b][1] + tets.vertices[c][1]) / 3.0,
            (tets.vertices[a][2] + tets.vertices[b][2] + tets.vertices[c][2]) / 3.0,
        ];
        // Insert the barycenter as a Steiner point rather than splitting the
        // facet in place. `split_face_on_constraint` cannot be used here and
        // never could: it looks for a live tet whose vertex set contains a, b
        // and c, which is the very predicate `face_exists_in_tets` uses -- and
        // we only reach this line because that returned false. So it found
        // nothing every single time, and this whole refinement path, budget and
        // child re-queue included, had never once run. Paraboloid punted its one
        // stubborn facet here and leaked 1.6e-7 through the carve.
        //
        // `insert_steiner_local` is the primitive that fits a MISSING facet: it
        // locates the tet containing the point and performs the minimal
        // conformal split for wherever the point actually lies (in-tet, on a
        // face, or on an edge), so it does not require the facet to already be
        // there. The segment pass has always used it for the same reason.
        let Some(m) = tets.insert_steiner_local(bary) else {
            if std::env::var("YAMM_CARVE_DBG").is_ok() {
                eprintln!("    [facet DBG] punt ({a},{b},{c}): barycenter insert failed");
            }
            final_faces.push(f);
            full_conform = false;
            continue;
        };
        if m == a || m == b || m == c {
            // Resolved onto an existing corner: the children would be
            // degenerate and re-queueing them would not terminate.
            if std::env::var("YAMM_CARVE_DBG").is_ok() {
                eprintln!("    [facet DBG] punt ({a},{b},{c}): barycenter hit a corner");
            }
            final_faces.push(f);
            full_conform = false;
            continue;
        }
        splits += 1;

        // Re-establish local Delaunay-ness around the new vertex `m` by Lawson
        // flips so the three child facets become real Delaunay faces and the
        // refinement converges. Protect the boundary: the active constraint is
        // the facet plane (a,b,c) that `m` lies in, and the protected segment/
        // face sets cover every boundary triangle still in flight (the already-
        // finalized faces, the pending queue, and this facet's three children).
        {
            let mut protected_faces: Vec<[usize; 3]> =
                Vec::with_capacity(final_faces.len() + queue.len() + 3);
            protected_faces.extend_from_slice(&final_faces);
            protected_faces.extend(queue.iter().copied());
            protected_faces.push([a, b, m]);
            protected_faces.push([b, c, m]);
            protected_faces.push([c, a, m]);
            let (psegs, pfaces, _pinc) = protected_sets_from_faces(&protected_faces);
            // `None` active constraint (see the segment-pass rationale): the
            // restoration around `m` (in the facet plane (a,b,c)) needs flips
            // whose new elements may cross that plane; integrity is preserved by
            // the protected segment/face SET checks (no recovered element is
            // destroyed).
            let protect = ProtectedBoundary {
                active: Constraint::None,
                segments: &psegs,
                faces: &pfaces,
            };
            lawson_restore(tets, m, &protect);
        }

        // Re-queue the three child facets.
        queue.push_back([a, b, m]);
        queue.push_back([b, c, m]);
        queue.push_back([c, a, m]);
    }

    *cur_faces = final_faces;

    // Final verification: every refined facet present AND the loop didn't punt.
    full_conform
        && cur_faces
            .iter()
            .all(|f| face_exists_in_tets(tets, f[0], f[1], f[2]))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::super::delaunay3d::Delaunay3D;
    use super::super::predicates3d::orient_3d;
    use super::*;

    /// Helper: build a minimal tet mesh from raw tet vertex lists.
    /// Re-builds adjacency from scratch (O(n^2) but fine for tests).
    fn build_test_mesh(vertices: Vec<[f64; 3]>, tet_verts: Vec<[usize; 4]>) -> Delaunay3D {
        // Create a Delaunay3D from the vertices (which adds super-tet
        // infrastructure), then clear its tets and inject ours.
        let mut dt = Delaunay3D::new(&vertices);
        let n_existing = dt.tets.len();
        for i in 0..n_existing {
            dt.free_tet(i);
        }

        // Add tets with correct orientation, tracking their actual indices
        let mut actual_indices: Vec<usize> = Vec::new();
        for mut v in tet_verts {
            let o = orient_3d(
                vertices[v[0]],
                vertices[v[1]],
                vertices[v[2]],
                vertices[v[3]],
            );
            if o < 0.0 {
                v.swap(2, 3);
            }
            let idx = dt.alloc_tet(Tet {
                verts: v,
                adj: [usize::MAX; 4],
            });
            actual_indices.push(idx);
        }

        // Build adjacency using the actual indices
        let n = actual_indices.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let ti = actual_indices[i];
                let tj = actual_indices[j];
                if let Some((fi, fj)) = delaunay3d::shared_face_indices(&dt.tets[ti], &dt.tets[tj])
                {
                    dt.tets[ti].adj[fi] = tj;
                    dt.tets[tj].adj[fj] = ti;
                }
            }
        }

        dt
    }

    #[test]
    fn test_edge_exists_simple() {
        // Two tets sharing a face: (0,1,2,3) and (0,1,2,4)
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [1.0, 0.0, 0.0],  // 1
            [0.5, 1.0, 0.0],  // 2
            [0.5, 0.4, 1.0],  // 3
            [0.5, 0.4, -1.0], // 4
        ];
        let dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // Edge (0, 1) exists
        assert!(edge_exists_in_tets(&dt, 0, 1));
        // Edge (3, 4) does NOT exist (they are in different tets, not sharing an edge)
        assert!(!edge_exists_in_tets(&dt, 3, 4));
        // Edge (0, 3) exists
        assert!(edge_exists_in_tets(&dt, 0, 3));
    }

    #[test]
    fn test_flip_2_to_3_basic() {
        // Two tets sharing face (0, 1, 2):
        //   T0 = (0, 1, 2, 3), T1 = (0, 1, 2, 4)
        // where 3 is above the face and 4 is below.
        // After 2-to-3 flip, we should get 3 tets sharing edge (3, 4).
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above)
            [1.0, 0.7, -1.5], // 4 (below)
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // Before flip: edge (3, 4) should NOT exist
        assert!(!edge_exists_in_tets(&dt, 3, 4));

        // Find the tets containing the right vertices
        let mut tet_with_3 = usize::MAX;
        let mut tet_with_4 = usize::MAX;
        for (i, tet) in dt.tets.iter().enumerate() {
            if !dt.is_live(i) {
                continue;
            }
            if tet.verts.contains(&3) {
                tet_with_3 = i;
            }
            if tet.verts.contains(&4) {
                tet_with_4 = i;
            }
        }
        assert_ne!(tet_with_3, usize::MAX);
        assert_ne!(tet_with_4, usize::MAX);

        let result = flip_2_to_3(&mut dt, tet_with_3, tet_with_4, &Constraint::None);
        assert!(result.is_some(), "2-to-3 flip should succeed");

        let new_tets = result.unwrap();

        // After flip: edge (3, 4) should exist
        assert!(edge_exists_in_tets(&dt, 3, 4));

        // Verify the new tets are valid.
        for &ni in &new_tets {
            assert!(dt.is_live(ni));
            let v = dt.tets[ni].verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(o > 0.0, "New tet should have positive orientation, got {o}");
        }

        // All new tets should share edge (3, 4)
        for &ni in &new_tets {
            let v = dt.tets[ni].verts;
            assert!(
                v.contains(&3) && v.contains(&4),
                "New tet should contain both vertices 3 and 4"
            );
        }
    }

    #[test]
    fn test_flip_3_to_2_basic() {
        // Three tets sharing edge (3, 4), ring vertices (0, 1, 2):
        //   T0 = (0, 1, 3, 4)
        //   T1 = (1, 2, 3, 4)
        //   T2 = (2, 0, 3, 4)
        // After 3-to-2 flip, we should get 2 tets sharing face (0, 1, 2).
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above)
            [1.0, 0.7, -1.5], // 4 (below)
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]]);

        // Before flip: edge (3, 4) should exist
        assert!(edge_exists_in_tets(&dt, 3, 4));

        // Find the three tets sharing edge (3, 4)
        let sharing = tets_sharing_edge(&dt, 3, 4);
        assert_eq!(
            sharing.len(),
            3,
            "Should have exactly 3 tets sharing edge (3,4)"
        );

        let result = flip_3_to_2(
            &mut dt,
            [sharing[0], sharing[1], sharing[2]],
            3,
            4,
            &Constraint::None,
        );
        assert!(result.is_some(), "3-to-2 flip should succeed");

        let new_tets = result.unwrap();

        // After flip: we should have 2 tets, both containing face (0, 1, 2)
        for &ni in &new_tets {
            assert!(dt.is_live(ni));
            let v = dt.tets[ni].verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(o > 0.0, "New tet should have positive orientation, got {o}");
            // Each new tet should contain 3 of {0, 1, 2}
            let ring_count = [0usize, 1, 2].iter().filter(|&&r| v.contains(&r)).count();
            assert_eq!(ring_count, 3, "New tet should contain all 3 ring vertices");
        }

        // One new tet should have vertex 3, the other vertex 4
        let has_3 = new_tets.iter().any(|&ni| dt.tets[ni].verts.contains(&3));
        let has_4 = new_tets.iter().any(|&ni| dt.tets[ni].verts.contains(&4));
        assert!(has_3 && has_4, "New tets should cover both apex vertices");
    }

    #[test]
    fn test_flip_2_to_3_then_3_to_2_roundtrip() {
        // Start with 2 tets, do 2-to-3, then 3-to-2, verify we get back
        // a valid mesh with the same edge set.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.7, 1.5],
            [1.0, 0.7, -1.5],
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // 2-to-3 flip
        let tet_with_3 = dt
            .tets
            .iter()
            .enumerate()
            .find(|(i, t)| dt.is_live(*i) && t.verts.contains(&3))
            .unwrap()
            .0;
        let tet_with_4 = dt
            .tets
            .iter()
            .enumerate()
            .find(|(i, t)| dt.is_live(*i) && t.verts.contains(&4))
            .unwrap()
            .0;
        let new3 = flip_2_to_3(&mut dt, tet_with_3, tet_with_4, &Constraint::None).unwrap();

        assert!(edge_exists_in_tets(&dt, 3, 4));

        // 3-to-2 flip to reverse
        let _new2 = flip_3_to_2(&mut dt, new3, 3, 4, &Constraint::None).unwrap();

        // Edge (3, 4) should no longer exist; edge (0, 1), (0, 2), (1, 2) should exist
        assert!(!edge_exists_in_tets(&dt, 3, 4));
        assert!(edge_exists_in_tets(&dt, 0, 1));
        assert!(edge_exists_in_tets(&dt, 0, 2));
        assert!(edge_exists_in_tets(&dt, 1, 2));
        assert!(edge_exists_in_tets(&dt, 0, 3));
        assert!(edge_exists_in_tets(&dt, 0, 4));

        // All live tets should have positive volume
        for (i, t) in dt.tets.iter().enumerate() {
            if !dt.is_live(i) {
                continue;
            }
            let v = t.verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(o > 0.0, "Tet {i} has non-positive orientation: {o}");
        }
    }

    #[test]
    fn test_find_tets_crossing_edge() {
        // Create a configuration where edge (3, 4) is missing and crosses
        // through the face (0, 1, 2).
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above z=0 plane)
            [1.0, 0.7, -1.5], // 4 (below z=0 plane)
        ];
        let dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // Edge (3, 4) is missing. Find crossing tets.
        assert!(!edge_exists_in_tets(&dt, 3, 4));
        let crossing = find_tets_crossing_edge(&dt, 3, 4);
        // Tets containing vertex 3 or 4 are excluded from crossing.
        // The segment from 3 to 4 passes through face (0,1,2) which is
        // shared by both tets. Since both tets CONTAIN 3 or 4, neither
        // appears in the crossing list.
        assert!(
            crossing.is_empty(),
            "Both tets contain endpoints, so crossing list should be empty"
        );
    }

    #[test]
    fn test_segment_intersects_triangle() {
        let a = [0.5, 0.5, 1.0];
        let b = [0.5, 0.5, -1.0];
        let p = [0.0, 0.0, 0.0];
        let q = [1.0, 0.0, 0.0];
        let r = [0.5, 1.0, 0.0];

        assert!(segment_intersects_triangle(a, b, p, q, r));

        // Point outside triangle
        let a2 = [5.0, 5.0, 1.0];
        let b2 = [5.0, 5.0, -1.0];
        assert!(!segment_intersects_triangle(a2, b2, p, q, r));

        // Segment parallel to triangle (same side)
        let a3 = [0.5, 0.5, 1.0];
        let b3 = [0.5, 0.5, 0.5];
        assert!(!segment_intersects_triangle(a3, b3, p, q, r));
    }

    #[test]
    fn test_recover_missing_edge_simple() {
        // Build a Delaunay mesh of 5 points where edge (3, 4) is missing.
        // Then use recover_edges to recover it.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above)
            [1.0, 0.7, -1.5], // 4 (below)
        ];

        // Create a mesh where (3, 4) is not an edge.
        // Two tets sharing face (0, 1, 2) with 3 above and 4 below.
        let mut dt = build_test_mesh(vertices.clone(), vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        assert!(!edge_exists_in_tets(&dt, 3, 4));

        let (recovered, failed, _) = recover_edges(&mut dt, &[(3, 4)]);

        assert_eq!(recovered, 1, "Should recover 1 edge");
        assert_eq!(failed, 0, "Should have 0 failures");
        assert!(edge_exists_in_tets(&dt, 3, 4));

        // Verify all live tets have positive orientation
        for (i, t) in dt.tets.iter().enumerate() {
            if !dt.is_live(i) {
                continue;
            }
            let v = t.verts;
            if t.is_hull() {
                continue;
            }
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(
                o > 0.0,
                "Tet {i} has non-positive orientation after recovery: {o}"
            );
        }
    }

    #[test]
    fn test_recover_already_present_edge() {
        // Edge (0, 1) is already in the mesh, recovery should be a no-op.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.7, 1.5],
            [1.0, 0.7, -1.5],
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        assert!(edge_exists_in_tets(&dt, 0, 1));

        let (recovered, failed, _) = recover_edges(&mut dt, &[(0, 1)]);
        assert_eq!(
            recovered, 0,
            "Already-present edge should not count as recovered"
        );
        assert_eq!(failed, 0, "Already-present edge should not count as failed");
    }

    #[test]
    fn test_recover_edge_in_delaunay_mesh() {
        // Use the actual Delaunay3D to build a mesh, then check if boundary
        // recovery can fix missing edges.
        let points = vec![
            [0.0, 0.0, 0.0],  // 0
            [1.0, 0.0, 0.0],  // 1
            [0.5, 1.0, 0.0],  // 2
            [0.5, 0.3, 0.8],  // 3
            [0.5, 0.3, -0.8], // 4
        ];
        let mut dt = Delaunay3D::new(&points);

        // Collect boundary edges from a hypothetical surface
        let boundary_edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4), // this one might be missing in Delaunay
        ];

        let (_recovered, failed, _) = recover_edges(&mut dt, &boundary_edges);

        // After recovery, all edges should exist
        for &(a, b) in &boundary_edges {
            {
                // The edge should exist (recovered or was already present)
                let exists = edge_exists_in_tets(&dt, a, b);
                if !exists {
                    // It's ok if it failed -- we just want to verify the
                    // function runs without panicking and returns sensible counts.
                    assert!(
                        failed > 0,
                        "Edge ({a}, {b}) not present but no failures reported"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2 tests: face recovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_exists_in_tets_present() {
        // Two tets sharing face (0,1,2): T0=(0,1,2,3), T1=(0,1,2,4)
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // Face (0,1,2) is the shared face -- it must exist.
        assert!(face_exists_in_tets(&dt, 0, 1, 2));
        // Face (0,1,3) is a face of T0.
        assert!(face_exists_in_tets(&dt, 0, 1, 3));
        // Face (0,2,3) is a face of T0.
        assert!(face_exists_in_tets(&dt, 0, 2, 3));
        // Face (1,2,3) is a face of T0.
        assert!(face_exists_in_tets(&dt, 1, 2, 3));
        // Face (0,1,4) is a face of T1.
        assert!(face_exists_in_tets(&dt, 0, 1, 4));
    }

    #[test]
    fn test_face_exists_in_tets_missing() {
        // Two tets sharing face (0,1,2): T0=(0,1,2,3), T1=(0,1,2,4)
        // Face (3,4,0) should NOT exist since 3 and 4 are never in the same tet.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        assert!(!face_exists_in_tets(&dt, 0, 3, 4));
        assert!(!face_exists_in_tets(&dt, 1, 3, 4));
        assert!(!face_exists_in_tets(&dt, 2, 3, 4));
    }

    #[test]
    fn test_face_exists_order_independent() {
        // Face existence should not depend on the order of the three vertices.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.7, 1.5],
            [1.0, 0.7, -1.5],
        ];
        let dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // All permutations of (0,1,2) should give the same result.
        assert!(face_exists_in_tets(&dt, 0, 1, 2));
        assert!(face_exists_in_tets(&dt, 0, 2, 1));
        assert!(face_exists_in_tets(&dt, 1, 0, 2));
        assert!(face_exists_in_tets(&dt, 1, 2, 0));
        assert!(face_exists_in_tets(&dt, 2, 0, 1));
        assert!(face_exists_in_tets(&dt, 2, 1, 0));
    }

    #[test]
    fn test_recover_face_already_present() {
        // If the face already exists, recover_faces should be a no-op.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.7, 1.5],
            [1.0, 0.7, -1.5],
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        assert!(face_exists_in_tets(&dt, 0, 1, 2));

        let (recovered, failed, _) = recover_faces(&mut dt, &[[0, 1, 2]]);
        assert_eq!(
            recovered, 0,
            "Already-present face should not count as recovered"
        );
        assert_eq!(failed, 0, "Already-present face should not count as failed");
    }

    #[test]
    fn test_recover_missing_face_simple() {
        // Build a mesh where face (0, 3, 4) is missing.
        //
        // Start with 3 tets sharing edge (3, 4):
        //   T0 = (0, 1, 3, 4)
        //   T1 = (1, 2, 3, 4)
        //   T2 = (2, 0, 3, 4)
        //
        // Face (0, 1, 2) does NOT exist. All edges of triangle (0,1,2) exist:
        //   (0,1) is an edge of T0
        //   (1,2) is an edge of T1
        //   (0,2) is an edge of T2
        // But face (0,1,2) is blocked by edge (3,4).
        //
        // A 3-to-2 flip around edge (3,4) should recover face (0,1,2).
        // However, face recovery uses 2-to-3 flips. Let's see if it works.
        //
        // The face (0,1,2) becomes a face when the 3-tet ring around (3,4)
        // is replaced by 2 tets sharing face (0,1,2). The recover_face_by_flips
        // algorithm works via the ring around edge (0,1): in the 3-tet config,
        // the ring around (0,1) contains T0 = (0,1,3,4). Vertex 2 is not in
        // that ring. A 2-to-3 flip on a blocking face can introduce vertex 2.

        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]]);

        // All edges of face (0,1,2) exist.
        assert!(edge_exists_in_tets(&dt, 0, 1));
        assert!(edge_exists_in_tets(&dt, 1, 2));
        assert!(edge_exists_in_tets(&dt, 0, 2));

        // But face (0,1,2) does NOT exist.
        assert!(
            !face_exists_in_tets(&dt, 0, 1, 2),
            "Face (0,1,2) should be missing initially"
        );

        let (recovered, failed, _) = recover_faces(&mut dt, &[[0, 1, 2]]);

        assert_eq!(failed, 0, "Should have 0 face recovery failures");
        assert_eq!(recovered, 1, "Should recover 1 face");
        assert!(
            face_exists_in_tets(&dt, 0, 1, 2),
            "Face (0,1,2) should exist after recovery"
        );

        // All live tets should have positive orientation
        for (i, t) in dt.tets.iter().enumerate() {
            if !dt.is_live(i) {
                continue;
            }
            let v = t.verts;
            if t.is_hull() {
                continue;
            }
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(
                o > 0.0,
                "Tet {i} has non-positive orientation after face recovery: {o}"
            );
        }
    }

    #[test]
    fn test_recover_faces_in_delaunay_mesh() {
        // Use Delaunay3D to build a mesh, recover edges, then recover faces.
        let points = vec![
            [0.0, 0.0, 0.0],  // 0
            [1.0, 0.0, 0.0],  // 1
            [0.5, 1.0, 0.0],  // 2
            [0.5, 0.3, 0.8],  // 3
            [0.5, 0.3, -0.8], // 4
        ];
        let mut dt = Delaunay3D::new(&points);

        let boundary_edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ];

        let boundary_faces: Vec<[usize; 3]> = vec![
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 4],
            [0, 2, 4],
            [1, 2, 4],
            [0, 3, 4],
            [1, 3, 4],
            [2, 3, 4],
        ];

        // Phase 1: recover edges
        let (_edge_ok, _edge_fail, _) = recover_edges(&mut dt, &boundary_edges);

        // Phase 2: recover faces
        let (_face_ok, face_fail, _) = recover_faces(&mut dt, &boundary_faces);

        // Verify as many faces as possible exist
        for face in &boundary_faces {
            let (a, b, c) = (face[0], face[1], face[2]);
            {
                let exists = face_exists_in_tets(&dt, a, b, c);
                if !exists {
                    assert!(
                        face_fail > 0,
                        "Face ({a}, {b}, {c}) not present but no failures reported"
                    );
                }
            }
        }

        // Verify no panics and sensible counts
        assert!(
            _face_ok + face_fail <= boundary_faces.len(),
            "Total should not exceed input count"
        );
    }

    #[test]
    fn test_recover_faces_counts() {
        // Mix of present and missing faces -- verify counting is correct.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        // Face (0,1,2) is present. Face (0,3,4) is missing (3 and 4 not in same tet).
        let boundary_faces: Vec<[usize; 3]> = vec![
            [0, 1, 2], // present
            [0, 1, 3], // present (face of T0)
        ];

        let (recovered, failed, _) = recover_faces(&mut dt, &boundary_faces);
        assert_eq!(recovered, 0, "Both faces are already present");
        assert_eq!(failed, 0, "No failures expected");
    }

    // -----------------------------------------------------------------------
    // P0/P1 tests: transactional flips + constraint guard + 2-2 diagonal swap
    // -----------------------------------------------------------------------

    /// Count live finite (non-hull) tets.
    fn live_finite_count(dt: &Delaunay3D) -> usize {
        (0..dt.tets.len())
            .filter(|&i| dt.is_live(i) && !dt.tets[i].is_hull())
            .count()
    }

    /// Snapshot of the live finite tets as a sorted vertex-multiset, for
    /// byte-for-byte (modulo ordering) no-op comparison.
    fn live_finite_snapshot(dt: &Delaunay3D) -> Vec<[usize; 4]> {
        let mut v: Vec<[usize; 4]> = (0..dt.tets.len())
            .filter(|&i| dt.is_live(i) && !dt.tets[i].is_hull())
            .map(|i| {
                let mut k = dt.tets[i].verts;
                k.sort();
                k
            })
            .collect();
        v.sort();
        v
    }

    /// Validate the live finite tet mesh: no negative-volume tets and
    /// symmetric adjacency among live tets.
    fn mesh_is_consistent(dt: &Delaunay3D) -> bool {
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            let v = dt.tets[i].verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            if o <= 0.0 {
                return false;
            }
            for fi in 0..4 {
                let nb = dt.tets[i].adj[fi];
                if nb == usize::MAX || !dt.is_live(nb) {
                    continue;
                }
                if !dt.tets[nb].adj.contains(&i) {
                    return false;
                }
            }
        }
        true
    }

    /// Build the LShaped-style coplanar sliver ring: a flat-face quad whose
    /// WRONG diagonal (1,3) is shared by a ring of FIVE tets (> 4, so the n==4
    /// `flip_ring4`/`flip_nm` cannot touch it). The quad corners 0,1,2,3 lie in
    /// z=0; the wanted diagonal (0,2) and the wrong diagonal (1,3) are the two
    /// in-plane diagonals (they cross at the origin, orient_3d(0,2,1,3)==0). The
    /// ring of (1,3) walks the apices [0, 4, 2, 6, 5] - one apex above (4) and
    /// two below (5,6) the plane plus the two in-plane corners 0 and 2.
    fn coplanar_sliver_ring() -> Delaunay3D {
        let vertices = vec![
            [-1.0, 0.0, 0.0],  // 0  (quad corner, -x, in plane)  wanted-diag end
            [0.0, -1.0, 0.0],  // 1  (quad corner, -y, in plane)  wrong-diag end
            [1.0, 0.0, 0.0],   // 2  (quad corner, +x, in plane)  wanted-diag end
            [0.0, 1.0, 0.0],   // 3  (quad corner, +y, in plane)  wrong-diag end
            [0.0, 0.0, 1.0],   // 4  apex above
            [-0.5, 0.0, -1.0], // 5  apex below (-x side)
            [0.5, 0.0, -1.0],  // 6  apex below (+x side)
        ];
        // Five tets around edge (1,3), cyclic apices [0,4,2,6,5].
        let tets = vec![
            [1, 3, 0, 4],
            [1, 3, 4, 2],
            [1, 3, 2, 6],
            [1, 3, 6, 5],
            [1, 3, 5, 0],
        ];
        build_test_mesh(vertices, tets)
    }

    /// A PINCHED ring around edge (1, 3): apex 4 sits at two positions of the
    /// cycle, so the link "polygon" is a figure-eight through vertex 4 rather
    /// than a simple cycle.
    ///
    /// Six tets with apex cycle `[0, 4, 2, 5, 4, 6]`. Every tet has a distinct
    /// vertex set and non-zero volume, so neither a duplicate-tet check nor a
    /// volume check can see the defect: it lives purely in the cyclic
    /// structure. This is the shape the coplanar-region rebuild produces when
    /// its 2-D flip log replays an inverse flip pair as coincident tets.
    fn pinched_apex_ring() -> Delaunay3D {
        let vertices = vec![
            [1.0, 0.0, 0.0],   // 0  apex
            [0.0, 0.0, -1.0],  // 1  ring edge end
            [-0.5, 0.9, 0.0],  // 2  apex
            [0.0, 0.0, 1.0],   // 3  ring edge end
            [0.5, 0.9, 0.0],   // 4  apex, REPEATED in the cycle
            [-1.0, 0.0, 0.0],  // 5  apex
            [-0.6, -0.8, 0.0], // 6  apex
        ];
        let tets = vec![
            [1, 3, 0, 4],
            [1, 3, 4, 2],
            [1, 3, 2, 5],
            [1, 3, 5, 4],
            [1, 3, 4, 6],
            [1, 3, 6, 0],
        ];
        build_test_mesh(vertices, tets)
    }

    /// Index of the live tet carrying exactly `verts`.
    fn find_tet(dt: &Delaunay3D, verts: [usize; 4]) -> usize {
        let mut want = verts;
        want.sort_unstable();
        (0..dt.tets.len())
            .find(|&i| {
                if !dt.is_live(i) {
                    return false;
                }
                let mut got = dt.tets[i].verts;
                got.sort_unstable();
                got == want
            })
            .unwrap_or_else(|| panic!("no live tet with vertex set {want:?}"))
    }

    #[test]
    fn test_ring_apex_cycle_rejects_a_pinched_link_polygon() {
        // The reducer must decline a non-simple cycle, which is what
        // `ring_apex_cycle`'s contract already promised. Handing it on would
        // make `flip_ring_general`'s polygon DP form a "triangle" with a
        // repeated vertex, and `orient_3d_sos` panics on that degenerate
        // simplex rather than invent a sign for it.
        let dt = pinched_apex_ring();
        let ring: Vec<usize> = [
            [1, 3, 0, 4],
            [1, 3, 4, 2],
            [1, 3, 2, 5],
            [1, 3, 5, 4],
            [1, 3, 4, 6],
            [1, 3, 6, 0],
        ]
        .iter()
        .map(|&v| find_tet(&dt, v))
        .collect();

        // The fixture is corrupt ONLY in its cycle: six distinct tets, none
        // coincident. A duplicate-tet check would pass on it.
        let mut snapshot = live_finite_snapshot(&dt);
        assert_eq!(snapshot.len(), 6, "fixture must hold exactly six tets");
        snapshot.dedup();
        assert_eq!(snapshot.len(), 6, "fixture must hold no coincident tets");

        assert!(
            ring_apex_cycle(&dt, 1, 3, &ring).is_none(),
            "a pinched link polygon (apex 4 at two positions) must be rejected"
        );
    }

    #[test]
    fn test_ring_apex_cycle_accepts_a_simple_link_polygon() {
        // The guard must not cost a healthy ring its flip: the 5-cycle of
        // `coplanar_sliver_ring` has apices [0, 4, 2, 6, 5], all distinct.
        let dt = coplanar_sliver_ring();
        let ring = edge_ring_cycle(&dt, 1, 3).expect("closed ring");
        let w = ring_apex_cycle(&dt, 1, 3, &ring).expect("simple ring must be accepted");
        let mut distinct = w.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            w.len(),
            "fixture ring must have distinct apices"
        );
    }

    #[test]
    fn test_coplanar_ring_flip_recovers_wanted_diagonal() {
        // The flat-face quad's wrong diagonal (1,3) has a 5-tet ring; the wanted
        // diagonal (0,2) is absent. The coplanar ring reducer must flip (1,3)
        // away so the wanted diagonal (0,2) appears - and do so deterministically
        // (identical result every run, since the only nondeterminism would come
        // from iteration order, which the DP/sort eliminate).
        let empty_segs = HashSet::default();
        let empty_faces = HashSet::default();

        // Sanity: the two diagonals are coplanar and cross in the interior.
        {
            let dt = coplanar_sliver_ring();
            assert_eq!(
                orient_3d(
                    dt.vertices[0],
                    dt.vertices[2],
                    dt.vertices[1],
                    dt.vertices[3]
                ),
                0.0,
                "wanted (0,2) and wrong (1,3) diagonals must be coplanar"
            );
            assert!(segments_cross_interior(&dt.vertices, [0, 2], [1, 3]));
            assert!(
                edge_exists_in_tets(&dt, 1, 3),
                "wrong diagonal present initially"
            );
            assert!(
                !edge_exists_in_tets(&dt, 0, 2),
                "wanted diagonal absent initially"
            );
            // The ring is genuinely larger than the n==4 reducer can handle.
            let ring = edge_ring_cycle(&dt, 1, 3).expect("closed ring");
            assert_eq!(ring.len(), 5, "wrong-diagonal ring must be a 5-cycle");
            assert!(
                flip_nm(&mut coplanar_sliver_ring(), 1, 3, &Constraint::None).is_none()
                    || ring.len() == 5
            );
        }

        // Run the reducer twice; both must recover (0,2) and yield IDENTICAL
        // tetrahedra (deterministic).
        let run = || {
            let mut dt = coplanar_sliver_ring();
            let protect = ProtectedBoundary {
                active: Constraint::Edge(0, 2),
                segments: &empty_segs,
                faces: &empty_faces,
            };
            let ok = flip_ring_general(&mut dt, 1, 3, &protect);
            (ok, dt)
        };
        let (ok1, dt1) = run();
        let (ok2, dt2) = run();
        assert!(
            ok1,
            "coplanar ring reducer must clear the wrong diagonal (1,3)"
        );
        assert!(ok2);
        assert!(
            edge_exists_in_tets(&dt1, 0, 2),
            "wanted diagonal (0,2) recovered"
        );
        assert!(
            !edge_exists_in_tets(&dt1, 1, 3),
            "wrong diagonal (1,3) removed"
        );
        assert!(
            mesh_is_consistent(&dt1),
            "reduced mesh must be orientation+adjacency consistent"
        );

        // Determinism: identical live finite tet sets (as sorted vertex tuples).
        assert_eq!(
            live_finite_snapshot(&dt1),
            live_finite_snapshot(&dt2),
            "coplanar reducer must be deterministic (identical mesh both runs)"
        );
    }

    #[test]
    fn test_coplanar_ring_flip_protects_recovered_boundary() {
        // If the wrong diagonal (1,3) is itself a PROTECTED (already-recovered)
        // boundary segment, the reducer must refuse to remove it (no-op), leaving
        // the mesh unchanged.
        let mut segs = HashSet::default();
        segs.insert((1usize, 3usize));
        let empty_faces = HashSet::default();

        let mut dt = coplanar_sliver_ring();
        let before = live_finite_snapshot(&dt);
        let protect = ProtectedBoundary {
            active: Constraint::Edge(0, 2),
            segments: &segs,
            faces: &empty_faces,
        };
        let ok = flip_ring_general(&mut dt, 1, 3, &protect);
        assert!(!ok, "must not remove a protected segment");
        assert_eq!(
            before,
            live_finite_snapshot(&dt),
            "mesh must be unchanged (no-op)"
        );
        assert!(
            edge_exists_in_tets(&dt, 1, 3),
            "protected (1,3) still present"
        );
    }

    #[test]
    fn test_flip_2_to_3_transactional_noop_on_degenerate() {
        // Two tets sharing face (0,1,2) where apices 3 and 4 are positioned so
        // the new edge (3,4) would be coplanar with a face edge → at least one
        // candidate new tet is degenerate (orient == 0). The flip must be a
        // true no-op: same live count, identical snapshot, still consistent.
        //
        // Put 3 and 4 in the SAME z=0 plane as edge (0,1) so that the
        // candidate tet (0,1,3,4) is flat.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.5],  // 2 (off the z=0 plane)
            [0.5, -1.0, 0.0], // 3 (in z=0 plane with 0,1)
            [1.5, -1.0, 0.0], // 4 (in z=0 plane with 0,1)
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        let before_count = live_finite_count(&dt);
        let before_snap = live_finite_snapshot(&dt);

        let t3 = (0..dt.tets.len())
            .find(|&i| dt.is_live(i) && dt.tets[i].verts.contains(&3))
            .unwrap();
        let t4 = (0..dt.tets.len())
            .find(|&i| dt.is_live(i) && dt.tets[i].verts.contains(&4))
            .unwrap();

        let result = flip_2_to_3(&mut dt, t3, t4, &Constraint::None);
        assert!(result.is_none(), "degenerate 2-3 flip must be rejected");
        assert_eq!(
            live_finite_count(&dt),
            before_count,
            "rejected flip must not change live tet count"
        );
        assert_eq!(
            live_finite_snapshot(&dt),
            before_snap,
            "rejected flip must leave the mesh unchanged"
        );
        assert!(mesh_is_consistent(&dt), "mesh must remain consistent");
    }

    #[test]
    fn test_flip_2_to_3_transactional_noop_on_constraint() {
        // A geometrically valid 2-3 flip that the constraint forbids: the new
        // edge (3,4) crosses the constraint face (0,1,2) interior. Must no-op.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above)
            [1.0, 0.7, -1.5], // 4 (below)
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);

        let before_count = live_finite_count(&dt);
        let before_snap = live_finite_snapshot(&dt);

        let t3 = (0..dt.tets.len())
            .find(|&i| dt.is_live(i) && dt.tets[i].verts.contains(&3))
            .unwrap();
        let t4 = (0..dt.tets.len())
            .find(|&i| dt.is_live(i) && dt.tets[i].verts.contains(&4))
            .unwrap();

        // The new edge (3,4) pierces triangle (0,1,2) → forbidden.
        let result = flip_2_to_3(&mut dt, t3, t4, &Constraint::Face([0, 1, 2]));
        assert!(
            result.is_none(),
            "constraint-violating 2-3 flip must be rejected"
        );
        assert_eq!(live_finite_count(&dt), before_count);
        assert_eq!(live_finite_snapshot(&dt), before_snap);
        assert!(mesh_is_consistent(&dt));

        // With no constraint, the same flip must succeed (sanity).
        assert!(flip_2_to_3(&mut dt, t3, t4, &Constraint::None).is_some());
        assert!(edge_exists_in_tets(&dt, 3, 4));
        assert!(mesh_is_consistent(&dt));
    }

    /// The 8 corner vertices of the unit cube.
    fn cube_vertices() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ]
    }

    /// The 12 triangles (2 per face) of the closed cube surface, oriented
    /// consistently (outward); orientation is irrelevant to face recovery,
    /// which is order-independent.
    fn cube_surface_tris() -> Vec<[usize; 3]> {
        vec![
            // bottom z=0 (quad 0,1,2,3) split on diagonal 0-2
            [0, 1, 2],
            [0, 2, 3],
            // top z=1 (quad 4,5,6,7) split on diagonal 4-6
            [4, 5, 6],
            [4, 6, 7],
            // front y=0 (0,1,5,4) split on diagonal 0-5
            [0, 1, 5],
            [0, 5, 4],
            // back y=1 (3,2,6,7) split on diagonal 3-6
            [3, 2, 6],
            [3, 6, 7],
            // left x=0 (0,3,7,4) split on diagonal 0-7
            [0, 3, 7],
            [0, 7, 4],
            // right x=1 (1,2,6,5) split on diagonal 1-6
            [1, 2, 6],
            [1, 6, 5],
        ]
    }

    #[test]
    fn test_cube_quad_diagonal_swap_milestone() {
        // Build the Delaunay of the 8 cube corners. The Delaunay frequently
        // picks the OPPOSITE diagonal on a quad face, so some wanted boundary
        // triangle is absent. Diagonal-swap recovery must restore it, and full
        // recover_faces_no_bail over all 12 cube tris must finish with failed==0.
        let pts = cube_vertices();
        let mut dt = Delaunay3D::new(&pts);

        let tris = cube_surface_tris();

        // Find at least one wanted triangle that is currently missing.
        let missing: Vec<[usize; 3]> = tris
            .iter()
            .copied()
            .filter(|t| !face_exists_in_tets(&dt, t[0], t[1], t[2]))
            .collect();
        assert!(
            !missing.is_empty(),
            "expected the Delaunay of cube corners to be missing at least one \
             boundary triangle (wrong diagonal); got none"
        );

        // Recover the first missing quad triangle by diagonal swap alone, and
        // assert BOTH triangles of that quad become present afterwards.
        let m = missing[0];
        let mate = quad_mate(&tris, m);
        assert!(
            recover_face_by_diagonal_swap(&mut dt, m[0], m[1], m[2], true),
            "diagonal-swap recovery should restore missing triangle {m:?}"
        );
        assert!(
            face_exists_in_tets(&dt, m[0], m[1], m[2]),
            "triangle {m:?} must exist after diagonal swap"
        );
        if let Some(mate) = mate {
            assert!(
                face_exists_in_tets(&dt, mate[0], mate[1], mate[2]),
                "the other triangle {mate:?} of the swapped quad must also be present"
            );
        }
        assert!(
            mesh_is_consistent(&dt),
            "mesh must stay consistent after the swap"
        );

        // Now run full no-bail recovery over all 12 cube triangles.
        let (_rec, failed, failed_faces) = recover_faces_no_bail(&mut dt, &tris);
        assert_eq!(
            failed, 0,
            "all 12 cube boundary triangles must be recoverable; failed={failed} {failed_faces:?}"
        );
        for t in &tris {
            assert!(
                face_exists_in_tets(&dt, t[0], t[1], t[2]),
                "cube triangle {t:?} must be present after recovery"
            );
        }
        assert!(mesh_is_consistent(&dt));
    }

    /// Given the wanted triangle `m` of a cube face, return the OTHER triangle
    /// of the same coplanar quad (the one sharing exactly the quad diagonal),
    /// if present in `tris`.
    fn quad_mate(tris: &[[usize; 3]], m: [usize; 3]) -> Option<[usize; 3]> {
        // Cube faces are listed in consecutive pairs in cube_surface_tris().
        for pair in tris.chunks(2) {
            if pair.len() == 2 {
                let mut a = pair[0];
                let mut b = pair[1];
                a.sort();
                b.sort();
                let mut ms = m;
                ms.sort();
                if a == ms {
                    return Some(pair[1]);
                }
                if b == ms {
                    return Some(pair[0]);
                }
            }
        }
        None
    }

    #[test]
    fn test_cube_recovery_is_deterministic() {
        // Determinism of the diagonal-swap recovery.
        //
        // (1) Same input ordering, run twice → identical recovered boundary
        //     face set (the swap's diagonal tie-break depends only on vertex
        //     ids, not on arena layout, so repeated runs agree exactly).
        //
        // (2) NOTE on reordering: the 8 cube corners are cospherical, so a
        //     different insertion order yields a *different but equally valid*
        //     Delaunay triangulation. In some orderings a wanted boundary quad's
        //     wrong diagonal is shared by a 3-tet RING (an interior edge with
        //     three incident tets), not by exactly two tets. A 2-2 flip cannot
        //     swap such a diagonal - that requires the edge-star reducer
        //     (flip_nm) scheduled for the next phase (P2/P3). We document this
        //     here rather than asserting full cross-ordering closure, which is
        //     out of scope for P0/P1. The canonical-ordering cube DOES reach
        //     failed==0 (see test_cube_quad_diagonal_swap_milestone).

        let canonical = cube_vertices();
        let tris = cube_surface_tris();

        let run = || {
            let mut dt = Delaunay3D::new(&canonical);
            let (rec, failed, _) = recover_faces_no_bail(&mut dt, &tris);
            let faces: HashSet<[usize; 3]> = boundary_face_set(&dt, |v| v);
            (rec, failed, faces)
        };

        let (rec_a, failed_a, faces_a) = run();
        let (rec_b, failed_b, faces_b) = run();

        assert_eq!(rec_a, rec_b, "recovered count must be deterministic");
        assert_eq!(failed_a, failed_b, "failed count must be deterministic");
        assert_eq!(
            faces_a, faces_b,
            "recovered face set must be identical across identical runs"
        );

        // For the canonical ordering the diagonal swap closes every quad.
        assert_eq!(failed_a, 0, "canonical cube must fully recover");
        let mut wanted: HashSet<[usize; 3]> = HashSet::default();
        for t in &tris {
            let mut s = *t;
            s.sort();
            wanted.insert(s);
        }
        for w in &wanted {
            assert!(faces_a.contains(w), "wanted face {w:?} must be present");
        }

        // Document the cross-ordering limitation as an observed (non-fatal)
        // fact: under this permutation some quads need the next-phase reducer.
        let perm = [3usize, 1, 7, 0, 5, 2, 6, 4];
        let permuted_pts: Vec<[f64; 3]> = perm.iter().map(|&old| canonical[old]).collect();
        let mut inv = [0usize; 8];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let tris_perm: Vec<[usize; 3]> = tris
            .iter()
            .map(|t| [inv[t[0]], inv[t[1]], inv[t[2]]])
            .collect();
        let mut dt_p = Delaunay3D::new(&permuted_pts);
        let (_pr, pf, _) = recover_faces_no_bail(&mut dt_p, &tris_perm);
        if pf != 0 {
            eprintln!(
                "cad-to-dagmc-mesher: NOTE: permuted cube leaves {pf} face(s) for the \
                 next-phase edge-star reducer (3-tet-ring diagonals are not 2-2-swappable). \
                 This is expected at P0/P1."
            );
        }
    }

    /// Collect the set of all live-tet faces, mapping each vertex through
    /// `map` (used to translate a permuted mesh back to canonical labels),
    /// then sorting each face for set membership.
    fn boundary_face_set(dt: &Delaunay3D, map: impl Fn(usize) -> usize) -> HashSet<[usize; 3]> {
        let mut set = HashSet::default();
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            for fi in 0..4 {
                let face = delaunay3d::opposite_face(dt.tets[i].verts, fi);
                let mut f = [map(face[0]), map(face[1]), map(face[2])];
                f.sort();
                set.insert(f);
            }
        }
        set
    }

    // -----------------------------------------------------------------------
    // ADHOC: ground-truth adjacency audit for flip relinking.
    // -----------------------------------------------------------------------

    /// Return a list of human-readable adjacency defects in the live finite mesh.
    ///
    /// Ground truth is recomputed from scratch (faces_match over all live finite
    /// tet pairs). Reports:
    ///  - dangling: adj points at a freed/out-of-range index
    ///  - asym: A->B recorded but B->A missing
    ///  - wrong: A->B recorded but A and B do NOT share a face
    ///  - missing: A and B truly share a face but the adj link is unset on one/both
    ///  - bad_face: adj[fi] set but the face it claims to be across isn't the
    ///    actual shared face with that neighbor
    fn adjacency_defects(dt: &Delaunay3D) -> Vec<String> {
        let mut out = Vec::new();
        let live: Vec<usize> = (0..dt.tets.len())
            .filter(|&i| dt.is_live(i) && !dt.tets[i].is_hull())
            .collect();

        // 1) Every recorded link must be valid, symmetric, on the right face.
        for &i in &live {
            for fi in 0..4 {
                let nb = dt.tets[i].adj[fi];
                if nb == usize::MAX {
                    continue;
                }
                if nb >= dt.tets.len() || !dt.is_live(nb) {
                    out.push(format!("dangling: tet {i} face {fi} -> dead/oob {nb}"));
                    continue;
                }
                if dt.tets[nb].is_hull() {
                    continue; // hull neighbors are allowed and not audited here
                }
                // The face this link claims must actually be shared with nb.
                let claimed = delaunay3d::opposite_face(dt.tets[i].verts, fi);
                let shares = (0..4).any(|fj| {
                    delaunay3d::faces_match(
                        &claimed,
                        &delaunay3d::opposite_face(dt.tets[nb].verts, fj),
                    )
                });
                if !shares {
                    out.push(format!(
                        "wrong: tet {i} face {fi} {claimed:?} -> {nb} (no shared face)"
                    ));
                }
                if !dt.tets[nb].adj.contains(&i) {
                    out.push(format!("asym: {i} -> {nb} but {nb} !-> {i}"));
                }
            }
        }

        // 2) Every truly-shared face between two live finite tets must be linked
        //    both ways.
        for ai in 0..live.len() {
            for bi in (ai + 1)..live.len() {
                let (a, b) = (live[ai], live[bi]);
                if let Some((fa, fb)) = delaunay3d::shared_face_indices(&dt.tets[a], &dt.tets[b]) {
                    if dt.tets[a].adj[fa] != b {
                        out.push(format!(
                            "missing: {a} shares face with {b} but adj[{fa}]={}",
                            dt.tets[a].adj[fa]
                        ));
                    }
                    if dt.tets[b].adj[fb] != a {
                        out.push(format!(
                            "missing: {b} shares face with {a} but adj[{fb}]={}",
                            dt.tets[b].adj[fb]
                        ));
                    }
                }
            }
        }
        out
    }

    /// A box made of 6 tets around the main diagonal (0..7 are cube corners,
    /// 8 is the center). Surrounds the two pivot tets with real neighbors so
    /// the external relinking path is fully exercised.
    fn surrounded_pair() -> (Delaunay3D, usize, usize) {
        // T1=(0,1,2,3) T2=(0,1,2,4) share face (0,1,2).
        // Add outer neighbors across each non-shared face of T1 and T2 so that
        // EVERY external face has a real neighbor whose back-pointer must be
        // fixed by the flip.
        let vertices = vec![
            [0.0, 0.0, 0.0],    // 0
            [2.0, 0.0, 0.0],    // 1
            [1.0, 2.0, 0.0],    // 2
            [1.0, 0.7, 2.0],    // 3 (above)
            [1.0, 0.7, -2.0],   // 4 (below)
            [1.0, 0.7, 5.0],    // 5 far above  -> neighbor of T1 across (0,1,3)? built generically
            [-3.0, -1.0, 1.0],  // 6
            [5.0, -1.0, 1.0],   // 7
            [1.0, 6.0, 1.0],    // 8
            [-3.0, -1.0, -1.0], // 9
            [5.0, -1.0, -1.0],  //10
            [1.0, 6.0, -1.0],   //11
        ];
        // Neighbors of T1=(0,1,2,3): across (1,2,3)->8, (0,2,3)->6, (0,1,3)->? use 7
        // Neighbors of T2=(0,1,2,4): across (1,2,4)->11,(0,2,4)->9,(0,1,4)->10
        let tets = vec![
            [0, 1, 2, 3], // T1
            [0, 1, 2, 4], // T2
            [1, 2, 3, 8],
            [0, 2, 3, 6],
            [0, 1, 3, 7],
            [1, 2, 4, 11],
            [0, 2, 4, 9],
            [0, 1, 4, 10],
        ];
        let dt = build_test_mesh(vertices, tets);
        let t1 = (0..dt.tets.len())
            .find(|&i| {
                let mut v = dt.tets[i].verts;
                v.sort();
                dt.is_live(i) && v == [0, 1, 2, 3]
            })
            .unwrap();
        let t2 = (0..dt.tets.len())
            .find(|&i| {
                let mut v = dt.tets[i].verts;
                v.sort();
                dt.is_live(i) && v == [0, 1, 2, 4]
            })
            .unwrap();
        (dt, t1, t2)
    }

    #[test]
    fn audit_flip_2_to_3_external_relink() {
        let (mut dt, t1, t2) = surrounded_pair();
        let pre = adjacency_defects(&dt);
        assert!(pre.is_empty(), "precondition mesh already broken: {pre:?}");
        let res = flip_2_to_3(&mut dt, t1, t2, &Constraint::None);
        assert!(res.is_some(), "2-3 flip should succeed here");
        let defects = adjacency_defects(&dt);
        assert!(
            defects.is_empty(),
            "2-3 flip left adjacency defects: {defects:#?}"
        );
    }

    #[test]
    fn adv_bad_ring_q_absent_from_one_tet() {
        // Adversarial: a "bad ring" where the union of non-{p,q} verts is
        // exactly 3 BUT one tet does not actually contain q.
        //   p=3, q=4.
        //   T1 = {p,q,a,b} = {3,4,0,1}
        //   T2 = {p,q,b,c} = {3,4,1,2}
        //   T3 = {p,a,b,c} = {3,0,1,2}  (NO q=4)
        // non-{3,4} verts: {0,1} u {1,2} u {0,1,2} = {0,1,2} -> len 3.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0 = a
            [2.0, 0.0, 0.0],  // 1 = b
            [1.0, 2.0, 0.0],  // 2 = c
            [1.0, 0.7, 2.0],  // 3 = p (above plane abc)
            [1.0, 0.7, -2.0], // 4 = q (below plane abc)
        ];
        let tets = vec![
            [0, 1, 3, 4], // T1: a,b,p,q
            [1, 2, 3, 4], // T2: b,c,p,q
            [0, 1, 2, 3], // T3: a,b,c,p  (no q!)
        ];
        let mut dt = build_test_mesh(vertices, tets);
        let ring: Vec<usize> = (0..dt.tets.len()).filter(|&i| dt.is_live(i)).collect();
        assert_eq!(ring.len(), 3);
        // Sanity: only 2 of the 3 tets actually contain q=4.
        let n_with_q = ring
            .iter()
            .filter(|&&i| dt.tets[i].verts.contains(&4))
            .count();
        assert_eq!(n_with_q, 2, "setup: exactly two tets should contain q");

        let res = flip_3_to_2(
            &mut dt,
            [ring[0], ring[1], ring[2]],
            3,
            4,
            &Constraint::None,
        );
        let defects = adjacency_defects(&dt);
        eprintln!("ADV result={res:?} defects={defects:#?}");
        let live_after: Vec<[usize; 4]> = (0..dt.tets.len())
            .filter(|&i| dt.is_live(i))
            .map(|i| dt.tets[i].verts)
            .collect();
        eprintln!("ADV live tets after = {live_after:?}");
        // If the claim is real, res is Some AND defects is non-empty.
        if res.is_some() {
            assert!(
                defects.is_empty(),
                "CLAIM CONFIRMED: returned Some but mesh is corrupt: {defects:#?}"
            );
        }
    }

    #[test]
    fn audit_flip_3_to_2_external_relink() {
        // Build a 3-tet ring around edge (p,q) surrounded by 6 outer neighbors.
        // Ring vertices a,b,c form a triangle; p above, q below.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0 = a
            [2.0, 0.0, 0.0],  // 1 = b
            [1.0, 2.0, 0.0],  // 2 = c
            [1.0, 0.7, 2.0],  // 3 = p (above)
            [1.0, 0.7, -2.0], // 4 = q (below)
            // outer neighbors, one per external face of the ring (6 faces)
            [-3.0, -1.0, 3.0],  // 5
            [5.0, -1.0, 3.0],   // 6
            [1.0, 6.0, 3.0],    // 7
            [-3.0, -1.0, -3.0], // 8
            [5.0, -1.0, -3.0],  // 9
            [1.0, 6.0, -3.0],   //10
        ];
        // Ring tets around edge (3,4): each is (edge endpoints? ) -- actually a
        // 3-2 ring is 3 tets each sharing edge (p,q)=(3,4):
        //   (a,b,p,q),(b,c,p,q),(c,a,p,q)
        let tets = vec![
            [0, 1, 3, 4], // a,b,p,q
            [1, 2, 3, 4], // b,c,p,q
            [2, 0, 3, 4], // c,a,p,q
            // external neighbors: faces containing p=3 -> (a,b,p),(b,c,p),(c,a,p)
            [0, 1, 3, 7], // across (0,1,3)
            [1, 2, 3, 5], // across (1,2,3)
            [2, 0, 3, 6], // across (2,0,3)
            // faces containing q=4 -> (a,b,q),(b,c,q),(c,a,q)
            [0, 1, 4, 10],
            [1, 2, 4, 8],
            [2, 0, 4, 9],
        ];
        let mut dt = build_test_mesh(vertices, tets);
        let pre = adjacency_defects(&dt);
        assert!(pre.is_empty(), "precondition mesh already broken: {pre:?}");

        let ring: Vec<usize> = (0..dt.tets.len())
            .filter(|&i| {
                let mut v = dt.tets[i].verts;
                v.sort();
                dt.is_live(i)
                    && v.contains(&3)
                    && v.contains(&4)
                    && (v == [0, 1, 3, 4] || v == [1, 2, 3, 4] || v == [0, 2, 3, 4])
            })
            .collect();
        assert_eq!(ring.len(), 3, "expected 3 ring tets, got {ring:?}");

        let res = flip_3_to_2(
            &mut dt,
            [ring[0], ring[1], ring[2]],
            3,
            4,
            &Constraint::None,
        );
        assert!(res.is_some(), "3-2 flip should succeed here");
        let defects = adjacency_defects(&dt);
        assert!(
            defects.is_empty(),
            "3-2 flip left adjacency defects: {defects:#?}"
        );
    }

    // Cheap deterministic LCG for the fuzz test.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn audit_random_flip_fuzz() {
        let mut failures: Vec<String> = Vec::new();
        for seed in 0u64..400 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let npts = 8 + (lcg(&mut s) % 8) as usize;
            let mut pts: Vec<[f64; 3]> = Vec::new();
            for _ in 0..npts {
                let x = (lcg(&mut s) % 1000) as f64 / 100.0;
                let y = (lcg(&mut s) % 1000) as f64 / 100.0;
                let z = (lcg(&mut s) % 1000) as f64 / 100.0;
                pts.push([x, y, z]);
            }
            let mut dt = Delaunay3D::new(&pts);

            for _ in 0..20 {
                let choose = lcg(&mut s) % 3;
                let live: Vec<usize> = (0..dt.tets.len())
                    .filter(|&i| dt.is_live(i) && !dt.tets[i].is_hull())
                    .collect();
                if live.is_empty() {
                    break;
                }
                let t1 = live[(lcg(&mut s) as usize) % live.len()];
                let fi = (lcg(&mut s) % 4) as usize;
                let t2 = dt.tets[t1].adj[fi];
                let interior = t2 != usize::MAX && dt.is_live(t2) && !dt.tets[t2].is_hull() && {
                    let mut ok = true;
                    for &t in &[t1, t2] {
                        for ff in 0..4 {
                            let nb = dt.tets[t].adj[ff];
                            if nb == usize::MAX || nb == t1 || nb == t2 {
                                continue;
                            }
                            if !dt.is_live(nb) || dt.tets[nb].is_hull() {
                                ok = false;
                            }
                        }
                    }
                    ok
                };
                let did = if choose == 0 {
                    if interior {
                        flip_2_to_3(&mut dt, t1, t2, &Constraint::None).is_some()
                    } else {
                        false
                    }
                } else if choose == 1 {
                    if t2 != usize::MAX && dt.is_live(t2) && !dt.tets[t2].is_hull() {
                        flip_2_to_2(&mut dt, t1, t2, &Constraint::None).is_some()
                    } else {
                        false
                    }
                } else {
                    let v = dt.tets[t1].verts;
                    let ei = (lcg(&mut s) % 6) as usize;
                    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
                    let (p, q) = (v[edges[ei].0], v[edges[ei].1]);
                    let ring = tets_sharing_edge(&dt, p, q);
                    if ring.len() == 3
                        && ring.iter().all(|&t| dt.is_live(t) && !dt.tets[t].is_hull())
                    {
                        flip_3_to_2(
                            &mut dt,
                            [ring[0], ring[1], ring[2]],
                            p,
                            q,
                            &Constraint::None,
                        )
                        .is_some()
                    } else {
                        false
                    }
                };
                if did {
                    let defects = adjacency_defects(&dt);
                    if !defects.is_empty() {
                        failures.push(format!(
                            "seed {seed} kind {choose}: {} defect(s); first: {}",
                            defects.len(),
                            defects[0]
                        ));
                        break;
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "fuzz found adjacency defects:\n{}",
            failures.join("\n")
        );
    }

    /// Hull-aware recovery on a cube WITH interior points: after recovery every
    /// one of the 12 boundary triangles must be a proper BOUNDARY INTERFACE -
    /// present as a face of a live finite tet whose neighbour across that face
    /// is a HULL tet (`verts[3] == INFINITE`). This is exactly the assertion the
    /// original P0/P1 unit test was missing: it checked face *presence* but not
    /// that the hull adopted the same diagonal, so the cube passed the unit test
    /// yet leaked end-to-end (the carve flooded the interior). The fix routes the
    /// wrong-diagonal edge through the hull-aware `flip_nm` so both the interior
    /// and the hull adopt the wanted diagonal.
    #[test]
    fn test_cube_with_interior_all_faces_are_hull_interfaces() {
        // 10-scale cube corners + a 3x3x3 grid of strictly-interior points.
        let mut pts: Vec<[f64; 3]> = cube_vertices()
            .iter()
            .map(|v| [v[0] * 10.0, v[1] * 10.0, v[2] * 10.0])
            .collect();
        for &x in &[2.5, 5.0, 7.5] {
            for &y in &[2.5, 5.0, 7.5] {
                for &z in &[2.5, 5.0, 7.5] {
                    pts.push([x, y, z]);
                }
            }
        }
        let tris = cube_surface_tris();

        let mut dt = Delaunay3D::new(&pts);

        // Edges first, then faces - the order the conforming volume path uses.
        let mut eset = HashSet::default();
        for t in &tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            eset.insert((a.min(b), a.max(b)));
            eset.insert((b.min(c), b.max(c)));
            eset.insert((a.min(c), a.max(c)));
        }
        // Sorted for the same reason the production path sorts: recovery order
        // changes the mesh, so an unsorted HashSet makes this test flaky.
        let mut eset: Vec<(usize, usize)> = eset.into_iter().collect();
        eset.sort_unstable();
        for (a, b) in eset {
            recover_edge_by_flips(&mut dt, a, b);
        }
        let (_rec, failed, failed_faces) = recover_faces_no_bail(&mut dt, &tris);
        assert_eq!(
            failed, 0,
            "all 12 cube faces must be recoverable with interior points; failed {failed_faces:?}"
        );

        // The mesh must remain a valid, overlap-free simplicial complex: every
        // live finite tet positively oriented, the total finite volume exactly
        // the cube volume (overlaps would inflate it), and adjacency consistent.
        assert!(mesh_is_consistent(&dt), "mesh must stay consistent");
        let mut vol = 0.0;
        for i in 0..dt.tets.len() {
            if dt.is_live(i) && !dt.tets[i].is_hull() {
                let v = dt.tets[i].verts;
                vol += orient_3d(
                    dt.vertices[v[0]],
                    dt.vertices[v[1]],
                    dt.vertices[v[2]],
                    dt.vertices[v[3]],
                )
                .abs()
                    / 6.0;
            }
        }
        assert!(
            (vol - 1000.0).abs() < 1e-6,
            "finite tets must tile the cube exactly (no overlaps): vol={vol:.4}"
        );

        // THE MISSING ASSERTION: every boundary triangle is a proper interface
        // between a live finite tet and a HULL tet. Find each boundary face on a
        // finite tet and require its neighbour across that face to be a hull tet.
        for tri in &tris {
            let mut want = *tri;
            want.sort();
            let mut found_interface = false;
            'search: for i in 0..dt.tets.len() {
                if !dt.is_live(i) || dt.tets[i].is_hull() {
                    continue;
                }
                let v = dt.tets[i].verts;
                for fi in 0..4 {
                    let mut f = delaunay3d::opposite_face(v, fi);
                    f.sort();
                    if f == want {
                        let nb = dt.tets[i].adj[fi];
                        if nb != usize::MAX && dt.is_live(nb) && dt.tets[nb].is_hull() {
                            found_interface = true;
                            break 'search;
                        }
                    }
                }
            }
            assert!(
                found_interface,
                "boundary triangle {tri:?} must be an interface to a HULL tet \
                 (so the carve flood is blocked); it is not - the hull kept the \
                 wrong diagonal"
            );
        }
    }

    #[test]
    fn test_recover_face_by_edge_flips_pierced_facet() {
        // A missing boundary facet (0,1,2) is pierced by the interior edge
        // (3,4): three tets form the ring around (3,4), with ring vertices
        // 0,1,2. The segment (3,4) crosses the interior of triangle (0,1,2),
        // which is exactly why the facet is absent. recover_face_by_edge_flips
        // must reduce the star of (3,4) (a 3-ring → 3-to-2 flip) and recover the
        // facet. This is the general piercing-edge case (step 2b) - not a
        // coplanar-quad diagonal swap.
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3 (above z=0)
            [1.0, 0.7, -1.5], // 4 (below z=0)
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]]);

        // The facet is missing; its three edges are present; (3,4) pierces it.
        assert!(!face_exists_in_tets(&dt, 0, 1, 2));
        assert!(edge_exists_in_tets(&dt, 0, 1));
        assert!(edge_exists_in_tets(&dt, 1, 2));
        assert!(edge_exists_in_tets(&dt, 2, 0));
        assert!(edge_exists_in_tets(&dt, 3, 4));

        assert!(
            recover_face_by_edge_flips(&mut dt, 0, 1, 2),
            "piercing-edge facet recovery should recover (0,1,2)"
        );
        assert!(face_exists_in_tets(&dt, 0, 1, 2));
        // The obstructing edge (3,4) must be gone after the reduction.
        assert!(!edge_exists_in_tets(&dt, 3, 4));

        // All live finite tets must remain positively oriented.
        for (i, t) in dt.tets.iter().enumerate() {
            if !dt.is_live(i) || t.is_hull() {
                continue;
            }
            let v = t.verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(
                o > 0.0,
                "tet {i} non-positive after edge-flip recovery: {o}"
            );
        }
    }

    #[test]
    fn test_recover_face_by_edge_flips_bails_when_edge_missing() {
        // If one of the facet's own edges is absent, the piercing-edge recovery
        // must bail (return false) without mutating the mesh - the global edge
        // recovery owns edges. Here (3,4) is never an edge (two tets share face
        // (0,1,2)), so facet (0,3,4) has a missing edge (3,4).
        let vertices = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let mut dt = build_test_mesh(vertices, vec![[0, 1, 2, 3], [0, 1, 2, 4]]);
        let live_before = (0..dt.tets.len()).filter(|&i| dt.is_live(i)).count();

        assert!(!edge_exists_in_tets(&dt, 3, 4));
        assert!(
            !recover_face_by_edge_flips(&mut dt, 0, 3, 4),
            "must bail when a facet edge is missing"
        );
        // Mesh unchanged (no flips committed).
        let live_after = (0..dt.tets.len()).filter(|&i| dt.is_live(i)).count();
        assert_eq!(
            live_before, live_after,
            "no-op bail must not change the mesh"
        );
    }

    // -----------------------------------------------------------------------
    // Lawson restoration: conforming refinement CONVERGES on a thin slab
    // -----------------------------------------------------------------------

    /// Build the closed triangulated surface of an axis-aligned box
    /// [0,lx]x[0,ly]x[0,lz] from its 8 corners (each face split on a diagonal).
    /// Returns (verts, tris).
    fn box_surface(lx: f64, ly: f64, lz: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, 0.0, 0.0], // 0
            [lx, 0.0, 0.0],  // 1
            [lx, ly, 0.0],   // 2
            [0.0, ly, 0.0],  // 3
            [0.0, 0.0, lz],  // 4
            [lx, 0.0, lz],   // 5
            [lx, ly, lz],    // 6
            [0.0, ly, lz],   // 7
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2], // z=0
            [4, 5, 6],
            [4, 6, 7], // z=lz
            [0, 1, 5],
            [0, 5, 4], // y=0
            [2, 3, 7],
            [2, 7, 6], // y=ly
            [0, 4, 7],
            [0, 7, 3], // x=0
            [1, 2, 6],
            [1, 6, 5], // x=lx
        ];
        (verts, tris)
    }

    /// Flood-carve the interior of a fully boundary-conforming tetrahedralization
    /// (mirror of `mod::carve_interior`, replicated here so the test stays in this
    /// module) and return the enclosed volume.
    fn carved_volume(dt: &Delaunay3D, faces: &[[usize; 3]]) -> f64 {
        use std::collections::VecDeque;
        let bset: HashSet<[usize; 3]> = faces
            .iter()
            .map(|f| {
                let mut s = *f;
                s.sort();
                s
            })
            .collect();
        let n = dt.tets.len();
        let mut outside = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if dt.is_live(i) && dt.tets[i].is_hull() {
                outside[i] = true;
                q.push_back(i);
            }
        }
        while let Some(ti) = q.pop_front() {
            let verts = dt.tets[ti].verts;
            for (fi, &nb) in dt.tets[ti].adj.iter().enumerate() {
                if nb == usize::MAX || nb >= n || !dt.is_live(nb) || outside[nb] {
                    continue;
                }
                let mut face = delaunay3d::opposite_face(verts, fi);
                face.sort();
                if bset.contains(&face) {
                    continue;
                }
                outside[nb] = true;
                q.push_back(nb);
            }
        }
        let mut seen = HashSet::default();
        let mut vol = 0.0;
        for i in 0..n {
            if !dt.is_live(i) || dt.tets[i].is_hull() || outside[i] {
                continue;
            }
            let mut k = dt.tets[i].verts;
            k.sort();
            if seen.insert(k) {
                let v = dt.tets[i].verts;
                vol += orient_3d(
                    dt.vertices[v[0]],
                    dt.vertices[v[1]],
                    dt.vertices[v[2]],
                    dt.vertices[v[3]],
                )
                .abs()
                    / 6.0;
            }
        }
        vol
    }

    #[test]
    fn lawson_restore_converges_on_thin_slab() {
        // A 1 x 10 x 10 slab is the canonical hard case for local-split-only
        // refinement: the Delaunay of its corners picks face diagonals that do
        // not match the wanted boundary triangulation, and a thin solid makes the
        // sub-segments through a Steiner point NON-Delaunay - so the
        // local-split-only refinement recurses without converging. With Lawson
        // restoration after each split it must converge in a BOUNDED number of
        // splits, every boundary face present, and the carved volume must equal
        // the analytical box volume to 1e-9.
        let (verts, tris) = box_surface(1.0, 10.0, 10.0);
        let mut all_points = verts.clone();
        // A few interior points so the slab has genuine interior tets.
        for &x in &[0.5_f64] {
            for &y in &[2.5_f64, 5.0, 7.5] {
                for &z in &[2.5_f64, 5.0, 7.5] {
                    all_points.push([x, y, z]);
                }
            }
        }
        let n_pts_before = all_points.len();

        let mut dt = Delaunay3D::new(&all_points);
        let mut cur_faces: Vec<[usize; 3]> = tris.clone();

        let seg_ok = recover_segments_with_steiner(&mut dt, &mut cur_faces);
        assert!(seg_ok, "segment recovery must converge on the thin slab");
        let facet_ok = recover_facets_with_steiner(&mut dt, &mut cur_faces);
        assert!(facet_ok, "facet recovery must converge on the thin slab");

        // BOUNDED splits: the refined surface must not have blown up. A healthy
        // slab adds at most a handful of Steiner points; a non-converging cascade
        // would add hundreds. (The original 8 corners + 9 interior = 17 points;
        // allow a generous bound that still fails loudly on a sliver cascade.)
        let n_steiner = dt.vertices.len() - n_pts_before;
        assert!(
            n_steiner < 200,
            "refinement must converge with bounded splits; added {n_steiner} Steiner points"
        );

        // Every refined boundary face must be present in the mesh.
        let mut failed = 0usize;
        for f in &cur_faces {
            if !face_exists_in_tets(&dt, f[0], f[1], f[2]) {
                failed += 1;
            }
        }
        assert_eq!(failed, 0, "all refined boundary faces must be present");

        // The carved interior volume must equal the analytical slab volume.
        let analytical = 1.0 * 10.0 * 10.0;
        let carved = carved_volume(&dt, &cur_faces);
        assert!(
            (carved - analytical).abs() < 1e-9,
            "carved volume {carved} must equal analytical {analytical} within 1e-9"
        );
    }

    #[test]
    fn lawson_restore_converges_on_small_box() {
        // A well-shaped 1x1x1 cube with interior points: a sanity check that the
        // Lawson-restored conforming refinement still converges and carves the
        // exact unit volume (no regression on the easy case).
        let (verts, tris) = box_surface(1.0, 1.0, 1.0);
        let mut all_points = verts.clone();
        all_points.push([0.5, 0.5, 0.5]);
        let mut dt = Delaunay3D::new(&all_points);
        let mut cur_faces: Vec<[usize; 3]> = tris.clone();
        assert!(recover_segments_with_steiner(&mut dt, &mut cur_faces));
        assert!(recover_facets_with_steiner(&mut dt, &mut cur_faces));
        for f in &cur_faces {
            assert!(
                face_exists_in_tets(&dt, f[0], f[1], f[2]),
                "cube boundary face {f:?} must be present"
            );
        }
        let carved = carved_volume(&dt, &cur_faces);
        assert!(
            (carved - 1.0).abs() < 1e-9,
            "carved cube volume {carved} must equal 1.0 within 1e-9"
        );
    }
}
