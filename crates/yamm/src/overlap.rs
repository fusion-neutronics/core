//! Inter-solid mesh overlap detection.
//!
//! DAGMC requires every point in space to belong to exactly one volume. Two
//! solids whose *meshes* interpenetrate break that even when the input CAD was
//! clean: a particle entering the doubly-claimed region is tracked in whichever
//! volume DAGMC thinks it is in, and when it hits the other volume's surface
//! from the inside the sense lookup hands it a volume that does not contain the
//! point. Sometimes that is a lost particle; sometimes it is silently the wrong
//! material, which is worse because nothing reports it.
//!
//! The CAD-level check catches overlapping *input*. This catches overlapping
//! *output*, which is a different failure: faceting two near-tangent curved
//! surfaces can make them cross by up to the chord tolerance even though the
//! solids only touch, and with `imprint=False` nothing has reconciled the
//! surfaces at all.
//!
//! # What is and is not an overlap
//!
//! Contact is not overlap, and a DAGMC assembly is mostly contact - a check
//! that flags touching solids is worse than no check. Two things separate them:
//!
//! * **Shared faces are excluded.** After imprinting, a face shared by two
//!   solids is stored once and both solids reference identical triangles. Those
//!   coincide exactly and would otherwise register as interpenetration. The
//!   caller passes the vertices lying on such faces and they are not tested.
//!   Note they are still part of the *surface* each solid is tested against:
//!   the inside/outside test needs each solid's full closed boundary.
//! * **Depth is reported, not a boolean.** Faceting artefacts penetrate by
//!   about the chord tolerance; a real overlap penetrates by a fraction of the
//!   geometry. Same test, magnitudes apart, so the caller thresholds on a
//!   number rather than this module guessing.
//!
//! # Coverage
//!
//! Two directional passes per pair, each testing one solid's features against
//! the other's closed surface:
//!
//! * vertex containment - catches deep overlap, and full containment via the
//!   opposite direction (if A swallows B, no vertex of A is inside B, but every
//!   vertex of B is inside A);
//! * edge crossing - catches shallow or glancing overlap where no vertex of
//!   either solid falls inside the other;
//! * face-interior sampling - catches the case both of the above miss, where
//!   the intersection curve runs exactly along shared edges so nothing on the
//!   1-skeleton is ever strictly inside.
//!
//! That third probe is not hypothetical padding. Two identical cubes offset
//! along one axis overlap in half their volume, yet every vertex and every edge
//! of each lies exactly on the other's surface, because their cross-sections
//! coincide: the only strictly-interior features are face interiors. An
//! axis-aligned duplicate of a part is an ordinary modelling slip, so a check
//! built only on vertices and edges would miss a 50% overlap.
//!
//! With face interiors included these probe the 0-, 1- and 2-dimensional
//! features of each mesh, which is what makes the pair equivalent to
//! triangle-triangle intersection without a tri-tri kernel: an interpenetration
//! of two closed surfaces puts some feature of one strictly inside the other.

use crate::volume::aabb_bvh::TriangleBvh;
use rayon::prelude::*;

/// A solid's id paired with its full closed boundary triangles.
pub type SolidBoundary = (usize, Vec<[usize; 3]>);

/// One overlapping pair.
#[derive(Debug, Clone)]
pub struct SolidOverlap {
    pub solid_a: usize,
    pub solid_b: usize,
    /// Deepest penetration found, in model units. The number to threshold on:
    /// compare against the meshing tolerance to tell a faceting artefact from a
    /// real overlap.
    pub max_depth: f64,
    /// Vertices of one solid found strictly inside the other.
    pub inside_vertices: usize,
    /// Edges of one solid that cross into the other.
    pub crossing_edges: usize,
    /// Triangles of one solid whose interior lies inside the other. The only
    /// signal when the two share cross-sections, where nothing on the
    /// 1-skeleton is ever strictly inside.
    pub inside_faces: usize,
    /// Boundary area of the doubly-claimed region, estimated by summing the
    /// triangle area each solid contributes inside the other.
    pub lens_area: f64,
    /// Estimated fraction of isotropic ray-fires that traverse the
    /// doubly-claimed region -- see [`find_solid_overlaps`].
    pub exposure: f64,
}

fn tri_area(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct Prepared {
    id: usize,
    bvh: TriangleBvh,
    tris: Vec<[usize; 3]>,
    /// Testable vertices: those used by this solid and not on a shared face.
    verts: Vec<usize>,
    /// Testable edges, deduplicated, both endpoints testable.
    edges: Vec<(usize, usize)>,
    lo: [f64; 3],
    hi: [f64; 3],
}

fn aabbs_disjoint(a: &Prepared, b: &Prepared, pad: f64) -> bool {
    (0..3).any(|k| a.hi[k] + pad < b.lo[k] || b.hi[k] + pad < a.lo[k])
}

/// How deep inside `bvh` the point is, or `None` if it is outside.
fn depth_inside(bvh: &TriangleBvh, p: &[f64; 3], eps: f64) -> Option<f64> {
    if !bvh.is_point_inside(p) {
        return None;
    }
    let d = dist(p, &bvh.nearest_point_on_surface(p));
    // A point exactly on the surface -- a touching contact, or a vertex of a
    // face shared with a third solid -- reads as "inside" from ray parity about
    // half the time. Depth separates that from real penetration.
    if d > eps {
        Some(d)
    } else {
        None
    }
}

/// Find pairs of solids whose meshes share volume.
///
/// `solids` gives each solid's id and its **full closed boundary** - shared
/// faces included, because the inside/outside test needs a closed surface.
/// `shared_vertices` lists vertices lying on faces shared between solids; those
/// are excluded from being *tested*, since a shared face's triangles coincide
/// exactly and are contact rather than overlap.
///
/// `depth_eps` is the absolute depth below which a penetration is treated as a
/// point sitting on the surface rather than inside it.
///
/// `exposure` estimates the fraction of isotropic ray-fires that traverse the
/// doubly-claimed region. By the Cauchy/Santaló random-chord result, the
/// fraction of random chords of a convex enclosure meeting a convex body inside
/// it is the ratio of their surface areas, so this is the lens boundary area
/// over the model bounding box's area.
///
/// Treat it as an order of magnitude, not a bound. It is wrong in both
/// directions: a non-convex lens has its mean projected area counted with
/// multiplicity, which pushes the estimate up, while any part of the lens
/// boundary that coincides with both surfaces is inside neither solid and is
/// counted by neither pass, which pushes it down - two identical cubes
/// half-overlapped come out about 2x low for exactly that reason. What it is
/// reliably good for is the job it exists to do: separating a faceting sliver
/// at 1e-8 from a real overlap at 1e-2 without anyone having to run a transport
/// check to find out.
///
/// It also estimates *affected* tracks, not lost ones. Some mis-tracked
/// particles are silently given the wrong material rather than being lost,
/// which is why a clean lost-particle count does not certify the geometry.
pub fn find_solid_overlaps(
    vertices: &[[f64; 3]],
    solids: &[SolidBoundary],
    shared_vertices: &[usize],
    depth_eps: f64,
) -> Vec<SolidOverlap> {
    if solids.len() < 2 || vertices.is_empty() {
        return Vec::new();
    }

    let mut shared: Vec<usize> = shared_vertices.to_vec();
    shared.sort_unstable();
    shared.dedup();
    let is_shared = |v: usize| shared.binary_search(&v).is_ok();

    let prepared: Vec<Prepared> = solids
        .iter()
        .map(|(id, tris)| {
            let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
                .iter()
                .map(|t| (vertices[t[0]], vertices[t[1]], vertices[t[2]]))
                .collect();
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            let mut verts: Vec<usize> = Vec::new();
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for t in tris {
                for &v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(vertices[v][k]);
                        hi[k] = hi[k].max(vertices[v][k]);
                    }
                    if !is_shared(v) {
                        verts.push(v);
                    }
                }
                for &(u, w) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    if !is_shared(u) && !is_shared(w) {
                        edges.push((u.min(w), u.max(w)));
                    }
                }
            }
            // Sorted rather than hashed: iteration order here reaches the
            // reported depth and counts, so it must not vary run to run.
            verts.sort_unstable();
            verts.dedup();
            edges.sort_unstable();
            edges.dedup();
            Prepared {
                id: *id,
                bvh: TriangleBvh::new(&tri_data),
                tris: tris.clone(),
                verts,
                edges,
                lo,
                hi,
            }
        })
        .collect();

    // Denominator for `exposure`: the model bounding box's surface area.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &prepared {
        for k in 0..3 {
            lo[k] = lo[k].min(p.lo[k]);
            hi[k] = hi[k].max(p.hi[k]);
        }
    }
    let (dx, dy, dz) = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    let bbox_area = 2.0 * (dx * dy + dy * dz + dz * dx);

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..prepared.len() {
        for j in (i + 1)..prepared.len() {
            if !aabbs_disjoint(&prepared[i], &prepared[j], depth_eps) {
                pairs.push((i, j));
            }
        }
    }

    let mut found: Vec<SolidOverlap> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = (&prepared[i], &prepared[j]);
            let (da, la) = directional(a, b, vertices, depth_eps);
            let (db, lb) = directional(b, a, vertices, depth_eps);
            let max_depth = da.0.max(db.0);
            let inside_vertices = da.1 + db.1;
            let crossing_edges = da.2 + db.2;
            let inside_faces = da.3 + db.3;
            if inside_vertices == 0 && crossing_edges == 0 && inside_faces == 0 {
                return None;
            }
            let lens_area = la + lb;
            Some(SolidOverlap {
                solid_a: a.id,
                solid_b: b.id,
                max_depth,
                inside_vertices,
                crossing_edges,
                inside_faces,
                lens_area,
                exposure: if bbox_area > 0.0 {
                    lens_area / bbox_area
                } else {
                    0.0
                },
            })
        })
        .collect();

    // Deterministic order regardless of how rayon scheduled the pairs.
    found.sort_by(|x, y| {
        (x.solid_a, x.solid_b)
            .cmp(&(y.solid_a, y.solid_b))
            .then(y.max_depth.total_cmp(&x.max_depth))
    });
    found
}

/// Test `a`'s features against `b`'s closed surface.
///
/// Returns `((max_depth, inside_vertices, crossing_edges, inside_faces), lens_area)`.
fn directional(
    a: &Prepared,
    b: &Prepared,
    vertices: &[[f64; 3]],
    eps: f64,
) -> ((f64, usize, usize, usize), f64) {
    let mut max_depth = 0.0f64;
    let mut n_inside = 0usize;

    let mut inside_vert: Vec<usize> = Vec::new();
    for &v in &a.verts {
        if let Some(d) = depth_inside(&b.bvh, &vertices[v], eps) {
            max_depth = max_depth.max(d);
            n_inside += 1;
            inside_vert.push(v);
        }
    }
    inside_vert.sort_unstable();

    // Shallow overlap can cross without putting any vertex inside (two thin
    // slabs meeting edge-on). `segment_crosses_boundary` finds those, but it
    // also fires on an edge lying in a contact plane, so confirm with sampled
    // depth along the edge -- which also yields a depth for a case the vertex
    // pass reports nothing for.
    let mut n_crossing = 0usize;
    for &(u, w) in &a.edges {
        let (p, q) = (vertices[u], vertices[w]);
        if !b.bvh.segment_crosses_boundary(&p, &q) {
            continue;
        }
        let mut deepest = 0.0f64;
        for s in 1..8 {
            let t = s as f64 / 8.0;
            let m = [
                p[0] + (q[0] - p[0]) * t,
                p[1] + (q[1] - p[1]) * t,
                p[2] + (q[2] - p[2]) * t,
            ];
            if let Some(d) = depth_inside(&b.bvh, &m, eps) {
                deepest = deepest.max(d);
            }
        }
        if deepest > 0.0 {
            n_crossing += 1;
            max_depth = max_depth.max(deepest);
        }
    }

    // Face interiors. Where two solids share cross-sections the intersection
    // curve runs along shared edges, so nothing on the 1-skeleton is ever
    // strictly inside and the two passes above both report nothing; the
    // centroid of a face lying inside the other solid still is. Also credits
    // the lens area, which the vertex pass alone would score as zero.
    let mut lens_area = 0.0f64;
    let mut n_inside_faces = 0usize;
    for t in &a.tris {
        let (p0, p1, p2) = (vertices[t[0]], vertices[t[1]], vertices[t[2]]);
        let k = t
            .iter()
            .filter(|v| inside_vert.binary_search(v).is_ok())
            .count();
        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        let centre_in = depth_inside(&b.bvh, &centroid, eps);
        if let Some(d) = centre_in {
            max_depth = max_depth.max(d);
            n_inside_faces += 1;
        }
        // A triangle is credited whole when its centre and all corners are
        // inside, and by the fraction of corners inside otherwise -- it is then
        // clipped by the intersection curve. A triangle whose corners all sit
        // exactly on the surface but whose interior is inside counts whole:
        // that is the coincident-cross-section case, fully inside despite
        // scoring zero corners.
        let frac = if k == 3 || centre_in.is_some() {
            // Whole. A triangle whose interior is inside counts fully even with
            // no corner inside -- that is the coincident-cross-section case,
            // where the corners sit exactly on the other surface.
            1.0
        } else {
            // Clipped by the intersection curve; credit the corners inside.
            k as f64 / 3.0
        };
        if frac > 0.0 {
            lens_area += tri_area(&p0, &p1, &p2) * frac;
        }
    }
    ((max_depth, n_inside, n_crossing, n_inside_faces), lens_area)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box as a closed triangle mesh, appended to `verts`.
    fn push_box(verts: &mut Vec<[f64; 3]>, lo: [f64; 3], hi: [f64; 3]) -> Vec<[usize; 3]> {
        let o = verts.len();
        for &(x, y, z) in &[
            (lo[0], lo[1], lo[2]),
            (hi[0], lo[1], lo[2]),
            (hi[0], hi[1], lo[2]),
            (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]),
            (hi[0], lo[1], hi[2]),
            (hi[0], hi[1], hi[2]),
            (lo[0], hi[1], hi[2]),
        ] {
            verts.push([x, y, z]);
        }
        let f = |a: usize, b: usize, c: usize| [o + a, o + b, o + c];
        vec![
            f(0, 2, 1),
            f(0, 3, 2),
            f(4, 5, 6),
            f(4, 6, 7),
            f(0, 1, 5),
            f(0, 5, 4),
            f(1, 2, 6),
            f(1, 6, 5),
            f(2, 3, 7),
            f(2, 7, 6),
            f(3, 0, 4),
            f(3, 4, 7),
        ]
    }

    fn two_boxes(offset: f64) -> (Vec<[f64; 3]>, Vec<SolidBoundary>) {
        let mut v = Vec::new();
        let a = push_box(&mut v, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = push_box(&mut v, [offset, 0.0, 0.0], [offset + 10.0, 10.0, 10.0]);
        (v, vec![(1, a), (2, b)])
    }

    #[test]
    fn overlapping_boxes_are_found_with_depth() {
        let (v, s) = two_boxes(5.0);
        let out = find_solid_overlaps(&v, &s, &[], 1e-9);
        assert_eq!(out.len(), 1, "expected one overlapping pair, got {out:?}");
        assert_eq!((out[0].solid_a, out[0].solid_b), (1, 2));
        // Deepest interior point of the lens is 5 from the nearest wall in x,
        // but only 5 from the y/z walls too -- the nearest surface bounds it.
        assert!(
            out[0].max_depth > 1.0,
            "depth should reflect a real overlap, got {}",
            out[0].max_depth
        );
        // Deliberately NOT asserting inside_vertices > 0: these two cubes share
        // y/z extents, so every vertex and edge of each lies exactly on the
        // other's surface and only the face interiors are strictly inside. That
        // degeneracy is why the face probe exists, and pinning it here stops
        // anyone "simplifying" the probe away.
        assert_eq!(
            out[0].inside_vertices, 0,
            "coincident cross-sections put no vertex strictly inside"
        );
        assert!(
            out[0].inside_faces > 0,
            "the overlap must be found via face interiors, got {:?}",
            out[0]
        );
    }

    #[test]
    fn touching_boxes_are_not_an_overlap() {
        let (v, s) = two_boxes(10.0);
        assert!(
            find_solid_overlaps(&v, &s, &[], 1e-9).is_empty(),
            "face-to-face contact must not report an overlap"
        );
    }

    #[test]
    fn disjoint_boxes_are_not_an_overlap() {
        let (v, s) = two_boxes(25.0);
        assert!(find_solid_overlaps(&v, &s, &[], 1e-9).is_empty());
    }

    #[test]
    fn full_containment_is_found() {
        // No vertex of the outer box is inside the inner one; only the opposite
        // direction sees it, which is why both are tested.
        let mut v = Vec::new();
        let outer = push_box(&mut v, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let inner = push_box(&mut v, [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]);
        let out = find_solid_overlaps(&v, &[(1, outer), (2, inner)], &[], 1e-9);
        assert_eq!(out.len(), 1, "containment must be reported");
        assert!(out[0].inside_vertices >= 8, "{:?}", out[0]);
    }

    #[test]
    fn shared_vertices_are_excluded() {
        // Two boxes meeting at x=10 whose contact vertices are declared shared:
        // still not an overlap, and excluding them must not break the surfaces
        // they are tested against.
        let (v, s) = two_boxes(10.0);
        let shared: Vec<usize> = (0..v.len())
            .filter(|&i| (v[i][0] - 10.0).abs() < 1e-12)
            .collect();
        assert!(!shared.is_empty());
        assert!(find_solid_overlaps(&v, &s, &shared, 1e-9).is_empty());
    }

    #[test]
    fn depth_scales_with_the_overlap() {
        let shallow = find_solid_overlaps(&two_boxes(9.5).0, &two_boxes(9.5).1, &[], 1e-9);
        let deep = find_solid_overlaps(&two_boxes(5.0).0, &two_boxes(5.0).1, &[], 1e-9);
        assert_eq!(shallow.len(), 1);
        assert_eq!(deep.len(), 1);
        assert!(
            shallow[0].max_depth < deep[0].max_depth,
            "a 0.5 overlap must read shallower than a 5.0 one: {} vs {}",
            shallow[0].max_depth,
            deep[0].max_depth
        );
        assert!(
            shallow[0].exposure < deep[0].exposure,
            "exposure must scale with the overlap too"
        );
    }

    #[test]
    fn results_are_deterministic() {
        let (v, s) = two_boxes(5.0);
        let a = find_solid_overlaps(&v, &s, &[], 1e-9);
        let b = find_solid_overlaps(&v, &s, &[], 1e-9);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].max_depth, b[0].max_depth);
        assert_eq!(a[0].inside_vertices, b[0].inside_vertices);
        assert_eq!(a[0].lens_area, b[0].lens_area);
    }

    #[test]
    fn single_solid_has_nothing_to_overlap() {
        let mut v = Vec::new();
        let a = push_box(&mut v, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(find_solid_overlaps(&v, &[(1, a)], &[], 1e-9).is_empty());
    }
}
