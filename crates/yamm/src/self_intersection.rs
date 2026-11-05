//! Crossings of a solid's surface mesh with itself.
//!
//! A wall thinner than the faceting can come out of the mesher with its two
//! faces crossed: a chord of one face, cut at a loose angular tolerance, sags
//! through the chord of the other, so over a strip the outer surface lies
//! inside the inner one. The solid's volume hardly changes, so a volume check
//! passes it, but transport inside the strip sees the solid's boundary in the
//! wrong order: particles are lost, or, when nothing notices, scored in the
//! wrong cell.
//!
//! The strips can be a few triangles wide on a mesh of millions, too few for
//! sampling to find, so this tests every pair of the solid's triangles whose
//! bounding boxes meet (through [`TriangleBvh::self_pairs`]) with an exact
//! triangle-triangle crossing test.
//!
//! # What counts as a crossing
//!
//! Neighbouring triangles touch by construction, and a closed mesh is mostly
//! neighbours, so contact must not count. A pair is reported only when:
//!
//! * the two share no vertex, by index or by position within the tolerance
//!   (a seam between two faces meshed with their own copies of the seam's
//!   vertices, as a discontinuous mesh stores them, is a shared edge);
//! * each triangle reaches more than the tolerance through the other's plane
//!   on both sides of it (one that only dips into the other's plane by less
//!   is touching it);
//! * the segment where the two cross runs more than the tolerance inside both
//!   triangles (a crossing along an edge, or at a corner, is contact).
//!
//! The reported depth is the smallest of those reaches, so it measures how
//! far the two triangles pass through each other.
//!
//! Coplanar overlap (a sheet folded flat onto itself) has no reach through
//! either plane and is not reported; that is a fold, which a dihedral angle
//! check finds. A degenerate triangle, whose smallest height is within the
//! tolerance, is skipped: nothing can run more than the tolerance inside it.
//!
//! # Tolerance
//!
//! The tolerance is `rel_tol` times the diagonal of the bounding box of every
//! vertex the solids use. At the default 1e-6 that is the tolerance the
//! surface assembler welds vertices at, so two vertices this close are one
//! point of the CAD, and anything nearer than that is rounding, not
//! geometry. A crossing from faceting is as deep as the chordal sag, which is
//! orders of magnitude larger.

use crate::overlap::SolidBoundary;
use crate::volume::aabb_bvh::TriangleBvh;
use crate::{MesherError, Result};
use rayon::prelude::*;

/// One crossing pair, the deepest of a solid's.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossingSample {
    /// Indices of the two triangles in the solid's triangle list.
    pub triangles: (usize, usize),
    /// How far the two pass through each other, in model units.
    pub depth: f64,
    /// The deepest point of the segment where they cross.
    pub point: [f64; 3],
}

/// A solid whose surface mesh crosses itself.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfIntersection {
    pub solid: usize,
    /// Number of crossing triangle pairs.
    pub pairs: usize,
    /// Deepest crossing, in model units.
    pub max_depth: f64,
    /// The deepest crossings, deepest first.
    pub samples: Vec<CrossingSample>,
}

type Tri = [[f64; 3]; 3];

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn scale(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn lerp(a: &[f64; 3], b: &[f64; 3], s: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}

/// Unit normal of `t`, or `None` when its smallest height is within `tol`.
fn unit_normal(t: &Tri, tol: f64) -> Option<[f64; 3]> {
    let n = cross(&sub(&t[1], &t[0]), &sub(&t[2], &t[0]));
    let len = norm(&n);
    let longest = (0..3)
        .map(|k| norm(&sub(&t[(k + 1) % 3], &t[k])))
        .fold(0.0, f64::max);
    // Twice the area over the longest edge is the smallest height.
    if len.is_nan() || len <= tol * longest || longest == 0.0 {
        return None;
    }
    Some(scale(&n, 1.0 / len))
}

/// The two points where `t` crosses the plane its signed vertex distances
/// `d` are measured from. `d` must have a vertex on each side.
fn plane_cut(t: &Tri, d: &[f64; 3]) -> [[f64; 3]; 2] {
    let mut out = [[0.0; 3]; 2];
    let mut n = 0;
    for k in 0..3 {
        let (i, j) = (k, (k + 1) % 3);
        if (d[i] > 0.0) != (d[j] > 0.0) && n < 2 {
            out[n] = lerp(&t[i], &t[j], d[i] / (d[i] - d[j]));
            n += 1;
        }
    }
    out
}

/// Inward unit normals, within the triangle's plane, of its three edges.
fn edge_normals(t: &Tri, n: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for k in 0..3 {
        let m = cross(n, &sub(&t[(k + 1) % 3], &t[k]));
        let len = norm(&m);
        out[k] = scale(&m, 1.0 / len);
    }
    out
}

/// How far `a` and `b` pass through each other, and the deepest point of the
/// segment where they cross, when that is more than `tol`.
fn crossing(a: &Tri, b: &Tri, tol: f64) -> Option<(f64, [f64; 3])> {
    let na = unit_normal(a, tol)?;
    let nb = unit_normal(b, tol)?;
    let db = [0, 1, 2].map(|k| dot(&sub(&b[k], &a[0]), &na));
    let da = [0, 1, 2].map(|k| dot(&sub(&a[k], &b[0]), &nb));
    let reach = |d: &[f64; 3]| {
        let hi = d.iter().copied().fold(f64::MIN, f64::max);
        let lo = d.iter().copied().fold(f64::MAX, f64::min);
        hi.min(-lo)
    };
    let through = reach(&db).min(reach(&da));
    if through.is_nan() || through <= tol {
        return None;
    }
    let line = cross(&na, &nb);
    if norm(&line) < 1e-12 {
        return None;
    }
    let sa = plane_cut(a, &da);
    let sb = plane_cut(b, &db);
    let ta = [dot(&sa[0], &line), dot(&sa[1], &line)];
    let tb = [dot(&sb[0], &line), dot(&sb[1], &line)];
    let lo = ta[0].min(ta[1]).max(tb[0].min(tb[1]));
    let hi = ta[0].max(ta[1]).min(tb[0].max(tb[1]));
    if hi.is_nan() || lo.is_nan() || hi <= lo || ta[0] == ta[1] {
        return None;
    }
    // The shared segment, located on a's cut.
    let at = |t: f64| lerp(&sa[0], &sa[1], (t - ta[0]) / (ta[1] - ta[0]));
    let (p, q) = (at(lo), at(hi));
    // Distance inside both triangles is the smallest of six edge-line
    // distances, each linear along the segment, so the deepest point is an
    // end or where two of them meet.
    let mut lines: Vec<(f64, f64)> = Vec::with_capacity(6);
    for (t, n) in [(a, &na), (b, &nb)] {
        let m = edge_normals(t, n);
        for k in 0..3 {
            let v0 = dot(&sub(&p, &t[k]), &m[k]);
            let v1 = dot(&sub(&q, &t[k]), &m[k]);
            lines.push((v0, v1 - v0));
        }
    }
    let depth_at = |u: f64| {
        lines
            .iter()
            .map(|(v0, s)| v0 + s * u)
            .fold(f64::MAX, f64::min)
    };
    let mut best = (depth_at(0.0), 0.0);
    let mut consider = |u: f64| {
        let d = depth_at(u);
        if d > best.0 {
            best = (d, u);
        }
    };
    consider(1.0);
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            let ds = lines[i].1 - lines[j].1;
            if ds != 0.0 {
                let u = (lines[j].0 - lines[i].0) / ds;
                if u > 0.0 && u < 1.0 {
                    consider(u);
                }
            }
        }
    }
    let depth = best.0.min(through);
    if depth.is_nan() || depth <= tol {
        return None;
    }
    Some((depth, lerp(&p, &q, best.1)))
}

/// Find the solids whose surface meshes cross themselves.
///
/// `solids` gives each solid's id and its full closed boundary, as indices
/// into `vertices`. Every pair of a solid's triangles that share no vertex is
/// tested; see the module documentation for what counts as a crossing. The
/// tolerance is `rel_tol` times the diagonal of the box around every vertex
/// the solids use. Each solid found reports its crossing pair count, the
/// deepest crossing and up to `samples` of the deepest crossings.
///
/// # Errors
///
/// [`MesherError::InvalidInput`] when a triangle indexes past `vertices`, a
/// vertex it uses is not finite, or `rel_tol` is negative or not finite.
pub fn find_self_intersections(
    vertices: &[[f64; 3]],
    solids: &[SolidBoundary],
    rel_tol: f64,
    samples: usize,
) -> Result<Vec<SelfIntersection>> {
    if !(rel_tol >= 0.0 && rel_tol.is_finite()) {
        return Err(MesherError::InvalidInput(format!(
            "rel_tol must be a non-negative number, got {rel_tol}"
        )));
    }
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for (id, tris) in solids {
        for t in tris {
            for &v in t {
                let p = vertices.get(v).ok_or_else(|| {
                    MesherError::InvalidInput(format!(
                        "solid {id}: triangle {t:?} indexes past the {} vertices",
                        vertices.len()
                    ))
                })?;
                if !p.iter().all(|x| x.is_finite()) {
                    return Err(MesherError::InvalidInput(format!(
                        "solid {id}: vertex {v} is not finite: {p:?}"
                    )));
                }
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
    }
    if lo[0] > hi[0] {
        return Ok(Vec::new());
    }
    let tol = rel_tol * norm(&sub(&hi, &lo));
    let tol_sq = tol * tol;

    let found = solids
        .par_iter()
        .filter_map(|(id, tris)| {
            let data: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
                .iter()
                .map(|t| (vertices[t[0]], vertices[t[1]], vertices[t[2]]))
                .collect();
            let bvh = TriangleBvh::new(&data);
            let mut hits = bvh.self_pairs(tol, |i, ti, j, tj| {
                let (vi, vj) = (&tris[i], &tris[j]);
                if vi.iter().any(|v| vj.contains(v)) {
                    return None;
                }
                let near = ti.iter().any(|p| {
                    tj.iter().any(|q| {
                        let d = sub(p, q);
                        dot(&d, &d) <= tol_sq
                    })
                });
                if near {
                    return None;
                }
                crossing(ti, tj, tol)
            });
            if hits.is_empty() {
                return None;
            }
            let pairs = hits.len();
            // Deepest first; ties in pair order, so the samples do not depend
            // on the thread scheduling.
            hits.sort_by(|x, y| y.2 .0.total_cmp(&x.2 .0).then((x.0, x.1).cmp(&(y.0, y.1))));
            let max_depth = hits[0].2 .0;
            let samples = hits
                .iter()
                .take(samples)
                .map(|&(i, j, (depth, point))| CrossingSample {
                    triangles: (i, j),
                    depth,
                    point,
                })
                .collect();
            Some(SelfIntersection {
                solid: *id,
                pairs,
                max_depth,
                samples,
            })
        })
        .collect();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed box `[0, 10] x [0, 10] x [0, 1]` whose top face is a fan of
    /// four triangles around a centre vertex at height `apex`. Below 0 the
    /// apex has gone through the bottom face, as a thin wall's two faces do
    /// when they cross.
    fn dented_box(apex: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let v = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [10.0, 10.0, 1.0],
            [0.0, 10.0, 1.0],
            [5.0, 5.0, apex],
        ];
        let t = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 4, 8],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (v, t)
    }

    fn find(v: &[[f64; 3]], t: Vec<[usize; 3]>) -> Vec<SelfIntersection> {
        find_self_intersections(v, &[(7, t)], 1e-6, 4).unwrap()
    }

    #[test]
    fn crossed_walls_are_found() {
        let (v, t) = dented_box(-0.5);
        let out = find(&v, t);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].solid, 7);
        // Each fan triangle crosses the bottom triangle under it; the one
        // across the diagonal it only touches at a corner of the crossing.
        assert_eq!(out[0].pairs, 4);
        assert!(out[0].max_depth > 0.1 && out[0].max_depth <= 0.5);
        for s in &out[0].samples {
            assert!(s.point[2].abs() < 1e-9, "{:?}", s.point);
        }
    }

    #[test]
    fn a_sound_dent_is_not_a_crossing() {
        let (v, t) = dented_box(0.5);
        assert!(find(&v, t).is_empty());
    }

    #[test]
    fn touching_is_not_crossing() {
        // The apex sits on the bottom face, and just short of through it.
        for apex in [0.0, -1e-7, 1e-7] {
            let (v, t) = dented_box(apex);
            assert!(find(&v, t).is_empty(), "apex {apex}");
        }
    }

    #[test]
    fn a_crossing_deeper_than_the_tolerance_is_found() {
        let (v, t) = dented_box(-1e-3);
        let out = find(&v, t);
        assert_eq!(out.len(), 1);
        assert!(out[0].max_depth > 1e-4 && out[0].max_depth <= 1e-3);
    }

    #[test]
    fn shared_edges_by_position_are_contact() {
        // Every triangle with its own copies of its vertices, as faces meshed
        // apart store their seams: the neighbours touch along whole edges.
        for apex in [0.5, -0.5] {
            let (v, t) = dented_box(apex);
            let mut own = Vec::new();
            let mut tris = Vec::new();
            for tri in &t {
                let base = own.len();
                own.extend(tri.iter().map(|&i| v[i]));
                tris.push([base, base + 1, base + 2]);
            }
            let out = find(&own, tris);
            assert_eq!(out.len(), usize::from(apex < 0.0), "apex {apex}");
        }
    }

    #[test]
    fn degenerate_triangles_are_skipped() {
        let (mut v, mut t) = dented_box(0.5);
        // A needle and a point through the middle of the box, crossing the
        // bottom and top faces.
        v.push([5.0, 5.0, -1.0]);
        v.push([5.0, 5.0, 2.0]);
        v.push([5.0, 5.0, 3.0]);
        v.push([4.0, 4.0, 0.0]);
        t.push([9, 10, 11]);
        t.push([12, 12, 12]);
        t.push([9, 9, 10]);
        assert!(find(&v, t).is_empty());
    }

    #[test]
    fn separate_shells_of_one_solid_crossing() {
        // A hollow solid whose void pokes out through the outer surface.
        let (mut v, mut t) = dented_box(1.0);
        let n = v.len();
        let (vv, tv) = dented_box(1.0);
        v.extend(
            vv.iter()
                .map(|p| [p[0] * 0.5 + 2.5, p[1] * 0.5 + 2.5, p[2] + 0.5]),
        );
        t.extend(tv.iter().map(|x| [x[0] + n, x[2] + n, x[1] + n]));
        let out = find(&v, t.clone());
        assert_eq!(out.len(), 1);
        // Moved fully inside, the void is sound.
        for p in &mut v[n..] {
            p[2] = 0.25 + (p[2] - 0.5) * 0.5;
        }
        assert!(find(&v, t).is_empty());
    }

    #[test]
    fn solids_are_tested_on_their_own() {
        // Two solids crossing each other are an overlap, not a self-crossing.
        let (v, t) = dented_box(0.5);
        let n = v.len();
        let mut all = v.clone();
        all.extend(v.iter().map(|p| [p[0] + 5.0, p[1], p[2]]));
        let moved: Vec<[usize; 3]> = t.iter().map(|x| [x[0] + n, x[1] + n, x[2] + n]).collect();
        let out = find_self_intersections(&all, &[(1, t), (2, moved)], 1e-6, 4).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn bad_input_is_an_error() {
        let (v, mut t) = dented_box(0.5);
        assert!(find_self_intersections(&v, &[(1, t.clone())], -1.0, 4).is_err());
        assert!(find_self_intersections(&v, &[(1, t.clone())], f64::NAN, 4).is_err());
        let mut bad = v.clone();
        bad[3][1] = f64::INFINITY;
        assert!(find_self_intersections(&bad, &[(1, t.clone())], 1e-6, 4).is_err());
        t.push([0, 1, 99]);
        assert!(find_self_intersections(&v, &[(1, t)], 1e-6, 4).is_err());
        assert!(find_self_intersections(&v, &[], 1e-6, 4)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn self_pairs_matches_brute_force() {
        // Every pair whose boxes meet, against testing all pairs.
        let mut tris = Vec::new();
        for i in 0..200 {
            let x = (i as f64 * 0.37).sin() * 5.0;
            let y = (i as f64 * 0.73).cos() * 5.0;
            let z = (i as f64 * 0.11).sin() * 5.0;
            tris.push(([x, y, z], [x + 1.0, y, z], [x, y + 1.0, z + 0.5]));
        }
        let bvh = TriangleBvh::new(&tris);
        let got: Vec<(usize, usize)> = bvh
            .self_pairs(0.0, |_, _, _, _| Some(()))
            .into_iter()
            .map(|(i, j, ())| (i, j))
            .collect();
        let bx = |t: &([f64; 3], [f64; 3], [f64; 3])| {
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for p in [t.0, t.1, t.2] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        };
        let mut want = Vec::new();
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                let (a, b) = (bx(&tris[i]), bx(&tris[j]));
                if (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k]) {
                    want.push((i, j));
                }
            }
        }
        assert!(!want.is_empty());
        assert_eq!(got, want);
    }

    #[test]
    fn the_result_does_not_depend_on_the_threads() {
        let (v, t) = dented_box(-0.5);
        let a = find(&v, t.clone());
        let b = find(&v, t);
        assert_eq!(a, b);
    }
}
