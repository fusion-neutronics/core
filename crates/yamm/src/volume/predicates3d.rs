#![allow(clippy::too_many_arguments)]
/// Robust 3D geometric predicates using Shewchuk's exact arithmetic.
use geometry_predicates::{insphere, orient2d, orient3d};

/// Returns positive if the tet (a, b, c, d) is positively oriented
/// (d is above the plane of a, b, c when a, b, c is CCW from d's view).
/// Negative if negatively oriented, zero if coplanar.
pub fn orient_3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    // geometry_predicates uses Shewchuk's convention where the sign is
    // opposite to the "positive tet orientation" convention. We negate.
    -orient3d(a, b, c, d)
}

/// Returns positive if e is inside the circumsphere of (a, b, c, d),
/// negative if outside, zero if on the sphere.
/// Assumes (a, b, c, d) are positively oriented (orient_3d > 0).
pub fn in_sphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], e: [f64; 3]) -> f64 {
    // With negated orient3d, insphere sign also flips.
    -insphere(a, b, c, d, e)
}

/// Simulation-of-Simplicity (SoS, Edelsbrunner-Mucke 1990) tie-break for the
/// 5-point insphere test, faithfully ported from TetGen's `insphere_s`
/// (tetgen.cxx:5141-5193). When the exact `in_sphere` returns exactly 0.0
/// (cospherical degeneracy), this resolves the ambiguous sign DETERMINISTICALLY
/// by the global vertex indices (`i*` = index into `Delaunay3D::vertices`,
/// i.e. `Tet.verts`). The same five points with the same indices always return
/// the same nonzero sign, which is what makes the cospherical flip decisions
/// globally consistent (breaks the LShaped livelock).
///
/// Built on the EXISTING WRAPPED (sign-negated) `in_sphere`/`orient_3d`, so it
/// keeps the SAME convention as the rest of this module: a positive result
/// means `e` is inside the circumsphere of the (positively-oriented) base
/// `{a,b,c,d}`, exactly like `in_sphere`. The single global -1 from negating
/// every cascade term does not change WHICH term is first-nonzero, only the
/// final sign (the desired behaviour: SoS sign agrees with the float sign on
/// non-degenerate input).
///
/// PRECONDITION (TetGen): the base simplex `{a,b,c,d}` (the FIRST FOUR args, NOT
/// sorted) must be positively oriented (`orient_3d(a,b,c,d) > 0`); the caller
/// guarantees this (or negates the result).
//
// NOTE: currently has no PRODUCTION caller. Wiring it into the base Delaunay /
// lawson Delaunay test (tetgen-style cospherical tie-break) was prototyped to
// close the LShaped coplanar sliver ring, but on the borderline thin-wall
// models (Pipe / ThinWalledCylinder) the SoS-resolved base mesh destabilised
// the downstream recovery and they stopped conforming reliably on the
// conforming path (a real regression). Per the no-regression rule that wiring
// was reverted; the validated, self-tested predicate is kept as the foundation
// for a future attempt that also stabilises recovery. Allowed-dead until then.
/// Robust in-sphere test with Simulation of Simplicity tie-breaking.
///
/// Returns whether point `e` lies inside the circumsphere of the
/// oriented tetrahedron `(a, b, c, d)`; exact-arithmetic predicate with
/// SoS perturbation (keyed on the vertex indices `ia..ie`) so a
/// cospherical degeneracy resolves consistently instead of ambiguously.
/// Currently allowed-dead - kept as the tested foundation for a future
/// conforming-recovery attempt (see the note below).
#[allow(dead_code)]
pub fn in_sphere_sos(
    a: [f64; 3],
    ia: usize,
    b: [f64; 3],
    ib: usize,
    c: [f64; 3],
    ic: usize,
    d: [f64; 3],
    id: usize,
    e: [f64; 3],
    ie: usize,
) -> f64 {
    let s = in_sphere(a, b, c, d, e); // wrapped/negated exact insphere
    if s != 0.0 {
        return s;
    }
    // five (coord, gidx) pairs, sorted ASCENDING by gidx via swap-counting
    // bubble sort (matches tetgen.cxx:5165-5177 byte-for-byte).
    let mut pt: [([f64; 3], usize); 5] = [(a, ia), (b, ib), (c, ic), (d, id), (e, ie)];
    let mut swaps = 0usize;
    let mut n = 5usize;
    loop {
        let mut count = 0usize;
        n -= 1;
        for i in 0..n {
            if pt[i].1 > pt[i + 1].1 {
                pt.swap(i, i + 1);
                count += 1;
            }
        }
        swaps += count;
        if count == 0 {
            break;
        }
    }
    let odd = !swaps.is_multiple_of(2);
    // term A: drop the smallest-index point pt[0]. orient_3d is the WRAPPED
    // (negated) one, same convention as the in_sphere above.
    let mut a_ori = orient_3d(pt[1].0, pt[2].0, pt[3].0, pt[4].0);
    if a_ori != 0.0 {
        if odd {
            a_ori = -a_ori;
        }
        return a_ori;
    }
    // term B: drop the 2nd-smallest pt[1], with an INTRINSIC leading minus
    // applied BEFORE the parity flip.
    let mut b_ori = -orient_3d(pt[0].0, pt[2].0, pt[3].0, pt[4].0);
    if b_ori == 0.0 {
        // Two of five points geometrically coincide => duplicate vertex bug.
        // Do NOT extend the cascade / loosen (no-hide-errors). Mirror TetGen's
        // terminatetetgen(this, 2) at tetgen.cxx:5188.
        panic!("in_sphere_sos: degenerate term B == 0 => duplicate vertex (ids {ia},{ib},{ic},{id},{ie})");
    }
    if odd {
        b_ori = -b_ori;
    }
    b_ori
}

/// Simulation-of-Simplicity tie-break for the 4-point orient3d test (coplanar
/// tie-break). TetGen ships no standalone version; this is hand-derived from
/// the same Edelsbrunner-Mucke construction (orient3d degenerate => the 4x4 det
/// is 0 => the perturbation reduces to the leading non-vanishing 3x3/2D minor
/// over the index-sorted points). Same negated convention as `orient_3d`, built
/// on the wrapped `orient_3d` and the wrapped (negated) `orient2d`. Reached only
/// when `orient_3d` returns exactly 0.0. Deterministic by global vertex index.
//
// NOTE: no PRODUCTION caller (see `in_sphere_sos`). Routing the flip-decision
// predicates (lawson flip-type / 2-3 convexity / 3-2 straddle) through this to
// close the LShaped coplanar sliver ring was prototyped, but it shifted the
// thin-wall recovery (Pipe / ThinWalledCylinder) off the conforming path, so it
// was reverted under the no-regression rule. Kept (allowed-dead) as the tested
// foundation for a future attempt.
/// Robust 3-D orientation test with Simulation of Simplicity.
///
/// Sign of the orientation determinant of `(a, b, c, d)` (positive =
/// `d` above the CCW plane `abc`); exact predicate with SoS
/// tie-breaking keyed on the vertex indices so coplanar degeneracies
/// resolve consistently. Allowed-dead - see the in-sphere note.
#[allow(dead_code)]
pub fn orient_3d_sos(
    a: [f64; 3],
    ia: usize,
    b: [f64; 3],
    ib: usize,
    c: [f64; 3],
    ic: usize,
    d: [f64; 3],
    id: usize,
) -> f64 {
    let s = orient_3d(a, b, c, d);
    if s != 0.0 {
        return s;
    }
    let mut pt: [([f64; 3], usize); 4] = [(a, ia), (b, ib), (c, ic), (d, id)];
    let mut swaps = 0usize;
    let mut n = 4usize;
    loop {
        let mut count = 0usize;
        n -= 1;
        for i in 0..n {
            if pt[i].1 > pt[i + 1].1 {
                pt.swap(i, i + 1);
                count += 1;
            }
        }
        swaps += count;
        if count == 0 {
            break;
        }
    }
    let odd = !swaps.is_multiple_of(2);
    // orient2d projected onto a coordinate pair (cx,cy). Negate to stay in the
    // same wrapped convention as orient_3d.
    let o2 = |p: [f64; 3], q: [f64; 3], r: [f64; 3], cx: usize, cy: usize| -> f64 {
        -orient2d([p[cx], p[cy]], [q[cx], q[cy]], [r[cx], r[cy]])
    };
    for &(cx, cy) in &[(0usize, 1usize), (0, 2), (1, 2)] {
        // xy, xz, yz
        let mut t1 = o2(pt[1].0, pt[2].0, pt[3].0, cx, cy);
        if t1 != 0.0 {
            if odd {
                t1 = -t1;
            }
            return t1;
        }
        let mut t2 = -o2(pt[0].0, pt[2].0, pt[3].0, cx, cy);
        if t2 != 0.0 {
            if odd {
                t2 = -t2;
            }
            return t2;
        }
        let mut t3 = o2(pt[0].0, pt[1].0, pt[3].0, cx, cy);
        if t3 != 0.0 {
            if odd {
                t3 = -t3;
            }
            return t3;
        }
        let mut t4 = -o2(pt[0].0, pt[1].0, pt[2].0, cx, cy);
        if t4 != 0.0 {
            if odd {
                t4 = -t4;
            }
            return t4;
        }
    }
    // All three projections vanish => duplicate point. Do NOT loosen.
    panic!(
        "orient_3d_sos: all projections degenerate => duplicate vertex (ids {ia},{ib},{ic},{id})"
    );
}

/// Compute the volume of a tetrahedron (a, b, c, d).
/// Positive if (a, b, c, d) is positively oriented.
pub fn tet_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    orient_3d(a, b, c, d) / 6.0
}

/// Squared distance between two 3D points.
pub fn dist_sq_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

/// Distance between two 3D points.
#[allow(dead_code)]
pub fn dist_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    dist_sq_3d(a, b).sqrt()
}

/// Cross product of (b-a) x (c-a).
#[allow(dead_code)]
pub fn cross(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
}

/// Dot product of two 3D vectors.
#[allow(dead_code)]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute circumcenter of tetrahedron (a, b, c, d).
pub fn circumcenter_tet(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    // Translate so a is at origin
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ca = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let da = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];

    let ba_sq = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
    let ca_sq = ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2];
    let da_sq = da[0] * da[0] + da[1] * da[1] + da[2] * da[2];

    let cross_cd = [
        ca[1] * da[2] - ca[2] * da[1],
        ca[2] * da[0] - ca[0] * da[2],
        ca[0] * da[1] - ca[1] * da[0],
    ];
    let cross_db = [
        da[1] * ba[2] - da[2] * ba[1],
        da[2] * ba[0] - da[0] * ba[2],
        da[0] * ba[1] - da[1] * ba[0],
    ];
    let cross_bc = [
        ba[1] * ca[2] - ba[2] * ca[1],
        ba[2] * ca[0] - ba[0] * ca[2],
        ba[0] * ca[1] - ba[1] * ca[0],
    ];

    let denom = 2.0 * (ba[0] * cross_cd[0] + ba[1] * cross_cd[1] + ba[2] * cross_cd[2]);

    if denom.abs() < f64::EPSILON * 1e6 {
        // Degenerate: return centroid
        return [
            (a[0] + b[0] + c[0] + d[0]) / 4.0,
            (a[1] + b[1] + c[1] + d[1]) / 4.0,
            (a[2] + b[2] + c[2] + d[2]) / 4.0,
        ];
    }

    let o = [
        (ba_sq * cross_cd[0] + ca_sq * cross_db[0] + da_sq * cross_bc[0]) / denom,
        (ba_sq * cross_cd[1] + ca_sq * cross_db[1] + da_sq * cross_bc[1]) / denom,
        (ba_sq * cross_cd[2] + ca_sq * cross_db[2] + da_sq * cross_bc[2]) / denom,
    ];

    [o[0] + a[0], o[1] + a[1], o[2] + a[2]]
}

/// Do the INTERIORS of triangle `tri` and tetrahedron `tet` overlap?
///
/// Separating-axis test over the 23 candidate axes of two convex hulls: the
/// triangle normal, the tet's four face normals, and the 3x6 edge-edge cross
/// products. Two convex sets have disjoint interiors exactly when some axis
/// separates them, so "no separating axis" is the answer we want.
///
/// TOUCHING COUNTS AS DISJOINT, which is the whole point. A tet whose face lies
/// exactly on a conformal boundary shares that face with a boundary triangle and
/// needs no cut; the cheap predicates (`segment_crosses_boundary`,
/// `is_point_inside`) cannot tell that apart from a real crossing, so they report
/// every boundary-adjacent tet as cut. Measured with those, Cuboid and Sphere
/// come out with an "interior share" of exactly 1.0000 - i.e. every tet they
/// flag is wholly inside. Contact is therefore admitted as separation here, via a
/// tolerance relative to the coordinate scale.
pub fn tri_tet_interiors_overlap(tri: &[[f64; 3]; 3], tet: &[[f64; 3]; 4]) -> bool {
    const TET_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let cr = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };

    let mut axes: Vec<[f64; 3]> = Vec::with_capacity(23);
    axes.push(cr(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
    for f in TET_FACES {
        axes.push(cr(sub(tet[f[1]], tet[f[0]]), sub(tet[f[2]], tet[f[0]])));
    }
    let tri_edges = [
        sub(tri[1], tri[0]),
        sub(tri[2], tri[1]),
        sub(tri[0], tri[2]),
    ];
    for e in TET_EDGES {
        let ev = sub(tet[e[1]], tet[e[0]]);
        for te in tri_edges {
            axes.push(cr(te, ev));
        }
    }

    let mut scale = 0.0f64;
    for p in tri.iter().chain(tet.iter()) {
        for c in p {
            scale = scale.max(c.abs());
        }
    }
    let eps = 1e-12 * if scale > 0.0 { scale } else { 1.0 };

    for ax in axes {
        let n2 = ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2];
        if n2 <= 0.0 {
            continue; // degenerate axis (parallel edges / sliver) carries no info
        }
        let inv = 1.0 / n2.sqrt();
        let proj = |p: &[f64; 3]| (p[0] * ax[0] + p[1] * ax[1] + p[2] * ax[2]) * inv;
        let (mut amin, mut amax) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in tri.iter() {
            let v = proj(p);
            amin = amin.min(v);
            amax = amax.max(v);
        }
        let (mut bmin, mut bmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in tet.iter() {
            let v = proj(p);
            bmin = bmin.min(v);
            bmax = bmax.max(v);
        }
        if amax <= bmin + eps || bmax <= amin + eps {
            return false;
        }
    }
    true
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {

    #[test]
    fn tri_tet_overlap_detects_crossing_but_not_contact() {
        // Unit tet at the origin.
        let tet = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        // A big triangle slicing horizontally through the middle: interiors
        // genuinely overlap, so this is a tet a clip would have to cut.
        let slicing = [[-1.0, -1.0, 0.25], [3.0, -1.0, 0.25], [-1.0, 3.0, 0.25]];
        assert!(tri_tet_interiors_overlap(&slicing, &tet));

        // The tet's own base face. Shared exactly => CONTACT, not a cut. This is
        // the case the cheap predicates get wrong.
        let coincident = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(!tri_tet_interiors_overlap(&coincident, &tet));

        // Coplanar with the base plane but offset sideways: still only contact at
        // most, never interior overlap.
        let beside = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
        assert!(!tri_tet_interiors_overlap(&beside, &tet));

        // Touching a single vertex.
        let at_vertex = [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]];
        assert!(!tri_tet_interiors_overlap(&at_vertex, &tet));

        // Well clear of the tet.
        let far = [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]];
        assert!(!tri_tet_interiors_overlap(&far, &tet));

        // A small triangle wholly INSIDE the tet: interiors overlap.
        let inside = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1]];
        assert!(tri_tet_interiors_overlap(&inside, &tet));
    }

    use super::*;

    #[test]
    fn orient3d_positive() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        assert!(orient_3d(a, b, c, d) > 0.0);
    }

    #[test]
    fn orient3d_negative() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, -1.0];
        assert!(orient_3d(a, b, c, d) < 0.0);
    }

    #[test]
    fn tet_volume_unit() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let vol = tet_volume(a, b, c, d);
        assert!((vol - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn circumcenter_regular_tet() {
        // Regular tet centered at origin
        let s = 1.0;
        let a = [s, s, s];
        let b = [s, -s, -s];
        let c = [-s, s, -s];
        let d = [-s, -s, s];
        let cc = circumcenter_tet(a, b, c, d);
        assert!(cc[0].abs() < 1e-10);
        assert!(cc[1].abs() < 1e-10);
        assert!(cc[2].abs() < 1e-10);
    }

    #[test]
    fn insphere_inside() {
        let a = [0.0, 0.0, 0.0];
        let b = [4.0, 0.0, 0.0];
        let c = [0.0, 4.0, 0.0];
        let d = [0.0, 0.0, 4.0];
        let e = [1.0, 1.0, 1.0]; // inside circumsphere
        assert!(in_sphere(a, b, c, d, e) > 0.0);
    }

    #[test]
    fn insphere_outside() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let e = [10.0, 10.0, 10.0]; // far outside
        assert!(in_sphere(a, b, c, d, e) < 0.0);
    }

    // ---- Simulation-of-Simplicity (SoS) tie-break tests ----

    // Helper: orient a base tet positively (orient_3d > 0) by swapping the last
    // two coords+indices if needed. SoS's precondition is the base {a,b,c,d}
    // must be positively oriented.
    fn orient_base(mut p: [([f64; 3], usize); 4]) -> [([f64; 3], usize); 4] {
        if orient_3d(p[0].0, p[1].0, p[2].0, p[3].0) < 0.0 {
            p.swap(2, 3);
        }
        p
    }

    #[test]
    fn in_sphere_sos_cospherical_nonzero_and_consistent() {
        // Six points of a regular octahedron on the unit sphere - maximally
        // cospherical. Any 5 of them are cospherical => exact in_sphere == 0.
        // gidx is the array index; the SoS answer must be deterministic and
        // CONSISTENT across input permutations (the sort canonicalizes order,
        // parity corrects for it). Base re-oriented positive before each call.
        let v: [[f64; 3]; 5] = [
            [1.0, 0.0, 0.0],  // 0
            [0.0, 1.0, 0.0],  // 1
            [0.0, 0.0, 1.0],  // 2
            [-1.0, 0.0, 0.0], // 3
            [0.0, -1.0, 0.0], // 4
        ];
        // sanity: these are cospherical => float predicate is exactly zero
        assert_eq!(in_sphere(v[0], v[1], v[2], v[3], v[4]), 0.0);

        // Build the base tet {0,1,2,3} positively oriented, e = vertex 4.
        let base = orient_base([(v[0], 0), (v[1], 1), (v[2], 2), (v[3], 3)]);
        let s0 = in_sphere_sos(
            base[0].0, base[0].1, base[1].0, base[1].1, base[2].0, base[2].1, base[3].0, base[3].1,
            v[4], 4,
        );
        assert!(s0 != 0.0, "SoS must resolve the cospherical tie to nonzero");

        // Permute the first four args (swap 0 and 1), re-orient base positive,
        // keep the same five-point set and same e. The inside/outside answer
        // must be IDENTICAL (deterministic, permutation-consistent).
        let base2 = orient_base([(v[1], 1), (v[0], 0), (v[2], 2), (v[3], 3)]);
        let s1 = in_sphere_sos(
            base2[0].0, base2[0].1, base2[1].0, base2[1].1, base2[2].0, base2[2].1, base2[3].0,
            base2[3].1, v[4], 4,
        );
        assert_eq!(
            s0.signum(),
            s1.signum(),
            "SoS in_sphere must be permutation-consistent (same 5 pts + ids => same side)"
        );

        // A different permutation of the base ({2,0,1,3}) - still same answer.
        let base3 = orient_base([(v[2], 2), (v[0], 0), (v[1], 1), (v[3], 3)]);
        let s2 = in_sphere_sos(
            base3[0].0, base3[0].1, base3[1].0, base3[1].1, base3[2].0, base3[2].1, base3[3].0,
            base3[3].1, v[4], 4,
        );
        assert_eq!(s0.signum(), s2.signum());

        // Determinism: identical call => identical value (bitwise).
        let s0b = in_sphere_sos(
            base[0].0, base[0].1, base[1].0, base[1].1, base[2].0, base[2].1, base[3].0, base[3].1,
            v[4], 4,
        );
        assert_eq!(s0, s0b);
    }

    #[test]
    fn orient_3d_sos_coplanar_nonzero_and_parity_flips() {
        // Four coplanar points (z=0 plane) => exact orient_3d == 0.0.
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 0.0];
        let p3 = [1.0, 1.0, 0.0];
        assert_eq!(
            orient_3d(p0, p1, p2, p3),
            0.0,
            "coplanar => float orient is 0"
        );

        let s = orient_3d_sos(p0, 0, p1, 1, p2, 2, p3, 3);
        assert!(s != 0.0, "SoS must resolve the coplanar tie to nonzero");

        // A single argument swap is one transposition => parity flips => the
        // SoS sign must invert (anti-symmetry).
        let s_swapped = orient_3d_sos(p1, 1, p0, 0, p2, 2, p3, 3);
        assert!(s_swapped != 0.0);
        assert_eq!(
            s.signum(),
            -s_swapped.signum(),
            "orient_3d_sos must flip sign under a single arg swap (parity)"
        );

        // Two swaps (even parity) => sign restored.
        let s_double = orient_3d_sos(p1, 1, p0, 0, p3, 3, p2, 2);
        assert_eq!(s.signum(), s_double.signum());

        // Determinism: identical call => identical bits.
        assert_eq!(s, orient_3d_sos(p0, 0, p1, 1, p2, 2, p3, 3));
    }

    #[test]
    fn sos_sign_agrees_with_float_on_nondegenerate() {
        // Non-degenerate base tet (positively oriented) and points clearly
        // inside / outside its circumsphere. SoS must return EXACTLY the float
        // value (it short-circuits when the float predicate is nonzero).
        let a = [0.0, 0.0, 0.0];
        let b = [4.0, 0.0, 0.0];
        let c = [0.0, 4.0, 0.0];
        let d = [0.0, 0.0, 4.0];
        assert!(orient_3d(a, b, c, d) > 0.0);

        let inside = [1.0, 1.0, 1.0];
        let f_in = in_sphere(a, b, c, d, inside);
        let s_in = in_sphere_sos(a, 0, b, 1, c, 2, d, 3, inside, 4);
        assert!(f_in > 0.0);
        assert_eq!(f_in, s_in, "SoS must equal float in_sphere when nonzero");

        let outside = [10.0, 10.0, 10.0];
        let f_out = in_sphere(a, b, c, d, outside);
        let s_out = in_sphere_sos(a, 0, b, 1, c, 2, d, 3, outside, 4);
        assert!(f_out < 0.0);
        assert_eq!(f_out, s_out);

        // orient_3d_sos likewise agrees with orient_3d on non-coplanar input.
        let above = [0.1, 0.1, 1.0];
        let f_ori = orient_3d(a, b, c, above);
        let s_ori = orient_3d_sos(a, 0, b, 1, c, 2, above, 3);
        assert!(f_ori != 0.0);
        assert_eq!(
            f_ori, s_ori,
            "orient_3d_sos must equal orient_3d when nonzero"
        );

        let below = [0.1, 0.1, -1.0];
        let f_ori2 = orient_3d(a, b, c, below);
        let s_ori2 = orient_3d_sos(a, 0, b, 1, c, 2, below, 3);
        assert_eq!(f_ori2.signum(), -f_ori.signum());
        assert_eq!(f_ori2, s_ori2);
    }
}
