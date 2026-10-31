//! Low-level intersection routines: Moller-Trumbore ray-triangle,
//! barycentric tet containment, and vector math helpers.

/// Near-coplanar tolerance for Plucker/barycentric tests.
/// 20 * f64::EPSILON ~ 4.44e-15  (from XDG constants.h)
pub const PLUCKER_ZERO_TOL: f64 = 20.0 * f64::EPSILON;

/// Nudge distance for stepping past a surface after crossing.
pub const SURFACE_BUMP: f64 = 1e-10;

/// Minimum positive ray parameter to accept (avoids self-intersection).
pub const RAY_MIN_T: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
pub fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
pub fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = length(v);
    if len < 1e-30 {
        return [0.0, 0.0, 0.0];
    }
    scale(v, 1.0 / len)
}

/// Distance between two points.
#[inline]
pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    length(sub(a, b))
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection
// ---------------------------------------------------------------------------

/// Moller-Trumbore ray-triangle intersection.
///
/// Returns `Some(t)` where `t > RAY_MIN_T` is the distance along the ray,
/// or `None` if no hit.
///
/// The hit point is `origin + t * direction`.
///
/// When the `simd` feature is enabled, this delegates to a runtime-dispatched
/// version compiled with the best available ISA extensions (AVX2+FMA, SSE4.1).
#[cfg(feature = "simd")]
#[inline]
pub fn ray_triangle_intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<f64> {
    crate::accel::simd::ray_triangle_intersect(origin, direction, v0, v1, v2)
}

/// Moller-Trumbore ray-triangle intersection (scalar fallback).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn ray_triangle_intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<f64> {
    let edge1 = sub(v1, v0);
    let edge2 = sub(v2, v0);
    let h = cross(direction, edge2);
    let a = dot(edge1, h);

    if a.abs() < PLUCKER_ZERO_TOL {
        return None; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = sub(origin, v0);
    let u = f * dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = cross(s, edge1);
    let v = f * dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot(edge2, q);
    if t > RAY_MIN_T {
        Some(t)
    } else {
        None
    }
}

/// Ray-triangle intersection that also returns the outward normal.
///
/// Returns `Some((t, normal))` where normal = normalize(edge1 × edge2).
pub fn ray_triangle_intersect_with_normal(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<(f64, [f64; 3])> {
    let t = ray_triangle_intersect(origin, direction, v0, v1, v2)?;
    let normal = triangle_normal(v0, v1, v2);
    Some((t, normal))
}

/// Compute the outward face normal of a triangle (not normalized to unit length).
#[inline]
pub fn triangle_normal_unnormalized(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> [f64; 3] {
    cross(sub(v1, v0), sub(v2, v0))
}

/// Compute the unit face normal of a triangle.
#[inline]
pub fn triangle_normal(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> [f64; 3] {
    normalize(triangle_normal_unnormalized(v0, v1, v2))
}

// ---------------------------------------------------------------------------
// Barycentric tet containment test (from XDG)
// ---------------------------------------------------------------------------

/// Test if a point is inside a tetrahedron using barycentric coordinates.
///
/// Solves T * lambda = (point - v0) where T = [v1-v0, v2-v0, v3-v0].
/// Returns `true` if all four barycentric coordinates are in
/// `[-PLUCKER_ZERO_TOL, 1 + PLUCKER_ZERO_TOL]`.
///
/// When the `simd` feature is enabled, this delegates to a runtime-dispatched
/// version compiled with the best available ISA extensions (AVX2+FMA, SSE4.1).
#[cfg(feature = "simd")]
pub fn point_in_tet(
    point: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> bool {
    crate::accel::simd::point_in_tet(point, v0, v1, v2, v3)
}

/// Test if a point is inside a tetrahedron (scalar fallback).
#[cfg(not(feature = "simd"))]
pub fn point_in_tet(
    point: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> bool {
    let e0 = sub(v1, v0);
    let e1 = sub(v2, v0);
    let e2 = sub(v3, v0);
    let rhs = sub(point, v0);

    // Solve 3x3 system using Cramer's rule
    let det = e0[0] * (e1[1] * e2[2] - e1[2] * e2[1]) - e0[1] * (e1[0] * e2[2] - e1[2] * e2[0])
        + e0[2] * (e1[0] * e2[1] - e1[1] * e2[0]);

    if det.abs() < 1e-30 {
        return false; // Degenerate tet
    }

    let inv_det = 1.0 / det;

    let lambda1 = inv_det
        * (rhs[0] * (e1[1] * e2[2] - e1[2] * e2[1]) - rhs[1] * (e1[0] * e2[2] - e1[2] * e2[0])
            + rhs[2] * (e1[0] * e2[1] - e1[1] * e2[0]));

    let lambda2 = inv_det
        * (e0[0] * (rhs[1] * e2[2] - rhs[2] * e2[1]) - e0[1] * (rhs[0] * e2[2] - rhs[2] * e2[0])
            + e0[2] * (rhs[0] * e2[1] - rhs[1] * e2[0]));

    let lambda3 = inv_det
        * (e0[0] * (e1[1] * rhs[2] - e1[2] * rhs[1]) - e0[1] * (e1[0] * rhs[2] - e1[2] * rhs[0])
            + e0[2] * (e1[0] * rhs[1] - e1[1] * rhs[0]));

    let lambda0 = 1.0 - lambda1 - lambda2 - lambda3;

    let tol = PLUCKER_ZERO_TOL;
    lambda0 >= -tol
        && lambda0 <= 1.0 + tol
        && lambda1 >= -tol
        && lambda1 <= 1.0 + tol
        && lambda2 >= -tol
        && lambda2 <= 1.0 + tol
        && lambda3 >= -tol
        && lambda3 <= 1.0 + tol
}

/// Ray-segment vs AABB slab test over `t` in `[0, t_max]`.
/// `bb` is `[xmin, ymin, zmin, xmax, ymax, zmax]`.
pub fn segment_hits_aabb(origin: [f64; 3], direction: [f64; 3], t_max: f64, bb: &[f64; 6]) -> bool {
    let mut t0 = 0.0_f64;
    let mut t1 = t_max;
    for i in 0..3 {
        if direction[i].abs() < 1e-300 {
            if origin[i] < bb[i] || origin[i] > bb[i + 3] {
                return false;
            }
            continue;
        }
        let inv = 1.0 / direction[i];
        let mut a = (bb[i] - origin[i]) * inv;
        let mut b = (bb[i + 3] - origin[i]) * inv;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return false;
        }
    }
    true
}

/// Triangle area: 0.5 * ||(v1 - v0) × (v2 - v0)||
#[inline]
pub fn triangle_area(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> f64 {
    0.5 * length(cross(sub(v1, v0), sub(v2, v0)))
}

/// Signed volume contribution of a triangle for the divergence theorem.
/// `v0 . ((v1 - v0) × (v2 - v0))`
#[inline]
pub fn triangle_volume_contribution(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> f64 {
    dot(v0, cross(sub(v1, v0), sub(v2, v0)))
}

/// Tetrahedron volume: |det([v1-v0, v2-v0, v3-v0])| / 6.0
#[inline]
pub fn tet_volume(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> f64 {
    signed_tet_volume(v0, v1, v2, v3).abs()
}

/// Signed tetrahedron volume: det([v1-v0, v2-v0, v3-v0]) / 6.0.
///
/// Positive when the vertices are in right-handed (positive) order, which is
/// the orientation [`crate::mesh::topology::TET_FACE_VERTICES`] assumes when
/// it produces outward-pointing face normals.
#[inline]
pub fn signed_tet_volume(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> f64 {
    let e0 = sub(v1, v0);
    let e1 = sub(v2, v0);
    let e2 = sub(v3, v0);
    dot(e0, cross(e1, e2)) / 6.0
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_triangle_hit() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let origin = [0.25, 0.25, -1.0];
        let direction = [0.0, 0.0, 1.0];
        let t = ray_triangle_intersect(origin, direction, v0, v1, v2);
        assert!(t.is_some());
        assert!((t.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ray_triangle_miss() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // Ray misses triangle
        let t = ray_triangle_intersect([2.0, 2.0, -1.0], [0.0, 0.0, 1.0], v0, v1, v2);
        assert!(t.is_none());
    }

    #[test]
    fn test_point_in_tet() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        // Center of tet
        assert!(point_in_tet([0.1, 0.1, 0.1], v0, v1, v2, v3));
        // Outside
        assert!(!point_in_tet([1.0, 1.0, 1.0], v0, v1, v2, v3));
    }

    #[test]
    fn test_triangle_area() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let area = triangle_area(v0, v1, v2);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tet_volume() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        let vol = tet_volume(v0, v1, v2, v3);
        assert!((vol - 1.0 / 6.0).abs() < 1e-10);
    }
}
