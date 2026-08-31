//! SIMD-accelerated intersection kernels via `multiversion` multiversioning.
//!
//! Each hot-path function is compiled for multiple ISA extensions:
//! - **AVX-512F+FMA**: 512-bit vectors for maximum throughput on supported CPUs
//! - **AVX2+FMA**: fused multiply-add for better throughput and accuracy
//! - **SSE4.1**: additional rounding/blending over baseline SSE2
//! - **Scalar/aarch64**: baseline fallback (works on all architectures)
//!
//! Runtime CPU detection selects the best available variant. The `multiversion`
//! crate handles all dispatch mechanics.

use multiversion::multiversion;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PLUCKER_ZERO_TOL: f64 = 20.0 * f64::EPSILON;
const RAY_MIN_T: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Vector helpers -- #[inline(always)] so they propagate into target-feature
// compiled functions, allowing the compiler to use FMA, etc.
// ---------------------------------------------------------------------------

#[inline(always)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// ===========================================================================
// Ray-Triangle intersection (Moller-Trumbore)
// ===========================================================================

/// SIMD-dispatched Moller-Trumbore ray-triangle intersection.
///
/// Identical semantics to [`crate::query::intersect::ray_triangle_intersect`]
/// but compiled with the best available ISA extensions.
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn ray_triangle_intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<f64> {
    let edge1 = sub3(v1, v0);
    let edge2 = sub3(v2, v0);
    let h = cross3(direction, edge2);
    let a = dot3(edge1, h);

    if a.abs() < PLUCKER_ZERO_TOL {
        return None; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = sub3(origin, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = cross3(s, edge1);
    let v = f * dot3(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot3(edge2, q);
    if t > RAY_MIN_T {
        Some(t)
    } else {
        None
    }
}

// ===========================================================================
// Ray-AABB slab test
// ===========================================================================

/// SIMD-dispatched ray-AABB slab test.
///
/// Identical semantics to [`crate::accel::bvh::Aabb::ray_intersect`].
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn ray_aabb_intersect(
    origin: [f64; 3],
    inv_dir: [f64; 3],
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
) -> Option<(f64, f64)> {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;

    for i in 0..3 {
        let t1 = (aabb_min[i] - origin[i]) * inv_dir[i];
        let t2 = (aabb_max[i] - origin[i]) * inv_dir[i];
        let ta = t1.min(t2);
        let tb = t1.max(t2);
        tmin = tmin.max(ta);
        tmax = tmax.min(tb);
    }

    if tmax >= tmin.max(0.0) {
        Some((tmin, tmax))
    } else {
        None
    }
}

// ===========================================================================
// Point-in-tet (barycentric containment via Cramer's rule)
// ===========================================================================

/// SIMD-dispatched barycentric tet containment test.
///
/// Identical semantics to [`crate::query::intersect::point_in_tet`].
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn point_in_tet(
    point: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> bool {
    let e0 = sub3(v1, v0);
    let e1 = sub3(v2, v0);
    let e2 = sub3(v3, v0);
    let rhs = sub3(point, v0);

    // 3x3 determinant via cofactor expansion
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

// ===========================================================================
// 4-wide Ray-Triangle intersection (batched Moller-Trumbore with precomputed edges)
// ===========================================================================

/// SIMD-dispatched 4-wide ray-triangle intersection with precomputed edges.
///
/// Tests a single ray against 4 triangles stored in SoA layout:
/// `v0[axis][tri]`, `edge1[axis][tri]`, `edge2[axis][tri]`.
///
/// Returns `(t_per_tri, hit_bitmask)` where bit `i` is set if triangle `i`
/// was hit at distance `t[i] >= min_t`.
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn ray_tri4_intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: &[[f64; 4]; 3],
    edge1: &[[f64; 4]; 3],
    edge2: &[[f64; 4]; 3],
    min_t: f64,
) -> ([f64; 4], u8) {
    // Inclusive barycentric slack. Large enough that a ray passing
    // exactly through a shared triangle edge cannot slip between both
    // facets' strict bounds (a missed crossing corrupts the adjacency
    // walk), small enough that the epsilon band beyond an edge does not
    // accept phantom crossings on the WRONG facet of a dihedral fold.
    // 1e-6 produced such phantom crossings at grazing incidence on fine
    // curved meshes (~2e-5 per history on a 1.5M-triangle torus shell
    // model, caught by the crossing verification); 1e-9 shows zero in
    // 1M histories with identical physics.
    const BARY_EPS: f64 = 1e-9;

    // h = cross(direction, edge2)
    let mut h = [[0.0; 4]; 3];
    for i in 0..4 {
        h[0][i] = direction[1] * edge2[2][i] - direction[2] * edge2[1][i];
        h[1][i] = direction[2] * edge2[0][i] - direction[0] * edge2[2][i];
        h[2][i] = direction[0] * edge2[1][i] - direction[1] * edge2[0][i];
    }

    // a = dot(edge1, h)
    let mut a = [0.0; 4];
    for i in 0..4 {
        a[i] = edge1[0][i] * h[0][i] + edge1[1][i] * h[1][i] + edge1[2][i] * h[2][i];
    }

    // f = 1/a (degenerate cases produce inf/NaN which fail bounds checks)
    let mut f = [0.0; 4];
    for i in 0..4 {
        f[i] = 1.0 / a[i];
    }

    // s = origin - v0
    let mut s = [[0.0; 4]; 3];
    for i in 0..4 {
        s[0][i] = origin[0] - v0[0][i];
        s[1][i] = origin[1] - v0[1][i];
        s[2][i] = origin[2] - v0[2][i];
    }

    // u = f * dot(s, h)
    let mut u = [0.0; 4];
    for i in 0..4 {
        u[i] = f[i] * (s[0][i] * h[0][i] + s[1][i] * h[1][i] + s[2][i] * h[2][i]);
    }

    // q = cross(s, edge1)
    let mut q = [[0.0; 4]; 3];
    for i in 0..4 {
        q[0][i] = s[1][i] * edge1[2][i] - s[2][i] * edge1[1][i];
        q[1][i] = s[2][i] * edge1[0][i] - s[0][i] * edge1[2][i];
        q[2][i] = s[0][i] * edge1[1][i] - s[1][i] * edge1[0][i];
    }

    // v = f * dot(direction, q)
    let mut v = [0.0; 4];
    for i in 0..4 {
        v[i] = f[i] * (direction[0] * q[0][i] + direction[1] * q[1][i] + direction[2] * q[2][i]);
    }

    // t = f * dot(edge2, q)
    let mut t = [0.0; 4];
    for i in 0..4 {
        t[i] = f[i] * (edge2[0][i] * q[0][i] + edge2[1][i] * q[1][i] + edge2[2][i] * q[2][i]);
    }

    // Build hit mask (branchless: all conditions checked per lane)
    let mut mask: u8 = 0;
    for i in 0..4 {
        if a[i].abs() >= PLUCKER_ZERO_TOL
            && u[i] >= -BARY_EPS
            && u[i] <= 1.0 + BARY_EPS
            && v[i] >= -BARY_EPS
            && u[i] + v[i] <= 1.0 + BARY_EPS
            && t[i] >= min_t
        {
            mask |= 1 << i;
        }
    }

    (t, mask)
}

// ===========================================================================
// 4-wide Ray-AABB slab test (for BVH4 traversal)
// ===========================================================================

/// SIMD-dispatched 4-wide ray-AABB slab test for BVH4 traversal.
///
/// Tests a single ray against 4 child AABBs stored in SoA layout.
/// Returns `(tmin_per_child, hit_bitmask)` where hit_bitmask bit `i`
/// is set if child `i` was hit.
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
#[allow(clippy::too_many_arguments)]
pub fn ray_aabb4_intersect(
    origin: [f64; 3],
    inv_dir: [f64; 3],
    min_x: &[f64; 4],
    min_y: &[f64; 4],
    min_z: &[f64; 4],
    max_x: &[f64; 4],
    max_y: &[f64; 4],
    max_z: &[f64; 4],
) -> ([f64; 4], u8) {
    let mins = [min_x, min_y, min_z];
    let maxs = [max_x, max_y, max_z];
    let mut tmin = [f64::NEG_INFINITY; 4];
    let mut tmax = [f64::INFINITY; 4];

    for axis in 0..3 {
        for i in 0..4 {
            let t1 = (mins[axis][i] - origin[axis]) * inv_dir[axis];
            let t2 = (maxs[axis][i] - origin[axis]) * inv_dir[axis];
            let ta = t1.min(t2);
            let tb = t1.max(t2);
            tmin[i] = tmin[i].max(ta);
            tmax[i] = tmax[i].min(tb);
        }
    }

    let mut mask: u8 = 0;
    for i in 0..4 {
        if tmax[i] >= tmin[i].max(0.0) {
            mask |= 1 << i;
        }
    }
    (tmin, mask)
}

// ===========================================================================
// 4-wide Point-AABB containment test (for BVH4 traversal)
// ===========================================================================

/// SIMD-dispatched 4-wide point-AABB containment test for BVH4 traversal.
///
/// Returns a bitmask where bit `i` is set if the point is inside child `i`.
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn point_aabb4_contains(
    point: [f64; 3],
    min_x: &[f64; 4],
    min_y: &[f64; 4],
    min_z: &[f64; 4],
    max_x: &[f64; 4],
    max_y: &[f64; 4],
    max_z: &[f64; 4],
) -> u8 {
    let mins = [min_x, min_y, min_z];
    let maxs = [max_x, max_y, max_z];
    let mut mask: u8 = 0xF; // start with all bits set
    for axis in 0..3 {
        for i in 0..4 {
            if point[axis] < mins[axis][i] || point[axis] > maxs[axis][i] {
                mask &= !(1 << i);
            }
        }
    }
    mask
}

// ===========================================================================
// 4-wide distance-to-AABB (for BVH4 closest-point traversal)
// ===========================================================================

/// SIMD-dispatched 4-wide squared distance from point to AABB.
///
/// Returns squared distance for each of 4 children (0 if point is inside).
#[multiversion(targets("x86_64+avx512f+fma", "x86_64+avx2+fma", "x86_64+sse4.1",))]
pub fn dist_aabb4_sq(
    point: [f64; 3],
    min_x: &[f64; 4],
    min_y: &[f64; 4],
    min_z: &[f64; 4],
    max_x: &[f64; 4],
    max_y: &[f64; 4],
    max_z: &[f64; 4],
) -> [f64; 4] {
    let mins = [min_x, min_y, min_z];
    let maxs = [max_x, max_y, max_z];
    let mut dist_sq = [0.0f64; 4];
    for axis in 0..3 {
        for i in 0..4 {
            if point[axis] < mins[axis][i] {
                let d = mins[axis][i] - point[axis];
                dist_sq[i] += d * d;
            } else if point[axis] > maxs[axis][i] {
                let d = point[axis] - maxs[axis][i];
                dist_sq[i] += d * d;
            }
        }
    }
    dist_sq
}

// ===========================================================================
// CPU detection info (for diagnostics)
// ===========================================================================

/// SIMD capability summary for diagnostics.
pub struct SimdInfo {
    /// What the CPU hardware supports (runtime detection).
    pub cpu_supports: &'static str,
    /// What the binary was compiled with (compile-time cfg).
    pub compiled_with: &'static str,
    /// What dispatched kernels will actually use at runtime.
    pub dispatch_uses: &'static str,
    /// True when `compiled_with` tier < `cpu_supports` tier -- non-dispatched
    /// code is leaving performance on the table.
    pub upgrade_available: bool,
}

/// Numeric tier for comparing SIMD levels.
fn compiled_tier() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "avx512f") && cfg!(target_feature = "fma") {
            return 4;
        }
        if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
            return 3;
        }
        if cfg!(target_feature = "sse4.1") {
            return 2;
        }
        return 1; // SSE2 baseline
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 3; // NEON is always available
    }
    #[allow(unreachable_code)]
    0
}

fn compiled_label() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "avx512f") && cfg!(target_feature = "fma") {
            return "AVX-512F+FMA";
        }
        if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
            return "AVX2+FMA";
        }
        if cfg!(target_feature = "sse4.1") {
            return "SSE4.1";
        }
        return "SSE2 (baseline)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON (baseline aarch64)";
    }
    #[allow(unreachable_code)]
    "scalar"
}

fn cpu_supports_label() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            return "AVX-512F+FMA";
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2+FMA";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
        return "SSE2 (baseline x86_64)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON (baseline aarch64)";
    }
    #[allow(unreachable_code)]
    "scalar"
}

fn cpu_tier() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            return 4;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return 3;
        }
        if is_x86_feature_detected!("sse4.1") {
            return 2;
        }
        return 1;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 3;
    }
    #[allow(unreachable_code)]
    0
}

/// Return a full SIMD diagnostics snapshot.
pub fn simd_info() -> SimdInfo {
    let cpu = cpu_supports_label();
    let dispatch = cpu; // dispatched kernels always pick the best
    let compiled = compiled_label();
    SimdInfo {
        cpu_supports: cpu,
        compiled_with: compiled,
        dispatch_uses: dispatch,
        upgrade_available: compiled_tier() < cpu_tier(),
    }
}

/// Return a description of the SIMD level in use for intersection kernels.
pub fn simd_level() -> &'static str {
    simd_info().dispatch_uses
}

/// Print a user-friendly SIMD diagnostics summary to stderr.
pub fn print_simd_info() {
    let tier = cpu_tier();

    #[cfg(target_arch = "x86_64")]
    {
        let ranking = match tier {
            4 => "the fastest option \u{1F389}",
            3 => "the 2nd fastest option",
            2 => "the 3rd fastest option",
            _ => "the baseline (slowest) option",
        };
        eprintln!("YAMT SIMD: supports (fastest to slowest) AVX-512F+FMA, AVX2+FMA, SSE4.1, SSE2");
        eprintln!(
            "  Your CPU supports {}, which is {}.",
            cpu_supports_label(),
            ranking,
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = tier;
        eprintln!("YAMT SIMD: using NEON (aarch64 baseline)");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = tier;
        eprintln!("YAMT SIMD: using scalar fallback");
    }
}

// ===========================================================================
// Tests
// ===========================================================================

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the phantom edge-band crossing (issue #256 residual).
    ///
    /// Two facets share the edge y=0 with a dihedral fold: F1 lies in the
    /// z=0 plane (y >= 0 side), F2 dips away on the y < 0 side. A grazing
    /// ray that crosses the surface on F2's side, a hair past the shared
    /// edge, also intersects F1's PLANE just outside F1's true extent.
    /// With a loose barycentric slack (the former 1e-6) the F1 hit is
    /// accepted: a phantom crossing far from the real surface point that
    /// desynchronizes adjacency tracking. The 4-wide intersector must
    /// reject it and report only the true F2 crossing.
    #[test]
    fn no_phantom_hit_in_edge_band_of_folded_neighbour() {
        // F1: (0,0,0) (4,0,0) (0,4,0)  in z = 0
        // F2: (4,0,0) (0,0,0) (2,-4,-0.008)  shallow fold across y = 0
        let v0 = [
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let e1 = [
            [4.0, -4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let e2 = [
            [0.0, -2.0, 0.0, 0.0],
            [4.0, -4.0, 0.0, 0.0],
            [0.0, -0.008, 0.0, 0.0],
        ];

        // Grazing ray in the x = 2 plane descending slowly: crosses z = 0
        // at y = -4e-6 (barycentric ~1e-6 beyond F1's edge, inside the old
        // slack), and crosses F2 (surface z = 0.002 * y for y < 0) later.
        let y_cross = -4.0e-6_f64;
        let origin = [2.0, 1.0, 0.01];
        let dz = -0.01 / (1.0 + y_cross.abs());
        let dir_raw = [0.0, -1.0, dz];
        let n = (dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]).sqrt();
        let direction = [0.0, dir_raw[1] / n, dir_raw[2] / n];

        let (t, mask) = ray_tri4_intersect(origin, direction, &v0, &e1, &e2, 1e-10);
        // F1 (lane 0) must NOT be hit: the plane crossing is beyond its
        // edge. F2 (lane 1) is the true crossing.
        assert_eq!(
            mask & 1,
            0,
            "phantom hit accepted on F1 in the edge band: t={:?}",
            t
        );
        assert_ne!(mask & 2, 0, "true F2 crossing missed: mask={mask:02b}");

        // And a ray exactly through the shared edge must still hit at
        // least one facet (no gap between strict bounds).
        let origin_edge = [2.0, 1.0, 0.0100000001];
        let dz_edge = -0.01;
        let n2 = (1.0f64 + dz_edge * dz_edge).sqrt();
        let dir_edge = [0.0, -1.0 / n2, dz_edge / n2];
        let (_t2, mask2) = ray_tri4_intersect(origin_edge, dir_edge, &v0, &e1, &e2, 1e-10);
        assert_ne!(
            mask2 & 3,
            0,
            "ray through the shared edge missed both facets"
        );
    }

    #[test]
    fn test_simd_level_reports_something() {
        let level = simd_level();
        assert!(!level.is_empty());
        println!("SIMD level: {level}");
    }

    // --- Ray-Triangle tests (must match scalar intersect.rs results) --------

    #[test]
    fn test_simd_ray_tri_hit() {
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
    fn test_simd_ray_tri_miss() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let t = ray_triangle_intersect([2.0, 2.0, -1.0], [0.0, 0.0, 1.0], v0, v1, v2);
        assert!(t.is_none());
    }

    #[test]
    fn test_simd_ray_tri_parallel() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        // Ray in the plane of the triangle
        let t = ray_triangle_intersect([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], v0, v1, v2);
        assert!(t.is_none());
    }

    #[test]
    fn test_simd_ray_tri_multiple_cases() {
        let cases: Vec<([f64; 3], [f64; 3])> = vec![
            ([0.25, 0.25, -1.0], [0.0, 0.0, 1.0]),
            ([0.1, 0.1, 5.0], [0.0, 0.0, -1.0]),
            ([2.0, 2.0, -1.0], [0.0, 0.0, 1.0]),
            ([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        for (origin, direction) in &cases {
            let result = ray_triangle_intersect(*origin, *direction, v0, v1, v2);
            // Just ensure it doesn't panic; correctness verified by other tests
            let _ = result;
        }
    }

    // --- Ray-AABB tests -----------------------------------------------------

    #[test]
    fn test_simd_ray_aabb_hit() {
        let hit = ray_aabb_intersect(
            [-1.0, 0.5, 0.5],
            [1.0, f64::INFINITY, f64::INFINITY], // inv_dir for [1,0,0]
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_some());
        let (tmin, _) = hit.unwrap();
        assert!((tmin - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_ray_aabb_miss() {
        let miss = ray_aabb_intersect(
            [-1.0, 0.5, 0.5],
            [-1.0, f64::INFINITY, f64::INFINITY],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert!(miss.is_none());
    }

    #[test]
    fn test_simd_ray_aabb_multiple_cases() {
        let cases: Vec<([f64; 3], [f64; 3])> = vec![
            ([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]),
            ([0.5, -1.0, 0.5], [0.0, 1.0, 0.0]),
            ([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]),
            ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            ([2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]),
        ];
        let bmin = [0.0, 0.0, 0.0];
        let bmax = [1.0, 1.0, 1.0];

        for (origin, direction) in &cases {
            let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];
            let _ = ray_aabb_intersect(*origin, inv_dir, bmin, bmax);
        }
    }

    // --- Point-in-tet tests -------------------------------------------------

    #[test]
    fn test_simd_point_in_tet_inside() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        assert!(point_in_tet([0.1, 0.1, 0.1], v0, v1, v2, v3));
    }

    #[test]
    fn test_simd_point_in_tet_outside() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        assert!(!point_in_tet([1.0, 1.0, 1.0], v0, v1, v2, v3));
    }

    // --- 4-wide Ray-Triangle tests --------------------------------------------

    #[test]
    fn test_ray_tri4_basic() {
        // 4 triangles: 3 hittable at various distances, 1 miss (degenerate)
        let tris: [([f64; 3], [f64; 3], [f64; 3]); 4] = [
            // tri 0: z=0 plane, hit at t=1 from (0.25, 0.25, -1)
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            // tri 1: z=2 plane, hit at t=3
            ([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]),
            // tri 2: z=5 plane, hit at t=6
            ([0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]),
            // tri 3: degenerate (zero edge1)
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ];

        // Convert to SoA with precomputed edges
        let mut v0 = [[0.0; 4]; 3];
        let mut e1 = [[0.0; 4]; 3];
        let mut e2 = [[0.0; 4]; 3];
        for (k, (p0, p1, p2)) in tris.iter().enumerate() {
            for axis in 0..3 {
                v0[axis][k] = p0[axis];
                e1[axis][k] = p1[axis] - p0[axis];
                e2[axis][k] = p2[axis] - p0[axis];
            }
        }

        let origin = [0.25, 0.25, -1.0];
        let direction = [0.0, 0.0, 1.0];
        let min_t = 1e-10;

        let (t4, mask) = ray_tri4_intersect(origin, direction, &v0, &e1, &e2, min_t);

        // tri 0,1,2 hit; tri 3 miss
        assert_eq!(mask, 0b0111);
        assert!((t4[0] - 1.0).abs() < 1e-12);
        assert!((t4[1] - 3.0).abs() < 1e-12);
        assert!((t4[2] - 6.0).abs() < 1e-12);
    }

    // --- 4-wide Ray-AABB tests -----------------------------------------------

    #[test]
    fn test_ray_aabb4_basic() {
        let boxes: [([f64; 3], [f64; 3]); 4] = [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
            ([0.0, 2.0, 0.0], [1.0, 3.0, 1.0]),
            ([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
        ];
        let min_x = [boxes[0].0[0], boxes[1].0[0], boxes[2].0[0], boxes[3].0[0]];
        let min_y = [boxes[0].0[1], boxes[1].0[1], boxes[2].0[1], boxes[3].0[1]];
        let min_z = [boxes[0].0[2], boxes[1].0[2], boxes[2].0[2], boxes[3].0[2]];
        let max_x = [boxes[0].1[0], boxes[1].1[0], boxes[2].1[0], boxes[3].1[0]];
        let max_y = [boxes[0].1[1], boxes[1].1[1], boxes[2].1[1], boxes[3].1[1]];
        let max_z = [boxes[0].1[2], boxes[1].1[2], boxes[2].1[2], boxes[3].1[2]];

        let origin = [0.5, 0.5, -1.0];
        let direction = [0.0, 0.0, 1.0];
        let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];

        let (_tmin4, mask) = ray_aabb4_intersect(
            origin, inv_dir, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z,
        );

        // Verify: should hit box 0 only (origin at 0.5,0.5, direction +z)
        assert_eq!(mask & 1, 1); // box 0 hit
        assert_eq!(mask & 2, 0); // box 1 miss (x=2..3)
        assert_eq!(mask & 4, 0); // box 2 miss (y=2..3)
        assert_eq!(mask & 8, 0); // box 3 miss (far away)
    }

    // --- 4-wide Point-AABB containment tests ---------------------------------

    #[test]
    fn test_point_aabb4_basic() {
        let boxes: [([f64; 3], [f64; 3]); 4] = [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
            ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]),
            (
                [f64::MAX, f64::MAX, f64::MAX],
                [f64::MIN, f64::MIN, f64::MIN],
            ),
        ];
        let min_x = [boxes[0].0[0], boxes[1].0[0], boxes[2].0[0], boxes[3].0[0]];
        let min_y = [boxes[0].0[1], boxes[1].0[1], boxes[2].0[1], boxes[3].0[1]];
        let min_z = [boxes[0].0[2], boxes[1].0[2], boxes[2].0[2], boxes[3].0[2]];
        let max_x = [boxes[0].1[0], boxes[1].1[0], boxes[2].1[0], boxes[3].1[0]];
        let max_y = [boxes[0].1[1], boxes[1].1[1], boxes[2].1[1], boxes[3].1[1]];
        let max_z = [boxes[0].1[2], boxes[1].1[2], boxes[2].1[2], boxes[3].1[2]];

        let point = [0.75, 0.75, 0.75];
        let mask = point_aabb4_contains(point, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z);

        // box 0 and 1 contain the point, box 2 and 3 do not
        assert_eq!(mask & 1, 1);
        assert_eq!(mask & 2, 2);
        assert_eq!(mask & 4, 0);
        assert_eq!(mask & 8, 0);
    }

    // --- 4-wide Distance-to-AABB tests ---------------------------------------

    #[test]
    fn test_dist_aabb4_basic() {
        let boxes: [([f64; 3], [f64; 3]); 4] = [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
            ([0.0, 2.0, 0.0], [1.0, 3.0, 1.0]),
            (
                [f64::MAX, f64::MAX, f64::MAX],
                [f64::MIN, f64::MIN, f64::MIN],
            ),
        ];
        let min_x = [boxes[0].0[0], boxes[1].0[0], boxes[2].0[0], boxes[3].0[0]];
        let min_y = [boxes[0].0[1], boxes[1].0[1], boxes[2].0[1], boxes[3].0[1]];
        let min_z = [boxes[0].0[2], boxes[1].0[2], boxes[2].0[2], boxes[3].0[2]];
        let max_x = [boxes[0].1[0], boxes[1].1[0], boxes[2].1[0], boxes[3].1[0]];
        let max_y = [boxes[0].1[1], boxes[1].1[1], boxes[2].1[1], boxes[3].1[1]];
        let max_z = [boxes[0].1[2], boxes[1].1[2], boxes[2].1[2], boxes[3].1[2]];

        let point = [0.5, 0.5, 0.5];
        let dists = dist_aabb4_sq(point, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z);

        // box 0 contains the point (dist=0), box 1 is 1.5 units away in x
        assert_eq!(dists[0], 0.0);
        assert!((dists[1] - 2.25).abs() < 1e-15); // (2.0 - 0.5)^2 = 2.25
    }

    #[test]
    fn test_simd_point_in_tet_multiple_cases() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];

        let points: Vec<[f64; 3]> = vec![
            [0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.33, 0.33, 0.33],
        ];

        // Inside: lambda_sum = sum of barycentric coords <= 1
        assert!(point_in_tet(points[0], v0, v1, v2, v3)); // 0.3 < 1
        assert!(point_in_tet(points[1], v0, v1, v2, v3)); // 0.75 < 1
        assert!(!point_in_tet(points[2], v0, v1, v2, v3)); // 1.5 > 1
        assert!(!point_in_tet(points[3], v0, v1, v2, v3)); // 3.0 > 1
        assert!(point_in_tet(points[4], v0, v1, v2, v3)); // vertex
        assert!(!point_in_tet(points[5], v0, v1, v2, v3)); // outside
        assert!(point_in_tet(points[6], v0, v1, v2, v3)); // 0.99 ~ 1
    }
}
