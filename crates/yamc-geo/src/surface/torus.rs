//! Ray-torus intersection helpers shared by the X/Y/Z-torus surface arms.

use super::roots::quartic_real_roots;

/// Implicit torus equation in axis-permuted coordinates:
/// `(rho - a)^2 / c^2 + ax^2 / b^2 - 1`, with `t1, t2` the transverse
/// displacements from the centre and `ax` the axial one. Shared by
/// X/Y/ZTorus.
pub(super) fn torus_evaluate(t1: f64, t2: f64, ax: f64, a: f64, b: f64, c: f64) -> f64 {
    let rho = (t1 * t1 + t2 * t2).sqrt();
    (rho - a).powi(2) / (c * c) + ax * ax / (b * b) - 1.0
}

/// Crossings closer than this fraction of the torus size `a + c` to
/// the point where a particle crossed the torus are that crossing and
/// are dropped. The quartic is solved in units of `a + c`, where the
/// factorising solver returns the root at the crossing point to a few
/// ulps (1e-16) unless the ray is within about 1e-7 rad of tangent, so
/// 1e-9 clears that error by orders of magnitude. It is also far below
/// any length transport resolves: 1.4e-6 cm for a torus of major radius
/// 907 cm and minor radius 468 cm. A genuine re-crossing that close
/// needs a ray within about 1e-9 rad of tangent.
const ON_SURFACE_TOLERANCE: f64 = 1e-9;

/// Relative slack on the bounding-ball test, far above round-off.
const BALL_SLACK: f64 = 1e-9;

/// Roots below this distance (cm) are behind or at the particle, the
/// floor every surface solver shares.
const MIN_DISTANCE: f64 = 1e-12;

/// Smallest positive ray-torus intersection in axis-permuted
/// coordinates (`pt*`/`dt*` transverse, `pax`/`dax` axial, relative to
/// the torus centre). Shared by X/Y/ZTorus.
///
/// The axial coordinate is scaled by `c/b`, which turns the elliptical
/// cross-section into a circle of radius `c`, and lengths are divided
/// by the torus size `a + c` so that the quartic's coefficients are of
/// order one. The quartic in the ray parameter,
/// `(|q|^2 + a^2 - c^2)^2 - 4 a^2 rho^2 = 0` with `q = p + t d`, is
/// solved by Algorithm 1010 (see [`quartic_real_roots`]).
///
/// `on_surface` is `Some(offset)` when the particle has just crossed
/// this torus and sits `offset` past the crossing along the ray
/// (transport moves a crossing particle a small fixed distance past
/// the surface; 0 for a particle exactly on it). The quartic is then
/// built at the crossing point with its constant term, the torus
/// function there, set to exactly zero, so the crossing just made is
/// the exact root `t = 0` instead of a round-off-sized root of either
/// sign; roots within [`ON_SURFACE_TOLERANCE`] of it are dropped, and
/// the rest are measured from the particle. Building the quartic at the
/// particle itself would not do: zeroing the constant term there moves
/// the torus through the particle, which shifts the next crossing by
/// up to the offset times the ratio of the torus function's slopes
/// along the ray, enough to stop a particle short of a surface.
///
/// Squaring the torus equation adds the roots of the mirror torus
/// `(rho + a)^2 + z^2 = c^2`. On the torus
/// `|q|^2 + a^2 - c^2 = 2 a rho >= 0` and on the mirror torus it is
/// `-2 a rho <= 0`, so a root is kept only where
/// `D s3^2 + s1^2 + s2^2 + a^2 - c^2 >= 0`, with `s` the hit point and
/// `D = c^2 / b^2`. Mirror roots are real only when `c > a`.
#[allow(clippy::too_many_arguments)]
pub(super) fn torus_distance(
    pt1: f64,
    pt2: f64,
    pax: f64,
    dt1: f64,
    dt2: f64,
    dax: f64,
    a: f64,
    b: f64,
    c: f64,
    on_surface: Option<f64>,
) -> Option<f64> {
    let offset = on_surface.unwrap_or(0.0);
    let size = a + c;
    let inv = 1.0 / size;
    let ax_scale = c / b;
    // Start at the crossing point, in units of the torus size; the ray
    // parameter is then in the same units, and the direction keeps its
    // components.
    let x = (pt1 - offset * dt1) * inv;
    let y = (pt2 - offset * dt2) * inv;
    let z = (pax - offset * dax) * ax_scale * inv;
    let w = dax * ax_scale;
    let ar = a * inv;
    let cr = c * inv;

    let m = dt1 * dt1 + dt2 * dt2 + w * w;
    let k = x * dt1 + y * dt2 + z * w;
    let n = x * x + y * y + z * z;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = x * dt1 + y * dt2;
    let n_t = x * x + y * y;
    let a2 = ar * ar;
    let big_s = n + a2 - cr * cr;

    let c4 = m * m;
    if c4 == 0.0 {
        return None;
    }
    // The torus lies in the ball |q| <= a + c, of radius 1 here: a ray
    // starting outside it that moves away or whose line misses it
    // cannot hit (the slack keeps points on the outer equator in).
    if n > 1.0 + BALL_SLACK && (k >= 0.0 || n - k * k / m > 1.0 + BALL_SLACK) {
        return None;
    }
    let c3 = 4.0 * m * k;
    let c2 = 4.0 * k * k + 2.0 * m * big_s - 4.0 * a2 * m_t;
    let c1 = 4.0 * k * big_s - 8.0 * a2 * k_t;
    let c0 = if on_surface.is_some() {
        0.0
    } else {
        big_s * big_s - 4.0 * a2 * n_t
    };
    let (roots, count) = quartic_real_roots(c3 / c4, c2 / c4, c1 / c4, c0 / c4);

    // Roots are from the crossing point; keep those ahead of the
    // particle, and past the tolerance when the crossing is its own.
    let mut floor = (MIN_DISTANCE + offset) * inv;
    if on_surface.is_some() {
        floor = floor.max(ON_SURFACE_TOLERANCE);
    }
    let mut min_t = f64::INFINITY;
    for &t in &roots[..count] {
        if t > floor && t < min_t {
            let (sx, sy, sz) = (x + t * dt1, y + t * dt2, z + t * w);
            if sx * sx + sy * sy + sz * sz + a2 - cr * cr >= 0.0 {
                min_t = t;
            }
        }
    }
    min_t.is_finite().then_some(min_t * size - offset)
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    /// Create a torus centered on the Z axis.
    /// a = major radius, b = minor radius in z direction, c = minor radius in xy plane.
    /// When b == c it is a circular torus.
    #[allow(clippy::too_many_arguments)]
    pub fn new_ztorus(
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::ZTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    /// `XTorus`: torus about the x axis through (x0, y0, z0).
    #[allow(clippy::too_many_arguments)]
    pub fn new_xtorus(
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::XTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    /// `YTorus`: torus about the y axis through (x0, y0, z0).
    #[allow(clippy::too_many_arguments)]
    pub fn new_ytorus(
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::YTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use crate::surface::Surface;
    fn direction_fan() -> Vec<[f64; 3]> {
        let mut dirs = Vec::new();
        for i in 0..8 {
            for j in 1..8 {
                let phi = std::f64::consts::TAU * i as f64 / 8.0;
                let theta = std::f64::consts::PI * j as f64 / 8.0;
                dirs.push([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        dirs
    }
    #[test]
    fn ztorus_evaluate_inside() {
        // Circular torus: major=3, minor=1, centered at origin
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Point at (3, 0, 0) is on the tube center (rho=3=a, z=0)
        assert!(torus.evaluate((3.0, 0.0, 0.0)) < 0.0);
    }
    #[test]
    fn ztorus_evaluate_outside() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Point at (0, 0, 0) center of the hole: rho=0, (0-3)^2/1 + 0 - 1 = 8
        assert!(torus.evaluate((0.0, 0.0, 0.0)) > 0.0);
        // Point far away
        assert!(torus.evaluate((10.0, 0.0, 0.0)) > 0.0);
    }
    #[test]
    fn ztorus_evaluate_on_surface() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Point at (4, 0, 0): rho=4, (4-3)^2/1 + 0 - 1 = 0
        assert!((torus.evaluate((4.0, 0.0, 0.0))).abs() < 1e-10);
        // Point at (2, 0, 0): rho=2, (2-3)^2/1 + 0 - 1 = 0
        assert!((torus.evaluate((2.0, 0.0, 0.0))).abs() < 1e-10);
        // Point at (3, 0, 1): rho=3, (3-3)^2/1 + 1/1 - 1 = 0
        assert!((torus.evaluate((3.0, 0.0, 1.0))).abs() < 1e-10);
    }
    #[test]
    fn ztorus_distance_from_outside_along_x() {
        // Circular torus: major=3, minor=1 at origin
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Ray from (10, 0, 0) in -x direction
        // Should hit outer surface at x=4, distance=6
        let d = torus.distance_to_surface([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 6.0).abs() < 1e-8);
    }
    #[test]
    fn ztorus_distance_through_hole_misses() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Ray along z-axis through center of hole: doesn't hit torus when hole radius > 0
        // Hole inner radius = a - c = 3 - 1 = 2 > 0
        let d = torus.distance_to_surface([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        assert!(d.is_none());
    }
    #[test]
    fn ztorus_distance_through_tube() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        // Ray from inside the hole at (1.5, 0, 0) in +x direction
        // Should hit inner surface of torus at x = 2 (rho = a - c = 2), distance = 0.5
        let d = torus.distance_to_surface([1.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 0.5).abs() < 1e-8);
    }
    #[test]
    fn ztorus_bounding_box() {
        let torus = Surface::new_ztorus(1.0, 2.0, 3.0, 5.0, 2.0, 1.5, None, None);
        let bbox = torus.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!((lo[0] - (1.0 - 6.5)).abs() < 1e-10);
        assert!((lo[1] - (2.0 - 6.5)).abs() < 1e-10);
        assert!((lo[2] - (3.0 - 2.0)).abs() < 1e-10);
        assert!((hi[0] - (1.0 + 6.5)).abs() < 1e-10);
        assert!((hi[1] - (2.0 + 6.5)).abs() < 1e-10);
        assert!((hi[2] - (3.0 + 2.0)).abs() < 1e-10);
    }
    #[test]
    fn ztorus_bounding_box_outside_is_none() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        assert!(torus.bounding_box(false).is_none());
    }
    #[test]
    fn ztorus_elliptical_evaluate() {
        // Elliptical torus: a=3, b=2 (z-extent), c=1 (xy-extent)
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 2.0, 1.0, None, None);
        // On surface: (4,0,0) -> (4-3)^2/1 + 0/4 - 1 = 0
        assert!((torus.evaluate((4.0, 0.0, 0.0))).abs() < 1e-10);
        // On surface: (3,0,2) -> (3-3)^2/1 + 4/4 - 1 = 0
        assert!((torus.evaluate((3.0, 0.0, 2.0))).abs() < 1e-10);
        // Inside: (3.5, 0, 0) -> (0.5)^2/1 + 0 - 1 = -0.75
        assert!(torus.evaluate((3.5, 0.0, 0.0)) < 0.0);
    }
    #[test]
    fn ztorus_offset_center_evaluate() {
        // Torus centered at (5, 5, 5), major=3, minor=1 (circular)
        let torus = Surface::new_ztorus(5.0, 5.0, 5.0, 3.0, 1.0, 1.0, None, None);
        // On surface: (9, 5, 5): rho = sqrt(16) = 4, (4-3)^2/1 + 0 - 1 = 0
        assert!((torus.evaluate((9.0, 5.0, 5.0))).abs() < 1e-10);
        // On surface: (3, 5, 5): rho = 2, (2-3)^2/1 + 0 - 1 = 0
        assert!((torus.evaluate((3.0, 5.0, 5.0))).abs() < 1e-10);
        // Inside tube center: (8, 5, 5): rho = 3 = a, (0)/1 + 0 - 1 = -1
        assert!(torus.evaluate((8.0, 5.0, 5.0)) < 0.0);
    }
    #[test]
    fn ztorus_offset_center_distance() {
        let torus = Surface::new_ztorus(5.0, 5.0, 5.0, 3.0, 1.0, 1.0, None, None);
        // Ray from (20, 5, 5) in -x direction, should hit outer surface at x = 5+4 = 9
        // distance = 20 - 9 = 11
        let d = torus.distance_to_surface([20.0, 5.0, 5.0], [-1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 11.0).abs() < 1e-8);
    }
    #[test]
    fn ztorus_elliptical_distance() {
        // Elliptical torus: a=3, b=2 (z-extent), c=1 (xy-extent)
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 2.0, 1.0, None, None);
        // Ray from (10, 0, 0) in -x direction, should hit outer surface at x=4 (a+c=4)
        // distance = 6
        let d = torus.distance_to_surface([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 6.0).abs() < 1e-6);
    }
    #[test]
    fn ztorus_offset_bounding_box() {
        let torus = Surface::new_ztorus(2.0, 3.0, 4.0, 5.0, 1.0, 2.0, None, None);
        // a+c = 7, b = 1
        let bbox = torus.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!((lo[0] - (2.0 - 7.0)).abs() < 1e-10);
        assert!((lo[1] - (3.0 - 7.0)).abs() < 1e-10);
        assert!((lo[2] - (4.0 - 1.0)).abs() < 1e-10);
        assert!((hi[0] - (2.0 + 7.0)).abs() < 1e-10);
        assert!((hi[1] - (3.0 + 7.0)).abs() < 1e-10);
        assert!((hi[2] - (4.0 + 1.0)).abs() < 1e-10);
    }
    #[test]
    fn xy_torus_permutation_oracle() {
        // The shared quartic is the ZTorus solver with permuted axes,
        // so X/YTorus must agree with ZTorus under coordinate
        // permutation EXACTLY (identical arithmetic): for XTorus the
        // axial coordinate is x (z's role) and the transverse pair is
        // (y, z); for YTorus the axial coordinate is y.
        let (a, b, c) = (3.0, 0.5, 0.8); // elliptical cross-section
        let zt = Surface::new_ztorus(0.0, 0.0, 0.0, a, b, c, None, None);
        let xt = Surface::new_xtorus(0.0, 0.0, 0.0, a, b, c, None, None);
        let yt = Surface::new_ytorus(0.0, 0.0, 0.0, a, b, c, None, None);
        let points = [
            [3.0, 0.0, 0.1],
            [0.5, 2.5, -0.2],
            [4.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for p in points {
            for d in direction_fan() {
                // ZTorus(t1=x, t2=y, ax=z) vs XTorus with (x,y,z) -> (ax, t1, t2)
                let z_ref = zt.distance_to_surface(p, d);
                let x_perm = xt.distance_to_surface([p[2], p[0], p[1]], [d[2], d[0], d[1]]);
                assert_eq!(z_ref, x_perm, "XTorus permutation mismatch at {p:?} {d:?}");
                // YTorus with (x,y,z) -> (t1, ax, t2)
                let y_perm = yt.distance_to_surface([p[0], p[2], p[1]], [d[0], d[2], d[1]]);
                assert_eq!(z_ref, y_perm, "YTorus permutation mismatch at {p:?} {d:?}");
                // evaluate must permute identically too.
                let ez = zt.evaluate((p[0], p[1], p[2]));
                let ex = xt.evaluate((p[2], p[0], p[1]));
                let ey = yt.evaluate((p[0], p[2], p[1]));
                assert_eq!(ez, ex);
                assert_eq!(ez, ey);
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Axis {
        X,
        Y,
        Z,
    }

    fn torus(axis: Axis, o: [f64; 3], a: f64, b: f64, c: f64) -> Surface {
        match axis {
            Axis::X => Surface::new_xtorus(o[0], o[1], o[2], a, b, c, None, None),
            Axis::Y => Surface::new_ytorus(o[0], o[1], o[2], a, b, c, None, None),
            Axis::Z => Surface::new_ztorus(o[0], o[1], o[2], a, b, c, None, None),
        }
    }

    /// Axis-frame `(t1, t2, ax)` to world coordinates, the inverse of
    /// the permutation each torus arm applies.
    fn to_world(axis: Axis, v: [f64; 3]) -> [f64; 3] {
        match axis {
            Axis::X => [v[2], v[0], v[1]],
            Axis::Y => [v[0], v[2], v[1]],
            Axis::Z => v,
        }
    }

    fn add(p: [f64; 3], d: [f64; 3], t: f64) -> [f64; 3] {
        [p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]]
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// Point on the torus at tube angle `theta` and azimuth `phi`, its
    /// outward unit normal, and two tangent vectors.
    #[allow(clippy::too_many_arguments)]
    fn surface_frame(
        axis: Axis,
        o: [f64; 3],
        a: f64,
        b: f64,
        c: f64,
        theta: f64,
        phi: f64,
    ) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let rho = a + c * theta.cos();
        let local = [rho * phi.cos(), rho * phi.sin(), b * theta.sin()];
        let p = to_world(axis, local);
        let p = [p[0] + o[0], p[1] + o[1], p[2] + o[2]];
        // Gradient of (rho - a)^2 / c^2 + ax^2 / b^2.
        let gr = theta.cos() / c;
        let n = unit(to_world(
            axis,
            [gr * phi.cos(), gr * phi.sin(), theta.sin() / b],
        ));
        let around = to_world(axis, [-phi.sin(), phi.cos(), 0.0]);
        let tube = unit(to_world(
            axis,
            [
                -theta.sin() * c * phi.cos(),
                -theta.sin() * c * phi.sin(),
                theta.cos() * b,
            ],
        ));
        (p, n, around, tube)
    }

    /// Independent oracle: the first sign change of the torus function
    /// along the ray beyond `skip`, found by marching in steps of
    /// `step` and bisecting.
    fn marched_distance(
        s: &Surface,
        p: [f64; 3],
        d: [f64; 3],
        skip: f64,
        step: f64,
        t_max: f64,
    ) -> Option<f64> {
        let f = |t: f64| {
            let q = add(p, d, t);
            s.evaluate((q[0], q[1], q[2]))
        };
        let mut t0 = skip;
        let f0 = f(t0).signum();
        while t0 < t_max {
            let t1 = t0 + step;
            if f(t1).signum() != f0 {
                let (mut lo, mut hi) = (t0, t1);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if f(mid).signum() == f0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
            t0 = t1;
        }
        None
    }

    fn assert_close(got: Option<f64>, want: Option<f64>, tol: f64, what: &str) {
        match (got, want) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!(
                (g - w).abs() <= tol * w.max(1.0),
                "{what}: got {g}, oracle {w}"
            ),
            _ => panic!("{what}: got {got:?}, oracle {want:?}"),
        }
    }

    /// Circular and elliptical tori, unit-sized and at the scale of the
    /// nested blanket tori that lost particles (major radius 907.2 cm).
    const SHAPES: [(f64, f64, f64); 5] = [
        (3.0, 1.0, 1.0),
        (3.0, 0.5, 0.8),
        (907.2, 467.7, 467.7),
        (907.2, 750.0, 467.7),
        (9.072, 3.177, 3.177),
    ];

    /// From points exactly on the torus, rays pointing in, out and
    /// obliquely, for every axis and shape: the distance flagged as on
    /// the surface matches a marching oracle, whether the particle is
    /// exactly on the surface or nudged past it as transport does.
    #[test]
    fn on_surface_rays_match_marching_oracle() {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for (a, b, c) in SHAPES {
                let size = a + c;
                let o = [0.1 * size, -0.2 * size, 0.05 * size];
                let s = torus(axis, o, a, b, c);
                for i in 0..12 {
                    let theta = std::f64::consts::TAU * (i as f64 + 0.3) / 12.0;
                    let phi = 0.7 + 0.9 * i as f64;
                    let (p, n, t1, t2) = surface_frame(axis, o, a, b, c, theta, phi);
                    let mut dirs = vec![n, [-n[0], -n[1], -n[2]]];
                    for (sn, s1, s2) in [
                        (-1.0, 0.7, 0.3),
                        (-1.0, -0.4, 0.9),
                        (1.0, 0.7, -0.3),
                        (1.0, -1.5, 0.4),
                        (-0.2, 1.0, 0.0),
                    ] {
                        dirs.push(unit([
                            sn * n[0] + s1 * t1[0] + s2 * t2[0],
                            sn * n[1] + s1 * t1[1] + s2 * t2[1],
                            sn * n[2] + s1 * t1[2] + s2 * t2[2],
                        ]));
                    }
                    for d in dirs {
                        let oracle =
                            marched_distance(&s, p, d, 1e-6 * size, size / 4000.0, 4.0 * size);
                        let what = format!("{axis:?} a={a} b={b} c={c} theta={theta} d={d:?}");
                        let got = s.distance_to_surface_on(p, d, Some(0.0));
                        assert_close(got, oracle, 1e-9, &what);
                        // Transport moves a crossing particle 1e-8 cm past.
                        let nudged = add(p, d, 1e-8);
                        let want = oracle.map(|t| t - 1e-8);
                        assert_close(
                            s.distance_to_surface_on(nudged, d, Some(1e-8)),
                            want,
                            1e-9,
                            &what,
                        );
                        // Pointing inward the ray must leave the tube again.
                        let inward = d[0] * n[0] + d[1] * n[1] + d[2] * n[2] < 0.0;
                        assert!(!inward || got.is_some(), "{what}: inward ray found no exit");
                    }
                }
            }
        }
    }

    /// The crossing just made is dropped when flagged: the exact answer
    /// from the outer equator straight inwards is the far side of the
    /// tube, never a round-off root at the particle.
    #[test]
    fn flagged_crossing_is_dropped() {
        let s = Surface::new_ztorus(0.0, 0.0, 0.0, 907.2, 467.7, 467.7, None, None);
        let r_out = 907.2 + 467.7;
        let p = [r_out * 0.6, r_out * 0.8, 0.0];
        let d = [-0.6, -0.8, 0.0];
        let got = s.distance_to_surface_on(p, d, Some(0.0)).unwrap();
        assert!((got - 2.0 * 467.7).abs() < 1e-9, "got {got}");
        // Pointing out of the outer equator there is nothing ahead.
        assert!(s
            .distance_to_surface_on(p, [0.6, 0.8, 0.0], Some(0.0))
            .is_none());
    }

    /// The torus that lost the particle quoted in the issue: from its
    /// position on the third torus, the flagged distance is the
    /// crossing a marching oracle finds.
    #[test]
    fn issue_lost_particle_ray() {
        let s = Surface::new_ztorus(0.0, 0.0, 0.0, 907.2, 417.7, 417.7, None, None);
        let p = [6.599275977704e2, 3.515844383589e2, -3.860647593018e2];
        let d = unit([-2.626239673012e-1, 2.390925578632e-1, -9.348066113231e-1]);
        let oracle = marched_distance(&s, p, d, 1e-3, 0.05, 4000.0);
        let got = s.distance_to_surface_on(p, d, Some(0.0));
        assert_close(got, oracle, 1e-9, "issue ray");
    }

    /// Squaring the torus equation adds the mirror torus
    /// `(rho + a)^2 + z^2 = c^2`, which has real points when `c > a`.
    /// Along the x axis of a spindle torus (a = 1, c = 1.5) the mirror
    /// roots sit at x = +-0.5, inside the torus, and must be rejected.
    #[test]
    fn spindle_torus_mirror_roots_rejected() {
        for (axis, b) in [(Axis::Z, 1.5), (Axis::X, 1.5), (Axis::Y, 2.5)] {
            let s = torus(axis, [0.0; 3], 1.0, b, 1.5);
            let p = to_world(axis, [0.2, 0.0, 0.0]);
            let d = to_world(axis, [1.0, 0.0, 0.0]);
            let got = s.distance_to_surface(p, d).unwrap();
            assert!((got - 2.3).abs() < 1e-12, "{axis:?}: got {got}");
            let p = to_world(axis, [5.0, 0.0, 0.0]);
            let d = to_world(axis, [-1.0, 0.0, 0.0]);
            let got = s.distance_to_surface(p, d).unwrap();
            assert!((got - 2.5).abs() < 1e-12, "{axis:?}: got {got}");
            // From the near side, flagged, the next real crossing is the
            // far side at x = -2.5, not the mirror roots in between.
            let p = to_world(axis, [2.5, 0.0, 0.0]);
            let got = s.distance_to_surface_on(p, d, Some(0.0)).unwrap();
            assert!((got - 5.0).abs() < 1e-12, "{axis:?}: got {got}");
        }
    }

    /// Rays whose two crossings nearly coincide (grazing the outer
    /// equator from outside) still return the first one accurately.
    #[test]
    fn grazing_rays() {
        let s = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        for h in [1e-2, 1e-4, 1e-6, 1e-8] {
            let y: f64 = 4.0 - h;
            let p = [10.0, y, 0.0];
            let want = 10.0 - (16.0 - y * y).sqrt();
            let got = s.distance_to_surface(p, [-1.0, 0.0, 0.0]).unwrap();
            assert!((got - want).abs() < 1e-9, "h={h}: got {got}, want {want}");
        }
        // Missing the torus by a hair, and through the hole along the axis.
        assert!(s
            .distance_to_surface([10.0, 4.0 + 1e-6, 0.0], [-1.0, 0.0, 0.0])
            .is_none());
        assert!(s
            .distance_to_surface([0.0, 0.0, 5.0], [0.0, 0.0, -1.0])
            .is_none());
        // Through the hole off-axis, and grazing the hole's rim inside.
        let got = s
            .distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            .unwrap();
        assert!((got - 2.0).abs() < 1e-12);
        let got = s
            .distance_to_surface([0.0, -5.0, 1.0 - 1e-6], [0.0, 1.0, 0.0])
            .unwrap();
        let want = 5.0 - (3.0 + (1.0 - (1.0f64 - 1e-6).powi(2)).sqrt());
        assert!((got - want).abs() < 1e-8, "got {got}, want {want}");
    }
}
