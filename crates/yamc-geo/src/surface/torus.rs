//! Ray–torus intersection helpers shared by the X/Y/Z-torus surface arms.

use super::roots::solve_quartic;

pub(super) fn torus_evaluate(t1: f64, t2: f64, ax: f64, a: f64, b: f64, c: f64) -> f64 {
    let rho = (t1 * t1 + t2 * t2).sqrt();
    (rho - a).powi(2) / (c * c) + ax * ax / (b * b) - 1.0
}

/// Smallest positive ray-torus intersection in axis-permuted
/// coordinates (`pt*`/`dt*` transverse, `pax`/`dax` axial), via the
/// same quartic the ZTorus arm has always used. Shared by X/Y/ZTorus.
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
) -> Option<f64> {
    // Scale the axial coordinate by c/b to transform the elliptical
    // cross-section to a circular torus with minor radius c.
    let ax_scale = if (b - c).abs() > 1e-14 { c / b } else { 1.0 };
    let pax_s = pax * ax_scale;
    let dax_s = dax * ax_scale;

    // Quartic coefficients for the circular torus implicit equation:
    // ((t1^2 + t2^2 + ax^2) + a^2 - c^2)^2 - 4 a^2 (t1^2 + t2^2) = 0
    let m = dt1 * dt1 + dt2 * dt2 + dax_s * dax_s;
    let k = pt1 * dt1 + pt2 * dt2 + pax_s * dax_s;
    let n = pt1 * pt1 + pt2 * pt2 + pax_s * pax_s;
    let m_t = dt1 * dt1 + dt2 * dt2;
    let k_t = pt1 * dt1 + pt2 * dt2;
    let n_t = pt1 * pt1 + pt2 * pt2;
    let big_s = n + a * a - c * c;

    let c4 = m * m;
    let c3 = 4.0 * m * k;
    let c2 = 4.0 * k * k + 2.0 * m * big_s - 4.0 * a * a * m_t;
    let c1 = 4.0 * k * big_s - 8.0 * a * a * k_t;
    let c0 = big_s * big_s - 4.0 * a * a * n_t;

    if c4.abs() < 1e-30 {
        return None;
    }
    let b_n = c3 / c4;
    let c_n = c2 / c4;
    let d_n = c1 / c4;
    let e_n = c0 / c4;
    let roots = solve_quartic(b_n, c_n, d_n, e_n);
    let mut min_t = f64::INFINITY;
    for &t in &roots {
        if t > 1e-12 && t < min_t {
            min_t = t;
        }
    }
    if min_t.is_finite() {
        Some(min_t)
    } else {
        None
    }
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
}
