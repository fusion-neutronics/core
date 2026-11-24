//! General quadric ray-intersection and signed evaluation.

#[allow(clippy::too_many_arguments)]
pub(super) fn distance(
    point: [f64; 3],
    direction: [f64; 3],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    j: f64,
    _k: f64,
) -> Option<f64> {
    // Substituting P + tD into the quadric gives A t^2 + B t + C = 0.
    let (x, y, z) = (point[0], point[1], point[2]);
    let (u, v, w) = (direction[0], direction[1], direction[2]);
    let qa = a * u * u + b * v * v + c * w * w + d * u * v + e * v * w + f * u * w;
    let qb = 2.0 * (a * x * u + b * y * v + c * z * w)
        + d * (x * v + y * u)
        + e * (y * w + z * v)
        + f * (x * w + z * u)
        + g * u
        + h * v
        + j * w;
    let qc = a * x * x
        + b * y * y
        + c * z * z
        + d * x * y
        + e * y * z
        + f * x * z
        + g * x
        + h * y
        + j * z
        + _k;
    if qa.abs() < 1e-30 {
        if qb.abs() < 1e-30 {
            return None;
        }
        let t = -qc / qb;
        return if t > 1e-12 { Some(t) } else { None };
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-qb - sqrt_disc) / (2.0 * qa);
    let t2 = (-qb + sqrt_disc) / (2.0 * qa);
    let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
    if lo > 1e-12 {
        Some(lo)
    } else if hi > 1e-12 {
        Some(hi)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    point: (f64, f64, f64),
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    j: f64,
    k: f64,
) -> f64 {
    let (x, y, z) = point;
    a * x * x
        + b * y * y
        + c * z * z
        + d * x * y
        + e * y * z
        + f * x * z
        + g * x
        + h * y
        + j * z
        + k
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    /// General quadric (`Quadric`):
    /// `a x^2 + b y^2 + c z^2 + d xy + e yz + f xz + g x + h y + j z + k = 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_quadric(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
        h: f64,
        j: f64,
        k: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Quadric {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
                j,
                k,
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
    /// Sphere of radius r at (x0,y0,z0) expressed as a quadric:
    /// x^2+y^2+z^2 - 2x0 x - 2y0 y - 2z0 z + (x0^2+y0^2+z0^2 - r^2) = 0.
    fn sphere_as_quadric(x0: f64, y0: f64, z0: f64, r: f64) -> Surface {
        Surface::new_quadric(
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -2.0 * x0,
            -2.0 * y0,
            -2.0 * z0,
            x0 * x0 + y0 * y0 + z0 * z0 - r * r,
            None,
            None,
        )
    }
    /// Deterministic direction fan over the unit sphere.
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
    fn quadric_matches_sphere_oracle() {
        // The dedicated Sphere variant is the oracle: the same surface
        // written as a general quadric must agree on the evaluate SIGN
        // (evaluate magnitudes differ -- sphere is signed distance, the
        // quadric is the implicit polynomial) and on ray distances.
        let sphere = Surface::new_sphere(1.0, -2.0, 0.5, 3.0, None, None);
        let quadric = sphere_as_quadric(1.0, -2.0, 0.5, 3.0);
        let points = [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [1.0, -2.0, 3.49],
            [1.0, -2.0, 3.51],
            [-3.0, 1.0, 0.0],
        ];
        for p in points {
            let es = sphere.evaluate((p[0], p[1], p[2]));
            let eq = quadric.evaluate((p[0], p[1], p[2]));
            assert_eq!(
                es > 0.0,
                eq > 0.0,
                "sign mismatch at {p:?}: sphere {es}, quadric {eq}"
            );
            for dir in direction_fan() {
                let ds = sphere.distance_to_surface(p, dir);
                let dq = quadric.distance_to_surface(p, dir);
                match (ds, dq) {
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() < 1e-9 * a.max(1.0),
                        "distance mismatch at {p:?} dir {dir:?}: {a} vs {b}"
                    ),
                    (None, None) => {}
                    other => panic!("hit/miss mismatch at {p:?} dir {dir:?}: {other:?}"),
                }
            }
        }
    }
    #[test]
    fn quadric_matches_plane_oracle() {
        // Degenerate (linear) quadric: 2x + 3y - z - 4 = 0 must agree
        // with the Plane variant on sign and ray distances.
        let plane = Surface::new_plane(2.0, 3.0, -1.0, 4.0, None, None);
        let quadric = Surface::new_quadric(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, -1.0, -4.0, None, None,
        );
        let points = [[0.0, 0.0, 0.0], [3.0, 1.0, -2.0], [1.0, 1.0, 1.0]];
        for p in points {
            let ep = plane.evaluate((p[0], p[1], p[2]));
            let eq = quadric.evaluate((p[0], p[1], p[2]));
            assert!(
                (ep - eq).abs() < 1e-12,
                "plane-form quadric must equal the plane evaluate exactly"
            );
            for dir in direction_fan() {
                let dp = plane.distance_to_surface(p, dir);
                let dq = quadric.distance_to_surface(p, dir);
                match (dp, dq) {
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() < 1e-9 * a.max(1.0),
                        "distance mismatch at {p:?} dir {dir:?}: {a} vs {b}"
                    ),
                    (None, None) => {}
                    other => panic!("hit/miss mismatch at {p:?} dir {dir:?}: {other:?}"),
                }
            }
        }
    }
    #[test]
    fn quadric_zcylinder_oracle() {
        // x^2 + y^2 - r^2 = 0 about the z axis at the origin.
        let cyl = Surface::new_cylinder([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 2.0, None, None);
        let quadric = Surface::new_quadric(
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, None, None,
        );
        let points = [[0.0, 0.0, 0.0], [3.0, 0.0, 5.0], [1.0, 1.0, -2.0]];
        for p in points {
            assert_eq!(
                cyl.evaluate((p[0], p[1], p[2])) > 0.0,
                quadric.evaluate((p[0], p[1], p[2])) > 0.0,
                "sign mismatch at {p:?}"
            );
            for dir in direction_fan() {
                let dc = cyl.distance_to_surface(p, dir);
                let dq = quadric.distance_to_surface(p, dir);
                match (dc, dq) {
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() < 1e-9 * a.max(1.0),
                        "distance mismatch at {p:?} dir {dir:?}: {a} vs {b}"
                    ),
                    (None, None) => {}
                    other => panic!("hit/miss mismatch at {p:?} dir {dir:?}: {other:?}"),
                }
            }
        }
    }
}
