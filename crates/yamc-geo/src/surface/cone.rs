//! Double-cone ray-intersection and signed evaluation.

pub(super) fn distance(
    point: [f64; 3],
    direction: [f64; 3],
    apex: [f64; 3],
    axis: [f64; 3],
    tan2_theta: f64,
) -> Option<f64> {
    // Substituting P + tD into perp^2 = tan2 * axial^2 gives a quadratic
    // with kappa = 1 + tan2.
    let v = [point[0] - apex[0], point[1] - apex[1], point[2] - apex[2]];
    let dv = direction;
    let kappa = 1.0 + tan2_theta;
    let va = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let da = dv[0] * axis[0] + dv[1] * axis[1] + dv[2] * axis[2];
    let vd = v[0] * dv[0] + v[1] * dv[1] + v[2] * dv[2];
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let qa = 1.0 - kappa * da * da;
    let qb = 2.0 * (vd - kappa * va * da);
    let qc = v2 - kappa * va * va;
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

pub(super) fn evaluate(
    point: (f64, f64, f64),
    apex: [f64; 3],
    axis: [f64; 3],
    tan2_theta: f64,
) -> f64 {
    let v = [point.0 - apex[0], point.1 - apex[1], point.2 - apex[2]];
    let s = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    (v2 - s * s) - tan2_theta * s * s
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    /// Arbitrary-axis double cone: `apex`, unit `axis`, and
    /// `tan2_theta` (the `r2` coefficient). Double-sheeted.
    pub fn new_cone(
        apex: [f64; 3],
        axis: [f64; 3],
        tan2_theta: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Cone {
                apex,
                axis,
                tan2_theta,
            },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    /// `XCone`: `(y-y0)^2 + (z-z0)^2 = r2 (x-x0)^2`.
    pub fn x_cone(
        x0: f64,
        y0: f64,
        z0: f64,
        r2: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cone([x0, y0, z0], [1.0, 0.0, 0.0], r2, surface_id, boundary)
    }
    /// `YCone`: `(x-x0)^2 + (z-z0)^2 = r2 (y-y0)^2`.
    pub fn y_cone(
        x0: f64,
        y0: f64,
        z0: f64,
        r2: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cone([x0, y0, z0], [0.0, 1.0, 0.0], r2, surface_id, boundary)
    }
    /// `ZCone`: `(x-x0)^2 + (y-y0)^2 = r2 (z-z0)^2`.
    pub fn z_cone(
        x0: f64,
        y0: f64,
        z0: f64,
        r2: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cone([x0, y0, z0], [0.0, 0.0, 1.0], r2, surface_id, boundary)
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
    /// ZCone (x-x0)^2 + (y-y0)^2 - r2 (z-z0)^2 = 0 as a quadric:
    /// a=1, b=1, c=-r2, g=-2x0, h=-2y0, j=2 r2 z0,
    /// k = x0^2 + y0^2 - r2 z0^2.
    fn zcone_as_quadric(x0: f64, y0: f64, z0: f64, r2: f64) -> Surface {
        Surface::new_quadric(
            1.0,
            1.0,
            -r2,
            0.0,
            0.0,
            0.0,
            -2.0 * x0,
            -2.0 * y0,
            2.0 * r2 * z0,
            x0 * x0 + y0 * y0 - r2 * z0 * z0,
            None,
            None,
        )
    }
    #[test]
    fn cone_matches_quadric_oracle() {
        // The general quadric (already oracle-tested against the other
        // dedicated variants) validates the cone solver: the same
        // double cone written both ways must agree on evaluate sign and
        // ray distances.
        let (x0, y0, z0, r2) = (0.5, -1.0, 2.0, 0.25);
        let cone = Surface::z_cone(x0, y0, z0, r2, None, None);
        let quadric = zcone_as_quadric(x0, y0, z0, r2);
        let points = [
            [0.0, 0.0, 0.0],
            [0.5, -1.0, 5.0],
            [3.0, 2.0, -4.0],
            [0.6, -1.0, 2.1],
        ];
        for p in points {
            let ec = cone.evaluate((p[0], p[1], p[2]));
            let eq = quadric.evaluate((p[0], p[1], p[2]));
            assert!(
                (ec - eq).abs() < 1e-9 * ec.abs().max(1.0),
                "evaluate mismatch at {p:?}: cone {ec}, quadric {eq}"
            );
            for dir in direction_fan() {
                let dc = cone.distance_to_surface(p, dir);
                let dq = quadric.distance_to_surface(p, dir);
                match (dc, dq) {
                    // 1e-6: the two algebraic arrangements round
                    // differently for near-surface origins (observed
                    // ~1e-8 here); a real solver error is orders of
                    // magnitude larger.
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() < 1e-6 * a.max(1.0),
                        "distance mismatch at {p:?} dir {dir:?}: {a} vs {b}"
                    ),
                    (None, None) => {}
                    other => panic!("hit/miss mismatch at {p:?} dir {dir:?}: {other:?}"),
                }
            }
        }
    }
    #[test]
    fn cone_is_double_sheeted() {
        // Both sheets exist. A +z ray from below the
        // apex of a z-cone crosses the lower sheet (negative-z half)
        // before the apex region and the upper sheet beyond it.
        let cone = Surface::z_cone(0.0, 0.0, 0.0, 1.0, None, None);
        // On-axis point below the apex is INSIDE the lower sheet
        // (evaluate < 0: perp^2 < tan2 * axial^2).
        assert!(cone.evaluate((0.0, 0.0, -3.0)) < 0.0);
        assert!(cone.evaluate((0.0, 0.0, 3.0)) < 0.0, "upper sheet interior");
        assert!(cone.evaluate((3.0, 0.0, 0.0)) > 0.0, "outside at the waist");
        // Ray through both sheets: from (-5, 0, -2) along +x crosses the
        // lower sheet twice (45-degree cone -> x = +-2 at z = -2).
        let d1 = cone
            .distance_to_surface([-5.0, 0.0, -2.0], [1.0, 0.0, 0.0])
            .expect("must hit the lower sheet");
        assert!((d1 - 3.0).abs() < 1e-12, "first crossing at x=-2, got {d1}");
    }
    #[test]
    fn axis_cone_constructors_permute() {
        // x_cone/y_cone are axis permutations of z_cone.
        let xc = Surface::x_cone(1.0, 2.0, 3.0, 0.5, None, None);
        let yc = Surface::y_cone(1.0, 2.0, 3.0, 0.5, None, None);
        let zc = Surface::z_cone(1.0, 2.0, 3.0, 0.5, None, None);
        // Points displaced along each axis from the apex are inside the
        // respective cone (axial term dominates).
        assert!(xc.evaluate((4.0, 2.0, 3.0)) < 0.0);
        assert!(yc.evaluate((1.0, 5.0, 3.0)) < 0.0);
        assert!(zc.evaluate((1.0, 2.0, 6.0)) < 0.0);
        // ... and outside the others.
        assert!(yc.evaluate((4.0, 2.0, 3.0)) > 0.0);
        assert!(zc.evaluate((4.0, 2.0, 3.0)) > 0.0);
    }
}
