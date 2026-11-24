//! Plane ray-intersection, signed evaluation, and axis constraint.

pub(super) fn distance(
    point: [f64; 3],
    direction: [f64; 3],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> Option<f64> {
    // Plane: ax + by + cz - d = 0
    let denom = a * direction[0] + b * direction[1] + c * direction[2];
    if denom.abs() < 1e-12 {
        // Parallel, no intersection
        return None;
    }
    let num = d - (a * point[0] + b * point[1] + c * point[2]);
    let t = num / denom;
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

pub(super) fn evaluate(point: (f64, f64, f64), a: f64, b: f64, c: f64, d: f64) -> f64 {
    a * point.0 + b * point.1 + c * point.2 - d
}

pub(super) fn axis_constraint(
    halfspace_above: bool,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> Option<(usize, bool, f64)> {
    if a == 1.0 && b == 0.0 && c == 0.0 {
        // X plane: x = d. Above (x > d) -> lower bound at d; below -> upper bound.
        Some((0, !halfspace_above, d))
    } else if a == 0.0 && b == 1.0 && c == 0.0 {
        Some((1, !halfspace_above, d))
    } else if a == 0.0 && b == 0.0 && c == 1.0 {
        Some((2, !halfspace_above, d))
    } else {
        None
    }
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    pub fn new_plane(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Plane { a, b, c, d },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    pub fn x_plane(x0: f64, surface_id: Option<usize>, boundary: Option<BoundaryType>) -> Self {
        Self::new_plane(1.0, 0.0, 0.0, x0, surface_id, boundary)
    }
    pub fn y_plane(y0: f64, surface_id: Option<usize>, boundary: Option<BoundaryType>) -> Self {
        Self::new_plane(0.0, 1.0, 0.0, y0, surface_id, boundary)
    }
    pub fn z_plane(z0: f64, surface_id: Option<usize>, boundary: Option<BoundaryType>) -> Self {
        Self::new_plane(0.0, 0.0, 1.0, z0, surface_id, boundary)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use crate::surface::Surface;
    #[test]
    fn general_plane_constructor_and_evaluate() {
        // Plane: 1*x + 1*y + 1*z - 3 = 0 (passes through (1,1,1))
        let plane = Surface::new_plane(1.0, 1.0, 1.0, 3.0, Some(10), None);
        assert_eq!(plane.get_surface_id(), Some(10));
        // Point on plane: x+y+z = 3
        assert!((plane.evaluate((1.0, 1.0, 1.0))).abs() < 1e-10);
        // Point above: x+y+z = 6 > 3 => positive
        assert!(plane.evaluate((2.0, 2.0, 2.0)) > 0.0);
        // Point below: x+y+z = 0 < 3 => negative
        assert!(plane.evaluate((0.0, 0.0, 0.0)) < 0.0);
    }
    #[test]
    fn general_plane_distance() {
        // Plane: x + y + z = 3
        let plane = Surface::new_plane(1.0, 1.0, 1.0, 3.0, None, None);
        // Ray from origin along (1,1,1) (not normalized, but direction doesn't need to be)
        // Intersection: t*(1+1+1) = 3, so t = 1
        let d = plane.distance_to_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn general_plane_distance_parallel_returns_none() {
        // Plane: z = 5
        let plane = Surface::z_plane(5.0, None, None);
        // Ray moving in x-direction (parallel to z-plane)
        let d = plane.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_none());
    }
    #[test]
    fn general_plane_distance_behind_returns_none() {
        // Plane: z = 5
        let plane = Surface::z_plane(5.0, None, None);
        // Ray from above the plane moving upward (away from plane)
        let d = plane.distance_to_surface([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
        assert!(d.is_none());
    }
    #[test]
    fn x_plane_evaluate_and_distance() {
        let plane = Surface::x_plane(3.0, Some(1), None);
        // Evaluate: x - 3. At x=3, evaluate=0
        assert!((plane.evaluate((3.0, 0.0, 0.0))).abs() < 1e-10);
        // x=5 => positive
        assert!(plane.evaluate((5.0, 0.0, 0.0)) > 0.0);
        // x=1 => negative
        assert!(plane.evaluate((1.0, 0.0, 0.0)) < 0.0);
        // Distance: from (0,0,0) in +x direction, hit at x=3, distance=3
        let d = plane.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((d.unwrap() - 3.0).abs() < 1e-10);
    }
    #[test]
    fn y_plane_evaluate_and_distance() {
        let plane = Surface::y_plane(-2.0, Some(2), None);
        assert!((plane.evaluate((0.0, -2.0, 0.0))).abs() < 1e-10);
        // Distance: from (0,0,0) in -y direction, hit at y=-2, distance=2
        let d = plane.distance_to_surface([0.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        assert!((d.unwrap() - 2.0).abs() < 1e-10);
    }
    #[test]
    fn z_plane_evaluate_and_distance() {
        let plane = Surface::z_plane(7.0, Some(3), None);
        assert!((plane.evaluate((0.0, 0.0, 7.0))).abs() < 1e-10);
        let d = plane.distance_to_surface([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((d.unwrap() - 7.0).abs() < 1e-10);
    }
    #[test]
    fn plane_bounding_box_is_none() {
        let plane = Surface::x_plane(5.0, None, None);
        assert!(plane.bounding_box(true).is_none());
        assert!(plane.bounding_box(false).is_none());
    }
    #[test]
    fn general_plane_bounding_box_is_none() {
        let plane = Surface::new_plane(1.0, 1.0, 1.0, 3.0, None, None);
        assert!(plane.bounding_box(true).is_none());
        assert!(plane.bounding_box(false).is_none());
    }
}
