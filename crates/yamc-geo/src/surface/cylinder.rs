//! Infinite-cylinder ray-intersection, signed evaluation, and halfspace bounds.

pub(super) fn distance(
    point: [f64; 3],
    direction: [f64; 3],
    axis: [f64; 3],
    origin: [f64; 3],
    radius: f64,
) -> Option<f64> {
    // Ray-cylinder intersection (infinite cylinder).
    let p = point;
    let v = direction;
    let c = origin;
    let a_axis = axis;
    let v_dot_a = v[0] * a_axis[0] + v[1] * a_axis[1] + v[2] * a_axis[2];
    let d = [
        v[0] - v_dot_a * a_axis[0],
        v[1] - v_dot_a * a_axis[1],
        v[2] - v_dot_a * a_axis[2],
    ];
    let delta_p = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let delta_p_dot_a = delta_p[0] * a_axis[0] + delta_p[1] * a_axis[1] + delta_p[2] * a_axis[2];
    let m = [
        delta_p[0] - delta_p_dot_a * a_axis[0],
        delta_p[1] - delta_p_dot_a * a_axis[1],
        delta_p[2] - delta_p_dot_a * a_axis[2],
    ];
    let a_c = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let b_c = 2.0 * (d[0] * m[0] + d[1] * m[1] + d[2] * m[2]);
    let c_c = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] - radius * radius;
    let disc = b_c * b_c - 4.0 * a_c * c_c;
    if disc < 0.0 || a_c.abs() < 1e-12 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b_c - sqrt_disc) / (2.0 * a_c);
    let t2 = (-b_c + sqrt_disc) / (2.0 * a_c);
    if t1 > 1e-12 {
        Some(t1)
    } else if t2 > 1e-12 {
        Some(t2)
    } else {
        None
    }
}

pub(super) fn evaluate(
    point: (f64, f64, f64),
    axis: [f64; 3],
    origin: [f64; 3],
    radius: f64,
) -> f64 {
    let v = [
        point.0 - origin[0],
        point.1 - origin[1],
        point.2 - origin[2],
    ];
    let dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let d = [
        v[0] - dot * axis[0],
        v[1] - dot * axis[1],
        v[2] - dot * axis[2],
    ];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - radius
}

pub(super) fn bounding_box(
    halfspace_below: bool,
    axis: [f64; 3],
    origin: [f64; 3],
    radius: f64,
) -> Option<([f64; 3], [f64; 3])> {
    if !halfspace_below {
        return None;
    }
    if axis[0].abs() < 1e-10 && axis[1].abs() < 1e-10 && (axis[2] - 1.0).abs() < 1e-10 {
        // Z-cylinder: bounded in X,Y, infinite in Z
        Some((
            [origin[0] - radius, origin[1] - radius, f64::NEG_INFINITY],
            [origin[0] + radius, origin[1] + radius, f64::INFINITY],
        ))
    } else if (axis[0] - 1.0).abs() < 1e-10 && axis[1].abs() < 1e-10 && axis[2].abs() < 1e-10 {
        // X-cylinder: bounded in Y,Z, infinite in X
        Some((
            [f64::NEG_INFINITY, origin[1] - radius, origin[2] - radius],
            [f64::INFINITY, origin[1] + radius, origin[2] + radius],
        ))
    } else if axis[0].abs() < 1e-10 && (axis[1] - 1.0).abs() < 1e-10 && axis[2].abs() < 1e-10 {
        // Y-cylinder: bounded in X,Z, infinite in Y
        Some((
            [origin[0] - radius, f64::NEG_INFINITY, origin[2] - radius],
            [origin[0] + radius, f64::INFINITY, origin[2] + radius],
        ))
    } else {
        None
    }
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    pub fn new_cylinder(
        axis: [f64; 3],
        origin: [f64; 3],
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        // The cylinder `evaluate` / `distance_to_surface` math rejects the axial
        // component as `v - (v·a)a`, which is the true perpendicular component only
        // when |a| = 1. Normalize here so every construction path (Python factory,
        // Rust callers, deserialization) gets correct geometry from any direction
        // vector. A zero-length axis can't be normalized; leave it untouched and let
        // callers (e.g. the Python `Cylinder` factory) reject it.
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let axis = if norm > 0.0 {
            [axis[0] / norm, axis[1] / norm, axis[2] / norm]
        } else {
            axis
        };
        Surface {
            surface_id,
            kind: SurfaceKind::Cylinder {
                axis,
                origin,
                radius,
            },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    /// Create a cylinder oriented along the Z axis, centered at (x0, y0), with given radius and surface_id
    pub fn z_cylinder(
        x0: f64,
        y0: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cylinder([0.0, 0.0, 1.0], [x0, y0, 0.0], radius, surface_id, boundary)
    }
    /// Create a cylinder oriented along the X axis, centered at (y0, z0), with given radius
    pub fn x_cylinder(
        y0: f64,
        z0: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cylinder([1.0, 0.0, 0.0], [0.0, y0, z0], radius, surface_id, boundary)
    }
    /// Create a cylinder oriented along the Y axis, centered at (x0, z0), with given radius
    pub fn y_cylinder(
        x0: f64,
        z0: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cylinder([0.0, 1.0, 0.0], [x0, 0.0, z0], radius, surface_id, boundary)
    }
    /// Create a cylinder with individual axis components with a specific boundary type
    #[allow(clippy::too_many_arguments)]
    pub fn cylinder(
        x0: f64,
        y0: f64,
        z0: f64,
        axis_x: f64,
        axis_y: f64,
        axis_z: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_cylinder(
            [axis_x, axis_y, axis_z],
            [x0, y0, z0],
            radius,
            surface_id,
            boundary,
        )
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use crate::surface::{Surface, SurfaceKind};
    #[test]
    fn x_cylinder_evaluate_inside() {
        let cyl = Surface::x_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (5, 1, 0): perpendicular distance to axis = 1 < 2
        assert!(cyl.evaluate((5.0, 1.0, 0.0)) < 0.0);
    }
    #[test]
    fn x_cylinder_evaluate_outside() {
        let cyl = Surface::x_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (0, 3, 0): perpendicular distance = 3 > 2
        assert!(cyl.evaluate((0.0, 3.0, 0.0)) > 0.0);
    }
    #[test]
    fn x_cylinder_evaluate_on_surface() {
        let cyl = Surface::x_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (0, 2, 0): perpendicular distance = 2 == radius
        assert!((cyl.evaluate((0.0, 2.0, 0.0))).abs() < 1e-10);
    }
    #[test]
    fn x_cylinder_distance_from_outside() {
        let cyl = Surface::x_cylinder(0.0, 0.0, 1.0, None, None);
        // Ray from (0, 5, 0) in -y direction should hit at distance 4
        let d = cyl.distance_to_surface([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
        assert!((d.unwrap() - 4.0).abs() < 1e-10);
    }
    #[test]
    fn y_cylinder_evaluate_inside() {
        let cyl = Surface::y_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (1, 5, 0): perpendicular distance to axis = 1 < 2
        assert!(cyl.evaluate((1.0, 5.0, 0.0)) < 0.0);
    }
    #[test]
    fn y_cylinder_evaluate_outside() {
        let cyl = Surface::y_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (3, 0, 0): perpendicular distance = 3 > 2
        assert!(cyl.evaluate((3.0, 0.0, 0.0)) > 0.0);
    }
    #[test]
    fn y_cylinder_evaluate_on_surface() {
        let cyl = Surface::y_cylinder(0.0, 0.0, 2.0, None, None);
        // Point at (2, 0, 0): perpendicular distance = 2 == radius
        assert!((cyl.evaluate((2.0, 0.0, 0.0))).abs() < 1e-10);
    }
    #[test]
    fn y_cylinder_distance_from_outside() {
        let cyl = Surface::y_cylinder(0.0, 0.0, 1.0, None, None);
        // Ray from (5, 0, 0) in -x direction should hit at distance 4
        let d = cyl.distance_to_surface([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!((d.unwrap() - 4.0).abs() < 1e-10);
    }
    #[test]
    fn cylinder_normalizes_non_unit_axis() {
        // A deliberately non-unit axis must produce the same geometry as the
        // equivalent unit axis (regression for #395: `v - (v·a)a` only rejects
        // the axial component correctly when |a| = 1).
        let unit = Surface::new_cylinder([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 2.0, None, None);
        let scaled = Surface::new_cylinder([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], 2.0, None, None);

        for p in [(1.5, 0.0, 0.7), (3.0, 0.0, -4.0), (0.0, 2.0, 10.0)] {
            assert!(
                (unit.evaluate(p) - scaled.evaluate(p)).abs() < 1e-12,
                "surface value mismatch at {p:?}"
            );
        }

        // Oblique axis: the stored axis should be unit length after construction.
        let oblique = Surface::new_cylinder([0.5, 0.6, 0.4], [0.0, 0.0, 0.0], 1.0, None, None);
        if let SurfaceKind::Cylinder { axis, .. } = oblique.kind {
            let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-12,
                "axis not normalized: len = {len}"
            );
        } else {
            panic!("expected a cylinder");
        }
    }
    #[test]
    fn cylinder_non_unit_axis_radius_correct() {
        // On-surface point: perpendicular distance to the axis equals the radius.
        // With a non-unit axis the old math scaled the projection by |a|², giving
        // a wrong surface value here.
        let cyl = Surface::new_cylinder([0.0, 0.0, 7.0], [0.0, 0.0, 0.0], 2.0, None, None);
        assert!(cyl.evaluate((2.0, 0.0, 3.0)).abs() < 1e-12);
        let d = cyl.distance_to_surface([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!((d.unwrap() - 3.0).abs() < 1e-12);
    }
    #[test]
    fn x_cylinder_bounding_box() {
        let cyl = Surface::x_cylinder(1.0, 2.0, 3.0, None, None);
        let bbox = cyl.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!(lo[0].is_infinite() && lo[0] < 0.0); // X is infinite
        assert!((lo[1] - (1.0 - 3.0)).abs() < 1e-10);
        assert!((lo[2] - (2.0 - 3.0)).abs() < 1e-10);
        assert!(hi[0].is_infinite() && hi[0] > 0.0);
        assert!((hi[1] - (1.0 + 3.0)).abs() < 1e-10);
        assert!((hi[2] - (2.0 + 3.0)).abs() < 1e-10);
    }
    #[test]
    fn y_cylinder_bounding_box() {
        let cyl = Surface::y_cylinder(1.0, 2.0, 3.0, None, None);
        let bbox = cyl.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!((lo[0] - (1.0 - 3.0)).abs() < 1e-10);
        assert!(lo[1].is_infinite() && lo[1] < 0.0); // Y is infinite
        assert!((lo[2] - (2.0 - 3.0)).abs() < 1e-10);
        assert!((hi[0] - (1.0 + 3.0)).abs() < 1e-10);
        assert!(hi[1].is_infinite() && hi[1] > 0.0);
        assert!((hi[2] - (2.0 + 3.0)).abs() < 1e-10);
    }
    #[test]
    fn z_cylinder_constructor_and_evaluate() {
        let cyl = Surface::z_cylinder(1.0, 2.0, 3.0, Some(5), None);
        assert_eq!(cyl.get_surface_id(), Some(5));
        // On surface: (4, 2, 0) has perpendicular distance = sqrt((4-1)^2 + (2-2)^2) = 3
        assert!((cyl.evaluate((4.0, 2.0, 0.0))).abs() < 1e-10);
        // Inside: (1, 2, 100) is on the axis regardless of z
        assert!(cyl.evaluate((1.0, 2.0, 100.0)) < 0.0);
        // Outside: (10, 2, 0) has perpendicular distance = 9 > 3
        assert!(cyl.evaluate((10.0, 2.0, 0.0)) > 0.0);
    }
    #[test]
    fn z_cylinder_distance_from_outside() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 2.0, None, None);
        // Ray from (10, 0, 0) in -x direction, hits at x=2, distance=8
        let d = cyl.distance_to_surface([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 8.0).abs() < 1e-10);
    }
    #[test]
    fn z_cylinder_distance_from_inside() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 5.0, None, None);
        // Ray from origin in +x direction, hits at x=5
        let d = cyl.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 5.0).abs() < 1e-10);
    }
    #[test]
    fn z_cylinder_distance_parallel_miss() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, None, None);
        // Ray along z-axis outside the cylinder (at x=5, y=0)
        let d = cyl.distance_to_surface([5.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(d.is_none());
    }
    #[test]
    fn z_cylinder_distance_along_axis() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, None, None);
        // Ray along z-axis from inside the cylinder: parallel to axis, never hits
        let d = cyl.distance_to_surface([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(d.is_none());
    }
    #[test]
    fn cylinder_individual_axis_components() {
        // Create a z-cylinder using the individual-component constructor
        let cyl = Surface::cylinder(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, Some(7), None);
        assert_eq!(cyl.get_surface_id(), Some(7));
        // Evaluate at (3, 0, 0): perpendicular distance = 3 = radius
        assert!((cyl.evaluate((3.0, 0.0, 0.0))).abs() < 1e-10);
    }
    #[test]
    fn z_cylinder_bounding_box_inside() {
        let cyl = Surface::z_cylinder(1.0, 2.0, 3.0, None, None);
        let bbox = cyl.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!((lo[0] - (-2.0)).abs() < 1e-10);
        assert!((lo[1] - (-1.0)).abs() < 1e-10);
        assert!(lo[2].is_infinite() && lo[2] < 0.0);
        assert!((hi[0] - 4.0).abs() < 1e-10);
        assert!((hi[1] - 5.0).abs() < 1e-10);
        assert!(hi[2].is_infinite() && hi[2] > 0.0);
    }
    #[test]
    fn cylinder_bounding_box_outside_is_none() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, None, None);
        assert!(cyl.bounding_box(false).is_none());
    }
    #[test]
    fn oblique_cylinder_bounding_box_is_none() {
        // Cylinder with a non-axis-aligned direction
        let cyl = Surface::new_cylinder([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.0, None, None);
        // Oblique cylinder should return None for bounding_box
        assert!(cyl.bounding_box(true).is_none());
    }
    #[test]
    fn x_cylinder_offset_center_distance() {
        // X-cylinder centered at y=3, z=4 with radius 1
        let cyl = Surface::x_cylinder(3.0, 4.0, 1.0, None, None);
        // Ray from (0, 3, 10) in -z direction, hits at z = 5 (top of cylinder), distance = 5
        let d = cyl.distance_to_surface([0.0, 3.0, 10.0], [0.0, 0.0, -1.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 5.0).abs() < 1e-10);
    }
    #[test]
    fn y_cylinder_offset_center_distance() {
        // Y-cylinder centered at x=2, z=3 with radius 1
        let cyl = Surface::y_cylinder(2.0, 3.0, 1.0, None, None);
        // Ray from (2, 0, 10) in -z direction, hits at z = 4, distance = 6
        let d = cyl.distance_to_surface([2.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 6.0).abs() < 1e-10);
    }
    #[test]
    fn z_cylinder_miss_perpendicular() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, None, None);
        // Ray from (5, 0, 0) in +y direction, misses the unit z-cylinder
        let d = cyl.distance_to_surface([5.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(d.is_none());
    }
}
