//! Sphere ray-intersection, signed evaluation, and halfspace bounds.

pub(super) fn distance(
    point: [f64; 3],
    direction: [f64; 3],
    x0: f64,
    y0: f64,
    z0: f64,
    radius: f64,
) -> Option<f64> {
    // Ray-sphere intersection: (p + t*v - c)·(p + t*v - c) = r^2
    let oc = [point[0] - x0, point[1] - y0, point[2] - z0];
    let a = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    let b = 2.0 * (oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2]);
    let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if t1 > 1e-12 {
        Some(t1)
    } else if t2 > 1e-12 {
        Some(t2)
    } else {
        None
    }
}

pub(super) fn evaluate(point: (f64, f64, f64), x0: f64, y0: f64, z0: f64, radius: f64) -> f64 {
    let dx = point.0 - x0;
    let dy = point.1 - y0;
    let dz = point.2 - z0;
    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

pub(super) fn bounding_box(
    halfspace_below: bool,
    x0: f64,
    y0: f64,
    z0: f64,
    radius: f64,
) -> Option<([f64; 3], [f64; 3])> {
    if halfspace_below {
        Some((
            [x0 - radius, y0 - radius, z0 - radius],
            [x0 + radius, y0 + radius, z0 + radius],
        ))
    } else {
        None
    }
}

use super::{BoundaryType, Surface, SurfaceKind};

impl Surface {
    pub fn new_sphere(
        x0: f64,
        y0: f64,
        z0: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Sphere { x0, y0, z0, radius },
            boundary: boundary.unwrap_or_default(),
            name: None,
        }
    }
    /// Create a sphere with a specific boundary type
    pub fn sphere(
        x0: f64,
        y0: f64,
        z0: f64,
        radius: f64,
        surface_id: Option<usize>,
        boundary: Option<BoundaryType>,
    ) -> Self {
        Self::new_sphere(x0, y0, z0, radius, surface_id, boundary)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use crate::surface::Surface;
    #[test]
    fn sphere_constructor_and_evaluate() {
        let s = Surface::sphere(1.0, 2.0, 3.0, 5.0, Some(1), None);
        // On surface: distance from (1,2,3) to (6,2,3) = 5 = radius
        assert!((s.evaluate((6.0, 2.0, 3.0))).abs() < 1e-10);
        // Inside: at center
        assert!(s.evaluate((1.0, 2.0, 3.0)) < 0.0);
        // Outside: far away
        assert!(s.evaluate((100.0, 0.0, 0.0)) > 0.0);
    }
    #[test]
    fn sphere_distance_from_outside() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 2.0, None, None);
        // Ray from (10,0,0) in -x direction, hits sphere at x=2, distance=8
        let d = s.distance_to_surface([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 8.0).abs() < 1e-10);
    }
    #[test]
    fn sphere_distance_from_inside() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 5.0, None, None);
        // Ray from origin in +x direction, hits sphere at x=5
        let d = s.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_some());
        assert!((d.unwrap() - 5.0).abs() < 1e-10);
    }
    #[test]
    fn sphere_distance_miss() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        // Ray from (5,5,0) going in +x direction - misses the unit sphere
        let d = s.distance_to_surface([5.0, 5.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_none());
    }
    #[test]
    fn sphere_distance_tangent_miss() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        // Ray from (0,2,0) in +x direction - passes above sphere, no intersection
        let d = s.distance_to_surface([0.0, 2.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(d.is_none());
    }
    #[test]
    fn sphere_bounding_box_inside() {
        let s = Surface::new_sphere(1.0, 2.0, 3.0, 4.0, None, None);
        let bbox = s.bounding_box(true).unwrap();
        let (lo, hi) = bbox;
        assert!((lo[0] - (-3.0)).abs() < 1e-10);
        assert!((lo[1] - (-2.0)).abs() < 1e-10);
        assert!((lo[2] - (-1.0)).abs() < 1e-10);
        assert!((hi[0] - 5.0).abs() < 1e-10);
        assert!((hi[1] - 6.0).abs() < 1e-10);
        assert!((hi[2] - 7.0).abs() < 1e-10);
    }
    #[test]
    fn sphere_bounding_box_outside_is_none() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        assert!(s.bounding_box(false).is_none());
    }
}
