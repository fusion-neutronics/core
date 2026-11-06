mod cone;
mod cylinder;
mod plane;
mod quadric;
mod roots;
mod sphere;
mod torus;

use crate::region::{HalfspaceType, RegionExpr};
pub use roots::quartic_real_roots;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use torus::{torus_distance, torus_evaluate};

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum BoundaryType {
    #[default]
    Transmission,
    Vacuum,
}

impl BoundaryType {
    /// Parse a boundary type from a string (case-sensitive), returning None for invalid strings
    /// Only accepts lowercase strings: "transmission" or "vacuum"
    pub fn from_str_option(s: &str) -> Option<Self> {
        match s {
            "transmission" => Some(BoundaryType::Transmission),
            "vacuum" => Some(BoundaryType::Vacuum),
            &_ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surface {
    pub surface_id: Option<usize>,
    pub kind: SurfaceKind,
    pub boundary: BoundaryType,
    /// Optional human-readable name (`"inner"`, `"outer_blanket"` …).
    /// Pure metadata: yamc itself uses `surface_id` for identity. Shown
    /// by the browser-side surface editor when present so recipients
    /// edit named surfaces instead of "Sphere #1".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurfaceKind {
    Plane {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    Sphere {
        x0: f64,
        y0: f64,
        z0: f64,
        radius: f64,
    },
    Cylinder {
        #[serde(deserialize_with = "unit_axis")]
        axis: [f64; 3],
        origin: [f64; 3],
        radius: f64,
    },
    ZTorus {
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
    },
    /// Torus about the x axis (`XTorus`):
    /// `((sqrt((y-y0)^2 + (z-z0)^2) - a)^2 / c^2 + (x-x0)^2 / b^2 = 1`.
    XTorus {
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
    },
    /// Torus about the y axis (`YTorus`):
    /// `((sqrt((x-x0)^2 + (z-z0)^2) - a)^2 / c^2 + (y-y0)^2 / b^2 = 1`.
    YTorus {
        x0: f64,
        y0: f64,
        z0: f64,
        a: f64,
        b: f64,
        c: f64,
    },
    /// General quadric (`Quadric`):
    /// `a x^2 + b y^2 + c z^2 + d xy + e yz + f xz + g x + h y + j z + k = 0`.
    /// The quadric coefficient naming skips `i`.
    Quadric {
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
    },
    /// Arbitrary-axis double cone (covers the `XCone`/`YCone`/
    /// `ZCone` like `Cylinder` covers the axis-aligned cylinders):
    /// points where `|perp(p - apex)|^2 = tan2_theta * ((p - apex) . axis)^2`.
    /// Double-sheeted; `tan2_theta` is the `r2` coefficient.
    Cone {
        apex: [f64; 3],
        #[serde(deserialize_with = "unit_axis")]
        axis: [f64; 3],
        tan2_theta: f64,
    },
}

/// A Cylinder or Cone axis as a model file holds it, made the unit vector
/// their evaluate and distance assume (as `Surface::new_cylinder` and the
/// Python factories make it): a hand-written model may give any length. One
/// already unit to rounding is kept as written, so a saved model loads with
/// the axes it was saved with. A zero axis has no direction and is refused.
fn unit_axis<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<[f64; 3], D::Error> {
    let axis = <[f64; 3]>::deserialize(deserializer)?;
    let n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if !(n2 > 0.0 && n2.is_finite()) {
        return Err(serde::de::Error::custom(format!(
            "a Cylinder or Cone axis must be a non-zero, finite vector, not {axis:?}"
        )));
    }
    if (n2 - 1.0).abs() <= 1e-12 {
        return Ok(axis);
    }
    let n = n2.sqrt();
    Ok([axis[0] / n, axis[1] / n, axis[2] / n])
}

/// Implicit torus equation in axis-permuted coordinates:
/// `(rho - a)^2 / c^2 + ax^2 / b^2 - 1`, with `t1, t2` the transverse
/// displacements from the centre and `ax` the axial one. Shared by
/// X/Y/ZTorus.
// Regular Rust implementation
impl Surface {
    /// Compute the distance from a point along a direction to the surface.
    /// Returns Some(distance) if intersection exists and distance > 0, else None.
    pub fn distance_to_surface(&self, point: [f64; 3], direction: [f64; 3]) -> Option<f64> {
        self.distance_to_surface_on(point, direction, None)
    }

    /// [`Surface::distance_to_surface`] for a particle that may have
    /// just crossed this surface: `on_surface` is `Some(offset)` when it
    /// did and has since moved `offset` past the crossing along
    /// `direction` (transport moves a crossing particle 1e-8 cm past the
    /// surface; 0 for a particle exactly on it), `None` otherwise. The
    /// tori then treat the crossing as an exact root and drop it, rather
    /// than solve for it in round-off (which can return it as a tiny
    /// positive distance). The other surfaces ignore it: a quadratic's
    /// root at the crossing comes back accurately, behind the particle.
    pub fn distance_to_surface_on(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<f64>,
    ) -> Option<f64> {
        match &self.kind {
            SurfaceKind::Plane { a, b, c, d } => plane::distance(point, direction, *a, *b, *c, *d),
            SurfaceKind::Sphere { x0, y0, z0, radius } => {
                sphere::distance(point, direction, *x0, *y0, *z0, *radius)
            }
            SurfaceKind::Cylinder {
                axis,
                origin,
                radius,
            } => cylinder::distance(point, direction, *axis, *origin, *radius),
            SurfaceKind::ZTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_distance(
                point[0] - x0,
                point[1] - y0,
                point[2] - z0,
                direction[0],
                direction[1],
                direction[2],
                *a,
                *b,
                *c,
                on_surface,
            ),
            SurfaceKind::XTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_distance(
                point[1] - y0,
                point[2] - z0,
                point[0] - x0,
                direction[1],
                direction[2],
                direction[0],
                *a,
                *b,
                *c,
                on_surface,
            ),
            SurfaceKind::YTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_distance(
                point[0] - x0,
                point[2] - z0,
                point[1] - y0,
                direction[0],
                direction[2],
                direction[1],
                *a,
                *b,
                *c,
                on_surface,
            ),
            SurfaceKind::Quadric {
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
            } => quadric::distance(point, direction, *a, *b, *c, *d, *e, *f, *g, *h, *j, *k),
            SurfaceKind::Cone {
                apex,
                axis,
                tan2_theta,
            } => cone::distance(point, direction, *apex, *axis, *tan2_theta),
        }
    }
    /// Builder-style setter for the optional `name` metadata field.
    /// `Surface::sphere(...).with_name("inner")` is the easiest way to
    /// label a surface without rewriting the per-kind constructors.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get the boundary type of the surface
    pub fn boundary(&self) -> &BoundaryType {
        &self.boundary
    }

    /// Set the boundary type of the surface
    pub fn set_boundary(&mut self, boundary: BoundaryType) {
        self.boundary = boundary;
    }

    /// Get the surface ID
    pub fn get_surface_id(&self) -> Option<usize> {
        self.surface_id
    }

    /// Set the surface ID
    pub fn set_surface_id(&mut self, surface_id: usize) {
        self.surface_id = Some(surface_id);
    }

    pub fn evaluate(&self, point: (f64, f64, f64)) -> f64 {
        match &self.kind {
            SurfaceKind::Plane { a, b, c, d } => plane::evaluate(point, *a, *b, *c, *d),
            SurfaceKind::Sphere { x0, y0, z0, radius } => {
                sphere::evaluate(point, *x0, *y0, *z0, *radius)
            }
            SurfaceKind::Cylinder {
                axis,
                origin,
                radius,
            } => cylinder::evaluate(point, *axis, *origin, *radius),
            SurfaceKind::ZTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_evaluate(point.0 - x0, point.1 - y0, point.2 - z0, *a, *b, *c),
            SurfaceKind::XTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_evaluate(point.1 - y0, point.2 - z0, point.0 - x0, *a, *b, *c),
            SurfaceKind::YTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => torus_evaluate(point.0 - x0, point.2 - z0, point.1 - y0, *a, *b, *c),
            SurfaceKind::Quadric {
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
            } => quadric::evaluate(point, *a, *b, *c, *d, *e, *f, *g, *h, *j, *k),
            SurfaceKind::Cone {
                apex,
                axis,
                tan2_theta,
            } => cone::evaluate(point, *apex, *axis, *tan2_theta),
        }
    }

    /// Get the bounding box for this surface when used as a halfspace.
    /// For finite surfaces like spheres and cylinders, returns the bounding box bounds for the negative halfspace (inside).
    /// For axis-aligned planes, returns None as they should only contribute axis constraints.
    /// halfspace_below: true for negative halfspace (inside), false for positive halfspace (outside)
    pub fn bounding_box(&self, halfspace_below: bool) -> Option<([f64; 3], [f64; 3])> {
        match &self.kind {
            // General quadrics are unbounded in general (an ellipsoid
            // special-case could tighten this later); cells using them
            // fall back to the unbounded BVH bucket.
            SurfaceKind::Quadric { .. } => None,
            // The double cone is unbounded along its axis.
            SurfaceKind::Cone { .. } => None,
            SurfaceKind::Plane { .. } => {
                // Planes contribute constraints through axis_constraint(), not bounding_box()
                None
            }
            SurfaceKind::Sphere { x0, y0, z0, radius } => {
                sphere::bounding_box(halfspace_below, *x0, *y0, *z0, *radius)
            }
            SurfaceKind::Cylinder {
                axis,
                origin,
                radius,
            } => cylinder::bounding_box(halfspace_below, *axis, *origin, *radius),
            SurfaceKind::ZTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => {
                if halfspace_below {
                    Some((
                        [x0 - (a + c), y0 - (a + c), z0 - b],
                        [x0 + (a + c), y0 + (a + c), z0 + b],
                    ))
                } else {
                    None
                }
            }
            SurfaceKind::XTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => {
                if halfspace_below {
                    Some((
                        [x0 - b, y0 - (a + c), z0 - (a + c)],
                        [x0 + b, y0 + (a + c), z0 + (a + c)],
                    ))
                } else {
                    None
                }
            }
            SurfaceKind::YTorus {
                x0,
                y0,
                z0,
                a,
                b,
                c,
            } => {
                if halfspace_below {
                    Some((
                        [x0 - (a + c), y0 - b, z0 - (a + c)],
                        [x0 + (a + c), y0 + b, z0 + (a + c)],
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Get the constraint this surface imposes on axis-aligned bounds when used as a halfspace.
    /// Returns (axis_index, is_upper_bound, value) or None if no axis constraint.
    pub fn axis_constraint(&self, halfspace_above: bool) -> Option<(usize, bool, f64)> {
        match &self.kind {
            SurfaceKind::Plane { a, b, c, d } => {
                plane::axis_constraint(halfspace_above, *a, *b, *c, *d)
            }
            _ => None,
        }
    }

    /// Compute a [`BoundingBox`] for the halfspace defined by this surface.
    ///
    /// Combines `axis_constraint` (for axis-aligned planes) and `bounding_box`
    /// (for spheres, cylinders, tori) into a single result.  Bounds that cannot
    /// be determined are set to ±infinity.
    pub fn halfspace_bounding_box(&self, is_above: bool) -> crate::bounding_box::BoundingBox {
        let mut lower = [f64::NEG_INFINITY; 3];
        let mut upper = [f64::INFINITY; 3];

        // Apply axis constraint (axis-aligned planes)
        if let Some((axis, is_upper, value)) = self.axis_constraint(is_above) {
            if is_upper {
                upper[axis] = value;
            } else {
                lower[axis] = value;
            }
        }

        // Apply finite bounding box (spheres, cylinders, tori)
        let halfspace_below = !is_above;
        if let Some((bb_lower, bb_upper)) = self.bounding_box(halfspace_below) {
            for i in 0..3 {
                lower[i] = lower[i].max(bb_lower[i]);
                upper[i] = upper[i].min(bb_upper[i]);
            }
        }

        crate::bounding_box::BoundingBox::new(lower, upper)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Halfspace {
    pub expr: RegionExpr,
}

impl Halfspace {
    pub fn new_above(surface: Arc<Surface>) -> Self {
        Halfspace {
            expr: RegionExpr::Halfspace(HalfspaceType::Above(surface)),
        }
    }

    pub fn new_below(surface: Arc<Surface>) -> Self {
        Halfspace {
            expr: RegionExpr::Halfspace(HalfspaceType::Below(surface)),
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── XCylinder tests ──

    // ── YCylinder tests ──

    // ── arbitrary-axis cylinder tests ──

    // ── ZTorus tests ──

    // ── BoundaryType tests ──

    #[test]
    fn boundary_default_is_transmission() {
        let bt: BoundaryType = Default::default();
        assert_eq!(bt, BoundaryType::Transmission);
    }

    #[test]
    fn boundary_from_str_option_valid() {
        assert_eq!(
            BoundaryType::from_str_option("transmission"),
            Some(BoundaryType::Transmission)
        );
        assert_eq!(
            BoundaryType::from_str_option("vacuum"),
            Some(BoundaryType::Vacuum)
        );
    }

    #[test]
    fn boundary_from_str_option_invalid() {
        assert_eq!(BoundaryType::from_str_option("Vacuum"), None);
        assert_eq!(BoundaryType::from_str_option("TRANSMISSION"), None);
        assert_eq!(BoundaryType::from_str_option("reflective"), None);
        assert_eq!(BoundaryType::from_str_option(""), None);
    }

    #[test]
    fn boundary_clone_and_eq() {
        let a = BoundaryType::Vacuum;
        let b = a.clone();
        assert_eq!(a, b);
        assert_ne!(BoundaryType::Transmission, BoundaryType::Vacuum);
    }

    // ── Surface ID and boundary type getter/setter tests ──

    #[test]
    fn surface_get_set_surface_id() {
        let mut s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        assert_eq!(s.get_surface_id(), None);
        s.set_surface_id(42);
        assert_eq!(s.get_surface_id(), Some(42));
    }

    #[test]
    fn surface_boundary_getter_setter() {
        let mut s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        assert_eq!(*s.boundary(), BoundaryType::Transmission);
        s.set_boundary(BoundaryType::Vacuum);
        assert_eq!(*s.boundary(), BoundaryType::Vacuum);
    }

    #[test]
    fn surface_constructed_with_vacuum_boundary() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 5.0, Some(1), Some(BoundaryType::Vacuum));
        assert_eq!(*s.boundary(), BoundaryType::Vacuum);
        assert_eq!(s.get_surface_id(), Some(1));
    }

    // ── General Plane tests ──

    // ── Axis-aligned plane tests ──

    // ── Sphere tests ──

    // ── ZCylinder tests ──

    // ── Cylinder with individual components ──

    // ── Bounding box tests ──

    // ── axis_constraint tests ──

    #[test]
    fn axis_constraint_x_plane_above() {
        let plane = Surface::x_plane(5.0, None, None);
        // halfspace_above=true: x > 5 => lower bound
        let constraint = plane.axis_constraint(true);
        assert!(constraint.is_some());
        let (axis, is_upper, val) = constraint.unwrap();
        assert_eq!(axis, 0);
        assert!(!is_upper); // lower bound
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn axis_constraint_x_plane_below() {
        let plane = Surface::x_plane(5.0, None, None);
        // halfspace_above=false: x < 5 => upper bound
        let constraint = plane.axis_constraint(false);
        assert!(constraint.is_some());
        let (axis, is_upper, val) = constraint.unwrap();
        assert_eq!(axis, 0);
        assert!(is_upper); // upper bound
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn axis_constraint_y_plane() {
        let plane = Surface::y_plane(-3.0, None, None);
        let constraint = plane.axis_constraint(true);
        assert!(constraint.is_some());
        let (axis, is_upper, val) = constraint.unwrap();
        assert_eq!(axis, 1);
        assert!(!is_upper);
        assert!((val - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn axis_constraint_z_plane() {
        let plane = Surface::z_plane(10.0, None, None);
        let constraint = plane.axis_constraint(false);
        assert!(constraint.is_some());
        let (axis, is_upper, val) = constraint.unwrap();
        assert_eq!(axis, 2);
        assert!(is_upper);
        assert!((val - 10.0).abs() < 1e-10);
    }

    #[test]
    fn axis_constraint_general_plane_is_none() {
        // A non-axis-aligned plane should return None
        let plane = Surface::new_plane(1.0, 1.0, 0.0, 5.0, None, None);
        assert!(plane.axis_constraint(true).is_none());
        assert!(plane.axis_constraint(false).is_none());
    }

    #[test]
    fn axis_constraint_sphere_is_none() {
        let s = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None);
        assert!(s.axis_constraint(true).is_none());
        assert!(s.axis_constraint(false).is_none());
    }

    #[test]
    fn axis_constraint_cylinder_is_none() {
        let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, None, None);
        assert!(cyl.axis_constraint(true).is_none());
    }

    #[test]
    fn axis_constraint_torus_is_none() {
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, 3.0, 1.0, 1.0, None, None);
        assert!(torus.axis_constraint(true).is_none());
    }

    // ── Halfspace tests ──

    #[test]
    fn halfspace_new_above_and_below() {
        let surface = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None));
        let above = Halfspace::new_above(Arc::clone(&surface));
        let below = Halfspace::new_below(Arc::clone(&surface));
        // Just verify they can be created without panic
        match &above.expr {
            RegionExpr::Halfspace(HalfspaceType::Above(_)) => {}
            _ => panic!("Expected Above halfspace"),
        }
        match &below.expr {
            RegionExpr::Halfspace(HalfspaceType::Below(_)) => {}
            _ => panic!("Expected Below halfspace"),
        }
    }

    // ── Torus with offset center ──

    // ── Cylinder offset center distance ──

    // ── Cylinder miss (perpendicular but offset) ──

    // ── Elliptical torus distance ──

    // ── General plane bounding box ──

    // ── Torus bounding box with offset ──

    // -- Quadric tests --

    // -- Cone tests --

    // -- X/Y torus tests --

    #[test]
    fn xy_torus_bounding_boxes_permute() {
        let zt = Surface::new_ztorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None);
        let xt = Surface::new_xtorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None);
        let yt = Surface::new_ytorus(1.0, 2.0, 3.0, 4.0, 0.5, 0.8, None, None);
        let (zlo, zhi) = zt.bounding_box(true).unwrap();
        let (xlo, xhi) = xt.bounding_box(true).unwrap();
        let (ylo, yhi) = yt.bounding_box(true).unwrap();
        // ZTorus: tight along z (+-b); XTorus tight along x; YTorus along y.
        assert_eq!(zhi[2] - zlo[2], 1.0);
        assert_eq!(xhi[0] - xlo[0], 1.0);
        assert_eq!(yhi[1] - ylo[1], 1.0);
        assert_eq!(zhi[0] - zlo[0], 9.6); // 2 (a + c)
        assert_eq!(xhi[1] - xlo[1], 9.6);
        assert_eq!(yhi[0] - ylo[0], 9.6);
    }

    // ── Loading a Cylinder or Cone axis ──

    fn load(kind: &str) -> Result<Surface, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"surface_id": null, "kind": {kind}, "boundary": "Transmission"}}"#
        ))
    }

    fn axis(s: &Surface) -> [f64; 3] {
        match s.kind {
            SurfaceKind::Cylinder { axis, .. } | SurfaceKind::Cone { axis, .. } => axis,
            _ => unreachable!(),
        }
    }

    #[test]
    fn loading_makes_a_cylinder_or_cone_axis_a_unit_vector() {
        // A radius-1 cylinder and a 45 degree cone about z, written with axes
        // twice the unit length.
        let cylinder =
            load(r#"{"Cylinder": {"axis": [0, 0, 2], "origin": [0, 0, 0], "radius": 1}}"#).unwrap();
        let cone =
            load(r#"{"Cone": {"apex": [0, 0, 0], "axis": [0, 0, -2], "tan2_theta": 1}}"#).unwrap();
        assert_eq!(axis(&cylinder), [0.0, 0.0, 1.0]);
        assert_eq!(axis(&cone), [0.0, 0.0, -1.0]);
        assert_eq!(cylinder.evaluate((0.5, 0.0, 1.0)), -0.5);
        assert_eq!(cone.evaluate((1.0, 0.0, 1.0)), 0.0);
    }

    #[test]
    fn a_saved_cylinder_axis_loads_bit_for_bit() {
        // [1, 1, 0] / sqrt(2) is an ulp short of unit length, so normalising
        // it again would move it.
        let saved = Surface::new_cylinder([1.0, 1.0, 0.0], [0.0; 3], 1.0, None, None);
        let a = axis(&saved);
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        assert_ne!([a[0] / n, a[1] / n, a[2] / n], a);
        let loaded: Surface =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(axis(&loaded), a);
    }

    #[test]
    fn loading_refuses_a_zero_cylinder_or_cone_axis() {
        for kind in [
            r#"{"Cylinder": {"axis": [0, 0, 0], "origin": [0, 0, 0], "radius": 1}}"#,
            r#"{"Cone": {"apex": [0, 0, 0], "axis": [0, 0, 0], "tan2_theta": 1}}"#,
        ] {
            let error = load(kind).unwrap_err().to_string();
            assert!(
                error.contains("axis must be a non-zero, finite vector"),
                "{error}"
            );
        }
    }
}
