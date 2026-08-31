//! Constructive-solid-geometry regions.
//!
//! [`Region`] is the high-level CSG type (an expression tree of halfspace
//! unions/intersections/complements). The expression tree lives in [`expr`];
//! its compiled, allocation-free RPN form for the transport hot loop lives in
//! [`flat`]. Both are re-exported here so `crate::region::<T>` paths stay stable.

use crate::surface::Surface;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod expr;
pub mod flat;

pub use expr::{HalfspaceType, RegionExpr};
pub use flat::{FlatRegion, RegionOp};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub expr: RegionExpr,
}

// Regular Rust implementation
// Regular Rust implementation
impl Region {
    /// Recursively collect all surfaces and their sense (true=Above, false=Below) in the region
    /// Fill a vector with surfaces and their sense (more efficient - reuses allocation)
    pub fn collect_surfaces_with_sense(&self, surfaces: &mut Vec<(Arc<Surface>, bool)>) {
        fn collect(expr: &RegionExpr, surfaces: &mut Vec<(Arc<Surface>, bool)>, sense: bool) {
            match expr {
                RegionExpr::Halfspace(hs) => match hs {
                    HalfspaceType::Above(surf) => surfaces.push((surf.clone(), sense)),
                    HalfspaceType::Below(surf) => surfaces.push((surf.clone(), !sense)),
                },
                RegionExpr::Union(a, b) | RegionExpr::Intersection(a, b) => {
                    collect(a, surfaces, sense);
                    collect(b, surfaces, sense);
                }
                RegionExpr::Complement(inner) => collect(inner, surfaces, !sense),
            }
        }
        surfaces.clear();
        collect(&self.expr, surfaces, true);
    }

    pub fn surfaces_with_sense(&self) -> Vec<(Arc<Surface>, bool)> {
        let mut result = Vec::new();
        self.collect_surfaces_with_sense(&mut result);
        result
    }

    /// Check if crossing the given surface at the intersection would exit the region
    /// For unions: exit only if outside all subregions. For intersections: exit if outside any subregion.
    pub fn is_exit_surface(
        &self,
        point: (f64, f64, f64),
        direction: (f64, f64, f64),
        _surface: &Surface,
        dist: f64,
        _sense: bool,
    ) -> bool {
        let eps = 1e-8;
        let p_next = (
            point.0 + direction.0 * (dist + eps),
            point.1 + direction.1 * (dist + eps),
            point.2 + direction.2 * (dist + eps),
        );
        fn check_exit(expr: &RegionExpr, p: (f64, f64, f64)) -> bool {
            match expr {
                RegionExpr::Halfspace(hs) => match hs {
                    HalfspaceType::Above(surf) => surf.evaluate(p) <= 0.0,
                    HalfspaceType::Below(surf) => surf.evaluate(p) >= 0.0,
                },
                RegionExpr::Union(a, b) => check_exit(a, p) && check_exit(b, p), // exit union if outside all
                RegionExpr::Intersection(a, b) => check_exit(a, p) || check_exit(b, p), // exit intersection if outside any
                RegionExpr::Complement(inner) => !check_exit(inner, p),
            }
        }
        check_exit(&self.expr, p_next)
    }
    pub fn new_from_halfspace(halfspace_type: HalfspaceType) -> Self {
        Region {
            expr: RegionExpr::Halfspace(halfspace_type),
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        Region {
            expr: RegionExpr::Intersection(
                Box::new(self.expr.clone()),
                Box::new(other.expr.clone()),
            ),
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        Region {
            expr: RegionExpr::Union(Box::new(self.expr.clone()), Box::new(other.expr.clone())),
        }
    }

    pub fn complement(&self) -> Self {
        Region {
            expr: RegionExpr::Complement(Box::new(self.expr.clone())),
        }
    }

    /// Compile the region expression into a flat RPN representation for
    /// iterative evaluation. Call once at setup time; use the returned
    /// `FlatRegion` for all hot-path containment tests.
    pub fn flatten(&self) -> FlatRegion {
        FlatRegion::from_expr(&self.expr)
    }

    // Updated contains method: no surface dictionary needed
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        self.expr.evaluate_contains(point)
    }

    // Updated evaluate_contains method: no surface dictionary needed
    pub fn evaluate_contains(&self, point: (f64, f64, f64)) -> bool {
        self.expr.evaluate_contains(point)
    }

    /// Axis-aligned bounding box that ENCLOSES this region.
    ///
    /// Computed recursively over the CSG expression so unions and complements
    /// compose correctly:
    /// - a half-space contributes its own [`Surface::halfspace_bounding_box`]
    ///   (axis constraints for axis-aligned planes; finite boxes for spheres /
    ///   cylinders / tori; `±inf` otherwise);
    /// - an `Intersection` is the [`intersection`](crate::bounding_box::BoundingBox::intersection)
    ///   of the child boxes;
    /// - a `Union` is the enclosing [`union`](crate::bounding_box::BoundingBox::union)
    ///   of the child boxes (a union may be disconnected, so the hull is the
    ///   smallest enclosing box);
    /// - a `Complement` is unbounded (`±inf`): the complement of a bounded
    ///   region has no finite bound; a surrounding intersection with a bounded
    ///   region re-tightens it.
    ///
    /// The box is a CONSERVATIVE bound, not an exact extent: for a ring/frame
    /// (`outer & (x<a | x>b | ...)`) it is the outer rectangle, which also
    /// covers the hole. Callers confirm membership with the exact CSG predicate
    /// ([`contains`](Self::contains)), so an over-approximating-but-enclosing
    /// box is always sound.
    ///
    /// A genuinely unbounded region (implicit complement, e.g. `Above(sphere)`
    /// or `~region`) or a genuinely empty one returns a non-finite box (see
    /// [`BoundingBox::is_finite`](crate::bounding_box::BoundingBox::is_finite)),
    /// preserving the sentinel behaviour the CPU BVH fallback and the GPU
    /// implicit-complement handling rely on.
    pub fn bounding_box(&self) -> crate::bounding_box::BoundingBox {
        use crate::bounding_box::BoundingBox;
        fn bbox(expr: &RegionExpr) -> BoundingBox {
            match expr {
                RegionExpr::Halfspace(hs) => match hs {
                    HalfspaceType::Above(surf) => surf.halfspace_bounding_box(true),
                    HalfspaceType::Below(surf) => surf.halfspace_bounding_box(false),
                },
                RegionExpr::Intersection(a, b) => bbox(a).intersection(&bbox(b)),
                RegionExpr::Union(a, b) => bbox(a).union(&bbox(b)),
                RegionExpr::Complement(_inner) => {
                    // The complement of a bounded region is unbounded; be
                    // conservative. An enclosing intersection re-tightens it.
                    BoundingBox::new([f64::NEG_INFINITY; 3], [f64::INFINITY; 3])
                }
            }
        }
        bbox(&self.expr)
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
#[test]
fn test_sphere_bb_moved_on_z_axis() {
    use crate::region::{HalfspaceType, Region};
    use crate::surface::Surface;
    use std::sync::Arc;
    // Sphere centered at (0, 0, 1) with radius 3
    let s2 = Surface::new_sphere(0.0, 0.0, 1.0, 3.0, Some(1), None);
    let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2)));
    let bbox = region2.bounding_box();
    assert_eq!(bbox.lower_left, [-3.0, -3.0, -2.0]);
    assert_eq!(bbox.upper_right, [3.0, 3.0, 4.0]);
}

#[test]
fn test_sphere_with_xplanes() {
    use crate::region::{HalfspaceType, Region};
    use crate::surface::Surface;
    use std::sync::Arc;
    // XPlane at x=2.1
    let s1 = Surface::x_plane(2.1, Some(5), None);
    // XPlane at x=-2.1
    let s2 = Surface::x_plane(-2.1, Some(6), None);
    // Sphere at (0,0,0) with radius 4.2
    let s3 = Surface::new_sphere(0.0, 0.0, 0.0, 4.2, Some(1), None);
    // Region: x <= 2.1 & x >= -2.1 & inside sphere
    let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1.clone())))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Above(Arc::new(
            s2.clone(),
        ))))
        .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
            s3.clone(),
        ))));
    let bbox = region1.bounding_box();
    assert_eq!(bbox.lower_left, [-2.1, -4.2, -4.2]);
    assert_eq!(bbox.upper_right, [2.1, 4.2, 4.2]);
}
mod tests {
    // Imports moved into individual test functions to avoid unused import warnings

    #[test]
    fn test_region_contains() {
        use super::Region;
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // Create two surfaces
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Plane {
                a: 0.0,
                b: 0.0,
                c: 1.0,
                d: -5.0,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 3.0,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };

        // Map of surfaces by surface_id
        let mut surfaces = HashMap::new();
        surfaces.insert(s1.surface_id, s1.clone());
        surfaces.insert(s2.surface_id, s2.clone());

        // Build a region: inside s2 AND above s1
        let region =
            Region::new_from_halfspace(crate::region::HalfspaceType::Above(Arc::new(s1.clone())))
                .intersection(&Region::new_from_halfspace(
                    crate::region::HalfspaceType::Below(Arc::new(s2.clone())),
                ));

        // Test a point inside both
        let point = (0.0, 0.0, 0.0);
        assert!(region.contains(point));

        // Test a point outside the sphere
        let point = (0.0, 0.0, 4.0);
        assert!(!region.contains(point));
    }

    #[test]
    fn test_sphere_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // Sphere of radius 2 at (0,0,0)
        let s = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let mut surfaces = HashMap::new();
        surfaces.insert(s.surface_id, s.clone());
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s.clone())));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left, [-2.0, -2.0, -2.0]);
        assert_eq!(bbox.upper_right, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_box_and_sphere_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // XPlanes at x=2.1 and x=-2.1, sphere at origin with radius 4.2
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Plane {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 2.1,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Plane {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: -2.1,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let s3 = Surface {
            surface_id: Some(3),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 4.2,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let mut surfaces = HashMap::new();
        surfaces.insert(s1.surface_id, s1.clone());
        surfaces.insert(s2.surface_id, s2.clone());
        surfaces.insert(s3.surface_id, s3.clone());
        // Region: x >= -2.1 & x <= 2.1 & inside sphere
        let region = Region::new_from_halfspace(HalfspaceType::Above(Arc::new(s2.clone())))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
                s1.clone(),
            ))))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
                s3.clone(),
            ))));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left, [-2.1, -4.2, -4.2]);
        assert_eq!(bbox.upper_right, [2.1, 4.2, 4.2]);
    }

    #[test]
    fn test_zplane_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // ZPlane at z=3.5
        let s = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Plane {
                a: 0.0,
                b: 0.0,
                c: 1.0,
                d: 3.5,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let mut surfaces = HashMap::new();
        surfaces.insert(s.surface_id, s.clone());
        // Region: z < 3.5 (Below ZPlane)
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s.clone())));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left[2], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[2], 3.5);
        assert_eq!(bbox.lower_left[0], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[0], f64::INFINITY);
        assert_eq!(bbox.lower_left[1], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[1], f64::INFINITY);
    }

    #[test]
    fn test_xplane_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // XPlane at x=1.5
        let s = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Plane {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 1.5,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let mut surfaces = HashMap::new();
        surfaces.insert(s.surface_id, s.clone());
        // Region: x < 1.5 (Below XPlane)
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s.clone())));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left[0], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[0], 1.5);
        assert_eq!(bbox.lower_left[1], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[1], f64::INFINITY);
        assert_eq!(bbox.lower_left[2], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[2], f64::INFINITY);
    }

    #[test]
    fn test_yplane_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::collections::HashMap;
        use std::sync::Arc;
        // YPlane at y=-2.0
        let s = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Plane {
                a: 0.0,
                b: 1.0,
                c: 0.0,
                d: -2.0,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        let mut surfaces = HashMap::new();
        surfaces.insert(s.surface_id, s.clone());
        // Region: y > -2.0 (Above YPlane)
        let region = Region::new_from_halfspace(HalfspaceType::Above(Arc::new(s.clone())));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left[1], -2.0);
        assert_eq!(bbox.upper_right[1], f64::INFINITY);
        assert_eq!(bbox.lower_left[0], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[0], f64::INFINITY);
        assert_eq!(bbox.lower_left[2], f64::NEG_INFINITY);
        assert_eq!(bbox.upper_right[2], f64::INFINITY);
    }

    #[test]
    fn test_zcylinder_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::{Surface, SurfaceKind};
        use std::sync::Arc;
        // Z-cylinder at (1, 2) with radius 3
        let s = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Cylinder {
                axis: [0.0, 0.0, 1.0],
                origin: [1.0, 2.0, 0.0],
                radius: 3.0,
            },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        };
        // Region: inside cylinder (Below)
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s.clone())));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left, [-2.0, -1.0, f64::NEG_INFINITY]);
        assert_eq!(bbox.upper_right, [4.0, 5.0, f64::INFINITY]);
    }

    #[test]
    fn test_flat_region_halfspace() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        let sphere = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 5.0, Some(1), None));
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere));
        let flat = region.flatten();

        // Inside sphere
        assert!(flat.contains((0.0, 0.0, 0.0)));
        assert!(flat.contains((3.0, 0.0, 0.0)));
        // On boundary (evaluate < 0 is false at r=5)
        assert!(!flat.contains((5.0, 0.0, 0.0)));
        // Outside sphere
        assert!(!flat.contains((6.0, 0.0, 0.0)));
    }

    #[test]
    fn test_flat_region_intersection() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        // Intersection of z > -5 and inside sphere(r=3)
        let plane = Arc::new(Surface::z_plane(-5.0, Some(1), None));
        let sphere = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 3.0, Some(2), None));
        let region = Region::new_from_halfspace(HalfspaceType::Above(plane))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(sphere)));
        let flat = region.flatten();

        // Inside both
        assert!(flat.contains((0.0, 0.0, 0.0)));
        // Inside sphere but below plane
        assert!(!flat.contains((0.0, 0.0, -6.0)));
        // Above plane but outside sphere
        assert!(!flat.contains((4.0, 0.0, 0.0)));
    }

    // Axis-aligned plane `a*x+b*y+c*z = d`; `Above` => coord >= d, `Below` =>
    // coord <= d (matches the existing bounding-box tests above).
    #[cfg(test)]
    fn axis_plane(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        id: usize,
    ) -> std::sync::Arc<crate::surface::Surface> {
        std::sync::Arc::new(crate::surface::Surface {
            surface_id: Some(id),
            kind: crate::surface::SurfaceKind::Plane { a, b, c, d },
            boundary: crate::surface::BoundaryType::default(),
            name: None,
        })
    }

    // Axis-aligned box [-h, h]^3 built from six planes.
    #[cfg(test)]
    fn cube_region(h: f64) -> super::Region {
        use super::{HalfspaceType, Region};
        Region::new_from_halfspace(HalfspaceType::Above(axis_plane(1.0, 0.0, 0.0, -h, 101)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                axis_plane(1.0, 0.0, 0.0, h, 102),
            )))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                axis_plane(0.0, 1.0, 0.0, -h, 103),
            )))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                axis_plane(0.0, 1.0, 0.0, h, 104),
            )))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(
                axis_plane(0.0, 0.0, 1.0, -h, 105),
            )))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                axis_plane(0.0, 0.0, 1.0, h, 106),
            )))
    }

    // Issue #272: a ring/frame region (`outer_box & (x<-a | x>a | y<-a | y>a)`)
    // is geometrically finite (the outer box bounds it), so its bounding box
    // must be the finite ENCLOSING box -- not the empty/inverted box the old
    // flatten-based algorithm produced by treating the union as an intersection.
    #[test]
    fn test_ring_via_union_bounding_box() {
        use super::{HalfspaceType, Region};
        // outside the inner room: x < -5 | x > 5 | y < -5 | y > 5
        let outside =
            Region::new_from_halfspace(HalfspaceType::Below(axis_plane(1.0, 0.0, 0.0, -5.0, 201)))
                .union(&Region::new_from_halfspace(HalfspaceType::Above(
                    axis_plane(1.0, 0.0, 0.0, 5.0, 202),
                )))
                .union(&Region::new_from_halfspace(HalfspaceType::Below(
                    axis_plane(0.0, 1.0, 0.0, -5.0, 203),
                )))
                .union(&Region::new_from_halfspace(HalfspaceType::Above(
                    axis_plane(0.0, 1.0, 0.0, 5.0, 204),
                )));
        let ring = cube_region(10.0).intersection(&outside);
        let bb = ring.bounding_box();
        assert!(bb.is_finite(), "ring bbox must be finite, got {bb:?}");
        assert_eq!(bb.lower_left, [-10.0, -10.0, -10.0]);
        assert_eq!(bb.upper_right, [10.0, 10.0, 10.0]);
        // The enclosing box covers the hole, but membership excludes it.
        assert!(
            !ring.contains((0.0, 0.0, 0.0)),
            "hole centre is not in the ring"
        );
        assert!(
            ring.contains((-8.0, 0.0, 0.0)),
            "wall interior is in the ring"
        );
    }

    // A frame built with a complement (`outer_box & ~inner_sphere`) is also
    // finite; the complement contributes ±inf and the outer box re-tightens.
    #[test]
    fn test_frame_via_complement_bounding_box() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;
        let inner = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
            Surface::new_sphere(0.0, 0.0, 0.0, 5.0, Some(301), None),
        )));
        let frame = cube_region(10.0).intersection(&inner.complement());
        let bb = frame.bounding_box();
        assert!(bb.is_finite(), "frame bbox must be finite, got {bb:?}");
        assert_eq!(bb.lower_left, [-10.0, -10.0, -10.0]);
        assert_eq!(bb.upper_right, [10.0, 10.0, 10.0]);
        assert!(
            !frame.contains((0.0, 0.0, 0.0)),
            "inner-sphere centre excluded"
        );
        assert!(
            frame.contains((8.0, 0.0, 0.0)),
            "outside sphere, inside box"
        );
    }

    // A union of two disjoint spheres has an enclosing box spanning both.
    #[test]
    fn test_union_bounding_box_encloses_both() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;
        let s1 = Arc::new(Surface::new_sphere(-3.0, 0.0, 0.0, 2.0, Some(1), None));
        let s2 = Arc::new(Surface::new_sphere(3.0, 0.0, 0.0, 2.0, Some(2), None));
        let region = Region::new_from_halfspace(HalfspaceType::Below(s1))
            .union(&Region::new_from_halfspace(HalfspaceType::Below(s2)));
        let bb = region.bounding_box();
        assert!(bb.is_finite());
        assert_eq!(bb.lower_left, [-5.0, -2.0, -2.0]);
        assert_eq!(bb.upper_right, [5.0, 2.0, 2.0]);
    }

    // A genuinely empty intersection still reports the non-finite sentinel.
    #[test]
    fn test_empty_intersection_bounding_box_is_non_finite() {
        use super::{HalfspaceType, Region};
        // x >= 5 AND x <= -5 : no points.
        let empty =
            Region::new_from_halfspace(HalfspaceType::Above(axis_plane(1.0, 0.0, 0.0, 5.0, 401)))
                .intersection(&Region::new_from_halfspace(HalfspaceType::Below(
                    axis_plane(1.0, 0.0, 0.0, -5.0, 402),
                )));
        assert!(!empty.bounding_box().is_finite());
    }

    #[test]
    fn test_flat_region_union() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        // Union of two spheres at different positions
        let s1 = Arc::new(Surface::new_sphere(-3.0, 0.0, 0.0, 2.0, Some(1), None));
        let s2 = Arc::new(Surface::new_sphere(3.0, 0.0, 0.0, 2.0, Some(2), None));
        let region = Region::new_from_halfspace(HalfspaceType::Below(s1))
            .union(&Region::new_from_halfspace(HalfspaceType::Below(s2)));
        let flat = region.flatten();

        // Inside first sphere
        assert!(flat.contains((-3.0, 0.0, 0.0)));
        // Inside second sphere
        assert!(flat.contains((3.0, 0.0, 0.0)));
        // Outside both
        assert!(!flat.contains((0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_flat_region_complement() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        // Complement of inside sphere = outside sphere
        let sphere = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 3.0, Some(1), None));
        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere)).complement();
        let flat = region.flatten();

        // Origin is inside sphere -> outside complement
        assert!(!flat.contains((0.0, 0.0, 0.0)));
        // Far away is outside sphere -> inside complement
        assert!(flat.contains((10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_flat_region_nested_intersection() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        // 6-plane box: -1 < x < 1, -1 < y < 1, -1 < z < 1
        let xlo = Arc::new(Surface::x_plane(-1.0, Some(1), None));
        let xhi = Arc::new(Surface::x_plane(1.0, Some(2), None));
        let ylo = Arc::new(Surface::y_plane(-1.0, Some(3), None));
        let yhi = Arc::new(Surface::y_plane(1.0, Some(4), None));
        let zlo = Arc::new(Surface::z_plane(-1.0, Some(5), None));
        let zhi = Arc::new(Surface::z_plane(1.0, Some(6), None));

        let region = Region::new_from_halfspace(HalfspaceType::Above(xlo))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(xhi)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(ylo)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(yhi)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(zlo)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(zhi)));
        let flat = region.flatten();

        assert!(flat.contains((0.0, 0.0, 0.0)));
        assert!(flat.contains((0.5, 0.5, 0.5)));
        assert!(!flat.contains((1.5, 0.0, 0.0)));
        assert!(!flat.contains((0.0, -1.5, 0.0)));
        assert!(!flat.contains((0.0, 0.0, 2.0)));

        // Compare with recursive
        for p in &[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.5, 0.0, 0.0),
            (0.0, -1.5, 0.0),
        ] {
            assert_eq!(flat.contains(*p), region.contains(*p));
        }
    }

    #[test]
    fn test_flat_matches_recursive() {
        use super::{HalfspaceType, Region};
        use crate::surface::Surface;
        use std::sync::Arc;

        // Complex region: (inside sphere & above z-plane) | complement(inside cylinder)
        let sphere = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 5.0, Some(1), None));
        let zplane = Arc::new(Surface::z_plane(-2.0, Some(2), None));
        let cyl = Arc::new(Surface::new_cylinder(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            1.0,
            Some(3),
            None,
        ));

        let region = Region::new_from_halfspace(HalfspaceType::Below(sphere))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(zplane)))
            .union(&Region::new_from_halfspace(HalfspaceType::Below(cyl)).complement());
        let flat = region.flatten();

        // Test a grid of points and compare recursive vs flat
        let test_points = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.0, 0.0, -5.0),
            (3.0, 3.0, 3.0),
            (0.5, 0.0, 0.0),    // inside cylinder
            (2.0, 0.0, 0.0),    // outside cylinder
            (10.0, 10.0, 10.0), // far outside
            (0.0, 0.0, -3.0),   // below z-plane
            (4.9, 0.0, 0.0),    // near sphere boundary
        ];
        for point in &test_points {
            assert_eq!(
                flat.contains(*point),
                region.contains(*point),
                "Mismatch at point {:?}: flat={}, recursive={}",
                point,
                flat.contains(*point),
                region.contains(*point)
            );
        }
    }
}
