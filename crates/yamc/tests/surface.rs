mod tests {
    use std::sync::Arc;
    use yamc::geo::*;
    use yamc::geo::{HalfspaceType, Region};

    #[test]
    fn test_zcylinder_with_zplanes_bounding_box() {
        // Z-cylinder centered at (1, 2) with radius 3
        let cyl = Surface::z_cylinder(1.0, 2.0, 3.0, Some(1), None);
        // Z planes at z = -5 and z = 5
        let z_bottom = Surface::z_plane(-5.0, Some(2), None);
        let z_top = Surface::z_plane(5.0, Some(3), None);

        // Region: inside cylinder and between Z planes
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(cyl)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(Arc::new(
                z_bottom,
            ))))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
                z_top,
            ))));
        let bbox = region.bounding_box();
        assert_eq!(bbox.lower_left, [-2.0, -1.0, -5.0]);
        assert_eq!(bbox.upper_right, [4.0, 5.0, 5.0]);
    }

    #[test]
    fn test_plane_creation() {
        let plane = Surface::new_plane(1.0, 0.0, 0.0, 2.0, Some(42), None);
        match plane.kind {
            SurfaceKind::Plane { a, b, c, d } => {
                assert_eq!(a, 1.0);
                assert_eq!(b, 0.0);
                assert_eq!(c, 0.0);
                assert_eq!(d, 2.0);
            }
            _ => panic!("Not a plane"),
        }
        assert_eq!(plane.surface_id, Some(42));
    }

    #[test]
    fn test_sphere_creation() {
        let sphere = Surface::new_sphere(1.0, 2.0, 3.0, 5.0, Some(7), None);
        match sphere.kind {
            SurfaceKind::Sphere { x0, y0, z0, radius } => {
                assert_eq!(x0, 1.0);
                assert_eq!(y0, 2.0);
                assert_eq!(z0, 3.0);
                assert_eq!(radius, 5.0);
            }
            _ => panic!("Not a sphere"),
        }
        assert_eq!(sphere.surface_id, Some(7));
    }

    #[test]
    fn test_cylinder_creation() {
        let axis = [0.0, 1.0, 0.0];
        let origin = [1.0, 2.0, 3.0];
        let cylinder = Surface::new_cylinder(axis, origin, 2.0, Some(99), None);
        match cylinder.kind {
            SurfaceKind::Cylinder {
                axis: a,
                origin: o,
                radius,
            } => {
                assert_eq!(a, axis);
                assert_eq!(o, origin);
                assert_eq!(radius, 2.0);
            }
            _ => panic!("Not a cylinder"),
        }
        assert_eq!(cylinder.surface_id, Some(99));
    }

    #[test]
    fn test_z_cylinder_creation() {
        let zcyl = Surface::z_cylinder(1.0, 2.0, 3.0, Some(123), None);
        match zcyl.kind {
            SurfaceKind::Cylinder {
                axis,
                origin,
                radius,
            } => {
                assert_eq!(axis, [0.0, 0.0, 1.0]);
                assert_eq!(origin, [1.0, 2.0, 0.0]);
                assert_eq!(radius, 3.0);
            }
            _ => panic!("Not a Z cylinder"),
        }
        assert_eq!(zcyl.surface_id, Some(123));
    }

    #[test]
    fn test_boundary_default() {
        let plane = Surface::new_plane(1.0, 0.0, 0.0, 2.0, Some(42), None);
        assert_eq!(*plane.boundary(), BoundaryType::Transmission);
    }

    #[test]
    fn test_boundary_vacuum() {
        let sphere = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, Some(1), Some(BoundaryType::Vacuum));
        assert_eq!(*sphere.boundary(), BoundaryType::Vacuum);
    }

    #[test]
    fn test_set_boundary() {
        let mut cylinder =
            Surface::new_cylinder([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 1.0, Some(2), None);
        assert_eq!(*cylinder.boundary(), BoundaryType::Transmission);

        cylinder.set_boundary(BoundaryType::Vacuum);
        assert_eq!(*cylinder.boundary(), BoundaryType::Vacuum);

        cylinder.set_boundary(BoundaryType::Transmission);
        assert_eq!(*cylinder.boundary(), BoundaryType::Transmission);
    }

    #[test]
    fn test_zcylinder_bounding_box() {
        let zcyl = Surface::z_cylinder(1.0, 2.0, 3.0, Some(123), None);

        // Test inside cylinder (halfspace_below = true)
        let bbox = zcyl.bounding_box(true);
        assert!(bbox.is_some());
        let (lower, upper) = bbox.unwrap();
        assert_eq!(lower, [-2.0, -1.0, f64::NEG_INFINITY]);
        assert_eq!(upper, [4.0, 5.0, f64::INFINITY]);

        // Test outside cylinder (halfspace_below = false)
        let bbox_outside = zcyl.bounding_box(false);
        assert!(bbox_outside.is_none()); // Outside cylinder is infinite
    }

    #[test]
    fn test_quadric_creation_and_transport_use() {
        // Quadric coefficients survive construction (issue #366).
        let q = Surface::new_quadric(
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            Some(11),
            None,
        );
        match q.kind {
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
            } => {
                assert_eq!(
                    (a, b, c, d, e, f, g, h, j, k),
                    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
                );
            }
            _ => panic!("Not a quadric"),
        }
        assert_eq!(q.surface_id, Some(11));

        // An ellipsoid quadric has no finite bbox contribution (falls in
        // the unbounded BVH bucket) but must still evaluate and ray-trace.
        let unit_sphere_q = Surface::new_quadric(
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, None, None,
        );
        assert!(unit_sphere_q.bounding_box(true).is_none());
        assert!(unit_sphere_q.evaluate((0.0, 0.0, 0.0)) < 0.0);
        assert!(unit_sphere_q.evaluate((2.0, 0.0, 0.0)) > 0.0);
        let d = unit_sphere_q
            .distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            .unwrap();
        assert!((d - 1.0).abs() < 1e-12);
    }
}
