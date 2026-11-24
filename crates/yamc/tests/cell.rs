mod tests {
    // --- Surface distance tests ---
    mod distance_tests {
        use yamc::geo::Surface;

        #[test]
        fn test_sphere_distance() {
            let sphere = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, Some(1), None);
            // From (2,0,0) toward center
            let d = sphere.distance_to_surface([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
            assert!((d.unwrap() - 1.0).abs() < 1e-10);
            // From (0,0,0) outward
            let d2 = sphere.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert!((d2.unwrap() - 1.0).abs() < 1e-10);
            // No intersection
            let d3 = sphere.distance_to_surface([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d3, None);
            // On surface, outward
            let d4 = sphere.distance_to_surface([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d4, None);
        }

        #[test]
        fn test_cylinder_distance() {
            // Z-cylinder at (0,0), r=1
            let cyl = Surface::z_cylinder(0.0, 0.0, 1.0, Some(1), None);
            // From (2,0,0) toward center
            let d = cyl.distance_to_surface([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
            assert!((d.unwrap() - 1.0).abs() < 1e-10);
            // From (0,2,0) toward center
            let d2 = cyl.distance_to_surface([0.0, 2.0, 0.0], [0.0, -1.0, 0.0]);
            assert!((d2.unwrap() - 1.0).abs() < 1e-10);
            // From (0,0,0) radially outward
            let d3 = cyl.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert!((d3.unwrap() - 1.0).abs() < 1e-10);
            // No intersection
            let d4 = cyl.distance_to_surface([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d4, None);
            // On surface, outward
            let d5 = cyl.distance_to_surface([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d5, None);
        }

        #[test]
        fn test_xplane_distance() {
            let plane = Surface::x_plane(5.0, Some(1), None);
            // From (0,0,0) in +x direction
            let d = plane.distance_to_surface([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d, Some(5.0));
            // From (0,0,0) in -x direction
            let d2 = plane.distance_to_surface([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
            assert_eq!(d2, None);
            // From (10,0,0) in -x direction
            let d3 = plane.distance_to_surface([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
            assert_eq!(d3, Some(5.0));
            // Parallel direction
            let d4 = plane.distance_to_surface([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
            assert_eq!(d4, None);
            // On plane, outward
            let d5 = plane.distance_to_surface([5.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert_eq!(d5, None);
        }
    }
    #[test]
    fn test_cell_fill_material() {
        use std::sync::Arc;
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};
        use yamc::geometry::Geometry;
        use yamc_materials::Material;

        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));

        let mat = Material::new(std::collections::HashMap::new(), "atom", "sum", None).unwrap();
        let mat_arc = Arc::new(mat);
        let cell = Cell::new(Some(1), region, Some("filled".to_string()), Some(0));
        assert_eq!(cell.material_idx, Some(0));

        // Optional fill
        let cell2 = Cell::new(
            Some(2),
            cell.region.clone(),
            Some("empty".to_string()),
            None,
        );
        assert!(cell2.material_idx.is_none());

        // Geometry resolves the index to the material in its flat store.
        let geometry = Geometry::new(vec![cell, cell2], vec![mat_arc]).unwrap();
        let resolved = geometry.material_for(&geometry.cells[0]).unwrap();
        assert_eq!(resolved.nuclides.len(), 0);
        assert!(geometry.material_for(&geometry.cells[1]).is_none());
    }
    #[test]
    fn test_cell_union_region() {
        // Union of two spheres
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 3.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2)));
        let region = region1.union(&region2);
        let cell = Cell::new(Some(100), region, Some("union".to_string()), None);
        assert!(cell.contains((0.0, 0.0, 0.0))); // inside first sphere
        assert!(cell.contains((3.0, 0.0, 0.0))); // inside second sphere
        assert!(!cell.contains((6.0, 0.0, 0.0))); // outside both
    }

    #[test]
    fn test_cell_intersection_region() {
        // Intersection of two spheres
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 1.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2)));
        let region = region1.intersection(&region2);
        let cell = Cell::new(Some(101), region, Some("intersection".to_string()), None);
        assert!(cell.contains((0.0, 0.0, 0.0))); // inside both
        assert!(cell.contains((1.0, 0.0, 0.0))); // inside both
        assert!(!cell.contains((3.0, 0.0, 0.0))); // outside both
    }

    #[test]
    fn test_cell_complement_region() {
        // Complement of a sphere
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let region_complement = region.complement();
        let cell = Cell::new(
            Some(102),
            region_complement,
            Some("complement".to_string()),
            None,
        );
        assert!(!cell.contains((0.0, 0.0, 0.0))); // inside original sphere
        assert!(cell.contains((3.0, 0.0, 0.0))); // outside original sphere
    }
    #[test]
    fn test_cell_complex_region() {
        // s1: x = 2.1, s2: x = -2.1, s3: sphere at origin, r=4.2
        let s1 = Surface {
            surface_id: Some(5),
            kind: SurfaceKind::Plane {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 2.1,
            }, // x = 2.1
            boundary: BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(6),
            kind: SurfaceKind::Plane {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: -2.1,
            }, // x = -2.1
            boundary: BoundaryType::default(),
            name: None,
        };
        let s3 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 4.2,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Above(Arc::new(
                s2,
            ))))
            .intersection(&Region::new_from_halfspace(HalfspaceType::Below(Arc::new(
                s3,
            ))));
        let cell = Cell::new(Some(42), region, Some("complex".to_string()), None);
        // Point inside all constraints
        assert!(cell.contains((0.0, 0.0, 0.0)));
        // Point outside s1 (x > 2.1)
        assert!(!cell.contains((3.0, 0.0, 0.0)));
        // Point outside s2 (x < -2.1)
        assert!(!cell.contains((-3.0, 0.0, 0.0)));
        // Point outside sphere (r > 4.2)
        assert!(!cell.contains((0.0, 0.0, 5.0)));
    }
    use std::sync::Arc;
    use yamc::geo::{BoundaryType, Surface, SurfaceKind};
    use yamc::geo::{HalfspaceType, Region};
    use yamc::geometry::cell::*;

    #[test]
    fn test_cell_contains_simple() {
        // Sphere of radius 2 at (0,0,0)
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        let cell = Cell::new(Some(1), region, None, None);
        assert!(cell.contains((0.0, 0.0, 0.0)));
        assert!(!cell.contains((3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_cell_union_intersection_complement() {
        // Two spheres
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 2.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1.clone())));
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2.clone())));
        // Union
        let union_cell = Cell::new(Some(2), region1.clone().union(&region2.clone()), None, None);
        assert!(union_cell.contains((0.0, 0.0, 0.0)));
        assert!(union_cell.contains((2.0, 0.0, 0.0)));
        assert!(!union_cell.contains((5.0, 0.0, 0.0)));
        // Intersection
        let intersection_cell = Cell::new(
            Some(3),
            region1.clone().intersection(&region2.clone()),
            None,
            None,
        );
        assert!(!intersection_cell.contains((0.0, 0.0, 0.0)));
        assert!(intersection_cell.contains((1.0, 0.0, 0.0)));
        // Complement
        let complement_cell = Cell::new(Some(4), region1.complement(), None, None);
        assert!(!complement_cell.contains((0.0, 0.0, 0.0)));
        assert!(complement_cell.contains((5.0, 0.0, 0.0)));
    }

    #[test]
    fn test_cell_naming() {
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        let cell = Cell::new(Some(1), region, Some("fuel".to_string()), None);
        assert_eq!(cell.name, Some("fuel".to_string()));
    }

    #[test]
    fn test_cell_id_default() {
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let cell = Cell::new(None, region, None, None);
        assert_eq!(cell.get_cell_id(), None, "Default cell_id should be None");
    }

    #[test]
    fn test_set_and_get_cell_id() {
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));

        let mut cell = Cell::new(None, region, None, None);

        // Test default value
        assert_eq!(cell.get_cell_id(), None);

        // Test setting and getting cell_id
        cell.set_cell_id(42);
        assert_eq!(cell.get_cell_id(), Some(42));

        // Test setting a different value
        cell.set_cell_id(999);
        assert_eq!(cell.get_cell_id(), Some(999));
    }

    #[test]
    fn test_cell_new_with_specific_id() {
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));

        // Test creating cell with specific ID
        let cell1 = Cell::new(Some(100), region.clone(), None, None);
        assert_eq!(cell1.get_cell_id(), Some(100));

        // Test creating cell with ID 0 (should be allowed since 0 can be a valid ID)
        let cell2 = Cell::new(Some(0), region.clone(), None, None);
        assert_eq!(cell2.get_cell_id(), Some(0));
    }

    #[test]
    fn test_cell_id_independence() {
        let s1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));

        // Test that different cells have independent IDs
        let mut cell1 = Cell::new(None, region.clone(), None, None);
        let mut cell2 = Cell::new(Some(50), region.clone(), None, None);

        cell1.set_cell_id(10);
        cell2.set_cell_id(20);

        assert_eq!(cell1.get_cell_id(), Some(10));
        assert_eq!(cell2.get_cell_id(), Some(20));

        // Ensure they don't affect each other
        cell1.set_cell_id(999);
        assert_eq!(cell1.get_cell_id(), Some(999));
        assert_eq!(
            cell2.get_cell_id(),
            Some(20),
            "Other cell's ID should not change"
        );
    }
}
