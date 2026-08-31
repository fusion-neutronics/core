mod tests {
    use std::sync::Arc;
    use yamc::geo::{BoundaryType, Surface, SurfaceKind};
    use yamc::geo::{HalfspaceType, Region};
    use yamc::geometry::cell::Cell;
    use yamc::geometry::*;

    #[test]
    fn test_find_cell() {
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
        let cell = Cell::new(Some(1), region, Some("cell1".to_string()), None);
        let geometry =
            Geometry::new(vec![cell.clone()], Vec::new()).expect("Failed to create geometry");
        // Point inside the sphere
        assert!(geometry.find_cell((0.0, 0.0, 0.0)).is_some());
        // Point outside the sphere
        assert!(geometry.find_cell((5.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn test_cell_id_validation() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

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

        // Test duplicate cell IDs
        let cell1 = Cell::new(Some(1), region.clone(), Some("cell1".to_string()), None);
        let cell2 = Cell::new(Some(1), region.clone(), Some("cell2".to_string()), None);
        let result = Geometry::new(vec![cell1, cell2], Vec::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Duplicate cell_id 1"));
    }

    #[test]
    fn test_cells_auto_assign_ids() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

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

        // Cells without IDs should get auto-assigned IDs
        let cell1 = Cell::new(None, region.clone(), Some("cell1".to_string()), None);
        let cell2 = Cell::new(None, region.clone(), Some("cell2".to_string()), None);
        let geometry =
            Geometry::new(vec![cell1, cell2], Vec::new()).expect("Auto-assign should succeed");
        assert_eq!(geometry.cells[0].cell_id, Some(1));
        assert_eq!(geometry.cells[1].cell_id, Some(2));

        // Mix of explicit and auto-assigned IDs
        let cell_a = Cell::new(Some(5), region.clone(), Some("a".to_string()), None);
        let cell_b = Cell::new(None, region.clone(), Some("b".to_string()), None);
        let geometry =
            Geometry::new(vec![cell_a, cell_b], Vec::new()).expect("Mixed IDs should succeed");
        assert_eq!(geometry.cells[0].cell_id, Some(5));
        assert_eq!(geometry.cells[1].cell_id, Some(6));
    }

    #[test]
    fn test_cell_ids_must_be_unique() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

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

        // All cells have IDs assigned
        let cell1 = Cell::new(Some(1), region.clone(), Some("cell1".to_string()), None);
        let cell2 = Cell::new(Some(2), region.clone(), Some("cell2".to_string()), None);
        let cell3 = Cell::new(Some(3), region.clone(), Some("cell3".to_string()), None);

        let geometry = Geometry::new(vec![cell1, cell2, cell3], Vec::new())
            .expect("Failed to create geometry");

        let ids: Vec<u32> = geometry.cells.iter().map(|c| c.cell_id.unwrap()).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_material_id_validation() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};
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

        // Test duplicate material IDs
        let mut m1 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m1.material_id = Some(10);
        let mat1 = Arc::new(m1);
        let mut m2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m2.material_id = Some(10); // Same ID - should fail
        let mat2 = Arc::new(m2);

        let cell1 = Cell::new(Some(1), region.clone(), Some("cell1".to_string()), Some(0));
        let cell2 = Cell::new(Some(2), region.clone(), Some("cell2".to_string()), Some(1));

        let result = Geometry::new(vec![cell1, cell2], vec![mat1, mat2]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Duplicate material_id 10"));
    }

    #[test]
    fn test_materials_auto_assign_ids() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};
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

        // Materials without IDs should get auto-assigned IDs
        let mat1 = Arc::new(Material::new(HashMap::new(), "atom", "sum", None).unwrap());
        let mat2 = Arc::new(Material::new(HashMap::new(), "atom", "sum", None).unwrap());

        let cell1 = Cell::new(None, region.clone(), Some("cell1".to_string()), Some(0));
        let cell2 = Cell::new(None, region.clone(), Some("cell2".to_string()), Some(1));

        let geometry = Geometry::new(vec![cell1, cell2], vec![mat1, mat2])
            .expect("Auto-assign should succeed");
        let id1 = geometry.materials[0].get_material_id().unwrap();
        let id2 = geometry.materials[1].get_material_id().unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        // Mix of explicit and auto-assigned material IDs
        let mut m3 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m3.material_id = Some(10);
        let mat3 = Arc::new(m3);
        let mat4 = Arc::new(Material::new(HashMap::new(), "atom", "sum", None).unwrap());

        let cell3 = Cell::new(None, region.clone(), Some("cell3".to_string()), Some(0));
        let cell4 = Cell::new(None, region.clone(), Some("cell4".to_string()), Some(1));

        let geometry =
            Geometry::new(vec![cell3, cell4], vec![mat3, mat4]).expect("Mixed IDs should succeed");
        let id3 = geometry.materials[0].get_material_id().unwrap();
        let id4 = geometry.materials[1].get_material_id().unwrap();
        assert_eq!(id3, 10);
        assert_eq!(id4, 11);
    }

    #[test]
    fn test_material_ids_must_be_unique() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};
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

        // All materials have IDs assigned
        let mut m1 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m1.material_id = Some(1);
        let mat1 = Arc::new(m1);
        let mut m2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m2.material_id = Some(2);
        let mat2 = Arc::new(m2);
        let mut m3 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m3.material_id = Some(3);
        let mat3 = Arc::new(m3);

        let cell1 = Cell::new(Some(1), region.clone(), Some("cell1".to_string()), Some(0));
        let cell2 = Cell::new(Some(2), region.clone(), Some("cell2".to_string()), Some(1));
        let cell3 = Cell::new(Some(3), region.clone(), Some("cell3".to_string()), Some(2));

        let geometry = Geometry::new(vec![cell1, cell2, cell3], vec![mat1, mat2, mat3])
            .expect("Failed to create geometry");

        // Check material IDs are preserved
        let mat1_id = geometry.materials[0].get_material_id().unwrap();
        let mat2_id = geometry.materials[1].get_material_id().unwrap();
        let mat3_id = geometry.materials[2].get_material_id().unwrap();
        assert_eq!(mat1_id, 1);
        assert_eq!(mat2_id, 2);
        assert_eq!(mat3_id, 3);
    }

    #[test]
    fn test_cells_without_materials() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

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

        // Cells without materials (void cells) should work fine
        let cell1 = Cell::new(
            Some(1),
            region.clone(),
            Some("void_cell1".to_string()),
            None,
        );
        let cell2 = Cell::new(
            Some(2),
            region.clone(),
            Some("void_cell2".to_string()),
            None,
        );

        let geometry =
            Geometry::new(vec![cell1, cell2], Vec::new()).expect("Failed to create geometry");
        assert_eq!(geometry.cells.len(), 2);
        assert_eq!(geometry.cells[0].get_cell_id(), Some(1));
        assert_eq!(geometry.cells[1].get_cell_id(), Some(2));
    }

    #[test]
    fn test_surface_id_validation() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

        // Test duplicate surface IDs
        let s1 = Surface {
            surface_id: Some(10),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };
        let s2 = Surface {
            surface_id: Some(10), // Same ID - should fail
            kind: SurfaceKind::Sphere {
                x0: 2.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };

        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2)));

        let cell1 = Cell::new(Some(1), region1, Some("cell1".to_string()), None);
        let cell2 = Cell::new(Some(2), region2, Some("cell2".to_string()), None);

        let result = Geometry::new(vec![cell1, cell2], Vec::new());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Duplicate surface_id 10 found"));
    }

    #[test]
    fn test_surface_without_id_validation() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

        // Test surface without ID - should be allowed (None IDs are valid)
        let s1 = Surface {
            surface_id: None, // No ID - should be fine
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
        let cell = Cell::new(Some(1), region, Some("cell1".to_string()), None);

        let result = Geometry::new(vec![cell], Vec::new());
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_surface_ids() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

        // Test valid unique surface IDs
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
        let s2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 2.0,
                y0: 0.0,
                z0: 0.0,
                radius: 1.0,
            },
            boundary: BoundaryType::default(),
            name: None,
        };

        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s1)));
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(s2)));

        let cell1 = Cell::new(Some(1), region1, Some("cell1".to_string()), None);
        let cell2 = Cell::new(Some(2), region2, Some("cell2".to_string()), None);

        let geometry =
            Geometry::new(vec![cell1, cell2], Vec::new()).expect("Failed to create geometry");
        assert_eq!(geometry.cells.len(), 2);
    }

    #[test]
    fn test_id_map_xy_basis() {
        use std::collections::HashMap;
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};
        use yamc_materials::Material;

        // Create a sphere at origin with radius 2
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
        let mut mat_inner = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        mat_inner.material_id = Some(10);
        let mat = Arc::new(mat_inner);
        let cell = Cell::new(Some(5), region, Some("sphere".to_string()), Some(0));

        let geometry = Geometry::new(vec![cell], vec![mat]).expect("Failed to create geometry");

        // Test xy basis at z=0 (through center of sphere)
        // origin=(0,0,0), width=(6,6), pixels=(5,5)
        let (cell_ids, material_ids) =
            geometry.sample_slice((0.0, 0.0, 0.0), (6.0, 6.0), (5, 5), "xy");

        // Check dimensions
        assert_eq!(cell_ids.len(), 5);
        assert_eq!(cell_ids[0].len(), 5);
        assert_eq!(material_ids.len(), 5);
        assert_eq!(material_ids[0].len(), 5);

        // Center point should be inside the sphere
        assert_eq!(cell_ids[2][2], 5);
        assert_eq!(material_ids[2][2], 10);

        // Corner points should be outside
        assert_eq!(cell_ids[0][0], -1);
        assert_eq!(material_ids[0][0], -1);
        assert_eq!(cell_ids[4][4], -1);
        assert_eq!(material_ids[4][4], -1);
    }

    #[test]
    fn test_id_map_xz_basis() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

        // Create a sphere at origin with radius 2
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
        let cell = Cell::new(Some(7), region, Some("sphere".to_string()), None);

        let geometry = Geometry::new(vec![cell], Vec::new()).expect("Failed to create geometry");

        // Test xz basis at y=0
        // origin=(0,0,0), width=(6,6), pixels=(5,5)
        let (cell_ids, material_ids) =
            geometry.sample_slice((0.0, 0.0, 0.0), (6.0, 6.0), (5, 5), "xz");

        // Center point should be inside
        assert_eq!(cell_ids[2][2], 7);
        // No material, so material_id should be -1
        assert_eq!(material_ids[2][2], -1);

        // Corners should be outside
        assert_eq!(cell_ids[0][0], -1);
    }

    #[test]
    fn test_id_map_yz_basis() {
        use yamc::geo::{BoundaryType, Surface, SurfaceKind};
        use yamc::geo::{HalfspaceType, Region};

        // Create a sphere at origin with radius 2
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
        let cell = Cell::new(Some(3), region, Some("sphere".to_string()), None);

        let geometry = Geometry::new(vec![cell], Vec::new()).expect("Failed to create geometry");

        // Test yz basis at x=0
        // origin=(0,0,0), width=(6,6), pixels=(5,5)
        let (cell_ids, _material_ids) =
            geometry.sample_slice((0.0, 0.0, 0.0), (6.0, 6.0), (5, 5), "yz");

        // Center point should be inside
        assert_eq!(cell_ids[2][2], 3);

        // Corners should be outside
        assert_eq!(cell_ids[0][0], -1);
        assert_eq!(cell_ids[4][0], -1);
    }
}
