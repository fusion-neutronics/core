mod tests {
    use std::sync::Arc;
    use yamc::geo::{BoundaryType, Surface, SurfaceKind};
    use yamc::geo::{HalfspaceType, Region};
    use yamc::geometry::cell::Cell;
    use yamc_tallies::filter::cell::*;

    #[test]
    fn test_cell_filter_creation() {
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        let cell = Cell::new(Some(42), region, Some("test_cell".to_string()), None);

        let filter = CellFilter::from_id(cell.cell_id.unwrap());
        assert_eq!(filter.cell_ids, vec![42]);
    }

    #[test]
    fn test_cell_filter_matching() {
        let sphere1 = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region1 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere1)));

        let sphere2 = Surface {
            surface_id: Some(2),
            kind: SurfaceKind::Sphere {
                x0: 1.0,
                y0: 1.0,
                z0: 1.0,
                radius: 3.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region2 = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere2)));

        let cell = Cell::new(Some(42), region1, Some("test_cell".to_string()), None);
        let filter = CellFilter::from_id(cell.cell_id.unwrap());

        assert!(filter.matches(42));
        assert!(!filter.matches(43));
        assert!(filter.matches(cell.cell_id.unwrap()));

        let other_cell = Cell::new(Some(99), region2, Some("other_cell".to_string()), None);
        assert!(!filter.matches(other_cell.cell_id.unwrap()));
    }

    #[test]
    fn test_cell_filter_equality() {
        let sphere = Surface {
            surface_id: Some(1),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));

        let cell1 = Cell::new(
            Some(42),
            region.clone(),
            Some("test_cell".to_string()),
            None,
        );
        let cell2 = Cell::new(
            Some(42),
            region.clone(),
            Some("another_name".to_string()),
            None,
        );

        let filter1 = CellFilter::from_id(cell1.cell_id.unwrap());
        let filter2 = CellFilter::from_id(cell2.cell_id.unwrap());

        assert_eq!(filter1, filter2);
    }

    fn make_cell(id: u32) -> Cell {
        let sphere = Surface {
            surface_id: Some(id as usize),
            kind: SurfaceKind::Sphere {
                x0: 0.0,
                y0: 0.0,
                z0: 0.0,
                radius: 2.0,
            },
            boundary: BoundaryType::Vacuum,
            name: None,
        };
        let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
        Cell::new(Some(id), region, None, None)
    }

    #[test]
    fn test_cell_filter_from_cells_multi() {
        let c1 = make_cell(10);
        let c2 = make_cell(20);
        let c3 = make_cell(30);
        let filter = CellFilter::from_ids(vec![
            c1.cell_id.unwrap(),
            c2.cell_id.unwrap(),
            c3.cell_id.unwrap(),
        ]);
        assert_eq!(filter.cell_ids, vec![10, 20, 30]);
        assert_eq!(filter.num_bins(), 3);
    }

    #[test]
    fn test_cell_filter_from_cells_single() {
        let c1 = make_cell(10);
        let filter = CellFilter::from_ids(vec![c1.cell_id.unwrap()]);
        assert_eq!(filter.cell_ids, vec![10]);
        assert_eq!(filter.num_bins(), 1);
    }

    #[test]
    #[should_panic(expected = "CellFilter requires at least one cell ID")]
    fn test_cell_filter_from_ids_empty_panics() {
        let _ = CellFilter::from_ids(vec![]);
    }

    #[test]
    fn test_cell_filter_get_bin_returns_position() {
        let c1 = make_cell(10);
        let c2 = make_cell(20);
        let c3 = make_cell(30);
        let filter = CellFilter::from_ids(vec![
            c1.cell_id.unwrap(),
            c2.cell_id.unwrap(),
            c3.cell_id.unwrap(),
        ]);
        assert_eq!(filter.get_bin(10), Some(0));
        assert_eq!(filter.get_bin(20), Some(1));
        assert_eq!(filter.get_bin(30), Some(2));
        assert_eq!(filter.get_bin(99), None);
    }

    #[test]
    fn test_cell_filter_matches_respects_list() {
        let c1 = make_cell(10);
        let c2 = make_cell(20);
        let filter = CellFilter::from_ids(vec![c1.cell_id.unwrap(), c2.cell_id.unwrap()]);
        assert!(filter.matches(10));
        assert!(filter.matches(20));
        assert!(!filter.matches(30));
    }
}
