mod tests {
    use yamc_tallies::filter::*;
    use yamc_tallies::{CellFilter, MaterialFilter};

    #[test]
    fn test_type_name_cell() {
        let cell_filter = CellFilter { cell_ids: vec![42] };
        let filter = Filter::Cell(cell_filter);
        assert_eq!(filter.type_name(), "CellFilter");
    }

    #[test]
    fn test_type_name_material() {
        let material_filter = MaterialFilter {
            material_ids: vec![99],
        };
        let filter = Filter::Material(material_filter);
        assert_eq!(filter.type_name(), "MaterialFilter");
    }
}
