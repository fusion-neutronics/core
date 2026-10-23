mod tests {
    use std::collections::HashMap;
    use yamc_materials::Material;
    use yamc_tallies::filter::material::*;

    #[test]
    fn test_material_filter_creation() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material.set_material_id(123);
        material.set_name("test_material");
        let filter = MaterialFilter::new(&material).unwrap();
        assert_eq!(filter.material_ids, vec![123]);
    }

    #[test]
    fn test_material_filter_creation_no_id_is_an_error() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material.set_name("steel");
        let err = MaterialFilter::new(&material).unwrap_err();
        assert!(
            err.contains("material \"steel\"") && err.contains("no id"),
            "{err}"
        );

        let unnamed = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        let err = MaterialFilter::from_materials(&[&make_material(11), &unnamed]).unwrap_err();
        assert!(err.contains("the material at position 1"), "{err}");
    }

    #[test]
    fn test_material_filter_matching() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material.set_material_id(123);
        material.set_name("test_material");
        let filter = MaterialFilter::new(&material).unwrap();

        assert!(filter.matches(Some(123)));
        assert!(!filter.matches(Some(124)));
        assert!(!filter.matches(None));
        assert!(filter.matches_material(&material));

        let mut other_material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        other_material.set_material_id(456);
        other_material.set_name("other_material");
        assert!(!filter.matches_material(&other_material));
    }

    #[test]
    fn test_material_filter_equality() {
        let mut material1 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material1.set_material_id(123);
        material1.set_name("material_1");
        let mut material2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material2.set_material_id(123);
        material2.set_name("material_2");

        let filter1 = MaterialFilter::new(&material1).unwrap();
        let filter2 = MaterialFilter::new(&material2).unwrap();

        assert_eq!(filter1, filter2);

        let mut material3 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material3.set_material_id(456);
        material3.set_name("material_3");
        let filter3 = MaterialFilter::new(&material3).unwrap();

        assert_ne!(filter1, filter3);
    }

    fn make_material(id: u32) -> Material {
        let mut m = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        m.set_material_id(id);
        m
    }

    #[test]
    fn test_material_filter_from_materials_multi() {
        let m1 = make_material(11);
        let m2 = make_material(22);
        let m3 = make_material(33);
        let filter = MaterialFilter::from_materials(&[&m1, &m2, &m3]).unwrap();
        assert_eq!(filter.material_ids, vec![11, 22, 33]);
        assert_eq!(filter.num_bins(), 3);
    }

    #[test]
    fn test_material_filter_from_materials_single() {
        let m1 = make_material(11);
        let filter = MaterialFilter::from_materials(&[&m1]).unwrap();
        assert_eq!(filter.material_ids, vec![11]);
        assert_eq!(filter.num_bins(), 1);
    }

    #[test]
    fn test_material_filter_from_materials_empty_is_an_error() {
        let err = MaterialFilter::from_materials(&[]).unwrap_err();
        assert!(err.contains("at least one material"), "{err}");
    }

    #[test]
    fn test_material_filter_from_materials_rejects_a_shared_id() {
        // The bins are found by id, so a second material with the first one's
        // id would get a bin that never scores.
        let m1 = make_material(11);
        let m2 = make_material(22);
        let clash = make_material(11);
        let err = MaterialFilter::from_materials(&[&m1, &m2, &clash]).unwrap_err();
        assert!(err.contains("positions 0 and 2 both have id 11"), "{err}");
        let err = MaterialFilter::from_materials(&[&m2, &m2]).unwrap_err();
        assert!(err.contains("both have id 22"), "{err}");
    }

    #[test]
    fn test_material_filter_get_bin_returns_position() {
        let m1 = make_material(11);
        let m2 = make_material(22);
        let m3 = make_material(33);
        let filter = MaterialFilter::from_materials(&[&m1, &m2, &m3]).unwrap();
        assert_eq!(filter.get_bin(Some(11)), Some(0));
        assert_eq!(filter.get_bin(Some(22)), Some(1));
        assert_eq!(filter.get_bin(Some(33)), Some(2));
        assert_eq!(filter.get_bin(Some(99)), None);
        assert_eq!(filter.get_bin(None), None);
    }

    #[test]
    fn test_material_filter_matches_respects_list() {
        let m1 = make_material(11);
        let m2 = make_material(22);
        let filter = MaterialFilter::from_materials(&[&m1, &m2]).unwrap();
        assert!(filter.matches(Some(11)));
        assert!(filter.matches(Some(22)));
        assert!(!filter.matches(Some(33)));
        assert!(!filter.matches(None));
    }
}
