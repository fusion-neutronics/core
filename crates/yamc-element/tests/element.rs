mod tests {

    use yamc_element::Element;

    #[test]
    fn test_element_struct_isotopes() {
        let fe = Element::new("Fe");
        let list = fe.get_nuclides();
        assert!(list.contains(&"Fe54".to_string()));
        assert!(list.contains(&"Fe56".to_string()));
        assert!(list.contains(&"Fe57".to_string()));
        assert!(list.contains(&"Fe58".to_string()));
    }

    #[test]
    fn test_unknown_element() {
        let fake = Element::new("Xx");
        assert!(fake.get_nuclides().is_empty());
    }

    #[test]
    fn test_get_nuclides_fe_full_list() {
        let fe = Element::new("Fe");
        let list = fe.get_nuclides();
        let expected = ["Fe54", "Fe56", "Fe57", "Fe58"];
        assert_eq!(
            list.len(),
            expected.len(),
            "Unexpected number of Fe isotopes: {:?}",
            list
        );
        assert_eq!(
            list,
            expected.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "Fe isotope list mismatch"
        );
    }
}
