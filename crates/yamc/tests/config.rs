mod tests {

    use yamc_nuclide::Config;

    #[test]
    fn test_set_cross_section_global_keyword() {
        for keyword in ["tendl-2025", "fendl-3.2d", "endf-b8.1"] {
            let mut config = Config::new();
            config.set_cross_section(keyword, None);
            assert_eq!(config.default_cross_section, Some(keyword.to_string()));
        }
    }

    #[test]
    #[should_panic(expected = "Invalid cross section source")]
    fn test_set_cross_section_invalid_keyword() {
        let mut config = Config::new();
        config.set_cross_section("invalid-keyword", None);
    }

    #[test]
    fn test_set_and_get_cross_section_for_nuclide() {
        let mut config = Config::new();
        config.set_cross_section("Li6", Some("tests/Li6.arrow"));
        assert_eq!(
            config.get_cross_section("Li6"),
            Some("tests/Li6.arrow".to_string())
        );
    }

    #[test]
    fn test_get_cross_section_fallback_to_global() {
        let mut config = Config::new();
        config.set_cross_section("tendl-2025", None);
        assert_eq!(
            config.get_cross_section("Fe56"),
            Some("tendl-2025".to_string())
        );
    }

    #[test]
    fn test_set_cross_section_path_to_file() {
        let mut config = Config::new();
        config.set_cross_section("Fe56", Some("tests/Fe56.arrow"));
        assert_eq!(
            config.get_cross_section("Fe56"),
            Some("tests/Fe56.arrow".to_string())
        );
    }

    #[test]
    fn test_set_cross_section_single_keyword() {
        let mut config = Config::new();
        config.set_cross_section("Fe56", Some("tendl-2025"));
        assert_eq!(
            config.get_cross_section("Fe56"),
            Some("tendl-2025".to_string())
        );
    }

    #[test]
    fn test_set_cross_sections_multiple() {
        let mut config = Config::new();
        let cross_sections = std::collections::HashMap::from([
            ("Li7".to_string(), "tests/Li7.arrow".to_string()),
            ("Li6".to_string(), "tendl-2025".to_string()),
        ]);
        config.set_cross_sections(cross_sections);
        assert_eq!(
            config.get_cross_section("Li7"),
            Some("tests/Li7.arrow".to_string())
        );
        assert_eq!(
            config.get_cross_section("Li6"),
            Some("tendl-2025".to_string())
        );
    }

    #[test]
    fn test_set_cross_section_global_keyword_for_all() {
        for keyword in ["tendl-2025", "fendl-3.2d", "endf-b8.1"] {
            let mut config = Config::new();
            config.set_cross_section(keyword, None);
            assert_eq!(config.get_cross_section("Li6"), Some(keyword.to_string()));
            assert_eq!(config.get_cross_section("Fe56"), Some(keyword.to_string()));
        }
    }

    #[test]
    fn test_set_cross_sections_with_string_keyword() {
        let mut config = Config::new();
        config.set_cross_sections("tendl-2025");
        assert_eq!(config.default_cross_section, Some("tendl-2025".to_string()));
        assert_eq!(
            config.get_cross_section("Li6"),
            Some("tendl-2025".to_string())
        );
        assert_eq!(
            config.get_cross_section("Fe56"),
            Some("tendl-2025".to_string())
        );
    }

    #[test]
    fn test_set_cross_sections_with_hashmap() {
        let mut config = Config::new();
        let cross_sections = std::collections::HashMap::from([
            ("Li6".to_string(), "tests/Li6.arrow".to_string()),
            ("Fe56".to_string(), "tendl-2025".to_string()),
        ]);
        config.set_cross_sections(cross_sections);
        assert_eq!(
            config.get_cross_section("Li6"),
            Some("tests/Li6.arrow".to_string())
        );
        assert_eq!(
            config.get_cross_section("Fe56"),
            Some("tendl-2025".to_string())
        );
        // When a keyword is in the hashmap, it should set the global default too
        assert_eq!(config.default_cross_section, Some("tendl-2025".to_string()));
    }

    #[test]
    #[should_panic(expected = "Invalid cross section source")]
    fn test_set_cross_sections_invalid_string_keyword() {
        let mut config = Config::new();
        config.set_cross_sections("invalid-keyword");
    }
}
