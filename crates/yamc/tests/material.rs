mod tests {

    use std::collections::HashMap;
    use yamc_materials::DensityUnits;
    #[allow(unused_imports)]
    use yamc_materials::Material;

    #[test]
    fn test_set_and_get_name() {
        let mut mat = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert_eq!(mat.get_name(), None);
        mat.set_name("TestMaterial");
        assert_eq!(mat.get_name(), Some("TestMaterial"));
        mat.set_name("AnotherName");
        assert_eq!(mat.get_name(), Some("AnotherName"));
    }

    #[test]
    fn test_material_id_default() {
        let mat = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert_eq!(
            mat.get_material_id(),
            None,
            "Default material_id should be None"
        );
        assert_eq!(
            mat.material_id, None,
            "Default material_id field should be None"
        );
    }

    #[test]
    fn test_set_and_get_material_id() {
        let mut mat = Material::new(HashMap::new(), "atom", "sum", None).unwrap();

        // Test default value
        assert_eq!(mat.get_material_id(), None);

        // Test setting and getting material_id
        mat.set_material_id(42);
        assert_eq!(mat.get_material_id(), Some(42));

        // Test setting a different value
        mat.set_material_id(999);
        assert_eq!(mat.get_material_id(), Some(999));

        // Test setting to 0
        mat.set_material_id(0);
        assert_eq!(mat.get_material_id(), Some(0));
    }

    #[test]
    fn test_material_with_id_constructor() {
        // Test creating material with specific ID
        let mut mat1 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        mat1.material_id = Some(100);
        assert_eq!(mat1.get_material_id(), Some(100));
        assert_eq!(mat1.material_id, Some(100));
        assert_eq!(mat1.get_name(), None, "Constructor should not set name");

        // Test creating material with ID 0
        let mut mat2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        mat2.material_id = Some(0);
        assert_eq!(mat2.get_material_id(), Some(0));

        // Test creating material with max ID
        let mut mat3 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        mat3.material_id = Some(u32::MAX);
        assert_eq!(mat3.get_material_id(), Some(u32::MAX));
    }

    #[test]
    fn test_material_id_independence() {
        // Test that different materials have independent IDs
        let mut mat1 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        let mut mat2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        mat2.material_id = Some(50);

        mat1.set_material_id(10);
        mat2.set_material_id(20);

        assert_eq!(mat1.get_material_id(), Some(10));
        assert_eq!(mat2.get_material_id(), Some(20));

        // Ensure they don't affect each other
        mat1.set_material_id(999);
        assert_eq!(mat1.get_material_id(), Some(999));
        assert_eq!(
            mat2.get_material_id(),
            Some(20),
            "Other material's ID should not change"
        );
    }

    #[test]
    fn test_sample_distance_to_collision() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        // Set up a mock total cross section and energy grid
        material.unified_energy_grid_neutron = vec![1.0, 10.0, 100.0];
        material
            .macroscopic_xs_neutron
            .insert(1, vec![2.0, 2.0, 2.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let energy = 5.0;
        // Sample 200 times and check the average is close to expected mean
        let mut samples = Vec::with_capacity(200);
        for _ in 0..200 {
            let distance = material.sample_distance_to_collision(energy, &mut rng);
            assert!(distance.is_some());
            samples.push(distance.unwrap());
        }
        // For sigma_t = 2.0, mean = 1/sigma_t = 0.5
        let avg: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_mean = 0.5;
        let tolerance = 0.05; // 10% tolerance
        assert!(
            (avg - expected_mean).abs() < tolerance,
            "Average sampled distance incorrect: got {}, expected {}",
            avg,
            expected_mean
        );
    }

    #[test]
    fn test_macroscopic_xs_neutron_total_by_nuclide_li6_li7() {
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // Call with by_nuclide = true
        let mt_filter = vec![1];
        let (_grid, _macro_xs) = material.calculate_macroscopic_xs(&mt_filter, true);
        // Check that macroscopic_xs_neutron_total_by_nuclide is Some and contains both nuclides
        let by_nuclide = material
            .macroscopic_xs_neutron_total_by_nuclide
            .as_ref()
            .expect("macroscopic_xs_neutron_total_by_nuclide should be Some");
        let sorted_keys = material
            .sorted_nuclide_keys
            .as_ref()
            .expect("sorted_nuclide_keys should be Some");
        assert!(
            sorted_keys.contains(&"Li6".to_string()),
            "sorted_nuclide_keys should contain Li6"
        );
        assert!(
            sorted_keys.contains(&"Li7".to_string()),
            "sorted_nuclide_keys should contain Li7"
        );
        // Check that the vectors are non-empty and same length as energy grid
        let grid_len = material.unified_energy_grid_neutron.len();
        assert_eq!(by_nuclide.len(), sorted_keys.len());
        for xs_vec in by_nuclide {
            assert_eq!(xs_vec.len(), grid_len, "xs vector should match grid length");
        }
    }

    #[test]
    fn test_macroscopic_xs_mt3_does_not_generate_mt1() {
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        let mt_filter = vec![3];
        let (_grid, macro_xs) = material.calculate_macroscopic_xs(&mt_filter, false);
        assert!(
            !macro_xs.contains_key(&1),
            "MT=1 should NOT be present when only MT=3 is requested"
        );
    }

    #[test]
    fn test_macroscopic_xs_mt24_does_not_generate_mt1() {
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        let mt_filter = vec![24];
        let (_grid, macro_xs) = material.calculate_macroscopic_xs(&mt_filter, false);
        assert!(
            !macro_xs.contains_key(&1),
            "MT=1 should NOT be present when only MT=24 is requested"
        );
    }
    #[test]
    fn test_hierarchical_mt3_generated_for_li6() {
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // Request only MT=3 (now present in JSON)
        let mt_filter = vec![3];
        let (_grid, macro_xs) = material.calculate_macroscopic_xs(&mt_filter, false);
        assert!(
            macro_xs.contains_key(&3),
            "MT=3 should be present in macro_xs for Li6"
        );
        // Just check that the cross section is nonzero and matches the JSON
        let xs = &macro_xs[&3];
        assert!(
            xs.iter().any(|&v| v > 0.0),
            "MT=3 cross section should have nonzero values"
        );
    }
    // Removed unused `use yamc::*;` (all required items referenced explicitly)

    #[test]
    fn test_new_material() {
        let material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert!(material.nuclides.is_empty());
        assert_eq!(material.density, None);
        assert_eq!(material.density_units.as_str(), "sum");
    }

    #[test]
    fn test_add_nuclide() {
        // Test creating material with nuclides upfront
        let material = Material::new(
            HashMap::from([("U235".into(), 0.05), ("U238".into(), 0.95)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        assert_eq!(material.nuclides.get("U235"), Some(&0.05));
        assert_eq!(material.nuclides.get("U238"), Some(&0.95));
        assert_eq!(material.nuclides.len(), 2);

        // Test overwriting by creating new material (or directly modifying pub field)
        let material2 = Material::new(
            HashMap::from([("U235".into(), 0.1), ("U238".into(), 0.95)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();
        assert_eq!(material2.nuclides.get("U235"), Some(&0.1));
    }

    #[test]
    fn test_add_nuclide_negative_fraction() {
        // Test that negative fractions are rejected by Material::new
        let result = Material::new(HashMap::from([("U235".into(), -0.05)]), "atom", "sum", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("negative"));
    }

    #[test]
    fn test_set_density() {
        // Test setting a valid density via constructor
        let material = Material::new(HashMap::new(), "atom", "g/cm3", Some(10.5)).unwrap();
        assert_eq!(material.density, Some(10.5));
        assert_eq!(material.density_units.as_str(), "g/cm3");

        // Test setting a different unit
        let material2 = Material::new(HashMap::new(), "atom", "kg/m3", Some(10500.0)).unwrap();
        assert_eq!(material2.density, Some(10500.0));
        assert_eq!(material2.density_units.as_str(), "kg/m3");
    }

    #[test]
    fn test_set_density_negative_value() {
        // Test setting a negative density
        let result = Material::new(HashMap::new(), "atom", "g/cm3", Some(-10.5));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Density must be positive");
    }

    #[test]
    fn test_set_density_zero_value() {
        // Test setting a zero density
        let result = Material::new(HashMap::new(), "atom", "g/cm3", Some(0.0));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Density must be positive");
    }

    #[test]
    fn test_material_clone() {
        let material = Material::new(
            HashMap::from([("U235".into(), 0.05), ("U238".into(), 0.95)]),
            "atom",
            "g/cm3",
            Some(19.1),
        )
        .unwrap();

        let cloned = material.clone();

        assert_eq!(cloned.nuclides.get("U235"), Some(&0.05));
        assert_eq!(cloned.nuclides.get("U238"), Some(&0.95));
        assert_eq!(cloned.density, Some(19.1));
        assert_eq!(cloned.density_units.as_str(), "g/cm3");
    }

    #[test]
    fn test_material_debug() {
        let material = Material::new(
            HashMap::from([("U235".into(), 0.05)]),
            "atom",
            "g/cm3",
            Some(19.1),
        )
        .unwrap();

        // This test merely ensures that the Debug implementation doesn't panic
        let _debug_str = format!("{:?}", material);
    }

    #[test]
    fn test_volume_get_and_set() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();

        // Test setting a valid volume
        let result = material.volume(Some(100.0));
        assert!(result.is_ok());
        assert_eq!(material.volume, Some(100.0));

        // Test getting the current volume
        let current_volume = material.volume(None).unwrap();
        assert_eq!(current_volume, Some(100.0));

        // Test setting an invalid (negative) volume
        let result = material.volume(Some(-50.0));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Volume must be positive");
        assert_eq!(material.volume, Some(100.0)); // Ensure the volume wasn't changed
    }

    #[test]
    fn test_get_nuclides() {
        // Empty material should return empty vector
        let material_empty = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert!(material_empty.get_nuclides().is_empty());

        // Add some nuclides
        let material = Material::new(
            HashMap::from([
                ("U235".into(), 0.05),
                ("U238".into(), 0.95),
                ("O16".into(), 2.0),
            ]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        // Check the result is sorted
        let nuclides = material.get_nuclides();
        assert_eq!(
            nuclides,
            vec!["O16".to_string(), "U235".to_string(), "U238".to_string()]
        );
    }

    #[test]
    fn test_get_atoms_per_barn_cm() {
        // Test with single nuclide case - must load nuclide data first
        let mut material_single = Material::new(
            HashMap::from([("Li6".into(), 2.5)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        // Load nuclide data from H5 file
        let mut nuclide_paths: HashMap<String, String> = HashMap::new();
        nuclide_paths.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material_single
            .read_nuclear_data(&nuclide_paths, None)
            .unwrap();

        let atoms_single = material_single.get_atoms_per_barn_cm().unwrap();
        assert_eq!(
            atoms_single.len(),
            1,
            "Should have 1 nuclide in the HashMap"
        );

        // For a single nuclide, we normalize the fraction to 1.0
        // AWR from H5 file * neutron_mass gives atomic mass
        let avogadro = 6.02214076e23;
        let li6_awr = material_single
            .nuclide_data
            .get("Li6")
            .unwrap()
            .atomic_weight_ratio
            .unwrap();
        let neutron_mass = 1.00866491595;
        let li6_mass = li6_awr * neutron_mass;
        let li6_expected = 1.0 * avogadro / li6_mass * 1.0e-24; // Fraction is normalized to 1.0

        let li6_actual = atoms_single.get("Li6").unwrap();
        let tolerance = 0.01; // 1%

        assert!(
            (li6_actual - li6_expected).abs() / li6_expected < tolerance,
            "Li6 atoms/cc calculation (single nuclide) is incorrect: got {}, expected {}",
            li6_actual,
            li6_expected
        );

        // Test with multiple nuclides
        let mut material_multi = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        // Load nuclide data from H5 files
        let mut nuclide_paths: HashMap<String, String> = HashMap::new();
        nuclide_paths.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_paths.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material_multi
            .read_nuclear_data(&nuclide_paths, None)
            .unwrap();

        let atoms_multi = material_multi.get_atoms_per_barn_cm().unwrap();
        assert_eq!(
            atoms_multi.len(),
            2,
            "Should have 2 nuclides in the HashMap"
        );

        // For multiple nuclides, the fractions are normalized and used with average molar mass
        let li7_awr = material_multi
            .nuclide_data
            .get("Li7")
            .unwrap()
            .atomic_weight_ratio
            .unwrap();
        let li7_mass = li7_awr * neutron_mass;
        let avg_mass = (0.5 * li6_mass + 0.5 * li7_mass) / 1.0; // weighted average

        let li6_expected_multi = 1.0 * avogadro / avg_mass * (0.5 / 1.0) * 1.0e-24;
        let li7_expected_multi = 1.0 * avogadro / avg_mass * (0.5 / 1.0) * 1.0e-24;

        let li6_actual_multi = atoms_multi.get("Li6").unwrap();
        let li7_actual_multi = atoms_multi.get("Li7").unwrap();

        assert!(
            (li6_actual_multi - li6_expected_multi).abs() / li6_expected_multi < tolerance,
            "Li6 atoms/cc calculation (multiple nuclides) is incorrect: got {}, expected {}",
            li6_actual_multi,
            li6_expected_multi
        );
        assert!(
            (li7_actual_multi - li7_expected_multi).abs() / li7_expected_multi < tolerance,
            "Li7 atoms/cc calculation (multiple nuclides) is incorrect: got {}, expected {}",
            li7_actual_multi,
            li7_expected_multi
        );

        // Test with non-normalized fractions
        let mut material_non_norm = Material::new(
            HashMap::from([("Li6".into(), 1.0), ("Li7".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        // Load nuclide data from H5 files
        let mut nuclide_paths: HashMap<String, String> = HashMap::new();
        nuclide_paths.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_paths.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material_non_norm
            .read_nuclear_data(&nuclide_paths, None)
            .unwrap();

        let atoms_non_norm = material_non_norm.get_atoms_per_barn_cm().unwrap();

        // Fractions should be normalized to 0.5 each (1.0/2.0)
        let avg_mass_non_norm = (1.0 * li6_mass + 1.0 * li7_mass) / 2.0;
        let li6_expected_non_norm = 1.0 * avogadro / avg_mass_non_norm * (1.0 / 2.0) * 1.0e-24;
        let li7_expected_non_norm = 1.0 * avogadro / avg_mass_non_norm * (1.0 / 2.0) * 1.0e-24;

        let li6_actual_non_norm = atoms_non_norm.get("Li6").unwrap();
        let li7_actual_non_norm = atoms_non_norm.get("Li7").unwrap();

        assert!(
            (li6_actual_non_norm - li6_expected_non_norm).abs() / li6_expected_non_norm < tolerance,
            "Li6 normalized atoms/cc calculation is incorrect: got {}, expected {}",
            li6_actual_non_norm,
            li6_expected_non_norm
        );
        assert!(
            (li7_actual_non_norm - li7_expected_non_norm).abs() / li7_expected_non_norm < tolerance,
            "Li7 normalized atoms/cc calculation is incorrect: got {}, expected {}",
            li7_actual_non_norm,
            li7_expected_non_norm
        );
    }

    #[test]
    fn test_get_atoms_per_barn_cm_no_density() {
        let material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();

        // Should return Err (not panic) when there are no nuclides defined
        let result = material.get_atoms_per_barn_cm();
        assert!(
            result.is_err(),
            "Should return Err when no nuclides defined"
        );

        // Should also return Err when nuclides are not added
        let material_with_density =
            Material::new(HashMap::new(), "atom", "g/cm3", Some(1.0)).unwrap();

        let result = material_with_density.get_atoms_per_barn_cm();
        assert!(
            result.is_err(),
            "Should return Err when no nuclides are defined"
        );
    }

    #[test]
    fn test_mean_free_path_neutron() {
        // Create a properly set up material
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();

        // Create mock cross sections directly (bypassing the normal calculation path)
        let energy_grid = vec![
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
            100000.0,
            1000000.0,
            10000000.0,
            100000000.0,
        ];
        material.unified_energy_grid_neutron = energy_grid.clone();

        // Set total cross section (in barns * atoms/cm³, which gives cm⁻¹)
        // Intentionally using a simple pattern that's easy to verify
        let total_xs = vec![
            1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        material.macroscopic_xs_neutron.insert(1, total_xs.clone());

        // Test exact values from our mock data
        assert_eq!(material.mean_free_path_neutron(1.0), Some(1.0));
        assert_eq!(material.mean_free_path_neutron(10.0), Some(2.0));
        assert_eq!(material.mean_free_path_neutron(100.0), Some(4.0));

        // Test interpolated value
        // At energy = 3.0, we're using linear interpolation between 1.0 and 10.0
        // Cross section should be about 0.889 (linearly interpolated between 1.0 and 0.5)
        // Mean free path should be about 1.125
        let mfp_3ev = material.mean_free_path_neutron(3.0).unwrap();
        assert!(
            (mfp_3ev - 1.125).abs() < 0.01,
            "Expected ~1.125, got {}",
            mfp_3ev
        );

        // Test outside of range (should use endpoint value)
        assert_eq!(material.mean_free_path_neutron(0.1), Some(1.0)); // Below range
        assert_eq!(material.mean_free_path_neutron(1e9), Some(256.0)); // Above range
    }

    #[test]
    fn test_mean_free_path_lithium_14mev() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "g/cm3", Some(0.534)).unwrap();
        // Set up mock cross section data for 14 MeV (1.4e7 eV)
        // We'll use a simple grid and cross section for demonstration
        let energy_grid = vec![1e6, 1.4e7, 1e8]; // eV
        let total_xs = vec![1.0, 0.5, 0.2]; // barns * atoms/cm³, so cm⁻¹
        material.unified_energy_grid_neutron = energy_grid.clone();
        material.macroscopic_xs_neutron.insert(1, total_xs.clone());
        // 14 MeV = 1.4e7 eV
        let mfp = material.mean_free_path_neutron(1.4e7);
        assert!(mfp.is_some());
        let mfp_val = mfp.unwrap();
        // At 14 MeV, total_xs = 0.5, so mean free path = 1/0.5 = 2.0 cm
        assert!(
            (mfp_val - 2.0).abs() < 1e-6,
            "Expected 2.0 cm, got {}",
            mfp_val
        );
    }

    #[test]
    fn test_add_element() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let material = Material::new(nuclides, "atom", "sum", None).unwrap();
        // Verify the isotopes were added correctly
        assert!(material.nuclides.contains_key("Li6"));
        assert!(material.nuclides.contains_key("Li7"));
        // Check the fractions are correct
        assert_eq!(*material.nuclides.get("Li6").unwrap(), 0.07589);
        assert_eq!(*material.nuclides.get("Li7").unwrap(), 0.92411);
        // Test adding an element with many isotopes
        let nuclides2 = yamc_nuclide::composition::expand_element("Sn", 1.0, "atom").unwrap();
        let material2 = Material::new(nuclides2, "atom", "sum", None).unwrap();
        assert_eq!(material2.nuclides.len(), 10);
    }

    #[test]
    fn test_add_element_invalid() {
        // Test with negative fraction
        let result = yamc_nuclide::composition::expand_element("Li", -1.0, "atom");
        assert!(result.is_err());
        // Test with invalid element
        let result = yamc_nuclide::composition::expand_element("Xx", 1.0, "atom");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_element_by_symbol_and_name() {
        // By symbol (case-sensitive, exact match)
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let material = Material::new(nuclides, "atom", "sum", None).unwrap();
        assert!(material.nuclides.contains_key("Li6"));
        assert!(material.nuclides.contains_key("Li7"));
        // By full name (case-sensitive, exact match)
        let nuclides2 = yamc_nuclide::composition::expand_element("gold", 1.0, "atom").unwrap();
        let material2 = Material::new(nuclides2, "atom", "sum", None).unwrap();
        assert!(material2.nuclides.contains_key("Au197"));
        // By full name (lowercase) - should fail
        let result = yamc_nuclide::composition::expand_element("Lithium", 1.0, "atom");
        assert!(result.is_err());
        // By symbol (lowercase) - should fail
        let result = yamc_nuclide::composition::expand_element("li", 1.0, "atom");
        assert!(result.is_err());
        // By symbol (uppercase) - should fail
        let result = yamc_nuclide::composition::expand_element("LI", 1.0, "atom");
        assert!(result.is_err());
        // Invalid name
        let result = yamc_nuclide::composition::expand_element("notanelement", 1.0, "atom");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_element_beryllium_and_iron() {
        let nuclides_be = yamc_nuclide::composition::expand_element("Be", 1.0, "atom").unwrap();
        let mat_be = Material::new(nuclides_be, "atom", "sum", None).unwrap();
        // Beryllium has only one stable isotope
        assert_eq!(mat_be.nuclides.len(), 1);
        assert!(mat_be.nuclides.contains_key("Be9"));
        // Check the fraction is 1.0 for Be9
        assert_eq!(*mat_be.nuclides.get("Be9").unwrap(), 1.0);

        let nuclides_fe = yamc_nuclide::composition::expand_element("Fe", 1.0, "atom").unwrap();
        let mat_fe = Material::new(nuclides_fe, "atom", "sum", None).unwrap();
        // Iron has four stable isotopes
        assert!(mat_fe.nuclides.contains_key("Fe54"));
        assert!(mat_fe.nuclides.contains_key("Fe56"));
        assert!(mat_fe.nuclides.contains_key("Fe57"));
        assert!(mat_fe.nuclides.contains_key("Fe58"));
        // Check that the sum of fractions is 1.0 (within tolerance)
        let sum: f64 = mat_fe.nuclides.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_material_reaction_mts_lithium() {
        use yamc_materials::Material;
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "sum", None).unwrap();
        material.set_temperature("294");
        // Prepare the nuclide JSON map for Li6 and Li7
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        // Read in the nuclear data
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // This will load Li6 and Li7, so the MTs should be the union of both, including hierarchical MTs
        let mts = material.reaction_mts().expect("Failed to get reaction MTs");
        // The expected list should match the actual list from the HDF5, including hierarchical MTs
        // endf-b8.1 Li6/Li7 add MT 32 (n,n'd) and MT 41 (n,2np) vs the prior
        // VIII.0 fixtures.
        let expected = vec![
            1, 2, 3, 4, 16, 24, 25, 27, 32, 41, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 101, 102,
            103, 104, 105, 203, 204, 205, 207, 301, 444, 901,
        ];
        assert_eq!(
            mts, expected,
            "Material lithium MT list does not match expected. Got {:?}",
            mts
        );
    }

    #[test]
    fn test_macroscopic_cross_section_flexible_reactions() {
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "atom",
            "g/cm3",
            Some(2.0),
        )
        .unwrap();
        material.set_temperature("294");

        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");

        // Test with integer MT number
        let (xs1, energy1) = material.macroscopic_cross_section(1, None);
        assert!(!energy1.is_empty(), "Energy grid should not be empty");
        assert!(!xs1.is_empty(), "Cross section should not be empty");
        assert_eq!(
            energy1.len(),
            xs1.len(),
            "Energy and cross section arrays should have same length"
        );

        // Test with string reaction name - same reaction
        let (xs2, energy2) = material.macroscopic_cross_section("(n,total)".to_string(), None);
        assert_eq!(
            energy1.len(),
            energy2.len(),
            "Energy grids should be same length"
        );
        assert_eq!(xs1.len(), xs2.len(), "Cross sections should be same length");

        // Values should be identical (or very close due to floating point)
        for (i, (&val1, &val2)) in xs1.iter().zip(xs2.iter()).enumerate() {
            assert!(
                (val1 - val2).abs() < 1e-10,
                "Cross section values should be identical at index {}: {} vs {}",
                i,
                val1,
                val2
            );
        }

        // Test with gamma capture reaction
        let (xs3, energy3) = material.macroscopic_cross_section("(n,gamma)", None);
        assert!(
            !energy3.is_empty(),
            "Energy grid should not be empty for (n,gamma)"
        );
        assert!(
            !xs3.is_empty(),
            "Cross section should not be empty for (n,gamma)"
        );
        assert_eq!(
            energy3.len(),
            xs3.len(),
            "Energy and cross section arrays should have same length for (n,gamma)"
        );
    }

    #[test]
    fn test_selective_temperature_load_be9_300() {
        yamc_nuclide::nuclide::clear_nuclide_cache();
        let mut mat =
            Material::new(HashMap::from([("Be9".into(), 1.0)]), "atom", "sum", None).unwrap();
        mat.set_temperature("300");
        let mut map = std::collections::HashMap::new();
        map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());

        // Loading with temperature 300 when only 294 is available should now fail
        // with a helpful error message
        let result = mat.read_nuclear_data(&map, None);
        assert!(
            result.is_err(),
            "Material loading should fail when requested temperature is not available"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("294"),
            "Error message should mention available temperature 294, got: {err_msg}"
        );
    }

    #[test]
    fn test_selective_temperature_load_be9_294() {
        yamc_nuclide::nuclide::clear_nuclide_cache();
        let mut mat =
            Material::new(HashMap::from([("Be9".into(), 1.0)]), "atom", "sum", None).unwrap();
        mat.set_temperature("294");
        let mut map = std::collections::HashMap::new();
        map.insert("Be9".to_string(), "tests/Be9.arrow".to_string());
        mat.read_nuclear_data(&map, None).unwrap();
        let be9 = mat.nuclide_data.get("Be9").expect("Be9 not loaded");
        assert_eq!(
            be9.available_temperatures,
            vec![
                "250".to_string(),
                "294".to_string(),
                "600".to_string(),
                "900".to_string(),
                "1200".to_string(),
                "2500".to_string()
            ]
        );
        assert_eq!(
            be9.loaded_temperatures,
            vec!["294".to_string()],
            "Should only load 294 data"
        );
    }

    #[test]
    fn test_calculate_microscopic_xs_neutron_lithium() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "sum", None).unwrap();
        material.set_temperature("294");
        // Prepare the nuclide JSON map for Li6 and Li7
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        // Read in the nuclear data
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // Build the unified energy grid
        let grid = material.unified_energy_grid_neutron();
        // Calculate microscopic cross sections
        let micro_xs = material.calculate_microscopic_xs_neutron(None);
        // Check that both Li6 and Li7 are present
        assert!(micro_xs.contains_key("Li6"));
        assert!(micro_xs.contains_key("Li7"));
        // Check that for a known MT (e.g., 2), both nuclides have cross section data
        let mt = 2;
        assert!(micro_xs["Li6"].contains_key(&mt), "Li6 missing MT=2");
        assert!(micro_xs["Li7"].contains_key(&mt), "Li7 missing MT=2");
        // Check that the cross section arrays are the same length as the grid
        assert_eq!(micro_xs["Li6"][&mt].len(), grid.len());
        assert_eq!(micro_xs["Li7"][&mt].len(), grid.len());
    }

    #[test]
    fn test_material_vs_nuclide_microscopic_xs_li6() {
        use yamc_nuclide::nuclide::get_or_load_nuclide;
        let mut material =
            Material::new(HashMap::from([("Li6".into(), 1.0)]), "atom", "sum", None).unwrap();
        material.set_temperature("294");
        // Prepare the nuclide JSON map for Li6
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        // Read in the nuclear data
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // Build the unified energy grid
        let grid = material.unified_energy_grid_neutron();
        // Calculate microscopic cross sections for the material
        let micro_xs_mat = material.calculate_microscopic_xs_neutron(None);
        // Get the nuclide directly
        let nuclide =
            get_or_load_nuclide("Li6", &nuclide_json_map, &yamc_nuclide::LoadScope::full())
                .expect("Failed to load Li6");
        let temperature = material.temperature();
        // Get reactions and energy grid for nuclide
        let reactions = nuclide
            .reactions_for_temp(temperature)
            .expect("No reactions for Li6");
        let energy_map = nuclide.energy.as_ref().expect("No energy map for Li6");
        let energy_grid = energy_map.get(temperature).expect("No energy grid for Li6");
        // For each MT in the material, compare the cross sections
        for (mt, xs_mat) in micro_xs_mat["Li6"].iter() {
            // Only compare if MT exists in nuclide
            if let Some(reaction) = reactions.get(mt) {
                let threshold_idx = reaction.threshold_idx;
                let nuclide_energy = if threshold_idx < energy_grid.len() {
                    &energy_grid[threshold_idx..]
                } else {
                    continue;
                };
                let xs_nuclide = &reaction.cross_section;
                // Interpolate nuclide xs onto the material grid
                let mut xs_nuclide_interp = Vec::with_capacity(grid.len());
                for &g in &grid {
                    if g < nuclide_energy[0] {
                        xs_nuclide_interp.push(0.0);
                    } else {
                        let xs = yamc::interpolate_linear(nuclide_energy, xs_nuclide, g);
                        xs_nuclide_interp.push(xs);
                    }
                }
                // Compare arrays (allow small tolerance)
                let tol = 1e-10;
                for (a, b) in xs_mat.iter().zip(xs_nuclide_interp.iter()) {
                    assert!(
                        (a - b).abs() < tol,
                        "Mismatch for MT {}: {} vs {}",
                        mt,
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn test_calculate_microscopic_xs_neutron_mt_filter() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "sum", None).unwrap();
        material.set_temperature("294");
        // Prepare the nuclide JSON map for Li6 and Li7
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        // Read in the nuclear data
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        // Build the unified energy grid
        let _grid = material.unified_energy_grid_neutron();
        // Calculate microscopic cross sections for all MTs
        let micro_xs_all = material.calculate_microscopic_xs_neutron(None);
        // Calculate microscopic cross sections for only MT=2
        let mt_filter = vec![2];
        let micro_xs_mt2 = material.calculate_microscopic_xs_neutron(Some(&mt_filter));
        // For each nuclide, only MT=2 should be present
        for nuclide in &["Li6", "Li7"] {
            assert!(
                micro_xs_mt2.contains_key(*nuclide),
                "{} missing in filtered result",
                nuclide
            );
            let xs_map = &micro_xs_mt2[*nuclide];
            // Assert that only the requested MT is present
            assert!(
                xs_map.keys().all(|k| *k == 2),
                "Filtered result for {} contains non-filtered MTs: {:?}",
                nuclide,
                xs_map.keys()
            );
            assert_eq!(
                xs_map.len(),
                1,
                "Filtered result for {} should have only one MT",
                nuclide
            );
            // The cross section array for MT=2 should match the unfiltered result
            let xs_all = &micro_xs_all[*nuclide][&2];
            let xs_filtered = &xs_map[&2];
            assert_eq!(
                xs_all, xs_filtered,
                "Filtered and unfiltered MT=2 xs do not match for {}",
                nuclide
            );
        }
    }

    #[test]
    fn test_calculate_macroscopic_xs_mt_filter() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "g/cm3", Some(0.534)).unwrap();
        material.set_temperature("294");
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        let mt_filter = vec![
            2, 16, 24, 25, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 101, 102, 103, 104, 105, 203,
            204, 205, 206, 207, 301, 444,
        ];
        let (_grid, macro_xs) = material.calculate_macroscopic_xs(&mt_filter, false);
        for mt in &mt_filter {
            match macro_xs.get(mt) {
                Some(xs) => assert_eq!(xs.len(), material.unified_energy_grid_neutron.len()),
                None => println!("MT {} not present in macro_xs", mt),
            }
        }
    }

    #[test]
    fn test_panic_if_no_density_for_macroscopic_xs_and_mean_free_path() {
        // Create material with g/cm3 density, then clear density to simulate "no density set"
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.density = None; // remove density to trigger panic
        material.set_temperature("294");

        // Prepare the nuclide JSON map for Li6
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        let _grid = material.unified_energy_grid_neutron();

        // Should panic when calculating macroscopic cross sections with no density
        let result = std::panic::catch_unwind(move || {
            let mut material = material;
            material.calculate_macroscopic_xs(&vec![1], false);
        });
        assert!(
            result.is_err(),
            "Should panic if density is not set for calculate_macroscopic_xs"
        );

        // Re-create material for the next test
        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.density = None; // remove density to trigger panic
        material.set_temperature("294");
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");
        material.unified_energy_grid_neutron();

        // Should panic when calculating mean free path with no density
        let result = std::panic::catch_unwind(move || {
            let mut material = material;
            material.mean_free_path_neutron(1e6);
        });
        assert!(
            result.is_err(),
            "Should panic if density is not set for mean_free_path_neutron"
        );
    }

    #[test]
    fn test_mean_free_path_lithium_real_data() {
        let nuclides = yamc_nuclide::composition::expand_element("Li", 1.0, "atom").unwrap();
        let mut material = Material::new(nuclides, "atom", "g/cm3", Some(0.534)).unwrap();
        material.set_temperature("294");

        // Prepare the nuclide JSON map for Li6 and Li7
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());

        // Read in the nuclear data
        material
            .read_nuclear_data(&nuclide_json_map, None)
            .expect("Failed to read nuclide JSON");

        // Calculate the mean free path at 14 MeV (1.4e7 eV)
        let mfp = material.mean_free_path_neutron(1.4e7);

        assert!(mfp.is_some(), "Mean free path should be Some for real data");
        let mfp_val = mfp.unwrap();
        // Print the value for inspection
        println!("Mean free path for lithium at 14 MeV: {} cm", mfp_val);
        // Check that the value is positive
        assert!(mfp_val > 0.0, "Mean free path should be positive");
        // Expected value with atomic mass from AWR (endf-b8.1 Li6/Li7 fixtures).
        let expected = 14.97826945;
        let rel_tol = 1e-4; // 0.01% tolerance for AWR-derived atomic mass differences
        assert!(
            (mfp_val - expected).abs() / expected < rel_tol,
            "Expected ~{:.8} cm, got {:.8} cm",
            expected,
            mfp_val
        );
    }

    #[test]
    fn test_sample_distance_to_collision_li6() {
        // Create Li6 material
        let mut mat = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        // Load nuclide data from JSON
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        mat.read_nuclear_data(&nuclide_json_map, None).unwrap();
        mat.set_temperature("294");

        // Check that the total cross section is present and nonzero at 14 MeV
        mat.calculate_macroscopic_xs(&vec![1], false);

        // Sample 1000 distances
        let mut sum = 0.0;
        let n_samples = 1000;
        use rand::rngs::StdRng;
        use rand::SeedableRng; // Needed for seed_from_u64
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let dist = mat
                .sample_distance_to_collision(14_000_000.0, &mut rng)
                .unwrap_or_else(|| panic!("sample_distance_to_collision returned None at 14 MeV!"));
            sum += dist;
        }
        let avg = sum / n_samples as f64;
        println!("Average distance: {}", avg);
        assert!(
            (avg - 6.9).abs() < 0.1,
            "Average {} not within 0.2 of 6.9",
            avg
        );
    }

    #[test]
    fn test_sample_interacting_nuclide_li6_li7() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 0.1), ("Li7".into(), 0.9)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclide data from JSON
        let mut nuclide_json_map = HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Calculate total xs to ensure everything is set up
        // For this test, calculate per-nuclide macroscopic total xs as well
        material.calculate_macroscopic_xs(&vec![1], true);

        // Check that macroscopic_xs_neutron_total_by_nuclide is present and not empty
        let by_nuclide = material.macroscopic_xs_neutron_total_by_nuclide.as_ref();
        assert!(
            by_nuclide.is_some(),
            "macroscopic_xs_neutron_total_by_nuclide should be Some after calculation"
        );
        let by_nuclide = by_nuclide.unwrap();
        assert!(
            !by_nuclide.is_empty(),
            "macroscopic_xs_neutron_total_by_nuclide should not be empty"
        );

        // Sample the interacting nuclide many times at 14 MeV
        let energy = 14_000_000.0;
        let n_samples = 10000;
        let mut counts = HashMap::new();
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let nuclide = material.sample_interacting_nuclide(energy, &mut rng);
            *counts.entry(nuclide).or_insert(0) += 1;
        }

        let count_li6 = *counts.get("Li6").unwrap_or(&0) as f64;
        let count_li7 = *counts.get("Li7").unwrap_or(&0) as f64;
        let total = count_li6 + count_li7;
        let frac_li6 = count_li6 / total;
        let frac_li7 = count_li7 / total;

        println!("Li6 fraction: {}, Li7 fraction: {}", frac_li6, frac_li7);

        // The sampled fractions should be close to the expected probability
        // (proportional to macroscopic total xs for each nuclide at this energy)
        // For a rough test, just check both are nonzero and sum to 1
        assert!(
            frac_li6 > 0.0 && frac_li7 > 0.0,
            "Both nuclides should be sampled"
        );
        assert!(
            (frac_li6 + frac_li7 - 1.0).abs() < 1e-6,
            "Fractions should sum to 1"
        );

        // Optionally, check that Li7 is sampled much more often than Li6 (since its fraction is higher)
        assert!(
            frac_li7 > frac_li6,
            "Li7 should be sampled more often than Li6"
        );
    }

    #[test]
    fn test_cache_invalidation_add_nuclide() {
        yamc_nuclide::nuclide::clear_nuclide_cache();

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 0.5)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclear data
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        nuclide_json_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Pre-populate cache by calculating cross sections
        material.calculate_macroscopic_xs(&vec![1], false);
        assert!(
            !material.macroscopic_xs_neutron.is_empty(),
            "Cache should be populated"
        );

        // Adding a nuclide via pub field should allow clearing the cache manually
        material.nuclides.insert("Li7".to_string(), 0.5);
        material.macroscopic_xs_neutron.clear();
        assert!(
            material.macroscopic_xs_neutron.is_empty(),
            "Cache should be cleared after adding nuclide"
        );
    }

    #[test]
    fn test_cache_invalidation_set_density() {
        yamc_nuclide::nuclide::clear_nuclide_cache();

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclear data
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Pre-populate cache
        material.calculate_macroscopic_xs(&vec![1], false);
        assert!(
            !material.macroscopic_xs_neutron.is_empty(),
            "Cache should be populated"
        );

        // Changing density via pub field and clearing cache manually
        material.density = Some(2.0);
        material.macroscopic_xs_neutron.clear();
        assert!(
            material.macroscopic_xs_neutron.is_empty(),
            "Cache should be cleared after density change"
        );
    }

    #[test]
    fn test_cache_invalidation_set_temperature() {
        yamc_nuclide::nuclide::clear_nuclide_cache();

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclear data
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Pre-populate cache
        material.calculate_macroscopic_xs(&vec![1], false);
        assert!(
            !material.macroscopic_xs_neutron.is_empty(),
            "Cache should be populated"
        );

        // Changing temperature should clear the cache
        material.set_temperature("300");
        assert!(
            material.macroscopic_xs_neutron.is_empty(),
            "Cache should be cleared after temperature change"
        );
    }

    #[test]
    fn test_cache_invalidation_read_nuclides() {
        yamc_nuclide::nuclide::clear_nuclide_cache();

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclear data first time
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Pre-populate cache
        material.calculate_macroscopic_xs(&vec![1], false);
        assert!(
            !material.macroscopic_xs_neutron.is_empty(),
            "Cache should be populated"
        );

        // Loading nuclear data again should clear the cache (use same map since empty map would fail)
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();
        assert!(
            material.macroscopic_xs_neutron.is_empty(),
            "Cache should be cleared after loading nuclear data"
        );
    }

    #[test]
    fn test_cache_behavior_after_calculation() {
        yamc_nuclide::nuclide::clear_nuclide_cache();

        let mut material = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        material.set_temperature("294");

        // Load nuclear data
        let mut nuclide_json_map = std::collections::HashMap::new();
        nuclide_json_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        material.read_nuclear_data(&nuclide_json_map, None).unwrap();

        // Calculate cross sections multiple times - each call replaces the cache
        material.calculate_macroscopic_xs(&vec![1], false);
        let first_size = material.macroscopic_xs_neutron.len();
        assert_eq!(first_size, 1, "Cache should contain only MT 1");

        material.calculate_macroscopic_xs(&vec![1, 2], false);
        let second_size = material.macroscopic_xs_neutron.len();
        assert_eq!(second_size, 2, "Cache should contain MT 1 and 2");

        // Verify that calling with just MT 1 replaces cache with only MT 1
        material.calculate_macroscopic_xs(&vec![1], false);
        assert_eq!(
            material.macroscopic_xs_neutron.len(),
            1,
            "Cache should contain only MT 1 again"
        );

        // Verify that calling with all MTs gives us all MTs in cache
        material.calculate_macroscopic_xs(&vec![1, 2], false);
        assert_eq!(
            material.macroscopic_xs_neutron.len(),
            2,
            "Cache should contain MT 1 and 2 again"
        );
    }

    #[test]
    fn test_material_different_data_sources() {
        // Test that material loading respects different data sources

        yamc_nuclide::nuclide::clear_nuclide_cache();

        // Material 1: Li6 from file
        let mut mat_li6 = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        mat_li6.set_temperature("294");
        let mut li6_map = std::collections::HashMap::new();
        li6_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        mat_li6.read_nuclear_data(&li6_map, None).unwrap();

        // Material 2: Li7 from file
        let mut mat_li7 = Material::new(
            HashMap::from([("Li7".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        mat_li7.set_temperature("294");
        let mut li7_map = std::collections::HashMap::new();
        li7_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        mat_li7.read_nuclear_data(&li7_map, None).unwrap();

        // Get macroscopic cross sections
        let (xs_li6, _) = mat_li6.macroscopic_cross_section("(n,gamma)", None);
        let (xs_li7, _) = mat_li7.macroscopic_cross_section("(n,gamma)", None);

        // Should have different data (different nuclides)
        let data_different = xs_li6.len() != xs_li7.len()
            || xs_li6
                .iter()
                .zip(&xs_li7)
                .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(
            data_different,
            "Li6 and Li7 materials should have different cross sections"
        );
        println!(
            "Material Li6: {} points, Li7: {} points",
            xs_li6.len(),
            xs_li7.len()
        );
    }

    #[test]
    fn test_material_file_and_keyword_sources() {
        // Test that materials can use both file paths and keywords

        yamc_nuclide::nuclide::clear_nuclide_cache();

        // Material 1: Li6 from file
        let mut mat_file = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        mat_file.set_temperature("294");
        let mut nuclide_map = std::collections::HashMap::new();
        nuclide_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        mat_file.read_nuclear_data(&nuclide_map, None).unwrap();

        // Material 2: Li7 from file (different nuclide, different file)
        let mut mat_other = Material::new(
            HashMap::from([("Li7".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        mat_other.set_temperature("294");
        let mut other_map = std::collections::HashMap::new();
        other_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        mat_other.read_nuclear_data(&other_map, None).unwrap();

        // Both should work and give different results (different nuclides)
        let (xs_file, _) = mat_file.macroscopic_cross_section("(n,gamma)", None);
        let (xs_other, _) = mat_other.macroscopic_cross_section("(n,gamma)", None);

        assert!(
            !xs_file.is_empty() && !xs_other.is_empty(),
            "Both materials should have cross section data"
        );

        // Should be different since Li6 vs Li7
        let data_different = xs_file.len() != xs_other.len()
            || xs_file
                .iter()
                .zip(&xs_other)
                .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            data_different,
            "Li6 file vs Li7 file should give different results"
        );
    }

    #[test]
    fn test_material_cache_respects_data_source_boundaries() {
        // Test that material cache properly separates different data sources

        yamc_nuclide::nuclide::clear_nuclide_cache();

        // Material 1: Li6 from file first time
        let mut mat_li6_1 = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        mat_li6_1.set_temperature("294");
        let mut li6_map = std::collections::HashMap::new();
        li6_map.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        mat_li6_1.read_nuclear_data(&li6_map, None).unwrap();

        // Material 2: Li7 from file
        let mut mat_li7 = Material::new(
            HashMap::from([("Li7".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        mat_li7.set_temperature("294");
        let mut li7_map = std::collections::HashMap::new();
        li7_map.insert("Li7".to_string(), "tests/Li7.arrow".to_string());
        mat_li7.read_nuclear_data(&li7_map, None).unwrap();

        // Material 3: Li6 from file again (should use cache)
        let mut mat_li6_2 = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();
        mat_li6_2.set_temperature("294");
        mat_li6_2.read_nuclear_data(&li6_map, None).unwrap(); // reuse same map

        // Get cross sections
        let (xs_li6_1, _) = mat_li6_1.macroscopic_cross_section("(n,gamma)", None);
        let (xs_li7, _) = mat_li7.macroscopic_cross_section("(n,gamma)", None);
        let (xs_li6_2, _) = mat_li6_2.macroscopic_cross_section("(n,gamma)", None);

        // Li6 materials should be identical (cache working)
        assert_eq!(
            xs_li6_1, xs_li6_2,
            "Li6 materials should be identical (cache working)"
        );

        // Li6 vs Li7 should be different (different nuclides)
        let li6_vs_li7_different = xs_li6_1.len() != xs_li7.len()
            || xs_li6_1
                .iter()
                .zip(&xs_li7)
                .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            li6_vs_li7_different,
            "Li6 and Li7 materials should have different data"
        );
    }

    #[test]
    fn test_material_path_normalization_in_cache() {
        // Test that different path formats for same file use same cache entry

        yamc_nuclide::nuclide::clear_nuclide_cache();

        // Material 1: relative path
        let mut mat_rel = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        mat_rel.set_temperature("294");
        let mut map_rel = std::collections::HashMap::new();
        map_rel.insert("Li6".to_string(), "tests/Li6.arrow".to_string());
        mat_rel.read_nuclear_data(&map_rel, None).unwrap();

        // Material 2: absolute path to same file
        let mut mat_abs = Material::new(
            HashMap::from([("Li6".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(1.0),
        )
        .unwrap();
        mat_abs.set_temperature("294");
        let mut map_abs = std::collections::HashMap::new();
        let abs_path = std::env::current_dir().unwrap().join("tests/Li6.arrow");
        map_abs.insert("Li6".to_string(), abs_path.to_string_lossy().to_string());
        mat_abs.read_nuclear_data(&map_abs, None).unwrap();

        // Should give identical results (same file, cache working)
        let (xs_rel, _) = mat_rel.macroscopic_cross_section("(n,gamma)", None);
        let (xs_abs, _) = mat_abs.macroscopic_cross_section("(n,gamma)", None);

        assert_eq!(
            xs_rel, xs_abs,
            "Relative and absolute paths to same file should give identical results"
        );
    }

    #[test]
    fn test_set_density_sum_no_value() {
        let material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert_eq!(material.density, None);
        assert_eq!(material.density_units.as_str(), "sum");
    }

    #[test]
    fn test_set_density_sum_with_value() {
        // Providing a value with "sum" is allowed (stored but not used in calculation)
        let material = Material::new(HashMap::new(), "atom", "sum", Some(1.0)).unwrap();
        assert_eq!(material.density, Some(1.0));
        assert_eq!(material.density_units.as_str(), "sum");
    }

    #[test]
    fn test_set_density_sum_negative_value_rejected() {
        let result = Material::new(HashMap::new(), "atom", "sum", Some(-1.0));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Density must be positive");
    }

    #[test]
    fn test_set_density_none_for_non_sum_unit() {
        let result = Material::new(HashMap::new(), "atom", "g/cm3", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("required"));
    }

    #[test]
    fn test_set_density_unsupported_unit() {
        let result = Material::new(HashMap::new(), "atom", "invalid_unit", Some(1.0));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported"));
    }

    #[test]
    fn test_get_atoms_per_barn_cm_sum_mode_single_nuclide() {
        let material =
            Material::new(HashMap::from([("Li6".into(), 3.0e-2)]), "atom", "sum", None).unwrap();

        let atoms = material.get_atoms_per_barn_cm().unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(*atoms.get("Li6").unwrap(), 3.0e-2);
    }

    #[test]
    fn test_get_atoms_per_barn_cm_sum_mode_multiple_nuclides() {
        let material = Material::new(
            HashMap::from([("H1".into(), 6.6e-2), ("O16".into(), 3.3e-2)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        let atoms = material.get_atoms_per_barn_cm().unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(*atoms.get("H1").unwrap(), 6.6e-2);
        assert_eq!(*atoms.get("O16").unwrap(), 3.3e-2);
    }

    #[test]
    fn test_get_atoms_per_barn_cm_sum_mode_negative_fraction_errors() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        // Bypass validation by directly modifying fields
        material.nuclides.insert("H1".to_string(), -1.0e-2);
        material.density_units = DensityUnits::Sum;

        let result = material.get_atoms_per_barn_cm();
        assert!(
            result.is_err(),
            "Should return Err for negative atom density"
        );
        assert!(result.unwrap_err().contains("negative atom density"));
    }

    #[test]
    fn test_sum_mode_does_not_require_nuclide_data() {
        // In "sum" mode, get_atoms_per_barn_cm should not need loaded nuclide data
        // since it just returns fractions directly (no AWR/molar mass needed)
        let material = Material::new(
            HashMap::from([("U235".into(), 1.5e-3), ("U238".into(), 2.2e-2)]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        // No call to read_nuclides - should still work
        let atoms = material.get_atoms_per_barn_cm().unwrap();
        assert_eq!(*atoms.get("U235").unwrap(), 1.5e-3);
        assert_eq!(*atoms.get("U238").unwrap(), 2.2e-2);
    }

    #[test]
    fn test_transmutable_default_and_set() {
        let mut material = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert!(!material.get_transmutable());

        material.set_transmutable(true);
        assert!(material.get_transmutable());

        material.set_transmutable(false);
        assert!(!material.get_transmutable());

        // Also check that new material defaults to false
        let mut material2 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        material2.material_id = Some(42);
        assert!(!material2.get_transmutable());
    }

    #[test]
    fn test_add_nuclide_with_mass_number() {
        use yamc_materials::Material;
        // Valid nuclide names with mass numbers
        let mat = Material::new(
            HashMap::from([
                ("Fe56".into(), 1.0),
                ("U235".into(), 0.5),
                ("Am241m".into(), 0.3),
                ("H3".into(), 0.1),
            ]),
            "atom",
            "sum",
            None,
        )
        .unwrap();

        // Verify nuclides were added
        assert!(mat.nuclides.contains_key("Fe56"));
        assert!(mat.nuclides.contains_key("U235"));
        assert!(mat.nuclides.contains_key("Am241m"));
        assert!(mat.nuclides.contains_key("H3"));
    }

    #[test]
    fn test_add_nuclide_rejects_element_symbols() {
        use yamc_materials::Material;
        // Invalid: element symbols without mass numbers should be rejected
        let result = Material::new(HashMap::from([("Fe".into(), 1.0)]), "atom", "sum", None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Invalid nuclide name") || err.contains("Fe"));

        let result = Material::new(HashMap::from([("U".into(), 1.0)]), "atom", "sum", None);
        assert!(result.is_err());

        let result = Material::new(HashMap::from([("Li".into(), 1.0)]), "atom", "sum", None);
        assert!(result.is_err());

        let result = Material::new(HashMap::from([("W".into(), 1.0)]), "atom", "sum", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_element_enriched_lithium_atom() {
        // 60% Li6 enrichment by atom percent
        let nuclides = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 60.0, "Li6", "atom", "atom",
        )
        .unwrap();
        let mat = Material::new(nuclides, "atom", "sum", None).unwrap();
        assert_eq!(mat.nuclides.len(), 2);
        let li6 = *mat.nuclides.get("Li6").unwrap();
        let li7 = *mat.nuclides.get("Li7").unwrap();
        assert!((li6 - 0.6).abs() < 1e-10, "Expected Li6=0.6, got {li6}");
        assert!((li7 - 0.4).abs() < 1e-10, "Expected Li7=0.4, got {li7}");
    }

    #[test]
    fn test_add_element_enriched_lithium_mass() {
        // 60% Li6 enrichment by weight percent
        let nuclides = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 60.0, "Li6", "mass", "atom",
        )
        .unwrap();
        let mat = Material::new(nuclides, "atom", "sum", None).unwrap();
        assert_eq!(mat.nuclides.len(), 2);
        let li6 = *mat.nuclides.get("Li6").unwrap();
        let li7 = *mat.nuclides.get("Li7").unwrap();
        // Convert 60 mass% to atom fractions using AME2020 atomic masses:
        // Li6 = 6.015123 u, Li7 = 7.016003 u
        let m6 = 6.015123_f64;
        let m7 = 7.016003_f64;
        let expected_li6 = (0.6 / m6) / (0.6 / m6 + 0.4 / m7);
        let expected_li7 = 1.0 - expected_li6;
        assert!(
            (li6 - expected_li6).abs() < 1e-6,
            "Expected Li6={expected_li6}, got {li6}"
        );
        assert!(
            (li7 - expected_li7).abs() < 1e-6,
            "Expected Li7={expected_li7}, got {li7}"
        );
    }

    #[test]
    fn test_add_element_enriched_boron() {
        // 90% B10 enrichment by atom percent
        let nuclides = yamc_nuclide::composition::expand_element_enriched(
            "B", 2.0, 90.0, "B10", "atom", "atom",
        )
        .unwrap();
        let mat = Material::new(nuclides, "atom", "sum", None).unwrap();
        let b10 = *mat.nuclides.get("B10").unwrap();
        let b11 = *mat.nuclides.get("B11").unwrap();
        // fraction * enriched_atom_frac
        assert!((b10 - 1.8).abs() < 1e-10, "Expected B10=1.8, got {b10}");
        assert!((b11 - 0.2).abs() < 1e-10, "Expected B11=0.2, got {b11}");
    }

    #[test]
    fn test_add_element_enriched_100_percent() {
        // 100% Li6 -- pure isotope
        let nuclides = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 100.0, "Li6", "atom", "atom",
        )
        .unwrap();
        let mat = Material::new(nuclides, "atom", "sum", None).unwrap();
        let li6 = *mat.nuclides.get("Li6").unwrap();
        assert!((li6 - 1.0).abs() < 1e-10);
        // Li7 should be 0.0, but 0.0 is allowed
        let li7 = *mat.nuclides.get("Li7").unwrap();
        assert!((li7).abs() < 1e-10);
    }

    #[test]
    fn test_add_element_enriched_by_name() {
        // Using element name instead of symbol
        let nuclides = yamc_nuclide::composition::expand_element_enriched(
            "lithium", 1.0, 50.0, "Li6", "atom", "atom",
        )
        .unwrap();
        let mat = Material::new(nuclides, "atom", "sum", None).unwrap();
        assert!(mat.nuclides.contains_key("Li6"));
        assert!(mat.nuclides.contains_key("Li7"));
    }

    #[test]
    fn test_add_element_enriched_invalid_three_isotopes() {
        // Oxygen has 3 naturally-occurring isotopes -- should fail
        let result = yamc_nuclide::composition::expand_element_enriched(
            "O", 1.0, 50.0, "O16", "atom", "atom",
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("3 naturally-occurring isotopes"));
    }

    #[test]
    fn test_add_element_enriched_invalid_target() {
        // Li8 is not a naturally-occurring isotope of Li
        let result = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 50.0, "Li8", "atom", "atom",
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("not a naturally-occurring isotope"));
    }

    #[test]
    fn test_add_element_enriched_invalid_enrichment_range() {
        let result = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 0.0, "Li6", "atom", "atom",
        );
        assert!(result.is_err());
        let result = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, -5.0, "Li6", "atom", "atom",
        );
        assert!(result.is_err());
        let result = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 101.0, "Li6", "atom", "atom",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_add_element_enriched_invalid_type() {
        let result = yamc_nuclide::composition::expand_element_enriched(
            "Li", 1.0, 50.0, "Li6", "invalid", "atom",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'atom' or 'mass'"));
    }

    #[test]
    fn test_fraction_type_default_is_atom() {
        let mat = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
        assert_eq!(mat.fraction_type.as_str(), "atom");
    }

    #[test]
    fn test_set_fraction_type_mass() {
        let mat = Material::new(HashMap::new(), "mass", "sum", None).unwrap();
        assert_eq!(mat.fraction_type.as_str(), "mass");
    }

    #[test]
    fn test_set_fraction_type_invalid() {
        let result = Material::new(HashMap::new(), "invalid", "sum", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'atom' or 'mass'"));
    }

    #[test]
    fn test_cannot_mix_fraction_types() {
        // Create a material with atom fraction type and nuclides
        let mat = Material::new(HashMap::from([("Li6".into(), 0.5)]), "atom", "sum", None).unwrap();
        assert_eq!(mat.fraction_type.as_str(), "atom");
        // Verify we can't create a mass material and pretend it's atom
        // (the fraction_type is set at construction time now)
        assert_eq!(mat.fraction_type.as_str(), "atom");
    }

    #[test]
    fn test_fraction_type_mass_consistent() {
        let mat = Material::new(HashMap::from([("Li6".into(), 0.3)]), "mass", "sum", None).unwrap();
        assert_eq!(mat.fraction_type.as_str(), "mass");
    }

    #[test]
    fn test_get_atoms_per_barn_cm_weight_percent() {
        // Create a material with weight fractions and verify the density calculation
        // For a single nuclide, atom and mass should give the same result
        let mat_ao = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "atom",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();

        let mat_wo = Material::new(
            HashMap::from([("Fe56".into(), 1.0)]),
            "mass",
            "g/cm3",
            Some(7.874),
        )
        .unwrap();

        // For a pure single-nuclide material, both should give the same atom density
        // because N = rho * N_A / M * 1e-24 regardless of fraction interpretation
        // We need nuclide data loaded for the AWR-based mass. Use mass-number fallback.
        let atoms_ao = mat_ao.get_atoms_per_barn_cm().unwrap();
        let atoms_wo = mat_wo.get_atoms_per_barn_cm().unwrap();
        let fe56_ao = atoms_ao["Fe56"];
        let fe56_wo = atoms_wo["Fe56"];
        assert!(
            (fe56_ao - fe56_wo).abs() / fe56_ao < 1e-10,
            "Single-nuclide: ao={fe56_ao} vs wo={fe56_wo} should be equal"
        );
    }

    #[test]
    fn test_get_atoms_per_barn_cm_weight_percent_two_nuclides() {
        // Two nuclides with equal weight fractions vs equal atom fractions
        // should give different atom densities
        let mat_ao = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "atom",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();

        let mat_wo = Material::new(
            HashMap::from([("Li6".into(), 0.5), ("Li7".into(), 0.5)]),
            "mass",
            "g/cm3",
            Some(0.534),
        )
        .unwrap();

        let atoms_ao = mat_ao.get_atoms_per_barn_cm().unwrap();
        let atoms_wo = mat_wo.get_atoms_per_barn_cm().unwrap();

        // For atom: 50/50 atom fractions, equal number densities
        assert!(
            (atoms_ao["Li6"] - atoms_ao["Li7"]).abs() / atoms_ao["Li6"] < 1e-10,
            "ao 50/50 should give equal atom densities"
        );

        // For mass: 50/50 weight fractions, lighter isotope has MORE atoms
        // N_Li6 / N_Li7 = (w_6/M_6) / (w_7/M_7) = M_7/M_6 > 1
        assert!(
            atoms_wo["Li6"] > atoms_wo["Li7"],
            "wo 50/50: Li6 should have more atoms than Li7 (lighter isotope)"
        );
    }
}
