//! Integration tests for `Material::init_photon_data` /
//! `calculate_photon_xs` / `sample_element`. Live here (rather than in
//! the yamc-materials crate) because they need on-disk Arrow nuclide /
//! element fixtures that ship with the yamc test directory.

use std::collections::HashMap;
use std::sync::Arc;
use yamc_materials::Material;

/// Helper: create a pure-Fe material with density and nuclide data loaded.
fn make_fe_material() -> Material {
    let mut mat = Material::new(
        HashMap::from([("Fe56".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    mat.set_temperature("294");

    let mut nuclide_map = HashMap::new();
    nuclide_map.insert("Fe56".to_string(), "tests/Fe56.arrow".to_string());
    mat.read_nuclear_data(&nuclide_map, None).unwrap();

    let mt_filter = vec![1];
    mat.calculate_macroscopic_xs(&mt_filter, false);

    mat
}

#[test]
fn test_init_photon_data_fe() {
    let mut mat = make_fe_material();
    let mut photon_paths = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());

    let result = mat.init_photon_data(&photon_paths);
    assert!(
        result.is_ok(),
        "init_photon_data failed: {:?}",
        result.err()
    );
    assert_eq!(mat.element_indices.len(), 1);
    assert_eq!(mat.element_indices[0].0, "Fe56");
}

#[test]
fn test_calculate_photon_xs_fe() {
    let mut mat = make_fe_material();
    let mut photon_paths = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    mat.init_photon_data(&photon_paths).unwrap();

    let xs = mat.calculate_photon_xs(100_000.0);
    assert!(xs.total > 0.0, "Total photon XS should be positive");
    assert!(xs.coherent > 0.0, "Coherent XS should be positive");
    assert!(xs.incoherent > 0.0, "Incoherent XS should be positive");
    assert!(
        xs.photoelectric > 0.0,
        "Photoelectric XS should be positive"
    );

    let sum = xs.coherent + xs.incoherent + xs.photoelectric + xs.pair_production;
    assert!(
        (xs.total - sum).abs() / sum < 1e-10,
        "Total ({}) should equal sum of components ({})",
        xs.total,
        sum
    );

    let fe_element = yamc_element::photon::get_element_by_name("Fe").unwrap();
    let micro = fe_element.calculate_xs(100_000.0);
    let atom_density = mat.cached_atoms_per_barn_cm.as_ref().unwrap()["Fe56"];
    assert!(
        (xs.total - atom_density * micro.total).abs() / xs.total < 1e-10,
        "Macro XS should equal N * micro XS"
    );
}

#[test]
fn test_sample_element_fe() {
    let mut mat = make_fe_material();
    let mut photon_paths = HashMap::new();
    photon_paths.insert("Fe".to_string(), "tests/Fe.arrow".to_string());
    mat.init_photon_data(&photon_paths).unwrap();

    let xs = mat.calculate_photon_xs(100_000.0);

    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);

    for _ in 0..10 {
        let (idx, element) = mat.sample_element(xs.total, 100_000.0, &mut rng);
        assert_eq!(element.name, "Fe");
        assert_eq!(element.atomic_number, 26);
        assert_eq!(idx, element.index);
    }
}

#[test]
fn test_photon_cache_keeps_sources_separate() {
    let original = yamc_element::photon::get_or_load_element("Fe", "tests/Fe.arrow").unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let other_path = temporary.path().join("Fe.arrow");
    std::fs::create_dir(&other_path).unwrap();
    for entry in std::fs::read_dir("tests/Fe.arrow").unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), other_path.join(entry.file_name())).unwrap();
        }
    }
    let other_source = other_path.to_str().unwrap();
    let other = yamc_element::photon::get_or_load_element("Fe", other_source).unwrap();
    assert!(!Arc::ptr_eq(&original, &other));
    assert_ne!(original.index, other.index);
    let cached = yamc_element::photon::get_or_load_element("Fe", other_source).unwrap();
    assert!(Arc::ptr_eq(&other, &cached));
}

#[test]
fn test_photon_cache_does_not_hide_a_different_source() {
    let element = yamc_element::photon::get_or_load_element("Fe", "tests/Fe.arrow").unwrap();
    assert_eq!(element.atomic_number, 26);
    let missing =
        yamc_element::photon::get_or_load_element("Fe", "tests/missing-photon-source/Fe.arrow");
    assert!(
        missing.is_err(),
        "a cached element must not replace a different data source"
    );
}

#[test]
fn test_init_photon_data_missing_path() {
    let mut mat = make_fe_material();
    let photon_paths = HashMap::new();

    let result = mat.init_photon_data(&photon_paths);
    assert!(
        result.is_err(),
        "Should fail when photon data path is missing"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No photon data path"),
        "Error should mention missing path, got: {err}"
    );
}
