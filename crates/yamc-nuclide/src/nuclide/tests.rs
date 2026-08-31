/// Test arrow datasets live in `crates/yamc/tests/` (e.g. `Li6.arrow`,
/// `Be9.arrow`). They were not duplicated when `nuclide.rs` moved into
/// this crate -- the data is ~545MB and yamc-side tests still need it.
/// Resolve relative to this crate's manifest dir so `cargo test` works
/// regardless of which crate's runner is invoked.
fn td(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("yamc")
        .join("tests")
        .join(name)
}

#[test]
fn test_sample_absorption_constituent_li6() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");
    let temperature = nuclide
        .loaded_temperatures
        .first()
        .expect("No temperatures loaded");
    let energy = 1e6; // 1 MeV, typical fast neutron
    let mut rng = StdRng::seed_from_u64(12345);
    // Should not panic and should return a Reaction
    let reaction = nuclide.sample_absorption_constituent(energy, temperature, &mut rng);
    // Check that the sampled MT is one of the expected absorption constituent MTs
    let valid_mts: Vec<i32> = {
        let mut mts = vec![
            102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 155, 182,
            191, 192, 193, 197,
        ];
        mts.extend(600..650);
        mts.extend(650..700);
        mts.extend(700..750);
        mts.extend(750..800);
        mts.extend(800..850);
        mts
    };
    assert!(
        valid_mts.contains(&reaction.mt_number),
        "Sampled absorption constituent MT {} not in valid list",
        reaction.mt_number
    );
    // Print for debug
    println!("Sampled absorption constituent MT: {}", reaction.mt_number);
}
#[test]
fn test_get_or_load_nuclide_uses_cache() {
    use std::collections::HashMap;
    let li6_path = td("Li6.arrow");
    assert!(li6_path.exists(), "tests/Li6.arrow missing");
    // Only remove Li6 from cache, don't clear all (avoid race with other tests)
    {
        let mut cache = match super::GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        // Remove any Li6 entries (handle both normalized and non-normalized paths)
        let keys_to_remove: Vec<String> = cache
            .keys()
            .filter(|k| k.starts_with("Li6@") && k.contains("Li6.arrow"))
            .cloned()
            .collect();
        for key in keys_to_remove {
            cache.remove(&key);
        }
    }
    let li6_path = std::fs::canonicalize(td("Li6.arrow")).expect("tests/Li6.arrow missing");
    let raw = crate::nuclide_loader::load_nuclide(&li6_path, &crate::LoadScope::full())
        .expect("Direct read failed");
    assert_eq!(raw.name.as_deref(), Some("Li6"));
    // Don't assert cache state here (other tests may be using it)
    let mut path_map = HashMap::new();
    path_map.insert("Li6".to_string(), li6_path.to_string_lossy().to_string());
    let first = super::get_or_load_nuclide("Li6", &path_map, &crate::LoadScope::full())
        .expect("Initial cached load failed");
    // Ensure Li6 now present in cache with the correct cache key
    {
        let cache = match super::GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Check for Li6 cache key (path may be normalized)
        let found = cache
            .keys()
            .any(|k| k.starts_with("Li6@") && k.contains("Li6.arrow"));
        assert!(found, "Li6 should be present after cached load");
    }
    let second = super::get_or_load_nuclide("Li6", &path_map, &crate::LoadScope::full())
        .expect("Second cached load failed");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "Expected identical Arc pointer from cache on second load"
    );
    assert_eq!(
        first.name.as_deref(),
        raw.name.as_deref(),
        "Names differ between raw and cached"
    );
}

#[cfg(test)]
#[test]
fn test_sample_reaction_li6() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li6.arrow");
    let temperature = nuclide
        .loaded_temperatures
        .first()
        .expect("No temperatures loaded")
        .clone();

    // Vary energy from 1.0 to 15e6 (10 steps)
    let energies = (0..10).map(|i| 1.0 + i as f64 * (15e6 - 1e6) / 9.0);

    for energy in energies {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let mut rng3 = StdRng::seed_from_u64(43); // Different seed

        let mt1 = nuclide
            .sample_reaction(energy, &temperature, &mut rng1)
            .map(|r| r.mt_number);
        let mt2 = nuclide
            .sample_reaction(energy, &temperature, &mut rng2)
            .map(|r| r.mt_number);
        let mt3 = nuclide
            .sample_reaction(energy, &temperature, &mut rng3)
            .map(|r| r.mt_number);

        // Ensure reactions were sampled successfully
        assert!(
            mt1.is_some(),
            "sample_reaction returned None at energy {energy}"
        );
        assert!(
            mt2.is_some(),
            "Repeat sample with same seed returned None at energy {energy}"
        );
        assert!(
            mt3.is_some(),
            "Sample with different seed returned None at energy {energy}"
        );

        let mt1 = mt1.unwrap();
        let mt2 = mt2.unwrap();
        let mt3 = mt3.unwrap();

        // Ensure same-seed reactions are the same (determinism)
        assert_eq!(mt1, mt2, "Different MT for same seed at energy {energy}");

        // Print info about the third reaction (with different seed)
        println!("Energy: {energy:e}, MT (seed 42): {mt1}, MT (seed 43): {mt3}");

        // Ensure basic validity
        assert!(
            mt1 > 0,
            "Sampled reaction has invalid MT number at energy {energy}"
        );
    }
}

#[test]
fn test_reaction_mts_li6() {
    // Load Li6 nuclide from test Arrow
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");
    let mts = nuclide.reaction_mts().expect("No MTs found");
    // Check for some expected MTs
    // These are common MTs that should be present in Li6
    let expected = vec![2, 102, 103, 105];
    for mt in &expected {
        assert!(mts.contains(mt), "Expected MT {mt} in Li6");
    }
    println!("Li6 MTs: {mts:?}");
}

#[test]
fn test_reaction_mts_li7() {
    // Load Li7 nuclide from test Arrow
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li7.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li7.arrow");
    let mts = nuclide.reaction_mts().expect("No MTs found");
    // Check for presence of key MTs
    assert!(mts.contains(&2), "MT=2 should be present");
    assert!(!mts.is_empty(), "MT list should not be empty");
    println!("Li7 MTs: {mts:?}");
}

#[test]
fn test_fissionable_false_for_be9_and_fe58() {
    let nuclide_be9 =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");
    assert!(!nuclide_be9.fissionable, "Be9 should not be fissionable");

    let nuclide_fe58 =
        crate::nuclide_loader::load_nuclide(td("Fe58.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Fe58.arrow");
    assert!(!nuclide_fe58.fissionable, "Fe58 should not be fissionable");
}

#[test]
fn test_li6_reactions_contain_specific_mts() {
    // Load Li6 nuclide from test Arrow
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");

    // Check that MT 2 (elastic) is present
    let required = [2];
    for mt in &required {
        let mut found = false;
        for temp_reactions in nuclide.reactions.iter() {
            if let Some(reaction) = temp_reactions.get(mt) {
                assert_ne!(reaction.mt_number, 0, "Reaction MT number for MT {mt} is 0");
                found = true;
                break;
            }
        }
        assert!(found, "MT {mt} not found in any temperature reactions");
    }
}

#[test]
fn test_available_temperatures_be9() {
    let nuclide_be9 =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");
    // Temperature keys are stored without 'K' suffix (e.g., "294" not "294K")
    assert!(
        nuclide_be9
            .available_temperatures
            .iter()
            .any(|t| t.contains("294")),
        "available_temperatures should contain 294 variant, got {:?}",
        nuclide_be9.available_temperatures
    );
    let temps_method = nuclide_be9
        .temperatures()
        .expect("temperatures() returned None");
    assert!(
        !temps_method.is_empty(),
        "temperatures() should return at least one temperature"
    );
}

#[test]
fn test_be9_mt_numbers_per_temperature() {
    let nuclide_be9 =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    // Get the temperature key (stored without 'K' suffix)
    let temp_key = nuclide_be9
        .loaded_temperatures
        .first()
        .expect("No temperature found");

    // Helper closure to extract and sort MT list for a temperature
    let get_sorted_mts = |temp: &str| -> Vec<i32> {
        let mut mts: Vec<i32> = nuclide_be9
            .reactions_for_temp(temp)
            .expect("Temperature not found in reactions")
            .keys()
            .cloned()
            .collect();
        mts.sort();
        mts
    };

    let mts = get_sorted_mts(temp_key);
    println!("Be9 MTs at {temp_key}: {mts:?}");

    // Check some expected MTs are present
    assert!(mts.contains(&2), "Be9 should have MT=2 (elastic)");
}

#[test]
fn test_available_temperatures_fe56_includes_294() {
    let nuclide_fe56 =
        crate::nuclide_loader::load_nuclide(td("Fe56.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Fe56.arrow");
    assert!(
        nuclide_fe56
            .available_temperatures
            .iter()
            .any(|t| t.contains("294")),
        "Fe56 available_temperatures should contain '294' variant"
    );
    let temps_method = nuclide_fe56
        .temperatures()
        .expect("temperatures() returned None");
    assert!(
        !temps_method.is_empty(),
        "Fe56 temperatures() should return at least one temperature"
    );
}

#[test]
fn test_clear_nuclide_cache() {
    // Insert a test nuclide into the cache
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");
    let nuclide_arc = std::sync::Arc::new(nuclide);

    {
        let mut cache = match super::GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.insert(
            "Test_Nuclide".to_string(),
            std::sync::Arc::downgrade(&nuclide_arc),
        );

        // Verify it's in the cache
        assert!(
            cache.contains_key("Test_Nuclide"),
            "Test nuclide should be in cache before clearing"
        );
    }

    // Call the clear function
    super::clear_nuclide_cache();

    // Verify cache is now empty
    let cache = match super::GLOBAL_NUCLIDE_CACHE.lock() {
        Ok(cache) => cache,
        Err(poisoned) => poisoned.into_inner(),
    };
    assert!(
        !cache.contains_key("Test_Nuclide"),
        "Test nuclide should be removed after clear_nuclide_cache"
    );
}

#[test]
fn test_microscopic_cross_section_with_temperature() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    // Get the actual temperature key from the data
    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Test with specific temperature
    let result = nuclide.microscopic_cross_section(2, Some(&temp_key), false);
    assert!(
        result.is_ok(),
        "Should successfully get MT=2 data for {}: {:?}",
        temp_key,
        result.err()
    );

    let (xs, energy) = result.unwrap();
    assert!(!xs.is_empty(), "Cross section data should not be empty");
    assert!(!energy.is_empty(), "Energy data should not be empty");
    assert_eq!(
        xs.len(),
        energy.len(),
        "Cross section and energy arrays should have same length"
    );

    // Test with invalid temperature
    let result_invalid = nuclide.microscopic_cross_section(2, Some("999K"), false);
    assert!(
        result_invalid.is_err(),
        "Should fail for unavailable temperature"
    );
}

#[test]
fn test_microscopic_cross_section_single_temperature() {
    // Load Be9 with only one temperature
    let temps_filter = std::collections::HashSet::from(["294".to_string()]);
    let mut nuclide = crate::nuclide_loader::load_nuclide(
        td("Be9.arrow"),
        &crate::LoadScope::full().with_temperatures(Some(temps_filter.clone())),
    )
    .expect("Failed to load Be9.arrow with temperature filter");

    // Should work without specifying temperature since only one is loaded
    let result = nuclide.microscopic_cross_section(2, None, false);
    assert!(
        result.is_ok(),
        "Should successfully get MT=2 data without temperature: {:?}",
        result.err()
    );

    let (xs, energy) = result.unwrap();
    assert!(!xs.is_empty(), "Cross section data should not be empty");
    assert!(!energy.is_empty(), "Energy data should not be empty");
    assert_eq!(
        xs.len(),
        energy.len(),
        "Cross section and energy arrays should have same length"
    );
}

#[test]
fn test_microscopic_cross_section_invalid_mt() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Should fail for non-existent MT
    let result = nuclide.microscopic_cross_section(9999, Some(&temp_key), false);
    assert!(result.is_err(), "Should error for invalid MT number");

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("MT 9999 not found"),
        "Error should mention MT not found: {error_msg}"
    );
}

#[test]
fn test_microscopic_cross_section_multiple_mt_numbers() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Test common MT numbers that should exist in Be9
    let test_mts = [2, 102]; // elastic and capture should exist

    for mt in test_mts {
        let result = nuclide.microscopic_cross_section(mt, Some(&temp_key), false);
        if let Ok((xs, energy)) = result {
            assert!(!xs.is_empty(), "MT={mt} should have cross section data");
            assert!(!energy.is_empty(), "MT={mt} should have energy data");
            assert_eq!(xs.len(), energy.len(), "MT={mt} data length mismatch");

            // Validate data quality
            for &e in &energy {
                assert!(e > 0.0, "MT={mt} energy values should be positive");
            }
            for &x in &xs {
                assert!(x >= 0.0, "MT={mt} cross sections should be non-negative");
            }
        }
    }
}

#[test]
fn test_microscopic_cross_section_lithium() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li6.arrow");

    // Use 294K (standard room temp in ENDF data)
    let result = nuclide.microscopic_cross_section(2, Some("294"), false);
    assert!(
        result.is_ok(),
        "Should successfully get Li6 elastic scattering data: {:?}",
        result.err()
    );

    let (xs, energy) = result.unwrap();
    assert!(
        !xs.is_empty(),
        "Li6 elastic scattering data should not be empty"
    );
    assert!(!energy.is_empty(), "Li6 energy data should not be empty");
}

#[test]
fn test_cache_optimization_keyword_vs_path() {
    // Test that accessing the same file via keyword vs direct path uses same cache entry
    use std::collections::HashMap;

    super::clear_nuclide_cache();

    // Load Li6 from local file first
    let mut li6_map = HashMap::new();
    li6_map.insert(
        "Li6".to_string(),
        td("Li6.arrow").to_string_lossy().into_owned(),
    );
    let _first = super::get_or_load_nuclide("Li6", &li6_map, &crate::LoadScope::full())
        .expect("Initial file load failed");

    // Now try to load the same file by its absolute path
    let absolute_path = std::fs::canonicalize(td("Li6.arrow")).unwrap();
    let mut abs_map = HashMap::new();
    abs_map.insert(
        "Li6".to_string(),
        absolute_path.to_string_lossy().to_string(),
    );
    let _second = super::get_or_load_nuclide("Li6", &abs_map, &crate::LoadScope::full())
        .expect("Absolute path load failed");

    // Verify both entries use the same cache key (should only be one entry)
    {
        let cache = match super::GLOBAL_NUCLIDE_CACHE.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };

        // Should only have one cache entry since both resolve to the same file
        let li6_entries: Vec<_> = cache
            .keys()
            .filter(|k| k.starts_with("Li6@") && k.contains("Li6.arrow"))
            .collect();

        assert_eq!(
            li6_entries.len(),
            1,
            "Expected exactly 1 cache entry for Li6, found: {li6_entries:?}"
        );
    }
}

#[test]
fn test_microscopic_cross_section_string_reactions() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Test with MT number (existing functionality)
    let result_mt = nuclide.microscopic_cross_section(2, Some(&temp_key), false);
    assert!(result_mt.is_ok(), "Should work with MT number");
    let (xs_mt, energy_mt) = result_mt.unwrap();

    // Test with reaction name string
    let result_str = nuclide.microscopic_cross_section("(n,elastic)", Some(&temp_key), false);
    assert!(result_str.is_ok(), "Should work with reaction name string");
    let (xs_str, energy_str) = result_str.unwrap();

    // Results should be identical
    assert_eq!(
        xs_mt, xs_str,
        "Cross section data should be identical for MT 2 and '(n,elastic)'"
    );
    assert_eq!(
        energy_mt, energy_str,
        "Energy data should be identical for MT 2 and '(n,elastic)'"
    );
}

#[test]
fn test_microscopic_cross_section_invalid_reaction_string() {
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Test with invalid reaction name
    let result = nuclide.microscopic_cross_section("(n,invalid)", Some(&temp_key), false);
    assert!(result.is_err(), "Should fail for invalid reaction name");

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("Unknown reaction name"),
        "Error should mention unknown reaction name"
    );
}

#[test]
fn test_microscopic_cross_section_fission_alias() {
    // Note: Be9 is not fissionable, so we'll just test the string recognition
    // The fission alias should map to MT 18
    let mut nuclide =
        crate::nuclide_loader::load_nuclide(td("Be9.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Be9.arrow");

    let temp_key = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Test that "fission" string is recognized (even though Be9 doesn't have fission reactions)
    let result = nuclide.microscopic_cross_section("fission", Some(&temp_key), false);

    // Should fail because Be9 doesn't have MT 18, but the error should be about missing MT, not unknown reaction
    assert!(
        result.is_err(),
        "Should fail because Be9 doesn't have fission reactions"
    );
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("MT 18 not found"),
        "Error should be about missing MT 18, not unknown reaction name"
    );
}

#[test]
fn test_nuclide_different_data_sources() {
    // Test that loading the same nuclide from different sources gives different results

    // Clear cache to ensure fresh loads
    crate::nuclide::clear_nuclide_cache();

    // For this test, we'll use different file paths as data sources
    let mut li6_file =
        crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li6 from file");

    let mut li7_file =
        crate::nuclide_loader::load_nuclide(td("Li7.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li7 from file");

    // Get cross sections from both (different nuclides will have different data)
    let (xs_li6, _) = li6_file
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li6 cross section");
    let (xs_li7, _) = li7_file
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li7 cross section");

    // Should have different data since they're different nuclides
    let data_different = xs_li6.len() != xs_li7.len()
        || xs_li6
            .iter()
            .zip(&xs_li7)
            .any(|(a, b)| (a - b).abs() > 1e-10);

    assert!(data_different, "Li6 and Li7 data should be different");
    println!("Li6: {} points, Li7: {} points", xs_li6.len(), xs_li7.len());
}

#[test]
fn test_nuclide_file_vs_keyword_sources() {
    // Test that file paths and keywords can coexist in cache

    crate::nuclide::clear_nuclide_cache();

    // Load Li6 from local file
    let mut li6_file =
        crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li6 from file");

    // Load Li7 from local file (different nuclide, different source)
    let mut li7_file =
        crate::nuclide_loader::load_nuclide(td("Li7.arrow"), &crate::LoadScope::full())
            .expect("Failed to load Li7 from file");

    assert_eq!(li6_file.name.as_deref(), Some("Li6"));
    assert_eq!(li7_file.name.as_deref(), Some("Li7"));

    // Verify we can load cross sections from both
    let (xs_li6, _) = li6_file
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li6 cross section");
    let (xs_li7, _) = li7_file
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li7 cross section");

    assert!(
        !xs_li6.is_empty() && !xs_li7.is_empty(),
        "Both should have cross section data"
    );
}

#[test]
fn test_nuclide_cache_respects_data_source_boundaries() {
    // Test that the cache properly separates different data sources

    crate::nuclide::clear_nuclide_cache();

    // Load Li6 from file first time
    let mut li6_1 = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6 first time");

    // Load Li7 from file (different nuclide/source)
    let mut li7 = crate::nuclide_loader::load_nuclide(td("Li7.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li7");

    // Load Li6 from file again (should use cache)
    let mut li6_2 = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6 second time");

    // Get cross sections
    let (xs_li6_1, _) = li6_1
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li6 cross section first time");
    let (xs_li7, _) = li7
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li7 cross section");
    let (xs_li6_2, _) = li6_2
        .microscopic_cross_section(102, Some("294"), false)
        .expect("Failed to get Li6 cross section second time");

    // Li6 loads should be identical (cache working)
    assert_eq!(
        xs_li6_1, xs_li6_2,
        "Li6 loads should be identical (cache working)"
    );

    // Li6 vs Li7 should be different (different nuclides)
    let li6_vs_li7_different = xs_li6_1.len() != xs_li7.len()
        || xs_li6_1
            .iter()
            .zip(&xs_li7)
            .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(
        li6_vs_li7_different,
        "Li6 and Li7 should have different data"
    );
}

#[test]
fn test_sample_inelastic_constituent() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Load Li6 which should have some inelastic reactions (MT 50-91)
    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");
    let temperature = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());

    // Check what inelastic constituent MTs are available
    let available_mts = nuclide.reaction_mts().unwrap_or_default();
    let inelastic_mts: Vec<i32> = available_mts
        .into_iter()
        .filter(|&mt| (50..92).contains(&mt))
        .collect();

    if inelastic_mts.is_empty() {
        println!("No inelastic constituent reactions (MT 50-91) found in Li6, skipping test");
        return;
    }

    println!("Available inelastic constituent MTs in Li6: {inelastic_mts:?}");

    // Test sampling above the inelastic threshold. endf-b8.1 Li6 has a single
    // inelastic constituent, MT 52, with a ~4.16 MeV threshold, so sample at
    // 7 MeV where it carries cross section. (The prior 3.75 MeV sat below the
    // VIII.1 threshold; VIII.0 had a lower-threshold level.)
    let energies = [0.7e7];
    for energy in energies {
        for i in 0..5 {
            let mut rng = StdRng::seed_from_u64(42 + i);
            let reaction = nuclide.sample_inelastic_constituent(energy, &temperature, &mut rng);
            println!(
                "Sample {}: Energy {:.1e}: Sampled inelastic MT {}",
                i + 1,
                energy,
                reaction.mt_number
            );
            // Verify the sampled MT is in the expected range
            assert!(
                reaction.mt_number >= 50 && reaction.mt_number < 92,
                "Sampled MT {} should be in range 50-91",
                reaction.mt_number
            );
        }
    }
}

#[test]
fn test_sample_inelastic_constituent_deterministic() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let nuclide = crate::nuclide_loader::load_nuclide(td("Li6.arrow"), &crate::LoadScope::full())
        .expect("Failed to load Li6.arrow");
    let temperature = nuclide
        .loaded_temperatures
        .first()
        .cloned()
        .unwrap_or("294".to_string());
    let energy = 1.0e7;

    // Same seed should give same result
    let mut rng1 = StdRng::seed_from_u64(12345);
    let mut rng2 = StdRng::seed_from_u64(12345);

    let reaction1 = nuclide.sample_inelastic_constituent(energy, &temperature, &mut rng1);
    let reaction2 = nuclide.sample_inelastic_constituent(energy, &temperature, &mut rng2);
    assert_eq!(
        reaction1.mt_number, reaction2.mt_number,
        "Same seed should give same inelastic constituent reaction"
    );
}

/// Issue #106: slotting MT 11 / 29 / 30 / 35 / 36 / 42 must not renumber any
/// stream. They used to fall into the trailing "not in the table" group, which
/// `build_inelastic_walk_order` appends in storage order; they now sit at the
/// end of the table in ascending MT order. Since libraries store reactions in
/// ascending MT order, the walk is the same sequence either way, so a fixed
/// `xi_mt` still selects the same channel as before.
#[test]
fn breakup_mts_keep_their_position_in_the_walk() {
    use crate::nuclide::{FastXSGrid, INELASTIC_MT_SLOTS};

    // A TENDL-shaped nuclide: elastic, the usual inelastic levels, and the
    // breakup channels, in the ascending MT order the data is stored in.
    let stored: Vec<i32> = vec![2, 5, 11, 16, 17, 22, 29, 30, 42, 51, 52, 91];
    let elastic_idx = stored.iter().position(|&mt| mt == 2);

    let order = FastXSGrid::build_inelastic_walk_order(&stored, elastic_idx);
    let walked: Vec<i32> = order.iter().map(|&j| stored[j]).collect();

    // Every non-elastic channel is visited exactly once, elastic never.
    assert_eq!(walked.len(), stored.len() - 1);
    assert!(
        !walked.contains(&2),
        "elastic must be dropped from the walk"
    );
    for &mt in stored.iter().filter(|&&mt| mt != 2) {
        assert_eq!(
            walked.iter().filter(|&&m| m == mt).count(),
            1,
            "MT {mt} should be walked exactly once"
        );
    }

    // Every walked MT is now tabled, so the trailing storage-order group is
    // empty for this nuclide and the walk follows the table exactly.
    for &mt in &walked {
        assert!(
            INELASTIC_MT_SLOTS.contains(&mt),
            "MT {mt} should be slotted after issue #106"
        );
    }
    let table_order: Vec<i32> = INELASTIC_MT_SLOTS
        .iter()
        .copied()
        .filter(|mt| walked.contains(mt))
        .collect();
    assert_eq!(walked, table_order);

    // And the breakup channels keep ascending MT order among themselves, which
    // is what makes this a no-op for the streams that already existed.
    let breakup: Vec<i32> = walked
        .iter()
        .copied()
        .filter(|mt| [11, 29, 30, 35, 36, 42].contains(mt))
        .collect();
    let mut sorted = breakup.clone();
    sorted.sort_unstable();
    assert_eq!(breakup, sorted);
}
