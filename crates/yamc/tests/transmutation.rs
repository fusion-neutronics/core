/// Integration tests for coupled transport-transmutation.
///
/// Tests the full transmutation pipeline using the actual chain file
/// and compares results against analytical solutions where possible.
use std::collections::HashMap;
use std::sync::Arc;
use yamc_materials::Material;
use yani_transmute::{
    cram48, load_chain_parts, ChainNuclide, ForwardEulerStepper, ReactionRates, TransmutationDriver,
};

/// Load the shared test chain from the v2 per-subsection fixture directory.
fn load_test_chain() -> Arc<HashMap<String, ChainNuclide>> {
    let base = format!(
        "{}/tests/transmutation-endf-b8.1-sfr.arrow",
        env!("CARGO_MANIFEST_DIR")
    );
    load_chain_parts(
        &format!("{base}/decay"),
        Some(&format!("{base}/reactions")),
        Some(&format!("{base}/fission_yields")),
        None,
    )
    .unwrap()
    .chain
}

// ============================================================
// Test 1: Pure decay (Co60 -> Ni60)
// ============================================================

#[test]
fn test_decay_co60_analytical() {
    let chain = load_test_chain();

    // Create material with Co60
    let mut material = Material::new(
        HashMap::from([("Co60".into(), 1.0e-3)]), // atoms/barn-cm
        "atom",
        "sum",
        None,
    )
    .unwrap();
    material.set_material_id(1);
    material.transmutable = true;

    let driver = TransmutationDriver::new(chain.clone(), Box::new(ForwardEulerStepper));

    // No reactions (decay only)
    let rates: ReactionRates = HashMap::new();

    // Co60 half-life from chain
    let co60 = chain.get("Co60").expect("Co60 not in chain");
    let half_life = co60.half_life.expect("Co60 should have half-life");

    // Transmute for one half-life
    let result = driver
        .transmute_material(&material, &rates, &HashMap::new(), half_life)
        .unwrap();

    // Verify Co60 decreased to ~50%
    let co60_final = *result.nuclides.get("Co60").unwrap_or(&0.0);
    let expected = 0.5 * 1.0e-3;
    let rel_error = (co60_final - expected).abs() / expected;
    assert!(
        rel_error < 1e-4,
        "Co60 after 1 half-life: expected ~{:.6e}, got {:.6e} (rel_err={:.2e})",
        expected,
        co60_final,
        rel_error
    );

    // Verify Ni60 produced (should be ~50% of initial Co60)
    let ni60_final = *result.nuclides.get("Ni60").unwrap_or(&0.0);
    let expected_ni60 = 1.0e-3 - co60_final;
    assert!(ni60_final > 0.0, "Ni60 should be produced from Co60 decay");
    let rel_error_ni = if expected_ni60 > 0.0 {
        (ni60_final - expected_ni60).abs() / expected_ni60
    } else {
        0.0
    };
    assert!(
        rel_error_ni < 1e-2,
        "Ni60 after 1 half-life: expected ~{:.6e}, got {:.6e} (rel_err={:.2e})",
        expected_ni60,
        ni60_final,
        rel_error_ni
    );
}

#[test]
fn test_decay_multiple_half_lives() {
    let chain = load_test_chain();

    let mut material = Material::new(
        HashMap::from([("Co60".into(), 1.0e-3)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();
    material.set_material_id(1);
    material.transmutable = true;

    let driver = TransmutationDriver::new(chain.clone(), Box::new(ForwardEulerStepper));
    let rates: ReactionRates = HashMap::new();

    let co60 = chain.get("Co60").unwrap();
    let half_life = co60.half_life.unwrap();

    // Transmute for 5 half-lives: should have ~3.125% remaining
    let dt = 5.0 * half_life;
    let result = driver
        .transmute_material(&material, &rates, &HashMap::new(), dt)
        .unwrap();

    let co60_final = *result.nuclides.get("Co60").unwrap_or(&0.0);
    let expected = (0.5_f64).powi(5) * 1.0e-3; // ~3.125e-5
    let rel_error = (co60_final - expected).abs() / expected;
    assert!(
        rel_error < 1e-4,
        "Co60 after 5 half-lives: expected ~{:.6e}, got {:.6e} (rel_err={:.2e})",
        expected,
        co60_final,
        rel_error
    );
}

// ============================================================
// Test 2: Simple activation with known reaction rates
// ============================================================

#[test]
fn test_activation_with_reaction_rates() {
    let chain = load_test_chain();

    // Use I135 which has (n,gamma) reaction in the chain
    let mut material = Material::new(
        HashMap::from([("I135".into(), 1.0e20)]), // atoms/barn-cm
        "atom",
        "sum",
        None,
    )
    .unwrap();
    material.set_material_id(1);
    material.transmutable = true;

    let driver = TransmutationDriver::new(chain.clone(), Box::new(ForwardEulerStepper));

    // Set up reaction rates: I135 (n,gamma) -> Xe136
    let mut rates: ReactionRates = HashMap::new();
    let mut i135_rates = HashMap::new();
    i135_rates.insert("(n,gamma)".to_string(), 2.0e-5); // sigma*phi = 2e-5 /s
    rates.insert("I135".to_string(), i135_rates);

    // Transmute for 1 day
    let dt = 86400.0;
    let result = driver
        .transmute_material(&material, &rates, &HashMap::new(), dt)
        .unwrap();

    // I135 should decrease (both from decay and reaction)
    let i135_final = *result.nuclides.get("I135").unwrap_or(&0.0);
    assert!(
        i135_final < 1.0e20,
        "I135 should decrease from activation + decay"
    );
    assert!(i135_final > 0.0, "I135 shouldn't be zero after 1 day");

    // Products should be created (Xe135 from decay, Xe136 from n,gamma)
    // Check I135 has the expected loss rate
    let i135_info = chain.get("I135").unwrap();
    let has_decay = i135_info.half_life.is_some();
    let has_reaction = i135_info.reactions.iter().any(|r| r.kind == "(n,gamma)");
    assert!(
        has_decay || has_reaction,
        "I135 should have decay or reaction channels"
    );
}

// ============================================================
// Test 3: Multi-step transmutation (irradiation + cooling)
// ============================================================

#[test]
fn test_multistep_irradiation_cooling() {
    let chain = load_test_chain();

    let initial_density = 1.0e-3;
    let mut material = Material::new(
        HashMap::from([("Co60".into(), initial_density)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();
    material.set_material_id(1);
    material.transmutable = true;

    let driver = TransmutationDriver::new(chain.clone(), Box::new(ForwardEulerStepper));

    // Step 1: "Irradiation" for 1 year (with zero reaction rate - just decay)
    let one_year = 365.25 * 24.0 * 3600.0;
    let rates_zero: ReactionRates = HashMap::new();

    let result1 = driver
        .transmute_material(&material, &rates_zero, &HashMap::new(), one_year)
        .unwrap();
    let co60_after_irrad = *result1.nuclides.get("Co60").unwrap_or(&0.0);

    // Step 2: Cooling for 1 year (also just decay for Co60)
    let result2 = driver
        .transmute_material(&result1, &rates_zero, &HashMap::new(), one_year)
        .unwrap();
    let co60_after_cool = *result2.nuclides.get("Co60").unwrap_or(&0.0);

    // Co60 half-life ~5.27 years
    // After 2 years: N = N0 * exp(-lambda * 2y)
    let co60_chain = chain.get("Co60").unwrap();
    let half_life = co60_chain.half_life.unwrap();
    let lambda = std::f64::consts::LN_2 / half_life;
    let expected_2y = initial_density * (-lambda * 2.0 * one_year).exp();

    let rel_error = (co60_after_cool - expected_2y).abs() / expected_2y;
    assert!(
        rel_error < 1e-3,
        "Co60 after 2 years: expected {:.6e}, got {:.6e} (rel_err={:.2e})",
        expected_2y,
        co60_after_cool,
        rel_error
    );

    // Verify monotonic decrease
    assert!(co60_after_irrad < initial_density);
    assert!(co60_after_cool < co60_after_irrad);
}

// ============================================================
// Test 4: Mass conservation
// ============================================================

#[test]
fn test_mass_conservation_simple_chain() {
    let chain = load_test_chain();

    // Simple case: one parent decaying to stable daughter
    let initial_density = 1.0e-3;
    let mut material = Material::new(
        HashMap::from([("Co60".into(), initial_density)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();
    material.set_material_id(1);
    material.transmutable = true;

    let driver = TransmutationDriver::new(chain.clone(), Box::new(ForwardEulerStepper));
    let rates: ReactionRates = HashMap::new();

    // Transmute for 2 half-lives
    let co60_chain = chain.get("Co60").unwrap();
    let half_life = co60_chain.half_life.unwrap();
    let dt = 2.0 * half_life;

    let result = driver
        .transmute_material(&material, &rates, &HashMap::new(), dt)
        .unwrap();

    // Sum of all nuclides should approximately equal initial (mass conservation)
    // Note: some nuclides might be below our cutoff threshold (1e-30)
    let total: f64 = result.nuclides.values().sum();
    let rel_error = (total - initial_density).abs() / initial_density;
    assert!(
        rel_error < 1e-3,
        "Mass conservation violated: initial={:.6e}, final_sum={:.6e} (rel_err={:.2e})",
        initial_density,
        total,
        rel_error
    );
}

// ============================================================
// Test 5: TransmutationResults structure
// ============================================================

#[test]
fn test_transmutation_results_structure() {
    use yani_transmute::TransmutationResults;

    let timesteps = vec![86400.0, 86400.0, 86400.0]; // 3 days
    let source_rates = vec![1e14, 1e14, 0.0]; // 2 days irradiation, 1 day cooling

    let mut results = TransmutationResults::new(timesteps.clone(), source_rates.clone());

    // Add initial material
    let mut mat0 = Material::new(
        HashMap::from([("Co60".into(), 1.0e-3)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();
    mat0.set_material_id(1);
    results.add_initial(1, mat0.clone());

    // Add step results
    let mut mat1 = mat0.clone();
    mat1.nuclides.insert("Co60".to_string(), 0.9e-3);
    results.add_step(1, mat1);

    let mut mat2 = mat0.clone();
    mat2.nuclides.insert("Co60".to_string(), 0.8e-3);
    results.add_step(1, mat2);

    let mut mat3 = mat0.clone();
    mat3.nuclides.insert("Co60".to_string(), 0.7e-3);
    results.add_step(1, mat3);

    // Test accessors
    assert_eq!(results.num_steps(), 3);

    // Times should be cumulative
    assert_eq!(results.times.len(), 4); // initial + 3 steps
    assert_eq!(results.times[0], 0.0);
    assert!((results.times[1] - 86400.0).abs() < 1e-10);
    assert!((results.times[2] - 172800.0).abs() < 1e-10);
    assert!((results.times[3] - 259200.0).abs() < 1e-10);

    // Get material at specific step
    let mat_step0 = results.get_material(1, 0).unwrap();
    assert!((mat_step0.nuclides["Co60"] - 1.0e-3).abs() < 1e-10);

    let mat_step3 = results.get_material(1, 3).unwrap();
    assert!((mat_step3.nuclides["Co60"] - 0.7e-3).abs() < 1e-10);

    // Get final material
    let final_mat = results.get_final_material(1).unwrap();
    assert!((final_mat.nuclides["Co60"] - 0.7e-3).abs() < 1e-10);

    // Get nuclide evolution
    let evolution = results.get_nuclide_evolution(1, "Co60").unwrap();
    assert_eq!(evolution.len(), 4);
    assert!((evolution[0] - 1.0e-3).abs() < 1e-10);
    assert!((evolution[3] - 0.7e-3).abs() < 1e-10);
}

// ============================================================
// Test 6: Validate material for transmutation
// ============================================================

#[test]
fn test_validate_for_transmutation() {
    // Valid transmutable material
    let mut mat =
        Material::new(HashMap::from([("U235".into(), 0.05)]), "atom", "sum", None).unwrap();
    mat.transmutable = true;
    mat.volume = Some(1.0);
    assert!(mat.validate_for_transmutation().is_ok());

    // Not transmutable
    let mat2 = Material::new(HashMap::from([("U235".into(), 0.05)]), "atom", "sum", None).unwrap();
    assert!(mat2.validate_for_transmutation().is_err());

    // No nuclides
    let mut mat3 = Material::new(HashMap::new(), "atom", "sum", None).unwrap();
    mat3.transmutable = true;
    assert!(mat3.validate_for_transmutation().is_err());
}

// ============================================================
// Test 7: to_sum_mode conversion
// ============================================================

#[test]
fn test_to_sum_mode_already_sum() {
    let mat = Material::new(
        HashMap::from([("U235".into(), 1.0e-4)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();

    let converted = mat.to_sum_mode().unwrap();
    assert_eq!(converted.density_units.as_str(), "sum");
    assert!((converted.nuclides["U235"] - 1.0e-4).abs() < 1e-15);
}

// ============================================================
// Test 8: Chain caching works correctly
// ============================================================

#[test]
fn test_chain_caching() {
    // Load twice - second should be from cache
    let chain1 = load_test_chain();
    let chain2 = load_test_chain();

    // Should be same Arc (pointer equality)
    assert!(Arc::ptr_eq(&chain1, &chain2));

    // Verify chain has expected nuclides
    assert!(chain1.contains_key("Co60"));
    assert!(chain1.contains_key("U235"));
    assert!(chain1.contains_key("Pu239"));
}

// ============================================================
// Test 9: CRAM48 accuracy for various timescales
// ============================================================

#[test]
fn test_cram48_long_time() {
    let half_life = 1.0e6;
    let lambda = std::f64::consts::LN_2 / half_life;
    let a = vec![-lambda];
    let n0 = vec![1.0e20];
    let dt = 10.0 * half_life;

    let result = cram48(&a, 1, &n0, dt).unwrap();
    let expected = n0[0] * (-lambda * dt).exp();
    let rel_error = (result[0] - expected).abs() / expected;
    assert!(
        rel_error < 1e-14,
        "CRAM48 10 half-lives: rel_error = {:.2e}",
        rel_error
    );
}

#[test]
fn test_cram48_short_time() {
    let half_life = 1.0e6;
    let lambda = std::f64::consts::LN_2 / half_life;
    let a = vec![-lambda];
    let n0 = vec![1.0e20];
    let dt = 0.001 * half_life;

    let result = cram48(&a, 1, &n0, dt).unwrap();
    let expected = n0[0] * (-lambda * dt).exp();
    let rel_error = (result[0] - expected).abs() / expected;
    assert!(
        rel_error < 1e-14,
        "CRAM48 0.001 half-lives: rel_error = {:.2e}",
        rel_error
    );
}

// ============================================================
// Test 10: Compute reaction rates with driver
// ============================================================

#[test]
fn test_compute_reaction_rates_zero_flux() {
    let chain = load_test_chain();

    let material = Material::new(
        HashMap::from([("U235".into(), 1.0e-3)]),
        "atom",
        "sum",
        None,
    )
    .unwrap();

    let driver = TransmutationDriver::new(chain, Box::new(ForwardEulerStepper));

    // Zero flux should give empty rates
    let rates = driver.compute_reaction_rates(&material, 0.0);
    assert!(rates.is_empty(), "Zero flux should give no reaction rates");
}
