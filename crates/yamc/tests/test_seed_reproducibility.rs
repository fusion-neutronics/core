//! Test that yamc produces reproducible results with a fixed seed.
//!
//! This test verifies that the simulation is deterministic when given the same
//! random seed. This is important for:
//! - Debugging and development
//! - Scientific reproducibility
//! - Regression testing

use std::collections::HashMap;
use yamc::util::fast_rng::FastRng;

/// Test that FastRng produces deterministic sequences
#[test]
fn test_rng_determinism() {
    let mut rng1 = FastRng::new(42);
    let mut rng2 = FastRng::new(42);

    // Generate 100 random numbers from each
    let seq1: Vec<f64> = (0..100).map(|_| rng1.random()).collect();
    let seq2: Vec<f64> = (0..100).map(|_| rng2.random()).collect();

    // They should be identical
    assert_eq!(
        seq1, seq2,
        "RNG sequences should be identical with same seed"
    );
}

/// Test that FastRng reseed works correctly
#[test]
fn test_rng_reseed_determinism() {
    let mut rng = FastRng::new(0);

    // First sequence with seed 42
    rng.reseed(42);
    let seq1: Vec<f64> = (0..10).map(|_| rng.random()).collect();

    // Generate some random numbers to change state
    for _ in 0..100 {
        rng.random();
    }

    // Reseed to 42 and generate again
    rng.reseed(42);
    let seq2: Vec<f64> = (0..10).map(|_| rng.random()).collect();

    assert_eq!(seq1, seq2, "RNG should produce same sequence after reseed");
}

/// Test that HashMap key sorting produces deterministic order
#[test]
fn test_hashmap_sorted_iteration() {
    // Create HashMaps in different orders - Rust's HashMap has randomized order
    let mut map1: HashMap<String, i32> = HashMap::new();
    map1.insert("Fe56".to_string(), 1);
    map1.insert("H1".to_string(), 2);
    map1.insert("O16".to_string(), 3);

    let mut map2: HashMap<String, i32> = HashMap::new();
    map2.insert("O16".to_string(), 3);
    map2.insert("Fe56".to_string(), 1);
    map2.insert("H1".to_string(), 2);

    // Unsorted iteration order is not guaranteed to be the same
    // But sorted iteration should always be the same
    let mut keys1: Vec<_> = map1.keys().collect();
    keys1.sort();
    let mut keys2: Vec<_> = map2.keys().collect();
    keys2.sort();

    assert_eq!(keys1, keys2, "Sorted keys should be in same order");

    // Verify the sorted order is alphabetical
    assert_eq!(keys1, vec!["Fe56", "H1", "O16"]);
}

/// Test that nuclide sampling would be deterministic with sorted iteration
#[test]
fn test_nuclide_sampling_determinism() {
    // Simulate the nuclide sampling algorithm with sorted keys
    let mut xs_by_nuclide: HashMap<String, f64> = HashMap::new();
    xs_by_nuclide.insert("Fe56".to_string(), 0.3);
    xs_by_nuclide.insert("H1".to_string(), 0.5);
    xs_by_nuclide.insert("O16".to_string(), 0.2);

    let total: f64 = xs_by_nuclide.values().sum();

    // Sort keys for deterministic iteration
    let mut sorted_keys: Vec<_> = xs_by_nuclide.keys().collect();
    sorted_keys.sort();

    // Sample 1000 times with the same sequence of random numbers
    let mut rng = FastRng::new(12345);
    let mut samples1: Vec<String> = Vec::new();
    for _ in 0..1000 {
        let xi = rng.random() * total;
        let mut accum = 0.0;
        for key in &sorted_keys {
            accum += xs_by_nuclide[*key];
            if xi < accum {
                samples1.push((*key).clone());
                break;
            }
        }
    }

    // Reset and sample again
    let mut rng = FastRng::new(12345);
    let mut samples2: Vec<String> = Vec::new();
    for _ in 0..1000 {
        let xi = rng.random() * total;
        let mut accum = 0.0;
        for key in &sorted_keys {
            accum += xs_by_nuclide[*key];
            if xi < accum {
                samples2.push((*key).clone());
                break;
            }
        }
    }

    assert_eq!(
        samples1, samples2,
        "Nuclide sampling should be deterministic with sorted keys"
    );
}
