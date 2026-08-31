//! Issue #502: the transmutation matrix must be bit-reproducible.
//!
//! `Material.transmute()` called twice on identical inputs, in the same
//! process, returned two different inventories. There is no Monte Carlo in that
//! path, so the only candidate was a float sum taking its order from a
//! container that does not promise one.
//!
//! It was the reaction-kind grouping in `accumulate_matrix`: a `HashMap` built
//! fresh per nuclide per call, feeding both the diagonal loss sum and the order
//! the matrix entries were emitted in. Rust seeds each `HashMap` instance
//! separately, so consecutive calls in one process walked the kinds differently
//! and rounded differently.
//!
//! This drives the real ENDF/B-VIII.1 SFR chain rather than a fixture with two
//! nuclides, because the reported reach scaled with the chain: 2 of 28 values
//! on a steel, 153 of 486 plus two membership flips on a fissile case, where
//! the fission-yield fold sums hundreds of products into each row.

use std::collections::HashMap;
use std::path::PathBuf;

use yani::{
    build_matrix_triplets, parse_chain_arrow, ChainNuclide, ChainParts, FissionYieldWeights,
};

fn chain() -> HashMap<String, ChainNuclide> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/transmutation-endf-b8.1-sfr.arrow");
    parse_chain_arrow(&path).expect("parse the test chain")
}

/// Every nuclide the chain names, in the sorted order the stepper uses.
fn all_names(chain: &HashMap<String, ChainNuclide>) -> Vec<String> {
    let mut names: Vec<String> = chain.keys().cloned().collect();
    names.sort();
    names
}

/// A rate on every kind every nuclide carries.
///
/// The magnitudes are deliberately spread: `1.0 + 1.0e-16` rounds back to
/// `1.0`, so a column that adds the small terms to the large one first loses
/// them, while one that adds them to each other first does not. That is the
/// same non-associativity the bug exposed, made loud enough to see in one step.
fn rates(chain: &HashMap<String, ChainNuclide>) -> HashMap<String, HashMap<String, f64>> {
    chain
        .iter()
        .map(|(name, nuc)| {
            let per_kind = nuc
                .reactions
                .iter()
                .enumerate()
                .map(|(i, rx)| {
                    let rate = if i % 4 == 0 { 1.0 } else { 1.0e-16 };
                    (rx.kind.clone(), rate)
                })
                .collect();
            (name.clone(), per_kind)
        })
        .collect()
}

/// All the weight on each fissionable nuclide's first tabulated energy.
fn weights(chain: &HashMap<String, ChainNuclide>) -> FissionYieldWeights {
    chain
        .iter()
        .filter_map(|(name, nuc)| {
            let fy = nuc.fission_yields.as_ref()?;
            let mut c = vec![0.0; fy.yields.len()];
            *c.first_mut()? = 1.0;
            Some((name.clone(), c))
        })
        .collect()
}

/// Rebuilding the same matrix must give the same bits, every time.
///
/// Rebuilds rather than comparing one result with itself: the grouping map is
/// constructed inside the builder, so a fresh call is the only way to get a
/// fresh iteration order.
#[test]
fn the_matrix_is_bit_identical_across_builds() {
    let chain = chain();
    let names = all_names(&chain);
    let rates = rates(&chain);
    let weights = weights(&chain);

    let (first, n) = build_matrix_triplets(&chain, &names, &rates, &weights, ChainParts::default())
        .expect("build");
    assert!(n > 100, "expected a chain worth testing, got {n} nuclides");

    // Compared as a dense accumulation rather than triplet by triplet: that is
    // what the solver consumes, and summing duplicate (row, col) entries is
    // itself order-sensitive, so it is the sharper comparison.
    let dense = |triplets: &[(usize, usize, f64)]| {
        let mut a = HashMap::new();
        for &(row, col, value) in triplets {
            *a.entry((row, col)).or_insert(0.0f64) += value;
        }
        let mut cells: Vec<((usize, usize), u64)> =
            a.into_iter().map(|(k, v)| (k, v.to_bits())).collect();
        cells.sort();
        cells
    };

    let expected = dense(&first);
    for i in 1..25 {
        let (again, _) =
            build_matrix_triplets(&chain, &names, &rates, &weights, ChainParts::default())
                .expect("build");
        assert_eq!(dense(&again), expected, "build {i} differs from the first");
    }
}
