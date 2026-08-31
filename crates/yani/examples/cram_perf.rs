//! CRAM perf probe -- timed against the full ENDF-b8.1 chain.
//!
//! Run as: `cargo run --release -p yani --example cram_perf`
//!
//! Use this when changing anything that touches faer features or the CRAM
//! solver path to check for regressions. Resolves the chain fixture relative
//! to the crate manifest, so it works from any cwd.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use yani::{
    build_matrix_triplets, cram48_sparse, parse_chain_arrow, ChainParts, FissionYieldWeights,
    ReactionRates,
};

fn main() {
    let chain_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "transmutation-endf-b8.1-sfr.arrow",
    ]
    .iter()
    .collect();
    let chain = parse_chain_arrow(&chain_path).expect("load chain");
    let mut names: Vec<String> = chain.keys().cloned().collect();
    names.sort();
    let n = names.len();

    // Plant some reaction rates on actinides and structural nuclides so the
    // matrix has a representative coupling pattern (not just decay-only).
    let mut rates: ReactionRates = HashMap::new();
    for name in [
        "U235", "U238", "Pu239", "Pu240", "Pu241", "Th232", "Fe56", "Cr52", "Ni58", "Li6",
    ] {
        let mut r = HashMap::new();
        r.insert("(n,gamma)".to_string(), 1.0e-9);
        // "fission", not "(n,fission)": the rate key has to match the chain's
        // own spelling of the reaction or the fission channel (and with it
        // every fission product) is silently absent from the matrix.
        r.insert("fission".to_string(), 5.0e-10);
        r.insert("(n,2n)".to_string(), 1.0e-11);
        rates.insert(name.to_string(), r);
    }

    // Spread the fission-yield fold evenly over every tabulated energy: the
    // worst case for issue #379's combination, since each product then draws
    // on all of a nuclide's yield vectors rather than the one or two a real
    // spectrum reaches.
    let mut fy_weights: FissionYieldWeights = HashMap::new();
    for name in rates.keys() {
        if let Some(fy) = chain.get(name).and_then(|c| c.fission_yields.as_ref()) {
            let k = fy.yields.len();
            if k > 0 {
                fy_weights.insert(name.clone(), vec![1.0 / k as f64; k]);
            }
        }
    }

    let (triplets, dim) =
        build_matrix_triplets(&chain, &names, &rates, &fy_weights, ChainParts::default())
            .expect("build matrix");
    println!("chain: {n} nuclides | nnz: {}", triplets.len());

    let mut n0 = vec![0.0; dim];
    if let Some(i) = names.iter().position(|s| s == "U235") {
        n0[i] = 1.0e22;
    }
    if let Some(i) = names.iter().position(|s| s == "U238") {
        n0[i] = 1.0e23;
    }
    let dt = 86400.0 * 30.0; // 30 days

    // Warm up (first call may do extra one-time work like trait dispatch)
    let _ = cram48_sparse(&triplets, dim, &n0, dt).unwrap();

    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(cram48_sparse(&triplets, dim, &n0, dt).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "cram48_sparse: {:?} per iter ({:?} total over {} iters)",
        elapsed / iters,
        elapsed,
        iters
    );
}
