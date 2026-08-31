//! Issue #379: fission product inventory must follow the neutron spectrum.
//!
//! Drives the matrix builder with the real ENDF/B-VIII.1 U235 and Pu239 yield
//! tables from the test chain, once with all the weight on the thermal point
//! (what `yields.first()` used to pick) and once on 14 MeV (what a fusion
//! source actually sees), and checks the symmetric-fission valley fills in.

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

/// Mass number from a GNDS name such as `Xe135` or `Te131_m1`.
fn mass_number(name: &str) -> Option<u32> {
    let digits: String = name
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Total production rate of the symmetric-fission valley (A 110 to 125) in
/// `parent`'s matrix column, with all the fold's weight on tabulated point `k`.
fn valley_rate(chain: &HashMap<String, ChainNuclide>, parent: &str, k: usize) -> f64 {
    let fy = chain[parent]
        .fission_yields
        .as_ref()
        .expect("parent carries fission yields");

    // Just the parent and its products: the same set the stepper's BFS walks,
    // and small enough to keep the matrix cheap.
    let mut names: Vec<String> = fy
        .yields
        .iter()
        .flat_map(|y| y.products.iter().map(|(p, _)| p.clone()))
        .filter(|p| chain.contains_key(p))
        .collect();
    names.push(parent.to_string());
    names.sort();
    names.dedup();

    let rate = 1.0;
    let rates = HashMap::from([(
        parent.to_string(),
        HashMap::from([("fission".to_string(), rate)]),
    )]);
    let mut c = vec![0.0; fy.yields.len()];
    c[k] = 1.0;
    let weights = FissionYieldWeights::from([(parent.to_string(), c)]);

    let (triplets, _) =
        build_matrix_triplets(chain, &names, &rates, &weights, ChainParts::default())
            .expect("build");
    let col = names.iter().position(|n| n == parent).unwrap();
    triplets
        .iter()
        .filter(|(row, c, _)| *c == col && *row != col)
        .filter(
            |(row, _, _)| matches!(mass_number(&names[*row]), Some(a) if (110..=125).contains(&a)),
        )
        .map(|(_, _, v)| v)
        .sum()
}

#[test]
fn u235_yields_are_tabulated_at_three_ascending_energies() {
    let chain = chain();
    let fy = chain["U235"].fission_yields.as_ref().unwrap();
    assert_eq!(fy.energies(), vec![0.0253, 5.0e5, 1.4e7]);
}

#[test]
fn u235_symmetric_valley_fills_in_at_14_mev() {
    // The defect this issue is about: a 14.06 MeV broomstick scored the
    // thermal yield vector, which is roughly two orders of magnitude too low
    // through the symmetric-fission valley.
    let chain = chain();
    let thermal = valley_rate(&chain, "U235", 0);
    let fast = valley_rate(&chain, "U235", 2);

    assert!(
        thermal > 0.0 && fast > 0.0,
        "both spectra produce something"
    );
    assert!(
        fast / thermal > 10.0,
        "U235 valley should rise by more than an order of magnitude, \
         got {fast:.4e} / {thermal:.4e} = {:.1}x",
        fast / thermal
    );
}

#[test]
fn pu239_symmetric_valley_fills_in_at_14_mev() {
    let chain = chain();
    let fy = chain["Pu239"].fission_yields.as_ref().unwrap();
    let top = fy.yields.len() - 1;
    assert_eq!(fy.yields[top].energy, 1.4e7, "top point is 14 MeV");

    let thermal = valley_rate(&chain, "Pu239", 0);
    let fast = valley_rate(&chain, "Pu239", top);
    assert!(
        fast / thermal > 5.0,
        "Pu239 valley should rise substantially, got {:.1}x",
        fast / thermal
    );
}

#[test]
fn every_tabulated_energy_conserves_products_per_fission() {
    // Whatever the spectrum, a fission still makes about two products. This is
    // what makes a mistaken fold visible as a conservation break rather than a
    // quiet redistribution.
    let chain = chain();
    for parent in ["U235", "Pu239"] {
        let fy = chain[parent].fission_yields.as_ref().unwrap();
        for k in 0..fy.yields.len() {
            let total: f64 = fy.yields[k].products.iter().map(|(_, y)| y).sum();
            assert!(
                (total - 2.0).abs() < 0.05,
                "{parent} point {k} sums to {total}, not ~2 products per fission"
            );
        }
    }
}
