//! Fission yield uncertainty on the inventory.
//!
//! Th232 irradiated for a day above its highest tabulated yield energy, so
//! every fission takes the 14 MeV yields, with a 5% DY placed on each of them. A
//! fission product's atom is never destroyed by the decay or capture a day
//! allows, so the fission products together count `F * sum(y_i m_i)` atoms:
//! independent draws give that total a relative sigma of
//! `0.05 sqrt(sum y^2) / sum y`, whatever the chain does with each product.
//!
//! The committed chain carries no evaluated yields, so the test attaches them,
//! with the tape's products the solver's own, which is the identity mapping.
//!
//! Self-skips when the Th232 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const DY: f64 = 0.05;

/// The fixture chain with Th232's yields given a DY of `DY` on every product.
fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let th232 = chain.get_mut("Th232").expect("Th232");
    let nominal = th232.fission_yields.as_ref().expect("Th232 yields");
    let yields = nominal
        .yields
        .iter()
        .map(|point| {
            let mut point = point.clone();
            point.independent = Some(yani::EvaluatedYields {
                products: point.products.iter().map(|(n, _)| n.clone()).collect(),
                yields: point.products.iter().map(|(_, y)| *y).collect(),
                uncertainties: point.products.iter().map(|(_, y)| Some(DY * y)).collect(),
                interpolation: None,
            });
            point
        })
        .collect();
    th232.fission_yields = Some(Arc::new(yani::FissionYieldSet::new(yields)));
    Arc::new(chain)
}

fn thorium() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Th232")?;
    let mut m = Material::new(
        HashMap::from([("Th232".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Th232 material");
    m.nuclides.insert("Th232".to_string(), 3.0e-2);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("Th232".to_string(), path)]), None)
        .expect("read Th232");
    Some(m)
}

fn is_actinide(name: &str) -> bool {
    let symbol: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    matches!(
        symbol.as_str(),
        "Ac" | "Th" | "Pa" | "U" | "Np" | "Pu" | "Am" | "Cm" | "Bk" | "Cf" | "Es" | "Fm"
    )
}

#[test]
fn the_fission_product_total_moves_by_the_yields_in_quadrature() {
    let Some(mut material) = thorium() else {
        eprintln!("skipping: Th232 fixture not available");
        return;
    };
    let chain = chain();
    // Above 14 MeV, where the yields are clamped to the last point.
    let spectrum = MultigroupSpectrum {
        boundaries: vec![1.4e7, 1.6e7],
        masses: vec![1.0],
        flux_error: None,
    };
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: Some((0, 1.0e14)),
    }];
    let request = DataUncertainty {
        seed: 3,
        samples: Some(2048),
        sources: vec![Source::FissionYield],
        attribution: false,
        ..Default::default()
    };
    let results = transmute_material(
        &mut material,
        &[spectrum],
        &steps,
        Arc::clone(&chain),
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute");

    let info = &results.uncertainty_info[&0];
    assert!(info.fission_yields_perturbed.contains("Th232"), "{info:?}");
    assert!(info.fission_yields_mapping_mismatch.is_empty(), "{info:?}");
    assert!(!info.not_perturbed.iter().any(|s| s == "fission yield"));
    assert_eq!(info.sources, vec!["fission_yield".to_string()]);

    let fast = &chain["Th232"].fission_yields.as_ref().unwrap().yields[1];
    assert_eq!(fast.energy, 1.4e7);
    let sum: f64 = fast.products.iter().map(|(_, y)| y).sum();
    let squares: f64 = fast.products.iter().map(|(_, y)| y * y).sum();
    let expected = DY * squares.sqrt() / sum;

    let totals: Vec<f64> = results
        .uncertainty_inventories(0, 1)
        .expect("an ensemble")
        .iter()
        .map(|inventory| {
            inventory
                .iter()
                .filter(|(name, _)| !is_actinide(name))
                .map(|(_, n)| n)
                .sum()
        })
        .collect();
    let mean = totals.iter().sum::<f64>() / totals.len() as f64;
    let sd =
        (totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (totals.len() - 1) as f64).sqrt();
    let got = sd / mean;
    assert!(
        (got / expected - 1.0).abs() < 0.08,
        "fission products total: {got:.5} against {expected:.5}"
    );

    // The actinide does not see its own yields.
    let th232 = results
        .get_nuclide_uncertainty(0, "Th232", 1)
        .unwrap_or(0.0);
    assert_eq!(th232, 0.0);
}
