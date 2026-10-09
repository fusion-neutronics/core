//! Gas production in appm on real data, against the hand calculation.
//!
//! Natural iron, one 365-day year at 1e14 n/cm^2/s spread evenly over 13 to
//! 15 MeV, TENDL-2025 cross sections and reactions with the ENDF/B-VIII.1
//! decay data and fission yields, the defaults a Python run takes. Issue #300
//! gives the gas this makes, computed by hand from the inventory, as H1 600.7,
//! H2 13.3 and He4 142.4 appm. The method has to agree with that arithmetic
//! exactly, and the inventory has to still say roughly those numbers.
//!
//! The data is read from the download cache and the test skips when it is
//! absent, as the other real-data tests here do.

use std::collections::HashMap;

use yamc_materials::Material;
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep, GAS_NUCLIDES};

const IRON: [&str; 4] = ["Fe54", "Fe56", "Fe57", "Fe58"];

fn cached(name: &str) -> Option<String> {
    let dir = yamc_test_cache::root().join(name);
    dir.is_dir().then(|| dir.to_string_lossy().into_owned())
}

#[test]
fn iron_gas_matches_the_hand_calculation() {
    let decay = cached("endf-b8.1-transmutation-decay.arrow");
    let reactions = cached("tendl-2025-transmutation-reactions.arrow");
    let yields = cached("endf-b8.1-transmutation-fission_yields.arrow");
    let paths: Option<HashMap<String, String>> = IRON
        .iter()
        .map(|n| Some((n.to_string(), cached(&format!("tendl-2025-{n}.arrow"))?)))
        .collect();
    let (Some(decay), Some(reactions), Some(yields), Some(paths)) =
        (decay, reactions, yields, paths)
    else {
        eprintln!("skipping gas_production (TENDL-2025 iron or the chain is not cached)");
        return;
    };
    let loaded =
        yani::load_chain_parts(&decay, Some(&reactions), Some(&yields), None).expect("chain");

    let composition: HashMap<String, f64> = IRON
        .iter()
        .map(|n| (n.to_string(), yamc_nuclide::data::NATURAL_ABUNDANCE[n]))
        .collect();
    let mut iron = Material::new(composition, "atom", "g/cm3", Some(7.874)).expect("iron");
    iron.volume = Some(1.0);
    // Through the configuration, as a Python run reads it, so the activation
    // load takes the reaction subset the cache holds. Every cached TENDL-2025
    // nuclide is mapped, not only the iron: the products' own (n,p) and
    // (n,a) add about 0.65 appm of H1 over the year.
    let mut library = paths;
    for entry in std::fs::read_dir(yamc_test_cache::root()).expect("cache") {
        let name = entry.expect("cache entry").file_name();
        let name = name.to_string_lossy();
        if let Some(nuclide) = name
            .strip_prefix("tendl-2025-")
            .and_then(|n| n.strip_suffix(".arrow"))
            .filter(|n| !n.starts_with("transmutation"))
        {
            library
                .entry(nuclide.to_string())
                .or_insert_with(|| cached(&name).unwrap());
        }
    }
    yamc_nuclide::config::CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .set_cross_sections(library);
    let id = iron.material_id.unwrap_or(0);

    let spectra = [MultigroupSpectrum {
        boundaries: vec![13.0e6, 15.0e6],
        masses: vec![1.0],
        flux_error: None,
    }];
    let steps = [
        TransmuteStep {
            dt: 365.0 * 86400.0,
            irradiation: Some((0, 1.0e14)),
        },
        TransmuteStep {
            dt: 86400.0,
            irradiation: None,
        },
    ];
    let results = transmute_material(
        &mut iron,
        &spectra,
        &steps,
        loaded.chain,
        &loaded.branch,
        loaded.parts,
        None,
    )
    .expect("transmute");

    let gas = results
        .gas_production(id, true)
        .expect("the chain holds the gas")
        .expect("the material is in the results");

    // The issue's few lines, on the same inventory.
    let a0 = results
        .get_material(id, 0)
        .unwrap()
        .get_atoms_per_barn_cm()
        .unwrap();
    let total: f64 = a0.values().sum();
    for (step, material) in results.materials[&id].iter().enumerate() {
        let a = material.get_atoms_per_barn_cm().unwrap();
        for n in GAS_NUCLIDES {
            let by_hand = (a.get(n).copied().unwrap_or(0.0) - a0.get(n).copied().unwrap_or(0.0))
                / total
                * 1e6;
            let got = gas[n][step];
            assert!(
                (got - by_hand).abs() <= 1e-9 * by_hand.abs().max(1.0),
                "{n} at step {step}: {got} against {by_hand}"
            );
        }
    }
    assert_eq!(gas["H1"].len(), steps.len() + 1);

    // And the numbers the issue quotes. Within 1 appm rather than to the
    // figure quoted: the products' share depends on which of their data the
    // cache holds, and with the iron alone H1 reads 600.06.
    for (n, expected) in [("H1", 600.7), ("H2", 13.3), ("He4", 142.4)] {
        let got = gas[n][1];
        assert!(
            (got - expected).abs() < 1.0,
            "{n}: {got} appm, the issue has {expected}"
        );
    }
    // Stable gas does not move over the cooling day.
    assert_eq!(gas["He4"][1], gas["He4"][2]);
}
