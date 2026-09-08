//! An energy-resolved rate must sum to the one-group rate it decomposes
//! (yani#27).
//!
//! `get_reaction_rate_spectrum` exists to say which part of a spectrum drove a
//! channel, which a collapsed rate cannot: 57 mb of `W186(n,gamma)` against a
//! spectrum that is 89% fast and 0.7% thermal is either a fast-capture rate or
//! a resonance-region one, and the two carry different consequences for a
//! disagreement. The number is only worth reading if it is a decomposition of
//! the rate the solve actually used, so that is what these check, on both the
//! dilute and the shielded weighting.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;

use yamc_materials::Material;
use yani_transmute::multigroup::compute_multigroup_reaction_rates_shielded;
use yani_transmute::{reaction_rate_spectrum, Shielding};

/// Total flux magnitude, so the rates are not all at unit flux.
const SOURCE_RATE: f64 = 3.0e13;

fn chain() -> HashMap<String, yani::ChainNuclide> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    yani::parse_chain_arrow(&path).expect("parse chain")
}

fn iron(data: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.density = Some(7.87);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), data.to_string())]),
        None,
    )
    .expect("read Fe56");
    m
}

fn ccfe_709() -> Vec<f64> {
    yamc_nuclide::group_structures::get_group_structure("CCFE-709")
        .expect("CCFE-709")
        .to_vec()
}

/// A flux per unit lethargy, so every decade carries some and the resonance
/// region is not empty. The top group takes a 14 MeV spike on top of it, which
/// is the shape an activation spectrum has.
fn broad_flux(boundaries: &[f64]) -> Vec<f64> {
    let n = boundaries.len() - 1;
    let mut flux: Vec<f64> = (0..n)
        .map(|g| (boundaries[g + 1] / boundaries[g]).ln())
        .collect();
    let spike = boundaries
        .iter()
        .position(|&e| e > 1.4e7)
        .expect("CCFE-709 reaches past 14 MeV")
        - 1;
    flux[spike] += 20.0;
    flux
}

/// Every channel the collapse produced, with its rate and its per-group terms.
fn checked_against_collapse(
    material: &Material,
    chain: &HashMap<String, yani::ChainNuclide>,
    flux: &[f64],
    boundaries: &[f64],
    shielding: Option<&Shielding>,
) -> usize {
    let (rates, _, _) = compute_multigroup_reaction_rates_shielded(
        material,
        chain,
        flux,
        boundaries,
        SOURCE_RATE,
        shielding,
    );
    assert!(!rates.is_empty(), "the spectrum must produce rates");

    let mut checked = 0;
    for (nuclide, per_kind) in &rates {
        for (kind, &rate) in per_kind {
            let terms = reaction_rate_spectrum(
                material,
                flux,
                boundaries,
                SOURCE_RATE,
                shielding,
                nuclide,
                kind,
            )
            .unwrap_or_else(|| panic!("{nuclide} {kind}: no spectrum for a channel with a rate"));
            assert_eq!(terms.len(), flux.len(), "one term per group");
            let summed: f64 = terms.iter().sum();
            assert!(
                (summed - rate).abs() <= 1.0e-12 * rate.abs(),
                "{nuclide} {kind}: collapse gives {rate}, the per-group terms sum to {summed}"
            );
            checked += 1;
        }
    }
    checked
}

#[test]
fn the_terms_sum_to_the_dilute_rate() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let boundaries = ccfe_709();
    let flux = broad_flux(&boundaries);
    let checked = checked_against_collapse(&iron(&data), &chain(), &flux, &boundaries, None);
    assert!(
        checked > 3,
        "expected several reactions to check, got {checked}"
    );
}

#[test]
fn the_terms_sum_to_the_shielded_rate() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let material = iron(&data);
    let chain = chain();
    let boundaries = ccfe_709();
    let flux = broad_flux(&boundaries);
    // Two centimetres of mean chord through solid iron, which is enough for its
    // keV resonances to depress the flux inside the lump.
    let shielding = Shielding::new(2.0).expect("a positive chord");

    let checked = checked_against_collapse(&material, &chain, &flux, &boundaries, Some(&shielding));
    assert!(
        checked > 3,
        "expected several reactions to check, got {checked}"
    );

    // The identity above would also hold if the shielded weighting were being
    // ignored on both sides, so check that it is not: capture is the channel
    // the resonances dominate, and shielding must move it.
    let dilute = reaction_rate_spectrum(
        &material,
        &flux,
        &boundaries,
        SOURCE_RATE,
        None,
        "Fe56",
        "(n,gamma)",
    )
    .expect("dilute capture");
    let shielded = reaction_rate_spectrum(
        &material,
        &flux,
        &boundaries,
        SOURCE_RATE,
        Some(&shielding),
        "Fe56",
        "(n,gamma)",
    )
    .expect("shielded capture");
    let (dilute_total, shielded_total): (f64, f64) = (dilute.iter().sum(), shielded.iter().sum());
    assert!(
        shielded_total < dilute_total,
        "shielding must depress capture: dilute {dilute_total}, shielded {shielded_total}"
    );
}

#[test]
fn a_group_with_no_flux_carries_no_rate() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let boundaries = ccfe_709();
    let n = boundaries.len() - 1;
    // Flux in three groups and nothing anywhere else, which is the shape a
    // few-line source has. The highest of the three is the last group that
    // lies wholly under 20 MeV, where the ENDF/B-VIII.1 evaluation ends:
    // CCFE-709 runs on to 1 GeV, and a group past the evaluation's last point
    // carries flux but no cross section, so it carries no rate by design.
    let top = boundaries.iter().rposition(|&e| e <= 2.0e7).unwrap() - 1;
    let carrying = [12usize, 400, top];
    let mut flux = vec![0.0; n];
    for (i, &g) in carrying.iter().enumerate() {
        flux[g] = (i + 1) as f64;
    }
    flux[n - 2] = 1.0;

    let terms = reaction_rate_spectrum(
        &iron(&data),
        &flux,
        &boundaries,
        SOURCE_RATE,
        None,
        "Fe56",
        "(n,gamma)",
    )
    .expect("capture on a three-line spectrum");

    for (g, &term) in terms.iter().enumerate() {
        if carrying.contains(&g) {
            assert!(term > 0.0, "group {g} carries flux and must carry rate");
        } else if g == n - 2 {
            assert_eq!(
                term, 0.0,
                "group {g} lies above the evaluation and must carry no rate"
            );
        } else {
            assert_eq!(
                term, 0.0,
                "group {g} carries no flux and must carry no rate"
            );
        }
    }
}

#[test]
fn an_unknown_nuclide_or_channel_is_none() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let material = iron(&data);
    let boundaries = ccfe_709();
    let flux = broad_flux(&boundaries);
    let ask = |nuclide: &str, kind: &str| {
        reaction_rate_spectrum(
            &material,
            &flux,
            &boundaries,
            SOURCE_RATE,
            None,
            nuclide,
            kind,
        )
    };

    assert!(
        ask("Fe56", "(n,gamma)").is_some(),
        "the channel it does have"
    );
    assert!(
        ask("U235", "(n,gamma)").is_none(),
        "a nuclide it never loaded"
    );
    assert!(
        ask("Fe56", "(n,banana)").is_none(),
        "a kind that names no MT"
    );
    // `(n,n')` has no transport total, so its rate comes from the branching
    // overlay's MF=10 partials rather than from a group average of this shape.
    assert!(
        ask("Fe56", "(n,n')").is_none(),
        "a kind with no MT to collapse"
    );

    // No flux is not a small flux: there is no spectrum to resolve onto, and a
    // vector of zeros would read as a rate resolved rather than as no rate.
    assert!(
        reaction_rate_spectrum(
            &material,
            &vec![0.0; flux.len()],
            &boundaries,
            SOURCE_RATE,
            None,
            "Fe56",
            "(n,gamma)",
        )
        .is_none(),
        "an empty spectrum drives nothing"
    );
    assert!(
        reaction_rate_spectrum(
            &material,
            &flux,
            &boundaries,
            0.0,
            None,
            "Fe56",
            "(n,gamma)",
        )
        .is_none(),
        "a pulse at zero rate drives nothing"
    );
}
