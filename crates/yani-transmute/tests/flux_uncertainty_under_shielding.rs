//! A flux perturbation must move the rates the run actually used.
//!
//! The flux replicas are built from the per-group terms of the collapse, and
//! those terms used to be computed dilute whatever the run asked for, then
//! ASSIGNED to the rate. So a run that combined a self-shielding chord with a
//! per-bin flux sigma had every replica's rate quietly replaced by the dilute
//! one, even at zero perturbation, and the ensemble drifted off the nominal it
//! is supposed to spread around.
//!
//! Both halves are checked: that the terms are shielded when the run is, and
//! that the perturbation is a factor on the rate rather than a substitute for
//! it.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::multigroup::{
    compute_multigroup_reaction_rates_shielded, per_group_reaction_rates,
};
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{
    transmute_material_shielded, MultigroupSpectrum, Shielding, TransmutationResults, TransmuteStep,
};

/// Three groups: thermal, epithermal, fast.
const GROUPS: [f64; 4] = [1.0e-5, 0.625, 1.0e5, 2.0e7];
const FLUX: [f64; 3] = [1.0e12, 5.0e12, 1.0e14];
/// Enough chord through solid iron for its keV resonances to bite, which on
/// this structure is a few percent on capture.
const CHORD_CM: f64 = 2.0;
/// Small enough that the ensemble mean sits far closer to its own nominal than
/// the shielded and dilute answers sit to each other.
const RELATIVE_SIGMA: f64 = 0.01;
const REPLICAS: usize = 64;
/// The product of Fe56 capture, so the nuclide the shielding moves.
const PRODUCT: &str = "Fe57";

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
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

fn run(data: &str, shielding: Option<&Shielding>, with_sigma: bool) -> TransmutationResults {
    let total: f64 = FLUX.iter().sum();
    let spectra = vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        relative_std_dev: with_sigma.then(|| vec![RELATIVE_SIGMA; FLUX.len()]),
    }];
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: Some((0, total)),
    }];
    let request = DataUncertainty {
        seed: 1,
        samples: Some(REPLICAS),
        // Only the flux, so nothing here needs MF=33 covariance and the
        // ensemble's spread is the flux sigma alone.
        sources: vec![Source::FluxSpectrum],
    };
    transmute_material_shielded(
        &mut iron(data),
        &spectra,
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        with_sigma.then_some(&request),
        shielding,
    )
    .expect("transmute")
}

/// The ensemble mean of the product's density after the irradiation step.
fn ensemble_mean(results: &TransmutationResults, id: u32) -> f64 {
    let samples = results
        .uncertainty
        .get(&id)
        .expect("an ensemble was recorded")
        .samples_at(0, PRODUCT);
    assert_eq!(samples.len(), REPLICAS, "every replica must be recorded");
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn density(results: &TransmutationResults, id: u32) -> f64 {
    results
        .get_nuclide_density(id, PRODUCT, 1)
        .expect("the product exists after irradiation")
}

/// The terms themselves, against the collapse they are supposed to decompose.
///
/// This is the direct form of the first half, and the sharp one: the ensemble
/// can only see the terms through the shape of its perturbation, which on this
/// fixture is a 3% difference inside a 1% sigma. The sums are exact.
#[test]
fn the_per_group_terms_carry_the_weighting_the_run_asked_for() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let material = iron(&data);
    let chain = chain();
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let total: f64 = FLUX.iter().sum();
    let masses: Vec<f64> = FLUX.iter().map(|f| f / total).collect();

    let mut checked = 0;
    for request in [None, Some(&shielding)] {
        let (rates, _, _) = compute_multigroup_reaction_rates_shielded(
            &material, &chain, &masses, &GROUPS, 1.0, request,
        );
        let terms = per_group_reaction_rates(&material, &chain, &masses, &GROUPS, request);
        assert!(!rates.is_empty(), "the spectrum must produce rates");
        for (nuclide, per_kind) in &rates {
            for (kind, &rate) in per_kind {
                let summed: f64 = terms[nuclide][kind].iter().sum();
                assert!(
                    (summed - rate).abs() <= 1.0e-12 * rate.abs(),
                    "{nuclide} {kind}: the collapse gives {rate}, its terms sum to {summed}"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 6,
        "expected several reactions under each weighting, got {checked}"
    );

    // And the two weightings are genuinely different, so the check above is
    // not passing twice on the same numbers.
    let dilute = per_group_reaction_rates(&material, &chain, &masses, &GROUPS, None);
    let shielded = per_group_reaction_rates(&material, &chain, &masses, &GROUPS, Some(&shielding));
    let sum = |t: &yani_transmute::flux_uncertainty::PerGroupRates| -> f64 {
        t["Fe56"]["(n,gamma)"].iter().sum()
    };
    assert!(
        sum(&shielded) < sum(&dilute) * 0.99,
        "shielded terms must sum below dilute ones: {:e} against {:e}",
        sum(&shielded),
        sum(&dilute)
    );
}

#[test]
fn a_shielded_run_perturbs_its_shielded_rates() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = iron(&data).material_id.unwrap_or(0);

    let dilute = density(&run(&data, None, false), id);
    let shielded = density(&run(&data, Some(&shielding), false), id);

    // Without a gap between the two answers the rest of this test cannot fail,
    // so it is asserted rather than assumed.
    let gap = (dilute - shielded) / shielded;
    assert!(
        gap > 0.02,
        "shielding must move {PRODUCT} for this test to discriminate: \
         dilute {dilute:e}, shielded {shielded:e}, gap {:.3}%",
        gap * 100.0
    );

    let mean = ensemble_mean(&run(&data, Some(&shielding), true), id);

    // The perturbation is symmetric and the rate is linear in the flux, so the
    // ensemble spreads around its own nominal. A tenth of the gap is far wider
    // than the sampling error at this sigma and far narrower than the gap.
    let off_shielded = (mean - shielded).abs() / shielded;
    assert!(
        off_shielded < gap / 10.0,
        "the ensemble must spread around the shielded nominal it came from: \
         mean {mean:e} is {:.3}% off shielded {shielded:e}, with the dilute \
         answer {dilute:e} a {:.3}% walk away",
        off_shielded * 100.0,
        gap * 100.0
    );
}

/// The other half, and the one that fails loudest: with every deviate at zero
/// the replicas must reproduce the nominal inventory exactly.
#[test]
fn an_unperturbed_shielded_replica_is_the_nominal_run() {
    let Some(data) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skipping: Fe56 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = iron(&data).material_id.unwrap_or(0);

    let total: f64 = FLUX.iter().sum();
    let spectra = vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        // A flux the caller states is exact still requests the source, so the
        // per-group terms are built and used, with every deviate zero.
        relative_std_dev: Some(vec![0.0; FLUX.len()]),
    }];
    let steps = vec![TransmuteStep {
        dt: 86400.0,
        irradiation: Some((0, total)),
    }];
    let request = DataUncertainty {
        seed: 1,
        samples: Some(2),
        sources: vec![Source::FluxSpectrum],
    };
    let results = transmute_material_shielded(
        &mut iron(&data),
        &spectra,
        &steps,
        chain(),
        &Default::default(),
        Default::default(),
        Some(&request),
        Some(&shielding),
    )
    .expect("transmute");

    let nominal = density(&results, id);
    for (replica, sample) in results
        .uncertainty
        .get(&id)
        .expect("an ensemble was recorded")
        .samples_at(0, PRODUCT)
        .iter()
        .enumerate()
    {
        assert_eq!(
            *sample, nominal,
            "replica {replica} perturbed nothing and must be the nominal run"
        );
    }
}
