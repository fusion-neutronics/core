//! Half-life uncertainty on the inventory and on what is derived from it.
//!
//! One case pins every rule the source has to keep. Fe56 irradiated long enough
//! to saturate Mn56 (about nineteen of its half-lives), with a 5% sigma placed
//! on Mn56's half-life:
//!
//! - at saturation `N = R / lambda`, so the Mn56 density moves with its
//!   half-life, one for one: about 5%;
//! - but `A = lambda N = R`, so its activity hardly moves at all. That holds
//!   only if each replica's activity is evaluated with that replica's own
//!   half-life. With the nominal one it would move by the full 5%, which is
//!   the mistake the per-replica chain exists to prevent;
//! - after a cooldown of `t` the activity is `R exp(-lambda t)`, whose
//!   relative sensitivity to the half-life is `lambda t`, so five hours of
//!   cooling (1.34 in `lambda t`) should read about 1.34 x 5%.
//!
//! The committed chain carries no half-life sigmas, so the test sets one and
//! leaves every other nuclide without, which the report must list.
//!
//! Self-skips when the Fe56 fixture is missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{transmute_material, MultigroupSpectrum, TransmuteStep};

const HOUR: f64 = 3600.0;
const RELATIVE_SIGMA: f64 = 0.05;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let mn56 = chain.get_mut("Mn56").expect("Mn56 in the chain");
    mn56.half_life_uncertainty = Some(mn56.half_life.unwrap() * RELATIVE_SIGMA);
    Arc::new(chain)
}

fn iron() -> Option<Material> {
    let path = yamc_test_cache::nuclide("Fe56")?;
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "sum",
        None,
    )
    .expect("Fe56 material");
    m.nuclides.insert("Fe56".to_string(), 8.5e-2);
    m.volume = Some(1.0);
    m.set_temperature("294");
    m.read_nuclear_data(&HashMap::from([("Fe56".to_string(), path)]), None)
        .expect("read Fe56");
    Some(m)
}

/// Two days of 14 MeV-ish flux, then five hours of cooling.
fn run(sources: Vec<Source>) -> Option<(yani_transmute::TransmutationResults, f64)> {
    let mut material = iron()?;
    let spectrum = MultigroupSpectrum {
        boundaries: vec![1.0e6, 2.0e7],
        masses: vec![1.0],
        flux_error: None,
    };
    let steps = vec![
        TransmuteStep {
            dt: 48.0 * HOUR,
            irradiation: Some((0, 1.0e14)),
        },
        TransmuteStep {
            dt: 5.0 * HOUR,
            irradiation: None,
        },
    ];
    let chain = chain();
    let lambda_t = std::f64::consts::LN_2 * 5.0 * HOUR / chain["Mn56"].half_life.unwrap();
    let request = DataUncertainty {
        seed: 11,
        samples: Some(512),
        sources,
        attribution: false,
    };
    let results = transmute_material(
        &mut material,
        &[spectrum],
        &steps,
        chain,
        &Default::default(),
        Default::default(),
        Some(&request),
    )
    .expect("transmute");
    Some((results, lambda_t))
}

fn info_of(results: &yani_transmute::TransmutationResults) -> &yani_transmute::uncertainty::Info {
    results
        .uncertainty_info
        .get(&0)
        .expect("uncertainty was asked for")
}

fn relative(estimate: &yani_transmute::Estimate) -> f64 {
    estimate.std_dev.expect("a spread") / estimate.nominal
}

#[test]
fn a_saturated_activity_is_insensitive_to_its_own_half_life() {
    let Some((results, lambda_t)) = run(vec![Source::HalfLife]) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let chain = results.chain.clone().unwrap();
    let info = info_of(&results);
    assert!(info.half_lives_perturbed.contains("Mn56"), "{info:?}");
    assert_eq!(
        info.half_lives_perturbed.len(),
        1,
        "only Mn56 carries a sigma"
    );
    assert!(
        info.no_half_life_uncertainty.contains("Fe55"),
        "an unstable nuclide without a sigma is reported, not assumed exact"
    );
    assert!(info.has_gaps());
    assert_eq!(info.half_lives_sampled, 512);

    // Step 1 is the end of irradiation, step 2 the end of the cooldown.
    let density = |step: usize| {
        let nominal = results.get_material(0, step).unwrap().nuclides["Mn56"];
        let sigma = results.uncertainty[&0].std_dev_at(step - 1)["Mn56"];
        sigma / nominal
    };
    let activity = |step: usize| {
        let by = results
            .activity_uncertainty_by_nuclide(0, step, &chain)
            .expect("volume is set")
            .expect("uncertainty was asked for");
        relative(&by["Mn56"])
    };

    // Sampling error on a sigma from 512 replicas is about 3%.
    let saturated = density(1);
    assert!(
        (saturated / RELATIVE_SIGMA - 1.0).abs() < 0.12,
        "saturated Mn56 density moves with its half-life: {saturated:.4}"
    );
    let saturated_activity = activity(1);
    assert!(
        saturated_activity < 0.02 * RELATIVE_SIGMA,
        "saturated Mn56 activity is lambda N = R, so it barely moves: {saturated_activity:.2e}"
    );
    let cooled = activity(2);
    assert!(
        (cooled / (lambda_t * RELATIVE_SIGMA) - 1.0).abs() < 0.12,
        "after cooling the activity moves by lambda t times the half-life sigma: \
         {cooled:.4} against {:.4}",
        lambda_t * RELATIVE_SIGMA
    );

    // Mn56's photon lines are its activity times a per-decay probability, so
    // at saturation they are as insensitive to its half-life as the activity
    // is. The chain stores the lines per atom per second, which scales with
    // the decay constant; a replica's half-life has to carry through to them.
    assert!(
        !chain["Mn56"].sources.is_empty(),
        "the fixture carries Mn56's lines"
    );
    let lines = results
        .photon_spectrum_uncertainty(0, 1, &chain)
        .expect("volume is set")
        .expect("uncertainty was asked for");
    let strongest = lines
        .iter()
        .max_by(|a, b| a.estimate.nominal.total_cmp(&b.estimate.nominal))
        .expect("Mn56 emits");
    let line = relative(&strongest.estimate);
    assert!(
        line < 0.02 * RELATIVE_SIGMA,
        "a saturated emitter's photon rate barely moves: {line:.2e}"
    );

    // The initial iron is stable, so step 0 has no spread to show.
    let initial = results.activity_uncertainty(0, 0, &chain).unwrap().unwrap();
    assert_eq!(initial.std_dev, Some(0.0));
}

/// Off unless asked for: without the half-life source nothing is sampled and
/// the report has nothing to say about half-lives.
#[test]
fn without_the_source_no_half_life_is_touched() {
    let Some((results, _)) = run(vec![Source::CrossSections]) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    let info = info_of(&results);
    assert!(info.half_lives_perturbed.is_empty());
    assert!(info.no_half_life_uncertainty.is_empty());
    assert_eq!(info.half_lives_sampled, 0);
    assert!(results.uncertainty[&0]
        .half_lives()
        .iter()
        .all(|h| h.is_empty()));
}

/// The same seed gives the same half-lives, and so the same inventories.
#[test]
fn the_draw_is_reproducible() {
    let (Some((a, _)), Some((b, _))) = (run(vec![Source::HalfLife]), run(vec![Source::HalfLife]))
    else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    assert_eq!(
        a.uncertainty[&0].half_lives(),
        b.uncertainty[&0].half_lives()
    );
    let sa = a.uncertainty[&0].std_dev_at(1)["Mn56"];
    let sb = b.uncertainty[&0].std_dev_at(1)["Mn56"];
    assert_eq!(sa.to_bits(), sb.to_bits());
}

/// Leaving the source out is recorded as a source not perturbed, and asking
/// for it removes it from that list.
#[test]
fn the_report_says_whether_half_lives_were_perturbed() {
    let (Some((off, _)), Some((on, _))) = (
        run(vec![Source::CrossSections]),
        run(vec![Source::HalfLife]),
    ) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    assert!(info_of(&off).not_perturbed.iter().any(|s| s == "half-life"));
    assert!(!info_of(&on).not_perturbed.iter().any(|s| s == "half-life"));
    assert!(info_of(&on).sources.contains(&"half_life".to_string()));
}
