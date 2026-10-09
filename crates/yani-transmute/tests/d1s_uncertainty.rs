//! Half-life uncertainty on D1S time-correction factors.
//!
//! A TCF is the emitter's activity over the schedule per unit production. For
//! Mn56 with a 5% half-life sigma:
//!
//! - at the end of a long irradiation it has saturated, `A = R`, so the TCF
//!   hardly moves;
//! - after cooling for `t` it is `R exp(-lambda t)`, which moves by
//!   `lambda t x 5%`;
//! - two campaigns of one schedule draw one set of half-lives per replica, so
//!   identical rate histories give identical TCFs, draw for draw.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yani_transmute::d1s_uncertainty::time_correction_factor_ensemble;
use yani_transmute::uncertainty::{DataUncertainty, Source};

const HOUR: f64 = 3600.0;
const SIGMA: f64 = 0.05;

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    let mut chain = yani::parse_chain_arrow(&path).expect("parse chain");
    let mn56 = chain.get_mut("Mn56").expect("Mn56");
    mn56.half_life_uncertainty = Some(mn56.half_life.unwrap() * SIGMA);
    Arc::new(chain)
}

fn relative_spread(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean
}

#[test]
fn the_time_correction_moves_with_the_half_life_it_reads() {
    let chain = chain();
    let emitters = vec!["Mn56".to_string()];
    let timesteps = [48.0 * HOUR, 5.0 * HOUR];
    let rates = vec![vec![1.0e10, 0.0], vec![1.0e10, 0.0]];
    let request = DataUncertainty {
        seed: 3,
        samples: Some(512),
        sources: vec![Source::HalfLife],
        attribution: false,
        ..Default::default()
    };
    let ensemble =
        time_correction_factor_ensemble(&emitters, &timesteps, &rates, &chain, &request).unwrap();
    assert_eq!(ensemble.replicas.len(), 512);
    assert!(ensemble.half_lives_perturbed.contains("Mn56"));
    assert_eq!(ensemble.sources, vec!["half_life".to_string()]);

    let at = |step: usize| -> Vec<f64> {
        ensemble
            .replicas
            .iter()
            .map(|r| r[0]["Mn56"][step])
            .collect()
    };
    let saturated = relative_spread(&at(1));
    assert!(
        saturated < 0.02 * SIGMA,
        "saturated TCF barely moves: {saturated:.2e}"
    );
    let lambda_t = std::f64::consts::LN_2 * 5.0 * HOUR / chain["Mn56"].half_life.unwrap();
    let cooled = relative_spread(&at(2));
    assert!(
        (cooled / (lambda_t * SIGMA) - 1.0).abs() < 0.12,
        "cooled TCF moves by lambda t x sigma: {cooled:.4} against {:.4}",
        lambda_t * SIGMA
    );
    // One set of half-lives per replica, shared by both campaigns.
    for r in &ensemble.replicas {
        assert_eq!(r[0]["Mn56"], r[1]["Mn56"]);
    }
}

#[test]
fn without_the_half_life_source_nothing_is_drawn() {
    let request = DataUncertainty {
        seed: 3,
        samples: Some(16),
        sources: vec![Source::CrossSections],
        attribution: false,
        ..Default::default()
    };
    let ensemble = time_correction_factor_ensemble(
        &["Mn56".to_string()],
        &[HOUR],
        &[vec![1.0e10]],
        &chain(),
        &request,
    )
    .unwrap();
    assert!(ensemble.replicas.is_empty());
    assert!(ensemble.sources.is_empty());
}

/// A half-life sigma no draw can carry reaches the ensemble's report as its
/// own gap, not as a nuclide perturbed or one that states none.
#[test]
fn a_half_life_sigma_no_draw_can_carry_is_reported() {
    let mut chain = Arc::unwrap_or_clone(chain());
    chain.get_mut("Mn56").expect("Mn56").half_life_uncertainty = Some(f64::INFINITY);
    let chain = Arc::new(chain);
    let emitters = vec!["Mn56".to_string()];
    let request = DataUncertainty {
        seed: 3,
        samples: Some(8),
        sources: vec![Source::HalfLife],
        attribution: false,
        ..Default::default()
    };
    let ensemble = time_correction_factor_ensemble(
        &emitters,
        &[48.0 * HOUR],
        &[vec![1.0e10]],
        &chain,
        &request,
    )
    .unwrap();
    assert!(ensemble.half_life_uncertainty_not_carried.contains("Mn56"));
    assert!(!ensemble.half_lives_perturbed.contains("Mn56"));
    assert!(!ensemble.no_half_life_uncertainty.contains("Mn56"));
}

/// Left to itself the ensemble stops when every TCF's sigma is known to the
/// convergence target, and names the ones that were not when the cap stops it.
#[test]
fn an_adaptive_ensemble_stops_on_the_standard_error_of_each_sigma() {
    use yani_transmute::uncertainty::Output;

    let chain = chain();
    let emitters = vec!["Mn56".to_string()];
    let timesteps = [48.0 * HOUR, 5.0 * HOUR];
    let rates = vec![vec![1.0e10, 0.0]];
    let request = DataUncertainty {
        seed: 3,
        sources: vec![Source::HalfLife],
        ..Default::default()
    };
    let ensemble =
        time_correction_factor_ensemble(&emitters, &timesteps, &rates, &chain, &request).unwrap();
    let n = ensemble.replicas.len();
    // About 201 replicas for a Gaussian sigma at 5%, so a block past it.
    assert!((192..=384).contains(&n), "{n}");
    assert!(ensemble.converged && !ensemble.hit_cap, "{ensemble:?}");
    assert_eq!(ensemble.convergence, 0.05);

    let tight = DataUncertainty {
        convergence: 0.01,
        ..request.clone()
    };
    let ensemble =
        time_correction_factor_ensemble(&emitters, &timesteps, &rates, &chain, &tight).unwrap();
    assert_eq!(
        ensemble.replicas.len(),
        1024,
        "1% cannot be met inside the cap"
    );
    assert!(ensemble.hit_cap && !ensemble.converged);
    let miss = &ensemble.unconverged[0];
    assert_eq!(
        miss.output,
        Output::TimeCorrectionFactor {
            campaign: 0,
            emitter: "Mn56".to_string()
        }
    );
    assert!(miss.relative_standard_error.unwrap() > 0.01);

    let bad = DataUncertainty {
        convergence: 1.5,
        ..request
    };
    let err =
        time_correction_factor_ensemble(&emitters, &timesteps, &rates, &chain, &bad).unwrap_err();
    assert!(err.contains("convergence"), "{err}");
}
