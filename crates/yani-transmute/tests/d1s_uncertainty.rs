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

use yani_transmute::d1s_uncertainty::{time_correction_factor_ensemble, D1S_LINES_NOT_PERTURBED};
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

/// Mn56's photon spectra, with relative normalisation sigmas `gamma` and
/// `xray` (`None` writes the ENDF/B way, FD = 1 with dFD = 0, and a 2% dRI on
/// every line, which no per-emitter tally can follow).
fn mn56_photons(gamma: Option<f64>, xray: Option<f64>) -> Arc<HashMap<String, yani::ChainNuclide>> {
    let mut chain = Arc::unwrap_or_clone(chain());
    let mn56 = chain.get_mut("Mn56").expect("Mn56");
    let mut kinds = ["gamma", "xray"].into_iter();
    let mut spectra = 0;
    for source in mn56.sources.iter_mut().filter(|s| s.particle == "photon") {
        let yani::DecaySourceDistribution::Discrete { intensities, .. } = &source.distribution
        else {
            continue;
        };
        // The fixture names no radiation; the strongest spectrum is the gamma
        // one, the rest x-rays.
        let kind = kinds.next().unwrap_or("xray");
        source.radiation = Some(kind.to_string());
        let sigma = if kind == "gamma" { gamma } else { xray };
        source.uncertainty = Some(Arc::new(yani::DecaySourceUncertainty {
            normalization: Some(if sigma.is_some() { 0.9 } else { 1.0 }),
            normalization_uncertainty: Some(sigma.map_or(0.0, |s| 0.9 * s)),
            intensity_uncertainties: Some(intensities.iter().map(|i| 0.02 * i).collect()),
            energy_uncertainties: None,
            covariance: None,
        }));
        spectra += 1;
    }
    assert!(spectra >= 1, "Mn56 has photon lines");
    Arc::new(chain)
}

fn photon_request(samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: 3,
        samples: Some(samples),
        sources: vec![Source::DecayPhotonLines],
        attribution: false,
    }
}

/// The normalisation scales an emitter's whole spectrum, so its tally: a
/// D1S dose of one emitter moves by FD's sigma, applied after the fact,
/// weighted by the spectrum's share of the emitter's photon energy. The line
/// intensities and energies are reported as held.
#[test]
fn a_d1s_dose_moves_by_the_photon_normalisation() {
    let sigma = 0.03;
    let chain = mn56_photons(Some(sigma), None);
    let emitters = vec!["Mn56".to_string()];
    let ensemble = time_correction_factor_ensemble(
        &emitters,
        &[48.0 * HOUR, 5.0 * HOUR],
        &[vec![1.0e10, 0.0]],
        &chain,
        &photon_request(4096),
    )
    .unwrap();
    assert_eq!(ensemble.sources, vec!["decay_photon_lines".to_string()]);
    assert!(ensemble
        .decay_photon_normalisations_perturbed
        .contains("Mn56"));
    assert!(ensemble
        .not_perturbed
        .iter()
        .any(|s| s == D1S_LINES_NOT_PERTURBED));
    assert!(ensemble.not_perturbed.iter().any(|s| s == "half-life"));
    assert_eq!(ensemble.replicas.len(), 4096);

    // The gamma spectrum carries nearly all of Mn56's photon energy.
    let mn56 = &chain["Mn56"];
    let energy = |s: &yani::DecaySource| match &s.distribution {
        yani::DecaySourceDistribution::Discrete {
            energies,
            intensities,
        } => energies.iter().zip(intensities).map(|(e, i)| e * i).sum(),
        _ => 0.0,
    };
    let photons = mn56.sources.iter().filter(|s| s.particle == "photon");
    let total: f64 = photons.clone().map(energy).sum();
    let gamma: f64 = photons
        .filter(|s| s.radiation.as_deref() == Some("gamma"))
        .map(energy)
        .sum();
    for step in [1, 2] {
        let values: Vec<f64> = ensemble
            .replicas
            .iter()
            .map(|r| r[0]["Mn56"][step])
            .collect();
        let spread = relative_spread(&values);
        let want = sigma * gamma / total;
        assert!(
            (spread / want - 1.0).abs() < 0.06,
            "step {step}: {spread:.4} against {want:.4}"
        );
    }
}

/// Gamma and x-ray normalisations of one emitter: drawn independently the
/// lower end, one deviate between them the upper, which is wider. The
/// fixture merges Mn56's photons into one spectrum, so an x-ray spectrum
/// carrying a fifth of the gamma one's photon energy is added beside it.
#[test]
fn a_d1s_dose_ranges_over_the_correlation_between_spectra() {
    let mut chain = Arc::unwrap_or_clone(mn56_photons(Some(0.03), None));
    let mn56 = chain.get_mut("Mn56").expect("Mn56");
    let gamma_energy: f64 = mn56
        .sources
        .iter()
        .filter(|s| s.particle == "photon")
        .map(|s| match &s.distribution {
            yani::DecaySourceDistribution::Discrete {
                energies,
                intensities,
            } => energies.iter().zip(intensities).map(|(e, i)| e * i).sum(),
            _ => 0.0,
        })
        .sum();
    let intensity = 0.2 * gamma_energy / 6.0e3;
    mn56.sources.push(yani::DecaySource {
        particle: "photon".to_string(),
        radiation: Some("xray".to_string()),
        distribution: yani::DecaySourceDistribution::Discrete {
            energies: vec![6.0e3],
            intensities: vec![intensity],
        },
        uncertainty: Some(Arc::new(yani::DecaySourceUncertainty {
            normalization: Some(0.5),
            normalization_uncertainty: Some(0.5 * 0.1),
            ..Default::default()
        })),
    });
    let chain = Arc::new(chain);
    let ensemble = time_correction_factor_ensemble(
        &["Mn56".to_string()],
        &[48.0 * HOUR],
        &[vec![1.0e10]],
        &chain,
        &photon_request(4096),
    )
    .unwrap();
    let at = |replicas: &[yani_transmute::d1s_uncertainty::ReplicaTcfs]| -> f64 {
        relative_spread(&replicas.iter().map(|r| r[0]["Mn56"][1]).collect::<Vec<_>>())
    };
    let (low, high) = (at(&ensemble.replicas), at(&ensemble.replicas_correlated));
    // Shares of the photon energy: 5/6 gamma at 3%, 1/6 x-ray at 10%.
    let (g, x): (f64, f64) = (5.0 / 6.0 * 0.03, 1.0 / 6.0 * 0.1);
    let quadrature = (g * g + x * x).sqrt();
    assert!(
        (low / quadrature - 1.0).abs() < 0.06,
        "{low} against {quadrature}"
    );
    assert!(
        (high / (g + x) - 1.0).abs() < 0.06,
        "{high} against {}",
        g + x
    );
}

/// A spectrum written the ENDF/B way has no normalisation sigma to draw, so
/// nothing moves the tally, and the report names it as folded.
#[test]
fn a_folded_spectrum_is_named_and_draws_nothing() {
    let chain = mn56_photons(None, None);
    let ensemble = time_correction_factor_ensemble(
        &["Mn56".to_string()],
        &[48.0 * HOUR],
        &[vec![1.0e10]],
        &chain,
        &photon_request(16),
    )
    .unwrap();
    assert!(ensemble.decay_photon_normalisations_perturbed.is_empty());
    assert!(ensemble.replicas.is_empty());
    assert!(ensemble.decay_photon_spectra_folded["Mn56"].contains(&"gamma".to_string()));
    assert!(ensemble
        .not_perturbed
        .iter()
        .any(|s| s == D1S_LINES_NOT_PERTURBED));
}
