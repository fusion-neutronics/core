//! Nuclear-data uncertainty on displacement damage, against the Fe56 fixture
//! with its MF=33 covariance written in from the committed evaluation.
//!
//! Its own binary, because it points the process-wide configuration at that
//! copy of Fe56. Self-skips when the Fe56 fixture is missing.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::damage::MT_DAMAGE_ENERGY;
use yani_transmute::flux_uncertainty::FluxError;
use yani_transmute::uncertainty::{DataUncertainty, Source, DEFAULT_CONVERGENCE};
use yani_transmute::{
    transmute_materials, DamageRequest, MultigroupSpectrum, TransmutationResults, TransmuteCase,
    TransmuteStep,
};

const BOUNDARIES: [f64; 3] = [1.0e5, 1.0e6, 1.4e7];
const MASSES: [f64; 2] = [0.3, 0.7];

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn iron() -> Material {
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
    m
}

/// Five minutes at 1e10 n/cm^2/s, then an hour of cooling, with `sources`
/// perturbed over 400 replicas.
fn run(sources: Vec<Source>, flux_error: Option<Vec<f64>>) -> TransmutationResults {
    let mut m = iron();
    transmute_materials(
        vec![TransmuteCase {
            material: &mut m,
            spectra: vec![MultigroupSpectrum {
                boundaries: BOUNDARIES.to_vec(),
                masses: MASSES.to_vec(),
                flux_error: flux_error.map(FluxError::RelativeStdDev),
            }],
            steps: vec![
                TransmuteStep {
                    dt: 300.0,
                    irradiation: Some((0, 1.0e10)),
                },
                TransmuteStep {
                    dt: 3600.0,
                    irradiation: None,
                },
            ],
            shielding: None,
            product_threshold: yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 7,
            samples: Some(400),
            sources,
            attribution: false,
            convergence: DEFAULT_CONVERGENCE,
        }),
        Some(&DamageRequest::default()),
    )
    .expect("transmute")
}

/// The group average of a lin-lin tabulation: the trapezoid over the group's
/// end points and every tabulated point between.
fn group_average(reaction: &yamc_nuclide::reaction::Reaction, lo: f64, hi: f64) -> f64 {
    let mut points = vec![(lo, reaction.cross_section_at(lo).unwrap())];
    for (&e, &xs) in reaction.energy.iter().zip(reaction.cross_section.iter()) {
        if e > lo && e < hi {
            points.push((e, xs));
        }
    }
    points.push((hi, reaction.cross_section_at(hi).unwrap()));
    let integral: f64 = points
        .windows(2)
        .map(|w| 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0))
        .sum();
    integral / (hi - lo)
}

#[test]
fn damage_energy_moves_with_the_cross_sections_and_the_flux() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = common::fe56_with_covariance(tmp.path()) else {
        eprintln!("skipping -- Fe56 fixture absent");
        return;
    };
    yamc_nuclide::config::CONFIG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .set_cross_section("Fe56", Some(dir.to_str().unwrap()));

    // The cross sections move the damage energy through each reaction's
    // deposit: the replicas' spread is the sigma the sampler reads off the
    // widened field, the deposits reproduce nearly all of the stored row, and
    // dpa moves with damage energy by the NRT factor alone. The trimmed
    // evaluation states MF=33 for (n,p) alone, so the sigma is that
    // reaction's share of the damage energy, small but not zero: (n,p) is not
    // a reaction the damage energy was ever read from before.
    let r = run(vec![Source::CrossSections], None);
    let fe56 = r.uncertainty_info[&0].damage_energy["Fe56"];
    assert!(fe56.explained > 0.95 && fe56.explained < 1.05, "{fe56:?}");
    assert!(fe56.sigma > 0.0, "{fe56:?}");
    let energy = r.damage_energy_uncertainty(0, 1, None).unwrap().unwrap();
    assert_eq!(
        energy.nominal,
        r.get_displacement_damage(0).unwrap().damage_energy[1]
    );
    let (mean, sd) = (energy.mean.unwrap(), energy.std_dev.unwrap());
    // 400 replicas: the spread is known to about 3.5%.
    let relative = sd / energy.nominal;
    assert!(
        (relative / fe56.sigma - 1.0).abs() < 0.15,
        "replica spread {relative} vs sampled sigma {}",
        fe56.sigma
    );
    assert!(
        (mean - energy.nominal).abs() < 4.0 * sd / 20.0,
        "{mean} vs {}",
        energy.nominal
    );
    let dpa = r.dpa_uncertainty(0, 1, Some("Fe")).unwrap().unwrap();
    assert!((dpa.std_dev.unwrap() / dpa.nominal - relative).abs() < 1e-6 * relative);
    // Cooling adds nothing, so the spread carries over unchanged.
    assert_eq!(
        r.dpa_uncertainty(0, 2, None).unwrap().unwrap().std_dev,
        r.dpa_uncertainty(0, 1, None).unwrap().unwrap().std_dev
    );
    assert!(r.dpa_uncertainty(0, 1, Some("W")).is_err());

    // A per-bin flux sigma moves it group by group: with both bins at 10%
    // independently, the relative spread is 10% times the root sum of squares
    // of the groups' shares.
    let r = run(vec![Source::FluxSpectrum], Some(vec![0.1, 0.1]));
    let data = yamc_nuclide::nuclide::get_or_load_nuclide(
        "Fe56",
        &HashMap::from([("Fe56".to_string(), dir.to_str().unwrap().to_string())]),
        &yamc_nuclide::load_scope::LoadScope::activation([MT_DAMAGE_ENERGY].into()),
    )
    .unwrap();
    let reaction = data.reactions_for_temp("294").unwrap()[&MT_DAMAGE_ENERGY].clone();
    let shares: Vec<f64> = BOUNDARIES
        .windows(2)
        .zip(MASSES)
        .map(|(w, m)| group_average(&reaction, w[0], w[1]) * m)
        .collect();
    let total: f64 = shares.iter().sum();
    let expected = 0.1
        * shares
            .iter()
            .map(|s| (s / total).powi(2))
            .sum::<f64>()
            .sqrt();
    let e = r.damage_energy_uncertainty(0, 1, None).unwrap().unwrap();
    let relative = e.std_dev.unwrap() / e.nominal;
    assert!(
        (relative / expected - 1.0).abs() < 0.15,
        "{relative} vs {expected}"
    );
    // No cross-section draw, so nothing to report on the deposits.
    assert!(r.uncertainty_info[&0].damage_energy.is_empty());
}
