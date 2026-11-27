//! A self-shielded rate follows the composition the irradiation leaves.
//!
//! The shielding mixture is the composition, so a shielded rate changes as
//! the irradiation changes what the material is made of. A shielded case used
//! to be collapsed once, at the starting composition, and every step reused
//! that collapse: splitting the timeline then changed nothing, and the answer
//! held to the starting composition's rates. Each step now collapses at the
//! composition it starts from, and its uncertainty replicas and its energy
//! breakdown read that collapse.
//!
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc_materials::Material;
use yani_transmute::uncertainty::{DataUncertainty, Source};
use yani_transmute::{
    transmute_material_shielded, MultigroupSpectrum, Shielding, TransmutationResults, TransmuteStep,
};

/// Thermal, a group holding the 18.8 eV resonance, the rest of the
/// epithermal range, and fast.
const GROUPS: [f64; 5] = [1.0e-5, 0.625, 1.0e2, 1.0e5, 2.0e7];
const FLUX: [f64; 4] = [2.0e14, 6.0e14, 6.0e14, 6.0e14];
/// Enough tungsten for the resonance to shield itself strongly.
const CHORD_CM: f64 = 2.0;
/// Long enough to burn about a third of the W186.
const IRRADIATION_S: f64 = 0.25 * 365.25 * 86400.0;
const ABSORBER: &str = "W186";
const CAPTURE: &str = "(n,gamma)";

fn chain() -> Arc<HashMap<String, yani::ChainNuclide>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    Arc::new(yani::parse_chain_arrow(&path).expect("parse chain"))
}

fn tungsten(data: &str) -> Material {
    let mut m = Material::new(
        HashMap::from([(ABSORBER.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .expect("W186 material");
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([(ABSORBER.to_string(), data.to_string())]),
        None,
    )
    .expect("read W186");
    m
}

fn spectra(sigma: Option<f64>) -> Vec<MultigroupSpectrum> {
    let total: f64 = FLUX.iter().sum();
    vec![MultigroupSpectrum {
        boundaries: GROUPS.to_vec(),
        masses: FLUX.iter().map(|f| f / total).collect(),
        flux_error: sigma.map(|s| {
            yani_transmute::flux_uncertainty::FluxError::RelativeStdDev(vec![s; FLUX.len()])
        }),
    }]
}

/// The irradiation in `n` equal steps.
fn steps(n: usize) -> Vec<TransmuteStep> {
    let total: f64 = FLUX.iter().sum();
    (0..n)
        .map(|_| TransmuteStep {
            dt: IRRADIATION_S / n as f64,
            irradiation: Some((0, total)),
        })
        .collect()
}

fn run(
    data: &str,
    n: usize,
    shielding: Option<&Shielding>,
    request: Option<&DataUncertainty>,
    sigma: Option<f64>,
) -> TransmutationResults {
    transmute_material_shielded(
        &mut tungsten(data),
        &spectra(sigma),
        &steps(n),
        chain(),
        &Default::default(),
        Default::default(),
        request,
        shielding,
    )
    .expect("transmute")
}

fn absorber_left(results: &TransmutationResults, id: u32, n: usize) -> f64 {
    results
        .get_nuclide_density(id, ABSORBER, n)
        .expect("W186 after the irradiation")
}

/// The absorber's capture rate per atom on step `k`, as the solve used it.
fn capture_rate(results: &TransmutationResults, id: u32, k: usize) -> f64 {
    results
        .get_reaction_rates(id, k)
        .and_then(|r| r.get(ABSORBER))
        .and_then(|r| r.get(CAPTURE))
        .map(|edges| edges.iter().map(|(_, rate)| rate).sum())
        .expect("the step drove W186 capture")
}

/// Refining the timeline converges, at first order, on one answer. With one
/// collapse for every step it changed nothing at all.
#[test]
fn refining_a_shielded_timeline_converges() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = tungsten(&data).get_material_id().unwrap_or(0);
    let left: Vec<f64> = [1, 2, 4, 8, 16]
        .iter()
        .map(|&n| absorber_left(&run(&data, n, Some(&shielding), None, None), id, n))
        .collect();
    let start = run(&data, 1, None, None, None)
        .get_nuclide_density(id, ABSORBER, 0)
        .unwrap();
    let burnt = 1.0 - left[4] / start;
    assert!(
        burnt > 0.2,
        "the test needs the absorber to burn: {:.1}% burnt",
        100.0 * burnt
    );
    let gaps: Vec<f64> = left
        .windows(2)
        .map(|w| (w[1] - w[0]).abs() / w[1])
        .collect();
    assert!(
        gaps[0] > 5e-4,
        "one step and two must differ once each step re-collapses: {left:?}"
    );
    for g in gaps.windows(2) {
        // First order: each halving of the step roughly halves the change.
        assert!(
            g[1] < 0.7 * g[0],
            "refinement must converge: changes {gaps:?} over {left:?}"
        );
    }
}

/// Each step's rate is the collapse of the composition that step started
/// from, exactly: what a standalone collapse of that state gives.
#[test]
fn each_steps_rate_is_its_starting_compositions_collapse() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let initial = tungsten(&data);
    let id = initial.get_material_id().unwrap_or(0);
    let results = run(&data, 4, Some(&shielding), None, None);
    let total: f64 = FLUX.iter().sum();
    let masses: Vec<f64> = FLUX.iter().map(|f| f / total).collect();
    let mut rates = Vec::new();
    for k in 0..4 {
        let mut at = results
            .get_material(id, k)
            .expect("the step's start")
            .clone();
        at.nuclide_data = initial.nuclide_data.clone();
        let (collapsed, _, _) =
            yani_transmute::multigroup::compute_multigroup_reaction_rates_shielded(
                &at,
                &chain(),
                &masses,
                &GROUPS,
                total,
                Some(&shielding),
            );
        let expected = collapsed[ABSORBER][CAPTURE];
        let rate = capture_rate(&results, id, k);
        assert!(
            (rate - expected).abs() <= 1e-12 * expected,
            "step {k}: solved with {rate:e}, its composition collapses to {expected:e}"
        );
        rates.push(rate);
    }
    assert!(
        rates.windows(2).all(|w| (w[1] - w[0]).abs() > 1e-4 * w[0]),
        "the composition moves the shielded rate: {rates:?}"
    );
    // A dilute rate per atom does not depend on the composition, so it holds.
    let dilute = run(&data, 4, None, None, None);
    let flat: Vec<f64> = (0..4).map(|k| capture_rate(&dilute, id, k)).collect();
    assert!(
        flat.iter().all(|r| *r == flat[0]),
        "dilute rates hold: {flat:?}"
    );
}

/// A dilute timeline split into steps is still the single-step solve: its
/// matrix does not depend on the composition, and nothing re-collapses.
#[test]
fn a_dilute_timeline_still_splits_to_the_same_answer() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let id = tungsten(&data).get_material_id().unwrap_or(0);
    let one = absorber_left(&run(&data, 1, None, None, None), id, 1);
    let many = absorber_left(&run(&data, 16, None, None, None), id, 16);
    assert!((one - many).abs() <= 1e-12 * one, "{one:e} vs {many:e}");
}

/// Replicas that perturb nothing reproduce the nominal run exactly at every
/// step, so each step's replicas perturb the collapse that step used. Replicas
/// left on the starting composition's collapse would sit on the frozen answer
/// instead, percents away.
#[test]
fn unperturbed_replicas_follow_every_steps_collapse() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = tungsten(&data).get_material_id().unwrap_or(0);
    let request = DataUncertainty {
        seed: 1,
        samples: Some(2),
        sources: vec![Source::FluxSpectrum],
        attribution: false,
        ..Default::default()
    };
    let n = 4;
    let results = run(&data, n, Some(&shielding), Some(&request), Some(0.0));
    let ensemble = results.uncertainty.get(&id).expect("an ensemble");
    for k in 0..n {
        let nominal = absorber_left(&results, id, k + 1);
        for (replica, sample) in ensemble.samples_at(k, ABSORBER).iter().enumerate() {
            assert_eq!(
                *sample, nominal,
                "step {k}, replica {replica}: perturbing nothing must be the nominal run"
            );
        }
    }
}

/// A perturbed ensemble spreads around the re-collapsed nominal, not the
/// frozen one.
#[test]
fn a_perturbed_ensemble_spreads_around_the_recollapsed_answer() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = tungsten(&data).get_material_id().unwrap_or(0);
    let n = 8;
    let request = DataUncertainty {
        seed: 3,
        samples: Some(64),
        sources: vec![Source::FluxSpectrum],
        attribution: false,
        ..Default::default()
    };
    // Small enough that the ensemble's sampling offset, one correlated draw
    // per replica over every step, sits far inside the frozen answer's gap.
    let results = run(&data, n, Some(&shielding), Some(&request), Some(1.0e-4));
    let nominal = absorber_left(&results, id, n);
    // The frozen answer: one step, one collapse.
    let frozen = absorber_left(&run(&data, 1, Some(&shielding), None, None), id, 1);
    let gap = (frozen - nominal).abs() / nominal;
    let samples = results.uncertainty[&id].samples_at(n - 1, ABSORBER);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let off = (mean - nominal).abs() / nominal;
    assert!(
        off < gap / 10.0,
        "ensemble mean {mean:e} is {:.3}% off the nominal {nominal:e}; the frozen \
         answer {frozen:e} is {:.3}% away",
        100.0 * off,
        100.0 * gap
    );
}

/// A later step's energy breakdown is the rate that step was solved with:
/// it is re-derived at the composition the step started from.
#[test]
fn a_later_steps_breakdown_sums_to_its_rate() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let id = tungsten(&data).get_material_id().unwrap_or(0);
    let results = run(&data, 4, Some(&shielding), None, None);
    for k in 0..4 {
        let spectrum = results
            .get_reaction_rate_spectrum(id, ABSORBER, CAPTURE, k)
            .expect("a breakdown");
        let sum: f64 = spectrum.rates.iter().sum();
        let rate = capture_rate(&results, id, k);
        assert!(
            (sum - rate).abs() <= 1e-9 * rate,
            "step {k}: breakdown {sum:e} against the rate {rate:e}"
        );
    }
}

/// Displacement damage over a shielded timeline: each step's damage energy is
/// collapsed where that step's rates were, and the replicas' damage factors
/// line up with those collapses, so replicas that perturb nothing reproduce
/// the nominal damage at every state.
#[test]
fn shielded_damage_follows_each_steps_collapse() {
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let mut m = tungsten(&data);
    let id = m.get_material_id().unwrap_or(0);
    let n = 4;
    let results = yani_transmute::transmute_materials(
        vec![yani_transmute::TransmuteCase {
            material: &mut m,
            spectra: spectra(Some(0.0)),
            steps: steps(n),
            shielding: Some(shielding),
            product_threshold: yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        Some(&DataUncertainty {
            seed: 1,
            samples: Some(2),
            sources: vec![Source::FluxSpectrum],
            attribution: false,
            ..Default::default()
        }),
        Some(&yani_transmute::DamageRequest::default()),
        None,
    );
    let results = match results {
        Ok(r) => r,
        Err(e) if e.to_string().contains("MT=444") => {
            eprintln!("skipping: no damage-energy data in the W186 fixture");
            return;
        }
        Err(e) => panic!("transmute: {e}"),
    };
    let damage = results
        .get_displacement_damage(id)
        .expect("damage reported");
    // The damage energy accrues every step.
    for k in 0..n {
        assert!(damage.damage_energy[k + 1] > damage.damage_energy[k]);
    }
    let replicas = damage.replicas.as_ref().expect("damage replicas");
    for (r, series) in replicas.damage_energy.iter().enumerate() {
        assert_eq!(
            series, &damage.damage_energy,
            "replica {r} perturbed nothing and must be the nominal damage"
        );
    }
}

/// A processed solve steps every material together, and re-collapses each
/// shielded step the same way: with a transfer that moves nothing, it is the
/// unprocessed shielded run at every step.
#[test]
fn a_processed_shielded_solve_recollapses_alike() {
    use yani_transmute::{Processing, ProcessingRun, Rate};
    let Some(data) = yamc_test_cache::nuclide(ABSORBER) else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let shielding = Shielding::new(CHORD_CM).expect("a positive chord");
    let n = 4;
    let mut plain = tungsten(&data);
    plain.set_material_id(1);
    let expected = transmute_material_shielded(
        &mut plain,
        &spectra(None),
        &steps(n),
        chain(),
        &Default::default(),
        Default::default(),
        None,
        Some(&shielding),
    )
    .expect("transmute");

    let mut tank = Material::new(HashMap::new(), "atom", "sum", None).expect("empty tank");
    tank.set_material_id(9);
    tank.volume = Some(1.0);
    let mut idle = Processing::new();
    idle.add_transfer(1, Some("W"), Rate::Constant(0.0), "1/s", Some(9), None)
        .unwrap();
    let mut processed = tungsten(&data);
    processed.set_material_id(1);
    processed.volume = Some(1.0);
    let got = yani_transmute::transmute_materials(
        vec![yani_transmute::TransmuteCase {
            material: &mut processed,
            spectra: spectra(None),
            steps: steps(n),
            shielding: Some(shielding),
            product_threshold: yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        }],
        chain(),
        &Default::default(),
        Default::default(),
        None,
        None,
        Some(ProcessingRun {
            processing: &idle,
            receivers: vec![tank],
        }),
    )
    .expect("processed solve");
    for k in 1..=n {
        let a = absorber_left(&expected, 1, k);
        let b = absorber_left(&got, 1, k);
        assert!(
            (a - b).abs() <= 1e-9 * a,
            "step {k}: processed {b:e} vs {a:e}"
        );
        assert_eq!(
            capture_rate(&got, 1, k - 1),
            capture_rate(&expected, 1, k - 1)
        );
    }
}
