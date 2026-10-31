//! Uncertainty on a coupled transport-transmutation, each replica the whole
//! coupled run again.
//!
//! The case is the one the method exists for: a B10 sphere burned hard enough
//! that it shapes its own flux. Every pulse burns a fifth of the B10, so the
//! sphere leaks more and the later steps' rates depend on what the earlier
//! steps left, noise and decay data included.
//!
//! Each replica must BE the coupled run, not an approximation of it: replica
//! `k` is bit-identical to a plain coupled run with replica `k`'s transport
//! seed and replica `k`'s decay data. On one pulse, where the two methods
//! solve the same problem, the coupled replicas' spread must match the
//! independent method's statistical sigma, which comes from the per-history
//! covariance of a single transport instead. Self-skips when the nuclear-data
//! fixtures are missing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yani_transmute::coupled_uncertainty::CoupledReplicas;
use yani_transmute::uncertainty::{DataUncertainty, FluxResponseReport, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep, TransmutationResults};

const RADIUS: f64 = 2.0;
const PARTICLES: usize = 2000;
const BASE_SEED: u64 = 11;

type Chain = Arc<HashMap<String, yani::ChainNuclide>>;

fn b10() -> Option<String> {
    let path = yamc_test_cache::nuclide("B10")?;
    std::path::Path::new(&path)
        .join("reactions.arrow")
        .exists()
        .then_some(path)
}

fn chain() -> Option<Chain> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    path.exists()
        .then(|| Arc::new(yani::parse_chain_arrow(&path).expect("chain")))
}

/// A bare B10 sphere around a 1 keV point source.
fn model(path: &str) -> Model {
    let mut m = Material::new(
        HashMap::from([("B10".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(2.3),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.volume = Some(4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3));
    m.transmutable = true;
    m.read_nuclear_data(
        &HashMap::from([("B10".to_string(), path.to_string())]),
        None,
    )
    .expect("read B10");
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("b10".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).expect("geometry");
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e3], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = Verbose::silent();
    model
}

fn settings(seed: u64) -> TransportSettings {
    TransportSettings {
        total_particles: Some(PARTICLES),
        seed,
        threads: Some(1),
        ..Default::default()
    }
}

/// Three pulses that each burn about a fifth of the B10, and a cooldown.
fn burnout() -> Schedule {
    let pulse = ScheduleStep {
        rate: 1.0e20,
        dt: 1.0e4,
        is_pulse: true,
    };
    let cooldown = ScheduleStep {
        rate: 0.0,
        dt: 3.0e8,
        is_pulse: false,
    };
    Schedule::new(vec![pulse.clone(), pulse.clone(), pulse, cooldown]).expect("schedule")
}

fn request(sources: Vec<UqSource>, samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: 3,
        samples: Some(samples),
        sources,
        ..Default::default()
    }
}

fn transmute(
    model: &mut Model,
    method: &str,
    schedule: &Schedule,
    chain: &Chain,
    seed: u64,
    request: Option<&DataUncertainty>,
) -> TransmutationResults {
    model
        .transmute(
            method,
            schedule,
            Arc::clone(chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &settings(seed),
            request,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        )
        .expect("transmute")
}

/// Every step's inventory of material 1.
fn inventories(results: &TransmutationResults, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.get_material(1, i).expect("step").nuclides.clone())
        .collect()
}

/// Replica `k`'s every-step inventory, as the ensemble holds it.
fn replica(results: &TransmutationResults, k: usize, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.uncertainty_inventories(1, i).expect("replicas")[k].clone())
        .collect()
}

/// The nominal results and the model left behind are those of a run without
/// uncertainty, bit for bit; the burnout really does move the flux; and each
/// replica is bit-identical to the plain coupled run with its transport seed.
#[test]
fn each_statistical_replica_is_a_whole_coupled_run() {
    let (Some(path), Some(chain)) = (b10(), chain()) else {
        eprintln!("skipping: B10 or chain fixture not fetched");
        return;
    };
    let schedule = burnout();
    let steps = 4;
    let request = request(vec![UqSource::Statistical], 3);

    let mut plain_model = model(&path);
    let plain = transmute(
        &mut plain_model,
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        None,
    );
    let mut uq_model = model(&path);
    let uq = transmute(
        &mut uq_model,
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        Some(&request),
    );
    assert_eq!(inventories(&plain, steps), inventories(&uq, steps));
    let left = |m: &Model| m.geometry.materials()[0].nuclides.clone();
    assert_eq!(
        left(&plain_model),
        left(&uq_model),
        "the model is left as the nominal run leaves it"
    );

    // Real feedback: the B10 burns down step on step, and the B10 that
    // remains captures at a rate per atom that rises as the sphere thins.
    let b10_at = |i: usize| plain.get_nuclide_density(1, "B10", i).expect("B10");
    let burned: Vec<f64> = (0..=3).map(b10_at).collect();
    assert!(
        burned[3] < 0.75 * burned[0],
        "the pulses burn the B10 down: {burned:?}"
    );
    let per_atom = |i: usize| (burned[i - 1] / burned[i]).ln();
    assert!(
        per_atom(3) > 1.01 * per_atom(1),
        "the capture per B10 atom rises as the sphere burns: {} then {}",
        per_atom(1),
        per_atom(3)
    );

    let info = &uq.uncertainty_info[&1];
    assert_eq!(info.samples, 3);
    assert_eq!(info.flux_response, FluxResponseReport::CoupledReplicas);
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s == "tallied-rate statistics"));

    let replicas = CoupledReplicas::new(&request, &chain).expect("carried");
    for k in 0..3 {
        let seed = replicas.transport_seed(BASE_SEED, k as u64);
        assert_ne!(seed, BASE_SEED);
        let rerun = transmute(&mut model(&path), "coupled", &schedule, &chain, seed, None);
        assert_eq!(
            replica(&uq, k, steps),
            inventories(&rerun, steps),
            "replica {k} is the coupled run with its own seed"
        );
    }
    let spread = uq.uncertainty_inventories(1, 3).expect("replicas");
    assert!(
        spread.iter().any(|r| r["B10"] != spread[0]["B10"]),
        "independent seeds must give different burnups"
    );
}

/// With the decay data drawn and no statistical source, a replica is the
/// coupled run on its own chain with the nominal random numbers, bit for bit,
/// and its tritium decays with the half-life the ensemble records for it.
///
/// The chain fixture states no half-life sigma, so the burnout's tritium is
/// given one of 10% here.
#[test]
fn each_decay_data_replica_is_the_coupled_run_on_its_own_chain() {
    let (Some(path), Some(chain)) = (b10(), chain()) else {
        eprintln!("skipping: B10 or chain fixture not fetched");
        return;
    };
    let mut edited = (*chain).clone();
    let h3 = edited.get_mut("H3").expect("H3 in the chain");
    h3.half_life_uncertainty = Some(0.1 * h3.half_life.expect("H3 decays"));
    let chain: Chain = Arc::new(edited);
    let schedule = burnout();
    let steps = 4;
    let request = request(vec![UqSource::HalfLife], 2);
    let uq = transmute(
        &mut model(&path),
        "coupled",
        &schedule,
        &chain,
        BASE_SEED,
        Some(&request),
    );
    let info = &uq.uncertainty_info[&1];
    assert!(info.half_lives_perturbed.contains("H3"), "{info:?}");
    assert!(info
        .not_perturbed
        .iter()
        .any(|s| s == "tallied-rate statistics"));

    let replicas = CoupledReplicas::new(&request, &chain).expect("carried");
    let half_lives = uq.uncertainty[&1].half_lives();
    for k in 0..2 {
        assert_eq!(replicas.transport_seed(BASE_SEED, k), BASE_SEED);
        let (edited, drawn) = replicas.chain(&chain, k);
        assert_eq!(half_lives[k as usize]["H3"], drawn.half_lives["H3"]);
        assert_ne!(drawn.half_lives["H3"], chain["H3"].half_life.unwrap());
        let rerun = transmute(
            &mut model(&path),
            "coupled",
            &schedule,
            &edited,
            BASE_SEED,
            None,
        );
        assert_eq!(
            replica(&uq, k as usize, steps),
            inventories(&rerun, steps),
            "replica {k} is the coupled run on its own decay data"
        );
    }
}

/// On a single pulse the coupled method's one transport is the independent
/// method's, so the spread of coupled replicas run on fresh seeds must agree
/// with the independent method's statistical sigma, drawn from the
/// per-history covariance of that one transport. 128 coupled replicas know a
/// sigma to about 6% and 1024 independent ones to about 2%; four of their
/// combined standard error, with the covariance's own, is the tolerance. Both
/// also agree with the spread of 64 plain runs on 64 seeds (5.1e-4 for B10).
#[test]
fn on_one_pulse_the_replicas_agree_with_the_per_history_covariance() {
    let (Some(path), Some(chain)) = (b10(), chain()) else {
        eprintln!("skipping: B10 or chain fixture not fetched");
        return;
    };
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e20,
        dt: 1.0e4,
        is_pulse: true,
    }])
    .expect("schedule");
    let run = |method: &str, samples: usize| {
        let request = request(vec![UqSource::Statistical], samples);
        transmute(
            &mut model(&path),
            method,
            &schedule,
            &chain,
            BASE_SEED,
            Some(&request),
        )
    };
    let coupled = run("coupled", 128);
    let independent = run("independent", 1024);
    assert_eq!(
        inventories(&coupled, 1),
        inventories(&independent, 1),
        "one pulse is the same calculation either way"
    );
    for nuclide in ["B10", "Li7"] {
        let a = coupled.uncertainty[&1].std_dev_at(0)[nuclide];
        let b = independent.uncertainty[&1].std_dev_at(0)[nuclide];
        assert!(
            (a / b - 1.0).abs() < 0.3,
            "{nuclide}: coupled replicas give {a:e}, the per-history covariance {b:e}"
        );
    }
}
