//! Each coupled step transports on its own seed.
//!
//! One seed reused for every step makes every step's Monte Carlo noise the
//! same draw, so it accumulates through the run instead of averaging out. At
//! a rate too low to change the composition, the old behaviour gave
//! bit-identical tallies on every step; now the first step keeps the user's
//! seed, later steps get their own, and a rerun repeats them.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportSettings, Verbose};
use yamc::StepTallies;
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani::{BranchTable, ChainNuclide, ChainReaction};
use yani_transmute::{Schedule, ScheduleStep};

const SEED: u64 = 3;

fn nuclide(name: &str, reactions: Vec<ChainReaction>) -> ChainNuclide {
    ChainNuclide {
        name: name.to_string(),
        half_life: None,
        decay_energy: 0.0,
        reactions: reactions.into(),
        decays: vec![],
        fission_yields: None,
        sources: Default::default(),
        half_life_uncertainty: None,
        decay_energy_uncertainty: None,
        decay_evaluated: true,
        excitation_energy: None,
        level_index: None,
        decay_energy_components: Default::default(),
    }
}

/// W186 captures to a stable W187, and nothing else happens.
fn chain() -> Arc<HashMap<String, ChainNuclide>> {
    let capture = ChainReaction {
        kind: "(n,gamma)".to_string(),
        target: Some("W187".to_string()),
        branching: 1.0,
        q_value: None,
        branching_uncertainty: None,
        evaluated_branching: None,
        multiplicity: None,
    };
    Arc::new(HashMap::from([
        ("W186".to_string(), nuclide("W186", vec![capture])),
        ("W187".to_string(), nuclide("W187", vec![])),
    ]))
}

/// Three coupled irradiation steps of a W186 sphere around a 14.1 MeV point
/// source, with a decay-only step between the second and third, at a rate
/// that leaves the composition unchanged to many digits.
fn run(path: &str, capture: &Arc<Tally>) -> StepTallies {
    let mut material = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .unwrap();
    material.set_temperature("294");
    material.set_material_id(1);
    material.transmutable = true;
    material.volume = Some(4.0 / 3.0 * std::f64::consts::PI * 1000.0);
    material
        .read_nuclear_data(
            &HashMap::from([("W186".to_string(), path.to_string())]),
            None,
        )
        .unwrap();
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("w".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![Arc::clone(capture)]);
    model.verbose = Verbose::silent();
    let step = |rate| ScheduleStep {
        rate,
        dt: 1.0,
        is_pulse: rate > 0.0,
    };
    let schedule = Schedule::new(vec![step(1.0e6), step(1.0e6), step(0.0), step(1.0e6)]).unwrap();
    let settings = TransportSettings {
        total_particles: Some(2_000),
        seed: SEED,
        ..Default::default()
    };
    model
        .transmute_with_tallies(
            "coupled",
            &schedule,
            chain(),
            Arc::new(BranchTable::new()),
            Default::default(),
            &settings,
            None,
            0.0,
            None,
        )
        .unwrap()
        .1
}

#[test]
fn every_coupled_step_transports_on_its_own_seed() {
    let Some(path) = yamc_test_cache::transport_nuclide("W186") else {
        eprintln!("skipping: W186 fixture missing");
        return;
    };
    let mut capture = Tally::new();
    capture.scores = vec!["(n,gamma)".parse::<Score>().unwrap()];
    capture.estimator = Estimator::TrackLength;
    let capture = Arc::new(capture);

    let tallies = run(&path, &capture);
    let seeds: Vec<u64> = tallies.runs.iter().map(|r| r.runs[0].seed).collect();
    assert_eq!(seeds.len(), 3, "one transport per irradiation step");
    assert_eq!(seeds[0], SEED, "the first step keeps the user's seed");
    assert!(
        seeds[0] != seeds[1] && seeds[1] != seeds[2] && seeds[0] != seeds[2],
        "every step has its own seed: {seeds:?}"
    );
    let (mean, _) = tallies.evolution(&capture).unwrap();
    assert_ne!(mean[0], mean[1], "the steps' noise is not the same draw");

    let again = run(&path, &capture);
    let (repeat, _) = again.evolution(&capture).unwrap();
    // Same random numbers; only the threads' summation order can differ.
    for (a, b) in mean.iter().flatten().zip(repeat.iter().flatten()) {
        assert!(
            (a - b).abs() <= 1e-12 * a.abs(),
            "a rerun repeats: {a} vs {b}"
        );
    }
    assert_eq!(
        seeds,
        again
            .runs
            .iter()
            .map(|r| r.runs[0].seed)
            .collect::<Vec<_>>()
    );
}
