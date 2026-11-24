//! MF=40 isomer production uncertainty on a coupled transport-transmutation:
//! each replica the whole coupled run again, every step tallying and splitting
//! its isomeric partials on production curves carrying that replica's draw.
//!
//! The case is ENDF/B-VIII.1 In115 at 14 MeV, whose `(n,2n)` to In114_m1 and
//! `(n,n')` to In115_m1 have MF=40 covariance. Self-skips when the
//! nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

const RADIUS: f64 = 3.0;
const SEED: u64 = 11;
const BASE_SEED: u64 = 7;
const PARTICLES: usize = 4000;
const ISOMERS: [&str; 2] = ["In114_m1", "In115_m1"];

/// The chain with the In115 and Mo92 branching subsection over it, or `None`
/// when a fixture is missing.
fn chain() -> Option<yani::LoadedChain> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let chain = manifest.join("tests/transmutation-endf-b8.1-sfr.arrow");
    let branch = manifest.join("../yani-transmute/tests/fixtures/endf-b8.1-branching-In115-Mo92");
    if !chain.join("reactions").is_dir() || !branch.is_dir() {
        return None;
    }
    let dir = |d: PathBuf| d.to_string_lossy().into_owned();
    Some(
        yani::load_chain_parts(
            &dir(chain.join("decay")),
            Some(&dir(chain.join("reactions"))),
            None,
            &[("branching", &dir(branch))],
        )
        .expect("chain"),
    )
}

/// A bare In115 sphere around a 14 MeV point source.
fn model(path: &str) -> Model {
    let mut m = Material::new(
        HashMap::from([("In115".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.31),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.volume = Some(4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3));
    m.transmutable = true;
    m.read_nuclear_data(
        &HashMap::from([("In115".to_string(), path.to_string())]),
        None,
    )
    .expect("read In115");
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
    let cell = Cell::new(Some(1), region, Some("in".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).expect("geometry");
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![1.406e7], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = Verbose::silent();
    model
}

/// `pulses` short pulses, so the isomers are their productions times the
/// pulse, and an hour's cooldown.
fn schedule(pulses: usize) -> Schedule {
    let pulse = ScheduleStep {
        rate: 1.0e12,
        dt: 1.0,
        is_pulse: true,
    };
    let mut steps = vec![pulse; pulses];
    steps.push(ScheduleStep {
        rate: 0.0,
        dt: 3600.0,
        is_pulse: false,
    });
    Schedule::new(steps).expect("schedule")
}

fn request(sources: Vec<UqSource>, samples: usize) -> DataUncertainty {
    DataUncertainty {
        seed: SEED,
        samples: Some(samples),
        sources,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn transmute(
    path: &str,
    method: &str,
    schedule: &Schedule,
    loaded: &yani::LoadedChain,
    branch: &Arc<yani::BranchTable>,
    seed: u64,
    request: Option<&DataUncertainty>,
) -> TransmutationResults {
    model(path)
        .transmute(
            method,
            schedule,
            Arc::clone(&loaded.chain),
            Arc::clone(branch),
            loaded.parts,
            &TransportSettings {
                total_particles: Some(PARTICLES),
                seed,
                threads: Some(1),
                ..Default::default()
            },
            request,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
            None,
        )
        .expect("transmute")
}

fn inventories(results: &TransmutationResults, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.get_material(1, i).expect("step").nuclides.clone())
        .collect()
}

fn replica(results: &TransmutationResults, k: usize, steps: usize) -> Vec<HashMap<String, f64>> {
    (1..=steps)
        .map(|i| results.uncertainty_inventories(1, i).expect("replicas")[k].clone())
        .collect()
}

/// The nominal results are a plain coupled run's, bit for bit, and replica
/// `k` is bit for bit the plain coupled run on the branching table carrying
/// replica `k`'s draw: the isomers it makes move, and nothing a curve does
/// not reach does. The report says what was drawn.
#[test]
fn each_replica_is_the_coupled_run_on_its_own_production_curves() {
    let (Some(path), Some(loaded)) = (yamc_test_cache::transport_nuclide("In115"), chain()) else {
        eprintln!("skipping: In115, chain or branching fixture not fetched");
        return;
    };
    let schedule = schedule(2);
    let steps = 3;
    let request = request(vec![UqSource::IsomerBranching], 3);
    let plain = transmute(
        &path,
        "coupled",
        &schedule,
        &loaded,
        &loaded.branch,
        BASE_SEED,
        None,
    );
    let uq = transmute(
        &path,
        "coupled",
        &schedule,
        &loaded,
        &loaded.branch,
        BASE_SEED,
        Some(&request),
    );
    assert_eq!(inventories(&plain, steps), inventories(&uq, steps));

    let info = &uq.uncertainty_info[&1];
    assert_eq!(info.flux_response, FluxResponseReport::CoupledReplicas);
    for kind in ["(n,2n)", "(n,n')"] {
        assert!(
            info.isomer_branchings_perturbed
                .contains(&("In115".to_string(), kind.to_string())),
            "{:?}",
            info.isomer_branchings_perturbed
        );
    }
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("isomeric branching covariance (MF=40)")));

    let replicas = CoupledReplicas::new(&request, &loaded.chain, &loaded.branch).expect("carried");
    let nominal = &inventories(&plain, steps)[0];
    for k in 0..3 {
        assert_eq!(replicas.transport_seed(BASE_SEED, k), BASE_SEED);
        let branch = replicas.branch(&loaded.branch, k);
        assert!(!Arc::ptr_eq(&branch, &loaded.branch));
        let rerun = transmute(
            &path, "coupled", &schedule, &loaded, &branch, BASE_SEED, None,
        );
        let ours = replica(&uq, k as usize, steps);
        assert_eq!(
            ours,
            inventories(&rerun, steps),
            "replica {k} is the coupled run on its own production curves"
        );
        for isomer in ISOMERS {
            assert_ne!(
                ours[0][isomer], nominal[isomer],
                "replica {k} moves {isomer}"
            );
        }
    }
}

/// On one pulse with the nominal random numbers, a coupled replica's
/// transport is the nominal one, history for history: a curve moves what is
/// tallied, not what is transported. Its isomers are then the independent
/// replica's of the same index, which folds the same draw against the
/// interval productions of that one transport, to the rounding of the two
/// routes: the coupled replica tallies the drawn curve, the independent one
/// sums the nominal curve's intervals times their multipliers.
///
/// Except where a draw takes a stated production above the reaction's total
/// at some energies. The coupled replica splits at each collision energy, so
/// the nominal rule clips it there; the independent fold sees only the
/// integral, and clips only when that exceeds the total. The coupled replica
/// then makes a little less, by what its rule reports clipped, and no more.
#[test]
fn on_one_pulse_each_replica_is_the_independent_one() {
    let (Some(path), Some(loaded)) = (yamc_test_cache::transport_nuclide("In115"), chain()) else {
        eprintln!("skipping: In115, chain or branching fixture not fetched");
        return;
    };
    const REPLICAS: usize = 8;
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e12,
        dt: 1.0,
        is_pulse: true,
    }])
    .expect("schedule");
    let request = request(vec![UqSource::IsomerBranching], REPLICAS);
    let run = |method: &str| {
        transmute(
            &path,
            method,
            &schedule,
            &loaded,
            &loaded.branch,
            BASE_SEED,
            Some(&request),
        )
    };
    let (coupled, independent) = (run("coupled"), run("independent"));
    assert_eq!(inventories(&coupled, 1), inventories(&independent, 1));
    let nominal = &inventories(&coupled, 1)[0];
    let replicas = CoupledReplicas::new(&request, &loaded.chain, &loaded.branch).expect("carried");
    let (mut moved, mut clipped_replicas) = (0.0_f64, 0);
    for k in 0..REPLICAS {
        // What the rule clipped on this replica's curves, from its rerun.
        let rerun = transmute(
            &path,
            "coupled",
            &schedule,
            &loaded,
            &replicas.branch(&loaded.branch, k as u64),
            BASE_SEED,
            None,
        );
        let clipped = rerun.branching_report[&1][0]
            .channels
            .iter()
            .any(|c| c.clipped_share > 0.0);
        clipped_replicas += usize::from(clipped);
        let (c, i) = (replica(&coupled, k, 1), replica(&independent, k, 1));
        for isomer in ISOMERS {
            let (c, i, n) = (c[0][isomer], i[0][isomer], nominal[isomer]);
            eprintln!(
                "replica {k} {isomer}: coupled {:+.6e}, independent {:+.6e}{}",
                c / n - 1.0,
                i / n - 1.0,
                if clipped { " (clipped)" } else { "" }
            );
            let difference = (c - i) / n;
            if clipped {
                assert!(
                    (-1e-4..1e-9).contains(&difference),
                    "replica {k} {isomer}: clipping takes {difference:e}, at most a little"
                );
            } else {
                assert!(
                    difference.abs() < 1e-9,
                    "replica {k} {isomer}: coupled {c:e} against independent {i:e}"
                );
            }
            moved = moved.max((c / n - 1.0).abs());
        }
    }
    assert!(
        moved > 1e-3,
        "the draws move the isomers, by up to {moved:e}"
    );
    assert!(
        clipped_replicas < REPLICAS,
        "most replicas are unclipped, so the comparison is exact for them"
    );
}
