//! A coupled transport-transmutation whose replicas ride one transport per
//! step as likelihood-ratio weights (option B of #462), against the reruns
//! of the whole coupled calculation per replica (option A) it stands in for.
//!
//! The case is a W186 sphere at 1 keV, its resonance parameters sampled,
//! burned hard enough that each replica's depletion of W186 moves the
//! self-shielding of the next step's flux: the feedback both methods carry,
//! one by rerunning and one by the replicas' compositions in their weights.
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yani_transmute::uncertainty::{DataUncertainty, FluxResponseReport, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep, TransmutationResults};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 31;
const BASE_SEED: u64 = 5;
const RADIUS: f64 = 5.0;

type Chain = Arc<HashMap<String, yani::ChainNuclide>>;

fn w186(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("W186")?);
    let dir = tmp.join("W186.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached W186") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    let evaluation =
        endf::Material::from_str(&String::from_utf8(raw).expect("text")).expect("parses");
    yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance");
    yamc_convert::resonance_parameters::write_resonance_parameters(&evaluation, &dir)
        .expect("resonance parameters");
    Some(dir)
}

fn chain() -> Option<Chain> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    path.exists()
        .then(|| Arc::new(yani::parse_chain_arrow(&path).expect("chain")))
}

fn model(dir: &Path) -> Model {
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.volume = Some(4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3));
    m.transmutable = true;
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read W186");
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
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).expect("geometry");
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e3], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
}

/// `pulses` pulses, each burning a few percent of the W186.
fn burnout(pulses: usize, rate: f64) -> Schedule {
    Schedule::new(vec![
        ScheduleStep {
            rate,
            dt: 1.0e4,
            is_pulse: true,
        };
        pulses
    ])
    .expect("schedule")
}

fn transmute(
    dir: &Path,
    schedule: &Schedule,
    chain: &Chain,
    particles: usize,
    request: Option<&DataUncertainty>,
) -> TransmutationResults {
    model(dir)
        .transmute(
            "coupled",
            schedule,
            Arc::clone(chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &TransportSettings {
                total_particles: Some(particles),
                seed: BASE_SEED,
                threads: Some(1),
                ..Default::default()
            },
            request,
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        )
        .expect("transmute")
}

/// Every replica's `nuclide` at step `step` over the nominal, minus one.
fn shifts(results: &TransmutationResults, nuclide: &str, step: usize) -> Vec<f64> {
    let nominal = results
        .get_nuclide_density(1, nuclide, step)
        .expect("nominal");
    results
        .uncertainty_inventories(1, step)
        .expect("replicas")
        .iter()
        .map(|i| i.get(nuclide).copied().unwrap_or(0.0) / nominal - 1.0)
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// The weighted replicas' nominal is the plain coupled run's, and each
/// replica follows the reruns' replica of the same index: the same draw,
/// and the same feedback through its own depleted composition, to within
/// the reruns' Monte Carlo noise.
///
/// Three pulses burn 46% of the W186. At 200 000 particles the draws spread
/// W186 and W187 by four to five times the reruns' noise, and the reruns
/// less the weighted replicas spread by that noise alone at every step
/// (W186 after the third pulse: 5.4e-3 of spread, 1.03e-3 of noise, 1.06e-3
/// of difference), so each weighted replica is its rerun. Minutes of
/// transport, so ignored by default; `PARTICLES` sets the count.
#[test]
#[ignore = "validation: minutes of transport"]
fn each_weighted_replica_follows_its_rerun() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    let particles: usize = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    const REPLICAS: usize = 12;
    let steps = 3;
    let schedule = burnout(steps, 4.0e21);
    let plain = transmute(&dir, &schedule, &chain, particles, None);
    let request = |sources, weights| DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources,
        coupled_replica_weights: weights,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let weighted = transmute(
        &dir,
        &schedule,
        &chain,
        particles,
        Some(&request(vec![UqSource::CrossSections], true)),
    );
    let weighted_time = t.elapsed();
    let t = std::time::Instant::now();
    let reruns = transmute(
        &dir,
        &schedule,
        &chain,
        particles,
        Some(&request(
            vec![UqSource::Statistical, UqSource::CrossSections],
            false,
        )),
    );
    let rerun_time = t.elapsed();
    let noise = transmute(
        &dir,
        &schedule,
        &chain,
        particles,
        Some(&request(vec![UqSource::Statistical], false)),
    );
    eprintln!("weighted {weighted_time:.2?}, reruns {rerun_time:.2?}");
    for i in 1..=steps {
        assert_eq!(
            plain.get_material(1, i).map(|m| &m.nuclides),
            weighted.get_material(1, i).map(|m| &m.nuclides),
            "the nominal is the plain run's"
        );
    }
    let info = &weighted.uncertainty_info[&1];
    assert_eq!(
        info.flux_response,
        FluxResponseReport::CoupledReplicaWeights { replicas: REPLICAS }
    );
    assert_eq!(info.replica_weight_noise.len(), steps);
    eprintln!(
        "replica weight noise by step: {:?}",
        info.replica_weight_noise
    );
    let burned = 1.0
        - plain.get_nuclide_density(1, "W186", steps).unwrap()
            / plain.get_nuclide_density(1, "W186", 0).unwrap();
    eprintln!("W186 burned: {burned:.3}");
    for step in 1..=steps {
        for nuclide in ["W186", "W187"] {
            let (b, a) = (
                shifts(&weighted, nuclide, step),
                shifts(&reruns, nuclide, step),
            );
            let nominal = noise.get_nuclide_density(1, nuclide, step).unwrap();
            let sigma = noise.get_nuclide_uncertainty(1, nuclide, step).unwrap() / nominal;
            let difference: Vec<f64> = a.iter().zip(&b).map(|(a, b)| a - b).collect();
            eprintln!(
                "step {step} {nuclide}: draws spread {:.3e} (weights) {:.3e} (reruns), noise \
                 {sigma:.3e}, reruns less weights {:+.3e} on average, {:.3e} about that",
                sd(&b),
                sd(&a),
                mean(&difference),
                sd(&difference)
            );
            for k in 0..REPLICAS {
                eprintln!(
                    "  replica {k:2}: reruns {:+.4e}, weights {:+.4e}",
                    a[k], b[k]
                );
            }
            assert!(
                sd(&b) > 3.0 * sigma,
                "step {step} {nuclide}: the draws must move it past the noise"
            );
            // Twelve differences of variance `sigma^2` (the weighted
            // replicas' own noise is a tenth of it, `replica_weight_noise`)
            // have a standard deviation under 1.6 of it with probability over
            // 0.995.
            assert!(
                sd(&difference) < 1.6 * sigma,
                "step {step} {nuclide}: the reruns and the weighted replicas differ by {:.3e} \
                 about the mean, past the reruns' noise {sigma:.3e}",
                sd(&difference)
            );
            assert!(mean(&difference).abs() < 4.0 * sigma);
        }
    }
}

/// On the first step every replica still has the nominal composition, so
/// the weighted coupled transport is independent mode's replica-weighted
/// one, draw for draw and history for history, and each replica's inventory
/// after it is the independent replica's of the same index.
#[test]
fn the_first_step_is_independent_modes_replica_weights() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    const REPLICAS: usize = 6;
    let schedule = burnout(1, 1.0e16);
    let request = |weights| DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources: vec![UqSource::CrossSections],
        coupled_replica_weights: weights,
        ..Default::default()
    };
    let weighted = transmute(&dir, &schedule, &chain, 4000, Some(&request(true)));
    let independent = model(&dir)
        .transmute(
            "independent",
            &schedule,
            Arc::clone(&chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &TransportSettings {
                total_particles: Some(4000),
                seed: BASE_SEED,
                threads: Some(1),
                ..Default::default()
            },
            Some(&request(false)),
            yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
        )
        .expect("transmute");
    assert_eq!(
        weighted.get_material(1, 1).map(|m| &m.nuclides),
        independent.get_material(1, 1).map(|m| &m.nuclides)
    );
    let (w, i) = (
        shifts(&weighted, "W187", 1),
        shifts(&independent, "W187", 1),
    );
    for k in 0..REPLICAS {
        assert!(
            (w[k] - i[k]).abs() < 1e-9,
            "replica {k}: weighted coupled {:+e}, independent {:+e}",
            w[k],
            i[k]
        );
    }
    assert!(
        w.iter().any(|x| x.abs() > 1e-3),
        "the draws move W187: {w:?}"
    );
}

/// What one weighted transport costs beside a plain one on this sphere: the
/// wall time of the run, of which the transport loop alone, with 12 replicas
/// and without. Prints, asserts nothing.
#[test]
#[ignore = "timings"]
fn cost_of_a_weighted_transport() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path()) else {
        return;
    };
    let particles: usize = std::env::var("PARTICLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    for replicas in [None, Some(12)] {
        let mut m = model(&dir);
        let t = std::time::Instant::now();
        m.simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed: BASE_SEED,
            threads: Some(1),
            data_uncertainty: replicas.map(|replicas| yamc::model::TransportDataUncertainty {
                seed: SEED,
                replicas,
            }),
            ..Default::default()
        })
        .expect("transport");
        eprintln!(
            "{replicas:?} replicas, {particles} particles: run {:.2?}, transport loop {:.2} s",
            t.elapsed(),
            m.last_transport_secs.unwrap_or(f64::NAN)
        );
    }
}
