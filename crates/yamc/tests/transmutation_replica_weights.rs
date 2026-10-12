//! A transport-coupled transmutation whose replicas carry the flux's
//! response to their own cross sections, through the transport's replica
//! weights.
//!
//! The fixture is the cached ENDF/B-VIII.1 W186 directory with the
//! `covariance.arrow` and `resonance_parameters.arrow` the real converter
//! writes from the committed MF=2 and MF=32 evaluation, in a pure tungsten
//! sphere driven at 1 keV, where capture is self-shielded: a fold of each
//! replica's capture against the nominal flux overstates its sigma, and the
//! replica weights do not. Self-skips when the nuclear-data fixtures are
//! missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::resonance_rates::ResonanceMethod;
use yani_transmute::uncertainty::{DataUncertainty, FluxResponseReport, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 31;
const REPLICAS: usize = 8;
const PARTICLES: usize = 4000;
const RADIUS: f64 = 5.0;

/// A copy of the cached W186 directory with the converter's covariance and
/// resonance parameters, or `None` when the fixtures are missing.
fn w186(tmp: &Path) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("W186")?);
    let dir = tmp.join("W186.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached W186") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy section");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    let text = String::from_utf8(raw).expect("ENDF is text");
    let evaluation = endf::Material::from_str(&text).expect("W186 parses");
    assert!(yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance"));
    assert!(
        yamc_convert::resonance_parameters::write_resonance_parameters(&evaluation, &dir)
            .expect("resonance parameters")
    );
    Some(dir)
}

/// The chain the transmutation reads, or `None` when it is not fetched.
fn chain() -> Option<Arc<HashMap<String, yani::ChainNuclide>>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    if !path.exists() {
        return None;
    }
    Some(Arc::new(yani::parse_chain_arrow(&path).expect("chain")))
}

/// A pure W186 sphere around a 1 keV point source, with `tallies`.
fn model(dir: &Path, tallies: Vec<Arc<Tally>>) -> Model {
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
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(PARTICLES),
        seed: 5,
        threads: Some(1),
        ..Default::default()
    }
}

fn request() -> DataUncertainty {
    DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources: vec![UqSource::CrossSections],
        ..Default::default()
    }
}

/// A one-second pulse, short enough that W187 is its capture rate times the
/// pulse, so each replica's W187 over the nominal is its capture rate over
/// the nominal one.
fn schedule() -> Schedule {
    Schedule::new(vec![ScheduleStep {
        rate: 1.0e10,
        dt: 1.0,
        is_pulse: true,
    }])
    .expect("schedule")
}

/// Each replica's W187 over the nominal, minus one, and the run's results.
fn transmute(
    dir: &Path,
    chain: &Arc<HashMap<String, yani::ChainNuclide>>,
    request: Option<&DataUncertainty>,
) -> yani_transmute::TransmutationResults {
    model(dir, vec![])
        .transmute(
            "independent",
            &schedule(),
            Arc::clone(chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &settings(),
            request,
        )
        .expect("transmute")
}

fn shifts(results: &yani_transmute::TransmutationResults) -> Vec<f64> {
    let nominal = results.get_nuclide_density(1, "W187", 1).expect("W187");
    results
        .uncertainty_inventories(1, 1)
        .expect("replicas")
        .iter()
        .map(|i| i.get("W187").copied().unwrap_or(0.0) / nominal - 1.0)
        .collect()
}

/// The transmutation's replica `k` is the transport's replica `k`: the same
/// draws on the same histories, so each replica's capture rate over the
/// nominal one is the capture tally's replica `k` over its nominal, to
/// rounding. The nominal inventory is the one a run without uncertainty
/// gives, bit for bit, and the report says the flux response was carried.
#[test]
fn each_replica_is_the_transports_replica() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };

    let plain = transmute(&dir, &chain, None);
    let weighted = transmute(&dir, &chain, Some(&request()));
    assert_eq!(
        plain.get_material(1, 1).map(|m| &m.nuclides),
        weighted.get_material(1, 1).map(|m| &m.nuclides),
        "replica weights must leave the nominal inventory alone"
    );
    let info = &weighted.uncertainty_info[&1];
    assert_eq!(
        info.flux_response,
        FluxResponseReport::ReplicaWeights { replicas: REPLICAS }
    );
    assert_eq!(info.samples, REPLICAS);
    assert!(
        matches!(
            info.resonance_parameters.get("W186"),
            Some(ResonanceMethod::Sampled { .. })
        ),
        "{:?}",
        info.resonance_parameters
    );
    assert!(!info
        .not_perturbed
        .iter()
        .any(|s| s.starts_with("flux response")));

    let mut capture = Tally::new();
    capture.filters.push(Filter::Cell(CellFilter::from_id(1)));
    capture.scores = vec!["(n,gamma)".parse::<Score>().expect("score")];
    capture.estimator = Estimator::TrackLength;
    capture.reset_accumulation();
    let capture = Arc::new(capture);
    model(&dir, vec![Arc::clone(&capture)])
        .simulate_transport(&TransportSettings {
            data_uncertainty: Some(TransportDataUncertainty {
                seed: SEED,
                replicas: REPLICAS,
            }),
            ..settings()
        })
        .expect("transport");
    let sums = capture.get_replica_sums().expect("replica sums");
    let nominal = capture.finalize().mean[0] * capture.finalize().n_histories as f64;

    let mut moved = 0.0_f64;
    for (k, shift) in shifts(&weighted).into_iter().enumerate() {
        let tally = sums.sums[k] / nominal - 1.0;
        eprintln!("replica {k}: transmutation {shift:+.6e}, capture tally {tally:+.6e}");
        assert!(
            (shift - tally).abs() < 1e-7,
            "replica {k}: the transmutation's capture moved by {shift:e}, the transport's by \
             {tally:e}"
        );
        moved = moved.max(shift.abs());
    }
    assert!(
        moved > 1e-3,
        "the replicas move capture, by up to {moved:e}"
    );
}

/// The Monte Carlo noise in the replicas' shifts is measured from the even
/// half of the histories and removed from every sigma: the reported sigma is
/// the replicas' spread less that noise in quadrature, for a density and for
/// a derived quantity alike, and the report says so.
#[test]
fn the_replicas_monte_carlo_noise_is_removed() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    let results = transmute(&dir, &chain, Some(&request()));
    let info = &results.uncertainty_info[&1];
    assert!(info.noise_removed);
    assert_eq!(results.uncertainty[&1].noise_scale(), Some(1.0));

    let nominal = results.get_nuclide_density(1, "W187", 1).expect("W187");
    let sigma = results
        .get_nuclide_uncertainty(1, "W187", 1)
        .expect("sigma");
    let (spread, noise) = results
        .get_nuclide_uncertainty_noise(1, "W187", 1)
        .expect("noise");
    let x = shifts(&results);
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let raw = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt();
    assert!((spread / nominal - raw).abs() < 1e-9 * raw.max(1e-12));
    assert!(noise > 0.0, "{PARTICLES} histories leave some noise");
    let left = (spread * spread - noise * noise).max(0.0).sqrt();
    assert!((sigma - left).abs() <= 1e-12 * spread);
    eprintln!(
        "W187: spread {:.4e}, noise {:.4e}, sigma {:.4e} (relative)",
        spread / nominal,
        noise / nominal,
        sigma / nominal
    );

    let activity = results
        .activity_uncertainty(1, 1, &chain)
        .expect("activity")
        .expect("uncertainty was asked for");
    let (Some(s), Some(raw), Some(n)) = (activity.std_dev, activity.replica_spread, activity.noise)
    else {
        panic!("activity carries its spread and noise: {activity:?}");
    };
    assert!((s * s - (raw * raw - n * n).max(0.0)).abs() <= 1e-9 * raw * raw);
    // At step 0 every replica is the initial composition, so nothing is
    // removed.
    let initial = results
        .activity_uncertainty(1, 0, &chain)
        .expect("activity")
        .expect("uncertainty was asked for");
    assert_eq!(initial.noise, None);
}
