//! A transport-coupled transmutation's replicas under delta tracking,
//! survival biasing and weight windows.
//!
//! The transmutation tally scores each replica with the same factors as the
//! transport's own tallies, so on one set of histories its replica `k`
//! capture rate over the nominal is the capture tally's replica `k` over its
//! nominal, to rounding, whatever game the transport plays. On the cached
//! ENDF/B-VIII.1 W186 directory with the converter's covariance and
//! resonance parameters, in a pure W186 sphere around a 1 keV source.
//! Self-skips when the nuclear-data fixtures are missing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc::variance_reduction::{SurvivalBiasing, VarianceReduction, WeightWindowBounds};
use yamc_materials::Material;
use yamc_particle::particle::ParticleType;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::cell::CellFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::score::Score;
use yamc_tallies::tally::Tally;
use yamc_tallies::Estimator;
use yani_transmute::uncertainty::{DataUncertainty, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep};

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 31;
const REPLICAS: usize = 8;
const PARTICLES: usize = 3000;
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

#[derive(Clone, Copy, Debug)]
enum Mode {
    Woodcock,
    Hybrid,
    SurvivalBiasing,
    WeightWindows,
}

/// Windows that roulette collisions near the centre up to a weight of 6 and
/// split them again further out.
fn weight_windows() -> WeightWindowBounds {
    let n = 4;
    let mesh = RegularRectangularMesh::new([-RADIUS; 3], [RADIUS; 3], [n, n, n]);
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let inner = [i, j, k].iter().all(|&x| x == 1 || x == 2);
                lower.push(if inner { 2.0 } else { 0.1 });
                upper.push(if inner { 10.0 } else { 0.5 });
            }
        }
    }
    WeightWindowBounds {
        mesh,
        particle: ParticleType::Neutron,
        energy_bins: None,
        lower_bounds: lower,
        upper_bounds: upper,
        survival_factor: 3.0,
        max_split: 10,
        weight_floor: 1e-38,
    }
}

/// A pure W186 sphere around a 1 keV point source, transported in `mode`.
fn model(dir: &Path, mode: Mode, tallies: Vec<Arc<Tally>>) -> Model {
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
    model.tracking_mode = match mode {
        Mode::Woodcock => TrackingMode::Woodcock,
        Mode::Hybrid => TrackingMode::Hybrid,
        _ => TrackingMode::Surface,
    };
    match mode {
        Mode::SurvivalBiasing => model
            .variance_reduction
            .push(VarianceReduction::SurvivalBiasing(
                SurvivalBiasing::default(),
            )),
        Mode::WeightWindows => {
            model
                .variance_reduction
                .push(VarianceReduction::WeightWindowBounds(Box::new(
                    weight_windows(),
                )))
        }
        _ => {}
    }
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

fn schedule() -> Schedule {
    Schedule::new(vec![ScheduleStep {
        rate: 1.0e10,
        dt: 1.0,
        is_pulse: true,
    }])
    .expect("schedule")
}

/// Under each transport game, the transmutation's replica `k` is the
/// transport capture tally's replica `k` on the same histories, and the
/// replicas move capture.
#[test]
fn each_replica_is_the_transports_replica_in_every_game() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let (Some(dir), Some(chain)) = (w186(tmp.path()), chain()) else {
        eprintln!("skipping: W186 or chain fixture not fetched");
        return;
    };
    let request = DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources: vec![UqSource::CrossSections],
        ..Default::default()
    };
    for mode in [
        Mode::Woodcock,
        Mode::Hybrid,
        Mode::SurvivalBiasing,
        Mode::WeightWindows,
    ] {
        let results = model(&dir, mode, vec![])
            .transmute(
                "independent",
                &schedule(),
                Arc::clone(&chain),
                Arc::new(yani::BranchTable::new()),
                Default::default(),
                &settings(),
                Some(&request),
                yani_transmute::DEFAULT_PRODUCT_THRESHOLD,
                None,
            )
            .expect("transmute");
        let nominal = results.get_nuclide_density(1, "W187", 1).expect("W187");
        let shifts: Vec<f64> = results
            .uncertainty_inventories(1, 1)
            .expect("replicas")
            .iter()
            .map(|i| i.get("W187").copied().unwrap_or(0.0) / nominal - 1.0)
            .collect();

        let mut capture = Tally::new();
        capture.filters.push(Filter::Cell(CellFilter::from_id(1)));
        capture.scores = vec!["(n,gamma)".parse::<Score>().expect("score")];
        capture.estimator = Estimator::TrackLength;
        capture.reset_accumulation();
        let capture = Arc::new(capture);
        model(&dir, mode, vec![Arc::clone(&capture)])
            .simulate_transport(&TransportSettings {
                data_uncertainty: Some(TransportDataUncertainty {
                    seed: SEED,
                    replicas: REPLICAS,
                }),
                ..settings()
            })
            .expect("transport");
        let sums = capture.get_replica_sums().expect("replica sums");
        let finalized = capture.finalize();
        let total = finalized.mean[0] * finalized.n_histories as f64;

        let mut moved = 0.0_f64;
        for (k, shift) in shifts.iter().enumerate() {
            let tally = sums.sums[k] / total - 1.0;
            eprintln!("{mode:?} replica {k}: transmutation {shift:+.6e}, capture {tally:+.6e}");
            assert!(
                (shift - tally).abs() < 1e-7,
                "{mode:?} replica {k}: transmutation {shift:e}, capture tally {tally:e}"
            );
            moved = moved.max(shift.abs());
        }
        assert!(moved > 1e-3, "{mode:?}: the replicas move capture");
    }
}
