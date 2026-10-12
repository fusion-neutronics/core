//! Nuclear-data replica weights under MPI give the serial answer: every rank
//! draws the same replicas from the same seed, transports its share of the
//! histories, and the per-replica sums of the tallies and of a
//! transport-coupled transmutation's tally are summed across ranks like the
//! nominal ones.
//!
//! `cargo test --features mpi` runs this binary as a single process, so the
//! test relaunches itself under `mpirun -np 2`. Each rank writes what it
//! got; the parent checks the ranks against each other and against a serial
//! run of the same histories in-process. On the cached ENDF/B-VIII.1 W186
//! directory with the covariance and resonance parameters the converter
//! writes from the committed MF=2 and MF=32 evaluation, around a 1 keV
//! source. Self-skips without the fixtures.
#![cfg(feature = "mpi")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TransportDataUncertainty, TransportSettings, Verbose};
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
use yani_transmute::uncertainty::{DataUncertainty, Source as UqSource};
use yani_transmute::{Schedule, ScheduleStep};

const TEST_NAME: &str = "replica_sums_are_the_serial_ones_on_every_rank";
const OUT_DIR_ENV: &str = "YAMC_MPI_REPLICA_OUT_DIR";
const RANKS: usize = 2;
const REPLICAS: usize = 4;
const SEED: u64 = 11;
const RADIUS: f64 = 5.0;
const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");

/// A copy of the cached W186 directory with the converter's covariance and
/// resonance parameters, under `tmp`.
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

fn chain() -> Option<Arc<HashMap<String, yani::ChainNuclide>>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../yani/tests/transmutation-endf-b8.1-sfr.arrow");
    path.exists()
        .then(|| Arc::new(yani::parse_chain_arrow(&path).expect("chain")))
}

/// A tungsten sphere around a 1 keV point source, transmutable, with
/// `tallies`.
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
    model
}

fn settings() -> TransportSettings {
    TransportSettings {
        total_particles: Some(2000),
        seed: 7,
        ..Default::default()
    }
}

/// The nominal means and replica sums of a flux and a capture tally, then
/// each replica's W187 over the nominal from a one-second pulse.
fn replica_results(dir: &Path, chain: &Arc<HashMap<String, yani::ChainNuclide>>) -> Vec<f64> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.scores = vec![
        "flux".parse::<Score>().expect("score"),
        "(n,gamma)".parse::<Score>().expect("score"),
    ];
    t.estimator = Estimator::TrackLength;
    t.reset_accumulation();
    let t = Arc::new(t);
    model(dir, vec![Arc::clone(&t)])
        .simulate_transport(&TransportSettings {
            data_uncertainty: Some(TransportDataUncertainty {
                seed: SEED,
                replicas: REPLICAS,
            }),
            ..settings()
        })
        .expect("transport");
    let mut out = t.get_mean();
    out.extend(t.get_replica_sums().expect("replica sums").sums);

    let request = DataUncertainty {
        seed: SEED,
        samples: Some(REPLICAS),
        sources: vec![UqSource::CrossSections],
        ..Default::default()
    };
    let schedule = Schedule::new(vec![ScheduleStep {
        rate: 1.0e10,
        dt: 1.0,
        is_pulse: true,
    }])
    .expect("schedule");
    let results = model(dir, vec![])
        .transmute(
            "independent",
            &schedule,
            Arc::clone(chain),
            Arc::new(yani::BranchTable::new()),
            Default::default(),
            &settings(),
            Some(&request),
        )
        .expect("transmute");
    let nominal = results.get_nuclide_density(1, "W187", 1).expect("W187");
    out.push(nominal);
    out.extend(
        results
            .uncertainty_inventories(1, 1)
            .expect("replicas")
            .iter()
            .map(|i| i.get("W187").copied().unwrap_or(0.0) / nominal),
    );
    out
}

fn write(path: &Path, values: &[f64]) {
    let text: Vec<String> = values
        .iter()
        .map(|b| format!("{:016x}", b.to_bits()))
        .collect();
    std::fs::write(path, text.join("\n")).expect("write rank values");
}

fn read(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .map(|l| f64::from_bits(u64::from_str_radix(l, 16).expect("hex f64 bits")))
        .collect()
}

fn under_mpi_launcher() -> bool {
    std::env::var("OMPI_COMM_WORLD_SIZE").is_ok() || std::env::var("PMI_SIZE").is_ok()
}

#[test]
fn replica_sums_are_the_serial_ones_on_every_rank() {
    // A fixed directory per launch, so every rank reads the same copy.
    let base = match std::env::var(OUT_DIR_ENV) {
        Ok(out) => PathBuf::from(out),
        Err(_) => std::env::temp_dir().join(format!("yamc_mpi_replicas_{}", std::process::id())),
    };
    std::fs::create_dir_all(&base).unwrap();
    // Each process its own copy of the data, named without touching MPI,
    // which the parent never initializes.
    let process = ["OMPI_COMM_WORLD_RANK", "PMI_RANK"]
        .iter()
        .find_map(|v| std::env::var(v).ok())
        .map_or("serial".to_string(), |r| format!("rank{r}"));
    let data = base.join(format!("data_{process}"));
    let (Some(dir), Some(chain)) = (w186(&data), chain()) else {
        eprintln!("skipping: W186 or chain fixture missing");
        return;
    };
    if under_mpi_launcher() {
        let values = replica_results(&dir, &chain);
        let rank = yamc::mpi_context::mpi_rank();
        write(&base.join(format!("rank{rank}.txt")), &values);
        yamc::mpi_context::mpi_finalize();
        return;
    }

    let out = base;
    let status = Command::new("mpirun")
        .args(["-np", &RANKS.to_string()])
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(OUT_DIR_ENV, &out)
        .status()
        .expect("run mpirun: the mpi feature needs an MPI launcher on PATH");
    assert!(status.success(), "mpirun -np {RANKS} failed: {status}");
    let per_rank: Vec<Vec<f64>> = (0..RANKS)
        .map(|r| read(&out.join(format!("rank{r}.txt"))))
        .collect();
    let serial = replica_results(&dir, &chain);
    std::fs::remove_dir_all(&out).ok();

    for (r, values) in per_rank.iter().enumerate().skip(1) {
        assert_eq!(values, &per_rank[0], "rank {r} differs from rank 0");
    }
    assert_eq!(per_rank[0].len(), serial.len());
    let mut moved = 0.0_f64;
    for (i, (m, s)) in per_rank[0].iter().zip(&serial).enumerate() {
        eprintln!("[{i}] MPI {m:.12e}, serial {s:.12e}");
        assert!(
            (m - s).abs() <= 1e-9 * s.abs().max(f64::MIN_POSITIVE),
            "[{i}]: MPI {m:e} against serial {s:e}"
        );
    }
    // The transmutation's replica ratios are the tail; they must move.
    for x in &serial[serial.len() - REPLICAS..] {
        moved = moved.max((x - 1.0).abs());
    }
    assert!(moved > 1e-3, "the replicas move W187, by up to {moved:e}");
}
