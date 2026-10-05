//! Resonance-parameter covariance (MF=32) through to transport.
//!
//! The published fixtures predate the converter writing MF=32's contribution,
//! so this reconverts ENDF/B-VIII.1 Pb208's `covariance.arrow` from a local
//! tape into a copy of the fixture and runs the replica weights on it. Pb208
//! puts its resolved-range uncertainty below 1.5 MeV in MF=32. Ignored unless
//! `ENDF_TAPES` points at the libraries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc::xs_perturbation::perturbed_material;
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
use yani_transmute::covariance_fold::transport_fields;
use yani_transmute::covariance_sample::Sampler;

const SEED: u64 = 11;
const RESOLVED_TOP: f64 = 1.5e6;

/// A copy of the Pb208 fixture in `dir` with its `covariance.arrow` reconverted
/// from the tape under `ENDF_TAPES`.
fn reconverted(dir: &Path) -> Option<PathBuf> {
    let tapes = PathBuf::from(std::env::var_os("ENDF_TAPES")?);
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Pb208.arrow");
    let out = dir.join("Pb208.arrow");
    std::fs::create_dir_all(&out).unwrap();
    for entry in std::fs::read_dir(&fixture).unwrap() {
        let p = entry.unwrap().path();
        std::fs::copy(&p, out.join(p.file_name().unwrap())).unwrap();
    }
    let tape = endf::material::Material::from_file(
        tapes.join("endfb-viii.1-endf/neutrons-version.VIII.1/n-082_Pb_208.endf"),
    )
    .unwrap();
    yamc_convert::covariance::write_covariance(&tape, &out).unwrap();
    Some(out)
}

fn lead(dir: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Pb208".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(11.35),
    )
    .unwrap();
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Pb208".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .unwrap();
    m.ensure_covariance_loaded().unwrap();
    m
}

/// The relative cells of `reaction` whose upper edge is inside the resolved
/// range.
fn resolved_cells(m: &Material, reaction: i32) -> usize {
    let (fields, _) = transport_fields(m);
    fields["Pb208"].field.as_ref().map_or(0, |f| {
        f.relative_cells
            .iter()
            .filter(|c| c.mt == reaction && c.hi <= RESOLVED_TOP)
            .count()
    })
}

fn tally(score: &str) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.scores = vec![score.parse::<Score>().unwrap()];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

fn run(m: Material, seed: u64, data: Option<TransportDataUncertainty>) -> Vec<Arc<Tally>> {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 15.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let cell = Cell::new(
        Some(1),
        Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere))),
        None,
        Some(0),
    );
    let geometry = Geometry::new(vec![cell], vec![Arc::new(m)]).unwrap();
    let tallies: Vec<Arc<Tally>> = ["flux", "elastic", "(n,gamma)"]
        .iter()
        .map(|s| tally(s))
        .collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![5.0e5], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(
        geometry,
        vec![source],
        tallies.iter().map(Arc::clone).collect(),
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(20_000),
            seed,
            threads: Some(1),
            data_uncertainty: data,
            ..Default::default()
        })
        .unwrap();
    tallies
}

fn replica(t: &Arc<Tally>, k: usize) -> (f64, f64) {
    let sums = t.get_replica_sums().unwrap();
    let h = t.finalize().n_histories as f64;
    let (s1, s2) = (sums.sums[k], sums.sums[sums.replicas + k]);
    let mean = s1 / h;
    ((mean), ((s2 - s1 * s1 / h) / (h - 1.0) / h).sqrt())
}

/// Pb208's resonance-parameter covariance reaches the fold as cells inside
/// the resolved range, and each replica's flux, elastic and capture there
/// match a rerun of its draw; the nuclear-data sigma it adds is reported.
#[test]
#[ignore = "reconverts from a local tape; set ENDF_TAPES and run with --ignored"]
fn resonance_parameter_covariance_reaches_transport() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(dir) = reconverted(tmp.path()) else {
        return;
    };
    let published = lead(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Pb208.arrow"));
    let m = lead(&dir);
    let (before, after) = (resolved_cells(&published, 2), resolved_cells(&m, 2));
    eprintln!("elastic cells in the resolved range: published {before}, reconverted {after}");
    assert!(
        after > before + 50,
        "MF=32's per-resonance cells reach the field"
    );
    assert!(resolved_cells(&m, 102) > resolved_cells(&published, 102));

    let replicas = 3;
    let data = Some(TransportDataUncertainty {
        seed: SEED,
        replicas,
    });
    let weighted = run(m.clone(), 21, data);
    let (fields, _) = transport_fields(&m);
    let cells = fields
        .iter()
        .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
        .collect();
    for k in 0..replicas {
        let draw = Sampler::new(&cells, &[]).draw(SEED, k as u64);
        let rerun = run(
            perturbed_material(&m, &fields, &draw).unwrap().0,
            1000 + k as u64,
            None,
        );
        for (i, name) in ["flux", "elastic", "capture"].iter().enumerate() {
            let (x, sx) = replica(&weighted[i], k);
            let (y, sy) = (rerun[i].get_mean()[0], rerun[i].get_std_dev()[0]);
            let z = (x - y) / (sx * sx + sy * sy).sqrt();
            eprintln!(
                "replica {k} {name}: weighted {x:e} ± {sx:e}, rerun {y:e} ± {sy:e} (z = {z:.2})"
            );
            assert!(z.abs() < 4.0, "replica {k} {name}: z = {z:.2}");
        }
    }
    let with_published = run(published, 21, data);
    for (i, name) in ["flux", "elastic", "capture"].iter().enumerate() {
        let sd = |t: &Arc<Tally>| t.finalize().nuclear_data_standard_deviation().unwrap()[0];
        let mean = weighted[i].get_mean()[0];
        eprintln!(
            "{name}: nuclear-data sigma {:.2}% with MF=32, {:.2}% with MF=33 alone",
            100.0 * sd(&weighted[i]) / mean,
            100.0 * sd(&with_published[i]) / mean
        );
    }
}
