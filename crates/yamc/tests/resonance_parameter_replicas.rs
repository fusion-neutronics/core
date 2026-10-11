//! Resonance parameters sampled exactly by the transport's replica weights.
//!
//! The fixture is the cached ENDF/B-VIII.1 W186 directory with the
//! `covariance.arrow` and `resonance_parameters.arrow` the real converter
//! writes from the committed MF=2 and MF=32 evaluation, which has no MF=33:
//! the resonance range's uncertainty is all there is. Each replica draws the
//! parameters, and its flux and capture through the resonances, under the
//! weights, must match a rerun of the same draw applied to the data
//! ([`perturbed_material`] with the replica's [`ResonanceChange`]). Self-skips
//! when the nuclear-data fixtures are missing.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportDataUncertainty, TransportSettings, Verbose};
use yamc::xs_perturbation::{
    perturbed_material, resonance_sampling, sampled_names, ResonanceChange,
};
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
use yani_transmute::resonance_rates::ResonanceMethod;

const W186_ENDF: &[u8] = include_bytes!("../../endf/fixtures/n-074_W_186_mf2_mf32.endf.xz");
const SEED: u64 = 31;
const SCORES: [&str; 2] = ["flux", "(n,gamma)"];

/// A copy of the cached W186 directory with the converter's
/// `covariance.arrow` and, when `parameters`, `resonance_parameters.arrow`,
/// or `None` when the fixtures are missing.
fn w186(tmp: &Path, parameters: bool) -> Option<PathBuf> {
    let cached = PathBuf::from(yamc_test_cache::nuclide("W186")?);
    let dir = tmp.join("W186.arrow");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for entry in std::fs::read_dir(&cached).expect("read cached W186") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name();
        if entry.path().is_file() && name != "resonance_parameters.arrow" {
            std::fs::copy(entry.path(), dir.join(name)).expect("copy section");
        }
    }
    let mut raw = Vec::new();
    lzma_rs::xz_decompress(&mut &W186_ENDF[..], &mut raw).expect("fixture decompresses");
    let text = String::from_utf8(raw).expect("ENDF is text");
    let evaluation = endf::Material::from_str(&text).expect("W186 parses");
    assert!(yamc_convert::covariance::write_covariance(&evaluation, &dir).expect("covariance"));
    if parameters {
        assert!(
            yamc_convert::resonance_parameters::write_resonance_parameters(&evaluation, &dir)
                .expect("resonance parameters")
        );
    }
    Some(dir)
}

fn tungsten(dir: &Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("W186".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(19.3),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("W186".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read W186");
    m.ensure_covariance_loaded().expect("covariance");
    m.ensure_resonance_parameters_loaded()
        .expect("resonance parameters");
    m
}

fn tally(score: &str) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.scores = vec![score.parse::<Score>().expect("score")];
    t.estimator = Estimator::TrackLength;
    t.reset_accumulation();
    Arc::new(t)
}

/// A sphere of `material` of `radius` cm with a point source at its centre,
/// emitting `lines` (eV) with equal probability.
fn model(material: Material, radius: f64, lines: &[f64]) -> (Model, Vec<Arc<Tally>>) {
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    };
    let region = Region::new_from_halfspace(HalfspaceType::Below(Arc::new(sphere)));
    let cell = Cell::new(Some(1), region, Some("c".into()), Some(0));
    let geometry = Geometry::new(vec![cell], vec![Arc::new(material)]).expect("geometry");
    let tallies: Vec<Arc<Tally>> = SCORES.iter().map(|s| tally(s)).collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(lines.to_vec(), vec![1.0; lines.len()]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(
        geometry,
        vec![source],
        tallies.iter().map(Arc::clone).collect(),
    );
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Surface;
    (model, tallies)
}

/// One single-threaded run of [`model`].
fn run(
    material: Material,
    (radius, lines): (f64, &[f64]),
    particles: usize,
    seed: u64,
    data: Option<TransportDataUncertainty>,
) -> Vec<Arc<Tally>> {
    let (mut model, tallies) = model(material, radius, lines);
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed,
            threads: Some(1),
            data_uncertainty: data,
            ..Default::default()
        })
        .expect("transport");
    tallies
}

/// Replica `k`'s mean and its Monte Carlo standard error.
fn replica(t: &Arc<Tally>, k: usize) -> (f64, f64) {
    let sums = t.get_replica_sums().expect("replica sums");
    let h = t.finalize().n_histories as f64;
    let (s1, s2) = (sums.sums[k], sums.sums[sums.replicas + k]);
    let mean = s1 / h;
    (mean, ((s2 - s1 * s1 / h) / (h - 1.0) / h).sqrt())
}

/// The material with replica `k`'s draw applied to its data: its MF=33 cells
/// as the replica weights draw them, without the first-order MF=32 rows of a
/// sampled nuclide, and its resonance parameters' change.
fn rerun_material(m: &Material, k: u64) -> Material {
    let sampling = resonance_sampling(m);
    let (fields, _) = transport_fields(m, &sampled_names(&sampling));
    let cells = fields
        .iter()
        .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
        .collect();
    let draw = Sampler::new(&cells, &[]).draw(SEED, k);
    let mut resonance = BTreeMap::new();
    for (name, sampled) in &sampling {
        let Ok(r) = sampled else { continue };
        let grid = m.nuclide_data[name].energy.as_ref().expect("grid")["294"].as_slice();
        let by_mt = r.changes(SEED, k, grid).expect("changes");
        resonance.insert(
            name.clone(),
            ResonanceChange {
                temperature: "294".to_string(),
                by_mt,
            },
        );
    }
    perturbed_material(m, &fields, &draw, &resonance)
        .expect("perturb")
        .0
}

/// The coverage report names W186's parameters as sampled, and a folder
/// without them as keeping the rows.
#[test]
fn the_coverage_report_says_which_parameters_are_sampled() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path(), true) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let (mut with, _) = model(tungsten(&dir), 1.0, &[30.0]);
    let report = with.data_uncertainty_coverage().expect("coverage");
    assert!(
        matches!(
            report.resonance_parameters.get("W186"),
            Some(ResonanceMethod::Sampled { .. })
        ),
        "{:?}",
        report.resonance_parameters
    );
    // Its only covariance is the rows the parameters replace.
    assert!(report.nuclides["W186"].perturbed.is_empty());

    let tmp_rows = tempfile::tempdir().expect("temp dir");
    let rows = w186(tmp_rows.path(), false).expect("fixtures");
    let (mut without, _) = model(tungsten(&rows), 1.0, &[30.0]);
    let report = without.data_uncertainty_coverage().expect("coverage");
    assert!(report.resonance_parameters.is_empty());
    assert!(report.nuclides["W186"].perturbed.contains_key(&102));
}

/// Lines on the flanks of W186's 18.8 eV resonance, where a drawn energy or
/// width moves the cross section by percents.
const FLANKS: [f64; 4] = [17.9, 18.4, 19.2, 19.8];

/// On a sphere thin enough that most neutrons leave uncollided, the same
/// seed gives nearly the same histories in a rerun, so each replica's
/// capture must match its rerun's closely: this checks the reaction-rate
/// factor with each replica's own rebuilt cross section.
#[test]
fn each_replica_matches_its_rerun_on_a_thin_target() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path(), true) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let m = tungsten(&dir);
    let replicas = 4;
    let geometry = (0.005, &FLANKS[..]);
    let weighted = run(
        m.clone(),
        geometry,
        20_000,
        7,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
    );
    let nominal = weighted[1].get_mean()[0];
    let mut moved = 0.0_f64;
    for k in 0..replicas {
        let (x, _) = replica(&weighted[1], k);
        let y = run(rerun_material(&m, k as u64), geometry, 20_000, 7, None)[1].get_mean()[0];
        moved = moved.max((y / nominal - 1.0).abs());
        eprintln!("replica {k}: weighted capture {x:e}, rerun {y:e}, nominal {nominal:e}");
        assert!(
            (x / y - 1.0).abs() < 2e-3,
            "replica {k}: weighted capture {x:e}, rerun {y:e} (nominal {nominal:e})"
        );
    }
    assert!(moved > 1e-2, "the parameters move the capture: {moved}");
}

/// Through a sphere the flux and the capture rate answer the drawn
/// parameters through every flight and collision, and every replica's, under
/// the weights, matches an independent rerun of its draw.
#[test]
fn each_replica_matches_its_rerun_through_a_thick_sphere() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let Some(dir) = w186(tmp.path(), true) else {
        eprintln!("skipping: nuclear-data fixtures missing");
        return;
    };
    let m = tungsten(&dir);
    let replicas = 4;
    let particles = 12_000;
    let geometry = (0.5, &FLANKS[..]);
    let weighted = run(
        m.clone(),
        geometry,
        particles,
        21,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
    );
    let mut moved = 0.0_f64;
    let mut failures = Vec::new();
    for k in 0..replicas {
        let rerun = run(
            rerun_material(&m, k as u64),
            geometry,
            particles,
            1000 + k as u64,
            None,
        );
        for (i, name) in SCORES.iter().enumerate() {
            let (x, sx) = replica(&weighted[i], k);
            let (y, sy) = (rerun[i].get_mean()[0], rerun[i].get_std_dev()[0]);
            let nominal = weighted[i].get_mean()[0];
            moved = moved.max((x / nominal - 1.0).abs());
            let z = (x - y) / (sx * sx + sy * sy).sqrt();
            eprintln!(
                "replica {k} {name}: weighted {x:e} ± {sx:e}, rerun {y:e} ± {sy:e}, \
                 nominal {nominal:e} (z = {z:.2})"
            );
            if z.abs() >= 4.0 {
                failures.push(format!("replica {k} {name}: z = {z:.2}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
    assert!(moved > 1e-3, "the parameters move the tallies: {moved}");
}
