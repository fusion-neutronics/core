//! Correlated nuclear-data replica weights, checked against reruns.
//!
//! A replica's tally under the weights must be, in expectation, the tally of
//! a run whose cross sections are that replica's draw. A rerun with the draw
//! applied to the data (`xs_perturbation`) is that run, exact by
//! construction, so every replica is compared with one. On ENDF/B-VIII.1
//! Fe56, whose covariance covers elastic, capture and the total inelastic.
//! Self-skips where the fixture carries no covariance.

use std::collections::HashMap;
use std::path::PathBuf;
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

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Fe56.arrow");
    dir.join("covariance.arrow").is_file().then_some(dir)
}

fn iron(dir: &std::path::Path) -> Material {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), dir.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Fe56");
    m.ensure_covariance_loaded().expect("read covariance");
    m
}

/// Fe56 with an equal share of Fe57, whose fixture carries no covariance:
/// Fe57 perturbs nothing but is still part of every reaction rate.
fn mixed_iron(dir: &std::path::Path) -> Option<Material> {
    let fe57 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Fe57.arrow");
    if !fe57.join("reactions.arrow").is_file() || fe57.join("covariance.arrow").is_file() {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 0.5), ("Fe57".to_string(), 0.5)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([
            ("Fe56".to_string(), dir.to_string_lossy().into_owned()),
            ("Fe57".to_string(), fe57.to_string_lossy().into_owned()),
        ]),
        None,
    )
    .expect("read iron");
    m.ensure_covariance_loaded().expect("read covariance");
    Some(m)
}

fn tally(score: &str) -> Arc<Tally> {
    let mut t = Tally::new();
    t.filters.push(Filter::Cell(CellFilter::from_id(1)));
    t.scores = vec![score.parse::<Score>().expect("score")];
    t.estimator = Estimator::TrackLength;
    t.initialize_batches(1);
    Arc::new(t)
}

/// One run of a sphere of `material`: flux and capture tallies, returned
/// with the model so a caller can inspect them.
fn run(
    material: Material,
    radius: f64,
    energy: f64,
    particles: usize,
    seed: u64,
    data: Option<TransportDataUncertainty>,
) -> Vec<Arc<Tally>> {
    run_scores(
        material,
        radius,
        energy,
        particles,
        seed,
        data,
        &["flux", "(n,gamma)"],
    )
}

/// [`run`] with the tallies' scores given.
fn run_scores(
    material: Material,
    radius: f64,
    energy: f64,
    particles: usize,
    seed: u64,
    data: Option<TransportDataUncertainty>,
    scores: &[&str],
) -> Vec<Arc<Tally>> {
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
    let tallies: Vec<Arc<Tally>> = scores.iter().map(|s| tally(s)).collect();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![energy], vec![1.0]).unwrap()),
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
            total_particles: Some(particles),
            seed,
            // One thread, so which worker folds which history, and with it
            // the last bit of a Welford sum, is the same run to run.
            threads: Some(1),
            data_uncertainty: data,
            ..Default::default()
        })
        .expect("transport");
    tallies
}

/// Replica `k`'s mean and its Monte Carlo standard error, from a replica run.
fn replica(t: &Arc<Tally>, k: usize) -> (f64, f64) {
    let sums = t.get_replica_sums().expect("replica sums");
    let h = t.finalize().n_histories as f64;
    let (s1, s2) = (sums.sums[k], sums.sums[sums.replicas + k]);
    let mean = s1 / h;
    let var = (s2 - s1 * s1 / h) / (h - 1.0);
    (mean, (var / h).sqrt())
}

/// The material with replica `k`'s draw applied to its data.
fn rerun_material(m: &Material, k: u64) -> Material {
    let (fields, _) = transport_fields(m);
    let cells = fields
        .iter()
        .filter_map(|(n, t)| t.field.clone().map(|f| (n.clone(), f)))
        .collect();
    let draw = Sampler::new(&cells, &[]).draw(SEED, k);
    perturbed_material(m, &fields, &draw).expect("perturb").0
}

#[test]
fn the_nominal_tally_is_unchanged_by_the_replicas() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let plain = run(m.clone(), 5.0, 14.1e6, 2_000, 3, None);
    let with = run(
        m,
        5.0,
        14.1e6,
        2_000,
        3,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas: 4,
        }),
    );
    for (a, b) in plain.iter().zip(&with) {
        assert_eq!(a.get_mean(), b.get_mean(), "the nominal mean moved");
        assert_eq!(a.get_std_dev(), b.get_std_dev(), "the nominal error moved");
        assert!(a.get_replica_sums().is_none());
        assert!(b.get_replica_sums().is_some());
    }
}

/// On a sphere thin enough that most neutrons leave uncollided, the same
/// seed gives nearly the same histories in a rerun, so each replica's
/// capture rate must match its rerun closely: this checks the reaction-rate
/// factor `Σ'_x / Σ_x` cell by cell.
#[test]
fn each_replica_matches_its_rerun_on_a_thin_target() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let replicas = 4;
    let weighted = run(
        m.clone(),
        0.2,
        0.0253,
        20_000,
        7,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
    );
    let nominal = weighted[1].get_mean()[0];
    for k in 0..replicas {
        let (x, _) = replica(&weighted[1], k);
        let y = run(rerun_material(&m, k as u64), 0.2, 0.0253, 20_000, 7, None)[1].get_mean()[0];
        assert!(
            (x / y - 1.0).abs() < 2e-3,
            "replica {k}: weighted capture {x:e}, rerun {y:e} (nominal {nominal:e})"
        );
    }
}

/// A nuclide with no covariance perturbs nothing, but its capture is part
/// of the material's: each replica's capture rate on a mixed target must
/// still match its rerun, the uncovered share diluting the change.
#[test]
fn a_nuclide_without_covariance_still_dilutes_the_rate_it_shares() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let Some(m) = mixed_iron(&dir) else {
        eprintln!("skipping: needs an Fe57 fixture without covariance");
        return;
    };
    let replicas = 3;
    let weighted = run(
        m.clone(),
        0.2,
        0.0253,
        20_000,
        7,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
    );
    for k in 0..replicas {
        let (x, _) = replica(&weighted[1], k);
        let y = run(rerun_material(&m, k as u64), 0.2, 0.0253, 20_000, 7, None)[1].get_mean()[0];
        assert!(
            (x / y - 1.0).abs() < 2e-3,
            "replica {k}: weighted capture {x:e}, rerun {y:e}"
        );
    }
}

/// On a thick sphere the flux responds to the cross sections through every
/// flight and collision, and each replica's flux and capture rate must match
/// an independent rerun within Monte Carlo error: this checks the flight and
/// collision weights.
#[test]
fn each_replica_matches_its_rerun_through_a_thick_sphere() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let replicas = 3;
    let particles = 6_000;
    let weighted = run(
        m.clone(),
        8.0,
        14.1e6,
        particles,
        21,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
    );
    for k in 0..replicas {
        let rerun = run(
            rerun_material(&m, k as u64),
            8.0,
            14.1e6,
            particles,
            1000 + k as u64,
            None,
        );
        for (i, name) in ["flux", "capture"].iter().enumerate() {
            let (x, sx) = replica(&weighted[i], k);
            let y = rerun[i].get_mean()[0];
            let sy = rerun[i].get_std_dev()[0];
            let z = (x - y) / (sx * sx + sy * sy).sqrt();
            assert!(
                z.abs() < 4.0,
                "replica {k} {name}: weighted {x:e} ± {sx:e}, rerun {y:e} ± {sy:e} (z = {z:.2})"
            );
        }
    }
}

#[test]
fn modes_the_weights_do_not_carry_yet_are_refused() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: Fe56 fixture carries no covariance");
        return;
    };
    let m = iron(&dir);
    let sphere = Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 1.0,
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
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1.0e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![tally("flux")]);
    model.verbose = Verbose::silent();
    model.tracking_mode = TrackingMode::Woodcock;
    let err = model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10),
            data_uncertainty: Some(TransportDataUncertainty {
                seed: 1,
                replicas: 4,
            }),
            ..Default::default()
        })
        .expect_err("delta tracking is refused");
    assert!(err.contains("delta tracking"), "{err}");

    model.tracking_mode = TrackingMode::Surface;
    let mut mesh = Tally::new();
    mesh.filters
        .push(Filter::Mesh(yamc_tallies::filter::mesh::MeshFilter::new(
            yamc_tallies::mesh::RegularRectangularMesh::new([-1.0; 3], [1.0; 3], [2, 2, 2]),
        )));
    mesh.scores = vec!["flux".parse::<Score>().unwrap()];
    mesh.initialize_batches(1);
    model.tallies = vec![Arc::new(mesh)];
    let err = model
        .simulate_transport(&TransportSettings {
            total_particles: Some(10),
            data_uncertainty: Some(TransportDataUncertainty {
                seed: 1,
                replicas: 4,
            }),
            ..Default::default()
        })
        .expect_err("a mesh tally is refused");
    assert!(err.contains("mesh tallies"), "{err}");
}

/// A model whose data carries no covariance anywhere is refused rather than
/// reported with a nuclear-data sigma of zero.
#[test]
fn a_model_with_no_covariance_is_refused_not_reported_exact() {
    let fe57 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/Fe57.arrow");
    if !fe57.join("reactions.arrow").is_file() || fe57.join("covariance.arrow").is_file() {
        eprintln!("skipping: needs an Fe57 fixture without covariance");
        return;
    }
    let mut m = Material::new(
        HashMap::from([("Fe57".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.87),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe57".to_string(), fe57.to_string_lossy().into_owned())]),
        None,
    )
    .expect("read Fe57");
    let result = std::panic::catch_unwind(|| {
        run(
            m,
            1.0,
            1.0e6,
            10,
            1,
            Some(TransportDataUncertainty {
                seed: 1,
                replicas: 4,
            }),
        )
    });
    let message = result
        .expect_err("a model with no covariance is refused")
        .downcast::<String>()
        .map(|s| *s)
        .unwrap_or_default();
    assert!(message.contains("no nuclide"), "{message}");
}

/// Natural lithium, from the fixtures.
fn lithium() -> Option<Material> {
    let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let (li6, li7) = (tests.join("Li6.arrow"), tests.join("Li7.arrow"));
    if !li6.join("covariance.arrow").is_file() || !li7.join("reactions.arrow").is_file() {
        return None;
    }
    let mut m = Material::new(
        HashMap::from([("Li6".to_string(), 0.0759), ("Li7".to_string(), 0.9241)]),
        "atom",
        "g/cm3",
        Some(0.5),
    )
    .expect("material");
    m.set_material_id(1);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([
            ("Li6".to_string(), li6.to_string_lossy().into_owned()),
            ("Li7".to_string(), li7.to_string_lossy().into_owned()),
        ]),
        None,
    )
    .expect("read lithium");
    m.ensure_covariance_loaded().expect("covariance");
    Some(m)
}

/// Tritium production on a thin lithium target matches each replica's rerun:
/// the production score's own cross-section change is carried, through the
/// reactions that emit tritons.
#[test]
fn tritium_production_matches_its_rerun_on_a_thin_lithium_target() {
    let Some(m) = lithium() else {
        eprintln!("skipping: lithium fixtures missing or without covariance");
        return;
    };
    let replicas = 3;
    let weighted = run_scores(
        m.clone(),
        0.2,
        0.0253,
        20_000,
        7,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
        &["H3-production"],
    );
    for k in 0..replicas {
        let (x, _) = replica(&weighted[0], k);
        let y = run_scores(
            rerun_material(&m, k as u64),
            0.2,
            0.0253,
            20_000,
            7,
            None,
            &["H3-production"],
        )[0]
        .get_mean()[0];
        assert!(
            (x / y - 1.0).abs() < 2e-3,
            "replica {k}: weighted tritium {x:e}, rerun {y:e}"
        );
    }
}

/// Through a thick lithium sphere at 14 MeV, where Li7's tritium and the
/// flux response both enter, each replica's tritium production matches an
/// independent rerun within Monte Carlo error.
#[test]
fn tritium_production_matches_its_rerun_through_a_thick_lithium_sphere() {
    let Some(m) = lithium() else {
        eprintln!("skipping: lithium fixtures missing or without covariance");
        return;
    };
    let replicas = 3;
    let particles = 6_000;
    let weighted = run_scores(
        m.clone(),
        30.0,
        14.1e6,
        particles,
        21,
        Some(TransportDataUncertainty {
            seed: SEED,
            replicas,
        }),
        &["H3-production"],
    );
    for k in 0..replicas {
        let rerun = run_scores(
            rerun_material(&m, k as u64),
            30.0,
            14.1e6,
            particles,
            1000 + k as u64,
            None,
            &["H3-production"],
        );
        let (x, sx) = replica(&weighted[0], k);
        let (y, sy) = (rerun[0].get_mean()[0], rerun[0].get_std_dev()[0]);
        let z = (x - y) / (sx * sx + sy * sy).sqrt();
        assert!(
            z.abs() < 4.0,
            "replica {k}: weighted {x:e} ± {sx:e}, rerun {y:e} ± {sy:e} (z = {z:.2})"
        );
    }
}
