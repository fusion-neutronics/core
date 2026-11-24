//! Per-tet density factors on a mesh geometry.
//!
//! On `two_region_tets.arrow` (a unit cube split at x = 0.5 into two tet
//! volumes), factors `a` on every tet of one volume and `b` on the other must
//! reproduce the same model with the two materials at `a·ρ` and `b·ρ`: flux and
//! heating per material, under both the delta-tracking flight estimator and
//! the collision estimator. Under pure Woodcock both runs share a majorant, so
//! they must agree history for history; factors of 1 likewise reproduce the
//! model without factors.
//!
//! The iron is contrived dense (40 g/cm3, about 0.8 cm mean free path at
//! 14 MeV) so the flux itself, not only the reaction rates, responds to the
//! density of a 1 cm cube.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;

use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::Tally;
use yamc_tallies::{Estimator, MaterialFilter};

const MESH: &str = "../yamt/tests/data/two_region_tets.arrow";
const RHO: f64 = 40.0;
const N: usize = 40_000;

fn iron(id: u32, density: f64, path: &str) -> Arc<Material> {
    let mut m = Material::new(
        HashMap::from([("Fe56".to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(density),
    )
    .unwrap();
    m.set_material_id(id);
    m.set_temperature("294");
    m.read_nuclear_data(
        &HashMap::from([("Fe56".to_string(), path.to_string())]),
        None,
    )
    .unwrap();
    Arc::new(m)
}

fn tally(estimator: Estimator) -> Arc<Tally> {
    let mut t = Tally::new();
    t.estimator = estimator;
    t.scores = vec!["flux".parse().unwrap(), "heating".parse().unwrap()];
    t.filters = vec![Filter::Material(MaterialFilter {
        material_ids: vec![1, 2],
    })];
    Arc::new(t)
}

/// The two-region cube with material densities `fuel` and `moderator`, under
/// `mode`, with a track-length and a collision tally.
fn model(path: &str, fuel: f64, moderator: f64, mode: TrackingMode) -> Model {
    let materials = HashMap::from([
        ("fuel".to_string(), iron(1, fuel, path)),
        ("moderator".to_string(), iron(2, moderator, path)),
    ]);
    let mesh =
        yamc::geometry::mesh::MeshGeometry::from_arrow(std::path::Path::new(MESH), &materials)
            .unwrap();
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.25, 0.5, 0.5])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut m = Model::new_with_mesh(
        mesh,
        vec![source],
        vec![tally(Estimator::TrackLength), tally(Estimator::Collision)],
    );
    m.verbose = Verbose::silent();
    m.tracking_mode = mode;
    m
}

/// Factor `fuel` on every tet of the volume holding x < 0.5, `moderator` on
/// the other.
fn factors(fuel: f64, moderator: f64) -> Vec<f64> {
    let mesh = yamt::MeshGeometry::from_arrow(std::path::Path::new(MESH)).unwrap();
    let fuel_vol = mesh.find_volume([0.25, 0.5, 0.5]);
    mesh.topology
        .tet_volume_ids
        .iter()
        .map(|v| {
            if *v == Some(fuel_vol) {
                fuel
            } else {
                moderator
            }
        })
        .collect()
}

/// (mean, std) per bin of both tallies.
fn run(model: &mut Model) -> Vec<(f64, f64)> {
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(N),
            seed: 3,
            ..Default::default()
        })
        .unwrap();
    model
        .tallies
        .iter()
        .flat_map(|t| t.get_mean().into_iter().zip(t.get_std_dev()))
        .collect()
}

fn z(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).abs() / (a.1 * a.1 + b.1 * b.1).sqrt()
}

/// Bins 2, 3 (track length) and 6, 7 (collision) are heating, which follows
/// density directly; flux in the denser half barely moves in a 1 cm cube.
const HEATING_BINS: [usize; 4] = [2, 3, 6, 7];

#[test]
fn factors_match_the_same_densities_set_on_the_materials() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let (a, b) = (0.5, 1.5);
    let unscaled = run(&mut model(&path, RHO, RHO, TrackingMode::Woodcock));
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let mut scaled = model(&path, RHO, RHO, mode);
        scaled.set_tet_density_factors(Some(factors(a, b))).unwrap();
        let scaled = run(&mut scaled);
        let reference = run(&mut model(&path, a * RHO, b * RHO, mode));
        for (i, (s, r)) in scaled.iter().zip(&reference).enumerate() {
            if mode == TrackingMode::Woodcock {
                // Same majorant (b·Σ) and the same acceptance probabilities,
                // so the same histories: equal up to the reduction's
                // thread-order rounding.
                assert!(
                    (s.0 - r.0).abs() <= 1e-12 * r.0.abs(),
                    "bin {i}: {s:?} != {r:?}"
                );
            } else {
                // Hybrid may surface-step the reference's cells but never a
                // cell with factors, so the histories differ.
                assert!(z(*s, *r) < 4.5, "hybrid bin {i}: {s:?} vs reference {r:?}");
            }
        }
        for i in HEATING_BINS {
            assert!(
                z(scaled[i], unscaled[i]) > 8.0,
                "{mode:?} bin {i}: the factors must visibly change the heating"
            );
        }
    }
}

#[test]
fn factors_of_one_reproduce_no_factors() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let mut with = model(&path, RHO, RHO, TrackingMode::Woodcock);
    with.set_tet_density_factors(Some(factors(1.0, 1.0)))
        .unwrap();
    let with = run(&mut with);
    let without = run(&mut model(&path, RHO, RHO, TrackingMode::Woodcock));
    for (w, o) in with.iter().zip(&without) {
        assert!((w.0 - o.0).abs() <= 1e-12 * w.0.abs(), "{w:?} != {o:?}");
    }
}

#[test]
fn bad_factors_and_surface_tracking_are_errors() {
    let Some(path) = yamc_test_cache::nuclide("Fe56") else {
        eprintln!("skip: Fe56 fixture missing");
        return;
    };
    let mut m = model(&path, RHO, RHO, TrackingMode::Surface);
    let n = factors(1.0, 1.0).len();
    assert!(m.set_tet_density_factors(Some(vec![1.0; n - 1])).is_err());
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut f = vec![1.0; n];
        f[n / 2] = bad;
        assert!(m.set_tet_density_factors(Some(f)).is_err());
    }
    m.set_tet_density_factors(Some(vec![1.0; n])).unwrap();
    let err = m
        .simulate_transport(&TransportSettings {
            total_particles: Some(10),
            ..Default::default()
        })
        .unwrap_err();
    assert!(err.contains("hybrid"), "{err}");
    m.set_tet_density_factors(None).unwrap();
    m.simulate_transport(&TransportSettings {
        total_particles: Some(10),
        ..Default::default()
    })
    .unwrap();
}
