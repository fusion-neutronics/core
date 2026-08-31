//! Unstructured-mesh tallies: estimator/mode support matrix (issues
//! #354/#355).
//!
//! Both estimators are supported in every tracking mode. The collision
//! path resolves the containing tetrahedron with yamt's element-BVH
//! point query (#354); under woodcock/hybrid, flux-score track-length
//! unstructured tallies take the true segment-scoring path (#355) and
//! everything else scores through the delta-collision estimator with
//! per-tet attribution. Estimators and modes must all agree within
//! statistics.
#![cfg(feature = "mesh")]

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::UnstructuredMeshFilter;

fn build_li6_sphere() -> Geometry {
    let surf = Arc::new(Surface {
        surface_id: Some(1),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: 10.0,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    });
    let region = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            surf.clone(),
        )))),
    };
    let mut material = Material::new(
        HashMap::from([("Li6".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(0.534),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nuclide_map = HashMap::from([("Li6".to_string(), "tests/Li6.arrow".to_string())]);
    material.read_nuclear_data(&nuclide_map, None).unwrap();
    let cell = Cell::new(Some(1), region, Some("sphere".to_string()), Some(0));
    Geometry::new(vec![cell], vec![Arc::new(material)]).unwrap()
}

fn make_source() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    })
}

fn make_unstructured_tally(estimator: yamc_tallies::Estimator) -> Tally {
    let mesh = Arc::new(
        yamt::MeshGeometry::from_arrow(std::path::Path::new("../yamt/tests/data/cube.arrow"))
            .unwrap(),
    );
    let mut tally = Tally::new();
    tally.estimator = estimator;
    tally.filters = vec![Filter::UnstructuredMesh(UnstructuredMeshFilter::new(
        mesh, 0,
    ))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("umesh_flux".to_string());
    tally
}

fn run(estimator: yamc_tallies::Estimator, mode: TrackingMode) -> Result<(), String> {
    let mut model = Model::new(
        build_li6_sphere(),
        vec![make_source()],
        vec![Arc::new(make_unstructured_tally(estimator))],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model.simulate_transport(&TransportSettings {
        total_particles: Some(100),
        seed: 42,
        threads: Some(1),
        ..Default::default()
    })
}

#[test]
fn unstructured_track_length_surface_still_works() {
    run(yamc_tallies::Estimator::TrackLength, TrackingMode::Surface)
        .expect("track-length + surface is the supported combination");
}

#[test]
fn unstructured_collision_estimator_runs_in_every_mode() {
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        run(yamc_tallies::Estimator::Collision, mode)
            .unwrap_or_else(|e| panic!("collision + unstructured must run under {mode:?}: {e}"));
    }
}

#[test]
fn unstructured_ineligible_runs_via_collision_density() {
    // A cell filter makes the tally ineligible for segment scoring (the
    // segment spans unidentified cells), so it falls back to the
    // delta-collision path -- which now attributes per tet (#354).
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let mut tally = make_unstructured_tally(yamc_tallies::Estimator::TrackLength);
        tally
            .filters
            .push(Filter::Cell(yamc_tallies::CellFilter::from_id(1)));
        let mut model = Model::new(
            build_li6_sphere(),
            vec![make_source()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(100),
                seed: 42,
                threads: Some(1),
                ..Default::default()
            })
            .unwrap_or_else(|e| {
                panic!("ineligible unstructured tally must run under {mode:?}: {e}")
            });
    }
}

/// Histories for the estimator-comparison runs. 20 000 (the count this rig
/// used when issue #316 was found) left the comparison at 2 to 5 sigma of
/// power, so a +44% bias sat inside a 4 sigma bound on a lucky seed. 200 000
/// puts the two estimators about 0.5 sigma apart and would show that bias at
/// well over 20 sigma.
const ESTIMATOR_HISTORIES: usize = 200_000;

fn run_estimator_total(
    estimator: yamc_tallies::Estimator,
    mode: TrackingMode,
    seed: u64,
) -> (f64, f64) {
    let mut model = Model::new(
        build_li6_sphere(),
        vec![make_source()],
        vec![Arc::new(make_unstructured_tally(estimator))],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(ESTIMATOR_HISTORIES),
            seed,
            ..Default::default()
        })
        .unwrap();
    let t = &model.tallies[0];
    (t.total_mean(), t.total_std())
}

/// The collision estimator must read the same flux under every tracking mode:
/// the per-tet attribution (#354) and the woodcock / hybrid segment path (#355)
/// are the same physics seen through different tracking. This half of the
/// original combined test holds: over 12 seeds the largest mode-to-mode gap
/// measured was 1.8 sigma, well inside the 4 sigma bound.
#[test]
fn unstructured_collision_agrees_across_tracking_modes() {
    let (surface_mean, surface_std) = run_estimator_total(
        yamc_tallies::Estimator::Collision,
        TrackingMode::Surface,
        42,
    );
    assert!(
        surface_mean > 0.0,
        "surface collision flux zero: rig broken"
    );
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let (mean, std) = run_estimator_total(yamc_tallies::Estimator::Collision, mode, 42);
        assert!(mean > 0.0, "{mode:?} collision flux zero");
        let tol = 4.0 * (surface_std.powi(2) + std.powi(2)).sqrt();
        assert!(
            (surface_mean - mean).abs() < tol,
            "{mode:?} collision {mean:.6e} vs surface collision {surface_mean:.6e} \
             exceeds 4 sigma {tol:.2e}",
        );
    }
}

/// The two estimators must converge to the same flux: the collision
/// estimator's per-tet attribution (#354) has to agree with the segment walk
/// in every tracking mode.
///
/// This once read +44% (about 30 sigma at 1M histories) because the WALK was
/// broken, not the collision estimator: it mis-stepped through negatively
/// oriented tets, dropped every track that started outside the mesh, and
/// leaked across volume boundaries (issue #316). It passed anyway at the
/// original 20 000 histories on the pinned seed 42, which sat at 3.4 of the
/// 4 sigma bound while 17 of 36 (seed, mode) combinations were already over
/// it. Hence 200 000 histories and three seeds here: at that count the old
/// bias would show at over 20 sigma.
#[test]
fn unstructured_collision_matches_track_length() {
    // The two estimators converge to the same flux; the collision
    // estimator's per-tet attribution (#354) must agree with the
    // segment walk in every tracking mode. Issue #316: the segment walk
    // was the broken side (it dropped every track that started outside
    // the mesh, and mis-walked negatively oriented tets), which read 33%
    // low and made the collision estimator look 44% high.
    //
    // Run over several seeds: one pinned seed only samples one
    // realisation, and that is how the 44% bias survived here.
    for seed in [42_u64, 7, 12345] {
        let (tl_mean, tl_std) = run_estimator_total(
            yamc_tallies::Estimator::TrackLength,
            TrackingMode::Surface,
            seed,
        );
        assert!(tl_mean > 0.0, "TL surface flux zero: rig broken");
        for mode in [
            TrackingMode::Surface,
            TrackingMode::Woodcock,
            TrackingMode::Hybrid,
        ] {
            let (c_mean, c_std) =
                run_estimator_total(yamc_tallies::Estimator::Collision, mode, seed);
            assert!(c_mean > 0.0, "{mode:?} collision flux zero");
            let tol = 4.0 * (tl_std.powi(2) + c_std.powi(2)).sqrt();
            assert!(
                (tl_mean - c_mean).abs() < tol,
                "seed {seed} {mode:?} collision {c_mean:.6e} vs surface TL {tl_mean:.6e} \
                 exceeds 4 sigma {tol:.2e}",
            );
        }
    }
}

/// Issue #316: independent reference for the tet-mesh path.
///
/// `cube.arrow` tiles the unit cube exactly, so a structured 1x1x1 rectangular
/// mesh over `[0, 1]^3` integrates the flux over the identical region. Both
/// tet-mesh estimators must land on it. This is what pinned down *which*
/// estimator was wrong: before the fix the tet track-length total was 0.67 of
/// this reference while the tet collision total was within its own statistics
/// of it.
#[test]
fn unstructured_total_matches_the_structured_mesh_over_the_same_volume() {
    fn structured_tally(estimator: yamc_tallies::Estimator) -> Tally {
        let mesh = yamc_tallies::mesh::RegularRectangularMesh::new(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1, 1, 1],
        );
        let mut tally = Tally::new();
        tally.estimator = estimator;
        tally.filters = vec![Filter::Mesh(yamc_tallies::MeshFilter::new(mesh))];
        tally.scores = vec![Score::Flux(FluxScore)];
        tally.name = Some("smesh_flux".to_string());
        tally
    }

    fn total(tally: Tally) -> (f64, f64) {
        let mut model = Model::new(
            build_li6_sphere(),
            vec![make_source()],
            vec![Arc::new(tally)],
        );
        model.verbose = yamc::model::Verbose::silent();
        model
            .simulate_transport(&TransportSettings {
                total_particles: Some(ESTIMATOR_HISTORIES),
                seed: 42,
                ..Default::default()
            })
            .unwrap();
        (model.tallies[0].total_mean(), model.tallies[0].total_std())
    }

    for estimator in [
        yamc_tallies::Estimator::TrackLength,
        yamc_tallies::Estimator::Collision,
    ] {
        let (tet_mean, tet_std) = total(make_unstructured_tally(estimator));
        let (ref_mean, ref_std) = total(structured_tally(estimator));
        assert!(ref_mean > 0.0, "structured reference flux zero: rig broken");
        // Same histories, same seed, same region: the two binnings see the
        // same tracks, so this is a tight agreement, not a noise budget.
        let tol = 3.0 * (tet_std.powi(2) + ref_std.powi(2)).sqrt();
        assert!(
            (tet_mean - ref_mean).abs() < tol,
            "{estimator:?}: tet mesh {tet_mean:.6e} vs structured {ref_mean:.6e} \
             over the same [0,1]^3 (3 sigma {tol:.2e})",
        );
    }
}

fn run_flux_total(mode: TrackingMode) -> (f64, f64) {
    let mut model = Model::new(
        build_li6_sphere(),
        vec![make_source()],
        vec![Arc::new(make_unstructured_tally(
            yamc_tallies::Estimator::TrackLength,
        ))],
    );
    model.verbose = yamc::model::Verbose::silent();
    model.tracking_mode = mode;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(20_000),
            seed: 42,
            ..Default::default()
        })
        .unwrap();
    let t = &model.tallies[0];
    (t.total_mean(), t.total_std())
}

#[test]
fn unstructured_track_length_woodcock_matches_surface() {
    // Flux-score TL unstructured tallies score true track length along
    // each delta flight segment (issue #355): same statistical-match
    // property as the regular-mesh #351 tests.
    let (mean_s, std_s) = run_flux_total(TrackingMode::Surface);
    assert!(mean_s > 0.0, "surface unstructured flux zero: rig broken");
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let (mean_m, std_m) = run_flux_total(mode);
        assert!(mean_m > 0.0, "{mode:?} unstructured flux zero");
        let tol = 3.0 * (std_s.powi(2) + std_m.powi(2)).sqrt();
        assert!(
            (mean_s - mean_m).abs() < tol,
            "{mode:?} {mean_m:.6e} vs surface {mean_s:.6e} exceeds 3 sigma {tol:.2e}",
        );
    }
}

/// Regression for issue #290: a model carrying a tet-mesh tally must be able to
/// fingerprint. Every Python `simulate_transport` fingerprints the model for
/// `combine_results` run provenance, and `UnstructuredMeshFilter::serialize`
/// used to error, so the transport completed and then the results call threw,
/// making per-tet tallies unusable from Python.
#[test]
fn unstructured_tally_model_fingerprints() {
    let model = Model::new(
        build_li6_sphere(),
        vec![make_source()],
        vec![Arc::new(make_unstructured_tally(
            yamc_tallies::Estimator::TrackLength,
        ))],
    );
    let fp = model
        .fingerprint()
        .expect("a model with a tet-mesh tally must fingerprint");
    assert_eq!(fp.len(), 64, "sha256 hex digest expected, got {fp:?}");
    assert_eq!(
        fp,
        model.fingerprint().unwrap(),
        "the fingerprint must be deterministic"
    );
}

/// The tally subtree is excluded from the fingerprint, so a tet tally must not
/// change a model's identity: the same physics with and without it pools under
/// `combine_results`.
#[test]
fn unstructured_tally_does_not_change_model_identity() {
    let with_tally = Model::new(
        build_li6_sphere(),
        vec![make_source()],
        vec![Arc::new(make_unstructured_tally(
            yamc_tallies::Estimator::TrackLength,
        ))],
    );
    let without = Model::new(build_li6_sphere(), vec![make_source()], vec![]);
    assert_eq!(
        with_tally.fingerprint().unwrap(),
        without.fingerprint().unwrap()
    );
}
