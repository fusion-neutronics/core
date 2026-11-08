//! Surface-current tallies on void geometry, where the answer is exact.
//!
//! Every particle from a point source inside a sphere crosses it outwards
//! exactly once, so the current through it is 1 per source particle, and
//! from the centre every crossing is along the normal (cosine 1). Through a
//! plane, the half of an isotropic source heading towards it crosses, with
//! the sign of the side it crosses to. Delta tracking, a surface the
//! geometry does not have, and a `current` score without surfaces are
//! refused. No nuclear data is needed.

use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{Score, Tally};
use yamc_tallies::{CosineFilter, CurrentScore, Estimator, SurfaceFilter};

const PARTICLES: usize = 20_000;

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

fn sphere(radius: f64, id: usize, boundary: Option<BoundaryType>) -> Arc<Surface> {
    Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        radius,
        Some(id),
        boundary,
    ))
}

/// Void inside sphere 1 (r = 5) and between it and the vacuum sphere 2 (r = 10).
fn nested_spheres() -> Geometry {
    let inner = sphere(5.0, 1, None);
    let outer = sphere(10.0, 2, Some(BoundaryType::Vacuum));
    let cells = vec![
        Cell::new(Some(1), below(&inner), None, None),
        Cell::new(
            Some(2),
            above(&inner).intersection(&below(&outer)),
            None,
            None,
        ),
    ];
    Geometry::new(cells, vec![]).unwrap()
}

/// Void in a vacuum sphere (r = 10) cut in two by the plane x = 2 (id 3).
fn split_sphere() -> Geometry {
    let outer = sphere(10.0, 2, Some(BoundaryType::Vacuum));
    let plane = Arc::new(Surface::x_plane(2.0, Some(3), None));
    let cells = vec![
        Cell::new(
            Some(1),
            below(&outer).intersection(&below(&plane)),
            None,
            None,
        ),
        Cell::new(
            Some(2),
            below(&outer).intersection(&above(&plane)),
            None,
            None,
        ),
    ];
    Geometry::new(cells, vec![]).unwrap()
}

fn current_tally(surfaces: Vec<usize>, cosine_bins: Option<Vec<f64>>) -> Arc<Tally> {
    let mut tally = Tally::new();
    tally.scores = vec![Score::Current(CurrentScore)];
    tally.estimator = Estimator::Surface;
    tally.filters = vec![Filter::Surface(SurfaceFilter::from_ids(surfaces))];
    if let Some(edges) = cosine_bins {
        tally
            .filters
            .push(Filter::Cosine(CosineFilter::new(edges).unwrap()));
    }
    tally.validate().unwrap();
    Arc::new(tally)
}

fn run(
    geometry: Geometry,
    source: [f64; 3],
    tallies: Vec<Arc<Tally>>,
    mode: TrackingMode,
) -> Result<Model, String> {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new(source)),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], tallies);
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.simulate_transport(&TransportSettings {
        total_particles: Some(PARTICLES),
        seed: 1,
        threads: Some(4),
        ..Default::default()
    })?;
    Ok(model)
}

#[test]
fn every_particle_crosses_a_sphere_around_it_once_outwards() {
    let tally = current_tally(vec![1, 2], Some(vec![-1.0, 0.0, 0.5, 1.0]));
    let model = run(
        nested_spheres(),
        [0.0, 0.0, 0.0],
        vec![tally],
        TrackingMode::Surface,
    )
    .unwrap();
    let mean = model.tallies[0].get_mean();
    // score -> surface (2) -> cosine (3): from the centre every crossing of
    // either sphere is along its outward normal, cosine 1, the last bin.
    assert_eq!(mean.len(), 6);
    for surface in 0..2 {
        assert!((mean[surface * 3 + 2] - 1.0).abs() < 1e-12, "{mean:?}");
        assert_eq!(mean[surface * 3], 0.0);
        assert_eq!(mean[surface * 3 + 1], 0.0);
    }
}

#[test]
fn an_off_centre_source_still_gives_a_current_of_one_with_no_inward_crossings() {
    let tally = current_tally(vec![1], Some(vec![-1.0, 0.0, 1.0]));
    let model = run(
        nested_spheres(),
        [3.0, 1.0, 0.0],
        vec![tally],
        TrackingMode::Surface,
    )
    .unwrap();
    let mean = model.tallies[0].get_mean();
    assert_eq!(mean[0], 0.0, "no crossing of sphere 1 is inwards: {mean:?}");
    assert!((mean[1] - 1.0).abs() < 1e-12, "{mean:?}");
}

#[test]
fn the_current_through_a_plane_has_the_sign_of_the_side_crossed_to() {
    // A particle heading towards the plane x = 2 crosses it only if it meets
    // it inside the vacuum sphere (r = 10), where the plane is a disc of
    // radius sqrt(96). From a source on the x axis a distance d from the
    // plane, that takes a direction cosine towards it above
    // d / sqrt(96 + d^2), so (1 - d / sqrt(96 + d^2)) / 2 of the particles
    // cross. From the origin (d = 2) that is 0.4, towards x > 2, the plane's
    // positive side; from x = 5 (d = 3) it is 0.354, towards the negative side.
    let crossing = |d: f64| (1.0 - d / (96.0 + d * d).sqrt()) / 2.0;
    let from_x5 = -crossing(3.0);
    let sigma = (0.25 / PARTICLES as f64).sqrt();
    for (source, expected) in [([0.0, 0.0, 0.0], crossing(2.0)), ([5.0, 0.0, 0.0], from_x5)] {
        let model = run(
            split_sphere(),
            source,
            vec![current_tally(vec![3], None)],
            TrackingMode::Surface,
        )
        .unwrap();
        let mean = model.tallies[0].get_mean()[0];
        assert!(
            (mean - expected).abs() < 5.0 * sigma,
            "source {source:?}: {mean}, expected {expected}"
        );
    }
}

#[test]
fn delta_tracking_is_refused() {
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let err = run(
            nested_spheres(),
            [0.0; 3],
            vec![current_tally(vec![1], None)],
            mode,
        )
        .unwrap_err();
        assert!(err.contains("does not support surface tallies"), "{err}");
    }
}

#[test]
fn a_surface_the_geometry_does_not_have_is_refused() {
    let err = run(
        nested_spheres(),
        [0.0; 3],
        vec![current_tally(vec![99], None)],
        TrackingMode::Surface,
    )
    .unwrap_err();
    assert!(err.contains("surface id 99"), "{err}");
}

#[test]
fn a_current_score_without_surfaces_is_refused() {
    let mut tally = Tally::new();
    tally.scores = vec![Score::Current(CurrentScore)];
    let err = tally.validate().unwrap_err();
    assert!(err.contains("pass surfaces="), "{err}");
}
