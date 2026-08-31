//! Woodcock flights terminate at true vacuum exits (issue #360).
//!
//! Delta tracking samples flights with no boundary checks, so before
//! the fix a flight could tunnel through a vacuum boundary, cross
//! out-of-geometry space, and land in a disjoint body that surface
//! tracking could never reach: the modes disagreed on MEANS, the gap
//! attenuated as exp(-sigma_maj*d), and the #351 mesh scorer deposited
//! into gap voxels. `woodcock_flight_exit` now checks each flight
//! against the geometry's precomputed vacuum surfaces and kills at the
//! first crossing whose far side is outside the defined geometry.
//!
//! Coverage:
//! - Repro matrix {woodcock, hybrid} x {track-length, collision}: gap
//!   and body-B mesh bins exactly zero, in-body bins and totals match
//!   surface tracking, the partial boundary bin is clipped correctly.
//! - False-kill regression: a crossing of another body's infinite
//!   vacuum-plane EXTENSION (still inside the current body) must not
//!   kill the flight.
//! - Leak events: leaking woodcock flights die AT body A's boundary
//!   (x <= 3), not in the gap or at body B, with the live weight.

use std::collections::HashMap;
use std::sync::Arc;
use yamc::geo::{BoundaryType, Surface, SurfaceKind};
use yamc::geo::{HalfspaceType, Region, RegionExpr};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings};
use yamc::track::{HistorySelection, TrackEventType};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::source::{ParticleSource, Source, SourceEnergyDistribution};
use yamc_tallies::filter::mesh::MeshFilter;
use yamc_tallies::filter::Filter;
use yamc_tallies::mesh::RegularRectangularMesh;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::CellFilter;

fn be9_material(id: u32) -> Arc<Material> {
    let mut mat = Material::new(
        HashMap::from([("Be9".into(), 1.0)]),
        "atom",
        "g/cm3",
        Some(1.85),
    )
    .unwrap();
    mat.set_material_id(id);
    mat.set_temperature("294");
    let map = HashMap::from([("Be9".to_string(), "tests/Be9.arrow".to_string())]);
    mat.read_nuclear_data(&map, None).unwrap();
    Arc::new(mat)
}

fn vacuum_sphere(id: usize, x0: f64, radius: f64) -> Arc<Surface> {
    Arc::new(Surface {
        surface_id: Some(id),
        kind: SurfaceKind::Sphere {
            x0,
            y0: 0.0,
            z0: 0.0,
            radius,
        },
        boundary: BoundaryType::Vacuum,
        name: None,
    })
}

/// Two disjoint vacuum Be9 spheres r=3 at x=0 and x=20; the space
/// between belongs to no cell.
fn disjoint_spheres_geometry() -> Geometry {
    let s_a = vacuum_sphere(1, 0.0, 3.0);
    let s_b = vacuum_sphere(2, 20.0, 3.0);
    let region_a = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            s_a.clone(),
        )))),
    };
    let region_b = Region {
        expr: RegionExpr::Complement(Box::new(RegionExpr::Halfspace(HalfspaceType::Above(
            s_b.clone(),
        )))),
    };
    let cell_a = Cell::new(Some(1), region_a, Some("body_a".to_string()), Some(0));
    let cell_b = Cell::new(Some(2), region_b, Some("body_b".to_string()), Some(1));
    Geometry::new(vec![cell_a, cell_b], vec![be9_material(1), be9_material(2)]).unwrap()
}

fn beam_source_14mev() -> ParticleSource {
    ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, 0.0, 0.0]),
        ),
        angle: AngularDistribution::new_monodirectional(1.0, 0.0, 0.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    })
}

/// 24x1x1 mesh along the beam: bins are 1 cm wide, x in [-2, 22].
fn make_beam_mesh_tally(estimator: yamc_tallies::Estimator) -> Tally {
    let mesh = RegularRectangularMesh::new([-2.0, -0.5, -0.5], [22.0, 0.5, 0.5], [24, 1, 1]);
    let mut tally = Tally::new();
    tally.estimator = estimator;
    tally.filters = vec![Filter::Mesh(MeshFilter::new(mesh))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("beam_mesh".to_string());
    tally
}

fn run_beam(mode: TrackingMode, estimator: yamc_tallies::Estimator) -> (Vec<f64>, Vec<f64>) {
    let mut model = Model::new(
        disjoint_spheres_geometry(),
        vec![beam_source_14mev()],
        vec![Arc::new(make_beam_mesh_tally(estimator))],
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
    (t.get_mean(), t.get_std_dev())
}

/// Bin index for the 1-cm slab containing x (mesh starts at -2).
fn bin_for_x(x: f64) -> usize {
    ((x + 2.0).floor() as usize).min(23)
}

fn assert_no_tunneling(mode: TrackingMode, estimator: yamc_tallies::Estimator) {
    let (mean_s, std_s) = run_beam(TrackingMode::Surface, estimator);
    let (mean_m, std_m) = run_beam(mode, estimator);

    // Gap (x in (3,17)) and body-B (x > 17) bins: exactly zero under
    // both modes. Bins fully inside the gap start at x=4.
    for bin in bin_for_x(4.0)..24 {
        assert_eq!(
            mean_s[bin], 0.0,
            "surface scored past the body-A boundary (bin {bin}): rig broken"
        );
        assert_eq!(
            mean_m[bin], 0.0,
            "{mode:?}/{estimator:?} deposited past the body-A boundary \
             (bin {bin}): flights are tunneling through the vacuum exit",
        );
    }

    // The partial boundary bin [2,3] carries the clipped final segment;
    // it must match surface tracking within statistics.
    let b = bin_for_x(2.5);
    let tol = 5.0 * (std_s[b].powi(2) + std_m[b].powi(2)).sqrt();
    assert!(
        (mean_s[b] - mean_m[b]).abs() < tol,
        "boundary bin: {mode:?}/{estimator:?} {m:.6e} vs surface {s:.6e}",
        m = mean_m[b],
        s = mean_s[b],
    );

    // In-body converged bins and the total agree with surface tracking.
    let mut checked = 0;
    for i in 0..bin_for_x(3.0) {
        if mean_s[i] <= 0.0 || std_s[i] / mean_s[i] > 0.10 {
            continue;
        }
        checked += 1;
        let tol = 5.0 * (std_s[i].powi(2) + std_m[i].powi(2)).sqrt();
        assert!(
            (mean_s[i] - mean_m[i]).abs() < tol,
            "bin {i}: {mode:?}/{estimator:?} {m:.6e} vs surface {s:.6e}",
            m = mean_m[i],
            s = mean_s[i],
        );
    }
    assert!(checked >= 3, "only {checked} converged in-body bins");

    let total_s: f64 = mean_s.iter().sum();
    let total_m: f64 = mean_m.iter().sum();
    let var_s: f64 = std_s.iter().map(|s| s * s).sum();
    let var_m: f64 = std_m.iter().map(|s| s * s).sum();
    let tol = 3.0 * (var_s + var_m).sqrt();
    assert!(
        (total_s - total_m).abs() < tol,
        "total: {mode:?}/{estimator:?} {total_m:.6e} vs surface {total_s:.6e}",
    );
}

#[test]
fn woodcock_track_length_does_not_tunnel() {
    assert_no_tunneling(TrackingMode::Woodcock, yamc_tallies::Estimator::TrackLength);
}

#[test]
fn woodcock_collision_does_not_tunnel() {
    assert_no_tunneling(TrackingMode::Woodcock, yamc_tallies::Estimator::Collision);
}

#[test]
fn hybrid_track_length_does_not_tunnel() {
    assert_no_tunneling(TrackingMode::Hybrid, yamc_tallies::Estimator::TrackLength);
}

#[test]
fn hybrid_collision_does_not_tunnel() {
    assert_no_tunneling(TrackingMode::Hybrid, yamc_tallies::Estimator::Collision);
}

/// Two vacuum-bounded boxes where B's infinite y=+-1 plane extensions
/// slice through A: a +y flight inside A crosses them and must NOT be
/// killed there (the far side is still inside A); it dies at A's y=6
/// face.
fn disjoint_boxes_geometry() -> Geometry {
    fn plane(a: f64, b: f64, c: f64, d: f64, id: usize) -> Arc<Surface> {
        Arc::new(Surface {
            surface_id: Some(id),
            kind: SurfaceKind::Plane { a, b, c, d },
            boundary: BoundaryType::Vacuum,
            name: None,
        })
    }
    fn boxed(id_base: usize, x: (f64, f64), y: (f64, f64), z: (f64, f64)) -> Cell {
        let expr = RegionExpr::Intersection(
            Box::new(RegionExpr::Intersection(
                Box::new(RegionExpr::Intersection(
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Above(plane(
                        1.0, 0.0, 0.0, x.0, id_base,
                    )))),
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Below(plane(
                        1.0,
                        0.0,
                        0.0,
                        x.1,
                        id_base + 1,
                    )))),
                )),
                Box::new(RegionExpr::Intersection(
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Above(plane(
                        0.0,
                        1.0,
                        0.0,
                        y.0,
                        id_base + 2,
                    )))),
                    Box::new(RegionExpr::Halfspace(HalfspaceType::Below(plane(
                        0.0,
                        1.0,
                        0.0,
                        y.1,
                        id_base + 3,
                    )))),
                )),
            )),
            Box::new(RegionExpr::Intersection(
                Box::new(RegionExpr::Halfspace(HalfspaceType::Above(plane(
                    0.0,
                    0.0,
                    1.0,
                    z.0,
                    id_base + 4,
                )))),
                Box::new(RegionExpr::Halfspace(HalfspaceType::Below(plane(
                    0.0,
                    0.0,
                    1.0,
                    z.1,
                    id_base + 5,
                )))),
            )),
        );
        Cell::new(Some(id_base as u32), Region { expr }, None, Some(0))
    }
    let mut cell_a = boxed(10, (-2.0, 2.0), (-6.0, 6.0), (-1.0, 1.0));
    cell_a.material_idx = Some(0);
    let mut cell_b = boxed(20, (4.0, 8.0), (-1.0, 1.0), (-1.0, 1.0));
    cell_b.material_idx = Some(1);
    Geometry::new(vec![cell_a, cell_b], vec![be9_material(1), be9_material(2)]).unwrap()
}

fn run_boxes(mode: TrackingMode) -> (f64, f64) {
    let geometry = disjoint_boxes_geometry();
    let mut tally = Tally::new();
    tally.estimator = yamc_tallies::Estimator::TrackLength;
    tally.filters = vec![Filter::Cell(CellFilter::from_id(10))];
    tally.scores = vec![Score::Flux(FluxScore)];
    tally.name = Some("box_a_flux".to_string());
    let source = ParticleSource::Neutron(Source {
        space: yamc_source::source::SourceSpatialDistribution::Point(
            yamc_source::distribution::spatial::Point::new([0.0, -3.0, 0.0]),
        ),
        angle: AngularDistribution::new_monodirectional(0.0, 1.0, 0.0),
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], vec![Arc::new(tally)]);
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
fn plane_extension_does_not_false_kill() {
    let (mean_s, std_s) = run_boxes(TrackingMode::Surface);
    for mode in [TrackingMode::Woodcock, TrackingMode::Hybrid] {
        let (mean_m, std_m) = run_boxes(mode);
        assert!(mean_s > 0.0, "surface flux zero: rig broken");
        assert!(
            mean_m > 0.0,
            "{mode:?} flux zero: false-killed at the plane extension"
        );
        let tol = 3.0 * (std_s.powi(2) + std_m.powi(2)).sqrt();
        assert!(
            (mean_s - mean_m).abs() < tol,
            "{mode:?} {mean_m:.6e} vs surface {mean_s:.6e}: y=1 extension \
             crossing was treated as an exit",
        );
    }
}

#[test]
fn leaks_die_at_the_body_boundary() {
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        let mut model = Model::new(
            disjoint_spheres_geometry(),
            vec![beam_source_14mev()],
            vec![],
        );
        model.verbose = yamc::model::Verbose::silent();
        model.tracking_mode = mode;
        let settings = TransportSettings {
            total_particles: Some(2_000),
            seed: 42,
            threads: Some(1),
            ..Default::default()
        };
        let tracks = model
            .run_with_tracking(&settings, HistorySelection::All)
            .unwrap();

        let mut leaks = 0;
        for track in &tracks.tracks {
            for ev in &track.events {
                if ev.event_type == TrackEventType::Leak {
                    leaks += 1;
                    let r = (ev.position[0] * ev.position[0]
                        + ev.position[1] * ev.position[1]
                        + ev.position[2] * ev.position[2])
                        .sqrt();
                    assert!(
                        r <= 3.0 + 1e-6,
                        "{mode:?}: leak at {:?} (r={r}), beyond body A's boundary",
                        ev.position,
                    );
                    assert!(
                        ev.weight > 0.0,
                        "{mode:?}: leak recorded with zeroed weight"
                    );
                }
            }
        }
        assert!(leaks > 0, "{mode:?}: no leak events recorded");
    }
}
