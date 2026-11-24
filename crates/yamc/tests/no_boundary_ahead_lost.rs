//! A particle with no boundary ahead is a lost particle, not a silent kill.
//!
//! An iron sphere whose surface is a transmission boundary, wrapped in an
//! unbounded void cell (`+sphere`) that nothing closes: once a particle
//! leaves the iron it is in a cell with no surface ahead of it. That is how a
//! missing vacuum boundary shows up, and it must be counted in
//! `model.lost_particles` (naming the void cell it is in) and trip
//! `max_lost_particles`, in surface tracking and in the hybrid tracker's
//! surface-stepped cells alike.
//!
//! Pure Woodcock never asks for a boundary in its delta flights; its twin
//! case is a flight that lands in no cell without having crossed a vacuum
//! surface, covered here with the same sphere and no outer cell at all.
//!
//! The same sphere with `boundary='vacuum'` and no outer cell is the control:
//! it must lose nothing in every mode.
//!
//! Self-skips if the endf-b8.1 Fe56 fixture is absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};

const NUCLIDE: &str = "Fe56";
const RADIUS: f64 = 5.0;
const SPHERE_ID: usize = 1;
const IRON_CELL_ID: u32 = 1;
const VOID_CELL_ID: u32 = 2;
const PARTICLES: usize = 200;

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

fn iron() -> Material {
    let mut material = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.874),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(NUCLIDE.to_string(), yamc_test_cache::nuclide_path(NUCLIDE))]);
    material.read_nuclear_data(&nm, None).unwrap();
    material
}

#[derive(Clone, Copy, PartialEq)]
enum Outside {
    /// Vacuum sphere, no outer cell: the healthy model.
    Vacuum,
    /// Transmission sphere inside an unbounded void cell.
    UnboundedVoid,
    /// Transmission sphere with no outer cell.
    NoCell,
}

fn geometry(outside: Outside) -> Geometry {
    let boundary = if outside == Outside::Vacuum {
        BoundaryType::Vacuum
    } else {
        BoundaryType::Transmission
    };
    let sphere = Arc::new(Surface {
        surface_id: Some(SPHERE_ID),
        kind: SurfaceKind::Sphere {
            x0: 0.0,
            y0: 0.0,
            z0: 0.0,
            radius: RADIUS,
        },
        boundary,
        name: None,
    });
    let mut cells = vec![Cell::new(
        Some(IRON_CELL_ID),
        Region::new_from_halfspace(HalfspaceType::Below(sphere.clone())),
        Some("iron".into()),
        Some(0),
    )];
    if outside == Outside::UnboundedVoid {
        cells.push(Cell::new(
            Some(VOID_CELL_ID),
            Region::new_from_halfspace(HalfspaceType::Above(sphere)),
            Some("outside".into()),
            None,
        ));
    }
    Geometry::new(cells, vec![Arc::new(iron())]).unwrap()
}

fn model(outside: Outside, mode: TrackingMode, max_lost: usize) -> Model {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(
            Discrete::new(vec![14.06e6], vec![1.0]).unwrap(),
        ),
        strength: 1.0,
    });
    let mut model = Model::new(geometry(outside), vec![source], vec![]);
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = max_lost;
    model
}

fn run(model: &mut Model) {
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(PARTICLES),
            seed: 7,
            threads: Some(1),
            ..Default::default()
        })
        .unwrap();
}

/// Run `model` expecting the `max_lost_particles` abort.
fn assert_aborts(mut model: Model) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut model)));
    let payload = result.expect_err("a model losing every escaping particle must abort");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(
        message.contains("Maximum lost particles exceeded (11 > 10)"),
        "unexpected panic: {message}"
    );
}

/// Every particle that leaves the iron is lost in the void cell. Surface
/// tracking loses it just past the sphere it crossed; a hybrid delta flight
/// out of the iron lands somewhere in the void first and is lost there.
fn assert_lost_in_void_cell(mode: TrackingMode) {
    let mut model = model(Outside::UnboundedVoid, mode, usize::MAX);
    run(&mut model);
    // 5 cm of iron passes most 14 MeV neutrons.
    assert!(
        model.lost_particles.len() > PARTICLES / 4,
        "{mode:?}: expected most of {PARTICLES} histories lost, got {}",
        model.lost_particles.len()
    );
    for lp in &model.lost_particles {
        assert_eq!(lp.last_cell_id, Some(VOID_CELL_ID), "{mode:?}: {lp:?}");
        assert_eq!(lp.last_cell_name.as_deref(), Some("outside"));
        assert_eq!(lp.last_cell_index, Some(1));
        let r = lp.position.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(r > RADIUS, "{mode:?}: lost at r = {r}, inside the iron");
        if mode == TrackingMode::Surface {
            assert_eq!(lp.surface_id, Some(SPHERE_ID));
            assert!(
                r - RADIUS < 1e-6,
                "lost at r = {r}, expected just past the sphere"
            );
        }
    }
}

#[test]
fn no_boundary_ahead_is_lost_surface_tracking() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    assert_lost_in_void_cell(TrackingMode::Surface);
}

#[test]
fn no_boundary_ahead_is_lost_hybrid() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    assert_lost_in_void_cell(TrackingMode::Hybrid);
}

#[test]
fn no_boundary_ahead_trips_default_cap_surface_tracking() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    assert_aborts(model(Outside::UnboundedVoid, TrackingMode::Surface, 10));
}

#[test]
fn no_boundary_ahead_trips_default_cap_hybrid() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    assert_aborts(model(Outside::UnboundedVoid, TrackingMode::Hybrid, 10));
}

/// Pure Woodcock: a flight that leaves the iron through the transmission
/// sphere lands in no cell. Lost, naming the cell the flight started in,
/// as surface tracking reports the same particle after its crossing.
#[test]
fn woodcock_flight_into_no_cell_is_lost() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for mode in [TrackingMode::Woodcock, TrackingMode::Surface] {
        let mut model = model(Outside::NoCell, mode, usize::MAX);
        run(&mut model);
        assert!(
            model.lost_particles.len() > PARTICLES / 4,
            "{mode:?}: expected most of {PARTICLES} histories lost, got {}",
            model.lost_particles.len()
        );
        for lp in &model.lost_particles {
            assert_eq!(lp.last_cell_id, Some(IRON_CELL_ID), "{mode:?}: {lp:?}");
            let r = lp.position.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(r > RADIUS, "{mode:?}: lost at r = {r}, inside the iron");
        }
    }
}

#[test]
fn woodcock_flight_into_no_cell_trips_default_cap() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    assert_aborts(model(Outside::NoCell, TrackingMode::Woodcock, 10));
}

/// The healthy model loses nothing: leaking through the vacuum sphere is a
/// normal termination in every mode.
#[test]
fn vacuum_boundary_loses_nothing() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Woodcock,
        TrackingMode::Hybrid,
    ] {
        let mut model = model(Outside::Vacuum, mode, 0);
        run(&mut model);
        assert!(
            model.lost_particles.is_empty(),
            "{mode:?}: lost {} particles",
            model.lost_particles.len()
        );
    }
}
