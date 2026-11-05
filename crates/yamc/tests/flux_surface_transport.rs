//! Transport through flux and offset surfaces.
//!
//! A layered stellarator model (plasma, scrape-off layer, iron first wall
//! and blanket, vacuum world) built from the rotating ellipse of the
//! stellarator source example: no particle is lost in any tracking mode,
//! and Woodcock tracking (which only asks the surfaces for their sign)
//! agrees cell by cell with surface tracking (which asks for ray
//! crossings). The same layers on a circular torus, as flux surfaces and
//! as ZTorus surfaces, give the same tallies from the same seed. The
//! source's plasma lies inside the plasma's flux surface. Self-skips if
//! the endf-b8.1 Fe56 fixture is absent.

use std::collections::HashMap;
use std::sync::Arc;

use yamc::geo::{BoundaryType, FluxBoundary, HalfspaceType, Region, Surface};
use yamc::geometry::cell::Cell;
use yamc::geometry::flux::stellarator_flux_surface;
use yamc::geometry::Geometry;
use yamc::model::{Model, TrackingMode, TransportSettings, Verbose};
use yamc_materials::Material;
use yamc_source::distribution::angular::AngularDistribution;
use yamc_source::distribution::energy::Discrete;
use yamc_source::distribution::spatial::Point;
use yamc_source::source::{
    ParticleSource, Source, SourceEnergyDistribution, SourceSpatialDistribution,
};
use yamc_source::stellarator::StellaratorShape;
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::CellFilter;

const NUCLIDE: &str = "Fe56";
const PARTICLES: usize = 4000;

fn data_present() -> bool {
    std::path::Path::new(&yamc_test_cache::nuclide_path(NUCLIDE)).is_dir()
}

fn iron() -> Arc<Material> {
    let mut material = Material::new(
        HashMap::from([(NUCLIDE.to_string(), 1.0)]),
        "atom",
        "g/cm3",
        Some(7.8),
    )
    .unwrap();
    material.set_material_id(1);
    material.set_temperature("294");
    let nm = HashMap::from([(NUCLIDE.to_string(), yamc_test_cache::nuclide_path(NUCLIDE))]);
    material.read_nuclear_data(&nm, None).unwrap();
    Arc::new(material)
}

/// The rotating ellipse of `examples/python/stellarator_plasma_source.py`.
fn shape() -> StellaratorShape {
    let (a, e) = (150.0, 50.0);
    StellaratorShape::new(
        5,
        vec![[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, -1, -1]],
        vec![600.0, a, e, e],
        vec![[1, -1, 0], [1, -1, 1], [1, 1, -1]],
        vec![a, -e, e],
    )
    .unwrap()
}

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

/// Plasma and scrape-off layer (void), 5 cm iron first wall, 40 cm iron
/// blanket, void out to a vacuum sphere. `layers` are the boundary and
/// its offsets at 15, 20 and 60 cm.
fn layered(layers: [Arc<Surface>; 4]) -> Geometry {
    let world = Arc::new(Surface::new_sphere(
        0.0,
        0.0,
        0.0,
        1100.0,
        Some(5),
        Some(BoundaryType::Vacuum),
    ));
    let [plasma, wall, wall_back, blanket_back] = layers;
    let cells = vec![
        Cell::new(Some(1), below(&plasma), Some("plasma".into()), None),
        Cell::new(
            Some(2),
            above(&plasma).intersection(&below(&wall)),
            Some("scrape_off".into()),
            None,
        ),
        Cell::new(
            Some(3),
            above(&wall).intersection(&below(&wall_back)),
            Some("first_wall".into()),
            Some(0),
        ),
        Cell::new(
            Some(4),
            above(&wall_back).intersection(&below(&blanket_back)),
            Some("blanket".into()),
            Some(0),
        ),
        Cell::new(
            Some(5),
            above(&blanket_back).intersection(&below(&world)),
            Some("outside".into()),
            None,
        ),
    ];
    Geometry::new(cells, vec![iron()]).unwrap()
}

fn offsets_of(plasma: Surface) -> [Arc<Surface>; 4] {
    let at = |d: f64, id: usize| Arc::new(plasma.offset(d, Some(id), None).unwrap());
    let layers = [at(15.0, 2), at(20.0, 3), at(60.0, 4)];
    let [wall, wall_back, blanket_back] = layers;
    [Arc::new(plasma), wall, wall_back, blanket_back]
}

fn stellarator() -> Geometry {
    layered(offsets_of(
        stellarator_flux_surface(&shape(), 1.0, Some(1), None).unwrap(),
    ))
}

/// Per-cell flux tallies, cells 1 to 5.
fn tallies() -> Vec<Arc<Tally>> {
    (1..=5)
        .map(|id| {
            let mut tally = Tally::new();
            tally.estimator = yamc_tallies::Estimator::TrackLength;
            tally.filters = vec![Filter::Cell(CellFilter::from_id(id))];
            tally.scores = vec![Score::Flux(FluxScore)];
            Arc::new(tally)
        })
        .collect()
}

/// Run with 14.1 MeV neutrons from the magnetic axis at `zeta = 0`;
/// return the lost-particle count and per-cell `(mean, std)`.
fn run(geometry: Geometry, mode: TrackingMode, particles: usize) -> (usize, Vec<(f64, f64)>) {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([600.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let mut model = Model::new(geometry, vec![source], tallies());
    model.verbose = Verbose::silent();
    model.tracking_mode = mode;
    model.max_lost_particles = usize::MAX;
    model
        .simulate_transport(&TransportSettings {
            total_particles: Some(particles),
            seed: 1,
            threads: Some(8),
            ..Default::default()
        })
        .unwrap();
    let flux = model
        .tallies
        .iter()
        .map(|t| (t.total_mean(), t.total_std()))
        .collect();
    (model.lost_particles.len(), flux)
}

#[test]
fn stellarator_layers_lose_no_particles_and_tracking_modes_agree() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let mut results = Vec::new();
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Hybrid,
        TrackingMode::Woodcock,
    ] {
        let (lost, flux) = run(stellarator(), mode, PARTICLES);
        assert_eq!(lost, 0, "{mode:?}: {lost} of {PARTICLES} histories lost");
        results.push((mode, flux));
    }
    let (_, surface) = &results[0];
    for (mode, flux) in &results[1..] {
        for (cell, (a, b)) in surface.iter().zip(flux).enumerate() {
            let sigma = (a.1 * a.1 + b.1 * b.1).sqrt();
            assert!(
                (a.0 - b.0).abs() <= 4.0 * sigma,
                "cell {}: surface tracking {} +/- {}, {mode:?} {} +/- {}",
                cell + 1,
                a.0,
                a.1,
                b.0,
                b.1
            );
        }
    }
}

#[test]
fn flux_layers_on_a_torus_track_like_ztorus() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let circle = FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![600.0, 150.0],
        vec![[-1, 0]],
        vec![150.0],
    )
    .unwrap();
    let flux = layered(offsets_of(Surface::new_flux(
        Arc::new(circle),
        Some(1),
        None,
    )));
    let torus = |minor: f64, id: usize| {
        Arc::new(Surface::new_ztorus(
            0.0,
            0.0,
            0.0,
            600.0,
            minor,
            minor,
            Some(id),
            None,
        ))
    };
    let tori = layered([
        torus(150.0, 1),
        torus(165.0, 2),
        torus(170.0, 3),
        torus(210.0, 4),
    ]);
    let (lost_flux, by_flux) = run(flux, TrackingMode::Surface, PARTICLES);
    let (lost_tori, by_tori) = run(tori, TrackingMode::Surface, PARTICLES);
    assert_eq!((lost_flux, lost_tori), (0, 0));
    // The same histories: crossings agree to round-off (about 1e-10 of the
    // distance), so the tallies differ only where a history's collision
    // fell within round-off of a crossing and the history went the other
    // way, which moves a tally by ~1e-6 of itself (1e-4 of its standard
    // error at this history count).
    for (cell, (a, b)) in by_flux.iter().zip(&by_tori).enumerate() {
        assert!(
            (a.0 - b.0).abs() <= 1e-5 * a.0.abs().max(b.0.abs()) + 1e-12,
            "cell {}: flux layers {} vs tori {}",
            cell + 1,
            a.0,
            b.0
        );
    }
}

#[test]
fn the_source_plasma_is_inside_the_plasma_surface() {
    let shape = shape();
    let plasma = stellarator_flux_surface(&shape, 1.0, None, None).unwrap();
    let edge = stellarator_flux_surface(&shape, 0.8, None, None).unwrap();
    for k in 0..400 {
        let (rho, theta, zeta) = (
            (k as f64 * 0.618_033_988_7) % 1.0,
            k as f64 * 0.37,
            k as f64 * 0.11,
        );
        let p = shape.point(rho, theta, zeta);
        let x = (p.r * zeta.cos(), p.r * zeta.sin(), p.z);
        assert!(
            plasma.evaluate(x) < 0.0,
            "rho {rho} at {x:?} outside the plasma"
        );
        assert_eq!(
            edge.evaluate(x) < 0.0,
            rho < 0.8,
            "rho {rho} at {x:?} vs the 0.8 surface"
        );
        let on = shape.point(1.0, theta, zeta);
        let x = (on.r * zeta.cos(), on.r * zeta.sin(), on.z);
        assert!(
            plasma.evaluate_exact(x).abs() < 1e-9,
            "rho 1 at {x:?} is off the surface"
        );
    }
}

/// The million-history check quoted in the pull request; run it in
/// release with `cargo test --release -p yamc --test flux_surface_transport
/// -- --ignored`.
#[test]
#[ignore]
fn stellarator_layers_lose_no_particles_at_scale() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for mode in [TrackingMode::Surface, TrackingMode::Woodcock] {
        let (lost, _) = run(stellarator(), mode, 1_000_000);
        assert_eq!(lost, 0, "{mode:?}: {lost} of 1e6 histories lost");
    }
}
