//! Transport through flux and offset surfaces.
//!
//! Layered models (plasma, scrape-off layer, iron first wall and blanket,
//! vacuum world) built from the rotating ellipse of the stellarator source
//! example and from a D-shaped tokamak plasma: no particle is lost in any
//! tracking mode, and Woodcock tracking (which only asks the surfaces for
//! their sign) agrees cell by cell with surface tracking (which asks for
//! ray crossings). The same layers on a circular tokamak, as flux surfaces
//! and as ZTorus surfaces, give the same tallies from the same seed, also
//! with the outermost layer as the vacuum boundary in place of the world
//! sphere. The source's plasma lies inside the plasma's flux surface, and
//! a cell's flux searches stopped at its bounding box still find the
//! nearest crossing. The transport tests self-skip if the endf-b8.1 Fe56
//! fixture is absent.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, Region, Surface, SurfaceKind};
use yamc::geometry::cell::Cell;
use yamc::geometry::flux::{flux_surface, PlasmaShape};
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
use yamc_source::tokamak::TokamakShape;
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

/// A D-shaped tokamak plasma about the same size, with an outward
/// Shafranov shift.
fn d_shape() -> TokamakShape {
    TokamakShape::new(600.0, 150.0, 1.7, 0.45, 20.0).unwrap()
}

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

/// Plasma and scrape-off layer (void), 5 cm iron first wall, 40 cm iron
/// blanket, void out to a vacuum sphere (or, if the blanket's back is
/// itself a vacuum surface, nothing beyond it). `layers` are the boundary
/// and its offsets at 15, 20 and 60 cm.
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
    let vacuum_back = blanket_back.boundary == BoundaryType::Vacuum;
    let mut cells = vec![
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
    ];
    if !vacuum_back {
        cells.push(Cell::new(
            Some(5),
            above(&blanket_back).intersection(&below(&world)),
            Some("outside".into()),
            None,
        ));
    }
    Geometry::new(cells, vec![iron()]).unwrap()
}

/// The layers of `layered`, with the blanket's back a vacuum surface if
/// `vacuum_back`.
fn offsets_of(plasma: Surface, vacuum_back: bool) -> [Arc<Surface>; 4] {
    let at = |d: f64, id: usize, boundary: Option<BoundaryType>| {
        Arc::new(plasma.offset(d, Some(id), boundary).unwrap())
    };
    let back = vacuum_back.then_some(BoundaryType::Vacuum);
    let layers = [at(15.0, 2, None), at(20.0, 3, None), at(60.0, 4, back)];
    let [wall, wall_back, blanket_back] = layers;
    [Arc::new(plasma), wall, wall_back, blanket_back]
}

fn stellarator(vacuum_back: bool) -> Geometry {
    layered(offsets_of(
        flux_surface(PlasmaShape::Stellarator(&shape()), 1.0, Some(1), None).unwrap(),
        vacuum_back,
    ))
}

fn tokamak(vacuum_back: bool) -> Geometry {
    layered(offsets_of(
        flux_surface(PlasmaShape::Tokamak(&d_shape()), 1.0, Some(1), None).unwrap(),
        vacuum_back,
    ))
}

/// Per-cell flux tallies, one per cell.
fn tallies(cells: u32) -> Vec<Arc<Tally>> {
    (1..=cells)
        .map(|id| {
            let mut tally = Tally::new();
            tally.estimator = yamc_tallies::Estimator::TrackLength;
            tally.filters = vec![Filter::Cell(CellFilter::from_id(id))];
            tally.scores = vec![Score::Flux(FluxScore)];
            Arc::new(tally)
        })
        .collect()
}

/// Run with 14.1 MeV neutrons from `(600, 0, 0)`, on the stellarator's
/// magnetic axis at `zeta = 0` and inside every plasma here; return the
/// lost-particle count and per-cell `(mean, std)`.
fn run(geometry: Geometry, mode: TrackingMode, particles: usize) -> (usize, Vec<(f64, f64)>) {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([600.0, 0.0, 0.0])),
        angle: AngularDistribution::Isotropic,
        energy: SourceEnergyDistribution::Discrete(Discrete::new(vec![14.1e6], vec![1.0]).unwrap()),
        strength: 1.0,
    });
    let cells = geometry.cells.len() as u32;
    let mut model = Model::new(geometry, vec![source], tallies(cells));
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
    for vacuum_back in [false, true] {
        tracking_modes_agree(stellarator, vacuum_back);
    }
}

#[test]
fn tokamak_layers_lose_no_particles_and_tracking_modes_agree() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for vacuum_back in [false, true] {
        tracking_modes_agree(tokamak, vacuum_back);
    }
}

fn tracking_modes_agree(geometry: fn(bool) -> Geometry, vacuum_back: bool) {
    let mut results = Vec::new();
    for mode in [
        TrackingMode::Surface,
        TrackingMode::Hybrid,
        TrackingMode::Woodcock,
    ] {
        let (lost, flux) = run(geometry(vacuum_back), mode, PARTICLES);
        assert_eq!(
            lost, 0,
            "{mode:?} (vacuum back {vacuum_back}): {lost} of {PARTICLES} histories lost"
        );
        results.push((mode, flux));
    }
    let (_, surface) = &results[0];
    for (mode, flux) in &results[1..] {
        for (cell, (a, b)) in surface.iter().zip(flux).enumerate() {
            let sigma = (a.1 * a.1 + b.1 * b.1).sqrt();
            assert!(
                (a.0 - b.0).abs() <= 4.0 * sigma,
                "cell {} (vacuum back {vacuum_back}): surface tracking {} +/- {}, {mode:?} {} \
                 +/- {}",
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
fn circular_tokamak_layers_track_like_ztorus() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let circle = TokamakShape::new(600.0, 150.0, 1.0, 0.0, 0.0).unwrap();
    for vacuum_back in [false, true] {
        let flux = layered(offsets_of(
            flux_surface(PlasmaShape::Tokamak(&circle), 1.0, Some(1), None).unwrap(),
            vacuum_back,
        ));
        let torus = |minor: f64, id: usize, boundary: Option<BoundaryType>| {
            Arc::new(Surface::new_ztorus(
                0.0,
                0.0,
                0.0,
                600.0,
                minor,
                minor,
                Some(id),
                boundary,
            ))
        };
        let tori = layered([
            torus(150.0, 1, None),
            torus(165.0, 2, None),
            torus(170.0, 3, None),
            torus(210.0, 4, vacuum_back.then_some(BoundaryType::Vacuum)),
        ]);
        same_tallies(flux, tori);
    }
}

fn same_tallies(flux: Geometry, tori: Geometry) {
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
    let plasma = flux_surface(PlasmaShape::Stellarator(&shape), 1.0, None, None).unwrap();
    let edge = flux_surface(PlasmaShape::Stellarator(&shape), 0.8, None, None).unwrap();
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

/// Small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }

    fn direction(&mut self) -> [f64; 3] {
        let mu = self.range(-1.0, 1.0);
        let phi = self.range(0.0, TAU);
        let s = (1.0 - mu * mu).sqrt();
        [s * phi.cos(), s * phi.sin(), mu]
    }
}

/// What `Cell::closest_surface_within` answers, from each surface's own
/// first crossing beyond 1e-10 cm: the nearest flux crossing if it is
/// nearer than the cap and every closed-form crossing, else the nearest
/// closed-form one, else the cap.
fn nearest_crossing(
    cell: &Cell,
    p: [f64; 3],
    u: [f64; 3],
    cap: f64,
) -> Option<(BoundaryType, f64, Option<usize>)> {
    let surfaces = cell.region.surfaces_with_sense();
    let mut closed: Option<(f64, &Surface)> = None;
    let mut free: Option<(f64, &Surface)> = None;
    for (surface, _) in &surfaces {
        let Some(d) = surface.distance_to_surface(p, u).filter(|&d| d > 1e-10) else {
            continue;
        };
        let best = if surface.is_free_form() {
            &mut free
        } else {
            &mut closed
        };
        if best.is_none_or(|(b, _)| d < b) {
            *best = Some((d, surface));
        }
    }
    let nearest_closed = closed.map_or(f64::INFINITY, |(d, _)| d);
    match (free, closed) {
        (Some((d, s)), _) if d < nearest_closed.min(cap) => {
            Some((s.boundary.clone(), d, s.surface_id))
        }
        (_, Some((d, s))) => Some((s.boundary.clone(), d, s.surface_id)),
        _ if cap < f64::INFINITY => Some((BoundaryType::Transmission, cap, None)),
        _ => None,
    }
}

/// A cell's flux searches stop where the ray leaves the cell's bounding
/// box, and the cell still finds the nearest crossing: from points in the
/// cell (where the box decides), in its box but not in the cell (where the
/// searches can run on past the box) and outside the box, with and without
/// a cap.
#[test]
fn cells_stop_flux_searches_at_their_bounding_box() {
    let plasma = flux_surface(PlasmaShape::Stellarator(&shape()), 1.0, Some(1), None).unwrap();
    let boundary = match &plasma.kind {
        SurfaceKind::Flux { boundary, .. } => Arc::clone(boundary),
        _ => unreachable!(),
    };
    let wall = Arc::new(plasma.offset(15.0, Some(2), None).unwrap());
    let plasma = Arc::new(plasma);
    let top = Arc::new(Surface::z_plane(60.0, Some(3), None));
    let bottom = Arc::new(Surface::z_plane(-60.0, Some(4), None));
    // A slice of the scrape-off layer.
    let region = above(&plasma)
        .intersection(&below(&wall))
        .intersection(&below(&top))
        .intersection(&above(&bottom));
    let cell = Cell::new(Some(1), region, None, None);
    let bx = cell.region.bounding_box();
    assert!(bx.is_finite());
    let exit = |p: [f64; 3], u: [f64; 3]| {
        (0..3)
            .filter(|&i| u[i] != 0.0)
            .map(|i| {
                let face = if u[i] > 0.0 {
                    bx.upper_right[i]
                } else {
                    bx.lower_left[i]
                };
                (face - p[i]) / u[i]
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut rng = Rng(2026);
    let (mut in_box, mut past_box) = (0, 0);
    for k in 0..300 {
        let p = match k % 3 {
            0 => {
                let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
                let (x, n) = (boundary.point(theta, zeta), boundary.normal(theta, zeta));
                let d = rng.range(0.5, 14.5);
                [x[0] + d * n[0], x[1] + d * n[1], x[2] + d * n[2]]
            }
            1 => std::array::from_fn(|i| rng.range(bx.lower_left[i], bx.upper_right[i])),
            _ => [
                rng.range(-800.0, 800.0),
                rng.range(-800.0, 800.0),
                rng.range(61.0, 200.0),
            ],
        };
        let u = rng.direction();
        let cap = if k % 2 == 0 {
            f64::INFINITY
        } else {
            rng.range(0.0, 400.0)
        };
        let got = cell.closest_surface_within(p, u, None, cap);
        let want = nearest_crossing(&cell, p, u, cap);
        let same = match (&got, &want) {
            (None, None) => true,
            (Some((bg, dg, ig)), Some((bw, dw, iw))) => {
                bg == bw && ig == iw && (dg - dw).abs() <= 1e-9 * dw.max(1.0)
            }
            _ => false,
        };
        assert!(
            same,
            "from {p:?} along {u:?}, cap {cap}: {got:?}, want {want:?}"
        );
        let inside = (0..3).all(|i| p[i] >= bx.lower_left[i] && p[i] <= bx.upper_right[i]);
        if inside && exit(p, u) < cap {
            match got {
                Some((_, d, _)) if d <= exit(p, u) => in_box += 1,
                _ => past_box += 1,
            }
        }
    }
    // Both ways out of the box-limited search are taken.
    assert!(in_box > 50 && past_box > 0, "{in_box} {past_box}");
}

/// The million-history checks quoted in the pull requests; run them in
/// release with `cargo test --release -p yamc --test flux_surface_transport
/// -- --ignored`.
#[test]
#[ignore]
fn stellarator_layers_lose_no_particles_at_scale() {
    lose_no_particles_at_scale(stellarator);
}

#[test]
#[ignore]
fn tokamak_layers_lose_no_particles_at_scale() {
    lose_no_particles_at_scale(tokamak);
}

fn lose_no_particles_at_scale(geometry: fn(bool) -> Geometry) {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for vacuum_back in [false, true] {
        for mode in [TrackingMode::Surface, TrackingMode::Woodcock] {
            let (lost, _) = run(geometry(vacuum_back), mode, 1_000_000);
            assert_eq!(
                lost, 0,
                "{mode:?} (vacuum back {vacuum_back}): {lost} of 1e6 histories lost"
            );
        }
    }
}
