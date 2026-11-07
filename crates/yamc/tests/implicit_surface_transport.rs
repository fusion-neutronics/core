//! Transport through implicit surfaces.
//!
//! Spheres and tori given as formulas track like Sphere and ZTorus
//! surfaces: the same histories from the same seed give the same tallies
//! cell by cell, in surface and in Woodcock tracking, with the outermost
//! layer the vacuum boundary or inside one. A gyroid lattice of iron and
//! void (in a block of vacuum planes, in a block of planes inside a vacuum
//! sphere, or filling an implicit superellipsoid that is itself the vacuum
//! boundary) and an iron slab with a port cut through it by a
//! superellipsoid lose no particle in any tracking mode, and Woodcock
//! tracking (which asks the surfaces only for their sign, and for the
//! distance to a vacuum boundary) agrees cell by cell with surface
//! tracking (which asks for every crossing). A cell must bound an implicit
//! surface that has no bounding box, and its searches stopped at the
//! cell's box still find the nearest crossing. The transport tests
//! self-skip if the endf-b8.1 Fe56 fixture is absent.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use yamc::geo::{BoundaryType, HalfspaceType, ImplicitFormula, Region, Surface};
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
use yamc_tallies::filter::Filter;
use yamc_tallies::tally::{FluxScore, Score, Tally};
use yamc_tallies::CellFilter;

const NUCLIDE: &str = "Fe56";
const PARTICLES: usize = 4000;
/// Histories per configuration and tracking mode in the at-scale test.
const AT_SCALE: usize = 1_000_000;
/// A gyroid of period 4 cm, its walls where the field is +-0.3.
const GYROID: &str = "sin(pi / 2 * x) * cos(pi / 2 * y) + sin(pi / 2 * y) * cos(pi / 2 * z) \
                      + sin(pi / 2 * z) * cos(pi / 2 * x) - 0.3";
/// A port 30 by 8 by 12 cm with rounded edges, centred on (15, 0, 0).
const PORT: &str = "((x - 15) / 15)^4 + (y / 4)^4 + (z / 6)^4 - 1";
const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];
/// The magnetic axis of the tori, at y = 0.
const AXIS: [f64; 3] = [600.0, 0.0, 0.0];

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

fn implicit(formula: &str, id: usize, boundary: Option<BoundaryType>) -> Arc<Surface> {
    let formula = Arc::new(ImplicitFormula::new(formula).unwrap());
    Arc::new(Surface::new_implicit(formula, Some(id), boundary))
}

/// A torus about the z axis, of major radius 600 cm and minor radius
/// `minor`, as its signed distance: as a user would write it.
fn torus_formula(minor: f64, id: usize, boundary: Option<BoundaryType>) -> Arc<Surface> {
    implicit(
        &format!("sqrt((sqrt(x^2 + y^2) - 600)^2 + z^2) - {minor}"),
        id,
        boundary,
    )
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

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

/// The box from `lo` to `hi` of planes, as a region.
fn block(lo: [f64; 3], hi: [f64; 3], first_id: usize, boundary: Option<BoundaryType>) -> Region {
    let planes = [
        Surface::x_plane(lo[0], Some(first_id), boundary.clone()),
        Surface::x_plane(hi[0], Some(first_id + 1), boundary.clone()),
        Surface::y_plane(lo[1], Some(first_id + 2), boundary.clone()),
        Surface::y_plane(hi[1], Some(first_id + 3), boundary.clone()),
        Surface::z_plane(lo[2], Some(first_id + 4), boundary.clone()),
        Surface::z_plane(hi[2], Some(first_id + 5), boundary),
    ]
    .map(Arc::new);
    above(&planes[0])
        .intersection(&below(&planes[1]))
        .intersection(&above(&planes[2]))
        .intersection(&below(&planes[3]))
        .intersection(&above(&planes[4]))
        .intersection(&below(&planes[5]))
}

/// The box `[-h, h]^3` of planes, as a region.
fn cube(h: f64, first_id: usize, boundary: Option<BoundaryType>) -> Region {
    block([-h; 3], [h; 3], first_id, boundary)
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

/// Run with 14.1 MeV neutrons from `source`; return the lost-particle
/// count and per-cell `(mean, std)`.
fn run(
    geometry: Geometry,
    mode: TrackingMode,
    particles: usize,
    source: [f64; 3],
) -> (usize, Vec<(f64, f64)>) {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new(source)),
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

/// Cells inside the first of the nested `layers` (innermost first),
/// between each layer and the next and, with a `world` (a vacuum
/// surface), from the last layer out to it; iron where `is_iron` says.
/// Without a world the last layer is the vacuum boundary.
fn nested(layers: &[Arc<Surface>], world: Option<&Arc<Surface>>, is_iron: &[bool]) -> Geometry {
    let mut regions = vec![below(&layers[0])];
    for pair in layers.windows(2) {
        regions.push(above(&pair[0]).intersection(&below(&pair[1])));
    }
    if let Some(world) = world {
        regions.push(above(&layers[layers.len() - 1]).intersection(&below(world)));
    }
    let cells = regions
        .into_iter()
        .zip(is_iron)
        .zip(1..)
        .map(|((region, &filled), id)| Cell::new(Some(id), region, None, filled.then_some(0)))
        .collect();
    Geometry::new(cells, vec![iron()]).unwrap()
}

/// Void inside 10 cm, iron to 15, void to 20, iron to 30, then void out
/// to a vacuum sphere at 35 cm, or nothing beyond 30 if that sphere is
/// itself the vacuum boundary. `sphere(radius, id, boundary)` makes the
/// surfaces, the vacuum sphere at 35 cm too.
fn shells(
    sphere: impl Fn(f64, usize, Option<BoundaryType>) -> Arc<Surface>,
    vacuum_back: bool,
) -> Geometry {
    let back = vacuum_back.then_some(BoundaryType::Vacuum);
    let layers: Vec<Arc<Surface>> = [(10.0, None), (15.0, None), (20.0, None), (30.0, back)]
        .into_iter()
        .zip(1..)
        .map(|((r, boundary), id)| sphere(r, id, boundary))
        .collect();
    let world = (!vacuum_back).then(|| sphere(35.0, 5, Some(BoundaryType::Vacuum)));
    nested(&layers, world.as_ref(), &[false, true, false, true, false])
}

/// The layers of a tokamak around a circular plasma of major radius
/// 600 cm and minor radius 150 cm: plasma and a 15 cm scrape-off layer
/// (void), a 5 cm iron first wall and a 40 cm iron blanket, then void out
/// to a vacuum sphere of radius 1100 cm, or nothing beyond the blanket if
/// its back is itself the vacuum boundary. `torus(minor, id, boundary)`
/// makes the layers.
fn tokamak(
    torus: impl Fn(f64, usize, Option<BoundaryType>) -> Arc<Surface>,
    vacuum_back: bool,
) -> Geometry {
    let back = vacuum_back.then_some(BoundaryType::Vacuum);
    let layers: Vec<Arc<Surface>> = [(150.0, None), (165.0, None), (170.0, None), (210.0, back)]
        .into_iter()
        .zip(1..)
        .map(|((minor, boundary), id)| torus(minor, id, boundary))
        .collect();
    let world = (!vacuum_back).then(|| sphere(1100.0, 5, Some(BoundaryType::Vacuum)));
    nested(&layers, world.as_ref(), &[false, false, true, true, false])
}

/// The same histories from the same seed, in surface tracking and in
/// Woodcock tracking (whose vacuum exits are searched for too): the
/// formulas' crossings agree with the closed forms' to round-off (about
/// 1e-11 cm), so the tallies differ only where a history's collision fell
/// within round-off of a crossing and the history went the other way,
/// which moves a tally by ~1e-6 of itself, far inside its statistics.
fn same_tallies(
    closed: impl Fn() -> Geometry,
    formulas: impl Fn() -> Geometry,
    source: [f64; 3],
    what: &str,
) {
    for mode in [TrackingMode::Surface, TrackingMode::Woodcock] {
        let (lost_closed, by_closed) = run(closed(), mode, PARTICLES, source);
        let (lost_formulas, by_formulas) = run(formulas(), mode, PARTICLES, source);
        assert_eq!((lost_closed, lost_formulas), (0, 0), "{mode:?} ({what})");
        for (cell, (a, b)) in by_closed.iter().zip(&by_formulas).enumerate() {
            assert!(
                (a.0 - b.0).abs() <= 1e-5 * a.0.abs().max(b.0.abs()) + 1e-12,
                "cell {} ({mode:?}, {what}): closed forms {} +/- {} vs formulas {} +/- {}",
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
fn implicit_spheres_track_like_spheres() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for vacuum_back in [false, true] {
        // Written as yamc evaluates a sphere, so the two agree in sign
        // everywhere and differ only in where a search puts a crossing.
        same_tallies(
            || shells(sphere, vacuum_back),
            || {
                shells(
                    |r, id, boundary| {
                        implicit(&format!("sqrt(x^2 + y^2 + z^2) - {r}"), id, boundary)
                    },
                    vacuum_back,
                )
            },
            ORIGIN,
            &format!("spheres, vacuum back {vacuum_back}"),
        );
    }
}

#[test]
fn implicit_tori_track_like_ztorus() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for vacuum_back in [false, true] {
        let ztorus = |minor: f64, id: usize, boundary: Option<BoundaryType>| {
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
        same_tallies(
            || tokamak(ztorus, vacuum_back),
            || tokamak(torus_formula, vacuum_back),
            AXIS,
            &format!("tori, vacuum back {vacuum_back}"),
        );
    }
}

/// What holds the gyroid lattice.
#[derive(Clone, Copy, Debug)]
enum Holder {
    /// The box `[-10, 10]^3` of vacuum planes.
    VacuumPlanes,
    /// That box of planes, with void around it inside a vacuum sphere of
    /// radius 20 cm.
    PlanesInVacuumSphere,
    /// The vacuum superellipsoid `x^4 + y^4 + z^4 = 10^4`, which the
    /// lattice fills.
    VacuumSuperellipsoid,
}

/// An iron gyroid lattice and the void between its walls, in `holder`.
fn lattice(holder: Holder) -> Geometry {
    let gyroid = implicit(GYROID, 1, None);
    let vacuum = Some(BoundaryType::Vacuum);
    let (inside, around) = match holder {
        Holder::VacuumPlanes => (cube(10.0, 2, vacuum), None),
        Holder::PlanesInVacuumSphere => {
            let block = cube(10.0, 2, None);
            let around = below(&sphere(20.0, 8, vacuum)).intersection(&block.complement());
            (block, Some(around))
        }
        Holder::VacuumSuperellipsoid => {
            (below(&implicit("x^4 + y^4 + z^4 - 10000", 2, vacuum)), None)
        }
    };
    let mut cells = vec![
        Cell::new(Some(1), inside.intersection(&below(&gyroid)), None, Some(0)),
        Cell::new(Some(2), inside.intersection(&above(&gyroid)), None, None),
    ];
    cells.extend(around.map(|region| Cell::new(Some(3), region, None, None)));
    Geometry::new(cells, vec![iron()]).unwrap()
}

/// An iron slab 20 cm thick (x from 5 to 25 cm) and 60 cm square, with a
/// void port through it cut by the superellipsoid `PORT`, which reaches
/// 5 cm out of either face; void around the slab, inside a vacuum sphere
/// of radius 60 cm.
fn port_through_a_slab() -> Geometry {
    let port = implicit(PORT, 1, None);
    let slab = block([5.0, -30.0, -30.0], [25.0, 30.0, 30.0], 2, None);
    let world = sphere(60.0, 8, Some(BoundaryType::Vacuum));
    let cells = vec![
        Cell::new(Some(1), slab.intersection(&above(&port)), None, Some(0)),
        Cell::new(Some(2), slab.intersection(&below(&port)), None, None),
        Cell::new(
            Some(3),
            below(&world).intersection(&slab.complement()),
            None,
            None,
        ),
    ];
    Geometry::new(cells, vec![iron()]).unwrap()
}

/// No history is lost in any of `modes`, and each of them agrees with the
/// first (surface tracking) cell by cell, within 4 standard deviations.
fn tracking_modes_agree(
    geometry: impl Fn() -> Geometry,
    source: [f64; 3],
    modes: &[TrackingMode],
    particles: usize,
    what: &str,
) {
    let mut results = Vec::new();
    for &mode in modes {
        let (lost, flux) = run(geometry(), mode, particles, source);
        assert_eq!(
            lost, 0,
            "{mode:?} ({what}): {lost} of {particles} histories lost"
        );
        results.push((mode, flux));
    }
    let (_, surface) = &results[0];
    for (mode, flux) in &results[1..] {
        for (cell, (a, b)) in surface.iter().zip(flux).enumerate() {
            let sigma = (a.1 * a.1 + b.1 * b.1).sqrt();
            assert!(
                (a.0 - b.0).abs() <= 4.0 * sigma,
                "cell {} ({what}): surface tracking {} +/- {}, {mode:?} {} +/- {}",
                cell + 1,
                a.0,
                a.1,
                b.0,
                b.1
            );
        }
    }
}

const ALL_MODES: [TrackingMode; 3] = [
    TrackingMode::Surface,
    TrackingMode::Hybrid,
    TrackingMode::Woodcock,
];

#[test]
fn a_gyroid_lattice_loses_no_particles_and_tracking_modes_agree() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for holder in [
        Holder::VacuumPlanes,
        Holder::PlanesInVacuumSphere,
        Holder::VacuumSuperellipsoid,
    ] {
        tracking_modes_agree(
            || lattice(holder),
            ORIGIN,
            &ALL_MODES,
            PARTICLES,
            &format!("{holder:?}"),
        );
    }
}

#[test]
fn a_port_through_a_slab_loses_no_particles_and_tracking_modes_agree() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    tracking_modes_agree(
        port_through_a_slab,
        ORIGIN,
        &ALL_MODES,
        PARTICLES,
        "port through a slab",
    );
}

#[test]
fn cells_must_bound_an_implicit_surface_without_a_box() {
    let gyroid = implicit(GYROID, 7, None);
    let refusal = |region: Region| {
        Geometry::new(vec![Cell::new(Some(1), region, None, None)], vec![]).unwrap_err()
    };
    let message = refusal(below(&gyroid));
    assert!(
        message.contains("Cell 1 uses ImplicitSurface 7, which has no bounding box")
            && message.contains("not bounded either"),
        "{message}"
    );
    let message = refusal(below(&gyroid).intersection(&cube(2e6, 10, None)));
    assert!(message.contains("reaches beyond 1048576 cm"), "{message}");
    // A surface without an id or a name is named by its formula, shortened.
    let unnamed = Arc::new(Surface::new_implicit(
        Arc::new(ImplicitFormula::new(GYROID).unwrap()),
        None,
        None,
    ));
    let message = refusal(above(&unnamed));
    assert!(
        message.contains("ImplicitSurface('sin(pi / 2 * x) * cos(pi / 2 * y)")
            && message.contains("...')"),
        "{message}"
    );
    // Bounded by planes, or by a surface with a box of its own.
    Geometry::new(
        vec![Cell::new(
            Some(1),
            below(&gyroid).intersection(&cube(10.0, 10, None)),
            None,
            None,
        )],
        vec![],
    )
    .unwrap();
    let ball = implicit("x^2 + y^2 + z^2 - 4", 8, None);
    Geometry::new(
        vec![
            Cell::new(
                Some(1),
                below(&ball).intersection(&below(&gyroid)),
                None,
                None,
            ),
            Cell::new(Some(2), above(&ball), None, None),
        ],
        vec![],
    )
    .unwrap();
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
/// first crossing beyond 1e-10 cm: the nearest free-form crossing if it
/// is nearer than the cap and every closed-form crossing, else the nearest
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

/// A gyroid cell's searches, which would otherwise run on to 1e6 cm, stop
/// where the ray leaves the cell's box of planes, and the cell still finds
/// the nearest crossing: from points in the cell, in the box but not in
/// the cell, and outside the box, with and without a cap.
#[test]
fn cells_stop_implicit_searches_at_their_bounding_box() {
    let gyroid = implicit(GYROID, 1, None);
    let cell = Cell::new(
        Some(1),
        below(&gyroid).intersection(&cube(5.0, 2, None)),
        None,
        None,
    );
    let bx = cell.region.bounding_box();
    assert!(bx.is_finite());
    let mut rng = Rng(2026);
    let (mut in_cell, mut elsewhere) = (0, 0);
    for k in 0..400 {
        let p: [f64; 3] = if k % 4 == 0 {
            std::array::from_fn(|_| rng.range(-9.0, 9.0))
        } else {
            std::array::from_fn(|i| rng.range(bx.lower_left[i], bx.upper_right[i]))
        };
        let u = rng.direction();
        let cap = if k % 2 == 0 {
            f64::INFINITY
        } else {
            rng.range(0.0, 12.0)
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
        if cell.contains((p[0], p[1], p[2])) {
            in_cell += 1;
        } else {
            elsewhere += 1;
        }
    }
    assert!(in_cell > 50 && elsewhere > 50, "{in_cell} {elsewhere}");
}

/// The check at scale quoted in the pull request: a million histories in
/// each configuration above, in surface and in Woodcock tracking, lose
/// none, and the two agree cell by cell. Run it in release (about four
/// minutes on 8 threads) with `cargo test --release -p yamc --test
/// implicit_surface_transport -- --ignored`.
#[test]
#[ignore]
fn implicit_surfaces_lose_no_particles_at_scale() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    let modes = [TrackingMode::Surface, TrackingMode::Woodcock];
    for vacuum_back in [false, true] {
        tracking_modes_agree(
            || tokamak(torus_formula, vacuum_back),
            AXIS,
            &modes,
            AT_SCALE,
            &format!("tori, vacuum back {vacuum_back}"),
        );
    }
    for holder in [
        Holder::VacuumPlanes,
        Holder::PlanesInVacuumSphere,
        Holder::VacuumSuperellipsoid,
    ] {
        tracking_modes_agree(
            || lattice(holder),
            ORIGIN,
            &modes,
            AT_SCALE,
            &format!("{holder:?}"),
        );
    }
    tracking_modes_agree(
        port_through_a_slab,
        ORIGIN,
        &modes,
        AT_SCALE,
        "port through a slab",
    );
}
