//! Transport through implicit surfaces.
//!
//! Spheres given as formulas track like Sphere surfaces: the same
//! histories from the same seed give the same tallies cell by cell, with
//! the outermost sphere the vacuum boundary or inside one. A gyroid
//! lattice of iron and void, in a box of planes or inside an implicit
//! superellipsoid that is itself the vacuum boundary, loses no particle in
//! any tracking mode, and Woodcock tracking (which only asks the surfaces
//! for their sign) agrees cell by cell with surface tracking (which asks
//! for ray crossings). A cell must bound an implicit surface that has no
//! bounding box, and its searches stopped at the cell's box still find the
//! nearest crossing. The transport tests self-skip if the endf-b8.1 Fe56
//! fixture is absent.

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
/// A gyroid of period 4 cm, its walls where the field is +-0.3.
const GYROID: &str = "sin(pi / 2 * x) * cos(pi / 2 * y) + sin(pi / 2 * y) * cos(pi / 2 * z) \
                      + sin(pi / 2 * z) * cos(pi / 2 * x) - 0.3";

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

fn below(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Below(s.clone()))
}

fn above(s: &Arc<Surface>) -> Region {
    Region::new_from_halfspace(HalfspaceType::Above(s.clone()))
}

/// The box `[-h, h]^3` of planes, as a region.
fn cube(h: f64, first_id: usize, boundary: Option<BoundaryType>) -> Region {
    let planes = [
        Surface::x_plane(-h, Some(first_id), boundary.clone()),
        Surface::x_plane(h, Some(first_id + 1), boundary.clone()),
        Surface::y_plane(-h, Some(first_id + 2), boundary.clone()),
        Surface::y_plane(h, Some(first_id + 3), boundary.clone()),
        Surface::z_plane(-h, Some(first_id + 4), boundary.clone()),
        Surface::z_plane(h, Some(first_id + 5), boundary),
    ]
    .map(Arc::new);
    above(&planes[0])
        .intersection(&below(&planes[1]))
        .intersection(&above(&planes[2]))
        .intersection(&below(&planes[3]))
        .intersection(&above(&planes[4]))
        .intersection(&below(&planes[5]))
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

/// Run with 14.1 MeV neutrons from the origin; return the lost-particle
/// count and per-cell `(mean, std)`.
fn run(geometry: Geometry, mode: TrackingMode, particles: usize) -> (usize, Vec<(f64, f64)>) {
    let source = ParticleSource::Neutron(Source {
        space: SourceSpatialDistribution::Point(Point::new([0.0, 0.0, 0.0])),
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

/// Void inside 10 cm, iron to 15, void to 20, iron to 30, then void out
/// to a vacuum sphere at 35 cm, or nothing beyond 30 if that sphere is
/// itself the vacuum boundary. `sphere(radius, id, boundary)` makes the
/// surfaces.
fn shells(
    sphere: impl Fn(f64, usize, Option<BoundaryType>) -> Arc<Surface>,
    vacuum_back: bool,
) -> Geometry {
    let back = vacuum_back.then_some(BoundaryType::Vacuum);
    let s: Vec<Arc<Surface>> = [(10.0, None), (15.0, None), (20.0, None), (30.0, back)]
        .into_iter()
        .enumerate()
        .map(|(i, (r, boundary))| sphere(r, i + 1, boundary))
        .collect();
    let mut cells = vec![
        Cell::new(Some(1), below(&s[0]), None, None),
        Cell::new(
            Some(2),
            above(&s[0]).intersection(&below(&s[1])),
            None,
            Some(0),
        ),
        Cell::new(
            Some(3),
            above(&s[1]).intersection(&below(&s[2])),
            None,
            None,
        ),
        Cell::new(
            Some(4),
            above(&s[2]).intersection(&below(&s[3])),
            None,
            Some(0),
        ),
    ];
    if !vacuum_back {
        let world = sphere(35.0, 5, Some(BoundaryType::Vacuum));
        cells.push(Cell::new(
            Some(5),
            above(&s[3]).intersection(&below(&world)),
            None,
            None,
        ));
    }
    Geometry::new(cells, vec![iron()]).unwrap()
}

#[test]
fn implicit_spheres_track_like_spheres() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for vacuum_back in [false, true] {
        let closed = shells(
            |r, id, boundary| Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, r, Some(id), boundary)),
            vacuum_back,
        );
        // Written as yamc evaluates a sphere, so the two agree in sign
        // everywhere and differ only in where a search puts a crossing.
        let formulas = shells(
            |r, id, boundary| implicit(&format!("sqrt(x^2 + y^2 + z^2) - {r}"), id, boundary),
            vacuum_back,
        );
        let (lost_closed, by_closed) = run(closed, TrackingMode::Surface, PARTICLES);
        let (lost_formulas, by_formulas) = run(formulas, TrackingMode::Surface, PARTICLES);
        assert_eq!((lost_closed, lost_formulas), (0, 0));
        // The same histories: crossings agree to round-off (about 1e-11
        // cm), so the tallies differ only where a history's collision fell
        // within round-off of a crossing and the history went the other
        // way.
        for (cell, (a, b)) in by_closed.iter().zip(&by_formulas).enumerate() {
            assert!(
                (a.0 - b.0).abs() <= 1e-5 * a.0.abs().max(b.0.abs()) + 1e-12,
                "cell {} (vacuum back {vacuum_back}): spheres {} vs formulas {}",
                cell + 1,
                a.0,
                b.0
            );
        }
    }
}

/// An iron gyroid lattice and the void between its walls, in the box
/// `[-10, 10]^3` of vacuum planes or inside the vacuum superellipsoid
/// `x^4 + y^4 + z^4 = 10^4`.
fn lattice(in_planes: bool) -> Geometry {
    let gyroid = implicit(GYROID, 1, None);
    let inside = if in_planes {
        cube(10.0, 2, Some(BoundaryType::Vacuum))
    } else {
        below(&implicit(
            "x^4 + y^4 + z^4 - 10000",
            2,
            Some(BoundaryType::Vacuum),
        ))
    };
    let cells = vec![
        Cell::new(Some(1), inside.intersection(&below(&gyroid)), None, Some(0)),
        Cell::new(Some(2), inside.intersection(&above(&gyroid)), None, None),
    ];
    Geometry::new(cells, vec![iron()]).unwrap()
}

#[test]
fn a_gyroid_lattice_loses_no_particles_and_tracking_modes_agree() {
    if !data_present() {
        eprintln!("skipping: {NUCLIDE} fixture absent");
        return;
    }
    for in_planes in [true, false] {
        let mut results = Vec::new();
        for mode in [
            TrackingMode::Surface,
            TrackingMode::Hybrid,
            TrackingMode::Woodcock,
        ] {
            let (lost, flux) = run(lattice(in_planes), mode, PARTICLES);
            assert_eq!(
                lost, 0,
                "{mode:?} (in planes {in_planes}): {lost} of {PARTICLES} histories lost"
            );
            results.push((mode, flux));
        }
        let (_, surface) = &results[0];
        for (mode, flux) in &results[1..] {
            for (cell, (a, b)) in surface.iter().zip(flux).enumerate() {
                let sigma = (a.1 * a.1 + b.1 * b.1).sqrt();
                assert!(
                    (a.0 - b.0).abs() <= 4.0 * sigma,
                    "cell {} (in planes {in_planes}): surface tracking {} +/- {}, {mode:?} {} \
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
