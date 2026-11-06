//! Cost of implicit surfaces against closed-form and flux surfaces.
//!
//! Per call of `Surface::evaluate` (the sign test of cell finding) and
//! `Surface::distance_to_surface` (the ray crossing of surface tracking),
//! for a sphere and a torus in closed form, the same torus as a flux
//! surface, and formulas: the sphere and the torus as signed distances, a
//! superellipsoid, two spheres joined by a smooth minimum and a gyroid
//! lattice. All but the gyroid are tokamak sized, and their points and
//! rays are the flux bench's: origins through a 1800 by 1800 by 600 cm
//! box, isotropic directions. The gyroid (period 4 cm) has no bounding
//! box, so its rays start in a 40 cm cube of it and stop where they leave
//! the cube, as the searches of a cell bounded by that cube do.
//!
//! `distance_to_surface` is timed over those rays, and over as many that
//! cross the surface (`distance_to_surface_hits`); the share of the first
//! that do is printed for each surface.
//!
//! Run with:
//!     cargo bench -p yamc --bench implicit_surface
//!
//! Each sample evaluates `CALLS` points or rays; divide criterion's time
//! by it for the per-call cost.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;
use yamc::geo::{FluxBoundary, ImplicitFormula, Surface};

const CALLS: usize = 200;

/// Where a case's points and rays start, and where its rays stop.
#[derive(Clone, Copy)]
enum Span {
    /// The flux bench's box, rays unbounded.
    Tokamak,
    /// The cube `[-20, 20]^3`, rays stopped where they leave it.
    Cube,
}

impl Span {
    fn point(self, rng: &mut StdRng) -> [f64; 3] {
        match self {
            Span::Tokamak => [
                rng.random_range(-900.0..900.0),
                rng.random_range(-900.0..900.0),
                rng.random_range(-300.0..300.0),
            ],
            Span::Cube => std::array::from_fn(|_| rng.random_range(-20.0..20.0)),
        }
    }

    /// Where a ray from `o` along `u` stops.
    fn cap(self, o: [f64; 3], u: [f64; 3]) -> f64 {
        match self {
            Span::Tokamak => f64::INFINITY,
            Span::Cube => (0..3)
                .filter(|&i| u[i] != 0.0)
                .map(|i| (20.0_f64.copysign(u[i]) - o[i]) / u[i])
                .fold(f64::INFINITY, f64::min),
        }
    }
}

struct Case {
    name: &'static str,
    surface: Surface,
    span: Span,
}

impl Case {
    fn distance(&self, (o, u, cap): ([f64; 3], [f64; 3], f64)) -> Option<f64> {
        self.surface.distance_to_surface_capped(o, u, None, cap)
    }
}

fn formula(name: &'static str, text: &str, span: Span) -> Case {
    let formula = Arc::new(ImplicitFormula::new(text).unwrap());
    eprintln!("{name}: {} instructions", formula.tape().len());
    Case {
        name,
        surface: Surface::new_implicit(formula, None, None),
        span,
    }
}

fn cases() -> Vec<Case> {
    let tokamak = |name, surface| Case {
        name,
        surface,
        span: Span::Tokamak,
    };
    let circle = FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![600.0, 150.0],
        vec![[-1, 0]],
        vec![150.0],
    )
    .unwrap();
    vec![
        tokamak(
            "sphere",
            Surface::new_sphere(0.0, 0.0, 0.0, 800.0, None, None),
        ),
        formula(
            "sphere_formula",
            "sqrt(x^2 + y^2 + z^2) - 800",
            Span::Tokamak,
        ),
        tokamak(
            "ztorus",
            Surface::new_ztorus(0.0, 0.0, 0.0, 600.0, 150.0, 150.0, None, None),
        ),
        tokamak(
            "flux_torus",
            Surface::new_flux(Arc::new(circle), None, None),
        ),
        formula(
            "torus_formula",
            "sqrt((sqrt(x^2 + y^2) - 600)^2 + z^2) - 150",
            Span::Tokamak,
        ),
        formula(
            "superellipsoid_formula",
            "(x / 800)^4 + (y / 800)^4 + (z / 300)^4 - 1",
            Span::Tokamak,
        ),
        formula(
            "smooth_union_formula",
            "smin(sqrt((x - 300)^2 + y^2 + z^2) - 400, sqrt((x + 300)^2 + y^2 + z^2) - 400, \
             100)",
            Span::Tokamak,
        ),
        formula(
            "gyroid_formula",
            "sin(pi / 2 * x) * cos(pi / 2 * y) + sin(pi / 2 * y) * cos(pi / 2 * z) \
             + sin(pi / 2 * z) * cos(pi / 2 * x) - 0.3",
            Span::Cube,
        ),
    ]
}

fn ray(span: Span, rng: &mut StdRng) -> ([f64; 3], [f64; 3], f64) {
    let o = span.point(rng);
    let mu: f64 = rng.random_range(-1.0..1.0);
    let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    let s = (1.0 - mu * mu).sqrt();
    let u = [s * phi.cos(), s * phi.sin(), mu];
    (o, u, span.cap(o, u))
}

fn bench(c: &mut Criterion) {
    let cases = cases();
    let points: Vec<Vec<(f64, f64, f64)>> = cases
        .iter()
        .map(|case| {
            let mut rng = StdRng::seed_from_u64(1);
            (0..CALLS)
                .map(|_| {
                    let p = case.span.point(&mut rng);
                    (p[0], p[1], p[2])
                })
                .collect()
        })
        .collect();
    let mut rays = Vec::new();
    let mut hits = Vec::new();
    for case in &cases {
        let mut rng = StdRng::seed_from_u64(2);
        let all: Vec<_> = (0..CALLS).map(|_| ray(case.span, &mut rng)).collect();
        let mut crossed: Vec<_> = all
            .iter()
            .copied()
            .filter(|&r| case.distance(r).is_some())
            .collect();
        eprintln!(
            "{}: {} of {CALLS} rays cross the surface",
            case.name,
            crossed.len()
        );
        while crossed.len() < CALLS {
            let r = ray(case.span, &mut rng);
            if case.distance(r).is_some() {
                crossed.push(r);
            }
        }
        rays.push(all);
        hits.push(crossed);
    }

    let mut group = c.benchmark_group("evaluate");
    group.sample_size(20);
    for (case, pts) in cases.iter().zip(&points) {
        group.bench_with_input(BenchmarkId::from_parameter(case.name), pts, |b, pts| {
            b.iter(|| {
                for &p in pts {
                    black_box(case.surface.evaluate(black_box(p)));
                }
            })
        });
    }
    group.finish();
    for (group_name, sets) in [
        ("distance_to_surface", &rays),
        ("distance_to_surface_hits", &hits),
    ] {
        let mut group = c.benchmark_group(group_name);
        group.sample_size(10);
        for (case, set) in cases.iter().zip(sets) {
            group.bench_with_input(BenchmarkId::from_parameter(case.name), set, |b, set| {
                b.iter(|| {
                    for &r in set {
                        black_box(case.distance(black_box(r)));
                    }
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
