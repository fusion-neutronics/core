//! Flux and offset surfaces of a `TokamakShape`.
//!
//! - A tokamak without triangularity is a torus: circular, or elliptical
//!   when elongated. Its flux surfaces give the signs and ray crossings of
//!   the `ZTorus` with the same radii, and the offsets of a circular one
//!   those of the `ZTorus` with minor radius `a + d`, with the exact torus
//!   distance.
//! - A D-shaped plasma has the same cross-section at every toroidal angle.
//!   A dense polygon through `TokamakShape::point` decides the side of a
//!   point, and gives its distance from the surface: the closest point of a
//!   surface of revolution lies in the point's own `(R, Z)` half-plane.
//! - Flux surfaces at increasing `rho` are nested, with the shape's points
//!   at `rho` on the `rho` surface, and a thin one keeps the reach of the
//!   boundary scaled by `rho`, so rays grazing it find its crossings.
//! - A surface loads back from JSON to the bit.

use std::f64::consts::TAU;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use yamc::geo::{Surface, SurfaceKind};
use yamc::geometry::flux::{flux_surface, PlasmaShape};
use yamc_source::stellarator::StellaratorShape;
use yamc_source::tokamak::TokamakShape;

fn tokamak(r0: f64, a: f64, elongation: f64, triangularity: f64, shafranov: f64) -> TokamakShape {
    TokamakShape::new(r0, a, elongation, triangularity, shafranov).unwrap()
}

fn surface(shape: &TokamakShape, rho: f64) -> Surface {
    flux_surface(PlasmaShape::Tokamak(shape), rho, None, None).unwrap()
}

fn direction(rng: &mut StdRng) -> [f64; 3] {
    let mu: f64 = rng.random_range(-1.0..1.0);
    let phi: f64 = rng.random_range(0.0..TAU);
    let s = (1.0 - mu * mu).sqrt();
    [s * phi.cos(), s * phi.sin(), mu]
}

/// A point at cylindrical `(R, phi, Z)`.
fn at(r: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    (r * phi.cos(), r * phi.sin(), z)
}

/// Compare `flux` with `torus` at random points and on random rays around
/// a torus of major radius `r0` and half-height `height`. `exact` is the
/// torus's exact signed distance, where it has one.
fn compare_with_ztorus(
    flux: &Surface,
    torus: &Surface,
    (r0, height): (f64, f64),
    exact: Option<&dyn Fn((f64, f64, f64)) -> f64>,
    rng: &mut StdRng,
) {
    let reach = 1.4 * r0;
    for _ in 0..400 {
        let p = (
            rng.random_range(-reach..reach),
            rng.random_range(-reach..reach),
            rng.random_range(-height..height),
        );
        let (value, want) = (flux.evaluate(p), torus.evaluate(p));
        assert_eq!(value < 0.0, want < 0.0, "{p:?}: flux {value}, torus {want}");
        if let Some(exact) = exact {
            let (got, d) = (flux.evaluate_exact(p), exact(p));
            assert!(
                (got - d).abs() <= 1e-9 + 1e-12 * d.abs(),
                "{p:?}: distance {got}, torus {d}"
            );
        }
    }
    for _ in 0..300 {
        let o = [
            rng.random_range(-reach..reach),
            rng.random_range(-reach..reach),
            rng.random_range(-height..height),
        ];
        let u = direction(rng);
        match (
            flux.distance_to_surface(o, u),
            torus.distance_to_surface(o, u),
        ) {
            (None, None) => {}
            (Some(g), Some(w)) => assert!(
                (g - w).abs() <= 1e-7 * w.max(1.0),
                "ray {o:?} {u:?} crossed at {g}, torus says {w}"
            ),
            (got, want) => panic!("ray {o:?} {u:?}: flux {got:?}, torus {want:?}"),
        }
    }
}

#[test]
fn a_circular_tokamak_and_its_offsets_are_ztori() {
    let mut rng = StdRng::seed_from_u64(1);
    // The boundary, and a surface inside a shifted plasma: the circle of
    // radius rho a about R0 + shafranov (1 - rho^2).
    for (shape, rho, offsets) in [
        (
            tokamak(600.0, 150.0, 1.0, 0.0, 0.0),
            1.0,
            &[0.0, 2.0, 15.0, 67.0, -20.0][..],
        ),
        (tokamak(600.0, 150.0, 1.0, 0.0, 40.0), 0.5, &[0.0, 10.0][..]),
    ] {
        let plasma = surface(&shape, rho);
        let a = rho * shape.minor_radius();
        let centre = shape.major_radius() + shape.shafranov_factor() * (1.0 - rho * rho);
        for &d in offsets {
            let flux = plasma.offset(d, None, None).unwrap();
            let minor = a + d;
            let torus = Surface::new_ztorus(0.0, 0.0, 0.0, centre, minor, minor, None, None);
            let exact =
                |p: (f64, f64, f64)| ((p.0.hypot(p.1) - centre).powi(2) + p.2 * p.2).sqrt() - minor;
            compare_with_ztorus(
                &flux,
                &torus,
                (centre, minor + 50.0),
                Some(&exact),
                &mut rng,
            );
        }
    }
}

#[test]
fn an_elongated_tokamak_without_triangularity_is_an_elliptical_ztorus() {
    let mut rng = StdRng::seed_from_u64(2);
    let shape = tokamak(600.0, 150.0, 1.8, 0.0, -30.0);
    for rho in [1.0, 0.6] {
        let (a, centre) = (
            rho * shape.minor_radius(),
            shape.major_radius() + shape.shafranov_factor() * (1.0 - rho * rho),
        );
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, centre, 1.8 * a, a, None, None);
        compare_with_ztorus(
            &surface(&shape, rho),
            &torus,
            (centre, 1.8 * a + 50.0),
            None,
            &mut rng,
        );
    }
}

/// The cross-section of `shape` at `rho` as a closed polygon in `(R, Z)`.
fn section(shape: &TokamakShape, rho: f64, n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| shape.point(rho, k as f64 * TAU / n as f64))
        .collect()
}

/// Signed distance (negative inside) from `q` to the polygon, by
/// crossing parity and the nearest segment.
fn polygon_distance(polygon: &[(f64, f64)], q: (f64, f64)) -> f64 {
    let n = polygon.len();
    let (mut inside, mut d2) = (false, f64::INFINITY);
    for k in 0..n {
        let (a, b) = (polygon[k], polygon[(k + 1) % n]);
        if (a.1 > q.1) != (b.1 > q.1) && a.0 + (q.1 - a.1) * (b.0 - a.0) / (b.1 - a.1) > q.0 {
            inside = !inside;
        }
        let (ex, ez) = (b.0 - a.0, b.1 - a.1);
        let s = (((q.0 - a.0) * ex + (q.1 - a.1) * ez) / (ex * ex + ez * ez)).clamp(0.0, 1.0);
        let (dx, dz) = (a.0 + s * ex - q.0, a.1 + s * ez - q.1);
        d2 = d2.min(dx * dx + dz * dz);
    }
    if inside {
        -d2.sqrt()
    } else {
        d2.sqrt()
    }
}

/// D-shapes across the allowed ranges, with the offsets their reach
/// allows.
fn d_shapes() -> Vec<(TokamakShape, &'static [f64])> {
    vec![
        (
            tokamak(906.0, 292.258, 1.557, 0.270, 0.44789),
            &[0.0, 15.0, -40.0],
        ),
        (tokamak(620.0, 200.0, 1.7, 0.5, 30.0), &[0.0, 15.0]),
        (tokamak(620.0, 200.0, 2.5, -0.95, -60.0), &[0.0, 15.0]),
        (tokamak(620.0, 200.0, 1.0, 1.0, 0.0), &[0.0, 60.0]),
    ]
}

#[test]
fn d_shaped_signs_and_distances_match_the_cross_section() {
    let mut rng = StdRng::seed_from_u64(3);
    for (shape, offsets) in d_shapes() {
        for rho in [1.0, 0.6] {
            let polygon = section(&shape, rho, 20_000);
            let plasma = surface(&shape, rho);
            let (r0, a) = (shape.major_radius(), shape.minor_radius());
            let height = shape.elongation() * a;
            for &d in offsets.iter().filter(|&&d| rho == 1.0 || d == 0.0) {
                let flux = plasma.offset(d, None, None).unwrap();
                let mut checked = 0;
                for _ in 0..300 {
                    let (r, z) = (
                        rng.random_range(r0 - a - 80.0..r0 + a + 80.0),
                        rng.random_range(-height - 80.0..height + 80.0),
                    );
                    let p = at(r, rng.random_range(0.0..TAU), z);
                    // The polygon is within 3e-5 cm of the curve (its chord sag).
                    let want = polygon_distance(&polygon, (r, z)) - d;
                    if want.abs() < 1e-3 {
                        continue;
                    }
                    checked += 1;
                    let value = flux.evaluate(p);
                    assert_eq!(
                        value < 0.0,
                        want < 0.0,
                        "{shape:?} rho {rho} offset {d} at {p:?}: evaluate {value}, polygon {want}"
                    );
                    let got = flux.evaluate_exact(p);
                    assert!(
                        (got - want).abs() < 1e-4,
                        "{shape:?} rho {rho} offset {d} at {p:?}: distance {got}, polygon {want}"
                    );
                }
                assert!(checked > 250, "only {checked} points checked");
            }
        }
    }
}

#[test]
fn d_shaped_ray_crossings_are_where_the_side_changes() {
    let mut rng = StdRng::seed_from_u64(4);
    for (shape, offsets) in d_shapes() {
        // Within 7e-4 cm of the curve (the chord sag bound).
        let polygon = section(&shape, 1.0, 4000);
        let plasma = surface(&shape, 1.0);
        let (r0, a) = (shape.major_radius(), shape.minor_radius());
        for &d in offsets {
            let flux = plasma.offset(d, None, None).unwrap();
            for _ in 0..25 {
                let o = [
                    rng.random_range(-1.3 * r0..1.3 * r0),
                    rng.random_range(-1.3 * r0..1.3 * r0),
                    rng.random_range(-a..a),
                ];
                let u = direction(&mut rng);
                let x = |t: f64| (o[0] + t * u[0], o[1] + t * u[1], o[2] + t * u[2]);
                let inside = |t: f64| flux.evaluate(x(t)) < 0.0;
                // Sphere tracing on the polygon's distance, which cannot step
                // over a crossing (bar a graze within 0.01 cm), with the side
                // from evaluate; the side change is then bisected.
                let start = inside(0.0);
                let mut t = 0.0;
                let mut oracle = None;
                while t < 5.0 * r0 {
                    let p = x(t);
                    let gap = (polygon_distance(&polygon, (p.0.hypot(p.1), p.2)) - d).abs();
                    let next = t + (gap - 1e-3).max(0.01);
                    if inside(next) != start {
                        let (mut lo, mut hi) = (t, next);
                        for _ in 0..60 {
                            let mid = 0.5 * (lo + hi);
                            if inside(mid) == start {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        oracle = Some(hi);
                        break;
                    }
                    t = next;
                }
                match (flux.distance_to_surface(o, u), oracle) {
                    (None, None) => {}
                    (Some(g), Some(w)) => assert!(
                        (g - w).abs() <= 1e-7 * w.max(1.0),
                        "{shape:?} offset {d}: ray {o:?} {u:?} crossed at {g}, side change at {w}"
                    ),
                    (got, want) => panic!(
                        "{shape:?} offset {d}: ray {o:?} {u:?}: crossing {got:?}, side change \
                         {want:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn flux_surfaces_at_increasing_rho_are_nested() {
    let mut rng = StdRng::seed_from_u64(5);
    let rhos = [0.2, 0.4, 0.6, 0.8, 1.0];
    for (shape, _) in d_shapes() {
        let surfaces: Vec<Surface> = rhos.iter().map(|&rho| surface(&shape, rho)).collect();
        for (k, &rho) in rhos.iter().enumerate() {
            for _ in 0..50 {
                let (r, z) = shape.point(rho, rng.random_range(0.0..TAU));
                let p = at(r, rng.random_range(0.0..TAU), z);
                for (j, other) in surfaces.iter().enumerate() {
                    let value = other.evaluate_exact(p);
                    if j == k {
                        assert!(value.abs() < 1e-9, "{shape:?}: rho {rho} point {value} off");
                    } else {
                        assert_eq!(
                            value < 0.0,
                            j > k,
                            "{shape:?}: a rho {rho} point is on the wrong side of rho {}",
                            rhos[j]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_thin_flux_surface_keeps_the_reach_of_the_boundary() {
    // The cross-section at rho is the boundary's scaled by rho, and so is
    // the depth inside up to which the distance is smooth, which bounds
    // the inward offsets allowed. At rho 0.02 a surface is a few cm across,
    // and with triangularity -1 it bends within 2 mm.
    let reach = |surface: &Surface| match &surface.kind {
        SurfaceKind::Flux { boundary, .. } => boundary.inner_reach(),
        _ => unreachable!(),
    };
    let shapes = d_shapes().into_iter().map(|(shape, _)| shape);
    for shape in shapes.chain([tokamak(900.0, 100.0, 2.0, -1.0, -49.0)]) {
        let outer = reach(&surface(&shape, 1.0));
        for rho in [0.2, 0.02] {
            let thin = surface(&shape, rho);
            let inner = reach(&thin);
            assert!(
                (inner / (rho * outer) - 1.0).abs() < 0.02,
                "{shape:?}: reach {inner} at rho {rho}, {outer} at rho 1"
            );
            assert!(thin.offset(-0.5 * inner, None, None).is_ok());
        }
    }
}

#[test]
fn rays_grazing_a_thin_flux_surface_find_its_crossings() {
    // Leaving the surface within 1e-4 to 1e-2 rad of its toroidal tangent,
    // a ray from the inboard side comes back across it within 20 cm.
    // Every crossing returned is on the surface: a march that ran out of
    // budget would return the point it had reached instead, from the capped
    // query that transport makes (`distance_to_surface` would find none).
    let shape = tokamak(900.0, 100.0, 2.0, -1.0, -49.0);
    let thin = surface(&shape, 0.02);
    let SurfaceKind::Flux { boundary, .. } = &thin.kind else {
        unreachable!()
    };
    let mut rng = StdRng::seed_from_u64(9);
    let mut crossings = 0;
    for _ in 0..100 {
        let theta = rng.random_range(0.0..TAU);
        // At zeta = 0 the normal lies in the x-z plane, at right angles
        // to the toroidal direction y.
        let (x, n) = (boundary.point(theta, 0.0), boundary.normal(theta, 0.0));
        let o = [x[0] + 1e-8 * n[0], x[1], x[2] + 1e-8 * n[2]];
        let g: f64 = rng.random_range(1e-4..1e-2);
        let u = [g * n[0], (1.0 - g * g).sqrt(), g * n[2]];
        if let Some(t) = thin.distance_to_surface_capped(o, u, None, f64::INFINITY) {
            let p = (o[0] + t * u[0], o[1] + t * u[1], o[2] + t * u[2]);
            let off = thin.evaluate_exact(p);
            assert!(
                off.abs() < 1e-9,
                "ray {o:?} {u:?}: crossing at {t} is {off} off"
            );
            crossings += 1;
        }
    }
    assert!(crossings >= 30, "{crossings} crossings");
}

#[test]
fn a_tokamak_flux_surface_loads_back_to_the_bit() {
    // Its mode coefficients are full 17-digit floats, which a parser that
    // does not round correctly can land one ulp off: serde_json's default
    // one does, so yamc enables its `float_roundtrip`.
    let shapes = [
        tokamak(600.0, 150.0, 1.7, 0.45, 20.0),
        tokamak(906.0, 292.258, 1.557, 0.27, 0.44789),
        tokamak(906.0, 292.258, 1.557, -0.3, 0.44789),
        tokamak(906.0, 292.258, 1.557, 0.9, 0.44789),
    ];
    for shape in shapes {
        for rho in [1.0, 0.5] {
            let json = serde_json::to_string(&surface(&shape, rho)).unwrap();
            let back: Surface = serde_json::from_str(&json).unwrap();
            assert_eq!(
                serde_json::to_string(&back).unwrap(),
                json,
                "{shape:?} rho {rho}"
            );
        }
    }
}

#[test]
fn rho_outside_the_plasma_is_refused_as_for_a_stellarator() {
    let shape = tokamak(600.0, 150.0, 1.7, 0.4, 20.0);
    let stellarator = StellaratorShape::new(
        1,
        vec![[0, 0, 0], [1, 1, 0]],
        vec![600.0, 150.0],
        vec![[1, -1, 0]],
        vec![150.0],
    )
    .unwrap();
    for rho in [0.0, -0.5, 1.5, f64::NAN] {
        let error = flux_surface(PlasmaShape::Tokamak(&shape), rho, None, None).unwrap_err();
        assert_eq!(
            error,
            format!(
                "rho must be in (0, 1] (got {rho}); a surface outside the plasma is an \
                 OffsetSurface of the rho = 1 flux surface"
            )
        );
        assert_eq!(
            flux_surface(PlasmaShape::Stellarator(&stellarator), rho, None, None).unwrap_err(),
            error
        );
    }
}
