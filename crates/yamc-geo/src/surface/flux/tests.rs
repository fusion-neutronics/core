//! Oracle tests for flux and offset surfaces.
//!
//! - A circular torus written as a Fourier series has the exact distance
//!   `sqrt((rho - R0)^2 + z^2) - a`, and its offsets are tori, so the
//!   existing `ZTorus` solver is an independent ray oracle (including the
//!   spindle torus past the fold, with its crease on the z axis).
//! - For a non-axisymmetric boundary, `P = X(theta, zeta) + d N(theta,
//!   zeta)` is at signed distance `d` with its closest point at `(theta,
//!   zeta)` while `|d|` is inside the local reach.
//! - Signs away from the surface are checked against a dense winding
//!   number in the point's own cut.

use super::FluxBoundary;
use crate::surface::{BoundaryType, Surface, SurfaceKind};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

/// Small deterministic generator (xorshift64*), so the tests need no
/// dependency.
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

fn circular_torus(r0: f64, a: f64) -> FluxBoundary {
    FluxBoundary::new(1, vec![[0, 0], [1, 0]], vec![r0, a], vec![[-1, 0]], vec![a]).unwrap()
}

/// The rotating ellipse of `examples/python/stellarator_plasma_source.py`
/// at rho = 1: `R = 600 + 150 cos(t) + 50 cos(t - 5 z)`,
/// `Z = 150 sin(t) - 50 sin(t - 5 z)`.
fn rotating_ellipse() -> FluxBoundary {
    FluxBoundary::new(
        5,
        vec![[0, 0], [1, 0], [1, 1], [-1, -1]],
        vec![600.0, 150.0, 50.0, 50.0],
        vec![[-1, 0], [-1, 1], [1, -1]],
        vec![150.0, -50.0, 50.0],
    )
    .unwrap()
}

/// A bean: concave on the inboard side, with a rotating perturbation.
/// `R = 600 + 120 cos(t) + 45 cos(2t) + 15 cos(t - 5z)`,
/// `Z = 190 sin(t) + 5 sin(2t) - 15 sin(t - 5z)`.
fn bean() -> FluxBoundary {
    FluxBoundary::new(
        5,
        vec![[0, 0], [1, 0], [2, 0], [1, 1], [-1, -1]],
        vec![600.0, 120.0, 45.0, 15.0, 15.0],
        vec![[-1, 0], [-2, 0], [-1, 1], [1, -1]],
        vec![190.0, 5.0, -15.0, 15.0],
    )
    .unwrap()
}

fn add(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

fn wrap(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

fn boundary_of(surface: &Surface) -> &FluxBoundary {
    match &surface.kind {
        SurfaceKind::Flux { boundary, .. } => boundary,
        _ => panic!("not a flux surface"),
    }
}

#[test]
fn circular_torus_distance_is_exact() {
    let (r0, a) = (600.0, 150.0);
    let torus = circular_torus(r0, a);
    let mut rng = Rng(7);
    let exact = |p: [f64; 3]| ((p[0].hypot(p[1]) - r0).powi(2) + p[2] * p[2]).sqrt() - a;
    for _ in 0..300 {
        let p = [
            rng.range(-900.0, 900.0),
            rng.range(-900.0, 900.0),
            rng.range(-300.0, 300.0),
        ];
        let d = torus.signed_distance(p);
        assert!(
            (d - exact(p)).abs() <= 1e-9 + 1e-12 * exact(p).abs(),
            "point {p:?}: {d} vs exact {}",
            exact(p)
        );
    }
    // On the z axis and on the magnetic axis, where whole circles of
    // surface points are equally close.
    for p in [
        [0.0, 0.0, 50.0],
        [0.0, 0.0, 0.0],
        [r0, 0.0, 0.0],
        [0.0, -r0, 0.0],
    ] {
        let d = torus.signed_distance(p);
        assert!(
            (d - exact(p)).abs() <= 1e-9,
            "point {p:?}: {d} vs {}",
            exact(p)
        );
    }
}

#[test]
fn normal_offsets_are_at_their_distance() {
    for (name, boundary, depths) in [
        (
            "rotating ellipse",
            rotating_ellipse(),
            vec![-30.0, -5.0, -1e-3, -1e-8, 0.0, 1e-8, 1e-3, 1.0, 15.0, 67.0],
        ),
        (
            "bean",
            bean(),
            vec![-10.0, -1e-3, -1e-8, 0.0, 1e-8, 1e-3, 1.0, 15.0, 67.0],
        ),
    ] {
        // The oracle holds while |d| is inside the reach on that side.
        let patches = &boundary.engine.patches;
        let (inner, outer) = (
            0.5 / patches.max_convex,
            0.5 / patches.max_concave.max(1e-12),
        );
        let depths: Vec<f64> = depths
            .into_iter()
            .filter(|&d| (d < 0.0 && -d < inner) || (d >= 0.0 && d < outer))
            .collect();
        assert!(
            depths.len() >= 7,
            "{name}: reach too small for the test ({inner}, {outer})"
        );
        let mut rng = Rng(11);
        for _ in 0..60 {
            let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
            let x = boundary.point(theta, zeta);
            let n = boundary.normal(theta, zeta);
            for &d in &depths {
                let p = add(x, n, d);
                let (got, t, z) = boundary.closest_point(p);
                assert!(
                    (got - d).abs() <= 1e-9 + 1e-12 * d.abs(),
                    "{name}: offset {d} at ({theta}, {zeta}) gave distance {got}"
                );
                if d.abs() >= 1e-3 {
                    assert!(
                        wrap(t - theta).abs() < 1e-6 && wrap(z - zeta).abs() < 1e-6,
                        "{name}: offset {d} at ({theta}, {zeta}) found the foot at ({t}, {z})"
                    );
                }
            }
        }
    }
}

/// Inside/outside by crossing parity against a dense polygon of the cut
/// through `p`; `None` when `p` is too close to the polygon for its chord
/// sag to be ignored.
fn inside_dense(boundary: &FluxBoundary, p: [f64; 3]) -> Option<bool> {
    let phi = p[1].atan2(p[0]);
    let q = (p[0].hypot(p[1]), p[2]);
    const N: usize = 20_000;
    let pts: Vec<(f64, f64)> = (0..N)
        .map(|k| {
            let x = boundary.point(k as f64 * TAU / N as f64, phi);
            (x[0].hypot(x[1]), x[2])
        })
        .collect();
    let mut inside = false;
    let mut nearest = f64::INFINITY;
    for k in 0..N {
        let (a, b) = (pts[k], pts[(k + 1) % N]);
        nearest = nearest.min((a.0 - q.0).hypot(a.1 - q.1));
        if (a.1 > q.1) != (b.1 > q.1) && a.0 + (q.1 - a.1) * (b.0 - a.0) / (b.1 - a.1) > q.0 {
            inside = !inside;
        }
    }
    (nearest > 0.1).then_some(inside)
}

#[test]
fn signs_match_a_dense_winding_number() {
    let surface = Surface::new_flux(Arc::new(bean()), None, None);
    let boundary = boundary_of(&surface);
    let mut rng = Rng(3);
    let mut checked = 0;
    for _ in 0..400 {
        // Concentrated on the cut region, including the concave dent.
        let (phi, rho, z) = (
            rng.range(0.0, TAU),
            rng.range(380.0, 820.0),
            rng.range(-240.0, 240.0),
        );
        let p = [rho * phi.cos(), rho * phi.sin(), z];
        let Some(inside) = inside_dense(boundary, p) else {
            continue;
        };
        checked += 1;
        let value = surface.evaluate((p[0], p[1], p[2]));
        assert_eq!(value < 0.0, inside, "point {p:?}: evaluate {value}");
        assert!(value != 0.0);
    }
    assert!(checked > 300, "only {checked} points checked");
}

#[test]
fn bounds_and_boxes_contain_the_truth() {
    for boundary in [rotating_ellipse(), bean()] {
        let e = &boundary.engine;
        let mut rng = Rng(5);
        for _ in 0..200 {
            let p = [
                rng.range(-850.0, 850.0),
                rng.range(-850.0, 850.0),
                rng.range(-300.0, 300.0),
            ];
            let (_, lo, hi) = e.bounds(p);
            let d = e.signed(p).d;
            assert!(
                lo <= d + 1e-9 && d <= hi + 1e-9,
                "{lo} <= {d} <= {hi} at {p:?}"
            );
        }
        // Every patch box holds its patch.
        let patches = &e.patches;
        for _ in 0..2000 {
            let (t, z) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
            let (i, j) = patches.patch_of(t, z);
            let x = e.series.point(t, z).x;
            assert_eq!(
                patches.box_of(i, j).distance2(x),
                0.0,
                "({t}, {z}) outside its box"
            );
        }
    }
}

/// Compare ray crossings of a circular torus's offsets with `ZTorus`.
fn compare_with_ztorus(r0: f64, a: f64, offsets: &[f64], rays: usize, seed: u64) {
    let base = Surface::new_flux(Arc::new(circular_torus(r0, a)), None, None);
    let mut rng = Rng(seed);
    for &d in offsets {
        let flux = base.offset(d, None, None).unwrap();
        let torus = Surface::new_ztorus(0.0, 0.0, 0.0, r0, a + d, a + d, None, None);
        let reach = 1.4 * (r0 + a + d.max(0.0));
        for _ in 0..rays {
            let o = [
                rng.range(-reach, reach),
                rng.range(-reach, reach),
                rng.range(-(a + d.abs() + 50.0), a + d.abs() + 50.0),
            ];
            let u = rng.direction();
            let got = flux.distance_to_surface(o, u);
            let want = torus.distance_to_surface(o, u);
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => assert!(
                    (g - w).abs() <= 1e-7 * w.max(1.0),
                    "offset {d}: ray {o:?} {u:?} crossed at {g}, torus says {w}"
                ),
                _ => panic!("offset {d}: ray {o:?} {u:?}: flux {got:?}, torus {want:?}"),
            }
        }
    }
}

#[test]
fn torus_offsets_cross_where_ztorus_does() {
    compare_with_ztorus(600.0, 150.0, &[0.0, 2.0, 15.0, 67.0, -20.0], 150, 21);
}

#[test]
fn spindle_torus_past_the_fold_crosses_where_ztorus_does() {
    // a + d > R0: the outward offset has folded onto the z axis and the
    // level set is a spindle torus with a crease there.
    compare_with_ztorus(600.0, 150.0, &[500.0], 150, 22);
    // Rays through the crease on the z axis, where ZTorus's root filter is
    // ill-conditioned: an oracle from the exact distance, sampled finely
    // and bisected.
    let base = Surface::new_flux(Arc::new(circular_torus(600.0, 150.0)), None, None);
    let flux = base.offset(500.0, None, None).unwrap();
    let g = |o: [f64; 3], u: [f64; 3], t: f64| {
        let p = add(o, u, t);
        ((p[0].hypot(p[1]) - 600.0).powi(2) + p[2] * p[2]).sqrt() - 150.0 - 500.0
    };
    let oracle = |o: [f64; 3], u: [f64; 3]| {
        let (step, mut t) = (0.01, 1e-6);
        let g0 = g(o, u, t);
        while t < 3000.0 {
            let next = t + step;
            if g(o, u, next).signum() != g0.signum() {
                let (mut a, mut b) = (t, next);
                for _ in 0..80 {
                    let m = 0.5 * (a + b);
                    if g(o, u, m).signum() == g0.signum() {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                return Some(b);
            }
            t = next;
        }
        None
    };
    for o in [[0.0, 0.0, 400.0], [10.0, 0.0, -400.0], [0.0, 3.0, 0.0]] {
        for u in [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.6, 0.0, -0.8]] {
            let (got, want) = (flux.distance_to_surface(o, u), oracle(o, u));
            match (got, want) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    assert!(
                        (g - w).abs() <= 1e-7 * w.max(1.0),
                        "{o:?} {u:?}: {g} vs {w}"
                    )
                }
                _ => panic!("{o:?} {u:?}: flux {got:?}, oracle {want:?}"),
            }
        }
    }
}

#[test]
fn a_particle_on_the_surface_finds_the_right_crossing() {
    let boundary = Arc::new(rotating_ellipse());
    let surface = Surface::new_flux(Arc::clone(&boundary), Some(9), None);
    let mut rng = Rng(13);
    for _ in 0..20 {
        let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
        let x = boundary.point(theta, zeta);
        let n = boundary.normal(theta, zeta);
        for s in [1e-9, 1e-8, 3e-8, -1e-9, -1e-8, -3e-8] {
            let p = add(x, n, s);
            // Straight at the surface: the crossing just ahead is returned.
            let towards = [-s.signum() * n[0], -s.signum() * n[1], -s.signum() * n[2]];
            let t = surface.distance_to_surface(p, towards).unwrap();
            assert!(
                (t - s.abs()).abs() <= 1e-10 + 1e-9 * s.abs(),
                "{s} from the surface, moving towards it: crossing at {t}"
            );
            // Straight away from it: the crossing just made is not repeated.
            let away = [s.signum() * n[0], s.signum() * n[1], s.signum() * n[2]];
            let next = surface.distance_to_surface(p, away);
            assert!(
                next.is_none_or(|t| t > 1.0),
                "moving away from {s}: {next:?}"
            );
            // With the transport hint that it was just crossed.
            let next = surface.distance_to_surface_on(p, away, Some(s.abs()));
            assert!(
                next.is_none_or(|t| t > 1.0),
                "hinted, moving away from {s}: {next:?}"
            );
        }
    }
}

#[test]
fn evaluate_is_a_pure_function_of_the_point() {
    let surface = Surface::new_flux(Arc::new(bean()), None, None)
        .offset(15.0, None, None)
        .unwrap();
    let mut rng = Rng(17);
    let points: Vec<(f64, f64, f64)> = (0..200)
        .map(|_| {
            (
                rng.range(-800.0, 800.0),
                rng.range(-800.0, 800.0),
                rng.range(-250.0, 250.0),
            )
        })
        .collect();
    let forward: Vec<u64> = points
        .iter()
        .map(|&p| surface.evaluate(p).to_bits())
        .collect();
    let mut backward: Vec<u64> = points
        .iter()
        .rev()
        .map(|&p| surface.evaluate(p).to_bits())
        .collect();
    backward.reverse();
    assert_eq!(forward, backward);
    let threaded: Vec<u64> = std::thread::scope(|scope| {
        let handles: Vec<_> = points
            .chunks(50)
            .map(|chunk| {
                let surface = &surface;
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&p| surface.evaluate(p).to_bits())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    assert_eq!(forward, threaded);
}

#[test]
fn offsets_add_up_and_are_checked() {
    let base = Surface::new_flux(Arc::new(rotating_ellipse()), None, None);
    let fw = base.offset(15.0, None, None).unwrap();
    let blanket = fw
        .offset(52.0, Some(3), Some(BoundaryType::Vacuum))
        .unwrap();
    let direct = base.offset(67.0, None, None).unwrap();
    let p = (1100.0, 30.0, 40.0);
    assert_eq!(blanket.evaluate(p), direct.evaluate(p));
    assert_eq!(blanket.surface_id, Some(3));
    assert_eq!(blanket.boundary, BoundaryType::Vacuum);
    // Deeper inward than the convex parts' radius of curvature folds.
    let error = base.offset(-60.0, None, None).unwrap_err();
    assert!(error.contains("FluxSurface(shape, rho < 1)"), "{error}");
    let error = Surface::new_sphere(0.0, 0.0, 0.0, 1.0, None, None)
        .offset(1.0, None, None)
        .unwrap_err();
    assert!(error.contains("not a Sphere"), "{error}");
    // The inside is bounded and grows with the offset; the outside is not.
    let (lo, hi) = fw.bounding_box(true).unwrap();
    assert!(hi[0] >= 800.0 + 15.0 && lo[2] <= -200.0 - 15.0);
    assert!(fw.bounding_box(false).is_none());
}

#[test]
fn bad_boundaries_are_refused() {
    let err = FluxBoundary::new(0, vec![[0, 0]], vec![1.0], vec![[-1, 0]], vec![1.0]).unwrap_err();
    assert!(err.contains("num_field_periods"), "{err}");
    let err = FluxBoundary::new(
        1,
        vec![[0, 0], [0, 0]],
        vec![600.0, 1.0],
        vec![[-1, 0]],
        vec![1.0],
    )
    .unwrap_err();
    assert!(err.contains("more than once"), "{err}");
    // Minor radius larger than the major radius: the cut crosses R = 0.
    let err = FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![100.0, 150.0],
        vec![[-1, 0]],
        vec![150.0],
    )
    .unwrap_err();
    assert!(err.contains("z axis"), "{err}");
    // A figure of eight crosses itself.
    let err = FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![600.0, 100.0],
        vec![[-2, 0]],
        vec![100.0],
    )
    .unwrap_err();
    assert!(
        err.contains("crosses itself") || err.contains("no area"),
        "{err}"
    );
}

#[test]
fn a_boundary_round_trips_through_json() {
    let surface = Surface::new_flux(Arc::new(bean()), Some(4), None)
        .offset(10.0, Some(5), None)
        .unwrap();
    let json = serde_json::to_string(&surface).unwrap();
    let back: Surface = serde_json::from_str(&json).unwrap();
    for p in [
        (700.0, 0.0, 0.0),
        (650.0, 100.0, 150.0),
        (500.0, -200.0, -50.0),
    ] {
        assert_eq!(surface.evaluate(p), back.evaluate(p));
    }
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
}

/// Axisymmetric dumbbell `R = r0 + l cos(t)`, `Z = sin(t) (w + h
/// cos^6(t))`: two bulbs joined by a bar `2 w` thick at `R = r0`.
fn dumbbell(r0: f64, l: f64, w: f64, h: f64) -> FluxBoundary {
    FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![r0, l],
        vec![[-1, 0], [-3, 0], [-5, 0], [-7, 0]],
        vec![
            w + 2.5 * h / 32.0,
            4.5 * h / 32.0,
            2.5 * h / 32.0,
            0.5 * h / 32.0,
        ],
    )
    .unwrap()
}

/// First crossing of `D = offset` along the ray beyond 1e-10 cm by brute
/// force: sphere tracing on the exact distance with steps of at least
/// `h_min`, then bisection.
fn brute_force_root(
    boundary: &FluxBoundary,
    offset: f64,
    o: [f64; 3],
    u: [f64; 3],
    t_max: f64,
    h_min: f64,
) -> Option<f64> {
    let g = |t: f64| boundary.signed_distance(add(o, u, t)) - offset;
    let (mut t, mut gt) = (1e-10, g(1e-10));
    while t < t_max {
        let next = t + gt.abs().max(h_min);
        let gn = g(next);
        if gn == 0.0 || gn.signum() != gt.signum() {
            let (mut a, mut b) = (t, next);
            while b - a > 1e-12 {
                let m = 0.5 * (a + b);
                if g(m).signum() == gt.signum() {
                    a = m;
                } else {
                    b = m;
                }
            }
            return Some(b);
        }
        (t, gt) = (next, gn);
    }
    None
}

/// The engine's crossing (transmission and vacuum alike) against the brute
/// force one; an earlier engine crossing is accepted if it is a genuine
/// sign change the brute force stepped over. Returns the evaluations used.
fn check_crossing(boundary: &FluxBoundary, offset: f64, o: [f64; 3], u: [f64; 3]) -> usize {
    let want = brute_force_root(boundary, offset, o, u, 2000.0, 1e-5);
    let g = |t: f64| boundary.signed_distance(add(o, u, t)) - offset;
    let mut evaluations = 0;
    for vacuum in [false, true] {
        let trace = boundary
            .engine
            .trace_ray(o, u, offset, None, vacuum, f64::INFINITY);
        assert!(
            !trace.exhausted,
            "budget ran out: {o:?} {u:?} offset {offset}"
        );
        evaluations = evaluations.max(trace.evaluations);
        match (trace.crossing, want) {
            (None, None) => {}
            (Some(got), Some(want)) if (got - want).abs() <= 1e-7 * want.max(1.0) => {}
            (Some(got), _) if g(got - 1e-7).signum() != g(got + 1e-7).signum() => {
                assert!(want.is_none_or(|w| got < w), "{got} after {want:?}");
            }
            (got, want) => panic!(
                "offset {offset} vacuum {vacuum} from {o:?} along {u:?}: engine {got:?}, brute \
                 force {want:?}"
            ),
        }
    }
    evaluations
}

#[test]
fn the_reach_stops_at_half_a_thin_part() {
    // At full size, and at a hundredth of it on the same major radius,
    // where a patch is far longer toroidally than poloidally.
    for scale in [1.0, 0.01] {
        for w in [1.0, 3.0].map(|w| w * scale) {
            let boundary = dumbbell(1000.0, 300.0 * scale, w, 100.0 * scale);
            // Half the bar's thickness bounds the depth up to which the
            // distance is smooth, and the inward offsets allowed.
            let reach = boundary.inner_reach();
            assert!(reach > 0.5 * w && reach <= w, "w {w}: reach {reach}");
            assert!(boundary.check_offset(-w).is_err());
            // Up through the bar from just above its bottom face.
            let o = [1000.0, 0.0, -w + 0.01 * scale];
            let t = boundary
                .trace(o, [0.0, 0.0, 1.0], 0.0, None, false, f64::INFINITY)
                .crossing;
            assert!(
                (t.unwrap() - (2.0 * w - 0.01 * scale)).abs() < 1e-9,
                "w {w}: {t:?}"
            );
            let mut rng = Rng(5);
            for _ in 0..20 {
                let u = normalize([rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), 1.0]);
                check_crossing(&boundary, 0.0, o, u);
            }
        }
    }
}

#[test]
fn an_axisymmetric_boundary_searches_one_row_for_gaps() {
    // Every row of patches of an axisymmetric boundary is a turned copy of
    // the first, so the first row alone bounds the same gaps as all of
    // them: as tightly or more (the boxes of a turned patch differ), and
    // without filling the search's piece budget, which on a thin surface
    // (a tokamak flux surface deep in the plasma) left pairs too coarse to
    // bound and the reach 0.
    for a in [150.0, 50.0] {
        let boundary = FluxBoundary::new(
            1,
            vec![[0, 0], [1, 0], [2, 0]],
            vec![600.0, a, 0.2 * a],
            vec![[-1, 0]],
            vec![2.5 * a],
        )
        .unwrap();
        let e = &boundary.engine;
        let gaps = |rows| e.patches.gaps(&e.series, rows, e.orientation_3d, e.margin);
        let (one, all) = (gaps(1), gaps(e.patches.nz));
        for (side, one, all) in [("inner", one.0, all.0), ("outer", one.1, all.1)] {
            assert!(
                one >= all && one <= 1.01 * all,
                "a {a}, {side}: one row {one}, all rows {all}"
            );
        }
        assert_eq!(
            boundary.inner_reach(),
            (1.0 / e.kappa_convex).min(0.5 * one.0)
        );
    }
}

#[test]
fn an_axisymmetric_boundary_forms_its_side_test_polyline_once() {
    // The same boundary with a zero toroidal mode takes the general path,
    // which forms the cut through every query point; the side test must
    // not tell them apart, to the bit.
    let modes = vec![[0, 0], [1, 0], [2, 0]];
    let axisymmetric = FluxBoundary::new(
        1,
        modes.clone(),
        vec![600.0, 150.0, 30.0],
        vec![[-1, 0], [-2, 0]],
        vec![250.0, 12.0],
    )
    .unwrap();
    let general = FluxBoundary::new(
        1,
        [modes, vec![[1, 1]]].concat(),
        vec![600.0, 150.0, 30.0, 0.0],
        vec![[-1, 0], [-2, 0]],
        vec![250.0, 12.0],
    )
    .unwrap();
    assert!(!axisymmetric.engine.poly_fixed.is_empty());
    assert!(general.engine.poly_fixed.is_empty());
    let mut rng = Rng(23);
    for _ in 0..500 {
        let p = [
            rng.range(-900.0, 900.0),
            rng.range(-900.0, 900.0),
            rng.range(-300.0, 300.0),
        ];
        let (a, b) = (axisymmetric.engine.in_plane(p), general.engine.in_plane(p));
        assert_eq!(
            [a.sign, a.upper, a.theta].map(f64::to_bits),
            [b.sign, b.upper, b.theta].map(f64::to_bits),
            "{p:?}"
        );
    }
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

#[test]
fn shallow_crossings_far_from_the_boundary_are_found() {
    // Rays the first version of the march lost: a 60 cm vacuum layer met
    // at about 4 degrees (it ran out of evaluations and lost the particle)
    // and a long shallow pass over the bean.
    let re = rotating_ellipse();
    let o = [503.5718055698, 8.117318195608, -150.4026246763];
    let u = normalize([0.7448347420947, -0.6642340783366, -0.06335847334877]);
    let t = re
        .trace(o, u, 60.0, None, true, f64::INFINITY)
        .crossing
        .unwrap();
    assert!((t - 35.00135837).abs() < 1e-6, "{t}");
    let cases = [
        (
            [589.7641814328, -161.6222524287, 147.9368027346],
            [0.511163065, 0.8592822908, 0.0186092164],
        ),
        (
            [676.228706698, -17.10159508002, -116.8618300965],
            [-0.2121457550, 0.9576837931, -0.1945146039],
        ),
        (
            [710.7376564341, 260.5423185976, -18.87414039906],
            [-0.5184685487, 0.3333175212, -0.7874578047],
        ),
        (
            [682.015489043, 61.38201297862, -110.7562139607],
            [-0.4881572650, 0.7686358709, -0.4134022043],
        ),
        (
            [83.66857412838, -484.8161908679, 166.5168101111],
            [0.5823117902, -0.8110722038, -0.0554514135],
        ),
    ];
    for (o, u) in cases {
        for offset in [20.0, 60.0] {
            check_crossing(&re, offset, o, normalize(u));
        }
    }
    let be = bean();
    let o = [-410.21558722345276, 303.055408290895, 151.3971638744561];
    let u = normalize([0.7404007893171879, 0.6522989420850496, -0.1622120875064814]);
    let t = be
        .trace(o, u, 0.0, None, false, f64::INFINITY)
        .crossing
        .unwrap();
    assert!((t - 282.8767171).abs() < 1e-6, "{t}");
}

#[test]
fn grazing_rays_find_their_crossing() {
    // Rays along the tangent plane of a level set, just inside or outside
    // it, 50 cm before the tangent point.
    let mut worst = 0;
    for boundary in [rotating_ellipse(), bean()] {
        for offset in [0.0, 15.0, 60.0, -10.0] {
            let mut rng = Rng(77);
            for height in [1e-7, -1e-7, 1e-3, -1e-3, 0.1, -0.1, 1.0, -1.0] {
                for _ in 0..4 {
                    let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
                    let sp = boundary.engine.series.point(theta, zeta);
                    let n = boundary.normal(theta, zeta);
                    let t = normalize(add(sp.x_t, sp.x_z, rng.range(-1.0, 1.0)));
                    let dot = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
                    let t = normalize(add(t, n, -dot));
                    let o = add(add(add(sp.x, n, offset), n, height), t, -50.0);
                    worst = worst.max(check_crossing(&boundary, offset, o, t));
                }
            }
        }
    }
    assert!(worst < 1000, "a grazing ray took {worst} evaluations");
}

#[test]
fn directions_are_normalised_and_checked() {
    let boundary = rotating_ellipse();
    let o = [503.5718055698, 8.117318195608, -150.4026246763];
    let u = normalize([0.7448347420947, -0.6642340783366, -0.06335847334877]);
    let unit = boundary
        .trace(o, u, 60.0, None, false, f64::INFINITY)
        .crossing
        .unwrap();
    let long = boundary
        .trace(o, u.map(|v| 7.0 * v), 60.0, None, false, f64::INFINITY)
        .crossing
        .unwrap();
    assert!((7.0 * long - unit).abs() < 1e-9 * unit, "{unit} {long}");
    let none = |o: [f64; 3], u: [f64; 3]| {
        boundary
            .trace(o, u, 0.0, None, false, f64::INFINITY)
            .crossing
    };
    assert_eq!(none(o, [0.0; 3]), None);
    assert_eq!(none(o, [f64::NAN, 0.0, 1.0]), None);
    assert_eq!(none([f64::INFINITY, 0.0, 0.0], u), None);
    // From far away the march ends (and finds the crossing).
    let t = none([1e9, 0.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
    assert!((t - (1e9 - 800.0)).abs() < 1e-3, "{t}");
}

#[test]
fn field_periods_are_limited() {
    let with = |nfp| {
        FluxBoundary::new(
            nfp,
            vec![[0, 0], [1, 0]],
            vec![600.0, 100.0],
            vec![[-1, 0]],
            vec![100.0],
        )
    };
    assert!(with(256).is_ok());
    let err = with(257).unwrap_err();
    assert!(err.contains("between 1 and 256"), "{err}");
}

#[test]
fn loaded_layers_share_one_boundary() {
    let base = Surface::new_flux(Arc::new(rotating_ellipse()), Some(1), None);
    let layers = vec![
        base.clone(),
        base.offset(15.0, Some(2), None).unwrap(),
        base.offset(60.0, Some(3), Some(BoundaryType::Vacuum))
            .unwrap(),
    ];
    let json = serde_json::to_string(&layers).unwrap();
    let back: Vec<Surface> = serde_json::from_str(&json).unwrap();
    let first = match &back[0].kind {
        SurfaceKind::Flux { boundary, .. } => Arc::clone(boundary),
        _ => unreachable!(),
    };
    for layer in &back[1..] {
        match &layer.kind {
            SurfaceKind::Flux { boundary, .. } => assert!(Arc::ptr_eq(boundary, &first)),
            _ => unreachable!(),
        }
    }
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
}

/// A digest of the ray search's answers (crossing, whether the budget ran
/// out, and evaluations, bit for bit) over a fixed set of queries: random
/// rays, capped ones, starts on the surface with and without the transport
/// hint, and grazing ones, on transmission and vacuum surfaces. A change
/// meant to leave the search alone, here or in the search shared with the
/// other free-form kinds, keeps it (`cargo test -p yamc-geo
/// flux::tests::ray_digest -- --ignored --nocapture`, before and after; it
/// takes minutes); libm differences between platforms change it.
#[test]
#[ignore]
fn ray_digest() {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut queries = 0_usize;
    // Random rays per offset: past its fold the spindle torus's search
    // often runs out of evaluations, which costs seconds on a vacuum surface.
    let cases = [
        (
            circular_torus(600.0, 150.0),
            vec![(0.0, 150), (15.0, 150), (-20.0, 150), (500.0, 8)],
        ),
        (
            rotating_ellipse(),
            vec![(0.0, 150), (15.0, 150), (60.0, 150), (-10.0, 150)],
        ),
        (bean(), vec![(0.0, 150), (15.0, 150), (-10.0, 150)]),
        (
            dumbbell(1000.0, 300.0, 3.0, 100.0),
            vec![(0.0, 150), (2.0, 150), (-1.0, 150)],
        ),
    ];
    for (boundary, offsets) in cases {
        let e = &boundary.engine;
        let (lo, hi) = boundary.bounding_box(100.0);
        let mut rng = Rng(2026);
        for (offset, random) in offsets {
            let clock = std::time::Instant::now();
            let mut evals = 0_usize;
            let mut worst = 0_usize;
            let mut record = |o: [f64; 3], u: [f64; 3], hint: Option<f64>, t_max: f64| {
                for vacuum in [false, true] {
                    let trace = e.trace_ray(o, u, offset, hint, vacuum, t_max);
                    let bits = [
                        trace.crossing.map_or(u64::MAX, f64::to_bits),
                        u64::from(trace.exhausted),
                        trace.evaluations as u64,
                    ];
                    for byte in bits.iter().flat_map(|b| b.to_le_bytes()) {
                        digest = (digest ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                    }
                    queries += 1;
                    evals += trace.evaluations;
                    worst = worst.max(trace.evaluations);
                }
            };
            for _ in 0..random {
                let o = [
                    rng.range(lo[0], hi[0]),
                    rng.range(lo[1], hi[1]),
                    rng.range(lo[2], hi[2]),
                ];
                let u = rng.direction();
                record(o, u, None, f64::INFINITY);
                record(o, u, None, rng.range(0.0, 300.0));
            }
            for _ in 0..60 {
                let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
                let x = boundary.point(theta, zeta);
                let n = boundary.normal(theta, zeta);
                let s = rng.range(-3e-8, 3e-8);
                let o = add(x, n, offset + s);
                let u = rng.direction();
                record(o, u, None, f64::INFINITY);
                record(o, u, Some(s.abs()), f64::INFINITY);
            }
            for k in 0..30 {
                let (theta, zeta) = (rng.range(0.0, TAU), rng.range(0.0, TAU));
                let sp = e.series.point(theta, zeta);
                let n = boundary.normal(theta, zeta);
                let t = normalize(add(sp.x_t, sp.x_z, rng.range(-1.0, 1.0)));
                let dot = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
                let t = normalize(add(t, n, -dot));
                let height = [1e-7, -1e-7, 1e-3, -1e-3, 0.5, -0.5][k % 6];
                let o = add(add(add(sp.x, n, offset), n, height), t, -50.0);
                record(o, t, None, f64::INFINITY);
            }
            println!(
                "offset {offset}: {:.1} s, {evals} evals, worst {worst}",
                clock.elapsed().as_secs_f64()
            );
        }
    }
    println!("{queries} queries, digest {digest:016x}");
}

/// Rough per-call costs (run in release: `cargo test --release -p yamc-geo
/// flux::tests::timing -- --ignored --nocapture`).
#[test]
#[ignore]
fn timing() {
    use std::time::Instant;
    for (name, boundary) in [
        ("circular torus", circular_torus(600.0, 150.0)),
        ("rotating ellipse", rotating_ellipse()),
        ("bean", bean()),
    ] {
        let base = Surface::new_flux(Arc::new(boundary), None, None);
        for d in [0.0, 15.0] {
            let s = base.offset(d, None, None).unwrap();
            let mut rng = Rng(99);
            let pts: Vec<(f64, f64, f64)> = (0..2000)
                .map(|_| {
                    (
                        rng.range(-900.0, 900.0),
                        rng.range(-900.0, 900.0),
                        rng.range(-300.0, 300.0),
                    )
                })
                .collect();
            let rays: Vec<([f64; 3], [f64; 3])> = (0..500)
                .map(|_| {
                    (
                        [
                            rng.range(-900.0, 900.0),
                            rng.range(-900.0, 900.0),
                            rng.range(-300.0, 300.0),
                        ],
                        rng.direction(),
                    )
                })
                .collect();
            let t = Instant::now();
            let mut acc = 0.0;
            for &p in &pts {
                acc += s.evaluate(p);
            }
            let eval_us = t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64;
            let t = Instant::now();
            for &p in &pts {
                acc += s.evaluate_exact(p);
            }
            let exact_us = t.elapsed().as_secs_f64() * 1e6 / pts.len() as f64;
            let t = Instant::now();
            let mut hits = 0;
            for &(o, u) in &rays {
                if s.distance_to_surface(o, u).is_some() {
                    hits += 1;
                }
            }
            let ray_us = t.elapsed().as_secs_f64() * 1e6 / rays.len() as f64;
            println!("{name:17} offset {d:4}: evaluate {eval_us:8.2} us, exact {exact_us:8.2} us, ray {ray_us:9.2} us ({hits} hits) [{acc:.1}]");
        }
    }
}
