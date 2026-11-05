//! Cost of flux and offset surfaces against closed-form surfaces.
//!
//! Per call of `Surface::evaluate` (the sign test of cell finding) and
//! `Surface::distance_to_surface` (the ray crossing of surface tracking),
//! for a sphere and a ZTorus (closed form) and for flux surfaces of
//! increasing mode count: a circular torus (2 modes), the rotating
//! ellipse of the stellarator source example (7) and a synthetic
//! stellarator boundary with modes up to m = n = 12 (about 300 R and Z
//! coefficients), each as the boundary and as a 15 cm offset layer.
//!
//! Run with:
//!     cargo bench -p yamc --bench flux_surface
//!
//! Each sample evaluates `CALLS` points or rays; divide criterion's time
//! by it for the per-call cost.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Arc;
use yamc::geo::{FluxBoundary, Surface};

const CALLS: usize = 200;

fn circular_torus() -> FluxBoundary {
    FluxBoundary::new(
        1,
        vec![[0, 0], [1, 0]],
        vec![600.0, 150.0],
        vec![[-1, 0]],
        vec![150.0],
    )
    .unwrap()
}

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

/// The rotating ellipse plus small stellarator-symmetric harmonics up to
/// m = n = 12, decaying as 0.5^(|m| + |n|) (cm).
fn synthetic_stellarator() -> FluxBoundary {
    let mut rng = StdRng::seed_from_u64(3);
    let (mut r_modes, mut r_mn) = (
        vec![[0, 0], [1, 0], [1, 1], [-1, -1]],
        vec![600.0, 150.0, 50.0, 50.0],
    );
    let (mut z_modes, mut z_mn) = (vec![[-1, 0], [-1, 1], [1, -1]], vec![150.0, -50.0, 50.0]);
    for m in 0..=12_i32 {
        for n in -12..=12_i32 {
            if m == 0 && n <= 0 {
                continue;
            }
            let amplitude = 4.0 * 0.5_f64.powi(m + n.abs());
            if amplitude < 1e-4 {
                continue;
            }
            // Stellarator symmetry: R takes cos(m t) cos(n N z) and
            // sin sin terms, Z the cross ones.
            let (rc, rs, zc, zs) = (
                rng.random_range(-1.0..1.0) * amplitude,
                rng.random_range(-1.0..1.0) * amplitude,
                rng.random_range(-1.0..1.0) * amplitude,
                rng.random_range(-1.0..1.0) * amplitude,
            );
            for (is_r, mode, value) in [
                (true, [m, n.abs()], rc),
                (true, [-m, -n.abs()], rs),
                (false, [-m, n.abs()], zc),
                (false, [m, -n.abs()], zs),
            ] {
                let (modes, values) = if is_r {
                    (&mut r_modes, &mut r_mn)
                } else {
                    (&mut z_modes, &mut z_mn)
                };
                if !modes.contains(&mode) {
                    modes.push(mode);
                    values.push(value);
                }
            }
        }
    }
    FluxBoundary::new(5, r_modes, r_mn, z_modes, z_mn).unwrap()
}

fn cases() -> Vec<(&'static str, Surface)> {
    let flux = |name: &'static str, b: FluxBoundary| {
        let base = Surface::new_flux(Arc::new(b), None, None);
        let layer = base.offset(15.0, None, None).unwrap();
        [(name, base), (name, layer)]
    };
    let mut out = vec![
        (
            "sphere",
            Surface::new_sphere(0.0, 0.0, 0.0, 800.0, None, None),
        ),
        (
            "ztorus",
            Surface::new_ztorus(0.0, 0.0, 0.0, 600.0, 150.0, 150.0, None, None),
        ),
    ];
    let [a, b] = flux("flux_torus_2_modes", circular_torus());
    out.extend([a, ("flux_torus_2_modes+15cm", b.1)]);
    let [a, b] = flux("rotating_ellipse_7_modes", rotating_ellipse());
    out.extend([a, ("rotating_ellipse_7_modes+15cm", b.1)]);
    let synthetic = synthetic_stellarator();
    let modes = synthetic.r_modes().len() + synthetic.z_modes().len();
    eprintln!("synthetic stellarator: {modes} modes");
    let [a, b] = flux("synthetic_stellarator", synthetic);
    out.extend([a, ("synthetic_stellarator+15cm", b.1)]);
    out
}

fn points(seed: u64) -> Vec<(f64, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..CALLS)
        .map(|_| {
            (
                rng.random_range(-900.0..900.0),
                rng.random_range(-900.0..900.0),
                rng.random_range(-300.0..300.0),
            )
        })
        .collect()
}

fn rays(seed: u64) -> Vec<([f64; 3], [f64; 3])> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..CALLS)
        .map(|_| {
            let o = [
                rng.random_range(-900.0..900.0),
                rng.random_range(-900.0..900.0),
                rng.random_range(-300.0..300.0),
            ];
            let mu: f64 = rng.random_range(-1.0..1.0);
            let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let s = (1.0 - mu * mu).sqrt();
            (o, [s * phi.cos(), s * phi.sin(), mu])
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let cases = cases();
    let (pts, rs) = (points(1), rays(2));
    let mut group = c.benchmark_group("evaluate");
    group.sample_size(20);
    for (name, surface) in &cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), surface, |b, s| {
            b.iter(|| {
                for &p in &pts {
                    black_box(s.evaluate(black_box(p)));
                }
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("distance_to_surface");
    group.sample_size(10);
    for (name, surface) in &cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), surface, |b, s| {
            b.iter(|| {
                for &(o, u) in &rs {
                    black_box(s.distance_to_surface(black_box(o), black_box(u)));
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
