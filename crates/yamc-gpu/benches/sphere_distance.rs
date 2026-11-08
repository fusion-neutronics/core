//! Particles/sec benchmark for the sphere-distance kernel -- first
//! sqrt-heavy GPU bench. Sqrt is exact on cubecl-spirv on this driver,
//! so this kernel uses *no* polyfill ops; comparing against the
//! transport-step bench tells us how much of the GPU's relative cost
//! comes from polyfill `ln` vs raw GPU compute.
//!
//! Workload: 1M particles, each on a hemisphere at random stand-off
//! pointing at the unit sphere at the origin. All rays hit by
//! construction.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench sphere_distance

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::common::probes::sphere_distance::{
    run_sphere_distance, run_sphere_distance_cpu, run_sphere_distance_cpu_rayon,
};
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;

fn build_workload() -> (Vec<f64>, Vec<f64>, [f64; 4]) {
    let mut positions = Vec::with_capacity(N * 3);
    let mut directions = Vec::with_capacity(N * 3);
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        let phi = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt()) * (i as f64);
        let theta = (1.0 - 2.0 * t).acos();
        let standoff = 2.5_f64 + (i as f64 % 3.0);
        let px = standoff * theta.sin() * phi.cos();
        let py = standoff * theta.sin() * phi.sin();
        let pz = standoff * theta.cos();
        let mag = (px * px + py * py + pz * pz).sqrt();
        positions.extend_from_slice(&[px, py, pz]);
        directions.extend_from_slice(&[-px / mag, -py / mag, -pz / mag]);
    }
    let params = [0.0, 0.0, 0.0, 1.0];
    (positions, directions, params)
}

fn bench_sphere_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere_distance_1M");
    group.throughput(Throughput::Elements(N as u64));
    let (positions, directions, params) = build_workload();

    group.bench_function("cpu_seq", |b| {
        b.iter(|| run_sphere_distance_cpu(&positions, &directions, &params));
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| run_sphere_distance_cpu_rayon(&positions, &directions, &params));
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| run_sphere_distance(&ctx, &positions, &directions, &params));
            });
        }
        Err(GpuInitError::NoF64Adapter) => {
            eprintln!("skipping GPU bench: no Vulkan f64 adapter on this host");
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sphere_distance);
criterion_main!(benches);
