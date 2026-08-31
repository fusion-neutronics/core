//! Particles/sec benchmark for the homogeneous-slab distance-to-collision
//! kernel -- first comparable performance number for the GPU port.
//!
//! Three variants on the same workload (`N = 1_000_000` free-flight
//! samples, single launch, single material with constant Σ_total):
//! - `gpu`: end-to-end through `run_distance_to_collision` (upload
//!   seeds + Σ, launch kernel, download distances).
//! - `cpu_seq`: single-threaded CPU baseline using the same PCG-32
//!   algorithm and libm `f64::ln`.
//! - `cpu_rayon`: rayon-parallel CPU baseline, also same algorithm.
//!
//! GPU "kernel only" timing isn't separated out -- `client.read_one`
//! synchronises and dominates measurement noise on small workloads.
//! The end-to-end number is the one that matters for `compute='gpu'`
//! anyway: real transport will need the upload/download path too.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench distance_to_collision

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::common::sampling::distance_to_collision::{
    run_distance_to_collision, run_distance_to_collision_cpu, run_distance_to_collision_cpu_rayon,
};
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;
const SIGMA_TOTAL: f64 = 0.5;

fn make_seeds(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect()
}

fn bench_distance_to_collision(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_to_collision_1M");
    group.throughput(Throughput::Elements(N as u64));
    let seeds = make_seeds(N);

    group.bench_function("cpu_seq", |b| {
        b.iter(|| run_distance_to_collision_cpu(&seeds, SIGMA_TOTAL));
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| run_distance_to_collision_cpu_rayon(&seeds, SIGMA_TOTAL));
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| run_distance_to_collision(&ctx, &seeds, SIGMA_TOTAL));
            });
        }
        Err(GpuInitError::NoF64Adapter) => {
            eprintln!("skipping GPU bench: no Vulkan f64 adapter on this host");
        }
    }

    group.finish();
}

criterion_group!(benches, bench_distance_to_collision);
criterion_main!(benches);
