//! Histories/sec benchmark for the multi-step transport kernel --
//! the most realistic transport workload we can run on the GPU
//! without geometry. Each particle runs up to `MAX_STEPS` collisions
//! until it's absorbed or hits the cap; "history" = one full particle.
//!
//! With constant Σ_a / Σ_t = 1/3 and `MAX_STEPS = 50`, the expected
//! mean steps per history is ~3 and the absorption tail past the cap
//! is `(2/3)^50 ≈ 1.6e-9` -- every history terminates in practice.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench multi_step

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::neutron::probes::multi_step::{
    run_multi_step, run_multi_step_cpu, run_multi_step_cpu_rayon,
};
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;
const TARGET_MASS: f64 = 12.0;
const MAX_STEPS: u32 = 50;
const N_BINS: usize = 8;

struct Workload {
    seeds: Vec<u32>,
    energies: Vec<f64>,
    directions: Vec<f64>,
    log_grid: Vec<f64>,
    xs_e: Vec<f64>,
    xs_a: Vec<f64>,
    log_e_min: f64,
    log_e_max: f64,
}

fn build_workload() -> Workload {
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let xs_e: Vec<f64> = vec![2.0; n_grid];
    let xs_a: Vec<f64> = vec![1.0; n_grid];

    let seeds: Vec<u32> = (0..N)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = vec![1.0; N];
    let mut directions = Vec::with_capacity(3 * N);
    for _ in 0..N {
        directions.extend_from_slice(&[0.0, 0.0, 1.0]);
    }

    Workload {
        seeds,
        energies,
        directions,
        log_grid,
        xs_e,
        xs_a,
        log_e_min,
        log_e_max,
    }
}

fn bench_multi_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_step_1M");
    group.throughput(Throughput::Elements(N as u64));
    let w = build_workload();

    group.bench_function("cpu_seq", |b| {
        b.iter(|| {
            run_multi_step_cpu(
                &w.seeds,
                &w.energies,
                &w.directions,
                &w.log_grid,
                &w.xs_e,
                &w.xs_a,
                TARGET_MASS,
                w.log_e_min,
                w.log_e_max,
                N_BINS,
                MAX_STEPS,
            )
        });
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| {
            run_multi_step_cpu_rayon(
                &w.seeds,
                &w.energies,
                &w.directions,
                &w.log_grid,
                &w.xs_e,
                &w.xs_a,
                TARGET_MASS,
                w.log_e_min,
                w.log_e_max,
                N_BINS,
                MAX_STEPS,
            )
        });
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| {
                    run_multi_step(
                        &ctx,
                        &w.seeds,
                        &w.energies,
                        &w.directions,
                        &w.log_grid,
                        &w.xs_e,
                        &w.xs_a,
                        TARGET_MASS,
                        w.log_e_min,
                        w.log_e_max,
                        N_BINS,
                        MAX_STEPS,
                    )
                });
            });
        }
        Err(GpuInitError::NoF64Adapter) => {
            eprintln!("skipping GPU bench: no Vulkan f64 adapter on this host");
        }
    }

    group.finish();
}

criterion_group!(benches, bench_multi_step);
criterion_main!(benches);
