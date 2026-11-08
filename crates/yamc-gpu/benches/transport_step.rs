//! Particles/sec benchmark for the combined transport-step kernel
//! (XS lookup + free-flight sampling). First "realistic transport
//! step" number for the GPU port.
//!
//! Same shape as `distance_to_collision.rs`: 1M particles per launch,
//! GPU end-to-end vs CPU sequential vs CPU rayon. The XS table is a
//! 100-point log-spaced grid sampling `Σ(E) = 2 + sin(ln E)` between
//! `E = 1e-3` and `E = 1e3`, with each particle at a distinct energy.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench transport_step

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::neutron::probes::transport_step::{
    run_transport_step, run_transport_step_cpu, run_transport_step_cpu_rayon,
};
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;

fn build_workload() -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_grid = 100usize;
    let log_e_min = (1e-3_f64).ln();
    let log_e_max = (1e3_f64).ln();
    let log_energy_grid: Vec<f64> = (0..n_grid)
        .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
        .collect();
    let xs_values: Vec<f64> = log_energy_grid.iter().map(|&le| 2.0 + le.sin()).collect();

    let seeds: Vec<u32> = (0..N)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    let energies: Vec<f64> = (0..N)
        .map(|i| {
            let frac = (i as f64 + 0.5) / N as f64;
            (log_e_min + 0.01 + (log_e_max - 0.01 - log_e_min) * frac).exp()
        })
        .collect();

    (seeds, energies, log_energy_grid, xs_values)
}

fn bench_transport_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport_step_1M");
    group.throughput(Throughput::Elements(N as u64));
    let (seeds, energies, log_grid, xs_values) = build_workload();

    group.bench_function("cpu_seq", |b| {
        b.iter(|| run_transport_step_cpu(&seeds, &energies, &log_grid, &xs_values));
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| run_transport_step_cpu_rayon(&seeds, &energies, &log_grid, &xs_values));
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| run_transport_step(&ctx, &seeds, &energies, &log_grid, &xs_values));
            });
        }
        Err(GpuInitError::NoF64Adapter) => {
            eprintln!("skipping GPU bench: no Vulkan f64 adapter on this host");
        }
    }

    group.finish();
}

criterion_group!(benches, bench_transport_step);
criterion_main!(benches);
