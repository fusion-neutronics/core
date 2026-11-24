//! Particles/sec benchmark for the composed full-step kernel
//! (XS lookup + free flight + MT sample + scatter/absorb).
//!
//! This is the most realistic transport-shape kernel we have so far.
//! Three RNG samples per particle, two ln_polyfill calls (one for
//! log-energy, one for the free-flight sample), one sqrt, a binary
//! search, divides, and a branch. Compared against single-thread CPU
//! and rayon-parallel CPU at 1M particles.
//!
//! Run with:
//!     cargo bench -p yamc-gpu --bench full_step

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use yamc_gpu::neutron::probes::full_step::{run_full_step, run_full_step_cpu};
use yamc_gpu::{GpuContext, GpuInitError};

const N: usize = 1_000_000;
const TARGET_MASS: f64 = 12.0;

struct Workload {
    seeds: Vec<u32>,
    energies: Vec<f64>,
    log_grid: Vec<f64>,
    xs_e: Vec<f64>,
    xs_a: Vec<f64>,
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
    let energies: Vec<f64> = (0..N)
        .map(|i| {
            let frac = (i as f64 + 0.5) / N as f64;
            (log_e_min + 0.05 + (log_e_max - 0.05 - log_e_min) * frac).exp()
        })
        .collect();
    Workload {
        seeds,
        energies,
        log_grid,
        xs_e,
        xs_a,
    }
}

fn run_cpu_rayon(
    seeds: &[u32],
    energies: &[f64],
    log_grid: &[f64],
    xs_e: &[f64],
    xs_a: &[f64],
    target_mass: f64,
) {
    use rayon::prelude::*;
    use yamc_gpu::common::rng::{expand_seed, pcg_xsh_rr, PCG_INCR, PCG_MULT};
    let n = seeds.len();
    let _: Vec<()> = (0..n)
        .into_par_iter()
        .map(|i| {
            let s0 = expand_seed(seeds[i]);
            let r1 = pcg_xsh_rr(s0);
            let s1 = s0.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r2 = pcg_xsh_rr(s1);
            let s2 = s1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r3 = pcg_xsh_rr(s2);

            let xi1 = (r1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let xi2 = (r2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let xi3 = (r3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

            let log_e = energies[i].ln();
            let lo = log_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, log_grid.len() - 1);
            let idx_lo = idx_hi - 1;
            let frac = (log_e - log_grid[idx_lo]) / (log_grid[idx_hi] - log_grid[idx_lo]);
            let sigma_e = xs_e[idx_lo] + (xs_e[idx_hi] - xs_e[idx_lo]) * frac;
            let sigma_a = xs_a[idx_lo] + (xs_a[idx_hi] - xs_a[idx_lo]) * frac;
            let sigma_t = sigma_e + sigma_a;

            let _distance = -xi1.ln() / sigma_t;
            let p_elastic = sigma_e / sigma_t;
            if xi2 < p_elastic {
                let mu_cm = 1.0 - 2.0 * xi3;
                let denom = (target_mass + 1.0) * (target_mass + 1.0);
                let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
                let _e_out = energies[i] * numer / denom;
                let _mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();
            }
        })
        .collect();
}

fn bench_full_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_step_1M");
    group.throughput(Throughput::Elements(N as u64));
    let w = build_workload();

    group.bench_function("cpu_seq", |b| {
        b.iter(|| {
            run_full_step_cpu(
                &w.seeds,
                &w.energies,
                &w.log_grid,
                &w.xs_e,
                &w.xs_a,
                TARGET_MASS,
            )
        });
    });

    group.bench_function("cpu_rayon", |b| {
        b.iter(|| {
            run_cpu_rayon(
                &w.seeds,
                &w.energies,
                &w.log_grid,
                &w.xs_e,
                &w.xs_a,
                TARGET_MASS,
            )
        });
    });

    match GpuContext::new() {
        Ok(ctx) => {
            group.bench_function("gpu", |b| {
                b.iter(|| {
                    run_full_step(
                        &ctx,
                        &w.seeds,
                        &w.energies,
                        &w.log_grid,
                        &w.xs_e,
                        &w.xs_a,
                        TARGET_MASS,
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

criterion_group!(benches, bench_full_step);
criterion_main!(benches);
