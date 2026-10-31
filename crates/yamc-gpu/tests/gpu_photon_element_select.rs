//! Per-mechanism test for the photon element-selection walk extracted into
//! `common::probes::photon_element_select`. Sweeps PCG seeds across a fixed
//! two-element material's per-element macroscopic totals and checks the GPU
//! `select_photon_element` matches the `select_photon_element_cpu` 32-bit-PCG
//! twin bit-for-bit, and that the empirical selection fractions match the
//! macro-XS weights (mirrors CPU `Material::sample_element`).
//!
//!   cargo test -p yamc-gpu --release --test gpu_photon_element_select -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use yamc_gpu::common::probes::photon_element_select::{
    run_photon_element_select, select_photon_element_cpu,
};
use yamc_gpu::{GpuContext, GpuInitError};

#[test]
fn gpu_photon_element_select_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // Two comparable-Z elements (Pb+W-ish macroscopic totals at ~8 MeV): a
    // 60/40 split in macro-XS contribution.
    let xs = [0.0312_f64, 0.0208_f64];
    let total: f64 = xs.iter().sum();

    // One sample per seed; all share the same two-element row (offset 0,
    // count 2, single-point grid so the probe collapses interpolation).
    let n = 200_000usize;
    let xs_flat: Vec<f64> = (0..n).flat_map(|_| xs.iter().copied()).collect();
    let offsets: Vec<u32> = (0..n as u32).map(|i| i * 2).collect();
    let counts: Vec<u32> = vec![2u32; n];
    let seeds: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();

    let gpu = run_photon_element_select(&ctx, &xs_flat, &offsets, &counts, &seeds);

    let mut mismatches = 0usize;
    let mut hist = [0usize; 2];
    for (i, &seed) in seeds.iter().enumerate() {
        let cpu = select_photon_element_cpu(&xs, yamc_gpu::common::rng::expand_seed(seed));
        if gpu[i] != cpu {
            mismatches += 1;
        }
        hist[gpu[i] as usize] += 1;
    }
    let denom = n as f64;
    println!(
        "photon element select: {mismatches} mismatches / {n} | GPU fractions \
         e0={:.3} e1={:.3} (expected {:.3}/{:.3})",
        hist[0] as f64 / denom,
        hist[1] as f64 / denom,
        xs[0] / total,
        xs[1] / total,
    );
    assert_eq!(
        mismatches, 0,
        "GPU element selection disagrees with CPU twin"
    );
    // Empirical fractions match the macro-XS weights to MC noise.
    assert!((hist[0] as f64 / denom - xs[0] / total).abs() < 0.005);
    assert!((hist[1] as f64 / denom - xs[1] / total).abs() < 0.005);
}
