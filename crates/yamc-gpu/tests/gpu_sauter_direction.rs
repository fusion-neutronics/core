//! Per-mechanism GPU-vs-CPU test for the photoelectron emission direction
//! extracted into `probes::sauter_direction`. Checks the GPU sampler's polar
//! cosine distribution against the CPU
//! `yamc_physics::photon::photoelectron::sample_photoelectron_direction` the
//! transport path runs, and checks the direction it builds from that cosine.
//!
//! The polar axis is the incident photon's direction, on both, so `mu` is the
//! direction's component along it and the rest carries the azimuth. The
//! incident direction here is off every axis, so a sampler that took `mu`
//! about lab +x (as OpenMC does), or that gave the photoelectron the parent
//! photon's direction, cannot produce this distribution.
//!
//!   cargo test -p yamc-gpu --release --test gpu_sauter_direction -- --nocapture

#![cfg(not(target_os = "macos"))]

use rand::SeedableRng;
use yamc_gpu::common::probes::sauter_direction::run_sauter_direction;
use yamc_gpu::{GpuContext, GpuInitError};
use yamc_physics::photon::photoelectron::sample_photoelectron_direction;

const N: usize = 400_000;

/// An incident photon direction off every lab axis.
fn incident() -> [f64; 3] {
    let v = [0.3_f64, -0.5, 0.7];
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn along(d: &[f64; 3], u: &[f64; 3]) -> f64 {
    d[0] * u[0] + d[1] * u[1] + d[2] * u[2]
}

/// Histogram `mu` over 20 uniform bins on [-1, 1].
fn mu_histogram(mus: impl Iterator<Item = f64>) -> [f64; 20] {
    let mut h = [0.0f64; 20];
    let mut n = 0.0;
    for mu in mus {
        let b = (((mu + 1.0) * 0.5 * 20.0) as usize).min(19);
        h[b] += 1.0;
        n += 1.0;
    }
    for v in &mut h {
        *v /= n;
    }
    h
}

#[test]
fn gpu_sauter_direction_matches_cpu_distribution() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // Kinetic energies spanning the regime a photoelectron lands in: a few keV
    // (forward-ish but broad) up to 5 MeV (sharply forward). The Sauter shape
    // moves a long way across this range, so matching at every point is a real
    // constraint on the sampler, not just on its mean.
    for &ke in &[1.0e3_f64, 5.0e4, 5.0e5, 5.0e6] {
        let seeds: Vec<u32> = (0..N as u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let u = incident();
        let gpu = run_sauter_direction(&ctx, &seeds, ke, u);
        assert_eq!(gpu.len(), N);

        let mut rng = rand::rngs::StdRng::seed_from_u64(20260918);
        let cpu: Vec<[f64; 3]> = (0..N)
            .map(|_| sample_photoelectron_direction(ke, u, &mut rng))
            .collect();

        // Every direction is a unit vector, and the polar cosine is its
        // component along the incident direction.
        for d in gpu.iter().take(1000) {
            let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "ke {ke}: GPU direction not a unit vector: {d:?} (|d| = {norm})"
            );
        }

        let g = mu_histogram(gpu.iter().map(|d| along(d, &u)));
        let c = mu_histogram(cpu.iter().map(|d| along(d, &u)));
        // 1/sqrt(N) on the fullest bin is ~0.2% here; 1.5% absolute per bin is
        // ~7 sigma of headroom and still far tighter than any plausible wrong
        // distribution (a parent-direction regression puts everything in one
        // bin).
        for b in 0..20 {
            assert!(
                (g[b] - c[b]).abs() < 0.015,
                "ke {ke}: mu bin {b} GPU {:.4} vs CPU {:.4}\nGPU {g:?}\nCPU {c:?}",
                g[b],
                c[b]
            );
        }

        let g_mean: f64 = gpu.iter().map(|d| along(d, &u)).sum::<f64>() / N as f64;
        let c_mean: f64 = cpu.iter().map(|d| along(d, &u)).sum::<f64>() / N as f64;
        assert!(
            (g_mean - c_mean).abs() < 0.01,
            "ke {ke}: mean mu GPU {g_mean:.5} vs CPU {c_mean:.5}"
        );
        // The azimuth is uniform about the incident direction, so the mean
        // direction is <mu> u, with nothing transverse; <sin^2 theta> carries
        // the rest. Guards the rotation, which the mu histogram alone says
        // nothing about.
        for (side, dirs, mean) in [("GPU", &gpu, g_mean), ("CPU", &cpu, c_mean)] {
            for k in 0..3 {
                let m: f64 = dirs.iter().map(|d| d[k]).sum::<f64>() / N as f64;
                assert!(
                    (m - mean * u[k]).abs() < 0.01,
                    "ke {ke}: {side} mean direction component {k} {m:.5}, expected {:.5}",
                    mean * u[k]
                );
            }
        }
        let perp = |dirs: &[[f64; 3]]| {
            dirs.iter().map(|d| 1.0 - along(d, &u).powi(2)).sum::<f64>() / N as f64
        };
        let (perp_gpu, perp_cpu) = (perp(&gpu), perp(&cpu));
        assert!(
            (perp_gpu - perp_cpu).abs() < 0.01,
            "ke {ke}: <sin^2 theta> GPU {perp_gpu:.5} vs CPU {perp_cpu:.5}"
        );

        println!(
            "ke {ke:>10.0} eV: <mu> GPU {g_mean:.5} CPU {c_mean:.5}, \
             <sin^2> GPU {perp_gpu:.5} CPU {perp_cpu:.5}"
        );
    }
}
