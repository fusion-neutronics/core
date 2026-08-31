//! Per-mechanism test for the photon interaction-selection extracted into
//! `kernels::photon_select`. Sweeps `cutoff` across `[0, sigma_total)` and
//! checks the GPU `select_photon_reaction` matches a CPU mirror of the
//! cumulative selection in `yamc::transport::handle_photon_collision`
//! (exact, since it's deterministic given the cutoff).
//!
//!   cargo test -p yamc-gpu --release --test gpu_photon_select -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use yamc_gpu::common::probes::photon_select::run_photon_select;
use yamc_gpu::{GpuContext, GpuInitError};

/// CPU mirror of `handle_photon_collision`'s cumulative reaction pick:
/// return the first channel whose running cumulative exceeds `cutoff`.
fn cpu_select(cutoff: f64, coh: f64, inc: f64, photo: f64) -> u32 {
    let mut prob = 0.0_f64;
    prob += coh;
    if prob > cutoff {
        return 0;
    }
    prob += inc;
    if prob > cutoff {
        return 1;
    }
    prob += photo;
    if prob > cutoff {
        return 2;
    }
    3
}

#[test]
fn gpu_photon_select_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // Representative Fe-ish partial macroscopic XS (cm^-1) around ~1 MeV.
    let coh = 1.56e-2_f64;
    let inc = 5.25e-1_f64;
    let photo = 1.29e-1_f64;
    let pair = 4.6e-4_f64;
    let sigma_t = coh + inc + photo + pair;

    // Dense cutoff sweep across [0, sigma_t), including the channel
    // boundaries (where an off-by-epsilon would flip the choice).
    let n = 200_000usize;
    let mut cutoffs: Vec<f64> = (0..n)
        .map(|i| sigma_t * (i as f64 + 0.5) / n as f64)
        .collect();
    // Add the exact cumulative boundaries.
    cutoffs.push(coh);
    cutoffs.push(coh + inc);
    cutoffs.push(coh + inc + photo);

    let gpu_kind = run_photon_select(&ctx, &cutoffs, coh, inc, photo);

    let mut mismatches = 0usize;
    let mut hist = [0usize; 4];
    for (i, &cut) in cutoffs.iter().enumerate() {
        let cpu = cpu_select(cut, coh, inc, photo);
        if gpu_kind[i] != cpu {
            mismatches += 1;
        }
        hist[gpu_kind[i] as usize] += 1;
    }
    let denom = cutoffs.len() as f64;
    println!(
        "photon select: {mismatches} mismatches / {} | GPU fractions coh={:.3} inc={:.3} photo={:.3} pair={:.3} (expected {:.3}/{:.3}/{:.3}/{:.3})",
        cutoffs.len(),
        hist[0] as f64 / denom,
        hist[1] as f64 / denom,
        hist[2] as f64 / denom,
        hist[3] as f64 / denom,
        coh / sigma_t,
        inc / sigma_t,
        photo / sigma_t,
        pair / sigma_t,
    );
    assert_eq!(mismatches, 0, "GPU selection disagrees with CPU mirror");
    // Sanity: the uniform-cutoff fractions match sigma_i / sigma_t.
    assert!((hist[1] as f64 / denom - inc / sigma_t).abs() < 0.005);
}
