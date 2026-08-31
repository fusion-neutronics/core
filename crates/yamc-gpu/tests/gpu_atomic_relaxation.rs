//! Per-mechanism test for the atomic-relaxation per-hop transition sampler
//! extracted into `kernels::atomic_relaxation`. The CPU `atomic_relaxation`
//! only exposes the full cascade, so this checks the GPU sampler's
//! transition-index distribution against the transition CDF it shares with
//! the CPU (built from Fe via `extract_atomic_relaxation_for_gpu`).
//!
//!   cargo test -p yamc-gpu --release --test gpu_atomic_relaxation -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use std::path::Path;
use std::sync::Arc;

use yamc_element::photon_arrow::read_photon_interaction_from_arrow;
use yamc_gpu::common::probes::atomic_relaxation::run_relax_transition;
use yamc_gpu::{extract_atomic_relaxation_for_gpu, MAX_AR_TRANS};
use yamc_gpu::{GpuContext, GpuInitError};

#[test]
fn gpu_relax_transition_matches_cdf() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let fe = Arc::new(
        read_photon_interaction_from_arrow(Path::new("../yamc/tests/Fe.arrow"))
            .expect("load Fe.arrow photon data"),
    );
    // Transition tables are energy-independent; the extractor still needs a
    // non-empty energy grid (for the PE-subshell XS we don't use here).
    let materials = vec![vec![("Fe".to_string(), fe.clone(), 1.0_f64)]];
    let ar = extract_atomic_relaxation_for_gpu(&materials, &fe.energy);

    // Pick the shell (material 0) with the most transitions.
    let stride = MAX_AR_TRANS;
    let mut best_shell = 0usize;
    let mut best_nt = 0u32;
    for (s, &nt) in ar.n_trans.iter().enumerate() {
        if nt > best_nt {
            best_nt = nt;
            best_shell = s;
        }
    }
    assert!(best_nt >= 2, "no Fe shell has >= 2 transitions ({best_nt})");
    let nt = best_nt as usize;
    let off = best_shell * stride;
    let cum_prob = ar.trans_cum_prob[off..off + nt].to_vec();
    let primary = ar.trans_primary[off..off + nt].to_vec();
    let secondary = ar.trans_secondary[off..off + nt].to_vec();
    let energy = ar.trans_energy[off..off + nt].to_vec();

    let n = 400_000usize;
    let seeds: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(17))
        .collect();
    let gpu_tidx = run_relax_transition(&ctx, &seeds, &cum_prob, &primary, &secondary, &energy);

    // Expected per-transition probability from the CDF (what the CPU uses).
    let mut expected = vec![0.0_f64; nt];
    expected[0] = cum_prob[0];
    for i in 1..nt {
        expected[i] = cum_prob[i] - cum_prob[i - 1];
    }

    let mut hist = vec![0usize; nt];
    for &t in &gpu_tidx {
        assert!((t as usize) < nt, "GPU t_idx {t} >= nt {nt}");
        hist[t as usize] += 1;
    }

    let mut max_dev = 0.0_f64;
    for i in 0..nt {
        let g = hist[i] as f64 / n as f64;
        max_dev = max_dev.max((g - expected[i]).abs());
    }
    let gfrac: Vec<String> = hist
        .iter()
        .map(|&h| format!("{:.3}", h as f64 / n as f64))
        .collect();
    let efrac: Vec<String> = expected.iter().map(|p| format!("{p:.3}")).collect();
    println!(
        "shell {best_shell} ({nt} transitions):\n  GPU {gfrac:?}\n  CDF {efrac:?}\n  max |Δ| = {max_dev:.4}"
    );
    assert!(
        max_dev < 0.01,
        "GPU transition distribution differs from CDF by {max_dev:.4} (> 0.01)"
    );
}
