//! Per-mechanism GPU-vs-CPU test for the incoherent scattering function
//! S(x, Z) lookup extracted into `kernels::incoherent_sf`. Deterministic
//! (no RNG): evaluates the GPU `incoherent_s_at` at a sweep of momentum
//! transfers and checks it matches the CPU
//! `incoherent_form_factor.evaluate(x)` on the same table.
//!
//!   cargo test -p yamc-gpu --release --test gpu_incoherent_sf -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use std::path::Path;

use yamc_element::photon_arrow::read_photon_interaction_from_arrow;
use yamc_gpu::common::probes::incoherent_sf::run_incoherent_s;
use yamc_gpu::{GpuContext, GpuInitError};
use yamc_nuclide::reaction_product::Tabulated1D;

#[test]
fn gpu_incoherent_s_matches_cpu_evaluate() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let fe = read_photon_interaction_from_arrow(Path::new("../yamc/tests/Fe.arrow"))
        .expect("load Fe.arrow photon data");
    let (iff_x, iff_s) = match &fe.incoherent_form_factor {
        Tabulated1D::Tabulated1D { x, y, .. } => (x.clone(), y.clone()),
    };
    let n_pts = iff_x.len();
    assert!(n_pts >= 2, "Fe incoherent form factor too short");
    let xmin = iff_x[0];
    let xmax = iff_x[n_pts - 1];

    // Query x: a dense sweep across the table, plus the exact table nodes and
    // below/above-range points (clamp behaviour).
    let mut xq: Vec<f64> = Vec::new();
    for i in 0..=500 {
        xq.push(xmin + (xmax - xmin) * (i as f64 / 500.0));
    }
    xq.extend_from_slice(&iff_x); // exact nodes
    xq.push(xmin - (xmax - xmin)); // below range -> clamp to first
    xq.push(xmax + (xmax - xmin)); // above range -> clamp to last

    let gpu_s = run_incoherent_s(&ctx, &xq, &iff_x, &iff_s);

    let mut max_rel = 0.0_f64;
    for (i, &x) in xq.iter().enumerate() {
        let cpu = match &fe.incoherent_form_factor {
            Tabulated1D::Tabulated1D { .. } => fe.incoherent_form_factor.evaluate(x),
        };
        let g = gpu_s[i];
        assert!(g.is_finite(), "GPU S non-finite at x={x}");
        let denom = cpu.abs().max(1e-30);
        let rel = (g - cpu).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    println!(
        "incoherent S(x): max rel |GPU-CPU| over {} queries = {max_rel:.3e}",
        xq.len()
    );
    assert!(
        max_rel < 1e-9,
        "GPU incoherent_s_at disagrees with CPU evaluate (max rel {max_rel:.3e})"
    );
}
