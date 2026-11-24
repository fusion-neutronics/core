//! Per-mechanism GPU-vs-CPU test for the polar (µ, φ) direction rotation
//! extracted into `kernels::rotate_mu_phi` (the formula the photon kernel
//! uses for Compton/Rayleigh/electron/positron/fluorescence directions).
//! Deterministic: sweeps directions (including near-axial, where the
//! broomstick's long scattered tracks live), µ, and φ, and compares
//! component-wise against the CPU reference
//! `yamc_physics::neutron::interaction::rotate_direction_fast`.
//!
//!   cargo test -p yamc-gpu --release --test gpu_rotate_mu_phi -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use yamc_gpu::common::probes::rotate_mu_phi::run_rotate_mu_phi;
use yamc_gpu::{GpuContext, GpuInitError};
use yamc_physics::neutron::interaction::rotate_direction_fast;

#[test]
fn gpu_rotate_mu_phi_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // Direction sweep: polar angles from exactly-axial through equatorial,
    // azimuthally rotated; plus exact poles. Near-axial is the regime the
    // broomstick scattered-flux deficit lives in.
    let mut dirs: Vec<[f64; 3]> = Vec::new();
    for &cz in &[
        1.0_f64, 0.999_999, 0.999, 0.99, 0.9, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.9, -0.999,
        -0.999_999, -1.0,
    ] {
        let sz = (1.0 - cz * cz).max(0.0).sqrt();
        for &az in &[0.0_f64, 0.7, 1.9, 3.6, 5.1] {
            dirs.push([sz * az.cos(), sz * az.sin(), cz]);
        }
    }
    let mus: Vec<f64> = vec![
        -1.0, -0.95, -0.7, -0.4, -0.1, 0.0, 0.1, 0.3, 0.6, 0.85, 0.99, 1.0,
    ];
    let phis: Vec<f64> = (0..16)
        .map(|i| i as f64 * std::f64::consts::TAU / 16.0)
        .collect();

    // Cartesian product.
    let mut d_flat = Vec::new();
    let mut m_flat = Vec::new();
    let mut p_flat = Vec::new();
    for d in &dirs {
        for &mu in &mus {
            for &phi in &phis {
                d_flat.extend_from_slice(d);
                m_flat.push(mu);
                p_flat.push(phi);
            }
        }
    }
    let n = m_flat.len();
    let gpu = run_rotate_mu_phi(&ctx, &d_flat, &m_flat, &p_flat);

    let mut max_abs = 0.0_f64;
    let mut max_case = (0usize, [0.0; 3], [0.0; 3]);
    let mut pole_mismatches = 0usize;
    for i in 0..n {
        let d = [d_flat[3 * i], d_flat[3 * i + 1], d_flat[3 * i + 2]];
        let cpu = rotate_direction_fast(d[0], d[1], d[2], m_flat[i], p_flat[i]);
        let g = [gpu[3 * i], gpu[3 * i + 1], gpu[3 * i + 2]];
        // GPU renormalizes; normalize the CPU result the same way for a
        // like-for-like comparison.
        let nrm = (cpu[0] * cpu[0] + cpu[1] * cpu[1] + cpu[2] * cpu[2]).sqrt();
        let c = [cpu[0] / nrm, cpu[1] / nrm, cpu[2] / nrm];
        let dev = (0..3).map(|k| (g[k] - c[k]).abs()).fold(0.0_f64, f64::max);
        // The exact-pole degenerate branch differs by construction (CPU
        // expands about y, GPU uses the simple pole form) -- both are valid
        // uniform-φ rotations; count separately rather than failing.
        let at_pole = d[2].abs() >= 1.0 - 1e-12;
        if at_pole {
            if dev > 1e-9 {
                pole_mismatches += 1;
            }
            continue;
        }
        if dev > max_abs {
            max_abs = dev;
            max_case = (i, g, c);
        }
    }
    println!(
        "rotate_mu_phi: {n} cases, max |GPU-CPU| component dev (off-pole) = {max_abs:.3e}, pole-branch diffs = {pole_mismatches}"
    );
    if max_abs > 1e-10 {
        let (i, g, c) = max_case;
        println!(
            "  worst case i={i}: dir=({:.6},{:.6},{:.6}) mu={} phi={}\n  GPU {g:?}\n  CPU {c:?}",
            d_flat[3 * i],
            d_flat[3 * i + 1],
            d_flat[3 * i + 2],
            m_flat[i],
            p_flat[i]
        );
    }
    assert!(
        max_abs < 1e-10,
        "GPU rotation deviates from CPU rotate_direction_fast by {max_abs:.3e}"
    );
}
