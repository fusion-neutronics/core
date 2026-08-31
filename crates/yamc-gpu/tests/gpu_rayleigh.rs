//! Per-mechanism GPU-vs-CPU test for the Rayleigh (coherent) angle sampler
//! extracted into `kernels::rayleigh_scatter`. Loads the Fe element's
//! integrated coherent form factor and checks the GPU helper's µ
//! distribution matches the CPU `PhotonInteraction::rayleigh_scatter` on the
//! same table.
//!
//! Run on a GPU box single-threaded:
//!   cargo test -p yamc-gpu --release --test gpu_rayleigh -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use std::path::Path;

use rand::SeedableRng;
use yamc_element::photon_arrow::read_photon_interaction_from_arrow;
use yamc_gpu::common::probes::rayleigh_scatter::run_rayleigh;
use yamc_gpu::{GpuContext, GpuInitError};
use yamc_nuclide::reaction_product::Tabulated1D;

const MASS_ELECTRON_EV: f64 = 0.510_998_950_00e6;

// #421 FIXED: `rayleigh_propose` now uses `f_max = F(x²_max)` (interpolated)
// instead of the full-table integral, matching the CPU. This test (GPU
// Rayleigh µ distribution == CPU `rayleigh_scatter`) now passes.
#[test]
fn gpu_rayleigh_matches_cpu_distribution() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    // Fe photon data (directory holding element.arrow). Path is relative to
    // the yamc-gpu crate root (test CWD).
    let fe = read_photon_interaction_from_arrow(Path::new("../yamc/tests/Fe.arrow"))
        .expect("load Fe.arrow photon data");

    // Integrated coherent form factor: x = x² axis, y = F(x²) CDF.
    let (x2, cdf) = match &fe.coherent_int_form_factor {
        Tabulated1D::Tabulated1D { x, y, .. } => (x.clone(), y.clone()),
    };
    assert!(x2.len() >= 2, "Fe coherent form factor too short");

    let n = 200_000usize;
    // Rayleigh is most relevant at low energy; pick a few keV-range points.
    for &e_ev in &[1.0e5_f64, 5.0e4, 2.0e4] {
        let alpha = e_ev / MASS_ELECTRON_EV;

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(7))
            .collect();
        let energies = vec![e_ev; n];
        let gpu_mu = run_rayleigh(&ctx, &seeds, &energies, &x2, &cdf);

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
        let mut cpu_sum = 0.0_f64;
        let mut cpu_fwd = 0usize; // mu > 0.9 -> forward
        for _ in 0..n {
            let mu = fe.rayleigh_scatter(alpha, &mut rng);
            cpu_sum += mu;
            if mu > 0.9 {
                cpu_fwd += 1;
            }
        }
        let cpu_mean = cpu_sum / n as f64;
        let cpu_frac_fwd = cpu_fwd as f64 / n as f64;

        let gpu_mean: f64 = gpu_mu.iter().sum::<f64>() / n as f64;
        let gpu_frac_fwd = gpu_mu.iter().filter(|&&m| m > 0.9).count() as f64 / n as f64;

        for (i, &m) in gpu_mu.iter().enumerate() {
            assert!((-1.0..=1.0).contains(&m), "mu={m} at i={i} outside [-1,1]");
        }

        println!(
            "E={e_ev:.2e} a={alpha:.4}: mean mu GPU={gpu_mean:.4} CPU={cpu_mean:.4} | frac(mu>0.9) GPU={gpu_frac_fwd:.4} CPU={cpu_frac_fwd:.4}"
        );
        assert!(
            (gpu_mean - cpu_mean).abs() < 0.02,
            "E={e_ev:.2e}: GPU mean mu {gpu_mean:.4} vs CPU {cpu_mean:.4} -- Rayleigh angle disagrees"
        );
        assert!(
            (gpu_frac_fwd - cpu_frac_fwd).abs() < 0.02,
            "E={e_ev:.2e}: GPU forward fraction {gpu_frac_fwd:.4} vs CPU {cpu_frac_fwd:.4}"
        );
    }
}
