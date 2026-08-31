//! Per-mechanism GPU-vs-CPU test for the photoelectric subshell sampler
//! extracted into `kernels::pe_subshell`. Loads the Fe element's per-subshell
//! photoelectric cross sections and checks the GPU helper's sampled-shell
//! distribution matches the CPU `PhotonInteraction::sample_photoelectric_subshell`
//! on the same table.
//!
//!   cargo test -p yamc-gpu --release --test gpu_pe_subshell -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use std::path::Path;

use rand::SeedableRng;
use yamc_element::photon_arrow::read_photon_interaction_from_arrow;
use yamc_gpu::common::probes::pe_subshell::run_pe_subshell;
use yamc_gpu::{GpuContext, GpuInitError};

const AR_MAX_SHELLS: usize = 16;

#[test]
fn gpu_pe_subshell_matches_cpu_distribution() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };

    let fe = read_photon_interaction_from_arrow(Path::new("../yamc/tests/Fe.arrow"))
        .expect("load Fe.arrow photon data");
    let ns = fe.shells.len();
    assert!(ns > 0 && ns <= AR_MAX_SHELLS, "Fe shells = {ns}");

    let n = 200_000usize;
    // A few energies above the Fe K-edge (~7.1 keV) where several subshells
    // are open, so the sampled-shell distribution is non-trivial.
    for &e_ev in &[5.0e4_f64, 2.0e4, 1.0e4] {
        let micro = fe.calculate_xs(e_ev);
        let i_g = micro.index_grid;
        let f = micro.interp_factor;

        // Build the [2 x AR_MAX_SHELLS] log-XS table (rows = the two energy
        // grid points bracketing e_ev), padded with zeros (skipped like the
        // CPU's below-threshold shells).
        let mut table = vec![0.0_f64; 2 * AR_MAX_SHELLS];
        table[..ns].copy_from_slice(&fe.cross_sections[i_g][..ns]);
        table[AR_MAX_SHELLS..AR_MAX_SHELLS + ns].copy_from_slice(&fe.cross_sections[i_g + 1][..ns]);

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(11))
            .collect();
        let gpu_shell = run_pe_subshell(&ctx, &seeds, ns as u32, f, &table);

        // Histogram both.
        let mut gpu_hist = vec![0usize; ns];
        for &sh in &gpu_shell {
            assert!((sh as usize) < ns, "GPU shell {sh} >= ns {ns}");
            gpu_hist[sh as usize] += 1;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED);
        let mut cpu_hist = vec![0usize; ns];
        for _ in 0..n {
            let sh = fe.sample_photoelectric_subshell(&micro, &mut rng);
            cpu_hist[sh] += 1;
        }

        let mut max_dev = 0.0_f64;
        for s in 0..ns {
            let g = gpu_hist[s] as f64 / n as f64;
            let c = cpu_hist[s] as f64 / n as f64;
            max_dev = max_dev.max((g - c).abs());
        }
        let gfrac: Vec<String> = gpu_hist
            .iter()
            .map(|&h| format!("{:.3}", h as f64 / n as f64))
            .collect();
        let cfrac: Vec<String> = cpu_hist
            .iter()
            .map(|&h| format!("{:.3}", h as f64 / n as f64))
            .collect();
        println!("E={e_ev:.2e}: shell fractions\n  GPU {gfrac:?}\n  CPU {cfrac:?}\n  max |Δ| = {max_dev:.4}");
        assert!(
            max_dev < 0.02,
            "E={e_ev:.2e}: PE subshell distribution differs by {max_dev:.4} (> 0.02)"
        );
    }
}
