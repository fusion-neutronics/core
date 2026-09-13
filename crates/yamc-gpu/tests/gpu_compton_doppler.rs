//! Per-mechanism GPU-vs-CPU test for the Compton Doppler broadening sampler
//! extracted into `kernels::compton_doppler`. Builds Fe's Compton-profile
//! tables (reusing `extract_compton_doppler_for_gpu`) and checks the GPU
//! helper's broadened-E_out distribution matches the CPU
//! `PhotonInteraction::compton_doppler` at a fixed incident energy + mu.
//!
//!   cargo test -p yamc-gpu --release --test gpu_compton_doppler -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use std::path::Path;
use std::sync::Arc;

use rand::SeedableRng;
use yamc_element::photon::compton_profile_pz;
use yamc_element::photon_arrow::read_photon_interaction_from_arrow;
use yamc_gpu::common::probes::compton_doppler::run_compton_doppler;
use yamc_gpu::extract_compton_doppler_for_gpu;
use yamc_gpu::{GpuContext, GpuInitError};

const MASS_ELECTRON_EV: f64 = 0.510_998_950_00e6;

#[test]
fn gpu_compton_doppler_matches_cpu_distribution() {
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
    let pz_grid = compton_profile_pz();
    let materials = vec![vec![("Fe".to_string(), fe.clone(), 1.0_f64)]];
    let dop = extract_compton_doppler_for_gpu(&materials, &pz_grid);
    assert_eq!(dop.n_shells.len(), 1);
    assert!(dop.n_shells[0] >= 1, "Fe has no Compton shells");

    let n = 200_000usize;
    // Doppler broadening matters at moderate energy + large scattering angle.
    for &(e_ev, mu) in &[(1.0e5_f64, -0.5_f64), (5.0e4, -0.8), (3.0e5, 0.0)] {
        let alpha = e_ev / MASS_ELECTRON_EV;
        let e_out_kn = alpha / (1.0 + alpha * (1.0 - mu)) * MASS_ELECTRON_EV;

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(13))
            .collect();
        let gpu_e = run_compton_doppler(
            &ctx,
            &seeds,
            e_ev,
            mu,
            &dop.pz_grid,
            dop.n_shells[0],
            &dop.electron_pdf,
            &dop.binding_energy,
            &dop.profile_pdf,
            &dop.profile_cdf,
            &dop.profile_tail_slope,
            &dop.profile_negative_mass,
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD0FF);
        let mut cpu_sum = 0.0_f64;
        let mut cpu_sq = 0.0_f64;
        for _ in 0..n {
            let (e_out, _shell) = fe.compton_doppler(alpha, mu, &mut rng);
            cpu_sum += e_out;
            cpu_sq += e_out * e_out;
        }
        let cpu_mean = cpu_sum / n as f64;
        let cpu_std = (cpu_sq / n as f64 - cpu_mean * cpu_mean).max(0.0).sqrt();

        let gpu_mean = gpu_e.iter().sum::<f64>() / n as f64;
        let gpu_sq: f64 = gpu_e.iter().map(|&e| e * e).sum::<f64>() / n as f64;
        let gpu_std = (gpu_sq - gpu_mean * gpu_mean).max(0.0).sqrt();
        for (i, &e) in gpu_e.iter().enumerate() {
            assert!(
                e.is_finite() && e > 0.0,
                "GPU E_out={e} at i={i} non-physical"
            );
        }

        // Compare the broadened distribution: mean and width (std), each
        // relative to the KN energy. Doppler is a ~few-% broadening; a 3%
        // window on mean and 20% on the (small) std catches a real
        // discrepancy without flaking on MC noise.
        let mean_rel = (gpu_mean - cpu_mean).abs() / e_out_kn;
        let std_rel = if cpu_std > 0.0 {
            (gpu_std - cpu_std).abs() / cpu_std
        } else {
            0.0
        };
        println!(
            "E={e_ev:.2e} mu={mu}: E_out_kn={e_out_kn:.4e} | mean GPU={gpu_mean:.4e} CPU={cpu_mean:.4e} (rel {mean_rel:.4}) | std GPU={gpu_std:.3e} CPU={cpu_std:.3e} (rel {std_rel:.3})"
        );
        assert!(
            mean_rel < 0.03,
            "E={e_ev:.2e} mu={mu}: Doppler mean E_out GPU {gpu_mean:.4e} vs CPU {cpu_mean:.4e} differs ({mean_rel:.4} of KN)"
        );
        assert!(
            std_rel < 0.20,
            "E={e_ev:.2e} mu={mu}: Doppler width GPU {gpu_std:.3e} vs CPU {cpu_std:.3e} differs ({std_rel:.3})"
        );
    }
}
