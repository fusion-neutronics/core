//! Per-mechanism GPU-vs-CPU test for the Compton (Klein-Nishina/Kahn)
//! sampler extracted into `kernels::compton_scatter`. Lives here (a
//! `tests/` integration test) rather than inline in the src file, and
//! exercises the helper through its public `run_compton_kahn` wrapper.
//!
//! Run on a GPU box single-threaded:
//!   cargo test -p yamc-gpu --release --test gpu_compton_scatter -- --test-threads=1 --nocapture

#![cfg(not(target_os = "macos"))]

use rand::SeedableRng;
use yamc_element::photon::klein_nishina;
use yamc_gpu::common::probes::compton_scatter::run_compton_kahn;
use yamc_gpu::{GpuContext, GpuInitError};

const MASS_ELECTRON_EV: f64 = 0.510_998_950_00e6;

/// The GPU free-electron Klein-Nishina sampler must reproduce the **same
/// scattered-energy distribution** as the CPU reference `klein_nishina`.
/// Both are sampled N times at a fixed incident energy; we compare the mean
/// `E'/E` and the large-energy-loss fraction. This is the per-scatter
/// energy-loss check whose absence let the #415 photoelectric deficit hide
/// behind loose integrated tallies.
///
/// Energies: 5 MeV (α≈9.8, where the CPU switches to a different sampler
/// than Kahn but the same Klein-Nishina distribution), 1.25 MeV (Co60-mean
/// source), 300 keV and 100 keV (energies a downscattered photon reaches).
#[test]
fn gpu_compton_kahn_matches_klein_nishina_distribution() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuInitError::NoF64Adapter) => {
            println!("no Vulkan f64 adapter -- skipping");
            return;
        }
    };
    let n = 200_000usize;

    for &e_ev in &[5.0e6_f64, 1.25e6, 3.0e5, 1.0e5] {
        let alpha = e_ev / MASS_ELECTRON_EV;

        // GPU: distinct PCG seeds per sample.
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(1))
            .collect();
        let alphas = vec![alpha; n];
        let (gpu_eratio, gpu_mu) = run_compton_kahn(&ctx, &seeds, &alphas);

        // CPU reference: yamc-element klein_nishina, N draws.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let mut cpu_sum = 0.0_f64;
        let mut cpu_below_half = 0usize; // E'/E < 0.5 -> large energy loss
        for _ in 0..n {
            let (alpha_out, _mu) = klein_nishina(alpha, &mut rng);
            let er = alpha_out / alpha;
            cpu_sum += er;
            if er < 0.5 {
                cpu_below_half += 1;
            }
        }
        let cpu_mean = cpu_sum / n as f64;
        let cpu_frac_big_loss = cpu_below_half as f64 / n as f64;

        let gpu_mean: f64 = gpu_eratio.iter().sum::<f64>() / n as f64;
        let gpu_frac_big_loss = gpu_eratio.iter().filter(|&&er| er < 0.5).count() as f64 / n as f64;

        // Every sample must be physical: E'/E in [1/(1+2α), 1], mu in [-1,1].
        let min_ratio = 1.0 / (1.0 + 2.0 * alpha);
        for (i, (&er, &m)) in gpu_eratio.iter().zip(gpu_mu.iter()).enumerate() {
            assert!(
                er >= min_ratio - 1e-9 && er <= 1.0 + 1e-9,
                "E'/E={er} at i={i} (E={e_ev}) outside [{min_ratio}, 1]"
            );
            assert!((-1.0..=1.0).contains(&m), "mu={m} at i={i} outside [-1,1]");
        }

        // MC means of E'/E: SEM is well under 0.001 at N=200k for both. A 2%
        // absolute window on the mean and on the large-loss fraction is far
        // tighter than the ~2.6x #415 bias and far looser than statistical
        // noise -- it catches a real per-scatter energy-loss discrepancy
        // without flaking.
        println!(
            "E={e_ev:.3e} a={alpha:.3}: mean E'/E GPU={gpu_mean:.4} CPU={cpu_mean:.4} | frac(E'/E<0.5) GPU={gpu_frac_big_loss:.4} CPU={cpu_frac_big_loss:.4}"
        );
        assert!(
            (gpu_mean - cpu_mean).abs() < 0.02,
            "E={e_ev:.2e}: GPU mean E'/E {gpu_mean:.4} vs CPU {cpu_mean:.4} -- per-scatter energy loss disagrees (#415)"
        );
        assert!(
            (gpu_frac_big_loss - cpu_frac_big_loss).abs() < 0.02,
            "E={e_ev:.2e}: GPU large-loss fraction {gpu_frac_big_loss:.4} vs CPU {cpu_frac_big_loss:.4}"
        );
    }
}
