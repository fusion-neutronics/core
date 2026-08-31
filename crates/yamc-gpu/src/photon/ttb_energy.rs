//! Thick-target-bremsstrahlung per-photon energy sampler.
//!
//! Shared by the photon transport kernel's photoelectron / Auger / positron
//! TTB loops. Given a uniform draw `c` in `[c_l, c_max]` and the bracketing
//! log-energy grid points plus the PDF/CDF values for the chosen
//! incident-energy row, [`ttb_photon_energy`] returns the sampled photon
//! energy (linear eV) via the same inverse-CDF interpolation the CPU uses.
//!
//! The CPU reference is
//! [`yamc_physics::photon::bremsstrahlung::sample_ttb_photon_energy`]; the
//! `gpu_ttb_energy_matches_cpu` test pins the GPU `#[cube]` form against it.
//! The two cannot be bit-identical (the GPU uses the `ln_f64` / `exp_f64`
//! polyfills and `exp(ln(x)/a)` where the CPU uses `std` `ln`/`exp` and
//! `powf`), but they agree to well within Monte-Carlo noise.

use crate::common::polyfills::{exp_f64, ln_f64};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Inverse-CDF TTB photon energy (linear eV). `w_l_log`/`w_r_log` are the
/// bracketing log-space photon-energy grid points; `p_l`/`p_r` the PDF at
/// those points and `c_l` the CDF at the lower point, for the chosen
/// incident-energy row. The `inside <= 0` guard avoids `ln` of a
/// non-positive argument (the CPU `powf` would return a NaN that the
/// caller's `w > cutoff` check already rejects, so this is behaviourally
/// equivalent and just keeps the GPU result finite).
#[cube]
pub fn ttb_photon_energy(c: f64, w_l_log: f64, w_r_log: f64, p_l: f64, p_r: f64, c_l: f64) -> f64 {
    let a = ln_f64(p_r / p_l) / (w_r_log - w_l_log) + 1.0;
    let exp_wl = exp_f64(w_l_log);
    let inside = a * (c - c_l) / (exp_wl * p_l) + 1.0;
    let mut w = 0.0_f64;
    if inside > 0.0 {
        w = exp_wl * exp_f64(ln_f64(inside) / a);
    }
    w
}

/// Test kernel: one TTB-energy sample per thread from a stride-6 input
/// buffer laid out as `[c, w_l_log, w_r_log, p_l, p_r, c_l]`.
#[cube(launch_unchecked)]
fn ttb_energy_kernel(inputs: &[f64], out: &mut [f64]) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let i = ABSOLUTE_POS * 6;
    out[ABSOLUTE_POS] = ttb_photon_energy(
        inputs[i],
        inputs[i + 1],
        inputs[i + 2],
        inputs[i + 3],
        inputs[i + 4],
        inputs[i + 5],
    );
}

/// Run [`ttb_photon_energy`] on the GPU for each stride-6 input tuple
/// (`[c, w_l_log, w_r_log, p_l, p_r, c_l]`). For test/validation use.
pub fn run_ttb_photon_energy(ctx: &GpuContext, inputs: &[f64]) -> Vec<f64> {
    assert!(inputs.len().is_multiple_of(6));
    let n = inputs.len() / 6;
    let client = ctx.client();
    let in_h = client.create_from_slice(bytemuck::cast_slice(inputs));
    let out_h = client.empty(n * core::mem::size_of::<f64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        ttb_energy_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(in_h, inputs.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};
    use yamc_physics::photon::bremsstrahlung::sample_ttb_photon_energy;

    /// The GPU `#[cube]` TTB energy sampler must match the CPU reference
    /// (`yamc_physics::...::sample_ttb_photon_energy`) for the same inputs.
    /// Sweeps the inverse-CDF formula across realistic photon-energy
    /// windows and PDF ratios (so the slope `a` spans `a<1`, `a≈1`,
    /// `a>1`) with the draw `c` walking across the bin. The two backends
    /// differ only in the transcendental implementation, so they agree to
    /// far tighter than Monte-Carlo noise.
    #[test]
    fn gpu_ttb_energy_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Build a battery of (c, w_l_log, w_r_log, p_l, p_r, c_l) tuples.
        // p_l is chosen so exp(w_l)*p_l = 1, giving inside = a*(c-c_l)+1;
        // c walks above c_l so `inside` stays positive across the sweep.
        let mut inputs: Vec<f64> = Vec::new();
        for &e_lo in &[1.0e3_f64, 1.0e4, 1.0e5, 1.0e6] {
            let w_l_log = e_lo.ln();
            let w_r_log = (e_lo * 3.0).ln();
            let exp_wl = w_l_log.exp();
            let p_l = 1.0 / exp_wl;
            for &ratio in &[0.25_f64, 0.6, 1.0, 1.7, 4.0] {
                let p_r = p_l * ratio;
                let c_l = 0.3;
                for k in 0..8 {
                    let c = c_l + 0.18 * k as f64;
                    inputs.extend_from_slice(&[c, w_l_log, w_r_log, p_l, p_r, c_l]);
                }
            }
        }

        let gpu = run_ttb_photon_energy(&ctx, &inputs);
        let mut max_rel = 0.0_f64;
        let n = inputs.len() / 6;
        for (i, t) in inputs.as_chunks::<6>().0.iter().enumerate() {
            let cpu = sample_ttb_photon_energy(t[0], t[1], t[2], t[3], t[4], t[5]);
            let g = gpu[i];
            assert!(
                cpu.is_finite() && g.is_finite(),
                "sample {i}: non-finite (gpu {g}, cpu {cpu}) for inputs {t:?}"
            );
            let rel = (g - cpu).abs() / cpu.abs().max(1.0);
            if rel > max_rel {
                max_rel = rel;
            }
        }
        println!("TTB photon energy: {n} samples, max relative GPU-vs-CPU drift = {max_rel}");
        assert!(
            max_rel < 1e-9,
            "GPU vs CPU TTB energy drift {max_rel} exceeds 1e-9"
        );
    }
}
