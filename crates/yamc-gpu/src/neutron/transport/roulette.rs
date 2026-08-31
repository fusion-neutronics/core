//! Weight-cutoff Russian roulette for survival biasing.
//!
//! The weight-cutoff roulette (delta-tracking aside, the standard analog
//! variance-reduction roulette) culls low-weight survivors that implicit
//! capture would otherwise let accumulate near zero weight. After a
//! collision is fully processed, a still-alive particle whose weight has
//! dropped below `weight_cutoff` survives with probability
//! `weight / weight_survive` (continuing at `weight_survive`) or is killed
//! (weight 0). The survival probability leaves the expected weight
//! unchanged, so the roulette is variance-neutral (unbiased).
//!
//! [`weight_cutoff_roulette`] is the single `#[cube]` source of the
//! decision arithmetic, shared by the transport kernel and its CPU twin
//! (`shared.rs`); [`weight_cutoff_roulette_cpu`] is its exact CPU mirror.
//! `gpu_weight_cutoff_roulette_matches_cpu` pins the GPU result to the CPU
//! twin bit-for-bit on real hardware (the helper is pure f64
//! compare/divide/select -- no FMA, no transcendentals -- so equality is
//! exact, not within a tolerance).

#[cfg(test)]
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Weight-cutoff Russian-roulette decision. Given a still-alive particle's
/// post-collision `weight`, the survivor weight `weight_survive`, and a
/// uniform draw `xi` in `(0, 1]`, returns the particle's new weight:
/// `weight_survive` when it survives (probability `weight / weight_survive`)
/// or `0.0` when it is killed. Pure: the caller owns the RNG draw, the
/// below-cutoff gate, and the `alive` flag.
#[cube]
pub fn weight_cutoff_roulette(weight: f64, weight_survive: f64, xi: f64) -> f64 {
    // `0.0` encodes a kill so the helper lowers to a single SPIR-V
    // comparison + select (no `Option`, which `#[cube]` cannot return).
    let mut out = 0.0_f64;
    if xi < weight / weight_survive {
        out = weight_survive;
    }
    out
}

/// CPU twin of [`weight_cutoff_roulette`]. Identical arithmetic so the
/// transport CPU mirror stays bit-for-bit with the kernel.
pub fn weight_cutoff_roulette_cpu(weight: f64, weight_survive: f64, xi: f64) -> f64 {
    if xi < weight / weight_survive {
        weight_survive
    } else {
        0.0
    }
}

/// Validation kernel: one [`weight_cutoff_roulette`] evaluation per thread.
/// Reads `(weight, weight_survive, xi)` triples and writes the helper's
/// output weight. For test/validation use only.
#[cfg(test)]
#[cube(launch_unchecked)]
fn weight_cutoff_roulette_kernel(weights: &[f64], survives: &[f64], xis: &[f64], out: &mut [f64]) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    out[ABSOLUTE_POS] = weight_cutoff_roulette(
        weights[ABSOLUTE_POS],
        survives[ABSOLUTE_POS],
        xis[ABSOLUTE_POS],
    );
}

/// Run [`weight_cutoff_roulette`] on the GPU for each `(weight,
/// weight_survive, xi)` triple. For test/validation use.
#[cfg(test)]
fn run_weight_cutoff_roulette(
    ctx: &GpuContext,
    weights: &[f64],
    survives: &[f64],
    xis: &[f64],
) -> Vec<f64> {
    let n = weights.len();
    assert_eq!(survives.len(), n);
    assert_eq!(xis.len(), n);
    let client = ctx.client();
    let w_h = client.create_from_slice(bytemuck::cast_slice(weights));
    let s_h = client.create_from_slice(bytemuck::cast_slice(survives));
    let x_h = client.create_from_slice(bytemuck::cast_slice(xis));
    let out_h = client.empty(core::mem::size_of_val(weights));
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        weight_cutoff_roulette_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(w_h, n),
            BufferArg::from_raw_parts(s_h, n),
            BufferArg::from_raw_parts(x_h, n),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice::<u8, f64>(&client.read_one(out_h).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuInitError;

    /// The roulette helper is pure f64 compare/divide/select (no FMA, no
    /// transcendentals), so the GPU result must equal the CPU twin
    /// bit-for-bit. Sweeps below-survive weights and `xi` values straddling
    /// the `weight / weight_survive` survival probability, including the
    /// boundary.
    #[test]
    fn gpu_weight_cutoff_roulette_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let weight_survive = 1.0_f64;
        let mut weights = Vec::new();
        let mut survives = Vec::new();
        let mut xis = Vec::new();
        for &w in &[0.01_f64, 0.1, 0.25, 0.5, 0.75, 0.9] {
            for k in 0..=20u32 {
                let xi = (k as f64 + 1.0) / 21.0; // (0, 1]
                weights.push(w);
                survives.push(weight_survive);
                xis.push(xi);
            }
        }

        let gpu = run_weight_cutoff_roulette(&ctx, &weights, &survives, &xis);
        assert_eq!(gpu.len(), weights.len());
        for i in 0..weights.len() {
            let cpu = weight_cutoff_roulette_cpu(weights[i], survives[i], xis[i]);
            assert_eq!(
                gpu[i].to_bits(),
                cpu.to_bits(),
                "roulette differs at i={i}: gpu {} cpu {cpu} (w {} ws {} xi {})",
                gpu[i],
                weights[i],
                survives[i],
                xis[i]
            );
        }
        println!(
            "weight_cutoff_roulette: {} cases, GPU==CPU bit-exact",
            weights.len()
        );
    }

    /// The CPU twin must be unbiased: averaged over uniform `xi`, the
    /// expected surviving weight equals the pre-roulette weight.
    #[test]
    fn roulette_cpu_is_variance_neutral() {
        let weight_survive = 1.0_f64;
        for &w in &[0.05_f64, 0.2, 0.4, 0.7] {
            let n = 1_000_000u32;
            let mut sum = 0.0_f64;
            for k in 0..n {
                let xi = (k as f64 + 0.5) / n as f64;
                sum += weight_cutoff_roulette_cpu(w, weight_survive, xi);
            }
            let mean = sum / n as f64;
            assert!(
                (mean - w).abs() < 1.0e-4,
                "roulette mean {mean} != pre-roulette weight {w}"
            );
        }
    }
}
