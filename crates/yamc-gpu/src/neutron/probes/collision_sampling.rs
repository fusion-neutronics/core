//! Reaction sampling at a collision site (Phase E.1).
//!
//! At a collision the particle picks one of several possible reactions
//! (elastic scatter, absorption, fission, ...) weighted by the partial
//! cross sections at its energy:
//!
//! ```text
//! P(MT_i) = Σ_{MT_i}(E) / Σ_total(E)
//! ```
//!
//! Sampling is the standard inverse-CDF construction: draw a uniform
//! `ξ ∈ [0, 1)` and pick the smallest `i` such that the cumulative
//! `Σ_{MT_0..i+1}` exceeds `ξ * Σ_total`.
//!
//! This first cut handles **two reactions only** -- elastic scatter
//! (output `1`) and absorption (output `0`). It also takes the
//! per-particle partial XS values *as inputs*, rather than looking
//! them up. Lookup is already validated (`xs_lookup.rs`); decoupling
//! the lookup from the sampling gives a cleaner test of the new
//! mechanic. Wiring lookup + sampling into a single kernel is the
//! next sub-branch.
//!
//! # Outputs
//!
//! `out_mt[tid]` is the chosen MT code as a `u32`:
//! - `0` = absorption  (terminate the particle)
//! - `1` = elastic scatter  (update energy and direction)
//!
//! Real production code will use the standard MT numbers (102 for
//! capture, 2 for elastic, 18 for fission, ...). For this first kernel
//! a packed enum-style 0/1 keeps the test simple.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Per-particle: PCG-32 → uniform `ξ` → compare to elastic fraction.
/// Output `1` if elastic was sampled, `0` if absorption.
#[cube(launch_unchecked)]
fn collision_sampling_kernel(
    seeds: &[u32],
    xs_elastic: &[f64],
    xs_absorption: &[f64],
    out_mt: &mut [u32],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    // Inline PCG-32 -- same constants as `kernels::pcg32`.
    let oldstate = expand_seed(seeds[ABSOLUTE_POS]);
    let _next_state = oldstate * PCG_MULT + PCG_INCR;
    let r_u32 = pcg_out(oldstate);
    let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    // Inverse-CDF on a 2-bin discrete distribution.
    let sigma_e = xs_elastic[ABSOLUTE_POS];
    let sigma_a = xs_absorption[ABSOLUTE_POS];
    let p_elastic = sigma_e / (sigma_e + sigma_a);

    let mut mt = 0u32;
    if xi < p_elastic {
        mt = 1u32;
    }
    out_mt[ABSOLUTE_POS] = mt;
}

/// Run the collision-sampling kernel. `xs_elastic` and `xs_absorption`
/// hold per-particle partial cross sections (the host has already
/// looked them up at each particle's energy). Output is one MT code
/// per particle: `0` = absorption, `1` = elastic.
pub fn run_collision_sampling(
    ctx: &GpuContext,
    seeds: &[u32],
    xs_elastic: &[f64],
    xs_absorption: &[f64],
) -> Vec<u32> {
    assert_eq!(seeds.len(), xs_elastic.len());
    assert_eq!(seeds.len(), xs_absorption.len());

    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let xs_e_handle = client.create_from_slice(bytemuck::cast_slice(xs_elastic));
    let xs_a_handle = client.create_from_slice(bytemuck::cast_slice(xs_absorption));
    let out_handle = client.empty(std::mem::size_of_val(seeds));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        collision_sampling_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(xs_e_handle, n),
            BufferArg::from_raw_parts(xs_a_handle, n),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// CPU equivalent of the collision-sampling kernel. Same PCG-32 and
/// same threshold comparison.
pub fn run_collision_sampling_cpu(
    seeds: &[u32],
    xs_elastic: &[f64],
    xs_absorption: &[f64],
) -> Vec<u32> {
    assert_eq!(seeds.len(), xs_elastic.len());
    assert_eq!(seeds.len(), xs_absorption.len());

    seeds
        .iter()
        .zip(xs_elastic.iter())
        .zip(xs_absorption.iter())
        .map(|((&seed, &sigma_e), &sigma_a)| {
            let oldstate = crate::common::rng::expand_seed(seed);
            let _next_state = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r_u32 = pcg_out(oldstate);
            let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let p_elastic = sigma_e / (sigma_e + sigma_a);
            if xi < p_elastic {
                1u32
            } else {
                0u32
            }
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// 100k particles, all with the same XS ratio. Expected fraction
    /// scattered = `σ_e / (σ_e + σ_a)`. Standard error of a
    /// proportion is `sqrt(p (1 - p) / N)`. With p ≈ 2/3 and N = 1e5
    /// the SEM is ~0.0015, so a 3-sigma tolerance of ~0.005 catches a
    /// real bug while leaving room for MC noise.
    #[test]
    fn gpu_scatter_fraction_matches_xs_ratio() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let n = 100_000usize;
        let sigma_e = 2.0_f64;
        let sigma_a = 1.0_f64;
        let p_expected = sigma_e / (sigma_e + sigma_a);

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let xs_e_arr: Vec<f64> = vec![sigma_e; n];
        let xs_a_arr: Vec<f64> = vec![sigma_a; n];

        let mt = run_collision_sampling(&ctx, &seeds, &xs_e_arr, &xs_a_arr);
        let n_scatter = mt.iter().filter(|&&m| m == 1).count();
        let p_observed = n_scatter as f64 / n as f64;
        let sem = (p_expected * (1.0 - p_expected) / n as f64).sqrt();
        let tol = 3.0 * sem;

        println!("scatter fraction: observed = {p_observed}, expected = {p_expected}, SEM = {sem}");
        assert!(
            (p_observed - p_expected).abs() < tol,
            "observed scatter fraction {p_observed} outside 3-sigma window of {p_expected}"
        );

        // Outputs must be only 0 or 1.
        for (i, &m) in mt.iter().enumerate() {
            assert!(m == 0 || m == 1, "out_mt[{i}] = {m}, expected 0 or 1");
        }
    }

    /// GPU and CPU must agree on every per-particle sample, given the
    /// same seeds and the same XS values. With identical PCG-32 and
    /// identical f64 division, the threshold comparison should agree
    /// bit-exactly. Allow a tiny number of disagreements (`< 0.1%`) in
    /// case the GPU's f64 division differs from CPU by 1 ulp at the
    /// boundary of `xi == p_elastic`.
    #[test]
    fn gpu_collision_sampling_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 10_000usize;
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        // Mix of XS ratios across particles -- exercises the divide
        // step at multiple values rather than testing one ratio.
        let xs_e_arr: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 % 10.0) * 0.5).collect();
        let xs_a_arr: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 % 7.0) * 0.3).collect();

        let gpu = run_collision_sampling(&ctx, &seeds, &xs_e_arr, &xs_a_arr);
        let cpu = run_collision_sampling_cpu(&seeds, &xs_e_arr, &xs_a_arr);
        let n_diff = gpu.iter().zip(cpu.iter()).filter(|(g, c)| g != c).count();
        let frac_diff = n_diff as f64 / n as f64;
        println!("GPU vs CPU sampling: {n_diff} disagreements out of {n} ({frac_diff:.4})");
        assert!(
            frac_diff < 0.001,
            "{frac_diff:.4} > 0.001 disagreement rate suggests a real bug, not just 1-ulp boundary noise"
        );
    }
}
