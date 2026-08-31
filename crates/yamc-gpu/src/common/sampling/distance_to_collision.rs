//! Distance to collision in a homogeneous medium -- first physics-shaped
//! kernel of the GPU port.
//!
//! Each thread is one particle. The thread reads its RNG state, samples
//! a uniform `ξ ∈ (0, 1)`, and computes the standard exponential
//! free-flight distance:
//!
//! ```text
//! d = -ln(ξ) / Σ_total
//! ```
//!
//! The kernel uses the inline PCG-32 step (same constants as
//! `kernels::pcg32` and CPU `GpuRng`, byte-identical sequence) and the
//! `ln_f64` polyfill from `kernels::polyfills` (because cubecl-spirv
//! emits invalid SPIR-V for native `f64::ln` on this driver, see
//! cubecl#1316).
//!
//! Test strategy: statistical only. The sampled distances follow an
//! exponential distribution with mean and standard deviation both
//! `1 / Σ_total`. With `N = 10_000` samples the standard error of the
//! mean is `(1 / Σ_total) / sqrt(N) ≈ 0.02 / Σ_total`, so a 3-sigma
//! check (`|mean - 1/Σ_total|  < 0.06 / Σ_total`) is tight enough to
//! catch a real bug while being well clear of MC noise.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Each thread reads its seed from `seeds[tid]`, samples one free
/// flight, writes to `distances[tid]`. `params[0]` carries `Σ_total`
/// (kept in a 1-element buffer rather than using comptime, so changing
/// `Σ_total` between launches doesn't trigger a JIT recompile).
///
/// The PCG-32 step is inlined directly here. cubecl `&mut` bindings
/// don't snapshot like Rust does -- see `kernels::pcg32` -- so factoring
/// the step into a helper would silently shift every output by one
/// state advance.
#[cube(launch_unchecked)]
fn distance_to_collision_kernel(seeds: &[u32], params: &[f64], distances: &mut [f64]) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let sigma_total = params[0];

    let oldstate = expand_seed(seeds[ABSOLUTE_POS]);
    let _next_state = oldstate * PCG_MULT + PCG_INCR;
    let r_u32 = pcg_out(oldstate);

    // u32 → f64 in (0, 1]. Adding 1.0 before the divide guarantees we
    // never feed `ln(0) = -∞` to the polyfill, at the cost of skipping
    // exactly 0.0 from the output domain.
    let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    distances[ABSOLUTE_POS] = -ln_f64(xi) / sigma_total;
}

/// CPU equivalent of `run_distance_to_collision` for benchmarking.
/// Uses the same PCG-32 algorithm and the same `(r as f64 + 1.0) /
/// (2^32 + 1)` mapping as the GPU kernel, with libm `f64::ln` standing
/// in for the polyfill. Output is statistically the same exponential
/// distribution; bit patterns differ by ~1 ulp because libm and the
/// polyfill round differently on some inputs.
pub fn run_distance_to_collision_cpu(seeds: &[u32], sigma_total: f64) -> Vec<f64> {
    seeds
        .iter()
        .map(|&seed| {
            let oldstate = crate::common::rng::expand_seed(seed);
            let _next_state = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r_u32 = pcg_out(oldstate);
            let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            -xi.ln() / sigma_total
        })
        .collect()
}

/// Rayon-parallel CPU equivalent. Uses the same per-seed algorithm as
/// the sequential version above; rayon splits the input slice across
/// the host's worker threads.
pub fn run_distance_to_collision_cpu_rayon(seeds: &[u32], sigma_total: f64) -> Vec<f64> {
    use rayon::prelude::*;
    seeds
        .par_iter()
        .map(|&seed| {
            let oldstate = crate::common::rng::expand_seed(seed);
            let _next_state = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r_u32 = pcg_out(oldstate);
            let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            -xi.ln() / sigma_total
        })
        .collect()
}

/// Sample one free flight per seed. Each `seeds[i]` becomes one PCG-32
/// step, one `ln_f64` evaluation, one division. Returns a `Vec<f64>`
/// of distances in the same order as the input seeds.
pub fn run_distance_to_collision(ctx: &GpuContext, seeds: &[u32], sigma_total: f64) -> Vec<f64> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let params = [sigma_total];
    let params_handle = client.create_from_slice(bytemuck::cast_slice(&params));
    let distances_handle = client.empty(n * core::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        distance_to_collision_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(params_handle, 1),
            BufferArg::from_raw_parts(distances_handle.clone(), n),
        );
    }

    let bytes = client.read_one(distances_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Sampled distances must be a proper exponential distribution with
    /// rate Σ_total. Mean ≈ 1/Σ, std ≈ 1/Σ. With 10k samples the
    /// standard error of the mean is about 0.02/Σ, so a 3-sigma window
    /// of 0.06/Σ catches any real bug while leaving MC noise alone.
    #[test]
    fn distance_distribution_matches_exponential() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let n = 10_000usize;
        let sigma_total = 0.5_f64;
        // Distinct seeds per thread so they generate independent
        // sequences. Any non-trivial seeding pattern works; this one
        // mixes a salt with the index for a quick spread.
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();

        let distances = run_distance_to_collision(&ctx, &seeds, sigma_total);

        assert_eq!(distances.len(), n);
        assert!(
            distances.iter().all(|d| d.is_finite() && *d > 0.0),
            "all distances should be finite and positive"
        );

        let mean: f64 = distances.iter().sum::<f64>() / n as f64;
        let variance: f64 = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        let expected_mean = 1.0 / sigma_total;
        let expected_std = 1.0 / sigma_total;
        let sem = expected_std / (n as f64).sqrt();

        println!(
            "distance to collision: mean = {mean} (expected {expected_mean}, SEM {sem}), \
             std = {std} (expected {expected_std})"
        );

        assert!(
            (mean - expected_mean).abs() < 3.0 * sem,
            "sample mean {mean} outside 3-sigma window of {expected_mean} (SEM {sem})"
        );
        // Std should be within ~5% of expected for n = 10_000 -- looser
        // than mean because the variance estimator has higher variance.
        assert!(
            ((std - expected_std).abs() / expected_std) < 0.05,
            "sample std {std} more than 5% from expected {expected_std}"
        );
    }
}
