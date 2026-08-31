//! First end-to-end transport step on the GPU. Combines the Phase A
//! free-flight sampling with the Phase B XS lookup into one kernel:
//! per particle, look up `Σ_total(E)` from a tabulated grid, then
//! sample `d = -ln(ξ) / Σ(E)`.
//!
//! This is the smallest kernel that does what a real transport step
//! does for the "distance to next collision" sub-problem. Subsequent
//! sub-branches add: per-cell tally accumulation (Phase D wired in),
//! geometry distance-to-surface (Phase C), collision physics and
//! reaction sampling (Phase E).
//!
//! The kernel uses only the validated subset of cubecl-spirv ops:
//! - Polyfill `ln_f64` for the natural log (cubecl-spirv's own `ln` is
//!   broken on AMD/RADV, see cubecl#1316).
//! - Inline PCG-32 for the random sample.
//! - u32 binary search with the counter-bounded `while` shape that
//!   cubecl's macro accepts.
//! - Plain f64 arithmetic (1 ulp drift from CPU due to FMA contraction).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Per-particle transport step. Each thread reads its RNG seed and
/// energy, looks up Σ(E) from the shared grid, samples one free flight,
/// writes the distance. Handles only the in-range case for the energy
/// query (same first-cut assumption as `xs_lookup.rs`).
#[cube(launch_unchecked)]
fn transport_step_kernel(
    seeds: &[u32],
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
    distances: &mut [f64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    // Inline PCG-32 -- same constants as `kernels::pcg32`. cubecl `&mut`
    // bindings alias live state instead of snapshotting, so factoring
    // this into a helper would silently shift every output by one step.
    let oldstate = expand_seed(seeds[ABSOLUTE_POS]);
    let _next_state = oldstate * PCG_MULT + PCG_INCR;
    let r_u32 = pcg_out(oldstate);
    let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    // XS lookup: log-energy binary search + linear interpolation. Same
    // shape as `xs_lookup.rs`; counter-bounded while loop to keep
    // cubecl's macro happy.
    let log_e = ln_f64(energies[ABSOLUTE_POS]);
    let mut lo = 0u32;
    let mut hi = log_energy_grid.len() as u32;
    let mut iter = 0u32;
    while iter < 32u32 && lo < hi {
        let mid: u32 = (lo + hi) / 2u32;
        if log_energy_grid[mid as usize] < log_e {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
        iter += 1u32;
    }
    let idx_hi: u32 = lo;
    let idx_lo: u32 = lo - 1u32;
    let x_lo = log_energy_grid[idx_lo as usize];
    let x_hi = log_energy_grid[idx_hi as usize];
    let y_lo = xs_values[idx_lo as usize];
    let y_hi = xs_values[idx_hi as usize];
    let sigma = y_lo + (y_hi - y_lo) * (log_e - x_lo) / (x_hi - x_lo);

    distances[ABSOLUTE_POS] = -ln_f64(xi) / sigma;
}

/// Run the combined transport-step kernel. Each `seeds[i]` and
/// `energies[i]` make one independent particle; outputs are sampled
/// free flights in the same order.
pub fn run_transport_step(
    ctx: &GpuContext,
    seeds: &[u32],
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
) -> Vec<f64> {
    assert_eq!(
        seeds.len(),
        energies.len(),
        "seeds and energies must be the same length"
    );
    assert_eq!(
        log_energy_grid.len(),
        xs_values.len(),
        "log_energy_grid and xs_values must have the same length"
    );
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_handle = client.create_from_slice(bytemuck::cast_slice(energies));
    let log_grid_handle = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_handle = client.create_from_slice(bytemuck::cast_slice(xs_values));
    let out_handle = client.empty(std::mem::size_of_val(energies));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        transport_step_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(energies_handle, n),
            BufferArg::from_raw_parts(log_grid_handle, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_handle, xs_values.len()),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// CPU equivalent of `run_transport_step` for benchmarking and testing.
/// Uses the same PCG-32 algorithm and the same binary-search +
/// log-linear interpolation as the GPU kernel; libm `ln` instead of
/// the polyfill, so results may differ by a few ulps.
pub fn run_transport_step_cpu(
    seeds: &[u32],
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
) -> Vec<f64> {
    assert_eq!(seeds.len(), energies.len());
    assert_eq!(log_energy_grid.len(), xs_values.len());
    let n_grid = log_energy_grid.len();

    seeds
        .iter()
        .zip(energies.iter())
        .map(|(&seed, &e)| {
            let oldstate = crate::common::rng::expand_seed(seed);
            let _next = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r_u32 = pcg_out(oldstate);
            let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

            let log_e = e.ln();
            let lo = log_energy_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, n_grid - 1);
            let idx_lo = idx_hi - 1;
            let x_lo = log_energy_grid[idx_lo];
            let x_hi = log_energy_grid[idx_hi];
            let y_lo = xs_values[idx_lo];
            let y_hi = xs_values[idx_hi];
            let sigma = y_lo + (y_hi - y_lo) * (log_e - x_lo) / (x_hi - x_lo);

            -xi.ln() / sigma
        })
        .collect()
}

/// Rayon-parallel CPU equivalent.
pub fn run_transport_step_cpu_rayon(
    seeds: &[u32],
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
) -> Vec<f64> {
    use rayon::prelude::*;
    assert_eq!(seeds.len(), energies.len());
    assert_eq!(log_energy_grid.len(), xs_values.len());
    let n_grid = log_energy_grid.len();

    seeds
        .par_iter()
        .zip(energies.par_iter())
        .map(|(&seed, &e)| {
            let oldstate = crate::common::rng::expand_seed(seed);
            let _next = oldstate.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let r_u32 = pcg_out(oldstate);
            let xi = (r_u32 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

            let log_e = e.ln();
            let lo = log_energy_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, n_grid - 1);
            let idx_lo = idx_hi - 1;
            let x_lo = log_energy_grid[idx_lo];
            let x_hi = log_energy_grid[idx_hi];
            let y_lo = xs_values[idx_lo];
            let y_hi = xs_values[idx_hi];
            let sigma = y_lo + (y_hi - y_lo) * (log_e - x_lo) / (x_hi - x_lo);

            -xi.ln() / sigma
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Smooth analytic XS: `Σ(E) = 2 + sin(ln E)`. Bounded in [1, 3];
    /// gives reasonable distances for any in-range energy.
    fn analytic_xs(log_e: f64) -> f64 {
        2.0 + log_e.sin()
    }

    /// GPU and CPU must agree per-particle to within ~10 ulps. Drift
    /// sources: GPU uses `ln_polyfill` twice (energy lookup + free
    /// flight) where CPU uses libm twice; FMA contraction in the
    /// interpolation and divide chain. 10 ulps is comfortable; a real
    /// codegen bug would push this much higher.
    #[test]
    fn gpu_transport_step_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let n_grid = 100usize;
        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();
        let log_energy_grid: Vec<f64> = (0..n_grid)
            .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
            .collect();
        let xs_values: Vec<f64> = log_energy_grid.iter().map(|&le| analytic_xs(le)).collect();

        let n_particles = 1000usize;
        let seeds: Vec<u32> = (0..n_particles)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies: Vec<f64> = (0..n_particles)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n_particles as f64;
                (log_e_min + 0.01 + (log_e_max - 0.01 - log_e_min) * frac).exp()
            })
            .collect();

        let gpu = run_transport_step(&ctx, &seeds, &energies, &log_energy_grid, &xs_values);
        let cpu = run_transport_step_cpu(&seeds, &energies, &log_energy_grid, &xs_values);

        let mut max_ulps = 0u64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(g.is_finite() && g > 0.0, "GPU dist[{i}] = {g} not positive");
            assert!(c.is_finite() && c > 0.0, "CPU dist[{i}] = {c} not positive");
            let gi = g.to_bits();
            let ci = c.to_bits();
            let d = gi.max(ci) - gi.min(ci);
            if d > max_ulps {
                max_ulps = d;
                worst_idx = i;
            }
        }

        println!(
            "transport_step: max ULP drift = {max_ulps} (E[{worst_idx}] = {}, GPU = {}, CPU = {})",
            energies[worst_idx], gpu[worst_idx], cpu[worst_idx]
        );
        assert!(
            max_ulps <= 10,
            "GPU vs CPU transport step drifts > 10 ulps (max {max_ulps})"
        );
    }
}
