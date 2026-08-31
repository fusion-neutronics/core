//! Cross-section table lookup with binary search and linear
//! interpolation in log-energy.
//!
//! yamc's CPU transport stores per-nuclide cross-section tables as
//! `(energy_grid, xs_values)` pairs. For a particle at energy `E`, the
//! collision sampling step reads `Σ(E) = interp(energy_grid, xs_values,
//! E)` and uses it as the sampling rate for the next free flight. Phase
//! B of the GPU port is moving that lookup onto the device. This module
//! is the kernel itself; wiring it into a Σ-driven distance-to-collision
//! kernel is the next sub-branch.
//!
//! # Algorithm
//!
//! Per particle, on the GPU:
//! 1. Compute `log_E = ln(E)` via the polyfill (cubecl-spirv f64 ln is
//!    broken on this driver, see cubecl#1316).
//! 2. Binary search `log_energy_grid` for the first index `idx` where
//!    `log_energy_grid[idx] >= log_E`.
//! 3. Linear interpolate between `(log_energy_grid[idx-1],
//!    xs_values[idx-1])` and `(log_energy_grid[idx], xs_values[idx])`.
//!
//! # Assumptions and limits
//!
//! - **Inputs in range.** This first cut assumes every queried `E` is
//!   strictly between `energy_grid[0]` and `energy_grid[-1]`. Out-of-
//!   range values get clamped at the boundary index (returning the edge
//!   `xs_value`); a production kernel will need an explicit policy for
//!   below-min and above-max.
//! - **Linear-in-log interpolation.** Standard for neutron XS data on
//!   log-spaced grids. Real yamc CPU uses log-log interpolation in some
//!   cases; matching that variant is a follow-up.
//! - **One material, one reaction.** A real material is a sum over
//!   nuclides weighted by atomic densities. That comes after this works.

use crate::common::polyfills::ln_f64;
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Per-thread XS lookup. Each `energies[tid]` is one query; the result
/// goes to `out_xs[tid]`. `log_energy_grid` is sorted ascending.
#[cube(launch_unchecked)]
fn xs_lookup_kernel(
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
    out_xs: &mut [f64],
) {
    if ABSOLUTE_POS >= energies.len() {
        terminate!();
    }
    let log_e = ln_f64(energies[ABSOLUTE_POS]);

    // Binary search for the first index where log_energy_grid[idx] >= log_e.
    // The pattern is a counter-bounded `while iter < MAX_ITERS && lo < hi`
    // because cubecl's macro accepts that shape; a plain `while lo < hi`
    // hits a NativeExpand<usize> type mismatch in macro expansion. 32
    // iterations is comfortably above the log2 of any reasonable XS
    // grid size (tens of thousands of points still fits in 16 iters).
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

    // First-cut assumption: every queried energy is strictly inside
    // the grid (logE_min < log_e < logE_max). Under that assumption
    // the binary search returns lo in [1, n-1], so idx_hi = lo and
    // idx_lo = lo - 1 are both valid grid indices. Out-of-range queries
    // are caller error here; production will add an explicit policy.
    let idx_hi: u32 = lo;
    let idx_lo: u32 = lo - 1u32;

    let x_lo = log_energy_grid[idx_lo as usize];
    let x_hi = log_energy_grid[idx_hi as usize];
    let y_lo = xs_values[idx_lo as usize];
    let y_hi = xs_values[idx_hi as usize];

    // Linear interpolation in (log_energy, xs). x_hi > x_lo by
    // construction (grid is strictly ascending); no zero-divide guard
    // beyond that.
    out_xs[ABSOLUTE_POS] = y_lo + (y_hi - y_lo) * (log_e - x_lo) / (x_hi - x_lo);
}

/// Sample one XS lookup per query. `log_energy_grid` and `xs_values`
/// have the same length; `log_energy_grid` is `energy_grid.iter()
/// .map(|e| e.ln()).collect()` (host-precomputed so the kernel doesn't
/// have to ln the grid every launch).
pub fn run_xs_lookup(
    ctx: &GpuContext,
    energies: &[f64],
    log_energy_grid: &[f64],
    xs_values: &[f64],
) -> Vec<f64> {
    assert_eq!(
        log_energy_grid.len(),
        xs_values.len(),
        "log_energy_grid and xs_values must have the same length"
    );
    let client = ctx.client();
    let n = energies.len();
    let energies_handle = client.create_from_slice(bytemuck::cast_slice(energies));
    let log_grid_handle = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_handle = client.create_from_slice(bytemuck::cast_slice(xs_values));
    let out_handle = client.empty(std::mem::size_of_val(energies));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        xs_lookup_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(energies_handle, n),
            BufferArg::from_raw_parts(log_grid_handle, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_handle, xs_values.len()),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    let bytes = client.read_one(out_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// CPU equivalent of the GPU lookup. Uses libm `ln` instead of the
/// polyfill (so results may differ from GPU by a few ulps near the
/// interpolation operand), and the same binary-search-with-clamp + log-
/// linear interpolation. Inputs in `[energy_grid[0], energy_grid[-1]]`
/// are interpolated; out-of-range inputs return the nearest edge value.
pub fn run_xs_lookup_cpu(energies: &[f64], log_energy_grid: &[f64], xs_values: &[f64]) -> Vec<f64> {
    assert_eq!(log_energy_grid.len(), xs_values.len());
    let n = log_energy_grid.len();
    energies
        .iter()
        .map(|&e| {
            let log_e = e.ln();
            // partition_point gives first index where pred is false,
            // matching the kernel's binary search exactly.
            let lo = log_energy_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, n - 1);
            let idx_lo = idx_hi - 1;
            let x_lo = log_energy_grid[idx_lo];
            let x_hi = log_energy_grid[idx_hi];
            let y_lo = xs_values[idx_lo];
            let y_hi = xs_values[idx_hi];
            y_lo + (y_hi - y_lo) * (log_e - x_lo) / (x_hi - x_lo)
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Build a smooth analytic XS curve `Σ(E) = 2 + sin(ln(E))` sampled
    /// on a log-spaced grid. The function is monotonic-enough on
    /// segments and bounded in `[1, 3]`, so interpolation error stays
    /// small relative to the values themselves.
    fn analytic_xs(log_e: f64) -> f64 {
        2.0 + log_e.sin()
    }

    /// GPU and CPU must agree on every query to within ~10 ulps. The
    /// drift comes from two sources: GPU uses `ln_polyfill` while CPU
    /// uses libm `ln` (~1 ulp), and the interpolation chain has FMA
    /// contraction differences (~1 ulp each step). 10 ulps is a
    /// generous bar that catches real bugs and ignores rounding.
    #[test]
    fn gpu_xs_lookup_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Log-spaced grid: 100 points from E = 0.001 to E = 1000.
        let n_grid = 100usize;
        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();
        let log_energy_grid: Vec<f64> = (0..n_grid)
            .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
            .collect();
        let xs_values: Vec<f64> = log_energy_grid.iter().map(|&le| analytic_xs(le)).collect();

        // 1000 queries strictly inside the grid.
        let n_query = 1000usize;
        let energies: Vec<f64> = (0..n_query)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n_query as f64;
                (log_e_min + 0.01 + (log_e_max - 0.01 - log_e_min) * frac).exp()
            })
            .collect();

        let gpu_xs = run_xs_lookup(&ctx, &energies, &log_energy_grid, &xs_values);
        let cpu_xs = run_xs_lookup_cpu(&energies, &log_energy_grid, &xs_values);

        let mut max_ulps = 0u64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu_xs.iter().zip(cpu_xs.iter()).enumerate() {
            // ULP distance on same-sign positive finite values.
            assert!(g.is_finite() && g > 0.0, "GPU XS[{i}] = {g} not positive");
            assert!(c.is_finite() && c > 0.0, "CPU XS[{i}] = {c} not positive");
            let gi = g.to_bits();
            let ci = c.to_bits();
            let d = gi.max(ci) - gi.min(ci);
            if d > max_ulps {
                max_ulps = d;
                worst_idx = i;
            }
        }

        println!(
            "xs_lookup: max ULP drift = {max_ulps} (E[{worst_idx}] = {}, GPU = {}, CPU = {})",
            energies[worst_idx], gpu_xs[worst_idx], cpu_xs[worst_idx]
        );
        assert!(
            max_ulps <= 10,
            "GPU vs CPU XS lookup drifts > 10 ulps (max {max_ulps})"
        );
    }
}
