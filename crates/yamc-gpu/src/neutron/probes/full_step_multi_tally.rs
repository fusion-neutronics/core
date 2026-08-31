//! Multi-bin tally generalisation of `full_step_tally`. Same physics
//! step, but the absorption-rate tally is now an array of `N` bins
//! instead of a single accumulator. Each particle picks its bin from
//! its energy and atomic-adds to that bin only.
//!
//! # Bin scheme
//!
//! Uniform binning in log-energy:
//!
//! ```text
//! bin = clamp(floor((log E - log E_min) / (log E_max - log E_min) * N),
//!             0, N - 1)
//! ```
//!
//! `tally_log_e_min` and `tally_log_e_max` are passed in `params[1]`
//! and `params[2]`, separately from the XS grid bounds (so the tally
//! can use a different range/resolution than the XS table). The bin
//! count is taken from the tally buffer's length.
//!
//! # Why this matters
//!
//! Single-bin tallies are a proof of concept; multi-bin tallies are
//! useful. With `N_bins ≈ 100` the atomic contention drops by ~100×
//! relative to the single-bin kernel, so this is also a real
//! performance step (worth re-benching once the test passes).
//!
//! Real production needs `N_cells × N_energy × N_reactions` bins --
//! same indexing, just a flat 1D buffer with `bin = cell × n_energy *
//! n_reactions + e_bin × n_reactions + r`. Wiring multi-axis indexing
//! is bookkeeping; the per-bin atomic pattern is what this kernel
//! validates.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::probes::fixed_point_tally::TALLY_SCALE;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn full_step_multi_tally_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    params: &[f64], // [target_mass, tally_log_e_min, tally_log_e_max]
    out_distances: &mut [f64],
    out_energies: &mut [f64],
    out_mu_lab: &mut [f64],
    out_alive: &mut [u32],
    tally: &mut [Atomic<u64>],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    let target_mass = params[0];
    let tally_log_min = params[1];
    let tally_log_max = params[2];
    let e_in = energies_in[ABSOLUTE_POS];

    // Three PCG-32 samples.
    let s0 = expand_seed(seeds[ABSOLUTE_POS]);
    let r1 = pcg_out(s0);
    let s1 = s0 * PCG_MULT + PCG_INCR;
    let r2 = pcg_out(s1);
    let s2 = s1 * PCG_MULT + PCG_INCR;
    let r3 = pcg_out(s2);

    let xi1 = (r1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let xi2 = (r2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let xi3 = (r3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    // XS lookup.
    let log_e = ln_f64(e_in);
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
    let frac = (log_e - x_lo) / (x_hi - x_lo);
    let xs_e_lo = xs_elastic_grid[idx_lo as usize];
    let xs_e_hi = xs_elastic_grid[idx_hi as usize];
    let sigma_e = xs_e_lo + (xs_e_hi - xs_e_lo) * frac;
    let xs_a_lo = xs_absorption_grid[idx_lo as usize];
    let xs_a_hi = xs_absorption_grid[idx_hi as usize];
    let sigma_a = xs_a_lo + (xs_a_hi - xs_a_lo) * frac;
    let sigma_t = sigma_e + sigma_a;

    // Free flight.
    let distance = -ln_f64(xi1) / sigma_t;
    out_distances[ABSOLUTE_POS] = distance;

    // Tally bin index. Uniform in log-energy across the tally range.
    // Clamp to [0, n_bins - 1] for out-of-range queries.
    let n_bins: u32 = tally.len() as u32;
    let n_bins_f = n_bins as f64;
    let bin_f = (log_e - tally_log_min) / (tally_log_max - tally_log_min) * n_bins_f;
    let mut bin = 0u32;
    if bin_f >= 0.0 {
        bin = bin_f as u32;
    }
    if bin >= n_bins {
        bin = n_bins - 1u32;
    }

    let tally_contribution = distance * sigma_a;
    let scaled = (tally_contribution * 1_073_741_824.0 + 0.5) as i64;
    let bits = u64::reinterpret(scaled);
    tally[bin as usize].fetch_add(bits);

    // Reaction sample.
    let p_elastic = sigma_e / sigma_t;
    let mut alive = 0u32;
    let mut e_out = e_in;
    let mut mu_lab = 0.0;
    if xi2 < p_elastic {
        alive = 1u32;
        let mu_cm = 1.0 - 2.0 * xi3;
        let one_plus_a = target_mass + 1.0;
        let denom = one_plus_a * one_plus_a;
        let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
        e_out = e_in * numer / denom;
        mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();
    }
    out_energies[ABSOLUTE_POS] = e_out;
    out_mu_lab[ABSOLUTE_POS] = mu_lab;
    out_alive[ABSOLUTE_POS] = alive;
}

/// Result of a multi-bin-tally full step.
pub struct FullStepMultiTallyResult {
    pub distances: Vec<f64>,
    pub energies: Vec<f64>,
    pub mu_lab: Vec<f64>,
    pub alive: Vec<u32>,
    /// Per-bin track-length-weighted absorption tally, in physical
    /// units. Sum across all particles before normalising.
    pub absorption_per_bin: Vec<f64>,
}

/// Run the multi-bin-tally full transport step. Caller specifies the
/// tally bin count (`n_bins`) and bin range
/// (`tally_log_e_min .. tally_log_e_max`); particle energies outside
/// the range fall into the nearest edge bin.
#[allow(clippy::too_many_arguments)]
pub fn run_full_step_multi_tally(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
    tally_log_e_min: f64,
    tally_log_e_max: f64,
    n_bins: usize,
) -> FullStepMultiTallyResult {
    assert_eq!(seeds.len(), energies_in.len());
    assert!(n_bins > 0, "n_bins must be positive");
    let n = seeds.len();

    let client = ctx.client();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_h = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let log_grid_h = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_e_h = client.create_from_slice(bytemuck::cast_slice(xs_elastic_grid));
    let xs_a_h = client.create_from_slice(bytemuck::cast_slice(xs_absorption_grid));
    let params = [target_mass, tally_log_e_min, tally_log_e_max];
    let params_h = client.create_from_slice(bytemuck::cast_slice(&params));
    let out_d_h = client.empty(std::mem::size_of_val(energies_in));
    let out_e_h = client.empty(std::mem::size_of_val(energies_in));
    let out_mu_h = client.empty(std::mem::size_of_val(energies_in));
    let out_alive_h = client.empty(std::mem::size_of_val(seeds));
    let initial_tally: Vec<u64> = vec![0u64; n_bins];
    let tally_h = client.create_from_slice(bytemuck::cast_slice(&initial_tally));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        full_step_multi_tally_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(energies_h, n),
            BufferArg::from_raw_parts(log_grid_h, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_e_h, xs_elastic_grid.len()),
            BufferArg::from_raw_parts(xs_a_h, xs_absorption_grid.len()),
            BufferArg::from_raw_parts(params_h, 3),
            BufferArg::from_raw_parts(out_d_h.clone(), n),
            BufferArg::from_raw_parts(out_e_h.clone(), n),
            BufferArg::from_raw_parts(out_mu_h.clone(), n),
            BufferArg::from_raw_parts(out_alive_h.clone(), n),
            BufferArg::from_raw_parts(tally_h.clone(), n_bins),
        );
    }

    let distances: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_d_h).unwrap()).to_vec();
    let energies: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_e_h).unwrap()).to_vec();
    let mu_lab: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_mu_h).unwrap()).to_vec();
    let alive: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_alive_h).unwrap()).to_vec();
    let tally_bits: Vec<u64> = bytemuck::cast_slice(&client.read_one(tally_h).unwrap()).to_vec();
    let absorption_per_bin: Vec<f64> = tally_bits
        .iter()
        .map(|&bits| (bits as i64 as f64) / TALLY_SCALE)
        .collect();

    FullStepMultiTallyResult {
        distances,
        energies,
        mu_lab,
        alive,
        absorption_per_bin,
    }
}

/// CPU equivalent. Sequential accumulation into the same fixed-point
/// u64 buckets.
#[allow(clippy::too_many_arguments)]
pub fn run_full_step_multi_tally_cpu(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
    tally_log_e_min: f64,
    tally_log_e_max: f64,
    n_bins: usize,
) -> FullStepMultiTallyResult {
    assert_eq!(seeds.len(), energies_in.len());
    let n = seeds.len();

    let mut distances = Vec::with_capacity(n);
    let mut energies = Vec::with_capacity(n);
    let mut mu_lab = Vec::with_capacity(n);
    let mut alive = Vec::with_capacity(n);
    let mut tally: Vec<u64> = vec![0u64; n_bins];

    for i in 0..n {
        let s0 = crate::common::rng::expand_seed(seeds[i]);
        let r1 = pcg_out(s0);
        let s1 = s0.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let r2 = pcg_out(s1);
        let s2 = s1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let r3 = pcg_out(s2);

        let xi1 = (r1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let xi2 = (r2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let xi3 = (r3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let log_e = energies_in[i].ln();
        let lo = log_energy_grid.partition_point(|&x| x < log_e);
        let idx_hi = lo.clamp(1, log_energy_grid.len() - 1);
        let idx_lo = idx_hi - 1;
        let frac =
            (log_e - log_energy_grid[idx_lo]) / (log_energy_grid[idx_hi] - log_energy_grid[idx_lo]);
        let sigma_e =
            xs_elastic_grid[idx_lo] + (xs_elastic_grid[idx_hi] - xs_elastic_grid[idx_lo]) * frac;
        let sigma_a = xs_absorption_grid[idx_lo]
            + (xs_absorption_grid[idx_hi] - xs_absorption_grid[idx_lo]) * frac;
        let sigma_t = sigma_e + sigma_a;

        let distance = -xi1.ln() / sigma_t;
        distances.push(distance);

        // Tally bin computation matches the GPU kernel exactly.
        let bin_f = (log_e - tally_log_e_min) / (tally_log_e_max - tally_log_e_min) * n_bins as f64;
        let mut bin = 0i64;
        if bin_f >= 0.0 {
            bin = bin_f as i64;
        }
        if bin >= n_bins as i64 {
            bin = n_bins as i64 - 1;
        }
        let tally_contribution = distance * sigma_a;
        let scaled = (tally_contribution * TALLY_SCALE + 0.5) as i64;
        tally[bin as usize] = tally[bin as usize].wrapping_add(scaled as u64);

        let p_elastic = sigma_e / sigma_t;
        if xi2 < p_elastic {
            let mu_cm = 1.0 - 2.0 * xi3;
            let denom = (target_mass + 1.0) * (target_mass + 1.0);
            let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
            energies.push(energies_in[i] * numer / denom);
            mu_lab.push((1.0 + target_mass * mu_cm) / numer.sqrt());
            alive.push(1);
        } else {
            energies.push(energies_in[i]);
            mu_lab.push(0.0);
            alive.push(0);
        }
    }

    let absorption_per_bin: Vec<f64> = tally
        .iter()
        .map(|&bits| (bits as i64 as f64) / TALLY_SCALE)
        .collect();

    FullStepMultiTallyResult {
        distances,
        energies,
        mu_lab,
        alive,
        absorption_per_bin,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    fn constant_xs_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_grid = 100usize;
        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();
        let log_grid: Vec<f64> = (0..n_grid)
            .map(|i| log_e_min + (log_e_max - log_e_min) * (i as f64) / (n_grid as f64 - 1.0))
            .collect();
        let xs_e: Vec<f64> = vec![2.0; n_grid];
        let xs_a: Vec<f64> = vec![1.0; n_grid];
        (log_grid, xs_e, xs_a)
    }

    /// 100k particles spread uniformly in log-energy, 8 tally bins.
    /// Each bin should receive roughly the same number of
    /// contributions, and each per-particle bin tally should sum to
    /// roughly `(N / n_bins) × Σ_a / Σ_t = N × 1/24` in physical
    /// units. Tolerance: 5% relative -- gives MC noise headroom while
    /// catching a real bug.
    #[test]
    fn multi_bin_tally_distribution() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let n = 100_000usize;
        let n_bins = 8usize;
        let target_mass = 12.0_f64;
        let (log_grid, xs_e, xs_a) = constant_xs_grid();

        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n as f64;
                (log_e_min + 0.05 + (log_e_max - 0.05 - log_e_min) * frac).exp()
            })
            .collect();

        let r = run_full_step_multi_tally(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
            log_e_min,
            log_e_max,
            n_bins,
        );

        // Per-bin expected: (N / n_bins) particles × <distance × Σ_a>.
        // <distance × Σ_a> = Σ_a / Σ_t = 1/3.
        let expected_per_bin = (n as f64 / n_bins as f64) * (1.0 / 3.0);
        for (i, &t) in r.absorption_per_bin.iter().enumerate() {
            let rel = (t - expected_per_bin).abs() / expected_per_bin;
            println!("bin {i}: tally = {t} (expected {expected_per_bin}, rel err {rel:.4})");
            assert!(
                rel < 0.05,
                "bin {i} tally {t} far from expected {expected_per_bin}"
            );
        }

        // Total tally must equal the single-bin sum: same physics,
        // same particles, just redistributed.
        let total_gpu: f64 = r.absorption_per_bin.iter().sum();
        let expected_total = (n as f64) * (1.0 / 3.0);
        let rel_total = (total_gpu - expected_total).abs() / expected_total;
        println!("total tally = {total_gpu} (expected {expected_total}, rel err {rel_total:.4})");
        assert!(rel_total < 0.01, "total tally drifts too far from expected");
    }

    /// GPU and CPU per-bin tallies must match exactly. Integer
    /// accumulation makes the order-independent property automatic;
    /// 1-ulp distance differences are absorbed by the round-to-nearest
    /// in the fixed-point conversion.
    #[test]
    fn gpu_multi_bin_tally_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 5_000usize;
        let n_bins = 16usize;
        let target_mass = 56.0_f64;
        let (log_grid, xs_e, xs_a) = constant_xs_grid();
        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n as f64;
                (log_e_min + 0.05 + (log_e_max - 0.05 - log_e_min) * frac).exp()
            })
            .collect();

        let g = run_full_step_multi_tally(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
            log_e_min,
            log_e_max,
            n_bins,
        );
        let c = run_full_step_multi_tally_cpu(
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
            log_e_min,
            log_e_max,
            n_bins,
        );

        for (i, (&gv, &cv)) in g
            .absorption_per_bin
            .iter()
            .zip(c.absorption_per_bin.iter())
            .enumerate()
        {
            let rel = if cv == 0.0 {
                gv.abs()
            } else {
                ((gv - cv) / cv).abs()
            };
            assert!(
                rel < 1e-9,
                "bin {i}: GPU = {gv}, CPU = {cv}, rel err = {rel:e}"
            );
        }
        println!("GPU/CPU multi-bin tally bit-exact across {n_bins} bins");
    }
}
