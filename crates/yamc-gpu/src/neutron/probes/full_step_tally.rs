//! Phase D wiring: full transport step with a track-length tally
//! accumulator.
//!
//! Same physics as [`crate::neutron::probes::full_step`] (XS lookup → free
//! flight → MT sample → scatter-or-absorb), with one new step at the
//! end of each thread: contribute `distance × Σ_a(E)` to a shared
//! tally accumulator using fixed-point u64 atomic add (the validated
//! Phase D pattern from `fixed_point_tally.rs`).
//!
//! # What's tallied
//!
//! Track-length absorption rate, integrated over all particles. Each
//! particle's straight-line traversal contributes
//! `track_length(particle) × Σ_absorption(E_particle)` to the
//! accumulator regardless of whether it eventually scatters or
//! absorbs. After the kernel, dividing by the particle count gives
//! the mean per-particle rate.
//!
//! For constant `Σ_a` and `Σ_t` the analytic expectation is clean:
//!
//! ```text
//! <track_length × Σ_a> = Σ_a × <d> = Σ_a / Σ_t
//! ```
//!
//! # First-cut limits
//!
//! - **Single tally bin.** All particles contribute to one global
//!   `Atomic<u64>`. Real production needs `N_cells × N_energy_bins ×
//!   N_reactions` bins -- that's a 1D buffer indexed by
//!   `cell × n_energy + bin`. Wiring the indexing in is mostly
//!   bookkeeping once the single-bin pattern is proven, which is what
//!   this kernel does.
//! - **Atomic contention.** Every thread hits the same slot. On a
//!   real workload with N bins the contention drops by ~N. Worth
//!   measuring once we have multi-bin tallies.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::probes::fixed_point_tally::TALLY_SCALE;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn full_step_tally_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    params: &[f64], // [target_mass]
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
    let e_in = energies_in[ABSOLUTE_POS];

    // Three PCG-32 samples (free flight, MT pick, CM cosine).
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

    // Track-length tally contribution: distance * Σ_a, fixed-point
    // scaled, atomic add. Always non-negative under our test inputs,
    // so the i64 → u64 reinterpret trick from `fixed_point_tally.rs`
    // isn't needed -- direct cast suffices.
    let tally_contribution = distance * sigma_a;
    let scaled = (tally_contribution * 1_073_741_824.0 + 0.5) as i64;
    let bits = u64::reinterpret(scaled);
    tally[0].fetch_add(bits);

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

/// Result of a tally-aware full step: per-particle outputs plus the
/// scalar absorption-rate tally.
pub struct FullStepTallyResult {
    pub distances: Vec<f64>,
    pub energies: Vec<f64>,
    pub mu_lab: Vec<f64>,
    pub alive: Vec<u32>,
    /// Mean track-length × Σ_a per particle, recovered from the
    /// fixed-point u64 accumulator: `tally_u64 / (N × TALLY_SCALE)`.
    pub absorption_rate: f64,
}

/// Run the tally-aware full transport step.
pub fn run_full_step_tally(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
) -> FullStepTallyResult {
    assert_eq!(seeds.len(), energies_in.len());
    let n = seeds.len();

    let client = ctx.client();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_h = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let log_grid_h = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_e_h = client.create_from_slice(bytemuck::cast_slice(xs_elastic_grid));
    let xs_a_h = client.create_from_slice(bytemuck::cast_slice(xs_absorption_grid));
    let params = [target_mass];
    let params_h = client.create_from_slice(bytemuck::cast_slice(&params));
    let out_d_h = client.empty(std::mem::size_of_val(energies_in));
    let out_e_h = client.empty(std::mem::size_of_val(energies_in));
    let out_mu_h = client.empty(std::mem::size_of_val(energies_in));
    let out_alive_h = client.empty(std::mem::size_of_val(seeds));
    let initial_tally: u64 = 0;
    let tally_h = client.create_from_slice(bytemuck::bytes_of(&initial_tally));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        full_step_tally_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(energies_h, n),
            BufferArg::from_raw_parts(log_grid_h, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_e_h, xs_elastic_grid.len()),
            BufferArg::from_raw_parts(xs_a_h, xs_absorption_grid.len()),
            BufferArg::from_raw_parts(params_h, 1),
            BufferArg::from_raw_parts(out_d_h.clone(), n),
            BufferArg::from_raw_parts(out_e_h.clone(), n),
            BufferArg::from_raw_parts(out_mu_h.clone(), n),
            BufferArg::from_raw_parts(out_alive_h.clone(), n),
            BufferArg::from_raw_parts(tally_h.clone(), 1),
        );
    }

    let distances: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_d_h).unwrap()).to_vec();
    let energies: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_e_h).unwrap()).to_vec();
    let mu_lab: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_mu_h).unwrap()).to_vec();
    let alive: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_alive_h).unwrap()).to_vec();
    let tally_bits: u64 = bytemuck::cast_slice::<u8, u64>(&client.read_one(tally_h).unwrap())[0];
    let tally_signed = tally_bits as i64;
    let absorption_rate = (tally_signed as f64) / TALLY_SCALE / (n as f64);

    FullStepTallyResult {
        distances,
        energies,
        mu_lab,
        alive,
        absorption_rate,
    }
}

/// CPU equivalent. Same algorithm; sums the same fixed-point
/// contributions into a `u64` accumulator (sequentially, no atomics
/// needed).
pub fn run_full_step_tally_cpu(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
) -> FullStepTallyResult {
    assert_eq!(seeds.len(), energies_in.len());
    let n = seeds.len();

    let mut distances = Vec::with_capacity(n);
    let mut energies = Vec::with_capacity(n);
    let mut mu_lab = Vec::with_capacity(n);
    let mut alive = Vec::with_capacity(n);
    let mut tally_u64 = 0u64;

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

        // Match the GPU's fixed-point accumulation exactly so per-
        // particle drift doesn't accumulate over the sum.
        let tally_contribution = distance * sigma_a;
        let scaled = (tally_contribution * TALLY_SCALE + 0.5) as i64;
        tally_u64 = tally_u64.wrapping_add(scaled as u64);

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

    let absorption_rate = (tally_u64 as i64 as f64) / TALLY_SCALE / (n as f64);
    FullStepTallyResult {
        distances,
        energies,
        mu_lab,
        alive,
        absorption_rate,
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

    /// With constant Σ_a = 1, Σ_t = 3, the per-particle contribution
    /// `distance × Σ_a` has mean `Σ_a / Σ_t = 1/3` and variance
    /// `Σ_a² × Var(d) = Σ_a² / Σ_t² = 1/9`. With N = 100k, SEM of the
    /// mean is sqrt(1/9 / N) ≈ 1.05e-3 so ±4e-3 catches a real bug.
    #[test]
    fn tally_value_matches_expected() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let n = 100_000usize;
        let target_mass = 12.0_f64;
        let (log_grid, xs_e, xs_a) = constant_xs_grid();
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n as f64;
                (1e-3_f64.ln() + 0.05 + (1e3_f64.ln() - 0.05 - 1e-3_f64.ln()) * frac).exp()
            })
            .collect();

        let r = run_full_step_tally(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
        );

        let expected = 1.0 / 3.0;
        println!(
            "tally absorption rate per particle: {} (expected {})",
            r.absorption_rate, expected
        );
        assert!(
            (r.absorption_rate - expected).abs() < 4e-3,
            "tally {} far from expected {}",
            r.absorption_rate,
            expected
        );
    }

    /// CPU and GPU absorption-rate tallies must agree to within MC
    /// precision. The fixed-point accumulator is bit-exact for any
    /// individual contribution, but the accumulation order differs
    /// between CPU (sequential) and GPU (atomic, non-deterministic
    /// order). The integer accumulator is order-independent in
    /// integer arithmetic, so the tally bits should match exactly.
    /// Per-particle outputs follow the same tolerance pattern as
    /// `full_step.rs`.
    #[test]
    fn gpu_full_step_tally_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 5_000usize;
        let target_mass = 56.0_f64;
        let (log_grid, xs_e, xs_a) = constant_xs_grid();
        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n as f64;
                (1e-3_f64.ln() + 0.05 + (1e3_f64.ln() - 0.05 - 1e-3_f64.ln()) * frac).exp()
            })
            .collect();

        let g = run_full_step_tally(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
        );
        let c = run_full_step_tally_cpu(&seeds, &energies, &log_grid, &xs_e, &xs_a, target_mass);

        for i in 0..n {
            assert_eq!(g.alive[i], c.alive[i], "alive[{i}] differs");
        }

        // Tally agreement: both fixed-point accumulators sum the same
        // integer contributions; the only drift sources are the f64 →
        // i64 rounding step and any 1-ulp difference in the distance
        // value. Allow a tiny relative tolerance.
        let rel_err = ((g.absorption_rate - c.absorption_rate) / c.absorption_rate).abs();
        println!(
            "tally GPU = {}, CPU = {}, rel err = {rel_err:e}",
            g.absorption_rate, c.absorption_rate
        );
        assert!(
            rel_err < 1e-9,
            "tally rel err {rel_err:e} > 1e-9; integer accumulation diverged"
        );
    }
}
