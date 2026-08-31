//! Full transport step composed from validated pieces -- Phase A
//! (free-flight sampling), Phase B (XS lookup), Phase E.1 (reaction
//! sampling), Phase E.2 (elastic-scatter kinematics) -- into a single
//! kernel.
//!
//! Per particle:
//! 1. **XS lookup**: log-energy binary search + linear interp on
//!    `xs_elastic_grid` and `xs_absorption_grid`. `Σ_t = Σ_e + Σ_a`.
//! 2. **Free flight**: `d = -ln(ξ₁) / Σ_t`.
//! 3. **MT sample**: scatter if `ξ₂ < Σ_e / Σ_t`, else absorption.
//! 4. **If scatter** (E.2): sample `µ_cm = 1 - 2 ξ₃`, compute new
//!    energy `E'` and lab cosine `µ_lab` via standard elastic
//!    formulas. **If absorbed**: leave energy unchanged, `µ_lab = 0`,
//!    flag the particle dead.
//!
//! Three RNG samples per particle are produced from one stored seed by
//! advancing the PCG-32 state explicitly across each step. Cubecl
//! `&mut` bindings alias live state instead of snapshotting like Rust
//! does (see `kernels::pcg32`), so we inline three full PCG-32 steps
//! rather than calling a helper. Verbose, but unambiguous.
//!
//! The 3D direction rotation (which uses the new `µ_lab` plus `sin`/
//! `cos` of the azimuthal angle) is intentionally not part of this
//! kernel -- Phase E.3 lands once the `sin`/`cos` story is settled.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Per-particle full step: XS lookup → free flight → MT sample →
/// scatter-or-absorb. See module docs for the algorithm.
#[cube(launch_unchecked)]
fn full_step_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    params: &[f64], // params[0] = target_mass A
    out_distances: &mut [f64],
    out_energies: &mut [f64],
    out_mu_lab: &mut [f64],
    out_alive: &mut [u32],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    let target_mass = params[0];
    let e_in = energies_in[ABSOLUTE_POS];

    // Three PCG-32 samples advanced from `seeds[tid]`. Each step uses
    // the previous state to produce its random word and computes the
    // next state for the following step.
    let s0 = expand_seed(seeds[ABSOLUTE_POS]);
    let r1 = pcg_out(s0);
    let s1 = s0 * PCG_MULT + PCG_INCR;
    let r2 = pcg_out(s1);
    let s2 = s1 * PCG_MULT + PCG_INCR;
    let r3 = pcg_out(s2);

    let xi1 = (r1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let xi2 = (r2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let xi3 = (r3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    // XS lookup: log-energy binary search + linear interpolation.
    // Counter-bounded `while iter < MAX && lo < hi` -- same shape that
    // cubecl's macro accepts elsewhere (see `xs_lookup.rs`).
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

    // Reaction sample: scatter if `xi2 < p_elastic`, else absorb.
    let p_elastic = sigma_e / sigma_t;

    // Default to absorbed; mutate inside the `if` block. Same
    // imperative pattern that worked in `collision_sampling.rs` and
    // `sphere_distance.rs`.
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

/// Result of a full step per particle.
pub struct FullStepResult {
    pub distances: Vec<f64>,
    pub energies: Vec<f64>,
    pub mu_lab: Vec<f64>,
    pub alive: Vec<u32>,
}

/// Run the full transport-step kernel.
pub fn run_full_step(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
) -> FullStepResult {
    assert_eq!(seeds.len(), energies_in.len());
    assert_eq!(log_energy_grid.len(), xs_elastic_grid.len());
    assert_eq!(log_energy_grid.len(), xs_absorption_grid.len());
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

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        full_step_kernel::launch_unchecked::<WgpuRuntime>(
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
        );
    }

    FullStepResult {
        distances: bytemuck::cast_slice(&client.read_one(out_d_h).unwrap()).to_vec(),
        energies: bytemuck::cast_slice(&client.read_one(out_e_h).unwrap()).to_vec(),
        mu_lab: bytemuck::cast_slice(&client.read_one(out_mu_h).unwrap()).to_vec(),
        alive: bytemuck::cast_slice(&client.read_one(out_alive_h).unwrap()).to_vec(),
    }
}

/// CPU equivalent. Same algorithm; libm `ln` and `sqrt` instead of the
/// polyfill / cubecl ops. For benchmarking and equivalence testing.
pub fn run_full_step_cpu(
    seeds: &[u32],
    energies_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
) -> FullStepResult {
    assert_eq!(seeds.len(), energies_in.len());
    assert_eq!(log_energy_grid.len(), xs_elastic_grid.len());
    assert_eq!(log_energy_grid.len(), xs_absorption_grid.len());
    let n = seeds.len();

    let mut distances = Vec::with_capacity(n);
    let mut energies = Vec::with_capacity(n);
    let mut mu_lab = Vec::with_capacity(n);
    let mut alive = Vec::with_capacity(n);

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
        let x_lo = log_energy_grid[idx_lo];
        let x_hi = log_energy_grid[idx_hi];
        let frac = (log_e - x_lo) / (x_hi - x_lo);
        let sigma_e =
            xs_elastic_grid[idx_lo] + (xs_elastic_grid[idx_hi] - xs_elastic_grid[idx_lo]) * frac;
        let sigma_a = xs_absorption_grid[idx_lo]
            + (xs_absorption_grid[idx_hi] - xs_absorption_grid[idx_lo]) * frac;
        let sigma_t = sigma_e + sigma_a;

        distances.push(-xi1.ln() / sigma_t);

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

    FullStepResult {
        distances,
        energies,
        mu_lab,
        alive,
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Build a 100-point log-spaced energy grid covering 1e-3..1e3
    /// with constant XS values: σ_e = 2.0, σ_a = 1.0 at every point.
    /// With constant XS the analytic expectations are clean:
    /// alive fraction = 2/3, free-flight mean = 1/3, etc.
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

    #[test]
    fn full_step_aggregate_statistics() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let n = 100_000usize;
        let target_mass = 12.0_f64; // carbon
        let (log_grid, xs_e, xs_a) = constant_xs_grid();

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        // Energies spread across the grid (avoids edge-of-grid).
        let energies: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n as f64;
                (1e-3_f64.ln() + 0.05 + (1e3_f64.ln() - 0.05 - 1e-3_f64.ln()) * frac).exp()
            })
            .collect();

        let r = run_full_step(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
        );

        // 1. Distances all positive and finite.
        for (i, &d) in r.distances.iter().enumerate() {
            assert!(d.is_finite() && d > 0.0, "distance[{i}] = {d} not positive");
        }
        // 2. Mean distance ≈ 1 / Σ_t = 1/3. With N = 1e5 and std = 1/3,
        // SEM ≈ 1.05e-3, so |mean - 1/3| < 4e-3 is comfortably > 3σ.
        let expected_mean_d = 1.0 / 3.0;
        let mean_d = r.distances.iter().sum::<f64>() / n as f64;
        println!("full_step: mean distance = {mean_d} (expected {expected_mean_d})");
        assert!(
            (mean_d - expected_mean_d).abs() < 4e-3,
            "mean distance {mean_d} far from expected {expected_mean_d}"
        );

        // 3. Alive fraction ≈ Σ_e / Σ_t = 2/3. SEM = sqrt(p(1-p)/N) ≈
        // 1.5e-3, so a 5e-3 window is > 3σ.
        let expected_alive = 2.0 / 3.0;
        let alive_count = r.alive.iter().filter(|&&a| a == 1).count();
        let alive_frac = alive_count as f64 / n as f64;
        println!("full_step: alive fraction = {alive_frac} (expected {expected_alive})");
        assert!(
            (alive_frac - expected_alive).abs() < 5e-3,
            "alive fraction {alive_frac} far from expected {expected_alive}"
        );

        // 4. For surviving particles, E'/E in [α, 1] with α = ((11/13))².
        let alpha = ((target_mass - 1.0) / (target_mass + 1.0)).powi(2);
        for (i, ((&a, &e_in), &e_out)) in r
            .alive
            .iter()
            .zip(energies.iter())
            .zip(r.energies.iter())
            .enumerate()
        {
            if a == 1 {
                let ratio = e_out / e_in;
                assert!(
                    ratio >= alpha - 1e-12 && ratio <= 1.0 + 1e-12,
                    "alive particle {i}: E'/E = {ratio} outside [{alpha}, 1]"
                );
            } else {
                // Absorbed -- energy unchanged, mu_lab is the default 0.
                assert_eq!(e_out, e_in, "absorbed particle {i} energy changed");
                assert_eq!(r.mu_lab[i], 0.0, "absorbed particle {i} mu_lab non-zero");
            }
        }
    }

    /// GPU and CPU per-particle agreement: distance and energy on the
    /// ulp budget; mu_lab on absolute (cancellation issue from
    /// `elastic_scatter.rs`); alive flag bit-exact.
    #[test]
    fn gpu_full_step_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let n = 5_000usize;
        let target_mass = 56.0_f64; // iron, A > 1 to avoid hydrogen edge case
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

        let g = run_full_step(
            &ctx,
            &seeds,
            &energies,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
        );
        let c = run_full_step_cpu(&seeds, &energies, &log_grid, &xs_e, &xs_a, target_mass);

        // alive flag must match exactly.
        for i in 0..n {
            assert_eq!(g.alive[i], c.alive[i], "alive[{i}] differs");
        }

        // Distance: ulp budget. The chain (ln, divide) is short; allow
        // 16 ulps for FMA contraction across multiple ops.
        let mut max_d = 0u64;
        for i in 0..n {
            let gb = g.distances[i].to_bits();
            let cb = c.distances[i].to_bits();
            let d = gb.max(cb) - gb.min(cb);
            if d > max_d {
                max_d = d;
            }
        }
        assert!(max_d <= 16, "distance ulp drift {max_d}");

        // Energy of alive particles: ulp budget.
        let mut max_e = 0u64;
        for i in 0..n {
            if g.alive[i] == 1 {
                let gb = g.energies[i].to_bits();
                let cb = c.energies[i].to_bits();
                let d = gb.max(cb) - gb.min(cb);
                if d > max_e {
                    max_e = d;
                }
            }
        }
        assert!(max_e <= 16, "energy ulp drift {max_e}");

        // mu_lab: absolute tolerance (see elastic_scatter docs).
        let mut max_mu = 0.0_f64;
        for i in 0..n {
            if g.alive[i] == 1 {
                let d = (g.mu_lab[i] - c.mu_lab[i]).abs();
                if d > max_mu {
                    max_mu = d;
                }
            }
        }
        println!("full_step: max distance ulp = {max_d}, max energy ulp = {max_e}, max |mu_lab| abs = {max_mu:e}");
        assert!(max_mu < 1e-10, "mu_lab abs drift {max_mu:e}");
    }
}
