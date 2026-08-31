//! Multi-step transport: each particle loops up to `MAX_STEPS` times
//! through `(XS lookup → free flight → tally → MT sample → scatter or
//! absorb)` until it dies or hits the step cap.
//!
//! This is the macro structure of MC neutron transport. A "transport
//! step" is one collision event; a particle's history is a sequence of
//! these until termination. The full-step kernel was one collision per
//! launch; this kernel iterates the same physics inside the kernel,
//! amortising upload/download cost across many physics events per
//! particle.
//!
//! # Scope of this first cut
//!
//! - **No geometry yet.** The medium is infinite and homogeneous.
//!   Position isn't tracked; we'd need cell-finding for that, which
//!   waits on Phase C. This is fine for benchmarking the inner
//!   transport loop and for thermalisation-style sanity checks.
//! - **Single material.** One target mass `A`; same XS curves as the
//!   `full_step_*` kernels.
//! - **Termination = absorption or step cap.** No energy-cutoff
//!   termination yet; for our test workloads the absorption
//!   probability per step (≈ 1/3) keeps histories short anyway.
//! - **`MAX_STEPS` is comptime.** Recompiles per distinct value, but
//!   for the test workload one value is enough.
//!
//! # Per-step algorithm (inside the loop)
//!
//! 1. XS lookup at current energy via log-energy binary search.
//! 2. Free-flight distance: `d = -ln(ξ₁) / Σ_t`. Score
//!    `track_length × Σ_a` into the energy-binned tally.
//! 3. MT sample. If `ξ₂ ≥ Σ_e / Σ_t`, mark absorbed and break.
//! 4. Elastic scatter:
//!    - `µ_cm = 1 − 2 ξ₃` → new energy and `µ_lab` via standard
//!      formulas.
//!    - Marsaglia rejection on the unit disk for `(cos φ, sin φ)`,
//!      then 3D direction rotation. Same algorithm as
//!      `direction_rotation.rs`.
//!
//! Each step needs ≥ 5 PCG-32 advances (free flight, MT, µ_cm, plus
//! at least one (a, b) pair for rejection). Some steps need more if
//! rejection fails the first attempt; the kernel maintains a single
//! per-particle PCG state and advances it ad-hoc.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::ln_f64;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn multi_step_kernel(
    seeds: &[u32],
    energies_in: &[f64],
    directions_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    params: &[f64], // [target_mass, tally_log_e_min, tally_log_e_max]
    out_energies: &mut [f64],
    out_directions: &mut [f64],
    out_alive: &mut [u32],
    out_n_steps: &mut [u32],
    tally: &mut [Atomic<u64>],
    #[comptime] max_steps: u32,
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    let target_mass = params[0];
    let tally_log_min = params[1];
    let tally_log_max = params[2];

    let i3 = ABSOLUTE_POS * 3;
    let mut energy = energies_in[ABSOLUTE_POS];
    let mut dir_x = directions_in[i3];
    let mut dir_y = directions_in[i3 + 1];
    let mut dir_z = directions_in[i3 + 2];
    let mut state = expand_seed(seeds[ABSOLUTE_POS]);
    let mut alive = 1u32;
    let mut n_steps = 0u32;

    let mut step = 0u32;
    while step < max_steps && alive == 1u32 {
        // XS lookup at current energy.
        let log_e = ln_f64(energy);
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

        // Free-flight sample.
        let s_xi1 = state;
        let r_xi1 = pcg_out(s_xi1);
        state = s_xi1 * PCG_MULT + PCG_INCR;
        let xi1 = (r_xi1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let distance = -ln_f64(xi1) / sigma_t;

        // Tally track-length × Σ_a in the right energy bin.
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

        // MT sample.
        let s_xi2 = state;
        let r_xi2 = pcg_out(s_xi2);
        state = s_xi2 * PCG_MULT + PCG_INCR;
        let xi2 = (r_xi2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
        let p_elastic = sigma_e / sigma_t;

        if xi2 >= p_elastic {
            // Absorbed: terminate this particle.
            alive = 0u32;
        } else {
            // Elastic scatter: sample µ_cm, update energy and µ_lab.
            let s_xi3 = state;
            let r_xi3 = pcg_out(s_xi3);
            state = s_xi3 * PCG_MULT + PCG_INCR;
            let xi3 = (r_xi3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let mu_cm = 1.0 - 2.0 * xi3;

            let one_plus_a = target_mass + 1.0;
            let denom = one_plus_a * one_plus_a;
            let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
            let new_energy = energy * numer / denom;
            let mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();

            // Marsaglia rejection for (cos_phi, sin_phi). Up to 8
            // attempts; bounded retry is required for cubecl's macro
            // and astronomically unlikely to need all 8.
            let mut accepted = false;
            let mut cos_phi = 1.0;
            let mut sin_phi = 0.0;
            let mut rej_iter = 0u32;
            while rej_iter < 8u32 && !accepted {
                let s_a = state;
                let r_a = pcg_out(s_a);
                state = s_a * PCG_MULT + PCG_INCR;
                let s_b = state;
                let r_b = pcg_out(s_b);
                state = s_b * PCG_MULT + PCG_INCR;

                let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                let s = a * a + b * b;
                if s > 0.0 && s <= 1.0 {
                    let inv_sqrt_s = 1.0 / s.sqrt();
                    cos_phi = a * inv_sqrt_s;
                    sin_phi = b * inv_sqrt_s;
                    accepted = true;
                }
                rej_iter += 1u32;
            }

            // 3D direction rotation by (µ_lab, cos_phi, sin_phi).
            let sin_theta_sq = 1.0 - mu_lab * mu_lab;
            let mut sin_theta = 0.0;
            if sin_theta_sq > 0.0 {
                sin_theta = sin_theta_sq.sqrt();
            }
            let one_minus_w_sq = 1.0 - dir_z * dir_z;

            let mut new_u = sin_theta * cos_phi;
            let mut new_v = sin_theta * sin_phi;
            let mut new_w = mu_lab;
            if dir_z < 0.0 {
                new_v = -new_v;
                new_w = -mu_lab;
            }
            if one_minus_w_sq > 1e-14 {
                let sin_phi_w = one_minus_w_sq.sqrt();
                new_u = mu_lab * dir_x
                    + sin_theta * (dir_x * dir_z * cos_phi - dir_y * sin_phi) / sin_phi_w;
                new_v = mu_lab * dir_y
                    + sin_theta * (dir_y * dir_z * cos_phi + dir_x * sin_phi) / sin_phi_w;
                new_w = mu_lab * dir_z - sin_theta * sin_phi_w * cos_phi;
            }

            energy = new_energy;
            dir_x = new_u;
            dir_y = new_v;
            dir_z = new_w;
        }

        n_steps = step + 1u32;
        step += 1u32;
    }

    out_energies[ABSOLUTE_POS] = energy;
    out_directions[i3] = dir_x;
    out_directions[i3 + 1] = dir_y;
    out_directions[i3 + 2] = dir_z;
    out_alive[ABSOLUTE_POS] = alive;
    out_n_steps[ABSOLUTE_POS] = n_steps;
}

/// Result of multi-step transport per particle.
pub struct MultiStepResult {
    pub final_energies: Vec<f64>,
    pub final_directions: Vec<f64>,
    pub alive: Vec<u32>,
    pub n_steps: Vec<u32>,
    pub absorption_per_bin: Vec<f64>,
}

/// Run the multi-step transport kernel.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_step(
    ctx: &GpuContext,
    seeds: &[u32],
    energies_in: &[f64],
    directions_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
    tally_log_e_min: f64,
    tally_log_e_max: f64,
    n_bins: usize,
    max_steps: u32,
) -> MultiStepResult {
    let n = seeds.len();
    assert_eq!(energies_in.len(), n);
    assert_eq!(directions_in.len(), 3 * n);

    let client = ctx.client();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_h = client.create_from_slice(bytemuck::cast_slice(energies_in));
    let dirs_h = client.create_from_slice(bytemuck::cast_slice(directions_in));
    let log_grid_h = client.create_from_slice(bytemuck::cast_slice(log_energy_grid));
    let xs_e_h = client.create_from_slice(bytemuck::cast_slice(xs_elastic_grid));
    let xs_a_h = client.create_from_slice(bytemuck::cast_slice(xs_absorption_grid));
    let params = [target_mass, tally_log_e_min, tally_log_e_max];
    let params_h = client.create_from_slice(bytemuck::cast_slice(&params));
    let out_e_h = client.empty(std::mem::size_of_val(energies_in));
    let out_d_h = client.empty(std::mem::size_of_val(directions_in));
    let out_alive_h = client.empty(std::mem::size_of_val(seeds));
    let out_n_steps_h = client.empty(std::mem::size_of_val(seeds));
    let initial_tally: Vec<u64> = vec![0u64; n_bins];
    let tally_h = client.create_from_slice(bytemuck::cast_slice(&initial_tally));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        multi_step_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(energies_h, n),
            BufferArg::from_raw_parts(dirs_h, 3 * n),
            BufferArg::from_raw_parts(log_grid_h, log_energy_grid.len()),
            BufferArg::from_raw_parts(xs_e_h, xs_elastic_grid.len()),
            BufferArg::from_raw_parts(xs_a_h, xs_absorption_grid.len()),
            BufferArg::from_raw_parts(params_h, 3),
            BufferArg::from_raw_parts(out_e_h.clone(), n),
            BufferArg::from_raw_parts(out_d_h.clone(), 3 * n),
            BufferArg::from_raw_parts(out_alive_h.clone(), n),
            BufferArg::from_raw_parts(out_n_steps_h.clone(), n),
            BufferArg::from_raw_parts(tally_h.clone(), n_bins),
            max_steps,
        );
    }

    let final_energies: Vec<f64> =
        bytemuck::cast_slice(&client.read_one(out_e_h).unwrap()).to_vec();
    let final_directions: Vec<f64> =
        bytemuck::cast_slice(&client.read_one(out_d_h).unwrap()).to_vec();
    let alive: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_alive_h).unwrap()).to_vec();
    let n_steps: Vec<u32> = bytemuck::cast_slice(&client.read_one(out_n_steps_h).unwrap()).to_vec();
    let tally_bits: Vec<u64> = bytemuck::cast_slice(&client.read_one(tally_h).unwrap()).to_vec();
    let absorption_per_bin: Vec<f64> = tally_bits
        .iter()
        .map(|&bits| (bits as i64 as f64) / 1_073_741_824.0)
        .collect();

    MultiStepResult {
        final_energies,
        final_directions,
        alive,
        n_steps,
        absorption_per_bin,
    }
}

/// CPU equivalent of `run_multi_step` for benchmarking and testing.
/// Same PCG-32, same XS lookup, same Marsaglia rejection, same
/// fixed-point u64 tally. Sequential per-particle.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_step_cpu(
    seeds: &[u32],
    energies_in: &[f64],
    directions_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
    tally_log_e_min: f64,
    tally_log_e_max: f64,
    n_bins: usize,
    max_steps: u32,
) -> MultiStepResult {
    let n = seeds.len();
    assert_eq!(energies_in.len(), n);
    assert_eq!(directions_in.len(), 3 * n);
    let mut tally: Vec<u64> = vec![0u64; n_bins];
    let mut final_energies = Vec::with_capacity(n);
    let mut final_directions = vec![0.0_f64; 3 * n];
    let mut alive_out = Vec::with_capacity(n);
    let mut n_steps_out = Vec::with_capacity(n);

    for i in 0..n {
        let i3 = i * 3;
        let mut energy = energies_in[i];
        let mut dx = directions_in[i3];
        let mut dy = directions_in[i3 + 1];
        let mut dz = directions_in[i3 + 2];
        let mut state = crate::common::rng::expand_seed(seeds[i]);
        let mut alive = 1u32;
        let mut n_steps = 0u32;

        for step in 0..max_steps {
            if alive == 0 {
                break;
            }
            // XS lookup.
            let log_e = energy.ln();
            let lo = log_energy_grid.partition_point(|&x| x < log_e);
            let idx_hi = lo.clamp(1, log_energy_grid.len() - 1);
            let idx_lo = idx_hi - 1;
            let frac = (log_e - log_energy_grid[idx_lo])
                / (log_energy_grid[idx_hi] - log_energy_grid[idx_lo]);
            let sigma_e = xs_elastic_grid[idx_lo]
                + (xs_elastic_grid[idx_hi] - xs_elastic_grid[idx_lo]) * frac;
            let sigma_a = xs_absorption_grid[idx_lo]
                + (xs_absorption_grid[idx_hi] - xs_absorption_grid[idx_lo]) * frac;
            let sigma_t = sigma_e + sigma_a;

            // Free-flight sample.
            let s_xi1 = state;
            let r_xi1 = pcg_out(s_xi1);
            state = s_xi1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let xi1 = (r_xi1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let distance = -xi1.ln() / sigma_t;

            // Tally.
            let bin_f =
                (log_e - tally_log_e_min) / (tally_log_e_max - tally_log_e_min) * n_bins as f64;
            let mut bin = 0i64;
            if bin_f >= 0.0 {
                bin = bin_f as i64;
            }
            if bin >= n_bins as i64 {
                bin = n_bins as i64 - 1;
            }
            let scaled = (distance * sigma_a * 1_073_741_824.0 + 0.5) as i64;
            tally[bin as usize] = tally[bin as usize].wrapping_add(scaled as u64);

            // MT sample.
            let s_xi2 = state;
            let r_xi2 = pcg_out(s_xi2);
            state = s_xi2.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
            let xi2 = (r_xi2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
            let p_elastic = sigma_e / sigma_t;

            if xi2 >= p_elastic {
                alive = 0;
            } else {
                // Scatter: sample µ_cm.
                let s_xi3 = state;
                let r_xi3 = pcg_out(s_xi3);
                state = s_xi3.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                let xi3 = (r_xi3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                let mu_cm = 1.0 - 2.0 * xi3;

                let one_plus_a = target_mass + 1.0;
                let denom = one_plus_a * one_plus_a;
                let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
                let new_energy = energy * numer / denom;
                let mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();

                // Marsaglia rejection.
                let mut accepted = false;
                let mut cos_phi = 1.0;
                let mut sin_phi = 0.0;
                for _ in 0..8 {
                    if accepted {
                        break;
                    }
                    let s_a = state;
                    let r_a = pcg_out(s_a);
                    state = s_a.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                    let s_b = state;
                    let r_b = pcg_out(s_b);
                    state = s_b.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);

                    let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                    let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                    let s = a * a + b * b;
                    if s > 0.0 && s <= 1.0 {
                        let inv_sqrt_s = 1.0 / s.sqrt();
                        cos_phi = a * inv_sqrt_s;
                        sin_phi = b * inv_sqrt_s;
                        accepted = true;
                    }
                }

                // 3D rotation.
                let sin_theta = (1.0 - mu_lab * mu_lab).max(0.0).sqrt();
                let one_minus_w_sq = 1.0 - dz * dz;
                let (new_u, new_v, new_w);
                if one_minus_w_sq > 1e-14 {
                    let sin_phi_w = one_minus_w_sq.sqrt();
                    new_u =
                        mu_lab * dx + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / sin_phi_w;
                    new_v =
                        mu_lab * dy + sin_theta * (dy * dz * cos_phi + dx * sin_phi) / sin_phi_w;
                    new_w = mu_lab * dz - sin_theta * sin_phi_w * cos_phi;
                } else {
                    let sgn = if dz >= 0.0 { 1.0 } else { -1.0 };
                    new_u = sin_theta * cos_phi;
                    new_v = sgn * sin_theta * sin_phi;
                    new_w = sgn * mu_lab;
                }

                energy = new_energy;
                dx = new_u;
                dy = new_v;
                dz = new_w;
            }

            n_steps = step + 1;
        }

        final_energies.push(energy);
        final_directions[i3] = dx;
        final_directions[i3 + 1] = dy;
        final_directions[i3 + 2] = dz;
        alive_out.push(alive);
        n_steps_out.push(n_steps);
    }

    let absorption_per_bin: Vec<f64> = tally
        .iter()
        .map(|&bits| (bits as i64 as f64) / 1_073_741_824.0)
        .collect();

    MultiStepResult {
        final_energies,
        final_directions,
        alive: alive_out,
        n_steps: n_steps_out,
        absorption_per_bin,
    }
}

/// Rayon-parallel CPU equivalent. Each particle's history is fully
/// independent so the parallelisation is trivial; the only catch is
/// per-thread tally accumulation followed by a reduction step.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_step_cpu_rayon(
    seeds: &[u32],
    energies_in: &[f64],
    directions_in: &[f64],
    log_energy_grid: &[f64],
    xs_elastic_grid: &[f64],
    xs_absorption_grid: &[f64],
    target_mass: f64,
    tally_log_e_min: f64,
    tally_log_e_max: f64,
    n_bins: usize,
    max_steps: u32,
) -> MultiStepResult {
    use rayon::prelude::*;

    /// Per-particle final state, returned from a rayon worker.
    struct ParticleResult {
        energy: f64,
        direction: [f64; 3],
        alive: u32,
        n_steps: u32,
        tally: Vec<u64>,
    }

    let n = seeds.len();
    // Per-particle final state computed in parallel.
    let per_particle: Vec<ParticleResult> = (0..n)
        .into_par_iter()
        .map(|i| {
            let i3 = i * 3;
            let mut energy = energies_in[i];
            let mut dx = directions_in[i3];
            let mut dy = directions_in[i3 + 1];
            let mut dz = directions_in[i3 + 2];
            let mut state = crate::common::rng::expand_seed(seeds[i]);
            let mut alive = 1u32;
            let mut n_steps = 0u32;
            let mut tally: Vec<u64> = vec![0u64; n_bins];

            for step in 0..max_steps {
                if alive == 0 {
                    break;
                }
                let log_e = energy.ln();
                let lo = log_energy_grid.partition_point(|&x| x < log_e);
                let idx_hi = lo.clamp(1, log_energy_grid.len() - 1);
                let idx_lo = idx_hi - 1;
                let frac = (log_e - log_energy_grid[idx_lo])
                    / (log_energy_grid[idx_hi] - log_energy_grid[idx_lo]);
                let sigma_e = xs_elastic_grid[idx_lo]
                    + (xs_elastic_grid[idx_hi] - xs_elastic_grid[idx_lo]) * frac;
                let sigma_a = xs_absorption_grid[idx_lo]
                    + (xs_absorption_grid[idx_hi] - xs_absorption_grid[idx_lo]) * frac;
                let sigma_t = sigma_e + sigma_a;

                let s_xi1 = state;
                let r_xi1 = pcg_out(s_xi1);
                state = s_xi1.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                let xi1 = (r_xi1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                let distance = -xi1.ln() / sigma_t;

                let bin_f =
                    (log_e - tally_log_e_min) / (tally_log_e_max - tally_log_e_min) * n_bins as f64;
                let mut bin = 0i64;
                if bin_f >= 0.0 {
                    bin = bin_f as i64;
                }
                if bin >= n_bins as i64 {
                    bin = n_bins as i64 - 1;
                }
                let scaled = (distance * sigma_a * 1_073_741_824.0 + 0.5) as i64;
                tally[bin as usize] = tally[bin as usize].wrapping_add(scaled as u64);

                let s_xi2 = state;
                let r_xi2 = pcg_out(s_xi2);
                state = s_xi2.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                let xi2 = (r_xi2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                let p_elastic = sigma_e / sigma_t;

                if xi2 >= p_elastic {
                    alive = 0;
                } else {
                    let s_xi3 = state;
                    let r_xi3 = pcg_out(s_xi3);
                    state = s_xi3.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                    let xi3 = (r_xi3 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
                    let mu_cm = 1.0 - 2.0 * xi3;

                    let one_plus_a = target_mass + 1.0;
                    let denom = one_plus_a * one_plus_a;
                    let numer = target_mass * target_mass + 2.0 * target_mass * mu_cm + 1.0;
                    let new_energy = energy * numer / denom;
                    let mu_lab = (1.0 + target_mass * mu_cm) / numer.sqrt();

                    let mut accepted = false;
                    let mut cos_phi = 1.0;
                    let mut sin_phi = 0.0;
                    for _ in 0..8 {
                        if accepted {
                            break;
                        }
                        let s_a = state;
                        let r_a = pcg_out(s_a);
                        state = s_a.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
                        let s_b = state;
                        let r_b = pcg_out(s_b);
                        state = s_b.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);

                        let a = (r_a as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                        let b = (r_b as f64 + 0.5) * (2.0 / 4_294_967_296.0) - 1.0;
                        let s = a * a + b * b;
                        if s > 0.0 && s <= 1.0 {
                            let inv_sqrt_s = 1.0 / s.sqrt();
                            cos_phi = a * inv_sqrt_s;
                            sin_phi = b * inv_sqrt_s;
                            accepted = true;
                        }
                    }

                    let sin_theta = (1.0 - mu_lab * mu_lab).max(0.0).sqrt();
                    let one_minus_w_sq = 1.0 - dz * dz;
                    let (new_u, new_v, new_w);
                    if one_minus_w_sq > 1e-14 {
                        let sin_phi_w = one_minus_w_sq.sqrt();
                        new_u = mu_lab * dx
                            + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / sin_phi_w;
                        new_v = mu_lab * dy
                            + sin_theta * (dy * dz * cos_phi + dx * sin_phi) / sin_phi_w;
                        new_w = mu_lab * dz - sin_theta * sin_phi_w * cos_phi;
                    } else {
                        let sgn = if dz >= 0.0 { 1.0 } else { -1.0 };
                        new_u = sin_theta * cos_phi;
                        new_v = sgn * sin_theta * sin_phi;
                        new_w = sgn * mu_lab;
                    }
                    energy = new_energy;
                    dx = new_u;
                    dy = new_v;
                    dz = new_w;
                }

                n_steps = step + 1;
            }

            ParticleResult {
                energy,
                direction: [dx, dy, dz],
                alive,
                n_steps,
                tally,
            }
        })
        .collect();

    let mut final_energies = Vec::with_capacity(n);
    let mut final_directions = vec![0.0_f64; 3 * n];
    let mut alive_out = Vec::with_capacity(n);
    let mut n_steps_out = Vec::with_capacity(n);
    let mut total_tally: Vec<u64> = vec![0u64; n_bins];
    for (i, p) in per_particle.into_iter().enumerate() {
        final_energies.push(p.energy);
        let i3 = i * 3;
        final_directions[i3] = p.direction[0];
        final_directions[i3 + 1] = p.direction[1];
        final_directions[i3 + 2] = p.direction[2];
        alive_out.push(p.alive);
        n_steps_out.push(p.n_steps);
        for (slot, contrib) in total_tally.iter_mut().zip(p.tally.iter()) {
            *slot = slot.wrapping_add(*contrib);
        }
    }
    let absorption_per_bin: Vec<f64> = total_tally
        .iter()
        .map(|&bits| (bits as i64 as f64) / 1_073_741_824.0)
        .collect();

    MultiStepResult {
        final_energies,
        final_directions,
        alive: alive_out,
        n_steps: n_steps_out,
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

    /// With constant Σ_a = 1, Σ_t = 3, absorption probability per
    /// collision is `Σ_a / Σ_t = 1/3`. Histories are geometric: the
    /// expected number of collisions per particle is `Σ_t / Σ_a = 3`,
    /// and the expected total absorption tally per particle equals
    /// 1.0 (each particle absorbs exactly once eventually, scoring
    /// `(track_length × Σ_a)` summed over its history → contributes
    /// `Σ_a / Σ_a = 1` in expectation).
    ///
    /// With max_steps high enough that almost no particles hit the
    /// cap, the tally total per particle should be close to 1.0.
    #[test]
    fn multi_step_aggregate_statistics() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let n = 100_000usize;
        let max_steps: u32 = 50;
        let target_mass = 12.0_f64;
        let n_bins = 8usize;
        let (log_grid, xs_e, xs_a) = constant_xs_grid();
        let log_e_min = (1e-3_f64).ln();
        let log_e_max = (1e3_f64).ln();

        let seeds: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let energies_in: Vec<f64> = vec![1.0; n];
        let mut dirs_in = Vec::with_capacity(3 * n);
        for _ in 0..n {
            dirs_in.extend_from_slice(&[0.0, 0.0, 1.0]);
        }

        let r = run_multi_step(
            &ctx,
            &seeds,
            &energies_in,
            &dirs_in,
            &log_grid,
            &xs_e,
            &xs_a,
            target_mass,
            log_e_min,
            log_e_max,
            n_bins,
            max_steps,
        );

        let mean_steps: f64 = r.n_steps.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let alive_at_end = r.alive.iter().filter(|&&a| a == 1).count();
        let total_tally: f64 = r.absorption_per_bin.iter().sum();
        let mean_tally_per_particle = total_tally / n as f64;

        // Expected: geometric with absorption rate 1/3 per step.
        // Mean steps before absorption = 3.
        // Tail at max_steps = (2/3)^50 ≈ 1.6e-9 -- vanishing.
        let expected_steps = 3.0_f64;
        let expected_tally = 1.0_f64;
        println!(
            "multi-step: mean steps = {mean_steps} (expected ~{expected_steps}), \
             alive at end = {alive_at_end} of {n}, \
             mean tally/particle = {mean_tally_per_particle} (expected ~{expected_tally})"
        );
        // Tolerance: 5% relative for tally (MC noise-dominated),
        // 10% for mean steps (more variance).
        let rel_tally = (mean_tally_per_particle - expected_tally).abs() / expected_tally;
        let rel_steps = (mean_steps - expected_steps).abs() / expected_steps;
        assert!(
            rel_tally < 0.05,
            "tally per particle {mean_tally_per_particle} far from 1.0 (rel err {rel_tally})"
        );
        assert!(
            rel_steps < 0.10,
            "mean steps {mean_steps} far from 3 (rel err {rel_steps})"
        );
        // With max_steps=50 and absorption rate 1/3, alive fraction
        // should be vanishingly small.
        assert!(
            alive_at_end < n / 1000,
            "{alive_at_end} particles alive at step cap suggests step cap was too low or absorption broken"
        );
    }
}
