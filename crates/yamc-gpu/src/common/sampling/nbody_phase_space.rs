//! Shared N-body phase-space (ENDF File-6 Law-6) outgoing-energy + angle sampler.
//!
//! For inelastic reactions that emit 3, 4, or 5 outgoing bodies (e.g. `(n, 2n)`
//! with the residual nucleus as a third body), the outgoing CM-frame energy is
//! drawn from the n-body phase-space distribution
//! `E_out = E_max * x / (x + y)`, where `x` is a Maxwell sample, `y` is a
//! per-`n_bodies` analytic draw, and `E_max` is the kinematically-available
//! band `(A_p - 1)/A_p * (m_t/(m_t+1) * E_in + Q)`. `mu` is isotropic in the CM
//! frame (NBPS provides no angular table), so it overrides any slice-B angular
//! sample. This is the kernel's `eout_kind == EOUT_KIND_NBODY_PHASE_SPACE (5)`
//! branch, faithful to `yamc_physics::gpu::flat::nbody_phase_space`.
//!
//! RNG draw order (preserved exactly from the inline kernel):
//!   1. Maxwell `x` sample: 3 uniforms (the third feeds `cos(pi/2 * xr3)`),
//!   2. `y` sample: 3 uniforms (n=3, Maxwell-shaped), 3 (n=4), or 6 (n=5),
//!   3. isotropic `mu`: 1 uniform.
//!
//! The draws happen ONLY when `n_bodies in [3, 5]`, `total_mass > 1`, and the
//! kinematic cap `e_max > 0`; otherwise no uniforms are consumed and the caller
//! leaves its current `e_cm` / `mu_sampled` untouched (`e_valid == 0`,
//! `mu_valid == 0`). The mu draw happens only when the energy denominator
//! `x + y > 0` (`mu_valid == 1`), matching the flat twin
//! `yamc_physics::gpu::flat::nbody_phase_space`, which returns `None` (no mu
//! draw) for the degenerate `x + y <= 0` case (issue #107). The energy is
//! applied only when `x + y > 0` and the result is `> 0` (`e_valid == 1`).
//! With the PCG's `(0, 1)` uniforms `x_m` is a strictly positive sum of
//! `-ln(xi)` terms, so `x + y > 0` always holds in practice; the guard exists
//! purely to keep the kernel and the flat twin bit-identical.
//!
//! `x_m` / `y_m` flow through the `ln`/`cos` polyfills (~16 ULP for ln, ~1e-13
//! abs for cos per `common/polyfills.rs`), so GPU vs CPU is NOT bit-exact for
//! `e_out`; the parity test uses ~1e-9 relative for `e_out` and ~1e-9 absolute
//! for `mu`, while the PCG `state` and the `e_valid` / `mu_valid` flags stay
//! bit-for-bit (`gpu_nbody_phase_space_matches_cpu`).

use crate::common::polyfills::{cos_f64, ln_f64};
use cubecl::prelude::*;

/// Result of [`sample_nbody_phase_space`]: the sampled outgoing energy (apply
/// only when `e_valid == 1`), the isotropic scattering cosine `mu` (apply only
/// when `mu_valid == 1`), and the advanced PCG-32 state.
///
/// `mu_valid == 1` iff the three input guards passed, `e_max > 0`, and the
/// energy denominator `x_m + y_m > 0` (the flat twin draws / overwrites
/// `mu_sampled` only then). `e_valid == 1` iff additionally the scaled energy
/// came out `> 0`.
#[derive(CubeType)]
pub struct NbodyPhaseSpaceSample {
    pub e_out: f64,
    pub mu: f64,
    pub e_valid: u32,
    pub mu_valid: u32,
    pub state: u64,
}

/// Sample an N-body phase-space (File-6 Law-6) outgoing energy and isotropic
/// `mu`. `n_bodies` is the slot's body count (valid 3, 4, or 5), `total_mass`
/// the total atomic-mass ratio of all outgoing bodies (`A_p`, must be `> 1`),
/// `target_mass` the target atomic-mass ratio, `energy` the incident energy,
/// and `q` the reaction Q-value. Returns the sampled energy + mu, their `valid`
/// flags, and the advanced `state`. Preserves the inline kernel's exact RNG
/// draw order (Maxwell `x`: 3 uniforms; `y`: 3/3/6 for n=3/4/5; `mu`: 1).
#[cube]
pub fn sample_nbody_phase_space(
    energy: f64,
    target_mass: f64,
    q: f64,
    n_bodies: u32,
    total_mass: f64,
    state_in: u64,
) -> NbodyPhaseSpaceSample {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu = 0.0_f64;
    let mut e_valid = 0u32;
    let mut mu_valid = 0u32;

    #[allow(clippy::manual_range_contains)]
    if n_bodies >= 3u32 && n_bodies <= 5u32 && total_mass > 1.0 {
        let e_max =
            (total_mass - 1.0) / total_mass * (target_mass / (target_mass + 1.0) * energy + q);
        if e_max > 0.0 {
            // Maxwell sample for x: 3 draws.
            let d_x1 = crate::common::pcg32::draw_uniform(state);
            state = d_x1.state;
            let xr1 = d_x1.xi;
            let d_x2 = crate::common::pcg32::draw_uniform(state);
            state = d_x2.state;
            let xr2 = d_x2.xi;
            let d_x3 = crate::common::pcg32::draw_uniform(state);
            state = d_x3.state;
            let xr3 = d_x3.xi;
            let cos_xr3 = cos_f64(std::f64::consts::FRAC_PI_2 * xr3);
            let x_m = -ln_f64(xr1) - ln_f64(xr2) * cos_xr3 * cos_xr3;

            #[allow(unused_assignments)]
            let mut y_m = 0.0_f64;
            if n_bodies == 3u32 {
                // Maxwell sample again.
                let d_y1 = crate::common::pcg32::draw_uniform(state);
                state = d_y1.state;
                let yr1 = d_y1.xi;
                let d_y2 = crate::common::pcg32::draw_uniform(state);
                state = d_y2.state;
                let yr2 = d_y2.xi;
                let d_y3 = crate::common::pcg32::draw_uniform(state);
                state = d_y3.state;
                let yr3 = d_y3.xi;
                let cos_yr3 = cos_f64(std::f64::consts::FRAC_PI_2 * yr3);
                y_m = -ln_f64(yr1) - ln_f64(yr2) * cos_yr3 * cos_yr3;
            } else if n_bodies == 4u32 {
                // y = -ln(r1*r2*r3)
                let d_y1 = crate::common::pcg32::draw_uniform(state);
                state = d_y1.state;
                let yr1 = d_y1.xi;
                let d_y2 = crate::common::pcg32::draw_uniform(state);
                state = d_y2.state;
                let yr2 = d_y2.xi;
                let d_y3 = crate::common::pcg32::draw_uniform(state);
                state = d_y3.state;
                let yr3 = d_y3.xi;
                y_m = -ln_f64(yr1 * yr2 * yr3);
            } else {
                // n_bodies == 5
                // y = -ln(r1*r2*r3*r4) - ln(r5)·cos(π/2·r6)²
                let d_y1 = crate::common::pcg32::draw_uniform(state);
                state = d_y1.state;
                let yr1 = d_y1.xi;
                let d_y2 = crate::common::pcg32::draw_uniform(state);
                state = d_y2.state;
                let yr2 = d_y2.xi;
                let d_y3 = crate::common::pcg32::draw_uniform(state);
                state = d_y3.state;
                let yr3 = d_y3.xi;
                let d_y4 = crate::common::pcg32::draw_uniform(state);
                state = d_y4.state;
                let yr4 = d_y4.xi;
                let d_y5 = crate::common::pcg32::draw_uniform(state);
                state = d_y5.state;
                let yr5 = d_y5.xi;
                let d_y6 = crate::common::pcg32::draw_uniform(state);
                state = d_y6.state;
                let yr6 = d_y6.xi;
                let cos_yr6 = cos_f64(std::f64::consts::FRAC_PI_2 * yr6);
                y_m = -ln_f64(yr1 * yr2 * yr3 * yr4) - ln_f64(yr5) * cos_yr6 * cos_yr6;
            }

            let denom_v = x_m + y_m;
            if denom_v > 0.0 {
                let v_frac = x_m / denom_v;
                let e_sampled = e_max * v_frac;
                if e_sampled > 0.0 {
                    e_out = e_sampled;
                    e_valid = 1u32;
                }

                // Mu is isotropic for NBPS: override slice-B sample. Drawn
                // ONLY when the energy denominator `x + y > 0`, matching the
                // flat twin `yamc_physics::gpu::flat::nbody_phase_space`,
                // which returns `None` (no mu draw) for the degenerate
                // `x + y <= 0` case (issue #107). With the PCG's `(0, 1)`
                // uniforms `x_m` is a strictly positive sum of `-ln(xi)`
                // terms, so this branch is always taken in practice; the
                // guard keeps the kernel and twin bit-identical regardless.
                let d_mu = crate::common::pcg32::draw_uniform(state);
                state = d_mu.state;
                let xi_nb_mu = d_mu.xi;
                mu = 1.0 - 2.0 * xi_nb_mu;
                mu_valid = 1u32;
            }
        }
    }

    NbodyPhaseSpaceSample {
        e_out,
        mu,
        e_valid,
        mu_valid,
        state,
    }
}

// ----------------------------- CPU twin -----------------------------

use crate::common::pcg32::draw_uniform_cpu;

/// CPU twin of [`sample_nbody_phase_space`]. Same algorithm and same 64-bit-PCG
/// draws (via [`draw_uniform_cpu`]); uses libm `ln`/`cos` rather than the
/// polyfills, so it matches the `#[cube]` kernel to within the polyfills' ULP
/// drift. Returns `(e_out, mu, e_valid, mu_valid, state)`.
pub fn sample_nbody_phase_space_cpu(
    energy: f64,
    target_mass: f64,
    q: f64,
    n_bodies: u32,
    total_mass: f64,
    state_in: u64,
) -> (f64, f64, u32, u32, u64) {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu = 0.0_f64;
    let mut e_valid = 0u32;
    let mut mu_valid = 0u32;

    if (3..=5).contains(&n_bodies) && total_mass > 1.0 {
        let e_max =
            (total_mass - 1.0) / total_mass * (target_mass / (target_mass + 1.0) * energy + q);
        if e_max > 0.0 {
            let (xr1, s) = draw_uniform_cpu(state);
            state = s;
            let (xr2, s) = draw_uniform_cpu(state);
            state = s;
            let (xr3, s) = draw_uniform_cpu(state);
            state = s;
            let cos_xr3 = (std::f64::consts::FRAC_PI_2 * xr3).cos();
            let x_m = -xr1.ln() - xr2.ln() * cos_xr3 * cos_xr3;

            let y_m = if n_bodies == 3 {
                let (yr1, s) = draw_uniform_cpu(state);
                state = s;
                let (yr2, s) = draw_uniform_cpu(state);
                state = s;
                let (yr3, s) = draw_uniform_cpu(state);
                state = s;
                let cos_yr3 = (std::f64::consts::FRAC_PI_2 * yr3).cos();
                -yr1.ln() - yr2.ln() * cos_yr3 * cos_yr3
            } else if n_bodies == 4 {
                let (yr1, s) = draw_uniform_cpu(state);
                state = s;
                let (yr2, s) = draw_uniform_cpu(state);
                state = s;
                let (yr3, s) = draw_uniform_cpu(state);
                state = s;
                -(yr1 * yr2 * yr3).ln()
            } else {
                let (yr1, s) = draw_uniform_cpu(state);
                state = s;
                let (yr2, s) = draw_uniform_cpu(state);
                state = s;
                let (yr3, s) = draw_uniform_cpu(state);
                state = s;
                let (yr4, s) = draw_uniform_cpu(state);
                state = s;
                let (yr5, s) = draw_uniform_cpu(state);
                state = s;
                let (yr6, s) = draw_uniform_cpu(state);
                state = s;
                let cos_yr6 = (std::f64::consts::FRAC_PI_2 * yr6).cos();
                -(yr1 * yr2 * yr3 * yr4).ln() - yr5.ln() * cos_yr6 * cos_yr6
            };

            let denom_v = x_m + y_m;
            if denom_v > 0.0 {
                let e_sampled = e_max * (x_m / denom_v);
                if e_sampled > 0.0 {
                    e_out = e_sampled;
                    e_valid = 1;
                }

                // Mu drawn ONLY when `x + y > 0` (issue #107), matching the
                // `#[cube]` kernel above and the flat twin's early-return.
                let (xi_nb_mu, s) = draw_uniform_cpu(state);
                state = s;
                mu = 1.0 - 2.0 * xi_nb_mu;
                mu_valid = 1;
            }
        }
    }

    (e_out, mu, e_valid, mu_valid, state)
}

// ----------------------------- Test kernel -----------------------------

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn nbody_test_kernel(
    energy: &[f64],
    target_mass: &[f64],
    q: &[f64],
    n_bodies: &[u32],
    total_mass: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_mu: &mut [f64],
    out_e_valid: &mut [u32],
    out_mu_valid: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = sample_nbody_phase_space(
        energy[ABSOLUTE_POS],
        target_mass[ABSOLUTE_POS],
        q[ABSOLUTE_POS],
        n_bodies[ABSOLUTE_POS],
        total_mass[ABSOLUTE_POS],
        crate::common::pcg32::expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_mu[ABSOLUTE_POS] = r.mu;
    out_e_valid[ABSOLUTE_POS] = r.e_valid;
    out_mu_valid[ABSOLUTE_POS] = r.mu_valid;
    out_state[ABSOLUTE_POS] = r.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    // Rel tolerance for the energy: x_m / y_m flow through the ln/cos
    // polyfills (~16 ULP for ln, ~1e-13 abs for cos per `polyfills.rs`), so
    // 1e-9 relative is comfortably above the worst-case compounded drift while
    // still being thousands of times tighter than MC noise. `mu` is a pure
    // u32 -> f64 draw (no transcendentals) so it is bit-exact, but we compare
    // it within the same absolute tolerance for uniformity. The PCG `state`
    // and the `e_valid` / `mu_valid` flags are bit-exact.
    const REL_TOL: f64 = 1e-9;

    fn seeds_vec(n: u32, salt: u32) -> Vec<u32> {
        (0..n)
            .map(|s| (s.wrapping_mul(7919).wrapping_add(salt)).wrapping_mul(2_654_435_761))
            .collect()
    }

    fn ctx_or_skip() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                None
            }
        }
    }

    /// GPU N-body phase-space draw must match the CPU twin: PCG `state` and the
    /// `e_valid` / `mu_valid` flags bit-exact; the energy within `REL_TOL`
    /// (two/four `ln` polyfill calls + one/two `cos`), `mu` within the same
    /// absolute tolerance. Sweeps n=3/4/5, several `(E_in, target_mass, Q,
    /// A_p)` shapes (incl. the invalid-body-count, degenerate-mass, and
    /// below-threshold no-draw cases), and many seeds.
    #[test]
    fn gpu_nbody_phase_space_matches_cpu() {
        let Some(ctx) = ctx_or_skip() else { return };
        // (E_in, target_mass, Q, total_mass, n_bodies)
        let shapes: &[(f64, f64, f64, f64, u32)] = &[
            (14.06e6, 56.0, -5.0e6, 57.0, 3),
            (14.06e6, 56.0, -5.0e6, 57.0, 4),
            (14.06e6, 56.0, -5.0e6, 57.0, 5),
            (20.0e6, 9.0, -1.6e6, 10.0, 3),
            (18.0e6, 12.0, -7.0e6, 13.0, 4),
            (14.1e6, 1.0, -2.2e6, 2.0, 5),
            // No-draw cases: invalid body count, degenerate mass, sub-threshold.
            (14.06e6, 56.0, -5.0e6, 57.0, 2),
            (14.06e6, 56.0, -5.0e6, 57.0, 6),
            (14.06e6, 56.0, -5.0e6, 1.0, 3),
            (1.0e6, 56.0, -100.0e6, 57.0, 3),
        ];
        let mut energy = Vec::new();
        let mut target_mass = Vec::new();
        let mut q = Vec::new();
        let mut total_mass = Vec::new();
        let mut n_bodies = Vec::new();
        let mut seeds = Vec::new();
        for (si, &(e, tm, qv, ap, nb)) in shapes.iter().enumerate() {
            for s in seeds_vec(400, si as u32) {
                energy.push(e);
                target_mass.push(tm);
                q.push(qv);
                total_mass.push(ap);
                n_bodies.push(nb);
                seeds.push(s);
            }
        }
        let n = seeds.len();

        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_mu = vec![0.0_f64; n];
        let mut cpu_ev = vec![0u32; n];
        let mut cpu_mv = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, mu, ev, mv, st) = sample_nbody_phase_space_cpu(
                energy[i],
                target_mass[i],
                q[i],
                n_bodies[i],
                total_mass[i],
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_mu[i] = mu;
            cpu_ev[i] = ev;
            cpu_mv[i] = mv;
            cpu_state[i] = st;
        }

        let client = ctx.client();
        let e_h = client.create_from_slice(bytemuck::cast_slice(&energy));
        let tm_h = client.create_from_slice(bytemuck::cast_slice(&target_mass));
        let q_h = client.create_from_slice(bytemuck::cast_slice(&q));
        let nb_h = client.create_from_slice(bytemuck::cast_slice(&n_bodies));
        let ap_h = client.create_from_slice(bytemuck::cast_slice(&total_mass));
        let s_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(n * core::mem::size_of::<f64>());
        let omu_h = client.empty(n * core::mem::size_of::<f64>());
        let oev_h = client.empty(n * core::mem::size_of::<u32>());
        let omv_h = client.empty(n * core::mem::size_of::<u32>());
        let os_h = client.empty(n * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            nbody_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(e_h, energy.len()),
                BufferArg::from_raw_parts(tm_h, target_mass.len()),
                BufferArg::from_raw_parts(q_h, q.len()),
                BufferArg::from_raw_parts(nb_h, n_bodies.len()),
                BufferArg::from_raw_parts(ap_h, total_mass.len()),
                BufferArg::from_raw_parts(s_h, seeds.len()),
                BufferArg::from_raw_parts(oe_h.clone(), n),
                BufferArg::from_raw_parts(omu_h.clone(), n),
                BufferArg::from_raw_parts(oev_h.clone(), n),
                BufferArg::from_raw_parts(omv_h.clone(), n),
                BufferArg::from_raw_parts(os_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_mu = bytemuck::cast_slice::<u8, f64>(&client.read_one(omu_h).unwrap()).to_vec();
        let gpu_ev = bytemuck::cast_slice::<u8, u32>(&client.read_one(oev_h).unwrap()).to_vec();
        let gpu_mv = bytemuck::cast_slice::<u8, u32>(&client.read_one(omv_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "sample {i}: PCG state diverged (gpu {} cpu {}, n_bodies {}, seed {})",
                gpu_state[i], cpu_state[i], n_bodies[i], seeds[i]
            );
            assert_eq!(
                gpu_ev[i], cpu_ev[i],
                "sample {i}: e_valid diverged (gpu {} cpu {}, n_bodies {}, seed {})",
                gpu_ev[i], cpu_ev[i], n_bodies[i], seeds[i]
            );
            assert_eq!(
                gpu_mv[i], cpu_mv[i],
                "sample {i}: mu_valid diverged (gpu {} cpu {}, n_bodies {}, seed {})",
                gpu_mv[i], cpu_mv[i], n_bodies[i], seeds[i]
            );
            if cpu_ev[i] == 1 {
                let tol = REL_TOL * cpu_e[i].abs().max(1.0);
                assert!(
                    (gpu_e[i] - cpu_e[i]).abs() <= tol,
                    "sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, n_bodies {}, seed {})",
                    gpu_e[i],
                    cpu_e[i],
                    n_bodies[i],
                    seeds[i]
                );
            }
            if cpu_mv[i] == 1 {
                assert!(
                    (gpu_mu[i] - cpu_mu[i]).abs() <= REL_TOL,
                    "sample {i}: mu diverged (gpu {} cpu {}, n_bodies {}, seed {})",
                    gpu_mu[i],
                    cpu_mu[i],
                    n_bodies[i],
                    seeds[i]
                );
                assert!(
                    (-1.0..=1.0).contains(&gpu_mu[i]),
                    "sample {i}: mu out of [-1, 1]: {}",
                    gpu_mu[i]
                );
            }
        }
        println!(
            "nbody phase-space: {} draws, state+e_valid+mu_valid bit-exact, energy within {:e} rel",
            n, REL_TOL
        );
    }
}
