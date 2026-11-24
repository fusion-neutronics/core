//! Shared outgoing-energy rejection samplers for the
//! Evaporation (ENDF File 5, Law 9), Maxwell (Law 7), and Watt (Law 11)
//! continuum laws.
//!
//! Each law's E_out is drawn by rejection: a fixed number of PCG-32 uniforms
//! per attempt, fed through the f64 `ln`/`exp`/`cos` polyfills, then tested
//! against a cap. Three layers:
//!
//! - `*_rejection_draw`: ONE attempt (evaporation 2 uniforms, Maxwell 3,
//!   Watt 4), returning the candidate, an `accepted` flag and the state.
//! - `*_rejection`: the cap check and up to 32 attempts at a known `theta`
//!   (or Watt `a`, `b`). The fission chi sampler calls these after its own
//!   interpolation.
//! - `sample_*_law`: `theta(E_in)` (or `a`, `b`) interpolated on the slot's
//!   grid, then the rejection loop. The transport kernel's inelastic
//!   Evaporation / Maxwell / Watt branches call these; they are the GPU form
//!   of `yamc_physics::gpu::flat::{evaporation, maxwell, watt}`, which the CPU
//!   transport uses, and `gpu_rejection_laws_match_flat` holds them to it.
//!
//! The draw order is the same as the `flat` samplers', so the RNG `state`
//! advances bit-for-bit. Because the candidate energy flows through
//! `ln`/`exp`/`cos`, GPU vs CPU is not bit-exact for the energy: the tests
//! use a relative tolerance, while `valid` and the PCG `state` must match
//! exactly. The single-shot fission Watt fallback (`watt_fission_draw`) is
//! checked against a CPU copy.

use crate::common::polyfills::{cos_f64, exp_f64, ln_f64};
use cubecl::prelude::*;

/// Result of a single Evaporation / Maxwell / Watt rejection attempt: the
/// candidate outgoing energy (only meaningful when `accepted == 1`), the
/// acceptance flag, and the advanced PCG-32 state.
#[derive(CubeType)]
pub struct RejectionDraw {
    pub e_out: f64,
    pub accepted: u32,
    pub state: u64,
}

/// One Evaporation rejection attempt (`p(E) proportional to E * exp(-E/theta)`,
/// capped at `theta * y = E_in - U`). Draws two PCG uniforms `xi1, xi2`,
/// forms `x = -ln(1 - v*xi1) - ln(1 - v*xi2)` (where `v = 1 - exp(-y)`), and
/// accepts when `x <= y`; the accepted energy is `x * theta`. `v` and `y` are
/// precomputed at the call site from the interpolated `theta`. Returns
/// `accepted == 0` (with `e_out == 0`) when the attempt is rejected.
#[cube]
pub fn evaporation_rejection_draw(
    v_e: f64,
    y: f64,
    theta_val: f64,
    state_in: u64,
) -> RejectionDraw {
    let mut state = state_in;

    let d_e1 = crate::common::pcg32::draw_uniform(state);
    state = d_e1.state;
    let xi_e1 = d_e1.xi;

    let d_e2 = crate::common::pcg32::draw_uniform(state);
    state = d_e2.state;
    let xi_e2 = d_e2.xi;

    // x = -ln((1 - v*xi1)(1 - v*xi2)) = -ln(1 - v*xi1) - ln(1 - v*xi2)
    let arg_a = 1.0 - v_e * xi_e1;
    let arg_b = 1.0 - v_e * xi_e2;
    let mut x_e = 0.0_f64;
    if arg_a > 0.0 && arg_b > 0.0 {
        x_e = -ln_f64(arg_a) - ln_f64(arg_b);
    }
    let mut accepted = 0u32;
    let mut e_out = 0.0_f64;
    if x_e <= y {
        accepted = 1u32;
        e_out = x_e * theta_val;
    }
    RejectionDraw {
        e_out,
        accepted,
        state,
    }
}

/// One Maxwell rejection attempt (`p(E) proportional to sqrt(E) * exp(-E/theta)`,
/// capped at `cap_e = E_in - U`). Draws three PCG uniforms; the third feeds
/// `cos(pi/2 * xi3)`. Forms `E = -theta * (ln xi1 + ln xi2 * cos^2)` and accepts
/// when `0 < E <= cap_e`. Mirrors `sample_maxwell_spectrum`.
#[cube]
pub fn maxwell_rejection_draw(theta_val: f64, cap_e: f64, state_in: u64) -> RejectionDraw {
    let mut state = state_in;

    let d_m1 = crate::common::pcg32::draw_uniform(state);
    state = d_m1.state;
    let xi_m1 = d_m1.xi;

    let d_m2 = crate::common::pcg32::draw_uniform(state);
    state = d_m2.state;
    let xi_m2 = d_m2.xi;

    let d_m3 = crate::common::pcg32::draw_uniform(state);
    state = d_m3.state;
    let xi_m3 = d_m3.xi;

    let cos_m3 = cos_f64(std::f64::consts::FRAC_PI_2 * xi_m3);
    let mut e_out_m = 0.0_f64;
    if xi_m1 > 0.0 && xi_m2 > 0.0 {
        e_out_m = -theta_val * (ln_f64(xi_m1) + ln_f64(xi_m2) * cos_m3 * cos_m3);
    }
    let mut accepted = 0u32;
    let mut e_out = 0.0_f64;
    if e_out_m > 0.0 && e_out_m <= cap_e {
        accepted = 1u32;
        e_out = e_out_m;
    }
    RejectionDraw {
        e_out,
        accepted,
        state,
    }
}

/// One inelastic Watt rejection attempt (`p(E) proportional to exp(-E/a) *
/// sinh(sqrt(b*E))`, capped at `cap_e = E_in - U`) via the Watt-Maxwell
/// relationship. Draws four PCG uniforms: a Maxwell `w` with parameter `a`
/// (uniforms 1-3, `cos(pi/2 * xi3)`), then the Watt correction
/// `u_xi = 2*xi4 - 1`. Forms `E = w + a2b/4 + u_xi * sqrt(a2b * w)` (with
/// `a2b = a*a*b` precomputed at the call site) and accepts when `0 < E <= cap_e`.
/// Mirrors `sample_watt_spectrum_params`.
#[cube]
pub fn watt_rejection_draw(watt_a_val: f64, a2b: f64, cap_e: f64, state_in: u64) -> RejectionDraw {
    let mut state = state_in;

    let d_w1 = crate::common::pcg32::draw_uniform(state);
    state = d_w1.state;
    let xi_w1 = d_w1.xi;

    let d_w2 = crate::common::pcg32::draw_uniform(state);
    state = d_w2.state;
    let xi_w2 = d_w2.xi;

    let d_w3 = crate::common::pcg32::draw_uniform(state);
    state = d_w3.state;
    let xi_w3 = d_w3.xi;

    let d_w4 = crate::common::pcg32::draw_uniform(state);
    state = d_w4.state;
    let xi_w4 = d_w4.xi;

    // Maxwell w with parameter a: w = -a * (ln xi1 + ln xi2 * cos^2(pi/2 * xi3)).
    let cos_w3 = cos_f64(std::f64::consts::FRAC_PI_2 * xi_w3);
    let mut w_m = 0.0_f64;
    if xi_w1 > 0.0 && xi_w2 > 0.0 {
        w_m = -watt_a_val * (ln_f64(xi_w1) + ln_f64(xi_w2) * cos_w3 * cos_w3);
    }
    // Watt correction: u_xi = 2*xi4 - 1.
    let u_xi = 2.0 * xi_w4 - 1.0;
    let prod = a2b * w_m;
    let mut e_out_w = 0.0_f64;
    if w_m > 0.0 && prod >= 0.0 {
        e_out_w = w_m + 0.25 * a2b + u_xi * prod.sqrt();
    }
    let mut accepted = 0u32;
    let mut e_out = 0.0_f64;
    if e_out_w > 0.0 && e_out_w <= cap_e {
        accepted = 1u32;
        e_out = e_out_w;
    }
    RejectionDraw {
        e_out,
        accepted,
        state,
    }
}

/// Result of [`watt_fission_draw`]: the (un-clamped) outgoing energy and the
/// advanced PCG-32 state. The fission-spectrum Watt fallback is a SINGLE
/// unconditional draw (no rejection loop and no `w > 0` guard), so there is no
/// `accepted` flag; the caller applies the `<= 0 -> 1e-6` floor.
#[derive(CubeType)]
pub struct WattFissionDraw {
    pub e_out: f64,
    pub state: u64,
}

/// Single-shot fission-spectrum Watt draw (the fallback used when a fissile
/// material carries only Watt `(a, b)` parameters and no tabulated chi-spectrum).
/// Draws four PCG uniforms exactly as the inelastic Watt body, but
/// UNCONDITIONALLY (no `w > 0` / `prod >= 0` guard) and with no acceptance test,
/// matching the kernel's inline fission fallback. The caller floors a
/// non-positive result to `1e-6`. Mirrors `sample_watt_spectrum_params` for the
/// single-attempt fission path.
#[cube]
pub fn watt_fission_draw(watt_a: f64, watt_b: f64, state_in: u64) -> WattFissionDraw {
    let mut state = state_in;

    let d_xw1 = crate::common::pcg32::draw_uniform(state);
    state = d_xw1.state;
    let xw1 = d_xw1.xi;

    let d_xw2 = crate::common::pcg32::draw_uniform(state);
    state = d_xw2.state;
    let xw2 = d_xw2.xi;

    let d_xw3 = crate::common::pcg32::draw_uniform(state);
    state = d_xw3.state;
    let xw3 = d_xw3.xi;

    let cos_arg = std::f64::consts::FRAC_PI_2 * xw3;
    let c = cos_f64(cos_arg);
    let w_max = -watt_a * (ln_f64(xw1) + ln_f64(xw2) * c * c);

    let d_xw4 = crate::common::pcg32::draw_uniform(state);
    state = d_xw4.state;
    let xw4 = d_xw4.xi;
    let a2b = watt_a * watt_a * watt_b;
    let u_sym = 2.0 * xw4 - 1.0;
    let e_out = w_max + 0.25 * a2b + u_sym * (a2b * w_max).sqrt();

    WattFissionDraw { e_out, state }
}

// ------------------------- Whole-law samplers -------------------------

/// An outgoing energy sampled by a whole law (interpolation, cap and the
/// rejection loop): `valid == 1` when an attempt was accepted with a positive
/// energy, matching `Some` from the `yamc_physics::gpu::flat` sampler of the
/// same law. When `valid == 0` the caller keeps its current energy.
#[derive(CubeType)]
pub struct LawSample {
    pub e_out: f64,
    pub valid: u32,
    pub state: u64,
}

/// Rejection attempts per law sample, as in the `flat` samplers.
const MAX_ATTEMPTS: u32 = 32;

/// Linearly interpolate a per-energy parameter at `e_in` on the `n`-point
/// grid starting at `grid[g_off]`, clamped to the end values (the `flat`
/// samplers' `linear_interp_on_grid`). Point `k`'s parameter is
/// `param[(p_off + k) * p_stride + p_lane]`, so interleaved parameters (Watt's
/// `a`, `b` at stride 2) read their own lane. Requires `n >= 1`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn interp_on_grid(
    e_in: f64,
    grid: &[f64],
    g_off: u32,
    n: u32,
    param: &[f64],
    p_off: u32,
    p_stride: u32,
    p_lane: u32,
) -> f64 {
    let e_first = grid[g_off as usize];
    let e_last = grid[(g_off + n - 1u32) as usize];
    let mut out = param[((p_off + n - 1u32) * p_stride + p_lane) as usize];
    if e_in <= e_first {
        out = param[(p_off * p_stride + p_lane) as usize];
    } else if e_in < e_last {
        let mut k = 0u32;
        while k + 1u32 < n {
            let e_k = grid[(g_off + k) as usize];
            let e_k1 = grid[(g_off + k + 1u32) as usize];
            if e_in >= e_k && e_in < e_k1 {
                let de = e_k1 - e_k;
                let mut f = e_in * 0.0;
                if de > 0.0 {
                    f = (e_in - e_k) / de;
                }
                let p_k = param[((p_off + k) * p_stride + p_lane) as usize];
                let p_k1 = param[((p_off + k + 1u32) * p_stride + p_lane) as usize];
                out = p_k + f * (p_k1 - p_k);
            }
            k += 1u32;
        }
    }
    out
}

/// Evaporation (Law 9) at a known `theta`: up to [`MAX_ATTEMPTS`]
/// [`evaporation_rejection_draw`]s below the cap `E_in - u`. No draws when
/// `theta <= 0` or `e_in <= u`.
#[cube]
pub fn evaporation_rejection(theta_val: f64, e_in: f64, u: f64, state_in: u64) -> LawSample {
    let mut state = state_in;
    let mut e_out = e_in * 0.0;
    let mut valid = 0u32;
    if theta_val > 0.0 && e_in > u {
        let y = (e_in - u) / theta_val;
        let v_e = 1.0 - exp_f64(-y);
        let mut accepted = 0u32;
        let mut iter = 0u32;
        while iter < MAX_ATTEMPTS && accepted == 0u32 {
            let rj = evaporation_rejection_draw(v_e, y, theta_val, state);
            state = rj.state;
            if rj.accepted == 1u32 {
                accepted = 1u32;
                if rj.e_out > 0.0 {
                    e_out = rj.e_out;
                    valid = 1u32;
                }
            }
            iter += 1u32;
        }
    }
    LawSample {
        e_out,
        valid,
        state,
    }
}

/// Maxwell (Law 7) at a known `theta`: up to [`MAX_ATTEMPTS`]
/// [`maxwell_rejection_draw`]s below the cap `E_in - u`. No draws when
/// `theta <= 0` or `e_in <= u`.
#[cube]
pub fn maxwell_rejection(theta_val: f64, e_in: f64, u: f64, state_in: u64) -> LawSample {
    let mut state = state_in;
    let mut e_out = e_in * 0.0;
    let mut valid = 0u32;
    if theta_val > 0.0 && e_in > u {
        let cap_e = e_in - u;
        let mut accepted = 0u32;
        let mut iter = 0u32;
        while iter < MAX_ATTEMPTS && accepted == 0u32 {
            let rj = maxwell_rejection_draw(theta_val, cap_e, state);
            state = rj.state;
            if rj.accepted == 1u32 {
                accepted = 1u32;
                if rj.e_out > 0.0 {
                    e_out = rj.e_out;
                    valid = 1u32;
                }
            }
            iter += 1u32;
        }
    }
    LawSample {
        e_out,
        valid,
        state,
    }
}

/// Watt (Law 11, inelastic) at known `a`, `b`: up to [`MAX_ATTEMPTS`]
/// [`watt_rejection_draw`]s below the cap `E_in - u`. No draws when `a <= 0`
/// or `e_in <= u`.
#[cube]
pub fn watt_rejection(a_val: f64, b_val: f64, e_in: f64, u: f64, state_in: u64) -> LawSample {
    let mut state = state_in;
    let mut e_out = e_in * 0.0;
    let mut valid = 0u32;
    if a_val > 0.0 && e_in > u {
        let cap_e = e_in - u;
        let a2b = a_val * a_val * b_val;
        let mut accepted = 0u32;
        let mut iter = 0u32;
        while iter < MAX_ATTEMPTS && accepted == 0u32 {
            let rj = watt_rejection_draw(a_val, a2b, cap_e, state);
            state = rj.state;
            if rj.accepted == 1u32 {
                accepted = 1u32;
                if rj.e_out > 0.0 {
                    e_out = rj.e_out;
                    valid = 1u32;
                }
            }
            iter += 1u32;
        }
    }
    LawSample {
        e_out,
        valid,
        state,
    }
}

/// `yamc_physics::gpu::flat::evaporation::sample_evaporation` on the GPU:
/// `theta(E_in)` from the `n`-point grid at `grid[g_off]` and the theta row at
/// `theta[theta_off]`, then [`evaporation_rejection`]. Requires `n >= 1`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn sample_evaporation_law(
    e_in: f64,
    grid: &[f64],
    g_off: u32,
    n: u32,
    theta: &[f64],
    theta_off: u32,
    u: f64,
    state: u64,
) -> LawSample {
    let theta_val = interp_on_grid(e_in, grid, g_off, n, theta, theta_off, 1u32, 0u32);
    evaporation_rejection(theta_val, e_in, u, state)
}

/// `yamc_physics::gpu::flat::maxwell::sample_maxwell` on the GPU: `theta(E_in)`
/// from the `n`-point grid and theta row both at offset `g_off`, then
/// [`maxwell_rejection`]. Requires `n >= 1`.
#[cube]
pub fn sample_maxwell_law(
    e_in: f64,
    grid: &[f64],
    g_off: u32,
    n: u32,
    theta: &[f64],
    u: f64,
    state: u64,
) -> LawSample {
    let theta_val = interp_on_grid(e_in, grid, g_off, n, theta, g_off, 1u32, 0u32);
    maxwell_rejection(theta_val, e_in, u, state)
}

/// `yamc_physics::gpu::flat::watt::sample_watt_inelastic` on the GPU: `a(E_in)`
/// and `b(E_in)` from the `n`-point grid at offset `g_off` and the interleaved
/// `ab` buffer (`a` at `2 * i`, `b` at `2 * i + 1`), then [`watt_rejection`].
/// Requires `n >= 1`.
#[cube]
pub fn sample_watt_law(
    e_in: f64,
    grid: &[f64],
    g_off: u32,
    n: u32,
    ab: &[f64],
    u: f64,
    state: u64,
) -> LawSample {
    let a_val = interp_on_grid(e_in, grid, g_off, n, ab, g_off, 2u32, 0u32);
    let b_val = interp_on_grid(e_in, grid, g_off, n, ab, g_off, 2u32, 1u32);
    watt_rejection(a_val, b_val, e_in, u, state)
}

// ----------------------------- CPU twins -----------------------------

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};

/// CPU `draw_uniform` twin (wrapping 64-bit PCG, `(r + 1)/(2^32 + 1)` mapping).
/// Returns `(xi, state)`. Bit-for-bit identical to the `#[cube]` draw.
#[inline]
fn cpu_draw_uniform(state_in: u64) -> (f64, u64) {
    let s = state_in;
    let rand = pcg_out(s);
    let state = s.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    ((rand as f64 + 1.0) * (1.0 / 4_294_967_297.0), state)
}

/// CPU twin of [`watt_fission_draw`]. Returns `(e_out, state)`.
pub fn watt_fission_draw_cpu(watt_a: f64, watt_b: f64, state_in: u64) -> (f64, u64) {
    let (xw1, state) = cpu_draw_uniform(state_in);
    let (xw2, state) = cpu_draw_uniform(state);
    let (xw3, state) = cpu_draw_uniform(state);
    let c = (std::f64::consts::FRAC_PI_2 * xw3).cos();
    let w_max = -watt_a * (xw1.ln() + xw2.ln() * c * c);
    let (xw4, state) = cpu_draw_uniform(state);
    let a2b = watt_a * watt_a * watt_b;
    let u_sym = 2.0 * xw4 - 1.0;
    let e_out = w_max + 0.25 * a2b + u_sym * (a2b * w_max).sqrt();
    (e_out, state)
}

// ----------------------------- Test kernels -----------------------------

#[cube(launch)]
fn watt_fission_test_kernel(
    watt_a: &[f64],
    watt_b: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = watt_fission_draw(
        watt_a[ABSOLUTE_POS],
        watt_b[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_state[ABSOLUTE_POS] = r.state;
}

/// Law selector for [`law_test_kernel`].
#[cfg(test)]
const LAW_EVAPORATION: u32 = 0;
#[cfg(test)]
const LAW_MAXWELL: u32 = 1;
#[cfg(test)]
const LAW_WATT: u32 = 2;

/// One whole-law sample per thread: [`sample_evaporation_law`],
/// [`sample_maxwell_law`] or [`sample_watt_law`] by `law[i]`, on shared grid
/// buffers with per-sample offsets.
#[cfg(test)]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn law_test_kernel(
    law: &[u32],
    e_in: &[f64],
    u: &[f64],
    g_off: &[u32],
    n: &[u32],
    theta_off: &[u32],
    grid: &[f64],
    theta: &[f64],
    evap_theta: &[f64],
    ab: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_valid: &mut [u32],
    out_state: &mut [u64],
) {
    let i = ABSOLUTE_POS;
    if i >= out_e.len() {
        terminate!();
    }
    let state = expand_seed(seeds[i]);
    if law[i] == LAW_EVAPORATION {
        let r = sample_evaporation_law(
            e_in[i],
            grid,
            g_off[i],
            n[i],
            evap_theta,
            theta_off[i],
            u[i],
            state,
        );
        out_e[i] = r.e_out;
        out_valid[i] = r.valid;
        out_state[i] = r.state;
    } else if law[i] == LAW_MAXWELL {
        let r = sample_maxwell_law(e_in[i], grid, g_off[i], n[i], theta, u[i], state);
        out_e[i] = r.e_out;
        out_valid[i] = r.valid;
        out_state[i] = r.state;
    } else {
        let r = sample_watt_law(e_in[i], grid, g_off[i], n[i], ab, u[i], state);
        out_e[i] = r.e_out;
        out_valid[i] = r.valid;
        out_state[i] = r.state;
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    // Rel tolerance for the energy: the candidate flows through the ln/exp/cos
    // polyfills (~16 ULP for ln/exp, ~1e-13 abs for cos per `polyfills.rs`), so
    // 1e-9 relative is comfortably above the worst-case compounded drift while
    // still being thousands of times tighter than MC noise. `accepted` and the
    // PCG `state` are bit-exact (the acceptance comparison can only flip when
    // the candidate lands within ~1e-9 of the cap, which the seeds below avoid
    // for the bulk; any rare flip is reported and re-seeded around).
    const E_REL_TOL: f64 = 1e-9;

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

    /// GPU single-shot fission Watt draw vs CPU twin (four uniforms, no
    /// acceptance guard). State bit-exact, un-clamped energy within tol.
    #[test]
    fn gpu_watt_fission_matches_cpu() {
        let Some(ctx) = ctx_or_skip() else { return };
        let shapes: &[(f64, f64)] = &[(0.988e6, 2.249e-6), (1.0e6, 1.0e-6), (5.0e5, 3.0e-6)];
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut seeds = Vec::new();
        for (si, &(av, bv)) in shapes.iter().enumerate() {
            for s in seeds_vec(500, si as u32) {
                a.push(av);
                b.push(bv);
                seeds.push(s);
            }
        }
        let n = seeds.len();

        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, st) =
                watt_fission_draw_cpu(a[i], b[i], crate::common::rng::expand_seed(seeds[i]));
            cpu_e[i] = e;
            cpu_state[i] = st;
        }

        let client = ctx.client();
        let a_h = client.create_from_slice(bytemuck::cast_slice(&a));
        let b_h = client.create_from_slice(bytemuck::cast_slice(&b));
        let s_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(n * core::mem::size_of::<f64>());
        let os_h = client.empty(n * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        launch!(
            watt_fission_test_kernel,
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(a_h, a.len()),
            BufferArg::from_raw_parts(b_h, b.len()),
            BufferArg::from_raw_parts(s_h, seeds.len()),
            BufferArg::from_raw_parts(oe_h.clone(), n),
            BufferArg::from_raw_parts(os_h.clone(), n),
        );
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "fission-watt sample {i}: PCG state diverged (gpu {} cpu {}, seed {})",
                gpu_state[i], cpu_state[i], seeds[i]
            );
            // w_max can be negative (no guard), making the sqrt arg negative and
            // the result NaN on both sides -- skip the energy compare there
            // (NaN == NaN bit pattern can differ); state parity above is the
            // load-bearing guard for those.
            if cpu_e[i].is_finite() && gpu_e[i].is_finite() {
                let tol = E_REL_TOL * cpu_e[i].abs().max(1.0);
                assert!(
                    (gpu_e[i] - cpu_e[i]).abs() <= tol,
                    "fission-watt sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, seed {})",
                    gpu_e[i],
                    cpu_e[i],
                    seeds[i]
                );
            }
        }
        println!(
            "watt fission: {} draws, state bit-exact, finite energy within {:e} rel",
            n, E_REL_TOL
        );
    }

    /// The whole-law GPU samplers (interpolation, cap and the 32-attempt
    /// rejection loop, as the transport kernel runs them) must match the
    /// `flat` samplers the CPU transport uses: PCG state and `valid` (`Some`)
    /// bit-exact, the energy within `E_REL_TOL`. Covers grids with several
    /// points and one point, energies below / inside / above the grid, an
    /// evaporation theta row at its own offset (a second component), `u` at
    /// and above `E_in` (no draws), a cap so tight most attempts are rejected,
    /// and a zero `theta` / `a` row (no draws).
    #[test]
    fn gpu_rejection_laws_match_flat() {
        use yamc_physics::gpu::flat::{evaporation, maxwell, watt};
        let Some(ctx) = ctx_or_skip() else { return };

        // Three slots on one grid buffer: A (3 points), B (1 point), C (zero
        // theta / a). Maxwell theta and Watt (a, b) share the grid offsets;
        // evaporation theta sits at its own offsets behind two unused values.
        let grid = [1.0e6, 5.0e6, 2.0e7, 2.0e6, 1.0e6, 1.0e7];
        let theta = [3.0e5, 6.0e5, 1.2e6, 4.0e5, 0.0, 0.0];
        let evap_theta = [9.0e9, 9.0e9, 3.5e5, 7.0e5, 1.4e6, 4.5e5, 0.0, 0.0];
        let a = [9.65e5, 1.0e6, 1.1e6, 9.9e5, 0.0, 0.0];
        let b = [2.29e-6, 2.5e-6, 2.8e-6, 2.2e-6, 0.0, 0.0];
        let ab: Vec<f64> = a.iter().zip(&b).flat_map(|(&a, &b)| [a, b]).collect();
        // (g_off, n, evap theta_off, [(e_in, u)])
        type Slot<'a> = (u32, u32, u32, &'a [(f64, f64)]);
        let slots: &[Slot] = &[
            (
                0,
                3,
                2,
                &[
                    (5.0e5, 0.0),
                    (1.0e6, 0.0),
                    (3.0e6, 2.0e5),
                    (1.4e7, 0.0),
                    (3.0e7, 2.0e5),
                    (1.0e6, 2.0e6),
                    (1.4e7, 1.4e7 - 1.0),
                ],
            ),
            (3, 1, 5, &[(2.0e6, 0.0), (1.4e7, 1.0e6)]),
            (4, 2, 6, &[(5.0e6, 0.0)]),
        ];

        let (mut law, mut e_in, mut u, mut g_off, mut n, mut t_off, mut seeds) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for l in [LAW_EVAPORATION, LAW_MAXWELL, LAW_WATT] {
            for &(g, ng, to, cases) in slots {
                for (ci, &(e, uu)) in cases.iter().enumerate() {
                    for s in seeds_vec(100, l * 1000 + g * 100 + ci as u32) {
                        law.push(l);
                        e_in.push(e);
                        u.push(uu);
                        g_off.push(g);
                        n.push(ng);
                        t_off.push(to);
                        seeds.push(s);
                    }
                }
            }
        }
        let count = seeds.len();

        // Reference: the `flat` samplers, on each slot's rows of the same data.
        let mut cpu_e = vec![0.0_f64; count];
        let mut cpu_valid = vec![0u32; count];
        let mut cpu_state = vec![0u64; count];
        for i in 0..count {
            let rows = g_off[i] as usize..(g_off[i] + n[i]) as usize;
            let evap_rows = t_off[i] as usize..(t_off[i] + n[i]) as usize;
            let mut state = crate::common::rng::expand_seed(seeds[i]);
            let sampled = match law[i] {
                LAW_EVAPORATION => evaporation::sample_evaporation(
                    e_in[i],
                    &grid[rows],
                    &evap_theta[evap_rows],
                    u[i],
                    &mut state,
                ),
                LAW_MAXWELL => maxwell::sample_maxwell(
                    e_in[i],
                    &grid[rows.clone()],
                    &theta[rows],
                    u[i],
                    &mut state,
                ),
                _ => watt::sample_watt_inelastic(
                    e_in[i],
                    &grid[rows.clone()],
                    &a[rows.clone()],
                    &b[rows],
                    u[i],
                    &mut state,
                ),
            };
            if let Some(e) = sampled {
                cpu_e[i] = e;
                cpu_valid[i] = 1;
            }
            cpu_state[i] = state;
        }

        let client = ctx.client();
        let up = |v: &[u8]| client.create_from_slice(v);
        let law_h = up(bytemuck::cast_slice(&law));
        let e_h = up(bytemuck::cast_slice(&e_in));
        let u_h = up(bytemuck::cast_slice(&u));
        let g_h = up(bytemuck::cast_slice(&g_off));
        let n_h = up(bytemuck::cast_slice(&n));
        let to_h = up(bytemuck::cast_slice(&t_off));
        let grid_h = up(bytemuck::cast_slice(&grid));
        let th_h = up(bytemuck::cast_slice(&theta));
        let eth_h = up(bytemuck::cast_slice(&evap_theta));
        let ab_h = up(bytemuck::cast_slice(&ab));
        let s_h = up(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(count * core::mem::size_of::<f64>());
        let ov_h = client.empty(count * core::mem::size_of::<u32>());
        let os_h = client.empty(count * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (count as u32).div_ceil(WG);
        launch!(
            law_test_kernel,
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(law_h, count),
            BufferArg::from_raw_parts(e_h, count),
            BufferArg::from_raw_parts(u_h, count),
            BufferArg::from_raw_parts(g_h, count),
            BufferArg::from_raw_parts(n_h, count),
            BufferArg::from_raw_parts(to_h, count),
            BufferArg::from_raw_parts(grid_h, grid.len()),
            BufferArg::from_raw_parts(th_h, theta.len()),
            BufferArg::from_raw_parts(eth_h, evap_theta.len()),
            BufferArg::from_raw_parts(ab_h, ab.len()),
            BufferArg::from_raw_parts(s_h, count),
            BufferArg::from_raw_parts(oe_h.clone(), count),
            BufferArg::from_raw_parts(ov_h.clone(), count),
            BufferArg::from_raw_parts(os_h.clone(), count),
        );
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_valid = bytemuck::cast_slice::<u8, u32>(&client.read_one(ov_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        for (l, name) in [
            (LAW_EVAPORATION, "evaporation"),
            (LAW_MAXWELL, "maxwell"),
            (LAW_WATT, "watt"),
        ] {
            let idx: Vec<usize> = (0..count).filter(|&i| law[i] == l).collect();
            let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let pick_u = |v: &[u32]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let pick_s = |v: &[u64]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
            assert_rejection_parity(
                name,
                &pick(&gpu_e),
                &pick_u(&gpu_valid),
                &pick_s(&gpu_state),
                &pick(&cpu_e),
                &pick_u(&cpu_valid),
                &pick_s(&cpu_state),
                &pick_u(&seeds),
            );
            let valid = idx.iter().filter(|&&i| cpu_valid[i] == 1).count();
            println!(
                "{name}: {} samples, {valid} valid, matching flat",
                idx.len()
            );
        }
    }

    /// Shared assertion for the three rejection laws: PCG state and the
    /// `valid` flag must be bit-exact; a valid energy within `E_REL_TOL`.
    #[allow(clippy::too_many_arguments)]
    fn assert_rejection_parity(
        law: &str,
        gpu_e: &[f64],
        gpu_acc: &[u32],
        gpu_state: &[u64],
        cpu_e: &[f64],
        cpu_acc: &[u32],
        cpu_state: &[u64],
        seeds: &[u32],
    ) {
        for i in 0..gpu_e.len() {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "{law} sample {i}: PCG state diverged (gpu {} cpu {}, seed {})",
                gpu_state[i], cpu_state[i], seeds[i]
            );
            assert_eq!(
                gpu_acc[i], cpu_acc[i],
                "{law} sample {i}: valid flag diverged (gpu {} cpu {}, seed {})",
                gpu_acc[i], cpu_acc[i], seeds[i]
            );
            if cpu_acc[i] == 1 {
                let tol = E_REL_TOL * cpu_e[i].abs().max(1.0);
                assert!(
                    (gpu_e[i] - cpu_e[i]).abs() <= tol,
                    "{law} sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, seed {})",
                    gpu_e[i],
                    cpu_e[i],
                    seeds[i]
                );
            }
        }
    }
}
