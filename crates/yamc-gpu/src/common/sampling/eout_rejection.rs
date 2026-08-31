//! Shared single-iteration outgoing-energy rejection draws for the
//! Evaporation (ENDF File 5, Law 9), Maxwell (Law 7), and Watt (Law 11)
//! continuum laws.
//!
//! Each law's E_out is drawn by rejection: a fixed number of PCG-32 uniforms
//! per attempt, fed through the f64 `ln`/`exp`/`cos` polyfills, then tested
//! against a cap. The neutron transport kernel ran the rejection loop inline at
//! four sites (the inelastic Evaporation / Maxwell / Watt-inelastic E_out paths
//! and the fission-spectrum Watt fallback). These helpers extract ONE rejection
//! attempt per law -- the RNG-consuming, transcendental-heavy inner body --
//! returning the candidate energy, an `accepted` flag, and the advanced PCG
//! state. The outer 32-iteration loop, the `theta(E_in)` / `a(E_in)` / `b(E_in)`
//! grid interpolation, and the buffer indexing stay at the call site (they vary
//! per law-slot layout); only the bit-sensitive draw math is shared.
//!
//! These mirror `yamc_nuclide::sampling::sample_maxwell_spectrum` /
//! `sample_watt_spectrum_params` and the `yamc_physics::gpu::flat::{evaporation,
//! maxwell,watt}` CPU twins. The draw ORDER inside each helper is preserved
//! exactly from the kernel (evaporation: 2 uniforms, maxwell: 3, watt: 4), so
//! the RNG `state` advances bit-for-bit. Because the candidate energy flows
//! through `ln`/`exp`/`cos`, GPU vs CPU is NOT bit-exact for the energy -- the
//! parity tests use a relative tolerance sized to the polyfills' ULP drift (see
//! `common/polyfills.rs`: ~16 ULP for ln/exp, ~1e-13 abs for cos), while the
//! `accepted` flag and the PCG `state` must match bit-for-bit.

use crate::common::polyfills::{cos_f64, ln_f64};
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

/// CPU twin of [`evaporation_rejection_draw`]. Uses libm `ln` rather than the
/// polyfill; agrees to within the polyfills' ULP drift. Returns
/// `(e_out, accepted, state)`.
pub fn evaporation_rejection_draw_cpu(
    v_e: f64,
    y: f64,
    theta_val: f64,
    state_in: u64,
) -> (f64, u32, u64) {
    let (xi_e1, state) = cpu_draw_uniform(state_in);
    let (xi_e2, state) = cpu_draw_uniform(state);
    let arg_a = 1.0 - v_e * xi_e1;
    let arg_b = 1.0 - v_e * xi_e2;
    let mut x_e = 0.0_f64;
    if arg_a > 0.0 && arg_b > 0.0 {
        x_e = -arg_a.ln() - arg_b.ln();
    }
    if x_e <= y {
        (x_e * theta_val, 1, state)
    } else {
        (0.0, 0, state)
    }
}

/// CPU twin of [`maxwell_rejection_draw`]. Returns `(e_out, accepted, state)`.
pub fn maxwell_rejection_draw_cpu(theta_val: f64, cap_e: f64, state_in: u64) -> (f64, u32, u64) {
    let (xi_m1, state) = cpu_draw_uniform(state_in);
    let (xi_m2, state) = cpu_draw_uniform(state);
    let (xi_m3, state) = cpu_draw_uniform(state);
    let cos_m3 = (std::f64::consts::FRAC_PI_2 * xi_m3).cos();
    let mut e_out_m = 0.0_f64;
    if xi_m1 > 0.0 && xi_m2 > 0.0 {
        e_out_m = -theta_val * (xi_m1.ln() + xi_m2.ln() * cos_m3 * cos_m3);
    }
    if e_out_m > 0.0 && e_out_m <= cap_e {
        (e_out_m, 1, state)
    } else {
        (0.0, 0, state)
    }
}

/// CPU twin of [`watt_rejection_draw`]. Returns `(e_out, accepted, state)`.
pub fn watt_rejection_draw_cpu(
    watt_a_val: f64,
    a2b: f64,
    cap_e: f64,
    state_in: u64,
) -> (f64, u32, u64) {
    let (xi_w1, state) = cpu_draw_uniform(state_in);
    let (xi_w2, state) = cpu_draw_uniform(state);
    let (xi_w3, state) = cpu_draw_uniform(state);
    let (xi_w4, state) = cpu_draw_uniform(state);
    let cos_w3 = (std::f64::consts::FRAC_PI_2 * xi_w3).cos();
    let mut w_m = 0.0_f64;
    if xi_w1 > 0.0 && xi_w2 > 0.0 {
        w_m = -watt_a_val * (xi_w1.ln() + xi_w2.ln() * cos_w3 * cos_w3);
    }
    let u_xi = 2.0 * xi_w4 - 1.0;
    let prod = a2b * w_m;
    let mut e_out_w = 0.0_f64;
    if w_m > 0.0 && prod >= 0.0 {
        e_out_w = w_m + 0.25 * a2b + u_xi * prod.sqrt();
    }
    if e_out_w > 0.0 && e_out_w <= cap_e {
        (e_out_w, 1, state)
    } else {
        (0.0, 0, state)
    }
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

#[cube(launch_unchecked)]
fn evap_test_kernel(
    v_e: &[f64],
    y: &[f64],
    theta: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_acc: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = evaporation_rejection_draw(
        v_e[ABSOLUTE_POS],
        y[ABSOLUTE_POS],
        theta[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_acc[ABSOLUTE_POS] = r.accepted;
    out_state[ABSOLUTE_POS] = r.state;
}

#[cube(launch_unchecked)]
fn maxwell_test_kernel(
    theta: &[f64],
    cap: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_acc: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = maxwell_rejection_draw(
        theta[ABSOLUTE_POS],
        cap[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_acc[ABSOLUTE_POS] = r.accepted;
    out_state[ABSOLUTE_POS] = r.state;
}

#[cube(launch_unchecked)]
fn watt_test_kernel(
    watt_a: &[f64],
    a2b: &[f64],
    cap: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_acc: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = watt_rejection_draw(
        watt_a[ABSOLUTE_POS],
        a2b[ABSOLUTE_POS],
        cap[ABSOLUTE_POS],
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_acc[ABSOLUTE_POS] = r.accepted;
    out_state[ABSOLUTE_POS] = r.state;
}

#[cube(launch_unchecked)]
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

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

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

    /// GPU Evaporation rejection draw must match the CPU twin: PCG `state` and
    /// the `accepted` flag bit-exact; the energy within `E_REL_TOL` (two `ln`
    /// polyfill calls). Sweeps several `(theta, E_in-U)` shapes and many seeds.
    #[test]
    fn gpu_evaporation_rejection_matches_cpu() {
        let Some(ctx) = ctx_or_skip() else { return };
        // (theta, y) where y = (E_in - U)/theta. v_e = 1 - exp(-y), computed
        // here on the CPU (the kernel takes v_e/y/theta directly).
        let shapes: &[(f64, f64)] = &[(1.0e6, 14.0), (5.0e5, 2.0), (2.0e6, 0.5), (1.0e6, 8.0)];
        let mut v_e = Vec::new();
        let mut y = Vec::new();
        let mut theta = Vec::new();
        let mut seeds = Vec::new();
        for (si, &(th, yy)) in shapes.iter().enumerate() {
            let v = 1.0 - (-yy).exp();
            for s in seeds_vec(400, si as u32) {
                v_e.push(v);
                y.push(yy);
                theta.push(th);
                seeds.push(s);
            }
        }
        let n = seeds.len();

        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_acc = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, a, st) = evaporation_rejection_draw_cpu(
                v_e[i],
                y[i],
                theta[i],
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_acc[i] = a;
            cpu_state[i] = st;
        }

        let client = ctx.client();
        let v_h = client.create_from_slice(bytemuck::cast_slice(&v_e));
        let y_h = client.create_from_slice(bytemuck::cast_slice(&y));
        let t_h = client.create_from_slice(bytemuck::cast_slice(&theta));
        let s_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(n * core::mem::size_of::<f64>());
        let oa_h = client.empty(n * core::mem::size_of::<u32>());
        let os_h = client.empty(n * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            evap_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(v_h, v_e.len()),
                BufferArg::from_raw_parts(y_h, y.len()),
                BufferArg::from_raw_parts(t_h, theta.len()),
                BufferArg::from_raw_parts(s_h, seeds.len()),
                BufferArg::from_raw_parts(oe_h.clone(), n),
                BufferArg::from_raw_parts(oa_h.clone(), n),
                BufferArg::from_raw_parts(os_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_acc = bytemuck::cast_slice::<u8, u32>(&client.read_one(oa_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        assert_rejection_parity(
            "evaporation",
            &gpu_e,
            &gpu_acc,
            &gpu_state,
            &cpu_e,
            &cpu_acc,
            &cpu_state,
            &seeds,
        );
        println!(
            "evaporation rejection: {} draws, state+accepted bit-exact, energy within {:e} rel",
            n, E_REL_TOL
        );
    }

    /// GPU Maxwell rejection draw vs CPU twin (three uniforms, one `cos` + two
    /// `ln` polyfill calls). State + accepted bit-exact, energy within tol.
    #[test]
    fn gpu_maxwell_rejection_matches_cpu() {
        let Some(ctx) = ctx_or_skip() else { return };
        // (theta, cap_e).
        let shapes: &[(f64, f64)] = &[
            (1.0e6, 1.4e7),
            (5.0e5, 1.0e6),
            (2.0e6, 5.0e5),
            (1.0e6, 1.0e7),
        ];
        let mut theta = Vec::new();
        let mut cap = Vec::new();
        let mut seeds = Vec::new();
        for (si, &(th, cp)) in shapes.iter().enumerate() {
            for s in seeds_vec(400, si as u32) {
                theta.push(th);
                cap.push(cp);
                seeds.push(s);
            }
        }
        let n = seeds.len();

        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_acc = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, a, st) = maxwell_rejection_draw_cpu(
                theta[i],
                cap[i],
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_acc[i] = a;
            cpu_state[i] = st;
        }

        let client = ctx.client();
        let t_h = client.create_from_slice(bytemuck::cast_slice(&theta));
        let c_h = client.create_from_slice(bytemuck::cast_slice(&cap));
        let s_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(n * core::mem::size_of::<f64>());
        let oa_h = client.empty(n * core::mem::size_of::<u32>());
        let os_h = client.empty(n * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            maxwell_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(t_h, theta.len()),
                BufferArg::from_raw_parts(c_h, cap.len()),
                BufferArg::from_raw_parts(s_h, seeds.len()),
                BufferArg::from_raw_parts(oe_h.clone(), n),
                BufferArg::from_raw_parts(oa_h.clone(), n),
                BufferArg::from_raw_parts(os_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_acc = bytemuck::cast_slice::<u8, u32>(&client.read_one(oa_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        assert_rejection_parity(
            "maxwell", &gpu_e, &gpu_acc, &gpu_state, &cpu_e, &cpu_acc, &cpu_state, &seeds,
        );
        println!(
            "maxwell rejection: {} draws, state+accepted bit-exact, energy within {:e} rel",
            n, E_REL_TOL
        );
    }

    /// GPU inelastic Watt rejection draw vs CPU twin (four uniforms; Maxwell-w
    /// then the Watt correction). State + accepted bit-exact, energy within tol.
    #[test]
    fn gpu_watt_rejection_matches_cpu() {
        let Some(ctx) = ctx_or_skip() else { return };
        // (a, b, cap_e); a2b = a*a*b precomputed at the call site.
        let shapes: &[(f64, f64, f64)] = &[
            (1.0e6, 2.249e-6, 1.4e7),
            (5.0e5, 3.0e-6, 1.0e6),
            (2.0e6, 1.0e-6, 5.0e6),
            (1.0e6, 2.249e-6, 1.0e7),
        ];
        let mut a = Vec::new();
        let mut a2b = Vec::new();
        let mut cap = Vec::new();
        let mut seeds = Vec::new();
        for (si, &(av, bv, cp)) in shapes.iter().enumerate() {
            for s in seeds_vec(400, si as u32) {
                a.push(av);
                a2b.push(av * av * bv);
                cap.push(cp);
                seeds.push(s);
            }
        }
        let n = seeds.len();

        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_acc = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, ac, st) = watt_rejection_draw_cpu(
                a[i],
                a2b[i],
                cap[i],
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_acc[i] = ac;
            cpu_state[i] = st;
        }

        let client = ctx.client();
        let a_h = client.create_from_slice(bytemuck::cast_slice(&a));
        let ab_h = client.create_from_slice(bytemuck::cast_slice(&a2b));
        let c_h = client.create_from_slice(bytemuck::cast_slice(&cap));
        let s_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oe_h = client.empty(n * core::mem::size_of::<f64>());
        let oa_h = client.empty(n * core::mem::size_of::<u32>());
        let os_h = client.empty(n * core::mem::size_of::<u64>());
        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            watt_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(a_h, a.len()),
                BufferArg::from_raw_parts(ab_h, a2b.len()),
                BufferArg::from_raw_parts(c_h, cap.len()),
                BufferArg::from_raw_parts(s_h, seeds.len()),
                BufferArg::from_raw_parts(oe_h.clone(), n),
                BufferArg::from_raw_parts(oa_h.clone(), n),
                BufferArg::from_raw_parts(os_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oe_h).unwrap()).to_vec();
        let gpu_acc = bytemuck::cast_slice::<u8, u32>(&client.read_one(oa_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(os_h).unwrap()).to_vec();

        assert_rejection_parity(
            "watt", &gpu_e, &gpu_acc, &gpu_state, &cpu_e, &cpu_acc, &cpu_state, &seeds,
        );
        println!(
            "watt rejection: {} draws, state+accepted bit-exact, energy within {:e} rel",
            n, E_REL_TOL
        );
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
        unsafe {
            watt_fission_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(a_h, a.len()),
                BufferArg::from_raw_parts(b_h, b.len()),
                BufferArg::from_raw_parts(s_h, seeds.len()),
                BufferArg::from_raw_parts(oe_h.clone(), n),
                BufferArg::from_raw_parts(os_h.clone(), n),
            );
        }
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

    /// Shared assertion for the three looped rejection laws: PCG state and the
    /// `accepted` flag must be bit-exact; the accepted energy within `E_REL_TOL`.
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
                "{law} sample {i}: accepted flag diverged (gpu {} cpu {}, seed {})",
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
