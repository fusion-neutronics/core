//! f64 transcendentals coverage check on cubecl-spirv → Vulkan.
//!
//! The cubecl book lists f64 on SPIR-V as `❔` ("partial"), with a note
//! that not all ops are implemented. `f64_arith` covered the core IEEE
//! ops (`+ - * cmp`); this kernel covers the next tier -- `sqrt`, `exp`,
//! `ln` -- which yamc cannot do without:
//! - `ln` is in the distance-to-collision sampling: `-ln(ξ) / Σ`.
//! - `sqrt` is in nearly every geometry path (vector magnitudes,
//!   sphere/cylinder distances).
//! - `exp` is in attenuation and several photon scoring helpers.
//!
//! # Empirical findings (AMD Strix Halo + RADV/ACO, Mesa)
//!
//! - **`sqrt` works perfectly.** 0 ULP drift across all tested inputs.
//!   This is the only transcendental this branch can rely on today.
//! - **`exp` and `ln` are broken on this driver.** Mesa NIR emits
//!   `Unimplemented NIR instr bit size: div 64 %N = flog2 %M` and the
//!   ACO compiler then errors with use-before-def
//!   (`aco_validate.cpp: %N never defined: v_mul_f64_e64`). The kernel
//!   returns garbage (`exp(1e-6) → NaN`, `ln(e) → 0.6534…`). Root
//!   cause is Mesa's NIR not implementing 64-bit `flog2` / `fexp2`,
//!   which cubecl-spirv (or the Vulkan extended-instruction-set
//!   lowering) requires for f64 exp/ln.
//! - **NVIDIA Vulkan untested here** -- likely works since NVIDIA's f64
//!   path is more mature, but until verified the safe assumption is
//!   "exp/ln may need a polyfill on at least one major Vulkan driver."
//!
//! sqrt and exp/ln are split into separate kernels so the passing
//! sqrt test runs without triggering the AMD driver's compile-time
//! warnings on the broken paths.
//!
//! `gpu_f64_sqrt_within_4_ulps` always asserts (passes here).
//! `gpu_f64_exp_and_ln_within_4_ulps` is `#[ignore]` until upstream
//! fixes -- runnable with `cargo test -- --include-ignored`.
//!
//! # Strategic implication for yamc
//!
//! The distance-to-collision kernel cannot be written using `x.ln()`
//! directly on AMD/RADV today. Mitigations, in increasing order of
//! complexity:
//! 1. Use NVIDIA hardware for production runs; document AMD/Mesa as
//!    unsupported until upstream fixes land.
//! 2. Polyfill `ln(x)` and `exp(x)` from basic arithmetic + bit
//!    manipulation (e.g., extract f64 exponent + Padé approximation
//!    on the mantissa). Adds ~50 lines per function, ~5–10x slowdown
//!    vs native, but works everywhere.
//! 3. Wait for / contribute to a Mesa NIR or cubecl-spirv fix.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn f64_sqrt_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS].sqrt();
}

#[cube(launch_unchecked)]
fn f64_exp_ln_kernel(input: &[f64], out_exp: &mut [f64], out_ln: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = input[ABSOLUTE_POS];
    out_exp[ABSOLUTE_POS] = x.exp();
    out_ln[ABSOLUTE_POS] = x.ln();
}

/// Run `sqrt` over the input array on the GPU.
pub fn run_f64_sqrt(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        f64_sqrt_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run `exp` and `ln` over the input array on the GPU. **Currently
/// broken on AMD/RADV** -- see module docs.
pub fn run_f64_exp_ln(ctx: &GpuContext, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let client = ctx.client();
    let n = input.len();
    let byte_len = std::mem::size_of_val(input);

    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let exp_handle = client.empty(byte_len);
    let ln_handle = client.empty(byte_len);

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        f64_exp_ln_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(exp_handle.clone(), n),
            BufferArg::from_raw_parts(ln_handle.clone(), n),
        );
    }
    let exp_out: Vec<f64> = bytemuck::cast_slice(&client.read_one(exp_handle).unwrap()).to_vec();
    let ln_out: Vec<f64> = bytemuck::cast_slice(&client.read_one(ln_handle).unwrap()).to_vec();
    (exp_out, ln_out)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// ULP distance between two same-sign finite f64s; `u64::MAX` for
    /// NaN / infinity / opposite-sign / non-finite mismatch (treated as
    /// "infinitely far apart" so the assert fires loudly).
    fn ulps_diff(a: f64, b: f64) -> u64 {
        if a.is_nan()
            || b.is_nan()
            || a.is_infinite() != b.is_infinite()
            || a.is_sign_negative() != b.is_sign_negative()
        {
            return u64::MAX;
        }
        let ai = a.to_bits();
        let bi = b.to_bits();
        ai.max(bi) - ai.min(bi)
    }

    fn worst_ulps(label: &str, gpu: &[f64], cpu: &[f64], input: &[f64]) -> u64 {
        let mut max_ulps = 0u64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = ulps_diff(g, c);
            if d > max_ulps {
                max_ulps = d;
                worst_idx = i;
            }
        }
        println!(
            "{label}: max ULP drift = {max_ulps} (input[{worst_idx}] = {}, GPU = {}, CPU = {})",
            input[worst_idx], gpu[worst_idx], cpu[worst_idx]
        );
        max_ulps
    }

    fn standard_inputs() -> Vec<f64> {
        // Positive inputs spanning yamc-relevant magnitudes. Avoid 0
        // (ln(0) = -inf) and avoid huge inputs (exp overflows ~700).
        vec![
            1e-6,
            1e-3,
            0.1,
            0.5,
            std::f64::consts::E.recip(),
            1.0,
            2.0,
            std::f64::consts::E,
            std::f64::consts::PI,
            10.0,
            100.0,
        ]
    }

    /// `sqrt` is the only transcendental we can verify works on every
    /// Vulkan driver tested so far (AMD/RADV gives 0 ULP drift). Strict
    /// 4-ULP bar applied -- if this ever loosens, something regressed.
    #[test]
    fn gpu_f64_sqrt_within_4_ulps() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let input = standard_inputs();
        let gpu_sqrt = run_f64_sqrt(&ctx, &input);
        let cpu_sqrt: Vec<f64> = input.iter().map(|x| x.sqrt()).collect();
        let max_ulps = worst_ulps("sqrt", &gpu_sqrt, &cpu_sqrt, &input);
        assert!(
            max_ulps <= 4,
            "GPU f64 sqrt drifts > 4 ulps from CPU (max {max_ulps})"
        );
    }

    /// **Currently expected to fail on AMD/RADV** -- see module docs.
    /// The kernel calls `x.exp()` and `x.ln()`; on the AMD ACO compiler
    /// these produce a use-before-def error and return garbage. The
    /// test stays in tree as documentation of what we tested and as a
    /// regression check that can be re-enabled (`cargo test --
    /// --include-ignored`) on hardware/drivers that handle f64 exp/ln
    /// correctly, or once an upstream fix lands.
    #[test]
    #[ignore = "f64 exp/ln broken on AMD RADV/ACO; see module docs"]
    fn gpu_f64_exp_and_ln_within_4_ulps() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let input = standard_inputs();
        let (gpu_exp, gpu_ln) = run_f64_exp_ln(&ctx, &input);
        let cpu_exp: Vec<f64> = input.iter().map(|x| x.exp()).collect();
        let cpu_ln: Vec<f64> = input.iter().map(|x| x.ln()).collect();
        let exp_ulps = worst_ulps("exp", &gpu_exp, &cpu_exp, &input);
        let ln_ulps = worst_ulps("ln (natural log)", &gpu_ln, &cpu_ln, &input);
        const TOL: u64 = 4;
        assert!(
            exp_ulps <= TOL,
            "GPU f64 exp drifts > {TOL} ulps from CPU (max {exp_ulps})"
        );
        assert!(
            ln_ulps <= TOL,
            "GPU f64 ln drifts > {TOL} ulps from CPU (max {ln_ulps})"
        );
    }
}
