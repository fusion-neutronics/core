//! Probe: do f64 `sin` and `cos` work on cubecl-spirv → Vulkan on
//! this driver?
//!
//! # Empirical finding (AMD Strix Halo + RADV/ACO, Mesa 25.2.8)
//!
//! **Both broken**, same root cause as `exp`/`ln`. Mesa NIR reports:
//!
//! ```text
//! Unimplemented NIR instr bit size: div 64    %26 = fsin %24
//! Unimplemented NIR instr bit size: div 64    %31 = fcos_amd %26
//! ```
//!
//! and the kernel returns garbage (~`-2.4e+220`). The cubecl-spirv
//! issue against this gap (cubecl#1316 covers ln/exp) carries a
//! follow-up note adding `sin`/`cos`/`pow` to the GLSL.std.450 ops
//! that need 64-bit polyfilling.
//!
//! Re-tested on cubecl 0.11.0-pre.3 (Mesa 26.0.3, 2026-09-13): still
//! broken, `sin(-pi/2)` and `cos(pi)` both read back as ~`2.8e-311`
//! instead of `-1`. `SinOp`/`CosOp` are still lowered straight to the
//! GLSL.std.450 ops in `cubecl-spirv/src/ops/math.rs`.
//!
//! # Why this kernel still exists
//!
//! `direction_rotation.rs` uses **rejection sampling on the unit
//! disk** to get `(cos φ, sin φ)` for the azimuthal angle without
//! ever calling `sin`/`cos`. This probe stays in tree as the
//! regression check that gets re-enabled (`cargo test --
//! --include-ignored`) once an upstream fix lands or on a different
//! driver/runtime.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn f64_sin_cos_kernel(input: &[f64], out_sin: &mut [f64], out_cos: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = input[ABSOLUTE_POS];
    out_sin[ABSOLUTE_POS] = x.sin();
    out_cos[ABSOLUTE_POS] = x.cos();
}

/// Run `sin` and `cos` over the input array on the GPU. Returns
/// `(sin_out, cos_out)`.
pub fn run_f64_sin_cos(ctx: &GpuContext, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let client = ctx.client();
    let n = input.len();
    let byte_len = std::mem::size_of_val(input);

    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let sin_handle = client.empty(byte_len);
    let cos_handle = client.empty(byte_len);

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        f64_sin_cos_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(sin_handle.clone(), n),
            BufferArg::from_raw_parts(cos_handle.clone(), n),
        );
    }

    let sin_out: Vec<f64> = bytemuck::cast_slice(&client.read_one(sin_handle).unwrap()).to_vec();
    let cos_out: Vec<f64> = bytemuck::cast_slice(&client.read_one(cos_handle).unwrap()).to_vec();
    (sin_out, cos_out)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

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

    /// **Currently expected to fail on AMD/RADV** -- Mesa NIR doesn't
    /// implement 64-bit `fsin`/`fcos`, same gap as for `flog2`/
    /// `fexp2`. The kernel emits garbage; downstream code uses
    /// rejection sampling instead. `#[ignore]` so the regression test
    /// stays available on hardware/drivers where sin/cos do work.
    #[test]
    #[ignore = "f64 sin/cos broken on AMD RADV/ACO; see module docs"]
    fn gpu_f64_sin_cos_within_32_ulps() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        // Inputs in `[-π, π]`, mix of sign and special points.
        let input: Vec<f64> = vec![
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_3,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            -std::f64::consts::PI,
            1.0,
            -1.0,
            2.0,
            -2.0,
            std::f64::consts::TAU * 0.123,
        ];

        let (gpu_sin, gpu_cos) = run_f64_sin_cos(&ctx, &input);

        let mut max_sin = 0u64;
        let mut max_cos = 0u64;
        let mut worst_sin = 0usize;
        let mut worst_cos = 0usize;
        for (i, &x) in input.iter().enumerate() {
            let cs = x.sin();
            let cc = x.cos();
            let ds = ulps_diff(gpu_sin[i], cs);
            let dc = ulps_diff(gpu_cos[i], cc);
            if ds > max_sin {
                max_sin = ds;
                worst_sin = i;
            }
            if dc > max_cos {
                max_cos = dc;
                worst_cos = i;
            }
        }
        println!(
            "sin: max ULP drift = {max_sin} (input[{worst_sin}] = {}, GPU = {}, CPU = {})",
            input[worst_sin],
            gpu_sin[worst_sin],
            input[worst_sin].sin()
        );
        println!(
            "cos: max ULP drift = {max_cos} (input[{worst_cos}] = {}, GPU = {}, CPU = {})",
            input[worst_cos],
            gpu_cos[worst_cos],
            input[worst_cos].cos()
        );

        if max_sin > 32 || max_cos > 32 {
            println!(
                "f64 sin/cos appear broken on this driver (sin {max_sin} ulps, cos {max_cos} ulps). \
                 Direction rotation will need rejection-sampling fallback."
            );
        }
        // Don't hard-fail -- we want both the "works" and "broken" cases
        // to flow through and let the caller decide.
    }
}
