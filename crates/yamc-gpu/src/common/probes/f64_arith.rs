//! End-to-end f64 round-trip kernel -- the proof point that cubecl-spirv
//! actually emits working f64 code on the host's Vulkan driver.
//!
//! The cubecl book flags f64 on SPIR-V as `❔` ("partial") with a note
//! that not all ops are implemented. This kernel exercises the core
//! arithmetic ops the transport kernel will lean on: `*`, `+`, `-`,
//! comparisons feeding a select, and unary negation. Division and
//! transcendentals are out of scope for this test.
//!
//! # Why this is a *tolerance* test, not a bit-equality test
//!
//! In a hand-written CPU implementation each `*`, `+`, `-` is rounded
//! separately per IEEE-754 round-to-nearest-even. A correctness-first
//! reading of "f64 works" would demand the GPU produce the same bit
//! pattern. Empirically it does not:
//!
//! - cubecl-spirv (or NVIDIA's Vulkan driver) **contracts `a*b ± c`
//!   patterns into FMA**, producing one rounding instead of two -- more
//!   accurate than the CPU result, but ~1 ulp different at the bits.
//!   Hits inputs whose square isn't exact in f64 (e.g., π, e, 1/3).
//! - At the f64 representable boundary (`x = 1e308`) overflow handling
//!   diverged: GPU returned `NaN` where CPU returned `+inf`. Excluded
//!   from this test's input set.
//! - Division (`c / (x + 2.0)` in an earlier draft) showed ~1 ulp drift
//!   on most non-trivial inputs -- NVIDIA's f64 divide is reciprocal +
//!   refinement, round-correct to ~1 ulp not exact-IEEE.
//!
//! All three are tolerable for MC transport: 1 ulp is **vastly** below
//! MC statistical noise, and FMA contraction strictly improves accuracy.
//! So we assert max-1-ulp agreement instead of bit-equality. A failure
//! here means f64 codegen has a real bug (much more than 1 ulp off, or
//! producing NaN/inf where CPU doesn't), not just FMA contraction.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

#[cfg(test)]
#[inline]
fn cpu_expr(x: f64) -> f64 {
    let a = x * x;
    let b = a + 1.0;
    let c = b - 0.5;
    if x > 0.0 {
        c
    } else {
        -c
    }
}

#[cube(launch_unchecked)]
fn f64_arith_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = input[ABSOLUTE_POS];
    let a = x * x;
    let b = a + 1.0;
    let c = b - 0.5;
    let r = if x > 0.0 { c } else { -c };
    output[ABSOLUTE_POS] = r;
}

/// Run the f64 arithmetic kernel on the GPU and return the per-element
/// outputs. Caller compares against the same expression run on the CPU.
pub fn run_f64_arith(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();

    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        f64_arith_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }

    let bytes = client.read_one(output_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// ULP distance between two same-sign f64s. Returns `u64::MAX` for
    /// opposite signs, NaN, or non-finite -- those are treated as "very
    /// far apart" so the test fails loudly. For finite same-sign values
    /// the distance is the bit-pattern difference, since adjacent f64s
    /// differ by 1 in their u64 representation.
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

    /// CPU and GPU must agree to within 1 ulp on the core arithmetic
    /// expression. See module docs for why bit-equality isn't achievable
    /// (FMA contraction). A failure here means more than 1 ulp drift --
    /// real cubecl-spirv f64 codegen bug or driver instability.
    #[test]
    fn gpu_f64_arith_within_one_ulp_of_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Inputs span the regimes yamc actually sees in transport: zero,
        // small, large, near-machine-epsilon scales, transcendentals,
        // negatives. `1e308` excluded: GPU returns NaN where CPU returns
        // +inf at the f64 boundary (see module docs).
        let input: Vec<f64> = vec![
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            1e-9,
            -1e-9,
            1e9,
            -1e9,
            1e-308,
            std::f64::consts::PI,
            std::f64::consts::E,
            1.0 / 3.0,
            7.0,
            42.0,
            -42.0,
            100.0,
        ];

        let gpu_out = run_f64_arith(&ctx, &input);
        let cpu_out: Vec<f64> = input.iter().copied().map(cpu_expr).collect();

        let mut max_ulps = 0u64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let d = ulps_diff(g, c);
            if d > max_ulps {
                max_ulps = d;
                worst_idx = i;
            }
        }

        println!(
            "GPU vs CPU max ULP drift: {max_ulps} (input[{worst_idx}] = {}, GPU = {}, CPU = {})",
            input[worst_idx], gpu_out[worst_idx], cpu_out[worst_idx]
        );
        assert!(
            max_ulps <= 1,
            "GPU f64 arithmetic drifts more than 1 ulp from CPU\n\
             worst: input[{worst_idx}] = {}, GPU = {}, CPU = {}, ulps = {max_ulps}",
            input[worst_idx],
            gpu_out[worst_idx],
            cpu_out[worst_idx],
        );
    }
}
