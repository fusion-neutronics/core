//! Software polyfills for f64 transcendentals on the cubecl-spirv path.
//!
//! Why these exist: cubecl-spirv emits invalid GLSL.std.450 `OpExtInst
//! Log` / `Exp` for f64 operands (the spec restricts those to f16/f32).
//! NVIDIA's Vulkan driver tolerates it; AMD/RADV correctly rejects it
//! and the kernel returns garbage. See:
//! - Mesa work item: <https://gitlab.freedesktop.org/mesa/mesa/-/work_items/15359>
//! - cubecl issue #1316: <https://github.com/tracel-ai/cubecl/issues/1316>
//!
//! Until cubecl-spirv emits a polyfill at codegen time, yamc-gpu does
//! it at the source level. These functions use only ops we've validated
//! work on cubecl-spirv via Vulkan: `+ - * /`, comparisons, unary `-`,
//! `sqrt` (verified in `f64_arith` and `f64_transcendentals`), plus
//! `Reinterpret` for bit-pattern manipulation of f64 ↔ u64.
//!
//! # Algorithm: `ln_f64`
//!
//! Standard fdlibm-style approach:
//! 1. Extract biased exponent and mantissa from the f64 bit pattern.
//!    With `m ∈ [1, 2)` and `e` the unbiased exponent: `x = m * 2^e`.
//! 2. If `m > sqrt(2)`, halve and bump `e` so `m ∈ [sqrt(2)/2, sqrt(2))`.
//!    Keeps the substitution variable small.
//! 3. Substitute `u = (m-1)/(m+1)`. Then `m = (1+u)/(1-u)`, and
//!    `ln(m) = 2*atanh(u) = 2*(u + u³/3 + u⁵/5 + …)`. With `m` reduced
//!    above, `|u| < 0.172f64` and the series converges to ~1 ulp in ~10
//!    terms.
//! 4. `ln(x) = ln(m) + e * ln(2)`.
//!
//! # Algorithm: `exp_f64`
//!
//! 1. Range reduction: `k = round(x / ln 2)`, `r = x - k * ln 2`. Then
//!    `r ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347f64, 0.347f64]` and
//!    `exp(x) = 2^k * exp(r)`.
//! 2. `exp(r)` via Taylor series. With `|r| ≤ 0.347f64`, ~14 terms hit
//!    ~1 ulp.
//! 3. `2^k` by direct construction of an f64 with biased exponent
//!    `k + 1023` and zero mantissa.
//!
//! # Accuracy and inputs
//!
//! Targets ~4–16 ulps vs CPU libm -- well below MC noise. Inputs that
//! would over/underflow are out of scope for the first cut: `ln_f64`
//! assumes `x > 0` and finite; `exp_f64` assumes `|x| < ~700` (no
//! overflow checks). yamc transport never calls these on out-of-range
//! values, so guarding adds cost without benefit.

use cubecl::prelude::*;
use std::f64::consts;

// Pull constants from std rather than re-typing approximations -- clippy
// flags hand-rolled literals for these. INV_LN_2 = log2(e).
const LN_2: f64 = consts::LN_2;
const INV_LN_2: f64 = consts::LOG2_E;
const SQRT_2: f64 = consts::SQRT_2;

/// Natural log polyfill. See module docs for the algorithm. Assumes
/// `x > 0` and finite.
#[cube]
pub fn ln_f64(x: f64) -> f64 {
    // 1. Extract exponent and mantissa from the f64 bit pattern.
    let bits = u64::reinterpret(x);
    let exp_bits = (bits >> 52u64) & 0x7FFu64;
    // Reset the exponent field to bias 1023 (i.e., 2^0) so the
    // reconstructed value is the mantissa m ∈ [1, 2).
    let mantissa_bits = (bits & 0x000F_FFFF_FFFF_FFFFu64) | 0x3FF0_0000_0000_0000u64;
    let mut m = f64::reinterpret(mantissa_bits);
    // Unbiased exponent as f64 (we'll multiply by ln 2 later anyway).
    // exp_bits is in [0, 2047]; for finite positive x with exponent in
    // the normal range, exp_bits ≥ 1.
    let mut e = exp_bits as f64 - 1023.0f64;

    // 2. If m > sqrt(2), halve it and bump e so |u| stays small.
    if m > SQRT_2 {
        m *= 0.5f64;
        e += 1.0f64;
    }

    // 3. u = (m - 1) / (m + 1), then ln(m) = 2*(u + u^3/3 + u^5/5 + …).
    // Polynomial in u^2: ln_m / (2u) = 1 + u^2/3 + u^4/5 + u^6/7 + …
    // 12 terms is comfortably below 1 ulp for |u| < 0.172f64.
    // Horner form expressed as sequential bindings to dodge bracket
    // depth (the deeply-nested form is unreadable and error-prone).
    let u = (m - 1.0f64) / (m + 1.0f64);
    let u2 = u * u;
    let h11 = 1.0f64 / 23.0f64;
    let h10 = 1.0f64 / 21.0f64 + u2 * h11;
    let h9 = 1.0f64 / 19.0f64 + u2 * h10;
    let h8 = 1.0f64 / 17.0f64 + u2 * h9;
    let h7 = 1.0f64 / 15.0f64 + u2 * h8;
    let h6 = 1.0f64 / 13.0f64 + u2 * h7;
    let h5 = 1.0f64 / 11.0f64 + u2 * h6;
    let h4 = 1.0f64 / 9.0f64 + u2 * h5;
    let h3 = 1.0f64 / 7.0f64 + u2 * h4;
    let h2 = 1.0f64 / 5.0f64 + u2 * h3;
    let h1 = 1.0f64 / 3.0f64 + u2 * h2;
    let p = 1.0f64 + u2 * h1;
    let ln_m = 2.0f64 * u * p;

    // 4. ln(x) = ln(m) + e * LN_2.
    ln_m + e * LN_2
}

/// Exponential polyfill. See module docs for the algorithm. Assumes
/// `|x| < ~700` so `2^k` does not over/underflow the f64 exponent.
#[cube]
pub fn exp_f64(x: f64) -> f64 {
    // 1. Range reduce: x = k * ln 2 + r, with r ∈ [-ln 2 / 2, ln 2 / 2].
    // round(y) = floor(y + 0.5f64) for y ≥ 0, else -floor(-y + 0.5f64).
    let kf = x * INV_LN_2;
    let k_int = if kf >= 0.0f64 {
        (kf + 0.5f64) as i64
    } else {
        -((-kf + 0.5f64) as i64)
    };
    let k_rounded = k_int as f64;
    let r = x - k_rounded * LN_2;

    // 2. exp(r) by Taylor series in Horner form: 1 + r + r²/2 + r³/6 + …
    // 14 terms cover |r| ≤ 0.347f64 (= ln(2)/2) to under 1 ulp.
    // Sequential bindings instead of nested parens, same reason as above.
    let h13 = 1.0f64 / 6_227_020_800.0f64;
    let h12 = 1.0f64 / 479_001_600.0f64 + r * h13;
    let h11 = 1.0f64 / 39_916_800.0f64 + r * h12;
    let h10 = 1.0f64 / 3_628_800.0f64 + r * h11;
    let h9 = 1.0f64 / 362_880.0f64 + r * h10;
    let h8 = 1.0f64 / 40_320.0f64 + r * h9;
    let h7 = 1.0f64 / 5_040.0f64 + r * h8;
    let h6 = 1.0f64 / 720.0f64 + r * h7;
    let h5 = 1.0f64 / 120.0f64 + r * h6;
    let h4 = 1.0f64 / 24.0f64 + r * h5;
    let h3 = 1.0f64 / 6.0f64 + r * h4;
    let h2 = 1.0f64 / 2.0f64 + r * h3;
    let h1 = 1.0f64 + r * h2;
    let exp_r = 1.0f64 + r * h1;

    // 3. 2^k by constructing an f64 with biased exponent k+1023.
    // (k_int + 1023) is in [0, 2046] for any k that doesn't over/underflow,
    // so it fits trivially in u64 for the 52-bit shift.
    let biased = (k_int + 1023i64) as u64;
    let two_to_k_bits = biased << 52u64;
    let two_to_k = f64::reinterpret(two_to_k_bits);

    two_to_k * exp_r
}

/// Cosine polyfill. f64 `.cos()` on cubecl-spirv → Vulkan is broken
/// on both AMD/RADV and NVIDIA: the driver returns ~`1.4f64e-289` for
/// every input. Without a working cos the Watt fission spectrum
/// sampler in the multi-cell transport kernel collapses to a single
/// exponential and downstream physics goes wrong for fissile
/// materials. This polyfill brings cos to ~1 ulp via:
///
/// 1. Range reduction: `x = k * (π/2) + r` with `r ∈ [-π/4, π/4]`.
///    cos(k*π/2 + r) cycles through `cos(r), -sin(r), -cos(r), sin(r)`
///    by `k mod 4`.
/// 2. cos(r) / sin(r) for small r via Taylor series (12 terms hit
///    <1 ulp for |r| ≤ π/4 ≈ 0.785f64).
///
/// Assumes finite `|x| ≲ 1e8`; yamc transport never calls cos on
/// out-of-range inputs.
#[cube]
pub fn cos_f64(x: f64) -> f64 {
    // π/2 and its reciprocal. Use std::f64 constants so the linter
    // doesn't flag the literal.
    let half_pi = std::f64::consts::FRAC_PI_2;
    let two_over_pi = std::f64::consts::FRAC_2_PI;
    // k = round(x / (π/2)), via the standard signed-rounding trick
    // (cubecl's `f64::round` lowering is also suspect -- keep arithmetic
    // we've verified works).
    let kf = x * two_over_pi;
    let k_int = if kf >= 0.0f64 {
        (kf + 0.5f64) as i64
    } else {
        -((-kf + 0.5f64) as i64)
    };
    let r = x - (k_int as f64) * half_pi;

    // cos(r) and sin(r) via Taylor series. For |r| ≤ π/4, 12 terms is
    // well under 1 ulp.
    let r2 = r * r;
    // cos(r) = 1 - r²/2! + r⁴/4! - r⁶/6! + ...
    let c_h6 = -1.0f64 / 87_178_291_200.0f64; // 1/14!
    let c_h5 = 1.0f64 / 479_001_600.0f64 + r2 * c_h6;
    let c_h4 = -1.0f64 / 3_628_800.0f64 + r2 * c_h5;
    let c_h3 = 1.0f64 / 40_320.0f64 + r2 * c_h4;
    let c_h2 = -1.0f64 / 720.0f64 + r2 * c_h3;
    let c_h1 = 1.0f64 / 24.0f64 + r2 * c_h2;
    let c_h0 = -1.0f64 / 2.0f64 + r2 * c_h1;
    let cos_r = 1.0f64 + r2 * c_h0;
    // sin(r) = r - r³/3! + r⁵/5! - r⁷/7! + ...
    let s_h6 = -1.0f64 / 1_307_674_368_000.0f64; // 1/15!
    let s_h5 = 1.0f64 / 6_227_020_800.0f64 + r2 * s_h6;
    let s_h4 = -1.0f64 / 39_916_800.0f64 + r2 * s_h5;
    let s_h3 = 1.0f64 / 362_880.0f64 + r2 * s_h4;
    let s_h2 = -1.0f64 / 5_040.0f64 + r2 * s_h3;
    let s_h1 = 1.0f64 / 120.0f64 + r2 * s_h2;
    let s_h0 = -1.0f64 / 6.0f64 + r2 * s_h1;
    let sin_r = r * (1.0f64 + r2 * s_h0);

    // Reconstruct cos(x) from k mod 4:
    //   k%4 == 0:  cos(r)
    //   k%4 == 1: -sin(r)
    //   k%4 == 2: -cos(r)
    //   k%4 == 3:  sin(r)
    // Note `k_int` can be negative; (k_int % 4 + 4) % 4 normalises.
    let kmod = ((k_int % 4) + 4) % 4;
    if kmod == 0 {
        cos_r
    } else if kmod == 1 {
        -sin_r
    } else if kmod == 2 {
        -cos_r
    } else {
        sin_r
    }
}

/// Sine polyfill -- symmetric to `cos_f64`. Same algorithm; `k mod 4`
/// rotation pattern is `sin, cos, -sin, -cos`.
#[cube]
pub fn sin_f64(x: f64) -> f64 {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let two_over_pi = std::f64::consts::FRAC_2_PI;
    let kf = x * two_over_pi;
    let k_int = if kf >= 0.0f64 {
        (kf + 0.5f64) as i64
    } else {
        -((-kf + 0.5f64) as i64)
    };
    let r = x - (k_int as f64) * half_pi;

    let r2 = r * r;
    let c_h6 = -1.0f64 / 87_178_291_200.0f64;
    let c_h5 = 1.0f64 / 479_001_600.0f64 + r2 * c_h6;
    let c_h4 = -1.0f64 / 3_628_800.0f64 + r2 * c_h5;
    let c_h3 = 1.0f64 / 40_320.0f64 + r2 * c_h4;
    let c_h2 = -1.0f64 / 720.0f64 + r2 * c_h3;
    let c_h1 = 1.0f64 / 24.0f64 + r2 * c_h2;
    let c_h0 = -1.0f64 / 2.0f64 + r2 * c_h1;
    let cos_r = 1.0f64 + r2 * c_h0;
    let s_h6 = -1.0f64 / 1_307_674_368_000.0f64;
    let s_h5 = 1.0f64 / 6_227_020_800.0f64 + r2 * s_h6;
    let s_h4 = -1.0f64 / 39_916_800.0f64 + r2 * s_h5;
    let s_h3 = 1.0f64 / 362_880.0f64 + r2 * s_h4;
    let s_h2 = -1.0f64 / 5_040.0f64 + r2 * s_h3;
    let s_h1 = 1.0f64 / 120.0f64 + r2 * s_h2;
    let s_h0 = -1.0f64 / 6.0f64 + r2 * s_h1;
    let sin_r = r * (1.0f64 + r2 * s_h0);

    let kmod = ((k_int % 4) + 4) % 4;
    if kmod == 0 {
        sin_r
    } else if kmod == 1 {
        cos_r
    } else if kmod == 2 {
        -sin_r
    } else {
        -cos_r
    }
}

/// `atan` for any real `t`, accurate to ~1e-14. Helper for [`atan2_f64`].
///
/// The plain 12-term Taylor for `atan` converges to only ~1e-6 near `|t| = 1`
/// (the series' radius-of-convergence edge), so this uses a two-level range
/// reduction to shrink the argument before the series:
/// 1. `atan(a) = π/2 - atan(1/a)` folds `|a| > 1` down to `[0, 1]`.
/// 2. `atan(y) = π/6 + atan((y·√3 - 1) / (√3 + y))` folds `y > tan(π/12)` down
///    to `|u| <= tan(π/12) ≈ 0.268`.
/// With `u² <= 0.072`, a 13-term Horner series in `u²` lands well under 1e-13.
/// Only the ops validated on cubecl-spirv (+ - * / sqrt, comparisons) appear.
#[cube]
pub fn atan_full(t: f64) -> f64 {
    let neg = t < 0.0f64;
    // |t| without abs(): explicit negate (abs is not validated on spirv).
    let mut a = t;
    if neg {
        a = -t;
    }
    // 1. Fold |a| > 1 into [0, 1] via atan(a) = π/2 - atan(1/a). Compute 1/a
    //    only inside the branch so a = 0 never divides (dead-branch inf would
    //    otherwise be produced by a select).
    let flip = a > 1.0f64;
    let mut y = a;
    if flip {
        y = 1.0f64 / a;
    }
    // 2. Fold y > tan(π/12) toward zero: atan(y) = π/6 + atan((y√3-1)/(√3+y)),
    //    leaving |u| <= tan(π/12) ≈ 0.268.
    let sqrt3 = 3.0f64.sqrt();
    let tan_pi_12 = 2.0f64 - sqrt3;
    let shift = y > tan_pi_12;
    let mut u = y;
    if shift {
        u = (y * sqrt3 - 1.0f64) / (sqrt3 + y);
    }

    // atan(u) = u·(1 - u²/3 + u⁴/5 - u⁶/7 + …). 13 Horner terms in u² give
    // ~1e-14 for |u| <= 0.268. Sequential bindings dodge bracket depth.
    let u2 = u * u;
    let c12 = 1.0f64 / 25.0f64;
    let c11 = -1.0f64 / 23.0f64 + u2 * c12;
    let c10 = 1.0f64 / 21.0f64 + u2 * c11;
    let c9 = -1.0f64 / 19.0f64 + u2 * c10;
    let c8 = 1.0f64 / 17.0f64 + u2 * c9;
    let c7 = -1.0f64 / 15.0f64 + u2 * c8;
    let c6 = 1.0f64 / 13.0f64 + u2 * c7;
    let c5 = -1.0f64 / 11.0f64 + u2 * c6;
    let c4 = 1.0f64 / 9.0f64 + u2 * c5;
    let c3 = -1.0f64 / 7.0f64 + u2 * c4;
    let c2 = 1.0f64 / 5.0f64 + u2 * c3;
    let c1 = -1.0f64 / 3.0f64 + u2 * c2;
    let p = 1.0f64 + u2 * c1;
    let mut r = u * p;

    // Undo the reductions in reverse order.
    if shift {
        r += std::f64::consts::FRAC_PI_6;
    }
    if flip {
        r = std::f64::consts::FRAC_PI_2 - r;
    }
    if neg {
        r = -r;
    }
    r
}

/// `atan2(y, x)` polyfill, ~1e-13 vs libm. cubecl-spirv has no validated
/// `atan`/`atan2` lowering on the AMD/RADV f64 path, and the cylindrical mesh
/// binning needs `φ = atan2(y, x)` on the track-length walk (issue #279). Builds
/// on [`atan_full`] with the standard quadrant assembly from the signs of x, y.
#[cube]
pub fn atan2_f64(y: f64, x: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let half_pi = std::f64::consts::FRAC_PI_2;
    if x > 0.0f64 {
        atan_full(y / x)
    } else if x < 0.0f64 {
        // Left half-plane: shift the principal value by ±π so the result lands
        // in (π/2, π] for y >= 0 and [-π, -π/2) for y < 0.
        if y >= 0.0f64 {
            atan_full(y / x) + pi
        } else {
            atan_full(y / x) - pi
        }
    } else {
        // On the y-axis (x == 0): ±π/2, or 0 for the degenerate (0, 0).
        if y > 0.0f64 {
            half_pi
        } else if y < 0.0f64 {
            -half_pi
        } else {
            0.0f64
        }
    }
}

/// Test kernel: `ln_polyfill_kernel(x) = ln_f64(x)` per element.
#[cube(launch_unchecked)]
fn ln_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = ln_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `exp_polyfill_kernel(x) = exp_f64(x)` per element.
#[cube(launch_unchecked)]
fn exp_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = exp_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `cos_polyfill_kernel(x) = cos_f64(x)` per element.
#[cube(launch_unchecked)]
fn cos_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = cos_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `sin_polyfill_kernel(x) = sin_f64(x)` per element.
#[cube(launch_unchecked)]
fn sin_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = sin_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `atan2_polyfill_kernel(y, x) = atan2_f64(y, x)` per element.
#[cube(launch_unchecked)]
fn atan2_polyfill_kernel(y_in: &[f64], x_in: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= y_in.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = atan2_f64(y_in[ABSOLUTE_POS], x_in[ABSOLUTE_POS]);
}

/// Probe: f64 → i64 → f64 round-trip -- verifies cubecl issue #1317
/// (the original "as i64 produces garbage on AMD RADV") on 0.10f64.0.
#[cube(launch_unchecked)]
fn i64_cast_probe(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = input[ABSOLUTE_POS];
    let k = (x + 0.5f64) as i64;
    output[ABSOLUTE_POS] = k as f64;
}

use crate::{GpuContext, WgpuRuntime};

/// Run the i64-cast probe on the GPU. Returns rounded values
/// `(input[i] + 0.5f64) as i64 as f64`.
pub fn run_i64_cast_probe(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
    unsafe {
        i64_cast_probe::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `ln` over the input array on the GPU.
pub fn run_ln_polyfill(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        ln_polyfill_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `exp` over the input array on the GPU.
pub fn run_exp_polyfill(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        exp_polyfill_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `cos` over the input array on the GPU.
pub fn run_cos_polyfill(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
    unsafe {
        cos_polyfill_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `sin` over the input array on the GPU.
pub fn run_sin_polyfill(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
    unsafe {
        sin_polyfill_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(input_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `atan2` over the paired input arrays on the GPU.
pub fn run_atan2_polyfill(ctx: &GpuContext, y_in: &[f64], x_in: &[f64]) -> Vec<f64> {
    assert_eq!(y_in.len(), x_in.len(), "atan2 inputs must be equal length");
    let client = ctx.client();
    let n = y_in.len();
    let y_handle = client.create_from_slice(bytemuck::cast_slice(y_in));
    let x_handle = client.create_from_slice(bytemuck::cast_slice(x_in));
    let output_handle = client.empty(std::mem::size_of_val(y_in));
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
    unsafe {
        atan2_polyfill_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(y_handle, n),
            BufferArg::from_raw_parts(x_handle, n),
            BufferArg::from_raw_parts(output_handle.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
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

    /// `ln_f64` polyfill must agree with libm `f64::ln` to within a
    /// reasonable tolerance. Targeting 16 ulps as a generous first
    /// bar; tighten if the algorithm is doing better than expected.
    /// Inputs span yamc-relevant magnitudes; avoids 0 and inf.
    #[test]
    fn gpu_ln_polyfill_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let input: Vec<f64> = vec![
            1e-9,
            1e-6,
            1e-3,
            0.1f64,
            0.5f64,
            std::f64::consts::E.recip(),
            1.0f64,
            2.0f64,
            std::f64::consts::E,
            std::f64::consts::PI,
            10.0f64,
            100.0f64,
            1000.0f64,
            1e6,
            1e9,
        ];

        let gpu_out = run_ln_polyfill(&ctx, &input);
        let cpu_out: Vec<f64> = input.iter().map(|x| x.ln()).collect();
        let max_ulps = worst_ulps("ln_polyfill", &gpu_out, &cpu_out, &input);
        assert!(
            max_ulps <= 16,
            "ln_f64 polyfill drifts > 16 ulps from libm (max {max_ulps})"
        );
    }

    /// `exp_f64` polyfill must agree with libm `f64::exp`. Same
    /// tolerance bar. Inputs avoid overflow (|x| < 700).
    #[test]
    fn gpu_exp_polyfill_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let input: Vec<f64> = vec![
            -50.0f64,
            -10.0f64,
            -1.0f64,
            -0.5f64,
            -0.1f64,
            -1e-3,
            0.0f64,
            1e-3,
            0.1f64,
            0.5f64,
            1.0f64,
            std::f64::consts::E,
            std::f64::consts::PI,
            10.0f64,
            50.0f64,
        ];

        let gpu_out = run_exp_polyfill(&ctx, &input);
        let cpu_out: Vec<f64> = input.iter().map(|x| x.exp()).collect();
        let max_ulps = worst_ulps("exp_polyfill", &gpu_out, &cpu_out, &input);
        assert!(
            max_ulps <= 16,
            "exp_f64 polyfill drifts > 16 ulps from libm (max {max_ulps})"
        );
    }

    /// Trig polyfills can land on `±0` where the value is
    /// mathematically zero (cos(π/2), sin(0)) and libm lands on a
    /// tiny non-zero due to π's f64 rounding. The strict `ulps_diff`
    /// flags sign-of-zero mismatches as u64::MAX; for sin/cos we
    /// treat both sides as "near zero" → equal.
    fn cmp_trig(label: &str, gpu: &[f64], cpu: &[f64], input: &[f64]) -> f64 {
        let mut max_abs = 0.0f64;
        let mut worst_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_abs {
                max_abs = d;
                worst_idx = i;
            }
        }
        println!(
            "{label}: max |Δ| = {max_abs:.3e} at input[{worst_idx}] = {} (GPU = {}, CPU = {})",
            input[worst_idx], gpu[worst_idx], cpu[worst_idx]
        );
        max_abs
    }

    /// `cos_f64` polyfill regression. Drives the Watt fission
    /// spectrum sampler and several inelastic continuum samplers
    /// (Maxwell, Watt, Evaporation, free-gas thermal). Native `.cos()`
    /// on cubecl-spirv → Vulkan is broken on both AMD/RADV and NVIDIA
    /// (returns ~1.4f64e-289 garbage), so the kernel uses this polyfill.
    /// Without it the actinide verification sphere shows 3-10×
    /// reaction-rate inflation on every nontrivial-σ inelastic MT.
    #[test]
    fn gpu_cos_polyfill_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let pi = std::f64::consts::PI;
        let input: Vec<f64> = vec![
            -2.0f64 * pi,
            -pi,
            -pi / 2.0f64,
            -pi / 3.0f64,
            -pi / 4.0f64,
            -0.5f64,
            -1e-3,
            0.0f64,
            1e-3,
            0.5f64,
            pi / 4.0f64,
            pi / 3.0f64,
            pi / 2.0f64,
            pi,
            1.5f64 * pi,
            2.0f64 * pi,
            10.0f64,
            100.0f64,
        ];
        let gpu_out = run_cos_polyfill(&ctx, &input);
        let cpu_out: Vec<f64> = input.iter().map(|x| x.cos()).collect();
        let max_abs = cmp_trig("cos_polyfill", &gpu_out, &cpu_out, &input);
        // 1e-13 absolute is well under MC noise; trig polyfills with
        // 12-term Taylor + range reduction land at ~1 ulp ≈ 1e-16
        // away from the libm value for inputs not near a zero.
        assert!(
            max_abs < 1e-13,
            "cos_f64 polyfill drifts > 1e-13 from libm (max |Δ| = {max_abs:.3e})"
        );
    }

    /// `sin_f64` polyfill regression. Same motivation as cos.
    #[test]
    fn gpu_sin_polyfill_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let pi = std::f64::consts::PI;
        let input: Vec<f64> = vec![
            -2.0f64 * pi,
            -pi,
            -pi / 2.0f64,
            -pi / 3.0f64,
            -pi / 4.0f64,
            -0.5f64,
            -1e-3,
            0.0f64,
            1e-3,
            0.5f64,
            pi / 4.0f64,
            pi / 3.0f64,
            pi / 2.0f64,
            pi,
            1.5f64 * pi,
            2.0f64 * pi,
            10.0f64,
            100.0f64,
        ];
        let gpu_out = run_sin_polyfill(&ctx, &input);
        let cpu_out: Vec<f64> = input.iter().map(|x| x.sin()).collect();
        let max_abs = cmp_trig("sin_polyfill", &gpu_out, &cpu_out, &input);
        assert!(
            max_abs < 1e-13,
            "sin_f64 polyfill drifts > 1e-13 from libm (max |Δ| = {max_abs:.3e})"
        );
    }

    /// `atan2_f64` polyfill regression. Drives the azimuthal binning of the
    /// cylindrical mesh DDA (`φ = atan2(y, x)`; issue #279). Native `.atan2()`
    /// has no validated cubecl-spirv lowering on the AMD/RADV f64 path, so the
    /// kernel uses this polyfill. Inputs span all four quadrants and both axes;
    /// diffs are wrapped into `[-π, π]` so a legitimate result that lands on the
    /// far side of the `±π` seam is not mistaken for a large error.
    #[test]
    fn gpu_atan2_polyfill_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let pi = std::f64::consts::PI;
        // Sample angles around the circle at three radii, plus the axes. Avoid
        // the exact (y = 0, x < 0) sign-of-zero seam (libm returns -π for -0.0).
        let angles: Vec<f64> = (0..24)
            .map(|k| -pi + (k as f64 + 0.37) * (2.0 * pi / 24.0))
            .collect();
        let mut y_in = Vec::new();
        let mut x_in = Vec::new();
        for &r in &[0.3f64, 1.0, 7.5] {
            for &th in &angles {
                y_in.push(r * th.sin());
                x_in.push(r * th.cos());
            }
        }
        // Explicit on-axis probes.
        for &(y, x) in &[
            (1.0f64, 0.0f64),
            (-1.0, 0.0),
            (0.0, 1.0),
            (1e-9, -1.0),
            (-1e-9, -1.0),
            (5.0, 5.0),
            (-5.0, 5.0),
            (5.0, -5.0),
            (-5.0, -5.0),
        ] {
            y_in.push(y);
            x_in.push(x);
        }

        let gpu_out = run_atan2_polyfill(&ctx, &y_in, &x_in);
        let cpu_out: Vec<f64> = y_in.iter().zip(&x_in).map(|(&y, &x)| y.atan2(x)).collect();

        let mut max_abs = 0.0f64;
        let mut worst = 0usize;
        for (i, (&g, &c)) in gpu_out.iter().zip(&cpu_out).enumerate() {
            let mut d = g - c;
            if d > pi {
                d -= 2.0 * pi;
            }
            if d < -pi {
                d += 2.0 * pi;
            }
            if d.abs() > max_abs {
                max_abs = d.abs();
                worst = i;
            }
        }
        println!(
            "atan2_polyfill: max |Δ| = {max_abs:.3e} at (y={}, x={}) GPU={} CPU={}",
            y_in[worst], x_in[worst], gpu_out[worst], cpu_out[worst]
        );
        assert!(
            max_abs < 1e-12,
            "atan2_f64 polyfill drifts > 1e-12 from libm (max |Δ| = {max_abs:.3e})"
        );
    }

    /// Verify cubecl issue #1317 on 0.10f64.0: `(f64_expr) as i64` round-
    /// trip produces correct integer values. Pre-0.10f64.0 produced
    /// garbage on AMD RADV.
    #[test]
    fn gpu_i64_cast_round_trip_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        let input: Vec<f64> = vec![
            -1000.5f64, -100.5f64, -1.5f64, -0.5f64, 0.0f64, 0.5f64, 1.5f64, 100.5f64, 1000.5f64,
        ];
        let gpu_out = run_i64_cast_probe(&ctx, &input);
        let cpu_out: Vec<f64> = input
            .iter()
            .map(|&x| ((x + 0.5f64) as i64) as f64)
            .collect();
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(
                g, c,
                "i64 cast diverges at idx={i}: input={}, gpu={g}, cpu={c}",
                input[i]
            );
        }
    }
}
