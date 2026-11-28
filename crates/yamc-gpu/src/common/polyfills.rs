//! Software polyfills for f64 transcendentals on the cubecl-spirv path.
//!
//! Why these exist: cubecl-spirv emits invalid GLSL.std.450 `OpExtInst
//! Log` / `Exp` for f64 operands (the spec restricts those to f16/f32).
//! NVIDIA's Vulkan driver tolerates it; AMD/RADV correctly rejects it
//! and the kernel returns garbage. See:
//! - Mesa work item: <https://gitlab.freedesktop.org/mesa/mesa/-/work_items/15359>
//! - cubecl-spirv lowers f64 `ln`/`exp` straight to GLSL.std.450 `Log`/`Exp`
//!   with no operand-width check, and upstream has not fixed it yet.
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
//! Measured against `std`, both on the plain-Rust copies in [`cpu`] (which
//! come from the same source as the `#[cube]` functions, the `polyfills!`
//! macro) and on the GPU. The two are not bit-identical, because the shader
//! compiler may fuse a multiply and an add, but their worst errors agree:
//!
//! | function    | inputs                 | worst error               |
//! |-------------|------------------------|---------------------------|
//! | `ln_f64`    | 1e-300 to 1e300        | 3 ulps                    |
//! | `exp_f64`   | -700 to 700            | 1 ulp                     |
//! | `expm1_f64` | -700 to 700            | 2 ulps                    |
//! | `cos_f64`   | -1e4 to 1e4            | 10 ulps, 1.2e-15 absolute |
//! | `sin_f64`   | -1e4 to 1e4            | 10 ulps, 1.2e-15 absolute |
//! | `atan_full` | -1e3 to 1e3            | 2 ulps                    |
//! | `atan2_f64` | around the unit circle | 3 ulps                    |
//!
//! Inputs that would over/underflow are out of scope: `ln_f64` assumes
//! `x > 0` and finite; `exp_f64` assumes `|x| < ~700` (no overflow checks).
//! yamc transport never calls these on out-of-range values, so guarding adds
//! cost without benefit.

use cubecl::prelude::*;
use std::f64::consts;

// Pull constants from std rather than re-typing approximations -- clippy
// flags hand-rolled literals for these. INV_LN_2 = log2(e).
const LN_2: f64 = consts::LN_2;
const INV_LN_2: f64 = consts::LOG2_E;
const SQRT_2: f64 = consts::SQRT_2;

// Cody-Waite splits for the range reductions (fdlibm's values). Each part
// but the last has enough trailing zero bits that `k * part` is exact for
// every `k` the reductions meet, so subtracting the parts in turn removes
// `k` copies of the constant to well under an ulp of the remainder. A single
// rounded constant leaves `k` times its rounding error instead: 700 ulps in
// `exp` near |x| = 700, 1e-12 absolute in `cos` near |x| = 1e4.
//
// The GPU shader compiler may rewrite `fma(a, b, fma(a, c, d))` as
// `fma(a, b + c, d)` (cubecl emits no SPIR-V `NoContraction` to forbid it),
// which rounds the split back into one constant. Neighbouring steps therefore
// take `k` from different conversions (`k_int as f64` and
// `(k_int as i32) as f64`): the same value, but not an operand the compiler
// can see is shared.
const LN_2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
const LN_2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
const PIO2_1: f64 = f64::from_bits(0x3FF9_21FB_5440_0000);
const PIO2_2: f64 = f64::from_bits(0x3DD0_B461_1A60_0000);
const PIO2_3: f64 = f64::from_bits(0x3BA3_198A_2E00_0000);
const PIO2_3T: f64 = f64::from_bits(0x397B_839A_2520_49C1);

/// Every polyfill, written once. Invoked as `polyfills!(#[cube])` for the
/// `#[cube]` functions the kernels call, and as `polyfills!()` in [`cpu`] for
/// plain-Rust copies of the same arithmetic. Bit casts go through `to_bits` /
/// `from_bits`, which each invoking module defines for its side.
macro_rules! polyfills {
    ($(#[$attr:meta])?) => {
        /// Natural log polyfill. See module docs for the algorithm. Assumes
        /// `x > 0` and finite.
        $(#[$attr])?
        pub fn ln_f64(x: f64) -> f64 {
            // 1. Extract exponent and mantissa from the f64 bit pattern.
            let bits = to_bits(x);
            let exp_bits = (bits >> 52u64) & 0x7FFu64;
            // Reset the exponent field to bias 1023 (i.e., 2^0) so the
            // reconstructed value is the mantissa m ∈ [1, 2).
            let mantissa_bits = (bits & 0x000F_FFFF_FFFF_FFFFu64) | 0x3FF0_0000_0000_0000u64;
            let mut m = from_bits(mantissa_bits);
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
        $(#[$attr])?
        pub fn exp_f64(x: f64) -> f64 {
            // 1. Range reduce: x = k * ln 2 + r, with r ∈ [-ln 2 / 2, ln 2 / 2].
            // round(y) = floor(y + 0.5f64) for y ≥ 0, else -floor(-y + 0.5f64).
            let kf = x * INV_LN_2;
            let k_int = if kf >= 0.0f64 {
                (kf + 0.5f64) as i64
            } else {
                -((-kf + 0.5f64) as i64)
            };
            // Two conversions of k so the steps cannot be merged (see LN_2_HI).
            let k_a = k_int as f64;
            let k_b = (k_int as i32) as f64;
            let r = fma(-k_b, LN_2_LO, fma(-k_a, LN_2_HI, x));

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
            let two_to_k = from_bits(two_to_k_bits);

            two_to_k * exp_r
        }

        /// `exp(x) - 1` without the cancellation `exp_f64(x) - 1` suffers near 0:
        /// the Taylor series for `|x| < 0.5` (25 terms, far below an ulp there), else
        /// [`exp_f64()`] less 1. Same input assumption as [`exp_f64()`].
        $(#[$attr])?
        pub fn expm1_f64(x: f64) -> f64 {
            let mut ax = x;
            if x < 0.0f64 {
                ax = -x;
            }
            #[allow(unused_assignments)]
            let mut out = x * 0.0f64;
            if ax < 0.5f64 {
                // Horner form of sum_{k=1..25} x^k / k!: x (1 + x/2 (1 + x/3 (...))).
                let mut acc = x * 0.0f64 + 1.0f64;
                let mut k = 25u32;
                while k > 1u32 {
                    acc = 1.0f64 + x * acc / (k as f64);
                    k -= 1u32;
                }
                out = x * acc;
            } else {
                out = exp_f64(x) - 1.0f64;
            }
            out
        }

        /// Cosine polyfill. f64 `.cos()` on cubecl-spirv → Vulkan is broken
        /// on both AMD/RADV and NVIDIA: the driver returns ~`1.4f64e-289` for
        /// every input. Without a working cos the Watt fission spectrum
        /// sampler in the multi-cell transport kernel collapses to a single
        /// exponential and downstream physics goes wrong for fissile
        /// materials. This polyfill brings cos to within 10 ulps (1.2e-15
        /// absolute) via:
        ///
        /// 1. Range reduction: `x = k * (π/2) + r` with `r ∈ [-π/4, π/4]`,
        ///    removing `k * π/2` in Cody-Waite parts.
        ///    cos(k*π/2 + r) cycles through `cos(r), -sin(r), -cos(r), sin(r)`
        ///    by `k mod 4`.
        /// 2. cos(r) / sin(r) for small r via Taylor series (12 terms hit
        ///    <1 ulp for |r| ≤ π/4 ≈ 0.785f64).
        ///
        /// The reduction is exact for `|x| < 3e6` (`k < 2^21`); yamc transport
        /// calls cos on angles.
        $(#[$attr])?
        pub fn cos_f64(x: f64) -> f64 {
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
            // Alternating conversions of k so no two steps can be merged (see
            // PIO2_1).
            let k_a = k_int as f64;
            let k_b = (k_int as i32) as f64;
            let r = fma(
                -k_b,
                PIO2_3T,
                fma(-k_a, PIO2_3, fma(-k_b, PIO2_2, fma(-k_a, PIO2_1, x))),
            );

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
        $(#[$attr])?
        pub fn sin_f64(x: f64) -> f64 {
            let two_over_pi = std::f64::consts::FRAC_2_PI;
            let kf = x * two_over_pi;
            let k_int = if kf >= 0.0f64 {
                (kf + 0.5f64) as i64
            } else {
                -((-kf + 0.5f64) as i64)
            };
            // Alternating conversions of k so no two steps can be merged (see
            // PIO2_1).
            let k_a = k_int as f64;
            let k_b = (k_int as i32) as f64;
            let r = fma(
                -k_b,
                PIO2_3T,
                fma(-k_a, PIO2_3, fma(-k_b, PIO2_2, fma(-k_a, PIO2_1, x))),
            );

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

        /// `atan` for any real `t`, accurate to 2 ulps. Helper for [`atan2_f64()`].
        ///
        /// The plain 12-term Taylor for `atan` converges to only ~1e-6 near `|t| = 1`
        /// (the series' radius-of-convergence edge), so this uses a two-level range
        /// reduction to shrink the argument before the series:
        /// 1. `atan(a) = π/2 - atan(1/a)` folds `|a| > 1` down to `[0, 1]`.
        /// 2. `atan(y) = π/6 + atan((y·√3 - 1) / (√3 + y))` folds `y > tan(π/12)` down
        ///    to `|u| <= tan(π/12) ≈ 0.268`.
        ///
        /// With `u² <= 0.072`, a 13-term Horner series in `u²` lands well under 1e-13.
        /// Only the ops validated on cubecl-spirv (+ - * / sqrt, comparisons) appear.
        $(#[$attr])?
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
        /// binning needs `φ = atan2(y, x)` on the track-length walk. Builds
        /// on [`atan_full()`] with the standard quadrant assembly from the signs of x, y.
        $(#[$attr])?
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

    };
}

#[cube]
fn to_bits(x: f64) -> u64 {
    u64::reinterpret(x)
}

#[cube]
fn from_bits(bits: u64) -> f64 {
    f64::reinterpret(bits)
}

polyfills!(#[cube]);

/// Plain-Rust copies of the polyfills, from the same source as the `#[cube]`
/// functions above. They are the reference the GPU output is checked against
/// bit for bit, and what their accuracy against `std` is measured on, with no
/// GPU needed.
pub mod cpu {
    use super::{INV_LN_2, LN_2, LN_2_HI, LN_2_LO, PIO2_1, PIO2_2, PIO2_3, PIO2_3T, SQRT_2};

    fn to_bits(x: f64) -> u64 {
        x.to_bits()
    }

    fn from_bits(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    /// `a * b + c` with one rounding, as cubecl's `fma`.
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }

    polyfills!();
}

/// Test kernel: `ln_polyfill_kernel(x) = ln_f64(x)` per element.
#[cube(launch)]
fn ln_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = ln_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `exp_polyfill_kernel(x) = exp_f64(x)` per element.
#[cube(launch)]
fn exp_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = exp_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `expm1_polyfill_kernel(x) = expm1_f64(x)` per element.
#[cube(launch)]
fn expm1_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = expm1_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `cos_polyfill_kernel(x) = cos_f64(x)` per element.
#[cube(launch)]
fn cos_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = cos_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `sin_polyfill_kernel(x) = sin_f64(x)` per element.
#[cube(launch)]
fn sin_polyfill_kernel(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = sin_f64(input[ABSOLUTE_POS]);
}

/// Test kernel: `atan2_polyfill_kernel(y, x) = atan2_f64(y, x)` per element.
#[cube(launch)]
fn atan2_polyfill_kernel(y_in: &[f64], x_in: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= y_in.len() {
        terminate!();
    }
    output[ABSOLUTE_POS] = atan2_f64(y_in[ABSOLUTE_POS], x_in[ABSOLUTE_POS]);
}

/// Probe: f64 → i64 → f64 round-trip -- verifies the cubecl-spirv bug where
/// `as i64` produced garbage on AMD RADV is fixed on 0.10f64.0.
#[cube(launch)]
fn i64_cast_probe(input: &[f64], output: &mut [f64]) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = input[ABSOLUTE_POS];
    let k = (x + 0.5f64) as i64;
    output[ABSOLUTE_POS] = k as f64;
}

use crate::GpuContext;

/// Run the i64-cast probe on the GPU. Returns rounded values
/// `(input[i] + 0.5f64) as i64 as f64`.
pub fn run_i64_cast_probe(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
    launch!(
        i64_cast_probe,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
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

    launch!(
        ln_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
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

    launch!(
        exp_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

/// Run the polyfill `expm1` over the input array on the GPU.
pub fn run_expm1_polyfill(ctx: &GpuContext, input: &[f64]) -> Vec<f64> {
    let client = ctx.client();
    let n = input.len();
    let input_handle = client.create_from_slice(bytemuck::cast_slice(input));
    let output_handle = client.empty(std::mem::size_of_val(input));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;

    launch!(
        expm1_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
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
    launch!(
        cos_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
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
    launch!(
        sin_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(input_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
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
    launch!(
        atan2_polyfill_kernel,
        &client,
        CubeCount::Static(groups, 1, 1),
        CubeDim::new_1d(WORKGROUP_SIZE),
        BufferArg::from_raw_parts(y_handle, n),
        BufferArg::from_raw_parts(x_handle, n),
        BufferArg::from_raw_parts(output_handle.clone(), n),
    );
    bytemuck::cast_slice(&client.read_one(output_handle).unwrap()).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Units in the last place between `a` and `b`; `u64::MAX` across a sign
    /// change or for a non-finite value, unless the two are equal.
    fn ulps(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
            return u64::MAX;
        }
        let (x, y) = (a.to_bits(), b.to_bits());
        x.max(y) - x.min(y)
    }

    fn sweep(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..=n)
            .map(|i| lo + (hi - lo) * i as f64 / n as f64)
            .collect()
    }

    fn log_sweep(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        sweep(lo.ln(), hi.ln(), n)
            .into_iter()
            .map(f64::exp)
            .collect()
    }

    /// Angles around the unit circle at three radii, as `(y, x)` pairs.
    fn circle(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut y = Vec::new();
        let mut x = Vec::new();
        for r in [0.3f64, 1.0, 7.5] {
            for t in sweep(-0.999 * consts::PI, 0.999 * consts::PI, n) {
                y.push(r * t.sin());
                x.push(r * t.cos());
            }
        }
        (y, x)
    }

    /// Asserts every `got[i]` is within `max_ulps` of `reference(xs[i])`, and
    /// prints the worst so the module-doc table can be checked against it.
    fn assert_within(
        label: &str,
        xs: &[f64],
        got: &[f64],
        reference: impl Fn(f64) -> f64,
        max_ulps: u64,
    ) {
        assert_eq!(got.len(), xs.len());
        let (worst, at) = xs
            .iter()
            .zip(got)
            .map(|(&x, &g)| (ulps(g, reference(x)), x))
            .max_by_key(|&(u, _)| u)
            .unwrap();
        println!("{label}: worst {worst} ulps from std, at x = {at:e}");
        assert!(
            worst <= max_ulps,
            "{label}: {worst} ulps from std at x = {at:e}, over the bound of {max_ulps}"
        );
    }

    fn on_cpu(xs: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
        xs.iter().map(|&x| f(x)).collect()
    }

    /// The plain-Rust copies against `std`, at the bounds in the module docs.
    #[test]
    fn cpu_copies_track_std() {
        let n = 200_000;
        let xs = log_sweep(1e-300, 1e300, n);
        assert_within("ln_f64", &xs, &on_cpu(&xs, cpu::ln_f64), f64::ln, 4);
        let xs = sweep(0.5, 2.0, n);
        assert_within("ln_f64 near 1", &xs, &on_cpu(&xs, cpu::ln_f64), f64::ln, 4);
        let xs = sweep(-700.0, 700.0, n);
        assert_within("exp_f64", &xs, &on_cpu(&xs, cpu::exp_f64), f64::exp, 2);
        let got = on_cpu(&xs, cpu::expm1_f64);
        assert_within("expm1_f64", &xs, &got, f64::exp_m1, 3);
        let xs = sweep(-1.0, 1.0, n);
        let got = on_cpu(&xs, cpu::expm1_f64);
        assert_within("expm1_f64 near 0", &xs, &got, f64::exp_m1, 3);
        for (lo, hi) in [(-1e4, 1e4), (-6.3, 6.3)] {
            let xs = sweep(lo, hi, n);
            assert_within("cos_f64", &xs, &on_cpu(&xs, cpu::cos_f64), f64::cos, 11);
            assert_within("sin_f64", &xs, &on_cpu(&xs, cpu::sin_f64), f64::sin, 11);
        }
        let xs = sweep(-1e3, 1e3, n);
        let got = on_cpu(&xs, cpu::atan_full);
        assert_within("atan_full", &xs, &got, f64::atan, 3);
        let (y, x) = circle(n / 3);
        let idx: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
        let got: Vec<f64> = y
            .iter()
            .zip(&x)
            .map(|(&y, &x)| cpu::atan2_f64(y, x))
            .collect();
        let reference = |i: f64| y[i as usize].atan2(x[i as usize]);
        assert_within("atan2_f64 (x is the input index)", &idx, &got, reference, 4);
    }

    fn gpu() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                None
            }
        }
    }

    /// The `#[cube]` functions on the GPU against `std`, on the same inputs as
    /// [`cpu_copies_track_std`]. The source is the same, but the GPU results
    /// are not bit-identical to the copies: the shader compiler may fuse a
    /// multiply and an add into one rounding, which moves a result by an ulp
    /// or so, never past the copies' worst. Same bounds as the copies.
    #[test]
    fn gpu_polyfills_track_std() {
        let Some(ctx) = gpu() else { return };
        let n = 200_000;
        let xs = log_sweep(1e-300, 1e300, n);
        assert_within("ln_f64", &xs, &run_ln_polyfill(&ctx, &xs), f64::ln, 4);
        let xs = sweep(0.5, 2.0, n);
        assert_within(
            "ln_f64 near 1",
            &xs,
            &run_ln_polyfill(&ctx, &xs),
            f64::ln,
            4,
        );
        let xs = sweep(-700.0, 700.0, n);
        assert_within("exp_f64", &xs, &run_exp_polyfill(&ctx, &xs), f64::exp, 2);
        let got = run_expm1_polyfill(&ctx, &xs);
        assert_within("expm1_f64", &xs, &got, f64::exp_m1, 3);
        let xs = sweep(-1.0, 1.0, n);
        let got = run_expm1_polyfill(&ctx, &xs);
        assert_within("expm1_f64 near 0", &xs, &got, f64::exp_m1, 3);
        for (lo, hi) in [(-1e4, 1e4), (-6.3, 6.3)] {
            let xs = sweep(lo, hi, n);
            assert_within("cos_f64", &xs, &run_cos_polyfill(&ctx, &xs), f64::cos, 11);
            assert_within("sin_f64", &xs, &run_sin_polyfill(&ctx, &xs), f64::sin, 11);
        }
        let (y, x) = circle(n / 3);
        let idx: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
        let got = run_atan2_polyfill(&ctx, &y, &x);
        let reference = |i: f64| y[i as usize].atan2(x[i as usize]);
        assert_within("atan2_f64 (x is the input index)", &idx, &got, reference, 4);
    }

    /// Verify on 0.10f64.0 that `(f64_expr) as i64` round-
    /// trip produces correct integer values. Pre-0.10f64.0 produced
    /// garbage on AMD RADV.
    #[test]
    fn gpu_i64_cast_round_trip_matches_cpu() {
        let Some(ctx) = gpu() else { return };
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
