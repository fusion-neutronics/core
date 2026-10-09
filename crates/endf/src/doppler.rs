//! Free-gas Doppler broadening of a pointwise cross section, and its adjoint.
//!
//! # The kernel
//!
//! A target nucleus of mass ratio `A` (the ENDF `AWR`) in thermal motion at
//! temperature `T` turns a 0 K cross section `σ₀` into
//!
//! ```text
//! σ_T(y) = 1/(y²√π) ∫₀^∞ x² σ₀(x) [exp(-(x - y)²) - exp(-(x + y)²)] dx
//! ```
//!
//! in the velocity-like variables `x = √(αE')`, `y = √(αE)`, with
//! `α = A / (kT)` per eV. This is the free-gas kernel of SIGMA1 (D. E. Cullen
//! and C. R. Weisbin, Nucl. Sci. Eng. 60, 199, 1976) that NJOY's BROADR
//! evaluates, and [`broaden`] follows BROADR in everything that defines the
//! operator:
//!
//! - `σ₀` is linear in energy between its points, so `x² σ₀` is a quartic in
//!   `x` on each interval and the integral against the Gaussian is exact,
//!   through the `F_n(a) = ∫_a^∞ zⁿ exp(-z²) dz/√π` functions (an `erfc` and
//!   an exponential) and their differences `H_n(a, b) = F_n(a) - F_n(b)`.
//! - Below the lowest point `σ₀` continues as `1/v` to zero energy, and above
//!   the highest it continues as a constant.
//! - The second exponential, the contribution of nuclei moving faster than
//!   the neutron and away from it, is kept wherever it is not negligible,
//!   which is only at energies of a few `kT/A` and below.
//!
//! Where a difference of `F_n` would lose its significance, an interval
//! narrower than [`SHORT`] Doppler widths, BROADR switches to a Taylor series
//! of the defining integral (its `hnabb`); this module integrates such an
//! interval with Gauss-Legendre instead, which is exact to rounding for a
//! quartic times a Gaussian over that width and needs no separate expansion of
//! the second exponential.
//!
//! The one place this module departs from BROADR is the kernel's reach. BROADR
//! stops summing intervals once the far end of one is [`BROADR_REACH`] (four)
//! Doppler widths from `y`, and drops the second exponential above `y = 4`.
//! The kernel there is `exp(-16) ≈ 1e-7` of its peak, but the cross section
//! there need not be small next to the one at `y`: in the minimum beside a
//! strong resonance it can be ten thousand times larger, and BROADR's result
//! is then low by up to about `1e-4` relative (`7e-5` in the elastic minima
//! of W186 below 1 keV at 293.6 K). [`broaden`] reaches [`WINDOW`], seven
//! widths, where what is neglected is below `exp(-49) ≈ 5e-22` of the peak, so
//! what it computes is the exact operator to double precision.
//! [`broaden_within`] takes the reach as an argument, and at
//! [`BROADR_REACH`] reproduces BROADR to the 7 significant figures a PENDF
//! file carries.
//!
//! # The adjoint
//!
//! A reaction rate against a flux density `ψ` is `R = ∫ σ_T(E) ψ(E) dE`.
//! Broadening is linear, so the same rate is `R = ∫ σ₀(E') w(E') dE'` with the
//! broadened weight
//!
//! ```text
//! w(x) = x/√π ∫₀^∞ ψ(y)/y [exp(-(x - y)²) - exp(-(x + y)²)] dy
//! ```
//!
//! which is what [`broadened_weight`] computes. The identity is exact for every
//! `σ₀`, so a quantity that needs the broadened rate of many perturbed 0 K
//! cross sections (sampled resonance parameters, say) computes `w` once and
//! then needs only a 0 K reconstruction and an integral per sample.
//!
//! `w` has no closed form when `ψ` does not vanish (the `1/y` stops it), so it
//! is integrated numerically and stored as a [`BroadenedWeight`]: a polynomial
//! of degree 15 in energy on each of a set of panels, fitted at Chebyshev
//! points and refined until its trailing coefficients are below `1e-14` of the
//! flux density nearby (or `16 ε y` of it, the rounding floor of any function
//! of energy that changes across a Doppler width, if that is larger: `3e-11`
//! at 10 keV in tungsten at room temperature). [`BroadenedWeight::integrate`]
//! then integrates a piecewise-linear `σ₀` against it with Gauss-Legendre
//! rules exact for that polynomial degree, so the only error left is the
//! fit's.
//!
//! `w` is not `ψ`, even far from any group edge: for a constant `ψ` it is
//! `ψ · 2x D(x)`, with `D` Dawson's integral, which is `ψ (1 + 1/(2x²) + ...)`.
//! The correction is the familiar rise of a broadened constant cross section at
//! low energy, `kT/(2AE)` relative, below `1e-6` above about 200 eV in tungsten
//! at room temperature but not zero, so it is computed rather than assumed
//! away. Outside `WINDOW` widths of the flux's support `w` is zero.
//!
//! # BROADR's upper limit
//!
//! BROADR broadens only below `thnmax`, normally the top of the resolved range,
//! and copies the 0 K data above it. [`broadened_weight`] takes that limit as
//! `broaden_below`: the flux below it is carried through the kernel, the flux
//! above it is weighted unbroadened, so the identity holds for the cross
//! section a PENDF file actually holds.
//!
//! # What the identity does not cover
//!
//! `w` is the adjoint of the exact kernel. A library's broadened cross section
//! differs from that in two ways the identity cannot see: BROADR's shorter
//! reach (above, up to about `1e-4` relative in deep minima), and the
//! linear interpolation BROADR fits to its result, which holds it only to the
//! tolerance it was run with (typically `1e-3`). A rate taken against a PENDF
//! table carries both; one taken through `w` carries neither.

use std::f64::consts::PI;

use crate::data::K_BOLTZMANN;
use crate::error::{Error, Result};

/// How far the kernel reaches, in Doppler widths (units of `x`).
///
/// Beyond it the kernel is below `exp(-49) ≈ 5e-22` of its peak.
pub const WINDOW: f64 = 7.0;

/// The reach of NJOY BROADR's kernel, in Doppler widths: `atop` in `bsigma`.
pub const BROADR_REACH: f64 = 4.0;

/// The width, in Doppler widths, below which an interval of the 0 K cross
/// section is integrated by Gauss-Legendre rather than by `F_n` differences.
pub const SHORT: f64 = 0.5;

/// Gauss-Legendre points for an interval narrower than [`SHORT`].
const SHORT_POINTS: usize = 12;

/// Gauss-Legendre points per sub-interval of the integral that defines `w`.
const WEIGHT_POINTS: usize = 12;

/// The widest sub-interval, in Doppler widths, of the integral that defines
/// `w`.
const WEIGHT_STEP: f64 = 0.5;

/// Chebyshev points, so polynomial degree plus one, on each panel of `w`.
const PANEL_POINTS: usize = 16;

/// A panel is accepted when its last three Chebyshev coefficients are below
/// this fraction of the flux density within [`WINDOW`] of it, or below the
/// rounding floor of `w` as a function of energy, whichever is larger.
///
/// That floor is intrinsic: `w` changes by about `ψ` across one Doppler width,
/// `δy = 1`, and an energy known to one ulp fixes `y` to `ε y`, so no function
/// of energy can be known better than `ε y ψ`.
const PANEL_TOLERANCE: f64 = 1e-14;

/// The inverse square root of pi.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// `α = A / (kT)`, per eV, so that `x = √(αE)`.
fn alpha(awr: f64, temperature: f64) -> f64 {
    awr / (K_BOLTZMANN * temperature)
}

fn invalid(what: String) -> Error {
    Error::InvalidArgument { what }
}

fn check_awr_temperature(awr: f64, temperature: f64) -> Result<()> {
    if !(awr.is_finite() && awr > 0.0) {
        return Err(invalid(format!(
            "the mass ratio must be positive and finite, got {awr}"
        )));
    }
    if !(temperature.is_finite() && temperature >= 0.0) {
        return Err(invalid(format!(
            "the temperature must be non-negative and finite, got {temperature} K"
        )));
    }
    Ok(())
}

/// Check a tabulated function: matching lengths, positive non-decreasing
/// energies (a repeated energy is a step), finite values.
fn check_table(what: &str, energy: &[f64], value: &[f64]) -> Result<()> {
    if energy.len() != value.len() {
        return Err(invalid(format!(
            "{what}: {} energies but {} values",
            energy.len(),
            value.len()
        )));
    }
    if energy.is_empty() {
        return Err(invalid(format!("{what}: no points")));
    }
    if let Some(e) = energy.iter().find(|e| !(e.is_finite() && **e > 0.0)) {
        return Err(invalid(format!(
            "{what}: energies must be positive and finite, got {e}"
        )));
    }
    if let Some(w) = energy.windows(2).find(|w| w[1] < w[0]) {
        return Err(invalid(format!(
            "{what}: energies must not decrease, got {} after {}",
            w[1], w[0]
        )));
    }
    if let Some(v) = value.iter().find(|v| !v.is_finite()) {
        return Err(invalid(format!("{what}: values must be finite, got {v}")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Special functions and quadrature
// ---------------------------------------------------------------------------

/// Complementary error function for `x >= 0`, to within an ulp or so.
///
/// Rust's `f64::erfc` is not yet stable, and this crate takes no dependencies,
/// so this is the FreeBSD `s_erf.c` rational approximation, as carried by the
/// `libm` crate, restricted to non-negative arguments:
///
/// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
/// Developed at SunPro, a Sun Microsystems, Inc. business.
/// Permission to use, copy, modify, and distribute this software is freely
/// granted, provided that this notice is preserved.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
fn erfc(x: f64) -> f64 {
    const ERX: f64 = 8.45062911510467529297e-01;
    const PP0: f64 = 1.28379167095512558561e-01;
    const PP1: f64 = -3.25042107247001499370e-01;
    const PP2: f64 = -2.84817495755985104766e-02;
    const PP3: f64 = -5.77027029648944159157e-03;
    const PP4: f64 = -2.37630166566501626084e-05;
    const QQ1: f64 = 3.97917223959155352819e-01;
    const QQ2: f64 = 6.50222499887672944485e-02;
    const QQ3: f64 = 5.08130628187576562776e-03;
    const QQ4: f64 = 1.32494738004321644526e-04;
    const QQ5: f64 = -3.96022827877536812320e-06;
    const PA0: f64 = -2.36211856075265944077e-03;
    const PA1: f64 = 4.14856118683748331666e-01;
    const PA2: f64 = -3.72207876035701323847e-01;
    const PA3: f64 = 3.18346619901161753674e-01;
    const PA4: f64 = -1.10894694282396677476e-01;
    const PA5: f64 = 3.54783043256182359371e-02;
    const PA6: f64 = -2.16637559486879084300e-03;
    const QA1: f64 = 1.06420880400844228286e-01;
    const QA2: f64 = 5.40397917702171048937e-01;
    const QA3: f64 = 7.18286544141962662868e-02;
    const QA4: f64 = 1.26171219808761642112e-01;
    const QA5: f64 = 1.36370839120290507362e-02;
    const QA6: f64 = 1.19844998467991074170e-02;
    const RA0: f64 = -9.86494403484714822705e-03;
    const RA1: f64 = -6.93858572707181764372e-01;
    const RA2: f64 = -1.05586262253232909814e+01;
    const RA3: f64 = -6.23753324503260060396e+01;
    const RA4: f64 = -1.62396669462573470355e+02;
    const RA5: f64 = -1.84605092906711035994e+02;
    const RA6: f64 = -8.12874355063065934246e+01;
    const RA7: f64 = -9.81432934416914548592e+00;
    const SA1: f64 = 1.96512716674392571292e+01;
    const SA2: f64 = 1.37657754143519042600e+02;
    const SA3: f64 = 4.34565877475229228821e+02;
    const SA4: f64 = 6.45387271733267880336e+02;
    const SA5: f64 = 4.29008140027567833386e+02;
    const SA6: f64 = 1.08635005541779435134e+02;
    const SA7: f64 = 6.57024977031928170135e+00;
    const SA8: f64 = -6.04244152148580987438e-02;
    const RB0: f64 = -9.86494292470009928597e-03;
    const RB1: f64 = -7.99283237680523006574e-01;
    const RB2: f64 = -1.77579549177547519889e+01;
    const RB3: f64 = -1.60636384855821916062e+02;
    const RB4: f64 = -6.37566443368389627722e+02;
    const RB5: f64 = -1.02509513161107724954e+03;
    const RB6: f64 = -4.83519191608651397019e+02;
    const SB1: f64 = 3.03380607434824582924e+01;
    const SB2: f64 = 3.25792512996573918826e+02;
    const SB3: f64 = 1.53672958608443695994e+03;
    const SB4: f64 = 3.19985821950859553908e+03;
    const SB5: f64 = 2.55305040643316442583e+03;
    const SB6: f64 = 4.74528541206955367215e+02;
    const SB7: f64 = -2.24409524465858183362e+01;

    debug_assert!(x >= 0.0);
    if x < 0.84375 {
        if x < 1.3877787807814457e-17 {
            return 1.0 - x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if x < 0.25 {
            return 1.0 - (x + x * y);
        }
        return 0.5 - (x - 0.5 + x * y);
    }
    if x < 1.25 {
        let s = x - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return 1.0 - ERX - p / q;
    }
    if x >= 28.0 {
        return 0.0;
    }
    let s = 1.0 / (x * x);
    let (r, big_s) = if x < 1.0 / 0.35 {
        (
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
            1.0 + s
                * (SA1
                    + s * (SA2
                        + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
        )
    } else {
        (
            RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
        )
    };
    // Split x into a part whose square is exact and a small remainder, so
    // that exp(-x²) does not lose the low bits of x².
    let z = f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000);
    (-z * z - 0.5625).exp() * ((z - x) * (z + x) + r / big_s).exp() / x
}

/// `F_n(a) = ∫_a^∞ zⁿ exp(-z²) dz/√π` for `n = 0..=4` and `a >= 0`.
///
/// Every term of the recurrence is positive, so nothing cancels.
fn f_functions(a: f64) -> [f64; 5] {
    if a == f64::INFINITY {
        return [0.0; 5];
    }
    let e = 0.5 * FRAC_1_SQRT_PI * (-a * a).exp();
    let f0 = 0.5 * erfc(a);
    let f1 = e;
    let f2 = a * e + 0.5 * f0;
    let f3 = a * a * e + f1;
    let f4 = a * a * a * e + 1.5 * f2;
    [f0, f1, f2, f3, f4]
}

/// `H_n(a, b) = ∫_a^b zⁿ exp(-z²) dz/√π` for `n = 0..=4`, `a <= b`, `b`
/// possibly infinite.
///
/// Each half-line is reduced to non-negative arguments, where `F_n` is a sum of
/// positive terms, so a difference of `F_n` loses no more than the ratio of
/// their sizes; across zero the two halves are added.
fn h_functions(a: f64, b: f64) -> [f64; 5] {
    let positive = |lo: f64, hi: f64| {
        let (fl, fh) = (f_functions(lo), f_functions(hi));
        std::array::from_fn(|n| fl[n] - fh[n])
    };
    let reflect = |h: [f64; 5]| -> [f64; 5] {
        std::array::from_fn(|n| if n % 2 == 1 { -h[n] } else { h[n] })
    };
    if a >= 0.0 {
        positive(a, b)
    } else if b <= 0.0 {
        reflect(positive(-b, -a))
    } else {
        let below = reflect(positive(0.0, -a));
        let above = positive(0.0, b);
        std::array::from_fn(|n| below[n] + above[n])
    }
}

/// Gauss-Legendre nodes and weights on `[-1, 1]`.
#[derive(Debug, Clone)]
struct GaussLegendre {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendre {
    fn new(n: usize) -> Self {
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let legendre = |z: f64| {
            // P_n(z) and its derivative by the three-term recurrence.
            let (mut p, mut p_prev) = (1.0, 0.0);
            for j in 1..=n {
                let jf = j as f64;
                let next = ((2.0 * jf - 1.0) * z * p - (jf - 1.0) * p_prev) / jf;
                p_prev = p;
                p = next;
            }
            let dp = n as f64 * (z * p - p_prev) / (z * z - 1.0);
            (p, dp)
        };
        for i in 0..n.div_ceil(2) {
            let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (p, dp) = legendre(z);
                let step = p / dp;
                z -= step;
                if step.abs() <= 1e-16 {
                    break;
                }
            }
            let (_, dp) = legendre(z);
            nodes[i] = -z;
            nodes[n - 1 - i] = z;
            let w = 2.0 / ((1.0 - z * z) * dp * dp);
            weights[i] = w;
            weights[n - 1 - i] = w;
        }
        Self { nodes, weights }
    }

    /// `∫_a^b f`.
    fn integrate(&self, a: f64, b: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
        let sum: f64 = self
            .nodes
            .iter()
            .zip(&self.weights)
            .map(|(&u, &w)| w * f(mid + half * u))
            .sum();
        sum * half
    }
}

/// `exp(-(s - t)²) - exp(-(s + t)²)` for `s, t >= 0`, without the cancellation
/// of subtracting the two when `st` is small.
fn kernel(s: f64, t: f64) -> f64 {
    let d = s - t;
    (-d * d).exp() * -(-4.0 * s * t).exp_m1()
}

// ---------------------------------------------------------------------------
// Forward broadening
// ---------------------------------------------------------------------------

/// A 0 K cross section in velocity units: `x = √(αE)`, values unchanged.
struct VelocityTable<'a> {
    x: Vec<f64>,
    sigma: &'a [f64],
}

/// One stretch of the 0 K cross section, in `x`.
#[derive(Debug, Clone, Copy)]
enum Stretch {
    /// `σ₀ = c/x` on `(0, x_first]`.
    OneOverV { c: f64 },
    /// Linear in `x²` (so in energy) from `s0` at `a` to `s1` at `b`.
    Linear { s0: f64, s1: f64 },
    /// A constant from the last point to infinity.
    Constant { s: f64 },
}

impl Stretch {
    /// `σ₀(x)` on `[a, b]`.
    fn at(self, a: f64, b: f64, x: f64) -> f64 {
        match self {
            Stretch::OneOverV { c } => c / x,
            Stretch::Linear { s0, s1 } => {
                s0 + (s1 - s0) * ((x - a) * (x + a)) / ((b - a) * (b + a))
            }
            Stretch::Constant { s } => s,
        }
    }

    /// The coefficients of `(x/t)² σ₀(x)` as a polynomial in `z = x - t`, for
    /// the stretch `[a, b]`.
    ///
    /// The linear case is written around `a` rather than expanded in powers of
    /// `x`, which would cancel catastrophically at high energy: with
    /// `σ₀ = s0 + m (x - a)(x + a)` and `x = t + z`, the factors become
    /// `(z - d)(z + p)` with `d = a - t` and `p = a + t`, both no larger than the
    /// stretch and the window.
    fn coefficients(self, a: f64, b: f64, t: f64) -> [f64; 5] {
        let u = 1.0 / t;
        match self {
            // (x/t)² c/x = c x/t² = c (t + z)/t².
            Stretch::OneOverV { c } => [c * u, c * u * u, 0.0, 0.0, 0.0],
            Stretch::Linear { s0, s1 } => {
                let m = (s1 - s0) / ((b - a) * (b + a));
                let (d, p) = (a - t, a + t);
                quadratic_times_square(u, s0 - m * d * p, m * (p - d), m)
            }
            Stretch::Constant { s } => quadratic_times_square(u, s, 0.0, 0.0),
        }
    }
}

/// `(1 + u z)² (c0 + c1 z + c2 z²)` as coefficients of `z⁰..z⁴`.
fn quadratic_times_square(u: f64, c0: f64, c1: f64, c2: f64) -> [f64; 5] {
    [
        c0,
        c1 + 2.0 * u * c0,
        c2 + 2.0 * u * c1 + u * u * c0,
        2.0 * u * c2 + u * u * c1,
        u * u * c2,
    ]
}

/// `∫_a^b (x/y)² σ₀(x) [exp(-(x - y)²) - exp(-(x + y)²)] dx/√π` over one
/// stretch.
fn stretch_integral(
    stretch: Stretch,
    a: f64,
    b: f64,
    y: f64,
    reach: f64,
    short: &GaussLegendre,
) -> f64 {
    if b - a < SHORT {
        let inv_y = 1.0 / y;
        return FRAC_1_SQRT_PI
            * short.integrate(a, b, |x| {
                let r = x * inv_y;
                r * r * stretch.at(a, b, x) * kernel(x, y)
            });
    }
    let mut total = 0.0;
    for (t, sign) in [(y, 1.0), (-y, -1.0)] {
        // The second exponential is exp(-(x + y)²), which is negligible once
        // the stretch starts `reach` widths above -y.
        if a - t > reach {
            continue;
        }
        let c = stretch.coefficients(a, b, t);
        let h = h_functions(a - t, b - t);
        let sum: f64 = c.iter().zip(&h).map(|(c, h)| c * h).sum();
        total += sign * sum;
    }
    total
}

impl VelocityTable<'_> {
    /// The broadened cross section at `y > 0`.
    fn broaden_at(&self, y: f64, reach: f64, short: &GaussLegendre) -> f64 {
        let x = &self.x;
        let s = self.sigma;
        let n = x.len();
        let (lo, hi) = (y - reach, y + reach);
        let mut total = 0.0;
        if x[0] > lo {
            let c = s[0] * x[0];
            total += stretch_integral(Stretch::OneOverV { c }, 0.0, x[0], y, reach, short);
        }
        // Intervals [x[i], x[i + 1]] that reach into the window.
        let first = x.partition_point(|&v| v <= lo).saturating_sub(1);
        for i in first..n.saturating_sub(1) {
            let (a, b) = (x[i], x[i + 1]);
            if a >= hi {
                break;
            }
            if b <= lo || b <= a {
                continue;
            }
            let stretch = Stretch::Linear {
                s0: s[i],
                s1: s[i + 1],
            };
            total += stretch_integral(stretch, a, b, y, reach, short);
        }
        if x[n - 1] < hi {
            let stretch = Stretch::Constant { s: s[n - 1] };
            total += stretch_integral(stretch, x[n - 1], f64::INFINITY, y, reach, short);
        }
        total
    }
}

/// Doppler-broaden a 0 K cross section to `temperature` at the energies `at`.
///
/// `energy` (eV, non-decreasing, a repeated energy being a step) and `sigma`
/// describe the cross section, linear-linear between points, continued as
/// `1/v` below the first point and as a constant above the last, as NJOY's
/// BROADR does. `awr` is the target's mass in neutron masses and
/// `temperature` is in kelvin; at 0 K the cross section is interpolated
/// unchanged. Every energy in `at` must be positive.
///
/// See the [module documentation](self) for the kernel and how it is
/// integrated.
pub fn broaden(
    energy: &[f64],
    sigma: &[f64],
    awr: f64,
    temperature: f64,
    at: &[f64],
) -> Result<Vec<f64>> {
    broaden_within(energy, sigma, awr, temperature, at, WINDOW)
}

/// [`broaden`] with the kernel cut where BROADR cuts it, after the first
/// interval of the 0 K table to end more than `reach` Doppler widths away.
///
/// At [`BROADR_REACH`] this is what NJOY computes; at [`WINDOW`] it is
/// [`broaden`].
pub fn broaden_within(
    energy: &[f64],
    sigma: &[f64],
    awr: f64,
    temperature: f64,
    at: &[f64],
    reach: f64,
) -> Result<Vec<f64>> {
    if reach.is_nan() || reach <= 0.0 {
        return Err(invalid(format!(
            "the kernel's reach must be positive, got {reach}"
        )));
    }
    check_table("the 0 K cross section", energy, sigma)?;
    check_awr_temperature(awr, temperature)?;
    if let Some(e) = at.iter().find(|e| !(e.is_finite() && **e > 0.0)) {
        return Err(invalid(format!(
            "broadening energies must be positive and finite, got {e}"
        )));
    }
    if temperature == 0.0 {
        return Ok(at.iter().map(|&e| interpolate(energy, sigma, e)).collect());
    }
    let alpha = alpha(awr, temperature);
    let table = VelocityTable {
        x: energy.iter().map(|&e| (alpha * e).sqrt()).collect(),
        sigma,
    };
    let short = GaussLegendre::new(SHORT_POINTS);
    Ok(at
        .iter()
        .map(|&e| table.broaden_at((alpha * e).sqrt(), reach, &short))
        .collect())
}

/// A 0 K cross section at `e` under the same continuation [`broaden`] uses.
fn interpolate(energy: &[f64], sigma: &[f64], e: f64) -> f64 {
    let n = energy.len();
    if e < energy[0] {
        return sigma[0] * (energy[0] / e).sqrt();
    }
    if e >= energy[n - 1] {
        return sigma[n - 1];
    }
    let i = energy.partition_point(|&v| v <= e) - 1;
    let (e0, e1) = (energy[i], energy[i + 1]);
    sigma[i] + (sigma[i + 1] - sigma[i]) * (e - e0) / (e1 - e0)
}

// ---------------------------------------------------------------------------
// The flux weight
// ---------------------------------------------------------------------------

/// A flux density `ψ(E)`, per eV, that a reaction rate `∫ σ ψ dE` is taken
/// against. It is zero outside the energies it is given on.
#[derive(Debug, Clone, PartialEq)]
pub enum Flux {
    /// Constant within each group: `density[g]` on `[edges[g], edges[g + 1])`.
    /// A multigroup flux `φ_g` flat in energy has `density[g] = φ_g / ΔE_g`.
    Histogram { edges: Vec<f64>, density: Vec<f64> },
    /// Flat in lethargy within each group: `per_lethargy[g] / E` on
    /// `[edges[g], edges[g + 1])`. A multigroup flux `φ_g` with a `1/E` shape
    /// inside each group has `per_lethargy[g] = φ_g / ln(E_{g+1} / E_g)`.
    Lethargy {
        edges: Vec<f64>,
        per_lethargy: Vec<f64>,
    },
    /// Linear-linear between points; a repeated energy is a step. This is the
    /// form of a self-shielded shape `φ_g s(E) / ∫_g s dE`.
    Pointwise { energy: Vec<f64>, value: Vec<f64> },
}

/// One stretch of a flux density in energy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    /// `start + slope (E - lo)`.
    Linear { start: f64, slope: f64 },
    /// `c / E`.
    InverseE { c: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Piece {
    lo: f64,
    hi: f64,
    shape: Shape,
}

impl Piece {
    fn at(&self, e: f64) -> f64 {
        match self.shape {
            Shape::Linear { start, slope } => start + slope * (e - self.lo),
            Shape::InverseE { c } => c / e,
        }
    }

    /// The largest `|ψ|` on the piece.
    fn magnitude(&self) -> f64 {
        self.at(self.lo).abs().max(self.at(self.hi).abs())
    }

    /// The piece cut to `[lo, hi]`, which must lie within it.
    fn cut(&self, lo: f64, hi: f64) -> Piece {
        let shape = match self.shape {
            Shape::Linear { start, slope } => Shape::Linear {
                start: start + slope * (lo - self.lo),
                slope,
            },
            other => other,
        };
        Piece { lo, hi, shape }
    }
}

impl Flux {
    /// The flux as pieces of positive width, in increasing energy.
    fn pieces(&self) -> Result<Vec<Piece>> {
        let groups = |edges: &[f64], values: &[f64], what: &str| -> Result<()> {
            if edges.len() != values.len() + 1 {
                return Err(invalid(format!(
                    "{what}: {} edges need {} values, got {}",
                    edges.len(),
                    edges.len().saturating_sub(1),
                    values.len()
                )));
            }
            if values.is_empty() {
                return Err(invalid(format!("{what}: no groups")));
            }
            check_table(what, &edges[1..], values)?;
            check_table(what, &edges[..1], &values[..1])?;
            if let Some(w) = edges.windows(2).find(|w| w[1] <= w[0]) {
                return Err(invalid(format!(
                    "{what}: group edges must increase, got {} after {}",
                    w[1], w[0]
                )));
            }
            Ok(())
        };
        match self {
            Flux::Histogram { edges, density } => {
                groups(edges, density, "histogram flux")?;
                Ok(edges
                    .windows(2)
                    .zip(density)
                    .map(|(w, &d)| Piece {
                        lo: w[0],
                        hi: w[1],
                        shape: Shape::Linear {
                            start: d,
                            slope: 0.0,
                        },
                    })
                    .collect())
            }
            Flux::Lethargy {
                edges,
                per_lethargy,
            } => {
                groups(edges, per_lethargy, "lethargy flux")?;
                Ok(edges
                    .windows(2)
                    .zip(per_lethargy)
                    .map(|(w, &c)| Piece {
                        lo: w[0],
                        hi: w[1],
                        shape: Shape::InverseE { c },
                    })
                    .collect())
            }
            Flux::Pointwise { energy, value } => {
                check_table("pointwise flux", energy, value)?;
                if let Some(w) = energy.windows(3).find(|w| w[0] == w[2]) {
                    return Err(invalid(format!(
                        "pointwise flux: energy {} appears more than twice",
                        w[0]
                    )));
                }
                Ok(energy
                    .windows(2)
                    .zip(value.windows(2))
                    .filter(|(e, _)| e[1] > e[0])
                    .map(|(e, v)| Piece {
                        lo: e[0],
                        hi: e[1],
                        shape: Shape::Linear {
                            start: v[0],
                            slope: (v[1] - v[0]) / (e[1] - e[0]),
                        },
                    })
                    .collect())
            }
        }
    }
}

/// A polynomial in energy on `[lo, hi]`, as Chebyshev coefficients in
/// `u = (E - mid) / half`.
#[derive(Debug, Clone, PartialEq)]
struct Panel {
    lo: f64,
    hi: f64,
    coefficients: [f64; PANEL_POINTS],
}

impl Panel {
    fn at(&self, e: f64) -> f64 {
        let (mid, half) = (0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo));
        let u = (e - mid) / half;
        // Clenshaw's recurrence.
        let (mut b1, mut b2) = (0.0, 0.0);
        for &c in self.coefficients[1..].iter().rev() {
            let b0 = 2.0 * u * b1 - b2 + c;
            b2 = b1;
            b1 = b0;
        }
        u * b1 - b2 + self.coefficients[0]
    }
}

/// The broadened flux weight `w`, such that `∫ σ_T ψ dE = ∫ σ₀ w dE` for every
/// 0 K cross section `σ₀`. Built by [`broadened_weight`].
///
/// It is the sum of two parts: polynomial panels carrying the flux that is
/// broadened, and the flux that is not (above `broaden_below`, or all of it at
/// 0 K) carried unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct BroadenedWeight {
    panels: Vec<Panel>,
    unbroadened: Vec<Piece>,
}

/// Build the broadened flux weight for a target of mass ratio `awr` at
/// `temperature` kelvin.
///
/// `broaden_below` is BROADR's `thnmax`: flux at energies above it weights the
/// 0 K cross section directly, as the PENDF file copies the 0 K data there.
/// `None` broadens everywhere.
///
/// See the [module documentation](self) for what `w` is and how it is
/// represented.
pub fn broadened_weight(
    flux: &Flux,
    awr: f64,
    temperature: f64,
    broaden_below: Option<f64>,
) -> Result<BroadenedWeight> {
    check_awr_temperature(awr, temperature)?;
    let pieces = flux.pieces()?;
    if let Some(e) = broaden_below {
        if e.is_nan() || e <= 0.0 {
            return Err(invalid(format!(
                "the upper limit of broadening must be positive, got {e}"
            )));
        }
    }
    // At 0 K nothing is broadened and w is the flux itself.
    let limit = if temperature == 0.0 {
        0.0
    } else {
        broaden_below.unwrap_or(f64::INFINITY)
    };
    let mut broadened = Vec::new();
    let mut unbroadened = Vec::new();
    for piece in pieces {
        if piece.hi <= limit {
            broadened.push(piece);
        } else if piece.lo >= limit {
            unbroadened.push(piece);
        } else {
            broadened.push(piece.cut(piece.lo, limit));
            unbroadened.push(piece.cut(limit, piece.hi));
        }
    }
    let panels = if broadened.is_empty() {
        Vec::new()
    } else {
        WeightBuilder::new(broadened, alpha(awr, temperature)).panels()
    };
    Ok(BroadenedWeight {
        panels,
        unbroadened,
    })
}

/// What the integral defining `w` needs: the flux in velocity units, and the
/// quadrature.
struct WeightBuilder {
    alpha: f64,
    pieces: Vec<Piece>,
    /// Each piece's ends in `y`.
    y_lo: Vec<f64>,
    y_hi: Vec<f64>,
    rule: GaussLegendre,
}

impl WeightBuilder {
    fn new(pieces: Vec<Piece>, alpha: f64) -> Self {
        let y_lo = pieces.iter().map(|p| (alpha * p.lo).sqrt()).collect();
        let y_hi = pieces.iter().map(|p| (alpha * p.hi).sqrt()).collect();
        Self {
            alpha,
            pieces,
            y_lo,
            y_hi,
            rule: GaussLegendre::new(WEIGHT_POINTS),
        }
    }

    /// The pieces that overlap `[lo, hi]` in `y`.
    fn overlapping(&self, lo: f64, hi: f64) -> impl Iterator<Item = usize> + '_ {
        let first = self.y_hi.partition_point(|&v| v <= lo);
        (first..self.pieces.len()).take_while(move |&i| self.y_lo[i] < hi)
    }

    /// `w(x)`, the defining integral cut to [`WINDOW`] widths either side.
    fn weight(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (lo, hi) = ((x - WINDOW).max(0.0), x + WINDOW);
        let mut total = 0.0;
        for i in self.overlapping(lo, hi) {
            let piece = &self.pieces[i];
            let (a, b) = (self.y_lo[i].max(lo), self.y_hi[i].min(hi));
            // Sub-intervals no wider than WEIGHT_STEP, and no wider than half
            // their distance from zero, where a 1/E flux varies fastest.
            let mut start = a;
            while start < b {
                let step = WEIGHT_STEP.min((0.5 * start).max(0.05 * WEIGHT_STEP));
                let end = (start + step).min(b);
                let end = if b - end < 1e-3 * step { b } else { end };
                total += self.rule.integrate(start, end, |y| {
                    piece.at(y * y / self.alpha) * kernel(y, x) / y
                });
                start = end;
            }
        }
        total * x * FRAC_1_SQRT_PI
    }

    /// The largest `|ψ|` within [`WINDOW`] of `[lo, hi]` in `y`.
    fn flux_scale(&self, lo: f64, hi: f64) -> f64 {
        self.overlapping(lo - WINDOW, hi + WINDOW)
            .map(|i| self.pieces[i].magnitude())
            .fold(0.0, f64::max)
    }

    /// Fit `w` on `[y_a, y_b]` as a Chebyshev series in energy.
    fn fit(&self, y_a: f64, y_b: f64) -> Panel {
        let (lo, hi) = (y_a * y_a / self.alpha, y_b * y_b / self.alpha);
        let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let n = PANEL_POINTS;
        let theta = |j: usize| PI * (j as f64 + 0.5) / n as f64;
        let values: Vec<f64> = (0..n)
            .map(|j| self.weight((self.alpha * (mid + half * theta(j).cos())).sqrt()))
            .collect();
        let mut coefficients = [0.0; PANEL_POINTS];
        for (k, c) in coefficients.iter_mut().enumerate() {
            let sum: f64 = values
                .iter()
                .enumerate()
                .map(|(j, v)| v * (k as f64 * theta(j)).cos())
                .sum();
            *c = 2.0 * sum / n as f64;
        }
        coefficients[0] *= 0.5;
        Panel {
            lo,
            hi,
            coefficients,
        }
    }

    /// The panels: started at every flux edge, at one Doppler width apart
    /// within a window of each, then halved in `y` until each fits.
    fn panels(&self) -> Vec<Panel> {
        let mut knots: Vec<f64> = self
            .y_lo
            .iter()
            .chain(&self.y_hi)
            .copied()
            .collect::<Vec<_>>();
        let support_lo = (self.y_lo[0] - WINDOW).max(0.0);
        let support_hi = self.y_hi[self.y_hi.len() - 1] + WINDOW;
        knots.push(support_lo);
        knots.push(support_hi);
        knots.sort_by(f64::total_cmp);
        knots.dedup();

        let mut starts = Vec::new();
        for w in knots.windows(2) {
            let (p, q) = (w[0], w[1]);
            starts.push(p);
            let reach = WINDOW + 1.0;
            let mut steps: Vec<f64> = Vec::new();
            let mut k = 1.0;
            while k <= reach && p + k < q {
                steps.push(p + k);
                steps.push(q - k);
                k += 1.0;
            }
            steps.retain(|&v| v > p && v < q);
            steps.sort_by(f64::total_cmp);
            steps.dedup_by(|a, b| (*a - *b).abs() < 0.5);
            starts.extend(steps);
        }
        starts.push(support_hi);

        let mut panels = Vec::new();
        for w in starts.windows(2) {
            let mut stack = vec![(w[0], w[1], 0u32)];
            while let Some((a, b, depth)) = stack.pop() {
                let panel = self.fit(a, b);
                let tail = panel.coefficients[PANEL_POINTS - 3..]
                    .iter()
                    .fold(0.0_f64, |m, c| m.max(c.abs()));
                let floor = 16.0 * f64::EPSILON * (1.0 + b);
                let scale = self.flux_scale(a, b) * PANEL_TOLERANCE.max(floor);
                if tail <= scale || depth >= 30 {
                    panels.push(panel);
                } else {
                    let m = 0.5 * (a + b);
                    // Pushed upper half first so the lower half is fitted
                    // first and the panels come out in order.
                    stack.push((m, b, depth + 1));
                    stack.push((a, m, depth + 1));
                }
            }
        }
        panels.retain(|p| p.hi > p.lo);
        panels
    }
}

/// One stretch of a 0 K cross section in energy, as [`BroadenedWeight::integrate`]
/// walks it: `start + slope (E - lo)`, or `c / √E` below the first point.
#[derive(Debug, Clone, Copy)]
enum Cross {
    Linear { lo: f64, start: f64, slope: f64 },
    InverseSqrt { c: f64 },
}

impl Cross {
    fn at(self, e: f64) -> f64 {
        match self {
            Cross::Linear { lo, start, slope } => start + slope * (e - lo),
            Cross::InverseSqrt { c } => c / e.sqrt(),
        }
    }
}

/// The stretches of a 0 K cross section, continued as [`broaden`] does: `1/v`
/// from zero, linear between points, constant to infinity.
fn cross_stretches(energy: &[f64], sigma: &[f64]) -> Vec<(f64, f64, Cross)> {
    let n = energy.len();
    let mut out = Vec::with_capacity(n + 1);
    out.push((
        0.0,
        energy[0],
        Cross::InverseSqrt {
            c: sigma[0] * energy[0].sqrt(),
        },
    ));
    for i in 0..n - 1 {
        let (a, b) = (energy[i], energy[i + 1]);
        if b > a {
            let slope = (sigma[i + 1] - sigma[i]) / (b - a);
            out.push((
                a,
                b,
                Cross::Linear {
                    lo: a,
                    start: sigma[i],
                    slope,
                },
            ));
        }
    }
    out.push((
        energy[n - 1],
        f64::INFINITY,
        Cross::Linear {
            lo: energy[n - 1],
            start: sigma[n - 1],
            slope: 0.0,
        },
    ));
    out
}

/// `∫_lo^hi σ₀ ψ dE` in closed form, for one stretch of each.
fn cross_times_piece(cross: Cross, piece: &Piece, lo: f64, hi: f64) -> f64 {
    let h = hi - lo;
    match (cross, piece.shape) {
        (Cross::Linear { slope: q, .. }, Shape::Linear { slope: t, .. }) => {
            let (s, p) = (cross.at(lo), piece.at(lo));
            h * (s * p + 0.5 * h * (s * t + p * q) + q * t * h * h / 3.0)
        }
        (Cross::Linear { slope: q, .. }, Shape::InverseE { c }) => {
            // c ∫ (s + q (E - lo)) / E dE = c [s ln(hi/lo) + q lo (r - ln(1 + r))],
            // r = h / lo, with the second term written so it does not cancel.
            let s = cross.at(lo);
            let r = h / lo;
            let log = r.ln_1p();
            let r_minus_log = if r < 1e-3 {
                r * r * (0.5 - r * (1.0 / 3.0 - r * (0.25 - r / 5.0)))
            } else {
                r - log
            };
            c * (s * log + q * lo * r_minus_log)
        }
        (Cross::InverseSqrt { c }, Shape::Linear { slope: t, .. }) => {
            // c ∫ (p + t (E - lo)) E^(-1/2) dE.
            let p = piece.at(lo);
            let (ra, rb) = (lo.sqrt(), hi.sqrt());
            let root = h / (ra + rb);
            let three_halves = (hi * rb - lo * ra) / 1.5;
            c * ((p - t * lo) * 2.0 * root + t * three_halves)
        }
        (Cross::InverseSqrt { c }, Shape::InverseE { c: k }) => {
            let (ra, rb) = (lo.sqrt(), hi.sqrt());
            2.0 * c * k * (rb - ra) / (ra * rb)
        }
    }
}

impl BroadenedWeight {
    /// `w` at energy `e`, per eV.
    pub fn at(&self, e: f64) -> f64 {
        let i = self.panels.partition_point(|p| p.hi <= e);
        let panel = self
            .panels
            .get(i)
            .filter(|p| p.lo <= e)
            .map_or(0.0, |p| p.at(e));
        let j = self.unbroadened.partition_point(|p| p.hi <= e);
        let direct = self
            .unbroadened
            .get(j)
            .filter(|p| p.lo <= e)
            .map_or(0.0, |p| p.at(e));
        panel + direct
    }

    /// The number of polynomial panels `w` is held on.
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// `∫ σ₀(E) w(E) dE` for a 0 K cross section given on `energy` and
    /// `sigma`, linear-linear between points and continued as [`broaden`]
    /// continues it, `1/v` below the first point and constant above the last.
    ///
    /// This equals `∫ σ_T ψ dE`, with `σ_T` what [`broaden`] returns for the
    /// same table, to the accuracy of the panel fit (about `1e-14` of the flux
    /// density) and the kernel's reach ([`WINDOW`]).
    pub fn integrate(&self, energy: &[f64], sigma: &[f64]) -> Result<f64> {
        check_table("the 0 K cross section", energy, sigma)?;
        let stretches = cross_stretches(energy, sigma);
        // Linear times a degree-15 polynomial needs 9 points; 1/√E times one,
        // taken in √E, is a degree-30 polynomial and needs 16.
        let linear_rule = GaussLegendre::new(PANEL_POINTS / 2 + 1);
        let root_rule = GaussLegendre::new(PANEL_POINTS);

        let mut total = sweep(
            &stretches,
            &self.panels,
            |p| (p.lo, p.hi),
            |cross, panel, lo, hi| match cross {
                Cross::Linear { .. } => {
                    linear_rule.integrate(lo, hi, |e| cross.at(e) * panel.at(e))
                }
                Cross::InverseSqrt { c } => {
                    root_rule.integrate(lo.sqrt(), hi.sqrt(), |r| 2.0 * c * panel.at(r * r))
                }
            },
        );
        total += sweep(
            &stretches,
            &self.unbroadened,
            |p| (p.lo, p.hi),
            cross_times_piece,
        );
        Ok(total)
    }
}

/// Walk the stretches of a cross section and a sorted run of non-overlapping
/// targets together, summing `f` over every overlap.
fn sweep<T>(
    stretches: &[(f64, f64, Cross)],
    targets: &[T],
    bounds: impl Fn(&T) -> (f64, f64),
    mut f: impl FnMut(Cross, &T, f64, f64) -> f64,
) -> f64 {
    let mut total = 0.0;
    let mut i = 0;
    for target in targets {
        let (t_lo, t_hi) = bounds(target);
        while i < stretches.len() && stretches[i].1 <= t_lo {
            i += 1;
        }
        let mut k = i;
        while k < stretches.len() && stretches[k].0 < t_hi {
            let (lo, hi) = (stretches[k].0.max(t_lo), stretches[k].1.min(t_hi));
            if hi > lo {
                total += f(stretches[k].2, target, lo, hi);
            }
            k += 1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_matches_reference_values() {
        // Values from a 50-digit evaluation.
        let cases = [
            (0.0, 1.0),
            (0.1, 0.887_537_083_981_715),
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 4.677_734_981_047_266e-3),
            (4.0, 1.541_725_790_028_002e-8),
            (7.0, 4.183_825_607_779_414e-23),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            assert!(
                ((got - want) / want).abs() < 4e-16,
                "erfc({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn gauss_legendre_is_exact_for_its_degree() {
        let rule = GaussLegendre::new(9);
        for k in 0..=17 {
            let got = rule.integrate(-1.0, 2.0, |x| x.powi(k));
            let want = (2f64.powi(k + 1) - (-1f64).powi(k + 1)) / (k + 1) as f64;
            assert!((got - want).abs() < 1e-13 * want.abs().max(1.0), "x^{k}");
        }
    }

    #[test]
    fn h_functions_match_quadrature_either_side_of_zero() {
        let rule = GaussLegendre::new(20);
        for (a, b) in [(-3.0, -0.5), (-1.2, 2.5), (0.3, 4.0), (-6.0, 6.0)] {
            let h = h_functions(a, b);
            for (n, h) in h.iter().enumerate() {
                let pieces = ((b - a) / 0.5).ceil() as usize;
                let step = (b - a) / pieces as f64;
                let want: f64 = (0..pieces)
                    .map(|k| {
                        let lo = a + k as f64 * step;
                        FRAC_1_SQRT_PI
                            * rule.integrate(lo, lo + step, |z| z.powi(n as i32) * (-z * z).exp())
                    })
                    .sum();
                assert!(
                    (h - want).abs() < 1e-15,
                    "H_{n}({a}, {b}) = {h}, want {want}"
                );
            }
        }
    }

    #[test]
    fn short_and_long_paths_agree() {
        // The same stretch integrated both ways, either side of SHORT.
        let rule = GaussLegendre::new(SHORT_POINTS);
        let stretch = Stretch::Linear { s0: 3.0, s1: 7.0 };
        for (a, y) in [(10.0, 10.2), (0.05, 0.3), (2.0, 1.0), (500.0, 503.0)] {
            let b = a + 0.999 * SHORT;
            let closed: f64 = [(y, 1.0), (-y, -1.0)]
                .iter()
                .map(|&(t, sign)| {
                    let c = stretch.coefficients(a, b, t);
                    let h = h_functions(a - t, b - t);
                    sign * c.iter().zip(&h).map(|(c, h)| c * h).sum::<f64>()
                })
                .sum();
            let quadrature = stretch_integral(stretch, a, b, y, WINDOW, &rule);
            assert!(
                (closed - quadrature).abs() < 1e-13 * quadrature.abs(),
                "a={a} y={y}: {closed} against {quadrature}"
            );
        }
    }

    #[test]
    fn a_constant_broadens_to_the_free_gas_closed_form() {
        // σ₀ = s everywhere broadens to s (1 + 1/(2y²)) erf(y) + s e^{-y²}/(y√π).
        let (awr, t) = (10.0, 1000.0);
        let alpha = alpha(awr, t);
        let energy = [1e-12, 1e9];
        let sigma = [5.0, 5.0];
        // A 1/v continuation below 1e-12 eV is invisible at these energies.
        let at = [1e-4, 1e-2, 0.1, 1.0, 30.0];
        let got = broaden(&energy, &sigma, awr, t, &at).unwrap();
        for (e, got) in at.iter().zip(got) {
            let y = (alpha * e).sqrt();
            let erf = 1.0 - erfc(y);
            let want = 5.0 * ((1.0 + 0.5 / (y * y)) * erf + (-y * y).exp() * FRAC_1_SQRT_PI / y);
            assert!(
                (got - want).abs() < 1e-13 * want,
                "E={e}: {got} against {want}"
            );
        }
    }

    #[test]
    fn one_over_v_is_unchanged() {
        // The free-gas kernel leaves a 1/v cross section exactly 1/v. One point
        // far above makes the table 1/v everywhere the kernel reaches.
        let energy = [1e3];
        let sigma = [2.0 / 1e3f64.sqrt()];
        let at = [1e-7, 1e-5, 1e-3, 0.1, 1.0];
        let got = broaden(&energy, &sigma, 50.0, 3000.0, &at).unwrap();
        for (e, got) in at.iter().zip(got) {
            let want = 2.0 / e.sqrt();
            assert!(
                (got / want - 1.0).abs() < 1e-13,
                "E={e}: {got} against {want}"
            );
        }
    }

    #[test]
    fn zero_kelvin_is_the_identity() {
        let energy = [1.0, 2.0, 2.0, 5.0];
        let sigma = [1.0, 3.0, 4.0, 2.0];
        let got = broaden(&energy, &sigma, 10.0, 0.0, &[1.5, 3.5, 10.0]).unwrap();
        assert_eq!(got, vec![2.0, 3.0, 2.0]);

        let flux = Flux::Histogram {
            edges: vec![1.0, 3.0, 6.0],
            density: vec![2.0, 0.5],
        };
        let weight = broadened_weight(&flux, 10.0, 0.0, None).unwrap();
        assert_eq!(weight.panel_count(), 0);
        assert_eq!(weight.at(2.0), 2.0);
        assert_eq!(weight.at(4.0), 0.5);
        let got = weight.integrate(&energy, &sigma).unwrap();
        // ∫ σ ψ by hand: 4 on [1, 2], 22/3 on [2, 3], 8/3 on [3, 5], 1 on [5, 6].
        let want = 15.0;
        assert!((got - want).abs() < 1e-12 * want, "{got} against {want}");
    }

    #[test]
    fn rejects_bad_input() {
        assert!(broaden(&[1.0, 0.5], &[1.0, 1.0], 1.0, 300.0, &[1.0]).is_err());
        assert!(broaden(&[1.0], &[1.0, 1.0], 1.0, 300.0, &[1.0]).is_err());
        assert!(broaden(&[1.0], &[1.0], -1.0, 300.0, &[1.0]).is_err());
        assert!(broaden(&[1.0], &[1.0], 1.0, -3.0, &[1.0]).is_err());
        assert!(broaden(&[1.0], &[1.0], 1.0, 300.0, &[0.0]).is_err());
        let bad = Flux::Histogram {
            edges: vec![1.0, 1.0, 2.0],
            density: vec![1.0, 1.0],
        };
        assert!(broadened_weight(&bad, 1.0, 300.0, None).is_err());
        let ok = Flux::Histogram {
            edges: vec![1.0, 2.0],
            density: vec![1.0],
        };
        assert!(broadened_weight(&ok, 1.0, 300.0, Some(0.0)).is_err());
    }
}
