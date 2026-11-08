//! Quartic factorisation for the GPU ray-torus intersection: Algorithm
//! 1010 of A. G. Orellana and C. De Michele, "Algorithm 1010: Boosting
//! Efficiency in Solving Quartic Equations with No Compromise in
//! Accuracy", ACM Transactions on Mathematical Software 46(2), article
//! 20 (2020), <https://doi.org/10.1145/3386241>.
//!
//! The same algorithm as the CPU solver in `yamc-geo`
//! (`surface::roots`), which carries the derivation and the paper's
//! equation numbers: the monic quartic is written as
//! `(x^2 + l1 x + l3)^2 + d2 (x + l2)^2` from `phi0`, the dominant root
//! of the resolvent cubic, factorised into two quadratics, and the
//! factors are polished by Newton-Raphson. It is rewritten here in
//! `#[cube]` form because the kernels cannot call `yamc-geo`.
//!
//! # Difference from the CPU solver
//!
//! cubecl-spirv emits no valid f64 `cbrt` on the AMD/RADV path (see
//! `polyfills`), so Cardano's formula for a resolvent cubic with one
//! real root takes its cube root from [`cbrt_f64`], a product of
//! repeated square roots refined by Newton. The three-real-root case is
//! Newton from a bound on both sides, as in `yamc-geo`. The paper's own
//! Newton polish of `phi0` follows, so the two solvers agree to
//! round-off. Each `#[cube]` function has a plain Rust `_cpu` twin with
//! identical arithmetic, which the rayon CPU transport path uses and
//! the parity tests compare against the GPU.
//!
//! cubecl keeps an f64 `let mut` seeded from a bare literal at compile
//! time, which then cannot take a runtime value; such variables are
//! seeded from `rt_zero`, a runtime zero made from an input.

// `x - x` is that runtime zero.
#![allow(clippy::eq_op)]

use cubecl::prelude::*;

/// `f64::MAX^(1/4)` divided by the golden ratio: the quartic rescaling
/// factor applied when `phi0` overflows.
const QUARTIC_RESCALE: f64 = 7.156_344_677_702_229e76;
/// `f64::MAX^(1/3)` divided by the golden ratio: the resolvent cubic's
/// rescaling factor.
const CUBIC_RESCALE: f64 = 3.488_062_137_979_395_6e102;
/// "No real root": the largest f64, which no root of a finite quartic
/// reaches.
pub const NO_ROOT: f64 = f64::MAX;

/// Real factorisation `(x^2 + alpha1 x + beta1)(x^2 + alpha2 x + beta2)`
/// of the quartic rescaled by `x -> x / scale`. `real == 0` when the
/// factors are complex (no real roots); the other fields are then
/// meaningless.
#[derive(CubeType)]
pub struct QuarticFactors {
    pub alpha1: f64,
    pub beta1: f64,
    pub alpha2: f64,
    pub beta2: f64,
    pub scale: f64,
    pub real: u32,
}

/// `|x|` by explicit negation (`abs` is not validated on spirv).
#[cube]
fn fabs(x: f64) -> f64 {
    let mut r = x;
    if x < 0.0 {
        r = -x;
    }
    r
}

/// `1` when `x` is finite (exponent field not all ones), else `0`.
#[cube]
fn finite(x: f64) -> u32 {
    let bits = u64::reinterpret(x);
    let mut r = 1u32;
    if ((bits >> 52u64) & 0x7FFu64) == 0x7FFu64 {
        r = 0u32;
    }
    r
}

/// Cube root of `y >= 0` with `+ - * / sqrt` only. The product of
/// `y^(1/4^k)` for `k = 1..6` is `y^(1/3)` short by a factor
/// `y^(-1/12288)` (within 6% for any finite `y`); four Newton steps on
/// `z^3 = y`, each squaring the relative error, then reach f64
/// precision.
#[cube]
pub fn cbrt_f64(y: f64) -> f64 {
    let rt_zero = y - y;
    let mut z = rt_zero;
    if y > 0.0 {
        let mut s = y;
        z = rt_zero + 1.0;
        let mut k = 0u32;
        while k < 6u32 {
            s = s.sqrt().sqrt();
            z *= s;
            k += 1u32;
        }
        let mut n = 0u32;
        while n < 4u32 {
            z = (2.0 * z + y / (z * z)) / 3.0;
            n += 1u32;
        }
    }
    z
}

/// Dominant (largest magnitude) real root of `x^3 + g x + h` with
/// `+ - * / sqrt` only: Newton from `-sign(R) 2 sqrt(Q)` for three real
/// roots (`Q = -g/3`, `R = h/2`), Cardano with [`cbrt_f64`] for one.
#[cube]
fn dominant_cubic_root(g: f64, h: f64) -> f64 {
    let q = -g / 3.0;
    let r = 0.5 * h;
    let rt_zero = g - g;
    let mut x = rt_zero;
    if r == 0.0 {
        if g <= 0.0 {
            x = (-g).sqrt();
        }
    } else {
        let qr = q / r;
        let mut kk = 1.0 - q * qr * qr;
        if fabs(q) >= fabs(r) {
            let rq = r / q;
            let mut sign_q = rt_zero + 1.0;
            if q < 0.0 {
                sign_q = rt_zero - 1.0;
            }
            kk = sign_q * (rq * rq / q - 1.0);
        }
        let mut sign_r = rt_zero + 1.0;
        if r < 0.0 {
            sign_r = rt_zero - 1.0;
        }
        if kk < 0.0 {
            x = -sign_r * 2.0 * q.sqrt();
            let mut n = 0u32;
            while n < 8u32 {
                let df = 3.0 * x * x + g;
                if df != 0.0 {
                    x -= (x * (x * x + g) + h) / df;
                }
                n += 1u32;
            }
        } else {
            let mut inner = fabs(r) * (1.0 + kk.sqrt());
            if fabs(q) >= fabs(r) {
                inner = fabs(r) + fabs(q).sqrt() * fabs(q) * kk.sqrt();
            }
            let big_a = -sign_r * cbrt_f64(inner);
            let mut big_b = rt_zero;
            if big_a != 0.0 {
                big_b = q / big_a;
            }
            x = big_a + big_b;
        }
    }
    x
}

/// The paper's Newton-Raphson polish of a root of `x^3 + g x + h`.
#[cube]
fn polish_cubic_root(x0: f64, g: f64, h: f64) -> f64 {
    let mut x = x0;
    let mut xsq = x * x;
    let mut f = x * (xsq + g) + h;
    let mut maxtt = fabs(x * xsq);
    if fabs(g * x) > maxtt {
        maxtt = fabs(g * x);
    }
    if fabs(h) > maxtt {
        maxtt = fabs(h);
    }
    if fabs(f) > 2.220_446_049_250_313e-16 * maxtt {
        let mut go = 1u32;
        let mut n = 0u32;
        while n < 8u32 && go == 1u32 {
            let df = 3.0 * xsq + g;
            if df == 0.0 {
                go = 0u32;
            } else {
                let x_old = x;
                x -= f / df;
                let f_old = f;
                xsq = x * x;
                f = x * (xsq + g) + h;
                if f == 0.0 {
                    go = 0u32;
                } else if fabs(f) >= fabs(f_old) {
                    x = x_old;
                    go = 0u32;
                }
            }
            n += 1u32;
        }
    }
    x
}

/// `phi0`, the dominant root of the resolvent cubic, from the quartic
/// shifted to cancel its `x^2` (or `x^3`) coefficient. `scaled == 1`
/// on the retry after the quartic was rescaled.
#[cube]
fn calc_phi0(a: f64, b: f64, c: f64, d: f64, scaled: u32) -> f64 {
    let diskr = 9.0 * a * a - 24.0 * b;
    let mut s = -a / 4.0;
    if diskr > 0.0 {
        let sq = diskr.sqrt();
        if a > 0.0 {
            s = -(2.0 * b) / (3.0 * a + sq);
        } else {
            s = -(2.0 * b) / (3.0 * a - sq);
        }
    }
    let aq = a + 4.0 * s;
    let bq = b + 3.0 * s * (a + 2.0 * s);
    let cq = c + s * (2.0 * b + s * (3.0 * a + 4.0 * s));
    let dq = d + s * (c + s * (b + s * (a + s)));
    let gg = bq * bq / 9.0;
    let hh = aq * cq;
    let g = hh - 4.0 * dq - 3.0 * gg;
    let h = (8.0 * dq + hh - 2.0 * gg) * bq / 3.0 - cq * cq - dq * aq * aq;
    let x = dominant_cubic_root(g, h);
    // If the root overflowed on the retry, rescale the cubic,
    // phi = CUBIC_RESCALE psi, forming g / CUBIC_RESCALE^2 and
    // h / CUBIC_RESCALE^3 without overflow.
    let mut root = x;
    let mut gx = g;
    let mut hx = h;
    let mut mult = x - x + 1.0;
    if finite(x) == 0u32 && scaled == 1u32 {
        let r = mult * CUBIC_RESCALE;
        let aqs = aq / r;
        let bqs = bq / r;
        let cqs = cq / r;
        let dqs = dq / r / r;
        let ggs = bqs * bqs / 9.0;
        let hhs = aqs * cqs;
        gx = hhs - 4.0 * dqs - 3.0 * ggs;
        hx = (8.0 * dqs + hhs - 2.0 * ggs) * bqs / 3.0 - cqs * (cqs / r) - (dq / r) * aqs * aqs;
        root = dominant_cubic_root(gx, hx);
        mult = r;
    }
    polish_cubic_root(root, gx, hx) * mult
}

/// `|computed - target| / |target|`, or `|computed|` for a zero target.
#[cube]
fn rel(computed: f64, target: f64) -> f64 {
    let mut e = fabs(computed);
    if target != 0.0 {
        e = fabs((computed - target) / target);
    }
    e
}

/// Residual of the LDL^T coefficients.
#[cube]
fn err_ldlt(b: f64, c: f64, d: f64, d2: f64, l1: f64, l2: f64, l3: f64) -> f64 {
    rel(d2 + l1 * l1 + 2.0 * l3, b)
        + rel(2.0 * d2 * l2 + 2.0 * l1 * l3, c)
        + rel(d2 * l2 * l2 + l3 * l3, d)
}

/// Residual of a real factorisation without the `d` equation.
#[cube]
fn err_abc(a: f64, b: f64, c: f64, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    rel(bq * cq + aq * dq, c) + rel(bq + aq * cq + dq, b) + rel(aq + cq, a)
}

/// Residual of a real factorisation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn err_abcd(a: f64, b: f64, c: f64, d: f64, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    rel(bq * dq, d) + err_abc(a, b, c, aq, bq, cq, dq)
}

/// `|z - target| / |target|` for complex `z = (re, im)`, or `|z|` for a
/// zero target.
#[cube]
fn crel(re: f64, im: f64, target: f64) -> f64 {
    let mut e = (re * re + im * im).sqrt();
    if target != 0.0 {
        let u = (re - target) / target;
        let v = im / target;
        e = (u * u + v * v).sqrt();
    }
    e
}

/// Residual of the complex factorisation `alpha1 = (ar, ai)`,
/// `beta1 = (br, bi)`, `alpha2 = (ar, -ai)`, `beta2 = (br, -bi)`: the
/// conjugate form both complex candidates of the solver take.
#[cube]
#[allow(clippy::too_many_arguments)]
fn err_abcd_conj(a: f64, b: f64, c: f64, d: f64, ar: f64, ai: f64, br: f64, bi: f64) -> f64 {
    // beta1 beta2 = |beta1|^2, real.
    let e0 = crel(br * br + bi * bi, 0.0, d);
    // beta1 alpha2 + alpha1 beta2 = 2 Re(beta1 conj(alpha1)), real.
    let e1 = crel(2.0 * (br * ar + bi * ai), 0.0, c);
    // beta1 + alpha1 alpha2 + beta2 = 2 Re(beta1) + |alpha1|^2, real.
    let e2 = crel(2.0 * br + ar * ar + ai * ai, 0.0, b);
    // alpha1 + alpha2 = 2 Re(alpha1), real.
    let e3 = crel(2.0 * ar, 0.0, a);
    e0 + e1 + e2 + e3
}

/// Recompute the smaller-magnitude beta from `beta1 beta2 = d`;
/// returns the new `beta1` (`which == 0`) or `beta2` (`which == 1`).
#[cube]
fn balanced(d: f64, bq: f64, dq: f64, which: u32) -> f64 {
    let mut b1 = bq;
    let mut b2 = dq;
    if fabs(dq) < fabs(bq) {
        b2 = d / bq;
    } else if fabs(dq) > fabs(bq) {
        b1 = d / dq;
    }
    let mut out = b1;
    if which == 1u32 {
        out = b2;
    }
    out
}

/// Newton-Raphson on the four factorisation equations with the paper's
/// closed-form inverse Jacobian; returns the refined factors.
#[cube]
#[allow(clippy::too_many_arguments)]
fn newton_refine(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    x0_in: f64,
    x1_in: f64,
    x2_in: f64,
    x3_in: f64,
    scale: f64,
    real: u32,
) -> QuarticFactors {
    let mut x0 = x0_in;
    let mut x1 = x1_in;
    let mut x2 = x2_in;
    let mut x3 = x3_in;
    let mut f0 = x1 * x3 - d;
    let mut f1 = x1 * x2 + x0 * x3 - c;
    let mut f2 = x1 + x0 * x2 + x3 - b;
    let mut f3 = x0 + x2 - a;
    let mut err = rel(f0 + d, d) + rel(f1 + c, c) + rel(f2 + b, b) + rel(f3 + a, a);
    let mut go = real;
    let mut n = 0u32;
    while n < 8u32 && go == 1u32 {
        let x02 = x0 - x2;
        let det = x1 * x1 + x1 * (-x2 * x02 - 2.0 * x3) + x3 * (x0 * x02 + x3);
        if det == 0.0 {
            go = 0u32;
        } else {
            let j00 = x02;
            let j01 = x3 - x1;
            let j02 = x1 * x2 - x0 * x3;
            let j03 = -x1 * j01 - x0 * j02;
            let j10 = x0 * j00 + j01;
            let j11 = -x1 * j00;
            let j12 = -x1 * j01;
            let j13 = -x1 * j02;
            let j23 = j02 * x2 + j01 * x3;
            let j30 = -x2 * j00 - j01;
            let j31 = j00 * x3;
            let j32 = x3 * j01;
            let j33 = x3 * j02;
            let o0 = x0;
            let o1 = x1;
            let o2 = x2;
            let o3 = x3;
            x0 -= (j00 * f0 + j01 * f1 + j02 * f2 + j03 * f3) / det;
            x1 -= (j10 * f0 + j11 * f1 + j12 * f2 + j13 * f3) / det;
            x2 -= (-j00 * f0 - j01 * f1 - j02 * f2 + j23 * f3) / det;
            x3 -= (j30 * f0 + j31 * f1 + j32 * f2 + j33 * f3) / det;
            f0 = x1 * x3 - d;
            f1 = x1 * x2 + x0 * x3 - c;
            f2 = x1 + x0 * x2 + x3 - b;
            f3 = x0 + x2 - a;
            let err_old = err;
            err = rel(f0 + d, d) + rel(f1 + c, c) + rel(f2 + b, b) + rel(f3 + a, a);
            if err == 0.0 {
                go = 0u32;
            } else if err >= err_old {
                x0 = o0;
                x1 = o1;
                x2 = o2;
                x3 = o3;
                go = 0u32;
            }
        }
        n += 1u32;
    }
    QuarticFactors {
        alpha1: x0,
        beta1: x1,
        alpha2: x2,
        beta2: x3,
        scale,
        real,
    }
}

/// Algorithm 1010's factorisation of `x^4 + a x^3 + b x^2 + c x + d`.
#[cube]
pub fn quartic_factors(a_in: f64, b_in: f64, c_in: f64, d_in: f64) -> QuarticFactors {
    let mut a = a_in;
    let mut b = b_in;
    let mut c = c_in;
    let mut d = d_in;
    // Coefficients are finite, so this is exactly zero.
    let rt_zero = a_in - a_in;
    let mut scale = rt_zero + 1.0;
    let mut phi0 = calc_phi0(a, b, c, d, 0u32);
    if finite(phi0) == 0u32 {
        scale = QUARTIC_RESCALE;
        let s2 = scale * scale;
        a /= scale;
        b /= s2;
        c /= s2 * scale;
        d /= s2 * s2;
        phi0 = calc_phi0(a, b, c, d, 1u32);
    }
    let l1 = a / 2.0;
    let l3 = b / 6.0 + phi0 / 2.0;
    let del2 = c - a * l3;
    let bl311 = 2.0 * b / 3.0 - phi0 - l1 * l1;
    let dml3l3 = d - l3 * l3;

    // (d2, l2) with the smallest LDL^T residual among eqs. (18)-(20).
    let mut have = 0u32;
    let mut best_err = rt_zero;
    let mut d2 = rt_zero;
    let mut l2 = rt_zero;
    if bl311 != 0.0 {
        d2 = bl311;
        l2 = del2 / (2.0 * bl311);
        best_err = err_ldlt(b, c, d, d2, l1, l2, l3);
        have = 1u32;
    }
    if del2 != 0.0 {
        let l2b = 2.0 * dml3l3 / del2;
        if l2b != 0.0 {
            let d2b = del2 / (2.0 * l2b);
            let e = err_ldlt(b, c, d, d2b, l1, l2b, l3);
            if have == 0u32 || e < best_err {
                d2 = d2b;
                l2 = l2b;
                best_err = e;
                have = 1u32;
            }
        }
        let e = err_ldlt(b, c, d, bl311, l1, l2b, l3);
        if have == 0u32 || e < best_err {
            d2 = bl311;
            l2 = l2b;
        }
    }

    // kind: 1 real factors, 2 complex, 0 d2 exactly zero.
    let mut kind = 0u32;
    let mut aq = rt_zero;
    let mut bq = rt_zero;
    let mut cq = rt_zero;
    let mut dq = rt_zero;
    // Complex factors in conjugate form: alpha1 = (zar, zai), beta1 = (zbr, zbi).
    let mut zar = rt_zero;
    let mut zai = rt_zero;
    let mut zbr = rt_zero;
    let mut zbi = rt_zero;
    if d2 < 0.0 {
        kind = 1u32;
        let gamma = (-d2).sqrt();
        aq = l1 + gamma;
        let bq0 = l3 + gamma * l2;
        let dq0 = l3 - gamma * l2;
        bq = balanced(d, bq0, dq0, 0u32);
        dq = balanced(d, bq0, dq0, 1u32);
        cq = l1 - gamma;
        if fabs(aq) < fabs(cq) {
            // Recompute alpha1 by the formula with the least residual.
            let mut pick = a - cq;
            let mut pick_err = rt_zero;
            let mut have_pick = 0u32;
            if dq != 0.0 {
                pick = (c - bq * cq) / dq;
                pick_err = err_abc(a, b, c, pick, bq, cq, dq);
                have_pick = 1u32;
            }
            if cq != 0.0 {
                let v = (b - dq - bq) / cq;
                let e = err_abc(a, b, c, v, bq, cq, dq);
                if have_pick == 0u32 || e < pick_err {
                    pick = v;
                    pick_err = e;
                    have_pick = 1u32;
                }
            }
            let v = a - cq;
            let e = err_abc(a, b, c, v, bq, cq, dq);
            if have_pick == 0u32 || e < pick_err {
                pick = v;
            }
            aq = pick;
        } else {
            // Recompute alpha2 likewise.
            let mut pick = a - aq;
            let mut pick_err = rt_zero;
            let mut have_pick = 0u32;
            if bq != 0.0 {
                pick = (c - aq * dq) / bq;
                pick_err = err_abc(a, b, c, aq, bq, pick, dq);
                have_pick = 1u32;
            }
            if aq != 0.0 {
                let v = (b - bq - dq) / aq;
                let e = err_abc(a, b, c, aq, bq, v, dq);
                if have_pick == 0u32 || e < pick_err {
                    pick = v;
                    pick_err = e;
                    have_pick = 1u32;
                }
            }
            let v = a - aq;
            let e = err_abc(a, b, c, aq, bq, v, dq);
            if have_pick == 0u32 || e < pick_err {
                pick = v;
            }
            cq = pick;
        }
    } else if d2 > 0.0 {
        kind = 2u32;
        let gamma = d2.sqrt();
        zar = l1;
        zai = gamma;
        zbr = l3;
        zbi = gamma * l2;
    }

    // d2 zero to machine precision: also try d2 = 0.
    let mut tiny = fabs(2.0 * b / 3.0);
    if fabs(phi0) > tiny {
        tiny = fabs(phi0);
    }
    if l1 * l1 > tiny {
        tiny = l1 * l1;
    }
    tiny *= 2.220_446_049_250_313e-16;
    if kind == 0u32 || fabs(d2) <= tiny {
        let d3 = d - l3 * l3;
        let mut err0 = rt_zero;
        if kind == 1u32 {
            err0 = err_abcd(a, b, c, d, aq, bq, cq, dq);
        } else if kind == 2u32 {
            err0 = err_abcd_conj(a, b, c, d, zar, zai, zbr, zbi);
        }
        if d3 <= 0.0 {
            let sq = (-d3).sqrt();
            let bq1 = balanced(d, l3 + sq, l3 - sq, 0u32);
            let dq1 = balanced(d, l3 + sq, l3 - sq, 1u32);
            let err1 = err_abcd(a, b, c, d, l1, bq1, l1, dq1);
            if kind == 0u32 || err1 < err0 {
                kind = 1u32;
                aq = l1;
                bq = bq1;
                cq = l1;
                dq = dq1;
            }
        } else {
            let sq = d3.sqrt();
            let err1 = err_abcd_conj(a, b, c, d, l1, 0.0, l3, sq);
            if kind == 0u32 || err1 < err0 {
                kind = 2u32;
            }
        }
    }

    let mut real = 0u32;
    if kind == 1u32 {
        real = 1u32;
    }
    newton_refine(a, b, c, d, aq, bq, cq, dq, scale, real)
}

/// Larger-magnitude real root of `x^2 + alpha x + beta`, or `NO_ROOT`
/// for a complex pair.
#[cube]
pub fn quadratic_root_max(alpha: f64, beta: f64) -> f64 {
    let disc = alpha * alpha - 4.0 * beta;
    let mut x = (alpha - alpha) + NO_ROOT;
    if disc >= 0.0 {
        if alpha >= 0.0 {
            x = (-alpha - disc.sqrt()) / 2.0;
        } else {
            x = (-alpha + disc.sqrt()) / 2.0;
        }
    }
    x
}

/// Smaller-magnitude real root of `x^2 + alpha x + beta` given the
/// larger one `x_max` (from `beta / x_max`, free of cancellation).
#[cube]
pub fn quadratic_root_min(beta: f64, x_max: f64) -> f64 {
    let mut x = x_max - x_max;
    if x_max != 0.0 {
        x = beta / x_max;
    }
    x
}

// ------------------------------ CPU twins ------------------------------

/// CPU twin of [`QuarticFactors`].
#[derive(Debug, Clone, Copy)]
pub struct QuarticFactorsCpu {
    pub alpha1: f64,
    pub beta1: f64,
    pub alpha2: f64,
    pub beta2: f64,
    pub scale: f64,
    pub real: bool,
}

fn finite_cpu(x: f64) -> bool {
    ((x.to_bits() >> 52) & 0x7FF) != 0x7FF
}

/// CPU twin of [`cbrt_f64`].
pub fn cbrt_f64_cpu(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    let mut s = y;
    let mut z = 1.0;
    for _ in 0..6 {
        s = s.sqrt().sqrt();
        z *= s;
    }
    for _ in 0..4 {
        z = (2.0 * z + y / (z * z)) / 3.0;
    }
    z
}

fn dominant_cubic_root_cpu(g: f64, h: f64) -> f64 {
    let q = -g / 3.0;
    let r = 0.5 * h;
    if r == 0.0 {
        return if g <= 0.0 { (-g).sqrt() } else { 0.0 };
    }
    let kk = if q.abs() < r.abs() {
        let qr = q / r;
        1.0 - q * qr * qr
    } else {
        let rq = r / q;
        let sign_q = if q < 0.0 { -1.0 } else { 1.0 };
        sign_q * (rq * rq / q - 1.0)
    };
    let sign_r = if r < 0.0 { -1.0 } else { 1.0 };
    if kk < 0.0 {
        let mut x = -sign_r * 2.0 * q.sqrt();
        for _ in 0..8 {
            let df = 3.0 * x * x + g;
            if df != 0.0 {
                x -= (x * (x * x + g) + h) / df;
            }
        }
        x
    } else {
        let big_a = if q.abs() < r.abs() {
            -sign_r * cbrt_f64_cpu(r.abs() * (1.0 + kk.sqrt()))
        } else {
            -sign_r * cbrt_f64_cpu(r.abs() + q.abs().sqrt() * q.abs() * kk.sqrt())
        };
        let big_b = if big_a != 0.0 { q / big_a } else { 0.0 };
        big_a + big_b
    }
}

fn polish_cubic_root_cpu(x0: f64, g: f64, h: f64) -> f64 {
    let mut x = x0;
    let mut xsq = x * x;
    let mut f = x * (xsq + g) + h;
    let maxtt = (x * xsq).abs().max((g * x).abs()).max(h.abs());
    if f.abs() > f64::EPSILON * maxtt {
        for _ in 0..8 {
            let df = 3.0 * xsq + g;
            if df == 0.0 {
                break;
            }
            let x_old = x;
            x -= f / df;
            let f_old = f;
            xsq = x * x;
            f = x * (xsq + g) + h;
            if f == 0.0 {
                break;
            }
            if f.abs() >= f_old.abs() {
                x = x_old;
                break;
            }
        }
    }
    x
}

fn calc_phi0_cpu(a: f64, b: f64, c: f64, d: f64, scaled: bool) -> f64 {
    let diskr = 9.0 * a * a - 24.0 * b;
    let mut s = -a / 4.0;
    if diskr > 0.0 {
        let sq = diskr.sqrt();
        s = if a > 0.0 {
            -2.0 * b / (3.0 * a + sq)
        } else {
            -2.0 * b / (3.0 * a - sq)
        };
    }
    let aq = a + 4.0 * s;
    let bq = b + 3.0 * s * (a + 2.0 * s);
    let cq = c + s * (2.0 * b + s * (3.0 * a + 4.0 * s));
    let dq = d + s * (c + s * (b + s * (a + s)));
    let gg = bq * bq / 9.0;
    let hh = aq * cq;
    let g = hh - 4.0 * dq - 3.0 * gg;
    let h = (8.0 * dq + hh - 2.0 * gg) * bq / 3.0 - cq * cq - dq * aq * aq;
    let x = dominant_cubic_root_cpu(g, h);
    if finite_cpu(x) || !scaled {
        return polish_cubic_root_cpu(x, g, h);
    }
    let r = CUBIC_RESCALE;
    let (aqs, bqs, cqs) = (aq / r, bq / r, cq / r);
    let dqs = dq / r / r;
    let ggs = bqs * bqs / 9.0;
    let hhs = aqs * cqs;
    let gs = hhs - 4.0 * dqs - 3.0 * ggs;
    let hs = (8.0 * dqs + hhs - 2.0 * ggs) * bqs / 3.0 - cqs * (cqs / r) - (dq / r) * aqs * aqs;
    polish_cubic_root_cpu(dominant_cubic_root_cpu(gs, hs), gs, hs) * r
}

fn rel_cpu(computed: f64, target: f64) -> f64 {
    if target != 0.0 {
        ((computed - target) / target).abs()
    } else {
        computed.abs()
    }
}

fn err_ldlt_cpu(b: f64, c: f64, d: f64, d2: f64, l1: f64, l2: f64, l3: f64) -> f64 {
    rel_cpu(d2 + l1 * l1 + 2.0 * l3, b)
        + rel_cpu(2.0 * d2 * l2 + 2.0 * l1 * l3, c)
        + rel_cpu(d2 * l2 * l2 + l3 * l3, d)
}

fn err_abc_cpu(a: f64, b: f64, c: f64, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    rel_cpu(bq * cq + aq * dq, c) + rel_cpu(bq + aq * cq + dq, b) + rel_cpu(aq + cq, a)
}

#[allow(clippy::too_many_arguments)]
fn err_abcd_cpu(a: f64, b: f64, c: f64, d: f64, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    rel_cpu(bq * dq, d) + err_abc_cpu(a, b, c, aq, bq, cq, dq)
}

fn crel_cpu(re: f64, im: f64, target: f64) -> f64 {
    if target != 0.0 {
        let u = (re - target) / target;
        let v = im / target;
        (u * u + v * v).sqrt()
    } else {
        (re * re + im * im).sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
fn err_abcd_conj_cpu(a: f64, b: f64, c: f64, d: f64, ar: f64, ai: f64, br: f64, bi: f64) -> f64 {
    crel_cpu(br * br + bi * bi, 0.0, d)
        + crel_cpu(2.0 * (br * ar + bi * ai), 0.0, c)
        + crel_cpu(2.0 * br + ar * ar + ai * ai, 0.0, b)
        + crel_cpu(2.0 * ar, 0.0, a)
}

fn balanced_cpu(d: f64, bq: f64, dq: f64) -> (f64, f64) {
    if dq.abs() < bq.abs() {
        (bq, d / bq)
    } else if dq.abs() > bq.abs() {
        (d / dq, dq)
    } else {
        (bq, dq)
    }
}

fn newton_refine_cpu(a: f64, b: f64, c: f64, d: f64, x: &mut [f64; 4]) {
    let res = |x: &[f64; 4]| {
        [
            x[1] * x[3] - d,
            x[1] * x[2] + x[0] * x[3] - c,
            x[1] + x[0] * x[2] + x[3] - b,
            x[0] + x[2] - a,
        ]
    };
    let size = |f: &[f64; 4]| {
        rel_cpu(f[0] + d, d) + rel_cpu(f[1] + c, c) + rel_cpu(f[2] + b, b) + rel_cpu(f[3] + a, a)
    };
    let mut f = res(x);
    let mut err = size(&f);
    for _ in 0..8 {
        let x02 = x[0] - x[2];
        let det = x[1] * x[1] + x[1] * (-x[2] * x02 - 2.0 * x[3]) + x[3] * (x[0] * x02 + x[3]);
        if det == 0.0 {
            break;
        }
        let j00 = x02;
        let j01 = x[3] - x[1];
        let j02 = x[1] * x[2] - x[0] * x[3];
        let j03 = -x[1] * j01 - x[0] * j02;
        let j10 = x[0] * j00 + j01;
        let j11 = -x[1] * j00;
        let j12 = -x[1] * j01;
        let j13 = -x[1] * j02;
        let j23 = j02 * x[2] + j01 * x[3];
        let j30 = -x[2] * j00 - j01;
        let j31 = j00 * x[3];
        let j32 = x[3] * j01;
        let j33 = x[3] * j02;
        let old = *x;
        x[0] -= (j00 * f[0] + j01 * f[1] + j02 * f[2] + j03 * f[3]) / det;
        x[1] -= (j10 * f[0] + j11 * f[1] + j12 * f[2] + j13 * f[3]) / det;
        x[2] -= (-j00 * f[0] - j01 * f[1] - j02 * f[2] + j23 * f[3]) / det;
        x[3] -= (j30 * f[0] + j31 * f[1] + j32 * f[2] + j33 * f[3]) / det;
        f = res(x);
        let err_old = err;
        err = size(&f);
        if err == 0.0 {
            break;
        }
        if err >= err_old {
            *x = old;
            break;
        }
    }
}

/// CPU twin of [`quartic_factors`].
pub fn quartic_factors_cpu(a_in: f64, b_in: f64, c_in: f64, d_in: f64) -> QuarticFactorsCpu {
    let (mut a, mut b, mut c, mut d) = (a_in, b_in, c_in, d_in);
    let mut scale = 1.0;
    let mut phi0 = calc_phi0_cpu(a, b, c, d, false);
    if !finite_cpu(phi0) {
        scale = QUARTIC_RESCALE;
        let s2 = scale * scale;
        a /= scale;
        b /= s2;
        c /= s2 * scale;
        d /= s2 * s2;
        phi0 = calc_phi0_cpu(a, b, c, d, true);
    }
    let l1 = a / 2.0;
    let l3 = b / 6.0 + phi0 / 2.0;
    let del2 = c - a * l3;
    let bl311 = 2.0 * b / 3.0 - phi0 - l1 * l1;
    let dml3l3 = d - l3 * l3;

    let mut best: Option<(f64, f64, f64)> = None;
    let mut consider = |d2: f64, l2: f64| {
        let e = err_ldlt_cpu(b, c, d, d2, l1, l2, l3);
        if best.is_none_or(|(be, _, _)| e < be) {
            best = Some((e, d2, l2));
        }
    };
    if bl311 != 0.0 {
        consider(bl311, del2 / (2.0 * bl311));
    }
    if del2 != 0.0 {
        let l2b = 2.0 * dml3l3 / del2;
        if l2b != 0.0 {
            consider(del2 / (2.0 * l2b), l2b);
        }
        consider(bl311, l2b);
    }
    let (d2, l2) = best.map_or((0.0, 0.0), |(_, d2, l2)| (d2, l2));

    // kind: 1 real factors, 2 complex, 0 d2 exactly zero.
    let mut kind = 0;
    let mut r = [0.0; 4];
    let mut z = [0.0; 4];
    if d2 < 0.0 {
        kind = 1;
        let gamma = (-d2).sqrt();
        let mut aq = l1 + gamma;
        let (bq, dq) = balanced_cpu(d, l3 + gamma * l2, l3 - gamma * l2);
        let mut cq = l1 - gamma;
        let pick = |cands: [Option<f64>; 3], err: &dyn Fn(f64) -> f64| {
            let mut best = (f64::NAN, f64::INFINITY);
            for v in cands.into_iter().flatten() {
                let e = err(v);
                if best.0.is_nan() || e < best.1 {
                    best = (v, e);
                }
            }
            best.0
        };
        if aq.abs() < cq.abs() {
            aq = pick(
                [
                    (dq != 0.0).then(|| (c - bq * cq) / dq),
                    (cq != 0.0).then(|| (b - dq - bq) / cq),
                    Some(a - cq),
                ],
                &|v| err_abc_cpu(a, b, c, v, bq, cq, dq),
            );
        } else {
            cq = pick(
                [
                    (bq != 0.0).then(|| (c - aq * dq) / bq),
                    (aq != 0.0).then(|| (b - bq - dq) / aq),
                    Some(a - aq),
                ],
                &|v| err_abc_cpu(a, b, c, aq, bq, v, dq),
            );
        }
        r = [aq, bq, cq, dq];
    } else if d2 > 0.0 {
        kind = 2;
        let gamma = d2.sqrt();
        z = [l1, gamma, l3, gamma * l2];
    }

    let tiny = f64::EPSILON * (2.0 * b / 3.0).abs().max(phi0.abs()).max(l1 * l1);
    if kind == 0 || d2.abs() <= tiny {
        let d3 = d - l3 * l3;
        let err0 = match kind {
            1 => err_abcd_cpu(a, b, c, d, r[0], r[1], r[2], r[3]),
            2 => err_abcd_conj_cpu(a, b, c, d, z[0], z[1], z[2], z[3]),
            _ => 0.0,
        };
        if d3 <= 0.0 {
            let sq = (-d3).sqrt();
            let (bq1, dq1) = balanced_cpu(d, l3 + sq, l3 - sq);
            let err1 = err_abcd_cpu(a, b, c, d, l1, bq1, l1, dq1);
            if kind == 0 || err1 < err0 {
                kind = 1;
                r = [l1, bq1, l1, dq1];
            }
        } else {
            let err1 = err_abcd_conj_cpu(a, b, c, d, l1, 0.0, l3, d3.sqrt());
            if kind == 0 || err1 < err0 {
                kind = 2;
            }
        }
    }

    if kind == 1 {
        newton_refine_cpu(a, b, c, d, &mut r);
        QuarticFactorsCpu {
            alpha1: r[0],
            beta1: r[1],
            alpha2: r[2],
            beta2: r[3],
            scale,
            real: true,
        }
    } else {
        QuarticFactorsCpu {
            alpha1: 0.0,
            beta1: 0.0,
            alpha2: 0.0,
            beta2: 0.0,
            scale,
            real: false,
        }
    }
}

/// CPU twin of [`quadratic_root_max`].
pub fn quadratic_root_max_cpu(alpha: f64, beta: f64) -> f64 {
    let disc = alpha * alpha - 4.0 * beta;
    if disc < 0.0 {
        NO_ROOT
    } else if alpha >= 0.0 {
        (-alpha - disc.sqrt()) / 2.0
    } else {
        (-alpha + disc.sqrt()) / 2.0
    }
}

/// CPU twin of [`quadratic_root_min`].
pub fn quadratic_root_min_cpu(beta: f64, x_max: f64) -> f64 {
    if x_max != 0.0 {
        beta / x_max
    } else {
        0.0
    }
}

/// The four real roots of `x^4 + a x^3 + b x^2 + c x + d` from the
/// CPU twins, `NO_ROOT` in the slots of complex roots: the layout the
/// test kernel writes.
pub fn quartic_roots_cpu(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
    let f = quartic_factors_cpu(a, b, c, d);
    let mut out = [NO_ROOT; 4];
    if f.real {
        for (i, (alpha, beta)) in [(f.alpha1, f.beta1), (f.alpha2, f.beta2)]
            .into_iter()
            .enumerate()
        {
            let x_max = quadratic_root_max_cpu(alpha, beta);
            if x_max < NO_ROOT {
                out[2 * i] = x_max * f.scale;
                out[2 * i + 1] = quadratic_root_min_cpu(beta, x_max) * f.scale;
            }
        }
    }
    out
}

/// Test kernel: the four root slots of a quartic per thread, as
/// [`quartic_roots_cpu`] lays them out.
#[cube(launch_unchecked)]
fn quartic_roots_kernel(coeffs: &[f64], roots: &mut [f64]) {
    if ABSOLUTE_POS * 4 >= roots.len() {
        terminate!();
    }
    let i4 = ABSOLUTE_POS * 4;
    let f = quartic_factors(coeffs[i4], coeffs[i4 + 1], coeffs[i4 + 2], coeffs[i4 + 3]);
    roots[i4] = NO_ROOT;
    roots[i4 + 1] = NO_ROOT;
    roots[i4 + 2] = NO_ROOT;
    roots[i4 + 3] = NO_ROOT;
    if f.real == 1u32 {
        let x1 = quadratic_root_max(f.alpha1, f.beta1);
        if x1 < NO_ROOT {
            roots[i4] = x1 * f.scale;
            roots[i4 + 1] = quadratic_root_min(f.beta1, x1) * f.scale;
        }
        let x2 = quadratic_root_max(f.alpha2, f.beta2);
        if x2 < NO_ROOT {
            roots[i4 + 2] = x2 * f.scale;
            roots[i4 + 3] = quadratic_root_min(f.beta2, x2) * f.scale;
        }
    }
}

use crate::GpuContext;

/// Run the quartic solver on the GPU for test/validation purposes.
/// `coeffs` is stride-4 `[a0, b0, c0, d0, a1, ...]` (monic quartics);
/// returns four root slots per quartic, `NO_ROOT` for complex roots.
pub fn run_quartic_roots(ctx: &GpuContext, coeffs: &[f64]) -> Vec<[f64; 4]> {
    assert!(coeffs.len().is_multiple_of(4));
    let n = coeffs.len() / 4;
    let client = ctx.client();
    let coeffs_h = client.create_from_slice(bytemuck::cast_slice(coeffs));
    let roots_h = client.empty(std::mem::size_of_val(coeffs));
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        quartic_roots_kernel::launch_unchecked(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(coeffs_h, coeffs.len()),
            BufferArg::from_raw_parts(roots_h.clone(), coeffs.len()),
        );
    }
    let flat: Vec<f64> = bytemuck::cast_slice(&client.read_one(roots_h).unwrap()).to_vec();
    flat.as_chunks::<4>().0.to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Monic quartic coefficients with the given roots (exact for
    /// small integers and powers of two).
    fn from_roots(r: [f64; 4]) -> [f64; 4] {
        let [r1, r2, r3, r4] = r;
        [
            -(r1 + r2 + r3 + r4),
            r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4,
            -(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4),
            r1 * r2 * r3 * r4,
        ]
    }

    fn sorted_real(slots: [f64; 4]) -> Vec<f64> {
        let mut v: Vec<f64> = slots.into_iter().filter(|r| *r < NO_ROOT).collect();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Hard quartics: roots across many decades, double and near-double
    /// roots, zero and near-zero roots, complex pairs and a torus-shaped
    /// quartic in cm. No overflowing quartic: the rescaling it triggers
    /// keys on an infinite `phi0`, and Vulkan does not promise to keep
    /// infinities, so the GPU can only be held to it on the CPU twin
    /// (`overflow_rescales_on_cpu_twin`). The torus never gets near it:
    /// its quartic is solved in units of the torus size.
    fn battery() -> Vec<[f64; 4]> {
        let eps = 2f64.powi(-20);
        vec![
            from_roots([1.0, 2.0, 3.0, 4.0]),
            from_roots([1e-6, 1e-3, 1.0, 1e3]),
            from_roots([1e-15, 1.0, 1e5, 1e15]),
            from_roots([-1e-5, 2e-2, -3e1, 4e4]),
            from_roots([1.0, 1.0, 3.0, 4.0]),
            from_roots([1.0, 1.0 + eps, 5.0, -7.0]),
            from_roots([0.0, 1.0, -2.0, 3.0]),
            from_roots([2f64.powi(-40), -(2f64.powi(-30)), 1.5, 2.5]),
            from_roots([625.1, 1560.5, 2439.5, 3374.9]),
            [0.0, -5.0, 0.0, 4.0],
            [0.0, -2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [4.0, 6.0, 4.0, -15.0],
            [3.0, 3.0, 3.0, 2.0],
        ]
    }

    #[test]
    fn overflow_rescales_on_cpu_twin() {
        let q = from_roots([1e70, 2e70, 3e70, 4e70]);
        let got = sorted_real(quartic_roots_cpu(q[0], q[1], q[2], q[3]));
        assert_eq!(got.len(), 4, "{got:?}");
        for (g, e) in got.iter().zip([1e70, 2e70, 3e70, 4e70]) {
            assert!(((g - e) / e).abs() < 1e-12, "{got:?}");
        }
    }

    #[test]
    fn cbrt_matches_std() {
        for y in [1e-300, 1e-20, 0.3, 1.0, 8.0, 27.0, 7.5e10, 1e300, 1.7e308] {
            let got = cbrt_f64_cpu(y);
            assert!(((got - y.cbrt()) / y.cbrt()).abs() < 4e-16, "{y}: {got}");
        }
        assert_eq!(cbrt_f64_cpu(0.0), 0.0);
    }

    #[test]
    fn dominant_cubic_root_branches() {
        // (x - 1)(x - 2)(x + 3): -3; (x + 1)(x + 2)(x - 3): 3;
        // x^3 + x + 2 = (x + 1)(x^2 - x + 2): -1; x^3 - 4x: 2.
        assert!((dominant_cubic_root_cpu(-7.0, 6.0) + 3.0).abs() < 1e-14);
        assert!((dominant_cubic_root_cpu(-7.0, -6.0) - 3.0).abs() < 1e-14);
        assert!((dominant_cubic_root_cpu(1.0, 2.0) + 1.0).abs() < 1e-14);
        assert_eq!(dominant_cubic_root_cpu(-4.0, 0.0), 2.0);
        // Two roots merging: (x - 2)(x + 1)^2 = x^3 - 3x - 2: 2.
        assert!((dominant_cubic_root_cpu(-3.0, -2.0) - 2.0).abs() < 1e-14);
    }

    /// The CPU twin agrees with the `yamc-geo` solver (which uses
    /// std `cbrt` for the resolvent) on the whole battery.
    #[test]
    fn cpu_twin_matches_geo_solver() {
        for q in battery() {
            let twin = sorted_real(quartic_roots_cpu(q[0], q[1], q[2], q[3]));
            let (roots, n) = yamc_geo::quartic_real_roots(q[0], q[1], q[2], q[3]);
            let mut geo = roots[..n].to_vec();
            geo.sort_by(f64::total_cmp);
            assert_eq!(twin.len(), geo.len(), "{q:?}: twin {twin:?} geo {geo:?}");
            for (t, g) in twin.iter().zip(&geo) {
                let tol = 1e-12 * g.abs().max(1e-300);
                assert!(
                    (t - g).abs() <= tol.max(1e-7 * g.abs()),
                    "{q:?}: {t} vs {g}"
                );
            }
        }
    }

    #[test]
    fn gpu_quartic_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };
        let quartics = battery();
        let coeffs: Vec<f64> = quartics.iter().flatten().copied().collect();
        let gpu = run_quartic_roots(&ctx, &coeffs);
        for (q, g) in quartics.iter().zip(&gpu) {
            let cpu = quartic_roots_cpu(q[0], q[1], q[2], q[3]);
            for (gr, cr) in g.iter().zip(&cpu) {
                let same = gr == cr || (gr - cr).abs() <= 1e-14 * cr.abs();
                assert!(same, "{q:?}: gpu {g:?} cpu {cpu:?}");
            }
        }
    }
}
