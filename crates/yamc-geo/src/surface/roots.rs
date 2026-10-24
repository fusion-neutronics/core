//! Real roots of a quartic, used by the ray-torus intersection.
//!
//! The solver is Algorithm 1010 of A. G. Orellana and C. De Michele,
//! "Algorithm 1010: Boosting Efficiency in Solving Quartic Equations
//! with No Compromise in Accuracy", ACM Transactions on Mathematical
//! Software 46(2), article 20 (2020), <https://doi.org/10.1145/3386241>.
//! Equation numbers below refer to that paper.
//!
//! The monic quartic `x^4 + a x^3 + b x^2 + c x + d` is written as the
//! LDL^T form `(x^2 + l1 x + l3)^2 + d2 (x + l2)^2`, where `l3` comes
//! from `phi0`, the dominant root of the depressed resolvent cubic. The
//! LDL^T form gives the factorisation
//! `(x^2 + alpha1 x + beta1)(x^2 + alpha2 x + beta2)`, whose
//! coefficients are polished by Newton-Raphson on the four equations
//! that tie them to `a, b, c, d` before the two quadratics are solved.
//! Unlike the closed-form Ferrari reduction this has no absolute
//! thresholds: every choice between equivalent formulas is made by
//! comparing relative residuals, so it keeps full accuracy when the
//! coefficients span many orders of magnitude, and for clustered,
//! double and near-zero roots.
//!
//! Only real roots are returned: callers here (ray intersections) have
//! no use for complex ones. When `d2 > 0` the quartic is
//! `(x^2 + l1 x + l3)^2 + d2 (x + l2)^2 > 0` and has no real roots, so
//! the complex branches of the paper are kept only to decide, by the
//! same residual comparison as the paper, which factorisation to trust
//! when `d2` is zero to machine precision.
//!
//! One departure from the paper: when the resolvent cubic has three
//! real roots its dominant root is found by Newton from a bound instead
//! of the trigonometric formula (see [`dominant_cubic_root`]). Both give
//! the same root after the paper's Newton polish of `phi0`; Newton is
//! cheaper here and needs no `acos`/`cos`, which the GPU twin of this
//! solver cannot use.

/// `f64::MAX^(1/4)` divided by the golden ratio: the factor the
/// quartic is rescaled by when `phi0` overflows.
const QUARTIC_RESCALE: f64 = 7.156_344_677_702_229e76;
/// `f64::MAX^(1/3)` divided by the golden ratio: the factor the
/// resolvent cubic is rescaled by when its root overflows.
const CUBIC_RESCALE: f64 = 3.488_062_137_979_395_6e102;

/// Real roots of `x^4 + a x^3 + b x^2 + c x + d = 0`, unsorted, with
/// multiplicity. The first `count` entries of the array are valid.
pub fn quartic_real_roots(a: f64, b: f64, c: f64, d: f64) -> ([f64; 4], usize) {
    let mut roots = [0.0; 4];
    let mut count = 0;
    if let Some(f) = quartic_factors(a, b, c, d) {
        for (alpha, beta) in [(f.alpha1, f.beta1), (f.alpha2, f.beta2)] {
            if let Some((x1, x2)) = quadratic_real_roots(alpha, beta) {
                roots[count] = x1 * f.scale;
                roots[count + 1] = x2 * f.scale;
                count += 2;
            }
        }
    }
    (roots, count)
}

/// Real factorisation `(x^2 + alpha1 x + beta1)(x^2 + alpha2 x + beta2)`
/// of a rescaled quartic: the roots of the original quartic are the
/// roots of the two quadratics multiplied by `scale`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuarticFactors {
    pub alpha1: f64,
    pub beta1: f64,
    pub alpha2: f64,
    pub beta2: f64,
    pub scale: f64,
}

/// Algorithm 1010's factorisation of `x^4 + a x^3 + b x^2 + c x + d`
/// into two real quadratics, or `None` when the factors it settles on
/// are complex (no real roots).
pub(crate) fn quartic_factors(a: f64, b: f64, c: f64, d: f64) -> Option<QuarticFactors> {
    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    let mut scale = 1.0;
    let mut phi0 = calc_phi0(a, b, c, d, false);
    // Simple polynomial rescaling (sec. 2.4): x -> x * scale.
    if !phi0.is_finite() {
        scale = QUARTIC_RESCALE;
        let s2 = scale * scale;
        a /= scale;
        b /= s2;
        c /= s2 * scale;
        d /= s2 * s2;
        phi0 = calc_phi0(a, b, c, d, true);
    }
    let w = Weights::new(a, b, c, d);
    let l1 = a / 2.0; // eq. (4)
    let l3 = b / 6.0 + phi0 / 2.0; // eq. (8)
    let del2 = c - a * l3; // after eq. (27)
    let bl311 = 2.0 * b / 3.0 - phi0 - l1 * l1; // d2 of eq. (20)
    let dml3l3 = d - l3 * l3; // d3 of eq. (15) with d2 = 0

    // Three ways to get (d2, l2), eqs. (18)-(20); keep the one with the
    // smallest relative residual.
    let mut best: Option<(f64, f64, f64)> = None;
    let mut consider = |d2: f64, l2: f64| {
        let err = err_ldlt(&w, d2, l1, l2, l3);
        if best.is_none_or(|(e, _, _)| err < e) {
            best = Some((err, d2, l2));
        }
    };
    if bl311 != 0.0 {
        consider(bl311, del2 / (2.0 * bl311));
    }
    if del2 != 0.0 {
        let l2 = 2.0 * dml3l3 / del2;
        if l2 != 0.0 {
            consider(del2 / (2.0 * l2), l2);
        }
        consider(bl311, l2);
    }
    let (d2, l2) = best.map_or((0.0, 0.0), |(_, d2, l2)| (d2, l2));

    let mut factors = if d2 < 0.0 {
        Factors::Real(real_factors_case1(&w, l1, l2, l3, d2))
    } else if d2 > 0.0 {
        // Case II, eqs. (53)-(56): complex conjugate factors.
        let gamma = d2.sqrt();
        Factors::Complex([
            (l1, gamma),
            (l3, gamma * l2),
            (l1, -gamma),
            (l3, -gamma * l2),
        ])
    } else {
        Factors::Zero
    };

    // d2 zero to machine precision: also try d2 = 0 (eqs. (63)-(68))
    // and keep whichever factorisation has the smaller residual.
    let tiny = f64::EPSILON * (2.0 * b / 3.0).abs().max(phi0.abs()).max(l1 * l1);
    if matches!(factors, Factors::Zero) || d2.abs() <= tiny {
        let d3 = d - l3 * l3;
        let err0 = match factors {
            Factors::Real(r) => err_abcd(&w, r[0], r[1], r[2], r[3]),
            Factors::Complex(z) => err_abcd_complex(&w, z),
            Factors::Zero => f64::INFINITY,
        };
        let (alt, err1) = if d3 <= 0.0 {
            let sq = (-d3).sqrt();
            let (bq, dq) = balance(d, l3 + sq, l3 - sq);
            (
                Factors::Real([l1, bq, l1, dq]),
                err_abcd(&w, l1, bq, l1, dq),
            )
        } else {
            let sq = d3.sqrt();
            let z = [(l1, 0.0), (l3, sq), (l1, 0.0), (l3, -sq)];
            (Factors::Complex(z), err_abcd_complex(&w, z))
        };
        if matches!(factors, Factors::Zero) || err1 < err0 {
            factors = alt;
        }
    }

    match factors {
        Factors::Real(mut r) => {
            newton_refine(&w, &mut r);
            Some(QuarticFactors {
                alpha1: r[0],
                beta1: r[1],
                alpha2: r[2],
                beta2: r[3],
                scale,
            })
        }
        Factors::Complex(_) | Factors::Zero => None,
    }
}

/// Real roots of `x^2 + alpha x + beta`, the larger-magnitude root
/// first, or `None` for a complex pair. The smaller root comes from
/// `beta / x_max`, which avoids cancellation.
pub(crate) fn quadratic_real_roots(alpha: f64, beta: f64) -> Option<(f64, f64)> {
    let disc = alpha * alpha - 4.0 * beta;
    if disc < 0.0 {
        return None;
    }
    let div = if alpha >= 0.0 {
        -alpha - disc.sqrt()
    } else {
        -alpha + disc.sqrt()
    };
    let x_max = div / 2.0;
    let x_min = if x_max == 0.0 { 0.0 } else { beta / x_max };
    Some((x_max, x_min))
}

/// The factorisation candidates while the solver decides between them:
/// real `[alpha1, beta1, alpha2, beta2]`, complex `(re, im)` pairs in
/// the same order, or `d2` exactly zero.
enum Factors {
    Real([f64; 4]),
    Complex([(f64, f64); 4]),
    Zero,
}

/// Case I (`d2 < 0`), eqs. (37)-(53): real factors, with `beta1 beta2
/// = d` enforced on the smaller of the betas and the smaller of the
/// alphas recomputed by the formula with the least residual.
fn real_factors_case1(w: &Weights, l1: f64, l2: f64, l3: f64, d2: f64) -> [f64; 4] {
    let (a, b, c, d) = (w.a, w.b, w.c, w.d);
    let gamma = (-d2).sqrt();
    let mut aq = l1 + gamma;
    let (bq, dq) = balance(d, l3 + gamma * l2, l3 - gamma * l2);
    let mut cq = l1 - gamma;
    if aq.abs() < cq.abs() {
        // eqs. (47)
        let candidates = [
            (dq != 0.0).then(|| (c - bq * cq) / dq),
            (cq != 0.0).then(|| (b - dq - bq) / cq),
            Some(a - cq),
        ];
        aq = least_residual(&candidates, |v| err_abc(w, v, bq, cq, dq));
    } else {
        // eqs. (53)
        let candidates = [
            (bq != 0.0).then(|| (c - aq * dq) / bq),
            (aq != 0.0).then(|| (b - bq - dq) / aq),
            Some(a - aq),
        ];
        cq = least_residual(&candidates, |v| err_abc(w, aq, bq, v, dq));
    }
    [aq, bq, cq, dq]
}

/// The candidate with the smallest residual (the first on ties).
fn least_residual(candidates: &[Option<f64>], err: impl Fn(f64) -> f64) -> f64 {
    let mut pick = (f64::NAN, f64::INFINITY);
    for &v in candidates.iter().flatten() {
        let e = err(v);
        if pick.0.is_nan() || e < pick.1 {
            pick = (v, e);
        }
    }
    pick.0
}

/// Recompute the smaller-magnitude of `beta1`, `beta2` from
/// `beta1 beta2 = d`, which is accurate where their difference is not.
fn balance(d: f64, bq: f64, dq: f64) -> (f64, f64) {
    if dq.abs() < bq.abs() {
        (bq, d / bq)
    } else if dq.abs() > bq.abs() {
        (d / dq, dq)
    } else {
        (bq, dq)
    }
}

/// The quartic's coefficients with the weights that turn an error in
/// each into a relative one (1 for a zero coefficient, where the error
/// stays absolute), so the residuals below need no division.
struct Weights {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    wa: f64,
    wb: f64,
    wc: f64,
    wd: f64,
}

impl Weights {
    fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        let weight = |t: f64| if t == 0.0 { 1.0 } else { 1.0 / t.abs() };
        Weights {
            a,
            b,
            c,
            d,
            wa: weight(a),
            wb: weight(b),
            wc: weight(c),
            wd: weight(d),
        }
    }
}

/// Residual of the LDL^T coefficients, eq. (21) and following.
fn err_ldlt(w: &Weights, d2: f64, l1: f64, l2: f64, l3: f64) -> f64 {
    (d2 + l1 * l1 + 2.0 * l3 - w.b).abs() * w.wb
        + (2.0 * d2 * l2 + 2.0 * l1 * l3 - w.c).abs() * w.wc
        + (d2 * l2 * l2 + l3 * l3 - w.d).abs() * w.wd
}

/// Residual of a real factorisation without the `d` equation,
/// eqs. (48)-(51).
fn err_abc(w: &Weights, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    (bq * cq + aq * dq - w.c).abs() * w.wc
        + (bq + aq * cq + dq - w.b).abs() * w.wb
        + (aq + cq - w.a).abs() * w.wa
}

/// Residual of a real factorisation, eq. (68).
fn err_abcd(w: &Weights, aq: f64, bq: f64, cq: f64, dq: f64) -> f64 {
    (bq * dq - w.d).abs() * w.wd + err_abc(w, aq, bq, cq, dq)
}

/// Residual of a complex factorisation `[alpha1, beta1, alpha2,
/// beta2]` as `(re, im)` pairs, eqs. (68)-(69).
fn err_abcd_complex(w: &Weights, z: [(f64, f64); 4]) -> f64 {
    let mul = |p: (f64, f64), q: (f64, f64)| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0);
    let add = |p: (f64, f64), q: (f64, f64)| (p.0 + q.0, p.1 + q.1);
    let crel = |p: (f64, f64), target: f64, weight: f64| (p.0 - target).hypot(p.1) * weight;
    let [aq, bq, cq, dq] = z;
    crel(mul(bq, dq), w.d, w.wd)
        + crel(add(mul(bq, cq), mul(aq, dq)), w.c, w.wc)
        + crel(add(add(bq, mul(aq, cq)), dq), w.b, w.wb)
        + crel(add(aq, cq), w.a, w.wa)
}

/// Newton-Raphson on the four equations `alpha1 + alpha2 = a`,
/// `beta1 + alpha1 alpha2 + beta2 = b`, `beta1 alpha2 + alpha1 beta2 = c`,
/// `beta1 beta2 = d` (sec. 2.3), with the paper's closed-form inverse
/// Jacobian. Stops after 8 steps, at an exact solution, or as soon as
/// a step fails to reduce the relative residual (and undoes that step).
fn newton_refine(w: &Weights, x: &mut [f64; 4]) {
    let (a, b, c, d) = (w.a, w.b, w.c, w.d);
    let residual = |x: &[f64; 4]| {
        [
            x[1] * x[3] - d,
            x[1] * x[2] + x[0] * x[3] - c,
            x[1] + x[0] * x[2] + x[3] - b,
            x[0] + x[2] - a,
        ]
    };
    let size = |f: &[f64; 4]| -> f64 {
        f[0].abs() * w.wd + f[1].abs() * w.wc + f[2].abs() * w.wb + f[3].abs() * w.wa
    };
    let mut f = residual(x);
    let mut err = size(&f);
    for _ in 0..8 {
        let x02 = x[0] - x[2];
        let det = x[1] * x[1] + x[1] * (-x[2] * x02 - 2.0 * x[3]) + x[3] * (x[0] * x02 + x[3]);
        if det == 0.0 {
            break;
        }
        let mut j = [[0.0; 4]; 4];
        j[0][0] = x02;
        j[0][1] = x[3] - x[1];
        j[0][2] = x[1] * x[2] - x[0] * x[3];
        j[0][3] = -x[1] * j[0][1] - x[0] * j[0][2];
        j[1][0] = x[0] * j[0][0] + j[0][1];
        j[1][1] = -x[1] * j[0][0];
        j[1][2] = -x[1] * j[0][1];
        j[1][3] = -x[1] * j[0][2];
        j[2][0] = -j[0][0];
        j[2][1] = -j[0][1];
        j[2][2] = -j[0][2];
        j[2][3] = j[0][2] * x[2] + j[0][1] * x[3];
        j[3][0] = -x[2] * j[0][0] - j[0][1];
        j[3][1] = j[0][0] * x[3];
        j[3][2] = x[3] * j[0][1];
        j[3][3] = x[3] * j[0][2];
        let old = *x;
        let inv_det = 1.0 / det;
        for k in 0..4 {
            let dx = j[k][0] * f[0] + j[k][1] * f[1] + j[k][2] * f[2] + j[k][3] * f[3];
            x[k] -= dx * inv_det;
        }
        f = residual(x);
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

/// `phi0`, the dominant root of the resolvent cubic, eqs. (79)-(88)
/// and sec. 2.2. The quartic is first shifted (`x -> x + s`, which
/// leaves `phi0` unchanged) so that its `x^2` coefficient (or failing
/// that its `x^3` coefficient) vanishes, which limits cancellation in
/// the depressed cubic `phi^3 + g phi + h`. The analytic root is then
/// polished by Newton-Raphson. `scaled` is true on the retry after the
/// quartic has been rescaled, when the cubic itself is rescaled too if
/// its root still overflows.
fn calc_phi0(a: f64, b: f64, c: f64, d: f64, scaled: bool) -> f64 {
    let diskr = 9.0 * a * a - 24.0 * b; // eq. (87)
    let s = if diskr > 0.0 {
        // eq. (88)
        let sq = diskr.sqrt();
        if a > 0.0 {
            -2.0 * b / (3.0 * a + sq)
        } else {
            -2.0 * b / (3.0 * a - sq)
        }
    } else {
        -a / 4.0
    };
    // eqs. (83): coefficients of the shifted quartic.
    let aq = a + 4.0 * s;
    let bq = b + 3.0 * s * (a + 2.0 * s);
    let cq = c + s * (2.0 * b + s * (3.0 * a + 4.0 * s));
    let dq = d + s * (c + s * (b + s * (a + s)));
    let gg = bq * bq / 9.0;
    let hh = aq * cq;
    let g = hh - 4.0 * dq - 3.0 * gg; // eq. (85)
    let h = (8.0 * dq + hh - 2.0 * gg) * bq / 3.0 - cq * cq - dq * aq * aq; // eq. (86)
    let x = dominant_cubic_root(g, h);
    if x.is_finite() || !scaled {
        return polish_cubic_root(x, g, h);
    }
    // Rescale the cubic, phi = CUBIC_RESCALE * psi: the depressed cubic
    // in psi has g / CUBIC_RESCALE^2 and h / CUBIC_RESCALE^3, formed
    // here without overflow.
    let r = CUBIC_RESCALE;
    let (aqs, bqs, cqs) = (aq / r, bq / r, cq / r);
    let dqs = dq / r / r;
    let ggs = bqs * bqs / 9.0;
    let hhs = aqs * cqs;
    let gs = hhs - 4.0 * dqs - 3.0 * ggs;
    let hs = (8.0 * dqs + hhs - 2.0 * ggs) * bqs / 3.0 - cqs * (cqs / r) - (dq / r) * aqs * aqs;
    polish_cubic_root(dominant_cubic_root(gs, hs), gs, hs) * r
}

/// Newton-Raphson polish of a root `x` of `x^3 + g x + h` (end of
/// sec. 2.2): at most 8 steps, stopping at an exact root or as soon as
/// a step fails to reduce `|f|` (and undoing that step).
fn polish_cubic_root(x: f64, g: f64, h: f64) -> f64 {
    let mut x = x;
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

/// Dominant (largest magnitude) real root of `x^3 + g x + h` (sec. 2.2),
/// arranged so that neither `Q^3` nor `R^2` is formed when they would
/// overflow: Cardano's formula for one real root, and in place of the
/// paper's trigonometric formula for three, Newton from a bound.
fn dominant_cubic_root(g: f64, h: f64) -> f64 {
    let q = -g / 3.0;
    let r = 0.5 * h;
    if r == 0.0 {
        return if g <= 0.0 { (-g).sqrt() } else { 0.0 };
    }
    // kk < 0 exactly when R^2 < Q^3 (three real roots); kk scales
    // R^2 - Q^3 by R^2 or |Q|^3, whichever is larger.
    let kk = if q.abs() < r.abs() {
        let qr = q / r;
        1.0 - q * qr * qr
    } else {
        let rq = r / q;
        q.signum() * (rq * rq / q - 1.0)
    };
    if kk < 0.0 {
        // Three real roots. The dominant one has the sign of -R and
        // magnitude in [sqrt(3 Q), 2 sqrt(Q)], where the cubic is convex
        // and increasing away from zero, so Newton from -sign(R) 2 sqrt(Q)
        // descends monotonically onto it; 8 steps reach f64 precision from
        // the worst start. This gives the root of the trigonometric
        // formula without acos and cos, at less cost (and is the form the
        // GPU solver can use).
        let mut x = -r.signum() * 2.0 * q.sqrt();
        for _ in 0..8 {
            let df = 3.0 * x * x + g;
            if df != 0.0 {
                x -= (x * (x * x + g) + h) / df;
            }
        }
        x
    } else {
        let big_a = if q.abs() < r.abs() {
            -r.signum() * (r.abs() * (1.0 + kk.sqrt())).cbrt()
        } else {
            -r.signum() * (r.abs() + q.abs().sqrt() * q.abs() * kk.sqrt()).cbrt()
        };
        let big_b = if big_a == 0.0 { 0.0 } else { q / big_a };
        big_a + big_b
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Coefficients `(a, b, c, d)` of the monic quartic with the given
    /// roots. With roots that are small integers or powers of two the
    /// expansion is exact in f64, so the roots are exact references.
    fn from_roots(r: [f64; 4]) -> (f64, f64, f64, f64) {
        let [r1, r2, r3, r4] = r;
        let a = -(r1 + r2 + r3 + r4);
        let b = r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4;
        let c = -(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4);
        let d = r1 * r2 * r3 * r4;
        (a, b, c, d)
    }

    fn sorted_roots(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
        let (roots, n) = quartic_real_roots(a, b, c, d);
        let mut v = roots[..n].to_vec();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Every expected root is matched, in order, to relative accuracy
    /// `rel_tol` (absolute for a zero root).
    fn assert_roots(got: &[f64], expected: &[f64], rel_tol: f64) {
        assert_eq!(
            got.len(),
            expected.len(),
            "got {got:?}, expected {expected:?}"
        );
        for (g, e) in got.iter().zip(expected) {
            let err = if *e == 0.0 {
                g.abs()
            } else {
                ((g - e) / e).abs()
            };
            assert!(
                err <= rel_tol,
                "root {g:e} vs {e:e} (rel err {err:e}); all {got:?}"
            );
        }
    }

    #[test]
    fn four_simple_roots() {
        let (a, b, c, d) = from_roots([1.0, 2.0, 3.0, 4.0]);
        assert_roots(&sorted_roots(a, b, c, d), &[1.0, 2.0, 3.0, 4.0], 1e-14);
    }

    #[test]
    fn biquadratic() {
        // (x^2 - 1)(x^2 - 4)
        assert_roots(
            &sorted_roots(0.0, -5.0, 0.0, 4.0),
            &[-2.0, -1.0, 1.0, 2.0],
            1e-15,
        );
    }

    #[test]
    fn no_real_roots() {
        // x^4 + 1 and (x^2 + 1)(x^2 + 4).
        assert!(sorted_roots(0.0, 0.0, 0.0, 1.0).is_empty());
        assert!(sorted_roots(0.0, 5.0, 0.0, 4.0).is_empty());
    }

    #[test]
    fn two_real_two_complex() {
        // (x - 1)(x + 3)(x^2 + 2x + 5): a = 4, b = 6, c = 4, d = -15.
        assert_roots(&sorted_roots(4.0, 6.0, 4.0, -15.0), &[-3.0, 1.0], 1e-15);
    }

    #[test]
    fn roots_spanning_many_orders_of_magnitude() {
        // Every root is recovered to near machine precision RELATIVE to
        // itself, although the coefficients span up to 60 decades.
        for roots in [
            [1e-6, 1e-3, 1.0, 1e3],
            [-1e-5, 2e-2, -3e1, 4e4],
            [2f64.powi(-20), 2f64.powi(-7), 2f64.powi(6), 2f64.powi(19)],
            [1e-15, 1.0, 1e5, 1e15],
        ] {
            let (a, b, c, d) = from_roots(roots);
            let mut expected = roots.to_vec();
            expected.sort_by(f64::total_cmp);
            assert_roots(&sorted_roots(a, b, c, d), &expected, 1e-12);
        }
    }

    #[test]
    fn double_roots() {
        // (x - 1)^2 (x - 3)(x - 4): a double root next to simple ones.
        let (a, b, c, d) = from_roots([1.0, 1.0, 3.0, 4.0]);
        assert_roots(&sorted_roots(a, b, c, d), &[1.0, 1.0, 3.0, 4.0], 1e-7);
        // (x^2 - 1)^2: two double roots.
        assert_roots(
            &sorted_roots(0.0, -2.0, 0.0, 1.0),
            &[-1.0, -1.0, 1.0, 1.0],
            1e-7,
        );
        // (x - 2)^2 (x^2 + 1): a double root and a complex pair. A double
        // root may come back as a complex pair, or as a close real pair.
        let got = sorted_roots(-4.0, 5.0, -4.0, 4.0);
        assert!(got.is_empty() || (got.len() == 2 && got.iter().all(|r| (r - 2.0).abs() < 1e-7)));
    }

    #[test]
    fn near_double_roots() {
        // Two roots 2^-20 apart (1e-6 relative) are still both found.
        let eps = 2f64.powi(-20);
        let (a, b, c, d) = from_roots([1.0, 1.0 + eps, 5.0, -7.0]);
        let got = sorted_roots(a, b, c, d);
        assert_roots(&got, &[-7.0, 1.0, 1.0 + eps, 5.0], 1e-9);
    }

    #[test]
    fn zero_and_near_zero_roots() {
        // An exact zero root (d = 0) comes back as zero or within
        // round-off of the other roots' scale, with the rest intact.
        let (a, b, c, d) = from_roots([0.0, 1.0, -2.0, 3.0]);
        assert_eq!(d, 0.0);
        assert_roots(&sorted_roots(a, b, c, d), &[-2.0, 0.0, 1.0, 3.0], 1e-15);
        // Near-zero roots keep their relative accuracy.
        let (a, b, c, d) = from_roots([2f64.powi(-40), -(2f64.powi(-30)), 1.5, 2.5]);
        assert_roots(
            &sorted_roots(a, b, c, d),
            &[-(2f64.powi(-30)), 2f64.powi(-40), 1.5, 2.5],
            1e-12,
        );
    }

    #[test]
    fn torus_shaped_quartic() {
        // The quartic of the ray x = 2000 - t, y = 0, z = 0 against the
        // z-torus with major radius 907.2 and minor radius 467.7, kept in
        // cm (coefficients from 1 to 1e13): crossings at
        // 2000 - 1374.9, 2000 - 439.5, 2000 + 439.5 and 2000 + 1374.9.
        let (big_r, r) = (907.2, 467.7);
        let roots = [
            2000.0 - (big_r + r),
            2000.0 - (big_r - r),
            2000.0 + (big_r - r),
            2000.0 + (big_r + r),
        ];
        let (a, b, c, d) = from_roots(roots);
        assert_roots(&sorted_roots(a, b, c, d), &roots, 1e-13);
    }

    #[test]
    fn huge_coefficients_rescale() {
        // Roots near 1e70 overflow the resolvent cubic (h ~ 1e420): the rescaling path recovers them.
        let roots = [1e70, 2e70, 3e70, 4e70];
        let (a, b, c, d) = from_roots(roots);
        assert!(d.is_finite());
        assert_roots(&sorted_roots(a, b, c, d), &roots, 1e-12);
    }

    #[test]
    fn quadratic_roots_avoid_cancellation() {
        // x^2 - 1e8 x + 1: roots 1e8 and 1e-8, the small one exact to
        // relative precision.
        let (big, small) = quadratic_real_roots(-1e8, 1.0).unwrap();
        assert!((big - 1e8).abs() / 1e8 < 1e-15);
        assert!((small - 1e-8).abs() / 1e-8 < 1e-15);
        assert!(quadratic_real_roots(0.0, 1.0).is_none());
    }

    #[test]
    fn dominant_cubic_root_branches() {
        // Three real roots, (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6: -3.
        assert!((dominant_cubic_root(-7.0, 6.0) + 3.0).abs() < 1e-14);
        // (x + 1)(x + 2)(x - 3) = x^3 - 7x - 6: 3.
        assert!((dominant_cubic_root(-7.0, -6.0) - 3.0).abs() < 1e-14);
        // One real root, x^3 + x + 2 = (x + 1)(x^2 - x + 2): -1.
        assert!((dominant_cubic_root(1.0, 2.0) + 1.0).abs() < 1e-14);
        // h = 0: x^3 - 4x has roots 0, +-2; +2 is returned.
        assert_eq!(dominant_cubic_root(-4.0, 0.0), 2.0);
    }
}
