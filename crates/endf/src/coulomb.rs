//! The Coulomb penetrability `P_l(rho, eta) = rho / (F_l^2 + G_l^2)` of a
//! charged-particle channel, for R-matrix reconstruction.
//!
//! At or beyond the classical turning point `rho_t = eta + sqrt(eta^2 +
//! l(l+1))` Steed's method gives it directly: continued fraction CF2 gives
//! `p + iq = H'/H` for `H = G + iF`, and `P = rho q`, since `q = 1/|H|^2` by
//! the Wronskian (Barnett, Comput. Phys. Commun. 21 (1981) 297; Thompson and
//! Barnett, J. Comput. Phys. 64 (1986) 490).
//!
//! Inside the barrier CF2 does not converge. There `G` grows towards the
//! origin, so its logarithmic derivative `g = G'/G` is integrated inward
//! from the turning point (the Riccati equation, stable in the direction the
//! solution grows), with `ln G` alongside, and `F` follows from CF1's
//! `f = F'/F` and the Wronskian `F'G - FG' = 1`: `F = 1/(G (f - g))`. Kept in
//! logarithms, a channel deep under its barrier gives a penetrability that
//! underflows to zero instead of overflowing on the way.

/// `f = F'_l/F_l` at `x`: CF1, `S_(l+1) - R_(l+1)^2 / (T_(l+1) - R_(l+2)^2 /
/// (T_(l+2) - ...))` with `S_k = k/x + eta/k`, `R_k^2 = 1 + eta^2/k^2` and
/// `T_k = S_k + S_(k+1)`, by the modified Lentz method.
fn cf1(l: usize, x: f64, eta: f64) -> f64 {
    let s = |k: f64| k / x + eta / k;
    let r2 = |k: f64| 1.0 + (eta / k).powi(2);
    const TINY: f64 = 1e-300;
    let mut k = l as f64 + 1.0;
    let mut f = s(k);
    if f == 0.0 {
        f = TINY;
    }
    let (mut c, mut d) = (f, 0.0);
    for _ in 0..1_000_000 {
        let a = -r2(k);
        let b = s(k) + s(k + 1.0);
        d = b + a * d;
        if d == 0.0 {
            d = TINY;
        }
        c = b + a / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        k += 1.0;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    f
}

/// `(p, q)` with `p + iq = H'_l/H_l` at `x`, `H = G + iF`: CF2,
/// `i(1 - eta/x) + (i/x) ab / (2(x - eta + i) + (a+1)(b+1) / (2(x - eta +
/// 2i) + ...))` with `a = i eta - l` and `b = i eta + l + 1`, by the
/// modified Lentz method in complex arithmetic. Converges for `x` at or
/// beyond the turning point.
fn cf2(l: usize, x: f64, eta: f64) -> (f64, f64) {
    type C = (f64, f64);
    let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let div = |a: C, b: C| {
        let n = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / n, (a.1 * b.0 - a.0 * b.1) / n)
    };
    let add = |a: C, b: C| (a.0 + b.0, a.1 + b.1);
    let small = |z: C| {
        if z.0 == 0.0 && z.1 == 0.0 {
            (1e-300, 0.0)
        } else {
            z
        }
    };
    let lf = l as f64;
    // K = alpha_1 / (beta_1 + alpha_2 / (beta_2 + ...)),
    // alpha_k = (a + k - 1)(b + k - 1), beta_k = 2(x - eta + k i).
    let alpha = |k: f64| mul((k - 1.0 - lf, eta), (k + lf, eta));
    let beta = |k: f64| (2.0 * (x - eta), 2.0 * k);
    let a1 = alpha(1.0);
    let k_value = if a1 == (0.0, 0.0) {
        (0.0, 0.0)
    } else {
        // Lentz on the fraction beta_1 + alpha_2/(beta_2 + ...), then
        // K = alpha_1 / that.
        let mut f = small(beta(1.0));
        let (mut c, mut d) = (f, (0.0, 0.0));
        let mut k = 2.0;
        for _ in 0..10_000_000 {
            let a = alpha(k);
            let b = beta(k);
            d = small(add(b, mul(a, d)));
            c = small(add(b, div(a, c)));
            d = div((1.0, 0.0), d);
            let delta = mul(c, d);
            f = mul(f, delta);
            k += 1.0;
            if (delta.0 - 1.0).abs() + delta.1.abs() < 1e-16 || a == (0.0, 0.0) {
                break;
            }
        }
        div(a1, f)
    };
    // p + iq = i(1 - eta/x) + (i/x) K
    let ik = (-k_value.1 / x, k_value.0 / x);
    (ik.0, 1.0 - eta / x + ik.1)
}

/// The Coulomb penetrability `rho / (F_l^2 + G_l^2)` for `rho > 0` and
/// `eta >= 0`; zero for `rho <= 0`. With `eta = 0` it is the hard-sphere
/// penetrability of any `l`.
pub(crate) fn penetrability(l: usize, rho: f64, eta: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }
    let ll = (l * (l + 1)) as f64;
    let turning = eta + (eta * eta + ll).sqrt();
    if rho >= turning {
        return rho * cf2(l, rho, eta).1;
    }
    // A barrier this thick leaves nothing to compute: the WKB exponent
    // 2 int sqrt(V) dx past ~700 underflows exp anyway.
    let v = |x: f64| ll / (x * x) + 2.0 * eta / x - 1.0;
    let (steps, mut wkb) = (2000usize, 0.0);
    let (a, b) = (rho.ln(), turning.ln());
    for i in 0..steps {
        let t = a + (b - a) * (i as f64 + 0.5) / steps as f64;
        let x = t.exp();
        wkb += v(x).max(0.0).sqrt() * x * (b - a) / steps as f64;
    }
    if 2.0 * wkb > 1400.0 {
        return 0.0;
    }
    // At the turning point: H'/H from CF2 and F'/F from CF1 give G and G'.
    let (p, q) = cf2(l, turning, eta);
    let f_m = cf1(l, turning, eta);
    let gamma = (f_m - p) / q;
    // |G|^2 = gamma^2 F^2 with F^2 = 1/(q(1 + gamma^2)).
    let ln_g_m = 0.5 * (gamma * gamma / (q * (1.0 + gamma * gamma))).ln();
    let g_m = p - q / gamma;
    // Inward, in t = ln x: dg/dt = x (V - g^2), d(ln G)/dt = x g.
    let rhs = |t: f64, y: [f64; 2]| -> [f64; 2] {
        let x = t.exp();
        [x * (v(x) - y[0] * y[0]), x * y[0]]
    };
    let mut y = [g_m, ln_g_m];
    let mut t = turning.ln();
    let end = rho.ln();
    let mut h = -(t - end).abs().max(1e-12) / 64.0;
    // Dormand-Prince 5(4), adaptive, to a relative tolerance of 1e-12.
    const A: [[f64; 6]; 6] = [
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];
    const C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
    const E: [f64; 7] = [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ];
    for _ in 0..1_000_000 {
        if (t - end).abs() <= 1e-15 * end.abs().max(1.0) {
            break;
        }
        if (t + h - end) * h > 0.0 {
            h = end - t;
        }
        let mut k = [[0.0; 2]; 7];
        k[0] = rhs(t, y);
        for s in 0..6 {
            let mut ys = y;
            for (j, kj) in k.iter().enumerate().take(s + 1) {
                ys[0] += h * A[s][j] * kj[0];
                ys[1] += h * A[s][j] * kj[1];
            }
            k[s + 1] = rhs(t + C[s] * h, ys);
        }
        // The fifth-order solution is the last stage's argument.
        let mut y5 = y;
        for (j, kj) in k.iter().enumerate().take(6) {
            y5[0] += h * A[5][j] * kj[0];
            y5[1] += h * A[5][j] * kj[1];
        }
        let mut err = 0.0f64;
        for i in 0..2 {
            let e: f64 = (0..7).map(|j| E[j] * k[j][i]).sum::<f64>() * h;
            let scale = 1e-12 * (1.0 + y[i].abs().max(y5[i].abs()));
            err = err.max((e / scale).abs());
        }
        if err <= 1.0 {
            t += h;
            y = y5;
        }
        let factor = if err == 0.0 {
            5.0
        } else {
            (0.9 * err.powf(-0.2)).clamp(0.2, 5.0)
        };
        h *= factor;
    }
    let (g, ln_g) = (y[0], y[1]);
    let f = cf1(l, rho, eta);
    // P = rho / (G^2 + F^2), F = 1/(G (f - g)).
    let inv_g2 = (-2.0 * ln_g).exp();
    rho * inv_g2 / (1.0 + inv_g2 * inv_g2 / ((f - g) * (f - g)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no charge the Coulomb functions are the Riccati-Bessel ones, and
    /// the penetrability is the hard-sphere one, inside the barrier and out.
    #[test]
    fn without_charge_it_is_the_hard_sphere_penetrability() {
        for l in 0..=4usize {
            for &rho in &[1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 3.5, 7.0, 20.0] {
                let want = crate::resonance::penetration_shift(l as i64, rho).0;
                let got = penetrability(l, rho, 0.0);
                assert!(
                    (got / want - 1.0).abs() < 1e-9,
                    "l={l} rho={rho}: {got:e} against {want:e}"
                );
            }
        }
    }

    /// Against mpmath's Coulomb functions at 30 digits, outside the barrier
    /// and deep inside it: to 1e-11 (over a grid of l <= 5, rho 0.05 to 20
    /// and eta 0.5 to 18 the worst is 8e-13).
    #[test]
    fn it_matches_mpmath() {
        for (l, rho, eta, want) in [
            (0, 1.0, 0.5, 0.5879405784031426),
            (1, 0.85, 3.2, 8.067475664089911e-6),
            (2, 3.0, 8.0, 8.985498827366621e-11),
            (0, 20.0, 3.0, 16.74115225820161),
            (5, 0.3, 18.0, 3.8510315290813e-50),
            (3, 5.0, 8.0, 6.380811824988564e-8),
        ] {
            let got = penetrability(l, rho, eta);
            assert!(
                (got / want - 1.0).abs() < 1e-11,
                "l={l} rho={rho} eta={eta}: {got:e} against {want:e}"
            );
        }
    }

    /// Across the turning point the two ways of computing it meet.
    #[test]
    fn it_is_continuous_at_the_turning_point() {
        for (l, eta) in [(0usize, 0.5), (1, 2.0), (2, 5.0), (5, 1.0)] {
            let t = eta + (eta * eta + (l * (l + 1)) as f64).sqrt();
            let (inside, outside) = (
                penetrability(l, t * (1.0 - 1e-9), eta),
                penetrability(l, t, eta),
            );
            assert!(
                (inside / outside - 1.0).abs() < 1e-7,
                "l={l} eta={eta}: {inside} {outside}"
            );
        }
    }

    /// Far under a Coulomb barrier `F_0 -> C_0 rho` and `G_0 -> 1/C_0` as
    /// `rho -> 0`, with `C_0^2 = 2 pi eta / (exp(2 pi eta) - 1)` (Gamow), so
    /// `P_0 -> C_0^2 rho`; and a barrier too thick to matter gives zero.
    #[test]
    fn deep_under_the_barrier_it_is_gamow_suppressed() {
        let (eta, rho) = (3.0, 1e-6);
        let two_pi_eta = 2.0 * std::f64::consts::PI * eta;
        let gamow = two_pi_eta / two_pi_eta.exp_m1() * rho;
        let got = penetrability(0, rho, eta);
        assert!(
            (got / gamow - 1.0).abs() < 1e-3,
            "{got:e} against {gamow:e}"
        );
        assert_eq!(penetrability(0, 1e-3, 3000.0), 0.0);
    }
}
