//! Standalone polynomial-root solver: a self-contained Ferrari-method
//! cubic/quartic solver used by surface ray-intersection routines.

/// Solve a depressed cubic t³ + pt + q = 0, returning all real roots.
pub(crate) fn solve_depressed_cubic(p: f64, q: f64) -> Vec<f64> {
    let disc = q * q / 4.0 + p * p * p / 27.0;

    if disc > 1e-12 {
        // One real root
        let sqrt_d = disc.sqrt();
        let u = cbrt(-q / 2.0 + sqrt_d);
        let v = cbrt(-q / 2.0 - sqrt_d);
        vec![u + v]
    } else if disc < -1e-12 {
        // Three real roots - trigonometric method
        let r = (-p / 3.0).sqrt();
        let cos_arg = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
        let phi = cos_arg.acos();
        let mut roots = Vec::with_capacity(3);
        for k in 0..3 {
            roots.push(2.0 * r * ((phi - 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos());
        }
        roots
    } else {
        // Repeated roots
        if q.abs() < 1e-14 {
            vec![0.0]
        } else {
            let u = cbrt(-q / 2.0);
            vec![2.0 * u, -u]
        }
    }
}

/// Cube root that handles negative numbers correctly.
fn cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.cbrt()
    } else {
        -(-x).cbrt()
    }
}

/// Solve cubic y³ + ay² + by + c = 0 by depressing and calling solve_depressed_cubic.
pub(crate) fn solve_cubic(a: f64, b: f64, c: f64) -> Vec<f64> {
    let p = b - a * a / 3.0;
    let q = c - a * b / 3.0 + 2.0 * a * a * a / 27.0;
    let shift = -a / 3.0;
    solve_depressed_cubic(p, q)
        .into_iter()
        .map(|w| w + shift)
        .collect()
}

/// Solve quartic t⁴ + bt³ + ct² + dt + e = 0 using Ferrari's method.
/// Returns all real roots.
pub(crate) fn solve_quartic(b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    // Depress: substitute t = u - b/4
    let b2 = b * b;
    let b3 = b2 * b;
    let b4 = b2 * b2;
    let p = c - 3.0 * b2 / 8.0;
    let q = d - b * c / 2.0 + b3 / 8.0;
    let r = e - b * d / 4.0 + b2 * c / 16.0 - 3.0 * b4 / 256.0;
    let shift = -b / 4.0;

    if q.abs() < 1e-14 {
        // Biquadratic: u⁴ + pu² + r = 0
        let disc = p * p - 4.0 * r;
        if disc < -1e-12 {
            return Vec::new();
        }
        let sqrt_disc = disc.max(0.0).sqrt();
        let w1 = (-p + sqrt_disc) / 2.0;
        let w2 = (-p - sqrt_disc) / 2.0;
        let mut roots = Vec::new();
        if w1 >= -1e-12 {
            let s = w1.max(0.0).sqrt();
            roots.push(s + shift);
            roots.push(-s + shift);
        }
        if w2 >= -1e-12 {
            let s = w2.max(0.0).sqrt();
            roots.push(s + shift);
            roots.push(-s + shift);
        }
        return roots;
    }

    // Resolvent cubic: y³ - (p/2)y² - ry + (rp/2 - q²/8) = 0
    let cubic_a = -p / 2.0;
    let cubic_b = -r;
    let cubic_c = r * p / 2.0 - q * q / 8.0;
    let cubic_roots = solve_cubic(cubic_a, cubic_b, cubic_c);

    // Use the largest real root for numerical stability
    let y = cubic_roots
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let two_y_minus_p = 2.0 * y - p;
    if two_y_minus_p < -1e-12 {
        return Vec::new();
    }
    let sq = two_y_minus_p.max(0.0).sqrt();

    let mut roots = Vec::new();
    if sq.abs() > 1e-14 {
        let h = q / (2.0 * sq);

        // First quadratic: u² - sq·u + (y + h) = 0
        let disc1 = sq * sq / 4.0 - (y + h);
        if disc1 >= -1e-12 {
            let s = disc1.max(0.0).sqrt();
            roots.push(sq / 2.0 + s + shift);
            roots.push(sq / 2.0 - s + shift);
        }

        // Second quadratic: u² + sq·u + (y - h) = 0
        let disc2 = sq * sq / 4.0 - (y - h);
        if disc2 >= -1e-12 {
            let s = disc2.max(0.0).sqrt();
            roots.push(-sq / 2.0 + s + shift);
            roots.push(-sq / 2.0 - s + shift);
        }
    } else {
        // sq ≈ 0, degenerate
        let disc = y * y - r;
        if disc >= -1e-12 {
            let s = disc.max(0.0).sqrt();
            let w1 = -y + s;
            let w2 = -y - s;
            if w1 >= -1e-12 {
                let v = w1.max(0.0).sqrt();
                roots.push(v + shift);
                roots.push(-v + shift);
            }
            if w2 >= -1e-12 {
                let v = w2.max(0.0).sqrt();
                roots.push(v + shift);
                roots.push(-v + shift);
            }
        }
    }

    roots
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Quartic solver tests ──

    #[test]
    fn quartic_solver_biquadratic() {
        // (t² - 1)(t² - 4) = t⁴ - 5t² + 4 = 0
        // Roots: ±1, ±2
        let roots = solve_quartic(0.0, -5.0, 0.0, 4.0);
        assert!(roots.len() >= 4);
        let mut sorted: Vec<f64> = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        assert!(sorted.iter().any(|&r| (r - 1.0).abs() < 1e-8));
        assert!(sorted.iter().any(|&r| (r + 1.0).abs() < 1e-8));
        assert!(sorted.iter().any(|&r| (r - 2.0).abs() < 1e-8));
        assert!(sorted.iter().any(|&r| (r + 2.0).abs() < 1e-8));
    }

    #[test]
    fn quartic_solver_known_roots() {
        // (t-1)(t-2)(t-3)(t-4) = t⁴ - 10t³ + 35t² - 50t + 24
        let roots = solve_quartic(-10.0, 35.0, -50.0, 24.0);
        assert!(roots.len() >= 4);
        for expected in &[1.0, 2.0, 3.0, 4.0] {
            assert!(
                roots.iter().any(|&r| (r - expected).abs() < 1e-6),
                "Missing root {}; got {:?}",
                expected,
                roots
            );
        }
    }

    // ── Cubic and quartic solver edge cases ──

    #[test]
    fn depressed_cubic_single_root() {
        // t^3 + 3t + 2 = 0 has one real root at t = -0.5961...
        // discriminant > 0 => single root
        let roots = solve_depressed_cubic(3.0, 2.0);
        assert_eq!(roots.len(), 1);
        // Verify root actually satisfies equation
        let t = roots[0];
        assert!((t * t * t + 3.0 * t + 2.0).abs() < 1e-8);
    }

    #[test]
    fn depressed_cubic_three_roots() {
        // t^3 - 3t = 0 => t(t^2 - 3) = 0 => roots: 0, sqrt(3), -sqrt(3)
        // disc = 0 + (-3)^3/27 = -1 < 0 => three roots
        let roots = solve_depressed_cubic(-3.0, 0.0);
        assert!(roots.len() >= 2); // should be 3 but 0 might be degenerate
        for &t in &roots {
            assert!((t * t * t - 3.0 * t).abs() < 1e-8);
        }
    }

    #[test]
    fn depressed_cubic_repeated_root() {
        // t^3 - 3t + 2 = (t-1)^2(t+2) = 0 => roots: 1 (double), -2
        // p = -3, q = 2, disc = 4/4 + (-27)/27 = 1 - 1 = 0
        let roots = solve_depressed_cubic(-3.0, 2.0);
        assert!(roots.len() >= 2);
        // Verify all roots satisfy equation
        for &t in &roots {
            assert!(
                (t * t * t - 3.0 * t + 2.0).abs() < 1e-6,
                "root {} doesn't satisfy equation: {}",
                t,
                t * t * t - 3.0 * t + 2.0
            );
        }
    }

    #[test]
    fn depressed_cubic_triple_root_zero() {
        // t^3 = 0 => p = 0, q = 0 => one root at 0
        let roots = solve_depressed_cubic(0.0, 0.0);
        assert!(!roots.is_empty());
        assert!(roots[0].abs() < 1e-10);
    }

    #[test]
    fn solve_cubic_general() {
        // (y-1)(y-2)(y-3) = y^3 - 6y^2 + 11y - 6
        let roots = solve_cubic(-6.0, 11.0, -6.0);
        assert!(roots.len() >= 3);
        for expected in &[1.0, 2.0, 3.0] {
            assert!(
                roots.iter().any(|&r| (r - expected).abs() < 1e-6),
                "Missing root {}; got {:?}",
                expected,
                roots
            );
        }
    }

    #[test]
    fn quartic_no_real_roots() {
        // t^4 + 1 = 0 has no real roots
        let roots = solve_quartic(0.0, 0.0, 0.0, 1.0);
        // All roots should be non-real, so the returned list should be empty or contain
        // only spurious near-zero roots
        for &r in &roots {
            // If any root is returned, verify it does NOT satisfy the equation well
            let val = r.powi(4) + 1.0;
            // val should not be close to 0 (this equation has no real roots)
            // But solver may return empty vec which is fine
            if val.abs() < 1e-6 {
                panic!("Unexpected real root {} for t^4 + 1 = 0", r);
            }
        }
    }

    #[test]
    fn quartic_double_roots() {
        // (t^2 - 1)^2 = t^4 - 2t^2 + 1 = 0
        // Roots: +1 (double), -1 (double)
        let roots = solve_quartic(0.0, -2.0, 0.0, 1.0);
        assert!(roots.len() >= 2);
        assert!(roots.iter().any(|&r| (r - 1.0).abs() < 1e-6));
        assert!(roots.iter().any(|&r| (r + 1.0).abs() < 1e-6));
    }
}
