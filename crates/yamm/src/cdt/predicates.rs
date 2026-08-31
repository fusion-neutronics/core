/// Compute circumcenter in metric-weighted space.
///
/// Transforms UV vertices by the first fundamental form metric tensor,
/// computes the circumcenter in the transformed (3D-isotropic) space,
/// then transforms back to UV. This produces Steiner points that create
/// equilateral triangles in 3D on curved surfaces.
///
/// Metric tensor: ds² = E du² + 2F du dv + G dv²
/// For F≈0 (orthogonal parameterization): transform is (u√E, v√G).
pub fn circumcenter_metric(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    e: f64,
    f: f64,
    g: f64,
) -> [f64; 2] {
    // Cholesky decomposition of [[E,F],[F,G]]:
    // L = [[l11, 0], [l21, l22]]
    // where l11 = sqrt(E), l21 = F/sqrt(E), l22 = sqrt(G - F²/E)
    let e_safe = e.max(1e-15);
    let l11 = e_safe.sqrt();
    let l21 = f / l11;
    let det = g - f * f / e_safe;
    let l22 = if det > 1e-15 { det.sqrt() } else { 1e-8 };

    // Transform to metric space: p' = L * p
    let transform =
        |p: [f64; 2]| -> [f64; 2] { [l11 * p[0] + 0.0 * p[1], l21 * p[0] + l22 * p[1]] };

    let a_t = transform(a);
    let b_t = transform(b);
    let c_t = transform(c);

    // Circumcenter in transformed space
    let cc_t = circumcenter(a_t, b_t, c_t);

    // Inverse transform: p = L^{-1} * p'
    // L^{-1} = [[1/l11, 0], [-l21/(l11*l22), 1/l22]]
    let inv_l11 = 1.0 / l11;
    let inv_l22 = 1.0 / l22;
    [
        inv_l11 * cc_t[0],
        -l21 * inv_l11 * inv_l22 * cc_t[0] + inv_l22 * cc_t[1],
    ]
}

/// Compute circumcenter of triangle (a, b, c) in UV space.
pub fn circumcenter(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let ax = a[0] - c[0];
    let ay = a[1] - c[1];
    let bx = b[0] - c[0];
    let by = b[1] - c[1];

    let d = 2.0 * (ax * by - ay * bx);

    if d.abs() < f64::EPSILON * 1e3 {
        // Degenerate: return centroid as fallback
        return [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
    }

    let a_sq = ax * ax + ay * ay;
    let b_sq = bx * bx + by * by;

    let ux = (a_sq * by - b_sq * ay) / d;
    let uy = (b_sq * ax - a_sq * bx) / d;

    [ux + c[0], uy + c[1]]
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circumcenter_equilateral() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.5, 0.866025403784];
        let cc = circumcenter(a, b, c);
        assert!((cc[0] - 0.5).abs() < 1e-6);
        assert!((cc[1] - 0.2886751).abs() < 1e-4);
    }

    #[test]
    fn circumcenter_right_triangle() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        let cc = circumcenter(a, b, c);
        // Circumcenter of right triangle is midpoint of hypotenuse
        assert!((cc[0] - 0.5).abs() < 1e-10);
        assert!((cc[1] - 0.5).abs() < 1e-10);
    }
}
