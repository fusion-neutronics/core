//! Parametric curve evaluation and adaptive discretization.
//!
//! First increment of moving edge discretization out of Python/OCC
//! (the `dag.rs` edge→face DAG TODO). Python extracts each OCC edge's
//! curve DEFINITION once (analytic parameters or B-spline data) and
//! Rust evaluates + discretizes - in parallel, without per-point PyO3
//! or OCC calls.
//!
//! Conformality between faces sharing an edge does NOT depend on this
//! sampler matching OCC's `GCPnts_TangentialDeflection` bit-for-bit:
//! shared edges are discretized once and cached (`edge_params_cache`),
//! so both faces always receive identical parameters. The sampler only
//! has to honour the chordal + angular tolerance bounds.

/// A 3-D parametric curve definition, mirroring the OCC curve types
/// that appear in STEP models. Exotic adaptor types (offset curves,
/// curves-on-surface) stay on the Python/OCC fallback path.
#[derive(Clone, Debug)]
pub enum Curve3 {
    /// `p(t) = origin + t * dir` (dir unit-length).
    Line { origin: [f64; 3], dir: [f64; 3] },
    /// `p(t) = center + r cos(t) x + r sin(t) y`.
    Circle {
        center: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        radius: f64,
    },
    /// `p(t) = center + a cos(t) x + b sin(t) y`.
    Ellipse {
        center: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        major: f64,
        minor: f64,
    },
    /// NURBS (rational when `weights` is `Some`); evaluated by de Boor.
    BSpline {
        degree: usize,
        /// Flattened knot vector WITH multiplicities expanded.
        knots: Vec<f64>,
        poles: Vec<[f64; 3]>,
        weights: Option<Vec<f64>>,
    },
}

impl Curve3 {
    /// Evaluate the curve position at parameter `t`.
    pub fn value(&self, t: f64) -> [f64; 3] {
        match self {
            Curve3::Line { origin, dir } => [
                origin[0] + t * dir[0],
                origin[1] + t * dir[1],
                origin[2] + t * dir[2],
            ],
            Curve3::Circle {
                center,
                x_dir,
                y_dir,
                radius,
            } => {
                let (s, c) = t.sin_cos();
                [
                    center[0] + radius * (c * x_dir[0] + s * y_dir[0]),
                    center[1] + radius * (c * x_dir[1] + s * y_dir[1]),
                    center[2] + radius * (c * x_dir[2] + s * y_dir[2]),
                ]
            }
            Curve3::Ellipse {
                center,
                x_dir,
                y_dir,
                major,
                minor,
            } => {
                let (s, c) = t.sin_cos();
                [
                    center[0] + major * c * x_dir[0] + minor * s * y_dir[0],
                    center[1] + major * c * x_dir[1] + minor * s * y_dir[1],
                    center[2] + major * c * x_dir[2] + minor * s * y_dir[2],
                ]
            }
            Curve3::BSpline {
                degree,
                knots,
                poles,
                weights,
            } => de_boor_3(*degree, knots, poles, weights.as_deref(), t),
        }
    }
}

/// A 2-D parametric curve (a pcurve in a face's UV space).
#[derive(Clone, Debug)]
pub enum Curve2 {
    Line {
        origin: [f64; 2],
        dir: [f64; 2],
    },
    Circle {
        center: [f64; 2],
        x_dir: [f64; 2],
        y_dir: [f64; 2],
        radius: f64,
    },
    Ellipse {
        center: [f64; 2],
        x_dir: [f64; 2],
        y_dir: [f64; 2],
        major: f64,
        minor: f64,
    },
    BSpline {
        degree: usize,
        knots: Vec<f64>,
        poles: Vec<[f64; 2]>,
        weights: Option<Vec<f64>>,
    },
}

impl Curve2 {
    /// Evaluate the pcurve at parameter `t`.
    pub fn value(&self, t: f64) -> [f64; 2] {
        match self {
            Curve2::Line { origin, dir } => [origin[0] + t * dir[0], origin[1] + t * dir[1]],
            Curve2::Circle {
                center,
                x_dir,
                y_dir,
                radius,
            } => {
                let (s, c) = t.sin_cos();
                [
                    center[0] + radius * (c * x_dir[0] + s * y_dir[0]),
                    center[1] + radius * (c * x_dir[1] + s * y_dir[1]),
                ]
            }
            Curve2::Ellipse {
                center,
                x_dir,
                y_dir,
                major,
                minor,
            } => {
                let (s, c) = t.sin_cos();
                [
                    center[0] + major * c * x_dir[0] + minor * s * y_dir[0],
                    center[1] + major * c * x_dir[1] + minor * s * y_dir[1],
                ]
            }
            Curve2::BSpline {
                degree,
                knots,
                poles,
                weights,
            } => de_boor_2(*degree, knots, poles, weights.as_deref(), t),
        }
    }
}

/// Find the knot span index for parameter `t` (clamped knot vector).
pub(crate) fn find_span(degree: usize, knots: &[f64], t: f64) -> usize {
    let n = knots.len() - degree - 2; // last pole index
    let t = t.clamp(knots[degree], knots[n + 1]);
    // Binary search for span k with knots[k] <= t < knots[k+1].
    let mut lo = degree;
    let mut hi = n + 1;
    if t >= knots[n + 1] {
        return n;
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if t < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

macro_rules! de_boor_impl {
    ($name:ident, $dim:literal) => {
        fn $name(
            degree: usize,
            knots: &[f64],
            poles: &[[f64; $dim]],
            weights: Option<&[f64]>,
            t: f64,
        ) -> [f64; $dim] {
            let k = find_span(degree, knots, t);
            let t = t.clamp(knots[degree], knots[knots.len() - degree - 1]);
            // Homogeneous de Boor: carry weight as an extra coordinate.
            let mut d: Vec<([f64; $dim], f64)> = (0..=degree)
                .map(|j| {
                    let i = j + k - degree;
                    let w = weights.map_or(1.0, |ws| ws[i]);
                    let mut p = poles[i];
                    for x in p.iter_mut() {
                        *x *= w;
                    }
                    (p, w)
                })
                .collect();
            for r in 1..=degree {
                for j in (r..=degree).rev() {
                    let i = j + k - degree;
                    let denom = knots[i + degree - r + 1] - knots[i];
                    let alpha = if denom.abs() < 1e-300 {
                        0.0
                    } else {
                        (t - knots[i]) / denom
                    };
                    let (p1, w1) = d[j - 1];
                    let (p0, w0) = d[j];
                    let mut p = [0.0; $dim];
                    for (x, (a, b)) in p.iter_mut().zip(p0.iter().zip(p1.iter())) {
                        *x = (1.0 - alpha) * b + alpha * a;
                    }
                    d[j] = (p, (1.0 - alpha) * w1 + alpha * w0);
                }
            }
            let (mut p, w) = d[degree];
            if weights.is_some() && w.abs() > 1e-300 {
                for x in p.iter_mut() {
                    *x /= w;
                }
            }
            p
        }
    };
}

de_boor_impl!(de_boor_3, 3);
de_boor_impl!(de_boor_2, 2);

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Adaptive discretization of a 3-D curve over `[t0, t1]` honouring a
/// chordal deflection bound and an angular (turn) bound - the same
/// guarantees as OCC's `GCPnts_TangentialDeflection`.
///
/// Recursive midpoint refinement: a segment splits while the midpoint's
/// chordal deviation exceeds `tolerance` or the tangent turn across it
/// exceeds `angular_tolerance` radians (tangents approximated by
/// finite differences, which is robust for all curve kinds including
/// rational B-splines).
pub fn discretize_curve3(
    curve: &Curve3,
    t0: f64,
    t1: f64,
    tolerance: f64,
    angular_tolerance: f64,
) -> Vec<f64> {
    // Straight lines satisfy both bounds with a single segment - emit
    // exactly the endpoints (matching OCC and preserving the minimal-
    // triangulation goal: a planar quad face must stay 2 triangles).
    if matches!(curve, Curve3::Line { .. }) {
        return vec![t0, t1];
    }
    let mut params = vec![t0];
    // Seed segments so the midpoint test cannot alias: conics get at
    // most a quarter turn per seed segment (a full circle's endpoints
    // coincide - pure endpoint subdivision would see chord 0); B-splines
    // seed at their distinct knot values (no span can hide between two
    // sample points).
    let seeds: Vec<f64> = match curve {
        Curve3::Line { .. } => unreachable!(),
        Curve3::Circle { .. } | Curve3::Ellipse { .. } => {
            let n = ((t1 - t0) / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
            (0..=n)
                .map(|i| t0 + (t1 - t0) * i as f64 / n as f64)
                .collect()
        }
        Curve3::BSpline { knots, degree, .. } => {
            let mut s: Vec<f64> = vec![t0];
            for &k in &knots[*degree..knots.len() - degree] {
                if k > t0 + 1e-12 && k < t1 - 1e-12 && (k - s[s.len() - 1]).abs() > 1e-12 {
                    s.push(k);
                }
            }
            s.push(t1);
            s
        }
    };
    let mut stack: Vec<(f64, f64)> = seeds.windows(2).rev().map(|w| (w[0], w[1])).collect();
    const MAX_DEPTH_SPAN: f64 = 1e-9;
    // Safety ceiling on emitted points per edge. Legitimate analytic / B-spline
    // edges at working tolerances emit O(1e2)-O(1e4) points; a single edge past
    // this is already pathological for the per-face CDT that consumes it. The
    // cap only ever trips when the tolerance test can never be satisfied (e.g. a
    // near-degenerate or mis-evaluated periodic seam on a high-pole surface),
    // where without it the loop refines to the MAX_DEPTH_SPAN floor and emits
    // ~(t1-t0)/1e-9 points (billions), pinning one core and exhausting memory.
    const MAX_CURVE_POINTS: usize = 100_000;
    while let Some((a, b)) = stack.pop() {
        let m = 0.5 * (a + b);
        let pa = curve.value(a);
        let pb = curve.value(b);
        let pm = curve.value(m);
        // Chordal deviation of the midpoint from segment (pa, pb).
        let chord = dist3(pa, pb);
        let dev = if chord < 1e-300 {
            dist3(pa, pm)
        } else {
            // Distance from pm to line (pa, pb).
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let am = [pm[0] - pa[0], pm[1] - pa[1], pm[2] - pa[2]];
            let cx = ab[1] * am[2] - ab[2] * am[1];
            let cy = ab[2] * am[0] - ab[0] * am[2];
            let cz = ab[0] * am[1] - ab[1] * am[0];
            (cx * cx + cy * cy + cz * cz).sqrt() / chord
        };
        // Turn angle between (pa->pm) and (pm->pb).
        let v1 = [pm[0] - pa[0], pm[1] - pa[1], pm[2] - pa[2]];
        let v2 = [pb[0] - pm[0], pb[1] - pm[1], pb[2] - pm[2]];
        let l1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let l2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
        let turn = if l1 > 1e-300 && l2 > 1e-300 {
            let dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (l1 * l2);
            dot.clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };
        let want_refine = dev > tolerance || turn > angular_tolerance;
        // `m > a && m < b` is the midpoint-degeneracy guard: at large |a|,|b| the
        // float midpoint `0.5*(a+b)` can round to exactly `a` or `b` while
        // `(b - a) > MAX_DEPTH_SPAN` still holds, which would re-queue an
        // identical segment forever. If the midpoint cannot split the interval,
        // stop refining and emit. `params.len() < MAX_CURVE_POINTS` bounds the
        // never-converging case. Both clauses are always true for normal curves
        // (the midpoint is strictly interior when (b - a) > 1e-9, and real edges
        // stay far below the cap), so this is bit-identical off the pathological
        // path.
        let can_refine =
            (b - a) > MAX_DEPTH_SPAN && m > a && m < b && params.len() < MAX_CURVE_POINTS;
        if want_refine && can_refine {
            stack.push((m, b));
            stack.push((a, m));
        } else {
            params.push(b);
        }
    }
    params
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_eval() {
        let c = Curve3::Line {
            origin: [1.0, 2.0, 3.0],
            dir: [1.0, 0.0, 0.0],
        };
        assert_eq!(c.value(2.5), [3.5, 2.0, 3.0]);
    }

    #[test]
    fn circle_eval() {
        let c = Curve3::Circle {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            radius: 2.0,
        };
        let p = c.value(std::f64::consts::FRAC_PI_2);
        assert!((p[0]).abs() < 1e-12 && (p[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bspline_endpoint_interpolation() {
        // Clamped quadratic over [0, 1]: must interpolate first/last poles.
        let c = Curve3::BSpline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            weights: None,
        };
        let p0 = c.value(0.0);
        let p1 = c.value(1.0);
        assert!(dist3(p0, [0.0, 0.0, 0.0]) < 1e-12);
        assert!(dist3(p1, [2.0, 0.0, 0.0]) < 1e-12);
        // Midpoint of a quadratic Bezier: 0.25*P0 + 0.5*P1 + 0.25*P2.
        let pm = c.value(0.5);
        assert!(dist3(pm, [1.0, 1.0, 0.0]) < 1e-12);
    }

    #[test]
    fn rational_bspline_quarter_circle() {
        // Exact quarter circle as a rational quadratic Bezier.
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let c = Curve3::BSpline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles: vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            weights: Some(vec![1.0, w, 1.0]),
        };
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let p = c.value(t);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "radius at t={t}: {r}");
        }
    }

    #[test]
    fn discretize_circle_meets_chordal_bound() {
        let c = Curve3::Circle {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            radius: 10.0,
        };
        let tol = 0.01;
        let ang = 0.3;
        let params = discretize_curve3(&c, 0.0, std::f64::consts::TAU, tol, ang);
        assert!(params.len() >= 8);
        // Verify the chordal bound on every emitted segment.
        for w in params.windows(2) {
            let pa = c.value(w[0]);
            let pb = c.value(w[1]);
            let pm = c.value(0.5 * (w[0] + w[1]));
            let chord = dist3(pa, pb);
            if chord > 1e-12 {
                let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let am = [pm[0] - pa[0], pm[1] - pa[1], pm[2] - pa[2]];
                let cx = ab[1] * am[2] - ab[2] * am[1];
                let cy = ab[2] * am[0] - ab[0] * am[2];
                let cz = ab[0] * am[1] - ab[1] * am[0];
                let dev = (cx * cx + cy * cy + cz * cz).sqrt() / chord;
                assert!(dev <= tol * 1.01, "segment deviation {dev} > {tol}");
            }
        }
        // Params strictly increasing.
        for w in params.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn discretize_line_is_minimal() {
        let c = Curve3::Line {
            origin: [0.0, 0.0, 0.0],
            dir: [1.0, 0.0, 0.0],
        };
        let params = discretize_curve3(&c, 0.0, 100.0, 0.01, 0.3);
        // A straight line is a single segment: endpoints only.
        assert_eq!(params, vec![0.0, 100.0]);
    }

    // A high-pole doubly-periodic seam curve (a stellarator plasma wall) drove
    // discretize_curve3 to subdivide forever because the tolerance test could
    // never be met. These two tests reproduce the two failure modes on a curve
    // that never satisfies the tolerance; both must return a BOUNDED number of
    // points (an unbounded run would emit ~(t1-t0)/1e-9 points and hang the test
    // process rather than fail an assert, so returning at all proves termination
    // and the bound proves it does not blow up).
    const BOUND: usize = 200_000; // comfortably above the internal cap, far below a runaway

    #[test]
    fn discretize_never_converging_curve_is_bounded() {
        // tolerance = 0 and angular = 0: a curved segment can never pass the
        // test, so pre-fix it refined to the 1e-9 span floor (~1e9 points).
        let c = Curve3::Circle {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            radius: 1.0,
        };
        let params = discretize_curve3(&c, 0.0, 1.0, 0.0, 0.0);
        assert!(
            params.len() < BOUND,
            "unbounded subdivision: {} points",
            params.len()
        );
        for w in params.windows(2) {
            assert!(w[1] >= w[0], "params must stay sorted");
        }
    }

    #[test]
    fn discretize_large_magnitude_params_terminate() {
        // At ~1e15 the float midpoint 0.5*(a+b) aliases to an endpoint while
        // (b - a) is still ~O(1) >> 1e-9, so pre-fix the loop re-queued the same
        // segment forever. The midpoint-degeneracy guard must emit instead.
        let c = Curve3::Circle {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            radius: 1.0,
        };
        let base = 1.0e15;
        let params = discretize_curve3(&c, base, base + 1.0, 1e-9, 1e-9);
        assert!(
            params.len() < BOUND,
            "did not terminate cleanly: {} points",
            params.len()
        );
        assert_eq!(params.first().copied(), Some(base));
        assert_eq!(params.last().copied(), Some(base + 1.0));
    }
}
