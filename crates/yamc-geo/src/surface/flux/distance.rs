//! Signed distance to the surface: a global inside/outside test in the
//! point's own toroidal cut, and the 3-D closest point with a
//! certificate that it is the global one.

use super::series::{Collapsed, SurfacePoint};
use super::Engine;

/// Largest number of polyline vertices in the side test.
pub(super) const MAX_POLY: usize = 512;

/// The point's relation to the cut through it (`zeta = phi` of the point).
#[derive(Debug, Clone, Copy)]
pub(super) struct InPlane {
    /// +1 outside, -1 inside, 0 exactly on the cut.
    pub sign: f64,
    /// Upper bound on the distance to the cut (exact within the sag band),
    /// which bounds the 3-D distance from above.
    pub upper: f64,
    /// Parameter of the (estimated) closest point of the cut.
    pub theta: f64,
    pub phi: f64,
}

/// The closest surface point to a query point.
#[derive(Debug, Clone, Copy)]
pub(super) struct Foot {
    pub theta: f64,
    pub zeta: f64,
    /// `|x - X(theta, zeta)|^2`.
    pub d2: f64,
    pub point: SurfacePoint,
    /// Newton stopped at a stationary point with a positive-definite
    /// Hessian.
    pub converged: bool,
}

/// Exact signed distance with its closest point and outward normal.
#[derive(Debug, Clone, Copy)]
pub(super) struct Signed {
    pub d: f64,
    pub theta: f64,
    pub zeta: f64,
    /// Outward unit normal at the closest point: the gradient of `d`.
    pub normal: [f64; 3],
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Squared distance from `p` to the segment `a b`, and the segment
/// parameter of the nearest point.
#[inline]
fn segment_distance2(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let len2 = ex * ex + ey * ey;
    let s = if len2 > 0.0 {
        (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (dx, dy) = (a.0 + s * ex - p.0, a.1 + s * ey - p.1);
    (dx * dx + dy * dy, s)
}

impl Engine {
    /// Inside/outside and the distance to the cut through `x`.
    ///
    /// The cut at `phi = atan2(y, x)` is a closed curve `C(theta)` in the
    /// `(R, Z)` half-plane, and the solid's intersection with that
    /// half-plane is its interior (the toroidal angle is the cylindrical
    /// one), so this is exact. A polyline through `C` on a fixed `theta`
    /// grid stays within the sag `s = (dtheta^2 / 8) sup|C''|` of the
    /// curve, so for a point farther than `s` from the polyline the
    /// crossing parity of a ray from it equals the curve's. Closer points
    /// solve for the closest curve point on the few segments that can
    /// hold it and take the side from the normal there.
    pub(super) fn in_plane(&self, x: [f64; 3]) -> InPlane {
        let r_x = x[0].hypot(x[1]);
        let phi = x[1].atan2(x[0]);
        let cut = self.series.collapse(phi, 1);
        let n = self.n_poly;
        let sag = self.dtheta_poly.powi(2) / 8.0 * cut.curvature_bound() + self.margin;
        let p = (r_x, x[2]);
        let stride = self.series.m_max + 1;
        let vertex = |k: usize| {
            cut.curve_from_table(
                &self.poly_cos[k * stride..(k + 1) * stride],
                &self.poly_sin[k * stride..(k + 1) * stride],
            )
        };
        let mut segment_d2 = [0.0; MAX_POLY];
        let mut prev = vertex(n - 1);
        let (mut best_d2, mut best_k, mut best_s) = (f64::INFINITY, 0, 0.0);
        let mut inside = false;
        for (k, slot) in segment_d2.iter_mut().enumerate().take(n) {
            let next = vertex(k);
            // Segment k runs from vertex k - 1 to vertex k.
            let (d2, s) = segment_distance2(p, prev, next);
            *slot = d2;
            if d2 < best_d2 {
                (best_d2, best_k, best_s) = (d2, k, s);
            }
            if (prev.1 > p.1) != (next.1 > p.1) {
                let r_cross = prev.0 + (p.1 - prev.1) * (next.0 - prev.0) / (next.1 - prev.1);
                if r_cross > p.0 {
                    inside = !inside;
                }
            }
            prev = next;
        }
        let dtheta = self.dtheta_poly;
        let theta_of = |k: usize, s: f64| (k as f64 - 1.0 + s) * dtheta;
        let d_poly = best_d2.sqrt();
        if d_poly > sag {
            return InPlane {
                sign: if inside { -1.0 } else { 1.0 },
                upper: d_poly + sag,
                theta: theta_of(best_k, best_s),
                phi,
            };
        }
        // Within the sag band: the closest curve point lies on an arc whose
        // chord is within d_poly + 2 sag of the point.
        let limit2 = (d_poly + 2.0 * sag).powi(2);
        let (mut d2, mut theta) = (f64::INFINITY, 0.0);
        for (k, &seg) in segment_d2.iter().enumerate().take(n) {
            if seg <= limit2 {
                let lo = (k as f64 - 2.0) * dtheta;
                let hi = (k as f64 + 1.0) * dtheta;
                let (cd2, ct) = closest_on_cut(&cut, p, lo, hi);
                if cd2 < d2 {
                    (d2, theta) = (cd2, ct);
                }
            }
        }
        let c = cut.curve(theta);
        let (dr, dz) = (p.0 - c[0], p.1 - c[1]);
        // Outward normal of the cut: orientation * (Z', -R').
        let side = self.orientation_in * (dr * c[3] - dz * c[2]);
        let dist = d2.sqrt();
        InPlane {
            sign: if dist == 0.0 || side == 0.0 {
                0.0
            } else {
                side.signum()
            },
            upper: dist,
            theta,
            phi,
        }
    }

    /// Newton's method for the closest surface point, from `(theta,
    /// zeta)`, optionally kept inside a parameter rectangle.
    pub(super) fn newton(
        &self,
        x: [f64; 3],
        theta: f64,
        zeta: f64,
        rect: Option<([f64; 2], [f64; 2])>,
        max_iter: usize,
    ) -> Foot {
        let clamp = |t: f64, z: f64| match rect {
            Some((tr, zr)) => (t.clamp(tr[0], tr[1]), z.clamp(zr[0], zr[1])),
            None => (t, z),
        };
        let (mut t, mut z) = clamp(theta, zeta);
        let mut sp = self.series.point(t, z);
        let mut r = sub(x, sp.x);
        let mut f = dot(r, r);
        let tol = self.newton_tolerance;
        let mut converged = false;
        for _ in 0..max_iter {
            let g = [-dot(r, sp.x_t), -dot(r, sp.x_z)];
            let h00 = dot(sp.x_t, sp.x_t) - dot(r, sp.x_tt);
            let h01 = dot(sp.x_t, sp.x_z) - dot(r, sp.x_tz);
            let h11 = dot(sp.x_z, sp.x_z) - dot(r, sp.x_zz);
            let det = h00 * h11 - h01 * h01;
            let pd = h00 > 0.0 && det > 0.0;
            // Newton where the Hessian is positive definite, otherwise a
            // Levenberg-Marquardt step that is a descent direction.
            let (a00, a11) = if pd {
                (h00, h11)
            } else {
                let lambda = h00.abs().max(h11.abs()) + 1.0;
                (h00 + lambda, h11 + lambda)
            };
            let det_a = a00 * a11 - h01 * h01;
            if det_a <= 0.0 || !det_a.is_finite() {
                break;
            }
            let mut dt = -(a11 * g[0] - h01 * g[1]) / det_a;
            let mut dz = -(a00 * g[1] - h01 * g[0]) / det_a;
            let mut improved = false;
            for _ in 0..40 {
                let (nt, nz) = clamp(t + dt, z + dz);
                let np = self.series.point(nt, nz);
                let nr = sub(x, np.x);
                let nf = dot(nr, nr);
                if nf <= f {
                    let moved = [np.x[0] - sp.x[0], np.x[1] - sp.x[1], np.x[2] - sp.x[2]];
                    (t, z, sp, r, f) = (nt, nz, np, nr, nf);
                    improved = true;
                    if pd && dot(moved, moved).sqrt() <= tol {
                        converged = true;
                    }
                    break;
                }
                dt *= 0.5;
                dz *= 0.5;
            }
            if converged || !improved {
                // No descent step: a stationary point (or the clamp edge).
                if !improved {
                    converged = pd
                        && rect.is_none_or(|(tr, zr)| {
                            t > tr[0] && t < tr[1] && z > zr[0] && z < zr[1]
                        });
                }
                break;
            }
        }
        Foot {
            theta: t,
            zeta: z,
            d2: f,
            point: sp,
            converged,
        }
    }

    /// Whether `foot` (at distance `dist`) is the global closest point of
    /// a point on `side`: no patch outside its 3 x 3 neighbourhood comes
    /// closer than `dist`, and inside that neighbourhood the surface does
    /// not curve towards the point enough for the normals to meet within
    /// `dist`.
    fn certified(&self, x: [f64; 3], foot: &Foot, dist: f64, side: f64) -> bool {
        if !foot.converged {
            return false;
        }
        let centre = self.patches.patch_of(foot.theta, foot.zeta);
        let kappa = self.patches.curvature_near(centre, 1, side);
        if dist * kappa >= 0.9 {
            return false;
        }
        let far2 = self
            .patches
            .min_distance2(x, Some((centre, 1)), dist * dist);
        far2 >= dist * dist
    }

    /// Exhaustive search over the patches whose box comes closer than the
    /// best point found so far, starting from `best`.
    fn branch_and_bound(&self, x: [f64; 3], mut best: Foot) -> Foot {
        for (lb2, i, j) in self.patches.within(x, best.d2) {
            if lb2 >= best.d2 {
                break;
            }
            let (tr, zr) = self.patches.rect(i, j);
            let (tm, zm) = (0.5 * (tr[0] + tr[1]), 0.5 * (zr[0] + zr[1]));
            let inset = |a: [f64; 2], f: f64| a[0] + (a[1] - a[0]) * f;
            for (t0, z0) in [
                (tm, zm),
                (inset(tr, 0.1), inset(zr, 0.1)),
                (inset(tr, 0.9), inset(zr, 0.1)),
                (inset(tr, 0.1), inset(zr, 0.9)),
                (inset(tr, 0.9), inset(zr, 0.9)),
            ] {
                let foot = self.newton(x, t0, z0, Some((tr, zr)), 30);
                if foot.d2 < best.d2 {
                    best = foot;
                }
            }
        }
        best
    }

    /// Exact signed distance, its closest point and the outward normal.
    pub(super) fn signed(&self, x: [f64; 3]) -> Signed {
        let cut = self.in_plane(x);
        self.signed_from(x, &cut)
    }

    pub(super) fn signed_from(&self, x: [f64; 3], cut: &InPlane) -> Signed {
        if cut.sign == 0.0 {
            let sp = self.series.point(cut.theta, cut.phi);
            return Signed {
                d: 0.0,
                theta: cut.theta,
                zeta: cut.phi,
                normal: self.normal(&sp),
            };
        }
        // The cut's closest point is a surface point: an upper bound.
        let seed_point = self.series.point(cut.theta, cut.phi);
        let seed_r = sub(x, seed_point.x);
        let seed = Foot {
            theta: cut.theta,
            zeta: cut.phi,
            d2: dot(seed_r, seed_r),
            point: seed_point,
            converged: false,
        };
        let mut foot = self.newton(x, cut.theta, cut.phi, None, 40);
        if foot.d2 > seed.d2 {
            foot = seed;
        }
        let dist = foot.d2.sqrt();
        if !self.certified(x, &foot, dist, cut.sign) {
            foot = self.branch_and_bound(x, foot);
        }
        // The normal comes from the surface derivatives at the foot: right
        // after a crossing the point is ~1e-8 cm from the surface, where
        // (x - X) / |x - X| is ill-conditioned.
        Signed {
            d: cut.sign * foot.d2.sqrt(),
            theta: foot.theta,
            zeta: foot.zeta,
            normal: self.normal(&foot.point),
        }
    }

    /// Outward unit normal from the parameter derivatives.
    pub(super) fn normal(&self, sp: &SurfacePoint) -> [f64; 3] {
        let n = cross(sp.x_t, sp.x_z);
        let len = dot(n, n).sqrt();
        let s = self.orientation_3d / len;
        [n[0] * s, n[1] * s, n[2] * s]
    }

    /// Rigorous bounds on the signed distance: `[lo, hi]` with the sign
    /// exact, from the cut and the patch boxes.
    pub(super) fn bounds(&self, x: [f64; 3]) -> (InPlane, f64, f64) {
        let cut = self.in_plane(x);
        if cut.sign == 0.0 {
            return (cut, 0.0, 0.0);
        }
        let lower2 = self.patches.min_distance2(x, None, cut.upper * cut.upper);
        let lower = lower2.sqrt().min(cut.upper);
        let (lo, hi) = if cut.sign > 0.0 {
            (lower, cut.upper)
        } else {
            (-cut.upper, -lower)
        };
        (cut, lo, hi)
    }
}

/// Closest point of the cut to `p` over `theta` in `[lo, hi]`, as
/// `(distance^2, theta)`: the smallest of the endpoints and every local
/// minimum, each refined by safeguarded Newton.
fn closest_on_cut(cut: &Collapsed, p: (f64, f64), lo: f64, hi: f64) -> (f64, f64) {
    // q = |C - p|^2 / 2, q' = (C - p).C', q'' = |C'|^2 + (C - p).C''.
    let eval = |t: f64| {
        let c = cut.curve(t);
        let (dr, dz) = (c[0] - p.0, c[1] - p.1);
        (
            dr * dr + dz * dz,
            dr * c[2] + dz * c[3],
            c[2] * c[2] + c[3] * c[3] + dr * c[4] + dz * c[5],
        )
    };
    const SAMPLES: usize = 12;
    let step = (hi - lo) / SAMPLES as f64;
    let mut prev_t = lo;
    let mut prev = eval(lo);
    let mut best = (prev.0, lo);
    for k in 1..=SAMPLES {
        let t = lo + k as f64 * step;
        let cur = eval(t);
        if cur.0 < best.0 {
            best = (cur.0, t);
        }
        if prev.1 < 0.0 && cur.1 >= 0.0 {
            let (d2, root) = rtsafe_min(&eval, prev_t, t);
            if d2 < best.0 {
                best = (d2, root);
            }
        }
        prev_t = t;
        prev = cur;
    }
    best
}

/// Minimise `q` on a bracket where `q'` goes from negative to
/// non-negative: Newton on `q'` with bisection fallback.
fn rtsafe_min(eval: &impl Fn(f64) -> (f64, f64, f64), lo: f64, hi: f64) -> (f64, f64) {
    let (mut a, mut b) = (lo, hi);
    let mut t = 0.5 * (a + b);
    let mut v = eval(t);
    for _ in 0..80 {
        if v.1 < 0.0 {
            a = t;
        } else {
            b = t;
        }
        let newton = if v.2 > 0.0 { t - v.1 / v.2 } else { f64::NAN };
        let next = if newton > a && newton < b {
            newton
        } else {
            0.5 * (a + b)
        };
        if (next - t).abs() <= 1e-15 * (1.0 + t.abs()) || b - a <= 1e-15 * (1.0 + t.abs()) {
            t = next;
            v = eval(t);
            break;
        }
        t = next;
        v = eval(t);
    }
    (v.0, t)
}
