//! First crossing of a ray with a level set `D(x) = offset` of the signed
//! distance to the surface.
//!
//! Every step is safe: `D` is 1-Lipschitz, so from a point where
//! `|D - offset| >= w` no root lies within `w` (the step uses rigorous
//! bounds where they decide the sign, the exact distance otherwise). Near
//! the surface, where the curvature bound `K` on `D` along the ray holds,
//! the step is the larger second-order one. A sign change is then refined
//! by safeguarded Newton inside a bracket whose left end is known to be
//! root-free, after checking that no earlier pair of crossings hides in it.

use super::Engine;

/// Roots at or below this ray distance are dropped and the search goes
/// on: `Cell::closest_surface` ignores distances up to 1e-10 cm, so
/// returning one would hide this surface's next crossing.
pub(super) const MIN_ROOT: f64 = 1e-10;
/// A pair of crossings closer together than this (cm) may be missed; a
/// ray would have to graze the surface within about 1e-9 rad of tangency
/// for it to matter.
const H_MIN: f64 = 1e-9;
/// Cap on distance evaluations per query.
const MAX_STEPS: usize = 400;
/// Distance from the level set within which a particle counts as on it
/// (well above the 1e-8 cm crossing nudge).
const ON_BAND: f64 = 1e-7;

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

struct Ray<'a> {
    engine: &'a Engine,
    origin: [f64; 3],
    direction: [f64; 3],
    offset: f64,
    steps: usize,
}

impl Ray<'_> {
    fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    /// `g = D - offset`, its slope along the ray, and `|D|`.
    fn exact(&mut self, t: f64) -> (f64, f64, f64) {
        self.steps += 1;
        let s = self.engine.signed(self.at(t));
        (s.d - self.offset, dot(s.normal, self.direction), s.d.abs())
    }

    fn budget_left(&self) -> bool {
        self.steps < MAX_STEPS
    }

    /// Curvature bound on `g` along the ray over a stretch where `|D|`
    /// stays below `d_max`, if the second-order bound holds there.
    fn curvature(&self, d_max: f64) -> Option<f64> {
        let k = self.engine.kappa_all;
        (k > 0.0 && d_max * k <= 0.5).then_some(2.0 * k)
    }

    /// A step from a point with `|g| = w`, approach speed `v` (positive
    /// when moving towards the level set) and `|D| = d`, within which `g`
    /// cannot vanish.
    fn safe_step(&self, w: f64, v: f64, d: f64) -> f64 {
        let k = self.engine.kappa_all;
        if k > 0.0 && d * k < 0.4 {
            let kk = 2.0 * k;
            // Positive root of w - v h - K h^2 / 2 (for v <= 0 this is the
            // longer (|v| + root) / K, moving away from the level set).
            let h = ((v * v + 2.0 * kk * w).sqrt() - v) / kk;
            let h_max = 0.5 / k - d;
            w.max(h.min(h_max))
        } else {
            w
        }
    }

    /// Distance the particle can move away from the level set it sits on,
    /// slope `gp` and `|g| = w`, before the level set can be met again.
    fn chord(&self, w: f64, gp: f64, d: f64) -> f64 {
        let k = self.engine.kappa_all;
        if k > 0.0 && d * k < 0.4 {
            let kk = 2.0 * k;
            let h = (gp.abs() + (gp * gp + 2.0 * kk * w).sqrt()) / kk;
            (0.99 * h).min(0.5 / k - d)
        } else {
            0.0
        }
    }

    fn tolerance(t: f64) -> f64 {
        (4.0 * f64::EPSILON * t.abs()).max(1e-12)
    }

    /// First root in `[a, b]` where `g(a)` and `g(b)` have the same sign
    /// and `a` is root-free: `None` if there is none (pairs of roots closer
    /// than `H_MIN` may be missed).
    fn scan_same_sign(&mut self, a: f64, ga: f64, b: f64, gb: f64) -> Option<f64> {
        if ga.abs() + gb.abs() > b - a || b - a <= H_MIN || !self.budget_left() {
            return None;
        }
        let m = 0.5 * (a + b);
        let (gm, _, _) = self.exact(m);
        if gm == 0.0 {
            return Some(m);
        }
        if gm.signum() != ga.signum() {
            return Some(self.bracket(a, ga, f64::NAN, m, gm));
        }
        self.scan_same_sign(a, ga, m, gm)
            .or_else(|| self.scan_same_sign(m, gm, b, gb))
    }

    /// The first root in `(a, b]` given a sign change and a root-free `a`:
    /// the far (sign-verified) end of a bracket narrowed to the
    /// tolerance.
    fn bracket(&mut self, mut a: f64, mut ga: f64, mut gpa: f64, mut b: f64, gb: f64) -> f64 {
        let mut monotone = false;
        while b - a > Self::tolerance(b) && self.budget_left() {
            if !monotone {
                // g is monotone on [a, b] (one root) if its slope at a
                // cannot reverse within the bracket.
                if gpa.is_finite() {
                    let d_max = (ga + self.offset).abs() + (b - a);
                    if let Some(kk) = self.curvature(d_max) {
                        monotone = gpa.abs() > kk * (b - a) && gpa.signum() != ga.signum();
                    }
                }
            }
            // Newton from a when monotone and it lands inside, else bisect.
            let newton = if monotone && gpa != 0.0 {
                a - ga / gpa
            } else {
                f64::NAN
            };
            let m = if newton > a && newton < b {
                newton
            } else {
                0.5 * (a + b)
            };
            let (gm, gpm, _) = self.exact(m);
            if gm == 0.0 {
                return m;
            }
            if gm.signum() != ga.signum() {
                b = m;
                continue;
            }
            if !monotone {
                // Same sign at m: an earlier pair of roots could hide in [a, m].
                if let Some(root) = self.scan_same_sign(a, ga, m, gm) {
                    return root;
                }
            }
            (a, ga, gpa) = (m, gm, gpm);
        }
        debug_assert!(gb != 0.0);
        b
    }

    fn run(&mut self, t_start: f64, t_end: f64) -> Outcome {
        let mut t = t_start;
        // The last exact evaluation `(t, g, slope)`: no root lies before it.
        let mut prev: Option<(f64, f64, f64)> = None;
        while self.budget_left() {
            if t > t_end {
                return Outcome::Miss;
            }
            let x = self.at(t);
            let (cut, lo, hi) = self.engine.bounds(x);
            let (glo, ghi) = (lo - self.offset, hi - self.offset);
            // Where the bounds decide the sign, `|g|` is at least the
            // bound, and `g` is 1-Lipschitz along the ray: no root within
            // that step, so no root is skipped and the last exact point
            // goes stale.
            if glo > self.engine.margin {
                t += glo;
                prev = None;
                continue;
            }
            if ghi < -self.engine.margin {
                t += -ghi;
                prev = None;
                continue;
            }
            self.steps += 1;
            let s = self.engine.signed_from(x, &cut);
            let g = s.d - self.offset;
            let gp = dot(s.normal, self.direction);
            let d = s.d.abs();
            // A sign change since the last exact point: the first root is
            // in between (round-off can carry a step across a root that
            // the step bound itself would have stopped at).
            if let Some((tp, gpv, gpp)) = prev
                && (g == 0.0 || g.signum() != gpv.signum())
            {
                let root = if g == 0.0 {
                    t
                } else {
                    self.bracket(tp, gpv, gpp, t, g)
                };
                match self.accept(root) {
                    Some(r) => return Outcome::Root(r),
                    None => {
                        t = self.past(root);
                        prev = None;
                        continue;
                    }
                }
            }
            let (w, v) = (g.abs(), -g.signum() * gp);
            if v > 0.0 {
                // Approaching: Newton's estimate of the crossing.
                let tn = t + w / v;
                if w / v <= Self::tolerance(tn) {
                    match self.accept(tn) {
                        Some(r) => return Outcome::Root(r),
                        None => {
                            t = self.past(tn);
                            prev = None;
                            continue;
                        }
                    }
                }
                if tn <= t_end {
                    let (gn, _, _) = self.exact(tn);
                    if gn == 0.0 || gn.signum() != g.signum() {
                        let root = if gn == 0.0 {
                            tn
                        } else {
                            self.bracket(t, g, gp, tn, gn)
                        };
                        match self.accept(root) {
                            Some(r) => return Outcome::Root(r),
                            None => {
                                t = self.past(root);
                                prev = None;
                                continue;
                            }
                        }
                    }
                }
            }
            prev = Some((t, g, gp));
            t += self.safe_step(w, v, d);
        }
        Outcome::Exhausted(t)
    }

    /// A root at or below `MIN_ROOT` is dropped (see there).
    fn accept(&self, root: f64) -> Option<f64> {
        (root > MIN_ROOT).then_some(root)
    }

    /// Where to resume after a dropped root: past the chord within which
    /// the level set cannot be met again.
    fn past(&mut self, root: f64) -> f64 {
        let (_, gp, d) = self.exact(root);
        root + self.chord(0.0, gp, d).max(H_MIN)
    }
}

enum Outcome {
    Root(f64),
    Miss,
    /// The evaluation budget ran out at this ray distance.
    Exhausted(f64),
}

impl Engine {
    /// Distance along the ray to the first crossing of the level set
    /// `D = offset` beyond `MIN_ROOT`, or `None` (also when the crossing is
    /// beyond `t_max`, where the search stops).
    ///
    /// `on_surface` is `Some(offset)` when the particle has just crossed
    /// this level set and sits that far past the crossing. Without it a
    /// particle within `ON_BAND` of the level set is recognised from the
    /// distance itself: moving towards the level set, the crossing ahead
    /// is returned; moving away, the search starts beyond the chord within
    /// which the level set cannot be met again. `vacuum` surfaces never
    /// return an unverified crossing when the evaluation budget runs out
    /// (transport kills a particle there); transmission surfaces return
    /// the point reached, which spatial relocation then resolves.
    pub(super) fn ray(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        offset: f64,
        on_surface: Option<f64>,
        vacuum: bool,
        t_max: f64,
    ) -> Option<f64> {
        let mut bx = self.patches.extents;
        bx.inflate(offset.max(0.0) + self.margin);
        let (t0, t1) = bx.clip(origin, direction)?;
        let t1 = t1.min(t_max);
        if t1 <= MIN_ROOT {
            return None;
        }
        let mut ray = Ray {
            engine: self,
            origin,
            direction,
            offset,
            steps: 0,
        };
        let mut start = t0.max(0.0);
        if t0 <= 0.0 {
            let (_, lo, hi) = self.bounds(origin);
            let near = lo - offset <= ON_BAND && hi - offset >= -ON_BAND;
            if on_surface.is_some() || near {
                let (g0, gp0, d0) = ray.exact(0.0);
                let leaving = g0 * gp0 > 0.0 || gp0 == 0.0;
                if on_surface.is_some() || (g0.abs() <= ON_BAND && leaving) {
                    // Moving off the level set (or told it was just
                    // crossed): skip the chord within which it cannot be
                    // met again.
                    start = ray.chord(g0.abs(), gp0, d0).max(H_MIN);
                } else if g0.abs() <= ON_BAND {
                    // Within the band and moving towards the level set:
                    // the crossing is just ahead and must be returned.
                    let t_hi = (2.0 * g0.abs() + self.margin) / gp0.abs();
                    let (g_hi, _, _) = ray.exact(t_hi);
                    if g_hi == 0.0 || g_hi.signum() != g0.signum() {
                        let root = if g_hi == 0.0 {
                            t_hi
                        } else {
                            ray.bracket(0.0, g0, gp0, t_hi, g_hi)
                        };
                        if root > MIN_ROOT {
                            return Some(root);
                        }
                        let (_, gpr, dr) = ray.exact(root);
                        start = root + ray.chord(0.0, gpr, dr).max(H_MIN);
                    } else {
                        start = t_hi;
                    }
                }
            }
        }
        match ray.run(start, t1) {
            Outcome::Root(t) => Some(t),
            Outcome::Miss => None,
            Outcome::Exhausted(_) if vacuum => None,
            Outcome::Exhausted(t) => Some(t.max(2.0 * MIN_ROOT)),
        }
    }
}
