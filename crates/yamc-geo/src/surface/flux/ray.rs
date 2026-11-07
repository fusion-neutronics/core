//! First crossing of a ray with a level set `D(x) = offset` of the signed
//! distance to the surface.
//!
//! `g(t) = D(o + t u) - offset` is 1-Lipschitz along a unit ray, so from a
//! point where `|g| >= w` no crossing lies within `w`; that is the step
//! wherever rigorous bounds decide the sign. Within the surface's reach on
//! each side `D` is smooth and `|g''|` is bounded by the curvature of its
//! level sets (side-specific: outside, convex parts flatten and only
//! concave ones tighten; inside, the reverse). That gives longer
//! second-order steps, rules out a pair of crossings between two samples
//! and certifies a bracket as holding a single crossing. Approaching a
//! crossing, a probe just past Newton's estimate forms the bracket, which
//! safeguarded Newton then narrows; a bracket is never returned unrefined.
//!
//! Beyond the reach (an offset layer past its fold, or past half a thin
//! part) only the Lipschitz steps apply. A single crossing is still never
//! missed, but a pair closer together than a probe's overshoot may be: a
//! sliver of the far side of a creased layer that the particle crosses
//! without stopping.

use super::distance::Signed;
use super::Engine;

/// Roots at or below this ray distance are dropped and the search goes
/// on: `Cell::closest_surface` ignores distances up to 1e-10 cm, so
/// returning one would hide this surface's next crossing.
pub(super) const MIN_ROOT: f64 = 1e-10;
/// A pair of crossings closer together than this (cm) may be missed; a
/// ray would have to graze the surface within about 1e-9 rad of tangency
/// for it to matter.
const H_MIN: f64 = 1e-9;
/// Evaluations per query before giving up. A transmission surface then
/// returns the root-free point reached, which spatial relocation resolves;
/// a vacuum surface, where a returned crossing kills the particle, gets the
/// far larger budget and reports no crossing only if that runs out too.
const BUDGET: usize = 4_000;
const VACUUM_BUDGET: usize = 400_000;
/// Distance from the level set within which a particle counts as on it
/// (well above the 1e-8 cm crossing nudge).
const ON_BAND: f64 = 1e-7;
/// A crossing is located to this (cm), far below the 1e-8 cm nudge.
const ROOT_TOL: f64 = 1e-11;
/// The probe lands this far past Newton's estimate, relatively.
const OVERSHOOT: f64 = 0.25;
/// Only probe when Newton's estimate is within this many safe steps.
const PROBE_REACH: f64 = 8.0;
/// Fraction of the reach within which the curvature bound is used.
const REACH_USE: f64 = 0.9;

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// How far a function at `|g| = w`, moving away from zero at rate `away`
/// (negative: towards it) with `|g''| <= k`, is sure not to reach zero.
fn clearance(w: f64, away: f64, k: f64) -> f64 {
    let root = (away * away + 2.0 * k * w).sqrt();
    if away >= 0.0 {
        if k > 0.0 {
            (away + root) / k
        } else {
            f64::INFINITY
        }
    } else {
        // (root - |away|) / k, without the cancellation.
        let den = root - away;
        if den > 0.0 {
            2.0 * w / den
        } else {
            f64::INFINITY
        }
    }
}

/// A point on the ray: `g`, its slope along the ray, and the signed `D`.
#[derive(Clone, Copy)]
struct Sample {
    t: f64,
    g: f64,
    slope: f64,
    d: f64,
}

struct Ray<'a> {
    engine: &'a Engine,
    origin: [f64; 3],
    direction: [f64; 3],
    offset: f64,
    evaluations: usize,
    budget: usize,
}

enum Outcome {
    Root(f64),
    Miss,
    /// The budget ran out; no crossing lies before this ray distance.
    Exhausted(f64),
}

impl Ray<'_> {
    fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    fn spent(&self) -> bool {
        self.evaluations >= self.budget
    }

    fn sample(&self, t: f64, s: Signed) -> Sample {
        Sample {
            t,
            g: s.d - self.offset,
            slope: dot(s.normal, self.direction),
            d: s.d,
        }
    }

    fn exact(&mut self, t: f64) -> Sample {
        self.evaluations += 1;
        let s = self.engine.signed(self.at(t));
        self.sample(t, s)
    }

    /// Bound on `|g''|` while the signed distance stays in `[lo, hi]`, if
    /// that range is inside the reach on each side it touches.
    fn curvature(&self, lo: f64, hi: f64) -> Option<f64> {
        let e = self.engine;
        let (kv, kc) = (e.kappa_convex, e.kappa_concave);
        let mut k = 0.0_f64;
        if hi > 0.0 {
            let (d1, d2) = (lo.max(0.0), hi);
            if d2 >= REACH_USE * e.reach_out {
                return None;
            }
            k = k.max(kv / (1.0 + d1 * kv)).max(kc / (1.0 - d2 * kc));
        }
        if lo < 0.0 {
            let (e1, e2) = ((-hi).max(0.0), -lo);
            if e2 >= REACH_USE * e.reach_in {
                return None;
            }
            k = k.max(kv / (1.0 - e2 * kv)).max(kc / (1.0 + e1 * kc));
        }
        Some(k)
    }

    /// Half the distance from `d` to the nearer edge of the range where
    /// the curvature bound holds (negative outside it): steps from a point
    /// at `d` use the bound over `[d - window, d + window]`.
    fn window(&self, d: f64) -> f64 {
        let e = self.engine;
        0.5 * (REACH_USE * e.reach_out - d).min(d + REACH_USE * e.reach_in)
    }

    /// The longest step from `s` within which `g` cannot vanish: the
    /// Lipschitz step `|g|`, or the second-order one where it is valid.
    fn safe_step(&self, s: &Sample) -> f64 {
        let w = s.g.abs();
        let h = self.window(s.d);
        if h <= w {
            return w;
        }
        match self.curvature(s.d - h, s.d + h) {
            Some(k) => w.max(clearance(w, s.g.signum() * s.slope, k).min(h)),
            None => w,
        }
    }

    /// How far a point on (or within `|g|` of) the level set, moving off
    /// it, can go before it could meet it again.
    fn chord(&self, s: &Sample) -> f64 {
        let h = self.window(s.d);
        if h <= 0.0 {
            return 0.0;
        }
        match self.curvature(s.d - h, s.d + h) {
            Some(k) => (0.99 * clearance(s.g.abs(), s.slope.abs(), k)).min(h),
            None => 0.0,
        }
    }

    fn tolerance(t: f64) -> f64 {
        (4.0 * f64::EPSILON * t.abs()).max(ROOT_TOL)
    }

    /// Bound on `|g''|` over `[a, b]`, if that stretch is inside the reach.
    fn curvature_between(&self, a: &Sample, b: &Sample) -> Option<f64> {
        let half = 0.5 * (b.t - a.t);
        self.curvature(a.d.min(b.d) - half, a.d.max(b.d) + half)
    }

    /// No crossing at all in `[a, b]` (same sign at both ends)? Proven by
    /// the 1-Lipschitz bound or, where valid, by the curvature bound from
    /// both ends.
    fn excluded(&self, a: &Sample, b: &Sample) -> bool {
        let width = b.t - a.t;
        if a.g.abs() + b.g.abs() > width {
            return true;
        }
        let Some(k) = self.curvature_between(a, b) else {
            return false;
        };
        let sign = a.g.signum();
        clearance(a.g.abs(), sign * a.slope, k) + clearance(b.g.abs(), -sign * b.slope, k) > width
    }

    /// The first crossing in `[a, b]`, both ends of the same sign and `a`
    /// root-free; `Ok(None)` if there is none (pairs closer than `H_MIN`
    /// may be missed, and beyond the reach, where only the Lipschitz bound
    /// holds and ruling a pair out would take `O(width / |g|)` evaluations,
    /// none is looked for), `Err(t)` if the budget ran out with no crossing
    /// before `t`.
    fn scan(&mut self, a: Sample, b: Sample) -> Result<Option<f64>, f64> {
        if b.t - a.t <= H_MIN || self.excluded(&a, &b) || self.curvature_between(&a, &b).is_none() {
            return Ok(None);
        }
        if self.spent() {
            return Err(a.t);
        }
        let m = self.exact(0.5 * (a.t + b.t));
        if m.g == 0.0 {
            return Ok(Some(m.t));
        }
        if m.g.signum() != a.g.signum() {
            return self.bracket(a, m).map(Some);
        }
        match self.scan(a, m)? {
            Some(root) => Ok(Some(root)),
            None => self.scan(m, b),
        }
    }

    /// The first crossing in `(a, b]` given a sign change and a root-free
    /// `a`, located to the tolerance (at or just past it), or `Err(t)` if
    /// the budget ran out with no crossing before `t`.
    fn bracket(&mut self, mut a: Sample, mut b: Sample) -> Result<f64, f64> {
        // Newton runs from the end evaluated last while it keeps halving
        // the bracket; otherwise the step bisects.
        let mut from_b = true;
        let mut halving = true;
        loop {
            let width = b.t - a.t;
            let tol = Self::tolerance(b.t);
            if width <= tol {
                return Ok(b.t);
            }
            // A floor on |g'| over the bracket: if positive, g is monotone
            // there, the crossing is the only one, and it lies within
            // |g| / floor of either end.
            let floor = self
                .curvature_between(&a, &b)
                .map_or(0.0, |k| a.slope.abs().max(b.slope.abs()) - k * width);
            if floor > 0.0 {
                if b.g.abs() <= tol * floor {
                    return Ok(b.t);
                }
                if a.g.abs() <= tol * floor {
                    return Ok((a.t + a.g.abs() / floor).min(b.t));
                }
            }
            if self.spent() {
                return Err(a.t);
            }
            let from = if from_b { &b } else { &a };
            let newton = from.t - from.g / from.slope;
            let tm = if halving && newton > a.t && newton < b.t {
                newton
            } else {
                0.5 * (a.t + b.t)
            };
            let m = self.exact(tm);
            if m.g == 0.0 {
                return Ok(m.t);
            }
            if m.g.signum() != a.g.signum() {
                b = m;
                from_b = true;
            } else {
                if floor <= 0.0
                    && let Some(root) = self.scan(a, m)?
                {
                    return Ok(root);
                }
                a = m;
                from_b = false;
            }
            halving = b.t - a.t <= 0.5 * width;
        }
    }

    /// Turn a found crossing into the outcome, or the ray distance to
    /// resume from when it is at or below `MIN_ROOT` (see there).
    fn settle(&mut self, found: Result<f64, f64>) -> Result<Outcome, f64> {
        match found {
            Ok(root) if root > MIN_ROOT => Ok(Outcome::Root(root)),
            Ok(root) => {
                // Past the chord within which the level set cannot be met
                // again.
                let s = self.exact(root);
                Err(root + self.chord(&s).max(H_MIN))
            }
            Err(safe) => Ok(Outcome::Exhausted(safe)),
        }
    }

    /// March from `start` (with the exact sample there, if taken) to `end`.
    fn run(&mut self, start: f64, end: f64, here: Option<Sample>) -> Outcome {
        let mut t = start;
        let mut next = here;
        // The last exact sample: no crossing lies between it and `t`.
        let mut prev: Option<Sample> = None;
        // The last step: a loose bound gives steps far shorter than an
        // exact sample would, and is then not worth taking.
        let mut last_step = 0.0_f64;
        // The bounds were no use at this `|g|`: nearer the level set than
        // twice that, sample exactly straight away.
        let mut loose_below = 0.0_f64;
        loop {
            if t > end {
                return Outcome::Miss;
            }
            if self.spent() {
                return Outcome::Exhausted(t);
            }
            let s = match next.take() {
                Some(s) => s,
                None if prev.is_some_and(|p| p.g.abs() < loose_below) => self.exact(t),
                None => {
                    self.evaluations += 1;
                    let x = self.at(t);
                    let (cut, lo, hi) = self.engine.bounds(x);
                    let (glo, ghi) = (lo - self.offset, hi - self.offset);
                    // Where the bounds decide the sign with room to spare,
                    // step by them (g is 1-Lipschitz, so no crossing lies
                    // within the bound), unless they are loose (the step
                    // short against the other bound) or the step far
                    // shorter than the last: an exact sample then steps
                    // much further.
                    let room = (1e-6 * t.abs()).max(1e-9);
                    let decided = if glo > room {
                        Some((1.0, glo, ghi))
                    } else if ghi < -room {
                        Some((-1.0, -ghi, -glo))
                    } else {
                        None
                    };
                    if let Some((sign, step, other)) = decided {
                        // A crossing since `prev` that a step went over (a
                        // safety net: steps are built not to).
                        if let Some(p) = prev
                            && p.g.signum() != sign
                        {
                            let s = self.exact(t);
                            let found = self.bracket(p, s);
                            match self.settle(found) {
                                Ok(outcome) => return outcome,
                                Err(resume) => {
                                    t = resume;
                                    prev = None;
                                    last_step = 0.0;
                                    continue;
                                }
                            }
                        }
                        if step >= 0.25 * other && step >= 0.5 * last_step {
                            t += step;
                            last_step = step;
                            continue;
                        }
                    }
                    let signed = self.engine.signed_from(x, &cut);
                    let s = self.sample(t, signed);
                    loose_below = 2.0 * s.g.abs();
                    s
                }
            };
            // On the level set, or a sign change since the last exact
            // sample (round-off can carry a step across a crossing).
            let found = if s.g == 0.0 {
                Some(Ok(t))
            } else {
                match prev {
                    Some(p) if p.g.signum() != s.g.signum() => Some(self.bracket(p, s)),
                    _ => None,
                }
            };
            if let Some(found) = found {
                match self.settle(found) {
                    Ok(outcome) => return outcome,
                    Err(resume) => {
                        t = resume;
                        prev = None;
                        last_step = 0.0;
                        continue;
                    }
                }
            }
            let step = self.safe_step(&s);
            let away = s.g.signum() * s.slope;
            if away < 0.0 {
                let newton = s.g.abs() / -away;
                let found = if newton <= Self::tolerance(t + newton) {
                    Some(Ok(t + newton))
                } else {
                    // Probe just past Newton's estimate to form a bracket.
                    let reach = newton * (1.0 + OVERSHOOT) + ROOT_TOL;
                    if reach > step && reach <= PROBE_REACH * step && t + reach <= end {
                        let probe = self.exact(t + reach);
                        if probe.g == 0.0 {
                            Some(Ok(probe.t))
                        } else if probe.g.signum() != s.g.signum() {
                            Some(self.bracket(s, probe))
                        } else if self.excluded(&s, &probe) {
                            // Nothing up to the probe: carry on from there.
                            prev = Some(s);
                            last_step = probe.t - t;
                            t = probe.t;
                            next = Some(probe);
                            continue;
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                if let Some(found) = found {
                    match self.settle(found) {
                        Ok(outcome) => return outcome,
                        Err(resume) => {
                            t = resume;
                            prev = None;
                            last_step = 0.0;
                            continue;
                        }
                    }
                }
            }
            prev = Some(s);
            last_step = step;
            t += step;
        }
    }
}

/// A ray query's answer and what it cost.
pub(super) struct Trace {
    pub crossing: Option<f64>,
    /// The evaluation budget ran out.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by the tests"))]
    pub exhausted: bool,
    /// Bound and exact distance evaluations.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by the tests"))]
    pub evaluations: usize,
}

impl Engine {
    /// Distance along the ray (`direction` of unit length) to the first
    /// crossing of the level set `D = offset` beyond `MIN_ROOT`, or `None`
    /// (also when the crossing is beyond `t_max`, where the search stops).
    ///
    /// A particle within `ON_BAND` of the level set and moving off it (it
    /// has just crossed, `on_surface` says so, or it was left there) starts
    /// beyond the chord within which the level set cannot be met again;
    /// one moving towards it marches from where it is, so the crossing just
    /// ahead is returned. If the evaluation budget runs out, a transmission
    /// surface returns the root-free point reached (spatial relocation
    /// resolves it) and a vacuum surface `None`; the vacuum budget is large
    /// enough that this does not happen in practice.
    pub(super) fn ray(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        offset: f64,
        on_surface: Option<f64>,
        vacuum: bool,
        t_max: f64,
    ) -> Option<f64> {
        self.trace_ray(origin, direction, offset, on_surface, vacuum, t_max)
            .crossing
    }

    /// [`Engine::ray`] with what the query cost.
    pub(super) fn trace_ray(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        offset: f64,
        on_surface: Option<f64>,
        vacuum: bool,
        t_max: f64,
    ) -> Trace {
        let none = Trace {
            crossing: None,
            exhausted: false,
            evaluations: 0,
        };
        let mut bx = self.patches.extents;
        bx.inflate(offset.max(0.0) + self.margin);
        let Some((t0, t1)) = bx.clip(origin, direction) else {
            return none;
        };
        let end = t1.min(t_max);
        if end <= MIN_ROOT {
            return none;
        }
        let mut ray = Ray {
            engine: self,
            origin,
            direction,
            offset,
            evaluations: 0,
            budget: if vacuum { VACUUM_BUDGET } else { BUDGET },
        };
        let mut start = t0.max(0.0);
        let mut here = None;
        if t0 <= 0.0 {
            let near = on_surface.is_some() || {
                ray.evaluations += 1;
                let (_, lo, hi) = self.bounds(origin);
                lo - offset <= ON_BAND && hi - offset >= -ON_BAND
            };
            if near {
                let s = ray.exact(0.0);
                if s.g.abs() <= ON_BAND && s.g * s.slope >= 0.0 {
                    start = ray.chord(&s).max(H_MIN);
                } else {
                    here = Some(s);
                }
            }
        }
        let outcome = ray.run(start, end, here);
        let (crossing, exhausted) = match outcome {
            Outcome::Root(t) => (Some(t), false),
            Outcome::Miss => (None, false),
            Outcome::Exhausted(_) if vacuum => (None, true),
            Outcome::Exhausted(t) => (Some(t.max(2.0 * MIN_ROOT)), true),
        };
        Trace {
            crossing,
            exhausted,
            evaluations: ray.evaluations,
        }
    }
}
