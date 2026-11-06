//! The certified ray search the free-form surface kinds share.
//!
//! A free-form kind finds where the ray `o + t u` (`u` of unit length, so
//! `t` is in cm) first crosses its surface by searching a function `g(t)`
//! whose sign along the ray is the sign of the surface's `evaluate`. Each
//! kind steps along the ray in its own way, ruling stretches out with its
//! own bounds; what they share is here:
//!
//! - the constants and the root [`tolerance`];
//! - the evaluation [`Budget`], and the [`Outcome`] of a search with the
//!   answer a surface gives when the budget runs out ([`Outcome::trace`]);
//! - dropping roots at or below [`MIN_ROOT`] and searching on
//!   ([`settle`]);
//! - the on-surface rule ([`leaving`]);
//! - narrowing a bracket to the crossing ([`bracket`]), with the kind's
//!   certificate that the bracket holds a single crossing;
//! - running a search on the unit direction ([`along_unit`]).

/// Roots at or below this ray distance are dropped and the search goes
/// on: `Cell::closest_surface` ignores distances up to 1e-10 cm, so
/// returning one would hide the surface's next crossing.
pub(super) const MIN_ROOT: f64 = 1e-10;
/// A pair of crossings closer together than this (cm) may be missed; a
/// ray would have to graze a smooth surface within about 1e-9 rad of
/// tangency for it to matter.
pub(super) const H_MIN: f64 = 1e-9;
/// A crossing is located to this (cm), far below the 1e-8 cm nudge.
pub(super) const ROOT_TOL: f64 = 1e-11;
/// Distance from the surface within which a particle counts as on it
/// (well above the 1e-8 cm crossing nudge).
pub(super) const ON_BAND: f64 = 1e-7;
/// Evaluations per query before giving up (see [`Outcome::trace`]).
const BUDGET: usize = 4_000;
const VACUUM_BUDGET: usize = 400_000;

/// What a crossing at ray distance `t` is located to: [`ROOT_TOL`], or a
/// few ulps of `t` where that is more.
pub(super) fn tolerance(t: f64) -> f64 {
    (4.0 * f64::EPSILON * t.abs()).max(ROOT_TOL)
}

/// The middle of the stretch `[a, b]` of the ray.
pub(super) fn midpoint(a: f64, b: f64) -> f64 {
    0.5 * (a + b)
}

/// Whether the stretch `[a, b]` is too short to split: no longer than
/// [`H_MIN`], or so far along the ray that its middle rounds onto an end.
pub(super) fn minimal(a: f64, b: f64) -> bool {
    let m = midpoint(a, b);
    !(b - a > H_MIN && a < m && m < b)
}

/// The evaluations a query may spend: [`BUDGET`] on a transmission
/// surface, [`VACUUM_BUDGET`] on a vacuum one.
pub(super) struct Budget {
    used: usize,
    limit: usize,
}

impl Budget {
    pub(super) fn new(vacuum: bool) -> Self {
        Self {
            used: 0,
            limit: if vacuum { VACUUM_BUDGET } else { BUDGET },
        }
    }

    /// Count one evaluation.
    pub(super) fn spend(&mut self) {
        self.used += 1;
    }

    /// Whether the budget has run out.
    pub(super) fn spent(&self) -> bool {
        self.used >= self.limit
    }

    /// Evaluations spent so far.
    pub(super) fn used(&self) -> usize {
        self.used
    }
}

/// How a search along a ray ended.
pub(super) enum Outcome {
    /// The first crossing beyond [`MIN_ROOT`].
    Root(f64),
    /// No crossing up to the end of the stretch searched.
    Miss,
    /// The budget ran out; no crossing lies before this ray distance.
    Exhausted(f64),
}

/// A ray query's answer and what it cost.
pub(super) struct Trace {
    pub crossing: Option<f64>,
    /// The evaluation budget ran out.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by the tests"))]
    pub exhausted: bool,
    /// Evaluations spent.
    #[cfg_attr(not(test), expect(dead_code, reason = "read by the tests"))]
    pub evaluations: usize,
}

impl Outcome {
    /// The answer of a surface whose search ended so, after `evaluations`.
    /// If the budget ran out, a transmission surface returns the root-free
    /// point reached, which spatial relocation resolves, and a vacuum
    /// surface, where a returned crossing kills the particle (its budget is
    /// large enough that this does not happen in practice), no crossing.
    pub(super) fn trace(self, vacuum: bool, evaluations: usize) -> Trace {
        let (crossing, exhausted) = match self {
            Outcome::Root(t) => (Some(t), false),
            Outcome::Miss => (None, false),
            Outcome::Exhausted(_) if vacuum => (None, true),
            Outcome::Exhausted(t) => (Some(t.max(2.0 * MIN_ROOT)), true),
        };
        Trace {
            crossing,
            exhausted,
            evaluations,
        }
    }
}

/// Turn a crossing found at or just past `t` (`Ok(t)`), or a budget that
/// ran out with no crossing before `t` (`Err(t)`), into the outcome. A
/// crossing at or below [`MIN_ROOT`] is dropped (see there): the result is
/// then `Err` with the ray distance to search on from, `resume(t)`, past
/// the stretch over which the kind shows the surface is not met again.
pub(super) fn settle(
    found: Result<f64, f64>,
    resume: impl FnOnce(f64) -> f64,
) -> Result<Outcome, f64> {
    match found {
        Ok(root) if root > MIN_ROOT => Ok(Outcome::Root(root)),
        Ok(root) => Err(resume(root)),
        Err(safe) => Ok(Outcome::Exhausted(safe)),
    }
}

/// The on-surface rule, for a particle at the start of the ray where `g`
/// has slope `slope` and the kind puts it `distance` (cm) from the
/// surface. On the surface to within [`ON_BAND`] and not moving towards it
/// (moving along it counts, as does a start exactly on it), the particle
/// has just crossed it, or was left there, so the crossing at its feet is
/// not one to return: its search starts past the stretch over which the
/// kind shows the surface is not met again. Any other particle searches
/// from where it is, so the crossing just ahead is returned. Signs are
/// compared, not multiplied: a product can underflow to zero and pass a
/// particle heading for the surface off as leaving it.
pub(super) fn leaving(distance: f64, g: f64, slope: f64) -> bool {
    use std::cmp::Ordering::Equal;
    let away = match (g.partial_cmp(&0.0), slope.partial_cmp(&0.0)) {
        (Some(side), Some(heading)) => side == Equal || heading == Equal || side == heading,
        _ => false,
    };
    distance <= ON_BAND && away
}

/// A point of the ray where a kind has evaluated `g` exactly.
pub(super) trait Point: Copy {
    /// Its ray distance.
    fn t(&self) -> f64;
    /// `g` there.
    fn g(&self) -> f64;
    /// The derivative of `g` along the ray there.
    fn slope(&self) -> f64;
}

/// What [`bracket`] needs of a kind's search along one ray.
pub(super) trait Line {
    type Point: Point;

    /// `g` at ray distance `t`, evaluated exactly rather than bounded (one
    /// evaluation).
    fn exact(&mut self, t: f64) -> Self::Point;

    /// Whether the evaluation budget has run out.
    fn spent(&self) -> bool;

    /// The certificate: a lower bound on `|g'|` over `[a, b]`, or 0 (or
    /// less) if the kind has none. Where it is positive `g` is strictly
    /// monotone, so the bracket holds a single crossing, within `|g| /
    /// floor` of either end.
    fn floor(&self, a: &Self::Point, b: &Self::Point) -> f64;

    /// The first crossing in `[a, b]`, where `g` has the same sign at both
    /// ends and `a` is root-free, or `Ok(None)` if there is none; `Err(t)`
    /// if the budget ran out with no crossing before `t`.
    fn scan(&mut self, a: Self::Point, b: Self::Point) -> Result<Option<f64>, f64>;

    /// Whether Newton's step is taken from `b` rather than `a`, `last_b`
    /// saying which was evaluated last: by default that one.
    fn newton_from_b(&self, a: &Self::Point, b: &Self::Point, last_b: bool) -> bool {
        let _ = (a, b);
        last_b
    }
}

/// The first crossing in `(a, b]`, where `g` is nonzero at both ends,
/// with opposite signs, and `a` is root-free: located to the [`tolerance`]
/// at or just past it; `Err(t)` if the budget ran out with no crossing
/// before `t`.
///
/// Newton's method runs from the end evaluated last (or the one the kind
/// picks, [`Line::newton_from_b`]) while each step at least halves the
/// bracket; otherwise the step bisects (so does a Newton step that is not
/// finite or does not land strictly inside), and the bracket is never
/// returned unrefined. The answer is its far end, whose sign has been
/// checked to be across the crossing, a probe where `g` is exactly 0, or,
/// where the certificate shows the single crossing lies within the
/// tolerance of the near end, the near end's first-order estimate of it.
/// A probe on the near side of an uncertified bracket gets the stretch
/// behind it scanned for a pair of crossings its ends cannot show.
pub(super) fn bracket<L: Line>(line: &mut L, mut a: L::Point, mut b: L::Point) -> Result<f64, f64> {
    let mut from_b = true;
    let mut halving = true;
    loop {
        let width = b.t() - a.t();
        let tol = tolerance(b.t());
        if width <= tol {
            return Ok(b.t());
        }
        let floor = line.floor(&a, &b);
        if floor > 0.0 {
            if b.g().abs() <= tol * floor {
                return Ok(b.t());
            }
            if a.g().abs() <= tol * floor {
                return Ok((a.t() + a.g().abs() / floor).min(b.t()));
            }
        }
        if line.spent() {
            return Err(a.t());
        }
        let from = if line.newton_from_b(&a, &b, from_b) {
            &b
        } else {
            &a
        };
        let newton = from.t() - from.g() / from.slope();
        let tm = if halving && newton > a.t() && newton < b.t() {
            newton
        } else {
            midpoint(a.t(), b.t())
        };
        let m = line.exact(tm);
        if m.g() == 0.0 {
            return Ok(m.t());
        }
        if m.g().signum() != a.g().signum() {
            b = m;
            from_b = true;
        } else {
            if floor <= 0.0
                && let Some(root) = line.scan(a, m)?
            {
                return Ok(root);
            }
            a = m;
            from_b = false;
        }
        halving = b.t() - a.t() <= 0.5 * width;
    }
}

/// Run `search` on the ray `point + t * direction` along the unit
/// direction, as the constants here are lengths: `search` is given the
/// unit direction and the cap `t_max` in its units and answers in them;
/// the answer comes back in the units of `direction`. `None` for a zero or
/// non-finite direction or a non-finite point.
pub(super) fn along_unit(
    point: [f64; 3],
    direction: [f64; 3],
    t_max: f64,
    search: impl FnOnce([f64; 3], f64) -> Option<f64>,
) -> Option<f64> {
    let len = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(len.is_finite() && len > 0.0) || !point.iter().all(|v| v.is_finite()) {
        return None;
    }
    let unit = direction.map(|v| v / len);
    search(unit, t_max * len).map(|t| t / len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `g` given as a function of the ray distance, returning `(g, g')`;
    /// Newton's step from the end nearer the crossing to first order if
    /// `nearer`.
    struct Analytic<F> {
        g: F,
        floor: f64,
        nearer: bool,
        budget: Budget,
    }

    #[derive(Clone, Copy)]
    struct At {
        t: f64,
        g: f64,
        slope: f64,
    }

    impl Point for At {
        fn t(&self) -> f64 {
            self.t
        }

        fn g(&self) -> f64 {
            self.g
        }

        fn slope(&self) -> f64 {
            self.slope
        }
    }

    impl<F: Fn(f64) -> (f64, f64)> Line for Analytic<F> {
        type Point = At;

        fn exact(&mut self, t: f64) -> At {
            self.budget.spend();
            let (g, slope) = (self.g)(t);
            At { t, g, slope }
        }

        fn spent(&self) -> bool {
            self.budget.spent()
        }

        fn floor(&self, _: &At, _: &At) -> f64 {
            self.floor
        }

        fn scan(&mut self, _: At, _: At) -> Result<Option<f64>, f64> {
            Ok(None)
        }

        fn newton_from_b(&self, a: &At, b: &At, last_b: bool) -> bool {
            if self.nearer {
                (b.g / b.slope).abs() <= (a.g / a.slope).abs()
            } else {
                last_b
            }
        }
    }

    /// The crossing `bracket` finds in `[a, b]`, and the evaluations it
    /// took.
    fn crossing(g: impl Fn(f64) -> (f64, f64), floor: f64, a: f64, b: f64) -> (f64, usize) {
        crossing_from(g, floor, false, a, b)
    }

    /// [`crossing`], with Newton's step from the nearer end if `nearer`.
    fn crossing_from(
        g: impl Fn(f64) -> (f64, f64),
        floor: f64,
        nearer: bool,
        a: f64,
        b: f64,
    ) -> (f64, usize) {
        let mut line = Analytic {
            g,
            floor,
            nearer,
            budget: Budget::new(false),
        };
        let (a, b) = (line.exact(a), line.exact(b));
        let root = bracket(&mut line, a, b).unwrap();
        (root, line.budget.used())
    }

    #[test]
    fn brackets_close_on_the_crossing() {
        // A root at 4/3, with and without a certificate (|g'| >= 2 on
        // [1, 3], = 1 for the line), located to the tolerance; the line's
        // is never short of its sign change.
        let r = 4.0 / 3.0;
        for certified in [false, true] {
            let floor = if certified { 2.0 } else { 0.0 };
            let (t, _) = crossing(|t| (t * t - r * r, 2.0 * t), floor, 1.0, 3.0);
            assert!((t - r).abs() <= tolerance(r), "{t}");
            let floor = if certified { 1.0 } else { 0.0 };
            let (t, _) = crossing(|t| (r - t, -1.0), floor, 0.25, 2.0);
            assert!(t >= r && t - r <= tolerance(r), "{t}");
        }
    }

    #[test]
    fn an_overflowing_slope_falls_back_to_bisection() {
        // exp(k (t - r)) - 1: Newton's step from the far end is tiny, or
        // inf / inf.
        let r = 1.1;
        for k in [300.0, 700.0, 1e4] {
            let (t, evaluations) = crossing(
                |t: f64| ((k * (t - r)).exp() - 1.0, k * (k * (t - r)).exp()),
                0.0,
                -1.0,
                3.0,
            );
            assert!((t - r).abs() <= tolerance(r), "k {k}: {t}");
            assert!(evaluations < 200, "k {k}: {evaluations} evaluations");
        }
    }

    #[test]
    fn a_flat_crossing_still_converges() {
        // A triple root: Newton creeps towards it from one side, so the
        // steps that fail to halve the bracket bisect it.
        let cube = |t: f64| ((t - 1.0).powi(3), 3.0 * (t - 1.0).powi(2));
        let (t, evaluations) = crossing(cube, 0.0, 0.0, 5.0);
        assert!((t - 1.0).abs() <= tolerance(1.0), "{t}");
        assert!(evaluations < 200, "{evaluations} evaluations");
    }

    #[test]
    fn a_wide_bracket_closes_from_its_nearer_end() {
        // A ray into a sphere of radius 4 whose bracket is a whole monotone
        // stretch, its near end 2.3 cm short of the crossing (|g'| >= 0.33
        // over it). After a bisection the end evaluated last is the near
        // one, whose Newton step overshoots the far end; the far end's
        // converges.
        let o = [0.9167010010071905, -1.7772549394664718, -0.2794539186099012];
        let u = [-0.3762601550027884, 0.5199688612358522, 0.7668511453355004];
        let g = |t: f64| {
            let p: [f64; 3] = std::array::from_fn(|k| o[k] + t * u[k] - [0.3, -0.2, 0.1][k]);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            (r - 4.0, (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / r)
        };
        let (last, _) = crossing_from(g, 0.33, false, 2.8615911610632, 5.7231823220265);
        let (nearer, evaluations) = crossing_from(g, 0.33, true, 2.8615911610632, 5.7231823220265);
        assert!(
            (nearer - 5.189186152299).abs() <= tolerance(5.2),
            "{nearer}"
        );
        assert!((last - nearer).abs() <= tolerance(5.2), "{last} {nearer}");
        assert!(evaluations <= 10, "{evaluations} evaluations");
    }

    #[test]
    fn only_a_particle_not_heading_for_the_surface_is_leaving_it() {
        assert!(leaving(1e-8, 1e-8, 0.5));
        assert!(leaving(1e-8, -1e-8, -0.5));
        assert!(!leaving(1e-8, 1e-8, -0.5));
        assert!(!leaving(1e-8, -1e-8, 0.5));
        // On it, or moving along it.
        assert!(leaving(0.0, 0.0, -1.0));
        assert!(leaving(1e-8, -1e-8, 0.0));
        // Too far from it, or no sign to go by.
        assert!(!leaving(1e-6, 1e-6, 0.5));
        assert!(!leaving(0.0, f64::NAN, 0.5));
        // g * slope underflows to -0.0 here, which a product test reads as
        // leaving.
        assert_eq!(-1e-170_f64 * 1e-160, 0.0);
        assert!(!leaving(1e-170, -1e-170, 1e-160));
    }

    #[test]
    fn a_stretch_too_short_to_split_is_minimal() {
        assert!(minimal(1.0, 1.0 + 0.5 * H_MIN));
        assert!(!minimal(1.0, 1.0 + 2.0 * H_MIN));
        // Far along the ray neighbouring floats are more than H_MIN apart.
        let t = 1e8_f64;
        assert!(t.next_up() - t > H_MIN);
        assert!(minimal(t, t.next_up()));
        assert!(!minimal(t, t.next_up().next_up()));
        assert!(minimal(f64::NAN, t));
    }

    #[test]
    fn a_spent_budget_is_answered_by_the_boundary_type() {
        let transmission = Outcome::Exhausted(5.0).trace(false, 4_000);
        assert_eq!(transmission.crossing, Some(5.0));
        assert!(transmission.exhausted);
        // Never closer than the cell would look.
        let at_start = Outcome::Exhausted(0.0).trace(false, 4_000);
        assert_eq!(at_start.crossing, Some(2.0 * MIN_ROOT));
        let vacuum = Outcome::Exhausted(5.0).trace(true, VACUUM_BUDGET);
        assert_eq!(vacuum.crossing, None);
        assert!(vacuum.exhausted);
        let mut budget = Budget::new(false);
        for _ in 0..BUDGET {
            assert!(!budget.spent());
            budget.spend();
        }
        assert!(budget.spent());
    }

    #[test]
    fn a_root_at_the_start_is_dropped_and_the_search_resumes() {
        assert!(matches!(
            settle(Ok(1.0), |_| unreachable!()),
            Ok(Outcome::Root(1.0))
        ));
        assert_eq!(
            settle(Ok(MIN_ROOT), |t| t + 1.0).err(),
            Some(1.0 + MIN_ROOT)
        );
        assert!(matches!(
            settle(Err(3.0), |_| unreachable!()),
            Ok(Outcome::Exhausted(3.0))
        ));
    }

    #[test]
    fn a_search_runs_on_the_unit_direction() {
        let seen = std::cell::Cell::new(([0.0; 3], 0.0));
        let t = along_unit([1.0, 2.0, 3.0], [0.0, 0.0, 2.0], 10.0, |unit, t_max| {
            seen.set((unit, t_max));
            Some(4.0)
        });
        assert_eq!(seen.get(), ([0.0, 0.0, 1.0], 20.0));
        assert_eq!(t, Some(2.0));
        let never = |_: [f64; 3], _: f64| -> Option<f64> { unreachable!() };
        assert_eq!(along_unit([0.0; 3], [0.0; 3], 1.0, never), None);
        assert_eq!(
            along_unit([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, never),
            None
        );
        assert_eq!(
            along_unit([0.0; 3], [f64::INFINITY, 0.0, 0.0], 1.0, never),
            None
        );
    }
}
