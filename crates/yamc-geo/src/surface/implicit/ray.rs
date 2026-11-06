//! The first crossing of a ray with an implicit surface.
//!
//! The cell test reads the sign of the formula's point value `g(t)` at the
//! point [`ray_point`] computes along the ray `o + t u` (`u` of unit
//! length), and the interval evaluators enclose those very values. So the
//! search rules stretches of the ray out with enclosures and confirms each
//! crossing with point values, and what it finds is the first sign change
//! the cell test would see. A crossing is a point where `g` is strictly
//! across from the side the ray is on: where `g` is exactly 0 there is no
//! side, so a ray that only touches the surface does not cross it, and the
//! point returned for a crossing is always one where `g` is strictly
//! across, so the particle moved just past it is in a half-space.
//!
//! The ray is clipped to the formula's bounding box, or with no box to the
//! cube of [`WORLD`] within which the formula is checked. The clip is
//! rounded outward, so every point before and after the stretch kept is
//! outside the box, where the formula is proved positive: a ray that starts
//! outside the box starts above the surface, whatever `g` is where it
//! enters (a box may touch the surface). The stretch is then swept from its
//! near end a piece at a time:
//!
//! - where the enclosure of `g` has no point strictly across there is no
//!   crossing;
//! - where that of `g'` excludes 0, `g` is strictly monotone and the piece
//!   holds at most one crossing, there if `g` at its far end is across: the
//!   bracket is narrowed by safeguarded Newton ([`search::bracket`]), with
//!   the least `|g'|` as its certificate;
//! - otherwise the piece is halved, down to [`H_MIN`]; a piece that short
//!   whose far end is across is bracketed, and one whose far end is not is
//!   passed over, which can miss only a pair of crossings closer together
//!   than that.
//!
//! The enclosure of `g` is first the plain one over the box around the
//! piece. Where neither it nor that of `g'` decides, and the spread of the
//! centred form (the bound on `|g'|` times the piece's length) is narrower
//! than the plain enclosure, the centred form ([`Tape::centre`]) is tried:
//! near a tangency it narrows like the square of the piece's length rather
//! than like its length, so a ray grazing a thin feature is cleared in a
//! few pieces rather than thousands. A long formula with `min` or `max` is
//! shortened for the rest of a piece, and its halves, by the choices its
//! enclosures decide there ([`Tape::simplify`]), which gives the same
//! values in the piece, bit for bit.
//!
//! A particle within [`search::ON_BAND`] of the surface (to first order,
//! `|g| / |g'|`, or flagged as having just crossed it) and not heading for
//! it skips the stretch over which `g` is certified monotone, so the
//! crossing at its feet is not returned (see [`search::leaving`]).

use super::interval::Interval;
use super::tape::{ray_point, Choice, Dual, SegmentSlots, Tape};
use super::WORLD;
use crate::surface::search::{
    self, bracket, leaving, midpoint, minimal, tolerance, Budget, Line, Outcome, Trace, H_MIN,
    MIN_ROOT,
};
use std::cell::RefCell;

/// The longest stretch a particle leaving the surface skips, and the
/// factor shorter ones tried shrink by.
const SKIP: f64 = 1e-6;
const SKIP_SHRINK: f64 = 0.125;
/// Tapes at least this long with a `min` or `max` are shortened per piece.
const SHORTEN: usize = 16;
/// The formula's own tape, as a piece's tape.
const ROOT: usize = usize::MAX;

/// The side of the surface `v` puts a point on: 1 above, -1 below, 0 for
/// neither (an exact zero, or NaN).
fn side(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Whether an enclosure of `g` along a piece holds no point strictly
/// across from side `side` (an exact zero is no crossing).
fn clear(v: Interval, side: f64) -> bool {
    if side > 0.0 {
        v.lo >= 0.0
    } else {
        v.hi <= 0.0
    }
}

/// Whether every point of an enclosure of `g` is strictly across from side
/// `side`.
fn across(v: Interval, side: f64) -> bool {
    if side > 0.0 {
        v.hi < 0.0
    } else {
        v.lo > 0.0
    }
}

/// `t` moved in direction `step`'s sign, by steps doubling from `step`,
/// until `outside(t)`.
fn outward(mut t: f64, mut step: f64, outside: impl Fn(f64) -> bool) -> f64 {
    while !outside(t) {
        t += step;
        step *= 2.0;
    }
    t
}

/// The stretch `[t0, t1]` of the ray `o + t u` outside which every point,
/// as [`ray_point`] computes it, is outside the closed box `[lo, hi]`, or
/// `None` if every point is. Each computed coordinate is monotone in `t`,
/// so moving an end out until its point is outside the box on the axis
/// that sets it puts every point beyond outside too.
fn clip(o: [f64; 3], u: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    let (mut enters, mut leaves) = (None, None);
    for k in 0..3 {
        if u[k] == 0.0 {
            if !(lo[k] <= o[k] && o[k] <= hi[k]) {
                return None;
            }
            continue;
        }
        let (near, far) = if u[k] > 0.0 {
            (lo[k], hi[k])
        } else {
            (hi[k], lo[k])
        };
        let (a, b) = ((near - o[k]) / u[k], (far - o[k]) / u[k]);
        if a > t0 {
            (t0, enters) = (a, Some(k));
        }
        if b < t1 {
            (t1, leaves) = (b, Some(k));
        }
    }
    if t0 == f64::INFINITY || t1 == f64::NEG_INFINITY {
        return None;
    }
    let at = |k: usize, t: f64| o[k] + t * u[k];
    // About an ulp of axis `k`'s coordinates, as a ray distance.
    let ulp = |k: usize, t: f64| {
        let scale = o[k].abs().max(lo[k].abs()).max(hi[k].abs()) / u[k].abs();
        (scale.max(t.abs()) * f64::EPSILON).max(f64::MIN_POSITIVE)
    };
    if let Some(k) = enters {
        let before = |t| {
            if u[k] > 0.0 {
                at(k, t) < lo[k]
            } else {
                at(k, t) > hi[k]
            }
        };
        t0 = outward(t0, -ulp(k, t0), before);
    }
    if let Some(k) = leaves {
        let after = |t| {
            if u[k] > 0.0 {
                at(k, t) > hi[k]
            } else {
                at(k, t) < lo[k]
            }
        };
        t1 = outward(t1, ulp(k, t1), after);
    }
    (t0 <= t1).then_some((t0, t1))
}

/// A point of the ray with `g` and its slope there, evaluated exactly.
#[derive(Clone, Copy)]
struct Sample {
    t: f64,
    g: f64,
    slope: f64,
}

impl search::Point for Sample {
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

/// A stretch `[a, b]` of the ray left to search, and the tape it uses.
#[derive(Clone, Copy)]
struct Piece {
    a: f64,
    b: f64,
    tape: usize,
}

/// Buffers a query reuses: kept per thread, so a query allocates nothing
/// once they have grown.
#[derive(Default)]
struct Scratch {
    duals: Vec<Dual>,
    slots: SegmentSlots,
    pieces: Vec<Piece>,
    /// Shortened tapes, those past the ones in use spare.
    tapes: Vec<Tape>,
    work: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// `ROOT` or a shortened tape.
fn pick<'a>(root: &'a Tape, tapes: &'a [Tape], i: usize) -> &'a Tape {
    if i == ROOT {
        root
    } else {
        &tapes[i]
    }
}

struct Ray<'a> {
    root: &'a Tape,
    origin: [f64; 3],
    direction: [f64; 3],
    budget: Budget,
    s: Scratch,
    /// Shortened tapes in use. A piece's tape is never older than those of
    /// the pieces still to search, which come after it, so once a piece is
    /// taken the tapes made after its own are free.
    made: usize,
    /// The side the ray is on before the stretch left to search: 1 or -1.
    side: f64,
    /// The tape and certificate of the bracket being narrowed, and the
    /// nearest point of it seen strictly across.
    tape: usize,
    floor: f64,
    across: Sample,
}

impl Ray<'_> {
    /// `g` and its slope at `t`, on tape `tape` (one evaluation).
    fn at(&mut self, tape: usize, t: f64) -> Sample {
        self.budget.spend();
        let tape = pick(self.root, &self.s.tapes, tape);
        let d = tape.eval_dual(
            ray_point(self.origin, self.direction, t),
            self.direction,
            &mut self.s.duals,
        );
        Sample {
            t,
            g: d.v,
            slope: d.d,
        }
    }

    /// The longest stretch `[0, w]` tried over which `g` is certified
    /// monotone, or 0 if none is.
    fn skip(&mut self, end: f64) -> f64 {
        let mut w = SKIP.min(end);
        while w >= H_MIN && !self.budget.spent() {
            self.budget.spend();
            let e = self.root.eval_stretch(
                self.origin,
                self.direction,
                Interval::new(0.0, w),
                &mut self.s.slots,
            );
            if !e.d.has_zero() {
                return w;
            }
            w *= SKIP_SHRINK;
        }
        0.0
    }

    /// The first point past `from` (where `g` has no side) that has one,
    /// stepping by lengths that double; `Ok(None)` if none is before `end`,
    /// `Err` if the budget ran out first.
    fn first_signed(&mut self, from: f64, end: f64) -> Result<Option<Sample>, f64> {
        let (mut t, mut step) = (from, tolerance(from));
        loop {
            t += step;
            step *= 2.0;
            if t > end {
                return Ok(None);
            }
            if self.budget.spent() {
                return Err(t);
            }
            let s = self.at(ROOT, t);
            if side(s.g) != 0.0 {
                return Ok(Some(s));
            }
        }
    }

    /// The crossing in `(near.t, far.t]`, `far` strictly across and `near`
    /// not, as a point strictly across. Where `near` is strictly on the
    /// ray's side [`bracket`] narrows it; where its answer is not a point
    /// it saw across (a probe where `g` was exactly 0, or the estimate from
    /// the near end that its certificate allows), or `near` has no side,
    /// the first point strictly across is looked for from there.
    fn refine(&mut self, near: Sample, far: Sample) -> Result<f64, f64> {
        self.across = far;
        if side(near.g) != self.side {
            return self.first_across(near.t);
        }
        let t = bracket(self, near, far)?;
        if t == self.across.t {
            return Ok(t);
        }
        let s = self.at(self.tape, t);
        if side(s.g) == -self.side {
            Ok(t)
        } else {
            self.first_across(t)
        }
    }

    /// The first point past `from` (where `g` is not strictly across) at
    /// which it is, no farther than `self.across`: steps doubling from the
    /// tolerance find one, and bisection narrows the last step to the
    /// tolerance. `Err` if the budget ran out first.
    fn first_across(&mut self, from: f64) -> Result<f64, f64> {
        let mut a = from;
        let mut step = tolerance(from);
        let mut far = loop {
            let t = a + step;
            if t >= self.across.t {
                break self.across;
            }
            if self.budget.spent() {
                return Err(a);
            }
            let s = self.at(self.tape, t);
            if side(s.g) == -self.side {
                break s;
            }
            a = t;
            step *= 2.0;
        };
        loop {
            let m = midpoint(a, far.t);
            if far.t - a <= tolerance(far.t) || !(a < m && m < far.t) {
                return Ok(far.t);
            }
            if self.budget.spent() {
                return Err(a);
            }
            let s = self.at(self.tape, m);
            if side(s.g) == -self.side {
                far = s;
            } else {
                a = m;
            }
        }
    }

    /// The tape for the rest of a piece, and its halves, whose enclosures
    /// on tape `from` left their choices in the slots: a shortened one if
    /// they decide any `min` or `max` of a long tape.
    fn shorten(&mut self, from: usize) -> usize {
        let tape = pick(self.root, &self.s.tapes, from);
        if tape.len() < SHORTEN || self.s.slots.choices().iter().all(|&c| c == Choice::Both) {
            return from;
        }
        if self.made == self.s.tapes.len() {
            self.s.tapes.push(Tape::default());
        }
        let mut out = std::mem::take(&mut self.s.tapes[self.made]);
        let tape = pick(self.root, &self.s.tapes, from);
        tape.simplify(self.s.slots.choices(), &mut self.s.work, &mut out);
        let shorter = out.len() < tape.len();
        self.s.tapes[self.made] = out;
        if shorter {
            self.made += 1;
            self.made - 1
        } else {
            from
        }
    }

    /// Search `[t0, end]` (from the floor). `entering`: the ray reaches the
    /// box at `t0 > 0`, so it is above the surface before then.
    fn run(&mut self, t0: f64, end: f64, entering: bool, on_surface: Option<f64>) -> Outcome {
        let mut start = t0.max(MIN_ROOT);
        // The side before the start, where known: above before the box,
        // or the particle's own side.
        let mut before = if entering { 1.0 } else { 0.0 };
        if !entering && t0 <= 0.0 {
            let here = self.at(ROOT, 0.0);
            let distance = if on_surface.is_some() || here.g == 0.0 {
                0.0
            } else {
                (here.g / here.slope).abs()
            };
            if leaving(distance, here.g, here.slope) {
                start = start.max(self.skip(end));
            }
            before = side(here.g);
        }
        if start >= end {
            return Outcome::Miss;
        }
        let mut first = self.at(ROOT, start);
        self.side = side(first.g);
        if entering && self.side == -1.0 && start > MIN_ROOT {
            // Below already where it enters, with everything before above.
            return Outcome::Root(start);
        }
        if self.side == 0.0 {
            if before != 0.0 {
                self.side = before;
            } else {
                match self.first_signed(start, end) {
                    Ok(Some(s)) => {
                        first = s;
                        self.side = side(s.g);
                    }
                    Ok(None) => return Outcome::Miss,
                    Err(t) => return Outcome::Exhausted(t),
                }
            }
        }
        self.sweep(first, end)
    }

    /// Sweep `[first.t, end]` from its near end, on the side `self.side`.
    fn sweep(&mut self, first: Sample, end: f64) -> Outcome {
        self.s.pieces.clear();
        self.s.pieces.push(Piece {
            a: first.t,
            b: end,
            tape: ROOT,
        });
        // The latest sample at the end of a piece: the near end of the
        // next one.
        let mut last = first;
        while let Some(piece) = self.s.pieces.pop() {
            if self.budget.spent() {
                return Outcome::Exhausted(piece.a);
            }
            self.made = if piece.tape == ROOT {
                0
            } else {
                piece.tape + 1
            };
            let Some(found) = self.piece(piece, &mut last) else {
                continue;
            };
            match search::settle(found, |_| piece.b) {
                Ok(outcome) => return outcome,
                // Across at or before MIN_ROOT, which does not count: search
                // on past the piece from the other side.
                Err(_) => self.side = -self.side,
            }
        }
        Outcome::Miss
    }

    /// Search one piece: `None` if it holds no crossing or has been split
    /// (its halves pushed to be searched next), else the crossing found
    /// (`Err` if the budget ran out). `last` is the latest sample at the
    /// far end of a piece.
    fn piece(&mut self, piece: Piece, last: &mut Sample) -> Option<Result<f64, f64>> {
        let t = Interval::new(piece.a, piece.b);
        let (o, u) = (self.origin, self.direction);
        let tape = pick(self.root, &self.s.tapes, piece.tape);
        self.budget.spend();
        let e = tape.eval_stretch(o, u, t, &mut self.s.slots);
        if clear(e.v, self.side) {
            return None;
        }
        if across(e.v, self.side) {
            // The near end never is, as the pieces before end on this side;
            // a crossing there all the same.
            return Some(Ok(piece.a));
        }
        let monotone = !e.d.has_zero();
        // The centred form, where its spread (the slope times the length)
        // is narrower than the plain enclosure: near a tangency, or on a
        // short piece.
        if !monotone && e.d.magnitude() * (piece.b - piece.a) < e.v.hi - e.v.lo {
            self.budget.spend();
            let c = tape.centre(o, u, t, &mut self.s.slots);
            if clear(c, self.side) {
                return None;
            }
            if across(c, self.side) {
                return Some(Ok(piece.a));
            }
        }
        let tape = self.shorten(piece.tape);
        if !(monotone || minimal(piece.a, piece.b)) {
            let m = midpoint(piece.a, piece.b);
            self.s.pieces.push(Piece {
                a: m,
                b: piece.b,
                tape,
            });
            self.s.pieces.push(Piece {
                a: piece.a,
                b: m,
                tape,
            });
            return None;
        }
        let far = self.at(tape, piece.b);
        if side(far.g) != -self.side {
            *last = far;
            return None;
        }
        self.tape = tape;
        self.floor = if monotone { e.d.mignitude() } else { 0.0 };
        let near = if last.t == piece.a {
            *last
        } else {
            self.at(tape, piece.a)
        };
        *last = far;
        Some(self.refine(near, far))
    }
}

impl Line for Ray<'_> {
    type Point = Sample;

    fn exact(&mut self, t: f64) -> Sample {
        let s = self.at(self.tape, t);
        if side(s.g) == -self.side && s.t < self.across.t {
            self.across = s;
        }
        s
    }

    fn spent(&self) -> bool {
        self.budget.spent()
    }

    /// The least `|g'|` over the piece being bracketed, if it is monotone.
    fn floor(&self, _: &Sample, _: &Sample) -> f64 {
        self.floor
    }

    /// Only a piece too short to split is bracketed without a certificate,
    /// and a pair of crossings in it is closer together than [`H_MIN`]: one
    /// the search may miss.
    fn scan(&mut self, _: Sample, _: Sample) -> Result<Option<f64>, f64> {
        Ok(None)
    }

    /// From the end whose Newton step is the shorter, the nearer to the
    /// crossing to first order. A bracket here is a whole monotone piece,
    /// so the end evaluated last can be a bisection's, far from the
    /// crossing, with a step beyond the other end.
    fn newton_from_b(&self, a: &Sample, b: &Sample, _: bool) -> bool {
        (b.g / b.slope).abs() <= (a.g / a.slope).abs()
    }
}

/// The first crossing of the surface `tape` (with the bounding box
/// `bounding_box`, if it has one) beyond [`MIN_ROOT`] along the ray `origin
/// + t direction` (`direction` of unit length), if before `t_max`, and what
/// the query cost. `on_surface` says the particle has just crossed the
/// surface; if the evaluation budget runs out, a transmission surface
/// returns the root-free point reached and a vacuum one `None` (see
/// [`Outcome::trace`]).
pub(super) fn trace(
    tape: &Tape,
    bounding_box: Option<([f64; 3], [f64; 3])>,
    origin: [f64; 3],
    direction: [f64; 3],
    on_surface: Option<f64>,
    vacuum: bool,
    t_max: f64,
) -> Trace {
    let none = Outcome::Miss.trace(vacuum, 0);
    let (lo, hi) = bounding_box.unwrap_or(([-WORLD; 3], [WORLD; 3]));
    let Some((t0, t1)) = clip(origin, direction, lo, hi) else {
        return none;
    };
    let end = t1.min(t_max);
    if end.is_nan() || end <= MIN_ROOT {
        return none;
    }
    let entering = bounding_box.is_some() && t0 > 0.0;
    SCRATCH.with(|scratch| {
        let unset = Sample {
            t: f64::INFINITY,
            g: f64::NAN,
            slope: f64::NAN,
        };
        let mut ray = Ray {
            root: tape,
            origin,
            direction,
            budget: Budget::new(vacuum),
            s: scratch.take(),
            made: 0,
            side: 0.0,
            tape: ROOT,
            floor: 0.0,
            across: unset,
        };
        let outcome = ray.run(t0, end, entering, on_surface);
        let evaluations = ray.budget.used();
        scratch.replace(ray.s);
        outcome.trace(vacuum, evaluations)
    })
}
