//! The compiled formula: a linear tape of instructions, and the evaluators
//! that run it.
//!
//! - [`Tape::eval`]: the f64 value at a point, the one the cell test reads.
//! - [`Tape::eval_interval`]: an [`Interval`] over a box, enclosing the
//!   point value at every point of it (see [`super::interval`]);
//!   [`Tape::eval_traced`] also records which operand each `min` / `max`
//!   takes throughout the box, and [`Tape::simplify`] drops the operands
//!   never taken, giving a shorter tape with the same value, bit for bit,
//!   everywhere in that box.
//! - [`Tape::eval_dual`]: the value and its derivative along a direction
//!   (forward mode), for Newton's method.
//! - [`Tape::eval_interval_dual`]: an interval value and an enclosure of the
//!   derivative along a direction over a box: where it excludes zero the
//!   formula is strictly monotone along every line in that direction
//!   through the box. At a kink (`abs`, `min`, `max`) it encloses the
//!   derivatives of both sides (Clarke's generalised gradient), so a kink
//!   is shown monotone only where both sides slope the same way.
//! - [`Tape::eval_segment`]: the same along a stretch of a ray, with the
//!   value narrowed by the centred (mean-value) form. A centred form built
//!   from the derivative alone encloses the exact values, not the computed
//!   ones whose signs the cell test reads; this one adds a bound on the
//!   rounding error, carried through the tape alongside the derivative.
//!
//! The point arithmetic of each instruction is in one place, [`point`],
//! which the constant folding uses too. Every transcendental function is
//! `libm`'s, the same on every platform; the rest is IEEE arithmetic,
//! which Rust never contracts into fused multiply-adds.

use super::interval::{Interval, LIBM_ULPS};

/// One tape instruction; operands are earlier slots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    X,
    Y,
    Z,
    Const(f64),
    Neg(u32),
    /// `a * a`.
    Square(u32),
    Sqrt(u32),
    Abs(u32),
    Exp(u32),
    Log(u32),
    Sin(u32),
    Cos(u32),
    /// `a^p` for a constant non-integer `p` (the base is never negative).
    Pow(u32, f64),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    /// The smaller operand: the first on a tie, NaN if either is NaN.
    Min(u32, u32),
    /// The larger operand: the first on a tie, NaN if either is NaN.
    Max(u32, u32),
}

impl Op {
    /// The operands (`u32::MAX` for none).
    pub fn args(&self) -> [u32; 2] {
        match *self {
            Op::X | Op::Y | Op::Z | Op::Const(_) => [u32::MAX; 2],
            Op::Neg(a)
            | Op::Square(a)
            | Op::Sqrt(a)
            | Op::Abs(a)
            | Op::Exp(a)
            | Op::Log(a)
            | Op::Sin(a)
            | Op::Cos(a)
            | Op::Pow(a, _) => [a, u32::MAX],
            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
            | Op::Div(a, b)
            | Op::Min(a, b)
            | Op::Max(a, b) => [a, b],
        }
    }

    /// The same instruction on renumbered operands.
    pub fn map(&self, f: impl Fn(u32) -> u32) -> Op {
        match *self {
            Op::X | Op::Y | Op::Z | Op::Const(_) => *self,
            Op::Neg(a) => Op::Neg(f(a)),
            Op::Square(a) => Op::Square(f(a)),
            Op::Sqrt(a) => Op::Sqrt(f(a)),
            Op::Abs(a) => Op::Abs(f(a)),
            Op::Exp(a) => Op::Exp(f(a)),
            Op::Log(a) => Op::Log(f(a)),
            Op::Sin(a) => Op::Sin(f(a)),
            Op::Cos(a) => Op::Cos(f(a)),
            Op::Pow(a, p) => Op::Pow(f(a), p),
            Op::Add(a, b) => Op::Add(f(a), f(b)),
            Op::Sub(a, b) => Op::Sub(f(a), f(b)),
            Op::Mul(a, b) => Op::Mul(f(a), f(b)),
            Op::Div(a, b) => Op::Div(f(a), f(b)),
            Op::Min(a, b) => Op::Min(f(a), f(b)),
            Op::Max(a, b) => Op::Max(f(a), f(b)),
        }
    }
}

/// Which operand a `min` / `max` takes throughout a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Either, or not a `min` / `max`.
    Both,
    Left,
    Right,
}

/// A value and its derivative along a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub d: f64,
}

/// Enclosures of a value and of its derivative along a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalDual {
    pub v: Interval,
    pub d: Interval,
}

/// The point at ray distance `t`, computed as the ray search does.
pub fn ray_point(origin: [f64; 3], direction: [f64; 3], t: f64) -> [f64; 3] {
    [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ]
}

/// A box enclosing [`ray_point`] for every `t` in `t`.
pub fn ray_box(origin: [f64; 3], direction: [f64; 3], t: Interval) -> [Interval; 3] {
    std::array::from_fn(|k| Interval::point(origin[k]) + t * Interval::point(direction[k]))
}

/// `min` of the tapes: the first operand on a tie and NaN if either is NaN
/// (`f64::min` drops a NaN, which no interval could enclose).
fn fmin(a: f64, b: f64) -> f64 {
    if a <= b {
        a
    } else if b < a {
        b
    } else {
        f64::NAN
    }
}

/// `max` counterpart of [`fmin`].
fn fmax(a: f64, b: f64) -> f64 {
    if a >= b {
        a
    } else if b > a {
        b
    } else {
        f64::NAN
    }
}

/// `factor * d` by the chain rule, 0 where `d` is: a slot that does not
/// change along the direction has no derivative along it, even where the
/// instruction's own is infinite (`sqrt` at 0, say).
fn chain(d: f64, factor: impl FnOnce() -> f64) -> f64 {
    if d == 0.0 {
        0.0
    } else {
        factor() * d
    }
}

/// [`chain`] for intervals.
fn chain_interval(d: Interval, factor: impl FnOnce() -> Interval) -> Interval {
    if d.lo == 0.0 && d.hi == 0.0 {
        Interval::point(0.0)
    } else {
        factor() * d
    }
}

/// The f64 value of `op` at `p`, its operands in `s`.
#[inline(always)]
pub(super) fn point(op: Op, s: &[f64], p: [f64; 3]) -> f64 {
    let at = |i: u32| s[i as usize];
    match op {
        Op::X => p[0],
        Op::Y => p[1],
        Op::Z => p[2],
        Op::Const(c) => c,
        Op::Neg(a) => -at(a),
        Op::Square(a) => at(a) * at(a),
        Op::Sqrt(a) => at(a).sqrt(),
        Op::Abs(a) => at(a).abs(),
        Op::Exp(a) => libm::exp(at(a)),
        Op::Log(a) => libm::log(at(a)),
        Op::Sin(a) => libm::sin(at(a)),
        Op::Cos(a) => libm::cos(at(a)),
        Op::Pow(a, e) => libm::pow(at(a), e),
        Op::Add(a, b) => at(a) + at(b),
        Op::Sub(a, b) => at(a) - at(b),
        Op::Mul(a, b) => at(a) * at(b),
        Op::Div(a, b) => at(a) / at(b),
        Op::Min(a, b) => fmin(at(a), at(b)),
        Op::Max(a, b) => fmax(at(a), at(b)),
    }
}

/// The value and derivative along `dir` of `op` at `p`.
fn dual(op: Op, s: &[Dual], p: [f64; 3], dir: [f64; 3]) -> Dual {
    let at = |i: u32| s[i as usize];
    let dual = |v: f64, d: f64| Dual { v, d };
    match op {
        Op::X => dual(p[0], dir[0]),
        Op::Y => dual(p[1], dir[1]),
        Op::Z => dual(p[2], dir[2]),
        Op::Const(c) => dual(c, 0.0),
        Op::Neg(a) => dual(-at(a).v, -at(a).d),
        Op::Square(a) => {
            let x = at(a);
            dual(x.v * x.v, 2.0 * x.v * x.d)
        }
        Op::Sqrt(a) => {
            let x = at(a);
            let r = x.v.sqrt();
            dual(r, chain(x.d, || 0.5 / r))
        }
        Op::Abs(a) => {
            let x = at(a);
            dual(x.v.abs(), if x.v < 0.0 { -x.d } else { x.d })
        }
        Op::Exp(a) => {
            let x = at(a);
            let e = libm::exp(x.v);
            dual(e, chain(x.d, || e))
        }
        Op::Log(a) => {
            let x = at(a);
            dual(libm::log(x.v), chain(x.d, || 1.0 / x.v))
        }
        Op::Sin(a) => {
            let x = at(a);
            dual(libm::sin(x.v), chain(x.d, || libm::cos(x.v)))
        }
        Op::Cos(a) => {
            let x = at(a);
            dual(libm::cos(x.v), chain(x.d, || -libm::sin(x.v)))
        }
        Op::Pow(a, e) => {
            let x = at(a);
            dual(
                libm::pow(x.v, e),
                chain(x.d, || e * libm::pow(x.v.abs(), e - 1.0)),
            )
        }
        Op::Add(a, b) => dual(at(a).v + at(b).v, at(a).d + at(b).d),
        Op::Sub(a, b) => dual(at(a).v - at(b).v, at(a).d - at(b).d),
        Op::Mul(a, b) => {
            let (x, y) = (at(a), at(b));
            dual(x.v * y.v, x.d * y.v + x.v * y.d)
        }
        Op::Div(a, b) => {
            let (x, y) = (at(a), at(b));
            let q = x.v / y.v;
            dual(q, (x.d - q * y.d) / y.v)
        }
        Op::Min(a, b) => {
            let (x, y) = (at(a), at(b));
            if x.v <= y.v {
                x
            } else if y.v < x.v {
                y
            } else {
                dual(f64::NAN, f64::NAN)
            }
        }
        Op::Max(a, b) => {
            let (x, y) = (at(a), at(b));
            if x.v >= y.v {
                x
            } else if y.v > x.v {
                y
            } else {
                dual(f64::NAN, f64::NAN)
            }
        }
    }
}

/// An enclosure of `op` over the box `b`.
#[inline(always)]
pub(super) fn interval(op: Op, s: &[Interval], b: &[Interval; 3]) -> Interval {
    let at = |i: u32| s[i as usize];
    match op {
        Op::X => b[0],
        Op::Y => b[1],
        Op::Z => b[2],
        Op::Const(c) => Interval::point(c),
        Op::Neg(a) => -at(a),
        Op::Square(a) => at(a).square(),
        Op::Sqrt(a) => at(a).sqrt(),
        Op::Abs(a) => at(a).abs(),
        Op::Exp(a) => at(a).exp(),
        Op::Log(a) => at(a).log(),
        Op::Sin(a) => at(a).sin(),
        Op::Cos(a) => at(a).cos(),
        Op::Pow(a, e) => at(a).pow(e),
        Op::Add(a, c) => at(a) + at(c),
        Op::Sub(a, c) => at(a) - at(c),
        Op::Mul(a, c) => at(a) * at(c),
        Op::Div(a, c) => at(a) / at(c),
        Op::Min(a, c) => at(a).min(at(c)),
        Op::Max(a, c) => at(a).max(at(c)),
    }
}

/// Which operand the `min` / `max` `op` takes throughout the box, from the
/// enclosures of its operands: decided only where one is strictly beyond
/// the other, so a tie (where `min` takes its first operand, and `+0` and
/// `-0` are equal but not the same) is never decided for the second.
fn choice(op: Op, at: impl Fn(u32) -> Interval) -> Choice {
    match op {
        Op::Min(a, b) if at(a).hi < at(b).lo => Choice::Left,
        Op::Min(a, b) if at(b).hi < at(a).lo => Choice::Right,
        Op::Max(a, b) if at(a).lo > at(b).hi => Choice::Left,
        Op::Max(a, b) if at(b).lo > at(a).hi => Choice::Right,
        _ => Choice::Both,
    }
}

/// Enclosures of `op` and its derivative along `dir` over the box `b`.
fn interval_dual(op: Op, s: &[IntervalDual], b: &[Interval; 3], dir: [f64; 3]) -> IntervalDual {
    let at = |i: u32| s[i as usize];
    let dual = |v: Interval, d: Interval| IntervalDual { v, d };
    let constant = |c: f64| Interval::point(c);
    match op {
        Op::X => dual(b[0], constant(dir[0])),
        Op::Y => dual(b[1], constant(dir[1])),
        Op::Z => dual(b[2], constant(dir[2])),
        Op::Const(c) => dual(Interval::point(c), constant(0.0)),
        Op::Neg(a) => dual(-at(a).v, -at(a).d),
        Op::Square(a) => {
            let x = at(a);
            dual(x.v.square(), constant(2.0) * x.v * x.d)
        }
        Op::Sqrt(a) => {
            let x = at(a);
            let r = x.v.sqrt();
            let d = chain_interval(x.d, || {
                if x.v.lo > 0.0 {
                    constant(0.5) / r
                } else {
                    Interval::ENTIRE
                }
            });
            dual(r, d)
        }
        Op::Abs(a) => {
            let x = at(a);
            let d = if x.v.lo >= 0.0 {
                x.d
            } else if x.v.hi < 0.0 {
                -x.d
            } else {
                x.d.hull(-x.d)
            };
            dual(x.v.abs(), d)
        }
        Op::Exp(a) => {
            let x = at(a);
            let e = x.v.exp();
            dual(e, chain_interval(x.d, || e))
        }
        Op::Log(a) => {
            let x = at(a);
            dual(x.v.log(), chain_interval(x.d, || constant(1.0) / x.v))
        }
        Op::Sin(a) => {
            let x = at(a);
            dual(x.v.sin(), chain_interval(x.d, || x.v.cos()))
        }
        Op::Cos(a) => {
            let x = at(a);
            dual(x.v.cos(), chain_interval(x.d, || -x.v.sin()))
        }
        Op::Pow(a, e) => {
            let x = at(a);
            let d = chain_interval(x.d, || constant(e) * x.v.pow(e - 1.0));
            dual(x.v.pow(e), d)
        }
        Op::Add(a, c) => dual(at(a).v + at(c).v, at(a).d + at(c).d),
        Op::Sub(a, c) => dual(at(a).v - at(c).v, at(a).d - at(c).d),
        Op::Mul(a, c) => {
            let (x, y) = (at(a), at(c));
            dual(x.v * y.v, x.d * y.v + x.v * y.d)
        }
        Op::Div(a, c) => {
            let (x, y) = (at(a), at(c));
            let q = x.v / y.v;
            dual(q, (x.d - q * y.d) / y.v)
        }
        Op::Min(a, c) => {
            let (x, y) = (at(a), at(c));
            if x.v.hi < y.v.lo {
                x
            } else if y.v.hi < x.v.lo {
                y
            } else {
                dual(x.v.min(y.v), x.d.hull(y.d))
            }
        }
        Op::Max(a, c) => {
            let (x, y) = (at(a), at(c));
            if x.v.lo > y.v.hi {
                x
            } else if y.v.lo > x.v.hi {
                y
            } else {
                dual(x.v.max(y.v), x.d.hull(y.d))
            }
        }
    }
}

/// `[-m, m]`.
fn spread(m: f64) -> Interval {
    Interval::new(-m, m)
}

/// `ulps` ulps of the largest member of `v`: a bound on the rounding error
/// of an operation whose result lies in `v`.
fn rounding(v: Interval, ulps: f64) -> Interval {
    let m = v.magnitude();
    if m.is_finite() {
        spread((m.next_up() - m) * ulps)
    } else {
        Interval::ENTIRE
    }
}

/// An enclosure of the error of slot `op` along a stretch of a ray: its
/// computed value at the computed point less its exact value at the exact
/// point. `s` holds the slots' enclosures over the stretch's box `b` (of
/// the computed and the exact values alike), `e` the operands' errors,
/// `own` the slot's enclosure, and `stretch` the ray distances times the
/// direction. The error is the operands' errors carried through the
/// operation (by its derivative over the operands' enclosures, or by a
/// Lipschitz bound) plus the operation's own rounding: within half an ulp
/// for IEEE arithmetic, taken as one, and one ulp for `libm`, taken as
/// [`LIBM_ULPS`].
fn error(
    op: Op,
    s: &[IntervalDual],
    e: &[Interval],
    own: Interval,
    b: &[Interval; 3],
    stretch: &[Interval; 3],
) -> Interval {
    let at = |i: u32| s[i as usize].v;
    let err = |i: u32| e[i as usize];
    let ieee = rounding(own, 1.0);
    let libm = rounding(own, f64::from(LIBM_ULPS));
    // The larger operand error, for the operations no steeper than 1.
    let lipschitz = |a: u32, c: u32| spread(err(a).magnitude().max(err(c).magnitude()));
    match op {
        // The point is `o + t u`, rounded twice.
        Op::X => rounding(stretch[0], 1.0) + rounding(b[0], 1.0),
        Op::Y => rounding(stretch[1], 1.0) + rounding(b[1], 1.0),
        Op::Z => rounding(stretch[2], 1.0) + rounding(b[2], 1.0),
        Op::Const(_) => Interval::point(0.0),
        Op::Neg(a) => -err(a),
        Op::Square(a) => err(a) * (at(a) + at(a)) + ieee,
        Op::Sqrt(a) => {
            // |sqrt(c) - sqrt(e)| = |c - e| / (sqrt(c) + sqrt(e)), and is at
            // most sqrt(|c - e|).
            let carried = if at(a).lo > 0.0 {
                let r = at(a).sqrt();
                err(a) / (r + r)
            } else {
                spread(err(a).magnitude().sqrt().next_up())
            };
            carried + ieee
        }
        Op::Abs(a) => lipschitz(a, a),
        Op::Exp(a) => at(a).exp() * err(a) + libm,
        Op::Log(a) => err(a) / at(a) + libm,
        Op::Sin(a) => at(a).cos() * err(a) + libm,
        Op::Cos(a) => -(at(a).sin() * err(a)) + libm,
        Op::Pow(a, p) => Interval::point(p) * at(a).pow(p - 1.0) * err(a) + libm,
        Op::Add(a, c) => err(a) + err(c) + ieee,
        Op::Sub(a, c) => err(a) - err(c) + ieee,
        Op::Mul(a, c) => at(a) * err(c) + at(c) * err(a) + ieee,
        Op::Div(a, c) => (err(a) * at(c) - at(a) * err(c)) / (at(c) * at(c)) + ieee,
        // Exact, and taking one operand throughout where the enclosures
        // (of computed and exact values alike) decide it.
        Op::Min(a, c) if at(a).hi < at(c).lo => err(a),
        Op::Min(a, c) if at(c).hi < at(a).lo => err(c),
        Op::Max(a, c) if at(a).lo > at(c).hi => err(a),
        Op::Max(a, c) if at(c).lo > at(a).hi => err(c),
        Op::Min(a, c) | Op::Max(a, c) => lipschitz(a, c),
    }
}

/// Buffers for [`Tape::eval_segment`], and the choices it records.
#[derive(Default)]
pub struct SegmentSlots {
    duals: Vec<IntervalDual>,
    errors: Vec<Interval>,
    middle: Vec<Interval>,
    choices: Vec<Choice>,
}

impl SegmentSlots {
    /// Which operand each `min` / `max` takes throughout the box of the
    /// last stretch evaluated (as [`Tape::eval_traced`] over that box
    /// records), for [`Tape::simplify`].
    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }
}

/// A formula compiled to straight-line code: slot `i` holds the result of
/// instruction `i`, whose operands are earlier slots, and the value of the
/// formula is the last slot. Each evaluator takes a buffer for the slots,
/// which it overwrites, so a caller can reuse one across evaluations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tape {
    pub(super) ops: Vec<Op>,
}

impl Tape {
    /// The instructions.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of `min` / `max` instructions.
    pub fn choices(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Min(..) | Op::Max(..)))
            .count()
    }

    /// The value at `p`.
    pub fn eval(&self, p: [f64; 3], slots: &mut Vec<f64>) -> f64 {
        slots.clear();
        for &op in &self.ops {
            let v = point(op, slots, p);
            slots.push(v);
        }
        slots.last().copied().unwrap_or(f64::NAN)
    }

    /// An enclosure of the value over the box `b`.
    pub fn eval_interval(&self, b: [Interval; 3], slots: &mut Vec<Interval>) -> Interval {
        slots.clear();
        for &op in &self.ops {
            let v = interval(op, slots, &b);
            slots.push(v);
        }
        slots.last().copied().unwrap_or(Interval::ENTIRE)
    }

    /// [`Tape::eval_interval`], also recording in `choices` (one per
    /// instruction) which operand each `min` / `max` takes throughout the
    /// box, for [`Tape::simplify`].
    pub fn eval_traced(
        &self,
        b: [Interval; 3],
        slots: &mut Vec<Interval>,
        choices: &mut Vec<Choice>,
    ) -> Interval {
        slots.clear();
        choices.clear();
        for &op in &self.ops {
            choices.push(choice(op, |i| slots[i as usize]));
            let v = interval(op, slots, &b);
            slots.push(v);
        }
        slots.last().copied().unwrap_or(Interval::ENTIRE)
    }

    /// The value at `p` and its derivative along `dir`.
    pub fn eval_dual(&self, p: [f64; 3], dir: [f64; 3], slots: &mut Vec<Dual>) -> Dual {
        slots.clear();
        for &op in &self.ops {
            let v = dual(op, slots, p, dir);
            slots.push(v);
        }
        slots.last().copied().unwrap_or(Dual {
            v: f64::NAN,
            d: f64::NAN,
        })
    }

    /// Enclosures of the value and of its derivative along `dir` over the
    /// box `b`.
    pub fn eval_interval_dual(
        &self,
        b: [Interval; 3],
        dir: [f64; 3],
        slots: &mut Vec<IntervalDual>,
    ) -> IntervalDual {
        slots.clear();
        for &op in &self.ops {
            let v = interval_dual(op, slots, &b, dir);
            slots.push(v);
        }
        slots.last().copied().unwrap_or(IntervalDual {
            v: Interval::ENTIRE,
            d: Interval::ENTIRE,
        })
    }

    /// Enclosures of the value at every point [`ray_point`] computes along
    /// the stretch `t` of the ray `origin + t direction`, and of the
    /// derivative along `direction` there. The value is the plain interval
    /// over the stretch's box intersected with the centred form: the exact
    /// value at the middle, plus the derivative times the distance from
    /// it, plus a bound on the rounding error of the computed values. Near
    /// a tangency the plain interval narrows like the length of the
    /// stretch, the centred form like its square. `slots` also records the
    /// `min` / `max` choices over the stretch's box.
    ///
    /// This is [`Tape::eval_stretch`] followed by [`Tape::centre`], which a
    /// caller can run apart, paying for the centred form only where the
    /// plain enclosures leave something undecided.
    pub fn eval_segment(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        t: Interval,
        slots: &mut SegmentSlots,
    ) -> IntervalDual {
        let whole = self.eval_stretch(origin, direction, t, slots);
        IntervalDual {
            v: self.centre(origin, direction, t, slots),
            d: whole.d,
        }
    }

    /// The plain enclosures of [`Tape::eval_segment`]: the interval dual
    /// over the stretch's box, which `slots` keeps for [`Tape::centre`],
    /// and the `min` / `max` choices over that box.
    pub fn eval_stretch(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        t: Interval,
        slots: &mut SegmentSlots,
    ) -> IntervalDual {
        let b = ray_box(origin, direction, t);
        let whole = self.eval_interval_dual(b, direction, &mut slots.duals);
        slots.choices.clear();
        for &op in &self.ops {
            slots
                .choices
                .push(choice(op, |a| slots.duals[a as usize].v));
        }
        whole
    }

    /// The value enclosure of [`Tape::eval_segment`], once
    /// [`Tape::eval_stretch`] has run over the same stretch of the same
    /// ray: the plain one it left in `slots` narrowed by the centred form.
    pub fn centre(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        t: Interval,
        slots: &mut SegmentSlots,
    ) -> Interval {
        let whole = slots.duals.last().copied().unwrap_or(IntervalDual {
            v: Interval::ENTIRE,
            d: Interval::ENTIRE,
        });
        let b = ray_box(origin, direction, t);
        let stretch = direction.map(|u| t * Interval::point(u));
        slots.errors.clear();
        for (i, &op) in self.ops.iter().enumerate() {
            let own = slots.duals[i].v;
            let e = error(op, &slots.duals, &slots.errors, own, &b, &stretch);
            slots.errors.push(e);
        }
        let error = slots.errors.last().copied().unwrap_or(Interval::ENTIRE);
        let m = 0.5 * (t.lo + t.hi);
        let middle = self.eval_interval(
            ray_box(origin, direction, Interval::point(m)),
            &mut slots.middle,
        );
        let centred = middle + whole.d * (t - Interval::point(m)) + error;
        let (lo, hi) = (centred.lo.max(whole.v.lo), centred.hi.min(whole.v.hi));
        if lo <= hi {
            Interval::new(lo, hi)
        } else {
            whole.v
        }
    }

    /// This tape with every `min` / `max` that `choices` (from
    /// [`Tape::eval_traced`] over some box) decides replaced by the operand
    /// it takes, and the instructions no longer needed dropped, written to
    /// `out`. Throughout that box the result has the same value as this
    /// tape, bit for bit. `work` is a buffer.
    pub fn simplify(&self, choices: &[Choice], work: &mut Vec<u32>, out: &mut Tape) {
        let n = self.ops.len();
        out.ops.clear();
        if n == 0 || choices.len() != n {
            out.ops.extend_from_slice(&self.ops);
            return;
        }
        work.clear();
        work.resize(2 * n, 0);
        // `alias[i]`: the slot whose value slot `i` takes; `slot[i]`: first
        // whether slot `i` is needed, then 1 + its index in `out`.
        let (alias, slot) = work.split_at_mut(n);
        for i in 0..n {
            alias[i] = match (self.ops[i], choices[i]) {
                (Op::Min(a, _) | Op::Max(a, _), Choice::Left) => alias[a as usize],
                (Op::Min(_, b) | Op::Max(_, b), Choice::Right) => alias[b as usize],
                _ => i as u32,
            };
        }
        slot[alias[n - 1] as usize] = 1;
        for i in (0..n).rev() {
            if slot[i] == 0 || alias[i] != i as u32 {
                continue;
            }
            for a in self.ops[i].args() {
                if a != u32::MAX {
                    slot[alias[a as usize] as usize] = 1;
                }
            }
        }
        for i in 0..n {
            if slot[i] != 0 && alias[i] == i as u32 {
                let op = self.ops[i].map(|a| slot[alias[a as usize] as usize] - 1);
                slot[i] = out.ops.len() as u32 + 1;
                out.ops.push(op);
            }
        }
    }
}
