//! Implicit surfaces: the zero set of a formula `f(x, y, z)` (cm), with
//! `f < 0` below the surface.
//!
//! A formula string is parsed (`parse`), compiled to a [`Tape`] of f64
//! instructions (`compile`) and checked (`bounds`). The tape's point
//! value is the surface's `evaluate`, whose sign is all the cell test
//! reads; its interval, dual and interval-dual evaluators enclose that
//! very value and its derivative along a direction, which is what a
//! certified ray search needs.
//!
//! The formula language: numbers, `x`, `y`, `z`, `pi`, `+ - * /`, unary
//! minus, `^` with a constant exponent, and the functions `sqrt`, `abs`,
//! `exp`, `log`, `sin`, `cos`, `min`, `max` (two or more arguments) and
//! `smin(a, b, k)` / `smax(a, b, k)` (polynomial smooth minimum and
//! maximum, blending over a distance `k > 0`). Integer powers are products
//! (`x^3` is `x^2 * x`, `x^-2` is `1 / x^2`); other powers need a base that
//! cannot be negative (`abs(x)^2.5`). Discontinuous functions (`floor`,
//! `mod`, `atan2`, comparisons) are left out: a jump in sign is a crossing
//! with no surface there.
//!
//! Determinism: `sin`, `cos`, `exp`, `log` and `pow` come from the pure-Rust
//! `libm`, so they give the same bits on every platform (std's do not);
//! the rest is IEEE arithmetic, which Rust never fuses or reorders, run in
//! the fixed order of the tape. The tape depends on nothing but the
//! formula, and a model stores the canonical formula string, which parses
//! back to the same tape.
//!
//! Construction refuses, with a message saying where:
//! - a formula that does not parse, or has a non-finite constant;
//! - one that interval arithmetic cannot show to be defined (never NaN)
//!   and free of poles (never a division by zero, whose sign flips with no
//!   surface there) throughout its bounding box, or, with no box, within
//!   [`WORLD`] of the origin; an overflow to an infinity is allowed;
//! - one that is never negative there (nothing below the surface) or, with
//!   no box, never positive there.
//!
//! The bounding box is proved by interval arithmetic too (see `bounds`);
//! a periodic field, a plane or a twist gets none, and the cells using it
//! must bound it instead. All of this holds within [`WORLD`] of the origin
//! (about 10 km), not beyond: far enough out a high power or an
//! exponential overflows, and two overflows can make a NaN.

mod bounds;
mod compile;
mod interval;
mod parse;
mod tape;
#[cfg(test)]
mod tests;

pub use bounds::WORLD;
use bounds::{Cell, Extent, Fault, Problem};
pub use interval::{Interval, LIBM_ULPS};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
pub use tape::{ray_box, ray_point, Choice, Dual, IntervalDual, Op, SegmentSlots, Tape};

thread_local! {
    /// Slots for [`ImplicitFormula::evaluate`], one buffer per thread.
    static SLOTS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// A checked formula `f(x, y, z)` (cm) whose zero set is an implicit
/// surface, `f < 0` below it; see the module docs. It is stored (and
/// serialised) as its canonical formula string, and rebuilt from it.
pub struct ImplicitFormula {
    text: String,
    tape: Tape,
    bounding_box: Option<([f64; 3], [f64; 3])>,
}

impl std::fmt::Debug for ImplicitFormula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImplicitFormula")
            .field("formula", &self.text)
            .field("instructions", &self.tape.len())
            .finish_non_exhaustive()
    }
}

/// `v` to 7 significant digits, for messages.
fn short(v: f64) -> String {
    let rounded: f64 = format!("{v:.6e}").parse().unwrap_or(v);
    if rounded == 0.0 {
        "0".to_string()
    } else if rounded < 0.0 {
        format!("-{}", parse::number(-rounded))
    } else {
        parse::number(rounded)
    }
}

impl ImplicitFormula {
    /// Parse, compile and check `formula`, and prove its bounding box.
    pub fn new(formula: &str) -> Result<Self, String> {
        let ast = parse::parse(formula).map_err(|e| parse::locate(formula, e.pos, &e.message))?;
        let at = |node: u32| ast.nodes[node as usize].pos;
        let compiled =
            compile::compile(&ast).map_err(|e| parse::locate(formula, at(e.node), &e.message))?;
        let text = ast.print(ast.root());
        let tape = &compiled.tape;
        let checked = match bounds::extent(tape) {
            Extent::Inside(cell) => {
                bounds::validate(tape, cell, true).map(|()| Some((cell.lo, cell.hi)))
            }
            Extent::Unbounded => bounds::validate(tape, Cell::cube(WORLD), false).map(|()| None),
            Extent::Empty => Err(Problem::NeverBelow),
        };
        let bounding_box = checked.map_err(|problem| match problem {
            Problem::Fault { slot, fault, at: p } => {
                let node = compiled.origin[slot];
                let what = match (tape.ops[slot], fault) {
                    (Op::Sqrt(_), _) => "take the square root of a negative number",
                    (Op::Log(_), _) => "take the logarithm of a negative number",
                    (Op::Div(..), Fault::Undefined) => "divide zero by zero",
                    (Op::Pow(..), Fault::Pole) => "raise zero to a negative power",
                    (_, Fault::Pole) => "divide by zero",
                    (_, Fault::Undefined) => "be undefined (NaN) after an overflow",
                };
                parse::locate(
                    formula,
                    at(node),
                    &format!(
                        "{} can {what} near ({}, {}, {}); a surface's formula must be \
                         defined, with no poles, wherever it is used",
                        ast.print(node),
                        short(p[0]),
                        short(p[1]),
                        short(p[2])
                    ),
                )
            }
            Problem::NeverBelow => format!(
                "{text} is never negative, so nothing would be below the surface (below is \
                 where the formula is negative)"
            ),
            Problem::NeverAbove => format!(
                "{text} is never positive within {WORLD} cm of the origin, so nothing would be \
                 above the surface (above is where the formula is positive)"
            ),
        })?;
        Ok(Self {
            text,
            tape: compiled.tape,
            bounding_box,
        })
    }

    /// The canonical formula string: it parses back to this formula.
    pub fn formula(&self) -> &str {
        &self.text
    }

    /// The compiled formula.
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// The formula's value at `point`. Only its sign is meaningful: it is
    /// a distance only if the formula is one.
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        SLOTS.with(|slots| match slots.try_borrow_mut() {
            Ok(mut slots) => self.tape.eval(point, &mut slots),
            Err(_) => self.tape.eval(point, &mut Vec::new()),
        })
    }

    /// A box holding every point where the formula is not positive (so
    /// everything below the surface), proved by interval arithmetic, or
    /// `None` if none could be.
    pub fn bounding_box(&self) -> Option<([f64; 3], [f64; 3])> {
        self.bounding_box
    }
}

impl Serialize for ImplicitFormula {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
    }
}

impl<'de> Deserialize<'de> for ImplicitFormula {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::new(&text).map_err(serde::de::Error::custom)
    }
}
