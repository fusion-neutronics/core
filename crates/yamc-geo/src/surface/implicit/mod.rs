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
//! - one shown to be zero, or below the smallest normal number, throughout
//!   some part of that region, whose points would be on neither side;
//! - one shown never to be negative there (nothing below the surface) or,
//!   with no box, never to be positive there.
//!
//! The bounding box is proved by interval arithmetic too (see `bounds`);
//! a periodic field, a plane or a twist gets none, and the cells using it
//! must bound it instead. All of this holds within [`WORLD`] of the origin
//! (about 10 km), not beyond: far enough out a high power or an
//! exponential overflows, and two overflows can make a NaN.
//!
//! Ray crossings come from a certified search on the same evaluators (see
//! `ray`): the first point where the sign the cell test reads changes,
//! found to 1e-11 cm, with the failure policy the free-form kinds share.
//! The formula's value is a distance only if the formula is one, so it is
//! never used as one: the search steps by enclosures, and a particle counts
//! as on the surface within 1e-7 cm of it to first order (`|f| / |f'|`).

mod bounds;
mod compile;
mod interval;
mod parse;
mod ray;
#[cfg(test)]
mod ray_tests;
mod tape;
#[cfg(test)]
mod tests;

use crate::surface::search::{along_unit, Trace};
use crate::surface::{BoundaryType, Surface, SurfaceKind};
pub use bounds::WORLD;
use bounds::{Cell, Extent, Fault, Problem};
pub use interval::{Interval, LIBM_ULPS};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
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
        // What slot `slot` can do wrong, and the formula node that made it.
        let explain = |slot: usize, fault: Fault| {
            let what = match (tape.ops[slot], fault) {
                (Op::Sqrt(_), _) => "take the square root of a negative number",
                (Op::Log(_), _) => "take the logarithm of a negative number",
                (Op::Div(..), Fault::Undefined) => "divide zero by zero",
                (Op::Pow(..), Fault::Pole) => "raise zero to a negative power",
                (_, Fault::Pole) => "divide by zero",
                (_, Fault::Undefined) => "be undefined (NaN) after an overflow",
            };
            (what, compiled.origin[slot])
        };
        let bounding_box = checked.map_err(|problem| match problem {
            Problem::Fault { slot, fault, at: p } => {
                let (what, node) = explain(slot, fault);
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
            Problem::Unproved { slot, fault } => {
                let (what, node) = explain(slot, fault);
                parse::locate(
                    formula,
                    at(node),
                    &format!(
                        "{} could not be shown never to {what} (a surface's formula must be \
                         shown to be defined, with no poles, wherever it is used); interval \
                         arithmetic, which checks it, loses track of a square written out: \
                         write x^2 - 2 * x + 2 as (x - 1)^2 + 1, say",
                        ast.print(node)
                    ),
                )
            }
            Problem::Flat { at: p } => format!(
                "{text} is zero, or too small for its sign to mean anything (below \
                 2.2e-308), throughout a region around ({}, {}, {}), whose points would be on \
                 neither side of the surface",
                short(p[0]),
                short(p[1]),
                short(p[2])
            ),
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

    /// The formula string for messages: its first 57 characters and `...`
    /// if it is longer than 60.
    pub fn brief(&self) -> String {
        if self.text.chars().count() <= 60 {
            self.text.clone()
        } else {
            let head: String = self.text.chars().take(57).collect();
            format!("{head}...")
        }
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

    /// Ray parameter `t` of the first crossing of the surface along
    /// `point + t * direction` beyond 1e-10 cm, if before `t_max`; none for
    /// a zero or non-finite direction or a non-finite point. A search with
    /// no bounding box to stop it stops `WORLD` from the origin. The
    /// search's answer, with whether it ran out of evaluations (see
    /// `ray::trace`).
    pub(super) fn trace(
        &self,
        point: [f64; 3],
        direction: [f64; 3],
        on_surface: Option<f64>,
        vacuum: bool,
        t_max: f64,
    ) -> Trace {
        // The search measures length, so it runs on the unit direction.
        along_unit(point, direction, t_max, |unit, t_max| {
            ray::trace(
                &self.tape,
                self.bounding_box,
                point,
                unit,
                on_surface,
                vacuum,
                t_max,
            )
        })
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

impl Surface {
    /// An implicit surface: the zero set of `formula`, below where it is
    /// negative.
    pub fn new_implicit(
        formula: Arc<ImplicitFormula>,
        surface_id: Option<usize>,
        boundary_type: Option<BoundaryType>,
    ) -> Self {
        Surface {
            surface_id,
            kind: SurfaceKind::Implicit { formula },
            boundary: boundary_type.unwrap_or_default(),
            name: None,
        }
    }
}

/// Formulas built while loading, by their text: building one can take a
/// tenth of a second, and a model loaded again while it is still in use, or
/// with surfaces that share a formula but not an id, needs it again.
static LOADED: OnceLock<Mutex<HashMap<String, Weak<ImplicitFormula>>>> = OnceLock::new();

/// Deserialize a formula, sharing it with every other live formula loaded
/// from the same text (for `SurfaceKind::Implicit`).
pub(crate) fn deserialize_shared<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Arc<ImplicitFormula>, D::Error> {
    let text = String::deserialize(deserializer)?;
    let mut loaded = LOADED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(formula) = loaded.get(&text).and_then(Weak::upgrade) {
        return Ok(formula);
    }
    let formula = Arc::new(ImplicitFormula::new(&text).map_err(serde::de::Error::custom)?);
    loaded.retain(|_, f| f.strong_count() > 0);
    loaded.insert(text, Arc::downgrade(&formula));
    Ok(formula)
}
