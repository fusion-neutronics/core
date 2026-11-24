//! What interval arithmetic proves about a formula at construction: a box
//! outside which it is positive, and that it is defined and changes sign
//! where it is used.
//!
//! Everything is proved within [`WORLD`] of the origin, far beyond any
//! model: past it a formula could overflow (`x^64` does at 6.5e4 cm, and
//! `inf - inf` is NaN), and nothing is claimed there.
//!
//! Both rest on [`check`], an interval evaluation over a box that also
//! looks, instruction by instruction, for a way the point evaluation could
//! give NaN there (a square root of a negative number, `0 / 0`, `inf - inf`
//! after an overflow, ...): an enclosure says nothing about a NaN, so a box
//! where one is possible proves nothing. Where the formula is used it must
//! also have no pole (a division by a value that can be zero), whose sign
//! flips with no surface there. An overflow to an infinity of the right
//! sign is allowed. Nor may it be zero throughout some part of the region,
//! or so small there that its sign is the noise of an underflow (`max(x,
//! 0) * y`, say, for every `x < 0`), as the points there would be on
//! neither side of the surface; such a part is refused where interval
//! arithmetic shows it.
//!
//! The box: `f > 0` is proved on the six slabs between the cube
//! `[-w, w]^3` and the world's faces, for the smallest power of two `w`
//! that allows it, and then each face of the box is pushed in by a
//! best-first search of an octree (cells where `f > 0` is proved are
//! dropped; the most outlying cell left at the finest level, or when the
//! search's budget runs out, bounds that face). The cells of size `h` the
//! surface passes through near its extreme number about `R / h` at a
//! curved face of radius `R` and `A / h^2` at a flat one of area `A`, so
//! the budget leaves the box loose by about a percent of its size at a
//! curved face and a few percent at a flat one. Where no `w` works (a
//! periodic field, a plane, a rotation whose angle grows without bound)
//! there is no box.

use super::interval::Interval;
use super::tape::{interval, Op, Tape};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Formulas are checked, and boxes proved, within this distance (cm) of
/// the origin along each axis: 2^20 cm, about 10 km.
pub const WORLD: f64 = 1_048_576.0;
/// Smallest half-width (cm) of the cube the slabs are proved outside:
/// 2^-10 cm.
const SMALLEST_CUBE: f64 = 1.0 / 1024.0;
/// Levels of the octree each face search goes down.
const FACE_DEPTH: u32 = 24;
/// Cells a face search examines before settling for the bound reached.
const FACE_BUDGET: usize = 4_096;
/// Levels the search for a NaN or a pole in a region goes down, and the
/// search for points of each sign.
const FAULT_DEPTH: u32 = 40;
const SIGN_DEPTH: u32 = 12;
/// Cells either search examines.
const CHECK_BUDGET: usize = 16_384;
/// Cells the search for a region where the formula is zero examines.
const FLAT_BUDGET: usize = 4_096;

/// An axis-aligned box `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Cell {
    pub(super) lo: [f64; 3],
    pub(super) hi: [f64; 3],
}

impl Cell {
    pub(super) fn cube(w: f64) -> Self {
        Self {
            lo: [-w; 3],
            hi: [w; 3],
        }
    }

    fn intervals(&self) -> [Interval; 3] {
        std::array::from_fn(|k| Interval::new(self.lo[k], self.hi[k]))
    }

    pub(super) fn centre(&self) -> [f64; 3] {
        std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k]))
    }

    /// The eight cells halving it along each axis (sharing their faces
    /// exactly).
    fn children(&self) -> [Cell; 8] {
        let m = self.centre();
        std::array::from_fn(|i| {
            let mut c = *self;
            for (k, &mid) in m.iter().enumerate() {
                if i >> k & 1 == 0 {
                    c.hi[k] = mid;
                } else {
                    c.lo[k] = mid;
                }
            }
            c
        })
    }
}

/// How the point evaluation can fail at an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fault {
    /// It can give NaN.
    Undefined,
    /// It can divide by zero (or raise zero to a negative power).
    Pole,
}

/// Whether `op` can give NaN for some operands within the enclosures in
/// `s` (none of which is NaN).
fn undefined(op: Op, s: &[Interval]) -> bool {
    let at = |i: u32| s[i as usize];
    let unbounded = |v: Interval| v.lo == f64::NEG_INFINITY || v.hi == f64::INFINITY;
    match op {
        Op::Sqrt(a) | Op::Log(a) | Op::Pow(a, _) => at(a).lo < 0.0,
        Op::Sin(a) | Op::Cos(a) => unbounded(at(a)),
        Op::Add(a, b) => {
            let (x, y) = (at(a), at(b));
            (x.hi == f64::INFINITY && y.lo == f64::NEG_INFINITY)
                || (x.lo == f64::NEG_INFINITY && y.hi == f64::INFINITY)
        }
        Op::Sub(a, b) => {
            let (x, y) = (at(a), at(b));
            (x.hi == f64::INFINITY && y.hi == f64::INFINITY)
                || (x.lo == f64::NEG_INFINITY && y.lo == f64::NEG_INFINITY)
        }
        Op::Mul(a, b) => {
            let (x, y) = (at(a), at(b));
            (x.has_zero() && unbounded(y)) || (y.has_zero() && unbounded(x))
        }
        Op::Div(a, b) => {
            let (x, y) = (at(a), at(b));
            (x.has_zero() && y.has_zero()) || (unbounded(x) && unbounded(y))
        }
        Op::X
        | Op::Y
        | Op::Z
        | Op::Const(_)
        | Op::Neg(_)
        | Op::Square(_)
        | Op::Abs(_)
        | Op::Exp(_)
        | Op::Min(..)
        | Op::Max(..) => false,
    }
}

/// Whether `op` divides by a value within the enclosures in `s` that can
/// be zero.
fn pole(op: Op, s: &[Interval]) -> bool {
    let at = |i: u32| s[i as usize];
    match op {
        Op::Div(_, b) => at(b).has_zero(),
        Op::Pow(a, p) => p < 0.0 && at(a).has_zero(),
        _ => false,
    }
}

/// An enclosure of `tape` over `cell` that holds for the point evaluation
/// everywhere in it: the slot and the way it fails if some instruction can
/// give NaN there or, with `poles`, divide by zero.
pub(super) fn check(
    tape: &Tape,
    cell: [Interval; 3],
    poles: bool,
    s: &mut Vec<Interval>,
) -> Result<Interval, (usize, Fault)> {
    s.clear();
    for (i, &op) in tape.ops.iter().enumerate() {
        if undefined(op, s) {
            return Err((i, Fault::Undefined));
        }
        if poles && pole(op, s) {
            return Err((i, Fault::Pole));
        }
        let v = interval(op, s, &cell);
        s.push(v);
    }
    s.last().copied().ok_or((0, Fault::Undefined))
}

/// Whether `f > 0` is proved throughout `cell` (an overflow to `+inf`
/// counts as positive).
fn positive(tape: &Tape, cell: [Interval; 3], s: &mut Vec<Interval>) -> bool {
    check(tape, cell, false, s).is_ok_and(|f| f.lo > 0.0)
}

/// Where a formula's non-positive points within the world are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Extent {
    /// Within this box: `f > 0` is proved everywhere else.
    Inside(Cell),
    /// Not provably within any box.
    Unbounded,
    /// Nowhere: `f > 0` throughout.
    Empty,
}

/// Whether `f > 0` is proved on the six slabs between `[-w, w]^3` and the
/// faces of the world.
fn slabs_positive(tape: &Tape, w: f64, s: &mut Vec<Interval>) -> bool {
    (0..3).all(|k| {
        [Interval::new(w, WORLD), Interval::new(-WORLD, -w)]
            .into_iter()
            .all(|far| {
                let mut slab = [Interval::new(-WORLD, WORLD); 3];
                slab[k] = far;
                positive(tape, slab, s)
            })
    })
}

/// A cell in a face search, the most outlying first: by how far it
/// reaches towards the face, then the deeper, then the earlier.
struct Candidate {
    reach: f64,
    depth: u32,
    order: usize,
    cell: Cell,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.reach
            .total_cmp(&o.reach)
            .then(self.depth.cmp(&o.depth))
            .then(o.order.cmp(&self.order))
    }
}

/// How far the non-positive points of `tape` within `root` reach along
/// axis `axis` (towards `+inf` if `upper`, else towards `-inf`, as a
/// bound on `x_axis` or `-x_axis`), or `None` if there are none.
fn face(tape: &Tape, root: Cell, axis: usize, upper: bool, s: &mut Vec<Interval>) -> Option<f64> {
    let reach = |c: &Cell| if upper { c.hi[axis] } else { -c.lo[axis] };
    let mut heap = BinaryHeap::new();
    heap.push(Candidate {
        reach: reach(&root),
        depth: 0,
        order: 0,
        cell: root,
    });
    let mut order = 0;
    let mut examined = 0;
    while let Some(c) = heap.pop() {
        // Every cell left reaches no further than this one.
        if examined == FACE_BUDGET {
            return Some(c.reach);
        }
        examined += 1;
        if positive(tape, c.cell.intervals(), s) {
            continue;
        }
        if c.depth == FACE_DEPTH {
            return Some(c.reach);
        }
        for child in c.cell.children() {
            order += 1;
            heap.push(Candidate {
                reach: reach(&child),
                depth: c.depth + 1,
                order,
                cell: child,
            });
        }
    }
    None
}

/// The box of the non-positive points within `root` (its points outside
/// the box are proved positive), or `Empty` if there are none.
fn hull(tape: &Tape, root: Cell, s: &mut Vec<Interval>) -> Extent {
    let mut cell = root;
    for axis in 0..3 {
        let Some(hi) = face(tape, root, axis, true, s) else {
            return Extent::Empty;
        };
        let Some(lo) = face(tape, root, axis, false, s) else {
            return Extent::Empty;
        };
        cell.hi[axis] = hi;
        cell.lo[axis] = -lo;
    }
    Extent::Inside(cell)
}

/// The box of the points where `f <= 0` (see the module docs).
pub(super) fn extent(tape: &Tape) -> Extent {
    let mut s = Vec::new();
    let mut w = SMALLEST_CUBE;
    while !slabs_positive(tape, w, &mut s) {
        w *= 2.0;
        if w >= WORLD {
            return Extent::Unbounded;
        }
    }
    hull(tape, Cell::cube(w), &mut s)
}

/// Why a formula was refused in a region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Problem {
    /// Slot `slot` can give NaN or divide by zero near `at`.
    Fault {
        slot: usize,
        fault: Fault,
        at: [f64; 3],
    },
    /// Slot `slot` could not be shown never to give NaN or divide by zero
    /// before the check's budget ran out.
    Unproved { slot: usize, fault: Fault },
    /// `f` is zero, or smaller than the smallest normal number, throughout
    /// a region around `at`.
    Flat { at: [f64; 3] },
    /// `f >= 0` throughout: nothing is below the surface.
    NeverBelow,
    /// `f <= 0` throughout: nothing is above the surface.
    NeverAbove,
}

/// Check that `tape` is defined and has no pole throughout `region`, takes
/// both signs in it (only negative ones need finding if `above` is known),
/// and is not zero throughout any part of it found.
pub(super) fn validate(tape: &Tape, region: Cell, above: bool) -> Result<(), Problem> {
    defined(tape, region)?;
    signs(tape, region, above)?;
    flat(tape, region)
}

/// Refine an octree, depth first, wherever [`check`] finds a possible NaN
/// or pole, until every cell is clear; refuse if one is left at the finest
/// level (a fault there) or when the budget runs out (none shown, but none
/// ruled out). The cell nearer the region's centre is refined first, so a
/// fault is reported at a point of it near there.
fn defined(tape: &Tape, region: Cell) -> Result<(), Problem> {
    let centre = region.centre();
    let distance = |c: &Cell| {
        let m = c.centre();
        let d: [f64; 3] = std::array::from_fn(|k| m[k] - centre[k]);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let mut s = Vec::new();
    let mut stack = vec![(region, 0)];
    let mut examined = 0;
    while let Some((cell, depth)) = stack.pop() {
        examined += 1;
        let Err((slot, fault)) = check(tape, cell.intervals(), true, &mut s) else {
            continue;
        };
        if depth == FAULT_DEPTH {
            return Err(Problem::Fault {
                slot,
                fault,
                at: cell.centre(),
            });
        }
        if examined >= CHECK_BUDGET {
            return Err(Problem::Unproved { slot, fault });
        }
        let mut children = cell.children();
        children.sort_by(|a, b| distance(b).total_cmp(&distance(a)));
        stack.extend(children.map(|c| (c, depth + 1)));
    }
    Ok(())
}

/// Look, breadth first, for a cell throughout which `tape` is zero or so
/// small that its sign is the noise of an underflow (its enclosure holds 0
/// and is within the smallest normal number of it): points there would be
/// on neither side of the surface, or on one by chance. Only the cells
/// whose enclosure holds 0 are refined, and the formula is accepted when
/// the budget runs out. `tape` is known to be defined throughout `region`,
/// so its plain enclosures hold.
fn flat(tape: &Tape, region: Cell) -> Result<(), Problem> {
    let mut s = Vec::new();
    let mut queue = VecDeque::from([(region, 0)]);
    let mut examined = 0;
    while let Some((cell, depth)) = queue.pop_front() {
        if examined == FLAT_BUDGET {
            break;
        }
        examined += 1;
        let f = tape.eval_interval(cell.intervals(), &mut s);
        if !f.has_zero() {
            continue;
        }
        if f.magnitude() < f64::MIN_POSITIVE {
            return Err(Problem::Flat { at: cell.centre() });
        }
        if depth < FAULT_DEPTH {
            queue.extend(cell.children().map(|c| (c, depth + 1)));
        }
    }
    Ok(())
}

/// Look for points of each sign, breadth first, refining the cells that
/// could hold a sign not yet seen. A sign is shown taken by a point that
/// has it and missing when every cell is shown not to have it; if the
/// refinement runs out first, it is taken to be there. `tape` is known to
/// be defined throughout `region`, so its plain enclosures hold.
fn signs(tape: &Tape, region: Cell, above: bool) -> Result<(), Problem> {
    let (mut s, mut slots) = (Vec::new(), Vec::new());
    let (mut below, mut above) = (false, above);
    // Whether a cell that might hold a sign not yet seen was left
    // unrefined.
    let (mut maybe_below, mut maybe_above) = (false, false);
    let mut queue = VecDeque::from([(region, 0)]);
    let mut examined = 0;
    while let Some((cell, depth)) = queue.pop_front() {
        if below && above {
            break;
        }
        if examined == CHECK_BUDGET {
            maybe_below = true;
            maybe_above = true;
            break;
        }
        examined += 1;
        let f = tape.eval_interval(cell.intervals(), &mut s);
        let v = tape.eval(cell.centre(), &mut slots);
        below |= v < 0.0;
        above |= v > 0.0;
        let want_below = !below && f.lo < 0.0;
        let want_above = !above && f.hi > 0.0;
        if !(want_below || want_above) {
            continue;
        }
        if depth == SIGN_DEPTH {
            maybe_below |= want_below;
            maybe_above |= want_above;
            continue;
        }
        queue.extend(cell.children().map(|c| (c, depth + 1)));
    }
    if !below && !maybe_below {
        return Err(Problem::NeverBelow);
    }
    if !above && !maybe_above {
        return Err(Problem::NeverAbove);
    }
    Ok(())
}
