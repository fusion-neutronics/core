//! Lowering a parsed formula to a [`Tape`].
//!
//! The instructions are built in a hash-consed arena: an instruction equal
//! to an earlier one (same operation, same operands, constants compared
//! bit for bit) is not stored again, so a subexpression written twice is
//! evaluated once. An instruction whose operands are all constants is
//! folded to a constant by [`point`], the point evaluator's own arithmetic,
//! and `a * a` is stored as `square(a)`, its exact equivalent with a
//! tighter interval. Nodes are appended in creation order, which is a
//! topological order, and the hash map is only looked up, never iterated,
//! so the tape (the reachable nodes in that order) depends on nothing but
//! the formula.
//!
//! `^` with an integer exponent becomes a chain of squarings and
//! multiplications (and a division for a negative one); with any other
//! exponent it needs a base that interval arithmetic shows is never
//! negative. `smin(a, b, k)` is `min(a, b) - max(k - |a - b|, 0)^2 / (4k)`
//! (the polynomial smooth minimum, which differs from `min` only where
//! `a` and `b` are within `k` of each other) and `smax` its mirror image.

use super::interval::Interval;
use super::parse::{Ast, BinOp, Func, Kind};
use super::tape::{point, Op, Tape};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Largest integer exponent written out as multiplications.
const MAX_INTEGER_EXPONENT: f64 = 1024.0;

/// A compiled formula: the tape and, for each slot, the formula node that
/// made it (for messages).
pub(super) struct Compiled {
    pub(super) tape: Tape,
    pub(super) origin: Vec<u32>,
}

/// A formula that does not compile: the node at fault and why.
pub(super) struct CompileError {
    pub(super) node: u32,
    pub(super) message: String,
}

/// The hash-consed instruction arena.
struct Builder {
    ops: Vec<Op>,
    origin: Vec<u32>,
    index: HashMap<(u8, u32, u64), u32>,
    /// The formula node being lowered.
    current: u32,
}

/// An instruction's identity: its kind, its operands and any constant, by
/// bits.
fn key(op: &Op) -> (u8, u32, u64) {
    let tag = match op {
        Op::X => 0,
        Op::Y => 1,
        Op::Z => 2,
        Op::Const(_) => 3,
        Op::Neg(_) => 4,
        Op::Square(_) => 5,
        Op::Sqrt(_) => 6,
        Op::Abs(_) => 7,
        Op::Exp(_) => 8,
        Op::Log(_) => 9,
        Op::Sin(_) => 10,
        Op::Cos(_) => 11,
        Op::Pow(..) => 12,
        Op::Add(..) => 13,
        Op::Sub(..) => 14,
        Op::Mul(..) => 15,
        Op::Div(..) => 16,
        Op::Min(..) => 17,
        Op::Max(..) => 18,
    };
    match *op {
        Op::Const(c) => (tag, 0, c.to_bits()),
        Op::Pow(a, p) => (tag, a, p.to_bits()),
        _ => {
            let [a, b] = op.args();
            (tag, a, u64::from(b))
        }
    }
}

impl Builder {
    fn constant_of(&self, n: u32) -> Option<f64> {
        match self.ops[n as usize] {
            Op::Const(c) => Some(c),
            _ => None,
        }
    }

    fn intern(&mut self, op: Op) -> u32 {
        let key = key(&op);
        if let Some(&n) = self.index.get(&key) {
            return n;
        }
        let n = self.ops.len() as u32;
        self.ops.push(op);
        self.origin.push(self.current);
        self.index.insert(key, n);
        n
    }

    fn constant(&mut self, c: f64) -> u32 {
        self.intern(Op::Const(c))
    }

    /// Add `op`, folded if its operands are all constants.
    fn push(&mut self, op: Op) -> u32 {
        let op = match op {
            Op::Mul(a, b) if a == b => Op::Square(a),
            _ => op,
        };
        let [a, b] = op.args();
        let values: Vec<f64> = [a, b]
            .into_iter()
            .filter(|&i| i != u32::MAX)
            .map_while(|i| self.constant_of(i))
            .collect();
        let operands = [a, b].iter().filter(|&&i| i != u32::MAX).count();
        if operands > 0 && values.len() == operands {
            let local = op.map(|i| if i == a { 0 } else { 1 });
            return self.constant(point(local, &values, [0.0; 3]));
        }
        self.intern(op)
    }

    /// The reachable instructions below `root`, as a tape, with each
    /// slot's origin.
    fn tape(&self, root: u32) -> (Tape, Vec<u32>) {
        let n = root as usize + 1;
        let mut live = vec![false; n];
        live[n - 1] = true;
        for i in (0..n).rev() {
            if live[i] {
                for a in self.ops[i].args() {
                    if a != u32::MAX {
                        live[a as usize] = true;
                    }
                }
            }
        }
        let mut slot = vec![u32::MAX; n];
        let mut ops = Vec::new();
        let mut origin = Vec::new();
        for i in 0..n {
            if live[i] {
                slot[i] = ops.len() as u32;
                ops.push(self.ops[i].map(|a| slot[a as usize]));
                origin.push(self.origin[i]);
            }
        }
        (Tape { ops }, origin)
    }

    /// An enclosure of node `n` over all of space.
    fn range(&self, n: u32) -> Interval {
        let (tape, _) = self.tape(n);
        tape.eval_interval([Interval::ENTIRE; 3], &mut Vec::new())
    }

    /// `a^n` by squaring and multiplying, `1 / a^|n|` for a negative `n`.
    fn integer_power(&mut self, a: u32, n: i64) -> u32 {
        if n == 0 {
            return self.constant(1.0);
        }
        let m = n.unsigned_abs();
        let mut r = a;
        for bit in (0..63 - m.leading_zeros()).rev() {
            r = self.push(Op::Square(r));
            if m >> bit & 1 == 1 {
                r = self.push(Op::Mul(r, a));
            }
        }
        if n < 0 {
            let one = self.constant(1.0);
            r = self.push(Op::Div(one, r));
        }
        r
    }

    /// `smin(a, b, k)`, or `smax` if not `min`.
    fn smooth(&mut self, a: u32, b: u32, k: f64, min: bool) -> u32 {
        let m = self.push(if min { Op::Min(a, b) } else { Op::Max(a, b) });
        let diff = self.push(Op::Sub(a, b));
        let gap = self.push(Op::Abs(diff));
        let k_node = self.constant(k);
        let overlap = self.push(Op::Sub(k_node, gap));
        let zero = self.constant(0.0);
        let overlap = self.push(Op::Max(overlap, zero));
        let squared = self.push(Op::Square(overlap));
        let scale = self.constant(4.0 * k);
        let bump = self.push(Op::Div(squared, scale));
        self.push(if min {
            Op::Sub(m, bump)
        } else {
            Op::Add(m, bump)
        })
    }
}

fn fail<T>(node: u32, message: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError {
        node,
        message: message.into(),
    })
}

/// Lower a parsed formula to a tape.
pub(super) fn compile(ast: &Ast) -> Result<Compiled, CompileError> {
    let mut b = Builder {
        ops: Vec::new(),
        origin: Vec::new(),
        index: HashMap::new(),
        current: 0,
    };
    let mut value: Vec<u32> = Vec::with_capacity(ast.nodes.len());
    for (i, node) in ast.nodes.iter().enumerate() {
        let i = i as u32;
        b.current = i;
        let at = |n: &u32| value[*n as usize];
        let r = match &node.kind {
            Kind::Num(v) => b.constant(*v),
            Kind::Pi => b.constant(PI),
            Kind::Var(k) => b.intern([Op::X, Op::Y, Op::Z][*k]),
            Kind::Neg(a) => b.push(Op::Neg(at(a))),
            Kind::Bin(op, l, r) => {
                let (l, r) = (at(l), at(r));
                b.push(match op {
                    BinOp::Add => Op::Add(l, r),
                    BinOp::Sub => Op::Sub(l, r),
                    BinOp::Mul => Op::Mul(l, r),
                    BinOp::Div => Op::Div(l, r),
                })
            }
            Kind::Pow(base, exponent) => {
                let Some(p) = b.constant_of(at(exponent)) else {
                    return fail(
                        *exponent,
                        "the exponent must be a constant (x, y and z cannot appear in it)",
                    );
                };
                let base_node = at(base);
                if p == p.trunc() {
                    if p.abs() > MAX_INTEGER_EXPONENT {
                        return fail(
                            *exponent,
                            format!(
                                "an integer exponent can be at most {MAX_INTEGER_EXPONENT} in \
                                 size"
                            ),
                        );
                    }
                    b.integer_power(base_node, p as i64)
                } else {
                    if b.range(base_node).lo < 0.0 {
                        return fail(
                            i,
                            format!(
                                "a power with a non-integer exponent needs a base that cannot \
                                 be negative, and {} can be: write abs(...) around it, or a \
                                 sum of squares",
                                ast.print(*base)
                            ),
                        );
                    }
                    b.push(Op::Pow(base_node, p))
                }
            }
            Kind::Call(f, args) => {
                let a = at(&args[0]);
                match f {
                    Func::Sqrt => b.push(Op::Sqrt(a)),
                    Func::Abs => b.push(Op::Abs(a)),
                    Func::Exp => b.push(Op::Exp(a)),
                    Func::Log => b.push(Op::Log(a)),
                    Func::Sin => b.push(Op::Sin(a)),
                    Func::Cos => b.push(Op::Cos(a)),
                    Func::Min | Func::Max => args[1..].iter().fold(a, |acc, n| {
                        let n = at(n);
                        b.push(if *f == Func::Min {
                            Op::Min(acc, n)
                        } else {
                            Op::Max(acc, n)
                        })
                    }),
                    Func::SMin | Func::SMax => {
                        let k = b.constant_of(at(&args[2]));
                        let Some(k) = k.filter(|k| *k > 0.0 && (4.0 * k).is_finite()) else {
                            return fail(
                                args[2],
                                format!(
                                    "the blending distance k of {} must be a positive constant \
                                     (below 4e307)",
                                    f.name()
                                ),
                            );
                        };
                        b.smooth(a, at(&args[1]), k, *f == Func::SMin)
                    }
                }
            }
        };
        if let Some(c) = b.constant_of(r)
            && !c.is_finite()
        {
            return fail(
                i,
                format!(
                    "{} is not a finite number (it evaluates to {c})",
                    ast.print(i)
                ),
            );
        }
        value.push(r);
    }
    let root = ast.root();
    let top = value[root as usize];
    if b.constant_of(top).is_some() {
        return fail(
            root,
            "the formula is a constant: it must depend on x, y or z",
        );
    }
    let (tape, origin) = b.tape(top);
    Ok(Compiled { tape, origin })
}
