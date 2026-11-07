//! Flattened RPN representation of a CSG region, compiled once from the
//! `RegionExpr` tree and evaluated iteratively (no recursion, GPU-ready).

use super::{HalfspaceType, RegionExpr};
use crate::surface::Surface;
use std::collections::HashMap;
use std::sync::Arc;

/// Instruction in the flat RPN evaluation of a CSG region.
///
/// The recursive `RegionExpr` tree is compiled into a linear sequence of these
/// ops at setup time. The transport loop then evaluates containment iteratively
/// with a small fixed-size stack -- no recursion, no heap allocation, GPU-ready.
#[derive(Debug, Clone, Copy)]
pub enum RegionOp {
    /// Evaluate surface at index, push `(evaluate(point) > 0.0)`.
    Above(u32),
    /// Evaluate surface at index, push `(evaluate(point) < 0.0)`.
    Below(u32),
    /// Pop two booleans, push `(a && b)`.
    And,
    /// Pop two booleans, push `(a || b)`.
    Or,
    /// Pop one boolean, push `(!a)`.
    Not,
}

/// Flattened RPN representation of a CSG region expression.
///
/// Built once at setup time from a `RegionExpr` tree. Evaluates containment
/// iteratively by scanning the `ops` array and maintaining a small boolean stack.
/// All surface references are stored in a flat `Vec` indexed by the `u32` operands
/// in `Above`/`Below` instructions.
#[derive(Debug, Clone)]
pub struct FlatRegion {
    ops: Vec<RegionOp>,
    surfaces: Vec<Arc<Surface>>,
}

/// Evaluation stack size. [`walk_rpn`] orders operands so that a region needs
/// more than `n` slots only if it has at least `2^(n-1)` half-spaces, so 64 is
/// never reached by a real model.
const MAX_STACK_DEPTH: usize = 64;

/// One step of a region's RPN program, in evaluation order.
#[derive(Debug, Clone, Copy)]
pub enum RpnStep<'a> {
    /// Push the half-space's truth value.
    Halfspace(&'a HalfspaceType),
    /// Pop two booleans, push `(a && b)`.
    And,
    /// Pop two booleans, push `(a || b)`.
    Or,
    /// Pop one boolean, push `(!a)`.
    Not,
}

/// Walk `expr` as an RPN program, calling `step` for each step in evaluation
/// order, and return the evaluation stack depth the program needs.
///
/// The operands of `&` and `|` commute, so the one needing the deeper stack is
/// emitted first (Sethi-Ullman ordering). A chain such as `a & (b & (c & ...))`
/// then needs two stack slots however long it is, where left-to-right
/// emission needs one slot per level. Every RPN compiler of a region (the CPU
/// [`FlatRegion`] and the GPU region program) uses this walk, so they agree.
pub fn walk_rpn<'a>(expr: &'a RegionExpr, step: &mut impl FnMut(RpnStep<'a>)) -> usize {
    fn need(expr: &RegionExpr, needs: &mut HashMap<*const RegionExpr, usize>) -> usize {
        let n = match expr {
            RegionExpr::Halfspace(_) => 1,
            RegionExpr::Complement(inner) => need(inner, needs),
            RegionExpr::Union(a, b) | RegionExpr::Intersection(a, b) => {
                let (na, nb) = (need(a, needs), need(b, needs));
                if na == nb {
                    na + 1
                } else {
                    na.max(nb)
                }
            }
        };
        needs.insert(expr as *const RegionExpr, n);
        n
    }

    fn emit<'a>(
        expr: &'a RegionExpr,
        needs: &HashMap<*const RegionExpr, usize>,
        step: &mut impl FnMut(RpnStep<'a>),
    ) {
        match expr {
            RegionExpr::Halfspace(hs) => step(RpnStep::Halfspace(hs)),
            RegionExpr::Union(a, b) | RegionExpr::Intersection(a, b) => {
                let (a, b): (&RegionExpr, &RegionExpr) = (a, b);
                let (first, second) =
                    if needs[&(b as *const RegionExpr)] > needs[&(a as *const RegionExpr)] {
                        (b, a)
                    } else {
                        (a, b)
                    };
                emit(first, needs, step);
                emit(second, needs, step);
                step(if matches!(expr, RegionExpr::Union(..)) {
                    RpnStep::Or
                } else {
                    RpnStep::And
                });
            }
            RegionExpr::Complement(inner) => {
                emit(inner, needs, step);
                step(RpnStep::Not);
            }
        }
    }

    let mut needs = HashMap::new();
    let depth = need(expr, &mut needs);
    emit(expr, &needs, step);
    depth
}

impl FlatRegion {
    /// Compile a `RegionExpr` tree into a flat RPN instruction sequence.
    pub fn from_expr(expr: &RegionExpr) -> Self {
        let mut ops = Vec::new();
        let mut surfaces: Vec<Arc<Surface>> = Vec::new();
        let mut surface_map: HashMap<*const Surface, u32> = HashMap::new();

        let depth = walk_rpn(expr, &mut |step| {
            ops.push(match step {
                RpnStep::Halfspace(hs) => {
                    let (surf, above) = match hs {
                        HalfspaceType::Above(s) => (s, true),
                        HalfspaceType::Below(s) => (s, false),
                    };
                    let idx = *surface_map.entry(Arc::as_ptr(surf)).or_insert_with(|| {
                        let idx = u32::try_from(surfaces.len())
                            .expect("a region references more than u32::MAX surfaces");
                        surfaces.push(surf.clone());
                        idx
                    });
                    if above {
                        RegionOp::Above(idx)
                    } else {
                        RegionOp::Below(idx)
                    }
                }
                RpnStep::And => RegionOp::And,
                RpnStep::Or => RegionOp::Or,
                RpnStep::Not => RegionOp::Not,
            });
        });
        assert!(
            depth <= MAX_STACK_DEPTH,
            "region needs an evaluation stack of {depth}, more than the {MAX_STACK_DEPTH} supported"
        );
        FlatRegion { ops, surfaces }
    }

    /// Test whether `point` is inside the region.
    #[inline]
    pub fn contains(&self, point: (f64, f64, f64)) -> bool {
        let mut stack = [false; MAX_STACK_DEPTH];
        let mut top: usize = 0;

        for op in &self.ops {
            match *op {
                RegionOp::Above(idx) => {
                    stack[top] = self.surfaces[idx as usize].evaluate(point) > 0.0;
                    top += 1;
                }
                RegionOp::Below(idx) => {
                    stack[top] = self.surfaces[idx as usize].evaluate(point) < 0.0;
                    top += 1;
                }
                RegionOp::And => {
                    top -= 1;
                    stack[top - 1] &= stack[top];
                }
                RegionOp::Or => {
                    top -= 1;
                    stack[top - 1] |= stack[top];
                }
                RegionOp::Not => {
                    stack[top - 1] = !stack[top - 1];
                }
            }
        }

        debug_assert_eq!(
            top, 1,
            "RPN stack should have exactly one value after evaluation"
        );
        stack[0]
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_region_compiles_and_evaluates_intersection() {
        // Region = inside sphere(r=2 @ origin) AND below plane z=0.
        let sphere = Arc::new(Surface::new_sphere(0.0, 0.0, 0.0, 2.0, None, None));
        let plane = Arc::new(Surface::new_plane(0.0, 0.0, 1.0, 0.0, None, None));
        let expr = RegionExpr::Intersection(
            Box::new(RegionExpr::Halfspace(HalfspaceType::Below(sphere))),
            Box::new(RegionExpr::Halfspace(HalfspaceType::Below(plane))),
        );
        let flat = FlatRegion::from_expr(&expr);

        // Inside the sphere and below z=0 -> contained.
        assert!(flat.contains((0.0, 0.0, -1.0)));
        // Inside the sphere but above z=0 -> excluded by the plane.
        assert!(!flat.contains((0.0, 0.0, 1.0)));
        // Below z=0 but outside the sphere -> excluded by the sphere.
        assert!(!flat.contains((5.0, 0.0, -1.0)));
        // A shared surface is interned once.
        assert!(flat.ops.len() >= 3);
    }

    fn below(radius: f64) -> RegionExpr {
        RegionExpr::Halfspace(HalfspaceType::Below(Arc::new(Surface::new_sphere(
            0.0, 0.0, 0.0, radius, None, None,
        ))))
    }

    fn rpn_depth(expr: &RegionExpr) -> usize {
        walk_rpn(expr, &mut |_| {})
    }

    /// Issue #510: `new & old` in a loop builds a right-deep tree, which
    /// overflowed the 64-slot stack when emitted left to right.
    #[test]
    fn deep_right_chain_needs_two_slots() {
        let mut expr = below(1.0);
        for i in 1..200 {
            expr = RegionExpr::Intersection(Box::new(below(i as f64 + 1.0)), Box::new(expr));
        }
        assert_eq!(rpn_depth(&expr), 2);
        let flat = FlatRegion::from_expr(&expr);
        assert!(flat.contains((0.0, 0.0, 0.0)));
        assert!(!flat.contains((0.0, 0.0, 1.5)));
        assert_eq!(
            flat.contains((0.0, 0.0, 0.5)),
            expr.evaluate_contains((0.0, 0.0, 0.5))
        );
    }

    #[test]
    fn deep_left_chain_and_complements_need_two_slots() {
        let mut expr = below(1.0);
        for i in 1..200 {
            let next = RegionExpr::Union(Box::new(expr), Box::new(below(i as f64 + 1.0)));
            expr = RegionExpr::Complement(Box::new(RegionExpr::Complement(Box::new(next))));
        }
        assert_eq!(rpn_depth(&expr), 2);
        let flat = FlatRegion::from_expr(&expr);
        for r in [0.5, 150.5, 250.0] {
            let p = (0.0, 0.0, r);
            assert_eq!(flat.contains(p), expr.evaluate_contains(p), "z = {r}");
        }
    }

    #[test]
    fn balanced_tree_depth_grows_with_log_of_leaves() {
        // A balanced tree of 2^k leaves needs k + 1 slots, the minimum.
        let mut level: Vec<RegionExpr> = (0..256).map(|i| below(i as f64 + 1.0)).collect();
        while level.len() > 1 {
            let mut next = Vec::new();
            let mut it = level.into_iter();
            while let (Some(a), Some(b)) = (it.next(), it.next()) {
                next.push(RegionExpr::Union(Box::new(a), Box::new(b)));
            }
            level = next;
        }
        assert_eq!(rpn_depth(&level[0]), 9);
    }

    /// Reordering commuted operands must not change what a region contains.
    #[test]
    fn reordered_program_matches_recursive_evaluation() {
        let shells: Vec<RegionExpr> = (1..=6).map(|i| below(i as f64)).collect();
        let not = |e: RegionExpr| RegionExpr::Complement(Box::new(e));
        let and = |a, b| RegionExpr::Intersection(Box::new(a), Box::new(b));
        let or = |a, b| RegionExpr::Union(Box::new(a), Box::new(b));
        // ((s2 & ~s1) | (s4 & ~(s3 | (s1 & s2)))) & ~(s6 & ~s5 & s4) style mix:
        // the operand needing the deeper stack sits on either side.
        let expr = and(
            or(
                and(shells[1].clone(), not(shells[0].clone())),
                and(
                    shells[3].clone(),
                    not(or(
                        shells[2].clone(),
                        and(shells[0].clone(), shells[1].clone()),
                    )),
                ),
            ),
            not(and(
                and(shells[5].clone(), not(shells[4].clone())),
                shells[3].clone(),
            )),
        );
        let flat = FlatRegion::from_expr(&expr);
        for i in 0..70 {
            let p = (0.0, 0.0, i as f64 * 0.1);
            assert_eq!(flat.contains(p), expr.evaluate_contains(p), "z = {}", p.2);
        }
    }
}
