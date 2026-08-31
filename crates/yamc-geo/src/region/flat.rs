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
    Above(u16),
    /// Evaluate surface at index, push `(evaluate(point) < 0.0)`.
    Below(u16),
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
/// All surface references are stored in a flat `Vec` indexed by the `u16` operands
/// in `Above`/`Below` instructions.
#[derive(Debug, Clone)]
pub struct FlatRegion {
    ops: Vec<RegionOp>,
    surfaces: Vec<Arc<Surface>>,
}

const MAX_STACK_DEPTH: usize = 64;

impl FlatRegion {
    /// Compile a `RegionExpr` tree into a flat RPN instruction sequence.
    pub fn from_expr(expr: &RegionExpr) -> Self {
        let mut ops = Vec::new();
        let mut surfaces: Vec<Arc<Surface>> = Vec::new();
        let mut surface_map: HashMap<*const Surface, u16> = HashMap::new();

        fn emit(
            expr: &RegionExpr,
            ops: &mut Vec<RegionOp>,
            surfaces: &mut Vec<Arc<Surface>>,
            surface_map: &mut HashMap<*const Surface, u16>,
        ) {
            match expr {
                RegionExpr::Halfspace(hs) => {
                    let (surf, above) = match hs {
                        HalfspaceType::Above(s) => (s, true),
                        HalfspaceType::Below(s) => (s, false),
                    };
                    let ptr = Arc::as_ptr(surf);
                    let idx = *surface_map.entry(ptr).or_insert_with(|| {
                        let idx = surfaces.len() as u16;
                        surfaces.push(surf.clone());
                        idx
                    });
                    ops.push(if above {
                        RegionOp::Above(idx)
                    } else {
                        RegionOp::Below(idx)
                    });
                }
                RegionExpr::Union(a, b) => {
                    emit(a, ops, surfaces, surface_map);
                    emit(b, ops, surfaces, surface_map);
                    ops.push(RegionOp::Or);
                }
                RegionExpr::Intersection(a, b) => {
                    emit(a, ops, surfaces, surface_map);
                    emit(b, ops, surfaces, surface_map);
                    ops.push(RegionOp::And);
                }
                RegionExpr::Complement(inner) => {
                    emit(inner, ops, surfaces, surface_map);
                    ops.push(RegionOp::Not);
                }
            }
        }

        emit(expr, &mut ops, &mut surfaces, &mut surface_map);
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
}
