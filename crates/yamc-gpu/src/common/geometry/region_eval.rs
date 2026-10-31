//! Point-in-region (CSG) evaluation on the GPU.
//!
//! The CPU finds a particle's cell by evaluating each cell's actual CSG
//! region -- the boolean tree of signed half-spaces (a surface's
//! negative/positive side, combined by intersection / union / complement),
//! compiled to a flat RPN program by `yamc_geo::region::FlatRegion`. The
//! AABB-only cell finding the GPU shipped with cannot disambiguate
//! nested / overlapping cells (concentric shells: the inner sphere's AABB
//! sits inside the shell's, and the shell's AABB also contains the inner
//! region, so multiple AABBs match a single point). This module ports the
//! CPU's region evaluation so the GPU resolves the same cell.
//!
//! # Region program layout (one flat `&[u32]` buffer)
//!
//! A single buffer encodes every cell's region, mirroring `FlatRegion`'s
//! RPN form but interned across the whole geometry:
//!
//! - **Header** -- the first `n_cells + 1` words are prefix offsets into
//!   the op stream. Cell `c`'s ops occupy op-indices
//!   `[program[c], program[c + 1])`.
//! - **Op stream** -- starts at word `n_cells + 1`. Op at op-index `k`
//!   lives at `program[(n_cells + 1) + k]`.
//!
//! Each op word packs an opcode in the top 4 bits and a surface index in
//! the low 28 bits: `word = (opcode << 28) | surf_idx`.
//!
//! | opcode | meaning | stack effect |
//! |--------|---------|--------------|
//! | 0 `ABOVE` | push `surface_evaluate(surf_idx) > 0` | +1 |
//! | 1 `BELOW` | push `surface_evaluate(surf_idx) < 0` | +1 |
//! | 2 `AND`   | pop two, push `a && b` | -1 |
//! | 3 `OR`    | pop two, push `a || b` | -1 |
//! | 4 `NOT`   | pop one, push `!a` | 0 |
//!
//! This matches `yamc_geo::region::flat::RegionOp` and its `contains`
//! evaluator one-for-one (same surface sense, same boolean order), so a
//! cell that the CPU says contains a point is the cell the GPU picks too.
//!
//! `surface_evaluate` is the implicit value `f(P)` of each surface --
//! exactly the `yamc_geo::Surface::evaluate` formulas, dispatched on the
//! same `SURFACE_*` discriminants and stride-10 `surface_params` layout the
//! boundary-distance kernel already uses. `f < 0` is inside (the `Below`
//! side), `f > 0` outside (`Above`), matching the CPU half-space sense.

use crate::common::geometry::boundary_distance::{
    SURFACE_CONE, SURFACE_CYLINDER, SURFACE_PLANE, SURFACE_QUADRIC, SURFACE_SPHERE, SURFACE_XTORUS,
    SURFACE_YTORUS, SURFACE_ZTORUS,
};
use cubecl::prelude::*;

/// Op-stack depth. Matches the CPU `FlatRegion`'s `MAX_STACK_DEPTH`; a
/// region tree deeper than this would need a bigger stack on both sides.
pub const REGION_STACK_DEPTH: usize = 64;

/// Opcode discriminants packed into the top 4 bits of each op word.
pub const REGION_OP_ABOVE: u32 = 0;
pub const REGION_OP_BELOW: u32 = 1;
pub const REGION_OP_AND: u32 = 2;
pub const REGION_OP_OR: u32 = 3;
pub const REGION_OP_NOT: u32 = 4;

/// Bit shift / mask splitting an op word into `(opcode, surf_idx)`.
pub const REGION_OP_SHIFT: u32 = 28;
pub const REGION_SURF_MASK: u32 = 0x0FFF_FFFF;

/// Implicit surface value `f(P)` for the surface at index `s` in the
/// shared `surface_params` (stride 10) / `surface_types` buffers. A
/// faithful port of `yamc_geo::Surface::evaluate`: `f < 0` is inside the
/// `Below` half-space, `f > 0` the `Above` half-space.
#[cube]
pub fn surface_evaluate(
    surface_types: &[u32],
    surface_params: &[f64],
    s: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> f64 {
    let stype = surface_types[s as usize];
    let base = (s * 10u32) as usize;
    let p0 = surface_params[base];
    let p1 = surface_params[base + 1];
    let p2 = surface_params[base + 2];
    let p3 = surface_params[base + 3];
    let p4 = surface_params[base + 4];
    let p5 = surface_params[base + 5];
    let p6 = surface_params[base + 6];
    let p7 = surface_params[base + 7];
    let p8 = surface_params[base + 8];
    let p9 = surface_params[base + 9];

    let mut value = 0.0_f64;

    if stype == SURFACE_SPHERE {
        // |P - C| - r
        let qx = px - p0;
        let qy = py - p1;
        let qz = pz - p2;
        value = (qx * qx + qy * qy + qz * qz).sqrt() - p3;
    }
    if stype == SURFACE_PLANE {
        // a·x + b·y + c·z - d  (n·P - d)
        value = p0 * px + p1 * py + p2 * pz - p3;
    }
    if stype == SURFACE_CYLINDER {
        // |q - (q·a)a| - r, with q = P - O, a the unit axis.
        let qx = px - p0;
        let qy = py - p1;
        let qz = pz - p2;
        let q_dot_a = qx * p4 + qy * p5 + qz * p6;
        let mx = qx - q_dot_a * p4;
        let my = qy - q_dot_a * p5;
        let mz = qz - q_dot_a * p6;
        value = (mx * mx + my * my + mz * mz).sqrt() - p3;
    }
    if stype == SURFACE_ZTORUS {
        // Axial coordinate z; transverse pair (x, y).
        value = torus_evaluate(px - p0, py - p1, pz - p2, p3, p4, p5);
    }
    if stype == SURFACE_XTORUS {
        // Axial coordinate x; transverse pair (y, z).
        value = torus_evaluate(py - p1, pz - p2, px - p0, p3, p4, p5);
    }
    if stype == SURFACE_YTORUS {
        // Axial coordinate y; transverse pair (x, z).
        value = torus_evaluate(px - p0, pz - p2, py - p1, p3, p4, p5);
    }
    if stype == SURFACE_QUADRIC {
        // a x² + b y² + c z² + d xy + e yz + f xz + g x + h y + j z + k
        value = p0 * px * px
            + p1 * py * py
            + p2 * pz * pz
            + p3 * px * py
            + p4 * py * pz
            + p5 * px * pz
            + p6 * px
            + p7 * py
            + p8 * pz
            + p9;
    }
    if stype == SURFACE_CONE {
        // (v² - s²) - tan²θ·s², with v = P - apex, s = v·axis.
        let vx = px - p0;
        let vy = py - p1;
        let vz = pz - p2;
        let s_ax = vx * p3 + vy * p4 + vz * p5;
        let v2 = vx * vx + vy * vy + vz * vz;
        value = (v2 - s_ax * s_ax) - p6 * s_ax * s_ax;
    }

    value
}

/// Torus implicit value `(ρ - a)²/c² + ax²/b² - 1` with `ρ = √(t1² + t2²)`.
/// Mirrors `yamc_geo::surface::torus::torus_evaluate`; the caller permutes
/// the transverse / axial coordinates per torus axis.
#[cube]
fn torus_evaluate(t1: f64, t2: f64, ax: f64, a: f64, b: f64, c: f64) -> f64 {
    let rho = (t1 * t1 + t2 * t2).sqrt();
    let r = rho - a;
    r * r / (c * c) + ax * ax / (b * b) - 1.0
}

/// Evaluate point-in-region for cell `cell` by running its RPN program.
/// Returns `true` when the point is inside the cell's CSG region -- the
/// exact predicate the CPU `FlatRegion::contains` computes.
#[cube]
pub fn region_contains(
    region_program: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    n_cells: u32,
    cell: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> bool {
    let header = n_cells + 1u32;
    let op_start = region_program[cell as usize];
    let op_end = region_program[(cell + 1u32) as usize];

    // Empty op-range = identity region (always contained). Used by the
    // AABB-only equivalence fixtures that pass a zero header so the cell
    // is selected on its bounding box alone.
    let mut result = true;

    // Boolean stack stored as 0/1 in a u32 array (cubecl handles the
    // dynamic push/pop indexing the same way the photon TTB stack does).
    let mut stack = Array::<u32>::new(REGION_STACK_DEPTH);
    let mut top = 0u32;

    let mut k = op_start;
    while k < op_end {
        let word = region_program[(header + k) as usize];
        let opcode = word >> REGION_OP_SHIFT;
        let surf_idx = word & REGION_SURF_MASK;

        if opcode == REGION_OP_ABOVE {
            let f = surface_evaluate(surface_types, surface_params, surf_idx, px, py, pz);
            let mut b = 0u32;
            if f > 0.0 {
                b = 1u32;
            }
            stack[top as usize] = b;
            top += 1u32;
        }
        if opcode == REGION_OP_BELOW {
            let f = surface_evaluate(surface_types, surface_params, surf_idx, px, py, pz);
            let mut b = 0u32;
            if f < 0.0 {
                b = 1u32;
            }
            stack[top as usize] = b;
            top += 1u32;
        }
        if opcode == REGION_OP_AND {
            top -= 1u32;
            let a = stack[top as usize];
            let mut r = 0u32;
            if a == 1u32 && stack[(top - 1u32) as usize] == 1u32 {
                r = 1u32;
            }
            stack[(top - 1u32) as usize] = r;
        }
        if opcode == REGION_OP_OR {
            top -= 1u32;
            let a = stack[top as usize];
            let mut r = 0u32;
            if a == 1u32 || stack[(top - 1u32) as usize] == 1u32 {
                r = 1u32;
            }
            stack[(top - 1u32) as usize] = r;
        }
        if opcode == REGION_OP_NOT {
            let mut r = 1u32;
            if stack[(top - 1u32) as usize] == 1u32 {
                r = 0u32;
            }
            stack[(top - 1u32) as usize] = r;
        }

        k += 1u32;
    }

    if op_end > op_start {
        result = stack[0] == 1u32;
    }
    result
}

// ----------------------- CPU mirrors -----------------------

/// CPU twin of [`surface_evaluate`]. Same formulas, plain Rust, so the
/// rayon / sequential CPU transport paths resolve the identical cell.
pub fn surface_evaluate_cpu(
    surface_types: &[u32],
    surface_params: &[f64],
    s: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> f64 {
    let stype = surface_types[s as usize];
    let base = (s as usize) * 10;
    let p0 = surface_params[base];
    let p1 = surface_params[base + 1];
    let p2 = surface_params[base + 2];
    let p3 = surface_params[base + 3];
    let p4 = surface_params[base + 4];
    let p5 = surface_params[base + 5];
    let p6 = surface_params[base + 6];
    let p7 = surface_params[base + 7];
    let p8 = surface_params[base + 8];
    let p9 = surface_params[base + 9];

    if stype == SURFACE_SPHERE {
        let qx = px - p0;
        let qy = py - p1;
        let qz = pz - p2;
        (qx * qx + qy * qy + qz * qz).sqrt() - p3
    } else if stype == SURFACE_PLANE {
        p0 * px + p1 * py + p2 * pz - p3
    } else if stype == SURFACE_CYLINDER {
        let qx = px - p0;
        let qy = py - p1;
        let qz = pz - p2;
        let q_dot_a = qx * p4 + qy * p5 + qz * p6;
        let mx = qx - q_dot_a * p4;
        let my = qy - q_dot_a * p5;
        let mz = qz - q_dot_a * p6;
        (mx * mx + my * my + mz * mz).sqrt() - p3
    } else if stype == SURFACE_ZTORUS {
        torus_evaluate_cpu(px - p0, py - p1, pz - p2, p3, p4, p5)
    } else if stype == SURFACE_XTORUS {
        torus_evaluate_cpu(py - p1, pz - p2, px - p0, p3, p4, p5)
    } else if stype == SURFACE_YTORUS {
        torus_evaluate_cpu(px - p0, pz - p2, py - p1, p3, p4, p5)
    } else if stype == SURFACE_QUADRIC {
        p0 * px * px
            + p1 * py * py
            + p2 * pz * pz
            + p3 * px * py
            + p4 * py * pz
            + p5 * px * pz
            + p6 * px
            + p7 * py
            + p8 * pz
            + p9
    } else if stype == SURFACE_CONE {
        let vx = px - p0;
        let vy = py - p1;
        let vz = pz - p2;
        let s_ax = vx * p3 + vy * p4 + vz * p5;
        let v2 = vx * vx + vy * vy + vz * vz;
        (v2 - s_ax * s_ax) - p6 * s_ax * s_ax
    } else {
        0.0
    }
}

fn torus_evaluate_cpu(t1: f64, t2: f64, ax: f64, a: f64, b: f64, c: f64) -> f64 {
    let rho = (t1 * t1 + t2 * t2).sqrt();
    let r = rho - a;
    r * r / (c * c) + ax * ax / (b * b) - 1.0
}

/// CPU twin of [`region_contains`]. Same RPN program, same boolean order.
#[allow(clippy::too_many_arguments)]
pub fn region_contains_cpu(
    region_program: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
    n_cells: u32,
    cell: u32,
    px: f64,
    py: f64,
    pz: f64,
) -> bool {
    let header = (n_cells + 1) as usize;
    let op_start = region_program[cell as usize] as usize;
    let op_end = region_program[(cell + 1) as usize] as usize;

    // Empty op-range = identity region (always contained), matching the
    // GPU `region_contains` fast path for AABB-only fixtures.
    if op_start == op_end {
        return true;
    }

    let mut stack = [false; REGION_STACK_DEPTH];
    let mut top = 0usize;

    for k in op_start..op_end {
        let word = region_program[header + k];
        let opcode = word >> REGION_OP_SHIFT;
        let surf_idx = word & REGION_SURF_MASK;

        if opcode == REGION_OP_ABOVE {
            let f = surface_evaluate_cpu(surface_types, surface_params, surf_idx, px, py, pz);
            stack[top] = f > 0.0;
            top += 1;
        } else if opcode == REGION_OP_BELOW {
            let f = surface_evaluate_cpu(surface_types, surface_params, surf_idx, px, py, pz);
            stack[top] = f < 0.0;
            top += 1;
        } else if opcode == REGION_OP_AND {
            top -= 1;
            stack[top - 1] = stack[top - 1] && stack[top];
        } else if opcode == REGION_OP_OR {
            top -= 1;
            stack[top - 1] = stack[top - 1] || stack[top];
        } else if opcode == REGION_OP_NOT {
            stack[top - 1] = !stack[top - 1];
        }
    }

    stack[0]
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::geometry::boundary_distance::SURFACE_PARAM_STRIDE;

    fn above(idx: u32) -> u32 {
        (REGION_OP_ABOVE << REGION_OP_SHIFT) | idx
    }
    fn below(idx: u32) -> u32 {
        (REGION_OP_BELOW << REGION_OP_SHIFT) | idx
    }
    fn and() -> u32 {
        REGION_OP_AND << REGION_OP_SHIFT
    }
    fn or() -> u32 {
        REGION_OP_OR << REGION_OP_SHIFT
    }
    fn not() -> u32 {
        REGION_OP_NOT << REGION_OP_SHIFT
    }

    /// Pad a list of sphere `(cx, cy, cz, r)` into the stride-10 params layout.
    fn spheres(specs: &[[f64; 4]]) -> (Vec<u32>, Vec<f64>) {
        let mut types = Vec::new();
        let mut params = Vec::new();
        for s in specs {
            types.push(SURFACE_SPHERE);
            params.extend_from_slice(s);
            params.extend(std::iter::repeat_n(0.0, SURFACE_PARAM_STRIDE - 4));
        }
        (types, params)
    }

    /// Three concentric spheres (r = 1, 2, 3) -> three nested shell cells:
    ///   cell 0 (core): below s0
    ///   cell 1 (mid):  above s0 AND below s1
    ///   cell 2 (out):  above s1 AND below s2
    /// Every cell's AABB contains the origin, so AABB-only finding can't tell
    /// them apart; the region test must.
    #[test]
    fn concentric_shells_resolve_by_region() {
        let (types, params) = spheres(&[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 3.0],
        ]);
        let n_cells = 3u32;
        // Header (4 offsets) + ops. Core: [below 0]. Mid: [above 0, below 1, and].
        // Out: [above 1, below 2, and].
        let ops = vec![
            below(0),
            above(0),
            below(1),
            and(),
            above(1),
            below(2),
            and(),
        ];
        let offsets = vec![0u32, 1, 4, 7];
        let mut program = offsets.clone();
        program.extend_from_slice(&ops);

        let contains = |cell: u32, p: [f64; 3]| {
            region_contains_cpu(&program, &types, &params, n_cells, cell, p[0], p[1], p[2])
        };

        // A point at radius 0.5 is in the core only.
        assert!(contains(0, [0.5, 0.0, 0.0]));
        assert!(!contains(1, [0.5, 0.0, 0.0]));
        assert!(!contains(2, [0.5, 0.0, 0.0]));
        // Radius 1.5 -> middle shell only.
        assert!(!contains(0, [1.5, 0.0, 0.0]));
        assert!(contains(1, [1.5, 0.0, 0.0]));
        assert!(!contains(2, [1.5, 0.0, 0.0]));
        // Radius 2.5 -> outer shell only.
        assert!(!contains(0, [2.5, 0.0, 0.0]));
        assert!(!contains(1, [2.5, 0.0, 0.0]));
        assert!(contains(2, [2.5, 0.0, 0.0]));
        // Radius 3.5 -> outside every cell.
        assert!(!contains(0, [3.5, 0.0, 0.0]));
        assert!(!contains(1, [3.5, 0.0, 0.0]));
        assert!(!contains(2, [3.5, 0.0, 0.0]));
    }

    /// Empty op-range = identity region (always contained). The AABB-only
    /// fixtures rely on this.
    #[test]
    fn empty_program_is_identity() {
        let program = vec![0u32, 0u32]; // 1 cell, zero ops.
        let types = vec![SURFACE_SPHERE];
        let params = vec![0.0; SURFACE_PARAM_STRIDE];
        assert!(region_contains_cpu(
            &program, &types, &params, 1, 0, 100.0, 0.0, 0.0
        ));
    }

    /// Union of two disjoint spheres + complement, checked against the
    /// boolean truth table so the `Or` / `Not` opcodes are exercised.
    #[test]
    fn union_and_complement() {
        let (types, params) = spheres(&[[-3.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]);
        // cell 0 = inside s0 OR inside s1 = [below 0, below 1, or]
        // cell 1 = complement of cell 0 = [below 0, below 1, or, not]
        let offsets = vec![0u32, 3, 7];
        let ops = vec![below(0), below(1), or(), below(0), below(1), or(), not()];
        let mut program = offsets.clone();
        program.extend_from_slice(&ops);
        let c = |cell: u32, p: [f64; 3]| {
            region_contains_cpu(&program, &types, &params, 2, cell, p[0], p[1], p[2])
        };
        assert!(c(0, [-3.0, 0.0, 0.0])); // in first sphere
        assert!(c(0, [3.0, 0.0, 0.0])); // in second sphere
        assert!(!c(0, [0.0, 0.0, 0.0])); // between -> not in union
        assert!(!c(1, [-3.0, 0.0, 0.0])); // complement excludes the union
        assert!(c(1, [0.0, 0.0, 0.0])); // complement includes the gap
    }
}
