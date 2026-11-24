//! Which surfaces bound each cell, for the boundary-distance step.
//!
//! A flight can only leave its cell through a surface the cell's region is
//! built from, so the transport kernels solve for those surfaces alone, as the
//! CPU's `Cell::closest_surface` does. Solving every surface in the model would
//! cost a distance solve per surface per step, and would stop the particle at
//! surfaces that do not bound its cell (a vacuum one among them would end the
//! history there).
//!
//! The lists are read off the region program
//! ([`super::region_eval`]), so they name exactly the surfaces the cell-finding
//! step tests.
//!
//! # Layout
//!
//! The same shape as the region program: `n_cells + 1` prefix offsets, then
//! the surface indices. Cell `c`'s surfaces are at
//! `lists[n_cells + 1 + k]` for `k` in `[lists[c], lists[c + 1])`, in
//! ascending order so that a tie between two surfaces goes to the lower index,
//! as the scan over every surface did.
//!
//! A cell with an empty program is the identity region (the AABB-only test
//! fixtures): nothing says which surfaces bound it, so it lists all of them.

use super::region_eval::{REGION_OP_ABOVE, REGION_OP_BELOW, REGION_OP_SHIFT, REGION_SURF_MASK};

/// Build the per-cell surface lists from a region program over `n_cells`
/// cells and `n_surfaces` surfaces. See the module docs for the layout.
pub fn cell_surface_lists(region_program: &[u32], n_cells: usize, n_surfaces: usize) -> Vec<u32> {
    let header = n_cells + 1;
    let mut offsets = Vec::with_capacity(header);
    let mut indices: Vec<u32> = Vec::new();
    let mut seen = vec![false; n_surfaces];
    for cell in 0..n_cells {
        offsets.push(indices.len() as u32);
        let op_start = region_program[cell] as usize;
        let op_end = region_program[cell + 1] as usize;
        let start = indices.len();
        if op_start == op_end {
            indices.extend(0..n_surfaces as u32);
            continue;
        }
        for &word in &region_program[header + op_start..header + op_end] {
            let opcode = word >> REGION_OP_SHIFT;
            if opcode == REGION_OP_ABOVE || opcode == REGION_OP_BELOW {
                let s = word & REGION_SURF_MASK;
                if !seen[s as usize] {
                    seen[s as usize] = true;
                    indices.push(s);
                }
            }
        }
        indices[start..].sort_unstable();
        for &s in &indices[start..] {
            seen[s as usize] = false;
        }
    }
    offsets.push(indices.len() as u32);
    let mut lists = offsets;
    lists.extend_from_slice(&indices);
    lists
}

/// Cell `cell`'s surfaces in a list built by [`cell_surface_lists`].
pub fn cell_surfaces(lists: &[u32], n_cells: usize, cell: usize) -> &[u32] {
    let header = n_cells + 1;
    &lists[header + lists[cell] as usize..header + lists[cell + 1] as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::geometry::region_eval::{REGION_OP_AND, REGION_OP_NOT, REGION_OP_OR};

    fn op(opcode: u32, s: u32) -> u32 {
        (opcode << REGION_OP_SHIFT) | s
    }

    #[test]
    fn each_cell_lists_its_own_surfaces_once_in_order() {
        // Cell 0: below 3 and above 1. Cell 1: (above 3 or below 0), not above 3.
        // Cell 2: identity region.
        let ops = [
            op(REGION_OP_BELOW, 3),
            op(REGION_OP_ABOVE, 1),
            op(REGION_OP_AND, 0),
            op(REGION_OP_ABOVE, 3),
            op(REGION_OP_BELOW, 0),
            op(REGION_OP_OR, 0),
            op(REGION_OP_ABOVE, 3),
            op(REGION_OP_NOT, 0),
            op(REGION_OP_AND, 0),
        ];
        let mut program = vec![0, 3, 9, 9];
        program.extend_from_slice(&ops);
        let lists = cell_surface_lists(&program, 3, 5);
        assert_eq!(cell_surfaces(&lists, 3, 0), &[1, 3]);
        assert_eq!(cell_surfaces(&lists, 3, 1), &[0, 3]);
        assert_eq!(cell_surfaces(&lists, 3, 2), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn no_cells_gives_a_lone_offset() {
        assert_eq!(cell_surface_lists(&[0], 0, 0), vec![0]);
    }
}
