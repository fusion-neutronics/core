//! Find which cell each particle is in (Phase C foundation).
//!
//! For multi-cell transport the kernel needs to know the cell index
//! for every particle, so that XS lookups and tally bin selection can
//! be per-cell. This kernel handles the lookup as a linear scan over
//! per-cell axis-aligned bounding boxes -- correct for any geometry
//! whose cells are non-overlapping AABBs (or where the CPU side has
//! already collapsed CSG cells into their AABB approximation).
//!
//! # First-cut limits
//!
//! - **Linear scan, not BVH traversal.** O(N) per particle in the
//!   number of cells. For test geometries (≤16 cells) this is fine;
//!   for production with thousands of cells the BVH already exists in
//!   `yamc-geo` and a stackful traversal kernel is a follow-up. The
//!   POD-friendly `BvhNode` derives in `yamc-geo` make the upload
//!   trivial; the algorithmic port is the work.
//! - **AABB-only containment.** The CPU `Bvh::find` calls a per-cell
//!   `contains_fn` that runs the cell's CSG predicate. We don't have
//!   CSG-on-GPU; the kernel relies on the AABB being the full cell
//!   shape (or being a superset that happens to be unique per
//!   particle position). Works for box-and-shell geometries; not for
//!   overlapping/CSG-heavy cells.
//! - **First match wins.** If two cell AABBs both contain the point
//!   (overlap), the kernel returns the lower index. CPU yamc would
//!   pick by CSG predicate; here it's caller's responsibility to
//!   give non-overlapping AABBs.
//!
//! # Layout
//!
//! `cell_aabbs` is stride-6 packed: `[min_x0, min_y0, min_z0, max_x0,
//! max_y0, max_z0, min_x1, …]`. Output is one `u32` per particle --
//! the cell index, or `CELL_NOT_FOUND = u32::MAX` if no cell contains
//! the point.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Sentinel returned when no cell AABB contains the particle's position.
pub const CELL_NOT_FOUND: u32 = u32::MAX;

// Re-exported as `MISS_SENTINEL` from the parent module would clash
// with `sphere_distance::MISS_SENTINEL` (which is an `f64` for ray
// misses), so we keep this u32 name distinct.

#[cube(launch_unchecked)]
fn cell_finding_kernel(positions: &[f64], cell_aabbs: &[f64], out_cell: &mut [u32]) {
    if ABSOLUTE_POS >= out_cell.len() {
        terminate!();
    }

    let i3 = ABSOLUTE_POS * 3;
    let px = positions[i3];
    let py = positions[i3 + 1];
    let pz = positions[i3 + 2];

    let n_cells: u32 = (cell_aabbs.len() / 6) as u32;
    let mut found = 4_294_967_295u32; // u32::MAX
    let mut i = 0u32;
    while i < n_cells && found == 4_294_967_295u32 {
        let base: u32 = i * 6u32;
        let min_x = cell_aabbs[base as usize];
        let min_y = cell_aabbs[(base + 1u32) as usize];
        let min_z = cell_aabbs[(base + 2u32) as usize];
        let max_x = cell_aabbs[(base + 3u32) as usize];
        let max_y = cell_aabbs[(base + 4u32) as usize];
        let max_z = cell_aabbs[(base + 5u32) as usize];
        if px >= min_x && px <= max_x && py >= min_y && py <= max_y && pz >= min_z && pz <= max_z {
            found = i;
        }
        i += 1u32;
    }
    out_cell[ABSOLUTE_POS] = found;
}

/// Run the cell-finding kernel. `positions` is stride-3 packed,
/// `cell_aabbs` is stride-6 (min then max). Returns one cell index
/// per particle, with `CELL_NOT_FOUND` for any position outside all
/// cell AABBs.
pub fn run_cell_finding(ctx: &GpuContext, positions: &[f64], cell_aabbs: &[f64]) -> Vec<u32> {
    assert!(
        positions.len().is_multiple_of(3),
        "positions length must be a multiple of 3"
    );
    assert!(
        cell_aabbs.len().is_multiple_of(6),
        "cell_aabbs length must be a multiple of 6"
    );
    let n = positions.len() / 3;

    let client = ctx.client();
    let positions_h = client.create_from_slice(bytemuck::cast_slice(positions));
    let aabbs_h = client.create_from_slice(bytemuck::cast_slice(cell_aabbs));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        cell_finding_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(positions_h, positions.len()),
            BufferArg::from_raw_parts(aabbs_h, cell_aabbs.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

/// CPU equivalent of `run_cell_finding`. Same algorithm.
pub fn run_cell_finding_cpu(positions: &[f64], cell_aabbs: &[f64]) -> Vec<u32> {
    assert!(positions.len().is_multiple_of(3));
    assert!(cell_aabbs.len().is_multiple_of(6));
    let n = positions.len() / 3;
    let n_cells = cell_aabbs.len() / 6;
    let mut out = Vec::with_capacity(n);
    for p in 0..n {
        let i3 = p * 3;
        let px = positions[i3];
        let py = positions[i3 + 1];
        let pz = positions[i3 + 2];
        let mut found = CELL_NOT_FOUND;
        for c in 0..n_cells {
            let base = c * 6;
            if px >= cell_aabbs[base]
                && px <= cell_aabbs[base + 3]
                && py >= cell_aabbs[base + 1]
                && py <= cell_aabbs[base + 4]
                && pz >= cell_aabbs[base + 2]
                && pz <= cell_aabbs[base + 5]
            {
                found = c as u32;
                break;
            }
        }
        out.push(found);
    }
    out
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Build 4 non-overlapping unit cubes at known positions, send a
    /// point per cube + a few misses; expect each in-cube point to
    /// land in its cube and miss points to return the sentinel.
    #[test]
    fn finds_cells_and_reports_misses() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Cells at (0,0,0)-(1,1,1), (2,0,0)-(3,1,1), (0,2,0)-(1,3,1),
        // (2,2,2)-(3,3,3).
        let cell_aabbs: Vec<f64> = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0
            2.0, 0.0, 0.0, 3.0, 1.0, 1.0, // cell 1
            0.0, 2.0, 0.0, 1.0, 3.0, 1.0, // cell 2
            2.0, 2.0, 2.0, 3.0, 3.0, 3.0, // cell 3
        ];
        let positions: Vec<f64> = vec![
            0.5, 0.5, 0.5, // in cell 0
            2.5, 0.5, 0.5, // in cell 1
            0.5, 2.5, 0.5, // in cell 2
            2.5, 2.5, 2.5, // in cell 3
            5.0, 5.0, 5.0, // miss
            -1.0, 0.5, 0.5, // miss
            1.5, 1.5, 1.5, // miss (between cells)
        ];

        let gpu = run_cell_finding(&ctx, &positions, &cell_aabbs);
        let cpu = run_cell_finding_cpu(&positions, &cell_aabbs);

        let expected: Vec<u32> = vec![0, 1, 2, 3, CELL_NOT_FOUND, CELL_NOT_FOUND, CELL_NOT_FOUND];
        assert_eq!(gpu, expected, "GPU output");
        assert_eq!(cpu, expected, "CPU output");
        assert_eq!(gpu, cpu, "GPU and CPU agree");
    }

    /// Boundary points (exactly on a cell face) must be assigned
    /// deterministically. With `>=` and `<=` comparisons both bounds
    /// are inclusive, so a point exactly at (1,0.5,0.5) (between
    /// cells 0 and 1 if they touched) lands in the lower-index cell.
    #[test]
    fn deterministic_boundary_handling() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };
        // Two touching cubes sharing the x=1 face.
        let cell_aabbs = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // cell 0
            1.0, 0.0, 0.0, 2.0, 1.0, 1.0, // cell 1
        ];
        // Exactly on the shared face.
        let positions = vec![1.0, 0.5, 0.5];
        let gpu = run_cell_finding(&ctx, &positions, &cell_aabbs);
        // First-match-wins → cell 0.
        assert_eq!(gpu, vec![0u32]);
    }

    /// Larger workload: 1000 random positions, 16 cells. GPU and CPU
    /// must agree on every assignment.
    #[test]
    fn gpu_matches_cpu_on_random_positions() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };

        // 16 unit cells in a 4×4×1 grid.
        let mut cell_aabbs = Vec::with_capacity(16 * 6);
        for ix in 0..4 {
            for iy in 0..4 {
                let x0 = ix as f64;
                let y0 = iy as f64;
                cell_aabbs.extend_from_slice(&[x0, y0, 0.0, x0 + 1.0, y0 + 1.0, 1.0]);
            }
        }

        // Pseudo-random positions, mostly inside the 0..4 × 0..4 ×
        // 0..1 box but with some that miss.
        let mut positions = Vec::with_capacity(1000 * 3);
        for i in 0..1000 {
            let f = i as f64;
            let px = ((f * 0.137) % 5.0) - 0.5;
            let py = ((f * 0.241) % 5.0) - 0.5;
            let pz = ((f * 0.197) % 1.5) - 0.25;
            positions.extend_from_slice(&[px, py, pz]);
        }

        let gpu = run_cell_finding(&ctx, &positions, &cell_aabbs);
        let cpu = run_cell_finding_cpu(&positions, &cell_aabbs);
        assert_eq!(gpu, cpu);
    }
}
