//! BVH-based cell-finding kernel.
//!
//! Replaces the linear AABB scan from `cell_finding.rs` with a
//! stackless BVH traversal. For geometries with thousands of cells
//! the BVH is asymptotically `O(log N)` per particle instead of
//! `O(N)`, while remaining a single launchable kernel with no
//! per-thread local memory.
//!
//! # Stackless rope encoding
//!
//! Standard BVH descent uses an explicit stack: when a node's AABB
//! contains the point, push both children; when it doesn't, do
//! nothing. cubecl's per-thread local memory story is awkward
//! enough that we sidestep it: we pre-compute, for every node `i`,
//! `escape_idx[i]` -- the depth-first index immediately AFTER the
//! whole subtree rooted at `i`. With that stored alongside each
//! node, the traversal collapses to a single pointer chase:
//!
//! ```text
//! let mut i = 0;
//! while i < n_nodes {
//!     if aabb_contains(node[i], point) {
//!         if leaf(node[i]) {
//!             // test primitives, optionally return
//!         }
//!         i += 1;            // descend left child (or skip leaf)
//!     } else {
//!         i = escape[i];     // skip the whole subtree
//!     }
//! }
//! ```
//!
//! Computing `escape_idx`: for each internal node `i` with right
//! child at index `R`, the left subtree ends at `R` and the right
//! subtree ends at `escape_idx[i]`. So `escape_idx[i + 1] = R` and
//! `escape_idx[R] = escape_idx[i]`. Top-down DFS pass, O(N).
//!
//! # Layout
//!
//! - `bvh_aabbs`: stride-6 packed, one `[min_x, min_y, min_z, max_x,
//!   max_y, max_z]` per node.
//! - `bvh_meta`: stride-3 u32 per node:
//!   `[right_or_first_prim, n_prims, escape_idx]`.
//! - `prim_indices`: leaf payloads (primitive indices, packed).
//! - `unbounded_indices`: primitives whose AABB was non-finite --
//!   linear post-traversal fallback.
//! - `cell_aabbs`: stride-6 per cell, used for the AABB-contains
//!   test on each leaf primitive.
//!
//! # Containment
//!
//! - The standalone `bvh_cell_finding_kernel` / `run_bvh_cell_finding*`
//!   helpers below do **AABB-only** containment (same as `cell_finding.rs`):
//!   they expect non-overlapping cell AABBs and exist for the standalone
//!   cell-finding diagnostics / benches.
//! - The transport path uses [`bvh_find_cell_at_point`], which confirms
//!   each AABB candidate with the cell's actual CSG region predicate
//!   ([`crate::common::geometry::region_eval::region_contains_cpu`]) -- so
//!   nested / overlapping cells (concentric shells) resolve to the cell
//!   that genuinely contains the point, matching the CPU. The GPU kernel's
//!   inlined cell-find (in `kernel.rs`) does the same with the `#[cube]`
//!   `region_contains`.
//! - **First-match-wins** within a leaf and across leaves visited in DFS
//!   order. With the region test this is the unique containing cell; for
//!   the AABB-only helpers it's the first AABB hit.

use crate::common::geometry::cell_finding::CELL_NOT_FOUND;
use crate::common::geometry::region_eval::region_contains_cpu;
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;
use yamc_geo::Bvh;

#[cube(launch_unchecked)]
fn bvh_cell_finding_kernel(
    positions: &[f64],
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    prim_indices: &[u32],
    unbounded_indices: &[u32],
    cell_aabbs: &[f64],
    out_cell: &mut [u32],
) {
    if ABSOLUTE_POS >= out_cell.len() {
        terminate!();
    }

    let i3 = ABSOLUTE_POS * 3;
    let px = positions[i3];
    let py = positions[i3 + 1];
    let pz = positions[i3 + 2];

    let n_nodes: u32 = (bvh_aabbs.len() / 6) as u32;
    let n_unbounded: u32 = unbounded_indices.len() as u32;

    let mut found = 4_294_967_295u32;
    let mut i = 0u32;
    while i < n_nodes && found == 4_294_967_295u32 {
        let aabb_base: u32 = i * 6u32;
        let min_x = bvh_aabbs[aabb_base as usize];
        let min_y = bvh_aabbs[(aabb_base + 1u32) as usize];
        let min_z = bvh_aabbs[(aabb_base + 2u32) as usize];
        let max_x = bvh_aabbs[(aabb_base + 3u32) as usize];
        let max_y = bvh_aabbs[(aabb_base + 4u32) as usize];
        let max_z = bvh_aabbs[(aabb_base + 5u32) as usize];
        let in_aabb =
            px >= min_x && px <= max_x && py >= min_y && py <= max_y && pz >= min_z && pz <= max_z;

        let meta_base: u32 = i * 3u32;
        let r_or_f = bvh_meta[meta_base as usize];
        let n_prims = bvh_meta[(meta_base + 1u32) as usize];
        let escape = bvh_meta[(meta_base + 2u32) as usize];

        if in_aabb {
            if n_prims > 0u32 {
                // Leaf: scan primitives, AABB-contains each.
                let mut p = 0u32;
                while p < n_prims && found == 4_294_967_295u32 {
                    let prim_idx = prim_indices[(r_or_f + p) as usize];
                    let cb: u32 = prim_idx * 6u32;
                    let cmin_x = cell_aabbs[cb as usize];
                    let cmin_y = cell_aabbs[(cb + 1u32) as usize];
                    let cmin_z = cell_aabbs[(cb + 2u32) as usize];
                    let cmax_x = cell_aabbs[(cb + 3u32) as usize];
                    let cmax_y = cell_aabbs[(cb + 4u32) as usize];
                    let cmax_z = cell_aabbs[(cb + 5u32) as usize];
                    if px >= cmin_x
                        && px <= cmax_x
                        && py >= cmin_y
                        && py <= cmax_y
                        && pz >= cmin_z
                        && pz <= cmax_z
                    {
                        found = prim_idx;
                    }
                    p += 1u32;
                }
            }
            i += 1u32;
        } else {
            i = escape;
        }
    }

    // Unbounded fallback: primitives the BVH couldn't index.
    if found == 4_294_967_295u32 {
        let mut u = 0u32;
        while u < n_unbounded && found == 4_294_967_295u32 {
            let prim_idx = unbounded_indices[u as usize];
            let cb: u32 = prim_idx * 6u32;
            let cmin_x = cell_aabbs[cb as usize];
            let cmin_y = cell_aabbs[(cb + 1u32) as usize];
            let cmin_z = cell_aabbs[(cb + 2u32) as usize];
            let cmax_x = cell_aabbs[(cb + 3u32) as usize];
            let cmax_y = cell_aabbs[(cb + 4u32) as usize];
            let cmax_z = cell_aabbs[(cb + 5u32) as usize];
            if px >= cmin_x
                && px <= cmax_x
                && py >= cmin_y
                && py <= cmax_y
                && pz >= cmin_z
                && pz <= cmax_z
            {
                found = prim_idx;
            }
            u += 1u32;
        }
    }

    out_cell[ABSOLUTE_POS] = found;
}

/// Build a BVH directly from a stride-6 packed cell-AABB slice and
/// flatten it. Convenience wrapper for callers (tests, bench, the
/// multi-cell transport runner) that already have AABBs in the
/// kernel's expected layout.
pub fn build_and_flatten_bvh(cell_aabbs: &[f64]) -> (Vec<f64>, Vec<u32>, Vec<u32>, Vec<u32>) {
    use yamc_geo::BoundingBox;
    let n_cells = cell_aabbs.len() / 6;
    let mut items: Vec<(u32, BoundingBox)> = Vec::with_capacity(n_cells);
    for (i, chunk) in cell_aabbs.as_chunks::<6>().0.iter().enumerate() {
        items.push((
            i as u32,
            BoundingBox::new(
                [chunk[0], chunk[1], chunk[2]],
                [chunk[3], chunk[4], chunk[5]],
            ),
        ));
    }
    let bvh = Bvh::build(&items);
    flatten_bvh(&bvh)
}

/// Flatten a `yamc_geo::Bvh` into the four arrays the GPU kernel
/// expects. Builds the rope-encoded `escape_idx` table on the way.
pub fn flatten_bvh(bvh: &Bvh) -> (Vec<f64>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = bvh.nodes.len();
    let mut aabbs: Vec<f64> = Vec::with_capacity(6 * n);
    let mut meta: Vec<u32> = Vec::with_capacity(3 * n);

    // Compute escape indices top-down: for each internal node, the
    // left child is at `i+1` and ends at the right child index R;
    // the right child ends at the parent's escape.
    let mut escape: Vec<u32> = vec![0u32; n];
    if n > 0 {
        escape[0] = n as u32;
        for (i, node) in bvh.nodes.iter().enumerate() {
            if node.n_prims == 0 {
                let r = node.right_or_first_prim as usize;
                if i + 1 < n {
                    escape[i + 1] = r as u32;
                }
                if r < n {
                    escape[r] = escape[i];
                }
            }
        }
    }

    for (i, node) in bvh.nodes.iter().enumerate() {
        aabbs.extend_from_slice(&node.aabb_min);
        aabbs.extend_from_slice(&node.aabb_max);
        meta.push(node.right_or_first_prim);
        meta.push(node.n_prims);
        meta.push(escape[i]);
    }

    (aabbs, meta, bvh.prim_indices.clone(), bvh.unbounded.clone())
}

/// Run BVH cell-finding on the GPU.
///
/// `cell_aabbs` is stride-6 per cell, parallel to the indices stored
/// in the BVH's `prim_indices` and `unbounded`. Returns one cell
/// index per particle, or `CELL_NOT_FOUND` if the point is outside
/// every cell's AABB.
pub fn run_bvh_cell_finding(
    ctx: &GpuContext,
    positions: &[f64],
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    prim_indices: &[u32],
    unbounded_indices: &[u32],
    cell_aabbs: &[f64],
) -> Vec<u32> {
    assert!(positions.len().is_multiple_of(3));
    assert!(bvh_aabbs.len().is_multiple_of(6));
    assert!(bvh_meta.len().is_multiple_of(3));
    assert!(cell_aabbs.len().is_multiple_of(6));
    let n = positions.len() / 3;

    let client = ctx.client();
    let pos_h = client.create_from_slice(bytemuck::cast_slice(positions));
    let aabbs_h = client.create_from_slice(bytemuck::cast_slice(bvh_aabbs));
    let meta_h = client.create_from_slice(bytemuck::cast_slice(bvh_meta));
    // cubecl rejects zero-sized buffers; pad a single sentinel u32
    // for empty prim/unbounded lists. The kernel guards on
    // `n_prims > 0` and `n_unbounded > 0` so the padding is never
    // actually read.
    let prim_data: &[u32] = if prim_indices.is_empty() {
        &[0]
    } else {
        prim_indices
    };
    let unbounded_data: &[u32] = if unbounded_indices.is_empty() {
        &[0]
    } else {
        unbounded_indices
    };
    let prim_h = client.create_from_slice(bytemuck::cast_slice(prim_data));
    let unb_h = client.create_from_slice(bytemuck::cast_slice(unbounded_data));
    let cell_h = client.create_from_slice(bytemuck::cast_slice(cell_aabbs));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);

    unsafe {
        bvh_cell_finding_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(pos_h, positions.len()),
            BufferArg::from_raw_parts(aabbs_h, bvh_aabbs.len()),
            BufferArg::from_raw_parts(meta_h, bvh_meta.len()),
            BufferArg::from_raw_parts(prim_h, prim_data.len()),
            BufferArg::from_raw_parts(unb_h, unbounded_data.len()),
            BufferArg::from_raw_parts(cell_h, cell_aabbs.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}

/// Single-point BVH lookup, used by the CPU mirrors of the
/// multi-cell transport runner so the cell-finding logic isn't
/// duplicated across the sequential and rayon paths.
///
/// The AABB test is a cheap pre-filter; the actual cell is confirmed by
/// evaluating the candidate cell's CSG region program ([`region_contains_cpu`]),
/// so nested / overlapping cells resolve exactly as on the CPU transport
/// and as the GPU kernel does. `region_program` / `surface_types` /
/// `surface_params` are the flat geometry buffers; `n_cells` is the cell
/// count (`cell_aabbs.len() / 6`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn bvh_find_cell_at_point(
    px: f64,
    py: f64,
    pz: f64,
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    prim_indices: &[u32],
    unbounded_indices: &[u32],
    cell_aabbs: &[f64],
    region_program: &[u32],
    surface_types: &[u32],
    surface_params: &[f64],
) -> u32 {
    let n_nodes = bvh_aabbs.len() / 6;
    let n_cells = (cell_aabbs.len() / 6) as u32;
    let in_cell = |prim_idx: u32| -> bool {
        let cb = (prim_idx as usize) * 6;
        px >= cell_aabbs[cb]
            && px <= cell_aabbs[cb + 3]
            && py >= cell_aabbs[cb + 1]
            && py <= cell_aabbs[cb + 4]
            && pz >= cell_aabbs[cb + 2]
            && pz <= cell_aabbs[cb + 5]
            && region_contains_cpu(
                region_program,
                surface_types,
                surface_params,
                n_cells,
                prim_idx,
                px,
                py,
                pz,
            )
    };
    let mut found = CELL_NOT_FOUND;
    // Fast path for tiny geometries: when the BVH has no
    // hierarchy (single leaf node, n_cells ≤ MAX_LEAF_SIZE = 4),
    // skip the descent overhead and scan cell AABBs directly.
    // This matches the GPU kernel's `n_bvh_nodes <= 1` branch.
    if n_nodes <= 1 {
        for c in 0..n_cells {
            if in_cell(c) {
                found = c;
                break;
            }
        }
        if found == CELL_NOT_FOUND {
            for &prim_idx in unbounded_indices {
                if in_cell(prim_idx) {
                    found = prim_idx;
                    break;
                }
            }
        }
        return found;
    }
    let mut i = 0usize;
    while i < n_nodes && found == CELL_NOT_FOUND {
        let ab = i * 6;
        let in_aabb = px >= bvh_aabbs[ab]
            && px <= bvh_aabbs[ab + 3]
            && py >= bvh_aabbs[ab + 1]
            && py <= bvh_aabbs[ab + 4]
            && pz >= bvh_aabbs[ab + 2]
            && pz <= bvh_aabbs[ab + 5];
        let mb = i * 3;
        let r_or_f = bvh_meta[mb] as usize;
        let n_prims = bvh_meta[mb + 1];
        let escape = bvh_meta[mb + 2] as usize;
        if in_aabb {
            if n_prims > 0 {
                for k in 0..n_prims as usize {
                    let prim_idx = prim_indices[r_or_f + k];
                    if in_cell(prim_idx) {
                        found = prim_idx;
                        break;
                    }
                }
            }
            i += 1;
        } else {
            i = escape;
        }
    }
    if found == CELL_NOT_FOUND {
        for &prim_idx in unbounded_indices {
            if in_cell(prim_idx) {
                found = prim_idx;
                break;
            }
        }
    }
    found
}

/// CPU mirror. Same flattened layout, same rope-traversal order.
pub fn run_bvh_cell_finding_cpu(
    positions: &[f64],
    bvh_aabbs: &[f64],
    bvh_meta: &[u32],
    prim_indices: &[u32],
    unbounded_indices: &[u32],
    cell_aabbs: &[f64],
) -> Vec<u32> {
    let n = positions.len() / 3;
    let n_nodes = bvh_aabbs.len() / 6;
    let mut out = Vec::with_capacity(n);
    for p in 0..n {
        let i3 = p * 3;
        let px = positions[i3];
        let py = positions[i3 + 1];
        let pz = positions[i3 + 2];
        let mut found = CELL_NOT_FOUND;
        let mut i = 0usize;
        while i < n_nodes && found == CELL_NOT_FOUND {
            let ab = i * 6;
            let in_aabb = px >= bvh_aabbs[ab]
                && px <= bvh_aabbs[ab + 3]
                && py >= bvh_aabbs[ab + 1]
                && py <= bvh_aabbs[ab + 4]
                && pz >= bvh_aabbs[ab + 2]
                && pz <= bvh_aabbs[ab + 5];
            let mb = i * 3;
            let r_or_f = bvh_meta[mb] as usize;
            let n_prims = bvh_meta[mb + 1];
            let escape = bvh_meta[mb + 2] as usize;
            if in_aabb {
                if n_prims > 0 {
                    for k in 0..n_prims as usize {
                        let prim_idx = prim_indices[r_or_f + k];
                        let cb = (prim_idx as usize) * 6;
                        if px >= cell_aabbs[cb]
                            && px <= cell_aabbs[cb + 3]
                            && py >= cell_aabbs[cb + 1]
                            && py <= cell_aabbs[cb + 4]
                            && pz >= cell_aabbs[cb + 2]
                            && pz <= cell_aabbs[cb + 5]
                        {
                            found = prim_idx;
                            break;
                        }
                    }
                }
                i += 1;
            } else {
                i = escape;
            }
        }
        if found == CELL_NOT_FOUND {
            for &prim_idx in unbounded_indices {
                let cb = (prim_idx as usize) * 6;
                if px >= cell_aabbs[cb]
                    && px <= cell_aabbs[cb + 3]
                    && py >= cell_aabbs[cb + 1]
                    && py <= cell_aabbs[cb + 4]
                    && pz >= cell_aabbs[cb + 2]
                    && pz <= cell_aabbs[cb + 5]
                {
                    found = prim_idx;
                    break;
                }
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
    use yamc_geo::BoundingBox;

    fn build_bvh_from_aabbs(aabbs: &[f64]) -> Bvh {
        let mut items: Vec<(u32, BoundingBox)> = Vec::new();
        for (i, chunk) in aabbs.as_chunks::<6>().0.iter().enumerate() {
            items.push((
                i as u32,
                BoundingBox::new(
                    [chunk[0], chunk[1], chunk[2]],
                    [chunk[3], chunk[4], chunk[5]],
                ),
            ));
        }
        Bvh::build(&items)
    }

    /// 4×4×1 grid of unit cells, 1000 random points. The BVH-based
    /// kernel must agree with the linear-scan kernel on every
    /// assignment.
    #[test]
    fn bvh_matches_linear_scan_on_grid() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => return,
        };

        let mut cell_aabbs = Vec::with_capacity(16 * 6);
        for ix in 0..4 {
            for iy in 0..4 {
                let x0 = ix as f64;
                let y0 = iy as f64;
                cell_aabbs.extend_from_slice(&[x0, y0, 0.0, x0 + 1.0, y0 + 1.0, 1.0]);
            }
        }
        let bvh = build_bvh_from_aabbs(&cell_aabbs);
        let (bvh_aabbs, bvh_meta, prims, unb) = flatten_bvh(&bvh);

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

        let bvh_gpu = run_bvh_cell_finding(
            &ctx,
            &positions,
            &bvh_aabbs,
            &bvh_meta,
            &prims,
            &unb,
            &cell_aabbs,
        );
        let bvh_cpu =
            run_bvh_cell_finding_cpu(&positions, &bvh_aabbs, &bvh_meta, &prims, &unb, &cell_aabbs);
        assert_eq!(bvh_gpu, bvh_cpu, "GPU and CPU BVH paths must agree");

        // Compare to the ground-truth linear scan.
        let linear =
            crate::common::geometry::cell_finding::run_cell_finding_cpu(&positions, &cell_aabbs);
        assert_eq!(
            bvh_cpu, linear,
            "BVH must give the same cell assignments as linear scan"
        );
    }

    /// Single primitive at origin, point inside.
    #[test]
    fn bvh_single_primitive() {
        let cell_aabbs = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bvh = build_bvh_from_aabbs(&cell_aabbs);
        let (a, m, p, u) = flatten_bvh(&bvh);
        let positions = vec![0.5, 0.5, 0.5];
        let cpu = run_bvh_cell_finding_cpu(&positions, &a, &m, &p, &u, &cell_aabbs);
        assert_eq!(cpu, vec![0u32]);
    }

    /// Empty BVH (no primitives at all) must return CELL_NOT_FOUND.
    #[test]
    fn bvh_empty() {
        let cell_aabbs: Vec<f64> = vec![];
        let bvh = build_bvh_from_aabbs(&cell_aabbs);
        let (a, m, p, u) = flatten_bvh(&bvh);
        let positions = vec![0.5, 0.5, 0.5];
        let cpu = run_bvh_cell_finding_cpu(&positions, &a, &m, &p, &u, &cell_aabbs);
        assert_eq!(cpu, vec![CELL_NOT_FOUND]);
    }
}
