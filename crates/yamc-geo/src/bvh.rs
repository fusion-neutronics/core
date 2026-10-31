//! Flat-array binary BVH for point-in-cell queries.
//!
//! Built once from a slice of `(primitive_index, bounding_box)` pairs and
//! queried by descending an iterative stack. The node format is fixed-size
//! and POD-friendly, so the same buffer can later be uploaded to a GPU
//! and traversed in WGSL.
//!
//! Primitives whose bounding box is non-finite (regions defined only by
//! infinite half-spaces, complements without finite bounds, or empty
//! regions whose `lower_left > upper_right`) cannot be sorted into the
//! BVH. They live in a separate `unbounded` list and are tested after the
//! spatial descent -- typically a handful per geometry.
//!
//! Build strategy: median split on the longest axis of the parent AABB,
//! sorted by primitive centroid. This is O(N log N) build, no SAH; quality
//! is ~indistinguishable from SAH for the tens-to-thousands-of-cells
//! regime we target, and it keeps the builder small and dependency-free.

use crate::bounding_box::BoundingBox;

/// One node in the flat BVH. Internal nodes have `n_prims == 0`; leaves
/// have `n_prims > 0`.
///
/// Layout is `#[repr(C)]` so the same `Vec<BvhNode>` can be uploaded to
/// a GPU buffer with `bytemuck` (when that integration lands).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BvhNode {
    pub aabb_min: [f64; 3],
    pub aabb_max: [f64; 3],
    /// Internal node: index of the right child (left child is at
    /// `self_idx + 1`). Leaf: starting index into `Bvh::prim_indices`.
    pub right_or_first_prim: u32,
    /// `0` = internal node; `> 0` = leaf with this many primitives.
    pub n_prims: u32,
}

/// A binary BVH built from primitive bounding boxes.
#[derive(Debug, Clone)]
pub struct Bvh {
    /// Nodes in depth-first order; `nodes[0]` is the root if non-empty.
    pub nodes: Vec<BvhNode>,
    /// Leaf payload: a packed list of primitive indices referenced by
    /// `BvhNode::right_or_first_prim` for leaf nodes.
    pub prim_indices: Vec<u32>,
    /// Primitive indices whose bounding box was non-finite. Tested
    /// linearly after the BVH descent fails to find a hit.
    pub unbounded: Vec<u32>,
}

/// Maximum primitives per leaf. A small leaf size means more AABB tests
/// but fewer wasted `Cell::contains` calls; tunable.
const MAX_LEAF_SIZE: usize = 4;

/// Maximum traversal stack depth. With binary splits and 32 levels we
/// can address 2³² primitives -- well past anything realistic.
const MAX_STACK: usize = 32;

impl Bvh {
    /// Build a BVH over `(primitive_index, bounding_box)` pairs.
    /// Bounded primitives go into the spatial tree; unbounded ones go
    /// into the `unbounded` fallback list.
    pub fn build(items: &[(u32, BoundingBox)]) -> Self {
        let mut bounded: Vec<(u32, BoundingBox)> = Vec::with_capacity(items.len());
        let mut unbounded: Vec<u32> = Vec::new();
        for (idx, bb) in items {
            if bb.is_finite() {
                bounded.push((*idx, bb.clone()));
            } else {
                unbounded.push(*idx);
            }
        }

        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * bounded.len().max(1));
        let mut prim_indices: Vec<u32> = Vec::with_capacity(bounded.len());

        if !bounded.is_empty() {
            build_recursive(&mut bounded, &mut nodes, &mut prim_indices);
        }

        Bvh {
            nodes,
            prim_indices,
            unbounded,
        }
    }

    /// Find the first primitive whose AABB contains `point` AND whose
    /// `contains_fn(idx)` returns true. After exhausting the BVH, falls
    /// through to the `unbounded` list.
    ///
    /// `contains_fn` is the user-supplied per-primitive test (typically
    /// `|i| cells[i as usize].contains(point)`).
    #[inline]
    pub fn find<F: FnMut(u32) -> bool>(&self, point: [f64; 3], mut contains_fn: F) -> Option<u32> {
        if !self.nodes.is_empty() {
            let mut stack = [0u32; MAX_STACK];
            let mut top = 1usize;
            // root at index 0
            while top > 0 {
                top -= 1;
                let node_idx = stack[top] as usize;
                let node = &self.nodes[node_idx];
                if !aabb_contains_point(&node.aabb_min, &node.aabb_max, &point) {
                    continue;
                }
                if node.n_prims > 0 {
                    let start = node.right_or_first_prim as usize;
                    for i in 0..node.n_prims as usize {
                        let prim = self.prim_indices[start + i];
                        if contains_fn(prim) {
                            return Some(prim);
                        }
                    }
                } else {
                    // Internal: push left (self+1) and right (stored).
                    debug_assert!(top + 2 <= MAX_STACK, "BVH traversal stack overflow");
                    stack[top] = node_idx as u32 + 1;
                    stack[top + 1] = node.right_or_first_prim;
                    top += 2;
                }
            }
        }
        self.unbounded
            .iter()
            .find(|&&idx| contains_fn(idx))
            .copied()
    }

    /// Number of bounded primitives indexed by the BVH (excluding the
    /// unbounded fallback list).
    pub fn bounded_count(&self) -> usize {
        self.prim_indices.len()
    }
}

#[inline]
fn aabb_contains_point(amin: &[f64; 3], amax: &[f64; 3], p: &[f64; 3]) -> bool {
    p[0] >= amin[0]
        && p[0] <= amax[0]
        && p[1] >= amin[1]
        && p[1] <= amax[1]
        && p[2] >= amin[2]
        && p[2] <= amax[2]
}

fn combined_aabb(items: &[(u32, BoundingBox)]) -> ([f64; 3], [f64; 3]) {
    let mut lo = items[0].1.lower_left;
    let mut hi = items[0].1.upper_right;
    for (_, bb) in &items[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(bb.lower_left[k]);
            hi[k] = hi[k].max(bb.upper_right[k]);
        }
    }
    (lo, hi)
}

fn build_recursive(
    items: &mut [(u32, BoundingBox)],
    nodes: &mut Vec<BvhNode>,
    prim_indices: &mut Vec<u32>,
) -> u32 {
    let self_idx = nodes.len() as u32;
    let (aabb_min, aabb_max) = combined_aabb(items);

    if items.len() <= MAX_LEAF_SIZE {
        let first = prim_indices.len() as u32;
        for (idx, _) in items.iter() {
            prim_indices.push(*idx);
        }
        nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            right_or_first_prim: first,
            n_prims: items.len() as u32,
        });
        return self_idx;
    }

    // Pick the longest axis of the combined AABB and median-split centroids.
    let extent = [
        aabb_max[0] - aabb_min[0],
        aabb_max[1] - aabb_min[1],
        aabb_max[2] - aabb_min[2],
    ];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    items.sort_by(|a, b| {
        let ca = 0.5 * (a.1.lower_left[axis] + a.1.upper_right[axis]);
        let cb = 0.5 * (b.1.lower_left[axis] + b.1.upper_right[axis]);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = items.len() / 2;

    // Reserve our slot before recursing so children land at the right
    // depth-first positions.
    nodes.push(BvhNode {
        aabb_min,
        aabb_max,
        right_or_first_prim: 0, // patched after right child is built
        n_prims: 0,             // marks internal node
    });

    // Left child immediately follows this node (`self_idx + 1`).
    let _left = build_recursive(&mut items[..mid], nodes, prim_indices);
    // Right child sits at whatever position the recursion lands on next.
    let right = build_recursive(&mut items[mid..], nodes, prim_indices);
    nodes[self_idx as usize].right_or_first_prim = right;

    self_idx
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(lo: [f64; 3], hi: [f64; 3]) -> BoundingBox {
        BoundingBox::new(lo, hi)
    }

    #[test]
    fn empty_bvh_returns_none() {
        let bvh = Bvh::build(&[]);
        assert_eq!(bvh.find([0.0, 0.0, 0.0], |_| true), None);
    }

    #[test]
    fn single_primitive_inside_returns_it() {
        let bvh = Bvh::build(&[(7, bb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]);
        assert_eq!(bvh.find([0.5, 0.5, 0.5], |_| true), Some(7));
    }

    #[test]
    fn single_primitive_outside_returns_none() {
        let bvh = Bvh::build(&[(7, bb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]);
        assert_eq!(bvh.find([5.0, 0.0, 0.0], |_| true), None);
    }

    #[test]
    fn many_disjoint_boxes_finds_correct_one() {
        // 27 unit cubes in a 3x3x3 grid. With MAX_LEAF_SIZE > 1 the leaves
        // hold multiple prims, so the contains_fn must implement the real
        // per-primitive test (here: per-prim AABB membership).
        let mut items = Vec::new();
        let mut bboxes: Vec<([f64; 3], [f64; 3])> = Vec::new();
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let idx = (i + 3 * (j + 3 * k)) as u32;
                    let lo = [i as f64, j as f64, k as f64];
                    let hi = [i as f64 + 1.0, j as f64 + 1.0, k as f64 + 1.0];
                    items.push((idx, bb(lo, hi)));
                    bboxes.push((lo, hi));
                }
            }
        }
        let bvh = Bvh::build(&items);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                    let want = (i + 3 * (j + 3 * k)) as u32;
                    let got = bvh.find(p, |idx| {
                        let (lo, hi) = bboxes[idx as usize];
                        aabb_contains_point(&lo, &hi, &p)
                    });
                    assert_eq!(got, Some(want), "miss at {p:?}");
                }
            }
        }
    }

    #[test]
    fn unbounded_primitives_go_to_fallback_list() {
        let bounded = (0u32, bb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        let unbounded = (
            1u32,
            bb([f64::NEG_INFINITY, 0.0, 0.0], [f64::INFINITY, 1.0, 1.0]),
        );
        let bvh = Bvh::build(&[bounded, unbounded]);
        assert_eq!(bvh.unbounded, vec![1]);
        // Inside bounded → bounded primitive.
        assert_eq!(bvh.find([0.5, 0.5, 0.5], |_| true), Some(0));
        // Outside bounded box but contains_fn says yes for index 1 →
        // fallback list resolves it.
        let outside = [10.0, 0.5, 0.5];
        assert_eq!(bvh.find(outside, |idx| idx == 1), Some(1));
    }

    #[test]
    fn contains_fn_can_reject_aabb_hit() {
        // Two overlapping AABBs, but only one of them actually contains
        // the point per the user-supplied contains_fn.
        let items = vec![
            (0u32, bb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])),
            (1u32, bb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])),
        ];
        let bvh = Bvh::build(&items);
        // Both AABBs contain [1.5, 1.5, 1.5]; contains_fn only accepts idx 1.
        assert_eq!(bvh.find([1.5, 1.5, 1.5], |idx| idx == 1), Some(1));
    }
}
