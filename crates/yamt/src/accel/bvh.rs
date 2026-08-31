//! SAH-based Bounding Volume Hierarchy for fast ray-AABB and point-AABB queries.
//!
//! Stores AABBs in a flat array for cache-friendly traversal. Each leaf
//! references a range of primitives. Interior nodes store the split axis
//! and child indices.

/// An axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: [f64::MAX, f64::MAX, f64::MAX],
        max: [f64::MIN, f64::MIN, f64::MIN],
    };

    /// Expand this AABB to include a point.
    #[inline]
    pub fn expand_point(&mut self, p: [f64; 3]) {
        for (i, &pi) in p.iter().enumerate() {
            self.min[i] = self.min[i].min(pi);
            self.max[i] = self.max[i].max(pi);
        }
    }

    /// Expand this AABB to include another AABB.
    #[inline]
    pub fn expand_aabb(&mut self, other: &Aabb) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    /// Surface area (used by SAH).
    #[inline]
    pub fn surface_area(&self) -> f64 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        2.0 * (dx * dy + dy * dz + dz * dx)
    }

    /// Centroid of this AABB.
    #[inline]
    pub fn centroid(&self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Longest axis: 0 = x, 1 = y, 2 = z.
    #[inline]
    pub fn longest_axis(&self) -> usize {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz {
            0
        } else if dy >= dz {
            1
        } else {
            2
        }
    }

    /// Test if a point is inside (inclusive).
    #[inline]
    pub fn contains_point(&self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    /// Ray-AABB slab test. Returns `Some((tmin, tmax))` if the ray intersects.
    ///
    /// When the `simd` feature is enabled, this delegates to a runtime-dispatched
    /// version compiled with the best available ISA extensions (AVX2+FMA, SSE4.1).
    #[cfg(feature = "simd")]
    #[inline]
    pub fn ray_intersect(&self, origin: [f64; 3], inv_dir: [f64; 3]) -> Option<(f64, f64)> {
        crate::accel::simd::ray_aabb_intersect(origin, inv_dir, self.min, self.max)
    }

    /// Ray-AABB slab test (scalar fallback).
    #[cfg(not(feature = "simd"))]
    #[inline]
    pub fn ray_intersect(&self, origin: [f64; 3], inv_dir: [f64; 3]) -> Option<(f64, f64)> {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;

        for i in 0..3 {
            let t1 = (self.min[i] - origin[i]) * inv_dir[i];
            let t2 = (self.max[i] - origin[i]) * inv_dir[i];
            let ta = t1.min(t2);
            let tb = t1.max(t2);
            tmin = tmin.max(ta);
            tmax = tmax.min(tb);
        }

        if tmax >= tmin.max(0.0) {
            Some((tmin, tmax))
        } else {
            None
        }
    }

    /// Minimum squared distance from a point to this AABB (0 if inside).
    #[inline]
    pub fn min_distance_sq(&self, p: [f64; 3]) -> f64 {
        let mut dist_sq = 0.0;
        for (pi, (lo, hi)) in p.iter().zip(self.min.iter().zip(self.max.iter())) {
            if *pi < *lo {
                let d = lo - pi;
                dist_sq += d * d;
            } else if *pi > *hi {
                let d = pi - hi;
                dist_sq += d * d;
            }
        }
        dist_sq
    }

    /// Dilate the AABB by a small factor to avoid missing primitives at boundaries.
    pub fn dilated(&self, factor: f64) -> Self {
        let dx = (self.max[0] - self.min[0]) * factor;
        let dy = (self.max[1] - self.min[1]) * factor;
        let dz = (self.max[2] - self.min[2]) * factor;
        // Ensure a minimum dilation for degenerate boxes
        let eps = 1e-10;
        let dx = dx.max(eps);
        let dy = dy.max(eps);
        let dz = dz.max(eps);
        Self {
            min: [self.min[0] - dx, self.min[1] - dy, self.min[2] - dz],
            max: [self.max[0] + dx, self.max[1] + dy, self.max[2] + dz],
        }
    }
}

impl From<[f64; 6]> for Aabb {
    fn from(v: [f64; 6]) -> Self {
        Self {
            min: [v[0], v[1], v[2]],
            max: [v[3], v[4], v[5]],
        }
    }
}

/// Dilation factor for BVH leaf bounds (from XDG constants).
const BOX_DILATION: f64 = 1e-7;

/// Number of SAH bins.
const NUM_BINS: usize = 12;

/// A flat BVH node with no enum discriminant.
///
/// `count == 0` means interior node:
///   - left child is always at `self_index + 1` (DFS build order)
///   - `offset` stores the right child index
///
/// `count > 0` means leaf node:
///   - `offset` is the start index into the primitive index buffer
///   - `count` is the number of primitives
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BvhNode {
    min: [f64; 3],
    max: [f64; 3],
    offset: u32,
    count: u32,
}

impl BvhNode {
    #[inline]
    fn bounds(&self) -> Aabb {
        Aabb {
            min: self.min,
            max: self.max,
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

// ===========================================================================
// BVH4 node (4-wide, SoA layout for SIMD traversal)
// ===========================================================================

/// A 4-wide BVH node storing up to 4 child AABBs in SoA layout.
///
/// `child_meta[i]` packs count (high 32 bits) and offset (low 32 bits):
/// - Leaf child: count > 0, offset = start index into `indices`
/// - Interior child: count == 0, offset = BVH4 node index
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Bvh4Node {
    child_min_x: [f64; 4],
    child_min_y: [f64; 4],
    child_min_z: [f64; 4],
    child_max_x: [f64; 4],
    child_max_y: [f64; 4],
    child_max_z: [f64; 4],
    child_meta: [u64; 4],
    num_children: u8,
}

impl Bvh4Node {
    const SENTINEL_META: u64 = (u32::MAX as u64) << 32;

    fn empty() -> Self {
        Self {
            child_min_x: [f64::MAX; 4],
            child_min_y: [f64::MAX; 4],
            child_min_z: [f64::MAX; 4],
            child_max_x: [f64::MIN; 4],
            child_max_y: [f64::MIN; 4],
            child_max_z: [f64::MIN; 4],
            child_meta: [Self::SENTINEL_META; 4],
            num_children: 0,
        }
    }

    #[inline]
    fn child_count(meta: u64) -> u32 {
        (meta >> 32) as u32
    }

    #[inline]
    fn child_offset(meta: u64) -> u32 {
        meta as u32
    }

    #[inline]
    fn is_child_leaf(meta: u64) -> bool {
        let c = Self::child_count(meta);
        c > 0 && c != u32::MAX
    }

    #[inline]
    fn make_leaf_meta(offset: u32, count: u32) -> u64 {
        ((count as u64) << 32) | offset as u64
    }

    #[inline]
    fn make_interior_meta(node_idx: u32) -> u64 {
        node_idx as u64
    }
}

/// A BVH over a set of primitives (triangles or tetrahedra).
///
/// Uses a 4-wide BVH (BVH4) internally: each node has up to 4 children
/// whose AABBs are tested simultaneously with SIMD. Built by constructing
/// a binary BVH (SAH) then collapsing pairs of interior nodes.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<Bvh4Node>,
    /// Reordered primitive indices.
    pub indices: Vec<u32>,
    root_bounds: Option<Aabb>,
}

/// Maximum primitives per leaf before splitting.
const MAX_LEAF_SIZE: usize = 4;

impl Bvh {
    /// Build a BVH over primitives with given AABBs.
    ///
    /// `aabbs[i]` is the bounding box for primitive `i`.
    pub fn build(aabbs: &[[f64; 6]]) -> Self {
        if aabbs.is_empty() {
            return Self {
                nodes: Vec::new(),
                indices: Vec::new(),
                root_bounds: None,
            };
        }

        let mut indices: Vec<u32> = (0..aabbs.len() as u32).collect();
        let bvh_aabbs: Vec<Aabb> = aabbs
            .iter()
            .map(|a| Aabb::from(*a).dilated(BOX_DILATION))
            .collect();
        let centroids: Vec<[f64; 3]> = bvh_aabbs.iter().map(|a| a.centroid()).collect();

        // Phase 1: Build BVH2
        let mut bvh2_nodes = Vec::with_capacity(2 * aabbs.len());
        Self::build_recursive(
            &mut bvh2_nodes,
            &mut indices,
            &bvh_aabbs,
            &centroids,
            0,
            aabbs.len(),
        );

        // Extract root bounds before collapse
        let root_bounds = Some(bvh2_nodes[0].bounds());

        // Phase 2: Collapse BVH2 → BVH4
        let mut bvh4_nodes = Vec::with_capacity(bvh2_nodes.len() / 2 + 1);
        Self::collapse_recursive(&bvh2_nodes, 0, &mut bvh4_nodes);

        Self {
            nodes: bvh4_nodes,
            indices,
            root_bounds,
        }
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        indices: &mut [u32],
        aabbs: &[Aabb],
        centroids: &[[f64; 3]],
        start: usize,
        end: usize,
    ) -> u32 {
        let count = end - start;

        // Compute bounds for this range
        let mut bounds = Aabb::EMPTY;
        for i in start..end {
            bounds.expand_aabb(&aabbs[indices[i] as usize]);
        }

        // Leaf
        if count <= MAX_LEAF_SIZE {
            let node_idx = nodes.len() as u32;
            nodes.push(BvhNode {
                min: bounds.min,
                max: bounds.max,
                offset: start as u32,
                count: count as u32,
            });
            return node_idx;
        }

        // SAH binned split
        let mut centroid_bounds = Aabb::EMPTY;
        for i in start..end {
            centroid_bounds.expand_point(centroids[indices[i] as usize]);
        }

        let axis = centroid_bounds.longest_axis();
        let axis_min = centroid_bounds.min[axis];
        let axis_max = centroid_bounds.max[axis];

        // If centroids are coincident, fall back to median split
        if (axis_max - axis_min).abs() < 1e-14 {
            let mid = (start + end) / 2;
            let node_idx = nodes.len() as u32;
            // Reserve slot (left child will be at node_idx + 1)
            nodes.push(BvhNode {
                min: bounds.min,
                max: bounds.max,
                offset: 0,
                count: 0,
            });
            let _left = Self::build_recursive(nodes, indices, aabbs, centroids, start, mid);
            let right = Self::build_recursive(nodes, indices, aabbs, centroids, mid, end);
            nodes[node_idx as usize].offset = right;
            return node_idx;
        }

        // Bin primitives
        let scale = NUM_BINS as f64 / (axis_max - axis_min);
        let mut bin_counts = [0usize; NUM_BINS];
        let mut bin_bounds = [Aabb::EMPTY; NUM_BINS];

        for idx in indices.iter().take(end).skip(start) {
            let idx = *idx as usize;
            let c = centroids[idx][axis];
            let bin = ((c - axis_min) * scale).min((NUM_BINS - 1) as f64) as usize;
            bin_counts[bin] += 1;
            bin_bounds[bin].expand_aabb(&aabbs[idx]);
        }

        // Evaluate SAH cost using prefix/suffix scans (O(1) per split)
        let mut left_bounds_prefix = [Aabb::EMPTY; NUM_BINS];
        let mut left_count_prefix = [0usize; NUM_BINS];
        left_bounds_prefix[0] = bin_bounds[0];
        left_count_prefix[0] = bin_counts[0];
        for i in 1..NUM_BINS {
            left_bounds_prefix[i] = left_bounds_prefix[i - 1];
            left_bounds_prefix[i].expand_aabb(&bin_bounds[i]);
            left_count_prefix[i] = left_count_prefix[i - 1] + bin_counts[i];
        }

        let mut right_bounds_suffix = [Aabb::EMPTY; NUM_BINS];
        let mut right_count_suffix = [0usize; NUM_BINS];
        right_bounds_suffix[NUM_BINS - 1] = bin_bounds[NUM_BINS - 1];
        right_count_suffix[NUM_BINS - 1] = bin_counts[NUM_BINS - 1];
        for i in (0..NUM_BINS - 1).rev() {
            right_bounds_suffix[i] = right_bounds_suffix[i + 1];
            right_bounds_suffix[i].expand_aabb(&bin_bounds[i]);
            right_count_suffix[i] = right_count_suffix[i + 1] + bin_counts[i];
        }

        let mut best_cost = f64::MAX;
        let mut best_split = 0;

        for split in 1..NUM_BINS {
            let left_count = left_count_prefix[split - 1];
            let right_count = right_count_suffix[split];

            if left_count == 0 || right_count == 0 {
                continue;
            }

            let cost = left_count as f64 * left_bounds_prefix[split - 1].surface_area()
                + right_count as f64 * right_bounds_suffix[split].surface_area();
            if cost < best_cost {
                best_cost = cost;
                best_split = split;
            }
        }

        // Partition primitives
        let threshold = axis_min + best_split as f64 / scale;
        let mut mid = start;
        for i in start..end {
            if centroids[indices[i] as usize][axis] < threshold {
                indices.swap(i, mid);
                mid += 1;
            }
        }

        // Fallback: if partition is degenerate, force median split
        if mid == start || mid == end {
            mid = (start + end) / 2;
        }

        let node_idx = nodes.len() as u32;
        // Reserve slot (left child will be at node_idx + 1 due to DFS order)
        nodes.push(BvhNode {
            min: bounds.min,
            max: bounds.max,
            offset: 0,
            count: 0,
        });
        let _left = Self::build_recursive(nodes, indices, aabbs, centroids, start, mid);
        let right = Self::build_recursive(nodes, indices, aabbs, centroids, mid, end);
        nodes[node_idx as usize].offset = right;

        node_idx
    }

    /// Collapse a BVH2 subtree into BVH4 nodes.
    ///
    /// For each interior node, greedily expands the interior child with
    /// the largest surface area until we have 4 children or all are leaves.
    fn collapse_recursive(bvh2: &[BvhNode], bvh2_idx: u32, bvh4: &mut Vec<Bvh4Node>) -> u32 {
        let node = &bvh2[bvh2_idx as usize];

        // If the BVH2 root itself is a leaf, wrap it in a single BVH4 node
        if node.is_leaf() {
            let bvh4_idx = bvh4.len() as u32;
            let mut n = Bvh4Node::empty();
            n.child_min_x[0] = node.min[0];
            n.child_min_y[0] = node.min[1];
            n.child_min_z[0] = node.min[2];
            n.child_max_x[0] = node.max[0];
            n.child_max_y[0] = node.max[1];
            n.child_max_z[0] = node.max[2];
            n.child_meta[0] = Bvh4Node::make_leaf_meta(node.offset, node.count);
            n.num_children = 1;
            bvh4.push(n);
            return bvh4_idx;
        }

        // Collect children by greedily expanding interior nodes up to 4
        // Each entry: (bvh2_index, is_leaf, surface_area)
        let left_idx = bvh2_idx + 1;
        let right_idx = node.offset;
        let mut children: Vec<(u32, bool, f64)> = vec![
            (
                left_idx,
                bvh2[left_idx as usize].is_leaf(),
                bvh2[left_idx as usize].bounds().surface_area(),
            ),
            (
                right_idx,
                bvh2[right_idx as usize].is_leaf(),
                bvh2[right_idx as usize].bounds().surface_area(),
            ),
        ];

        while children.len() < 4 {
            // Find the interior child with the largest surface area
            let best = children
                .iter()
                .enumerate()
                .filter(|(_, (_, is_leaf, _))| !is_leaf)
                .max_by(|(_, (_, _, sa_a)), (_, (_, _, sa_b))| {
                    sa_a.partial_cmp(sa_b).unwrap_or(std::cmp::Ordering::Equal)
                });

            let best_pos = match best {
                Some((idx, _)) => idx,
                None => break, // all children are leaves
            };

            let (child_bvh2_idx, _, _) = children.remove(best_pos);
            let child_node = &bvh2[child_bvh2_idx as usize];
            let cl = child_bvh2_idx + 1;
            let cr = child_node.offset;
            children.push((
                cl,
                bvh2[cl as usize].is_leaf(),
                bvh2[cl as usize].bounds().surface_area(),
            ));
            children.push((
                cr,
                bvh2[cr as usize].is_leaf(),
                bvh2[cr as usize].bounds().surface_area(),
            ));
        }

        // Reserve a BVH4 node slot
        let bvh4_idx = bvh4.len() as u32;
        bvh4.push(Bvh4Node::empty());

        let num = children.len();
        for (i, &(child_bvh2_idx, is_leaf, _)) in children.iter().enumerate() {
            let cn = &bvh2[child_bvh2_idx as usize];
            bvh4[bvh4_idx as usize].child_min_x[i] = cn.min[0];
            bvh4[bvh4_idx as usize].child_min_y[i] = cn.min[1];
            bvh4[bvh4_idx as usize].child_min_z[i] = cn.min[2];
            bvh4[bvh4_idx as usize].child_max_x[i] = cn.max[0];
            bvh4[bvh4_idx as usize].child_max_y[i] = cn.max[1];
            bvh4[bvh4_idx as usize].child_max_z[i] = cn.max[2];

            if is_leaf {
                bvh4[bvh4_idx as usize].child_meta[i] =
                    Bvh4Node::make_leaf_meta(cn.offset, cn.count);
            } else {
                let child_bvh4_idx = Self::collapse_recursive(bvh2, child_bvh2_idx, bvh4);
                bvh4[bvh4_idx as usize].child_meta[i] =
                    Bvh4Node::make_interior_meta(child_bvh4_idx);
            }
        }
        bvh4[bvh4_idx as usize].num_children = num as u8;

        bvh4_idx
    }

    /// Traverse the BVH with a ray, calling `test_fn` for each candidate
    /// primitive index. The `test_fn` should return `Some(distance)` on hit.
    ///
    /// Returns the result from `test_fn` with the smallest positive distance.
    ///
    /// Uses a fixed-size stack (no heap allocation) and ordered traversal
    /// (closer child first) for better early-exit pruning.
    pub fn ray_traverse<F>(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        test_fn: F,
    ) -> Option<(u32, f64)>
    where
        F: FnMut(u32) -> Option<f64>,
    {
        self.ray_traverse_seeded(origin, direction, f64::INFINITY, test_fn)
    }

    /// Like [`Bvh::ray_traverse`] but the search is seeded with a virtual
    /// hit at `t_max`: BVH nodes whose entry distance exceeds `t_max` are
    /// never visited, so short-segment queries stay cheap. Only hits
    /// strictly closer than `t_max` are returned.
    pub fn ray_traverse_upto<F>(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
        test_fn: F,
    ) -> Option<(u32, f64)>
    where
        F: FnMut(u32) -> Option<f64>,
    {
        self.ray_traverse_seeded(origin, direction, t_max, test_fn)
    }

    fn ray_traverse_seeded<F>(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        seed_t: f64,
        mut test_fn: F,
    ) -> Option<(u32, f64)>
    where
        F: FnMut(u32) -> Option<f64>,
    {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];

        // Check root bounds first
        if let Some(ref rb) = self.root_bounds {
            rb.ray_intersect(origin, inv_dir)?;
        }

        // A finite seed acts as a virtual hit at `seed_t`: the ordered
        // traversal then culls every node farther than the seed.
        let mut closest: Option<(u32, f64)> = if seed_t.is_finite() {
            Some((u32::MAX, seed_t))
        } else {
            None
        };
        let mut stack: [(u32, f64); 64] = [(0, 0.0); 64];
        let mut stack_ptr = 1usize;
        stack[0] = (0, 0.0);
        // Deep or skimming queries can exceed the fixed stack; spill to the
        // heap instead of silently dropping subtrees (missed hits).
        let mut overflow: Vec<(u32, f64)> = Vec::new();

        while stack_ptr > 0 || !overflow.is_empty() {
            let (node_idx, node_tmin) = if let Some(x) = overflow.pop() {
                x
            } else {
                stack_ptr -= 1;
                stack[stack_ptr]
            };

            if let Some((_, best_t)) = closest {
                if node_tmin > best_t {
                    continue;
                }
            }

            let node = &self.nodes[node_idx as usize];
            let nc = node.num_children as usize;

            // 4-wide ray-AABB test
            #[cfg(feature = "simd")]
            let (tmin4, hit_mask) = {
                let (t, m) = crate::accel::simd::ray_aabb4_intersect(
                    origin,
                    inv_dir,
                    &node.child_min_x,
                    &node.child_min_y,
                    &node.child_min_z,
                    &node.child_max_x,
                    &node.child_max_y,
                    &node.child_max_z,
                );
                // Mask off unused children
                (t, m & ((1u8 << nc) - 1))
            };

            #[cfg(not(feature = "simd"))]
            let (tmin4, hit_mask) = {
                let mut tmin_arr = [f64::MAX; 4];
                let mut mask: u8 = 0;
                for i in 0..nc {
                    let aabb = Aabb {
                        min: [
                            node.child_min_x[i],
                            node.child_min_y[i],
                            node.child_min_z[i],
                        ],
                        max: [
                            node.child_max_x[i],
                            node.child_max_y[i],
                            node.child_max_z[i],
                        ],
                    };
                    if let Some((t, _)) = aabb.ray_intersect(origin, inv_dir) {
                        tmin_arr[i] = t;
                        mask |= 1 << i;
                    }
                }
                (tmin_arr, mask)
            };

            if hit_mask == 0 {
                continue;
            }

            // Collect hits: (child_slot, tmin)
            let mut hits: [(usize, f64); 4] = [(0, 0.0); 4];
            let mut nhits = 0usize;
            for (i, &t) in tmin4.iter().enumerate().take(nc) {
                if hit_mask & (1 << i) != 0 {
                    hits[nhits] = (i, t);
                    nhits += 1;
                }
            }

            // Process hits: leaves inline, interiors sorted onto stack
            // Sort by tmin descending so nearest is pushed last (popped first)
            // Simple insertion sort (max 4 elements)
            for a in 1..nhits {
                let key = hits[a];
                let mut b = a;
                while b > 0 && hits[b - 1].1 < key.1 {
                    hits[b] = hits[b - 1];
                    b -= 1;
                }
                hits[b] = key;
            }

            for &(slot, tmin_val) in hits.iter().take(nhits) {
                let meta = node.child_meta[slot];

                if let Some((_, best_t)) = closest {
                    if tmin_val > best_t {
                        continue;
                    }
                }

                if Bvh4Node::is_child_leaf(meta) {
                    let offset = Bvh4Node::child_offset(meta);
                    let count = Bvh4Node::child_count(meta);
                    for j in offset..(offset + count) {
                        let prim_idx = self.indices[j as usize];
                        if let Some(t) = test_fn(prim_idx) {
                            if t > 0.0 {
                                match closest {
                                    None => closest = Some((prim_idx, t)),
                                    Some((_, best_t)) if t < best_t => {
                                        closest = Some((prim_idx, t));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                } else if stack_ptr < 64 {
                    stack[stack_ptr] = (Bvh4Node::child_offset(meta), tmin_val);
                    stack_ptr += 1;
                } else {
                    overflow.push((Bvh4Node::child_offset(meta), tmin_val));
                }
            }
        }

        closest.filter(|&(prim, _)| prim != u32::MAX)
    }

    /// Traverse every node the ray touches and call `test_fn` for every
    /// candidate primitive, with NO nearest-hit pruning and no fixed-size
    /// stack drop. Parity (point containment) queries must enumerate every
    /// hit along the ray; routing them through `ray_traverse` silently
    /// dropped far hits (its ordered traversal prunes nodes beyond the
    /// current best) and truncated subtrees when the 64-slot stack filled,
    /// which corrupted crossing counts on large meshes.
    pub fn ray_traverse_collect<F>(&self, origin: [f64; 3], direction: [f64; 3], mut test_fn: F)
    where
        F: FnMut(u32),
    {
        if self.nodes.is_empty() {
            return;
        }
        let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];
        if let Some(ref rb) = self.root_bounds {
            if rb.ray_intersect(origin, inv_dir).is_none() {
                return;
            }
        }
        let mut stack: Vec<u32> = Vec::with_capacity(128);
        stack.push(0);
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            let nc = node.num_children as usize;
            for i in 0..nc {
                let aabb = Aabb {
                    min: [
                        node.child_min_x[i],
                        node.child_min_y[i],
                        node.child_min_z[i],
                    ],
                    max: [
                        node.child_max_x[i],
                        node.child_max_y[i],
                        node.child_max_z[i],
                    ],
                };
                if aabb.ray_intersect(origin, inv_dir).is_none() {
                    continue;
                }
                let meta = node.child_meta[i];
                if Bvh4Node::is_child_leaf(meta) {
                    let offset = Bvh4Node::child_offset(meta);
                    let count = Bvh4Node::child_count(meta);
                    for j in offset..(offset + count) {
                        test_fn(self.indices[j as usize]);
                    }
                } else {
                    stack.push(Bvh4Node::child_offset(meta));
                }
            }
        }
    }

    /// Find all primitives whose AABB contains the given point.
    /// Calls `test_fn` for each candidate and returns the first that returns `true`.
    pub fn point_query<F>(&self, point: [f64; 3], mut test_fn: F) -> Option<u32>
    where
        F: FnMut(u32) -> bool,
    {
        if self.nodes.is_empty() {
            return None;
        }

        // Quick root bounds check
        if let Some(ref rb) = self.root_bounds {
            if !rb.contains_point(point) {
                return None;
            }
        }

        let mut stack = [0u32; 64];
        let mut stack_ptr = 1usize;
        stack[0] = 0;
        // Spill to the heap instead of silently dropping subtrees.
        let mut overflow: Vec<u32> = Vec::new();

        while stack_ptr > 0 || !overflow.is_empty() {
            let node_idx = if let Some(x) = overflow.pop() {
                x
            } else {
                stack_ptr -= 1;
                stack[stack_ptr]
            };
            let node = &self.nodes[node_idx as usize];
            let nc = node.num_children as usize;

            // Test children inline and process immediately (single pass)
            for i in 0..nc {
                if point[0] < node.child_min_x[i]
                    || point[0] > node.child_max_x[i]
                    || point[1] < node.child_min_y[i]
                    || point[1] > node.child_max_y[i]
                    || point[2] < node.child_min_z[i]
                    || point[2] > node.child_max_z[i]
                {
                    continue;
                }
                let meta = node.child_meta[i];
                if Bvh4Node::is_child_leaf(meta) {
                    let offset = Bvh4Node::child_offset(meta);
                    let count = Bvh4Node::child_count(meta);
                    for j in offset..(offset + count) {
                        let prim_idx = self.indices[j as usize];
                        if test_fn(prim_idx) {
                            return Some(prim_idx);
                        }
                    }
                } else if stack_ptr < 64 {
                    stack[stack_ptr] = Bvh4Node::child_offset(meta);
                    stack_ptr += 1;
                } else {
                    overflow.push(Bvh4Node::child_offset(meta));
                }
            }
        }

        None
    }

    /// Find the closest primitive to a query point using BVH pruning.
    ///
    /// `test_fn(prim_idx)` should return the squared distance from the query point
    /// to the primitive. Returns `Some((prim_idx, dist_sq))` for the closest hit.
    pub fn closest_point_traverse<F>(&self, point: [f64; 3], mut test_fn: F) -> Option<(u32, f64)>
    where
        F: FnMut(u32) -> f64,
    {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_dist_sq = f64::MAX;
        let mut best_prim: u32 = 0;
        let mut found = false;

        let mut stack: [(u32, f64); 64] = [(0, 0.0); 64];
        let mut stack_ptr = 1usize;
        stack[0] = (0, 0.0);
        // Deep or skimming queries can exceed the fixed stack; spill to the
        // heap instead of silently dropping subtrees (missed hits).
        let mut overflow: Vec<(u32, f64)> = Vec::new();

        while stack_ptr > 0 || !overflow.is_empty() {
            let (node_idx, parent_dist_sq) = if let Some(x) = overflow.pop() {
                x
            } else {
                stack_ptr -= 1;
                stack[stack_ptr]
            };

            if parent_dist_sq >= best_dist_sq {
                continue;
            }

            let node = &self.nodes[node_idx as usize];
            let nc = node.num_children as usize;

            // Inline 4-wide distance computation (too cheap for SIMD dispatch overhead)
            let mut dists = [f64::MAX; 4];
            for (i, dist_out) in dists.iter_mut().enumerate().take(nc) {
                let mut d = 0.0f64;
                let mins = [
                    node.child_min_x[i],
                    node.child_min_y[i],
                    node.child_min_z[i],
                ];
                let maxs = [
                    node.child_max_x[i],
                    node.child_max_y[i],
                    node.child_max_z[i],
                ];
                for axis in 0..3 {
                    if point[axis] < mins[axis] {
                        let delta = mins[axis] - point[axis];
                        d += delta * delta;
                    } else if point[axis] > maxs[axis] {
                        let delta = point[axis] - maxs[axis];
                        d += delta * delta;
                    }
                }
                *dist_out = d;
            }

            // Collect valid children that pass pruning
            let mut hits: [(usize, f64); 4] = [(0, 0.0); 4];
            let mut nhits = 0usize;
            for (i, &d) in dists.iter().enumerate().take(nc) {
                if d < best_dist_sq {
                    hits[nhits] = (i, d);
                    nhits += 1;
                }
            }

            // Sort by distance descending so nearest is pushed last (popped first)
            for a in 1..nhits {
                let key = hits[a];
                let mut b = a;
                while b > 0 && hits[b - 1].1 < key.1 {
                    hits[b] = hits[b - 1];
                    b -= 1;
                }
                hits[b] = key;
            }

            for &(slot, dist) in hits.iter().take(nhits) {
                if dist >= best_dist_sq {
                    continue;
                }

                let meta = node.child_meta[slot];
                if Bvh4Node::is_child_leaf(meta) {
                    let offset = Bvh4Node::child_offset(meta);
                    let count = Bvh4Node::child_count(meta);
                    for j in offset..(offset + count) {
                        let prim_idx = self.indices[j as usize];
                        let d = test_fn(prim_idx);
                        if d < best_dist_sq {
                            best_dist_sq = d;
                            best_prim = prim_idx;
                            found = true;
                        }
                    }
                } else if stack_ptr < 64 {
                    stack[stack_ptr] = (Bvh4Node::child_offset(meta), dist);
                    stack_ptr += 1;
                } else {
                    overflow.push((Bvh4Node::child_offset(meta), dist));
                }
            }
        }

        if found {
            Some((best_prim, best_dist_sq))
        } else {
            None
        }
    }

    /// Traverse the BVH with a ray, testing leaf triangles in 4-wide batches.
    ///
    /// Takes precomputed triangle data (v0, edge1, edge2) indexed by the BVH's
    /// primitive indices. Uses SIMD-dispatched 4-wide Moller-Trumbore for leaf
    /// testing. The `accept` closure is called only for actual hits and can
    /// reject specific primitives (e.g. history exclusion in ray_fire).
    ///
    /// Returns the closest accepted hit `(prim_idx, distance)`.
    #[allow(clippy::too_many_arguments)]
    pub fn ray_traverse_tris<F>(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        tri_v0: &[[f64; 3]],
        tri_edge1: &[[f64; 3]],
        tri_edge2: &[[f64; 3]],
        min_t: f64,
        mut accept: F,
    ) -> Option<(u32, f64)>
    where
        F: FnMut(u32) -> bool,
    {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = [1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2]];

        if let Some(ref rb) = self.root_bounds {
            rb.ray_intersect(origin, inv_dir)?;
        }

        let mut closest: Option<(u32, f64)> = None;
        let mut stack: [(u32, f64); 64] = [(0, 0.0); 64];
        let mut stack_ptr = 1usize;
        stack[0] = (0, 0.0);
        // Deep or skimming queries can exceed the fixed stack; spill to the
        // heap instead of silently dropping subtrees (missed hits).
        let mut overflow: Vec<(u32, f64)> = Vec::new();

        while stack_ptr > 0 || !overflow.is_empty() {
            let (node_idx, node_tmin) = if let Some(x) = overflow.pop() {
                x
            } else {
                stack_ptr -= 1;
                stack[stack_ptr]
            };

            if let Some((_, best_t)) = closest {
                if node_tmin > best_t {
                    continue;
                }
            }

            let node = &self.nodes[node_idx as usize];
            let nc = node.num_children as usize;

            // 4-wide ray-AABB test for children
            #[cfg(feature = "simd")]
            let (tmin4, hit_mask) = {
                let (t, m) = crate::accel::simd::ray_aabb4_intersect(
                    origin,
                    inv_dir,
                    &node.child_min_x,
                    &node.child_min_y,
                    &node.child_min_z,
                    &node.child_max_x,
                    &node.child_max_y,
                    &node.child_max_z,
                );
                (t, m & ((1u8 << nc) - 1))
            };

            #[cfg(not(feature = "simd"))]
            let (tmin4, hit_mask) = {
                let mut tmin_arr = [f64::MAX; 4];
                let mut mask: u8 = 0;
                for i in 0..nc {
                    let aabb = Aabb {
                        min: [
                            node.child_min_x[i],
                            node.child_min_y[i],
                            node.child_min_z[i],
                        ],
                        max: [
                            node.child_max_x[i],
                            node.child_max_y[i],
                            node.child_max_z[i],
                        ],
                    };
                    if let Some((t, _)) = aabb.ray_intersect(origin, inv_dir) {
                        tmin_arr[i] = t;
                        mask |= 1 << i;
                    }
                }
                (tmin_arr, mask)
            };

            if hit_mask == 0 {
                continue;
            }

            // Collect hits sorted by tmin descending (nearest popped first)
            let mut hits: [(usize, f64); 4] = [(0, 0.0); 4];
            let mut nhits = 0usize;
            for (i, &t) in tmin4.iter().enumerate().take(nc) {
                if hit_mask & (1 << i) != 0 {
                    hits[nhits] = (i, t);
                    nhits += 1;
                }
            }
            for a in 1..nhits {
                let key = hits[a];
                let mut b = a;
                while b > 0 && hits[b - 1].1 < key.1 {
                    hits[b] = hits[b - 1];
                    b -= 1;
                }
                hits[b] = key;
            }

            for &(slot, tmin_val) in hits.iter().take(nhits) {
                if let Some((_, best_t)) = closest {
                    if tmin_val > best_t {
                        continue;
                    }
                }

                let meta = node.child_meta[slot];
                if Bvh4Node::is_child_leaf(meta) {
                    let offset = Bvh4Node::child_offset(meta);
                    let count = Bvh4Node::child_count(meta);

                    // Gather triangle data into SoA [f64; 4] for batched test
                    let mut soa_v0 = [[0.0f64; 4]; 3];
                    let mut soa_e1 = [[0.0f64; 4]; 3];
                    let mut soa_e2 = [[0.0f64; 4]; 3];
                    let cnt = count as usize;
                    for k in 0..cnt {
                        let idx = self.indices[(offset as usize) + k] as usize;
                        for axis in 0..3 {
                            soa_v0[axis][k] = tri_v0[idx][axis];
                            soa_e1[axis][k] = tri_edge1[idx][axis];
                            soa_e2[axis][k] = tri_edge2[idx][axis];
                        }
                    }
                    // Unused slots stay zero: edge1=[0,0,0] → det≈0 → miss

                    // 4-wide Moller-Trumbore via SIMD dispatch (picks AVX2/SSE4.1 at runtime)
                    #[cfg(feature = "simd")]
                    let (t4, tri_mask) = crate::accel::simd::ray_tri4_intersect(
                        origin, direction, &soa_v0, &soa_e1, &soa_e2, min_t,
                    );

                    #[cfg(not(feature = "simd"))]
                    let (t4, tri_mask) = {
                        const BARY_EPS: f64 = 1e-6;
                        const PLUCKER_ZERO_TOL: f64 = 20.0 * f64::EPSILON;

                        let mut h = [[0.0; 4]; 3];
                        for i in 0..4 {
                            h[0][i] = direction[1] * soa_e2[2][i] - direction[2] * soa_e2[1][i];
                            h[1][i] = direction[2] * soa_e2[0][i] - direction[0] * soa_e2[2][i];
                            h[2][i] = direction[0] * soa_e2[1][i] - direction[1] * soa_e2[0][i];
                        }
                        let mut a_val = [0.0; 4];
                        for i in 0..4 {
                            a_val[i] = soa_e1[0][i] * h[0][i]
                                + soa_e1[1][i] * h[1][i]
                                + soa_e1[2][i] * h[2][i];
                        }
                        let mut f_val = [0.0; 4];
                        for i in 0..4 {
                            f_val[i] = 1.0 / a_val[i];
                        }
                        let mut sv = [[0.0; 4]; 3];
                        for i in 0..4 {
                            sv[0][i] = origin[0] - soa_v0[0][i];
                            sv[1][i] = origin[1] - soa_v0[1][i];
                            sv[2][i] = origin[2] - soa_v0[2][i];
                        }
                        let mut u_val = [0.0; 4];
                        for i in 0..4 {
                            u_val[i] = f_val[i]
                                * (sv[0][i] * h[0][i] + sv[1][i] * h[1][i] + sv[2][i] * h[2][i]);
                        }
                        let mut q = [[0.0; 4]; 3];
                        for i in 0..4 {
                            q[0][i] = sv[1][i] * soa_e1[2][i] - sv[2][i] * soa_e1[1][i];
                            q[1][i] = sv[2][i] * soa_e1[0][i] - sv[0][i] * soa_e1[2][i];
                            q[2][i] = sv[0][i] * soa_e1[1][i] - sv[1][i] * soa_e1[0][i];
                        }
                        let mut v_val = [0.0; 4];
                        for i in 0..4 {
                            v_val[i] = f_val[i]
                                * (direction[0] * q[0][i]
                                    + direction[1] * q[1][i]
                                    + direction[2] * q[2][i]);
                        }
                        let mut t_val = [0.0; 4];
                        for i in 0..4 {
                            t_val[i] = f_val[i]
                                * (soa_e2[0][i] * q[0][i]
                                    + soa_e2[1][i] * q[1][i]
                                    + soa_e2[2][i] * q[2][i]);
                        }
                        let mut m: u8 = 0;
                        for i in 0..4 {
                            if a_val[i].abs() >= PLUCKER_ZERO_TOL
                                && u_val[i] >= -BARY_EPS
                                && u_val[i] <= 1.0 + BARY_EPS
                                && v_val[i] >= -BARY_EPS
                                && u_val[i] + v_val[i] <= 1.0 + BARY_EPS
                                && t_val[i] >= min_t
                            {
                                m |= 1 << i;
                            }
                        }
                        (t_val, m)
                    };

                    // Post-filter hits and update closest
                    for (k, &t) in t4.iter().enumerate().take(cnt) {
                        if tri_mask & (1 << k) == 0 {
                            continue;
                        }
                        if let Some((_, best_t)) = closest {
                            if t >= best_t {
                                continue;
                            }
                        }
                        let prim_idx = self.indices[(offset as usize) + k];
                        if accept(prim_idx) {
                            closest = Some((prim_idx, t));
                        }
                    }
                } else if stack_ptr < 64 {
                    stack[stack_ptr] = (Bvh4Node::child_offset(meta), tmin_val);
                    stack_ptr += 1;
                } else {
                    overflow.push((Bvh4Node::child_offset(meta), tmin_val));
                }
            }
        }

        closest
    }

    /// Return the root AABB, or `None` if the BVH is empty.
    pub fn bounds(&self) -> Option<Aabb> {
        self.root_bounds
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_ray_intersect() {
        let aabb = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        // Ray from (-1, 0.5, 0.5) in +x direction
        let hit = aabb.ray_intersect([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0].map(|x: f64| 1.0 / x));
        assert!(hit.is_some());
        let (tmin, _tmax) = hit.unwrap();
        assert!((tmin - 1.0).abs() < 1e-10);

        // Ray pointing away
        let miss = aabb.ray_intersect([-1.0, 0.5, 0.5], [-1.0, 0.0, 0.0].map(|x: f64| 1.0 / x));
        assert!(miss.is_none());
    }

    #[test]
    fn test_aabb_contains_point() {
        let aabb = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(aabb.contains_point([0.5, 0.5, 0.5]));
        assert!(!aabb.contains_point([1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_bvh_build_and_ray_traverse() {
        // 4 non-overlapping boxes along x-axis
        let aabbs: Vec<[f64; 6]> = (0..4)
            .map(|i| {
                let x = i as f64 * 2.0;
                [x, 0.0, 0.0, x + 1.0, 1.0, 1.0]
            })
            .collect();

        let bvh = Bvh::build(&aabbs);

        // Ray through box 2 (x=4..5)
        let result = bvh.ray_traverse([4.5, 0.5, -1.0], [0.0, 0.0, 1.0], |prim| {
            // All primitives "hit" at distance 1.0
            if prim == 2 {
                Some(1.0)
            } else {
                None
            }
        });
        assert_eq!(result, Some((2, 1.0)));
    }

    #[test]
    fn test_bvh_point_query() {
        let aabbs: Vec<[f64; 6]> = (0..4)
            .map(|i| {
                let x = i as f64 * 2.0;
                [x, 0.0, 0.0, x + 1.0, 1.0, 1.0]
            })
            .collect();

        let bvh = Bvh::build(&aabbs);

        // Point in box 1 (x=2..3)
        let result = bvh.point_query([2.5, 0.5, 0.5], |prim| prim == 1);
        assert_eq!(result, Some(1));

        // Point outside all boxes
        let result = bvh.point_query([10.0, 0.5, 0.5], |_| true);
        assert_eq!(result, None);
    }

    #[test]
    fn test_empty_bvh() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.bounds().is_none());
        assert!(bvh
            .ray_traverse([0.0; 3], [1.0, 0.0, 0.0], |_| Some(1.0))
            .is_none());
        assert!(bvh.point_query([0.0; 3], |_| true).is_none());
    }
}
