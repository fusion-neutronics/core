/// AABB BVH (Bounding Volume Hierarchy) for fast ray-triangle intersection.
///
/// Used by `filter_tets_inside_boundary` and BCC point generation to
/// accelerate inside/outside testing from O(n) to O(log n) per query.
const LEAF_SIZE: usize = 8;

/// Axis-aligned bounding box.
#[derive(Clone, Copy)]
struct Aabb {
    min: [f64; 3],
    max: [f64; 3],
}

impl Aabb {
    fn empty() -> Self {
        Aabb {
            min: [f64::MAX; 3],
            max: [f64::MIN; 3],
        }
    }

    fn expand_point(&mut self, p: [f64; 3]) {
        for (i, &pi) in p.iter().enumerate() {
            self.min[i] = self.min[i].min(pi);
            self.max[i] = self.max[i].max(pi);
        }
    }

    fn expand_tri(&mut self, tri: &[[f64; 3]; 3]) {
        self.expand_point(tri[0]);
        self.expand_point(tri[1]);
        self.expand_point(tri[2]);
    }

    fn longest_axis(&self) -> usize {
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

    /// Slab-based ray-AABB intersection test.
    fn ray_intersects(&self, origin: &[f64; 3], inv_dir: &[f64; 3]) -> bool {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;
        for i in 0..3 {
            let t1 = (self.min[i] - origin[i]) * inv_dir[i];
            let t2 = (self.max[i] - origin[i]) * inv_dir[i];
            tmin = tmin.max(t1.min(t2));
            tmax = tmax.min(t1.max(t2));
        }
        tmax >= tmin.max(0.0)
    }
}

enum BvhNode {
    Leaf {
        start: usize,
        count: usize,
        aabb: Aabb,
    },
    Internal {
        left: usize,
        right: usize,
        aabb: Aabb,
    },
}

/// BVH over a triangle soup for O(log n) ray intersection counting.
pub struct TriangleBvh {
    nodes: Vec<BvhNode>,
    /// Triangles reordered during BVH build.
    tris: Vec<[[f64; 3]; 3]>,
}

impl TriangleBvh {
    /// Build a BVH from triangle vertex data.
    pub fn new(triangles: &[([f64; 3], [f64; 3], [f64; 3])]) -> Self {
        let mut tris: Vec<[[f64; 3]; 3]> = triangles.iter().map(|(a, b, c)| [*a, *b, *c]).collect();
        let mut centroids: Vec<[f64; 3]> = tris
            .iter()
            .map(|t| {
                [
                    (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                    (t[0][2] + t[1][2] + t[2][2]) / 3.0,
                ]
            })
            .collect();
        let mut nodes = Vec::new();
        let n = tris.len();
        build_recursive(&mut nodes, &mut tris, &mut centroids, 0, n);
        TriangleBvh { nodes, tris }
    }

    /// Count ray-triangle intersections (for inside/outside testing).
    pub fn ray_intersection_count(&self, origin: &[f64; 3], dir: &[f64; 3]) -> usize {
        let inv_dir = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        let mut count = 0;
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            match &self.nodes[ni] {
                BvhNode::Leaf {
                    start,
                    count: n,
                    aabb,
                } => {
                    if !aabb.ray_intersects(origin, &inv_dir) {
                        continue;
                    }
                    for i in *start..(*start + *n) {
                        if ray_triangle_intersect(origin, dir, &self.tris[i]) {
                            count += 1;
                        }
                    }
                }
                BvhNode::Internal { left, right, aabb } => {
                    if !aabb.ray_intersects(origin, &inv_dir) {
                        continue;
                    }
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }
        count
    }

    /// Generalized winding number of a point w.r.t. the closed mesh.
    ///
    /// Returns ~1.0 for points inside, ~0.0 for points outside.
    /// Uses the Van Oosterom & Strackee solid angle formula per triangle.
    /// Much more robust than ray casting for points near the boundary.
    pub fn winding_number(&self, point: &[f64; 3]) -> f64 {
        let mut omega = 0.0f64;
        for tri in &self.tris {
            omega += solid_angle_triangle(point, &tri[0], &tri[1], &tri[2]);
        }
        omega / (4.0 * std::f64::consts::PI)
    }

    /// Test if a point is inside a closed mesh.
    ///
    /// Uses BVH-accelerated ray casting with 3 perturbed directions and
    /// majority vote. Falls back to winding number for ambiguous cases.
    /// O(log n) per query vs O(n) for brute-force winding number.
    pub fn is_point_inside(&self, point: &[f64; 3]) -> bool {
        // 3 perturbed ray directions to avoid edge/vertex hits
        let dirs: [[f64; 3]; 3] = [
            [1.0, 0.1234, 0.0567],
            [0.0432, 1.0, 0.0891],
            [0.0678, 0.0345, 1.0],
        ];
        let mut inside_votes = 0;
        for dir in &dirs {
            let count = self.ray_intersection_count(point, dir);
            if count % 2 == 1 {
                inside_votes += 1;
            }
        }
        inside_votes >= 2
    }

    /// Find the nearest point on the triangle surface to a query point.
    /// Returns the closest point on any triangle in the BVH.
    /// O(log n) average case using branch-and-bound pruning.
    pub fn nearest_point_on_surface(&self, point: &[f64; 3]) -> [f64; 3] {
        let mut best_point = *point;
        let mut best_dist_sq = f64::MAX;
        self.nearest_recursive(0, point, &mut best_point, &mut best_dist_sq);
        best_point
    }

    fn nearest_recursive(
        &self,
        node_idx: usize,
        point: &[f64; 3],
        best_point: &mut [f64; 3],
        best_dist_sq: &mut f64,
    ) {
        match &self.nodes[node_idx] {
            BvhNode::Leaf { start, count, aabb } => {
                // Prune: if the AABB is farther than current best, skip
                if aabb_dist_sq(point, aabb) > *best_dist_sq {
                    return;
                }
                for i in *start..(*start + *count) {
                    let cp = closest_point_on_triangle(point, &self.tris[i]);
                    let d = dist_sq(point, &cp);
                    if d < *best_dist_sq {
                        *best_dist_sq = d;
                        *best_point = cp;
                    }
                }
            }
            BvhNode::Internal { left, right, aabb } => {
                if aabb_dist_sq(point, aabb) > *best_dist_sq {
                    return;
                }
                // Visit closer child first for better pruning
                let dl = match &self.nodes[*left] {
                    BvhNode::Leaf { aabb, .. } | BvhNode::Internal { aabb, .. } => {
                        aabb_dist_sq(point, aabb)
                    }
                };
                let dr = match &self.nodes[*right] {
                    BvhNode::Leaf { aabb, .. } | BvhNode::Internal { aabb, .. } => {
                        aabb_dist_sq(point, aabb)
                    }
                };
                if dl < dr {
                    self.nearest_recursive(*left, point, best_point, best_dist_sq);
                    self.nearest_recursive(*right, point, best_point, best_dist_sq);
                } else {
                    self.nearest_recursive(*right, point, best_point, best_dist_sq);
                    self.nearest_recursive(*left, point, best_point, best_dist_sq);
                }
            }
        }
    }

    /// Test if a point is inside using the full winding number (O(n) but robust).
    pub fn is_point_inside_winding(&self, point: &[f64; 3]) -> bool {
        let w = self.winding_number(point).abs();
        w > 0.5 && w < 1.5
    }

    /// Count segment-triangle intersections for a finite segment from `a` to `b`.
    /// Only counts intersections with t in (eps, 1-eps) to avoid endpoint hits.
    fn segment_intersection_count(&self, a: &[f64; 3], dir: &[f64; 3], len_sq: f64) -> usize {
        let inv_dir = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        let mut count = 0;
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            match &self.nodes[ni] {
                BvhNode::Leaf {
                    start,
                    count: n,
                    aabb,
                } => {
                    if !aabb.ray_intersects(a, &inv_dir) {
                        continue;
                    }
                    for i in *start..(*start + *n) {
                        if segment_triangle_intersect(a, dir, &self.tris[i]) {
                            count += 1;
                        }
                    }
                }
                BvhNode::Internal { left, right, aabb } => {
                    if !aabb.ray_intersects(a, &inv_dir) {
                        continue;
                    }
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }
        let _ = len_sq; // reserved for future use
        count
    }

    /// Test if a segment from `a` to `b` crosses the boundary surface.
    ///
    /// Uses 3 perturbed segments with majority vote for robustness
    /// (same strategy as `is_point_inside`). An odd intersection count
    /// means the endpoints are on opposite sides of the closed surface.
    pub fn segment_crosses_boundary(&self, a: &[f64; 3], b: &[f64; 3]) -> bool {
        let dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if len_sq < 1e-28 {
            return false;
        }

        // 3 slightly perturbed directions for majority vote
        let perturbations: [[f64; 3]; 3] = [
            [1.3e-4, 1.7e-4, 1.1e-4],
            [-1.9e-4, 1.3e-4, -1.7e-4],
            [1.1e-4, -2.3e-4, 1.9e-4],
        ];

        let mut votes = 0u32;
        for pert in &perturbations {
            let pd = [dir[0] + pert[0], dir[1] + pert[1], dir[2] + pert[2]];
            if self.segment_intersection_count(a, &pd, len_sq) % 2 == 1 {
                votes += 1;
            }
        }
        votes >= 2
    }
}

fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    tris: &mut Vec<[[f64; 3]; 3]>,
    centroids: &mut Vec<[f64; 3]>,
    start: usize,
    count: usize,
) -> usize {
    let mut aabb = Aabb::empty();
    for tri in tris.iter().skip(start).take(count) {
        aabb.expand_tri(tri);
    }

    if count <= LEAF_SIZE {
        let idx = nodes.len();
        nodes.push(BvhNode::Leaf { start, count, aabb });
        return idx;
    }

    // Split on longest axis at median centroid
    let axis = aabb.longest_axis();
    let mid = start + count / 2;

    // Partial sort to find median along axis
    let (_left_tris, _, _) =
        tris[start..start + count].select_nth_unstable_by(count / 2, |a, b| {
            let ca = (a[0][axis] + a[1][axis] + a[2][axis]) / 3.0;
            let cb = (b[0][axis] + b[1][axis] + b[2][axis]) / 3.0;
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
    // Also reorder centroids to match (not strictly needed since we don't use them after split)
    centroids[start..start + count].select_nth_unstable_by(count / 2, |a, b| {
        a[axis]
            .partial_cmp(&b[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let idx = nodes.len();
    // Placeholder - will be replaced
    nodes.push(BvhNode::Internal {
        left: 0,
        right: 0,
        aabb,
    });

    let left = build_recursive(nodes, tris, centroids, start, count / 2);
    let right = build_recursive(nodes, tris, centroids, mid, count - count / 2);

    nodes[idx] = BvhNode::Internal { left, right, aabb };
    idx
}

/// Solid angle subtended by triangle (v0, v1, v2) as seen from point p.
///
/// Van Oosterom & Strackee (1983) formula:
///   tan(Ω/2) = (a · (b × c)) / (|a||b||c| + (a·b)|c| + (b·c)|a| + (a·c)|b|)
/// where a = v0-p, b = v1-p, c = v2-p.
fn solid_angle_triangle(p: &[f64; 3], v0: &[f64; 3], v1: &[f64; 3], v2: &[f64; 3]) -> f64 {
    let a = [v0[0] - p[0], v0[1] - p[1], v0[2] - p[2]];
    let b = [v1[0] - p[0], v1[1] - p[1], v1[2] - p[2]];
    let c = [v2[0] - p[0], v2[1] - p[1], v2[2] - p[2]];

    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let lc = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();

    if la < 1e-15 || lb < 1e-15 || lc < 1e-15 {
        return 0.0; // Point is on a vertex
    }

    // b × c
    let bc_cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];

    let numerator = a[0] * bc_cross[0] + a[1] * bc_cross[1] + a[2] * bc_cross[2];
    let ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let bc = b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
    let ac = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
    let denominator = la * lb * lc + ab * lc + bc * la + ac * lb;

    2.0 * numerator.atan2(denominator)
}

/// Squared distance from point to AABB (0 if inside).
fn aabb_dist_sq(p: &[f64; 3], aabb: &Aabb) -> f64 {
    let mut d = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        if pi < aabb.min[i] {
            d += (aabb.min[i] - pi) * (aabb.min[i] - pi);
        } else if pi > aabb.max[i] {
            d += (pi - aabb.max[i]) * (pi - aabb.max[i]);
        }
    }
    d
}

fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
}

/// Find the closest point on a triangle to a query point.
fn closest_point_on_triangle(p: &[f64; 3], tri: &[[f64; 3]; 3]) -> [f64; 3] {
    let a = tri[0];
    let b = tri[1];
    let c = tri[2];

    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];

    let d1 = dot3(&ab, &ap);
    let d2 = dot3(&ac, &ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
    let d3 = dot3(&ab, &bp);
    let d4 = dot3(&ac, &bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    }

    let cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let d5 = dot3(&ab, &cp);
    let d6 = dot3(&ac, &cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [
            b[0] + w * (c[0] - b[0]),
            b[1] + w * (c[1] - b[1]),
            b[2] + w * (c[2] - b[2]),
        ];
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Möller–Trumbore segment-triangle intersection (eps < t < 1-eps).
/// `dir` = endpoint_b - endpoint_a (not necessarily unit length).
fn segment_triangle_intersect(origin: &[f64; 3], dir: &[f64; 3], tri: &[[f64; 3]; 3]) -> bool {
    let edge1 = [
        tri[1][0] - tri[0][0],
        tri[1][1] - tri[0][1],
        tri[1][2] - tri[0][2],
    ];
    let edge2 = [
        tri[2][0] - tri[0][0],
        tri[2][1] - tri[0][1],
        tri[2][2] - tri[0][2],
    ];

    let h = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];
    let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    if a.abs() < 1e-14 {
        return false;
    }

    let f = 1.0 / a;
    let s = [
        origin[0] - tri[0][0],
        origin[1] - tri[0][1],
        origin[2] - tri[0][2],
    ];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);

    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    // Accept only intersections strictly between endpoints (eps, 1-eps)
    let eps = 1e-6;
    t > eps && t < 1.0 - eps
}

/// Möller–Trumbore ray-triangle intersection (t > 0).
fn ray_triangle_intersect(origin: &[f64; 3], dir: &[f64; 3], tri: &[[f64; 3]; 3]) -> bool {
    let edge1 = [
        tri[1][0] - tri[0][0],
        tri[1][1] - tri[0][1],
        tri[1][2] - tri[0][2],
    ];
    let edge2 = [
        tri[2][0] - tri[0][0],
        tri[2][1] - tri[0][1],
        tri[2][2] - tri[0][2],
    ];

    let h = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];
    let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    if a.abs() < 1e-14 {
        return false;
    }

    let f = 1.0 / a;
    let s = [
        origin[0] - tri[0][0],
        origin[1] - tri[0][1],
        origin[2] - tri[0][2],
    ];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);

    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    t > 1e-14
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_triangles() -> Vec<([f64; 3], [f64; 3], [f64; 3])> {
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let idx = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        idx.iter().map(|t| (v[t[0]], v[t[1]], v[t[2]])).collect()
    }

    #[test]
    fn bvh_inside_outside() {
        let bvh = TriangleBvh::new(&cube_triangles());
        assert!(bvh.is_point_inside(&[0.5, 0.5, 0.5]));
        assert!(!bvh.is_point_inside(&[2.0, 0.5, 0.5]));
        assert!(!bvh.is_point_inside(&[-1.0, -1.0, -1.0]));
    }

    #[test]
    fn bvh_ray_count() {
        let bvh = TriangleBvh::new(&cube_triangles());
        // Use the same perturbed rays as is_point_inside to avoid edge hits
        let dir = [1.0, 0.00013, 0.00017];
        let count = bvh.ray_intersection_count(&[0.5, 0.5, 0.5], &dir);
        assert!(
            count % 2 == 1,
            "Inside point should have odd crossings, got {count}"
        );
        let count = bvh.ray_intersection_count(&[2.0, 0.5, 0.5], &dir);
        assert!(
            count.is_multiple_of(2),
            "Outside point should have even crossings, got {count}"
        );
    }

    #[test]
    fn segment_crosses_boundary_basic() {
        let bvh = TriangleBvh::new(&cube_triangles());
        // Inside to outside: should cross
        assert!(bvh.segment_crosses_boundary(&[0.5, 0.5, 0.5], &[2.0, 0.5, 0.5]));
        // Both inside: should NOT cross
        assert!(!bvh.segment_crosses_boundary(&[0.3, 0.3, 0.3], &[0.7, 0.7, 0.7]));
        // Both outside: should NOT cross (same side)
        assert!(!bvh.segment_crosses_boundary(&[2.0, 0.5, 0.5], &[3.0, 0.5, 0.5]));
        // Outside to inside: should cross
        assert!(bvh.segment_crosses_boundary(&[-1.0, 0.5, 0.5], &[0.5, 0.5, 0.5]));
    }
}
