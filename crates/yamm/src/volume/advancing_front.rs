/// Advancing-front tetrahedral mesher.
///
/// Grows equilateral tetrahedra inward from the boundary surface, producing
/// uniform-quality tets with edges near `target_edge_length`.
///
/// Algorithm (simplified Netgen-style):
/// 1. Init front from boundary triangles with inward normals
/// 2. Pick lowest quality-class, outermost face from priority queue
/// 3. Place ideal equilateral tet apex; validate via BVH + KD-tree
/// 4. Fallback to nearby existing point if ideal fails
/// 5. Front collision: matching faces zip up automatically
/// 6. Star-shaped closure for endgame cavities
use super::dethash::{HashMap, HashSet};
use std::collections::BinaryHeap;

use kiddo::SquaredEuclidean;
use smallvec::SmallVec;

use super::aabb_bvh::TriangleBvh;
use super::optimize;
use super::predicates3d::{self, dist_sq_3d, orient_3d};
use super::types::{BoundaryRecoveryStats, VolumeInput, VolumeOutput};
use crate::error::{MesherError, Result};

/// Height of an equilateral tetrahedron with edge length 1.
const TET_HEIGHT: f64 = 0.8164965809277261; // sqrt(2/3)

// ── Data structures ─────────────────────────────────────────────────────

/// Sorted face key for HashMap lookups.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct FaceKey([usize; 3]);

impl FaceKey {
    fn new(a: usize, b: usize, c: usize) -> Self {
        let mut arr = [a, b, c];
        arr.sort();
        FaceKey(arr)
    }
}

/// Sorted edge key.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct EdgeKey(usize, usize);

impl EdgeKey {
    fn new(a: usize, b: usize) -> Self {
        EdgeKey(a.min(b), a.max(b))
    }
}

/// A triangle on the advancing front.
#[derive(Clone, Debug)]
struct FrontFace {
    verts: [usize; 3],
    normal: [f64; 3],
    quality_class: u8,
}

/// Priority queue entry: process low quality_class first, then small area.
#[derive(Clone, Debug)]
struct PqEntry {
    quality_class: u8,
    area_x1000: u32,
    key: FaceKey,
}

impl PartialEq for PqEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for PqEntry {}
impl PartialOrd for PqEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PqEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: we want min quality_class and min area first
        (other.quality_class, other.area_x1000).cmp(&(self.quality_class, self.area_x1000))
    }
}

/// The main advancing-front state.
struct AdvancingFront {
    vertices: Vec<[f64; 3]>,
    n_boundary: usize,
    front: HashMap<FaceKey, FrontFace>,
    tetrahedra: Vec<[usize; 4]>,
    kdtree: kiddo::float::kdtree::KdTree<f64, u64, 3, 256, u32>,
    edge_faces: HashMap<EdgeKey, SmallVec<[FaceKey; 2]>>,
    bvh: TriangleBvh,
    target_h: f64,
    pq: BinaryHeap<PqEntry>,
    /// Tracks how many tets are adjacent to each face (0, 1, or 2).
    face_tet_count: HashMap<FaceKey, u8>,
}

// ── Quality class thresholds ────────────────────────────────────────────

fn edge_range(qc: u8, h: f64) -> (f64, f64) {
    match qc {
        0..=1 => (0.4 * h, 1.8 * h),
        2 => (0.3 * h, 2.0 * h),
        3 => (0.2 * h, 2.5 * h),
        _ => (0.1 * h, 3.0 * h),
    }
}

fn min_quality(qc: u8) -> f64 {
    match qc {
        0..=1 => 0.10,
        2 => 0.05,
        3 => 0.01,
        _ => 1e-6,
    }
}

const MAX_QUALITY_CLASS: u8 = 8;
const STAR_CLOSE_CLASS: u8 = 4;
const MAX_STAR_FACES: usize = 30;

// ── Helpers ─────────────────────────────────────────────────────────────

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-15 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn tri_area_sq(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    0.25 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
}

fn centroid3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

// ── Implementation ──────────────────────────────────────────────────────

impl AdvancingFront {
    fn new(input: &VolumeInput) -> Self {
        let n_boundary = input.boundary_vertices.len();
        let vertices = input.boundary_vertices.clone();
        let target_h = input.target_edge_length;

        // Build BVH for inside/outside tests
        let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = input
            .boundary_triangles
            .iter()
            .map(|tri| (vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]))
            .collect();
        let bvh = TriangleBvh::new(&tri_data);

        // Build KD-tree with boundary vertices
        let mut kdtree = kiddo::float::kdtree::KdTree::<f64, u64, 3, 256, u32>::new();
        for (i, v) in vertices.iter().enumerate() {
            kdtree.add(v, i as u64);
        }

        // Initialize front from boundary triangles with inward normals
        let mut front = HashMap::default();
        let mut edge_faces: HashMap<EdgeKey, SmallVec<[FaceKey; 2]>> = HashMap::default();
        let mut pq = BinaryHeap::new();

        for tri in &input.boundary_triangles {
            let [a, b, c] = *tri;
            let va = vertices[a];
            let vb = vertices[b];
            let vc = vertices[c];

            // Geometric normal
            let n = normalize(cross(sub(vb, va), sub(vc, va)));

            // Check if normal points inward: offset centroid along normal, test inside
            let cen = centroid3(va, vb, vc);
            let test_pt = [
                cen[0] + n[0] * target_h * 0.01,
                cen[1] + n[1] * target_h * 0.01,
                cen[2] + n[2] * target_h * 0.01,
            ];
            let inward = if bvh.is_point_inside(&test_pt) {
                n
            } else {
                [-n[0], -n[1], -n[2]]
            };

            let key = FaceKey::new(a, b, c);

            // Orient verts so that orient_3d(a,b,c, centroid+inward) > 0
            let mut verts = [a, b, c];
            let apex_test = [
                cen[0] + inward[0] * target_h * 0.1,
                cen[1] + inward[1] * target_h * 0.1,
                cen[2] + inward[2] * target_h * 0.1,
            ];
            if orient_3d(
                vertices[verts[0]],
                vertices[verts[1]],
                vertices[verts[2]],
                apex_test,
            ) < 0.0
            {
                verts.swap(0, 1);
            }

            let ff = FrontFace {
                verts,
                normal: inward,
                quality_class: 1,
            };

            let area = tri_area_sq(va, vb, vc).sqrt();
            pq.push(PqEntry {
                quality_class: 1,
                area_x1000: (area * 1000.0).min(u32::MAX as f64) as u32,
                key,
            });

            front.insert(key, ff);

            // Edge adjacency
            for i in 0..3 {
                let ek = EdgeKey::new(verts[i], verts[(i + 1) % 3]);
                edge_faces.entry(ek).or_default().push(key);
            }
        }

        AdvancingFront {
            vertices,
            n_boundary,
            front,
            tetrahedra: Vec::new(),
            kdtree,
            edge_faces,
            bvh,
            target_h,
            pq,
            face_tet_count: HashMap::default(),
        }
    }

    /// Add a vertex to vertices + KD-tree and return its index.
    fn add_vertex(&mut self, p: [f64; 3]) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(p);
        self.kdtree.add(&p, idx as u64);
        idx
    }

    /// Compute the ideal equilateral tet apex for a front face.
    fn ideal_point(&self, ff: &FrontFace) -> [f64; 3] {
        let a = self.vertices[ff.verts[0]];
        let b = self.vertices[ff.verts[1]];
        let c = self.vertices[ff.verts[2]];
        let cen = centroid3(a, b, c);
        let h = self.target_h * TET_HEIGHT;
        [
            cen[0] + ff.normal[0] * h,
            cen[1] + ff.normal[1] * h,
            cen[2] + ff.normal[2] * h,
        ]
    }

    /// Check if a new tet [face.verts + apex] is valid.
    fn validate_tet(&self, ff: &FrontFace, apex: usize) -> bool {
        let (a, b, c) = (ff.verts[0], ff.verts[1], ff.verts[2]);
        let pa = self.vertices[a];
        let pb = self.vertices[b];
        let pc = self.vertices[c];
        let pp = self.vertices[apex];

        // Positive volume
        if orient_3d(pa, pb, pc, pp) <= 0.0 {
            return false;
        }

        // Edge lengths in range
        let qc = ff.quality_class;
        let (lo, hi) = edge_range(qc, self.target_h);
        let lo_sq = lo * lo;
        let hi_sq = hi * hi;

        for &v in &[a, b, c] {
            let d = dist_sq_3d(self.vertices[v], pp);
            if d < lo_sq || d > hi_sq {
                return false;
            }
        }

        // Base edges also in range (relaxed - boundary edges may already be set)
        let base_edges = [(a, b), (b, c), (a, c)];
        for (u, v) in base_edges {
            let d = dist_sq_3d(self.vertices[u], self.vertices[v]);
            if d > hi_sq * 1.5 {
                return false;
            } // very relaxed for base
        }

        // Quality check
        let q = optimize::tet_quality(pa, pb, pc, pp);
        if q < min_quality(qc) {
            return false;
        }

        true
    }

    /// Find the best existing nearby point to connect to a front face.
    fn find_existing_point(&self, ff: &FrontFace) -> Option<usize> {
        let ideal = self.ideal_point(ff);
        let search_r = self.target_h * 1.5;

        let neighbors = self
            .kdtree
            .within::<SquaredEuclidean>(&ideal, search_r * search_r);

        let face_verts: HashSet<usize> = ff.verts.iter().copied().collect();
        let mut best: Option<(usize, f64)> = None;

        for nb in &neighbors {
            let idx = nb.item as usize;
            if idx >= self.vertices.len() {
                continue;
            } // stale KD-tree entry
            if face_verts.contains(&idx) {
                continue;
            }

            // Must be on the inward side
            let pa = self.vertices[ff.verts[0]];
            let pb = self.vertices[ff.verts[1]];
            let pc = self.vertices[ff.verts[2]];
            let pp = self.vertices[idx];
            if orient_3d(pa, pb, pc, pp) <= 0.0 {
                continue;
            }

            // Edge length check
            let (lo, hi) = edge_range(ff.quality_class, self.target_h);
            let lo_sq = lo * lo;
            let hi_sq = hi * hi;
            let edges_ok = ff.verts.iter().all(|&v| {
                let d = dist_sq_3d(self.vertices[v], pp);
                d >= lo_sq && d <= hi_sq
            });
            if !edges_ok {
                continue;
            }

            // Quality
            let q = optimize::tet_quality(pa, pb, pc, pp);
            if q < min_quality(ff.quality_class) {
                continue;
            }

            // Score: prefer closest to ideal distance
            let ideal_dist_sq = (self.target_h * TET_HEIGHT).powi(2);
            let cen = centroid3(pa, pb, pc);
            let actual_dist_sq = dist_sq_3d(cen, pp);
            let score = (actual_dist_sq - ideal_dist_sq).abs();

            if best.is_none() || score < best.unwrap().1 {
                best = Some((idx, score));
            }
        }

        best.map(|(idx, _)| idx)
    }

    /// Remove a face from the front and update edge adjacency.
    fn remove_front_face(&mut self, key: &FaceKey) {
        if let Some(ff) = self.front.remove(key) {
            for i in 0..3 {
                let ek = EdgeKey::new(ff.verts[i], ff.verts[(i + 1) % 3]);
                if let Some(faces) = self.edge_faces.get_mut(&ek) {
                    faces.retain(|k| k != key);
                    if faces.is_empty() {
                        self.edge_faces.remove(&ek);
                    }
                }
            }
        }
    }

    /// Add a face to the front and update edge adjacency.
    fn add_front_face(&mut self, key: FaceKey, ff: FrontFace) {
        let area = tri_area_sq(
            self.vertices[ff.verts[0]],
            self.vertices[ff.verts[1]],
            self.vertices[ff.verts[2]],
        )
        .sqrt();

        self.pq.push(PqEntry {
            quality_class: ff.quality_class,
            area_x1000: (area * 1000.0).min(u32::MAX as f64) as u32,
            key,
        });

        for i in 0..3 {
            let ek = EdgeKey::new(ff.verts[i], ff.verts[(i + 1) % 3]);
            self.edge_faces.entry(ek).or_default().push(key);
        }

        self.front.insert(key, ff);
    }

    /// Check if a tet would overlap with an already-meshed region.
    /// Increment the tet count for a face. Returns the new count.
    fn inc_face_tet_count(&mut self, key: FaceKey) -> u8 {
        let count = self.face_tet_count.entry(key).or_insert(0);
        *count += 1;
        *count
    }

    /// Create a tet and update the front.
    ///
    /// Uses face_tet_count to track adjacency:
    /// - 0 → face never seen (shouldn't happen for non-base faces)
    /// - 1 → face has 1 tet, should be on the front
    /// - 2 → face has 2 tets, fully internal → remove from front
    fn create_tet(&mut self, base_key: FaceKey, base_ff: FrontFace, apex: usize) {
        let [a, b, c] = base_ff.verts;
        let pa = self.vertices[a];
        let pb = self.vertices[b];
        let pc = self.vertices[c];
        let pp = self.vertices[apex];

        // Ensure positive orientation
        let mut tet = [a, b, c, apex];
        if orient_3d(pa, pb, pc, pp) < 0.0 {
            tet.swap(0, 1);
        }
        self.tetrahedra.push(tet);

        // Remove base face from front (it now has a tet on BOTH sides if it
        // was a boundary face with one tet, or its second tet if internal)
        self.remove_front_face(&base_key);

        // Process the 4 faces of this tet
        let all_faces = [
            FaceKey::new(a, b, c),    // base face
            FaceKey::new(b, c, apex), // opposite a
            FaceKey::new(a, c, apex), // opposite b
            FaceKey::new(a, b, apex), // opposite c
        ];

        let next_qc = base_ff
            .quality_class
            .saturating_add(1)
            .min(MAX_QUALITY_CLASS);
        let tet_cen = [
            (pa[0] + pb[0] + pc[0] + pp[0]) / 4.0,
            (pa[1] + pb[1] + pc[1] + pp[1]) / 4.0,
            (pa[2] + pb[2] + pc[2] + pp[2]) / 4.0,
        ];

        for key in &all_faces {
            let count = self.inc_face_tet_count(*key);

            if count >= 2 {
                // Face now has 2 adjacent tets → fully internal, remove from front
                self.remove_front_face(key);
            } else if count == 1 && *key != base_key {
                // Face has 1 tet → it's a new front face (unless it's the base
                // which we already removed)
                if !self.front.contains_key(key) {
                    // Compute inward normal (pointing away from the tet just formed)
                    let fv = key.0;
                    let va = self.vertices[fv[0]];
                    let vb = self.vertices[fv[1]];
                    let vc = self.vertices[fv[2]];
                    let n_raw = cross(sub(vb, va), sub(vc, va));
                    let n = normalize(n_raw);

                    let face_cen = centroid3(va, vb, vc);
                    let to_face = sub(face_cen, tet_cen);
                    let inward = if dot(n, to_face) > 0.0 {
                        [-n[0], -n[1], -n[2]]
                    } else {
                        n
                    };

                    let mut oriented = fv;
                    let test_pt = [
                        face_cen[0] + inward[0] * self.target_h * 0.01,
                        face_cen[1] + inward[1] * self.target_h * 0.01,
                        face_cen[2] + inward[2] * self.target_h * 0.01,
                    ];
                    if orient_3d(
                        self.vertices[oriented[0]],
                        self.vertices[oriented[1]],
                        self.vertices[oriented[2]],
                        test_pt,
                    ) < 0.0
                    {
                        oriented.swap(0, 1);
                    }

                    let ff = FrontFace {
                        verts: oriented,
                        normal: inward,
                        quality_class: next_qc,
                    };
                    self.add_front_face(*key, ff);
                }
            }
        }
    }

    /// Try star-shaped closure: if a connected group of front faces is small,
    /// compute an interior point and create radial tets (Netgen approach).
    fn try_star_close(&mut self, seed_key: &FaceKey) -> bool {
        // Collect connected face group via BFS through shared edges
        let mut group_keys: Vec<FaceKey> = Vec::new();
        let mut visited: HashSet<FaceKey> = HashSet::default();
        let mut stack = vec![*seed_key];

        while let Some(k) = stack.pop() {
            if !visited.insert(k) {
                continue;
            }
            if !self.front.contains_key(&k) {
                continue;
            }
            group_keys.push(k);
            if group_keys.len() > MAX_STAR_FACES {
                return false;
            }

            let ff = &self.front[&k];
            for i in 0..3 {
                let ek = EdgeKey::new(ff.verts[i], ff.verts[(i + 1) % 3]);
                if let Some(faces) = self.edge_faces.get(&ek) {
                    for fk in faces {
                        if !visited.contains(fk) {
                            stack.push(*fk);
                        }
                    }
                }
            }
        }

        if group_keys.len() < 4 {
            return false;
        }

        // Collect unique vertices
        let mut group_verts: HashSet<usize> = HashSet::default();
        for k in &group_keys {
            if let Some(ff) = self.front.get(k) {
                for &v in &ff.verts {
                    group_verts.insert(v);
                }
            }
        }

        // Special case: 4 faces, 4 vertices → single tet
        if group_keys.len() == 4 && group_verts.len() == 4 {
            let v: Vec<usize> = group_verts.iter().copied().collect();
            let vol = predicates3d::tet_volume(
                self.vertices[v[0]],
                self.vertices[v[1]],
                self.vertices[v[2]],
                self.vertices[v[3]],
            );
            if vol.abs() > 1e-15 {
                let mut tet = [v[0], v[1], v[2], v[3]];
                if vol < 0.0 {
                    tet.swap(2, 3);
                }
                self.tetrahedra.push(tet);
                for k in &group_keys {
                    self.remove_front_face(k);
                }
                return true;
            }
        }

        // Compute interior point (centroid of group vertices)
        let nv = group_verts.len() as f64;
        let mut center = [0.0, 0.0, 0.0];
        for &vi in &group_verts {
            let p = self.vertices[vi];
            center[0] += p[0];
            center[1] += p[1];
            center[2] += p[2];
        }
        center[0] /= nv;
        center[1] /= nv;
        center[2] /= nv;

        // Verify center is inside boundary
        if !self.bvh.is_point_inside(&center) {
            return false;
        }

        // Verify all tets from group faces to center have positive volume
        let keys_copy: Vec<FaceKey> = group_keys.clone();
        for k in &keys_copy {
            if let Some(ff) = self.front.get(k) {
                let vol = orient_3d(
                    self.vertices[ff.verts[0]],
                    self.vertices[ff.verts[1]],
                    self.vertices[ff.verts[2]],
                    center,
                );
                if vol <= 0.0 {
                    return false;
                }
            }
        }

        // All valid: create center vertex and radial tets
        let center_idx = self.add_vertex(center);
        for k in &keys_copy {
            if let Some(ff) = self.front.get(k).cloned() {
                let mut tet = [ff.verts[0], ff.verts[1], ff.verts[2], center_idx];
                if orient_3d(
                    self.vertices[tet[0]],
                    self.vertices[tet[1]],
                    self.vertices[tet[2]],
                    self.vertices[tet[3]],
                ) < 0.0
                {
                    tet.swap(0, 1);
                }
                self.tetrahedra.push(tet);
            }
        }
        for k in &keys_copy {
            self.remove_front_face(k);
        }

        true
    }

    /// Run the full advancing-front loop with vertex merging.
    ///
    /// Key difference from naive AF: when the ideal point is near an existing
    /// vertex (within merge_radius), SNAP to that vertex. This ensures fronts
    /// from opposite sides share vertices, enabling face-based zipping via
    /// face_tet_count.
    fn run(&mut self) {
        let max_iters = (self.front.len() as u64) * 100;
        let merge_radius_sq = (0.4 * self.target_h).powi(2);
        let mut iters = 0u64;

        while !self.front.is_empty() && iters < max_iters {
            iters += 1;

            let entry = loop {
                match self.pq.pop() {
                    None => return,
                    Some(e) => {
                        if self.front.contains_key(&e.key) {
                            break e;
                        }
                    }
                }
            };

            let key = entry.key;
            let ff = match self.front.get(&key) {
                Some(f) => f.clone(),
                None => continue,
            };

            // Check if this face already has 2 adjacent tets (fully internal)
            if let Some(&count) = self.face_tet_count.get(&key) {
                if count >= 2 {
                    self.remove_front_face(&key);
                    continue;
                }
            }

            let ideal = self.ideal_point(&ff);
            let mut formed = false;

            // Strategy 1: SNAP to existing nearby vertex.
            // This is the key to front collision: vertices get shared → faces zip.
            if !formed {
                let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&ideal);
                let snap_idx = nearest.item as usize;

                if nearest.distance < merge_radius_sq
                    && snap_idx < self.vertices.len()
                    && !ff.verts.contains(&snap_idx)
                    && self.validate_tet(&ff, snap_idx)
                {
                    self.create_tet(key, ff.clone(), snap_idx);
                    formed = true;
                }
            }

            // Strategy 2: Try existing point with broader search
            if !formed {
                if let Some(idx) = self.find_existing_point(&ff) {
                    if self.validate_tet(&ff, idx) {
                        self.create_tet(key, ff.clone(), idx);
                        formed = true;
                    }
                }
            }

            // Strategy 3: Create new vertex at ideal position
            if !formed && self.bvh.is_point_inside(&ideal) {
                let nearest = self.kdtree.nearest_one::<SquaredEuclidean>(&ideal);
                if nearest.distance >= merge_radius_sq {
                    let tentative_idx = self.vertices.len();
                    self.vertices.push(ideal);
                    if self.validate_tet(&ff, tentative_idx) {
                        self.kdtree.add(&ideal, tentative_idx as u64);
                        self.create_tet(key, ff.clone(), tentative_idx);
                        formed = true;
                    } else {
                        self.vertices.pop();
                    }
                }
            }

            if !formed {
                let new_qc = ff.quality_class.saturating_add(1);
                if new_qc >= STAR_CLOSE_CLASS && self.try_star_close(&key) {
                    continue;
                }
                if new_qc <= MAX_QUALITY_CLASS {
                    if let Some(ff_mut) = self.front.get_mut(&key) {
                        ff_mut.quality_class = new_qc;
                    }
                    let area = tri_area_sq(
                        self.vertices[ff.verts[0]],
                        self.vertices[ff.verts[1]],
                        self.vertices[ff.verts[2]],
                    )
                    .sqrt();
                    self.pq.push(PqEntry {
                        quality_class: new_qc,
                        area_x1000: (area * 1000.0).min(u32::MAX as f64) as u32,
                        key,
                    });
                }
            }
        }
    }
}

/// Test if point p is strictly inside tetrahedron t.
fn point_in_tet_orient(p: &[f64; 3], tet: &[usize; 4], verts: &[[f64; 3]]) -> bool {
    let pp = [p[0], p[1], p[2]];
    let a = verts[tet[0]];
    let b = verts[tet[1]];
    let c = verts[tet[2]];
    let d = verts[tet[3]];
    let eps = 1e-10;
    let o0 = orient_3d(pp, b, c, d);
    let o1 = orient_3d(a, pp, c, d);
    let o2 = orient_3d(a, b, pp, d);
    let o3 = orient_3d(a, b, c, pp);
    (o0 > eps && o1 > eps && o2 > eps && o3 > eps)
        || (o0 < -eps && o1 < -eps && o2 < -eps && o3 < -eps)
}

// ── Public entry point ──────────────────────────────────────────────────

/// Mesh a volume using advancing-front with vertex merging.
///
/// Grows uniform equilateral tets inward from the boundary. Vertex merging
/// ensures fronts from opposite sides share vertices, enabling face zipping
/// via face_tet_count to prevent overlaps.
pub fn mesh_advancing_front(input: &VolumeInput) -> Result<VolumeOutput> {
    if input.boundary_vertices.len() < 4 {
        return Err(MesherError::InvalidInput(
            "Need at least 4 boundary vertices".into(),
        ));
    }
    if input.boundary_triangles.is_empty() {
        return Err(MesherError::InvalidInput(
            "Need at least 1 boundary triangle".into(),
        ));
    }

    let mut af = AdvancingFront::new(input);
    af.run();

    // Remove overlapping tets via centroid-in-tet test with spatial grid.
    // When overlap is detected, keep the tet with better quality.
    let mut tets = af.tetrahedra;
    {
        let verts = &af.vertices;
        let n_tets = tets.len();
        if n_tets > 1 {
            let centroids: Vec<[f64; 3]> = tets
                .iter()
                .map(|t| {
                    [
                        (verts[t[0]][0] + verts[t[1]][0] + verts[t[2]][0] + verts[t[3]][0]) / 4.0,
                        (verts[t[0]][1] + verts[t[1]][1] + verts[t[2]][1] + verts[t[3]][1]) / 4.0,
                        (verts[t[0]][2] + verts[t[1]][2] + verts[t[2]][2] + verts[t[3]][2]) / 4.0,
                    ]
                })
                .collect();

            let h = af.target_h;
            let inv_cell = 1.0 / (h * 2.0);
            let cell_key = |p: &[f64; 3]| -> (i64, i64, i64) {
                (
                    (p[0] * inv_cell).floor() as i64,
                    (p[1] * inv_cell).floor() as i64,
                    (p[2] * inv_cell).floor() as i64,
                )
            };
            let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::default();
            for (i, c) in centroids.iter().enumerate() {
                grid.entry(cell_key(c)).or_default().push(i);
            }

            let qualities: Vec<f64> = tets
                .iter()
                .map(|t| optimize::tet_quality(verts[t[0]], verts[t[1]], verts[t[2]], verts[t[3]]))
                .collect();

            let mut to_remove: HashSet<usize> = HashSet::default();
            for i in 0..n_tets {
                if to_remove.contains(&i) {
                    continue;
                }
                let (cx, cy, cz) = cell_key(&centroids[i]);
                for dx in -1..=1_i64 {
                    for dy in -1..=1_i64 {
                        for dz in -1..=1_i64 {
                            if let Some(js) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &j in js {
                                    if j <= i || to_remove.contains(&j) {
                                        continue;
                                    }
                                    let ci_in_j =
                                        point_in_tet_orient(&centroids[i], &tets[j], verts);
                                    let cj_in_i =
                                        point_in_tet_orient(&centroids[j], &tets[i], verts);
                                    // Only flag MUTUAL containment (near-total overlap)
                                    if ci_in_j && cj_in_i {
                                        // Keep the better quality tet
                                        if qualities[i] >= qualities[j] {
                                            to_remove.insert(j);
                                        } else {
                                            to_remove.insert(i);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !to_remove.is_empty() {
                let mut idx = 0;
                tets.retain(|_| {
                    let keep = !to_remove.contains(&idx);
                    idx += 1;
                    keep
                });
            }
        }
    }

    finalize(af.vertices, tets, af.n_boundary, af.front.len())
}

fn finalize(
    mut vertices: Vec<[f64; 3]>,
    mut tets: Vec<[usize; 4]>,
    n_boundary: usize,
    remaining_front: usize,
) -> Result<VolumeOutput> {
    // Laplacian smoothing (boundary vertices fixed)
    if !tets.is_empty() {
        optimize::laplacian_smooth(&mut vertices, &tets, n_boundary, 3);
    }

    // Fix orientation
    for t in &mut tets {
        let vol = predicates3d::tet_volume(
            vertices[t[0]],
            vertices[t[1]],
            vertices[t[2]],
            vertices[t[3]],
        );
        if vol < 0.0 {
            t.swap(2, 3);
        }
    }

    // Remove degenerate tets
    tets.retain(|t| {
        predicates3d::tet_volume(
            vertices[t[0]],
            vertices[t[1]],
            vertices[t[2]],
            vertices[t[3]],
        )
        .abs()
            > 1e-15
    });

    let interior_vertices = vertices[n_boundary..].to_vec();

    Ok(VolumeOutput {
        interior_vertices,
        tetrahedra: tets,
        boundary_recovery_stats: BoundaryRecoveryStats {
            edges_recovered: 0,
            edges_failed: 0,
            faces_recovered: 0,
            faces_failed: remaining_front,
            failed_edges: vec![],
            failed_faces: vec![],
            tets_cut_by_boundary: 0,
        },
    })
}

// ── Tests ───────────────────────────────────────────────────────────────

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube boundary: 8 vertices, 12 triangles.
    fn unit_cube() -> VolumeInput {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let tris = vec![
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
        VolumeInput {
            boundary_vertices: verts,
            boundary_triangles: tris,
            target_edge_length: 0.5,
        }
    }

    #[test]
    fn advancing_front_unit_cube() {
        let input = unit_cube();
        let output = mesh_advancing_front(&input).unwrap();

        assert!(!output.tetrahedra.is_empty(), "Should produce tets, got 0");

        // All tets should have positive volume
        let mut all_verts = input.boundary_vertices.clone();
        all_verts.extend_from_slice(&output.interior_vertices);

        for t in &output.tetrahedra {
            let vol = predicates3d::tet_volume(
                all_verts[t[0]],
                all_verts[t[1]],
                all_verts[t[2]],
                all_verts[t[3]],
            );
            assert!(vol > 0.0, "Tet has non-positive volume: {vol}");
        }
    }

    #[test]
    fn advancing_front_cube_edge_uniformity() {
        let input = unit_cube();
        let output = mesh_advancing_front(&input).unwrap();

        let mut all_verts = input.boundary_vertices.clone();
        all_verts.extend_from_slice(&output.interior_vertices);

        let target = input.target_edge_length;
        let mut max_edge = 0.0_f64;
        let mut min_edge = f64::MAX;

        for t in &output.tetrahedra {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let d = dist_sq_3d(all_verts[t[i]], all_verts[t[j]]).sqrt();
                    max_edge = max_edge.max(d);
                    min_edge = min_edge.min(d);
                }
            }
        }

        // Check that most edges are reasonable (advancing front may have
        // some short edges from endgame closure, but should be rare)
        assert!(
            max_edge < target * 4.0,
            "Max edge {max_edge:.3} exceeds 4x target {target}"
        );
        eprintln!("Edge range: [{min_edge:.3}, {max_edge:.3}], target={target}");
    }
}
