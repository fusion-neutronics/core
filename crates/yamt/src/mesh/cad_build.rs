//! Pure mesh-construction kernels used by the CAD export pipeline.
//!
//! These were previously per-element Python loops in `yamc.cad`; they are the
//! compiled core the Python side now calls. `weld_vertices` merges the copies
//! the faces of a model hold of each point they share into a watertight,
//! index-shared mesh; `tet_adjacency_flat` exposes the adjacency builder on
//! the flat arrays the Arrow writer uses.

use std::collections::{BTreeSet, HashMap};

use crate::mesh::topology::build_tet_adjacency;
use crate::types::VertexId;

/// Weld the per-face triangulations of a model into a watertight boundary.
///
/// `vertex_faces[v]` is the face vertex `v` was meshed on, one per vertex, and
/// each triangle indexes the vertices of one face. Faces that meet each hold
/// a copy of every point along their common edges, and the weld merges those
/// copies: the vertices on the open boundary of their faces' triangles within
/// `rel_tol` times the extent of the used vertices of each other. A vertex
/// inside its face's triangles is no copy; it merges only with vertices at
/// exactly its position.
///
/// Two vertices of one face merge only within `seam_rel_tol` times the
/// extent, as the copies of a point on the face's seam or at a pole do (they
/// differ by rounding); further apart they are two points of the face. The
/// closest pairs merge first, and no merge joins two such points of one face,
/// so a solid thinner than `rel_tol` keeps its sides apart: the faces around
/// its rim hold points of both. Whether two copies merge does not depend on
/// where they fall on a grid.
///
/// Each merged vertex takes the position of its copy with the lowest index.
/// Triangles left with two corners on one vertex are dropped.
///
/// Returns `(welded_vertices, welded_triangles, kept)` where `welded_triangles`
/// index into `welded_vertices` and `kept` lists the original triangle indices
/// that survived, so callers can filter parallel per-triangle arrays.
pub fn weld_vertices(
    vertices: &[[f64; 3]],
    triangles: &[[VertexId; 3]],
    vertex_faces: &[u32],
    rel_tol: f64,
    seam_rel_tol: f64,
) -> (Vec<[f64; 3]>, Vec<[VertexId; 3]>, Vec<usize>) {
    // Sorted-unique vertex ids actually referenced by the triangles.
    let used: Vec<VertexId> = triangles
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if used.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Bounding extent over the used vertices sets the tolerances.
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &g in &used {
        let v = vertices[g as usize];
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    let max_extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    let scale = if max_extent != 0.0 { max_extent } else { 1.0 };
    let tolerance = scale * rel_tol;
    let seam_tolerance = scale * seam_rel_tol;

    // Vertices at one position are one vertex. The copies on the faces'
    // boundaries go on to the tolerance, one per position.
    let open = open_boundary(triangles, vertices.len());
    let mut copies = Copies::new(vertices.len());
    let mut at: HashMap<[u64; 3], (VertexId, bool)> = HashMap::new();
    let mut candidates: Vec<VertexId> = Vec::new();
    for &g in &used {
        let (first, has_candidate) = at
            .entry(vertices[g as usize].map(f64::to_bits))
            .or_insert((g, false));
        let (root_first, root_g) = (copies.find(*first), copies.find(g));
        if root_first != root_g {
            copies.union(root_first, root_g);
        }
        if open[g as usize] && !*has_candidate {
            *has_candidate = true;
            candidates.push(g);
        }
    }
    drop(at);

    for (_, a, b) in close_pairs(vertices, &candidates, tolerance) {
        let (root_a, root_b) = (copies.find(a), copies.find(b));
        if root_a == root_b {
            continue;
        }
        // Two points of one face, further apart than its seam copies.
        let splits_a_face = copies.members(root_a).iter().any(|&p| {
            copies.members(root_b).iter().any(|&q| {
                vertex_faces[p as usize] == vertex_faces[q as usize]
                    && distance(vertices[p as usize], vertices[q as usize]) > seam_tolerance
            })
        });
        if !splits_a_face {
            copies.union(root_a, root_b);
        }
    }

    // `used` is ascending, so each merged vertex meets its lowest copy first.
    let mut local_of_root = vec![VertexId::MAX; vertices.len()];
    let mut local = vec![VertexId::MAX; vertices.len()];
    let mut local_verts: Vec<[f64; 3]> = Vec::new();
    for &g in &used {
        let root = copies.find(g) as usize;
        if local_of_root[root] == VertexId::MAX {
            local_of_root[root] = local_verts.len() as VertexId;
            local_verts.push(vertices[g as usize]);
        }
        local[g as usize] = local_of_root[root];
    }

    let mut welded_tris: Vec<[VertexId; 3]> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    for (t, tri) in triangles.iter().enumerate() {
        let [a, b, c] = tri.map(|g| local[g as usize]);
        // Drop seams that collapsed to a degenerate triangle.
        if a != b && b != c && a != c {
            welded_tris.push([a, b, c]);
            kept.push(t);
        }
    }

    (local_verts, welded_tris, kept)
}

/// Whether each of the `n` vertices lies on the open boundary of `triangles`:
/// on an edge that other than two of them use. As each face's triangles index
/// vertices of its own, these are the vertices on its edges, along its seam
/// and at its poles.
fn open_boundary(triangles: &[[VertexId; 3]], n: usize) -> Vec<bool> {
    let mut edges: Vec<(VertexId, VertexId)> = triangles
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    let mut open = vec![false; n];
    for uses in edges.chunk_by(|a, b| a == b) {
        if uses.len() != 2 {
            let (a, b) = uses[0];
            open[a as usize] = true;
            open[b as usize] = true;
        }
    }
    open
}

/// One of each opposite pair of the 26 cells around a cell, so that a pass
/// over the cells meets every two neighbouring cells once.
const FORWARD_NEIGHBOURS: [[i64; 3]; 13] = [
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
];

/// Every pair of `points` (vertex ids) within `tolerance` of each other, as
/// `(distance, lower id, higher id)`, closest first and ties by id.
fn close_pairs(
    vertices: &[[f64; 3]],
    points: &[VertexId],
    tolerance: f64,
) -> Vec<(f64, VertexId, VertexId)> {
    // Cells as wide as the tolerance: a point's neighbours within it lie in
    // its own cell or the 26 around it.
    let mut cells: HashMap<[i64; 3], Vec<VertexId>> = HashMap::new();
    for &v in points {
        let cell = vertices[v as usize].map(|c| (c / tolerance).floor() as i64);
        cells.entry(cell).or_default().push(v);
    }
    let mut pairs = Vec::new();
    let mut add = |a: VertexId, b: VertexId| {
        let d = distance(vertices[a as usize], vertices[b as usize]);
        if d <= tolerance {
            pairs.push((d, a.min(b), a.max(b)));
        }
    };
    for (cell, members) in &cells {
        for (i, &a) in members.iter().enumerate() {
            for &b in &members[i + 1..] {
                add(a, b);
            }
        }
        for offset in FORWARD_NEIGHBOURS {
            let next = [0, 1, 2].map(|i| cell[i].saturating_add(offset[i]));
            if let Some(others) = cells.get(&next) {
                for &a in members {
                    for &b in others {
                        add(a, b);
                    }
                }
            }
        }
    }
    pairs.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    pairs
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Vertices gathered into sets of copies of one point (union-find), each set
/// with the list of its members.
struct Copies {
    parent: Vec<VertexId>,
    /// Members of each set of two or more, by its root.
    members: HashMap<VertexId, Vec<VertexId>>,
}

impl Copies {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as VertexId).collect(),
            members: HashMap::new(),
        }
    }

    fn find(&mut self, mut v: VertexId) -> VertexId {
        while self.parent[v as usize] != v {
            let grandparent = self.parent[self.parent[v as usize] as usize];
            self.parent[v as usize] = grandparent;
            v = grandparent;
        }
        v
    }

    /// The members of the set whose root is `root`.
    fn members(&self, root: VertexId) -> &[VertexId] {
        match self.members.get(&root) {
            Some(members) => members,
            // A root is its own parent.
            None => std::slice::from_ref(&self.parent[root as usize]),
        }
    }

    /// Join the sets with roots `a` and `b`, the smaller into the larger.
    fn union(&mut self, a: VertexId, b: VertexId) {
        let (big, small) = if self.members(a).len() >= self.members(b).len() {
            (a, b)
        } else {
            (b, a)
        };
        let moved = self.members.remove(&small).unwrap_or_else(|| vec![small]);
        self.members
            .entry(big)
            .or_insert_with(|| vec![big])
            .extend(moved);
        self.parent[small as usize] = big;
    }
}

/// Tet-to-tet adjacency from flat connectivity, as `[n0,n1,n2,n3, ...]` per tet
/// (face order matches [`crate::mesh::topology::TET_FACE_VERTICES`]); `-1` marks
/// a boundary face.
pub fn tet_adjacency_flat(tetrahedra: &[VertexId]) -> Vec<i32> {
    let tets: Vec<[VertexId; 4]> = tetrahedra.as_chunks::<4>().0.to_vec();
    let adjacency = build_tet_adjacency(&tets);
    let mut out = Vec::with_capacity(adjacency.len() * 4);
    for tet in adjacency {
        for face in tet {
            out.push(face.map_or(-1, |t| t as i32));
        }
    }
    out
}

/// Map each BRep face id to the solids that own it, preserving the order the
/// solids were given in (the CAD pipeline passes solids in ascending solid-id
/// order, so `owners[fid][0]` is the winner for single-owner lookups).
pub fn face_owners(solid_faces: &[(u32, Vec<i64>)]) -> HashMap<i64, Vec<u32>> {
    let mut owners: HashMap<i64, Vec<u32>> = HashMap::new();
    for (solid_id, face_ids) in solid_faces {
        for &fid in face_ids {
            owners.entry(fid).or_default().push(*solid_id);
        }
    }
    owners
}

/// Owning solid id per triangle: the first solid (in `solid_faces` order)
/// whose face list contains the triangle's face id, `-1` when unowned.
pub fn triangle_owner_ids(triangle_face_ids: &[i64], owners: &HashMap<i64, Vec<u32>>) -> Vec<i32> {
    triangle_face_ids
        .iter()
        .map(|fid| {
            owners
                .get(fid)
                .and_then(|s| s.first())
                .map_or(-1, |&s| s as i32)
        })
        .collect()
}

/// Surface-to-volume topology pairs: `pairs[surface_id - 1] = [fwd, rev]`,
/// where `fwd` (slot 0) is the 0-based volume whose outward normal matches the
/// triangle normal (face forward in that solid) and `rev` (slot 1) the volume
/// facing the other way; `None` marks no volume on that side (implicit
/// complement). `face_solid_reversed` is keyed by `(solid_id, face_id)`.
pub fn surface_volume_pairs(
    face_to_surface_id: &[(i64, u32)],
    owners: &HashMap<i64, Vec<u32>>,
    face_solid_reversed: &HashMap<(u32, i64), bool>,
) -> Vec<[Option<u32>; 2]> {
    let num_surfaces = face_to_surface_id
        .iter()
        .map(|&(_, sid)| sid)
        .max()
        .unwrap_or(0);
    let mut pairs = vec![[None, None]; num_surfaces as usize];
    for &(fid, sid) in face_to_surface_id {
        if sid == 0 {
            continue; // surface ids are 1-based by construction
        }
        let Some(solids) = owners.get(&fid) else {
            continue;
        };
        for &solid_id in solids {
            let vol_id = solid_id - 1;
            let reversed = face_solid_reversed
                .get(&(solid_id, fid))
                .copied()
                .unwrap_or(false);
            let slot = if reversed { 1 } else { 0 };
            pairs[(sid - 1) as usize][slot] = Some(vol_id);
        }
    }
    pairs
}

/// Flatten per-solid tet blocks `(solid_id, vertex_offset, tets)` into one
/// connectivity array, offsetting each block's local vertex indices to its
/// position in the global vertex array. Returns the flattened tets and the
/// owning solid id per tet.
pub fn flatten_tet_blocks(blocks: &[(u32, usize, Vec<[u32; 4]>)]) -> (Vec<[u32; 4]>, Vec<u32>) {
    let total: usize = blocks.iter().map(|(_, _, t)| t.len()).sum();
    let mut tets = Vec::with_capacity(total);
    let mut volume_ids = Vec::with_capacity(total);
    for (solid_id, offset, block) in blocks {
        let off = *offset as u32;
        for t in block {
            tets.push([t[0] + off, t[1] + off, t[2] + off, t[3] + off]);
            volume_ids.push(*solid_id);
        }
    }
    (tets, volume_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quads as the CAD pipeline hands faces to the weld: each with its own
    /// copies of its corners and two triangles, and the face of every vertex.
    fn faces(quads: &[[[f64; 3]; 4]]) -> (Vec<[f64; 3]>, Vec<[VertexId; 3]>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let mut vertex_faces = Vec::new();
        for (face, quad) in quads.iter().enumerate() {
            let base = vertices.len() as VertexId;
            vertices.extend_from_slice(quad);
            vertex_faces.extend([face as u32; 4]);
            triangles.push([base, base + 1, base + 2]);
            triangles.push([base, base + 2, base + 3]);
        }
        (vertices, triangles, vertex_faces)
    }

    /// The six faces of the box from `lo` to `hi`.
    fn box_faces(lo: [f64; 3], hi: [f64; 3]) -> Vec<[[f64; 3]; 4]> {
        let c = |i: usize, j: usize, k: usize| {
            [[lo[0], hi[0]][i], [lo[1], hi[1]][j], [lo[2], hi[2]][k]]
        };
        vec![
            [c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)],
            [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],
            [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],
            [c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)],
            [c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)],
            [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],
        ]
    }

    #[test]
    fn weld_merges_the_copies_of_a_shared_edge() {
        let (vertices, triangles, vertex_faces) = faces(&[
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ]);
        let (welded, tris, kept) = weld_vertices(&vertices, &triangles, &vertex_faces, 1e-6, 1e-9);
        assert_eq!(welded.len(), 6);
        assert_eq!(kept, vec![0, 1, 2, 3]);
        // The second face's copies take the first face's vertices.
        assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]);
    }

    #[test]
    fn weld_merges_copies_either_side_of_a_grid_line() {
        // Two copies of one point 2e-12 apart that rounding to a grid of the
        // tolerance put in different cells, leaving a crack.
        let x = 0.5 + 0.5e-6;
        let (mut vertices, triangles, vertex_faces) = faces(&[
            [
                [0.0, 0.0, 0.0],
                [x, 0.0, 0.0],
                [x, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [x, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [x, 1.0, 0.0],
            ],
        ]);
        vertices[1][0] = x - 1e-12;
        vertices[4][0] = x + 1e-12;
        assert_ne!(
            (vertices[1][0] / 1e-6).round(),
            (vertices[4][0] / 1e-6).round()
        );
        let (welded, _, _) = weld_vertices(&vertices, &triangles, &vertex_faces, 1e-6, 1e-9);
        assert_eq!(welded.len(), 6);
    }

    #[test]
    fn weld_keeps_the_sides_of_a_solid_thinner_than_its_tolerance_apart() {
        // A plate down to a hundredth of the weld's tolerance thick: the
        // faces around its rim hold copies of both sides' corners.
        for thickness in [1e-5, 1e-6, 1e-7, 1e-8] {
            let (vertices, triangles, vertex_faces) =
                faces(&box_faces([0.0, 0.0, 0.0], [1.0, 1.0, thickness]));
            let (welded, tris, kept) =
                weld_vertices(&vertices, &triangles, &vertex_faces, 1e-6, 1e-9);
            assert_eq!(welded.len(), 8, "thickness {thickness}");
            assert_eq!(tris.len(), 12, "thickness {thickness}");
            assert_eq!(kept, (0..12).collect::<Vec<_>>());
            let top = welded.iter().filter(|v| v[2] == thickness).count();
            assert_eq!(top, 4, "thickness {thickness}");
        }
    }

    #[test]
    fn weld_collapses_a_solid_within_its_seam_tolerance() {
        // Closer than the seam tolerance, two points of one face are one: the
        // plate's rim faces collapse and its sides fall onto each other.
        let (vertices, triangles, vertex_faces) =
            faces(&box_faces([0.0, 0.0, 0.0], [1.0, 1.0, 1e-10]));
        let (welded, tris, kept) = weld_vertices(&vertices, &triangles, &vertex_faces, 1e-6, 1e-9);
        assert_eq!(welded.len(), 4);
        assert_eq!(kept, vec![0, 1, 2, 3]);
        assert_eq!(tris.len(), 4);
    }

    #[test]
    fn weld_merges_seam_copies_of_one_face_only_within_the_seam_tolerance() {
        // A face of two triangles whose open boundary meets itself, as a
        // seam does: vertices 0 and 3 are two copies of one point.
        let weld = |gap: f64| {
            let vertices = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [gap, 0.0, 0.0],
            ];
            weld_vertices(&vertices, &[[0, 1, 2], [3, 2, 1]], &[7; 4], 1e-6, 1e-9)
        };
        assert_eq!(weld(1e-13).0.len(), 3);
        assert_eq!(weld(1e-8).0.len(), 4);
    }

    #[test]
    fn weld_keeps_vertices_inside_their_face() {
        // Two faces sharing their rim, each a fan around a centre of its own
        // `gap` from the other's: the rims are copies, and the centres merge
        // only at one position.
        let weld = |gap: f64| {
            let mut vertices = Vec::new();
            let mut triangles = Vec::new();
            let mut vertex_faces = Vec::new();
            for (face, z) in [(0u32, 0.0), (1, gap)] {
                let base = vertices.len() as VertexId;
                vertices.extend([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]);
                vertices.push([0.5, 0.5, z]);
                vertex_faces.extend([face; 5]);
                for i in 0..4 {
                    triangles.push([base + i, base + (i + 1) % 4, base + 4]);
                }
            }
            weld_vertices(&vertices, &triangles, &vertex_faces, 1e-6, 1e-9)
        };
        let (welded, tris, _) = weld(1e-9);
        assert_eq!(welded.len(), 6);
        assert_eq!(tris.len(), 8);
        assert_eq!(weld(0.0).0.len(), 5);
    }

    fn two_solids() -> Vec<(u32, Vec<i64>)> {
        // Solid 1 owns faces 10, 11; solid 2 owns faces 11 (shared), 12.
        vec![(1, vec![10, 11]), (2, vec![11, 12])]
    }

    #[test]
    fn triangle_owner_first_solid_wins_on_shared_faces() {
        let owners = face_owners(&two_solids());
        let ids = triangle_owner_ids(&[10, 11, 12, 99], &owners);
        assert_eq!(ids, vec![1, 1, 2, -1]);
    }

    #[test]
    fn surface_volume_pairs_fwd_rev_slots() {
        let owners = face_owners(&two_solids());
        // Face 11 is shared: forward in solid 1, reversed in solid 2.
        // Face 10 is reversed in solid 1; face 12 forward in solid 2.
        let reversed: HashMap<(u32, i64), bool> = [
            ((1, 10), true),
            ((1, 11), false),
            ((2, 11), true),
            ((2, 12), false),
        ]
        .into_iter()
        .collect();
        let f2s = vec![(10, 1), (11, 2), (12, 3)];
        let pairs = surface_volume_pairs(&f2s, &owners, &reversed);
        assert_eq!(
            pairs,
            vec![
                [None, Some(0)],    // surface 1: solid 1 reversed
                [Some(0), Some(1)], // surface 2: shared, fwd solid 1, rev solid 2
                [Some(1), None],    // surface 3: solid 2 forward
            ]
        );
    }

    #[test]
    fn surface_volume_pairs_unlisted_face_defaults_forward() {
        let owners = face_owners(&[(1, vec![10])]);
        let pairs = surface_volume_pairs(&[(10, 1)], &owners, &HashMap::new());
        assert_eq!(pairs, vec![[Some(0), None]]);
    }

    #[test]
    fn flatten_tet_blocks_offsets_and_labels() {
        let blocks = vec![
            (1, 0, vec![[0, 1, 2, 3]]),
            (2, 100, vec![[0, 1, 2, 3], [1, 2, 3, 4]]),
        ];
        let (tets, vols) = flatten_tet_blocks(&blocks);
        assert_eq!(
            tets,
            vec![[0, 1, 2, 3], [100, 101, 102, 103], [101, 102, 103, 104]]
        );
        assert_eq!(vols, vec![1, 2, 2]);
    }
}
