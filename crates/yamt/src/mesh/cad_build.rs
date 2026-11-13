//! Pure mesh-construction kernels used by the CAD export pipeline.
//!
//! These were previously per-element Python loops in `yamc.cad`; they are the
//! compiled core the Python side now calls. `weld_vertices` merges coincident
//! boundary vertices into a watertight, index-shared mesh; `tet_adjacency_flat`
//! exposes the adjacency builder on the flat arrays the Arrow writer uses.

use std::collections::{BTreeSet, HashMap};

use crate::mesh::topology::build_tet_adjacency;
use crate::query::intersect::{add, dot, length, scale, sub};
use crate::types::VertexId;

/// Weld coincident vertices of a triangle mesh into a watertight boundary.
///
/// Only the vertices referenced by `triangles` are considered. Their bounding
/// extent sets a quantization step (`max_extent * rel_tol`); vertices whose
/// quantized coordinates match are merged, keeping the first as representative.
/// Triangles that collapse to a degenerate (two shared indices) after welding
/// are dropped.
///
/// Returns `(welded_vertices, welded_triangles, kept)` where `welded_triangles`
/// index into `welded_vertices` and `kept` lists the original triangle indices
/// that survived, so callers can filter parallel per-triangle arrays.
pub fn weld_vertices(
    vertices: &[[f64; 3]],
    triangles: &[[VertexId; 3]],
    rel_tol: f64,
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

    // Bounding extent over the used vertices sets the merge quantum.
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
    let quantum = if max_extent != 0.0 { max_extent } else { 1.0 } * rel_tol;

    let mut key_to_local: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut gid_to_local: HashMap<VertexId, VertexId> = HashMap::new();
    let mut local_verts: Vec<[f64; 3]> = Vec::new();
    for &g in &used {
        let v = vertices[g as usize];
        let key = (
            (v[0] / quantum).round() as i64,
            (v[1] / quantum).round() as i64,
            (v[2] / quantum).round() as i64,
        );
        let local = *key_to_local.entry(key).or_insert_with(|| {
            let idx = local_verts.len() as VertexId;
            local_verts.push(v);
            idx
        });
        gid_to_local.insert(g, local);
    }

    let mut welded_tris: Vec<[VertexId; 3]> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    for (t, tri) in triangles.iter().enumerate() {
        let a = gid_to_local[&tri[0]];
        let b = gid_to_local[&tri[1]];
        let c = gid_to_local[&tri[2]];
        // Drop seams that collapsed to a degenerate triangle.
        if a != b && b != c && a != c {
            welded_tris.push([a, b, c]);
            kept.push(t);
        }
    }

    (local_verts, welded_tris, kept)
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

/// Edges along which each volume's surface is open, every volume bounded by
/// the surfaces `surface_volumes` gives it (as [`surface_volume_pairs`] writes
/// them) with their triangles wound outward from it.
///
/// A closed surface runs along each edge of its triangles as often one way as
/// the other. An edge a volume's triangles run along more often one way is
/// open: a hole, a face left one-sided (a face inside a solid written as a
/// boundary with the implicit complement), a face wound the wrong way round,
/// or a crack between two facetings of one CAD edge (see
/// [`unmatched_edges`]). A surface with the volume on both sides runs both
/// ways at once and closes on itself.
///
/// Returns `(volume, edges)` for every volume with an open edge, in volume
/// order, each edge as `[from, to]` the way its triangles run in excess,
/// sorted.
pub fn open_volume_edges(
    triangles: &[[VertexId; 3]],
    triangle_surface_ids: &[u32],
    surface_volumes: &[[Option<u32>; 2]],
) -> Vec<(u32, Vec<[VertexId; 2]>)> {
    // Per volume and edge, lower vertex first: the runs from the lower vertex
    // to the higher, less the runs back.
    let mut balance: HashMap<(u32, VertexId, VertexId), i64> = HashMap::new();
    for (tri, &surface) in triangles.iter().zip(triangle_surface_ids) {
        let Some(volumes) = (surface as usize)
            .checked_sub(1)
            .and_then(|s| surface_volumes.get(s))
        else {
            continue;
        };
        for (volume, sense) in [(volumes[0], 1), (volumes[1], -1)] {
            let Some(volume) = volume else { continue };
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                if a < b {
                    *balance.entry((volume, a, b)).or_insert(0) += sense;
                } else if b < a {
                    *balance.entry((volume, b, a)).or_insert(0) -= sense;
                }
            }
        }
    }
    let mut open: Vec<(u32, [VertexId; 2])> = balance
        .into_iter()
        .filter(|&(_, n)| n != 0)
        .map(|((volume, a, b), n)| (volume, if n > 0 { [a, b] } else { [b, a] }))
        .collect();
    open.sort_unstable();
    let mut by_volume: Vec<(u32, Vec<[VertexId; 2]>)> = Vec::new();
    for (volume, edge) in open {
        match by_volume.last_mut() {
            Some((v, edges)) if *v == volume => edges.push(edge),
            _ => by_volume.push((volume, vec![edge])),
        }
    }
    by_volume
}

/// The open edges of one volume (as [`open_volume_edges`] gives them) that no
/// open edge running back alongside them accounts for.
///
/// The two faces on a CAD edge can facet it without sharing their vertices
/// along it: copies of a vertex the weld does not merge, a face whose surface
/// puts its boundary slightly off the edge (an offset surface), or one face
/// with points along the edge that the other lacks. The two sides of such a
/// crack are open edges running opposite ways, as far apart as the facetings
/// of the edge. The rim of a hole, or of a face left one-sided, has nothing
/// running back alongside it. An edge counts as matched when an open edge
/// running the other way (their directions more than 90 degrees apart)
/// passes within `reach` of its midpoint; a `reach` of 0 matches none.
pub fn unmatched_edges(
    vertices: &[[f64; 3]],
    edges: &[[VertexId; 2]],
    reach: f64,
) -> Vec<[VertexId; 2]> {
    if reach.is_nan() || reach <= 0.0 || edges.is_empty() {
        return edges.to_vec();
    }
    let ends: Vec<([f64; 3], [f64; 3])> = edges
        .iter()
        .map(|&[a, b]| (vertices[a as usize], vertices[b as usize]))
        .collect();
    let mean = ends.iter().map(|&(p, q)| length(sub(q, p))).sum::<f64>() / ends.len() as f64;
    // Points along every edge less than a cell apart, each filed under its
    // cell. A midpoint within `reach` (at most half a cell) of an edge is
    // then less than a cell from one of the edge's points, so in the same
    // cell as it or a neighbour.
    let cell = (2.0 * reach).max(0.25 * mean);
    let key = |p: [f64; 3]| p.map(|x| (x / cell).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, &(p, q)) in ends.iter().enumerate() {
        let n = (length(sub(q, p)) / cell).ceil() as usize + 1;
        for k in 0..=n {
            let here = grid
                .entry(key(add(p, scale(sub(q, p), k as f64 / n as f64))))
                .or_default();
            if here.last() != Some(&i) {
                here.push(i);
            }
        }
    }
    let matched = |i: usize| {
        let (p, q) = ends[i];
        let mid = scale(add(p, q), 0.5);
        let runs_back = |j: usize| {
            let (a, b) = ends[j];
            let ab = sub(b, a);
            let t = (dot(sub(mid, a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
            j != i && dot(sub(q, p), ab) < 0.0 && length(sub(mid, add(a, scale(ab, t)))) <= reach
        };
        let [x, y, z] = key(mid);
        (x - 1..=x + 1).any(|cx| {
            (y - 1..=y + 1).any(|cy| {
                (z - 1..=z + 1).any(|cz| {
                    grid.get(&[cx, cy, cz])
                        .is_some_and(|near| near.iter().any(|&j| runs_back(j)))
                })
            })
        })
    };
    edges
        .iter()
        .enumerate()
        .filter(|&(i, _)| !matched(i))
        .map(|(_, &edge)| edge)
        .collect()
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

    /// A unit cube wound outward, vertex `x + 2y + 4z` at corner `(x, y, z)`,
    /// two triangles to a face in the order -z, +z, -y, +y, -x, +x.
    const CUBE: [[VertexId; 3]; 12] = [
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
    ];

    /// The cube with each face its own surface (1 to 6), plus a square sheet
    /// inside it (vertices 8 to 11) as surface 7.
    fn cube_with_sheet() -> (Vec<[VertexId; 3]>, Vec<u32>) {
        let mut triangles = CUBE.to_vec();
        triangles.extend([[8, 9, 10], [8, 10, 11]]);
        let surfaces = (0..14).map(|t| t / 2 + 1).collect();
        (triangles, surfaces)
    }

    #[test]
    fn open_volume_edges_finds_a_sheet_left_one_sided() {
        let (triangles, surfaces) = cube_with_sheet();
        let mut pairs = vec![[Some(0), None]; 7];
        // Only the sheet's rim is open, the way the sheet runs.
        assert_eq!(
            open_volume_edges(&triangles, &surfaces, &pairs),
            vec![(0, vec![[8, 9], [9, 10], [10, 11], [11, 8]])]
        );
        // With the cube on both sides the sheet closes on itself.
        pairs[6] = [Some(0), Some(0)];
        assert_eq!(open_volume_edges(&triangles, &surfaces, &pairs), vec![]);
        // So does a face wound inward that bounds the cube from its back.
        let mut inward = triangles.clone();
        for tri in &mut inward[10..12] {
            tri.swap(1, 2);
        }
        pairs[5] = [None, Some(0)];
        assert_eq!(open_volume_edges(&inward, &surfaces, &pairs), vec![]);
    }

    #[test]
    fn open_volume_edges_finds_a_hole_and_a_face_held_the_wrong_way() {
        let (triangles, surfaces) = cube_with_sheet();
        let (cube, surfaces) = (&triangles[..12], &surfaces[..12]);
        let rim = vec![[4, 6], [5, 4], [6, 7], [7, 5]];
        // The +z face (surface 2) bounds nothing: its rim is open.
        let mut pairs = vec![[Some(0), None]; 6];
        pairs[1] = [None, None];
        assert_eq!(
            open_volume_edges(cube, surfaces, &pairs),
            vec![(0, rim.clone())]
        );
        // Bounding the cube from its back, the face runs its rim the way its
        // neighbours do.
        pairs[1] = [None, Some(0)];
        assert_eq!(
            open_volume_edges(cube, surfaces, &pairs),
            vec![(0, rim.clone())]
        );
        // Facing forward it closes the cube, and leaves volume 1 behind it,
        // which it bounds alone, open.
        pairs[1] = [Some(0), Some(1)];
        assert_eq!(open_volume_edges(cube, surfaces, &pairs), vec![(1, rim)]);
    }

    #[test]
    fn unmatched_edges_passes_a_crack_and_keeps_a_rim() {
        // The edge from (0, 0, 0) to (2, 0, 0) faceted by two faces 1e-7
        // apart, one with a point at x = 1 that the other lacks; and the rim
        // of a unit square.
        let vertices = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1e-7, 0.0],
            [1.0, 1e-7, 0.0],
            [0.0, 1e-7, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let crack = [[0, 1], [2, 3], [3, 4]];
        let rim = [[5, 6], [6, 7], [7, 8], [8, 5]];
        let both: Vec<_> = crack.iter().chain(&rim).copied().collect();
        assert_eq!(unmatched_edges(&vertices, &both, 1e-6), rim.to_vec());
        // Nothing matches out of reach, or running the same way.
        assert_eq!(unmatched_edges(&vertices, &crack, 0.0), crack.to_vec());
        assert_eq!(unmatched_edges(&vertices, &crack, 1e-8), crack.to_vec());
        let same_way = [[0, 1], [4, 2]];
        assert_eq!(
            unmatched_edges(&vertices, &same_way, 1e-6),
            same_way.to_vec()
        );
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
