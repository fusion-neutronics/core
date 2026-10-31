//! Pure mesh-construction kernels used by the CAD export pipeline.
//!
//! These were previously per-element Python loops in `yamc.cad`; they are the
//! compiled core the Python side now calls. `weld_vertices` merges coincident
//! boundary vertices into a watertight, index-shared mesh; the `*_flat`
//! wrappers expose the topology builders (adjacency, AABBs) on the flat arrays
//! the Arrow writer uses.

use std::collections::{BTreeSet, HashMap};

use crate::mesh::topology::{build_tet_aabbs, build_tet_adjacency, build_triangle_aabbs};
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

/// Per-triangle AABBs from flat vertex/triangle arrays, as
/// `[min_x,min_y,min_z,max_x,max_y,max_z, ...]`.
pub fn tri_aabbs_flat(vertices: &[f64], triangles: &[VertexId]) -> Vec<f64> {
    let verts: Vec<[f64; 3]> = vertices.as_chunks::<3>().0.to_vec();
    let tris: Vec<[VertexId; 3]> = triangles.as_chunks::<3>().0.to_vec();
    build_triangle_aabbs(&tris, &verts)
        .into_iter()
        .flatten()
        .collect()
}

/// Per-tet AABBs from flat vertex/tet arrays, as
/// `[min_x,min_y,min_z,max_x,max_y,max_z, ...]`.
pub fn tet_aabbs_flat(vertices: &[f64], tetrahedra: &[VertexId]) -> Vec<f64> {
    let verts: Vec<[f64; 3]> = vertices.as_chunks::<3>().0.to_vec();
    let tets: Vec<[VertexId; 4]> = tetrahedra.as_chunks::<4>().0.to_vec();
    build_tet_aabbs(&tets, &verts)
        .into_iter()
        .flatten()
        .collect()
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
