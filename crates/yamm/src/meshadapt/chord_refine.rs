//! Chord-error refinement: split edges where flat triangles deviate from
//! the curved surface by more than a tolerance.
//!
//! This is a two-phase process:
//! 1. [`chord_refine_edges`] - find unique edge midpoints that need surface evaluation
//! 2. [`chord_refine_split`] - given the evaluated midpoints, split edges exceeding tolerance

use std::collections::HashMap;

/// Collect unique edge midpoint UV coordinates for surface evaluation.
///
/// Returns `(edge_keys, midpoint_uvs)` where each edge_key is `(min_idx, max_idx)`
/// and midpoint_uvs are the UV midpoints that Python should evaluate on the surface.
pub fn chord_refine_edges(
    verts_uv: &[[f64; 2]],
    tris: &[[usize; 3]],
) -> (Vec<[usize; 2]>, Vec<[f64; 2]>) {
    chord_refine_edges_inner(verts_uv, tris, 0)
}

/// Collect edges for chord refinement, skipping boundary edges.
///
/// `n_boundary_verts`: edges where both endpoints have index < this value
/// are skipped (they are boundary polygon vertices that must not be split
/// to preserve watertight stitching between adjacent faces).
pub fn chord_refine_edges_skip_boundary(
    verts_uv: &[[f64; 2]],
    tris: &[[usize; 3]],
    n_boundary_verts: usize,
) -> (Vec<[usize; 2]>, Vec<[f64; 2]>) {
    chord_refine_edges_inner(verts_uv, tris, n_boundary_verts)
}

fn chord_refine_edges_inner(
    verts_uv: &[[f64; 2]],
    tris: &[[usize; 3]],
    n_boundary_verts: usize,
) -> (Vec<[usize; 2]>, Vec<[f64; 2]>) {
    let mut seen = HashMap::new();
    let mut edge_keys = Vec::new();
    let mut midpoint_uvs = Vec::new();

    for tri in tris {
        for ei in 0..3 {
            let a = tri[ei];
            let b = tri[(ei + 1) % 3];
            let key = if a < b { [a, b] } else { [b, a] };
            if seen.contains_key(&key) {
                continue;
            }
            seen.insert(key, edge_keys.len());

            // Skip boundary edges (both endpoints are boundary polygon vertices)
            if n_boundary_verts > 0 && a < n_boundary_verts && b < n_boundary_verts {
                continue;
            }

            let um = 0.5 * (verts_uv[a][0] + verts_uv[b][0]);
            let vm = 0.5 * (verts_uv[a][1] + verts_uv[b][1]);
            edge_keys.push(key);
            midpoint_uvs.push([um, vm]);
        }
    }

    (edge_keys, midpoint_uvs)
}

/// Split edges where the surface midpoint deviates from the linear midpoint
/// by more than `chord_tol`.
///
/// `midpoint_xyz` must be the same length as `edge_keys` and contain the
/// surface-evaluated 3D positions (from Python OCP calls).
///
/// Returns `(new_verts_uv, new_verts_3d, new_tris)`.
#[allow(clippy::type_complexity)]
pub fn chord_refine_split(
    verts_uv: &[[f64; 2]],
    verts_3d: &[[f64; 3]],
    tris: &[[usize; 3]],
    edge_keys: &[[usize; 2]],
    midpoint_xyz: &[[f64; 3]],
    chord_tol: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let n_verts = verts_uv.len();

    // Build edge_key → index map
    let _edge_map: HashMap<[usize; 2], usize> =
        edge_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();

    // Determine which edges to split
    let mut split_map: HashMap<[usize; 2], usize> = HashMap::new(); // edge_key → new_vert_idx
    let mut new_uv = Vec::new();
    let mut new_xyz = Vec::new();

    for (i, &key) in edge_keys.iter().enumerate() {
        let a = key[0];
        let b = key[1];
        let mx = 0.5 * (verts_3d[a][0] + verts_3d[b][0]);
        let my = 0.5 * (verts_3d[a][1] + verts_3d[b][1]);
        let mz = 0.5 * (verts_3d[a][2] + verts_3d[b][2]);

        let sx = midpoint_xyz[i][0];
        let sy = midpoint_xyz[i][1];
        let sz = midpoint_xyz[i][2];

        let err = ((sx - mx).powi(2) + (sy - my).powi(2) + (sz - mz).powi(2)).sqrt();

        if err > chord_tol {
            let new_idx = n_verts + new_uv.len();
            split_map.insert(key, new_idx);
            let um = 0.5 * (verts_uv[a][0] + verts_uv[b][0]);
            let vm = 0.5 * (verts_uv[a][1] + verts_uv[b][1]);
            new_uv.push([um, vm]);
            new_xyz.push([sx, sy, sz]);
        }
    }

    if split_map.is_empty() {
        // No splits needed - return cloned input
        return (verts_uv.to_vec(), verts_3d.to_vec(), tris.to_vec());
    }

    // Build output vertex arrays
    let mut out_uv: Vec<[f64; 2]> = verts_uv.to_vec();
    let mut out_xyz: Vec<[f64; 3]> = verts_3d.to_vec();
    out_uv.extend_from_slice(&new_uv);
    out_xyz.extend_from_slice(&new_xyz);

    // Split triangles
    let mut out_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 2);

    for tri in tris {
        // Check which edges of this triangle are split
        let mut splits = [None; 3]; // splits[ei] = Some(new_vert_idx)
        for ei in 0..3 {
            let a = tri[ei];
            let b = tri[(ei + 1) % 3];
            let key = if a < b { [a, b] } else { [b, a] };
            if let Some(&new_idx) = split_map.get(&key) {
                splits[ei] = Some(new_idx);
            }
        }

        let n_splits = splits.iter().filter(|s| s.is_some()).count();

        match n_splits {
            0 => {
                out_tris.push(*tri);
            }
            1 => {
                let ei = splits.iter().position(|s| s.is_some()).unwrap();
                let m = splits[ei].unwrap();
                let v0 = tri[ei];
                let v1 = tri[(ei + 1) % 3];
                let v2 = tri[(ei + 2) % 3];
                out_tris.push([v0, m, v2]);
                out_tris.push([m, v1, v2]);
            }
            2 => {
                let v0 = tri[0];
                let v1 = tri[1];
                let v2 = tri[2];
                let m0 = splits[0];
                let m1 = splits[1];
                let m2 = splits[2];

                match (m0, m1, m2) {
                    (Some(m0), Some(m1), None) => {
                        out_tris.push([v0, m0, v2]);
                        out_tris.push([m0, v1, m1]);
                        out_tris.push([m0, m1, v2]);
                    }
                    (None, Some(m1), Some(m2)) => {
                        out_tris.push([v0, v1, m1]);
                        out_tris.push([v0, m1, m2]);
                        out_tris.push([m2, m1, v2]);
                    }
                    (Some(m0), None, Some(m2)) => {
                        out_tris.push([v0, m0, m2]);
                        out_tris.push([m0, v1, v2]);
                        out_tris.push([m0, v2, m2]);
                    }
                    _ => unreachable!(),
                }
            }
            3 => {
                let v0 = tri[0];
                let v1 = tri[1];
                let v2 = tri[2];
                let m0 = splits[0].unwrap();
                let m1 = splits[1].unwrap();
                let m2 = splits[2].unwrap();
                out_tris.push([v0, m0, m2]);
                out_tris.push([m0, v1, m1]);
                out_tris.push([m2, m1, v2]);
                out_tris.push([m0, m1, m2]);
            }
            _ => unreachable!(),
        }
    }

    (out_uv, out_xyz, out_tris)
}
