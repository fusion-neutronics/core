//! Chordal offset application and vertex compaction utilities.

use std::collections::HashMap;

/// Apply chordal offsets to 3D vertices along their normals.
///
/// For each vertex `i`, if `offsets[i] > 0` and a normal is available,
/// the vertex is shifted by `offset * normal`.  Vertices beyond the
/// length of `offsets` or `normals` are left unchanged.
///
/// Operates in-place semantics but returns a new Vec for FFI convenience.
pub fn apply_chordal_offsets(
    verts_3d: &[[f64; 3]],
    offsets: &[f64],
    normals: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    let mut out = verts_3d.to_vec();
    let n = out.len().min(offsets.len()).min(normals.len());
    for i in 0..n {
        let off = offsets[i];
        if off > 0.0 {
            out[i][0] += off * normals[i][0];
            out[i][1] += off * normals[i][1];
            out[i][2] += off * normals[i][2];
        }
    }
    out
}

/// Compact a triangle mesh: extract only the used vertices and remap
/// triangle indices to a contiguous 0-based range.
///
/// Given global `vertices` and `triangles` (with indices into `vertices`),
/// returns `(local_verts, local_tris)` where `local_verts` contains only
/// the vertices referenced by `triangles`, and `local_tris` has remapped
/// indices.
pub fn compact_mesh(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    // Collect and sort unique vertex indices
    let mut used: Vec<usize> = Vec::new();
    for tri in triangles {
        used.push(tri[0]);
        used.push(tri[1]);
        used.push(tri[2]);
    }
    used.sort_unstable();
    used.dedup();

    // Build global → local map
    let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(used.len());
    let mut local_verts: Vec<[f64; 3]> = Vec::with_capacity(used.len());
    for (local_idx, &global_idx) in used.iter().enumerate() {
        g2l.insert(global_idx, local_idx);
        local_verts.push(vertices[global_idx]);
    }

    // Remap triangles
    let local_tris: Vec<[usize; 3]> = triangles
        .iter()
        .map(|t| [g2l[&t[0]], g2l[&t[1]], g2l[&t[2]]])
        .collect();

    (local_verts, local_tris)
}
