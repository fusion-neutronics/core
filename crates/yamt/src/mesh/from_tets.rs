//! A mesh geometry from a bare tet mesh: every volume tet-filled.
//!
//! Another code's tet mesh (FEniCSx, gmsh) gives vertices, tets and a
//! material per tet. The volumes are the materials, and their boundary
//! surfaces are the tet faces where the material changes, plus the outer
//! faces, so the surfaces are conformal with the tets by construction and
//! nothing has to be mapped between the two codes.

use crate::types::VertexId;

/// What [`from_tets`] builds: the tables of a mesh file (see
/// `crate::io::arrow::MeshData`), unflattened.
#[derive(Debug, Clone, Default)]
pub struct TetMeshTables {
    /// The tets, each positively oriented (re-wound where it was not).
    pub tetrahedra: Vec<[VertexId; 4]>,
    /// Boundary triangles over the same vertices, wound so their normal
    /// points out of the first volume of their surface.
    pub triangles: Vec<[VertexId; 3]>,
    /// 1-based surface id of each triangle.
    pub triangle_surface_ids: Vec<u32>,
    /// `[out of, into]` 0-based volumes of each surface, `None` outside.
    pub surface_volumes: Vec<[Option<u32>; 2]>,
    /// The volume (cm³) of each volume, summed over its tets.
    pub volume_measures: Vec<f64>,
    /// Tets that were wound negatively and re-wound.
    pub rewound: usize,
}

fn signed_volume(v: &[[f64; 3]], t: [VertexId; 4]) -> f64 {
    let [a, b, c, d] = t.map(|i| v[i as usize]);
    let e = |p: [f64; 3]| [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let (b, c, d) = (e(b), e(c), e(d));
    (b[0] * (c[1] * d[2] - c[2] * d[1]) - b[1] * (c[0] * d[2] - c[2] * d[0])
        + b[2] * (c[0] * d[1] - c[1] * d[0]))
        / 6.0
}

/// The faces of a positively oriented tet `[a, b, c, d]`, each wound so its
/// normal points out of the tet.
fn outward_faces(t: [VertexId; 4]) -> [[VertexId; 3]; 4] {
    let [a, b, c, d] = t;
    [[a, c, b], [a, b, d], [a, d, c], [b, c, d]]
}

/// Build the boundary surfaces of a tet mesh whose tet `i` is in volume
/// `tet_volume[i]` (0-based, `< num_volumes`).
///
/// Errors on a degenerate tet, a vertex index out of range, or a face shared
/// by more than two tets (a non-manifold mesh).
pub fn from_tets(
    vertices: &[[f64; 3]],
    tets: &[[VertexId; 4]],
    tet_volume: &[u32],
    num_volumes: u32,
) -> Result<TetMeshTables, String> {
    if tets.len() != tet_volume.len() {
        return Err(format!(
            "{} tets but {} volume ids: give one per tet",
            tets.len(),
            tet_volume.len()
        ));
    }
    if let Some(&v) = tet_volume.iter().find(|&&v| v >= num_volumes) {
        return Err(format!(
            "volume id {v} is not below the {num_volumes} volumes"
        ));
    }
    let n = vertices.len() as VertexId;
    let mut out = TetMeshTables {
        volume_measures: vec![0.0; num_volumes as usize],
        ..Default::default()
    };
    out.tetrahedra.reserve(tets.len());
    for (i, &t) in tets.iter().enumerate() {
        if let Some(&bad) = t.iter().find(|&&j| j >= n) {
            return Err(format!(
                "tet {i} uses vertex {bad}, but there are {n} vertices"
            ));
        }
        let mut t = t;
        let mut vol = signed_volume(vertices, t);
        if vol < 0.0 {
            t.swap(0, 1);
            vol = -vol;
            out.rewound += 1;
        }
        if vol.is_nan() || vol == 0.0 {
            return Err(format!("tet {i} has no volume"));
        }
        out.volume_measures[tet_volume[i] as usize] += vol;
        out.tetrahedra.push(t);
    }

    // Every face once per tet, sorted so a face's two sides are adjacent.
    let mut faces: Vec<([VertexId; 3], u32, u8)> = Vec::with_capacity(4 * tets.len());
    for (i, &t) in out.tetrahedra.iter().enumerate() {
        for (k, f) in outward_faces(t).into_iter().enumerate() {
            let mut key = f;
            key.sort_unstable();
            faces.push((key, i as u32, k as u8));
        }
    }
    faces.sort_unstable_by_key(|f| f.0);

    // A boundary face: one tet, or two tets in different volumes. Surfaces
    // are grouped by the (out of, into) volume pair, in order of appearance.
    let mut surface_of: std::collections::HashMap<[Option<u32>; 2], u32> =
        std::collections::HashMap::new();
    let mut add = |out: &mut TetMeshTables, tri: [VertexId; 3], pair: [Option<u32>; 2]| {
        let next = out.surface_volumes.len() as u32;
        let s = *surface_of.entry(pair).or_insert(next);
        if s == next {
            out.surface_volumes.push(pair);
        }
        out.triangles.push(tri);
        out.triangle_surface_ids.push(s + 1);
    };
    let mut i = 0;
    while i < faces.len() {
        let mut j = i + 1;
        while j < faces.len() && faces[j].0 == faces[i].0 {
            j += 1;
        }
        let tri = |f: &([VertexId; 3], u32, u8)| {
            outward_faces(out.tetrahedra[f.1 as usize])[f.2 as usize]
        };
        match j - i {
            1 => {
                let f = faces[i];
                let a = tet_volume[f.1 as usize];
                let t = tri(&f);
                add(&mut out, t, [Some(a), None]);
            }
            2 => {
                let (f, g) = (faces[i], faces[i + 1]);
                let (a, b) = (tet_volume[f.1 as usize], tet_volume[g.1 as usize]);
                if a != b {
                    // one triangle per shared face, out of the lower volume
                    let (f, a, b) = if a < b { (f, a, b) } else { (g, b, a) };
                    let t = tri(&f);
                    add(&mut out, t, [Some(a), Some(b)]);
                }
            }
            k => {
                return Err(format!(
                    "a face of tet {} is shared by {k} tets: the mesh is not manifold",
                    faces[i].1
                ));
            }
        }
        i = j;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube of 6 tets around its diagonal, the first three in volume
    /// 0 and the rest in volume 1.
    fn cube() -> (Vec<[f64; 3]>, Vec<[VertexId; 4]>) {
        let v: Vec<[f64; 3]> = (0..8)
            .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
            .collect();
        let t = vec![
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ];
        (v, t)
    }

    #[test]
    fn surfaces_bound_each_volume() {
        let (v, t) = cube();
        let out = from_tets(&v, &t, &[0, 0, 0, 1, 1, 1], 2).unwrap();
        // 12 outer triangles and 2 on the interface (the cube's diagonal planes)
        let outer = out
            .triangle_surface_ids
            .iter()
            .filter(|&&s| out.surface_volumes[s as usize - 1][1].is_none());
        assert_eq!(outer.count(), 12);
        assert_eq!(out.triangles.len(), 14);
        assert_eq!(out.surface_volumes.len(), 3);
        let total: f64 = out.volume_measures.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((out.volume_measures[0] - 0.5).abs() < 1e-12);
        // every triangle points out of its first volume: the divergence
        // theorem gives each volume back
        for vol in 0..2u32 {
            let mut sum = 0.0;
            for (tri, &s) in out.triangles.iter().zip(&out.triangle_surface_ids) {
                let [a, b] = out.surface_volumes[s as usize - 1];
                let sign = if a == Some(vol) {
                    1.0
                } else if b == Some(vol) {
                    -1.0
                } else {
                    continue;
                };
                let [p, q, r] = tri.map(|i| v[i as usize]);
                let c = [
                    q[1] * r[2] - q[2] * r[1],
                    q[2] * r[0] - q[0] * r[2],
                    q[0] * r[1] - q[1] * r[0],
                ];
                sum += sign * (p[0] * c[0] + p[1] * c[1] + p[2] * c[2]) / 6.0;
            }
            assert!(
                (sum - out.volume_measures[vol as usize]).abs() < 1e-12,
                "{vol}: {sum}"
            );
        }
    }

    #[test]
    fn negative_tets_are_rewound() {
        let (v, mut t) = cube();
        let before = from_tets(&v, &t, &[0; 6], 1).unwrap().rewound;
        let flipped = signed_volume(&v, t[2]) > 0.0;
        t[2].swap(1, 2);
        let out = from_tets(&v, &t, &[0; 6], 1).unwrap();
        assert_eq!(out.rewound, if flipped { before + 1 } else { before - 1 });
        assert!(out.tetrahedra.iter().all(|&t| signed_volume(&v, t) > 0.0));
        assert_eq!(out.triangles.len(), 12);
    }

    #[test]
    fn bad_input_is_refused() {
        let (v, t) = cube();
        assert!(from_tets(&v, &t, &[0; 5], 1).is_err());
        assert!(from_tets(&v, &t, &[2; 6], 2).is_err());
        let mut flat = t.clone();
        flat[0] = [0, 1, 2, 3];
        assert!(from_tets(&v, &flat, &[0; 6], 1)
            .unwrap_err()
            .contains("no volume"));
        let mut twice = t.clone();
        twice.push(t[0]);
        assert!(from_tets(&v, &twice, &[0; 7], 1)
            .unwrap_err()
            .contains("manifold"));
    }
}
