/// Tet mesh optimization: ODT smoothing, topological flips, sliver elimination.
use super::predicates3d::{circumcenter_tet, dist_sq_3d, tet_volume};

/// Quality metric for a tetrahedron.
/// Returns a value in [0, 1] where 1 = ideal equilateral tet.
/// Uses the ratio of actual volume to the volume of an equilateral tet
/// with the same RMS edge length.
pub fn tet_quality(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let vol = tet_volume(a, b, c, d).abs();

    // RMS edge length
    let edges = [
        dist_sq_3d(a, b),
        dist_sq_3d(a, c),
        dist_sq_3d(a, d),
        dist_sq_3d(b, c),
        dist_sq_3d(b, d),
        dist_sq_3d(c, d),
    ];
    let rms_sq = edges.iter().sum::<f64>() / 6.0;
    let rms = rms_sq.sqrt();

    if rms < 1e-15 {
        return 0.0;
    }

    // Volume of ideal equilateral tet with edge length rms:
    // V_ideal = rms^3 / (6 * sqrt(2))
    let ideal_vol = rms * rms * rms / (6.0 * 2.0_f64.sqrt());

    if ideal_vol < 1e-30 {
        return 0.0;
    }

    (vol / ideal_vol).min(1.0)
}

/// Compute minimum dihedral angle of a tetrahedron (in radians).
pub fn min_dihedral_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    // The 6 edges and their opposite edges form 6 dihedral angles.
    // Each dihedral angle is computed from the normals of the two adjacent faces.
    let faces: [([f64; 3], [f64; 3], [f64; 3]); 4] = [
        (b, c, d), // opposite a
        (a, d, c), // opposite b
        (a, b, d), // opposite c
        (a, c, b), // opposite d
    ];

    let normals: Vec<[f64; 3]> = faces
        .iter()
        .map(|(p, q, r)| {
            let u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        })
        .collect();

    let mut min_angle = std::f64::consts::PI;

    // 6 edges, each shared by 2 faces
    let edge_face_pairs = [
        (0, 1), // edge cd: faces opp a, opp b
        (0, 2), // edge bd: faces opp a, opp c
        (0, 3), // edge bc: faces opp a, opp d
        (1, 2), // edge ad: faces opp b, opp c
        (1, 3), // edge ac: faces opp b, opp d
        (2, 3), // edge ab: faces opp c, opp d
    ];

    for &(i, j) in &edge_face_pairs {
        let ni = &normals[i];
        let nj = &normals[j];
        let dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2];
        let li = (ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]).sqrt();
        let lj = (nj[0] * nj[0] + nj[1] * nj[1] + nj[2] * nj[2]).sqrt();
        if li > 1e-15 && lj > 1e-15 {
            // Dihedral angle is PI minus the angle between outward normals
            let cos_angle = (dot / (li * lj)).clamp(-1.0, 1.0);
            let angle = std::f64::consts::PI - cos_angle.acos();
            min_angle = min_angle.min(angle);
        }
    }

    min_angle
}

/// ODT (Optimal Delaunay Triangulation) smoothing.
///
/// Moves interior vertices toward the weighted average of circumcenters of their
/// incident tets (weighted by tet volume). This minimizes the interpolation error
/// of the Delaunay triangulation and naturally produces uniform tet distributions.
///
/// `boundary_count` is the number of fixed boundary vertices (indices 0..boundary_count).
pub fn odt_smooth(
    vertices: &mut [[f64; 3]],
    tets: &[[usize; 4]],
    boundary_count: usize,
    iterations: usize,
) {
    for _ in 0..iterations {
        // Accumulate weighted circumcenter contributions per vertex
        let n = vertices.len();
        let mut new_pos = vec![[0.0_f64; 3]; n];
        let mut weights = vec![0.0_f64; n];

        for t in tets {
            let a = vertices[t[0]];
            let b = vertices[t[1]];
            let c = vertices[t[2]];
            let d = vertices[t[3]];

            let vol = tet_volume(a, b, c, d).abs();
            if vol < 1e-30 {
                continue;
            }

            let cc = circumcenter_tet(a, b, c, d);

            for &vi in t {
                if vi >= boundary_count {
                    new_pos[vi][0] += cc[0] * vol;
                    new_pos[vi][1] += cc[1] * vol;
                    new_pos[vi][2] += cc[2] * vol;
                    weights[vi] += vol;
                }
            }
        }

        // Update interior vertices
        for vi in boundary_count..n {
            if weights[vi] > 1e-30 {
                vertices[vi] = [
                    new_pos[vi][0] / weights[vi],
                    new_pos[vi][1] / weights[vi],
                    new_pos[vi][2] / weights[vi],
                ];
            }
        }
    }
}

/// Laplacian smoothing: move each interior vertex to the average of its neighbors.
pub fn laplacian_smooth(
    vertices: &mut [[f64; 3]],
    tets: &[[usize; 4]],
    boundary_count: usize,
    iterations: usize,
) {
    for _ in 0..iterations {
        let n = vertices.len();
        let mut new_pos = vec![[0.0_f64; 3]; n];
        let mut neighbor_count = vec![0u32; n];

        for t in tets {
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        let vi = t[i];
                        let vj = t[j];
                        if vi >= boundary_count {
                            new_pos[vi][0] += vertices[vj][0];
                            new_pos[vi][1] += vertices[vj][1];
                            new_pos[vi][2] += vertices[vj][2];
                            neighbor_count[vi] += 1;
                        }
                    }
                }
            }
        }

        for vi in boundary_count..n {
            if neighbor_count[vi] > 0 {
                let c = neighbor_count[vi] as f64;
                vertices[vi] = [new_pos[vi][0] / c, new_pos[vi][1] / c, new_pos[vi][2] / c];
            }
        }
    }
}

/// Remove inverted or degenerate tets from the mesh.
pub fn remove_bad_tets(vertices: &[[f64; 3]], tets: &mut Vec<[usize; 4]>) {
    tets.retain(|t| {
        let vol = tet_volume(
            vertices[t[0]],
            vertices[t[1]],
            vertices[t[2]],
            vertices[t[3]],
        );
        vol > 1e-14
    });
}

/// Compute mesh quality statistics.
pub struct MeshStats {
    pub num_tets: usize,
    pub min_quality: f64,
    pub avg_quality: f64,
    pub min_dihedral_deg: f64,
    pub avg_dihedral_deg: f64,
    pub vol_std_dev: f64,
}

/// Aggregate quality statistics (volumes, edge-ratio extremes, mean
/// and standard deviation) over a tetrahedral mesh - the metrics the
/// improve pass reports and tests assert against.
pub fn compute_stats(vertices: &[[f64; 3]], tets: &[[usize; 4]]) -> MeshStats {
    if tets.is_empty() {
        return MeshStats {
            num_tets: 0,
            min_quality: 0.0,
            avg_quality: 0.0,
            min_dihedral_deg: 0.0,
            avg_dihedral_deg: 0.0,
            vol_std_dev: 0.0,
        };
    }

    let mut min_q = f64::MAX;
    let mut sum_q = 0.0;
    let mut min_dh = f64::MAX;
    let mut sum_dh = 0.0;
    let mut volumes = Vec::with_capacity(tets.len());

    for t in tets {
        let a = vertices[t[0]];
        let b = vertices[t[1]];
        let c = vertices[t[2]];
        let d = vertices[t[3]];

        let q = tet_quality(a, b, c, d);
        min_q = min_q.min(q);
        sum_q += q;

        let dh = min_dihedral_angle(a, b, c, d);
        min_dh = min_dh.min(dh);
        sum_dh += dh;

        volumes.push(tet_volume(a, b, c, d).abs());
    }

    let n = tets.len() as f64;
    let avg_vol = volumes.iter().sum::<f64>() / n;
    let vol_var = volumes.iter().map(|v| (v - avg_vol).powi(2)).sum::<f64>() / n;

    MeshStats {
        num_tets: tets.len(),
        min_quality: min_q,
        avg_quality: sum_q / n,
        min_dihedral_deg: min_dh.to_degrees(),
        avg_dihedral_deg: (sum_dh / n).to_degrees(),
        vol_std_dev: vol_var.sqrt(),
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_equilateral() {
        // Regular tetrahedron centered at origin
        let s = 1.0;
        let a = [s, s, s];
        let b = [s, -s, -s];
        let c = [-s, s, -s];
        let d = [-s, -s, s];
        let q = tet_quality(a, b, c, d);
        assert!(
            q > 0.9,
            "Equilateral tet should have quality near 1.0, got {q}"
        );
    }

    #[test]
    fn quality_degenerate() {
        // Nearly flat tet
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.5, 0.5, 0.001];
        let q = tet_quality(a, b, c, d);
        assert!(q < 0.1, "Flat tet should have low quality, got {q}");
    }

    #[test]
    fn dihedral_angle_positive() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let angle = min_dihedral_angle(a, b, c, d);
        assert!(angle > 0.0 && angle < std::f64::consts::PI);
    }

    #[test]
    fn laplacian_smooth_moves_interior() {
        // Simple case: 1 interior vertex surrounded by 4 boundary vertices
        let mut vertices = vec![
            [0.0, 0.0, 0.0], // 0 boundary
            [2.0, 0.0, 0.0], // 1 boundary
            [1.0, 2.0, 0.0], // 2 boundary
            [1.0, 1.0, 2.0], // 3 boundary
            [1.5, 1.0, 0.8], // 4 interior (slightly off-center)
        ];
        let tets = vec![[0, 1, 2, 4], [0, 1, 4, 3], [0, 2, 3, 4], [1, 2, 3, 4]];
        let old_pos = vertices[4];
        laplacian_smooth(&mut vertices, &tets, 4, 5);
        // The interior point should move toward the centroid of its neighbors
        let new_pos = vertices[4];
        assert!(
            (new_pos[0] - old_pos[0]).abs() > 1e-6
                || (new_pos[1] - old_pos[1]).abs() > 1e-6
                || (new_pos[2] - old_pos[2]).abs() > 1e-6,
            "Interior vertex should have moved"
        );
    }
}
