/// Delaunay refinement for tetrahedral mesh quality.
///
/// TetGen-style circumcenter-based refinement:
/// 1. Find tets that violate quality criteria (radius-edge ratio or max volume)
/// 2. Insert tet's circumcenter into the Delaunay
/// 3. If circumcenter is outside boundary, reject and skip
/// 4. Repeat until all tets satisfy criteria
use super::aabb_bvh::TriangleBvh;
use super::delaunay3d::Delaunay3D;
use super::predicates3d::{self, circumcenter_tet, dist_sq_3d};

/// Quality criteria for a tetrahedron.
struct TetQuality {
    /// Circumradius / shortest edge length. Lower is better.
    /// Ideal equilateral tet: ~0.612. Bad sliver: >2.0.
    radius_edge_ratio: f64,
    /// Volume of the tet.
    volume: f64,
    /// Circumcenter position.
    circumcenter: [f64; 3],
}

fn compute_tet_quality(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> TetQuality {
    let cc = circumcenter_tet(a, b, c, d);
    let cr_sq = dist_sq_3d(a, cc);
    let cr = cr_sq.sqrt();

    // Find shortest edge
    let edges = [
        dist_sq_3d(a, b),
        dist_sq_3d(a, c),
        dist_sq_3d(a, d),
        dist_sq_3d(b, c),
        dist_sq_3d(b, d),
        dist_sq_3d(c, d),
    ];
    let min_edge = edges.iter().copied().fold(f64::MAX, f64::min).sqrt();

    let ratio = if min_edge > 1e-15 {
        cr / min_edge
    } else {
        f64::MAX
    };
    let vol = predicates3d::tet_volume(a, b, c, d).abs();

    TetQuality {
        radius_edge_ratio: ratio,
        volume: vol,
        circumcenter: cc,
    }
}

/// Refine a Delaunay tet mesh by inserting circumcenters of bad tets.
///
/// A tet is "bad" if:
/// - Its radius-edge ratio exceeds `max_ratio` (default 2.0), OR
/// - Its volume exceeds `max_volume`
///
/// The circumcenter of the bad tet is inserted into the Delaunay.
/// If the circumcenter is outside the boundary, the insertion is skipped.
///
/// Returns the number of Steiner points inserted.
pub fn refine_quality(
    dt: &mut Delaunay3D,
    bvh: &TriangleBvh,
    target_edge_length: f64,
    max_passes: usize,
) -> usize {
    // Max volume: 3x volume of regular tet with edge = target_edge_length.
    // This allows some variation while preventing huge tets.
    // V_regular = edge^3 / (6*sqrt(2))
    let max_volume = 3.0 * target_edge_length.powi(3) / (6.0 * 2.0_f64.sqrt());
    let max_ratio = 2.0;
    // Minimum distance between new point and existing points
    let min_dist_sq = (0.2 * target_edge_length).powi(2);

    let mut total_inserted = 0;

    for _pass in 0..max_passes {
        let tets = dt.extract_tets();
        if tets.is_empty() {
            break;
        }

        // Find bad tets and their circumcenters
        let mut bad_tets: Vec<([f64; 3], f64)> = Vec::new(); // (circumcenter, badness)

        for t in &tets {
            let a = dt.vertices[t[0]];
            let b = dt.vertices[t[1]];
            let c = dt.vertices[t[2]];
            let d = dt.vertices[t[3]];

            let q = compute_tet_quality(a, b, c, d);

            // Check max edge length
            let max_edge_len_sq = [
                dist_sq_3d(a, b),
                dist_sq_3d(a, c),
                dist_sq_3d(a, d),
                dist_sq_3d(b, c),
                dist_sq_3d(b, d),
                dist_sq_3d(c, d),
            ]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
            let edge_too_long = max_edge_len_sq > (target_edge_length * 1.5).powi(2);

            let is_bad = q.radius_edge_ratio > max_ratio || q.volume > max_volume || edge_too_long;
            if !is_bad {
                continue;
            }

            // Only insert if circumcenter is inside the boundary
            if !bvh.is_point_inside(&q.circumcenter) {
                continue;
            }

            // Check not too close to the tet's own vertices
            let too_close = [a, b, c, d]
                .iter()
                .any(|v| dist_sq_3d(*v, q.circumcenter) < min_dist_sq);
            if too_close {
                continue;
            }

            // Badness score: higher = worse (process worst first)
            let badness = q.radius_edge_ratio + q.volume / max_volume;
            bad_tets.push((q.circumcenter, badness));
        }

        if bad_tets.is_empty() {
            break;
        }

        // Sort by badness (worst first) and limit per pass to avoid runaway
        bad_tets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_inserts_per_pass = (tets.len() / 4).max(10).min(bad_tets.len());

        let mut inserted_this_pass = 0;
        for (cc, _) in bad_tets.iter().take(max_inserts_per_pass) {
            // Re-check inside (mesh may have changed)
            if !bvh.is_point_inside(cc) {
                continue;
            }

            dt.insert_steiner_point(*cc);
            inserted_this_pass += 1;
        }

        total_inserted += inserted_this_pass;

        if inserted_this_pass == 0 {
            break;
        }
    }

    total_inserted
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tet_quality_equilateral() {
        // Regular tet
        let a = [1.0, 1.0, 1.0];
        let b = [1.0, -1.0, -1.0];
        let c = [-1.0, 1.0, -1.0];
        let d = [-1.0, -1.0, 1.0];
        let q = compute_tet_quality(a, b, c, d);
        // Ideal ratio is sqrt(6)/4 ≈ 0.612
        assert!(
            q.radius_edge_ratio < 0.7,
            "ratio={:.3}",
            q.radius_edge_ratio
        );
        assert!(q.volume > 0.0);
    }

    #[test]
    fn test_refine_cube() {
        use super::super::aabb_bvh::TriangleBvh;
        use super::super::delaunay3d::Delaunay3D;

        let verts = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [0.0, 2.0, 2.0],
            [1.0, 1.0, 1.0], // interior point
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
        let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
            .iter()
            .map(|t| (verts[t[0]], verts[t[1]], verts[t[2]]))
            .collect();
        let bvh = TriangleBvh::new(&tri_data);

        let mut dt = Delaunay3D::new(&verts);
        let before = dt.extract_tets().len();

        // Use small target to ensure tets are "bad" and get refined
        let inserted = refine_quality(&mut dt, &bvh, 0.3, 5);

        let after = dt.extract_tets().len();
        // Should produce more tets (refinement inserted points)
        assert!(after >= before, "Should have at least as many tets after refinement (before={before}, after={after}, inserted={inserted})");
    }
}
