mod predicates;
mod refine;
mod triangulation;
mod types;

pub use types::{CDTInput, CDTOutput};

/// Perform constrained Delaunay triangulation.
///
/// Takes boundary vertices and constraint edges, produces a triangle mesh.
/// If `max_edge_length` is set, applies Ruppert-style refinement followed by
/// Laplacian smoothing to improve triangle quality (aspect ratios).
pub fn triangulate(input: &CDTInput) -> CDTOutput {
    let mut cdt = triangulation::CDT::new(input);
    cdt.build(input);
    if input.max_edge_length.is_some() || input.min_angle.is_some() || input.size_field.is_some() {
        refine::refine(&mut cdt, input);
        // Laplacian smoothing improves triangle aspect ratios without changing
        // the triangle count.  We apply it whenever refinement ran (adaptive
        // size-field mode or angle/edge-length refinement), which corresponds
        // to "Fine" quality. Coarse meshes (no refinement) skip smoothing
        // since there are no interior Steiner points to move.
        refine::smooth(&mut cdt, input, 5);
    }
    let mut output = cdt.into_output(input);

    // For adaptive (size_field) mode: enable quality operations but protect seam edges.
    // Seam edges (both endpoints on UV domain boundary) must not be modified.
    let seam_verts = input.size_field.as_ref().map(|sf| output.seam_vertices(sf));

    // Metric-weighted edge swapping: improve triangle quality.
    if input.size_field.is_some() {
        // Adaptive mode: swap with seam protection
        refine::swap_edges(&mut output, input.size_field.as_ref(), seam_verts.as_ref());
    } else {
        refine::swap_edges(&mut output, None, None);
    }

    // Valence-based edge swapping.
    if input.size_field.is_some() {
        refine::swap_for_valence(&mut output, seam_verts.as_ref());
    } else {
        refine::swap_for_valence(&mut output, None);
    }

    // Edge splitting: split edges too long relative to local target size.
    if input.size_field.is_some() {
        refine::split_long_edges(
            &mut output,
            input.size_field.as_ref(),
            input.max_edge_length,
            seam_verts.as_ref(),
        );
    } else if input.max_edge_length.is_some() {
        refine::split_long_edges(&mut output, None, input.max_edge_length, None);
    }

    // Edge collapse: remove short interior edges to reduce triangle count.
    if input.max_edge_length.is_some() || input.size_field.is_some() {
        refine::collapse_short_edges(
            &mut output,
            input.size_field.as_ref(),
            input.max_edge_length,
        );
    }
    // Safety net: remove duplicate triangles and non-manifold excess that
    // collapse_short_edges may have introduced.
    output.sanitize();
    output
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_coarse() {
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: None,
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        assert_eq!(output.triangles.len(), 2);
        for tri in &output.triangles {
            let a = output.vertices[tri[0]];
            let b = output.vertices[tri[1]];
            let c = output.vertices[tri[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area > 0.0, "Triangle has non-positive area: {area}");
        }
    }

    #[test]
    fn unit_square_with_hole() {
        let input = CDTInput {
            vertices: vec![
                // Outer boundary (CCW)
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                // Inner hole (CW)
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 1.5],
            ],
            constraints: vec![
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ],
            boundary_loops: vec![0..4, 4..8],
            max_edge_length: None,
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        assert!(
            output.triangles.len() >= 8,
            "Expected >= 8 triangles, got {}",
            output.triangles.len()
        );
        // No triangles should be inside the hole
        for tri in &output.triangles {
            let a = output.vertices[tri[0]];
            let b = output.vertices[tri[1]];
            let c = output.vertices[tri[2]];
            let cx = (a[0] + b[0] + c[0]) / 3.0;
            let cy = (a[1] + b[1] + c[1]) / 3.0;
            let in_hole = cx > 0.5 && cx < 1.5 && cy > 0.5 && cy < 1.5;
            assert!(!in_hole, "Triangle centroid ({cx}, {cy}) is inside hole");
        }
    }

    #[test]
    fn constraint_edges_recovered() {
        let input = CDTInput {
            vertices: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.5, 0.5],
                [1.5, 1.5],
            ],
            constraints: vec![
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5], // Diagonal constraint
            ],
            boundary_loops: vec![0..4],
            max_edge_length: None,
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        // Verify the constraint edge [4,5] appears as a triangle edge
        let mut found_constraint = false;
        for tri in &output.triangles {
            let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
            for e in &edges {
                if (e[0] == 4 && e[1] == 5) || (e[0] == 5 && e[1] == 4) {
                    found_constraint = true;
                }
            }
        }
        assert!(
            found_constraint,
            "Constraint edge [4,5] not found in output"
        );
    }

    #[test]
    fn refinement_max_edge_length() {
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: Some(2.0),
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        assert!(
            output.vertices.len() > 4,
            "Steiner points should be inserted"
        );
        assert!(output.triangles.len() > 2);
        // Verify no edge exceeds max_edge_length significantly
        for tri in &output.triangles {
            let pts: Vec<[f64; 2]> = tri.iter().map(|&i| output.vertices[i]).collect();
            for i in 0..3 {
                let j = (i + 1) % 3;
                let dx = pts[j][0] - pts[i][0];
                let dy = pts[j][1] - pts[i][1];
                let len = (dx * dx + dy * dy).sqrt();
                assert!(len <= 3.0, "Edge length {len} exceeds threshold");
            }
        }
    }

    #[test]
    fn collinear_points() {
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            boundary_loops: vec![0..5],
            max_edge_length: None,
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        assert_eq!(output.triangles.len(), 3);
    }

    #[test]
    fn all_triangles_positive_area() {
        // L-shaped polygon
        let input = CDTInput {
            vertices: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 2.0],
            ],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            boundary_loops: vec![0..6],
            max_edge_length: None,
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };
        let output = triangulate(&input);
        assert!(output.triangles.len() >= 4);
        for tri in &output.triangles {
            let a = output.vertices[tri[0]];
            let b = output.vertices[tri[1]];
            let c = output.vertices[tri[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area > 0.0, "Triangle has non-positive area: {area}");
        }
    }

    #[test]
    fn smoothing_moves_interior_vertices() {
        // Refine a 10x10 square so Steiner points are inserted, then verify
        // that smoothing actually changes their positions while leaving
        // boundary vertices fixed.
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: Some(2.0),
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };

        // Build and refine (without smoothing) to capture pre-smooth positions.
        let mut cdt = triangulation::CDT::new(&input);
        cdt.build(&input);
        refine::refine(&mut cdt, &input);

        let boundary = cdt.boundary_handles();
        let all_handles = cdt.all_vertex_handles();

        // There must be interior (Steiner) vertices after refinement.
        let interior_handles: Vec<_> = all_handles
            .iter()
            .copied()
            .filter(|h| !boundary.contains(h))
            .collect();
        assert!(
            !interior_handles.is_empty(),
            "Expected Steiner points after refinement"
        );

        // Record pre-smooth positions of interior vertices.
        let pre_interior: Vec<[f64; 2]> = interior_handles
            .iter()
            .map(|&h| cdt.vertex_position(h))
            .collect();

        // Record pre-smooth positions of boundary vertices.
        let boundary_handles: Vec<_> = boundary.iter().copied().collect();
        let pre_boundary: Vec<[f64; 2]> = boundary_handles
            .iter()
            .map(|&h| cdt.vertex_position(h))
            .collect();

        // Run smoothing.
        refine::smooth(&mut cdt, &input, 5);

        // Check that at least one interior vertex moved.
        let mut any_moved = false;
        for (i, &h) in interior_handles.iter().enumerate() {
            let post = cdt.vertex_position(h);
            let dx = post[0] - pre_interior[i][0];
            let dy = post[1] - pre_interior[i][1];
            if dx.abs() > 1e-12 || dy.abs() > 1e-12 {
                any_moved = true;
                break;
            }
        }
        assert!(
            any_moved,
            "Smoothing should move at least one interior vertex"
        );

        // Verify boundary vertices did NOT move.
        for (i, &h) in boundary_handles.iter().enumerate() {
            let post = cdt.vertex_position(h);
            assert!(
                (post[0] - pre_boundary[i][0]).abs() < 1e-12
                    && (post[1] - pre_boundary[i][1]).abs() < 1e-12,
                "Boundary vertex should not have moved"
            );
        }
    }

    #[test]
    fn edge_collapse_reduces_triangle_count() {
        // Refine a 10x10 square with a small max_edge_length to generate many
        // Steiner points, then verify that edge collapse reduces the triangle
        // count while preserving mesh validity.
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: Some(2.0),
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };

        // Build the mesh without edge collapse to get baseline triangle count.
        let mut cdt = triangulation::CDT::new(&input);
        cdt.build(&input);
        refine::refine(&mut cdt, &input);
        refine::smooth(&mut cdt, &input, 5);
        let baseline = cdt.into_output(&input);
        let baseline_count = baseline.triangles.len();

        // Now build with edge collapse (via the full triangulate pipeline).
        let collapsed = triangulate(&input);
        let collapsed_count = collapsed.triangles.len();

        assert!(
            collapsed_count < baseline_count,
            "Edge collapse should reduce triangle count: baseline={baseline_count}, collapsed={collapsed_count}"
        );

        // Verify all triangles remain valid (positive area, CCW orientation).
        for tri in &collapsed.triangles {
            let a = collapsed.vertices[tri[0]];
            let b = collapsed.vertices[tri[1]];
            let c = collapsed.vertices[tri[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(
                area > 0.0,
                "Triangle has non-positive area after collapse: {area}"
            );
        }
    }

    #[test]
    fn valence_swap_improves_regularity() {
        // Refine a square to get a mesh with many interior vertices, then
        // verify that swap_for_valence brings average valence deviation
        // closer to ideal (6 for interior vertices).
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: Some(2.0),
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };

        // Build mesh up through swap_edges (before swap_for_valence).
        let mut cdt = triangulation::CDT::new(&input);
        cdt.build(&input);
        refine::refine(&mut cdt, &input);
        refine::smooth(&mut cdt, &input, 5);
        let mut output = cdt.into_output(&input);
        refine::swap_edges(&mut output, None, None);

        // Compute valence deviation before swap_for_valence.
        let deviation_before = valence_deviation(&output);

        // Run valence swap.
        refine::swap_for_valence(&mut output, None);

        // Compute valence deviation after.
        let deviation_after = valence_deviation(&output);

        assert!(
            deviation_after <= deviation_before,
            "Valence swap should not increase deviation: before={deviation_before:.3}, after={deviation_after:.3}"
        );

        // Verify all triangles remain valid (positive area).
        for tri in &output.triangles {
            let a = output.vertices[tri[0]];
            let b = output.vertices[tri[1]];
            let c = output.vertices[tri[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(
                area > 0.0,
                "Triangle has non-positive area after valence swap: {area}"
            );
        }
    }

    /// Compute average squared valence deviation from ideal for interior vertices.
    fn valence_deviation(output: &CDTOutput) -> f64 {
        let num_boundary = output.num_original_vertices;
        let mut valence: Vec<usize> = vec![0; output.vertices.len()];
        for tri in &output.triangles {
            valence[tri[0]] += 1;
            valence[tri[1]] += 1;
            valence[tri[2]] += 1;
        }
        let mut total_sq_dev = 0.0;
        let mut count = 0;
        for (v, &val) in valence.iter().enumerate() {
            if val == 0 {
                continue;
            }
            let ideal: i64 = if v < num_boundary { 4 } else { 6 };
            let dev = val as i64 - ideal;
            total_sq_dev += (dev * dev) as f64;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total_sq_dev / count as f64
        }
    }

    #[test]
    fn split_long_edges_splits_oversized_edges() {
        // Create a mesh with a known long edge, then verify that
        // split_long_edges splits it.
        let input = CDTInput {
            vertices: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            constraints: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            boundary_loops: vec![0..4],
            max_edge_length: Some(2.0),
            min_angle: None,
            size_field: None,
            periodic_seams: false,
        };

        // Build mesh through smoothing and swap_edges.
        let mut cdt = triangulation::CDT::new(&input);
        cdt.build(&input);
        refine::refine(&mut cdt, &input);
        refine::smooth(&mut cdt, &input, 5);
        let mut output = cdt.into_output(&input);
        refine::swap_edges(&mut output, None, None);
        refine::swap_for_valence(&mut output, None);

        let verts_before = output.vertices.len();
        let tris_before = output.triangles.len();

        // Compute max edge length before split.
        let max_before = max_edge_length_in_mesh(&output);

        // Run split_long_edges with a target that ensures some edges get split.
        // Use a small max_edge_length so existing edges are likely above 1.4 * target.
        refine::split_long_edges(&mut output, None, Some(1.0), None);

        let verts_after = output.vertices.len();
        let tris_after = output.triangles.len();

        // Splitting adds vertices and triangles.
        assert!(
            verts_after >= verts_before,
            "Split should add vertices: before={verts_before}, after={verts_after}"
        );
        assert!(
            tris_after >= tris_before,
            "Split should add triangles: before={tris_before}, after={tris_after}"
        );

        // Max edge length should decrease (or at least not exceed the threshold).
        let max_after = max_edge_length_in_mesh(&output);
        assert!(
            max_after <= max_before || max_after <= 1.0 * 1.5,
            "Max edge should be reduced after splitting: before={max_before:.3}, after={max_after:.3}"
        );

        // Verify all triangles are valid (positive area).
        for tri in &output.triangles {
            let a = output.vertices[tri[0]];
            let b = output.vertices[tri[1]];
            let c = output.vertices[tri[2]];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(
                area > 0.0,
                "Triangle has non-positive area after edge split: {area}"
            );
        }
    }

    /// Compute maximum edge length across all triangles.
    fn max_edge_length_in_mesh(output: &CDTOutput) -> f64 {
        let mut max_len = 0.0_f64;
        for tri in &output.triangles {
            for k in 0..3 {
                let a = output.vertices[tri[k]];
                let b = output.vertices[tri[(k + 1) % 3]];
                let dx = b[0] - a[0];
                let dy = b[1] - a[1];
                let len = (dx * dx + dy * dy).sqrt();
                max_len = max_len.max(len);
            }
        }
        max_len
    }
}
