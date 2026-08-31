use crate::cdt::{self, CDTInput};
use crate::error::{MesherError, Result};

use super::types::{FaceInput, FaceOutput, MeshMode};

/// Mesh a single face in UV space.
///
/// Assembles boundary polylines from edges, runs CDT, and returns
/// the interior vertices + triangle connectivity.
///
/// # Errors
///
/// Returns `MesherError::InvalidInput` if:
/// - `boundary_edges` is empty
/// - any edge has fewer than 2 `uv_points`
/// - `target_edge_length` is not positive in `Fine` mode
pub fn mesh_face(input: &FaceInput) -> Result<FaceOutput> {
    // --- Input validation ---
    if input.boundary_edges.is_empty() {
        return Err(MesherError::InvalidInput(
            "boundary_edges must not be empty".into(),
        ));
    }
    for (i, edge) in input.boundary_edges.iter().enumerate() {
        if edge.uv_points.len() < 2 {
            return Err(MesherError::InvalidInput(format!(
                "boundary_edges[{i}] must have at least 2 uv_points, got {}",
                edge.uv_points.len()
            )));
        }
    }
    for (hi, hole) in input.holes.iter().enumerate() {
        for (ei, edge) in hole.iter().enumerate() {
            if edge.uv_points.len() < 2 {
                return Err(MesherError::InvalidInput(format!(
                    "holes[{hi}][{ei}] must have at least 2 uv_points, got {}",
                    edge.uv_points.len()
                )));
            }
        }
    }
    if let MeshMode::Fine { target_edge_length } = &input.mode {
        if *target_edge_length <= 0.0 {
            return Err(MesherError::InvalidInput(format!(
                "target_edge_length must be positive, got {target_edge_length}"
            )));
        }
    }
    // Step 1: Collect all boundary vertices from edges
    let (boundary_vertices, outer_constraints, outer_loop_len) =
        collect_wire_vertices(&input.boundary_edges);

    let mut all_vertices = boundary_vertices;
    let mut all_constraints = outer_constraints;
    #[allow(clippy::single_range_in_vec_init)]
    let mut boundary_loops = vec![0..outer_loop_len];

    // Step 2: Add hole wires
    for hole in &input.holes {
        let constraint_offset = all_constraints.len();
        let vertex_offset = all_vertices.len();
        let (hole_verts, hole_constraints, hole_loop_len) = collect_wire_vertices(hole);

        // Offset vertex indices in hole constraints
        for c in &hole_constraints {
            all_constraints.push([c[0] + vertex_offset, c[1] + vertex_offset]);
        }
        all_vertices.extend(hole_verts);
        boundary_loops.push(constraint_offset..constraint_offset + hole_loop_len);
    }

    // Step 3: Build CDT input
    let (max_edge_length, min_angle, size_field) = match &input.mode {
        MeshMode::Coarse => (None, None, None),
        MeshMode::Fine { target_edge_length } => (Some(*target_edge_length), Some(20.0), None),
        MeshMode::Adaptive { size_field } => (None, Some(20.0), Some((**size_field).clone())),
    };

    // Periodic seam exclusion is disabled - the CDT sanitize() +
    // boundary-only merge + close_open_edges pipeline handles seam
    // topology robustly without needing to exclude Steiner points.
    let periodic_seams = false;

    let cdt_input = CDTInput {
        vertices: all_vertices,
        constraints: all_constraints,
        boundary_loops,
        max_edge_length,
        min_angle,
        periodic_seams,
        size_field,
    };

    // Step 4: Triangulate
    let cdt_output = cdt::triangulate(&cdt_input);

    // Step 5: Extract interior vertices (Steiner points from refinement).
    // cdt_output.num_original_vertices tells us how many are input boundary vertices.
    // Any vertices beyond that are Steiner points inserted during refinement.
    //
    // CLAMPED: on degenerate faces (self-touching UV boundaries from dirty
    // CAD - GEOUNED's FWTBM1.step), spade DEDUPLICATES coincident input
    // points, so the output can hold FEWER vertices than
    // `num_original_vertices`; the unchecked slice panicked through PyO3 and
    // killed the whole assembly. Such a face yields no Steiner points.
    let n_orig = cdt_output
        .num_original_vertices
        .min(cdt_output.vertices.len());
    let interior_uv_vertices: Vec<[f64; 2]> = cdt_output.vertices[n_orig..].to_vec();

    Ok(FaceOutput {
        face_id: input.face_id,
        interior_uv_vertices,
        triangles: cdt_output.triangles,
        num_boundary_vertices: n_orig,
        boundary_uv_vertices: cdt_output.vertices[..n_orig].to_vec(),
    })
}

/// Collect vertices and constraint edges from a closed wire of edges.
/// Returns (vertices, constraints, num_constraints).
fn collect_wire_vertices(
    edges: &[super::types::EdgeInput],
) -> (Vec<[f64; 2]>, Vec<[usize; 2]>, usize) {
    let mut vertices: Vec<[f64; 2]> = Vec::new();
    let mut constraints: Vec<[usize; 2]> = Vec::new();

    // Junction weld tolerance. DELIBERATELY absolute and tight: a
    // scale-aware weld (1e-9 x UV span) was tried for GEOUNED's
    // FWTBM1.step (pcurve corner jogs ~3e-10 that make the boundary
    // self-intersect), but GEOUNED's RJ24.stp has REAL self-touch gaps
    // down to the same relative scale (2.4e-13 of span) that must stay
    // distinct - welding them pinches loops and broke its triangulation
    // (43 tris lost=0 -> 26 tris lost=6221). No threshold separates the
    // two regimes; corner jogs are instead handled by the
    // can_add_constraint guard in the CDT (skip the crossing
    // constraint), which degrades gracefully. The principled upgrade
    // EXISTS upstream: the extraction layer's topology-aware junction
    // snap (apply_junction_snaps - gated on a shared OCC vertex plus a
    // relative jog-size bound) repairs true jogs before the UVs ever
    // reach this weld, so this stays a pure exact-duplicate filter.
    let weld_eps = 1e-12;

    for edge in edges {
        let points = if edge.reversed {
            edge.uv_points.iter().rev().cloned().collect::<Vec<_>>()
        } else {
            edge.uv_points.clone()
        };

        // Add points, skipping the first if it duplicates the last added vertex
        for (i, &pt) in points.iter().enumerate() {
            if i == 0 && !vertices.is_empty() {
                let last = vertices.last().unwrap();
                let dx = (last[0] - pt[0]).abs();
                let dy = (last[1] - pt[1]).abs();
                if dx < weld_eps && dy < weld_eps {
                    continue; // Skip duplicate at edge junction
                }
            }
            vertices.push(pt);
        }
    }

    // Remove closing vertex if it duplicates the first (wire is closed)
    if vertices.len() >= 2 {
        let first = vertices[0];
        let last = vertices[vertices.len() - 1];
        let dx = (first[0] - last[0]).abs();
        let dy = (first[1] - last[1]).abs();
        if dx < weld_eps && dy < weld_eps {
            vertices.pop();
        }
    }

    // Build constraint edges along the polyline (closed loop)
    let n = vertices.len();
    for i in 0..n {
        constraints.push([i, (i + 1) % n]);
    }

    let num_constraints = constraints.len();
    (vertices, constraints, num_constraints)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::super::types::{EdgeInput, MeshMode};
    use super::*;

    #[test]
    fn mesh_square_face_coarse() {
        let input = FaceInput {
            face_id: 1,
            boundary_edges: vec![
                EdgeInput {
                    edge_id: 1,
                    uv_points: vec![[0.0, 0.0], [1.0, 0.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 2,
                    uv_points: vec![[1.0, 0.0], [1.0, 1.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 3,
                    uv_points: vec![[1.0, 1.0], [0.0, 1.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 4,
                    uv_points: vec![[0.0, 1.0], [0.0, 0.0]],
                    reversed: false,
                },
            ],
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        };
        let output = mesh_face(&input).unwrap();
        assert_eq!(output.face_id, 1);
        assert_eq!(output.num_boundary_vertices, 4);
        assert_eq!(output.triangles.len(), 2);
        assert!(output.interior_uv_vertices.is_empty());
    }

    #[test]
    fn mesh_square_face_fine() {
        let input = FaceInput {
            face_id: 2,
            boundary_edges: vec![
                EdgeInput {
                    edge_id: 1,
                    uv_points: vec![[0.0, 0.0], [10.0, 0.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 2,
                    uv_points: vec![[10.0, 0.0], [10.0, 10.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 3,
                    uv_points: vec![[10.0, 10.0], [0.0, 10.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 4,
                    uv_points: vec![[0.0, 10.0], [0.0, 0.0]],
                    reversed: false,
                },
            ],
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Fine {
                target_edge_length: 2.0,
            },
        };
        let output = mesh_face(&input).unwrap();
        assert!(output.triangles.len() > 2);
        assert!(!output.interior_uv_vertices.is_empty());
    }

    #[test]
    fn mesh_face_subnormal_coordinates() {
        // GEOUNED FWTBM1.step regression: a plane face whose pcurve emits
        // SUBNORMAL v-coordinates (3.2e-45). spade rejects subnormals, and
        // the old fallback collapsed the boundary loop into a single
        // vertex - the face silently emitted zero triangles and the solid
        // leaked thousands of particles. The clamp maps subnormals to 0.
        let denormal = 3.196712121740989e-45_f64;
        let pts = [
            [-448.9999999998905, denormal],
            [-434.917000000607, denormal],
            [-434.9170000006065, -1470.0],
            [-448.99999999989006, -1470.0],
        ];
        let input = FaceInput {
            face_id: 29,
            boundary_edges: (0..4)
                .map(|i| EdgeInput {
                    edge_id: i as u64,
                    uv_points: vec![pts[i], pts[(i + 1) % 4]],
                    reversed: false,
                })
                .collect(),
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        };
        let output = mesh_face(&input).unwrap();
        assert_eq!(output.triangles.len(), 2);
    }

    #[test]
    fn mesh_face_self_intersecting_boundary_no_panic() {
        // GEOUNED FWTBM1.step / RJ24.stp regression: a self-touching UV
        // boundary produces intersecting constraint edges, and spade's
        // add_constraint PANICS on those - crashing the whole assembly
        // through PyO3. The can_add_constraint guard skips the offending
        // constraint instead; the face may degrade but the run survives.
        let pts = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]; // bowtie
        let input = FaceInput {
            face_id: 7,
            boundary_edges: (0..4)
                .map(|i| EdgeInput {
                    edge_id: i as u64,
                    uv_points: vec![pts[i], pts[(i + 1) % 4]],
                    reversed: false,
                })
                .collect(),
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        };
        // Must not panic.
        let _ = mesh_face(&input);
    }

    #[test]
    fn mesh_face_junction_jog_no_panic() {
        // GEOUNED FWTBM1.step: consecutive wire edges evaluate their
        // shared corner to UVs differing by ~3e-10 on a face whose UV span
        // is ~1500 - a microscopic "jog" that makes the boundary
        // self-intersect under spade's exact predicates. These jogs are
        // NOT welded away (see collect_wire_vertices for why a scale-aware
        // weld is unsafe); the can_add_constraint guard skips the crossing
        // constraint instead, and the face must still produce a usable
        // triangulation without panicking.
        let noise = 3.3e-10_f64;
        let input = FaceInput {
            face_id: 173,
            boundary_edges: vec![
                EdgeInput {
                    edge_id: 1,
                    uv_points: vec![[34.0, -735.0], [34.0, 735.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 2,
                    uv_points: vec![[34.0 + noise, 735.0], [0.0, 735.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 3,
                    uv_points: vec![[0.0 - noise, 735.0], [0.0, -735.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 4,
                    uv_points: vec![[0.0, -735.0 - noise], [34.0, -735.0]],
                    reversed: false,
                },
            ],
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        };
        let output = mesh_face(&input).unwrap();
        assert!(!output.triangles.is_empty());
    }

    #[test]
    fn mesh_reversed_edge() {
        // Same square but one edge is reversed
        let input = FaceInput {
            face_id: 3,
            boundary_edges: vec![
                EdgeInput {
                    edge_id: 1,
                    uv_points: vec![[0.0, 0.0], [1.0, 0.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 2,
                    uv_points: vec![[1.0, 1.0], [1.0, 0.0]], // stored reversed
                    reversed: true,                          // flip it back
                },
                EdgeInput {
                    edge_id: 3,
                    uv_points: vec![[1.0, 1.0], [0.0, 1.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: 4,
                    uv_points: vec![[0.0, 1.0], [0.0, 0.0]],
                    reversed: false,
                },
            ],
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        };
        let output = mesh_face(&input).unwrap();
        assert_eq!(output.triangles.len(), 2);
    }
}
