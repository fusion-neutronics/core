use rayon::prelude::*;

use crate::error::Result;

use super::face::mesh_face;
use super::types::{FaceInput, FaceOutput};

/// Mesh multiple faces sequentially.
pub fn mesh_faces(inputs: &[FaceInput]) -> Result<Vec<FaceOutput>> {
    inputs.iter().map(mesh_face).collect()
}

/// Mesh multiple faces in parallel using rayon.
///
/// Faces are sorted by estimated cost (largest first) to keep all cores busy.
pub fn mesh_faces_parallel(inputs: &[FaceInput]) -> Result<Vec<FaceOutput>> {
    if inputs.len() <= 1 {
        return mesh_faces(inputs);
    }

    // Sort by estimated cost descending (more edges = likely more expensive)
    let mut indexed: Vec<(usize, &FaceInput)> = inputs.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        let cost_a = estimate_cost(a);
        let cost_b = estimate_cost(b);
        cost_b
            .partial_cmp(&cost_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mesh in parallel - collect Results
    let results: Vec<(usize, Result<FaceOutput>)> = indexed
        .par_iter()
        .map(|(orig_idx, input)| (*orig_idx, mesh_face(input)))
        .collect();

    // Check for errors and restore original order
    let mut ordered: Vec<(usize, FaceOutput)> = Vec::with_capacity(results.len());
    for (idx, res) in results {
        ordered.push((idx, res?));
    }
    ordered.sort_by_key(|(idx, _)| *idx);
    Ok(ordered.into_iter().map(|(_, output)| output).collect())
}

/// Estimate the cost of meshing a face (higher = more expensive).
fn estimate_cost(input: &FaceInput) -> f64 {
    let total_points: usize = input.boundary_edges.iter().map(|e| e.uv_points.len()).sum();
    let hole_points: usize = input
        .holes
        .iter()
        .flat_map(|w| w.iter())
        .map(|e| e.uv_points.len())
        .sum();
    let is_fine = matches!(
        input.mode,
        super::types::MeshMode::Fine { .. } | super::types::MeshMode::Adaptive { .. }
    );
    let base = (total_points + hole_points) as f64;
    if is_fine {
        base * 10.0
    } else {
        base
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::types::{EdgeInput, MeshMode};

    fn make_square(id: u64) -> FaceInput {
        FaceInput {
            face_id: id,
            boundary_edges: vec![
                EdgeInput {
                    edge_id: id * 10 + 1,
                    uv_points: vec![[0.0, 0.0], [1.0, 0.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: id * 10 + 2,
                    uv_points: vec![[1.0, 0.0], [1.0, 1.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: id * 10 + 3,
                    uv_points: vec![[1.0, 1.0], [0.0, 1.0]],
                    reversed: false,
                },
                EdgeInput {
                    edge_id: id * 10 + 4,
                    uv_points: vec![[0.0, 1.0], [0.0, 0.0]],
                    reversed: false,
                },
            ],
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
        }
    }

    #[test]
    fn batch_two_squares() {
        let inputs: Vec<FaceInput> = (1..=2).map(make_square).collect();
        let outputs = mesh_faces(&inputs).unwrap();
        assert_eq!(outputs.len(), 2);
        for out in &outputs {
            assert_eq!(out.num_boundary_vertices, 4);
            assert_eq!(out.triangles.len(), 2);
        }
    }

    #[test]
    fn parallel_same_as_sequential() {
        let inputs: Vec<FaceInput> = (1..=10).map(make_square).collect();
        let seq = mesh_faces(&inputs).unwrap();
        let par = mesh_faces_parallel(&inputs).unwrap();
        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.face_id, p.face_id);
            assert_eq!(s.triangles.len(), p.triangles.len());
            assert_eq!(s.num_boundary_vertices, p.num_boundary_vertices);
        }
    }
}
