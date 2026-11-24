use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon;

use crate::error::{MesherError, Result};

use super::face::mesh_face;
use super::types::{FaceInput, FaceOutput, MeshMode};

/// A face task with its priority for the scheduling queue.
struct PrioritizedFace {
    index: usize,
    cost: f64,
}

impl PartialEq for PrioritizedFace {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for PrioritizedFace {}

impl PartialOrd for PrioritizedFace {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedFace {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.cost
            .partial_cmp(&other.cost)
            .unwrap_or(CmpOrdering::Equal)
    }
}

/// DAG-based mesh scheduler.
///
/// Tracks edge->face dependencies. All edge tasks (discretization) are assumed
/// done before entering Rust (Python handles edge discretization). In this
/// implementation, we focus on face-level scheduling with cost-based priority.
///
/// The true edge->face dependency DAG (edge discretization in Rust,
/// atomic-counter face unblocking) lives in [`super::scene`]; this
/// scheduler remains the path for callers that arrive with fully
/// evaluated `FaceInput`s.
pub struct MeshDag {
    faces: Vec<FaceInput>,
    estimated_costs: Vec<f64>,
}

impl MeshDag {
    /// Create a new DAG from face inputs.
    pub fn new(faces: Vec<FaceInput>) -> Self {
        let estimated_costs: Vec<f64> = faces.iter().map(estimate_cost).collect();
        MeshDag {
            faces,
            estimated_costs,
        }
    }

    /// Execute the DAG, meshing faces in priority order (most expensive first).
    ///
    /// Uses rayon's work-stealing to keep all cores busy. The priority queue
    /// ensures expensive faces start first, so shorter faces fill in gaps.
    pub fn execute(self) -> Result<Vec<FaceOutput>> {
        let n = self.faces.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            return Ok(vec![mesh_face(&self.faces[0])?]);
        }

        // Build priority queue (most expensive first)
        let queue = Mutex::new(BinaryHeap::new());
        {
            let mut q = queue.lock().unwrap();
            for (i, cost) in self.estimated_costs.iter().enumerate() {
                q.push(PrioritizedFace {
                    index: i,
                    cost: *cost,
                });
            }
        }

        // Results array (initialized with None, filled by workers)
        let results: Vec<Mutex<Option<Result<FaceOutput>>>> =
            (0..n).map(|_| Mutex::new(None)).collect();

        let completed = AtomicUsize::new(0);

        // Use rayon scope for work-stealing parallelism
        rayon::scope(|s| {
            let num_threads = rayon::current_num_threads();
            for _ in 0..num_threads {
                let queue = &queue;
                let results = &results;
                let faces = &self.faces;
                let completed = &completed;

                s.spawn(move |_| {
                    loop {
                        // Grab next task from priority queue
                        let task = {
                            let mut q = queue.lock().unwrap();
                            q.pop()
                        };

                        match task {
                            Some(pf) => {
                                let output = mesh_face(&faces[pf.index]);
                                *results[pf.index].lock().unwrap() = Some(output);
                                completed.fetch_add(1, Ordering::Relaxed);
                            }
                            None => {
                                // Queue empty - check if all done
                                if completed.load(Ordering::Relaxed) >= faces.len() {
                                    break;
                                }
                                // Yield to let other threads finish
                                std::thread::yield_now();
                                // Double-check queue
                                let still_empty = queue.lock().unwrap().is_empty();
                                if still_empty && completed.load(Ordering::Relaxed) >= faces.len() {
                                    break;
                                }
                                if still_empty {
                                    break; // No more work to do
                                }
                            }
                        }
                    }
                });
            }
        });

        // Collect results in original order, propagating any errors
        results
            .into_iter()
            .map(|r| {
                r.into_inner().unwrap().unwrap_or_else(|| {
                    Err(MesherError::MeshingFailed(
                        "face was not processed by any worker".into(),
                    ))
                })
            })
            .collect()
    }
}

/// Estimate the cost of meshing a face.
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
        MeshMode::Fine { .. } | MeshMode::Adaptive { .. }
    );
    let base = (total_points + hole_points) as f64;
    if is_fine {
        base * 10.0
    } else {
        base
    }
}

/// Mesh faces using the DAG scheduler.
pub fn mesh_faces_dag(inputs: Vec<FaceInput>) -> Result<Vec<FaceOutput>> {
    let dag = MeshDag::new(inputs);
    dag.execute()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::types::EdgeInput;

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
    fn dag_single_face() {
        let inputs = vec![make_square(1)];
        let outputs = mesh_faces_dag(inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].triangles.len(), 2);
    }

    #[test]
    fn dag_many_faces() {
        let inputs: Vec<FaceInput> = (1..=50).map(make_square).collect();
        let outputs = mesh_faces_dag(inputs).unwrap();
        assert_eq!(outputs.len(), 50);
        for (i, out) in outputs.iter().enumerate() {
            assert_eq!(out.face_id, (i + 1) as u64);
            assert_eq!(out.triangles.len(), 2);
        }
    }

    #[test]
    fn dag_deterministic() {
        let inputs: Vec<FaceInput> = (1..=20).map(make_square).collect();
        let out1 = mesh_faces_dag(inputs.clone()).unwrap();
        let out2 = mesh_faces_dag(inputs).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.face_id, b.face_id);
            assert_eq!(a.triangles, b.triangles);
        }
    }
}
