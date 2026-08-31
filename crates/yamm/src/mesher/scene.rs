//! True edge→face DAG scheduler over an OCC-free scene description.
//!
//! Fulfils the long-standing `dag.rs` TODO ("move edge discretization to
//! Rust and track true edge→face dependencies with atomic counters").
//! The scene carries curve DEFINITIONS, not evaluated points:
//!
//! - **Edge tasks** (DAG roots): each unique edge either discretizes its
//!   [`Curve3`] (chordal + angular bounds) or accepts precomputed
//!   parameters (the OCC-fallback path for exotic curve types).
//! - **Face tasks**: an atomic counter holds the number of unresolved
//!   edges a face references; the edge task that drops it to zero
//!   spawns the face, which evaluates its pcurves ([`Curve2`]) at the
//!   shared parameters and runs the CDT. No barrier exists between edge
//!   and face work - faces mesh while other edges still discretize.
//!
//! Because every face referencing an edge evaluates the SAME parameter
//! list through its own pcurve, shared-edge discretizations agree by
//! construction (the conformality invariant the Python
//! `edge_params_cache` provides today).
//!
//! This is also a complete CAD-kernel-free meshing API: a consumer with
//! its own geometry source (no OCC) can drive the mesher entirely
//! through curve definitions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::curves::{discretize_curve3, Curve2, Curve3};
use crate::error::{MesherError, Result};

use super::face::mesh_face;
use super::types::{EdgeInput, FaceInput, FaceOutput, MeshMode};

/// How an edge's discretization parameters are obtained.
pub enum SceneEdgeSource {
    /// Discretize this curve over `[t0, t1]` (an edge task).
    Curve { curve: Curve3, t0: f64, t1: f64 },
    /// Parameters already known (OCC fallback / external cache).
    Params(Vec<f64>),
}

/// One unique edge in the scene.
pub struct SceneEdge {
    pub source: SceneEdgeSource,
}

/// A face's reference to an edge within one of its wires.
pub struct SceneFaceEdge {
    /// Index into the scene's edge table, or `None` when `uv_points`
    /// is pre-resolved (exotic pcurve, periodic synthetic edge, ...).
    pub edge: Option<usize>,
    pub edge_id: u64,
    pub reversed: bool,
    /// pcurve to evaluate at the edge's parameters (`edge` must be
    /// `Some`); ignored when `uv_points` is provided.
    pub pcurve: Option<Curve2>,
    /// Pre-resolved UV polyline (used when `edge` is `None` or
    /// `pcurve` is `None`).
    pub uv_points: Vec<[f64; 2]>,
}

/// One face in the scene.
pub struct SceneFace {
    pub face_id: u64,
    pub boundary: Vec<SceneFaceEdge>,
    pub holes: Vec<Vec<SceneFaceEdge>>,
    pub is_planar: bool,
    pub mode: MeshMode,
    /// Junction-snap flags (issue #70): `boundary_snaps[k]` means edge
    /// `k`'s effective head must be snapped to edge `k-1`'s effective
    /// tail after UV evaluation - consecutive edges share an OCC
    /// vertex, but each pcurve evaluates it with its own slop, and the
    /// leftover micro-jog self-intersects under exact predicates.
    /// Empty = no snaps.
    pub boundary_snaps: Vec<bool>,
    pub hole_snaps: Vec<Vec<bool>>,
}

/// Snap flagged junctions: edge `k`'s effective head := edge `k-1`'s
/// effective tail (cyclic). Mirrors the Python `apply_junction_snaps`.
fn apply_junction_snaps(edges: &mut [EdgeInput], snaps: &[bool]) {
    let n = edges.len();
    if n < 2 || snaps.len() != n {
        return;
    }
    let chord = |pts: &[[f64; 2]]| -> f64 {
        let a = pts[0];
        let b = pts[pts.len() - 1];
        ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    };
    for k in 0..n {
        if !snaps[k] {
            continue;
        }
        let prev = (k + n - 1) % n;
        let tail = if edges[prev].reversed {
            edges[prev].uv_points[0]
        } else {
            *edges[prev].uv_points.last().unwrap()
        };
        let prev_span = chord(&edges[prev].uv_points);
        let e = &mut edges[k];
        let head_idx = if e.reversed { e.uv_points.len() - 1 } else { 0 };
        let head = e.uv_points[head_idx];
        let gap = ((head[0] - tail[0]).powi(2) + (head[1] - tail[1]).powi(2)).sqrt();
        if gap == 0.0 {
            continue;
        }
        // Relative gate (see the Python twin): true shared-vertex jogs
        // are vertex-tolerance-sized; gaps comparable to the edge
        // length are REAL dirty-wire jumps that must not be pinched.
        let span = prev_span.max(chord(&e.uv_points));
        if gap > 1e-3 * span {
            continue;
        }
        e.uv_points[head_idx] = tail;
    }
}

fn resolve_edge(fe: &SceneFaceEdge, params: &[OnceLock<Vec<f64>>]) -> EdgeInput {
    let uv_points = match (fe.edge, &fe.pcurve) {
        (Some(ei), Some(pc)) => {
            let ts = params[ei].get().expect("edge task completed");
            ts.iter().map(|&t| pc.value(t)).collect()
        }
        _ => fe.uv_points.clone(),
    };
    EdgeInput {
        edge_id: fe.edge_id,
        uv_points,
        reversed: fe.reversed,
    }
}

/// Execute the scene DAG. Returns face outputs in input order.
pub fn mesh_faces_scene(
    edges: Vec<SceneEdge>,
    faces: Vec<SceneFace>,
    tolerance: f64,
    angular_tolerance: f64,
) -> Result<Vec<FaceOutput>> {
    let n_edges = edges.len();
    let n_faces = faces.len();
    if n_faces == 0 {
        return Ok(vec![]);
    }

    // Edge parameter slots, filled exactly once by edge tasks.
    let params: Vec<OnceLock<Vec<f64>>> = (0..n_edges).map(|_| OnceLock::new()).collect();

    // edge index -> dependent face indices (deduplicated per face).
    let mut edge_faces: Vec<Vec<usize>> = vec![Vec::new(); n_edges];
    let mut counters: Vec<AtomicUsize> = Vec::with_capacity(n_faces);
    for (fi, face) in faces.iter().enumerate() {
        // Only Curve-source edges have tasks that decrement counters;
        // Params edges are pre-resolved and must not count as pending
        // dependencies (a face with only Params deps would never spawn).
        let mut deps: Vec<usize> = face
            .boundary
            .iter()
            .chain(face.holes.iter().flatten())
            .filter_map(|fe| match (fe.edge, &fe.pcurve) {
                (Some(ei), Some(_)) => {
                    if matches!(edges[ei].source, SceneEdgeSource::Curve { .. }) {
                        Some(ei)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        deps.sort_unstable();
        deps.dedup();
        for &ei in &deps {
            edge_faces[ei].push(fi);
        }
        counters.push(AtomicUsize::new(deps.len()));
    }

    let results: Vec<Mutex<Option<Result<FaceOutput>>>> =
        (0..n_faces).map(|_| Mutex::new(None)).collect();

    // Resolve immediate (Params) edges before entering the scope so
    // their dependent faces start as ready.
    for (ei, e) in edges.iter().enumerate() {
        if let SceneEdgeSource::Params(p) = &e.source {
            let _ = params[ei].set(p.clone());
        }
    }

    rayon::scope(|s| {
        let params = &params;
        let edge_faces = &edge_faces;
        let counters = &counters;
        let results = &results;
        let faces = &faces;

        // Closure to run a face task (used by both the ready-at-start
        // pass and edge-completion unblocking).
        fn run_face<'s>(
            s: &rayon::Scope<'s>,
            fi: usize,
            faces: &'s [SceneFace],
            params: &'s [OnceLock<Vec<f64>>],
            results: &'s [Mutex<Option<Result<FaceOutput>>>],
        ) {
            s.spawn(move |_| {
                let face = &faces[fi];
                let mut boundary_edges: Vec<EdgeInput> = face
                    .boundary
                    .iter()
                    .map(|fe| resolve_edge(fe, params))
                    .collect();
                apply_junction_snaps(&mut boundary_edges, &face.boundary_snaps);
                let mut holes: Vec<Vec<EdgeInput>> = face
                    .holes
                    .iter()
                    .map(|h| h.iter().map(|fe| resolve_edge(fe, params)).collect())
                    .collect();
                for (h, hs) in holes.iter_mut().zip(face.hole_snaps.iter()) {
                    apply_junction_snaps(h, hs);
                }
                let input = FaceInput {
                    face_id: face.face_id,
                    boundary_edges,
                    holes,
                    is_planar: face.is_planar,
                    mode: face.mode.clone(),
                };
                *results[fi].lock().unwrap() = Some(mesh_face(&input));
            });
        }

        // Faces with zero pending deps (all edges pre-resolved) start
        // immediately. This pass runs BEFORE any edge task spawns:
        // ready faces have no Curve dependencies by definition, so no
        // edge task can race a decrement against it, and faces with
        // deps > 0 are spawned exactly once by the edge task whose
        // fetch_sub returns 1.
        for (fi, counter) in counters.iter().enumerate() {
            if counter.load(Ordering::Acquire) == 0 {
                run_face(s, fi, faces, params, results);
            }
        }

        // Spawn an edge task per Curve edge; each completion decrements
        // its dependent faces' counters and spawns those that hit zero.
        for (ei, e) in edges.iter().enumerate() {
            if let SceneEdgeSource::Curve { curve, t0, t1 } = &e.source {
                s.spawn(move |s| {
                    let p = discretize_curve3(curve, *t0, *t1, tolerance, angular_tolerance);
                    let _ = params[ei].set(p);
                    for &fi in &edge_faces[ei] {
                        if counters[fi].fetch_sub(1, Ordering::AcqRel) == 1 {
                            run_face(s, fi, faces, params, results);
                        }
                    }
                });
            }
        }
    });

    results
        .into_iter()
        .map(|r| {
            r.into_inner().unwrap().unwrap_or_else(|| {
                Err(MesherError::MeshingFailed(
                    "scene face was not processed".into(),
                ))
            })
        })
        .collect()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square_face(face_id: u64, edge_base: usize, with_pcurves: bool) -> SceneFace {
        // Four straight pcurves around the unit square; the 3-D edges
        // are lines so each contributes exactly its endpoints.
        let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let boundary = (0..4)
            .map(|k| {
                let a = corners[k];
                let b = corners[(k + 1) % 4];
                if with_pcurves {
                    SceneFaceEdge {
                        edge: Some(edge_base + k),
                        edge_id: k as u64,
                        reversed: false,
                        pcurve: Some(Curve2::Line {
                            origin: a,
                            dir: [b[0] - a[0], b[1] - a[1]],
                        }),
                        uv_points: vec![],
                    }
                } else {
                    SceneFaceEdge {
                        edge: None,
                        edge_id: k as u64,
                        reversed: false,
                        pcurve: None,
                        uv_points: vec![a, b],
                    }
                }
            })
            .collect();
        SceneFace {
            face_id,
            boundary,
            holes: vec![],
            is_planar: true,
            mode: MeshMode::Coarse,
            boundary_snaps: vec![],
            hole_snaps: vec![],
        }
    }

    fn square_edges() -> Vec<SceneEdge> {
        // 3-D unit-square edges in the z=0 plane, parameterized [0, 1].
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        (0..4)
            .map(|k| {
                let a = corners[k];
                let b = corners[(k + 1) % 4];
                SceneEdge {
                    source: SceneEdgeSource::Curve {
                        curve: Curve3::Line {
                            origin: a,
                            dir: [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                        },
                        t0: 0.0,
                        t1: 1.0,
                    },
                }
            })
            .collect()
    }

    #[test]
    fn scene_dag_meshes_square_via_edge_tasks() {
        let outputs = mesh_faces_scene(
            square_edges(),
            vec![unit_square_face(1, 0, true)],
            0.01,
            0.3,
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].triangles.len(), 2);
        assert_eq!(outputs[0].num_boundary_vertices, 4);
    }

    #[test]
    fn scene_dag_params_only_edges_spawn_faces() {
        // Regression: a face whose edges are ALL Params-source (already
        // discretized) has no edge task to decrement its counter - it
        // must start as ready, not hang as "not processed".
        let edges: Vec<SceneEdge> = (0..4)
            .map(|_| SceneEdge {
                source: SceneEdgeSource::Params(vec![0.0, 1.0]),
            })
            .collect();
        let outputs =
            mesh_faces_scene(edges, vec![unit_square_face(3, 0, true)], 0.01, 0.3).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].triangles.len(), 2);
    }

    #[test]
    fn scene_dag_pre_resolved_faces_start_immediately() {
        let outputs =
            mesh_faces_scene(vec![], vec![unit_square_face(7, 0, false)], 0.01, 0.3).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].triangles.len(), 2);
    }

    #[test]
    fn scene_dag_shared_edges_give_identical_params() {
        // Two faces sharing all four edges: both must see the same
        // parameter lists (conformality by construction).
        let edges = square_edges();
        let faces = vec![unit_square_face(1, 0, true), unit_square_face(2, 0, true)];
        let outputs = mesh_faces_scene(edges, faces, 0.01, 0.3).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(
            outputs[0].num_boundary_vertices,
            outputs[1].num_boundary_vertices
        );
        assert_eq!(outputs[0].triangles.len(), outputs[1].triangles.len());
    }

    #[test]
    fn scene_dag_many_faces_deterministic_order() {
        let edges = square_edges();
        let faces: Vec<SceneFace> = (0..40).map(|i| unit_square_face(i, 0, true)).collect();
        let outputs = mesh_faces_scene(edges, faces, 0.01, 0.3).unwrap();
        assert_eq!(outputs.len(), 40);
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(o.face_id, i as u64);
            assert_eq!(o.triangles.len(), 2);
        }
    }
}
