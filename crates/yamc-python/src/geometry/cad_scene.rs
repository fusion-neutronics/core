//! Bindings over the `yamm` meshing core: the scene surface mesher (replacing
//! OCC `BRepMesh`) and the Rust tet mesher (replacing netgen).
//!
//! yamc owns its OCP extraction in Python. It dedups edges and
//! discretizes them once (in OCP), then evaluates each face's pcurve at the
//! shared parameters to get UV polylines. So the Rust side only needs the
//! *pre-resolved* scene path - no curve serialization, no size-field plumbing:
//! every boundary edge arrives as `(edge_id, reversed, uv_points)` and the
//! edge table is empty. Conformality across shared edges is achieved upstream
//! by feeding both faces the identical UV samples for the shared edge.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use yamm::mesher::{mesh_faces_scene, MeshMode, SceneFace, SceneFaceEdge};
use yamm::volume::{mesh_volume, VolumeInput};

/// Output from meshing one face in UV space (new scene mesher).
///
/// Triangle indices `0..num_boundary_vertices` reference `boundary_uv_vertices`;
/// indices `>= num_boundary_vertices` reference `interior_uv_vertices`. The
/// caller evaluates UV -> 3D on the OCC surface itself.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core")]
pub struct SceneFaceOutput {
    #[pyo3(get)]
    pub face_id: u64,
    #[pyo3(get)]
    pub boundary_uv_vertices: Vec<[f64; 2]>,
    #[pyo3(get)]
    pub interior_uv_vertices: Vec<[f64; 2]>,
    #[pyo3(get)]
    pub triangles: Vec<[usize; 3]>,
    #[pyo3(get)]
    pub num_boundary_vertices: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl SceneFaceOutput {
    fn __repr__(&self) -> String {
        format!(
            "SceneFaceOutput(face_id={}, boundary={}, interior={}, triangles={})",
            self.face_id,
            self.boundary_uv_vertices.len(),
            self.interior_uv_vertices.len(),
            self.triangles.len()
        )
    }
}

/// One pre-resolved boundary edge: `(edge_id, reversed, uv_points)`.
type PreEdge = (u64, bool, Vec<[f64; 2]>);

/// One face: `(face_id, boundary, holes, is_planar, mode)`.
/// `mode` is `"coarse"` or `"fine:<edge_length>"`.
type PreFace = (u64, Vec<PreEdge>, Vec<Vec<PreEdge>>, bool, String);

fn pre_edge(e: &PreEdge) -> SceneFaceEdge {
    SceneFaceEdge {
        edge: None,
        edge_id: e.0,
        reversed: e.1,
        pcurve: None,
        uv_points: e.2.clone(),
    }
}

fn parse_mode(mode: &str) -> PyResult<MeshMode> {
    if mode == "coarse" {
        Ok(MeshMode::Coarse)
    } else if let Some(s) = mode.strip_prefix("fine:") {
        let target_edge_length: f64 = s
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid fine length: {e}")))?;
        Ok(MeshMode::Fine { target_edge_length })
    } else {
        Err(PyValueError::new_err(format!(
            "invalid mesh mode '{mode}'; use 'coarse' or 'fine:<length>'"
        )))
    }
}

/// Mesh faces from pre-resolved UV boundary loops (empty edge table; yamc owns
/// edge discretization in OCP). Returns one output per input face, in order.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (faces, tolerance=0.01, angular_tolerance=0.3))]
pub fn mesh_faces_scene_resolved(
    py: Python<'_>,
    faces: Vec<PreFace>,
    tolerance: f64,
    angular_tolerance: f64,
) -> PyResult<Vec<SceneFaceOutput>> {
    let scene_faces: Vec<SceneFace> = faces
        .iter()
        .map(|(face_id, boundary, holes, is_planar, mode)| {
            Ok(SceneFace {
                face_id: *face_id,
                boundary: boundary.iter().map(pre_edge).collect(),
                holes: holes
                    .iter()
                    .map(|h| h.iter().map(pre_edge).collect())
                    .collect(),
                is_planar: *is_planar,
                mode: parse_mode(mode)?,
                boundary_snaps: vec![],
                hole_snaps: vec![],
            })
        })
        .collect::<PyResult<_>>()?;

    let outputs = py
        .detach(|| mesh_faces_scene(vec![], scene_faces, tolerance, angular_tolerance))
        .map_err(|e| PyRuntimeError::new_err(format!("mesh_faces_scene failed: {e}")))?;

    Ok(outputs
        .into_iter()
        .map(|o| SceneFaceOutput {
            face_id: o.face_id,
            boundary_uv_vertices: o.boundary_uv_vertices,
            interior_uv_vertices: o.interior_uv_vertices,
            triangles: o.triangles,
            num_boundary_vertices: o.num_boundary_vertices,
        })
        .collect())
}

// The "pre-oriented to positive volume" claim below is an INVARIANT yamm
// enforces and re-checks at its own emission boundary (see
// `yamm::volume::VolumeOutput::first_non_positive_tet`), not an incidental
// property: transport reads tet face normals off a fixed face table that only
// points outward for positively oriented tets, and an inverted tet makes the
// element walk pick an entry face as its exit, so unstructured track-length
// tallies read far too low (issue #316). A violation surfaces here as a
// RuntimeError out of `mesh_volume`. Connectivity then reaches the Arrow file
// unchanged apart from a constant per-solid vertex offset
// (`yamt::mesh::cad_build::flatten_tet_blocks`), so the invariant survives
// export.
/// Tetrahedralize a solid from its conformal boundary surface mesh (replaces
/// netgen). Returns `(interior_vertices, tetrahedra)`. Boundary vertices are
/// preserved exactly (tet indices `0..len(boundary_vertices)` reference them,
/// `>=` reference the returned interior vertices), and tets are pre-oriented
/// to positive volume - so no winding swap or face remapping is needed.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn mesh_volume_rs(
    py: Python<'_>,
    boundary_vertices: Vec<[f64; 3]>,
    boundary_triangles: Vec<[usize; 3]>,
    target_edge_length: f64,
) -> PyResult<(Vec<[f64; 3]>, Vec<[usize; 4]>)> {
    let input = VolumeInput {
        boundary_vertices,
        boundary_triangles,
        target_edge_length,
    };
    let out = py
        .detach(|| mesh_volume(&input))
        .map_err(|e| PyRuntimeError::new_err(format!("mesh_volume failed: {e}")))?;
    Ok((out.interior_vertices, out.tetrahedra))
}
