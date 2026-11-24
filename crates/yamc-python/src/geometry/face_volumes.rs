use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use yamt::mesh::face_volumes::FaceVolumeMesh;

/// A welded surface mesh being corrected so each curved CAD face encloses its
/// CAD volume (the compiled core of :mod:`yamc.cad._face_volumes`).
///
/// It holds the mesh and its adjacency and does the volume sums, the root
/// finding and the vertex moves; the caller evaluates the CAD (curve points
/// and normals, a face's exact volume) and hands it in.
///
/// Args:
///     vertices: List of ``[x, y, z]`` coordinates.
///     triangles: List of ``[i, j, k]`` vertex indices.
///     face_ids: The CAD face of each triangle.
///
/// Raises:
///     ValueError: If ``face_ids`` is not one per triangle or a triangle
///         indexes past the vertices.
#[gen_stub_pyclass]
#[pyclass(module = "yamc._core", name = "FaceVolumeMesh")]
pub struct PyFaceVolumeMesh {
    inner: FaceVolumeMesh,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFaceVolumeMesh {
    #[new]
    fn new(
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
        face_ids: Vec<u32>,
    ) -> PyResult<Self> {
        let inner =
            FaceVolumeMesh::new(vertices, &triangles, &face_ids).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// The face ids, in order of first appearance among the triangles.
    fn faces(&self) -> Vec<u32> {
        self.inner.faces().to_vec()
    }

    /// The vertices, as moved so far.
    fn vertices(&self) -> Vec<[f64; 3]> {
        self.inner.vertices().to_vec()
    }

    /// ``faces`` and every face sharing a mesh edge of their boundaries,
    /// sorted.
    fn neighbours(&self, faces: Vec<u32>) -> Vec<u32> {
        self.inner.neighbours(&faces)
    }

    /// ``[(vertex, [x, y, z]), ...]``: the vertices on the boundaries of
    /// ``faces`` inside the box from ``lo`` to ``hi`` (bounds included), by
    /// vertex index.
    fn boundary_points(&self, faces: Vec<u32>, lo: [f64; 3], hi: [f64; 3]) -> Vec<(u32, [f64; 3])> {
        self.inner.boundary_points(&faces, lo, hi)
    }

    /// Whether every two consecutive vertices of ``chain`` share a mesh edge.
    fn is_chain(&self, chain: Vec<u32>) -> bool {
        self.inner.is_chain(&chain)
    }

    /// Record a matched CAD edge: its mesh vertices ``chain`` in curve order
    /// and ``arcs``, per segment, points of the edge from the segment's first
    /// vertex to its second, ends included. Returns the edge's index.
    ///
    /// Raises:
    ///     ValueError: If there is not one arc of at least two points per
    ///         segment, or a vertex is not in the mesh.
    fn add_edge(&mut self, chain: Vec<u32>, arcs: Vec<Vec<[f64; 3]>>) -> PyResult<usize> {
        self.inner
            .add_edge(chain, arcs)
            .map_err(PyValueError::new_err)
    }

    /// Move the vertices inside edge ``edge`` along ``outward`` (per interior
    /// vertex of its chain, the unit vector away from the centre of
    /// curvature, or None), all by one distance of at most ``cap``, so the
    /// polyline cuts off the arc's area. Only vertices on no face but
    /// ``faces`` move. Returns whether the move hit the cap, or None when
    /// nothing moved.
    ///
    /// Raises:
    ///     ValueError: If there is no such edge or ``outward`` is not one per
    ///         interior vertex.
    fn move_edge(
        &mut self,
        edge: usize,
        faces: Vec<u32>,
        outward: Vec<Option<[f64; 3]>>,
        cap: f64,
    ) -> PyResult<Option<bool>> {
        self.inner
            .move_edge(edge, &faces, &outward, cap)
            .map_err(PyValueError::new_err)
    }

    /// Start correcting face ``face_id``: the point to take the CAD face's
    /// volume from, or None (with the reason counted) when the face has no
    /// vertex it may move or its boundary is not matched to added edges.
    ///
    /// Raises:
    ///     ValueError: If the mesh has no such face.
    fn face_apex(&mut self, face_id: u32) -> PyResult<Option<[f64; 3]>> {
        self.inner.face_apex(face_id).map_err(PyValueError::new_err)
    }

    /// Finish the face ``face_apex`` started: move its own vertices along
    /// their normals, all by one distance of at most ``cap``, so its
    /// triangles enclose ``cad_volume`` (the CAD face's volume from the apex)
    /// once closed to the CAD edges. Returns whether the move hit the cap.
    ///
    /// Raises:
    ///     ValueError: If no face was started since the last move.
    fn move_face(&mut self, cad_volume: f64, cap: f64) -> PyResult<bool> {
        self.inner
            .move_face(cad_volume, cap)
            .map_err(PyValueError::new_err)
    }

    /// Counts of what the moves did: ``[(name, count), ...]``.
    fn stats(&self) -> Vec<(String, usize)> {
        let s = self.inner.stats();
        [
            ("edges_moved", s.edges_moved),
            ("edges_capped", s.edges_capped),
            ("faces_moved", s.faces_moved),
            ("faces_capped", s.faces_capped),
            ("faces_no_interior", s.faces_no_interior),
            ("faces_unmatched", s.faces_unmatched),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
}
