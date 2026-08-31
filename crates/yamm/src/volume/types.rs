/// Input for volume (tetrahedral) meshing.
#[derive(Clone, Debug)]
pub struct VolumeInput {
    /// Boundary vertices in 3D (the surface mesh vertices).
    pub boundary_vertices: Vec<[f64; 3]>,
    /// Boundary triangles as indices into `boundary_vertices`.
    /// Must form a closed, manifold surface.
    pub boundary_triangles: Vec<[usize; 3]>,
    /// Target edge length for interior tetrahedra.
    pub target_edge_length: f64,
}

/// Statistics from the boundary recovery phase of volume meshing.
#[derive(Clone, Debug, Default)]
pub struct BoundaryRecoveryStats {
    /// Number of boundary edges that were successfully recovered via flips.
    pub edges_recovered: usize,
    /// Number of boundary faces that were successfully recovered via flips.
    pub faces_recovered: usize,
    /// Number of boundary edges that could not be recovered.
    pub edges_failed: usize,
    /// Number of boundary faces that could not be recovered.
    pub faces_failed: usize,
    /// Specific edges that could not be recovered (vertex index pairs).
    pub failed_edges: Vec<[usize; 2]>,
    /// Specific faces that could not be recovered (vertex index triples).
    pub failed_faces: Vec<[usize; 3]>,
    /// Tets of the FINAL mesh whose interior the boundary passes through.
    ///
    /// This is an exact predictor of an inexact fill, and the only honest
    /// watertightness signal this struct carries. A tet the boundary cuts is
    /// kept or dropped WHOLESALE by the inside/outside filter, so its volume is
    /// counted entirely or not at all; zero cut tets means the tet mesh
    /// partitions exactly the region its boundary encloses. Measured over ten
    /// zoo solids, `tets_cut_by_boundary == 0` iff the fill error is at machine
    /// precision (issue #136).
    ///
    /// Note the contrast with `faces_failed`, which counts boundary faces not
    /// present as tet faces and fires on healthy meshes - a plain cylinder
    /// reports hundreds while filling exactly, because mesh improvement
    /// subdivides boundary faces (issue #105).
    pub tets_cut_by_boundary: usize,
}

/// Output from volume meshing.
#[derive(Clone, Debug)]
pub struct VolumeOutput {
    /// Interior vertices (not on the boundary surface).
    /// The full vertex list is: boundary_vertices ++ interior_vertices.
    /// Indices 0..N reference boundary vertices, N.. reference these.
    pub interior_vertices: Vec<[f64; 3]>,
    /// Tetrahedra as 4-tuples of vertex indices into the combined list.
    ///
    /// INVARIANT: every tet is POSITIVELY ORIENTED, i.e. its signed volume
    /// `det([v1-v0, v2-v0, v3-v0]) / 6` is strictly positive. See
    /// [`VolumeOutput::first_non_positive_tet`] for why this matters and
    /// [`crate::volume::mesh_volume`] for where it is enforced.
    pub tetrahedra: Vec<[usize; 4]>,
    /// Statistics from the boundary recovery phase.
    pub boundary_recovery_stats: BoundaryRecoveryStats,
}

impl VolumeOutput {
    /// The full vertex list this output's tet indices address:
    /// `boundary_vertices ++ interior_vertices`.
    ///
    /// Allocates. Prefer [`VolumeOutput::vertex`] on hot paths.
    pub fn all_vertices(&self, boundary_vertices: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut all = boundary_vertices.to_vec();
        all.extend_from_slice(&self.interior_vertices);
        all
    }

    /// One vertex of the combined `boundary_vertices ++ interior_vertices`
    /// list, without materialising it.
    #[inline]
    pub fn vertex(&self, boundary_vertices: &[[f64; 3]], index: usize) -> [f64; 3] {
        match boundary_vertices.get(index) {
            Some(v) => *v,
            None => self.interior_vertices[index - boundary_vertices.len()],
        }
    }

    /// Index and signed volume of the first tet that is not positively
    /// oriented, or `None` when the whole mesh satisfies the invariant.
    ///
    /// WHY THE INVARIANT EXISTS. Downstream transport (the `yamt` crate) reads
    /// tet face normals straight off a fixed face table,
    /// `yamt::mesh::topology::TET_FACE_VERTICES`, whose orderings yield
    /// OUTWARD-pointing normals only for a positively oriented tet; for a
    /// negatively oriented one every normal points inward instead. The element
    /// walk picks its exit face with `dot(direction, normal) > 0`, so a
    /// negatively oriented tet makes it select an ENTRY face: the walk hops
    /// backwards or stops at the mesh boundary early, and unstructured
    /// track-length tallies read far too low (issue #316, which measured -33%).
    /// `yamt` refuses to load a mesh with an inverted tet, but a mesher that
    /// quietly starts emitting them should fail HERE, where the regression is,
    /// not later at load time or as a mysterious flux deficit.
    ///
    /// The sign test is Shewchuk's exact `orient3d`, the same predicate the
    /// mesher's own orientation passes use, so this can only fire on a tet no
    /// pass fixed. Cost is O(number of tets) with no allocation.
    pub fn first_non_positive_tet(&self, boundary_vertices: &[[f64; 3]]) -> Option<(usize, f64)> {
        for (i, t) in self.tetrahedra.iter().enumerate() {
            let signed = super::predicates3d::tet_volume(
                self.vertex(boundary_vertices, t[0]),
                self.vertex(boundary_vertices, t[1]),
                self.vertex(boundary_vertices, t[2]),
                self.vertex(boundary_vertices, t[3]),
            );
            if signed <= 0.0 {
                return Some((i, signed));
            }
        }
        None
    }
}
