/// A boundary edge of a face, with its UV polyline.
#[derive(Clone, Debug)]
pub struct EdgeInput {
    /// Unique edge ID (shared edges have the same ID).
    pub edge_id: u64,
    /// UV-space polyline points for this edge.
    pub uv_points: Vec<[f64; 2]>,
    /// Whether this edge is reversed in the context of this face boundary.
    pub reversed: bool,
}

use crate::size_field::SizeField;

/// Mesh quality mode.
#[derive(Clone, Debug)]
pub enum MeshMode {
    /// Minimal triangles, no refinement. Good for transport-only surfaces.
    Coarse,
    /// Refine to target edge length in UV space.
    Fine {
        /// Target maximum edge length in UV space.
        target_edge_length: f64,
    },
    /// Curvature-adaptive refinement using a spatially-varying size field.
    /// Edges are measured in 3D via the metric tensor; target sizes vary
    /// across the UV domain based on surface curvature.
    Adaptive {
        /// The size field grid with target sizes and metric tensor data.
        size_field: Box<SizeField>,
    },
}

/// Input for meshing a single face.
#[derive(Clone, Debug)]
pub struct FaceInput {
    /// Unique face ID.
    pub face_id: u64,
    /// Boundary edges forming the outer wire (in order).
    pub boundary_edges: Vec<EdgeInput>,
    /// Hole wires (each is a list of edges forming a closed loop).
    pub holes: Vec<Vec<EdgeInput>>,
    /// Whether the face is planar.
    pub is_planar: bool,
    /// Mesh mode.
    pub mode: MeshMode,
}

/// Output from meshing a single face.
#[derive(Clone, Debug)]
pub struct FaceOutput {
    /// Face ID (same as input).
    pub face_id: u64,
    /// New interior UV vertices (not on boundary edges).
    /// These need UV→3D evaluation by the caller.
    pub interior_uv_vertices: Vec<[f64; 2]>,
    /// Triangle connectivity. Indices reference a combined vertex list:
    /// - Indices 0..N reference boundary vertices (from edges, in order)
    /// - Indices N..N+M reference interior_uv_vertices
    pub triangles: Vec<[usize; 3]>,
    /// Total number of boundary vertices (from all edges combined).
    pub num_boundary_vertices: usize,
    /// The boundary UV vertices the CDT actually used (indices
    /// `0..num_boundary_vertices` of the combined vertex list). Lets
    /// callers evaluate UV->3D without re-walking the input edges -
    /// required for scene-DAG faces whose inputs carry curve
    /// definitions instead of evaluated points.
    pub boundary_uv_vertices: Vec<[f64; 2]>,
}
