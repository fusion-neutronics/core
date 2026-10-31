//! Core type aliases and enums for mesh geometry.

/// Index into the vertex array.
pub type VertexId = u32;
/// Index into the triangle array.
pub type TriangleId = u32;
/// Index into the tetrahedron array.
pub type TetrahedronId = u32;
/// Logical volume identifier (0-based).
pub type VolumeId = u32;
/// Logical surface identifier (0-based).
pub type SurfaceId = u32;

/// Orientation sense of a surface relative to a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sense {
    /// Surface normal points outward from the volume.
    Forward,
    /// Surface normal points inward to the volume.
    Reverse,
}

/// Boundary condition at a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryCondition {
    Transmission,
    Vacuum,
    Reflective,
}

/// Material and temperature properties for a volume.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialProps {
    pub material_name: Option<String>,
    pub temperature: Option<f64>,
}
