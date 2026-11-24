/// Error types for the cad-to-dagmc-mesher library.
use std::fmt;

/// Errors that can occur during meshing operations.
#[derive(Debug, Clone)]
pub enum MesherError {
    /// Invalid input parameters (e.g., empty arrays, bad indices).
    InvalidInput(String),
    /// Degenerate geometry detected (e.g., zero-area triangles, coplanar points).
    DegenerateGeometry(String),
    /// Meshing algorithm failed to produce a valid result.
    MeshingFailed(String),
}

impl fmt::Display for MesherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MesherError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            MesherError::DegenerateGeometry(msg) => write!(f, "degenerate geometry: {msg}"),
            MesherError::MeshingFailed(msg) => write!(f, "meshing failed: {msg}"),
        }
    }
}

impl std::error::Error for MesherError {}

/// A specialized Result type for meshing operations.
pub type Result<T> = std::result::Result<T, MesherError>;
