//! Geometry: CSG `Geometry`/`Cell`, the CSG↔mesh backend, mesh geometry,
//! CSG↔yamc-geo conversion, and the cell-neighbour acceleration lists.

pub mod backend;
pub mod cell;
pub mod conversion;
pub mod csg;
#[cfg(feature = "mesh")]
pub mod fill;
#[cfg(feature = "mesh")]
pub mod mesh;
pub mod neighbor_lists;

// Preserve the historical `crate::geometry::Geometry` path.
pub use csg::*;
