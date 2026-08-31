//! Geometry wrappers: CSG geometry/cells/regions/surfaces, meshes, and CAD.

mod bounding_box;
mod cell;
mod csg;
mod mesh;
mod region;
mod surface;

pub use bounding_box::*;
pub use cell::*;
pub use csg::*;
pub use mesh::*;
pub use region::*;
pub use surface::*;

#[cfg(feature = "mesh")]
mod mesh_geometry;
#[cfg(feature = "mesh")]
pub use mesh_geometry::*;

#[cfg(feature = "cad")]
mod cad;
#[cfg(feature = "cad")]
mod cad_scene;
#[cfg(feature = "cad")]
pub use cad::*;
#[cfg(feature = "cad")]
pub use cad_scene::*;
