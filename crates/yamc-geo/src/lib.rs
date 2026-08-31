//! Shared geometry layer: cells, the geometry container, BVH acceleration,
//! bounding boxes, mesh hooks, and slice plotting.
pub mod bounding_box;
pub mod bvh;
pub mod cell;
pub mod geometry;
pub mod mesh;
pub mod plot;
pub mod region;
pub mod surface;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use bounding_box::BoundingBox;
pub use bvh::{Bvh, BvhNode};
pub use cell::GeoCell;
pub use geometry::CsgGeometry;
pub use mesh::{GeoMesh, GeoMeshVolume};
pub use plot::{PlotGrid, PlotParams, PlotSample};
pub use region::{FlatRegion, HalfspaceType, Region, RegionExpr};
pub use surface::{BoundaryType, Halfspace, Surface, SurfaceKind};
