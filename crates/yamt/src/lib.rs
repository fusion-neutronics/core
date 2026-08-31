//! Pure-Rust geometry for mesh-based particle transport: surface / tet
//! meshes, BVH acceleration, spatial queries, and Arrow I/O.
pub mod accel;
pub mod geometry;
pub mod io;
pub mod mesh;
pub mod query;
pub mod types;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use geometry::MeshGeometry;
pub use mesh::physical_groups::PhysicalGroupData;
pub use mesh::topology::MeshTopology;
pub use query::ray_history::RayHistory;
pub use types::*;

#[cfg(feature = "arrow")]
pub use io::arrow::{
    add_vacuum_boundary, build_topology, read_arrow_mesh, write_arrow_mesh, ArrowMeshData, MeshData,
};

/// Report the SIMD ISA level used for intersection kernels.
#[cfg(feature = "simd")]
pub fn simd_level() -> &'static str {
    accel::simd::simd_level()
}

/// SIMD capability summary for diagnostics.
#[cfg(feature = "simd")]
pub use accel::simd::SimdInfo;

/// Return a full SIMD diagnostics snapshot.
#[cfg(feature = "simd")]
pub fn simd_info() -> SimdInfo {
    accel::simd::simd_info()
}

/// Print a user-friendly SIMD diagnostics summary to stderr.
#[cfg(feature = "simd")]
pub fn print_simd_info() {
    accel::simd::print_simd_info()
}
