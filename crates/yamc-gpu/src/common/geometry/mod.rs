//! Shared `#[cube]` geometry helpers used by both transport kernels:
//! BVH and linear cell finding, surface distance, and 3D direction
//! rotation.

pub mod birth;
pub mod boundary_distance;
pub mod bvh_cell_finding;
pub mod cell_finding;
pub mod cell_surfaces;
pub mod direction_rotation;
pub mod region_eval;
pub mod surface_distance;
