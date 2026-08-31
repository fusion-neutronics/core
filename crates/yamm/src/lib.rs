//! Conformal surface and volume meshing for DAGMC neutronics geometry.
//!
//! This crate is the compute core: it has **no CAD-kernel and no Python
//! dependency** (only `spade`, `rayon`, geometry predicates, a kd-tree).
//! Geometry enters as parametric DEFINITIONS - [`curves::Curve3`],
//! [`curves::Curve2`], [`surfaces::Surface3`] - not as a CAD handle, so a
//! consumer with any geometry source can drive it without OpenCASCADE.
//!
//! Entry points, in increasing order of orchestration:
//! - [`mesher::mesh_face`] - triangulate one face from its boundary edges.
//! - [`mesher::mesh_faces_dag`] - schedule many pre-evaluated faces in
//!   parallel (cost-prioritized work queue).
//! - [`mesher::mesh_faces_scene`] - the true edge→face DAG: edges
//!   discretize their curves as roots, faces unblock via atomic counters
//!   the moment their boundary is ready, then evaluate their pcurves and
//!   triangulate. Fully CAD-kernel-free.
//! - [`volume`] - tetrahedral meshing of a closed boundary surface. Every tet
//!   it emits is positively oriented, checked at the emission boundary; see
//!   [`volume::VolumeOutput::first_non_positive_tet`] for why transport depends
//!   on that.
//!
//! Supporting modules: [`cdt`] (constrained Delaunay), [`meshadapt`]
//! (BRepMesh-style adaptation), [`size_field`] (curvature sizing),
//! [`utils`] (merge / remap / T-junction repair primitives shared with
//! the Python layer over PyO3).

pub mod cdt;
pub mod curves;
pub mod error;
pub mod mesh3d;
pub mod meshadapt;
pub mod mesher;
pub mod overlap;
pub mod size_field;
pub mod surfaces;
pub mod utils;
pub mod volume;

pub use error::{MesherError, Result};

/// Configure the number of threads used for parallel meshing.
///
/// Must be called before any parallel meshing function. Panics if the
/// global thread pool has already been initialised (e.g., by a prior call).
pub fn set_thread_count(n: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .expect("rayon global thread pool already initialised");
}
