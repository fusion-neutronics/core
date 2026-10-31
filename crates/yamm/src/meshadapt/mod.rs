//! MeshAdapt-style surface mesher.
//!
//! This module implements a Rust-native surface mesher inspired by gmsh's
//! MeshAdapt algorithm (Shephard & Beall's Bidirectional Data Structure).
//!
//! # Phase 1: BDS half-edge mesh
//!
//! The [`bds`] module provides a half-edge data structure optimised for the
//! split / collapse / swap / smooth operations that will be added in later
//! phases.

pub mod bds;
pub mod chord_refine;
pub mod driver;
pub mod offsets;
pub mod ops;
