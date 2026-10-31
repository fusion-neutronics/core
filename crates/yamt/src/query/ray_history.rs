//! Ray history tracking for avoiding re-intersection when streaming.
//!
//! Three states (mirrors DAGMC's `RayHistory`):
//! - New particle: empty history
//! - Streaming (same direction): reuse history, skip last-hit triangle
//! - Scattered (direction changed): reset history

use crate::types::TriangleId;

/// Tracks recently hit triangles to avoid re-intersection.
#[derive(Debug, Clone, Default)]
pub struct RayHistory {
    /// Recently hit triangles to exclude from intersection tests.
    pub exclude: Vec<TriangleId>,
}

impl RayHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a triangle hit. Call after a successful `ray_fire`.
    pub fn add(&mut self, tri: TriangleId) {
        self.exclude.push(tri);
    }

    /// Reset the history (e.g., after scattering).
    pub fn reset(&mut self) {
        self.exclude.clear();
    }

    /// Check if a triangle is in the exclusion set.
    #[inline]
    pub fn contains(&self, tri: TriangleId) -> bool {
        self.exclude.contains(&tri)
    }
}
