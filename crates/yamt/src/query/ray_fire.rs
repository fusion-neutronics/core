//! Ray-surface intersection query (DAGMC-style ray_fire).

use crate::accel::bvh::Bvh;
use crate::query::ray_history::RayHistory;
use crate::types::*;

/// Result of a ray fire query.
#[derive(Debug, Clone, Copy)]
pub struct RayFireResult {
    /// Distance along the ray to the hit surface.
    pub distance: f64,
    /// The surface that was hit.
    pub surface_id: SurfaceId,
    /// The specific triangle that was hit.
    pub triangle_id: TriangleId,
}

/// Minimum intersection distance for ray fire queries.
///
/// Small positive value to avoid self-intersection when a particle sits
/// exactly on a triangle face. The transport tolerance (1e-8) pushes
/// particles past the surface after crossing.
const RAY_FIRE_MIN_T: f64 = 1e-10;

/// Fire a ray within a volume and find the nearest surface crossing.
///
/// Uses precomputed triangle edge data for cache-friendly traversal.
/// Leaf triangles are tested in 4-wide SIMD batches.
#[allow(clippy::too_many_arguments)]
pub fn ray_fire(
    surface_bvh: &Bvh,
    surface_bvh_tri_ids: &[TriangleId],
    surface_bvh_surf_ids: &[SurfaceId],
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    origin: [f64; 3],
    direction: [f64; 3],
    history: Option<&RayHistory>,
) -> Option<RayFireResult> {
    // Use 4-wide batched ray-triangle testing for leaf nodes.
    // The accept filter handles history exclusion.
    let result = surface_bvh.ray_traverse_tris(
        origin,
        direction,
        tri_v0,
        tri_edge1,
        tri_edge2,
        RAY_FIRE_MIN_T,
        |bvh_prim_idx| {
            if let Some(hist) = history {
                !hist.contains(surface_bvh_tri_ids[bvh_prim_idx as usize])
            } else {
                true
            }
        },
    );

    result.map(|(prim_idx, distance)| {
        let idx = prim_idx as usize;
        RayFireResult {
            distance,
            surface_id: surface_bvh_surf_ids[idx],
            triangle_id: surface_bvh_tri_ids[idx],
        }
    })
}
