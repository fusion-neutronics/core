//! Point containment and volume location queries.

use crate::accel::bvh::Bvh;
use crate::mesh::topology::MeshTopology;
use crate::query::intersect::{self, dot, ray_triangle_intersect};
use crate::types::*;

/// Test if a point is inside a volume using ray casting.
///
/// Fires a ray from the point and counts surface crossings.
/// Odd = inside, even = outside. Tries multiple directions to handle
/// edge cases where the ray hits an edge/vertex.
pub fn point_in_volume(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    point: [f64; 3],
) -> bool {
    let directions = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.57735026919, 0.57735026919, 0.57735026919],
    ];

    for direction in &directions {
        match count_crossings_bvh(surface_bvh, tri_v0, tri_edge1, tri_edge2, point, *direction) {
            Some(count) => return count % 2 == 1,
            None => continue,
        }
    }

    false
}

/// Max inline hits for stack-allocated crossing buffer.
const MAX_INLINE_HITS: usize = 64;

fn count_crossings_bvh(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    origin: [f64; 3],
    direction: [f64; 3],
) -> Option<usize> {
    let mut hit_buf = [0.0f64; MAX_INLINE_HITS];
    let mut hit_count = 0usize;
    let mut ambiguous = false;

    // Collect-all traversal: parity needs EVERY hit along the ray, so the
    // nearest-hit pruning of ray_traverse must not be used here.
    surface_bvh.ray_traverse_collect(origin, direction, |bvh_prim_idx| {
        let idx = bvh_prim_idx as usize;
        let v0 = tri_v0[idx];
        let v1 = intersect::add(v0, tri_edge1[idx]);
        let v2 = intersect::add(v0, tri_edge2[idx]);

        if let Some(t) = ray_triangle_intersect(origin, direction, v0, v1, v2) {
            let normal = intersect::cross(tri_edge1[idx], tri_edge2[idx]);
            let d = dot(direction, normal);
            if d * d < 1e-10 * dot(normal, normal) {
                ambiguous = true;
                return;
            }
            if hit_count < MAX_INLINE_HITS {
                hit_buf[hit_count] = t;
                hit_count += 1;
            }
        }
    });

    if ambiguous {
        return None;
    }

    // Sort and deduplicate: hits at nearly the same distance are a single crossing
    let hits = &mut hit_buf[..hit_count];
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut crossings = 0usize;
    let mut prev_t = f64::NEG_INFINITY;
    for &t in hits.iter() {
        if (t - prev_t).abs() > 1e-8 {
            crossings += 1;
            prev_t = t;
        }
    }

    Some(crossings)
}

/// Find which volume contains a point by testing each volume.
///
/// Returns the volume ID, or the implicit complement if no volume matches.
pub(crate) fn find_volume(
    topo: &MeshTopology,
    surface_bvhs: &[crate::geometry::SurfaceBvhData],
    point: [f64; 3],
) -> VolumeId {
    let bb = &topo.global_aabb;
    if point[0] < bb[0]
        || point[0] > bb[3]
        || point[1] < bb[1]
        || point[1] > bb[4]
        || point[2] < bb[2]
        || point[2] > bb[5]
    {
        return topo.implicit_complement;
    }

    for vol_id in 0..topo.num_volumes {
        let vbb = &topo.volume_aabbs[vol_id as usize];
        if point[0] < vbb[0]
            || point[0] > vbb[3]
            || point[1] < vbb[1]
            || point[1] > vbb[4]
            || point[2] < vbb[2]
            || point[2] > vbb[5]
        {
            continue;
        }

        let sd = &surface_bvhs[vol_id as usize];
        if point_in_volume(
            &sd.bvh,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            point,
        ) {
            return vol_id;
        }
    }

    topo.implicit_complement
}
