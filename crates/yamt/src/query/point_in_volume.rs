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

/// Like [`point_in_volume`], but `None` when the mesh does not say.
///
/// Parity alone takes the first direction whose ray crosses no facet edge-on,
/// and answers outside when every direction does. Here each direction's
/// parity is also checked against the facet nearest the point along it: from
/// inside a closed, outward-wound volume the ray leaves through that facet,
/// from outside it enters (`exit_sign` gives each facet's winding relative to
/// the volume, `+1` outward and `-1` inward). Parity and facing disagree
/// near a folded or flipped facet, or where the ray slips through a crack, so
/// that direction is passed over too. When no direction gives a consistent
/// answer the point cannot be placed in or out of the volume, and the caller
/// is told so instead of being told it is outside.
pub fn point_in_volume_checked(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    exit_sign: &[i8],
    point: [f64; 3],
) -> Option<bool> {
    let directions = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.57735026919, 0.57735026919, 0.57735026919],
    ];
    for direction in &directions {
        let Some((count, nearest_leaves)) = count_crossings_facing(
            surface_bvh,
            tri_v0,
            tri_edge1,
            tri_edge2,
            exit_sign,
            point,
            *direction,
        ) else {
            continue;
        };
        let inside = count % 2 == 1;
        match nearest_leaves {
            Some(leaves) if leaves != inside => continue,
            _ => return Some(inside),
        }
    }
    None
}

/// Crossing count along a ray, as [`count_crossings_bvh`], and whether the
/// ray leaves the volume through the nearest facet it crosses (`None` with
/// no facet ahead, or a facet whose winding is unknown). `None` overall when
/// the direction cannot be used: a facet is crossed edge-on, or two facets
/// at the nearest crossing face opposite ways.
fn count_crossings_facing(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    exit_sign: &[i8],
    origin: [f64; 3],
    direction: [f64; 3],
) -> Option<(usize, Option<bool>)> {
    // Inline buffer as in count_crossings_bvh, spilling past it: this runs
    // on every point location, so it should not allocate in the common case.
    let mut hit_buf = [(0.0f64, 0i8); MAX_INLINE_HITS];
    let mut hit_count = 0usize;
    let mut spill: Vec<(f64, i8)> = Vec::new();
    let mut ambiguous = false;
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
            // +1 when the ray leaves the volume through this facet, -1 when
            // it enters, 0 when the facet's winding is unknown.
            let sign = exit_sign.get(idx).copied().unwrap_or(0);
            let facing = if d > 0.0 { sign } else { -sign };
            if hit_count < MAX_INLINE_HITS {
                hit_buf[hit_count] = (t, facing);
                hit_count += 1;
            } else {
                spill.push((t, facing));
            }
        }
    });
    if ambiguous {
        return None;
    }
    let hits: &mut [(f64, i8)] = if spill.is_empty() {
        &mut hit_buf[..hit_count]
    } else {
        spill.extend_from_slice(&hit_buf[..hit_count]);
        &mut spill
    };
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut crossings = 0usize;
    let mut prev_t = f64::NEG_INFINITY;
    for &(t, _) in hits.iter() {
        if (t - prev_t).abs() > 1e-8 {
            crossings += 1;
            prev_t = t;
        }
    }
    let Some(&(t0, _)) = hits.first() else {
        return Some((0, None));
    };
    // Every facet hit at the nearest crossing (one, or several meeting at an
    // edge or vertex there) must face the same way.
    let mut nearest: i8 = 0;
    for &(_, facing) in hits.iter().take_while(|h| h.0 - t0 <= 1e-8) {
        if facing == 0 {
            continue;
        }
        if nearest != 0 && nearest != facing {
            return None;
        }
        nearest = facing;
    }
    Some((crossings, (nearest != 0).then_some(nearest > 0)))
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
    // Hits past the inline buffer (a ray through a volume made of many lumps
    // or channels): dropping them would flip the parity.
    let mut spill: Vec<f64> = Vec::new();
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
            } else {
                spill.push(t);
            }
        }
    });

    if ambiguous {
        return None;
    }

    // Sort and deduplicate: hits at nearly the same distance are a single crossing
    let hits: &mut [f64] = if spill.is_empty() {
        &mut hit_buf[..hit_count]
    } else {
        spill.extend_from_slice(&hit_buf[..hit_count]);
        &mut spill
    };
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

/// Which volume contains a point, as [`find_volume_ranked`] (with no ranks,
/// every volume ranks the same and the first that contains the point is
/// it), using [`point_in_volume_checked`]. `None` when the point cannot be
/// placed: no volume is known to contain it, and at least one that might
/// (one ranked above any that does) cannot say.
pub(crate) fn try_find_volume(
    topo: &MeshTopology,
    surface_bvhs: &[crate::geometry::SurfaceBvhData],
    point: [f64; 3],
    ranks: &[u32],
) -> Option<VolumeId> {
    let bb = &topo.global_aabb;
    if point[0] < bb[0]
        || point[0] > bb[3]
        || point[1] < bb[1]
        || point[1] > bb[4]
        || point[2] < bb[2]
        || point[2] > bb[5]
    {
        return Some(topo.implicit_complement);
    }
    let rank = |v: VolumeId| ranks.get(v as usize).copied().unwrap_or(0);
    let mut best: Option<VolumeId> = None;
    let mut undecided: Option<u32> = None;
    for vol_id in 0..topo.num_volumes {
        if best.is_some_and(|b| rank(b) >= rank(vol_id)) {
            continue;
        }
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
        match point_in_volume_checked(
            &sd.bvh,
            &sd.precomputed.v0,
            &sd.precomputed.edge1,
            &sd.precomputed.edge2,
            &sd.exit_sign,
            point,
        ) {
            Some(true) => best = Some(vol_id),
            Some(false) => {}
            None => undecided = Some(undecided.map_or(rank(vol_id), |r| r.max(rank(vol_id)))),
        }
    }
    match (best, undecided) {
        (Some(b), Some(r)) if rank(b) < r => None,
        (Some(b), _) => Some(b),
        (None, Some(_)) => None,
        (None, None) => Some(topo.implicit_complement),
    }
}

/// Like [`find_volume`], but where several volumes contain the point (solids
/// whose overlaps were recorded), the one with the highest `ranks` entry.
pub(crate) fn find_volume_ranked(
    topo: &MeshTopology,
    surface_bvhs: &[crate::geometry::SurfaceBvhData],
    point: [f64; 3],
    ranks: &[u32],
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
    let mut best: Option<VolumeId> = None;
    for vol_id in 0..topo.num_volumes {
        if best.is_some_and(|b| ranks[b as usize] >= ranks[vol_id as usize]) {
            continue;
        }
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
            best = Some(vol_id);
        }
    }
    best.unwrap_or(topo.implicit_complement)
}
