//! XDG-style element walking through a tet mesh.
//!
//! A ray enters a tet, exits through one of its 4 faces, and enters
//! the adjacent tet. This produces `(TetrahedronId, path_length)` segments
//! used for mesh tally scoring.

use crate::accel::bvh::Bvh;
use crate::mesh::topology::{MeshTopology, TET_FACE_VERTICES};
use crate::query::intersect::{self, cross, dot, point_in_tet, ray_triangle_intersect, sub};
use crate::types::*;

/// Find the next element along a ray from the current element.
///
/// Tests all 4 faces of the tetrahedron, finds the exit face (minimum
/// positive intersection distance with outward-pointing normal check),
/// and returns the adjacent element and exit distance.
///
/// Returns `(next_tet, exit_distance)` where `next_tet` is `None` if
/// the ray exits the mesh boundary.
///
/// Relies on `topo.tetrahedra` being positively oriented (the invariant
/// `MeshTopology` enforces with `validate_tet_orientation`, which rejects a
/// mesh that breaks it): the outward-normal test reads face normals straight
/// off `TET_FACE_VERTICES`, which point inward on a negatively oriented tet.
pub fn next_element(
    topo: &MeshTopology,
    current: TetrahedronId,
    origin: [f64; 3],
    direction: [f64; 3],
) -> (Option<TetrahedronId>, f64) {
    let verts = &topo.tetrahedra[current as usize];
    let v = [
        topo.vertices[verts[0] as usize],
        topo.vertices[verts[1] as usize],
        topo.vertices[verts[2] as usize],
        topo.vertices[verts[3] as usize],
    ];

    let mut min_dist = f64::MAX;
    let mut exit_face = None;

    for (face_idx, face_verts) in TET_FACE_VERTICES.iter().enumerate() {
        let fv0 = v[face_verts[0]];
        let fv1 = v[face_verts[1]];
        let fv2 = v[face_verts[2]];

        // Compute face normal (outward from the tet -- away from opposite vertex)
        let normal = intersect::triangle_normal_unnormalized(fv0, fv1, fv2);

        // Only consider faces where the ray is exiting (dot > 0)
        if dot(direction, normal) <= 0.0 {
            continue;
        }

        if let Some(t) = ray_triangle_intersect(origin, direction, fv0, fv1, fv2) {
            if t < min_dist {
                min_dist = t;
                exit_face = Some(face_idx);
            }
        } else {
            // Try with relaxed epsilon for near-boundary cases
            // Compute intersection manually for nearly-zero distance
            let edge1 = sub(fv1, fv0);
            let edge2 = sub(fv2, fv0);
            let h = cross(direction, edge2);
            let a = dot(edge1, h);
            if a.abs() > 1e-30 {
                let f = 1.0 / a;
                let s = sub(origin, fv0);
                let u = f * dot(s, h);
                if (-1e-8..=1.0 + 1e-8).contains(&u) {
                    let q = cross(s, edge1);
                    let vv = f * dot(direction, q);
                    if vv >= -1e-8 && u + vv <= 1.0 + 1e-8 {
                        let t = f * dot(edge2, q);
                        if t >= 0.0 && t < min_dist {
                            min_dist = t;
                            exit_face = Some(face_idx);
                        }
                    }
                }
            }
        }
    }

    match exit_face {
        Some(fi) => {
            let next = topo.tet_adjacency[current as usize][fi];
            (next, min_dist.max(0.0))
        }
        None => {
            // Shouldn't happen for a valid mesh, but handle gracefully
            (None, 0.0)
        }
    }
}

/// Walk a ray through the tet mesh, accumulating segment lengths.
///
/// Walks wherever tet adjacency leads, which on a conformal multi-volume mesh
/// includes tets of other volumes; [`segments`] is the volume-scoped form.
///
/// Returns `Vec<(TetrahedronId, path_length)>`.
pub fn walk_elements(
    topo: &MeshTopology,
    start_element: TetrahedronId,
    origin: [f64; 3],
    direction: [f64; 3],
    max_distance: f64,
) -> Vec<(TetrahedronId, f64)> {
    let mut segments = Vec::new();
    walk_into(
        topo,
        start_element,
        origin,
        direction,
        max_distance,
        None,
        &mut segments,
    );
    segments
}

/// Walk a ray through the tet mesh, appending `(tet, path_length)` to `out`,
/// and return how far along `direction` the walk advanced before it left the
/// mesh or ran out of `max_distance`.
///
/// `volume`, when given, confines the walk to that volume's tets: tet
/// adjacency is built globally by shared face, so on a conformal mesh it links
/// tets across a volume interface and an unconfined walk would keep scoring
/// into the neighbouring volume's tets (issue #316).
///
/// The returned distance is what a caller needs to resume the ray past the
/// meshed region: it counts the per-face `SURFACE_BUMP` nudges as well as the
/// scored segment lengths, so `origin + returned * direction` is at or beyond
/// the exit point rather than a hair short of it (which would re-enter the tet
/// just left and double-count it).
fn walk_into(
    topo: &MeshTopology,
    start_element: TetrahedronId,
    origin: [f64; 3],
    direction: [f64; 3],
    max_distance: f64,
    volume: Option<VolumeId>,
    out: &mut Vec<(TetrahedronId, f64)>,
) -> f64 {
    let mut current = Some(start_element);
    let mut travelled = 0.0_f64;
    let mut steps = 0usize;
    const MAX_STEPS: usize = 100_000;

    while let Some(tet) = current {
        let remaining = max_distance - travelled;
        if remaining <= 1e-14 || steps >= MAX_STEPS {
            break;
        }
        steps += 1;

        // Recompute the position from the ray parameter each step rather than
        // accumulating it, so a long walk does not drift off the ray.
        let pos = intersect::add(origin, intersect::scale(direction, travelled));
        let (next, exit_dist) = next_element(topo, tet, pos, direction);
        let segment_len = exit_dist.min(remaining);

        if segment_len > 1e-14 {
            out.push((tet, segment_len));
            travelled += segment_len;
        }
        // Step across the face (also un-sticks a zero-distance step).
        travelled += intersect::SURFACE_BUMP;

        current = match (next, volume) {
            (Some(n), Some(vol)) if topo.tet_volume_ids[n as usize] != Some(vol) => None,
            (next, _) => next,
        };
    }

    travelled
}

/// Ray versus tetrahedron overlap, by clipping the ray against the tet's four
/// outward half-spaces.
///
/// Returns `Some((t_enter, t_exit))` in units of `direction` (which must be
/// normalised), with `t_enter <= t_exit`. `t_enter` is negative when `origin`
/// is already inside. Returns `None` when the ray misses the tet.
///
/// Half-space clipping rather than four ray-triangle tests: it has no
/// edge/vertex special cases, which matters because the entry face is exactly
/// where a track enters the meshed region. Takes bare vertices, so unlike
/// [`next_element`] it does not lean on the positive-orientation invariant:
/// each face normal is flipped to point away from the opposite vertex.
pub fn ray_tet_interval(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> Option<(f64, f64)> {
    const PARALLEL_TOL: f64 = 1e-12;
    let v = [v0, v1, v2, v3];
    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;

    for (face_idx, face_verts) in TET_FACE_VERTICES.iter().enumerate() {
        let a = v[face_verts[0]];
        let mut normal = intersect::triangle_normal(a, v[face_verts[1]], v[face_verts[2]]);
        // Face `i` is opposite vertex `i`: outward points away from it.
        if dot(normal, sub(v[face_idx], a)) > 0.0 {
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        // Inside the half-space is `dot(n, p - a) <= 0` (n points outward).
        let denom = dot(direction, normal);
        let num = dot(normal, sub(origin, a));
        if denom.abs() < PARALLEL_TOL {
            if num > PARALLEL_TOL {
                return None; // parallel to the face and outside it
            }
            continue;
        }
        let t = -num / denom;
        if denom < 0.0 {
            t_enter = t_enter.max(t); // entering through this face
        } else {
            t_exit = t_exit.min(t); // leaving through this face
        }
    }

    (t_enter <= t_exit).then_some((t_enter, t_exit))
}

/// First tetrahedron the ray enters within `max_distance`, and the distance to
/// its entry face. `origin` is assumed to be outside every tet of the volume
/// (the caller has already tried [`find_element`]).
fn first_element_along_ray(
    topo: &MeshTopology,
    element_bvh: &Bvh,
    element_bvh_tet_ids: &[TetrahedronId],
    origin: [f64; 3],
    direction: [f64; 3],
    max_distance: f64,
) -> Option<(TetrahedronId, f64)> {
    element_bvh
        .ray_traverse_upto(origin, direction, max_distance, |bvh_prim_idx| {
            let tet_id = element_bvh_tet_ids[bvh_prim_idx as usize];
            let verts = &topo.tetrahedra[tet_id as usize];
            let (t_enter, t_exit) = ray_tet_interval(
                origin,
                direction,
                topo.vertices[verts[0] as usize],
                topo.vertices[verts[1] as usize],
                topo.vertices[verts[2] as usize],
                topo.vertices[verts[3] as usize],
            )?;
            // Behind the origin, or a grazing touch with no path in it.
            if t_exit <= intersect::SURFACE_BUMP || t_exit - t_enter.max(0.0) <= 1e-14 {
                return None;
            }
            let t = t_enter.max(0.0);
            // `ray_traverse` only accepts strictly positive hit distances, and
            // the caller has already established that `origin` is outside the
            // mesh, so a t of exactly 0 is a boundary-grazing artefact.
            (t > 0.0 && t < max_distance).then_some(t)
        })
        .map(|(bvh_prim_idx, t)| (element_bvh_tet_ids[bvh_prim_idx as usize], t))
}

/// Find which tetrahedron contains a point, using the element BVH.
pub fn find_element(
    topo: &MeshTopology,
    element_bvh: &Bvh,
    element_bvh_tet_ids: &[TetrahedronId],
    point: [f64; 3],
) -> Option<TetrahedronId> {
    element_bvh
        .point_query(point, |bvh_prim_idx| {
            let tet_id = element_bvh_tet_ids[bvh_prim_idx as usize];
            let verts = &topo.tetrahedra[tet_id as usize];
            let v0 = topo.vertices[verts[0] as usize];
            let v1 = topo.vertices[verts[1] as usize];
            let v2 = topo.vertices[verts[2] as usize];
            let v3 = topo.vertices[verts[3] as usize];
            point_in_tet(point, v0, v1, v2, v3)
        })
        .map(|bvh_idx| element_bvh_tet_ids[bvh_idx as usize])
}

/// Compute all (element, distance) segments along a line from start to end.
///
/// `start` does not have to be inside the mesh: the track is ray-cast onto the
/// first tetrahedron it enters and walked from there, and the walk is resumed
/// after every exit until the segment is spent. A tet-mesh tally overlays a
/// small part of a much bigger model, so the usual case is a track that begins
/// outside it; scoring only tracks that *begin* in a tet made the track-length
/// estimator miss every entering track and read low (issue #316).
///
/// Only `volume`'s own tets are returned, matching what [`find_element`] (and
/// so the collision estimator) resolves for the same volume.
pub fn segments(
    topo: &MeshTopology,
    element_bvh: &Bvh,
    element_bvh_tet_ids: &[TetrahedronId],
    volume: VolumeId,
    start: [f64; 3],
    end: [f64; 3],
) -> Vec<(TetrahedronId, f64)> {
    // A track may leave the meshed region and come back (concave or
    // multi-part volumes); each pass costs one BVH ray query, so cap the
    // number of re-entries rather than let a grazing track spin.
    const MAX_ENTRIES: usize = 256;

    let diff = sub(end, start);
    let total_dist = intersect::length(diff);
    if total_dist < 1e-15 {
        return Vec::new();
    }
    let direction = intersect::normalize(diff);

    // A tally overlay is small next to the model, so most tracks come nowhere
    // near it. Reject those on the volume's AABB before touching the BVH.
    if let Some(bb) = topo.volume_aabbs.get(volume as usize) {
        if !intersect::segment_hits_aabb(start, direction, total_dist, bb) {
            return Vec::new();
        }
    }

    let mut out: Vec<(TetrahedronId, f64)> = Vec::new();
    let mut travelled = 0.0_f64;

    for _ in 0..MAX_ENTRIES {
        let remaining = total_dist - travelled;
        if remaining <= 1e-14 {
            break;
        }
        let pos = intersect::add(start, intersect::scale(direction, travelled));
        let entered = match find_element(topo, element_bvh, element_bvh_tet_ids, pos) {
            Some(tet) => (tet, 0.0),
            None => {
                match first_element_along_ray(
                    topo,
                    element_bvh,
                    element_bvh_tet_ids,
                    pos,
                    direction,
                    remaining,
                ) {
                    // Step a hair past the entry face so the walk starts
                    // inside the tet it was handed.
                    Some((tet, t)) => (tet, t + intersect::SURFACE_BUMP),
                    None => break,
                }
            }
        };
        let (tet, entry_dist) = entered;
        travelled += entry_dist;
        if total_dist - travelled <= 1e-14 {
            break;
        }
        let origin = intersect::add(start, intersect::scale(direction, travelled));
        let consumed = walk_into(
            topo,
            tet,
            origin,
            direction,
            total_dist - travelled,
            Some(volume),
            &mut out,
        );
        // `walk_into` advances at least one SURFACE_BUMP per step, so this
        // makes progress and the loop terminates.
        travelled += consumed.max(intersect::SURFACE_BUMP);
    }

    out
}

// ----------------------------- Tests -----------------------------

#[cfg(all(test, feature = "arrow"))]
mod tests {
    use super::*;

    fn load(name: &str) -> MeshTopology {
        let path = format!("tests/data/{name}");
        crate::io::arrow::build_topology(
            crate::io::arrow::read_arrow_mesh(std::path::Path::new(&path)).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_next_element_single_tet() {
        // Build a minimal topology with a single tet
        let topo = load("cube.arrow");

        // Pick the first tet and fire a ray through it
        if topo.tetrahedra.is_empty() {
            return;
        }

        // Compute centroid of first tet
        let verts = &topo.tetrahedra[0];
        let v0 = topo.vertices[verts[0] as usize];
        let v1 = topo.vertices[verts[1] as usize];
        let v2 = topo.vertices[verts[2] as usize];
        let v3 = topo.vertices[verts[3] as usize];
        let centroid = [
            (v0[0] + v1[0] + v2[0] + v3[0]) / 4.0,
            (v0[1] + v1[1] + v2[1] + v3[1]) / 4.0,
            (v0[2] + v1[2] + v2[2] + v3[2]) / 4.0,
        ];

        let (_next, dist) = next_element(&topo, 0, centroid, [0.0, 0.0, 1.0]);
        assert!(dist > 0.0, "Should exit through a face");
        // next may be Some (adjacent tet) or None (boundary)
    }

    #[test]
    fn test_walk_through_cube() {
        let topo = load("cube.arrow");

        // Build element BVH for the single volume
        let tet_aabbs: Vec<[f64; 6]> = topo.tet_aabbs.clone();
        let bvh = Bvh::build(&tet_aabbs);
        let tet_ids: Vec<TetrahedronId> = (0..topo.tetrahedra.len() as u32).collect();

        // Strictly inside one tet: the cube's 6 tets all meet along the
        // x = y = z main diagonal, so a point with equal coordinates sits on
        // a shared edge and gives zero-distance exits.
        let start = [0.5, 0.35, 0.3];
        let start_tet = find_element(&topo, &bvh, &tet_ids, start);
        assert!(start_tet.is_some(), "Should find a tet at {start:?}");

        // Walk in +z direction -- should cross tets until hitting z=1
        let segs = walk_elements(&topo, start_tet.unwrap(), start, [0.0, 0.0, 1.0], 10.0);
        assert!(!segs.is_empty());

        // The tets tile the cube, so the walk covers the whole way to z = 1.
        let total: f64 = segs.iter().map(|(_, d)| d).sum();
        assert!(
            (total - 0.7).abs() < 1e-6,
            "Walk should cover z = 0.3 to z = 1, got {total}"
        );
    }

    /// Issue #316: the element walk read face normals off `TET_FACE_VERTICES`,
    /// which point inward on a negatively oriented tet, so it picked an entry
    /// face as its exit and stopped short. Building a `MeshTopology` now fails
    /// on a negatively oriented tet, so the loaded fixtures are positive.
    #[test]
    fn every_tet_is_positively_oriented() {
        for name in ["cube.arrow", "two_region_tets.arrow"] {
            let topo = load(name);
            for (tet_idx, verts) in topo.tetrahedra.iter().enumerate() {
                let signed = intersect::signed_tet_volume(
                    topo.vertices[verts[0] as usize],
                    topo.vertices[verts[1] as usize],
                    topo.vertices[verts[2] as usize],
                    topo.vertices[verts[3] as usize],
                );
                assert!(
                    signed > 0.0,
                    "{name} tet {tet_idx} has signed volume {signed}"
                );
            }
        }
    }

    /// Issue #316: a track that starts outside the mesh must still be scored.
    /// `segments` used to return nothing unless its start point was already
    /// inside a tet, which is the minority case for a tally overlay.
    ///
    /// Reference is analytic: `cube.arrow` tiles the unit cube exactly, so the
    /// walked path length must equal the ray's chord through `[0, 1]^3`.
    #[test]
    fn segments_match_the_analytic_chord_from_outside() {
        let topo = load("cube.arrow");
        let bvh = Bvh::build(&topo.tet_aabbs);
        let tet_ids: Vec<TetrahedronId> = (0..topo.tetrahedra.len() as u32).collect();

        // Deterministic xorshift so the sweep is reproducible.
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut checked = 0usize;
        for _ in 0..5000 {
            let start = [
                -2.0 + 6.0 * next(),
                -2.0 + 6.0 * next(),
                -2.0 + 6.0 * next(),
            ];
            let end = [
                -2.0 + 6.0 * next(),
                -2.0 + 6.0 * next(),
                -2.0 + 6.0 * next(),
            ];
            let d = sub(end, start);
            let len = intersect::length(d);
            if len < 1e-9 {
                continue;
            }
            let dir = intersect::normalize(d);

            // Analytic slab clip of the segment against the unit cube.
            let (mut t0, mut t1) = (0.0_f64, len);
            for i in 0..3 {
                if dir[i].abs() < 1e-12 {
                    if !(0.0..=1.0).contains(&start[i]) {
                        t0 = 1.0;
                        t1 = 0.0;
                    }
                    continue;
                }
                let inv = 1.0 / dir[i];
                let (mut a, mut b) = ((0.0 - start[i]) * inv, (1.0 - start[i]) * inv);
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                t0 = t0.max(a);
                t1 = t1.min(b);
            }
            let chord = (t1 - t0).max(0.0);
            if chord < 1e-6 {
                continue; // grazing: skip the tolerance-dominated cases
            }
            checked += 1;

            let segs = segments(&topo, &bvh, &tet_ids, 0, start, end);
            let walked: f64 = segs.iter().map(|&(_, l)| l).sum();
            assert!(
                (walked - chord).abs() < 1e-6,
                "segment {start:?} -> {end:?}: walked {walked}, analytic chord {chord}"
            );
        }
        assert!(checked > 200, "sweep degenerated: only {checked} rays hit");
    }

    /// Issue #316: tet adjacency is global and links tets across a volume
    /// interface on a conformal mesh, so an unconfined walk kept scoring into
    /// the neighbouring volume's tets. A tally on volume V must see only V's
    /// tets, matching what `find_element(V, ..)` resolves for the collision
    /// estimator.
    #[test]
    fn segments_stay_inside_the_queried_volume() {
        let topo = load("two_region_tets.arrow");

        // Both half-boxes of two_region_tets.arrow span x in [0, 0.5] and [0.5, 1].
        for volume in 0..topo.num_volumes {
            let range = &topo.volume_tet_ranges[volume as usize];
            let tet_ids: Vec<TetrahedronId> =
                topo.volume_tet_indices[range.start as usize..range.end as usize].to_vec();
            let aabbs: Vec<[f64; 6]> = tet_ids
                .iter()
                .map(|&t| topo.tet_aabbs[t as usize])
                .collect();
            let bvh = Bvh::build(&aabbs);

            // Starts outside the mesh and crosses both halves end to end.
            let segs = segments(
                &topo,
                &bvh,
                &tet_ids,
                volume,
                [-0.5, 0.4, 0.3],
                [1.5, 0.4, 0.3],
            );
            let total: f64 = segs.iter().map(|&(_, l)| l).sum();
            assert!(
                (total - 0.5).abs() < 1e-6,
                "volume {volume} should see only its own 0.5 cm half, got {total}"
            );
            for &(tet, _) in &segs {
                assert_eq!(
                    topo.tet_volume_ids[tet as usize],
                    Some(volume),
                    "tet {tet} is not in volume {volume}"
                );
            }
        }
    }
}
