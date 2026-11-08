//! Point containment and volume location queries.

use crate::accel::bvh::Bvh;
use crate::mesh::topology::MeshTopology;
use crate::query::intersect::{self, dot, ray_triangle_parity, ParityHit, RAY_MIN_T};
use crate::types::*;

/// The directions parity rays are cast along, in the order they are tried.
///
/// No direction lies along an axis or a face or body diagonal, or in a
/// coordinate plane, where the edges and vertices of a mesh with symmetry
/// planes or axis-aligned faces line up. A ray meets those as a rule only
/// when it runs exactly along such a line or plane, so a tilt of a few
/// degrees keeps clear of them as well as a larger one.
///
/// The first is 2.5 degrees off +z, at an azimuth 8 degrees from that of
/// every direction whose x and y components are integers up to 2. A model
/// lying in the xy plane, as a fusion device does, is thinnest along z, and
/// the closer a ray is to it, the fewer BVH nodes it visits: in the meshes of
/// issue #502 this one visits 3 to 4% more than +z, and one 10 degrees off
/// 11 to 17% more. The others are each at least 8 degrees from every
/// direction whose components are integers up to 2, 4 degrees from those up
/// to 4, and 14 degrees out of every coordinate plane. The lines all seven
/// lie along are at least 27 degrees apart, so a ray that runs along some
/// feature of the mesh is followed by one well away from it.
const PARITY_DIRECTIONS: [[f64; 3]; 7] = [
    [0.0357309103309484, 0.0250190527608253, 0.9990482215818578],
    [-0.569848302323069, 0.6847158248944284, -0.45435355339059336],
    [0.5612734249900038, 0.4465056186957721, -0.6968535533905933],
    [-0.6960436780244805, 0.5561242696683923, 0.45414644660940673],
    [0.46236804345907595, 0.6778171821860707, 0.5716464466094067],
    [0.37435049357199485, -0.257374541036516, -0.8908535533905932],
    [0.2622308380377364, 0.8925544565721028, -0.3668535533905932],
];

/// How far along a parity ray a crossing must be to count: first any
/// distance, then, when no ray can be counted that way, more than
/// [`RAY_MIN_T`].
///
/// A point a hair's breadth off a surface is on its side of it, so the
/// first pass counts a crossing however close it is: the volume on the
/// other side would not even be asked when the point is outside its
/// bounding box. A point within round-off of an edge or vertex meets the
/// facets there along every ray, as close to the edge as it is itself, so
/// often no ray can be counted. Such a point is on the surface, where either
/// side is a fair answer, and the second pass leaves out a facet that close,
/// as a ray fired from a surface does.
const PARITY_MIN_T: [f64; 2] = [0.0, RAY_MIN_T];

/// Test if a point is inside a volume using ray casting.
///
/// Fires a ray from the point and counts surface crossings: odd is inside,
/// even outside. A ray that grazes a facet, or runs through or within
/// round-off of an edge or vertex, cannot be counted, and the next direction
/// is tried (see [`PARITY_MIN_T`] for a point on an edge or vertex). Outside
/// when no direction can be counted.
pub fn point_in_volume(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    point: [f64; 3],
) -> bool {
    for min_t in PARITY_MIN_T {
        for direction in PARITY_DIRECTIONS {
            let count = count_crossings_bvh(
                surface_bvh,
                tri_v0,
                tri_edge1,
                tri_edge2,
                point,
                direction,
                min_t,
            );
            if let Some(count) = count {
                return count % 2 == 1;
            }
        }
    }
    false
}

/// Like [`point_in_volume`], but `None` when the mesh does not say.
///
/// Parity alone takes the first direction whose ray can be counted, and
/// answers outside when none can. Here each direction's parity is also
/// checked against the facet nearest the point along it: from inside a
/// closed, outward-wound volume the ray leaves through that facet, from
/// outside it enters (`exit_sign` gives each facet's winding relative to the
/// volume, `+1` outward and `-1` inward). Parity and facing disagree near a
/// folded or flipped facet, or where the ray slips through a crack, so that
/// direction is passed over too. When no direction gives a consistent answer
/// (on either pass of [`PARITY_MIN_T`]) the point cannot be placed in or out
/// of the volume, and the caller is told so instead of being told it is
/// outside.
pub fn point_in_volume_checked(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    exit_sign: &[i8],
    point: [f64; 3],
) -> Option<bool> {
    for min_t in PARITY_MIN_T {
        for direction in PARITY_DIRECTIONS {
            let Some((count, nearest_leaves)) = count_crossings_facing(
                surface_bvh,
                tri_v0,
                tri_edge1,
                tri_edge2,
                exit_sign,
                point,
                direction,
                min_t,
            ) else {
                continue;
            };
            let inside = count % 2 == 1;
            match nearest_leaves {
                Some(leaves) if leaves != inside => continue,
                _ => return Some(inside),
            }
        }
    }
    None
}

/// Crossing count along a ray, as [`count_crossings_bvh`], and whether the
/// ray leaves the volume through the nearest facet it crosses (`None` with
/// no facet ahead, or a facet whose winding is unknown). `None` overall when
/// the direction cannot be used: the ray cannot be counted, or two facets at
/// the nearest crossing face opposite ways.
#[allow(clippy::too_many_arguments)]
fn count_crossings_facing(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    exit_sign: &[i8],
    origin: [f64; 3],
    direction: [f64; 3],
    min_t: f64,
) -> Option<(usize, Option<bool>)> {
    // Every crossing and its facing, in an inline buffer that spills to the
    // heap (a ray through a volume made of many lumps or channels): this runs
    // on every point location, so it should not allocate in the common case.
    let mut hit_buf = [(0.0f64, 0i8); MAX_INLINE_HITS];
    let mut hit_count = 0usize;
    let mut spill: Vec<(f64, i8)> = Vec::new();
    let mut ambiguous = false;
    surface_bvh.ray_traverse_collect(origin, direction, |bvh_prim_idx| {
        if ambiguous {
            return;
        }
        let idx = bvh_prim_idx as usize;
        let (e1, e2) = (tri_edge1[idx], tri_edge2[idx]);
        match ray_triangle_parity(origin, direction, tri_v0[idx], e1, e2, min_t) {
            ParityHit::Miss => {}
            ParityHit::Ambiguous => ambiguous = true,
            ParityHit::Cross(t) => {
                // +1 when the ray leaves the volume through this facet, -1
                // when it enters, 0 when the facet's winding is unknown.
                let sign = exit_sign.get(idx).copied().unwrap_or(0);
                let facing = if dot(direction, intersect::cross(e1, e2)) > 0.0 {
                    sign
                } else {
                    -sign
                };
                if hit_count < MAX_INLINE_HITS {
                    hit_buf[hit_count] = (t, facing);
                    hit_count += 1;
                } else {
                    spill.push((t, facing));
                }
            }
        }
    });
    if ambiguous {
        return None;
    }
    let hits: &[(f64, i8)] = if spill.is_empty() {
        &hit_buf[..hit_count]
    } else {
        spill.extend_from_slice(&hit_buf[..hit_count]);
        &spill
    };
    let Some(t0) = hits.iter().map(|h| h.0).min_by(f64::total_cmp) else {
        return Some((0, None));
    };
    // Every facet hit at the nearest crossing (one, or several lying on top of
    // each other) must face the same way. A ray in and out of the volume
    // within 1e-8 cm, clipping a corner, is passed over by this too.
    let mut nearest: i8 = 0;
    for &(t, facing) in hits {
        if t - t0 > 1e-8 || facing == 0 {
            continue;
        }
        if nearest != 0 && nearest != facing {
            return None;
        }
        nearest = facing;
    }
    Some((hits.len(), (nearest != 0).then_some(nearest > 0)))
}

/// Max inline hits for stack-allocated crossing buffer.
const MAX_INLINE_HITS: usize = 64;

/// How many times a ray crosses the facets, `None` when it cannot be counted
/// (see [`ray_triangle_parity`]).
///
/// Every crossing counts, however close to the next. A ray through an edge
/// that both facets would report is not counted at all, so two crossings a
/// hair apart are two: a ray clipping a corner of the surface, entering and
/// leaving, or a face with the volume on both sides, whose facets come twice
/// in the volume's surface, once each way. Taken as one, either flips the
/// parity.
fn count_crossings_bvh(
    surface_bvh: &Bvh,
    tri_v0: &[[f64; 3]],
    tri_edge1: &[[f64; 3]],
    tri_edge2: &[[f64; 3]],
    origin: [f64; 3],
    direction: [f64; 3],
    min_t: f64,
) -> Option<usize> {
    let mut crossings = 0usize;
    let mut ambiguous = false;
    // Collect-all traversal: parity needs EVERY hit along the ray, so the
    // nearest-hit pruning of ray_traverse must not be used here.
    surface_bvh.ray_traverse_collect(origin, direction, |bvh_prim_idx| {
        if ambiguous {
            return;
        }
        let idx = bvh_prim_idx as usize;
        let (v0, e1, e2) = (tri_v0[idx], tri_edge1[idx], tri_edge2[idx]);
        match ray_triangle_parity(origin, direction, v0, e1, e2, min_t) {
            ParityHit::Miss => {}
            ParityHit::Ambiguous => ambiguous = true,
            ParityHit::Cross(_) => crossings += 1,
        }
    });
    (!ambiguous).then_some(crossings)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::topology::build_triangle_aabbs;
    use std::f64::consts::PI;

    /// A triangle soup as one volume's surface BVH, wound outward.
    struct Surface {
        bvh: Bvh,
        v0: Vec<[f64; 3]>,
        edge1: Vec<[f64; 3]>,
        edge2: Vec<[f64; 3]>,
        exit_sign: Vec<i8>,
    }

    impl Surface {
        fn new(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Self {
            let corner = |t: &[u32; 3], k: usize| vertices[t[k] as usize];
            let edge = |k: usize| -> Vec<[f64; 3]> {
                triangles
                    .iter()
                    .map(|t| intersect::sub(corner(t, k), corner(t, 0)))
                    .collect()
            };
            Self {
                bvh: Bvh::build(&build_triangle_aabbs(triangles, vertices)),
                v0: triangles.iter().map(|t| corner(t, 0)).collect(),
                edge1: edge(1),
                edge2: edge(2),
                exit_sign: vec![1; triangles.len()],
            }
        }

        /// Crossings counted at any distance, as the first pass does.
        fn count(&self, origin: [f64; 3], direction: [f64; 3]) -> Option<usize> {
            count_crossings_bvh(
                &self.bvh,
                &self.v0,
                &self.edge1,
                &self.edge2,
                origin,
                direction,
                0.0,
            )
        }

        fn contains(&self, p: [f64; 3]) -> bool {
            point_in_volume(&self.bvh, &self.v0, &self.edge1, &self.edge2, p)
        }

        fn contains_checked(&self, p: [f64; 3]) -> Option<bool> {
            point_in_volume_checked(
                &self.bvh,
                &self.v0,
                &self.edge1,
                &self.edge2,
                &self.exit_sign,
                p,
            )
        }
    }

    /// The octahedron |x| + |y| + |z| <= 1.
    fn octahedron() -> Surface {
        let vertices = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut triangles = Vec::new();
        for (x, y) in [(0, 2), (2, 1), (1, 3), (3, 0)] {
            triangles.push([x, y, 4]);
            triangles.push([y, x, 5]);
        }
        Surface::new(&vertices, &triangles)
    }

    /// The cube [-1, 1]^3, each face split along the diagonal through its
    /// centre from its first corner to its third.
    fn cube() -> Surface {
        let mut vertices = Vec::new();
        for z in [-1.0, 1.0] {
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push([x, y, z]);
            }
        }
        let mut triangles = Vec::new();
        for [a, b, c, d] in [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ] {
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
        Surface::new(&vertices, &triangles)
    }

    /// A torus as in issue #502: major radius 600, minor radius 150, on a
    /// 24 x 120 (theta x zeta) grid, so its zeta = 0 grid line lies in the
    /// plane y = 0 and the vertex (600, 0, 150) is straight above (600, 0, 0).
    fn torus() -> Surface {
        let (n_theta, n_zeta) = (24u32, 120u32);
        let mut vertices = Vec::new();
        for i in 0..n_theta {
            let theta = 2.0 * PI * f64::from(i) / f64::from(n_theta);
            let r = 600.0 + 150.0 * theta.cos();
            for j in 0..n_zeta {
                let zeta = 2.0 * PI * f64::from(j) / f64::from(n_zeta);
                vertices.push([r * zeta.cos(), r * zeta.sin(), 150.0 * theta.sin()]);
            }
        }
        let id = |i: u32, j: u32| (i % n_theta) * n_zeta + j % n_zeta;
        let mut triangles = Vec::new();
        for i in 0..n_theta {
            for j in 0..n_zeta {
                let (a, b, c, d) = (id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j));
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        Surface::new(&vertices, &triangles)
    }

    const UP: [f64; 3] = [0.0, 0.0, 1.0];

    #[test]
    fn a_ray_through_a_shared_vertex_or_edge_is_not_counted() {
        let oct = octahedron();
        // Only touching the octahedron, at a corner or an edge of its equator:
        // counted once, this put the point inside.
        assert_eq!(oct.count([1.0, 0.0, -2.0], UP), None);
        assert_eq!(oct.count([0.5, 0.5, -2.0], UP), None);
        // Leaving it through the top corner, or an edge down from it.
        assert_eq!(oct.count([0.0, 0.0, -0.5], UP), None);
        assert_eq!(oct.count([0.5, 0.0, 0.2], UP), None);
        // Clear of every edge.
        assert_eq!(oct.count([0.2, 0.1, -2.0], UP), Some(2));
        assert_eq!(oct.count([0.2, 0.1, 0.0], UP), Some(1));
    }

    #[test]
    fn a_ray_within_round_off_of_an_edge_is_not_counted() {
        // 1e-15 to 1e-16 cm off the plane y = 0, a ray up from (600, 0, 0)
        // runs that close to the edges of the zeta = 0 grid line, next to the
        // vertex (600, 0, 150). It used to slip between the facets there at
        // some of these points, and count no crossing from inside the torus.
        let torus = torus();
        for y in [1e-15, -1e-15, 1e-16, -1e-16] {
            for m in 0..200 {
                let phi = 2.0 * PI * (f64::from(m) + 0.5) / 200.0;
                let p = [600.0 + 1e-10 * phi.cos(), y, 1e-10 * phi.sin()];
                assert_ne!(torus.count(p, UP), Some(0), "{p:?}");
                assert!(torus.contains(p), "{p:?}");
                assert_eq!(torus.contains_checked(p), Some(true), "{p:?}");
            }
        }
    }

    #[test]
    fn a_point_whose_first_ray_meets_a_vertex_or_edge_is_placed_by_the_next() {
        let cube = cube();
        for d in PARITY_DIRECTIONS[..2].iter().copied() {
            let s = d.map(f64::signum);
            let back = |c: [f64; 3], t: f64| [0, 1, 2].map(|k| c[k] - t * d[k]);
            // From inside, out through the corner d points to, an edge from
            // that corner, or the centre of a face meeting there, on the
            // diagonal the face is split along.
            for target in [
                s,
                [s[0], s[1], 0.0],
                [s[0], 0.0, s[2]],
                [0.0, s[1], s[2]],
                [s[0], 0.0, 0.0],
                [0.0, s[1], 0.0],
                [0.0, 0.0, s[2]],
            ] {
                let p = back(target, 0.5);
                assert_eq!(cube.count(p, d), None, "{p:?}");
                assert!(cube.contains(p), "{p:?}");
                assert_eq!(cube.contains_checked(p), Some(true), "{p:?}");
            }
            // From outside, only touching the cube at a corner or an edge.
            for target in [[s[0], s[1], -s[2]], [s[0], 0.0, -s[2]]] {
                let p = back(target, 2.0);
                assert_eq!(cube.count(p, d), None, "{p:?}");
                assert!(!cube.contains(p), "{p:?}");
                assert_eq!(cube.contains_checked(p), Some(false), "{p:?}");
            }
        }
    }

    #[test]
    fn a_ray_aimed_within_round_off_of_a_vertex_or_edge_is_passed_over() {
        // From well inside or outside the torus, the first two parity rays
        // aimed at its vertices and at the middles of its edges, which they
        // pass within round-off of. Facets that each decided alone could let
        // such a ray slip through the surface, or only touch it.
        let torus = torus();
        // The facets lie between 148.4 and 150 cm from the torus's core circle.
        let inside = |p: [f64; 3]| {
            let r = (p[0].hypot(p[1]) - 600.0).hypot(p[2]);
            (r < 147.0 || r > 151.0).then_some(r < 147.0)
        };
        let mut aimed = 0;
        for d in PARITY_DIRECTIONS[..2].iter().copied() {
            for k in (0..torus.v0.len()).step_by(7) {
                let (v0, e1, e2) = (torus.v0[k], torus.edge1[k], torus.edge2[k]);
                for (target, back) in [
                    (v0, 40.0),
                    ([0, 1, 2].map(|i| v0[i] + 0.5 * e1[i]), 90.0),
                    ([0, 1, 2].map(|i| v0[i] + 0.5 * e2[i]), 250.0),
                ] {
                    let p = [0, 1, 2].map(|i| target[i] - back * d[i]);
                    let Some(want) = inside(p) else { continue };
                    aimed += 1;
                    assert_eq!(torus.count(p, d), None, "{p:?}");
                    assert_eq!(torus.contains(p), want, "{p:?}");
                    assert_eq!(torus.contains_checked(p), Some(want), "{p:?}");
                }
            }
        }
        assert!(aimed > 2000, "{aimed}");
    }

    #[test]
    fn a_ray_clipping_a_corner_crosses_the_surface_twice() {
        // From outside the octahedron, the first parity ray passing a hair
        // inside a corner or an edge of its equator: it enters and leaves
        // within 1e-8 cm. Taken as one crossing, that put the point inside.
        let oct = octahedron();
        let d = PARITY_DIRECTIONS[0];
        for target in [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ] {
            for depth in [1e-10, 1e-9, 3e-9] {
                let p = [0, 1, 2].map(|k| target[k] * (1.0 - depth) - 0.5 * d[k]);
                assert_eq!(oct.count(p, d), Some(2), "{p:?}");
                assert!(!oct.contains(p), "{p:?}");
                assert_eq!(oct.contains_checked(p), Some(false), "{p:?}");
            }
        }
    }

    #[test]
    fn a_point_on_an_edge_or_vertex_is_put_on_one_side_of_the_surface() {
        // A ray from a point on an edge or vertex, within round-off, meets the
        // facets there as close to the edge as the point is, so often no ray
        // can be counted. The point is on the surface, where either side will
        // do, and the second pass, which leaves out facets that close, places
        // it.
        let torus = torus();
        let mut uncounted = 0;
        for k in (0..torus.v0.len()).step_by(5) {
            let (v0, e1, e2) = (torus.v0[k], torus.edge1[k], torus.edge2[k]);
            for p in [
                v0,
                [0, 1, 2].map(|i| v0[i] + 0.5 * e1[i]),
                [0, 1, 2].map(|i| v0[i] + 0.5 * e2[i]),
            ] {
                if PARITY_DIRECTIONS
                    .iter()
                    .all(|&d| torus.count(p, d).is_none())
                {
                    uncounted += 1;
                }
                assert!(torus.contains_checked(p).is_some(), "{p:?}");
            }
        }
        assert!(uncounted > 100, "{uncounted}");
    }

    #[test]
    fn the_parity_directions_keep_clear_of_lattice_directions_and_each_other() {
        let lattice = |m: i32| -> Vec<[f64; 3]> {
            (-m..=m)
                .flat_map(|x| (-m..=m).flat_map(move |y| (-m..=m).map(move |z| [x, y, z])))
                .filter(|v| v != &[0, 0, 0])
                .map(|v| intersect::normalize(v.map(f64::from)))
                .collect()
        };
        let (near, far) = (lattice(2), lattice(4));
        for (i, d) in PARITY_DIRECTIONS.iter().enumerate() {
            assert!((dot(*d, *d) - 1.0).abs() < 1e-15, "{d:?}");
            for e in &PARITY_DIRECTIONS[i + 1..] {
                assert!(dot(*d, *e).abs() < 27f64.to_radians().cos(), "{d:?} {e:?}");
            }
            if i == 0 {
                continue;
            }
            assert!(
                d.iter().all(|c| c.abs() >= 14f64.to_radians().sin()),
                "{d:?}"
            );
            for l in &near {
                assert!(dot(*d, *l) < 8f64.to_radians().cos(), "{d:?} {l:?}");
            }
            for l in &far {
                assert!(dot(*d, *l) < 4f64.to_radians().cos(), "{d:?} {l:?}");
            }
        }
        // The first, a few degrees off +z, out of the coordinate planes, and
        // at an azimuth 8 degrees from that of every direction whose x and y
        // components are integers up to 2.
        let [x, y, z] = PARITY_DIRECTIONS[0];
        let off_z = z.acos().to_degrees();
        assert!((2.0..3.0).contains(&off_z), "{off_z}");
        assert!(x.abs().min(y.abs()) >= 1f64.to_radians().sin(), "{x} {y}");
        let across = intersect::normalize([x, y, 0.0]);
        for l in near.iter().filter(|l| l[2] == 0.0) {
            assert!(dot(across, *l).abs() < 8f64.to_radians().cos(), "{l:?}");
        }
    }
}
