//! Contacts between independently meshed solids.
//!
//! A discontinuous (non-imprinted) mesh gives every solid its own closed
//! boundary, so two solids that touch in the CAD meet along two separate
//! surfaces: coincident where both were faceted the same way, crossing each
//! other where they were not (thin overlap and gap slivers up to the faceting
//! tolerance either side). Neither surface knows about the other, so the
//! surface topology alone cannot say which solid a particle enters when it
//! leaves one through such a surface.
//!
//! [`Contacts`] records, for every triangle that faces the implicit
//! complement, the other solids that come within the contact tolerance of it.
//! Those are the only solids a particle leaving through that triangle can be
//! inside, so relocation tests them and nothing else, and a triangle with no
//! contacts (the outside of the model) hands the particle to the implicit
//! complement without any test at all.

use crate::accel::bvh::Bvh;
use crate::mesh::topology::MeshTopology;
use crate::query::closest::closest_point_on_triangle;
use crate::query::intersect::{add, cross, dot, scale, sub};
use crate::types::{TriangleId, VolumeId};

/// Per-triangle lists of touching solids, in compressed-row form.
#[derive(Clone, Debug, Default)]
pub struct Contacts {
    /// `offsets[t]..offsets[t + 1]` slices `volumes` for triangle `t`. Empty
    /// when no contacts were computed.
    offsets: Vec<u32>,
    volumes: Vec<VolumeId>,
}

impl Contacts {
    /// The solids touching triangle `tri`, ascending by id.
    #[inline]
    pub fn of(&self, tri: TriangleId) -> &[VolumeId] {
        let t = tri as usize;
        if t + 1 >= self.offsets.len() {
            return &[];
        }
        &self.volumes[self.offsets[t] as usize..self.offsets[t + 1] as usize]
    }

    /// Number of (triangle, solid) contact entries.
    pub fn len(&self) -> usize {
        self.volumes.len()
    }

    /// True when no triangle touches another solid.
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Find, for every triangle bounding exactly one solid, the other solids
    /// with a triangle within `tolerance` of it (touching, crossing or
    /// overlapping it). Triangles of shared (imprinted) surfaces and of the
    /// graveyard get no contacts: the topology already says what is across a
    /// shared surface, and nothing is across the graveyard.
    pub fn build(topo: &MeshTopology, tolerance: f64) -> Self {
        let n_tri = topo.triangles.len();
        let num_volumes = topo.num_volumes;
        let real = |v: Option<VolumeId>| v.filter(|&v| v < num_volumes);

        let mut tri_surface = vec![u32::MAX; n_tri];
        for (s, range) in topo.surface_tri_ranges.iter().enumerate() {
            for i in range.start..range.end {
                tri_surface[topo.surface_tri_indices[i as usize] as usize] = s as u32;
            }
        }
        let owners = |t: usize| -> [Option<VolumeId>; 2] {
            match tri_surface[t] {
                u32::MAX => [None, None],
                s => {
                    let (fwd, rev) = topo.surface_volumes[s as usize];
                    [real(fwd), real(rev)]
                }
            }
        };
        let vertices = |t: usize| {
            let tri = topo.triangles[t];
            [
                topo.vertices[tri[0] as usize],
                topo.vertices[tri[1] as usize],
                topo.vertices[tri[2] as usize],
            ]
        };

        // One BVH over every triangle that bounds a solid.
        let solid_tris: Vec<u32> = (0..n_tri as u32)
            .filter(|&t| owners(t as usize).iter().any(Option::is_some))
            .collect();
        let aabbs: Vec<[f64; 6]> = solid_tris
            .iter()
            .map(|&t| topo.triangle_aabbs[t as usize])
            .collect();
        let bvh = Bvh::build(&aabbs);

        let touching = |t: usize| -> Vec<VolumeId> {
            let owner = match owners(t) {
                [Some(v), None] | [None, Some(v)] => v,
                _ => return Vec::new(),
            };
            let bb = topo.triangle_aabbs[t];
            let q = [
                bb[0] - tolerance,
                bb[1] - tolerance,
                bb[2] - tolerance,
                bb[3] + tolerance,
                bb[4] + tolerance,
                bb[5] + tolerance,
            ];
            let a = vertices(t);
            let mut out: Vec<VolumeId> = Vec::new();
            bvh.aabb_query(q, |prim| {
                let s = solid_tris[prim as usize] as usize;
                let fresh = |v: &VolumeId| *v != owner && !out.contains(v);
                if !owners(s).iter().flatten().any(fresh) {
                    return;
                }
                let sb = topo.triangle_aabbs[s];
                if sb[3] < q[0]
                    || sb[0] > q[3]
                    || sb[4] < q[1]
                    || sb[1] > q[4]
                    || sb[5] < q[2]
                    || sb[2] > q[5]
                {
                    return;
                }
                if triangles_within(&a, &vertices(s), tolerance) {
                    let add: Vec<VolumeId> =
                        owners(s).into_iter().flatten().filter(fresh).collect();
                    out.extend(add);
                }
            });
            out.sort_unstable();
            out
        };

        #[cfg(feature = "parallel")]
        let per_tri: Vec<Vec<VolumeId>> = {
            use rayon::prelude::*;
            (0..n_tri).into_par_iter().map(touching).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let per_tri: Vec<Vec<VolumeId>> = (0..n_tri).map(touching).collect();

        let mut offsets = Vec::with_capacity(n_tri + 1);
        let mut volumes = Vec::new();
        offsets.push(0u32);
        for list in per_tri {
            volumes.extend(list);
            offsets.push(volumes.len() as u32);
        }
        Contacts { offsets, volumes }
    }
}

/// True when triangles `a` and `b` come within `tol` of each other, including
/// when they intersect. Exact: the closest pair of points of two disjoint
/// triangles is a vertex against the other triangle or an edge against an
/// edge, and two intersecting triangles have an edge of one crossing the
/// other.
pub fn triangles_within(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], tol: f64) -> bool {
    for i in 0..3 {
        if closest_point_on_triangle(a[i], b[0], b[1], b[2]).1 <= tol
            || closest_point_on_triangle(b[i], a[0], a[1], a[2]).1 <= tol
        {
            return true;
        }
    }
    let tol2 = tol * tol;
    for i in 0..3 {
        let (p1, q1) = (a[i], a[(i + 1) % 3]);
        for j in 0..3 {
            if segment_distance2(p1, q1, b[j], b[(j + 1) % 3]) <= tol2 {
                return true;
            }
        }
    }
    (0..3).any(|i| {
        segment_crosses_triangle(a[i], a[(i + 1) % 3], b)
            || segment_crosses_triangle(b[i], b[(i + 1) % 3], a)
    })
}

/// Squared distance between segments `p1 q1` and `p2 q2` (Ericson,
/// Real-Time Collision Detection, 5.1.9).
fn segment_distance2(p1: [f64; 3], q1: [f64; 3], p2: [f64; 3], q2: [f64; 3]) -> f64 {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    const EPS: f64 = 1e-300;
    let (s, t) = if a <= EPS && e <= EPS {
        (0.0, 0.0)
    } else if a <= EPS {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= EPS {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s = if denom > 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    let c1 = add(p1, scale(d1, s));
    let c2 = add(p2, scale(d2, t));
    let d = sub(c1, c2);
    dot(d, d)
}

/// True when segment `p q` crosses triangle `tri` (Moller-Trumbore over the
/// segment's own parameter range). A segment lying in the triangle's plane is
/// left to the distance tests.
fn segment_crosses_triangle(p: [f64; 3], q: [f64; 3], tri: &[[f64; 3]; 3]) -> bool {
    let d = sub(q, p);
    let e1 = sub(tri[1], tri[0]);
    let e2 = sub(tri[2], tri[0]);
    let h = cross(d, e2);
    let det = dot(e1, h);
    if det.abs() < 1e-300 {
        return false;
    }
    let f = 1.0 / det;
    let s = sub(p, tri[0]);
    let u = f * dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qv = cross(s, e1);
    let v = f * dot(d, qv);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    (0.0..=1.0).contains(&(f * dot(e2, qv)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    fn shifted(dz: f64, dx: f64) -> [[f64; 3]; 3] {
        T.map(|v| [v[0] + dx, v[1], v[2] + dz])
    }

    #[test]
    fn coincident_triangles_touch() {
        assert!(triangles_within(&T, &T, 0.0));
    }

    #[test]
    fn parallel_offset_is_measured_exactly() {
        let b = shifted(0.01, 0.0);
        assert!(triangles_within(&T, &b, 0.0101));
        assert!(!triangles_within(&T, &b, 0.0099));
    }

    #[test]
    fn crossing_triangles_touch_with_zero_tolerance() {
        // b stands upright through the middle of a, far from a's edges and
        // vertices, so only the segment-crossing test can see it.
        let b = [[0.25, 0.25, -1.0], [0.3, 0.2, 1.0], [0.2, 0.3, 1.0]];
        assert!(triangles_within(&T, &b, 0.0));
    }

    #[test]
    fn edge_to_edge_gap_is_measured() {
        // b is a's mirror image across x + y = 1.5, so the closest pair is
        // a's hypotenuse (on x + y = 1) against b's (on x + y = 2).
        let b = T.map(|v| [1.5 - v[1], 1.5 - v[0], v[2]]);
        let gap = 1.0 / 2f64.sqrt();
        assert!(triangles_within(&T, &b, gap + 1e-9));
        assert!(!triangles_within(&T, &b, gap - 1e-9));
    }
}
