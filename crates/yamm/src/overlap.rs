//! Inter-solid mesh overlap detection.
//!
//! DAGMC requires every point in space to belong to exactly one volume. Two
//! solids whose *meshes* interpenetrate break that even when the input CAD was
//! clean: a particle entering the doubly-claimed region is tracked in whichever
//! volume DAGMC thinks it is in, and when it hits the other volume's surface
//! from the inside the sense lookup hands it a volume that does not contain the
//! point. Sometimes that is a lost particle; sometimes it is silently the wrong
//! material, which is worse because nothing reports it.
//!
//! The CAD-level check catches overlapping *input*. This catches overlapping
//! *output*, which is a different failure: faceting two near-tangent curved
//! surfaces can make them cross by up to the chord tolerance even though the
//! solids only touch, and with `imprint=False` nothing has reconciled the
//! surfaces at all.
//!
//! # What is and is not an overlap
//!
//! Contact is not overlap, and a DAGMC assembly is mostly contact - a check
//! that flags touching solids is worse than no check. Two things separate them:
//!
//! * **Shared faces are excluded.** After imprinting, a face shared by two
//!   solids is stored once and both solids reference identical triangles. Those
//!   coincide exactly and would otherwise register as interpenetration. The
//!   caller passes the vertices lying on such faces and they are not tested.
//!   Note they are still part of the *surface* each solid is tested against:
//!   the inside/outside test needs each solid's full closed boundary.
//! * **Depth is reported, not a boolean.** Faceting artefacts penetrate by
//!   about the chord tolerance; a real overlap penetrates by a fraction of the
//!   geometry. Same test, magnitudes apart, so the caller thresholds on a
//!   number rather than this module guessing.
//!
//! # Coverage
//!
//! Two directional passes per pair, each testing one solid's features against
//! the other's closed surface:
//!
//! * vertex containment - catches deep overlap, and full containment via the
//!   opposite direction (if A swallows B, no vertex of A is inside B, but every
//!   vertex of B is inside A);
//! * edge crossing - catches shallow or glancing overlap where no vertex of
//!   either solid falls inside the other;
//! * face-interior sampling - catches the case both of the above miss, where
//!   the intersection curve runs exactly along shared edges so nothing on the
//!   1-skeleton is ever strictly inside.
//!
//! That third probe is not hypothetical padding. Two identical cubes offset
//! along one axis overlap in half their volume, yet every vertex and every edge
//! of each lies exactly on the other's surface, because their cross-sections
//! coincide: the only strictly-interior features are face interiors. An
//! axis-aligned duplicate of a part is an ordinary modelling slip, so a check
//! built only on vertices and edges would miss a 50% overlap.
//!
//! With face interiors included these probe the 0-, 1- and 2-dimensional
//! features of each mesh, which is what makes the pair equivalent to
//! triangle-triangle intersection without a tri-tri kernel: an interpenetration
//! of two closed surfaces puts some feature of one strictly inside the other.

use crate::volume::aabb_bvh::TriangleBvh;
use rayon::prelude::*;

/// A solid's id paired with its full closed boundary triangles.
pub type SolidBoundary = (usize, Vec<[usize; 3]>);

/// One overlapping pair.
#[derive(Debug, Clone)]
pub struct SolidOverlap {
    pub solid_a: usize,
    pub solid_b: usize,
    /// Deepest penetration found, in model units. The number to threshold on:
    /// compare against the meshing tolerance to tell a faceting artefact from a
    /// real overlap.
    pub max_depth: f64,
    /// Vertices of one solid found strictly inside the other.
    pub inside_vertices: usize,
    /// Edges of one solid that cross into the other.
    pub crossing_edges: usize,
    /// Triangles of one solid whose interior lies inside the other. The only
    /// signal when the two share cross-sections, where nothing on the
    /// 1-skeleton is ever strictly inside.
    pub inside_faces: usize,
    /// Boundary area of the doubly-claimed region, estimated by summing the
    /// triangle area each solid contributes inside the other.
    pub lens_area: f64,
    /// Estimated fraction of isotropic ray-fires that traverse the
    /// doubly-claimed region -- see [`find_solid_overlaps`].
    pub exposure: f64,
}

fn tri_area(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct Prepared {
    id: usize,
    bvh: TriangleBvh,
    tris: Vec<[usize; 3]>,
    /// Testable vertices: those used by this solid and not on a shared face.
    verts: Vec<usize>,
    /// Testable edges, deduplicated, both endpoints testable.
    edges: Vec<(usize, usize)>,
    /// One vertex of each connected piece of the boundary (a hollow solid's
    /// outer surface and each of its voids).
    shells: Vec<usize>,
    lo: [f64; 3],
    hi: [f64; 3],
}

/// One vertex of each connected piece (shell) of a triangle mesh.
fn shell_representatives(tris: &[[usize; 3]]) -> Vec<usize> {
    let mut ids: Vec<usize> = tris.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let local = |v: usize| ids.binary_search(&v).expect("a vertex of these triangles");
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for t in tris {
        let a = root(&mut parent, local(t[0]));
        for &v in &t[1..] {
            let b = root(&mut parent, local(v));
            parent[b] = a;
        }
    }
    (0..ids.len())
        .filter(|&i| root(&mut parent, i) == i)
        .map(|i| ids[i])
        .collect()
}

fn aabbs_disjoint(a: &Prepared, b: &Prepared, pad: f64) -> bool {
    (0..3).any(|k| a.hi[k] + pad < b.lo[k] || b.hi[k] + pad < a.lo[k])
}

/// True when `p` lies outside `b`'s bounding box grown by `pad`, so it cannot
/// be inside `b`. Lets the per-feature tests skip, without a ray cast, the
/// features of a large solid that are nowhere near a small one it reaches.
fn outside_box(p: &[f64; 3], b: &Prepared, pad: f64) -> bool {
    (0..3).any(|k| p[k] < b.lo[k] - pad || p[k] > b.hi[k] + pad)
}

/// True when segment `p q` lies wholly to one side of `b`'s bounding box grown
/// by `pad` along some axis, so it cannot cross `b`'s boundary.
fn segment_misses_box(p: &[f64; 3], q: &[f64; 3], b: &Prepared, pad: f64) -> bool {
    (0..3).any(|k| {
        (p[k] < b.lo[k] - pad && q[k] < b.lo[k] - pad)
            || (p[k] > b.hi[k] + pad && q[k] > b.hi[k] + pad)
    })
}

/// How deep inside `bvh` the point is, or `None` if it is outside.
fn depth_inside(bvh: &TriangleBvh, p: &[f64; 3], eps: f64) -> Option<f64> {
    // A point within eps of the surface is contact, whatever parity says: a
    // point exactly on it (a touching contact, or a vertex of a face shared
    // with a third solid) reads as "inside" about half the time. Asking that
    // first is also the cheap question, and on touching solids it settles
    // nearly every feature without the three parity rays.
    if bvh.any_within(p, eps) || !bvh.is_point_inside(p) {
        return None;
    }
    Some(dist(p, &bvh.nearest_point_on_surface(p)))
}

/// Find pairs of solids whose meshes share volume.
///
/// `solids` gives each solid's id and its **full closed boundary** - shared
/// faces included, because the inside/outside test needs a closed surface.
/// `shared_vertices` lists vertices lying on faces shared between solids; those
/// are excluded from being *tested*, since a shared face's triangles coincide
/// exactly and are contact rather than overlap.
///
/// `depth_eps` is the absolute depth below which a penetration is treated as a
/// point sitting on the surface rather than inside it.
///
/// `exposure` estimates the fraction of isotropic ray-fires that traverse the
/// doubly-claimed region. By the Cauchy/Santaló random-chord result, the
/// fraction of random chords of a convex enclosure meeting a convex body inside
/// it is the ratio of their surface areas, so this is the lens boundary area
/// over the model bounding box's area.
///
/// Treat it as an order of magnitude, not a bound. It is wrong in both
/// directions: a non-convex lens has its mean projected area counted with
/// multiplicity, which pushes the estimate up, while any part of the lens
/// boundary that coincides with both surfaces is inside neither solid and is
/// counted by neither pass, which pushes it down - two identical cubes
/// half-overlapped come out about 2x low for exactly that reason. What it is
/// reliably good for is the job it exists to do: separating a faceting sliver
/// at 1e-8 from a real overlap at 1e-2 without anyone having to run a transport
/// check to find out.
///
/// It also estimates *affected* tracks, not lost ones. Some mis-tracked
/// particles are silently given the wrong material rather than being lost,
/// which is why a clean lost-particle count does not certify the geometry.
pub fn find_solid_overlaps(
    vertices: &[[f64; 3]],
    solids: &[SolidBoundary],
    shared_vertices: &[usize],
    depth_eps: f64,
) -> Vec<SolidOverlap> {
    if solids.len() < 2 || vertices.is_empty() {
        return Vec::new();
    }

    let mut shared: Vec<usize> = shared_vertices.to_vec();
    shared.sort_unstable();
    shared.dedup();
    let is_shared = |v: usize| shared.binary_search(&v).is_ok();

    let prepared: Vec<Prepared> = solids
        .par_iter()
        .map(|(id, tris)| {
            let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
                .iter()
                .map(|t| (vertices[t[0]], vertices[t[1]], vertices[t[2]]))
                .collect();
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            let mut verts: Vec<usize> = Vec::new();
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for t in tris {
                for &v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(vertices[v][k]);
                        hi[k] = hi[k].max(vertices[v][k]);
                    }
                    if !is_shared(v) {
                        verts.push(v);
                    }
                }
                for &(u, w) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    if !is_shared(u) && !is_shared(w) {
                        edges.push((u.min(w), u.max(w)));
                    }
                }
            }
            // Sorted rather than hashed: iteration order here reaches the
            // reported depth and counts, so it must not vary run to run.
            verts.sort_unstable();
            verts.dedup();
            edges.sort_unstable();
            edges.dedup();
            Prepared {
                id: *id,
                bvh: TriangleBvh::new(&tri_data),
                tris: tris.clone(),
                verts,
                edges,
                shells: shell_representatives(tris),
                lo,
                hi,
            }
        })
        .collect();

    // Denominator for `exposure`: the model bounding box's surface area.
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in &prepared {
        for k in 0..3 {
            lo[k] = lo[k].min(p.lo[k]);
            hi[k] = hi[k].max(p.hi[k]);
        }
    }
    let (dx, dy, dz) = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    let bbox_area = 2.0 * (dx * dy + dy * dz + dz * dx);

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..prepared.len() {
        for j in (i + 1)..prepared.len() {
            if !aabbs_disjoint(&prepared[i], &prepared[j], depth_eps) {
                pairs.push((i, j));
            }
        }
    }

    let mut found: Vec<SolidOverlap> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = (&prepared[i], &prepared[j]);
            // Surfaces that never come within depth_eps of each other neither
            // touch nor cross, so each connected piece of one solid's boundary
            // lies wholly inside or wholly outside the other, and one vertex
            // of each piece says which. Only a direction with a piece inside
            // needs the per-feature pass. On nested shells this turns every
            // non-adjacent pair from a ray cast per vertex into a few. (One
            // vertex per solid is not enough: a hollow solid's outer surface
            // can be outside the other while its void's surface is inside.)
            let (run_ab, run_ba) = if a.bvh.surfaces_within(&b.bvh, depth_eps) {
                (true, true)
            } else {
                let inside = |x: &Prepared, y: &Prepared| {
                    x.shells
                        .iter()
                        .any(|&v| y.bvh.is_point_inside(&vertices[v]))
                };
                (inside(a, b), inside(b, a))
            };
            if !run_ab && !run_ba {
                return None;
            }
            let none = ((0.0, 0, 0, 0), 0.0);
            let (da, la) = if run_ab {
                directional(a, b, vertices, depth_eps)
            } else {
                none
            };
            let (db, lb) = if run_ba {
                directional(b, a, vertices, depth_eps)
            } else {
                none
            };
            let max_depth = da.0.max(db.0);
            let inside_vertices = da.1 + db.1;
            let crossing_edges = da.2 + db.2;
            let inside_faces = da.3 + db.3;
            if inside_vertices == 0 && crossing_edges == 0 && inside_faces == 0 {
                return None;
            }
            let lens_area = la + lb;
            Some(SolidOverlap {
                solid_a: a.id,
                solid_b: b.id,
                max_depth,
                inside_vertices,
                crossing_edges,
                inside_faces,
                lens_area,
                exposure: if bbox_area > 0.0 {
                    lens_area / bbox_area
                } else {
                    0.0
                },
            })
        })
        .collect();

    // Deterministic order regardless of how rayon scheduled the pairs.
    found.sort_by(|x, y| {
        (x.solid_a, x.solid_b)
            .cmp(&(y.solid_a, y.solid_b))
            .then(y.max_depth.total_cmp(&x.max_depth))
    });
    found
}

/// Volume shared by two closed triangle meshes.
///
/// When the surfaces never come within `tol` of each other and one lies wholly
/// inside the other, the shared volume is all of the inner solid, which the
/// caller knows exactly; that case returns `SharedVolume::Embedded` with the
/// inner solid's index (0 for `a`, 1 for `b`). When they never come close and
/// neither is inside the other, nothing is shared. Otherwise (the surfaces
/// meet, or hollow solids interleave) the shared volume is estimated by
/// sampling `samples` points uniformly in the intersection of the two
/// bounding boxes (deterministic for a given `seed`).
pub fn shared_volume(
    vertices: &[[f64; 3]],
    a: &[[usize; 3]],
    b: &[[usize; 3]],
    tol: f64,
    samples: usize,
    seed: u64,
) -> SharedVolume {
    let bvh = |tris: &[[usize; 3]]| {
        let t: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
            .iter()
            .map(|t| (vertices[t[0]], vertices[t[1]], vertices[t[2]]))
            .collect();
        TriangleBvh::new(&t)
    };
    let (ba, bb) = (bvh(a), bvh(b));
    if !ba.surfaces_within(&bb, tol) {
        // Each connected piece (shell) of either boundary then lies wholly
        // inside or outside the other solid, and one vertex of it says which.
        // `b` lies wholly inside `a` when all of `b`'s shells are inside `a`
        // and none of `a`'s is inside `b` (a void of `a` inside `b` is a hole
        // in what they share). With no shell of either inside the other they
        // are apart; anything else (hollow solids interleaved) is sampled.
        let inside = |shells: &[usize], other: &TriangleBvh| {
            shells
                .iter()
                .filter(|&&v| other.is_point_inside(&vertices[v]))
                .count()
        };
        let (shells_a, shells_b) = (shell_representatives(a), shell_representatives(b));
        let (a_in_b, b_in_a) = (inside(&shells_a, &bb), inside(&shells_b, &ba));
        if !shells_b.is_empty() && b_in_a == shells_b.len() && a_in_b == 0 {
            return SharedVolume::Embedded(1);
        }
        if !shells_a.is_empty() && a_in_b == shells_a.len() && b_in_a == 0 {
            return SharedVolume::Embedded(0);
        }
        if a_in_b == 0 && b_in_a == 0 {
            return SharedVolume::Estimate {
                volume: 0.0,
                std_err: 0.0,
            };
        }
    }
    let bounds = |tris: &[[usize; 3]]| {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for t in tris {
            for &v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(vertices[v][k]);
                    hi[k] = hi[k].max(vertices[v][k]);
                }
            }
        }
        (lo, hi)
    };
    let ((la, ha), (lb, hb)) = (bounds(a), bounds(b));
    let lo = [la[0].max(lb[0]), la[1].max(lb[1]), la[2].max(lb[2])];
    let hi = [ha[0].min(hb[0]), ha[1].min(hb[1]), ha[2].min(hb[2])];
    if (0..3).any(|k| hi[k] <= lo[k]) || samples == 0 {
        return SharedVolume::Estimate {
            volume: 0.0,
            std_err: 0.0,
        };
    }
    let box_volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    const CHUNK: usize = 4096;
    let chunks = samples.div_ceil(CHUNK);
    let hits: usize = (0..chunks)
        .into_par_iter()
        .map(|c| {
            // splitmix64, each chunk starting from its own mixed state so that
            // the chunks' streams do not overlap (seed ^ c * step shifted one
            // chunk's stream by a few draws from another's): deterministic
            // whatever the threads.
            let mut state = {
                let mut z = seed.wrapping_add((c as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            let mut next = || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
            };
            let n = CHUNK.min(samples - c * CHUNK);
            (0..n)
                .filter(|_| {
                    let p = [
                        lo[0] + (hi[0] - lo[0]) * next(),
                        lo[1] + (hi[1] - lo[1]) * next(),
                        lo[2] + (hi[2] - lo[2]) * next(),
                    ];
                    ba.is_point_inside(&p) && bb.is_point_inside(&p)
                })
                .count()
        })
        .sum();
    let p = hits as f64 / samples as f64;
    SharedVolume::Estimate {
        volume: box_volume * p,
        std_err: box_volume * (p * (1.0 - p) / samples as f64).sqrt(),
    }
}

/// Result of [`shared_volume`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SharedVolume {
    /// One solid (0 = `a`, 1 = `b`) lies wholly inside the other.
    Embedded(usize),
    /// Monte Carlo estimate, with its standard error.
    Estimate { volume: f64, std_err: f64 },
}

/// Points where two solids' closed surface meshes come within `tolerance` of
/// each other.
///
/// For every pair of solids whose meshes come that close, up to `samples`
/// mesh vertices of one of them that lie within `tolerance` of the other's
/// surface, as `(solid_a, solid_b, points on solid_a)`. A pair whose meshes
/// come close only between vertices is returned with no points, and so is a
/// pair whose meshes share a vertex: a mesh vertex lies on its solid's CAD
/// surface, so one that belongs to both lies on both and the two touch.
/// Ordered by pair, whatever the thread scheduling.
///
/// The CAD layer uses these to tell solids that touch (each sample lies on
/// the other CAD solid) from solids separated by a real gap that faceting has
/// brought within the tolerance.
#[allow(clippy::type_complexity)]
pub fn contact_samples(
    vertices: &[[f64; 3]],
    solids: &[SolidBoundary],
    tolerance: f64,
    samples: usize,
) -> Vec<(usize, usize, Vec<[f64; 3]>)> {
    struct Solid {
        id: usize,
        bvh: TriangleBvh,
        verts: Vec<usize>,
        lo: [f64; 3],
        hi: [f64; 3],
    }
    let prepared: Vec<Solid> = solids
        .par_iter()
        .map(|(id, tris)| {
            let t: Vec<([f64; 3], [f64; 3], [f64; 3])> = tris
                .iter()
                .map(|t| (vertices[t[0]], vertices[t[1]], vertices[t[2]]))
                .collect();
            let mut verts: Vec<usize> = tris.iter().flatten().copied().collect();
            verts.sort_unstable();
            verts.dedup();
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for &v in &verts {
                for k in 0..3 {
                    lo[k] = lo[k].min(vertices[v][k]);
                    hi[k] = hi[k].max(vertices[v][k]);
                }
            }
            Solid {
                id: *id,
                bvh: TriangleBvh::new(&t),
                verts,
                lo,
                hi,
            }
        })
        .collect();
    let mut pairs = Vec::new();
    for i in 0..prepared.len() {
        for j in (i + 1)..prepared.len() {
            let (a, b) = (&prepared[i], &prepared[j]);
            if (0..3).any(|k| a.hi[k] + tolerance < b.lo[k] || b.hi[k] + tolerance < a.lo[k]) {
                continue;
            }
            pairs.push((i, j));
        }
    }
    let near = |a: &Solid, b: &Solid| -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for &v in &a.verts {
            let p = vertices[v];
            if (0..3).any(|k| p[k] < b.lo[k] - tolerance || p[k] > b.hi[k] + tolerance) {
                continue;
            }
            if dist(&p, &b.bvh.nearest_point_on_surface(&p)) <= tolerance {
                out.push(p);
                if out.len() >= samples {
                    break;
                }
            }
        }
        out
    };
    let mut found: Vec<(usize, usize, Vec<[f64; 3]>)> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = (&prepared[i], &prepared[j]);
            if !a.bvh.surfaces_within(&b.bvh, tolerance) {
                return None;
            }
            let (small, large) = if a.verts.len() <= b.verts.len() {
                (a, b)
            } else {
                (b, a)
            };
            if small
                .verts
                .iter()
                .any(|v| large.verts.binary_search(v).is_ok())
            {
                return Some((a.id, b.id, Vec::new()));
            }
            let from_a = near(a, b);
            if !from_a.is_empty() {
                return Some((a.id, b.id, from_a));
            }
            let from_b = near(b, a);
            if !from_b.is_empty() {
                return Some((b.id, a.id, from_b));
            }
            Some((a.id, b.id, Vec::new()))
        })
        .collect();
    found.sort_by_key(|&(a, b, _)| (a.min(b), a.max(b)));
    found
}

/// Test `a`'s features against `b`'s closed surface.
///
/// Returns `((max_depth, inside_vertices, crossing_edges, inside_faces), lens_area)`.
fn directional(
    a: &Prepared,
    b: &Prepared,
    vertices: &[[f64; 3]],
    eps: f64,
) -> ((f64, usize, usize, usize), f64) {
    // Each pass runs over a's features in parallel (a pair of large touching
    // solids is otherwise one core's work while the rest wait) and collects
    // per-feature results in feature order, then reduces them serially, so
    // the counts, depths and area sum do not depend on thread scheduling.
    let vertex_depths: Vec<Option<f64>> = a
        .verts
        .par_iter()
        .map(|&v| {
            if outside_box(&vertices[v], b, eps) {
                None
            } else {
                depth_inside(&b.bvh, &vertices[v], eps)
            }
        })
        .collect();
    let mut max_depth = 0.0f64;
    let mut inside_vert: Vec<usize> = Vec::new();
    for (&v, d) in a.verts.iter().zip(&vertex_depths) {
        if let Some(d) = *d {
            max_depth = max_depth.max(d);
            inside_vert.push(v);
        }
    }
    let n_inside = inside_vert.len();
    // a.verts is sorted, so inside_vert is too.

    // Shallow overlap can cross without putting any vertex inside (two thin
    // slabs meeting edge-on). `segment_crosses_boundary` finds those, but it
    // also fires on an edge lying in a contact plane, so confirm with sampled
    // depth along the edge -- which also yields a depth for a case the vertex
    // pass reports nothing for.
    let edge_depths: Vec<f64> = a
        .edges
        .par_iter()
        .map(|&(u, w)| {
            let (p, q) = (vertices[u], vertices[w]);
            if segment_misses_box(&p, &q, b, eps) || !b.bvh.segment_crosses_boundary(&p, &q) {
                return 0.0;
            }
            let mut deepest = 0.0f64;
            for s in 1..8 {
                let t = s as f64 / 8.0;
                let m = [
                    p[0] + (q[0] - p[0]) * t,
                    p[1] + (q[1] - p[1]) * t,
                    p[2] + (q[2] - p[2]) * t,
                ];
                if let Some(d) = depth_inside(&b.bvh, &m, eps) {
                    deepest = deepest.max(d);
                }
            }
            deepest
        })
        .collect();
    let mut n_crossing = 0usize;
    for &d in &edge_depths {
        if d > 0.0 {
            n_crossing += 1;
            max_depth = max_depth.max(d);
        }
    }

    // Face interiors. Where two solids share cross-sections the intersection
    // curve runs along shared edges, so nothing on the 1-skeleton is ever
    // strictly inside and the two passes above both report nothing; the
    // centroid of a face lying inside the other solid still is. Also credits
    // the lens area, which the vertex pass alone would score as zero.
    let faces: Vec<(Option<f64>, f64)> = a
        .tris
        .par_iter()
        .map(|t| {
            let (p0, p1, p2) = (vertices[t[0]], vertices[t[1]], vertices[t[2]]);
            let k = t
                .iter()
                .filter(|v| inside_vert.binary_search(v).is_ok())
                .count();
            let centroid = [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ];
            let centre_in = if outside_box(&centroid, b, eps) {
                None
            } else {
                depth_inside(&b.bvh, &centroid, eps)
            };
            // A triangle is credited whole when its centre and all corners
            // are inside, and by the fraction of corners inside otherwise --
            // it is then clipped by the intersection curve. A triangle whose
            // corners all sit exactly on the surface but whose interior is
            // inside counts whole: that is the coincident-cross-section case,
            // fully inside despite scoring zero corners.
            let frac = if k == 3 || centre_in.is_some() {
                1.0
            } else {
                k as f64 / 3.0
            };
            let area = if frac > 0.0 {
                tri_area(&p0, &p1, &p2) * frac
            } else {
                0.0
            };
            (centre_in, area)
        })
        .collect();
    let mut lens_area = 0.0f64;
    let mut n_inside_faces = 0usize;
    for &(centre_in, area) in &faces {
        if let Some(d) = centre_in {
            max_depth = max_depth.max(d);
            n_inside_faces += 1;
        }
        lens_area += area;
    }
    ((max_depth, n_inside, n_crossing, n_inside_faces), lens_area)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box as a closed triangle mesh, appended to `verts`.
    fn push_box(verts: &mut Vec<[f64; 3]>, lo: [f64; 3], hi: [f64; 3]) -> Vec<[usize; 3]> {
        let o = verts.len();
        for &(x, y, z) in &[
            (lo[0], lo[1], lo[2]),
            (hi[0], lo[1], lo[2]),
            (hi[0], hi[1], lo[2]),
            (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]),
            (hi[0], lo[1], hi[2]),
            (hi[0], hi[1], hi[2]),
            (lo[0], hi[1], hi[2]),
        ] {
            verts.push([x, y, z]);
        }
        let f = |a: usize, b: usize, c: usize| [o + a, o + b, o + c];
        vec![
            f(0, 2, 1),
            f(0, 3, 2),
            f(4, 5, 6),
            f(4, 6, 7),
            f(0, 1, 5),
            f(0, 5, 4),
            f(1, 2, 6),
            f(1, 6, 5),
            f(2, 3, 7),
            f(2, 7, 6),
            f(3, 0, 4),
            f(3, 4, 7),
        ]
    }

    fn two_boxes(offset: f64) -> (Vec<[f64; 3]>, Vec<SolidBoundary>) {
        let mut v = Vec::new();
        let a = push_box(&mut v, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = push_box(&mut v, [offset, 0.0, 0.0], [offset + 10.0, 10.0, 10.0]);
        (v, vec![(1, a), (2, b)])
    }

    #[test]
    fn overlapping_boxes_are_found_with_depth() {
        let (v, s) = two_boxes(5.0);
        let out = find_solid_overlaps(&v, &s, &[], 1e-9);
        assert_eq!(out.len(), 1, "expected one overlapping pair, got {out:?}");
        assert_eq!((out[0].solid_a, out[0].solid_b), (1, 2));
        // Deepest interior point of the lens is 5 from the nearest wall in x,
        // but only 5 from the y/z walls too -- the nearest surface bounds it.
        assert!(
            out[0].max_depth > 1.0,
            "depth should reflect a real overlap, got {}",
            out[0].max_depth
        );
        // Deliberately NOT asserting inside_vertices > 0: these two cubes share
        // y/z extents, so every vertex and edge of each lies exactly on the
        // other's surface and only the face interiors are strictly inside. That
        // degeneracy is why the face probe exists, and pinning it here stops
        // anyone "simplifying" the probe away.
        assert_eq!(
            out[0].inside_vertices, 0,
            "coincident cross-sections put no vertex strictly inside"
        );
        assert!(
            out[0].inside_faces > 0,
            "the overlap must be found via face interiors, got {:?}",
            out[0]
        );
    }

    #[test]
    fn touching_boxes_are_not_an_overlap() {
        let (v, s) = two_boxes(10.0);
        assert!(
            find_solid_overlaps(&v, &s, &[], 1e-9).is_empty(),
            "face-to-face contact must not report an overlap"
        );
    }

    #[test]
    fn disjoint_boxes_are_not_an_overlap() {
        let (v, s) = two_boxes(25.0);
        assert!(find_solid_overlaps(&v, &s, &[], 1e-9).is_empty());
    }

    #[test]
    fn full_containment_is_found() {
        // No vertex of the outer box is inside the inner one; only the opposite
        // direction sees it, which is why both are tested.
        let mut v = Vec::new();
        let outer = push_box(&mut v, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let inner = push_box(&mut v, [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]);
        let out = find_solid_overlaps(&v, &[(1, outer), (2, inner)], &[], 1e-9);
        assert_eq!(out.len(), 1, "containment must be reported");
        assert!(out[0].inside_vertices >= 8, "{:?}", out[0]);
    }

    #[test]
    fn shared_vertices_are_excluded() {
        // Two boxes meeting at x=10 whose contact vertices are declared shared:
        // still not an overlap, and excluding them must not break the surfaces
        // they are tested against.
        let (v, s) = two_boxes(10.0);
        let shared: Vec<usize> = (0..v.len())
            .filter(|&i| (v[i][0] - 10.0).abs() < 1e-12)
            .collect();
        assert!(!shared.is_empty());
        assert!(find_solid_overlaps(&v, &s, &shared, 1e-9).is_empty());
    }

    #[test]
    fn depth_scales_with_the_overlap() {
        let shallow = find_solid_overlaps(&two_boxes(9.5).0, &two_boxes(9.5).1, &[], 1e-9);
        let deep = find_solid_overlaps(&two_boxes(5.0).0, &two_boxes(5.0).1, &[], 1e-9);
        assert_eq!(shallow.len(), 1);
        assert_eq!(deep.len(), 1);
        assert!(
            shallow[0].max_depth < deep[0].max_depth,
            "a 0.5 overlap must read shallower than a 5.0 one: {} vs {}",
            shallow[0].max_depth,
            deep[0].max_depth
        );
        assert!(
            shallow[0].exposure < deep[0].exposure,
            "exposure must scale with the overlap too"
        );
    }

    #[test]
    fn results_are_deterministic() {
        let (v, s) = two_boxes(5.0);
        let a = find_solid_overlaps(&v, &s, &[], 1e-9);
        let b = find_solid_overlaps(&v, &s, &[], 1e-9);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].max_depth, b[0].max_depth);
        assert_eq!(a[0].inside_vertices, b[0].inside_vertices);
        assert_eq!(a[0].lens_area, b[0].lens_area);
    }

    #[test]
    fn shared_volume_of_two_offset_boxes() {
        // Two 10 cm cubes offset by 5 along x share a 5 x 10 x 10 block.
        let (v, s) = two_boxes(5.0);
        match shared_volume(&v, &s[0].1, &s[1].1, 1e-9, 200_000, 7) {
            SharedVolume::Estimate { volume, std_err } => {
                assert!(
                    (volume - 500.0).abs() < 4.0 * std_err + 1.0,
                    "{volume} +- {std_err}"
                );
            }
            other => panic!("expected an estimate, got {other:?}"),
        }
    }

    #[test]
    fn shared_volume_of_interleaved_hollow_solids_is_sampled() {
        // a: [-5,5]^3 with a [-3,3]^3 void, b: [-6,6]^3 with a [-4,4]^3 void.
        // No surface of one comes near the other's, neither lies inside the
        // other, and they share the shell between 4 and 5: 1000 - 512 = 488.
        // Whichever shell of each is listed first.
        for (a_void_first, b_void_first) in [(false, false), (true, false), (false, true)] {
            let mut v = Vec::new();
            let hollow = |v: &mut Vec<[f64; 3]>, outer: f64, inner: f64, void_first: bool| {
                let (o, i) = if void_first {
                    let i = push_box(v, [-inner; 3], [inner; 3]);
                    (push_box(v, [-outer; 3], [outer; 3]), i)
                } else {
                    let o = push_box(v, [-outer; 3], [outer; 3]);
                    (o, push_box(v, [-inner; 3], [inner; 3]))
                };
                if void_first {
                    [i, o].concat()
                } else {
                    [o, i].concat()
                }
            };
            let a = hollow(&mut v, 5.0, 3.0, a_void_first);
            let b = hollow(&mut v, 6.0, 4.0, b_void_first);
            for (x, y) in [(&a, &b), (&b, &a)] {
                match shared_volume(&v, x, y, 1e-6, 400_000, 3) {
                    SharedVolume::Estimate { volume, std_err } => assert!(
                        (volume - 488.0).abs() < 5.0 * std_err,
                        "{volume} +- {std_err}"
                    ),
                    other => panic!("expected an estimate, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn shared_volume_errors_are_honest_across_seeds() {
        // The interleaved hollow cubes share 488 of the 1000 cm^3 box sampled.
        // Each seed's estimate must sit within its own stated error: chunks
        // drawing overlapping streams used to understate it several times over.
        let mut v = Vec::new();
        let a = [
            push_box(&mut v, [-5.0; 3], [5.0; 3]),
            push_box(&mut v, [-3.0; 3], [3.0; 3]),
        ]
        .concat();
        let b = [
            push_box(&mut v, [-6.0; 3], [6.0; 3]),
            push_box(&mut v, [-4.0; 3], [4.0; 3]),
        ]
        .concat();
        for seed in 1..=6 {
            match shared_volume(&v, &a, &b, 1e-6, 1_000_000, seed) {
                SharedVolume::Estimate { volume, std_err } => assert!(
                    (volume - 488.0).abs() < 4.5 * std_err,
                    "seed {seed}: {volume} +- {std_err}"
                ),
                other => panic!("expected an estimate, got {other:?}"),
            }
        }
    }

    #[test]
    fn shared_volume_of_an_embedded_box_is_the_whole_box() {
        let mut v = Vec::new();
        let outer = push_box(&mut v, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let inner = push_box(&mut v, [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]);
        assert_eq!(
            shared_volume(&v, &outer, &inner, 1e-9, 1000, 7),
            SharedVolume::Embedded(1)
        );
        assert_eq!(
            shared_volume(&v, &inner, &outer, 1e-9, 1000, 7),
            SharedVolume::Embedded(0)
        );
    }

    #[test]
    fn contact_samples_find_touching_and_near_pairs_only() {
        // Face to face: samples on the shared plane. 0.05 apart within a 0.1
        // tolerance: samples too (the caller decides whether the CAD
        // touches). 5 apart: no pair.
        for (offset, expect) in [(10.0, true), (10.05, true), (15.0, false)] {
            let (v, s) = two_boxes(offset);
            let found = contact_samples(&v, &s, 0.1, 4);
            assert_eq!(!found.is_empty(), expect, "offset {offset}: {found:?}");
            if let Some((_, _, points)) = found.first() {
                assert!(!points.is_empty() && points.len() <= 4);
            }
        }
    }

    #[test]
    fn contact_samples_need_no_points_for_solids_sharing_a_vertex() {
        // The second box reuses the first one's corner at (10, 0, 0), as
        // welding does where two meshes meet: the pair is listed, touching,
        // with no points to check against the CAD.
        let (v, mut s) = two_boxes(10.0);
        let shared = s[0]
            .1
            .iter()
            .flatten()
            .copied()
            .find(|&i| v[i] == [10.0, 0.0, 0.0]);
        let own = s[1]
            .1
            .iter()
            .flatten()
            .copied()
            .find(|&i| v[i] == [10.0, 0.0, 0.0]);
        let (shared, own) = (shared.unwrap(), own.unwrap());
        for tri in s[1].1.iter_mut() {
            for i in tri.iter_mut() {
                if *i == own {
                    *i = shared;
                }
            }
        }
        let found = contact_samples(&v, &s, 0.1, 4);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].2.is_empty(), "{found:?}");
    }

    #[test]
    fn an_overlap_of_hollow_solids_whose_surfaces_stay_apart_is_found() {
        // a: [0,10]^3 with a [2,8]^3 void; b: [1,9]^3 with a [3,7]^3 void,
        // its void listed first. No surface of one comes near the other's,
        // yet the bodies overlap (for 1 < x < 2, say): a's void surface is
        // inside b, although a's outer surface and b's void surface are not.
        let mut v = Vec::new();
        let mut a = push_box(&mut v, [0.0; 3], [10.0; 3]);
        a.extend(push_box(&mut v, [2.0; 3], [8.0; 3]));
        let mut b = push_box(&mut v, [3.0; 3], [7.0; 3]);
        b.extend(push_box(&mut v, [1.0; 3], [9.0; 3]));
        let out = find_solid_overlaps(&v, &[(1, a), (2, b)], &[], 1e-6);
        assert_eq!(out.len(), 1, "{out:?}");
    }

    #[test]
    fn shell_representatives_find_each_connected_piece() {
        let mut v = Vec::new();
        let mut t = push_box(&mut v, [0.0; 3], [10.0; 3]);
        t.extend(push_box(&mut v, [2.0; 3], [8.0; 3]));
        assert_eq!(shell_representatives(&t).len(), 2);
        assert_eq!(shell_representatives(&t[..12]).len(), 1);
    }

    #[test]
    fn single_solid_has_nothing_to_overlap() {
        let mut v = Vec::new();
        let a = push_box(&mut v, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(find_solid_overlaps(&v, &[(1, a)], &[], 1e-9).is_empty());
    }
}
