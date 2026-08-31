//! Convex-cell clipping, for the exact cut-cell fill (issue #136).
//!
//! THE PROBLEM. `mesh_volume_delaunay`'s centroid filter keeps or drops each tet
//! WHOLESALE, so a tet the boundary passes through has its volume counted
//! entirely or not at all. Measured over ten zoo solids (#147), that is an exact
//! predictor of an inexact fill: zero cut tets iff the fill error is at machine
//! precision. BlanketModule carries 11208 of its 12576 units of enclosed volume
//! on cut tets and is 5.4% short; AnnularSector is 5.0% cut and 0.43% short.
//!
//! THE FIX. Cut those tets and keep only the interior pieces. The Delaunay
//! tetrahedralises the convex hull of all points and the boundary lies inside
//! that hull, so every interior point lies in exactly one tet and therefore
//! `enclosed volume == Σ vol(tet ∩ interior)` identically - the fill becomes
//! exact by construction rather than by the cancellation it currently relies on.
//!
//! WHY CUTTING BY PLANES IS ENOUGH. Within one tet, take the supporting planes of
//! every boundary triangle overlapping it. The boundary inside the tet is a
//! subset of those planes, so the interiors of the resulting sub-cells avoid the
//! boundary entirely - each sub-cell is wholly inside or wholly outside, and one
//! centroid test classifies it. This holds even though the surface is bounded by
//! triangle EDGES rather than whole planes: the neighbouring triangles supplying
//! those edges also overlap the tet, so their planes are in the set.
//!
//! Unlike the conforming carve this needs no Delaunay property and no
//! convergence, which is exactly why the carve cannot get here (it diverges,
//! #139).
//!
//! Plane counts per cut tet were measured before any of this was written (#136
//! step 0): k <= 3 for BlanketModule, NestedCylinder and ToroidalSector, with
//! AnnularSector the outlier at k <= 14 because its boundary carries sub-degree
//! sliver triangles.

use super::predicates3d::{orient_3d, tet_volume};

/// Vertices snapped onto a cut plane. Diagnostic for #151.
pub(super) static SNAPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Snap tolerance as a fraction of the cell's own diagonal: a vertex within this
/// distance of a cut plane is treated as lying on it. Overridable with
/// YAMM_CLIP_SNAP (0 disables).
///
/// CALIBRATED by sweep, not chosen by taste - the same discipline that caught a
/// bad work bound in #142 and a REGION_CAP that could never bind in #144:
///
///   snap        AnnularSector min-vol   tets    fill err
///   0 (off)     3.860e-09               12940   1.310e-14
///   1e-12       5.731e-08               11796   1.377e-14
///   1e-9        5.731e-08               11796   1.377e-14
///   1e-6        5.731e-08               11796   1.377e-14
///   1e-4        5.731e-08               11796   1.377e-14
///
/// 5.731e-08 is exactly AnnularSector's UNCLIPPED minimum tet volume, so the snap
/// restores the size tail to baseline while also emitting 1144 fewer tets, and the
/// fill error barely moves. Every value from 1e-12 to 1e-4 gives the identical
/// result, which says the vertices being snapped sit at essentially exact-zero
/// orient values rather than spread across a range - so this is not a trade-off
/// dial. 1e-9 is the middle of that plateau: ~1000x margin over what the zoo
/// needs, while still a vanishingly small distance on other geometry.
///
/// It does nothing for BlanketModule, whose result is unchanged at every value -
/// none of its 45086 crossings lands near a vertex (closest 6.632e-3 of an edge),
/// so its thin cells come from cuts near a tet EDGE and are a separate matter
/// (#151).
fn snap_rel() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("YAMM_CLIP_SNAP")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-9)
    })
}

/// Smallest dihedral angle of a tet, in radians. Used to pick a fan apex, so only
/// the ordering matters, but a real angle keeps the score interpretable against
/// the numbers in #151.
fn min_dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let cr = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let norm = |u: [f64; 3]| dot(u, u).sqrt();
    let p = [a, b, c, d];
    // Outward-ish normals of the four faces; sign is irrelevant since the angle is
    // taken as pi minus the normal-to-normal angle and then folded to (0, pi/2].
    let faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut n = [[0.0f64; 3]; 4];
    for (k, f) in faces.iter().enumerate() {
        n[k] = cr(sub(p[f[1]], p[f[0]]), sub(p[f[2]], p[f[0]]));
        let l = norm(n[k]);
        if l == 0.0 {
            return 0.0; // degenerate face: worst possible score
        }
        for x in 0..3 {
            n[k][x] /= l;
        }
    }
    let mut worst = std::f64::consts::PI;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let c = dot(n[i], n[j]).clamp(-1.0, 1.0);
            let ang = std::f64::consts::PI - c.acos();
            let ang = ang.min(std::f64::consts::PI - ang);
            worst = worst.min(ang);
        }
    }
    worst
}

/// Smallest `min(t, 1-t)` seen over all edge crossings, scaled by 1e9 and stored
/// as an integer so it can live in an atomic. Diagnostic for #151: a crossing that
/// lands close to an endpoint is what produces a sliver cell, so this distribution
/// is what a snap tolerance has to be chosen against.
pub(super) static CROSSING_MIN_T_E9: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);
/// Crossings whose `min(t, 1-t)` fell below 1e-3, 1e-6 and 1e-9 respectively.
pub(super) static CROSSING_NEAR: [std::sync::atomic::AtomicUsize; 3] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];
pub(super) static CROSSING_TOTAL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// A convex polyhedron: vertices plus faces as ordered index loops.
///
/// Face loops carry no orientation guarantee. Nothing here needs one: volume and
/// tetrahedralisation both fan from the cell centroid, which is interior for a
/// convex cell, and take absolute volumes. Avoiding an orientation invariant
/// removes a whole class of sign bugs from the clipping, and the caller
/// (`mesh_volume`) enforces positive orientation on the tets we emit anyway.
#[derive(Clone, Debug)]
pub(super) struct ConvexCell {
    pub verts: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
}

impl ConvexCell {
    /// The four faces of a tet. Loop order is irrelevant (see the type docs).
    pub fn from_tet(p: &[[f64; 3]; 4]) -> Self {
        ConvexCell {
            verts: p.to_vec(),
            faces: vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        }
    }

    pub fn centroid(&self) -> [f64; 3] {
        let n = self.verts.len() as f64;
        let mut c = [0.0; 3];
        for v in &self.verts {
            for k in 0..3 {
                c[k] += v[k];
            }
        }
        for k in 0..3 {
            c[k] /= n;
        }
        c
    }

    /// Volume, by fanning every face into triangles from the cell centroid.
    /// Absolute per sub-tet, which is valid because the centroid of a convex cell
    /// is interior, so the sub-tets partition it.
    pub fn volume(&self) -> f64 {
        let c = self.centroid();
        let mut vol = 0.0;
        for f in &self.faces {
            if f.len() < 3 {
                continue;
            }
            for i in 1..f.len() - 1 {
                vol +=
                    tet_volume(c, self.verts[f[0]], self.verts[f[i]], self.verts[f[i + 1]]).abs();
            }
        }
        vol
    }

    /// Append this cell's tetrahedralisation. `voff` is the index this cell's
    /// first vertex will occupy in the caller's list.
    ///
    /// Fans from an existing VERTEX rather than from the centroid: for every face
    /// not containing the apex, triangulate the face and join it to the apex.
    /// That partitions a convex cell without inventing a vertex, and it emits far
    /// fewer tets - a centroid fan turns an unsplit tet into 4 tets, a vertex fan
    /// into 1. Measured over the cut tets of the zoo, the centroid fan cost 10x
    /// (BlanketModule) to 105x (AnnularSector) the tets it replaced.
    ///
    /// The APEX is chosen to maximise the worst aspect ratio of the emitted tets
    /// rather than fixed at vertex 0. Any vertex of a convex cell partitions it,
    /// so this costs nothing in exactness or element count - the fan from a good
    /// apex has the same number of tets as the fan from a bad one - and it is the
    /// only lever on #151 that touches BlanketModule, whose crossings are all
    /// healthy (closest to a vertex: 6.632e-3 of an edge) yet whose smallest tet
    /// volume still fell to 7.8e-9. A thin tet there comes from fanning a face
    /// the apex is nearly coplanar with, which choosing the apex avoids.
    ///
    /// Emitted tets are POSITIVELY oriented, which `mesh_volume` requires at its
    /// emission boundary (issue #108). Zero-volume fan triangles are skipped
    /// rather than emitted for `drop_degenerate_tets` to sweep up later.
    pub fn tetrahedralise(&self, voff: usize, out: &mut Vec<[usize; 4]>) {
        let apex = self.best_apex();
        self.fan_from(apex, voff, out);
    }

    /// Vertex whose fan maximises the worst MINIMUM DIHEDRAL ANGLE among the tets
    /// it emits.
    ///
    /// Scored on the dihedral angle directly rather than on a volume proxy. A
    /// normalised-volume score was tried first and gave a mixed result - it
    /// restored BlanketModule (0.0123 -> 0.0602 degrees, against 0.0626 unclipped)
    /// but made AnnularSector worse (0.0049 -> 0.0001), because on the near-flat
    /// cells its degenerate boundary produces, volume-over-edge-cubed stops
    /// tracking angle. Optimising the measured quantity avoids that whole class of
    /// proxy mismatch.
    fn best_apex(&self) -> usize {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for apex in 0..self.verts.len() {
            let pa = self.verts[apex];
            let mut worst = f64::INFINITY;
            let mut any = false;
            for f in &self.faces {
                if f.len() < 3 || f.contains(&apex) {
                    continue;
                }
                for i in 1..f.len() - 1 {
                    let (pb, pc, pd) = (self.verts[f[0]], self.verts[f[i]], self.verts[f[i + 1]]);
                    if tet_volume(pa, pb, pc, pd) == 0.0 {
                        continue;
                    }
                    any = true;
                    worst = worst.min(min_dihedral(pa, pb, pc, pd));
                }
            }
            if any && worst > best.0 {
                best = (worst, apex);
            }
        }
        best.1
    }

    fn fan_from(&self, apex: usize, voff: usize, out: &mut Vec<[usize; 4]>) {
        let pa = self.verts[apex];
        for f in &self.faces {
            if f.len() < 3 || f.contains(&apex) {
                continue;
            }
            for i in 1..f.len() - 1 {
                let (b, c, d) = (f[0], f[i], f[i + 1]);
                let (pb, pc, pd) = (self.verts[b], self.verts[c], self.verts[d]);
                if tet_volume(pa, pb, pc, pd) == 0.0 {
                    continue;
                }
                let mut t = [voff + apex, voff + b, voff + c, voff + d];
                if orient_3d(pa, pb, pc, pd) < 0.0 {
                    t.swap(2, 3);
                }
                out.push(t);
            }
        }
    }

    /// Split by the plane through `tri`, returning `(positive side, negative
    /// side)` where the sign is that of `orient_3d(tri, point)`. A side is `None`
    /// when the cell does not reach it.
    ///
    /// Sides are classified with the EXACT `orient_3d` against the triangle
    /// rather than a float plane equation, so "on the plane" is exact and the
    /// coplanar-boundary case cannot be misjudged. The crossing parameter comes
    /// from the same values: `orient_3d` is proportional to signed distance for a
    /// fixed plane, so `t = o_a / (o_a - o_b)` is the true crossing fraction.
    /// Bounding-box diagonal, the cell's own length scale. Used to turn the snap
    /// tolerance into something scale-free.
    fn diagonal(&self) -> f64 {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for v in &self.verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
    }

    pub fn split_by_plane(&self, tri: &[[f64; 3]; 3]) -> (Option<ConvexCell>, Option<ConvexCell>) {
        // SNAP (issue #151, size half). A vertex sitting a hair off the cut plane
        // produces a crossing a hair away from it, and that is a sliver cell.
        // Measured over the clip's crossings: AnnularSector puts 6042 of 24796
        // within 1e-3 of an edge endpoint, and the count barely moves down to 1e-9
        // (6002) - so those are not spread out, they are vertices lying within
        // floating-point noise of the plane while the classification uses a strict
        // > 0 / < 0.
        //
        // Treating such a vertex as ON the plane is what removes the sliver, and it
        // is cheap in a way that is easy to miss: both halves then SHARE that
        // vertex, so the two still partition the cell exactly and volume
        // conservation - and therefore `split_conserving` - is untouched. What the
        // snap costs is fidelity of the cut to the true boundary, bounded by
        // roughly (face area x snap distance), which is why the tolerance is
        // relative to the cell's own size and small.
        //
        // orient_3d is 2 * area(tri) * signed distance, so scaling the threshold by
        // twice the triangle area converts a DISTANCE tolerance into orient units.
        let two_area = {
            let u = [
                tri[1][0] - tri[0][0],
                tri[1][1] - tri[0][1],
                tri[1][2] - tri[0][2],
            ];
            let w = [
                tri[2][0] - tri[0][0],
                tri[2][1] - tri[0][1],
                tri[2][2] - tri[0][2],
            ];
            let c = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
        };
        let snap = snap_rel() * two_area * self.diagonal();
        let o: Vec<f64> = self
            .verts
            .iter()
            .map(|v| {
                let x = orient_3d(tri[0], tri[1], tri[2], *v);
                if snap > 0.0 && x.abs() <= snap {
                    SNAPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    0.0
                } else {
                    x
                }
            })
            .collect();
        let any_pos = o.iter().any(|&x| x > 0.0);
        let any_neg = o.iter().any(|&x| x < 0.0);
        match (any_pos, any_neg) {
            (false, false) => (None, None), // entire cell lies in the plane
            (true, false) => (Some(self.clone()), None),
            (false, true) => (None, Some(self.clone())),
            (true, true) => {
                let pos = self.clip_side(&o, true);
                let neg = self.clip_side(&o, false);
                (pos, neg)
            }
        }
    }

    /// Keep the half of the cell on one side of the plane. `keep_pos` selects
    /// which. Vertices exactly ON the plane are kept by BOTH sides, which is what
    /// makes the two halves share the cut face rather than leaving a gap.
    fn clip_side(&self, o: &[f64], keep_pos: bool) -> Option<ConvexCell> {
        let keep = |i: usize| -> bool {
            if keep_pos {
                o[i] >= 0.0
            } else {
                o[i] <= 0.0
            }
        };
        let mut verts: Vec<[f64; 3]> = Vec::new();
        let mut remap: Vec<Option<usize>> = vec![None; self.verts.len()];
        // Crossing points are shared between the two faces meeting on an edge, so
        // they are keyed by the edge and reused - otherwise the cap polygon has
        // duplicate corners and the cell is not closed.
        let mut on_edge: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let push_orig = |i: usize, verts: &mut Vec<[f64; 3]>, remap: &mut Vec<Option<usize>>| {
            if let Some(x) = remap[i] {
                return x;
            }
            let x = verts.len();
            verts.push(self.verts[i]);
            remap[i] = Some(x);
            x
        };

        let mut faces: Vec<Vec<usize>> = Vec::new();
        // Vertices lying in the cut plane, which form the new cap face.
        let mut cap: Vec<usize> = Vec::new();
        let note_cap = |x: usize, cap: &mut Vec<usize>| {
            if !cap.contains(&x) {
                cap.push(x);
            }
        };

        for f in &self.faces {
            let mut loop_out: Vec<usize> = Vec::new();
            for k in 0..f.len() {
                let (i, j) = (f[k], f[(k + 1) % f.len()]);
                if keep(i) {
                    let x = push_orig(i, &mut verts, &mut remap);
                    loop_out.push(x);
                    if o[i] == 0.0 {
                        note_cap(x, &mut cap);
                    }
                }
                // Strict sign change means the edge crosses; an endpoint exactly
                // on the plane is handled by the `o[i] == 0.0` case above.
                let crosses = (o[i] > 0.0 && o[j] < 0.0) || (o[i] < 0.0 && o[j] > 0.0);
                if crosses {
                    let key = (i.min(j), i.max(j));
                    let x = if let Some(&x) = on_edge.get(&key) {
                        x
                    } else {
                        let t = o[i] / (o[i] - o[j]);
                        {
                            use std::sync::atomic::Ordering::Relaxed;
                            let m = t.min(1.0 - t).max(0.0);
                            CROSSING_TOTAL.fetch_add(1, Relaxed);
                            CROSSING_MIN_T_E9.fetch_min((m * 1e9) as u64, Relaxed);
                            for (k, thr) in [1e-3, 1e-6, 1e-9].iter().enumerate() {
                                if m < *thr {
                                    CROSSING_NEAR[k].fetch_add(1, Relaxed);
                                }
                            }
                        }
                        let (a, b) = (self.verts[i], self.verts[j]);
                        let p = [
                            a[0] + t * (b[0] - a[0]),
                            a[1] + t * (b[1] - a[1]),
                            a[2] + t * (b[2] - a[2]),
                        ];
                        let x = verts.len();
                        verts.push(p);
                        on_edge.insert(key, x);
                        x
                    };
                    loop_out.push(x);
                    note_cap(x, &mut cap);
                }
            }
            loop_out.dedup();
            if loop_out.len() > 1 && loop_out.first() == loop_out.last() {
                loop_out.pop();
            }
            if loop_out.len() >= 3 {
                faces.push(loop_out);
            }
        }

        if cap.len() >= 3 {
            faces.push(order_planar_loop(&verts, &cap));
        }
        if verts.len() < 4 || faces.len() < 4 {
            return None; // nothing of substance on this side
        }
        Some(ConvexCell { verts, faces })
    }
}

/// Order coplanar points into a convex loop, by angle about their centroid in a
/// basis on their own plane. Valid because the points are the corners of a convex
/// cross-section of a convex cell.
fn order_planar_loop(verts: &[[f64; 3]], idx: &[usize]) -> Vec<usize> {
    let n = idx.len() as f64;
    let mut c = [0.0; 3];
    for &i in idx {
        for k in 0..3 {
            c[k] += verts[i][k];
        }
    }
    for k in 0..3 {
        c[k] /= n;
    }
    // Plane basis: first spoke, then a normal from the most independent second
    // spoke, then the in-plane perpendicular.
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let e1 = sub(verts[idx[0]], c);
    let mut nrm = [0.0; 3];
    for &i in idx.iter().skip(1) {
        let cand = cross(e1, sub(verts[i], c));
        if dot(cand, cand) > dot(nrm, nrm) {
            nrm = cand;
        }
    }
    let e2 = cross(nrm, e1);
    let mut with_angle: Vec<(f64, usize)> = idx
        .iter()
        .map(|&i| {
            let r = sub(verts[i], c);
            (dot(r, e2).atan2(dot(r, e1)), i)
        })
        .collect();
    with_angle.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    with_angle.into_iter().map(|(_, i)| i).collect()
}

/// Split `cell` by `plane`, but only ACCEPT the split if it conserves volume.
///
/// The splitter is exact on well-behaved geometry (measured: 6.674e-14 worst
/// relative error over every cut tet of ToroidalSector) but not on degenerate
/// input: on AnnularSector, whose boundary carries sub-degree sliver triangles,
/// the worst cell lost 4.851e-2 of its volume. Near-parallel cut planes there
/// produce cross-sections thin enough that the cap-polygon ordering and the
/// centroid fan stop being reliable.
///
/// Rather than let that leak into the mesh silently - which is the exact failure
/// mode of the centroid filter this work replaces - a non-conserving split is
/// REJECTED and the cell is kept whole. The caller then classifies it by centroid
/// as today, so such a cell is no worse than the status quo, and the count of
/// rejections is reported so the loss is visible instead of hidden.
///
/// Returns `(cells, rejected)`.
pub(super) fn split_conserving(
    cell: &ConvexCell,
    plane: &[[f64; 3]; 3],
    rel_tol: f64,
) -> (Vec<ConvexCell>, bool) {
    let v0 = cell.volume();
    let (a, b) = cell.split_by_plane(plane);
    if a.is_none() && b.is_none() {
        return (vec![cell.clone()], true);
    }
    let va = a.as_ref().map(|c| c.volume()).unwrap_or(0.0);
    let vb = b.as_ref().map(|c| c.volume()).unwrap_or(0.0);
    if v0 > 0.0 && ((va + vb) - v0).abs() > rel_tol * v0 {
        return (vec![cell.clone()], true);
    }
    let mut out = Vec::with_capacity(2);
    out.extend(a);
    out.extend(b);
    if out.is_empty() {
        return (vec![cell.clone()], true);
    }
    (out, false)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_conserving_split_is_rejected_not_absorbed() {
        // The guard must return the cell intact rather than a lossy pair, so a
        // failure degrades to the status quo instead of quietly losing volume.
        let c = ConvexCell::from_tet(&UNIT_TET);
        let v0 = c.volume();
        let (cells, rejected) = split_conserving(&c, &horizontal_plane(0.4), 1e-9);
        assert!(!rejected, "a clean mid-tet cut must be accepted");
        let total: f64 = cells.iter().map(|x| x.volume()).sum();
        assert!((total - v0).abs() < 1e-12 * v0);

        // An impossibly tight tolerance forces rejection; volume must survive.
        let (cells, rejected) = split_conserving(&c, &horizontal_plane(0.4), 0.0);
        assert!(rejected, "zero tolerance must reject");
        assert_eq!(cells.len(), 1);
        assert!((cells[0].volume() - v0).abs() < 1e-15);
    }

    const UNIT_TET: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    fn horizontal_plane(z: f64) -> [[f64; 3]; 3] {
        [[-9.0, -9.0, z], [9.0, -9.0, z], [-9.0, 9.0, z]]
    }

    #[test]
    fn tet_cell_volume_matches_the_tet() {
        let c = ConvexCell::from_tet(&UNIT_TET);
        // 1/6 for the corner tet.
        assert!((c.volume() - 1.0 / 6.0).abs() < 1e-15, "{}", c.volume());
    }

    #[test]
    fn split_conserves_volume() {
        // The invariant that makes the clip trustworthy: cutting moves volume
        // between the two sides but never creates or destroys any.
        let c = ConvexCell::from_tet(&UNIT_TET);
        let v0 = c.volume();
        for z in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let (pos, neg) = c.split_by_plane(&horizontal_plane(z));
            let vp = pos.as_ref().map(|x| x.volume()).unwrap_or(0.0);
            let vn = neg.as_ref().map(|x| x.volume()).unwrap_or(0.0);
            assert!(
                (vp + vn - v0).abs() < 1e-12 * v0.max(1.0),
                "z={z}: {vp} + {vn} != {v0}"
            );
            assert!(vp > 0.0 && vn > 0.0, "z={z} should cut both sides");
        }
    }

    #[test]
    fn plane_missing_the_cell_returns_one_side() {
        let c = ConvexCell::from_tet(&UNIT_TET);
        let (pos, neg) = c.split_by_plane(&horizontal_plane(5.0));
        assert!(pos.is_none() != neg.is_none(), "exactly one side");
        let kept = pos.or(neg).unwrap();
        assert!((kept.volume() - c.volume()).abs() < 1e-15);
    }

    #[test]
    fn plane_on_a_face_does_not_split() {
        // z = 0 is the tet's own base face: contact, not a cut. One side must
        // come back whole, which is what keeps a conformal boundary from
        // generating spurious cells.
        let c = ConvexCell::from_tet(&UNIT_TET);
        let (pos, neg) = c.split_by_plane(&horizontal_plane(0.0));
        assert!(pos.is_some() && neg.is_none(), "contact must not cut");
        assert!((pos.unwrap().volume() - c.volume()).abs() < 1e-15);
    }

    #[test]
    fn repeated_splits_conserve_volume() {
        // Sequential cutting is how a tet with k planes is handled, so the
        // invariant has to survive cells that are no longer tets.
        let c = ConvexCell::from_tet(&UNIT_TET);
        let v0 = c.volume();
        let mut cells = vec![c];
        for (i, z) in [0.2, 0.4, 0.6].iter().enumerate() {
            let plane = if i % 2 == 0 {
                horizontal_plane(*z)
            } else {
                // A slanted plane, so the cells stop being axis-aligned slabs.
                [[-9.0, -9.0, *z], [9.0, -9.0, *z + 0.3], [-9.0, 9.0, *z]]
            };
            let mut next = Vec::new();
            for cell in &cells {
                let (p, n) = cell.split_by_plane(&plane);
                next.extend(p);
                next.extend(n);
            }
            cells = next;
            let total: f64 = cells.iter().map(|c| c.volume()).sum();
            assert!(
                (total - v0).abs() < 1e-11 * v0.max(1.0),
                "after {} planes: {total} != {v0} ({} cells)",
                i + 1,
                cells.len()
            );
        }
    }

    #[test]
    fn tetrahedralisation_reproduces_the_cell_volume() {
        let c = ConvexCell::from_tet(&UNIT_TET);
        for z in [0.35, 0.5, 0.8] {
            let (pos, _) = c.split_by_plane(&horizontal_plane(z));
            let cell = pos.unwrap();
            let pts = cell.verts.clone();
            let mut tets: Vec<[usize; 4]> = Vec::new();
            cell.tetrahedralise(0, &mut tets);
            assert!(!tets.is_empty(), "z={z}");
            let mut sum = 0.0;
            for t in &tets {
                let v = tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
                // Every emitted tet must be positively oriented (issue #108).
                assert!(v > 0.0, "z={z} tet {t:?} has signed volume {v}");
                sum += v;
            }
            assert!(
                (sum - cell.volume()).abs() < 1e-12 * cell.volume(),
                "z={z}: {sum} != {}",
                cell.volume()
            );
        }
    }

    #[test]
    fn vertex_fan_emits_one_tet_for_a_tet() {
        // The whole point of fanning from a vertex instead of the centroid.
        let c = ConvexCell::from_tet(&UNIT_TET);
        let mut tets: Vec<[usize; 4]> = Vec::new();
        c.tetrahedralise(0, &mut tets);
        assert_eq!(tets.len(), 1, "an unsplit tet must stay one tet");
    }
}
