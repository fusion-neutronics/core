//! Make per-solid tet meshes conformal with each other and with their surface.
//!
//! Each solid is tetrahedralised on its own (in parallel), from the shared,
//! welded surface triangulation. Two steps put new vertices ON that surface,
//! inside one solid only:
//!
//! - boundary recovery's Steiner refinement (a segment or facet the Delaunay
//!   does not contain is split until it is), and
//! - quality improvement's long-edge splits, which split surface edges too.
//!
//! Neither changes the geometry, so each solid still fills its surface exactly,
//! but the triangles its tets present on the surface are then a refinement of
//! the surface triangles. On a surface two solids share, the other solid still
//! presents the unrefined triangle: the two sides do not share faces (hanging
//! nodes), and in either case the tets' faces no longer match the surface
//! triangles transport tracks against.
//!
//! [`make_conformal`] repairs this after all solids are meshed, using only the
//! points the solids already added, by exact local operations:
//!
//! 1. **Edge points.** A point on a surface edge splits that edge everywhere:
//!    every tet of every solid containing the edge, and every surface triangle
//!    containing it, becomes one element per sub-edge (substituting the edge's
//!    end points in order, which keeps orientation and volume exactly).
//! 2. **Face points.** Where one owner of a surface triangle refined it and the
//!    other did not, the refinement becomes the surface triangulation there,
//!    and the other owner's single tet on that triangle is replaced by the cone
//!    from its opposite vertex over the refined triangles. The cone tiles the
//!    tet exactly, so this too preserves volume.
//!
//! A triangle refined differently by both owners is left as it is and counted
//! in [`ConformStats::conflicts`]: this pass never moves a vertex or changes a
//! solid's volume, so the cost of a case it does not cover is that case alone.

use super::dethash::{HashMap, HashSet};
use super::predicates3d::{cross, dot, orient_3d};

/// One tet-meshed solid: its tets over the shared vertex array, and the
/// indices of the surface triangles that bound it.
#[derive(Debug, Clone, Default)]
pub struct SolidTets {
    pub tets: Vec<[usize; 4]>,
    pub surface: Vec<usize>,
}

/// What [`make_conformal`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConformStats {
    /// Surface edges split at points one solid had added on them.
    pub edges_split: usize,
    /// Tets replaced by a cone over another solid's refinement of their face.
    pub cones: usize,
    /// Surface triangles replaced by the refinement the tets present.
    pub surface_triangles_refined: usize,
    /// Surface triangles refined differently by their two owners, or not in a
    /// way that tiles them (left as is).
    pub conflicts: usize,
    /// Solids whose tets are not conformal with their own surface (a
    /// legacy-clip fallback, say): left out of the repair entirely.
    pub skipped_solids: usize,
    /// The repair would have left a surface that is not closed, so nothing
    /// was changed.
    pub reverted: bool,
}

pub(super) fn sorted3(f: [usize; 3]) -> [usize; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

fn tet_faces(t: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [t[1], t[2], t[3]],
        [t[0], t[2], t[3]],
        [t[0], t[1], t[3]],
        [t[0], t[1], t[2]],
    ]
}

/// Faces bounding exactly one tet of `tets` (sorted vertex ids).
fn boundary_faces(tets: &[[usize; 4]]) -> Vec<[usize; 3]> {
    let mut faces: Vec<[usize; 3]> = tets
        .iter()
        .flat_map(|&t| tet_faces(t).map(sorted3))
        .collect();
    faces.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < faces.len() {
        let mut j = i + 1;
        while j < faces.len() && faces[j] == faces[i] {
            j += 1;
        }
        if j - i == 1 {
            out.push(faces[i]);
        }
        i = j;
    }
    out
}

/// Where a point lies on a triangle (a, b, c).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum OnTriangle {
    Interior,
    /// On the edge between the triangle's corners `.0` and `.1` (0..3).
    Edge(usize, usize),
    Corner,
}

/// Uniform grid over triangle bounding boxes, for point location.
pub(super) struct TriGrid {
    lo: [f64; 3],
    cell: f64,
    cells: HashMap<[i64; 3], Vec<u32>>,
}

impl TriGrid {
    pub(super) fn new(vertices: &[[f64; 3]], tris: &[[usize; 3]]) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut total_edge = 0.0;
        for t in tris {
            for &v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(vertices[v][k]);
                    hi[k] = hi[k].max(vertices[v][k]);
                }
            }
            total_edge += dist(vertices[t[0]], vertices[t[1]]);
        }
        let diag = dist(lo, hi);
        // about one median-sized triangle per cell, but never a degenerate grid
        let mean_edge = if tris.is_empty() {
            diag
        } else {
            total_edge / tris.len() as f64
        };
        let cell = mean_edge.max(diag / 4096.0).max(1e-12);
        let mut g = TriGrid {
            lo,
            cell,
            cells: HashMap::default(),
        };
        for (ti, t) in tris.iter().enumerate() {
            let (mut tlo, mut thi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for &v in t {
                for k in 0..3 {
                    tlo[k] = tlo[k].min(vertices[v][k]);
                    thi[k] = thi[k].max(vertices[v][k]);
                }
            }
            let (l, h) = (g.key(tlo), g.key(thi));
            for x in l[0]..=h[0] {
                for y in l[1]..=h[1] {
                    for z in l[2]..=h[2] {
                        g.cells.entry([x, y, z]).or_default().push(ti as u32);
                    }
                }
            }
        }
        g
    }

    fn key(&self, p: [f64; 3]) -> [i64; 3] {
        [0, 1, 2].map(|k| ((p[k] - self.lo[k]) / self.cell).floor() as i64)
    }

    pub(super) fn near(&self, p: [f64; 3]) -> &[u32] {
        self.cells.get(&self.key(p)).map_or(&[], |v| v.as_slice())
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Locate `p` on triangle `t`, or `None` when it is off the triangle.
///
/// The points this pass places were created on the surface by linear splits
/// of its edges and faces, so they lie on their triangle up to rounding. The
/// tolerances are relative to the triangle's size, far below any real
/// separation between surfaces.
pub(super) fn locate(vertices: &[[f64; 3]], t: [usize; 3], p: [f64; 3]) -> Option<OnTriangle> {
    let (a, b, c) = (vertices[t[0]], vertices[t[1]], vertices[t[2]]);
    let n = cross(a, b, c);
    let n2 = dot(n, n);
    if n2 == 0.0 {
        return None;
    }
    let size = dist(a, b).max(dist(b, c)).max(dist(c, a));
    let off_plane = dot(n, [p[0] - a[0], p[1] - a[1], p[2] - a[2]]) / n2.sqrt();
    if off_plane.abs() > 1e-9 * size {
        return None;
    }
    // barycentric weights from sub-triangle areas, signed along n
    let w = [cross(p, b, c), cross(a, p, c), cross(a, b, p)].map(|m| dot(m, n) / n2);
    let tol = 1e-9;
    if w.iter().any(|&x| x < -tol) {
        return None;
    }
    let zero: Vec<usize> = (0..3).filter(|&k| w[k].abs() <= tol).collect();
    Some(match zero.len() {
        0 => OnTriangle::Interior,
        // zero weight on corner k: on the edge opposite it
        1 => {
            let k = zero[0];
            OnTriangle::Edge((k + 1) % 3, (k + 2) % 3)
        }
        _ => OnTriangle::Corner,
    })
}

/// Split `elem` (a tet or a triangle, as its ordered vertex list) along every
/// edge in `splits` that it contains, recursively, into elements over the
/// sub-edges. Substituting the edge's ends keeps orientation and measure.
fn split_by_edges<const N: usize>(
    elem: [usize; N],
    splits: &HashMap<(usize, usize), Vec<usize>>,
    out: &mut Vec<[usize; N]>,
) -> bool {
    for i in 0..N {
        for j in (i + 1)..N {
            let (u, v) = (elem[i], elem[j]);
            let key = (u.min(v), u.max(v));
            if let Some(chain) = splits.get(&key) {
                // chain runs from key.0 to key.1 through the points on the edge
                let chain: Vec<usize> = if u == key.0 {
                    chain.clone()
                } else {
                    chain.iter().rev().copied().collect()
                };
                for w in chain.windows(2) {
                    let mut child = elem;
                    child[i] = w[0];
                    child[j] = w[1];
                    if !split_by_edges(child, splits, out) {
                        out.push(child);
                    }
                }
                return true;
            }
        }
    }
    false
}

/// True when `faces` tile triangle `t` exactly: their areas sum to its area,
/// every edge inside it is shared by two of them, and every edge used once
/// lies on one of its edges. A refinement that fails this (a legacy-clip
/// fallback's faces, say) is not adopted as the surface.
fn tiles(vertices: &[[f64; 3]], t: [usize; 3], faces: &[[usize; 3]]) -> bool {
    let area = |f: [usize; 3]| {
        let n = cross(vertices[f[0]], vertices[f[1]], vertices[f[2]]);
        dot(n, n).sqrt() / 2.0
    };
    // every vertex on the triangle (its plane, inside it): the surface must
    // keep its geometry exactly, whatever the solid's tets did
    if faces
        .iter()
        .flatten()
        .any(|&v| !t.contains(&v) && locate(vertices, t, vertices[v]).is_none())
    {
        return false;
    }
    let whole = area(t);
    let parts: f64 = faces.iter().map(|&f| area(f)).sum();
    if (parts - whole).abs() > 1e-9 * whole {
        return false;
    }
    let mut uses: HashMap<(usize, usize), u32> = HashMap::default();
    for f in faces {
        for k in 0..3 {
            let (u, v) = (f[k], f[(k + 1) % 3]);
            *uses.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    uses.into_iter().all(|((u, v), n)| match n {
        2 => true,
        1 => {
            // both ends on one edge of t
            let on = |w: usize| locate(vertices, t, vertices[w]);
            matches!((on(u), on(v)), (Some(a), Some(b)) if a != OnTriangle::Interior
                && b != OnTriangle::Interior
                && shares_edge(vertices, t, u, v))
        }
        _ => false,
    })
}

/// True when `u` and `v` both lie on one edge (or corner) of triangle `t`.
fn shares_edge(vertices: &[[f64; 3]], t: [usize; 3], u: usize, v: usize) -> bool {
    (0..3).any(|k| {
        let edge = [t[k], t[(k + 1) % 3]];
        let on_edge = |w: usize| {
            edge.contains(&w)
                || matches!(locate(vertices, t, vertices[w]),
                Some(OnTriangle::Edge(i, j)) if {
                    let e = [t[i], t[j]];
                    (e[0] == edge[0] && e[1] == edge[1]) || (e[0] == edge[1] && e[1] == edge[0])
                })
        };
        on_edge(u) && on_edge(v)
    })
}

/// Make `solids`' tets conformal with each other and with `surface`.
///
/// `surface` is the shared surface triangulation over `vertices`, `parent[i]`
/// the original index of surface triangle `i` (pass `0..surface.len()`), and
/// each solid lists the surface triangles bounding it. Surface triangles are
/// refined in place, new ones appended with their parent recorded, and every
/// solid's `surface` list is updated to the refined triangles.
pub fn make_conformal(
    vertices: &[[f64; 3]],
    surface: &mut Vec<[usize; 3]>,
    parent: &mut Vec<usize>,
    solids: &mut [SolidTets],
) -> ConformStats {
    let (s0, p0, t0) = (surface.clone(), parent.clone(), solids.to_vec());
    let mut stats = repair(vertices, surface, parent, solids);
    // the surface is what transport tracks against: it must stay closed
    if !solids.iter().all(|s| closed(surface, &s.surface)) {
        *surface = s0;
        *parent = p0;
        solids.clone_from_slice(&t0);
        stats = ConformStats {
            reverted: true,
            ..stats
        };
    }
    stats
}

/// Every edge of the listed triangles is used by exactly two of them.
fn closed(surface: &[[usize; 3]], tris: &[usize]) -> bool {
    let mut uses: HashMap<(usize, usize), u32> = HashMap::default();
    for &i in tris {
        let t = surface[i];
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            *uses.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    uses.values().all(|&n| n == 2)
}

fn repair(
    vertices: &[[f64; 3]],
    surface: &mut Vec<[usize; 3]>,
    parent: &mut Vec<usize>,
    solids: &mut [SolidTets],
) -> ConformStats {
    let mut stats = ConformStats::default();
    if solids.is_empty() || surface.is_empty() {
        return stats;
    }
    let surface_key: HashMap<[usize; 3], usize> = surface
        .iter()
        .enumerate()
        .map(|(i, &t)| (sorted3(t), i))
        .collect();
    let grid = TriGrid::new(vertices, surface);

    // ── 1. Each solid's refinement of each surface triangle it owns ────────
    // A boundary face of a solid's tets that is not a surface triangle lies
    // inside one: find which (its centroid is interior to exactly one).
    let mut refinement: HashMap<usize, Vec<(usize, Vec<[usize; 3]>)>> = HashMap::default();
    let mut messy = vec![false; solids.len()];
    for (si, s) in solids.iter().enumerate() {
        let own: HashSet<usize> = s.surface.iter().copied().collect();
        let mut per_tri: HashMap<usize, Vec<[usize; 3]>> = HashMap::default();
        for f in boundary_faces(&s.tets) {
            if surface_key.contains_key(&f) {
                continue;
            }
            let c = [0, 1, 2]
                .map(|k| (vertices[f[0]][k] + vertices[f[1]][k] + vertices[f[2]][k]) / 3.0);
            match grid.near(c).iter().find(|&&ti| {
                own.contains(&(ti as usize))
                    && locate(vertices, surface[ti as usize], c) == Some(OnTriangle::Interior)
            }) {
                Some(&ti) => per_tri.entry(ti as usize).or_default().push(f),
                // a boundary face off its own surface: the solid's tets are
                // not conformal with it, so leave this solid alone
                None => messy[si] = true,
            }
        }
        if messy[si] {
            stats.skipped_solids += 1;
            continue;
        }
        for (ti, mut faces) in per_tri {
            faces.sort_unstable();
            refinement.entry(ti).or_default().push((si, faces));
        }
    }
    if refinement.is_empty() {
        return stats;
    }
    let mut owners: HashMap<usize, Vec<usize>> = HashMap::default();
    for (si, s) in solids.iter().enumerate() {
        for &ti in &s.surface {
            owners.entry(ti).or_default().push(si);
        }
    }

    // ── 2. Adopt one refinement per triangle; cone the other owner's tet ──
    let mut replace: Vec<(usize, Vec<[usize; 3]>)> = Vec::new();
    let mut cones: HashMap<usize, Vec<([usize; 3], Vec<[usize; 3]>)>> = HashMap::default();
    let mut edge_points: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    let mut tris: Vec<usize> = refinement.keys().copied().collect();
    tris.sort_unstable();
    for ti in tris {
        let refined = &refinement[&ti];
        if owners
            .get(&ti)
            .is_some_and(|o| o.iter().any(|&si| messy[si]))
        {
            continue;
        }
        let canonical = refined[0].1.clone();
        if refined.iter().any(|(_, faces)| *faces != canonical)
            || !tiles(vertices, surface[ti], &canonical)
        {
            stats.conflicts += 1;
            continue;
        }
        for &si in owners.get(&ti).map_or(&[][..], |v| v.as_slice()) {
            if !refined.iter().any(|(s, _)| *s == si) {
                cones
                    .entry(si)
                    .or_default()
                    .push((sorted3(surface[ti]), canonical.clone()));
            }
        }
        // points the refinement put on this triangle's edges
        let t = surface[ti];
        for v in canonical.iter().flatten().copied() {
            if t.contains(&v) {
                continue;
            }
            if let Some(OnTriangle::Edge(i, j)) = locate(vertices, t, vertices[v]) {
                let key = (t[i].min(t[j]), t[i].max(t[j]));
                let e = edge_points.entry(key).or_default();
                if !e.contains(&v) {
                    e.push(v);
                }
            }
        }
        replace.push((ti, canonical));
    }

    let mut cone_list: Vec<_> = cones.into_iter().collect();
    cone_list.sort_unstable_by_key(|(si, _)| *si);
    for (si, list) in cone_list {
        let s = &mut solids[si];
        let by_face: HashMap<[usize; 3], &Vec<[usize; 3]>> =
            list.iter().map(|(f, r)| (*f, r)).collect();
        let mut tets = Vec::with_capacity(s.tets.len() + list.len() * 4);
        for &t in &s.tets {
            let hit = tet_faces(t)
                .into_iter()
                .enumerate()
                .find_map(|(k, f)| by_face.get(&sorted3(f)).map(|c| (t[k], *c)));
            match hit {
                Some((apex, children)) => {
                    for c in children {
                        let mut nt = [c[0], c[1], c[2], apex];
                        if orient_3d(
                            vertices[nt[0]],
                            vertices[nt[1]],
                            vertices[nt[2]],
                            vertices[nt[3]],
                        ) < 0.0
                        {
                            nt.swap(0, 1);
                        }
                        tets.push(nt);
                    }
                    stats.cones += 1;
                }
                None => tets.push(t),
            }
        }
        s.tets = tets;
    }

    // the surface takes each adopted refinement, oriented like its triangle
    let mut children: Vec<Vec<usize>> = (0..surface.len()).map(|i| vec![i]).collect();
    for (ti, faces) in replace {
        let t = surface[ti];
        let n = cross(vertices[t[0]], vertices[t[1]], vertices[t[2]]);
        let oriented: Vec<[usize; 3]> = faces
            .iter()
            .map(|&f| {
                if dot(cross(vertices[f[0]], vertices[f[1]], vertices[f[2]]), n) < 0.0 {
                    [f[0], f[2], f[1]]
                } else {
                    f
                }
            })
            .collect();
        let p = parent[ti];
        surface[ti] = oriented[0];
        for &f in &oriented[1..] {
            children[ti].push(surface.len());
            surface.push(f);
            parent.push(p);
        }
        stats.surface_triangles_refined += 1;
    }
    for s in solids.iter_mut() {
        s.surface = s
            .surface
            .iter()
            .flat_map(|&i| children[i].clone())
            .collect();
    }

    // ── 3. Split what still has an edge a refinement put points on ────────
    // tets around that edge in every solid, and surface triangles beside it
    if edge_points.is_empty() {
        return stats;
    }
    let mut chains: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    for (&key, pts) in &edge_points {
        let (a, b) = (vertices[key.0], vertices[key.1]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let along = |v: usize| {
            let p = vertices[v];
            dot([p[0] - a[0], p[1] - a[1], p[2] - a[2]], ab)
        };
        let mut pts = pts.clone();
        pts.sort_by(|&x, &y| along(x).total_cmp(&along(y)).then(x.cmp(&y)));
        let mut chain = vec![key.0];
        chain.extend(pts);
        chain.push(key.1);
        chains.insert(key, chain);
    }
    stats.edges_split = chains.len();
    let mut children: Vec<Vec<usize>> = Vec::with_capacity(surface.len());
    let mut new_surface = Vec::with_capacity(surface.len());
    let mut new_parent = Vec::with_capacity(surface.len());
    for (i, &t) in surface.iter().enumerate() {
        let mut out = Vec::new();
        if !split_by_edges(t, &chains, &mut out) {
            out.push(t);
        }
        children.push((new_surface.len()..new_surface.len() + out.len()).collect());
        new_parent.extend(std::iter::repeat_n(parent[i], out.len()));
        new_surface.extend(out);
    }
    *surface = new_surface;
    *parent = new_parent;
    for s in solids.iter_mut() {
        s.surface = s
            .surface
            .iter()
            .flat_map(|&i| children[i].clone())
            .collect();
        let mut tets = Vec::with_capacity(s.tets.len());
        for &t in &s.tets {
            if !split_by_edges(t, &chains, &mut tets) {
                tets.push(t);
            }
        }
        s.tets = tets;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(v: &[[f64; 3]], tets: &[[usize; 4]]) -> f64 {
        tets.iter()
            .map(|t| orient_3d(v[t[0]], v[t[1]], v[t[2]], v[t[3]]) / 6.0)
            .sum()
    }

    /// Two tets sharing face (0,1,2); the lower one has split that face at
    /// its centroid (vertex 5) and its edge (0,1) at its midpoint (vertex 6).
    fn two_tets() -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<SolidTets>) {
        let v = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
        // surface: the shared triangle plus each tet's three other faces
        let surface = vec![
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 0, 4],
        ];
        let upper = SolidTets {
            tets: vec![[0, 1, 2, 3]],
            surface: vec![0, 1, 2, 3],
        };
        // lower: the tet (0,1,2,4) split at 6 on edge (0,1), then at 5 on face
        // (0,1,2): children cone from 4 over the refined face
        let lower = SolidTets {
            tets: vec![[0, 6, 5, 4], [6, 1, 5, 4], [1, 2, 5, 4], [2, 0, 5, 4]],
            surface: vec![0, 4, 5, 6],
        };
        (v, surface, vec![upper, lower])
    }

    #[test]
    fn repairs_a_one_sided_refinement_exactly() {
        let (v, mut surface, mut solids) = two_tets();
        for s in &mut solids {
            for t in &mut s.tets {
                if orient_3d(v[t[0]], v[t[1]], v[t[2]], v[t[3]]) < 0.0 {
                    t.swap(0, 1);
                }
            }
        }
        let before: Vec<f64> = solids.iter().map(|s| volume(&v, &s.tets)).collect();
        let mut parent: Vec<usize> = (0..surface.len()).collect();
        let stats = make_conformal(&v, &mut surface, &mut parent, &mut solids);
        assert_eq!(stats.edges_split, 1, "{stats:?}");
        assert_eq!(stats.conflicts, 0);
        assert!(stats.cones >= 1);
        // volumes unchanged, every tet positive
        for (s, b) in solids.iter().zip(&before) {
            assert!((volume(&v, &s.tets) - b).abs() < 1e-15);
            for t in &s.tets {
                assert!(orient_3d(v[t[0]], v[t[1]], v[t[2]], v[t[3]]) > 0.0);
            }
        }
        // conformal: the two solids' faces on z = 0 coincide, and each solid's
        // tet boundary is exactly its surface triangles
        let on_plane = |f: &[usize; 3]| f.iter().all(|&i| v[i][2] == 0.0);
        let upper: HashSet<[usize; 3]> = boundary_faces(&solids[0].tets)
            .into_iter()
            .filter(on_plane)
            .collect();
        let lower: HashSet<[usize; 3]> = boundary_faces(&solids[1].tets)
            .into_iter()
            .filter(on_plane)
            .collect();
        assert_eq!(upper, lower);
        for s in &solids {
            let mut b = boundary_faces(&s.tets);
            let mut surf: Vec<[usize; 3]> =
                s.surface.iter().map(|&i| sorted3(surface[i])).collect();
            b.sort_unstable();
            surf.sort_unstable();
            assert_eq!(b, surf);
        }
        // every refined triangle remembers its parent
        assert_eq!(parent.len(), surface.len());
        assert!(parent.iter().all(|&p| p < 7));
    }

    #[test]
    fn leaves_a_solid_alone_when_its_tets_leave_its_surface() {
        let (v, surface, mut solids) = two_tets();
        let mut v = v;
        // the lower solid's tets now reach a point below its surface: one of
        // its boundary faces lies on no surface triangle
        v[5] = [1.0 / 3.0, 1.0 / 3.0, -0.25];
        for s in &mut solids {
            for t in &mut s.tets {
                if orient_3d(v[t[0]], v[t[1]], v[t[2]], v[t[3]]) < 0.0 {
                    t.swap(0, 1);
                }
            }
        }
        let (s0, t0) = (surface.clone(), solids.clone());
        let mut surface = surface;
        let mut parent: Vec<usize> = (0..surface.len()).collect();
        let stats = make_conformal(&v, &mut surface, &mut parent, &mut solids);
        assert_eq!(stats.skipped_solids, 1, "{stats:?}");
        assert_eq!(stats.cones, 0);
        assert_eq!(surface, s0);
        assert_eq!(solids[0].tets, t0[0].tets);
    }
}
