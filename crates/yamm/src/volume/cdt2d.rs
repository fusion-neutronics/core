//! 2-D constrained Delaunay triangulation (CDT).
//!
//! Recovering a flat boundary FACE in the 3-D volume mesher (issue #31) reduces
//! to a 2-D problem: the face's vertices are coplanar, and we need a
//! triangulation of that plane in which every boundary SEGMENT of the face is
//! an edge. The 3-D bipyramid edge-removal (`flip_ring_general`) provably cannot
//! do this for reflex faces - there is no valid 3-D flip when the whole
//! configuration is coplanar. In 2-D, by contrast, **constrained edge recovery
//! always succeeds**: any segment between two vertices can be made an edge by
//! repeatedly flipping the edges that cross it, and each such flip is on a convex
//! quad (Lawson). That is why this 2-D step is the right tool.
//!
//! This module is pure 2-D geometry (no 3-D mesh interaction) so it is small and
//! directly unit-testable. The caller projects a flat face to 2-D, runs
//! [`triangulate`], and lifts the resulting triangles back (the 3-D installation
//! into the tet mesh - cavity facet recovery - is a separate step).
//!
//! Predicates are plain `f64` (orient2d / in-circle). The flat-face vertices a
//! caller passes are real surface points projected onto the face plane, so they
//! are well separated; exact predicates can be swapped in later if a degenerate
//! input is ever observed.

use super::dethash::{HashMap, HashSet};

/// Twice the signed area of triangle (a, b, c); > 0 iff CCW. EXACT (Shewchuk):
/// the flat-face inputs this module exists for (issue #31) are full of
/// collinear triples - straight subdivided cap edges - where the naive f64
/// determinant returns noise instead of 0. Every orientation decision in this
/// module (crossing tests, flip convexity) goes through this predicate.
#[inline]
pub(super) fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    geometry_predicates::orient2d(a, b, c)
}

/// orient2d with Simulation-of-Simplicity tie-breaks (Edelsbrunner–Mücke):
/// the exact sign when nonzero; otherwise the sign the determinant takes
/// under a symbolic perturbation p_i → (x_i + ε_i, y_i + ε'_i) whose
/// magnitudes decrease strictly with VERTEX INDEX (x-perturbation dominating
/// y per point). Expanding the perturbed determinant for index-sorted points
/// 1 < 2 < 3 orders the monomials as: det, then (y2−y3), (x3−x2), (y3−y1),
/// then the constant −1 (the ε₁'ε₂ cross term) - so the result is NEVER zero
/// and all decisions are mutually consistent (they are realized by one
/// genuine perturbed point set; every classical general-position theorem -
/// termination and success of constrained edge insertion by flips - then
/// holds verbatim).
///
/// SAFE HERE, unlike the historical 3-D global-SoS attempt: this module's
/// triangulations are purely COMBINATORIAL bridges (the 3-D caller realizes
/// them as zero-volume stack tets, which have no geometric validity
/// requirements), so no symbolic convention ever crosses into the non-SoS
/// 3-D Delaunay structure.
fn orient2d_sos(pts: &[[f64; 2]], i: u32, j: u32, k: u32) -> f64 {
    let (mut a, mut b, mut c) = (i, j, k);
    let mut sign = 1.0f64;
    if a > b {
        std::mem::swap(&mut a, &mut b);
        sign = -sign;
    }
    if b > c {
        std::mem::swap(&mut b, &mut c);
        sign = -sign;
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
        sign = -sign;
    }
    let (p1, p2, p3) = (pts[a as usize], pts[b as usize], pts[c as usize]);
    let d0 = orient2d(p1, p2, p3);
    if d0 != 0.0 {
        return sign * d0;
    }
    let t1 = p2[1] - p3[1];
    if t1 != 0.0 {
        return sign * t1;
    }
    let t2 = p3[0] - p2[0];
    if t2 != 0.0 {
        return sign * t2;
    }
    let t3 = p3[1] - p1[1];
    if t3 != 0.0 {
        return sign * t3;
    }
    -sign
}

/// > 0 iff `d` lies strictly inside the circumcircle of CCW triangle (a, b, c).
#[inline]
fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let ad = adx * adx + ady * ady;
    let bd = bdx * bdx + bdy * bdy;
    let cd = cdx * cdx + cdy * cdy;
    adx * (bdy * cd - cdy * bd) - ady * (bdx * cd - cdx * bd) + ad * (bdx * cdy - cdx * bdy)
}

/// Do open segments (p1,p2) and (q1,q2) cross at an interior point?
#[inline]
fn segments_cross(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> bool {
    let d1 = orient2d(q1, q2, p1);
    let d2 = orient2d(q1, q2, p2);
    let d3 = orient2d(p1, p2, q1);
    let d4 = orient2d(p1, p2, q2);
    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0) && d1 != 0.0 && d2 != 0.0
}

/// `segments_cross` under the SoS predicate (never a tie). Vertex indices
/// index into `pts`.
#[inline]
fn segments_cross_sos(pts: &[[f64; 2]], a: u32, b: u32, x: u32, y: u32) -> bool {
    let d1 = orient2d_sos(pts, x, y, a);
    let d2 = orient2d_sos(pts, x, y, b);
    let d3 = orient2d_sos(pts, a, b, x);
    let d4 = orient2d_sos(pts, a, b, y);
    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
}

/// A triangle-based 2-D triangulation with neighbour adjacency. Vertices are
/// indices into `pts`; the first three are a bounding super-triangle (removed at
/// the end). `adj[t][i]` is the triangle opposite local vertex `i` of triangle
/// `t` (or `NONE`), kept symmetric.
const NONE: u32 = u32::MAX;

pub(super) struct Mesh {
    pts: Vec<[f64; 2]>,
    tri: Vec<[u32; 3]>,
    adj: Vec<[u32; 3]>,
    dead: Vec<bool>,
}

/// One logged diagonal flip: the quad's OLD diagonal (s1, s2) was replaced by
/// the NEW diagonal (x, y) - i.e. triangles (s1,s2,x), (s1,s2,y) became
/// (x,y,s1), (x,y,s2). Indices are vertex indices of the mesh's point list.
/// The 3-D caller (issue #31 coplanar-region recovery) replays a flip log as a
/// stack of zero-volume tets {s1,s2,x,y}: each 2-D flip IS one flat tet whose
/// bottom faces are the pre-flip triangle pair and whose top faces are the
/// post-flip pair - the classic triangulation-flip ↔ stacked-tet
/// correspondence that lets a purely 2-D recovery be imposed on the 3-D mesh.
#[derive(Clone, Copy, Debug)]
pub(super) struct FlipLog {
    pub old_diag: [u32; 2],
    pub new_diag: [u32; 2],
}

impl Mesh {
    /// Local index (0..3) of vertex `v` in triangle `t`, or None.
    fn local(&self, t: usize, v: u32) -> Option<usize> {
        (0..3).find(|&i| self.tri[t][i] == v)
    }

    /// Load an EXISTING triangulation (no super-triangle, vertex indices are
    /// the caller's, no offset). Each triangle is re-oriented CCW; adjacency is
    /// built from an edge map. Returns None when the input is not a valid
    /// 2-manifold triangulation: a zero-area triangle, or an edge shared by
    /// more than two triangles. Used by the issue-#31 coplanar-region recovery
    /// to load the 3-D mesh's in-plane sheet triangulations for 2-D flipping.
    pub(super) fn from_triangulation(pts: &[[f64; 2]], tris: &[[usize; 3]]) -> Option<Mesh> {
        let mut mesh = Mesh {
            pts: pts.to_vec(),
            tri: Vec::with_capacity(tris.len()),
            adj: vec![[NONE; 3]; tris.len()],
            dead: vec![false; tris.len()],
        };
        for t in tris {
            let mut v = [t[0] as u32, t[1] as u32, t[2] as u32];
            let o = orient2d(pts[t[0]], pts[t[1]], pts[t[2]]);
            if o == 0.0 {
                return None; // degenerate triangle
            }
            if o < 0.0 {
                v.swap(1, 2);
            }
            mesh.tri.push(v);
        }
        // Adjacency: map each undirected edge to the (triangle, local edge)
        // that first registered it; the second occurrence links the pair.
        let mut emap: HashMap<(u32, u32), (usize, usize)> = HashMap::default();
        for t in 0..mesh.tri.len() {
            for e in 0..3 {
                let (a, b) = (mesh.tri[t][(e + 1) % 3], mesh.tri[t][(e + 2) % 3]);
                let key = (a.min(b), a.max(b));
                match emap.get(&key) {
                    Some(&(ot, oe)) => {
                        if mesh.adj[t][e] != NONE || mesh.adj[ot][oe] != NONE {
                            return None; // edge in >2 triangles
                        }
                        mesh.adj[t][e] = ot as u32;
                        mesh.adj[ot][oe] = t as u32;
                    }
                    None => {
                        emap.insert(key, (t, e));
                    }
                }
            }
        }
        Some(mesh)
    }

    /// Does undirected edge (a, b) exist in a live triangle?
    pub(super) fn has_edge(&self, a: u32, b: u32) -> bool {
        self.tri.iter().enumerate().any(|(t, tri)| {
            !self.dead[t]
                && (0..3).any(|e| {
                    let (x, y) = (tri[(e + 1) % 3], tri[(e + 2) % 3]);
                    (x == a && y == b) || (x == b && y == a)
                })
        })
    }

    /// All live triangles as SORTED vertex triples (set semantics).
    pub(super) fn triangles_sorted(&self) -> HashSet<[u32; 3]> {
        self.tri
            .iter()
            .enumerate()
            .filter(|&(t, _)| !self.dead[t])
            .map(|(_, tri)| {
                let mut s = *tri;
                s.sort_unstable();
                s
            })
            .collect()
    }

    /// Set the neighbour of triangle `t` across local edge `e` to `nb`, and (if
    /// `nb` is real) make `nb` point back. Keeps adjacency symmetric.
    fn link(&mut self, t: usize, e: usize, nb: u32) {
        self.adj[t][e] = nb;
        if nb != NONE {
            // edge of t across local e = (tri[t][(e+1)%3], tri[t][(e+2)%3]); the
            // mirror edge in nb is the one opposite the vertex nb does NOT share.
            let (a, b) = (self.tri[t][(e + 1) % 3], self.tri[t][(e + 2) % 3]);
            for j in 0..3 {
                let (x, y) = (
                    self.tri[nb as usize][(j + 1) % 3],
                    self.tri[nb as usize][(j + 2) % 3],
                );
                if (x == a && y == b) || (x == b && y == a) {
                    self.adj[nb as usize][j] = t as u32;
                    break;
                }
            }
        }
    }
}

/// Build a 2-D constrained Delaunay triangulation of `pts` in which every
/// segment in `segments` (index pairs into `pts`) is an edge, then return the
/// triangles strictly INSIDE the region bounded by `segments` (the face
/// interior), as vertex-index triples into the original `pts`.
///
/// `segments` must form the (possibly non-convex, possibly multiply-connected)
/// closed boundary of the face. Returns an empty vec on degenerate input
/// (< 3 points, all collinear, or a constraint that cannot be represented).
pub fn triangulate(pts: &[[f64; 2]], segments: &[[usize; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    // Super-triangle covering all points (indices 0..3 in the mesh point list;
    // real points are offset by 3).
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let dx = (hi[0] - lo[0]).max(1.0);
    let dy = (hi[1] - lo[1]).max(1.0);
    let m = dx.max(dy) * 1000.0;
    let cx = (lo[0] + hi[0]) * 0.5;
    let cy = (lo[1] + hi[1]) * 0.5;
    let mut mp: Vec<[f64; 2]> = vec![[cx - m, cy - m], [cx + m, cy - m], [cx, cy + m]];
    mp.extend_from_slice(pts);
    let mut mesh = Mesh {
        pts: mp,
        tri: vec![[0, 1, 2]],
        adj: vec![[NONE; 3]],
        dead: vec![false],
    };

    // ── Incremental Bowyer-Watson insertion of the real points ──
    for raw in 0..n {
        let v = (raw + 3) as u32;
        let p = mesh.pts[v as usize];
        // Find one triangle whose circumcircle contains p.
        let mut seed = NONE;
        for (t, tri) in mesh.tri.iter().enumerate() {
            if mesh.dead[t] {
                continue;
            }
            let (a, b, c) = (
                mesh.pts[tri[0] as usize],
                mesh.pts[tri[1] as usize],
                mesh.pts[tri[2] as usize],
            );
            if in_circle(a, b, c, p) > 0.0 {
                seed = t as u32;
                break;
            }
        }
        if seed == NONE {
            continue; // duplicate / outside all circumcircles (degenerate)
        }
        // Grow the cavity (all triangles whose circumcircle contains p, connected).
        let mut cavity = vec![seed as usize];
        let mut stack = vec![seed as usize];
        let mut in_cav: HashSet<usize> = [seed as usize].into_iter().collect();
        while let Some(t) = stack.pop() {
            for e in 0..3 {
                let nb = mesh.adj[t][e];
                if nb == NONE || in_cav.contains(&(nb as usize)) {
                    continue;
                }
                let tr = mesh.tri[nb as usize];
                let (a, b, c) = (
                    mesh.pts[tr[0] as usize],
                    mesh.pts[tr[1] as usize],
                    mesh.pts[tr[2] as usize],
                );
                if in_circle(a, b, c, p) > 0.0 {
                    in_cav.insert(nb as usize);
                    cavity.push(nb as usize);
                    stack.push(nb as usize);
                }
            }
        }
        // Boundary edges of the cavity (edge, outer neighbour) → new triangles.
        let mut bedges: Vec<(u32, u32, u32)> = Vec::new(); // (a, b, outer_nb)
        for &t in &cavity {
            for e in 0..3 {
                let nb = mesh.adj[t][e];
                if nb == NONE || !in_cav.contains(&(nb as usize)) {
                    bedges.push((mesh.tri[t][(e + 1) % 3], mesh.tri[t][(e + 2) % 3], nb));
                }
            }
        }
        for &t in &cavity {
            mesh.dead[t] = true;
        }
        // Create new triangles (a, b, p), recording them for adjacency linking.
        let mut created: Vec<usize> = Vec::with_capacity(bedges.len());
        for &(a, b, outer) in &bedges {
            let t = mesh.tri.len();
            mesh.tri.push([a, b, v]);
            mesh.adj.push([NONE; 3]);
            mesh.dead.push(false);
            // local edge 2 of (a,b,v) is the (a,b) edge → links to outer neighbour.
            mesh.link(t, 2, outer);
            created.push(t);
        }
        // Link the new triangles to each other along their shared (v-incident) edges.
        for ia in 0..created.len() {
            for ib in (ia + 1)..created.len() {
                let (ta, tb) = (created[ia], created[ib]);
                // shared edge = the one containing v and a common other vertex.
                for ea in 0..3 {
                    if mesh.adj[ta][ea] != NONE {
                        continue;
                    }
                    let (a1, a2) = (mesh.tri[ta][(ea + 1) % 3], mesh.tri[ta][(ea + 2) % 3]);
                    for eb in 0..3 {
                        if mesh.adj[tb][eb] != NONE {
                            continue;
                        }
                        let (b1, b2) = (mesh.tri[tb][(eb + 1) % 3], mesh.tri[tb][(eb + 2) % 3]);
                        if (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1) {
                            mesh.adj[ta][ea] = tb as u32;
                            mesh.adj[tb][eb] = ta as u32;
                        }
                    }
                }
            }
        }
    }

    // ── Recover each constraint segment by flipping the edges that cross it ──
    for seg in segments {
        let (va, vb) = ((seg[0] + 3) as u32, (seg[1] + 3) as u32);
        recover_edge_logged(&mut mesh, va, vb, &mut Vec::new());
    }

    // ── Region extraction: flood "outside" from the super-triangle across
    //    non-constraint edges; keep the triangles NOT reached and not touching
    //    the super-triangle. ──
    let cset: HashSet<(u32, u32)> = segments
        .iter()
        .map(|s| {
            let (a, b) = ((s[0] + 3) as u32, (s[1] + 3) as u32);
            (a.min(b), a.max(b))
        })
        .collect();
    let mut outside = vec![false; mesh.tri.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (t, tri) in mesh.tri.iter().enumerate() {
        if !mesh.dead[t] && tri.iter().any(|&x| x < 3) {
            outside[t] = true;
            stack.push(t);
        }
    }
    while let Some(t) = stack.pop() {
        for e in 0..3 {
            let nb = mesh.adj[t][e];
            if nb == NONE || mesh.dead[nb as usize] || outside[nb as usize] {
                continue;
            }
            let (a, b) = (mesh.tri[t][(e + 1) % 3], mesh.tri[t][(e + 2) % 3]);
            if cset.contains(&(a.min(b), a.max(b))) {
                continue; // a constraint edge blocks the flood
            }
            outside[nb as usize] = true;
            stack.push(nb as usize);
        }
    }

    let mut out = Vec::new();
    for (t, tri) in mesh.tri.iter().enumerate() {
        if mesh.dead[t] || outside[t] || tri.iter().any(|&x| x < 3) {
            continue;
        }
        out.push([
            (tri[0] - 3) as usize,
            (tri[1] - 3) as usize,
            (tri[2] - 3) as usize,
        ]);
    }
    out
}

/// Make segment (va, vb) an edge of the mesh by flipping the current edges that
/// cross it, appending every executed flip to `log`. Returns true iff the edge
/// is present afterwards.
///
/// In 2-D, when no vertex lies on the open segment, the crossing edges always
/// include at least one whose quad is strictly convex, and flipping it cannot
/// increase the crossing count - so scanning ALL crossing edges each round and
/// flipping the first flippable one terminates with the segment recovered
/// (Anglada's constrained-edge insertion). The previous first-crossing-only
/// variant stalled on the exactly-collinear configurations the flat-face
/// inputs are full of. Returns false on a stall (a vertex on the segment, or
/// the segment leaving the triangulated region) - callers treat that as a
/// clean bail.
pub(super) fn recover_edge_logged(
    mesh: &mut Mesh,
    va: u32,
    vb: u32,
    log: &mut Vec<FlipLog>,
) -> bool {
    // TRANSACTIONAL: a failed recovery must leave neither the mesh nor the
    // log polluted - the 3-D caller replays the log as a stack of flat tets,
    // so junk flips become junk mesh layers. Every 2-D flip is exactly
    // invertible (flip the new diagonal back), so on failure we unwind to the
    // entry state. The flip budget is progress-bound, not topology-bound: a
    // degenerate configuration can cycle, and cycling must fail fast.
    let log_start = log.len();
    let mut budget: Option<usize> = None;
    let unwind = |mesh: &mut Mesh, log: &mut Vec<FlipLog>| {
        while log.len() > log_start {
            let fl = log.pop().unwrap();
            let undone = flip_edge(mesh, fl.new_diag[0], fl.new_diag[1]);
            debug_assert!(matches!(
                undone,
                Some(FlipLog { new_diag, .. }) if {
                    let mut a = new_diag; a.sort_unstable();
                    let mut b = fl.old_diag; b.sort_unstable();
                    a == b
                }
            ));
        }
        false
    };
    loop {
        if mesh.has_edge(va, vb) {
            return true;
        }
        // Collect ALL edges crossing the open segment this round. Crossing
        // tests use the SoS predicate: a collinear configuration (subdivided
        // straight cap edges, BCC lattice rows) is decided symbolically, so
        // every quad along the segment is strictly convex or strictly not -
        // the recovery can never stall on a tie (the flat-cap regions are
        // full of them).
        let mut crossings: Vec<(usize, usize)> = Vec::new(); // (tri, local edge)
        for (t, tri) in mesh.tri.iter().enumerate() {
            if mesh.dead[t] {
                continue;
            }
            for e in 0..3 {
                let (x, y) = (tri[(e + 1) % 3], tri[(e + 2) % 3]);
                if x == va || x == vb || y == va || y == vb {
                    continue; // shares an endpoint
                }
                if x < y && segments_cross_sos(&mesh.pts, va, vb, x, y) {
                    crossings.push((t, e));
                }
            }
        }
        if crossings.is_empty() {
            return unwind(mesh, log); // nothing crosses - vertex on segment
        }
        // Budget: proportional to the INITIAL crossing count; in general
        // position each crossing needs ~1 flip, so a generous multiple covers
        // the degenerate detours while a true cycle exhausts it quickly.
        let b = *budget.get_or_insert(crossings.len() * 8 + 32);
        if b == 0 {
            return unwind(mesh, log); // cycling - fail clean
        }
        budget = Some(b - 1);
        let mut flipped = false;
        for &(t, e) in &crossings {
            if let Some(fl) = flip(mesh, t, e) {
                log.push(fl);
                flipped = true;
                break;
            }
        }
        if !flipped {
            return unwind(mesh, log); // every crossing quad non-convex - stalled
        }
    }
}

/// Flip the diagonal (x, y): locate the live triangle pair sharing that edge
/// and flip it. Used to UNDO a logged flip (the inverse of a flip is the flip
/// of its new diagonal).
fn flip_edge(mesh: &mut Mesh, x: u32, y: u32) -> Option<FlipLog> {
    for t in 0..mesh.tri.len() {
        if mesh.dead[t] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (mesh.tri[t][(e + 1) % 3], mesh.tri[t][(e + 2) % 3]);
            if (a == x && b == y) || (a == y && b == x) {
                return flip(mesh, t, e);
            }
        }
    }
    None
}

/// Flip the shared edge (local edge `e` of triangle `t`) with its neighbour, if
/// the union quad is STRICTLY convex. Returns the executed flip, or None (no
/// change) if there is no neighbour or the quad is non-convex/degenerate.
fn flip(mesh: &mut Mesh, t: usize, e: usize) -> Option<FlipLog> {
    let nb = mesh.adj[t][e];
    if nb == NONE {
        return None;
    }
    let nb = nb as usize;
    // Shared edge endpoints, and the two apexes.
    let apex_t = mesh.tri[t][e];
    let s1 = mesh.tri[t][(e + 1) % 3];
    let s2 = mesh.tri[t][(e + 2) % 3];
    let apex_n = (0..3)
        .map(|i| mesh.tri[nb][i])
        .find(|&v| v != s1 && v != s2)
        .unwrap();
    // Convex iff the two apexes are on opposite sides of the shared edge AND
    // the shared-edge endpoints are on opposite sides of the new edge -
    // under the SoS predicate, so a collinear quad corner is decided
    // symbolically instead of stalling the recovery. A symbolically-convex,
    // geometrically-flat flip commits a zero-AREA triangle, which is fine:
    // these triangulations are combinatorial bridges (3-D zero-volume stack
    // layers), and every later decision uses the same consistent predicate.
    let (o1, o2) = (
        orient2d_sos(&mesh.pts, s1, s2, apex_t),
        orient2d_sos(&mesh.pts, s1, s2, apex_n),
    );
    if (o1 > 0.0) == (o2 > 0.0) {
        return None;
    }
    let (o3, o4) = (
        orient2d_sos(&mesh.pts, apex_t, apex_n, s1),
        orient2d_sos(&mesh.pts, apex_t, apex_n, s2),
    );
    if (o3 > 0.0) == (o4 > 0.0) {
        return None; // not convex - flipping would invert
    }
    // Outer neighbours of the four boundary edges of the quad.
    // In t: edges opposite s1 and s2. In nb: edges opposite s1 and s2.
    let t_opp_s1 = mesh.adj[t][mesh.local(t, s1).unwrap()];
    let t_opp_s2 = mesh.adj[t][mesh.local(t, s2).unwrap()];
    let n_opp_s1 = mesh.adj[nb][mesh.local(nb, s1).unwrap()];
    let n_opp_s2 = mesh.adj[nb][mesh.local(nb, s2).unwrap()];
    // Rebuild the two triangles around the new diagonal, KEEPING them CCW
    // (o3/o4 are exact and nonzero, so the orientation is decided, not
    // guessed). Storing the apexes in a fixed order regardless of o3's sign -
    // the original code's behaviour - emitted CW triangles, breaking every
    // consumer that relies on consistent orientation (e.g. an undirected edge
    // appearing once per direction across its two owners).
    mesh.adj[t] = [NONE; 3];
    mesh.adj[nb] = [NONE; 3];
    if o3 > 0.0 {
        // (apex_t, apex_n, s1) is CCW: edge opp apex_t = (apex_n,s1) →
        // n_opp_s2; opp apex_n = (s1,apex_t) → t_opp_s2; opp s1 = diagonal.
        mesh.tri[t] = [apex_t, apex_n, s1];
        mesh.link(t, 0, n_opp_s2);
        mesh.link(t, 1, t_opp_s2);
        mesh.adj[t][2] = nb as u32;
    } else {
        mesh.tri[t] = [apex_n, apex_t, s1];
        mesh.link(t, 0, t_opp_s2);
        mesh.link(t, 1, n_opp_s2);
        mesh.adj[t][2] = nb as u32;
    }
    if o4 < 0.0 {
        // (apex_n, apex_t, s2) is CCW: edge opp apex_n = (apex_t,s2) →
        // t_opp_s1; opp apex_t = (s2,apex_n) → n_opp_s1; opp s2 = diagonal.
        mesh.tri[nb] = [apex_n, apex_t, s2];
        mesh.link(nb, 0, t_opp_s1);
        mesh.link(nb, 1, n_opp_s1);
        mesh.adj[nb][2] = t as u32;
    } else {
        mesh.tri[nb] = [apex_t, apex_n, s2];
        mesh.link(nb, 0, n_opp_s1);
        mesh.link(nb, 1, t_opp_s1);
        mesh.adj[nb][2] = t as u32;
    }
    Some(FlipLog {
        old_diag: [s1, s2],
        new_diag: [apex_t, apex_n],
    })
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn has_edge(tris: &[[usize; 3]], a: usize, b: usize) -> bool {
        tris.iter().any(|t| {
            (t.contains(&a) && t.contains(&b))
                && (0..3).any(|i| {
                    let (x, y) = (t[(i + 1) % 3], t[(i + 2) % 3]);
                    (x == a && y == b) || (x == b && y == a)
                })
        })
    }

    fn area(pts: &[[f64; 2]], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| orient2d(pts[t[0]], pts[t[1]], pts[t[2]]).abs() * 0.5)
            .sum()
    }

    #[test]
    fn square_constrained_diagonal() {
        // Unit square; constrain the 0-2 diagonal (Delaunay is ambiguous here).
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let segs = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
        let tris = triangulate(&pts, &segs);
        assert_eq!(tris.len(), 2, "square splits into 2 triangles");
        assert!(
            has_edge(&tris, 0, 2),
            "the constrained 0-2 diagonal is present"
        );
        assert!(
            (area(&pts, &tris) - 1.0).abs() < 1e-9,
            "covers the unit square"
        );
    }

    #[test]
    fn reflex_l_shape() {
        // An L-shaped (non-convex) polygon - the case 3-D flips cannot do.
        //  (0,2)6  5(1,2)
        //    |      |
        //  (0,1)7   4(1,1)___3(2,1)
        //    |                |
        //  (0,0)0__1(1,0)__2(2,0)
        let pts = [
            [0.0, 0.0], // 0
            [1.0, 0.0], // 1
            [2.0, 0.0], // 2
            [2.0, 1.0], // 3
            [1.0, 1.0], // 4
            [1.0, 2.0], // 5
            [0.0, 2.0], // 6
            [0.0, 1.0], // 7
        ];
        let segs = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
        ];
        let tris = triangulate(&pts, &segs);
        // L-shape area = 3 (a 2x2 square minus a 1x1 corner).
        assert!(
            (area(&pts, &tris) - 3.0).abs() < 1e-9,
            "covers exactly the L region (area 3), got {}",
            area(&pts, &tris)
        );
        // Every boundary segment must be an edge.
        for s in &segs {
            assert!(
                has_edge(&tris, s[0], s[1]),
                "boundary segment {s:?} recovered"
            );
        }
        // The reflex vertex is 4; no triangle should poke into the missing corner
        // (x>1 && y>1) - checked via total area above (would exceed 3 if it did).
    }

    #[test]
    fn collinear_is_empty() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert!(triangulate(&pts, &[]).is_empty());
    }

    /// Replay a flip log on a triangle SET (sorted triples): each flip must
    /// find its pre-flip pair present and replaces it by the post-flip pair -
    /// exactly the simulation the 3-D caller runs before committing a stack.
    fn replay(start: &HashSet<[u32; 3]>, log: &[FlipLog]) -> Option<HashSet<[u32; 3]>> {
        let mut cur = start.clone();
        let key = |a: u32, b: u32, c: u32| {
            let mut s = [a, b, c];
            s.sort_unstable();
            s
        };
        for f in log {
            let [s1, s2] = f.old_diag;
            let [x, y] = f.new_diag;
            if !cur.remove(&key(s1, s2, x)) || !cur.remove(&key(s1, s2, y)) {
                return None;
            }
            if !cur.insert(key(x, y, s1)) || !cur.insert(key(x, y, s2)) {
                return None;
            }
        }
        Some(cur)
    }

    #[test]
    fn from_triangulation_and_logged_recovery() {
        // Unit square triangulated with the 1-3 diagonal; recover 0-2 and check
        // the log replays the transformation exactly.
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tris = [[0, 1, 3], [1, 2, 3]];
        let mut mesh = Mesh::from_triangulation(&pts, &tris).expect("valid triangulation");
        assert!(mesh.has_edge(1, 3));
        assert!(!mesh.has_edge(0, 2));
        let start = mesh.triangles_sorted();
        let mut log = Vec::new();
        assert!(recover_edge_logged(&mut mesh, 0, 2, &mut log));
        assert!(mesh.has_edge(0, 2));
        assert_eq!(log.len(), 1, "one diagonal flip");
        assert_eq!(
            replay(&start, &log).expect("log replays cleanly"),
            mesh.triangles_sorted()
        );
    }

    #[test]
    fn from_triangulation_rejects_degenerate() {
        // Zero-area triangle (collinear) must be rejected.
        let pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert!(Mesh::from_triangulation(&pts, &[[0, 1, 2]]).is_none());
        // An edge shared by three triangles must be rejected.
        let pts = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0], [0.5, 2.0]];
        assert!(Mesh::from_triangulation(&pts, &[[0, 1, 2], [0, 1, 3], [0, 1, 4]]).is_none());
    }

    #[test]
    fn logged_recovery_with_collinear_lattice() {
        // A 3x3 lattice (collinear triples everywhere - the flat-cap
        // configuration). Recover the long diagonal of the lower-left 2x1
        // strip: (0,0)-(2,1). Vertex layout:
        //   6 7 8
        //   3 4 5
        //   0 1 2
        let pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ];
        let tris = [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ];
        let mut mesh = Mesh::from_triangulation(&pts, &tris).expect("valid lattice");
        let start = mesh.triangles_sorted();
        let mut log = Vec::new();
        // (0,0)-(2,1) passes strictly between vertices (no on-segment vertex).
        assert!(recover_edge_logged(&mut mesh, 0, 5, &mut log));
        assert!(mesh.has_edge(0, 5));
        assert_eq!(
            replay(&start, &log).expect("log replays cleanly"),
            mesh.triangles_sorted()
        );
        // (0,0)-(2,2) passes exactly THROUGH vertex 4. Under the SoS
        // predicates the recovery "succeeds" symbolically (the edge threads
        // past the coincident vertex) - which is exactly why CALLERS must
        // pre-detect on-segment vertices exactly (coplanar.rs
        // on_segment_vertex → SplitSegment) before invoking recovery. Here we
        // just pin the symbolic behaviour: it terminates and the log replays.
        let mut mesh2 = Mesh::from_triangulation(&pts, &tris).expect("valid lattice");
        let start2 = mesh2.triangles_sorted();
        let mut log2 = Vec::new();
        let ok = recover_edge_logged(&mut mesh2, 0, 8, &mut log2);
        assert_eq!(
            replay(&start2, &log2).expect("log replays cleanly"),
            mesh2.triangles_sorted()
        );
        assert!(ok, "SoS recovery threads past the on-segment vertex");
    }

    #[test]
    fn lshaped_cap_cascade_recovery() {
        // The exact dumped region where recovery of (13,15) stalled on
        // LShaped@1.25 (issue #31): geometric-grading cascades make the first
        // crossing edges unflippable (collinear quads) while a later one IS
        // flippable. The recovery must find it.
        let pts: Vec<[f64; 2]> = vec![
            [-5.0, -5.0],
            [-5.0, 5.0],
            [5.0, -5.0],
            [5.0, -2.0],
            [-2.0, -2.0],
            [-2.0, 5.0],
            [-3.5, 1.5],
            [1.5, -3.5],
            [-4.25, -1.75],
            [-1.75, -4.25],
            [-4.625, 1.625],
            [1.625, -4.625],
            [-4.8125, -1.6875],
            [-1.6875, -4.8125],
            [-4.90625, 1.65625],
            [1.65625, -4.90625],
            [-4.953125, -1.671875],
            [-1.671875, -4.953125],
            [-4.9765625, 1.6640625],
            [1.6640625, -4.9765625],
            [-4.98828125, -1.66796875],
            [-1.66796875, -4.98828125],
            [-4.994140625, 1.666015625],
            [1.666015625, -4.994140625],
            [-4.9970703125, -1.6669921875],
            [-1.6669921875, -4.9970703125],
        ];
        let tris: Vec<[usize; 3]> = vec![
            [0, 1, 24],
            [0, 2, 25],
            [0, 4, 8],
            [0, 4, 9],
            [0, 8, 12],
            [0, 9, 13],
            [0, 12, 16],
            [0, 13, 17],
            [0, 16, 20],
            [0, 17, 21],
            [0, 20, 24],
            [0, 21, 25],
            [1, 5, 6],
            [1, 6, 10],
            [1, 10, 14],
            [1, 14, 18],
            [1, 18, 22],
            [1, 22, 24],
            [2, 3, 7],
            [2, 7, 11],
            [2, 11, 15],
            [2, 15, 19],
            [2, 19, 23],
            [2, 23, 25],
            [3, 4, 5],
            [3, 4, 7],
            [4, 5, 6],
            [4, 6, 8],
            [4, 7, 9],
            [6, 8, 10],
            [7, 9, 11],
            [8, 10, 12],
            [9, 11, 13],
            [10, 12, 16],
            [10, 14, 24],
            [10, 16, 20],
            [10, 20, 24],
            [11, 13, 17],
            [11, 15, 25],
            [11, 17, 21],
            [11, 21, 25],
            [14, 18, 24],
            [15, 19, 25],
            [18, 22, 24],
            [19, 23, 25],
        ];
        let mut mesh = Mesh::from_triangulation(&pts, &tris).expect("valid region");
        let start = mesh.triangles_sorted();
        let mut log = Vec::new();
        assert!(
            recover_edge_logged(&mut mesh, 13, 15, &mut log),
            "the (13,15) cascade chord must recover (flip sequence exists)"
        );
        assert_eq!(
            replay(&start, &log).expect("log replays cleanly"),
            mesh.triangles_sorted()
        );
    }

    #[test]
    fn sheet_to_sheet_transformation_via_edge_recovery() {
        // Two triangulations of the same square fan differing in both quads;
        // recovering every edge of the target reproduces it EXACTLY - the
        // path-2 step of the #31 coplanar-region recovery.
        let pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ];
        let bot = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]];
        let top = [[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];
        let mut mesh = Mesh::from_triangulation(&pts, &bot).expect("valid bot");
        let target = Mesh::from_triangulation(&pts, &top)
            .expect("valid top")
            .triangles_sorted();
        let start = mesh.triangles_sorted();
        let mut log = Vec::new();
        let mut edges: Vec<(u32, u32)> = top
            .iter()
            .flat_map(|t| {
                [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                    .map(|(a, b)| ((a as u32).min(b as u32), (a as u32).max(b as u32)))
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        for (a, b) in edges {
            recover_edge_logged(&mut mesh, a, b, &mut log);
        }
        assert_eq!(
            mesh.triangles_sorted(),
            target,
            "ends exactly at the target sheet"
        );
        assert_eq!(replay(&start, &log).expect("log replays cleanly"), target);
    }
}
