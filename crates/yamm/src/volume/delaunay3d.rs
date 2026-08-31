#![allow(clippy::needless_range_loop, clippy::doc_lazy_continuation)]
use super::dethash::{HashMap, HashSet};
use super::predicates3d::{in_sphere, orient_3d};
/// Bowyer-Watson 3D Delaunay tetrahedralization.
///
/// Uses an infinite vertex approach (INFINITE = usize::MAX). Hull tets always
/// have INFINITE at vertex position 3. The visibility test for hull tets is
/// orient_3d(v0, v1, v2, p) < 0 (TetGen convention).
///
/// Sentinel for the infinite (ghost) vertex.
pub const INFINITE: usize = usize::MAX;

/// A tetrahedron with adjacency information.
#[derive(Clone, Debug)]
pub struct Tet {
    pub verts: [usize; 4],
    pub adj: [usize; 4],
}

impl Tet {
    /// Whether this tet is a hull (ghost) tet - one whose fourth
    /// vertex is the symbolic point at infinity.
    #[inline]
    pub fn is_hull(&self) -> bool {
        self.verts[3] == INFINITE
    }
}

/// Fast free-list: O(1) push/pop/contains using a Vec stack + Vec<bool> bitmap.
pub(super) struct FreeList {
    stack: Vec<usize>,
    is_free: Vec<bool>,
}

impl FreeList {
    fn new() -> Self {
        FreeList {
            stack: Vec::new(),
            is_free: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, len: usize) {
        if self.is_free.len() < len {
            self.is_free.resize(len, false);
        }
    }

    #[inline]
    pub(super) fn contains(&self, idx: &usize) -> bool {
        *idx < self.is_free.len() && self.is_free[*idx]
    }

    fn insert(&mut self, idx: usize) {
        self.ensure_capacity(idx + 1);
        if !self.is_free[idx] {
            self.is_free[idx] = true;
            self.stack.push(idx);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        while let Some(idx) = self.stack.pop() {
            if self.is_free[idx] {
                self.is_free[idx] = false;
                return Some(idx);
            }
        }
        None
    }

    #[allow(dead_code)]
    fn iter(&self) -> impl Iterator<Item = &usize> {
        self.stack
            .iter()
            .filter(|&&idx| idx < self.is_free.len() && self.is_free[idx])
    }
}

/// 3D Delaunay tetrahedralization.
pub struct Delaunay3D {
    pub vertices: Vec<[f64; 3]>,
    pub tets: Vec<Tet>,
    #[allow(dead_code)]
    pub(super) ghost_idx: usize,
    /// A point known to be in the interior of the convex hull.
    /// Used for orienting hull tets when no neighbor reference is available.
    interior_seed: [f64; 3],
    pub(super) free_list: FreeList,
    /// Hint for point location: index of last inserted tet (for spatial walk).
    last_inserted_tet: usize,
    // ── Reusable buffers (avoid per-insert allocation) ──
    /// Epoch-based visited tracking: visited iff `visited_epoch[ti] == current_epoch`.
    visited_epoch: Vec<u32>,
    current_epoch: u32,
    /// Epoch-based cavity membership (replaces HashSet in star-shape validation).
    cavity_epoch: Vec<u32>,
    cavity_epoch_counter: u32,
    /// Reusable scratch buffers cleared each insert.
    cavity_buf: Vec<usize>,
    stack_buf: Vec<usize>,
    bfaces_buf: Vec<(usize, [usize; 3])>,
    new_tets_buf: Vec<usize>,
    /// Reusable face-map for inter-new-tet adjacency: matches each internal face
    /// (sorted vertex triple) to the (tet_idx, face_idx) that first registered
    /// it, so the second occurrence links the pair. Only ever holds the current
    /// insert's new-tet faces (≈ a handful), so a flat `Vec` linear-scanned is
    /// both faster and far more cache-friendly than a `HashMap` here - no
    /// hashing, contiguous memory, and O(1) `clear()` that never retains an
    /// over-large capacity from a one-off big cavity.
    face_map_buf: Vec<([usize; 3], (usize, usize))>,
    /// Vertex→incident-tet adjacency index. `vert_tets[v]` lists the indices of
    /// every LIVE tet (finite AND hull) that has finite vertex `v`. Lets the
    /// boundary-recovery edge/face/ring existence queries run in O(degree)
    /// instead of O(#tets) - the cost that made the conforming path too slow on
    /// fine meshes (issue #30).
    ///
    /// INACTIVE (empty) during base construction: `new()` builds via `insert()`
    /// (Bowyer-Watson), which is hot and runs *before* the index exists, so it
    /// pays nothing. The recovery driver calls `build_vert_tets()` once after
    /// construction to activate it; thereafter it is maintained incrementally at
    /// the alloc/free/push_vertex choke points - the ONLY sites that change a
    /// slot's vertex set (there are no in-place `verts[]` writes anywhere, and
    /// recovery mutates exclusively through `alloc_tet`/`free_tet`). The INFINITE
    /// (ghost) vertex is never indexed.
    vert_tets: Vec<Vec<u32>>,
}

impl Delaunay3D {
    fn empty_with_vertices(vertices: Vec<[f64; 3]>) -> Self {
        Delaunay3D {
            vertices,
            tets: vec![],
            ghost_idx: INFINITE,
            interior_seed: [0.; 3],
            free_list: FreeList::new(),
            last_inserted_tet: 0,
            visited_epoch: Vec::new(),
            current_epoch: 0,
            cavity_epoch: Vec::new(),
            cavity_epoch_counter: 0,
            cavity_buf: Vec::new(),
            stack_buf: Vec::new(),
            bfaces_buf: Vec::new(),
            new_tets_buf: Vec::new(),
            face_map_buf: Vec::new(),
            vert_tets: Vec::new(),
        }
    }

    /// Build an incremental Delaunay tetrahedralization seeded from
    /// the first non-degenerate tetrahedron of `points`.
    pub fn new(points: &[[f64; 3]]) -> Self {
        if points.len() < 4 {
            return Self::empty_with_vertices(points.to_vec());
        }
        let (mut dt, seeds) = Self::init_first4(points);
        // Insert the remaining points in SPATIAL (Morton / Z-order) order rather
        // than input order. Bowyer-Watson locates each new point's containing tet
        // by walking from the previously-inserted tet (`last_inserted_tet`). In
        // input order, consecutive boundary vertices can be far apart spatially,
        // so the walk overruns its step budget and falls back to a full O(#tets)
        // linear scan - making the whole build O(n^2) (e.g. 116s for ~184k pts).
        // Z-ordering keeps successive insertions spatially adjacent, so the walk
        // is short and the build is near-linear. The vertex indices and the
        // `vertices` array are untouched (we only permute the insertion ORDER),
        // so geometry/indexing is identical; only walk locality improves.
        let order = morton_order(points);
        for &i in &order {
            if seeds.contains(&i) {
                continue;
            }
            dt.insert(i);
        }
        dt
    }

    fn init_first4(points: &[[f64; 3]]) -> (Self, [usize; 4]) {
        let n = points.len();
        let (i0, i1) = farthest_pair(points);
        let dir = sub(points[i1], points[i0]);
        let mut i2 = 0;
        let mut best = 0.0f64;
        for i in 0..n {
            if i == i0 || i == i1 {
                continue;
            }
            let d = dist_to_line_sq(points[i], points[i0], &dir);
            if d > best {
                best = d;
                i2 = i;
            }
        }
        let e1 = sub(points[i1], points[i0]);
        let e2 = sub(points[i2], points[i0]);
        let normal = cross(&e1, &e2);
        let mut i3 = 0;
        best = 0.0;
        for i in 0..n {
            if i == i0 || i == i1 || i == i2 {
                continue;
            }
            let v = sub(points[i], points[i0]);
            let d = dot(&v, &normal).abs();
            if d > best {
                best = d;
                i3 = i;
            }
        }

        let mut rv = [i0, i1, i2, i3];
        if orient_3d(points[rv[0]], points[rv[1]], points[rv[2]], points[rv[3]]) < 0.0 {
            rv.swap(2, 3);
        }

        let vertices: Vec<[f64; 3]> = points.to_vec();

        // Real tet (index 0) + 4 hull tets (indices 1-4).
        // Convention: INFINITE always at position 3 in hull tets.
        // For hull tet at face fi: vertices are the 3 face vertices at [0,1,2] + INFINITE at [3].
        // The face from opposite_face(rv, fi) has the real tet's vertex fi on the POSITIVE side.
        // For the hull tet, INFINITE (exterior) should also be on the positive side.
        // opposite_face returns a face where rv[fi] is on positive side.
        // orient_3d(face[0], face[1], face[2], rv[fi]) > 0.
        // We need orient_3d(hull[0], hull[1], hull[2], INFINITE_side_point) > 0.
        // Since INFINITE is at position 3 and represents exterior (same side as... hmm,
        // we can't compute orient_3d with INFINITE).
        //
        // TetGen convention: orient_3d(v0, v1, v2, p) < 0 means p is visible (exterior).
        // So for hull tet [v0, v1, v2, INFINITE], we need interior points to have
        // orient_3d(v0, v1, v2, interior_point) > 0.
        // The interior point is rv[fi] (the real tet's vertex opposite this face).
        // We need orient_3d(hull[0], hull[1], hull[2], rv[fi]) > 0.
        // face = opposite_face(rv, fi) gives us orient_3d(face[0], face[1], face[2], rv[fi]) > 0.
        // So hull[0..2] = face[0..2] directly works!

        let mut tets = vec![Tet {
            verts: rv,
            adj: [usize::MAX; 4],
        }];
        for fi in 0..4 {
            let face = opposite_face(rv, fi);
            let mut hv = [face[0], face[1], face[2], INFINITE];
            // Ensure interior (rv[fi]) is on the NEGATIVE side of hull face.
            // This makes outside points orient > 0 → in conflict.
            let o = orient_3d(
                vertices[hv[0]],
                vertices[hv[1]],
                vertices[hv[2]],
                vertices[rv[fi]],
            );
            if o > 0.0 {
                hv.swap(0, 1);
            } // flip so interior is on negative side
            tets.push(Tet {
                verts: hv,
                adj: [usize::MAX; 4],
            });
        }
        build_adjacency(&mut tets);

        // Interior seed: centroid of the 4 seed points (guaranteed inside the convex hull)
        let interior_seed = [
            (vertices[i0][0] + vertices[i1][0] + vertices[i2][0] + vertices[i3][0]) / 4.0,
            (vertices[i0][1] + vertices[i1][1] + vertices[i2][1] + vertices[i3][1]) / 4.0,
            (vertices[i0][2] + vertices[i1][2] + vertices[i2][2] + vertices[i3][2]) / 4.0,
        ];

        (
            {
                let mut dt = Self::empty_with_vertices(vertices);
                dt.tets = tets;
                dt.interior_seed = interior_seed;
                dt
            },
            [i0, i1, i2, i3],
        )
    }

    #[allow(dead_code)]
    pub(super) fn insert_steiner_point(&mut self, point: [f64; 3]) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(point);
        self.insert(idx);
        idx
    }

    fn insert(&mut self, idx: usize) {
        // Bowyer-Watson insert runs ONLY during base construction (`new()`),
        // before the incidence index is built - it mutates `self.tets` /
        // `free_list` directly (not via alloc_tet/free_tet), so it intentionally
        // does NOT maintain `vert_tets`. The index is built afterwards by
        // `build_vert_tets()`. This invariant keeps base construction allocation-
        // free; assert it so a future caller can't silently desync the index.
        debug_assert!(
            self.vert_tets.is_empty(),
            "insert() must not run with the incidence index active"
        );
        let p = self.vertices[idx];
        let n_tets = self.tets.len();
        self.free_list.ensure_capacity(n_tets);

        let start = self.find_containing_tet(p);
        if start == usize::MAX {
            return;
        }

        // ── Epoch-based visited tracking (no allocation) ──
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            self.visited_epoch.fill(0);
            self.current_epoch = 1;
        }
        self.visited_epoch.resize(n_tets, 0);

        // ── Cavity BFS (reuse buffers) ──
        self.cavity_buf.clear();
        self.stack_buf.clear();
        self.stack_buf.push(start);
        let epoch = self.current_epoch;
        while let Some(ti) = self.stack_buf.pop() {
            if ti == usize::MAX || ti >= self.visited_epoch.len() || self.visited_epoch[ti] == epoch
            {
                continue;
            }
            self.visited_epoch[ti] = epoch;
            if self.tet_conflicts(ti, p) {
                self.cavity_buf.push(ti);
                for &ni in &self.tets[ti].adj {
                    if ni != usize::MAX
                        && (ni >= self.visited_epoch.len() || self.visited_epoch[ni] != epoch)
                    {
                        self.stack_buf.push(ni);
                    }
                }
            }
        }
        if self.cavity_buf.is_empty() {
            return;
        }

        // ── Epoch-based cavity membership (replaces HashSet) ──
        self.cavity_epoch_counter = self.cavity_epoch_counter.wrapping_add(1);
        if self.cavity_epoch_counter == 0 {
            self.cavity_epoch.fill(0);
            self.cavity_epoch_counter = 1;
        }
        self.cavity_epoch.resize(self.tets.len(), 0);
        let cepoch = self.cavity_epoch_counter;
        for &ti in &self.cavity_buf {
            self.cavity_epoch[ti] = cepoch;
        }

        // ── Star-shape cavity repair by EXPANSION (never cut, never drop) ──
        //
        // With EXACT predicates the in-sphere conflict cavity is already
        // star-shaped from p, so every finite boundary face yields a non-flat
        // new tet [face, p] - EXCEPT when p lies exactly in a boundary face's
        // plane (orient_3d(face, p) == 0). That single face would spawn a
        // degenerate (zero-volume) tet. The CORRECT repair is NOT to remove the
        // cavity tet (the old code's cut, which cascades to an empty cavity and
        // silently drops p). Instead we ABSORB the neighbour tet across that
        // coplanar face into the cavity, so p re-connects to the neighbour's
        // (non-coplanar) faces and the on-face degeneracy is dissolved. This
        // only ever GROWS the cavity, so it can never empty and never drops a
        // distinct point. See tetgen insertpoint() (Bowyer-Watson + ONFACE).
        //
        // When the coplanar boundary face's neighbour is the exterior (a hull
        // tet, i.e. p lies exactly on the convex-hull boundary plane), we absorb
        // that HULL tet into the cavity too: the flat finite `face` then stops
        // being a boundary face, and p re-attaches through the hull tet's other
        // (hull) faces as a new hull vertex. This keeps the expansion purely
        // additive and hull-aware - p is still never dropped.
        loop {
            let mut grow_target = usize::MAX;
            for ci in 0..self.cavity_buf.len() {
                let ti = self.cavity_buf[ci];
                if self.cavity_epoch[ti] != cepoch {
                    continue;
                }
                for fi in 0..4 {
                    let nb = self.tets[ti].adj[fi];
                    let nb_in_cavity = nb != usize::MAX
                        && self.cavity_epoch.get(nb).copied().unwrap_or(0) == cepoch;
                    if nb_in_cavity {
                        continue; // interior face
                    }
                    let face = opposite_face(self.tets[ti].verts, fi);
                    if face.contains(&INFINITE) {
                        continue; // hull face: a flat hull child is impossible here
                    }
                    let o = orient_3d(
                        self.vertices[face[0]],
                        self.vertices[face[1]],
                        self.vertices[face[2]],
                        p,
                    );
                    if o == 0.0 {
                        // p is coplanar with this boundary face → flat new tet.
                        // Absorb the neighbour across it (finite OR hull) so the
                        // degeneracy is dissolved. A live neighbour always exists
                        // (every face of a live tet has a recorded neighbour, the
                        // hull tet being the exterior one); if for any reason it
                        // is missing, skip - the new-tet builder drops only this
                        // single flat child, never the whole point.
                        if nb != usize::MAX && nb < self.tets.len() && !self.free_list.contains(&nb)
                        {
                            grow_target = nb;
                            break;
                        }
                    }
                }
                if grow_target != usize::MAX {
                    break;
                }
            }
            if grow_target == usize::MAX {
                break;
            }
            // Absorb the neighbour across the coplanar face into the cavity.
            self.cavity_epoch.resize(self.tets.len(), 0);
            if self.cavity_epoch[grow_target] != cepoch {
                self.cavity_epoch[grow_target] = cepoch;
                self.cavity_buf.push(grow_target);
            }
        }

        // ── Collect boundary faces (reuse buffer) ──
        self.bfaces_buf.clear();
        for &ti in &self.cavity_buf {
            if self.cavity_epoch[ti] != cepoch {
                continue;
            }
            for fi in 0..4 {
                let nb = self.tets[ti].adj[fi];
                if nb == usize::MAX || self.cavity_epoch.get(nb).copied().unwrap_or(0) != cepoch {
                    self.bfaces_buf
                        .push((nb, opposite_face(self.tets[ti].verts, fi)));
                }
            }
        }
        if self.bfaces_buf.is_empty() {
            return;
        }

        // Free cavity tets
        for &ti in &self.cavity_buf {
            if self.cavity_epoch[ti] == cepoch {
                self.free_list.insert(ti);
            }
        }

        // ── Create new tets with face-map adjacency ──
        self.new_tets_buf.clear();
        self.face_map_buf.clear();

        // Take bfaces out to avoid borrow conflict
        let bfaces = std::mem::take(&mut self.bfaces_buf);

        for &(nb, ref face) in &bfaces {
            let is_hull_face = face.contains(&INFINITE);

            let v = if is_hull_face {
                let mut reals = [0usize; 2];
                let mut ri = 0;
                for &fv in face {
                    if fv != INFINITE {
                        if ri < 2 {
                            reals[ri] = fv;
                        }
                        ri += 1;
                    }
                }
                if ri != 2 {
                    continue;
                }
                let mut hv = [reals[0], reals[1], idx, INFINITE];
                let o = orient_3d(
                    self.vertices[hv[0]],
                    self.vertices[hv[1]],
                    self.vertices[hv[2]],
                    self.interior_seed,
                );
                if o > 0.0 {
                    hv.swap(0, 1);
                }
                hv
            } else {
                let mut fv = [face[0], face[1], face[2], idx];
                let o = orient_3d(
                    self.vertices[fv[0]],
                    self.vertices[fv[1]],
                    self.vertices[fv[2]],
                    self.vertices[fv[3]],
                );
                if o < 0.0 {
                    fv.swap(0, 1);
                } else if o == 0.0 {
                    continue;
                }
                fv
            };

            let tet = Tet {
                verts: v,
                adj: [usize::MAX; 4],
            };
            let ti = if let Some(free) = self.free_list.pop() {
                self.tets[free] = tet;
                free
            } else {
                self.tets.push(tet);
                self.tets.len() - 1
            };

            // Adjacency to outer neighbor
            if nb != usize::MAX {
                for fi in 0..4 {
                    if faces_match(&opposite_face(self.tets[nb].verts, fi), face) {
                        self.tets[nb].adj[fi] = ti;
                        for fi2 in 0..4 {
                            if faces_match(&opposite_face(self.tets[ti].verts, fi2), face) {
                                self.tets[ti].adj[fi2] = nb;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            // Face-map: register this tet's internal faces for O(1) adjacency linking
            for fi2 in 0..4 {
                if self.tets[ti].adj[fi2] != usize::MAX {
                    continue; // already linked to outer neighbor
                }
                let mut sf = opposite_face(self.tets[ti].verts, fi2);
                sf.sort();
                if let Some(pos) = self.face_map_buf.iter().position(|&(f, _)| f == sf) {
                    // Found matching face from a previously created new tet
                    let (_, (other_ti, other_fi)) = self.face_map_buf.swap_remove(pos);
                    self.tets[ti].adj[fi2] = other_ti;
                    self.tets[other_ti].adj[other_fi] = ti;
                } else {
                    self.face_map_buf.push((sf, (ti, fi2)));
                }
            }

            self.new_tets_buf.push(ti);
        }

        // Put bfaces buffer back
        self.bfaces_buf = bfaces;

        // Update spatial walk hint: pick the first finite new tet
        for &ti in &self.new_tets_buf {
            if !self.tets[ti].is_hull() {
                self.last_inserted_tet = ti;
                break;
            }
        }
    }

    /// Conflict test. Hull tets: orient_3d(v0,v1,v2,p) < 0.
    fn tet_conflicts(&self, ti: usize, p: [f64; 3]) -> bool {
        let t = &self.tets[ti];
        if t.is_hull() {
            // INFINITE at position 3. Vertices [0,1,2] are the hull face.
            // Interior is on positive side (orient_3d(v0,v1,v2,interior) > 0).
            // Visible (exterior, in conflict) if orient_3d(v0,v1,v2,p) < 0.
            // Outside points are on the POSITIVE side of the hull face
            orient_3d(
                self.vertices[t.verts[0]],
                self.vertices[t.verts[1]],
                self.vertices[t.verts[2]],
                p,
            ) > 0.0
        } else {
            let (a, b, c, d) = (
                self.vertices[t.verts[0]],
                self.vertices[t.verts[1]],
                self.vertices[t.verts[2]],
                self.vertices[t.verts[3]],
            );
            let o = orient_3d(a, b, c, d);
            if o > 0.0 {
                in_sphere(a, b, c, d, p) > 0.0
            } else if o < 0.0 {
                in_sphere(a, c, b, d, p) > 0.0
            } else {
                true
            }
        }
    }

    fn find_containing_tet(&self, p: [f64; 3]) -> usize {
        // Start from the last inserted tet (spatial locality hint from Hilbert-sorted points)
        let mut current = usize::MAX;
        let hint = self.last_inserted_tet;
        if hint < self.tets.len() && !self.free_list.contains(&hint) && !self.tets[hint].is_hull() {
            current = hint;
        }
        if current == usize::MAX {
            for i in 0..self.tets.len() {
                if !self.free_list.contains(&i) && !self.tets[i].is_hull() {
                    current = i;
                    break;
                }
            }
        }
        if current == usize::MAX {
            for i in 0..self.tets.len() {
                if !self.free_list.contains(&i) {
                    return i;
                }
            }
            return usize::MAX;
        }

        let max_steps = self.tets.len() * 2;
        for _ in 0..max_steps {
            if self.free_list.contains(&current) {
                let mut found = false;
                for &a in &self.tets[current].adj {
                    if a != usize::MAX && !self.free_list.contains(&a) {
                        current = a;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return usize::MAX;
                }
                continue;
            }
            let t = &self.tets[current];
            if t.is_hull() {
                return current;
            } // outside hull → start cavity from here

            let v = t.verts;
            let (a, b, c, d) = (
                self.vertices[v[0]],
                self.vertices[v[1]],
                self.vertices[v[2]],
                self.vertices[v[3]],
            );
            let (o0, o1, o2, o3) = (
                orient_3d(p, b, c, d),
                orient_3d(a, p, c, d),
                orient_3d(a, b, p, d),
                orient_3d(a, b, c, p),
            );

            if o0 >= 0.0 && o1 >= 0.0 && o2 >= 0.0 && o3 >= 0.0 {
                return current;
            }

            let min_o = o0.min(o1).min(o2).min(o3);
            let fi = if o0 == min_o {
                0
            } else if o1 == min_o {
                1
            } else if o2 == min_o {
                2
            } else {
                3
            };
            let next = t.adj[fi];
            if next == usize::MAX || next == current {
                return self.find_containing_tet_linear(p);
            }
            current = next;
        }
        self.find_containing_tet_linear(p)
    }

    fn find_containing_tet_linear(&self, p: [f64; 3]) -> usize {
        // Try finite tets first
        for (i, t) in self.tets.iter().enumerate() {
            if self.free_list.contains(&i) || t.is_hull() {
                continue;
            }
            let v = t.verts;
            let (a, b, c, d) = (
                self.vertices[v[0]],
                self.vertices[v[1]],
                self.vertices[v[2]],
                self.vertices[v[3]],
            );
            if orient_3d(p, b, c, d) >= 0.0
                && orient_3d(a, p, c, d) >= 0.0
                && orient_3d(a, b, p, d) >= 0.0
                && orient_3d(a, b, c, p) >= 0.0
            {
                return i;
            }
        }
        // Outside hull: find a conflicting hull tet
        for (i, t) in self.tets.iter().enumerate() {
            if self.free_list.contains(&i) || !t.is_hull() {
                continue;
            }
            if self.tet_conflicts(i, p) {
                return i;
            }
        }
        usize::MAX
    }

    /// Whether tet slot `ti` is in range and not on the free list
    /// (i.e. a currently-valid tet, not a recycled hole).
    pub fn is_live(&self, ti: usize) -> bool {
        ti < self.tets.len() && !self.free_list.contains(&ti)
    }

    /// Build a correctly-oriented hull tet for the finite boundary face
    /// `[f0, f1, f2]`.
    ///
    /// Convention (see `init_first4` and `tet_conflicts`): a hull tet
    /// `[v0, v1, v2, INFINITE]` keeps INFINITE at slot 3 and the finite face
    /// oriented so that interior points are on the NEGATIVE side, i.e.
    /// `orient_3d(v0, v1, v2, interior) < 0` (outside points → `> 0`). This is
    /// the combinatorial replacement for `orient_3d` when the apex is INFINITE.
    pub(super) fn oriented_hull_tet(&self, f0: usize, f1: usize, f2: usize) -> [usize; 4] {
        let mut hv = [f0, f1, f2, INFINITE];
        let o = orient_3d(
            self.vertices[f0],
            self.vertices[f1],
            self.vertices[f2],
            self.interior_seed,
        );
        if o > 0.0 {
            hv.swap(0, 1);
        }
        hv
    }

    /// Allocate a tet into a free slot (reused) or a fresh one, and
    /// register its finite vertices in the incidence index.
    pub fn alloc_tet(&mut self, tet: Tet) -> usize {
        let ti = if let Some(free) = self.free_list.pop() {
            self.tets[free] = tet;
            free
        } else {
            self.tets.push(tet);
            self.tets.len() - 1
        };
        // Register the new tet's finite vertices in the incidence index. When a
        // freed slot is reused, its previous occupant was already unregistered
        // by `free_tet`, so this never leaves a stale entry.
        self.vt_register(ti);
        ti
    }

    /// Retire tet slot `ti`: unregister its vertices from the
    /// incidence index and return the slot to the free list.
    pub fn free_tet(&mut self, ti: usize) {
        // Unregister BEFORE the slot can be reused - reads the tet's current
        // (about-to-die) vertices, which are still intact in `self.tets[ti]`.
        self.vt_unregister(ti);
        self.free_list.insert(ti);
    }

    /// Returns true if the vertex→incident-tet index is active (built).
    #[inline]
    pub fn index_active(&self) -> bool {
        !self.vert_tets.is_empty()
    }

    /// Live tets incident to finite vertex `v` (finite AND hull tets). Empty if
    /// the index is inactive or `v` is out of range. Indices are kept live by
    /// the alloc/free maintenance; callers that want only finite tets still
    /// filter `is_hull` themselves (matching the pre-index full-scan behaviour).
    #[inline]
    pub fn incident_tets(&self, v: usize) -> &[u32] {
        match self.vert_tets.get(v) {
            Some(list) => list.as_slice(),
            None => &[],
        }
    }

    /// Add `ti`'s finite vertices to the incidence index (no-op when inactive).
    #[inline]
    fn vt_register(&mut self, ti: usize) {
        if self.vert_tets.is_empty() {
            return;
        }
        for &v in &self.tets[ti].verts {
            if v != INFINITE {
                self.vert_tets[v].push(ti as u32);
            }
        }
    }

    /// Remove `ti` from the incidence index of each of its finite vertices
    /// (no-op when inactive). Must be called while `self.tets[ti]` still holds
    /// the tet's vertices.
    #[inline]
    fn vt_unregister(&mut self, ti: usize) {
        if self.vert_tets.is_empty() {
            return;
        }
        let tu = ti as u32;
        for &v in &self.tets[ti].verts {
            if v != INFINITE {
                let list = &mut self.vert_tets[v];
                if let Some(pos) = list.iter().position(|&x| x == tu) {
                    list.swap_remove(pos);
                }
            }
        }
    }

    /// Build (activate) the vertex→incident-tet index from the current mesh.
    /// Call ONCE after base construction and before boundary recovery; from then
    /// on it is maintained incrementally at `alloc_tet`/`free_tet`/`push_vertex`.
    /// O(#tets). Idempotent (rebuilds from scratch each call).
    pub fn build_vert_tets(&mut self) {
        let nv = self.vertices.len();
        let mut idx: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for ti in 0..self.tets.len() {
            if self.free_list.contains(&ti) {
                continue;
            }
            for &v in &self.tets[ti].verts {
                if v != INFINITE {
                    idx[v].push(ti as u32);
                }
            }
        }
        self.vert_tets = idx;
    }

    /// Append a new vertex and return its index (does NOT insert it into the
    /// triangulation - used by the local-split Steiner primitives, which place
    /// the vertex by deterministic local re-meshing rather than Bowyer-Watson).
    pub(super) fn push_vertex(&mut self, p: [f64; 3]) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(p);
        // Keep the incidence index aligned with the vertex array when active -
        // Steiner refinement pushes vertices during recovery, then connects them
        // via alloc_tet, which will index into this new (initially empty) slot.
        if !self.vert_tets.is_empty() {
            self.vert_tets.push(Vec::new());
        }
        idx
    }

    /// Insert `point` lying ON the (open) edge (a, b) by a deterministic LOCAL
    /// split of every tet in the edge's ring - finite AND hull. This is the
    /// conforming-Delaunay-refinement Steiner primitive: the new vertex `m`
    /// lies on segment (a, b), so each ring tet `{a, b, x, y}` is replaced by
    /// two tets `{m, b, x, y}` (a→m) and `{a, m, x, y}` (b→m); the union covers
    /// the same region (m is on edge a-b, so both sub-tets are non-degenerate
    /// and positively oriented after orientation fixing). Adjacency is rebuilt
    /// conformally: internal new↔new faces via a face-map, external links to the
    /// ring's outer neighbours (the neighbours across the two faces of each ring
    /// tet NOT containing edge (a,b)), all hull-aware.
    ///
    /// Returns the new vertex index, or `None` (mesh unchanged) if the edge has
    /// no ring, the point is not strictly interior to (a,b), or any sub-tet
    /// would be degenerate.
    pub(super) fn split_edge_on_constraint(
        &mut self,
        a: usize,
        b: usize,
        point: [f64; 3],
    ) -> Option<usize> {
        if a == b {
            return None;
        }
        // Collect the ring: every live tet containing both a and b. Only tets
        // incident to `a` can qualify → O(degree) via the index when active;
        // full scan otherwise (e.g. unit tests that skip build_vert_tets).
        let ring: Vec<usize> = if self.index_active() {
            self.incident_tets(a)
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| self.is_live(i) && self.tets[i].verts.contains(&b))
                .collect()
        } else {
            (0..self.tets.len())
                .filter(|&i| {
                    self.is_live(i) && {
                        let v = self.tets[i].verts;
                        v.contains(&a) && v.contains(&b)
                    }
                })
                .collect()
        };
        if ring.is_empty() {
            return None;
        }

        // ── VALIDATION PHASE (no mutation) ──
        // Build the two child tets of every ring tet, oriented; reject the whole
        // split if any child is degenerate (the point is not truly on the edge).
        // Each entry: (child_verts oriented, [the two faces of the parent that do
        // NOT contain a or b respectively - for external relinking]).
        struct Child {
            verts: [usize; 4],
            // External face this child inherits from the parent, with the
            // parent's neighbor across it. `None` for the internal new↔new face
            // handled by the face-map.
            ext_face: [usize; 3],
            ext_neighbor: usize,
        }
        let mut children: Vec<Child> = Vec::with_capacity(ring.len() * 2);
        let m = self.vertices.len(); // tentative new index (pushed after validation)

        // Temporarily make `point` queryable via a closure capturing it for `m`.
        let vert = |idx: usize, m_pt: [f64; 3]| -> [f64; 3] {
            if idx == m {
                m_pt
            } else {
                self.vertices[idx]
            }
        };

        for &ti in &ring {
            let v = self.tets[ti].verts;
            // The two "other" vertices x, y (may include INFINITE for hull tets).
            let mut others = [usize::MAX; 2];
            let mut k = 0;
            for &x in &v {
                if x != a && x != b {
                    if k < 2 {
                        others[k] = x;
                    }
                    k += 1;
                }
            }
            if k != 2 {
                return None;
            }
            let (x, y) = (others[0], others[1]);
            // A ZERO-VOLUME ring tet (a flat-sandwich member, issue #31) is
            // allowed to produce zero-volume children: flat parent → flat
            // children is exactly volume-preserving and keeps the sandwich
            // conformal as the surface refines. Only a POSITIVE parent
            // rejecting a degenerate child signals "point not on the edge".
            let parent_flat = !v.contains(&INFINITE)
                && orient_3d(
                    self.vertices[v[0]],
                    self.vertices[v[1]],
                    self.vertices[v[2]],
                    self.vertices[v[3]],
                ) == 0.0;

            // Locate the parent faces opposite a and opposite b, plus their
            // external neighbors.
            let mut fa_idx = usize::MAX; // face opposite a = {b,x,y}
            let mut fb_idx = usize::MAX; // face opposite b = {a,x,y}
            for fi in 0..4 {
                if v[fi] == a {
                    fa_idx = fi;
                } else if v[fi] == b {
                    fb_idx = fi;
                }
            }
            if fa_idx == usize::MAX || fb_idx == usize::MAX {
                return None;
            }
            let nb_a = self.tets[ti].adj[fa_idx]; // neighbor across {b,x,y}
            let nb_b = self.tets[ti].adj[fb_idx]; // neighbor across {a,x,y}

            // Child 1: a→m → {m, b, x, y}. Inherits the {b,x,y} face (opp a).
            // Child 2: b→m → {a, m, x, y}. Inherits the {a,x,y} face (opp b).
            let c1 = [m, b, x, y];
            let c2 = [a, m, x, y];

            for (cv, ext_face, ext_neighbor) in [(c1, [b, x, y], nb_a), (c2, [a, x, y], nb_b)] {
                // Orient / reject degenerate.
                let oriented = if cv.contains(&INFINITE) {
                    let finite: Vec<usize> =
                        cv.iter().copied().filter(|&z| z != INFINITE).collect();
                    if finite.len() != 3 {
                        return None;
                    }
                    // Reject collinear hull face.
                    let pa = vert(finite[0], point);
                    let pb = vert(finite[1], point);
                    let pc = vert(finite[2], point);
                    let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let cr = cross(&ab, &ac);
                    if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] == 0.0 {
                        return None;
                    }
                    self.oriented_hull_tet_pt(finite[0], finite[1], finite[2], point)
                } else {
                    let mut w = cv;
                    let o = orient_3d(
                        vert(w[0], point),
                        vert(w[1], point),
                        vert(w[2], point),
                        vert(w[3], point),
                    );
                    if o < 0.0 {
                        w.swap(0, 1);
                    } else if o == 0.0 && !parent_flat {
                        return None; // degenerate child → not on the edge
                    }
                    w
                };
                children.push(Child {
                    verts: oriented,
                    ext_face,
                    ext_neighbor,
                });
            }
        }

        // ── COMMIT PHASE ──
        let m = self.push_vertex(point);
        for &ti in &ring {
            self.free_tet(ti);
        }
        let mut new_indices: Vec<usize> = Vec::with_capacity(children.len());
        for c in &children {
            new_indices.push(self.alloc_tet(Tet {
                verts: c.verts,
                adj: [usize::MAX; 4],
            }));
        }

        // Inter-new-tet adjacency via face-map (sorted face → (tet, face_idx)).
        let mut fmap: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
        for &ni in &new_indices {
            for fi in 0..4 {
                let mut sf = opposite_face(self.tets[ni].verts, fi);
                sf.sort();
                if let Some((other_ti, other_fi)) = fmap.remove(&sf) {
                    self.tets[ni].adj[fi] = other_ti;
                    self.tets[other_ti].adj[other_fi] = ni;
                } else {
                    fmap.insert(sf, (ni, fi));
                }
            }
        }

        // External adjacency: each child inherits one outer face from its parent.
        for (k, c) in children.iter().enumerate() {
            let ni = new_indices[k];
            let nb = c.ext_neighbor;
            if nb == usize::MAX || !self.is_live(nb) {
                continue;
            }
            // Find the child's face matching its inherited external face.
            for fi in 0..4 {
                if faces_match(&opposite_face(self.tets[ni].verts, fi), &c.ext_face) {
                    self.tets[ni].adj[fi] = nb;
                    for nfi in 0..4 {
                        if faces_match(&opposite_face(self.tets[nb].verts, nfi), &c.ext_face) {
                            self.tets[nb].adj[nfi] = ni;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        Some(m)
    }

    /// Split edge (a, b) at the EXISTING vertex `w`, whose position lies on
    /// the open segment (a, b): every ring tet `{a, b, x, y}` is replaced by
    /// `{a, w, x, y}` and `{w, b, x, y}`. Exactly volume-preserving (w is on
    /// the segment), no new vertices. This resolves the VERTEX-ON-EDGE
    /// degeneracy behind issue #47's flat-cap unproductive class (= #31's
    /// coplanar family): a surface Steiner vertex sits exactly ON an in-plane
    /// edge of the cap triangulation, every crossing query for segments ending
    /// at that vertex degenerates (crossing parameter t == 1 within fp), and
    /// recovery churns midpoint splits forever. Splitting the edge AT the
    /// vertex removes the obstruction topologically.
    ///
    /// Transactional: validates every child (rejecting the whole operation if
    /// any would be degenerate - including the case where a ring tet already
    /// CONTAINS `w`, i.e. a zero-volume sliver) before mutating. Returns true
    /// on success.
    pub(super) fn split_edge_at_vertex(&mut self, a: usize, b: usize, w: usize) -> bool {
        if a == b || w == a || w == b {
            return false;
        }
        let ring: Vec<usize> = if self.index_active() {
            self.incident_tets(a)
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| self.is_live(i) && self.tets[i].verts.contains(&b))
                .collect()
        } else {
            (0..self.tets.len())
                .filter(|&i| {
                    self.is_live(i) && {
                        let v = self.tets[i].verts;
                        v.contains(&a) && v.contains(&b)
                    }
                })
                .collect()
        };
        if ring.is_empty() {
            return false;
        }

        // Partition the ring: a tet that already CONTAINS w is a zero-content
        // sliver spanned by the split (w lies on the open segment (a, b), so
        // {a, b, w, x} has three collinear vertices and decomposes into the
        // two triangles {a,w,x}, {w,b,x} - no tet at all). These are DELETED;
        // their external faces re-glue to the split children (or to each
        // other where the whole face neighbourhood is deleted). This was
        // previously an unconditional transactional bail - the dominant
        // `voe split REJECTED` mode of the issue-#31/#47 flat-cap class.
        let mut kept: Vec<usize> = Vec::new();
        let mut deleted: Vec<usize> = Vec::new();
        for &ti in &ring {
            if self.tets[ti].verts.contains(&w) {
                deleted.push(ti);
            } else {
                kept.push(ti);
            }
        }
        let sev_dbg = std::env::var("YAMM_SEV_DBG").is_ok();
        macro_rules! sbail {
            ($why:expr) => {{
                if sev_dbg {
                    eprintln!(
                        "    [sev DBG] split_edge_at_vertex({a},{b};{w}): bail - {}",
                        $why
                    );
                }
                return false;
            }};
        }
        if !deleted.is_empty() {
            // Deleting the slivers is volume-preserving up to the SAME dust
            // the split itself accepts: require w on the line of (a, b) within
            // the caller's own criterion (dist² ≤ |ab|²·1e-24, the
            // vertex-on-edge trigger tolerance). Each {a,b,w,x} sliver's
            // volume is then bounded by that dust. INFINITE-containing
            // "slivers" are out of model.
            if deleted
                .iter()
                .any(|&ti| self.tets[ti].verts.contains(&INFINITE))
            {
                sbail!("hull tet contains w");
            }
            let (pa, pb, pw) = (self.vertices[a], self.vertices[b], self.vertices[w]);
            let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let pa_pw = [pw[0] - pa[0], pw[1] - pa[1], pw[2] - pa[2]];
            let t_on = if l2 > 0.0 {
                (pa_pw[0] * d[0] + pa_pw[1] * d[1] + pa_pw[2] * d[2]) / l2
            } else {
                -1.0
            };
            let proj = [
                pa[0] + t_on * d[0],
                pa[1] + t_on * d[1],
                pa[2] + t_on * d[2],
            ];
            let dist2 =
                (pw[0] - proj[0]).powi(2) + (pw[1] - proj[1]).powi(2) + (pw[2] - proj[2]).powi(2);
            if !(t_on > 0.0 && t_on < 1.0) || dist2 > l2 * 1e-24 {
                sbail!("w not on the open edge within tolerance");
            }
        }
        if kept.is_empty() {
            sbail!("entire ring is degenerate"); // nothing to realize the split with
        }
        // External faces of the deleted cluster: every face of a deleted tet
        // that does not carry the split edge (those are zero-area internal
        // interfaces) and whose neighbour is outside the cluster. Grouped by
        // face key: a singleton re-glues to the unique child owning that
        // face; a pair means the cluster separated two outside tets that now
        // glue directly. Anything else is a configuration we don't model.
        let del_set: HashSet<usize> = deleted.iter().copied().collect();
        let mut patch_map: HashMap<[usize; 3], Vec<usize>> = HashMap::default();
        for &ti in &deleted {
            let v = self.tets[ti].verts;
            for fi in 0..4 {
                let f = opposite_face(v, fi);
                if f.contains(&a) && f.contains(&b) {
                    continue;
                }
                let nb = self.tets[ti].adj[fi];
                if nb != usize::MAX && del_set.contains(&nb) {
                    continue;
                }
                let mut key = f;
                key.sort_unstable();
                patch_map.entry(key).or_default().push(nb);
            }
        }

        struct Child {
            verts: [usize; 4],
            ext_face: [usize; 3],
            ext_neighbor: usize,
        }
        let mut children: Vec<Child> = Vec::with_capacity(kept.len() * 2);
        for &ti in &kept {
            let v = self.tets[ti].verts;
            // Flat parent → flat children allowed (issue #31 sandwich; see
            // split_edge_on_constraint).
            let parent_flat = !v.contains(&INFINITE)
                && orient_3d(
                    self.vertices[v[0]],
                    self.vertices[v[1]],
                    self.vertices[v[2]],
                    self.vertices[v[3]],
                ) == 0.0;
            let mut others = [usize::MAX; 2];
            let mut k = 0;
            for &x in &v {
                if x != a && x != b {
                    if k < 2 {
                        others[k] = x;
                    }
                    k += 1;
                }
            }
            if k != 2 {
                return false;
            }
            let (x, y) = (others[0], others[1]);
            let mut fa_idx = usize::MAX;
            let mut fb_idx = usize::MAX;
            for fi in 0..4 {
                if v[fi] == a {
                    fa_idx = fi;
                } else if v[fi] == b {
                    fb_idx = fi;
                }
            }
            if fa_idx == usize::MAX || fb_idx == usize::MAX {
                return false;
            }
            let nb_a = self.tets[ti].adj[fa_idx];
            let nb_b = self.tets[ti].adj[fb_idx];
            let c1 = [w, b, x, y];
            let c2 = [a, w, x, y];
            for (cv, ext_face, ext_neighbor) in [(c1, [b, x, y], nb_a), (c2, [a, x, y], nb_b)] {
                let oriented = if cv.contains(&INFINITE) {
                    let finite: Vec<usize> =
                        cv.iter().copied().filter(|&z| z != INFINITE).collect();
                    if finite.len() != 3 {
                        return false;
                    }
                    let pa = self.vertices[finite[0]];
                    let pb = self.vertices[finite[1]];
                    let pc = self.vertices[finite[2]];
                    let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let cr = cross(&ab, &ac);
                    if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] == 0.0 {
                        return false;
                    }
                    self.oriented_hull_tet(finite[0], finite[1], finite[2])
                } else {
                    let mut cw = cv;
                    let o = orient_3d(
                        self.vertices[cw[0]],
                        self.vertices[cw[1]],
                        self.vertices[cw[2]],
                        self.vertices[cw[3]],
                    );
                    if o < 0.0 {
                        cw.swap(0, 1);
                    } else if o == 0.0 && !parent_flat {
                        return false; // degenerate child - w not truly interior
                    }
                    cw
                };
                children.push(Child {
                    verts: oriented,
                    ext_face,
                    ext_neighbor,
                });
            }
        }
        // Validate the patches against the planned children (still no
        // mutation): a singleton patch face must be owned by EXACTLY one
        // child; a pair must join two live outside tets.
        let mut patches: Vec<([usize; 3], Vec<usize>)> = patch_map.into_iter().collect();
        patches.sort_unstable(); // deterministic commit order
        for (key, nbs) in &patches {
            match nbs.len() {
                1 => {
                    let owners = children
                        .iter()
                        .filter(|c| key.iter().all(|z| c.verts.contains(z)))
                        .count();
                    if owners != 1 {
                        sbail!(format!("patch face {key:?} owned by {owners} children"));
                    }
                }
                2 => {
                    if nbs
                        .iter()
                        .any(|&n| n == usize::MAX || !self.is_live(n) || del_set.contains(&n))
                    {
                        sbail!("paired patch with unusable neighbour");
                    }
                }
                _ => sbail!(format!("patch face {key:?} with {} neighbours", nbs.len())),
            }
        }

        // ── COMMIT ──
        for &ti in deleted.iter().chain(kept.iter()) {
            self.free_tet(ti);
        }
        let mut new_indices: Vec<usize> = Vec::with_capacity(children.len());
        for c in &children {
            new_indices.push(self.alloc_tet(Tet {
                verts: c.verts,
                adj: [usize::MAX; 4],
            }));
        }
        let mut fmap: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
        for &ni in &new_indices {
            for fi in 0..4 {
                let mut sf = opposite_face(self.tets[ni].verts, fi);
                sf.sort();
                if let Some((other_ti, other_fi)) = fmap.remove(&sf) {
                    self.tets[ni].adj[fi] = other_ti;
                    self.tets[other_ti].adj[other_fi] = ni;
                } else {
                    fmap.insert(sf, (ni, fi));
                }
            }
        }
        for (k, c) in children.iter().enumerate() {
            let ni = new_indices[k];
            let nb = c.ext_neighbor;
            if nb == usize::MAX || !self.is_live(nb) {
                continue;
            }
            for fi in 0..4 {
                if faces_match(&opposite_face(self.tets[ni].verts, fi), &c.ext_face) {
                    self.tets[ni].adj[fi] = nb;
                    for nfi in 0..4 {
                        if faces_match(&opposite_face(self.tets[nb].verts, nfi), &c.ext_face) {
                            self.tets[nb].adj[nfi] = ni;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        // Re-glue the deleted cluster's external faces.
        let slot_of = |tets: &Vec<Tet>, ti: usize, key: &[usize; 3]| -> usize {
            (0..4)
                .find(|&fi| {
                    let mut sf = opposite_face(tets[ti].verts, fi);
                    sf.sort();
                    sf == *key
                })
                .unwrap_or(usize::MAX)
        };
        for (key, nbs) in &patches {
            if nbs.len() == 2 {
                // The cluster separated two outside tets - glue them directly.
                let (n1, n2) = (nbs[0], nbs[1]);
                let (s1, s2) = (slot_of(&self.tets, n1, key), slot_of(&self.tets, n2, key));
                if s1 != usize::MAX && s2 != usize::MAX {
                    self.tets[n1].adj[s1] = n2;
                    self.tets[n2].adj[s2] = n1;
                }
            } else {
                // Singleton: glue the unique owning child to the outside tet.
                let ci = (0..children.len())
                    .find(|&k| key.iter().all(|z| children[k].verts.contains(z)))
                    .unwrap();
                let ni = new_indices[ci];
                let sc = slot_of(&self.tets, ni, key);
                let nb = nbs[0];
                if sc != usize::MAX {
                    self.tets[ni].adj[sc] = nb;
                    if nb != usize::MAX && self.is_live(nb) {
                        let sn = slot_of(&self.tets, nb, key);
                        if sn != usize::MAX {
                            self.tets[nb].adj[sn] = ni;
                        }
                    }
                }
            }
        }
        true
    }

    /// Insert `point` lying IN the (open) interior of triangle (a, b, c)'s plane
    /// by a deterministic LOCAL split of the (≤2) tets incident to face
    /// (a, b, c) - finite AND hull. Each incident tet `{a, b, c, apex}` is
    /// replaced by three tets `{a, b, m, apex}`, `{b, c, m, apex}`,
    /// `{c, a, m, apex}` (m on the face, so all three are non-degenerate). The
    /// shared face (a, b, c) becomes three sub-faces (a, b, m), (b, c, m),
    /// (c, a, m) shared between the two incident tets' children. Adjacency is
    /// rebuilt conformally and hull-aware.
    ///
    /// Returns the new vertex index, or `None` (mesh unchanged) if the face has
    /// no incident tet, the point is not strictly interior to the triangle, or
    /// any sub-tet would be degenerate.
    pub(super) fn split_face_on_constraint(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        point: [f64; 3],
    ) -> Option<usize> {
        if a == b || b == c || a == c {
            return None;
        }
        // The (≤2) live tets that own face (a, b, c) as a face. Only tets
        // incident to `a` can contain all three vertices → O(degree) via the
        // index when active; full scan otherwise. For each, the apex is the 4th.
        let mut incident: Vec<(usize, usize)> = Vec::new(); // (tet, apex)
        let candidates: Vec<usize> = if self.index_active() {
            self.incident_tets(a).iter().map(|&i| i as usize).collect()
        } else {
            (0..self.tets.len()).collect()
        };
        for i in candidates {
            if !self.is_live(i) {
                continue;
            }
            let v = self.tets[i].verts;
            if v.contains(&a) && v.contains(&b) && v.contains(&c) {
                // Confirm (a,b,c) is an actual FACE (apex is the remaining vert).
                let apex = v.iter().copied().find(|&z| z != a && z != b && z != c);
                if let Some(apex) = apex {
                    incident.push((i, apex));
                }
            }
        }
        if incident.is_empty() {
            return None;
        }

        let m = self.vertices.len(); // tentative
        let vert = |idx: usize, m_pt: [f64; 3]| -> [f64; 3] {
            if idx == m {
                m_pt
            } else {
                self.vertices[idx]
            }
        };

        struct Child {
            verts: [usize; 4],
            ext_face: [usize; 3],
            ext_neighbor: usize,
        }
        let mut children: Vec<Child> = Vec::new();

        for &(ti, apex) in &incident {
            let v = self.tets[ti].verts;
            // Flat parent → flat children allowed (issue #31 sandwich; see
            // split_edge_on_constraint).
            let parent_flat = !v.contains(&INFINITE)
                && orient_3d(
                    self.vertices[v[0]],
                    self.vertices[v[1]],
                    self.vertices[v[2]],
                    self.vertices[v[3]],
                ) == 0.0;
            // External neighbors across the three faces of {a,b,c,apex} that
            // each contain `apex` (the face opposite apex IS (a,b,c) - the split
            // plane, an internal new↔new interface, not external).
            // Faces: opp a = {b,c,apex}, opp b = {a,c,apex}, opp c = {a,b,apex}.
            let mut nb_a = usize::MAX; // across {b,c,apex}
            let mut nb_b = usize::MAX; // across {a,c,apex}
            let mut nb_c = usize::MAX; // across {a,b,apex}
            for fi in 0..4 {
                let f = opposite_face(v, fi);
                if !f.contains(&apex) {
                    continue; // that's the (a,b,c) face
                }
                let nb = self.tets[ti].adj[fi];
                if !f.contains(&a) {
                    nb_a = nb; // face {b,c,apex}
                } else if !f.contains(&b) {
                    nb_b = nb; // face {a,c,apex}
                } else if !f.contains(&c) {
                    nb_c = nb; // face {a,b,apex}
                }
            }

            // Three children, each carrying ONE edge of the face + m + apex.
            // Child for edge (a,b): {a,b,m,apex}, inherits external face {a,b,apex} (nb_c).
            // Child for edge (b,c): {b,c,m,apex}, inherits {b,c,apex} (nb_a).
            // Child for edge (c,a): {c,a,m,apex}, inherits {c,a,apex} (nb_b).
            for (cv, ext_face, ext_neighbor) in [
                ([a, b, m, apex], [a, b, apex], nb_c),
                ([b, c, m, apex], [b, c, apex], nb_a),
                ([c, a, m, apex], [c, a, apex], nb_b),
            ] {
                let oriented = if cv.contains(&INFINITE) {
                    let finite: Vec<usize> =
                        cv.iter().copied().filter(|&z| z != INFINITE).collect();
                    if finite.len() != 3 {
                        return None;
                    }
                    let pa = vert(finite[0], point);
                    let pb = vert(finite[1], point);
                    let pc = vert(finite[2], point);
                    let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let cr = cross(&ab, &ac);
                    if cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2] == 0.0 {
                        return None;
                    }
                    self.oriented_hull_tet_pt(finite[0], finite[1], finite[2], point)
                } else {
                    let mut w = cv;
                    let o = orient_3d(
                        vert(w[0], point),
                        vert(w[1], point),
                        vert(w[2], point),
                        vert(w[3], point),
                    );
                    if o < 0.0 {
                        w.swap(0, 1);
                    } else if o == 0.0 && !parent_flat {
                        return None; // degenerate child → point not in triangle interior
                    }
                    w
                };
                children.push(Child {
                    verts: oriented,
                    ext_face,
                    ext_neighbor,
                });
            }
        }

        // ── COMMIT PHASE ──
        let m = self.push_vertex(point);
        for &(ti, _) in &incident {
            self.free_tet(ti);
        }
        let mut new_indices: Vec<usize> = Vec::with_capacity(children.len());
        for c in &children {
            new_indices.push(self.alloc_tet(Tet {
                verts: c.verts,
                adj: [usize::MAX; 4],
            }));
        }

        let mut fmap: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
        for &ni in &new_indices {
            for fi in 0..4 {
                let mut sf = opposite_face(self.tets[ni].verts, fi);
                sf.sort();
                if let Some((other_ti, other_fi)) = fmap.remove(&sf) {
                    self.tets[ni].adj[fi] = other_ti;
                    self.tets[other_ti].adj[other_fi] = ni;
                } else {
                    fmap.insert(sf, (ni, fi));
                }
            }
        }

        for (k, c) in children.iter().enumerate() {
            let ni = new_indices[k];
            let nb = c.ext_neighbor;
            if nb == usize::MAX || !self.is_live(nb) {
                continue;
            }
            for fi in 0..4 {
                if faces_match(&opposite_face(self.tets[ni].verts, fi), &c.ext_face) {
                    self.tets[ni].adj[fi] = nb;
                    for nfi in 0..4 {
                        if faces_match(&opposite_face(self.tets[nb].verts, nfi), &c.ext_face) {
                            self.tets[nb].adj[nfi] = ni;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        Some(m)
    }

    /// Insert a Steiner `point` into the tetrahedralization by a deterministic
    /// LOCAL split based on where the point lies - used by conforming-Delaunay
    /// segment recovery, where the point is the (clamped) midpoint of a MISSING
    /// boundary segment and therefore generally falls strictly inside a finite
    /// tet (the segment is straight; its midpoint is interior to the enclosed
    /// region). Classifies the location and dispatches:
    ///   * strictly inside a finite tet → 1-to-4 split,
    ///   * on a finite face of the located tet → `split_face_on_constraint`,
    ///   * on a finite edge of the located tet → `split_edge_on_constraint`.
    /// Hull-aware via the underlying primitives. Returns the new vertex index,
    /// or `None` (mesh unchanged) if the point lands outside the convex hull
    /// (located in a hull tet) or any sub-tet would be degenerate.
    ///
    /// This is NOT generic Bowyer-Watson: it never removes circumsphere-
    /// conflicting tets (which would drop on-boundary points); it performs the
    /// minimal conformal split for the point's exact location.
    pub(super) fn insert_steiner_local(&mut self, point: [f64; 3]) -> Option<usize> {
        let idbg = std::env::var("YAMM_INS_DBG").is_ok();
        macro_rules! ifail {
            ($($arg:tt)*) => {
                if idbg {
                    eprintln!("    [ins DBG] {}", format!($($arg)*));
                }
            };
        }
        let mut ti = self.find_containing_tet(point);
        if ti == usize::MAX || !self.is_live(ti) {
            ifail!("find_containing_tet failed for {point:?}");
            return None;
        }
        // BOUNDARY-ON-HULL case: a Steiner point on a CONVEX part of the surface
        // lies on the convex-hull boundary, so point location lands in a HULL
        // tet. The point then sits on that hull tet's finite face - which is a
        // boundary face shared with a finite tet just inside. Redirect to the
        // finite neighbour and split that shared face.
        if self.tets[ti].is_hull() {
            let hv = self.tets[ti].verts;
            // The finite face is the one NOT containing INFINITE.
            let mut fin = usize::MAX;
            for fi in 0..4 {
                if !opposite_face(hv, fi).contains(&INFINITE) {
                    fin = fi;
                    break;
                }
            }
            if fin == usize::MAX {
                ifail!("hull tet without finite face");
                return None;
            }
            let f = opposite_face(hv, fin);
            // Split the boundary face directly (handles both the hull and the
            // finite incident tet, conformally).
            let r = self.split_face_on_constraint(f[0], f[1], f[2], point);
            if r.is_none() {
                ifail!("hull-redirect split_face_on_constraint({f:?}) failed at {point:?}");
            }
            return r;
        }

        let v = self.tets[ti].verts;
        let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
        let (pa, pb, pc, pd) = (
            self.vertices[a],
            self.vertices[b],
            self.vertices[c],
            self.vertices[d],
        );
        // Barycentric-style location via the four face orientations. With the
        // tet positively oriented (orient_3d(a,b,c,d) > 0), the point is inside
        // iff all four are >= 0; a (near-)zero indicates the point lies ON that
        // face; two zeros ⇒ on the shared edge. Point location is geometric, so
        // we classify a face/edge incidence with a RELATIVE tolerance against
        // the tet's own scale (6*volume); this only affects WHICH conformal
        // split runs - the gate still validates the final volume exactly.
        let vol6 = orient_3d(pa, pb, pc, pd).abs();
        if vol6 <= 0.0 {
            // The located tet is DEGENERATE (exactly zero volume - a flat
            // sliver the flips left behind). The point then lies in the
            // sliver's PLANE, i.e. on/near faces it shares with non-degenerate
            // neighbours. Walk the local patch (deterministic BFS, bounded)
            // for a finite, non-degenerate tet that contains the point and
            // classify there. Without this, the four face orientations are
            // all -0.0 with tol = 0, the classifier reads "on a vertex" and
            // bails - the terminal `insert failed` mode behind issue #47
            // cluster B (NestedCylinder & friends) after hundreds of good
            // splits.
            let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            seen.insert(ti);
            for fi in 0..4 {
                let nb = self.tets[ti].adj[fi];
                if nb != usize::MAX && self.is_live(nb) && seen.insert(nb) {
                    queue.push_back(nb);
                }
            }
            let mut visited = 0usize;
            while let Some(ci) = queue.pop_front() {
                visited += 1;
                if visited > 64 {
                    break;
                }
                if !self.tets[ci].is_hull() {
                    let cv = self.tets[ci].verts;
                    let (qa, qb, qc, qd) = (
                        self.vertices[cv[0]],
                        self.vertices[cv[1]],
                        self.vertices[cv[2]],
                        self.vertices[cv[3]],
                    );
                    let v6 = orient_3d(qa, qb, qc, qd).abs();
                    if v6 > 0.0 {
                        let oc = [
                            orient_3d(point, qb, qc, qd),
                            orient_3d(qa, point, qc, qd),
                            orient_3d(qa, qb, point, qd),
                            orient_3d(qa, qb, qc, point),
                        ];
                        let tc = v6 * 1e-9;
                        if oc.iter().all(|&x| x >= -tc) {
                            let r = self.classify_and_split(ci, &oc, tc, point);
                            if r.is_none() {
                                ifail!("degenerate-located: neighbour classify failed oc={oc:?}");
                            }
                            return r;
                        }
                    }
                }
                for fi in 0..4 {
                    let nb = self.tets[ci].adj[fi];
                    if nb != usize::MAX && self.is_live(nb) && seen.insert(nb) {
                        queue.push_back(nb);
                    }
                }
            }
            // The local patch has no containing tet - the zero-volume tet
            // TRAPPED the location walk (its faces are coplanar, the exit
            // test degenerates, and it falsely reports containment far from
            // the true location). Recover with a deterministic full scan for
            // a finite, non-degenerate tet containing the point. O(#tets),
            // but it only runs when a walk lands in a flat sliver - rare -
            // and correctness here decides whether the whole conforming
            // attempt survives (issue #47 cluster B).
            let mut found: Option<(usize, [f64; 4], f64)> = None;
            for ci in 0..self.tets.len() {
                if !self.is_live(ci) || self.tets[ci].is_hull() {
                    continue;
                }
                let cv = self.tets[ci].verts;
                let (qa, qb, qc, qd) = (
                    self.vertices[cv[0]],
                    self.vertices[cv[1]],
                    self.vertices[cv[2]],
                    self.vertices[cv[3]],
                );
                let v6 = orient_3d(qa, qb, qc, qd).abs();
                if v6 <= 0.0 {
                    continue;
                }
                let oc = [
                    orient_3d(point, qb, qc, qd),
                    orient_3d(qa, point, qc, qd),
                    orient_3d(qa, qb, point, qd),
                    orient_3d(qa, qb, qc, point),
                ];
                let tc = v6 * 1e-9;
                if oc.iter().all(|&x| x >= -tc) {
                    found = Some((ci, oc, tc));
                    break;
                }
            }
            if let Some((ci, oc, tc)) = found {
                let r = self.classify_and_split(ci, &oc, tc, point);
                if r.is_none() {
                    ifail!("degenerate-trap full-scan: classify failed oc={oc:?}");
                }
                return r;
            }
            if idbg {
                let mut n_hull = 0;
                let mut n_degen = 0;
                let mut n_fat = 0;
                let mut best: f64 = f64::NEG_INFINITY;
                for &ci in &seen {
                    if !self.is_live(ci) {
                        continue;
                    }
                    if self.tets[ci].is_hull() {
                        n_hull += 1;
                        continue;
                    }
                    let cv = self.tets[ci].verts;
                    let v6 = orient_3d(
                        self.vertices[cv[0]],
                        self.vertices[cv[1]],
                        self.vertices[cv[2]],
                        self.vertices[cv[3]],
                    )
                    .abs();
                    if v6 <= 0.0 {
                        n_degen += 1;
                    } else {
                        n_fat += 1;
                        let oc = [
                            orient_3d(
                                point,
                                self.vertices[cv[1]],
                                self.vertices[cv[2]],
                                self.vertices[cv[3]],
                            ),
                            orient_3d(
                                self.vertices[cv[0]],
                                point,
                                self.vertices[cv[2]],
                                self.vertices[cv[3]],
                            ),
                            orient_3d(
                                self.vertices[cv[0]],
                                self.vertices[cv[1]],
                                point,
                                self.vertices[cv[3]],
                            ),
                            orient_3d(
                                self.vertices[cv[0]],
                                self.vertices[cv[1]],
                                self.vertices[cv[2]],
                                point,
                            ),
                        ];
                        let worst = oc.iter().cloned().fold(f64::INFINITY, f64::min) / v6;
                        best = best.max(worst);
                    }
                }
                ifail!(
                    "located tet degenerate (vol=0), no containing neighbour: patch hull={n_hull} degen={n_degen} fat={n_fat} best_rel={best:e} point={point:?}"
                );
            }
            return None;
        }
        let tol = vol6 * 1e-9;
        let o = [
            orient_3d(point, pb, pc, pd), // opposite a
            orient_3d(pa, point, pc, pd), // opposite b
            orient_3d(pa, pb, point, pd), // opposite c
            orient_3d(pa, pb, pc, point), // opposite d
        ];
        if o.iter().any(|&x| x < -tol) {
            // Borderline/outside the located tet - the walk's tolerance and ours
            // disagree. Retry by nudging into the most-negative face's neighbour
            // once; if still bad, bail.
            let worst = (0..4)
                .min_by(|&i, &j| o[i].partial_cmp(&o[j]).unwrap())
                .unwrap();
            let nb = self.tets[ti].adj[worst];
            if nb != usize::MAX && self.is_live(nb) && !self.tets[nb].is_hull() {
                ti = nb;
                let v2 = self.tets[ti].verts;
                let o2 = [
                    orient_3d(
                        point,
                        self.vertices[v2[1]],
                        self.vertices[v2[2]],
                        self.vertices[v2[3]],
                    ),
                    orient_3d(
                        self.vertices[v2[0]],
                        point,
                        self.vertices[v2[2]],
                        self.vertices[v2[3]],
                    ),
                    orient_3d(
                        self.vertices[v2[0]],
                        self.vertices[v2[1]],
                        point,
                        self.vertices[v2[3]],
                    ),
                    orient_3d(
                        self.vertices[v2[0]],
                        self.vertices[v2[1]],
                        self.vertices[v2[2]],
                        point,
                    ),
                ];
                let vol6b = orient_3d(
                    self.vertices[v2[0]],
                    self.vertices[v2[1]],
                    self.vertices[v2[2]],
                    self.vertices[v2[3]],
                )
                .abs();
                let tolb = vol6b * 1e-9;
                if o2.iter().any(|&x| x < -tolb) {
                    ifail!("retry neighbour still outside (o2={o2:?} tolb={tolb:e})");
                    return None;
                }
                let r = self.classify_and_split(ti, &o2, tolb, point);
                if r.is_none() {
                    ifail!("classify_and_split (retry) failed o2={o2:?} tolb={tolb:e}");
                }
                return r;
            }
            ifail!("outside located tet, neighbour unusable (o={o:?} tol={tol:e})");
            return None;
        }
        let r = self.classify_and_split(ti, &o, tol, point);
        if r.is_none() {
            ifail!("classify_and_split failed o={o:?} tol={tol:e}");
        }
        r
    }

    /// Given the four face-orientation values `o` of point relative to finite
    /// tet `ti` (all >= -tol), dispatch the conformal split: interior → 1-4,
    /// on-face → face split, on-edge → edge split.
    fn classify_and_split(
        &mut self,
        ti: usize,
        o: &[f64; 4],
        tol: f64,
        point: [f64; 3],
    ) -> Option<usize> {
        let v = self.tets[ti].verts;
        let zeros: Vec<usize> = (0..4).filter(|&i| o[i].abs() <= tol).collect();
        match zeros.len() {
            0 => self.split_tet_1_to_4(ti, point),
            1 => {
                let f = opposite_face(v, zeros[0]);
                if f.contains(&INFINITE) {
                    return None;
                }
                self.split_face_on_constraint(f[0], f[1], f[2], point)
            }
            2 => {
                let on_face: [usize; 2] = [zeros[0], zeros[1]];
                let edge: Vec<usize> = (0..4)
                    .filter(|i| !on_face.contains(i))
                    .map(|i| v[i])
                    .collect();
                if edge.len() != 2 || edge.contains(&INFINITE) {
                    return None;
                }
                match self.split_edge_on_constraint(edge[0], edge[1], point) {
                    Some(m) => Some(m),
                    None => {
                        // The split refused - typically because the point
                        // divides the edge OUTSIDE the splittable interior,
                        // i.e. it coincides with an endpoint within fp (the
                        // almost-3-zeros configuration of issue #57's
                        // near-tangent class). Resolve to that existing
                        // vertex, exactly like the 3-zeros arm below.
                        let (pu, pw) = (self.vertices[edge[0]], self.vertices[edge[1]]);
                        let d = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2]];
                        let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if l2 <= 0.0 {
                            return None;
                        }
                        let r = [point[0] - pu[0], point[1] - pu[1], point[2] - pu[2]];
                        let s = (r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / l2;
                        if s <= 1e-6 {
                            Some(edge[0])
                        } else if s >= 1.0 - 1e-6 {
                            Some(edge[1])
                        } else {
                            None
                        }
                    }
                }
            }
            3 => {
                // Three zero faces share exactly one tet vertex - the point
                // IS (within tolerance) that existing vertex. Inserting would
                // create a near-duplicate vertex; instead REPORT the existing
                // one, turning the caller's split into the vertex-on-segment
                // resolution (issue #57: a Steiner crossing point landing on
                // an existing vertex previously hard-failed the whole
                // conforming attempt). The caller's follow-ups - facet split
                // at m, Lawson restoration around m, child recoveries - are
                // all valid for an existing vertex.
                let nz = (0..4).find(|i| !zeros.contains(i))?;
                let w = v[nz];
                if w == INFINITE {
                    None
                } else {
                    Some(w)
                }
            }
            _ => None, // 4 zeros ⇒ degenerate located tet - nothing to do
        }
    }

    /// 1-to-4 split: replace the single finite tet `ti = {a,b,c,d}` (which must
    /// strictly contain `point` in its interior) by the four tets
    /// `{point,b,c,d}`, `{a,point,c,d}`, `{a,b,point,d}`, `{a,b,c,point}`,
    /// relinking adjacency conformally (internal new↔new via face-map; external
    /// to `ti`'s four neighbours, hull-aware). Returns the new vertex index, or
    /// `None` (mesh unchanged) if any sub-tet is degenerate.
    fn split_tet_1_to_4(&mut self, ti: usize, point: [f64; 3]) -> Option<usize> {
        let v = self.tets[ti].verts;
        if v.contains(&INFINITE) {
            return None;
        }
        let m = self.vertices.len();
        // Build the four children, replacing each vertex slot by m in turn. The
        // external face each child inherits is the parent face opposite the
        // replaced slot.
        let mut children: Vec<([usize; 4], [usize; 3], usize)> = Vec::with_capacity(4);
        for slot in 0..4 {
            let mut cv = v;
            cv[slot] = m;
            let ext_face = opposite_face(v, slot); // does NOT contain the replaced vertex
            let ext_nb = self.tets[ti].adj[slot];
            // Orient (m substituted for index m).
            let coord = |idx: usize| -> [f64; 3] {
                if idx == m {
                    point
                } else {
                    self.vertices[idx]
                }
            };
            let mut w = cv;
            let oo = orient_3d(coord(w[0]), coord(w[1]), coord(w[2]), coord(w[3]));
            if oo < 0.0 {
                w.swap(0, 1);
            } else if oo == 0.0 {
                return None; // degenerate child
            }
            children.push((w, ext_face, ext_nb));
        }

        // ── COMMIT ──
        let m = self.push_vertex(point);
        self.free_tet(ti);
        let mut new_indices = Vec::with_capacity(4);
        for (cv, _, _) in &children {
            new_indices.push(self.alloc_tet(Tet {
                verts: *cv,
                adj: [usize::MAX; 4],
            }));
        }
        // Internal new↔new adjacency.
        let mut fmap: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
        for &ni in &new_indices {
            for fi in 0..4 {
                let mut sf = opposite_face(self.tets[ni].verts, fi);
                sf.sort();
                if let Some((other_ti, other_fi)) = fmap.remove(&sf) {
                    self.tets[ni].adj[fi] = other_ti;
                    self.tets[other_ti].adj[other_fi] = ni;
                } else {
                    fmap.insert(sf, (ni, fi));
                }
            }
        }
        // External adjacency to ti's old neighbours.
        for (k, (_, ext_face, ext_nb)) in children.iter().enumerate() {
            let ni = new_indices[k];
            let nb = *ext_nb;
            if nb == usize::MAX || !self.is_live(nb) {
                continue;
            }
            for fi in 0..4 {
                if faces_match(&opposite_face(self.tets[ni].verts, fi), ext_face) {
                    self.tets[ni].adj[fi] = nb;
                    for nfi in 0..4 {
                        if faces_match(&opposite_face(self.tets[nb].verts, nfi), ext_face) {
                            self.tets[nb].adj[nfi] = ni;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        Some(m)
    }

    /// Like [`oriented_hull_tet`] but any vertex index equal to
    /// `self.vertices.len()` (the not-yet-pushed Steiner point `m`) resolves to
    /// the transient coordinate `pt`. Used by the split primitives during their
    /// validation phase, where a hull child can legitimately have `m` as one of
    /// its three hull-face corners (splitting a hull edge/face). The orientation
    /// convention is identical to [`oriented_hull_tet`]: the finite face is
    /// arranged so interior points (here the `interior_seed`) are on the
    /// NEGATIVE side.
    fn oriented_hull_tet_pt(&self, f0: usize, f1: usize, f2: usize, pt: [f64; 3]) -> [usize; 4] {
        let m = self.vertices.len();
        let coord = |idx: usize| -> [f64; 3] {
            if idx == m {
                pt
            } else {
                self.vertices[idx]
            }
        };
        let mut hv = [f0, f1, f2, INFINITE];
        let o = orient_3d(coord(f0), coord(f1), coord(f2), self.interior_seed);
        if o > 0.0 {
            hv.swap(0, 1);
        }
        hv
    }

    /// Extract finite tets (no INFINITE vertex), deduplicated.
    ///
    /// Vertex order is whatever the incremental construction left behind, so
    /// these tets are NOT yet guaranteed positively oriented. The pipelines in
    /// [`crate::volume`] establish that invariant afterwards and
    /// [`crate::volume::mesh_volume`] re-checks it before returning; anything
    /// that starts routing these tets to a caller must do the same.
    pub fn extract_tets(&self) -> Vec<[usize; 4]> {
        let cap = self.tets.len() - self.free_list.stack.len();
        let mut result = Vec::with_capacity(cap);
        let mut seen = HashSet::with_capacity_and_hasher(cap, Default::default());
        for (i, t) in self.tets.iter().enumerate() {
            if self.free_list.contains(&i) || t.is_hull() {
                continue;
            }
            let mut k = t.verts;
            k.sort();
            if seen.insert(k) {
                result.push(t.verts);
            }
        }
        result
    }
}

// --- Helpers ---
/// Spread the low 21 bits of `n` so each occupies every 3rd bit position
/// (`b20 b19 ... b0` → `b20 0 0 b19 0 0 ... b0`). Building block for a 3-D
/// Morton (Z-order) code: OR three such spreads at offsets 0/1/2.
#[inline]
fn part1by2(n: u64) -> u64 {
    let mut x = n & 0x1f_ffff; // keep 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

/// Lower corner + per-axis quantization scale of `points`' bounding box for a
/// 21-bit-per-axis Morton grid. Zero-extent axes get scale 0 (quantize to 0).
fn morton_scale(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            if p[k] < lo[k] {
                lo[k] = p[k];
            }
            if p[k] > hi[k] {
                hi[k] = p[k];
            }
        }
    }
    const MAXQ: f64 = ((1u32 << 21) - 1) as f64; // 21-bit grid
    let mut scale = [0.0f64; 3];
    for k in 0..3 {
        let ext = hi[k] - lo[k];
        scale[k] = if ext > 0.0 { MAXQ / ext } else { 0.0 };
    }
    (lo, scale)
}

/// 3-D Morton (Z-order) code of `p` under the given bbox quantization.
#[inline]
fn morton_code(p: &[f64; 3], lo: &[f64; 3], scale: &[f64; 3]) -> u64 {
    let qx = (((p[0] - lo[0]) * scale[0]) as u64).min(0x1f_ffff);
    let qy = (((p[1] - lo[1]) * scale[1]) as u64).min(0x1f_ffff);
    let qz = (((p[2] - lo[2]) * scale[2]) as u64).min(0x1f_ffff);
    part1by2(qx) | (part1by2(qy) << 1) | (part1by2(qz) << 2)
}

/// Indices of `points` sorted by 3-D Morton (Z-order) code. Gives spatially-
/// coherent insertion order so Bowyer-Watson's point-location walk stays local
/// (near-linear build instead of the O(n^2) linear-scan-fallback path).
fn morton_order(points: &[[f64; 3]]) -> Vec<usize> {
    let (lo, scale) = morton_scale(points);
    let mut order: Vec<usize> = (0..points.len()).collect();
    // Stable sort by Morton code; ties keep input order (deterministic).
    order.sort_by_key(|&i| morton_code(&points[i], &lo, &scale));
    order
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}
fn farthest_pair(pts: &[[f64; 3]]) -> (usize, usize) {
    let (mut i0, mut i1, mut best) = (0, 1, 0.0f64);
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = dist2(pts[i], pts[j]);
            if d > best {
                best = d;
                i0 = i;
                i1 = j;
            }
        }
    }
    (i0, i1)
}
fn dist_to_line_sq(p: [f64; 3], o: [f64; 3], d: &[f64; 3]) -> f64 {
    let v = sub(p, o);
    let c = cross(&v, d);
    let d2 = dot(d, d);
    if d2 < 1e-30 {
        0.0
    } else {
        dot(&c, &c) / d2
    }
}

pub(super) fn opposite_face(v: [usize; 4], fi: usize) -> [usize; 3] {
    match fi {
        0 => [v[1], v[2], v[3]],
        1 => [v[0], v[3], v[2]],
        2 => [v[0], v[1], v[3]],
        3 => [v[0], v[2], v[1]],
        _ => unreachable!(),
    }
}
pub(super) fn faces_match(a: &[usize; 3], b: &[usize; 3]) -> bool {
    let (mut sa, mut sb) = (*a, *b);
    sa.sort();
    sb.sort();
    sa == sb
}
pub(super) fn shared_face_indices(a: &Tet, b: &Tet) -> Option<(usize, usize)> {
    for fi in 0..4 {
        let fa = opposite_face(a.verts, fi);
        for fj in 0..4 {
            if faces_match(&fa, &opposite_face(b.verts, fj)) {
                return Some((fi, fj));
            }
        }
    }
    None
}
fn build_adjacency(tets: &mut [Tet]) {
    for i in 0..tets.len() {
        for j in (i + 1)..tets.len() {
            if let Some((fi, fj)) = shared_face_indices(&tets[i], &tets[j]) {
                tets[i].adj[fi] = j;
                tets[j].adj[fj] = i;
            }
        }
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build-time scaling probe for `Delaunay3D::new` on a boundary-dominated
    /// cloud (two jittered cylinder shells, emitted band by band). Ignored by
    /// default (it allocates ~150k points); run with
    /// `cargo test --release delaunay_build_scaling -- --ignored --nocapture`.
    ///
    /// Profiling findings (kept here as the record behind issue #30's base-build
    /// ceiling):
    /// - Point LOCATION is O(1)/point with Morton insertion order: ~10 walk
    ///   steps/point, ZERO linear-scan fallbacks, ~1-3% of build time. A
    ///   point-location hierarchy would therefore buy nothing.
    /// - The build is nonetheless ~O(n^1.6): new-tets-per-insert is CONSTANT
    ///   (~6) yet the cavity-free + new-tet-create + adjacency-link phase's wall
    ///   time grows superlinearly. Constant ops/insert + superlinear wall time =
    ///   MEMORY-LATENCY bound - the tet array (~64 B/tet → >100 MB at ~300k pts)
    ///   outgrows CPU cache, so the per-insert random accesses (neighbour tets,
    ///   freed-slot writes, the index-keyed epoch/free-list arrays) become RAM
    ///   round-trips. Only bites at fine targets (large n), where the legacy
    ///   filter is already accurate.
    /// - Cache-locality fixes were TRIED and did NOT help: replacing the per-
    ///   insert face-map `HashMap` with a flat `Vec` (kept - cleaner, frees the
    ///   never-shrinking capacity, but no measurable speedup), and periodic
    ///   tet-array compaction + Morton-reorder so spatially-near tets are
    ///   index-near (reverted - 0 benefit). Reordering INDICES doesn't shrink
    ///   the WORKING SET: an incremental Delaunay over a surface-distributed
    ///   cloud touches the whole growing "active front," which exceeds cache
    ///   regardless of index order. A real win would need a fundamentally
    ///   different cache-aware construction (block/divide-and-conquer or BRIO
    ///   with spatial blocking to bound the active front) - a major rewrite, and
    ///   unwarranted since fine targets don't need the conforming path. See #30.
    #[test]
    #[ignore]
    fn delaunay_build_scaling() {
        use std::time::Instant;
        for &n_band in &[2_000usize, 20_000, 80_000, 150_000] {
            // Two concentric cylinder shells (mimics ThinWalledCylinder), points
            // emitted shell-by-shell then ring-by-ring: locally coherent, but the
            // axial sweep makes input order poor for a naive walk.
            let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n_band * 2 + 8);
            let rings = ((n_band as f64).sqrt() as usize).max(4);
            let per_ring = (n_band / rings).max(4);
            // Deterministic sub-permille jitter per point breaks the perfectly
            // cospherical/coplanar rings (which would otherwise trigger large
            // coplanar-absorption cavities) - real CAD boundary meshes are not
            // exactly cospherical. jit(k) ∈ [-5e-4, 5e-4].
            let jit = |k: usize| -> f64 {
                (((k.wrapping_mul(2654435761) % 1000) as f64) / 1000.0 - 0.5) * 1e-3
            };
            let mut k = 0usize;
            for &r in &[4.8f64, 5.0] {
                for ir in 0..rings {
                    let z = 20.0 * (ir as f64) / (rings as f64) + jit(k);
                    for ip in 0..per_ring {
                        let th = 2.0 * std::f64::consts::PI * (ip as f64) / (per_ring as f64);
                        let rr = r + jit(k.wrapping_add(7));
                        pts.push([rr * th.cos() + jit(k + 3), rr * th.sin(), z]);
                        k += 1;
                    }
                }
            }
            // caps
            pts.push([0.0, 0.0, -0.1]);
            pts.push([0.0, 0.0, 20.1]);
            let t = Instant::now();
            let dt = Delaunay3D::new(&pts);
            let secs = t.elapsed().as_secs_f64();
            let live = dt.extract_tets().len();
            eprintln!(
                "  [build scaling] {} pts -> {:.3}s ({} finite tets, {:.1} pts/ms)",
                pts.len(),
                secs,
                live,
                pts.len() as f64 / (secs * 1000.0)
            );
            assert!(live > 0);
        }
    }

    #[test]
    fn single_point() {
        assert!(Delaunay3D::new(&[[0.; 3]]).extract_tets().is_empty());
    }

    #[test]
    fn five_points() {
        let pts = vec![
            [0., 0., 0.],
            [2., 0., 0.],
            [1., 2., 0.],
            [1., 1., 2.],
            [1., 0.8, 0.5],
        ];
        let tets = Delaunay3D::new(&pts).extract_tets();
        assert!(!tets.is_empty());
        for t in &tets {
            for &v in t {
                assert!(v < 5);
            }
        }
    }

    #[test]
    fn cube_corners() {
        let pts = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ];
        assert!(Delaunay3D::new(&pts).extract_tets().len() >= 5);
    }

    #[test]
    fn positive_volume() {
        let pts = vec![
            [0., 0., 0.],
            [3., 0., 0.],
            [0., 3., 0.],
            [0., 0., 3.],
            [1., 1., 1.],
            [2., 0.5, 0.5],
            [0.5, 2., 0.5],
            [0.5, 0.5, 2.],
        ];
        for t in Delaunay3D::new(&pts).extract_tets() {
            assert!(orient_3d(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]) > -1e-10);
        }
    }

    #[test]
    fn cube_volume() {
        let pts = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ];
        let vol: f64 = Delaunay3D::new(&pts)
            .extract_tets()
            .iter()
            .map(|t| orient_3d(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs() / 6.0)
            .sum();
        assert!((vol - 1.0).abs() < 0.01, "vol={vol:.4}");
    }

    /// Conformity check: every finite-tet face is shared by exactly two finite
    /// tets OR exactly one finite tet (a hull/boundary face). No face may be
    /// shared by three or more finite tets (overlap), and adjacency must be
    /// symmetric among live finite tets.
    fn assert_conforming(dt: &Delaunay3D) {
        let mut face_count: HashMap<[usize; 3], usize> = HashMap::default();
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            for fi in 0..4 {
                let mut f = opposite_face(dt.tets[i].verts, fi);
                if f.contains(&INFINITE) {
                    continue;
                }
                f.sort();
                *face_count.entry(f).or_insert(0) += 1;
            }
        }
        for (f, &c) in &face_count {
            assert!(
                c <= 2,
                "face {f:?} shared by {c} finite tets (>2 ⇒ overlap)"
            );
        }
        // Symmetric adjacency among live finite tets.
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            for fi in 0..4 {
                let nb = dt.tets[i].adj[fi];
                if nb == usize::MAX || !dt.is_live(nb) {
                    continue;
                }
                assert!(
                    dt.tets[nb].adj.contains(&i),
                    "adjacency not symmetric: {i} -> {nb} but not back"
                );
            }
        }
        // Positive volume.
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            let v = dt.tets[i].verts;
            let o = orient_3d(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            );
            assert!(o > 0.0, "tet {i} non-positive orientation {o}");
        }
    }

    fn edge_present(dt: &Delaunay3D, a: usize, b: usize) -> bool {
        (0..dt.tets.len()).any(|i| {
            dt.is_live(i) && {
                let v = dt.tets[i].verts;
                v.contains(&a) && v.contains(&b)
            }
        })
    }

    #[test]
    fn split_interior_edge_conformal() {
        // A bipyramid: two tets sharing face (0,1,2), apices 3 (above) and 4
        // (below). Edge (0,1) is interior to that shared structure; split its
        // midpoint and assert both sub-edges exist and the mesh is conformal.
        let mut pts = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        // Surround with more points so (0,1) has a proper interior ring.
        pts.push([1.0, -1.0, 0.5]); // 5
        pts.push([1.0, -1.0, -0.5]); // 6
        let mut dt = Delaunay3D::new(&pts);
        assert!(
            edge_present(&dt, 0, 1),
            "edge (0,1) must be present pre-split"
        );
        let mid = [
            (pts[0][0] + pts[1][0]) / 2.0,
            (pts[0][1] + pts[1][1]) / 2.0,
            (pts[0][2] + pts[1][2]) / 2.0,
        ];
        let m = dt
            .split_edge_on_constraint(0, 1, mid)
            .expect("edge split should succeed on a real interior edge");
        assert_eq!(m, pts.len(), "new vertex appended at end");
        assert!(edge_present(&dt, 0, m), "sub-edge (0,m) must exist");
        assert!(edge_present(&dt, m, 1), "sub-edge (m,1) must exist");
        assert!(!edge_present(&dt, 0, 1), "original edge (0,1) must be gone");
        assert_conforming(&dt);
    }

    #[test]
    fn split_boundary_edge_with_hull_conformal() {
        // A single tetrahedron: its edges are all hull edges (each edge's ring
        // includes hull tets). Splitting one must stay conformal and hull-aware.
        let pts = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut dt = Delaunay3D::new(&pts);
        assert!(edge_present(&dt, 0, 1));
        let mid = [1.5, 0.0, 0.0];
        let m = dt
            .split_edge_on_constraint(0, 1, mid)
            .expect("hull edge split should succeed");
        assert!(edge_present(&dt, 0, m));
        assert!(edge_present(&dt, m, 1));
        assert!(!edge_present(&dt, 0, 1));
        assert_conforming(&dt);
    }

    #[test]
    fn split_interior_face_conformal() {
        // Two tets sharing face (0,1,2); split its barycenter. Each tet → 3,
        // so 6 finite tets; the three sub-faces (0,1,m),(1,2,m),(2,0,m) each
        // shared by two finite tets.
        let pts = vec![
            [0.0, 0.0, 0.0],  // 0
            [2.0, 0.0, 0.0],  // 1
            [1.0, 2.0, 0.0],  // 2
            [1.0, 0.7, 1.5],  // 3
            [1.0, 0.7, -1.5], // 4
        ];
        let mut dt = Delaunay3D::new(&pts);
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 0, 1, 2
        ));
        let bary = [
            (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0,
            (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0,
            (pts[0][2] + pts[1][2] + pts[2][2]) / 3.0,
        ];
        let m = dt
            .split_face_on_constraint(0, 1, 2, bary)
            .expect("face split should succeed");
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 0, 1, m
        ));
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 1, 2, m
        ));
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 2, 0, m
        ));
        assert!(
            !super::super::boundary_recovery::face_exists_in_tets(&dt, 0, 1, 2),
            "original face must be replaced by its three children"
        );
        assert_conforming(&dt);
    }

    #[test]
    fn split_boundary_face_with_hull_conformal() {
        // Single tet: face (1,2,3) is a hull face (one finite tet incident, plus
        // a hull tet on the outside). Split its barycenter; stay conformal.
        let pts = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let mut dt = Delaunay3D::new(&pts);
        let bary = [
            (pts[1][0] + pts[2][0] + pts[3][0]) / 3.0,
            (pts[1][1] + pts[2][1] + pts[3][1]) / 3.0,
            (pts[1][2] + pts[2][2] + pts[3][2]) / 3.0,
        ];
        let m = dt
            .split_face_on_constraint(1, 2, 3, bary)
            .expect("hull face split should succeed");
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 1, 2, m
        ));
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 2, 3, m
        ));
        assert!(super::super::boundary_recovery::face_exists_in_tets(
            &dt, 3, 1, m
        ));
        assert_conforming(&dt);
    }

    /// Every DISTINCT input vertex must appear in at least one live finite tet.
    /// Returns the set of input indices that were dropped (should be empty).
    fn dropped_vertices(dt: &Delaunay3D, n_input: usize) -> Vec<usize> {
        let mut seen = vec![false; n_input];
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) || dt.tets[i].is_hull() {
                continue;
            }
            for &v in &dt.tets[i].verts {
                if v != INFINITE && v < n_input {
                    seen[v] = true;
                }
            }
        }
        (0..n_input).filter(|&i| !seen[i]).collect()
    }

    /// De-duplicate coincident input points (within an absolute tolerance) and
    /// return, for each surviving (distinct) point, one representative index.
    fn distinct_indices(pts: &[[f64; 3]], tol: f64) -> Vec<usize> {
        let mut reps: Vec<usize> = Vec::new();
        'outer: for i in 0..pts.len() {
            for &j in &reps {
                if dist2(pts[i], pts[j]).sqrt() <= tol {
                    continue 'outer;
                }
            }
            reps.push(i);
        }
        reps
    }

    fn assert_all_distinct_preserved(pts: &[[f64; 3]]) {
        let dt = Delaunay3D::new(pts);
        let dropped = dropped_vertices(&dt, pts.len());
        // A dropped index is only acceptable if it is a duplicate of another,
        // PRESERVED point (coincident within tolerance).
        let reps = distinct_indices(pts, 1e-12);
        let bad: Vec<usize> = dropped
            .iter()
            .copied()
            .filter(|d| reps.contains(d))
            .collect();
        assert!(
            bad.is_empty(),
            "distinct input vertices dropped: {bad:?} (total dropped {}, n={})",
            dropped.len(),
            pts.len()
        );
    }

    /// Convex-hull volume equality: for a convex point set, the sum of finite
    /// tet volumes must equal the true region volume `expected` (no gaps, no
    /// overlaps). Also asserts conformity (≤2 finite tets per interior face).
    fn assert_convex_fill(pts: &[[f64; 3]], expected: f64) {
        let dt = Delaunay3D::new(pts);
        assert_conforming(&dt);
        assert!(dropped_vertices(&dt, pts.len()).is_empty());
        let vol: f64 = dt
            .extract_tets()
            .iter()
            .map(|t| orient_3d(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs() / 6.0)
            .sum();
        assert!(
            (vol - expected).abs() < expected * 1e-9 + 1e-9,
            "filled vol {vol:.12} != expected {expected:.12}"
        );
    }

    /// 3x3x3 structured grid: maximally cospherical / coplanar - the exact
    /// stress case that made the old cut-cascade empty cavities and drop points.
    #[test]
    fn grid_3x3x3_preserves_all_vertices_and_fills() {
        let mut pts = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    pts.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        assert_all_distinct_preserved(&pts);
        // Convex hull is the 2x2x2 cube.
        assert_convex_fill(&pts, 8.0);
    }

    /// Cube corners PLUS an interior structured grid (mixed boundary + interior
    /// cospherical layout).
    #[test]
    fn cube_plus_grid_preserves_all_vertices_and_fills() {
        let mut pts = vec![
            [0., 0., 0.],
            [4., 0., 0.],
            [4., 4., 0.],
            [0., 4., 0.],
            [0., 0., 4.],
            [4., 0., 4.],
            [4., 4., 4.],
            [0., 4., 4.],
        ];
        for x in [1.0, 2.0, 3.0] {
            for y in [1.0, 2.0, 3.0] {
                for z in [1.0, 2.0, 3.0] {
                    pts.push([x, y, z]);
                }
            }
        }
        assert_all_distinct_preserved(&pts);
        assert_convex_fill(&pts, 64.0);
    }

    /// Hand-built THIN SLAB: a wide, near-degenerate (very thin in z) box of
    /// grid points. Thin/flat sets are exactly where the old code cut the cavity
    /// to empty. The convex hull is the slab box.
    #[test]
    fn thin_slab_preserves_all_vertices_and_fills() {
        let mut pts = Vec::new();
        let nz = 0.05_f64;
        for x in 0..4 {
            for y in 0..4 {
                for &z in &[0.0_f64, nz] {
                    pts.push([x as f64, y as f64, z]);
                }
            }
        }
        assert_all_distinct_preserved(&pts);
        // Hull is 3 x 3 x nz box.
        assert_convex_fill(&pts, 9.0 * nz);
    }

    /// Hand-built REFLEX (L-shaped) point set. The Delaunay tetrahedralization
    /// fills the CONVEX HULL of the L (the notch is filled by Delaunay; boundary
    /// recovery later carves it). The point of THIS test is solely that NO
    /// distinct vertex is dropped on a reflex/non-convex layout - the failure
    /// mode the diagnosis measured (LShaped dropped 42/52 boundary verts).
    #[test]
    fn lshape_reflex_preserves_all_vertices() {
        // An L-profile (in xy) extruded in z, sampled on a unit grid so the
        // layout is heavily coplanar/cospherical along the prism.
        let profile: &[[f64; 2]] = &[
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
            // grid samples on the faces to add coplanar stress
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.5],
            [0.5, 2.0],
            [1.0, 0.5],
            [0.5, 1.0],
        ];
        let mut pts = Vec::new();
        for &z in &[0.0_f64, 1.0, 2.0] {
            for &[x, y] in profile {
                pts.push([x, y, z]);
            }
        }
        assert_all_distinct_preserved(&pts);
    }

    /// Regular grid where many points are EXACTLY cospherical at every step:
    /// re-uses the conformity + no->2-share invariant directly.
    #[test]
    fn grid_no_overlapping_tets() {
        let mut pts = Vec::new();
        for x in 0..4 {
            for y in 0..3 {
                for z in 0..3 {
                    pts.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let dt = Delaunay3D::new(&pts);
        assert_conforming(&dt); // ≤2 finite tets per interior face ⇒ no overlap
        assert!(dropped_vertices(&dt, pts.len()).is_empty());
    }

    #[test]
    fn many_points() {
        let mut pts = vec![
            [0., 0., 0.],
            [10., 0., 0.],
            [10., 10., 0.],
            [0., 10., 0.],
            [0., 0., 10.],
            [10., 0., 10.],
            [10., 10., 10.],
            [0., 10., 10.],
        ];
        for x in [2.5, 5., 7.5] {
            for y in [2.5, 5., 7.5] {
                for z in [2.5, 5., 7.5] {
                    pts.push([x, y, z]);
                }
            }
        }
        let tets = Delaunay3D::new(&pts).extract_tets();
        assert!(tets.len() > 20);
        let vol: f64 = tets
            .iter()
            .map(|t| orient_3d(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs() / 6.0)
            .sum();
        assert!((vol - 1000.0).abs() < 1.0, "vol={vol:.1}");
    }
}
