//! Mesh-side kernels of the CAD pipeline's face-volume correction.
//!
//! A chordal mesh inscribes a curved surface, so a curved solid's mesh is too
//! small. The correction (driven from `yamc.cad._face_volumes`) moves the
//! vertices inside each curved CAD edge, then inside each curved face, by one
//! common distance per edge or face, until each edge polyline cuts off the
//! arc's area and each face's facets enclose the CAD face's volume.
//!
//! Everything that needs the CAD (curve points and normals, a face's exact
//! volume) is evaluated by the caller and handed in; this holds the mesh, its
//! adjacency and the arcs of CAD edge each boundary segment spans, and does
//! the volume sums, the root finding and the vertex moves.

use std::collections::HashMap;

use crate::types::VertexId;

/// Secant steps on the exact enclosed volume (or cut-off area) after a first
/// linear step. The measure is linear in the distance to first order, so a few
/// steps meet it to rounding.
const ITERATIONS: usize = 3;

type Point = [f64; 3];

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add_scaled(a: Point, s: f64, d: Point) -> Point {
    [a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2]]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

/// Signed volume of the tetrahedron from `apex` over triangle pqr.
fn cone(p: Point, q: Point, r: Point, apex: Point) -> f64 {
    dot(sub(p, apex), cross(sub(q, apex), sub(r, apex))) / 6.0
}

/// Signed volume from `apex` over the closed polygon `points`, fanned from its
/// first point.
fn fan_cone(points: impl Iterator<Item = Point>, apex: Point) -> f64 {
    let mut points = points;
    let Some(first) = points.next() else {
        return 0.0;
    };
    let Some(mut prev) = points.next() else {
        return 0.0;
    };
    let mut total = 0.0;
    for p in points {
        total += cone(first, prev, p, apex);
        prev = p;
    }
    total
}

/// Vector area of the closed polygon `points`.
fn vector_area(points: impl Iterator<Item = Point>) -> Point {
    let mut points = points;
    let mut area = [0.0; 3];
    let Some(first) = points.next() else {
        return area;
    };
    let Some(mut prev) = points.next() else {
        return area;
    };
    for p in points {
        let c = cross(sub(prev, first), sub(p, first));
        area[0] += c[0];
        area[1] += c[1];
        area[2] += c[2];
        prev = p;
    }
    [0.5 * area[0], 0.5 * area[1], 0.5 * area[2]]
}

/// `(h, capped)`: the distance within `[-cap, cap]` that zeroes `residual`, a
/// function of one distance close to linear with `slope` at zero, by a Newton
/// step and then secant steps.
fn solve(mut residual: impl FnMut(f64) -> f64, slope: f64, cap: f64) -> (f64, bool) {
    let (mut h0, mut r0) = (0.0, residual(0.0));
    if slope == 0.0 || r0 == 0.0 {
        return (0.0, false);
    }
    let mut h1 = (-r0 / slope).clamp(-cap, cap);
    let mut r1 = residual(h1);
    for _ in 0..ITERATIONS {
        if r1 == r0 || r1.abs() <= 1e-12 * r0.abs() {
            break;
        }
        let h2 = (h1 - r1 * (h1 - h0) / (r1 - r0)).clamp(-cap, cap);
        if h2 == h1 {
            break;
        }
        (h0, r0) = (h1, r1);
        h1 = h2;
        r1 = residual(h2);
    }
    (h1, h1.abs() >= cap)
}

fn key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Counts of what the correction did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceVolumeStats {
    /// Curved edges whose interior vertices moved.
    pub edges_moved: usize,
    /// Of those, the ones whose move hit the deflection cap.
    pub edges_capped: usize,
    /// Curved faces whose interior vertices moved.
    pub faces_moved: usize,
    /// Of those, the ones whose move hit the deflection cap.
    pub faces_capped: usize,
    /// Curved faces with no vertex of their own to move.
    pub faces_no_interior: usize,
    /// Curved faces whose boundary could not be matched to their CAD edges.
    pub faces_unmatched: usize,
}

/// A matched CAD edge: its mesh vertices in curve order and, per segment
/// between two of them, points of the CAD edge from one to the other.
struct Edge {
    chain: Vec<VertexId>,
    arcs: Vec<Vec<Point>>,
}

/// A welded surface mesh with the adjacency the correction needs.
pub struct FaceVolumeMesh {
    vertices: Vec<Point>,
    /// Face ids in order of first appearance among the triangles.
    faces: Vec<u32>,
    face_tris: HashMap<u32, Vec<[VertexId; 3]>>,
    /// `(mesh edge, face)` for each face whose triangles use the edge,
    /// sorted.
    edge_faces: Vec<((VertexId, VertexId), u32)>,
    /// Each vertex's faces, sorted.
    vert_faces: Vec<Vec<u32>>,
    /// Each face's boundary: the directed edges of its triangles that no other
    /// of its triangles shares, or None where an edge has more than two.
    boundary: HashMap<u32, Option<Vec<(VertexId, VertexId)>>>,
    /// The faces meshed onto the same set of vertices as each face, sorted.
    twins: HashMap<u32, Vec<u32>>,
    edges: Vec<Edge>,
    /// Per boundary segment (low, high vertex): its edge and segment index.
    seg_arc: HashMap<(VertexId, VertexId), (usize, usize)>,
    /// The face `face_apex` started: its id, vertices that may move, apex.
    pending: Option<(u32, Vec<VertexId>, Point)>,
    stats: FaceVolumeStats,
}

impl FaceVolumeMesh {
    /// Build the adjacency of a mesh of `vertices` and `triangles`, each
    /// triangle on the CAD face of the same index in `face_ids`.
    ///
    /// # Errors
    ///
    /// If `face_ids` and `triangles` differ in length or a triangle indexes
    /// past the vertices.
    pub fn new(
        vertices: Vec<Point>,
        triangles: &[[VertexId; 3]],
        face_ids: &[u32],
    ) -> Result<Self, String> {
        if triangles.len() != face_ids.len() {
            return Err(format!(
                "{} face ids for {} triangles",
                face_ids.len(),
                triangles.len()
            ));
        }
        let n = vertices.len();
        if let Some(t) = triangles
            .iter()
            .find(|t| t.iter().any(|&v| v as usize >= n))
        {
            return Err(format!("triangle {t:?} indexes past the {n} vertices"));
        }
        let mut faces = Vec::new();
        let mut face_tris: HashMap<u32, Vec<[VertexId; 3]>> = HashMap::new();
        for (tri, &fid) in triangles.iter().zip(face_ids) {
            face_tris
                .entry(fid)
                .or_insert_with(|| {
                    faces.push(fid);
                    Vec::new()
                })
                .push(*tri);
        }

        let mut edge_faces = Vec::with_capacity(3 * triangles.len() / 2 + 1);
        let mut vert_faces: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut boundary = HashMap::new();
        let mut groups: HashMap<Vec<VertexId>, Vec<u32>> = HashMap::new();
        let mut keys: Vec<(VertexId, VertexId)> = Vec::new();
        let mut counts: Vec<((VertexId, VertexId), u32)> = Vec::new();
        for &fid in &faces {
            let tris = &face_tris[&fid];
            // How many of the face's triangles use each of its mesh edges.
            keys.clear();
            for tri in tris {
                for k in 0..3 {
                    let (a, b) = (tri[k], tri[(k + 1) % 3]);
                    keys.push(key(a, b));
                    let fs = &mut vert_faces[a as usize];
                    if fs.last() != Some(&fid) {
                        fs.push(fid);
                    }
                }
            }
            keys.sort_unstable();
            counts.clear();
            for &k in &keys {
                match counts.last_mut() {
                    Some((last, c)) if *last == k => *c += 1,
                    _ => counts.push((k, 1)),
                }
            }
            edge_faces.extend(counts.iter().map(|&(k, _)| (k, fid)));
            let segs = if counts.iter().any(|&(_, c)| c > 2) {
                None
            } else {
                let count = |k| counts[counts.partition_point(|&(e, _)| e < k)].1;
                let mut segs = Vec::new();
                for tri in tris {
                    for k in 0..3 {
                        let (a, b) = (tri[k], tri[(k + 1) % 3]);
                        if count(key(a, b)) == 1 {
                            segs.push((a, b));
                        }
                    }
                }
                Some(segs)
            };
            boundary.insert(fid, segs);
            let mut verts: Vec<VertexId> = tris.iter().flatten().copied().collect();
            verts.sort_unstable();
            verts.dedup();
            groups.entry(verts).or_default().push(fid);
        }
        edge_faces.sort_unstable();
        for fs in &mut vert_faces {
            fs.sort_unstable();
        }
        let mut twins = HashMap::new();
        for mut group in groups.into_values() {
            group.sort_unstable();
            for &fid in &group {
                twins.insert(fid, group.clone());
            }
        }
        Ok(Self {
            vertices,
            faces,
            face_tris,
            edge_faces,
            vert_faces,
            boundary,
            twins,
            edges: Vec::new(),
            seg_arc: HashMap::new(),
            pending: None,
            stats: FaceVolumeStats::default(),
        })
    }

    /// The face ids, in order of first appearance among the triangles.
    pub fn faces(&self) -> &[u32] {
        &self.faces
    }

    /// The vertices, as moved so far.
    pub fn vertices(&self) -> &[Point] {
        &self.vertices
    }

    /// What the moves so far did.
    pub fn stats(&self) -> &FaceVolumeStats {
        &self.stats
    }

    /// `faces` and every face that shares a mesh edge of their boundaries,
    /// sorted.
    pub fn neighbours(&self, faces: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = faces.to_vec();
        for fid in faces {
            if let Some(Some(segs)) = self.boundary.get(fid) {
                for &(a, b) in segs {
                    out.extend(self.faces_on(key(a, b)));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The vertices on the boundaries of `faces` inside the box from `lo` to
    /// `hi` (bounds included), with their positions, by vertex index.
    pub fn boundary_points(&self, faces: &[u32], lo: Point, hi: Point) -> Vec<(VertexId, Point)> {
        let mut verts: Vec<VertexId> = faces
            .iter()
            .filter_map(|f| self.boundary.get(f).and_then(Option::as_ref))
            .flat_map(|segs| segs.iter().flat_map(|&(a, b)| [a, b]))
            .collect();
        verts.sort_unstable();
        verts.dedup();
        verts
            .into_iter()
            .map(|v| (v, self.vertices[v as usize]))
            .filter(|(_, p)| (0..3).all(|i| lo[i] <= p[i] && p[i] <= hi[i]))
            .collect()
    }

    /// Whether every two consecutive vertices of `chain` share a mesh edge.
    pub fn is_chain(&self, chain: &[VertexId]) -> bool {
        chain
            .windows(2)
            .all(|w| self.faces_on(key(w[0], w[1])).next().is_some())
    }

    /// The faces whose triangles use the mesh edge `edge` (low, high vertex).
    fn faces_on(&self, edge: (VertexId, VertexId)) -> impl Iterator<Item = u32> + '_ {
        let start = self.edge_faces.partition_point(|&(e, _)| e < edge);
        self.edge_faces[start..]
            .iter()
            .take_while(move |&&(e, _)| e == edge)
            .map(|&(_, f)| f)
    }

    /// Record a matched CAD edge: its mesh vertices `chain` in curve order and
    /// `arcs`, per segment, points of the edge from the segment's first vertex
    /// to its second, ends included. Returns the edge's index.
    ///
    /// # Errors
    ///
    /// If there is not one arc of at least two points per segment, or a
    /// vertex is not in the mesh.
    pub fn add_edge(
        &mut self,
        chain: Vec<VertexId>,
        arcs: Vec<Vec<Point>>,
    ) -> Result<usize, String> {
        if chain.len() < 2 || arcs.len() + 1 != chain.len() {
            return Err(format!(
                "{} arcs for a chain of {} vertices",
                arcs.len(),
                chain.len()
            ));
        }
        if arcs.iter().any(|a| a.len() < 2) {
            return Err("an arc needs at least its two ends".to_string());
        }
        if let Some(v) = chain.iter().find(|&&v| v as usize >= self.vertices.len()) {
            return Err(format!("vertex {v} is not in the mesh"));
        }
        let index = self.edges.len();
        for (s, w) in chain.windows(2).enumerate() {
            self.seg_arc.insert(key(w[0], w[1]), (index, s));
        }
        self.edges.push(Edge { chain, arcs });
        Ok(index)
    }

    /// Move the vertices inside edge `edge` (as added) along `outward` (per
    /// interior vertex of the chain, the unit vector away from the centre of
    /// curvature, or None where the edge is straight there), all by one
    /// distance of at most `cap`, so the polyline cuts off the arc's area. A
    /// vertex moves only if it lies on no face but `faces`, those the edge
    /// bounds. Returns whether the move hit the cap, or None when nothing
    /// moved.
    ///
    /// # Errors
    ///
    /// If there is no edge `edge` or `outward` is not one entry per interior
    /// vertex.
    pub fn move_edge(
        &mut self,
        edge: usize,
        faces: &[u32],
        outward: &[Option<Point>],
        cap: f64,
    ) -> Result<Option<bool>, String> {
        let Some(Edge { chain, arcs }) = self.edges.get(edge) else {
            return Err(format!("no edge {edge}"));
        };
        if outward.len() + 2 != chain.len().max(2) {
            return Err(format!(
                "{} outward directions for {} interior vertices",
                outward.len(),
                chain.len().saturating_sub(2)
            ));
        }
        if chain.len() < 3 {
            return Ok(None);
        }
        let mut moving: HashMap<VertexId, Point> = HashMap::new();
        for (&v, dir) in chain[1..chain.len() - 1].iter().zip(outward) {
            if !self.vert_faces[v as usize]
                .iter()
                .all(|f| faces.contains(f))
            {
                continue;
            }
            if let Some(d) = dir {
                moving.insert(v, *d);
            }
        }
        if moving.is_empty() {
            return Ok(None);
        }
        let verts = &self.vertices;
        let origin = |v: VertexId| verts[v as usize];
        let mut segments = Vec::new();
        let mut slope = 0.0;
        for (w, arc) in chain.windows(2).zip(arcs) {
            let (a, b) = (w[0], w[1]);
            let inner = &arc[1..arc.len() - 1];
            let area = vector_area(
                std::iter::once(origin(a))
                    .chain(inner.iter().copied())
                    .chain(std::iter::once(origin(b))),
            );
            let size = norm(area);
            if size == 0.0 {
                continue;
            }
            let normal = [area[0] / size, area[1] / size, area[2] / size];
            segments.push((a, b, arc, normal));
            // Moving a chord end by h sweeps about half the chord times h.
            let half = 0.5 * norm(sub(origin(b), origin(a)));
            let ends = usize::from(moving.contains_key(&a)) + usize::from(moving.contains_key(&b));
            slope -= half * ends as f64;
        }
        let at = |v: VertexId, h: f64| match moving.get(&v) {
            Some(&d) => add_scaled(origin(v), h, d),
            None => origin(v),
        };
        let residual = |h: f64| {
            let mut total = 0.0;
            for &(a, b, arc, normal) in &segments {
                // The arc from a to b, then back along the chord.
                let polygon = std::iter::once(at(a, h))
                    .chain(arc.iter().copied())
                    .chain(std::iter::once(at(b, h)));
                total += dot(vector_area(polygon), normal);
            }
            total
        };
        let (h, capped) = solve(residual, slope, cap);
        for (v, d) in moving {
            self.vertices[v as usize] = add_scaled(self.vertices[v as usize], h, d);
        }
        self.stats.edges_moved += 1;
        self.stats.edges_capped += usize::from(capped);
        Ok(Some(capped))
    }

    /// The vertices of face `fid` and those of them that may move: inside it
    /// (on none of its boundary segments) and on no face but it and its
    /// twins. None, with the reason counted, when the face cannot be
    /// corrected.
    fn interior(&mut self, fid: u32) -> Option<(Vec<VertexId>, Vec<VertexId>)> {
        let Some(segs) = self.boundary[&fid].as_ref() else {
            self.stats.faces_unmatched += 1;
            return None;
        };
        let mut on_boundary: Vec<VertexId> = segs.iter().flat_map(|&(a, b)| [a, b]).collect();
        on_boundary.sort_unstable();
        on_boundary.dedup();
        let mut face_verts: Vec<VertexId> =
            self.face_tris[&fid].iter().flatten().copied().collect();
        face_verts.sort_unstable();
        face_verts.dedup();
        let twins = &self.twins[&fid];
        let interior: Vec<VertexId> = face_verts
            .iter()
            .copied()
            .filter(|v| on_boundary.binary_search(v).is_err())
            .filter(|&v| {
                self.vert_faces[v as usize]
                    .iter()
                    .all(|f| twins.contains(f))
            })
            .collect();
        if interior.is_empty() {
            self.stats.faces_no_interior += 1;
            return None;
        }
        if segs
            .iter()
            .any(|&(a, b)| !self.seg_arc.contains_key(&key(a, b)))
        {
            self.stats.faces_unmatched += 1;
            return None;
        }
        Some((face_verts, interior))
    }

    /// Start correcting face `fid`: the point its volume is to be taken from,
    /// off the face along its mean normal and a little to one side, so no cone
    /// is flat. `move_face` finishes it. None, with the reason counted, when
    /// the face has no vertex it may move or its boundary is not matched to
    /// added edges.
    ///
    /// # Errors
    ///
    /// If the mesh has no face `fid`.
    pub fn face_apex(&mut self, fid: u32) -> Result<Option<Point>, String> {
        self.pending = None;
        if !self.face_tris.contains_key(&fid) {
            return Err(format!("the mesh has no face {fid}"));
        }
        let Some((face_verts, interior)) = self.interior(fid) else {
            return Ok(None);
        };
        let apex = self.apex(fid, &face_verts);
        self.pending = Some((fid, interior, apex));
        Ok(Some(apex))
    }

    fn apex(&self, fid: u32, face_verts: &[VertexId]) -> Point {
        let verts = &self.vertices;
        let mut centre = [0.0; 3];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &v in face_verts {
            let p = verts[v as usize];
            for i in 0..3 {
                centre[i] += p[i];
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let count = face_verts.len() as f64;
        let centre = centre.map(|c| c / count);
        let mut size = norm(sub(hi, lo));
        if size == 0.0 {
            size = 1.0;
        }
        let mut mean = [0.0; 3];
        let mut area_sum = 0.0;
        for &[p, q, r] in &self.face_tris[&fid] {
            let (p, q, r) = (verts[p as usize], verts[q as usize], verts[r as usize]);
            let c = cross(sub(q, p), sub(r, p));
            mean = add_scaled(mean, 0.5, c);
            area_sum += 0.5 * norm(c);
        }
        let mut offset = [0.0137 * size, 0.0291 * size, 0.0213 * size];
        if norm(mean) > 0.1 * area_sum {
            offset = add_scaled(offset, 0.5 * size / norm(mean), mean);
        }
        [
            centre[0] + offset[0],
            centre[1] + offset[1],
            centre[2] + offset[2],
        ]
    }

    /// Finish the face `face_apex` started: move its vertices that may move
    /// along their area-weighted normals, all by one distance of at most
    /// `cap`, so its triangles enclose `cad_volume`, the CAD face's volume
    /// from the apex with the face's own normal. The CAD face is closed to
    /// the mesh boundary by the slivers between each boundary segment and its
    /// arc. Returns whether the move hit the cap.
    ///
    /// # Errors
    ///
    /// If no face was started since the last move.
    pub fn move_face(&mut self, cad_volume: f64, cap: f64) -> Result<bool, String> {
        let Some((fid, interior, apex)) = self.pending.take() else {
            return Err("move_face needs a face started by face_apex".to_string());
        };
        let verts = &self.vertices;
        let tris = &self.face_tris[&fid];

        let mut target = cad_volume;
        for &(u, w) in self.boundary[&fid].as_ref().into_iter().flatten() {
            let (edge, s) = self.seg_arc[&key(u, w)];
            let arc = &self.edges[edge].arcs[s];
            // The sliver from the chord u-w back along the arc from w to u.
            let ends = [verts[u as usize], verts[w as usize]].into_iter();
            target += if self.edges[edge].chain[s] == w {
                fan_cone(ends.chain(arc.iter().copied()), apex)
            } else {
                fan_cone(ends.chain(arc.iter().rev().copied()), apex)
            };
        }

        let mut normals: HashMap<VertexId, Point> =
            interior.iter().map(|&v| (v, [0.0; 3])).collect();
        for &[p, q, r] in tris {
            let (pp, qq, rr) = (verts[p as usize], verts[q as usize], verts[r as usize]);
            let c = cross(sub(qq, pp), sub(rr, pp));
            for v in [p, q, r] {
                if let Some(n) = normals.get_mut(&v) {
                    *n = add_scaled(*n, 0.5, c);
                }
            }
        }
        let mut slope = 0.0;
        let mut moving: HashMap<VertexId, Point> = HashMap::new();
        for &v in &interior {
            let n = normals[&v];
            let size = norm(n);
            if size > 0.0 {
                moving.insert(v, [n[0] / size, n[1] / size, n[2] / size]);
                slope += size / 3.0;
            }
        }
        // The corners of each triangle that moves, each with its direction
        // if it moves, so the residual below is arithmetic only.
        let mut fixed = 0.0;
        let mut changing: Vec<[(Point, Option<Point>); 3]> = Vec::new();
        for &tri in tris {
            if tri.iter().any(|v| moving.contains_key(v)) {
                changing.push(tri.map(|v| (verts[v as usize], moving.get(&v).copied())));
            } else {
                let [p, q, r] = tri.map(|v| verts[v as usize]);
                fixed += cone(p, q, r, apex);
            }
        }
        let at = |(p, d): (Point, Option<Point>), h: f64| match d {
            Some(d) => add_scaled(p, h, d),
            None => p,
        };
        let residual = |h: f64| {
            let mut sum = 0.0;
            for &[p, q, r] in &changing {
                sum += cone(at(p, h), at(q, h), at(r, h), apex);
            }
            fixed + sum - target
        };
        let (h, capped) = solve(residual, slope, cap);
        for (v, d) in moving {
            self.vertices[v as usize] = add_scaled(self.vertices[v as usize], h, d);
        }
        if capped {
            self.stats.faces_capped += 1;
        } else {
            self.stats.faces_moved += 1;
        }
        Ok(capped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square in z = 0 as two triangles on face 1, and the same square
    /// split into four around a centre vertex 4 on face 2, sharing its
    /// boundary.
    fn pillow() -> FaceVolumeMesh {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let triangles = [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ];
        FaceVolumeMesh::new(vertices, &triangles, &[1, 1, 2, 2, 2, 2]).unwrap()
    }

    fn straight(a: Point, b: Point) -> Vec<Point> {
        (0..=8)
            .map(|k| {
                let s = f64::from(k) / 8.0;
                add_scaled(a, s, sub(b, a))
            })
            .collect()
    }

    #[test]
    fn solve_finds_root_of_linear_and_caps() {
        let (h, capped) = solve(|h| 2.0 * h - 1.0, 2.0, 10.0);
        assert!((h - 0.5).abs() < 1e-15 && !capped);
        let (h, capped) = solve(|h| 2.0 * h - 1.0, 2.0, 0.1);
        assert!(h == 0.1 && capped);
        assert_eq!(solve(|h| h, 1.0, 1.0), (0.0, false));
    }

    #[test]
    fn adjacency_and_boundary() {
        let m = pillow();
        assert_eq!(m.faces(), &[1, 2]);
        assert_eq!(m.boundary[&1].as_ref().unwrap().len(), 4);
        assert_eq!(m.boundary[&2].as_ref().unwrap().len(), 4);
        assert_eq!(m.neighbours(&[2]), vec![1, 2]);
        assert!(m.is_chain(&[0, 1, 2]));
        assert!(!m.is_chain(&[1, 3]));
        let pts = m.boundary_points(&[2], [-1.0; 3], [0.5, 2.0, 2.0]);
        assert_eq!(pts.iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 3]);
    }

    #[test]
    fn face_moves_to_target_volume() {
        let mut m = pillow();
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            m.add_edge(
                vec![i, j],
                vec![straight(corners[i as usize], corners[j as usize])],
            )
            .unwrap();
        }
        // Face 2's own vertex 4 is the only one that may move, and face 1
        // stays flat. A pyramid over the unit square of height h has volume
        // h / 3 from any apex in its base plane.
        assert!(m.move_face(1.0, 1.0).is_err());
        let apex = m.face_apex(2).unwrap().unwrap();
        let base: f64 = m.face_tris[&2]
            .iter()
            .map(|&t| {
                let [p, q, r] = t.map(|v| m.vertices[v as usize]);
                cone(p, q, r, apex)
            })
            .sum();
        let capped = m.move_face(base + 0.1 / 3.0, 1.0).unwrap();
        assert!(!capped);
        let z = m.vertices()[4][2];
        assert!((z.abs() - 0.1).abs() < 1e-12, "{z}");
        assert_eq!(m.vertices()[0], [0.0, 0.0, 0.0]);
        assert_eq!(m.stats().faces_moved, 1);
        assert_eq!(m.face_apex(1), Ok(None));
        assert_eq!(m.stats().faces_no_interior, 1);
        assert!(m.face_apex(7).is_err());
    }

    #[test]
    fn edge_moves_to_arc_area() {
        // A polyline 0-1-2 under a circular arc of radius 1 through 0 and 2.
        let r: f64 = 1.0;
        let a = std::f64::consts::FRAC_PI_4;
        let p = |t: f64| [r * t.sin(), r * t.cos(), 0.0];
        let vertices = vec![p(-a), p(0.0), p(a), [0.0, 0.0, 0.0]];
        let triangles = [[0, 1, 3], [1, 2, 3]];
        let mut m = FaceVolumeMesh::new(vertices, &triangles, &[1, 1]).unwrap();
        let arcs = [(-a, 0.0), (0.0, a)]
            .iter()
            .map(|&(t0, t1)| {
                (0..=8)
                    .map(|k| p(t0 + (t1 - t0) * f64::from(k) / 8.0))
                    .collect()
            })
            .collect();
        let e = m.add_edge(vec![0, 1, 2], arcs).unwrap();
        assert!(m.move_edge(e, &[1], &[], 0.5).is_err());
        let capped = m
            .move_edge(e, &[1], &[Some([0.0, 1.0, 0.0])], 0.5)
            .unwrap()
            .unwrap();
        assert!(!capped);
        // The polygon 3-0-1-2 now has the area of the sector, to the
        // resolution of the eight-segment arcs (about 2e-3 here).
        let v = m.vertices();
        let poly = vector_area([v[3], v[0], v[1], v[2]].into_iter());
        let sector = a * r * r;
        let error = poly[2].abs() / sector - 1.0;
        assert!(error.abs() < 3e-3, "{error}");
        assert!(v[1][1] > 1.0);
    }
}
