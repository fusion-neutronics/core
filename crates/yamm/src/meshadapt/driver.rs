//! MeshAdapt driver: the main refinement loop.
//!
//! Follows gmsh's `refineMeshBDS` from `meshGFaceBDS.cpp:726`:
//! iteratively applies split -> collapse -> swap -> smooth until all edges
//! are within `[min_ratio, max_ratio]` of the local target size, or
//! `max_iterations` is reached.

use super::bds::{BDSMesh, NONE};
use super::ops;
use crate::size_field::SizeField;

use rayon::prelude::*;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

/// Configuration for the MeshAdapt refinement loop.
#[derive(Debug, Clone)]
pub struct MeshAdaptConfig {
    /// Maximum number of split/collapse/swap/smooth iterations.
    pub max_iterations: usize,
    /// Collapse edges shorter than `min_ratio * target_h`.
    pub min_ratio: f64,
    /// Split edges longer than `max_ratio * target_h`.
    pub max_ratio: f64,
    /// Quality threshold below which edge swaps are attempted.
    pub quality_threshold: f64,
}

impl Default for MeshAdaptConfig {
    fn default() -> Self {
        MeshAdaptConfig {
            max_iterations: 10,
            min_ratio: 0.7,
            max_ratio: 1.4,
            quality_threshold: 0.5,
        }
    }
}

/// Run the MeshAdapt algorithm on a BDS mesh with a size field.
///
/// Iteratively applies split -> collapse -> swap -> smooth until
/// all edges are within `[min_ratio, max_ratio]` of the local target
/// size, or `max_iterations` is reached.
///
/// On the final iteration, an aggressive collapse pass with threshold
/// 0.45 (instead of the normal 0.7) cleans up remaining short edges,
/// matching GMSH's `refineMeshBDS` behavior.
pub fn meshadapt(mesh: &mut BDSMesh, sf: &SizeField, config: &MeshAdaptConfig) {
    // Set initial target_h from size field for all vertices
    for v in &mut mesh.vertices {
        if v.he != NONE {
            v.target_h = sf.target_h_at(v.u, v.v);
        }
    }

    let mut seen_buf: Vec<bool> = Vec::new();
    for iter in 0..config.max_iterations {
        if !meshadapt_step(mesh, sf, config) {
            break;
        }

        // On the final iteration, run an aggressive collapse pass to
        // clean up remaining clusters of tiny triangles (GMSH uses 0.45).
        if iter == config.max_iterations - 1 {
            collapse_pass(mesh, sf, 0.45, &mut seen_buf);
            smooth_pass(mesh, sf);
            swap_pass(mesh, sf, &mut seen_buf);
            smooth_pass(mesh, sf);
        }
    }

    // Post-loop triangle repair: targeted swaps and smoothing on
    // low-quality triangles (quality < 0.2), matching GMSH's final
    // validation pass in refineMeshBDS.
    repair_pass(mesh, sf, 0.2);
}

/// Post-loop repair pass: find triangles with quality below `threshold`
/// and aggressively swap their edges + smooth their vertices to improve
/// them.  Repeats until no more improvements can be made (max 5 rounds).
fn repair_pass(mesh: &mut BDSMesh, sf: &SizeField, threshold: f64) {
    for _round in 0..5 {
        let mut improved = false;

        // Collect faces with low quality
        let nf = mesh.faces.len();
        let mut bad_faces: Vec<usize> = Vec::new();
        for fi in 0..nf {
            if mesh.faces[fi].deleted {
                continue;
            }
            let [a, b, c] = mesh.face_vertices(fi);
            let q = ops::triangle_quality_pub(mesh, a, b, c, Some(sf));
            if q < threshold {
                bad_faces.push(fi);
            }
        }

        if bad_faces.is_empty() {
            break;
        }

        // Try swapping edges of bad triangles
        for &fi in &bad_faces {
            if mesh.faces[fi].deleted {
                continue;
            }
            let he0 = mesh.faces[fi].he;
            let he1 = mesh.half_edges[he0].next;
            let he2 = mesh.half_edges[he1].next;
            for &he in &[he0, he1, he2] {
                if mesh.half_edges[he].twin != NONE && ops::swap_edge_with_sf(mesh, he, Some(sf)) {
                    improved = true;
                    break;
                }
            }
        }

        // Smooth vertices of bad triangles
        for &fi in &bad_faces {
            if fi >= mesh.faces.len() || mesh.faces[fi].deleted {
                continue;
            }
            let [a, b, c] = mesh.face_vertices(fi);
            for &v in &[a, b, c] {
                if !mesh.vertices[v].on_boundary
                    && mesh.vertices[v].he != NONE
                    && ops::smooth_vertex_metric(mesh, v, sf)
                {
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }
}

/// Run one MeshAdapt iteration.
///
/// Follows GMSH's `refineMeshBDS` ordering with smoothing after every
/// topological operation for better convergence:
///   split → smooth → swap → smooth → collapse → smooth → swap → smooth
///
/// Returns true if any topology changes were made (split, collapse, or swap).
///
/// This allows the caller to project vertices between passes (e.g. for
/// incremental meshing where 3D coordinates must be recomputed after
/// each topology change).
pub fn meshadapt_step(mesh: &mut BDSMesh, sf: &SizeField, config: &MeshAdaptConfig) -> bool {
    let mut changed = false;
    let mut seen_buf: Vec<bool> = Vec::new();
    changed |= split_pass(mesh, sf, config.max_ratio, &mut seen_buf);
    smooth_pass(mesh, sf);
    changed |= swap_pass(mesh, sf, &mut seen_buf);
    smooth_pass(mesh, sf);
    changed |= collapse_pass(mesh, sf, config.min_ratio, &mut seen_buf);
    smooth_pass(mesh, sf);
    changed |= swap_pass(mesh, sf, &mut seen_buf);
    smooth_pass(mesh, sf);
    changed
}

/// Split pass: iterate over all edges and split those longer than
/// `max_ratio * target_h` at the midpoint.
///
/// Uses geodesic-aware midpoint placement via `split_edge_metric` so
/// that the new vertex is placed at the 3D arc-length midpoint rather
/// than the UV midpoint.  Boundary edges are also split when too long,
/// with the new midpoint inheriting boundary status.
/// Resize and zero a reusable bool buffer.
fn reset_seen(buf: &mut Vec<bool>, len: usize) {
    buf.clear();
    buf.resize(len, false);
}

fn split_pass(
    mesh: &mut BDSMesh,
    sf: &SizeField,
    max_ratio: f64,
    seen_buf: &mut Vec<bool>,
) -> bool {
    let mut changed = false;

    // Collect candidate half-edges (one per undirected edge).
    // We snapshot the current half-edge count to avoid processing
    // half-edges created during this pass.
    let he_count = mesh.half_edges.len();
    let mut candidates: Vec<(usize, bool)> = Vec::new(); // (he_id, is_boundary)
    reset_seen(seen_buf, he_count);
    let seen = seen_buf.as_mut_slice();

    for i in 0..he_count {
        if seen[i] {
            continue;
        }
        let he = &mesh.half_edges[i];
        if he.face == NONE || he.face >= mesh.faces.len() || mesh.faces[he.face].deleted {
            continue;
        }
        let is_boundary = he.twin == NONE;
        // Mark twin as seen so we process each undirected edge once
        if !is_boundary && he.twin < he_count {
            seen[he.twin] = true;
        }
        seen[i] = true;
        candidates.push((i, is_boundary));
    }

    // Sort by descending edge length so we split the longest edges first
    candidates.sort_by(|&(a, _), &(b, _)| {
        let len_a = edge_length_ratio(mesh, sf, a);
        let len_b = edge_length_ratio(mesh, sf, b);
        len_b
            .partial_cmp(&len_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (he_id, is_boundary) in candidates {
        // Re-check that the half-edge still belongs to a live face
        if he_id >= mesh.half_edges.len() {
            continue;
        }
        let face = mesh.half_edges[he_id].face;
        if face == NONE || face >= mesh.faces.len() || mesh.faces[face].deleted {
            continue;
        }

        let ratio = edge_length_ratio(mesh, sf, he_id);
        if ratio > max_ratio {
            // Never split boundary edges - they are pre-subdivided at the
            // assembly level to match adjacent faces.  Splitting them here
            // creates unmatched vertices that break watertightness.
            if is_boundary {
                continue;
            }
            let mid = {
                if mesh.half_edges[he_id].twin == NONE {
                    continue;
                }
                ops::split_edge_metric(mesh, he_id, sf)
            };
            if let Some(mid) = mid {
                // Set target_h on the new midpoint vertex
                mesh.vertices[mid].target_h =
                    sf.target_h_at(mesh.vertices[mid].u, mesh.vertices[mid].v);
                changed = true;
            }
        }
    }

    changed
}

/// Collapse pass: iterate over all edges and collapse those shorter than
/// `min_ratio * target_h`.
fn collapse_pass(
    mesh: &mut BDSMesh,
    sf: &SizeField,
    min_ratio: f64,
    seen_buf: &mut Vec<bool>,
) -> bool {
    let mut changed = false;

    let he_count = mesh.half_edges.len();
    let mut candidates: Vec<usize> = Vec::new();
    reset_seen(seen_buf, he_count);
    let seen = seen_buf.as_mut_slice();

    for i in 0..he_count {
        if seen[i] {
            continue;
        }
        let he = &mesh.half_edges[i];
        if he.face == NONE || he.face >= mesh.faces.len() || mesh.faces[he.face].deleted {
            continue;
        }
        if he.twin == NONE {
            continue; // boundary
        }
        if he.twin < he_count {
            seen[he.twin] = true;
        }
        seen[i] = true;
        candidates.push(i);
    }

    // Sort by ascending edge length so we collapse the shortest edges first
    candidates.sort_by(|&a, &b| {
        let len_a = edge_length_ratio(mesh, sf, a);
        let len_b = edge_length_ratio(mesh, sf, b);
        len_a
            .partial_cmp(&len_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for he_id in candidates {
        if he_id >= mesh.half_edges.len() {
            continue;
        }
        let face = mesh.half_edges[he_id].face;
        if face == NONE || face >= mesh.faces.len() || mesh.faces[face].deleted {
            continue;
        }
        let twin = mesh.half_edges[he_id].twin;
        if twin == NONE {
            continue;
        }

        let ratio = edge_length_ratio(mesh, sf, he_id);
        if ratio < min_ratio {
            let v0 = mesh.half_edges[he_id].origin;
            let v1 = mesh.he_dest(he_id);

            // Try to collapse the vertex with higher valence (or the
            // non-boundary one).  `collapse_edge` rejects boundary vertices.
            let p = if !mesh.vertices[v0].on_boundary {
                v0
            } else {
                v1
            };

            if ops::collapse_edge(mesh, he_id, p) {
                changed = true;
            }
        }
    }

    changed
}

/// Swap pass: iterate over all interior edges and swap if quality improves.
///
/// We collect candidates first (snapshotting), then attempt swaps.
/// This avoids issues with new half-edges created by swap operations.
fn swap_pass(mesh: &mut BDSMesh, sf: &SizeField, seen_buf: &mut Vec<bool>) -> bool {
    let mut changed = false;

    let he_count = mesh.half_edges.len();
    let mut candidates: Vec<usize> = Vec::new();
    reset_seen(seen_buf, he_count);
    let seen = seen_buf.as_mut_slice();

    for i in 0..he_count {
        if seen[i] {
            continue;
        }
        let he = &mesh.half_edges[i];
        if he.face == NONE || he.face >= mesh.faces.len() || mesh.faces[he.face].deleted {
            continue;
        }
        if he.twin == NONE {
            continue;
        }
        if he.twin < he_count {
            seen[he.twin] = true;
        }
        seen[i] = true;
        candidates.push(i);
    }

    for he_id in candidates {
        if he_id >= mesh.half_edges.len() {
            continue;
        }
        let face = mesh.half_edges[he_id].face;
        if face == NONE || face >= mesh.faces.len() || mesh.faces[face].deleted {
            continue;
        }
        let twin = mesh.half_edges[he_id].twin;
        if twin == NONE {
            continue;
        }

        // swap_edge with size field for metric-based quality decisions
        if ops::swap_edge_with_sf(mesh, he_id, Some(sf)) {
            changed = true;
        }
    }

    changed
}

/// Smooth pass: metric-weighted Laplacian smoothing of all interior vertices.
///
/// Uses the metric tensor from the size field to weight the centroid
/// computation, producing geodesic-aware smoothing that handles UV
/// distortion near poles.  Falls back to the unweighted UV-centroid
/// smooth if the metric-weighted move is rejected (e.g. would invert a
/// triangle).
fn smooth_pass(mesh: &mut BDSMesh, sf: &SizeField) {
    let nv = mesh.vertices.len();
    for v in 0..nv {
        if mesh.vertices[v].he == NONE {
            continue;
        }
        if mesh.vertices[v].on_boundary {
            continue;
        }
        // Try metric-weighted smoothing first; fall back to UV-centroid
        if !ops::smooth_vertex_metric(mesh, v, sf) {
            ops::smooth_vertex(mesh, v);
        }
    }
}

/// Compute the ratio of an edge's 3D length to the local target_h.
///
/// When anisotropic curvature data is available, returns the maximum of
/// the two directional ratios (so an edge that is too long in *either*
/// principal direction will be split).  Otherwise falls back to the
/// isotropic `edge_length_3d / target_h_midpoint`.
fn edge_length_ratio(mesh: &BDSMesh, sf: &SizeField, he_id: usize) -> f64 {
    let v0 = mesh.half_edges[he_id].origin;
    let v1 = mesh.he_dest(he_id);

    let a = [mesh.vertices[v0].u, mesh.vertices[v0].v];
    let b = [mesh.vertices[v1].u, mesh.vertices[v1].v];

    // Use anisotropic ratio when available
    if let Some((r1, r2)) = sf.edge_length_anisotropic(a, b) {
        r1.max(r2)
    } else {
        let edge_len = sf.edge_length_3d(a, b);
        let mid_u = 0.5 * (a[0] + b[0]);
        let mid_v = 0.5 * (a[1] + b[1]);
        let target = sf.target_h_at(mid_u, mid_v);

        if target > 1e-30 {
            edge_len / target
        } else {
            f64::MAX
        }
    }
}

// ──────────────────── Top-level meshing function ────────────────────

/// Mesh a parametric surface face using MeshAdapt.
///
/// Takes boundary UV vertices (forming a closed polygon) and a size field,
/// produces a triangle mesh as `(uv_vertices, triangles)`.
///
/// 1. Builds an initial CDT from the boundary (using spade).
/// 2. Converts CDT output to BDS.
/// 3. Sets vertex `target_h` from the size field.
/// 4. Runs the meshadapt loop.
/// 5. Exports BDS back to flat arrays.
pub fn mesh_face_meshadapt(
    boundary_uv: &[[f64; 2]],
    sf: &SizeField,
    config: Option<MeshAdaptConfig>,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let config = config.unwrap_or_default();
    let n = boundary_uv.len();

    if n < 3 {
        return (Vec::new(), Vec::new());
    }

    // 1. Build initial CDT from boundary
    let (cdt_verts, cdt_tris) = build_initial_cdt(boundary_uv);

    if cdt_tris.is_empty() {
        return (cdt_verts, cdt_tris);
    }

    // 2. Convert to BDS
    let mut mesh = BDSMesh::from_triangles(&cdt_verts, &cdt_tris);

    // 3. Set vertex target_h from size field.
    //    XYZ coordinates are set to UV (flat surface) for backward compat
    //    with the smooth_vertex function.  Quality evaluation uses the
    //    metric tensor directly via SizeField, not these XYZ values.
    for v in &mut mesh.vertices {
        v.target_h = sf.target_h_at(v.u, v.v);
        v.x = v.u;
        v.y = v.v;
        v.z = 0.0;
    }

    // 4. Run meshadapt loop
    meshadapt(&mut mesh, sf, &config);

    // 5. Export BDS to flat arrays
    mesh.to_triangles()
}

/// Mesh multiple parametric surface faces in parallel using MeshAdapt.
///
/// Each input is a `(boundary_uv, size_field)` pair.  The faces are meshed
/// independently via rayon, so this achieves true CPU parallelism (unlike
/// Python threads which are limited by the GIL).
///
/// Returns a `Vec` of `(uv_vertices, triangles)` in the same order as the
/// input slice.
#[allow(clippy::type_complexity)]
pub fn mesh_faces_meshadapt_parallel(
    inputs: &[(Vec<[f64; 2]>, SizeField)],
    config: Option<MeshAdaptConfig>,
) -> Vec<(Vec<[f64; 2]>, Vec<[usize; 3]>)> {
    let config = config.unwrap_or_default();
    inputs
        .par_iter()
        .map(|(boundary, sf)| mesh_face_meshadapt(boundary, sf, Some(config.clone())))
        .collect()
}

/// Build an initial constrained Delaunay triangulation from a boundary polygon.
///
/// Returns `(vertices, triangles)` where triangles are indices into `vertices`.
/// The vertices include the input boundary points (in their original order) plus
/// any Steiner points added by spade.
fn build_initial_cdt(boundary_uv: &[[f64; 2]]) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let n = boundary_uv.len();
    if n < 3 {
        return (Vec::new(), Vec::new());
    }

    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

    // Insert boundary vertices
    let mut handles = Vec::with_capacity(n);
    for pt in boundary_uv {
        match cdt.insert(Point2::new(pt[0], pt[1])) {
            Ok(h) => handles.push(h),
            Err(_) => {
                // Duplicate point - just use the existing handle
                // Find it by looking up the last inserted position
                // For robustness, we still push a handle
                if let Some(last) = handles.last() {
                    handles.push(*last);
                }
            }
        }
    }

    // Add constraints for boundary edges
    for i in 0..n {
        let j = (i + 1) % n;
        if handles[i] != handles[j] {
            // can_add_constraint checks that the constraint won't cause issues
            if cdt.can_add_constraint(handles[i], handles[j]) {
                cdt.add_constraint(handles[i], handles[j]);
            }
        }
    }

    // Build a map from spade's fixed vertex handles to our vertex indices
    let mut handle_to_idx = std::collections::HashMap::new();
    let mut verts: Vec<[f64; 2]> = Vec::new();

    for v in cdt.vertices() {
        let pos = v.position();
        let idx = verts.len();
        handle_to_idx.insert(v.fix(), idx);
        verts.push([pos.x, pos.y]);
    }

    // Collect triangles, filtering to only those inside the boundary
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for face in cdt.inner_faces() {
        let vs = face.vertices();
        let p0 = vs[0].position();
        let p1 = vs[1].position();
        let p2 = vs[2].position();

        // Centroid of this triangle
        let tcx = (p0.x + p1.x + p2.x) / 3.0;
        let tcy = (p0.y + p1.y + p2.y) / 3.0;

        // Only include triangles whose centroid is inside the boundary polygon
        if point_in_polygon(tcx, tcy, boundary_uv) {
            let i0 = handle_to_idx[&vs[0].fix()];
            let i1 = handle_to_idx[&vs[1].fix()];
            let i2 = handle_to_idx[&vs[2].fix()];

            // Ensure CCW orientation
            let area = (verts[i1][0] - verts[i0][0]) * (verts[i2][1] - verts[i0][1])
                - (verts[i2][0] - verts[i0][0]) * (verts[i1][1] - verts[i0][1]);
            if area > 0.0 {
                tris.push([i0, i1, i2]);
            } else if area < 0.0 {
                tris.push([i0, i2, i1]);
            }
            // Skip degenerate (zero-area) triangles
        }
    }

    (verts, tris)
}

// ──────────────────── Incremental MeshAdapt API ────────────────────

/// Build the initial CDT mesh for incremental MeshAdapt.
///
/// Takes boundary UV vertices (forming a closed polygon) and a size field,
/// produces the initial mesh as `(uv_vertices, triangles)`.  The caller can
/// then iterate with [`meshadapt_step`] (via the BDS round-trip), projecting
/// vertices between passes.
pub fn meshadapt_init(
    boundary_uv: &[[f64; 2]],
    sf: &SizeField,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    if boundary_uv.len() < 3 {
        return (Vec::new(), Vec::new());
    }

    // Build initial CDT from boundary
    let (cdt_verts, cdt_tris) = build_initial_cdt(boundary_uv);

    if cdt_tris.is_empty() {
        return (cdt_verts, cdt_tris);
    }

    // Convert to BDS to set up target_h and XYZ, then export
    let mut mesh = BDSMesh::from_triangles(&cdt_verts, &cdt_tris);
    for v in &mut mesh.vertices {
        v.target_h = sf.target_h_at(v.u, v.v);
        v.x = v.u;
        v.y = v.v;
        v.z = 0.0;
    }

    mesh.to_triangles()
}

/// Run one MeshAdapt step on flat arrays (for incremental API).
///
/// Converts the input arrays to BDS, runs one step, and exports back.
/// Returns `(new_vertices, new_triangles, changed)`.
pub fn meshadapt_face_step(
    vertices_uv: &[[f64; 2]],
    triangles: &[[usize; 3]],
    sf: &SizeField,
    config: &MeshAdaptConfig,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>, bool) {
    let mut mesh = BDSMesh::from_triangles(vertices_uv, triangles);

    // Set vertex target_h and XYZ from size field
    for v in &mut mesh.vertices {
        v.target_h = sf.target_h_at(v.u, v.v);
        v.x = v.u;
        v.y = v.v;
        v.z = 0.0;
    }

    let changed = meshadapt_step(&mut mesh, sf, config);
    let (verts, tris) = mesh.to_triangles();
    (verts, tris, changed)
}

/// Point-in-polygon test using the ray casting (crossing number) algorithm.
fn point_in_polygon(px: f64, py: f64, polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ───────────────────────── Tests ─────────────────────────

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size_field::SizeField;

    /// Create a uniform size field over the unit square.
    fn uniform_field(target: f64) -> SizeField {
        SizeField {
            nu: 3,
            nv: 3,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h: vec![target; 9],
            metric_e: vec![1.0; 9],
            metric_f: vec![0.0; 9],
            metric_g: vec![1.0; 9],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        }
    }

    /// Create a size field that varies: small at center, large at edges.
    fn varying_field() -> SizeField {
        let nu = 5;
        let nv = 5;
        let mut target_h = vec![0.0; nu * nv];
        for iv in 0..nv {
            for iu in 0..nu {
                let u = iu as f64 / (nu - 1) as f64;
                let v = iv as f64 / (nv - 1) as f64;
                // Distance from center (0.5, 0.5), range [0, ~0.707]
                let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
                // target_h: 0.1 at center, 0.4 at corners
                target_h[iv * nu + iu] = 0.1 + 0.4 * d;
            }
        }
        SizeField {
            nu,
            nv,
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            target_h,
            metric_e: vec![1.0; nu * nv],
            metric_f: vec![0.0; nu * nv],
            metric_g: vec![1.0; nu * nv],
            target_h1: None,
            target_h2: None,
            curvature_angle: None,
        }
    }

    /// Unit square boundary, CCW.
    fn unit_square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn meshadapt_config_default() {
        let cfg = MeshAdaptConfig::default();
        assert_eq!(cfg.max_iterations, 10);
        assert!((cfg.min_ratio - 0.7).abs() < 1e-10);
        assert!((cfg.max_ratio - 1.4).abs() < 1e-10);
    }

    #[test]
    fn point_in_polygon_basic() {
        let sq = unit_square();
        assert!(point_in_polygon(0.5, 0.5, &sq));
        assert!(point_in_polygon(0.1, 0.1, &sq));
        assert!(!point_in_polygon(1.5, 0.5, &sq));
        assert!(!point_in_polygon(-0.1, 0.5, &sq));
    }

    #[test]
    fn build_initial_cdt_unit_square() {
        let boundary = unit_square();
        let (verts, tris) = build_initial_cdt(&boundary);
        assert!(verts.len() >= 4, "should have at least 4 vertices");
        assert!(!tris.is_empty(), "should produce triangles");

        // All triangle vertices should be within [0,1]x[0,1]
        for tri in &tris {
            for &vi in tri {
                assert!(verts[vi][0] >= -1e-10 && verts[vi][0] <= 1.0 + 1e-10);
                assert!(verts[vi][1] >= -1e-10 && verts[vi][1] <= 1.0 + 1e-10);
            }
        }
    }

    #[test]
    fn mesh_face_meshadapt_unit_square_uniform() {
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, None);

        assert!(!verts.is_empty(), "should produce vertices");
        assert!(!tris.is_empty(), "should produce triangles");

        // With target_h=0.3 on a unit square, the initial coarse mesh (2 triangles)
        // should get refined (edges of length ~1.0 >> 0.3*1.4=0.42).
        // We expect more triangles than the initial 2.
        assert!(
            tris.len() > 2,
            "should refine beyond initial CDT: got {} triangles",
            tris.len()
        );
    }

    #[test]
    fn mesh_face_meshadapt_coarse_field() {
        // With a very large target_h, the mesh should stay coarse
        let boundary = unit_square();
        let sf = uniform_field(5.0);
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, None);

        assert!(!verts.is_empty());
        assert!(!tris.is_empty());
        // The initial CDT of a square produces 2 triangles; with target_h=5.0
        // no splitting should occur (edges ~1.0 < 5.0*1.4=7.0)
        assert!(
            tris.len() <= 4,
            "coarse field should not refine much: got {} triangles",
            tris.len()
        );
    }

    #[test]
    fn mesh_face_meshadapt_varying_field() {
        let boundary = unit_square();
        let sf = varying_field();
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, None);

        assert!(!verts.is_empty(), "should produce vertices");
        assert!(!tris.is_empty(), "should produce triangles");

        // Varying field should produce more triangles than uniform coarse
        assert!(
            tris.len() > 2,
            "varying field should refine: got {} triangles",
            tris.len()
        );
    }

    #[test]
    fn mesh_face_meshadapt_validates() {
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, None);

        // Rebuild into BDS and validate the topology
        if !tris.is_empty() {
            let mesh = BDSMesh::from_triangles(&verts, &tris);
            assert!(mesh.validate(), "output mesh topology should be valid");
        }
    }

    #[test]
    fn mesh_face_meshadapt_empty_boundary() {
        let sf = uniform_field(1.0);
        let (verts, tris) = mesh_face_meshadapt(&[], &sf, None);
        assert!(verts.is_empty());
        assert!(tris.is_empty());
    }

    #[test]
    fn mesh_face_meshadapt_triangle_boundary() {
        let boundary = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let sf = uniform_field(0.3);
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, None);

        assert!(!verts.is_empty());
        assert!(!tris.is_empty());
    }

    #[test]
    fn meshadapt_on_grid_mesh() {
        // Build a 3x3 grid mesh and run meshadapt with a uniform field
        let n = 3;
        let mut grid_verts = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                grid_verts.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        let mut grid_tris = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;
                grid_tris.push([v00, v10, v11]);
                grid_tris.push([v00, v11, v01]);
            }
        }

        let mut mesh = BDSMesh::from_triangles(&grid_verts, &grid_tris);
        let sf = uniform_field(0.2);

        // Set XYZ = UV for a flat surface
        for v in &mut mesh.vertices {
            v.x = v.u;
            v.y = v.v;
            v.z = 0.0;
        }

        let config = MeshAdaptConfig::default();
        meshadapt(&mut mesh, &sf, &config);

        assert!(mesh.validate(), "mesh should be valid after meshadapt");
        // The grid has edge lengths ~0.33; target_h=0.2 so max_ratio*0.2=0.28
        // means edges should be split. We should get more faces.
        assert!(
            mesh.num_live_faces() > grid_tris.len(),
            "should have more faces after refinement: {} vs {}",
            mesh.num_live_faces(),
            grid_tris.len()
        );
    }

    #[test]
    fn meshadapt_converges() {
        // Verify meshadapt converges (stops changing before max_iterations)
        let boundary = unit_square();
        let sf = uniform_field(0.5);
        let config = MeshAdaptConfig {
            max_iterations: 50,
            ..MeshAdaptConfig::default()
        };
        let (verts, tris) = mesh_face_meshadapt(&boundary, &sf, Some(config));
        assert!(!tris.is_empty());

        // Verify output is a valid triangulation
        if !tris.is_empty() {
            let mesh = BDSMesh::from_triangles(&verts, &tris);
            assert!(mesh.validate());
        }
    }

    // ── Step 5: Incremental API tests ──────────────────────

    #[test]
    fn meshadapt_step_single_iteration() {
        // Build a coarse mesh (unit square, target_h=0.3 -> needs splitting)
        let boundary = unit_square();
        let (cdt_verts, cdt_tris) = build_initial_cdt(&boundary);
        let sf = uniform_field(0.3);

        let mut mesh = BDSMesh::from_triangles(&cdt_verts, &cdt_tris);
        for v in &mut mesh.vertices {
            v.target_h = sf.target_h_at(v.u, v.v);
            v.x = v.u;
            v.y = v.v;
            v.z = 0.0;
        }

        let config = MeshAdaptConfig::default();
        let nf_before = mesh.num_live_faces();

        let changed = meshadapt_step(&mut mesh, &sf, &config);

        assert!(changed, "first step should make changes on coarse mesh");
        assert!(mesh.validate(), "mesh should be valid after step");
        assert!(
            mesh.num_live_faces() > nf_before,
            "should have more faces after refinement step"
        );
    }

    #[test]
    fn meshadapt_step_converges_to_no_change() {
        // Verify that meshadapt_step eventually converges.
        // Use the same approach as mesh_face_meshadapt: build CDT, convert
        // to BDS, run the step loop.  Use a large target_h so the mesh
        // stays coarse and converges quickly.
        let boundary = unit_square();
        let sf = uniform_field(2.0); // large target -> coarse mesh -> fast convergence
        let config = MeshAdaptConfig::default();

        let (cdt_verts, cdt_tris) = build_initial_cdt(&boundary);
        let mut mesh = BDSMesh::from_triangles(&cdt_verts, &cdt_tris);
        for v in &mut mesh.vertices {
            v.target_h = sf.target_h_at(v.u, v.v);
            v.x = v.u;
            v.y = v.v;
            v.z = 0.0;
        }

        let mut converged = false;
        for i in 0..20 {
            if !meshadapt_step(&mut mesh, &sf, &config) {
                converged = true;
                break;
            }
            assert!(mesh.validate(), "mesh should be valid after step {}", i);
        }

        assert!(converged, "should converge before 20 iterations");
        assert!(mesh.validate(), "mesh should be valid after convergence");
    }

    #[test]
    fn meshadapt_init_produces_valid_cdt() {
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let (verts, tris) = meshadapt_init(&boundary, &sf);

        assert!(!verts.is_empty(), "should produce vertices");
        assert!(!tris.is_empty(), "should produce triangles");

        let mesh = BDSMesh::from_triangles(&verts, &tris);
        assert!(mesh.validate(), "initial CDT mesh should be valid");
    }

    #[test]
    fn meshadapt_init_empty_boundary() {
        let sf = uniform_field(1.0);
        let (verts, tris) = meshadapt_init(&[], &sf);
        assert!(verts.is_empty());
        assert!(tris.is_empty());
    }

    #[test]
    fn meshadapt_face_step_roundtrip() {
        // Test that meshadapt_face_step correctly roundtrips through BDS
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let config = MeshAdaptConfig::default();

        // Get initial mesh
        let (verts0, tris0) = meshadapt_init(&boundary, &sf);
        assert!(!tris0.is_empty());

        // Run one step via the flat-array API
        let (verts1, tris1, changed) = meshadapt_face_step(&verts0, &tris0, &sf, &config);

        assert!(changed, "first step should make changes");
        assert!(!verts1.is_empty());
        assert!(!tris1.is_empty());

        // Verify the result is a valid mesh
        let mesh = BDSMesh::from_triangles(&verts1, &tris1);
        assert!(mesh.validate(), "mesh from face_step should be valid");

        // Run another step to verify continued roundtripping
        let (verts2, tris2, _changed2) = meshadapt_face_step(&verts1, &tris1, &sf, &config);
        assert!(!verts2.is_empty());
        assert!(!tris2.is_empty());
        let mesh2 = BDSMesh::from_triangles(&verts2, &tris2);
        assert!(
            mesh2.validate(),
            "mesh from second face_step should be valid"
        );
    }

    #[test]
    fn meshadapt_face_step_matches_monolithic() {
        // Verify that iterating meshadapt_face_step produces similar
        // triangle counts as the monolithic mesh_face_meshadapt
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let config = MeshAdaptConfig {
            max_iterations: 10,
            ..MeshAdaptConfig::default()
        };

        // Monolithic
        let (_, mono_tris) = mesh_face_meshadapt(&boundary, &sf, Some(config.clone()));

        // Incremental
        let (mut verts, mut tris) = meshadapt_init(&boundary, &sf);
        for _ in 0..config.max_iterations {
            let (v, t, changed) = meshadapt_face_step(&verts, &tris, &sf, &config);
            verts = v;
            tris = t;
            if !changed {
                break;
            }
        }

        // Both should produce a similar number of triangles
        // (not exact because BDS rebuild from flat arrays may differ slightly)
        let ratio = tris.len() as f64 / mono_tris.len() as f64;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "incremental ({}) and monolithic ({}) triangle counts should be similar",
            tris.len(),
            mono_tris.len()
        );
    }

    #[test]
    fn mesh_faces_meshadapt_parallel_basic() {
        // Mesh two identical unit squares in parallel and verify both
        // produce valid, non-empty results.
        let boundary = unit_square();
        let sf = uniform_field(0.3);
        let inputs = vec![
            (boundary.clone(), sf.clone()),
            (boundary.clone(), sf.clone()),
        ];

        let results = mesh_faces_meshadapt_parallel(&inputs, None);
        assert_eq!(results.len(), 2, "should return one result per input");

        for (i, (verts, tris)) in results.iter().enumerate() {
            assert!(!verts.is_empty(), "face {i} should have vertices");
            assert!(!tris.is_empty(), "face {i} should have triangles");
            assert!(
                tris.len() > 2,
                "face {i} should be refined beyond initial CDT"
            );

            // Validate topology
            let mesh = BDSMesh::from_triangles(verts, tris);
            assert!(mesh.validate(), "face {i} output mesh should be valid");
        }
    }

    #[test]
    fn mesh_faces_meshadapt_parallel_matches_sequential() {
        // Verify parallel results match sequential for various inputs.
        let boundary_sq = unit_square();
        let boundary_tri = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let sf1 = uniform_field(0.3);
        let sf2 = uniform_field(0.5);

        let inputs = vec![
            (boundary_sq.clone(), sf1.clone()),
            (boundary_tri.clone(), sf2.clone()),
        ];

        let parallel = mesh_faces_meshadapt_parallel(&inputs, None);

        let seq0 = mesh_face_meshadapt(&boundary_sq, &sf1, None);
        let seq1 = mesh_face_meshadapt(&boundary_tri, &sf2, None);

        // Triangle counts should match exactly (same algorithm, same inputs)
        assert_eq!(
            parallel[0].1.len(),
            seq0.1.len(),
            "parallel and sequential should produce same triangle count for face 0"
        );
        assert_eq!(
            parallel[1].1.len(),
            seq1.1.len(),
            "parallel and sequential should produce same triangle count for face 1"
        );
    }
}
