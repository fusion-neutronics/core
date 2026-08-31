use std::collections::HashSet;

use super::predicates::{circumcenter, circumcenter_metric};
use super::triangulation::CDT;
use super::types::{CDTInput, CDTOutput};
use crate::size_field::SizeField;

/// Ruppert-style mesh refinement with optional curvature adaptation.
///
/// When a `size_field` is present on the input, edges are measured in 3D
/// (via the metric tensor) and compared against a spatially-varying target
/// size.  Otherwise, edges are measured in raw UV coordinates against the
/// uniform `max_edge_length`.
///
/// Inserts circumcenters of triangles that violate:
/// - Edge length exceeding the (local) target
/// - Angles smaller than `min_angle`
pub fn refine(cdt: &mut CDT, input: &CDTInput) {
    let max_edge_len = input.max_edge_length.unwrap_or(f64::MAX);
    let min_angle_rad = input.min_angle.unwrap_or(20.0).to_radians();
    let size_field = input.size_field.as_ref();

    // Compute domain bounding box from input vertices for quick rejection
    let (bb_min, bb_max) = bounding_box(&input.vertices);

    // For periodic surfaces (size_field present), compute a seam exclusion
    // margin.  Steiner points within this margin of a seam edge are NOT
    // inserted, forcing the seam-adjacent triangles to use only boundary
    // vertices.  This ensures that stitching (merging opposite seam boundary
    // vertices) produces a watertight mesh.
    let seam_margin = if !input.periodic_seams {
        None
    } else {
        size_field
    }
    .map(|sf| {
        let u_range = sf.u_max - sf.u_min;
        let v_range = sf.v_max - sf.v_min;
        // Use the median target_h as the margin - excludes roughly one row
        // of cells from each seam edge.
        let mut th_vals: Vec<f64> = sf
            .target_h
            .iter()
            .copied()
            .filter(|&h| h > 0.0 && h < f64::MAX)
            .collect();
        th_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_th = if th_vals.is_empty() {
            (u_range.min(v_range)) * 0.05
        } else {
            th_vals[th_vals.len() / 2]
        };
        // Convert target_h (3D) to UV margin using metric tensor at midpoint
        let _mid_u = (sf.u_min + sf.u_max) * 0.5;
        let _mid_v = (sf.v_min + sf.v_max) * 0.5;
        let mid_idx = (sf.nv / 2) * sf.nu + (sf.nu / 2);
        let sqrt_e = sf
            .metric_e
            .get(mid_idx)
            .copied()
            .unwrap_or(1.0)
            .max(1e-10)
            .sqrt();
        let sqrt_g = sf
            .metric_g
            .get(mid_idx)
            .copied()
            .unwrap_or(1.0)
            .max(1e-10)
            .sqrt();
        // margin_u = target_h / sqrt(E), margin_v = target_h / sqrt(G)
        let margin_u = (median_th / sqrt_e).min(u_range * 0.15);
        let margin_v = (median_th / sqrt_g).min(v_range * 0.15);
        (sf.u_min, sf.u_max, sf.v_min, sf.v_max, margin_u, margin_v)
    });

    let max_iterations = 50_000;

    for _ in 0..max_iterations {
        let tris = cdt.get_triangle_positions();

        let mut worst_score: f64 = 0.0;
        let mut worst_cc: Option<[f64; 2]> = None;

        // Metric-aware circumcenter: use size field metric when available
        let compute_cc = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> [f64; 2] {
            if let Some(sf) = size_field {
                let cx = (a[0] + b[0] + c[0]) / 3.0;
                let cy = (a[1] + b[1] + c[1]) / 3.0;
                let (e, f, g) = sf.metric_at(cx, cy);
                circumcenter_metric(a, b, c, e, f, g)
            } else {
                circumcenter(a, b, c)
            }
        };

        for (a, b, c) in &tris {
            // Skip triangles with centroid outside the domain bbox
            let cx = (a[0] + b[0] + c[0]) / 3.0;
            let cy = (a[1] + b[1] + c[1]) / 3.0;
            if cx < bb_min[0] || cx > bb_max[0] || cy < bb_min[1] || cy > bb_max[1] {
                continue;
            }

            let (max_edge, target) = if let Some(sf) = size_field {
                let ab = sf.edge_length_3d(*a, *b);
                let bc = sf.edge_length_3d(*b, *c);
                let ca = sf.edge_length_3d(*c, *a);
                let max_e = ab.max(bc).max(ca);
                (max_e, sf.target_h_at(cx, cy))
            } else {
                let ab_sq = dist_sq(*a, *b);
                let bc_sq = dist_sq(*b, *c);
                let ca_sq = dist_sq(*c, *a);
                (ab_sq.max(bc_sq).max(ca_sq).sqrt(), max_edge_len)
            };

            // Check edge length criterion
            if max_edge > target {
                let score = max_edge / target;
                if score > worst_score {
                    let cc = compute_cc(*a, *b, *c);
                    if cc[0] >= bb_min[0]
                        && cc[0] <= bb_max[0]
                        && cc[1] >= bb_min[1]
                        && cc[1] <= bb_max[1]
                    {
                        // Reject circumcenters near periodic seam edges
                        if let Some((u0, u1, v0, v1, mu, mv)) = seam_margin {
                            if (cc[0] - u0).abs() < mu
                                || (cc[0] - u1).abs() < mu
                                || (cc[1] - v0).abs() < mv
                                || (cc[1] - v1).abs() < mv
                            {
                                continue;
                            }
                        }
                        worst_score = score;
                        worst_cc = Some(cc);
                    }
                }
                continue;
            }

            // Check angle criterion
            let min_a = smallest_angle(*a, *b, *c);
            if min_a < min_angle_rad {
                let score = max_edge / target;
                if score > worst_score {
                    let cc = compute_cc(*a, *b, *c);
                    if cc[0] >= bb_min[0]
                        && cc[0] <= bb_max[0]
                        && cc[1] >= bb_min[1]
                        && cc[1] <= bb_max[1]
                    {
                        // Reject circumcenters near periodic seam edges
                        if let Some((u0, u1, v0, v1, mu, mv)) = seam_margin {
                            if (cc[0] - u0).abs() < mu
                                || (cc[0] - u1).abs() < mu
                                || (cc[1] - v0).abs() < mv
                                || (cc[1] - v1).abs() < mv
                            {
                                continue;
                            }
                        }
                        worst_score = score;
                        worst_cc = Some(cc);
                    }
                }
            }
        }

        if let Some(cc) = worst_cc {
            cdt.insert_steiner(cc);
        } else {
            break;
        }
    }
}

fn bounding_box(vertices: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::MAX; 2];
    let mut max = [f64::MIN; 2];
    for v in vertices {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
    }
    (min, max)
}

/// Compute the unsigned area of a UV triangle (half the cross product).
fn tri_area_uv(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

fn dist_sq(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    dx * dx + dy * dy
}

fn smallest_angle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let ab = dist_sq(a, b).sqrt();
    let bc = dist_sq(b, c).sqrt();
    let ca = dist_sq(c, a).sqrt();

    let cos_a = ((ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca)).clamp(-1.0, 1.0);
    let cos_b = ((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc)).clamp(-1.0, 1.0);
    let angle_a = cos_a.acos();
    let angle_b = cos_b.acos();
    let angle_c = std::f64::consts::PI - angle_a - angle_b;

    angle_a.min(angle_b).min(angle_c.max(0.0))
}

/// Laplacian smoothing of interior vertices.
///
/// Moves each non-boundary vertex toward the centroid of its neighbors.
/// Uses a damping factor of 0.5 (moves halfway to the centroid each
/// iteration) for stability.  Boundary vertices (those from the original
/// input) are never moved.
///
/// The vertex positions are updated in-place via spade's `vertex_data_mut`.
/// This breaks the Delaunay property but preserves the triangle connectivity,
/// which is exactly what we want: the subsequent `into_output` call reads
/// topology and positions without relying on Delaunay queries.
pub fn smooth(cdt: &mut CDT, _input: &CDTInput, iterations: usize) {
    let boundary = cdt.boundary_handles();
    let damping = 0.5;

    for _ in 0..iterations {
        let adj = cdt.vertex_adjacency();

        // Collect the interior vertices and their target positions before
        // mutating anything (two-phase update avoids order-dependent drift).
        let mut moves: Vec<(spade::handles::FixedVertexHandle, [f64; 2])> = Vec::new();

        for &handle in adj.keys() {
            // Skip boundary / constraint vertices.
            if boundary.contains(&handle) {
                continue;
            }

            let neighbours = &adj[&handle];
            if neighbours.is_empty() {
                continue;
            }

            // Compute centroid of neighbours.
            let mut cx = 0.0;
            let mut cy = 0.0;
            for &nh in neighbours {
                let p = cdt.vertex_position(nh);
                cx += p[0];
                cy += p[1];
            }
            let n = neighbours.len() as f64;
            cx /= n;
            cy /= n;

            let old = cdt.vertex_position(handle);
            let new_pos = [
                old[0] + damping * (cx - old[0]),
                old[1] + damping * (cy - old[1]),
            ];

            moves.push((handle, new_pos));
        }

        // Apply all moves.
        for (handle, pos) in moves {
            cdt.set_vertex_position(handle, pos);
        }
    }
}

/// Enforce periodic-seam compatibility on a CDTOutput.
///
/// For periodic surfaces, the CDT boundary has vertices on opposite seam
/// edges (u_min vs u_max, v_min vs v_max) that will be merged during
/// stitching.  After CDT refinement, Steiner points near these seam edges
/// create mismatched triangulations that leave gaps after stitching.
///
/// Strategy: remove ALL Steiner points within a threshold distance of
/// each seam edge (and all triangles referencing them).  This creates a
/// clean gap at each seam that only has boundary vertices on its edges.
/// The Python stitching pipeline's `fill_boundary_holes` then closes
/// these gaps using only boundary + far-interior vertices, producing
/// compatible triangulations on both seam sides.
#[allow(dead_code)]
pub fn enforce_periodic_seams(output: &mut CDTOutput, sf: &SizeField) {
    let num_orig = output.num_original_vertices;
    let u_range = sf.u_max - sf.u_min;
    let v_range = sf.v_max - sf.v_min;
    let _eps_u = u_range * 1e-6;
    let _eps_v = v_range * 1e-6;

    // Threshold: remove Steiner points within this fraction of the
    // domain width from each seam edge.  The threshold is based on the
    // typical Steiner point spacing - we want to remove roughly the
    // first 1-2 rows of Steiner points adjacent to each seam.
    let n_steiner = output.vertices.len() - num_orig;
    if n_steiner == 0 {
        return;
    }
    // Estimate average Steiner spacing from density
    let n_tris = output.triangles.len().max(1);
    let domain_area = u_range * v_range;
    let avg_tri_area = domain_area / n_tris as f64;
    let avg_spacing = (avg_tri_area * 2.0).sqrt(); // ~edge length of equilateral
    let margin = avg_spacing * 1.5; // remove ~1.5 rows from each seam

    // Identify Steiner vertices to remove: those within `margin` of any seam
    let mut remove_verts: HashSet<usize> = HashSet::new();
    for i in num_orig..output.vertices.len() {
        let u = output.vertices[i][0];
        let v = output.vertices[i][1];
        let near_u_min = (u - sf.u_min).abs() < margin;
        let near_u_max = (u - sf.u_max).abs() < margin;
        let near_v_min = (v - sf.v_min).abs() < margin;
        let near_v_max = (v - sf.v_max).abs() < margin;
        if near_u_min || near_u_max || near_v_min || near_v_max {
            remove_verts.insert(i);
        }
    }

    if remove_verts.is_empty() {
        return;
    }

    // Remove all triangles that reference any removed Steiner vertex
    output.triangles.retain(|tri| {
        !remove_verts.contains(&tri[0])
            && !remove_verts.contains(&tri[1])
            && !remove_verts.contains(&tri[2])
    });

    // Clean up
    output.sanitize();
}

/// Compute triangle quality using the metric tensor from the size field.
///
/// Quality metric: 4*sqrt(3)*area / (a^2 + b^2 + c^2)
/// Returns 1.0 for equilateral, 0.0 for degenerate.
fn triangle_quality_metric(a: [f64; 2], b: [f64; 2], c: [f64; 2], sf: Option<&SizeField>) -> f64 {
    let (ab, bc, ca) = if let Some(sf) = sf {
        (
            sf.edge_length_3d(a, b),
            sf.edge_length_3d(b, c),
            sf.edge_length_3d(c, a),
        )
    } else {
        (
            dist_sq(a, b).sqrt(),
            dist_sq(b, c).sqrt(),
            dist_sq(c, a).sqrt(),
        )
    };
    let s = (ab + bc + ca) / 2.0;
    let area = (s * (s - ab) * (s - bc) * (s - ca)).max(0.0).sqrt();
    let denom = ab * ab + bc * bc + ca * ca;
    if denom < 1e-30 {
        return 0.0;
    }
    4.0 * 3.0_f64.sqrt() * area / denom // 1.0 = equilateral
}

/// Metric-weighted edge swapping to improve triangle quality.
///
/// For each interior edge shared by two triangles, checks if swapping
/// the diagonal improves the minimum quality of the pair. Quality is
/// measured using the metric tensor from the size field (3D lengths)
/// or Euclidean distance if no size field is provided.
///
/// Quality metric: 4*sqrt(3)*area / (a^2 + b^2 + c^2)
/// Returns 1.0 for equilateral, 0.0 for degenerate.
pub fn swap_edges(
    output: &mut CDTOutput,
    size_field: Option<&SizeField>,
    seam_verts: Option<&HashSet<usize>>,
) {
    let num_boundary = output.num_original_vertices;
    let max_passes = 10;

    for _ in 0..max_passes {
        let mut swapped_any = false;

        // Build edge-to-triangle adjacency.
        // For each edge (lo, hi), store the indices of the (up to 2) triangles sharing it.
        let mut edge_to_tris: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();

        for (ti, tri) in output.triangles.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_to_tris.entry((lo, hi)).or_default().push(ti);
            }
        }

        // Collect candidate edges: interior edges shared by exactly 2 triangles.
        // Sort for deterministic processing order - HashMap iteration order is
        // random per-process, and the swap order matters because swaps mutate
        // shared triangles in place (see the `dirty` guard below).
        let mut candidates: Vec<(usize, usize)> = edge_to_tris
            .iter()
            .filter(|(_, tris)| tris.len() == 2)
            .map(|(&edge, _)| edge)
            .collect();
        candidates.sort_unstable();

        // Track triangles already rewritten this pass. A candidate's cached
        // (ti0, ti1) become stale once either triangle is swapped by an
        // earlier candidate that shared it; applying a swap on stale adjacency
        // overwrites a triangle based on the wrong diagonal and tears interior
        // 2-triangle holes (the root cause of the nondeterministic
        // Ellipticaltorus open edges). Defer such candidates to a later pass.
        let mut dirty = vec![false; output.triangles.len()];

        for (ea, eb) in candidates {
            // Skip edges touching any seam vertex - modifying connectivity
            // near the UV domain boundary can create asymmetric topology
            // that breaks periodic seam stitching.
            if let Some(seam) = seam_verts {
                if seam.contains(&ea) || seam.contains(&eb) {
                    continue;
                }
            }
            let tri_indices = &edge_to_tris[&(ea, eb)];
            let ti0 = tri_indices[0];
            let ti1 = tri_indices[1];

            // Skip if either cached triangle was already rewritten this pass,
            // or no longer actually contains the shared edge (stale adjacency).
            if dirty[ti0] || dirty[ti1] {
                continue;
            }
            let t0 = output.triangles[ti0];
            let t1 = output.triangles[ti1];
            if !(t0.contains(&ea) && t0.contains(&eb) && t1.contains(&ea) && t1.contains(&eb)) {
                continue;
            }

            // Find the opposite vertex in each triangle (the vertex not on the shared edge).
            let tri0 = output.triangles[ti0];
            let tri1 = output.triangles[ti1];

            let opp0 = tri0.iter().find(|&&v| v != ea && v != eb).copied();
            let opp1 = tri1.iter().find(|&&v| v != ea && v != eb).copied();

            let (opp0, opp1) = match (opp0, opp1) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            // Never create edges between two boundary vertices that don't already exist.
            if opp0 < num_boundary && opp1 < num_boundary {
                continue;
            }

            // Get vertex positions.
            let pa = output.vertices[ea];
            let pb = output.vertices[eb];
            let p0 = output.vertices[opp0];
            let p1 = output.vertices[opp1];

            // Check that the new triangles have positive area (CCW orientation).
            // New tri A: opp0, opp1, ea
            let cross_a = (p1[0] - p0[0]) * (pa[1] - p0[1]) - (p1[1] - p0[1]) * (pa[0] - p0[0]);
            // New tri B: opp1, opp0, eb
            let cross_b = (p0[0] - p1[0]) * (pb[1] - p1[1]) - (p0[1] - p1[1]) * (pb[0] - p1[0]);

            if cross_a <= 0.0 || cross_b <= 0.0 {
                continue;
            }

            // Compute quality of the current pair.
            let q_old_0 = triangle_quality_metric(pa, pb, p0, size_field);
            let q_old_1 = triangle_quality_metric(pa, pb, p1, size_field);
            let q_old_min = q_old_0.min(q_old_1);

            // Compute quality of the swapped pair.
            let q_new_0 = triangle_quality_metric(p0, p1, pa, size_field);
            let q_new_1 = triangle_quality_metric(p1, p0, pb, size_field);
            let q_new_min = q_new_0.min(q_new_1);

            // Only swap if quality improves by at least 1%.
            if q_new_min <= q_old_min * 1.01 {
                continue;
            }

            // Perform the swap: replace the two triangles.
            // New tri A: opp0, opp1, ea  (CCW guaranteed by cross_a > 0)
            output.triangles[ti0] = [opp0, opp1, ea];
            // New tri B: opp1, opp0, eb  (CCW guaranteed by cross_b > 0)
            output.triangles[ti1] = [opp1, opp0, eb];
            dirty[ti0] = true;
            dirty[ti1] = true;

            swapped_any = true;
        }

        if !swapped_any {
            break;
        }
    }
}

/// Valence-based edge swapping to improve vertex regularity.
///
/// After metric-quality swap_edges, this pass targets vertex valence:
/// the ideal valence is 6 for interior vertices and 4 for boundary vertices.
/// For each interior edge shared by two triangles, we compute the sum of
/// squared deviations from ideal valence for the four vertices of the
/// quadrilateral, both before and after a hypothetical swap.  If the swap
/// reduces the total cost (and produces valid CCW triangles), it is performed.
///
/// This follows the wildmeshing approach of iterating until convergence
/// (no more beneficial swaps) or a maximum number of passes.
pub fn swap_for_valence(output: &mut CDTOutput, seam_verts: Option<&HashSet<usize>>) {
    let num_boundary = output.num_original_vertices;
    let max_passes = 5;

    for _ in 0..max_passes {
        let mut swapped = false;

        // Build vertex valence map (number of triangles incident to each vertex).
        let mut valence: Vec<usize> = vec![0; output.vertices.len()];
        for tri in &output.triangles {
            valence[tri[0]] += 1;
            valence[tri[1]] += 1;
            valence[tri[2]] += 1;
        }

        // Build edge-to-triangle adjacency.
        let mut edge_to_tris: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();

        for (ti, tri) in output.triangles.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_to_tris.entry((lo, hi)).or_default().push(ti);
            }
        }

        // Collect candidate edges: interior edges shared by exactly 2 triangles.
        // Sorted for deterministic order (HashMap iteration is random) since
        // swaps mutate shared triangles in place - see the `dirty` guard.
        let mut candidates: Vec<(usize, usize)> = edge_to_tris
            .iter()
            .filter(|(_, tris)| tris.len() == 2)
            .map(|(&edge, _)| edge)
            .collect();
        candidates.sort_unstable();

        // Defer candidates whose cached triangles were already rewritten this
        // pass: applying a swap on stale adjacency corrupts both the
        // triangulation (interior holes) and the valence accounting below.
        let mut dirty = vec![false; output.triangles.len()];

        for (ea, eb) in candidates {
            // Skip edges touching any seam vertex - modifying connectivity
            // near the UV domain boundary can create asymmetric topology
            // that breaks periodic seam stitching.
            if let Some(seam) = seam_verts {
                if seam.contains(&ea) || seam.contains(&eb) {
                    continue;
                }
            }
            let tri_indices = &edge_to_tris[&(ea, eb)];
            let ti0 = tri_indices[0];
            let ti1 = tri_indices[1];

            if dirty[ti0] || dirty[ti1] {
                continue;
            }
            let tri0 = output.triangles[ti0];
            let tri1 = output.triangles[ti1];
            if !(tri0.contains(&ea)
                && tri0.contains(&eb)
                && tri1.contains(&ea)
                && tri1.contains(&eb))
            {
                continue;
            }

            // Find opposite vertices.
            let opp0 = match tri0.iter().find(|&&v| v != ea && v != eb) {
                Some(&v) => v,
                None => continue,
            };
            let opp1 = match tri1.iter().find(|&&v| v != ea && v != eb) {
                Some(&v) => v,
                None => continue,
            };

            // Never create an edge between two boundary vertices.
            if opp0 < num_boundary && opp1 < num_boundary {
                continue;
            }

            // Compute ideal valence for each vertex.
            let ideal = |v: usize| -> i64 {
                if v < num_boundary {
                    4
                } else {
                    6
                }
            };

            // Current valences.
            let v_ea = valence[ea] as i64;
            let v_eb = valence[eb] as i64;
            let v_o0 = valence[opp0] as i64;
            let v_o1 = valence[opp1] as i64;

            // Cost before swap: sum of (valence - ideal)^2 for the 4 vertices.
            let cost_before = (v_ea - ideal(ea)).pow(2)
                + (v_eb - ideal(eb)).pow(2)
                + (v_o0 - ideal(opp0)).pow(2)
                + (v_o1 - ideal(opp1)).pow(2);

            // After swap: ea and eb each lose one triangle, opp0 and opp1 each gain one.
            let cost_after = (v_ea - 1 - ideal(ea)).pow(2)
                + (v_eb - 1 - ideal(eb)).pow(2)
                + (v_o0 + 1 - ideal(opp0)).pow(2)
                + (v_o1 + 1 - ideal(opp1)).pow(2);

            if cost_after >= cost_before {
                continue;
            }

            // Geometry check: new triangles must have positive area.
            let pa = output.vertices[ea];
            let pb = output.vertices[eb];
            let p0 = output.vertices[opp0];
            let p1 = output.vertices[opp1];

            // New tri A: opp0, opp1, ea
            let cross_a = (p1[0] - p0[0]) * (pa[1] - p0[1]) - (p1[1] - p0[1]) * (pa[0] - p0[0]);
            // New tri B: opp1, opp0, eb
            let cross_b = (p0[0] - p1[0]) * (pb[1] - p1[1]) - (p0[1] - p1[1]) * (pb[0] - p1[0]);

            if cross_a <= 0.0 || cross_b <= 0.0 {
                continue;
            }

            // Perform the swap.
            output.triangles[ti0] = [opp0, opp1, ea];
            output.triangles[ti1] = [opp1, opp0, eb];
            dirty[ti0] = true;
            dirty[ti1] = true;

            // Update valence counts.
            valence[ea] -= 1;
            valence[eb] -= 1;
            valence[opp0] += 1;
            valence[opp1] += 1;

            swapped = true;
        }

        if !swapped {
            break;
        }
    }
}

/// Split edges that are longer than the local target size.
///
/// For each INTERIOR edge longer than `split_ratio * target_h`, inserts
/// a midpoint vertex and splits the two triangles sharing that edge:
/// (a,b,c) and (a,b,d) become (a,m,c), (m,b,c), (a,m,d), (m,b,d).
///
/// BOUNDARY edges (single adjacent triangle - the face's constraint
/// polyline) are NEVER split (issue #70): a midpoint inserted here
/// exists only on THIS face, while the neighbouring face keeps the
/// shared edge's cached discretization - a unilateral Steiner point
/// that no repair can reconcile on curved shared edges (the sagitta
/// exceeds any T-junction eps). Conformality comes from both faces
/// receiving identical edge parameters; the boundary's density is the
/// input's density by design.
///
/// This runs BEFORE collapse to ensure sufficient vertex density in
/// under-refined regions.
pub fn split_long_edges(
    output: &mut CDTOutput,
    size_field: Option<&SizeField>,
    max_edge_length: Option<f64>,
    seam_verts: Option<&HashSet<usize>>,
) {
    let split_ratio = 1.4; // gmsh's MAXE_ threshold

    loop {
        // Build edge-to-triangle adjacency.
        let mut edge_to_tris: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();

        for (ti, tri) in output.triangles.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edge_to_tris.entry((lo, hi)).or_default().push(ti);
            }
        }

        // Find the longest edge that exceeds the threshold.
        // We process one edge at a time per pass to keep adjacency consistent.
        let mut best_edge: Option<(usize, usize)> = None;
        let mut best_ratio: f64 = 0.0;

        for (&(ea, eb), tris) in edge_to_tris.iter() {
            // Never split a boundary edge (single adjacent triangle):
            // it lies on the face's constraint polyline, which adjacent
            // faces share point-for-point (issue #70 - see above).
            if tris.len() < 2 {
                continue;
            }
            // Skip edges touching any seam vertex - modifying connectivity
            // near the UV domain boundary can create asymmetric topology
            // that breaks periodic seam stitching.
            if let Some(seam) = seam_verts {
                if seam.contains(&ea) || seam.contains(&eb) {
                    continue;
                }
            }
            let pa = output.vertices[ea];
            let pb = output.vertices[eb];

            let (edge_len, target_h) = if let Some(sf) = size_field {
                let el = sf.edge_length_3d(pa, pb);
                let um = (pa[0] + pb[0]) * 0.5;
                let vm = (pa[1] + pb[1]) * 0.5;
                (el, sf.target_h_at(um, vm))
            } else if let Some(mel) = max_edge_length {
                let dx = pb[0] - pa[0];
                let dy = pb[1] - pa[1];
                ((dx * dx + dy * dy).sqrt(), mel)
            } else {
                continue;
            };

            let ratio = edge_len / target_h;
            if ratio > split_ratio && ratio > best_ratio {
                best_ratio = ratio;
                best_edge = Some((ea, eb));
            }
        }

        // No edge exceeds the threshold - done.
        let (ea, eb) = match best_edge {
            Some(e) => e,
            None => break,
        };

        // Insert midpoint vertex.
        let pa = output.vertices[ea];
        let pb = output.vertices[eb];
        let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
        let m = output.vertices.len();
        output.vertices.push(mid);

        // Get the triangle indices sharing this edge.
        let tris = edge_to_tris[&(ea, eb)].clone();

        // Collect new triangles and mark old ones for replacement.
        let mut new_tris: Vec<[usize; 3]> = Vec::new();
        let mut removed_tis: Vec<usize> = Vec::new();

        for &ti in &tris {
            let tri = output.triangles[ti];
            // Find the opposite vertex (the one not on the edge).
            let opp = match tri.iter().find(|&&v| v != ea && v != eb) {
                Some(&v) => v,
                None => continue,
            };

            // Split into (ea, m, opp) and (m, eb, opp), choosing the winding
            // that maintains CCW orientation.
            let po = output.vertices[opp];

            let cross1 = (mid[0] - pa[0]) * (po[1] - pa[1]) - (mid[1] - pa[1]) * (po[0] - pa[0]);
            let cross2 = (pb[0] - mid[0]) * (po[1] - mid[1]) - (pb[1] - mid[1]) * (po[0] - mid[0]);

            if cross1 > 0.0 {
                new_tris.push([ea, m, opp]);
            } else {
                new_tris.push([ea, opp, m]);
            }

            if cross2 > 0.0 {
                new_tris.push([m, eb, opp]);
            } else {
                new_tris.push([m, opp, eb]);
            }

            removed_tis.push(ti);
        }

        // Mark removed triangles with sentinel, then add new ones.
        for &ti in &removed_tis {
            output.triangles[ti] = [usize::MAX, usize::MAX, usize::MAX];
        }
        output.triangles.retain(|tri| tri[0] != usize::MAX);
        output.triangles.extend(new_tris);
    }
}

/// Collapse short interior edges as a post-processing step.
///
/// After Ruppert refinement and Laplacian smoothing, the mesh may contain
/// edges that are significantly shorter than the local target size.  This
/// function iteratively collapses such edges (merging vertex `b` into vertex
/// `a`) to reduce triangle count while maintaining mesh quality.
///
/// Only edges where **both** endpoints are interior (Steiner) vertices are
/// eligible for collapse.  Boundary vertices (indices < `num_boundary_vertices`)
/// are never touched.  Each collapse is validated: the resulting triangles must
/// have positive area (no inversions or degeneracies).
///
/// The collapse ratio is 0.7 of the local target edge length, matching the
/// heuristic used by gmsh's MeshAdapt.
pub fn collapse_short_edges(
    output: &mut CDTOutput,
    size_field: Option<&SizeField>,
    max_edge_length: Option<f64>,
) {
    // Use a more conservative ratio for adaptive (size_field) mode to
    // avoid over-collapsing on curved surfaces where triangle count
    // directly impacts geometric accuracy.
    let collapse_ratio = if size_field.is_some() { 0.3 } else { 0.7 };
    let num_boundary = output.num_original_vertices;

    // Iterate until no more collapses are possible.
    loop {
        let mut collapsed_any = false;

        // Collect all unique edges with their lengths and target thresholds.
        // We process shortest-first so each pass catches the most obvious
        // candidates and avoids cascading problems.
        let mut edge_candidates: Vec<(usize, usize, f64)> = Vec::new();

        // Use a set to avoid duplicate edges.
        let mut seen_edges = std::collections::HashSet::new();

        for tri in &output.triangles {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if !seen_edges.insert((lo, hi)) {
                    continue;
                }
                // Both endpoints must be interior (Steiner) vertices.
                if lo < num_boundary || hi < num_boundary {
                    continue;
                }
                let pa = output.vertices[a];
                let pb = output.vertices[b];

                let (edge_len, target_h) = if let Some(sf) = size_field {
                    let el = sf.edge_length_3d(pa, pb);
                    let um = (pa[0] + pb[0]) * 0.5;
                    let vm = (pa[1] + pb[1]) * 0.5;
                    (el, sf.target_h_at(um, vm))
                } else if let Some(mel) = max_edge_length {
                    let dx = pb[0] - pa[0];
                    let dy = pb[1] - pa[1];
                    ((dx * dx + dy * dy).sqrt(), mel)
                } else {
                    continue;
                };

                let threshold = collapse_ratio * target_h;
                if edge_len < threshold {
                    edge_candidates.push((a, b, edge_len));
                }
            }
        }

        // Sort by edge length ascending - collapse shortest first.
        edge_candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Track which vertices have been removed this pass so we don't
        // collapse into already-removed vertices.
        let mut removed = std::collections::HashSet::new();

        for (a, b, _len) in &edge_candidates {
            let a = *a;
            let b = *b;

            // Skip if either vertex was already removed this pass.
            if removed.contains(&a) || removed.contains(&b) {
                continue;
            }

            // Try to collapse edge: merge vertex b into vertex a.
            // 1. Identify triangles that share the edge (a,b) - these are removed.
            // 2. Identify triangles that reference b but not a - these get b remapped to a.
            // 3. Validate that remapped triangles remain valid (positive area).

            let mut edge_tris = Vec::new(); // indices of triangles sharing edge (a,b)
            let mut remap_tris = Vec::new(); // indices of triangles referencing b but not a

            for (ti, tri) in output.triangles.iter().enumerate() {
                let has_a = tri[0] == a || tri[1] == a || tri[2] == a;
                let has_b = tri[0] == b || tri[1] == b || tri[2] == b;
                if has_a && has_b {
                    edge_tris.push(ti);
                } else if has_b {
                    remap_tris.push(ti);
                }
            }

            // Validate: check that remapping b -> a in each remap_tri produces
            // a valid triangle (positive area, no duplicate vertices, no
            // excessively long edges, and no duplicate of an existing triangle).
            let mut valid = true;

            // Build a set of existing triangle keys (sorted vertex triples)
            // so we can detect duplicates created by the remap.
            let mut existing_tris: std::collections::HashSet<[usize; 3]> =
                std::collections::HashSet::new();
            for (ti, tri) in output.triangles.iter().enumerate() {
                if tri[0] == usize::MAX {
                    continue; // sentinel
                }
                // Skip edge_tris (they'll be removed) and remap_tris (they'll change)
                if edge_tris.contains(&ti) || remap_tris.contains(&ti) {
                    continue;
                }
                let mut key = [tri[0], tri[1], tri[2]];
                key.sort();
                existing_tris.insert(key);
            }

            // When a size_field is present, compute the total 3D area of the
            // old triangles that will be modified (remap_tris) so we can check
            // the area-ratio constraint afterwards.
            let mut old_area_sum = 0.0;
            let mut new_area_sum = 0.0;

            for &ti in &remap_tris {
                let mut new_tri = output.triangles[ti];

                // Compute old triangle area (before remap) for area-ratio check.
                if size_field.is_some() {
                    let op0 = output.vertices[new_tri[0]];
                    let op1 = output.vertices[new_tri[1]];
                    let op2 = output.vertices[new_tri[2]];
                    old_area_sum += tri_area_uv(op0, op1, op2);
                }

                for v in new_tri.iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }

                // Check for degenerate triangle (duplicate vertex indices).
                if new_tri[0] == new_tri[1] || new_tri[1] == new_tri[2] || new_tri[0] == new_tri[2]
                {
                    valid = false;
                    break;
                }

                // Check for duplicate triangle (would create non-manifold edges).
                let mut key = [new_tri[0], new_tri[1], new_tri[2]];
                key.sort();
                if existing_tris.contains(&key) {
                    valid = false;
                    break;
                }
                existing_tris.insert(key);

                let p0 = output.vertices[new_tri[0]];
                let p1 = output.vertices[new_tri[1]];
                let p2 = output.vertices[new_tri[2]];

                // Check orientation (must remain positive / CCW).
                let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
                if cross <= 0.0 {
                    valid = false;
                    break;
                }

                // Accumulate new area for area-ratio check.
                if size_field.is_some() {
                    new_area_sum += tri_area_uv(p0, p1, p2);
                }

                // Check that no new edge exceeds the target edge length.
                // For adaptive mode (size_field), use a tighter 1.2× bound
                // to prevent creating over-sized triangles on curved surfaces.
                // For uniform mode, use the original 1.5× bound.
                let edges = [[p0, p1], [p1, p2], [p2, p0]];
                for [ea, eb] in &edges {
                    if let Some(sf) = size_field {
                        let el = sf.edge_length_3d(*ea, *eb);
                        let um = (ea[0] + eb[0]) * 0.5;
                        let vm = (ea[1] + eb[1]) * 0.5;
                        let th = sf.target_h_at(um, vm);
                        if el > th * 1.2 {
                            valid = false;
                            break;
                        }
                    } else if let Some(mel) = max_edge_length {
                        let dx = eb[0] - ea[0];
                        let dy = eb[1] - ea[1];
                        let el = (dx * dx + dy * dy).sqrt();
                        if el > mel * 1.5 {
                            valid = false;
                            break;
                        }
                    }
                }
                if !valid {
                    break;
                }
            }

            // Area-ratio check: when a size_field is present, reject the
            // collapse if the total UV area of the modified triangles changes
            // too much.  This prevents collapses that would distort the
            // surface approximation on curved geometry.
            if valid && size_field.is_some() && old_area_sum > 0.0 {
                let ratio = new_area_sum / old_area_sum;
                if !(0.3..=3.0).contains(&ratio) {
                    valid = false;
                }
            }

            if !valid {
                continue;
            }

            // Perform the collapse.
            // Remap b -> a in the remap triangles.
            for &ti in &remap_tris {
                for v in output.triangles[ti].iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
            }

            // Mark edge triangles for removal (set sentinel).
            for &ti in &edge_tris {
                output.triangles[ti] = [usize::MAX, usize::MAX, usize::MAX];
            }

            removed.insert(b);
            collapsed_any = true;
        }

        // Remove sentinel triangles.
        output.triangles.retain(|tri| tri[0] != usize::MAX);

        if !collapsed_any {
            break;
        }
    }
}
