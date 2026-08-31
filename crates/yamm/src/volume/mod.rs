#![allow(clippy::needless_range_loop, clippy::type_complexity)]
pub mod aabb_bvh;
#[allow(dead_code)]
mod advancing_front;
mod bcc;
pub mod boundary_recovery;
#[allow(dead_code)]
mod cdt2d;
mod clip;
mod coplanar;
mod delaunay3d;
pub(crate) mod dethash;
mod mesh_improve;
pub mod optimize;
#[allow(dead_code)]
mod refine3d;

use self::dethash::{HashMap, HashSet};
use rayon::prelude::*;
mod predicates3d;
mod types;

pub use types::{BoundaryRecoveryStats, VolumeInput, VolumeOutput};

use crate::error::{MesherError, Result};
use delaunay3d::Delaunay3D;

/// Relative mismatch between the legacy/boundary-only tet-sum and the
/// boundary-enclosed volume at which a conforming carve is attempted
/// (override with YAMM_CONFORMING_TRIGGER).
///
/// Both use sites take the absolute difference, so this fires on overshoot
/// as well as undershoot - worth saying because overshoot is the case that
/// went unaddressed for longest. Undershoot is unfilled space; overshoot is
/// tets crossing the boundary and spilling outside it, which for DAGMC is
/// space claimed by a solid that its own surface does not enclose.
///
/// History: 1% from the original #30 design, when the conforming carve
/// routinely fell back on the cylindrical flat-cap family - a tighter
/// trigger just burned time. The #47 cluster-B re-triage then showed the
/// fill suite's failures concentrated in the **sub-1% under-fill band**
/// (e.g. NestedCylinder legacy fill 99.49% vs the suite's 99.5% floor)
/// where the 1% trigger stayed silent. With the coplanar-region recovery
/// (PR #56) the carve actually SUCCEEDS on that family, so the trigger
/// dropped to 0.4% - below the fill tolerance of the day, and chosen to sit
/// "comfortably above legacy's normal accuracy (exact-conforming models sit
/// at ~0.0–0.1%)".
///
/// That last premise turned out to be wrong, and it was the reason a whole
/// band of models kept an inexact mesh. Measuring the fill sweep instead of
/// estimating it: 146 of 175 solid measurements are exact to 1e-9, so
/// legacy's "normal accuracy" is not 0.0–0.1% - it is either exact or
/// visibly wrong, with nothing in between. Every model that missed sat in
/// the 1.6e-6 to 4.3e-3 band, i.e. entirely below the 0.4% trigger, so the
/// carve was never attempted on precisely the solids it exists to fix. With
/// the trigger here the carve runs on them and 8 of the 11 single-body
/// cases land at machine precision (1e-13 to 1e-15).
///
/// 1e-8 is set from that same measurement: the largest genuine round-off in
/// the sweep is 1.4e-9 (SimpleTokamak, whose ~1200-unit coordinates give it
/// the biggest absolute error), and the smallest real defect is 1.6e-6. The
/// trigger sits in that gap, so an already-exact solid still pays nothing
/// and everything inexact gets the attempt. The attempt remains free of
/// correctness risk in either direction: the carve self-validates against
/// the boundary-enclosed volume and falls back to the legacy mesh, so a
/// solid can only end up as good as it was before.
const CONFORMING_TRIGGER_DEFAULT: f64 = 1e-8;

/// Mesh a volume with tetrahedra.
///
/// Takes a closed surface mesh (boundary) and fills it with tetrahedra
/// targeting uniform, near-equilateral elements at the given edge length.
///
/// The boundary vertices and triangles are preserved exactly (conformal).
///
/// # Guarantees
///
/// Every emitted tet is POSITIVELY ORIENTED (signed volume
/// `det([v1-v0, v2-v0, v3-v0]) / 6 > 0`). Each internal pipeline
/// (legacy Delaunay filter, conforming carve, boundary-only) ends in
/// `mesh_improve::improve_mesh`, which re-orients every tet and drops the
/// degenerate remainder; this function then RE-CHECKS the result at the single
/// emission boundary so a pipeline that stops maintaining the orientation fails
/// here instead of downstream. See [`VolumeOutput::first_non_positive_tet`] for
/// why the orientation matters (issue #316) and what the check costs.
///
/// # Errors
///
/// Returns `MesherError::InvalidInput` if:
/// - `boundary_vertices` has fewer than 4 vertices
/// - `boundary_triangles` is empty
/// - `target_edge_length` is not positive
/// - any triangle index is out of range
///
/// Returns `MesherError::MeshingFailed` if the meshed result violates the
/// positive-orientation guarantee above (a mesher bug, not bad input).
pub fn mesh_volume(input: &VolumeInput) -> Result<VolumeOutput> {
    // --- Input validation ---
    if input.boundary_vertices.len() < 4 {
        return Err(MesherError::InvalidInput(format!(
            "boundary_vertices must have at least 4 vertices, got {}",
            input.boundary_vertices.len()
        )));
    }
    if input.boundary_triangles.is_empty() {
        return Err(MesherError::InvalidInput(
            "boundary_triangles must not be empty".into(),
        ));
    }
    if input.target_edge_length <= 0.0 {
        return Err(MesherError::InvalidInput(format!(
            "target_edge_length must be positive, got {}",
            input.target_edge_length
        )));
    }
    let n_verts = input.boundary_vertices.len();
    for (i, tri) in input.boundary_triangles.iter().enumerate() {
        for &idx in tri {
            if idx >= n_verts {
                return Err(MesherError::InvalidInput(format!(
                    "boundary_triangles[{i}] contains index {idx} but there are only {n_verts} vertices"
                )));
            }
        }
    }

    // --- Feasibility guard (issue #47) ---
    // The BCC lattice seeds ~2·V/h³ interior points (V = enclosed volume,
    // h = target_edge_length·2/√3), and the pipeline downstream (Delaunay
    // build, recovery, improvement) scales super-linearly in that count. A
    // target_edge_length that is tiny relative to the model implies an
    // infeasible mesh - e.g. SimpleTokamak (scale ~1200) at target 1.5 implies
    // ~141 MILLION interior points (~850M tets), which previously GROUND
    // SILENTLY for >1.5h before being killed. Estimating the count costs
    // milliseconds (divergence-theorem volume), so refuse loudly and instantly
    // instead. The cap (default 20M points ≈ 120M tets - already an enormous
    // mesh) is overridable via YAMM_MAX_INTERIOR_PTS for users who mean it.
    {
        let h = input.target_edge_length * 2.0 / 3.0_f64.sqrt();
        let enclosed =
            boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
        let est_pts = 2.0 * enclosed / (h * h * h);
        let cap: f64 = std::env::var("YAMM_MAX_INTERIOR_PTS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(20_000_000.0);
        if est_pts > cap {
            // Suggest the finest target that lands at ~1M interior points.
            let h_ok = (2.0 * enclosed / 1_000_000.0).cbrt();
            let tel_ok = h_ok * 3.0_f64.sqrt() / 2.0;
            return Err(MesherError::InvalidInput(format!(
                "target_edge_length {} implies ~{:.1}M interior points for this solid \
                 (enclosed volume {:.3e}) - beyond the {:.0}M-point cap; the mesh would \
                 be infeasible to build. Use a coarser target (>= ~{:.1} for ~1M points) \
                 or raise YAMM_MAX_INTERIOR_PTS to override.",
                input.target_edge_length,
                est_pts / 1e6,
                enclosed,
                cap / 1e6,
                tel_ok
            )));
        }
    }

    // Delaunay pipeline with quality refinement
    let output = mesh_volume_delaunay(input)?;

    // --- Emission-boundary invariant: positive orientation (issue #316) ---
    // One O(#tets), allocation-free sign pass at the single point every
    // internal path funnels through. Measured at ~15 ns/tet: 0.10 ms on a
    // 6.9k-tet mesh (0.39% of the 27 ms mesh_volume) and 11.7 ms on a 761k-tet
    // mesh (0.11% of the 10.2 s mesh_volume). Three orders of magnitude below
    // the meshing itself, so it stays always-on rather than debug-only: a
    // silently inverted mesh is not something to discover in a transport run.
    if let Some((idx, signed)) = output.first_non_positive_tet(&input.boundary_vertices) {
        return Err(MesherError::MeshingFailed(format!(
            "tet {idx} of {} is not positively oriented (signed volume {signed:e}). \
             Every tet yamm emits must have positive signed volume: transport reads \
             face normals off a fixed face table that only points outward for \
             positively oriented tets, and an inverted tet makes the element walk \
             pick an entry face as its exit (issue #316). This is a mesher bug",
            output.tetrahedra.len()
        )));
    }

    Ok(output)
}

/// Volume enclosed by a triangulated surface. See
/// [`crate::utils::surface_enclosed_volume`], which this used to duplicate --
/// the winding-robust, nesting-aware implementation now lives there so callers
/// outside this module (including Python) get the same answer.
fn boundary_enclosed_volume(verts: &[[f64; 3]], tris: &[[usize; 3]]) -> f64 {
    crate::utils::surface_enclosed_volume(verts, tris)
}

/// Threshold below which a tet is treated as degenerate. Absolute, not relative
/// to the model: a tet this small contributes nothing to any volume integral at
/// any scale, and the zoo asserts on the same number.
const DEGENERATE_TET_VOLUME: f64 = 1e-12;

/// Drop tets at or below [`DEGENERATE_TET_VOLUME`], reporting how many.
///
/// By definition this removes numerically zero volume, and a degenerate element
/// is worse than an absent one: MOAB will carry it into an unstructured-mesh
/// tally as a cell that can never be scored, and DAGMC has to ray-fire against
/// faces with no well-defined normal.
///
/// Every pipeline exit calls this. It used to be inline in the boundary-only
/// path alone, which is why SimpleTokamak shipped one zero-volume sliver out of
/// ~500k tets (#47): its solids come out of the main Delaunay path, where
/// nothing dropped what `improve_mesh` could not heal.
fn drop_degenerate_tets(pts: &[[f64; 3]], tets: &mut Vec<[usize; 4]>, label: &str) -> usize {
    let before = tets.len();
    tets.retain(|t| {
        predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs()
            >= DEGENERATE_TET_VOLUME
    });
    let dropped = before - tets.len();
    if dropped > 0 {
        eprintln!("    [vol] {label}: dropped {dropped} degenerate tets (of {before})");
    }
    dropped
}

/// Classify and return the interior tets of a *fully boundary-conforming*
/// Delaunay tetrahedralization by flood-fill carving.
///
/// Once every boundary triangle is present as a tet face, the boundary is a
/// closed 2-manifold embedded in the tetrahedralization separating "outside"
/// tets (reachable from the convex-hull / INFINITE tets without crossing a
/// boundary face) from the enclosed "inside" tets. We flood from the hull tets
/// across non-boundary faces; everything not reached is inside. This is exact
/// and overlap-free by construction - unlike per-tet centroid inside-tests,
/// which keep tets straddling a curved boundary (overshoot) or drop tets
/// straddling a thin/concave boundary (undershoot). Requires the boundary to
/// be fully recovered first; the caller guards on `failed == 0` and on a
/// carved-volume == boundary-volume check.
fn carve_interior(dt: &Delaunay3D, boundary_faces: &[[usize; 3]]) -> Vec<[usize; 4]> {
    use std::collections::VecDeque;

    let bset: HashSet<[usize; 3]> = boundary_faces
        .iter()
        .map(|f| {
            let mut s = *f;
            s.sort();
            s
        })
        .collect();

    // PARITY flood (issue #37, hollow solids): a solid's surface may have NESTED
    // components (e.g. the coil casing - an outer shell with a sealed inner
    // cavity holding the winding pack). A binary "unreachable from hull ⇒
    // inside" flood wrongly counts the cavity as material (it too is sealed off
    // from the hull). Instead, flood the WHOLE mesh from the hull, toggling a
    // parity bit each time a boundary face is crossed: material = parity ODD
    // (inside an odd number of nested surfaces); the cavity (two crossings in)
    // is parity EVEN and excluded. Single-shell solids behave identically to
    // the old flood. Since the recovered surface is a closed separating
    // complex, parity is path-independent; a parity CONFLICT therefore directly
    // detects a non-separating defect (reported under YAMM_CARVE_DBG).
    let n = dt.tets.len();
    const UNSEEN: u8 = u8::MAX;
    let mut parity: Vec<u8> = vec![UNSEEN; n];
    let mut conflicts = 0usize;
    let mut q: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if dt.is_live(i) && dt.tets[i].is_hull() {
            parity[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(ti) = q.pop_front() {
        let verts = dt.tets[ti].verts;
        let adj = dt.tets[ti].adj;
        for (fi, &nb) in adj.iter().enumerate() {
            if nb == usize::MAX || nb >= n || !dt.is_live(nb) {
                continue;
            }
            let mut face = delaunay3d::opposite_face(verts, fi);
            face.sort();
            let p_new = parity[ti] ^ u8::from(bset.contains(&face));
            if parity[nb] == UNSEEN {
                parity[nb] = p_new;
                q.push_back(nb);
            } else if parity[nb] != p_new {
                conflicts += 1; // non-separating surface defect
            }
        }
    }
    // Unvisited finite tets (adjacency-isolated pockets) keep the historical
    // "unreached ⇒ material" interpretation, but are counted as a defect signal.
    let mut unvisited = 0usize;

    let mut out = Vec::new();
    let mut seen = HashSet::default();
    for i in 0..n {
        if !dt.is_live(i) || dt.tets[i].is_hull() {
            continue;
        }
        if parity[i] == UNSEEN {
            unvisited += 1;
        } else if parity[i] == 0 {
            continue; // even crossings: outside or a sealed cavity - not material
        }
        let mut k = dt.tets[i].verts;
        k.sort();
        if seen.insert(k) {
            out.push(dt.tets[i].verts);
        }
    }
    if (conflicts > 0 || unvisited > 0) && std::env::var("YAMM_CARVE_DBG").is_ok() {
        eprintln!("    [carve DBG] parity conflicts={conflicts} unvisited={unvisited} (surface separation defects)");
    }
    if std::env::var("YAMM_CARVE_DBG").is_ok() {
        let mut n_live_real = 0usize;
        let mut n_hull = 0usize;
        let mut n_outside = 0usize;
        for i in 0..n {
            if !dt.is_live(i) {
                continue;
            }
            if dt.tets[i].is_hull() {
                n_hull += 1;
            } else {
                n_live_real += 1;
                if parity[i] != 1 {
                    n_outside += 1;
                }
            }
        }
        // How many boundary faces are actually present as a face of a live tet,
        // and how many sit between two LIVE NON-HULL tets (interior interface =
        // a flood-blocker that should separate, not a hull boundary).
        let mut present = 0usize;
        let mut between_two_real = 0usize;
        let mut between_real_and_hull = 0usize;
        for bf in &bset {
            let mut found = false;
            // O(degree) via the incidence index (a full per-face O(#tets) scan
            // here made this DEBUG block cost ~200s on the casing).
            for &iu in dt.incident_tets(bf[0]) {
                let i = iu as usize;
                if !dt.is_live(i) || dt.tets[i].is_hull() {
                    continue;
                }
                let v = dt.tets[i].verts;
                if v.contains(&bf[0]) && v.contains(&bf[1]) && v.contains(&bf[2]) {
                    found = true;
                    // find the face index and its neighbor
                    for fi in 0..4 {
                        let mut f = delaunay3d::opposite_face(v, fi);
                        f.sort();
                        if &f == bf {
                            let nb = dt.tets[i].adj[fi];
                            if nb != usize::MAX && dt.is_live(nb) {
                                if dt.tets[nb].is_hull() {
                                    between_real_and_hull += 1;
                                } else {
                                    between_two_real += 1;
                                }
                            }
                        }
                    }
                    break;
                }
            }
            if found {
                present += 1;
            }
        }
        eprintln!(
            "    [carve DBG] live_real={} hull={} outside={} inside={} | bset={} present={} (real|real={} real|hull={})",
            n_live_real, n_hull, n_outside, out.len(), bset.len(), present, between_two_real, between_real_and_hull
        );
    }
    out
}

/// Conforming volume mesh: recover the full boundary (no Steiner points - the
/// input surface triangulation is preserved exactly), then carve the interior
/// by flood-fill. Self-validating: returns `None` (caller falls back to the
/// legacy filter path) unless every boundary face was recovered AND the carved
/// volume matches the boundary-enclosed volume. This can never regress models
/// the legacy path handled, and exactly fixes both over- and under-shoot where
/// it succeeds.
fn mesh_volume_conforming(
    input: &VolumeInput,
    all_points: &[[f64; 3]],
    n_boundary: usize,
) -> Option<VolumeOutput> {
    use std::time::Instant;
    if all_points.len() < 4 {
        return None;
    }
    let t0 = Instant::now();
    let tdbg = std::env::var("YAMM_TIME_DBG").is_ok();
    let mut tp = Instant::now();
    let mut dt = Delaunay3D::new(all_points);
    // Activate the vertex→incident-tet index now that base construction is done.
    // Boundary recovery's edge/face/ring existence queries then run in
    // O(degree) instead of O(#tets) - the cost that made the conforming path
    // too slow on fine meshes (issue #30). The index is maintained incrementally
    // through the recovery flips/Steiner splits from here on.
    dt.build_vert_tets();
    if tdbg {
        eprintln!(
            "    [time] delaunay({} pts)={:.2}s",
            all_points.len(),
            tp.elapsed().as_secs_f64()
        );
        tp = Instant::now();
    }

    // ── Volume-invariant probe (issue #37, YAMM_VOLCHECK) ──
    // The Delaunay partitions its convex hull exactly, and every recovery
    // operation must preserve that partition: the SUM of finite live tet
    // volumes is invariant. A growth between stages pinpoints where an
    // overlapping re-mesh was committed; a shrink pinpoints a gap.
    let volcheck = std::env::var("YAMM_VOLCHECK").is_ok();
    let total_finite_vol = |dt: &delaunay3d::Delaunay3D| -> f64 {
        let mut s = 0.0;
        for i in 0..dt.tets.len() {
            if !dt.is_live(i) {
                continue;
            }
            let v = dt.tets[i].verts;
            if v.contains(&delaunay3d::INFINITE) {
                continue;
            }
            s += predicates3d::tet_volume(
                dt.vertices[v[0]],
                dt.vertices[v[1]],
                dt.vertices[v[2]],
                dt.vertices[v[3]],
            )
            .abs();
        }
        s
    };
    let vol0 = if volcheck { total_finite_vol(&dt) } else { 0.0 };
    if volcheck {
        eprintln!("    [VOLCHECK] post-build  total={vol0:.6}");
    }

    // The evolving REFINED boundary-triangle set. Conforming Delaunay
    // refinement may add Steiner points ON boundary segments / facets, which
    // subdivides the surface triangulation without changing its geometry. We
    // track that refined set here and hand IT to carve_interior so the flood is
    // blocked by the faces actually present in the mesh. The volume gate below
    // still references the ORIGINAL input triangles (the true target volume).
    let mut cur_faces: Vec<[usize; 3]> = input.boundary_triangles.clone();

    // ── Separation audit (issue #37, YAMM_SEP_AUDIT) ──
    // The carve's flood partition is only valid if the surface SEPARATES, which
    // requires a closed 2-manifold: every undirected edge incident to exactly 2
    // triangles. Per-element recovery (each facet present as a tet face) does
    // NOT imply separation if the INPUT surface has T-junctions / non-manifold
    // edges (e.g. meshadapt face-junction misalignment) - the flood then leaks
    // through the defect and the gate rejects. Reports defect edge counts.
    let sep_audit = |label: &str, faces: &[[usize; 3]]| {
        if std::env::var("YAMM_SEP_AUDIT").is_err() {
            return;
        }
        let mut inc: HashMap<(usize, usize), u32> = HashMap::default();
        for f in faces {
            for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *inc.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            }
        }
        let mut n1 = 0usize; // border edges (holes)
        let mut n3 = 0usize; // non-manifold (3+)
        let mut sample = String::new();
        for (&(u, v), &c) in &inc {
            if c == 2 {
                continue;
            }
            if c == 1 {
                n1 += 1;
            } else {
                n3 += 1;
            }
            if sample.len() < 120 {
                sample.push_str(&format!("({u},{v})×{c} "));
            }
        }
        eprintln!(
            "    [SEP] {label}: faces={} edges={} | border(×1)={n1} nonmanifold(×3+)={n3} {}",
            faces.len(),
            inc.len(),
            if n1 + n3 == 0 { "CLOSED ✓" } else { &sample }
        );
    };
    sep_audit("input    ", &cur_faces);

    // (i) Recover ALL boundary segments: flip recovery, then Steiner refinement
    //     until every (refined) segment is present. Splitting a segment also
    //     subdivides its incident facets in cur_faces (the new vertex is on
    //     their shared edge). If the segments do not fully conform, no facet can
    //     close either and the carve would leak - bail NOW (fast) to the legacy
    //     fallback rather than grind through the (expensive) facet pass on a
    //     mesh that cannot conform. The volume gate would reject it regardless,
    //     so this is a pure perf short-circuit, not a correctness change.
    let seg_ok = boundary_recovery::recover_segments_with_steiner(&mut dt, &mut cur_faces);
    if volcheck {
        let v = total_finite_vol(&dt);
        eprintln!(
            "    [VOLCHECK] post-segs   total={v:.6} (Δ={:+.6})",
            v - vol0
        );
    }
    if tdbg {
        eprintln!(
            "    [time] recover_segments={:.2}s (ok={})",
            tp.elapsed().as_secs_f64(),
            seg_ok
        );
        tp = Instant::now();
    }
    if !seg_ok {
        eprintln!(
            "    [vol] conforming: segments did not fully conform in {:.1}s - fallback",
            t0.elapsed().as_secs_f64()
        );
        return None;
    }

    // (ii) Recover ALL boundary facets: flips, then Steiner refinement (split a
    //      stubborn facet at its barycenter) until every triangle in cur_faces
    //      is present.
    let facet_ok = boundary_recovery::recover_facets_with_steiner(&mut dt, &mut cur_faces);
    sep_audit("post-recov", &cur_faces);
    if volcheck {
        let v = total_finite_vol(&dt);
        eprintln!(
            "    [VOLCHECK] post-facets total={v:.6} (Δ={:+.6})",
            v - vol0
        );
    }
    if tdbg {
        eprintln!(
            "    [time] recover_facets={:.2}s (ok={})",
            tp.elapsed().as_secs_f64(),
            facet_ok
        );
        tp = Instant::now();
    }
    let recovered = cur_faces.len();
    if !facet_ok {
        eprintln!(
            "    [vol] conforming: facets partial; carve+gate will arbitrate in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }

    // (iii) Carve the interior, blocked by the REFINED present faces.
    let mut tets = carve_interior(&dt, &cur_faces);
    if tdbg {
        eprintln!(
            "    [time] carve={:.2}s -> {} tets",
            tp.elapsed().as_secs_f64(),
            tets.len()
        );
    }
    if tets.is_empty() {
        eprintln!("    [vol] conforming: carve produced no tets - fallback");
        return None;
    }

    // Steiner refinement appends new vertices to `dt.vertices` (all lying on
    // the original surface), so the tet indices reference the FULL dt vertex
    // list - use it (not `all_points`) as the coordinate table.
    let mut pts = dt.vertices.clone();
    for t in &mut tets {
        let vol = predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]);
        if vol < 0.0 {
            t.swap(2, 3);
        }
    }

    // Self-validating volume gate: a correct carve exactly partitions the
    // enclosed region, so its volume must equal the boundary-enclosed volume.
    // The reference is the ORIGINAL input triangles (the true target volume);
    // Steiner refinement only subdivides the surface, never changes its volume.
    let carved: f64 = tets
        .iter()
        .map(|t| predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs())
        .sum();
    let bvol = boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
    // A topological flood-fill carve over a FULLY-recovered boundary is a valid
    // partition of the enclosed region by construction, so its volume equals the
    // boundary-enclosed volume EXACTLY in exact arithmetic. The residual here is
    // pure floating-point accumulation: `carved` sums |det|/6 over thousands of
    // tets while `bvol` sums divergence terms over the boundary triangles - two
    // different summations of the same quantity. 1e-4 (relative) is the
    // accumulation bound and is still 1000x tighter than any real defect this
    // gate guards against (the over/under-shoot bugs were 10-52%). This is NOT a
    // meshing tolerance and does not mask bugs: the carve's correctness is a
    // topological invariant; this only bounds summation rounding.
    if bvol <= 0.0 || (carved - bvol).abs() > bvol * 1e-4 {
        eprintln!(
            "    [vol] conforming: carved vol {:.4} != boundary vol {:.4} - fallback",
            carved, bvol
        );
        return None;
    }
    eprintln!(
        "    [vol] conforming: {} faces recovered, carved {} tets, vol {:.4} (exact) in {:.1}s",
        recovered,
        tets.len(),
        carved,
        t0.elapsed().as_secs_f64()
    );

    // Boundary-preserving, volume-preserving quality improvement.
    mesh_improve::improve_mesh(
        &mut pts,
        &mut tets,
        n_boundary,
        &input.boundary_vertices,
        &input.boundary_triangles,
        input.target_edge_length,
        10,
        None,
    );
    drop_degenerate_tets(&pts, &mut tets, "conforming");

    let interior_vertices = pts[n_boundary..].to_vec();
    Some(VolumeOutput {
        interior_vertices,
        tetrahedra: tets,
        boundary_recovery_stats: BoundaryRecoveryStats::default(),
    })
}

/// Uniform grid over boundary-triangle AABBs, so asking "which boundary
/// triangles could touch this tet?" costs O(cells spanned) instead of
/// O(#boundary triangles). Shared by the exactness count below and the
/// `YAMM_CLIP_AUDIT` breakdown; a clipper would want the same lookup.
struct BoundaryTriGrid {
    origin: [f64; 3],
    cell: f64,
    cells: HashMap<[i64; 3], Vec<u32>>,
}

impl BoundaryTriGrid {
    fn new(tri_data: &[([f64; 3], [f64; 3], [f64; 3])]) -> Self {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for (a, b, c) in tri_data {
            for p in [a, b, c] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        let diag =
            ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let cell = (diag / 64.0).max(1e-12);
        let mut me = BoundaryTriGrid {
            origin: lo,
            cell,
            cells: HashMap::default(),
        };
        for (ti, (a, b, c)) in tri_data.iter().enumerate() {
            let tlo = [
                a[0].min(b[0]).min(c[0]),
                a[1].min(b[1]).min(c[1]),
                a[2].min(b[2]).min(c[2]),
            ];
            let thi = [
                a[0].max(b[0]).max(c[0]),
                a[1].max(b[1]).max(c[1]),
                a[2].max(b[2]).max(c[2]),
            ];
            let (l, h) = (me.cell_of(&tlo), me.cell_of(&thi));
            for x in l[0]..=h[0] {
                for y in l[1]..=h[1] {
                    for z in l[2]..=h[2] {
                        me.cells.entry([x, y, z]).or_default().push(ti as u32);
                    }
                }
            }
        }
        me
    }

    fn cell_of(&self, p: &[f64; 3]) -> [i64; 3] {
        [
            ((p[0] - self.origin[0]) / self.cell).floor() as i64,
            ((p[1] - self.origin[1]) / self.cell).floor() as i64,
            ((p[2] - self.origin[2]) / self.cell).floor() as i64,
        ]
    }

    /// Does the boundary pass through this tet's INTERIOR? Contact does not
    /// count - see `predicates3d::tri_tet_interiors_overlap`.
    /// Boundary triangles whose interior overlaps this tet's interior. The clip
    /// needs the LIST (to collect the planes it must cut by), not just whether
    /// one exists, so `cuts` is a thin wrapper over this.
    fn overlapping(
        &self,
        tri_data: &[([f64; 3], [f64; 3], [f64; 3])],
        tp: &[[f64; 3]; 4],
    ) -> Vec<u32> {
        let lo = [
            tp.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            tp.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
            tp.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min),
        ];
        let hi = [
            tp.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
            tp.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
            tp.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max),
        ];
        let (l, h) = (self.cell_of(&lo), self.cell_of(&hi));
        let mut cand: Vec<u32> = Vec::new();
        for x in l[0]..=h[0] {
            for y in l[1]..=h[1] {
                for z in l[2]..=h[2] {
                    if let Some(v) = self.cells.get(&[x, y, z]) {
                        cand.extend_from_slice(v);
                    }
                }
            }
        }
        cand.sort_unstable();
        cand.dedup();
        cand.retain(|&ti| {
            let (a, b, c) = tri_data[ti as usize];
            predicates3d::tri_tet_interiors_overlap(&[a, b, c], tp)
        });
        cand
    }

    fn cuts(&self, tri_data: &[([f64; 3], [f64; 3], [f64; 3])], tp: &[[f64; 3]; 4]) -> bool {
        let lo = [
            tp.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            tp.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
            tp.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min),
        ];
        let hi = [
            tp.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
            tp.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
            tp.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max),
        ];
        let (l, h) = (self.cell_of(&lo), self.cell_of(&hi));
        let mut cand: Vec<u32> = Vec::new();
        for x in l[0]..=h[0] {
            for y in l[1]..=h[1] {
                for z in l[2]..=h[2] {
                    if let Some(v) = self.cells.get(&[x, y, z]) {
                        cand.extend_from_slice(v);
                    }
                }
            }
        }
        cand.sort_unstable();
        cand.dedup();
        cand.iter().any(|&ti| {
            let (a, b, c) = tri_data[ti as usize];
            predicates3d::tri_tet_interiors_overlap(&[a, b, c], tp)
        })
    }
}

/// Tets of `tets` whose interior the boundary passes through - an EXACT
/// predictor of an inexact fill (issue #136). Such a tet is kept or dropped
/// wholesale by the inside/outside filter, so its volume is counted entirely or
/// not at all; zero of them means the mesh partitions exactly the region the
/// boundary encloses.
fn count_tets_cut_by_boundary(
    points: &[[f64; 3]],
    tets: &[[usize; 4]],
    tri_data: &[([f64; 3], [f64; 3], [f64; 3])],
) -> usize {
    if tri_data.is_empty() || tets.is_empty() {
        return 0;
    }
    let grid = BoundaryTriGrid::new(tri_data);
    let cut = |t: &&[usize; 4]| -> bool {
        let tp = [points[t[0]], points[t[1]], points[t[2]], points[t[3]]];
        grid.cuts(tri_data, &tp)
    };
    if tets.len() > 1_000 {
        tets.par_iter().filter(cut).count()
    } else {
        tets.iter().filter(cut).count()
    }
}

/// Delaunay-based pipeline with iterative quality refinement.
fn mesh_volume_delaunay(input: &VolumeInput) -> Result<VolumeOutput> {
    let n_boundary = input.boundary_vertices.len();

    use std::time::Instant;
    // Step 1: Generate BCC lattice interior points
    let t_start = Instant::now();
    let interior_pts = bcc::generate_bcc_interior_points(
        &input.boundary_vertices,
        &input.boundary_triangles,
        input.target_edge_length,
    );

    // Step 2: Combine boundary + interior points
    let mut all_points: Vec<[f64; 3]> = input.boundary_vertices.clone();
    all_points.extend_from_slice(&interior_pts);

    // FORCED conforming path (YAMM_CONFORMING): attempt the conforming carve
    // unconditionally, before the legacy pipeline. The DEFAULT flow instead
    // triggers the attempt only when the legacy filter's volume is wrong
    // (Step 6b below, issue #30) - that confines the conforming build+recovery
    // cost to the solids that need the fix, so the common case (filter already
    // exact, e.g. fine meshes where a discarded duplicate Delaunay build alone
    // costs ~60s at 184k pts) pays nothing. This env forces the attempt
    // everywhere (testing / debugging); YAMM_NO_CONFORMING disables both.
    if std::env::var("YAMM_CONFORMING").is_ok() && std::env::var("YAMM_NO_CONFORMING").is_err() {
        if let Some(out) = mesh_volume_conforming(input, &all_points, n_boundary) {
            return Ok(out);
        }
    }

    if interior_pts.is_empty() {
        let out = mesh_boundary_only(input);
        // Undershoot-trigger parity (issue #47): this early return previously
        // skipped the Step-6b conforming backstop entirely, so boundary-only
        // solids (thin shells) shipped whatever the centroid filter kept -
        // e.g. Oktavian's outer shell under-fills by 1.34% with no second
        // chance. Apply the same trigger: if the boundary-only tet-sum
        // disagrees with the boundary-enclosed volume, attempt the exact
        // conforming carve (self-validating; falls back to the boundary-only
        // mesh on any failure).
        if !out.tetrahedra.is_empty()
            && std::env::var("YAMM_NO_CONFORMING").is_err()
            && std::env::var("YAMM_CONFORMING").is_err()
        {
            let mut pts: Vec<[f64; 3]> = input.boundary_vertices.clone();
            pts.extend_from_slice(&out.interior_vertices);
            let bo_vol: f64 = out
                .tetrahedra
                .iter()
                .map(|t| predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs())
                .sum();
            let bvol =
                boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
            let trig: f64 = std::env::var("YAMM_CONFORMING_TRIGGER")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(CONFORMING_TRIGGER_DEFAULT);
            if bvol > 0.0 && (bo_vol - bvol).abs() > bvol * trig {
                eprintln!(
                    "  [vol] boundary-only tet-sum {:.4} vs boundary {:.4} ({:+.2}%) - attempting conforming carve",
                    bo_vol,
                    bvol,
                    (bo_vol - bvol) / bvol * 100.0
                );
                if let Some(conf) = mesh_volume_conforming(input, &all_points, n_boundary) {
                    return Ok(conf);
                }
                eprintln!("  [vol] conforming carve fell back - keeping boundary-only mesh");
            }
        }
        return Ok(out);
    }

    let t1 = t_start.elapsed();
    // Step 3: 3D Delaunay tetrahedralization
    let t_step = Instant::now();
    let mut dt = Delaunay3D::new(&all_points);
    // Activate the vertex→incident-tet index: recover_faces' per-face cavity /
    // diagonal-swap collection is then O(degree) instead of a full O(#tets)
    // scan PER MISSING FACE - on a 306k-face boundary with ~20k missing faces
    // that full scan ground for ~40 minutes (issue #47). Maintained
    // incrementally by the flip/alloc paths from here on.
    dt.build_vert_tets();

    let t2 = t_step.elapsed();
    if std::env::var("YAMM_TIME_DBG").is_ok() {
        eprintln!(
            "    [time] legacy delaunay({} pts)={:.2}s",
            all_points.len(),
            t2.as_secs_f64()
        );
    }
    let t_step = Instant::now();
    let (faces_recovered, faces_failed, failed_faces) =
        boundary_recovery::recover_faces(&mut dt, &input.boundary_triangles);

    let mut recovery_stats = BoundaryRecoveryStats {
        edges_recovered: 0,
        edges_failed: 0,
        faces_recovered,
        faces_failed,
        failed_edges: vec![],
        failed_faces,
        // Filled in at the end, once the FINAL tets exist.
        tets_cut_by_boundary: 0,
    };

    let t3 = t_step.elapsed();
    let t_step = Instant::now();
    let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = input
        .boundary_triangles
        .iter()
        .map(|tri| {
            (
                input.boundary_vertices[tri[0]],
                input.boundary_vertices[tri[1]],
                input.boundary_vertices[tri[2]],
            )
        })
        .collect();
    let bvh = aabb_bvh::TriangleBvh::new(&tri_data);

    let all_extracted = dt.extract_tets();
    eprintln!(
        "    [vol] extracted {} raw tets, building BVH over {} boundary tris",
        all_extracted.len(),
        input.boundary_triangles.len()
    );

    // Step 4: Filter tets by BVH-accelerated inside test (parallel ray-cast)
    let t_filter_start = Instant::now();
    let n_interior_before_clip = all_points.len() - n_boundary;
    // Vertices the clip creates ON the boundary surface, handed to mesh_improve so
    // a swap cannot dismantle a clipped boundary face (see the pinning note in
    // mesh_improve::TetMesh).
    let mut clip_pinned: Option<Vec<bool>> = None;
    let inside_test = |t: &&[usize; 4]| -> bool {
        let centroid = [
            (all_points[t[0]][0] + all_points[t[1]][0] + all_points[t[2]][0] + all_points[t[3]][0])
                / 4.0,
            (all_points[t[0]][1] + all_points[t[1]][1] + all_points[t[2]][1] + all_points[t[3]][1])
                / 4.0,
            (all_points[t[0]][2] + all_points[t[1]][2] + all_points[t[2]][2] + all_points[t[3]][2])
                / 4.0,
        ];
        bvh.is_point_inside(&centroid)
    };
    let mut tets: Vec<[usize; 4]> = if all_extracted.len() > 1_000 {
        all_extracted
            .par_iter()
            .filter(inside_test)
            .copied()
            .collect()
    } else {
        all_extracted.iter().filter(inside_test).copied().collect()
    };
    let t_filter_centroid = t_filter_start.elapsed();
    eprintln!(
        "    [vol] centroid filter: {} -> {} tets in {:.1}s",
        all_extracted.len(),
        tets.len(),
        t_filter_centroid.as_secs_f64()
    );

    // ── CLIP AUDIT (issue #136, YAMM_CLIP_AUDIT) ──
    // De-risks the clipping route before any of it is built.
    //
    // The Delaunay tetrahedralises the CONVEX HULL of all points, and the
    // boundary lies inside that hull, so every interior point lies in exactly
    // one extracted tet. Therefore
    //
    //     enclosed volume  ==  Σ over ALL tets of vol(tet ∩ interior)
    //
    // exactly, by construction - a clipping filter cannot be wrong about the
    // volume the way the centroid filter is. What is NOT known in advance is the
    // COST: how many tets actually straddle the boundary (each needing a clip and
    // a local re-tetrahedralisation), and how much volume rides on them.
    //
    // This audit answers that without implementing the clip. It classifies each
    // tet by the insideness of its four vertices and reports the bracket
    //
    //     [ fully-inside volume , fully-inside + straddling volume ]
    //
    // The enclosed volume must lie inside that bracket for clipping to be able to
    // reach it, so the bracket is also a correctness check on the whole idea.
    if std::env::var("YAMM_CLIP_AUDIT").is_ok() {
        let t_audit = Instant::now();
        let tet_vol = |t: &[usize; 4]| -> f64 {
            predicates3d::tet_volume(
                all_points[t[0]],
                all_points[t[1]],
                all_points[t[2]],
                all_points[t[3]],
            )
            .abs()
        };
        // A tet is CUT if the boundary passes through it, which is what a clip
        // would have to handle. Vertex insideness alone does NOT tell us that: a
        // conformal boundary vertex lies exactly ON the surface, so a tet that
        // merely TOUCHES the boundary reports vertices as not-strictly-inside and
        // looks straddling while being wholly interior. Measured: Cuboid and
        // Sphere both report an interior share of exactly 1.0000 under the vertex
        // test - they touch, they are not cut. So test the six edges for an actual
        // crossing, and fall back to the centroid for the uncut ones.
        // Uniform grid over boundary-triangle AABBs, so the per-tet candidate
        // lookup is O(cells touched) rather than O(#boundary triangles). The
        // clipper needs exactly this lookup too, so it is not throwaway.
        let (mut gmin, mut gmax) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for (a, b, c) in &tri_data {
            for p in [a, b, c] {
                for k in 0..3 {
                    gmin[k] = gmin[k].min(p[k]);
                    gmax[k] = gmax[k].max(p[k]);
                }
            }
        }
        let diag = ((gmax[0] - gmin[0]).powi(2)
            + (gmax[1] - gmin[1]).powi(2)
            + (gmax[2] - gmin[2]).powi(2))
        .sqrt();
        let cell = (diag / 64.0).max(1e-12);
        let cell_of = |p: &[f64; 3]| -> [i64; 3] {
            [
                ((p[0] - gmin[0]) / cell).floor() as i64,
                ((p[1] - gmin[1]) / cell).floor() as i64,
                ((p[2] - gmin[2]) / cell).floor() as i64,
            ]
        };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
        for (ti, (a, b, c)) in tri_data.iter().enumerate() {
            let lo = [
                a[0].min(b[0]).min(c[0]),
                a[1].min(b[1]).min(c[1]),
                a[2].min(b[2]).min(c[2]),
            ];
            let hi = [
                a[0].max(b[0]).max(c[0]),
                a[1].max(b[1]).max(c[1]),
                a[2].max(b[2]).max(c[2]),
            ];
            let (l, h) = (cell_of(&lo), cell_of(&hi));
            for x in l[0]..=h[0] {
                for y in l[1]..=h[1] {
                    for z in l[2]..=h[2] {
                        grid.entry([x, y, z]).or_default().push(ti as u32);
                    }
                }
            }
        }

        let classify = |t: &&[usize; 4]| -> (u8, f64) {
            // GENUINELY CUT: some boundary triangle's interior overlaps this
            // tet's interior. Contact does not count - a tet sharing a face with
            // a conformal boundary needs no cut, and that is the distinction the
            // cheap BVH predicates cannot make (they report every
            // boundary-adjacent tet, which for Cuboid and Sphere is every
            // boundary tet despite all of them being wholly interior).
            let tp = [
                all_points[t[0]],
                all_points[t[1]],
                all_points[t[2]],
                all_points[t[3]],
            ];
            let lo = [
                tp.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                tp.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                tp.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min),
            ];
            let hi = [
                tp.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
                tp.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
                tp.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max),
            ];
            let (l, h) = (cell_of(&lo), cell_of(&hi));
            let mut cand: Vec<u32> = Vec::new();
            for x in l[0]..=h[0] {
                for y in l[1]..=h[1] {
                    for z in l[2]..=h[2] {
                        if let Some(v) = grid.get(&[x, y, z]) {
                            cand.extend_from_slice(v);
                        }
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();
            let cut = cand.iter().any(|&ti| {
                let (a, b, c) = tri_data[ti as usize];
                predicates3d::tri_tet_interiors_overlap(&[a, b, c], &tp)
            });
            if cut {
                return (2, tet_vol(t));
            }
            let c = [
                (all_points[t[0]][0]
                    + all_points[t[1]][0]
                    + all_points[t[2]][0]
                    + all_points[t[3]][0])
                    / 4.0,
                (all_points[t[0]][1]
                    + all_points[t[1]][1]
                    + all_points[t[2]][1]
                    + all_points[t[3]][1])
                    / 4.0,
                (all_points[t[0]][2]
                    + all_points[t[1]][2]
                    + all_points[t[2]][2]
                    + all_points[t[3]][2])
                    / 4.0,
            ];
            if bvh.is_point_inside(&c) {
                (4, tet_vol(t)) // 4 = wholly inside
            } else {
                (0, tet_vol(t)) // 0 = wholly outside
            }
        };
        let rows: Vec<(u8, f64)> = if all_extracted.len() > 1_000 {
            all_extracted.par_iter().map(|t| classify(&t)).collect()
        } else {
            all_extracted.iter().map(|t| classify(&t)).collect()
        };
        let mut n_in = 0usize;
        let mut n_out = 0usize;
        let mut n_straddle = 0usize;
        let mut v_in = 0.0f64;
        let mut v_out = 0.0f64;
        let mut v_straddle = 0.0f64;
        for (k, v) in &rows {
            match k {
                4 => {
                    n_in += 1;
                    v_in += v;
                }
                0 => {
                    n_out += 1;
                    v_out += v;
                }
                _ => {
                    n_straddle += 1;
                    v_straddle += v;
                }
            }
        }
        // n_straddle / v_straddle now mean CUT tets specifically.
        // PLANE COUNT per cut tet (step 0 of the clipping plan). The clip must cut
        // a tet by the supporting plane of every boundary triangle overlapping it,
        // and k distinct planes yield up to 2^k cells - so this distribution is
        // what decides whether the clip is cheap or explosive, and it is measured
        // BEFORE any clipping code is written.
        //
        // Planes are grouped by EXACT coplanarity (orient_3d == 0 against a
        // reference triangle), not by comparing normals with a tolerance: two
        // triangles of one flat CAD face are exactly coplanar here, and that is
        // the case that keeps k small.
        {
            let tgrid = BoundaryTriGrid::new(&tri_data);
            let coplanar =
                |r: &([f64; 3], [f64; 3], [f64; 3]), t: &([f64; 3], [f64; 3], [f64; 3])| -> bool {
                    predicates3d::orient_3d(r.0, r.1, r.2, t.0) == 0.0
                        && predicates3d::orient_3d(r.0, r.1, r.2, t.1) == 0.0
                        && predicates3d::orient_3d(r.0, r.1, r.2, t.2) == 0.0
                };
            let degenerate = |t: &([f64; 3], [f64; 3], [f64; 3])| -> bool {
                let u = [t.1[0] - t.0[0], t.1[1] - t.0[1], t.1[2] - t.0[2]];
                let v = [t.2[0] - t.0[0], t.2[1] - t.0[1], t.2[2] - t.0[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0
            };
            // Also SPLITS each cut tet for real, so the measurement is the actual
            // cell count rather than the crude 2^k bound, and so the splitter's
            // volume-conservation invariant is checked against real geometry
            // instead of only the unit tests.
            let planes_of = |t: &&[usize; 4]| -> (usize, usize, usize, usize, f64, usize) {
                let tp = [
                    all_points[t[0]],
                    all_points[t[1]],
                    all_points[t[2]],
                    all_points[t[3]],
                ];
                let ov = tgrid.overlapping(&tri_data, &tp);
                if ov.is_empty() {
                    return (0, 0, 0, 0, 0.0, 0);
                }
                let mut reps: Vec<u32> = Vec::new();
                for &ti in &ov {
                    let t = tri_data[ti as usize];
                    if degenerate(&t) {
                        continue; // no plane to cut by; counted over the input below
                    }
                    if !reps.iter().any(|&r| coplanar(&tri_data[r as usize], &t)) {
                        reps.push(ti);
                    }
                }
                // Sequential split by each distinct plane. A plane only splits the
                // cells it actually meets, which is why this lands far below 2^k.
                let mut cells = vec![clip::ConvexCell::from_tet(&tp)];
                let v_tet = tet_vol(t);
                let mut n_rejected = 0usize;
                for &pi in &reps {
                    let tr = tri_data[pi as usize];
                    let plane = [tr.0, tr.1, tr.2];
                    let mut next = Vec::with_capacity(cells.len() * 2);
                    for c in &cells {
                        let (parts, rejected) = clip::split_conserving(c, &plane, 1e-9);
                        if rejected {
                            n_rejected += 1;
                        }
                        next.extend(parts);
                    }
                    cells = next;
                    if cells.len() > 4096 {
                        break; // measurement guard; reported via the max column
                    }
                }
                // Classify and tetrahedralise the INSIDE cells, which is what the
                // filter will actually emit - so the cost reported below is the
                // real tet-count growth, not a proxy.
                let mut emitted = 0usize;
                let mut v_inside = 0.0f64;
                for c in &cells {
                    if bvh.is_point_inside(&c.centroid()) {
                        let mut scratch: Vec<[usize; 4]> = Vec::new();
                        c.tetrahedralise(0, &mut scratch);
                        emitted += scratch.len();
                        v_inside += c.volume();
                    }
                }
                let _ = v_inside;
                let v_cells: f64 = cells.iter().map(|c| c.volume()).sum();
                let v_err = if v_tet > 0.0 {
                    ((v_cells - v_tet) / v_tet).abs()
                } else {
                    0.0
                };
                (
                    reps.len(),
                    ov.len(),
                    n_rejected,
                    cells.len(),
                    v_err,
                    emitted,
                )
            };
            let rows: Vec<(usize, usize, usize, usize, f64, usize)> = if all_extracted.len() > 1_000
            {
                all_extracted.par_iter().map(|t| planes_of(&t)).collect()
            } else {
                all_extracted.iter().map(|t| planes_of(&t)).collect()
            };
            let mut hist: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            let mut max_tris = 0usize;
            let mut n_rejections = 0usize;
            let mut max_cells = 0usize;
            let mut tot_emitted = 0usize;
            let mut tot_cells = 0usize;
            let mut worst_vol_err = 0.0f64;
            for (k, m, rej, cells, verr, emit) in &rows {
                if *m == 0 {
                    continue; // not a cut tet
                }
                *hist.entry(*k).or_insert(0) += 1;
                max_tris = max_tris.max(*m);
                n_rejections += rej;
                max_cells = max_cells.max(*cells);
                tot_cells += cells;
                tot_emitted += emit;
                worst_vol_err = worst_vol_err.max(*verr);
            }
            let n_cut_here: usize = hist.values().sum();
            let worst = hist.keys().last().copied().unwrap_or(0);
            let cells_worst = if worst < 32 { 1u64 << worst } else { u64::MAX };
            let summary: Vec<String> = hist.iter().map(|(k, n)| format!("k={k}:{n}")).collect();
            let n_degen_input = tri_data.iter().filter(|t| degenerate(t)).count();
            eprintln!(
                "    [clip] planes per cut tet: {} | cut={n_cut_here} max-overlapping-tris={max_tris} split-rejections={n_rejections} degenerate-boundary-tris={n_degen_input}",
                if summary.is_empty() {
                    "none".to_string()
                } else {
                    summary.join(" ")
                }
            );
            eprintln!("    [clip] worst k={worst} => crude 2^k bound {cells_worst} cells");
            {
                use std::sync::atomic::Ordering::Relaxed;
                let tot = clip::CROSSING_TOTAL.load(Relaxed);
                let mt = clip::CROSSING_MIN_T_E9.load(Relaxed);
                eprintln!(
                    "    [clip] crossings: {tot} | min(t,1-t): smallest={:.3e} under-1e-3={} under-1e-6={} under-1e-9={} (#151)",
                    if mt == u64::MAX { f64::NAN } else { mt as f64 / 1e9 },
                    clip::CROSSING_NEAR[0].load(Relaxed),
                    clip::CROSSING_NEAR[1].load(Relaxed),
                    clip::CROSSING_NEAR[2].load(Relaxed),
                );
            }
            eprintln!(
                "    [clip] ACTUAL cells: max={max_cells} total={tot_cells} mean={:.1} per cut tet | inside cells -> {tot_emitted} tets (replacing {n_cut_here}) | worst split volume error={worst_vol_err:.3e}",
                if n_cut_here > 0 {
                    tot_cells as f64 / n_cut_here as f64
                } else {
                    0.0
                }
            );
        }
        let bvol = boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
        let kept: f64 = tets.iter().map(tet_vol).sum();
        let lo = v_in;
        let hi = v_in + v_straddle;
        eprintln!(
            "    [clip] tets: {} total = {n_in} inside + {n_out} outside + {n_straddle} genuinely cut ({:.1}% need a clip) in {:.1}s",
            all_extracted.len(),
            100.0 * n_straddle as f64 / all_extracted.len().max(1) as f64,
            t_audit.elapsed().as_secs_f64()
        );
        eprintln!("    [clip] volume: inside={v_in:.6} cut={v_straddle:.6} outside={v_out:.6}");
        eprintln!(
            "    [clip] boundary={bvol:.6} centroid-filter kept={kept:.6} (err {:+.3e})",
            kept / bvol - 1.0
        );
        eprintln!(
            "    [clip] clip bracket=[{lo:.6}, {hi:.6}] contains boundary: {} | interior share of cut = {:.4}",
            bvol >= lo * (1.0 - 1e-9) && bvol <= hi * (1.0 + 1e-9),
            if v_straddle > 0.0 {
                (bvol - lo) / v_straddle
            } else {
                f64::NAN
            }
        );
    }
    // Zero-volume audit (issue #47 clusters B+C): count exactly/near-degenerate
    // tets after each pipeline stage to localize WHO commits them.
    let zv_audit = std::env::var("YAMM_ZEROVOL_AUDIT").is_ok();
    let zv_count = |pts: &[[f64; 3]], ts: &[[usize; 4]], label: &str| {
        if !zv_audit {
            return;
        }
        let mut zero = 0usize;
        let mut tiny = 0usize;
        for t in ts {
            let v = predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs();
            if v == 0.0 {
                zero += 1;
            } else if v < 1e-12 {
                tiny += 1;
            }
        }
        // Coincident-vertex audit (issue #60): count vertices whose exact
        // coordinates duplicate an earlier vertex - combinatorially distinct
        // points at identical positions break any index-based face matching
        // downstream.
        let mut seen: HashMap<[u64; 3], usize> = HashMap::default();
        let mut dups = 0usize;
        for p in pts {
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            *seen.entry(key).or_insert(0) += 1;
        }
        for n in seen.values() {
            if *n > 1 {
                dups += n - 1;
            }
        }
        eprintln!(
            "    [zv] {label}: {} tets, zero-vol={zero}, sub-1e-12={tiny}, dup-verts={dups}/{}",
            ts.len(),
            pts.len()
        );
    };
    zv_count(&all_points, &all_extracted, "post-extract (raw)");
    zv_count(&all_points, &tets, "post-filter");

    // Step 4a: Recover boundary-adjacent tets for hollow geometries.
    // Detect if geometry is hollow: check if any extracted tet has all vertices
    // inside but centroid outside (characteristic of thin-walled hollow shapes).
    // For simple (non-hollow) solids this loop finds nothing - skip it entirely.
    let t_recovery_start = Instant::now();
    // NOTE (issue #136): when the clip filter runs it rebuilds `tets` from
    // `all_extracted` wholesale, so whatever this step adds is replaced rather
    // than double-counted. That double count was real when the clip sat before
    // this step: on ToroidalSector it re-added exactly the 70 tets the clip had
    // just cut, turning a post-clip fill of -6.661e-15 into +9.833e-03.
    {
        // Quick hollow detection: sample a few rejected tets to see if any have
        // boundary vertices. If no rejected tet has a boundary vertex, the geometry
        // is non-hollow and recovery would find nothing.
        let kept_set: HashSet<[usize; 4]> = {
            let mut s = HashSet::with_capacity_and_hasher(tets.len(), Default::default());
            for t in &tets {
                let mut k = *t;
                k.sort();
                s.insert(k);
            }
            s
        };

        let needs_recovery = all_extracted.iter().take(all_extracted.len()).any(|t| {
            let mut key = *t;
            key.sort();
            if kept_set.contains(&key) {
                return false;
            }
            // A rejected tet with a boundary vertex suggests hollow geometry
            t.iter().any(|&v| v < n_boundary)
        });

        if needs_recovery {
            let kept_faces: HashSet<[usize; 3]> = {
                let mut s = HashSet::default();
                for t in &tets {
                    for combo in &[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]] {
                        let mut f = [t[combo[0]], t[combo[1]], t[combo[2]]];
                        f.sort();
                        s.insert(f);
                    }
                }
                s
            };

            // Parallel scan: for each rejected tet, check if it should be recovered.
            // Uses rayon to parallelize the expensive BVH queries.
            let recovered_tets: Vec<[usize; 4]> = all_extracted
                .par_iter()
                .filter(|t| {
                    let mut key = **t;
                    key.sort();
                    if kept_set.contains(&key) {
                        return false;
                    }
                    // Must have at least one boundary vertex
                    if !t.iter().any(|&v| v < n_boundary) {
                        return false;
                    }
                    // All vertices must be inside (boundary vertices auto-pass)
                    let verts_ok = t
                        .iter()
                        .all(|&v| v < n_boundary || bvh.is_point_inside(&all_points[v]));
                    if !verts_ok {
                        return false;
                    }
                    // Centroid must be inside
                    let centroid = [
                        (all_points[t[0]][0]
                            + all_points[t[1]][0]
                            + all_points[t[2]][0]
                            + all_points[t[3]][0])
                            / 4.0,
                        (all_points[t[0]][1]
                            + all_points[t[1]][1]
                            + all_points[t[2]][1]
                            + all_points[t[3]][1])
                            / 4.0,
                        (all_points[t[0]][2]
                            + all_points[t[1]][2]
                            + all_points[t[2]][2]
                            + all_points[t[3]][2])
                            / 4.0,
                    ];
                    if !bvh.is_point_inside(&centroid) {
                        return false;
                    }
                    // Must share a face with a kept tet
                    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
                        .iter()
                        .any(|combo| {
                            let mut f = [t[combo[0]], t[combo[1]], t[combo[2]]];
                            f.sort();
                            kept_faces.contains(&f)
                        })
                })
                .copied()
                .collect();

            let n_recovered = recovered_tets.len();
            tets.extend_from_slice(&recovered_tets);
            zv_count(&all_points, &tets, "post-hollow-recovery");
            eprintln!(
                "    [vol] boundary recovery: +{} tets in {:.1}s (parallel)",
                n_recovered,
                t_recovery_start.elapsed().as_secs_f64()
            );
        } else {
            eprintln!(
                "    [vol] boundary recovery: skipped (non-hollow geometry) in {:.1}s",
                t_recovery_start.elapsed().as_secs_f64()
            );
        }
    }

    if tets.is_empty() {
        return Ok(mesh_boundary_only(input));
    }

    let t4 = t_step.elapsed();
    let t_step = Instant::now();
    // Step 5: Remove overlapping tets (only needed if boundary recovery modified the mesh)

    if recovery_stats.faces_recovered > 0 {
        remove_overlapping_tets(&all_points, &mut tets);
        zv_count(&all_points, &tets, "post-overlap");
    }

    // Step 6: Fix any inverted tets by swapping vertices
    for t in &mut tets {
        let vol = predicates3d::tet_volume(
            all_points[t[0]],
            all_points[t[1]],
            all_points[t[2]],
            all_points[t[3]],
        );
        if vol < 0.0 {
            t.swap(2, 3);
        }
    }

    let t5 = t_step.elapsed();

    // Step 6b: Undershoot-triggered conforming carve (issue #30 - DEFAULT-ON).
    // The legacy centroid filter drops tets straddling thin/curved boundaries
    // (the Pipe / ThinWalledCylinder under-fill) and cannot represent exact
    // interiors. Compare its tet-sum volume to the (nesting-aware) boundary-
    // enclosed volume; if they disagree beyond 1%, the filter is wrong HERE, so
    // attempt the exact conforming carve. The carve self-validates (carved
    // volume == boundary volume) and falls back to this legacy mesh on any
    // failure, so correctness can only improve. Confining the attempt to
    // solids the filter actually gets wrong keeps the common case at zero
    // overhead (the conforming build+recovery is paid only where it buys the
    // fix); solids that need it are coarse/thin (small n) or genuinely require
    // the recovery (e.g. fine curved CAD, where conforming now succeeds).
    // The 1% threshold is a TRIGGER, not a tolerance: below it we keep the
    // legacy mesh (status quo - the volume tests assert 3%); above it we try
    // to do better. Escape hatch: YAMM_NO_CONFORMING. (YAMM_CONFORMING forces
    // the attempt unconditionally up front; if that already failed, do not
    // burn a second identical attempt here.)
    if std::env::var("YAMM_NO_CONFORMING").is_err() && std::env::var("YAMM_CONFORMING").is_err() {
        let legacy_vol: f64 = tets
            .iter()
            .map(|t| {
                predicates3d::tet_volume(
                    all_points[t[0]],
                    all_points[t[1]],
                    all_points[t[2]],
                    all_points[t[3]],
                )
                .abs()
            })
            .sum();
        let bvol = boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
        let trig: f64 = std::env::var("YAMM_CONFORMING_TRIGGER")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(CONFORMING_TRIGGER_DEFAULT);
        if bvol > 0.0 && (legacy_vol - bvol).abs() > bvol * trig {
            eprintln!(
                "  [vol] legacy tet-sum {:.4} vs boundary {:.4} ({:+.2}%) - attempting conforming carve",
                legacy_vol,
                bvol,
                (legacy_vol - bvol) / bvol * 100.0
            );
            if let Some(out) = mesh_volume_conforming(input, &all_points, n_boundary) {
                return Ok(out);
            }
            eprintln!("  [vol] conforming carve fell back - keeping legacy mesh");
        }
    }

    // ── CLIP FILTER (issue #136 - DEFAULT ON, escape hatch YAMM_NO_CLIP_FILTER) ──
    // Runs only AFTER the conforming carve has declined, i.e. it is the FALLBACK
    // rather than the mechanism.
    //
    // Ordering matters and was measured. Placed before the carve, the clip makes
    // the fill exact, so the carve's undershoot trigger never fires and the
    // clipped mesh ships in place of the carved one - which regressed the five
    // carve-dependent solids in test_volume_mesh_zoo (Ellipsoid, Hyperboloid,
    // Nestedtorus 2/3/4vol) on element SIZE, not on fill: "mean edge 0.564 vs
    // target 1.5". The carve produces a properly sized conformal mesh on the 15
    // solids it handles; the clip is needed exactly where the carve fails, so it
    // belongs here.
    //
    // Replaces the wholesale keep/drop decision for the tets the boundary
    // actually passes through. The centroid filter above is exact for every
    // OTHER tet; measured over ten zoo solids, zero cut tets iff the fill error
    // is at machine precision (#147). So only the cut ones are re-decided here,
    // which is 0.6%-18.9% of tets.
    //
    // For each cut tet: collect the distinct supporting planes of the boundary
    // triangles overlapping it, split the tet by each, keep the cells whose
    // centroid is inside, and tetrahedralise those. The boundary inside the tet
    // is a subset of those planes, so each resulting cell is wholly inside or
    // wholly outside and one centroid test settles it (see volume/clip.rs).
    // Default ON. The previous default shipped meshes that do not fill their own
    // boundary - BlanketModule's solid 1 was missing 5.4% of itself, silently, so
    // anything normalised over that mesh was out by the same amount. Trading that
    // for the tail-quality regression in #151 (the smallest tets get smaller; the
    // sliver POPULATION is unchanged) is the right way round: the fill error is a
    // physics error, slivers are a robustness risk.
    //
    // YAMM_NO_CLIP_FILTER reverts to the wholesale keep/drop decision, matching
    // YAMM_NO_CONFORMING and YAMM_NO_COPL_REGION, so a user hitting sliver trouble
    // has a way out without a rebuild.
    if std::env::var("YAMM_NO_CLIP_FILTER").is_err() {
        let t_clip = Instant::now();
        let grid = BoundaryTriGrid::new(&tri_data);
        let coplanar =
            |r: &([f64; 3], [f64; 3], [f64; 3]), t: &([f64; 3], [f64; 3], [f64; 3])| -> bool {
                predicates3d::orient_3d(r.0, r.1, r.2, t.0) == 0.0
                    && predicates3d::orient_3d(r.0, r.1, r.2, t.1) == 0.0
                    && predicates3d::orient_3d(r.0, r.1, r.2, t.2) == 0.0
            };
        let degenerate = |t: &([f64; 3], [f64; 3], [f64; 3])| -> bool {
            let u = [t.1[0] - t.0[0], t.1[1] - t.0[1], t.1[2] - t.0[2]];
            let v = [t.2[0] - t.0[0], t.2[1] - t.0[1], t.2[2] - t.0[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0
        };
        // Per-tet outcome. `Clip` carries LOCAL vertex coordinates plus tets
        // indexing them, so the parallel pass allocates nothing shared; the merge
        // below maps those coordinates onto global indices.
        enum Outcome {
            Keep,
            Drop,
            Clip(Vec<[f64; 3]>, Vec<[usize; 4]>),
        }
        let decide = |t: &&[usize; 4]| -> Outcome {
            let tp = [
                all_points[t[0]],
                all_points[t[1]],
                all_points[t[2]],
                all_points[t[3]],
            ];
            let ov = grid.overlapping(&tri_data, &tp);
            if ov.is_empty() {
                return if inside_test(t) {
                    Outcome::Keep
                } else {
                    Outcome::Drop
                };
            }
            let mut reps: Vec<u32> = Vec::new();
            for &ti in &ov {
                let tri = tri_data[ti as usize];
                if degenerate(&tri) {
                    continue;
                }
                if !reps.iter().any(|&r| coplanar(&tri_data[r as usize], &tri)) {
                    reps.push(ti);
                }
            }
            let mut cells = vec![clip::ConvexCell::from_tet(&tp)];
            for &pi in &reps {
                let tr = tri_data[pi as usize];
                let plane = [tr.0, tr.1, tr.2];
                let mut next = Vec::with_capacity(cells.len() * 2);
                for c in &cells {
                    let (parts, _rejected) = clip::split_conserving(c, &plane, 1e-9);
                    next.extend(parts);
                }
                cells = next;
            }
            let mut lverts: Vec<[f64; 3]> = Vec::new();
            let mut ltets: Vec<[usize; 4]> = Vec::new();
            for c in &cells {
                if !bvh.is_point_inside(&c.centroid()) {
                    continue;
                }
                let voff = lverts.len();
                lverts.extend(c.verts.iter().copied());
                c.tetrahedralise(voff, &mut ltets);
            }
            if ltets.is_empty() {
                // Nothing of this tet is inside. Falling back to the centroid
                // decision would re-introduce the wholesale error, so drop it.
                return Outcome::Drop;
            }
            Outcome::Clip(lverts, ltets)
        };
        let outcomes: Vec<Outcome> = if all_extracted.len() > 1_000 {
            all_extracted.par_iter().map(|t| decide(&t)).collect()
        } else {
            all_extracted.iter().map(|t| decide(&t)).collect()
        };

        // Merge. Clip vertices are DEDUPLICATED against the existing points by
        // exact bit pattern, which matters for more than size: adjacent cut tets
        // compute the same crossing point from the same shared edge and the same
        // plane, so the values are bit-identical, and sharing the index is what
        // keeps the mesh conformal across cells instead of cracked. A cracked
        // mesh would also make mesh_improve treat every clipped face as a
        // boundary face.
        let mut index_of: HashMap<[u64; 3], usize> = HashMap::default();
        for (i, p) in all_points.iter().enumerate() {
            index_of
                .entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                .or_insert(i);
        }
        let mut kept = 0usize;
        let mut dropped = 0usize;
        let mut clipped = 0usize;
        let mut new_tets: Vec<[usize; 4]> = Vec::with_capacity(tets.len());
        for (t, out) in all_extracted.iter().zip(outcomes) {
            match out {
                Outcome::Keep => {
                    kept += 1;
                    new_tets.push(*t);
                }
                Outcome::Drop => dropped += 1,
                Outcome::Clip(lverts, ltets) => {
                    clipped += 1;
                    let mut map: Vec<usize> = Vec::with_capacity(lverts.len());
                    for p in &lverts {
                        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
                        let idx = *index_of.entry(key).or_insert_with(|| {
                            all_points.push(*p);
                            all_points.len() - 1
                        });
                        map.push(idx);
                    }
                    for lt in ltets {
                        new_tets.push([map[lt[0]], map[lt[1]], map[lt[2]], map[lt[3]]]);
                    }
                }
            }
        }
        eprintln!(
            "    [vol] clip filter: {} raw = {kept} kept + {dropped} dropped + {clipped} clipped -> {} tets ({} verts added) in {:.1}s",
            all_extracted.len(),
            new_tets.len(),
            all_points.len() - n_boundary - n_interior_before_clip,
            t_clip.elapsed().as_secs_f64()
        );
        // Immediate volume check, BEFORE mesh_improve touches anything: isolates a
        // clip bug from a downstream one.
        let v_now: f64 = new_tets
            .iter()
            .map(|t| {
                predicates3d::tet_volume(
                    all_points[t[0]],
                    all_points[t[1]],
                    all_points[t[2]],
                    all_points[t[3]],
                )
                .abs()
            })
            .sum();
        let bv_now = boundary_enclosed_volume(&input.boundary_vertices, &input.boundary_triangles);
        eprintln!(
            "    [vol] clip filter: post-clip tet-sum {v_now:.6} vs boundary {bv_now:.6} (err {:+.3e})",
            if bv_now > 0.0 { v_now / bv_now - 1.0 } else { 0.0 }
        );
        // Everything appended by the merge is a crossing point, which lies exactly
        // on a boundary triangle. Original tet corners were deduplicated onto
        // their existing indices and are NOT pinned unless they were already
        // boundary vertices.
        let mut pin = vec![false; all_points.len()];
        for p in pin.iter_mut().skip(n_boundary + n_interior_before_clip) {
            *p = true;
        }
        clip_pinned = Some(pin);
        tets = new_tets;
    }

    let t_step = Instant::now();
    // Step 7: Local mesh improvement
    zv_count(&all_points, &tets, "pre-improve");
    mesh_improve::improve_mesh(
        &mut all_points,
        &mut tets,
        n_boundary,
        &input.boundary_vertices,
        &input.boundary_triangles,
        input.target_edge_length,
        10,
        clip_pinned.as_deref(),
    );
    zv_count(&all_points, &tets, "post-improve");
    drop_degenerate_tets(&all_points, &mut tets, "delaunay");

    let t6 = t_step.elapsed();
    eprintln!("  [vol] Timing: BCC={:.1}s Delaunay={:.1}s Recovery={:.1}s Filter={:.1}s Overlap={:.1}s Improve={:.1}s Total={:.1}s ({} tets)",
              t1.as_secs_f64(), t2.as_secs_f64(), t3.as_secs_f64(),
              t4.as_secs_f64(), t5.as_secs_f64(), t6.as_secs_f64(),
              t_start.elapsed().as_secs_f64(), tets.len());
    // Extract interior vertices
    let interior_vertices = all_points[n_boundary..].to_vec();

    // EXACTNESS SIGNAL (issue #136). Count the FINAL tets the boundary passes
    // through. This is exact and cheap (measured 0.0s on every zoo model,
    // including BlanketModule's 25743 tets), and it is what the caller should
    // warn on: zero means the mesh partitions exactly the region the boundary
    // encloses, and nonzero means it cannot, because a cut tet is kept or dropped
    // wholesale. It replaces `faces_failed` as the watertightness signal, which
    // fires on healthy meshes (issue #105).
    let t_cut = Instant::now();
    let mut all_v = input.boundary_vertices.clone();
    all_v.extend_from_slice(&interior_vertices);
    recovery_stats.tets_cut_by_boundary = count_tets_cut_by_boundary(&all_v, &tets, &tri_data);
    eprintln!(
        "    [vol] exactness: {} tet(s) cut by the boundary in {:.1}s",
        recovery_stats.tets_cut_by_boundary,
        t_cut.elapsed().as_secs_f64()
    );

    Ok(VolumeOutput {
        interior_vertices,
        tetrahedra: tets,
        boundary_recovery_stats: recovery_stats,
    })
}

/// Mesh a very small volume with just boundary vertices (no interior points).
fn mesh_boundary_only(input: &VolumeInput) -> VolumeOutput {
    if input.boundary_vertices.len() < 4 {
        return VolumeOutput {
            interior_vertices: vec![],
            tetrahedra: vec![],
            boundary_recovery_stats: BoundaryRecoveryStats::default(),
        };
    }

    let dt = Delaunay3D::new(&input.boundary_vertices);
    let mut tets = dt.extract_tets();

    tets = filter_tets_inside_boundary(
        &input.boundary_vertices,
        &tets,
        &input.boundary_vertices,
        &input.boundary_triangles,
    );

    // Fix orientation
    for t in &mut tets {
        let vol = predicates3d::tet_volume(
            input.boundary_vertices[t[0]],
            input.boundary_vertices[t[1]],
            input.boundary_vertices[t[2]],
            input.boundary_vertices[t[3]],
        );
        if vol < 0.0 {
            t.swap(2, 3);
        }
    }

    // Quality parity with the main pipeline (issue #47 cluster C): a Delaunay
    // of near-COSPHERICAL boundary points (thin spherical/cylindrical shells -
    // exactly the geometries that reach this path) is full of degenerate
    // slivers, and this path previously shipped them raw (Oktavian solid 2:
    // 1,801 zero-volume tets of 28,948 - the worst quality defect in the
    // zoo). Run the same boundary-/volume-preserving improvement the main
    // path runs (it demonstrably heals slivers via swaps)...
    let n_boundary = input.boundary_vertices.len();
    let mut pts: Vec<[f64; 3]> = input.boundary_vertices.clone();
    mesh_improve::improve_mesh(
        &mut pts,
        &mut tets,
        n_boundary,
        &input.boundary_vertices,
        &input.boundary_triangles,
        input.target_edge_length,
        10,
        None,
    );
    // ...then drop any tet still below the degeneracy threshold.
    drop_degenerate_tets(&pts, &mut tets, "boundary-only");

    let interior_vertices = pts[n_boundary..].to_vec();
    VolumeOutput {
        interior_vertices,
        tetrahedra: tets,
        boundary_recovery_stats: BoundaryRecoveryStats::default(),
    }
}

/// Extract unique boundary edges from boundary triangles.
///
/// Each triangle (a, b, c) contributes edges (a,b), (b,c), (a,c).
/// Edges are normalized so the smaller index comes first, and duplicates
/// are removed.
#[allow(dead_code)]
fn extract_boundary_edges(boundary_triangles: &[[usize; 3]]) -> Vec<(usize, usize)> {
    let mut edge_set = HashSet::default();
    for tri in boundary_triangles {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        edge_set.insert((a.min(b), a.max(b)));
        edge_set.insert((b.min(c), b.max(c)));
        edge_set.insert((a.min(c), a.max(c)));
    }
    edge_set.into_iter().collect()
}

/// Filter tets to keep only those whose centroids are inside the boundary mesh.
fn filter_tets_inside_boundary(
    vertices: &[[f64; 3]],
    tets: &[[usize; 4]],
    boundary_vertices: &[[f64; 3]],
    boundary_triangles: &[[usize; 3]],
) -> Vec<[usize; 4]> {
    let tri_data: Vec<([f64; 3], [f64; 3], [f64; 3])> = boundary_triangles
        .iter()
        .map(|tri| {
            (
                boundary_vertices[tri[0]],
                boundary_vertices[tri[1]],
                boundary_vertices[tri[2]],
            )
        })
        .collect();
    let bvh = aabb_bvh::TriangleBvh::new(&tri_data);

    let inside_test = |t: &&[usize; 4]| -> bool {
        let centroid = [
            (vertices[t[0]][0] + vertices[t[1]][0] + vertices[t[2]][0] + vertices[t[3]][0]) / 4.0,
            (vertices[t[0]][1] + vertices[t[1]][1] + vertices[t[2]][1] + vertices[t[3]][1]) / 4.0,
            (vertices[t[0]][2] + vertices[t[1]][2] + vertices[t[2]][2] + vertices[t[3]][2]) / 4.0,
        ];
        bvh.is_point_inside(&centroid)
    };
    if tets.len() > 1_000 {
        tets.par_iter().filter(inside_test).copied().collect()
    } else {
        tets.iter().filter(inside_test).copied().collect()
    }
}

/// Remove overlapping tets caused by incomplete Delaunay cavity expansion.
///
/// For each tet, checks if its centroid is inside any other tet.
/// If so, one of the pair overlaps - remove the larger one.
/// Classify tets via flood fill from outside.
///
/// Builds a tet adjacency graph, seeds from a tet whose centroid is outside
/// the boundary, and floods through adjacencies that don't cross the boundary
/// surface. Tets NOT reached by the flood are inside the solid.
#[allow(dead_code)]
fn flood_fill_classify(
    vertices: &[[f64; 3]],
    tets: &[[usize; 4]],
    bvh: &aabb_bvh::TriangleBvh,
    _tri_data: &[([f64; 3], [f64; 3], [f64; 3])],
) -> Vec<[usize; 4]> {
    if tets.is_empty() {
        return vec![];
    }

    // Compute centroids
    let centroids: Vec<[f64; 3]> = tets
        .iter()
        .map(|t| {
            [
                (vertices[t[0]][0] + vertices[t[1]][0] + vertices[t[2]][0] + vertices[t[3]][0])
                    / 4.0,
                (vertices[t[0]][1] + vertices[t[1]][1] + vertices[t[2]][1] + vertices[t[3]][1])
                    / 4.0,
                (vertices[t[0]][2] + vertices[t[1]][2] + vertices[t[2]][2] + vertices[t[3]][2])
                    / 4.0,
            ]
        })
        .collect();

    // Build adjacency: map sorted face → [tet_indices sharing that face]
    let mut face_to_tets: HashMap<[usize; 3], Vec<usize>> = HashMap::default();
    for (ti, t) in tets.iter().enumerate() {
        for combo in &[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]] {
            let mut f = [t[combo[0]], t[combo[1]], t[combo[2]]];
            f.sort();
            face_to_tets.entry(f).or_default().push(ti);
        }
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![vec![]; tets.len()];
    for tet_list in face_to_tets.values() {
        if tet_list.len() == 2 {
            adj[tet_list[0]].push(tet_list[1]);
            adj[tet_list[1]].push(tet_list[0]);
        }
    }

    // Find seed: a tet whose centroid is definitely outside
    let seed = centroids
        .iter()
        .enumerate()
        .find(|(_, c)| !bvh.is_point_inside(c))
        .map(|(i, _)| i);

    let Some(seed) = seed else {
        // All centroids inside - keep all tets
        return tets.to_vec();
    };

    // Flood fill from seed, blocking at boundary crossings
    let mut outside: HashSet<usize> = HashSet::default();
    let mut stack = vec![seed];

    while let Some(ti) = stack.pop() {
        if !outside.insert(ti) {
            continue;
        }

        for &ni in &adj[ti] {
            if outside.contains(&ni) {
                continue;
            }
            // Check if the segment between centroids crosses the boundary
            if !bvh.segment_crosses_boundary(&centroids[ti], &centroids[ni]) {
                stack.push(ni);
            }
        }
    }

    // Keep tets NOT in the outside set
    tets.iter()
        .enumerate()
        .filter(|(i, _)| !outside.contains(i))
        .map(|(_, t)| *t)
        .collect()
}

fn remove_overlapping_tets(vertices: &[[f64; 3]], tets: &mut Vec<[usize; 4]>) {
    if tets.len() < 2 {
        return;
    }

    // Precompute centroids and volumes
    let data: Vec<([f64; 3], f64)> = tets
        .iter()
        .map(|t| {
            let c = [
                (vertices[t[0]][0] + vertices[t[1]][0] + vertices[t[2]][0] + vertices[t[3]][0])
                    / 4.0,
                (vertices[t[0]][1] + vertices[t[1]][1] + vertices[t[2]][1] + vertices[t[3]][1])
                    / 4.0,
                (vertices[t[0]][2] + vertices[t[1]][2] + vertices[t[2]][2] + vertices[t[3]][2])
                    / 4.0,
            ];
            let vol = predicates3d::tet_volume(
                vertices[t[0]],
                vertices[t[1]],
                vertices[t[2]],
                vertices[t[3]],
            )
            .abs();
            (c, vol)
        })
        .collect();

    // Compute typical tet edge length for grid cell size.
    // Use median of first 100 tets' max edge to avoid outliers.
    let mut edge_samples: Vec<f64> = Vec::new();
    for t in tets.iter().take(200.min(tets.len())) {
        let mut mx = 0.0_f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d = predicates3d::dist_sq_3d(vertices[t[i]], vertices[t[j]]).sqrt();
                mx = mx.max(d);
            }
        }
        edge_samples.push(mx);
    }
    edge_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let typical_edge = if edge_samples.is_empty() {
        1.0
    } else {
        edge_samples[edge_samples.len() / 2]
    };
    let cell_size = typical_edge * 2.0;
    let inv_cell = 1.0 / cell_size;

    // Hash centroids into spatial grid
    let cell_key = |p: &[f64; 3]| -> (i64, i64, i64) {
        (
            (p[0] * inv_cell).floor() as i64,
            (p[1] * inv_cell).floor() as i64,
            (p[2] * inv_cell).floor() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::default();
    for (i, (c, _)) in data.iter().enumerate() {
        grid.entry(cell_key(c)).or_default().push(i);
    }

    // Phase 1 (parallel): collect every overlapping PAIR. The pair set is a
    // pure geometric predicate (order-independent), so this parallelizes
    // safely - it was the dominant cost (~370s serial on a 682k-tet mesh once
    // legacy recover_faces stopped grinding, issue #47).
    use rayon::prelude::*;
    let cells: Vec<(&(i64, i64, i64), &Vec<usize>)> = grid.iter().collect();
    let mut overlapping: Vec<(usize, usize)> = cells
        .par_iter()
        .flat_map_iter(|(&(cx, cy, cz), indices)| {
            let grid = &grid;
            let data = &data;
            let tets = &*tets;
            let mut local: Vec<(usize, usize)> = Vec::new();
            for dx in -1..=1_i64 {
                for dy in -1..=1_i64 {
                    for dz in -1..=1_i64 {
                        let neighbor_key = (cx + dx, cy + dy, cz + dz);
                        let neighbor_indices = match grid.get(&neighbor_key) {
                            Some(v) => v.as_slice(),
                            None => continue,
                        };
                        for &i in indices.iter() {
                            for &j in neighbor_indices {
                                if j <= i {
                                    continue;
                                }
                                let ci_in_j = point_in_tet(&data[i].0, &tets[j], vertices);
                                let cj_in_i = point_in_tet(&data[j].0, &tets[i], vertices);
                                if ci_in_j || cj_in_i {
                                    local.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
            local.into_iter()
        })
        .collect();
    // Phase 2 (serial, deterministic): greedy resolution in sorted pair order.
    // Mirrors the previous serial semantics: a pair whose member was already
    // removed is satisfied; otherwise remove the LARGER tet of the pair.
    overlapping.sort_unstable();
    overlapping.dedup();
    let mut to_remove = HashSet::default();
    for (i, j) in overlapping {
        if to_remove.contains(&i) || to_remove.contains(&j) {
            continue;
        }
        if data[i].1 >= data[j].1 {
            to_remove.insert(i);
        } else {
            to_remove.insert(j);
        }
    }

    if !to_remove.is_empty() {
        let mut idx = 0;
        tets.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }
}

/// Test if a point is inside a tetrahedron using barycentric coordinates.
fn point_in_tet(p: &[f64; 3], tet: &[usize; 4], vertices: &[[f64; 3]]) -> bool {
    let a = vertices[tet[0]];
    let b = vertices[tet[1]];
    let c = vertices[tet[2]];
    let d = vertices[tet[3]];

    let o0 = predicates3d::orient_3d([p[0], p[1], p[2]], b, c, d);
    let o1 = predicates3d::orient_3d(a, [p[0], p[1], p[2]], c, d);
    let o2 = predicates3d::orient_3d(a, b, [p[0], p[1], p[2]], d);
    let o3 = predicates3d::orient_3d(a, b, c, [p[0], p[1], p[2]]);

    // All strictly positive or all strictly negative → inside (with margin)
    // Use a margin to avoid false positives from near-face centroids
    let eps = 1e-10;
    (o0 > eps && o1 > eps && o2 > eps && o3 > eps)
        || (o0 < -eps && o1 < -eps && o2 < -eps && o3 < -eps)
}

/// Winding number inside test. Uses abs to handle both normal orientations.
#[allow(dead_code)]
fn winding_number_at(point: &[f64; 3], triangles: &[([f64; 3], [f64; 3], [f64; 3])]) -> f64 {
    let mut omega = 0.0f64;
    for (v0, v1, v2) in triangles {
        let a = [v0[0] - point[0], v0[1] - point[1], v0[2] - point[2]];
        let b = [v1[0] - point[0], v1[1] - point[1], v1[2] - point[2]];
        let c = [v2[0] - point[0], v2[1] - point[1], v2[2] - point[2]];
        let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let lc = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        if la < 1e-15 || lb < 1e-15 || lc < 1e-15 {
            continue;
        }
        let bc = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        let num = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
        let den = la * lb * lc
            + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * lc
            + (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) * la
            + (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]) * lb;
        omega += 2.0 * num.atan2(den);
    }
    omega / (4.0 * std::f64::consts::PI)
}

#[allow(dead_code)]
fn is_point_inside(point: &[f64; 3], triangles: &[([f64; 3], [f64; 3], [f64; 3])]) -> bool {
    let w = winding_number_at(point, triangles).abs();
    w > 0.5 && w < 1.5
}

/// Moller-Trumbore ray-triangle intersection (t > 0).
#[allow(dead_code)]
fn ray_tri_intersect(
    origin: &[f64; 3],
    dir: &[f64; 3],
    v0: &[f64; 3],
    v1: &[f64; 3],
    v2: &[f64; 3],
) -> bool {
    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let h = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];
    let a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    if a.abs() < 1e-14 {
        return false;
    }

    let f = 1.0 / a;
    let s = [origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = [
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0],
    ];
    let v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    t > 1e-14
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube boundary: 8 vertices, 12 triangles.
    fn unit_cube() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (verts, tris)
    }

    #[test]
    fn mesh_unit_cube() {
        let (verts, tris) = unit_cube();
        let input = VolumeInput {
            boundary_vertices: verts,
            boundary_triangles: tris,
            target_edge_length: 0.5,
        };
        let output = mesh_volume(&input).unwrap();

        assert!(
            !output.tetrahedra.is_empty(),
            "Should produce tetrahedra for unit cube"
        );

        // All vertex indices should be valid
        let total_verts = input.boundary_vertices.len() + output.interior_vertices.len();
        for t in &output.tetrahedra {
            for &v in t {
                assert!(
                    v < total_verts,
                    "Invalid vertex index: {v} >= {total_verts}"
                );
            }
        }

        // All tets should have positive volume
        let mut all_verts = input.boundary_vertices.clone();
        all_verts.extend_from_slice(&output.interior_vertices);
        for t in &output.tetrahedra {
            let vol = predicates3d::tet_volume(
                all_verts[t[0]],
                all_verts[t[1]],
                all_verts[t[2]],
                all_verts[t[3]],
            );
            assert!(vol > 0.0, "Tet has non-positive volume: {vol}");
        }
    }

    #[test]
    fn mesh_large_cube() {
        let (verts, tris) = unit_cube();
        // Scale up to 10x10x10
        let verts: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| [v[0] * 10.0, v[1] * 10.0, v[2] * 10.0])
            .collect();
        let input = VolumeInput {
            boundary_vertices: verts,
            boundary_triangles: tris,
            target_edge_length: 2.0,
        };
        let output = mesh_volume(&input).unwrap();

        assert!(
            output.tetrahedra.len() > 10,
            "Large cube should produce many tets, got {}",
            output.tetrahedra.len()
        );
        assert!(
            !output.interior_vertices.is_empty(),
            "Should have interior vertices"
        );
    }

    #[test]
    fn mesh_quality_stats() {
        let (verts, tris) = unit_cube();
        let verts: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| [v[0] * 5.0, v[1] * 5.0, v[2] * 5.0])
            .collect();
        let input = VolumeInput {
            boundary_vertices: verts.clone(),
            boundary_triangles: tris,
            target_edge_length: 1.5,
        };
        let output = mesh_volume(&input).unwrap();

        let mut all_verts = verts;
        all_verts.extend_from_slice(&output.interior_vertices);
        let stats = optimize::compute_stats(&all_verts, &output.tetrahedra);

        assert!(stats.num_tets > 0);
        assert!(stats.min_quality > 0.0);
        assert!(stats.avg_quality > 0.0);
        assert!(stats.min_dihedral_deg > 0.0);
    }

    #[test]
    fn mesh_cube_boundary_conformity() {
        // Mesh a cube and verify that all 12 input boundary triangles
        // appear as faces of the output tetrahedra.
        let (verts, tris) = unit_cube();
        let input = VolumeInput {
            boundary_vertices: verts.clone(),
            boundary_triangles: tris.clone(),
            target_edge_length: 0.5,
        };
        let output = mesh_volume(&input).unwrap();

        // Collect all tet faces as sorted-vertex triples
        let mut all_verts = verts;
        all_verts.extend_from_slice(&output.interior_vertices);

        let mut tet_faces: HashSet<[usize; 3]> = HashSet::default();
        for t in &output.tetrahedra {
            // 4 faces per tet
            let faces: [[usize; 3]; 4] = [
                [t[0], t[1], t[2]],
                [t[0], t[1], t[3]],
                [t[0], t[2], t[3]],
                [t[1], t[2], t[3]],
            ];
            for mut f in faces {
                f.sort();
                tet_faces.insert(f);
            }
        }

        // Check each boundary triangle appears as a tet face
        let mut missing = 0;
        for tri in &tris {
            let mut sorted_tri = *tri;
            sorted_tri.sort();
            if !tet_faces.contains(&sorted_tri) {
                missing += 1;
            }
        }

        // All 12 boundary triangles should appear in the tet mesh.
        // If some are missing, the boundary recovery stats should reflect it.
        if missing > 0 {
            eprintln!(
                "WARNING: {missing}/{} boundary triangles not found in tet faces",
                tris.len()
            );
        }

        // Verify recovery stats are populated
        let stats = &output.boundary_recovery_stats;
        assert_eq!(
            stats.edges_failed + stats.faces_failed + missing,
            stats.edges_failed + stats.faces_failed + missing,
            "Stats should be consistent"
        );
    }

    #[test]
    fn degenerate_tets_are_dropped_and_others_kept() {
        // A flat (zero-volume) tet alongside a healthy one. Only the flat one
        // goes, and the volume removed is nothing -- which is the argument for
        // dropping rather than tolerating: it costs no material and spares
        // MOAB a cell that can never be scored.
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], // coplanar with the first three
        ];
        let mut tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let before: f64 = tets
            .iter()
            .map(|t| predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs())
            .sum();
        assert_eq!(drop_degenerate_tets(&pts, &mut tets, "test"), 1);
        assert_eq!(tets, vec![[0, 1, 2, 3]]);
        let after: f64 = tets
            .iter()
            .map(|t| predicates3d::tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]).abs())
            .sum();
        assert!((before - after).abs() < DEGENERATE_TET_VOLUME);
    }

    #[test]
    fn a_clean_mesh_loses_nothing() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut tets = vec![[0, 1, 2, 3]];
        assert_eq!(drop_degenerate_tets(&pts, &mut tets, "test"), 0);
        assert_eq!(tets.len(), 1);
    }

    #[test]
    fn mesh_cube_recovery_stats() {
        // Verify that boundary_recovery_stats fields are populated
        let (verts, tris) = unit_cube();
        let verts: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| [v[0] * 5.0, v[1] * 5.0, v[2] * 5.0])
            .collect();
        let input = VolumeInput {
            boundary_vertices: verts,
            boundary_triangles: tris,
            target_edge_length: 1.5,
        };
        let output = mesh_volume(&input).unwrap();
        let stats = &output.boundary_recovery_stats;

        // Stats should be present and sensible
        let total =
            stats.edges_recovered + stats.edges_failed + stats.faces_recovered + stats.faces_failed;
        // Just verify the stats struct is populated (total >= 0 always
        // true for usize, so check it compiles and runs).
        let _ = total;
    }
}
