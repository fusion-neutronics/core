//! Coplanar-region 2-D constrained re-triangulation (issue #31 endgame).
//!
//! THE PROBLEM. A flat boundary face (LShaped's reflex caps, cylinder end
//! caps) puts many mesh vertices EXACTLY in one plane. The base Delaunay's
//! exact-zero orientation/insphere ties there produce "pillow" tets:
//! zero-volume tets bridging two different triangulations of the same planar
//! region (the sheet seen from above vs from below). A boundary segment whose
//! diagonal lost those ties is then missing from BOTH sheets, and every 3-D
//! repair tool dead-ends on the exact-zero predicates: edge splits make
//! degenerate children, flips reject zero-volume configurations (#42 guards),
//! cavity re-triangulation rebuilds the same ties in its local DT, and SoS
//! merely renames the flat tets without removing them (see the #31 design
//! notes).
//!
//! THE FIX (this module). Stop fighting the flats in 3-D. The in-plane
//! structure is purely 2-D, and in 2-D constrained edge recovery by flips
//! ALWAYS succeeds (no flat triangles can exist; `cdt2d`):
//!
//! 1. Extract the two SHEETS of the maximal coplanar region around the
//!    missing segment: `S_top` = in-plane faces whose up-side tet is non-flat
//!    (finite-above or hull), `S_bot` symmetric. The flat tets between them -
//!    the sandwich - are deleted wholesale; their internal structure is
//!    irrelevant. Side tets are NEVER touched (the historical Pipe/TWC
//!    regressions all came from rebuilding healthy neighbours).
//! 2. In exact 2-D (drop-axis projection preserves the f64 coordinates;
//!    orientation via Shewchuk's `orient2d`), copy `S_bot` and recover every
//!    surface segment of the region by logged flips → sheet `T_c` containing
//!    every constraint. Then recover every edge of `S_top` → the log ends
//!    EXACTLY at `S_top`.
//! 3. Replay the flip log as a stack of zero-volume tets: one 2-D flip IS one
//!    flat tet (old diagonal pair = bottom faces, new pair = top faces) - the
//!    classic triangulation-flip ↔ stacked-tet correspondence. The stack
//!    glues to the untouched below-tets via `S_bot` and to the untouched
//!    above-tets (or hull tets) via `S_top`; every constraint exists as an
//!    edge of the `T_c` layer, and every surface facet of the plane exists as
//!    a face between consecutive layers.
//!
//! The operation is EXACTLY volume-preserving (every removed and added tet
//! has zero volume), fully transactional (all 2-D work and a combinatorial
//! wiring simulation happen before any 3-D mutation), and deterministic
//! (sorted iteration everywhere). Anchoring the constraint recovery at
//! `S_bot` (the interior side for a hull-supporting cap) keeps the
//! carved-inside part of the stack minimal.

use super::dethash::{HashMap, HashSet};
use std::collections::VecDeque;

use super::cdt2d::{recover_edge_logged, FlipLog, Mesh};
use super::delaunay3d::{opposite_face, Delaunay3D, Tet, INFINITE};
use super::predicates3d::orient_3d;

/// Sorted-triple key for a face.
#[inline]
fn fkey(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut s = [a, b, c];
    s.sort_unstable();
    s
}

/// Which kind of tet sits on one side of an in-plane face.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SideKind {
    /// Finite tet with its apex strictly on the positive side of the plane.
    Up,
    /// Finite tet with its apex strictly on the negative side.
    Down,
    /// Zero-volume tet entirely in the plane (sandwich member).
    Flat,
    /// Hull tet (INFINITE apex) - the plane is hull-supporting there.
    Hull,
}

/// Hard cap on the size of a coplanar region (faces); a region larger than
/// this is not the flat-cap class and the analysis bails.
///
/// This was 50_000, which is LARGER than the entire boundary of a fine model
/// (a target-sized BlanketModule boundary is 48140 faces), so it never bit: the
/// region BFS walked the whole connected coplanar complex, and a
/// planar-dominated solid is one enormous complex. Combined with the fact that
/// this path is entered once per segment, that was the dominant cost of the
/// conforming carve on such models.
///
/// CALIBRATED against `copl_region_max` (reported on the `[seg DBG] budget:`
/// line) over every zoo solid whose carve succeeds - see the commit message for
/// the table. The cap only needs to admit the flat-cap class this module exists
/// for; anything bigger is by definition not that class.
/// Overridable with `YAMM_COPL_REGION_CAP` so the calibration above can be
/// re-run (measure with a deliberately huge cap, read the high-water, set the
/// default) without a rebuild. Read once - this is on a per-segment path.
fn region_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("YAMM_COPL_REGION_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096)
    })
}

/// High-water mark of coplanar-region size (faces) actually reached, over all
/// attempts. Diagnostic only - it is what the cap is calibrated against.
pub(super) static REGION_HIGH_WATER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Outcome of a coplanar-region recovery attempt. Anything but `Recovered`
/// is a strict no-op on the 3-D mesh.
pub(super) enum CoplanarOutcome {
    /// The region was rebuilt and (a, b) now exists as a mesh edge.
    Recovered,
    /// A region vertex `at` lies EXACTLY on the open surface segment `seg`
    /// (2-D collinear and strictly between its endpoints), so `seg` can never
    /// be an edge of any triangulation. The caller must split `seg` at `at`
    /// (the in-plane analogue of the AcrossVert / vertex-on-edge resolutions)
    /// and re-attempt. Typical source: a BCC interior point that landed
    /// exactly on the flat cap plane AND on the segment's line - invisible to
    /// `finddirection` because the in-plane march exits the hull.
    SplitSegment { seg: (usize, usize), at: usize },
    /// Analysis or 2-D recovery failed; fall through to the Steiner path.
    Failed,
}

/// Try to recover missing segment (a, b) by re-triangulating the maximal
/// coplanar region it lies in (module docs above). On STRUCTURAL analysis
/// failures (the region is not the clean two-sheets-plus-sandwich shape) the
/// region's vertices are added to `failed_verts` so later segments of the
/// same broken region skip the analysis cost; constraint-level failures do
/// NOT memo (other segments of the same region may still succeed).
pub(super) fn recover_segment_coplanar_region(
    tets: &mut Delaunay3D,
    a: usize,
    b: usize,
    inc_faces: &HashMap<(usize, usize), Vec<[usize; 3]>>,
    psegs: &HashSet<(usize, usize)>,
    failed_verts: &mut HashSet<usize>,
) -> CoplanarOutcome {
    use CoplanarOutcome::*;
    if failed_verts.contains(&a) || failed_verts.contains(&b) {
        return Failed;
    }
    let dbg = std::env::var("YAMM_COPL_DBG").is_ok();

    // ── 1. The plane: from the surface faces incident to (a, b). Fire only
    // when ALL incident surface faces are exactly coplanar - a segment in the
    // interior of a flat surface region. Crease segments (cap rim, curved
    // walls) bail here, which is what keeps Pipe/TWC off this path.
    // O(degree) via the index rather than an O(#faces) scan of the whole surface.
    let empty: Vec<[usize; 3]> = Vec::new();
    let inc: Vec<&[usize; 3]> = inc_faces
        .get(&(a.min(b), a.max(b)))
        .unwrap_or(&empty)
        .iter()
        .collect();
    let Some(f0) = inc.first() else {
        return Failed;
    };
    let (p0, p1, p2) = (
        tets.vertices[f0[0]],
        tets.vertices[f0[1]],
        tets.vertices[f0[2]],
    );
    let on_p =
        |v: usize| -> bool { v != INFINITE && orient_3d(p0, p1, p2, tets.vertices[v]) == 0.0 };
    if inc.iter().any(|f| f.iter().any(|&v| !on_p(v))) {
        return Failed; // crease / curved - not the coplanar class
    }

    // ── 2. Region growth: BFS over in-plane edges. Around each in-plane edge
    // walk the full incident-tet ring; every face with 3 on-plane vertices is
    // a region face (recording its owning (tet, face) slots), every tet with 4
    // on-plane vertices is a sandwich flat tet, and each new region face
    // queues its edges. This reaches the whole connected planar complex -
    // lateral sheet connectivity goes through the edge rings, which a
    // face-adjacency BFS would miss.
    let mut faces: HashMap<[usize; 3], Vec<(usize, usize)>> = HashMap::default();
    let mut flat: HashSet<usize> = HashSet::default();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::default();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let register = |ti: usize,
                    tets: &Delaunay3D,
                    faces: &mut HashMap<[usize; 3], Vec<(usize, usize)>>,
                    flat: &mut HashSet<usize>,
                    queue: &mut VecDeque<(usize, usize)>,
                    seen_edges: &mut HashSet<(usize, usize)>| {
        let v = tets.tets[ti].verts;
        if v.iter().all(|&x| on_p(x)) {
            flat.insert(ti);
        }
        for fi in 0..4 {
            let f = opposite_face(v, fi);
            if !f.iter().all(|&x| on_p(x)) {
                continue;
            }
            let key = fkey(f[0], f[1], f[2]);
            let owners = faces.entry(key).or_default();
            if !owners.contains(&(ti, fi)) {
                owners.push((ti, fi));
            }
            for &(u, w) in &[(key[0], key[1]), (key[1], key[2]), (key[0], key[2])] {
                if seen_edges.insert((u, w)) {
                    queue.push_back((u, w));
                }
            }
        }
    };
    // Seed from the stars of a and b.
    if !tets.index_active() {
        return Failed;
    }
    for &s in &[a, b] {
        for &ti in tets.incident_tets(s) {
            let ti = ti as usize;
            if tets.is_live(ti) {
                register(ti, tets, &mut faces, &mut flat, &mut queue, &mut seen_edges);
            }
        }
    }
    while let Some((u, w)) = queue.pop_front() {
        REGION_HIGH_WATER.fetch_max(faces.len(), std::sync::atomic::Ordering::Relaxed);
        if faces.len() > region_cap() {
            if dbg {
                eprintln!("    [copl DBG] ({a},{b}): region exceeds cap - bail");
            }
            failed_verts.extend(faces.keys().flat_map(|k| k.iter().copied()));
            return Failed;
        }
        // All live tets around edge (u, w), hull tets included.
        let ring: Vec<usize> = tets
            .incident_tets(u)
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| tets.is_live(i) && tets.tets[i].verts.contains(&w))
            .collect();
        for ti in ring {
            register(ti, tets, &mut faces, &mut flat, &mut queue, &mut seen_edges);
        }
    }

    // The memo for analysis failures from here on: the region is known.
    let mut vset: Vec<usize> = faces.keys().flat_map(|k| k.iter().copied()).collect();
    vset.sort_unstable();
    vset.dedup();
    macro_rules! bail {
        ($why:expr) => {{
            if dbg {
                eprintln!(
                    "    [copl DBG] ({a},{b}): bail - {} (region: {} faces, {} flat, {} verts)",
                    $why,
                    faces.len(),
                    flat.len(),
                    vset.len()
                );
            }
            failed_verts.extend(vset.iter().copied());
            return Failed;
        }};
    }
    if !vset.contains(&a) || !vset.contains(&b) {
        bail!("segment endpoints not in region");
    }

    // ── 3. Sheets: classify each region face's two sides; the up-side
    // non-flat owner puts it in S_top, the down-side non-flat owner in S_bot.
    // Hull tets are the empty side of a hull-supporting plane and must sit on
    // ONE consistent side across the region.
    let side_of = |ti: usize, fi: usize| -> SideKind {
        let apex = tets.tets[ti].verts[fi];
        if apex == INFINITE {
            SideKind::Hull
        } else {
            let o = orient_3d(p0, p1, p2, tets.vertices[apex]);
            if o > 0.0 {
                SideKind::Up
            } else if o < 0.0 {
                SideKind::Down
            } else {
                SideKind::Flat
            }
        }
    };
    // A face KEY can carry more than two owners: zero-volume stack tets (a
    // previous commit of this very machinery, or native pillow stacks) repeat
    // the same vertex triple at several stack heights. Only the NON-FLAT
    // owners matter for the sheets - everything flat between them is sandwich
    // and gets deleted. Per key: 2 non-flat owners (opposite sides) → in both
    // sheets; 1 → that side's sheet (the other side is sandwich-internal);
    // 0 → fully internal.
    let mut s_top: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
    let mut s_bot: HashMap<[usize; 3], (usize, usize)> = HashMap::default();
    let mut hull_up: Option<bool> = None;
    // First pass: fix the hull side from any face with a finite non-flat
    // owner opposite a hull owner, and validate global consistency below.
    for owners in faces.values() {
        let kinds: Vec<SideKind> = owners.iter().map(|&(ti, fi)| side_of(ti, fi)).collect();
        if kinds.contains(&SideKind::Hull) {
            if kinds.contains(&SideKind::Down) {
                if hull_up == Some(false) {
                    bail!("hull on both sides of the plane");
                }
                hull_up = Some(true);
            }
            if kinds.contains(&SideKind::Up) {
                if hull_up == Some(true) {
                    bail!("hull on both sides of the plane");
                }
                hull_up = Some(false);
            }
        }
    }
    // Global consistency: a hull-supporting plane has NO finite tet on the
    // hull's side anywhere in the region.
    let any_kind = |k: SideKind| {
        faces
            .values()
            .flatten()
            .any(|&(ti, fi)| side_of(ti, fi) == k)
    };
    if hull_up == Some(true) && any_kind(SideKind::Up) {
        bail!("finite tet above a hull-supporting plane");
    }
    if hull_up == Some(false) && any_kind(SideKind::Down) {
        bail!("finite tet below a hull-supporting plane");
    }
    if hull_up.is_none() && any_kind(SideKind::Hull) {
        bail!("hull side undetermined");
    }
    for (key, owners) in &faces {
        let nf: Vec<(usize, usize)> = owners
            .iter()
            .copied()
            .filter(|&(ti, fi)| side_of(ti, fi) != SideKind::Flat)
            .collect();
        match nf.len() {
            0 => {} // sandwich-internal
            1 => {
                let up_side = match side_of(nf[0].0, nf[0].1) {
                    SideKind::Up => true,
                    SideKind::Down => false,
                    SideKind::Hull => hull_up.unwrap(), // validated above
                    SideKind::Flat => unreachable!(),
                };
                if up_side {
                    s_top.insert(*key, nf[0]);
                } else {
                    s_bot.insert(*key, nf[0]);
                }
            }
            2 => {
                let (k1, k2) = (side_of(nf[0].0, nf[0].1), side_of(nf[1].0, nf[1].1));
                let up1 = match k1 {
                    SideKind::Up => true,
                    SideKind::Down => false,
                    SideKind::Hull => hull_up.unwrap(),
                    SideKind::Flat => unreachable!(),
                };
                let up2 = match k2 {
                    SideKind::Up => true,
                    SideKind::Down => false,
                    SideKind::Hull => hull_up.unwrap(),
                    SideKind::Flat => unreachable!(),
                };
                if up1 == up2 {
                    bail!("same-side non-flat owners");
                }
                let (up, down) = if up1 { (nf[0], nf[1]) } else { (nf[1], nf[0]) };
                s_top.insert(*key, up);
                s_bot.insert(*key, down);
            }
            _ => bail!("more than two non-flat owners"),
        }
    }
    // Both sheets must triangulate the same region: same vertex set and same
    // rim (edges with exactly one incident sheet triangle). Interior coverage
    // is then forced; the 2-D loader validates manifoldness.
    let sheet_check =
        |sheet: &HashMap<[usize; 3], (usize, usize)>| -> (Vec<usize>, Vec<(usize, usize)>) {
            let mut vs: Vec<usize> = sheet.keys().flat_map(|k| k.iter().copied()).collect();
            vs.sort_unstable();
            vs.dedup();
            let mut ecount: HashMap<(usize, usize), usize> = HashMap::default();
            for k in sheet.keys() {
                for &(u, w) in &[(k[0], k[1]), (k[1], k[2]), (k[0], k[2])] {
                    *ecount.entry((u, w)).or_insert(0) += 1;
                }
            }
            let mut rim: Vec<(usize, usize)> = ecount
                .iter()
                .filter(|&(_, &c)| c == 1)
                .map(|(&e, _)| e)
                .collect();
            if ecount.values().any(|&c| c > 2) {
                rim.push((usize::MAX, usize::MAX)); // poison → mismatch below
            }
            rim.sort_unstable();
            (vs, rim)
        };
    let (top_vs, top_rim) = sheet_check(&s_top);
    let (bot_vs, bot_rim) = sheet_check(&s_bot);
    if top_vs != bot_vs || top_rim != bot_rim || s_top.len() != s_bot.len() {
        bail!("sheets do not triangulate the same region");
    }
    if top_vs != vset {
        bail!("sheet vertices != region vertices");
    }

    // ── 4. Exact 2-D projection: drop the dominant axis of the plane normal
    // (a coordinate drop preserves the f64 values exactly, and for points
    // exactly on the plane it is injective).
    let n = {
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let drop_k = (0..3)
        .max_by(|&i, &j| n[i].abs().partial_cmp(&n[j].abs()).unwrap())
        .unwrap();
    if n[drop_k] == 0.0 {
        bail!("degenerate anchor face");
    }
    let proj = |v: usize| -> [f64; 2] {
        let p = tets.vertices[v];
        [p[(drop_k + 1) % 3], p[(drop_k + 2) % 3]]
    };
    let g2l: HashMap<usize, u32> = vset
        .iter()
        .enumerate()
        .map(|(li, &g)| (g, li as u32))
        .collect();
    let pts2d: Vec<[f64; 2]> = vset.iter().map(|&g| proj(g)).collect();
    let to_local_tris = |sheet: &HashMap<[usize; 3], (usize, usize)>| -> Vec<[usize; 3]> {
        let mut keys: Vec<&[usize; 3]> = sheet.keys().collect();
        keys.sort_unstable();
        keys.iter()
            .map(|k| {
                [
                    g2l[&k[0]] as usize,
                    g2l[&k[1]] as usize,
                    g2l[&k[2]] as usize,
                ]
            })
            .collect()
    };
    let bot_tris = to_local_tris(&s_bot);
    let top_tris = to_local_tris(&s_top);
    let Some(mut mesh) = Mesh::from_triangulation(&pts2d, &bot_tris) else {
        bail!("S_bot is not a valid 2-D triangulation");
    };
    let Some(top_mesh) = Mesh::from_triangulation(&pts2d, &top_tris) else {
        bail!("S_top is not a valid 2-D triangulation");
    };
    let top_set = top_mesh.triangles_sorted();

    // ── 5. Constraints: every surface segment with both endpoints in the
    // region (they lie in the plane automatically). REQUIRED ones - the
    // target (a, b) and any segment that currently exists ONLY in sandwich
    // flat tets (deleting the sandwich must not lose it) - must recover.
    // Segments missing from the mesh entirely are attempted BEST-EFFORT: a
    // stubborn one must not block the commit that fixes everything else (it
    // stays missing either way, and gets its own attempt - typically against
    // a cleaner mesh - when its turn comes). Segments existing in side tets
    // survive untouched and are skipped.
    let mut cons: Vec<(usize, usize)> = psegs
        .iter()
        .filter(|&&(u, w)| g2l.contains_key(&u) && g2l.contains_key(&w))
        .copied()
        .collect();
    cons.sort_unstable();
    let exists_outside_sandwich = |u: usize, w: usize| -> bool {
        tets.incident_tets(u).iter().any(|&ti| {
            let ti = ti as usize;
            tets.is_live(ti) && !flat.contains(&ti) && tets.tets[ti].verts.contains(&w)
        })
    };
    let exists_in_sandwich = |u: usize, w: usize| -> bool {
        tets.incident_tets(u).iter().any(|&ti| {
            let ti = ti as usize;
            flat.contains(&ti) && tets.is_live(ti) && tets.tets[ti].verts.contains(&w)
        })
    };
    let mut log: Vec<FlipLog> = Vec::new();
    let mut required: Vec<((u32, u32), (usize, usize))> = vec![((g2l[&a], g2l[&b]), (a, b))];
    let mut optional: Vec<(u32, u32)> = Vec::new();
    for &(u, w) in &cons {
        let lc = (g2l[&u], g2l[&w]);
        if (u, w) == (a.min(b), a.max(b)) {
            continue; // already in `required`
        }
        if exists_outside_sandwich(u, w) {
            continue; // survives in side tets untouched by this operation
        }
        if exists_in_sandwich(u, w) {
            required.push((lc, (u, w))); // would be LOST with the sandwich
        } else {
            optional.push(lc); // missing today, missing on failure - no harm
        }
    }
    // A vertex lying EXACTLY on the open 2-D segment (lu, lw): exact
    // collinearity + strictly between along the dominant axis. Such a segment
    // can never be an edge of ANY triangulation - report it to the caller for
    // the standard split-at-vertex resolution.
    let on_segment_vertex = |lu: u32, lw: u32| -> Option<usize> {
        let (pu, pw) = (pts2d[lu as usize], pts2d[lw as usize]);
        let ax = if (pw[0] - pu[0]).abs() >= (pw[1] - pu[1]).abs() {
            0
        } else {
            1
        };
        let (lo, hi) = (pu[ax].min(pw[ax]), pu[ax].max(pw[ax]));
        (0..pts2d.len() as u32)
            .find(|&z| {
                z != lu
                    && z != lw
                    && super::cdt2d::orient2d(pu, pw, pts2d[z as usize]) == 0.0
                    && pts2d[z as usize][ax] > lo
                    && pts2d[z as usize][ax] < hi
            })
            .map(|z| vset[z as usize])
    };
    // A vertex exactly ON a constraint must be resolved by the CALLER
    // (split-at-vertex) - the SoS recovery below would thread the edge
    // through the coincident vertex. Required → report; optional → skip.
    for &((lu, lw), (gu, gw)) in &required {
        if let Some(z) = on_segment_vertex(lu, lw) {
            if dbg {
                eprintln!(
                    "    [copl DBG] ({a},{b}): vertex {z} ON required segment ({gu},{gw}) - split"
                );
            }
            return SplitSegment {
                seg: (gu, gw),
                at: z,
            };
        }
    }
    let optional: Vec<(u32, u32)> = optional
        .into_iter()
        .filter(|&(u, w)| on_segment_vertex(u, w).is_none())
        .collect();
    // Recover to a FIXED POINT: an edge whose recovery stalls against the
    // current intermediate triangulation often becomes recoverable after
    // later edges land (order dependence of constrained insertion) - retry
    // failed edges until a full round makes no progress.
    for _round in 0..8 {
        let mut progress = false;
        let mut all_ok = true;
        for &((lu, lw), _) in &required {
            if mesh.has_edge(lu, lw) {
                continue;
            }
            if recover_edge_logged(&mut mesh, lu, lw, &mut log) {
                progress = true;
            } else {
                all_ok = false;
            }
        }
        for &(u, w) in &optional {
            if !mesh.has_edge(u, w) && recover_edge_logged(&mut mesh, u, w, &mut log) {
                progress = true;
            }
        }
        if all_ok || !progress {
            break;
        }
    }
    if let Some(&((lu, lw), (gu, gw))) = required
        .iter()
        .find(|&&((lu, lw), _)| !mesh.has_edge(lu, lw))
    {
        // Constraint-level failure: no memo (other segments of this region
        // may still succeed).
        if dbg {
            eprintln!(
                "    [copl DBG] ({a},{b}): required constraint ({gu},{gw}) not recoverable in 2-D - bail"
            );
            if std::env::var("YAMM_COPL_DUMP").is_ok() {
                eprintln!("    [copl DUMP] seg=({lu},{lw})");
                eprintln!("    [copl DUMP] pts={pts2d:?}");
                let mut tris: Vec<[u32; 3]> = mesh.triangles_sorted().into_iter().collect();
                tris.sort_unstable();
                eprintln!("    [copl DUMP] tris={tris:?}");
            }
        }
        return Failed;
    }
    let n_l1 = log.len();

    // ── 6. Path 2: transform T_c into S_top by recovering every S_top edge
    // (none can cross another, so one pass lands exactly on S_top).
    let mut top_edges: Vec<(u32, u32)> = top_tris
        .iter()
        .flat_map(|t| {
            [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].map(|(x, y)| {
                let (x, y) = (x as u32, y as u32);
                (x.min(y), x.max(y))
            })
        })
        .collect();
    top_edges.sort_unstable();
    top_edges.dedup();
    let mut p2_failed: Vec<(u32, u32)> = Vec::new();
    for _round in 0..8 {
        p2_failed.clear();
        let mut progress = false;
        for &(u, w) in &top_edges {
            if mesh.has_edge(u, w) {
                continue;
            }
            if recover_edge_logged(&mut mesh, u, w, &mut log) {
                progress = true;
            } else {
                p2_failed.push((u, w));
            }
        }
        if p2_failed.is_empty() || !progress {
            break;
        }
    }
    if mesh.triangles_sorted() != top_set {
        if dbg && std::env::var("YAMM_COPL_DUMP").is_ok() {
            eprintln!("    [copl DUMP] path2 failed edges={p2_failed:?}");
            eprintln!("    [copl DUMP] pts={pts2d:?}");
            let mut cur: Vec<[u32; 3]> = mesh.triangles_sorted().into_iter().collect();
            cur.sort_unstable();
            eprintln!("    [copl DUMP] cur_tris={cur:?}");
            let mut top: Vec<[usize; 3]> = top_tris.clone();
            top.sort_unstable();
            eprintln!("    [copl DUMP] top_tris={top:?}");
        }
        bail!("path 2 did not reach S_top");
    }

    // ── 7. Wiring simulation (pure, pre-commit): replay the log on the
    // S_bot face-key set; every flip's bottom pair must be exposed and the
    // final exposed set must be exactly S_top. Guarantees the 3-D commit
    // below cannot fail halfway.
    let l2g = |li: u32| vset[li as usize];
    let mut sim: HashSet<[usize; 3]> = s_bot.keys().copied().collect();
    for fl in &log {
        let [s1, s2] = fl.old_diag.map(l2g);
        let [x, y] = fl.new_diag.map(l2g);
        if !sim.remove(&fkey(s1, s2, x))
            || !sim.remove(&fkey(s1, s2, y))
            || !sim.insert(fkey(x, y, s1))
            || !sim.insert(fkey(x, y, s2))
        {
            bail!("wiring simulation failed");
        }
    }
    let top_keys: HashSet<[usize; 3]> = s_top.keys().copied().collect();
    if sim != top_keys {
        bail!("wiring simulation does not end at S_top");
    }

    if dbg {
        eprintln!(
            "    [copl DBG] ({a},{b}): COMMIT - region {} faces / {} flat / {} verts; req={} opt={} |L1|={} |L2|={}",
            faces.len(),
            flat.len(),
            vset.len(),
            required.len(),
            optional.len(),
            n_l1,
            log.len() - n_l1
        );
    }

    // ── 8. Commit: delete the sandwich, build the stack bottom-up, glue.
    let mut flat_sorted: Vec<usize> = flat.iter().copied().collect();
    flat_sorted.sort_unstable();
    for &ti in &flat_sorted {
        tets.free_tet(ti);
    }
    // The exposed sheet: face key → (tet, face slot) awaiting its up-side
    // neighbour. Starts as S_bot's owners (untouched below tets).
    let mut exposed: HashMap<[usize; 3], (usize, usize)> = s_bot.clone();
    for fl in &log {
        let [s1, s2] = fl.old_diag.map(l2g);
        let [x, y] = fl.new_diag.map(l2g);
        // verts = [s1, s2, x, y]: opposite_face slots - 0:(s2,x,y) top,
        // 1:(s1,y,x) top, 2:(s1,s2,y) bottom, 3:(s1,x,s2) bottom.
        let ni = tets.alloc_tet(Tet {
            verts: [s1, s2, x, y],
            adj: [usize::MAX; 4],
        });
        for (slot, key) in [(2usize, fkey(s1, s2, y)), (3, fkey(s1, s2, x))] {
            let (lo, lo_fi) = exposed.remove(&key).expect("simulated");
            tets.tets[ni].adj[slot] = lo;
            tets.tets[lo].adj[lo_fi] = ni;
        }
        exposed.insert(fkey(s2, x, y), (ni, 0));
        exposed.insert(fkey(s1, x, y), (ni, 1));
    }
    // Final glue: each exposed face meets its untouched up-side owner.
    for (key, &(up, up_fi)) in &s_top {
        let (lo, lo_fi) = exposed.remove(key).expect("simulated");
        tets.tets[up].adj[up_fi] = lo;
        tets.tets[lo].adj[lo_fi] = up;
    }
    debug_assert!(exposed.is_empty());

    if super::boundary_recovery::edge_exists_in_tets(tets, a, b) {
        Recovered
    } else {
        Failed
    }
}
