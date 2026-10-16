//! Discontinuous tracking: solids meshed one by one, without imprinting.
//!
//! Two touching boxes each carry their own closed surface, so where they meet
//! there are two surfaces, each bounding one box and the implicit complement.
//! These pin what a particle leaving one box through such a surface ends up
//! in: the other box when the faces coincide or overlap, the implicit
//! complement when there is a gap, and never a box it is not inside.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{add_vacuum_boundary, build_topology, ArrowMeshData, Beyond, MeshGeometry};

/// Closed axis-aligned box with outward winding. `split` picks which diagonal
/// splits the +x face, so two boxes can mesh a shared plane differently.
fn push_box(
    vertices: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[u32; 3]>,
    lo: [f64; 3],
    hi: [f64; 3],
    split_plus_x_other_way: bool,
    split_minus_x_other_way: bool,
) -> usize {
    let o = vertices.len() as u32;
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
        vertices.push([x, y, z]);
    }
    let quad = |a: u32, b: u32, c: u32, d: u32, other: bool| -> [[u32; 3]; 2] {
        if other {
            [[o + a, o + b, o + d], [o + b, o + c, o + d]]
        } else {
            [[o + a, o + b, o + c], [o + a, o + c, o + d]]
        }
    };
    let faces = [
        quad(0, 3, 2, 1, false),                   // z = lo, normal -z
        quad(4, 5, 6, 7, false),                   // z = hi, normal +z
        quad(0, 1, 5, 4, false),                   // y = lo, normal -y
        quad(2, 3, 7, 6, false),                   // y = hi, normal +y
        quad(0, 4, 7, 3, split_minus_x_other_way), // x = lo, normal -x
        quad(1, 2, 6, 5, split_plus_x_other_way),  // x = hi, normal +x
    ];
    let before = triangles.len();
    for f in faces {
        triangles.extend_from_slice(&f);
    }
    triangles.len() - before
}

/// Box A = [0,1]^3 and box B = [1 + shift, 2 + shift] x [0,1]^2, each its own
/// solid with its own surface (no imprint). B's -x face is split along the
/// other diagonal from A's +x face when `cross_split` is set.
fn two_boxes(shift: f64, cross_split: bool, imprinted: Option<bool>) -> MeshGeometry {
    let data = two_boxes_data(shift, cross_split, imprinted, imprinted.map(|_| 0.05));
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

fn two_boxes_data(
    shift: f64,
    cross_split: bool,
    imprinted: Option<bool>,
    contact_tolerance: Option<f64>,
) -> ArrowMeshData {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let na = push_box(
        &mut vertices,
        &mut triangles,
        [0.0; 3],
        [1.0; 3],
        false,
        false,
    );
    let nb = push_box(
        &mut vertices,
        &mut triangles,
        [1.0 + shift, 0.0, 0.0],
        [2.0 + shift, 1.0, 1.0],
        false,
        cross_split,
    );
    let mut triangle_surface_ids = vec![1u32; na];
    triangle_surface_ids.extend(vec![2u32; nb]);
    let mut triangle_physical_groups = vec![1i32; na];
    triangle_physical_groups.extend(vec![2i32; nb]);
    ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        tet_adjacency: vec![],
        tetrahedra: vec![],
        tet_volume_ids: vec![],
        tet_physical_groups: vec![],
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        vertices,
        physical_groups_json:
            r#"{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}"#.to_string(),
        // Outward winding: each box is the forward volume of its own surface,
        // the implicit complement the other side.
        surface_volumes_json: "[[0, null], [1, null]]".to_string(),
        volume_measures: vec![],
        imprinted,
        contact_tolerance,
        overlaps: vec![],
        separated: vec![],
    }
}

const A: u32 = 0;
const B: u32 = 1;
const IC: u32 = 2;

/// The next crossing and the volume beyond it, relocating when the tracker
/// defers that to the crossing being taken.
fn next(geom: &MeshGeometry, vol: u32, origin: [f64; 3], dir: [f64; 3]) -> (f64, Option<u32>) {
    let (t, _surf, beyond) = geom.next_crossing(vol, origin, dir).expect("a crossing");
    let next = match beyond {
        Beyond::Volume(v) => v,
        Beyond::Relocate(tri) => {
            let x = [
                origin[0] + t * dir[0],
                origin[1] + t * dir[1],
                origin[2] + t * dir[2],
            ];
            Some(geom.relocate(vol, tri, x, dir))
        }
    };
    (t, next)
}

#[test]
fn metadata_selects_discontinuous_tracking() {
    let geom = two_boxes(0.0, false, Some(false));
    assert!(geom.is_discontinuous());
    assert_eq!(geom.contact_tolerance(), Some(0.05));
    assert!(!geom.contacts().is_empty());

    let conformal = two_boxes(0.0, false, None);
    assert!(!conformal.is_discontinuous());
    assert_eq!(conformal.contact_tolerance(), None);
}

#[test]
fn conformal_tracking_cannot_cross_an_unshared_interface() {
    // The failure discontinuous tracking exists for: read off the topology,
    // the other side of A's +x face is the implicit complement, not B.
    let geom = two_boxes(0.0, false, None);
    let (_, surf) = geom
        .ray_fire(A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], None)
        .unwrap();
    assert_eq!(geom.next_volume(surf, A), Some(IC));
}

#[test]
fn coincident_faces_hand_over_both_ways() {
    let geom = two_boxes(0.0, false, Some(false));
    let (t, n) = next(&geom, A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(B));
    let (t, n) = next(&geom, B, [1.5, 0.5, 0.5], [-1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(A));
}

#[test]
fn coincident_faces_split_differently_still_hand_over() {
    // A's +x face and B's -x face are the same square split along opposite
    // diagonals, so a ray can cross A's facet where B's facets meet. Sweep
    // the face, including the diagonals themselves.
    let geom = two_boxes(0.0, true, Some(false));
    for i in 0..=20 {
        for j in 0..=20 {
            let (y, z) = (
                0.025 + 0.95 * i as f64 / 20.0,
                0.025 + 0.95 * j as f64 / 20.0,
            );
            let dir = [0.8, 0.36, 0.48];
            let start = [1.0 - 0.1 * dir[0], y - 0.1 * dir[1], z - 0.1 * dir[2]];
            if !(0.0..=1.0).contains(&start[1]) || !(0.0..=1.0).contains(&start[2]) {
                continue;
            }
            let (_, n) = next(&geom, A, start, dir);
            assert_eq!(n, Some(B), "ray through ({y}, {z}) did not enter B");
        }
    }
}

#[test]
fn an_overlap_sliver_belongs_to_the_solid_left() {
    // B pushed 0.01 into A: the sliver x in [0.99, 1] is inside both. Leaving
    // A at x = 1 the particle is inside B already, and B is where it goes.
    let geom = two_boxes(-0.01, false, Some(false));
    let (t, n) = next(&geom, A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!(
        (t - 0.5).abs() < 1e-12,
        "A's own face is the boundary, t = {t}"
    );
    assert_eq!(n, Some(B));
    // And from B the other way, the sliver is B's until B's face at 0.99.
    let (t, n) = next(&geom, B, [1.49, 0.5, 0.5], [-1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(A));
}

#[test]
fn a_gap_within_the_contact_tolerance_goes_to_the_solid_ahead() {
    // 0.01 apart with a 0.05 contact tolerance: touching, so the particle
    // leaving A is in B, and from B the gap is B's until B's own facet.
    let geom = two_boxes(0.01, false, Some(false));
    let (t, n) = next(&geom, A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(B));
    let (t, n) = next(&geom, B, [1.0 + 1e-8, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 1.01).abs() < 1e-7, "B's far face, t = {t}");
    assert_eq!(n, Some(IC));
}

#[test]
fn a_gap_wider_than_the_contact_tolerance_is_the_implicit_complement() {
    let geom = two_boxes(0.1, false, Some(false));
    let (t, n) = next(&geom, A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(IC));
    let (t, n) = next(&geom, IC, [1.0 + 1e-8, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.1).abs() < 1e-7, "B starts 0.1 on, t = {t}");
    assert_eq!(n, Some(B));
}

#[test]
fn leaving_the_model_goes_to_the_implicit_complement() {
    let geom = two_boxes(0.0, false, Some(false));
    let (t, n) = next(&geom, A, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(n, Some(IC));
    // Along the interface plane's edge: B touches A's top face only along
    // the line x = 1, which a ray leaving through the middle of it misses.
    let (_, n) = next(&geom, A, [0.9, 0.5, 0.5], [0.0, 0.6, 0.8]);
    assert_eq!(n, Some(IC));
}

#[test]
fn a_particle_outside_its_solid_is_relocated_in_place() {
    // Tracked in B but standing in the gap in front of it (given to B as
    // touching) and heading away, after a collision there: B has no facet
    // ahead to leave through, so the particle is re-located where it stands
    // (a zero-distance crossing) instead of being lost.
    let geom = two_boxes(0.01, false, Some(false));
    let (t, n) = next(&geom, B, [1.005, 0.5, 0.5], [-1.0, 0.0, 0.0]);
    assert_eq!(t, 0.0);
    assert_eq!(n, Some(IC));
}

#[test]
fn foreign_surface_verification_is_off() {
    // B overlapping A puts B's facet inside A by construction; for a
    // discontinuous mesh that is not an error.
    let geom = two_boxes(-0.01, false, Some(false));
    assert!(geom
        .segment_blocked(A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 0.5)
        .is_none());
    let conformal = two_boxes(-0.01, false, None);
    assert!(conformal
        .segment_blocked(A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 0.5)
        .is_some());
}

#[test]
fn the_exit_found_by_relocation_answers_the_next_query() {
    // Relocating from A into B finds B's far face; the ray-fire that follows
    // in B, from just past the crossing, reuses it. Its answer must be what a
    // fresh query gives (the cache is one-shot, so asking again is fresh).
    let geom = two_boxes(0.0, true, Some(false));
    let dir = [0.8, 0.36, 0.48];
    let start = [0.5, 0.3, 0.2];
    let (t, n) = next(&geom, A, start, dir);
    assert_eq!(n, Some(B));
    let past = [
        start[0] + (t + 1e-8) * dir[0],
        start[1] + (t + 1e-8) * dir[1],
        start[2] + (t + 1e-8) * dir[2],
    ];
    let (t_cached, s_cached, b_cached) = geom.next_crossing(B, past, dir).unwrap();
    let (t_fresh, s_fresh, b_fresh) = geom.next_crossing(B, past, dir).unwrap();
    assert!(
        (t_cached - t_fresh).abs() < 1e-12,
        "{t_cached} vs {t_fresh}"
    );
    assert_eq!((s_cached, b_cached), (s_fresh, b_fresh));
}

/// Boxes `[lo, hi]`, each its own solid and surface (no imprint), with the
/// given recorded overlaps `[low, high]`.
fn boxes(spec: &[([f64; 3], [f64; 3])], overlaps: Vec<[u32; 2]>) -> Result<MeshGeometry, String> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut triangle_surface_ids = Vec::new();
    let mut triangle_physical_groups = Vec::new();
    let mut groups = Vec::new();
    let mut sv = Vec::new();
    for (i, &(lo, hi)) in spec.iter().enumerate() {
        let n = push_box(&mut vertices, &mut triangles, lo, hi, false, false);
        triangle_surface_ids.extend(vec![i as u32 + 1; n]);
        triangle_physical_groups.extend(vec![i as i32 + 1; n]);
        groups.push(format!(r#""{}": {{"name": "mat:m{i}", "dim": 3}}"#, i + 1));
        sv.push(format!("[{i}, null]"));
    }
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        tet_adjacency: vec![],
        tetrahedra: vec![],
        tet_volume_ids: vec![],
        tet_physical_groups: vec![],
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        vertices,
        physical_groups_json: format!("{{{}}}", groups.join(", ")),
        surface_volumes_json: format!("[{}]", sv.join(", ")),
        volume_measures: vec![],
        imprinted: Some(false),
        contact_tolerance: Some(0.01),
        overlaps,
        separated: vec![],
    };
    build_topology(data)
        .map(MeshGeometry::from_topology)
        .map_err(|e| e.to_string())
}

const OUTER: ([f64; 3], [f64; 3]) = ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]);
const INNER: ([f64; 3], [f64; 3]) = ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);

#[test]
fn an_embedded_solid_is_invisible_without_a_recorded_overlap() {
    // The behaviour the overlap ranking exists to fix: an overlap goes to the
    // solid the particle is in, so from the outer box the inner one is never
    // entered and the ray leaves at the outer box's far face.
    let geom = boxes(&[OUTER, INNER], vec![]).unwrap();
    let (t, n) = next(&geom, 0, [0.5, 1.5, 1.5], [1.0, 0.0, 0.0]);
    assert!((t - 2.5).abs() < 1e-12, "t = {t}");
    assert_eq!(n, Some(2), "out into the implicit complement");
}

#[test]
fn a_recorded_overlap_gives_the_shared_region_to_the_higher_solid() {
    let geom = boxes(&[OUTER, INNER], vec![[0, 1]]).unwrap();
    // From the outer box, the inner one is entered at its face.
    let (t, n) = next(&geom, 0, [0.5, 1.5, 1.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    assert_eq!(n, Some(1));
    // Leaving the inner box lands back in the outer one, which touches it
    // nowhere: the recorded overlap is what offers it to relocation.
    let (t, n) = next(&geom, 1, [1.5, 1.5, 1.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    assert_eq!(n, Some(0));
    // A point in both belongs to the higher-ranked solid.
    assert_eq!(geom.find_volume([1.5, 1.5, 1.5]), 1);
    assert_eq!(geom.find_volume([0.5, 0.5, 0.5]), 0);
}

#[test]
fn a_partial_overlap_is_ranked_both_ways() {
    // B pokes 0.5 into A: the shared slab x in [2.5, 3] is B's.
    let a = ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]);
    let b = ([2.5, 1.0, 1.0], [4.0, 2.0, 2.0]);
    let geom = boxes(&[a, b], vec![[0, 1]]).unwrap();
    let (t, n) = next(&geom, 0, [1.0, 1.5, 1.5], [1.0, 0.0, 0.0]);
    assert!((t - 1.5).abs() < 1e-12, "A hands over at B's face, t = {t}");
    assert_eq!(n, Some(1));
    let (t, n) = next(&geom, 1, [3.5, 1.5, 1.5], [-1.0, 0.0, 0.0]);
    assert!(
        (t - 1.0).abs() < 1e-12,
        "B keeps the slab until its own face, t = {t}"
    );
    assert_eq!(n, Some(0));
    assert_eq!(geom.find_volume([2.75, 1.5, 1.5]), 1);
}

#[test]
fn cyclic_overlaps_are_refused() {
    let err = boxes(&[OUTER, INNER], vec![[0, 1], [1, 0]]).err().unwrap();
    assert!(err.contains("cyclic"), "{err}");
}

#[test]
fn touching_solids_without_a_shared_surface_are_flagged() {
    // Touching boxes with separate surfaces, tracked conformally (nothing in
    // the file, or a file claiming an imprint that did not share the face):
    // conformal tracking cannot cross the interface.
    assert!(two_boxes(0.0, false, None).touches_without_shared_surface(1000));
    assert!(two_boxes(-0.01, true, None).touches_without_shared_surface(1000));
    assert!(two_boxes(0.0, false, Some(true)).touches_without_shared_surface(1000));
    // Separated by a real gap: nothing to cross, nothing to flag.
    assert!(!two_boxes(0.5, false, None).touches_without_shared_surface(1000));
    // Recorded as non-imprinted: already tracked discontinuously.
    assert!(!two_boxes(0.0, false, Some(false)).touches_without_shared_surface(1000));
}

/// Boxes `[lo, hi]`, each its own solid and surface (no imprint), tracked
/// discontinuously with the given contact tolerance.
fn layered(
    spec: &[([f64; 3], [f64; 3])],
    tolerance: f64,
    separated: Vec<[u32; 2]>,
) -> MeshGeometry {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let counts: Vec<usize> = spec
        .iter()
        .map(|&(lo, hi)| push_box(&mut vertices, &mut triangles, lo, hi, false, false))
        .collect();
    solids(vertices, triangles, &counts, tolerance, separated)
}

/// Solids pushed one after another (`counts` triangles each), each its own
/// surface with the implicit complement outside, meshed without imprinting.
fn solids(
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    counts: &[usize],
    tolerance: f64,
    separated: Vec<[u32; 2]>,
) -> MeshGeometry {
    let mut triangle_surface_ids = Vec::new();
    let mut triangle_physical_groups = Vec::new();
    let mut groups = Vec::new();
    let mut sv = Vec::new();
    for (i, &n) in counts.iter().enumerate() {
        triangle_surface_ids.extend(vec![i as u32 + 1; n]);
        triangle_physical_groups.extend(vec![i as i32 + 1; n]);
        groups.push(format!(r#""{}": {{"name": "mat:m{i}", "dim": 3}}"#, i + 1));
        sv.push(format!("[{i}, null]"));
    }
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        tet_adjacency: vec![],
        tetrahedra: vec![],
        tet_volume_ids: vec![],
        tet_physical_groups: vec![],
        triangles,
        triangle_surface_ids,
        triangle_physical_groups,
        vertices,
        physical_groups_json: format!("{{{}}}", groups.join(", ")),
        surface_volumes_json: format!("[{}]", sv.join(", ")),
        volume_measures: vec![],
        imprinted: Some(false),
        contact_tolerance: Some(tolerance),
        overlaps: vec![],
        separated,
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

/// Square tube along z: outer [-1, 2]^2, a square hole [-g, 1 + g]^2 around
/// the unit box, z in [0, 1], wound outward (into the hole on its inner wall).
fn push_tube(vertices: &mut Vec<[f64; 3]>, triangles: &mut Vec<[u32; 3]>, g: f64) -> usize {
    let o = vertices.len() as u32;
    let outer = [(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)];
    let inner = [(-g, -g), (1.0 + g, -g), (1.0 + g, 1.0 + g), (-g, 1.0 + g)];
    for z in [0.0, 1.0] {
        for &(x, y) in &outer {
            vertices.push([x, y, z]);
        }
    }
    for z in [0.0, 1.0] {
        for &(x, y) in &inner {
            vertices.push([x, y, z]);
        }
    }
    let oi = |i: u32, z: u32| o + (i % 4) + 4 * z;
    let ii = |i: u32, z: u32| o + 8 + (i % 4) + 4 * z;
    let before = triangles.len();
    for i in 0..4u32 {
        triangles.push([oi(i, 1), oi(i + 1, 1), ii(i + 1, 1)]);
        triangles.push([oi(i, 1), ii(i + 1, 1), ii(i, 1)]);
        triangles.push([oi(i, 0), ii(i, 0), ii(i + 1, 0)]);
        triangles.push([oi(i, 0), ii(i + 1, 0), oi(i + 1, 0)]);
        triangles.push([oi(i, 0), oi(i + 1, 0), oi(i + 1, 1)]);
        triangles.push([oi(i, 0), oi(i + 1, 1), oi(i, 1)]);
        triangles.push([ii(i, 0), ii(i, 1), ii(i + 1, 1)]);
        triangles.push([ii(i, 0), ii(i + 1, 1), ii(i + 1, 0)]);
    }
    triangles.len() - before
}

#[test]
fn a_layer_thinner_than_the_contact_tolerance_is_not_skipped() {
    // Block A, a 0.05 thick foil F against it, block B against the foil,
    // with a 0.1 contact tolerance: B is within the tolerance of A's face
    // too. Volume ids put B before F, so a first-come choice would snap from
    // A straight into B and lose the foil.
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([1.05, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let foil = ([1.0, 0.0, 0.0], [1.05, 1.0, 1.0]);
    let geom = layered(&[a, b, foil], 0.1, vec![]);
    let (t, n) = next(&geom, 0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.5).abs() < 1e-12);
    assert_eq!(
        n,
        Some(2),
        "leaving A enters the foil, not the block beyond it"
    );
    let (t, n) = next(&geom, 2, [1.0 + 1e-8, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.05).abs() < 1e-7, "t = {t}");
    assert_eq!(n, Some(1));
}

#[test]
fn a_gap_the_cad_has_is_not_filled() {
    // Two blocks 0.03 apart, within the 0.05 contact tolerance. Unless the
    // writer recorded them as separated in the CAD, the gap counts as a
    // faceting gap and is given to the block ahead; recorded, it stays void.
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([1.03, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let touching = layered(&[a, b], 0.05, vec![]);
    let (_, n) = next(&touching, 0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert_eq!(n, Some(1));
    let separated = layered(&[a, b], 0.05, vec![[1, 0]]);
    let (_, n) = next(&separated, 0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert_eq!(n, Some(2), "the modelled gap is the implicit complement");
    let (t, n) = next(&separated, 2, [1.0 + 1e-8, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.03).abs() < 1e-7, "t = {t}");
    assert_eq!(n, Some(1));
}

#[test]
fn a_particle_heading_back_across_a_hole_leaves_the_solid_it_was_given() {
    // Block A in the square hole of tube B, with a 0.01 gap all round and a
    // 0.05 contact tolerance: leaving A, the particle is given B in the gap.
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let na = push_box(
        &mut vertices,
        &mut triangles,
        [0.0; 3],
        [1.0; 3],
        false,
        false,
    );
    let nb = push_tube(&mut vertices, &mut triangles, 0.01);
    let geom = solids(vertices, triangles, &[na, nb], 0.05, vec![]);
    assert_eq!(next(&geom, A, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]).1, Some(B));
    // Carrying on, it crosses the gap and the tube to the tube's outer wall.
    let (t, _) = next(&geom, B, [1.005, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.995).abs() < 1e-9, "t = {t}");
    // Scattered back in the gap, its next facet of B is the far wall of the
    // hole, an entry: it has left B's side and is re-located where it is (in
    // the gap), rather than crossing A and the hole as B.
    let (t, cell) = next(&geom, B, [1.005, 0.5, 0.5], [-1.0, 0.0, 0.0]);
    assert_eq!((t, cell), (0.0, Some(IC)));
}

#[test]
fn a_separated_layer_in_front_of_a_touching_solid_keeps_its_gap() {
    // Block A, a foil C 0.02 off it that the CAD says is apart from A, and
    // block B 0.06 off it that the CAD does not (they touch elsewhere, say),
    // all within the 0.1 contact tolerance. The gap to C is real: leaving A
    // goes to the implicit complement, and from there into C, not past it.
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([1.06, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let c = ([1.02, 0.0, 0.0], [1.04, 1.0, 1.0]);
    let geom = layered(&[a, b, c], 0.1, vec![[0, 2]]);
    let ic = geom.implicit_complement();
    assert_eq!(next(&geom, 0, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]).1, Some(ic));
    let (t, cell) = next(&geom, ic, [1.0 + 1e-8, 0.5, 0.5], [1.0, 0.0, 0.0]);
    assert!((t - 0.02).abs() < 1e-6, "t = {t}");
    assert_eq!(cell, Some(2));
}

#[test]
fn the_default_contact_tolerance_ignores_a_graveyard() {
    // A file that says it was not imprinted but records no tolerance gets a
    // ten-thousandth of its solids' bounding-box diagonal ([0,2]x[0,1]^2
    // here), whatever box a graveyard adds around them.
    let expected = 1e-4 * 6f64.sqrt();
    for graveyard in [None, Some(10.0)] {
        let mut data = two_boxes_data(0.0, false, Some(false), None);
        if let Some(offset) = graveyard {
            add_vacuum_boundary(&mut data, offset);
        }
        let geom = MeshGeometry::from_topology(build_topology(data).unwrap());
        let tol = geom.contact_tolerance().unwrap();
        assert!((tol - expected).abs() < 1e-15, "{graveyard:?}: {tol}");
    }
}

/// The cells a ray passes through, stepping 1e-8 past each crossing as the
/// transport does. Zero-length visits (a hand-over at the point of arrival)
/// are left out.
fn walk(geom: &MeshGeometry, mut vol: u32, mut pos: [f64; 3], dir: [f64; 3]) -> Vec<u32> {
    let mut cells = Vec::new();
    for _ in 0..12 {
        let Some((t, _, beyond)) = geom.next_crossing(vol, pos, dir) else {
            break;
        };
        let x = [
            pos[0] + t * dir[0],
            pos[1] + t * dir[1],
            pos[2] + t * dir[2],
        ];
        let next = match beyond {
            Beyond::Volume(v) => v,
            Beyond::Relocate(tri) => Some(geom.relocate(vol, tri, x, dir)),
        };
        if t > 1e-6 {
            cells.push(vol);
        }
        pos = [
            x[0] + 1e-8 * dir[0],
            x[1] + 1e-8 * dir[1],
            x[2] + 1e-8 * dir[2],
        ];
        match next {
            Some(v) => vol = v,
            None => break,
        }
    }
    cells
}

#[test]
fn touching_parts_inside_a_host_hand_over_to_each_other() {
    // Host L = [0,10]^3 holds parts P and C, which touch at x = 5; both take
    // their overlap with L. Crossing from P into C enters C: L contains the
    // crossing point, but C's coincident entry is taken at once.
    let l = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let p = ([2.0, 2.0, 2.0], [5.0, 8.0, 8.0]);
    let c = ([5.0, 2.0, 2.0], [8.0, 8.0, 8.0]);
    let geom = boxes(&[l, p, c], vec![[0, 1], [0, 2]]).unwrap();
    let cells = walk(&geom, 1, [3.0, 5.0, 5.0], [1.0, 0.0, 0.0]);
    assert_eq!(&cells[..3], &[1, 2, 0], "{cells:?}");
    let back = walk(&geom, 2, [7.0, 5.0, 5.0], [-1.0, 0.0, 0.0]);
    assert_eq!(&back[..3], &[2, 1, 0], "{back:?}");
}

#[test]
fn a_part_flush_with_its_hosts_face_is_entered_from_outside() {
    // H = [1,2]^2 x [1,3] reaches the top face of L = [0,3]^3 and takes its
    // overlap with L. Coming down through that face, whichever of the two
    // coplanar facets the ray meets first, the particle is in H.
    let l = ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]);
    let h = ([1.0, 1.0, 1.0], [2.0, 2.0, 3.0]);
    let geom = boxes(&[l, h], vec![[0, 1]]).unwrap();
    let ic = geom.implicit_complement();
    for (x, y) in [(1.5, 1.5), (1.2, 1.7), (1.9, 1.1), (1.25, 1.25)] {
        let cells = walk(&geom, ic, [x, y, 4.0], [0.0, 0.0, -1.0]);
        assert_eq!(&cells[..3], &[ic, 1, 0], "at ({x}, {y}): {cells:?}");
    }
}

#[test]
fn a_particle_outside_its_solid_is_not_carried_to_a_far_overlap() {
    // L = [0,3]^3, with H = [-3,0.5]x[1,2]^2 taking its overlap. A particle
    // tracked in L but standing just outside it (as after a collision in a
    // gap given to L), heading away, has no exit of L ahead: it is re-located
    // where it stands, not carried to where the ray would enter H.
    let l = ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]);
    let h = ([-3.0, 1.0, 1.0], [0.5, 2.0, 2.0]);
    let geom = boxes(&[l, h], vec![[0, 1]]).unwrap();
    let d = 0.5f64.sqrt();
    let (t, cell) = next(&geom, 0, [-0.001, 2.5, 1.5], [-d, -d, 0.0]);
    assert_eq!((t, cell), (0.0, Some(geom.implicit_complement())));
}

#[test]
fn an_internal_face_does_not_move_a_particle_out_of_its_solid() {
    // One solid, the unit box, with an internal face at z = 0.5 that has the
    // box on both sides, as a file from another writer can have; the face is
    // in the solid's surface list twice, once each way. Crossing it, whichever
    // copy the ray meets first, the particle stays in the box.
    for flip in [false, true] {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let n_box = push_box(
            &mut vertices,
            &mut triangles,
            [0.0; 3],
            [1.0; 3],
            false,
            false,
        );
        let o = vertices.len() as u32;
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            vertices.push([x, y, 0.5]);
        }
        if flip {
            triangles.push([o, o + 2, o + 1]);
            triangles.push([o, o + 3, o + 2]);
        } else {
            triangles.push([o, o + 1, o + 2]);
            triangles.push([o, o + 2, o + 3]);
        }
        let mut triangle_surface_ids = vec![1u32; n_box];
        triangle_surface_ids.extend([2, 2]);
        let data = ArrowMeshData {
            triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
            tet_aabbs: build_tet_aabbs(&[], &vertices),
            tet_adjacency: vec![],
            tetrahedra: vec![],
            tet_volume_ids: vec![],
            tet_physical_groups: vec![],
            triangle_physical_groups: vec![1; triangles.len()],
            triangles,
            triangle_surface_ids,
            vertices,
            physical_groups_json: r#"{"1": {"name": "mat:a", "dim": 3}}"#.to_string(),
            surface_volumes_json: "[[0, null], [0, 0]]".to_string(),
            volume_measures: vec![],
            imprinted: Some(false),
            contact_tolerance: Some(0.01),
            overlaps: vec![],
            separated: vec![],
        };
        let geom = MeshGeometry::from_topology(build_topology(data).unwrap());
        let n = (0.1f64 * 0.1 + 0.05 * 0.05 + 1.0).sqrt();
        let dir = [0.1 / n, 0.05 / n, 1.0 / n];
        for i in 0..20 {
            for j in 0..20 {
                // Points whose ray meets the internal face well inside the box.
                let p = [
                    0.05 + 0.045 * f64::from(i),
                    0.05 + 0.045 * f64::from(j),
                    0.25,
                ];
                let (t, cell) = next(&geom, 0, p, dir);
                assert!(
                    t > 0.0 && cell == Some(0),
                    "flip {flip}, from {p:?}: {t} into {cell:?}"
                );
            }
        }
    }
}
