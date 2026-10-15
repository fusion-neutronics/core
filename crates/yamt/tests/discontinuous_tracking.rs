//! Discontinuous tracking: solids meshed one by one, without imprinting.
//!
//! Two touching boxes each carry their own closed surface, so where they meet
//! there are two surfaces, each bounding one box and the implicit complement.
//! These pin what a particle leaving one box through such a surface ends up
//! in: the other box when the faces coincide or overlap, the implicit
//! complement when there is a gap, and never a box it is not inside.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, Beyond, MeshGeometry};

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
        physical_groups_json:
            r#"{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}"#.to_string(),
        // Outward winding: each box is the forward volume of its own surface,
        // the implicit complement the other side.
        surface_volumes_json: "[[0, null], [1, null]]".to_string(),
        volume_measures: vec![],
        imprinted,
        contact_tolerance: imprinted.map(|_| 0.05),
        overlaps: vec![],
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
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
fn a_mesh_without_provenance_whose_solids_touch_is_flagged() {
    // Touching boxes with separate surfaces and nothing in the file to say
    // so: tracked conformally, which cannot cross the interface.
    assert!(two_boxes(0.0, false, None).looks_unimprinted(1000));
    assert!(two_boxes(-0.01, true, None).looks_unimprinted(1000));
    // Separated by a real gap: nothing to cross, nothing to flag.
    assert!(!two_boxes(0.5, false, None).looks_unimprinted(1000));
    // Recorded as non-imprinted: already tracked discontinuously.
    assert!(!two_boxes(0.0, false, Some(false)).looks_unimprinted(1000));
}

/// Boxes `[lo, hi]`, each its own solid and surface (no imprint), tracked
/// discontinuously with the given contact tolerance.
fn layered(spec: &[([f64; 3], [f64; 3])], tolerance: f64) -> MeshGeometry {
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
        contact_tolerance: Some(tolerance),
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
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
    let geom = layered(&[a, b, foil], 0.1);
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
