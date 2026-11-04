//! Conformal tracking from a point that is not inside the volume it is
//! tracked in.
//!
//! One volume made of two separate boxes, A = [0,1]^3 and B = [2,3] x [0,1]^2,
//! with a triangle missing from A's +x face. A ray from inside A towards the
//! hole leaves A through it and next meets B's -x face from outside: the
//! nearest facet of the volume ahead is one the ray enters it through.
//! `ray_fire` takes that hit as a crossing, which puts the particle on the
//! wrong side of B's face; `ray_fire_leaving` flags it.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// Closed axis-aligned box with outward winding; the +x face is split into
/// [1, 2, 6] (z <= y) and [1, 6, 5] (z >= y).
fn push_box(
    vertices: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[u32; 3]>,
    lo: [f64; 3],
    hi: [f64; 3],
) {
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
    for (a, b, c, d) in [
        (0, 3, 2, 1), // z = lo, normal -z
        (4, 5, 6, 7), // z = hi, normal +z
        (0, 1, 5, 4), // y = lo, normal -y
        (2, 3, 7, 6), // y = hi, normal +y
        (0, 4, 7, 3), // x = lo, normal -x
        (1, 2, 6, 5), // x = hi, normal +x
    ] {
        triangles.push([o + a, o + b, o + c]);
        triangles.push([o + a, o + c, o + d]);
    }
}

/// Boxes A and B as one volume, its surface bounded by the implicit
/// complement. `hole` drops A's +x triangle [1, 6, 5].
fn two_lumps(hole: bool) -> MeshGeometry {
    lumps(hole, false)
}

/// As [`two_lumps`]; `flip` winds A's +x triangle [1, 6, 5] the wrong way
/// round instead, as a folded facet is.
fn lumps(hole: bool, flip: bool) -> MeshGeometry {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    push_box(&mut vertices, &mut triangles, [0.0; 3], [1.0; 3]);
    if hole {
        triangles.remove(11);
    } else if flip {
        triangles[11].swap(1, 2);
    }
    push_box(
        &mut vertices,
        &mut triangles,
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 1.0],
    );
    let n = triangles.len();
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        tet_adjacency: vec![],
        tetrahedra: vec![],
        tet_volume_ids: vec![],
        tet_physical_groups: vec![],
        triangles,
        triangle_surface_ids: vec![1; n],
        triangle_physical_groups: vec![1; n],
        vertices,
        physical_groups_json: r#"{"1": {"name": "mat:a", "dim": 3}}"#.to_string(),
        surface_volumes_json: "[[0, null]]".to_string(),
        volume_measures: vec![],
        imprinted: Some(true),
        contact_tolerance: None,
        overlaps: vec![],
        separated: vec![],
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

/// Inside A, heading +x through the triangle that `hole` removes.
const ORIGIN: [f64; 3] = [0.5, 0.3, 0.7];
const PLUS_X: [f64; 3] = [1.0, 0.0, 0.0];

#[test]
fn a_ray_leaving_through_the_surface_is_a_crossing() {
    let geom = two_lumps(false);
    let (t, _, leaves) = geom.ray_fire_leaving(0, ORIGIN, PLUS_X).unwrap();
    assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    assert!(leaves);
    // The other way round, from inside B, the first facet ahead is B's +x face.
    let (t, _, leaves) = geom.ray_fire_leaving(0, [2.5, 0.3, 0.7], PLUS_X).unwrap();
    assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    assert!(leaves);
}

#[test]
fn a_ray_out_through_a_crack_meets_the_volume_from_outside() {
    let geom = two_lumps(true);
    // ray_fire takes B's -x face, 1.5 away, as the next crossing.
    let (t, _) = geom.ray_fire(0, ORIGIN, PLUS_X, None).unwrap();
    assert!((t - 1.5).abs() < 1e-12, "t = {t}");
    // The ray enters the volume there, so the particle was not inside it.
    let (t, _, leaves) = geom.ray_fire_leaving(0, ORIGIN, PLUS_X).unwrap();
    assert!((t - 1.5).abs() < 1e-12, "t = {t}");
    assert!(!leaves);
}

#[test]
fn a_wrong_wound_facet_ahead_of_a_particle_inside_is_still_its_way_out() {
    // A folded facet faces the wrong way, but the particle short of it is
    // inside the volume, so this is where it leaves, as ray_fire has it.
    let geom = lumps(false, true);
    let (t, _, leaves) = geom.ray_fire_leaving(0, ORIGIN, PLUS_X).unwrap();
    assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    assert!(leaves);
}
