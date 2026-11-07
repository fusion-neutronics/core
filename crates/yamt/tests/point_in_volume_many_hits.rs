//! Point containment by ray parity, for a volume whose parity ray crosses
//! more triangles than the inline hit buffer holds.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// One volume bounded by `n` nested cube surfaces about the origin (cube i
/// spans [-i, i]^3), as a model of many lumps or channels: shells alternate
/// in and out of the volume, and a ray from near the centre crosses every
/// cube once, whichever way it goes.
fn nested_cubes(n: u32) -> MeshGeometry {
    let mut vertices = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for i in 1..=n {
        let (lo, hi) = (-f64::from(i), f64::from(i));
        let o = vertices.len() as u32;
        for &(x, y, z) in &[
            (lo, lo, lo),
            (hi, lo, lo),
            (hi, hi, lo),
            (lo, hi, lo),
            (lo, lo, hi),
            (hi, lo, hi),
            (hi, hi, hi),
            (lo, hi, hi),
        ] {
            vertices.push([x, y, z]);
        }
        for [a, b, c, d] in [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ] {
            triangles.push([o + a, o + b, o + c]);
            triangles.push([o + a, o + c, o + d]);
        }
    }
    let n_tris = triangles.len();
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        triangles,
        triangle_surface_ids: vec![1; n_tris],
        triangle_physical_groups: vec![1; n_tris],
        vertices,
        physical_groups_json: r#"{"1": {"name": "mat:a", "dim": 3}}"#.to_string(),
        surface_volumes_json: "[[0, null]]".to_string(),
        ..Default::default()
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

#[test]
fn a_parity_ray_through_many_lumps_keeps_every_crossing() {
    // From inside the smallest cube a ray crosses all n cubes: 65 crossings
    // for 65 cubes. From between the first two it crosses the n - 1 outer
    // ones, and the first twice or not at all. The checked count keeps every
    // crossing too (it leaves through the smallest cube, as an odd count says).
    for n in [63, 65, 79] {
        let geom = nested_cubes(n);
        let centre = [0.1, 0.2, 0.3];
        assert!(geom.point_in_volume(0, centre), "{n} cubes");
        assert_eq!(geom.find_volume(centre), 0, "{n} cubes");
        assert_eq!(
            geom.point_in_volume_checked(0, centre),
            Some(true),
            "{n} cubes"
        );
        assert!(
            !geom.point_in_volume(0, [1.5, 0.2, 0.3]),
            "{n} cubes, in a gap"
        );
    }
}
