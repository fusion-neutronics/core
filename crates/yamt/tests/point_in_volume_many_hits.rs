//! Point containment by ray parity, for a volume whose parity ray crosses
//! more triangles than the inline hit buffer holds.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// One volume made of `n` unit boxes stacked along z with gaps between them
/// (box i spans z in [2i, 2i + 1]), as a model of many lumps or channels.
fn stacked_boxes(n: u32) -> MeshGeometry {
    let mut vertices = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for i in 0..n {
        let (lo, hi) = (2.0 * f64::from(i), 2.0 * f64::from(i) + 1.0);
        let o = vertices.len() as u32;
        for &(x, y, z) in &[
            (0.0, 0.0, lo),
            (1.0, 0.0, lo),
            (1.0, 1.0, lo),
            (0.0, 1.0, lo),
            (0.0, 0.0, hi),
            (1.0, 0.0, hi),
            (1.0, 1.0, hi),
            (0.0, 1.0, hi),
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
    // From the lowest box straight up, the ray crosses that box's top and
    // then two faces of each box above it: 65 crossings for 33 boxes.
    for n in [32, 33, 40] {
        let geom = stacked_boxes(n);
        assert!(geom.point_in_volume(0, [0.5, 0.5, 0.5]), "{n} boxes");
        assert_eq!(geom.find_volume([0.5, 0.5, 0.5]), 0, "{n} boxes");
        assert!(
            !geom.point_in_volume(0, [0.5, 0.5, 1.5]),
            "{n} boxes, in a gap"
        );
    }
}
