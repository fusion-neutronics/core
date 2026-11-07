//! Point location in a solid with a face that has the solid on both sides
//! (an internal face, as a file from another writer can have).
//!
//! The face's facets come twice in the solid's surface, once each way, so a
//! parity ray through the face crosses two facets at the same distance. Both
//! crossings count, and leave the parity as it was: taken as one, they put
//! the part of the solid beyond the face outside it.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// The box [0, 2]^3 as one volume, its outer faces on surface 1, wound
/// outward, and an internal face at z = 1 on surface 2.
fn box_with_internal_face() -> MeshGeometry {
    let mut vertices = Vec::new();
    for z in [0.0, 1.0, 2.0] {
        for (x, y) in [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)] {
            vertices.push([x, y, z]);
        }
    }
    let (mut triangles, mut surface_ids) = (Vec::new(), Vec::new());
    let mut quad = |q: [u32; 4], surface: u32| {
        triangles.push([q[0], q[1], q[2]]);
        triangles.push([q[0], q[2], q[3]]);
        surface_ids.extend([surface, surface]);
    };
    quad([0, 3, 2, 1], 1);
    quad([8, 9, 10, 11], 1);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        quad([a, b, b + 4, a + 4], 1);
        quad([a + 4, b + 4, b + 8, a + 8], 1);
    }
    quad([4, 5, 6, 7], 2);
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        triangle_physical_groups: surface_ids.iter().map(|&s| s as i32).collect(),
        triangle_surface_ids: surface_ids,
        triangles,
        vertices,
        physical_groups_json: r#"{"1": {"name": "mat:a", "dim": 3}}"#.to_string(),
        surface_volumes_json: "[[0, null], [0, 0]]".to_string(),
        imprinted: Some(true),
        ..Default::default()
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

#[test]
fn a_face_with_the_solid_on_both_sides_leaves_the_parity_as_it_was() {
    let geom = box_with_internal_face();
    let ic = geom.implicit_complement();
    // A grid of points through the box, none on a face. Parity counted the
    // two crossings of the internal face as one, and put about half of them
    // outside.
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                let p = [i, j, k].map(|c| 0.2 * f64::from(c) + 0.0731);
                assert!(geom.point_in_volume(0, p), "{p:?}");
                assert_eq!(geom.find_volume(p), 0, "{p:?}");
                assert_eq!(geom.point_in_volume_checked(0, p), Some(true), "{p:?}");
                assert_eq!(geom.try_find_volume_from(u32::MAX, p), Some(0), "{p:?}");
            }
        }
    }
    for p in [[1.1, 0.7, 2.5], [0.3, 1.6, -0.5], [2.4, 0.9, 1.3]] {
        assert!(!geom.point_in_volume(0, p), "{p:?}");
        assert_eq!(geom.find_volume(p), ic, "{p:?}");
        assert_eq!(geom.try_find_volume_from(u32::MAX, p), Some(ic), "{p:?}");
    }
}
