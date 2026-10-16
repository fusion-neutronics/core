//! Touching solids that do not share a surface where they meet.
//!
//! Tracking crosses from one solid into the next only through a surface the
//! two share. Two touching boxes that each carry their own closed surface
//! meet along two surfaces, each bounding one box and the implicit
//! complement, so a particle leaving one box is read into the implicit
//! complement and never enters the other. These pin the probe that flags
//! such a mesh on load.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// The six faces of the box `[lo, hi]`, each as two triangles wound with an
/// outward normal, in the order -z, +z, -y, +y, -x, +x.
fn box_faces(vertices: &mut Vec<[f64; 3]>, lo: [f64; 3], hi: [f64; 3]) -> Vec<[[u32; 3]; 2]> {
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
    let quad = |a: u32, b: u32, c: u32, d: u32| [[o + a, o + b, o + c], [o + a, o + c, o + d]];
    vec![
        quad(0, 3, 2, 1),
        quad(4, 5, 6, 7),
        quad(0, 1, 5, 4),
        quad(2, 3, 7, 6),
        quad(0, 4, 7, 3),
        quad(1, 2, 6, 5),
    ]
}

fn geometry(
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    triangle_surface_ids: Vec<u32>,
    surface_volumes_json: &str,
) -> MeshGeometry {
    let n = triangles.len();
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        tet_adjacency: vec![],
        tetrahedra: vec![],
        tet_volume_ids: vec![],
        tet_physical_groups: vec![],
        triangle_physical_groups: vec![1; n],
        triangles,
        triangle_surface_ids,
        vertices,
        physical_groups_json:
            r#"{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}"#.to_string(),
        surface_volumes_json: surface_volumes_json.to_string(),
        volume_measures: vec![],
        imprinted: None,
        contact_tolerance: None,
        overlaps: vec![],
        separated: vec![],
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

/// Box A = [0,1]^3 and box B = [1 + shift, 2 + shift] x [0,1]^2, each with
/// its own closed surface (surface 1 for A, 2 for B).
fn separate_surfaces(shift: f64) -> MeshGeometry {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut ids = Vec::new();
    for (surface, lo, hi) in [
        (1u32, [0.0; 3], [1.0; 3]),
        (2, [1.0 + shift, 0.0, 0.0], [2.0 + shift, 1.0, 1.0]),
    ] {
        for face in box_faces(&mut vertices, lo, hi) {
            triangles.extend_from_slice(&face);
            ids.extend([surface; 2]);
        }
    }
    geometry(vertices, triangles, ids, "[[0, null], [1, null]]")
}

#[test]
fn touching_solids_with_separate_surfaces_are_flagged() {
    assert!(separate_surfaces(0.0).touches_without_shared_surface(1000));
    // Overlapping a little, as two facetings of one CAD surface can.
    assert!(separate_surfaces(-0.01).touches_without_shared_surface(1000));
}

#[test]
fn solids_apart_are_not_flagged() {
    assert!(!separate_surfaces(0.5).touches_without_shared_surface(1000));
}

#[test]
fn touching_solids_sharing_their_interface_are_not_flagged() {
    // The imprinted form of the same two boxes: A's +x face is B's -x face,
    // one surface with A in front of it and B behind it.
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut ids = Vec::new();
    let a = box_faces(&mut vertices, [0.0; 3], [1.0; 3]);
    let b = box_faces(&mut vertices, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    for (k, face) in a.iter().enumerate() {
        triangles.extend_from_slice(face);
        ids.extend([if k == 5 { 3u32 } else { 1 }; 2]);
    }
    for face in &b[..4] {
        triangles.extend_from_slice(face);
        ids.extend([2u32; 2]);
    }
    triangles.extend_from_slice(&b[5]);
    ids.extend([2u32; 2]);
    let geom = geometry(vertices, triangles, ids, "[[0, null], [1, null], [0, 1]]");
    assert!(!geom.touches_without_shared_surface(1000));
}
