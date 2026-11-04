//! Point location that cannot say where a point is.
//!
//! Parity point location takes the first test ray that crosses no facet
//! edge-on and, when every ray does, answers "outside". A point it cannot
//! place inside a solid then lands in the implicit complement, and a particle
//! there carries on as void. `point_in_volume_checked` also checks each ray's
//! parity against the facet nearest the point along it (a ray from inside
//! leaves through it), passes over a ray where the two disagree, and returns
//! `None` when no ray gives a consistent answer; `try_find_volume_from` turns
//! that into "cannot place" rather than the implicit complement.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

/// The unit cube [0,1]^3 as one volume against the implicit complement,
/// wound outward. `flip` reverses the winding of the triangles it names.
fn cube(flip: &[usize]) -> MeshGeometry {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let mut triangles = Vec::new();
    for (a, b, c, d) in [
        (0, 3, 2, 1), // z = 0, normal -z: triangles 0, 1
        (4, 5, 6, 7), // z = 1, normal +z: triangles 2, 3
        (0, 1, 5, 4), // y = 0, normal -y: triangles 4, 5
        (2, 3, 7, 6), // y = 1, normal +y: triangles 6, 7
        (0, 4, 7, 3), // x = 0, normal -x: triangles 8, 9
        (1, 2, 6, 5), // x = 1, normal +x: triangles 10, 11
    ] {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    }
    for &i in flip {
        triangles[i].swap(1, 2);
    }
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

const INSIDE: [f64; 3] = [0.3, 0.4, 0.6];
const OUTSIDE_BELOW: [f64; 3] = [0.3, 0.4, -1.0];
const OUTSIDE_AWAY: [f64; 3] = [3.0, 3.0, 3.0];

#[test]
fn a_sound_cube_is_placed_as_before() {
    let geom = cube(&[]);
    let ic = geom.implicit_complement();
    for (point, inside) in [
        (INSIDE, true),
        (OUTSIDE_BELOW, false),
        (OUTSIDE_AWAY, false),
    ] {
        assert_eq!(
            geom.point_in_volume_checked(0, point),
            Some(inside),
            "{point:?}"
        );
        assert_eq!(geom.point_in_volume(0, point), inside, "{point:?}");
        let expected = if inside { 0 } else { ic };
        for hint in [0, ic, u32::MAX] {
            assert_eq!(
                geom.try_find_volume_from(hint, point),
                Some(expected),
                "{point:?}"
            );
            assert_eq!(geom.find_volume_from(hint, point), expected, "{point:?}");
        }
    }
}

#[test]
fn a_flipped_facet_on_the_first_ray_is_passed_over() {
    // The +z ray from INSIDE (y > x there, so it meets the top face's
    // triangle [4, 6, 7]) crosses a facet wound inward. Its parity still says
    // inside, its facing says the ray enters there: the next ray decides.
    let geom = cube(&[3]);
    assert_eq!(geom.point_in_volume_checked(0, INSIDE), Some(true));
    assert_eq!(geom.try_find_volume_from(u32::MAX, INSIDE), Some(0));
}

#[test]
fn a_point_no_ray_can_place_is_reported_as_unplaceable() {
    // Every facet wound inward: each ray's parity and facing disagree, so the
    // mesh cannot say whether INSIDE is in the volume. Parity alone answers
    // from the first ray (here, inside), and falls back to the implicit
    // complement when no ray can be used.
    let geom = cube(&(0..12).collect::<Vec<_>>());
    assert_eq!(geom.point_in_volume_checked(0, INSIDE), None);
    for hint in [0, geom.implicit_complement(), u32::MAX] {
        assert_eq!(geom.try_find_volume_from(hint, INSIDE), None);
    }
    // A point whose rays miss the cube is still placed outside it.
    assert_eq!(
        geom.try_find_volume_from(u32::MAX, OUTSIDE_AWAY),
        Some(geom.implicit_complement())
    );
}
