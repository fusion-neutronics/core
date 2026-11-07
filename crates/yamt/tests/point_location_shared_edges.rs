//! Point location when a parity ray runs through, or within round-off of, an
//! edge or vertex that facets share (issue #502).
//!
//! Parity counts the facets a ray from the point crosses. Facets that share
//! an edge or vertex each decide alone whether a ray through it meets them,
//! so round-off can have the ray meet none of them and slip through the
//! surface, and a ray that only touches the surface there, at a fold, meets
//! it once. Either puts the point on the wrong side. Such a ray is passed over
//! for the next one, and the rays are cast along directions that the edges of
//! a mesh with a symmetry plane or axis-aligned faces do not line up with.

use std::f64::consts::PI;

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

fn mesh(
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    triangle_surface_ids: Vec<u32>,
    physical_groups_json: &str,
    surface_volumes_json: &str,
) -> MeshGeometry {
    let n = triangles.len();
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        triangle_physical_groups: triangle_surface_ids.iter().map(|&s| s as i32).collect(),
        triangle_surface_ids,
        triangles,
        vertices,
        physical_groups_json: physical_groups_json.to_string(),
        surface_volumes_json: surface_volumes_json.to_string(),
        imprinted: Some(true),
        ..Default::default()
    };
    assert_eq!(data.triangle_surface_ids.len(), n);
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

/// A closed faceted torus about the z axis, major radius 600 cm, wound
/// outward, on an `n_theta` x `n_zeta` grid: vertex (i, j) at poloidal angle
/// 2 pi i / n_theta and toroidal angle 2 pi j / n_zeta, each grid quad split
/// into two triangles.
fn push_torus(
    vertices: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[u32; 3]>,
    minor: f64,
    n_theta: u32,
    n_zeta: u32,
) {
    let o = vertices.len() as u32;
    for i in 0..n_theta {
        let theta = 2.0 * PI * f64::from(i) / f64::from(n_theta);
        let r = 600.0 + minor * theta.cos();
        for j in 0..n_zeta {
            let zeta = 2.0 * PI * f64::from(j) / f64::from(n_zeta);
            vertices.push([r * zeta.cos(), r * zeta.sin(), minor * theta.sin()]);
        }
    }
    let id = |i: u32, j: u32| o + (i % n_theta) * n_zeta + j % n_zeta;
    for i in 0..n_theta {
        for j in 0..n_zeta {
            let (a, b, c, d) = (id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j));
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
}

/// Two nested tori, minor radii 150 and 165 cm on a 24 x 120 grid, as an
/// imprinted mesh: volume 0 is inside the inner torus, volume 1 between the
/// two. As in the stellarator layers of #502, the zeta = 0 grid line of both
/// lies in the plane y = 0, and theta = pi / 2 is a grid angle, so the
/// vertices (600, 0, 150) and (600, 0, 165) are straight above (600, 0, 0).
fn nested_tori() -> MeshGeometry {
    let (mut vertices, mut triangles) = (Vec::new(), Vec::new());
    push_torus(&mut vertices, &mut triangles, 150.0, 24, 120);
    let inner = triangles.len();
    push_torus(&mut vertices, &mut triangles, 165.0, 24, 120);
    let surface_ids = (0..triangles.len())
        .map(|k| if k < inner { 1 } else { 2 })
        .collect();
    mesh(
        vertices,
        triangles,
        surface_ids,
        r#"{"1": {"name": "mat:plasma", "dim": 3}, "2": {"name": "mat:sol", "dim": 3}}"#,
        "[[0, 1], [1, null]]",
    )
}

#[test]
fn a_point_just_off_a_symmetry_plane_is_put_in_the_volume_it_is_in() {
    // Where yamc reads the cell of a particle born at (600, 0, 0): 1e-10 cm
    // along its flight. With a small y component of flight the point is
    // 1e-11 to 1e-16 cm off the plane y = 0, and a ray along +z from it runs
    // that close to the edges along the zeta = 0 grid line, and within
    // 1e-10 cm of the vertices on it. Parity first counted along +z, and that
    // ray slipped between the facets of the inner torus at 64 of these
    // points, which were then put between the tori. The rays now cast from
    // them keep clear of those edges, and an edge a ray does run that close
    // to is passed over (see the parity tests in point_in_volume.rs).
    let geom = nested_tori();
    let mut points = Vec::new();
    for k in 11..=16 {
        for y in [10f64.powi(-k), -10f64.powi(-k)] {
            for m in 0..200 {
                let phi = 2.0 * PI * (f64::from(m) + 0.5) / 200.0;
                points.push([600.0 + 1e-10 * phi.cos(), y, 1e-10 * phi.sin()]);
            }
        }
    }
    for p in points {
        assert_eq!(geom.point_in_volume_checked(0, p), Some(true), "{p:?}");
        assert_eq!(geom.point_in_volume_checked(1, p), Some(false), "{p:?}");
        assert!(geom.point_in_volume(0, p), "{p:?}");
        assert!(!geom.point_in_volume(1, p), "{p:?}");
        assert_eq!(geom.try_find_volume_from(u32::MAX, p), Some(0), "{p:?}");
        assert_eq!(geom.find_volume(p), 0, "{p:?}");
    }
}

#[test]
fn a_point_on_a_vertex_or_edge_is_put_on_one_side_of_the_surface() {
    // On a vertex, or within round-off of the middle of an edge, as a
    // particle born there is if the cell 1e-10 cm along its flight cannot be
    // found. Every ray from there meets the facets at the point itself, as
    // close to an edge as the point is, so often no ray can be counted. The
    // point is on the surface, where either side will do, and it is put on
    // one, not left unplaced (a lost particle).
    let geom = nested_tori();
    let ic = geom.implicit_complement();
    let topo = &geom.topology;
    for tri in topo.triangles.iter().step_by(3) {
        let [a, b, _] = tri.map(|v| topo.vertices[v as usize]);
        let minor = (a[0].hypot(a[1]) - 600.0).hypot(a[2]);
        let sides = if minor < 157.5 { [0, 1] } else { [1, ic] };
        for p in [a, [0, 1, 2].map(|k| 0.5 * (a[k] + b[k]))] {
            let placed = geom.try_find_volume_from(u32::MAX, p);
            assert!(
                placed.is_some_and(|v| sides.contains(&v)),
                "{p:?} {placed:?}"
            );
            assert!(sides.contains(&geom.find_volume(p)), "{p:?}");
        }
    }
}

/// The octahedron |x| + |y| + |z| <= 1 as one volume, wound outward.
fn octahedron() -> MeshGeometry {
    let vertices = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut triangles = Vec::new();
    for (x, y) in [(0, 2), (2, 1), (1, 3), (3, 0)] {
        triangles.push([x, y, 4]);
        triangles.push([y, x, 5]);
    }
    let n = triangles.len();
    mesh(
        vertices,
        triangles,
        vec![1; n],
        r#"{"1": {"name": "mat:a", "dim": 3}}"#,
        "[[0, null]]",
    )
}

#[test]
fn a_point_below_a_corner_or_edge_of_the_equator_is_outside() {
    // Straight below the corners and the edge midpoints of the octahedron's
    // equator, where a ray along +z, once the first parity ray, only touches
    // the octahedron, at the vertex or edge right above, and was counted as
    // crossing it.
    let geom = octahedron();
    let ic = geom.implicit_complement();
    for (x, y) in [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ] {
        let p = [x, y, -2.0];
        assert!(!geom.point_in_volume(0, p), "{p:?}");
        assert_eq!(geom.point_in_volume_checked(0, p), Some(false), "{p:?}");
        assert_eq!(geom.find_volume(p), ic, "{p:?}");
        assert_eq!(geom.try_find_volume_from(u32::MAX, p), Some(ic), "{p:?}");
    }
}

#[test]
fn a_point_below_the_top_corner_or_an_edge_from_it_is_inside() {
    // Straight below the top vertex and the midpoints of the edges down from
    // it, inside the octahedron, where a ray along +z leaves through the
    // vertex or edge right above.
    let geom = octahedron();
    for (x, y) in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.0, -0.5)] {
        let p = [x, y, 0.2];
        assert!(geom.point_in_volume(0, p), "{p:?}");
        assert_eq!(geom.point_in_volume_checked(0, p), Some(true), "{p:?}");
        assert_eq!(geom.find_volume(p), 0, "{p:?}");
        assert_eq!(geom.try_find_volume_from(u32::MAX, p), Some(0), "{p:?}");
    }
}

/// Unit boxes A = [0, 1]^3 (volume 0) and B, A moved 1 along `axis` (volume
/// 1), sharing the face between them, as an imprinted mesh.
fn two_boxes(axis: usize) -> MeshGeometry {
    // Vertex (a, b, c) is a along `axis` (0, 1 or 2) and b, c along the other
    // two (0 or 1).
    let other = [(axis + 1) % 3, (axis + 2) % 3];
    let mut vertices = Vec::new();
    for a in 0..3 {
        for b in 0..2 {
            for c in 0..2 {
                let mut v = [0.0; 3];
                v[axis] = f64::from(a);
                v[other[0]] = f64::from(b);
                v[other[1]] = f64::from(c);
                vertices.push(v);
            }
        }
    }
    let id = |a: u32, b: u32, c: u32| a * 4 + b * 2 + c;
    let (mut triangles, mut surface_ids) = (Vec::new(), Vec::new());
    // A quad as two triangles, wound so that their normals point away from
    // `inside`.
    let mut quad = |q: [u32; 4], inside: [f64; 3], surface: u32| {
        let p = |i: u32| vertices[i as usize];
        let (e1, e2) = (
            [0, 1, 2].map(|k| p(q[1])[k] - p(q[0])[k]),
            [0, 1, 2].map(|k| p(q[2])[k] - p(q[0])[k]),
        );
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let out: f64 = (0..3).map(|k| n[k] * (p(q[0])[k] - inside[k])).sum();
        for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            triangles.push(if out > 0.0 { t } else { [t[0], t[2], t[1]] });
            surface_ids.push(surface);
        }
    };
    let centre = |a: f64| {
        let mut c = [0.5; 3];
        c[axis] = a;
        c
    };
    for (lo, surface) in [(0, 1), (1, 3)] {
        let inside = centre(f64::from(lo) + 0.5);
        // The end face away from the other box.
        let a = if lo == 0 { 0 } else { 2 };
        quad(
            [id(a, 0, 0), id(a, 1, 0), id(a, 1, 1), id(a, 0, 1)],
            inside,
            surface,
        );
        // The four side faces.
        for (b0, c0, b1, c1) in [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)] {
            quad(
                [
                    id(lo, b0, c0),
                    id(lo, b1, c1),
                    id(lo + 1, b1, c1),
                    id(lo + 1, b0, c0),
                ],
                inside,
                surface,
            );
        }
    }
    // The shared face, wound out of A into B.
    quad(
        [id(1, 0, 0), id(1, 1, 0), id(1, 1, 1), id(1, 0, 1)],
        centre(0.5),
        2,
    );
    mesh(
        vertices,
        triangles,
        surface_ids,
        r#"{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}"#,
        "[[0, null], [0, 1], [1, null]]",
    )
}

#[test]
fn a_point_a_hair_off_a_shared_face_is_put_on_its_side_of_it() {
    // Where a particle born on the face is read, 1e-10 cm along its flight:
    // closer to the face than that when the flight is not square to it. A
    // ray from A crossing the face that close used to be passed over, so A
    // did not claim the point, and B, whose bounding box it is outside, was
    // never asked: the point was put in the implicit complement.
    for axis in 0..3 {
        let geom = two_boxes(axis);
        for gap in [1e-13, 1e-12, 1e-11, 5e-11] {
            for (side, volume) in [(-1.0, 0), (1.0, 1)] {
                let mut p = [0.3, 0.6, 0.45];
                p[axis] = 1.0 + side * gap;
                assert_eq!(geom.find_volume(p), volume, "{p:?}");
                assert_eq!(
                    geom.try_find_volume_from(u32::MAX, p),
                    Some(volume),
                    "{p:?}"
                );
                assert_eq!(geom.point_in_volume_checked(volume, p), Some(true), "{p:?}");
                assert_eq!(
                    geom.point_in_volume_checked(1 - volume, p),
                    Some(false),
                    "{p:?}"
                );
            }
        }
    }
}
