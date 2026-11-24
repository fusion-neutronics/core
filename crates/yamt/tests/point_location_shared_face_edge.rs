//! Point location on, or a hair beside, the edge of a face two volumes share
//! (issue #533, part 1).
//!
//! Each volume casts its own parity rays and passes over a ray that runs
//! within round-off of one of its edges. At a point on the edge of a shared
//! face the two volumes then decided along different rays: one counted a ray
//! leaving through the shared face at t = 0, the other passed its first ray
//! over and counted one leaving through the same face further on. Neither
//! claimed the point, which was put in the implicit complement or not placed
//! at all.

use yamt::mesh::topology::{build_tet_aabbs, build_triangle_aabbs};
use yamt::{build_topology, ArrowMeshData, MeshGeometry};

const X: [f64; 2] = [-25000.0, -23000.0];
/// The floor runs from Y[0] to Y[2], the wall stands on it from Y[1] to Y[2].
const Y: [f64; 3] = [-24000.0, -22100.0, -22000.0];
/// The floor from Z[0] to Z[1], the wall from Z[1] to Z[2].
const Z: [f64; 3] = [-6100.0, -6000.0, -5000.0];

struct Builder {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    surface_ids: Vec<u32>,
}

impl Builder {
    fn vertex(&mut self, p: [f64; 3]) -> u32 {
        if let Some(i) = self.vertices.iter().position(|v| *v == p) {
            return i as u32;
        }
        self.vertices.push(p);
        (self.vertices.len() - 1) as u32
    }

    /// A convex planar polygon, fanned from its first corner and wound so
    /// that its normal points along `out`.
    fn face(&mut self, corners: &[[f64; 3]], out: [f64; 3], surface: u32) {
        let ids: Vec<u32> = corners.iter().map(|&p| self.vertex(p)).collect();
        for k in 1..ids.len() - 1 {
            let (a, b, c) = (corners[0], corners[k], corners[k + 1]);
            let e1 = [0, 1, 2].map(|i| b[i] - a[i]);
            let e2 = [0, 1, 2].map(|i| c[i] - a[i]);
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let facing = n[0] * out[0] + n[1] * out[1] + n[2] * out[2];
            assert!(facing != 0.0, "degenerate triangle");
            let t = if facing > 0.0 {
                [ids[0], ids[k], ids[k + 1]]
            } else {
                [ids[0], ids[k + 1], ids[k]]
            };
            self.triangles.push(t);
            self.surface_ids.push(surface);
        }
    }
}

/// A floor slab and a wall standing on it along its +y edge, flush with it,
/// at the coordinates of the bioshield model of #533, as an imprinted mesh:
/// volume 0 the floor, volume 1 the wall, surface 2 the face they share.
fn floor_and_wall() -> MeshGeometry {
    let mut b = Builder {
        vertices: Vec::new(),
        triangles: Vec::new(),
        surface_ids: Vec::new(),
    };
    let ([x0, x1], [ya, yb, yc], [z0, z1, z2]) = (X, Y, Z);
    // The floor's own boundary.
    b.face(
        &[[x0, ya, z0], [x1, ya, z0], [x1, yc, z0], [x0, yc, z0]],
        [0.0, 0.0, -1.0],
        1,
    );
    b.face(
        &[[x0, ya, z0], [x1, ya, z0], [x1, ya, z1], [x0, ya, z1]],
        [0.0, -1.0, 0.0],
        1,
    );
    b.face(
        &[[x0, yc, z0], [x1, yc, z0], [x1, yc, z1], [x0, yc, z1]],
        [0.0, 1.0, 0.0],
        1,
    );
    for (x, out) in [(x0, -1.0), (x1, 1.0)] {
        // Its top edge has a vertex where the wall's face meets it.
        b.face(
            &[
                [x, ya, z0],
                [x, yc, z0],
                [x, yc, z1],
                [x, yb, z1],
                [x, ya, z1],
            ],
            [out, 0.0, 0.0],
            1,
        );
    }
    b.face(
        &[[x0, ya, z1], [x1, ya, z1], [x1, yb, z1], [x0, yb, z1]],
        [0.0, 0.0, 1.0],
        1,
    );
    // The shared face, wound out of the floor.
    b.face(
        &[[x0, yb, z1], [x1, yb, z1], [x1, yc, z1], [x0, yc, z1]],
        [0.0, 0.0, 1.0],
        2,
    );
    // The wall's own boundary.
    b.face(
        &[[x0, yb, z2], [x1, yb, z2], [x1, yc, z2], [x0, yc, z2]],
        [0.0, 0.0, 1.0],
        3,
    );
    b.face(
        &[[x0, yb, z1], [x1, yb, z1], [x1, yb, z2], [x0, yb, z2]],
        [0.0, -1.0, 0.0],
        3,
    );
    b.face(
        &[[x0, yc, z1], [x1, yc, z1], [x1, yc, z2], [x0, yc, z2]],
        [0.0, 1.0, 0.0],
        3,
    );
    for (x, out) in [(x0, -1.0), (x1, 1.0)] {
        b.face(
            &[[x, yb, z1], [x, yc, z1], [x, yc, z2], [x, yb, z2]],
            [out, 0.0, 0.0],
            3,
        );
    }
    let Builder {
        vertices,
        triangles,
        surface_ids,
    } = b;
    let data = ArrowMeshData {
        triangle_aabbs: build_triangle_aabbs(&triangles, &vertices),
        tet_aabbs: build_tet_aabbs(&[], &vertices),
        triangle_physical_groups: surface_ids.iter().map(|&s| s as i32).collect(),
        triangle_surface_ids: surface_ids,
        triangles,
        vertices,
        physical_groups_json:
            r#"{"1": {"name": "mat:floor", "dim": 3}, "2": {"name": "mat:wall", "dim": 3}}"#
                .to_string(),
        surface_volumes_json: "[[0, null], [0, 1], [1, null]]".to_string(),
        imprinted: Some(true),
        ..Default::default()
    };
    MeshGeometry::from_topology(build_topology(data).unwrap())
}

/// Offsets from an edge, out to well beyond round-off at these coordinates.
const OFFSETS: [f64; 9] = [
    -1e-9, -1e-10, -1e-11, -1e-12, 0.0, 1e-12, 1e-11, 1e-10, 1e-9,
];

/// The volumes a point at `offset` (dy, dz) from an edge at (y, z) of the
/// shared face may fairly be put in: every one whose closure it is within
/// round-off of.
fn fair(geom: &MeshGeometry, y: f64, dy: f64, dz: f64) -> Vec<u32> {
    let ic = geom.implicit_complement();
    let near = |d: f64| d.abs() <= 1e-11;
    let (above, below) = (dz >= 0.0 || near(dz), dz <= 0.0 || near(dz));
    let mut out = Vec::new();
    if y == Y[2] {
        // The outer edge: beyond it (dy > 0) is outside both.
        let inside_y = dy <= 0.0 || near(dy);
        if inside_y && below {
            out.push(0);
        }
        if inside_y && above {
            out.push(1);
        }
        if dy >= 0.0 || near(dy) {
            out.push(ic);
        }
    } else {
        // The inner edge, where the wall's -y face meets the floor's top:
        // below is the floor, above it the wall (dy > 0) or outside.
        if below {
            out.push(0);
        }
        if above && (dy >= 0.0 || near(dy)) {
            out.push(1);
        }
        if above && (dy <= 0.0 || near(dy)) {
            out.push(ic);
        }
    }
    out
}

#[test]
fn a_point_on_or_beside_an_edge_of_a_shared_face_is_placed() {
    let geom = floor_and_wall();
    let (mut unplaced, mut wrong, mut total) = (Vec::new(), Vec::new(), 0);
    for i in 0..97 {
        // Along the edge, clear of its ends.
        let x = X[0] + (X[1] - X[0]) * (f64::from(i) + 0.5) / 97.0;
        for y in [Y[1], Y[2]] {
            for dy in OFFSETS {
                for dz in OFFSETS {
                    let p = [x, y + dy, Z[1] + dz];
                    total += 1;
                    match geom.try_find_volume_from(u32::MAX, p) {
                        None => unplaced.push(p),
                        Some(v) if !fair(&geom, y, dy, dz).contains(&v) => wrong.push((p, v)),
                        Some(_) => {}
                    }
                }
            }
        }
    }
    assert!(
        unplaced.is_empty() && wrong.is_empty(),
        "of {total}: {} unplaced, e.g. {:?}; {} in a volume they are not beside, e.g. {:?}",
        unplaced.len(),
        unplaced.first(),
        wrong.len(),
        wrong.first()
    );
}
