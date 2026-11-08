//! Shared fixtures for the tet-orientation tests.
//!
//! Compiled into every integration-test binary that declares `mod common`, so
//! each binary uses only part of it.
#![allow(dead_code)]

use yamm::volume::{mesh_volume, VolumeInput};

/// Signed tet volume, `det([v1-v0, v2-v0, v3-v0]) / 6`, in PLAIN f64.
///
/// This is deliberately a re-derivation of the consumer's own formula
/// (`yamt::query::intersect::signed_tet_volume`) rather than a call into yamm:
/// the guard inside `mesh_volume` uses Shewchuk's exact `orient3d`, so checking
/// with it here would only restate the guard. Plain f64 is both independent and
/// strictly the harder bar, since it is the arithmetic the walk actually runs.
pub fn signed_tet_volume(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> f64 {
    let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let e2 = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    (e0[0] * cx + e0[1] * cy + e0[2] * cz) / 6.0
}

/// A copy of `yamt::mesh::topology::TET_FACE_VERTICES`, the face table the
/// consumer reads normals off. Face `i` is opposite vertex `i`, and the winding
/// is chosen so the normal points AWAY from that opposite vertex, which only
/// holds when the tet is positively oriented.
///
/// Duplicated rather than imported: `yamm` is deliberately transport-neutral and
/// does not depend on `yamt`. Restating the table is also what makes the check
/// meaningful, since it pins the exact convention the walk assumes.
const TET_FACE_VERTICES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Index of the first face of `tet` whose `TET_FACE_VERTICES` normal points
/// INWARD, or `None` when all four point outward.
///
/// This is the property the element walk actually consumes, one step closer to
/// the failure than the signed volume: it picks its exit face with
/// `dot(direction, normal) > 0`, so an inward normal makes an entry face look
/// like the exit.
pub fn first_inward_face(vertices: &[[f64; 3]], tet: [usize; 4]) -> Option<usize> {
    (0..4).find(|&f| {
        let [i, j, k] = TET_FACE_VERTICES[f];
        let (a, b, c) = (vertices[tet[i]], vertices[tet[j]], vertices[tet[k]]);
        let normal = cross(sub(b, a), sub(c, a));
        // Face f is opposite vertex f: outward means the normal leans away from
        // it, so the dot product against `opposite - a` must be negative.
        dot(normal, sub(vertices[tet[f]], a)) >= 0.0
    })
}

// --------------------------------------------------------------------------
// Boundary-surface fixtures
// --------------------------------------------------------------------------

/// Axis-aligned cube of side `s` at the origin: 8 vertices, 12 triangles.
pub fn cube(s: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [s, 0.0, 0.0],
        [s, s, 0.0],
        [0.0, s, 0.0],
        [0.0, 0.0, s],
        [s, 0.0, s],
        [s, s, s],
        [0.0, s, s],
    ];
    let tris = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    (verts, tris)
}

/// Icosphere of radius `r` after `subdiv` rounds of 4-way subdivision.
pub fn icosphere(r: f64, subdiv: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut verts: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut tris: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdiv {
        let mut mid: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for tri in &tris {
            let mut m = [0usize; 3];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                m[k] = *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                    verts.push([
                        (verts[a][0] + verts[b][0]) / 2.0,
                        (verts[a][1] + verts[b][1]) / 2.0,
                        (verts[a][2] + verts[b][2]) / 2.0,
                    ]);
                    verts.len() - 1
                });
            }
            next.push([tri[0], m[0], m[2]]);
            next.push([m[0], tri[1], m[1]]);
            next.push([m[2], m[1], tri[2]]);
            next.push([m[0], m[1], m[2]]);
        }
        tris = next;
    }
    for v in &mut verts {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        *v = [v[0] / n * r, v[1] / n * r, v[2] / n * r];
    }
    (verts, tris)
}

/// Closed cylinder along +z: radius `r`, height `h`, `n` circumferential
/// segments. Curved and capped, so it exercises the sliver-prone paths the cube
/// does not.
pub fn cylinder(r: f64, h: f64, n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut verts = Vec::with_capacity(2 * n + 2);
    for z in [0.0, h] {
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            verts.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    let bottom_hub = verts.len();
    verts.push([0.0, 0.0, 0.0]);
    let top_hub = verts.len();
    verts.push([0.0, 0.0, h]);

    let mut tris = Vec::with_capacity(4 * n);
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([i, j, n + j]);
        tris.push([i, n + j, n + i]);
        tris.push([bottom_hub, j, i]);
        tris.push([top_hub, n + i, n + j]);
    }
    (verts, tris)
}

// --------------------------------------------------------------------------
// Assertion
// --------------------------------------------------------------------------

/// Mesh one solid and assert every emitted tet is positively oriented.
pub fn assert_positively_oriented(
    case: &str,
    boundary_vertices: Vec<[f64; 3]>,
    boundary_triangles: Vec<[usize; 3]>,
    target_edge_length: f64,
) {
    let input = VolumeInput {
        boundary_vertices,
        boundary_triangles,
        target_edge_length,
    };
    let out = mesh_volume(&input).unwrap_or_else(|e| panic!("{case}: mesh_volume failed: {e}"));
    assert!(!out.tetrahedra.is_empty(), "{case}: no tets emitted");

    let all = out.all_vertices(&input.boundary_vertices);
    let mut negative = 0usize;
    let mut zero = 0usize;
    let mut worst = f64::INFINITY;
    for t in &out.tetrahedra {
        let v = signed_tet_volume(all[t[0]], all[t[1]], all[t[2]], all[t[3]]);
        if v < 0.0 {
            negative += 1;
        } else if v == 0.0 {
            zero += 1;
        }
        worst = worst.min(v);
    }
    assert_eq!(
        (negative, zero),
        (0, 0),
        "{case}: {negative} inverted and {zero} zero-volume tets of {} \
         (min signed volume {worst:e}); TET_FACE_VERTICES normals point inward \
         for these and the yamt element walk will pick an entry face as its exit",
        out.tetrahedra.len()
    );

    // The public guard must agree with the independent plain-f64 verdict.
    assert!(
        out.first_non_positive_tet(&input.boundary_vertices)
            .is_none(),
        "{case}: first_non_positive_tet disagrees with the plain-f64 sign test"
    );

    // The consequence the transport walk depends on: read through the face
    // table, every face normal must point outward.
    for (i, t) in out.tetrahedra.iter().enumerate() {
        assert!(
            first_inward_face(&all, *t).is_none(),
            "{case}: tet {i} has an inward TET_FACE_VERTICES normal on face {:?}",
            first_inward_face(&all, *t)
        );
    }
}

/// Hollow cylinder (tube) along +z: inner radius `ri`, outer radius `ro`,
/// height `h`, `n` circumferential segments. Annular caps, no hub vertex.
///
/// A solid with a bore is the shape that broke the old wholesale
/// inside/outside filter: the inner wall gives the boundary a second sheet
/// running through the interior, so tets straddle it and get counted or
/// discarded whole. Upstream measured its CAD equivalent (NestedCylinder)
/// under-filling by 0.2595%.
pub fn tube(ri: f64, ro: f64, h: f64, n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut verts = Vec::with_capacity(4 * n);
    // Rings, in order: outer-bottom, inner-bottom, outer-top, inner-top.
    for (r, z) in [(ro, 0.0), (ri, 0.0), (ro, h), (ri, h)] {
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            verts.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    let (ob, ib, ot, it) = (0, n, 2 * n, 3 * n);

    let mut tris = Vec::with_capacity(8 * n);
    for i in 0..n {
        let j = (i + 1) % n;
        // Outer wall, normals out.
        tris.push([ob + i, ob + j, ot + j]);
        tris.push([ob + i, ot + j, ot + i]);
        // Inner wall, normals in toward the axis (out of the solid).
        tris.push([ib + i, it + j, ib + j]);
        tris.push([ib + i, it + i, it + j]);
        // Bottom annulus, normals -z.
        tris.push([ob + i, ib + j, ob + j]);
        tris.push([ob + i, ib + i, ib + j]);
        // Top annulus, normals +z.
        tris.push([ot + i, ot + j, it + j]);
        tris.push([ot + i, it + j, it + i]);
    }
    (verts, tris)
}
