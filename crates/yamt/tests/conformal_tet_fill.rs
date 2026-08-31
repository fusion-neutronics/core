//! A tet fill must reproduce the cell it fills, exactly.
//!
//! The headline invariant is `sum(tets) == cell`: the tets tagged to a volume
//! must sum to the volume its bounding surface encloses. That alone is a weak
//! test, though, and issue #316 is why. The `cube.msh` fixture it was traced
//! to left 8.4 percent of the cube uncovered and double-covered 8.3 percent;
//! the two nearly cancelled, the total volume looked right, and the mesh was
//! badly broken. Unstructured track-length tallies read 33 percent low for
//! months behind that cancellation.
//!
//! So the volume identity is checked alongside the topology that has to hold
//! for it to mean anything:
//!
//! * no face is shared by three or more tets (a fill cannot overlap itself),
//! * the once-used faces form a closed, consistently wound surface (the fill
//!   has no interior gap, and no cavity),
//! * that surface encloses the same volume as the tets sum to, which is the
//!   algebraic identity that only holds when every interior face cancels
//!   against its twin,
//! * and every one of those faces lies on a triangle of the CAD surface, so
//!   the fill conforms to the cell's actual boundary rather than to some
//!   coincidentally equal-volume shape.
//!
//! `nested_cylinders_tets.arrow` is the fixture that makes this worth running.
//! The flat-sided fixtures (`cube`, `two_region_tets`) are axis-aligned boxes,
//! where a tet fill conforms almost by accident. The nested cylinders share a
//! *curved* surface: surface 1 bounds both the inner rod and the outer shell,
//! and a conformal fill has to land on the same faceted cylinder from each
//! side. Both radii tessellate to regular polygons, so each volume also has a
//! closed-form expected volume that is independent of this whole pipeline.

use std::collections::HashMap;

use yamt::mesh::topology::TET_FACE_VERTICES;
use yamt::query::intersect::{signed_tet_volume, triangle_volume_contribution};
use yamt::types::Sense;
use yamt::MeshGeometry;

/// Fixtures that carry tets, with the volumes each one should measure.
///
/// The nested-cylinder volumes are the exact areas of the regular polygons OCC
/// tessellates the two cylinders into, times the height: a 12-gon at r = 1 and
/// a 15-gon at r = 2, both 4 cm tall. `0.5 * n * r^2 * sin(2*pi/n) * h`. The
/// 12-gon term is exactly 12.0 because `sin(pi/6)` is exactly 0.5.
const FIXTURES: &[(&str, &[f64])] = &[
    ("cube.arrow", &[1.0]),
    ("two_region_tets.arrow", &[0.5, 0.5]),
    (
        "nested_cylinders_tets.arrow",
        &[12.0, 36.808_397_169_118_36],
    ),
];

/// A vertex position quantised to a 1e-9 cm lattice, used as a map key.
type Point = (i64, i64, i64);

/// A face identified by its three vertex positions, sorted so the two tets
/// sharing it agree on the key regardless of winding.
type FaceKey = [Point; 3];

/// Face key to (how many tets used it, the winding the first one gave it).
type FaceUses = HashMap<FaceKey, (u32, [u32; 3])>;

/// Directed edge to how many boundary faces walked it that way.
type EdgeUses = HashMap<(Point, Point), u32>;

/// Vertex position rounded to a lattice, so faces can be matched by position.
///
/// Positional matching is required, not a convenience: the surface mesh and
/// the tet mesh occupy disjoint vertex index blocks in the Arrow file (the
/// triangles of `nested_cylinders_tets.arrow` use indices 0..54, the tets use
/// 54..250), so a shared face has two different index triples. 1e-9 cm is far
/// below any real feature and far above the f64 noise in a coordinate that has
/// been through a rotation.
fn key(v: [f64; 3]) -> Point {
    let q = |x: f64| (x * 1e9).round() as i64;
    (q(v[0]), q(v[1]), q(v[2]))
}

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

/// Is `p` on the triangle `a b c`, within `tol` of its plane and inside its
/// edges?
fn point_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3], tol: f64) -> bool {
    let n = cross(sub(b, a), sub(c, a));
    let area2 = dot(n, n).sqrt();
    if area2 < 1e-15 {
        return false;
    }
    if dot(sub(p, a), n).abs() / area2 > tol {
        return false;
    }
    // Same-sign edge tests, scaled by the triangle size so the tolerance is
    // a length rather than an area.
    let e0 = dot(cross(sub(b, a), sub(p, a)), n);
    let e1 = dot(cross(sub(c, b), sub(p, b)), n);
    let e2 = dot(cross(sub(a, c), sub(p, c)), n);
    let slack = -tol * area2;
    e0 >= slack && e1 >= slack && e2 >= slack
}

#[test]
fn tet_fill_conforms_to_the_cell_it_fills() {
    for (name, expected_volumes) in FIXTURES {
        let path = format!("tests/data/{name}");
        let geom = MeshGeometry::from_arrow(std::path::Path::new(&path))
            .unwrap_or_else(|e| panic!("{name} should load: {e}"));
        let topo = &geom.topology;

        assert!(!topo.tetrahedra.is_empty(), "{name} has no tets");
        assert_eq!(
            topo.num_volumes as usize,
            expected_volumes.len(),
            "{name} volume count changed"
        );

        for vol in 0..topo.num_volumes {
            let vertex = |i: u32| topo.vertices[i as usize];

            // -- The tets of this volume --------------------------------
            let range = &topo.volume_tet_ranges[vol as usize];
            let tets: Vec<[u32; 4]> = (range.start..range.end)
                .map(|i| topo.tetrahedra[topo.volume_tet_indices[i as usize] as usize])
                .collect();
            assert!(!tets.is_empty(), "{name} volume {vol} has no tets");

            // -- Face bookkeeping ---------------------------------------
            // Key each face by its sorted vertex POSITIONS, but keep the
            // winding the owning tet gave it. A face interior to the fill is
            // seen twice, once from each side and with opposite winding, so
            // the two cancel in the divergence sum below. A face used once is
            // on the boundary of the fill.
            let mut faces: FaceUses = HashMap::new();
            for tet in &tets {
                for face in TET_FACE_VERTICES.iter() {
                    let tri = [tet[face[0]], tet[face[1]], tet[face[2]]];
                    let mut k = [
                        key(vertex(tri[0])),
                        key(vertex(tri[1])),
                        key(vertex(tri[2])),
                    ];
                    k.sort();
                    let entry = faces.entry(k).or_insert((0, tri));
                    entry.0 += 1;
                    assert!(
                        entry.0 <= 2,
                        "{name} volume {vol}: face {k:?} is shared by {} tets, so the \
                         fill overlaps itself",
                        entry.0
                    );
                }
            }

            let boundary: Vec<[u32; 3]> = faces
                .values()
                .filter(|(count, _)| *count == 1)
                .map(|(_, tri)| *tri)
                .collect();
            assert!(
                !boundary.is_empty(),
                "{name} volume {vol}: the tet fill has no boundary at all"
            );

            // -- The boundary of the fill is closed and consistently wound --
            // Every directed edge appears exactly once, and its reverse
            // exists. An interior gap would open a second boundary component
            // (still closed, but caught by the volume identity below); a
            // winding flip or a hole shows up right here.
            let mut directed: EdgeUses = HashMap::new();
            for tri in &boundary {
                let p = [
                    key(vertex(tri[0])),
                    key(vertex(tri[1])),
                    key(vertex(tri[2])),
                ];
                for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                    *directed.entry((a, b)).or_insert(0) += 1;
                }
            }
            for (&(a, b), &count) in &directed {
                assert_eq!(
                    count, 1,
                    "{name} volume {vol}: directed edge {a:?} -> {b:?} is used {count} \
                     times on the fill boundary, so two faces are wound against each other"
                );
                assert!(
                    directed.contains_key(&(b, a)),
                    "{name} volume {vol}: directed edge {a:?} -> {b:?} has no opposite, \
                     so the tet fill is not closed"
                );
            }

            // -- sum(tets) == cell --------------------------------------
            // Three independent numbers that all have to agree: what the tets
            // sum to, what their own boundary encloses, and what the CAD
            // surface encloses. Volumes here are order 1 to 40 cm3 and every
            // coordinate is exact in f64, so the tolerance is tight.
            let mut tet_sum = 0.0;
            for (i, tet) in tets.iter().enumerate() {
                let signed = signed_tet_volume(
                    vertex(tet[0]),
                    vertex(tet[1]),
                    vertex(tet[2]),
                    vertex(tet[3]),
                );
                assert!(
                    signed > 0.0,
                    "{name} volume {vol} tet {i}: signed volume {signed} is not positive"
                );
                tet_sum += signed;
            }

            let boundary_volume: f64 = boundary
                .iter()
                .map(|t| triangle_volume_contribution(vertex(t[0]), vertex(t[1]), vertex(t[2])))
                .sum::<f64>()
                / 6.0;

            let cell_volume = topo.volume_measures[vol as usize];
            let tol = 1e-9 * cell_volume.max(1.0);

            assert!(
                (tet_sum - boundary_volume).abs() < tol,
                "{name} volume {vol}: tets sum to {tet_sum} but their own boundary \
                 encloses {boundary_volume}, so interior faces did not cancel"
            );
            assert!(
                (tet_sum - cell_volume).abs() < tol,
                "{name} volume {vol}: tets fill {tet_sum} but the cell measures \
                 {cell_volume} (difference {:.3e})",
                tet_sum - cell_volume
            );
            assert!(
                (cell_volume - expected_volumes[vol as usize]).abs() < tol,
                "{name} volume {vol}: cell measures {cell_volume}, expected {}",
                expected_volumes[vol as usize]
            );

            // -- The fill conforms to the CAD surface, not just its volume --
            // Equal volume is not conformity: a shape of the right size in the
            // wrong place passes every check above. The tet mesher refines the
            // CAD triangles (the inner rod's 44 surface triangles carry 92 tet
            // faces), so this is containment, not a one-to-one match.
            let mut cad: Vec<[u32; 3]> = Vec::new();
            for &(surf_id, sense) in &topo.volume_surfaces[vol as usize] {
                let r = &topo.surface_tri_ranges[surf_id as usize];
                for i in r.start..r.end {
                    let t = topo.triangles[topo.surface_tri_indices[i as usize] as usize];
                    cad.push(match sense {
                        Sense::Forward => t,
                        Sense::Reverse => [t[0], t[2], t[1]],
                    });
                }
            }
            assert!(
                !cad.is_empty(),
                "{name} volume {vol} has no bounding CAD triangles"
            );

            for tri in &boundary {
                let centroid = {
                    let (a, b, c) = (vertex(tri[0]), vertex(tri[1]), vertex(tri[2]));
                    [
                        (a[0] + b[0] + c[0]) / 3.0,
                        (a[1] + b[1] + c[1]) / 3.0,
                        (a[2] + b[2] + c[2]) / 3.0,
                    ]
                };
                let on_surface = cad.iter().any(|s| {
                    point_on_triangle(centroid, vertex(s[0]), vertex(s[1]), vertex(s[2]), 1e-9)
                });
                assert!(
                    on_surface,
                    "{name} volume {vol}: a tet-fill boundary face centred at \
                     {centroid:?} does not lie on any bounding CAD triangle, so the \
                     fill does not conform to the cell"
                );
            }
        }
    }
}

/// The curved fixture has to stay curved.
///
/// Every other tet fixture is an axis-aligned box, where conformity is nearly
/// automatic. If a regeneration ever flattened the cylinders (a tessellation
/// change, a wrong builder), the test above would still pass and would quietly
/// stop testing the thing it exists for. Pin the geometry that makes it a
/// curved case: the interface bounds both volumes, and it is a 12-sided prism
/// wall, so it presents 12 distinct facet normals rather than one.
#[test]
fn nested_cylinders_interface_is_curved_and_shared() {
    let path = std::path::Path::new("tests/data/nested_cylinders_tets.arrow");
    let geom = MeshGeometry::from_arrow(path).expect("fixture should load");
    let topo = &geom.topology;

    // Both sides have to be real volumes. Every exterior surface also has two
    // sides once the topology is built, because the open side is filled in
    // with the implicit complement, whose id is `num_volumes`.
    let shared: Vec<usize> = (0..topo.num_surfaces as usize)
        .filter(|&s| {
            let (fwd, rev) = topo.surface_volumes[s];
            matches!((fwd, rev), (Some(f), Some(r))
                if f < topo.num_volumes && r < topo.num_volumes)
        })
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "expected exactly one surface bounding two volumes, found {shared:?}"
    );

    let surf = shared[0];
    let r = &topo.surface_tri_ranges[surf];
    let mut normals = std::collections::HashSet::new();
    for i in r.start..r.end {
        let t = topo.triangles[topo.surface_tri_indices[i as usize] as usize];
        let (a, b, c) = (
            topo.vertices[t[0] as usize],
            topo.vertices[t[1] as usize],
            topo.vertices[t[2] as usize],
        );
        let n = cross(sub(b, a), sub(c, a));
        let len = dot(n, n).sqrt();
        assert!(len > 1e-15, "degenerate triangle on the shared surface");
        // Round the unit normal so the two triangles of one prism facet agree.
        normals.insert(key([n[0] / len, n[1] / len, n[2] / len]));
    }
    assert_eq!(
        normals.len(),
        12,
        "the shared surface should be a 12-sided prism wall, got {} distinct \
         facet normals; if this is a deliberate retessellation, update FIXTURES \
         in conformal_tet_fill.rs and the expected volumes with it",
        normals.len()
    );
}
