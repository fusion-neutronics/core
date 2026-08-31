//! Locks the positive-orientation invariant of every tet `yamm` emits.
//!
//! Downstream transport (`yamt`) reads tet face normals off a fixed face table,
//! `yamt::mesh::topology::TET_FACE_VERTICES`, whose orderings yield OUTWARD
//! normals only for a positively oriented tet. The element walk picks its exit
//! face with `dot(direction, normal) > 0`, so an inverted tet makes it choose an
//! ENTRY face: the walk hops backwards or leaves the mesh early, and
//! unstructured track-length tallies read far too low (issue #316 measured
//! -33%). `yamt` now refuses a mesh with an inverted tet outright, so these
//! tests keep the source honest and a mesher regression fails where it is
//! introduced rather than at load.
//!
//! This binary covers the default Delaunay-plus-centroid-filter pipeline and the
//! boundary-only pipeline. The third emission path, the exact conforming carve,
//! is selected by a process-global environment variable and so lives in its own
//! test binary (`tet_orientation_conforming.rs`).

mod common;

use common::{
    assert_positively_oriented, cube, cylinder, first_inward_face, icosphere, signed_tet_volume,
};
use yamm::volume::VolumeOutput;

/// Default pipeline: Delaunay fill plus centroid filter, over flat-faced,
/// spherical and cylindrical solids at several scales and resolutions.
#[test]
fn emitted_tets_are_positively_oriented() {
    let (v, t) = cube(1.0);
    assert_positively_oriented("cube s=1 tel=0.5", v, t, 0.5);

    let (v, t) = cube(5.0);
    assert_positively_oriented("cube s=5 tel=1.5", v, t, 1.5);

    let (v, t) = cube(10.0);
    assert_positively_oriented("cube s=10 tel=2.0", v, t, 2.0);

    let (v, t) = icosphere(1.0, 2);
    assert_positively_oriented("sphere r=1 tel=0.35", v, t, 0.35);

    let (v, t) = icosphere(10.0, 3);
    assert_positively_oriented("sphere r=10 tel=2.0", v, t, 2.0);

    let (v, t) = cylinder(1.0, 4.0, 24);
    assert_positively_oriented("cylinder r=1 h=4 tel=0.5", v, t, 0.5);

    // Thin disc: an aspect ratio that pushes the filter towards flat slivers.
    let (v, t) = cylinder(50.0, 5.0, 48);
    assert_positively_oriented("cylinder r=50 h=5 tel=6.0", v, t, 6.0);
}

/// Boundary-only pipeline: a target edge length coarser than the solid seeds no
/// interior points, so `mesh_boundary_only` (a separate orientation pass and a
/// separate degenerate-tet drop) emits the mesh.
#[test]
fn boundary_only_tets_are_positively_oriented() {
    let (v, t) = cube(1.0);
    assert_positively_oriented("cube s=1 tel=10 (boundary-only)", v, t, 10.0);
}

/// The guard is not vacuous: it accepts a positively oriented tet and rejects
/// the same tet with two vertices swapped (the inversion `yamt` refuses to
/// load).
#[test]
fn guard_rejects_an_inverted_tet() {
    let boundary = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    assert!(signed_tet_volume(boundary[0], boundary[1], boundary[2], boundary[3]) > 0.0);

    let good = VolumeOutput {
        interior_vertices: vec![],
        tetrahedra: vec![[0, 1, 2, 3]],
        boundary_recovery_stats: Default::default(),
    };
    assert!(good.first_non_positive_tet(&boundary).is_none());

    let inverted = VolumeOutput {
        tetrahedra: vec![[0, 1, 3, 2]],
        ..good
    };
    let (idx, signed) = inverted
        .first_non_positive_tet(&boundary)
        .expect("inverted tet must be reported");
    assert_eq!(idx, 0);
    assert!(
        signed < 0.0,
        "expected a negative signed volume, got {signed}"
    );
}

/// The face-normal check is not vacuous either, and the copied face table is
/// the right one: a positively oriented reference tet has four outward normals,
/// and swapping two vertices turns them inward.
#[test]
fn face_normals_follow_orientation() {
    let v = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    assert!(signed_tet_volume(v[0], v[1], v[2], v[3]) > 0.0);
    assert_eq!(first_inward_face(&v, [0, 1, 2, 3]), None);
    assert_eq!(first_inward_face(&v, [0, 1, 3, 2]), Some(0));
}

/// `first_non_positive_tet` addresses interior vertices through the same
/// `boundary ++ interior` indexing the tets use, so it must see a bad tet whose
/// apex is an interior vertex too (an off-by-one there would silently pass every
/// real mesh, where most tets touch interior points).
#[test]
fn guard_indexes_interior_vertices() {
    let boundary = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let out = VolumeOutput {
        interior_vertices: vec![[0.0, 0.0, -1.0]],
        // Apex below the base plane: negatively oriented.
        tetrahedra: vec![[0, 1, 2, 3]],
        boundary_recovery_stats: Default::default(),
    };
    assert_eq!(out.vertex(&boundary, 3), [0.0, 0.0, -1.0]);
    let (idx, signed) = out
        .first_non_positive_tet(&boundary)
        .expect("inverted tet with an interior apex must be reported");
    assert_eq!(idx, 0);
    assert!(signed < 0.0);
}
