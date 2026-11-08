//! Locks the tet fill against the boundary it is meant to fill.
//!
//! `mesh_volume` fills a closed surface with tets; the sum of their volumes
//! must equal the volume that surface encloses. Until the cut-cell clip landed
//! it did not: the inside/outside filter kept or dropped each tet WHOLESALE, so
//! a tet the boundary passes through had its volume counted entirely or not at
//! all. Upstream measured a zoo solid missing 5.4% of itself that way, with
//! nothing raised -- and for transport that is not a meshing detail, it is a
//! flux normalised over the wrong volume.
//!
//! Nothing in the suite covered this. The orientation tests
//! (`tet_orientation.rs`) check that each tet is wound correctly, which a mesh
//! that fills 94.6% of its solid passes cleanly.
//!
//! Two independent checks, because they fail differently:
//!
//! * the fill ratio itself, `tet_mesh_volume / surface_enclosed_volume`, which
//!   is the quantity a tally is normalised by;
//! * `tets_cut_by_boundary`, which upstream measured to be zero exactly when
//!   the fill error is at machine precision. It is the structural reason for
//!   the ratio, so a regression that reintroduces cut tets fails here even on
//!   a solid whose over- and under-counting happen to cancel.
//!
//! Contrast `faces_failed`, which is nonzero on healthy meshes (mesh
//! improvement subdivides boundary faces) and so cannot serve as this signal.

mod common;

use common::{cube, cylinder, icosphere, tube};
use yamm::utils::{surface_enclosed_volume, tet_mesh_volume};
use yamm::volume::{mesh_volume, VolumeInput};

/// The fill must be exact to round-off, not merely close. The defect this
/// guards against was 5.4%, but the band between "exact" and "visibly wrong" is
/// empty in practice: upstream's sweep put 146 of 175 solid measurements at
/// 1e-9 or better and every miss at 1.6e-6 or worse. A loose bar here would
/// therefore buy no robustness and would hide the whole failure mode.
const FILL_TOL: f64 = 1e-9;

fn assert_fills_its_boundary(case: &str, verts: Vec<[f64; 3]>, tris: Vec<[usize; 3]>, tel: f64) {
    let enclosed = surface_enclosed_volume(&verts, &tris);
    assert!(enclosed > 0.0, "{case}: boundary encloses no volume");

    let input = VolumeInput {
        boundary_vertices: verts,
        boundary_triangles: tris,
        target_edge_length: tel,
    };
    let out = mesh_volume(&input).unwrap_or_else(|e| panic!("{case}: mesh_volume failed: {e}"));
    assert!(!out.tetrahedra.is_empty(), "{case}: no tets emitted");

    let all = out.all_vertices(&input.boundary_vertices);
    let filled = tet_mesh_volume(&all, &out.tetrahedra);
    let err = (filled - enclosed).abs() / enclosed;
    assert!(
        err < FILL_TOL,
        "{case}: tets fill {filled:.9} of the {enclosed:.9} its boundary encloses \
         (relative error {err:.3e}, {:+.4}%); a tally normalised over this mesh \
         is wrong by the same fraction",
        (filled - enclosed) / enclosed * 100.0
    );

    assert_eq!(
        out.boundary_recovery_stats.tets_cut_by_boundary,
        0,
        "{case}: the boundary passes through {} of {} tets; each is kept or \
         dropped whole, so the fill cannot be exact by construction even when \
         the error happens to cancel",
        out.boundary_recovery_stats.tets_cut_by_boundary,
        out.tetrahedra.len()
    );
}

/// Flat-faced, spherical, cylindrical and bored solids at several scales.
///
/// The tubes are the cases that earn this test. Measured on the mesher as it
/// stood before the cut-cell clip, with everything else identical:
///
///   ri=1   ro=2 h=4 n=24 tel=0.8   fill error   -1.5e-3
///   ri=2   ro=3 h=6 n=32 tel=1.0   fill error   -1.07e-1   (-10.7%)
///   ri=0.5 ro=2 h=3 n=20 tel=0.7   fill error   -2.75e-2   (-2.75%)
///
/// all three now at 1e-14 or better. A bore gives the boundary a second sheet
/// running through the interior, so tets straddle it -- which is exactly the
/// case the old wholesale keep/drop filter could not represent. The cube,
/// icosphere and cylinder were already exact and stay here as the control:
/// they are what makes a regression legible as "the bored solids only".
#[test]
fn tets_fill_the_volume_their_boundary_encloses() {
    let (v, t) = cube(1.0);
    assert_fills_its_boundary("cube s=1 tel=0.5", v, t, 0.5);

    let (v, t) = cube(10.0);
    assert_fills_its_boundary("cube s=10 tel=2.0", v, t, 2.0);

    let (v, t) = icosphere(1.0, 2);
    assert_fills_its_boundary("icosphere r=1 subdiv=2 tel=0.4", v, t, 0.4);

    let (v, t) = icosphere(5.0, 2);
    assert_fills_its_boundary("icosphere r=5 subdiv=2 tel=1.5", v, t, 1.5);

    let (v, t) = cylinder(2.0, 6.0, 24);
    assert_fills_its_boundary("cylinder r=2 h=6 n=24 tel=1.0", v, t, 1.0);

    let (v, t) = tube(1.0, 2.0, 4.0, 24);
    assert_fills_its_boundary("tube ri=1 ro=2 h=4 n=24 tel=0.8", v, t, 0.8);

    let (v, t) = tube(2.0, 3.0, 6.0, 32);
    assert_fills_its_boundary("tube ri=2 ro=3 h=6 n=32 tel=1.0", v, t, 1.0);

    let (v, t) = tube(0.5, 2.0, 3.0, 20);
    assert_fills_its_boundary("tube ri=0.5 ro=2 h=3 n=20 tel=0.7", v, t, 0.7);
}

/// The same input must give the same mesh, run to run.
///
/// It did not. `std`'s `HashSet`/`HashMap` seed themselves per process, and the
/// cavity re-triangulation in boundary recovery fed that iteration order
/// straight into the vertex, face and tet lists it re-meshes -- so a solid
/// could land on a different tetrahedralisation each run, sometimes filling its
/// boundary exactly and sometimes leaking. A transport result that moves when
/// nothing changed is not reproducible, and a fill check is worth little if the
/// mesh under it is a different mesh each time.
///
/// Within one process this compares two calls rather than two runs, which is
/// the weaker of the two claims but the one a test can make: a container
/// iterated in seed order gives the same order twice inside a process. What it
/// does catch is any NEW nondeterminism that is not seed-derived -- thread
/// interleaving in the parallel paths, or an address-dependent order.
#[test]
fn the_tet_mesh_is_reproducible() {
    for (case, (verts, tris), tel) in [
        ("icosphere r=1 subdiv=2", icosphere(1.0, 2), 0.4),
        ("cylinder r=2 h=6 n=24", cylinder(2.0, 6.0, 24), 1.0),
        ("tube ri=1 ro=2 h=4 n=24", tube(1.0, 2.0, 4.0, 24), 0.8),
    ] {
        let input = VolumeInput {
            boundary_vertices: verts,
            boundary_triangles: tris,
            target_edge_length: tel,
        };
        let a = mesh_volume(&input).unwrap_or_else(|e| panic!("{case}: mesh_volume failed: {e}"));
        let b = mesh_volume(&input).unwrap_or_else(|e| panic!("{case}: mesh_volume failed: {e}"));

        assert_eq!(
            a.interior_vertices, b.interior_vertices,
            "{case}: interior vertices differ between two meshings of one input"
        );
        assert_eq!(
            a.tetrahedra, b.tetrahedra,
            "{case}: tet connectivity differs between two meshings of one input"
        );
    }
}
