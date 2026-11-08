"""Tet meshing two solids that share a curved interface meshed by the CDT.

A 30 mm box with a 10 mm sphere cut out of it and a lofted notch cut into the
other side, plus the sphere sitting in its hole. Both solids are tet meshed.

WHAT THIS PINS. With the pinched-link-polygon guard in `ring_apex_cycle`
removed, this geometry aborts the volume mesh with a panic out of the tet
mesher::

    orient_3d_sos: all projections degenerate => duplicate vertex
        (ids 5554,5554,5554,81)

The ring reducer walked a PINCHED link polygon (the same finite apex at two
positions of the cycle), and `flip_ring_general`'s polygon DP then formed a
"triangle" with a repeated vertex. No orientation predicate can sign a repeated
point. Measured on this geometry: 61 pinched rings, then the panic.

WHY THE NOTCH IS HERE. The notch is what makes the case reproduce. Its loft
gives the box a b-spline face, so the box is no longer "all primitive" and
`_route_faces` sends every face it owns, including the shared spherical
interface, to the scene CDT instead of OCC BRepMesh. That CDT boundary is what
drives the coplanar-region path in the volume mesher. Without the notch both
solids are all-primitive, BRepMesh produces a different boundary triangulation,
and no ring is ever pinched.

WHY THE ASSERTIONS ARE STRUCTURAL. The tets involved in the corruption all have
zero volume, so it is exactly volume neutral: the carve's volume gate and every
volume-based assertion stay green while the complex is already invalid at the
CELL level. `test_shared_interface_adjacency` covers two planar touching cuboids
and checks face pairing; nothing checked the cells of a shared curved interface.

WHY `fidelity_check=False`. Routing that shared sphere face to the CDT meshes it
about 15.7% off CAD, and tightening `tolerance` to 0.001 does not move it. That
is a real and separate defect in the CDT's handling of a primitive face pulled
in by a non-primitive co-owner; it is not what this test is about, and gating on
it here would only hide the volume mesher behind an unrelated failure. A CAD
volume assertion is omitted for the same reason: it would pin the surface
mesher's error, not the tet mesher's.
"""

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402

BOX_SIDE = 30.0
SPHERE_R = 10.0
SPHERE_X = 20.0
NOTCH_X = -20.0
TARGET_EDGE_LENGTH = 2.0


@pytest.fixture(scope="module")
def shared_interface_tets():
    """Tet mesh both solids, keyed by solid id.

    Module scoped: the volume mesh takes about 25 s (481k tets across the two
    solids), and every assertion below reads the same one.
    """
    notch = (
        cq.Workplane("XY")
        .moveTo(NOTCH_X, 0)
        .rect(12, 12)
        .workplane(offset=14)
        .moveTo(NOTCH_X, 0)
        .circle(4)
        .loft()
        .translate((0, 0, -7))
    )
    sphere = cq.Workplane("XY").moveTo(SPHERE_X, 0).sphere(SPHERE_R)
    box = cq.Workplane("XY").box(BOX_SIDE, BOX_SIDE, BOX_SIDE).cut(sphere).cut(notch)

    assembly = cq.Assembly()
    assembly.add(box, name="box")
    assembly.add(sphere, name="sphere")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["box_mat", "sphere_mat"])
    c2y.mesh_surfaces(tolerance=0.01, angular_tolerance=0.2, fidelity_check=False)
    return c2y.mesh_volumes(tet_target_edge_length=TARGET_EDGE_LENGTH)


def _tet_volumes(vertices, tetrahedra):
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(tetrahedra, dtype=int)
    a, b, c, d = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]], v[t[:, 3]]
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def test_both_solids_are_tet_meshed(shared_interface_tets):
    """The mesh completes at all. This alone is the regression."""
    assert sorted(shared_interface_tets) == [1, 2]
    for sid, (_verts, tets) in shared_interface_tets.items():
        assert len(tets) > 0, f"solid {sid}: no tets"


@pytest.mark.parametrize("sid", [1, 2])
def test_no_duplicate_tets(shared_interface_tets, sid):
    """Two tets on the same four vertices are coincident.

    The face between them is then owned by four tets and the complex is no
    longer a valid simplicial complex. This is the direct signature of the
    replayed inverse-flip pair that pinches the rings.
    """
    _verts, tets = shared_interface_tets[sid]
    keys = [tuple(sorted(t)) for t in tets]
    duplicates = len(keys) - len(set(keys))
    assert duplicates == 0, f"solid {sid}: {duplicates} duplicate tets"


@pytest.mark.parametrize("sid", [1, 2])
def test_no_tet_has_a_repeated_vertex(shared_interface_tets, sid):
    """A tet with a repeated vertex is what the orientation predicate chokes on."""
    _verts, tets = shared_interface_tets[sid]
    bad = [t for t in tets if len(set(t)) != 4]
    assert not bad, (
        f"solid {sid}: {len(bad)} tets with a repeated vertex, e.g. {bad[0]}"
    )


@pytest.mark.parametrize("sid", [1, 2])
def test_no_zero_volume_tets(shared_interface_tets, sid):
    """The flat stack tets must never reach the output mesh."""
    verts, tets = shared_interface_tets[sid]
    vols = np.abs(_tet_volumes(verts, tets))
    flat = int((vols <= 1e-12).sum())
    assert flat == 0, f"solid {sid}: {flat} zero-volume tets"
