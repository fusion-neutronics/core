"""Closed doubly-periodic b-spline faces must route to BRepMesh.

These faces (full-torus shells, common in imported reactor CAD) are the
known scene-CDT gap: meshed from their seam wires they come out as a
coarse twisted shell that ignores the deflection tolerance (a metre-scale
radial build produced 2.6k triangles with 12 m edges at a 0.1 cm
tolerance request). Routed to BRepMesh with interior deflection control
they mesh cleanly and track the tolerance.
"""
from collections import defaultdict

import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad.mesher import _is_closed_doubly_periodic  # noqa: E402


def _nurbs_torus(major=100.0, minor=30.0):
    """A full torus as a closed doubly-periodic b-spline solid."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

    torus = cq.Solid.makeTorus(major, minor)
    conv = BRepBuilderAPI_NurbsConvert(torus.wrapped, True)
    return cq.Shape.cast(conv.Shape())


def _mesh(solid, tolerance, angular_tolerance):
    assy = cq.Assembly()
    assy.add(solid, name="shell")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["mat_shell"])
    return c2y.mesh(tolerance=tolerance, angular_tolerance=angular_tolerance)


def _census(mesh):
    """(n_triangles, n_edges_not_shared_by_exactly_2_triangles)."""
    edges = defaultdict(int)
    for t in mesh.triangles:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(e), max(e))] += 1
    return len(mesh.triangles), sum(1 for c in edges.values() if c != 2)


def test_nurbs_torus_faces_detected():
    """The b-spline torus faces qualify as closed doubly-periodic."""
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    solid = _nurbs_torus()
    exp = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
    n = 0
    while exp.More():
        assert _is_closed_doubly_periodic(TopoDS.Face(exp.Current()))
        n += 1
        exp.Next()
    assert n >= 1


def test_nurbs_torus_mesh_closed_and_refined():
    """Meshed b-spline torus must be a closed manifold that tracks the
    tolerance, not the fixed coarse shell the scene CDT used to emit."""
    solid = _nurbs_torus()
    coarse = _mesh(solid, tolerance=1.0, angular_tolerance=0.4)
    n_coarse, bad_coarse = _census(coarse)
    assert bad_coarse == 0, f"{bad_coarse} edges not shared by exactly 2 tris"
    assert n_coarse > 1000, f"suspiciously coarse: {n_coarse} triangles"

    fine = _mesh(solid, tolerance=0.2, angular_tolerance=0.2)
    n_fine, bad_fine = _census(fine)
    assert bad_fine == 0
    assert n_fine > n_coarse, f"tolerance had no effect: {n_coarse} -> {n_fine}"
