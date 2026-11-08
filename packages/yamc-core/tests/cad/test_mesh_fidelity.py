"""Tests for the mesh fidelity gate.

The gate compares the assembled surface mesh against the face areas and
solid volumes OCC integrates from the CAD, so a topologically consistent but
geometrically wrong mesh raises at meshing time instead of silently
transporting particles through a wrong model.
"""
import dataclasses

import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import (  # noqa: E402
    MeshFidelityError, _occ_face_area, _occ_signed_volume, check_mesh_fidelity,
)


def _meshed_spheres():
    """Two touching spheres, meshed with the gate bypassed."""
    left = cq.Workplane().sphere(10)
    right = cq.Workplane().sphere(10).translate((20, 0, 0))
    assy = cq.Assembly()
    assy.add(left, name="left")
    assy.add(right, name="right")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["left", "right"])
    mesh = c2y.mesh(tolerance=0.1, angular_tolerance=0.2, fidelity_check=False)
    return mesh, c2y._processed


def test_faithful_mesh_passes():
    """A correctly meshed model produces no fidelity problems."""
    mesh, pa = _meshed_spheres()
    assert check_mesh_fidelity(mesh, pa, tolerance=0.1) == []


def test_default_gate_active_on_mesh():
    """mesh() runs the gate by default and passes on good geometry."""
    assy = cq.Assembly()
    assy.add(cq.Workplane().box(10, 10, 10), name="cube")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["cube"])
    mesh = c2y.mesh()
    assert len(mesh.triangles) > 0


def test_distorted_mesh_flagged():
    """Shrinking the mesh 30% (wrong areas and volumes, topology intact)
    must be reported for every face and solid."""
    mesh, pa = _meshed_spheres()
    shrunk = dataclasses.replace(
        mesh, vertices=[[0.7 * x, 0.7 * y, 0.7 * z] for x, y, z in mesh.vertices]
    )
    problems = check_mesh_fidelity(shrunk, pa, tolerance=0.1)
    flagged = " ".join(problems)
    for fid in pa.face_to_occ:
        assert f"face {fid}:" in flagged
    for sid in mesh.solid_faces:
        assert f"solid {sid}:" in flagged


def test_locally_collapsed_solid_flagged():
    """Collapsing one solid's vertices onto its centroid (the other solid
    untouched) is caught, and only that solid's volume is reported."""
    mesh, pa = _meshed_spheres()
    verts = [list(v) for v in mesh.vertices]
    tris_of_s1 = [
        t for t, fid in zip(mesh.triangles, mesh.triangle_face_ids)
        if fid in mesh.solid_faces[1]
    ]
    idx = sorted({i for t in tris_of_s1 for i in t})
    cx = sum(verts[i][0] for i in idx) / len(idx)
    cy = sum(verts[i][1] for i in idx) / len(idx)
    cz = sum(verts[i][2] for i in idx) / len(idx)
    for i in idx:
        verts[i] = [
            cx + 0.3 * (verts[i][0] - cx),
            cy + 0.3 * (verts[i][1] - cy),
            cz + 0.3 * (verts[i][2] - cz),
        ]
    crushed = dataclasses.replace(mesh, vertices=verts)
    problems = check_mesh_fidelity(crushed, pa, tolerance=0.1)
    flagged = " ".join(problems)
    assert "solid 1:" in flagged
    assert "solid 2:" not in flagged


def test_a_revolved_spline_profile_meshes_faithfully():
    """A D-shaped profile (spline outboard, straight inboard) revolved half a
    turn about the z axis, the way paramak builds tokamak components. Its
    surface of revolution used to go to the scene CDT, which meshed the solid
    49% short in volume; it now goes to BRepMesh with the rest of the solid,
    and the default fidelity gate passes."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceOfRevolution

    profile = (
        cq.Workplane("XZ")
        .moveTo(300, -200)
        .spline([(450, -120), (520, 0), (450, 120)], includeCurrent=True)
        .lineTo(300, 200)
        .close()
    )
    solid = profile.revolve(180, (0, 0, 0), (0, 1, 0))
    assert any(
        BRepAdaptor_Surface(f.wrapped).GetType() == GeomAbs_SurfaceOfRevolution
        for f in solid.val().Faces()
    )
    assy = cq.Assembly()
    assy.add(solid, name="blanket")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["blanket"])
    c2y.mesh(tolerance=0.5)


def test_a_revolved_solid_touching_a_spline_extrusion_does_not_crack():
    """A revolved D-shape whose end face is shared with a solid extruded from
    it (a surface of extrusion, not routed to BRepMesh): the revolved solid
    must not mix meshers, which would crack it along the shared face's edges
    unseen by the gate. It is meshed whole on one side: either the mesh is
    watertight or the gate refuses it."""
    from collections import Counter

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    profile = (
        cq.Workplane("XZ")
        .moveTo(300, -200)
        .spline([(450, -120), (520, 0), (450, 120)], includeCurrent=True)
        .lineTo(300, 200)
        .close()
    )
    revolved = profile.revolve(180, (0, 0, 0), (0, 1, 0)).val()
    end = next(
        f for f in revolved.Faces()
        if BRepAdaptor_Surface(f.wrapped).GetType() == GeomAbs_Plane
    )
    extruded = cq.Solid.extrudeLinear(end, end.normalAt() * 100)
    assy = cq.Assembly()
    assy.add(revolved, name="revolved")
    assy.add(extruded, name="extruded")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["revolved", "extruded"])
    try:
        mesh = c2y.mesh(tolerance=0.5)
    except MeshFidelityError:
        return
    faces = set(mesh.solid_faces[1])
    uses = Counter()
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        if fid in faces:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                uses[(min(a, b), max(a, b))] += 1
    assert not [e for e, n in uses.items() if n != 2], "open edges in the revolved solid"


def _bspline_prism():
    """The surface z = 10 sin(6 pi x/100) cos(6 pi y/100) over 100 x 100,
    fitted through 60 x 60 points (OCC makes it cubic with a knot at every
    point, dense like the faces of a stellarator coil) and extruded 50 along
    z. The fit runs along the edges of the square, so the prism holds
    100 x 100 x 50 and each of its four walls is 100 x 50 (issue #421)."""
    import math

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt, gp_Vec

    n, k = 60, 6
    points = TColgp_Array2OfPnt(1, n, 1, n)
    for i in range(n):
        for j in range(n):
            x, y = 100 * i / (n - 1), 100 * j / (n - 1)
            z = 10 * math.sin(k * math.pi * x / 100) * math.cos(k * math.pi * y / 100)
            points.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))
    surface = GeomAPI_PointsToBSplineSurface(points, 8, 8, GeomAbs_C2, 1e-3).Surface()
    face = BRepBuilderAPI_MakeFace(surface, 1e-6).Face()
    return cq.Shape.cast(BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, 50)).Shape())


def _spline_face_area(face):
    """Area of a face that is a whole B-spline surface, integrated here:
    |S_u x S_v| by 4-point Gauss-Legendre on every knot span, within 1e-8 of
    the converged value on the prism's top face."""
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.gp import gp_Pnt, gp_Vec

    surface = BRepAdaptor_Surface(face)
    spline = surface.BSpline()
    x, w = np.polynomial.legendre.leggauss(4)

    def nodes(knots):
        return [
            (0.5 * (a + b + (b - a) * xi), 0.5 * (b - a) * wi)
            for a, b in zip(knots, knots[1:])
            for xi, wi in zip(x, w)
        ]

    us = nodes([spline.UKnot(i) for i in range(1, spline.NbUKnots() + 1)])
    vs = nodes([spline.VKnot(i) for i in range(1, spline.NbVKnots() + 1)])
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    area = 0.0
    for u, wu in us:
        for v, wv in vs:
            surface.D1(u, v, p, du, dv)
            area += wu * wv * du.Crossed(dv).Magnitude()
    return area


def test_the_cad_volume_of_a_dense_spline_solid_converges():
    """OCC's fixed Gauss rule puts the prism 2.4% low and its adaptive form
    0.6% high; Gauss-Kronrod on every knot span gets it."""
    assert _occ_signed_volume(_bspline_prism()) == pytest.approx(500000, rel=1e-5)


def test_the_cad_areas_of_swept_spline_faces_converge():
    """The prism's walls are splines swept along z. OCC's fixed Gauss rule
    puts them 3% and 4% low, and its adaptive form, which does not split a
    swept face at its knots, puts two of them 1.8% high at every precision
    from 1e-5 to 1e-7."""
    walls = [f for f in _bspline_prism().Faces() if f.geomType() == "EXTRUSION"]
    assert [_occ_face_area(f.wrapped) for f in walls] == pytest.approx(
        [5000.0] * 4, rel=1e-6
    )


def test_the_cad_area_of_a_dense_spline_face_converges():
    """OCC's fixed Gauss rule puts the prism's top face 4e-4 low."""
    top = max(_bspline_prism().Faces(), key=lambda f: f.Center().z)
    assert _occ_face_area(top.wrapped) == pytest.approx(
        _spline_face_area(top.wrapped), rel=1e-5
    )


def test_the_cad_volume_keeps_its_sign_on_a_reversed_or_mirrored_solid():
    """The gate tells a solid that is inside out in the CAD by its negative
    volume, and Gauss-Kronrod keeps that sign."""
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
    from OCP.TopLoc import TopLoc_Location

    prism = _bspline_prism().wrapped
    mirror = gp_Trsf()
    mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    mirrored = prism.Moved(TopLoc_Location(mirror))
    assert _occ_signed_volume(prism.Reversed()) == pytest.approx(-500000, rel=1e-5)
    assert _occ_signed_volume(mirrored) == pytest.approx(-500000, rel=1e-5)
    assy = cq.Assembly()
    assy.add(cq.Solid(prism.Reversed()), name="bump")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["bump"])
    mesh = c2y.mesh(tolerance=0.05, fidelity_check=False)
    problems = check_mesh_fidelity(mesh, c2y._processed, tolerance=0.05)
    assert len(problems) == 1 and "inside out in the CAD" in problems[0], problems


def test_a_dense_spline_solid_passes_the_gate():
    """A correct mesh of the prism is not refused for the CAD's own
    integration error. At this deflection the gate allows 2% in every face
    area and in the volume, and OCC's default integration put the walls 3%
    and 4% low and the volume 2.4% low (issue #421)."""
    assy = cq.Assembly()
    assy.add(_bspline_prism(), name="bump")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["bump"])
    c2y.mesh(tolerance=0.05)


def test_the_cad_volume_of_a_tilted_half_cylinder_comes_back_at_once():
    """Every vertex of a half cylinder lies in its flat faces, and OCC's own
    Gauss-Kronrod integrates about the centroid of the vertices: tilted off
    the axes, a flat face spans a cone of no volume to it, round-off keeps
    that from being exactly zero, and the relative precision it chases is
    never met. It came back right, after 20 to 40 seconds."""
    import math
    import time

    half = cq.Solid.makeCylinder(10, 20, angleDegrees=180).rotate(
        cq.Vector(0, 0, 0), cq.Vector(1, 2, 3), 17
    )
    start = time.perf_counter()
    volume = _occ_signed_volume(half)
    assert time.perf_counter() - start < 5.0
    assert volume == pytest.approx(1000 * math.pi, rel=1e-9)


def test_a_shelled_spline_solid_passes_the_gate():
    """The inner wall of a shelled spline extrusion is an offset surface,
    which OCC integrates as a single piece: its fixed Gauss rule puts the
    wall 2.1% low and its adaptive form 7.1% low, either enough to refuse a
    correct mesh."""
    import math

    angles = [2 * math.pi * i / 72 for i in range(72)]
    radii = [50 + 5 * math.cos(8 * a) for a in angles]
    profile = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
    casing = (
        cq.Workplane("XY").spline(profile, periodic=True).close()
        .extrude(50).faces(">Z").shell(-1.5)
    )
    assy = cq.Assembly()
    assy.add(casing, name="casing")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["casing"])
    c2y.mesh(tolerance=0.1)
