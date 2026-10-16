"""Tests for the mesh fidelity gate.

The gate compares the assembled surface mesh against exact OCC face areas
and solid volumes, so a topologically consistent but geometrically wrong
mesh raises at meshing time instead of silently transporting particles
through a wrong model.
"""
import dataclasses

import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import MeshFidelityError, check_mesh_fidelity  # noqa: E402


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
