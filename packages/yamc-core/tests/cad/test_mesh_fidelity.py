"""Tests for the mesh fidelity gate (issue #252).

The gate compares the assembled surface mesh against exact OCC face areas
and solid volumes, so a topologically consistent but geometrically wrong
mesh raises at meshing time instead of silently transporting particles
through a wrong model.
"""
import dataclasses

import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import check_mesh_fidelity  # noqa: E402


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
