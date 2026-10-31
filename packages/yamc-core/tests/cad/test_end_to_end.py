"""End-to-end tests: CadQuery assembly -> surface mesh via BRepMesh."""

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402


def test_single_box_surface_mesh():
    """A single box should produce 12 triangles (2 per face)."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    mesh = c2y.mesh()

    # 6 faces, each with 2 triangles = 12 triangles
    assert mesh.num_faces == 6
    assert mesh.num_solids == 1
    assert len(mesh.triangles) == 12
    # 8 unique corner vertices (though we may have duplicates at shared edges)
    assert len(mesh.vertices) >= 8
    # All vertices should be 3D
    for v in mesh.vertices:
        assert len(v) == 3
    # All triangles reference valid vertices
    n = len(mesh.vertices)
    for tri in mesh.triangles:
        assert all(0 <= idx < n for idx in tri)
    # Material tags preserved
    assert mesh.material_tags == ["steel"]


def test_two_boxes_shared_face():
    """Two touching boxes should share a face."""
    box1 = cq.Workplane("XY").box(1, 1, 1)
    box2 = cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0))
    assy = cq.Assembly()
    assy.add(box1, name="left")
    assy.add(box2, name="right")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel", "aluminum"])
    mesh = c2y.mesh()

    assert mesh.num_solids == 2
    # Should have at least 1 shared face
    assert len(mesh.shared_faces) >= 1
    # 2 boxes = 12 unique faces (6+6, minus shared), ~11 faces
    # After imprinting, the shared face counts once
    assert mesh.num_faces >= 10


def test_triangle_areas_positive():
    """All triangles should have positive area in 3D."""
    box = cq.Workplane("XY").box(2, 3, 4)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["material"])
    mesh = c2y.mesh()

    for tri in mesh.triangles:
        v0 = mesh.vertices[tri[0]]
        v1 = mesh.vertices[tri[1]]
        v2 = mesh.vertices[tri[2]]
        # Cross product magnitude
        e1 = [v1[i] - v0[i] for i in range(3)]
        e2 = [v2[i] - v0[i] for i in range(3)]
        cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
        area = 0.5 * sum(c * c for c in cross) ** 0.5
        assert area > 1e-12, f"Degenerate triangle with area {area}"


def test_smaller_tolerance_more_triangles():
    """Smaller tolerance should produce more triangles on curved surfaces."""
    cyl = cq.Workplane("XY").cylinder(2, 0.5)
    assy = cq.Assembly()
    assy.add(cyl, name="cyl")

    c2y_coarse = CadToYamc()
    c2y_coarse.add_cadquery_object(assy, material_tags=["m"])
    mesh_coarse = c2y_coarse.mesh(tolerance=1.0)

    c2y_fine = CadToYamc()
    c2y_fine.add_cadquery_object(assy, material_tags=["m"])
    mesh_fine = c2y_fine.mesh(tolerance=0.01)

    assert len(mesh_fine.triangles) > len(mesh_coarse.triangles)


def test_cylinder_mesh():
    """A cylinder has curved faces -- verify they mesh correctly."""
    cyl = cq.Workplane("XY").cylinder(2, 0.5)
    assy = cq.Assembly()
    assy.add(cyl, name="cyl")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    mesh = c2y.mesh()

    # A cylinder has 3 faces (top, bottom, lateral)
    assert mesh.num_faces == 3
    assert len(mesh.triangles) >= 3  # At minimum
    # All vertices should be near the cylinder
    for v in mesh.vertices:
        x, y, z = v
        r = (x * x + y * y) ** 0.5
        assert r <= 0.5 + 1e-6, f"Vertex outside cylinder: radius={r}"
        assert -1.0 - 1e-6 <= z <= 1.0 + 1e-6, f"Vertex outside cylinder: z={z}"


def test_sphere_mesh():
    """A sphere should mesh correctly (the UV CDT approach failed here)."""
    sphere = cq.Workplane("XY").sphere(5.0)
    assy = cq.Assembly()
    assy.add(sphere, name="sphere")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    mesh = c2y.mesh()

    assert len(mesh.triangles) > 0
    # All vertices should be approximately on the sphere surface
    for v in mesh.vertices:
        r = sum(c * c for c in v) ** 0.5
        assert abs(r - 5.0) < 0.2, f"Vertex not on sphere: radius={r}"

    # All triangles should have positive area
    for tri in mesh.triangles:
        v0 = mesh.vertices[tri[0]]
        v1 = mesh.vertices[tri[1]]
        v2 = mesh.vertices[tri[2]]
        e1 = [v1[i] - v0[i] for i in range(3)]
        e2 = [v2[i] - v0[i] for i in range(3)]
        cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
        area = 0.5 * sum(c * c for c in cross) ** 0.5
        assert area > 1e-12, f"Degenerate triangle on sphere with area {area}"
