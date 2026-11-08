"""Tests for Arrow IPC mesh export."""

import json
import os

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc, mesh_to_arrow  # noqa: E402


def test_to_arrow_basic(tmp_path):
    """Export a box mesh to Arrow and verify file exists."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    c2y.mesh_surfaces()

    path = str(tmp_path / "box.arrow")
    c2y.to_arrow(path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_to_arrow_two_materials(tmp_path):
    """Export two-box assembly with different materials."""
    box1 = cq.Workplane("XY").box(1, 1, 1)
    box2 = cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0))
    assy = cq.Assembly()
    assy.add(box1, name="left")
    assy.add(box2, name="right")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel", "moderator"])
    c2y.mesh_surfaces()

    path = str(tmp_path / "two_boxes.arrow")
    c2y.to_arrow(path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def _physical_groups(path):
    """The ``yamc.physical_groups`` metadata of a written Arrow mesh."""
    ipc = pytest.importorskip("pyarrow.ipc")
    metadata = ipc.open_file(path).schema.metadata
    return json.loads(metadata[b"yamc.physical_groups"])


def _two_box_converter():
    """Two imprinted unit boxes sharing an internal face at x = 0.5."""
    box1 = cq.Workplane("XY").box(1, 1, 1)
    box2 = cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0))
    assy = cq.Assembly()
    assy.add(box1, name="left")
    assy.add(box2, name="right")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel", "moderator"])
    return c2y, c2y.mesh_surfaces()


def test_exterior_surface_ids_excludes_the_shared_face():
    """The imprinted interface belongs to two solids, so it is not exterior."""
    c2y, mesh = _two_box_converter()

    exterior = c2y.exterior_surface_ids()
    interface = set(mesh.triangle_surface_ids) - set(exterior)
    assert len(interface) == 1, f"expected one shared face, got {interface}"
    assert len(exterior) == 10, f"expected the 10 outer faces, got {exterior}"


def test_boundary_tags_write_a_vacuum_group_on_the_exterior(tmp_path):
    """boundary_tags become a dim=2 ``boundary:<name>`` physical group."""
    c2y, _ = _two_box_converter()
    exterior = c2y.exterior_surface_ids()

    path = str(tmp_path / "tagged.arrow")
    c2y.to_arrow(path, boundary_tags={"vacuum": exterior})

    groups = _physical_groups(path)
    materials = [g for g in groups.values() if g["dim"] == 3]
    boundaries = [g for g in groups.values() if g["dim"] == 2]
    assert [g["name"] for g in materials] == ["mat:fuel", "mat:moderator"]
    assert len(boundaries) == 1
    assert boundaries[0]["name"] == "boundary:vacuum"
    assert boundaries[0]["surface_ids"] == exterior


def test_no_boundary_tags_leaves_the_mesh_without_a_boundary_group(tmp_path):
    """Omitting boundary_tags must not invent a boundary group."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    c2y.mesh_surfaces()

    path = str(tmp_path / "untagged.arrow")
    c2y.to_arrow(path)

    groups = _physical_groups(path)
    assert [g["name"] for g in groups.values()] == ["mat:steel"]
    assert all(g["dim"] == 3 for g in groups.values())


def test_mesh_to_arrow_raw(tmp_path):
    """Test the low-level mesh_to_arrow function directly."""
    path = str(tmp_path / "raw.arrow")
    mesh_to_arrow(
        path=path,
        vertices=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        triangles=[0, 1, 2],
        triangle_surface_ids=[1],
        triangle_physical_groups=[1],
        physical_groups_json='{"1": {"name": "mat:steel", "dim": 2}}',
    )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
