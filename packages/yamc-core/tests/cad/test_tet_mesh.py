"""Tests for tetrahedral volume meshing via the yamm tet mesher."""

import os

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402


def test_tet_mesh_single_box():
    """Tet mesh a single box."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel"])
    c2y.mesh_surfaces()
    tet_data = c2y.mesh_volumes()

    assert len(tet_data) == 1
    solid_id = list(tet_data.keys())[0]
    tet_v, tet_t = tet_data[solid_id]
    assert len(tet_v) > 4  # Should have more vertices than a single tet
    assert len(tet_t) > 0  # Should have at least one tet


def test_tet_mesh_selective(tmp_path):
    """Tet mesh only one of two volumes."""
    box1 = cq.Workplane("XY").box(1, 1, 1)
    box2 = cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0))
    assy = cq.Assembly()
    assy.add(box1, name="left")
    assy.add(box2, name="right")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel", "moderator"])
    c2y.mesh_surfaces()
    tet_data = c2y.mesh_volumes(volumes_to_tet=["fuel"])

    # Only one volume should have tets
    assert len(tet_data) == 1

    # Export should work with mixed tri+tet data
    path = str(tmp_path / "mixed.arrow")
    c2y.to_arrow(path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_tet_volumes_picks_solids_by_their_material_tags():
    """A part holding two solids gives the parts after it later solid ids,
    so tet_volumes picks solids by the tags the mesh gives them, not by
    their place in the list of material tags."""
    pair = cq.Workplane().sphere(2).union(cq.Workplane().sphere(2).translate((10, 0, 0)))
    assy = cq.Assembly()
    assy.add(pair, name="pair")
    assy.add(cq.Workplane().sphere(2).translate((0, 20, 0)), name="single")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["pair_tag", "single_tag"])
    mesh = c2y.mesh(tet_volumes=["single_tag"], target_edge_length=2.0)
    assert mesh.material_tags == ["pair_tag", "pair_tag", "single_tag"]
    assert list(c2y._tet_data) == [3]


def test_meshing_again_drops_the_tets_of_the_earlier_mesh(tmp_path):
    """Tets index the vertices of the mesh they were made for: a second mesh
    without tet_volumes has none (the export used to panic on them)."""
    assy = cq.Assembly()
    assy.add(cq.Workplane("XY").box(1, 1, 1), name="left")
    assy.add(cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0)), name="right")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel", "moderator"])
    c2y.mesh(tet_volumes=["fuel"], target_edge_length=0.5, tolerance={"fuel": 0.05})
    assert len(c2y._tet_data) == 1
    c2y.mesh()
    assert not c2y._tet_data
    c2y.to_arrow(str(tmp_path / "remeshed.arrow"))


def test_tet_mesh_with_export(tmp_path):
    """Full pipeline: surface mesh + tet mesh + arrow export."""
    box = cq.Workplane("XY").box(2, 2, 2)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    c2y.mesh_surfaces()
    c2y.mesh_volumes()

    path = str(tmp_path / "full_pipeline.arrow")
    c2y.to_arrow(path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
