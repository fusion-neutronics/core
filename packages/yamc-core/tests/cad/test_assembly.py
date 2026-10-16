"""Tests for assembly processing and topology extraction."""

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad.assembly_processor import process_assembly  # noqa: E402


def make_single_box_assembly():
    """A single box assembly."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box1")
    return assy


def make_two_touching_boxes():
    """Two boxes sharing a face at x=0.5."""
    box1 = cq.Workplane("XY").box(1, 1, 1).translate((0, 0, 0))
    box2 = cq.Workplane("XY").box(1, 1, 1).translate((1, 0, 0))
    assy = cq.Assembly()
    assy.add(box1, name="left")
    assy.add(box2, name="right")
    return assy


def test_single_box_faces():
    assy = make_single_box_assembly()
    result = process_assembly(assy, material_tags=["steel"])

    # A box has 6 faces
    assert len(result.faces) == 6
    # One solid
    assert len(result.solid_faces) == 1
    # No shared faces
    assert len(result.shared_faces) == 0
    # Each face should be planar (box)
    for fid, info in result.faces.items():
        assert info.is_planar  # Box faces are planar


def test_two_boxes_shared_face():
    assy = make_two_touching_boxes()
    result = process_assembly(assy, material_tags=["steel", "aluminum"])

    # Two solids
    assert len(result.solid_faces) == 2
    # Should have shared faces (at least 1 at the interface)
    assert len(result.shared_faces) >= 1, f"Expected shared faces, got {result.shared_faces}"
    # Each shared face should belong to exactly 2 solids
    for fid, solids in result.shared_faces.items():
        assert len(solids) == 2


def test_face_to_occ_present():
    assy = make_single_box_assembly()
    result = process_assembly(assy, material_tags=["steel"])

    # Every face should have an OCC face reference
    for fid in result.faces:
        assert fid in result.face_to_occ


def test_material_tags_preserved():
    assy = make_two_touching_boxes()
    result = process_assembly(assy, material_tags=["steel", "aluminum"])

    assert len(result.material_tags) == 2


def test_imprinted_compound_returned():
    assy = make_single_box_assembly()
    result = process_assembly(assy, material_tags=["steel"])

    # The imprinted compound should be present for BRepMesh
    assert result.imprinted_compound is not None


def test_face_solid_reversed_tracked():
    assy = make_two_touching_boxes()
    result = process_assembly(assy, material_tags=["steel", "aluminum"])

    # Every face/solid pair should have orientation tracked
    for solid_id, face_ids in result.solid_faces.items():
        for fid in face_ids:
            assert (solid_id, fid) in result.face_solid_reversed


def test_a_solid_inside_another_is_refused():
    """A cylinder inside a box that was never cut: the imprint returns the
    cylinder's region as a piece of the box, so the cylinder's material
    vanished (zero flux, no lost particles). Refused, naming it."""
    from yamc.cad import OverlappingSolidsError

    box = cq.Workplane("XY").box(30, 30, 30)
    cylinder = cq.Workplane("XY").circle(4).extrude(20).translate((0, 0, -10))
    assy = cq.Assembly()
    assy.add(cylinder, name="cylinder")
    assy.add(box, name="box")
    with pytest.raises(OverlappingSolidsError, match="'cylinder' was absorbed"):
        process_assembly(assy, material_tags=["a", "b"])


def test_one_of_two_solids_under_a_name_being_absorbed_is_refused():
    """One child holds two cubes, one inside a box that was never cut and one
    far away: the far cube still carries the child's name out of the imprint,
    but the child's volume does not, so the absorbed cube is still caught."""
    from yamc.cad import OverlappingSolidsError

    box = cq.Workplane("XY").box(30, 30, 30)
    pellets = (
        cq.Workplane("XY").box(4, 4, 4)
        .add(cq.Workplane("XY").box(4, 4, 4).translate((100, 0, 0)))
        .combine(glue=False)
    )
    assy = cq.Assembly()
    assy.add(box, name="box")
    assy.add(pellets, name="pellets")
    assert len(pellets.solids().vals()) == 2
    with pytest.raises(OverlappingSolidsError, match="'pellets' came out with 64"):
        process_assembly(assy, material_tags=["a", "b"])


def test_an_assembly_node_without_solids_is_not_an_input():
    """A face in the assembly (a tally plane, say) has no solid to come out of
    the imprint, and must not read as absorbed."""
    a = cq.Workplane("XY").box(10, 10, 10)
    b = cq.Workplane("XY").box(10, 10, 10).translate((10, 0, 0))
    plane = cq.Face.makePlane(5, 5)
    assy = cq.Assembly()
    assy.add(a, name="a")
    assy.add(b, name="b")
    assy.add(plane, name="tally_plane")
    result = process_assembly(assy, material_tags=["a", "b", "c"])
    assert len(result.solid_faces) == 2
