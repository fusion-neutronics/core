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


@pytest.fixture
def imprint_pool():
    from OCP.OSD import OSD_ThreadPool

    pool = OSD_ThreadPool.DefaultPool_s()
    previous_threads = pool.NbThreads()
    previous_launch_threads = pool.NbDefaultThreadsToLaunch()
    pool.Init(3)
    pool.SetNbDefaultThreadsToLaunch(2)
    yield pool
    pool.Init(previous_threads)
    pool.SetNbDefaultThreadsToLaunch(previous_launch_threads)


@pytest.mark.parametrize("threads", [1, 2, "all"])
def test_imprint_threads_are_scoped_and_topology_is_preserved(
    threads, imprint_pool, monkeypatch
):
    from OCP.OSD import OSD_Parallel
    from yamc.cad import assembly_processor

    expected = OSD_Parallel.NbLogicalProcessors_s() if threads == "all" else threads
    original_imprint = cq.occ_impl.assembly.imprint
    original_strip = assembly_processor._strip_free_internal_edges
    calls = []

    def imprint(assembly):
        calls.append((imprint_pool.NbThreads(), imprint_pool.NbDefaultThreadsToLaunch()))
        return original_imprint(assembly)

    def strip(compound, solids):
        assert imprint_pool.NbThreads() == 3
        assert imprint_pool.NbDefaultThreadsToLaunch() == 2
        return original_strip(compound, solids)

    monkeypatch.setattr(cq.occ_impl.assembly, "imprint", imprint)
    monkeypatch.setattr(assembly_processor, "_strip_free_internal_edges", strip)
    result = process_assembly(
        make_two_touching_boxes(), ["steel", "aluminum"], imprint_threads=threads
    )
    assert calls == [(expected, expected)]
    assert imprint_pool.NbThreads() == 3
    assert imprint_pool.NbDefaultThreadsToLaunch() == 2
    assert len(result.solid_faces) == 2
    assert len(result.shared_faces) == 1


def test_imprint_uses_a_bounded_pool_by_default(imprint_pool, monkeypatch):
    from yamc.cad import assembly_processor

    original_imprint = cq.occ_impl.assembly.imprint
    calls = []

    def imprint(assembly):
        calls.append(imprint_pool.NbThreads())
        return original_imprint(assembly)

    monkeypatch.setattr(cq.occ_impl.assembly, "imprint", imprint)
    process_assembly(make_two_touching_boxes(), ["steel", "aluminum"])
    assert calls == [assembly_processor.IMPRINT_THREADS] == [4]


def test_imprint_threads_are_restored_on_error(imprint_pool, monkeypatch):
    def fail(assembly):
        assert imprint_pool.NbThreads() == 1
        raise RuntimeError("imprint failed")

    monkeypatch.setattr(cq.occ_impl.assembly, "imprint", fail)
    with pytest.raises(RuntimeError, match="imprint failed"):
        process_assembly(make_two_touching_boxes(), ["a", "b"], imprint_threads=1)
    assert imprint_pool.NbThreads() == 3
    assert imprint_pool.NbDefaultThreadsToLaunch() == 2


def test_imprint_threads_preserve_partial_glue(imprint_pool, monkeypatch):
    original_imprint = cq.occ_impl.assembly.imprint
    calls = []

    def imprint(assembly, glue=None):
        calls.append((glue, imprint_pool.NbThreads()))
        return original_imprint(assembly)

    monkeypatch.setattr(cq.occ_impl.assembly, "imprint", imprint)
    result = process_assembly(make_two_touching_boxes(), ["a", "b"], imprint_threads=2)
    assert calls == [("partial", 2)]
    assert len(result.shared_faces) == 1
    assert imprint_pool.NbThreads() == 3
    assert imprint_pool.NbDefaultThreadsToLaunch() == 2


@pytest.mark.parametrize("imprint", [True, False])
@pytest.mark.parametrize("threads", [0, -1, True, False, 1.5, "2", None])
def test_invalid_imprint_threads_are_rejected_even_without_imprinting(imprint, threads):
    with pytest.raises(ValueError, match="imprint_threads"):
        process_assembly(
            make_single_box_assembly(), ["steel"], imprint=imprint,
            imprint_threads=threads,
        )


@pytest.mark.parametrize("imprint", [True, False])
def test_skipped_imprints_leave_threads_untouched(imprint, imprint_pool, monkeypatch):
    from yamc.cad import assembly_processor

    def unexpected(threads):
        pytest.fail("a skipped imprint must not change OCC thread settings")

    monkeypatch.setattr(assembly_processor, "setThreads", unexpected)
    process_assembly(
        make_single_box_assembly(), ["steel"], imprint=imprint, imprint_threads=1
    )
    assert imprint_pool.NbThreads() == 3
    assert imprint_pool.NbDefaultThreadsToLaunch() == 2


@pytest.mark.parametrize("method", ["mesh", "mesh_surfaces"])
def test_public_mesher_forwards_imprint_threads(method, monkeypatch):
    from yamc.cad import CadToYamc

    mesher = CadToYamc()
    mesher.add_cadquery_object(make_two_touching_boxes(), ["a", "b"])
    calls = []

    def mesh_surfaces(**kwargs):
        calls.append(kwargs["imprint_threads"])

    monkeypatch.setattr(mesher, "_mesh_surfaces", mesh_surfaces)
    getattr(mesher, method)(imprint_threads=2)
    assert calls == [2]


@pytest.mark.parametrize("method", ["mesh", "mesh_surfaces"])
def test_public_mesher_imprint_threads_are_restored_before_meshing(
    method, imprint_pool, monkeypatch
):
    from yamc.cad import CadToYamc
    from yamc.cad import _brepmesh

    original_mesh = _brepmesh.brepmesh_face_meshes
    calls = []

    def mesh_faces(*args, **kwargs):
        calls.append((imprint_pool.NbThreads(), imprint_pool.NbDefaultThreadsToLaunch()))
        return original_mesh(*args, **kwargs)

    monkeypatch.setattr(_brepmesh, "brepmesh_face_meshes", mesh_faces)
    mesher = CadToYamc()
    mesher.add_cadquery_object(make_two_touching_boxes(), ["a", "b"])
    result = getattr(mesher, method)(imprint_threads=1)
    assert calls == [(3, 2)]
    assert result.num_solids == 2
    assert len(result.shared_faces) == 1
    assert result.triangles


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


def test_imprinting_a_revolved_spline_solid_keeps_its_volume():
    """A revolved spline solid inside an enclosure split at its midplane: the
    imprint splits the solid's faces where the pieces meet, and OCC's default
    volume integration of the result drifts by a fraction of a percent. The
    volume check must not read that as part of the solid being absorbed."""
    import math

    def d_profile(scale, r0=600.0, n=12):
        return [
            (r0 + scale * math.cos(a) + 0.25 * scale * math.cos(2 * a),
             1.7 * scale * math.sin(a))
            for a in (2 * math.pi * i / n for i in range(n))
        ]

    plasma = (
        cq.Workplane("XZ").spline(d_profile(200.0), periodic=True).close()
        .revolve(90, (0, 0, 0), (0, 1, 0))
    )
    bb = plasma.val().BoundingBox()
    enclosure = cq.Workplane("XY").box(bb.xlen + 100, bb.ylen + 100, bb.zlen + 100).translate(
        (bb.center.x, bb.center.y, bb.center.z)
    )
    wall = enclosure.cut(plasma, clean=False)
    half = cq.Workplane("XY").box(1e5, 1e5, 1e5, centered=(True, True, False)).translate(
        (0, 0, bb.center.z)
    )
    assy = cq.Assembly()
    assy.add(plasma, name="plasma")
    assy.add(wall.intersect(half, clean=False), name="upper")
    assy.add(wall.cut(half, clean=False), name="lower")
    result = process_assembly(assy, material_tags=["a", "b", "c"])
    assert len(result.solid_faces) == 3


def test_imprinting_a_dense_spline_extrusion_keeps_its_volume():
    """A 150-point spline extruded, with a box on part of its top: the
    imprint splits that face. OCC's adaptive integration put the extrusion
    69% high before the split and 33% high after it, which read as a fifth
    of it absorbed into the box."""
    import math

    angles = [2 * math.pi * i / 150 for i in range(150)]
    radii = [50 + 8 * math.cos(15 * a) for a in angles]
    profile = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
    extrusion = cq.Workplane("XY").spline(profile, periodic=True).close().extrude(100)
    box = cq.Workplane("XY").box(40, 40, 30, centered=(True, True, False)).translate(
        (0, 0, 100)
    )
    assy = cq.Assembly()
    assy.add(extrusion, name="extrusion")
    assy.add(box, name="box")
    result = process_assembly(assy, material_tags=["a", "b"])
    assert len(result.solid_faces) == 2


def test_a_mirrored_part_is_not_read_as_absorbed():
    """A mirroring location turns OCC's integrated volume negative."""
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf

    mirror = gp_Trsf()
    mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    assy = cq.Assembly()
    assy.add(cq.Workplane("XY").box(10, 20, 5).translate((10, 0, 0)), name="left",
             loc=cq.Location(mirror))
    result = process_assembly(assy, material_tags=["a"])
    assert len(result.solid_faces) == 1


def test_overlapping_solids_under_one_name_are_accepted():
    """Two overlapping boxes in one child (one material): the imprint splits
    them into three pieces, the shared one listing the child's name twice, and
    nothing of either is lost."""
    a = cq.Workplane("XY").box(10, 10, 10).val()
    b = cq.Workplane("XY").box(10, 10, 10).translate((5, 0, 0)).val()
    assy = cq.Assembly()
    assy.add(cq.Workplane("XY").add(cq.Compound.makeCompound([a, b])), name="body")
    assy.add(cq.Workplane("XY").box(4, 4, 4).translate((30, 0, 0)), name="other")
    result = process_assembly(assy, material_tags=["a", "b"])
    assert len(result.solid_faces) == 4
