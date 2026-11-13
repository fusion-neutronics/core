"""Faces of a solid that bound nothing (#518).

OCC keeps a face that lies inside a solid in that solid as an INTERNAL face:
CadQuery's ``Shape.split`` with a face that stops short of the solid's boundary
leaves one, and so does a general fuse of a solid with a face. Such a face, with
the solid on both sides, bounds nothing. It used to be meshed as a boundary
between the solid and the implicit complement, so the solid was left open, its
volume came out wrong, the fidelity gate passed it, and surface tracking lost
the particles that crossed the face. It is left out of the mesh now, as
cad-to-dagmc leaves it out, and so is a face a solid holds both ways round or
as EXTERNAL. And the fidelity gate refuses a solid whose surface is open.
"""

import math
import os
from collections import Counter

import pytest
import yamc

cq = pytest.importorskip("cadquery")
from OCP.BOPAlgo import BOPAlgo_Builder  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.TopAbs import (  # noqa: E402
    TopAbs_EXTERNAL, TopAbs_FACE, TopAbs_FORWARD, TopAbs_INTERNAL, TopAbs_REVERSED,
)
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import MeshFidelityError  # noqa: E402


def _square(z, lo, hi):
    """A square face from lo to hi in x and y, at height z."""
    return cq.Face.makeFromWires(cq.Wire.makePolygon(
        [(lo, lo, z), (hi, lo, z), (hi, hi, z), (lo, hi, z)], close=True
    ))


def _box():
    return cq.Solid.makeBox(10, 10, 10)


def _spline_box():
    """The 10 cm box with every face a B-spline surface, which the scene CDT
    meshes (BRepMesh meshes the plain box)."""
    return cq.Shape.cast(BRepBuilderAPI_NurbsConvert(_box().wrapped, True).Shape())


def _split_fin(box):
    """*box* split by a 6 cm square at z = 5 that stops short of its walls:
    one solid, holding the square as an INTERNAL face."""
    return box.split(_square(5, 2, 8))


def _split_slit(box):
    """*box* split by a face that enters through three of its walls at z = 5
    and stops half way across: one solid, whose walls carry the face's
    edges."""
    slit = cq.Face.makeFromWires(cq.Wire.makePolygon(
        [(-1, -1, 5), (5, -1, 5), (5, 11, 5), (-1, 11, 5)], close=True
    ))
    return box.split(slit)


def _fused_sheet():
    """A 10 cm box general-fused with a 6 cm square at z = 5.7."""
    builder = BOPAlgo_Builder()
    builder.AddArgument(_box().wrapped)
    builder.AddArgument(_square(5.7, 2, 8).wrapped)
    builder.Perform()
    [solid] = cq.Shape.cast(builder.Shape()).Solids()
    return solid


def _cylinder_fin():
    """A cylinder of radius 5 split by a 6 cm square across its axis."""
    return cq.Solid.makeCylinder(5, 10).split(_square(5, -3, 3))


def _box_with(*faces):
    """The box as a solid built by hand, with *faces* added to its shell."""
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in [f.wrapped for f in _box().Faces()] + list(faces):
        builder.Add(shell, face)
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, shell)
    return cq.Solid(solid)


def _held_both_ways():
    """The box holding the 6 cm square once each way round: a face with the
    solid on both sides, not marked INTERNAL."""
    square = _square(5, 2, 8).wrapped
    return _box_with(square, square.Reversed())


def _external():
    """The box holding a square above it as an EXTERNAL face, with the solid
    on neither side."""
    return _box_with(_square(12, 2, 8).wrapped.Oriented(TopAbs_EXTERNAL))


CASES = {
    "split by a short face": (lambda: _split_fin(_box()), _box),
    "split by a face from the walls": (lambda: _split_slit(_box()), _box),
    "general fuse with a face": (_fused_sheet, _box),
    "cylinder split by a short face": (_cylinder_fin, lambda: cq.Solid.makeCylinder(5, 10)),
    "spline box split by a short face": (lambda: _split_fin(_spline_box()), _box),
    "spline box split by a face from the walls": (
        lambda: _split_slit(_spline_box()), _box
    ),
    "a face held both ways": (_held_both_ways, _box),
    "an EXTERNAL face": (_external, _box),
}


def _orientations(solid):
    """Each face occurrence's orientation in *solid*, by face."""
    held = {}
    explorer = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        held.setdefault(hash(face), set()).add(face.Orientation())
        explorer.Next()
    return held


def _cad_volume(solid):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    return props.Mass()


def _mesh(parts, tmp_path, name, **kwargs):
    """Mesh *parts* ({name: shape}) and write it; returns the mesh and path."""
    assembly = cq.Assembly()
    for part, shape in parts.items():
        assembly.add(shape, name=part)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=list(parts))
    mesh = c2y.mesh(tolerance=0.1, **kwargs)
    path = os.path.join(tmp_path, f"{name}.arrow")
    c2y.to_arrow(path)
    return mesh, path


def _open_edges(mesh, sid):
    """Edges of solid *sid*'s facets, each face's taken outward, that its
    facets do not run along as often one way as the other."""
    faces = set(mesh.solid_faces[sid])
    runs = Counter()
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        if fid in faces:
            a, b, c = tri[::-1] if mesh.face_solid_reversed.get((sid, fid)) else tri
            runs.update([(a, b), (b, c), (c, a)])
    return [e for e, n in runs.items() if n != runs[(e[1], e[0])]]


@pytest.mark.parametrize("case", CASES)
def test_a_face_that_bounds_nothing_is_left_out(tmp_path, case):
    """The solid is meshed from the faces that bound it: closed, and with the
    volume of the same solid without the extra face."""
    make, outline = CASES[case]
    solid = make()
    held = _orientations(solid)
    bounding = ({TopAbs_FORWARD}, {TopAbs_REVERSED})
    assert sum(o not in bounding for o in held.values()) == 1

    mesh, path = _mesh({"steel": solid}, tmp_path, "inside")
    assert len(mesh.solid_faces[1]) == len(held) - 1
    assert _open_edges(mesh, 1) == []
    # The divergence-theorem volume, as transport reads the file.
    [volume] = yamc.MeshGeometry(path, {}).volume_measures
    _, plain = _mesh({"steel": outline()}, tmp_path, "plain")
    [plain_volume] = yamc.MeshGeometry(plain, {}).volume_measures
    assert _cad_volume(solid) == pytest.approx(_cad_volume(outline()), rel=1e-9)
    assert volume == pytest.approx(plain_volume, rel=1e-9)


@pytest.mark.parametrize("imprint", [True, False])
def test_a_face_inside_a_solid_is_left_out_next_to_another(tmp_path, imprint):
    """With a box on top, so the imprint runs (the face survives it), and
    without imprinting."""
    top = cq.Solid.makeBox(10, 10, 10).translate(cq.Vector(0, 0, 10))
    mesh, path = _mesh(
        {"steel": _split_fin(_box()), "lead": top}, tmp_path, "pair", imprint=imprint
    )
    assert [len(mesh.solid_faces[s]) for s in (1, 2)] == [6, 6]
    assert _open_edges(mesh, 1) == _open_edges(mesh, 2) == []
    volumes = yamc.MeshGeometry(path, {}).volume_measures
    assert volumes == pytest.approx([1000.0, 1000.0], rel=1e-9)


def _steel():
    steel = yamc.Material(
        composition={"Fe56": 1.0}, density=7.874, name="steel", temperature=294
    )
    steel.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    return steel


def _flux(path, position, tracking_mode):
    """Flux in the steel and the particles lost, for 2000 neutrons from
    *position*."""
    steel = _steel()
    tally = yamc.Tally(scores=["flux"], materials=steel, name="steel")
    model = yamc.Model(
        geometry=yamc.MeshGeometry(path, {"steel": steel}),
        tallies=[tally],
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=position
        ),
        tracking_mode=tracking_mode,
        max_lost_particles=10**6,
        verbose=[],
    )
    results = model.simulate_transport(total_particles=2_000, seed=1)
    result = results["steel"]
    return result.mean[0], result.standard_deviation[0], len(model.lost_particles)


@pytest.mark.parametrize("tracking_mode", ["surface", "woodcock"])
def test_no_particle_is_lost_crossing_a_face_inside_a_solid(tmp_path, tracking_mode):
    """Sources under, over and beside the sheet the general fuse leaves in the
    box. Surface tracking used to hand a particle crossing the sheet to the
    implicit complement and lose it, as it did 972, 722 and 89 of 2000 in
    the issue."""
    _, path = _mesh({"steel": _fused_sheet()}, tmp_path, "sheet")
    _, plain = _mesh({"steel": _box()}, tmp_path, "plain")
    for position in [(5.0, 5.0, 5.0), (5.0, 5.0, 7.0), (9.2, 9.2, 8.0)]:
        mean, std, lost = _flux(path, position, tracking_mode)
        plain_mean, plain_std, plain_lost = _flux(plain, position, tracking_mode)
        assert lost == plain_lost == 0, f"{lost} lost from {position}"
        z = (mean - plain_mean) / math.hypot(std, plain_std)
        assert abs(z) < 4.0, f"from {position}: flux {mean:.5g} vs {plain_mean:.5g}"


def test_the_gate_refuses_a_solid_left_open(monkeypatch):
    """The square put back into the box as one of its boundary faces, as it
    used to be meshed: the facets no longer close, which the fidelity gate
    refuses. Its volume alone comes out 6% high, inside the gate's allowance
    for a box this small, which is how the open box used to pass."""
    import yamc.cad.mesher as mesher
    from yamc.cad.assembly_processor import FaceInfo

    process = mesher.process_assembly

    def keeping_the_square(*args, **kwargs):
        pa = process(*args, **kwargs)
        explorer = TopExp_Explorer(pa.solid_shapes[1].wrapped, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            explorer.Next()
            if face.Orientation() == TopAbs_INTERNAL and not any(
                face.IsSame(f) for f in pa.face_to_occ.values()
            ):
                fid = max(pa.faces) + 1
                pa.faces[fid] = FaceInfo(fid, True, 36.0, [1])
                pa.face_to_occ[fid] = face
                pa.solid_faces[1].append(fid)
                pa.face_solid_reversed[(1, fid)] = False  # not REVERSED
        return pa

    monkeypatch.setattr(mesher, "process_assembly", keeping_the_square)
    assembly = cq.Assembly()
    assembly.add(_split_fin(_box()), name="steel")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    with pytest.raises(MeshFidelityError, match=r"solid 1 \('steel'\): its surface is open"):
        c2y.mesh(tolerance=0.1)
