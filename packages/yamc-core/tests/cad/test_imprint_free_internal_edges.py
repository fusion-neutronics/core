"""Imprinted faces carrying an INTERNAL edge that no other face uses.

When two solids share a spherical face whose poles do not line up (one is
rotated against the other), the imprint keeps one shared face but leaves the
other solid's seam on it as INTERNAL edges, used by no other face. BRepMesh
meshes such a face short (about half the inner sphere's volume is lost), so the
edges are dropped after the imprint. INTERNAL edges that other faces use (a
line contact between solids) are kept.
"""
import math

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")

from OCP.BRepCheck import BRepCheck_Analyzer  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_INTERNAL  # noqa: E402
from OCP.TopExp import TopExp, TopExp_Explorer  # noqa: E402
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # noqa: E402

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad.assembly_processor import _imprint_assembly  # noqa: E402


def _cad_volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props, 1e-7)
    return props.Mass()


def _mesh_volumes(mesh):
    """Divergence-theorem volume of every solid in the surface mesh."""
    v = np.asarray(mesh.vertices, dtype=float)
    t = np.asarray(mesh.triangles)
    signed = np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])) / 6.0
    per_face = {}
    for fid, vol in zip(mesh.triangle_face_ids, signed):
        per_face[int(fid)] = per_face.get(int(fid), 0.0) + vol
    return {
        sid: sum(
            (-1.0 if mesh.face_solid_reversed.get((sid, fid)) else 1.0)
            * per_face.get(fid, 0.0)
            for fid in fids
        )
        for sid, fids in mesh.solid_faces.items()
    }


def _internal_edge_face_counts(compound):
    """For every INTERNAL edge in the compound, how many faces use it."""
    occ = compound.wrapped
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndUniqueAncestors_s(occ, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    counts = []
    explorer = TopExp_Explorer(occ, TopAbs_EDGE)
    while explorer.More():
        if explorer.Current().Orientation() == TopAbs_INTERNAL:
            counts.append(edge_faces.FindFromKey(explorer.Current()).Size())
        explorer.Next()
    return counts


def _sphere_in_shell(axis=(1, 0, 0), angle=0.0):
    """A radius 5 sphere inside a 5 to 10 shell rotated by *angle* degrees."""
    shell = cq.Workplane().sphere(10).cut(cq.Workplane().sphere(5))
    if angle:
        shell = shell.rotate((0, 0, 0), axis, angle)
    assy = cq.Assembly()
    assy.add(cq.Workplane().sphere(5), name="inner")
    assy.add(shell, name="shell")
    return assy


def _revolved_sphere_in_box():
    """A sphere revolved from a semicircle about x (poles on x) in a box whose
    cavity is cut with sphere() (poles on z)."""
    sphere = (
        cq.Workplane("XY").moveTo(-5, 0).threePointArc((0, 5), (5, 0)).close()
        .revolve(360, (-5, 0, 0), (5, 0, 0))
    )
    box = cq.Workplane().box(20, 20, 20).cut(cq.Workplane().sphere(5))
    assy = cq.Assembly()
    assy.add(sphere, name="sphere")
    assy.add(box, name="box")
    return assy


def _sphere_in_two_shells():
    """Sphere, shell rotated 11 degrees, second shell placed 22 degrees round
    by its assembly location."""
    assy = cq.Assembly()
    assy.add(cq.Workplane().sphere(5), name="inner")
    assy.add(
        cq.Workplane().sphere(10).cut(cq.Workplane().sphere(5))
        .rotate((0, 0, 0), (1, 0, 0), 11),
        name="middle",
    )
    assy.add(
        cq.Workplane().sphere(15).cut(cq.Workplane().sphere(10)),
        name="outer",
        loc=cq.Location((0, 0, 0), (1, 0, 0), 22),
    )
    return assy


@pytest.mark.parametrize(
    "assy",
    [
        pytest.param(_sphere_in_shell((0, 1, 0), 11), id="shell-rotated-11deg-about-y"),
        pytest.param(_sphere_in_shell((1, 0, 0), 0.1), id="shell-rotated-0.1deg-about-x"),
        pytest.param(_revolved_sphere_in_box(), id="revolved-sphere-in-box"),
        pytest.param(_sphere_in_two_shells(), id="two-rotated-shells"),
    ],
)
def test_misaligned_shared_sphere_meshes_faithfully(assy):
    """Before the INTERNAL edges were dropped, the inner sphere came out with
    0 to 290 of its 524 cm3 and the fidelity gate refused the mesh."""
    compound, solids = _imprint_assembly(assy)
    assert 1 not in _internal_edge_face_counts(compound)
    assert BRepCheck_Analyzer(compound.wrapped).IsValid()
    for solid in solids:
        assert BRepCheck_Analyzer(solid.wrapped).IsValid()

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags="assembly_names")
    mesh = c2y.mesh()  # fidelity gate on
    shapes = c2y._processed.solid_shapes
    for sid, vol_mesh in _mesh_volumes(mesh).items():
        vol_cad = _cad_volume(shapes[sid])
        assert vol_mesh == pytest.approx(vol_cad, rel=0.01), sid


def test_aligned_spheres_are_left_alone():
    """Aligned seams leave no INTERNAL edge: the imprint result is returned as
    it is and meshes as before."""
    assy = _sphere_in_shell()
    compound, _ = _imprint_assembly(assy)
    assert _internal_edge_face_counts(compound) == []

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags="assembly_names")
    mesh = c2y.mesh()
    shapes = c2y._processed.solid_shapes
    for sid, vol_mesh in _mesh_volumes(mesh).items():
        assert vol_mesh == pytest.approx(_cad_volume(shapes[sid]), rel=0.01)


def test_internal_edge_of_a_line_contact_is_kept():
    """A cube standing on one edge on a box's top face: the contact line is an
    INTERNAL edge of the top face that the cube's two faces also use. It is
    kept, while the free seam edges of a misaligned sphere pair in the same
    assembly are dropped."""
    assy = _sphere_in_shell((0, 1, 0), 11)
    assy.add(cq.Workplane().box(10, 10, 10).translate((30, 0, 0)), name="base")
    assy.add(
        cq.Workplane().box(4, 4, 4).rotate((0, 0, 0), (1, 0, 0), 45)
        .translate((30, 0, 5 + 2 * math.sqrt(2))),
        name="cube_on_edge",
    )
    compound, solids = _imprint_assembly(assy)
    assert _internal_edge_face_counts(compound) == [3]
    assert len(solids) == 4
