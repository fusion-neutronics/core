"""The compiled face-volume correction against the pure-Python reference it
replaced (``_face_volumes_reference``, next to this file).

Both correct copies of one uncorrected mesh. They differ only in the order a
few sums are taken (a vertex set's centre, a slope), so every vertex agrees
to rounding and every count exactly. The solids exercise both passes:
spheres (interior face vertices), a cylinder and a cone (edges only), a
torus (seam edges), B-spline lofts (faces the scene mesher meshes), and
nested spheres imprinted and not (one shared face, or two faces welded onto
one set of vertices).
"""

import copy

import pytest

cq = pytest.importorskip("cadquery")

from _face_volumes_reference import preserve_face_volumes as reference  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad import mesher as mesher_module  # noqa: E402
from yamc.cad._face_volumes import preserve_face_volumes  # noqa: E402

_COARSE = dict(tolerance=0.3, angular_tolerance=0.5)


def _loft():
    return cq.Workplane().circle(5).workplane(offset=10).ellipse(3, 6).loft()


def _twisted():
    return (
        cq.Workplane().rect(8, 4).workplane(offset=6)
        .transformed(rotate=(0, 0, 40)).ellipse(5, 2.5).loft()
    )


def _nested():
    core = cq.Workplane().sphere(5)
    return dict(core=core, shell=cq.Workplane().sphere(10).cut(core))


_CASES = {
    "sphere": (lambda: dict(part=cq.Workplane().sphere(10)), {}),
    "cylinder": (lambda: dict(part=cq.Workplane().cylinder(20, 5)), {}),
    "drilled_sphere": (
        lambda: dict(
            part=cq.Workplane().sphere(10).cut(cq.Workplane().cylinder(30, 4))
        ),
        {},
    ),
    "cone": (lambda: dict(part=cq.Workplane().add(cq.Solid.makeCone(6, 2, 10))), {}),
    "torus": (lambda: dict(part=cq.Workplane().add(cq.Solid.makeTorus(10, 3))), {}),
    "loft": (lambda: dict(part=_loft()), {}),
    "twisted_loft": (lambda: dict(part=_twisted()), {}),
    "nested_imprinted": (_nested, {}),
    "nested_not_imprinted": (_nested, dict(imprint=False, overlaps="ignore")),
}


def _uncorrected(parts, monkeypatch, **kwargs):
    monkeypatch.setattr(mesher_module, "_PRESERVE_FACE_VOLUMES", False)
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags="assembly_names")
    return c2y.mesh(**_COARSE, **kwargs), c2y._processed


@pytest.mark.parametrize("case", list(_CASES))
def test_matches_python_reference(case, monkeypatch):
    build, kwargs = _CASES[case]
    mesh, pa = _uncorrected(build(), monkeypatch, **kwargs)
    tolerances = {
        fid: (_COARSE["tolerance"], _COARSE["angular_tolerance"]) for fid in pa.faces
    }
    original = [list(v) for v in mesh.vertices]
    expected_mesh = copy.deepcopy(mesh)
    expected = reference(expected_mesh, pa, tolerances)
    got = preserve_face_volumes(mesh, pa, tolerances)

    assert got.pop("seconds") >= 0.0
    assert got == expected
    assert expected["edges_moved"] + expected["faces_moved"] > 0
    size = max(max(abs(x) for x in v) for v in original)
    worst = max(
        abs(a - b)
        for p, q in zip(mesh.vertices, expected_mesh.vertices)
        for a, b in zip(p, q)
    )
    moved = max(
        abs(a - b)
        for p, q in zip(mesh.vertices, original)
        for a, b in zip(p, q)
    )
    assert moved > 1e-3
    assert worst <= 1e-12 * size
