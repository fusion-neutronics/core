"""A solid far thinner than the model around it (issue #522).

CadToYamc welds the faces' meshes together, merging the copy each face holds
of a point they share. The weld used to merge any two vertices within 1e-6 of
the model's extent of each other, so a plate thinner than that between two
blocks had its two sides welded onto the same vertices: a shell of facets
listed both ways that enclosed nothing, lost every particle entering it and
passed the fidelity gate. The weld now never merges two points of one face,
so the faces around the plate's rim keep its sides apart, down to a
billionth of the model's extent; and the gate refuses a solid thinner than
that, which collapses.
"""
import dataclasses
from collections import Counter

import numpy as np
import pytest
import yamc

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import MeshFidelityError, check_mesh_fidelity  # noqa: E402

SIZE = 1000.0  # cm: the blocks' width, and the model's extent
DEPTH = 100.0  # cm: each block's height


def _sandwich(thickness, size=SIZE, depth=DEPTH):
    """Blocks "a" and "b", size x size x depth, with a "plate" *thickness*
    thick between them."""
    a = cq.Solid.makeBox(size, size, depth)
    plate = cq.Solid.makeBox(size, size, thickness).translate(cq.Vector(0, 0, depth))
    b = cq.Solid.makeBox(size, size, depth).translate(cq.Vector(0, 0, depth + thickness))
    c2y = CadToYamc()
    assembly = cq.Assembly()
    for name, solid in (("a", a), ("plate", plate), ("b", b)):
        assembly.add(solid, name=name)
    c2y.add_cadquery_object(assembly, ["a", "plate", "b"])
    return c2y


def _geometry(c2y, tmp_path):
    path = str(tmp_path / "sandwich.arrow")
    c2y.to_arrow(path)
    return yamc.MeshGeometry(path, {})


def test_a_plate_a_ten_millionth_of_the_model_thick_loses_no_particles(tmp_path):
    """The issue's model: a 1e-4 cm plate between two 1000 cm blocks, which
    welded to nothing and lost every particle crossing into it."""
    c2y = _sandwich(1e-4)
    c2y.mesh()
    geometry = _geometry(c2y, tmp_path)
    assert geometry.volume_measures == pytest.approx(
        [SIZE * SIZE * DEPTH, SIZE * SIZE * 1e-4, SIZE * SIZE * DEPTH], rel=1e-6
    )
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(
            position=(SIZE / 2, SIZE / 2, DEPTH / 2),
            energy=yamc.sources.Discrete([14.06e6], [1.0]),
        ),
        max_lost_particles=10_000,
    )
    model.simulate_transport(total_particles=2_000, seed=1)
    assert len(model.lost_particles) == 0


def _open_edges(mesh, sid):
    """How many edges of solid *sid*'s facets other than two of them use."""
    faces = set(mesh.solid_faces[sid])
    uses = Counter()
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        if fid in faces:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                uses[min(a, b), max(a, b)] += 1
    return sum(1 for n in uses.values() if n != 2)


@pytest.mark.parametrize("fraction", [1e-3, 1e-5, 1e-6, 1e-7, 1e-8])
def test_a_plate_down_to_a_hundred_millionth_of_the_model_keeps_its_volume(
    tmp_path, fraction
):
    """From a thousandth of the model's extent, far above the old weld's
    1e-6, to a hundredth of it; every solid stays closed."""
    thickness = fraction * SIZE
    c2y = _sandwich(thickness)
    mesh = c2y.mesh()
    assert [_open_edges(mesh, sid) for sid in (1, 2, 3)] == [0, 0, 0]
    assert _geometry(c2y, tmp_path).volume_measures == pytest.approx(
        [SIZE * SIZE * DEPTH, SIZE * SIZE * thickness, SIZE * SIZE * DEPTH], rel=1e-6
    )


def test_a_plate_thinner_than_the_mesh_resolves_is_refused():
    """1e-10 of a 1e5 cm model: below the billionth of the extent within
    which the weld takes two points of one face for one, the plate's sides
    are welded together, and the gate names it and its thickness."""
    c2y = _sandwich(1e-5, size=1e5, depth=1e4)
    with pytest.raises(MeshFidelityError) as err:
        c2y.mesh()
    message = str(err.value)
    assert "solid 2 ('plate'): collapsed" in message
    assert "It is 1e-05 thick" in message
    assert "solid 1 " not in message and "solid 3 " not in message


def test_the_gate_refuses_a_solid_whose_sides_lie_on_each_other():
    """The plate of the issue's model, its top moved down onto its bottom as
    the old weld left it, encloses nothing: the gate reports it collapsed and
    leaves the blocks alone."""
    c2y = _sandwich(1e-4)
    mesh = c2y.mesh()
    top = DEPTH + 1e-4
    collapsed = dataclasses.replace(
        mesh, vertices=[[x, y, DEPTH if z == top else z] for x, y, z in mesh.vertices]
    )
    problems = check_mesh_fidelity(collapsed, c2y._processed, 0.5)
    assert [p for p in problems if p.startswith("solid 2 ('plate'): collapsed")]
    assert not [p for p in problems if p.startswith(("solid 1", "solid 3"))]


def test_a_plate_a_ten_millionth_of_the_model_thick_tet_meshes_to_its_volume():
    """The tet mesher takes the plate's boundary from the welded surface as it
    is. Welding it again by position, as it was, left four vertices and no
    tets."""
    c2y = _sandwich(1e-4)
    c2y.mesh(tet_volumes=["plate"], target_edge_length=SIZE / 2)
    vertices, tets = c2y._tet_data[2]
    assert tets, "the plate has no tets"
    vertices, tets = np.asarray(vertices), np.asarray(tets)
    a, b, c, d = (vertices[tets[:, k]] for k in range(4))
    volumes = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    assert volumes.min() > 0.0
    assert volumes.sum() == pytest.approx(SIZE * SIZE * 1e-4, rel=1e-6)
