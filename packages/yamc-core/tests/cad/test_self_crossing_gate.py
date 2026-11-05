"""The fidelity gate refuses a solid whose surface mesh crosses itself.

A wall thinner than the sag of its facets, meshed at a loose angular
tolerance, can have its two faces cut through each other. The volume hardly
changes, so the volume check passes it; the crossing check finds it.
"""
import math

import pytest

cq = pytest.importorskip("cadquery")

from yamc._core import solid_self_intersections  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import MeshFidelityError, check_mesh_fidelity  # noqa: E402


def _thin_wall():
    """Half a tube 0.05 thick at radius 10, its two curved faces splines
    through different points, so their facets do not line up."""
    def arc(radius, points):
        return [
            cq.Vector(
                radius * math.cos(math.pi * k / (points - 1)),
                radius * math.sin(math.pi * k / (points - 1)),
                0,
            )
            for k in range(points)
        ]

    outer, inner = arc(10.0, 9), arc(9.95, 6)
    profile = cq.Wire.assembleEdges([
        cq.Edge.makeSpline(outer),
        cq.Edge.makeLine(outer[-1], inner[-1]),
        cq.Edge.makeSpline(inner[::-1]),
        cq.Edge.makeLine(inner[0], outer[0]),
    ])
    wall = cq.Solid.extrudeLinear(cq.Face.makeFromWires(profile), cq.Vector(0, 0, 5))
    assy = cq.Assembly()
    assy.add(cq.Workplane().add(wall), name="wall")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["wall"])
    return c2y


@pytest.mark.parametrize("imprint", [True, False])
def test_a_finely_faceted_thin_wall_passes(imprint):
    mesh = _thin_wall().mesh(tolerance=0.01, angular_tolerance=0.2, imprint=imprint)
    assert len(mesh.triangles) > 0


@pytest.mark.parametrize("imprint", [True, False])
def test_a_coarsely_faceted_thin_wall_is_refused(imprint):
    with pytest.raises(MeshFidelityError) as err:
        _thin_wall().mesh(tolerance=10.0, angular_tolerance=1.0, imprint=imprint)
    message = str(err.value)
    assert "solid 1 ('wall'): its surface crosses itself" in message
    assert "angular_tolerance tighter than 1" in message
    assert "fidelity_check=False" in message


@pytest.mark.parametrize("imprint", [True, False])
def test_the_crossing_is_what_the_area_and_volume_checks_miss(imprint):
    c2y = _thin_wall()
    mesh = c2y.mesh(
        tolerance=10.0, angular_tolerance=1.0, imprint=imprint, fidelity_check=False
    )
    assert check_mesh_fidelity(mesh, c2y._processed, 10.0) == []


def _box(apex):
    """A closed box whose top is a fan around a centre vertex at *apex*."""
    vertices = [
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 1], [10, 0, 1], [10, 10, 1], [0, 10, 1], [5, 5, apex],
    ]
    triangles = [
        [0, 2, 1], [0, 3, 2], [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ]
    return vertices, triangles


def test_solid_self_intersections_finds_the_crossing():
    vertices, triangles = _box(-0.5)
    [(solid, pairs, depth, points)] = solid_self_intersections(vertices, [(3, triangles)])
    assert solid == 3 and pairs == 4
    assert 0.1 < depth <= 0.5
    assert all(abs(p[2]) < 1e-9 for p in points)


@pytest.mark.parametrize("apex", [0.5, 0.0])
def test_solid_self_intersections_passes_sound_and_touching_walls(apex):
    vertices, triangles = _box(apex)
    assert solid_self_intersections(vertices, [(1, triangles)]) == []


def test_solid_self_intersections_rejects_bad_indices():
    vertices, triangles = _box(0.5)
    with pytest.raises(ValueError, match="indexes past"):
        solid_self_intersections(vertices, [(1, triangles + [[0, 1, 99]])])
