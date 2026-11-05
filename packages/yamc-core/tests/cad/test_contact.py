"""The separated-pair check against the CAD.

``find_separated`` decides, for each sample point, whether it is inside, on,
or within a small distance of another CAD solid. It finds the nearest boundary
point and reads the side from the outward normals there, and leaves what that
cannot settle to the solid classifier. These tests pin that it gives the same
answer as one distance query per point against the whole solid (zero inside),
on planar, curved and B-spline faces, convex and concave edges, and points
inside, outside and on the surface.
"""

import random

import pytest

cq = pytest.importorskip("cadquery")
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex  # noqa: E402
from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # noqa: E402
from OCP.gp import gp_Pnt  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._contact import _CadSolid  # noqa: E402

EPS = 1e-3


def _solids():
    spline = (
        cq.Workplane("XY")
        .spline([(0, 0), (4, 1.5), (8, -1), (10, 2)], includeCurrent=False)
        .lineTo(10, 6)
        .spline([(6, 7), (2, 5), (0, 6)], includeCurrent=True)
        .close()
        .extrude(3)
    )
    lofted = cq.Workplane("XY").rect(6, 4).workplane(offset=5).circle(2.5).loft()
    return {
        "box": cq.Workplane().box(4, 3, 2),
        "filleted box": cq.Workplane().box(4, 3, 2).edges().fillet(0.5),
        "box with a notch": cq.Workplane()
        .box(4, 4, 4)
        .cut(cq.Workplane().box(2, 2, 2).translate((1, 1, 1))),
        "cylinder": cq.Workplane().circle(2).extrude(3),
        "sphere": cq.Workplane().sphere(2),
        "spline extrusion": spline,
        "loft": lofted,
    }


def _touches_whole_solid(shape, p):
    dist = BRepExtrema_DistShapeShape(
        BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex(), shape
    )
    return not dist.IsDone() or dist.Value() <= EPS


@pytest.mark.parametrize("name", list(_solids()))
@pytest.mark.parametrize("as_solid", [True, False])
def test_nearest_point_side_matches_the_whole_solid_query(name, as_solid):
    """Points inside, outside and on the boundary of each solid, near it and
    far from it, get the same answer as a distance query against the whole
    shape: zero inside a solid, and the distance to the faces of a compound
    (some cadquery operations return a solid wrapped in one), which the query
    does not fill."""
    shape = _solids()[name].val()
    bb = shape.BoundingBox()
    shape = shape.Solids()[0].wrapped if as_solid else shape.wrapped
    rng = random.Random(1)
    lo = (bb.xmin - 1.5, bb.ymin - 1.5, bb.zmin - 1.5)
    hi = (bb.xmax + 1.5, bb.ymax + 1.5, bb.zmax + 1.5)
    points = [tuple(rng.uniform(lo[i], hi[i]) for i in range(3)) for _ in range(150)]
    # Points on the boundary: the solid's own vertices.
    points += [v.toTuple() for v in _solids()[name].vertices().vals()]
    cad = _CadSolid(1, shape)
    got = [cad.touches(p, EPS) for p in points]
    want = [_touches_whole_solid(shape, p) for p in points]
    assert got == want
    assert any(want) and not all(want)


def test_shallow_overlap_touches_and_a_gap_does_not():
    """A cube pushed 0.05 into another touches (every sample of one lies
    inside the other, none on its surface); a 0.05 gap does not."""

    def cubes(shift):
        asm = cq.Assembly()
        asm.add(cq.Workplane().box(10, 10, 10), name="a")
        asm.add(cq.Workplane().box(10, 10, 10).translate((10 + shift, 0, 0)), name="b")
        return asm

    for shift, separated in ((-0.05, []), (0.05, [(1, 2)])):
        c2y = CadToYamc()
        c2y.add_cadquery_object(cubes(shift), material_tags=["a", "b"])
        c2y.mesh(tolerance=0.1, imprint=False, overlaps="smaller_wins")
        assert c2y._separated == separated
