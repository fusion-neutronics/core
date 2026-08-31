"""Watertightness regression for the scene surface mesher.

No prior cad test asserted closed surfaces directly -- transport flux only
catches gross leaks. These assert zero open (boundary) edges after welding
coincident vertices, across primitives that exercise the different face kinds:
planar (cube), ruled + periodic seam (cylinder), and doubly-curved polar
(sphere, which falls back to the structured-grid path).
"""

from collections import defaultdict

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402


def _open_edge_count(vertices, triangles, rel_tol=1e-6):
    """Count edges used by != 2 triangles after welding coincident vertices."""
    extents = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
    quantum = (max(extents) or 1.0) * rel_tol
    key_to_id = {}
    remap = [
        key_to_id.setdefault(
            (round(v[0] / quantum), round(v[1] / quantum), round(v[2] / quantum)),
            len(key_to_id),
        )
        for v in vertices
    ]
    edge_uses = defaultdict(int)
    for tri in triangles:
        a, b, c = remap[tri[0]], remap[tri[1]], remap[tri[2]]
        for x, y in ((a, b), (b, c), (c, a)):
            edge_uses[(min(x, y), max(x, y))] += 1
    return sum(1 for uses in edge_uses.values() if uses != 2)


@pytest.mark.parametrize(
    "name, solid",
    [
        ("cube", cq.Workplane("XY").box(10, 10, 10)),
        ("cylinder", cq.Workplane("XY").cylinder(10, 5)),
        ("sphere", cq.Workplane("XY").sphere(5)),
    ],
)
def test_surface_mesh_watertight(name, solid):
    assy = cq.Assembly()
    assy.add(solid, name=name)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=[name])
    mesh = c2y.mesh()

    assert len(mesh.triangles) > 0, f"{name}: no triangles produced"
    opens = _open_edge_count(mesh.vertices, mesh.triangles)
    assert opens == 0, f"{name} surface mesh has {opens} open edges (not watertight)"
