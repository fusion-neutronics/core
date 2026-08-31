"""Local-only GEOUNED watertightness sweep.

Mirrors cad-to-dagmc-mesher's ``test_local_regression``: mesh real GEOUNED
``inputSTEP`` CAD and assert each solid's boundary is watertight (per-volume,
so internal shared faces don't count as leaks). Skipped unless the GEOUNED CAD
is on disk; set ``YAMM_GEOUNED_DIR`` to override the path.

Dirty-CAD files carry a per-volume open-edge budget (the residual shared-edge
discretization mismatch that cad-to-dagmc-mesher also only meets with a budget);
the budgets exist to catch *regressions*, not to claim perfection.
"""

import os
from collections import defaultdict

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402

_GEOUNED_DIR = os.path.expanduser(
    os.environ.get("YAMM_GEOUNED_DIR", "~/GEOUNED/testing/inputSTEP")
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_GEOUNED_DIR),
    reason=f"GEOUNED inputSTEP dir not present ({_GEOUNED_DIR})",
)

# relpath -> per-volume open-edge budget (0 = must be fully watertight).
_BUDGETS = {
    "cylBox.stp": 0,
    "placa.stp": 0,
    "biseau.stp": 0,
    "FWTBM1.step": 0,
    "tubos.stp": 0,
    "Misc/RJ24.stp": 0,
    "large/Triangle.stp": 130,  # dirty CAD: ~108 residual (cf ~/yamm budget 122)
}


def _per_volume_open_edges(mesh, rel_tol=1e-6):
    verts = mesh.vertices
    if not verts:
        return {None: 1}
    extents = [max(v[i] for v in verts) - min(v[i] for v in verts) for i in range(3)]
    quantum = (max(extents) or 1.0) * rel_tol
    key_to_id = {}
    remap = [
        key_to_id.setdefault(
            (round(v[0] / quantum), round(v[1] / quantum), round(v[2] / quantum)),
            len(key_to_id),
        )
        for v in verts
    ]
    per_solid = {}
    for sid, fids in mesh.solid_faces.items():
        fset = set(fids)
        edge_uses = defaultdict(int)
        for i, tri in enumerate(mesh.triangles):
            if mesh.triangle_face_ids[i] in fset:
                a, b, c = remap[tri[0]], remap[tri[1]], remap[tri[2]]
                for x, y in ((a, b), (b, c), (c, a)):
                    edge_uses[(min(x, y), max(x, y))] += 1
        per_solid[sid] = sum(1 for n in edge_uses.values() if n != 2)
    return per_solid


@pytest.mark.parametrize("relpath, budget", sorted(_BUDGETS.items()))
def test_geouned_watertight(relpath, budget):
    path = os.path.join(_GEOUNED_DIR, relpath)
    if not os.path.isfile(path):
        pytest.skip(f"{relpath} not present")

    solids = cq.importers.importStep(path).solids().vals()
    assert solids, f"{relpath}: no solids imported"
    tags = [f"m{i}" for i in range(len(solids))]
    assy = cq.Assembly()
    for tag, solid in zip(tags, solids):
        assy.add(solid, name=tag)

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, tags)
    mesh = c2y.mesh()

    total = sum(_per_volume_open_edges(mesh).values())
    assert total <= budget, (
        f"{relpath}: {total} per-volume open edges exceeds budget {budget} "
        f"({len(solids)} solids)"
    )
