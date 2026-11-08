"""Watertight + volume-vs-analytic coverage for every primitive surface type.

Mirrors (scaled down) cad-to-dagmc-mesher's surface zoo: each primitive OCC
surface (plane / cylinder / cone / sphere / torus) must mesh watertight with a
surface-enclosed volume matching the analytic value. These exercise the BRepMesh
primitive route; the torus in particular silently regressed (CDT-only) before
the hybrid routing, with no test to catch it -- this is that test.
"""

import math
from collections import defaultdict

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402


def _open_edges_and_volume(mesh, rel_tol=1e-6):
    """(open-edge count after vertex welding, enclosed volume).

    yamc keeps each face's parametric triangle winding (sense is carried by
    ``surface_volumes`` / ``face_solid_reversed``, not winding), so REVERSED
    faces point inward. We orient every triangle outward via the per-face
    reversed flag before the divergence-theorem sum, so the volume is the true
    enclosed volume regardless of yamc's winding convention.
    """
    verts = mesh.vertices
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
    edge_uses = defaultdict(int)
    for tri in mesh.triangles:
        a, b, c = remap[tri[0]], remap[tri[1]], remap[tri[2]]
        for x, y in ((a, b), (b, c), (c, a)):
            edge_uses[(min(x, y), max(x, y))] += 1
    open_edges = sum(1 for n in edge_uses.values() if n != 2)

    # face_id -> owning solid (single-solid models here).
    fid_to_solid = {fid: s for s, fids in mesh.solid_faces.items() for fid in fids}

    vol = 0.0
    for i, tri in enumerate(mesh.triangles):
        fid = mesh.triangle_face_ids[i]
        solid = fid_to_solid.get(fid)
        reversed_ = mesh.face_solid_reversed.get((solid, fid), False)
        a = verts[tri[0]]
        b = verts[tri[2] if reversed_ else tri[1]]
        c = verts[tri[1] if reversed_ else tri[2]]
        vol += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
    return open_edges, abs(vol)


# (name, solid factory, analytic volume). Factories build a fresh solid per
# test so the cadquery objects aren't shared/consumed across cases.
_PRIMITIVES = [
    ("cuboid", lambda: cq.Workplane().box(10, 10, 10), 10.0 ** 3),
    ("cylinder", lambda: cq.Workplane("XY").cylinder(20, 5), math.pi * 5 ** 2 * 20),
    ("cone", lambda: cq.Solid.makeCone(5, 0, 12), math.pi * 5 ** 2 * 12 / 3),
    ("sphere", lambda: cq.Workplane().sphere(10), 4 / 3 * math.pi * 10 ** 3),
    ("torus", lambda: cq.Solid.makeTorus(10, 3), 2 * math.pi ** 2 * 10 * 3 ** 2),
]


@pytest.mark.parametrize("name, factory, analytic", _PRIMITIVES)
def test_primitive_watertight_and_volume(name, factory, analytic):
    assy = cq.Assembly()
    assy.add(factory(), name=name)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, [name])
    mesh = c2y.mesh()

    open_edges, vol = _open_edges_and_volume(mesh)
    assert open_edges == 0, f"{name}: {open_edges} open edges (not watertight)"
    rel = abs(vol - analytic) / analytic
    assert rel < 0.02, (
        f"{name}: surface volume {vol:.2f} differs from analytic {analytic:.2f} "
        f"by {rel * 100:.2f}%"
    )


def _per_volume_open_edges(mesh, rel_tol=1e-6):
    """Open (boundary) edges of each solid's OWN boundary, after welding.

    The correct watertightness test for multi-material meshes: a shared internal
    face is non-manifold *globally* (3 faces meet at its perimeter), but each
    volume's boundary must still be a closed 2-manifold. Returns {solid_id: n}.
    """
    verts = mesh.vertices
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


_MULTISOLID = [
    ("two_touching_cubes", lambda: [
        cq.Workplane().box(10, 10, 10),
        cq.Workplane().box(10, 10, 10).translate((10, 0, 0)),
    ]),
    ("two_stacked_cylinders", lambda: [
        cq.Workplane("XY").cylinder(10, 5),
        cq.Workplane("XY").cylinder(10, 5).translate((0, 0, 10)),
    ]),
]


@pytest.mark.parametrize("name, factory", _MULTISOLID)
def test_multisolid_shared_face_watertight(name, factory):
    """Multi-solid models with a shared imprinted face: each volume's boundary
    must be closed (the shared face is a valid internal interface)."""
    solids = factory()
    assy = cq.Assembly()
    for i, solid in enumerate(solids):
        assy.add(solid, name=f"s{i}")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, [f"s{i}" for i in range(len(solids))])
    mesh = c2y.mesh()

    assert len(mesh.shared_faces) >= 1, f"{name}: expected an imprinted shared face"
    per_vol = _per_volume_open_edges(mesh)
    assert all(n == 0 for n in per_vol.values()), (
        f"{name}: per-volume open edges {per_vol} (a volume boundary is not closed)"
    )
