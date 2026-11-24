"""Tet meshes of imprinted solids are conformal (issue #706).

Each solid's tet mesher can add points on its boundary, so before the repair
its tets presented a refined surface where the other solid sharing that
surface did not (hanging faces), and the tets' boundary faces did not match
the surface triangles transport tracks against. After meshing, every tet
volume's boundary faces must be exactly its surface triangles, and the solids'
tets must meet face to face on the surfaces they share.
"""

from collections import Counter

import numpy as np
import pytest

import yamc

cq = pytest.importorskip("cadquery")
from yamc.cad.mesher import _CadMesher  # noqa: E402


def _faces(vertices, cells, k):
    """The faces of *cells* (k = 3 for triangles, 4 for tets), as sorted
    coordinate tuples, counted."""
    v = [tuple(p) for p in vertices]
    out = Counter()
    for c in cells:
        if k == 3:
            out[tuple(sorted(v[i] for i in c))] += 1
        else:
            for drop in range(4):
                out[tuple(sorted(v[c[j]] for j in range(4) if j != drop))] += 1
    return out


def _check(assembly, tags, edge, tmp_path):
    mesher = _CadMesher()
    mesher.add_cadquery_object(assembly, material_tags=tags)
    mesher.mesh(tet_volumes=tags, target_edge_length=edge)
    mesher.to_arrow(tmp_path / "m.arrow")
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.8)
    mesh = yamc.MeshGeometry.from_arrow(str(tmp_path / "m.arrow"), {t: iron for t in tags})

    vertices = mesh.vertices
    tets = np.array(mesh.tetrahedra)
    volume_ids = np.array(mesh.tet_volume_ids)
    all_tet_faces = _faces(vertices, tets, 4)
    for v, name in enumerate(mesh.volume_names):
        own = _faces(vertices, tets[volume_ids == v], 4)
        boundary = {f for f, n in own.items() if n == 1}
        surface = set(_faces(vertices, [mesh.triangles[i] for i in mesh.volume_triangle_indices(v)], 3))
        assert boundary == surface, f"{name}: tet boundary faces differ from its surface"
        # tet volume equals the surface's
        a, b, c, d = (np.array(vertices)[tets[volume_ids == v][:, i]] for i in range(4))
        tet_volume = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6
        assert tet_volume == pytest.approx(mesh.volume_measures[v], rel=1e-9)
    # every face is in one tet (a boundary) or two (inside, or on a shared
    # surface): a face in three would be overlapping tets
    assert max(all_tet_faces.values()) <= 2


def test_sphere_in_box(tmp_path):
    sphere = cq.Workplane("XY").sphere(4.0)
    box = cq.Workplane("XY").box(12, 12, 12).cut(sphere)
    assembly = cq.Assembly()
    assembly.add(sphere, name="sphere")
    assembly.add(box, name="box")
    _check(assembly, ["sphere", "box"], 1.04, tmp_path)


def test_touching_cuboids(tmp_path):
    a = cq.Workplane("XY").box(4, 4, 4)
    b = cq.Workplane("XY").box(4, 4, 4).translate((4, 0, 0))
    assembly = cq.Assembly()
    assembly.add(a, name="a")
    assembly.add(b, name="b")
    _check(assembly, ["a", "b"], 0.8, tmp_path)
