"""MeshGeometry.add_tets: tet-mesh volumes of a surface mesh, one at a time or
together, conformally, without changing the geometry it is called on."""

from collections import Counter

import numpy as np
import pytest

import yamc

cq = pytest.importorskip("cadquery")


def _assembly():
    sphere = cq.Workplane("XY").sphere(3.0)
    box = cq.Workplane("XY").box(10, 10, 10).cut(sphere)
    assembly = cq.Assembly()
    assembly.add(sphere, name="sphere")
    assembly.add(box, name="box")
    return assembly


def _surface():
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.8)
    return yamc.MeshGeometry.from_cadquery(
        _assembly(), {"sphere": iron, "box": iron}, tolerance=0.2
    )


def _faces(vertices, cells, k):
    v = [tuple(p) for p in vertices]
    out = Counter()
    for c in cells:
        if k == 3:
            out[tuple(sorted(v[i] for i in c))] += 1
        else:
            for drop in range(4):
                out[tuple(sorted(v[c[j]] for j in range(4) if j != drop))] += 1
    return out


def _assert_conformal(mesh, meshed):
    vertices = np.array(mesh.vertices)
    tets = np.array(mesh.tetrahedra)
    ids = np.array([-1 if v is None else v for v in mesh.tet_volume_ids])
    assert max(_faces(mesh.vertices, tets, 4).values()) <= 2
    for v, name in enumerate(mesh.volume_names):
        if name not in meshed:
            assert not (ids == v).any()
            continue
        own = _faces(mesh.vertices, tets[ids == v], 4)
        boundary = {f for f, n in own.items() if n == 1}
        surface = set(
            _faces(mesh.vertices, [mesh.triangles[i] for i in mesh.volume_triangle_indices(v)], 3)
        )
        assert boundary == surface, name
        a, b, c, d = (vertices[tets[ids == v][:, i]] for i in range(4))
        volume = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6
        assert volume == pytest.approx(mesh.volume_measures[v], rel=1e-9)


def test_surface_mesh_then_tets():
    surface = _surface()
    assert surface.tetrahedra == []
    assert set(surface.mesh_report) >= {"overlaps", "face_volumes"}

    meshed = surface.add_tets(["sphere", "box"], target_edge_length=1.5)
    _assert_conformal(meshed, {"sphere", "box"})
    assert meshed.volume_names == surface.volume_names
    # the refined surface encloses the same volumes, to rounding
    assert meshed.volume_measures == pytest.approx(surface.volume_measures, rel=1e-12)
    assert set(meshed.mesh_report["conform"]) == {
        "edges_split", "cones", "surface_triangles_refined", "conflicts",
        "skipped_solids", "reverted",
    }
    # the geometry it was called on is unchanged
    assert surface.tetrahedra == []


def test_one_volume_at_a_time():
    surface = _surface()
    sphere = surface.add_tets(["sphere"], target_edge_length=1.0)
    _assert_conformal(sphere, {"sphere"})
    both = sphere.add_tets(["box"], target_edge_length={"box": 2.0})
    _assert_conformal(both, {"sphere", "box"})


def test_default_is_every_volume_without_tets():
    meshed = _surface().add_tets(target_edge_length=1.5)
    _assert_conformal(meshed, {"sphere", "box"})


def test_bad_volumes_raise():
    surface = _surface()
    with pytest.raises(ValueError, match="no volume named 'nope'"):
        surface.add_tets(["nope"])
    with pytest.raises(ValueError, match="already has tets"):
        surface.add_tets(["sphere"], 1.5).add_tets(["sphere"])


def test_a_round_trip_keeps_the_tets(tmp_path):
    meshed = _surface().add_tets(["sphere", "box"], target_edge_length=1.5)
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.8)
    again = yamc.MeshGeometry.from_arrow(meshed.to_arrow(), {"sphere": iron, "box": iron})
    assert again.tetrahedra == meshed.tetrahedra
    assert again.tet_volume_names == meshed.tet_volume_names
    assert again.mesh_report is None
