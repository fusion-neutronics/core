"""A mesh volume that lists the same facet twice is refused at load (issue #515).

A copy of a facet is counted twice in the volume's measure and when locating
points, so particles born in the volume are lost. Two facets in the same place
facing opposite ways say the volume lies on both sides of one face. Both are
refused with a message naming the triangles, rather than loading and giving
wrong volumes and lost particles.
"""
import os

import pytest
import yamc

VERTICES = [
    [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
    [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
]
QUADS = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
CUBE = [t for a, b, c, d in QUADS for t in ([a, b, c], [a, c, d])]  # wound outward


def _load(tmp_path, triangles):
    path = os.path.join(tmp_path, "cube.arrow")
    yamc.mesh_to_arrow(
        path,
        [float(x) for p in VERTICES for x in p],
        [i for t in triangles for i in t],
        [1] * len(triangles),
        [1] * len(triangles),
        physical_groups_json='{"1": {"name": "mat:void", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
        imprinted=True,
    )
    return yamc.MeshGeometry(path, {})


def test_a_cube_with_each_facet_once_loads(tmp_path):
    assert _load(tmp_path, CUBE).volume_measures[0] == pytest.approx(1000.0)


def test_a_boundary_listed_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="lists the same facet twice: triangles 0 and 12"):
        _load(tmp_path, CUBE + CUBE)


def test_one_copied_facet_is_refused(tmp_path):
    with pytest.raises(ValueError, match="lists the same facet twice: triangles 5 and 12"):
        _load(tmp_path, CUBE + [CUBE[5]])


def test_a_reversed_copy_is_refused(tmp_path):
    a, b, c = CUBE[5]
    with pytest.raises(ValueError, match="two facets in the same place facing opposite ways"):
        _load(tmp_path, CUBE + [[a, c, b]])
