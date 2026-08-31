"""A negatively oriented tetrahedron is refused at load, not silently fixed.

Transport reads outward tet face normals off a fixed vertex ordering, which
only points outward for a positively oriented tet. An inverted tet made the
element walk pick an entry face as its exit and unstructured track-length
tallies read about 33 percent low (issue #316). yamc used to re-wind such a
tet on read, which hid the producing writer's bug; it now raises instead.
"""

import os
import tempfile

import pytest
import yamc
from yamc._core import mesh_to_arrow

# Unit-cube corners in (i + 2j + 4k) order, scaled to 10 cm.
SCALE = 10.0
VERTICES = [
    [SCALE * i, SCALE * j, SCALE * k]
    for k in (0, 1)
    for j in (0, 1)
    for i in (0, 1)
]

# Outward-wound boundary of that cube, two triangles per face.
TRIANGLES = [
    [0, 2, 3], [0, 3, 1],  # z = 0, normal -z
    [4, 5, 7], [4, 7, 6],  # z = 1, normal +z
    [0, 1, 5], [0, 5, 4],  # y = 0, normal -y
    [2, 6, 7], [2, 7, 3],  # y = 1, normal +y
    [0, 4, 6], [0, 6, 2],  # x = 0, normal -x
    [1, 3, 7], [1, 7, 5],  # x = 1, normal +x
]

# Kuhn triangulation of the cube: six positively oriented tets.
TETS = [
    [0, 1, 3, 7],
    [0, 1, 7, 5],
    [0, 2, 7, 3],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 7, 6],
]


def _write_cube(path, invert_tet=None):
    """Write the tetrahedralized cube, optionally inverting one tet.

    Swapping the last two vertices of a tet flips the sign of its volume,
    which is exactly the defect the loader has to catch.
    """
    tets = [list(t) for t in TETS]
    if invert_tet is not None:
        tets[invert_tet][2], tets[invert_tet][3] = (
            tets[invert_tet][3],
            tets[invert_tet][2],
        )
    mesh_to_arrow(
        path,
        [c for v in VERTICES for c in v],
        [i for t in TRIANGLES for i in t],
        [1] * len(TRIANGLES),
        [1] * len(TRIANGLES),
        tetrahedra=[i for t in tets for i in t],
        tet_volume_ids=[1] * len(tets),
        tet_physical_groups=[1] * len(tets),
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
    )


def _material():
    return yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="a")


def _path(name):
    return os.path.join(tempfile.mkdtemp(prefix="yamc_tet_orient_"), name)


def test_positively_oriented_mesh_loads():
    """The same mesh, unmodified, is accepted."""
    path = _path("cube.arrow")
    _write_cube(path)
    geom = yamc.MeshGeometry(path, {"a": _material()})
    assert geom.num_volumes == 1


def test_negative_tet_raises_with_an_actionable_message():
    """A single inverted tet is rejected, and the message names it."""
    path = _path("inverted.arrow")
    _write_cube(path, invert_tet=4)

    with pytest.raises(ValueError, match="tetrahedron 4"):
        yamc.MeshGeometry(path, {"a": _material()})

    try:
        yamc.MeshGeometry(path, {"a": _material()})
    except ValueError as exc:
        message = str(exc)
    assert "negatively oriented" in message
    assert "#316" in message
    assert "yamm" in message
    assert os.path.basename(path) in message
