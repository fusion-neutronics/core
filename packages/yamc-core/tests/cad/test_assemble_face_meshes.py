"""The compiled assembly of per-face triangulations into one welded mesh."""

import pytest

pytest.importorskip("cadquery")
from yamc._core import assemble_face_meshes  # noqa: E402

SQUARE = ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 1, 2), (0, 2, 3)])


def test_degenerate_triangles_are_dropped_and_empty_faces_get_no_surface():
    collinear = ([(0, 0, 1), (1, 0, 1), (2, 0, 1)], [(0, 1, 2)])
    pinched = ([(0, 0, 2), (1, 0, 2), (1, 1, 2)], [(0, 0, 1), (0, 1, 2)])
    _, triangles, face_ids, surface_ids, face_surface = assemble_face_meshes(
        [(7, *SQUARE), (8, *collinear), (9, *pinched)]
    )
    # Face 8 keeps nothing, so face 9 is surface 2.
    assert face_surface == [(7, 1), (9, 2)]
    assert face_ids == [7, 7, 9]
    assert surface_ids == [1, 1, 2]
    assert len(triangles) == 3


def test_coincident_vertices_of_adjacent_faces_are_welded():
    # A second square sharing the first one's x = 1 edge, with its own copies
    # of the two shared corners.
    right = ([(1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0)], [(0, 1, 2), (0, 2, 3)])
    vertices, triangles, *_ = assemble_face_meshes([(1, *SQUARE), (2, *right)])
    assert len(vertices) == 6
    assert len({i for tri in triangles for i in tri}) == 6


def test_a_triangle_past_its_face_vertices_is_refused():
    with pytest.raises(ValueError, match="face 3"):
        assemble_face_meshes([(3, SQUARE[0], [(0, 1, 4)])])
