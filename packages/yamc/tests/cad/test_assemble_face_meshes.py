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


def _slab(thickness):
    """The six faces of a 1 x 1 x *thickness* box, each with its own corners."""
    def corner(i, j, k):
        return (float(i), float(j), thickness * k)

    quads = [
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    ]
    return [
        (fid, [corner(*c) for c in quad], [(0, 1, 2), (0, 2, 3)])
        for fid, quad in enumerate(quads, start=1)
    ]


def test_a_slab_thinner_than_the_weld_keeps_both_sides():
    # 1e-8 thick, a hundredth of the weld's 1e-6: only the copies of each
    # corner are merged, not the corners above and below each other.
    vertices, triangles, *_ = assemble_face_meshes(_slab(1e-8))
    assert len(vertices) == 8
    assert len(triangles) == 12


def test_a_slab_thinner_than_the_seam_tolerance_collapses():
    vertices, triangles, *_ = assemble_face_meshes(_slab(1e-8), seam_rel_tol=1e-7)
    assert len(vertices) == 4
    assert len(triangles) == 4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rel_tol": 0.0}, "rel_tol must be a positive number"),
        ({"rel_tol": float("nan")}, "rel_tol must be a positive number"),
        ({"seam_rel_tol": -1e-9}, "seam_rel_tol must be a non-negative number"),
    ],
)
def test_a_weld_tolerance_out_of_range_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        assemble_face_meshes([(1, *SQUARE)], **kwargs)


def test_a_vertex_that_is_not_finite_is_refused():
    vertices = [(0, 0, 0), (1, 0, 0), (1, float("inf"), 0), (0, 1, 0)]
    with pytest.raises(ValueError, match="face 4: triangle .* not finite"):
        assemble_face_meshes([(1, *SQUARE), (4, vertices, SQUARE[1])])
