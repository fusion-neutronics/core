import pytest

pytest.importorskip("cadquery")
import yamc.cad as cad_to_yamc  # noqa: E402


def test_mesh_face_coarse():
    result = cad_to_yamc.mesh_face(
        face_id=1,
        boundary_edges=[
            {"edge_id": 1, "uv_points": [[0.0, 0.0], [1.0, 0.0]], "reversed": False},
            {"edge_id": 2, "uv_points": [[1.0, 0.0], [1.0, 1.0]], "reversed": False},
            {"edge_id": 3, "uv_points": [[1.0, 1.0], [0.0, 1.0]], "reversed": False},
            {"edge_id": 4, "uv_points": [[0.0, 1.0], [0.0, 0.0]], "reversed": False},
        ],
        mesh_mode="coarse",
    )
    assert result.face_id == 1
    assert result.num_boundary_vertices == 4
    assert len(result.triangles) == 2
    assert len(result.interior_uv_vertices) == 0


def test_mesh_face_fine():
    result = cad_to_yamc.mesh_face(
        face_id=2,
        boundary_edges=[
            {"edge_id": 1, "uv_points": [[0.0, 0.0], [10.0, 0.0]], "reversed": False},
            {"edge_id": 2, "uv_points": [[10.0, 0.0], [10.0, 10.0]], "reversed": False},
            {"edge_id": 3, "uv_points": [[10.0, 10.0], [0.0, 10.0]], "reversed": False},
            {"edge_id": 4, "uv_points": [[0.0, 10.0], [0.0, 0.0]], "reversed": False},
        ],
        mesh_mode="fine:2.0",
    )
    assert result.face_id == 2
    assert len(result.triangles) > 2
    assert len(result.interior_uv_vertices) > 0


def test_mesh_face_with_curved_edge():
    """Edge with multiple polyline points (discretized curve)."""
    import math

    # Half-circle arc as a polyline
    n = 10
    arc_points = [
        [math.cos(math.pi * i / n), math.sin(math.pi * i / n)] for i in range(n + 1)
    ]

    result = cad_to_yamc.mesh_face(
        face_id=3,
        boundary_edges=[
            {"edge_id": 1, "uv_points": arc_points, "reversed": False},
            {
                "edge_id": 2,
                "uv_points": [arc_points[-1], arc_points[0]],
                "reversed": False,
            },
        ],
        mesh_mode="coarse",
    )
    assert len(result.triangles) >= 9  # At least n-1 triangles for the arc


def test_mesh_face_reversed_edge():
    result = cad_to_yamc.mesh_face(
        face_id=4,
        boundary_edges=[
            {"edge_id": 1, "uv_points": [[0.0, 0.0], [1.0, 0.0]], "reversed": False},
            {
                "edge_id": 2,
                "uv_points": [[1.0, 1.0], [1.0, 0.0]],
                "reversed": True,
            },  # stored reversed
            {"edge_id": 3, "uv_points": [[1.0, 1.0], [0.0, 1.0]], "reversed": False},
            {"edge_id": 4, "uv_points": [[0.0, 1.0], [0.0, 0.0]], "reversed": False},
        ],
        mesh_mode="coarse",
    )
    assert len(result.triangles) == 2
