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


_SQUARE_WITH_HOLE = """
import sys
import yamc.cad as cad_to_yamc

def wire(size, offset, first_id):
    v = [(offset, offset), (offset + size, offset), (offset + size, offset + size), (offset, offset + size)]
    return [
        {"edge_id": first_id + i, "uv_points": [list(v[(i + 1) % 4]), list(v[i])], "reversed": True}
        for i in range(4)
    ]

target = float(sys.argv[1])
out = cad_to_yamc.mesh_face(
    face_id=5, boundary_edges=wire(1.0, 0.0, 1), holes=[wire(0.2, 0.4, 5)],
    mesh_mode=f"fine:{target}",
)
# The boundary vertices: each wire's points as traversed, shared corners once.
pts = []
for w in (wire(1.0, 0.0, 1), wire(0.2, 0.4, 5)):
    loop = []
    for e in w:
        for p in reversed(e["uv_points"]):
            if not loop or loop[-1] != p:
                loop.append(p)
    if loop[0] == loop[-1]:
        loop.pop()
    pts += loop
pts = pts[: out.num_boundary_vertices] + [list(p) for p in out.interior_uv_vertices]
area, longest, in_hole = 0.0, 0.0, 0
for a, b, c in out.triangles:
    (ax, ay), (bx, by), (cx, cy) = pts[a], pts[b], pts[c]
    area += 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    in_hole += 0.4 < (ax + bx + cx) / 3 < 0.6 and 0.4 < (ay + by + cy) / 3 < 0.6
    for (px, py), (qx, qy) in (((ax, ay), (bx, by)), ((bx, by), (cx, cy)), ((cx, cy), (ax, ay))):
        longest = max(longest, ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5)
print(out.num_boundary_vertices, area, longest, in_hole)
"""


@pytest.mark.parametrize("target", [0.5, 0.3, 0.1, 0.05])
def test_mesh_face_fine_with_a_hole_reaches_the_target(target):
    """A unit square with a square hole in fine mode (cad-to-dagmc-mesher#184).

    At 0.3 the long-edge split looped forever and at 0.1 refinement ran to its
    iteration cap. The face stands alone, so refinement now splits a boundary
    segment it comes too close to, and the mesh reaches the target. Run in a
    subprocess with a time limit so a regression fails rather than hangs.
    """
    import subprocess
    import sys

    run = subprocess.run(
        [sys.executable, "-c", _SQUARE_WITH_HOLE, str(target)],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    n_boundary, area, longest, in_hole = run.stdout.split()
    assert int(n_boundary) == 8
    assert int(in_hole) == 0
    assert float(area) == pytest.approx(0.96, abs=1e-9)
    assert float(longest) <= 1.5 * target
