"""Meshing gives the same triangulation on any number of threads.

Faces are meshed in parallel on ``mesh_threads`` threads (all by default),
OCC BRepMesh and the Rust scene CDT alike, which only changes time and memory
if the triangulation does not depend on the thread count. These mesh
assemblies whose solids share curved faces and edges, imprinted and not, on
1, 2, 4 and 8 threads, and require identical meshes.
"""
import pytest

cq = pytest.importorskip("cadquery")
from yamc._core import solid_self_intersections  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad.mesher import _route_faces  # noqa: E402

_PROFILE = [(0, 0), (12, -3), (20, 6), (10, 14), (-4, 8)]


def _assembly():
    """Nested spherical shells, a torus around them and a box cut by a
    cylinder: shared curved faces, seams and poles, and a solid touching
    two others. Two stacked extrusions of a spline profile, sharing their
    cap, are not primitive, so their faces go to the scene CDT."""
    inner = cq.Workplane().sphere(10)
    shell = cq.Workplane().sphere(14).cut(inner)
    torus = cq.Workplane().add(cq.Solid.makeTorus(30, 6))
    box = cq.Workplane().box(20, 20, 20).translate((0, 0, 40))
    rod = cq.Workplane().cylinder(30, 4).translate((0, 0, 40))
    asm = cq.Assembly()
    asm.add(inner, name="inner")
    asm.add(shell, name="shell")
    asm.add(torus, name="torus")
    asm.add(box.cut(rod), name="box")
    asm.add(rod.intersect(cq.Workplane().box(20, 20, 20).translate((0, 0, 40))), name="rod")
    for k in range(2):
        prism = cq.Workplane().spline(_PROFILE, periodic=True).close().extrude(10)
        asm.add(prism.translate((40, 0, 10 * k)), name=f"prism{k}")
    return asm


def test_both_meshers_are_exercised():
    m = CadToYamc()
    asm = _assembly()
    m.add_cadquery_object(asm, material_tags=[c.name for c in asm.children])
    m.mesh(tolerance=0.2, angular_tolerance=0.3, mesh_threads=1)
    brep, scene = _route_faces(m._processed)
    assert brep and scene


def _mesh(threads, imprint):
    m = CadToYamc()
    asm = _assembly()
    m.add_cadquery_object(asm, material_tags=[c.name for c in asm.children])
    mesh = m.mesh(
        tolerance=0.2, angular_tolerance=0.3, imprint=imprint, mesh_threads=threads
    )
    return [tuple(v) for v in mesh.vertices], [tuple(t) for t in mesh.triangles]


@pytest.mark.parametrize("imprint", [True, False], ids=["imprinted", "not imprinted"])
def test_the_mesh_does_not_depend_on_the_thread_count(imprint):
    reference = _mesh(1, imprint)
    for threads in (2, 4, 8, "all"):
        assert _mesh(threads, imprint) == reference, threads


@pytest.mark.parametrize("threads", [0, -1, 2.0, "some"])
def test_mesh_threads_must_be_all_or_a_positive_integer(threads):
    m = CadToYamc()
    asm = _assembly()
    m.add_cadquery_object(asm, material_tags=[c.name for c in asm.children])
    with pytest.raises(ValueError, match="mesh_threads"):
        m.mesh(mesh_threads=threads)


def test_the_rust_bindings_refuse_zero_threads():
    with pytest.raises(ValueError, match="threads"):
        solid_self_intersections([[0.0, 0.0, 0.0]], [], threads=0)
