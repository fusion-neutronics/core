"""BRepMesh gives the same triangulation on any number of threads.

Faces are meshed in parallel on a fixed pool (``_brepmesh.MESH_THREADS``),
which only changes time and memory if the triangulation does not depend on the
thread count. These mesh assemblies whose solids share curved faces and edges,
imprinted and not, on several counts, each twice, and require identical meshes.
"""
import cadquery as cq
import pytest
from yamc.cad import CadToYamc, _brepmesh


def _assembly():
    """Nested spherical shells, a torus around them and a box cut by a
    cylinder: shared curved faces, seams and poles, and a solid touching
    two others."""
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
    return asm


def _mesh(threads, imprint, monkeypatch):
    monkeypatch.setattr(_brepmesh, "MESH_THREADS", threads)
    m = CadToYamc()
    asm = _assembly()
    m.add_cadquery_object(asm, material_tags=[c.name for c in asm.children])
    mesh = m.mesh(tolerance=0.2, angular_tolerance=0.3, imprint=imprint)
    return [tuple(v) for v in mesh.vertices], [tuple(t) for t in mesh.triangles]


@pytest.mark.parametrize("imprint", [True, False], ids=["imprinted", "not imprinted"])
def test_the_mesh_does_not_depend_on_the_thread_count(imprint, monkeypatch):
    reference = _mesh(1, imprint, monkeypatch)
    for threads in (1, 2, 4, 8):
        assert _mesh(threads, imprint, monkeypatch) == reference, threads
