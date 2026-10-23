"""Surface meshing tolerances."""

import pytest

cq = pytest.importorskip("cadquery")

from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402

from yamc.cad import CadToYamc  # noqa: E402

FINE = 0.01
COARSE = 0.5
FINE_ANGLE = 0.1
COARSE_ANGLE = 1.0


def _assembly(**parts):
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    return assembly


def _mesher(assembly):
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags="assembly_names")
    return c2y


def _sphere_in_shell():
    return _assembly(
        inner=cq.Workplane().sphere(5),
        outer=cq.Workplane().sphere(8).cut(cq.Workplane().sphere(5)),
    )


def _split_torus():
    """A torus cut in two by a plane through its axis: the halves share two
    discs, bounded by circles that the toroidal faces use too."""
    torus = cq.Workplane().add(cq.Solid.makeTorus(10, 3))
    half = cq.Workplane().box(40, 20, 20, centered=(True, False, True))
    return _assembly(inner=torus.intersect(half), outer=torus.cut(half))


@pytest.mark.parametrize("imprint", [True, False])
def test_meshing_again_with_other_values_meshes_afresh(imprint):
    """BRepMesh keeps a triangulation already on a shape when it is finer
    than asked, and a second mesh of the same assembly meshes the same shapes
    (imprinting leaves the faces it does not cut alone). Meshing again at
    coarser values gives what a fresh mesher gives, not the earlier mesh."""
    def triangles(c2y, **settings):
        return c2y.mesh(imprint=imprint, **settings).triangles

    settings = dict(tolerance=COARSE, angular_tolerance=COARSE_ANGLE)
    c2y = _mesher(_split_torus())
    triangles(c2y, tolerance=FINE, angular_tolerance=FINE_ANGLE)
    assert triangles(c2y, **settings) == triangles(_mesher(_split_torus()), **settings)


@pytest.mark.parametrize("imprint", [True, False])
def test_shapes_tessellated_before_are_meshed_afresh(imprint):
    """Shapes an STL export or a viewer already tessellated finely mesh as
    fresh ones do."""
    def tessellated():
        assembly = _sphere_in_shell()
        for child in assembly.children:
            BRepMesh_IncrementalMesh(child.obj.val().wrapped, 0.001, False, 0.05)
        return assembly

    settings = dict(tolerance=COARSE, angular_tolerance=COARSE_ANGLE)
    fresh = _mesher(_sphere_in_shell()).mesh(imprint=imprint, **settings)
    assert _mesher(tessellated()).mesh(imprint=imprint, **settings).triangles == fresh.triangles
