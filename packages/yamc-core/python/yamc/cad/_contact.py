"""Which solids of a mesh made without imprinting really touch.

Discontinuous tracking treats two solids whose meshes come within the contact
tolerance as touching, and gives a gap thinner than that to the solid ahead:
that is how the thin gaps two facetings of one CAD surface leave between them
are closed. A gap that is in the CAD, narrower than the tolerance (an assembly
clearance meshed coarsely), would be filled the same way. The CAD settles it:
a mesh vertex lies on its solid's CAD surface, so its exact distance to the
other CAD solid is zero where the two touch and the real gap where they do
not. Pairs separated in the CAD are written to the mesh file, and relocation
never fills the gap between them.
"""

from yamc._core import solid_contact_samples

from ._overlap import _solid_triangles


def find_separated(mesh, solid_shapes, contact_tolerance):
    """Solid pairs within *contact_tolerance* in *mesh* whose CAD solids do not
    touch, as 1-based ``(a, b)`` solid ids.

    A pair touches when any of its sample points (mesh vertices of one solid
    near the other) lies within a thousandth of the tolerance of the other CAD
    solid, inside or on it. A pair whose meshes share a vertex, or only come
    close between vertices, has no samples and counts as touching.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    solids = sorted(_solid_triangles(mesh).items())
    eps = 1e-3 * contact_tolerance
    separated = []
    for a, b, points in solid_contact_samples(mesh.vertices, solids, contact_tolerance):
        shape = solid_shapes.get(b)
        if not points or shape is None:
            continue
        shape = getattr(shape, "wrapped", shape)
        touching = False
        for p in points:
            dist = BRepExtrema_DistShapeShape(
                BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex(), shape
            )
            if not dist.IsDone() or dist.Value() <= eps:
                touching = True
                break
        if not touching:
            separated.append((min(a, b), max(a, b)))
    return separated
