"""Overlaps between the solids of a surface mesh.

The imprint shares the surfaces where solids coincide, but it does not split
solids whose surfaces cross: a rod poking out of a box comes back as the whole
box and the whole rod, overlapping. Transport then gives the overlap to
whichever solid a particle was already in, and the tallies come out
confidently wrong with no lost particles. So :meth:`CadToYamc.mesh` checks the
mesh for overlaps deeper than twice the deflection and refuses them by
default.
"""

from yamc._core import solid_overlaps


def _solid_triangles(mesh):
    """Each solid's closed boundary, as ``{solid_id: [tri, ...]}``."""
    face_tris = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        face_tris.setdefault(fid, []).append(tri)
    return {
        sid: [tri for fid in fids for tri in face_tris.get(fid, [])]
        for sid, fids in mesh.solid_faces.items()
    }


def _name(mesh, sid):
    if 0 < sid <= len(mesh.material_tags):
        return f"solid {sid} ({mesh.material_tags[sid - 1]!r})"
    return f"solid {sid}"


def find_overlaps(mesh, depth_tolerance):
    """Solid pairs of *mesh* overlapping deeper than *depth_tolerance*, as
    ``[(solid_a, solid_b, max_depth, exposure), ...]``."""
    solids = sorted(_solid_triangles(mesh).items())
    return solid_overlaps(mesh.vertices, solids, depth_tolerance)


def describe(mesh, found):
    """Human-readable lines for :func:`find_overlaps` results."""
    return [
        f"{_name(mesh, a)} and {_name(mesh, b)} overlap {depth:.4g} deep "
        f"(roughly {100 * exposure:.2g}% of tracks cross the overlap)"
        for a, b, depth, exposure in found
    ]
