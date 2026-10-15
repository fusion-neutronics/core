"""Overlap gate for meshes made without imprinting.

Without imprinting, a particle leaving one solid is placed in whichever
touching solid contains it, and an overlap belongs to the solid the particle
was already in. That is harmless for the thin slivers two facetings of one CAD
surface leave between them, which is what the contact tolerance is sized for.
It is not harmless for a real overlap: a solid sitting inside another one
(a part embedded in a body that was never cut) is not touching it anywhere, so
a particle reaching it from the outside never enters it, and the tallies come
out confidently wrong with no lost particles. This gate makes such overlaps
loud at meshing time.
"""

from yamc._core import solid_overlaps


class OverlappingSolidsError(ValueError):
    """Two solids overlap deeper than the contact tolerance."""


def _solid_triangles(mesh):
    """Each solid's closed boundary, as ``{solid_id: [tri, ...]}``."""
    face_tris = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        face_tris.setdefault(fid, []).append(tri)
    return {
        sid: [tri for fid in fids for tri in face_tris.get(fid, [])]
        for sid, fids in mesh.solid_faces.items()
    }


def check_solid_overlaps(mesh, contact_tolerance):
    """Solid pairs of *mesh* that overlap deeper than *contact_tolerance*.

    Args:
        mesh: SurfaceMesh from the assembler.
        contact_tolerance: Penetration (3D units) up to which two solids
            count as touching.

    Returns:
        List of human-readable problem strings (empty when no two solids
        overlap deeper than the tolerance).
    """
    solids = sorted(_solid_triangles(mesh).items())

    def name(sid):
        if 0 < sid <= len(mesh.material_tags):
            return f"solid {sid} ({mesh.material_tags[sid - 1]!r})"
        return f"solid {sid}"

    return [
        f"{name(a)} and {name(b)} overlap {depth:.4g} deep "
        f"(roughly {100 * exposure:.2g}% of tracks cross the overlap)"
        for a, b, depth, exposure in solid_overlaps(
            mesh.vertices, solids, contact_tolerance
        )
    ]
