"""Overlaps between solids of a mesh made without imprinting.

Without imprinting, a particle leaving one solid is placed in whichever
touching solid contains it, and an overlap belongs to the solid the particle
was already in. That is harmless for the thin slivers two facetings of one CAD
surface leave between them, which is what the contact tolerance is sized for.
It is not harmless for a real overlap: a solid sitting inside another one
(a part embedded in a body that was never cut) is not touching it anywhere, so
a particle reaching it from the outside never enters it, and the tallies come
out confidently wrong with no lost particles.

So real overlaps are either refused (the default) or resolved by rank: with
``overlaps="smaller_wins"`` each overlapping pair is recorded in the mesh
file, the smaller solid takes the shared region wherever a particle comes
from, and the larger solid's volume is reduced by what it gives up.
"""

import warnings

from yamc._core import solid_overlaps, solid_shared_volume


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


def _solid_volumes(mesh):
    """Each solid's mesh volume, by the divergence theorem over its faces,
    each face signed by whether it is reversed in that solid."""
    verts = mesh.vertices
    face_vol = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        bxc = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        face_vol[fid] = face_vol.get(fid, 0.0) + (
            a[0] * bxc[0] + a[1] * bxc[1] + a[2] * bxc[2]
        ) / 6.0
    return {
        sid: abs(sum(
            (-1.0 if mesh.face_solid_reversed.get((sid, fid)) else 1.0)
            * face_vol.get(fid, 0.0)
            for fid in fids
        ))
        for sid, fids in mesh.solid_faces.items()
    }


def _name(mesh, sid):
    if 0 < sid <= len(mesh.material_tags):
        return f"solid {sid} ({mesh.material_tags[sid - 1]!r})"
    return f"solid {sid}"


def find_overlaps(mesh, contact_tolerance):
    """Solid pairs of *mesh* overlapping deeper than *contact_tolerance*, as
    ``[(solid_a, solid_b, max_depth, exposure), ...]``."""
    solids = sorted(_solid_triangles(mesh).items())
    return solid_overlaps(mesh.vertices, solids, contact_tolerance)


def describe(mesh, found):
    """Human-readable lines for :func:`find_overlaps` results."""
    return [
        f"{_name(mesh, a)} and {_name(mesh, b)} overlap {depth:.4g} deep "
        f"(roughly {100 * exposure:.2g}% of tracks cross the overlap)"
        for a, b, depth, exposure in found
    ]


def resolve_smaller_wins(mesh, found, contact_tolerance, samples=200_000):
    """Rank each overlapping pair so the smaller solid takes the shared region.

    Returns ``(pairs, volume_measures, lines)``: ``(low, high)`` 1-based solid
    pairs (``high`` takes the overlap), the volume of every solid in solid
    order with what the larger solid of each pair gives up taken off, and a
    report line per pair.
    """
    tris = _solid_triangles(mesh)
    volumes = _solid_volumes(mesh)
    pairs = []
    lines = []
    given_up = {sid: 0.0 for sid in volumes}
    for a, b, depth, _exposure in found:
        low, high = sorted((a, b), key=lambda s: (-volumes[s], s))
        shared, err, embedded = solid_shared_volume(
            mesh.vertices, tris[low], tris[high], contact_tolerance, samples
        )
        if embedded is not None:
            inner = (low, high)[embedded]
            shared, err = volumes[inner], 0.0
            how = f"{_name(mesh, inner)} lies wholly inside"
        else:
            how = f"shared {shared:.4g} +- {err:.2g} (sampled)"
        given_up[low] += shared
        pairs.append((low, high))
        lines.append(
            f"{_name(mesh, high)} takes its overlap with {_name(mesh, low)} "
            f"({depth:.4g} deep; {how}; volume of {_name(mesh, low)} "
            f"{volumes[low]:.6g} -> {volumes[low] - given_up[low]:.6g})"
        )
    # A solid giving up overlaps to two solids that overlap each other has the
    # triple region taken off twice.
    winners = {}
    for low, high in pairs:
        winners.setdefault(low, set()).add(high)
    paired = {frozenset(p) for p in pairs}
    for low, highs in winners.items():
        highs = sorted(highs)
        for i, h1 in enumerate(highs):
            for h2 in highs[i + 1:]:
                if frozenset((h1, h2)) in paired:
                    warnings.warn(
                        f"{_name(mesh, h1)} and {_name(mesh, h2)} overlap each "
                        f"other and both take an overlap from {_name(mesh, low)}: "
                        "its corrected volume counts their common region twice",
                        stacklevel=3,
                    )
    measures = [max(volumes[sid] - given_up[sid], 0.0) for sid in sorted(volumes)]
    return pairs, measures, lines
