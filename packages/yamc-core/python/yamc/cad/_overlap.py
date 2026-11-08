"""Overlaps between the solids of a surface mesh.

The imprint shares the surfaces where solids coincide, but it does not split
solids whose surfaces cross: a rod poking out of a box comes back as the whole
box and the whole rod, overlapping. Transport then gives the overlap to
whichever solid a particle was already in, and the tallies come out
confidently wrong with no lost particles. So :meth:`CadToYamc.mesh` checks the
mesh for overlaps deeper than the two solids' facetings explain (twice the
larger of their deflections) and refuses them by default.

Without imprinting, a particle leaving one solid is placed in whichever
touching solid contains it, and an overlap belongs to the solid the particle
was already in. That is harmless for the thin slivers two facetings of one CAD
surface leave between them, which is what the contact tolerance is sized for.
It is not harmless for a real overlap: a solid sitting inside another one
(a part embedded in a body that was never cut) is not touching it anywhere, so
a particle reaching it from the outside never enters it. Overlaps deeper than
the facetings explain, or than a contact tolerance the caller gives, are
refused by default, or resolved by rank: with
``overlaps="smaller_wins"`` each overlapping pair is recorded in the mesh
file, the smaller solid takes the shared region wherever a particle comes
from, and the larger solid's volume is reduced by what it gives up.
"""

import warnings

from yamc._core import solid_overlaps, solid_shared_volume


def _solid_triangles(mesh):
    """Each solid's closed boundary, as ``{solid_id: [tri, ...]}``."""
    face_tris = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        face_tris.setdefault(fid, []).append(tri)
    return {
        sid: [tri for fid in fids for tri in face_tris.get(fid, [])]
        for sid, fids in mesh.solid_faces.items()
    }


def _own_vertices(vertices, tris):
    """*tris* re-indexed over just the vertices they use, as
    ``(vertices, tris)``."""
    index = {}
    local = [[index.setdefault(i, len(index)) for i in tri] for tri in tris]
    own = [None] * len(index)
    for i, j in index.items():
        own[j] = vertices[i]
    return own, local


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


def find_overlaps(mesh, depth_tolerance, threads=None):
    """Solid pairs of *mesh* overlapping deeper than *depth_tolerance*, as
    ``[(solid_a, solid_b, max_depth, exposure), ...]``, found on *threads*
    threads (``None``: rayon's global pool)."""
    solids = sorted(_solid_triangles(mesh).items())
    return solid_overlaps(mesh.vertices, solids, depth_tolerance, threads)


def describe(mesh, found, deflection=None):
    """Human-readable lines for :func:`find_overlaps` results; with
    *deflection* (``{solid_id: deflection}``), each also gives the depth the
    two solids' facetings explain, twice the larger deflection."""
    return [
        f"{_name(mesh, a)} and {_name(mesh, b)} overlap {depth:.4g} deep"
        + (f", more than the {2 * max(deflection[a], deflection[b]):.4g} their "
           "facetings explain" if deflection else "")
        + f" (roughly {100 * exposure:.2g}% of tracks cross the overlap)"
        for a, b, depth, exposure in found
    ]


def resolve_smaller_wins(mesh, found, contact_tolerance, samples=200_000, threads=None):
    """Rank each overlapping pair so the smaller solid takes the shared region,
    sampling on *threads* threads (``None``: rayon's global pool).

    Returns ``(pairs, volume_measures, lines)``: ``(low, high)`` 1-based solid
    pairs (``high`` takes the overlap), the volume of every solid in solid
    order with what the larger solid of each pair gives up taken off, and a
    report line per pair.
    """
    tris = _solid_triangles(mesh)
    volumes = _solid_volumes(mesh)
    own = {}  # solid -> its triangles over its own vertices
    pairs = []
    lines = []
    given_up = {sid: 0.0 for sid in volumes}
    shared_with = {}  # (low, high) -> volume high takes from low
    inside = set()  # (outer, inner): inner lies wholly inside outer
    for a, b, depth, _exposure in found:
        low, high = sorted((a, b), key=lambda s: (-volumes[s], s))
        for sid in (low, high):
            if sid not in own:
                own[sid] = _own_vertices(mesh.vertices, tris[sid])
        # Only the pair's own vertices go across: the whole model's made every
        # pair cost as much as the model, which a part in a big assembly,
        # overlapping many others, paid many times over.
        (verts_low, tris_low), (verts_high, tris_high) = own[low], own[high]
        n = len(verts_low)
        shared, err, embedded = solid_shared_volume(
            verts_low + verts_high,
            tris_low,
            [[i + n for i in tri] for tri in tris_high],
            contact_tolerance,
            samples,
            threads=threads,
        )
        if embedded is not None:
            inner = (low, high)[embedded]
            inside.add(((low, high)[1 - embedded], inner))
            shared, err = volumes[inner], 0.0
            how = f"{_name(mesh, inner)} lies wholly inside"
        else:
            how = f"shared {shared:.4g} +- {err:.2g} (sampled)"
        given_up[low] += shared
        shared_with[(low, high)] = shared
        pairs.append((low, high))
        lines.append(
            f"{_name(mesh, high)} takes its overlap with {_name(mesh, low)} "
            f"({depth:.4g} deep; {how}; volume of {_name(mesh, low)} "
            f"{volumes[low]:.6g} -> {volumes[low] - given_up[low]:.6g})"
        )
    # A solid giving up overlaps to two solids that overlap each other would
    # have their common region taken off twice. When one lies wholly inside
    # the other (a nested chain: a bolt in a sleeve in a block), its share is
    # all within the outer one's and is not taken off again. Partial overlaps
    # between them are not resolved and are reported.
    winners = {}
    for low, high in pairs:
        winners.setdefault(low, set()).add(high)
    paired = {frozenset(p) for p in pairs}
    for low, highs in winners.items():
        nested = {h for h in highs if any((o, h) in inside for o in highs if o != h)}
        for h in nested:
            given_up[low] -= shared_with[(low, h)]
        highs = sorted(highs - nested)
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
