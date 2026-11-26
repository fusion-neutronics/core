"""``MeshGeometry.add_tets``: fill volumes of a mesh geometry with tetrahedra.

Needs no CAD: it works on the mesh's own surface triangles, so it serves a
surface mesh made by ``MeshGeometry.from_cadquery`` (which calls it for
``tet_volumes``) as well as one loaded with ``from_arrow``.
"""

from __future__ import annotations

from typing import Any, Literal


def _solid_boundary(vertices, triangles):
    """A solid's surface triangles over just the vertices they use, as
    ``(boundary_vertices, boundary_triangles)``. The surface is already
    index-shared, so no welding by position: that would merge the two sides of
    a solid thinner than the weld's tolerance."""
    used = sorted({i for tri in triangles for i in tri})
    local = {i: k for k, i in enumerate(used)}
    return [vertices[i] for i in used], [[local[i] for i in tri] for tri in triangles]


def _default_tet_edge_length(vertices):
    """Default target tet edge length: ~1/10 of the bounding-box diagonal."""
    extents = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
    diag = sum(e * e for e in extents) ** 0.5
    return max(diag / 10.0, 1e-9)


def _threads(mesh_threads):
    """``None`` (rayon's global pool) for ``"all"``, else the positive count."""
    if isinstance(mesh_threads, str) and mesh_threads == "all":
        return None
    if type(mesh_threads) is int and mesh_threads > 0:
        return mesh_threads
    raise ValueError(f"mesh_threads must be 'all' or a positive integer, got {mesh_threads!r}")


def add_tets(
    self,
    volumes: list[str] | None = None,
    target_edge_length: float | dict[str, float] | None = None,
    *,
    mesh_threads: int | Literal["all"] = "all",
) -> Any:
    """A copy of this geometry with *volumes* filled with tetrahedra.

    The geometry itself is left as it is, so a :class:`Model` holding it is
    unaffected. Each volume's tets are made conformal with the surface and
    with the tets of the volumes it shares surfaces with, including those
    already tet-meshed; ``mesh_report["conform"]`` says what that repaired.

    Args:
        volumes: Volume names (``volume_names``) to tet-mesh; ``None`` for
            every named volume without tets.
        target_edge_length: Target tet edge length (cm), one for every
            volume or a dict by volume name. ``None``, or a volume the dict
            leaves out, takes a tenth of the volume's bounding-box diagonal.
        mesh_threads: ``"all"`` (default) or a positive number of threads.

    Returns:
        MeshGeometry: The new geometry, with the same materials, implicit
        complement, graveyard and contact tolerance.
    """
    from yamc._core import conform_tet_meshes, mesh_volume_rs

    threads = _threads(mesh_threads)
    names = self.volume_names
    vertices, triangles, tets, tet_ids = self._file_mesh()
    vertices = [list(v) for v in vertices]
    triangles = [list(t) for t in triangles]
    n_file_tris = len(triangles)
    has_tets = {v - 1 for v in tet_ids if v > 0}

    if volumes is None:
        volumes = [n for i, n in enumerate(names) if n is not None and i not in has_tets]
    wanted = []
    for name in volumes:
        if name not in names:
            raise ValueError(
                f"no volume named {name!r}; the volumes are "
                + ", ".join(repr(n) for n in names if n is not None)
            )
        v = names.index(name)
        if v in has_tets:
            raise ValueError(f"volume {name!r} already has tets")
        wanted.append(v)

    tris_of = {
        v: [t for t in self.volume_triangle_indices(v) if t < n_file_tris]
        for v in set(wanted) | has_tets
    }

    # Tet vertices are their own entities, apart from the surface's (as the
    # file stores them): merged among the tet blocks by exact coordinate, so a
    # shared surface meets bit-identically from both sides.
    tet_vertex = {}
    for t in tets:
        for i in t:
            tet_vertex.setdefault(tuple(vertices[i]), i)
    solid_tets = {v: [] for v in tris_of}
    for t, vid in zip(tets, tet_ids):
        if vid > 0:
            solid_tets[vid - 1].append(list(t))
    for v in wanted:
        boundary_verts, boundary_tris = _solid_boundary(
            vertices, [triangles[t] for t in tris_of[v]]
        )
        tel = target_edge_length
        if isinstance(tel, dict):
            tel = tel.get(names[v])
        if tel is None:
            tel = _default_tet_edge_length(boundary_verts)
        interior, new = mesh_volume_rs(boundary_verts, boundary_tris, float(tel), threads)
        index = []
        for p in list(boundary_verts) + [list(p) for p in interior]:
            key = (float(p[0]), float(p[1]), float(p[2]))
            g = tet_vertex.get(key)
            if g is None:
                g = len(vertices)
                tet_vertex[key] = g
                vertices.append(list(key))
            index.append(g)
        solid_tets[v] = [[index[i] for i in t] for t in new]

    # Repair: the surface in tet numbering, then back.
    order = sorted({t for v in tris_of for t in tris_of[v]})
    position = {t: k for k, t in enumerate(order)}
    # (a surface vertex no tet uses, as after a legacy-clip fallback, stays)
    surface = [[tet_vertex.get(tuple(vertices[i]), i) for i in triangles[t]] for t in order]
    solids = sorted(solid_tets)
    refined, parent, repaired, stats = conform_tet_meshes(
        vertices,
        surface,
        [([position[t] for t in tris_of[v]], solid_tets[v]) for v in solids],
    )

    new_triangles = list(triangles)
    triangle_parent = list(range(n_file_tris))
    seen = set()
    for tri, par in zip(refined, parent):
        src = order[par]
        corner = dict(zip(surface[par], triangles[src]))
        child = [corner.get(i, i) for i in tri]
        if par not in seen:
            seen.add(par)
            new_triangles[src] = child
        else:
            new_triangles.append(child)
            triangle_parent.append(src)
    all_tets, tet_volumes = [], []
    for v, (_own, t) in zip(solids, repaired):
        all_tets += [list(x) for x in t]
        tet_volumes += [v] * len(t)

    geometry = self._with_tets(vertices, new_triangles, triangle_parent, all_tets, tet_volumes)
    report = dict(self.mesh_report or {})
    report["conform"] = dict(
        zip(
            (
                "edges_split",
                "cones",
                "surface_triangles_refined",
                "conflicts",
                "skipped_solids",
                "reverted",
            ),
            stats,
        )
    )
    return geometry._with_report(report)


def unstructured_mesh_to_vtkhdf(self, path, *, length_units="cm") -> None:
    """Write the cells, with each one's ``region_id`` (0-based index into
    ``regions``), to a VTKHDF UnstructuredGrid file for ParaView. Requires
    ``h5py`` (``pip install yamc[viz]``). ``length_units`` is the unit of the
    written coordinates, ``"cm"`` (default), ``"m"``, ``"mm"`` or ``"km"``."""
    import os

    import numpy as np

    from yamc._vtkhdf_writer import tet_to_vtkhdf
    from yamc.vtkhdf import _length_scale

    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    index = {r: i for i, r in enumerate(self.regions)}
    tet_to_vtkhdf(
        path,
        np.asarray(self.vertices, dtype="float64") * _length_scale(length_units),
        self.cells,
        cell_data={"region_id": [index[r] for r in self.cell_regions]},
        length_units=length_units,
    )
