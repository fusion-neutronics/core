"""Scene-based surface meshing: yamc's own OCP extraction feeding the ``yamm``
scene mesher (replaces OCC ``BRepMesh``).

For each BRep face, the outer wire and any hole wires are walked and each edge
is discretized in 3D (``GCPnts_TangentialDeflection``), then those parameters
are evaluated through the face's pcurve to a UV polyline
(``BRepAdaptor_Curve2d``). A per-edge parameter cache keyed by the *global* edge
index makes edges shared between faces use identical discretization, so adjacent
faces meet on coincident boundary points (conformal, watertight surfaces). The
scene mesher triangulates each face in UV; we evaluate UV -> 3D back onto the
OCC surface (``BRepAdaptor_Surface``).

No winding flip is applied: scene triangles are CCW-in-UV, which maps to the
``+du x dv`` 3D normal -- the face's natural orientation, matching the
``surface_volumes`` sense convention that ``CadToYamc.to_arrow`` derives from
``face_solid_reversed``.
"""

import math

from OCP.TopAbs import TopAbs_REVERSED, TopAbs_EDGE, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.GCPnts import GCPnts_TangentialDeflection, GCPnts_AbscissaPoint
from OCP.TopoDS import TopoDS

from yamc._core import mesh_faces_scene_resolved, weld_mesh

# Max structured-grid divisions per parametric direction (fallback meshing).
_MAX_GRID = 256

# Degenerate-edge arc-length threshold (3D units); below this an edge carries
# no usable discretization (e.g. seam/pole collapses) and is dropped.
_MIN_EDGE_LENGTH = 1e-10

# Squared cross-product magnitude below which a 3D triangle is treated as
# degenerate (zero-area) and dropped. Triangles touching a parametric
# singularity (e.g. a sphere pole, where several UV points map to one 3D point)
# collapse to ~zero area; matches the legacy BRepMesh path's filter.
_MIN_TRI_CROSS_SQ = 1e-24


def _tri_cross_sq(a, b, c):
    """Squared magnitude of (b-a) x (c-a) for 3D points a, b, c."""
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cx = e1[1] * e2[2] - e1[2] * e2[1]
    cy = e1[2] * e2[0] - e1[0] * e2[2]
    cz = e1[0] * e2[1] - e1[1] * e2[0]
    return cx * cx + cy * cy + cz * cz


def _triangle_area(vertices, triangles):
    """Total 3D area of a triangle list."""
    return 0.5 * sum(
        _tri_cross_sq(vertices[t[0]], vertices[t[1]], vertices[t[2]]) ** 0.5
        for t in triangles
    )


# A non-planar face whose CDT mesh covers less than this fraction of its true
# (analytic) surface area is treated as botched -> remeshed as a structured
# grid. Correct-but-coarse meshes cover ~the full area; an incomplete one
# (sphere=0%, doubly-periodic torus~46%) falls far below.
_MIN_AREA_COVERAGE = 0.5


def _extract_wire_uv(occ_face, occ_wire, edge_map, params_cache,
                     tolerance, angular_tolerance):
    """Walk a wire -> list of ``(edge_gid, reversed, uv_points)``.

    Shared edges reuse cached 3D parameters (keyed by global edge index) so
    every face discretizes a shared edge identically.
    """
    edges = []
    explorer = BRepTools_WireExplorer(occ_wire, occ_face)
    while explorer.More():
        occ_edge = explorer.Current()
        reversed_ = occ_edge.Orientation() == TopAbs_REVERSED

        curve3d = BRepAdaptor_Curve(occ_edge)
        if GCPnts_AbscissaPoint.Length_s(curve3d) < _MIN_EDGE_LENGTH:
            explorer.Next()
            continue

        gid = edge_map.FindIndex(occ_edge)
        params = params_cache.get(gid)
        if params is None:
            disc = GCPnts_TangentialDeflection(curve3d, tolerance, angular_tolerance)
            params = [disc.Parameter(i) for i in range(1, disc.NbPoints() + 1)]
            params_cache[gid] = params

        pcurve = BRepAdaptor_Curve2d(occ_edge, occ_face)
        uv_points = [[pcurve.Value(p).X(), pcurve.Value(p).Y()] for p in params]
        if len(uv_points) >= 2:
            edges.append((int(gid), bool(reversed_), uv_points))

        explorer.Next()
    return edges


def _structured_grid_face(surf, tolerance, angular_tolerance):
    """Fallback mesh for a non-planar face the CDT can't bound (a parametric
    singularity: sphere/torus poles + seam leave no closed UV boundary).

    Builds a structured ``n_u x n_v`` grid over the surface's UV domain,
    evaluates each node to 3D, and emits two triangles per cell, dropping the
    degenerate ones (pole rows). Returns ``(vertices, triangles)``.
    """
    u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
    v0, v1 = surf.FirstVParameter(), surf.LastVParameter()
    step = max(angular_tolerance, 1e-3)
    n_u = min(_MAX_GRID, max(8, int(math.ceil((u1 - u0) / step))))
    n_v = min(_MAX_GRID, max(4, int(math.ceil((v1 - v0) / step))))

    verts = []
    for i in range(n_u + 1):
        u = u0 + (u1 - u0) * i / n_u
        for j in range(n_v + 1):
            v = v0 + (v1 - v0) * j / n_v
            pnt = surf.Value(u, v)
            verts.append([pnt.X(), pnt.Y(), pnt.Z()])

    def node(i, j):
        return i * (n_v + 1) + j

    triangles = []
    for i in range(n_u):
        for j in range(n_v):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            for tri in ([a, b, c], [a, c, d]):
                if _tri_cross_sq(verts[tri[0]], verts[tri[1]], verts[tri[2]]) \
                        >= _MIN_TRI_CROSS_SQ:
                    triangles.append(tri)
    return verts, triangles


def scene_face_meshes(pa, tolerance, angular_tolerance, face_ids=None):
    """Mesh the given faces with the scene CDT (default: all faces).

    Returns ``{face_id: (vertices_3d, triangles_local)}`` -- per-face meshes with
    0-based, face-local triangle indices. Non-planar faces the CDT can't close
    from their (degenerate seam/pole) wire are remeshed as a structured UV grid:
    either they collapse to nothing (sphere poles) or cover too little of the
    true analytic area. Doubly-periodic tori are a known gap here (left on the
    CDT path); they are routed to BRepMesh upstream instead.
    """
    if face_ids is None:
        face_ids = list(pa.faces.keys())
    face_ids = list(face_ids)
    if not face_ids:
        return {}

    edge_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(pa.imprinted_compound, TopAbs_EDGE, edge_map)
    params_cache = {}

    pre_faces = []
    for fid in face_ids:
        occ_face = pa.face_to_occ[fid]
        outer = BRepTools.OuterWire_s(occ_face)
        boundary = _extract_wire_uv(
            occ_face, outer, edge_map, params_cache, tolerance, angular_tolerance
        )
        holes = []
        wire_exp = TopExp_Explorer(occ_face, TopAbs_WIRE)
        while wire_exp.More():
            wire = TopoDS.Wire(wire_exp.Current())
            if not wire.IsSame(outer):
                holes.append(_extract_wire_uv(
                    occ_face, wire, edge_map, params_cache,
                    tolerance, angular_tolerance,
                ))
            wire_exp.Next()
        pre_faces.append(
            (int(fid), boundary, holes, bool(pa.faces[fid].is_planar), "coarse")
        )

    outputs = mesh_faces_scene_resolved(pre_faces, tolerance, angular_tolerance)
    out_by_fid = {o.face_id: o for o in outputs}

    result = {}
    for fid in face_ids:
        surf = BRepAdaptor_Surface(pa.face_to_occ[fid])
        is_planar = pa.faces[fid].is_planar
        out = out_by_fid.get(fid)

        face_verts, kept = [], []
        if out is not None and out.triangles:
            for (u, v) in list(out.boundary_uv_vertices) + list(out.interior_uv_vertices):
                pnt = surf.Value(u, v)
                face_verts.append([pnt.X(), pnt.Y(), pnt.Z()])
            kept = [
                tri for tri in out.triangles
                if _tri_cross_sq(face_verts[tri[0]], face_verts[tri[1]], face_verts[tri[2]])
                >= _MIN_TRI_CROSS_SQ
            ]

        if not is_planar:
            true_area = pa.faces[fid].surface_area
            if not kept or (
                true_area > 0.0
                and _triangle_area(face_verts, kept) < _MIN_AREA_COVERAGE * true_area
            ):
                face_verts, kept = _structured_grid_face(
                    surf, tolerance, angular_tolerance
                )

        if kept:
            result[fid] = (face_verts, kept)
    return result


def assemble_surface_mesh(pa, per_face_meshes):
    """Combine per-face ``{fid: (verts_3d, tris_local)}`` meshes into welded
    global arrays.

    Surface ids are 1-indexed and assigned in ``pa.faces`` order to faces that
    produced triangles. Coincident vertices are then welded so the surface is
    watertight at the INDEX level (faces append their own boundary vertices and
    the structured-grid fallback duplicates seam/pole vertices ~1e-12 apart;
    transport ray-fire leaks through those micro-gaps otherwise). Returns
    ``(vertices, triangles, triangle_face_ids, triangle_surface_ids,
    face_to_surface_id)``.
    """
    vertices, triangles, triangle_face_ids, triangle_surface_ids = [], [], [], []
    face_to_surface_id = {}
    surface_id = 0
    for fid in pa.faces:
        mesh = per_face_meshes.get(fid)
        if not mesh or not mesh[1]:
            continue
        face_verts, kept = mesh
        surface_id += 1
        face_to_surface_id[fid] = surface_id
        base = len(vertices)
        vertices.extend(face_verts)
        for tri in kept:
            triangles.append([tri[0] + base, tri[1] + base, tri[2] + base])
            triangle_face_ids.append(fid)
            triangle_surface_ids.append(surface_id)

    vertices, triangles, triangle_face_ids, triangle_surface_ids = _weld_surface(
        vertices, triangles, triangle_face_ids, triangle_surface_ids
    )
    return vertices, triangles, triangle_face_ids, triangle_surface_ids, face_to_surface_id


def _weld_surface(vertices, triangles, face_ids, surface_ids, rel_tol=1e-6):
    """Merge coincident vertices and reindex triangles (in the Rust core),
    keeping the per-triangle face/surface id arrays aligned. Triangles that
    collapse to degenerate after welding are dropped (with their ids)."""
    if not vertices:
        return vertices, triangles, face_ids, surface_ids
    new_verts, new_tris, kept = weld_mesh(vertices, triangles, rel_tol)
    new_face_ids = [face_ids[i] for i in kept]
    new_surf_ids = [surface_ids[i] for i in kept]
    return new_verts, new_tris, new_face_ids, new_surf_ids
