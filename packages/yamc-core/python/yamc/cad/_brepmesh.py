"""BRepMesh surface meshing for all-primitive solids.

Mirrors cad-to-dagmc-mesher's routing: solids whose faces are all primitive OCC
surfaces (plane / cylinder / cone / sphere / torus, plus surfaces of
revolution, which the scene CDT meshes short) are meshed with OCC
``BRepMesh``. BRepMesh uses the *harmonized OCC edge discretization*, so
adjacent faces share boundary vertices and closed curved surfaces (sphere,
torus) come out watertight -- whereas the scene CDT struggles to close them
from their degenerate seam/pole wires. Imported / b-spline faces stay on the
scene CDT (see :mod:`yamc.cad._scene_extract`), where BRepMesh is poor.
"""

from collections import Counter

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound

from ._scene_extract import TopTools_IndexedMapOfShape, face_edge_ids


def run_brepmesh(shape, tolerance, angular_tolerance, threads):
    """Run ``BRepMesh_IncrementalMesh`` with interior deflection control.

    The simple ``(shape, deflection, relative, angle)`` constructor leaves
    ``IMeshTools_Parameters.AngleInterior`` unset, and OCC then grids the
    interior of periodic BSpline faces (full-torus shells) with cells that
    violate the angular tolerance by an order of magnitude in a band next
    to the u-seam (metre-long chords at a millimetre deflection request).
    On nested thin shells the seam bands of adjacent surfaces cross each
    other even though every shell is watertight. Setting ``AngleInterior``
    (and ``DeflectionInterior``) explicitly enforces the tolerances on
    interior nodes as well.

    Faces are meshed in parallel on *threads* threads. The triangulation does
    not depend on the count, imprinted or not (see
    tests/cad/test_mesh_threads.py); time and peak memory do. On a 64-solid
    imported model meshed without imprinting, 22 threads took 35 s and peaked
    at 11.4GB, 4 took 43 s at 4.4GB and 1 took 197 s at 1.9GB.
    """
    from OCP.IMeshTools import IMeshTools_Parameters

    from .assembly_processor import occ_threads

    params = IMeshTools_Parameters()
    params.Deflection = tolerance
    params.DeflectionInterior = tolerance
    params.Angle = angular_tolerance
    params.AngleInterior = angular_tolerance
    params.InParallel = True
    with occ_threads(threads):
        BRepMesh_IncrementalMesh(shape, params)


def _compound(faces):
    """The faces gathered into one compound, sharing their edges."""
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face)
    return compound


def _face_mesh(face):
    """*face*'s triangulation in 3D, as ``(vertices, triangles)`` with 0-based
    triangle indices, or None when it has none."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    nodes = range(1, tri.NbNodes() + 1)
    if loc.IsIdentity():
        verts = [tri.Node(i).Coord() for i in nodes]
    else:
        transform = loc.Transformation()
        verts = [tri.Node(i).Transformed(transform).Coord() for i in nodes]
    tris = [
        (i1 - 1, i2 - 1, i3 - 1)
        for i1, i2, i3 in (tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1))
    ]
    return verts, tris


def _edge_points(face, edge):
    """The 3D points of *face*'s triangulation along *edge*, or None."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    poly = None if tri is None else BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
    if poly is None:
        return None
    nodes = poly.Nodes()
    points = [tri.Node(nodes.Value(i)) for i in range(nodes.Lower(), nodes.Upper() + 1)]
    if not loc.IsIdentity():
        transform = loc.Transformation()
        points = [p.Transformed(transform) for p in points]
    return [p.Coord() for p in points]


def _same_points(a, b):
    """Whether two point lists agree point by point, to rounding."""
    return all(
        abs(x - y) <= 1e-9 * (1.0 + abs(x))
        for p, q in zip(a, b)
        for x, y in zip(p, q)
    )


def _same_edge(a, b):
    """Whether two faces' points along an edge are the same, in either order."""
    if a is None or b is None:
        return False
    return len(a) == len(b) and (_same_points(a, b) or _same_points(a, b[::-1]))


def _follows_its_edges(face, tris):
    """Whether the open boundary of *face*'s triangles (0-based, as
    :func:`_face_mesh` gives them) is exactly the polygons on its edges.

    A seam edge is explored twice, once per side, and gets each side's
    polygon. BRepMesh can fail a face without reporting it, leaving one stray
    triangle while the face's edges keep polygons that match its neighbours'.
    """
    uses = Counter()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            uses[(u, v) if u < v else (v, u)] += 1
    if any(n > 2 for n in uses.values()):
        return False
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    segments = Counter()
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        poly = BRep_Tool.PolygonOnTriangulation_s(TopoDS.Edge_s(explorer.Current()), tri, loc)
        if poly is None:
            return False
        nodes = poly.Nodes()
        ids = [nodes.Value(i) - 1 for i in range(nodes.Lower(), nodes.Upper() + 1)]
        for u, v in zip(ids, ids[1:]):
            if u != v:
                segments[(u, v) if u < v else (v, u)] += 1
        explorer.Next()
    boundary = {edge for edge, n in uses.items() if n == 1}
    return set(segments) == boundary and all(n == 1 for n in segments.values())


def _finest(values):
    """The smallest deflection and the smallest angle among ``(deflection,
    angle)`` pairs."""
    values = list(values)
    return min(v[0] for v in values), min(v[1] for v in values)


def _finest_first_on_shared_edges(mesh_at, shared_edges, placements=()):
    """Make the first face meshed on each shared edge the finest on it.

    That face sets the edge's points for every face on it, so it is meshed at
    the smallest deflection and the smallest angle among them. They are
    usually one face's values; they differ where two solids are given
    deflection and angle in opposite orders. The placements of one shape
    (lists of face ids) share a triangulation, so each takes the finest values
    among them. Updates *mesh_at* in place.
    """
    changed = True
    while changed:
        changed = False
        for fids in placements:
            finest = _finest(mesh_at[f] for f in fids)
            for fid in fids:
                if mesh_at[fid] != finest:
                    mesh_at[fid] = finest
                    changed = True
        for fids in shared_edges.values():
            first = min(fids, key=mesh_at.get)
            finest = _finest(mesh_at[f] for f in fids)
            if mesh_at[first] != finest:
                mesh_at[first] = finest
                changed = True


def _groups(face_ids, links):
    """*face_ids* in the groups that *links* (lists of face ids) join."""
    parent = {fid: fid for fid in face_ids}

    def root(fid):
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    for fids in links:
        for fid in fids[1:]:
            parent[root(fid)] = root(fids[0])
    groups = {}
    for fid in face_ids:
        groups.setdefault(root(fid), []).append(fid)
    return list(groups.values())


def _mesh_in_passes(pa, face_ids, mesh_at, face_edges, edge_faces, threads):
    """Mesh *face_ids* in passes, finest first, each pass together with the
    faces meshed before it that share an edge with it."""
    meshed = set()
    for value in sorted({mesh_at[fid] for fid in face_ids}):
        group = [fid for fid in face_ids if mesh_at[fid] == value]
        reused = sorted({
            other
            for fid in group
            for gid in face_edges[fid]
            for other in edge_faces[gid]
            if other in meshed
        })
        run_brepmesh(
            _compound(pa.face_to_occ[fid] for fid in group + reused), *value, threads
        )
        meshed.update(group)


def _mesh_face_by_face(pa, face_ids, face_tolerances, threads):
    """Mesh each face at its own ``(deflection, angle)``, keeping every shared
    edge conformal. Returns ``{face_id: mesh}`` as :func:`_face_mesh` gives.

    BRepMesh takes one deflection per call, and a call on some faces alone
    discretizes their edges afresh: faces meshed in separate calls end up with
    different points along an edge they share, an open crack (as
    cad-to-dagmc-mesher found). But a call reuses the triangulation of a face
    passed to it that is already finer than asked, and gives the faces it
    meshes that face's points along their common edges. So the faces are
    meshed in passes, finest first, each pass together with the faces meshed
    before it that share an edge with it: an edge is discretized once, by the
    first pass to reach it, and that pass is the finest of its faces (see
    :func:`_finest_first_on_shared_edges`).

    OCC does not always manage it. It discretizes an edge finer again when a
    face's boundary crosses itself at the coarser points (a thin annulus
    between a fine and a coarse circle), and a neighbour meshed before keeps
    the old points; or it fails a face, leaving a stray triangle whose edges
    still match. So each group of faces joined by edges, or by one shape
    placed several times (its placements share a triangulation), that mixes
    values is checked: every face's triangles must close onto the polygons on
    its edges, and every face on an edge must have the same points along it.
    A group that fails is meshed again in one call at the finest values in it,
    as with one value everywhere.
    """
    edge_map, face_edges = face_edge_ids(pa, face_ids)
    edge_faces = {}
    for fid in face_ids:
        for gid in dict.fromkeys(face_edges[fid]):
            edge_faces.setdefault(gid, []).append(fid)
    shared = {gid: fids for gid, fids in edge_faces.items() if len(fids) > 1}
    shapes = TopTools_IndexedMapOfShape()
    by_shape = {}
    for fid in face_ids:
        unplaced = pa.face_to_occ[fid].Located(TopLoc_Location())
        by_shape.setdefault(shapes.Add(unplaced), []).append(fid)
    placements = [fids for fids in by_shape.values() if len(fids) > 1]

    mesh_at = {fid: face_tolerances[fid] for fid in face_ids}
    _finest_first_on_shared_edges(mesh_at, shared, placements)
    _mesh_in_passes(pa, face_ids, mesh_at, face_edges, edge_faces, threads)
    meshes = {fid: _face_mesh(pa.face_to_occ[fid]) for fid in face_ids}

    def watertight(group):
        if not all(
            meshes[fid] is not None and _follows_its_edges(pa.face_to_occ[fid], meshes[fid][1])
            for fid in group
        ):
            return False
        for gid in dict.fromkeys(gid for fid in group for gid in face_edges[fid]):
            first, *others = edge_faces[gid]
            if others:
                edge = TopoDS.Edge_s(edge_map.FindKey(gid))
                points = _edge_points(pa.face_to_occ[first], edge)
                if not all(
                    _same_edge(_edge_points(pa.face_to_occ[fid], edge), points)
                    for fid in others
                ):
                    return False
        return True

    redo = [
        group
        for group in _groups(face_ids, list(shared.values()) + placements)
        if len({mesh_at[fid] for fid in group}) > 1 and not watertight(group)
    ]
    if redo:
        for group in redo:
            mesh_at.update(dict.fromkeys(group, _finest(mesh_at[fid] for fid in group)))
        faces = [fid for group in redo for fid in group]
        BRepTools.Clean_s(_compound(pa.face_to_occ[fid] for fid in faces))
        _mesh_in_passes(pa, faces, mesh_at, face_edges, edge_faces, threads)
        meshes.update({fid: _face_mesh(pa.face_to_occ[fid]) for fid in faces})
    return meshes


def brepmesh_face_meshes(pa, face_ids, face_tolerances, threads=1):
    """Per-face BRepMesh triangulations for ``face_ids``.

    *face_tolerances* maps each face id to the ``(deflection, angle)`` it is
    meshed at, on *threads* threads. When every face asks for the same, ``BRepMesh_IncrementalMesh``
    runs once on the imprinted compound (so the discretization is harmonized
    across faces); otherwise each face is meshed at its own, see
    :func:`_mesh_face_by_face`. Triangulations already on the shapes (a
    previous mesh of the same assembly, which shares them) are dropped first:
    BRepMesh would keep any finer than asked. Returns ``{face_id: (vertices,
    triangles)}`` in 3D with face-local 0-based triangle indices, degenerate
    triangles included (:func:`~yamc.cad._scene_extract.assemble_surface_mesh`
    drops them).
    """
    face_ids = list(face_ids)
    if not face_ids:
        return {}

    BRepTools.Clean_s(pa.imprinted_compound)
    values = {face_tolerances[fid] for fid in face_ids}
    if len(values) == 1:
        run_brepmesh(pa.imprinted_compound, *values.pop(), threads)
        meshes = {fid: _face_mesh(pa.face_to_occ[fid]) for fid in face_ids}
    else:
        meshes = _mesh_face_by_face(pa, face_ids, face_tolerances, threads)
    return {fid: mesh for fid, mesh in meshes.items() if mesh is not None and mesh[1]}
