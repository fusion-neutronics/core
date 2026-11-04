"""Mesh every curved CAD face to the volume the face itself encloses.

A chordal mesh inscribes a curved surface: its vertices lie on the surface
and its facets cut inside every convex part, so each curved solid comes out
too small (2 to 5% on the benchmark zoo at a chordal tolerance of 1% of the
model size), and a transported flux is low by as much. Moving vertices off
the surface by a fraction of the deflection, outward where the facets sag
inward, gives the mesh the CAD volume without adding a triangle.

Two passes. Each moves a welded vertex, which every triangle using it
follows, so shared edges, and the faces two imprinted solids share, stay
watertight and conformal:

- Edges. The vertices inside each curved CAD edge (its end vertices stay
  put) move along the curve's outward principal normal, all by one distance,
  until the polyline cuts off as much area as the arc. A flat face bounded by
  a circle then has its exact area, and a cylinder its exact volume, though
  its curved face has no vertex of its own to move.
- Faces. The vertices inside each curved face (on none of its edges) move
  along their area-weighted vertex normal, all by one distance, until the
  facets enclose what the CAD face does. What a face encloses is measured
  against the polyline its facets end on: the CAD face is closed off to that
  polyline by the thin sliver between each mesh segment and the arc of CAD
  edge it spans, so the measure does not depend on the point the volume is
  taken from, and the slivers cancel between the two faces of each edge.

No vertex moves further than the chordal deflection its face or edge was
meshed at, which the mesh already deviates by. Flat faces, faces with no
vertex of their own, and faces whose boundary cannot be matched to their CAD
edges are left as meshed.
"""

import math

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp_Domain, BRepGProp_Face, BRepGProp_VinertGK
from OCP.BRepLProp import BRepLProp_CLProps
from OCP.GeomAbs import GeomAbs_Line
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Dir, gp_Pnt

from ._fidelity import _GPROP_RELATIVE_PRECISION

# Points along each arc of CAD edge between two mesh vertices, for the sliver
# between the arc and its chord. The sliver is close to a parabolic segment,
# whose area this resolves to about 1e-3 of itself, far inside the mesh error.
_ARC_SAMPLES = 8

# Secant steps on the exact enclosed volume (or cut-off area) after a first
# linear step. The measure is linear in the distance to first order, so a
# few steps meet it to rounding.
_ITERATIONS = 3


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add_scaled(a, s, d):
    return (a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _cone(p, q, r, apex):
    """Signed volume of the tetrahedron from *apex* over triangle pqr."""
    p, q, r = _sub(p, apex), _sub(q, apex), _sub(r, apex)
    return _dot(p, _cross(q, r)) / 6.0


def _fan_cone(loop, apex):
    """Signed volume from *apex* over the closed polygon *loop*, fanned from
    its first point."""
    first = loop[0]
    return sum(_cone(first, loop[j], loop[j + 1], apex) for j in range(1, len(loop) - 1))


def _vector_area(loop):
    """Vector area of the closed polygon *loop*."""
    first = loop[0]
    ax = ay = az = 0.0
    for j in range(1, len(loop) - 1):
        c = _cross(_sub(loop[j], first), _sub(loop[j + 1], first))
        ax += c[0]
        ay += c[1]
        az += c[2]
    return (0.5 * ax, 0.5 * ay, 0.5 * az)


def _solve(residual, slope, cap):
    """``(h, capped)``: the distance within [-cap, cap] that zeroes
    *residual*, a function of one distance that is close to linear with
    *slope* at zero, by a Newton step and then secant steps."""
    h0, r0 = 0.0, residual(0.0)
    if slope == 0.0 or r0 == 0.0:
        return 0.0, False
    h1 = max(-cap, min(cap, -r0 / slope))
    r1 = residual(h1)
    for _ in range(_ITERATIONS):
        if r1 == r0 or abs(r1) <= 1e-12 * abs(r0):
            break
        h2 = max(-cap, min(cap, h1 - r1 * (h1 - h0) / (r1 - r0)))
        if h2 == h1:
            break
        h0, r0 = h1, r1
        h1, r1 = h2, residual(h2)
    return h1, abs(h1) >= cap


class _CadEdge:
    """A CAD edge and the mesh vertices along it, in curve order."""

    def __init__(self, edge):
        self.edge = edge
        self.curve = BRepAdaptor_Curve(edge)
        self.first = self.curve.FirstParameter()
        self.last = self.curve.LastParameter()
        self.straight = self.curve.GetType() == GeomAbs_Line
        self.faces = []  # face ids it bounds
        self.seam = False  # bounds a face on both sides
        self.points = None  # [(param, vertex)] from first to last, or None

    def point(self, t):
        return self.curve.Value(t).Coord()

    def arc(self, t0, t1):
        """Points of the edge from parameter *t0* to *t1*, ends included."""
        return [
            self.point(t0 + (t1 - t0) * k / _ARC_SAMPLES) for k in range(_ARC_SAMPLES + 1)
        ]

    def outward(self, t):
        """Unit vector away from the centre of curvature at *t*, or None
        where the edge is straight."""
        props = BRepLProp_CLProps(self.curve, t, 2, 1e-12)
        if not props.IsTangentDefined() or props.Curvature() <= 1e-12:
            return None
        normal = gp_Dir()
        props.Normal(normal)
        x, y, z = normal.Coord()
        return (-x, -y, -z)

    def locate(self, candidates, verts, edges):
        """Find the mesh vertices on this edge among *candidates* and order
        them along it into ``self.points``; leave it None when they do not
        make a chain of mesh *edges* from one end to the other."""
        box = Bnd_Box()
        BRepBndLib.Add_s(self.edge, box, False)
        x0, y0, z0, x1, y1, z1 = box.Get()
        size = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        tol = max(10.0 * BRep_Tool.Tolerance_s(self.edge), 1e-6 * size)
        box.Enlarge(tol)
        geom = BRep_Tool.Curve_s(self.edge, self.first, self.last)
        start, end = self.point(self.first), self.point(self.last)
        closed = _norm(_sub(start, end)) <= tol
        found = []
        for v in candidates:
            p = verts[v]
            pnt = gp_Pnt(*p)
            if box.IsOut(pnt):
                continue
            if _norm(_sub(p, start)) <= tol:
                found.append((self.first, v))
                if closed:
                    found.append((self.last, v))
                continue
            if _norm(_sub(p, end)) <= tol:
                found.append((self.last, v))
                continue
            proj = GeomAPI_ProjectPointOnCurve(pnt, geom, self.first, self.last)
            if proj.NbPoints() == 0 or proj.LowerDistance() > tol:
                continue
            found.append((proj.LowerDistanceParameter(), v))
        found.sort()
        if len(found) < 2 or found[0][0] != self.first or found[-1][0] != self.last:
            return
        for (_, a), (_, b) in zip(found, found[1:]):
            if (min(a, b), max(a, b)) not in edges:
                return
        self.points = found


def preserve_face_volumes(mesh, pa, face_tolerances, edges=True, faces=True):
    """Move vertices of *mesh* (SurfaceMesh, in place) so each curved face of
    *pa* (ProcessedAssembly) encloses its CAD volume; see the module docstring.

    *face_tolerances* is ``{face_id: (deflection, angle)}`` as each face was
    meshed. *edges* and *faces* switch the two passes. Returns a dict of
    counts of what was done.
    """
    stats = dict(
        edges_moved=0, edges_capped=0, edges_unmatched=0,
        faces_moved=0, faces_capped=0, faces_no_interior=0,
        faces_unmatched=0, faces_integration_failed=0,
    )
    verts = [tuple(v) for v in mesh.vertices]
    face_tris = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        face_tris.setdefault(fid, []).append(tuple(tri))
    curved = [fid for fid in face_tris if not pa.faces[fid].is_planar]
    if not curved:
        return stats

    # Each mesh edge's triangles per face, and each vertex's faces.
    edge_uses = {}
    vert_faces = {}
    for fid, tris in face_tris.items():
        for tri in tris:
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                key = (a, b) if a < b else (b, a)
                uses = edge_uses.setdefault(key, {})
                uses[fid] = uses.get(fid, 0) + 1
                vert_faces.setdefault(a, set()).add(fid)

    # A face's boundary: the directed edges of its triangles that no other of
    # its triangles shares. None when an edge has more than two of them.
    boundary = {}
    for fid, tris in face_tris.items():
        segs = []
        for tri in tris:
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                n = edge_uses[(a, b) if a < b else (b, a)][fid]
                if n == 1:
                    segs.append((a, b))
                elif n > 2:
                    segs = None
                    break
            if segs is None:
                break
        boundary[fid] = segs

    # The CAD edges of every face next to a curved one, each once.
    wanted = set(curved)
    for fid in curved:
        for seg in boundary[fid] or ():
            key = (min(seg), max(seg))
            wanted.update(edge_uses[key])
    edge_map = TopTools_IndexedMapOfShape()
    cad_edges = {}
    for fid in sorted(wanted & set(face_tris)):
        explorer = TopExp_Explorer(pa.face_to_occ[fid], TopAbs_EDGE)
        while explorer.More():
            edge = TopoDS.Edge_s(explorer.Current())
            explorer.Next()
            if BRep_Tool.Degenerated_s(edge):
                continue
            index = edge_map.Add(edge)
            if index not in cad_edges:
                cad_edges[index] = _CadEdge(edge)
            if fid in cad_edges[index].faces:
                cad_edges[index].seam = True
            else:
                cad_edges[index].faces.append(fid)

    # Order the mesh vertices along each edge, from the boundary vertices of
    # the faces it bounds, and record which edge and parameters every
    # boundary segment spans. Without the imprint, two solids that touch have
    # an edge each where they meet, and the weld can give both the same
    # vertices: those are one edge here.
    seg_arc = {}  # (a, b), a < b -> (cad_edge, t_a, t_b)
    chains = {}
    for cad in list(cad_edges.values()):
        # A seam is explored twice in its face, once per side, and lies
        # inside it. (OCC also calls an edge closed on a face where it only
        # lies along the face's own seam line, as the end circle of a
        # sector of a torus does, so that is not the test.)
        if cad.seam:
            continue
        candidates = set()
        for fid in cad.faces:
            for seg in boundary.get(fid) or ():
                candidates.update(seg)
        cad.locate(candidates, verts, edge_uses)
        if cad.points is None:
            stats["edges_unmatched"] += 1
            continue
        chain = tuple(v for _, v in cad.points)
        twin = chains.get(chain) or chains.get(chain[::-1])
        if twin is not None:
            twin.faces.extend(f for f in cad.faces if f not in twin.faces)
            cad.points = None
            continue
        chains[chain] = cad
        for (ta, a), (tb, b) in zip(cad.points, cad.points[1:]):
            seg_arc[(a, b) if a < b else (b, a)] = (cad, ta, tb) if a < b else (cad, tb, ta)

    # Likewise two faces can be meshed alike and welded onto one set of
    # vertices: moving them moves both.
    twins = {}
    for fid, tris in face_tris.items():
        twins.setdefault(frozenset(v for tri in tris for v in tri), set()).add(fid)
    face_twins = {fid: group for group in twins.values() for fid in group}

    if edges:
        for cad in cad_edges.values():
            moved = _move_edge(cad, verts, vert_faces, face_tolerances)
            if moved is not None:
                stats["edges_moved"] += 1
                stats["edges_capped"] += moved

    if faces:
        for fid in curved:
            outcome = _move_face(
                fid, face_tris[fid], boundary[fid], seg_arc, verts, vert_faces,
                face_twins[fid], face_tolerances[fid][0], pa.face_to_occ[fid],
            )
            stats[outcome] += 1

    mesh.vertices = [list(v) for v in verts]
    return stats


def _move_edge(cad, verts, vert_faces, face_tolerances):
    """Move the vertices inside the curved edge *cad* so its polyline cuts
    off the arc's area. Returns whether the move hit its cap, or None when
    nothing moved."""
    if cad.points is None or cad.straight or len(cad.points) < 3:
        return None
    own = set(cad.faces)
    moving = {}
    for t, v in cad.points[1:-1]:
        if not vert_faces.get(v, set()) <= own:
            continue
        direction = cad.outward(t)
        if direction is not None:
            moving[v] = direction
    if not moving:
        return None
    origin = {v: verts[v] for _, v in cad.points}
    segments = []
    slope = 0.0
    for (ta, a), (tb, b) in zip(cad.points, cad.points[1:]):
        arc = cad.arc(ta, tb)
        area = _vector_area([origin[a]] + arc[1:-1] + [origin[b]])
        size = _norm(area)
        if size == 0.0:
            continue
        normal = (area[0] / size, area[1] / size, area[2] / size)
        segments.append((a, b, arc, normal))
        # Moving a chord end by h sweeps about half the chord times h.
        half = 0.5 * _norm(_sub(origin[b], origin[a]))
        slope -= half * ((a in moving) + (b in moving))

    def at(v, h):
        return _add_scaled(origin[v], h, moving[v]) if v in moving else origin[v]

    def residual(h):
        total = 0.0
        for a, b, arc, normal in segments:
            pa_, pb = at(a, h), at(b, h)
            # The arc from a to b, then back along the chord.
            loop = [pa_] + arc + [pb]
            total += _dot(_vector_area(loop), normal)
        return total

    cap = min(face_tolerances[fid][0] for fid in cad.faces)
    h, capped = _solve(residual, slope, cap)
    for v, direction in moving.items():
        verts[v] = _add_scaled(origin[v], h, direction)
    return capped


def _move_face(fid, tris, segs, seg_arc, verts, vert_faces, twins, cap, occ_face):
    """Move the vertices inside curved face *fid* so its triangles enclose
    the CAD face's volume. Returns the name of the count it adds to."""
    if segs is None:
        return "faces_unmatched"
    on_boundary = {v for seg in segs for v in seg}
    face_verts = {v for tri in tris for v in tri}
    interior = [
        v for v in face_verts
        if v not in on_boundary and vert_faces.get(v, twins) <= twins
    ]
    if not interior:
        return "faces_no_interior"
    if any((min(s), max(s)) not in seg_arc for s in segs):
        return "faces_unmatched"

    # The point the volumes are taken from: off the face along its mean
    # normal, and a little to one side, so no cone is flat.
    pts = [verts[v] for v in face_verts]
    centre = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    size = _norm(_sub(hi, lo)) or 1.0
    mean = (0.0, 0.0, 0.0)
    area_sum = 0.0
    normals = {v: (0.0, 0.0, 0.0) for v in interior}
    for p, q, r in tris:
        c = _cross(_sub(verts[q], verts[p]), _sub(verts[r], verts[p]))
        mean = _add_scaled(mean, 0.5, c)
        area_sum += 0.5 * _norm(c)
        for v in (p, q, r):
            if v in normals:
                normals[v] = _add_scaled(normals[v], 0.5, c)
    offset = (0.0137 * size, 0.0291 * size, 0.0213 * size)
    if _norm(mean) > 0.1 * area_sum:
        offset = _add_scaled(offset, 0.5 * size / _norm(mean), mean)
    apex = (centre[0] + offset[0], centre[1] + offset[1], centre[2] + offset[2])

    # The CAD face's volume from the apex, with the face's own normal (the
    # direction the stored triangles wind), closed to the mesh boundary.
    face = TopoDS.Face_s(occ_face.Oriented(TopAbs_FORWARD))
    cone = BRepGProp_VinertGK()
    cone.SetLocation(gp_Pnt(*apex))
    surface = BRepGProp_Face(face, True)
    if face.NbChildren() == 0:
        error = cone.Perform(surface, _GPROP_RELATIVE_PRECISION)
    else:
        error = cone.Perform(surface, BRepGProp_Domain(face), _GPROP_RELATIVE_PRECISION)
    if error < 0.0:
        return "faces_integration_failed"
    target = cone.Mass()
    for u, w in segs:
        cad, ta, tb = seg_arc[(u, w) if u < w else (w, u)]
        if u > w:
            ta, tb = tb, ta
        # The sliver from the chord u-w back along the arc from w to u.
        target += _fan_cone([verts[u], verts[w]] + cad.arc(tb, ta), apex)

    slope = 0.0
    for v in interior:
        n = _norm(normals[v])
        if n > 0.0:
            normals[v] = (normals[v][0] / n, normals[v][1] / n, normals[v][2] / n)
            slope += n / 3.0
        else:
            normals[v] = None
    moving = {v: d for v, d in normals.items() if d is not None}
    fixed = 0.0
    changing = []
    for tri in tris:
        if any(v in moving for v in tri):
            changing.append(tri)
        else:
            fixed += _cone(verts[tri[0]], verts[tri[1]], verts[tri[2]], apex)
    origin = {v: verts[v] for tri in changing for v in tri}

    def residual(h):
        def at(v):
            return _add_scaled(origin[v], h, moving[v]) if v in moving else origin[v]
        return fixed + sum(_cone(at(p), at(q), at(r), apex) for p, q, r in changing) - target

    h, capped = _solve(residual, slope, cap)
    for v, direction in moving.items():
        verts[v] = _add_scaled(origin[v], h, direction)
    return "faces_capped" if capped else "faces_moved"
