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

The CAD side (curve points and normals, each face's exact volume) is
evaluated here with OCC; the mesh side (adjacency, boundaries, volume sums,
root finding, vertex moves) is the compiled ``FaceVolumeMesh``.
"""

import math
import time

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

from yamc._core import FaceVolumeMesh

from ._fidelity import _GPROP_RELATIVE_PRECISION

# Points along each arc of CAD edge between two mesh vertices, for the sliver
# between the arc and its chord. The sliver is close to a parabolic segment,
# whose area this resolves to about 1e-3 of itself, far inside the mesh error.
_ARC_SAMPLES = 8


def _distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) * (a[i] - b[i]) for i in range(3)))


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
        self.index = None  # its index in the FaceVolumeMesh, once added

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

    def locate(self, fvm):
        """Find the mesh vertices of *fvm* (FaceVolumeMesh) on this edge,
        among those on the boundaries of its faces, and order them along it
        into ``self.points``; leave it None when they do not make a chain of
        mesh edges from one end to the other."""
        box = Bnd_Box()
        BRepBndLib.Add_s(self.edge, box, False)
        x0, y0, z0, x1, y1, z1 = box.Get()
        size = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        tol = max(10.0 * BRep_Tool.Tolerance_s(self.edge), 1e-6 * size)
        box.Enlarge(tol)
        x0, y0, z0, x1, y1, z1 = box.Get()
        geom = BRep_Tool.Curve_s(self.edge, self.first, self.last)
        start, end = self.point(self.first), self.point(self.last)
        closed = _distance(start, end) <= tol
        found = []
        for v, p in fvm.boundary_points(self.faces, [x0, y0, z0], [x1, y1, z1]):
            if _distance(p, start) <= tol:
                found.append((self.first, v))
                if closed:
                    found.append((self.last, v))
                continue
            if _distance(p, end) <= tol:
                found.append((self.last, v))
                continue
            proj = GeomAPI_ProjectPointOnCurve(gp_Pnt(*p), geom, self.first, self.last)
            if proj.NbPoints() == 0 or proj.LowerDistance() > tol:
                continue
            found.append((proj.LowerDistanceParameter(), v))
        found.sort()
        if len(found) < 2 or found[0][0] != self.first or found[-1][0] != self.last:
            return
        if not fvm.is_chain([v for _, v in found]):
            return
        self.points = found


def preserve_face_volumes(mesh, pa, face_tolerances, edges=True, faces=True):
    """Move vertices of *mesh* (SurfaceMesh, in place) so each curved face of
    *pa* (ProcessedAssembly) encloses its CAD volume; see the module docstring.

    *face_tolerances* is ``{face_id: (deflection, angle)}`` as each face was
    meshed. *edges* and *faces* switch the two passes. Returns a dict of
    counts of what was done, and ``seconds``, the time it took.
    """
    start = time.perf_counter()
    stats = dict(
        edges_moved=0, edges_capped=0, edges_unmatched=0,
        faces_moved=0, faces_capped=0, faces_no_interior=0,
        faces_unmatched=0, faces_integration_failed=0,
    )
    fvm = FaceVolumeMesh(mesh.vertices, mesh.triangles, mesh.triangle_face_ids)
    curved = [fid for fid in fvm.faces() if not pa.faces[fid].is_planar]
    if not curved:
        stats["seconds"] = time.perf_counter() - start
        return stats

    # The CAD edges of every face next to a curved one, each once.
    edge_map = TopTools_IndexedMapOfShape()
    cad_edges = {}
    for fid in fvm.neighbours(curved):
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
    # the faces it bounds. Without the imprint, two solids that touch have an
    # edge each where they meet, and the weld can give both the same
    # vertices: those are one edge here.
    chains = {}
    for cad in list(cad_edges.values()):
        # A seam is explored twice in its face, once per side, and lies
        # inside it. (OCC also calls an edge closed on a face where it only
        # lies along the face's own seam line, as the end circle of a
        # sector of a torus does, so that is not the test.)
        if cad.seam:
            continue
        cad.locate(fvm)
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

    # The arc of CAD edge every boundary segment spans, for the edges that
    # move and the slivers of the curved faces.
    for chain, cad in chains.items():
        arcs = [cad.arc(ta, tb) for (ta, _), (tb, _) in zip(cad.points, cad.points[1:])]
        cad.index = fvm.add_edge(list(chain), arcs)

    if edges:
        for cad in cad_edges.values():
            if cad.points is None or cad.straight or len(cad.points) < 3:
                continue
            outward = [cad.outward(t) for t, _ in cad.points[1:-1]]
            cap = min(face_tolerances[fid][0] for fid in cad.faces)
            fvm.move_edge(cad.index, cad.faces, outward, cap)

    if faces:
        for fid in curved:
            apex = fvm.face_apex(fid)
            if apex is None:
                continue
            volume = _cad_face_volume(pa.face_to_occ[fid], apex)
            if volume is None:
                stats["faces_integration_failed"] += 1
                continue
            fvm.move_face(volume, face_tolerances[fid][0])

    mesh.vertices = fvm.vertices()
    stats.update(fvm.stats())
    stats["seconds"] = time.perf_counter() - start
    return stats


def _cad_face_volume(occ_face, apex):
    """The volume from *apex* over *occ_face*, with the face's own normal
    (the direction the stored triangles wind), or None when OCC cannot
    integrate it."""
    face = TopoDS.Face_s(occ_face.Oriented(TopAbs_FORWARD))
    cone = BRepGProp_VinertGK()
    cone.SetLocation(gp_Pnt(*apex))
    surface = BRepGProp_Face(face, True)
    if face.NbChildren() == 0:
        error = cone.Perform(surface, _GPROP_RELATIVE_PRECISION)
    else:
        error = cone.Perform(surface, BRepGProp_Domain(face), _GPROP_RELATIVE_PRECISION)
    if error < 0.0:
        return None
    return cone.Mass()
