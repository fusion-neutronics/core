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


def find_separated(mesh, solid_shapes, contact_tolerance, threads=None):
    """Solid pairs within *contact_tolerance* in *mesh* whose CAD solids do not
    touch, as 1-based ``(a, b)`` solid ids.

    A pair touches when any of its sample points (mesh vertices of one solid
    near the other) lies within a thousandth of the tolerance of the other CAD
    solid, inside or on it. A pair whose meshes share a vertex, or only come
    close between vertices, has no samples and counts as touching. The search
    runs on *threads* threads (``None``: rayon's global pool).
    """
    solids = sorted(_solid_triangles(mesh).items())
    eps = 1e-3 * contact_tolerance
    samples = list(
        solid_contact_samples(mesh.vertices, solids, contact_tolerance, threads=threads)
    )
    separated = []
    cad = None
    # Grouped by the other solid, so one solid's extrema are held at a time.
    for k in sorted(range(len(samples)), key=lambda k: samples[k][1]):
        a, b, points = samples[k]
        shape = solid_shapes.get(b)
        if not points or shape is None:
            continue
        if cad is None or cad.id != b:
            cad = _CadSolid(b, getattr(shape, "wrapped", shape))
        if not any(cad.touches(p, eps) for p in points):
            separated.append((k, (min(a, b), max(a, b))))
    return [pair for _, pair in sorted(separated)]


class _CadSolid:
    """One CAD solid, set up once for many "inside, on, or within eps of it"
    queries.

    The nearest boundary point is found exactly: faces and edges are visited
    in order of their bounding box distance, and the search stops when no box
    can hold anything nearer. Where the nearest point is the foot of a
    perpendicular inside a face, the outward normal there says which side the
    query point is on; where it is on an edge between two faces, the sum of
    their outward normals does. For a valid solid that is the solid
    classifier's answer, at a small fraction of its cost on B-spline faces.
    Whatever this cannot settle (a nearest vertex, a degenerate normal, an
    edge with other than two faces, nearest points that disagree) goes to the
    classifier, and an extrema search that fails goes to the distance query
    against the whole solid.
    """

    def __init__(self, solid_id, shape):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve2d, BRepAdaptor_Surface
        from OCP.BRepExtrema import BRepExtrema_ExtPC, BRepExtrema_ExtPF
        from OCP.Extrema import Extrema_ExtAlgo, Extrema_ExtFlag
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_IndexedMapOfShape,
        )

        self.id, self.shape, self.classifier = solid_id, shape, None
        # The distance query treats only a solid as filled: a compound
        # holding one is just its faces, and nothing is inside it.
        self.filled = shape.ShapeType() == TopAbs_SOLID
        self.faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            extrema = BRepExtrema_ExtPF()
            extrema.Initialize(
                face,
                Extrema_ExtFlag.Extrema_ExtFlag_MINMAX,
                Extrema_ExtAlgo.Extrema_ExtAlgo_Grad,
            )
            self.faces.append((_bounds(face), face, extrema, BRepAdaptor_Surface(face)))
            explorer.Next()
        ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)
        self.edges = []
        for k in range(1, ancestors.Extent() + 1):
            edge = TopoDS.Edge_s(ancestors.FindKey(k))
            if BRep_Tool.Degenerated_s(edge):
                continue
            extrema = BRepExtrema_ExtPC()
            extrema.Initialize(edge)
            sides = [
                (face, BRepAdaptor_Curve2d(edge, face), BRepAdaptor_Surface(face))
                for face in map(TopoDS.Face_s, ancestors.FindFromIndex(k))
            ]
            self.edges.append((_bounds(edge), extrema, sides))
        vertices = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_VERTEX, vertices)
        self.vertices = [
            BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.FindKey(k))).Coord()
            for k in range(1, vertices.Extent() + 1)
        ]

    def touches(self, p, eps):
        """Is *p* inside this solid, on it, or within *eps* of it?"""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.gp import gp_Pnt

        point = gp_Pnt(*p)
        vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
        best = min((_box_gap((*v, *v), p) for v in self.vertices), default=float("inf"))
        nearest = [(best, None, None)]
        queue = sorted(
            [(_box_gap(box, p), True, k) for k, (box, *_) in enumerate(self.faces)]
            + [(_box_gap(box, p), False, k) for k, (box, *_) in enumerate(self.edges)]
        )
        for gap, is_face, k in queue:
            if gap > _near(best):
                break
            if is_face:
                _, face, extrema, _ = self.faces[k]
                extrema.Perform(vertex, face)
            else:
                extrema = self.edges[k][1]
                extrema.Perform(vertex)
            if not extrema.IsDone():
                dist = BRepExtrema_DistShapeShape(vertex, self.shape)
                return not dist.IsDone() or dist.Value() <= eps
            for n in range(1, extrema.NbExt() + 1):
                d = extrema.SquareDistance(n) ** 0.5
                if d <= _near(best):
                    nearest.append((d, is_face, (k, extrema.Parameter(n))))
                    best = min(best, d)
        if best <= eps or not self.filled:
            return best <= eps
        outside = {
            self._outside(point, d, is_face, where)
            for d, is_face, where in nearest
            if d <= _near(best)
        }
        outside.discard("boundary")
        if len(outside) == 1 and None not in outside:
            return not outside.pop()
        return self._classify(point)

    def _outside(self, point, d, is_face, where):
        """Which side *point* is on, judged at one of its nearest boundary
        points, *d* away: True outside, False inside, None when this point
        cannot tell, and "boundary" for a face extremum on the face's edge,
        which the edge extrema judge instead."""
        from OCP.gp import gp_Vec

        if is_face is None:
            return None
        if is_face:
            k, (u, v) = where
            _, face, _, surface = self.faces[k]
            q, normal = _outward(face, surface, u, v)
            if normal is None:
                return None
            s = gp_Vec(q, point).Dot(normal)
            # The foot of a perpendicular lies along the normal; an extremum
            # that does not is on the face's boundary.
            return s > 0 if abs(s) >= 0.9 * d else "boundary"
        k, t = where
        sides = self.edges[k][2]
        if len(sides) != 2:
            return None
        total = gp_Vec(0.0, 0.0, 0.0)
        for face, curve, surface in sides:
            uv = curve.Value(t)
            q, normal = _outward(face, surface, uv.X(), uv.Y())
            if normal is None:
                return None
            total.Add(normal)
        if total.Magnitude() < 1e-6:
            return None
        total.Normalize()
        s = gp_Vec(q, point).Dot(total)
        return s > 0 if abs(s) >= 0.1 * d else None

    def _classify(self, point):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON

        if self.classifier is None:
            self.classifier = BRepClass3d_SolidClassifier(self.shape)
        self.classifier.Perform(point, 1e-7)
        return self.classifier.State() in (TopAbs_IN, TopAbs_ON)


def _near(best):
    """Distances this close to the nearest so far count as equally near."""
    return best * (1 + 1e-6) + 1e-12


def _box_gap(box, p):
    """Distance from *p* to an axis-aligned box, a lower bound on its distance
    to anything inside the box."""
    x0, y0, z0, x1, y1, z1 = box
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    dz = max(z0 - p[2], 0.0, p[2] - z1)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _bounds(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box.Get()


def _outward(face, surface, u, v):
    """The point at (*u*, *v*) on *face* and the unit outward normal there,
    None where the surface is degenerate."""
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_REVERSED

    q, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface.D1(u, v, q, du, dv)
    normal = du.Crossed(dv)
    if normal.Magnitude() < 1e-12:
        return q, None
    normal.Normalize()
    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()
    return q, normal
