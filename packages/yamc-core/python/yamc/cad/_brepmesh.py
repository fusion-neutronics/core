"""BRepMesh surface meshing for all-primitive solids.

Mirrors cad-to-dagmc-mesher's routing: solids whose faces are all primitive OCC
surfaces (plane / cylinder / cone / sphere / torus) are meshed with OCC
``BRepMesh``. BRepMesh uses the *harmonized OCC edge discretization*, so
adjacent faces share boundary vertices and closed curved surfaces (sphere,
torus) come out watertight -- whereas the scene CDT struggles to close them
from their degenerate seam/pole wires. Imported / b-spline faces stay on the
scene CDT (see :mod:`yamc.cad._scene_extract`), where BRepMesh is poor.
"""

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location

# Squared cross-product magnitude below which a triangle is degenerate.
_MIN_TRI_CROSS_SQ = 1e-24


def run_brepmesh(shape, tolerance, angular_tolerance):
    """Run ``BRepMesh_IncrementalMesh`` with interior deflection control.

    The simple ``(shape, deflection, relative, angle)`` constructor leaves
    ``IMeshTools_Parameters.AngleInterior`` unset, and OCC then grids the
    interior of periodic BSpline faces (full-torus shells) with cells that
    violate the angular tolerance by an order of magnitude in a band next
    to the u-seam (metre-long chords at a millimetre deflection request).
    On nested thin shells the seam bands of adjacent surfaces cross each
    other even though every shell is watertight. Setting ``AngleInterior``
    (and ``DeflectionInterior``) explicitly enforces the tolerances on
    interior nodes as well; see cad-to-dagmc-mesher issue #100.
    ``InParallel`` stays False to keep the triangulation reproducible.
    """
    from OCP.IMeshTools import IMeshTools_Parameters

    params = IMeshTools_Parameters()
    params.Deflection = tolerance
    params.DeflectionInterior = tolerance
    params.Angle = angular_tolerance
    params.AngleInterior = angular_tolerance
    params.InParallel = False
    BRepMesh_IncrementalMesh(shape, params)


def brepmesh_face_meshes(pa, face_ids, tolerance, angular_tolerance):
    """Per-face BRepMesh triangulations for ``face_ids``.

    Runs ``BRepMesh_IncrementalMesh`` once on the imprinted compound (so the
    discretization is harmonized across faces) and extracts each requested
    face's triangulation in 3D. Returns ``{face_id: (vertices, triangles)}``
    with face-local 0-based triangle indices; degenerate triangles are dropped.
    """
    face_ids = list(face_ids)
    if not face_ids:
        return {}

    run_brepmesh(pa.imprinted_compound, tolerance, angular_tolerance)

    result = {}
    for fid in face_ids:
        occ_face = pa.face_to_occ[fid]
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(occ_face, loc)
        if tri is None:
            continue

        transform = loc.Transformation()
        has_transform = not loc.IsIdentity()
        verts = []
        for i in range(1, tri.NbNodes() + 1):
            pnt = tri.Node(i)
            if has_transform:
                pnt = pnt.Transformed(transform)
            verts.append([pnt.X(), pnt.Y(), pnt.Z()])

        tris = []
        for i in range(1, tri.NbTriangles() + 1):
            i1, i2, i3 = tri.Triangle(i).Get()
            if i1 == i2 or i2 == i3 or i1 == i3:
                continue
            a, b, c = verts[i1 - 1], verts[i2 - 1], verts[i3 - 1]
            e1 = [b[j] - a[j] for j in range(3)]
            e2 = [c[j] - a[j] for j in range(3)]
            cross_sq = (
                (e1[1] * e2[2] - e1[2] * e2[1]) ** 2
                + (e1[2] * e2[0] - e1[0] * e2[2]) ** 2
                + (e1[0] * e2[1] - e1[1] * e2[0]) ** 2
            )
            if cross_sq < _MIN_TRI_CROSS_SQ:
                continue
            tris.append([i1 - 1, i2 - 1, i3 - 1])

        if tris:
            result[fid] = (verts, tris)
    return result
