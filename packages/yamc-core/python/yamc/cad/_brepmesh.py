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

from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location


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
    interior nodes as well.
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
    with face-local 0-based triangle indices, degenerate triangles included
    (:func:`~yamc.cad._scene_extract.assemble_surface_mesh` drops them).
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

        if tris:
            result[fid] = (verts, tris)
    return result
