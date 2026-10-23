"""Mesh fidelity gate: compare the surface mesh against the CAD it came from.

yamc's surface-adjacency tracker never re-locates particles in space, so a
topologically consistent but geometrically wrong surface mesh transports
cleanly, loses no particles, and produces confident wrong tallies (a
reactor radial build meshed into 2.6k twisted triangles at a 0.1 cm tolerance
request ran at full speed with the thermal shield flux low by 2.3x). This gate makes such failures loud at meshing time.

Two deterministic checks, both against the face areas and solid volumes OCC
integrates from the CAD (see _occ_face_area and _occ_signed_volume):

- per-face area: the triangulated area of every face against the OCC face
  area. Chordal meshes inscribe, so a correct mesh is low by O(d/r); the
  degenerate seam-wire meshes this catches sit near half the true area.
- per-solid volume: the divergence-theorem volume of every solid shell
  against the OCC solid volume, catching global distortion that happens to
  preserve area. A negative mesh volume means the facets enclose the solid
  the wrong way round (inside out, or with surfaces that cross), which
  transport reads as the solid's complement. A CAD solid whose own volume is
  negative (a reversed solid) is reported as such.

Tolerances scale with the requested deflection so legitimately coarse
meshes of small parts do not trip the gate.
"""

from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCP.BRepGProp import (
    BRepGProp, BRepGProp_Domain, BRepGProp_Face, BRepGProp_VinertGK,
)
from OCP.GeomAbs import (
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder,
    GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus,
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt

# OCC's default integration is a fixed Gauss rule, which misses dense spline
# faces badly (a stellarator coil casing came out 31% low in volume), so every
# face is integrated adaptively to this relative precision instead.
_GPROP_RELATIVE_PRECISION = 1e-6

# Surfaces OCC integrates piece by piece: the analytic ones, and splines,
# which it splits at their knots. Any other face (swept, offset) it
# integrates as a single piece.
_PIECEWISE_SURFACES = frozenset({
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
})


class MeshFidelityError(ValueError):
    """The surface mesh deviates grossly from the CAD it was meshed from."""


def _occ_face_area(occ_face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(occ_face, props, _GPROP_RELATIVE_PRECISION)
    if BRepAdaptor_Surface(occ_face).GetType() in _PIECEWISE_SURFACES:
        return props.Mass()
    # A single piece over a dense profile can settle on a value tens of
    # percent off that no tighter precision moves (a spline swept into a
    # wall comes out 1.8% high). Where OCC's fixed rule agrees to 1e-3, as it
    # does on smooth faces, the value is kept: right to about that, well
    # inside what the gate allows. Elsewhere the face is integrated again as
    # a B-spline surface, which OCC splits at its knots. Converting projects
    # every edge onto the new surface, up to a second a face on a reactor
    # model, so only the faces that need it are converted.
    fixed = GProp_GProps()
    BRepGProp.SurfaceProperties_s(occ_face, fixed)
    if abs(fixed.Mass() - props.Mass()) <= 1e-3 * props.Mass():
        return props.Mass()
    spline = TopoDS.Face_s(BRepBuilderAPI_NurbsConvert(occ_face, True).Shape())
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(spline, props, _GPROP_RELATIVE_PRECISION)
    return props.Mass()


def _occ_signed_volume(shape):
    """Volume of *shape*, negative when its faces point into it.

    Summed over the faces, each contributing the cone it spans to an apex,
    integrated by Gauss-Kronrod on every knot span (the adaptive form of
    VolumeProperties can settle on a wrong value for dense splines: 19% high
    on a test surface, 69% on an extruded spline profile). An offset face is
    integrated in one piece, which on some dense ones takes seconds (9 s for
    the inner wall of a shelled 32-point spline extrusion). A B-spline copy,
    as the area uses, only approximates it (3e-3 off on a thickened dense
    spline face), too loosely for the overlap check, which compares volumes
    to 1e-4.
    """
    occ = getattr(shape, "wrapped", shape)
    # Gauss-Kronrod meets its relative precision face by face, so it never
    # converges on a face whose cone is flat, such as a planar face through
    # the apex. OCC's own apex, the centroid of the vertices, lies in the
    # flat face of a half cylinder or of a half turn, and tilted off the axes
    # those took from 20 s to 3 min. The middle of the bounding box, moved off
    # by a few percent in an arbitrary direction, lies in the plane of no face
    # of ordinary CAD.
    box = Bnd_Box()
    BRepBndLib.Add_s(occ, box, False)
    if box.IsVoid():  # no faces, so no volume
        return 0.0
    x0, y0, z0, x1, y1, z1 = box.Get()
    size = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
    apex = gp_Pnt(
        0.5 * (x0 + x1) + 0.0137 * size,
        0.5 * (y0 + y1) + 0.0291 * size,
        0.5 * (z0 + z1) + 0.0213 * size,
    )
    volume = 0.0
    faces = TopExp_Explorer(occ, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        faces.Next()
        if face.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
            continue
        cone = BRepGProp_VinertGK()
        cone.SetLocation(apex)
        surface = BRepGProp_Face(face, True)  # split splines at their knots
        if face.NbChildren() == 0:  # no boundary: the whole surface
            error = cone.Perform(surface, _GPROP_RELATIVE_PRECISION)
        else:
            error = cone.Perform(
                surface, BRepGProp_Domain(face), _GPROP_RELATIVE_PRECISION
            )
        if error < 0.0:
            raise ValueError(
                "OCC could not integrate the volume of a CAD solid: Gauss-Kronrod "
                "failed on one of its faces"
            )
        volume += cone.Mass()
    return volume


def check_mesh_fidelity(mesh, pa, tolerance):
    """Compare *mesh* (SurfaceMesh) against the OCC shapes in *pa*.

    Args:
        mesh: SurfaceMesh from the assembler.
        pa: ProcessedAssembly (provides face_to_occ, solid_shapes and,
            when the imprint ran, solid_volumes).
        tolerance: the chordal deflection the mesh was requested at
            (3D units), one for every solid or ``{solid_id: deflection}``;
            fidelity thresholds scale with it. A solid's volume is held to
            its own deflection, a face's area to the finest of the solids
            it bounds (it was meshed at that or finer).

    Returns:
        List of human-readable problem strings (empty when the mesh is
        faithful).
    """
    problems = []
    if not isinstance(tolerance, dict):
        tolerance = {sid: tolerance for sid in mesh.solid_faces}
    face_tolerance = {
        fid: min(tolerance[sid] for sid in info.owning_solids)
        for fid, info in pa.faces.items()
    }

    # Group triangle areas and signed volume contributions per face.
    face_area = {}
    face_signed_vol = {}
    verts = mesh.vertices
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        a = verts[tri[0]]
        b = verts[tri[1]]
        c = verts[tri[2]]
        abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cx = aby * acz - abz * acy
        cy = abz * acx - abx * acz
        cz = abx * acy - aby * acx
        face_area[fid] = face_area.get(fid, 0.0) + 0.5 * (
            cx * cx + cy * cy + cz * cz
        ) ** 0.5
        # (1/6) a . (b x c): signed volume of the origin tetrahedron
        bxc_x = b[1] * c[2] - b[2] * c[1]
        bxc_y = b[2] * c[0] - b[0] * c[2]
        bxc_z = b[0] * c[1] - b[1] * c[0]
        face_signed_vol[fid] = face_signed_vol.get(fid, 0.0) + (
            a[0] * bxc_x + a[1] * bxc_y + a[2] * bxc_z
        ) / 6.0

    # Per-face area check. A chordal mesh inscribes the true surface, so a
    # correct mesh's area deficit is O(tolerance / curvature radius); allow a
    # generous multiple of that (using an area-derived length scale), floored
    # at 2%. The seam-wire degenerate meshes lose 30-60% of the area.
    for fid, occ_face in pa.face_to_occ.items():
        area_occ = _occ_face_area(occ_face)
        if area_occ <= 0.0:
            continue
        area_mesh = face_area.get(fid, 0.0)
        scale = area_occ ** 0.5
        allowed = max(0.02, 4.0 * face_tolerance[fid] / max(scale, 1e-12))
        rel = abs(area_mesh - area_occ) / area_occ
        if rel > allowed:
            problems.append(
                f"face {fid}: mesh area {area_mesh:.6g} vs CAD {area_occ:.6g} "
                f"({rel * 100:.1f}% off, allowed {allowed * 100:.1f}%)"
            )

    # Per-solid volume check. Face contributions are signed by the stored
    # winding; face_solid_reversed says whether that winding points into the
    # solid. The length scale 3V/A (the inradius for simple shapes, the
    # half-thickness for thin shells) converts the deflection into an
    # expected relative volume error.
    for sid, fids in mesh.solid_faces.items():
        shape = pa.solid_shapes.get(sid)
        if shape is None:
            continue
        vol_occ = pa.solid_volumes.get(sid)
        if vol_occ is None:
            vol_occ = _occ_signed_volume(shape)
        if vol_occ < 0.0:
            problems.append(
                f"solid {sid}: inside out in the CAD (volume {vol_occ:.6g}), "
                f"its faces point into it, as for a reversed solid"
            )
            continue
        if vol_occ == 0.0:
            continue
        total = 0.0
        area_sum = 0.0
        for fid in fids:
            sign = -1.0 if mesh.face_solid_reversed.get((sid, fid)) else 1.0
            total += sign * face_signed_vol.get(fid, 0.0)
            area_sum += face_area.get(fid, 0.0)
        r_eff = 3.0 * vol_occ / max(area_sum, 1e-12)
        allowed = max(0.02, 8.0 * tolerance[sid] / max(r_eff, 1e-12))
        if total < 0.0:
            # Transport would read the solid from the wrong side of its
            # facets. A mesh pointing the wrong way everywhere has the right
            # volume, negated; anything else has surfaces that cross.
            if abs(-total - vol_occ) <= min(allowed, 0.1) * vol_occ:
                problems.append(
                    f"solid {sid}: meshed inside out, its facets point into it "
                    f"(mesh volume {total:.6g} vs CAD {vol_occ:.6g})"
                )
            else:
                problems.append(
                    f"solid {sid}: mesh volume {total:.6g} vs CAD {vol_occ:.6g} "
                    f"is negative: its surfaces cross, or its facets point into it"
                )
            continue
        rel = abs(total - vol_occ) / vol_occ
        if rel > allowed:
            problems.append(
                f"solid {sid}: mesh volume {total:.6g} vs CAD {vol_occ:.6g} "
                f"({rel * 100:.1f}% off, allowed {allowed * 100:.1f}%)"
            )

    return problems
