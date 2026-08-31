"""Mesh fidelity gate: compare the surface mesh against the CAD it came from.

yamc's surface-adjacency tracker never re-locates particles in space, so a
topologically consistent but geometrically wrong surface mesh transports
cleanly, loses no particles, and produces confident wrong tallies (issue
#252: a reactor radial build meshed into 2.6k twisted triangles at a 0.1 cm
tolerance request ran at full speed with the thermal shield flux low by
2.3x). This gate makes such failures loud at meshing time.

Two deterministic checks, both against exact OCC properties:

- per-face area: the triangulated area of every face against the OCC face
  area. Chordal meshes inscribe, so a correct mesh is low by O(d/r); the
  degenerate seam-wire meshes this catches sit near half the true area.
- per-solid volume: the divergence-theorem volume of every solid shell
  against the OCC solid volume, catching global distortion that happens to
  preserve area.

Tolerances scale with the requested deflection so legitimately coarse
meshes of small parts do not trip the gate.
"""

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


class MeshFidelityError(ValueError):
    """The surface mesh deviates grossly from the CAD it was meshed from."""


def _occ_face_area(occ_face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(occ_face, props)
    return props.Mass()


def _occ_solid_volume(shape):
    occ = getattr(shape, "wrapped", shape)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(occ, props)
    return abs(props.Mass())


def check_mesh_fidelity(mesh, pa, tolerance):
    """Compare *mesh* (SurfaceMesh) against the OCC shapes in *pa*.

    Args:
        mesh: SurfaceMesh from the assembler.
        pa: ProcessedAssembly (provides face_to_occ, solid_shapes,
            face_solid_reversed).
        tolerance: the chordal deflection the mesh was requested at
            (3D units); fidelity thresholds scale with it.

    Returns:
        List of human-readable problem strings (empty when the mesh is
        faithful).
    """
    problems = []

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
        allowed = max(0.02, 4.0 * tolerance / max(scale, 1e-12))
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
        vol_occ = _occ_solid_volume(shape)
        if vol_occ <= 0.0:
            continue
        total = 0.0
        area_sum = 0.0
        for fid in fids:
            sign = -1.0 if mesh.face_solid_reversed.get((sid, fid)) else 1.0
            total += sign * face_signed_vol.get(fid, 0.0)
            area_sum += face_area.get(fid, 0.0)
        vol_mesh = abs(total)
        r_eff = 3.0 * vol_occ / max(area_sum, 1e-12)
        allowed = max(0.02, 8.0 * tolerance / max(r_eff, 1e-12))
        rel = abs(vol_mesh - vol_occ) / vol_occ
        if rel > allowed:
            problems.append(
                f"solid {sid}: mesh volume {vol_mesh:.6g} vs CAD {vol_occ:.6g} "
                f"({rel * 100:.1f}% off, allowed {allowed * 100:.1f}%)"
            )

    return problems
