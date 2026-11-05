"""Main CadToYamc class -- end-to-end surface meshing pipeline.

Takes a CadQuery assembly, imprints it (unless ``imprint=False``), meshes each
face with the ``cad-to-dagmc-mesher`` scene mesher (see
:mod:`yamc.cad._scene_extract`), and produces a global vertex + triangle mesh
with per-face and per-volume metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Literal

import math
import numbers
import os
import warnings

from .assembly_processor import process_assembly
from . import cad_mesh_labels as _cad_mesh_labels
from . import cad_mesh_to_arrow as _cad_mesh_to_arrow
from . import weld_mesh as _weld_mesh

if TYPE_CHECKING:
    import cadquery

# Surface meshing tolerances of a solid that the caller does not set: the
# chordal deflection (3D units) and the angular deflection (radians).
_DEFAULT_TOLERANCE = 0.1
_DEFAULT_ANGULAR_TOLERANCE = 0.1


@dataclass
class SurfaceMesh:
    """Result of surface meshing an assembly."""
    vertices: list  # list of [x, y, z]
    triangles: list  # list of [v0, v1, v2] (indices into vertices)
    triangle_face_ids: list  # face_id for each triangle
    triangle_surface_ids: list  # surface_id for each triangle (1-indexed)
    face_to_surface_id: dict  # face_id -> surface_id
    solid_faces: dict  # solid_id -> list of face_ids
    shared_faces: dict  # face_id -> [solid_id, solid_id]
    material_tags: list  # ordered material tags
    num_solids: int
    num_faces: int
    face_solid_reversed: dict = field(default_factory=dict)
    # (solid_id, face_id) -> bool: True when face is REVERSED in that solid
    # (UV surface normal points inward to the solid)


def _get_leaf_children(assembly):
    """Yield all leaf children (parts without sub-children) of an assembly."""
    for child in assembly.children:
        if hasattr(child, "children") and len(child.children) > 0:
            yield from _get_leaf_children(child)
        else:
            yield child


def _resolve_material_tags(assembly, shortcut):
    """Resolve a string shortcut into a list of material tag strings.

    Supported shortcuts:
        ``"assembly_names"``      -- use each part's ``.name``
        ``"assembly_materials"``  -- use each part's ``.material.name``
    """
    if shortcut == "assembly_names":
        return [child.name for child in _get_leaf_children(assembly)]

    if shortcut == "assembly_materials":
        tags = []
        for child in _get_leaf_children(assembly):
            if child.material is not None and child.material.name is not None:
                tags.append(str(child.material.name))
            else:
                raise ValueError(
                    f"Part {child.name!r} has no material assigned. "
                    "Assign materials with cq.Material('name') or provide "
                    "an explicit material_tags list."
                )
        return tags

    raise ValueError(
        f"material_tags string must be 'assembly_names' or "
        f"'assembly_materials', got {shortcut!r}"
    )


def _positive_finite(value, label, expected="a number"):
    """*value* as a float, refusing anything but a positive finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be {expected}, got {type(value).__name__}")
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"{label} must be positive and finite, got {value!r}")
    return float(value)


def _check_per_solid(value, name, material_tags):
    """Check a per-solid setting before the imprint, so a mistake fails fast.

    *value* is a number for every solid, or a dict from 1-based solid id or
    material tag to a number. Returns it with its keys as int or str and its
    numbers as floats. Solid ids are only checked to be positive here: how
    many solids there are is known after the imprint, see :func:`_per_solid`.
    """
    if not isinstance(value, dict):
        return _positive_finite(
            value, name, "a number, or a dict from solid id or material tag to a number"
        )
    checked = {}
    for key, number in value.items():
        if isinstance(key, bool) or not isinstance(key, (numbers.Integral, str)):
            raise TypeError(
                f"{name} keys must be int (solid id) or str (material tag), "
                f"got {type(key).__name__}"
            )
        if isinstance(key, str):
            key = str(key)
            if key not in material_tags:
                raise ValueError(
                    f"Material tag {key!r} in {name} not found. Available material "
                    f"tags are: {sorted(set(material_tags))}"
                )
        else:
            key = int(key)
            if key < 1:
                raise ValueError(f"{name} key {key} is not a solid id: solid ids start at 1")
        checked[key] = _positive_finite(number, f"{name}[{key!r}]")
    return checked


def _per_solid(value, name, default, solid_tags):
    """``{solid_id: value}`` for every solid, from a setting checked by
    :func:`_check_per_solid`.

    *solid_tags* is each solid's material tag in solid id order. A material
    tag sets every solid with that tag; solids the dict does not name take
    *default*.
    """
    count = len(solid_tags)
    if not isinstance(value, dict):
        return {sid: value for sid in range(1, count + 1)}
    resolved = {}
    for key, number in value.items():
        if isinstance(key, int):
            if key > count:
                raise ValueError(
                    f"{name} key {key} is not a solid id: the largest solid id is {count}"
                )
            sids = [key]
        else:
            sids = [i + 1 for i, tag in enumerate(solid_tags) if tag == key]
            if not sids:
                raise ValueError(
                    f"Material tag {key!r} in {name} is on no solid. The solids' "
                    f"material tags are: {sorted(set(solid_tags))}"
                )
        for sid in sids:
            if resolved.setdefault(sid, number) != number:
                raise ValueError(
                    f"Solid {sid} ({solid_tags[sid - 1]!r}) is given two {name} "
                    f"values, {resolved[sid]!r} and {number!r}, by solid id and "
                    "by material tag. Give each solid one value."
                )
    return {sid: resolved.get(sid, default) for sid in range(1, count + 1)}


# A solid's surfaces are meshed at its own values, except those it shares with
# a finer solid, at that one's. Where a solid is thin and has curved surfaces of
# both kinds, its own facets sag toward the finer ones by up to the difference,
# which comes off its thickness (one value everywhere sags both sides alike),
# and they can cross. So such a solid is meshed no coarser than this fraction
# of 3V/A, 1.5 times the thickness of a thin wall, which keeps what it loses
# (of its volume, and of the area of a flat rim between a fine and a coarse
# edge) within the fidelity gate's 2%; a solid too thin for that takes the
# finest values of its curved surfaces, as if meshed at one value.
_THIN_WALL_FRACTION = 0.02


def _cap_thin_solids(pa, solid_tolerance, solid_angular_tolerance):
    """``(deflections, angles)`` per solid, with every solid whose curved faces
    are meshed at different values held to a fraction of its thickness; see
    :data:`_THIN_WALL_FRACTION`."""
    from ._fidelity import _occ_signed_volume

    lin, ang = dict(solid_tolerance), dict(solid_angular_tolerance)
    thickness = {}
    changed = True
    while changed:
        changed = False
        for sid, fids in pa.solid_faces.items():
            curved = {
                (min(lin[s] for s in pa.faces[fid].owning_solids),
                 min(ang[s] for s in pa.faces[fid].owning_solids))
                for fid in fids
                if not pa.faces[fid].is_planar
            }
            if len(curved) < 2:
                continue
            if sid not in thickness:
                area = sum(pa.faces[fid].surface_area for fid in dict.fromkeys(fids))
                volume = abs(_occ_signed_volume(pa.solid_shapes[sid]))
                thickness[sid] = 3.0 * volume / max(area, 1e-300)
            cap = _THIN_WALL_FRACTION * thickness[sid]
            finest = min(v[0] for v in curved)
            if cap > finest:
                capped = (min(lin[sid], cap), ang[sid])
            else:
                capped = (min(lin[sid], finest), min(ang[sid], *(v[1] for v in curved)))
            if capped != (lin[sid], ang[sid]):
                lin[sid], ang[sid] = capped
                changed = True
    return lin, ang


def _weld_solid_boundary(vertices, triangles, rel_tol=1e-6):
    """Weld a solid's surface triangles into a watertight boundary.

    The scene surface mesh stores per-face-duplicated boundary vertices
    (coincident at shared edges but with distinct indices). The yamm tet
    mesher needs a closed, index-shared boundary, so coincident vertices are
    merged by coordinate in the Rust core. Returns ``(boundary_vertices,
    boundary_triangles)`` with triangles reindexed into the local welded vertex
    list (seams that collapse to degenerate triangles are dropped).
    """
    local_verts, boundary_triangles, _kept = _weld_mesh(vertices, triangles, rel_tol)
    return local_verts, boundary_triangles


def _default_tet_edge_length(vertices):
    """Default target tet edge length: ~1/10 of the bounding-box diagonal."""
    extents = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
    diag = sum(e * e for e in extents) ** 0.5
    return max(diag / 10.0, 1e-9)


def _route_faces(pa):
    """Route faces to BRepMesh vs the scene CDT, mirroring cad-to-dagmc-mesher.

    A face goes to BRepMesh iff every solid owning it is "all-primitive" -- all
    of that solid's faces are primitive OCC surfaces (plane / cylinder / cone /
    sphere / torus) or surfaces of revolution -- or the face is a closed
    doubly-periodic surface (see ``_is_closed_doubly_periodic``). Faces of
    solids with any other b-spline / imported surface go to the scene CDT.
    Returns ``(brep_face_ids, scene_face_ids)``.

    Surfaces of revolution count as primitive because the scene CDT meshes
    them short: a profile revolved about an axis is u-periodic, and the CDT
    closes its seam on itself (it logs "self-touching face boundary"), leaving
    paramak tokamaks 40% short in face area and up to 99.9% short in solid
    volume. BRepMesh meshes them to the requested deflection.

    A face shared with a solid that is not all-primitive goes to the scene
    CDT, and the two meshers discretize the edges between it and its
    neighbours differently, which leaves cracks the fidelity gate cannot see.
    So a solid with a surface of revolution goes to BRepMesh only if every
    solid it shares a face with does too; otherwise it stays on the scene CDT
    entirely, as before surfaces of revolution were routed here, where a
    short mesh fails the fidelity gate rather than cracking.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_Plane,
        GeomAbs_Cylinder,
        GeomAbs_Cone,
        GeomAbs_Sphere,
        GeomAbs_Torus,
        GeomAbs_SurfaceOfRevolution,
    )

    primitive_types = {
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
        GeomAbs_SurfaceOfRevolution,
    }
    surface_types = {
        sid: [BRepAdaptor_Surface(pa.face_to_occ[fid]).GetType() for fid in fids]
        for sid, fids in pa.solid_faces.items()
    }
    brep_solids = {
        sid for sid, types in surface_types.items()
        if all(t in primitive_types for t in types)
    }
    revolved = {
        sid for sid, types in surface_types.items()
        if GeomAbs_SurfaceOfRevolution in types
    }
    neighbours = {sid: set() for sid in pa.solid_faces}
    for info in pa.faces.values():
        owners = set(info.owning_solids)
        for sid in owners:
            neighbours.setdefault(sid, set()).update(owners - {sid})
    # Dropping one solid can expose another, so repeat until nothing changes.
    changed = True
    while changed:
        drop = {
            sid for sid in brep_solids & revolved
            if not neighbours.get(sid, set()) <= brep_solids
        }
        brep_solids -= drop
        changed = bool(drop)

    brep_fids, scene_fids = [], []
    for fid, info in pa.faces.items():
        owners = info.owning_solids
        if owners and all(s in brep_solids for s in owners):
            brep_fids.append(fid)
        elif _is_closed_doubly_periodic(pa.face_to_occ[fid]):
            # Closed doubly-periodic b-spline faces (full-torus shells,
            # common in imported reactor CAD) also go to BRepMesh: the scene
            # CDT meshes them from their seam wires as a coarse twisted
            # shell that ignores the deflection tolerance (the known gap
            # noted in _scene_extract). Such a face has no real boundary
            # edges (only seams / degenerate edges), so it shares no edge
            # discretization with scene-CDT neighbours and mixing backends
            # cannot open a crack.
            brep_fids.append(fid)
        else:
            scene_fids.append(fid)
    return brep_fids, scene_fids


def _is_closed_doubly_periodic(occ_face):
    """True for a doubly-periodic face whose only edges are seams or
    degenerate edges, i.e. a closed surface like a full torus shell."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    surf = BRepAdaptor_Surface(occ_face)
    if not (surf.IsUPeriodic() and surf.IsVPeriodic()):
        return False
    exp = TopExp_Explorer(occ_face, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        if not (
            BRep_Tool.IsClosed_s(edge, occ_face)
            or BRep_Tool.Degenerated_s(edge)
        ):
            return False
        exp.Next()
    return True


class CadToYamc:
    """Convert CadQuery assemblies to surface/volume meshes."""

    def __init__(self) -> None:
        """Initialize an empty CadToYamc converter."""
        self._assembly = None
        self._material_tags = None
        self._processed = None
        self._surface_mesh = None
        self._tet_data = {}  # solid_id -> (tet_vertices, tet_connectivity)
        self._tet_index_maps = {}  # solid_id -> local vertex -> index in mesh.vertices
        self._imprint = True
        self._contact_tolerance = None
        self._overlaps = []  # (low, high) 1-based solid pairs, high takes the overlap
        self._volume_measures = None  # per-solid volumes when overlaps are resolved
        self._separated = []  # (a, b) 1-based solid pairs a real gap apart

    def add_cadquery_object(
        self,
        assembly: cadquery.Assembly,
        material_tags: list[str] | str,
    ) -> None:
        """Add a CadQuery assembly with material tags.

        Args:
            assembly: CadQuery Assembly with named parts.
            material_tags: List of material tag strings, one per solid,
                or the string ``"assembly_names"`` to use each part's
                name, or ``"assembly_materials"`` to use each part's
                ``cq.Material`` name.
        """
        if isinstance(material_tags, str):
            material_tags = _resolve_material_tags(assembly, material_tags)

        self._assembly = assembly
        self._material_tags = material_tags

    def mesh(
        self,
        tet_volumes: list[str] | None = None,
        target_edge_length: float | None = None,
        tolerance: float | dict[int | str, float] = _DEFAULT_TOLERANCE,
        angular_tolerance: float | dict[int | str, float] = _DEFAULT_ANGULAR_TOLERANCE,
        fidelity_check: bool = True,
        imprint: bool = True,
        contact_tolerance: float | None = None,
        overlaps: str = "error",
        *,
        imprint_threads: int | Literal["all"] = "all",
    ) -> SurfaceMesh:
        """Mesh the assembly: surfaces + optional tet volumes.

        Surfaces are meshed with the yamm scene mesher; volumes listed in
        *tet_volumes* are then filled with tetrahedra by the yamm tet mesher.

        Args:
            tet_volumes: List of material tag names to tet-mesh, or None
                for surface-only meshing.
            target_edge_length: Target tet edge length (3D units). Required
                when *tet_volumes* is set.
            tolerance: Chordal deflection tolerance for surface meshing
                (3D units).  Smaller values produce more triangles on
                curved surfaces.  A number applies to every solid. A dict
                sets it per solid, keyed by material tag (every solid with
                that tag), for instance ``{"first_wall": 0.01, "coil": 1.0}``,
                or by 1-based solid id (the keys of ``SurfaceMesh.solid_faces``,
                in the order of ``SurfaceMesh.material_tags``, which has an
                entry per solid, so a part holding several solids takes
                consecutive ids); solids it does not name take 0.1. A surface
                two solids share is meshed at the finer of their values, and
                an edge at the finest of its surfaces', so the mesh stays
                watertight. Some solids are meshed finer than they ask. A
                solid whose curved surfaces end up at different values is
                held to a fiftieth of 3V/A (a measure of its thickness), so
                its own facets cannot sag through a thin wall, and takes the
                finest of those values if thinner still. The placements of
                one shape (``assembly.add(shape, loc=...)``) can share their
                triangles, and then all take the finest of their values.
                Where OCC cannot keep touching surfaces watertight at
                different values, the surfaces joined to them are meshed at
                the finest value among them. Default 0.1.
            angular_tolerance: Angular deflection tolerance for surface
                meshing (radians), a number or a per-solid dict as for
                *tolerance*.  Default 0.1 (~5.7 degrees).
            fidelity_check: Refuse, raising ``MeshFidelityError``, a mesh
                whose face areas or solid volumes are far from the CAD's, or
                in which a solid's surface crosses itself (a wall thinner than
                its facets' sag, meshed at too loose an angular tolerance).
                Default True.
            imprint: Imprint the assembly first, so solids that touch share
                one surface where they meet (a conformal mesh). ``False``
                skips the imprint and meshes every solid on its own: much
                faster on a large assembly, and it accepts CAD that will not
                imprint cleanly. Touching solids then meet along two separately
                faceted surfaces that may overlap or leave a thin gap, and
                :class:`yamc.MeshGeometry` tracks the exported mesh
                discontinuously to suit. Default True.
            imprint_threads: ``"all"`` (default) uses all logical processors
                for imprinting, or pass a positive integer to limit its threads
                and reduce peak RAM. Does not change subsequent meshing. The
                prior OCC thread settings are restored even if imprinting fails.
                Validated but unused when imprinting is skipped.
            contact_tolerance: Only with ``imprint=False``: the distance
                (3D units) within which two solids count as touching, one for
                the whole model. A particle crosses an overlap or a gap
                thinner than this straight into the next solid. Defaults to
                twice *tolerance* (with per-solid values, twice the largest
                deflection of any solid, those the dict does not name
                counting at 0.1), which covers the slivers two facetings of
                one CAD surface leave between them; raise it to accept
                sloppier CAD. A gap that is in the CAD stays a gap whatever
                the tolerance: the solids either side are checked against
                the CAD and recorded as separated.
            overlaps: What to do when two solids of the mesh overlap deeper
                than their facetings explain, twice the larger deflection of
                the two (twice *tolerance* when it is a number), or deeper
                than *contact_tolerance* when that is given. The imprint
                shares the surfaces where solids coincide, but it does not
                split solids whose surfaces cross (a rod poking out of a
                box), so both come out of it whole and still overlap;
                without imprinting, a part embedded in a body that was never
                cut overlaps it in the same way.
                Transport would give each overlap to whichever solid a
                particle was already in. ``"error"`` (default) raises
                :class:`OverlappingSolidsError`. ``"smaller_wins"``, only with
                ``imprint=False``, gives every such overlap to the smaller
                solid of the pair, wherever a particle arrives from, and takes
                the shared volume off the larger one; the pairs travel to the
                transport in the mesh file. ``"ignore"`` skips the check.

        Returns:
            SurfaceMesh with global vertex/triangle arrays.
        """
        if self._assembly is None:
            raise RuntimeError("No assembly added. Call add_cadquery_object first.")

        if tet_volumes and target_edge_length is None:
            raise ValueError(
                "target_edge_length is required when tet_volumes is specified"
            )

        self._mesh_surfaces(
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            fidelity_check=fidelity_check,
            imprint=imprint,
            contact_tolerance=contact_tolerance,
            overlaps=overlaps,
            imprint_threads=imprint_threads,
        )

        if tet_volumes:
            self.mesh_volumes(tet_volumes, target_edge_length)

        return self._surface_mesh

    # ------------------------------------------------------------------
    # Lower-level methods
    # ------------------------------------------------------------------

    def mesh_surfaces(
        self,
        tolerance: float | dict[int | str, float] = _DEFAULT_TOLERANCE,
        angular_tolerance: float | dict[int | str, float] = _DEFAULT_ANGULAR_TOLERANCE,
        fidelity_check: bool = True,
        imprint: bool = True,
        contact_tolerance: float | None = None,
        overlaps: str = "error",
        *,
        imprint_threads: int | Literal["all"] = "all",
    ) -> SurfaceMesh:
        """Mesh all surfaces of the assembly.

        Args:
            tolerance: Chordal deflection tolerance for surface meshing
                (3D units): a number for every solid, or a dict from solid
                id or material tag to a number; see :meth:`mesh`.
            angular_tolerance: Angular deflection tolerance (radians), a
                number or a per-solid dict; see :meth:`mesh`.
            imprint: Imprint the assembly first; see :meth:`mesh`.
            imprint_threads: Threads for imprinting only; see :meth:`mesh`.
            contact_tolerance: See :meth:`mesh`.
            overlaps: See :meth:`mesh`.

        Returns:
            SurfaceMesh with global vertex/triangle arrays.
        """
        if self._assembly is None:
            raise RuntimeError("No assembly added. Call add_cadquery_object first.")

        return self._mesh_surfaces(
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            fidelity_check=fidelity_check,
            imprint=imprint,
            contact_tolerance=contact_tolerance,
            overlaps=overlaps,
            imprint_threads=imprint_threads,
        )

    def mesh_volumes(
        self,
        volumes_to_tet: list[str] | None = None,
        tet_target_edge_length: float | None = None,
    ) -> dict[int, tuple[list[list[float]], list[list[int]]]]:
        """Generate tetrahedral meshes for selected volumes.

        Args:
            volumes_to_tet: List of material tag names to tet-mesh.
                If None, tet-meshes all volumes.
            tet_target_edge_length: Target tet edge length (3D units). If
                None, a default derived from each solid's size is used.

        Returns:
            Dict mapping solid_id -> (tet_vertices, tet_connectivity).
        """
        if self._surface_mesh is None:
            raise RuntimeError("No surface mesh. Call mesh() or mesh_surfaces() first.")

        # Determine which solids to tet-mesh
        solids_to_tet = set()
        mesh = self._surface_mesh
        for solid_id in mesh.solid_faces:
            tag_idx = solid_id - 1
            if tag_idx < len(mesh.material_tags):
                tag = mesh.material_tags[tag_idx]
            else:
                tag = None
            if volumes_to_tet is None or tag in volumes_to_tet:
                solids_to_tet.add(solid_id)

        self._mesh_volumes(
            solids_to_tet,
            tet_target_edge_length=tet_target_edge_length,
        )
        return self._tet_data

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _mesh_surfaces(
        self,
        tolerance=_DEFAULT_TOLERANCE,
        angular_tolerance=_DEFAULT_ANGULAR_TOLERANCE,
        fidelity_check=True,
        imprint=True,
        contact_tolerance=None,
        overlaps="error",
        imprint_threads="all",
    ):
        """Mesh surfaces with a hybrid backend, mirroring cad-to-dagmc-mesher.

        Solids whose faces are all primitive OCC surfaces (plane / cylinder /
        cone / sphere / torus) are meshed with OCC ``BRepMesh`` -- its harmonized
        edge discretization keeps closed curved surfaces (sphere, torus)
        watertight, where the CDT struggles. Faces of solids with any b-spline /
        imported face go to the shared Rust scene mesher (the CDT), where
        BRepMesh is poor. Both feed one welded assembler so yamc and
        cad-to-dagmc use the same per-face mesher for matching geometry. See
        :mod:`yamc.cad._brepmesh` and :mod:`yamc.cad._scene_extract`.
        """
        from ._brepmesh import brepmesh_face_meshes
        from ._scene_extract import assemble_surface_mesh, scene_face_meshes

        if imprint and contact_tolerance is not None:
            raise ValueError(
                "contact_tolerance only applies with imprint=False: an imprinted "
                "mesh shares the surface between touching solids, so there is "
                "no contact to resolve"
            )
        if overlaps not in ("error", "smaller_wins", "ignore"):
            raise ValueError(
                f"overlaps must be 'error', 'smaller_wins' or 'ignore', got {overlaps!r}"
            )
        if imprint and overlaps == "smaller_wins":
            raise ValueError(
                "overlaps='smaller_wins' needs imprint=False: an imprinted mesh "
                "is tracked across the surfaces its solids share, which cannot "
                "give an overlap to one of them"
            )
        if contact_tolerance is not None:
            if isinstance(contact_tolerance, bool) or not isinstance(
                contact_tolerance, numbers.Real
            ):
                raise TypeError(
                    "contact_tolerance must be a number, one distance for the "
                    f"whole model, got {type(contact_tolerance).__name__}"
                )
            if not (contact_tolerance >= 0.0 and math.isfinite(contact_tolerance)):
                raise ValueError(
                    f"contact_tolerance must be a non-negative distance, got "
                    f"{contact_tolerance!r}"
                )
        tolerance = _check_per_solid(tolerance, "tolerance", self._material_tags)
        angular_tolerance = _check_per_solid(
            angular_tolerance, "angular_tolerance", self._material_tags
        )

        self._processed = process_assembly(
            self._assembly, self._material_tags, imprint=imprint,
            imprint_threads=imprint_threads,
        )
        pa = self._processed
        solid_tolerance, solid_angular_tolerance = _cap_thin_solids(
            pa,
            _per_solid(tolerance, "tolerance", _DEFAULT_TOLERANCE, pa.material_tags),
            _per_solid(
                angular_tolerance, "angular_tolerance", _DEFAULT_ANGULAR_TOLERANCE,
                pa.material_tags,
            ),
        )
        # A face bounding two solids is meshed at the finer of their values,
        # the deflection and the angle each on its own.
        face_tolerances = {
            fid: (
                min(solid_tolerance[sid] for sid in info.owning_solids),
                min(solid_angular_tolerance[sid] for sid in info.owning_solids),
            )
            for fid, info in pa.faces.items()
        }
        # Two facetings of one CAD surface each stay within the coarser of
        # their chordal deflections of it, so they sit at most twice that
        # apart. An overlap between two solids deeper than that is in the CAD,
        # unless the caller accepts slack up to contact_tolerance.
        overlap_limit = {}
        if contact_tolerance is None:
            overlap_limit = solid_tolerance
            contact_tolerance = 2.0 * max(
                solid_tolerance.values(), default=_DEFAULT_TOLERANCE
            )
        self._imprint = imprint
        self._contact_tolerance = None if imprint else float(contact_tolerance)
        self._overlaps = []
        self._volume_measures = None
        self._separated = []
        self._tet_data = {}
        self._tet_index_maps = {}

        brep_fids, scene_fids = _route_faces(pa)

        per_face = {}
        per_face.update(
            brepmesh_face_meshes(pa, brep_fids, face_tolerances, parallel=not imprint)
        )
        per_face.update(scene_face_meshes(pa, face_tolerances, scene_fids))

        (vertices, triangles, triangle_face_ids, triangle_surface_ids,
         face_to_surface_id) = assemble_surface_mesh(pa, per_face)

        self._surface_mesh = SurfaceMesh(
            vertices=vertices,
            triangles=triangles,
            triangle_face_ids=triangle_face_ids,
            triangle_surface_ids=triangle_surface_ids,
            face_to_surface_id=face_to_surface_id,
            solid_faces=pa.solid_faces,
            shared_faces=pa.shared_faces,
            material_tags=pa.material_tags,
            num_solids=len(pa.solid_faces),
            num_faces=len(pa.faces),
            face_solid_reversed=pa.face_solid_reversed,
        )

        if fidelity_check:
            from ._fidelity import (
                MeshFidelityError, check_mesh_fidelity, check_self_crossing,
            )

            problems = check_mesh_fidelity(self._surface_mesh, pa, solid_tolerance)
            problems += check_self_crossing(self._surface_mesh, solid_angular_tolerance)
            if problems:
                detail = "\n  ".join(problems)
                raise MeshFidelityError(
                    "Surface mesh deviates grossly from the CAD. "
                    "Transport on such a mesh runs without lost particles but "
                    "produces wrong physics. Problems:\n  " + detail + "\n"
                    "Pass fidelity_check=False to bypass this gate."
                )

        if overlaps != "ignore":
            from ._overlap import describe, find_overlaps, resolve_smaller_wins
            from .assembly_processor import OverlappingSolidsError

            if overlap_limit:
                # One pass at the smallest limit finds every pair deeper than
                # its own.
                found = [
                    pair
                    for pair in find_overlaps(
                        self._surface_mesh, 2.0 * min(overlap_limit.values())
                    )
                    if pair[2] > 2.0 * max(overlap_limit[pair[0]], overlap_limit[pair[1]])
                ]
                limit = "twice the larger deflection of the two solids"
            else:
                found = find_overlaps(self._surface_mesh, contact_tolerance)
                limit = f"the contact tolerance ({contact_tolerance:g})"
            if found and imprint:
                detail = "\n  ".join(describe(self._surface_mesh, found, overlap_limit))
                raise OverlappingSolidsError(
                    f"Solids overlap deeper than {limit}:\n  {detail}\n"
                    "The imprint shares the surfaces where solids coincide, but "
                    "does not split solids whose surfaces cross: both come out "
                    "whole and still overlap. Transport then gives the overlap "
                    "to whichever solid a particle was already in, which gives "
                    "wrong tallies and no lost particles. Cut one solid from the "
                    "other in the CAD (for instance b = b.cut(a)), or mesh with "
                    "imprint=False and overlaps='smaller_wins' to give each "
                    "overlap to the smaller solid of the pair. Pass "
                    "overlaps='ignore' to bypass this gate."
                )
            if found and overlaps == "error":
                detail = "\n  ".join(describe(self._surface_mesh, found, overlap_limit))
                raise OverlappingSolidsError(
                    f"Solids overlap deeper than {limit}:\n  {detail}\n"
                    "Without imprinting, an overlap belongs to whichever solid a "
                    "particle was already in. That is harmless for the slivers "
                    "two facetings of one surface leave between them, but a "
                    "solid sitting inside another is never entered by particles "
                    "arriving from outside it, which gives wrong tallies and no lost "
                    "particles. Cut one solid from the other in the CAD (for "
                    "instance b = b.cut(a)); or pass overlaps='smaller_wins' to "
                    "give each overlap to the smaller solid of the pair; or, if "
                    "the overlap is modelling slack you accept, pass a "
                    "contact_tolerance above its depth. Pass overlaps='ignore' "
                    "to bypass this gate."
                )
            if found:
                pairs, measures, lines = resolve_smaller_wins(
                    self._surface_mesh, found, contact_tolerance
                )
                self._overlaps = pairs
                self._volume_measures = measures
                warnings.warn(
                    "Resolving overlapping solids, smaller solid wins:\n  "
                    + "\n  ".join(lines),
                    stacklevel=3,
                )

        if not imprint:
            from ._contact import find_separated

            self._separated = find_separated(
                self._surface_mesh, pa.solid_shapes, contact_tolerance
            )

        return self._surface_mesh

    def _mesh_volumes(self, solids_to_tet, tet_target_edge_length=None):
        """Tetrahedralize each solid with the yamm tet mesher (mesh_volume).

        Each solid's surface triangles (from the scene surface mesh) are welded
        into a watertight boundary and passed to ``mesh_volume_rs``, which
        preserves the boundary exactly and fills the interior. The tet vertex
        block (boundary + new interior) is appended to the global vertex array
        and the connectivity reindexed into it. The surface mesh is already
        conformal with this boundary, so no surface replacement is needed.

        Blocks share vertices across an imprinted internal interface. Appending
        each block whole gives the interface two coincident-but-distinct copies
        of every vertex, and ``build_tet_adjacency`` pairs tet faces by their
        sorted vertex INDICES, so the tets either side never link: the exported
        mesh has a hole along every internal interface, and carries a second
        copy of the vertices there (10.8% of them on an eight-slab stack).

        This does not currently change an answer, and the merge is not a bug
        fix dressed as one. Transport tracks on the surface representation, and
        the tet mesh is a tally overlay walked one volume at a time --
        ``element_walk::walk_into`` stops at a tet belonging to another volume
        whether or not the two are linked. Measured either way, a two-cuboid
        model matches its CSG twin identically (z = 0.10). What the merge buys
        is a conformal mesh: a consumer that walks the tets without a per-volume
        guard, or any tool reading the Arrow file, sees a closed interface
        rather than a hole.

        The merge keys on the exact coordinate rather than a tolerance. Both
        blocks weld their boundary from the same surface triangles, so a shared
        interface arrives bit-identical on both sides and an exact key merges
        all of it; anything a tolerance would additionally catch is two
        genuinely distinct points, which is what a narrow gap between solids is
        made of.
        """
        from yamc._core import mesh_volume_rs

        mesh = self._surface_mesh
        # Coordinate -> index in mesh.vertices, over the tet blocks only. Not
        # seeded from the surface vertices already in mesh.vertices: surface
        # triangles and tets are separate entities in the export and a tet
        # vertex that happens to land on one carries no adjacency with it.
        tet_vertex_index: dict[tuple[float, float, float], int] = {}

        for solid_id in sorted(solids_to_tet):
            face_ids = set(mesh.solid_faces.get(solid_id, []))
            solid_tris = [
                tri for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids)
                if fid in face_ids
            ]
            if not solid_tris:
                continue

            boundary_verts, boundary_tris = _weld_solid_boundary(
                mesh.vertices, solid_tris
            )

            tel = tet_target_edge_length
            if tel is None:
                tel = _default_tet_edge_length(boundary_verts)

            interior_verts, tets = mesh_volume_rs(
                boundary_verts, boundary_tris, float(tel)
            )

            # tets index boundary_verts (0..N) then interior_verts (N..).
            tet_v = boundary_verts + [
                [float(p[0]), float(p[1]), float(p[2])] for p in interior_verts
            ]
            tet_t = [[int(t[0]), int(t[1]), int(t[2]), int(t[3])] for t in tets]

            index_map = []
            for p in tet_v:
                key = (p[0], p[1], p[2])
                global_idx = tet_vertex_index.get(key)
                if global_idx is None:
                    global_idx = len(mesh.vertices)
                    tet_vertex_index[key] = global_idx
                    mesh.vertices.append(p)
                index_map.append(global_idx)

            self._tet_index_maps[solid_id] = index_map
            self._tet_data[solid_id] = (tet_v, tet_t)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _tet_blocks(self):
        """Per-solid ``(solid_id, vertex_offset, tets)`` blocks for the Rust core.

        The connectivity is already reindexed into ``mesh.vertices``, so the
        offset is 0 and ``flatten_tet_blocks`` passes it through unchanged. A
        scalar offset per block cannot express the sharing: solids that meet at
        an imprinted interface index the same vertices there, which no single
        per-block base can produce.
        """
        blocks = []
        for solid_id, (_tet_v, tet_t) in self._tet_data.items():
            index_map = self._tet_index_maps.get(solid_id)
            if index_map is None:
                blocks.append((solid_id, 0, tet_t))
                continue
            blocks.append((
                solid_id,
                0,
                [[index_map[t[0]], index_map[t[1]], index_map[t[2]], index_map[t[3]]]
                 for t in tet_t],
            ))
        return blocks

    def to_arrow(
        self, path: str, boundary_tags: dict[str, Any] | None = None
    ) -> None:
        """Export the mesh to an Arrow IPC file.

        Args:
            path: Output file path (e.g. "model.arrow").
            boundary_tags: Optional dict of boundary condition name (e.g.
                ``"vacuum"``) to the surface ids it applies to, written as
                ``dim=2`` ``boundary:<name>`` physical groups. Surface ids are
                1-based, matching ``SurfaceMesh.triangle_surface_ids``; pass
                :meth:`exterior_surface_ids` for the model's outer shell.
        """
        if self._surface_mesh is None:
            raise RuntimeError("No surface mesh. Call mesh() first.")

        mesh = self._surface_mesh
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        tags = [
            (str(name), sorted(int(s) for s in surface_ids))
            for name, surface_ids in (boundary_tags or {}).items()
        ]

        # Physical-group assignment, tet offsetting, surface-to-volume topology
        # and the Arrow write all run in the Rust core in one call. A mesh made
        # without imprinting records that, and its contact tolerance.
        _cad_mesh_to_arrow(
            path=str(path),
            vertices=mesh.vertices,
            triangles=mesh.triangles,
            triangle_surface_ids=mesh.triangle_surface_ids,
            triangle_face_ids=mesh.triangle_face_ids,
            solid_faces=list(mesh.solid_faces.items()),
            material_tags=mesh.material_tags,
            face_to_surface_id=list(mesh.face_to_surface_id.items()),
            face_solid_reversed=mesh.face_solid_reversed,
            tet_blocks=self._tet_blocks(),
            boundary_tags=tags,
            imprinted=self._imprint,
            contact_tolerance=self._contact_tolerance,
            overlaps=self._overlaps,
            volume_measures=self._volume_measures,
            separated=self._separated,
        )

    def exterior_surface_ids(self) -> list[int]:
        """Surface ids on the model's outer shell.

        A surface is exterior when its CAD face is owned by a single solid, so
        the surfaces bounding two solids (imprinted internal interfaces) are
        excluded. These are the surfaces a standalone mesh model wants tagged
        with a boundary condition.

        Raises:
            RuntimeError: For a mesh made with ``imprint=False``. Every surface
                of such a mesh belongs to one solid, including the surfaces
                where solids touch, so tagging these vacuum would kill
                particles inside the model. Bound it with
                ``MeshGeometry(..., graveyard_offset=...)`` instead.
        """
        if self._surface_mesh is None:
            raise RuntimeError("No surface mesh. Call mesh() first.")
        if not self._imprint:
            raise RuntimeError(
                "exterior_surface_ids() is not defined for a mesh made with "
                "imprint=False: every surface there belongs to a single solid, "
                "including the surfaces where solids touch, so tagging them "
                "vacuum would kill particles inside the model. Bound the model "
                "with yamc.MeshGeometry(..., graveyard_offset=...) instead."
            )

        mesh = self._surface_mesh
        return sorted(
            sid
            for fid, sid in mesh.face_to_surface_id.items()
            if len(mesh.shared_faces.get(fid, ())) < 2
        )

    def to_vtkhdf(self, path: str) -> None:
        """Export mesh to VTKHDF v2.1 UnstructuredGrid for ParaView.

        Args:
            path: Output file path (e.g. "model.vtkhdf").
        """
        if self._surface_mesh is None:
            raise RuntimeError("No surface mesh. Call mesh() first.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        from .vtkhdf_writer import (
            surface_to_vtkhdf,
            mixed_to_vtkhdf,
        )

        mesh = self._surface_mesh

        # Per-triangle volume/material labels and tet offsetting run in the
        # Rust core; Python keeps only the h5py write.
        volume_ids, material_ids, all_tets, tet_volume_ids = _cad_mesh_labels(
            mesh.triangle_face_ids,
            list(mesh.solid_faces.items()),
            self._tet_blocks(),
        )

        if self._tet_data:
            # Mixed mesh: tris + tets.
            # Tet vertices are already in mesh.vertices (conformal).
            mixed_to_vtkhdf(
                str(path),
                mesh.vertices,
                mesh.triangles,
                all_tets,
                cell_data={"volume_id": volume_ids + tet_volume_ids},
            )
        else:
            # Surface-only mesh
            surface_to_vtkhdf(
                str(path),
                mesh.vertices,
                mesh.triangles,
                cell_data={
                    "volume_id": volume_ids,
                    "material_id": material_ids,
                },
            )
