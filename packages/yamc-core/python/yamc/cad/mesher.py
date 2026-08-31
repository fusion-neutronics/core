"""Main CadToYamc class -- end-to-end surface meshing pipeline.

Takes a CadQuery assembly, imprints it, meshes each face with the
``cad-to-dagmc-mesher`` scene mesher (see :mod:`yamc.cad._scene_extract`), and
produces a global vertex + triangle mesh with per-face and per-volume metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import os

from .assembly_processor import process_assembly
from . import cad_mesh_labels as _cad_mesh_labels
from . import cad_mesh_to_arrow as _cad_mesh_to_arrow
from . import weld_mesh as _weld_mesh

if TYPE_CHECKING:
    import cadquery


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
    sphere / torus) -- or the face is a closed doubly-periodic surface (see
    ``_is_closed_doubly_periodic``). Faces of solids with any b-spline /
    imported surface otherwise go to the scene CDT. Returns
    ``(brep_face_ids, scene_face_ids)``.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_Plane,
        GeomAbs_Cylinder,
        GeomAbs_Cone,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )

    primitive_types = {
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    }
    solid_all_primitive = {
        sid: all(
            BRepAdaptor_Surface(pa.face_to_occ[fid]).GetType() in primitive_types
            for fid in fids
        )
        for sid, fids in pa.solid_faces.items()
    }

    brep_fids, scene_fids = [], []
    for fid, info in pa.faces.items():
        owners = info.owning_solids
        if owners and all(solid_all_primitive.get(s, False) for s in owners):
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
        tolerance: float = 0.1,
        angular_tolerance: float = 0.1,
        fidelity_check: bool = True,
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
                curved surfaces.  Default 0.1.
            angular_tolerance: Angular deflection tolerance for surface
                meshing (radians).  Default 0.1 (~5.7 degrees).

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
        )

        if tet_volumes:
            solids_to_tet = set()
            for i, tag in enumerate(self._material_tags):
                if tag in tet_volumes:
                    solids_to_tet.add(i + 1)  # solid IDs are 1-indexed

            self._mesh_volumes(
                solids_to_tet,
                tet_target_edge_length=target_edge_length,
            )

        return self._surface_mesh

    # ------------------------------------------------------------------
    # Lower-level methods
    # ------------------------------------------------------------------

    def mesh_surfaces(
        self,
        tolerance: float = 0.1,
        angular_tolerance: float = 0.1,
        fidelity_check: bool = True,
    ) -> SurfaceMesh:
        """Mesh all surfaces of the assembly.

        Args:
            tolerance: Chordal deflection tolerance for surface meshing
                (3D units).
            angular_tolerance: Angular deflection tolerance (radians).

        Returns:
            SurfaceMesh with global vertex/triangle arrays.
        """
        if self._assembly is None:
            raise RuntimeError("No assembly added. Call add_cadquery_object first.")

        return self._mesh_surfaces(
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
            fidelity_check=fidelity_check,
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

    def _mesh_surfaces(self, tolerance=0.1, angular_tolerance=0.1, fidelity_check=True):
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

        self._processed = process_assembly(self._assembly, self._material_tags)
        pa = self._processed

        brep_fids, scene_fids = _route_faces(pa)

        per_face = {}
        per_face.update(
            brepmesh_face_meshes(pa, brep_fids, tolerance, angular_tolerance)
        )
        per_face.update(
            scene_face_meshes(pa, tolerance, angular_tolerance, scene_fids)
        )

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
            from ._fidelity import MeshFidelityError, check_mesh_fidelity

            problems = check_mesh_fidelity(self._surface_mesh, pa, tolerance)
            if problems:
                detail = "\n  ".join(problems)
                raise MeshFidelityError(
                    "Surface mesh deviates grossly from the CAD (issue #252). "
                    "Transport on such a mesh runs without lost particles but "
                    "produces wrong physics. Problems:\n  " + detail + "\n"
                    "Pass fidelity_check=False to bypass this gate."
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
        # and the Arrow write all run in the Rust core in one call (issue #246).
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
        )

    def exterior_surface_ids(self) -> list[int]:
        """Surface ids on the model's outer shell.

        A surface is exterior when its CAD face is owned by a single solid, so
        the surfaces bounding two solids (imprinted internal interfaces) are
        excluded. These are the surfaces a standalone mesh model wants tagged
        with a boundary condition.
        """
        if self._surface_mesh is None:
            raise RuntimeError("No surface mesh. Call mesh() first.")

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
        # Rust core (issue #246); Python keeps only the h5py write.
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
