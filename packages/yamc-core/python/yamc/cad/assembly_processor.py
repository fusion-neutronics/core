"""Assembly processing: imprinting and topology extraction.

Takes a CadQuery assembly, imprints it, and extracts the face/solid topology
needed for meshing: which faces belong to which solids, shared faces, and
face orientation per solid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cadquery as cq
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS


@dataclass
class FaceInfo:
    """Information about a single BRep face."""
    face_id: int
    is_planar: bool
    surface_area: float
    owning_solids: list = field(default_factory=list)  # solid IDs this face belongs to


@dataclass
class ProcessedAssembly:
    """Result of processing a CadQuery assembly."""
    faces: dict  # face_id -> FaceInfo
    face_to_occ: dict  # face_id -> OCC TopoDS_Face
    material_tags: list  # ordered material tags matching solid order
    solid_faces: dict  # solid_id -> list of face_ids
    shared_faces: dict  # face_id -> [solid_id, solid_id] for faces shared between 2 volumes
    face_solid_reversed: dict = field(default_factory=dict)
    # (solid_id, face_id) -> bool: True when face is REVERSED in that solid
    # (i.e. the UV surface normal points inward to the solid)
    imprinted_compound: object = None  # OCC compound (source of the global edge map)
    solid_shapes: dict = field(default_factory=dict)  # solid_id -> CadQuery Shape


def _imprint_assembly(assembly):
    """Imprint a CadQuery assembly into a connected compound.

    Uses the BOPAlgo_Builder based imprint with glue="partial" when the
    installed cadquery supports it (CadQuery/cadquery#2069, faster and
    lower RAM than BOPAlgo_MakeConnected, same result for touching,
    non-overlapping solids). Older cadquery versions fall back to the
    original single-argument imprint.

    Single-solid assemblies skip imprinting entirely: it is a no-op for
    one solid, and the BOPAlgo_Builder imprint returns a Null shape when
    given fewer than two arguments.
    """
    import inspect

    id_map = {}
    for obj, name, loc, _ in assembly:
        for solid in obj.moved(loc).Solids():
            id_map[solid] = name
    if len(id_map) < 2:
        solids = list(id_map)
        compound = cq.occ_impl.shapes.Compound.makeCompound(solids)
        return compound, {s: (id_map[s],) for s in solids}

    imprint = cq.occ_impl.assembly.imprint
    if "glue" in inspect.signature(imprint).parameters:
        return imprint(assembly, glue="partial")
    return imprint(assembly)


def process_assembly(
    assembly: cq.Assembly, material_tags: list[str]
) -> ProcessedAssembly:
    """Process a CadQuery assembly for meshing.

    Imprints the assembly and extracts face/solid topology (shared faces,
    orientation per solid).  Surface meshing is done separately by the yamm
    scene mesher (see :mod:`yamc.cad._scene_extract`).

    Args:
        assembly: CadQuery Assembly with named parts.
        material_tags: List of material tag strings, one per part.

    Returns:
        ProcessedAssembly with topology data and imprinted compound.
    """
    # Step 1: Imprint the assembly
    imprinted_compound, imprinted_solids_with_ids = _imprint_assembly(assembly)

    # Get the OCC compound shape
    if hasattr(imprinted_compound, 'wrapped'):
        occ_compound = imprinted_compound.wrapped
    else:
        occ_compound = imprinted_compound

    # Step 2: Extract solids and reorder material tags
    # imprint() returns (Shape, Dict[Shape, Tuple[str, ...]]) where the dict
    # maps each imprinted solid to the original part name(s) it came from.
    solids = []
    ordered_tags = []

    # Build name -> material tag lookup from the assembly
    name_to_tag = {}
    for idx, (_, name, _, _) in enumerate(assembly):
        if idx < len(material_tags):
            name_to_tag[name] = material_tags[idx]

    # imprinted_solids_with_ids is a dict: Shape -> Tuple[str, ...]
    for imprinted_solid, origin_names in imprinted_solids_with_ids.items():
        solids.append(imprinted_solid)
        # Use the first origin name to find the material tag
        if origin_names and origin_names[0] in name_to_tag:
            ordered_tags.append(name_to_tag[origin_names[0]])
        else:
            ordered_tags.append(f"material_{len(ordered_tags)}")

    # Step 3: Extract faces from each solid -- track ownership and orientation
    faces = {}
    face_to_occ = {}
    solid_faces = {}
    solid_shapes = {}  # solid_id -> CadQuery Shape (for tet meshing)
    face_id_counter = 1

    seen_faces = {}  # hash -> face_id
    face_solid_reversed = {}  # (solid_id, face_id) -> bool

    for solid_idx, solid in enumerate(solids):
        solid_id = solid_idx + 1
        solid_face_ids = []

        if hasattr(solid, 'wrapped'):
            shape = solid.wrapped
        else:
            shape = solid

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            occ_face = TopoDS.Face(face_explorer.Current())
            face_hash = hash(occ_face)

            # Check if we've seen this face before (shared face)
            if face_hash in seen_faces:
                fid = seen_faces[face_hash]
                faces[fid].owning_solids.append(solid_id)
                face_solid_reversed[(solid_id, fid)] = (
                    occ_face.Orientation() == TopAbs_REVERSED
                )
                solid_face_ids.append(fid)
                face_explorer.Next()
                continue

            fid = face_id_counter
            face_id_counter += 1
            seen_faces[face_hash] = fid

            # Get face properties
            surface = BRepAdaptor_Surface(occ_face)
            is_planar = surface.GetType().value == 0  # GeomAbs_Plane = 0

            # Compute surface area (approximate)
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(occ_face, props)
            area = props.Mass()

            face_solid_reversed[(solid_id, fid)] = (
                occ_face.Orientation() == TopAbs_REVERSED
            )

            face_info = FaceInfo(
                face_id=fid,
                is_planar=is_planar,
                surface_area=area,
                owning_solids=[solid_id],
            )

            faces[fid] = face_info
            face_to_occ[fid] = occ_face
            solid_face_ids.append(fid)

            face_explorer.Next()

        solid_faces[solid_id] = solid_face_ids
        solid_shapes[solid_id] = solid

    # Step 4: Identify shared faces
    shared = {}
    for fid, info in faces.items():
        if len(info.owning_solids) >= 2:
            shared[fid] = info.owning_solids[:2]

    return ProcessedAssembly(
        faces=faces,
        face_to_occ=face_to_occ,
        material_tags=ordered_tags,
        solid_faces=solid_faces,
        shared_faces=shared,
        face_solid_reversed=face_solid_reversed,
        imprinted_compound=occ_compound,
        solid_shapes=solid_shapes,
    )
