"""Assembly processing: imprinting and topology extraction.

Takes a CadQuery assembly, imprints it (or, with ``imprint=False``, keeps every
solid as it is), and extracts the face/solid topology needed for meshing: which
faces belong to which solids, shared faces, and face orientation per solid.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

import cadquery as cq
from cadquery.func import setThreads
from OCP.OSD import OSD_Parallel, OSD_ThreadPool
from OCP.TopAbs import (
    TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_INTERNAL, TopAbs_REVERSED,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape,
)
from OCP.BRep import BRep_Builder
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf
from OCP.TopoDS import TopoDS, TopoDS_Iterator

from ._fidelity import _occ_signed_volume


_IMPRINT_THREAD_LOCK = RLock()


@contextmanager
def occ_threads(threads: int):
    """Run the OCC work inside on at most *threads* threads of the default
    pool, restoring the pool afterwards. Serialised, since the pool is global."""
    with _IMPRINT_THREAD_LOCK:
        pool = OSD_ThreadPool.DefaultPool_s()
        previous_threads = pool.NbThreads()
        previous_launch_threads = pool.NbDefaultThreadsToLaunch()
        try:
            setThreads(threads)
            pool.SetNbDefaultThreadsToLaunch(threads)
            yield
        finally:
            setThreads(previous_threads)
            pool.SetNbDefaultThreadsToLaunch(previous_launch_threads)


def release_freed_memory():
    """Hand memory the process has freed back to the operating system.

    glibc keeps freed heap memory for reuse, so after a large OCC operation the
    process still holds its peak: a 64-solid imprint left 9.5GB resident that
    nothing referenced, and every later stage of meshing ran on top of it.
    ``malloc_trim`` returns it (0.6GB left). Linux with glibc only; elsewhere
    this does nothing.
    """
    if not sys.platform.startswith("linux"):
        return
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)


class OverlappingSolidsError(ValueError):
    """Two input solids claim the same region of space."""


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
    # solid_id -> signed CAD volume, for the solids the imprint check integrated
    solid_volumes: dict = field(default_factory=dict)


def _assembly_solids(assembly):
    """Every solid of the assembly, placed, mapped to the name of its part."""
    id_map = {}
    for obj, name, loc, _ in assembly:
        for solid in obj.moved(loc).Solids():
            id_map[solid] = name
    return id_map


def _compound_without_imprint(assembly):
    """The assembly's solids gathered into one compound, untouched.

    Returns the same ``(compound, {solid: (name,)})`` pair as the imprint, so
    the rest of the pipeline does not care which one ran. No face is shared
    between solids: two solids that touch keep a face each where they meet.
    """
    id_map = _assembly_solids(assembly)
    solids = list(id_map)
    compound = cq.occ_impl.shapes.Compound.makeCompound(solids)
    return compound, {s: (id_map[s],) for s in solids}


def _is_mirrored(solid):
    """Whether *solid* is placed by a mirroring (negative determinant) location.

    ``gp_Trsf.IsNegative`` is not enough: it reads only the sign of the scale
    factor, and OCC can keep a reflection in the matrix instead, for instance
    for a placement in a mirrored frame.
    """
    trsf = solid.wrapped.Location().Transformation()
    return trsf.VectorialPart().Determinant() < 0.0


def _unmirrored(solid):
    """*solid* with its mirroring placement copied into its geometry."""
    placed = solid.wrapped.Location().Transformation()
    # The copy turns the faces round only when the scale factor is negative.
    # Rebuilding the transformation from its coefficients moves a reflection
    # held in the matrix into the scale factor's sign.
    trsf = gp_Trsf()
    trsf.SetValues(*(placed.Value(r, c) for r in (1, 2, 3) for c in (1, 2, 3, 4)))
    bare = solid.wrapped.Located(TopLoc_Location())
    baked = BRepBuilderAPI_Transform(bare, trsf, True).Shape()
    if _occ_signed_volume(baked) < 0.0:
        # It was inside out under its mirror too: a reversed solid.
        baked = baked.Reversed()
    return cq.Solid(baked)


def _bake_mirrored_parts(assembly):
    """The assembly's parts as iterating it gives them, with every mirrored
    solid turned into an ordinary one in the same place.

    A solid placed by a mirroring location, the usual way to place the left
    and right copies of a symmetric part, is inside out to OCC: its face
    normals point into it and its volume integrates negative. Meshed as it
    is, its facets point into it, and transport through it is wrong without
    losing a particle. Imprinting it goes wrong as well:
    BOPAlgo_MakeConnected returns extra solids and names every unmodified
    output after the mirrored part. Copying the geometry through the mirror
    gives an ordinary solid that OCC and the mesher both read the right way
    round. Parts without a mirrored solid pass through untouched.

    Returns:
        List of ``(shape, name, location, color)``, one per part, in the
        assembly's order.
    """
    parts = []
    for obj, name, loc, color in assembly:
        solids = obj.moved(loc).Solids()
        if not any(_is_mirrored(s) for s in solids):
            parts.append((obj, name, loc, color))
            continue
        baked = [_unmirrored(s) if _is_mirrored(s) else s for s in solids]
        parts.append((cq.Compound.makeCompound(baked), name, cq.Location(), color))
    return parts


def resolve_threads(threads: int | Literal["all"], name: str = "threads") -> int:
    """A thread count from ``"all"`` (every logical processor) or a positive
    integer; *name* is the argument named in the error otherwise."""
    if isinstance(threads, str) and threads == "all":
        return max(1, OSD_Parallel.NbLogicalProcessors_s())
    if type(threads) is int and threads > 0:
        return threads
    raise ValueError(f"{name} must be 'all' or a positive integer, got {threads!r}")


def _resolve_imprint_threads(imprint_threads: int | Literal["all"]) -> int:
    return resolve_threads(imprint_threads, "imprint_threads")


def _imprint_assembly(assembly, imprint_threads: int | Literal["all"] = "all"):
    """Imprint a CadQuery assembly into a connected compound.

    Uses the BOPAlgo_Builder based imprint with glue="partial" when the
    installed cadquery supports it (faster and
    lower RAM than BOPAlgo_MakeConnected, same result for touching,
    non-overlapping solids). Older cadquery versions fall back to the
    original single-argument imprint.

    Single-solid assemblies skip imprinting entirely: it is a no-op for
    one solid, and the BOPAlgo_Builder imprint returns a Null shape when
    given fewer than two arguments.
    """
    import inspect

    threads = _resolve_imprint_threads(imprint_threads)
    if len(_assembly_solids(assembly)) < 2:
        return _compound_without_imprint(assembly)

    imprint = cq.occ_impl.assembly.imprint
    with occ_threads(threads):
        if "glue" in inspect.signature(imprint).parameters:
            compound, solids = imprint(assembly, glue="partial")
        else:
            compound, solids = imprint(assembly)
    release_freed_memory()
    return _strip_free_internal_edges(compound, solids)


def _sub_shapes(stored, placed):
    """Yield each sub-shape of a shape twice over: as stored in it (location
    and orientation relative to it, for rebuilding it) and as placed in the
    model (for looking it up in maps of the whole model). *stored* and
    *placed* are the same shape, *placed* with its location and orientation
    in the model.
    """
    relative = TopoDS_Iterator(stored, False, False)
    absolute = TopoDS_Iterator(placed)
    while relative.More():
        yield relative.Value(), absolute.Value()
        relative.Next()
        absolute.Next()


def _strip_free_internal_edges(compound, solids_with_ids):
    """Drop INTERNAL edges that no other face uses from the imprinted faces.

    Where two solids share a spherical face whose poles do not line up (the
    same sphere, one copy rotated), the imprint keeps one shared face but
    leaves the other copy's seam on it as INTERNAL edges, a half meridian
    between poles that are not the face's own, which no other face uses.
    BRepMesh's sphere mesher then places interior nodes on only one side of
    that open wire and spans the rest of the face with long chords through
    the sphere, losing up to all of the inner solid's volume. Such an edge
    bounds nothing, so dropping it changes neither the geometry nor which
    faces the solids share. INTERNAL edges that other faces use too (a line
    contact between solids) are kept.

    The rebuilt faces replace the old ones in the compound and in every
    imprinted solid alike, so the solids keep sharing them.

    Returns:
        ``(compound, {solid: origin names})`` as the imprint does; the inputs
        themselves when no face has such an edge.
    """
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndUniqueAncestors_s(
        compound.wrapped, TopAbs_EDGE, TopAbs_FACE, edge_faces
    )
    # An edge used by a single face is keyed with its orientation in it.
    faces = TopTools_IndexedMapOfShape()  # faces with a free INTERNAL edge
    drops = {}  # index in faces -> its free INTERNAL edges
    for i in range(1, edge_faces.Extent() + 1):
        edge = edge_faces.FindKey(i)
        owners = edge_faces.FindFromIndex(i)
        if owners.Size() == 1 and edge.Orientation() == TopAbs_INTERNAL:
            drops.setdefault(faces.Add(owners.First()), []).append(edge)
    if not drops:
        return compound, solids_with_ids

    builder = BRep_Builder()
    reshape = ShapeBuild_ReShape()
    for index, drop in drops.items():
        face = faces.FindKey(index).Oriented(TopAbs_FORWARD)
        new_face = face.EmptyCopied()
        for wire, placed_wire in _sub_shapes(face, face):
            if wire.ShapeType() != TopAbs_WIRE:
                builder.Add(new_face, wire)
                continue
            new_wire = wire.EmptyCopied()
            kept = 0
            for edge, placed_edge in _sub_shapes(wire, placed_wire):
                if placed_edge.Orientation() == TopAbs_INTERNAL and any(
                    placed_edge.IsSame(d) for d in drop
                ):
                    continue
                builder.Add(new_wire, edge)
                kept += 1
            if kept:
                new_wire.Closed(wire.Closed())
                builder.Add(new_face, new_wire)
        reshape.Replace(face, new_face)

    def rebuilt(shape):
        return cq.Shape.cast(reshape.Apply(shape.wrapped))

    return rebuilt(compound), {
        rebuilt(solid): origins for solid, origins in solids_with_ids.items()
    }


def _bounding_faces(shape):
    """The faces that bound the solid *shape*, each once, as the solid holds it.

    A face the solid lies on both sides of bounds nothing: one OCC keeps in
    the solid as INTERNAL (``Shape.split`` with a face that stops short of the
    solid's boundary leaves one, and so does a general fuse with a face), or
    one the solid holds both FORWARD and REVERSED. Nor does an EXTERNAL face,
    with the solid on neither side. Meshed as a boundary, such a face opened
    the solid onto the implicit complement, and transport lost the particles
    that crossed it. OCC leaves these faces out of the solid's volume, and
    cad-to-dagmc leaves an INTERNAL face out of its mesh.
    """
    held = {}  # hash -> (face, the orientations the solid holds it in)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        held.setdefault(hash(face), (face, set()))[1].add(face.Orientation())
        explorer.Next()
    return [
        face for face, orientations in held.values()
        if orientations in ({TopAbs_FORWARD}, {TopAbs_REVERSED})
    ]


def _refuse_lost_or_shared_inputs(assembly, imprinted_solids_with_ids):
    """Raise unless every input solid comes out of the imprint as its own.

    Two ways an overlap between inputs goes wrong, both silently: an imprinted
    solid with more than one input as its origin (tagging it with the first
    origin's material removes it from the others), and an input that comes
    out smaller than it went in (a part sitting inside a body that was never
    cut comes back as a piece of the body, so its material vanishes).
    Either way transport loses no particles to show it. cad-to-dagmc-mesher
    refuses the same things.

    Origins are assembly names, so the second is checked by volume: the
    imprinted solids an input name is an origin of must hold all the volume
    its solids had, a piece counted once per time the name is among its
    origins (two overlapping solids under one name share a piece that lists
    the name twice). That also catches one of several solids under a name
    being absorbed while the others survive. Volumes are integrated the way
    the fidelity gate integrates them: OCC's default and adaptive integrations
    can be off by tens of percent or more on dense spline solids (the default
    one even in sign), and differently once the imprint splits their faces.
    Assembly nodes without solids (a face, a wire) are not inputs.

    Overlaps whose surfaces cross without coinciding are not split by the
    imprint at all: both solids come back whole, each with its own origin,
    and are not caught here.

    Returns:
        ``{imprinted solid: its signed volume}``, for the fidelity gate. A
        solid the imprint left as it was is integrated once.
    """
    def short(name):
        return name.rsplit("/", 1)[-1]

    volumes = {}  # (solid, orientation) -> signed volume

    def volume_of(solid):
        key = (solid, solid.wrapped.Orientation())
        if key not in volumes:
            volumes[key] = _occ_signed_volume(solid)
        # A reversed solid integrates negative.
        return abs(volumes[key])

    volume_in = {}
    for obj, name, loc, _ in assembly:
        for solid in obj.moved(loc).Solids():
            volume_in[name] = volume_in.get(name, 0.0) + volume_of(solid)
    volume_out = {}
    shared = []
    for solid, origins in imprinted_solids_with_ids.items():
        volume = volume_of(solid)
        for name in origins:
            volume_out[name] = volume_out.get(name, 0.0) + volume
        names = list(dict.fromkeys(origins))
        if len(names) > 1:
            shared.append((volume, [short(n) for n in names]))
    lost = []
    for name, v_in in volume_in.items():
        v_out = volume_out.get(name, 0.0)
        if v_out <= 0.0:
            lost.append(f"{short(name)!r} was absorbed into another solid")
        elif v_out < v_in * (1.0 - 1e-4):
            lost.append(
                f"{short(name)!r} came out with {v_out:.6g} of its {v_in:.6g} volume "
                "(part of it was absorbed into another solid)"
            )
    if not lost and not shared:
        return {
            solid: volumes[(solid, solid.wrapped.Orientation())]
            for solid in imprinted_solids_with_ids
        }
    lines = lost + [
        f"{' and '.join(repr(n) for n in names)} came out as one region of "
        f"volume {volume:.6g}"
        for volume, names in shared
    ]
    raise OverlappingSolidsError(
        "The imprint did not give every input solid a region of its own:\n  "
        + "\n  ".join(lines)
        + "\nSuch a region gets only one of its solids' materials, so the other "
        "solid comes out smaller than the CAD says or disappears altogether, "
        "and transport loses no particles to show it. Usually the solids "
        "overlap in the CAD: cut one from the other (for instance "
        "b = b.cut(a)) so every point belongs to exactly one solid. If they "
        "only touch, the imprint itself has failed on them."
    )


def process_assembly(
    assembly: cq.Assembly,
    material_tags: list[str],
    imprint: bool = True,
    *,
    imprint_threads: int | Literal["all"] = "all",
) -> ProcessedAssembly:
    """Process a CadQuery assembly for meshing.

    Imprints the assembly and extracts face/solid topology (shared faces,
    orientation per solid).  Surface meshing is done separately by the yamm
    scene mesher (see :mod:`yamc.cad._scene_extract`).

    Args:
        assembly: CadQuery Assembly with named parts.
        material_tags: List of material tag strings, one per part.
        imprint: Imprint the assembly so touching solids share the face where
            they meet. ``False`` skips the imprint (the expensive step on a
            large assembly) and keeps every solid whole, so each solid is
            meshed on its own.
        imprint_threads: ``"all"`` uses all logical processors, or a positive
            integer limits only imprinting. The prior OCC pool size and default
            launch count are restored afterward, including on failure. The
            value is validated even when no imprint operation is needed.

    Returns:
        ProcessedAssembly with topology data and imprinted compound.
    """
    threads = _resolve_imprint_threads(imprint_threads)
    # Everything below only iterates the parts, and no mirrored solid may
    # reach any of it.
    assembly = _bake_mirrored_parts(assembly)

    # Step 1: Imprint the assembly (or gather its solids as they are)
    volumes = {}
    if imprint:
        imprinted_compound, imprinted_solids_with_ids = _imprint_assembly(
            assembly, imprint_threads=threads
        )
        volumes = _refuse_lost_or_shared_inputs(assembly, imprinted_solids_with_ids)
    else:
        imprinted_compound, imprinted_solids_with_ids = _compound_without_imprint(
            assembly
        )

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

    # Step 3: Extract the faces bounding each solid -- track ownership and
    # orientation
    faces = {}
    face_to_occ = {}
    solid_faces = {}
    solid_shapes = {}  # solid_id -> CadQuery Shape (for tet meshing)
    solid_volumes = {}
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

        for occ_face in _bounding_faces(shape):
            face_hash = hash(occ_face)

            # Check if we've seen this face before (shared face)
            if face_hash in seen_faces:
                fid = seen_faces[face_hash]
                faces[fid].owning_solids.append(solid_id)
                face_solid_reversed[(solid_id, fid)] = (
                    occ_face.Orientation() == TopAbs_REVERSED
                )
                solid_face_ids.append(fid)
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

        solid_faces[solid_id] = solid_face_ids
        solid_shapes[solid_id] = solid
        if solid in volumes:
            solid_volumes[solid_id] = volumes[solid]

    # Step 4: Identify shared faces. A face bounds at most two solids; a third
    # is an imprint left in a broken state (a zero-thickness sliver split off
    # one solid, say), which the mesh cannot represent.
    shared = {}
    for fid, info in faces.items():
        owners = list(dict.fromkeys(info.owning_solids))
        if len(owners) > 2:
            names = [repr(ordered_tags[s - 1]) for s in owners]
            raise OverlappingSolidsError(
                f"The imprint left a face bounding {len(owners)} solids, made "
                f"from {', '.join(names[:-1])} and {names[-1]}; a face can bound "
                "two. The imprint has failed on these solids: it can split a "
                "zero-thickness sliver off one where curved faces touch (a name "
                "listed twice is two pieces of one input)."
            )
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
        solid_volumes=solid_volumes,
    )
