"""CAD meshing pipeline -- requires cadquery."""

from __future__ import annotations

# Rust CDT functions (always available, no cadquery needed)
from yamc._core import (
    cad_mesh_labels,
    cad_mesh_to_arrow,
    mesh_face,
    mesh_faces,
    mesh_to_arrow,
    mesh_faces_scene_resolved,
    mesh_volume_rs,
    solid_overlaps,
)

# Meshing CAD is MeshGeometry.from_cadquery; the mesher behind it
# (mesher._CadMesher) is internal.
from .assembly_processor import OverlappingSolidsError, ProcessedAssembly

__all__ = [
    "cad_mesh_labels", "cad_mesh_to_arrow",
    "mesh_face", "mesh_faces", "mesh_to_arrow",
    "mesh_faces_scene_resolved", "mesh_volume_rs", "solid_overlaps",
    "ProcessedAssembly", "OverlappingSolidsError",
]
