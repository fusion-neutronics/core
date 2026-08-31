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
    weld_mesh,
)

# High-level API (cadquery imported lazily at class instantiation)
from .mesher import CadToYamc, SurfaceMesh
from .assembly_processor import ProcessedAssembly

__all__ = [
    "cad_mesh_labels", "cad_mesh_to_arrow",
    "mesh_face", "mesh_faces", "mesh_to_arrow",
    "mesh_faces_scene_resolved", "mesh_volume_rs", "weld_mesh",
    "CadToYamc", "SurfaceMesh", "ProcessedAssembly",
]
