"""The VTKHDF UnstructuredGrid writers, kept here for the CAD pipeline.

They live in :mod:`yamc._vtkhdf_writer`, outside :mod:`yamc.cad`, so
``MeshGeometry.to_vtkhdf`` and ``UnstructuredMesh.to_vtkhdf`` work without cadquery
(importing anything under ``yamc.cad`` imports it).
"""

from yamc._vtkhdf_writer import (  # noqa: F401
    VTK_TETRA,
    VTK_TRIANGLE,
    mixed_to_vtkhdf,
    surface_to_vtkhdf,
    tet_to_vtkhdf,
)
