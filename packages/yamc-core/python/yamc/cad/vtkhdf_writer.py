"""VTKHDF v2.1 UnstructuredGrid writer for ParaView visualization."""

import numpy as np

VTK_TRIANGLE = 5
VTK_TETRA = 10


def _write_vtkhdf_header(root, n_points, n_cells, n_connectivity_ids):
    """Write the VTKHDF header attributes and metadata datasets."""
    root.attrs["Version"] = np.array([2, 1], dtype="int64")
    ascii_type = "UnstructuredGrid".encode("ascii")
    import h5py
    root.attrs.create(
        "Type",
        ascii_type,
        dtype=h5py.string_dtype("ascii", len(ascii_type)),
    )
    root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype="int64"))
    root.create_dataset("NumberOfCells", data=np.array([n_cells], dtype="int64"))
    root.create_dataset(
        "NumberOfConnectivityIds",
        data=np.array([n_connectivity_ids], dtype="int64"),
    )


def surface_to_vtkhdf(filename, vertices, triangles, cell_data=None):
    """Write surface triangle mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: List of [x, y, z] or numpy array (N, 3).
        triangles: List of [i, j, k] or numpy array (M, 3).
        cell_data: Optional dict of name -> array (one value per triangle).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    verts = np.array(vertices, dtype="float64")
    tris = np.array(triangles, dtype="int64")
    n_tris = len(tris)
    n_points = len(verts)

    connectivity = tris.flatten()
    offsets = np.arange(0, n_tris * 3 + 1, 3, dtype="int64")
    types = np.full(n_tris, VTK_TRIANGLE, dtype="uint8")

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_tris, len(connectivity))
        root.create_dataset("Points", data=verts)
        root.create_dataset("Connectivity", data=connectivity)
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Types", data=types)

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=np.array(data))


def tet_to_vtkhdf(filename, vertices, tets, cell_data=None):
    """Write tetrahedral mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: numpy array (N, 3) float64.
        tets: numpy array (M, 4) int.
        cell_data: Optional dict of name -> array (one value per tet).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    verts = np.array(vertices, dtype="float64")
    tets_arr = np.array(tets, dtype="int64")
    n_tets = len(tets_arr)
    n_points = len(verts)

    connectivity = tets_arr.flatten()
    offsets = np.arange(0, n_tets * 4 + 1, 4, dtype="int64")
    types = np.full(n_tets, VTK_TETRA, dtype="uint8")

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_tets, len(connectivity))
        root.create_dataset("Points", data=verts)
        root.create_dataset("Connectivity", data=connectivity)
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Types", data=types)

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=np.array(data))


def mixed_to_vtkhdf(filename, vertices, triangles, tets, cell_data=None):
    """Write mixed triangle + tetrahedron mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: numpy array (N, 3) float64.
        triangles: numpy array (M_tri, 3) int.
        tets: numpy array (M_tet, 4) int.
        cell_data: Optional dict of name -> array (one value per cell,
            triangles first then tets).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    verts = np.array(vertices, dtype="float64")
    tris = np.array(triangles, dtype="int64") if len(triangles) > 0 else np.empty((0, 3), dtype="int64")
    tets_arr = np.array(tets, dtype="int64") if len(tets) > 0 else np.empty((0, 4), dtype="int64")

    n_tris = len(tris)
    n_tets = len(tets_arr)
    n_cells = n_tris + n_tets
    n_points = len(verts)

    # Build connectivity, offsets, types for mixed mesh
    parts = []
    if n_tris > 0:
        parts.append(tris.flatten())
    if n_tets > 0:
        parts.append(tets_arr.flatten())
    connectivity = np.concatenate(parts) if parts else np.empty(0, dtype="int64")

    offsets = [0]
    for _ in range(n_tris):
        offsets.append(offsets[-1] + 3)
    for _ in range(n_tets):
        offsets.append(offsets[-1] + 4)
    offsets = np.array(offsets, dtype="int64")

    types = np.concatenate([
        np.full(n_tris, VTK_TRIANGLE, dtype="uint8"),
        np.full(n_tets, VTK_TETRA, dtype="uint8"),
    ])

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_cells, len(connectivity))
        root.create_dataset("Points", data=verts)
        root.create_dataset("Connectivity", data=connectivity)
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Types", data=types)

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=np.array(data))
