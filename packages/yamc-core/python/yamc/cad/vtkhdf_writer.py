"""VTKHDF v2.1 UnstructuredGrid writer for ParaView visualization.

No numpy. Every buffer here is built once and handed straight to h5py, which
takes an ordinary sequence as long as the dtype is given explicitly, so the
arrays this used to build were converted twice: the mesh arrives from the Rust
core as `list[list[...]]` (see `SurfaceMesh.vertices`), numpy copied it, and
h5py copied it again. Dropping the copy is marginally faster and leaves the
cad extra with no numpy dependency at all.

The `dtype=` arguments are load-bearing, not decoration. They pin what lands on
disk, and VTKHDF readers care: without them an int buffer's width would follow
the platform. Do not remove them.
"""

VTK_TRIANGLE = 5
VTK_TETRA = 10


def _flatten(rows):
    """Flatten a sequence of index tuples into one connectivity list."""
    return [i for row in rows for i in row]


def _write_vtkhdf_header(root, n_points, n_cells, n_connectivity_ids):
    """Write the VTKHDF header attributes and metadata datasets."""
    import h5py
    # `attrs.create` rather than `attrs["Version"] = [2, 1]`: plain assignment
    # would let h5py pick the integer width, and VTKHDF wants int64.
    root.attrs.create("Version", [2, 1], dtype="int64")
    ascii_type = "UnstructuredGrid".encode("ascii")
    root.attrs.create(
        "Type",
        ascii_type,
        dtype=h5py.string_dtype("ascii", len(ascii_type)),
    )
    root.create_dataset("NumberOfPoints", data=[n_points], dtype="int64")
    root.create_dataset("NumberOfCells", data=[n_cells], dtype="int64")
    root.create_dataset(
        "NumberOfConnectivityIds",
        data=[n_connectivity_ids],
        dtype="int64",
    )


def surface_to_vtkhdf(filename, vertices, triangles, cell_data=None):
    """Write surface triangle mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: Sequence of [x, y, z] triples, shape (N, 3).
        triangles: Sequence of [i, j, k] triples, shape (M, 3).
        cell_data: Optional dict of name -> sequence (one value per triangle).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    n_tris = len(triangles)
    n_points = len(vertices)

    connectivity = _flatten(triangles)
    offsets = list(range(0, n_tris * 3 + 1, 3))
    types = [VTK_TRIANGLE] * n_tris

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_tris, len(connectivity))
        root.create_dataset("Points", data=vertices, dtype="float64")
        root.create_dataset("Connectivity", data=connectivity, dtype="int64")
        root.create_dataset("Offsets", data=offsets, dtype="int64")
        root.create_dataset("Types", data=types, dtype="uint8")

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=data)


def tet_to_vtkhdf(filename, vertices, tets, cell_data=None):
    """Write tetrahedral mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: Sequence of [x, y, z] triples, shape (N, 3).
        tets: Sequence of [i, j, k, l] quadruples, shape (M, 4).
        cell_data: Optional dict of name -> sequence (one value per tet).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    n_tets = len(tets)
    n_points = len(vertices)

    connectivity = _flatten(tets)
    offsets = list(range(0, n_tets * 4 + 1, 4))
    types = [VTK_TETRA] * n_tets

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_tets, len(connectivity))
        root.create_dataset("Points", data=vertices, dtype="float64")
        root.create_dataset("Connectivity", data=connectivity, dtype="int64")
        root.create_dataset("Offsets", data=offsets, dtype="int64")
        root.create_dataset("Types", data=types, dtype="uint8")

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=data)


def mixed_to_vtkhdf(filename, vertices, triangles, tets, cell_data=None):
    """Write mixed triangle + tetrahedron mesh to VTKHDF UnstructuredGrid.

    Args:
        filename: Output file path.
        vertices: Sequence of [x, y, z] triples, shape (N, 3).
        triangles: Sequence of [i, j, k] triples, shape (M_tri, 3).
        tets: Sequence of [i, j, k, l] quadruples, shape (M_tet, 4).
        cell_data: Optional dict of name -> sequence (one value per cell,
            triangles first then tets).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTKHDF export. Install with: pip install h5py"
        ) from None

    n_tris = len(triangles)
    n_tets = len(tets)
    n_cells = n_tris + n_tets
    n_points = len(vertices)

    # Build connectivity, offsets, types for mixed mesh. Tet vertices are
    # already in `vertices` (the mesh is conformal), so this is a plain append.
    connectivity = _flatten(triangles) + _flatten(tets)

    offsets = [0]
    for _ in range(n_tris):
        offsets.append(offsets[-1] + 3)
    for _ in range(n_tets):
        offsets.append(offsets[-1] + 4)

    types = [VTK_TRIANGLE] * n_tris + [VTK_TETRA] * n_tets

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_vtkhdf_header(root, n_points, n_cells, len(connectivity))
        root.create_dataset("Points", data=vertices, dtype="float64")
        root.create_dataset("Connectivity", data=connectivity, dtype="int64")
        root.create_dataset("Offsets", data=offsets, dtype="int64")
        root.create_dataset("Types", data=types, dtype="uint8")

        if cell_data:
            cd = root.create_group("CellData")
            for name, data in cell_data.items():
                cd.create_dataset(name, data=data)
