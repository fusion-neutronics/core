"""VTK-HDF export for ParaView visualization.

Provides ``to_vtkhdf`` methods on Tally, Geometry, Model, Cell,
Region, NeutronSource/PhotonSource and Tracks objects.  Requires ``h5py``
(install with ``pip install yamc[viz]``).

Usage::

    tally.to_vtkhdf("heating.vtkhdf", scaling_factor=1.6e-19 * 1e21)
    results.with_source_strength(1e20)[tally].to("W/m3").to_vtkhdf(
        "heating.vtkhdf", length_units="m"
    )
    geometry.to_vtkhdf("geometry.vtkhdf", resolution=10000)
    model.to_vtkhdf("model.vtkhdf", samples=1000)  # geometry + source blocks
    model.to_vtkhdf("source.vtkhdf", include=["source"])
    results.tracks.to_vtkhdf("tracks.vtkhdf")  # one polyline per particle
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from yamc._core import reduce_mesh_tally_block

if TYPE_CHECKING:
    import yamc


VTK_TETRA = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_h5py():
    """Import and return h5py, raising a helpful error if not installed."""
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for VTK-HDF export. "
            "Install with: pip install yamc[viz]"
        ) from None


_CM_TO = {"mm": 10.0, "cm": 1.0, "m": 0.01}


def _length_scale(length_units):
    """The factor that turns yamc's cm into ``length_units`` for written
    coordinates. Only positions are scaled; the data arrays are unchanged."""
    if length_units not in _CM_TO:
        raise ValueError(
            f"length_units must be one of {sorted(_CM_TO)}, got {length_units!r}"
        )
    return _CM_TO[length_units]



def _score_name(score):
    """Convert a tally score (int or str) to a dataset-friendly string."""
    if isinstance(score, int):
        return f"MT{score}"
    return str(score)


def _write_imagedata_header(root, origin, spacing, shape, h5py, scale):
    """Write VTK-HDF ImageData header attributes, positions times ``scale``."""
    root.attrs["Version"] = np.array([2, 1], dtype="int64")
    ascii_type = b"ImageData"
    root.attrs.create(
        "Type",
        ascii_type,
        dtype=h5py.string_dtype("ascii", len(ascii_type)),
    )
    root.attrs["Direction"] = np.array(
        [1, 0, 0, 0, 1, 0, 0, 0, 1], dtype="float64"
    )
    root.attrs["Origin"] = np.array(origin, dtype="float64") * scale
    root.attrs["Spacing"] = np.array(spacing, dtype="float64") * scale
    root.attrs["WholeExtent"] = np.array(
        [0, shape[0], 0, shape[1], 0, shape[2]], dtype="int32"
    )


def _write_unstructured_header(root, tally, h5py, scale):
    """Write VTK-HDF UnstructuredGrid header from tally unstructured mesh data."""
    root.attrs["Version"] = np.array([2, 1], dtype="int64")
    ascii_type = b"UnstructuredGrid"
    root.attrs.create(
        "Type",
        ascii_type,
        dtype=h5py.string_dtype("ascii", len(ascii_type)),
    )

    vertices = np.array(tally.unstructured_mesh_vertices, dtype="float64")
    tets = np.array(tally.unstructured_mesh_connectivity, dtype="int64")
    n_points = len(vertices)
    n_cells = len(tets)

    root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype="int64"))
    root.create_dataset("NumberOfCells", data=np.array([n_cells], dtype="int64"))

    connectivity = tets.flatten()
    root.create_dataset(
        "NumberOfConnectivityIds",
        data=np.array([len(connectivity)], dtype="int64"),
    )
    root.create_dataset("Points", data=vertices * scale)
    root.create_dataset("Connectivity", data=connectivity)

    offsets = np.arange(0, n_cells * 4 + 1, 4, dtype="int64")
    root.create_dataset("Offsets", data=offsets)

    types = np.full(n_cells, VTK_TETRA, dtype="uint8")
    root.create_dataset("Types", data=types)


def _compute_grid(ll, ur, resolution):
    """Compute (nx, ny, nz, dx, dy, dz) from bounding box and total voxel count."""
    wx = ur[0] - ll[0]
    wy = ur[1] - ll[1]
    wz = ur[2] - ll[2]
    vol = wx * wy * wz
    if vol <= 0:
        raise ValueError("Bounding box has zero or negative volume")
    voxel_size = (vol / resolution) ** (1 / 3)
    nx = max(1, round(wx / voxel_size))
    ny = max(1, round(wy / voxel_size))
    nz = max(1, round(wz / voxel_size))
    return nx, ny, nz, wx / nx, wy / ny, wz / nz


def _check_finite_bbox(bb, label="Object"):
    """Raise if the bounding box contains infinities."""
    if not bb.is_finite():
        raise ValueError(
            f"{label} bounding box is infinite. "
            "Use a finite geometry or provide bounds explicitly."
        )
    return list(bb.lower_left), list(bb.upper_right)


# ---------------------------------------------------------------------------
# Tally writer
# ---------------------------------------------------------------------------


def mesh_tally_to_vtkhdf(
    filename: str,
    tally: yamc.Tally,
    *,
    scores: Sequence[int | str] | None = None,
    datasets: Sequence[str] = ("mean", "standard_deviation", "relative_error"),
    sum_energy: bool = True,
    sum_nuclides: bool = True,
    volume_normalization: bool = True,
    scaling_factor: float = 1.0,
    length_units: str = "cm",
) -> None:
    """Write mesh tally results to a VTK-HDF file for ParaView visualization.

    Args:
        filename: Output file path (recommended extension: ``.vtkhdf``).
        tally: A :class:`yamc.Tally` with a ``mesh`` or
            ``mesh`` set.
        scores: Which scores to include -- by name (str) or index (int).
            Default ``None`` writes all scores.
        datasets: Which statistics to write as CellData arrays.  Any
            combination of ``"mean"``, ``"standard_deviation"``,
            ``"relative_error"``.
        sum_energy: If *True* (default), sum over energy bins.  If
            *False*, write per-energy-bin arrays (``{score}_E0_mean``, …).
        sum_nuclides: If *True* (default), sum over parent nuclide bins.
            If *False*, write per-nuclide arrays
            (``{score}_{nuclide}_mean``, …).
        volume_normalization: If *True* (default), divide results by the
            mesh element volume.  For regular meshes every voxel has the
            same volume (dx·dy·dz); for unstructured meshes each
            tetrahedron volume is used.
        scaling_factor: Multiplicative factor applied to mean and
            standard_deviation values (default 1.0).  Useful for unit
            conversions (e.g. source strength).  Applied after volume
            normalization.
        length_units: Unit of the written mesh coordinates, ``"cm"``
            (default), ``"m"`` or ``"mm"``. Only the coordinates are scaled:
            volume-normalized values stay per cm³ whatever this is.

    The values are per source particle. For physical units (``"W/m3"``,
    ``"1/(cm2 s)"``, ...) write the result instead, with
    :func:`tally_result_to_vtkhdf`.
    """
    _write_mesh_tally(
        filename,
        tally,
        tally.mean,
        tally.standard_deviation,
        units=None,
        scores=scores,
        datasets=datasets,
        sum_energy=sum_energy,
        sum_nuclides=sum_nuclides,
        volume_normalization=volume_normalization,
        scaling_factor=scaling_factor,
        length_units=length_units,
    )


def tally_result_to_vtkhdf(
    filename: str,
    result: yamc.TallyResult,
    *,
    scores: Sequence[int | str] | None = None,
    datasets: Sequence[str] = ("mean", "standard_deviation", "relative_error"),
    sum_energy: bool = True,
    sum_nuclides: bool = True,
    length_units: str = "cm",
) -> None:
    """Write a mesh tally result, in its own units, to a VTK-HDF file.

    The values are written exactly as ``result`` holds them, so the units are
    whatever ``result.units`` says: set a source strength and convert first to
    get physical units, e.g. ``"W/m3"``, ``"1/(cm2 s)"`` or
    ``"reactions/(m3 s)"`` (``TallyResult.to`` divides by each element's
    volume for a per-volume unit). Each mean and standard deviation dataset
    carries a ``units`` attribute; relative errors carry ``"1"``.

    Pair a per-m³ unit with ``length_units="m"`` so the coordinates and the
    values agree; the two are independent.

    Args:
        filename: Output file path (recommended extension: ``.vtkhdf``).
        result: A mesh tally's :class:`yamc.TallyResult`, e.g.
            ``results.with_source_strength(1e20)[tally].to("W/m3")``.
        scores: Which scores to include -- by name (str) or index (int).
            Default ``None`` writes all scores.
        datasets: Which statistics to write as CellData arrays.  Any
            combination of ``"mean"``, ``"standard_deviation"``,
            ``"relative_error"``.
        sum_energy: If *True* (default), sum over energy bins.
        sum_nuclides: If *True* (default), sum over parent nuclide bins.
        length_units: Unit of the written mesh coordinates, ``"cm"``
            (default), ``"m"`` or ``"mm"``.
    """
    _write_mesh_tally(
        filename,
        result.tally,
        result.mean,
        result.standard_deviation,
        units=result.units,
        scores=scores,
        datasets=datasets,
        sum_energy=sum_energy,
        sum_nuclides=sum_nuclides,
        volume_normalization=False,
        scaling_factor=1.0,
        length_units=length_units,
    )


def _write_mesh_tally(
    filename,
    tally,
    mean,
    standard_deviation,
    *,
    units,
    scores,
    datasets,
    sum_energy,
    sum_nuclides,
    volume_normalization,
    scaling_factor,
    length_units,
):
    """Write ``mean``/``standard_deviation`` (flat, in ``tally``'s bin order)
    on ``tally``'s mesh. ``units`` is one string per score, written as each
    dataset's ``units`` attribute, or ``None`` to write none."""
    _h5py = _require_h5py()
    scale = _length_scale(length_units)

    # --- Discover tally dimensions via properties ----------------------------
    unstructured = tally.has_unstructured_mesh
    mesh = tally.mesh  # RegularRectangularMesh or None

    if getattr(tally, "cylindrical_mesh", None) is not None:
        # ParaView ImageData assumes a regular Cartesian grid, which cannot
        # represent a cylindrical mesh. Volume-normalised results are still
        # available directly via ``tally`` + ``mesh.element_volume(i)``.
        raise NotImplementedError(
            "VTK-HDF export is not yet supported for cylindrical meshes; "
            "access tally results and RegularCylindricalMesh.element_volume() "
            "directly instead."
        )

    if mesh is None and not unstructured:
        raise ValueError("Tally must have a mesh set")

    all_scores = tally.scores
    n_scores = len(all_scores)
    n_parent_bins = tally.n_parent_bins
    n_energy_bins = tally.n_energy_bins
    n_mesh_bins = tally.n_mesh_bins

    # --- Resolve requested score indices ------------------------------------
    if scores is None:
        score_indices = list(range(n_scores))
    else:
        score_indices = []
        for s in scores:
            if isinstance(s, int):
                if s < 0 or s >= n_scores:
                    raise ValueError(
                        f"Score index {s} out of range (0..{n_scores - 1})"
                    )
                score_indices.append(s)
            else:
                for i, sc in enumerate(all_scores):
                    if str(sc) == str(s):
                        score_indices.append(i)
                        break
                else:
                    raise ValueError(f"Score '{s}' not found in tally")

    # --- Fetch and reshape raw data to 4-D ----------------------------------
    # Flat layout: [score][parent][energy][mesh]  (mesh fastest)
    raw_mean = np.array(mean)
    expected_bins = n_scores * n_parent_bins * n_energy_bins * n_mesh_bins
    if raw_mean.size != expected_bins:
        raise ValueError(
            f"Tally has {raw_mean.size} result bins but "
            f"scores x parent_nuclides x energy_bins x mesh_bins = {expected_bins}. "
            "Tallies with additional bin dimensions (e.g. a per-nuclide "
            "``nuclides`` list) are not supported by VTK-HDF export."
        )
    raw_mean = raw_mean.reshape(
        n_scores, n_parent_bins, n_energy_bins, n_mesh_bins
    )
    raw_std = np.array(standard_deviation).reshape(
        n_scores, n_parent_bins, n_energy_bins, n_mesh_bins
    )

    # --- Iteration labels for parent / energy dimensions --------------------
    parent_nuclide_names = tally.parent_nuclides
    if sum_nuclides or parent_nuclide_names is None:
        parent_iter = [(None, None)]  # will sum
    else:
        parent_iter = [
            (nuc, i) for i, nuc in enumerate(parent_nuclide_names)
        ]

    energy_bin_edges = tally.energy_bins
    if sum_energy or energy_bin_edges is None:
        energy_iter = [(None, None)]  # will sum
    else:
        energy_iter = [(f"E{i}", i) for i in range(n_energy_bins)]

    # --- Mesh 3-D shape for ImageData ---------------------------------------
    if not unstructured:
        nx, ny, nz = mesh.shape

    # --- Compute per-element volumes for normalization ----------------------
    if volume_normalization:
        if not unstructured:
            dx, dy, dz = mesh.width
            vol = np.full(n_mesh_bins, dx * dy * dz)
        else:
            # One boundary crossing for all element volumes.
            vol = np.asarray(tally.mesh_element_volumes())

    # --- Build CellData dict ------------------------------------------------
    # The numeric reduction (sum over parent/energy bins in quadrature, volume
    # normalization, scaling, relative error) runs in the Rust core; Python
    # keeps the dataset naming and the h5py write. The whole score block crosses
    # the boundary once per score.
    vol_arg = vol.tolist() if volume_normalization else None
    parent_indices = [p_idx for _, p_idx in parent_iter]
    energy_indices = [e_idx for _, e_idx in energy_iter]

    cell_data = {}

    for si in score_indices:
        sname = _score_name(all_scores[si])
        # raw_mean[si] is (n_parent, n_energy, n_mesh); flatten C-order so the
        # layout matches the Rust reducer's [parent][energy][mesh] expectation.
        block_mean = raw_mean[si].reshape(-1).tolist()
        block_std = raw_std[si].reshape(-1).tolist()

        reductions = reduce_mesh_tally_block(
            block_mean,
            block_std,
            n_parent_bins,
            n_energy_bins,
            n_mesh_bins,
            parent_indices,
            energy_indices,
            vol_arg,
            scaling_factor,
        )

        red_idx = 0
        for p_label, p_idx in parent_iter:
            for e_label, e_idx in energy_iter:
                m, s, r = reductions[red_idx]
                red_idx += 1
                m = np.asarray(m, dtype="float64")
                s = np.asarray(s, dtype="float64")
                r = np.asarray(r, dtype="float64")

                # Dataset name: {score}[_{nuclide}][_E{i}]_{value}
                parts = [sname]
                if p_label is not None:
                    parts.append(p_label)
                if e_label is not None:
                    parts.append(e_label)
                prefix = "_".join(parts)

                # Energy bin boundaries and units as dataset attribute metadata
                attrs = {}
                if e_idx is not None and energy_bin_edges is not None:
                    attrs["energy_low_eV"] = energy_bin_edges[e_idx]
                    attrs["energy_high_eV"] = energy_bin_edges[e_idx + 1]

                for val in datasets:
                    if val == "mean":
                        arr = m
                    elif val == "standard_deviation":
                        arr = s
                    elif val == "relative_error":
                        arr = r
                    else:
                        raise ValueError(f"Unknown value type '{val}'")

                    ds_attrs = dict(attrs)
                    if units is not None:
                        ds_attrs["units"] = "1" if val == "relative_error" else units[si]
                    ds_name = f"{prefix}_{val}"
                    if not unstructured:
                        cell_data[ds_name] = (arr.reshape(nz, ny, nx), ds_attrs)
                    else:
                        cell_data[ds_name] = (arr, ds_attrs)

    # --- Write HDF5 file ----------------------------------------------------
    with _h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")

        if not unstructured:
            ll = mesh.lower_left
            _write_imagedata_header(
                root, ll, mesh.width, mesh.shape, _h5py, scale
            )
        else:
            _write_unstructured_header(root, tally, _h5py, scale)

        cd = root.create_group("CellData")
        for name, (data, ds_attrs) in cell_data.items():
            ds = cd.create_dataset(name, data=data.astype("float64"))
            for key, value in ds_attrs.items():
                ds.attrs[key] = value


# ---------------------------------------------------------------------------
# Geometry writer
# ---------------------------------------------------------------------------


def _sample_geometry_grid(geometry, resolution, datasets):
    """Voxelize a geometry; return (ll, spacing, shape, cell_data dict).

    The per-voxel cell/material lookup runs in the Rust core
    (``Geometry.voxelize``); Python only computes the grid sizing and reshapes
    the flat result.
    """
    ds_set = set(datasets)
    _GEOM_DATASETS = {"material_id", "cell_id"}
    unknown = ds_set - _GEOM_DATASETS
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}. Valid options: {_GEOM_DATASETS}"
        )

    bb = geometry.bounding_box()
    ll, ur = _check_finite_bbox(bb, "Geometry")
    nx, ny, nz, dx, dy, dz = _compute_grid(ll, ur, resolution)

    vox = geometry.voxelize(ll, (dx, dy, dz), (nx, ny, nz))

    cell_data = {}
    if "material_id" in ds_set:
        cell_data["material_id"] = np.asarray(
            vox.material_ids, dtype="int32"
        ).reshape(nz, ny, nx)
    if "cell_id" in ds_set:
        cell_data["cell_id"] = np.asarray(
            vox.cell_ids, dtype="int32"
        ).reshape(nz, ny, nx)
    return ll, [dx, dy, dz], [nx, ny, nz], cell_data


def _write_geometry_group(root, ll, spacing, shape, cell_data, h5py, scale):
    """Write a voxelized geometry as ImageData into an open VTKHDF group."""
    _write_imagedata_header(root, ll, spacing, shape, h5py, scale)
    cd = root.create_group("CellData")
    for name, arr in cell_data.items():
        cd.create_dataset(name, data=arr)


def geometry_to_vtkhdf(
    filename: str,
    geometry: yamc.Geometry,
    *,
    resolution: int = 10000,
    datasets: Sequence[str] = ("material_id", "cell_id"),
    length_units: str = "cm",
) -> None:
    """Write geometry to VTK-HDF ImageData with material and cell IDs.

    Samples the geometry on a 3-D regular grid and writes the requested
    CellData arrays.

    Args:
        filename: Output ``.vtkhdf`` path.
        geometry: A :class:`yamc.Geometry` object.
        resolution: Approximate total number of voxels (default 10 000).
        datasets: Which CellData arrays to write.  Any combination of
            ``"material_id"`` and ``"cell_id"``.
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    _h5py = _require_h5py()
    scale = _length_scale(length_units)
    ll, spacing, shape, cell_data = _sample_geometry_grid(
        geometry, resolution, datasets
    )
    with _h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_geometry_group(root, ll, spacing, shape, cell_data, _h5py, scale)


# ---------------------------------------------------------------------------
# Cell writer
# ---------------------------------------------------------------------------


def cell_to_vtkhdf(
    filename: str,
    cell: yamc.Cell,
    *,
    resolution: int = 10000,
    datasets: Sequence[str] = ("material_id", "cell_id"),
    length_units: str = "cm",
) -> None:
    """Write a single cell to VTK-HDF ImageData.

    Samples the cell on a 3-D regular grid covering its bounding box and
    writes the requested CellData arrays.

    Args:
        filename: Output ``.vtkhdf`` path.
        cell: A :class:`yamc.Cell` object.
        resolution: Approximate total number of voxels (default 10 000).
        datasets: Which CellData arrays to write.  Any combination of
            ``"material_id"`` and ``"cell_id"``.
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    _h5py = _require_h5py()
    scale = _length_scale(length_units)
    ds_set = set(datasets)
    _CELL_DATASETS = {"material_id", "cell_id"}
    unknown = ds_set - _CELL_DATASETS
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}. Valid options: {_CELL_DATASETS}"
        )

    bb = cell.bounding_box()
    ll, ur = _check_finite_bbox(bb, "Cell")
    nx, ny, nz, dx, dy, dz = _compute_grid(ll, ur, resolution)

    want_cell = "cell_id" in ds_set
    want_mat = "material_id" in ds_set

    cid = cell.id if cell.id is not None else 1
    mid = -1
    if want_mat:
        mat = cell.material
        if mat is not None:
            mid = mat.id

    # The per-voxel inside/outside test runs in the Rust core; Python paints
    # the resolved cell/material id onto the returned mask.
    inside = np.asarray(
        cell.voxelize(ll, (dx, dy, dz), (nx, ny, nz)), dtype="int32"
    ).reshape(nz, ny, nx)

    with _h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_imagedata_header(root, ll, [dx, dy, dz], [nx, ny, nz], _h5py, scale)
        cd = root.create_group("CellData")
        if want_mat:
            cd.create_dataset(
                "material_id", data=np.where(inside == 1, mid, -1).astype("int32")
            )
        if want_cell:
            cd.create_dataset(
                "cell_id", data=np.where(inside == 1, cid, -1).astype("int32")
            )


# ---------------------------------------------------------------------------
# Region writer
# ---------------------------------------------------------------------------


def region_to_vtkhdf(
    filename: str,
    region: yamc.Region | yamc.Halfspace,
    *,
    resolution: int = 10000,
    length_units: str = "cm",
) -> None:
    """Write a region to VTK-HDF ImageData.

    Samples the region on a 3-D regular grid and writes a ``region``
    CellData array (1 = inside, 0 = outside).

    Args:
        filename: Output ``.vtkhdf`` path.
        region: A :class:`yamc.Region` or halfspace object.
        resolution: Approximate total number of voxels (default 10 000).
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    _h5py = _require_h5py()
    scale = _length_scale(length_units)

    bb = region.bounding_box()
    ll, ur = _check_finite_bbox(bb, "Region")
    nx, ny, nz, dx, dy, dz = _compute_grid(ll, ur, resolution)

    # The per-voxel inside/outside test runs in the Rust core.
    inside = np.asarray(
        region.voxelize(ll, (dx, dy, dz), (nx, ny, nz)), dtype="int32"
    ).reshape(nz, ny, nx)

    with _h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_imagedata_header(root, ll, [dx, dy, dz], [nx, ny, nz], _h5py, scale)
        cd = root.create_group("CellData")
        cd.create_dataset("region", data=inside)


# ---------------------------------------------------------------------------
# Source writer
# ---------------------------------------------------------------------------

VTK_VERTEX = 1


def _sample_source_points(sources, samples):
    """Sample sources; return (points, energies, source_ids) arrays."""
    # Accept a single source or a list
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    all_points = []
    all_energies = []
    all_source_ids = []

    for src_idx, src in enumerate(sources):
        # Batch-sample in the Rust core instead of one crossing per particle.
        positions, energies = src.sample_n(samples)
        all_points.extend(positions)
        all_energies.extend(energies)
        all_source_ids.extend([src_idx] * len(positions))

    points = np.asarray(all_points, dtype="float64").reshape(-1, 3)
    energies = np.asarray(all_energies, dtype="float64")
    source_ids = np.asarray(all_source_ids, dtype="int32")
    return points, energies, source_ids


def _write_source_group(root, points, energies, source_ids, h5py, scale):
    """Write sampled source points as a vertex cloud into an open VTKHDF group."""
    n = len(points)
    root.attrs["Version"] = np.array([2, 1], dtype="int64")
    ascii_type = b"UnstructuredGrid"
    root.attrs.create(
        "Type",
        ascii_type,
        dtype=h5py.string_dtype("ascii", len(ascii_type)),
    )

    root.create_dataset(
        "NumberOfPoints", data=np.array([n], dtype="int64")
    )
    root.create_dataset(
        "NumberOfCells", data=np.array([n], dtype="int64")
    )
    root.create_dataset(
        "NumberOfConnectivityIds", data=np.array([n], dtype="int64")
    )
    root.create_dataset("Points", data=points * scale)
    root.create_dataset(
        "Connectivity", data=np.arange(n, dtype="int64")
    )
    root.create_dataset(
        "Offsets", data=np.arange(n + 1, dtype="int64")
    )
    root.create_dataset(
        "Types", data=np.full(n, VTK_VERTEX, dtype="uint8")
    )

    pd = root.create_group("PointData")
    pd.create_dataset("energy", data=energies)
    pd.create_dataset("source_id", data=source_ids)


def source_to_vtkhdf(
    filename: str,
    sources: yamc.NeutronSource
    | yamc.PhotonSource
    | Sequence[yamc.NeutronSource | yamc.PhotonSource],
    *,
    samples: int = 1000,
    length_units: str = "cm",
) -> None:
    """Write source sample points to a VTK-HDF UnstructuredGrid file.

    Samples each source ``samples`` times and writes the positions as a
    point cloud.  Each point carries ``energy`` and ``source_id``
    PointData so sources can be distinguished in ParaView.

    Args:
        filename: Output ``.vtkhdf`` path.
        sources: A single :class:`yamc.NeutronSource` (or :class:`yamc.PhotonSource`) or a list of
            them.
        samples: Number of sample points *per source* (default 1 000).
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    _h5py = _require_h5py()
    scale = _length_scale(length_units)
    points, energies, source_ids = _sample_source_points(sources, samples)
    with _h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        _write_source_group(root, points, energies, source_ids, _h5py, scale)


# ---------------------------------------------------------------------------
# Model writer
# ---------------------------------------------------------------------------


def model_to_vtkhdf(
    filename: str,
    model: yamc.Model,
    *,
    include: str | Sequence[str] = ("geometry", "source"),
    resolution: int = 10000,
    datasets: Sequence[str] = ("material_id", "cell_id"),
    samples: int = 1000,
    length_units: str = "cm",
) -> None:
    """Write model geometry and/or sampled source points to one VTK-HDF file.

    With both blocks selected (the default) the file is a VTK-HDF
    PartitionedDataSetCollection holding an ImageData block named
    ``geometry`` and a point-cloud block named ``source``; opening it in
    ParaView (5.12+) shows both together.  With a single entry the file
    is a plain dataset, identical to ``Geometry.to_vtkhdf`` /
    ``NeutronSource.to_vtkhdf`` output.

    Args:
        filename: Output ``.vtkhdf`` path.
        model: A :class:`yamc.Model`.
        include: Which blocks to write -- an iterable containing
            ``"geometry"`` and/or ``"source"`` (default both).
        resolution: Approximate total voxel count for the geometry block
            (default 10 000).
        datasets: Geometry CellData arrays to write.  Any combination of
            ``"material_id"`` and ``"cell_id"``.
        samples: Number of sample points per source for the source block
            (default 1 000).
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    _h5py = _require_h5py()
    scale = _length_scale(length_units)

    if isinstance(include, str):
        include = (include,)
    include = tuple(dict.fromkeys(include))  # dedupe, preserve order
    unknown = set(include) - {"geometry", "source"}
    if unknown:
        raise ValueError(
            f"Unknown include entries {unknown}. "
            "Valid options: 'geometry', 'source'"
        )
    if not include:
        raise ValueError("include must contain 'geometry' and/or 'source'")

    if len(include) == 1:
        if include[0] == "geometry":
            geometry_to_vtkhdf(
                filename,
                model.geometry,
                resolution=resolution,
                datasets=datasets,
                length_units=length_units,
            )
        else:
            source_to_vtkhdf(
                filename, model.source, samples=samples, length_units=length_units
            )
        return

    # Sample everything before touching the file so bad inputs fail fast
    # without leaving a partial file behind.
    blocks = {
        "geometry": _sample_geometry_grid(model.geometry, resolution, datasets),
        "source": _sample_source_points(model.source, samples),
    }

    with _h5py.File(filename, "w") as f:
        # The spec requires creation-order tracking on the VTKHDF group,
        # the Assembly group, and its children.
        root = f.create_group("VTKHDF", track_order=True)
        root.attrs["Version"] = np.array([2, 2], dtype="int64")
        ascii_type = b"PartitionedDataSetCollection"
        root.attrs.create(
            "Type",
            ascii_type,
            dtype=_h5py.string_dtype("ascii", len(ascii_type)),
        )

        for idx, name in enumerate(include):
            block = root.create_group(name, track_order=True)
            block.attrs["Index"] = np.int64(idx)
            if name == "geometry":
                _write_geometry_group(block, *blocks["geometry"], _h5py, scale)
            else:
                _write_source_group(block, *blocks["source"], _h5py, scale)

        assembly = root.create_group("Assembly", track_order=True)
        for name in include:
            assembly[name] = _h5py.SoftLink(f"/VTKHDF/{name}")


# ---------------------------------------------------------------------------
# Polyline writer (particle tracks)
# ---------------------------------------------------------------------------


def polylines_to_vtkhdf(
    filename: str,
    points: Sequence[Sequence[float]],
    offsets: Sequence[int],
    *,
    point_data: dict | None = None,
    cell_data: dict | None = None,
    length_units: str = "cm",
) -> None:
    """Write polylines to a VTK-HDF PolyData file.

    Line ``j`` runs through ``points[offsets[j]:offsets[j + 1]]``.

    Args:
        filename: Output ``.vtkhdf`` path.
        points: ``[x, y, z]`` per point (cm).
        offsets: Start of each line in ``points``, plus the point count
            (length = number of lines + 1, first entry 0).
        point_data: Arrays of one value per point, keyed by name.
        cell_data: Arrays of one value per line, keyed by name.
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    h5py = _require_h5py()
    scale = _length_scale(length_units)
    points = np.asarray(points, dtype="float64").reshape(-1, 3)
    offsets = np.asarray(offsets, dtype="int64")
    n_points, n_lines = len(points), len(offsets) - 1
    if n_lines < 0 or offsets[0] != 0 or offsets[-1] != n_points:
        raise ValueError(
            f"offsets must start at 0 and end at the point count ({n_points}), "
            f"got {offsets[:1].tolist()} ... {offsets[-1:].tolist()}"
        )
    for kind, data, n in (("point", point_data, n_points), ("cell", cell_data, n_lines)):
        for name, values in (data or {}).items():
            if len(values) != n:
                raise ValueError(f"{kind}_data[{name!r}] has {len(values)} values, expected {n}")

    with h5py.File(filename, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 1], dtype="int64")
        root.attrs.create("Type", b"PolyData", dtype=h5py.string_dtype("ascii", 8))
        root.create_dataset("NumberOfPoints", data=np.array([n_points], dtype="int64"))
        root.create_dataset("Points", data=points * scale)
        # PolyData requires all four topology groups; only Lines is populated.
        for name in ("Vertices", "Lines", "Polygons", "Strips"):
            lines = name == "Lines"
            g = root.create_group(name)
            g.create_dataset("NumberOfCells", data=np.array([n_lines if lines else 0], dtype="int64"))
            g.create_dataset(
                "NumberOfConnectivityIds", data=np.array([n_points if lines else 0], dtype="int64")
            )
            g.create_dataset("Offsets", data=offsets if lines else np.zeros(1, dtype="int64"))
            g.create_dataset(
                "Connectivity", data=np.arange(n_points if lines else 0, dtype="int64")
            )
        for group, data in (("PointData", point_data), ("CellData", cell_data)):
            g = root.create_group(group)
            for name, values in (data or {}).items():
                g.create_dataset(name, data=np.asarray(values))


def tracks_to_vtkhdf(
    filename: str, tracks: yamc.Tracks, *, length_units: str = "cm"
) -> None:
    """Write particle tracks to a VTK-HDF PolyData file, one polyline per track.

    Per point (``PointData``): ``energy`` (eV, after the event), ``weight``,
    ``event_type`` and ``cell_id`` (-1 if unknown). Per line (``CellData``):
    ``particle_type``, ``particle_id``, ``parent_id`` (-1 for source
    particles), ``generation`` and ``history``. ``event_type`` and
    ``particle_type`` are integer codes (ParaView colours by numbers); their
    names are stored in the root attributes ``event_type_names`` and
    ``particle_type_names``. Coordinates are in ``length_units``, so tracks
    overlay ``Model.to_vtkhdf`` output written with the same unit in ParaView.

    Args:
        filename: Output ``.vtkhdf`` path.
        tracks: The ``Tracks`` from ``Model.simulate_transport(capture_tracks=...)``.
        length_units: Unit of the written coordinates, ``"cm"``
            (default, yamc's own), ``"m"`` or ``"mm"``. Only positions are
            scaled, so files written with the same unit overlay in ParaView;
            data arrays are written exactly as without it.
    """
    h5py = _require_h5py()
    p = tracks.polylines()
    polylines_to_vtkhdf(
        filename,
        p["points"],
        p["offsets"],
        point_data={k: p[k] for k in ("energy", "weight", "event_type", "cell_id")},
        cell_data={
            k: p[k] for k in ("particle_type", "particle_id", "parent_id", "generation", "history")
        },
        length_units=length_units,
    )
    with h5py.File(filename, "a") as f:
        for key in ("event_type_names", "particle_type_names"):
            f["VTKHDF"].attrs[key] = np.array(p[key], dtype=h5py.string_dtype("ascii"))
