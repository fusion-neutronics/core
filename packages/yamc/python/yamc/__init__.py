"""yamc -- Yet Another Monte Carlo code."""

import sys
import types

# Single source of truth for the version: the installed package metadata
# (populated by maturin from pyproject's ``[project].version``). Read it at
# import time so there is no second hardcoded version string to keep in sync.
# Falls back gracefully when the package isn't installed (e.g. imported
# straight from a source checkout).
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _pkg_version,
)

try:
    __version__ = _pkg_version("yamc")
except _PackageNotFoundError:  # pragma: no cover - only without an installed dist
    __version__ = "0.0.0+unknown"

from yamc._core import *  # noqa: F401,F403

# The git commit the extension was built from, and whether a tracked file
# differed from it. None when it was not built from a git checkout of core.
from yamc._core import __commit__, __commit_dirty__  # noqa: F401

# Re-export submodules. sources (distribution zoo), parallel (MPI/GPU
# helpers), materials (curated material collections) and shapes (the lumps
# self-shielding takes) are PyO3 submodules of _core, like data.
from yamc._core import data, materials, parallel, shapes, sources  # noqa: F401

# Explicit imports for names used below (silences ruff F405). ``X as X``
# marks intentional re-exports (PEP 484) for type checkers; griffe doesn't
# honor that convention yet, so the API docs additionally mark these public
# via docs/griffe_ext.py:PublicApi.
from yamc._core import (  # noqa: F811
    Cell as Cell,
    DataUncertainty as DataUncertainty,
    Geometry as Geometry,
    Halfspace as Halfspace,
    Model as Model,
    NeutronSource as NeutronSource,
    PhotonSource as PhotonSource,
    Processing as Processing,
    Region as Region,
    Tally as Tally,
    Tracks as Tracks,
    get_transmutation_decay_data as _get_transmutation_decay_data,
    get_transmutation_reactions as _get_transmutation_reactions,
    get_transmutation_fission_yields as _get_transmutation_fission_yields,
    get_transmutation_branch_ratios as _get_transmutation_branch_ratios,
    get_neutron_cross_section_data as _get_neutron_cross_section_data,
    get_photon_cross_section_data as _get_photon_cross_section_data,
    available_nuclear_data_libraries as available_nuclear_data_libraries,  # noqa: F401
    lookup_photon_cross_section_data as lookup_photon_cross_section_data,  # noqa: F401
    lookup_neutron_cross_section_data as lookup_neutron_cross_section_data,  # noqa: F401  (public helper)
    set_transmutation_decay_data as _set_transmutation_decay_data,
    set_transmutation_reactions as _set_transmutation_reactions,
    set_transmutation_fission_yields as _set_transmutation_fission_yields,
    set_transmutation_branch_ratios as _set_transmutation_branch_ratios,
    set_neutron_cross_section_data as _set_neutron_cross_section_data,
    set_photon_cross_section_data as _set_photon_cross_section_data,
    set_neutron_cross_section_data_entry as set_neutron_cross_section_data_entry,  # noqa: F401  (public helper)
)

# The dotted paths (`from yamc.sources import Discrete`) are registered by the
# extension itself, next to the `yamc._core.<name>` registration it has always
# done, so a new submodule cannot arrive with only one of the two names.


# ---------------------------------------------------------------------------
# Module-level properties for transport and transmutation data
# ---------------------------------------------------------------------------

class _YamcModule(types.ModuleType):
    """Module subclass exposing particle data and the per-subsection
    transmutation_* sources as properties."""

    def __setattr__(self, name, value):
        if name == "cross_section_data":
            raise AttributeError(
                "cross_section_data was removed; set neutron_cross_section_data "
                "and photon_cross_section_data independently"
            )
        super().__setattr__(name, value)

    @property
    def neutron_cross_section_data(self):
        return _get_neutron_cross_section_data()

    @neutron_cross_section_data.setter
    def neutron_cross_section_data(self, value):
        _set_neutron_cross_section_data(value)

    @property
    def photon_cross_section_data(self):
        return _get_photon_cross_section_data()

    @photon_cross_section_data.setter
    def photon_cross_section_data(self, value):
        _set_photon_cross_section_data(value)

    @property
    def transmutation_decay_data(self):
        return _get_transmutation_decay_data()

    @transmutation_decay_data.setter
    def transmutation_decay_data(self, value):
        _set_transmutation_decay_data(value)

    @property
    def transmutation_reactions(self):
        return _get_transmutation_reactions()

    @transmutation_reactions.setter
    def transmutation_reactions(self, value):
        _set_transmutation_reactions(value)

    @property
    def transmutation_fission_yields(self):
        return _get_transmutation_fission_yields()

    @transmutation_fission_yields.setter
    def transmutation_fission_yields(self, value):
        _set_transmutation_fission_yields(value)

    @property
    def transmutation_branch_ratios(self):
        return _get_transmutation_branch_ratios()

    @transmutation_branch_ratios.setter
    def transmutation_branch_ratios(self, value):
        _set_transmutation_branch_ratios(value)


sys.modules[__name__].__class__ = _YamcModule



# `Enriched` and `enriched()` are a pyclass and pyfunction in the bindings
# crate, arriving through the `_core` star-import above. They were a dataclass
# here and a second, unrelated dataclass in yani, so the two wheels disagreed on
# the type and the generated stub had to name one wheel and import it into the
# other.


# ---------------------------------------------------------------------------
# to_vtkhdf methods (lazy-imported to avoid hard h5py/numpy dependency)
# ---------------------------------------------------------------------------

def _geometry_to_vtkhdf(self, filename, **kwargs):
    """Write geometry to a VTK-HDF ImageData file with material and cell IDs.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.geometry_to_vtkhdf`` (``resolution``, ``datasets``,
            ``length_units``).
    """
    from yamc.vtkhdf import geometry_to_vtkhdf
    geometry_to_vtkhdf(filename, self, **kwargs)


def _cell_to_vtkhdf(self, filename, **kwargs):
    """Write cell to a VTK-HDF ImageData file with material and cell IDs.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.cell_to_vtkhdf`` (``resolution``, ``datasets``,
            ``length_units``).
    """
    from yamc.vtkhdf import cell_to_vtkhdf
    cell_to_vtkhdf(filename, self, **kwargs)


def _region_to_vtkhdf(self, filename, *, resolution=10000, length_units="cm"):
    """Write region to a VTK-HDF ImageData file.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        resolution (int): Approximate total number of voxels (default 10 000).
        length_units (str): Unit of the written coordinates, ``"cm"``
            (default), ``"m"`` or ``"mm"``.
    """
    from yamc.vtkhdf import region_to_vtkhdf
    region_to_vtkhdf(filename, self, resolution=resolution, length_units=length_units)


def _model_to_vtkhdf(self, filename, **kwargs):
    """Write model geometry and/or sampled source points to one VTK-HDF file.

    By default writes both as a two-block PartitionedDataSetCollection
    (``geometry`` voxel grid + ``source`` point cloud, ParaView 5.12+).
    Pass ``include=("geometry",)`` or ``include=("source",)`` to write a
    plain single-dataset file instead.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.model_to_vtkhdf`` (``include``, ``resolution``,
            ``datasets``, ``samples``, ``length_units``).
    """
    from yamc.vtkhdf import model_to_vtkhdf
    model_to_vtkhdf(filename, self, **kwargs)


def _source_to_vtkhdf(self, filename, **kwargs):
    """Write sampled source points to a VTK-HDF file for ParaView visualization.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended extension: ``.vtkhdf``)
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.source_to_vtkhdf`` (``samples``, ``length_units``).
    """
    from yamc.vtkhdf import source_to_vtkhdf
    source_to_vtkhdf(filename, self, **kwargs)


def _tally_to_vtkhdf(self, filename, **kwargs):
    """Write mesh tally results to a VTK-HDF file for ParaView visualization.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended extension: ``.vtkhdf``)
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.mesh_tally_to_vtkhdf`` (``scores``, ``datasets``,
            ``sum_energy``, ``sum_nuclides``, ``volume_normalization``,
            ``scaling_factor``, ``length_units``).
    """
    from yamc.vtkhdf import mesh_tally_to_vtkhdf
    mesh_tally_to_vtkhdf(filename, self, **kwargs)


def _tracks_to_vtkhdf(self, filename, *, length_units="cm"):
    """Write particle tracks to a VTK-HDF PolyData file, one polyline per track.

    Opens in ParaView alongside ``model.to_vtkhdf(...)`` output. Per-point
    data: energy, weight, event type, cell ID; per-line data: particle type,
    particle ID, parent ID, generation, history.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        length_units (str): Unit of the written coordinates, ``"cm"``
            (default), ``"m"`` or ``"mm"``. Use the model's unit to overlay.
    """
    from yamc.vtkhdf import tracks_to_vtkhdf
    tracks_to_vtkhdf(filename, self, length_units=length_units)


def _mesh_geometry_to_vtkhdf(self, path, *, length_units="cm"):
    """Write the mesh's triangles and tets to a VTKHDF UnstructuredGrid file,
    for ParaView.

    Each cell carries ``volume_id``: the 1-based volume it belongs to (the
    solid id the mesher gave it), or -1 for a triangle in no volume. A
    surface two volumes share is written once, with the first of them. Its
    ``material_id`` is the 0-based index into ``volume_names``, or -1.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        path (str | os.PathLike): Output file path (e.g. ``"model.vtkhdf"``).
        length_units (str): Unit of the written coordinates, ``"cm"``
            (default), ``"m"`` or ``"mm"``.
    """
    import os

    import numpy as np

    from yamc._vtkhdf_writer import mixed_to_vtkhdf, surface_to_vtkhdf
    from yamc.vtkhdf import _length_scale

    vertices = np.asarray(self.vertices, dtype="float64") * _length_scale(length_units)

    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    triangles = self.triangles
    tri_volume = [-1] * len(triangles)
    for v in range(self.num_volumes):
        for t in self.volume_triangle_indices(v):
            if tri_volume[t] < 0:
                tri_volume[t] = v
    tets = self.tetrahedra
    tet_volume = [-1 if v is None else v for v in self.tet_volume_ids]
    material_id = tri_volume + tet_volume
    volume_id = [v + 1 if v >= 0 else -1 for v in material_id]
    cell_data = {"volume_id": volume_id, "material_id": material_id}
    if tets:
        mixed_to_vtkhdf(path, vertices, triangles, tets, cell_data=cell_data)
    else:
        surface_to_vtkhdf(path, vertices, triangles, cell_data=cell_data)


Geometry.to_vtkhdf = _geometry_to_vtkhdf
Cell.to_vtkhdf = _cell_to_vtkhdf
Region.to_vtkhdf = _region_to_vtkhdf
Halfspace.to_vtkhdf = _region_to_vtkhdf
Model.to_vtkhdf = _model_to_vtkhdf
NeutronSource.to_vtkhdf = _source_to_vtkhdf
PhotonSource.to_vtkhdf = _source_to_vtkhdf
Tally.to_vtkhdf = _tally_to_vtkhdf
# The Pyodide wheel is built without the mesh feature, so has no MeshGeometry.
if "MeshGeometry" in globals():
    from yamc._tets import add_tets as _add_tets  # noqa: E402

    globals()["MeshGeometry"].to_vtkhdf = _mesh_geometry_to_vtkhdf
    globals()["MeshGeometry"].add_tets = _add_tets

    from yamc._tets import unstructured_mesh_to_vtkhdf as _um_to_vtkhdf  # noqa: E402

    globals()["UnstructuredMesh"].to_vtkhdf = _um_to_vtkhdf
Tracks.to_vtkhdf = _tracks_to_vtkhdf

# Browser-runnable HTML export -- defined in pure Python so it can build
# the wasm bundle + optional cross-section embedding from the templates
# that ship inside the wheel.
from yamc._export import to_html as _model_to_html  # noqa: E402

Model.to_html = _model_to_html


# ---------------------------------------------------------------------------
# Labelled results: TallyResult.to_xarray / to_dataset
# ---------------------------------------------------------------------------
# yamc knows the coordinate values for every tally axis (score names, nuclide
# names, energy bin edges, ...) at result-construction time. These helpers
# attach them to an xarray object so results are self-describing and
# n-dimensional-native -- the modern replacement for OpenMC's pandas
# MultiIndex, with no hard xarray dependency (imported lazily).
from yamc._core import TallyResult as _TallyResult  # noqa: E402


def _tally_axis_coords(tally, dims, shape):
    """Best-effort coordinate values per tally axis.

    Attaches a coordinate only when its length matches the axis size; axes
    without retrievable coordinate values (e.g. mesh voxels) are left as plain
    integer indices, which is still a valid xarray.
    """
    coords = {}
    for dim, size in zip(dims, shape):
        values = None
        if dim == "score":
            values = [str(s) for s in (tally.scores or [])]
        elif dim == "nuclide":
            values = list(tally.nuclides or [])
        elif dim == "parent_nuclide":
            values = list(tally.parent_nuclides or [])
        elif dim == "energy":
            edges = tally.energy_bins
            if edges is not None and len(edges) == size + 1:
                # one coordinate per bin -- label by the bin's lower edge (eV)
                values = list(edges[:-1])
        if values is not None and len(values) == size:
            coords[dim] = values
    return coords


def _units_attrs(result):
    """The ``units`` attribute (CF convention) when every score shares one,
    plus the source strength the values carry."""
    units = result.units
    attrs = {"units": units[0]} if len(set(units)) == 1 else {}
    if result.source_strength is not None:
        attrs["source_strength"], attrs["source_strength_per"] = result.source_strength
    return attrs


def _with_score_units(obj, result):
    """Scores with different units get them as a ``units`` coordinate on the
    ``score`` axis, since one ``units`` attribute cannot hold them. The
    energy axis is labelled in eV."""
    units = result.units
    if len(set(units)) > 1 and "score" in obj.dims:
        obj = obj.assign_coords(units=("score", units))
    if "energy" in obj.coords:
        obj["energy"].attrs["units"] = "eV"
    return obj


def _tallyresult_to_xarray(self, field="mean"):
    """Return one tally field as a labelled ``xarray.DataArray``.

    Dimensions are named from ``dim_labels`` and carry real coordinate values
    (score names, nuclide names, energy-bin lower edges in eV) wherever yamc
    knows them. ``attrs["units"]`` holds the unit of the values (see
    ``TallyResult.units``), and ``attrs["source_strength"]`` the source
    strength when one is set. Requires xarray (``pip install yamc[xarray]``).

    Args:
        field (str): which field -- ``"mean"`` (default), ``"std_dev"``,
            ``"rel_err"``, ``"variance"``, or ``"count"``.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - only without xarray installed
        raise ImportError(
            "to_xarray() requires xarray. Install with `pip install xarray` "
            "(or `pip install yamc[xarray]`)."
        ) from exc
    arr = self.to_numpy(field)
    dims = list(self.dim_labels)
    coords = _tally_axis_coords(self.tally, dims, self.shape)
    attrs = _units_attrs(self)
    if field in ("rel_err", "count"):
        attrs["units"] = "1"
    elif field == "variance" and "units" in attrs:
        attrs["units"] = f"({attrs['units']})2"
    da = xr.DataArray(arr, dims=dims, coords=coords, name=self.tally.name or field, attrs=attrs)
    if field not in ("mean", "std_dev"):
        if "energy" in da.coords:
            da["energy"].attrs["units"] = "eV"
        return da
    return _with_score_units(da, self)


def _tallyresult_to_dataset(self):
    """Return ``mean``, ``std_dev`` and ``rel_err`` as one labelled
    ``xarray.Dataset`` sharing the tally's named, coordinate-bearing axes.

    ``mean`` and ``std_dev`` carry ``attrs["units"]`` (see
    ``TallyResult.units``), ``rel_err`` is ``"1"``, and the dataset's attrs
    hold the source strength when one is set. Requires xarray
    (``pip install yamc[xarray]``).
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - only without xarray installed
        raise ImportError(
            "to_dataset() requires xarray. Install with `pip install xarray` "
            "(or `pip install yamc[xarray]`)."
        ) from exc
    dims = list(self.dim_labels)
    coords = _tally_axis_coords(self.tally, dims, self.shape)
    attrs = _units_attrs(self)
    units = {"units": attrs.pop("units")} if "units" in attrs else {}
    data_vars = {
        "mean": (dims, self.to_numpy("mean"), units),
        "std_dev": (dims, self.to_numpy("std_dev"), units),
        "rel_err": (dims, self.to_numpy("rel_err"), {"units": "1"}),
    }
    return _with_score_units(xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs), self)


_TallyResult.to_xarray = _tallyresult_to_xarray
_TallyResult.to_dataset = _tallyresult_to_dataset


# ---------------------------------------------------------------------------
# Typed signatures for Jupyter (?/shift-tab)
# ---------------------------------------------------------------------------
# PyO3 exposes ``__text_signature__`` (parameter names + defaults) but no type
# annotations, so the signature line shown by Jupyter's ``?`` / shift-tab is
# untyped. Overlay the annotations from the shipped ``_core`` type stub onto
# each class's existing (correctly structured) signature and attach the result
# as ``Cls.__signature__`` so the signature line carries real types.
#
# Best-effort and fully guarded: tab-completion already reads the stub via
# Jedi and ``?`` already shows the docstring -- this only upgrades the
# signature line. Compiled methods (``method_descriptor``, no ``__dict__``) and
# module-level functions (``builtin_function_or_method``) cannot take an
# injected ``__signature__`` and are intentionally left untouched.

def _inject_signatures():
    try:
        import ast
        import inspect
        from pathlib import Path

        import yamc._core as _core
    except Exception:
        return

    stub = Path(__file__).parent / "_core" / "__init__.pyi"
    try:
        tree = ast.parse(stub.read_text())
    except Exception:
        return

    class _Ann(str):
        # Render the annotation text verbatim, without the surrounding quotes a
        # plain ``str`` annotation would get when inspect formats the signature.
        __slots__ = ()

        def __repr__(self):
            return str(self)

    _qualifier_modules = {"builtins", "typing", "yamc"}

    class _StripQualifiers(ast.NodeTransformer):
        # Rewrite ``builtins.str`` / ``typing.Optional`` / ``yamc.Enriched`` to
        # the bare name so the signature line matches the docs. Operating on the
        # AST (rather than string replacement) means a string inside a
        # ``Literal[...]`` is never accidentally rewritten.
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in _qualifier_modules:
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return node

    def _annotation(node):
        return _Ann(ast.unparse(_StripQualifiers().visit(node)))

    for cls_node in tree.body:
        if not isinstance(cls_node, ast.ClassDef):
            continue
        target = getattr(_core, cls_node.name, None)
        if target is None:
            continue
        ctor = next(
            (
                f
                for f in cls_node.body
                if isinstance(f, ast.FunctionDef) and f.name in ("__new__", "__init__")
            ),
            None,
        )
        if ctor is None:
            continue
        args = ctor.args
        annotations = {
            a.arg: _annotation(a.annotation)
            for a in (
                *args.posonlyargs,
                *args.args,
                args.vararg,
                *args.kwonlyargs,
                args.kwarg,
            )
            if a is not None and a.annotation is not None
        }
        if not annotations:
            continue
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            continue
        params = [
            p.replace(annotation=annotations[p.name]) if p.name in annotations else p
            for p in sig.parameters.values()
        ]
        try:
            target.__signature__ = sig.replace(parameters=params)
        except Exception:
            continue


_inject_signatures()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Define ``__all__`` so ``from yamc import *`` and tab-completion expose only
# the public surface: everything re-exported from the Rust core (`yamc._core`)
# plus the helpers defined here, minus the stdlib modules imported for internal
# use (``sys``/``types``/``dataclass``) and the private ``_``-prefixed config
# aliases. Computed (rather than hand-listed) so it tracks new ``_core``
# exports automatically. ``dir()`` is captured at module scope first because a

# comprehension has its own scope in Python 3.
_public_names = dir()
__all__ = sorted(
    name
    for name in _public_names
    if not name.startswith("_") and name not in {"sys", "types", "dataclass"}
)
del _public_names
