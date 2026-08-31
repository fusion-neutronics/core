"""yamc -- Yet Another Monte Carlo code."""

import pathlib as _pathlib
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
    __version__ = _pkg_version("yamc-core")
except _PackageNotFoundError:  # pragma: no cover - only without an installed dist
    __version__ = "0.0.0+unknown"

from yamc._core import *  # noqa: F401,F403

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
    Region as Region,
    Tally as Tally,
    get_transmutation_decay_data as _get_transmutation_decay_data,
    get_transmutation_reactions as _get_transmutation_reactions,
    get_transmutation_fission_yields as _get_transmutation_fission_yields,
    get_transmutation_branch_ratios as _get_transmutation_branch_ratios,
    get_cross_section_data as _get_cross_section_data,
    lookup_cross_section_data as lookup_cross_section_data,  # noqa: F401  (public helper)
    set_transmutation_decay_data as _set_transmutation_decay_data,
    set_transmutation_reactions as _set_transmutation_reactions,
    set_transmutation_fission_yields as _set_transmutation_fission_yields,
    set_transmutation_branch_ratios as _set_transmutation_branch_ratios,
    set_cross_section_data as _set_cross_section_data,
    set_cross_section_data_entry as set_cross_section_data_entry,  # noqa: F401  (public helper)
)

# The dotted paths (`from yamc.sources import Discrete`) are registered by the
# extension itself, next to the `yamc._core.<name>` registration it has always
# done, so a new submodule cannot arrive with only one of the two names.


# ---------------------------------------------------------------------------
# Module-level properties for transport and transmutation data
# ---------------------------------------------------------------------------

class _YamcModule(types.ModuleType):
    """Module subclass exposing cross_section_data and the per-subsection
    transmutation_* sources as properties."""

    @property
    def cross_section_data(self):
        return _get_cross_section_data()

    @cross_section_data.setter
    def cross_section_data(self, value):
        _set_cross_section_data(value)

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
            ``yamc.vtkhdf.geometry_to_vtkhdf`` (``resolution``, ``datasets``).
    """
    from yamc.vtkhdf import geometry_to_vtkhdf
    geometry_to_vtkhdf(filename, self, **kwargs)


def _cell_to_vtkhdf(self, filename, **kwargs):
    """Write cell to a VTK-HDF ImageData file with material and cell IDs.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.cell_to_vtkhdf`` (``resolution``, ``datasets``).
    """
    from yamc.vtkhdf import cell_to_vtkhdf
    cell_to_vtkhdf(filename, self, **kwargs)


def _region_to_vtkhdf(self, filename, *, resolution=10000):
    """Write region to a VTK-HDF ImageData file.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended: ``.vtkhdf``).
        resolution (int): Approximate total number of voxels (default 10 000).
    """
    from yamc.vtkhdf import region_to_vtkhdf
    region_to_vtkhdf(filename, self, resolution=resolution)


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
            ``datasets``, ``samples``).
    """
    from yamc.vtkhdf import model_to_vtkhdf
    model_to_vtkhdf(filename, self, **kwargs)


def _source_to_vtkhdf(self, filename, **kwargs):
    """Write sampled source points to a VTK-HDF file for ParaView visualization.

    Requires ``h5py`` (install with ``pip install yamc[viz]``).

    Args:
        filename (str): Output file path (recommended extension: ``.vtkhdf``)
        **kwargs: Keyword arguments forwarded to
            ``yamc.vtkhdf.source_to_vtkhdf`` (``samples``).
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
            ``scaling_factor``).
    """
    from yamc.vtkhdf import mesh_tally_to_vtkhdf
    mesh_tally_to_vtkhdf(filename, self, **kwargs)


Geometry.to_vtkhdf = _geometry_to_vtkhdf
Cell.to_vtkhdf = _cell_to_vtkhdf
Region.to_vtkhdf = _region_to_vtkhdf
Halfspace.to_vtkhdf = _region_to_vtkhdf
Model.to_vtkhdf = _model_to_vtkhdf
NeutronSource.to_vtkhdf = _source_to_vtkhdf
PhotonSource.to_vtkhdf = _source_to_vtkhdf
Tally.to_vtkhdf = _tally_to_vtkhdf

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


def _tallyresult_to_xarray(self, field="mean"):
    """Return one tally field as a labelled ``xarray.DataArray``.

    Dimensions are named from ``dim_labels`` and carry real coordinate values
    (score names, nuclide names, energy-bin lower edges in eV) wherever yamc
    knows them. Requires xarray (``pip install yamc[xarray]``).

    Args:
        field (str): which field -- ``"mean"`` (default),
            ``"standard_deviation"``, ``"relative_error"``, ``"variance"``,
            or ``"total_count"``.
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
    return xr.DataArray(arr, dims=dims, coords=coords, name=self.tally.name or field)


def _tallyresult_to_dataset(self):
    """Return mean, standard deviation and relative error as one labelled
    ``xarray.Dataset`` sharing the tally's named, coordinate-bearing axes.

    Requires xarray (``pip install yamc[xarray]``).
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
    data_vars = {
        "mean": (dims, self.to_numpy("mean")),
        "std_dev": (dims, self.to_numpy("standard_deviation")),
        "relative_error": (dims, self.to_numpy("relative_error")),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


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

# The auxiliary photon tabulations this package ships, so a photon conversion
# works without the caller locating three text files.
#
# None of this data is in ENDF: the photoatomic sublibrary carries no Compton
# profiles, no bremsstrahlung cross sections and no density effect correction,
# so a transport code takes them from separate published tabulations. yamc
# ships them; yani does not, because transmutation is driven by the neutron
# flux and never needs them. That is why this default lives here and not in the
# Rust binding, which both wheels share.
_PHOTON_DATA = _pathlib.Path(__file__).parent / "data"

_convert_photon_core = convert_photon  # noqa: F405  (from _core)


def convert_photon(
    photoatomic_path,
    output_dir,
    relaxation_path=None,
    compton_profiles=None,
    density_effect=None,
    bremsstrahlung=None,
    library="",
    data_version="",
    created_utc=None,
):
    """Convert a photoatomic evaluation into the per-element photon sections.

    Writes ``element.arrow`` always, and ``subshells.arrow``, ``compton.arrow``
    and ``bremsstrahlung.arrow`` where there is data for them. No NJOY: a
    photoatomic evaluation is already pointwise.

    The three auxiliary tabulations default to the copies shipped in this
    package, so the common case is two arguments. Pass them explicitly to use a
    different release; they are read as a set, so give all three or none.

    Parameters
    ----------
    photoatomic_path : str or Path
        The photoatomic evaluation. One file may hold several elements, and
        each is written to its own directory.
    output_dir : str or Path
        Directory to write ``{Element}.arrow/`` into.
    relaxation_path : str or Path, optional
        The atomic relaxation evaluation for the same element, carrying binding
        energies, occupancies and the transition cascade. Without it the
        subshells are written with zero binding energy and no transitions,
        which is a fluorescence-free atom rather than an error.
    compton_profiles, density_effect, bremsstrahlung : str or Path, optional
        Override the bundled tabulations.
    library, data_version, created_utc
        Recorded in ``version.json``.

    Returns
    -------
    list[str]
        One path per element written.
    """
    given = (compton_profiles, density_effect, bremsstrahlung)
    if any(g is None for g in given) and any(g is not None for g in given):
        raise ValueError(
            "compton_profiles, density_effect and bremsstrahlung are read as one "
            "set: give all three or none. A partial set would silently drop a "
            "section rather than fail."
        )
    if compton_profiles is None:
        compton_profiles = _PHOTON_DATA / "compton_profiles_biggs1975.txt"
        density_effect = _PHOTON_DATA / "density_effect_sternheimer1982.txt"
        bremsstrahlung = _PHOTON_DATA / "bremsstrahlung_seltzer_berger1986.txt"
        missing = [p.name for p in (compton_profiles, density_effect, bremsstrahlung)
                   if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"the bundled photon tabulations are missing from the installed "
                f"package ({', '.join(missing)} under {_PHOTON_DATA}). Pass the "
                f"three paths explicitly, or reinstall."
            )
    return _convert_photon_core(
        str(photoatomic_path),
        str(output_dir),
        relaxation_path=None if relaxation_path is None else str(relaxation_path),
        compton_profiles=str(compton_profiles),
        density_effect=str(density_effect),
        bremsstrahlung=str(bremsstrahlung),
        library=library,
        data_version=data_version,
        created_utc=created_utc,
    )


# comprehension has its own scope in Python 3.
_public_names = dir()
__all__ = sorted(
    name
    for name in _public_names
    if not name.startswith("_") and name not in {"sys", "types", "dataclass"}
)
del _public_names
