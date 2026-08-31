import sys as _sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

# The readers import stdlib and pyarrow and nothing else, so they are eager.
from .neutron_reader import read_neutron_from_arrow
from .photon_reader import read_photon_from_arrow

try:
    __version__ = version("nuclear_data_to_arrow")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Writers and verification, resolved on first attribute access rather than at
# import (issue #525).
#
# Every one of these modules does `from endf... import ...` at module level, and
# `photon_writer` reaches for `endf.incident_photon`, which PyPI's `endf` does
# not have: the import fails outright rather than at first use. Importing them
# eagerly therefore made the whole package, INCLUDING the two readers above and
# `schemas`, unimportable without the fork. Reading an Arrow directory back
# parses no ENDF and has no use for it, and on #524 that coupling stopped pytest
# COLLECTING `tests/test_schema_manifest.py`, a test that touches nothing but
# `manifest.json` and pyarrow.
#
# name -> module it comes from. Lazy, not optional: nothing here is any less
# supported than it was, it is simply not paid for until asked for.
_LAZY = {
    "export_neutron_to_arrow": ".neutron_writer",
    "export_photon_to_arrow": ".photon_writer",
    "export_transmutation_to_arrow": ".transmutation_writer",
    "verify_neutron": ".verify",
    "verify_photon": ".verify",
}


class MissingEndfFork(ImportError):
    """The endf reader the writers are written against is not installed.

    Raised by the `convert_*` entry points. The lazy ATTRIBUTES raise an
    `AttributeError` carrying the same message instead, with one of these as
    its `__cause__`, because a module `__getattr__` that raises anything else
    breaks every tool that walks a module: `help()`, `pydoc`,
    `inspect.getmembers`, Sphinx autodoc and IPython completion all getattr
    everything `__dir__` advertises and expect AttributeError for what is not
    there, so an ImportError from there makes `help(nuclear_data_to_arrow)`
    print this message and zero lines of documentation.

    The two cannot be one exception: ImportError and AttributeError have
    conflicting C-level layouts, so a class inheriting both is a TypeError at
    definition time.
    """


def _endf_message(what, exc):
    """What to say when `endf` is absent, or is the wrong `endf`."""
    return (
        f"{what} needs the endf reader this converter is written against, "
        "which is the local-develop branch of "
        "https://github.com/shimwell/endf-python and not PyPI's `endf`:\n"
        "    pip install 'endf @ git+https://github.com/shimwell/"
        "endf-python@local-develop'\n"
        f"The underlying import failed with: {exc}"
    )


def _require_endf():
    """Import `endf`, or say which one is wanted and how to install it.

    The `convert_*` functions below import endf directly rather than through a
    lazy attribute, so without this they raise a bare
    `ModuleNotFoundError: No module named 'endf'` and the caller is back to
    guessing a git URL, which is the failure this package set out to stop
    handing people.
    """
    try:
        import endf
    except ImportError as exc:
        raise MissingEndfFork(_endf_message("Converting", exc)) from exc
    return endf


def __getattr__(name):
    """PEP 562 lazy import for the endf-dependent half of the package."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        value = getattr(importlib.import_module(module, __name__), name)
    except ImportError as exc:
        # Say which package, because the failure names `endf.incident_photon`
        # and the fix is a git URL nobody guesses.
        # AttributeError, not ImportError: this is a module `__getattr__` and
        # every module-walking tool expects that type for a name it cannot
        # get. The message says what really happened and the ImportError is
        # the `__cause__`, so nothing is hidden.
        raise AttributeError(_endf_message(name, exc)) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


def _writer(name):
    """A lazy writer, resolved through this module's own attribute lookup.

    Not a global read: a module-level `__getattr__` is consulted for ATTRIBUTE
    access on the module, not for global name lookup inside a function defined
    in it, so `export_neutron_to_arrow(...)` in a body below would raise
    NameError however the lazy table is written.

    Not a direct `from .neutron_writer import ...` either: the writers are the
    seam the tests substitute (`monkeypatch.setattr(nda,
    "export_neutron_to_arrow", ...)`), and importing the submodule steps past
    whatever is set on the module.
    """
    return getattr(_sys.modules[__name__], name)


def convert_neutron(input_path, output_dir, *, source_format="endf",
                    temperatures=None, library="", data_version="",
                    njoy_exec="njoy"):
    """Convert an ENDF or ACE neutron file to simulation-ready Arrow format.

    Parameters
    ----------
    input_path : str or Path
        Path to an ENDF or ACE neutron file.
    output_dir : str or Path
        Directory to write the Arrow output into.
        A subdirectory named ``{nuclide}.arrow/`` will be created.
    source_format : {"endf", "ace"}
        Input file format. ``"endf"`` runs NJOY internally to reconstruct the
        resonances and Doppler broaden to each requested temperature, and is
        preferred: it is the only route that yields MT 901 (heating-local),
        which is built from two HEATR passes plus the MF=1/458 fission energy
        release and so cannot be recovered from an ACE table. ``"ace"`` reads a
        pre-processed table, fixing the temperature at whatever it was built
        with, and is useful when NJOY is unavailable. Note that ``"endf"`` runs
        through the ACE reader internally, since NJOY's output is ACE.
    temperatures : list of float, optional
        Temperatures in Kelvin (only used with ``source_format="endf"``).
    library : str, optional
        Library name (e.g., "endfb-8.0", "fendl-3.2c").
    data_version : str, optional
        Identifier of the published release this output belongs to, stamped
        into the completion marker. yamc compares a cached copy against it to
        decide whether a re-published library has invalidated the cache; see
        ``completion.write_completion_marker``.
    njoy_exec : str, optional
        NJOY executable to run, only used with ``source_format="endf"``.
        Defaults to ``"njoy"``, resolved on PATH.

        Which NJOY this is matters. FENDL is processed by the IAEA-NDS fork
        (https://github.com/IAEA-NDS/NJOY2016) with local modifications, and
        upstream NJOY does not reproduce the official release: measured on
        FENDL 3.2d against LANL 2016.78, the fork changes URR probability
        tables by up to 100% (La139) and MT 301 heating / MT 444 damage by up
        to 77% below roughly 1 keV. Both builds succeed, so the wrong choice
        is quiet. Pass the NJOY the library was evaluated with.

    Returns
    -------
    Path
        Path to the created Arrow directory.
    """
    endf = _require_endf()

    export_neutron_to_arrow = _writer("export_neutron_to_arrow")

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if source_format == "ace":
        if temperatures is not None:
            raise ValueError(
                "temperatures cannot be set with source_format='ace': an ACE "
                "table carries the temperature it was processed at. Use "
                "source_format='endf' to Doppler broaden to chosen temperatures."
            )
        data = endf.IncidentNeutron.from_ace(input_path)
    elif source_format == "endf":
        kwargs = {"njoy_exec": njoy_exec}
        if temperatures is not None:
            kwargs["temperatures"] = temperatures
        data = endf.IncidentNeutron.from_njoy(input_path, **kwargs)
    else:
        raise ValueError(f"Unknown source_format: {source_format!r}")

    arrow_path = output_dir / f"{data.name}.arrow"
    export_neutron_to_arrow(data, arrow_path, library=library,
                            data_version=data_version)
    return arrow_path


def convert_photon(input_path, output_dir, *, atom_path=None, library="",
                   data_version=""):
    """Convert an ENDF photon file to simulation-ready Arrow format.

    Parameters
    ----------
    input_path : str or Path
        Path to a photoatomic ENDF file.
    output_dir : str or Path
        Directory to write the Arrow output into.
        A subdirectory named ``{element}.arrow/`` will be created.
    atom_path : str or Path, optional
        Path to an atomic relaxation ENDF file.
    library : str, optional
        Library name (e.g., "endfb-8.0", "fendl-3.2c").
    data_version : str, optional
        Identifier of the published release this output belongs to, stamped
        into the completion marker. yamc compares a cached copy against it to
        decide whether a re-published library has invalidated the cache; see
        ``completion.write_completion_marker``.

    Returns
    -------
    Path
        Path to the created Arrow directory.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    endf = _require_endf()

    export_photon_to_arrow = _writer("export_photon_to_arrow")

    if atom_path is not None:
        data = endf.IncidentPhoton.from_endf(input_path, atom_path)
    else:
        data = endf.IncidentPhoton.from_endf(input_path)

    arrow_path = output_dir / f"{data.name}.arrow"
    export_photon_to_arrow(data, arrow_path, library=library,
                           data_version=data_version)
    return arrow_path


def convert_photon_endf(input_path, output_dir, *, library="", data_version=""):
    """Convert a multi-evaluation ENDF photon file to Arrow format.

    Some libraries (e.g. FENDL) bundle multiple elements into a single
    ENDF file.  This function extracts each evaluation and writes a
    separate Arrow directory for each element.

    Parameters
    ----------
    input_path : str or Path
        Path to an ENDF file containing one or more photoatomic evaluations.
    output_dir : str or Path
        Directory to write the Arrow output into.
    library : str, optional
        Library name (e.g., "fendl-3.2c").
    data_version : str, optional
        Identifier of the published release this output belongs to, stamped
        into the completion marker. yamc compares a cached copy against it to
        decide whether a re-published library has invalidated the cache; see
        ``completion.write_completion_marker``.

    Returns
    -------
    list of Path
        Paths to the created Arrow directories.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    endf = _require_endf()

    export_photon_to_arrow = _writer("export_photon_to_arrow")

    materials = endf.get_materials(input_path)
    paths = []
    for material in materials:
        data = endf.IncidentPhoton.from_endf(material)
        arrow_path = output_dir / f"{data.name}.arrow"
        export_photon_to_arrow(data, arrow_path, library=library,
                               data_version=data_version)
        paths.append(arrow_path)
    return paths


def convert_transmutation(output_path, *, decay_files=None, fpy_files=None,
                          neutron_files=None, branch_ratios=None,
                          library="", data_version="", subsections=None):
    """Convert a transmutation network to simulation-ready Arrow format.

    The network is built from the ENDF decay, fission-yield and neutron
    sub-libraries. The full reaction set is requested
    (``reactions=list(endf.chain.REACTIONS.keys())``) rather than the default
    subset.

    The output is written as a set of independently library-sourced
    subsections (``decay/``, ``reactions/``, ``fission_yields/``,
    ``branching/``) plus a top-level ``manifest.json``; see
    :func:`export_transmutation_to_arrow`.

    ``branching/`` is emitted from the same ``neutron_files``, for the chain's
    own reaction parents. It is not optional, because a chain without it is
    quietly wrong for any material whose activity comes from an isomer: the
    fraction of an (n,2n) that leaves the product metastable is energy
    dependent, and substituting a scalar branching table for it puts FNS
    decay heat 6x low on niobium. Pass ``subsections`` to emit less, but know
    what a partial chain does and does not say.

    Parameters
    ----------
    output_path : str or Path
        Target ``transmutation_{library}.arrow/`` root directory.
    decay_files, fpy_files, neutron_files : list of path-like
        ENDF inputs used to build the network. All three are required.
        ``decay_files`` is also the isomer table branching is mapped through;
        only the metastable evaluations in it are read for that.
    branch_ratios : str or Path, optional
        Path to a JSON file of branching ratios in ``openmc_data`` format
        (``{reaction: {parent: {target: ratio}}}``). Applied after the network
        is built/loaded.
    library : str, optional
        Library name (e.g., "endfb-8.0").
    data_version : str, optional
        Identifier of the published release this output belongs to, stamped
        into the completion marker. yamc compares a cached copy against it to
        decide whether a re-published library has invalidated the cache; see
        ``completion.write_completion_marker``.
    subsections : iterable of str, optional
        Which subsections to emit (defaults to all: decay, reactions,
        fission_yields, branching).

    Returns
    -------
    Path
        Path to the created ``transmutation_{library}.arrow/`` directory.
    """
    import fnmatch
    import json

    _require_endf()
    from endf.chain import Chain, REACTIONS

    export_transmutation_to_arrow = _writer("export_transmutation_to_arrow")

    output_path = Path(output_path)

    if decay_files is None or fpy_files is None or neutron_files is None:
        raise ValueError(
            "Building the network requires decay_files, fpy_files and neutron_files."
        )
    chain = Chain.from_endf(
        decay_files=list(decay_files),
        fpy_files=list(fpy_files),
        neutron_files=list(neutron_files),
        reactions=list(REACTIONS.keys()),
    )

    if branch_ratios is not None:
        with open(branch_ratios) as fh:
            all_ratios = json.load(fh)
        for reaction, ratios in all_ratios.items():
            chain.set_branch_ratios(
                branch_ratios=ratios, reaction=reaction, strict=False
            )

    export_transmutation_to_arrow(
        chain, output_path,
        library=library,
        data_version=data_version,
        source="endf",
        branch_ratios_applied=branch_ratios is not None,
        subsections=subsections,
    )

    if subsections is None or "branching" in subsections:
        # branching_extractor imports endf at module level, so it gets the
        # same guidance rather than a bare ModuleNotFoundError.
        _require_endf()
        from .branching_extractor import extract_branching, export_branching_to_arrow

        # Scoped to this chain's own reaction parents, so the subsection
        # describes the network it ships beside. Only the metastable decay
        # evaluations carry the level energies the isomer table is built from,
        # and matching them here keeps the isomer_table call off the other
        # ~3000 files.
        parents = [n.name for n in chain.nuclides if n.reactions]
        isomer_sources = [
            p for p in decay_files
            if fnmatch.fnmatch(Path(p).name, "dec-*m[0-9].endf")
        ] or list(decay_files)
        rows, _ = extract_branching(list(neutron_files), isomer_sources, parents)
        export_branching_to_arrow(rows, output_path, library=library,
                                  decay_library=library,
                                  data_version=data_version)

    return output_path


def convert_branching(output_path, *, neutron_dir, nuclides, decay_dir=None,
                      decay_files=None, library="tendl-2025", decay_library="",
                      data_version="", tol_eV=3000.0, linearize_tol=1e-3):
    """Extract isomeric branching from TENDL neutron files and write a
    ``branching/`` subsection.

    Reads MF=8/9/10 radionuclide production from each parent's TENDL neutron
    file, maps product levels to metastable states using decay data, and writes
    the verbatim energy-dependent yields/cross sections.

    Parameters
    ----------
    output_path : str or Path
        Target ``transmutation_{library}.arrow/`` root (``branching/`` and a
        manifest are written under it).
    neutron_dir : str or Path
        Directory of neutron ENDF files (TENDL ``n-*.tendl``, ENDF/B
        ``n-*.endf`` or JEFF ``n_*.jeff``; all three naming conventions are
        recognized).
    nuclides : iterable of str
        Parent nuclide GNDS names to process (e.g. the reaction parents in the
        chain).
    decay_dir : str or Path, optional
        Directory of decay ENDF files; metastable files (``dec-*m[0-9].endf``)
        are used to build the isomer table. Provide this or ``decay_files``.
    decay_files : list of path-like, optional
        Explicit decay files (overrides ``decay_dir``).
    library : str, optional
        Branching source library name (default "tendl-2025").
    data_version : str, optional
        Identifier of the published release this output belongs to, stamped
        into the completion marker. yamc compares a cached copy against it to
        decide whether a re-published library has invalidated the cache; see
        ``completion.write_completion_marker``.
    decay_library : str, optional
        Library the decay data (isomer mapping) came from, recorded in
        provenance.
    tol_eV : float, optional
        Energy-match tolerance for level -> isomer assignment.
    linearize_tol : float, optional
        Relative tolerance for resampling non-lin-lin ENDF interpolation
        regions onto the lin-lin pairs the arrow format stores.

    Returns
    -------
    (Path, dict)
        The created directory and an extraction ``stats`` dict.
    """
    # branching_extractor imports endf at module level, so it gets the same
    # guidance rather than a bare ModuleNotFoundError.
    _require_endf()
    from .branching_extractor import extract_branching, export_branching_to_arrow

    output_path = Path(output_path)
    if decay_files is None:
        if decay_dir is None:
            raise ValueError("Provide either decay_files or decay_dir.")
        decay_files = sorted(Path(decay_dir).glob("dec-*m[0-9].endf"))
    rows, stats = extract_branching(neutron_dir, decay_files, list(nuclides),
                                    tol_eV=tol_eV, linearize_tol=linearize_tol)
    export_branching_to_arrow(rows, output_path, library=library,
                              decay_library=decay_library,
                              data_version=data_version, tol_eV=tol_eV,
                              linearize_tol=linearize_tol)
    return output_path, stats


# The public surface. Several names are re-exports or lazy attributes rather
# than uses, so this is what makes them intentional rather than stray imports.
#
# It used to be assigned twice, here and at the top of the file, and this one
# won. It named `split_endf_tape` and `download_jeff_sublibrary`, which the
# package defines nowhere, so `from nuclear_data_to_arrow import *` raised
# AttributeError.
__all__ = [
    "__version__",
    "MissingEndfFork",
    # conversion entry points, defined above
    "convert_neutron",
    "convert_photon",
    "convert_photon_endf",
    "convert_transmutation",
    "convert_branching",
    # writers (lazy, see `_LAZY`)
    "export_neutron_to_arrow",
    "export_photon_to_arrow",
    "export_transmutation_to_arrow",
    # readers and verification
    "read_neutron_from_arrow",
    "read_photon_from_arrow",
    "verify_neutron",
    "verify_photon",
]
