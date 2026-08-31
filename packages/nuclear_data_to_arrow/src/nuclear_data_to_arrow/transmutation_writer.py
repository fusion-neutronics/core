"""Export an endf-python Chain to a simulation-ready transmutation directory.

The network is written as a set of independently library-sourced *subsections*,
each in its own subdirectory with a self-describing ``provenance.json``:

- ``decay/`` (nuclides, decay modes, decay photon/electron sources)
- ``reactions/`` (transmutation reaction topology + Q)
- ``fission_yields/`` (fission product yields + inheritance aliases)
- ``branching/`` (energy-dependent isomeric branching, written by
  :mod:`branching_extractor` rather than this module)

A top-level ``manifest.json`` lists the subsections produced by a run, merged
across writers.  See issue #10 for the full design.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from endf.univariate import Discrete, Tabular, Mixture

from .schemas import (
    DECAY_NUCLIDES_SCHEMA,
    DECAY_MODES_SCHEMA,
    DECAY_SOURCES_SCHEMA,
    TRANSMUTATION_REACTIONS_SCHEMA,
    FISSION_YIELDS_SCHEMA,
    FISSION_YIELD_ALIASES_SCHEMA,
)

# Subsections *this writer* can emit.  ``branching`` is a subsection of the same
# format but comes out of branching_extractor, which reads MF=8/9/10 from the
# neutron evaluations; convert_transmutation calls both.
SUBSECTIONS = ("decay", "reactions", "fission_yields")

FORMAT_VERSION = 2


# Option-D hosting format: LZ4-compressed Arrow IPC (pure-Rust decode in yamc).
# Transmutation sections stay single-batch (unlike neutron reactions.arrow).
_LZ4 = ipc.IpcWriteOptions(compression="lz4")


def _write_arrow_ipc(table, filepath):
    with pa.OSFile(str(filepath), 'wb') as f:
        writer = ipc.new_file(f, table.schema, options=_LZ4)
        writer.write_table(table)
        writer.close()


def _table(rows, schema):
    return pa.table(
        {col: [r[col] for r in rows] for col in schema.names},
        schema=schema,
    )


def _write_json(obj, filepath):
    Path(filepath).write_text(json.dumps(obj, indent=2))


def _write_manifest(path, library, emitted, created_utc):
    """Record *emitted* in the chain's manifest, keeping what is already there.

    A chain is assembled by more than one writer, and no writer sees all of it:
    :func:`export_transmutation_to_arrow` emits decay, reactions and
    fission_yields, :func:`export_branching_to_arrow` emits branching, and a
    library calls whichever subsections it can supply. Each wrote the manifest
    from its own results alone and overwrote the file, so the writer that
    happened to run last decided what the chain claimed to contain:
    ENDF/B-VIII.1 built all four subsections and listed only branching,
    TENDL-2017 built two and listed only reactions. Every subsection was
    present on disk and correct; only the index of them was wrong.

    Merging rather than overwriting is what makes the manifest describe the
    directory instead of the last call. Two cases are deliberately not merged:
    a directory rebuilt for a different library starts over rather than
    accumulating another library's subsections, and an entry whose directory
    has gone is dropped rather than left advertising something removed.
    """
    from . import __version__

    path = Path(path)
    manifest_path = path / "manifest.json"

    subsections = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            # An unreadable manifest is rebuilt from what is on disk rather
            # than aborting a chain that is otherwise written and valid.
            existing = {}
        if existing.get("library") == library:
            subsections = dict(existing.get("subsections") or {})

    subsections.update(emitted)
    subsections = {
        name: entry for name, entry in subsections.items()
        if (path / entry.get("path", name)).is_dir()
    }

    _write_json({
        "format_version": FORMAT_VERSION,
        "library": library,
        "converter_version": __version__,
        "created_utc": created_utc,
        "subsections": dict(sorted(subsections.items())),
    }, manifest_path)


def _source_rows(nuclide_name, particle, dist):
    """Yield (type, energies, intensities) tuples from a source distribution.

    Mixture distributions are flattened into one row per component, with the
    component probability multiplied into its intensities.
    """
    if isinstance(dist, Discrete):
        yield (
            "discrete",
            np.asarray(dist.x, dtype=np.float64).tolist(),
            np.asarray(dist.p, dtype=np.float64).tolist(),
        )
    elif isinstance(dist, Tabular):
        yield (
            "tabular",
            np.asarray(dist.x, dtype=np.float64).tolist(),
            np.asarray(dist.p, dtype=np.float64).tolist(),
        )
    elif isinstance(dist, Mixture):
        for prob, sub in zip(dist.probability, dist.distribution):
            for sub_type, energies, intensities in _source_rows(
                nuclide_name, particle, sub
            ):
                scaled = (np.asarray(intensities, dtype=np.float64) * float(prob)).tolist()
                yield (sub_type, energies, scaled)
    else:
        raise NotImplementedError(
            f"{nuclide_name} source ({particle}): unsupported distribution "
            f"type {type(dist).__name__}"
        )


def export_transmutation_to_arrow(chain, path, *, library="", data_version="",
                                  source="", branch_ratios_applied=False,
                                  subsections=None):
    """Export an endf-python Chain to a split transmutation directory.

    Parameters
    ----------
    chain : endf.Chain
        The transmutation network to export.
    path : str or Path
        Output root directory (e.g., "transmutation_endfb-8.1.arrow"). One
        subdirectory is written per emitted subsection, plus a manifest.
    library : str, optional
        Library name (e.g., "endfb-8.1"). Recorded in each subsection's
        provenance and in the manifest.
    data_version : str, optional
        Identifier of the published release these subsections belong to,
        recorded in each subsection's provenance. yamc caches a transmutation
        subsection as its own directory and compares this to decide whether a
        re-published library has invalidated it (yamc issue #366).
    source : str, optional
        How the chain was built. Recorded in provenance; currently always
        "endf", since the ENDF sub-libraries are the only supported input.
    branch_ratios_applied : bool, optional
        Whether a branch-ratios override was applied to the reactions.
        Recorded in the reactions subsection provenance.
    subsections : iterable of str, optional
        Which subsections to emit. Defaults to all of ``SUBSECTIONS``.

    Returns
    -------
    Path
        Path to the created root directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    requested = SUBSECTIONS if subsections is None else tuple(subsections)
    unknown = [s for s in requested if s not in SUBSECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown subsection(s) {unknown}; valid subsections are {list(SUBSECTIONS)}."
        )

    from . import __version__
    created_utc = datetime.now(timezone.utc).isoformat()

    # Collect rows for every subsection in a single pass over the network, then
    # write only the requested ones.
    decay_nuclide_rows = []
    decay_mode_rows = []
    source_rows = []
    reaction_rows = []
    fy_rows = []
    fy_alias_rows = []

    for nuc in chain.nuclides:
        # ``half_life_uncertainty`` is optional on the source object: this
        # writer is fed by more than one producer, and one that predates the
        # column must keep working. Absent becomes null, never 0.0 -- an
        # unstated uncertainty and one measured to be negligible are different
        # claims downstream.
        half_life_uncertainty = getattr(nuc, "half_life_uncertainty", None)
        decay_energy_uncertainty = getattr(nuc, "decay_energy_uncertainty", None)
        decay_nuclide_rows.append({
            "name": nuc.name,
            "half_life": float(nuc.half_life) if nuc.half_life is not None else None,
            "decay_energy": float(nuc.decay_energy),
            "half_life_uncertainty": (
                float(half_life_uncertainty)
                if half_life_uncertainty is not None
                else None
            ),
            "decay_energy_uncertainty": (
                float(decay_energy_uncertainty)
                if decay_energy_uncertainty is not None
                else None
            ),
        })

        for d in nuc.decay_modes:
            decay_mode_rows.append({
                "nuclide": nuc.name,
                "type": d.type,
                "target": d.target,
                "branching_ratio": float(d.branching_ratio),
            })

        if nuc.sources:
            for particle, dist in nuc.sources.items():
                for src_type, energies, intensities in _source_rows(nuc.name, particle, dist):
                    source_rows.append({
                        "nuclide": nuc.name,
                        "particle": particle,
                        "type": src_type,
                        "energies": energies,
                        "intensities": intensities,
                    })

        for r in nuc.reactions:
            reaction_rows.append({
                "nuclide": nuc.name,
                "type": r.type,
                "target": r.target,
                "Q": float(r.Q),
                "branching_ratio": float(r.branching_ratio),
            })

        fy_parent = getattr(nuc, "_fpy", None)
        has_fy = nuc.yield_data is not None
        if fy_parent is not None:
            fy_alias_rows.append({
                "nuclide": nuc.name,
                "fission_yield_parent": fy_parent,
            })
        elif has_fy:
            for energy, fy in nuc.yield_data.items():
                fy_rows.append({
                    "nuclide": nuc.name,
                    "energy": float(energy),
                    "products": list(fy.products),
                    "yields": np.asarray(fy.yields, dtype=np.float64).tolist(),
                })

    def _base_provenance(subsection):
        return {
            "subsection": subsection,
            "library": library,
            "data_version": data_version,
            "source": source,
            "converter_version": __version__,
            "created_utc": created_utc,
        }

    emitted = {}

    if "decay" in requested:
        decay_dir = path / "decay"
        decay_dir.mkdir(parents=True, exist_ok=True)
        # nuclides.arrow is the primary index of the decay subsection and is
        # always written (one row per nuclide in the network).
        _write_arrow_ipc(_table(decay_nuclide_rows, DECAY_NUCLIDES_SCHEMA),
                         decay_dir / "nuclides.arrow")
        if decay_mode_rows:
            _write_arrow_ipc(_table(decay_mode_rows, DECAY_MODES_SCHEMA),
                             decay_dir / "decay_modes.arrow")
        if source_rows:
            _write_arrow_ipc(_table(source_rows, DECAY_SOURCES_SCHEMA),
                             decay_dir / "sources.arrow")
        _write_json(_base_provenance("decay"), decay_dir / "provenance.json")
        emitted["decay"] = {"path": "decay"}

    if "reactions" in requested:
        reactions_dir = path / "reactions"
        reactions_dir.mkdir(parents=True, exist_ok=True)
        _write_arrow_ipc(_table(reaction_rows, TRANSMUTATION_REACTIONS_SCHEMA),
                         reactions_dir / "reactions.arrow")
        prov = _base_provenance("reactions")
        prov["branch_ratios_applied"] = bool(branch_ratios_applied)
        _write_json(prov, reactions_dir / "provenance.json")
        emitted["reactions"] = {"path": "reactions"}

    if "fission_yields" in requested:
        fy_dir = path / "fission_yields"
        fy_dir.mkdir(parents=True, exist_ok=True)
        _write_arrow_ipc(_table(fy_rows, FISSION_YIELDS_SCHEMA),
                         fy_dir / "fission_yields.arrow")
        if fy_alias_rows:
            _write_arrow_ipc(_table(fy_alias_rows, FISSION_YIELD_ALIASES_SCHEMA),
                             fy_dir / "aliases.arrow")
        _write_json(_base_provenance("fission_yields"), fy_dir / "provenance.json")
        emitted["fission_yields"] = {"path": "fission_yields"}

    _write_manifest(path, library, emitted, created_utc)

    return path
