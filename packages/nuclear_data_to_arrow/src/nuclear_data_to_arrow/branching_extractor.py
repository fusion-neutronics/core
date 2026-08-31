"""Extract isomeric branching data from neutron ENDF files (MF=8/9/10).

The best isomeric branching data comes from libraries like TENDL-2025, whose
neutron evaluations carry per-final-state (ground + metastable) radionuclide
production (MF=8/9/10).  This module reads that data with the endf-python
``radionuclide_production`` reader, maps each product nuclear level to a
metastable state using decay data, and emits a ``branching/`` subsection
(verbatim energy-dependent MF=9 yields / MF=10 cross sections) for the split
transmutation format.  See issue #10.

The level -> isomer mapping is the subtle part, and lives in endf-python as
``isomer_table`` and ``level_to_isomeric_state``: MF=8 LFS is a *nuclear level*
index, not an isomeric-state ordinal, so a level is matched to a metastable
nuclide by comparing its excitation energy against the decay-data isomer
energies.
"""

import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from endf.data import ATOMIC_SYMBOL, gnds_name, zam
from endf.radionuclide_production import isomer_table, level_to_isomeric_state

from .schemas import BRANCHING_SCHEMA
from .transmutation_writer import (
    _write_arrow_ipc, _table, _write_json, _write_manifest,
)


def _mt_to_type():
    """Map ENDF MT number -> transmutation reaction type string.

    Built from endf-python's chain reaction set so the type strings match
    the reactions subsection exactly, plus (n,n') (MT=4) for inelastic isomeric
    transitions, which the default set omits.
    """
    from endf.chain import REACTIONS
    mapping = {}
    for name, info in REACTIONS.items():
        for mt in getattr(info, "mts", ()) or ():
            mapping.setdefault(mt, name)
    mapping.setdefault(4, "(n,n')")
    return mapping


def tendl_filename(nuclide):
    """TENDL neutron filename for a GNDS nuclide name (e.g. Nb93 -> n-Nb093.tendl,
    Ag110_m1 -> n-Ag110m.tendl). Returns the bare filename, not a path."""
    Z, A, m = zam(nuclide)
    suffix = {0: "", 1: "m", 2: "n", 3: "o"}.get(m, "")
    return f"n-{ATOMIC_SYMBOL[Z]}{A:03d}{suffix}.tendl"


def endf_neutron_filename(nuclide):
    """ENDF/B neutron sublibrary filename for a GNDS nuclide name
    (e.g. Nb93 -> n-041_Nb_093.endf, Ag110_m1 -> n-047_Ag_110m1.endf)."""
    Z, A, m = zam(nuclide)
    suffix = "" if m == 0 else f"m{m}"
    return f"n-{Z:03d}_{ATOMIC_SYMBOL[Z]}_{A:03d}{suffix}.endf"


def jeff_filename(nuclide):
    """JEFF neutron sublibrary filename for a GNDS nuclide name
    (e.g. Nb93 -> n_41-Nb-093g.jeff, Ag110_m1 -> n_47-Ag-110m.jeff).

    JEFF spells the isomeric state as a letter rather than an ordinal, and the
    ground state carries an explicit ``g``: Hf178 / Hf178_m1 / Hf178_m2 are
    ``n_72-Hf-178g`` / ``n_72-Hf-178m`` / ``n_72-Hf-178n``. Z is not zero-padded.
    """
    Z, A, m = zam(nuclide)
    state = {0: "g", 1: "m", 2: "n", 3: "o"}.get(m)
    if state is None:
        return None
    return f"n_{Z}-{ATOMIC_SYMBOL[Z]}-{A:03d}{state}.jeff"


@lru_cache(maxsize=8)
def _neutron_file_index(neutron_dir_str):
    """Map ``n-*.tendl`` / ``n-*.endf`` / ``n_*.jeff`` basename -> path, searched
    recursively under ``neutron_dir``. Some libraries (e.g. TENDL-2017) nest
    files as ``neutron_file/<El>/<Nuclide>/lib/endf/n-<Nuclide>.tendl`` rather
    than a flat directory; one recursive walk (cached per directory) handles
    both."""
    d = Path(neutron_dir_str)
    index = {}
    for pattern in ("n-*.tendl", "n-*.endf", "n_*.jeff"):
        for p in d.rglob(pattern):
            index.setdefault(p.name, p)
    return index


@lru_cache(maxsize=8)
def _explicit_file_index(paths):
    """Map basename -> path for an explicit list of neutron files."""
    index = {}
    for p in paths:
        index.setdefault(Path(p).name, Path(p))
    return index


def _find_neutron_file(neutron_source, nuclide):
    """Locate a nuclide's neutron ENDF file, trying the TENDL, ENDF/B and JEFF
    naming conventions. Returns a Path or None.

    ``neutron_source`` is either a directory to search, in which case nested
    layouts (TENDL-2017) work as well as flat ones, or an explicit iterable of
    files. The second form is what ``convert_transmutation`` has: it is handed
    the same ``neutron_files`` list it assembles the chain from, which may not
    share one parent directory."""
    names = tuple(n for n in (tendl_filename(nuclide),
                              endf_neutron_filename(nuclide),
                              jeff_filename(nuclide)) if n)
    if isinstance(neutron_source, (str, Path)):
        neutron_dir = Path(neutron_source)
        for name in names:
            p = neutron_dir / name
            if p.is_file():
                return p
        index = _neutron_file_index(str(neutron_dir))
    else:
        index = _explicit_file_index(tuple(sorted(str(p) for p in neutron_source)))
    for name in names:
        hit = index.get(name)
        if hit is not None:
            return hit
    return None


def _eval_left_right(energy, values, u):
    """Left and right limits of a stored curve at ``u`` under the consumer
    conventions (yamc ``curve_interp``): zero below the first point, lin-lin
    between points, flat above the last, with duplicated breakpoints carrying
    step jumps."""
    lo = np.searchsorted(energy, u, side="left")
    hi = np.searchsorted(energy, u, side="right")
    if hi == 0:
        return 0.0, 0.0
    if lo >= len(energy):
        return values[-1], values[-1]
    if lo == hi:
        # Strictly inside a segment (u is not a breakpoint).
        x0, x1 = energy[lo - 1], energy[lo]
        y0, y1 = values[lo - 1], values[lo]
        v = y0 if x1 == x0 else y0 + (u - x0) / (x1 - x0) * (y1 - y0)
        return v, v
    # u coincides with breakpoints lo..hi-1.
    left = values[lo] if lo > 0 else 0.0
    right = values[hi - 1]
    return left, right


def _merge_duplicate_rows(rows):
    """Sum rows sharing ``(nuclide, reaction, target, quantity)`` (issue #16).

    Duplicates are real: two nuclear levels can map to the same isomeric state
    (e.g. Ac225 (n,2p) emits two LFS levels that both resolve to Fr224), and
    the physical production is their sum. The merged curve is built on the
    union grid with each curve evaluated under the consumer conventions, so
    folding the merged curve gives exactly the sum of folding the originals;
    step jumps are preserved as duplicated breakpoints.

    Returns ``(merged_rows, n_merged_groups)``.
    """
    groups = {}
    order = []
    for row in rows:
        key = (row["nuclide"], row["reaction"], row["target"], row["quantity"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    merged_rows = []
    n_merged = 0
    for key in order:
        group = groups[key]
        if len(group) == 1:
            merged_rows.append(group[0])
            continue
        n_merged += 1
        curves = [
            (np.asarray(r["energy"], dtype=np.float64),
             np.asarray(r["values"], dtype=np.float64))
            for r in group
        ]
        grid = np.unique(np.concatenate([e for e, _ in curves]))
        out_x, out_y = [], []
        for i, u in enumerate(grid):
            lefts, rights = zip(*(_eval_left_right(e, v, u) for e, v in curves))
            left, right = sum(lefts), sum(rights)
            # A jump in the sum needs the duplicated-breakpoint form, except at
            # the first grid point where the below-threshold zero is implicit.
            if i > 0 and left != right:
                out_x.append(u)
                out_y.append(left)
            out_x.append(u)
            out_y.append(right)
        merged = dict(group[0])
        merged["energy"] = out_x
        merged["values"] = out_y
        merged_rows.append(merged)
    return merged_rows, n_merged


def extract_branching(neutron_dir, decay_files, nuclides, *, tol_eV=3000.0,
                      linearize_tol=1e-3):
    """Extract verbatim isomeric-branching rows for the given parent nuclides.

    Parameters
    ----------
    neutron_dir : path-like or iterable of path-like
        Directory of neutron ENDF files (TENDL ``n-*.tendl``, ENDF/B
        ``n-*.endf`` or JEFF ``n_*.jeff``; all three naming conventions are
        recognized).
    decay_files : list of path-like
        Decay ENDF files used to build the isomer table (metastable files
        suffice; ground states are implicit).
    nuclides : iterable of str
        Parent nuclide GNDS names to process (typically the reaction parents in
        the chain).
    tol_eV : float
        Energy-match tolerance for level -> isomer assignment.
    linearize_tol : float
        Relative tolerance for resampling non-lin-lin ENDF interpolation
        regions onto lin-lin pairs (see ``Tabulated1D.linearize``).

    Returns
    -------
    (rows, stats) : (list of dict, dict)
        ``rows`` follow ``BRANCHING_SCHEMA``; ``stats`` reports coverage,
        including how many curves needed linearization and how many
        duplicate-target groups were merged.
    """
    import endf
    from endf import radionuclide_production

    mt2type = _mt_to_type()
    isomers = isomer_table(decay_files)

    rows = []
    stats = {"parents": 0, "parents_with_data": 0, "missing_files": [],
             "metastable_targets": set(), "linearized_curves": 0,
             "merged_duplicate_groups": 0}

    for nuc in nuclides:
        fpath = _find_neutron_file(neutron_dir, nuc)
        if fpath is None:
            stats["missing_files"].append(nuc)
            continue
        stats["parents"] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = endf.Material(str(fpath))
            prod = radionuclide_production(mat)
        if not prod:
            continue
        emitted_any = False
        for mt, states in prod.items():
            rtype = mt2type.get(mt)
            if rtype is None:
                continue
            for s in states:
                Z, A = divmod(int(s.ZAP), 1000)
                liso = level_to_isomeric_state(Z, A, int(s.LFS),
                                               s.excitation_energy, isomers,
                                               tol_eV=tol_eV)
                target = gnds_name(Z, A, liso)
                if liso > 0:
                    stats["metastable_targets"].add(target)
                for quantity, tab in (("yield", s.yields), ("cross_section", s.cross_section)):
                    if tab is None:
                        continue
                    was_linearized = not tab.is_linear
                    if was_linearized:
                        stats["linearized_curves"] += 1
                    lin = tab.linearize(rel_tol=linearize_tol)
                    ex, vy = lin.x, lin.y
                    rows.append({
                        "nuclide": nuc,
                        "reaction": rtype,
                        "target": target,
                        "quantity": quantity,
                        "energy": ex.tolist(),
                        "values": vy.tolist(),
                    })
                    emitted_any = True
        if emitted_any:
            stats["parents_with_data"] += 1

    rows, stats["merged_duplicate_groups"] = _merge_duplicate_rows(rows)
    stats["metastable_targets"] = sorted(stats["metastable_targets"])
    return rows, stats


def export_branching_to_arrow(rows, path, *, library="", decay_library="",
                              data_version="", source="endf", tol_eV=3000.0,
                              linearize_tol=1e-3):
    """Write a branching subsection (branching/branching.arrow + provenance)
    plus a top-level manifest, mirroring the split transmutation layout."""
    path = Path(path)
    branching_dir = path / "branching"
    branching_dir.mkdir(parents=True, exist_ok=True)

    from . import __version__
    created_utc = datetime.now(timezone.utc).isoformat()

    _write_arrow_ipc(_table(rows, BRANCHING_SCHEMA), branching_dir / "branching.arrow")
    provenance = {
        "subsection": "branching",
        "library": library,
        "data_version": data_version,
        "source": source,
        "decay_library": decay_library,
        "isomer_energy_tol_eV": tol_eV,
        "linearize_tol": linearize_tol,
        "converter_version": __version__,
        "created_utc": created_utc,
    }
    _write_json(provenance, branching_dir / "provenance.json")

    _write_manifest(path, library, {"branching": {"path": "branching"}},
                    created_utc)
    return path
