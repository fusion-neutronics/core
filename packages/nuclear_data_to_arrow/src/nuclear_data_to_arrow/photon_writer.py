"""Export an IncidentPhoton object to a simulation-ready .arrow/ directory.

Stores energy and cross sections in both linear and log space for fast
interpolation.  Pre-computes Compton profile CDFs via trapezoidal integration.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from endf.function import Tabulated1D
from endf.incident_photon import (
    PHOTON_REACTION_NAME, _SUBSHELLS, compton_profile_cdfs, compton_subshell_map,
)

from .completion import write_completion_marker
from .schemas import (
    ELEMENT_SCHEMA,
    SUBSHELLS_SCHEMA,
    COMPTON_SCHEMA,
    BREMSSTRAHLUNG_SCHEMA,
)

# Map MT numbers to (name, key) -- mirrors _REACTION_NAME in photon.py
_MT_KEY_MAP = {
    502: ("coherent", "coherent"),
    504: ("incoherent", "incoherent"),
    515: ("pair_production_electron", "pair_production_electron"),
    517: ("pair_production_nuclear", "pair_production_nuclear"),
    522: ("photoelectric", "photoelectric"),
    525: ("heating", "heating"),
}


_SUBSHELL_INDEX = {designator: i for i, designator in enumerate(_SUBSHELLS)}


def _transitions_array(transitions):
    """Build the (n_transitions, 4) relaxation table for one subshell.

    Columns are secondary subshell, tertiary subshell, transition energy in eV
    and probability. The two subshell columns are stored as indices into
    ``_SUBSHELLS`` rather than names, so the whole table is one float array.
    A radiative transition has no tertiary subshell and is recorded as index 0
    (``_SUBSHELLS[0]`` is ``None``).
    """
    secondary = [_SUBSHELL_INDEX[s] for s in transitions["secondary_subshell"]]
    tertiary = [_SUBSHELL_INDEX[s] for s in transitions["tertiary_subshell"]]
    return np.column_stack([
        secondary,
        tertiary,
        np.asarray(transitions["energy"], dtype=np.float64),
        np.asarray(transitions["probability"], dtype=np.float64),
    ]).astype(np.float64)


def _safe_log(arr):
    """Compute log, replacing zeros/negatives with a very small value."""
    arr = np.asarray(arr, dtype=np.float64)
    safe = np.where(arr > 0, arr, 1e-300)
    return np.log(safe)


# Option-D hosting format: LZ4-compressed Arrow IPC (pure-Rust decode in yamc).
_LZ4 = ipc.IpcWriteOptions(compression="lz4")


def _write_arrow_ipc(table, filepath):
    """Write a PyArrow table as Arrow IPC (option-D: LZ4-compressed)."""
    with pa.OSFile(str(filepath), 'wb') as f:
        writer = ipc.new_file(f, table.schema, options=_LZ4)
        writer.write_table(table)
        writer.close()


def export_photon_to_arrow(data, path, *, library="", data_version=""):
    """Export an IncidentPhoton object to a simulation-ready .arrow/ directory.

    Parameters
    ----------
    data : endf.IncidentPhoton
        The incident photon data to export.
    path : str or Path
        Directory path to write Arrow files to (e.g., "Fe.arrow").
    library : str, optional
        Library name (e.g., "endfb-8.0", "fendl-3.2c").
    data_version : str, optional
        Identifier of the published release this directory belongs to, stamped
        into ``version.json``. See :func:`completion.write_completion_marker`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    Z = data.atomic_number

    # version.json is written at the end, not here: it is the completion
    # marker the resume path skips on, so it has to mean every table below
    # made it to disk. See completion.py.

    # Build union energy grid (same as export_to_hdf5)
    union_grid = np.array([])
    for rx in data:
        union_grid = np.union1d(union_grid, rx.xs.x)

    ln_energy = _safe_log(union_grid)

    # ------------------------------------------------------------------
    # Collect cross sections and form factors from reactions
    # ------------------------------------------------------------------
    xs_data = {}  # key -> xs array on union grid
    coherent_rx = None
    incoherent_rx = None
    subshell_rows = []

    for mt, rx in data.reactions.items():
        key = PHOTON_REACTION_NAME[mt]

        if mt in (502, 504, 515, 517, 522, 525):
            if mt >= 534 and mt <= 572:
                pass  # handled below
            else:
                xs_data[key] = np.asarray(rx.xs(union_grid), dtype=np.float64)

            if mt == 502:
                coherent_rx = rx
            elif mt == 504:
                incoherent_rx = rx

        elif mt >= 534 and mt <= 572:
            threshold = rx.xs.x[0]
            idx = int(np.searchsorted(union_grid, threshold, side='right') - 1)
            photoionization = np.asarray(rx.xs(union_grid[idx:]), dtype=np.float64)
            ln_photoionization = _safe_log(photoionization)

            binding_energy = 0.0
            num_electrons = 0.0
            transitions_data = None
            transitions_shape = None

            if data.atomic_relaxation is not None:
                if key in data.atomic_relaxation.subshells:
                    ar = data.atomic_relaxation
                    binding_energy = float(ar.binding_energy[key])
                    num_electrons = float(ar.num_electrons[key])
                    if key in ar.transitions:
                        t_arr = _transitions_array(ar.transitions[key])
                        transitions_data = t_arr.ravel(order='C').tolist()
                        transitions_shape = list(t_arr.shape)

            subshell_rows.append({
                "designator": key,
                "binding_energy": binding_energy,
                "num_electrons": num_electrons,
                "xs": photoionization.tolist(),
                "ln_xs": ln_photoionization.tolist(),
                "threshold_idx": idx,
                "transitions_data": transitions_data,
                "transitions_shape": transitions_shape,
            })

    # ------------------------------------------------------------------
    # element.arrow
    # ------------------------------------------------------------------
    element_row = {
        "name": data.name,
        "Z": int(Z),
        "ln_energy": ln_energy.tolist(),
    }

    _xs_key_map = {
        "coherent": "coherent_xs",
        "incoherent": "incoherent_xs",
        "photoelectric": "photoelectric_xs",
        "pair_production_nuclear": "pair_production_nuclear_xs",
        "pair_production_electron": "pair_production_electron_xs",
        "heating": "heating_xs",
    }
    for key, col_name in _xs_key_map.items():
        if key in xs_data:
            element_row[col_name] = xs_data[key].tolist()
        else:
            element_row[col_name] = []

    # Form factors for coherent scattering
    if coherent_rx is not None and coherent_rx.scattering_factor is not None:
        ff = coherent_rx.scattering_factor
        ff_copy = deepcopy(ff)
        ff_copy.x = ff_copy.x * ff_copy.x
        ff_copy.y = ff_copy.y * ff_copy.y / Z**2
        int_ff = Tabulated1D(ff_copy.x, ff_copy.integral())
        element_row["coherent_int_ff_x"] = np.asarray(int_ff.x, dtype=np.float64).tolist()
        element_row["coherent_int_ff_y"] = np.asarray(int_ff.y, dtype=np.float64).tolist()
        element_row["coherent_ff_x"] = np.asarray(ff.x, dtype=np.float64).tolist()
        element_row["coherent_ff_y"] = np.asarray(ff.y, dtype=np.float64).tolist()
    else:
        element_row["coherent_int_ff_x"] = []
        element_row["coherent_int_ff_y"] = []
        element_row["coherent_ff_x"] = []
        element_row["coherent_ff_y"] = []

    if coherent_rx is not None and coherent_rx.anomalous_real is not None:
        element_row["coherent_anomalous_real_x"] = np.asarray(
            coherent_rx.anomalous_real.x, dtype=np.float64).tolist()
        element_row["coherent_anomalous_real_y"] = np.asarray(
            coherent_rx.anomalous_real.y, dtype=np.float64).tolist()
    else:
        element_row["coherent_anomalous_real_x"] = []
        element_row["coherent_anomalous_real_y"] = []

    if coherent_rx is not None and coherent_rx.anomalous_imag is not None:
        element_row["coherent_anomalous_imag_x"] = np.asarray(
            coherent_rx.anomalous_imag.x, dtype=np.float64).tolist()
        element_row["coherent_anomalous_imag_y"] = np.asarray(
            coherent_rx.anomalous_imag.y, dtype=np.float64).tolist()
    else:
        element_row["coherent_anomalous_imag_x"] = []
        element_row["coherent_anomalous_imag_y"] = []

    if incoherent_rx is not None and incoherent_rx.scattering_factor is not None:
        element_row["incoherent_ff_x"] = np.asarray(
            incoherent_rx.scattering_factor.x, dtype=np.float64).tolist()
        element_row["incoherent_ff_y"] = np.asarray(
            incoherent_rx.scattering_factor.y, dtype=np.float64).tolist()
    else:
        element_row["incoherent_ff_x"] = []
        element_row["incoherent_ff_y"] = []

    element_table = pa.table(
        {col: [element_row[col]] for col in ELEMENT_SCHEMA.names},
        schema=ELEMENT_SCHEMA,
    )
    _write_arrow_ipc(element_table, path / "element.arrow")

    # ------------------------------------------------------------------
    # subshells.arrow
    # ------------------------------------------------------------------
    if subshell_rows:
        subshells_table = pa.table(
            {col: [r[col] for r in subshell_rows]
             for col in SUBSHELLS_SCHEMA.names},
            schema=SUBSHELLS_SCHEMA,
        )
        _write_arrow_ipc(subshells_table, path / "subshells.arrow")

    # ------------------------------------------------------------------
    # compton.arrow
    # ------------------------------------------------------------------
    if data.compton_profiles:
        profile = data.compton_profiles
        J_arr = np.array([Jk.y for Jk in profile['J']], dtype=np.float64)
        pz = np.asarray(profile['J'][0].x, dtype=np.float64)

        # Pre-compute the cumulative distributions, left un-normalised
        J_cdf = compton_profile_cdfs(J_arr, pz)

        # subshells.arrow rows must be in canonical (most-bound-first) order for
        # the occupancy grouping to align; they are, because they are built from
        # MT 534..572 in ascending order. Assert it so a future ordering change
        # fails loudly instead of writing a silently wrong map.
        _positions = [_SUBSHELL_INDEX[r["designator"]] for r in subshell_rows
                      if r["designator"] in _SUBSHELL_INDEX]
        assert _positions == sorted(_positions), (
            "subshell rows are not in canonical order; the Compton subshell map "
            "would be wrong"
        )
        map_offsets, map_indices, map_weights = compton_subshell_map(
            profile['num_electrons'],
            [r["num_electrons"] for r in subshell_rows],
        )

        compton_row = {
            "num_electrons": np.asarray(profile['num_electrons'], dtype=np.float64).tolist(),
            "binding_energy": np.asarray(profile['binding_energy'], dtype=np.float64).tolist(),
            "pz": np.asarray(profile['J'][0].x, dtype=np.float64).tolist(),
            "J_data": J_arr.ravel(order='C').tolist(),
            "J_shape": list(J_arr.shape),
            "J_cdf_data": J_cdf.ravel(order='C').tolist(),
            "J_cdf_shape": list(J_cdf.shape),
            "subshell_map_offsets": map_offsets,
            "subshell_map_indices": map_indices,
            "subshell_map_weights": map_weights,
        }
        compton_table = pa.table(
            {col: [compton_row[col]] for col in COMPTON_SCHEMA.names},
            schema=COMPTON_SCHEMA,
        )
        _write_arrow_ipc(compton_table, path / "compton.arrow")

    # ------------------------------------------------------------------
    # bremsstrahlung.arrow
    # ------------------------------------------------------------------
    if data.bremsstrahlung:
        brem = data.bremsstrahlung
        dcs_arr = np.asarray(brem['dcs'], dtype=np.float64)
        brem_row = {
            "I": float(brem['I']),
            "electron_energy": np.asarray(brem['electron_energy'], dtype=np.float64).tolist(),
            "photon_energy": np.asarray(brem['photon_energy'], dtype=np.float64).tolist(),
            "num_electrons": np.asarray(brem['num_electrons'], dtype=np.float64).tolist(),
            "ionization_energy": np.asarray(brem['ionization_energy'], dtype=np.float64).tolist(),
            "dcs_data": dcs_arr.ravel(order='C').tolist(),
            "dcs_shape": list(dcs_arr.shape),
        }
        brem_table = pa.table(
            {col: [brem_row[col]] for col in BREMSSTRAHLUNG_SCHEMA.names},
            schema=BREMSSTRAHLUNG_SCHEMA,
        )
        _write_arrow_ipc(brem_table, path / "bremsstrahlung.arrow")

    # ------------------------------------------------------------------
    # version.json, last, so it means the whole directory is there
    # ------------------------------------------------------------------
    write_completion_marker(path, library, data_version)
