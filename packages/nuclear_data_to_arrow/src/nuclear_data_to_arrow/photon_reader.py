"""Read a simulation-ready .arrow/ photon directory back into Python dicts."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def _read_arrow_ipc(filepath):
    """Read an Arrow IPC file into a PyArrow table."""
    with pa.OSFile(str(filepath), 'rb') as f:
        reader = ipc.open_file(f)
        return reader.read_all()


def _to_list(val):
    """Convert pyarrow scalar/list to Python list or value."""
    if val is None:
        return None
    v = val.as_py() if hasattr(val, 'as_py') else val
    return v


def _read_single_row(table):
    """Read a single-row table into a dict."""
    assert len(table) == 1
    row = {}
    for name in table.schema.names:
        row[name] = _to_list(table.column(name)[0])
    return row


def read_photon_from_arrow(path):
    """Read a .arrow/ photon directory into a dict of dicts/lists.

    Parameters
    ----------
    path : str or Path
        Path to the Arrow directory (e.g., "Fe.arrow").

    Returns
    -------
    dict
        Dictionary with keys: "version", "element", "subshells",
        "compton" (optional), "bremsstrahlung" (optional).
    """
    path = Path(path)
    result = {}

    # version.json
    version_path = path / "version.json"
    if version_path.exists():
        result["version"] = json.loads(version_path.read_text())
        fmt_ver = result["version"].get("format_version")
        if fmt_ver is not None and fmt_ver > 1:
            raise ValueError(
                f"Unsupported format_version {fmt_ver} in {version_path}. "
                f"This reader supports format_version <= 1."
            )

    # element
    element_file = path / "element.arrow"
    if element_file.exists():
        element_table = _read_arrow_ipc(element_file)
        result["element"] = _read_single_row(element_table)
        result["element"]["_metadata"] = element_table.schema.metadata

    # subshells
    subshells_file = path / "subshells.arrow"
    if subshells_file.exists():
        subshells_table = _read_arrow_ipc(subshells_file)
        result["subshells"] = []
        for i in range(len(subshells_table)):
            row = {}
            for name in subshells_table.schema.names:
                row[name] = _to_list(subshells_table.column(name)[i])
            result["subshells"].append(row)
    else:
        result["subshells"] = []

    # compton (optional)
    compton_file = path / "compton.arrow"
    if compton_file.exists():
        compton_table = _read_arrow_ipc(compton_file)
        result["compton"] = _read_single_row(compton_table)

    # bremsstrahlung (optional)
    brem_file = path / "bremsstrahlung.arrow"
    if brem_file.exists():
        brem_table = _read_arrow_ipc(brem_file)
        result["bremsstrahlung"] = _read_single_row(brem_table)

    return result
