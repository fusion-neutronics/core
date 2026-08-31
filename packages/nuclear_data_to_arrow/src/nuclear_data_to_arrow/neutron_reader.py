"""Read a simulation-ready .arrow/ neutron directory back into Python dicts."""

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


def _read_multi_row(table):
    """Read a multi-row table into a list of dicts."""
    rows = []
    for i in range(len(table)):
        row = {}
        for name in table.schema.names:
            row[name] = _to_list(table.column(name)[i])
        rows.append(row)
    return rows


def read_neutron_from_arrow(path):
    """Read a .arrow/ neutron directory into a dict of dicts/lists.

    Parameters
    ----------
    path : str or Path
        Path to the Arrow directory (e.g., "Li6.arrow").

    Returns
    -------
    dict
        Dictionary with keys: "version", "nuclide", "reactions", "products",
        "distributions", "fast_xs", "urr" (optional), "total_nu" (optional).
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

    # nuclide
    nuclide_file = path / "nuclide.arrow"
    if nuclide_file.exists():
        nuclide_table = _read_arrow_ipc(nuclide_file)
        result["nuclide"] = _read_single_row(nuclide_table)
        result["nuclide"]["_metadata"] = nuclide_table.schema.metadata

    # reactions
    reactions_file = path / "reactions.arrow"
    if reactions_file.exists():
        reactions_table = _read_arrow_ipc(reactions_file)
        result["reactions"] = _read_multi_row(reactions_table)
    else:
        result["reactions"] = []

    # products
    products_file = path / "products.arrow"
    if products_file.exists():
        products_table = _read_arrow_ipc(products_file)
        result["products"] = _read_multi_row(products_table)
    else:
        result["products"] = []

    # distributions
    distributions_file = path / "distributions.arrow"
    if distributions_file.exists():
        distributions_table = _read_arrow_ipc(distributions_file)
        result["distributions"] = _read_multi_row(distributions_table)
    else:
        result["distributions"] = []

    # fast_xs
    fast_xs_file = path / "fast_xs.arrow"
    if fast_xs_file.exists():
        fast_xs_table = _read_arrow_ipc(fast_xs_file)
        result["fast_xs"] = _read_multi_row(fast_xs_table)

    # urr (optional)
    urr_file = path / "urr.arrow"
    if urr_file.exists():
        urr_table = _read_arrow_ipc(urr_file)
        result["urr"] = _read_multi_row(urr_table)

    # total_nu (optional)
    total_nu_file = path / "total_nu.arrow"
    if total_nu_file.exists():
        total_nu_table = _read_arrow_ipc(total_nu_file)
        result["total_nu"] = _read_single_row(total_nu_table)

    return result
