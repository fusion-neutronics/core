"""Tests for the TENDL isomeric-branching extractor.

The level -> isomer mapping and the interpolation linearization now live in
endf-python and are tested there.  What remains here is the file discovery, the
duplicate-target merging, and an end-to-end extraction that runs against the
local TENDL-2025 and ENDF/B-8.1 decay data when present and skips otherwise
(the raw libraries are too large to vendor).
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from nuclear_data_to_arrow.branching_extractor import (
    tendl_filename,
    endf_neutron_filename,
    jeff_filename,
    _find_neutron_file,
    _neutron_file_index,
    _merge_duplicate_rows,
    _eval_left_right,
)

# The end-to-end extraction needs a TENDL neutron library and a decay library,
# neither of which can be vendored. Both are searched for in a few plausible
# places and can be pointed at explicitly with TENDL_NEUTRON_DIR and
# ENDF_DECAY_DIR. When absent the test skips, and the skip message names what is
# missing rather than leaving it a mystery.
_TENDL_CANDIDATES = [
    os.environ.get("TENDL_NEUTRON_DIR"),
    Path.home() / "yamc-org" / "nuclear_data_generation_scripts" / "data" / "tendl-2025-endf",
    "/home/jon/yamc-org/cross_section_data_tendl_2025_arrow/tendl-2025-endf",
    Path.home() / "nuclear_data" / "tendl-2025-endf",
]
_DECAY_CANDIDATES = [
    os.environ.get("ENDF_DECAY_DIR"),
    "/home/jon/nuclear_data/endfb-viii.1-endf/decay-version.VIII.1",
    Path.home() / "nuclear_data" / "decay-version.VIII.1",
]


def _first_dir_with(candidates, pattern):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir() and next(path.rglob(pattern), None) is not None:
            return path
    return None


_FPY_CANDIDATES = [
    os.environ.get("ENDF_FPY_DIR"),
    "/home/jon/nuclear_data/endfb-viii.1-endf/nfy-version.VIII.1",
    Path.home() / "nuclear_data" / "nfy-version.VIII.1",
]

_TENDL = _first_dir_with(_TENDL_CANDIDATES, "n-*.tendl")
_DECAY = _first_dir_with(_DECAY_CANDIDATES, "dec-*.endf")
_FPY = _first_dir_with(_FPY_CANDIDATES, "nfy-*.endf")
_MISSING = ", ".join(
    name for name, found in (("TENDL_NEUTRON_DIR", _TENDL),
                             ("ENDF_DECAY_DIR", _DECAY)) if found is None
)
# Assembling a whole chain needs fission yields on top of the above.
_MISSING_CHAIN = ", ".join(
    name for name, found in (("TENDL_NEUTRON_DIR", _TENDL),
                             ("ENDF_DECAY_DIR", _DECAY),
                             ("ENDF_FPY_DIR", _FPY)) if found is None
)


def test_tendl_filename():
    assert tendl_filename("Nb93") == "n-Nb093.tendl"
    assert tendl_filename("Ag109") == "n-Ag109.tendl"
    assert tendl_filename("In115") == "n-In115.tendl"
    assert tendl_filename("Ag110_m1") == "n-Ag110m.tendl"
    assert tendl_filename("In116_m2") == "n-In116n.tendl"


def test_endf_neutron_filename():
    assert endf_neutron_filename("Nb93") == "n-041_Nb_093.endf"
    assert endf_neutron_filename("Ag109") == "n-047_Ag_109.endf"
    assert endf_neutron_filename("Ag110_m1") == "n-047_Ag_110m1.endf"
    assert endf_neutron_filename("Co58_m1") == "n-027_Co_058m1.endf"


def test_jeff_filename():
    assert jeff_filename("Nb93") == "n_41-Nb-093g.jeff"
    assert jeff_filename("H1") == "n_1-H-001g.jeff"
    assert jeff_filename("Fm255") == "n_100-Fm-255g.jeff"
    # JEFF spells the isomeric state as a letter, not an ordinal.
    assert jeff_filename("Ag110_m1") == "n_47-Ag-110m.jeff"
    assert jeff_filename("Hf178_m2") == "n_72-Hf-178n.jeff"


def test_find_neutron_file_jeff(tmp_path):
    d = tmp_path / "jeff"
    d.mkdir()
    (d / "n_26-Fe-056g.jeff").write_text("x")
    (d / "n_47-Ag-110m.jeff").write_text("x")
    _neutron_file_index.cache_clear()
    assert _find_neutron_file(d, "Fe56") == d / "n_26-Fe-056g.jeff"
    assert _find_neutron_file(d, "Ag110_m1") == d / "n_47-Ag-110m.jeff"
    assert _find_neutron_file(d, "Xe135") is None


def test_find_neutron_file_flat_and_nested(tmp_path):
    # Flat layout (TENDL-2025 style): n-*.tendl directly in the dir.
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "n-Fe056.tendl").write_text("x")
    assert _find_neutron_file(flat, "Fe56") == flat / "n-Fe056.tendl"

    # Nested layout (TENDL-2017 style): neutron_file/<El>/<Nuclide>/lib/endf/.
    nested = tmp_path / "nested"
    leaf = nested / "neutron_file" / "Pb" / "Pb204" / "lib" / "endf"
    leaf.mkdir(parents=True)
    (leaf / "n-Pb204.tendl").write_text("x")
    _neutron_file_index.cache_clear()
    assert _find_neutron_file(nested, "Pb204") == leaf / "n-Pb204.tendl"
    # A nuclide with no file returns None.
    assert _find_neutron_file(nested, "Xe135") is None


def test_find_neutron_file_from_an_explicit_list(tmp_path):
    """An explicit list of files, which is what convert_transmutation has.

    It is handed the same neutron_files it builds the chain from, and those need
    not share a parent directory, so there is no directory to walk.
    """
    one = tmp_path / "a"
    two = tmp_path / "b" / "c"
    one.mkdir()
    two.mkdir(parents=True)
    (one / "n-Fe056.tendl").write_text("x")
    (two / "n-041_Nb_093.endf").write_text("x")
    files = [one / "n-Fe056.tendl", two / "n-041_Nb_093.endf"]

    # Both are found despite living under different parents, and each is matched
    # by its own library's naming convention.
    assert _find_neutron_file(files, "Fe56") == one / "n-Fe056.tendl"
    assert _find_neutron_file(files, "Nb93") == two / "n-041_Nb_093.endf"
    assert _find_neutron_file(files, "Xe135") is None

    # A directory is still accepted, and a str is a directory rather than a
    # one-character iterable of files.
    assert _find_neutron_file(str(one), "Fe56") == one / "n-Fe056.tendl"


@pytest.mark.skipif(bool(_MISSING),
                    reason=f"set {_MISSING} to run the end-to-end extraction")
def test_extract_branching_integration(tmp_path):
    import glob
    from nuclear_data_to_arrow.branching_extractor import (
        extract_branching, export_branching_to_arrow,
    )

    decay_files = sorted(glob.glob(str(_DECAY / "dec-*m[0-9].endf")))
    rows, stats = extract_branching(_TENDL, decay_files, ["Ag109", "In115", "Nb93"])

    triples = {(r["nuclide"], r["reaction"], r["target"]) for r in rows}
    # (n,gamma) isomeric capture on Ag109 -> Ag110 ground + metastable
    assert ("Ag109", "(n,gamma)", "Ag110") in triples
    assert ("Ag109", "(n,gamma)", "Ag110_m1") in triples
    # (n,n') inelastic isomeric transition Nb93 -> Nb93_m1
    assert ("Nb93", "(n,n')", "Nb93_m1") in triples
    # (n,gamma) on Nb93 -> Nb94_m1
    assert ("Nb93", "(n,gamma)", "Nb94_m1") in triples
    # every metastable target is a valid GNDS name with an _m suffix
    assert all("_m" in t for t in stats["metastable_targets"])

    out = tmp_path / "transmutation_tendl-2025.arrow"
    export_branching_to_arrow(rows, out, library="tendl-2025", decay_library="endfb-8.1")
    with pa.memory_map(str(out / "branching" / "branching.arrow"), "r") as s:
        tbl = pa.ipc.open_file(s).read_all()
    assert tbl.num_rows == len(rows)
    assert tbl.schema.names == ["nuclide", "reaction", "target", "quantity", "energy", "values"]
    assert tbl.schema.metadata[b"filetype"] == b"transmutation-branching"

    import json
    prov = json.loads((out / "branching" / "provenance.json").read_text())
    assert prov["subsection"] == "branching"
    assert prov["library"] == "tendl-2025"
    assert prov["decay_library"] == "endfb-8.1"


# ---------------------------------------------------------------------------
# Linearization + duplicate merging (issue #16)
# ---------------------------------------------------------------------------


def _linlin_eval(x, y, u):
    """Reference lin-lin evaluation of the OUTPUT pairs (yamc conventions:
    zero below first point, flat above last; duplicate x = jump, right value
    wins at the jump energy)."""
    left, right = _eval_left_right(np.asarray(x, float), np.asarray(y, float), u)
    return right





def test_merge_duplicate_rows_sums_on_union_grid():
    row = {"nuclide": "Ac225", "reaction": "(n,2p)", "target": "Fr224",
           "quantity": "cross_section"}
    a = dict(row, energy=[1.0, 3.0, 5.0], values=[0.0, 2.0, 2.0])
    b = dict(row, energy=[2.0, 6.0], values=[1.0, 1.0])
    other = {"nuclide": "Nb93", "reaction": "(n,n')", "target": "Nb93_m1",
             "quantity": "cross_section", "energy": [1.0, 2.0], "values": [1.0, 1.0]}
    merged, n = _merge_duplicate_rows([a, other, b])
    assert n == 1
    assert len(merged) == 2
    assert merged[1] == other, "singletons pass through untouched"
    m = merged[0]
    # The merged curve must equal the sum of the separate curves under the
    # consumer conventions (zero below threshold, flat above the last point).
    for u in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 9.0]:
        expect = _linlin_eval(a["energy"], a["values"], u) + \
            _linlin_eval(b["energy"], b["values"], u)
        got = _linlin_eval(m["energy"], m["values"], u)
        assert abs(got - expect) < 1e-12, (u, got, expect)
    # Below both thresholds the merged curve is zero by the threshold rule.
    assert _linlin_eval(m["energy"], m["values"], 0.5) == 0.0


def test_merge_preserves_step_jumps():
    row = {"nuclide": "X", "reaction": "(n,2n)", "target": "Y",
           "quantity": "cross_section"}
    # Curve with an interior step (duplicated breakpoint).
    a = dict(row, energy=[1.0, 2.0, 2.0, 4.0], values=[1.0, 1.0, 3.0, 3.0])
    b = dict(row, energy=[1.0, 4.0], values=[10.0, 10.0])
    merged, n = _merge_duplicate_rows([a, b])
    assert n == 1
    m = merged[0]
    left, right = _eval_left_right(np.asarray(m["energy"], float),
                                   np.asarray(m["values"], float), 2.0)
    assert left == 11.0 and right == 13.0, "the jump must survive the merge"


def test_eval_left_right_conventions():
    e = np.array([2.0, 4.0])
    v = np.array([1.0, 3.0])
    assert _eval_left_right(e, v, 1.0) == (0.0, 0.0)     # below threshold
    assert _eval_left_right(e, v, 2.0) == (0.0, 1.0)     # threshold jump
    assert _eval_left_right(e, v, 3.0) == (2.0, 2.0)     # lin-lin interior
    assert _eval_left_right(e, v, 4.0) == (3.0, 3.0)     # last point
    assert _eval_left_right(e, v, 9.0) == (3.0, 3.0)     # flat above


@pytest.mark.skipif(bool(_MISSING_CHAIN),
                    reason=f"set {_MISSING_CHAIN} to build a whole chain")
def test_convert_transmutation_emits_branching(tmp_path):
    """A chain built through convert_transmutation carries branching.

    Not optional: without it a chain is quietly wrong for any material whose
    activity comes from an isomer, because the metastable fraction of an (n,2n)
    is energy dependent and a scalar branching table cannot express it. Niobium
    is the case that shows it, 6x low on FNS decay heat when the branching comes
    from the wrong library.
    """
    from nuclear_data_to_arrow import convert_transmutation

    out = tmp_path / "transmutation_test.arrow"
    convert_transmutation(
        out,
        decay_files=sorted(_DECAY.glob("dec-*.endf")),
        fpy_files=sorted(_FPY.glob("nfy-*.endf")),
        neutron_files=[_TENDL / "n-Nb093.tendl"],
        library="test",
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert "branching" in manifest["subsections"]
    assert (out / "branching" / "branching.arrow").is_file()

    rows = ipc.open_file(out / "branching" / "branching.arrow").read_all().to_pydict()
    # Scoped to the chain's own reaction parents, so Nb93 is in and nothing else
    # is, and the (n,2n) appears split between the ground and metastable target.
    assert set(rows["nuclide"]) == {"Nb93"}
    n2n = {t for n, r, t in zip(rows["nuclide"], rows["reaction"], rows["target"])
           if r == "(n,2n)"}
    assert n2n == {"Nb92", "Nb92_m1"}
