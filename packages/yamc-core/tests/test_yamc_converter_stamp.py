"""Data written by the yamc wheel's converters names the yamc wheel.

The converters are shared with the yani wheel, but the ``converter`` stamp is
the wheel doing the writing: ``yamc-python`` and the installed yamc-core
version, then the core commit the extension was compiled from.
"""

import json
import lzma
from importlib.metadata import version
from pathlib import Path

import pytest
import yamc

DECAY = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "endf"
    / "fixtures"
    / "dec-055_Cs_137.endf.xz"
)


def expected_stamp():
    """The stamp this extension writes, from its own attributes."""
    package = f"yamc-python {version('yamc-core')}"
    commit, dirty = yamc.__commit__, yamc.__commit_dirty__
    if commit is None:
        return package
    return f"{package} (core {commit}{', dirty' if dirty else ''})"


def test_a_chain_converted_by_yamc_records_the_yamc_build(tmp_path):
    if not DECAY.is_file():
        pytest.skip(f"missing fixture {DECAY}")
    decay = tmp_path / "dec-055_Cs_137.endf"
    decay.write_bytes(lzma.decompress(DECAY.read_bytes()))
    converted = tmp_path / "converted"
    yamc.convert_transmutation(
        [str(decay)], [], [], str(converted), library="test", subsections=["decay"]
    )
    stamp = expected_stamp()
    for record in (converted / "decay" / "provenance.json", converted / "manifest.json"):
        assert json.loads(record.read_text())["converter"] == stamp, record
