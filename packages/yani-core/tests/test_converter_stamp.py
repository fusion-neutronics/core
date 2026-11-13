"""Data written by this extension records the build that wrote it.

The ``converter`` stamp is the package, its version and the core commit the
extension was compiled from (``yani.__commit__``), so data from a wheel built
from source, whose version is always 0.0.0.dev0, still traces to a commit.
"""

import json
import lzma
from pathlib import Path

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

DECAY = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "endf"
    / "fixtures"
    / "dec-055_Cs_137.endf.xz"
)


def expected_stamp():
    """The stamp this extension writes, from its own attributes."""
    from importlib.metadata import version

    package = f"yani-python {version('yani-core')}"
    commit, dirty = yani.__commit__, yani.__commit_dirty__
    if commit is None:
        return package
    return f"{package} (core {commit}{', dirty' if dirty else ''})"


def test_a_converted_and_exported_chain_records_the_build_that_wrote_it(tmp_path):
    if not DECAY.is_file():
        pytest.skip(f"missing fixture {DECAY}")
    decay = tmp_path / "dec-055_Cs_137.endf"
    decay.write_bytes(lzma.decompress(DECAY.read_bytes()))
    converted = tmp_path / "converted"
    yani.convert_transmutation(
        [str(decay)], [], [], str(converted), library="test", subsections=["decay"]
    )
    stamp = expected_stamp()
    for record in (converted / "decay" / "provenance.json", converted / "manifest.json"):
        written = json.loads(record.read_text())
        assert written["converter"] == stamp, record
        assert "converter_version" not in written, record

    exported = tmp_path / "exported"
    yani.TransmutationChain(str(converted)).export_to_arrow(str(exported))
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["converter"] == stamp
