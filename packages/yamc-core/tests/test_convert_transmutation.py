"""`convert_transmutation` from Python, with no `endf` package installed.

The whole point of moving the parser into this repository. Before it, producing
transmutation data meant installing the `endf` package from a branch of a fork,
plus a separate converter distribution resolved by relative path out of a
sibling checkout. This test asserts the replacement: import the wheel, call one
function, get a directory yani reads.

It deliberately does NOT import `endf`. If that import ever becomes necessary
again, this file stops being a fair test of the claim.
"""

import json
import lzma
from pathlib import Path

import pytest

import yamc

# The committed evaluations the Rust round-trip tests use, decompressed into a
# temp directory because the Python entry point takes paths on disk.
FIXTURES = Path(__file__).resolve().parents[3] / "crates" / "endf" / "fixtures"

DECAY = [
    "dec-055_Cs_137.endf.xz",
    "dec-054_Xe_136.endf.xz",
    "dec-054_Xe_137.endf.xz",
    "dec-049_In_115.endf.xz",
    "dec-049_In_116.endf.xz",
    "dec-049_In_116m1.endf.xz",
    "dec-049_In_116m2.endf.xz",
    "dec-050_Sn_115.endf.xz",
    "dec-050_Sn_116.endf.xz",
    "dec-048_Cd_116.endf.xz",
]
NEUTRON = ["n-049_In-115_trimmed.endf.xz", "n-054_Xe_136_trimmed.endf.xz"]
FPY = ["synthetic-nfy.endf.xz"]


def _plain(names, into):
    """Decompress fixtures into *into*, returning the written paths."""
    out = []
    for name in names:
        source = FIXTURES / name
        assert source.is_file(), f"missing fixture {source}"
        target = into / name.removesuffix(".xz")
        target.write_bytes(lzma.decompress(source.read_bytes()))
        out.append(str(target))
    return out


@pytest.fixture
def converted(tmp_path):
    inputs = tmp_path / "endf"
    inputs.mkdir()
    out = tmp_path / "transmutation_endf-b8.1.arrow"
    n = yamc.convert_transmutation(
        decay_files=_plain(DECAY, inputs),
        fpy_files=_plain(FPY, inputs),
        neutron_files=_plain(NEUTRON, inputs),
        output_path=str(out),
        library="endf-b8.1",
        data_version="2026-08-09.1",
        created_utc="2026-08-09T00:00:00+00:00",
    )
    return out, n


def test_converts_without_the_endf_package(converted):
    """One call, from the wheel, produces a complete directory."""
    out, n = converted
    assert n >= 10, f"only {n} nuclides; the conversion did almost nothing"
    for rel in [
        "manifest.json",
        "decay/nuclides.arrow",
        "decay/decay_modes.arrow",
        "decay/sources.arrow",
        "decay/provenance.json",
        "reactions/reactions.arrow",
        "reactions/provenance.json",
        "fission_yields/fission_yields.arrow",
        "fission_yields/provenance.json",
    ]:
        assert (out / rel).is_file(), f"{rel} was not written"


def test_the_output_carries_its_provenance(converted):
    """`data_version` is what invalidates a stale cache (#366).

    A directory without it is one a consumer can never be told to refetch.
    """
    out, _ = converted
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["library"] == "endf-b8.1"
    assert manifest["data_version"] == "2026-08-09.1"
    assert manifest["format_version"] == 2
    assert set(manifest["subsections"]) == {"decay", "reactions", "fission_yields"}

    for subsection in ("decay", "reactions", "fission_yields"):
        p = json.loads((out / subsection / "provenance.json").read_text())
        assert p["data_version"] == "2026-08-09.1"
        assert p["source"] == "endf"


def test_reactions_carry_q(converted):
    """The column yani's own writer used to drop.

    Read with pyarrow rather than through yamc, so this checks the file rather
    than agreeing with whatever the reader chooses to expose.
    """
    pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    out, _ = converted
    table = ipc.open_file(out / "reactions" / "reactions.arrow").read_all()
    assert "Q" in table.schema.names, "reactions.arrow has no Q column"
    assert table.num_rows > 0, "no reactions were written, so this proves nothing"
    q = table.column("Q").to_pylist()
    assert any(v != 0.0 for v in q), "every Q is zero, which is not real data"


def test_missing_inputs_are_refused(tmp_path):
    """A partial chain is a wrong chain, not a smaller one."""
    with pytest.raises(ValueError, match="decay_files"):
        yamc.convert_transmutation(
            decay_files=[],
            fpy_files=[],
            neutron_files=[],
            output_path=str(tmp_path / "out"),
        )


def test_branching_subsection(tmp_path):
    """Isomeric branching, from Python, merged into the same manifest.

    In115 (n,gamma) populates both In116 and In116_m1, so the level-to-isomer
    resolution has something real to resolve rather than a ground state to fall
    back to.
    """
    inputs = tmp_path / "endf"
    inputs.mkdir()
    out = tmp_path / "transmutation_endf-b8.1.arrow"
    decay = _plain(DECAY, inputs)
    neutron = _plain(NEUTRON, inputs)

    yamc.convert_transmutation(
        decay_files=decay,
        fpy_files=_plain(FPY, inputs),
        neutron_files=neutron,
        output_path=str(out),
        library="endf-b8.1",
        data_version="2026-08-09.1",
        created_utc="2026-08-09T00:00:00+00:00",
    )
    stats = yamc.convert_branching(
        neutron_files=neutron,
        decay_files=decay,
        output_path=str(out),
        library="endf-b8.1",
        data_version="2026-08-09.1",
        created_utc="2026-08-09T00:00:00+00:00",
    )

    assert stats["parents"] >= 2, f"only {stats['parents']} parents read"
    assert (out / "branching" / "branching.arrow").is_file()

    # The branching call must merge into the manifest, not overwrite it. A
    # library advertising one subsection while shipping four is a real failure
    # mode this format has had.
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["subsections"]) == {
        "decay",
        "reactions",
        "fission_yields",
        "branching",
    }
