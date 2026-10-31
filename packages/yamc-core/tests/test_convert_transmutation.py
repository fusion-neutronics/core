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
    """`data_version` is what invalidates a stale cache.

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


def test_fission_yield_evaluations_are_stored_as_the_tape_gives_them(tmp_path):
    """Both MT=454 and MT=459, with DY, beside the nominal yields.

    U235 joins the decay set so the synthetic yields reach the chain. Read with
    pyarrow, so this checks the file rather than yamc's reading of it.
    """
    pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    inputs = tmp_path / "endf"
    inputs.mkdir()
    out = tmp_path / "out"
    yamc.convert_transmutation(
        decay_files=_plain([*DECAY, "dec-092_U_235.endf.xz"], inputs),
        fpy_files=_plain(FPY, inputs),
        neutron_files=_plain([*NEUTRON, "n-092_U_235_trimmed.endf.xz"], inputs),
        output_path=str(out),
        library="endf-b8.1",
        data_version="2026-08-09.1",
        subsections=["fission_yields"],
    )

    nominal = ipc.open_file(out / "fission_yields" / "fission_yields.arrow").read_all()
    assert nominal.schema.names == [
        "nuclide",
        "energy",
        "products",
        "yields",
        "interpolation",
    ]
    # The law of each energy's MT=454 LIST, which the solver reads the yields
    # with, null at the lowest energy where the tape states LE instead.
    energies = nominal.column("energy").to_pylist()
    laws = dict(zip(energies, nominal.column("interpolation").to_pylist()))
    assert laws == {0.0253: None, 5.0e5: 2}

    table = ipc.open_file(out / "fission_yields" / "evaluated_yields.arrow").read_all()
    assert table.schema.names == [
        "nuclide",
        "energy",
        "kind",
        "interpolation",
        "products",
        "yields",
        "yield_uncertainties",
    ]
    rows = {(r["kind"], r["energy"]): r for r in table.to_pylist()}
    assert set(rows) == {
        (kind, energy)
        for kind in ("independent", "cumulative")
        for energy in (0.0253, 5.0e5)
    }
    assert {r["nuclide"] for r in rows.values()} == {"U235"}

    thermal = rows[("independent", 0.0253)]
    assert thermal["interpolation"] is None
    assert rows[("independent", 5.0e5)]["interpolation"] == 2
    # Xe135_m1 has no decay data here, so the evaluated file is the only place
    # its yield survives under its own name.
    i = thermal["products"].index("Xe135_m1")
    assert (thermal["yields"][i], thermal["yield_uncertainties"][i]) == (0.0134, 0.0006)
    cumulative = rows[("cumulative", 0.0253)]
    i = cumulative["products"].index("Zr95")
    assert (cumulative["yields"][i], cumulative["yield_uncertainties"][i]) == (0.0605, 0.0018)


def test_decay_mode_sigmas_are_stored_as_the_tape_gives_them(converted):
    """The dBR of every decay mode is in the file, and a 0.0 stays a 0.0.

    MT=457 writes 0.0 for an uncertainty it does not state. The file keeps
    that number rather than a null standing in for it, and readers take both
    as "not stated". Read with pyarrow, so this checks the file itself.
    """
    pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    out, _ = converted
    modes = ipc.open_file(out / "decay" / "decay_modes.arrow").read_all()
    assert modes.schema.names[-2:] == [
        "branching_ratio_uncertainty",
        "evaluated_branching_ratio",
    ]
    assert modes.column("branching_ratio_uncertainty").null_count == 0
    dbr = {}
    for nuclide, sigma in zip(
        modes.column("nuclide").to_pylist(),
        modes.column("branching_ratio_uncertainty").to_pylist(),
    ):
        dbr.setdefault(nuclide, []).append(sigma)
    # Cs137's two modes carry one stated number each; In116_m1's one mode
    # states none.
    assert dbr["Cs137"] == [1.999988e-3, 1.999988e-3]
    assert dbr["In116_m1"] == [0.0]


def test_decay_modes_keep_the_tapes_branching_ratio(converted):
    """The tape's BR is stored beside the normalised one the solver uses.

    Cs137's two modes sum to 1 - 1e-8 on the tape, so the residual is put on
    the larger one. ``evaluated_branching_ratio`` keeps the tape's number,
    which is the one its dBR was evaluated on.
    """
    pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    out, _ = converted
    modes = ipc.open_file(out / "decay" / "decay_modes.arrow").read_all()
    assert modes.column("evaluated_branching_ratio").null_count == 0
    rows = {}
    for nuclide, normalised, evaluated in zip(
        modes.column("nuclide").to_pylist(),
        modes.column("branching_ratio").to_pylist(),
        modes.column("evaluated_branching_ratio").to_pylist(),
    ):
        rows.setdefault(nuclide, []).append((normalised, evaluated))
    moved = sorted(n for n, r in rows.items() for a, b in r if a != b)
    assert moved == ["Cs137"]
    assert [e for _, e in rows["Cs137"]] == [0.05300549, 0.9469945]
    assert rows["Cs137"][1][0] == pytest.approx(0.94699451, abs=1e-15)


def test_nuclide_sigmas_are_stored_as_the_tape_gives_them(converted):
    """A decay-energy sigma the tape writes as 0.0 is 0.0 in the file too."""
    pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    out, _ = converted
    nuclides = ipc.open_file(out / "decay" / "nuclides.arrow").read_all()
    row = nuclides.column("name").to_pylist().index("Cs137")
    # Cs137 emits no heavy particles: the tape gives 0.0 +- 0.0.
    assert nuclides.column("decay_energy_alpha")[row].as_py() == 0.0
    assert nuclides.column("decay_energy_alpha_uncertainty")[row].as_py() == 0.0


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
    # The per-list facts reach Python: In115 lists only the isomer, so every
    # list it gives is isomers only, and each has a line of its own.
    counts = stats["list_counts"]
    assert counts.get("MF=10 isomers only", 0) >= 1, counts
    assert len(stats["list_facts"]) == sum(
        n for kind, n in counts.items() if kind.startswith("MF=")
    ), stats["list_facts"]

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
    # In115 carries no MF=40, so there is no covariance file beside the curves.
    assert stats["mf40_sections"] == 0
    assert not (out / "branching" / "branching_covariance.arrow").exists()


def test_a_level_far_from_the_only_isomer_is_flagged(tmp_path):
    """With In116_m2 the only In116 isomer in the decay data, In115's capture
    level at 127.3 keV is booked to it, 162.4 keV away, and the stats say so."""
    inputs = tmp_path / "endf"
    inputs.mkdir()
    stats = yamc.convert_branching(
        neutron_files=_plain(["n-049_In-115_trimmed.endf.xz"], inputs),
        decay_files=_plain(["dec-049_In_116m2.endf.xz"], inputs),
        output_path=str(tmp_path / "transmutation_endf-b8.1.arrow"),
        library="endf-b8.1",
        data_version="2026-10-09.1",
        created_utc="2026-10-09T00:00:00+00:00",
    )
    assert stats["level_routes"]["single_isomer"] == 1
    assert (
        "In115 MT102 -> In116_m2: level 1 at 127.3 keV, taken as the only isomer, "
        "-162.4 keV from it"
    ) in stats["flagged_levels"], stats["flagged_levels"]


def test_branching_covariance_is_written_beside_the_curves(tmp_path):
    """MF=40, from Python: TENDL-2017 Nb93 states the covariance of the ground
    and isomer partials of (n,n') and (n,2n), one LB=5 block each, and every
    one is written and placed on its chain target."""
    inputs = tmp_path / "endf"
    inputs.mkdir()
    out = tmp_path / "transmutation_tendl-2017.arrow"
    stats = yamc.convert_branching(
        neutron_files=_plain(["n-041_Nb_093_tendl2017_trimmed.endf.xz"], inputs),
        decay_files=_plain(
            [
                "dec-041_Nb_092.endf.xz",
                "dec-041_Nb_092m1.endf.xz",
                "dec-041_Nb_093m1.endf.xz",
            ],
            inputs,
        ),
        output_path=str(out),
        library="tendl-2017",
        data_version="2026-09-27.1",
        created_utc="2026-09-27T00:00:00+00:00",
    )
    assert (out / "branching" / "branching_covariance.arrow").is_file()
    assert stats["mf40_sections"] == 2
    assert stats["mf40_blocks"] == 4
    assert stats["mf40_blocks_by_lb"] == {5: 4}
    assert stats["mf40_nc_blocks"] == 0
    assert stats["mf40_cross_state_blocks"] == 0
    assert stats["mf40_unmatched_states"] == []
    assert stats["mf40_without_blocks"] == []
    assert stats["mf40_blocks_outside_mf10"] == 0
    assert stats["mf40_partner_unresolved"] == []
    assert stats["mf40_states_placed_by_excitation"] == []
    provenance = json.loads((out / "branching" / "provenance.json").read_text())
    assert provenance["mf40_without_blocks"] == []
    assert provenance["mf40_unmatched_states"] == []
    assert provenance["mf40_partner_unresolved"] == []
    assert provenance["mf40_states_placed_by_excitation"] == []
