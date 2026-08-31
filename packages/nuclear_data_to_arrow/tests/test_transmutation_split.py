"""Tests for the split transmutation output (format_version 2 subsections)."""

import json
from types import SimpleNamespace

import endf.univariate
import pyarrow as pa
import pytest

from nuclear_data_to_arrow import export_transmutation_to_arrow


def _decay_mode(type_, target, br=1.0):
    return SimpleNamespace(type=type_, target=target, branching_ratio=br)


def _reaction(type_, target, Q=0.0, br=1.0):
    return SimpleNamespace(type=type_, target=target, Q=Q, branching_ratio=br)


def _fy_entry(products, yields):
    return SimpleNamespace(products=list(products), yields=list(yields))


def _make_chain():
    """A tiny network exercising every subsection and the fy-alias path."""
    fe56 = SimpleNamespace(
        name="Fe56", half_life=None, decay_energy=0.0,
        decay_modes=[], sources={},
        reactions=[_reaction("(n,gamma)", "Fe57", Q=7.6e6)],
        yield_data=None,
    )
    co60 = SimpleNamespace(
        name="Co60", half_life=1.66e8, decay_energy=2.5e6,
        decay_modes=[_decay_mode("beta-", "Ni60")],
        sources={"photon": endf.univariate.Discrete([1.17e6, 1.33e6], [1.0, 1.0])},
        reactions=[],
        yield_data=None,
    )
    # U235 carries fission yields at two incident energies (thermal + fast).
    u235 = SimpleNamespace(
        name="U235", half_life=2.2e16, decay_energy=5.0e6,
        decay_modes=[_decay_mode("alpha", "Th231")],
        sources={},
        reactions=[_reaction("(n,gamma)", "U236", Q=6.5e6)],
        yield_data={
            0.0253: _fy_entry(["Xe135", "Cs137"], [0.06, 0.062]),
            5.0e5: _fy_entry(["Xe135", "Cs137"], [0.05, 0.064]),
        },
    )
    # U236 inherits U235's yields -> should become an alias row, not fy rows.
    u236 = SimpleNamespace(
        name="U236", half_life=7.4e14, decay_energy=4.6e6,
        decay_modes=[_decay_mode("alpha", "Th232")],
        sources={},
        reactions=[],
        yield_data=None,
        _fpy="U235",
    )
    return SimpleNamespace(nuclides=[fe56, co60, u235, u236])


def _read(path):
    with pa.memory_map(str(path), "r") as src:
        return pa.ipc.open_file(src).read_all()


def _load_json(path):
    return json.loads(path.read_text())


def test_full_layout(tmp_path):
    out = tmp_path / "transmutation_endfb-8.1.arrow"
    export_transmutation_to_arrow(
        _make_chain(), out, library="endfb-8.1", source="endf",
    )

    # Top-level manifest and the three subsection dirs exist; branching does not.
    assert (out / "manifest.json").is_file()
    for sub in ("decay", "reactions", "fission_yields"):
        assert (out / sub).is_dir()
        assert (out / sub / "provenance.json").is_file()
    assert not (out / "branching").exists()

    manifest = _load_json(out / "manifest.json")
    assert manifest["format_version"] == 2
    assert manifest["library"] == "endfb-8.1"
    # The subsection path values are load-bearing (consumers navigate by them),
    # so assert the full mapping, not just the keys.
    assert manifest["subsections"] == {
        "decay": {"path": "decay"},
        "reactions": {"path": "reactions"},
        "fission_yields": {"path": "fission_yields"},
    }


def test_decay_subsection(tmp_path):
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(_make_chain(), out, library="endfb-8.1", source="endf")

    nuclides = _read(out / "decay" / "nuclides.arrow")
    # Only decay-scoped columns; the old derived counts are gone.
    # `half_life_uncertainty` is nullable and last, so a file written without
    # it still reads (issue #515).
    assert nuclides.schema.names == [
        "name",
        "half_life",
        "decay_energy",
        "half_life_uncertainty",
        "decay_energy_uncertainty",
    ]
    assert nuclides.num_rows == 4
    by_name = {r["name"]: r for r in nuclides.to_pylist()}
    assert by_name["Fe56"]["half_life"] is None      # stable -> null
    assert by_name["Co60"]["half_life"] == pytest.approx(1.66e8)
    # The fixture states no uncertainty, and that must arrive as null rather
    # than 0.0: unstated and measured-to-be-negligible are different claims.
    assert by_name["Co60"]["half_life_uncertainty"] is None
    assert by_name["Co60"]["decay_energy_uncertainty"] is None

    # filetype metadata lives on the subsection's primary table.
    assert nuclides.schema.metadata[b"filetype"] == b"transmutation-decay"

    modes = _read(out / "decay" / "decay_modes.arrow").to_pylist()
    assert {"nuclide": "Co60", "type": "beta-", "target": "Ni60",
            "branching_ratio": 1.0} in modes

    sources = _read(out / "decay" / "sources.arrow").to_pylist()
    assert len(sources) == 1
    assert sources[0]["nuclide"] == "Co60"
    assert sources[0]["particle"] == "photon"
    assert sources[0]["type"] == "discrete"
    assert sources[0]["energies"] == [1.17e6, 1.33e6]
    assert sources[0]["intensities"] == [1.0, 1.0]

    # decay provenance content (not just existence).
    prov = _load_json(out / "decay" / "provenance.json")
    assert prov["subsection"] == "decay"
    assert prov["library"] == "endfb-8.1"
    assert prov["source"] == "endf"
    assert prov["converter_version"]
    assert prov["created_utc"]
    assert "branch_ratios_applied" not in prov   # reactions-only key


def test_mixture_source_flattening(tmp_path):
    """A Mixture source flattens to one row per component, scaled by probability."""
    nuc = SimpleNamespace(
        name="X1", half_life=1.0, decay_energy=1.0e6,
        decay_modes=[],
        sources={"electron": endf.univariate.Mixture(
            [0.3, 0.7],
            [endf.univariate.Discrete([1.0e6], [2.0]),
             endf.univariate.Discrete([2.0e6], [4.0])],
        )},
        reactions=[], yield_data=None,
    )
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(SimpleNamespace(nuclides=[nuc]), out,
                                  library="endfb-8.1", source="endf")

    rows = _read(out / "decay" / "sources.arrow").to_pylist()
    assert len(rows) == 2
    by_energy = {r["energies"][0]: r for r in rows}
    # intensities are the component intensity times the component probability.
    assert by_energy[1.0e6]["intensities"] == pytest.approx([2.0 * 0.3])
    assert by_energy[2.0e6]["intensities"] == pytest.approx([4.0 * 0.7])
    assert all(r["particle"] == "electron" and r["type"] == "discrete" for r in rows)


def test_reactions_subsection(tmp_path):
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(_make_chain(), out, library="endfb-8.1", source="endf",
                                  branch_ratios_applied=True)

    reactions = _read(out / "reactions" / "reactions.arrow")
    assert reactions.schema.names == ["nuclide", "type", "target", "Q", "branching_ratio"]
    rows = reactions.to_pylist()
    assert {"nuclide": "Fe56", "type": "(n,gamma)", "target": "Fe57",
            "Q": 7.6e6, "branching_ratio": 1.0} in rows

    prov = _load_json(out / "reactions" / "provenance.json")
    assert prov["subsection"] == "reactions"
    assert prov["library"] == "endfb-8.1"
    assert prov["source"] == "endf"
    assert prov["branch_ratios_applied"] is True
    assert prov["converter_version"]
    assert prov["created_utc"]


def test_fission_yields_subsection(tmp_path):
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(_make_chain(), out, library="endfb-8.1", source="endf")

    fy = _read(out / "fission_yields" / "fission_yields.arrow").to_pylist()
    # U235 has two incident energies -> two rows, distinguished by `energy`.
    assert len(fy) == 2
    assert {r["nuclide"] for r in fy} == {"U235"}
    by_energy = {r["energy"]: r for r in fy}
    assert set(by_energy) == {0.0253, 5.0e5}
    thermal = by_energy[0.0253]
    fast = by_energy[5.0e5]
    assert thermal["products"] == ["Xe135", "Cs137"]
    assert thermal["yields"] == pytest.approx([0.06, 0.062])
    assert fast["yields"] == pytest.approx([0.05, 0.064])

    # U236 inherits, so it is an alias, not its own yields.
    aliases = _read(out / "fission_yields" / "aliases.arrow").to_pylist()
    assert aliases == [{"nuclide": "U236", "fission_yield_parent": "U235"}]

    prov = _load_json(out / "fission_yields" / "provenance.json")
    assert prov["subsection"] == "fission_yields"
    assert prov["library"] == "endfb-8.1"
    assert prov["source"] == "endf"
    assert "branch_ratios_applied" not in prov


def test_structural_only_chain(tmp_path):
    """A nuclide with no decay/reactions/yields: primary tables are still written
    (empty), and the optional secondary tables are omitted."""
    fe56 = SimpleNamespace(
        name="Fe56", half_life=None, decay_energy=0.0,
        decay_modes=[], sources={}, reactions=[], yield_data=None,
    )
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(SimpleNamespace(nuclides=[fe56]), out,
                                  library="endfb-8.1", source="endf")

    # Primary tables are always present, even with zero rows.
    assert _read(out / "reactions" / "reactions.arrow").num_rows == 0
    assert _read(out / "fission_yields" / "fission_yields.arrow").num_rows == 0
    assert _read(out / "decay" / "nuclides.arrow").num_rows == 1

    # Optional secondary tables are omitted when there is nothing to write.
    assert not (out / "decay" / "decay_modes.arrow").exists()
    assert not (out / "decay" / "sources.arrow").exists()
    assert not (out / "fission_yields" / "aliases.arrow").exists()

    # The manifest still lists every emitted subsection.
    manifest = _load_json(out / "manifest.json")
    assert set(manifest["subsections"]) == {"decay", "reactions", "fission_yields"}


def test_subsection_subset(tmp_path):
    out = tmp_path / "t.arrow"
    export_transmutation_to_arrow(_make_chain(), out, library="endfb-8.1",
                                  source="endf", subsections=["decay"])

    assert (out / "decay").is_dir()
    assert not (out / "reactions").exists()
    assert not (out / "fission_yields").exists()
    manifest = _load_json(out / "manifest.json")
    assert manifest["subsections"] == {"decay": {"path": "decay"}}


def test_unknown_subsection_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown subsection"):
        export_transmutation_to_arrow(_make_chain(), tmp_path / "t.arrow",
                                      subsections=["branching"])
