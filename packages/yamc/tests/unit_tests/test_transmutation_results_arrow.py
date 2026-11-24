"""TransmutationResults.to_arrow / from_arrow.

A saved result must load back answering every getter as before, bit for bit,
and save again to the same bytes. The file pins rather than stores the chain and
the initial composition's cross sections: when the configured data is not what
the run read, the three getters that re-derive from it raise instead of
answering from other data, and a warning says why.
"""

import shutil
import warnings
from pathlib import Path

import pytest
import yamc

FIXTURES = Path(__file__).resolve().parents[4] / "tests"
DAY = 86400.0
ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
FLUX = [1e12, 5e12, 1e14]


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.neutron_cross_section_data = str(FIXTURES)
    yield
    yamc.neutron_cross_section_data = None


def _solve(cross_sections=None):
    iron = yamc.Material(
        composition={"Fe56": 0.9, "Fe54": 0.1},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )
    if cross_sections is not None:
        iron.read_nuclear_data(cross_sections)
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, FLUX))
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=sum(FLUX), duration=DAY, source=spectrum),
            yamc.Pulse(rate=0.5 * sum(FLUX), duration=DAY, source=spectrum),
            yamc.Cooldown(duration=DAY),
        ]
    )
    return iron.id, iron.transmute(schedule=schedule)


def _answers(results, mid):
    """Everything the getters say about one material, for comparison."""
    steps = results.num_steps
    out = {
        "times": results.times,
        "timesteps": results.timesteps,
        "ids": results.material_ids,
        "releases": results.data_releases,
        "gaps": results.data_gaps,
        "source": results.get_source_rates(mid),
    }
    for k in range(steps + 1):
        out[f"nuclides {k}"] = results.get_material_nuclides(mid, k)
        out[f"material {k}"] = results.get_material(mid, k).get_atoms_per_barn_cm()
    for k in range(steps):
        out[f"rates {k}"] = results.get_reaction_rates(mid, k)
        out[f"branching {k}"] = results.get_isomeric_branching(mid, k)
        out[f"routes {k}"] = results.get_production_routes(mid, "Fe55", k)
        for nuclide, kind in [
            ("Fe56", "(n,gamma)"),
            ("Fe54", "(n,p)"),
            ("Fe56", "(n,2n)"),
        ]:
            out[f"spectrum {nuclide} {kind} {k}"] = results.get_reaction_rate_spectrum(
                mid, nuclide, kind, k
            )
    return out


def _load(source):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = yamc.TransmutationResults.from_arrow(source)
    return loaded, [str(w.message) for w in caught]


def test_a_saved_result_answers_as_before(tmp_path):
    mid, results = _solve()
    before = _answers(results, mid)
    assert before["routes 0"], "the test needs a route to compare"
    assert any(v for k, v in before.items() if k.startswith("spectrum")), (
        "and a spectrum"
    )

    path = tmp_path / "part.arrow"
    assert results.to_arrow(path) is None
    data = results.to_arrow()
    assert isinstance(data, bytes) and path.read_bytes() == data

    for source in (path, str(path), data):
        loaded, caught = _load(source)
        assert caught == [], "the same data is configured, so everything reattaches"
        # Equal floats, not close ones: the file stores every bit.
        assert _answers(loaded, mid) == before
        assert loaded.to_arrow() == data, "a loaded result saves to the same bytes"


def test_the_chain_is_pinned_by_its_contents(tmp_path, monkeypatch):
    """The run reads a copy of the chain that is gone by the load; the
    configured original is the same data, so it is accepted."""
    chain = tmp_path / "chain"
    shutil.copytree(yamc.transmutation_decay_data, chain)
    original = yamc.transmutation_decay_data
    for name in (
        "transmutation_decay_data",
        "transmutation_reactions",
        "transmutation_fission_yields",
    ):
        monkeypatch.setattr(yamc, name, str(chain))
    mid, results = _solve()
    before = _answers(results, mid)
    data = results.to_arrow()

    for name in (
        "transmutation_decay_data",
        "transmutation_reactions",
        "transmutation_fission_yields",
    ):
        monkeypatch.setattr(yamc, name, original)
    shutil.rmtree(chain)
    loaded, caught = _load(data)
    assert caught == []
    assert _answers(loaded, mid) == before


def test_a_chain_that_cannot_be_given_back_makes_its_getters_raise(monkeypatch):
    mid, results = _solve()
    before = _answers(results, mid)
    data = results.to_arrow()

    # The load cannot assemble the chain the run read: the reactions are unset.
    monkeypatch.setattr(yamc, "transmutation_reactions", None)
    loaded, caught = _load(data)
    assert len(caught) == 1 and "chain" in caught[0], caught
    for getter, args in [
        (loaded.get_production_routes, (mid, "Fe55", 0)),
        (loaded.get_isomeric_branching, (mid, 0)),
    ]:
        with pytest.raises(RuntimeError, match=getter.__name__):
            getter(*args)
    # The cross sections were read again from their files, and the rest is
    # what the file holds.
    assert (
        loaded.get_reaction_rate_spectrum(mid, "Fe56", "(n,gamma)", 0)
        == before["spectrum Fe56 (n,gamma) 0"]
    )
    assert loaded.get_reaction_rates(mid, 1) == before["rates 1"]
    assert loaded.to_arrow() == data


def test_missing_data_is_warned_about_and_only_those_getters_refuse(tmp_path):
    # The cross sections come from a folder that is gone by the time of the load.
    folder = tmp_path / "xs"
    folder.mkdir()
    for nuclide in ("Fe56", "Fe54"):
        shutil.copytree(FIXTURES / f"{nuclide}.arrow", folder / f"{nuclide}.arrow")
    mid, results = _solve({n: str(folder / f"{n}.arrow") for n in ("Fe56", "Fe54")})
    before = _answers(results, mid)
    data = results.to_arrow()
    shutil.rmtree(folder)

    loaded, caught = _load(data)
    assert len(caught) == 1 and "get_reaction_rate_spectrum" in caught[0], caught
    with pytest.raises(RuntimeError, match="get_reaction_rate_spectrum"):
        loaded.get_reaction_rate_spectrum(mid, "Fe56", "(n,gamma)", 0)
    # The chain was still there, so its two getters answer, as does the rest.
    assert loaded.get_production_routes(mid, "Fe55", 0) == before["routes 0"]
    assert loaded.get_material_nuclides(mid, 3) == before["nuclides 3"]
    # Saving the loaded result keeps the pin, so a later load with the data
    # back would reattach it.
    assert loaded.to_arrow() == data


def test_a_wrong_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a yani transmutation results file"):
        yamc.TransmutationResults.from_arrow(b"not an arrow file")


def test_parts_scan_as_one_table(tmp_path):
    """Results saved one per part read back as one table with plain pyarrow."""
    ds = pytest.importorskip("pyarrow.dataset")
    mid, results = _solve()
    for part in range(2):
        results.to_arrow(tmp_path / f"part_{part}.arrow")
    table = ds.dataset(str(tmp_path), format="ipc").to_table()
    assert table.column_names == [
        "material_id",
        "material_name",
        "step",
        "quantity",
        "nuclide",
        "reaction",
        "product",
        "value",
    ]
    final = [
        r
        for r in table.to_pylist()
        if r["quantity"] == "density"
        and r["step"] == results.num_steps
        and r["nuclide"] == "Fe56"
    ]
    assert [r["value"] for r in final] == [
        results.get_material_nuclides(mid, results.num_steps)["Fe56"]
    ] * 2
    assert {r["material_name"] for r in final} == {"iron"}
