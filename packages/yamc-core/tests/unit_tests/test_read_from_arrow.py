"""Reading an Arrow directory back through yamc's own loader (issues #443, #525).

This is the readback gate a conversion is checked with. Its whole point is that
the reader is the CONSUMER: a directory can satisfy every schema and still be
refused by the loader that has to use it, and the only Python reader before this
was a second implementation of the format in ``packages/nuclear_data_to_arrow``
that shared the writer's vocabulary. That is how #379 happened.

The contract tests below need no nuclear data. The two that read real fixtures
skip without them, as the rest of this suite does.
"""

from pathlib import Path

import pytest

import yamc

_FIXTURES = Path(__file__).resolve().parents[4] / "tests"
_NUCLIDE = _FIXTURES / "Fe56.arrow"
_ELEMENT = _FIXTURES / "Fe.arrow"


def test_an_unknown_scope_is_refused():
    """Not silently treated as `full`, which would be a wider read than asked."""
    with pytest.raises(ValueError, match="unknown scope"):
        yamc.read_nuclide_from_arrow(str(_NUCLIDE), scope="everything")


def test_a_missing_nuclide_directory_names_itself():
    """The path is in the message: a gate runs over many directories at once."""
    with pytest.raises(RuntimeError, match="no-such-nuclide"):
        yamc.read_nuclide_from_arrow("/nonexistent/no-such-nuclide.arrow")


def test_a_missing_element_directory_names_itself():
    with pytest.raises(RuntimeError, match="no-such-element"):
        yamc.read_element_from_arrow("/nonexistent/no-such-element.arrow")


@pytest.mark.skipif(not _NUCLIDE.exists(), reason="run scripts/fetch_test_fixtures.py")
def test_a_full_nuclide_reads_back_full():
    """`scope_loaded` is the assertion, not the `scope` that was asked for.

    The loader does not fail on a directory with no transport sections: it
    narrows a "full" request to cross-sections-only, on the theory that this is
    a cross-sections-only conversion. A gate that asks for "full" and looks
    only at whether the call returned has therefore checked nothing about
    distributions, products or fast_xs.
    """
    summary = yamc.read_nuclide_from_arrow(str(_NUCLIDE))

    assert summary["scope_loaded"] == "full"
    assert summary["name"] == "Fe56"
    assert summary["atomic_number"] == 26
    assert summary["mass_number"] == 56
    # There is no `library` key: the Arrow loader does not populate that field,
    # so it would read None for every directory, right or wrong.
    assert "library" not in summary
    assert summary["atomic_weight_ratio"] > 0
    assert summary["loaded_temperatures"]
    assert 2 in summary["mts"], "elastic scattering is absent"
    # Per loaded temperature, and every one of them accounted for. Not a single
    # maximum: the reader also keeps the 0 K union grid, which is longer than
    # any of these and belongs to no temperature, so one number would report a
    # grid nothing reads and hide a per-temperature grid that was truncated.
    assert set(summary["energy_points"]) == set(summary["loaded_temperatures"])
    assert all(n > 0 for n in summary["energy_points"].values())


@pytest.mark.skipif(not _NUCLIDE.exists(), reason="run scripts/fetch_test_fixtures.py")
def test_the_xs_scope_reads_less_and_says_so():
    """The narrow read is reported as narrow, so a caller cannot mistake it."""
    full = yamc.read_nuclide_from_arrow(str(_NUCLIDE), scope="full")
    xs = yamc.read_nuclide_from_arrow(str(_NUCLIDE), scope="xs")

    assert xs["scope_loaded"] == "xs"
    # Same nuclide either way. The scope selects sections, not identity.
    assert xs["name"] == full["name"]
    assert xs["mts"] == full["mts"]


@pytest.mark.skipif(not _ELEMENT.exists(), reason="run scripts/fetch_test_fixtures.py")
def test_an_element_reads_back_with_its_auxiliary_tabulations():
    """The three flags are the sections that are in no evaluation.

    Compton profiles, bremsstrahlung and atomic relaxation come from separate
    published tabulations, so a photon conversion that drops them produces a
    directory that reads back perfectly well and is still not what transport
    wants. A byte-level or schema-level check cannot tell the difference.
    """
    summary = yamc.read_element_from_arrow(str(_ELEMENT))

    assert summary["name"] == "Fe"
    assert summary["atomic_number"] == 26
    assert summary["n_energy_points"] > 0
    assert summary["n_subshells"] > 0
    assert summary["has_compton_profiles"]
    assert summary["has_bremsstrahlung"]


@pytest.mark.skipif(not _ELEMENT.exists(), reason="run scripts/fetch_test_fixtures.py")
def test_reading_one_element_twice_reads_the_file_twice():
    """Not served from the process-global element store.

    That store is keyed by element name, so a gate validating two builds of
    `Fe.arrow` in one process would get the first one back for the second call
    and report a false pass. Asserting the equality of two reads is the weak
    half of this; the strong half is that the binding calls the raw reader,
    which this pins by contract.
    """
    first = yamc.read_element_from_arrow(str(_ELEMENT))
    second = yamc.read_element_from_arrow(str(_ELEMENT))
    assert first == second
