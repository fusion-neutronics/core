"""The built-in group structures, read as data rather than passed as a name.

`group_structure(name)` hands back the edges behind a name like `"CCFE-709"`,
which is what lets a caller relate a multigroup spectrum to the energies it
belongs to (issue #492). These tests pin the registry contents, the shape of
what comes back, and that it is the same data the solver bins on.
"""

import pytest

import yamc

# Neutron first by increasing group count, then photon: the order the registry
# lists them in, which is the order the error message and the docs table use.
EXPECTED = [
    "XMAS-172",
    "VITAMIN-J-175",
    "SCALE-252",
    "TRIPOLI-315",
    "SHEM-361",
    "LLNL-616",
    "CCFE-709",
    "SCALE-999",
    "UKAEA-1102",
    "ECCO-1968",
    "CCFE-24-PHOTON",
    "VITAMIN-J-42",
]


def test_names_are_the_registry():
    assert yamc.group_structure_names() == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_every_name_resolves_to_usable_edges(name):
    """Ascending, positive, and as many groups as the name claims."""
    edges = yamc.group_structure(name)

    groups = int(name.rsplit("-")[-1] if name != "CCFE-24-PHOTON" else 24)
    assert len(edges) == groups + 1

    assert edges[0] >= 0.0
    assert all(hi > lo for lo, hi in zip(edges, edges[1:]))


@pytest.mark.parametrize("name", EXPECTED)
def test_edges_match_what_the_histogram_bins_on(name):
    """The function and the `boundaries` argument are the same data.

    Reading the edges off a throwaway `Histogram` was the only way to get at
    them before this function existed, so the two must agree, or the plotted
    energies would not be the ones the source sampled.
    """
    edges = yamc.group_structure(name)
    histogram = yamc.sources.Histogram(name, [1.0] * (len(edges) - 1))
    assert histogram.boundaries == edges


@pytest.mark.parametrize("name", ["VITAMIN-J-175", "UKAEA-1102"])
def test_edges_match_what_the_tally_bins_on(name):
    """Same data again on the tally side, where a spectrum is actually scored."""
    edges = yamc.group_structure(name)
    tally = yamc.Tally(scores=["flux"], energy_group_structure=name)
    assert tally.energy_bins == edges
    assert tally.n_energy_bins == len(edges) - 1


def test_ukaea_1102():
    """The 1102-group UKAEA structure: 1e-5 eV to 1 GeV, DT peak in 14.0-14.2 MeV."""
    edges = yamc.group_structure("UKAEA-1102")
    assert len(edges) == 1103
    assert edges[0] == 1e-5
    assert edges[-1] == 1e9

    i = max(k for k, e in enumerate(edges) if e <= 14.1e6)
    assert (edges[i], edges[i + 1]) == (14.0e6, 14.2e6)


def test_dt_peak_is_inside_every_neutron_structure():
    """A structure that stops below 14.1 MeV cannot hold a DT source spectrum."""
    for name in yamc.group_structure_names():
        if "PHOTON" in name or name == "VITAMIN-J-42":
            continue
        assert yamc.group_structure(name)[-1] > 14.1e6, name


def test_returned_list_is_a_copy():
    """Mutating what comes back must not corrupt the next caller's edges."""
    edges = yamc.group_structure("VITAMIN-J-175")
    edges[0] = -1.0
    assert yamc.group_structure("VITAMIN-J-175")[0] == 1e-5


def test_unknown_name_lists_the_known_ones():
    with pytest.raises(ValueError, match="Unknown group structure: 'CASMO-70'"):
        yamc.group_structure("CASMO-70")
    with pytest.raises(ValueError, match="CCFE-709"):
        yamc.group_structure("CASMO-70")


def test_names_are_case_sensitive():
    with pytest.raises(ValueError, match="Unknown group structure"):
        yamc.group_structure("ukaea-1102")
