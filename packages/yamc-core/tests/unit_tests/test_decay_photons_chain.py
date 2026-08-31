"""Tests for D1S chain data access (half-lives, reactions, decays, photon sources)."""

from pathlib import Path

import pytest
import yamc

CHAIN_FILE = str(
    Path(__file__).resolve().parents[4]
    / "crates"
    / "yamc"
    / "tests"
    / "transmutation-endf-b8.1-sfr.arrow"
)


def test_chain_half_lives():
    """Unstable nuclides appear with sensible half-lives; stable ones do not."""
    hl = yamc.TransmutationChain(CHAIN_FILE).half_lives

    # Unstable
    assert "Mn56" in hl
    assert "Co60" in hl
    assert "Fe55" in hl

    # Stable
    assert "Fe56" not in hl
    assert "Fe58" not in hl

    # Known ENDF/B-VIII.1 half-lives (seconds)
    assert abs(hl["Mn56"] - 9284.04) < 10.0
    assert abs(hl["Co60"] - 1.66344e8) < 1e5
    assert abs(hl["Fe55"] - 8.6594e7) < 1e4


def test_chain_decay_energies():
    """Decay energies are exposed in eV for decay-heat calculations."""
    energies = yamc.TransmutationChain(CHAIN_FILE).decay_energies

    assert energies["Mn56"] == pytest.approx(2522640.3)
    assert energies["Co60"] == pytest.approx(2600613.1)
    assert "Fe56" not in energies


def test_chain_reactions():
    """Reactions are loaded with (kind, target, branching) tuples."""
    rxns = yamc.TransmutationChain(CHAIN_FILE).reactions

    assert "Fe56" in rxns
    fe56 = [(r[0], r[1]) for r in rxns["Fe56"]]
    assert ("(n,gamma)", "Fe57") in fe56
    assert ("(n,2n)", "Fe55") in fe56
    assert ("(n,p)", "Mn56") in fe56


def test_chain_photon_sources():
    """Decay photon sources attach to nuclides that have decay gamma data."""
    sources = yamc.TransmutationChain(CHAIN_FILE).photon_sources

    assert "Mn56" in sources
    assert "Co60" in sources
    assert "Fe56" not in sources

    mn56 = sources["Mn56"]
    assert len(mn56) >= 1
    energies, intensities = mn56[0]
    assert len(energies) == len(intensities)
    assert len(energies) > 0
    assert all(e > 0 for e in energies)
    assert all(i > 0 for i in intensities)


def test_load_nonexistent_chain():
    """Loading a nonexistent chain path raises an error."""
    with pytest.raises(OSError):
        yamc.TransmutationChain("/nonexistent/chain.arrow")


def test_chain_decays():
    """Decay modes are exposed with the same (kind, target, branching) shape."""
    chain = yamc.TransmutationChain(CHAIN_FILE)
    decays = chain.decays

    assert decays["Mn56"] == [("beta-", "Fe56", 1.0)]
    assert decays["Co60"] == [("beta-", "Ni60", 1.0)]

    # Stable nuclides have no decay modes and are left out, as they are from
    # half_lives.
    assert "Fe56" not in decays
    assert set(decays) < set(chain.nuclide_names)

    # Branchings for one parent partition it.
    for name, modes in decays.items():
        assert sum(branching for _, _, branching in modes) == pytest.approx(1.0), name


def test_chain_decays_complete_a_two_step_route():
    """The point of the getter: a route whose second step is a decay.

    W186(n,a)Hf183(beta-)Ta183. With `reactions` alone a walk stops at Hf183 and
    the Ta183 it produces is attributed to nothing.
    """
    chain = yamc.TransmutationChain(CHAIN_FILE)

    step1 = [(kind, target) for kind, target, _ in chain.reactions["W186"]]
    assert ("(n,a)", "Hf183") in step1

    step2 = [(kind, target) for kind, target, _ in chain.decays["Hf183"]]
    assert ("beta-", "Ta183") in step2


def test_chain_isomer_decays_to_its_ground_state():
    """An IT edge, which is the one a naming convention could almost fake."""
    decays = yamc.TransmutationChain(CHAIN_FILE).decays

    assert decays["W185_m1"] == [("IT", "W185", 1.0)]
    assert decays["Ta182_m1"] == [("IT", "Ta182", 1.0)]


def test_chain_absent_target_is_none_not_the_string():
    """A channel with no named product carries None, in both getters.

    The declared type has always been `str | None`; `reactions` used to put the
    string "None" there, which is a nuclide name that matches nothing.
    """
    chain = yamc.TransmutationChain(CHAIN_FILE)

    fission = [e for e in chain.reactions["U235"] if e[0] == "fission"]
    assert fission == [("fission", None, 1.0)]

    assert chain.decays["He5"] == [("alpha", None, 1.0)]

    for source in (chain.reactions, chain.decays):
        for edges in source.values():
            for _, target, _ in edges:
                assert target is None or isinstance(target, str)
                assert target != "None"
