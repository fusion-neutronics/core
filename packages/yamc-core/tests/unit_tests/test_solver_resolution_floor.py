"""A decayed-away nuclide must not come back as decay heat (issue #410).

CRAM48's LU fill couples every row to the largest density in the solve, so a
component roughly 22 decades below it carries the solve's arithmetic rather than
an inventory. That is invisible in the inventory and loud in `decay_heat()`,
which weights by a decay constant spanning ten orders of magnitude: cooled
reactor graphite used to report two thirds of its decay heat from B12, a 20.2 ms
emitter, after a cooldown that should have reduced it by exp(-2.96e6).

The two halves of this file are the two directions the floor has to get right:
drop the residue after cooling, keep the same nuclide while it is really there.
"""

from pathlib import Path

import pytest

import yamc

_CHAIN = str(
    Path(__file__).resolve().parents[4] / "tests" / "transmutation-endf-b8.1-sfr.arrow"
)

# C12 (n,p) B12. 20.2 ms, and a ~13 MeV decay energy, which is what makes its
# residue outweigh every real emitter in the sample.
SHORT_LIVED = "B12"


def _graphite_steps():
    yamc.cross_section_data = "endf-b8.1"
    material = yamc.materials.pnnl.material(
        "Carbon, Graphite (reactor grade)", volume=1000.0, temperature=294
    )
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram([1e-5, 1e5, 1e6, 1.5e7], [1e12, 1e13, 1e14])
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1.11e14, duration=(1, "a"), source=spectrum),
            yamc.Cooldown(duration=(1, "d")),
        ]
    )
    results = material.transmute(schedule=schedule)
    return results.step_materials(material.id or 0)


@pytest.fixture(scope="module")
def graphite():
    return _graphite_steps()


@pytest.fixture(scope="module")
def chain():
    return yamc.TransmutationChain(_CHAIN)


def test_a_fully_decayed_nuclide_is_gone_after_cooling(graphite):
    cooled = dict(graphite[-1].nuclides)
    assert SHORT_LIVED not in cooled, (
        f"{SHORT_LIVED} survived 86400 s at a 20.2 ms half-life "
        f"({cooled.get(SHORT_LIVED):.3e} atoms/b-cm), which is the solver's "
        "floor, not an inventory"
    )


def test_the_residue_is_not_the_decay_heat(graphite):
    """The whole point: the heat has to come from nuclides that still exist."""
    per_nuclide = graphite[-1].decay_heat(by_nuclide=True)
    assert SHORT_LIVED not in per_nuclide
    total = graphite[-1].decay_heat()
    assert total == pytest.approx(sum(per_nuclide.values()), rel=1e-12)
    # H3 (12.3 y), Be10 (1.5 My) and C14 (5730 y) are what a cooled graphite
    # sample actually holds; one of them has to lead.
    hottest = max(per_nuclide, key=per_nuclide.get)
    assert hottest in {"H3", "Be10", "C14"}, f"unexpected leading emitter {hottest}"


def test_it_is_kept_while_it_is_really_there(graphite):
    """The floor is relative to the solve, not a blanket ban on short-lived
    nuclides. During irradiation B12 sits in equilibrium with its production and
    must survive the cut."""
    irradiated = dict(graphite[0].nuclides)
    assert SHORT_LIVED in irradiated
    assert irradiated[SHORT_LIVED] > 0.0


def test_equilibrium_short_lived_nuclides_survive_the_cut(graphite, chain):
    """A short-lived nuclide held in equilibrium by its own production is a real
    inventory, however fast it decays.

    This is the direction the cut is easiest to get wrong. Over a year-long
    pulse every sub-hour nuclide has a vanishing carry-over, so any rule phrased
    on half-life alone throws these away: graphite holds Li9 at 0.18 s and B12
    at 20 ms during the irradiation, both produced continuously, and both belong
    in the answer.
    """
    irradiated = dict(graphite[0].nuclides)
    short_lived = [
        (name, chain.half_lives[name])
        for name in irradiated
        if chain.half_lives.get(name) is not None and chain.half_lives[name] < 60.0
    ]
    assert short_lived, "the irradiated inventory should hold sub-minute nuclides"
    for name, half_life in short_lived:
        assert irradiated[name] > 0.0, (
            f"{name} (t1/2 {half_life} s) was produced throughout the pulse and "
            "must not be cut for being short-lived"
        )
