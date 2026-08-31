"""Tests for standalone material.transmute() over a PulseSchedule.

Each irradiation Pulse carries a NeutronSource whose energy is a Histogram (the
spectrum); the pulse rate is the total flux magnitude [n/cm^2/s]. Verifies:
- Fe56 density decreases under irradiation
- Daughter nuclides appear after transmutation
- Stable isotopes remain constant during cooling steps
- a golden characterization of the exact per-step densities (the tight guard
  for behaviour-preserving refactors)
- per-pulse multi-spectrum schedules
"""

import pytest
import yamc

NUC_DATA = "tests"
CHAIN_FILE = "tests/transmutation-endf-b8.1-sfr.arrow"
DAY = 86400.0

# Simple 3-group flux: thermal / epithermal / fast.
ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
# Total flux magnitude. Histogram normalizes the spectrum shape, so a rate equal
# to sum(flux) reproduces the absolute per-group flux MULTIGROUP_FLUX exactly.
RATE = sum(MULTIGROUP_FLUX)


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _make_iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )


def _spectrum():
    """A NeutronSource carrying the 3-group flux shape as a Histogram."""
    return yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )


@pytest.fixture
def transmute_results():
    iron = _make_iron()
    spectrum = _spectrum()
    results = iron.transmute(
        schedule=yamc.PulseSchedule([
            yamc.Pulse(rate=RATE, duration=DAY, source=spectrum),
            yamc.Pulse(rate=RATE, duration=DAY, source=spectrum),
            yamc.Cooldown(duration=DAY),
            yamc.Cooldown(duration=DAY),
        ]),
    )
    return iron, results.step_materials(iron.id or 0)


def _get_density(material, nuclide):
    """Get the density of a nuclide from a material, or 0.0 if absent."""
    for name, density in material.nuclides:
        if name == nuclide:
            return density
    return 0.0


# --- validation: irradiation pulses must carry a Histogram-energy source ------

def test_transmute_requires_source_on_irradiation_pulse():
    # An irradiation Pulse needs a NeutronSource (the spectrum); a sourceless
    # Pulse is rejected. Use Cooldown for decay-only steps.
    iron = _make_iron()
    with pytest.raises(ValueError, match="must carry a NeutronSource"):
        iron.transmute(
            schedule=yamc.PulseSchedule([yamc.Pulse(rate=RATE, duration=DAY)]),
        )


def test_transmute_rejects_non_histogram_energy():
    # The pulse source energy must be a Histogram (binned groups are needed to
    # collapse the multigroup cross sections); a Discrete energy is rejected.
    iron = _make_iron()
    src = yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0]))
    with pytest.raises(ValueError, match="must be a Histogram"):
        iron.transmute(
            schedule=yamc.PulseSchedule([yamc.Pulse(rate=RATE, duration=DAY, source=src)]),
        )


# --- qualitative behaviour ----------------------------------------------------

def test_fe56_decreases_under_irradiation(transmute_results):
    """Fe56 density should decrease after irradiation steps."""
    initial_mat, results = transmute_results
    initial_fe56 = _get_density(initial_mat, "Fe56")
    after_irradiation_fe56 = _get_density(results[1], "Fe56")  # after 2 days
    assert initial_fe56 > 0
    assert after_irradiation_fe56 < initial_fe56, (
        f"Fe56 should decrease: initial={initial_fe56}, after={after_irradiation_fe56}"
    )


def test_daughter_nuclides_appear(transmute_results):
    """Daughter nuclides should appear after transmutation."""
    _, results = transmute_results
    final = results[-1]
    assert len(final.nuclides) > 1, (
        f"Expected daughter nuclides, got only {[n for n, _ in final.nuclides]}"
    )


def test_stable_isotopes_constant_during_cooling(transmute_results):
    """Stable isotopes should not change between consecutive cooling steps."""
    _, results = transmute_results
    # results[2] = after 1st cooling day, results[3] = after 2nd cooling day
    fe56_cool1 = _get_density(results[2], "Fe56")
    fe56_cool2 = _get_density(results[3], "Fe56")
    assert fe56_cool1 == pytest.approx(fe56_cool2, rel=1e-6), (
        f"Fe56 should be constant during cooling: "
        f"cool1={fe56_cool1}, cool2={fe56_cool2}"
    )


# --- golden characterization (tight numerical guard) --------------------------

# Exact per-step densities for the fixture schedule, captured from the validated
# implementation. Material.transmute is deterministic (multigroup collapse +
# Bateman, no Monte Carlo), so these are reproducible to machine precision; the
# tight tolerance below guards against any numerical regression.
_GOLDEN = [
    {  # step 0 (after 1 day irradiation)
        "Fe56": 0.08473086419379175, "Fe55": 1.4826824510346344e-07,
        "H1": 3.8525315306199334e-08, "Fe57": 8.812607323112645e-09,
        "Mn56": 5.95841861173335e-09, "Cr53": 3.3779833922452382e-09,
    },
    {  # step 1 (after 2 days irradiation)
        "Fe56": 0.08473070366015778, "Fe55": 2.9643370173356404e-07,
        "Fe57": 1.7625161441770162e-08, "Mn56": 5.967818791153792e-09,
    },
    {  # step 2 (after 1 cooling day)
        "Fe56": 0.08473070961855034, "Fe55": 2.9622876087100245e-07,
        "Mn55": 4.3892277114235013e-10, "Mn56": 9.426319839739035e-12,
    },
    {  # step 3 (after 2 cooling days)
        "Fe56": 0.08473070962796193, "Fe55": 2.960239616952899e-07,
        "Mn55": 6.437219468548537e-10,
    },
]


def test_transmute_golden(transmute_results):
    """Per-step densities match the captured golden values to ~machine precision."""
    _, results = transmute_results
    assert len(results) == len(_GOLDEN)
    for i, (mat, expected) in enumerate(zip(results, _GOLDEN)):
        for nuclide, want in expected.items():
            got = _get_density(mat, nuclide)
            assert got == pytest.approx(want, rel=1e-9), (
                f"step {i} {nuclide}: got {got!r}, want {want!r}"
            )


# --- per-pulse multi-spectrum -------------------------------------------------

def test_transmute_multi_spectrum_runs():
    """Different pulses may carry different spectra (e.g. a soft and a hard one)."""
    iron = _make_iron()
    thermal = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, [1.0, 0.0, 0.0]))
    fast = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, [0.0, 0.0, 1.0]))
    results = iron.transmute(
        schedule=yamc.PulseSchedule([
            yamc.Pulse(rate=RATE, duration=DAY, source=thermal),
            yamc.Pulse(rate=RATE, duration=DAY, source=fast),
            yamc.Cooldown(duration=DAY),
        ]),
    ).step_materials(iron.id or 0)
    assert len(results) == 3
    # Irradiation still burns Fe56 and breeds daughters.
    assert _get_density(results[1], "Fe56") < _get_density(_make_iron(), "Fe56")
    assert len(results[-1].nuclides) > 1


# --- the transmutable flag is a Model concern, not a Material.transmute() one --

def test_transmute_does_not_require_the_transmutable_flag():
    """``transmutable=True`` is only for ``Model.simulate_transmutation()``.

    That flag picks which materials in a geometry to deplete. ``transmute()`` is
    called on one material and depletes it, so requiring the flag would be a
    redundant line in every standalone script. Guards against it creeping back.
    """
    inert = _make_iron()
    assert inert.transmutable is False, "the fixture must not set the flag"

    flagged = _make_iron()
    flagged.transmutable = True

    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=DAY, source=_spectrum()),
        yamc.Cooldown(duration=DAY),
    ])
    without = inert.transmute(schedule=schedule).step_materials(inert.id or 0)
    with_flag = flagged.transmute(schedule=schedule).step_materials(flagged.id or 0)

    # Not merely "both ran": the flag must not perturb the answer either.
    assert len(without) == len(with_flag) == 2
    for step, (a, b) in enumerate(zip(without, with_flag)):
        assert dict(a.nuclides) == dict(b.nuclides), f"step {step} differs"


# --- per-edge reaction rates (issue #505) ------------------------------------

def test_transmute_returns_results_carrying_per_edge_rates():
    """``transmute()`` hands back the object whose methods it advertises.

    ``get_reaction_rates`` was registered and documented on the yani wheel but
    had no producer there: the only thing that built a ``TransmutationResults``
    was ``Model.simulate_transmutation``, and ``Model`` is yamc-only. This is
    the transport-free entry point reaching the same quantity.
    """
    iron = _make_iron()
    results = iron.transmute(
        schedule=yamc.PulseSchedule([
            yamc.Pulse(rate=RATE, duration=DAY, source=_spectrum()),
            yamc.Cooldown(duration=DAY),
        ]),
    )

    mat_id = iron.id or 0
    rates = results.get_reaction_rates(mat_id, 0)
    assert rates, "an irradiation step drove reactions"
    assert "Fe56" in rates, f"expected the seed nuclide as a parent, got {list(rates)}"

    # parent -> kind -> [(target, rate)], with the rate strictly positive.
    for kind, edges in rates["Fe56"].items():
        assert edges, f"{kind} has no edges"
        for target, rate in edges:
            assert target is None or isinstance(target, str)
            assert rate > 0.0, f"Fe56 {kind} -> {target} has rate {rate}"


def test_a_decay_only_step_drives_no_reactions():
    """A cooldown records an empty map rather than shifting the step indexing."""
    iron = _make_iron()
    results = iron.transmute(
        schedule=yamc.PulseSchedule([
            yamc.Pulse(rate=RATE, duration=DAY, source=_spectrum()),
            yamc.Cooldown(duration=DAY),
        ]),
    )

    mat_id = iron.id or 0
    assert results.get_reaction_rates(mat_id, 1) == {}
    # Step 2 does not exist: two schedule steps means indices 0 and 1.
    assert results.get_reaction_rates(mat_id, 2) is None


def test_step_materials_pairs_with_the_rate_index():
    """The composition getters offset by the initial state; the rates do not."""
    iron = _make_iron()
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=DAY, source=_spectrum()),
        yamc.Cooldown(duration=DAY),
    ])
    results = iron.transmute(schedule=schedule)

    mat_id = iron.id or 0
    steps = results.step_materials(mat_id)
    assert len(steps) == len(results.timesteps) == 2
    # get_material counts the initial composition as 0, so it runs one ahead.
    assert dict(steps[0].nuclides) == dict(results.get_material(mat_id, 1).nuclides)
    assert dict(steps[-1].nuclides) == dict(
        results.get_final_material(mat_id).nuclides
    )
