"""API gaps that made a supported thing unaskable (issues #339, #453, and part of #338).

None of these change what the engine does. They change whether a user can say
what they want: two entry points had no `compute=` at all, so the GPU was
unreachable by construction rather than by a refusal anyone could read, one flag
was hardcoded so a documented workaround named something Python could not do,
and a decay curve meant differencing a log series by hand.

The `gpu_fission_bank` part does NOT resolve #338. Turning the bank off is an
escape hatch, and the wanted outcome is the bank working WITH mesh tallies on
the GPU, which is a device-side change to the fissile per-source path. What is
fixed here is narrower and true regardless of that: the dispatch error told the
user to "disable the fission bank" using a flag Python had no way to reach.
"""

from pathlib import Path

import pytest
import yamc

_TESTS = Path(__file__).resolve().parents[4] / "tests"
HOUR = 3600.0
YEAR = 365.25 * 86400.0


# --------------------------------------------------------------------------
# #453: cooldown_steps
# --------------------------------------------------------------------------


def test_cooldown_steps_reproduces_the_hand_written_series():
    """The eight lines from the getting-started page, as one call.

    Cumulative times are what a user thinks in; `Cooldown` takes the duration of
    its own step, so the two differ by a difference.
    """
    n = 14
    steps = yamc.cooldown_steps(start=(1, "h"), stop=(10, "a"), n=n)
    assert len(steps) == n

    expected_cumulative = [HOUR * (10 * YEAR / HOUR) ** (k / (n - 1)) for k in range(n)]
    expected_gaps = [expected_cumulative[0]] + [
        expected_cumulative[k] - expected_cumulative[k - 1] for k in range(1, n)
    ]
    for step, want in zip(steps, expected_gaps):
        assert step.duration == pytest.approx(want, rel=1e-12)


def test_cooldown_steps_durations_sum_to_stop():
    """The first step runs from the end of irradiation to `start`, so the total
    is `stop` rather than `stop - start`."""
    steps = yamc.cooldown_steps(start=(1, "h"), stop=(10, "a"), n=14)
    assert sum(s.duration for s in steps) == pytest.approx(10 * YEAR, rel=1e-12)


def test_cooldown_steps_are_log_spaced_by_default():
    """Equal ratios between consecutive cumulative times, which is what makes
    the points evenly spaced on the log-log axes these are always plotted on."""
    steps = yamc.cooldown_steps(start=(1, "h"), stop=(10, "a"), n=8)
    cumulative, total = [], 0.0
    for s in steps:
        total += s.duration
        cumulative.append(total)
    ratios = [cumulative[k] / cumulative[k - 1] for k in range(1, len(cumulative))]
    assert all(r == pytest.approx(ratios[0], rel=1e-9) for r in ratios)


def test_cooldown_steps_linear_spacing():
    steps = yamc.cooldown_steps(start=(1, "d"), stop=(10, "d"), n=10, spacing="linear")
    assert sum(s.duration for s in steps) == pytest.approx(10 * 86400.0, rel=1e-12)
    # After the first step, every gap is the same.
    gaps = [s.duration for s in steps[1:]]
    assert all(g == pytest.approx(gaps[0], rel=1e-9) for g in gaps)


def test_cooldown_steps_accepts_plain_seconds():
    steps = yamc.cooldown_steps(start=1.0, stop=100.0, n=3)
    assert sum(s.duration for s in steps) == pytest.approx(100.0, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start": (1, "h"), "stop": (10, "a"), "n": 1}, "at least 2"),
        ({"start": (10, "a"), "stop": (1, "h"), "n": 5}, "must be after"),
        ({"start": 0.0, "stop": (10, "a"), "n": 5}, "positive for log spacing"),
        (
            {"start": (1, "h"), "stop": (10, "a"), "n": 5, "spacing": "quadratic"},
            "unknown spacing",
        ),
    ],
)
def test_cooldown_steps_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        yamc.cooldown_steps(**kwargs)


def test_cooldown_steps_zero_start_is_allowed_when_linear():
    """Only log spacing needs a positive start, since it takes a ratio from it."""
    steps = yamc.cooldown_steps(start=0.0, stop=(1, "d"), n=5, spacing="linear")
    assert sum(s.duration for s in steps) == pytest.approx(86400.0, rel=1e-12)


# --------------------------------------------------------------------------
# #338: gpu_fission_bank
# --------------------------------------------------------------------------


def _rod_model():
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.87, temperature=294)
    mat.read_nuclear_data({"Fe56": str(_TESTS / "Fe56.arrow")})
    sphere = yamc.Sphere(radius=5.0, boundary="vacuum")
    cell = yamc.Cell(name="c", region=sphere.below, material=mat)
    src = yamc.NeutronSource(energy=yamc.sources.Discrete([1.0e6], [1.0]))
    return yamc.Geometry([cell]), src


def test_gpu_fission_bank_defaults_on_and_is_reachable():
    geom, src = _rod_model()
    model = yamc.Model(geometry=geom, source=src)
    assert model.gpu_fission_bank is True, "the default must not change"

    off = yamc.Model(geometry=geom, source=src, gpu_fission_bank=False)
    assert off.gpu_fission_bank is False


def test_gpu_fission_bank_is_settable_after_construction():
    geom, src = _rod_model()
    model = yamc.Model(geometry=geom, source=src)
    model.gpu_fission_bank = False
    assert model.gpu_fission_bank is False


# --------------------------------------------------------------------------
# #339: compute= on the two entry points that lacked it
# --------------------------------------------------------------------------


def test_simulate_transmutation_names_the_reason_for_cpu_only():
    """Previously there was no `compute` argument at all, so asking was a
    TypeError about an unexpected keyword rather than an answer."""
    geom, src = _rod_model()
    model = yamc.Model(geometry=geom, source=src)
    schedule = yamc.PulseSchedule([yamc.Cooldown(duration=(1, "d"))])
    with pytest.raises(ValueError, match="only supports compute='cpu'"):
        model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=10, compute="gpu"
        )


def test_generate_weight_windows_names_the_reason_for_cpu_only():
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, -3], upper_right=[3, 3, 3], shape=[1, 1, 2]
    )
    geom, src = _rod_model()
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    model = yamc.Model(geometry=geom, source=src, tallies=[tally])
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron")
    with pytest.raises(ValueError, match="only supports compute='cpu'"):
        model.generate_weight_windows(gen, total_particles=100, compute="gpu")


def test_compute_cpu_is_not_the_thing_that_gets_refused():
    """The refusal must be about the VALUE, not about the argument existing.

    Asserted by what the error is NOT: with `compute="cpu"` these calls fail for
    their own reasons (this model carries no transmutation chain, and the
    generator wants a stop condition), and never with the compute rejection.
    Before this change the same call raised `TypeError: unexpected keyword
    argument 'compute'`, which is the state issue #339 is about.
    """
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, -3], upper_right=[3, 3, 3], shape=[1, 1, 2]
    )
    geom, src = _rod_model()
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    model = yamc.Model(geometry=geom, source=src, tallies=[tally])
    schedule = yamc.PulseSchedule([yamc.Cooldown(duration=(1, "d"))])

    with pytest.raises((ValueError, RuntimeError)) as caught:
        model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=10, compute="cpu"
        )
    assert "compute" not in str(caught.value)

    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron")
    with pytest.raises((ValueError, RuntimeError)) as caught:
        model.generate_weight_windows(gen, compute="cpu")
    assert "compute" not in str(caught.value)
