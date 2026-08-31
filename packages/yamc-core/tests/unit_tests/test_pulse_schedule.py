"""PulseSchedule: decay-photon shutdown-dose-rate post-processing (#483).

PulseSchedule replaces the three verbatim openmc.deplete.d1s free functions
(get_radionuclides_from_chain / time_correction_factors / apply_time_correction)
with one object that owns the irradiation history. Each Pulse carries its own
source + rate + (value, unit) duration; one method, time_correct_tally, returns
a named DoseResult. Mixed-spectrum schedules are handled exactly (one transport
run per distinct source, summed).
"""
import numpy as np
import pytest

import yamc

CHAIN = "tests/transmutation-endf-b8.1-sfr.arrow"


def _dt_source():
    return yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )


def _broomstick_cell():
    mat = yamc.Material(composition={"Fe56": 1.0}, density=1.0, temperature=294)
    mat.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    cyl = yamc.Cylinder(axis="z", radius=5.0, boundary="vacuum")
    zb = yamc.Plane(axis="z", offset=-5.0, boundary="vacuum")
    zt = yamc.Plane(axis="z", offset=5.0, boundary="vacuum")
    return yamc.Cell(name="bs", region=cyl.below & zb.above & zt.below, material=mat)


def _run_decay_photons(source):
    cell = _broomstick_cell()
    geom = yamc.Geometry([cell])
    radionuclides = yamc.Model(geometry=geom, source=source).radionuclides()
    tally = yamc.Tally(
        scores=["flux"], name="dp", cells=cell, particle="photon",
        energy_group_structure="CCFE-24-PHOTON", parent_nuclides=radionuclides,
    )
    model = yamc.Model(
        geometry=geom, tallies=[tally], source=source,
        transport_secondary_photons=True, use_decay_photons=True, verbose=[],
    )
    return model.simulate_transport(total_particles=4000, seed=1)[tally]


# One transport run reused across the read-only assertions below.
@pytest.fixture(scope="module")
def dp_result():
    return _run_decay_photons(_dt_source())


def _sched(source):
    return yamc.PulseSchedule([
        yamc.Pulse(source=source, rate=1.0, duration=(1, "h")),
        yamc.Pulse(source=source, rate=1.0, duration=(1, "h")),
        yamc.Pulse(source=source, rate=1.0, duration=(1, "h")),
        yamc.Cooldown(duration=(15, "d")),
        yamc.Cooldown(duration=(15, "d")),
    ])


# --- duration parsing -------------------------------------------------------

def test_duration_units():
    s = _dt_source()
    assert yamc.Pulse(source=s, rate=1.0, duration=(1, "h")).duration == 3600.0
    assert yamc.Pulse(source=s, rate=1.0, duration=(2, "d")).duration == 2 * 86400.0
    assert yamc.Pulse(source=s, rate=1.0, duration=(1, "a")).duration == 365.25 * 86400.0
    assert yamc.Cooldown(duration=(30, "min")).duration == 1800.0
    # A plain number is seconds.
    assert yamc.Cooldown(duration=7200.0).duration == 7200.0


def test_unknown_unit_rejected():
    with pytest.raises(ValueError, match="unknown duration unit"):
        yamc.Cooldown(duration=(1, "fortnight"))


def test_pulse_rejects_non_neutron_source():
    with pytest.raises(TypeError, match="NeutronSource"):
        yamc.Pulse(source="not a source", rate=1.0, duration=(1, "h"))


def test_pulse_rejects_negative_rate():
    with pytest.raises(ValueError, match="rate must be non-negative"):
        yamc.Pulse(source=_dt_source(), rate=-1.0, duration=(1, "h"))


def test_pulse_source_optional_roundtrip():
    # A sourceless pulse (source=None) is what lets Material.transmute reuse
    # Pulse; its rate is preserved (it is the flux multiplier, not coerced to 0).
    p = yamc.Pulse(rate=0.5, duration=86400.0)
    assert p.source is None
    assert p.rate == 0.5
    assert p.duration == 86400.0
    # A sourced pulse keeps its source object.
    assert yamc.Pulse(rate=1e20, duration=(1, "d"), source=_dt_source()).source is not None


def test_empty_schedule_rejected():
    with pytest.raises(ValueError, match="at least one step"):
        yamc.PulseSchedule([])


# --- schedule structure -----------------------------------------------------

def test_schedule_len_and_sources():
    sched = _sched(_dt_source())
    assert len(sched) == 5
    assert len(sched.sources) == 1  # the three pulses share one source


# --- time correction (single source) ---------------------------------------

def test_dose_shape_times_and_decay(dp_result):
    sched = _sched(_dt_source())
    dose = sched.time_correct_tally(dp_result)
    mean = np.array(dose.mean)
    # 5 schedule steps; the pre-irradiation baseline is never included.
    assert mean.shape[0] == 5
    assert dose.times == [3600.0, 7200.0, 10800.0, 3 * 3600.0 + 15 * 86400.0,
                          3 * 3600.0 + 30 * 86400.0]
    totals = mean.sum(axis=1)
    # Activity builds up over the three irradiation steps...
    assert totals[0] < totals[1] < totals[2]
    # ...and decays away over the two long cooling steps.
    assert totals[3] < totals[2]
    assert totals[4] < totals[3]


def test_by_nuclide_breakdown(dp_result):
    sched = _sched(_dt_source())
    dose = sched.time_correct_tally(dp_result)
    # Fe56 activates to (among others) Mn56, a photon emitter.
    assert "Mn56" in dose.by_nuclide
    # Each nuclide breakdown has one row per selected step.
    assert np.array(dose.by_nuclide["Mn56"]).shape[0] == 5


def test_single_int_step_drops_dimension(dp_result):
    sched = _sched(_dt_source())
    dose_all = sched.time_correct_tally(dp_result)
    dose_last = sched.time_correct_tally(dp_result, steps=-1)
    # A bare int selects one step and drops the step dimension.
    assert np.array(dose_last.mean).ndim == 1
    assert np.allclose(dose_last.mean, np.array(dose_all.mean)[-1])
    assert isinstance(dose_last.times, float)


def test_step_subset(dp_result):
    sched = _sched(_dt_source())
    dose = sched.time_correct_tally(dp_result, steps=[0, 2])
    assert np.array(dose.mean).shape[0] == 2


def test_out_of_range_step_raises(dp_result):
    sched = _sched(_dt_source())
    with pytest.raises(IndexError):
        sched.time_correct_tally(dp_result, steps=99)


def test_tally_without_parent_nuclides_raises(dp_result):
    # A schedule needs a parent-nuclide-binned tally to time-correct.
    cell = _broomstick_cell()
    geom = yamc.Geometry([cell])
    src = _dt_source()
    plain = yamc.Tally(scores=["flux"], name="plain", cells=cell, particle="photon")
    model = yamc.Model(geometry=geom, tallies=[plain], source=src,
                       verbose=[])
    res = model.simulate_transport(total_particles=200, seed=1)[plain]
    with pytest.raises(ValueError, match="parent_nuclides"):
        _sched(src).time_correct_tally(res)


# --- mixed-spectrum (multi-source) ------------------------------------------

def test_multi_source_requires_dict(dp_result):
    s1 = _dt_source()
    s2 = _dt_source()
    sched = yamc.PulseSchedule([
        yamc.Pulse(source=s1, rate=1.0, duration=(1, "h")),
        yamc.Pulse(source=s2, rate=1.0, duration=(1, "h")),
        yamc.Cooldown(duration=(15, "d")),
    ])
    assert len(sched.sources) == 2
    with pytest.raises(ValueError, match="distinct sources"):
        sched.time_correct_tally(dp_result)  # single result is ambiguous


def test_multi_source_sum_equals_single(dp_result):
    """TCF is linear in source rate, so splitting one campaign's pulses across
    two distinct sources of the same spectrum must reproduce the single-source
    mean exactly (reusing the same tally result for both)."""
    s1 = _dt_source()
    s2 = _dt_source()
    split = yamc.PulseSchedule([
        yamc.Pulse(source=s1, rate=1.0, duration=(1, "h")),
        yamc.Pulse(source=s1, rate=1.0, duration=(1, "h")),
        yamc.Pulse(source=s2, rate=1.0, duration=(1, "h")),
        yamc.Cooldown(duration=(15, "d")),
        yamc.Cooldown(duration=(15, "d")),
    ])
    single_mean = np.array(_sched(_dt_source()).time_correct_tally(dp_result).mean)
    split_mean = np.array(split.time_correct_tally({s1: dp_result, s2: dp_result}).mean)
    assert np.allclose(split_mean, single_mean, rtol=1e-9)


# --- transmutation: schedule drives the timeline, model drives the source ----

def test_simulate_transmutation_rejects_multiple_distinct_sources():
    # simulate_transmutation honours only the model's configured source; a
    # schedule mixing two distinct Pulse sources is rejected (multi-spectrum
    # transmutation is deferred). The check fires before any transport runs.
    cell = _broomstick_cell()
    geom = yamc.Geometry([cell])
    src_a, src_b = _dt_source(), _dt_source()
    model = yamc.Model(geometry=geom, source=src_a, verbose=[])
    sched = yamc.PulseSchedule([
        yamc.Pulse(rate=1e18, duration=(1, "d"), source=src_a),
        yamc.Pulse(rate=1e18, duration=(1, "d"), source=src_b),
    ])
    with pytest.raises(ValueError, match="multiple distinct Pulse sources"):
        model.simulate_transmutation(method="coupled", schedule=sched, total_particles=1)
