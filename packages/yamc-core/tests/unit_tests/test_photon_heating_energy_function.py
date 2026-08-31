"""Issues #378 / #382: photon heating must apply an energy function as a weight.

The analog photon-heat arm of the collision estimator scores an already-deposited
eV value. It used to be returned straight through, so an ``energy_function=``
dropped out-of-range collisions (the gate) but never scaled the ones it kept (the
weight), while the track-length arm applied both. The same tally therefore
answered differently based only on the estimator.

The fix separates the two factors that had been fused together: ``weight/Sigma_t``,
which converts a collision into a track-length equivalent and must not touch an eV
deposit, and ``f(E)``, the user's response function, which applies to every score.

These tests run real photon transport rather than poking the scorer, so they cover
the whole path from the transport dispatch through to the finalised tally.
"""

import pytest

import yamc

FLAT = 7.0
ENERGY_GRID = [1.0e3, 1.0e4, 1.0e5, 1.0e7]


def _model(estimator, energy_function=None, seed_particles=4000):
    """A photon-transporting iron sphere carrying one photon heating tally."""
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=5.0, boundary="vacuum")
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.0, temperature=294)
    material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry(cells=[cell])

    kwargs = dict(
        scores=["heating"], name="h", cells=cell, particle="photon", estimator=estimator
    )
    if energy_function is not None:
        kwargs["energy_function"] = energy_function
    tally = yamc.Tally(**kwargs)

    source = yamc.PhotonSource(
        position=(0.0, 0.0, 0.0),
        energy=yamc.sources.Discrete([1.0e6], [1.0]),
    )
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source, verbose=[])
    return model, tally, seed_particles


def _score(estimator, energy_function=None):
    model, tally, n = _model(estimator, energy_function)
    results = model.simulate_transport(total_particles=n, seed=1)
    return float(results[tally].mean[0])


def test_flat_energy_function_scales_collision_heating_exactly():
    """A constant response of c must scale the tally by exactly c.

    Same seed, and applying the weight adds no random draws, so the histories are
    identical and the two runs must be exactly proportional. That makes this an
    equality test rather than a statistical one.
    """
    plain = _score("collision")
    weighted = _score("collision", (ENERGY_GRID, [FLAT] * len(ENERGY_GRID)))
    assert plain > 0.0, "the model must actually deposit photon heat to be a test"
    assert weighted == pytest.approx(plain * FLAT, rel=1e-9), (
        f"a flat energy function of {FLAT} must scale collision-estimator photon "
        f"heating by exactly {FLAT}: {plain} -> {weighted}"
    )


def test_track_length_scales_the_same_way():
    """The track-length arm already weighted correctly and must be unchanged."""
    plain = _score("track-length")
    weighted = _score("track-length", (ENERGY_GRID, [FLAT] * len(ENERGY_GRID)))
    assert plain > 0.0
    assert weighted == pytest.approx(plain * FLAT, rel=1e-9)


def test_both_estimators_respond_to_the_energy_function_identically():
    """The property that motivates the fix, and #382's acceptance criterion.

    Absolute collision and track-length photon heating need not agree (they carry
    a known convention difference, see #356/#357), so this compares each
    estimator's RESPONSE to the energy function -- the weighted/unweighted ratio.
    That isolates the energy-function handling from everything else, and it is
    exactly what disagreed before: one estimator's ratio was f, the other's was 1.
    """
    ef = (ENERGY_GRID, [FLAT] * len(ENERGY_GRID))
    collision_ratio = _score("collision", ef) / _score("collision")
    tracklength_ratio = _score("track-length", ef) / _score("track-length")
    assert collision_ratio == pytest.approx(FLAT, rel=1e-9)
    assert tracklength_ratio == pytest.approx(FLAT, rel=1e-9)
    assert collision_ratio == pytest.approx(tracklength_ratio, rel=1e-9), (
        "the two estimators must respond identically to a response function; "
        f"collision scaled by {collision_ratio}, track-length by {tracklength_ratio}"
    )


def test_sloped_energy_function_is_not_a_constant_rescale():
    """A varying f(E) must land between its extremes, not on one of them.

    A flat table cannot tell "evaluated at the right energy" from "evaluated at
    some fixed energy", so this uses a table spanning three decades and checks the
    result sits strictly inside the range the table allows.
    """
    lo, hi = 1.0, 1000.0
    plain = _score("collision")
    sloped = _score("collision", (ENERGY_GRID, [lo, 10.0, 100.0, hi]))
    ratio = sloped / plain
    assert lo < ratio < hi, (
        f"a response running {lo} to {hi} must give a weighting strictly inside "
        f"that range, got {ratio}"
    )


def test_out_of_range_energy_function_scores_nothing():
    """The gate must survive the fix: off the table drops the whole event."""
    # A 1 MeV source cannot reach this window, so every collision is out of range.
    scored = _score("collision", ([1.0e-3, 1.0e-2, 1.0e-1, 1.0], [1.0, 1.0, 1.0, 1.0]))
    assert scored == 0.0, (
        "every collision lies outside the table, so the tally must be empty, "
        f"got {scored}"
    )
