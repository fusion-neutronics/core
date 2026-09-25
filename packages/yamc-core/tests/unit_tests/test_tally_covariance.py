"""``Tally(covariance=True)``: the covariance of the bin means (issue #140, item 6).

A per-bin standard deviation treats the bins as independent. Bins scored by
the same histories are not, and anything summed over them inherits the
correlation. The check that matters is against what the covariance claims to
predict: the spread, across independent seeds, of a sum over the bins must
match the variance the covariance gives that sum, off-diagonal terms and all,
where the standard deviations alone would not.
"""

import numpy as np
import pytest

import yamc

GROUPS = [1.0e-5, 1.0e2, 1.0e4, 1.0e5, 1.0e6, 3.0e6, 1.0e7, 2.0e7]


def _run(seed, *, covariance=True, particles=2000, bins=GROUPS):
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.87, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    cell = yamc.Cell(name="iron", region=sphere.below, material=iron)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([1.406e7], [1.0])
    )
    tally = yamc.Tally(
        name="spectrum", cells=cell, scores=["flux"], energy_bins=bins,
        covariance=covariance,
    )
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source, verbose=[]
    )
    return model.simulate_transport(total_particles=particles, seed=seed)["spectrum"]


def test_off_by_default():
    assert _run(1, covariance=False).covariance is None


def test_the_diagonal_is_the_standard_deviation_squared():
    r = _run(1)
    cov = np.array(r.covariance)
    n = len(r.mean)
    assert cov.shape == (n, n)
    assert np.allclose(np.diag(cov), np.array(r.standard_deviation) ** 2, rtol=1e-9)
    assert np.allclose(cov, cov.T)


def test_the_covariance_predicts_the_spread_of_a_sum():
    """64 seeds. The total over the spectrum rests on the cross terms: slowing
    down scores one history in many bins, so they are correlated.

    The measured variance over the predicted one is a chi-squared on 63
    degrees of freedom over 63, in [0.68, 1.36] 95% of the time; the seeds are
    fixed, and the bounds a little wider. 16 seeds is too few here: the
    thermal bins are scored rarely and heavy-tailed, and their noise swamps a
    16-sample variance.
    """
    totals, predicted, independent = [], [], []
    for seed in range(1, 65):
        r = _run(seed)
        cov = np.array(r.covariance)
        totals.append(sum(r.mean))
        predicted.append(cov.sum())
        independent.append(np.trace(cov))
    ratio = np.var(totals, ddof=1) / np.mean(predicted)
    assert 0.6 <= ratio <= 1.5, f"measured over predicted variance {ratio:.3f}"
    # The cross terms matter: here the bins anticorrelate (a history's path
    # length shared between energy bins trades one against another), so the
    # sum's variance sits well below the standard deviations' quadrature.
    assert abs(np.mean(predicted) / np.mean(independent) - 1.0) > 0.1


def test_a_spectrum_covariance_drives_a_flux_error():
    """What it is for: a transmutation pulse takes it as the flux's error."""
    r = _run(1)
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(GROUPS, r.mean))
    pulse = yamc.Pulse(
        rate=1.0e14, duration=3600.0, source=spectrum, flux_covariance=r.covariance
    )
    assert pulse.flux_covariance == r.covariance


def test_too_many_bins_is_refused():
    bins = list(np.logspace(-5, np.log10(2.0e7), 2100))
    with pytest.raises(Exception, match="limited to 2048 bins"):
        _run(1, bins=bins, particles=10)


def test_the_gpu_refuses_rather_than_ignoring_it():
    # Checked before any GPU is initialized, so this runs on any machine.
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.87, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    cell = yamc.Cell(name="iron", region=sphere.below, material=iron)
    geometry = yamc.Geometry([cell])
    tally = yamc.Tally(cells=cell, scores=["flux"], energy_bins=GROUPS, covariance=True)
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=yamc.NeutronSource(
            position=(0, 0, 0), energy=yamc.sources.Discrete([1.406e7], [1.0])
        ),
        verbose=[],
    )
    with pytest.raises(Exception, match="covariance=True needs the per-history"):
        model.simulate_transport(total_particles=100, compute="gpu")
