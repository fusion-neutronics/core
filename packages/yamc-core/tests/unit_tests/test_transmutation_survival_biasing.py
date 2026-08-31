"""Transport-coupled transmutation under survival biasing.

The transmutation reaction-rate tallies score weighted track length
(sigma x track x weight). Under survival biasing every collision
multiplies the neutron weight by the scattering probability, so an
unweighted score would systematically overestimate the transmutation
rates. This compares the daughter inventory after one irradiation step
between an analog run and a survival-biasing run of the same model:
both are unbiased estimators of the same rates, so they must agree
within statistics.
"""
import pytest
import yamc

CHAIN_FILE = "tests/transmutation-endf-b8.1-sfr.arrow"
DAY = 86400.0


def _transmute(survival_biasing):
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        volume=4188.79,  # 4/3 pi 10^3
        temperature=294)
    material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    # Thermal source in a thick iron sphere: sigma_a/sigma_t is ~18% per
    # collision over many collisions, so survival-biasing weights decay
    # GRADUALLY through (weight_cutoff, 1) -- the regime where an
    # unweighted transmutation score is biased high by the mean inverse
    # weight. (A 14 MeV source would not discriminate: histories leak
    # after a few collisions with weights still near 1.)
    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([0.0253], [1.0]))
    vr = [yamc.SurvivalBiasing()] if survival_biasing else []
    model = yamc.Model(
        geometry=geometry,
        source=source,
        verbose=[],
        variance_reduction=vr)
    return model.simulate_transmutation(
        method="independent",
        schedule=yamc.PulseSchedule([yamc.Pulse(rate=1.0e18, duration=DAY, source=source)]),
        total_particles=20_000,
        seed=42)


@pytest.mark.parametrize("daughter", ["Fe57"])
def test_survival_biasing_transmutation_matches_analog(daughter):
    analog = _transmute(survival_biasing=False)
    biased = _transmute(survival_biasing=True)

    mat_id = 1
    a = analog.get_nuclide_evolution(mat_id, daughter)
    b = biased.get_nuclide_evolution(mat_id, daughter)
    assert a is not None and b is not None, f"{daughter} missing from results"
    a_final, b_final = a[-1], b[-1]
    assert a_final > 0.0, f"analog produced no {daughter}"
    assert b_final > 0.0, f"survival-biasing run produced no {daughter}"

    # Both estimators are unbiased; 20k histories converge the dominant
    # channels to the percent level. An unweighted transmutation score under
    # survival biasing overestimates by the mean inverse weight, far
    # outside this tolerance.
    ratio = b_final / a_final
    assert 0.9 < ratio < 1.1, (
        f"{daughter}: survival-biasing/analog inventory ratio {ratio:.3f} "
        f"(analog {a_final:.4e}, biased {b_final:.4e} atoms/barn-cm)")
