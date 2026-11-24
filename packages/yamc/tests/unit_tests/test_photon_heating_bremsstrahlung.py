"""Track-length photon heating with thick-target bremsstrahlung (issue #600).

The track-length coefficient deposits every electron's and positron's kinetic
energy at the collision. With the default ``electron_treatment="ttb"`` the
transport also banks their bremsstrahlung, which is tracked and scored again
where it is absorbed, so the track-length estimator counted it twice: in
tungsten at 6 MeV it deposited 12% more energy than the source emitted. The
collision estimator scores what each collision leaves behind and was right.

In a sphere thick enough to absorb everything the collision estimator returns
the source energy, so the two estimators must agree with either treatment.
"""

import pytest

import yamc

SOURCE_ENERGY = 6.0e6


def _heating(electron_treatment):
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.874, name="fe", temperature=294)
    material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    sphere = yamc.Sphere(radius=100.0, boundary="vacuum")
    cell = yamc.Cell(name="fe", region=sphere.below, material=material)
    geometry = yamc.Geometry(cells=[cell])
    tallies = [
        yamc.Tally(scores=["heating"], cells=cell, particle="photon", estimator=estimator, name=estimator)
        for estimator in ("track-length", "collision")
    ]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.PhotonSource(
            position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([SOURCE_ENERGY], [1.0])
        ),
        electron_treatment=electron_treatment,
        verbose=[],
    )
    results = model.simulate_transport(total_particles=40_000, seed=11)
    return {
        estimator: (float(results[estimator].mean[0]), float(results[estimator].standard_deviation[0]))
        for estimator in ("track-length", "collision")
    }


@pytest.mark.parametrize("electron_treatment", ["ttb", "local"])
def test_track_length_photon_heating_matches_the_collision_estimator(electron_treatment):
    heating = _heating(electron_treatment)
    collision, _ = heating["collision"]
    track, sigma = heating["track-length"]
    # Nothing leaks from a metre of iron, so every eV the source emits is
    # deposited.
    assert collision == pytest.approx(SOURCE_ENERGY, rel=2e-3)
    assert abs(track - collision) < max(4.0 * sigma, 3e-3 * collision), (track, collision, sigma)
