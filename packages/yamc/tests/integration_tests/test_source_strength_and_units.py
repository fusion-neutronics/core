"""with_source_strength and TallyResult.units / to (core#710).

A Li6 sphere under a 1 MeV point source, tallied in its one cell (which has
a volume) and on a mesh, so every conversion has a volume to divide by.
"""

import math

import pytest
import yamc

RADIUS = 10.0
VOLUME = 4.0 / 3.0 * math.pi * RADIUS**3
J_PER_EV = 1.602176634e-19


@pytest.fixture(scope="module")
def results():
    material = yamc.Material(composition={"Li6": 1.0}, density=2.0, temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    sphere = yamc.Sphere(radius=RADIUS, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material, volume=VOLUME)
    geometry = yamc.Geometry([cell])
    mesh = yamc.RegularRectangularMesh([-RADIUS] * 3, [RADIUS] * 3, [2, 1, 1])
    tallies = [
        yamc.Tally(name="heating", scores=["heating"], cells=cell),
        yamc.Tally(name="flux", scores=["flux"], cells=cell),
        yamc.Tally(name="mesh", scores=["heating"], mesh=mesh),
        yamc.Tally(name="both", scores=["flux", "heating"], cells=cell),
    ]
    source = yamc.NeutronSource(position=(0, 0, 0), energy=yamc.sources.Discrete([1e6], [1]))
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source, verbose=[])
    return model.simulate_transport(total_particles=2000, seed=1, threads=1)


def test_strength_scales_mean_and_std_not_relative_error(results):
    raw = results["heating"]
    scaled = results.with_source_strength(5e15)["heating"]
    assert scaled.mean[0] == pytest.approx(raw.mean[0] * 5e15)
    assert scaled.standard_deviation[0] == pytest.approx(raw.standard_deviation[0] * 5e15)
    assert scaled.relative_error == raw.relative_error
    assert scaled.statistical_checks.summary() == raw.statistical_checks.summary()
    assert results.source_strength is None, "the original is left unchanged"


def test_strength_replaces_rather_than_multiplies(results):
    twice = results.with_source_strength(5e15).with_source_strength(2.0)
    assert twice.source_strength == (2.0, "second")
    assert twice["heating"].mean[0] == pytest.approx(results["heating"].mean[0] * 2.0)
    back = twice.with_source_strength(None)
    assert back.source_strength is None
    assert back["heating"].mean == results["heating"].mean
    assert "source_strength=5e15 per second" in repr(results.with_source_strength(5e15))


def test_units_follow_the_strength(results):
    assert results["heating"].units == ["eV/source-particle"]
    assert results.with_source_strength(1.0)["heating"].units == ["eV/s"]
    assert results.with_source_strength(1.0, per="pulse")["flux"].units == ["cm/pulse"]
    assert results["both"].units == ["cm/source-particle", "eV/source-particle"]


def test_heating_to_watts_and_watts_per_cm3(results):
    eV = results["heating"].mean[0]
    heating = results.with_source_strength(5e15)["heating"]
    assert heating.to("W").mean[0] == pytest.approx(eV * 5e15 * J_PER_EV)
    w_cm3 = heating.to("W/cm3")
    assert w_cm3.units == ["W/cm3"]
    assert w_cm3.mean[0] == pytest.approx(eV * 5e15 * J_PER_EV / VOLUME)
    assert heating.to("W/m3").mean[0] == pytest.approx(w_cm3.mean[0] * 1e6)
    assert w_cm3.relative_error == heating.relative_error


def test_conversion_without_strength_is_per_source_particle(results):
    mev = results["heating"].to("MeV")
    assert mev.units == ["MeV/source-particle"]
    assert mev.mean[0] == pytest.approx(results["heating"].mean[0] * 1e-6)


def test_mesh_bins_divide_by_their_own_volume(results):
    mesh = results.with_source_strength(1.0)["mesh"]
    volumes = mesh.bin_volumes
    assert volumes == pytest.approx([4000.0, 4000.0])
    per_cm3 = mesh.to("eV/(cm3 s)")
    assert per_cm3.mean == pytest.approx([m / v for m, v in zip(mesh.mean, volumes)])


def test_flux_per_area(results):
    flux = results.with_source_strength(1e10)["flux"].to("1/(cm2 s)")
    assert flux.mean[0] == pytest.approx(results["flux"].mean[0] * 1e10 / VOLUME)
    assert flux.to_numpy("mean").shape == tuple(flux.shape)


def test_one_unit_per_score(results):
    both = results.with_source_strength(1.0)["both"].to(["m/s", "W"])
    assert both.units == ["m/s", "W"]
    raw = results["both"].mean
    assert both.mean == pytest.approx([raw[0] / 100.0, raw[1] * J_PER_EV])
    with pytest.raises(ValueError, match="2 scores"):
        results["both"].to(["W", "W", "W"])


def test_errors_say_what_is_missing(results):
    with pytest.raises(ValueError, match="with_source_strength"):
        results["heating"].to("W")
    with pytest.raises(ValueError, match="per pulse"):
        results.with_source_strength(1e19, per="pulse")["heating"].to("W")
    with pytest.raises(ValueError, match="per second"):
        results.with_source_strength(1.0)["heating"].to("J")
    with pytest.raises(ValueError, match="unknown unit"):
        results["heating"].to("furlong")
    with pytest.raises(ValueError, match="cannot convert"):
        results.with_source_strength(1.0)["flux"].to("W")
    with pytest.raises(ValueError, match="positive"):
        results.with_source_strength(-1.0)
    with pytest.raises(ValueError, match="per="):
        results.with_source_strength(1.0, per="hour")


def test_per_pulse_energy(results):
    shot = results.with_source_strength(1e19, per="pulse")["heating"].to("J/pulse")
    assert shot.mean[0] == pytest.approx(results["heating"].mean[0] * 1e19 * J_PER_EV)


def test_aggregate_values_scale_until_bins_differ_in_volume(results):
    heating = results.with_source_strength(2.0)["heating"]
    assert heating.aggregate_mean == pytest.approx(results["heating"].aggregate_mean * 2.0)
    mesh = results.with_source_strength(1.0)["mesh"]
    # Every voxel of this mesh has the same volume, so the scale is uniform.
    assert mesh.to("eV/(cm3 s)").aggregate_mean == pytest.approx(mesh.aggregate_mean / 4000.0)


def test_arrow_round_trip_keeps_strength_and_units(results):
    converted = results.with_source_strength(5e15)
    back = yamc.TransportResults.from_arrow(converted.to_arrow())
    assert back.source_strength == (5e15, "second")
    assert back["heating"].mean == pytest.approx(converted["heating"].mean)
    assert back["heating"].to("W/cm3").mean == pytest.approx(
        converted["heating"].to("W/cm3").mean
    )


def test_to_numpy_uses_the_export_field_names(results):
    r = results.with_source_strength(3.0)["heating"]
    assert r.to_numpy("std_dev").tolist() == r.standard_deviation
    assert r.to_numpy("rel_err").tolist() == r.relative_error
    assert r.to_numpy("count").tolist() == r.total_count
    with pytest.raises(ValueError, match="std_dev"):
        r.to_numpy("standard_deviation")
