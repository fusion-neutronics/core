"""Model.set_material_temperature / set_material_density: changing a material on
an existing model and running again matches a model built at the new state."""

import pytest

import yamc


def _iron(temperature=294, density=7.8):
    iron = yamc.Material(composition={"Fe56": 1.0}, density=density, temperature=temperature)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    return iron


def _model(iron):
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=iron)
    geometry = yamc.Geometry([cell])
    tally = yamc.Tally(scores=["flux"], cells=cell)
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(energy=14.06e6),
        tallies=[tally],
        verbose=[],
    )
    return model, tally


def _flux(model, tally):
    return list(model.simulate_transport(total_particles=2000, seed=7)[tally].mean)


def test_temperature_matches_a_fresh_model():
    iron = _iron()
    model, tally = _model(iron)
    before = _flux(model, tally)
    model.set_material_temperature(iron, 900.0)
    updated = _flux(model, tally)
    assert updated != pytest.approx(before, rel=1e-9)
    assert updated == pytest.approx(_flux(*_model(_iron(temperature=900))), rel=1e-12)
    assert iron.temperature == "294"


def test_density_matches_a_fresh_model():
    iron = _iron()
    model, tally = _model(iron)
    before = _flux(model, tally)
    model.set_material_density(iron, 3.9)
    updated = _flux(model, tally)
    assert updated != pytest.approx(before, rel=1e-9)
    assert updated == pytest.approx(_flux(*_model(_iron(density=3.9))), rel=1e-12)


def test_bad_requests_raise():
    iron = _iron()
    model, _ = _model(iron)
    with pytest.raises(ValueError, match="no material with id"):
        model.set_material_temperature(_iron(), 600.0)
    with pytest.raises(ValueError, match="positive finite"):
        model.set_material_density(iron, -1.0)
    with pytest.raises(ValueError, match="finite"):
        model.set_material_temperature(iron, float("nan"))
