"""Round-trip tests for the run-control and label setters on Model, Cell, and
Tally. These attributes are configuration knobs (not structural geometry), so
they are editable after construction and a setter must reflect on the getter."""

import yamc


def _geometry():
    sphere = yamc.Sphere(radius=2.0, boundary="vacuum")
    return yamc.Geometry([yamc.Cell(region=sphere.below)])


def test_total_particles_is_a_per_run_argument():
    """total_particles is chosen per run via simulate_transport() rather than
    stored on the Model: it is neither a constructor kwarg nor a Model attribute,
    but it is a parameter of the run call."""
    import inspect
    import pytest

    # No longer accepted as a constructor kwarg.
    with pytest.raises(TypeError):
        yamc.Model(_geometry(), total_particles=1000)

    # No longer a readable/settable Model attribute.
    model = yamc.Model(_geometry())
    assert not hasattr(model, "total_particles")

    # It is a parameter of the run call instead.
    run_params = inspect.signature(model.simulate_transport).parameters
    assert "total_particles" in run_params
    assert "seed" in run_params


def test_model_run_guard_setters():
    """max_lost_particles / max_steps_per_particle are editable run guards."""
    model = yamc.Model(_geometry())
    model.max_lost_particles = 42
    model.max_steps_per_particle = 7777
    assert model.max_lost_particles == 42
    assert model.max_steps_per_particle == 7777


def test_cell_name_setter():
    """Cell.name is a relabel-able metadata field (like Cell.id)."""
    sphere = yamc.Sphere(radius=2.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, name="inner")
    assert cell.name == "inner"
    cell.name = "outer"
    assert cell.name == "outer"
    cell.name = None
    assert cell.name is None


def test_tally_name_and_id_setters():
    """Tally.name and Tally.id are relabel-able metadata fields."""
    tally = yamc.Tally(scores=["flux"], name="a", id=1)
    assert tally.name == "a"
    assert tally.id == 1
    tally.name = "b"
    tally.id = 99
    assert tally.name == "b"
    assert tally.id == 99
