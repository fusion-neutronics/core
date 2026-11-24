"""A particle with no surface ahead of it is reported as lost.

A void sphere with a transmission (not vacuum) surface, wrapped in an
unbounded void cell that nothing closes: every particle that crosses the
sphere is in a cell with no boundary ahead. That is how a missing vacuum
boundary shows up, so each history must land in ``model.lost_particles``
naming the outer cell, and the default ``max_lost_particles`` must abort.
The model is all void, so no nuclear data is needed.
"""

import pytest
import yamc


def _model(boundary, outer_cell, **model_kwargs):
    sphere = yamc.Sphere(radius=5.0, boundary=boundary, surface_id=1)
    cells = [yamc.Cell(name="inside", region=sphere.below, id=1)]
    if outer_cell:
        cells.append(yamc.Cell(name="outside", region=sphere.above, id=2))
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(0, 0, 0)
    )
    return yamc.Model(yamc.Geometry(cells), source=source, **model_kwargs)


def test_no_boundary_ahead_is_lost_in_the_outer_cell():
    model = _model("transmission", outer_cell=True, max_lost_particles=1000)
    model.simulate_transport(total_particles=20, seed=1)
    lost = model.lost_particles
    assert len(lost) == 20
    for lp in lost:
        assert lp.last_cell_id == 2
        assert lp.last_cell_name == "outside"
        assert lp.surface_id == 1
        r = sum(x * x for x in lp.position) ** 0.5
        assert r == pytest.approx(5.0, abs=1e-6)


def test_no_boundary_ahead_trips_the_default_cap():
    model = _model("transmission", outer_cell=True)
    with pytest.raises(BaseException, match="Maximum lost particles exceeded"):
        model.simulate_transport(total_particles=20, seed=1, threads=1)


def test_vacuum_boundary_loses_nothing():
    model = _model("vacuum", outer_cell=False)
    model.simulate_transport(total_particles=20, seed=1)
    assert len(model.lost_particles) == 0
