"""Surface-current tallies: Tally(surfaces=..., scores=["current-net", ...])."""

import math
import re

import pytest
import yamc


def _void_spheres():
    inner = yamc.Sphere(radius=5.0)
    outer = yamc.Sphere(radius=10.0, boundary="vacuum")
    cells = [
        yamc.Cell(region=inner.below),
        yamc.Cell(region=inner.above & outer.below),
    ]
    return inner, outer, yamc.Geometry(cells)


def _run(geometry, tallies, position=(0.0, 0.0, 0.0), particles=5000):
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.NeutronSource(position=position, energy=14.1e6),
    )
    return model.simulate_transport(total_particles=particles, seed=1)


def test_leakage_from_a_void_sphere_is_one_per_source_particle():
    inner, outer, geometry = _void_spheres()
    tally = yamc.Tally(scores=["current-net"], surfaces=[inner, outer])
    result = _run(geometry, [tally])[tally]
    assert result.dim_labels == ["score", "surface"]
    assert result.mean == pytest.approx([1.0, 1.0], abs=1e-12)
    assert tally.estimator == "surface"
    assert tally.surfaces == [inner.surface_id, outer.surface_id]


def test_above_below_and_net_currents_through_a_plane():
    plane = yamc.Plane(axis="x", offset=2.0)
    world = yamc.Sphere(radius=10.0, boundary="vacuum")
    geometry = yamc.Geometry([
        yamc.Cell(region=world.below & plane.below),
        yamc.Cell(region=world.below & plane.above),
    ])
    tally = yamc.Tally(scores=["current-net", "current-above", "current-below"], surfaces=plane)
    # From x = 5 (above the plane) every crossing is towards .below.
    net, above, below = _run(geometry, [tally], position=(5.0, 0.0, 0.0))[tally].mean
    assert above == 0.0
    assert below == pytest.approx((1 - 3 / math.sqrt(105)) / 2, abs=0.03)
    assert net == pytest.approx(above - below, abs=1e-12)
    assert tally.scores == ["current-net", "current-above", "current-below"]


def test_cosine_bins_split_crossings_by_angle_to_the_normal():
    inner, _, geometry = _void_spheres()
    # From the centre every crossing is along the outward normal.
    tally = yamc.Tally(scores=["current-net"], surfaces=inner, cosine_bins=[-1.0, 0.0, 0.5, 1.0])
    result = _run(geometry, [tally])[tally]
    assert result.mean == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    # A detector viewing within 17.28 degrees of the normal, off-centre source.
    window = [math.cos(math.radians(17.28)), 1.0]
    partial = yamc.Tally(scores=["current-net"], surfaces=inner, cosine_bins=window)
    off_centre = _run(geometry, [partial], position=(3.0, 0.0, 0.0))[partial]
    assert 0.0 < off_centre.mean[0] < 1.0
    assert partial.cosine_bins == window


def test_energy_bins_bin_the_crossing_energy():
    inner, _, geometry = _void_spheres()
    tally = yamc.Tally(scores=["current-net"], surfaces=inner, energy_bins=[1e6, 1e7, 2e7])
    result = _run(geometry, [tally])[tally]
    assert result.mean == pytest.approx([0.0, 1.0], abs=1e-12)


def test_units_are_particles_per_source_particle():
    inner, _, _ = _void_spheres()
    tally = yamc.Tally(scores=["current-net"], surfaces=inner)
    assert tally.units == ["particles / source-particle"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"scores": ["current-net"]}, "pass surfaces="),
        ({"scores": ["flux"], "surfaces": "S"}, "'current-net', 'current-above' and"),
        ({"scores": ["current-net"], "surfaces": "S", "estimator": "collision"}, "takes no estimator"),
        ({"scores": ["current-net"], "cosine_bins": [0.0, 1.0]}, "pass surfaces="),
        ({"scores": ["current-net"], "surfaces": "S", "cosine_bins": [0.5, 1.5]}, "within [-1, 1]"),
        ({"scores": ["current-net"], "surfaces": "S", "cells": "C"}, "cannot also take cells="),
        ({"scores": ["current-net"], "surfaces": "S", "nuclides": ["Fe56"]}, "cannot apply to the 'current-net' score"),
    ],
)
def test_combinations_a_surface_tally_cannot_take_are_refused(kwargs, message):
    sphere = yamc.Sphere(radius=1.0)
    cell = yamc.Cell(region=sphere.below)
    yamc.Geometry([cell])
    kwargs = {k: {"S": sphere, "C": cell}.get(v, v) if isinstance(v, str) else v for k, v in kwargs.items()}
    with pytest.raises(ValueError, match=re.escape(message)):
        yamc.Tally(**kwargs)


def test_delta_tracking_is_refused():
    inner, _, geometry = _void_spheres()
    tally = yamc.Tally(scores=["current-net"], surfaces=inner)
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=yamc.NeutronSource(position=(0.0, 0.0, 0.0), energy=14.1e6),
        tracking_mode="woodcock",
    )
    with pytest.raises(ValueError, match="does not support surface tallies"):
        model.simulate_transport(total_particles=100)


def test_a_surface_not_in_the_geometry_is_refused():
    _, _, geometry = _void_spheres()
    stray = yamc.Sphere(radius=3.0)
    tally = yamc.Tally(scores=["current-net"], surfaces=stray)
    with pytest.raises(ValueError, match=f"surface id {stray.surface_id}"):
        _run(geometry, [tally], particles=100)
