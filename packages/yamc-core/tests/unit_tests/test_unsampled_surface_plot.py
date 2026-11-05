"""Plots of a geometry with a surface kind the browser view cannot evaluate.

The interactive HTML re-samples CSG geometry in JavaScript, whose
``surfaceEval`` knows planes, spheres, cylinders and z-tori only. For any other
kind (a flux surface here) the HTML must carry the raster yamc rendered, which
the viewer then keeps instead of re-sampling.
"""

import base64
import re

import numpy as np
import pytest
import yamc

PRESAMPLED = re.compile(r'const PRESAMPLED_B64 = "([^"]*)"')


def _flux_geometry():
    # A circular torus, R0 = 600 cm and a = 150 cm, as a stellarator shape.
    shape = yamc.sources.StellaratorShape(
        num_field_periods=1,
        r_modes=[(0, 0, 0), (1, 1, 0)],
        r_lmn=[600.0, 150.0],
        z_modes=[(1, -1, 0)],
        z_lmn=[150.0],
    )
    plasma = yamc.FluxSurface(shape)
    world = yamc.Sphere(radius=1000.0, boundary="vacuum")
    return yamc.Geometry(
        [
            yamc.Cell(region=plasma.below, id=1, name="plasma"),
            yamc.Cell(region=plasma.above & world.below, id=2, name="wall"),
        ]
    )


@pytest.mark.parametrize("via", ["model", "geometry"])
def test_flux_surface_plot_embeds_the_rendered_image(via):
    geometry = _flux_geometry()
    kwargs = dict(
        origin=(600.0, 0.0, 0.0), width=(400.0, 400.0), resolution=(40, 40), basis="xz"
    )
    if via == "model":
        source = yamc.NeutronSource(
            position=[600.0, 0.0, 0.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
        )
        plot = yamc.Model(geometry=geometry, source=source, verbose=[]).plot(**kwargs)
    else:
        plot = geometry.plot(**kwargs)

    match = PRESAMPLED.search(plot.html)
    assert match, (
        "the browser cannot evaluate a flux surface, so the HTML must carry the "
        "image yamc rendered"
    )
    data = np.frombuffer(base64.b64decode(match.group(1)), dtype="<i4")
    assert set(data[0::2].tolist()) == {1, 2}, "plasma and wall both in the image"
