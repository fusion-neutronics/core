"""Plots of a mesh-filled CSG cell must show the mesh body (issue #291).

``Model.plot`` / ``Geometry.plot`` return an interactive plot whose HTML is
normally re-sampled in the browser from the geometry JSON. A fill serializes as
an identity fingerprint rather than triangles, so that sampler cannot see a fill
body: the viewer drew the bare CSG frame while the legend listed embedded mesh
cells that never appeared, and the PNG (rendered server-side) disagreed with the
HTML for the same model.

The fix ships the server-rendered, fill-resolving raster for filled models and
has the viewer refuse to re-sample. These tests pin both halves.
"""

import base64
import re

import numpy as np
import pytest
import yamc

DATA = "crates/yamc/tests"
TWO_REGION = "crates/yamt/tests/data/two_region.arrow"
PRESAMPLED = re.compile(r'const PRESAMPLED_B64 = "([^"]*)"')


def _material(name, nuclide, density, mat_id):
    m = yamc.Material(
        composition={nuclide: 1.0},
        density=density,
        temperature=294,
        name=name,
        id=mat_id,
    )
    m.read_nuclear_data({nuclide: f"{DATA}/{nuclide}.arrow"})
    return m


def _filled_geometry():
    complement = _material("complement", "Fe56", 7.874, 101)
    fuel = _material("fuel", "Li6", 0.534, 102)
    moderator = _material("moderator", "Be9", 1.85, 103)
    mesh = yamc.MeshGeometry(TWO_REGION, {"fuel": fuel, "moderator": moderator})
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    chamber = yamc.Cell(
        region=sphere.below, material=complement, name="chamber", fill=mesh
    )
    return yamc.Geometry([chamber])


def _plain_geometry():
    complement = _material("complement", "Fe56", 7.874, 101)
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    return yamc.Geometry([yamc.Cell(region=sphere.below, material=complement)])


def _plot_kwargs():
    return dict(
        origin=(0.5, 0.5, 0.5), width=(2.0, 2.0), resolution=(80, 80), basis="xy"
    )


def _presampled_cell_ids(html):
    """Cell ids in the raster embedded in `html` (interleaved cell, material)."""
    match = PRESAMPLED.search(html)
    assert match, "no pre-sampled raster embedded in the HTML"
    data = np.frombuffer(base64.b64decode(match.group(1)), dtype="<i4")
    return set(data[0::2].tolist())


@pytest.mark.parametrize("via", ["model", "geometry"])
def test_filled_cell_plot_shows_the_mesh_body(via):
    geometry = _filled_geometry()
    # Embedded mesh volumes are ordinary cells appended after the user's cells.
    cell_ids = [c.id for c in geometry.cells]
    assert len(cell_ids) == 3, f"expected chamber + 2 mesh volumes, got {cell_ids}"

    if via == "model":
        source = yamc.NeutronSource(
            position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
        )
        plot = yamc.Model(geometry=geometry, source=source, verbose=[]).plot(
            **_plot_kwargs()
        )
    else:
        plot = geometry.plot(**_plot_kwargs())

    html = plot.html
    assert "const HAS_MESH_FILLS = true" in html, (
        "the viewer must know the model has fills, else it re-samples the bare "
        "CSG frame in the browser"
    )
    rendered = _presampled_cell_ids(html)
    for cid in cell_ids:
        assert cid in rendered, (
            f"cell {cid} is in the geometry (and in the plot legend) but not in "
            f"the rendered raster {sorted(rendered)}"
        )


def test_plain_csg_plot_is_unchanged():
    """No fills, no embedded raster: the browser sampler is correct and fast."""
    html = _plain_geometry().plot(**_plot_kwargs()).html
    assert "const HAS_MESH_FILLS = false" in html
    assert "const PRESAMPLED_B64 = null" in html, (
        "embedding the raster for plain CSG would re-add megabytes of base64 for "
        "no benefit"
    )


def test_filled_cell_sample_slice_resolves_fills():
    """`sample_slice` is the data behind the raster: it must resolve fills."""
    geometry = _filled_geometry()
    source = yamc.NeutronSource(
        position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(geometry=geometry, source=source, verbose=[])
    data = model.sample_slice(**_plot_kwargs())
    material_ids = set(np.asarray(data.material_ids).ravel().tolist())
    assert {101, 102, 103} <= material_ids, (
        f"complement + both mesh materials expected, got {sorted(material_ids)}"
    )
