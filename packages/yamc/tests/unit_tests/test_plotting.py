import pytest

import yamc


def _make_cell_and_geometry():
    sphere = yamc.Sphere(radius=1.0)
    region = sphere.below
    cell = yamc.Cell(region=region, name="cell1")
    geometry = yamc.Geometry([cell])
    return cell, region, geometry


def test_geometry_plot_returns_interactive():
    _, _, geometry = _make_cell_and_geometry()
    plot = geometry.plot(basis="xy", color_by="cell", outline="cell")
    assert isinstance(plot, yamc.InteractivePlot)
    html = plot._repr_html_()
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_cell_plot_returns_interactive():
    cell, _, _ = _make_cell_and_geometry()
    plot = cell.plot(basis="xy", color_by="cell", outline="cell")
    assert isinstance(plot, yamc.InteractivePlot)
    html = plot._repr_html_()
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_region_plot_returns_interactive():
    _, region, _ = _make_cell_and_geometry()
    plot = region.plot(basis="xy", outline="cell")
    assert isinstance(plot, yamc.InteractivePlot)
    html = plot._repr_html_()
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_region_plot_refuses_an_unknown_axis_unit():
    """An unknown unit used to plot cm values labelled with it; now it is
    refused, naming the units plots know."""
    _, region, _ = _make_cell_and_geometry()
    with pytest.raises(ValueError, match="'mm', 'cm', 'm' or 'km'"):
        region.plot(axis_units="inch")
