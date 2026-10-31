"""Tests for the interactive tally viewer (tally.plot)."""

import os
import tempfile

import yamc


def _make_mesh_tally():
    """Create a mesh tally with flux score (data will be zeros)."""
    mesh = yamc.RegularRectangularMesh([0, 0, 0], [10, 10, 10], [5, 5, 5])
    tally = yamc.Tally(scores=["flux"], mesh=mesh)
    return tally


def _make_geometry():
    """Simple sphere geometry for outline testing."""
    sphere = yamc.Sphere(radius=5.0)
    region = sphere.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, name="test_mat")
    cell = yamc.Cell(region=region, material=mat)
    return yamc.Geometry([cell])


class TestInteractiveTallyPlotReturn:
    """Test that interactive_plot returns an InteractiveTallyPlot object."""

    def test_returns_plot(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert isinstance(plot, yamc.InteractiveTallyPlot)

    def test_repr_html(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        html = plot._repr_html_()
        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "canvas" in html.lower()

    def test_str(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert "<html" in str(plot).lower()

    def test_repr(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        r = repr(plot)
        assert "InteractiveTallyPlot" in r
        assert "bytes" in r

    def test_html_property(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert plot.html == plot._repr_html_()


class TestInteractiveTallyPlotSave:
    """Test saving to HTML."""

    def test_save_html(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            fname = f.name
        try:
            plot.save(fname)
            assert os.path.getsize(fname) > 1000
            with open(fname) as f:
                content = f.read()
            assert "<html" in content.lower()
            assert "SLICES" in content
        finally:
            os.unlink(fname)


class TestInteractiveTallyPlotBases:
    """Test all three slice bases."""

    def test_basis_xy(self):
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy")
        assert "xy" in plot.html.lower()

    def test_basis_xz(self):
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xz")
        assert len(plot.html) > 0

    def test_basis_yz(self):
        tally = _make_mesh_tally()
        plot = tally.plot(basis="yz")
        assert len(plot.html) > 0

    def test_all_basis_buttons_enabled(self):
        """All 3 basis buttons should be present without disabled attribute."""
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy")
        html = plot.html
        # All buttons present
        assert 'data-val="xy"' in html
        assert 'data-val="xz"' in html
        assert 'data-val="yz"' in html
        # No disabled attribute on any basis button
        assert "disabled" not in html.split("basis-btns")[1].split("</div>")[0]

    def test_native_basis_embedded(self):
        """NATIVE_BASIS JS constant matches the requested basis."""
        for basis in ("xy", "xz", "yz"):
            tally = _make_mesh_tally()
            plot = tally.plot(basis=basis)
            assert f'NATIVE_BASIS = "{basis}"' in plot.html


class TestInteractiveTallyPlotSlices:
    """Test various slice specifications."""

    def test_single_slice_default(self):
        """Default: single center slice."""
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy")
        # Should have exactly one xy slice embedded
        assert '"xy:' in plot.html

    def test_specific_slices(self):
        """Specific bin indices."""
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy", slices=[0, 2, 4])
        assert '"xy:0"' in plot.html
        assert '"xy:2"' in plot.html
        assert '"xy:4"' in plot.html

    def test_all_slices(self):
        """All slices."""
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy", slices="all")
        # 5 bins along z for xy basis
        for i in range(5):
            assert f'"xy:{i}"' in plot.html

    def test_slice_coord(self):
        """Explicit slice coordinate."""
        tally = _make_mesh_tally()
        plot = tally.plot(basis="xy", slice_coord=2.0)
        # bin 1 (coords: 0, 2, 4, 6, 8 → bin 0 covers 0-2, bin 1 covers 2-4)
        assert '"xy:1"' in plot.html


class TestInteractiveTallyPlotWithGeometry:
    """Test with geometry overlay."""

    def test_with_csg_geometry(self):
        tally = _make_mesh_tally()
        geom = _make_geometry()
        plot = tally.plot(geometry=geom, outline="material")
        assert "GEOMETRY_JSON" in plot.html
        assert '"csg"' in plot.html

    def test_without_geometry(self):
        tally = _make_mesh_tally()
        plot = tally.plot(outline=None)
        assert "GEOMETRY_JSON = null" in plot.html


class TestInteractiveTallyPlotParameters:
    """Test various parameter combinations."""

    def test_log_scale(self):
        tally = _make_mesh_tally()
        plot = tally.plot(log_scale=True)
        assert "logScale: true" in plot.html

    def test_linear_scale(self):
        tally = _make_mesh_tally()
        plot = tally.plot(log_scale=False)
        assert "logScale: false" in plot.html

    def test_colorscale(self):
        tally = _make_mesh_tally()
        plot = tally.plot(colorscale="Hot")
        assert '"Hot"' in plot.html

    def test_axis_units(self):
        tally = _make_mesh_tally()
        plot = tally.plot(axis_units="mm")
        assert '"mm"' in plot.html

    def test_contour_kwargs(self):
        tally = _make_mesh_tally()
        geom = _make_geometry()
        plot = tally.plot(
            geometry=geom,
            contour_kwargs={"colors": "#ff0000", "linewidths": 3},
        )
        assert "ff0000" in plot.html.lower()

    def test_custom_title(self):
        tally = _make_mesh_tally()
        plot = tally.plot(title="My Custom Title")
        assert "My Custom Title" in plot.html

    def test_custom_colorbar_title(self):
        tally = _make_mesh_tally()
        plot = tally.plot(colorbar_title="Flux [n/cm²/s]")
        assert "Flux [n/cm" in plot.html

    def test_outline_none(self):
        tally = _make_mesh_tally()
        plot = tally.plot(outline=None)
        assert len(plot.html) > 0

    def test_outline_cell(self):
        tally = _make_mesh_tally()
        geom = _make_geometry()
        plot = tally.plot(geometry=geom, outline="cell")
        assert len(plot.html) > 0

    def test_value_mean(self):
        tally = _make_mesh_tally()
        plot = tally.plot(value="mean")
        assert len(plot.html) > 0

    def test_value_std_dev(self):
        tally = _make_mesh_tally()
        plot = tally.plot(value="standard_deviation")
        assert len(plot.html) > 0

    def test_value_rel_error(self):
        tally = _make_mesh_tally()
        plot = tally.plot(value="relative_error")
        assert len(plot.html) > 0

    def test_scaling_factor(self):
        tally = _make_mesh_tally()
        plot = tally.plot(scaling_factor=1000.0)
        assert "INITIAL_SCALING_FACTOR = 1000" in plot.html

    def test_font_size(self):
        tally = _make_mesh_tally()
        plot = tally.plot(font_size=16)
        assert "INITIAL_FONT_SIZE = 16" in plot.html

    def test_outline_pixels_in_html(self):
        tally = _make_mesh_tally()
        plot = tally.plot(resolution=100000)
        assert "OUTLINE_PIXELS = 100000" in plot.html
        assert 'id="outline-pixels"' in plot.html

    def test_colorbar_label_editable(self):
        tally = _make_mesh_tally()
        plot = tally.plot(colorbar_title="Custom Label")
        assert 'id="colorbar-label-input"' in plot.html
        assert "Custom Label" in plot.html


class TestInteractiveTallyPlotMeshData:
    """Test that mesh metadata is correctly embedded."""

    def test_mesh_bounds(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert "MESH_LL = [0,0,0]" in plot.html
        assert "MESH_UR = [10,10,10]" in plot.html

    def test_mesh_dimensions(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert "MESH_DIM = [5,5,5]" in plot.html

    def test_mesh_width(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        assert "MESH_WIDTH = [2,2,2]" in plot.html

    def test_slice_dims(self):
        tally = _make_mesh_tally()
        plot = tally.plot()
        # xy: [nx, ny] = [5, 5]
        assert '"xy":[5,5]' in plot.html


class TestInteractiveTallyPlotEdgeCases:
    """Test edge cases."""

    def test_single_bin_mesh(self):
        """1x1x1 mesh should work."""
        mesh = yamc.RegularRectangularMesh([0, 0, 0], [1, 1, 1], [1, 1, 1])
        tally = yamc.Tally(scores=["flux"], mesh=mesh)
        plot = tally.plot()
        assert len(plot.html) > 0

    def test_asymmetric_mesh(self):
        """Non-cubic mesh should work."""
        mesh = yamc.RegularRectangularMesh([0, 0, 0], [10, 20, 30], [2, 4, 6])
        tally = yamc.Tally(scores=["flux"], mesh=mesh)
        plot = tally.plot(basis="xy")
        assert "MESH_DIM = [2,4,6]" in plot.html
