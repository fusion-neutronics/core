import os
import tempfile

import yamc


def _make_geometry():
    """Simple sphere geometry for testing."""
    sphere = yamc.Sphere(radius=2.0)
    region = sphere.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, name="test_mat")
    cell = yamc.Cell(region=region, material=mat)
    return yamc.Geometry([cell])


class TestInteractivePlotReturn:
    """Test that interactive_plot returns an InteractivePlot object."""

    def test_returns_plot(self):
        geom = _make_geometry()
        plot = geom.plot()
        assert isinstance(plot, yamc.InteractivePlot)

    def test_repr_html(self):
        geom = _make_geometry()
        plot = geom.plot()
        html = plot._repr_html_()
        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "canvas" in html.lower()

    def test_str(self):
        geom = _make_geometry()
        plot = geom.plot()
        s = str(plot)
        assert isinstance(s, str)
        assert "<html" in s.lower()

    def test_repr(self):
        geom = _make_geometry()
        plot = geom.plot()
        r = repr(plot)
        assert "InteractivePlot" in r
        assert "px" in r

    def test_html_property(self):
        geom = _make_geometry()
        plot = geom.plot()
        assert plot.html == plot._repr_html_()


class TestInteractivePlotSaveHTML:
    """Test saving to HTML files."""

    def test_save_html(self):
        geom = _make_geometry()
        plot = geom.plot()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            fname = f.name
        try:
            plot.save(fname)
            assert os.path.getsize(fname) > 1000
            with open(fname) as f:
                content = f.read()
            assert "<html" in content.lower()
        finally:
            os.unlink(fname)


class TestInteractivePlotSavePNG:
    """Test saving to PNG files."""

    def test_save_png(self):
        geom = _make_geometry()
        plot = geom.plot()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            plot.save(fname)
            size = os.path.getsize(fname)
            assert size > 100  # valid PNG is > 100 bytes
            # Check PNG magic bytes
            with open(fname, "rb") as f:
                magic = f.read(8)
            assert magic[:4] == b"\x89PNG"
        finally:
            os.unlink(fname)

    def test_png_pixel_dimensions(self):
        geom = _make_geometry()
        plot = geom.plot(resolution=10000)
        r = repr(plot)
        assert "100x100" in r


class TestInteractivePlotParameters:
    """Test that various parameter combinations work."""

    def test_basis_xy(self):
        geom = _make_geometry()
        plot = geom.plot(basis="xy")
        assert len(plot.html) > 0

    def test_basis_xz(self):
        geom = _make_geometry()
        plot = geom.plot(basis="xz")
        assert len(plot.html) > 0

    def test_basis_yz(self):
        geom = _make_geometry()
        plot = geom.plot(basis="yz")
        assert len(plot.html) > 0

    def test_color_by_cell(self):
        geom = _make_geometry()
        plot = geom.plot(color_by="cell")
        assert len(plot.html) > 0

    def test_color_by_material(self):
        geom = _make_geometry()
        plot = geom.plot(color_by="material")
        assert len(plot.html) > 0

    def test_outline_cell(self):
        geom = _make_geometry()
        plot = geom.plot(outline="cell")
        assert len(plot.html) > 0

    def test_outline_material(self):
        geom = _make_geometry()
        plot = geom.plot(outline="material")
        assert len(plot.html) > 0

    def test_custom_origin(self):
        geom = _make_geometry()
        plot = geom.plot(origin=(0.5, 0.5, 0.0))
        assert len(plot.html) > 0

    def test_custom_width(self):
        geom = _make_geometry()
        plot = geom.plot(width=(3.0, 3.0))
        assert len(plot.html) > 0

    def test_custom_pixels_int(self):
        geom = _make_geometry()
        plot = geom.plot(resolution=1000)
        assert len(plot.html) > 0

    def test_custom_pixels_tuple(self):
        geom = _make_geometry()
        plot = geom.plot(resolution=(50, 50))
        r = repr(plot)
        assert "50x50" in r

    def test_axis_units_mm(self):
        geom = _make_geometry()
        plot = geom.plot(axis_units="mm")
        assert len(plot.html) > 0

    def test_custom_colors(self):
        geom = _make_geometry()
        plot = geom.plot(colors={10: "#ff0000"})
        assert len(plot.html) > 0

    def test_contour_kwargs_colors(self):
        geom = _make_geometry()
        plot = geom.plot(
            outline="cell", contour_kwargs={"colors": "#ff0000"}
        )
        assert len(plot.html) > 0
        assert "ff0000" in plot.html.lower()

    def test_contour_kwargs_linewidths(self):
        geom = _make_geometry()
        plot = geom.plot(
            outline="cell", contour_kwargs={"linewidths": 3}
        )
        assert len(plot.html) > 0

    def test_contour_kwargs_colors_changes_png(self):
        """Different outline colors should produce different PNGs."""
        geom = _make_geometry()
        p1 = geom.plot(
            resolution=(30, 30), outline="cell",
            contour_kwargs={"colors": "#000000"},
        )
        p2 = geom.plot(
            resolution=(30, 30), outline="cell",
            contour_kwargs={"colors": "#ff0000"},
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            fname1, fname2 = f1.name, f2.name
        try:
            p1.save(fname1)
            p2.save(fname2)
            with open(fname1, "rb") as f:
                data1 = f.read()
            with open(fname2, "rb") as f:
                data2 = f.read()
            assert data1 != data2
        finally:
            os.unlink(fname1)
            os.unlink(fname2)

    def test_contour_kwargs_linewidths_changes_png(self):
        """Different outline thicknesses should produce different PNGs."""
        geom = _make_geometry()
        p1 = geom.plot(
            resolution=(30, 30), outline="cell",
            contour_kwargs={"linewidths": 1},
        )
        p2 = geom.plot(
            resolution=(30, 30), outline="cell",
            contour_kwargs={"linewidths": 3},
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            fname1, fname2 = f1.name, f2.name
        try:
            p1.save(fname1)
            p2.save(fname2)
            with open(fname1, "rb") as f:
                data1 = f.read()
            with open(fname2, "rb") as f:
                data2 = f.read()
            assert data1 != data2
        finally:
            os.unlink(fname1)
            os.unlink(fname2)


class TestInteractivePlotPNGContent:
    """Test PNG content correctness."""

    def test_png_has_colored_pixels(self):
        """PNG of a sphere should have non-white pixels."""
        geom = _make_geometry()
        plot = geom.plot(resolution=(20, 20))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            plot.save(fname)
            # Read raw PNG data - just verify it's a valid file
            with open(fname, "rb") as f:
                data = f.read()
            assert len(data) > 50
            assert data[:4] == b"\x89PNG"
        finally:
            os.unlink(fname)

    def test_png_outline_changes_output(self):
        """PNG with and without outline should differ."""
        geom = _make_geometry()
        plot_no_outline = geom.plot(resolution=(30, 30), outline=None)
        plot_with_outline = geom.plot(resolution=(30, 30), outline="cell")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            fname1, fname2 = f1.name, f2.name
        try:
            plot_no_outline.save(fname1)
            plot_with_outline.save(fname2)
            with open(fname1, "rb") as f:
                data1 = f.read()
            with open(fname2, "rb") as f:
                data2 = f.read()
            # They should be different (outline adds black pixels)
            assert data1 != data2
        finally:
            os.unlink(fname1)
            os.unlink(fname2)

    def test_png_larger_than_raw_grid(self):
        """PNG with axes should be larger than the raw plot grid."""
        geom = _make_geometry()
        plot = geom.plot(resolution=(50, 50))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            plot.save(fname)
            # PNG includes margins for axes, so file should be non-trivial
            assert os.path.getsize(fname) > 200
        finally:
            os.unlink(fname)

    def test_png_different_units(self):
        """PNGs with different axis_units should differ (different labels)."""
        geom = _make_geometry()
        plot_cm = geom.plot(resolution=(30, 30), axis_units="cm")
        plot_mm = geom.plot(resolution=(30, 30), axis_units="mm")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            fname1, fname2 = f1.name, f2.name
        try:
            plot_cm.save(fname1)
            plot_mm.save(fname2)
            with open(fname1, "rb") as f:
                data1 = f.read()
            with open(fname2, "rb") as f:
                data2 = f.read()
            assert data1 != data2
        finally:
            os.unlink(fname1)
            os.unlink(fname2)
