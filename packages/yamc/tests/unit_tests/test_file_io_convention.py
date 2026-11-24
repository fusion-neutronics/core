"""Every save and load follows one convention (issue #655).

``obj.to_<format>(path=None)`` writes the file at ``path`` and returns None,
or returns the file's bytes when there is no path, so a browser, wasm or
Pyodide session needs no file system; ``Type.from_<format>(source)`` reads a
path or those bytes.
"""

import json

import pytest
import yamc


def _model():
    sphere = yamc.Sphere(radius=2.0, boundary="vacuum")
    return yamc.Model(
        yamc.Geometry([yamc.Cell(region=sphere.below)]),
        source=yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0])),
    )


def test_model_json_round_trips_by_path_and_by_bytes(tmp_path):
    model = _model()
    data = model.to_json()
    assert isinstance(data, bytes)
    json.loads(data)
    path = tmp_path / "model.json"
    assert model.to_json(path) is None
    assert path.read_bytes() == data
    for source in (path, str(path), data):
        assert yamc.Model.from_json(source).to_json() == data
    # Compact, for embedding; the same model.
    assert yamc.Model.from_json(model.to_json(pretty=False)).to_json() == data


def test_a_str_is_a_path_not_json():
    """No guessing: text that is not a file is refused, not parsed."""
    with pytest.raises(OSError, match="failed to read"):
        yamc.Model.from_json(_model().to_json().decode())


def test_plots_save_as_html_and_png(tmp_path):
    plot = _model().geometry.plot(basis="xy", width=(4.0, 4.0))
    html = plot.to_html()
    assert html == plot.html.encode()
    assert plot.to_html(tmp_path / "plot.html") is None
    assert (tmp_path / "plot.html").read_bytes() == html
    png = plot.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert plot.to_png(tmp_path / "plot.png") is None
    assert (tmp_path / "plot.png").read_bytes() == png


@pytest.mark.parametrize("old", ["save", "load"])
def test_the_old_names_are_gone(old):
    assert not hasattr(yamc.Model, old)
