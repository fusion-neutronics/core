"""A surface shared between cells keeps its explicit id through save and load.

JSON writes a shared surface once per cell that uses it. Loading used to read
the copies back as separate surfaces, so the one ``surface_id`` they carry
was refused as a duplicate.
"""

import pytest

import yamc


def _model():
    inner = yamc.Sphere(radius=5.0, surface_id=1, name="inner")
    outer = yamc.Sphere(radius=10.0, surface_id=2, boundary="vacuum")
    cells = [
        yamc.Cell(region=inner.below, name="core"),
        yamc.Cell(region=inner.above & outer.below, name="shell"),
    ]
    return yamc.Model(
        geometry=yamc.Geometry(cells),
        source=yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0])),
    )


def test_a_shared_surface_with_an_id_loads_back(tmp_path):
    model = _model()
    assert yamc.Model.load(model.to_json()).to_json() == model.to_json()

    path = tmp_path / "model.json"
    model.save(str(path))
    assert yamc.Model.load(str(path)).to_json() == model.to_json()


def test_one_id_on_two_different_surfaces_is_still_refused():
    cells = [
        yamc.Cell(region=yamc.Sphere(radius=5.0, surface_id=7).below),
        yamc.Cell(region=yamc.Sphere(x0=20.0, radius=5.0, surface_id=7).below),
    ]
    with pytest.raises(ValueError, match="Duplicate surface_id 7"):
        yamc.Geometry(cells)
