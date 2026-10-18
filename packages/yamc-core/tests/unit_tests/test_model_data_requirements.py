"""``Model.required_nuclides`` / ``required_elements`` / ``has_photons``.

These three used to be computed in Python, in two different places and two
different ways: ``_export.py`` round-tripped the whole model through
``json.loads(model.to_json())`` and re-derived ``has_photons`` by sniffing for
the substring ``"Photon"`` in the serialized sources, while ``__init__.py``
walked ``geometry.cells`` and deep-cloned a ``Material`` per cell just to read
name strings. Both keyed on the ``"Csg"`` branch of the serialized geometry, so
both silently returned ``[]`` for a mesh-backed model.

They now read the model's flat material store in Rust, which both backends
share, so the mesh cases below are the regression net.
"""

import pytest
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"


def _material(name, composition):
    return yamc.Material(
        composition=composition, density=1.0, density_units="g/cc", name=name
    )


def _csg_model(**kwargs):
    mat = _material("steel", {"Fe56": 0.9, "Li6": 0.1})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=mat)
    return yamc.Model(yamc.Geometry([cell]), **kwargs)


def test_required_nuclides_are_sorted_and_deduplicated():
    model = _csg_model(source=yamc.NeutronSource())
    assert model.required_nuclides() == ["Fe56", "Li6"]


def test_a_neutron_only_model_needs_no_photon_data():
    model = _csg_model(source=yamc.NeutronSource())
    assert model.has_photons() is False
    assert model.required_elements() == []


def test_secondary_photon_production_pulls_in_element_data():
    model = _csg_model(
        source=yamc.NeutronSource(), transport_secondary_photons=True
    )
    assert model.has_photons() is True
    assert model.required_elements() == ["Fe", "Li"]


def test_a_photon_source_needs_element_data_with_no_flag_set():
    """A pure photon source needs photon data even though
    ``transport_secondary_photons`` is False. This is the case the old
    substring-sniffing Python had to special-case; ``has_photons`` covers it."""
    model = _csg_model(source=yamc.PhotonSource(energy=1e6))
    assert model.has_photons() is True
    assert model.required_elements() == ["Fe", "Li"]


def test_required_temperatures_follow_the_materials_holding_each_nuclide():
    """Each nuclide is loaded at the temperature of every material holding it,
    in ascending order, and at every temperature (``None``) when one of those
    materials has none set. This is what the browser fetch ranges each
    nuclide's data to, and what ``to_html`` checks embedded data against."""
    hot = yamc.Material(
        composition={"Li6": 0.5, "Fe56": 0.5}, density=1.0, temperature=600, name="hot"
    )
    cold = yamc.Material(composition={"Li6": 1.0}, density=1.0, temperature=294, name="cold")
    unset = yamc.Material(composition={"Fe56": 0.5, "O16": 0.5}, density=1.0, name="unset")
    spheres = [yamc.Sphere(radius=r) for r in (1.0, 2.0)] + [
        yamc.Sphere(radius=3.0, boundary="vacuum")
    ]
    cells = [
        yamc.Cell(region=spheres[0].below, material=hot),
        yamc.Cell(region=spheres[0].above & spheres[1].below, material=cold),
        yamc.Cell(region=spheres[1].above & spheres[2].below, material=unset),
    ]
    model = yamc.Model(yamc.Geometry(cells), source=yamc.NeutronSource())
    assert model.required_temperatures() == {
        "Fe56": None,
        "Li6": ["294", "600"],
        "O16": None,
    }


def test_element_symbol_strips_the_mass_number():
    mat = _material("odd", {"Am241": 0.5, "H1": 0.5})
    sphere = yamc.Sphere(radius=1.0, boundary="vacuum")
    model = yamc.Model(
        yamc.Geometry([yamc.Cell(region=sphere.below, material=mat)]),
        source=yamc.PhotonSource(energy=1e6),
    )
    assert model.required_elements() == ["Am", "H"]


# ---------------------------------------------------------------------------
# Mesh-backed models: the case the Python implementations got wrong.
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_model():
    a = _material("a", {"Fe56": 1.0})
    b = _material("b", {"Li6": 1.0})
    geom = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": a, "moderator": b})
    return yamc.Model(geom, source=yamc.NeutronSource())


def test_mesh_model_reports_its_nuclides(mesh_model):
    """Previously ``[]``: the Python keyed on the serialized ``"Csg"`` branch,
    which a mesh model does not have."""
    assert mesh_model.required_nuclides() == ["Fe56", "Li6"]


def test_mesh_model_reports_its_elements(mesh_model):
    mesh_model_with_photons = yamc.Model(
        yamc.MeshGeometry(
            TWO_REGION_ARROW,
            {
                "fuel": _material("a", {"Fe56": 1.0}),
                "moderator": _material("b", {"Li6": 1.0}),
            },
        ),
        source=yamc.NeutronSource(),
        transport_secondary_photons=True,
    )
    assert mesh_model_with_photons.required_elements() == ["Fe", "Li"]


def test_mesh_model_does_not_raise_on_data_queries(mesh_model):
    """``Model.geometry`` raises on a mesh-backed model, so anything that got
    at the materials through it was broken for mesh. These go through the
    material store instead and must simply work."""
    assert mesh_model.has_photons() is False
    assert mesh_model.required_elements() == []
    # Neither material sets a temperature, so both load every one.
    assert mesh_model.required_temperatures() == {"Fe56": None, "Li6": None}
