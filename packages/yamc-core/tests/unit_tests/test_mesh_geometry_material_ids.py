"""Material ids in yamc.MeshGeometry.

Every Material has an id from construction, so a MeshGeometry neither assigns
nor changes any: it keeps the ids the materials were built with, and refuses
two different materials that share one. Absolute automatic ids depend on what
ran earlier in the process, so these tests compare ids rather than assert them.

Lives in ``unit_tests/`` (not ``cad/``) so it runs in CI, which installs no
cadquery.
"""

import pytest
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"


def _dummy_material(name, **kwargs):
    """Minimal Material (no nuclear data) used purely to test ID handling."""
    return yamc.Material(
        composition={"H": 1.0}, density=1.0, density_units="g/cc", name=name, **kwargs,
    )


def test_materials_keep_their_ids():
    """The materials have distinct ids before the mesh geometry is built, and
    building it does not change them."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    ids = (fuel.id, moderator.id)
    assert fuel.id != moderator.id

    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert (fuel.id, moderator.id) == ids


def test_explicit_id_is_preserved():
    """An explicit id survives construction, and a material built after it
    does not get it."""
    probe = _dummy_material("probe")
    fuel = _dummy_material("fuel", id=probe.id + 1)
    moderator = _dummy_material("moderator")

    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert fuel.id == probe.id + 1
    assert moderator.id not in (probe.id, fuel.id)


def test_duplicate_ids_rejected():
    """Two materials with the same explicit ID should raise."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    fuel.id = 5
    moderator.id = 5

    with pytest.raises(ValueError, match="Duplicate Material id"):
        yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})


def test_explicit_id_repeating_an_automatic_one_is_rejected():
    """An explicit id set later can repeat an automatic one; the mesh
    geometry refuses the two different materials that share it."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator", id=fuel.id)

    with pytest.raises(ValueError, match=f"Duplicate Material id {fuel.id}"):
        yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})


def test_bounding_box_for_material_works_after_construction():
    """Looking up a Material by reference finds its volumes by its id."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    bbox = mesh.bounding_box_for_material(fuel)
    assert bbox is not None
    # Two-region mesh: fuel occupies the left half (x in [0, 0.5]).
    assert bbox.lower_left[0] >= -1e-9
    assert bbox.upper_right[0] <= 0.5 + 1e-9


def test_calculate_volume_uses_analytic_values():
    """MeshGeometry.calculate_volume should return the exact analytic volumes
    (already known via the divergence theorem when the mesh was loaded), with
    std_dev == 0 -- no stochastic sampling is performed."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    # samples/bounding_box/seed are accepted for signature compatibility but ignored
    volumes = mesh.calculate_volume(samples=12345, seed=99)

    assert isinstance(volumes, dict)
    assert len(volumes) == mesh.num_volumes

    expected_measures = mesh.volume_measures
    for vol_id, result in volumes.items():
        assert result.volume == expected_measures[vol_id]
        assert result.std_dev == 0.0
        assert result.num_hits == 0


def test_calculate_volume_signature_matches_csg():
    """Calling calculate_volume with no args (the typical CSG idiom) works."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    mesh = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    volumes = mesh.calculate_volume()
    assert len(volumes) == mesh.num_volumes
    assert all(r.std_dev == 0.0 for r in volumes.values())


def test_implicit_complement_material_keeps_its_id():
    """The optional implicit-complement material follows the same rule as the
    named-dict materials."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    air = _dummy_material("air")
    air_id = air.id

    yamc.MeshGeometry(
        TWO_REGION_ARROW,
        {"fuel": fuel, "moderator": moderator},
        implicit_complement_material=air,
    )

    assert air.id == air_id
    assert air.id not in (fuel.id, moderator.id)


def test_same_material_under_multiple_keys_is_one_material():
    """If the user maps two names to the same ``Material`` object, it is one
    material with one id, not a duplicate."""
    shared = _dummy_material("shared")
    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": shared, "moderator": shared})
