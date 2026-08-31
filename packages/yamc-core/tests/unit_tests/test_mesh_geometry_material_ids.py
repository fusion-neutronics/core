"""Verify that yamc.MeshGeometry auto-assigns material IDs.

This is the regression net for the bug fixed in PR #42: previously the
constructor took ``HashMap<String, PyMaterial>`` (cloning), so any
material ID it assigned lived only on the clone -- the user's Python
``Material`` object kept a ``None`` ID, and a later call like
``mesh_geom.bounding_box_for_material(li_mat)`` raised
``ValueError: Material has no id``. The fix uses ``&Bound<PyDict>``
to mutate the user's Python objects in place.

Lives in ``unit_tests/`` (not ``cad/``) so it runs in CI, which installs no
cadquery.
"""

import pytest
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"


def _dummy_material(name):
    """Minimal Material (no nuclear data) used purely to test ID handling."""
    return yamc.Material(
        composition={"H": 1.0}, density=1.0, units="g/cc", name=name,
    )


def test_materials_get_ids_assigned_in_user_objects():
    """Both materials in the user's dict should have their IDs set after
    ``MeshGeometry`` construction. The user keeps their original object;
    the assignment must propagate to it."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    assert fuel.id is None
    assert moderator.id is None

    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert fuel.id is not None, "fuel.id should be auto-assigned"
    assert moderator.id is not None, "moderator.id should be auto-assigned"
    assert fuel.id != moderator.id


def test_explicit_id_is_preserved():
    """If the user pre-assigns an ID, it must survive construction."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    fuel.id = 42

    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})

    assert fuel.id == 42
    assert moderator.id is not None and moderator.id != 42


def test_duplicate_ids_rejected():
    """Two materials with the same explicit ID should raise."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    fuel.id = 5
    moderator.id = 5

    with pytest.raises(ValueError, match="Duplicate Material id"):
        yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})


def test_bounding_box_for_material_works_after_construction():
    """The original failure mode: looking up a Material by reference must
    succeed because the Material now has its auto-assigned ID."""
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


def test_implicit_complement_material_gets_id():
    """The optional implicit-complement material follows the same auto-assign
    rule as the named-dict materials."""
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    air = _dummy_material("air")
    assert air.id is None

    yamc.MeshGeometry(
        TWO_REGION_ARROW,
        {"fuel": fuel, "moderator": moderator},
        implicit_complement_material=air,
    )

    assert air.id is not None
    assert air.id not in (fuel.id, moderator.id)


def test_same_material_under_multiple_keys_gets_one_id():
    """If the user maps two names to the same ``Material`` object, they
    share a single ID -- auto-assign must dedup by Python object identity."""
    shared = _dummy_material("shared")
    yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": shared, "moderator": shared})
    # Just one ID was assigned; no duplicate-id error fired.
    assert shared.id is not None
