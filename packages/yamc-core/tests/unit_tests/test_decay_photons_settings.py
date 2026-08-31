import pytest
import yamc as mmc


def _minimal_geometry():
    """Create a minimal geometry for Model construction."""
    sphere = mmc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, boundary='vacuum')
    cell = mmc.Cell(region=sphere.below)
    return mmc.Geometry([cell])


def test_decay_photons_settings_defaults():
    """Test that D1S settings have correct defaults."""
    source = mmc.NeutronSource()
    geometry = _minimal_geometry()
    model = mmc.Model(geometry, source=source)
    assert not model.use_decay_photons


def test_decay_photons_settings_enabled():
    """Test that D1S settings can be enabled properly."""
    source = mmc.NeutronSource()
    geometry = _minimal_geometry()
    model = mmc.Model(
        geometry, source=source,
        transport_secondary_photons=True,
        use_decay_photons=True)
    assert model.use_decay_photons


def test_decay_photons_requires_transport_secondary_photons():
    """Test that use_decay_photons=True requires transport_secondary_photons=True."""
    source = mmc.NeutronSource()
    geometry = _minimal_geometry()
    with pytest.raises(ValueError, match="transport_secondary_photons"):
        mmc.Model(
            geometry, source=source,
            transport_secondary_photons=False,
            use_decay_photons=True)


def test_decay_photons_settings_getters():
    """Test getters for D1S-related model properties."""
    source = mmc.NeutronSource()
    geometry = _minimal_geometry()
    model = mmc.Model(
        geometry, source=source,
        transport_secondary_photons=True,
        use_decay_photons=True)
    assert model.use_decay_photons
    assert model.transport_secondary_photons


def test_transmutation_data():
    """Test the per-subsection yamc.transmutation_* module attributes.

    Only the two-state sources round-trip ``None`` this way. ``reactions`` and
    ``fission_yields`` can also be turned off, so ``None`` puts them back to
    the default library rather than clearing them; they are covered in
    test_transmutation_subsection_settings.py.
    """
    for attr in (
        "transmutation_decay_data",
        "transmutation_branch_ratios",
    ):
        setattr(mmc, attr, None)
        assert getattr(mmc, attr) is None

        setattr(mmc, attr, "chain.xml")
        assert getattr(mmc, attr) == "chain.xml"

        setattr(mmc, attr, "other_chain.xml")
        assert getattr(mmc, attr) == "other_chain.xml"

        setattr(mmc, attr, None)
        assert getattr(mmc, attr) is None
