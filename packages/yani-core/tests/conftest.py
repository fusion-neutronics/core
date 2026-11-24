"""Put yani's process-global nuclear-data settings back after every test.

Nothing in this suite may import yamc, and that includes this file. The suite
exists to run in an interpreter that has the yani wheel and nothing else, which
is what proves yani-core ships none of the transport stack; importing yamc here
to reuse its restore fixture would make the transport wheel a precondition for
running the very tests that show it is not one.

The settings are process-global (one `CONFIG` behind the module properties in
packages/yani-core/python/yani/__init__.py), so a test that sets one and does not
put it back changes what every later test in the same process resolves. Tests
here set them (test_branching_sources.py, test_no_transmutation_data.py), and
this puts them back after each one.

yani is imported inside the fixture rather than at module scope. A missing wheel
raised from a conftest aborts the whole session at collection, where the same
failure in a test module is confined to that one file, and that is what the
`importorskip` in test_yani_public_surface.py relies on.
"""

import pytest

_TRANSMUTATION_SETTINGS = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
    "transmutation_branch_ratios",
)

# Reading `neutron_cross_section_data` cannot tell "per-nuclide map set" from "global
# keyword set": the property returns the map whenever it is non-empty and the
# keyword only when it is not. Looking up a nuclide that will never have an
# explicit entry applies the global default and nothing else, so it reports the
# keyword the property hides.
_ABSENT_NUCLIDE = "__yani_conftest_probe__"


def _cross_section_data_state(yani):
    """Both halves of `neutron_cross_section_data`: (per-nuclide view, global keyword)."""
    return yani.neutron_cross_section_data, yani.lookup_neutron_cross_section_data(_ABSENT_NUCLIDE)


@pytest.fixture(autouse=True)
def _restore_data_settings():
    """Snapshot yani's data settings, and undo whatever the test moved."""
    try:
        import yani
    except ImportError:
        # The wheel is absent, so there is nothing to save and the test modules
        # will report the absence themselves.
        yield
        return

    saved_transport = _cross_section_data_state(yani)
    saved_photon = yani.photon_cross_section_data
    saved_transmutation = {n: getattr(yani, n) for n in _TRANSMUTATION_SETTINGS}

    yield

    # Restore each setting only if that setting actually moved. Reading is free,
    # writing is not, because a nuclide is cached under a key derived from the
    # source the config resolved it from.
    for name, value in saved_transmutation.items():
        if getattr(yani, name) != value:
            setattr(yani, name, value)

    saved_property, saved_keyword = saved_transport
    if _cross_section_data_state(yani) != saved_transport:
        yani.neutron_cross_section_data = saved_property
        if isinstance(saved_property, dict) and saved_keyword is not None:
            yani.set_neutron_cross_section_data_entry(saved_keyword)
    if yani.photon_cross_section_data != saved_photon:
        yani.photon_cross_section_data = saved_photon
