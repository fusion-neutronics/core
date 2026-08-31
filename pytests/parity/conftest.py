"""Put the process-global nuclear-data settings back after every parity test.

This directory belongs to neither package, so it inherits no conftest. The
autouse restore fixture in packages/yamc-core/tests/conftest.py covered these
files while they lived under pytests/ next to it, and after the split it does
not reach them: `_inventory` in test_yani_package_parity.py sets
`cross_section_data` and three `transmutation_*` sources on yamc AND on yani and
puts neither back, so without this nothing cleans up after it. One `pytest` run
walks all three suites in a single process and whichever suite runs next
inherits whatever was left set, which is a failure that shows up nowhere near
the test that caused it.

The two packages are imported inside the fixture rather than at module scope. A
missing wheel raised from a conftest aborts the whole session at collection,
where the same failure in a test module is confined to that one file, and that
is what the `importorskip` calls in these tests still rely on.

`YAMC_PARITY_REQUIRE_WHEELS` turns that around for the one job whose whole
purpose is to have both, which is what issue #535 is about: these tests had
never run anywhere, because no CI job installed both wheels and `importorskip`
therefore skipped on every run, including the test whose docstring says it
guards that splitting the bindings out (#381) changed no physics.
"""

import importlib
import os

import pytest

# yani's module-property shim (packages/yani-core/python/yani/__init__.py)
# mirrors yamc's setter for setter, and each extension module carries its own
# copy of the `CONFIG` behind them, so the two are saved and restored
# independently.
_PACKAGES = ("yamc", "yani")

_TRANSMUTATION_SETTINGS = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
    "transmutation_branch_ratios",
)

# Reading `cross_section_data` cannot tell "per-nuclide map set" from "global
# keyword set": the property returns the map whenever it is non-empty and the
# keyword only when it is not. Looking up a nuclide that will never have an
# explicit entry applies the global default and nothing else, so it reports the
# keyword the property hides. Same probe the yamc suite's conftest uses.
_ABSENT_NUCLIDE = "__parity_conftest_probe__"


def _installed():
    """The packages whose globals these tests can move.

    A wheel that is not installed is skipped rather than reported: whether a
    missing wheel is a skip or a failure is each test module's decision, and
    raising it from here would take the whole session down instead of one file.
    """
    found = []
    for name in _PACKAGES:
        try:
            found.append(importlib.import_module(name))
        except ImportError:
            continue
    return found


# Everything this directory needs in one interpreter. `yamc` and `yani` are the
# two wheels; pyarrow is here because test_yani_yamc_converter_parity reads the
# written Arrow back through it, and an environment missing it can compare
# nothing.
_REQUIRED = _PACKAGES + ("pyarrow.ipc",)


def pytest_configure(config):
    """Under `YAMC_PARITY_REQUIRE_WHEELS=1`, a missing wheel is the failure.

    Skipping is the right answer on a developer machine with one wheel
    installed, and the wrong one in the job that exists to install both: a run
    that skips every test reports green, which is how these tests went
    unexecuted for as long as they did. The switch is set by that job and by
    nothing else, so `pytest pytests/parity` on a laptop behaves as before.

    Really imported, not merely located. A `find_spec` hit says a wheel is on
    the path, not that its extension module loads, and a wheel built against
    the wrong ABI fails at exactly that step.
    """
    if os.environ.get("YAMC_PARITY_REQUIRE_WHEELS") != "1":
        return
    missing = []
    for name in _REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name} ({exc})")
    if missing:
        raise pytest.UsageError(
            "YAMC_PARITY_REQUIRE_WHEELS=1, so the parity suite must run with "
            "everything it compares installed, but these did not import: "
            + "; ".join(missing)
        )


def _cross_section_data_state(pkg):
    """Both halves of `cross_section_data`: (per-nuclide view, global keyword)."""
    return pkg.cross_section_data, pkg.lookup_cross_section_data(_ABSENT_NUCLIDE)


def _snapshot(pkg):
    return (
        _cross_section_data_state(pkg),
        {name: getattr(pkg, name) for name in _TRANSMUTATION_SETTINGS},
    )


def _restore(pkg, saved):
    saved_transport, saved_transmutation = saved

    # Restore each setting only if that setting actually moved. Reading is free,
    # writing is not, because a nuclide is cached under a key derived from the
    # source the config resolved it from.
    for name, value in saved_transmutation.items():
        if getattr(pkg, name) != value:
            setattr(pkg, name, value)

    saved_property, saved_keyword = saved_transport
    if _cross_section_data_state(pkg) != saved_transport:
        # `None` is the only input that clears both halves; setting a dict
        # merges into the existing map and setting a keyword leaves the map
        # alone. So clear first, then put back whichever halves were set,
        # keyword first so the map is not overwritten by it.
        pkg.cross_section_data = None
        if saved_keyword is not None:
            pkg.cross_section_data = saved_keyword
        if saved_property is not None:
            pkg.cross_section_data = saved_property


@pytest.fixture(autouse=True)
def _restore_data_settings():
    """Snapshot both packages' data settings, and undo whatever the test moved."""
    saved = [(pkg, _snapshot(pkg)) for pkg in _installed()]
    yield
    for pkg, state in saved:
        _restore(pkg, state)
