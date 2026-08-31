# Shared pytest fixtures for yamc test suite

import os
from pathlib import Path

import pytest

import yamc

# The repo root: tests -> yamc-core -> packages -> root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root():
    """Run the suite from the repo root, wherever pytest was launched.

    About 175 paths across this suite are written CWD-relative ("tests/Li6.arrow"
    and friends), which resolve only when pytest is started from the repo root.
    Since the packages were split out, `cd packages/yamc-core && pytest` looks
    like it should work -- the root `pytest.ini` is still discovered from there
    -- and it failed on every one of them (issue #539).

    Rewriting all 175 would be a large diff for no behaviour change, so the CWD
    is normalised here instead and the literals are left alone. Constants that
    are read at COLLECTION time cannot wait for this fixture and are resolved
    from `__file__` directly; see `regression_tests/conftest.py`.
    """
    previous = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)

# The process-global data settings, all backed by the one `CONFIG` in
# yamc-nuclide. A test that sets one of these and does not put it back changes
# what every later test in the same pytest process resolves.
_TRANSMUTATION_SETTINGS = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
    "transmutation_branch_ratios",
)

# `cross_section_data` is really two fields: a per-nuclide path map and a global
# library keyword. `get_cross_section_data` returns the map when it is non-empty
# and the keyword only when it is not, so reading the property alone cannot
# tell "map set" from "map and keyword set". Looking up a nuclide that will
# never have an explicit entry applies the global default and nothing else, so
# it reports the keyword the property hides.
_ABSENT_NUCLIDE = "__yamc_conftest_probe__"


def _cross_section_data_state():
    """Both halves of `cross_section_data`: (per-nuclide view, global keyword)."""
    return yamc.cross_section_data, yamc.lookup_cross_section_data(_ABSENT_NUCLIDE)


@pytest.fixture(autouse=True)
def _restore_data_settings():
    """Put the global data settings back after every test.

    Most test files already set what they need in their own autouse fixture,
    but not all of them undid it, and a leak is invisible in the file that
    causes it: it only ever shows up in whichever file happens to run next.

    The case that motivated this: `test_material_transmute.py` set
    `cross_section_data = "tests"` with no teardown, so `test_transmutation[Co58]`
    inherited it and found a `tests/Fe58.arrow` evaluation it does not see on
    its own. Fe59 went from absent to 2.26e-14 and was then compared against a
    reference built from a different library, failing by 2.55%. Both files
    passed when run separately.

    Restoring centrally rather than per file means tests nobody has written
    yet get the guarantee too. This runs outside any conftest or module
    fixture (a parent autouse fixture tears down last), so it undoes their
    writes as well as the test body's.
    """
    saved_transport = _cross_section_data_state()
    saved_transmutation = {n: getattr(yamc, n) for n in _TRANSMUTATION_SETTINGS}

    yield

    # Restore each setting only if that setting actually moved. Reading the
    # settings is free (measured: no change to suite wall-clock), but writing
    # them is not, because a nuclide is cached under a key derived from the
    # source the config resolves it to. 104 of ~1160 tests really do change
    # something, and putting those back costs the suite ~15s of reloading;
    # writing the settings back unconditionally instead of only when they
    # moved made it far worse. The 15s buys order-independence, so that a
    # subset run (a single file, `-k`, or one of the path-selected tiers CI
    # runs) gives the same answer as a full run.
    for name, value in saved_transmutation.items():
        if getattr(yamc, name) != value:
            setattr(yamc, name, value)

    saved_property, saved_keyword = saved_transport
    if _cross_section_data_state() != saved_transport:
        # `None` is the only input that clears both halves; setting a dict
        # merges into the existing map and setting a keyword leaves the map
        # alone. So clear first, then put back whichever halves were set,
        # keyword first so the map is not overwritten by it.
        yamc.cross_section_data = None
        if saved_keyword is not None:
            yamc.cross_section_data = saved_keyword
        if saved_property is not None:
            yamc.cross_section_data = saved_property
