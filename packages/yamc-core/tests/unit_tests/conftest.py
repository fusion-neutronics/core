"""Point the per-subsection transmutation settings at the local v2 chain
fixture for unit tests.

Replaces the old single ``yamc.transmutation_data`` / per-call
``transmutation_file`` API: sources are now the four ``yamc.transmutation_*``
settings, so tests set the three network-forming parts to the local fixture.
"""
from pathlib import Path

import pytest

import yamc

_CHAIN = str(
    Path(__file__).resolve().parents[4] / "tests" / "transmutation-endf-b8.1-sfr.arrow"
)


@pytest.fixture(autouse=True)
def _configure_transmutation_chain():
    yamc.transmutation_decay_data = _CHAIN
    yamc.transmutation_reactions = _CHAIN
    yamc.transmutation_fission_yields = _CHAIN
    yield
    yamc.transmutation_decay_data = None
    yamc.transmutation_reactions = None
    yamc.transmutation_fission_yields = None
