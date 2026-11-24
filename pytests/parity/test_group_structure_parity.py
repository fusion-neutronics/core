"""The two wheels must register the same group structures.

`group_structure` and `group_structure_names` are added outside the
`if transport` gate in `register_classes` (crates/yani-python/src/lib.rs), so
both wheels carry the pair and neither may drift from the other.

This is the one group-structure test that needs both wheels in one interpreter,
which is why it lives here instead of alongside the yamc-only tests in
packages/yamc-core/tests/unit_tests/test_group_structures.py: that suite is run
with the yamc wheel alone.
"""

import pytest

import yamc


def test_the_transmutation_wheel_has_the_same_registry():
    """Both wheels register the pair from the one bindings crate."""
    yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")
    assert yani.group_structure_names() == yamc.group_structure_names()
    assert yani.group_structure("CCFE-709") == yamc.group_structure("CCFE-709")
