"""The transmutation wheel must not re-export the transport surface (#452).

`yani` and `yamc` are built from one bindings crate, so every class the crate
defines is a candidate for both wheels. `register_classes` gates the
transport-only ones on the package name, and `build_stubs.py` gates the stub to
match. This is the test that notices when a new class is added without deciding
which wheel it belongs to.

Skipped unless the standalone yani wheel is installed. This suite runs against
that wheel alone, with no yamc in the interpreter, which is what makes the
absences asserted below mean something.
"""
import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

# Nothing an inventory calculation does needs these, and one of them is a test
# helper. They stay usable in yamc.
#
# The five model-level names came from an inline heredoc in the `yani-wheel` CI
# job, which asserted the same thing this file does and was the only check that
# job ever ran. This suite is run there now instead (issue #535), so the names
# live in one place.
TRANSPORT_ONLY = [
    "AngleDistribution",
    "Cell",
    "Geometry",
    "Model",
    "Particle",
    "PhotonSource",
    "ReactionProduct",
    "Sphere",
    "Tabulated",
    "Tally",
    "create_test_reaction_product",
    "sample_scatter_cosine",
]

# The stdlib imports the module-property shim needs. They are attributes of the
# module either way; what matters is that they are not advertised.
STDLIB = ["sys", "types"]


@pytest.mark.parametrize("name", TRANSPORT_ONLY)
def test_transport_surface_absent(name):
    assert not hasattr(yani, name), f"yani exposes the transport-only {name}"


@pytest.mark.parametrize("name", STDLIB)
def test_stdlib_not_public(name):
    assert name not in yani.__all__, f"`from yani import *` would bind {name}"


def test_all_is_defined_and_matches_the_module():
    # Without __all__, `import *` takes every non-underscore attribute, which is
    # how `sys` became public in the first place.
    assert yani.__all__, "yani defines no __all__"
    for name in yani.__all__:
        assert hasattr(yani, name), f"__all__ advertises the absent {name}"


def test_the_transmutation_surface_is_intact():
    for name in ("Material", "Enriched", "enriched", "Pulse", "Cooldown",
                 "PulseSchedule", "NeutronSource", "TransmutationResults",
                 "convert_transmutation", "data", "materials", "shapes",
                 "sources"):
        assert name in yani.__all__, f"yani no longer advertises {name}"


def test_the_stub_reexports_every_submodule():
    """Working at runtime is the half that hides the other half.

    `from yani._core import *` puts each submodule on the package, so
    `yani.shapes.CubeLump()` runs and every runtime assertion above passes,
    while type-checkers read the stubs and reject it. `shapes` shipped that
    way. This asserts against the stub the wheel installs, in the `X as X`
    spelling PEP 484 requires for a re-export.
    """
    stub = Path(yani.__file__).with_name("__init__.pyi")
    assert stub.is_file(), f"the wheel ships no top-level stub at {stub}"
    reexported = {
        alias.name
        for node in ast.parse(stub.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.level > 0
        for alias in node.names
        if alias.asname == alias.name
    }
    core = importlib.import_module("yani._core")
    submodules = {
        name
        for name in dir(core)
        if not name.startswith("_") and isinstance(getattr(core, name), ModuleType)
    }
    missing = sorted(submodules - reexported)
    assert not missing, f"yani/__init__.pyi does not re-export {missing}"


def test_the_shared_type_chain_still_resolves():
    # ReactionProduct is not top-level, but Nuclide.reactions reaches it, so the
    # types must still be registered and usable even when they are not exported.
    assert yani.Nuclide is not None and yani.Reaction is not None
