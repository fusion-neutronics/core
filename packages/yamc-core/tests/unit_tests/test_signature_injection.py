"""Typed ``__signature__`` injection for Jupyter (``?`` / shift-tab).

PyO3 exposes parameter names and defaults via ``__text_signature__`` but no
type annotations. ``yamc`` overlays the annotations from the shipped ``_core``
type stub onto each class's signature at import, so the signature line carries
real, cleanly rendered types. These tests guard that behaviour and that
construction is unaffected by the metadata.
"""
import inspect

import yamc


def test_material_signature_is_typed():
    sig = inspect.signature(yamc.Material)
    density = sig.parameters["density"]
    assert density.annotation is not inspect.Parameter.empty
    # density is a plain float (the units= kwarg carries its meaning).
    assert "float" in str(density.annotation)
    composition = sig.parameters["composition"]
    assert "dict" in str(composition.annotation)


def test_signature_renders_cleanly():
    rendered = str(inspect.signature(yamc.Material))
    # The injected, docs-matching type for the density parameter.
    assert "density: float" in rendered
    # No import-qualifier noise.
    assert "builtins." not in rendered
    assert "typing." not in rendered
    # A plain str annotation would render quoted (``density: 'float ...'``);
    # the unquoted-repr wrapper must prevent that.
    assert ": 'float" not in rendered


def test_multiple_classes_have_typed_signatures():
    for name in ("Material", "Model", "Tally", "Cell", "NeutronSource"):
        cls = getattr(yamc, name)
        sig = inspect.signature(cls)
        typed = [
            p
            for p in sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        ]
        assert typed, f"{name} should expose at least one typed parameter"


def test_injection_does_not_break_construction():
    # Setting ``__signature__`` is metadata only; construction must still work.
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.0)
    assert mat is not None
