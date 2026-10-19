"""Core expectations must not depend on another transport code."""

import ast
from pathlib import Path

import pytest

from .conftest import assert_self_reference


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_core_does_not_import_openmc():
    imports = []
    for directory in ("packages", "scripts", "examples", "pytests", "tools"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Call) and node.args:
                    function = node.func
                    name = getattr(function, "attr", getattr(function, "id", ""))
                    argument = node.args[0]
                    if name in ("__import__", "import_module", "importorskip"):
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                            modules = [argument.value]
                if any(module.split(".")[0] == "openmc" for module in modules):
                    imports.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not imports, "OpenMC comparisons belong in V&V:\n" + "\n".join(imports)


def test_self_reference_comparison_rejects_small_changes():
    with pytest.raises(AssertionError, match="flux"):
        assert_self_reference({"flux": 1.0}, {"flux": 1.00001}, "neutron")


def test_self_reference_comparison_checks_zero_bins():
    with pytest.raises(AssertionError, match="spectrum"):
        assert_self_reference([0.0, 1.0], [1e-15, 1.0], "spectrum")


def test_self_reference_comparison_requires_all_products():
    with pytest.raises(AssertionError, match="keys changed"):
        assert_self_reference({"Fe55": 1e-20}, {}, "transmutation")


def test_self_reference_comparison_allows_reduction_noise():
    assert_self_reference({"flux": [1.0]}, {"flux": [1.0 + 1e-14]}, "neutron")