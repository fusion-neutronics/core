#!/usr/bin/env python3
"""Check the typed public surface is consistent between the stubs and runtime.

Compares the symbols the ``.pyi`` stubs advertise against what the compiled
extension actually exposes, in both directions:

  * advertised-but-missing -- a name in a stub's ``__all__`` that cannot be
    imported at runtime. The stub lies: e.g. a ``#[gen_stub_pyclass]`` on a
    class that is never ``add_class``'d into the module, so type-checkers and
    the docs show a symbol users cannot import.
  * importable-but-unadvertised -- a public runtime class/function missing from
    the stub's ``__all__`` (a new binding that never got the ``gen_stub``
    macros), so type-checkers and the docs do not see it.
  * a submodule ``_core`` registers that no stub re-exports -- it reaches the
    package through ``from ._core import *`` and works when run, so neither
    check above has anything to say, while type-checkers reject
    ``yani.shapes.CubeLump``.

Runs over both wheels this repo builds, `yamc` and `yani` (issue #381); the
`yani` half is skipped when that wheel is not installed.

Exits non-zero with a report on any inconsistency. Run in CI next to the
stub-drift check.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = {
    "yamc": ROOT / "packages" / "yamc-core" / "python" / "yamc",
    "yani": ROOT / "packages" / "yani-core" / "python" / "yani",
}


def stub_all(pyi: Path) -> set[str]:
    """Return the union of every ``__all__`` (re)assignment in a stub file.

    Handles ``__all__ = [...]`` / ``= (...)`` and ``__all__ += [...]`` so a
    hand-edit that splits the list does not silently shrink the checked set.
    """
    names: set[str] = set()
    tree = ast.parse(pyi.read_text())
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AugAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and (
            isinstance(node.value, (ast.List, ast.Tuple))
        ):
            names.update(
                el.value
                for el in node.value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            )
    return names


def reexported(pyi: Path) -> set[str]:
    """Names a stub re-exports in the form type-checkers honour.

    PEP 484 counts an import in a stub as part of the public interface only
    when it is spelled ``X as X`` (or listed in ``__all__``). Plain
    ``from ._core import shapes`` binds the name for the stub's own use, and a
    checker still rejects ``yani.shapes``. ``declared_top_level`` below cannot
    tell the two apart, which is why this is separate: it would accept the
    spelling that leaves the defect in place.
    """
    names: set[str] = set()
    for node in ast.parse(pyi.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            names.update(a.name for a in node.names if a.asname == a.name)
    return names


def declared_top_level(pyi: Path) -> set[str]:
    """Public names a top-level stub explicitly adds (beyond ``from ._core import *``).

    Collects classes, functions, annotated module globals, and the names bound
    by relative imports (the re-exported submodules) -- i.e. exactly the
    yamc-side additions declared in ``yamc/__init__.pyi``. Stdlib imports such
    as ``from dataclasses import dataclass`` are excluded.
    """
    names: set[str] = set()
    for node in ast.parse(pyi.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            # relative import (e.g. ``from ._core import data as data``)
            names.update(
                a.asname or a.name for a in node.names if a.name != "*"
            )
    return {n for n in names if not n.startswith("_")}


def check(name: str, root: Path) -> list[str]:
    """Every inconsistency between `name`'s stubs and its runtime."""
    import importlib

    pkg = importlib.import_module(name)
    core = importlib.import_module(f"{name}._core")
    CORE_PYI = root / "_core" / "__init__.pyi"
    INIT_PYI = root / "__init__.pyi"
    yamc = pkg  # the messages below name the package they are checking

    errors: list[str] = []

    advertised = stub_all(CORE_PYI)
    # Asking the runtime which attributes are modules keeps this in step with
    # the bindings by construction. A hardcoded list here was a second copy of
    # scripts/build_stubs.py's, and a submodule missing from it was reported as
    # a missing gen_stub macro, which sends the reader to the wrong file.
    submodules = {
        n
        for n in dir(core)
        if not n.startswith("_") and isinstance(getattr(core, n), ModuleType)
    }
    runtime_public = {
        n for n in dir(core) if not n.startswith("_") and n not in submodules
    }

    missing = sorted(advertised - runtime_public)
    if missing:
        errors.append(
            "Advertised in the _core stub __all__ but NOT importable from "
                f"{name}._core (a gen_stub'd class that is never add_class'd, or a "
            "stale stub entry):\n" + "\n".join(f"  - {n}" for n in missing)
        )

    unadvertised = sorted(runtime_public - advertised)
    if unadvertised:
        errors.append(
            f"Importable from {name}._core but NOT in the _core stub __all__ "
            "(missing gen_stub macros, so type-checkers/docs miss it):\n"
            + "\n".join(f"  - {n}" for n in unadvertised)
        )

    # Re-export integrity: yamc/__init__.py does ``from ._core import *``, so
    # every advertised _core symbol must be reachable as ``yamc.X`` too.
    reexport_missing = sorted(n for n in advertised if not hasattr(yamc, n))
    if reexport_missing:
        errors.append(
            "Advertised in the _core stub __all__ but NOT re-exported onto the "
            f"top-level {name} module (the `from ._core import *` is incomplete):\n"
            + "\n".join(f"  - {n}" for n in reexport_missing)
        )

    # The Python-side additions declared in __init__.pyi (the data-config
    # globals, __version__, the re-exported submodules) must exist on the module.
    #
    # A missing stub is an error rather than a skip. It used to be `if
    # INIT_PYI.exists()`, which meant the check quietly did nothing when the file
    # was absent or the path was wrong, and reported success either way. yani had
    # no __init__.pyi at all, so its top-level surface went unchecked from the day
    # the wheel was split out. A guard that passes when its input is missing is
    # not a guard.
    if not INIT_PYI.exists():
        errors.append(
            f"{name} has no __init__.pyi at {INIT_PYI}, so its top-level surface "
            "is unchecked. Add the stub, declaring whatever __init__.py adds on "
            "top of `from ._core import *`."
        )
    else:
        declared = declared_top_level(INIT_PYI)
        top_missing = sorted(n for n in declared if not hasattr(yamc, n))
        if top_missing:
            errors.append(
                f"Declared in {name}/__init__.pyi but not present on the "
                f"{name} module:\n" + "\n".join(f"  - {n}" for n in top_missing)
            )

        # And the other direction, which is how `shapes` shipped unseen: a
        # submodule the extension registers arrives on the package through
        # `from ._core import *` and works perfectly at runtime, so nothing
        # here complained, while type-checkers rejected `yani.shapes.CubeLump`
        # because no stub re-exported it. `X as X` is the spelling checked,
        # since it is the only one they honour.
        undeclared = sorted(submodules - reexported(INIT_PYI))
        if undeclared:
            errors.append(
                f"Registered on {name}._core at runtime but NOT re-exported by "
                f"{name}/__init__.pyi as `from ._core import <name> as <name>`, "
                f"so type-checkers cannot see {name}.<submodule>:\n"
                + "\n".join(f"  - {n}" for n in undeclared)
            )

    if not errors:
        print(
            f"{name}: public surface OK, {len(advertised)} _core symbols "
            "advertised and all agree with the runtime."
        )
    return errors


def main() -> int:
    failed = False
    for name, root in PACKAGES.items():
        try:
            __import__(name)
        except ImportError:
            print(f"{name}: not installed, skipped")
            continue
        errors = check(name, root)
        if errors:
            failed = True
            print(f"\n{name}: public surface drift detected:\n")
            print("\n\n".join(errors))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
