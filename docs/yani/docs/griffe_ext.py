"""griffe extension: present PyO3's ``__new__`` as the class constructor.

PyO3 maps ``#[new]`` to ``__new__``, but mkdocstrings builds the class call
signature from ``__init__`` (and ``merge_init_into_class`` only merges
``__init__``). So a PyO3 class's typed constructor shows up as an awkward
``__new__`` member instead of on the class -- which no Python user calls
directly.

For any class that has a typed ``__new__`` and no ``__init__``, copy ``__new__``
to ``__init__``. With ``merge_init_into_class: true`` the reference then renders
the real constructor on the class itself -- ``Material(composition: ...,
density: float | Literal['sum'], ...)`` -- and the ``__new__`` member is hidden
by the normal dunder filter.
"""
from __future__ import annotations

import copy

import griffe


class PyO3Constructor(griffe.Extension):
    def on_class(self, *, cls: griffe.Class, **kwargs) -> None:
        members = cls.members
        new = members.get("__new__")
        if "__init__" in members or not isinstance(new, griffe.Function):
            return
        init = copy.copy(new)
        init.name = "__init__"
        cls.set_member("__init__", init)


class PublicApi(griffe.Extension):
    """Mark the wheels' top-level names as public for the API reference.

    ``yamc.__all__`` is computed at runtime from ``dir()`` (so it tracks new
    ``_core`` exports automatically); static analysis reduces it to an empty
    list, and griffe then falls back to import heuristics that hide every
    explicitly imported name (``Model``, ``Cell``, ``Tally``, ...) -- griffe
    does not yet honor the ``from x import y as y`` re-export convention.
    Mirror the runtime ``__all__`` rule instead: every non-underscore
    top-level member (minus the stdlib imports) is public.

    ``yani`` re-exports explicitly (``X as X``) rather than through a dynamic
    ``__all__``, so it needs this for the second reason only: the re-export
    convention griffe does not yet honor.
    """

    def on_package(self, *, pkg: griffe.Module, **kwargs) -> None:
        if pkg.name not in {"yamc", "yani"}:
            return
        for name, member in pkg.members.items():
            if not name.startswith("_") and name not in {"sys", "types", "dataclass"}:
                member.public = True
