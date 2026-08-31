# Public type surface for the `yani` package.
#
# Hand-maintained, but the bulk is the `from ._core import *` re-export, which
# auto-tracks the generated `_core` stub. Only the genuinely Python-side
# additions from __init__.py are declared explicitly below. Type-checkers use
# this file in preference to __init__.py.
#
# Its absence used to be invisible: scripts/check_public_surface.py guarded the
# top-level check on this file existing, so yani's top-level surface went
# unchecked while reporting success. That guard is now a hard failure, so this
# file has to keep up with __init__.py.
import typing

from ._core import *  # noqa: F401,F403

# PyO3 submodules of _core, re-exported by __init__.py under the names people
# write (`yani.sources.Histogram`, not `yani._core.sources.Histogram`).
from ._core import data as data
from ._core import materials as materials
from ._core import shapes as shapes
from ._core import sources as sources

# Package version, read at import time from the installed distribution metadata
# (importlib.metadata) -- single source of truth, no hardcoded duplicate.
__version__: str

# Module-level data configuration, backed by property getters/setters on the
# module object (see _YaniModule in __init__.py). Modelled as annotated module
# globals -- the idiomatic stub for a module-level settable value.
#
# Each extension module carries its own copy of the CONFIG behind these, so
# setting one here does not touch yamc's when both wheels are installed.
#
# `transmutation_reactions` and `transmutation_fission_yields` accept `False` as
# well as a library keyword or path: False turns the subsection off, `None`
# resets to the default library. The other three take no bool.
#
# `Literal[False]` and not `bool`, matching the setters: `True` is rejected at
# runtime, so a plain bool here would advertise a spelling that raises.
cross_section_data: str | dict[str, str] | None
transmutation_decay_data: str | None
transmutation_reactions: str | typing.Literal[False] | None
transmutation_fission_yields: str | typing.Literal[False] | None
transmutation_branch_ratios: str | None
