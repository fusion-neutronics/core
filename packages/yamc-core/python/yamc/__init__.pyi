# Public type surface for the `yamc` package.
#
# Hand-maintained, but the bulk is the `from ._core import *` re-export, which
# auto-tracks the generated `_core` stub. Only the genuinely Python-side
# additions from __init__.py are declared explicitly below. Type-checkers use
# this file in preference to __init__.py.
import typing

from ._core import *  # noqa: F401,F403
from ._core import data as data
from ._core import materials as materials
from ._core import parallel as parallel
from ._core import shapes as shapes
from ._core import sources as sources

# Package version, read at import time from the installed distribution metadata
# (importlib.metadata) -- single source of truth, no hardcoded duplicate.
__version__: str

# `Enriched` and `enriched` are not declared here. They stopped being a
# Python-side dataclass when they became a pyclass and a pyfunction in the
# bindings crate, and they arrive through the star-import above, in
# `_core`'s `__all__`. The hand-written pair that stayed behind shadowed them
# with a different type: it advertised `yamc.Enriched(0.5, "Li6", 90.0)`,
# which type-checked and then raised at runtime. yani's stub dropped them at
# the time; this one did not.

# Module-level data configuration, backed by property getters/setters on the
# module object (see _YamcModule in __init__.py). Modelled as annotated module
# globals -- the idiomatic stub for a module-level settable value.
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
