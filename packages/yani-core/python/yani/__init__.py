"""yani -- transmutation and activation without transport.

A material, an irradiation schedule and a neutron spectrum in; inventories,
activities and decay heat out. No geometry, no transport, no Monte Carlo.

``yamc`` is a superset of this package: it ships the same classes plus the
transport stack. The two are alternatives, not companions. Installing both puts
two extension modules in one process, so ``yamc.Material`` and ``yani.Material``
are distinct types and the nuclear-data configuration exists twice.
"""

import sys
import types
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _pkg_version,
)

try:
    __version__ = _pkg_version("yani-core")
except _PackageNotFoundError:  # pragma: no cover - only without an installed dist
    __version__ = "0.0.0+unknown"

from yani._core import *  # noqa: F401,F403

# PyO3 submodules of _core, same shape as yamc's.
from yani._core import data, materials, shapes, sources  # noqa: F401

# ``X as X`` marks intentional re-exports (PEP 484) for type checkers.
from yani._core import (  # noqa: F811
    Cooldown as Cooldown,
    DataUncertainty as DataUncertainty,
    Estimate as Estimate,
    LineEstimate as LineEstimate,
    Material as Material,
    NeutronSource as NeutronSource,
    Pulse as Pulse,
    PulseSchedule as PulseSchedule,
    TransmutationResults as TransmutationResults,
    get_cross_section_data as _get_cross_section_data,
    get_transmutation_branch_ratios as _get_transmutation_branch_ratios,
    get_transmutation_decay_data as _get_transmutation_decay_data,
    get_transmutation_fission_yields as _get_transmutation_fission_yields,
    get_transmutation_reactions as _get_transmutation_reactions,
    lookup_cross_section_data as lookup_cross_section_data,  # noqa: F401
    set_cross_section_data as _set_cross_section_data,
    set_cross_section_data_entry as set_cross_section_data_entry,  # noqa: F401
    set_transmutation_branch_ratios as _set_transmutation_branch_ratios,
    set_transmutation_decay_data as _set_transmutation_decay_data,
    set_transmutation_fission_yields as _set_transmutation_fission_yields,
    set_transmutation_reactions as _set_transmutation_reactions,
)

# Dotted imports (`from yani.sources import Histogram`) are registered by the
# extension itself, next to the `yani._core.<name>` registration it has always
# done, so a new submodule cannot arrive with only one of the two names.


class _YaniModule(types.ModuleType):
    """Module subclass exposing the nuclear-data sources as properties, so
    ``yani.cross_section_data = "endf-b8.1"`` reads like a setting rather than
    a function call. Same five settings, and the same global state, as yamc.
    """

    @property
    def cross_section_data(self):
        return _get_cross_section_data()

    @cross_section_data.setter
    def cross_section_data(self, value):
        _set_cross_section_data(value)

    @property
    def transmutation_decay_data(self):
        return _get_transmutation_decay_data()

    @transmutation_decay_data.setter
    def transmutation_decay_data(self, value):
        _set_transmutation_decay_data(value)

    @property
    def transmutation_reactions(self):
        return _get_transmutation_reactions()

    @transmutation_reactions.setter
    def transmutation_reactions(self, value):
        _set_transmutation_reactions(value)

    @property
    def transmutation_fission_yields(self):
        return _get_transmutation_fission_yields()

    @transmutation_fission_yields.setter
    def transmutation_fission_yields(self, value):
        _set_transmutation_fission_yields(value)

    @property
    def transmutation_branch_ratios(self):
        return _get_transmutation_branch_ratios()

    @transmutation_branch_ratios.setter
    def transmutation_branch_ratios(self, value):
        _set_transmutation_branch_ratios(value)


sys.modules[__name__].__class__ = _YaniModule


# `Enriched` and `enriched()` are a pyclass and pyfunction in the bindings
# crate, arriving through the `_core` star-import above. They were a dataclass
# here and a second, unrelated dataclass in yamc, so the two wheels disagreed on
# the type and the generated stub had to name one wheel and import it into the
# other.


# Without this, `sys` and `types` above are public attributes of the package and
# `from yani import *` drags them in. yamc has had the same guard since it grew
# its own module-property shim. The transport-only classes are no longer a
# concern here: `register_classes` stopped adding them to this wheel (#452).
#
# `dir()` inside a comprehension has its own scope in Python 3, hence the temp.
_public_names = dir()
__all__ = sorted(
    name
    for name in _public_names
    if not name.startswith("_") and name not in {"sys", "types"}
)
del _public_names
