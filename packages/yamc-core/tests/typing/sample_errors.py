"""Negative type-checker cases: each line is a deliberate type error.

The CI typing gate runs a checker over this file and asserts it reports an
error on every marked line -- proving the stubs are *precise* (they actually
reject wrong types), not merely present. This file is never executed.
"""
from __future__ import annotations

import yamc

# density is a float; a bare str like "dense" is invalid.
yamc.Material(composition={"Fe": 1.0}, density="dense")  # ERROR

# scores expects a sequence, not a bare int.
yamc.Tally(scores=5)  # ERROR

# the cross_section_data setter rejects an int (module-level union attribute).
yamc.cross_section_data = 123  # ERROR

# a group structure is named with a str, not with its group count.
yamc.group_structure(1102)  # ERROR

# `True` is not a source: the subsections that take a bool take only False,
# which is why the stub says Literal[False] and not bool.
yamc.transmutation_reactions = True  # ERROR

# and the other three subsections take no bool at all.
yamc.transmutation_decay_data = False  # ERROR

# Weight windows are defined on a rectangular mesh only. A cylindrical mesh is
# the mistake the runtime refusal was written for, and with `mesh: typing.Any`
# in the stub it type-checked clean and only failed when run (issue #121).
_cyl = yamc.RegularCylindricalMesh(r_bounds=(0.0, 1.0), z_bounds=(0.0, 1.0), shape=[2, 2, 2])
yamc.WeightWindowBounds(mesh=_cyl, lower_bounds=[0.5])  # ERROR
yamc.WeightWindowGeneratorDeGVR(mesh=_cyl)  # ERROR

# `particle` is a name or a list of names, never a number.
_rect = yamc.RegularRectangularMesh(lower_left=[0.0] * 3, upper_right=[1.0] * 3, shape=[1, 1, 1])
yamc.WeightWindowGeneratorDeGVR(mesh=_rect, particle=5)  # ERROR
