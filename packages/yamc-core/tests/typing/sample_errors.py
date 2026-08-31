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
