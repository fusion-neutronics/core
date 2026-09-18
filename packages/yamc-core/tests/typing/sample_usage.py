"""Static type-checker smoke test for the shipped stubs (pyright + mypy).

Every statement here is valid usage and must type-check with zero errors. The
``assert_type`` calls are load-bearing: ``assert_type(x, T)`` requires the
inferred static type to be exactly ``T``, so if the stubs failed to resolve
(no ``py.typed``) and ``yamc`` fell back to ``Any``, every one of them would
fail. This proves the stubs deliver real types, not ``Any``.

This file is never executed; it exists only to be checked statically. It does
not match pytest's ``test_*`` collection pattern, so pytest ignores it.
"""
from __future__ import annotations

from typing import assert_type

import yamc

# --- Constructors resolve to their real classes, not Any --------------------
mat = yamc.Material(composition={"Fe": 1.0}, density=7.874)
assert_type(mat, yamc.Material)

# raw per-nuclide atom densities via the from_atom_densities constructor
mat_sum = yamc.Material.from_atom_densities({"Fe": 1.0})
assert_type(mat_sum, yamc.Material)

tally = yamc.Tally(scores=["flux"])
assert_type(tally, yamc.Tally)

nuc = yamc.Nuclide("Fe56")
assert_type(nuc, yamc.Nuclide)

# --- Module-level data configuration is typed -------------------------------
yamc.cross_section_data = "endf-b8.1"
yamc.cross_section_data = {"Li6": "/data/Li6"}
yamc.cross_section_data = None
yamc.transmutation_decay_data = None

# `False` turns a subsection off, and only these two accept it. Without the
# `Literal[False]` in the stub these read as ordinary assignments at runtime
# and as type errors here, which is what the docs hit.
yamc.transmutation_reactions = False
yamc.transmutation_fission_yields = False
yamc.transmutation_reactions = "endf-b8.1"

# --- Pure-Python helper is typed --------------------------------------------
spec = yamc.enriched(0.5, target="Li6", percent=90.0)
assert_type(spec, yamc.Enriched)

# --- Curated material collections resolve to Material -----------------------
steel = yamc.materials.pnnl["Steel, Stainless 304"]
assert_type(steel, yamc.Material)
steel.name = "firstwall_material"

shield = yamc.materials.pnnl.material("Concrete, Ordinary (NIST)", density=1.8)
assert_type(shield, yamc.Material)

# --- Lump shapes resolve through the submodule, not as Any ------------------
# The submodules reach the package through `from ._core import *`, so this line
# runs whether or not a stub mentions them; it is the type-checkers that notice
# a missing re-export, and only if something asks them to.
assert_type(yamc.shapes.FoilLump(thickness=0.1), yamc.shapes.FoilLump)
assert_type(yamc.shapes.SphereLump(), yamc.shapes.SphereLump)

# --- Group structure edges come back as a real list[float] ------------------
assert_type(yamc.group_structure("UKAEA-1102"), list[float])
assert_type(yamc.group_structure_names(), list[str])

assert_type(yamc.materials.pnnl.search("concrete"), list[str])
assert_type(yamc.materials.pnnl.names(), list[str])
assert_type(yamc.materials.collections(), list[str])
assert_type(yamc.materials.pnnl.citation, str)

# --- Weight-window constructors take a rectangular mesh, not Any ------------
# `mesh` is `&Bound<PyAny>` in the bindings so the constructor can raise a
# refusal that names the rectangular-mesh-only limitation, which pyo3-stub-gen
# cannot see through. A `#[gen_stub(override_type(...))]` on the argument puts
# the real type back (issue #121), and `assert_type` below would not fail if it
# regressed to `Any`, so the negative cases in sample_errors.py are what guard
# it. These prove the accepting side still type-checks.
ww_mesh = yamc.RegularRectangularMesh(
    lower_left=[0.0, 0.0, 0.0], upper_right=[1.0, 1.0, 1.0], shape=[2, 2, 2]
)
assert_type(ww_mesh, yamc.RegularRectangularMesh)

bounds = yamc.WeightWindowBounds(mesh=ww_mesh, lower_bounds=[0.5] * 8)
assert_type(bounds, yamc.WeightWindowBounds)

# `particle` takes a single name or a list of them, which the stub now says.
gen_one = yamc.WeightWindowGeneratorDeGVR(mesh=ww_mesh, particle="photon")
assert_type(gen_one, yamc.WeightWindowGeneratorDeGVR)
gen_both = yamc.WeightWindowGeneratorDeGVR(
    mesh=ww_mesh, particle=["neutron", "photon"]
)
assert_type(gen_both, yamc.WeightWindowGeneratorDeGVR)
