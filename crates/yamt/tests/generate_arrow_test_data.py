#!/usr/bin/env python3
"""Rebuild every Arrow mesh test fixture from CAD.

Run from the yamc repo root, in a venv with cadquery and a built yamc:

    maturin develop --release
    python crates/yamt/tests/generate_arrow_test_data.py

Pass fixture filenames to rebuild only those:

    python crates/yamt/tests/generate_arrow_test_data.py box.arrow

Each fixture is built as a CadQuery solid and pushed through the real
``yamc.cad`` pipeline (imprint, surface mesh, optional tet mesh, Arrow write),
so the files always match what the shipping code emits.

This replaces the old ``regenerate_arrow_test_data.py``, which read the
existing files and rewrote them in the current schema. That reformatting could
not repair (and so silently preserved) geometric defects in the data: three of
``box.arrow``'s six cube faces were wound inward while all six senses said
outward, which made the unit cube measure 0.0 cm3, and it survived every past
"regeneration" (issue #324).

After writing, each fixture is validated: the triangles bounding every volume
must be a closed, consistently oriented manifold, wound outward for the sense
recorded in ``yamc.surface_volumes``, giving a positive divergence-theorem
volume that matches the CAD; every tet must be positively oriented and the tets
must sum to the volume they fill. ``fixture_winding_agrees_with_recorded_senses``
in ``crates/yamt/src/io/arrow.rs`` re-checks the committed files on every
``cargo test``, so a stale or hand-edited fixture cannot pass CI.

The mesh content is reproducible run to run, but the files are not byte-stable:
arrow-rs writes schema metadata by iterating a ``HashMap``, so the three
``yamc.*`` key/value pairs land in a different order each process. Expect a
small binary diff from a no-op regeneration; compare the decoded content, not
the bytes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import cadquery as cq
import pyarrow.ipc as ipc

from yamc.cad import CadToYamc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Coarsest tet edge length that still tetrahedralizes the unit-scale fixture
# boxes; larger values do not reduce the tet count further (a 1 cm box gives the
# minimal 6-tet Kuhn-style decomposition, a 0.5 cm half-box gives 6 tets each).
TET_EDGE_LENGTH = 2.0

# Sphere faceting for sphere_in_cube: 1.1 rad of angular deflection gives 78
# triangles on the r=5 sphere, matching the triangle budget the fixture has
# always had (the flat cube faces need only 2 triangles each).
SPHERE_TOLERANCE = 1.0
SPHERE_ANGULAR_TOLERANCE = 1.1

# Curved faceting for nested_cylinders_tets, chosen so both cylinders come out
# as regular polygons with a closed-form volume (see build_nested_cylinders).
CYLINDER_ANGULAR_TOLERANCE = 1.1


# ---------------------------------------------------------------------------
# CAD models
# ---------------------------------------------------------------------------


def build_box():
    """Unit cube centred on the origin, one material."""
    asm = cq.Assembly()
    asm.add(cq.Workplane("XY").box(1, 1, 1), name="box")
    return asm, ["water"]


def build_two_region():
    """Unit cube split at x = 0.5 into two half-boxes, two materials."""
    asm = cq.Assembly()
    asm.add(cq.Workplane("XY").box(0.5, 1, 1).translate((0.25, 0.5, 0.5)), name="fuel")
    asm.add(
        cq.Workplane("XY").box(0.5, 1, 1).translate((0.75, 0.5, 0.5)), name="moderator"
    )
    return asm, ["fuel", "moderator"]


def build_cube():
    """Unit cube spanning [0, 1]^3, one material, tet filled."""
    asm = cq.Assembly()
    asm.add(cq.Workplane("XY").box(1, 1, 1).translate((0.5, 0.5, 0.5)), name="cube")
    return asm, ["water"]


def build_sphere_in_cube():
    """Sphere of radius 5 inside a cube spanning [-10, 10]^3."""
    sphere = cq.Workplane().sphere(5)
    shell = cq.Workplane().box(20, 20, 20).cut(cq.Workplane().sphere(5))
    asm = cq.Assembly()
    asm.add(sphere, name="fuel")
    asm.add(shell, name="moderator")
    return asm, ["fuel", "moderator"]


def ngon_prism_volume(n, radius, height):
    """Volume of a regular ``n``-gon prism of circumradius ``radius``.

    A cylinder is tessellated as a prism over the inscribed regular polygon, so
    this is the exact volume of the faceted solid (not of the true cylinder).
    """
    return 0.5 * n * radius * radius * math.sin(2.0 * math.pi / n) * height


# Facet counts OCC produces for the two cylinders at
# CYLINDER_ANGULAR_TOLERANCE. They are asserted through the expected volumes
# below, so a tessellation change fails this script rather than silently
# rewriting the fixture.
INNER_CYLINDER_SIDES = 12
OUTER_CYLINDER_SIDES = 15
CYLINDER_HEIGHT = 4.0
INNER_CYLINDER_RADIUS = 1.0
OUTER_CYLINDER_RADIUS = 2.0

_INNER_PRISM = ngon_prism_volume(
    INNER_CYLINDER_SIDES, INNER_CYLINDER_RADIUS, CYLINDER_HEIGHT
)
_OUTER_PRISM = ngon_prism_volume(
    OUTER_CYLINDER_SIDES, OUTER_CYLINDER_RADIUS, CYLINDER_HEIGHT
)


def build_nested_cylinders():
    """Rod inside a shell, sharing one curved surface, both tet filled.

    The point of this fixture is the interface: surface 1 bounds both volumes,
    it is curved, and a conformal tet fill has to reproduce it exactly from
    each side. The flat-sided fixtures cannot exercise that.

    Both radii land on regular polygons (12 sides at r=1, 15 at r=2), so each
    volume has a closed-form expected volume and the inner rod comes out at
    exactly 12.0 cm3.
    """
    solid = cq.Workplane("XY").cylinder(CYLINDER_HEIGHT, INNER_CYLINDER_RADIUS)
    shell = (
        cq.Workplane("XY")
        .cylinder(CYLINDER_HEIGHT, OUTER_CYLINDER_RADIUS)
        .cut(cq.Workplane("XY").cylinder(CYLINDER_HEIGHT, INNER_CYLINDER_RADIUS))
    )
    asm = cq.Assembly()
    asm.add(solid, name="fuel")
    asm.add(shell, name="moderator")
    return asm, ["fuel", "moderator"]


FIXTURES = [
    # (filename, builder, tet_volumes, tag_exterior_vacuum, mesh kwargs, expected volumes)
    ("box.arrow", build_box, None, False, {}, [1.0]),
    ("two_region.arrow", build_two_region, None, False, {}, [0.5, 0.5]),
    ("cube.arrow", build_cube, ["water"], True, {}, [1.0]),
    (
        "two_region_tets.arrow",
        build_two_region,
        ["fuel", "moderator"],
        True,
        {},
        [0.5, 0.5],
    ),
    (
        "sphere_in_cube.arrow",
        build_sphere_in_cube,
        None,
        True,
        {
            "tolerance": SPHERE_TOLERANCE,
            "angular_tolerance": SPHERE_ANGULAR_TOLERANCE,
        },
        # Faceted, so filled in from the mesh itself; see check() below.
        [None, None],
    ),
    (
        "nested_cylinders_tets.arrow",
        build_nested_cylinders,
        ["fuel", "moderator"],
        True,
        {"angular_tolerance": CYLINDER_ANGULAR_TOLERANCE},
        [_INNER_PRISM, _OUTER_PRISM - _INNER_PRISM],
    ),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(name, builder, tet_volumes, tag_exterior_vacuum, mesh_kwargs):
    """Build one fixture from CAD and write it to ``data/<name>``."""
    path = os.path.join(DATA_DIR, name)
    assembly, material_tags = builder()

    converter = CadToYamc()
    converter.add_cadquery_object(assembly, material_tags)
    converter.mesh(
        tet_volumes=tet_volumes,
        target_edge_length=TET_EDGE_LENGTH if tet_volumes else None,
        **mesh_kwargs,
    )

    boundary_tags = None
    if tag_exterior_vacuum:
        boundary_tags = {"vacuum": converter.exterior_surface_ids()}

    converter.to_arrow(path, boundary_tags=boundary_tags)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _read(path):
    f = ipc.open_file(path)
    meta = {k.decode(): v.decode() for k, v in (f.schema.metadata or {}).items()}
    verts, tris, tri_sid, tets, tet_vid = [], [], [], [], []
    for i in range(f.num_record_batches):
        batch = f.get_batch(i)
        col = batch.column(0)
        if col.null_count < len(col):
            verts += [tuple(v) for v in col.to_pylist()]
        col = batch.column(1)
        if col.null_count < len(col):
            n = len(col) - col.null_count
            tris += col.to_pylist()[:n]
            tri_sid += batch.column(2).to_pylist()[:n]
        col = batch.column(4)
        if col.null_count < len(col):
            n = len(col) - col.null_count
            tets += col.to_pylist()[:n]
            tet_vid += batch.column(5).to_pylist()[:n]
    return meta, verts, tris, tri_sid, tets, tet_vid


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cone_volume(v0, v1, v2, origin=(0.0, 0.0, 0.0)):
    """Signed volume of the tetrahedron (origin, v0, v1, v2)."""
    a, b, c = _sub(v0, origin), _sub(v1, origin), _sub(v2, origin)
    return _dot(a, _cross(_sub(b, a), _sub(c, a))) / 6.0


def _signed_tet_volume(v0, v1, v2, v3):
    return _dot(_sub(v1, v0), _cross(_sub(v2, v0), _sub(v3, v0))) / 6.0


def check(path, expected_volumes):
    """Validate winding, senses, volumes and tet orientation. Returns volumes."""
    meta, verts, tris, tri_sid, tets, tet_vid = _read(path)
    surface_volumes = json.loads(meta["yamc.surface_volumes"])
    groups = json.loads(meta["yamc.physical_groups"])
    n_volumes = sum(1 for g in groups.values() if g.get("dim") == 3)
    implicit_complement = n_volumes

    tris_by_surface = defaultdict(list)
    for tri, sid in zip(tris, tri_sid):
        tris_by_surface[sid].append(tri)

    problems = []
    measured = []
    for vol in range(n_volumes):
        # Boundary of this volume, flipped where the stored sense is Reverse so
        # the whole set should end up outward-oriented.
        oriented = []
        for index, (fwd, rev) in enumerate(surface_volumes):
            sid = index + 1
            if fwd is None:
                fwd = implicit_complement
            elif rev is None:
                rev = implicit_complement
            for tri in tris_by_surface[sid]:
                if fwd == vol:
                    oriented.append((tuple(tri), sid))
                if rev == vol:
                    oriented.append(((tri[0], tri[2], tri[1]), sid))

        # Weld by coordinate: the surface mesh may carry per-face duplicated
        # vertices, so edge matching has to be positional, not by index.
        key = {}
        for i, v in enumerate(verts):
            key[i] = tuple(round(c, 9) + 0.0 for c in v)

        directed = defaultdict(int)
        for tri, _ in oriented:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                directed[(key[a], key[b])] += 1
        if any(n != 1 for n in directed.values()):
            problems.append(f"volume {vol}: repeated directed edge (non-manifold)")
        if any((b, a) not in directed for a, b in directed):
            problems.append(f"volume {vol}: open boundary (unmatched directed edge)")

        total = sum(
            _cone_volume(verts[t[0]], verts[t[1]], verts[t[2]]) for t, _ in oriented
        )
        measured.append(total)

        # A closed, consistently oriented boundary has a well-defined inside,
        # and a positive divergence sum means that orientation is the outward
        # one. Together those two are what the stored senses must satisfy.
        # (A per-surface sign test is not valid here: an enclosed cavity
        # surface, like the sphere inside the shell, is legitimately negative.)
        if total <= 0.0:
            problems.append(f"volume {vol}: divergence volume {total:+.6f} not positive")

        expected = expected_volumes[vol] if vol < len(expected_volumes) else None
        if expected is not None and not math.isclose(
            total, expected, rel_tol=1e-9, abs_tol=1e-9
        ):
            problems.append(f"volume {vol}: volume {total} != expected {expected}")

    if tets:
        negative = sum(
            1 for t in tets if _signed_tet_volume(*[verts[i] for i in t]) <= 0.0
        )
        if negative:
            problems.append(f"{negative} negatively oriented tets")
        summed = defaultdict(float)
        for tet, vid in zip(tets, tet_vid):
            summed[vid] += abs(_signed_tet_volume(*[verts[i] for i in tet]))
        for vid, filled in summed.items():
            surface_volume = measured[vid - 1]
            if not math.isclose(filled, surface_volume, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(
                    f"tets of volume {vid - 1} fill {filled}, "
                    f"bounding surface encloses {surface_volume}"
                )

    print(
        f"  {os.path.basename(path):<22} "
        f"verts={len(verts):>4} tris={len(tris):>4} tets={len(tets):>3} "
        f"surfaces={len(surface_volumes):>2} "
        f"volumes=[{', '.join(f'{v:.6f}' for v in measured)}]"
    )
    for problem in problems:
        print(f"    FAIL: {problem}")
    return problems


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Named fixtures only, when asked. The files are not byte-stable (see the
    # module docstring), so regenerating all six to add or fix one would leave
    # five files with a binary diff and no content change.
    wanted = sys.argv[1:]
    known = {name for name, *_ in FIXTURES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        print(f"unknown fixture(s): {', '.join(unknown)}")
        print(f"known: {', '.join(sorted(known))}")
        return 2
    selected = [f for f in FIXTURES if not wanted or f[0] in wanted]

    failures = []
    for name, builder, tets, vacuum, kwargs, expected in selected:
        path = generate(name, builder, tets, vacuum, kwargs)
        failures += check(path, expected)
    if failures:
        print(f"\n{len(failures)} problem(s) found")
        return 1
    print(f"\nAll {len(selected)} fixture(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
