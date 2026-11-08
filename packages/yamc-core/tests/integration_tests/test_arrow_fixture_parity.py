"""Transport parity for the ``box.arrow`` mesh fixture.

Before this, ``box.arrow`` was never transported through. Its only references
were the Arrow loader unit tests in ``crates/yamt/src/io/arrow.rs`` and the
graveyard triangle-count and bounding-box checks in
``packages/yamc-core/tests/unit_tests/test_vacuum_boundary.py``: all of them
assert that the file parses and that adding a vacuum boundary appends twelve
triangles, none of them that a particle tracked through the mesh ends up in
the right place.

The fixture is an exact axis-aligned unit cube centred on the origin with a
single ``mat:water`` volume, so a pure-CSG twin built from six planes is the
same solid with no discretisation error and no volume mismatch. It is a
surface mesh with no tetrahedra and no ``boundary:vacuum`` group, which makes
it the one fixture that exercises the ``graveyard_offset`` path in transport:
particles leave the mesh skin into the implicit complement and are killed on
the generated graveyard box. The CSG twin reproduces that exactly, void gap
included.

This test lives in Python because ``yamc.MeshGeometry(..., graveyard_offset=)``
is the public entry point for both halves of what it covers, and because the
graveyard is a Python-facing constructor argument whose existing coverage
(triangle counts) stops at the geometry.
"""

import numpy as np
import yamc

BOX_ARROW = "crates/yamt/tests/data/box.arrow"

# Histories per run. Chosen for POWER, not for speed: the measured relative
# error at this count is 0.025% on each side, a combined standard error of
# 0.035%, so a 5% bias would land at 141 sigma. The pair of runs takes about
# 0.31 s (a 1 cm cube with a vacuum boundary makes for short histories).
N_PARTICLES = 1_000_000

# Gap between the mesh skin and the generated vacuum boundary. Anything
# positive works; the gap is void, so it only delays leakage.
GRAVEYARD_OFFSET = 2.0

# Half width of the fixture's cube. The CSG twin is built from this so the
# teeth of the test can be re-checked by perturbing one number.
HALF = 0.5

# Agreement bound on the combined standard error, |a - b| / sqrt(sa^2 + sb^2).
# Not a fixed percentage: the two models are physically identical, so the only
# scale the difference can be judged against is their own Monte Carlo error.
Z_MAX = 4.0


def _material(name, nuclide, density):
    m = yamc.Material(
        composition={nuclide: 1.0}, density=density, name=name, temperature=294
    )
    m.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
    return m


def _source():
    return yamc.NeutronSource(
        position=[0.0, 0.0, 0.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )


def _run(geometry, tally):
    """Tally total and its standard error.

    ``aggregate_*`` is the per-history total statistic, so it carries the
    within-history correlation between bins; summing per-bin errors in
    quadrature would assume independence and over-state the error.
    """
    model = yamc.Model(geometry=geometry, tallies=[tally], source=_source())
    results = model.simulate_transport(total_particles=N_PARTICLES, seed=42)
    assert not model.lost_particles, f"run lost particles: {model.lost_particles}"
    result = results[tally]
    return float(result.aggregate_mean), float(result.aggregate_std_dev)


def _assert_agrees(label, mesh, csg):
    mesh_mean, mesh_std = mesh
    csg_mean, csg_std = csg
    assert mesh_mean > 0.0 and csg_mean > 0.0, (
        f"{label}: both sides must score (mesh {mesh_mean}, CSG {csg_mean})"
    )
    combined = np.sqrt(mesh_std**2 + csg_std**2)
    assert combined > 0.0, (
        f"{label}: both sides must report a standard error "
        f"(mesh {mesh_std}, CSG {csg_std})"
    )
    z = abs(mesh_mean - csg_mean) / combined
    rel = abs(mesh_mean - csg_mean) / csg_mean
    assert z < Z_MAX, (
        f"{label}: mesh {mesh_mean} +/- {mesh_std} vs CSG {csg_mean} +/- {csg_std} "
        f"differ by {rel:.3%} ({z:.2f} sigma)"
    )


def test_box_arrow_transport_matches_the_pure_csg_twin():
    """Flux through the meshed cube equals flux through the CSG cube.

    Measured: 0.025% relative error per side, agreeing at 0.00 sigma. The two
    runs share a seed, a source and the same physics along the same path, so
    they draw the same random numbers in the same order and currently agree to
    the last bit. The z test is still the criterion rather than an equality
    check, because renumbering the shared RNG stream in one path and not the
    other would decorrelate them without anything being wrong.

    Teeth: shrinking the CSG twin's inner cube from a half width of 0.5 to
    0.485 (3%) moves it by 3.21%, which is 88.8 sigma and fails; a 1% shrink
    (0.495) is still 29.2 sigma.
    """
    # "water" is the fixture's physical-group tag; any material may be bound
    # to it. Fe56 is used because it is one of the nuclides with test data.
    mesh_material = _material("steel", "Fe56", 7.8)
    mesh = yamc.MeshGeometry(
        BOX_ARROW, {"water": mesh_material}, graveyard_offset=GRAVEYARD_OFFSET
    )
    mesh_tally = yamc.Tally(scores=["flux"], materials=mesh_material, name="box")
    mesh_result = _run(mesh, mesh_tally)

    # Pure-CSG twin: the same cube, inside the same void gap, inside the same
    # vacuum-bounded graveyard box.
    csg_material = _material("steel", "Fe56", 7.8)
    inner = [yamc.Plane(axis=a, offset=s * HALF) for a in "xyz" for s in (-1, 1)]
    outer = [
        yamc.Plane(axis=a, offset=s * (HALF + GRAVEYARD_OFFSET), boundary="vacuum")
        for a in "xyz"
        for s in (-1, 1)
    ]

    def _box(planes):
        region = planes[0].above & planes[1].below
        for lo, hi in zip(planes[2::2], planes[3::2]):
            region = region & lo.above & hi.below
        return region

    cube = _box(inner)
    graveyard = _box(outer)
    twin = yamc.Geometry([
        yamc.Cell(region=cube, material=csg_material, name="box"),
        yamc.Cell(region=graveyard & ~cube, name="void"),
    ])
    csg_tally = yamc.Tally(scores=["flux"], materials=csg_material, name="box")
    csg_result = _run(twin, csg_tally)

    _assert_agrees("box.arrow flux", mesh_result, csg_result)
