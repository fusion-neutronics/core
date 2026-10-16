"""Transport agreement for mesh-filled cells.

The two-region fixture is an exact axis-aligned unit cube split at x=0.5,
so a pure-CSG twin of the hybrid model exists: identical materials,
identical surfaces. Every per-region flux must agree within statistics.
This exercises particle location, min-of-two-boundaries tracking,
material assignment and tally addressing against an independent
representation of the same model.
"""

import numpy as np
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"

# Raised from 40 000 together with the change of criterion below. At 40 000
# "fuel" and "moderator" are thin low-flux regions carrying about 3% relative
# error each, so no honest criterion could have detected a 5% bias there. At
# this count they are at 0.58% and 0.60%, so the 4 sigma bound below fires on
# a bias of 3.3% or more (0.38% on "complement"). The pair of runs takes about
# 0.18 s.
N_PARTICLES = 1_000_000

# Agreement bound on the combined standard error, |a - b| / sqrt(sa^2 + sb^2).
# This replaced a flat `rel < 0.05`, which could not fail for the right reason:
# at the original 40 000 histories the fuel and moderator regions had a 3%
# relative error apiece, so a 5% bound sat inside their own noise and would
# fire or not depending on the RNG realisation. The Rust twin of this test
# (`crates/yamc/tests/hybrid_mesh_fill.rs::hybrid_matches_pure_csg_twin`) uses
# the z criterion for the same reason.
#
# Measured z: complement 0.77, fuel 0.73, moderator 0.19. Unlike the fixture
# parity tests in `test_arrow_fixture_parity.py`, the two models here differ
# enough (mesh fill inside a CSG sphere versus pure CSG) to decorrelate their
# random streams, so this really is a statistical comparison.
#
# Teeth: moving the twin's split plane from x = 0.5 to 0.55 fails at 11.3
# sigma (fuel) and 13.0 sigma (moderator); shrinking the twin's box to
# x <= 0.97 fails at 6.8 sigma (moderator).
Z_MAX = 4.0


def _material(name, nuclide, density):
    m = yamc.Material(
        composition={nuclide: 1.0}, density=density, name=name, temperature=294
    )
    m.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
    return m


def _source():
    return yamc.NeutronSource(
        position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )


def _run(geometry, tallies):
    """Tally total and its standard error per tally: {tally: (mean, std)}.

    ``aggregate_*`` is the per-history total statistic, so it carries the
    within-history correlation between bins.
    """
    model = yamc.Model(geometry=geometry, tallies=tallies, source=_source())
    results = model.simulate_transport(total_particles=N_PARTICLES, seed=42)
    return {
        t: (float(results[t].aggregate_mean), float(results[t].aggregate_std_dev))
        for t in tallies
    }


def test_hybrid_matches_pure_csg_twin():
    complement = _material("complement", "Be9", 1.85)
    fuel = _material("fuel", "Li6", 1.5)
    moderator = _material("moderator", "Be9", 3.7)

    # Hybrid: mesh cube embedded in a CSG sphere.
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    mesh = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": fuel, "moderator": moderator})
    chamber = yamc.Cell(
        region=sphere.below, material=complement, name="chamber", fill=mesh
    )
    hybrid = yamc.Geometry([chamber])
    # Embedded volumes are ordinary cells: address them via geometry.cells.
    by_name = {c.name: c for c in hybrid.cells}
    fuel_cell = next(c for n, c in by_name.items() if "fuel" in n)
    mod_cell = next(c for n, c in by_name.items() if "moderator" in n)
    hybrid_tallies = [
        yamc.Tally(scores=["flux"], name="complement", cells=by_name["chamber"]),
        yamc.Tally(scores=["flux"], name="fuel", cells=fuel_cell),
        yamc.Tally(scores=["flux"], name="moderator", cells=mod_cell),
    ]
    hybrid_flux = _run(hybrid, hybrid_tallies)

    # Pure-CSG twin: the same cube as two plane-bounded half boxes.
    x0, xm, x1 = (yamc.Plane(axis="x", offset=v) for v in (0.0, 0.5, 1.0))
    y0, y1 = (yamc.Plane(axis="y", offset=v) for v in (0.0, 1.0))
    z0, z1 = (yamc.Plane(axis="z", offset=v) for v in (0.0, 1.0))
    yz = y0.above & y1.below & z0.above & z1.below
    fuel_box = x0.above & xm.below & yz
    mod_box = xm.above & x1.below & yz
    whole_box = x0.above & x1.below & yz
    sphere2 = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    twin_cells = [
        yamc.Cell(region=sphere2.below & ~whole_box, material=complement,
                  name="chamber"),
        yamc.Cell(region=fuel_box, material=fuel, name="fuel"),
        yamc.Cell(region=mod_box, material=moderator, name="moderator"),
    ]
    twin = yamc.Geometry(twin_cells)
    twin_tallies = [
        yamc.Tally(scores=["flux"], name=c.name, cells=c) for c in twin_cells
    ]
    twin_flux = _run(twin, twin_tallies)

    for h, t in zip(hybrid_tallies, twin_tallies):
        hybrid_mean, hybrid_std = hybrid_flux[h]
        twin_mean, twin_std = twin_flux[t]
        assert twin_mean > 0.0 and hybrid_mean > 0.0
        combined = np.sqrt(hybrid_std**2 + twin_std**2)
        assert combined > 0.0, (
            f"{h.name}: both runs must report a standard error "
            f"(hybrid {hybrid_std}, twin {twin_std})"
        )
        z = abs(hybrid_mean - twin_mean) / combined
        rel = abs(hybrid_mean - twin_mean) / twin_mean
        assert z < Z_MAX, (
            f"{h.name}: hybrid {hybrid_mean} +/- {hybrid_std} vs twin "
            f"{twin_mean} +/- {twin_std} differ by {rel:.2%} ({z:.2f} sigma)"
        )


def test_same_material_fill_is_invisible():
    iron = _material("beryllium", "Be9", 1.85)
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")

    mesh = yamc.MeshGeometry(TWO_REGION_ARROW, {"fuel": iron, "moderator": iron})
    filled_cell = yamc.Cell(region=sphere.below, material=iron, fill=mesh)
    filled_tally = yamc.Tally(scores=["flux"], name="flux")
    filled = _run(yamc.Geometry([filled_cell]), [filled_tally])[filled_tally]

    sphere2 = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    plain_cell = yamc.Cell(region=sphere2.below, material=iron)
    plain_tally = yamc.Tally(scores=["flux"], name="flux")
    plain = _run(yamc.Geometry([plain_cell]), [plain_tally])[plain_tally]

    filled_mean, filled_std = filled
    plain_mean, plain_std = plain
    assert plain_mean > 0.0 and filled_mean > 0.0
    combined = np.sqrt(filled_std**2 + plain_std**2)
    assert combined > 0.0
    z = abs(filled_mean - plain_mean) / combined
    assert z < Z_MAX, (
        f"filled {filled_mean} +/- {filled_std} vs plain {plain_mean} +/- "
        f"{plain_std} ({z:.2f} sigma)"
    )


def _touching_boxes_arrow(path):
    """Two 10 cm cubes meeting at x = 10, each with its own surface, written
    as a mesh made without imprinting (tracked discontinuously)."""
    from yamc._core import mesh_to_arrow

    def box(x0, x1):
        v = [
            [x0, 0, 0], [x1, 0, 0], [x1, 10, 0], [x0, 10, 0],
            [x0, 0, 10], [x1, 0, 10], [x1, 10, 10], [x0, 10, 10],
        ]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
        return v, [t for a, b, c, d in quads for t in ([a, b, c], [a, c, d])]

    va, ta = box(0.0, 10.0)
    vb, tb = box(10.0, 20.0)
    n = len(ta)
    mesh_to_arrow(
        path,
        [c for v in va + vb for c in v],
        [i for t in ta for i in t] + [i + 8 for t in tb for i in t],
        [1] * n + [2] * n,
        [1] * n + [2] * n,
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}',
        surface_volumes_json="[[0, null], [1, null]]",
        imprinted=False,
        contact_tolerance=1e-3,
    )


def test_a_mesh_made_without_imprinting_fills_a_cell_like_its_csg_twin(tmp_path):
    """Two touching cubes, each with its own surface, filling a CSG sphere,
    give the same fluxes as the cubes built from planes: crossings between the
    fill's solids are resolved as in a whole-model mesh."""
    path = str(tmp_path / "boxes.arrow")
    _touching_boxes_arrow(path)

    def materials():
        return {"a": _material("a", "Be9", 1.85), "b": _material("b", "Fe56", 7.874)}

    def run(geometry, cells, seed):
        tallies = [yamc.Tally(scores=["flux"], cells=c, name=c.name) for c in cells]
        model = yamc.Model(
            geometry=geometry,
            tallies=tallies,
            source=yamc.NeutronSource(
                energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(5.0, 5.0, 5.0)
            ),
        )
        results = model.simulate_transport(total_particles=100_000, seed=seed)
        assert not model.lost_particles
        return [
            (float(results[t].mean[0]), float(results[t].standard_deviation[0]))
            for t in tallies
        ]

    def chamber():
        return yamc.Sphere(x0=10.0, y0=5.0, z0=5.0, radius=30.0, boundary="vacuum").below

    mesh = yamc.MeshGeometry(path, materials())
    assert mesh.discontinuous
    filled = yamc.Geometry([yamc.Cell(region=chamber(), name="chamber", fill=mesh)])
    fill_cells = [next(c for c in filled.cells if c.name.endswith(f"_{t}")) for t in "ab"]
    fill_flux = run(filled, fill_cells, 1)

    mats = materials()
    x = [yamc.Plane(axis="x", offset=v) for v in (0.0, 10.0, 20.0)]
    y = [yamc.Plane(axis="y", offset=v) for v in (0.0, 10.0)]
    z = [yamc.Plane(axis="z", offset=v) for v in (0.0, 10.0)]
    yz = y[0].above & y[1].below & z[0].above & z[1].below
    twin = [
        yamc.Cell(region=chamber() & ~(x[0].above & x[2].below & yz), name="chamber"),
        yamc.Cell(region=x[0].above & x[1].below & yz, material=mats["a"], name="a"),
        yamc.Cell(region=x[1].above & x[2].below & yz, material=mats["b"], name="b"),
    ]
    twin_flux = run(yamc.Geometry(twin), twin[1:], 2)

    for tag, (mean, sd), (ref, ref_sd) in zip("ab", fill_flux, twin_flux):
        zscore = (mean - ref) / np.hypot(sd, ref_sd)
        assert abs(zscore) < Z_MAX, (
            f"{tag}: fill {mean} +- {sd}, CSG twin {ref} +- {ref_sd} ({zscore:.1f} sigma)"
        )

