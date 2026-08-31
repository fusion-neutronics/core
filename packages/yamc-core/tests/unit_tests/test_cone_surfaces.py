"""Cone surfaces (issue #365): the XCone/YCone/ZCone plus the
arbitrary-axis Cone, double-sheeted.

The general quadric (#366) is the oracle: a finite cone-capped cell
bounded by a ZCone written both ways must transport bit-identically.
"""
import math

import yamc


def _build(cone_surface):
    material = yamc.Material(composition={"Li6": 1.0}, density=0.46,
                             temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    # Finite region: inside the lower cone sheet, between two z-planes.
    z_lo = yamc.Plane(axis="z", offset=-10.0, boundary="vacuum")
    z_hi = yamc.Plane(axis="z", offset=-1.0, boundary="vacuum")
    cell = yamc.Cell(name="cone_frustum",
                     region=cone_surface.below & z_lo.above & z_hi.below,
                     material=material, id=1)
    source = yamc.NeutronSource(position=[0.0, 0.0, -5.0],
                                energy=yamc.sources.Discrete([14.06e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", cells=cell,
                       particle="neutron")
    return yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                      tallies=[tally], source=source,
                      verbose=[])


def test_zcone_as_quadric_is_bit_identical():
    r2 = 0.25
    cone = yamc.Cone(axis="z", x0=0.0, y0=0.0, z0=0.0,
                     opening_angle=math.degrees(math.atan(r2**0.5)),
                     boundary="vacuum")
    assert "Cone" in repr(cone)
    # (x)^2 + (y)^2 - r2 z^2 = 0 as a quadric.
    quadric = yamc.Quadric(a=1.0, b=1.0, c=-r2, boundary="vacuum")

    res_c = _build(cone).simulate_transport(total_particles=20_000, seed=42, threads=1)
    res_q = _build(quadric).simulate_transport(total_particles=20_000, seed=42, threads=1)
    c, q = res_c["flux"].mean[0], res_q["flux"].mean[0]
    assert c > 0.0
    # The two solvers are different algebraic arrangements of the same
    # quadratic, so boundary distances can differ in the last ULP;
    # identical histories agree to ~1e-15, statistics differ at ~1e-2.
    assert abs(c - q) < 1e-12 * c, f"cone {c!r} vs quadric {q!r}"


def test_axis_cones_construct():
    for axis in ("x", "y", "z"):
        s = yamc.Cone(axis=axis, x0=1.0, y0=2.0, z0=3.0,
                      opening_angle=math.degrees(math.atan(0.5**0.5)))
        assert "Cone" in repr(s)
    s = yamc.Cone(axis=(0.0, 1.0, 0.0), x0=1.0, y0=2.0, z0=3.0,
                  opening_angle=math.degrees(math.atan(0.5**0.5)))
    assert "Cone" in repr(s)
    # A non-unit axis is normalized by the constructor (no longer an error).
    s = yamc.Cone(axis=(0.0, 2.0, 0.0), x0=0.0, y0=0.0, z0=0.0,
                  opening_angle=math.degrees(math.atan(0.5**0.5)))
    assert "Cone" in repr(s)
