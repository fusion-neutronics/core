"""General quadric surface (issue #366).

The quadric is a catch-all primitive; the dedicated variants
are its oracles. A sphere written as a quadric must reproduce the
Sphere-based model bit-for-bit: same seed, same physics, the only
difference is which distance solver the geometry calls.
"""
import yamc


def _build(boundary_surface):
    material = yamc.Material(composition={"Li6": 1.0}, density=0.46,
                             temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    cell = yamc.Cell(name="ball", region=boundary_surface.below,
                     material=material, id=1)
    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                energy=yamc.sources.Discrete([14.06e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", cells=cell,
                       particle="neutron")
    return yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                      tallies=[tally], source=source,
                      verbose=[])


def test_quadric_accepts_name():
    # name= is accepted for consistency with Sphere/Cylinder/Plane/Cone/Torus
    # (issue #504). It is additive: a named quadric is otherwise identical to
    # an unnamed one.
    named = yamc.Quadric(a=1.0, b=1.0, c=1.0, k=-100.0, boundary="vacuum",
                         name="ellipsoid")
    unnamed = yamc.Quadric(a=1.0, b=1.0, c=1.0, k=-100.0, boundary="vacuum")
    assert "Quadric" in repr(named)
    assert repr(named) == repr(unnamed)


def test_sphere_as_quadric_is_bit_identical():
    r = 10.0
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=r,
                         boundary="vacuum")
    # x^2 + y^2 + z^2 - r^2 = 0
    quadric = yamc.Quadric(a=1.0, b=1.0, c=1.0, k=-r * r,
                           boundary="vacuum")
    assert "Quadric" in repr(quadric)

    res_s = _build(sphere).simulate_transport(total_particles=20_000, seed=42, threads=1)
    res_q = _build(quadric).simulate_transport(total_particles=20_000, seed=42, threads=1)
    assert res_s["flux"].mean[0] > 0.0
    assert repr(res_s["flux"].mean) == repr(res_q["flux"].mean), (
        f"sphere {res_s['flux'].mean[0]!r} vs quadric "
        f"{res_q['flux'].mean[0]!r}"
    )
