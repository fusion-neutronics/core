"""X/Y torus surfaces (issue #367).

The X/Y tori share the ZTorus quartic with permuted axes (the exact
permutation identity is pinned by the rust unit tests). Here the
end-to-end check is statistical: the same ring rotated onto each axis
with the source moved along is physically identical, so the fluxes
agree within statistics (the isotropic source directions do not rotate
with the geometry, so the histories differ).
"""
import yamc


def _build(torus, source_pos):
    material = yamc.Material(composition={"Li6": 1.0}, density=0.46,
                             temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    cell = yamc.Cell(name="ring", region=torus.below, material=material, id=1)
    source = yamc.NeutronSource(position=source_pos,
                                energy=yamc.sources.Discrete([14.06e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", cells=cell,
                       particle="neutron")
    return yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                      tallies=[tally], source=source,
                      verbose=[])


def test_axis_tori_permutation_identity():
    a, b, c = 10.0, 2.0, 3.0
    zt = yamc.Torus(axis="z", r_major=a, r_minor=b, r_minor_2=c, boundary="vacuum")
    xt = yamc.Torus(axis="x", r_major=a, r_minor=b, r_minor_2=c, boundary="vacuum")
    yt = yamc.Torus(axis="y", r_major=a, r_minor=b, r_minor_2=c, boundary="vacuum")
    assert "Torus(axis=x)" in repr(xt) and "Torus(axis=y)" in repr(yt)

    # Source inside the ring at (10, 0, 0) for ZTorus; the permuted
    # positions put it at the same location relative to each torus.
    res_z = _build(zt, [10.0, 0.0, 0.0]).simulate_transport(
        total_particles=20_000, seed=42, threads=1)
    res_x = _build(xt, [0.0, 10.0, 0.0]).simulate_transport(
        total_particles=20_000, seed=42, threads=1)
    res_y = _build(yt, [10.0, 0.0, 0.0]).simulate_transport(
        total_particles=20_000, seed=42, threads=1)

    z = res_z["flux"].mean[0]
    z_std = res_z["flux"].standard_deviation[0]
    assert z > 0.0
    for name, r in (("XTorus", res_x), ("YTorus", res_y)):
        v = r["flux"].mean[0]
        v_std = r["flux"].standard_deviation[0]
        tol = 4.0 * (z_std**2 + v_std**2) ** 0.5
        assert abs(v - z) < tol, (
            f"{name} {v:.6e} vs ZTorus {z:.6e} exceeds 4 sigma {tol:.2e}"
        )
