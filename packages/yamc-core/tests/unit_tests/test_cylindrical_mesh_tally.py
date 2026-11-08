"""
Tests for RegularCylindricalMesh and cylindrical mesh tally functionality.
"""

import math

import pytest
import yamc


def _keywords_available():
    try:
        m = yamc.Material(composition={"Li6": 1.0}, density=1.0, temperature=294)
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(), reason="keyword download requires download feature"
)


# --- construction / properties (no simulation needed) ---------------------


def test_cylindrical_mesh_creation():
    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(0.0, 10.0),
        z_bounds=(-5.0, 5.0),
        shape=(5, 4, 2),
    )
    assert mesh.num_bins == 40  # 5 * 4 * 2
    assert mesh.shape == [5, 4, 2]
    assert mesh.r_bounds == (0.0, 10.0)
    assert mesh.z_bounds == (-5.0, 5.0)
    assert mesh.phi_bounds == pytest.approx((0.0, 2 * math.pi))
    assert mesh.origin == [0.0, 0.0, 0.0]


def test_cylindrical_partial_phi_sector():
    # A sector that neither starts at 0 nor ends at 2*pi.
    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(2.0, 8.0),
        z_bounds=(0.0, 10.0),
        shape=(3, 2, 3),
        phi_bounds=(math.pi, 2 * math.pi),
    )
    assert mesh.num_bins == 18
    assert mesh.phi_bounds == pytest.approx((math.pi, 2 * math.pi))


def test_cylindrical_volume_is_non_uniform():
    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(0.0, 10.0),
        z_bounds=(0.0, 10.0),
        shape=(5, 1, 1),
    )
    # Outer rings are larger than inner rings.
    inner = mesh.element_volume(0)
    outer = mesh.element_volume(4)
    assert outer > inner
    # Sum of all ring volumes == full cylinder volume pi r^2 h.
    total = sum(mesh.element_volume(b) for b in range(mesh.num_bins))
    assert total == pytest.approx(math.pi * 10.0**2 * 10.0)


def test_cylindrical_mesh_validation():
    with pytest.raises(ValueError):
        yamc.RegularCylindricalMesh(r_bounds=(-1.0, 10.0), z_bounds=(0, 1), shape=(2, 1, 1))
    with pytest.raises(ValueError):
        yamc.RegularCylindricalMesh(r_bounds=(5.0, 1.0), z_bounds=(0, 1), shape=(2, 1, 1))
    with pytest.raises(ValueError):
        yamc.RegularCylindricalMesh(
            r_bounds=(0, 10), z_bounds=(0, 1), shape=(2, 1, 1), phi_bounds=(0.0, 10.0)
        )


def test_cylindrical_tally_creation():
    mesh = yamc.RegularCylindricalMesh(r_bounds=(0, 10), z_bounds=(-5, 5), shape=(4, 8, 3))
    tally = yamc.Tally(scores=["flux"], mesh=mesh)
    assert tally.n_mesh_bins == 4 * 8 * 3
    # The rectangular accessor is None; the cylindrical one round-trips.
    assert tally.mesh is None
    assert tally.cylindrical_mesh.shape == [4, 8, 3]
    # Per-bin volume getter works through the tally for any mesh type.
    assert tally.mesh_element_volume(0) == pytest.approx(mesh.element_volume(0))


# --- end-to-end simulation (needs nuclear data) ---------------------------


@requires_keywords
def test_cylindrical_tally_bin_count_matches_mesh():
    yamc.set_cross_section_data_entry("fendl-3.2d")

    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=12.0, boundary="vacuum")
    region = sphere.below
    material = yamc.Material(composition={"H1": 1.0}, density=0.001, temperature=294)
    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0),
    )

    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(0.0, 10.0),
        z_bounds=(-10.0, 10.0),
        shape=(5, 4, 5),  # 100 cells
    )
    tally = yamc.Tally(scores=["flux"], mesh=mesh)

    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source
    )
    results = model.simulate_transport(total_particles=200, seed=42)

    result = results[tally]
    assert len(result.mean) == 100
    assert len(result.standard_deviation) == 100
    assert all(v >= 0.0 for v in result.mean)
    assert any(v > 0.0 for v in result.mean)
