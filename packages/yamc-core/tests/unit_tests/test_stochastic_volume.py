import math
import yamc


def test_sphere_volume():
    """Sphere volume should be within 2% of 4/3 * pi * r^3."""
    r = 5.0
    s = yamc.Sphere(x0=0, y0=0, z0=0, radius=r)
    region = s.below
    result = region.calculate_volume(samples=1_000_000, seed=42)
    analytical = 4.0 / 3.0 * math.pi * r**3
    assert abs(result.volume - analytical) / analytical < 0.02


def test_cell_volume_attribute_populated():
    """After cell.calculate_volume(), cell.volume should be set."""
    s = yamc.Sphere(x0=0, y0=0, z0=0, radius=3.0)
    cell = yamc.Cell(region=s.below)
    assert cell.volume is None
    result = cell.calculate_volume(samples=500_000, seed=42)
    assert cell.volume is not None
    assert cell.volume == result.volume


def test_geometry_populates_all_cells():
    """After geometry.calculate_volume(), all cell volumes should be set."""
    s1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=3.0)
    s2 = yamc.Sphere(x0=10, y0=0, z0=0, radius=2.0)
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="Test")
    cell1 = yamc.Cell(region=s1.below, material=mat)
    cell2 = yamc.Cell(region=s2.below, material=mat)
    geometry = yamc.Geometry([cell1, cell2])
    volumes = geometry.calculate_volume(samples=1_000_000, seed=42)

    assert 1 in volumes
    assert 2 in volumes
    # Check volumes are reasonable
    analytical_1 = 4.0 / 3.0 * math.pi * 3.0**3
    analytical_2 = 4.0 / 3.0 * math.pi * 2.0**3
    assert abs(volumes[1].volume - analytical_1) / analytical_1 < 0.02
    assert abs(volumes[2].volume - analytical_2) / analytical_2 < 0.05  # smaller sphere, more variance
    # Check cells in geometry have volumes set
    for c in geometry.cells:
        assert c.volume is not None


def test_region_volume():
    """Region.calculate_volume returns correct VolumeResult."""
    s = yamc.Sphere(x0=0, y0=0, z0=0, radius=4.0)
    region = s.below
    result = region.calculate_volume(samples=500_000, seed=42)
    analytical = 4.0 / 3.0 * math.pi * 4.0**3
    assert abs(result.volume - analytical) / analytical < 0.02
    assert result.std_dev > 0
    assert result.num_hits > 0


def test_custom_bounding_box():
    """Explicit bounding box works for regions with infinite extents."""
    # Half-space: everything below z=5. Infinite region.
    zp = yamc.Plane(axis="z", offset=5.0)
    region = zp.below  # below z=5

    # Providing a finite bounding box should work
    bbox = yamc.BoundingBox([-1, -1, 0], [1, 1, 5])
    result = region.calculate_volume(samples=100_000, bounding_box=bbox, seed=42)
    # The entire bbox is inside the region (z ranges 0..5, all < 5)
    bbox_vol = 2.0 * 2.0 * 5.0  # 20 cm^3
    assert abs(result.volume - bbox_vol) / bbox_vol < 0.01


def test_concentric_shells():
    """Inner sphere + shell volumes should sum to outer sphere volume."""
    r_inner = 3.0
    r_outer = 5.0
    s_inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=r_inner)
    s_outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=r_outer)

    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="Test")

    inner_cell = yamc.Cell(region=s_inner.below, material=mat)
    shell_cell = yamc.Cell(region=s_inner.above & s_outer.below, material=mat)
    geometry = yamc.Geometry([inner_cell, shell_cell])
    volumes = geometry.calculate_volume(samples=1_000_000, seed=42)

    total = volumes[1].volume + volumes[2].volume
    analytical_outer = 4.0 / 3.0 * math.pi * r_outer**3
    assert abs(total - analytical_outer) / analytical_outer < 0.02

    analytical_inner = 4.0 / 3.0 * math.pi * r_inner**3
    assert abs(volumes[1].volume - analytical_inner) / analytical_inner < 0.03


def test_seed_reproducibility():
    """Same seed should produce same results."""
    s = yamc.Sphere(x0=0, y0=0, z0=0, radius=5.0)
    cell = yamc.Cell(region=s.below)
    r1 = cell.calculate_volume(samples=100_000, seed=123)
    r2 = cell.calculate_volume(samples=100_000, seed=123)
    assert r1.volume == r2.volume
    assert r1.num_hits == r2.num_hits


def test_geometry_propagates_material_volume():
    """geometry.calculate_volume() should set material.volume inside cells."""
    r = 5.0
    s = yamc.Sphere(x0=0, y0=0, z0=0, radius=r)
    mat = yamc.Material(
        composition={"Fe": 1.0},
        density=7.87, units="g/cc",
        name="Iron",
        transmutable=True,
    )

    cell = yamc.Cell(region=s.below, material=mat)
    geometry = yamc.Geometry([cell])
    geometry.calculate_volume(samples=1_000_000, seed=42)

    # The material inside the geometry cell should now have volume set
    cell_from_geom = geometry.cells[0]
    assert cell_from_geom.material is not None
    assert cell_from_geom.material.volume is not None
    analytical = 4.0 / 3.0 * math.pi * r**3
    assert abs(cell_from_geom.material.volume - analytical) / analytical < 0.02


def test_shared_transmutable_material_errors():
    """Shared transmutable material across cells should raise on calculate_volume."""
    import pytest

    s1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=3.0)
    s2 = yamc.Sphere(x0=10, y0=0, z0=0, radius=2.0)
    mat = yamc.Material(
        composition={"Fe": 1.0},
        density=7.87, units="g/cc",
        name="Iron",
        transmutable=True,
    )

    cell1 = yamc.Cell(region=s1.below, material=mat)
    cell2 = yamc.Cell(region=s2.below, material=mat)
    geometry = yamc.Geometry([cell1, cell2])

    with pytest.raises(ValueError, match="Transmutable material 1 is used in 2 cells"):
        geometry.calculate_volume(samples=100_000, seed=42)
