"""Tests for RegularRectangularMesh.from_domain()."""

import pytest
import yamc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_region(radius=10.0):
    """Return a spherical region centred at the origin."""
    s = yamc.Sphere(radius=radius)
    return s.below


def _box_region(x0, x1, y0, y1, z0, z1):
    """Return a box region bounded by planes."""
    return (
        yamc.Plane(axis="x", offset=x0).above & yamc.Plane(axis="x", offset=x1).below
        & yamc.Plane(axis="y", offset=y0).above & yamc.Plane(axis="y", offset=y1).below
        & yamc.Plane(axis="z", offset=z0).above & yamc.Plane(axis="z", offset=z1).below
    )


# ---------------------------------------------------------------------------
# Test: from_domain with Region
# ---------------------------------------------------------------------------

def test_from_domain_region():
    r = 10.0
    region = _sphere_region(r)
    mesh = yamc.RegularRectangularMesh.from_domain(region, shape=100)
    assert mesh.lower_left == pytest.approx([-r, -r, -r])
    assert mesh.upper_right == pytest.approx([r, r, r])
    # Cubic sphere → equal bins on each axis
    assert mesh.shape[0] == mesh.shape[1] == mesh.shape[2]


# ---------------------------------------------------------------------------
# Test: from_domain with Cell
# ---------------------------------------------------------------------------

def test_from_domain_cell():
    region = _sphere_region(5.0)
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="dummy")
    cell = yamc.Cell(region=region, material=mat)
    mesh = yamc.RegularRectangularMesh.from_domain(cell, shape=100)
    assert mesh.lower_left == pytest.approx([-5, -5, -5])
    assert mesh.upper_right == pytest.approx([5, 5, 5])


# ---------------------------------------------------------------------------
# Test: from_domain with Geometry (multi-cell)
# ---------------------------------------------------------------------------

def test_from_domain_geometry():
    r1 = _box_region(0, 10, 0, 5, 0, 2)
    r2 = _box_region(20, 30, 10, 15, 5, 8)
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="Test")
    c1 = yamc.Cell(region=r1, material=mat)
    c2 = yamc.Cell(region=r2, material=mat)
    geom = yamc.Geometry([c1, c2])
    mesh = yamc.RegularRectangularMesh.from_domain(geom, shape=100)
    assert mesh.lower_left == pytest.approx([0, 0, 0])
    assert mesh.upper_right == pytest.approx([30, 15, 8])


# ---------------------------------------------------------------------------
# Test: from_domain with BoundingBox
# ---------------------------------------------------------------------------

def test_from_domain_bounding_box():
    bb = yamc.BoundingBox([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
    mesh = yamc.RegularRectangularMesh.from_domain(bb, shape=200)
    assert mesh.lower_left == pytest.approx([-1, -2, -3])
    assert mesh.upper_right == pytest.approx([1, 2, 3])


# ---------------------------------------------------------------------------
# Test: integer dimension → roughly cubic voxels for asymmetric bbox
# ---------------------------------------------------------------------------

def test_from_domain_single_int_dimension():
    """An asymmetric bbox (10 × 20 × 30) with N=1000 should produce
    bins where each axis count is proportional to its width."""
    bb = yamc.BoundingBox([0, 0, 0], [10, 20, 30])
    mesh = yamc.RegularRectangularMesh.from_domain(bb, shape=1000)
    nx, ny, nz = mesh.shape
    # Ratios should be roughly 1:2:3
    assert ny == pytest.approx(2 * nx, abs=1)
    assert nz == pytest.approx(3 * nx, abs=1)
    # Total should be in the right ballpark
    total = nx * ny * nz
    assert 500 < total < 2000, f"Total bins {total} not near 1000"


# ---------------------------------------------------------------------------
# Test: default dimension (no dimension arg) → ~1000 total cells
# ---------------------------------------------------------------------------

def test_from_domain_default_dimension():
    bb = yamc.BoundingBox([0, 0, 0], [10, 10, 10])
    mesh = yamc.RegularRectangularMesh.from_domain(bb)
    total = mesh.num_bins
    assert 500 < total < 2000, f"Default total bins {total} not near 1000"


# ---------------------------------------------------------------------------
# Test: explicit [nx, ny, nz] dimension
# ---------------------------------------------------------------------------

def test_from_domain_list_dimension():
    bb = yamc.BoundingBox([0, 0, 0], [10, 20, 30])
    mesh = yamc.RegularRectangularMesh.from_domain(bb, shape=[5, 10, 15])
    assert mesh.shape == [5, 10, 15]


# ---------------------------------------------------------------------------
# Test: infinite bounding box raises ValueError
# ---------------------------------------------------------------------------

def test_from_domain_infinite_bbox_raises():
    # A single halfspace has infinite extent
    plane = yamc.Plane(axis="x", offset=0.0)
    region = plane.above  # everything above x=0 → infinite in all other axes
    with pytest.raises(ValueError, match="infinite"):
        yamc.RegularRectangularMesh.from_domain(region, shape=100)


# ---------------------------------------------------------------------------
# Test: invalid domain type raises TypeError
# ---------------------------------------------------------------------------

def test_from_domain_invalid_type_raises():
    with pytest.raises(TypeError):
        yamc.RegularRectangularMesh.from_domain("not a domain", shape=100)


# ---------------------------------------------------------------------------
# Test: material without MeshGeometry raises ValueError
# ---------------------------------------------------------------------------

def test_from_domain_material_without_meshgeom_raises():
    region = _sphere_region(5.0)
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="water")
    with pytest.raises(ValueError, match="MeshGeometry"):
        yamc.RegularRectangularMesh.from_domain(region, shape=100, material=mat)


# ---------------------------------------------------------------------------
# Test: the explicit constructor also accepts a single int (total voxel count),
# distributed over the box the corners already define.
# ---------------------------------------------------------------------------

def test_explicit_constructor_single_int_cubic():
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-1000, -1000, -1000], upper_right=[1000, 1000, 1000], shape=1_000_000)
    nx, ny, nz = mesh.shape
    assert nx == ny == nz                       # cubic box -> equal per-axis counts
    assert 0.5e6 < nx * ny * nz < 2e6           # ~ the requested total


def test_explicit_constructor_single_int_asymmetric():
    # 10 x 20 x 30 box, N=1000 -> per-axis counts in ~1:2:3 ratio
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0, 0, 0], upper_right=[10, 20, 30], shape=1000)
    nx, ny, nz = mesh.shape
    assert ny == pytest.approx(2 * nx, abs=1)
    assert nz == pytest.approx(3 * nx, abs=1)


def test_explicit_constructor_list_shape_unchanged():
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0, 0, 0], upper_right=[10, 10, 10], shape=[8, 8, 8])
    assert mesh.shape == [8, 8, 8]


def test_explicit_constructor_tuple_shape():
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0, 0, 0], upper_right=[10, 10, 10], shape=(4, 5, 6))
    assert mesh.shape == [4, 5, 6]
