import pytest
from yamc import Plane, Geometry, Cell, Material, Sphere, Cylinder, Torus

def test_xplane_creation():
    s = Plane(axis="x", offset=1.0, surface_id=42)
    assert s.surface_id == 42
    assert s.evaluate((1.0, 0.0, 0.0)) == pytest.approx(0.0)

def test_yplane_creation():
    s = Plane(axis="y", offset=2.0, surface_id=43)
    assert s.surface_id == 43
    assert s.evaluate((0.0, 2.0, 0.0)) == pytest.approx(0.0)

def test_zplane_creation():
    s = Plane(axis="z", offset=3.0, surface_id=44)
    assert s.surface_id == 44
    assert s.evaluate((0.0, 0.0, 3.0)) == pytest.approx(0.0)

def test_sphere_creation():
    s = Sphere(x0=1.0, y0=2.0, z0=3.0, radius=5.0, surface_id=45)
    assert s.surface_id == 45
    assert s.evaluate((1.0, 2.0, 8.0)) == pytest.approx(0.0)

def test_cylinder_creation():
    s = Cylinder(axis="y", x0=1.0, z0=3.0, radius=2.0, surface_id=46)
    assert s.surface_id == 46
    # Point at radius from origin, perpendicular to axis (Y axis)
    assert s.evaluate((3.0, 2.0, 3.0)) == pytest.approx(0.0)

def test_zcylinder_creation():
    s = Cylinder(axis="z", x0=1.0, y0=2.0, radius=3.0, surface_id=47)
    assert s.surface_id == 47
    # Point at radius from center in XY plane
    assert s.evaluate((4.0, 2.0, 0.0)) == pytest.approx(0.0)

def test_boundary_default():
    """Test that surfaces default to None boundary type."""
    s = Plane(axis="x", offset=1.0, surface_id=42)
    assert s.boundary is None

def test_xcylinder_creation():
    s = Cylinder(axis="x", y0=1.0, z0=2.0, radius=3.0, surface_id=50)
    assert s.surface_id == 50
    # Point at (0, 1, 5): perpendicular distance to axis = 3 = radius
    assert s.evaluate((0.0, 1.0, 5.0)) == pytest.approx(0.0)

def test_ycylinder_creation():
    s = Cylinder(axis="y", x0=1.0, z0=2.0, radius=3.0, surface_id=51)
    assert s.surface_id == 51
    # Point at (4, 0, 2): perpendicular distance to axis = 3 = radius
    assert s.evaluate((4.0, 0.0, 2.0)) == pytest.approx(0.0)

def test_ztorus_creation():
    s = Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0, surface_id=52)
    assert s.surface_id == 52
    # Point on outer equator: (4, 0, 0)
    assert s.evaluate((4.0, 0.0, 0.0)) == pytest.approx(0.0)
    # Point on inner equator: (2, 0, 0)
    assert s.evaluate((2.0, 0.0, 0.0)) == pytest.approx(0.0)

def test_ztorus_evaluate_signs():
    s = Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0)
    # Inside tube: (3, 0, 0) is on the tube center axis
    assert s.evaluate((3.0, 0.0, 0.0)) < 0.0
    # Outside: origin
    assert s.evaluate((0.0, 0.0, 0.0)) > 0.0
    # Outside: far away
    assert s.evaluate((10.0, 0.0, 0.0)) > 0.0

def test_boundary_default_all_surfaces():
    """Test that all surface types default to None boundary type."""
    surfaces = [
        Plane(axis="x", offset=1.0),
        Plane(axis="y", offset=2.0),
        Plane(axis="z", offset=3.0),
        Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0),
        Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0),
        Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0),
        Cylinder(axis="y", x0=0.0, z0=0.0, radius=1.0),
        Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0),
        Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0),
        Plane(axis="x", offset=0.0),
    ]
    for s in surfaces:
        assert s.boundary is None, f"Expected None boundary for {type(s)}"

def test_boundary_vacuum():
    """Test that vacuum boundary type can be explicitly set."""
    s = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1, boundary="vacuum")
    assert s.boundary == "vacuum"

def test_boundary_explicit_none():
    """Test that None boundary type can be explicitly set."""
    s = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1, boundary=None)
    assert s.boundary is None

def test_set_boundary():
    """Test that boundary type can be changed after creation."""
    s = Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0, surface_id=2)
    assert s.boundary is None

    s.boundary = "vacuum"
    assert s.boundary == "vacuum"

    s.boundary = None
    assert s.boundary is None

def test_invalid_boundary():
    with pytest.raises(ValueError):
        Plane(axis="x", offset=1.0, surface_id=42, boundary="invalid")

def test_invalid_set_boundary():
    s = Plane(axis="x", offset=1.0, surface_id=42)
    with pytest.raises(ValueError):
        s.boundary = "invalid"


def test_surface_optional_id():
    """Test that surface IDs are optional and default to None."""
    # Create surfaces without specifying surface_id
    sphere = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    plane = Plane(axis="x", offset=0.0)
    cylinder = Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0)

    # Surface IDs should default to None
    assert sphere.surface_id is None
    assert plane.surface_id is None
    assert cylinder.surface_id is None


def test_surface_explicit_id():
    """Test that surface IDs can be explicitly set."""
    # Create surfaces with specific IDs
    sphere = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1)
    plane = Plane(axis="x", offset=0.0, surface_id=2)
    cylinder = Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0, surface_id=3)
    
    # Surface IDs should match what we set
    assert sphere.surface_id == 1
    assert plane.surface_id == 2
    assert cylinder.surface_id == 3


def test_surface_id_setter():
    """Test that surface IDs can be changed after creation."""
    sphere = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    
    # Initially None
    assert sphere.surface_id is None
    
    # Set an ID
    sphere.surface_id = 42
    assert sphere.surface_id == 42
    
    # Change the ID
    sphere.surface_id = 100
    assert sphere.surface_id == 100


def test_geometry_duplicate_surface_ids():
    """Test that geometry validation detects duplicate surface IDs."""
    # Create surfaces with duplicate IDs
    surface1 = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1)
    surface2 = Plane(axis="x", offset=1.0, surface_id=1)  # Same ID!
    
    # Create regions using these surfaces
    region1 = surface1.below  # Inside sphere
    region2 = surface2.above  # Above plane
    
    # Create materials and cells
    material1 = Material(
        composition={"H1": 1.0},
        density=1.0)
    material2 = Material(
        composition={"He4": 1.0},
        density=1.0)

    cell1 = Cell(name="cell1", region=region1, material=material1, id=1)
    cell2 = Cell(name="cell2", region=region2, material=material2, id=2)

    # Geometry validation should fail due to duplicate surface IDs
    with pytest.raises(ValueError, match="Duplicate surface_id 1 found"):
        Geometry(cells=[cell1, cell2])


def test_geometry_mixed_surface_ids():
    """Test geometry with mix of surfaces with and without IDs."""
    # Create surfaces - some with IDs, some without
    surface1 = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1)
    surface2 = Plane(axis="x", offset=1.0)  # No ID
    surface3 = Cylinder(axis="z", x0=0.0, y0=0.0, radius=0.5, surface_id=3)
    
    # Create regions
    region1 = surface1.below & surface2.above  # Inside sphere and above plane
    region2 = surface3.below & surface2.below  # Inside cylinder and below plane
    
    # Create materials and cells
    material1 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=10)
    material2 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=20)

    cell1 = Cell(name="cell1", region=region1, material=material1, id=10)
    cell2 = Cell(name="cell2", region=region2, material=material2, id=20)

    # This should work fine - no duplicate surface IDs
    geometry = Geometry(cells=[cell1, cell2])
    assert geometry is not None


def test_geometry_no_surface_ids():
    """Test geometry where no surfaces have IDs."""
    # Create surfaces without IDs
    surface1 = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    surface2 = Plane(axis="x", offset=1.0)

    # Create regions
    region1 = surface1.below
    region2 = surface2.above
    
    # Create materials and cells
    material1 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=1)
    material2 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=2)

    cell1 = Cell(name="cell1", region=region1, material=material1, id=1)
    cell2 = Cell(name="cell2", region=region2, material=material2, id=2)

    # This should work fine - no surface ID conflicts
    geometry = Geometry(cells=[cell1, cell2])
    assert geometry is not None


def test_geometry_unique_surface_ids():
    """Test geometry with all surfaces having unique IDs."""
    # Create surfaces with unique IDs
    surface1 = Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=10)
    surface2 = Plane(axis="x", offset=1.0, surface_id=20)
    surface3 = Cylinder(axis="z", x0=0.0, y0=0.0, radius=0.5, surface_id=30)
    
    # Create regions
    region1 = surface1.below & surface2.above
    region2 = surface3.below & surface2.below
    
    # Create materials and cells
    material1 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=1)
    material2 = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=2)

    cell1 = Cell(name="cell1", region=region1, material=material1, id=1)
    cell2 = Cell(name="cell2", region=region2, material=material2, id=2)

    # This should work fine - all unique surface IDs
    geometry = Geometry(cells=[cell1, cell2])
    assert geometry is not None


def test_surface_types_with_ids():
    """Test that all surface types support optional IDs."""
    surfaces = [
        Plane(axis="x", offset=1.0, surface_id=1),
        Plane(axis="y", offset=2.0, surface_id=2),
        Plane(axis="z", offset=3.0, surface_id=3),
        Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=4),
        Cylinder(axis="z", x0=0.0, y0=0.0, radius=0.5, surface_id=5),
        Plane(axis=(1.0, 1.0, 1.0), offset=0.0 / (1.0**2 + 1.0**2 + 1.0**2) ** 0.5, surface_id=6),
        Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0, surface_id=7),
    ]

    # Check that all surfaces have the expected IDs
    for i, surface in enumerate(surfaces, 1):
        assert surface.surface_id == i


def test_surface_types_without_ids():
    """Test that all surface types work without IDs."""
    surfaces = [
        Plane(axis="x", offset=1.0),
        Plane(axis="y", offset=2.0),
        Plane(axis="z", offset=3.0),
        Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0),
        Cylinder(axis="z", x0=0.0, y0=0.0, radius=0.5),
        Plane(axis=(1.0, 1.0, 1.0), offset=0.0 / (1.0**2 + 1.0**2 + 1.0**2) ** 0.5),
        Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0),
    ]

    # Check that all surfaces have None IDs
    for surface in surfaces:
        assert surface.surface_id is None


def test_complex_geometry_surface_validation():
    """Test surface ID validation in a more complex geometry."""
    # Create multiple surfaces with some duplicate IDs
    surfaces = [
        Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0, surface_id=1),
        Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0, surface_id=2),
        Plane(axis="z", offset=1.0, surface_id=3),
        Plane(axis="z", offset=-1.0, surface_id=3),  # Duplicate ID!
        Plane(axis="x", offset=0.0),  # No ID
    ]
    
    # Create a complex region using all surfaces
    region = (surfaces[0].below & surfaces[1].below & surfaces[2].above & surfaces[3].below) | surfaces[4].above
    
    # Create material and cell
    material = Material(
        composition={"H1": 1.0},
        density=1.0,
        id=1)
    cell = Cell(name="test_cell", region=region, material=material, id=1)

    # Should fail due to duplicate surface ID 3
    with pytest.raises(ValueError, match="Duplicate surface_id 3 found"):
        Geometry(cells=[cell])

