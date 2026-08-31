"""
Test surface ID validation functionality.
Tests optional surface IDs and duplicate surface ID detection in geometry validation.
"""

import pytest
import yamc


def test_surface_optional_id():
    """Test that surface IDs are optional and default to None."""
    # Create surface without specifying ID
    surface = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    assert surface.surface_id is None
    
    # Create surface with explicit None
    surface2 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=None)
    assert surface2.surface_id is None


def test_surface_explicit_id():
    """Test that surface IDs can be set explicitly."""
    surface = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=42)
    assert surface.surface_id == 42
    
    # Test with different surface types
    plane = yamc.Plane(axis="x", offset=1.0, surface_id=100)
    assert plane.surface_id == 100


def test_surface_id_setter():
    """Test that surface IDs can be modified after creation."""
    surface = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    assert surface.surface_id is None
    
    # Set ID
    surface.surface_id = 123
    assert surface.surface_id == 123
    
    # Set back to None
    surface.surface_id = None
    assert surface.surface_id is None


def test_geometry_duplicate_surface_ids():
    """Test that duplicate surface IDs are detected in geometry validation."""
    # Create surfaces with same ID
    surface1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=10)
    surface2 = yamc.Sphere(x0=2.0, y0=0.0, z0=0.0, radius=1.0, surface_id=10)  # Same ID
    
    # Use different materials to avoid duplicate material ID error
    material1 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=1)
    material2 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=2)
    
    # Create cells using these surfaces
    cell1 = yamc.Cell(name='cell1', region=surface1.below, material=material1, id=1)
    cell2 = yamc.Cell(name='cell2', region=surface2.below, material=material2, id=2)

    # Should raise error due to duplicate surface IDs
    with pytest.raises(ValueError, match="Duplicate surface_id 10 found"):
        yamc.Geometry(cells=[cell1, cell2])


def test_geometry_mixed_surface_ids():
    """Test geometry with mix of surfaces with and without IDs."""
    # Create surfaces - mix of with/without IDs
    surface1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=10)  # With ID
    surface2 = yamc.Sphere(x0=2.0, y0=0.0, z0=0.0, radius=1.0)  # Without ID
    surface3 = yamc.Sphere(x0=4.0, y0=0.0, z0=0.0, radius=1.0, surface_id=20)  # With ID
    
    # Use different materials to avoid duplicate material ID error
    material1 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=1)
    material2 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=2)
    material3 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=3)
    
    # Create cells
    cell1 = yamc.Cell(name='cell1', region=surface1.below, material=material1, id=1)
    cell2 = yamc.Cell(name='cell2', region=surface2.below, material=material2, id=2)
    cell3 = yamc.Cell(name='cell3', region=surface3.below, material=material3, id=3)

    # Should work fine - no duplicates
    geometry = yamc.Geometry(cells=[cell1, cell2, cell3])
    assert len(geometry.cells) == 3


def test_geometry_no_surface_ids():
    """Test geometry with all surfaces having None IDs."""
    # Create surfaces without IDs
    surface1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    surface2 = yamc.Sphere(x0=2.0, y0=0.0, z0=0.0, radius=1.0)
    
    # Use different materials to avoid duplicate material ID error
    material1 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=1)
    material2 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=2)
    
    # Create cells
    cell1 = yamc.Cell(name='cell1', region=surface1.below, material=material1, id=1)
    cell2 = yamc.Cell(name='cell2', region=surface2.below, material=material2, id=2)

    # Should work fine - None IDs are allowed
    geometry = yamc.Geometry(cells=[cell1, cell2])
    assert len(geometry.cells) == 2


def test_geometry_unique_surface_ids():
    """Test geometry with all unique surface IDs."""
    # Create surfaces with unique IDs
    surface1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=1)
    surface2 = yamc.Sphere(x0=2.0, y0=0.0, z0=0.0, radius=1.0, surface_id=2)
    surface3 = yamc.Sphere(x0=4.0, y0=0.0, z0=0.0, radius=1.0, surface_id=3)
    
    # Use different materials to avoid duplicate material ID error
    material1 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=100)
    material2 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=200)
    material3 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=300)
    
    # Create cells
    cell1 = yamc.Cell(name='cell1', region=surface1.below, material=material1, id=1)
    cell2 = yamc.Cell(name='cell2', region=surface2.below, material=material2, id=2)
    cell3 = yamc.Cell(name='cell3', region=surface3.below, material=material3, id=3)

    # Should work fine
    geometry = yamc.Geometry(cells=[cell1, cell2, cell3])
    assert len(geometry.cells) == 3


def test_surface_types_with_ids():
    """Test that all surface types support surface_id parameter."""
    # Test various surface types with IDs
    sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=1)
    plane = yamc.Plane(axis="x", offset=1, surface_id=2)
    xplane = yamc.Plane(axis="x", offset=1, surface_id=3)
    yplane = yamc.Plane(axis="y", offset=2, surface_id=4)
    zplane = yamc.Plane(axis="z", offset=3, surface_id=5)
    zcylinder = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1, surface_id=6)
    cylinder = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1, surface_id=7)
    
    assert sphere.surface_id == 1
    assert plane.surface_id == 2
    assert xplane.surface_id == 3
    assert yplane.surface_id == 4
    assert zplane.surface_id == 5
    assert zcylinder.surface_id == 6
    assert cylinder.surface_id == 7


def test_surface_types_without_ids():
    """Test that all surface types default to None for surface_id."""
    # Test various surface types without IDs
    sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=1)
    plane = yamc.Plane(axis="x", offset=1)
    xplane = yamc.Plane(axis="x", offset=1)
    yplane = yamc.Plane(axis="y", offset=2)
    zplane = yamc.Plane(axis="z", offset=3)
    zcylinder = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1)
    cylinder = yamc.Cylinder(axis="z", x0=0, y0=0, radius=1)
    
    assert sphere.surface_id is None
    assert plane.surface_id is None
    assert xplane.surface_id is None
    assert yplane.surface_id is None
    assert zplane.surface_id is None
    assert zcylinder.surface_id is None
    assert cylinder.surface_id is None


def test_complex_geometry_surface_validation():
    """Test surface validation in a complex geometry with shared surfaces."""
    # Create surfaces with unique IDs
    surface1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, surface_id=10)
    surface2 = yamc.Plane(axis="x", offset=1.0, surface_id=20)
    surface3 = yamc.Cylinder(axis="z", x0=0.0, y0=0.0, radius=0.5, surface_id=30)
    
    material1 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=1)
    material2 = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=2)
    
    # Create regions that share surfaces (this tests Arc deduplication)
    region1 = surface1.below & surface2.above   # Uses surface1(10) and surface2(20)
    region2 = surface3.below & surface2.below   # Uses surface3(30) and surface2(20) - SHARED!
    
    cell1 = yamc.Cell(name='cell1', region=region1, material=material1, id=1)
    cell2 = yamc.Cell(name='cell2', region=region2, material=material2, id=2)
    
    # Should work fine - shared surfaces are correctly deduplicated
    geometry = yamc.Geometry(cells=[cell1, cell2])
    assert len(geometry.cells) == 2


if __name__ == "__main__":
    pytest.main([__file__])
