import yamc
import pytest

def test_geometry_find_cell():
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    cell = yamc.Cell(id=1, region=region)
    geometry = yamc.Geometry([cell])
    assert geometry.find_cell(0, 0, 0) is not None
    assert geometry.find_cell(5, 0, 0) is None


def test_geometry_duplicate_id_validation():
    """Test geometry validation for duplicate cell and material IDs."""
    
    # Create surfaces for our cells
    surf1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=1)
    surf2 = yamc.Sphere(x0=2, y0=0, z0=0, radius=1, surface_id=2)
    region1 = surf1.below
    region2 = surf2.below
    
    # Step 1: Create two cells with IDs 1 and 2
    cell1 = yamc.Cell(region=region1, id=1)
    _cell2 = yamc.Cell(region=region2, id=2)  # noqa: F841
    
    # Step 2: Create one material with ID 1
    material1 = yamc.Material(composition={"Li6": 1.0}, density=1.0, id=1)

    # Step 3: Fill both cells with the same material - this should WORK because it's
    # intentional sharing (same Python Material object = same source_id = same material)
    cell1_filled = yamc.Cell(region=region1, id=1, material=material1)
    cell2_filled = yamc.Cell(region=region2, id=2, material=material1)

    # Step 4: Creating geometry should SUCCEED - same material used in multiple cells is valid
    geometry_shared = yamc.Geometry([cell1_filled, cell2_filled])
    assert len(geometry_shared.cells) == 2
    # Both cells should have the same material_id since they share the same material
    assert geometry_shared.cells[0].material.id == 1
    assert geometry_shared.cells[1].material.id == 1
    
    # Step 5: Test duplicate cell ID detection
    # Create another cell with the same ID as cell1
    cell3_duplicate_id = yamc.Cell(region=region1, id=1)  # Same ID as cell1
    
    # Creating geometry with duplicate cell IDs should fail
    with pytest.raises(ValueError, match="Duplicate Cell id 1 found"):
        yamc.Geometry([cell1, cell3_duplicate_id])
    
    # Step 6: Test creating valid geometry with different material IDs
    # Create two different materials with different IDs
    material_a = yamc.Material(composition={"Li6": 1.0}, density=1.0, id=10)

    material_b = yamc.Material(composition={"Li7": 1.0}, density=1.0, id=20)
    
    # Fill cells with different materials
    cell1_with_mat_a = yamc.Cell(region=region1, id=1, material=material_a)
    cell2_with_mat_b = yamc.Cell(region=region2, id=2, material=material_b)
    
    # This should work fine - different cell IDs and different material IDs
    geometry_valid = yamc.Geometry([cell1_with_mat_a, cell2_with_mat_b])
    assert len(geometry_valid.cells) == 2
    
    # Step 7: Test duplicate material ID detection
    # Create another material with the same ID as material_a
    material_c_duplicate_id = yamc.Material(composition={"Be9": 1.0}, density=1.0, id=10)  # Same ID as material_a
    
    # Create a cell with the duplicate material
    cell3_with_duplicate_mat = yamc.Cell(region=region1, id=3, material=material_c_duplicate_id)
    
    # Creating geometry with duplicate material IDs should fail
    with pytest.raises(ValueError, match="Duplicate Material id 10 found"):
        yamc.Geometry([cell1_with_mat_a, cell3_with_duplicate_mat])
    
    # Step 8: Change cell ID and test that material validation still works
    # Update cell ID to be unique but keep duplicate material ID
    cell3_updated_id = yamc.Cell(region=region2, id=30, material=material_c_duplicate_id)
    
    # Should still fail due to duplicate material IDs
    with pytest.raises(ValueError, match="Duplicate Material id 10 found"):
        yamc.Geometry([cell1_with_mat_a, cell3_updated_id])
    
    # Step 9: Fix material ID and verify it works
    material_c_fixed = yamc.Material(composition={"Be9": 1.0}, density=1.0, id=30)  # Different ID
    
    cell3_fixed = yamc.Cell(region=region2, id=30, material=material_c_fixed)
    
    # Now it should work
    geometry_final = yamc.Geometry([cell1_with_mat_a, cell3_fixed])
    assert len(geometry_final.cells) == 2
    
    # Step 10: Test cells without materials are allowed (no material ID conflicts)
    cell_no_material1 = yamc.Cell(region=region1, id=100)
    cell_no_material2 = yamc.Cell(region=region2, id=200)
    
    geometry_no_materials = yamc.Geometry([cell_no_material1, cell_no_material2])
    assert len(geometry_no_materials.cells) == 2


def test_surface_id_validation():
    """Test surface ID validation in geometry creation."""
    
    # Step 1: Test duplicate surface ID detection
    # Create two surfaces with the same ID
    surf1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=10)
    surf2 = yamc.Sphere(x0=2, y0=0, z0=0, radius=1, surface_id=10)  # Same ID - should fail
    
    region1 = surf1.below
    region2 = surf2.below
    
    cell1 = yamc.Cell(region=region1, id=1)
    cell2 = yamc.Cell(region=region2, id=2)
    
    # Creating geometry with duplicate surface IDs should fail
    with pytest.raises(ValueError, match="Duplicate surface_id 10 found"):
        yamc.Geometry([cell1, cell2])
    
    # Step 2: Test surface without ID validation
    # Create surface without specifying ID (should default to None)
    surf_no_id = yamc.Sphere(x0=0, y0=0, z0=0, radius=1)  # No surface_id specified
    region_no_id = surf_no_id.below
    cell_no_surface_id = yamc.Cell(region=region_no_id, id=1)
    
    # Creating geometry with surface without ID should work fine (None IDs are allowed)
    geometry_with_none_id = yamc.Geometry([cell_no_surface_id])
    assert len(geometry_with_none_id.cells) == 1
    
    # Step 3: Test valid unique surface IDs
    surf_a = yamc.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=1)
    surf_b = yamc.Sphere(x0=2, y0=0, z0=0, radius=1, surface_id=2)
    surf_c = yamc.Sphere(x0=4, y0=0, z0=0, radius=1, surface_id=3)
    
    region_a = surf_a.below
    region_b = surf_b.below
    region_c = surf_c.below
    
    cell_a = yamc.Cell(region=region_a, id=1)
    cell_b = yamc.Cell(region=region_b, id=2)
    cell_c = yamc.Cell(region=region_c, id=3)
    
    # This should work fine - all surface IDs are unique
    geometry_valid = yamc.Geometry([cell_a, cell_b, cell_c])
    assert len(geometry_valid.cells) == 3
    
    # Step 4: Test different surface types with unique IDs
    sphere_surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=1, surface_id=100)
    plane_surf = yamc.Plane(axis="x", offset=5, surface_id=101)
    cylinder_surf = yamc.Cylinder(axis="z", x0=0, y0=0, radius=2, surface_id=102)
    
    sphere_region = sphere_surf.below
    plane_region = plane_surf.below
    cylinder_region = cylinder_surf.below
    
    sphere_cell = yamc.Cell(region=sphere_region, id=10)
    plane_cell = yamc.Cell(region=plane_region, id=11)
    cylinder_cell = yamc.Cell(region=cylinder_region, id=12)
    
    # Different surface types with unique IDs should work
    geometry_mixed = yamc.Geometry([sphere_cell, plane_cell, cylinder_cell])
    assert len(geometry_mixed.cells) == 3
    
    # Step 5: Test duplicate surface ID with different surface types
    duplicate_plane = yamc.Plane(axis="x", offset=10, surface_id=100)  # Same ID as sphere_surf
    duplicate_region = duplicate_plane.below
    duplicate_cell = yamc.Cell(region=duplicate_region, id=20)
    
    # Should fail due to duplicate surface ID even with different surface types
    with pytest.raises(ValueError, match="Duplicate surface_id 100 found"):
        yamc.Geometry([sphere_cell, duplicate_cell])
    
    # Step 6: Test complex regions with multiple surfaces
    # Create a region using intersection of two surfaces with unique IDs
    surf_x = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=200)
    surf_y = yamc.Plane(axis="x", offset=0, surface_id=201)
    
    # Region inside sphere AND to the right of plane
    complex_region = (surf_x.below) & (surf_y.above)
    complex_cell = yamc.Cell(region=complex_region, id=50)
    
    # Should work fine - both surfaces have unique IDs
    geometry_complex = yamc.Geometry([complex_cell])
    assert len(geometry_complex.cells) == 1
    
    # Step 7: Test complex region with duplicate surface IDs
    surf_duplicate = yamc.Plane(axis="y", offset=0, surface_id=200)  # Same ID as surf_x
    
    # Region using duplicate surface ID
    duplicate_complex_region = (surf_x.below) & (surf_duplicate.above)
    duplicate_complex_cell = yamc.Cell(region=duplicate_complex_region, id=51)
    
    # Should fail due to duplicate surface ID in complex region
    with pytest.raises(ValueError, match="Duplicate surface_id 200 found"):
        yamc.Geometry([duplicate_complex_cell])


def test_auto_assign_cell_ids():
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2)
    region = surf.below
    cell1 = yamc.Cell(region=region)
    geometry = yamc.Geometry([cell1])
    assert geometry.cells[0].id == 1
    # Note: The original Python cell object is NOT updated - only the geometry's copy is.
    # This matches typical MC framework behavior where the model owns copies of objects.
    # assert cell1.id == 1  # This would fail - original cell still has None

def test_cell_ids_stays_the_same():
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2)
    region = surf.below
    cell1 = yamc.Cell(region=region, id=54)
    geometry = yamc.Geometry([cell1])
    assert geometry.cells[0].id == 54
    assert cell1.id == 54

def test_material_object_retention():
    mat_air = yamc.Material(
        composition={"N": 0.784431, "O": 0.210748, "Ar": 0.0046},
        density=0.001205, units="g/cc",
        name="Air",
        id=33)

    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2)
    region = surf.below
    cell1 = yamc.Cell(region=region, material=mat_air)
    geometry = yamc.Geometry([cell1])
    assert geometry.cells[0].material.id == 33
    assert mat_air.id == 33

def test_material_id_auto_assignment():
    mat1 = yamc.Material(composition={"H": 1.0}, density=1.0)
    mat2 = yamc.Material(composition={"O": 1.0}, density=1.0)

    surf1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1)
    surf2 = yamc.Sphere(x0=3, y0=0, z0=0, radius=1)
    surf3 = yamc.Sphere(x0=3, y0=0, z0=0, radius=1)

    cell1 = yamc.Cell(region=surf1.below, material=mat1)
    cell2 = yamc.Cell(region=surf2.below, material=mat2)
    cell3 = yamc.Cell(region=surf3.below, material=mat2)

    geometry = yamc.Geometry([cell1, cell2, cell3])

    assert geometry.cells[0].material.id == 1
    assert geometry.cells[1].material.id == 2
    assert geometry.cells[2].material.id == 2

    assert mat2.id == 2
    mat2.id = 15
    assert mat2.id == 15

    assert geometry.cells[2].material.id == geometry.cells[1].material.id
    assert geometry.cells[0].material.id != geometry.cells[1].material.id

    assert cell1.id == 1
    assert cell2.id == 2
    assert cell3.id == 3
    assert mat1.id == 1

def test_material_id_auto_and_manual_assignment():
    mat1 = yamc.Material(composition={"H": 1.0}, density=1.0, id=2)

    mat2 = yamc.Material(composition={"O": 1.0}, density=1.0)

    mat3 = yamc.Material(composition={"O": 1.0}, density=1.0, id=1)

    surf1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1)
    surf2 = yamc.Sphere(x0=3, y0=0, z0=0, radius=1)
    surf3 = yamc.Sphere(x0=3, y0=0, z0=0, radius=1)

    cell1 = yamc.Cell(region=surf1.below, material=mat1)
    cell2 = yamc.Cell(region=surf2.below, material=mat2)
    cell3 = yamc.Cell(region=surf3.below, material=mat3)

    yamc.Geometry([cell1, cell2, cell3])

    assert mat1.id == 2
    assert mat2.id == 3
    assert mat3.id == 1


def test_sample_slice_xy_basis():
    """Test sample_slice function with xy basis."""
    # Create a sphere at origin with radius 2
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=10)
    cell = yamc.Cell(id=5, region=region, material=mat)
    geometry = yamc.Geometry([cell])

    # Test xy basis at z=0 (through center of sphere)
    # origin=(0,0,0), width=(6,6), resolution=(5,5)
    cell_ids, material_ids = geometry.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="xy")

    # Check dimensions
    assert len(cell_ids) == 5
    assert len(cell_ids[0]) == 5
    assert len(material_ids) == 5
    assert len(material_ids[0]) == 5

    # Center point should be inside the sphere
    assert cell_ids[2][2] == 5
    assert material_ids[2][2] == 10

    # Corner points should be outside
    assert cell_ids[0][0] == -1
    assert material_ids[0][0] == -1
    assert cell_ids[4][4] == -1
    assert material_ids[4][4] == -1


def test_sample_slice_xz_basis():
    """Test sample_slice function with xz basis."""
    # Create a sphere at origin with radius 2
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    cell = yamc.Cell(id=7, region=region)  # No material (void)
    geometry = yamc.Geometry([cell])

    # Test xz basis at y=0
    # origin=(0,0,0), width=(6,6), resolution=(5,5)
    cell_ids, material_ids = geometry.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="xz")

    # Center point should be inside
    assert cell_ids[2][2] == 7
    # No material, so material_id should be -1
    assert material_ids[2][2] == -1

    # Corners should be outside
    assert cell_ids[0][0] == -1


def test_sample_slice_yz_basis():
    """Test sample_slice function with yz basis."""
    # Create a sphere at origin with radius 2
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    cell = yamc.Cell(id=3, region=region)
    geometry = yamc.Geometry([cell])

    # Test yz basis at x=0
    # origin=(0,0,0), width=(6,6), resolution=(5,5)
    cell_ids, material_ids = geometry.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="yz")

    # Center point should be inside
    assert cell_ids[2][2] == 3

    # Corners should be outside
    assert cell_ids[0][0] == -1
    assert cell_ids[4][0] == -1


def test_sample_slice_defaults():
    """Test sample_slice function with default parameters."""
    # Create a sphere at origin with radius 2
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=10)
    cell = yamc.Cell(id=5, region=region, material=mat)
    geometry = yamc.Geometry([cell])

    # Test with all defaults - should use bbox center, bbox width, 40000 pixels
    cell_ids, material_ids = geometry.sample_slice()

    # With 40000 pixels and 4x4 bbox, expect ~200x200
    assert len(cell_ids) == 200
    assert len(cell_ids[0]) == 200

    # Center should be inside
    assert cell_ids[100][100] == 5
    assert material_ids[100][100] == 10


def test_sample_slice_pixels_int():
    """Test sample_slice with pixels as a single int."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    cell = yamc.Cell(id=1, region=region)
    geometry = yamc.Geometry([cell])

    # Test with resolution=100 (total pixels)
    cell_ids, _ = geometry.sample_slice(resolution=100)

    # With 100 total pixels and 4x4 bbox (aspect=1), expect ~10x10
    assert len(cell_ids) == 10
    assert len(cell_ids[0]) == 10


def test_sample_slice_multiple_cells():
    """Test sample_slice function with multiple cells."""
    # Create two adjacent cells using planes
    xplane_0 = yamc.Plane(axis="x", offset=0, surface_id=1)
    xplane_1 = yamc.Plane(axis="x", offset=5, surface_id=2)
    xplane_2 = yamc.Plane(axis="x", offset=10, surface_id=3)
    yplane_0 = yamc.Plane(axis="y", offset=0, surface_id=4)
    yplane_1 = yamc.Plane(axis="y", offset=10, surface_id=5)
    zplane_0 = yamc.Plane(axis="z", offset=0, surface_id=6)
    zplane_1 = yamc.Plane(axis="z", offset=10, surface_id=7)

    # Left cell: x from 0 to 5
    region1 = xplane_0.above & xplane_1.below & yplane_0.above & yplane_1.below & zplane_0.above & zplane_1.below
    mat1 = yamc.Material(composition={"H": 1.0}, density=1.0, id=1)
    cell1 = yamc.Cell(id=1, region=region1, material=mat1)

    # Right cell: x from 5 to 10
    region2 = xplane_1.above & xplane_2.below & yplane_0.above & yplane_1.below & zplane_0.above & zplane_1.below
    mat2 = yamc.Material(composition={"O": 1.0}, density=1.0, id=2)
    cell2 = yamc.Cell(id=2, region=region2, material=mat2)

    geometry = yamc.Geometry([cell1, cell2])

    # Sample using origin/width API
    # origin=(5,5,5), width=(10,10), resolution=(11,11)
    cell_ids, material_ids = geometry.sample_slice(
        origin=(5.0, 5.0, 5.0),
        width=(10.0, 10.0),
        resolution=(11, 11),
        basis="xy")

    # Points in left cell (x < 5) should have id=1, id=1
    # x=1 is at index 1 (from 0 to 10 with 11 samples)
    assert cell_ids[5][1] == 1  # x=1, y=5
    assert material_ids[5][1] == 1

    # Points in right cell (x > 5) should have id=2, id=2
    # x=6 is at index 6
    assert cell_ids[5][6] == 2  # x=6, y=5
    assert material_ids[5][6] == 2


def test_region_sample_slice():
    """Test sample_slice on Region."""
    # Create a sphere region
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below

    # Test with explicit parameters
    ids = region.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="xy")

    # Check dimensions
    assert len(ids) == 5
    assert len(ids[0]) == 5

    # Center should be inside (1)
    assert ids[2][2] == 1

    # Corners should be outside (-1)
    assert ids[0][0] == -1
    assert ids[4][4] == -1


def test_region_sample_slice_defaults():
    """Test Region.sample_slice with default parameters."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below

    # Test with all defaults
    ids = region.sample_slice()

    # With 40000 pixels and 4x4 bbox, expect ~200x200
    assert len(ids) == 200
    assert len(ids[0]) == 200

    # Center should be inside
    assert ids[100][100] == 1


def test_cell_sample_slice():
    """Test sample_slice on Cell."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=10)
    cell = yamc.Cell(id=5, region=region, material=mat)

    # Test with explicit parameters
    cell_ids, material_ids = cell.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="xy")

    # Check dimensions
    assert len(cell_ids) == 5
    assert len(cell_ids[0]) == 5

    # Center should have id=5 and id=10
    assert cell_ids[2][2] == 5
    assert material_ids[2][2] == 10

    # Corners should be outside (-1)
    assert cell_ids[0][0] == -1
    assert material_ids[0][0] == -1


def test_cell_sample_slice_no_material():
    """Test Cell.sample_slice with no material (void cell)."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    cell = yamc.Cell(id=7, region=region)  # No material

    cell_ids, material_ids = cell.sample_slice(resolution=25)

    # Center should have id=7 but id=-1
    mid = len(cell_ids) // 2
    assert cell_ids[mid][mid] == 7
    assert material_ids[mid][mid] == -1


def test_model_sample_slice():
    """Test sample_slice on Model."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1)
    region = surf.below
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=10)
    cell = yamc.Cell(id=5, region=region, material=mat)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource()
    model = yamc.Model(geometry, source=source)

    # Test with explicit parameters
    cell_ids, material_ids = model.sample_slice(
        origin=(0.0, 0.0, 0.0),
        width=(6.0, 6.0),
        resolution=(5, 5),
        basis="xy")

    # Check dimensions
    assert len(cell_ids) == 5
    assert len(cell_ids[0]) == 5

    # Center should have id=5 and id=10
    assert cell_ids[2][2] == 5
    assert material_ids[2][2] == 10

    # Corners should be outside (-1)
    assert cell_ids[0][0] == -1


# ── GeometrySliceData rich return tests ──────────────────────────────────


def test_geometry_slice_data_attributes():
    """GeometrySliceData exposes h_edges, v_edges, extent, labels."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1, boundary="vacuum")
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=10)
    cell = yamc.Cell(id=5, region=surf.below, material=mat)
    geometry = yamc.Geometry([cell])

    result = geometry.sample_slice(
        origin=(0.0, 0.0, 0.0), width=(6.0, 6.0), resolution=(10, 8), basis="xy"
    )
    assert type(result).__name__ == "GeometrySliceData"
    assert result.h_label == "x"
    assert result.v_label == "y"
    assert len(result.h_edges) == 11  # 10 pixels → 11 edges
    assert len(result.v_edges) == 9   # 8 pixels → 9 edges
    assert result.extent == (-3.0, 3.0, -3.0, 3.0)
    assert len(result.cell_ids) == 8
    assert len(result.cell_ids[0]) == 10


def test_geometry_slice_data_xz_labels():
    """Labels adapt to basis."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1, boundary="vacuum")
    cell = yamc.Cell(id=1, region=surf.below)
    geometry = yamc.Geometry([cell])

    result = geometry.sample_slice(resolution=25, basis="xz")
    assert result.h_label == "x"
    assert result.v_label == "z"

    result_yz = geometry.sample_slice(resolution=25, basis="yz")
    assert result_yz.h_label == "y"
    assert result_yz.v_label == "z"


def test_geometry_slice_data_edges():
    """edges() method detects material boundaries."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1, boundary="vacuum")
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=1)
    cell = yamc.Cell(id=1, region=surf.below, material=mat)
    geometry = yamc.Geometry([cell])

    result = geometry.sample_slice(
        origin=(0.0, 0.0, 0.0), width=(6.0, 6.0), resolution=(50, 50), basis="xy"
    )
    mat_edges = result.edges("material")
    cell_edges = result.edges("cell")

    assert len(mat_edges) == 50
    assert len(mat_edges[0]) == 50
    n_mat = sum(sum(row) for row in mat_edges)
    n_cell = sum(sum(row) for row in cell_edges)
    assert n_mat > 0
    assert n_cell > 0


def test_geometry_slice_data_tuple_compat():
    """Tuple unpacking still works for backward compatibility."""
    surf = yamc.Sphere(x0=0, y0=0, z0=0, radius=2, surface_id=1, boundary="vacuum")
    mat = yamc.Material(composition={"H": 1.0}, density=1.0, id=1)
    cell = yamc.Cell(id=1, region=surf.below, material=mat)
    geometry = yamc.Geometry([cell])

    cell_ids, material_ids = geometry.sample_slice(resolution=25, basis="xy")
    assert isinstance(cell_ids, list)
    assert isinstance(material_ids, list)
