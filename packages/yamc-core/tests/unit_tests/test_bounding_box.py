import yamc

def test_sphere_bb_moved_on_z_axis():
    s2 = yamc.Sphere(x0=0, y0=0, z0=1, radius=3)
    region2 = s2.below
    bb = region2.bounding_box()
    assert bb.lower_left == [-3.0, -3.0, -2.0]
    assert bb.upper_right == [3.0, 3.0, 4.0]

def test_sphere_with_xplanes():
    s1 = yamc.Plane(axis="x", offset=2.1)
    s2 = yamc.Plane(axis="x", offset=-2.1)
    s3 = yamc.Sphere(x0=0, y0=0, z0=0, radius=4.2)

    region1 = s1.below & s2.above & s3.below
    assert region1.contains((0, 0, 0))
    bb = region1.bounding_box()
    assert bb.lower_left == [-2.1, -4.2, -4.2]
    assert bb.upper_right == [2.1, 4.2, 4.2]

def test_zcylinder_with_zplanes():
    # Create a Z-cylinder centered at (1, 2) with radius 3
    cyl = yamc.Cylinder(axis="z", x0=1.0, y0=2.0, radius=3.0)
    # Create Z planes to bound the cylinder in Z direction
    z_bottom = yamc.Plane(axis="z", offset=-5.0)
    z_top = yamc.Plane(axis="z", offset=5.0)
    
    # Region inside cylinder and between the Z planes
    region = cyl.below & z_bottom.above & z_top.below
    
    # Test that points are contained as expected
    assert region.contains((1.0, 2.0, 0.0))  # Center of cylinder
    assert region.contains((3.0, 2.0, 0.0))  # On cylinder surface in +X
    assert region.contains((1.0, 4.0, 0.0))  # On cylinder surface in +Y
    assert not region.contains((5.0, 2.0, 0.0))  # Outside cylinder
    assert not region.contains((1.0, 2.0, 6.0))  # Above Z plane
    
    # Get bounding box - should be bounded by planes in X, Y and Z planes in Z
    bb = region.bounding_box()
    # X bounds: cylinder center (1) ± radius (3) = [-2, 4]
    # Y bounds: cylinder center (2) ± radius (3) = [-1, 5] 
    # Z bounds: between the Z planes = [-5, 5]
    assert bb.lower_left == [-2.0, -1.0, -5.0]
    assert bb.upper_right == [4.0, 5.0, 5.0]

def test_geometry_bounding_box():
    # First cell region: box from (0,0,0) to (10,5,2)
    xplane0 = yamc.Plane(axis="x", offset=0)
    xplane1 = yamc.Plane(axis="x", offset=10)
    yplane0 = yamc.Plane(axis="y", offset=0)
    yplane1 = yamc.Plane(axis="y", offset=5)
    zplane0 = yamc.Plane(axis="z", offset=0)
    zplane1 = yamc.Plane(axis="z", offset=2)
    region1 = (xplane0.above & xplane1.below & yplane0.above & yplane1.below & zplane0.above & zplane1.below)

    # Second cell region: box from (20,10,5) to (30,15,8)
    xplane2 = yamc.Plane(axis="x", offset=20)
    xplane3 = yamc.Plane(axis="x", offset=30)
    yplane2 = yamc.Plane(axis="y", offset=10)
    yplane3 = yamc.Plane(axis="y", offset=15)
    zplane2 = yamc.Plane(axis="z", offset=5)
    zplane3 = yamc.Plane(axis="z", offset=8)
    region2 = (xplane2.above & xplane3.below & yplane2.above & yplane3.below & zplane2.above & zplane3.below)

    mat = yamc.Material(composition={"H": 1.0}, density=1.0, units="g/cc", name="Test")
    cell1 = yamc.Cell(region=region1, material=mat)
    cell2 = yamc.Cell(region=region2, material=mat)

    geometry = yamc.Geometry([cell1, cell2])
    bbox = geometry.bounding_box()

    # The bounding box should cover both regions: from (0,0,0) to (30,15,8)
    assert bbox.lower_left == [0.0, 0.0, 0.0]
    assert bbox.upper_right == [30.0, 15.0, 8.0]
    assert bbox.center == [15.0, 7.5, 4.0]
    assert bbox.width == [30.0, 15.0, 8.0]