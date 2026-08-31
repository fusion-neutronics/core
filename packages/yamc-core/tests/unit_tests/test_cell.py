import yamc as csg

def test_cell_contains_simple():
    # Sphere of radius 2 at (0,0,0)
    sphere = csg.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below
    cell = csg.Cell(id=1, region=region, name=None)
    assert cell.contains(0.0, 0.0, 0.0)
    assert not cell.contains(3.0, 0.0, 0.0)

def test_cell_naming():
    sphere = csg.Sphere(surface_id=101, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below
    cell = csg.Cell(id=222, region=region, name="fuel")
    assert cell.name == "fuel"
    assert cell.id == 222

def test_cell_complex_region():
    # s1: x = 2.1, s2: x = -2.1, s3: sphere at origin, radius=4.2
    s1 = csg.Plane(axis="x", offset=2.1, surface_id=5)
    s2 = csg.Plane(axis="x", offset=-2.1, surface_id=6)
    s3 = csg.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=4.2, surface_id=1)
    region = s1.below & s2.above & s3.below
    cell = csg.Cell(id=42, region=region, name="complex")
    # Point inside all constraints
    assert cell.contains(0.0, 0.0, 0.0)
    # Point outside s1 (x > 2.1)
    assert not cell.contains(3.0, 0.0, 0.0)
    # Point outside s2 (x < -2.1)
    assert not cell.contains(-3.0, 0.0, 0.0)
    # Point outside sphere (r > 4.2)
    assert not cell.contains(0.0, 0.0, 5.0)

def test_cell_union_region():
    # Union of two spheres
    s1 = csg.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    s2 = csg.Sphere(surface_id=2, x0=3.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = s1.below | s2.below
    cell = csg.Cell(id=100, region=region, name="union")
    assert cell.contains(0.0, 0.0, 0.0)  # inside first sphere
    assert cell.contains(3.0, 0.0, 0.0)  # inside second sphere
    assert not cell.contains(6.0, 0.0, 0.0)  # outside both

def test_cell_intersection_region():
    # Intersection of two spheres
    s1 = csg.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    s2 = csg.Sphere(surface_id=2, x0=1.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = s1.below & s2.below
    cell = csg.Cell(id=101, region=region, name="intersection")
    assert cell.contains(0.0, 0.0, 0.0)  # inside both
    assert cell.contains(1.0, 0.0, 0.0)  # inside both
    assert not cell.contains(3.0, 0.0, 0.0)  # outside both

def test_cell_complement_region():
    # Complement of a sphere
    s1 = csg.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = s1.below
    region_complement = ~region
    cell = csg.Cell(id=102, region=region_complement, name="complement")
    assert not cell.contains(0.0, 0.0, 0.0)  # inside original sphere
    assert cell.contains(3.0, 0.0, 0.0)  # outside original sphere
