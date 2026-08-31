import yamc
import math

def test_xplane_distance():
    plane = yamc.Plane(axis="x", offset=5.0)
    # From (0,0,0) in +x direction
    d = plane.distance_to_surface((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d == 5.0
    # From (0,0,0) in -x direction
    d2 = plane.distance_to_surface((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert d2 is None
    # From (10,0,0) in -x direction
    d3 = plane.distance_to_surface((10.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert d3 == 5.0
    # Parallel direction
    d4 = plane.distance_to_surface((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert d4 is None

def test_sphere_distance():
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    d = sphere.distance_to_surface((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert math.isclose(d, 1.0, abs_tol=1e-10)
    d2 = sphere.distance_to_surface((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert math.isclose(d2, 1.0, abs_tol=1e-10)
    d3 = sphere.distance_to_surface((2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d3 is None

def test_cylinder_distance():
    cyl = yamc.Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0)
    d = cyl.distance_to_surface((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert math.isclose(d, 1.0, abs_tol=1e-10)
    d2 = cyl.distance_to_surface((0.0, 2.0, 0.0), (0.0, -1.0, 0.0))
    assert math.isclose(d2, 1.0, abs_tol=1e-10)
    d3 = cyl.distance_to_surface((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert math.isclose(d3, 1.0, abs_tol=1e-10)
    d4 = cyl.distance_to_surface((2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d4 is None