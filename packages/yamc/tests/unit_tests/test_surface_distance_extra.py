import yamc
import math

def test_plane_parallel_no_intersection():
    plane = yamc.Plane(axis="x", offset=1.0)
    d = plane.distance_to_surface((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert d is None

def test_plane_on_surface():
    plane = yamc.Plane(axis="x", offset=1.0)
    d = plane.distance_to_surface((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # Accept None (no intersection if starting on surface and moving outward)
    assert d is None or math.isclose(d, 0.0, abs_tol=1e-10) or d > 0.0

def test_sphere_inside_out():
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0)
    d = sphere.distance_to_surface((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert math.isclose(d, 2.0, abs_tol=1e-10)

def test_sphere_outside_away():
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0)
    d = sphere.distance_to_surface((3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d is None

def test_cylinder_axis_parallel():
    cyl = yamc.Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0)
    d = cyl.distance_to_surface((0.5, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert d is None

def test_cylinder_on_surface():
    cyl = yamc.Cylinder(axis="z", x0=0.0, y0=0.0, radius=1.0)
    d = cyl.distance_to_surface((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # Accept None (no intersection if starting on surface and moving outward)
    assert d is None or math.isclose(d, 0.0, abs_tol=1e-10) or d > 0.0

def test_plane_negative_direction():
    plane = yamc.Plane(axis="x", offset=2.0)
    d = plane.distance_to_surface((3.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert math.isclose(d, 1.0, abs_tol=1e-10)

def test_plane_no_intersection_behind():
    plane = yamc.Plane(axis="x", offset=2.0)
    d = plane.distance_to_surface((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert d is None

# ── XCylinder distance tests ──

def test_xcylinder_distance_from_outside():
    cyl = yamc.Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0)
    # Ray from (0, 5, 0) in -y direction should hit at distance 4
    d = cyl.distance_to_surface((0.0, 5.0, 0.0), (0.0, -1.0, 0.0))
    assert math.isclose(d, 4.0, abs_tol=1e-10)

def test_xcylinder_parallel_miss():
    cyl = yamc.Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0)
    # Ray parallel to axis (x direction) from inside
    d = cyl.distance_to_surface((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))
    assert d is None

def test_xcylinder_away_miss():
    cyl = yamc.Cylinder(axis="x", y0=0.0, z0=0.0, radius=1.0)
    # Ray from outside, moving away
    d = cyl.distance_to_surface((0.0, 5.0, 0.0), (0.0, 1.0, 0.0))
    assert d is None

# ── YCylinder distance tests ──

def test_ycylinder_distance_from_outside():
    cyl = yamc.Cylinder(axis="y", x0=0.0, z0=0.0, radius=1.0)
    # Ray from (5, 0, 0) in -x direction should hit at distance 4
    d = cyl.distance_to_surface((5.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert math.isclose(d, 4.0, abs_tol=1e-10)

def test_ycylinder_parallel_miss():
    cyl = yamc.Cylinder(axis="y", x0=0.0, z0=0.0, radius=1.0)
    # Ray parallel to axis (y direction)
    d = cyl.distance_to_surface((0.5, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert d is None

# ── ZTorus distance tests ──

def test_ztorus_distance_from_outside():
    torus = yamc.Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0)
    # Ray from (10, 0, 0) in -x direction, hits outer surface at x=4
    d = torus.distance_to_surface((10.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert d is not None
    assert math.isclose(d, 6.0, abs_tol=1e-6)

def test_ztorus_distance_from_hole():
    torus = yamc.Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0)
    # Ray from inside hole at (1.5, 0, 0) in +x direction
    # Should hit inner surface at x=2 (a-c=2), distance=0.5
    d = torus.distance_to_surface((1.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d is not None
    assert math.isclose(d, 0.5, abs_tol=1e-6)

def test_ztorus_through_hole_center_miss():
    torus = yamc.Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0)
    # Ray along z-axis (through center of hole) doesn't intersect torus
    d = torus.distance_to_surface((0.0, 0.0, 10.0), (0.0, 0.0, -1.0))
    assert d is None

def test_ztorus_distance_inside_tube_outward():
    torus = yamc.Torus(axis="z", x0=0.0, y0=0.0, z0=0.0, r_major=3.0, r_minor=1.0)
    # From tube center (3, 0, 0) in +x direction, should hit at x=4, distance=1
    d = torus.distance_to_surface((3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert d is not None
    assert math.isclose(d, 1.0, abs_tol=1e-6)
