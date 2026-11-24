import yamc

def test_region_contains_debug():
    s = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=2.0)
    region = s.below
    cell = yamc.Cell(region=region)
    # Should be inside
    assert cell.contains(0.0, 0.0, 0.0)
    # Should be outside
    assert not cell.contains(3.0, 0.0, 0.0)
    # Should be on surface (may be inside or outside depending on convention)
    print('contains(2,0,0):', cell.contains(2.0, 0.0, 0.0))
    # Print region type for debugging
    print('region:', region)
