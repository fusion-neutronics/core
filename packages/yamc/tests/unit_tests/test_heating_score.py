import yamc

def test_heating_score():
    """Test that model accepts tallies in constructor."""
    # Create minimal geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0, boundary='vacuum')
    region = sphere.below
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    cell = yamc.Cell(name="test", region=region, material=material)
    geometry = yamc.Geometry(cells=[cell])
    
    # Create source
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        direction=yamc.sources.Monodirectional([0.0,0.0,1.0]),
        energy=yamc.sources.Discrete([1400000.0], [1.0])
    )
    tally = yamc.Tally(scores=['heating'])
    tallies = [tally]

    # Model should accept tallies
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
    results = model.simulate_transport(total_particles=10)
    assert results[tally].mean[0] > 0.1
