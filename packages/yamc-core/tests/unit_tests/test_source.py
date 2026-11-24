import yamc

def test_python_source_construction():
    # Test NeutronSource API
    src = yamc.NeutronSource()
    src.position = [1.0, 2.0, 3.0]
    src.direction = yamc.sources.Monodirectional([0.0, 0.0, 1.0])
    src.energy = yamc.sources.Discrete([2e6], [1.0])

    particle = src.sample()
    assert particle.position == [1.0, 2.0, 3.0]
    assert particle.direction == [0.0, 0.0, 1.0]
    assert particle.energy == 2e6
