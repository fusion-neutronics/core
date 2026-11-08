import yamc


def _minimal_geometry():
    """Create a minimal geometry for Model construction."""
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=100.0, boundary='vacuum')
    cell = yamc.Cell(region=sphere.below)
    return yamc.Geometry([cell])


def test_python_settings_construction():
    # Use NeutronSource API
    src = yamc.NeutronSource()
    src.position = [0.0, 0.0, 0.0]
    src.direction = yamc.sources.Monodirectional([0.0, 1.0, 0.0])
    src.energy = yamc.sources.Discrete([1e5], [1.0])

    geometry = _minimal_geometry()
    model = yamc.Model(geometry, source=src)
    # source getter now returns a list
    assert len(model.source) == 1
    assert isinstance(model.source[0].position, list)
    assert model.source[0].position == [0.0, 0.0, 0.0]
    assert model.source[0].energy.energies == [1e5]
    assert model.source[0].energy.probabilities == [1.0]

def test_python_settings_source_construction():
    # Test source is accessible after Model construction
    src = yamc.NeutronSource(
        position=[1.0, 1.0, 1.0],
        energy=yamc.sources.Discrete([1e6], [1.0])
    )

    geometry = _minimal_geometry()
    model = yamc.Model(geometry, source=src)
    assert len(model.source) == 1
    assert isinstance(model.source[0].position, list)
    assert model.source[0].position == [1.0, 1.0, 1.0]
    assert model.source[0].energy.energies == [1e6]
    assert model.source[0].energy.probabilities == [1.0]

def test_python_settings_multi_source():
    # Test passing a list of sources
    src1 = yamc.NeutronSource(energy=1e6, strength=2.0)
    src2 = yamc.NeutronSource(energy=14e6, strength=3.0)

    geometry = _minimal_geometry()
    model = yamc.Model(geometry, source=[src1, src2])
    assert len(model.source) == 2
    assert model.source[0].strength == 2.0
    assert model.source[1].strength == 3.0

def test_python_source_strength_default():
    src = yamc.NeutronSource()
    assert src.strength == 1.0
    src.strength = 5.0
    assert src.strength == 5.0


def test_single_source_with_nondefault_strength_rejected():
    """A bare single source with strength != 1.0 is rejected at Model construction.

    Strength is only meaningful for multi-source weighting. To use it, wrap
    the source in a list (so the multi-source intent is explicit).
    """
    import pytest
    src = yamc.NeutronSource(strength=2.0)
    geometry = _minimal_geometry()
    with pytest.raises(ValueError, match="strength is only meaningful"):
        yamc.Model(geometry, source=src)


def test_single_source_in_list_with_nondefault_strength_allowed():
    """A list with one source can have any strength (caller is opting in)."""
    src = yamc.NeutronSource(strength=2.0)
    geometry = _minimal_geometry()
    model = yamc.Model(geometry, source=[src])
    assert model.source[0].strength == 2.0


def test_single_source_default_strength_allowed():
    """A bare single source with default strength is the common case -- must work."""
    src = yamc.NeutronSource()
    geometry = _minimal_geometry()
    model = yamc.Model(geometry, source=src)
    assert model.source[0].strength == 1.0

def test_multi_source_strength_sampling():
    """Sampling frequency should be proportional to relative source strength.

    Two point sources at distinct locations with strength 0.1 and 0.9.
    Run a void simulation with tracking and count source particles by
    starting position --- the second source should be sampled ~9x more often.
    """
    src1 = yamc.NeutronSource(position=[10, 10, 10], strength=0.1)
    src2 = yamc.NeutronSource(position=[-20, -30, -40], strength=0.9)

    sphere = yamc.Sphere(radius=100.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below)
    geometry = yamc.Geometry([cell])

    n_particles = 10_000

    model = yamc.Model(geometry, tallies=[], source=[src1, src2])
    tracks = model.simulate_transport(
        total_particles=n_particles * 1, seed=42, capture_tracks='all'
    ).tracks

    # Count source particles by exact starting position (generation-0 tracks)
    pos1 = [10.0, 10.0, 10.0]
    pos2 = [-20.0, -30.0, -40.0]
    count_src1 = 0
    count_src2 = 0
    for track in tracks.tracks:
        if track.generation == 0:
            start_pos = list(track.events[0].position)
            if start_pos == pos1:
                count_src1 += 1
            elif start_pos == pos2:
                count_src2 += 1
            else:
                raise AssertionError(
                    f"Unexpected source position {start_pos}, "
                    f"expected {pos1} or {pos2}"
                )

    assert count_src1 + count_src2 == n_particles
    # Expected ratio src2/src1 = 0.9/0.1 = 9
    ratio = count_src2 / max(count_src1, 1)
    assert 6.0 < ratio < 13.0, f"Expected ratio ~9, got {ratio}"
