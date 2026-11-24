"""Fixed-point sources: position=(x, y, z) tuples and the position getter.

yamc.Point was removed; a fixed point source is given as a plain (x, y, z)
tuple/list, and ``source.position`` returns that position as a list.
"""
import yamc


def test_tuple_position_construction():
    """A position tuple sets a fixed-point source."""
    source = yamc.NeutronSource(
        position=(1.0, 2.0, 3.0),
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
    )
    assert source.position == [1.0, 2.0, 3.0]


def test_fixed_point_sampling():
    """A fixed-point source always samples the same position."""
    source = yamc.NeutronSource(
        position=(5.0, -3.0, 7.5),
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
    )
    for _ in range(100):
        assert source.sample().position == [5.0, -3.0, 7.5]


def test_position_getter_returns_list():
    """source.position returns the (x, y, z) as a list."""
    source = yamc.NeutronSource(position=[1.0, 2.0, 3.0])
    retrieved = source.position
    assert isinstance(retrieved, list)
    assert retrieved == [1.0, 2.0, 3.0]


def test_position_setter():
    """Setting position accepts a tuple/list."""
    source = yamc.NeutronSource()
    source.position = [1.0, 2.0, 3.0]
    assert source.position == [1.0, 2.0, 3.0]


def test_default_position_is_origin():
    """A default source sits at the origin."""
    source = yamc.NeutronSource()
    assert source.position == [0.0, 0.0, 0.0]
    assert source.sample().position == [0.0, 0.0, 0.0]


def test_position_can_be_reassigned():
    """The position can be switched between fixed points."""
    source = yamc.NeutronSource()
    source.position = [1.0, 2.0, 3.0]
    assert source.position == [1.0, 2.0, 3.0]
    source.position = [4.0, 5.0, 6.0]
    assert source.position == [4.0, 5.0, 6.0]


def test_source_repr_shows_position_tuple():
    """The source repr shows the position as a tuple, not a Point class."""
    source = yamc.NeutronSource(position=(1.0, 2.0, 3.0))
    repr_str = repr(source)
    assert "NeutronSource" in repr_str
    assert "(1, 2, 3)" in repr_str


def test_various_positions():
    """A range of fixed positions sample correctly."""
    for position in [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-5.0, 0.0, 10.0],
        [100.0, -200.0, 300.0],
    ]:
        source = yamc.NeutronSource(position=position)
        assert source.position == position
        assert source.sample().position == position


def test_photon_source_tuple_position():
    """Tuple position works for PhotonSource too."""
    source = yamc.PhotonSource(position=(1.0, 0.0, 0.0), energy=1.0e6)
    assert source.position == [1.0, 0.0, 0.0]
    assert source.sample().position == [1.0, 0.0, 0.0]


def test_integration_with_discrete_energy():
    """Fixed-point position with a Discrete energy distribution."""
    source = yamc.NeutronSource(
        position=(1.0, 2.0, 3.0),
        energy=yamc.sources.Discrete([14.06e6, 2.5e6], [0.9, 0.1]),
    )
    for _ in range(10):
        particle = source.sample()
        assert particle.position == [1.0, 2.0, 3.0]
        assert particle.energy in [14.06e6, 2.5e6]
