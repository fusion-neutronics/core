"""Tests for the typed ``simulate_transport(capture_tracks=...)`` selection.

``capture_tracks`` replaces the old overloaded ``tracks=`` argument. Histories
are addressed by their global 0-based index, so the captured set is exact and
independent of thread scheduling:

    None            -> off
    int N           -> the first N histories (range(0, N))
    range(a, b, s)  -> histories a..b-1, with stride s
    [i, j, ...]     -> exactly those indices
    'all'           -> every history (warns)
"""
import pytest

import yamc


def _model(total_particles=100):
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=iron)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(position=(0, 0, 0), energy=1.0e6)
    model = yamc.Model(
        geometry=geometry,
        source=source,
        verbose=[],
    )
    return model, total_particles


def _simulate(**run_kwargs):
    """Build the standard model and run it at the helper's particle count."""
    model, total_particles = _model()
    return model.simulate_transport(
        total_particles=total_particles, seed=42, **run_kwargs
    )


def _captured_indices(capture_tracks):
    """Sorted set of global history indices captured for this selection."""
    results = _simulate(capture_tracks=capture_tracks)
    if results.tracks is None:
        return None
    return sorted({rec["history"] for rec in results.tracks.to_dataframe_records()})


def test_none_disables_tracking():
    results = _simulate()
    assert results.tracks is None
    assert _simulate(capture_tracks=None).tracks is None


def test_int_captures_first_n():
    assert _captured_indices(5) == [0, 1, 2, 3, 4]


def test_range_is_half_open():
    assert _captured_indices(range(2, 6)) == [2, 3, 4, 5]


def test_range_with_stride():
    assert _captured_indices(range(0, 20, 5)) == [0, 5, 10, 15]


def test_explicit_index_list():
    assert _captured_indices([3, 7, 11]) == [3, 7, 11]


def test_all_captures_every_history_and_warns():
    with pytest.warns(UserWarning, match="capture_tracks='all'"):
        indices = _captured_indices("all")
    assert indices == list(range(100))


def test_bool_is_rejected():
    # bool is a subclass of int; capture_tracks=True must not silently mean "1".
    with pytest.raises(TypeError, match="True/False"):
        _simulate(capture_tracks=True)


def test_negative_count_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _simulate(capture_tracks=-1)


def test_unknown_string_rejected():
    with pytest.raises(ValueError, match="must be 'all'"):
        _simulate(capture_tracks="everything")


def test_descending_range_rejected():
    with pytest.raises(ValueError, match="positive step"):
        _simulate(capture_tracks=range(30, 20, -1))


def test_negative_indices_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _simulate(capture_tracks=[1, -2, 3])
