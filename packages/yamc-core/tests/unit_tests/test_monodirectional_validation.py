"""A monodirectional source direction that points nowhere is refused when it
is made, with a ValueError that says why.

The zero vector used to reach a Rust panic (``PanicException``, which is not
an ``Exception``), and a NaN or infinite component was kept and surfaced
mid-transport as "Maximum lost particles exceeded". A vector too large to
square became the zero direction, and one too small to square was called the
zero vector.
"""

import math

import pytest
import yamc


@pytest.mark.parametrize(
    ("direction", "says"),
    [
        ((0.0, 0.0, 0.0), "is the zero vector"),
        ((0, 0, 0), "is the zero vector"),
        ((-0.0, 0.0, -0.0), "is the zero vector"),
        ((math.nan, 0.0, 1.0), r"has a NaN component \(u = NaN\)"),
        ((math.inf, 0.0, 0.0), r"has an infinite component \(u = inf\)"),
        ((0.0, -math.inf, 0.0), r"has an infinite component \(v = -inf\)"),
    ],
)
def test_a_direction_that_points_nowhere_raises_value_error(direction, says):
    with pytest.raises(ValueError, match=says):
        yamc.sources.Monodirectional(direction)


def test_setting_a_direction_that_points_nowhere_raises_and_keeps_the_old_one():
    mono = yamc.sources.Monodirectional((0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="is the zero vector"):
        mono.direction = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="has a NaN component"):
        mono.direction = (math.nan, 1.0, 0.0)
    assert mono.direction == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ((0.0, 0.0, 2.0), (0.0, 0.0, 1.0)),
        ((3.0, 4.0, 0.0), (0.6, 0.8, 0.0)),
        ((1e200, 1e200, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0)),
        ((1e-200, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ],
)
def test_any_other_direction_is_normalized(direction, expected):
    got = yamc.sources.Monodirectional(direction).direction
    assert got == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_a_model_file_with_a_zero_direction_fails_to_load():
    sphere = yamc.Sphere(radius=2.0, boundary="vacuum")
    geometry = yamc.Geometry([yamc.Cell(region=sphere.below)])
    source = yamc.NeutronSource(direction=yamc.sources.Monodirectional((0.0, 0.0, 1.0)))
    saved = yamc.Model(geometry, source=source).to_json()
    assert yamc.Model.load(saved).source[0].direction.direction == [0.0, 0.0, 1.0]

    edited = saved.replace('"direction":[0.0,0.0,1.0]', '"direction":[0.0,0.0,0.0]')
    assert edited != saved
    with pytest.raises(ValueError, match="is the zero vector"):
        yamc.Model.load(edited)
