"""ImplicitSurface: the zero set of a formula as an ordinary CSG surface."""

import math

import pytest

import yamc

GYROID = (
    "sin(pi * x) * cos(pi * y) + sin(pi * y) * cos(pi * z)"
    " + sin(pi * z) * cos(pi * x) - 0.3"
)


def _box(half):
    """The cube [-half, half]^3 as a region of planes."""
    region = yamc.Plane("x", -half).above & yamc.Plane("x", half).below
    region = region & yamc.Plane("y", -half).above & yamc.Plane("y", half).below
    return region & yamc.Plane("z", -half).above & yamc.Plane("z", half).below


def test_an_implicit_surface_is_an_ordinary_surface():
    ball = yamc.ImplicitSurface("sqrt(x^2+y^2+z^2)-5", surface_id=1, name="ball")
    assert isinstance(ball, yamc.Surface)
    # Kept in canonical form, which is what repr shows.
    assert ball.formula == "sqrt(x^2 + y^2 + z^2) - 5"
    assert repr(ball) == (
        "Surface(id=1, type=ImplicitSurface('sqrt(x^2 + y^2 + z^2) - 5'), boundary=None)"
    )
    assert yamc.Sphere(radius=5.0).formula is None
    # evaluate is the formula's value; here a signed distance.
    assert ball.evaluate((0.0, 0.0, 0.0)) == -5.0
    assert ball.evaluate((0.0, 6.0, 0.0)) == 1.0
    assert ball.below.contains((1.0, 2.0, 3.0))
    assert ball.above.contains((5.0, 1.0, 0.0))
    shell = ball.above & yamc.ImplicitSurface("x^2 + y^2 + z^2 - 49").below
    assert shell.contains((0.0, 0.0, 6.0)) and not shell.contains((0.0, 0.0, 4.0))
    assert (~ball.below | shell).contains((0.0, 0.0, 8.0))
    # A box proved from the formula, a little larger than the sphere.
    box = ball.bounding_box_inside()
    assert all(-5.2 < lo <= -5.0 for lo in box.lower_left)
    assert all(5.0 <= hi < 5.2 for hi in box.upper_right)
    assert ball.bounding_box_outside() is None
    assert ball.distance_to_surface((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)) == pytest.approx(
        5.0, abs=1e-10
    )
    assert ball.distance_to_surface((0.0, 0.0, 6.0), (0.0, 0.0, 1.0)) is None
    vacuum = yamc.ImplicitSurface("x^4 + y^4 + z^4 - 16", boundary="vacuum")
    assert repr(vacuum).endswith("boundary='vacuum')")
    # Long formulas are shortened in repr, but kept whole.
    long = " + ".join(f"(x - {i})^2" for i in range(20)) + " - 1e6"
    surface = yamc.ImplicitSurface(long)
    assert repr(surface).endswith("...'), boundary=None)") and len(repr(surface)) < 120
    assert yamc.ImplicitSurface(surface.formula).formula == surface.formula


def test_a_search_that_runs_out_finds_no_crossing():
    # x^2 - x^2 is 0, which interval arithmetic cannot see, so a search
    # parallel to this plane runs out of evaluations long before 2^20 cm:
    # there is no crossing to report, rather than the point it reached.
    plane = yamc.ImplicitSurface("x^2 - x^2 + y - 1")
    assert plane.distance_to_surface((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is None
    assert plane.distance_to_surface((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == (
        pytest.approx(1.0, abs=1e-10)
    )


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        # Parse errors say where, with a caret under the place.
        ("x + * y", r"found '\*' \(column 5\):\n  x \+ \* y\n      \^"),
        ("(x + y", r"expected '\)' to close the '\(' at column 1"),
        ("foo(x) - 1", r"unknown name 'foo' \(the coordinates are x, y and z"),
        ("1e400 * x - 1", r"too large for a 64-bit float \(column 1\)"),
        ("x^2.5 - 1", r"needs a base that cannot be negative, and x can be"),
        ("smin(x, y, 0) - 1", r"must be a positive constant .* \(column 12\)"),
        # Operations that can be undefined, or divide by zero, where the
        # formula is used, by name and place.
        (
            "sqrt(1 - x^2 - y^2) - z",
            r"sqrt\(1 - x\^2 - y\^2\) can take the square root",
        ),
        ("log(x) - 1", r"log\(x\) can take the logarithm of a negative number"),
        ("1 / (x - 1) - 2", r"1 / \(x - 1\) can divide by zero .* \(column 3\)"),
        # Formulas that never change sign.
        ("x^2 + y^2 + z^2 + 1", "is never negative, so nothing would be below"),
        ("-x^2 - 1", "is never positive within 1048576 cm of the origin"),
        # A formula that is zero wherever x < 0.
        ("max(x, 0) * y", "is zero, or too small for its sign to mean anything"),
        # One interval arithmetic cannot show is defined, and how to fix it.
        (
            "sqrt(x^2 - 2 * x + 2) - 5",
            r"could not be shown never to take the square root .* write "
            r"x\^2 - 2 \* x \+ 2 as \(x - 1\)\^2 \+ 1",
        ),
    ],
)
def test_bad_formulas_fail_with_clear_messages(formula, message):
    with pytest.raises(ValueError, match=message):
        yamc.ImplicitSurface(formula)


def test_bad_options_fail_with_clear_messages():
    with pytest.raises(ValueError, match="boundary must be None or 'vacuum'"):
        yamc.ImplicitSurface("x - 1", boundary="reflective")
    with pytest.raises(TypeError):
        yamc.ImplicitSurface("x - 1", 3)
    with pytest.raises(ValueError, match="ImplicitSurface cannot be offset"):
        yamc.OffsetSurface(yamc.ImplicitSurface("sqrt(x^2 + y^2 + z^2) - 5"), 1.0)


def test_the_docstring_examples_build():
    gyroid = yamc.ImplicitSurface(GYROID)
    assert gyroid.bounding_box_inside() is None
    box = _box(5.0)
    yamc.Geometry(
        [yamc.Cell(region=box & gyroid.below), yamc.Cell(region=box & gyroid.above)]
    )

    port = yamc.ImplicitSurface("((x - 225) / 60)^4 + (y / 30)^4 + (z / 50)^4 - 1")
    shell = yamc.Sphere(radius=200.0).above & yamc.Sphere(radius=250.0).below
    yamc.Geometry(
        [yamc.Cell(region=shell & port.above), yamc.Cell(region=shell & port.below)]
    )
    assert (shell & port.below).contains((225.0, 0.0, 0.0))
    assert (shell & port.above).contains((0.0, 225.0, 0.0))

    pair = yamc.ImplicitSurface(
        "smin(sqrt((x - 4)^2 + y^2 + z^2) - 5, sqrt((x + 4)^2 + y^2 + z^2) - 5, 2)"
    )
    # The fillet fills the waist between the spheres: inside the blend,
    # outside either sphere.
    assert pair.below.contains((0.0, 3.2, 0.0))
    assert yamc.Sphere(x0=4.0, radius=5.0).above.contains((0.0, 3.2, 0.0))


def test_a_cell_must_bound_a_formula_without_a_box():
    gyroid = yamc.ImplicitSurface(GYROID, surface_id=3)
    with pytest.raises(
        ValueError, match="uses ImplicitSurface 3, which has no bounding box"
    ):
        yamc.Geometry([yamc.Cell(region=gyroid.below)])
    with pytest.raises(ValueError, match="reaches beyond 1048576 cm"):
        yamc.Geometry([yamc.Cell(region=_box(2e6) & gyroid.below)])


def test_the_formula_cell_holds_the_shape_volume():
    ball = yamc.ImplicitSurface("sqrt(x^2 + y^2 + z^2) - 5")
    result = ball.below.calculate_volume(samples=200_000, seed=7)
    assert abs(result.volume - 4.0 / 3.0 * math.pi * 125.0) < 4.0 * result.std_dev


def test_a_model_with_implicit_surfaces_saves_and_loads(tmp_path):
    gyroid = yamc.ImplicitSurface(GYROID, surface_id=1, name="lattice")
    world = yamc.ImplicitSurface(
        "x^4 + y^4 + z^4 - 1e4", surface_id=2, boundary="vacuum"
    )
    cells = [
        yamc.Cell(region=world.below & gyroid.below, name="struts"),
        yamc.Cell(region=world.below & gyroid.above, name="pores"),
    ]
    model = yamc.Model(
        geometry=yamc.Geometry(cells),
        source=yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0])),
    )
    # The formulas are stored as their canonical text.
    assert GYROID in model.to_json()
    loaded = yamc.Model.load(model.to_json())
    assert loaded.to_json() == model.to_json()
    path = tmp_path / "lattice.json"
    model.save(str(path))
    assert yamc.Model.load(str(path)).to_json() == model.to_json()
    # And read back to the same surfaces.
    for point in [(0.1, 0.2, 0.3), (0.6, 0.2, 0.3), (-3.3, 4.1, 0.7), (9.9, 0.0, 0.0)]:
        assert repr(loaded.geometry.find_cell(*point)) == repr(
            model.geometry.find_cell(*point)
        )
    assert loaded.geometry.find_cell(10.1, 0.0, 0.0) is None
