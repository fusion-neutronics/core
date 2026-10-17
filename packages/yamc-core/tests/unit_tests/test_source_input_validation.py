"""Source inputs that are NaN or infinite (and probabilities or strengths that
are negative) raise ValueError where they are given, naming the parameter and
the value, and a model file is checked the same way as it is loaded.

Before, all of these were accepted: a NaN position or energy surfaced
mid-transport as lost particles or a NaN tally, a negative strength picked
sources wrongly without a word, and a NaN tokamak parameter passed checks
written as ``value <= 0``.
"""

import math

import pytest
import yamc

NAN = math.nan
INF = math.inf


@pytest.mark.parametrize("source_type", [yamc.NeutronSource, yamc.PhotonSource])
@pytest.mark.parametrize(
    ("position", "says"),
    [
        ((NAN, 0.0, 0.0), r"position \[NaN, 0.0, 0.0\] has a NaN coordinate \(x = NaN\)"),
        ((0.0, INF, 0.0), r"has an infinite coordinate \(y = inf\)"),
        ((0.0, 0.0, -INF), r"has an infinite coordinate \(z = -inf\)"),
    ],
)
def test_a_position_that_is_not_finite_raises(source_type, position, says):
    with pytest.raises(ValueError, match=says):
        source_type(position=position, energy=1e6)


def test_setting_a_position_that_is_not_finite_raises_and_keeps_the_old_one():
    source = yamc.NeutronSource(position=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match=r"has a NaN coordinate \(z = NaN\)"):
        source.position = (0.0, 0.0, NAN)
    assert source.position == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("strength", "says"),
    [
        (NAN, r"strength must be a finite number \(got NaN\)"),
        (INF, r"strength must be a finite number \(got inf\)"),
        (-1.0, r"strength must not be negative \(got -1\)"),
    ],
)
def test_a_strength_that_is_nan_infinite_or_negative_raises(strength, says):
    with pytest.raises(ValueError, match=says):
        yamc.NeutronSource(strength=strength)
    source = yamc.NeutronSource(strength=2.0)
    with pytest.raises(ValueError, match=says):
        source.strength = strength
    assert source.strength == 2.0


def test_a_strength_of_zero_is_allowed():
    assert yamc.NeutronSource(strength=0.0).strength == 0.0


@pytest.mark.parametrize("energy", [NAN, INF])
def test_an_energy_that_is_not_finite_raises(energy):
    with pytest.raises(ValueError, match=r"energy must be a finite number \(got (NaN|inf)\)"):
        yamc.NeutronSource(energy=energy)
    with pytest.raises(ValueError, match="energy must be a finite number"):
        yamc.PhotonSource(energy=energy)


@pytest.mark.parametrize(
    ("energies", "probabilities", "says"),
    [
        ([1e6, NAN], [1.0, 1.0], r"Discrete energies\[1\] must be a finite number \(got NaN\)"),
        ([INF], [1.0], r"Discrete energies\[0\] must be a finite number \(got inf\)"),
        ([1e6, 2e6], [1.0, NAN], r"Discrete probabilities\[1\] must be a finite number \(got NaN\)"),
        ([1e6], [INF], r"Discrete probabilities\[0\] must be a finite number \(got inf\)"),
        ([1e6, 2e6], [1.0, -0.5], r"Discrete probabilities\[1\] is -0.5: probabilities cannot be negative"),
    ],
)
def test_a_discrete_entry_that_is_not_finite_or_is_negative_raises(energies, probabilities, says):
    with pytest.raises(ValueError, match=says):
        yamc.sources.Discrete(energies, probabilities)


@pytest.mark.parametrize(
    ("make", "says"),
    [
        (lambda: yamc.sources.Uniform(NAN, 1.0), r"Uniform low must be a finite number \(got NaN\)"),
        (lambda: yamc.sources.Uniform(0.0, INF), r"Uniform high must be a finite number \(got inf\)"),
        (lambda: yamc.sources.Normal(NAN, 1.0), r"Normal mean must be a finite number \(got NaN\)"),
        (lambda: yamc.sources.Normal(1.0, INF), r"Normal std_dev must be a finite number \(got inf\)"),
        (lambda: yamc.sources.Normal(1.0, NAN), r"Normal std_dev must be a finite number \(got NaN\)"),
        (lambda: yamc.sources.Histogram([0.0, NAN], [1.0]), r"Histogram boundary 1 is NaN"),
    ],
)
def test_a_distribution_parameter_that_is_not_finite_raises(make, says):
    with pytest.raises(ValueError, match=says):
        make()


def _ring(**overrides):
    kwargs = {
        "radius": yamc.sources.Discrete([300.0], [1.0]),
        "phi": yamc.sources.Uniform(0.0, 2 * math.pi),
        "z": yamc.sources.Discrete([0.0], [1.0]),
    }
    kwargs.update(overrides)
    return yamc.sources.CylindricalRing(**kwargs)


def test_a_cylindrical_ring_with_an_input_that_is_not_finite_raises():
    with pytest.raises(ValueError, match=r"CylindricalRing origin \[0.0, NaN, 0.0\] has a NaN coordinate \(y = NaN\)"):
        _ring(origin=(0.0, NAN, 0.0))
    with pytest.raises(ValueError, match=r"radius must be a finite number \(got inf\)"):
        _ring(radius=INF)
    with pytest.raises(ValueError, match=r"z must be a finite number \(got NaN\)"):
        _ring(z=NAN)
    assert _ring(origin=(1.0, 2.0, 3.0)).origin == [1.0, 2.0, 3.0]


_TOKAMAK = {
    "major_radius": 906.0,
    "minor_radius": 292.258,
    "elongation": 1.557,
    "triangularity": 0.270,
    "mode": "H",
    "ion_density_centre": 1.09e20,
    "ion_density_peaking_factor": 1.0,
    "ion_density_pedestal": 1.09e20,
    "ion_density_separatrix": 3e19,
    "ion_temperature_centre": 45.9e3,
    "ion_temperature_peaking_factor": 8.06,
    "ion_temperature_beta": 6.0,
    "ion_temperature_pedestal": 6.09e3,
    "ion_temperature_separatrix": 0.1e3,
    "pedestal_radius": 0.8 * 292.258,
    "shafranov_factor": 0.44789,
    "mesh_resolution": (4, 4),
    "grid_density": 10,
}


@pytest.mark.parametrize(
    "name",
    [
        "major_radius",
        "elongation",
        "ion_density_centre",
        "ion_density_peaking_factor",
        "ion_temperature_centre",
        "ion_temperature_beta",
        "pedestal_radius",
        "shafranov_factor",
        "start_angle",
    ],
)
def test_a_tokamak_parameter_that_is_nan_raises_naming_it(name):
    with pytest.raises(ValueError, match=rf"^{name} must be a finite number \(got NaN\)$"):
        yamc.sources.tokamak_source(**{**_TOKAMAK, name: NAN})


def test_a_nan_tokamak_fuel_fraction_raises():
    with pytest.raises(ValueError, match="fuel fractions must be greater than 0 and at most 1"):
        yamc.sources.tokamak_source(**_TOKAMAK, fuel={"D": NAN, "T": 0.5})


def _saved_model():
    sphere = yamc.Sphere(radius=2.0, boundary="vacuum")
    geometry = yamc.Geometry([yamc.Cell(region=sphere.below)])
    source = yamc.NeutronSource(
        position=(1.0, 2.0, 3.0),
        energy=yamc.sources.Discrete([1e6, 2e6], [0.25, 0.75]),
        strength=2.0,
    )
    # A list: a model refuses an explicit strength on a lone bare source.
    return yamc.Model(geometry, source=[source]).to_json()


@pytest.mark.parametrize(
    ("before", "after", "says"),
    [
        ('"strength":2.0', '"strength":-1.0', r"strength must not be negative \(got -1\)"),
        ('"probabilities":[0.25,0.75]', '"probabilities":[0.25,-0.75]',
         r"Discrete probabilities\[1\] is -0.75: probabilities cannot be negative"),
    ],
)
def test_a_model_file_is_checked_as_a_constructed_source_is(before, after, says):
    saved = _saved_model()
    assert yamc.Model.load(saved).source[0].strength == 2.0
    edited = saved.replace(before, after)
    assert edited != saved
    with pytest.raises(ValueError, match=says):
        yamc.Model.load(edited)
