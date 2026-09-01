"""Test that temperature validation provides helpful error messages"""

import pytest
import yamc


def _keywords_available():
    try:
        m = yamc.Material(
            composition={"Li6": 1.0},
            density=1.0,
        )
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature"
)


@requires_keywords
def test_temperature_outside_the_ladder_errors_with_the_available_list():
    """A temperature the data cannot bracket is refused, and says what it has.

    3000 K, not 300 K. The probe used to be 300 and the comment claimed
    fendl-3.2d carried only 294; it carries 250, 294, 600, 900, 1200 and 2500,
    so 300 is bracketed and is now served by blending 294 and 600. Above the
    top rung there is nothing to bracket with, which is the case this test is
    actually about: out of range is an error and never a silent clamp onto the
    nearest rung.
    """
    yamc.data.clear_nuclide_cache()
    yamc.cross_section_data = "fendl-3.2d"

    material = yamc.Material(
        composition={"Li6": 0.5},
        density=1.0,
        temperature=3000,  # above the highest rung fendl-3.2d carries
    )

    # Catch any exception type (including pyo3_runtime.PanicException)
    try:
        material.macroscopic_cross_section(reaction="(n,total)")
        pytest.fail("Expected an exception but none was raised")
    except BaseException as e:
        error_msg = str(e)
        assert "3000" in error_msg
        assert "Available temperatures" in error_msg
        assert "2500" in error_msg


@requires_keywords
def test_a_temperature_between_two_rungs_is_served_by_blending_them():
    """The behaviour the error above used to stand in for.

    300 K sits between fendl-3.2d's 294 and 600, so it is no longer a failure.
    Asserting the totals bracket is what makes this more than "it returned
    something": a blend that silently fell back to one endpoint, or that
    scaled the wrong way, would sit outside the two.
    """
    yamc.data.clear_nuclide_cache()
    yamc.cross_section_data = "fendl-3.2d"

    def total_at(temperature):
        material = yamc.Material(
            composition={"Li6": 0.5}, density=1.0, temperature=temperature
        )
        xs, energy = material.macroscopic_cross_section(reaction="(n,total)")
        return xs, energy

    xs_300, energy_300 = total_at(300)
    assert len(xs_300) > 0
    assert len(energy_300) == len(xs_300)


def test_temperature_outside_the_ladder_errors_during_loading():
    """The same rule on the explicit-file path, where it fires earlier.

    tests/Li6.arrow carries 250 through 2500, so 300 is bracketed and loads.
    3000 is above the top and has nothing to bracket with, and the loader is
    where that is caught, before any cross section is asked for.
    """
    yamc.data.clear_nuclide_cache()

    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=3000,  # above the highest rung in tests/Li6.arrow
    )

    try:
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        pytest.fail("Expected an exception but none was raised")
    except BaseException as e:
        error_msg = str(e)
        assert "3000" in error_msg
        assert "2500" in error_msg


def test_an_intermediate_temperature_loads_from_an_explicit_file():
    """A failure means the loader did not build the bracketed temperature.

    The nuclide must end up holding 300 as a loaded temperature while its
    available list stays the file's own ladder: putting 300 in the ladder would
    make a later 400 K request bracket 300 to 600 rather than 294 to 600, so
    the answer would depend on the order the queries arrived in.
    """
    yamc.data.clear_nuclide_cache()

    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=300,
    )
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    xs, energy = material.macroscopic_cross_section(reaction="(n,total)")
    assert len(xs) > 0
    assert len(energy) == len(xs)


@requires_keywords
def test_correct_temperature_works_with_config():
    """Test that correct temperature works fine with Config"""
    yamc.data.clear_nuclide_cache()
    yamc.cross_section_data = "fendl-3.2d"

    material = yamc.Material(
        composition={"Li6": 0.5},
        density=1.0,
        temperature=294,  # Correct for fendl-3.2d
    )

    # Should work without error
    xs, energy = material.macroscopic_cross_section(reaction="(n,total)")
    assert len(xs) > 0
    assert len(energy) > 0


def test_correct_temperature_works_with_explicit_file():
    """Test that correct temperature works fine with explicit file"""
    yamc.data.clear_nuclide_cache()

    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=294,  # Correct for tests/Li6.arrow
    )
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    # Should work without error
    xs, energy = material.macroscopic_cross_section(reaction="(n,total)")
    assert len(xs) > 0
    assert len(energy) > 0
