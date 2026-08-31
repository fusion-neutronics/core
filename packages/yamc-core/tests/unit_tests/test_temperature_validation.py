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
def test_temperature_error_message_with_config():
    """Test that using wrong temperature with Config gives helpful error"""
    yamc.data.clear_nuclide_cache()
    yamc.cross_section_data = "fendl-3.2d"

    material = yamc.Material(
        composition={"Li6": 0.5},
        density=1.0,
        temperature=300,  # fendl-3.2d has 294, not 300
    )

    # Should raise error with helpful message about available temperatures
    # Catch any exception type (including pyo3_runtime.PanicException)
    try:
        material.macroscopic_cross_section(reaction="(n,total)")
        pytest.fail("Expected an exception but none was raised")
    except BaseException as e:
        error_msg = str(e)
        assert "Temperature '300' not available" in error_msg
        assert "Available temperatures" in error_msg
        assert "294" in error_msg


def test_temperature_error_message_with_explicit_file():
    """Test that using wrong temperature with explicit file gives helpful error"""
    yamc.data.clear_nuclide_cache()

    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=300,  # tests/Li6.arrow has 294, not 300
    )

    # Should raise error with helpful message during loading
    try:
        material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
        pytest.fail("Expected an exception but none was raised")
    except BaseException as e:
        error_msg = str(e)
        assert "300" in error_msg or "No matching temperatures" in error_msg
        assert "294" in error_msg


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
