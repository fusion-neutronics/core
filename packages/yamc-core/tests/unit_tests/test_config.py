import pytest
import yamc


def _keywords_available():
    """Check if keyword download tests can work (requires download feature)."""
    try:
        nuc = yamc.Nuclide("Li6")
        nuc.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature"
)

@pytest.fixture(autouse=True)
def clear_config():
    """Clear config state before each test"""
    yamc.cross_section_data = None
    yamc.transmutation_decay_data = None
    yield
    # No cleanup needed after test

def test_set_cross_sections_with_dict():
    """Test setting cross sections with a dictionary"""
    cross_sections = {
        "Li6": "tendl-2025",
        "Li7": "tests/Li7.arrow"
    }
    yamc.cross_section_data = cross_sections

    # Verify the cross sections were set
    assert yamc.lookup_cross_section_data("Li6") == "tendl-2025"
    assert yamc.lookup_cross_section_data("Li7") == "tests/Li7.arrow"

def test_set_cross_sections_with_string():
    """Test setting cross sections with a global keyword string"""
    yamc.cross_section_data = "tendl-2025"

    # Any nuclide should now return the global keyword
    assert yamc.lookup_cross_section_data("Fe56") == "tendl-2025"
    assert yamc.lookup_cross_section_data("Li6") == "tendl-2025"

def test_set_cross_section_single_nuclide_path():
    """Test setting a single nuclide with a file path"""
    yamc.set_cross_section_data_entry("Fe56", "tests/Fe56.arrow")
    assert yamc.lookup_cross_section_data("Fe56") == "tests/Fe56.arrow"

def test_set_cross_section_single_nuclide_keyword():
    """Test setting a single nuclide with a keyword"""
    yamc.set_cross_section_data_entry("Fe56", "tendl-2025")
    assert yamc.lookup_cross_section_data("Fe56") == "tendl-2025"

def test_set_cross_section_global_keyword():
    """Test setting a global keyword using set_cross_section_data_entry"""
    yamc.set_cross_section_data_entry("tendl-2025")
    assert yamc.lookup_cross_section_data("Li6") == "tendl-2025"
    assert yamc.lookup_cross_section_data("Fe56") == "tendl-2025"

def test_set_cross_sections_invalid_type():
    """Test that assigning an invalid type raises TypeError"""
    with pytest.raises(TypeError):
        yamc.cross_section_data = 123  # Invalid type

def test_mixed_global_and_specific_config():
    """Test mixing global default with specific nuclide overrides"""
    # Set global default to TENDL
    yamc.cross_section_data = "tendl-2025"

    # Override specific nuclides to FENDL
    yamc.set_cross_section_data_entry("Fe56", "fendl-3.2d")
    yamc.set_cross_section_data_entry("Li6", "tests/Li6.arrow")

    # Check that specific overrides work
    assert yamc.lookup_cross_section_data("Fe56") == "fendl-3.2d"
    assert yamc.lookup_cross_section_data("Li6") == "tests/Li6.arrow"

    # Check that other nuclides fall back to global default
    assert yamc.lookup_cross_section_data("Be9") == "tendl-2025"
    assert yamc.lookup_cross_section_data("U235") == "tendl-2025"

def test_explicit_path_override_in_nuclide_loading():
    """Test that explicit paths in load work correctly"""
    # Set up global config with local HDF5 files
    yamc.cross_section_data = {
        "Li6": "tests/Li6.arrow",
        "Be9": "tests/Be9.arrow"
    }

    # Create nuclide and load from explicit path
    li6_explicit = yamc.Nuclide("Li6")
    li6_explicit.read_nuclear_data("tests/Li6.arrow")

    # Should load successfully
    assert li6_explicit.name == "Li6"

    # Verify it loaded
    assert len(li6_explicit.available_temperatures) > 0

def test_file_validation_valid_path():
    """Test that setting a valid file path succeeds"""
    # Use a file that exists in the test directory
    yamc.cross_section_data = {
        "Li6": "tests/Li6.arrow",
        "Fe56": "tests/Fe56.arrow"
    }
    assert yamc.lookup_cross_section_data("Li6") == "tests/Li6.arrow"
    assert yamc.lookup_cross_section_data("Fe56") == "tests/Fe56.arrow"

def test_file_validation_invalid_path():
    """Test that setting a nonexistent file path panics"""
    # Rust panics from pyo3 come through as BaseException subclasses
    with pytest.raises(BaseException) as exc_info:
        yamc.cross_section_data = {
            "Fe56": "/this/path/does/not/exist.arrow"
        }
    # Verify it's the right kind of error
    assert "does not exist" in str(exc_info.value)

def test_file_validation_keyword_bypasses_check():
    """Test that keywords bypass file existence validation"""
    # These should not check file existence
    yamc.cross_section_data = {"Fe56": "tendl-2025"}
    assert yamc.lookup_cross_section_data("Fe56") == "tendl-2025"

    yamc.cross_section_data = None
    yamc.set_cross_section_data_entry("Fe56", "fendl-3.2d")
    assert yamc.lookup_cross_section_data("Fe56") == "fendl-3.2d"

def test_file_validation_single_nuclide():
    """Test file validation in set_cross_section_data_entry with single nuclide"""
    # Valid file should work
    yamc.set_cross_section_data_entry("Li6", "tests/Li6.arrow")
    assert yamc.lookup_cross_section_data("Li6") == "tests/Li6.arrow"

    # Invalid file should panic
    with pytest.raises(BaseException) as exc_info:
        yamc.set_cross_section_data_entry("Fe56", "/nonexistent/file.arrow")
    assert "does not exist" in str(exc_info.value)

def test_set_cross_sections_with_directory():
    """Test setting cross sections with a directory path"""
    yamc.cross_section_data = "tests"

    # Any nuclide should now return the directory
    assert yamc.lookup_cross_section_data("Li6") == "tests"
    assert yamc.lookup_cross_section_data("Fe56") == "tests"

def test_set_cross_sections_with_directory_in_dict():
    """Test setting a directory path as a dict value"""
    yamc.cross_section_data = {"Li6": "tests"}
    assert yamc.lookup_cross_section_data("Li6") == "tests"

def test_directory_config_loads_nuclide():
    """Test that directory-based config correctly loads nuclide data"""
    yamc.cross_section_data = "tests"

    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data()

    assert nuc.name == "Li6"
    assert len(nuc.available_temperatures) > 0

def test_directory_config_loads_material():
    """Test that directory-based config correctly loads material data"""
    yamc.cross_section_data = "tests"

    mat = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=1.0,
        temperature=294,
    )
    mat.read_nuclear_data()

    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0
    assert len(energy) > 0


# ---------------------------------------------------------------------------
# Combination tests: Config source × loading method
# Config sources: file path, directory, keyword
# Loading methods:
#   Nuclide.read_nuclear_data(path)   -- explicit path/keyword/directory
#   Nuclide.read_nuclear_data()       -- auto from cross_section_data
#   Nuclide auto-load via microscopic_cross_section  -- auto from cross_section_data
#   Material.read_nuclear_data()      -- auto from cross_section_data
#   Material.read_nuclear_data({...}) -- explicit dict
# ---------------------------------------------------------------------------

def _assert_nuclide_loaded(nuc):
    """Helper: verify a nuclide has data."""
    assert nuc.name is not None
    assert len(nuc.available_temperatures) > 0


# --- Nuclide.read_nuclear_data(explicit) ---

def test_nuclide_explicit_file_path():
    """Nuclide.read_nuclear_data with explicit file path (no Config)."""
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data("tests/Li6.arrow")
    _assert_nuclide_loaded(nuc)

def test_nuclide_explicit_directory():
    """Nuclide.read_nuclear_data with explicit directory path (no Config)."""
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data("tests")
    _assert_nuclide_loaded(nuc)

@requires_keywords
def test_nuclide_explicit_keyword():
    """Nuclide.read_nuclear_data with explicit keyword (no Config)."""
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data("endf-b8.1")
    _assert_nuclide_loaded(nuc)


# --- Nuclide.read_nuclear_data() via cross_section_data ---

def test_nuclide_config_file_path_auto():
    """cross_section_data has file path → Nuclide.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = {"Li6": "tests/Li6.arrow"}
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data()
    _assert_nuclide_loaded(nuc)

def test_nuclide_config_directory_auto():
    """cross_section_data has directory → Nuclide.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = "tests"
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data()
    _assert_nuclide_loaded(nuc)

@requires_keywords
def test_nuclide_config_keyword_auto():
    """cross_section_data has keyword → Nuclide.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = "endf-b8.1"
    nuc = yamc.Nuclide("Li6")
    nuc.read_nuclear_data()
    _assert_nuclide_loaded(nuc)


# --- Nuclide auto-load via microscopic_cross_section ---

def test_nuclide_autoload_config_file_path():
    """cross_section_data has file path → microscopic_cross_section triggers auto-load."""
    yamc.cross_section_data = {"Li6": "tests/Li6.arrow"}
    nuc = yamc.Nuclide("Li6")
    xs, energy = nuc.microscopic_cross_section(reaction=1, temperature='294')
    assert len(xs) > 0

def test_nuclide_autoload_config_directory():
    """cross_section_data has directory → microscopic_cross_section triggers auto-load."""
    yamc.cross_section_data = "tests"
    nuc = yamc.Nuclide("Li6")
    xs, energy = nuc.microscopic_cross_section(reaction=1, temperature='294')
    assert len(xs) > 0

@requires_keywords
def test_nuclide_autoload_config_keyword():
    """cross_section_data has keyword → microscopic_cross_section triggers auto-load."""
    yamc.cross_section_data = "endf-b8.1"
    nuc = yamc.Nuclide("Li6")
    xs, energy = nuc.microscopic_cross_section(reaction=1, temperature="294")
    assert len(xs) > 0


# --- Material.read_nuclear_data() via cross_section_data ---

def _make_li_material():
    mat = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=1.0,
        temperature=294,
    )
    return mat

def test_material_config_file_paths_auto():
    """cross_section_data has file paths → Material.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = {"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"}
    mat = _make_li_material()
    mat.read_nuclear_data()
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0

def test_material_config_directory_auto():
    """cross_section_data has directory → Material.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = "tests"
    mat = _make_li_material()
    mat.read_nuclear_data()
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0

@requires_keywords
def test_material_config_keyword_auto():
    """cross_section_data has keyword → Material.read_nuclear_data() auto-resolves."""
    yamc.cross_section_data = "endf-b8.1"
    mat = _make_li_material()
    mat.read_nuclear_data()
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0


# --- Material.read_nuclear_data(explicit dict) ---

def test_material_explicit_file_paths():
    """Material.read_nuclear_data with explicit file path dict (no Config)."""
    mat = _make_li_material()
    mat.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0

def test_material_explicit_directory():
    """Material.read_nuclear_data with explicit directory string (no Config)."""
    mat = _make_li_material()
    mat.read_nuclear_data("tests")
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0

@requires_keywords
def test_material_explicit_keyword():
    """Material.read_nuclear_data with explicit keyword string (no Config)."""
    mat = _make_li_material()
    mat.read_nuclear_data("endf-b8.1")
    xs, energy = mat.macroscopic_cross_section(reaction=1)
    assert len(xs) > 0
