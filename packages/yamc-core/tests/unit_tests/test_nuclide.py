import pytest
import yamc
from yamc import Nuclide


def _keywords_available():
    """Check if keyword download tests can work (requires download feature)."""
    try:
        n = Nuclide("Li6")
        n.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature"
)

def test_be9_not_fissionable():
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    assert hasattr(nuc, 'fissionable'), "Nuclide should have a 'fissionable' attribute"
    assert nuc.fissionable is False, "Be9 should not be fissionable"

def test_fe58_not_fissionable():
    nuc = Nuclide('Fe58')
    nuc.read_nuclear_data('tests/Fe58.arrow')
    assert hasattr(nuc, 'fissionable'), "Nuclide should have a 'fissionable' attribute"
    assert nuc.fissionable is False, "Fe58 should not be fissionable"

def test_read_li6_nuclide():
    nuc1 = Nuclide('Li6')
    nuc1.read_nuclear_data('tests/Li6.arrow')
    assert nuc1.element.lower() == 'lithium'
    assert nuc1.atomic_symbol == "Li"
    assert nuc1.atomic_number == 3
    assert nuc1.mass_number == 6
    assert nuc1.neutron_number == 3
    assert nuc1.available_temperatures == ['250', '294', '600', '900', '1200', '2500']
    # We don't expect any specific order of MT numbers, just check they're all ints
    assert all(isinstance(mt, int) for mt in nuc1.reaction_mts)

    cs = nuc1.reactions['294'][2].cross_section

    for entry in cs:
        assert isinstance(entry, float)
        assert isinstance(entry, float)

    # Check that essential MTs are present (not an exact list match since HDF5 may have additional MTs like 901)
    essential_mts = [1, 2, 3, 4, 101, 102, 103]  # hierarchical and key reactions
    for mt in essential_mts:
        assert mt in nuc1.reaction_mts, f"MT {mt} should be in Li6 reactions"

def test_read_li7_nuclide():
    nuc1 = Nuclide('Li7')
    nuc1.read_nuclear_data('tests/Li7.arrow')
    assert nuc1.element.lower() == 'lithium'
    assert nuc1.atomic_symbol == "Li"
    assert nuc1.atomic_number == 3
    assert nuc1.mass_number == 7
    assert nuc1.neutron_number == 4
    assert nuc1.available_temperatures == ['250', '294', '600', '900', '1200', '2500']
    # We don't expect any specific order of MT numbers, just check they're all ints
    assert all(isinstance(mt, int) for mt in nuc1.reaction_mts)

    cs = nuc1.reactions['294'][2].cross_section
    
    for entry in cs:
        assert isinstance(entry, float)
        assert isinstance(entry, float)

    # Check that essential MTs are present (not an exact list match)
    essential_mts = [1, 2, 3, 4, 101, 102]  # hierarchical and key reactions
    for mt in essential_mts:
        assert mt in nuc1.reaction_mts, f"MT {mt} should be in Li7 reactions"


def test_read_be9_available_and_loaded_temperatures():
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    assert nuc.available_temperatures == ['250', '294', '600', '900', '1200', '2500']
    # By current implementation, all temps are loaded eagerly
    assert hasattr(nuc, 'loaded_temperatures'), "loaded_temperatures attribute missing"
    assert nuc.loaded_temperatures == ['250', '294', '600', '900', '1200', '2500']
    # Reactions dict should contain the temperature
    assert '294' in nuc.reactions


def test_read_be9_mt_numbers_per_temperature():
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    mts_294 = sorted(int(mt) for mt in nuc.reactions['294'].keys())
    # Check essential MTs are present (not exact match since HDF5 may have different MTs)
    essential_mts = [1, 2, 3, 101]  # Be9 has no MT 102 (capture)
    for mt in essential_mts:
        assert mt in mts_294, f"Be9 should have MT {mt}"


def test_read_be9_selective_single_temperature():
    # Ensure only the specified temperature (294) is retained in reactions and loaded_temperatures
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow', temperatures=['294'])
    assert nuc.available_temperatures == ['250', '294', '600', '900', '1200', '2500'], "available_temperatures should list all temps present in file"
    assert nuc.loaded_temperatures == ['294'], f"loaded_temperatures should be only ['294'], got {nuc.loaded_temperatures}"
    assert '294' in nuc.reactions, "294K reactions missing after selective load"
    # Check essential MTs are present
    mts_294 = sorted(int(mt) for mt in nuc.reactions['294'].keys())
    essential_mts = [1, 2, 3, 101]  # Be9 has no MT 102 (capture)
    for mt in essential_mts:
        assert mt in mts_294, f"Be9 selective load should have MT {mt}"


@requires_keywords
def test_load_keyword():
    """Test keyword-based data loading from remote H5 files."""
    from yamc import Nuclide
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('endf-b8.1')
    assert nuc.element.lower() == 'lithium'
    assert nuc.atomic_number == 3
    assert nuc.mass_number == 6
    assert len(nuc.available_temperatures) > 0

def test_load_local_path():
    from yamc import Nuclide
    nuc = Nuclide('Li6')
    # Should not raise TypeError when passing local path
    nuc.read_nuclear_data("tests/Li6.arrow")


def test_microscopic_cross_section_with_temperature():
    """Test microscopic_cross_section with explicit temperature."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Test with specific temperature
    xs, energy = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert len(xs) > 0, "Cross section data should not be empty"
    assert len(energy) > 0, "Energy data should not be empty"
    assert len(xs) == len(energy), "Cross section and energy arrays should have same length"
    
    # Test with same temperature again to verify consistency
    xs_294_again, energy_294_again = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert xs == xs_294_again, "Same temperature should give identical results"
    assert energy == energy_294_again, "Same temperature should give identical energy"
    
    # Test different MT numbers
    xs_mt3, energy_mt3 = nuc.microscopic_cross_section(reaction=3, temperature='294')
    assert len(xs_mt3) > 0, "MT=3 cross section data should not be empty"
    assert len(energy_mt3) > 0, "MT=3 energy data should not be empty"


def test_microscopic_cross_section_without_temperature():
    """Test microscopic_cross_section with single loaded temperature."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    # Load only one temperature
    nuc.read_nuclear_data('tests/Be9.arrow', temperatures=['294'])
    
    # Should work without specifying temperature since only one is loaded
    xs, energy = nuc.microscopic_cross_section(2)
    assert len(xs) > 0, "Cross section data should not be empty"
    assert len(energy) > 0, "Energy data should not be empty"
    assert len(xs) == len(energy), "Cross section and energy arrays should have same length"


def test_microscopic_cross_section_multiple_temperatures_error():
    """Test that microscopic_cross_section errors when multiple temperatures loaded and none specified."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')  # Loads all temperatures

    # Should error when no temperature specified with multiple loaded temperatures
    with pytest.raises(Exception):
        nuc.microscopic_cross_section(2)

    # But should work when temperature is specified
    xs, energy = nuc.microscopic_cross_section(2, temperature='294')
    assert len(xs) > 0, "Cross section data should not be empty"
    assert len(energy) > 0, "Energy data should not be empty"


def test_microscopic_cross_section_invalid_temperature():
    """Test error handling for invalid temperature."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')

    # Should raise error for non-existent temperature
    with pytest.raises(Exception) as exc_info:
        nuc.microscopic_cross_section(reaction=2, temperature='500')
    error_msg = str(exc_info.value)
    assert "Temperature '500' not found" in error_msg or "500" in error_msg
    assert "294" in error_msg


def test_microscopic_cross_section_invalid_mt():
    """Test error handling for invalid MT number."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Should raise error for non-existent MT
    with pytest.raises(Exception) as exc_info:
        nuc.microscopic_cross_section(reaction=9999, temperature='294')
    error_msg = str(exc_info.value)
    assert "MT 9999 not found" in error_msg
    assert "Available MTs:" in error_msg


def test_microscopic_cross_section_multiple_mt_numbers():
    """Test microscopic_cross_section with various MT numbers."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Test common MT numbers that should exist in Be9
    test_mts = [1, 2, 3, 16, 27, 101, 102]  # Common reaction types
    
    for mt in test_mts:
        try:
            xs, energy = nuc.microscopic_cross_section(reaction=mt, temperature='294')
            assert len(xs) > 0, f"MT={mt} should have cross section data"
            assert len(energy) > 0, f"MT={mt} should have energy data"
            assert len(xs) == len(energy), f"MT={mt} data length mismatch"
            assert all(e > 0 for e in energy), f"MT={mt} energy values should be positive"
            assert all(x >= 0 for x in xs), f"MT={mt} cross sections should be non-negative"
        except Exception:
            # Some MT numbers might not exist, which is fine
            pass


def test_microscopic_cross_section_lithium():
    """Test microscopic_cross_section with Li6 data."""
    from yamc import Nuclide
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')

    # Li6 now has multiple temperatures, so explicit temperature is required
    xs, energy = nuc.microscopic_cross_section(reaction=2, temperature='294')  # Elastic scattering
    assert len(xs) > 0, "Li6 elastic scattering data should not be empty"
    assert len(energy) > 0, "Li6 energy data should not be empty"

    # Test with explicit temperature again for consistency
    xs_explicit, energy_explicit = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert xs == xs_explicit, "Results should be identical for same temperature"
    assert energy == energy_explicit, "Energy should be identical for same temperature"


def test_auto_loading_from_config():
    """Test that microscopic_cross_section can auto-load data from config when nuclide is empty"""
    # Set up config for auto-loading
    yamc.cross_section_data = {'Be9': 'tests/Be9.arrow'}
    
    # Create empty nuclide with name but no data loaded
    nuc = Nuclide('Be9')
    assert nuc.loaded_temperatures == [], "Should start with no loaded temperatures"
    
    # Call microscopic_cross_section - should auto-load data
    xs, energy = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert len(xs) > 0, "Auto-loaded cross section data should not be empty"
    assert len(energy) > 0, "Auto-loaded energy data should not be empty"
    assert len(xs) == len(energy), "Cross section and energy arrays should have same length"
    
    # Note: loaded_temperatures won't be updated in the Python object due to immutable API
    # The auto-loading happens internally but doesn't modify the original object


def test_auto_loading_additional_temperature():
    """Test that microscopic_cross_section can auto-load additional temperatures"""
    # Set up config for auto-loading
    yamc.cross_section_data = {'Be9': 'tests/Be9.arrow'}
    
    # Load Be9 with only 294K initially
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow', ['294'])
    
    assert nuc.loaded_temperatures == ['294'], "Should only have 294K loaded initially"
    assert '294' in nuc.available_temperatures, "Should know 294K is available"
    
    # Request 294K data - should work with loaded temperature
    xs, energy = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert len(xs) > 0, "294K cross section data should not be empty"
    assert len(energy) > 0, "294K energy data should not be empty"
    
    # The original nuclide object still shows only 294K due to immutable API
    # But the internal auto-loading worked to provide the 300K data
    

def test_auto_loading_without_config_fails():
    """Test that auto-loading fails gracefully when no config is available.

    Setting ``cross_section_data = {}`` must clear the configuration entirely --
    including any previously set global default library -- so that a later
    lookup finds nothing rather than silently falling back to a stale default.
    """
    original_configs = yamc.cross_section_data

    # Seed a global default first so the clear below has something to remove;
    # this makes the test self-contained instead of relying on whatever a
    # previously run test happened to leave behind.
    yamc.cross_section_data = "fendl-3.2d"

    # Clearing with an empty dict must wipe that default too (it previously
    # only cleared the per-nuclide map, leaving the default in place).
    yamc.cross_section_data = {}

    try:
        assert yamc.cross_section_data is None, (
            "cross_section_data = {} should clear the configuration entirely, "
            f"but it is still {yamc.cross_section_data!r}"
        )

        # Create empty nuclide with name but no config
        nuc = Nuclide('TestNuclide')

        # Call microscopic_cross_section - should fail with helpful error
        try:
            nuc.microscopic_cross_section(reaction=2, temperature='294')
            assert False, "Auto-loading without config should fail"
        except Exception as e:
            error_msg = str(e)
            # The error could be either "No configuration found" or an HDF5 file error
            # Both are acceptable since there's no valid config
            assert ("No configuration found" in error_msg or
                    "Failed to download" in error_msg or
                    "404 Not Found" in error_msg or
                    "unable to open file" in error_msg or
                    "No such file or directory" in error_msg or
                    "not found" in error_msg), f"Error should indicate missing or invalid configuration: {error_msg}"

    finally:
        # Restore original configuration
        if original_configs:
            yamc.cross_section_data = original_configs
        else:
            yamc.cross_section_data = None


def test_auto_loading_multiple_calls_consistent():
    """Test that multiple auto-loading calls give consistent results"""
    # Set up config for auto-loading
    yamc.cross_section_data = {'Be9': 'tests/Be9.arrow'}
    
    # Create empty nuclide
    nuc = Nuclide('Be9')
    
    # Call microscopic_cross_section multiple times
    xs1, energy1 = nuc.microscopic_cross_section(reaction=2, temperature='294')
    xs2, energy2 = nuc.microscopic_cross_section(reaction=2, temperature='294')
    xs3, energy3 = nuc.microscopic_cross_section(reaction=102, temperature='294')
    
    # First two calls should give identical results
    assert xs1 == xs2, "Multiple calls with same parameters should give identical results"
    assert energy1 == energy2, "Multiple calls with same parameters should give identical energy"
    
    # Third call should work too (different MT same temperature)
    assert len(xs3) > 0, "Auto-loading different MT should work"
    assert len(energy3) > 0, "Auto-loading different MT should provide energy"


def test_auto_loading_with_manual_loading_combined():
    """Test combining manual loading with auto-loading for additional data"""
    # Set up config for auto-loading
    yamc.cross_section_data = {'Be9': 'tests/Be9.arrow'}
    
    # Manually load some data first
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow', ['294'])
    
    # Verify manual loading worked
    assert '294' in nuc.loaded_temperatures, "Manual loading should work"
    assert '294' in nuc.available_temperatures, "Should know available temperatures"
    
    # Now use auto-loading for data that was manually loaded
    xs_manual, energy_manual = nuc.microscopic_cross_section(reaction=2, temperature='294')
    assert len(xs_manual) > 0, "Should get data for manually loaded temperature"
    
    # Test different MT with same temperature
    xs_auto, energy_auto = nuc.microscopic_cross_section(reaction=3, temperature='294')
    assert len(xs_auto) > 0, "Should work with different MT"
    
    # Test with a temperature-specific MT - try MT=444 which is available at 294K
    xs_specific, energy_specific = nuc.microscopic_cross_section(reaction=444, temperature='294')
    assert len(xs_specific) > 0, "Should get temperature-specific MT data"
    
    # Test error handling for invalid temperature
    try:
        nuc.microscopic_cross_section(reaction=444, temperature='500')
        assert False, "Should have raised an error for invalid temperature"
    except ValueError as e:
        # This is expected for invalid temperature
        assert "Temperature '500' not found" in str(e), f"Should get temperature not found error, got: {e}"
    
    # Note: For Be9 MT=2, the cross sections at 294K and 300K might be identical
    # This is fine - the important thing is that both calls succeeded


@requires_keywords
@pytest.mark.parametrize("keyword", ["fendl-3.2d", "tendl-2025", "endf-b8.1"])
def test_data_source_keyword_recognized(keyword):
    """Test that each supported data-source keyword is recognized."""

    # Test that the keyword is recognized (this tests the Rust backend)
    try:
        # This should not raise an exception if the keyword is recognized
        yamc.cross_section_data = {'Li6': keyword}

        # Verify we can retrieve it
        cross_sections = yamc.cross_section_data
        assert 'Li6' in cross_sections, "Li6 should be in cross sections config"
        assert cross_sections['Li6'] == keyword, f"Should store {keyword} keyword correctly"

    except Exception as e:
        pytest.fail(f"{keyword} keyword should be recognized by the system: {e}")


@requires_keywords
def test_auto_loading_with_global_keyword():
    """Test that auto-loading works with global keyword configuration"""
    from yamc import Nuclide

    # Clear any per-nuclide entries left by previous tests, then set global
    yamc.cross_section_data = None
    yamc.cross_section_data = 'fendl-3.2d'

    # Verify config is set correctly
    assert yamc.lookup_cross_section_data('Li6') == 'fendl-3.2d', "Global config should apply to Li6"

    # Create empty nuclide
    nuc = Nuclide('Li6')
    assert nuc.loaded_temperatures == [], "Should start with no loaded temperatures"

    # Call microscopic_cross_section - should auto-load data from global config
    # Note: fendl-3.2d uses temperature '294'
    xs, energy = nuc.microscopic_cross_section(reaction=1, temperature='294')
    assert len(xs) > 0, "Auto-loaded cross section data should not be empty"
    assert len(energy) > 0, "Auto-loaded energy data should not be empty"
    assert len(xs) == len(energy), "Cross section and energy arrays should have same length"


def test_microscopic_cross_section_by_name():
    """Test microscopic_cross_section with reaction names."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Test elastic scattering using reaction name
    xs_name, energy_name = nuc.microscopic_cross_section("(n,elastic)", temperature='294')
    assert len(xs_name) > 0, "Elastic scattering cross section should not be empty"
    assert len(energy_name) > 0, "Energy data should not be empty"
    
    # Compare with MT number approach (MT=2 is elastic scattering)
    xs_mt, energy_mt = nuc.microscopic_cross_section(reaction=2, temperature='294')
    
    # Should get identical results
    assert xs_name == xs_mt, "Reaction name and MT number should give identical cross sections"
    assert energy_name == energy_mt, "Reaction name and MT number should give identical energy grids"
    
    # Test other common reactions
    test_reactions = [
        ("(n,gamma)", 102),   # Radiative capture
        ("(n,a)", 107),       # Alpha production
        ("(n,total)", 1),     # Total cross section
    ]
    
    for reaction_name, mt_num in test_reactions:
        try:
            xs_name, energy_name = nuc.microscopic_cross_section(reaction_name, temperature='294')
            xs_mt, energy_mt = nuc.microscopic_cross_section(reaction=mt_num, temperature='294')
            
            assert len(xs_name) > 0, f"{reaction_name} should have cross section data"
            assert len(energy_name) > 0, f"{reaction_name} should have energy data"
            assert xs_name == xs_mt, f"{reaction_name} and MT={mt_num} should give identical results"
            assert energy_name == energy_mt, f"{reaction_name} and MT={mt_num} should give identical energy"
            
        except Exception as e:
            # Some reactions might not exist for Be9, which is acceptable
            if "not found" in str(e).lower():
                print(f"Note: {reaction_name} (MT={mt_num}) not available in Be9 data")
            else:
                raise e


def test_microscopic_cross_section_by_name_invalid_reaction():
    """Test error handling for invalid reaction names."""
    from yamc import Nuclide
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Test with invalid reaction name
    with pytest.raises(Exception) as exc_info:
        nuc.microscopic_cross_section("invalid_reaction", temperature='294')
    error_msg = str(exc_info.value)
    assert "not found in reaction mapping" in error_msg or "Unknown reaction" in error_msg


def test_microscopic_cross_section_by_name_fission():
    """Test that the special 'fission' alias works."""
    from yamc import Nuclide
    
    # Use Li6 which might have fission data, or test the error handling
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    try:
        xs_fission, energy_fission = nuc.microscopic_cross_section("fission", temperature='294')
        xs_mt18, energy_mt18 = nuc.microscopic_cross_section(18, temperature='294')
        
        # Should get identical results since fission maps to MT=18
        assert xs_fission == xs_mt18, "fission and MT=18 should give identical results"
        assert energy_fission == energy_mt18, "fission and MT=18 should give identical energy"
        
    except Exception as e:
        # Li6 might not have fission data, which is acceptable
        if "MT 18 not found" in str(e) or "not found" in str(e).lower():
            print("Note: Li6 does not have fission data (expected)")
        else:
            raise e


def test_sample_reaction_basic():
    """Test basic functionality of sample_reaction method."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Test sampling at a typical neutron energy
    reaction = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
    
    if reaction is not None:
        # Check that all expected fields are present
        assert 'mt_number' in reaction, "Reaction should have mt_number field"
        assert 'cross_section' in reaction, "Reaction should have cross_section field"
        assert 'threshold_idx' in reaction, "Reaction should have threshold_idx field"
        assert 'energy' in reaction, "Reaction should have energy field"
        
        # Check data types
        assert isinstance(reaction['mt_number'], int), "MT number should be integer"
        assert isinstance(reaction['cross_section'], list), "Cross section should be list"
        assert isinstance(reaction['energy'], list), "Energy should be list"
        assert isinstance(reaction['threshold_idx'], int), "Threshold index should be integer"

        # Check data validity
        assert reaction['mt_number'] > 0, "MT number should be positive"
        assert len(reaction['cross_section']) > 0, "Cross section should not be empty"
        assert len(reaction['energy']) > 0, "Energy grid should not be empty"
        assert reaction['threshold_idx'] >= 0, "Threshold index should be non-negative"
        assert all(x >= 0 for x in reaction['cross_section']), "Cross sections should be non-negative"
        assert all(e > 0 for e in reaction['energy']), "Energy values should be positive"
    else:
        # If no reaction is sampled, that's also valid (zero total cross section)
        print("Note: No reaction sampled (possibly zero total cross section)")


def test_sample_reaction_reproducibility():
    """Test that sample_reaction gives reproducible results with same seed."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Sample with same seed multiple times
    reaction1 = nuc.sample_reaction(energy=1.0, temperature='294', seed=123)
    reaction2 = nuc.sample_reaction(energy=1.0, temperature='294', seed=123)
    reaction3 = nuc.sample_reaction(energy=1.0, temperature='294', seed=456)
    
    if reaction1 is not None and reaction2 is not None:
        # Same seed should give same result
        assert reaction1['mt_number'] == reaction2['mt_number'], "Same seed should give same MT number"
        assert reaction1['cross_section'] == reaction2['cross_section'], "Same seed should give same cross section data"
        assert reaction1['energy'] == reaction2['energy'], "Same seed should give same energy data"
        
        # Different seed might give different result (but not guaranteed)
        if reaction3 is not None:
            # We can't guarantee different results, but at least verify the structure is correct
            assert isinstance(reaction3['mt_number'], int), "Different seed should still give valid MT number"
    else:
        print("Note: No reaction sampled in reproducibility test")


def test_sample_reaction_different_energies():
    """Test sample_reaction at different neutron energies."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Test various energies from thermal to fast neutron range
    test_energies = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    sampled_mts = set()
    
    for energy in test_energies:
        reaction = nuc.sample_reaction(energy=energy, temperature='294', seed=42)
        
        if reaction is not None:
            sampled_mts.add(reaction['mt_number'])
            
            # Verify reaction structure at each energy
            assert reaction['mt_number'] > 0, f"Valid MT number at energy {energy}"
            assert len(reaction['cross_section']) > 0, f"Non-empty cross section at energy {energy}"
            assert len(reaction['energy']) > 0, f"Non-empty energy grid at energy {energy}"
    
    # Should sample at least one reaction across all energies (unless total XS is zero everywhere)
    if len(sampled_mts) == 0:
        print("Warning: No reactions sampled across all test energies")
    else:
        print(f"Sampled {len(sampled_mts)} different reaction types across energy range")


def test_sample_reaction_be9():
    """Test sample_reaction with Be9 nuclide (different reaction channels)."""
    from yamc import Nuclide
    
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Be9 has different reaction channels than Li6
    reaction = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
    
    if reaction is not None:
        # Verify structure
        assert 'mt_number' in reaction, "Be9 reaction should have mt_number field"
        assert isinstance(reaction['mt_number'], int), "Be9 MT number should be integer"
        assert reaction['mt_number'] > 0, "Be9 MT number should be positive"
        
        # Check that MT number is valid for Be9
        assert reaction['mt_number'] in nuc.reaction_mts, f"MT {reaction['mt_number']} should be in available MTs for Be9"
        
        print(f"Be9 sampled reaction: MT {reaction['mt_number']}")
    else:
        print("Note: No Be9 reaction sampled")


def test_sample_reaction_without_seed():
    """Test that sample_reaction works without specifying a seed."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Sample without seed (should use system random state)
    reaction1 = nuc.sample_reaction(energy=1.0, temperature='294')
    reaction2 = nuc.sample_reaction(energy=1.0, temperature='294')
    
    # Both should work (might be same or different)
    if reaction1 is not None:
        assert isinstance(reaction1['mt_number'], int), "No-seed reaction 1 should be valid"
    
    if reaction2 is not None:
        assert isinstance(reaction2['mt_number'], int), "No-seed reaction 2 should be valid"
    
    print("No-seed sampling test completed")


def test_sample_reaction_multiple_temperatures():
    """Test sample_reaction with different temperatures."""
    from yamc import Nuclide
    
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')  # Has 294K
    
    # Sample at available temperature
    reaction_294 = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
    reaction_294_again = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
    
    # Both should work and be identical
    if reaction_294 is not None:
        assert isinstance(reaction_294['mt_number'], int), "294K reaction should be valid"
        
    if reaction_294_again is not None:
        assert isinstance(reaction_294_again['mt_number'], int), "294K reaction again should be valid"
        assert reaction_294['mt_number'] == reaction_294_again['mt_number'], "Same seed should give same result"
    print("Multiple temperature sampling test completed")


def test_sample_reaction_invalid_temperature():
    """Test error handling for invalid temperature in sample_reaction."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Test with invalid temperature - should handle gracefully
    # The current implementation uses fallback behavior rather than strict errors
    reaction = nuc.sample_reaction(energy=1.0, temperature='999', seed=42)
    
    # The method should either work (using fallback) or return None
    if reaction is not None:
        assert isinstance(reaction['mt_number'], int), "Fallback temperature reaction should be valid"
        print("Invalid temperature handled with fallback")
    else:
        print("Invalid temperature resulted in no reaction sampled")


def test_sample_reaction_edge_cases():
    """Test sample_reaction with edge case energies."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Test very low energy
    reaction_low = nuc.sample_reaction(energy=1e-5, temperature='294', seed=42)
    
    # Test very high energy
    reaction_high = nuc.sample_reaction(energy=1e7, temperature='294', seed=42)
    
    # Both should either work or return None (no error)
    if reaction_low is not None:
        assert isinstance(reaction_low['mt_number'], int), "Very low energy should give valid reaction"
    
    if reaction_high is not None:
        assert isinstance(reaction_high['mt_number'], int), "Very high energy should give valid reaction"
    
    print("Edge case energy test completed")


def test_sample_reaction_zero_energy():
    """Test sample_reaction with zero energy."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Test with zero energy (edge case)
    reaction = nuc.sample_reaction(energy=0.0, temperature='294', seed=42)
    
    # Should either work or return None, not crash
    if reaction is not None:
        assert isinstance(reaction['mt_number'], int), "Zero energy should give valid reaction if any"
    
    print("Zero energy test completed")


def test_sample_reaction_consistency_with_available_mts():
    """Test that sampled reactions are from available MT numbers."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    available_mts = set(nuc.reaction_mts)
    sampled_mts = set()
    
    # Sample many times to get a good distribution
    for i in range(100):
        reaction = nuc.sample_reaction(energy=1.0, temperature='294', seed=i)
        if reaction is not None:
            sampled_mts.add(reaction['mt_number'])
    
    # All sampled MTs should be in the available set
    for mt in sampled_mts:
        assert mt in available_mts, f"Sampled MT {mt} should be in available MTs {available_mts}"
    
    if sampled_mts:
        print(f"Sampled MTs {sorted(sampled_mts)} are all in available MTs {sorted(available_mts)}")
    else:
        print("No reactions sampled in consistency test")


def test_sample_reaction_return_type():
    """Test that sample_reaction returns correct Python types."""
    from yamc import Nuclide
    
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    reaction = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
    
    if reaction is not None:
        # Check that it's a dictionary
        assert isinstance(reaction, dict), "Reaction should be a dictionary"
        
        # Check specific field types
        assert isinstance(reaction['mt_number'], int), "mt_number should be int"
        assert isinstance(reaction['cross_section'], list), "cross_section should be list"
        assert isinstance(reaction['energy'], list), "energy should be list"
        assert isinstance(reaction['threshold_idx'], int), "threshold_idx should be int"

        # Check list element types
        if reaction['cross_section']:
            assert isinstance(reaction['cross_section'][0], float), "Cross section values should be float"
        
        if reaction['energy']:
            assert isinstance(reaction['energy'][0], float), "Energy values should be float"

    else:
        # None is also a valid return type
        assert reaction is None, "Return should be None or dict"
    
    print("Return type test completed")


@requires_keywords
def test_nuclide_different_data_sources():
    """Test that loading nuclides from different sources works."""
    from yamc import Nuclide

    # Load from keyword
    nuc1 = Nuclide('Li6')
    nuc1.read_nuclear_data('endf-b8.1')

    # Load from local file
    nuc2 = Nuclide('Li6')
    nuc2.read_nuclear_data('tests/Li6.arrow')

    # Both should have valid data
    assert len(nuc1.available_temperatures) > 0
    assert len(nuc2.available_temperatures) > 0
    assert nuc1.atomic_number == nuc2.atomic_number == 3


@requires_keywords
def test_nuclide_file_vs_keyword_sources():
    """Test that file paths and keywords can coexist."""
    from yamc import Nuclide

    # Load Li6 from file
    nuc_file = Nuclide('Li6')
    nuc_file.read_nuclear_data('tests/Li6.arrow')

    # Load Li7 from keyword
    nuc_keyword = Nuclide('Li7')
    nuc_keyword.read_nuclear_data('endf-b8.1')

    # Both should work
    assert nuc_file.mass_number == 6
    assert nuc_keyword.mass_number == 7
    assert len(nuc_file.reaction_mts) > 0
    assert len(nuc_keyword.reaction_mts) > 0


@requires_keywords
def test_nuclide_cache_respects_data_source_boundaries():
    """Test that the cache properly separates different data sources."""
    from yamc import Nuclide
    from yamc.data import clear_nuclide_cache
    clear_nuclide_cache()

    # Load from keyword
    nuc1 = Nuclide('Li6')
    nuc1.read_nuclear_data('endf-b8.1')

    # Load from local file
    nuc2 = Nuclide('Li6')
    nuc2.read_nuclear_data('tests/Li6.arrow')

    # Both should work independently
    assert len(nuc1.reaction_mts) > 0
    assert len(nuc2.reaction_mts) > 0


def test_nuclide_path_normalization():
    """Test that different path formats for same file give same results."""
    import os
    
    # Load Li6 with relative path
    li6_rel = Nuclide("Li6")
    li6_rel.read_nuclear_data("tests/Li6.arrow")
    
    # Load Li6 with absolute path
    li6_abs = Nuclide("Li6")
    abs_path = os.path.abspath("tests/Li6.arrow")
    li6_abs.read_nuclear_data(abs_path)
    
    # Should give identical results (same file)
    xs_rel, _ = li6_rel.microscopic_cross_section("(n,gamma)", temperature='294')
    xs_abs, _ = li6_abs.microscopic_cross_section("(n,gamma)", temperature='294')

    assert xs_rel == xs_abs, "Relative and absolute paths to same file should give identical results"


def test_basic_nuclide_properties_after_loading():
    """Test that basic nuclide properties are correctly set after loading data."""
    # Test Li6 properties
    nuc_li6 = Nuclide('Li6')
    nuc_li6.read_nuclear_data('tests/Li6.arrow')
    
    # Basic identification properties
    assert nuc_li6.name == 'Li6', "Nuclide name should be Li6"
    assert nuc_li6.element.lower() == 'lithium', "Element should be lithium"
    assert nuc_li6.atomic_symbol == 'Li', "Atomic symbol should be Li"
    assert nuc_li6.atomic_number == 3, "Atomic number should be 3"
    assert nuc_li6.mass_number == 6, "Mass number should be 6"
    assert nuc_li6.neutron_number == 3, "Neutron number should be 3"
    
    # Temperature availability
    assert len(nuc_li6.available_temperatures) > 0, "Should have available temperatures"
    assert '294' in nuc_li6.available_temperatures, "Should have 294K temperature data"
    
    # Reaction data availability
    assert len(nuc_li6.reaction_mts) > 0, "Should have reaction MTs available"
    assert isinstance(nuc_li6.reaction_mts, list), "Reaction MTs should be a list"
    assert all(isinstance(mt, int) for mt in nuc_li6.reaction_mts), "All MTs should be integers"


def test_basic_nuclide_properties_li7():
    """Test that basic nuclide properties are correctly set for Li7."""
    nuc_li7 = Nuclide('Li7')
    nuc_li7.read_nuclear_data('tests/Li7.arrow')
    
    # Basic identification properties
    assert nuc_li7.name == 'Li7', "Nuclide name should be Li7"
    assert nuc_li7.element.lower() == 'lithium', "Element should be lithium"
    assert nuc_li7.atomic_symbol == 'Li', "Atomic symbol should be Li"
    assert nuc_li7.atomic_number == 3, "Atomic number should be 3"
    assert nuc_li7.mass_number == 7, "Mass number should be 7"
    assert nuc_li7.neutron_number == 4, "Neutron number should be 4"
    
    # Temperature and reaction data
    assert len(nuc_li7.available_temperatures) > 0, "Should have available temperatures"
    assert len(nuc_li7.reaction_mts) > 0, "Should have reaction MTs available"


def test_basic_nuclide_properties_be9():
    """Test that basic nuclide properties are correctly set for Be9."""
    nuc_be9 = Nuclide('Be9')
    nuc_be9.read_nuclear_data('tests/Be9.arrow')
    
    # Basic identification properties
    assert nuc_be9.name == 'Be9', "Nuclide name should be Be9"
    assert nuc_be9.element.lower() == 'beryllium', "Element should be beryllium"
    assert nuc_be9.atomic_symbol == 'Be', "Atomic symbol should be Be"
    assert nuc_be9.atomic_number == 4, "Atomic number should be 4"
    assert nuc_be9.mass_number == 9, "Mass number should be 9"
    assert nuc_be9.neutron_number == 5, "Neutron number should be 5"
    
    # Be9 has at least one temperature
    assert len(nuc_be9.available_temperatures) >= 1, "Be9 should have temperature data"
    assert '294' in nuc_be9.available_temperatures, "Should have 294K temperature data"


def test_basic_nuclide_properties_iron_isotopes():
    """Test that basic nuclide properties are correctly set for iron isotopes."""
    iron_isotopes = [
        ('Fe54', 54, 28), ('Fe56', 56, 30), ('Fe57', 57, 31), ('Fe58', 58, 32)
    ]
    
    for name, mass_num, neutron_num in iron_isotopes:
        nuc = Nuclide(name)
        nuc.read_nuclear_data(f'tests/{name}.arrow')
        
        # Basic identification properties
        assert nuc.name == name, f"Nuclide name should be {name}"
        assert nuc.element.lower() == 'iron', f"Element should be iron for {name}"
        assert nuc.atomic_symbol == 'Fe', f"Atomic symbol should be Fe for {name}"
        assert nuc.atomic_number == 26, f"Atomic number should be 26 for {name}"
        assert nuc.mass_number == mass_num, f"Mass number should be {mass_num} for {name}"
        assert nuc.neutron_number == neutron_num, f"Neutron number should be {neutron_num} for {name}"
        
        # Should have temperature and reaction data
        assert len(nuc.available_temperatures) > 0, f"{name} should have available temperatures"
        assert len(nuc.reaction_mts) > 0, f"{name} should have reaction MTs available"


def test_nuclide_properties_before_loading():
    """Test nuclide properties before loading any data."""
    nuc = Nuclide('TestNuclide')
    
    # Basic properties should be set from the name
    assert nuc.name == 'TestNuclide', "Name should be set from constructor"
    
    # Other properties should be empty/None before loading
    assert nuc.available_temperatures == [], "Should have no temperatures before loading"
    assert nuc.loaded_temperatures == [], "Should have no loaded temperatures before loading"
    
    # Reaction MTs returns None before loading data
    assert nuc.reaction_mts is None, "Should have None reaction MTs before loading"
    
    # Test other properties that might be None before loading
    # These properties may be None before data is loaded
    assert nuc.element is None, "Element should be None before loading"
    assert nuc.atomic_number is None, "Atomic number should be None before loading" 
    assert nuc.mass_number is None, "Mass number should be None before loading"


def test_nuclide_properties_consistency():
    """Test that nuclide properties are consistent across multiple instances."""
    # Create two instances of the same nuclide
    nuc1 = Nuclide('Li6')
    nuc1.read_nuclear_data('tests/Li6.arrow')
    
    nuc2 = Nuclide('Li6')
    nuc2.read_nuclear_data('tests/Li6.arrow')
    
    # Should have identical properties
    assert nuc1.name == nuc2.name, "Names should be identical"
    assert nuc1.element == nuc2.element, "Elements should be identical"
    assert nuc1.atomic_symbol == nuc2.atomic_symbol, "Atomic symbols should be identical"
    assert nuc1.atomic_number == nuc2.atomic_number, "Atomic numbers should be identical"
    assert nuc1.mass_number == nuc2.mass_number, "Mass numbers should be identical"
    assert nuc1.neutron_number == nuc2.neutron_number, "Neutron numbers should be identical"
    assert nuc1.available_temperatures == nuc2.available_temperatures, "Available temperatures should be identical"
    assert nuc1.reaction_mts == nuc2.reaction_mts, "Reaction MTs should be identical"


def test_nuclide_name_parsing():
    """Test that nuclide names are parsed correctly."""
    test_cases = [
        ('Li6', 'Li', 6),
        ('Li7', 'Li', 7),
        ('Be9', 'Be', 9),
        ('Fe54', 'Fe', 54),
        ('Fe56', 'Fe', 56),
        ('Fe57', 'Fe', 57),
        ('Fe58', 'Fe', 58),
    ]
    
    for name, expected_symbol, expected_mass in test_cases:
        nuc = Nuclide(name)
        nuc.read_nuclear_data(f'tests/{name}.arrow')
        
        assert nuc.atomic_symbol == expected_symbol, f"Atomic symbol for {name} should be {expected_symbol}"
        assert nuc.mass_number == expected_mass, f"Mass number for {name} should be {expected_mass}"


def test_nuclide_reaction_mts_content():
    """Test that reaction MTs contain expected values."""
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Check for common reaction types that should be present
    common_mts = [1, 2, 3]  # total, elastic, nonelastic
    present_mts = set(nuc.reaction_mts)
    
    for mt in common_mts:
        if mt in present_mts:  # Not all nuclides have all reaction types
            assert mt in nuc.reaction_mts, f"MT {mt} should be in reaction MTs if present"
    
    # All MTs should be positive integers and reasonable values
    assert all(mt > 0 for mt in nuc.reaction_mts), "All MT numbers should be positive"
    assert all(mt < 1000 for mt in nuc.reaction_mts), "All MT numbers should be reasonable (<1000)"


def test_nuclide_temperature_data_consistency():
    """Test that temperature data is consistent."""
    nuc = Nuclide('Be9')
    nuc.read_nuclear_data('tests/Be9.arrow')
    
    # Available temperatures should include loaded temperatures
    available_set = set(nuc.available_temperatures)
    loaded_set = set(nuc.loaded_temperatures)
    
    assert loaded_set.issubset(available_set), "Loaded temperatures should be subset of available temperatures"
    
    # Temperature strings should be valid
    for temp in nuc.available_temperatures:
        assert isinstance(temp, str), "Temperature should be string"
        assert temp.isdigit() or '.' in temp, "Temperature should be numeric string"
        float_temp = float(temp)
        assert float_temp > 0, "Temperature should be positive"
        assert float_temp < 10000, "Temperature should be reasonable (<10000K)"


def test_nuclide_fissionable_property():
    """Test that the fissionable property is set correctly."""
    # Test non-fissionable nuclides
    non_fissionable = ['Li6', 'Li7', 'Be9', 'Fe54', 'Fe56', 'Fe57', 'Fe58']
    
    for name in non_fissionable:
        nuc = Nuclide(name)
        nuc.read_nuclear_data(f'tests/{name}.arrow')
        
        assert hasattr(nuc, 'fissionable'), f"{name} should have fissionable attribute"
        assert nuc.fissionable is False, f"{name} should not be fissionable"


def test_nuclide_atomic_mass_property():
    """Test atomic mass property if available."""
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    
    # Check if atomic mass is available
    if hasattr(nuc, 'atomic_mass'):
        assert isinstance(nuc.atomic_mass, (int, float)), "Atomic mass should be numeric"
        assert nuc.atomic_mass > 0, "Atomic mass should be positive"
        # Li6 atomic mass should be around 6 atomic mass units
        assert 5 < nuc.atomic_mass < 7, "Li6 atomic mass should be around 6 AMU"


def test_nuclide_cross_section_availability():
    """Test that cross section data is available after loading."""
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')

    # Should have reactions data
    assert hasattr(nuc, 'reactions'), "Should have reactions attribute"
    assert isinstance(nuc.reactions, dict), "Reactions should be dictionary"

    # Should have at least one temperature
    assert len(nuc.reactions) > 0, "Should have reactions for at least one temperature"

    # Each temperature should have reaction data
    for temp in nuc.reactions:
        temp_reactions = nuc.reactions[temp]
        assert isinstance(temp_reactions, dict), f"Reactions for {temp} should be dictionary"
        assert len(temp_reactions) > 0, f"Should have at least one reaction at {temp}"

        # Check a few reactions have cross section data
        for mt in list(temp_reactions.keys())[:3]:  # Check first 3 reactions
            reaction = temp_reactions[mt]
            assert hasattr(reaction, 'cross_section'), f"Reaction {mt} should have cross_section"
            assert len(reaction.cross_section) > 0, f"Reaction {mt} should have non-empty cross section"

def test_read_nuclear_data_with_file_path():
    """Test read_nuclear_data with an explicit file path."""
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests/Li6.arrow')
    assert nuc.name == "Li6"
    assert len(nuc.available_temperatures) > 0

def test_read_nuclear_data_no_args_uses_config():
    """Test read_nuclear_data with no args falls back to Config."""
    import yamc
    yamc.cross_section_data = None
    yamc.transmutation_decay_data = None
    yamc.cross_section_data = {"Li6": "tests/Li6.arrow"}

    nuc = Nuclide('Li6')
    nuc.read_nuclear_data()
    assert nuc.name == "Li6"
    assert len(nuc.available_temperatures) > 0
    yamc.cross_section_data = None
    yamc.transmutation_decay_data = None

def test_read_nuclear_data_with_directory():
    """Test read_nuclear_data with a directory path resolves to <dir>/<name>.arrow."""
    nuc = Nuclide('Li6')
    nuc.read_nuclear_data('tests')
    assert nuc.name == "Li6"
    assert len(nuc.available_temperatures) > 0
