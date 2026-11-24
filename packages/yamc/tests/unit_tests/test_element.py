#!/usr/bin/env python3
"""
Tests for the element module Python bindings.
"""
import pytest

import yamc

def test_element_class():
    """Test the Element class functionality."""
    # Create elements with different symbols
    h = yamc.Element("H")
    fe = yamc.Element("Fe")
    u = yamc.Element("U")
    
    # Check that the element names are correctly stored
    assert h.name == "H"
    assert fe.name == "Fe"
    assert u.name == "U"
    
    # Test getting nuclides for each element
    h_nuclides = h.get_nuclides()
    fe_nuclides = fe.get_nuclides()
    u_nuclides = u.get_nuclides()
    
    # Hydrogen should have at least H1 and H2 (protium and deuterium)
    assert len(h_nuclides) >= 2
    assert "H1" in h_nuclides
    assert "H2" in h_nuclides
    
    # Iron should have its natural isotopes
    assert len(fe_nuclides) >= 4  # Fe54, Fe56, Fe57, Fe58
    assert "Fe54" in fe_nuclides
    assert "Fe56" in fe_nuclides
    assert "Fe57" in fe_nuclides
    assert "Fe58" in fe_nuclides
    
    # Uranium should have at least U235 and U238
    assert len(u_nuclides) >= 2
    assert "U235" in u_nuclides
    assert "U238" in u_nuclides

def test_get_nuclides():
    """Test per-element get_nuclides method returns list for that element."""
    al_isotopes = yamc.Element('Al').get_nuclides()
    assert isinstance(al_isotopes, list)
    assert all(iso.startswith('Al') for iso in al_isotopes)
    assert len(al_isotopes) > 0

def test_nonexistent_element():
    """Test behavior with a non-existent element."""
    # Create an element with a symbol that doesn't exist
    fake_element = yamc.Element("Zz")
    
    # Should return an empty list of nuclides
    assert fake_element.get_nuclides() == []

if __name__ == "__main__":
    pytest.main(["-v", __file__])
