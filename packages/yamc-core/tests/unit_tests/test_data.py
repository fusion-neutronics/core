import pytest

from yamc.data import (atomic_number, atomic_symbol, element_nuclides,
                       natural_abundance, reaction_names)

def test_lithium_natural_abundance():
    abund = natural_abundance()
    li6 = float(abund['Li6'])
    li7 = float(abund['Li7'])
    assert abs(li6 - 0.0759) < 1e-4, f"Li6 abundance incorrect: {li6}"
    assert abs(li7 - 0.9241) < 1e-4, f"Li7 abundance incorrect: {li7}"
    assert abs(li6 + li7 - 1.0) < 1e-3, f"Li6 + Li7 should sum to 1, got {li6 + li7}"

def test_element_nuclides_li_and_be():
    nuclides = element_nuclides()
    assert sorted(nuclides['Li']) == ['Li6', 'Li7']
    assert sorted(nuclides['Be']) == ['Be9']

def test_reaction_names_returns_dict():
    names = reaction_names()
    assert isinstance(names, dict)
    assert len(names) > 50, f"Expected >50 MT entries, got {len(names)}"

def test_reaction_names_key_entries():
    names = reaction_names()
    # Keys are MT numbers (int), values are ENDF notation strings.
    # MT 3, 18 have known short-name aliases - accept either.
    assert names[2]   == '(n,elastic)'
    assert names[18]  in ('(n,fission)', 'fission')
    assert names[102] == '(n,gamma)'
    assert names[3]   in ('(n,nonelastic)', 'nonelastic')
    assert names[103] == '(n,p)'
    assert names[107] == '(n,a)'
    assert names[16]  == '(n,2n)'
    assert names[203] == '(n,Xp)'
    assert names[207] == '(n,Xa)'
    assert names[301] == 'heating'

def test_reaction_names_keys_are_ints():
    names = reaction_names()
    for k in names:
        assert isinstance(k, int), f"Expected int key, got {type(k)}: {k}"

def test_reaction_names_values_are_strings():
    names = reaction_names()
    for v in names.values():
        assert isinstance(v, str), f"Expected str value, got {type(v)}: {v}"


def test_atomic_symbol_covers_the_whole_table():
    assert atomic_symbol(26) == 'Fe'
    assert atomic_symbol(1) == 'H'
    # The last element the table carries, so the upper bound is exercised
    # rather than assumed.
    assert atomic_symbol(118) == 'Og'

def test_atomic_symbol_of_zero_is_the_neutron():
    """Z = 0 is a real ENDF material, not an error.

    JENDL ships a neutron sublibrary entry, so anything walking a library will
    meet Z = 0 and must not have it raise.
    """
    assert atomic_symbol(0) == 'n'

def test_atomic_symbol_refuses_an_element_that_does_not_exist():
    with pytest.raises(ValueError, match='119'):
        atomic_symbol(119)

def test_atomic_number_inverts_atomic_symbol():
    # Every Z the table holds, not a sample: the two are each other's inverse
    # or neither can be trusted to name a file.
    for z in range(119):
        assert atomic_number(atomic_symbol(z)) == z

def test_atomic_number_is_the_symbol_only_and_case_sensitive():
    """Refused rather than guessed at, because a wrong Z silently misnames data."""
    for wrong in ('fe', 'FE', 'iron', 'Xx', ''):
        with pytest.raises(ValueError):
            atomic_number(wrong)
