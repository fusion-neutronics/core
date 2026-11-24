"""The reactions and fission_yields settings have three states, so they need
three spellings.

Both take a library keyword or a path, and both can be turned off for a
calculation that genuinely has no reaction rates or no fissioning nuclide.
That is one state more than ``Optional[str]`` can carry, and for a while
``None`` was spelling "off" -- so clearing the setting, the obvious way to put
it back, silently dropped the subsection and the solve returned no activation
products at all. ``None`` unsets (there is no default library, so a
calculation that reads an unset one is refused); ``False`` turns off. Each
reads back as it was assigned.
"""

import pytest

import yamc


@pytest.fixture(params=["transmutation_reactions", "transmutation_fission_yields"])
def setting(request):
    """Each three-state setting, restored afterwards."""
    yield request.param
    setattr(yamc, request.param, None)


def test_none_unsets(setting):
    setattr(yamc, setting, "jeff-4.0")
    assert getattr(yamc, setting) == "jeff-4.0"

    setattr(yamc, setting, None)
    assert getattr(yamc, setting) is None


def test_false_turns_the_subsection_off(setting):
    setattr(yamc, setting, False)
    assert getattr(yamc, setting) is False


def test_off_is_not_a_one_way_door(setting):
    """Turning it off and clearing it unsets it, rather than leaving a chain
    permanently short a subsection."""
    setattr(yamc, setting, False)
    setattr(yamc, setting, None)
    assert getattr(yamc, setting) is None


def test_each_state_reads_back_as_assigned(setting, tmp_path):
    """Reading a setting and assigning the result again changes nothing, which
    is what lets a caller save and restore it."""
    for value in ("endf-b8.1", str(tmp_path), False, None):
        setattr(yamc, setting, value)
        setattr(yamc, setting, getattr(yamc, setting))
        assert getattr(yamc, setting) == value
        assert type(getattr(yamc, setting)) is type(value)


def test_true_is_refused_by_name(setting):
    """``True`` is the one bool with no meaning here; the error says what the
    three spellings are rather than being taken as a source named 'true'."""
    with pytest.raises(ValueError, match="True is not a source"):
        setattr(yamc, setting, True)


def test_decay_data_takes_none_as_unset():
    """The two-state siblings agree with the three-state ones about what
    ``None`` means."""
    yamc.transmutation_decay_data = "jeff-4.0"
    assert yamc.transmutation_decay_data == "jeff-4.0"
    yamc.transmutation_decay_data = None
    assert yamc.transmutation_decay_data is None
