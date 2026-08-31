"""The reactions and fission_yields settings have three states, so they need
three spellings.

Both take a library keyword or a path, and both can be turned off for a
calculation that genuinely has no reaction rates or no fissioning nuclide.
That is one state more than ``Optional[str]`` can carry, and for a while
``None`` was spelling "off" -- so clearing the setting, the obvious way to put
it back, silently dropped the subsection and the solve returned no activation
products at all. ``None`` resets; ``False`` turns off.
"""

import pytest

import yamc

_DEFAULT = "endf-b8.1"


@pytest.fixture(params=["transmutation_reactions", "transmutation_fission_yields"])
def setting(request):
    """Each three-state setting, restored afterwards."""
    yield request.param
    setattr(yamc, request.param, None)


def test_none_restores_the_default(setting):
    setattr(yamc, setting, "jeff-4.0")
    assert getattr(yamc, setting) == "jeff-4.0"

    setattr(yamc, setting, None)
    assert getattr(yamc, setting) == _DEFAULT


def test_false_turns_the_subsection_off(setting):
    setattr(yamc, setting, False)
    assert getattr(yamc, setting) is None


def test_off_is_not_a_one_way_door(setting):
    """Turning it off and clearing it gets the default back, not a chain that
    stays permanently short a subsection."""
    setattr(yamc, setting, False)
    setattr(yamc, setting, None)
    assert getattr(yamc, setting) == _DEFAULT


def test_a_path_is_kept_verbatim(setting, tmp_path):
    setattr(yamc, setting, str(tmp_path))
    assert getattr(yamc, setting) == str(tmp_path)


def test_true_is_refused_by_name(setting):
    """``True`` is the one bool with no meaning here; the error says what the
    three spellings are rather than being taken as a source named 'true'."""
    with pytest.raises(ValueError, match="True is not a source"):
        setattr(yamc, setting, True)


def test_decay_data_still_takes_none_as_the_default():
    """The two-state siblings are unchanged, and agree with the three-state
    ones about what ``None`` means."""
    yamc.transmutation_decay_data = "jeff-4.0"
    assert yamc.transmutation_decay_data == "jeff-4.0"
    yamc.transmutation_decay_data = None
    assert yamc.transmutation_decay_data is None
