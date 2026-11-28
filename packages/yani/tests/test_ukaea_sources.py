"""UKAEA's decay data and UKFY fission yields as keywords.

``ukdd-2012`` and ``ukdd-2020`` publish decay data and nothing else, and
``ukfy-4.2`` fission yields and nothing else. Each is taken by the one setting
that reads what it publishes and refused by every other, as it is assigned,
with a message naming the setting that does take it. Nothing is downloaded.
"""

import pytest
import yani

TAKES = {
    "ukdd-2012": "transmutation_decay_data",
    "ukdd-2020": "transmutation_decay_data",
    "ukfy-4.2": "transmutation_fission_yields",
}
SAYS = {
    "transmutation_decay_data": "supplies decay data only",
    "transmutation_fission_yields": "supplies fission yields only",
}
SUBSECTION_SETTINGS = [
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
]


@pytest.mark.parametrize("keyword", sorted(TAKES))
def test_each_is_taken_by_the_setting_that_reads_it(keyword):
    setting = TAKES[keyword]
    before = getattr(yani, setting)
    try:
        setattr(yani, setting, keyword)
        assert getattr(yani, setting) == keyword
    finally:
        setattr(yani, setting, before)


@pytest.mark.parametrize("keyword", sorted(TAKES))
@pytest.mark.parametrize("setting", SUBSECTION_SETTINGS)
def test_each_is_refused_by_every_other_setting(keyword, setting):
    if setting == TAKES[keyword]:
        pytest.skip("the setting that takes it")
    before = getattr(yani, setting)
    with pytest.raises(ValueError, match=SAYS[TAKES[keyword]]) as refused:
        setattr(yani, setting, keyword)
    assert TAKES[keyword] in str(refused.value)
    assert getattr(yani, setting) == before


@pytest.mark.parametrize("keyword", sorted(TAKES))
def test_none_is_a_cross_section_library(keyword):
    with pytest.raises(ValueError, match=SAYS[TAKES[keyword]]):
        yani.neutron_cross_section_data = keyword
    with pytest.raises(ValueError, match=SAYS[TAKES[keyword]]):
        yani.set_neutron_cross_section_data_entry("Fe56", keyword)
