"""Isomeric branching from several sources, in order.

``transmutation_branch_ratios`` takes one branching source or several. Each
reaction takes its isomeric split, and that split's MF=40 covariance, from the
first source that evaluates it, so the order is the whole meaning of the list
and is kept as given. ``irdff-ii``, the IAEA dosimetry library IRDFF-II,
publishes isomeric branching and nothing else, so naming it as any other kind
of source is refused before anything is downloaded.

The end-to-end case downloads ENDF/B-VIII.1 and IRDFF-II data, as the other
transmutation tests download theirs.
"""

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")


@pytest.mark.parametrize(
    "assigned",
    ["endf-b8.1", ["endf-b8.1"], ["irdff-ii", "endf-b8.1"], [], None],
)
def test_the_setting_reads_back_as_it_was_assigned(assigned):
    """A string stays a string and a list a list, a list of one included."""
    yani.transmutation_branch_ratios = assigned
    assert yani.transmutation_branch_ratios == assigned
    assert type(yani.transmutation_branch_ratios) is type(assigned)


def test_several_sources_keep_their_order():
    yani.transmutation_branch_ratios = ("endf-b8.1", "irdff-ii")
    assert yani.transmutation_branch_ratios == ["endf-b8.1", "irdff-ii"]


@pytest.mark.parametrize(
    "sources, message",
    [(["irdff-ii", "irdff-ii"], "listed twice"), (["irdff-ii", ""], "empty")],
)
def test_a_list_that_cannot_mean_anything_is_refused_and_changes_nothing(sources, message):
    yani.transmutation_branch_ratios = "endf-b8.1"
    with pytest.raises(ValueError, match=message):
        yani.transmutation_branch_ratios = sources
    assert yani.transmutation_branch_ratios == "endf-b8.1"


def test_a_source_is_a_string():
    with pytest.raises(TypeError):
        yani.transmutation_branch_ratios = True


@pytest.mark.parametrize(
    "setting",
    [
        "cross_section_data",
        "transmutation_decay_data",
        "transmutation_reactions",
        "transmutation_fission_yields",
    ],
)
def test_irdff_ii_is_refused_as_anything_but_a_branching_source(setting):
    before = getattr(yani, setting)
    with pytest.raises(ValueError, match="supplies isomeric branching only"):
        setattr(yani, setting, "irdff-ii")
    assert getattr(yani, setting) == before


def test_irdff_ii_is_refused_as_one_nuclide_s_source_too():
    with pytest.raises(ValueError, match="supplies isomeric branching only"):
        yani.cross_section_data = {"Fe56": "irdff-ii"}
    with pytest.raises(ValueError, match="supplies isomeric branching only"):
        yani.set_cross_section_data_entry("Fe56", "irdff-ii")


def _nb93_report(sources):
    """The branching report of a Nb93 foil under a 13.5-14.5 MeV group."""
    yani.cross_section_data = "endf-b8.1"
    yani.transmutation_branch_ratios = sources
    foil = yani.Material({"Nb93": 1.0}, density=8.57, temperature=294, volume=1.0)
    spectrum = yani.NeutronSource(energy=yani.sources.Histogram([1.35e7, 1.45e7], [1.0]))
    schedule = yani.PulseSchedule([yani.Pulse(rate=1.0e10, duration=(1, "h"), source=spectrum)])
    results = foil.transmute(schedule=schedule)
    return results.get_branching_report(foil.id, 0)


def test_the_report_names_the_source_of_each_split():
    """IRDFF-II first: Nb93 (n,n') and (n,2n) take IRDFF-II's splits, on
    ENDF/B-VIII.1 totals, so the run says it is mixed. ENDF/B-VIII.1 alone: its
    own (n,n') split, no (n,2n) isomer split (it evaluates none), one library."""
    report = _nb93_report(["irdff-ii", "endf-b8.1"])
    source = {(c["parent"], c["reaction"]): c["source"] for c in report["channels"]}
    assert source[("Nb93", "(n,n')")] == "irdff-ii"
    assert source[("Nb93", "(n,2n)")] == "irdff-ii"
    assert report["sources"][0] == "irdff-ii"
    assert report["mixed"] is True
    nb93 = [c for c in report["channels"] if c["parent"] == "Nb93" and c["source"]]
    assert {c["cross_section_library"] for c in nb93} == {"endf-b8.1"}

    report = _nb93_report("endf-b8.1")
    source = {(c["parent"], c["reaction"]): c["source"] for c in report["channels"]}
    assert source[("Nb93", "(n,n')")] == "endf-b8.1"
    assert ("Nb93", "(n,2n)") not in source
    assert report["sources"] == ["endf-b8.1"]
    assert report["mixed"] is False
