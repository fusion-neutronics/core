"""Tests for the chain manifest, which no single writer can see all of.

``export_transmutation_to_arrow`` emits decay, reactions and fission_yields;
``export_branching_to_arrow`` emits branching; a library calls whichever
subsections it can supply, into the same directory. Each writer used to build
the manifest from its own results and overwrite the file, so the one that ran
last decided what the chain claimed to hold: ENDF/B-VIII.1 wrote all four
subsections and listed one, TENDL-2017 wrote two and listed the other one.

The subsections themselves were always written correctly. Only the index of
them was wrong, which is the kind of defect that survives a long time.
"""

import json
import shutil

from nuclear_data_to_arrow.transmutation_writer import _write_manifest


def subsections(path):
    return sorted(json.loads((path / "manifest.json").read_text())["subsections"])


def make(path, *names):
    path.mkdir(parents=True, exist_ok=True)
    for n in names:
        (path / n).mkdir(exist_ok=True)
    return path


def test_endf_ordering_keeps_all_four(tmp_path):
    """The ENDF/B-VIII.1 regression: three subsections, then branching."""
    p = make(tmp_path / "chain", "decay", "reactions", "fission_yields", "branching")

    _write_manifest(p, "endf-b8.1", {
        "decay": {"path": "decay"},
        "reactions": {"path": "reactions"},
        "fission_yields": {"path": "fission_yields"},
    }, "t1")
    _write_manifest(p, "endf-b8.1", {"branching": {"path": "branching"}}, "t2")

    assert subsections(p) == ["branching", "decay", "fission_yields", "reactions"]


def test_tendl_ordering_keeps_both(tmp_path):
    """The TENDL-2017 regression: branching first, then reactions."""
    p = make(tmp_path / "chain", "branching", "reactions")

    _write_manifest(p, "tendl-2017", {"branching": {"path": "branching"}}, "t1")
    _write_manifest(p, "tendl-2017", {"reactions": {"path": "reactions"}}, "t2")

    assert subsections(p) == ["branching", "reactions"]


def test_a_different_library_starts_over(tmp_path):
    """Rebuilding a directory for another library must not inherit subsections."""
    p = make(tmp_path / "chain", "branching", "reactions")

    _write_manifest(p, "tendl-2017", {"branching": {"path": "branching"}}, "t1")
    _write_manifest(p, "jeff-4.0", {"reactions": {"path": "reactions"}}, "t2")

    manifest = json.loads((p / "manifest.json").read_text())
    assert manifest["library"] == "jeff-4.0"
    assert subsections(p) == ["reactions"]


def test_a_removed_subsection_is_dropped(tmp_path):
    """The manifest must not keep advertising a directory that has gone."""
    p = make(tmp_path / "chain", "branching", "reactions")
    _write_manifest(p, "tendl-2017", {"branching": {"path": "branching"}}, "t1")
    _write_manifest(p, "tendl-2017", {"reactions": {"path": "reactions"}}, "t2")

    shutil.rmtree(p / "branching")
    _write_manifest(p, "tendl-2017", {"reactions": {"path": "reactions"}}, "t3")

    assert subsections(p) == ["reactions"]


def test_an_unreadable_manifest_is_rebuilt(tmp_path):
    """A corrupt index must not abort a chain that is otherwise written."""
    p = make(tmp_path / "chain", "reactions")
    (p / "manifest.json").write_text("{ not json")

    _write_manifest(p, "tendl-2017", {"reactions": {"path": "reactions"}}, "t1")

    assert subsections(p) == ["reactions"]


def test_rewriting_one_subsection_leaves_the_others(tmp_path):
    """Re-emitting a single subsection must not drop the rest."""
    p = make(tmp_path / "chain", "decay", "reactions", "fission_yields", "branching")
    _write_manifest(p, "endf-b8.1", {
        "decay": {"path": "decay"},
        "reactions": {"path": "reactions"},
        "fission_yields": {"path": "fission_yields"},
        "branching": {"path": "branching"},
    }, "t1")

    _write_manifest(p, "endf-b8.1", {"reactions": {"path": "reactions"}}, "t2")

    assert subsections(p) == ["branching", "decay", "fission_yields", "reactions"]
