"""Tests for the completion marker that the resume path skips on.

The regression these exist for: the writers used to drop ``version.json`` in
first, so every directory an interrupted run left behind looked finished. A
rebuild killed partway through U238 left 2 of its 8 tables on disk, and a
plain re-run skipped it and called the library complete.
"""

import json

import pytest

from nuclear_data_to_arrow.completion import (
    MARKER, REQUIRED_TABLES, is_complete, write_completion_marker,
)


def make_folder(tmp_path, particle, *, marker=True, tables=None):
    """Build a .arrow folder holding *tables* (default: all required ones)."""
    folder = tmp_path / "Fe56.arrow"
    folder.mkdir()
    if tables is None:
        tables = REQUIRED_TABLES[particle]
    for name in tables:
        (folder / name).write_bytes(b"not really arrow, presence is the point")
    if marker:
        write_completion_marker(folder, "endf-b8.1")
    return folder


# --- is_complete ------------------------------------------------------------

@pytest.mark.parametrize("particle", ["neutron", "photon"])
def test_complete_folder_is_complete(tmp_path, particle):
    assert is_complete(make_folder(tmp_path, particle), particle)


@pytest.mark.parametrize("particle", ["neutron", "photon"])
def test_missing_marker_is_incomplete(tmp_path, particle):
    folder = make_folder(tmp_path, particle, marker=False)
    assert not is_complete(folder, particle)


@pytest.mark.parametrize("missing", REQUIRED_TABLES["neutron"])
def test_a_missing_required_table_is_incomplete(tmp_path, missing):
    """Catches the directories the old marker-first ordering left behind: the
    marker is there and the tables are not."""
    tables = [t for t in REQUIRED_TABLES["neutron"] if t != missing]
    folder = make_folder(tmp_path, "neutron", tables=tables)
    assert (folder / MARKER).is_file()
    assert not is_complete(folder, "neutron")


def test_the_actual_u238_shape_is_incomplete(tmp_path):
    """What the OOM-killed rebuild left on disk: marker plus nuclide.arrow,
    and nothing else. The old check called this done."""
    folder = make_folder(tmp_path, "neutron", tables=["nuclide.arrow"])
    assert not is_complete(folder, "neutron")


def test_optional_tables_are_not_required(tmp_path):
    """urr, total_nu and fission_photon are written only when the evaluation
    has that data, so a nuclide without them is still complete."""
    folder = make_folder(tmp_path, "neutron")
    assert not (folder / "urr.arrow").exists()
    assert is_complete(folder, "neutron")


def test_absent_folder_is_incomplete(tmp_path):
    assert not is_complete(tmp_path / "NoSuchNuclide.arrow", "neutron")


# --- write_completion_marker ------------------------------------------------

def test_marker_records_the_library_and_converter(tmp_path):
    folder = make_folder(tmp_path, "neutron")
    info = json.loads((folder / MARKER).read_text())
    assert info["library"] == "endf-b8.1"
    assert info["format_version"] == 1
    assert info["converter_version"]
    assert info["created_utc"]


def test_marker_write_leaves_no_temporary_behind(tmp_path):
    """It is renamed into place, so a kill mid-write leaves either no marker
    or a whole one, never a half-written file that still passes is_file()."""
    folder = make_folder(tmp_path, "neutron")
    assert not list(folder.glob("*.tmp"))


def test_marker_overwrites_a_previous_one(tmp_path):
    folder = make_folder(tmp_path, "neutron")
    write_completion_marker(folder, "jeff-4.0")
    assert json.loads((folder / MARKER).read_text())["library"] == "jeff-4.0"


# --- data_version (yamc issue #366) -----------------------------------------

def test_marker_records_the_data_version(tmp_path):
    """The field yamc compares a cached copy against.

    Distinct from converter_version on purpose: that identifies the code, and
    two rebuilds from the same converter are different data. The hosted objects
    are overwritten in place on a re-publish, so the URL and the cache key are
    unchanged and this stamp is the only thing that differs.
    """
    folder = make_folder(tmp_path, "neutron")
    write_completion_marker(folder, "fendl-3.2d", "2026-08-09.1")
    info = json.loads((folder / MARKER).read_text())
    assert info["data_version"] == "2026-08-09.1"
    assert info["library"] == "fendl-3.2d"


def test_data_version_is_present_even_when_not_supplied(tmp_path):
    """Always written, so a consumer reads a field rather than guessing whether
    an absent key means unstamped or means an old converter. Empty is what yamc
    treats as unstamped."""
    folder = make_folder(tmp_path, "neutron")
    info = json.loads((folder / MARKER).read_text())
    assert info["data_version"] == ""


def test_data_version_is_not_the_converter_version(tmp_path):
    """A rebuild from the same converter must be able to claim a new release,
    which is the whole reason the field exists."""
    folder = make_folder(tmp_path, "neutron")
    write_completion_marker(folder, "fendl-3.2d", "2026-08-09.2")
    info = json.loads((folder / MARKER).read_text())
    assert info["data_version"] != info["converter_version"]
