"""yani.generate_library refuses a build it cannot do before it fetches anything.

The builder is shared with the other wheel, which registers its own copy, so
these check the registration and the refusals that need no network, no NJOY
and no source archive.
"""

from pathlib import Path

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")


def test_an_unknown_library_is_refused(tmp_path):
    with pytest.raises(ValueError, match="endf-b8.1, jeff-4.0"):
        yani.generate_library("endf-b8.0", str(tmp_path / "lib"))


def test_an_unknown_part_is_refused(tmp_path):
    with pytest.raises(ValueError, match="photon, photonuclear"):
        yani.generate_library("endf-b8.1", str(tmp_path / "lib"), parts=["photons"])


def test_a_part_the_release_does_not_carry_is_refused(tmp_path):
    # TENDL's chain is assembled with another library's decay data, which it
    # never publishes as its own.
    with pytest.raises(RuntimeError, match="has no decay part"):
        yani.generate_library("tendl-2025", str(tmp_path / "lib"), parts=["decay"])


def test_bad_temperatures_are_refused(tmp_path):
    with pytest.raises(ValueError, match="above"):
        yani.generate_library("endf-b8.1", str(tmp_path / "lib"), temperatures=[0.0])


def test_a_folder_holding_other_files_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_text("not a library")
    with pytest.raises(RuntimeError, match="not a library build"):
        yani.generate_library("irdff-ii", str(tmp_path))
    assert not (tmp_path / ".build").exists()


def test_the_download_cache_is_refused():
    # Refused before anything is made, so pointing it inside the real cache
    # leaves the cache as it was.
    inside = Path(yani.cache_dir()) / "generate-library-test"
    with pytest.raises(RuntimeError, match="inside the nuclear data cache"):
        yani.generate_library("irdff-ii", str(inside))
    assert not inside.exists()
