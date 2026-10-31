"""verify_library re-hashes a library folder against its manifest.json.

The manifest is written here the way the publishing scripts write one
(release_manifest.py's yamc-data-manifest v1 shape), so these tests need no
data download and no converter.
"""

import hashlib
import json

import pytest

import yamc

FILES = {
    "neutron/Li6.arrow/version.json": b'{"format_version": 2}',
    "neutron/Li6.arrow/reactions.arrow": b"Li6-reactions",
    "neutron/index.json": b'{"format":1}\n',
    "transmutation/decay.arrow/nuclides.arrow": b"decay-table",
}


def _library(root, format_version=2):
    """A library folder holding FILES and the manifest that lists them.

    Returns the folder and the sha256 of its manifest.
    """
    for path, data in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    files = [
        {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(FILES.items())
    ]
    manifest = {
        "schema": "yamc-data-manifest",
        "schema_version": 1,
        "keyword": "endf-b8.1",
        "release": "2026-10-01",
        "format_version": format_version,
        "converter_versions": ["yamc-convert 0.1.0"],
        "created": "2026-10-01T00:00:00Z",
        "provenance": {},
        "particles": ["neutron"],
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    (root / "manifest.json").write_bytes(text.encode())
    return root, hashlib.sha256(text.encode()).hexdigest()


def test_an_intact_library_verifies(tmp_path):
    root, manifest_sha256 = _library(tmp_path)
    assert yamc.verify_library(str(root)) == {
        "keyword": "endf-b8.1",
        "release": "2026-10-01",
        "format_version": 2,
        "manifest_sha256": manifest_sha256,
        "file_count": 4,
        "total_bytes": sum(len(data) for data in FILES.values()),
        "missing": [],
        "changed": [],
        "unlisted": [],
        "ok": True,
    }


def test_changed_missing_and_unlisted_files_are_all_reported(tmp_path):
    root, _ = _library(tmp_path)
    # The same size with other bytes, which only the hash catches.
    (root / "neutron/Li6.arrow/reactions.arrow").write_bytes(b"Li6-REACTIONS")
    (root / "neutron/index.json").write_bytes(b"{}")
    (root / "transmutation/decay.arrow/nuclides.arrow").unlink()
    (root / "neutron/Li7.arrow").mkdir()
    (root / "neutron/Li7.arrow/version.json").write_bytes(b"{}")
    # Not part of the release, so not looked at.
    (root / ".build").mkdir()
    (root / ".build/state.json").write_bytes(b"{}")
    (root / "notes.txt").write_bytes(b"mine")

    report = yamc.verify_library(str(root))
    assert report["changed"] == [
        "neutron/Li6.arrow/reactions.arrow",
        "neutron/index.json",
    ]
    assert report["missing"] == ["transmutation/decay.arrow/nuclides.arrow"]
    assert report["unlisted"] == ["neutron/Li7.arrow/version.json"]
    assert report["ok"] is False


def test_an_unlisted_file_alone_fails_the_check(tmp_path):
    root, _ = _library(tmp_path)
    (root / "neutron/Li6.arrow/extra.arrow").write_bytes(b"x")
    report = yamc.verify_library(str(root))
    assert (report["missing"], report["changed"]) == ([], [])
    assert report["unlisted"] == ["neutron/Li6.arrow/extra.arrow"]
    assert report["ok"] is False


def test_a_folder_without_a_release_manifest_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="holds no manifest.json"):
        yamc.verify_library(str(tmp_path))
    # A converted chain's own manifest is not a release's.
    (tmp_path / "manifest.json").write_text('{"format_version": 2, "subsections": {}}')
    with pytest.raises(RuntimeError, match="not a release manifest"):
        yamc.verify_library(str(tmp_path))


def test_a_library_in_a_format_this_build_does_not_read_is_refused(tmp_path):
    root, _ = _library(tmp_path, format_version=99)
    with pytest.raises(RuntimeError, match="format_version 99"):
        yamc.verify_library(str(root))
