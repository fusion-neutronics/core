"""yani.verify_library re-hashes a library folder against its manifest.json.

The binding is shared with yamc, but yani's wheel registers its own copy, so
this checks the registration on this wheel rather than the function itself.
"""

import hashlib
import json

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

PATH = "transmutation/branching.arrow/branching.arrow"
DATA = b"branching-table"


def test_verify_library_rehashes_a_chain_only_library(tmp_path):
    (tmp_path / PATH).parent.mkdir(parents=True)
    (tmp_path / PATH).write_bytes(DATA)
    entry = {
        "path": PATH,
        "bytes": len(DATA),
        "sha256": hashlib.sha256(DATA).hexdigest(),
    }
    manifest = {
        "schema": "yamc-data-manifest",
        "schema_version": 1,
        "keyword": "irdff-ii",
        "release": "2026-10-01",
        "format_version": 2,
        "converter_versions": ["yani-convert 0.1.0"],
        "created": "2026-10-01T00:00:00Z",
        "provenance": {},
        "particles": [],
        "file_count": 1,
        "total_bytes": len(DATA),
        "files": [entry],
    }
    (tmp_path / "manifest.json").write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )

    report = yani.verify_library(str(tmp_path))
    assert (report["keyword"], report["ok"]) == ("irdff-ii", True)

    (tmp_path / PATH).write_bytes(DATA.upper())
    report = yani.verify_library(str(tmp_path))
    assert (report["changed"], report["ok"]) == ([PATH], False)
