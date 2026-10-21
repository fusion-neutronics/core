#!/usr/bin/env python3
"""Require binding version bumps when a PR changes the published data format.

Run from a checkout with the PR base available:
    python scripts/check_format_version_bumps.py --base <base-commit>

Only Git-tracked commits are compared. No network or installed wheel is needed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_SOURCE = "crates/yamc-nuclide/src/storage/release.rs"
BINDINGS = {
    "yamc-core": "crates/yamc-python/Cargo.toml",
    "yani-core": "crates/yani-python/Cargo.toml",
}


def supported_formats(source: str) -> tuple[int, int]:
    match = re.search(
        r"\bpub\s+const\s+SUPPORTED_FORMAT_VERSIONS\s*:\s*RangeInclusive<u32>"
        r"\s*=\s*(\d+)\s*\.\.=\s*(\d+)\s*;",
        source,
    )
    if match is None:
        raise ValueError("cannot read SUPPORTED_FORMAT_VERSIONS; update the format-bump check")
    return int(match[1]), int(match[2])


def check_bumps(changed: set[str], before: dict[str, str], after: dict[str, str]) -> list[str]:
    schema_changed = any(
        path.startswith("crates/nuclear-data-schema/src/") and path.endswith(".rs")
        for path in changed
    )
    formats_changed = RELEASE_SOURCE in changed and (
        supported_formats(before[RELEASE_SOURCE]) != supported_formats(after[RELEASE_SOURCE])
    )
    if not schema_changed and not formats_changed:
        return []
    failures = []
    for distribution, path in BINDINGS.items():
        old = tomllib.loads(before[path])["package"]["version"]
        new = tomllib.loads(after[path])["package"]["version"]
        if old == new:
            failures.append(
                f"{distribution}: published data format changed but {path} remains at {old}; "
                "bump the binding version and update Cargo.lock in this PR"
            )
    return failures


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="PR base commit to compare with HEAD")
    arguments = parser.parse_args()
    base = git("rev-parse", "--verify", f"{arguments.base}^{{commit}}").strip()
    changed = set(filter(None, git("diff", "--name-only", "-z", base, "HEAD", "--").split("\0")))
    relevant = RELEASE_SOURCE in changed or any(
        path.startswith("crates/nuclear-data-schema/src/") and path.endswith(".rs")
        for path in changed
    )
    if not relevant:
        print("Published data format unchanged; no binding version bump required.")
        return 0
    paths = [*BINDINGS.values()]
    if RELEASE_SOURCE in changed:
        paths.append(RELEASE_SOURCE)
    before = {path: git("show", f"{base}:{path}") for path in paths}
    after = {path: git("show", f"HEAD:{path}") for path in paths}
    failures = check_bumps(changed, before, after)
    for failure in failures:
        print(f"::error::{failure}", file=sys.stderr)
    if failures:
        return 1
    print("Published data format and binding versions are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())