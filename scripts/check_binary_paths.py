#!/usr/bin/env python3
"""Fail if a shipped binary carries the absolute path of the machine that built it.

cargo bakes the registry path of every dependency into panic-location strings,
so an unremapped build leaks either the CI container layout or, for anything
built by hand, a developer's home directory. `.cargo/config.toml` remaps the
three runner layouts to `/cargo`; this is the check that says the remap is
actually taking effect, because a `--remap-path-prefix` whose prefix does not
match is a silent no-op and the next toolchain or runner-image change could move
the prefix without anything noticing.

What this deliberately does NOT check:

* Dependency names and versions. Those survive the remap as
  `/cargo/registry/.../serde_json-1.0.151/src/read.rs`, and they are public by
  design: THIRD-PARTY-LICENSES.html ships in every wheel and lists both.
* `/rustc/<hash>` paths. Baked into the precompiled std, so removing them needs
  -Zbuild-std on nightly. They name the compiler commit, not the machine.
* Workspace-relative paths such as `crates/endf/src/urr.rs`. cargo emits those
  relative already, so they name no machine.

Usage:
    python scripts/check_binary_paths.py dist/*.whl
    python scripts/check_binary_paths.py path/to/some.wasm path/to/some.so
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

# A home directory, anyone's. `/root` is the manylinux container's, and the
# reason the published wheels leak today; the other two are the GitHub-hosted
# runner layouts. Trailing separator so `/rootfs` does not match `/root`.
LEAKS = re.compile(rb"(/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|/root)/[A-Za-z0-9._/-]*")

# Binary members worth scanning inside a wheel. Everything else in there is
# text we author.
BINARY_SUFFIXES = (".so", ".pyd", ".dylib", ".wasm")

# Members known to carry builder paths, with the issue that removes them.
#
# Deliberately EMPTY. It held the three `yamc/_wasm/*.wasm` blobs, which were
# committed and built by hand so the remaps in `.cargo/config.toml` could not
# reach them: those cover the CI runner layouts and a workstation's home is not
# one. They are built in CI now and come out as `/build/...`, and
# `yamc_geo_bg.wasm` is gone entirely because nothing referenced it. That was
# issue #16.
#
# The mechanism is kept rather than deleted, because the empty set is still
# doing work: a new blob, or a leak in the extension module itself, fails. And
# the check below treats a listed-but-clean member as a failure too, which is
# what forced this list to be emptied in the same change that fixed the blobs
# rather than left behind as a stale excuse. Keep that property if you ever add
# an entry.
KNOWN_DIRTY: set[str] = set()


def scan(name: str, blob: bytes) -> list[str]:
    """Return a deduplicated, truncated list of leaked paths in one blob."""
    found = {m.group(0).decode("utf-8", "replace") for m in LEAKS.finditer(blob)}
    return sorted(found)


def check(path: Path) -> int:
    """Report on one file. Returns the number of distinct leaks found."""
    targets: list[tuple[str, bytes]] = []

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.endswith(BINARY_SUFFIXES):
                    targets.append((f"{path.name}::{member}", zf.read(member)))
        if not targets:
            print(f"::error::{path.name} contains no binary members to check")
            return 1
    else:
        targets.append((str(path), path.read_bytes()))

    total = 0
    for name, blob in targets:
        member = name.split("::", 1)[-1]
        known = member in KNOWN_DIRTY
        leaks = scan(name, blob)

        if leaks and known:
            print(f"known (see issue 16): {member} carries {len(leaks)} path(s)")
        elif leaks:
            total += len(leaks)
            print(f"::error::{name} carries {len(leaks)} builder path(s):")
            for leak in leaks[:10]:
                print(f"    {leak}")
            if len(leaks) > 10:
                print(f"    ... and {len(leaks) - 10} more")
        elif known:
            total += 1
            print(
                f"::error::{member} is listed in KNOWN_DIRTY and is now clean. "
                "Remove it from the list rather than leaving a stale excuse."
            )
        else:
            print(f"ok: {name}")
    return total


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_binary_paths.py <file> [file ...]", file=sys.stderr)
        return 2

    paths = [Path(a) for a in argv]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"::error::{p} is not a file")
        return 2

    total = sum(check(p) for p in paths)
    if total:
        print(
            "::error::builder paths reached a shipped binary. Either the remaps "
            "in .cargo/config.toml no longer match the build layout, or this "
            "artifact was built by hand outside CI."
        )
        return 1
    # "no new": a KNOWN_DIRTY member that leaked is reported above, not here.
    print(f"checked {len(paths)} file(s): no new builder paths")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
