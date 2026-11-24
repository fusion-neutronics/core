"""Check that every ``yamc[...]`` extra the README and docs name is one yamc declares.

pip treats an undeclared extra as a warning, not an error, so a page telling
people to run ``pip install "yamc[cad]"`` for an extra that does not exist
installs yamc without cadquery and exits 0. This reads the extras from
packages/yamc/pyproject.toml and fails on any page that names another.

Run: python docs/check_extras.py   (from docs/yamc)
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / "docs" / "yamc"

# `yamc[cad]` or `yamc[all,pandas]`. An extra starts with a letter, which keeps
# indexing such as `yamc[0]` out.
MENTION = re.compile(r"\byamc\[\s*([A-Za-z][\w.\-]*(?:\s*,\s*[A-Za-z][\w.\-]*)*)\s*\]")


def normalize(extra: str) -> str:
    """Compare extra names the way pip does (PEP 685)."""
    return re.sub(r"[-_.]+", "-", extra).lower()


def main() -> int:
    project = tomllib.loads((ROOT / "packages" / "yamc" / "pyproject.toml").read_text())["project"]
    declared = {normalize(name) for name in project.get("optional-dependencies", {})}
    pages = [ROOT / "packages" / "yamc" / "README.md", *sorted((SITE / "docs").rglob("*.md"))]
    failures = [
        f"{page.relative_to(ROOT)}:{number}: yamc[{name}] is not an extra packages/yamc declares"
        for page in pages
        for number, line in enumerate(page.read_text().splitlines(), 1)
        for names in MENTION.findall(line)
        for name in (normalize(n.strip()) for n in names.split(","))
        if name not in declared
    ]
    if failures:
        print("::error::extras that `pip install` would not honour:", *failures, sep="\n  ")
        return 1
    print(f"extras named across {len(pages)} pages are all declared ({', '.join(sorted(declared))}): OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
