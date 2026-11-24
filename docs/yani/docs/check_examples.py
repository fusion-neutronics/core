"""Type-check the Python examples in the prose docs against yani's stubs.

The API reference is generated from the installed wheel, so a renamed or
retyped member shows up there on its own. The hand-written examples on the
other pages do not: they are prose, and prose keeps compiling. yani 0.11.0
changed ``Material.transmute()`` from returning ``list[Material]`` to returning
``TransmutationResults``, which has no ``__getitem__``, and every
``results[-1]`` in these pages -- the README's first example among them -- went
on rendering perfectly while raising TypeError for anyone who typed it in.

So the examples are extracted and run past mypy against the ``.pyi`` stubs the
wheel ships. This is a type check, not an execution: it needs no nuclear data
and no solve, which is what makes it cheap enough to sit in the docs job. It
catches the class of breakage above -- indexing what is not indexable, calling
what is gone, passing what no longer fits -- and it catches none of the physics.

Each page is concatenated into one module in source order, because that is how
the page reads: a name bound in one block is used in a later one. Pages that
open on a name they never bind get it from PLACEHOLDERS below.

Run: python docs/check_examples.py [--keep]
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Discover guide pages so new examples cannot silently fall outside this check.
# Pages without plain Python fences, including API directives, are skipped.
PAGES = [
    "../../packages/yani/README.md",
    *(page.relative_to(ROOT).as_posix() for page in sorted((ROOT / "docs").rglob("*.md"))),
]

# Names a page uses without binding, because the surrounding prose introduces
# them rather than the code. Typed rather than stubbed as Any: an annotation is
# what lets mypy check the calls made on them, which is the whole point.
PLACEHOLDERS = """\
import yani

material: yani.Material          # "a material you already have"
flux_709: list[float]            # a 709-group flux, tabulated elsewhere
neutron: list[str]               # ENDF evaluation paths, per convert_* call
decay: list[str]
fpy: list[str]
jendl_decay: list[str]           # a second library's decay files, for the fill
"""

# Errors that say something about the page rather than about yani's API.
# `import matplotlib` is not the docs' problem, and a fragment that opens
# mid-narrative will report names bound in prose rather than in code.
IGNORED_CODES = ("import-not-found", "import-untyped")

FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def extract(page: Path) -> str | None:
    blocks = FENCE.findall(page.read_text())
    return "\n\n".join(blocks) if blocks else None


def main() -> int:
    keep = "--keep" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="yani-doc-examples-"))
    written = []

    for rel in PAGES:
        page = ROOT / rel
        if not page.exists():
            print(f"::error::{rel} is listed in check_examples.py and is not there")
            return 1
        code = extract(page)
        if code is None:
            continue
        out = tmp / (rel.replace("/", "_").removesuffix(".md") + ".py")
        out.write_text(PLACEHOLDERS + "\n" + code)
        written.append((rel, out))

    if not written:
        print("::error::no Python examples found -- has the fence syntax changed?")
        return 1

    # The preamble is prepended, so mypy's line numbers are offset from the
    # page's. The page name is what matters for finding the block.
    offset = len(PLACEHOLDERS.splitlines()) + 1

    proc = subprocess.run(
        [sys.executable, "-m", "mypy",
         "--no-error-summary", "--no-color-output",
         # A doc example is illustrative: it may leave a value unused or call
         # something that returns Any. Those are not what this guards.
         "--disable-error-code=name-defined",
         "--disable-error-code=used-before-def",
         "--disable-error-code=has-type",
         "--disable-error-code=annotation-unchecked",
         # Independent blocks on one page reuse short names for different
         # things -- usage.md binds `edges` to a group structure in one and to
         # a dict of chain edges in another. Concatenating them is what makes
         # a later block see an earlier one's objects, and this is its price.
         "--allow-redefinition",
         # `edges = {}` in a doc example is clear to a reader and bare to
         # mypy. Annotating empty literals is not what these pages are for.
         "--disable-error-code=var-annotated",
         *(str(p) for _, p in written)],
        capture_output=True, text=True, cwd=ROOT,
    )

    failures = []
    keeping = True
    for line in (proc.stdout + proc.stderr).splitlines():
        if not line.strip():
            continue
        # mypy trails a `note:` after some errors. A note belongs to the error
        # above it, so it is dropped with one.
        is_note = " note:" in line
        if not is_note:
            keeping = not any(c in line for c in IGNORED_CODES)
        if not keeping:
            continue
        for rel, path in written:
            if line.startswith(str(path)):
                rest = line[len(str(path)):]
                m = re.match(r":(\d+):(.*)", rest)
                if m:
                    line = f"{rel} (example line {int(m.group(1)) - offset}):{m.group(2)}"
                else:
                    line = f"{rel}{rest}"
                break
        failures.append(line)

    pages = ", ".join(rel for rel, _ in written)
    print(f"type-checked the examples in {pages}")

    if failures:
        print("::error::the documented examples do not type-check against the "
              "installed yani:", *failures, sep="\n  ")
        if keep:
            print(f"\nextracted modules kept in {tmp}")
        return 1

    print("examples type-check against the installed yani: OK")
    if keep:
        print(f"extracted modules kept in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
