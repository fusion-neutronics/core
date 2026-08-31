# SPDX-License-Identifier: MIT
"""Dump a golden reference for a depletion chain built from several files.

`dump_golden.py` writes one dump per fixture, which suits a reader that takes
one file. A chain is the join of three sub-libraries, so it needs its own
golden naming all of them.

    python tools/dump_chain_golden.py
"""

from __future__ import annotations

import difflib
import io
import lzma
import sys
from pathlib import Path

import endf
from endf.chain import Chain

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_golden import Dump  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "crates" / "endf" / "tests" / "golden"

#: The fixtures the chain is built from. Upstream keeps these in `tests/` at
#: the repository root; here that name is a symlink to `crates/yamc/tests`, so
#: they live inside the crate. The DECAY and NEUTRON lines record the path
#: relative to ROOT, which is how `golden.rs::repo_root()` resolves them.
REL = "crates/endf/fixtures"
FIXTURES = ROOT / REL

#: The decay evaluations of the chain. Between them these close every path the
#: chain follows, so no product needs a stand-in except Cs137's, which is
#: deliberate: barium is absent, so the stand-in walk runs.
DECAY = [
    "dec-048_Cd_116.endf.xz",
    "dec-049_In_115.endf.xz",
    "dec-049_In_116.endf.xz",
    "dec-049_In_116m1.endf.xz",
    "dec-049_In_116m2.endf.xz",
    "dec-050_Sn_115.endf.xz",
    "dec-050_Sn_116.endf.xz",
    "dec-054_Xe_136.endf.xz",
    "dec-054_Xe_137.endf.xz",
    "dec-055_Cs_137.endf.xz",
]

#: The neutron evaluations, which supply the transmutation Q values.
NEUTRON = [
    "n-049_In-115_trimmed.endf.xz",
    "n-054_Xe_136_trimmed.endf.xz",
]

#: Capture only. The fixtures do not close the (n,2n) and charged-particle
#: paths, and an unclosed path exercises the stand-in walk rather than the
#: chain, which is what the decay fixtures are already for.
REACTIONS = ["(n,gamma)"]


def dump_chain(d: Dump, path: str, chain) -> None:
    d.int(f"{path}/n", len(chain.nuclides))
    for i, nuclide in enumerate(chain.nuclides):
        np_ = f"{path}/{i}"
        d.text(f"{np_}/name", nuclide.name)
        if nuclide.half_life is not None:
            d.float(f"{np_}/half_life", nuclide.half_life)
        d.float(f"{np_}/decay_energy", nuclide.decay_energy)
        for j, mode in enumerate(nuclide.decay_modes):
            mp = f"{np_}/decay/{j}"
            d.text(f"{mp}/type", mode.type)
            if mode.target is not None:
                d.text(f"{mp}/target", mode.target)
            d.float(f"{mp}/branching_ratio", mode.branching_ratio)
        for j, rx in enumerate(nuclide.reactions):
            rp = f"{np_}/reaction/{j}"
            d.text(f"{rp}/type", rx.type)
            if rx.target is not None:
                d.text(f"{rp}/target", rx.target)
            d.float(f"{rp}/Q", rx.Q)
            d.float(f"{rp}/branching_ratio", rx.branching_ratio)


def generate(out) -> None:
    """Write the whole dump to an open text stream."""
    decay = [endf.Material(FIXTURES / n) for n in DECAY]
    neutron = [endf.Material(FIXTURES / n) for n in NEUTRON]
    chain = Chain.from_endf(
        decay, [], neutron, reactions=tuple(REACTIONS), progress=False
    )

    out.write("# golden reference for a depletion chain, by the Python reader\n")
    out.write("# regenerate with: python tools/dump_chain_golden.py\n")
    out.write("KIND chain\n")
    for name in DECAY:
        out.write(f"DECAY {REL}/{name}\n")
    for name in NEUTRON:
        out.write(f"NEUTRON {REL}/{name}\n")
    for name in REACTIONS:
        out.write(f"REACTION {name}\n")
    dump_chain(Dump(out), "chain", chain)


def main() -> None:
    target = GOLDEN_DIR / "chain.txt.xz"

    # As in dump_golden.py, the check compares text rather than compressed
    # bytes, which are not portable between liblzma versions.
    if "--check" in sys.argv[1:]:
        fresh = io.StringIO()
        generate(fresh)
        with lzma.open(target, "rt") as fh:
            stored = fh.read()
        if fresh.getvalue() == stored:
            print(f"{target.relative_to(ROOT)} matches", file=sys.stderr)
            return
        print(f"{target.relative_to(ROOT)}: drifted", file=sys.stderr)
        diff = difflib.unified_diff(
            stored.splitlines(),
            fresh.getvalue().splitlines(),
            fromfile="stored",
            tofile="regenerated",
            lineterm="",
            n=1,
        )
        for i, line in enumerate(diff):
            if i >= 40:
                print("  ...", file=sys.stderr)
                break
            print(f"  {line}", file=sys.stderr)
        sys.exit("the chain golden no longer matches the reader")

    with lzma.open(target, "wt", preset=9) as out:
        generate(out)
    print(f"{target.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
