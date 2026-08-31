# SPDX-License-Identifier: MIT
"""Write ENDF-6 records, for building synthetic fixtures.

Some shapes the readers handle have no fixture that is small enough to keep in
the repository, and a few have no public evaluation at all. The ACE side solved
that with `make_urr_ace.py` and friends; this is the same idea for ENDF, shared
by the `make_*_endf.py` tools so each of those is only the data it is about.

Nothing here validates: the point is to write exactly the bytes the format
specifies, including the fields evaluators overload, so both readers are held
to the same interpretation of them.
"""

from __future__ import annotations

from pathlib import Path


def endf_float(value: float) -> str:
    """An 11-character ENDF float, e.g. ``' 9.223500+4'``.

    The format has no room for the ``e``, so the exponent is written bare. Six
    decimal places fit a two-digit exponent; a three-digit one costs a place.
    """
    if value == 0.0:
        return " 0.000000+0"
    digits, exponent = f"{value:.6E}".split("E")
    power = int(exponent)
    sign = "+" if power >= 0 else "-"
    text = f"{digits}{sign}{abs(power)}"
    if len(text) > 11:
        digits = f"{value:.5E}".split("E")[0]
        text = f"{digits}{sign}{abs(power)}"
    return text.rjust(11)


class Section:
    """Accumulates the records of one section, numbering them as it goes."""

    def __init__(self, mat: int, mf: int, mt: int) -> None:
        self.mat = mat
        self.mf = mf
        self.mt = mt
        self.lines: list[str] = []

    def record(self, body: str) -> None:
        n = (len(self.lines) + 1) % 100000
        self.lines.append(f"{body:<66}{self.mat:>4}{self.mf:>2}{self.mt:>3}{n:>5}")

    def cont(self, c1: float, c2: float, l1: int, l2: int, n1: int, n2: int) -> None:
        """A CONT record, which HEAD, LIST, TAB1 and TAB2 all open with."""
        self.record(f"{endf_float(c1)}{endf_float(c2)}{l1:>11}{l2:>11}{n1:>11}{n2:>11}")

    def floats(self, values: list[float]) -> None:
        """Six floats to a line, as every list of values is written."""
        for i in range(0, len(values), 6):
            self.record("".join(endf_float(v) for v in values[i : i + 6]))

    def pairs(self, values: list[int]) -> None:
        """Six integers to a line, for the interpolation ranges of TAB1/TAB2."""
        for i in range(0, len(values), 6):
            self.record("".join(f"{v:>11}" for v in values[i : i + 6]))

    def text(self, body: str) -> None:
        self.record(f"{body:<66}")

    def list_record(
        self,
        c1: float,
        c2: float,
        l1: int,
        l2: int,
        n2: int,
        values: list[float],
    ) -> None:
        """A LIST record: a CONT whose N1 is the value count, then the values."""
        self.cont(c1, c2, l1, l2, len(values), n2)
        self.floats(values)

    def tab1(
        self,
        c1: float,
        c2: float,
        l1: int,
        l2: int,
        x: list[float],
        y: list[float],
        interpolation: int = 2,
    ) -> None:
        """A TAB1 record with a single interpolation range."""
        self.cont(c1, c2, l1, l2, 1, len(x))
        self.pairs([len(x), interpolation])
        interleaved: list[float] = []
        for xi, yi in zip(x, y):
            interleaved += [xi, yi]
        self.floats(interleaved)

    def tab2(
        self,
        c1: float,
        c2: float,
        l1: int,
        l2: int,
        n2: int,
        interpolation: int = 2,
    ) -> None:
        """A TAB2 record: the interpolation across the outer variable."""
        self.cont(c1, c2, l1, l2, 1, n2)
        self.pairs([n2, interpolation])

    def finish(self) -> list[str]:
        """The section's records plus its SEND terminator."""
        send = f"{'':<66}{self.mat:>4}{self.mf:>2}{0:>3}{99999:>5}"
        return self.lines + [send]


def terminator(mat: int) -> str:
    """A FEND, MEND or TEND record; which one is decided by `mat`."""
    return f"{'':<66}{mat:>4}{0:>2}{0:>3}{0:>5}"


def descriptive(
    mat: int,
    za: float,
    awr: float,
    nsub: int,
    directory: list[tuple[int, int, int]],
    zsymam: str,
    lines: list[str],
    *,
    author: str = "endf-python tools",
    lrp: int = -1,
    lfi: int = 0,
    emax: float = 2.0e7,
    awi: float = 1.0,
) -> list[str]:
    """MF=1 MT=451, the section every material must have.

    The first two text records are read by column rather than by whitespace, so
    they are laid out here the way a real evaluation lays them out: ZSYMAM,
    ALAB, EDATE and AUTH on the first, then REF, DDATE, RDATE and ENDATE.
    """
    s = Section(mat, 1, 451)
    s.cont(za, awr, lrp, lfi, 0, 0)
    s.cont(0.0, 1.0 if lfi else 0.0, 0, 0, 0, 6)
    s.cont(awi, emax, 0, 0, nsub, 8)
    s.cont(0.0, 0.0, 0, 0, len(lines) + 2, len(directory))
    s.text(f"{zsymam:<11}{'SYNTH':<11}{'EVAL-JAN26':<10} {author:<33}")
    s.text(f" {'synthetic, not real':<21}{'DIST-JAN26':<10} {'':<22}{'20260101':<8}")
    for line in lines:
        s.text(line)
    for mf, mt, count in directory:
        s.record(f"{'':>22}{mf:>11}{mt:>11}{count:>11}{0:>11}")
    return s.finish()


def write_material(target: Path, mat: int, sections: list[list[str]]) -> None:
    """Wrap sections in the TPID/FEND/MEND/TEND records and write the file.

    Sections must arrive grouped by MF and in ascending order, since one FEND
    is written per file.
    """
    out = [terminator(1)]  # TPID
    previous_mf = None
    for section in sections:
        mf = int(section[0][70:72])
        if previous_mf is not None and mf != previous_mf:
            out.append(terminator(mat))  # FEND, closing the previous file
        out += section
        previous_mf = mf
    out.append(terminator(mat))  # FEND, closing the last file
    out.append(terminator(0))  # MEND
    out.append(terminator(-1))  # TEND
    target.write_text("\n".join(out) + "\n")
