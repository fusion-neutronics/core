"""The converters must not hold the GIL while they work.

A library build converts thousands of evaluations, and NJOY alone runs for
minutes over some of them. Held for the length of a call, the GIL stops every
other thread in the interpreter, so a thread pool over evaluations converts
them one at a time.

Timed, because nothing else is observable from Python: a conversion runs in a
worker thread while this one keeps reading the clock, and the readings must not
stop for the length of the conversion. The fixtures convert in milliseconds,
too short to time, so another process holds each call inside Rust for `HOLD`
seconds whatever the machine: NJOY is a shell script that sleeps, and every
other input is a pipe that a second process fills `HOLD` seconds after the
converter opens it. Both need POSIX, so Windows skips this file; the bindings
it guards are the same on every platform.
"""

import lzma
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import yamc

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-ins are a POSIX pipe and a shell script"
)

FIXTURES = Path(__file__).resolve().parents[3] / "crates" / "endf" / "fixtures"

#: How long each conversion is held inside Rust, in seconds.
HOLD = 0.3

#: Fills a pipe: once the converter opens it, waits HOLD, writes the file and
#: closes, then serves the next open the same way. So a converter that read its
#: input twice would be held twice rather than hang, even one that stopped part
#: way through the first time: only `cat` sees the broken pipe, and the loop
#: goes on. The pause before serving again lets the reader see the end first:
#: reopened at once, the pipe would hand it the file again.
FEED = 'while :; do { sleep "$3"; cat "$1"; } > "$2"; sleep 1; done'

FE56 = "n-026_Fe_056_trimmed.endf.xz"
NB93 = "n-041_Nb_093_tendl2017_trimmed.endf.xz"
NB_DECAY = [
    "dec-041_Nb_092.endf.xz",
    "dec-041_Nb_092m1.endf.xz",
    "dec-041_Nb_093m1.endf.xz",
]


@pytest.fixture
def held(tmp_path):
    """Inputs for one conversion, and the stand-ins that hold it up."""
    writers = []

    def plain(name):
        target = tmp_path / name.removesuffix(".xz")
        target.write_bytes(lzma.decompress((FIXTURES / name).read_bytes()))
        return str(target)

    def late(path):
        pipe = f"{path}.pipe"
        os.mkfifo(pipe)
        feed = ["sh", "-c", FEED, "feed", path, pipe, str(HOLD)]
        writers.append(subprocess.Popen(feed))
        return pipe

    def photonuclear():
        # The Li6 neutron evaluation with NSUB set to 0, as
        # test_convert_photonuclear.py builds it: no photonuclear fixture exists.
        path = Path(plain("n-003_Li_006_trimmed.endf.xz"))
        lines = path.read_text().splitlines()
        header = next(i for i, line in enumerate(lines) if line[70:75] == " 1451")
        line = lines[header + 2]
        lines[header + 2] = f"{line[:44]}{0:11}{line[55:]}"
        path.write_text("\n".join(lines))
        return str(path)

    njoy = tmp_path / "njoy"
    njoy.write_text(f"#!/bin/sh\ncat > /dev/null\nsleep {HOLD}\nexit 1\n")
    njoy.chmod(0o755)

    yield SimpleNamespace(
        plain=plain,
        late=late,
        photonuclear=photonuclear,
        njoy=str(njoy),
        out=str(tmp_path / "out"),
    )
    for writer in writers:
        writer.kill()
        writer.wait()


def _in_njoy(convert):
    """The stand-in NJOY exits 1 once it has slept, and the conversion says so."""

    def run():
        with pytest.raises(RuntimeError, match="NJOY exited"):
            convert()

    return run


#: Each binding in crates/yani-python/src/convert.rs, as a call held up by a
#: stand-in; a binding added there belongs here too. Built on the main thread,
#: so the worker only converts.
CASES = {
    "convert_neutron_xs": lambda h: _in_njoy(
        partial(yamc.convert_neutron_xs, h.plain(FE56), h.out, njoy_exec=h.njoy)
    ),
    "convert_neutron_transport": lambda h: _in_njoy(
        partial(yamc.convert_neutron_transport, h.plain(FE56), h.out, njoy_exec=h.njoy)
    ),
    "convert_transmutation": lambda h: partial(
        yamc.convert_transmutation,
        decay_files=[h.late(h.plain("dec-055_Cs_137.endf.xz"))],
        fpy_files=[],
        neutron_files=[],
        output_path=h.out,
        subsections=["decay"],
    ),
    "convert_branching": lambda h: partial(
        yamc.convert_branching,
        neutron_files=[h.late(h.plain(NB93))],
        decay_files=[h.plain(name) for name in NB_DECAY],
        output_path=h.out,
    ),
    "radionuclide_production": lambda h: partial(
        yamc.radionuclide_production, [h.late(h.plain(NB93))]
    ),
    "convert_photon": lambda h: partial(
        yamc.convert_photon, h.late(h.plain("photoat-001_H_000.endf.xz")), h.out
    ),
    "convert_photonuclear": lambda h: partial(
        yamc.convert_photonuclear, h.late(h.photonuclear()), h.out
    ),
}


def _longest_stall(convert):
    """Run `convert` in a worker thread while this one keeps reading the clock.

    Returns the longest gap between two readings, and how long the conversion
    took. The clock starts before the worker does: starting a thread waits for
    it to begin, and a worker that takes the GIL straight into a conversion
    would otherwise stop this thread before its first reading, hiding the
    stall it is here to see. For the same reason the clock is read once more
    after the conversion is seen to be done, so a stall that lasts until the
    end is counted too.
    """

    def timed():
        start = time.perf_counter()
        convert()
        return time.perf_counter() - start

    longest = 0.0
    last = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(timed)
        done = False
        while not done:
            done = future.done()
            now = time.perf_counter()
            longest = max(longest, now - last)
            last = now
    return longest, future.result()


@pytest.mark.parametrize("binding", CASES)
def test_the_main_thread_keeps_running_during_a_conversion(binding, held):
    """Holding the GIL stops this thread for the whole conversion.

    Released, it stops only for thread switches: under 17 ms in 16 runs of all
    seven, pinned to one core beside a busy loop, against at least 93% of the
    call when held.
    """
    stall, duration = _longest_stall(CASES[binding](held))
    assert duration >= HOLD, (
        f"{binding} took {duration * 1e3:.0f} ms, less than the "
        f"{HOLD * 1e3:.0f} ms its stand-in holds it for, so it was never held "
        "and this measured nothing"
    )
    assert stall < duration / 2, (
        f"the main thread stopped for {stall * 1e3:.0f} ms of a "
        f"{duration * 1e3:.0f} ms {binding}, which is what holding the GIL "
        "through it looks like"
    )


def test_every_converter_binding_has_a_case():
    """A binding added to convert.rs without a case here would go unchecked."""
    source = (
        Path(__file__).resolve().parents[3]
        / "crates"
        / "yani-python"
        / "src"
        / "convert.rs"
    ).read_text()
    bindings = set(re.findall(r"#\[pyfunction\][^{]*?pub fn (\w+)\(", source))
    assert bindings, "no #[pyfunction] found in convert.rs, so this checks nothing"
    assert bindings == set(CASES)
