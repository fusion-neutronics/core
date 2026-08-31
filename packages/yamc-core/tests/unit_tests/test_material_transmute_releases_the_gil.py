"""``Material.transmute()`` must not hold the GIL for the length of the solve.

``Model.transmute`` has released it for as long as it has existed;
``Material.transmute`` was the outlier, so one call froze the whole interpreter
and a Python-level ``ThreadPoolExecutor`` over cases could not overlap them at
all (issue #576, finding 7).

Timed rather than inspected, because there is nothing to inspect from Python:
the observable consequence is that two solves in two threads take about as long
as one, and that a plain Python thread keeps running while a solve is in
progress.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
RATE = sum(MULTIGROUP_FLUX)

#: Enough steps that one solve takes long enough to time (~0.2 s here) without
#: making the suite slow. The nominal step loop is a CRAM48 per step and is
#: deliberately never parallelised -- the state of one step is the next step's
#: input -- so this stays a real duration however much of the rest of the solve
#: later grows threads.
STEPS = 3000


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )


def _schedule():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.PulseSchedule(
        [yamc.Pulse(rate=RATE, duration=DAY, source=spectrum) for _ in range(STEPS)]
    )


def _one_solve():
    material = _iron()
    schedule = _schedule()
    # Warm: the first call loads the nuclear data, which is not what is being
    # timed and would swamp it.
    material.transmute(schedule=schedule)

    start = time.perf_counter()
    material.transmute(schedule=schedule)
    return time.perf_counter() - start, material, schedule


def test_a_python_thread_keeps_running_during_a_solve():
    """The direct statement: another thread makes progress while a solve runs.

    If the GIL were held for the solve, the counter would not move until it
    finished, so the count at the end would be whatever a single scheduling
    slice managed.
    """
    solo, material, schedule = _one_solve()
    if solo < 0.05:
        pytest.skip(f"one solve is only {solo * 1e3:.1f} ms; too short to time")

    ticks = 0
    stop = threading.Event()

    def counter():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    thread = threading.Thread(target=counter)
    thread.start()
    try:
        material.transmute(schedule=schedule)
    finally:
        stop.set()
        thread.join()

    # A 1 ms sleep means the counter can tick at most ~1000 times a second, and
    # the GIL is handed over at every sleep, so a solve of `solo` seconds should
    # see hundreds of them. Asserted loosely -- the point is "many, not none".
    assert ticks > 10, (
        f"the counter thread ticked {ticks} times during a {solo * 1e3:.0f} ms "
        "solve, which is what holding the GIL through it looks like"
    )


#: How many times each timing below is repeated, taking the fastest. A timing
#: floor is the meaningful statistic on a shared runner: noise only ever makes a
#: measurement slower, so the minimum is the one that reflects the machine.
REPEATS = 3


def _fastest(measure):
    """The quickest of `REPEATS` runs of `measure`."""
    return min(measure() for _ in range(REPEATS))


def _usable_cpus():
    """Cores this process may actually run on, not what the box advertises.

    `sched_getaffinity` is what a container or a `taskset` restricts, and it is
    Linux-only; `cpu_count` is the portable fall-back.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def test_two_solves_in_two_threads_overlap():
    """And the consequence: two of them beat running them back to back.

    Measured against a serial baseline taken on the SAME machine rather than
    against a multiple of one solve. The two are not equivalent: a fixed
    multiple of `solo` bakes in how much of a call is GIL-held, which is a
    property of the machine and the build, so a threshold calibrated on a fast
    many-core developer box does not survive a small shared CI runner. This
    lands at 0.69 of serial on a 32-core box and 0.84 on a 2-core CI runner,
    where the old `together < 1.6 * solo` form read 1.38 and 1.66 against a
    threshold of 1.60 -- passing and failing for reasons that had nothing to do
    with the GIL (issue #344 is the same class of defect, and this test had
    never once run in CI to show it).

    Serialized on the GIL, two threads can do no better than back to back, so
    `together` would land at or above `serial`. Overlapped, it is below it.
    """
    cpus = _usable_cpus()
    if cpus < 2:
        # Two threads cannot overlap on one core whatever the GIL does, so this
        # would fail for a reason it is not about. Measured: pinned to a single
        # core it reads 1.08 of serial, against 0.68 on two.
        pytest.skip(f"only {cpus} usable core; two solves cannot overlap")

    solo, _, _ = _one_solve()
    if solo < 0.05:
        pytest.skip(f"one solve is only {solo * 1e3:.1f} ms; too short to time")

    # Built and warmed outside the timed region, so what is timed is the solve
    # and not the nuclear-data load that would swamp it. Separate materials
    # because the solve loads into the one it is given.
    cases = []
    for _ in range(2):
        material, schedule = _iron(), _schedule()
        material.transmute(schedule=schedule)
        cases.append((material, schedule))

    def back_to_back():
        start = time.perf_counter()
        for material, schedule in cases:
            material.transmute(schedule=schedule)
        return time.perf_counter() - start

    def in_two_threads():
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda case: case[0].transmute(schedule=case[1]), cases))
        return time.perf_counter() - start

    serial = _fastest(back_to_back)
    together = _fastest(in_two_threads)

    assert together < 0.9 * serial, (
        f"two threads took {together:.3f} s where the same two solves back to "
        f"back take {serial:.3f} s ({together / serial:.2f} of serial); "
        "overlapping should beat it, and not doing so is what serializing on "
        "the GIL looks like"
    )
