"""Wall-clock benchmark for ``Material.transmute`` on a steel, plus a golden
inventory dump that pins the answer while the speed moves.

Four numbers, which are the four shapes a caller actually meets (issue #576):

===============  ==================================================
``once``         the first ``transmute`` in a fresh interpreter, so
                 it carries the Arrow decode of every reachable
                 chain nuclide
``repeat``       the median of the calls after it, each one dropping
                 the previous results object, which is what a sweep
                 over compositions does
``once_unc``     the same first call with ``data_uncertainty=``, so
                 it also carries the fold, the factorization and the
                 replica ensemble
``repeat_unc``   the median of the uncertainty calls after it
===============  ==================================================

The plain and the uncertainty cases run in separate interpreters, because
"first call" means nothing once a previous case has warmed the global nuclide
cache.

Alongside every timing it writes the final inventory as **hex floats**. Decimal
printing hides the last bits, which is exactly what a "no accuracy loss" claim
is about, so the comparison the speed work is judged against is a byte
comparison of that dump.

Usage::

    python tools/bench_transmute.py --label baseline
    python tools/bench_transmute.py --label "finding 1a" --compare baseline

Results accumulate in ``bench_transmute.json`` (``--out`` to move it), one
record per label, so the table and the plot can be regenerated at any point
from the file alone::

    python tools/plot_transmute_bench.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --- the case ----------------------------------------------------------------

#: Nuclear-data library. Everything here resolves through the same one, so a
#: run is reproducible from the cache without touching the network twice.
LIBRARY = os.environ.get("YANI_BENCH_LIBRARY", "endf-b8.1")

#: A 709-group structure is where the collapse cost actually lives; the coarse
#: structures the regression tests use hide it.
GROUPS = "CCFE-709"

#: SS316-like, by mass. Six elements is enough to pull a realistic reachable
#: set out of the chain (~430 nuclides on ENDF/B-8.1) without the run being
#: dominated by one long grid.
STEEL = {
    "Fe": 0.65,
    "Cr": 0.17,
    "Ni": 0.12,
    "Mo": 0.025,
    "Mn": 0.02,
    "Si": 0.01,
}
DENSITY = 7.9  # g/cm3
TEMPERATURE = 294.0

#: Total flux magnitude [n/cm2/s] and the schedule shape.
RATE = 1.0e14
FLUX_RELATIVE_SIGMA = 0.05

#: Fixed rather than adaptive: an adaptive count makes the replica loop's cost
#: depend on how quickly the sigmas settle, which is not what is being timed.
SAMPLES = 128

#: How many calls follow the first one. The plain case is cheap enough to
#: repeat properly; the uncertainty case is not.
REPEATS = 5
REPEATS_UNC = 3


def fusion_spectrum(boundaries: list[float]) -> list[float]:
    """A 1/E slowing-down tail with a 14 MeV peak on top.

    Deterministic, and non-trivial in the way that matters here: every group
    carries flux, so nothing in the collapse can be skipped as a zero, and the
    peak sits where the threshold reactions are.
    """
    flux = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mid = math.sqrt(max(lo, 1.0e-5) * hi)
        value = (hi - lo) / mid
        if 1.3e7 < mid < 1.5e7:
            value += 50.0
        flux.append(value)
    return flux


# --- the child process, one case per interpreter ------------------------------


def run_case(uncertainty: bool) -> dict:
    """Time one case and dump its inventory. Runs in its own interpreter."""
    import yani

    yani.cross_section_data = LIBRARY
    for setting in (
        "transmutation_decay_data",
        "transmutation_reactions",
        "transmutation_fission_yields",
        "transmutation_branch_ratios",
    ):
        setattr(yani, setting, LIBRARY)

    boundaries = yani.group_structure(GROUPS)
    flux = fusion_spectrum(boundaries)
    source = yani.NeutronSource(energy=yani.sources.Histogram(GROUPS, flux))

    pulse_kwargs = {"rate": RATE, "duration": (1, "h"), "source": source}
    if uncertainty:
        pulse_kwargs["flux_std_dev"] = [FLUX_RELATIVE_SIGMA * f for f in flux]
    schedule = yani.PulseSchedule([
        yani.Pulse(**pulse_kwargs),
        yani.Cooldown(duration=(1, "h")),
    ])
    request = yani.DataUncertainty(seed=1, samples=SAMPLES) if uncertainty else None

    def steel():
        return yani.Material(
            composition=dict(STEEL),
            density=DENSITY,
            fraction_type="mass",
            name="steel",
            volume=1.0,
            temperature=TEMPERATURE,
        )

    def one_call(material):
        start = time.perf_counter()
        results = material.transmute(schedule=schedule, data_uncertainty=request)
        return time.perf_counter() - start, results

    # The first call in this interpreter: nothing is cached, so this is the
    # number a script that transmutes one material sees.
    material = steel()
    cold, results = one_call(material)
    golden = inventory_of(results, material, uncertainty)
    del results

    # And calling it again on the same material, with the previous results
    # dropped each time. The same material on purpose: "transmute this steel
    # again" is the shape a sweep over schedules, fluxes or seeds has, and the
    # results object is dropped because holding every one of them is not.
    #
    # Until issue #576's finding 3 this made no difference either way --
    # `transmute` did not touch the material it was given, so a repeat call
    # re-decoded every reachable nuclide's Arrow directory whatever the caller
    # kept alive.
    warm = []
    for _ in range(REPEATS_UNC if uncertainty else REPEATS):
        elapsed, results = one_call(material)
        del results
        warm.append(elapsed)

    return {
        "once": cold,
        "repeat": statistics.median(warm),
        "repeat_all": warm,
        "golden": golden,
    }


def inventory_of(results, material, uncertainty: bool) -> dict:
    """The final inventory as hex floats, and the sigmas beside it.

    ``float.hex`` is exact, so this dump changes if and only if a bit does.
    """
    material_id = material.id or 0
    step = results.num_steps
    densities = results.get_material_nuclides(material_id, step) or {}
    out = {"densities": {n: float(v).hex() for n, v in sorted(densities.items())}}
    if uncertainty:
        sigmas = {}
        for name in sorted(densities):
            sigma = results.get_nuclide_uncertainty(material_id, name, step)
            if sigma is not None:
                sigmas[name] = float(sigma).hex()
        out["sigmas"] = sigmas
        info = results.data_uncertainty_info or {}
        out["samples"] = info.get("samples")
        out["converged"] = info.get("converged")
    return out


# --- the parent process -------------------------------------------------------


def child(case: str) -> int:
    payload = run_case(uncertainty=(case == "uncertainty"))
    sys.stdout.write("\n@@BENCH@@" + json.dumps(payload) + "\n")
    return 0


def spawn(case: str) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--case", case],
        capture_output=True,
        text=True,
        check=False,
    )
    marker = "@@BENCH@@"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise SystemExit(
        f"the {case} case produced no result\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("plain", "uncertainty"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help="name this measurement in the results file")
    parser.add_argument(
        "--out",
        default="bench_transmute.json",
        help="results file to append to (default: bench_transmute.json)",
    )
    parser.add_argument(
        "--compare",
        help="an earlier label whose golden inventory this one must reproduce bit for bit",
    )
    args = parser.parse_args()

    if args.case:
        return child(args.case)
    if not args.label:
        parser.error("--label is required")

    plain = spawn("plain")
    unc = spawn("uncertainty")

    record = {
        "label": args.label,
        "once": plain["once"],
        "repeat": plain["repeat"],
        "once_unc": unc["once"],
        "repeat_unc": unc["repeat"],
        "repeat_all": plain["repeat_all"],
        "repeat_unc_all": unc["repeat_all"],
        "golden": plain["golden"],
        "golden_unc": unc["golden"],
        "host": {
            "cpus": os.cpu_count(),
            "platform": platform.platform(),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        },
    }

    path = Path(args.out)
    records = json.loads(path.read_text()) if path.exists() else []
    records = [r for r in records if r["label"] != args.label]
    records.append(record)
    path.write_text(json.dumps(records, indent=1) + "\n")

    print(f"{'case':<14}{'seconds':>10}")
    for key in ("once", "repeat", "once_unc", "repeat_unc"):
        print(f"{key:<14}{record[key]:>10.3f}")

    if args.compare:
        earlier = next((r for r in records if r["label"] == args.compare), None)
        if earlier is None:
            raise SystemExit(f"no record labelled {args.compare!r} in {path}")
        return report_golden_diff(earlier, record)
    return 0


def report_golden_diff(earlier: dict, record: dict) -> int:
    """Compare two golden dumps bit for bit and say what moved."""
    bad = 0
    for key in ("golden", "golden_unc"):
        for section in ("densities", "sigmas"):
            was = earlier[key].get(section, {})
            now = record[key].get(section, {})
            if was == now:
                continue
            bad = 1
            names = sorted(set(was) | set(now))
            differing = [n for n in names if was.get(n) != now.get(n)]
            print(
                f"\n{key}.{section}: {len(differing)} of {len(names)} entries moved"
                f" against {earlier['label']!r}"
            )
            for name in differing[:20]:
                print(f"  {name:<12} {was.get(name)} -> {now.get(name)}")
    if bad:
        print("\nNOT bit-identical")
    else:
        print(f"\nbit-identical to {earlier['label']!r}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
