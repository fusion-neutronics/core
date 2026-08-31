"""Turn ``bench_transmute.json`` into the running table and the graph.

``tools/bench_transmute.py`` appends one record per improvement; this reads them
in the order they were measured and produces:

* a markdown table of seconds and speedups, on stdout, and
* ``transmute_speed.png``: wall time (log scale, because the uncertainty cases
  are several times the plain ones and a linear axis would flatten the plain
  ones into a line) beside cumulative speedup against the first record.

Usage::

    python tools/plot_transmute_bench.py
    python tools/plot_transmute_bench.py --in bench_transmute.json --out transmute_speed.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: The four cases, in the order they are drawn and legended.
SERIES = [
    ("once", "once"),
    ("repeat", "repeat"),
    ("once_unc", "once, uncertainty"),
    ("repeat_unc", "repeat, uncertainty"),
]

#: Categorical slots 1-4. Fixed order, assigned per series and never cycled, so
#: a case keeps its colour however many records there are.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"


def markdown_table(records: list[dict]) -> str:
    base = records[0]
    head = "| step | " + " | ".join(label for _, label in SERIES) + " |"
    rule = "|---" * (len(SERIES) + 1) + "|"
    lines = [head, rule]
    for record in records:
        cells = []
        for key, _ in SERIES:
            seconds = record[key]
            speedup = base[key] / seconds if seconds else float("nan")
            cells.append(f"{seconds:.2f} s ({speedup:.2f}x)")
        lines.append(f"| {record['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(records: list[dict], out: Path) -> None:
    base = records[0]
    x = list(range(len(records)))
    labels = [r["label"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)

    for (key, label), color in zip(SERIES, COLORS):
        seconds = [r[key] for r in records]
        speedup = [base[key] / s if s else float("nan") for s in seconds]
        for ax, values in ((axes[0], seconds), (axes[1], speedup)):
            ax.plot(
                x,
                values,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=label,
                zorder=3,
            )
        # Direct labels, mandatory at four series: two of these hues sit below
        # 3:1 against the surface, so identity must not rest on colour alone.
        # In text ink rather than the series colour, with the marker beside them
        # carrying the identity.
        # Two significant figures, so 0.13 s does not render as "0.1 s" beside a
        # 16 s baseline.
        last = seconds[-1]
        axes[0].annotate(
            f"{label}  {last:.2f} s" if last < 1.0 else f"{label}  {last:.1f} s",
            (x[-1], last),
            textcoords="offset points",
            xytext=(9, 0),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
        axes[1].annotate(
            f"{label}  {speedup[-1]:.1f}x",
            (x[-1], speedup[-1]),
            textcoords="offset points",
            xytext=(9, 0),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel("wall time (s, log scale)", color=INK_SECONDARY, fontsize=10)
    axes[0].set_title(
        "Material.transmute() on a steel, CCFE-709",
        color=INK,
        fontsize=13,
        loc="left",
        pad=14,
    )
    axes[1].axhline(1.0, color=GRID, linewidth=1.5, zorder=1)
    axes[1].set_ylabel(f"speedup vs {base['label']}", color=INK_SECONDARY, fontsize=10)
    axes[1].set_title(
        "cumulative speedup", color=INK, fontsize=13, loc="left", pad=14
    )

    # Room on the right for the direct labels.
    for ax in axes:
        ax.set_xlim(-0.4, len(records) - 1 + 0.45 * len(records))
    axes[0].legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        loc="upper right",
        ncols=2,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", default="bench_transmute.json")
    parser.add_argument("--out", default="transmute_speed.png")
    args = parser.parse_args()

    records = json.loads(Path(args.source).read_text())
    if not records:
        raise SystemExit(f"{args.source} has no records")

    print(markdown_table(records))
    plot(records, Path(args.out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
