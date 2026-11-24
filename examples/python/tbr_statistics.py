"""
Tritium-breeding (TBR) cell tallies, showing the full statistical-reliability
output that yamc now reports for every tally:

    mean +/- std, variance, figure of merit, variance of the variance,
    skewness, kurtosis, the large-score PDF tail slope, the convergence
    history (statistics vs number of histories), and a PASS/FAIL verdict.

Just printing a tally result now shows all of it. Two tallies are run to
contrast convergence: the total TBR (well converged) and the high-energy
(>10 MeV) contribution only, which scores far fewer events and converges
less well -- so the reliability checks have something to flag.
"""

import matplotlib.pyplot as plt

import yamc

# --- Geometry: point fusion source inside a Li/Be breeder sphere ---
inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0, boundary="vacuum")

breeder = yamc.Material(
    composition={"Li6": 0.07 / 2, "Li7": 0.93 / 2, "Be9": 0.5},
    density=2.0,
    temperature=294,
)
breeder.read_nuclear_data(
    {
        "Be9": "tests/Be9.arrow",
        "Li6": "tests/Li6.arrow",
        "Li7": "tests/Li7.arrow",
    }
)

cell_void = yamc.Cell(name="void", region=inner.below)
cell_breeder = yamc.Cell(
    name="breeder", region=inner.above & outer.below, material=breeder
)
geometry = yamc.Geometry([cell_void, cell_breeder])

source = yamc.NeutronSource(
    energy=yamc.sources.fusion_neutron_spectrum(20000.0),
    position=(0, 0, 0),
)

# Tally 1: total tritium production -- well converged.
tbr_total = yamc.Tally(
    name="TBR_total",
    cells=cell_breeder,
    scores=["H3-production"]
)

# Tally 2: tritium bred only by the rare highest-energy neutrons in the
# source's upper tail (>15.2 MeV). Very few neutrons reach these energies
# (a single high-energy bin acts as the filter), so this contribution is
# sparsely sampled, converges poorly, and the reliability checks flag it --
# the deliberate contrast with the well-converged total above.
tbr_tail = yamc.Tally(
    name="TBR_high_energy",
    cells=cell_breeder,
    scores=["H3-production"],
    energy_bins=[1.52e7, 3.0e7],
)

model = yamc.Model(
    geometry=geometry,
    tallies=[tbr_total, tbr_tail],
    source=source,
)
results = model.simulate_transport(total_particles=2000000, seed=1)


def report(r):
    # Printing the tally shows the full reliability summary.
    print()
    print(r)
    print("  programmatic:")
    print(f"    variance (per bin) . . . . {r.variance}")
    print(f"    aggregate rel. error . . . {r.aggregate_relative_error:.4%}")
    print(f"    variance of the variance . {r.aggregate_variance_of_variance:.4e}")
    print(
        f"    skewness / kurtosis  . . . {r.aggregate_skewness:.3f} / {r.aggregate_kurtosis:.3f}"
    )
    print(f"    large-score tail slope . . {r.aggregate_tail_slope:.3f}")
    # Convergence history: statistics versus number of histories.
    print("  convergence (rel.err / FOM vs N):")
    for p in r.convergence_history:
        print(
            f"    N={p.n_histories:>8}  rel.err={p.relative_error:.3%}  FOM={p.figure_of_merit:.2e}"
        )
    pdf = r.score_pdf
    populated = sum(1 for c in pdf.counts if c > 0)
    print(
        f"  empirical PDF: {populated} populated bins, "
        f"{pdf.zero} zero-score histories, tail slope {pdf.tail_slope:.2f}"
    )


# report(results["TBR_total"])
# report(results["TBR_high_energy"])

# --- Plot the empirical history-score PDF and CDF side by side ---
fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(13, 5))
for key in ("TBR_total", "TBR_high_energy"):
    pdf = results[key].score_pdf
    pts = [(c, n) for c, n in zip(pdf.bin_centers, pdf.counts) if n > 0]
    if not pts:
        continue
    xs = [c for c, _ in pts]
    ys = [n for _, n in pts]
    label = f"{key} (tail slope {pdf.tail_slope:.1f})"
    # PDF: counts per log-spaced magnitude bin.
    ax_pdf.step(xs, ys, where="mid", marker="o", label=label)
    # CDF: cumulative fraction of histories with score <= x. The zero-score
    # histories form the floor at the left.
    total = sum(pdf.counts) + pdf.zero
    running = pdf.zero
    cdf = []
    for _, n in pts:
        running += n
        cdf.append(running / total)
    ax_cdf.step(xs, cdf, where="post", marker="o", label=label)

ax_pdf.set_xscale("log")
ax_pdf.set_yscale("log")
ax_pdf.set_xlabel("per-history score magnitude (H3 / source neutron)")
ax_pdf.set_ylabel("number of histories")
ax_pdf.set_title("Empirical history-score PDF (H3-production)")
ax_pdf.legend()

ax_cdf.set_xscale("log")
ax_cdf.set_xlabel("per-history score magnitude (H3 / source neutron)")
ax_cdf.set_ylabel("cumulative fraction of histories")
ax_cdf.set_title("Empirical history-score CDF (H3-production)")
ax_cdf.set_ylim(0, 1.02)
ax_cdf.legend()

fig.tight_layout()
# plt.show()
