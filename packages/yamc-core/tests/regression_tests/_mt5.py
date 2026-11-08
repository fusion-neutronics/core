"""MT=5, ``(n,X)``, for the transmutation regression tests.

The contribution of MT=5 is checked independently of stored reference values.
The tests run yamc with and without the ``(n,X)`` reaction rows, then check
the difference against what MT=5 makes, computed here from the data alone
(:func:`predicted_mt5_densities`): the MT=5 cross
  section of the seed's own neutron data times each ``(n,X)`` row's curve,
  group-averaged flat in energy over the test spectrum (yamc's default
  within-group weight), and carried through the schedule with each product's
  own decay.

The prediction is first order in the flux: the parent is not depleted (its
removal over the schedule is below 1e-6 of it here) and MT=5's products are
not irradiated further. A product fed by the decay of another MT=5 product
whose half-life is neither negligible against an irradiation step nor long
against the schedule is not predicted at that step; see
:func:`predicted_mt5_densities`.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

ANYTHING = "(n,X)"
MT_ANYTHING = 5
BARN_CM2 = 1.0e-24

# A product with a half-life below this share of the shortest irradiation
# step is treated as decaying the moment it is made.
INSTANT_FRACTION = 1.0e-3


def _read(path):
    """Read an Arrow IPC file or stream; return the table and whether it was
    the file format."""
    try:
        with ipc.open_file(path) as reader:
            return reader.read_all(), True
    except pa.ArrowInvalid:
        with ipc.open_stream(path) as reader:
            return reader.read_all(), False


def _write(path, table, file_format):
    sink = ipc.new_file if file_format else ipc.new_stream
    with pa.OSFile(str(path), "wb") as out, sink(out, table.schema) as writer:
        writer.write_table(table)


def write_without_anything(chain_dir):
    """Take the ``(n,X)`` rows out of an exported chain's reactions subsection.

    The remaining reactions provide the baseline for the independent MT=5
    contribution check. The MT=5 columns stay, null on every row left.
    """
    path = Path(chain_dir) / "reactions" / "reactions.arrow"
    table, file_format = _read(path)
    keep = pc.not_equal(table.column("type"), ANYTHING)
    _write(path, table.filter(keep), file_format)


def anything_rows(chain_dir, parent):
    """The ``(n,X)`` rows of ``parent`` with a product: (target, energy,
    multiplicity, share)."""
    table, _ = _read(Path(chain_dir) / "reactions" / "reactions.arrow")
    rows = []
    for row in table.to_pylist():
        if row["nuclide"] != parent or row["type"] != ANYTHING or row["target"] is None:
            continue
        rows.append(
            (
                row["target"],
                np.asarray(row["energy"], dtype=float),
                np.asarray(row["multiplicity"], dtype=float),
                bool(row["share"]),
            )
        )
    return rows


def mt5_cross_section(fixture_dir, temperature):
    """MT=5 of a neutron fixture on its energy grid, or None if it has none."""
    fixture_dir = Path(fixture_dir)
    energies, _ = _read(fixture_dir / "energy.arrow")
    grid = None
    for row in energies.to_pylist():
        if row["temperature"] == temperature:
            grid = np.asarray(row["energy_values"], dtype=float)
    if grid is None:
        raise ValueError(f"{fixture_dir} has no energy grid at {temperature}")
    reactions, _ = _read(fixture_dir / "reactions.arrow")
    for row in reactions.to_pylist():
        if row["mt"] != MT_ANYTHING or temperature not in row["xs_temperatures"]:
            continue
        k = row["xs_temperatures"].index(temperature)
        values = np.asarray(row["xs_values"][k], dtype=float)
        start = int(row["xs_threshold_idx"][k])
        sigma = np.zeros_like(grid)
        sigma[start:start + len(values)] = values
        return grid, sigma
    return None


def _curve(energy, values, at):
    """A curve as the branching rule reads it: zero below its first point,
    linear between points, flat above its last."""
    return np.interp(at, energy, values, left=0.0, right=values[-1])


def mt5_production_rates(fixture_dir, chain_dir, parent, groups, flux, temperature="294K"):
    """Each MT=5 product's production rate per parent atom [1/s].

    A residual of a list written as shares (``share`` true) is
    ``sigma_5 * y_r / sum_r' y_r'``, every other row ``sigma_5 * m``, each
    group-averaged flat in energy and summed over the groups' fluxes.

    The two factors are each linear between the merged points, so on every
    segment their product is integrated exactly, as the branching rule does;
    a segment past the end of the list's curves, where the value is held
    flat, takes the trapezoid instead, as the rule does there too.
    """
    xs = mt5_cross_section(fixture_dir, temperature)
    rows = [r for r in anything_rows(chain_dir, parent) if r[0] != parent]
    if xs is None or not rows:
        return {}
    grid, sigma = xs
    lists = [[r for r in rows if r[3]], [r for r in rows if not r[3]]]
    rates = {}
    for lo, hi, phi in zip(groups[:-1], groups[1:], flux):
        for rows_of_list in lists:
            if not rows_of_list:
                continue
            last = min(e[-1] for _, e, _, _ in rows_of_list)
            points = [grid[(grid > lo) & (grid < hi)], [lo, hi]]
            points += [e[(e > lo) & (e < hi)] for _, e, _, _ in rows_of_list]
            at = np.unique(np.concatenate(points))
            s5 = np.interp(at, grid, sigma)
            values = [(t, _curve(e, m, at)) for t, e, m, _ in rows_of_list]
            if rows_of_list[0][3]:
                total = sum((y for _, y in values), np.zeros_like(at))
                safe = np.where(total > 0.0, total, 1.0)
                values = [(t, np.where(total > 0.0, y / safe, 0.0)) for t, y in values]
            de = np.diff(at)
            b0, b1 = s5[:-1], s5[1:]
            held = at[1:] > last
            for target, f in values:
                f0, f1 = f[:-1], f[1:]
                exact = de * (2.0 * f0 * b0 + f0 * b1 + f1 * b0 + 2.0 * f1 * b1) / 6.0
                trapezoid = 0.5 * (f0 * b0 + f1 * b1) * de
                average = np.sum(np.where(held, trapezoid, exact)) / (hi - lo)
                rates[target] = rates.get(target, 0.0) + phi * average * BARN_CM2
    return rates


# Light particles a decay mode emits beside its daughter.
_DECAY_EJECTILES = {"alpha": "He4", "p": "H1"}


def _ejectiles(mode):
    return [_DECAY_EJECTILES[m] for m in mode.split(",") if m in _DECAY_EJECTILES]


def predicted_mt5_densities(rates, n0, steps, half_lives, decays):
    """What MT=5 adds to each product at the end of each step, and which
    products cannot be predicted at each step.

    ``rates`` are :func:`mt5_production_rates`, ``n0`` the parent's density,
    ``steps`` ``(dt, irradiated)`` pairs. A product with a half-life below
    :data:`INSTANT_FRACTION` of the shortest irradiation step is passed on to
    its daughters (and the light particles its decay emits) as it is made.
    Every other product is made at its rate while the flux is on and decays
    with its own half-life. Its daughters are fed by that decay, which this
    does not follow, so a daughter is listed as unpredicted from the first
    step on which its parent's decayed amount is above ``1e-4`` of what it
    makes; the caller compares the rest.

    Returns ``(added, unpredicted)``, one dict and one set per step.
    """
    shortest = min(dt for dt, on in steps if on)
    instant = INSTANT_FRACTION * shortest

    def short_lived(name):
        h = half_lives.get(name)
        return h is not None and h < instant

    sources = dict((k, v * n0) for k, v in rates.items())
    for _ in range(64):
        fast = [q for q, s in sources.items() if s > 0.0 and short_lived(q)]
        if not fast:
            break
        for q in fast:
            s = sources.pop(q)
            for mode, daughter, branching in decays.get(q, []):
                for product in ([daughter] if daughter else []) + _ejectiles(mode):
                    sources[product] = sources.get(product, 0.0) + branching * s
    else:
        raise RuntimeError("short-lived MT=5 products decay into each other without end")

    density = dict.fromkeys(sources, 0.0)
    made = dict.fromkeys(sources, 0.0)
    added, unpredicted = [], []
    for dt, on in steps:
        for q, s in sources.items():
            h = half_lives.get(q)
            lam = np.log(2.0) / h if h else 0.0
            n = density[q]
            if lam == 0.0:
                density[q] = n + (s * dt if on else 0.0)
            else:
                decay = np.exp(-lam * dt)
                density[q] = n * decay + (s * -np.expm1(-lam * dt) / lam if on else 0.0)
            if on:
                made[q] += s * dt
        added.append(dict(density))
        fed = set()
        for q, n in density.items():
            if made[q] > 0.0 and made[q] - n > 1.0e-4 * made[q]:
                for mode, daughter, _ in decays.get(q, []):
                    fed.update(([daughter] if daughter else []) + _ejectiles(mode))
        unpredicted.append(fed)
    return added, unpredicted
