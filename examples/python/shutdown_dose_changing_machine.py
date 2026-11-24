"""Shutdown dose rate of a machine that changes along its schedule.

A small "reactor": a steel first-wall shell around a plasma, and a steel port
plug. The campaign is

1. a DD phase (2.45 MeV neutrons), with the port plug out,
2. a DT phase (14.06 MeV neutrons), with the port plug in,
3. a day of cooling as it stands,
4. maintenance: the port plug is pulled 300 cm out along x to a hot-cell
   position, and the dose is watched for a week,
5. decommissioning: the plug is taken away altogether.

Each step names the Model the machine is in. Cells are matched across the
models by name, so the plug's activation follows it when it moves, starts only
from the DT phase (it was absent during DD), and stops counting once it is gone.
The dose is read off the decay-photon flux in the hot cell, a 50 cm box around
the plug's parked position.

Uses the ``tests/`` fixtures (Fe54-58 neutron + Fe photon data + the SFR chain).

Run (from the repo root):  python examples/python/shutdown_dose_changing_machine.py
"""
import numpy as np
import yamc

DATA_DIR = "tests"
CHAIN = f"{DATA_DIR}/transmutation-endf-b8.1-sfr.arrow"
yamc.transmutation_decay_data = CHAIN
yamc.transmutation_reactions = CHAIN
yamc.transmutation_fission_yields = CHAIN


IRON = {"Fe54": 0.0585, "Fe56": 0.9175, "Fe57": 0.0212, "Fe58": 0.0028}


def steel():
    """Iron with its natural isotopes: Fe54(n,p) makes Mn54 (312 d) and Fe58
    capture makes Fe59 (44 d), so the dose outlives the hours-long Mn56."""
    mat = yamc.Material(composition=IRON, density=7.874, temperature=294)
    mat.read_nuclear_data({n: f"{DATA_DIR}/{n}.arrow" for n in IRON},
                          photon_data={"Fe": f"{DATA_DIR}/Fe.arrow"})
    return mat


def machine(plug_x=None, source_energy=None):
    """The reactor, with the port plug centred at x=plug_x (None: no plug)."""
    plasma = yamc.Sphere(radius=50.0)
    wall = yamc.Sphere(radius=60.0)
    world = yamc.Sphere(radius=500.0, boundary="vacuum")
    cells = [
        yamc.Cell(name="plasma", region=plasma.below),
        yamc.Cell(name="first wall", region=plasma.above & wall.below, material=steel()),
    ]
    outside = wall.above & world.below
    if plug_x is not None:
        plug = yamc.Sphere(x0=plug_x, radius=8.0)
        cells.append(yamc.Cell(name="port plug", region=plug.below, material=steel()))
        outside = outside & plug.above
    cells.append(yamc.Cell(name="hall", region=outside))
    source = None
    if source_energy is not None:
        source = yamc.NeutronSource(energy=yamc.sources.Discrete([source_energy], [1.0]))
    return yamc.Model(geometry=yamc.Geometry(cells), source=source, verbose=[])


dd_phase = machine(source_energy=2.45e6)                   # no plug yet
dt_phase = machine(plug_x=70.0, source_energy=14.06e6)     # plug in its port
maintenance = machine(plug_x=370.0)                        # plug pulled 300 cm out
decommissioned = machine()                                 # plug gone

schedule = yamc.PulseSchedule([
    yamc.Pulse(model=dd_phase, rate=1e16, duration=(30, "d")),
    yamc.Pulse(model=dt_phase, rate=1e17, duration=(30, "d")),
    yamc.Cooldown(duration=(1, "d")),                      # still the DT layout
    *[yamc.Cooldown(model=maintenance, duration=(1, "d"), moves={"port plug": (300, 0, 0)})
      for _ in range(7)],
    yamc.Cooldown(model=decommissioned, duration=(30, "d")),
])

# The hot cell: a box around the plug's parked position.
hot_cell = yamc.RegularRectangularMesh(lower_left=[345, -25, -25], upper_right=[395, 25, 25],
                                       shape=[1, 1, 1])
dose_tally = yamc.Tally(scores=["flux"], name="hot cell", mesh=hot_cell, particle="photon")

dose = yamc.simulate_shutdown_dose(schedule, tallies=[dose_tally], total_particles=100_000)["hot cell"]

labels = ["end of DD", "end of DT", "1 d cooled"] + [f"maint. day {d}" for d in range(1, 8)] \
    + ["decommissioned"]
for label, t, mean, std in zip(labels, dose.times, dose.mean, dose.std_dev):
    print(f"{label:>15}  t={t / 86400:6.1f} d   flux {mean[0]:.3e} +/- {std[0]:.1e} photon-cm/s")

# The plug dominates once it is parked in the hot cell.
worst = max(dose.by_nuclide, key=lambda n: np.array(dose.by_nuclide[n])[3].sum())
print(f"largest contributor on the first maintenance day: {worst}")
