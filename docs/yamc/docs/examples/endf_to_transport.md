# From an ENDF file to a transport simulation

Nuclear data evaluations are published as ENDF files, and yamc reads Arrow. This
example takes one evaluation, aluminium-27 from TENDL-2025, converts it to Arrow
on your own machine and uses it in a transport simulation: a sphere of
aluminium three mean free paths in radius, with a 14.06 MeV point source at its
centre, tallying the spectrum of the neutrons that leak out of it. Finally the same model is run on
yamc's published TENDL-2025 data, so you can check the conversion against it.

The same steps work for any neutron evaluation: one from another library, a
patched one, or a new one produced by a code such as TALYS.

## Installing

yamc, plus NJOY to reconstruct the resonances and Doppler broaden the
evaluation. NJOY2016 is not on PyPI; [this wheel](https://github.com/shimwell/wheels)
puts an `njoy` executable in the environment's `bin/`:

```bash
pip install yamc matplotlib
pip install --extra-index-url https://shimwell.github.io/wheels njoy2016
```

Any NJOY build works too. Pass its path to the converter as `njoy_exec`.

## Download the evaluation

TENDL serves each evaluation as its own file, about 4 MB for Al27:

```python
import urllib.request

url = "https://tendl.imperial.ac.uk/tendl_2025/neutron_files/Al/Al027/lib/endf/n-Al027.tendl"
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(request) as response, open("n-Al027.tendl", "wb") as f:
    f.write(response.read())
```

## Convert it to Arrow

`convert_neutron_transport` runs NJOY on the evaluation and writes everything a
transport simulation reads: the cross sections, the secondary angle and energy
distributions, and the lookup tables. Ask for the temperature the material will
be simulated at, here room temperature.

```python
import yamc

yamc.convert_neutron_transport(
    "n-Al027.tendl",
    "my_data/neutron",
    temperatures=[294.0],
)
```

This takes about 20 seconds and writes `my_data/neutron/Al27.arrow/`. The
nuclide's name comes from the evaluation itself, not from the file name.

## Simulate the leakage spectrum

Point yamc at the folder of converted data. Any nuclide the model uses is then
read from `<folder>/<Nuclide>.arrow`.

The sphere is sized from the material itself: three mean free paths at the
source energy, rounded up to a whole centimetre. That is thick enough that most
neutrons collide before they get out, so the leakage spectrum runs from the
14 MeV source peak down through the slowing-down range and tests elastic and
inelastic scattering, (n,2n) and the other exit channels, and the same recipe
gives a comparable sphere for any nuclide.

```python
import math

yamc.neutron_cross_section_data = "my_data/neutron"

aluminium = yamc.Material(composition={"Al27": 1.0}, density=2.7)
radius = math.ceil(3 * aluminium.mean_free_path_neutron(14.06e6))  # 3 x 9.48 cm -> 29 cm
sphere = yamc.Sphere(radius=radius, boundary="vacuum")
geometry = yamc.Geometry([yamc.Cell(region=sphere.below, material=aluminium)])

# The leakage spectrum rather than the flux in the sphere: it is what sphere
# benchmark experiments measure, it depends on the angles secondaries are
# emitted at as well as their energies, and its total per source neutron
# checks the multiplying reactions such as (n,2n). The cell flux converges
# only slightly faster and tests less.
leakage = yamc.Tally(
    scores=["current-above"],
    surfaces=sphere,
    energy_group_structure="CCFE-709",
)

model = yamc.Model(geometry=geometry, tallies=[leakage], source=yamc.NeutronSource())
converted = model.simulate_transport(total_particles=1_000_000)
```

`current-above` on the outer sphere counts every neutron crossing it outwards,
and with a vacuum boundary nothing comes back in, so this is the leakage.

One million neutrons take a few seconds, and leave every bin that holds more
than 0.1% of the leakage with a relative error under 4%. The run is given a
particle count rather than left to stop on its own: with no stop condition,
`simulate_transport` runs until every scored bin passes the statistical checks,
and of the 709 groups the few at eV and keV energies that a neutron reaches
only now and then would keep it going for hours.

## Compare with the published data

yamc's published TENDL-2025 data is processed from the same evaluation, so the
two spectra should lie on top of each other. A model keeps the nuclear data it
first ran with, so build a new one on the same geometry for the published data,
run it, and plot both spectra per unit lethargy:

```python
import matplotlib.pyplot as plt
import numpy as np

yamc.neutron_cross_section_data = "tendl-2025"
model = yamc.Model(geometry=geometry, tallies=[leakage], source=yamc.NeutronSource())
published = model.simulate_transport(total_particles=1_000_000)

bins = np.array(leakage.energy_bins)
lethargy = np.log(bins[1:] / bins[:-1])

fig, ax = plt.subplots()
for label, run in [("converted here", converted), ("published tendl-2025", published)]:
    current = np.array(run[leakage].mean)
    print(f"{label}: {current.sum():.4f} neutrons leak per source neutron")
    ax.stairs(current / lethargy, bins, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Leakage per unit lethargy per source neutron")
ax.legend()
fig.savefig("leakage_spectrum.png", dpi=150)
```

On the published data about 0.79 neutrons leak per source neutron, and about
half of them below 1 MeV. The peak at 14 MeV is the neutrons that crossed the
sphere without a collision; below it are those that scattered on the way out. If your own conversion differs from
the published one by more than the Monte Carlo noise, the two were processed
differently, for example at another temperature.
