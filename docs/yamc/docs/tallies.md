# Tallies

### Scores

Tallies accumulate results during particle transport. Scores can be specified by name or ENDF MT number.

<!-- doctest: skip -->
```python
# Named scores
tally = yamc.Tally(scores=["flux"], cells=cell)
tally = yamc.Tally(scores=["heating"], cells=cell)
tally = yamc.Tally(scores=["heating-local"], cells=cell)     # no secondary photon contribution
tally = yamc.Tally(scores=["absorption"], cells=cell)
tally = yamc.Tally(scores=["fission"], cells=cell)
tally = yamc.Tally(scores=["damage-energy"], cells=cell)     # DPA (MT 444)
tally = yamc.Tally(scores=["H3-production"], cells=cell)     # TBR

# MT numbers
tally = yamc.Tally(scores=[2], cells=cell)    # elastic
tally = yamc.Tally(scores=[18], cells=cell)   # fission
tally = yamc.Tally(scores=[102], cells=cell)  # (n,gamma)
tally = yamc.Tally(scores=[105], cells=cell)  # (n,t) tritium production

# Multiple scores in one tally
tally = yamc.Tally(scores=["flux", "heating", 105], cells=cell)
```

#### Virtual response overlays (``response=``)

By default a tally is macroscopic: cross sections are multiplied by the cell's atom density, so it only scores where the responding material physically exists. Passing ``response=`` scores a virtual response everywhere instead, even in void, as if a detector that is not in the geometry were present. ``response`` and ``nuclides`` are mutually exclusive (``nuclides`` remains the macroscopic per-nuclide breakdown at the real in-material density).

The type of ``response`` selects the regime:

- ``response="Si28"`` or ``response=["Li6", "Li7"]``: microscopic cross sections (barns) at unit density, with one result bin per nuclide. Weighting by atom density is left to post-processing.
- ``response=<Material>``: the material's macroscopic response, weighting each nuclide's microscopic XS by its real atom density and collapsing to a single combined bin (e.g. a SiO2 dose map). No post-processing is needed.

<!-- doctest: skip -->
```python
# Nuclide overlay (unit density), one bin per nuclide:
tally = yamc.Tally(
    scores=["heating"],
    response="Si28",
    mesh=mesh,
    name="silicon_dose_overlay",
)

# Material overlay, one combined macroscopic bin weighted by atom density:
silicon = yamc.Material(composition={"Si28": 1.0}, density=2.33)
tally = yamc.Tally(
    scores=["heating"],
    response=silicon,
    mesh=mesh,
    name="silicon_dose",
)
```

### Estimator

Each tally has an ``estimator`` controlling how a particle history is turned into score contributions:

- ``"track-length"`` (default): every cell crossing contributes ``X · weight · Δs``, where ``X`` is the relevant cross section (for ``flux`` the XS drops out, leaving a pure length integral; ``heating_xs`` for ``heating``; ``σ_r`` for a reaction rate; and so on). It has lower variance in optically thin regions and touches more bins per history.
- ``"collision"``: scores at every collision site with contribution ``(X / Σ_t) · weight``, where ``Σ_t`` is the URR-modified macroscopic total cross section the collision was sampled against. It has lower variance in optically thick regions and touches one bin per collision rather than one per crossing. Both estimators converge to the same expected value, so identical seeds must agree to within statistical noise.

<!-- doctest: skip -->
```python
# Default: track-length flux
tally = yamc.Tally(scores=["flux"], cells=cell)

# Collision-estimator flux, set explicitly
tally = yamc.Tally(scores=["flux"], cells=cell, estimator="collision")

# Collision-estimator reaction rate (URR-aware for total / elastic /
# fission / absorption / capture; smooth XS otherwise)
tally = yamc.Tally(scores=["(n,gamma)"], cells=cell, estimator="collision")
```

Score / estimator compatibility:

| Score | Track-length | Collision |
| --- | --- | --- |
| ``flux`` | ✓ (default) | ✓ |
| ``heating`` / ``heating-local`` | ✓: neutron KERMA (MT 301/901) and photon collision KERMA, both scored along tracks | ✓: neutrons use ``(heating_xs / Σ_t) · weight``; photons use the analog absorbed energy ``(E_in − E_out − Σ E_banked_photons) · weight`` at each collision |
| ``reaction-rate`` (``total`` / ``elastic`` / ``fission`` / ``absorption`` / arbitrary MT) | ✓ | ✓, URR-aware for MTs 1 / 2 / 18 / 27 / 102 |
| ``H1-production`` / ``H2-production`` / ``H3-production`` / ``He3-production`` / ``He4-production`` | ✓ | ✓ |
| ``damage-energy`` (MT 444) | ✓ | ✓: ``(damage_energy_xs / Σ_t) · weight`` per neutron collision |
| ``coherent-scatter`` / ``incoherent-scatter`` / ``photoelectric`` / ``pair-production`` (photon XS) | ✓ | ✓: ``(σ_component / Σ_t) · weight`` per photon collision, where ``Σ_t`` is the macroscopic photon total |

Every score type supports both estimators, with the following caveats. In each case the unsupported combination is refused or documented, never silently substituted.

- Photon heating depends on the estimator, because the two follow different physical conventions. ``estimator="collision"`` scores the analog absorbed energy at each photon collision: ``E_in − E_out − Σ E_banked_photons``, which is the electron/positron kinetic energy deposited locally minus the banked TTB bremsstrahlung (the coupled neutron-photon local-deposition convention). This matches a reference code's unfiltered heating to 0.01% on an absorbing Fe sphere, but not its photon-filtered share, which that code books on its electron pseudo-particles. ``estimator="track-length"`` scores collision KERMA along tracks, which has lower variance and suits mesh tallies. It counts the same electron energy, but the TTB-radiated part is deposited locally instead of where those photons land. On thick absorbing bodies the two agree within ~1%. In thin or streaming geometries the track-length value exceeds the absorbed energy by the escaping radiative fraction, which is the usual distinction between kerma and dose (4.7% on a 5 MeV broomstick). Photon ``heating-local`` reports the same values as ``heating``, since with no electron transport there is nothing to separate.
- Unstructured-mesh tallies support both estimators in every mode. The collision estimator resolves the containing tetrahedron with a BVH point query; under ``woodcock`` / ``hybrid``, flux-score track-length unstructured tallies score true track length along each flight segment (like regular meshes) and other scores use the delta-collision estimator with per-tet attribution.

After running, the chosen estimator is reported on the tally:

<!-- doctest: skip -->
```python
print(tally.estimator)  # "track-length" or "collision"
```

### Variance estimator

Each tally reports a per-bin standard deviation alongside its mean. Variance is estimated with per-history Welford accumulation: each source-particle history is one variance sample, each rayon worker updates its private scratch online with Welford's algorithm, and a pairwise reduction then combines the workers. There are no user settings for this (no batch size and no choice of method).

### Spatial binning

Each tally scores in one spatial domain family: cells, materials, mesh, or unstructured mesh. ``cells`` and ``materials`` are mutually exclusive: a single tally cannot bin over both.

<!-- doctest: skip -->
```python
# Cell binning, single cell (one result bin)
tally = yamc.Tally(scores=["flux"], cells=cell)

# Cell binning, list of cells (one result bin per cell)
tally = yamc.Tally(scores=["flux"], cells=[cell1, cell2, cell3])
# After results = model.simulate_transport():
#   results[tally].mean[0] is the flux in cell1
#   results[tally].mean[1] is the flux in cell2
#   results[tally].mean[2] is the flux in cell3
# A bare single cell is also accepted: cells=cell1 gives one bin.

# Material binning, single material
tally = yamc.Tally(scores=["flux"], materials=material)

# Material binning, list of materials (one result bin per material)
tally = yamc.Tally(scores=["flux"], materials=[m1, m2])

# Regular mesh binning. `shape` is either per-axis voxel counts [nx, ny, nz]
# or a single int (total voxel count, split into roughly cubic bins over the box).
mesh = yamc.RegularRectangularMesh(
    lower_left=[-10, -10, -10],
    upper_right=[10, 10, 10],
    shape=[20, 20, 20],           # or shape=8000 for ~8000 voxels
)
tally = yamc.Tally(scores=["flux"], mesh=mesh)

# Mesh from geometry bounding box (same shape rules: int total or [nx, ny, nz])
mesh = yamc.RegularRectangularMesh.from_domain(geometry, shape=1_000_000)
tally = yamc.Tally(scores=["flux"], mesh=mesh)

# Regular cylindrical (r, phi, z) mesh, suited to tokamak and cylindrical
# geometries. phi_bounds defaults to the full circle (0, 2*pi); pass a
# sub-interval (e.g. (0, math.pi)) for a sector.
import math
mesh = yamc.RegularCylindricalMesh(
    r_bounds=(0.0, 10.0),
    z_bounds=(-5.0, 5.0),
    shape=(10, 16, 20),                 # (nr, nphi, nz)
    phi_bounds=(0.0, 2 * math.pi),
    origin=(0.0, 0.0, 0.0),
)
tally = yamc.Tally(scores=["flux"], mesh=mesh)
# Cylindrical cells are not equal in volume: outer radial rings are larger.
# Normalise per cell when comparing flux across radii:
#   results[tally].mean[b] / mesh.element_volume(b)

# Unstructured mesh binning: a bin per tet of a MeshGeometry, in the
# order of mesh_geom.tetrahedra
tally = yamc.Tally(scores=["flux"], mesh=mesh_geom)
tally = yamc.Tally(scores=["flux"], mesh=(mesh_geom, "fuel"))  # one volume's tets

# A tally-only mesh, from any mesher and at any resolution: it need not match
# the geometry, and a bin per cell, in the order given
thermal = yamc.UnstructuredMesh(vertices, cells)
tally = yamc.Tally(scores=["heating"], mesh=thermal)
```

``mesh=`` takes every kind of mesh. A regular mesh bins on its grid. A
``MeshGeometry`` or an ``UnstructuredMesh`` has a bin per cell, and with a
region name, ``(mesh_geom, "fuel")``, only that volume's cells score (a volume
ID is accepted for a ``MeshGeometry`` too). ``UnstructuredMesh`` takes
``cell_regions=`` to name regions, and has ``cells``, ``cell_volumes``,
``cell_regions`` and ``to_vtkhdf``.

The ``mean``/``std_dev``/``relative_error`` arrays are laid out flat. When multiple binning axes are present, the bin order from slowest to fastest varying is ``score``, ``surface``, ``cosine``, ``cell``, ``material``, ``nuclide``, ``parent_nuclide``, ``energy``, ``mesh``. The ``surface`` and ``cosine`` axes only appear on [surface current](#surface-currents) tallies.

### Energy binning

<!-- doctest: skip -->
```python
# Custom bin boundaries (eV)
tally = yamc.Tally(scores=["flux"], cells=cell, energy_bins=[0, 0.625, 1e5, 20e6])

# Named group structure
tally = yamc.Tally(scores=["flux"], cells=cell, energy_group_structure="VITAMIN-J-175")
```

#### The built-in group structures

| name | groups | range | bin holding 14.1 MeV |
| --- | ---: | --- | --- |
| ``XMAS-172`` | 172 | 1.00001e-5 eV to 19.6403 MeV | 13.84 to 14.92 MeV |
| ``VITAMIN-J-175`` | 175 | 1e-5 eV to 19.64 MeV | 13.84 to 14.19 MeV |
| ``SCALE-252`` | 252 | 0 eV to 20 MeV | 13.84 to 14.55 MeV |
| ``TRIPOLI-315`` | 315 | 1e-5 eV to 19.64 MeV | 13.84 to 14.19 MeV |
| ``SHEM-361`` | 361 | 0 eV to 19.6403 MeV | 13.84 to 14.92 MeV |
| ``LLNL-616`` | 616 | 1e-5 eV to 20 MeV | 13.80 to 14.45 MeV |
| ``CCFE-709`` | 709 | 1e-5 eV to 1 GeV | 14.00 to 14.20 MeV |
| ``SCALE-999`` | 999 | 1e-5 eV to 20 MeV | 13.84 to 14.19 MeV |
| ``UKAEA-1102`` | 1102 | 1e-5 eV to 1 GeV | 14.00 to 14.20 MeV |
| ``ECCO-1968`` | 1968 | 1.00001e-5 eV to 19.64033 MeV | 14.07 to 14.19 MeV |
| ``CCFE-24-PHOTON`` | 24 | 1 keV to 20 MeV | photon structure |
| ``VITAMIN-J-42`` | 42 | 1 keV to 50 MeV | photon structure |

Two of them, ``SCALE-252`` and ``SHEM-361``, have a lowest boundary of exactly 0 eV. That is OpenMC's value and is kept, but it sits below the 1e-5 eV floor of the transport data, so treat their bottom bin as open-ended rather than as a real energy range.

Passing a name that is not in the table raises an error listing the valid names, so a typo cannot silently select a different structure. Every name except ``CCFE-24-PHOTON`` carries the same boundaries as OpenMC's ``openmc.mgxs.GROUP_STRUCTURES`` entry of that name, so a spectrum tabulated for one code can be handed to the other without re-binning.

The structures can also be read as data, so a multigroup result can be plotted or folded against the energies it was scored on:

<!-- doctest: skip -->
```python
yamc.group_structure_names()          # every name in the table above
edges = yamc.group_structure("UKAEA-1102")
len(edges)                            # 1103, one more than the group count
edges[0], edges[-1]                   # 1e-05, 1000000000.0
```

``Tally.energy_bins`` returns the same list once a group structure has been set on a tally.

#### What the choice of structure costs

Choosing a structure trades resolution against bin count, and the effect is easiest to show with an example. Below, a 14.06 MeV point source sits at the centre of a 50 cm sphere of one homogenized mixture of tungsten, water, steel, lithium and beryllium, and the same cell flux is tallied ten times over, once per neutron structure in the table. Flux is divided by bin width in lethargy, so that structures with different bin widths land on one axis.

The left panel shows the whole slowing-down spectrum. All ten agree on it, but the coarse structures step over the resonance dips and the thermal end that the fine ones resolve. The right panel shows the source line itself, which each structure smears across whichever bin contains 14.06 MeV. The wider the bin, the lower and flatter the peak, from ``XMAS-172`` at the bottom to ``ECCO-1968`` at the top. This figure is rendered live when these docs are built.

```python exec="1" html="1" source="above"
import base64
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yamc

yamc.neutron_cross_section_data = "endf-b8.1"

RADIUS = 50.0
mixture = yamc.Material(
    composition={"W": 0.10, "Fe": 0.20, "Cr": 0.05, "Ni": 0.03,
                 "H2O": 0.30, "Li": 0.20, "Be": 0.12},
    density=4.0,
)
sphere = yamc.Sphere(radius=RADIUS, boundary="vacuum")
cell = yamc.Cell(region=sphere.below, material=mixture)
geometry = yamc.Geometry([cell])            # before the tallies: filtering needs the cell id
source = yamc.NeutronSource()

names = [n for n in yamc.group_structure_names() if "PHOTON" not in n and n != "VITAMIN-J-42"]
tallies = [yamc.Tally(scores=["flux"], cells=cell, energy_group_structure=n, name=n)
           for n in names]
results = yamc.Model(
    geometry=geometry, source=source, tallies=tallies, verbose=[]
).simulate_transport(total_particles=500_000)

volume = 4 / 3 * np.pi * RADIUS**3
fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.9, 1]})
for color, name, tally in zip(plt.cm.viridis(np.linspace(0, 0.9, len(names))), names, tallies):
    edges = np.array(tally.energy_bins)
    # SCALE-252 and SHEM-361 start at 0 eV, which has no lethargy; widen that
    # bottom bin to the 1e-5 eV transport floor instead.
    lethargy = np.diff(np.log(np.maximum(edges, 1e-5)))
    flux = np.array(results[tally].mean) / volume / lethargy
    flux[flux <= 0] = np.nan                # bins no particle reached, not zero flux
    ax.stairs(flux, edges, color=color, lw=1.1, label=f"{name} ({len(edges) - 1})")
    axz.stairs(flux, edges, color=color, lw=1.4)

ax.set(xscale="log", yscale="log", xlim=(1e-3, 3e7), ylim=(1e-10, 5e-3),
       xlabel="Energy [eV]",
       ylabel=r"Flux per unit lethargy [cm$^{-2}$ per source neutron]",
       title="Cell-averaged spectrum, one tally per group structure")
ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25)
axz.set(xlim=(1.30e7, 1.50e7), xlabel="Energy [eV]", title="The 14.06 MeV source peak")
axz.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
axz.grid(alpha=0.25)
fig.tight_layout()

buf = BytesIO()
fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
print(
    '<img alt="the same spectrum tallied on ten group structures" '
    f'src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
    'style="max-width:100%">'
)
```

### Nuclide binning

Score per-nuclide contributions separately.

<!-- doctest: skip -->
```python
tally = yamc.Tally(
    scores=["H3-production"],
    cells=cell,
    nuclides=["Li6", "Li7", "Be9"],
)
# results[tally].mean returns [Li6_score, Li7_score, Be9_score]
```

### Particle binning

Restrict scoring to neutrons or photons. This is required for separate neutron and photon spectra.

<!-- doctest: skip -->
```python
tally_n = yamc.Tally(scores=["flux"], cells=cell, particle="neutron")
tally_p = yamc.Tally(scores=["flux"], cells=cell, particle="photon")
```

### Surface currents

A tally with ``surfaces=`` counts particles crossing those surfaces, instead of scoring along tracks or at collisions. This gives leakage through a boundary directly, and the current through any surface inside a model. There is one result bin per surface, in the order given, and the results are particles per source particle.

The scores count crossings by the side of the surface they cross to:

| Score | Counts |
|---|---|
| ``"current-above"`` | crossings towards the surface's ``.above`` side, each adding the particle weight |
| ``"current-below"`` | crossings towards ``.below``, each adding the particle weight |
| ``"current-net"`` | above minus below |

Which side is ``.above`` is the same as when building regions: outside a ``Sphere``, ``Cylinder``, ``Cone`` or ``Torus``, and the positive side of a ``Plane`` (``x > offset`` for ``axis="x"``). On a sphere, ``"current-above"`` is the outward current and ``"current-below"`` the inward one. On an x plane they are the currents towards +x and -x. A particle that crosses the same surface several times is counted at every crossing.

<!-- doctest: skip -->
```python
# Leakage out of a sphere in vacuum. Nothing comes back through the outer
# surface, so above and net are both the leakage and below is zero.
outer = yamc.Sphere(radius=30.0, boundary="vacuum")
leakage = yamc.Tally(
    scores=["current-above", "current-below", "current-net"],
    surfaces=outer,
)
# results[leakage].mean is [above, below, net] per source particle

# Current through a plane inside a model, towards +x and towards -x
wall = yamc.Plane(axis="x", offset=50.0)
through_wall = yamc.Tally(scores=["current-above", "current-below"], surfaces=wall)

# Several surfaces in one tally: one bin per surface
layers = yamc.Tally(scores=["current-above"], surfaces=[inner_sphere, outer_sphere])
```

A surface tally also takes ``energy_bins`` (the energy of the particle as it crosses), ``particle``, and an energy function or dose coefficients. ``cosine_bins`` bins the crossings by the cosine of the angle between the particle's direction and the surface normal, which points to ``.above``. A cosine of 1 is straight along the normal and 0 is grazing; negative cosines are crossings towards ``.below``.

<!-- doctest: skip -->
```python
import math

# Outward leakage spectrum
leakage_spectrum = yamc.Tally(
    scores=["current-above"],
    surfaces=outer,
    energy_group_structure="VITAMIN-J-175",
    particle="neutron",
)

# Only particles leaving within 17.28 degrees of the normal, as a detector
# viewing part of the surface sees them
detector_view = yamc.Tally(
    scores=["current-above"],
    surfaces=outer,
    cosine_bins=[math.cos(math.radians(17.28)), 1.0],
)
```

A surface tally cannot also take ``cells``, ``materials``, ``mesh``, ``nuclides``, ``response`` or ``estimator``. It needs ``tracking_mode="surface"`` (the default) and CSG surfaces. It runs on the CPU and with ``compute="gpu"``, except that ``cosine_bins`` is CPU only. Each surface needs an id; one is assigned automatically when a surface is created without ``surface_id=``.

#### Example: leakage spectrum from an iron sphere

<!-- doctest: skip -->
```python
import matplotlib.pyplot as plt
import numpy as np
import yamc

yamc.neutron_cross_section_data = "endf-b8.1"

outer = yamc.Sphere(radius=30.0, boundary="vacuum")
iron = yamc.Material(composition={"Fe": 1.0}, density=7.874)
cell = yamc.Cell(region=outer.below, material=iron)
geometry = yamc.Geometry([cell])

source = yamc.NeutronSource()

leakage = yamc.Tally(
    scores=["current-above"],
    surfaces=outer,
    energy_group_structure="VITAMIN-J-175",
    name="neutron_leakage",
)

model = yamc.Model(geometry=geometry, tallies=[leakage], source=source)
results = model.simulate_transport(total_particles=200000)

bins = np.array(leakage.energy_bins)
current = np.array(results[leakage].mean)
lethargy = np.log(bins[1:] / bins[:-1])
print(f"neutrons leaking per source neutron: {current.sum():.3f}")

fig, ax = plt.subplots()
ax.stairs(current / lethargy, bins)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Leakage per unit lethargy per source neutron")
fig.savefig("leakage_spectrum.png", dpi=150)
```

### Energy function weighting

Apply an energy-dependent weighting function to the tally (e.g. response functions, DPA cross sections).

``interpolation`` says how the values are read between points: ``"cubic"``, a
natural cubic spline in linear energy and the default, which needs at least four
points; ``"linear"``; or ``"log-log"``, the convention for tabulated
coefficients, which falls back to linear over an interval with a zero value.
The last two need two points. An ``energy_function`` given as
``(energies, values, units, interpolation)``, the tuple ``as_energy_function()``
returns, carries its own. A particle whose energy falls outside the tabulated
range scores nothing into that tally, so make the grid span the whole spectrum
you expect.

<!-- doctest: skip -->
```python
tally = yamc.Tally(
    scores=["flux"],
    cells=cell,
    # (energies, values); at least 4 points, energies strictly increasing
    energy_function=([1e-5, 1e3, 1e6, 20e6], [1.0, 1.5, 2.0, 0.5]),
)
```

### Dose coefficients

Apply effective dose conversion coefficients directly. They are ICRP-116 by default; add
``"icrp74"`` as a third element, ``("neutron", "AP", "icrp74")``, for ICRP-74. They are
interpolated log-log, as contact dose reads the same tables.

<!-- doctest: skip -->
```python
tally = yamc.Tally(
    scores=["flux"],
    mesh=mesh,
    dose_coefficients=("neutron", "AP"),  # (particle, geometry[, data_source])
)
```

A dose tally scores only the particle its coefficients are for, so a neutron
dose tally in a coupled neutron-photon run leaves the photons out. A
``particle=`` naming the other particle is refused; use ``energy_function=`` to
weight a particle by some other curve.

For the total dose from neutrons and photons, use ``"neutron_photon"`` as the
particle. Each particle is weighted by its own coefficients and both go into
the same bin, so the standard deviation is that of the total:

<!-- doctest: skip -->
```python
total_dose = yamc.Tally(scores=["flux"], mesh=mesh, dose_coefficients=("neutron_photon", "AP"))
```

Adding a neutron dose tally and a photon dose tally gives the same mean, but
not the error of the sum: the photons of a history are correlated with the
neutrons that made them, so adding the two errors in quadrature can be wrong in
either direction. Add separate neutron and photon tallies as well if the
breakdown is wanted.

For ambient dose equivalent H*(10), the quantity most regulations still
reference, fetch the ICRP-74 coefficients with
``yamc.data.dose_coefficients(..., dose_quantity="ambient")`` and fold them in
as an energy function, whose tuple names the log-log interpolation itself. H*(10)
is defined in an aligned and expanded field, so it takes no geometry, and ICRP-116
does not tabulate it.

<!-- doctest: skip -->
```python
h10 = yamc.data.dose_coefficients("neutron", dose_quantity="ambient")
tally = yamc.Tally(
    scores=["flux"],
    mesh=mesh,
    energy_function=h10.as_energy_function(),
)
```

The tables cover 1 meV to 20 MeV for neutrons and 10 keV to 10 MeV for photons
and are not extrapolated, so a particle outside that range scores nothing. H*(10)
is designed to over-estimate effective dose for area monitoring, but for
high-energy neutrons it under-reads it.

### Bin covariance

``standard_deviation`` treats each bin on its own, but the bins of one tally are
scored by the same histories and move together, so anything summed over them, a
reaction rate over a spectrum for instance, inherits an understated error from
the standard deviations alone. ``covariance=True`` also accumulates the
covariance of the bin means, history by history:

<!-- doctest: skip -->
```python
tally = yamc.Tally(
    scores=["flux"],
    cells=cell,
    energy_group_structure="VITAMIN-J-175",
    covariance=True,
)
# after the run: results[tally].covariance, a bins x bins nested list
```

It is meant for spectra: it is limited to 2048 bins and runs on the CPU only.
It is what ``Pulse(flux_covariance=...)`` takes, as [Nuclear-data
uncertainty](materials.md#nuclear-data-uncertainty) describes.

### Neutron and photon flux spectra

A complete example scoring energy-resolved neutron and photon flux with coupled transport.

<!-- doctest: skip -->
```python
import yamc

# Coupled transport reads neutron and photon data.
yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"

# Geometry: iron sphere
sphere = yamc.Sphere(radius=30.0, boundary="vacuum")
iron = yamc.Material(composition={"Fe": 1.0}, density=7.874)
cell = yamc.Cell(region=sphere.below, material=iron)
geometry = yamc.Geometry([cell])

# Source: 14 MeV neutrons
source = yamc.NeutronSource()

# Tallies: energy-binned neutron and photon flux
tally_n = yamc.Tally(
    scores=["flux"],
    cells=cell,
    energy_group_structure="VITAMIN-J-175",
    particle="neutron",
    name="neutron_spectrum",
)
tally_p = yamc.Tally(
    scores=["flux"],
    cells=cell,
    energy_group_structure="CCFE-24-PHOTON",
    particle="photon",
    name="photon_spectrum",
)

model = yamc.Model(
    geometry=geometry,
    tallies=[tally_n, tally_p],
    source=source,
    transport_secondary_photons=True,
)
results = model.simulate_transport(total_particles=500000)

# Plot spectra
import numpy as np
import matplotlib.pyplot as plt

n_bins = np.array(tally_n.energy_bins)
n_flux = np.array(results[tally_n].mean)
n_std = np.array(results[tally_n].standard_deviation)

p_bins = np.array(tally_p.energy_bins)
p_flux = np.array(results[tally_p].mean)
p_std = np.array(results[tally_p].standard_deviation)

fig, ax = plt.subplots()
for bins, flux, std, label in [
    (n_bins, n_flux, n_std, "neutron"),
    (p_bins, p_flux, p_std, "photon"),
]:
    centers = np.sqrt(bins[:-1] * bins[1:])      # geometric bin centres
    ax.step(bins[:-1], flux, where="post", label=label)
    ax.errorbar(centers, flux, yerr=std, fmt="none", capsize=2, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Flux [cm / source particle]")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig("flux_spectra.png", dpi=150)
```


