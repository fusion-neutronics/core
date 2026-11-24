# Sources

### Source

A source defines where, in what direction, and at what energy particles are born. All parameters are keyword-only.

```python
source = yamc.NeutronSource(
    position=(0, 0, 0),
    energy=14.06e6,
)
```

Defaults: origin at ``[0, 0, 0]``, 14.06 MeV, isotropic, neutron, strength 1.0.

``position`` accepts a plain ``(x, y, z)`` tuple and ``energy`` a single value in
eV as shorthand for a fixed point and a monoenergetic source. Pass an explicit
spatial or energy distribution (see below) when you need a ring, a spectrum, or
any other multi-location / multi-energy source.

The spatial, energy and angular distributions below live in the ``yamc.sources``
submodule (``yamc.sources.Discrete``, ``yamc.sources.Isotropic``, ...). The
``NeutronSource`` / ``PhotonSource`` builders themselves stay at the top level;
you can also ``from yamc import sources`` and write ``sources.Discrete(...)``.

### Spatial distributions

```python
# Fixed point, using the (x, y, z) tuple shorthand
source = yamc.NeutronSource(position=(0, 0, 0), energy=14.06e6)

# Cylindrical ring source (a bare ring). For a real tokamak plasma, with
# its shape and its density/temperature profiles, see
# "Parametric tokamak plasma source" below.
import math
source = yamc.NeutronSource(
    position=yamc.sources.CylindricalRing(
        radius=yamc.sources.Discrete([300], [1]),   # radius in cm
        phi=yamc.sources.Uniform(0, 2 * math.pi),   # full ring
        z=yamc.sources.Discrete([0], [1]),          # midplane
    ),
    energy=14.06e6,
)
```

### Source on a mesh

``yamc.sources.MeshDistribution`` spreads a source over the elements of a mesh:
the voxels of a ``RegularRectangularMesh``, the cells of a
``RegularCylindricalMesh``, or the tetrahedra of a tet ``MeshGeometry``. Each
element emits in proportion to its strength, from a point uniform within it.
Pass it as ``position=`` to ``NeutronSource`` or ``PhotonSource``.

```python
mesh = yamc.RegularRectangularMesh([0, 0, 0], [10, 10, 10], [2, 1, 1])
space = yamc.sources.MeshDistribution(mesh, strengths=[3.0, 1.0])
source = yamc.PhotonSource(position=space, energy=1e6)

print(space.num_elements)    # 2
print(space.total_strength)  # 4.0
```

Here three quarters of the photons are born in the voxel with ``x`` from 0 to
5 cm and one quarter in the other.

``strengths`` takes one non-negative value per element, in the order a tally
on the same mesh numbers its bins, so a mesh tally's ``mean`` can be passed in
directly. Omitted, each element's strength is its volume, which makes the
source uniform over the mesh. Elements with zero strength never emit. A list of
the wrong length, a NaN, infinite or negative value, or strengths that sum to
zero raise ``ValueError``. When the strengths are rates (particles per second),
``total_strength`` is the source rate to scale per-source-particle tallies by.

``cells=`` (a ``Cell`` or a list of them) or ``materials=`` (a ``Material`` or
a list) keeps each element's strength but has it emit only from the parts of it
inside those cells, or inside the cells filled with those materials. A decay
photon source on a voxel half filled with steel is then born in the steel, not
in the void beside it. These need the geometry, so they apply in a ``Model``
run; ``space.sample()`` on its own draws from the whole element.

### Energy distributions

```python
# Monoenergetic
energy = 14.06e6

# Discrete line spectrum, e.g. Co-60's two main decay gammas,
# 1.17 MeV and 1.33 MeV, emitted in ~equal intensity
energy = yamc.sources.Discrete([1.1732e6, 1.3325e6], [0.5, 0.5])

# Histogram (piecewise-constant) spectrum: n+1 ascending bin edges (eV) and one
# per-bin weight. Weights are per-bin probability mass and are normalized
# internally, so a raw multigroup flux per group can be passed directly. Here
# 30% of neutrons are born uniformly in [0, 1] MeV and 70% in [1, 20] MeV.
energy = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])

# The bin edges can also be a built-in group structure name instead of a list
# (e.g. "CCFE-709", "UKAEA-1102"); pass one weight per group. yamc.group_structure()
# returns the edges of any of them, and yamc.group_structure_names() lists them;
# see the table under Tallies > Energy binning.
energy = yamc.sources.Histogram("CCFE-709", flux_per_group)   # 709 weights

# Uniform range
energy = yamc.sources.Uniform(1e6, 20e6)

# Gaussian
energy = yamc.sources.Normal(14.06e6, 3e5)

# DT fusion neutron spectrum (Ballabio et al.)
energy = yamc.sources.fusion_neutron_spectrum(ion_temp=20000.0)
# DD fusion neutron spectrum
energy = yamc.sources.fusion_neutron_spectrum(ion_temp=20000.0, reactants='DD')
```

### Angular distributions

```python
direction = yamc.sources.Isotropic()                 # default, usually omitted
direction = yamc.sources.Monodirectional([0, 0, 1])  # beam in +z
```

### Parametric tokamak plasma source

`yamc.sources.tokamak_source()` builds a tokamak's neutron source from the
plasma itself (its shape, confinement mode, and ion density and temperature
profiles) rather than from a hand-placed ring.

Bind the common parameters with `functools.partial`, then override the sector,
fuel or mesh resolution when calling the source factory.

```python
from functools import partial

import yamc

# Shape (cm, dimensionless)
shape = yamc.sources.TokamakShape(
    major_radius=906.0,
    minor_radius=292.258,
    elongation=1.557,
    triangularity=0.270,
    shafranov_factor=0.44789,
)

make_plasma_source = partial(
    yamc.sources.tokamak_source,
    shape=shape,
    # Confinement mode: "L", "H" or "A"
    mode="H",
    # Ion density profile (m^-3, cm for the pedestal radius)
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    pedestal_radius=0.8 * 292.258,
    # Ion temperature profile (eV)
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
)

sources = make_plasma_source()
model = yamc.Model(geometry=geometry, source=sources, tallies=tallies)
```

Lengths are in cm, ion densities in m^-3, ion temperatures in eV and angles in
radians. The full argument list is in the [API reference](api.md).

The shape is its own object, `TokamakShape`, so it can be checked before any
source is built. It refuses an unphysical shape (a minor radius not less than
the major radius, a triangularity outside -1 to 1, a Shafranov factor of half
the minor radius or more) when it is constructed, and gives the plasma volume
and points on any flux surface:

```python
shape.volume()                  # plasma volume inside the boundary, cm^3
r, z = shape.evaluate(1.0, 0.0) # outboard midplane of the boundary, cm
```

`evaluate(rho, theta)` takes the normalised radius `rho` (0 on the magnetic
axis, 1 on the boundary) and the poloidal angle `theta` in radians. Either may
be a sequence, so a whole flux surface comes back in one call.

It returns a list of `NeutronSource` ring sources. The plasma is
discretised onto a cylindrical (R, phi, Z) mesh and each non-empty voxel
becomes one source per reaction, carrying the Ballabio spectrum at that voxel's
emission-weighted ion temperature. Strengths are normalised to sum to 1, so the
absolute neutron rate comes from the simulation's source rate. Pass the list
straight to `Model(source=...)`, as under
"Multiple sources and strength weighting" below.

The mesh defaults to `(100, 100)`, which is about 15000 sources for D-T fuel.
Every non-empty voxel costs one source, so pass a coarser
`mesh_resolution=(50, 50)` when a smoother source is not worth the memory.

#### Toroidal sectors

A wedge model only needs the matching wedge of plasma. `start_angle` and
`rotation_angle` are in radians; a negative `rotation_angle` sweeps the other
way.

```python
import math

sources = make_plasma_source(
    start_angle=0.0,
    rotation_angle=math.pi / 4,   # a 45 degree sector
)
```

#### Fuel

`fuel` takes atom fractions that sum to 1, and defaults to 50:50 D-T. The
supported species are `"D"` and `"T"`, giving D(d,n)3He and T(d,n)4He neutrons.

```python
sources = make_plasma_source(fuel={"D": 1.0})          # D-D only
sources = make_plasma_source(fuel={"D": 0.4, "T": 0.6})
```

T(t,2n)4He is not included because yamc has no T-T neutron spectrum model, so
D-D and D-T carry the whole yield. For 50:50 D-T fuel at reactor temperatures
the T-T channel is well under a percent of the total (0.7% for the parameters
above), and pure-tritium fuel is rejected rather than silently returning no
neutrons.

#### Plasma profiles

The profiles the source is built from are exposed on their own, for plotting
and for sanity-checking a parameter set. They depend only on the minor radius,
and each accepts a float or a sequence of floats.

```python
densities = yamc.sources.tokamak_ion_density(
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    minor_radius=292.258,
    pedestal_radius=0.8 * 292.258,
    r=[0.0, 100.0, 292.258],           # cm
)

temperatures = yamc.sources.tokamak_ion_temperature(
    r=[0.0, 100.0, 292.258],
    mode="H",
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    minor_radius=292.258,
    pedestal_radius=0.8 * 292.258,
)
```

`tokamak_neutron_source_density()` completes the set: it gives the Bosch-Hale
reaction rate density in neutrons/s/m^3. Mapping flux coordinates onto
`(R, Z)`, with the elongation, triangularity and Shafranov shift applied, is
`TokamakShape.evaluate()` above.

Plotting the two profiles across the plasma:

```python
import matplotlib.pyplot as plt
import yamc

minor_radius = 292.258
pedestal_radius = 0.8 * minor_radius
radii = [i * minor_radius / 200 for i in range(201)]

densities = yamc.sources.tokamak_ion_density(
    mode="H",
    minor_radius=minor_radius,
    pedestal_radius=pedestal_radius,
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    r=radii,
)
temperatures = yamc.sources.tokamak_ion_temperature(
    mode="H",
    minor_radius=minor_radius,
    pedestal_radius=pedestal_radius,
    r=radii,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
)

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
top.plot(radii, densities)
top.set_ylabel("Ion density [m$^{-3}$]")
bottom.plot(radii, temperatures)
bottom.set_xlabel("Minor radius [cm]")
bottom.set_ylabel("Ion temperature [eV]")
for axis in (top, bottom):
    axis.axvline(0.8 * minor_radius, ls="--", c="grey", lw=1, label="pedestal")
    axis.grid(True, alpha=0.3)
top.legend()
plt.show()
```

#### Plotting the emission

Sampling the sources shows the plasma's poloidal cross-section: the D-shape
from the triangularity, the elongation, and the emission concentrated near the
magnetic axis.

```python
import math

import matplotlib.pyplot as plt
import numpy as np
import yamc

sources = make_plasma_source(mesh_resolution=(60, 60))

# Share the points out between the sources by strength, the way the transport
# picks them. Sampling every source equally would instead give each voxel the
# same weight, hiding both the peaked emission and the true D-D fraction.
n_points = 40_000
strengths = np.array([source.strength for source in sources])
counts = np.random.default_rng(1).multinomial(n_points, strengths / strengths.sum())

radii, heights, energies = [], [], []
for source, n in zip(sources, counts):
    if n == 0:
        continue
    positions, sampled = source.sample_n(int(n))
    for (x, y, z), energy in zip(positions, sampled):
        radii.append(math.hypot(x, y))
        heights.append(z)
        energies.append(energy / 1e6)

fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter(radii, heights, c=energies, s=2, alpha=0.4, cmap="plasma")
ax.set_xlabel("R [cm]")
ax.set_ylabel("Z [cm]")
ax.set_aspect("equal")
ax.set_title("Neutron birth sites, coloured by energy")
fig.colorbar(scatter, ax=ax, label="Birth energy [MeV]")
plt.show()
```

Emission crowds towards the magnetic axis, where the plasma is hottest, and
the sparse 2.5 MeV points are the D-D component, about 0.5% of the yield for
50:50 D-T fuel at these temperatures.

### Stellarator plasma source

`yamc.sources.stellarator_source()` builds a stellarator's neutron source from
its shape and the same ion density and temperature profiles as the tokamak
source.

The shape is a `StellaratorShape`: R and Z as Fourier-Zernike series, in the
conventions of [DESC](https://github.com/PlasmaControl/DESC),

$$
R(\rho, \theta, \zeta) = \sum_{lmn} R_{lmn}\, \mathcal{R}_l^{|m|}(\rho)\, \Theta_m(\theta)\, \Phi_n(\zeta)
$$

and likewise for Z. $\mathcal{R}_l^{|m|}$ is the Zernike radial polynomial
($\mathcal{R}_l^{|m|}(1) = 1$, $l \ge |m|$, $l - |m|$ even), $\Theta_m$ is
$\cos(m\theta)$ for $m \ge 0$ and $\sin(|m|\theta)$ for $m < 0$, and $\Phi_n$
is $\cos(n N \zeta)$ for $n \ge 0$ and $\sin(|n| N \zeta)$ for $n < 0$, with
$N$ the number of field periods. $\rho$ is the square root of the normalised
toroidal flux and $\zeta$ the cylindrical toroidal angle. Each mode is an
`(l, m, n)` triple and coefficients are in cm. A rotating ellipse with five
field periods, $R = R_0 + a\rho\cos\theta + e\rho\cos(\theta - N\zeta)$ and
$Z = a\rho\sin\theta - e\rho\sin(\theta - N\zeta)$, is seven coefficients:

```python
import yamc

shape = yamc.sources.StellaratorShape(
    num_field_periods=5,
    r_modes=[(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, -1)],
    r_lmn=[600.0, 150.0, 50.0, 50.0],
    z_modes=[(1, -1, 0), (1, -1, 1), (1, 1, -1)],
    z_lmn=[150.0, -50.0, 50.0],
)

sources = yamc.sources.stellarator_source(
    shape=shape,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_rho=0.8,
)
model = yamc.Model(geometry=geometry, source=sources, tallies=tallies)
```

The constructor refuses a shape whose surfaces reach $R \le 0$, self-intersect
or are not nested. `shape.evaluate(rho, theta, zeta)` gives $(R, Z)$ for floats
or sequences, and `shape.volume()` the plasma volume.

The plasma is split into 50 radial shells in $\rho$, each one source per
reaction. A shell samples positions with density $S(\rho)\, R\, |\tau|$, the
neutron source density times the volume element of the shape, and emits the
Ballabio spectrum at its emission-weighted ion temperature. Strengths are
normalised to sum to 1, and `fuel` works as for the tokamak source.

The profiles are the tokamak ones with the minor radius replaced by $\rho$, so
`pedestal_rho` (between 0 and 1) places the pedestal. To plot them, call
`tokamak_ion_density()` / `tokamak_ion_temperature()` with `minor_radius=1`,
`pedestal_radius=pedestal_rho` and `r` set to the $\rho$ values.

#### From DESC

A DESC equilibrium already is this series. Its coefficients are in m:

<!-- doctest: skip -->
```python
import desc.io

eq = desc.io.load("equilibrium.h5")
shape = yamc.sources.StellaratorShape(
    num_field_periods=int(eq.NFP),
    r_modes=eq.R_basis.modes.tolist(),
    r_lmn=(100 * eq.R_lmn).tolist(),
    z_modes=eq.Z_basis.modes.tolist(),
    z_lmn=(100 * eq.Z_lmn).tolist(),
)
```

The W7-X equilibrium that ships with DESC, converted this way, is in
`docs/data/w7x_fourier_zernike.json`. Its flux surfaces at the start, a
quarter and half way through a field period, and where its neutrons are born:

```python exec="1" html="1" source="above"
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import yamc

shape = yamc.sources.StellaratorShape(**json.load(open("docs/data/w7x_fourier_zernike.json")))
sources = yamc.sources.stellarator_source(
    shape=shape,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_rho=0.8,
)

# Birth sites, shared out between the sources by strength as the transport does.
strengths = np.array([source.strength for source in sources])
counts = np.random.default_rng(1).multinomial(300_000, strengths / strengths.sum())
points = np.concatenate(
    [np.array(source.sample_n(int(n))[0]) for source, n in zip(sources, counts) if n]
)
phi = np.arctan2(points[:, 1], points[:, 0])
major = np.hypot(points[:, 0], points[:, 1])

period = 2 * math.pi / shape.num_field_periods
thetas = np.linspace(0, 2 * math.pi, 200)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, fraction in zip(axes, (0.0, 0.25, 0.5)):
    zeta = fraction * period
    near = np.abs((phi - zeta + math.pi) % (2 * math.pi) - math.pi) < 0.01
    ax.scatter(major[near], points[near, 2], s=1, alpha=0.3, color="tab:orange", rasterized=True)
    for rho in (0.25, 0.5, 0.75, 1.0):
        r, z = shape.evaluate(rho, thetas, zeta)
        ax.plot(r, z, color="k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("R [cm]")
    ax.set_title(f"{fraction:g} of a field period")
axes[0].set_ylabel("Z [cm]")
axes[3].scatter(points[::10, 0], points[::10, 1], s=0.5, alpha=0.2, color="tab:orange", rasterized=True)
axes[3].set_aspect("equal")
axes[3].set_xlabel("x [cm]")
axes[3].set_ylabel("y [cm]")
axes[3].set_title("birth sites, top view")
fig.tight_layout()

from io import StringIO as _S
_b = _S(); plt.savefig(_b, format="svg", bbox_inches="tight"); print(_b.getvalue())
```

Emission is concentrated on the magnetic axis, where the plasma is hottest.

#### From VMEC and VMEC++

A wout file stores $R$ and $Z$ on a set of radial surfaces in the basis
$\cos(m\theta - n\zeta)$ / $\sin(m\theta - n\zeta)$. VMEC2000, VMEC++ and
DESC's VMEC export all write it as classic NetCDF, which
`scipy.io.netcdf_file` reads:

<!-- doctest: skip -->
```python
from scipy.io import netcdf_file

with netcdf_file("wout_example.nc", mmap=False) as wout:
    arrays = {name: wout.variables[name].data.copy() for name in ("nfp", "xm", "xn", "rmnc", "zmns")}
```

The function below turns those arrays into a `StellaratorShape`. It splits
each term into the product basis, then fits every $(m, n)$ coefficient across
the surfaces with Zernike radial polynomials, choosing the degree by
cross-validation on alternate surfaces. It needs only numpy; copy it as it
stands. Here it runs on the arrays of VMEC++'s CTH-like example
(`docs/data/cth_like_wout.json`), and the fitted surfaces (lines) are drawn
over the wout's own (dots):

```python exec="1" html="1" source="above"
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import yamc


def zernike_radial(rho, ell, m):
    """R_l^m(rho) = (-1)^k rho^m P_k^(m, 0)(1 - 2 rho^2), k = (l - m) / 2."""
    x, p_prev, p = 1.0 - 2.0 * rho**2, np.zeros_like(rho), np.ones_like(rho)
    for j in range(1, (ell - m) // 2 + 1):
        if j == 1:
            p_prev, p = p, (m + 1) + 0.5 * (m + 2) * (x - 1.0)
            continue
        s = 2 * j + m
        p_prev, p = p, (
            (s - 1) * (s * (s - 2) * x + m**2) * p - 2 * (j + m - 1) * (j - 1) * s * p_prev
        ) / (2 * j * (j + m) * (s - 2))
    return (-1) ** ((ell - m) // 2) * rho**m * p


def shape_from_wout(nfp, xm, xn, rmnc, zmns, rmns=None, zmnc=None, max_degree=20):
    """A StellaratorShape from wout arrays (m, shape (ns, mn_mode))."""
    rho = np.sqrt(np.linspace(0.0, 1.0, len(rmnc)))
    folds = (np.arange(len(rho))[0::2], np.arange(len(rho))[1::2])

    def fit(values, m):
        def design(r, degree):
            return np.column_stack([zernike_radial(r, m + 2 * i, m) for i in range(degree + 1)])

        def held_out_error(degree):
            return max(
                np.abs(design(rho[b], degree) @ np.linalg.lstsq(design(rho[a], degree), values[a], rcond=None)[0] - values[b]).max()
                for a, b in (folds, folds[::-1])
            )

        degree = min(range(min(max_degree, len(folds[1]) - 2) + 1), key=held_out_error)
        return [m + 2 * i for i in range(degree + 1)], np.linalg.lstsq(design(rho, degree), values, rcond=None)[0]

    shape = {"num_field_periods": int(nfp)}
    for name, cos, sin in (("r", rmnc, rmns), ("z", zmnc, zmns)):
        columns = {}
        for j, (m, n) in enumerate(zip(np.asarray(xm, int), np.asarray(xn, int) // int(nfp))):
            c = 0.0 if cos is None else np.asarray(cos)[:, j]
            s = 0.0 if sin is None else np.asarray(sin)[:, j]
            m, sign, n = int(m), int(np.sign(n)), int(abs(n))
            # cos(a - b) = cos a cos b + sin a sin b, sin(a - b) = sin a cos b - cos a sin b
            terms = [((m, n), c), ((m, -n), -sign * s)]
            if m > 0:
                terms += [((-m, -n), sign * c), ((-m, n), s)]
            for mode, value in terms:
                columns[mode] = columns.get(mode, 0.0) + value
        modes, lmn = [], []
        for (m, n), values in columns.items():
            if np.all(np.asarray(values) == 0.0):
                continue
            ells, coefficients = fit(values, abs(m))
            modes += [(ell, m, n) for ell in ells]
            lmn += (100.0 * coefficients).tolist()
        shape[f"{name}_modes"], shape[f"{name}_lmn"] = modes, lmn
    return yamc.sources.StellaratorShape(**shape)


wout = json.load(open("docs/data/cth_like_wout.json"))
shape = shape_from_wout(**wout)

# The wout's own surfaces, summed from its Fourier coefficients (m to cm).
xm, xn = np.array(wout["xm"]), np.array(wout["xn"])
rmnc, zmns = 100 * np.array(wout["rmnc"]), 100 * np.array(wout["zmns"])
thetas = np.linspace(0, 2 * math.pi, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, zeta in zip(axes, (0.0, math.pi / wout["nfp"])):
    phase = np.outer(thetas, xm) - xn * zeta
    for j in range(0, len(rmnc), 2):
        rho = math.sqrt(j / (len(rmnc) - 1))
        ax.plot(np.cos(phase) @ rmnc[j], np.sin(phase) @ zmns[j], ".", ms=1.5, color="tab:blue", rasterized=True)
        r, z = shape.evaluate(rho, thetas, zeta)
        ax.plot(r, z, color="k", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("R [cm]")
    ax.set_title(f"zeta = {zeta:.2f}")
axes[0].set_ylabel("Z [cm]")
fig.tight_layout()

from io import StringIO as _S
_b = _S(); plt.savefig(_b, format="svg", bbox_inches="tight"); print(_b.getvalue())
```

The fit reproduces the stored surfaces to well under a millimetre, and unlike
interpolation between surfaces it follows the $\rho^{|m|}$ behaviour of the
coefficients near the magnetic axis, where emission peaks. For equilibria with
`lasym` set, pass `rmns` and `zmnc` too.

### Multiple sources and strength weighting

Pass a list of sources with relative strengths to ``Model()``.

<!-- doctest: skip -->
```python
source_a = yamc.NeutronSource(
    position=(0, 0, -7),
    energy=14.06e6,
    strength=2.0,
)
source_b = yamc.NeutronSource(
    position=(0, 0, 7),
    energy=2.5e6,
    strength=1.0,
)
model = yamc.Model(geometry=geometry, source=[source_a, source_b])
```

### VTK-HDF source export

Sampled source points can be exported for [ParaView](https://www.paraview.org/) visualization.

<!-- doctest: skip -->
```python
source.to_vtkhdf("source_points.vtkhdf", samples=1000)
```

The viewer below is rendered live when these docs are built: it samples a tokamak-like cylindrical-ring plasma source, writes ``source_points.vtkhdf``, reads it back, and shows the sampled emission sites as a point cloud coloured by energy (drag to rotate, scroll to zoom).

```python exec="1" html="1"
import html
import math

import numpy as np
import pyvista as pv
from vtkmodules.vtkIOHDF import vtkHDFReader

import yamc

pv.OFF_SCREEN = True
source = yamc.NeutronSource(
    position=yamc.sources.CylindricalRing(
        radius=yamc.sources.Normal(300, 25),        # fuzzy major radius (cm)
        phi=yamc.sources.Uniform(0, 2 * math.pi),   # full ring
        z=yamc.sources.Normal(0, 20),               # spread about the midplane
    ),
    energy=yamc.sources.Normal(14.06e6, 3e5),
)
source.to_vtkhdf("source_points.vtkhdf", samples=4000)

reader = vtkHDFReader()
reader.SetFileName("source_points.vtkhdf")
reader.Update()
points = pv.wrap(reader.GetOutput())

plotter = pv.Plotter(window_size=(760, 560))
plotter.add_mesh(points, scalars="energy", cmap="plasma", render_points_as_spheres=True,
                 point_size=5, show_scalar_bar=False)
plotter.add_axes()
plotter.camera_position = "iso"

doc = plotter.export_html(filename=None).read()
print(
    f'<iframe srcdoc="{html.escape(doc, quote=True)}" width="100%" height="580" '
    'style="border:1px solid #ddd;border-radius:4px"></iframe>'
)
```



