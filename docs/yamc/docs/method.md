# Method

This page describes what yamc computes and in what order. The usage guide
covers how to call it.

yamc is a continuous-energy Monte Carlo code for fixed-source neutron and
photon transport. It has no eigenvalue (criticality) mode: fission progeny are
followed as part of the history that created them, and a model that is
supercritical is stopped with an error rather than run.

## Overview

A run has these stages:

1. **Load.** Each material's nuclides are read from the nuclear data library
   and the cross sections are prepared at the material temperature
   ([Nuclear data](#nuclear-data)).
2. **Source.** Each history starts with one particle drawn from the sources
   ([Sources](#sources)).
3. **Transport.** The particle is moved from collision to collision through
   the geometry ([Geometry](#geometry), [Tracking](#tracking)). At each
   collision a nuclide and a reaction are sampled
   ([Neutron collisions](#neutron-collisions),
   [Photon transport](#photon-transport)), and secondary particles are put on
   a per-history bank.
4. **Scoring.** Tallies are scored along tracks or at collisions
   ([Tallies](#tallies)).
5. **History end.** When the bank is empty, the history's contribution to each
   tally bin is folded into the running statistics
   ([Statistics](#statistics)).
6. **Stop.** The run ends when the particle count is reached, every
   convergence target is met, or the runtime budget is spent, whichever comes
   first.

Transmutation and shutdown dose runs build on this loop
([Transmutation coupling](#transmutation-coupling),
[Shutdown dose rate](#shutdown-dose-rate)).

## Nuclear data

Each nuclide is a directory of Apache Arrow IPC files, one per section:
metadata and temperatures, energy grid, reactions, secondary products and
distributions, unresolved resonance probability tables, total $\nu$, fission
photon data and (opt-in) covariances. Only the sections a run needs are
loaded: transport loads every section but the covariances, which only a
nuclear-data uncertainty run reads, and for a material with a temperature set
only that temperature or the two published ones that bracket it; activation
loads only the reaction cross sections.

### Cross sections

Cross sections are interpolated linearly in energy (lin-lin) everywhere.
Threshold reactions are zero below threshold.

The summed reactions are rebuilt from the partials at load, using the ENDF sum
rules, rather than taken from the file:

- total is elastic plus non-elastic (MT 2 + MT 3);
- absorption is disappearance (MT 27) minus fission, so it never includes
  fission;
- scattering is elastic plus every neutron-emitting non-elastic partial;
- fission is MT 18, or the sum of MT 19/20/21/38 where only the partials are
  given.

A redundant total is skipped when its own levels are present (for example
MT 16 next to MT 875 to 890), so nothing is counted twice.

There are two lookup levels:

- **Material.** Each material has a union grid of its nuclides' grids, with
  the macroscopic total $\Sigma_t$ and each nuclide's macroscopic total on it.
  These give the flight distance and the nuclide choice. The macroscopic sum
  is taken in a fixed nuclide order so that it is reproducible.
- **Nuclide.** Each nuclide keeps its own grid, holding total, absorption,
  scattering and fission, plus the per-reaction partials. These give the
  reaction choice and the secondary channel.

Both use a logarithmic index (8000 to 8192 bins) to bracket the energy,
followed by a short search. Energies off the end of a grid take the end value.

### Temperature

If a material's temperature matches a library temperature, that data is used.
If it falls between two library temperatures, a new temperature is made once
at load by blending the two data sets pointwise, linearly in $T$, on the union
of their grids. A temperature outside the library range is an error.

The blend is deterministic. Picking one of the two bracketing temperatures at
random for each lookup would bias deep-penetration results, because the
attenuation through a thick shield is an average of exponentials, not the
exponential of the average cross section. Probability tables are not blended;
the new temperature takes the nearer neighbour's tables.

There is no on-the-fly Doppler broadening. The default temperature is 294 K.

## Sources

A history starts by choosing a source in proportion to its strength, then
drawing its position, direction and energy.

- **Position:** a point, or a cylindrical ring with independent $r$,
  $\varphi$ and $z$ distributions about an axis, or a mesh. On a mesh the
  element is chosen by a binary search over the cumulative strengths, and the
  point is uniform within it: directly in a box, uniform in $r^2$ (and in
  $\varphi$ and $z$) in a cylindrical cell, and from sorted uniform
  barycentric weights in a tetrahedron. With ``cells`` or ``materials`` given,
  points outside them are rejected and redrawn in the same element.
- **Direction:** isotropic or monodirectional.
- **Energy:** discrete lines and histogram bins are drawn by the alias
  method. Uniform, normal and the D-D and D-T fusion spectra of
  [Ballabio et al. 1998](#references) are also available. The default neutron
  energy is 14.06 MeV.

The tokamak source discretises the plasma profiles of
[Fausser et al. 2012](#references) into one ring source per (voxel,
reaction), each with a Ballabio spectrum at the voxel's ion temperature.

The stellarator source takes the plasma shape as a Fourier-Zernike series in
the conventions of [DESC](#references), $R$ and $Z$ as functions of $\rho$,
$\theta$ and $\zeta$, evaluated with a Jacobi recurrence for the Zernike
radial polynomials. The volume element $R\,\tau$, with
$\tau = R_\rho Z_\theta - R_\theta Z_\rho$, is evaluated analytically, and
a shape whose volume element changes sign (self-intersecting or non-nested
surfaces) is refused. The plasma is split into radial shells; shell strengths
are Gauss-Legendre integrals in $\rho$ of the emission density times
$\mathrm{d}V/\mathrm{d}\rho$, the angular integral taken by a trapezoid
rule fine enough to be exact for the trigonometric polynomial $R\,\tau$.
Within a shell, $\rho$ is proposed with density $\propto\rho$ and the
angles uniformly over one field period, accepted against a majorant of
$S(\rho)\,R\,|\tau|/\rho$, and the field period is then chosen at random.

Source strength is a relative selection probability only. Results are per
source particle and are scaled to an absolute rate afterwards.

## Geometry

### CSG

Surfaces are planes, spheres, cylinders and cones on any axis, elliptical tori
about x, y or z, and general quadrics. Distances to tori come from a
closed-form quartic solve. A cell's region is a tree of half-spaces combined by
intersection, union and complement. At setup it is compiled into a flat
reverse-Polish program, so a point-in-cell test needs no recursion and no
allocation.

To find the cell containing a point, yamc first tries the particle's current
cell, then the neighbours learned for that cell during the run
([Harper et al. 2020](#references)), then a bounding-volume hierarchy over the
cells' bounding boxes. Bounding boxes are computed recursively from the
region tree, so ring and frame cells built from unions and complements still
have finite boxes.

The nearest surface along the flight is the smallest positive distance over
the cell's surfaces. After a crossing the particle is moved $10^{-8}$ cm past
the surface and its cell is found again.

### Mesh geometry

Triangulated surface meshes are tracked directly, with one surface BVH per
volume: a binned SAH build, collapsed to a 4-wide tree tested with SIMD.
Tetrahedra, when present, are used for unstructured mesh tallies, not for
locating particles. The space between meshed volumes is an implicit
complement, void unless given a material. After a crossing the next volume
comes from the surface-to-volume topology, so no point location is needed.

Before a crossing is accepted, the segment is checked for any other surface
it passes through. A segment that skips a surface is recorded as lost rather
than continuing in the wrong volume.

A defect in the mesh loses the particle where the defect is, rather than
letting it fly on in the wrong volume:

- **A point the mesh cannot place.** Point location (at a source site, or
  after a Woodcock flight) casts test rays and takes the parity of their
  crossings. A ray whose parity disagrees with the way the nearest facet along
  it faces (a stray facet, a flipped facet or a crack) is passed over. A point
  that no ray places consistently is lost there, instead of being put in the
  implicit complement and carried on as void.
- **A crack in a volume's surface.** A particle that slips out through a
  missing facet next meets the volume's far side from outside. That facet is
  checked just short of the crossing: if the particle is still inside the
  volume, the facet is only wound the wrong way and the particle leaves
  through it; if it is outside, the particle left through the crack and is
  lost where it left.

A sound mesh takes neither path, so these checks cost it nothing.

### Mesh inside CSG

A mesh can fill a CSG cell, with a translation and rotation. Its volumes
become cells inside the host, and the host's material fills the space between
them. The next boundary is the nearer of the CSG exit and the mesh crossing.
The mesh crossing wins only if it is at least $10^{-8}$ cm nearer, so a mesh
face lying on the host surface does not trap the particle. A mesh that sticks
out of its host is refused unless clipping is explicitly allowed.

### Boundaries and lost particles

A surface is either transmitting (the default) or vacuum. A particle that
crosses a vacuum surface leaks and is killed. There are no reflective,
white or periodic boundaries.

A particle whose cell cannot be found is recorded as lost (position,
direction, energy, last cell and surface) and killed. The run stops once more
than `max_lost_particles` (default 10) are lost, because a lost particle
almost always means a geometry error rather than a rare event.

## Tracking

The distance to the next collision is

$$
d = -\frac{\ln \xi}{\Sigma_t(E)},
$$

with $\xi$ uniform on $(0, 1]$. Three tracking modes are available.

**Surface tracking** (the default) compares $d$ with the distance to the
nearest surface. If the surface is nearer, the particle crosses it and $d$ is
redrawn in the next cell; otherwise it collides.

**Woodcock (delta) tracking** samples $d$ against a majorant
$\Sigma_\text{maj}(E) \ge \Sigma_t(E)$ that holds in every material, moves the
particle without looking at surfaces, then accepts a real collision with
probability $\Sigma_t / \Sigma_\text{maj}$. Otherwise the collision is virtual
and the flight continues at the same energy. A landing point in void is always
virtual. The majorant is global: the union of all the materials' energy grids,
with the largest macroscopic total at each point, using the worst-case
probability-table band inside the unresolved range. A per-material majorant
would not be a bound once the flight crosses into another material. Vacuum
exits are still found exactly by testing the flight against the vacuum
surfaces only.

**Hybrid tracking** chooses per cell. A cell is surface-tracked when the
expected number of virtual collisions across it,
$(\Sigma_\text{maj} - \Sigma_t)\,\bar{\ell}$, exceeds 1, where $\bar{\ell} = 4V/S$
is the mean chord of its bounding box. Thin, dense cells in a mostly light
model are then tracked by surfaces, and everything else by Woodcock.

Woodcock and hybrid tracking are CPU only and do not support mesh fills.

## Neutron collisions

### Choosing the nuclide and the reaction

The nuclide is chosen in proportion to its macroscopic total. The reaction is
then chosen from four partials: elastic, inelastic (every neutron-emitting
non-elastic reaction), fission and absorption. Within inelastic or absorption,
the specific reaction is chosen in proportion to its cross section.

### Elastic scattering

Below $400\,kT$ (the `free_gas_threshold`), the target's thermal motion is
sampled from a free gas with the constant cross section approximation. For
hydrogen (atomic weight ratio $A \le 1$) it is always sampled. Above the
threshold the target is at rest. The centre-of-mass angle comes from the
tabulated distribution, and the outgoing energy and lab angle from two-body
kinematics:

$$
E' = E\,\frac{A^2 + 2A\mu + 1}{(A+1)^2}, \qquad
\mu_\text{lab} = \frac{1 + A\mu}{\sqrt{A^2 + 2A\mu + 1}}.
$$

Thermal scattering laws ($S(\alpha,\beta)$) and resonance-corrected elastic
kernels (DBRC, RVS) are not used.

### Unresolved resonances

In the unresolved resonance range, cross sections come from probability
tables. One random number is drawn when the particle reaches a new energy and
is held until the energy changes, including across surfaces and through void,
so the flight, the nuclide choice, the reaction choice, the secondary photon
production, the tallies and the transmutation reaction rates all see the same
band. Each nuclide derives its own band from that number with a hash
of its ZA, so the bands of different nuclides are independent.

The band is read at the two bracketing table energies with the same random
number and interpolated between them. The band total is always the sum of its
partials (elastic, inelastic, capture, fission), not the table's own total
column. Where the tables are relative to the smooth cross sections, capture is
scaled from the smooth absorption, which in yamc already excludes fission, so
fission is not subtracted twice.

### Secondary neutrons

Outgoing angle and energy come from the product distributions in the data:
discrete levels, tabulated and continuous tabular energy laws, Maxwell, Watt
and evaporation spectra, Kalbach-Mann, correlated angle-energy and N-body
phase space. Tabulated distributions pick an incident-energy bracket at random
and scale the sampled energy to the interpolated bracket limits. Laws given in
the centre-of-mass frame are transformed to the lab with

$$
E_\text{lab} = E_\text{cm} + \frac{E + 2\mu_\text{cm}(A+1)\sqrt{E E_\text{cm}}}{(A+1)^2},
\qquad
\mu_\text{lab} = \mu_\text{cm}\sqrt{\frac{E_\text{cm}}{E_\text{lab}}} + \frac{1}{A+1}\sqrt{\frac{E}{E_\text{lab}}}.
$$

For an integer neutron yield, such as (n,2n), the extra neutrons are banked
at the particle's weight, which is analog. A non-integer yield multiplies the
weight instead. Secondaries go on a last-in, first-out bank and are followed
to completion within the same history.

### Fission

The number of neutrons is $\bar\nu_\text{total}(E)$, rounded stochastically.
Each neutron is delayed with probability $\beta(E) = \bar\nu_d(E)/\bar\nu_\text{total}(E)$.
Delayed neutrons are emitted at once (there is no precursor time) from a single
delayed spectrum: the six group spectra summed with their group yields.
Prompt neutrons are drawn from the prompt spectrum of the sampled fission
channel. Emission is isotropic in the lab. Using the prompt spectrum for every
neutron would make the fission spectrum too hard.

A history whose live fission bank passes 20,000 neutrons is stopped with an
error, because the model is then supercritical or close to it.

### Energy limits

There is no neutron energy cutoff. Neutrons are followed until they are
absorbed or leak.

## Photon transport

The element is chosen in proportion to its macroscopic total, then the
interaction from its cross sections (interpolated log-log):

- **Coherent (Rayleigh):** angle from the integrated form factor, then
  rejection on $(1+\mu^2)/2$. The energy is unchanged.
- **Incoherent (Compton):** Klein-Nishina with rejection on the incoherent
  scattering function. Doppler broadening is applied using the relativistic
  impulse approximation of [Kaltiaisenaho 2016](#references), with the Compton
  profiles of [Biggs et al. 1975](#references). The vacancy left in the
  struck shell is relaxed.
- **Photoelectric:** the subshell is chosen from the subshell cross sections,
  the photoelectron gets $E - E_b$ and leaves at a Sauter angle about the
  photon's direction, and the vacancy is relaxed.
- **Pair production:** the electron and positron energies are sampled with the
  PENELOPE method ([Salvat 2011](#references)). The positron annihilates at
  rest into two back-to-back 511 keV photons. Triplet production is included
  in the pair cross section.

Atomic relaxation follows the full cascade of radiative and non-radiative
transitions. Fluorescence photons are banked and Auger electrons are treated
as electrons, below.

Electrons and positrons are not transported. By default their energy is
converted to bremsstrahlung photons using thick-target bremsstrahlung (TTB)
yields from [Seltzer and Berger 1986](#references); the photons start at the
electron's position and direction and the rest of the energy is deposited
locally. With `electron_treatment="local"` all of it is deposited locally.

Photons below `photon_cutoff_energy` (default 1 keV) are killed.

The photon data are read directly from the ENDF photo-atomic and atomic
relaxation sublibraries, with no NJOY step.

## Coupled neutron-photon transport

With secondary photons on, photons are sampled at every neutron collision,
before the reaction is processed. The mean number is
$\sigma_{\gamma,\text{prod}}/\sigma_t$, made an integer by a Bernoulli draw on
its fractional part, and each photon carries the neutron's weight. Here
$\sigma_t$ is the total the collision was sampled from; in the unresolved range
that is the probability-table band's, and each reaction's photon production is
moved by its own channel's band factor (capture by the capture factor, and so
on). The producing reaction is chosen in proportion to its photon production, and the
photon's energy and angle come from that reaction's distribution, relative to
the incoming neutron direction. Fission photon yields are scaled up by
$(\bar\nu_p + \bar\nu_d)/\bar\nu_p$ to include delayed fission photons.

## Variance reduction

The default is fully analog. Two methods can be switched on. Survival biasing
runs on the CPU and the GPU, and weight windows on the CPU only.

### Survival biasing

At a neutron collision absorption is not sampled. The weight is multiplied by
the scattering probability, $(\sigma_e + \sigma_i)/\sigma_t$, and fission neutrons
are banked from the weight before that discount, in proportion to
$\sigma_f/\sigma_t$. Secondary photons are sampled before the discount. Below
`weight_cutoff` (default 0.25) the particle plays Russian roulette and
survives with probability $w/w_s$ at weight $w_s$ (`weight_survive`,
default 1). Photons stay analog.

### Weight windows

A weight window gives lower and upper weight bounds per mesh voxel and energy
group, for each particle type. Windows are applied after each collision:

- above the upper bound, the particle is split into
  $n = \min(\lceil w / w_u \rceil, n_\text{max})$ copies of weight $w/n$, with
  $n_\text{max}$ = `max_split` (default 10);
- at or below the lower bound, it plays roulette with survival weight
  $\min(w\,n_\text{max},\; c\,w_l)$, with $c$ = `survival_factor` (default 3).

Splitting stops for a history once it has made 10,000 extra particles, which
caps the cost of a badly placed window without changing the expected result.

### Generating windows (DeGVR)

Windows are generated by density-extrapolation global variance reduction
([Pan et al. 2023](#references)). The model is first run analog with every
density divided by $N$, which is cheap because particles go further. A second
run at density divided by $N/2$ uses the first run's windows. With $F$ and $H$
the two flux maps, the flux at full density is extrapolated voxel by voxel as

$$
G = F\left(\frac{H}{F}\right)^{N-1}
$$

in log space. The lower bound is the extrapolated flux divided by the survival
factor and the group maximum; the upper bound is 5 times the lower.

If `density_reduction` is not given, $N$ is chosen without transport: a ray
trace from the strongest source to the voxel centres gives the optical depth
$\tau$ at the 90th percentile, and $N = \max(\tau / 7,\; 2)$. Coupled
neutron-photon windows are generated from one pair of runs with a flux mesh
per particle type.

## Tallies

### Estimators

A tally score has the form $R(E)$ times a flux contribution, where $R$ is 1
for flux, a macroscopic cross section for a reaction rate, or a KERMA factor
for heating. Two estimators are available.

**Track-length** (the default) scores every track segment:

$$
\text{score} = w\,\ell\,R(E),
$$

with $w$ the weight and $\ell$ the length of the segment in the bin.

**Collision** scores every real collision:

$$
\text{score} = w\,\frac{R(E)}{\Sigma_t(E)},
$$

with $\Sigma_t$ the total the collision was sampled from, including the
probability-table band in the unresolved range.

Under Woodcock tracking there is no track in a cell to score, so track-length
tallies switch to the collision-density estimator, scored at every collision,
real or virtual:

$$
\text{score} = w\,\frac{R(E)}{\Sigma_\text{maj}(E)}.
$$

The exception is a flux tally on a mesh with no cell or material filter. Its
true track length can be found by walking the flight through the mesh, so it
keeps the track-length estimator.

### Mesh tallies

A track crossing a mesh is split between voxels in proportion to the length in
each. Regular meshes are walked with the Amanatides-Woo traversal
([Amanatides and Woo 1987](#references)); cylindrical meshes are intersected
analytically; unstructured tetrahedral meshes are walked element to element.

### Heating

Neutron heating uses the KERMA cross sections (MT 301, or MT 901 for local
heating). Photon heating depends on the estimator:

- **Collision:** the energy absorbed at each collision,
  $w\,(E_\text{in} - E_\text{out} - \sum E_\text{banked photons})$. This is
  analog.
- **Track-length:** the photon KERMA, built from the cross sections: for
  Compton, the mean energy transfer of Klein-Nishina scattering weighted by the
  incoherent scattering function (the distribution Compton scattering samples),
  less the fluorescence energy; for the photoelectric effect, $E$ less the expected fluorescence
  energy; for pair production, $E - 2m_ec^2$. Coherent scattering deposits
  nothing.

Both include the energy of the electrons. They differ in where the
bremsstrahlung energy ends up: the collision estimator follows the TTB photons,
the KERMA counts that energy where the electron was made. In a thick body they
agree to about 1%. In a thin or streaming one the KERMA is higher by the
fraction that escapes, which is the usual difference between kerma and
absorbed dose.

### Dose

Dose is a flux tally weighted by a dose coefficient $f(E)$: ICRP 74 or
ICRP 116 effective dose per fluence, for a chosen irradiation geometry. The
coefficients are interpolated log-log, the way the tables are read everywhere
else; events outside the table are not scored.

## Statistics

### Mean and standard deviation

One history is one sample. During a history, its contributions to each bin are
summed in a scratch buffer, and at the end of the history that sum $x$ is
folded into the bin's running mean $\mu$ and sum of squared deviations $M_2$
with Welford's update:

$$
\delta = x - \mu, \qquad \mu \leftarrow \mu + \frac{\delta}{n}, \qquad M_2 \leftarrow M_2 + \delta\,(x - \mu).
$$

Histories that do not touch a bin are folded in as zeros at the end, in one
step. Each thread keeps its own accumulators, which are merged pairwise with
the parallel form of the update ([Chan et al. 1979](#references)), so the result is exact in real
arithmetic whatever the split.

The reported standard deviation is that of the mean,

$$
s_{\bar x} = \sqrt{\frac{M_2}{N(N-1)}},
$$

and the relative error is $s_{\bar x}/\bar x$. There are no batches and no
batch size to choose.

This sample-per-history estimate is correct when secondaries are correlated
with their parent, as fission and (n,xn) progeny are, because all of a
history's particles add to the same sample.

The GPU accumulates $\sum x$ and $\sum x^2$ per bin instead and forms
$M_2 = \sum x^2 - (\sum x)^2/N$.

### Convergence checks

For each tally, the per-history total over all bins is also tracked, with its
third and fourth moments. From it yamc reports:

- the **variance of the variance**, $\text{VoV} = M_4/M_2^2 - 1/N$;
- the **figure of merit**, $\text{FOM} = 1/(R^2 t)$, with $R$ the relative
  error and $t$ the wall-clock time;
- the **tail slope** of the score distribution, from a Hill estimate over the
  201 largest scores.

The checks passed are: relative error below 0.1, in total and in every
populated bin, VoV below 0.1, tail slope above 3, and, over the second half of
the run, a mean that is not trending (a 95% Student-t test on the slope of the
means of the blocks of histories between snapshots), relative error and VoV
that do not grow, a FOM that stays within a factor of 2,
a relative error falling as $1/\sqrt{N}$ (a log-log slope between $-0.75$ and
$-0.25$), a VoV falling as $1/N$ (a slope between $-1.5$ and $-0.5$) and a FOM
with no trend in $N$ (a slope within $\pm 0.5$). The three rate checks read as
not evaluated with fewer than four snapshots in that half, and both FOM checks
also when the half spans less than 0.1 s. A tally passes only if its tail slope
could be estimated.

A run can stop early on convergence targets: a relative error, standard
deviation or VoV threshold, on all tallies or a named one. Targets are checked
between chunks of histories, as is the runtime budget, so a run can go over
either by up to one chunk.

### Bin covariance

On request, a tally also accumulates $P_{ij} = \sum_h x_{hi} x_{hj}$ over
histories, giving

$$
\operatorname{Cov}(\bar x_i, \bar x_j) = \frac{P_{ij} - N\bar x_i \bar x_j}{N(N-1)},
$$

for up to 2048 bins.

### Combining runs

Separate runs of the same model can be combined exactly: the means, $M_2$,
higher moments, covariance products and score distributions are merged with
the same pairwise update. Runs are refused if they could share random number
streams, if the model, data library or tally definitions differ, if they read
different published releases of one library (each run records its releases in
``runs[i]["data_releases"]``), or if they came from the GPU.

## Random numbers and reproducibility

Every history has its own random number stream, seeded from the base seed and
the history's index in the run. The thread, chunk and MPI rank it runs on do
not enter, so a history is the same whatever the parallel layout. Collisions
use a 64-bit PCG generator shared with the GPU kernels. Each secondary gets
its own seed from its parent's seed and its birth order, so the order in which
the bank is emptied does not change it. The CPU empties the bank last in,
first out and the GPU first in, first out, and both follow the same histories.

Histories run in parallel on a rayon thread pool and, with MPI, across ranks.
Per-thread statistics are merged at the end. Because floating-point addition
is not associative, results are statistically identical across thread counts
but may differ in the last bits.

## Transmutation coupling

Reaction rates for transmutation are scored directly from continuous-energy
cross sections, $w\,\ell\,\sigma(E)$ per nuclide and reaction on every track,
not by folding a group flux. In the unresolved range $\sigma$ is the
probability-table band the flight drew, as for the other tallies. They are normalised per source particle, then by
the step's source rate and the material volume. Fission yields are weighted by
the continuous-energy fission rate. Isomeric branching ratios are applied at
the collision energies.

A short scouting run first finds the nuclides the material can populate, so
only those are tallied. Each step is then solved with the
[yani](https://fusion-neutronics.github.io/yamc-and-yani/yani/) CRAM-48 solver
([Pusa 2016](#references)). In "coupled" mode transport is re-run every step
with the updated compositions, using the rates at the start of the step. In
"independent" mode transport runs once and its rates are scaled by each step's
source rate.

## Shutdown dose rate

Shutdown dose uses the direct one-step method (D1S). During transport,
activation reactions that lead (through any decay chain) to a photon emitter
are treated as prompt photon sources. One decay photon is emitted per
collision, isotropically, with weight $w\,\sigma_\text{D1S}/\sigma_t$, and
tagged with its decay path: the emitter alone when the reaction makes it, the
chain from the reaction product to the emitter (``Cs137>Ba137m``) when it is
made through decay. The path's branching ratios are in the photon yield. After
the run, each path's bin is multiplied by its time correction factor, the
Bateman activity along that path over the irradiation and cooling schedule,
which carries the time dependence alone. One transport run therefore gives
the dose at any number of cooling times.

When the machine changes between steps (`simulate_shutdown_dose`), the run is
split in two. A neutron run per irradiating model records each decay photon at
birth (history, position, direction, energy, weight, decay path and cell) instead of
tracking it. A photon run per observing configuration $o$ then makes photon
history $i$ out of every photon that neutron history $i$ recorded, in every
irradiating configuration $c$, each moved by its cell's rigid motion
$T_o T_c^{-1}$ (direction by the rotation alone; dropped if the cell is absent)
and transported once at its own weight. The history's dose at the end of each
step $s$ that $o$ observes is formed at the end of the history,

$$
D_i(s) = \sum_c \sum_p \mathrm{TCF}_{c,p}(s)\, x_{i,c,p},
$$

where $x_{i,c,p}$ is the history's tally of the photons of decay path $p$ recorded
in configuration $c$, and $\mathrm{TCF}_{c,p}$ uses only that configuration's
pulses. The histories are independent, so the mean and standard error of
$D_i(s)$ over them are those of the whole calculation, the neutron stage and
every correlation within a history included. The photon runs use random-number
streams independent of the neutron runs', since transporting history $i$'s
photons on the stream that made them would correlate the two.

The rigorous two-step method (R2S, [Chen and Fischer 2002](#references)) is
offered beside the one-step method. One neutron transport tallies the spectrum of every
activated region, each region's inventory is solved through the schedule with
the transmutation solver, and a photon transport per step samples the decay
photons uniformly over each region at that step's activity, with the line and
continuum spectra of the decay data. A region is a material-filled cell, or with
a mesh each part of such a cell under a voxel (cell-under-mesh), its spectrum
then the voxel's in VITAMIN-J-175 groups and its volume estimated from $8^3$
jittered points per voxel. Products that are themselves activated, and burnup,
enter through the inventory solve.

## Nuclear-data uncertainty

Transport can estimate the uncertainty from the cross-section covariances in
one run. Each history carries a likelihood ratio for each of a set of
perturbed data replicas ([Rief 1984](#references)): with surface tracking a
flight of length $d$ multiplies it by $e^{-\Delta\Sigma_t d}$ and a collision by
$\sigma'_\text{MT}/\sigma_\text{MT}$. Under delta tracking the majorant is
raised to cover every replica's total, so a flight moves no ratio, a real
collision takes the same $\sigma'_\text{MT}/\sigma_\text{MT}$ and a virtual one
multiplies it by $(\Sigma_\text{maj}-\Sigma'_t)/(\Sigma_\text{maj}-\Sigma_t)$.
Track-length scores are weighted by the mean survival over the track.
Collision-estimator scores take the ratio a history arrives with, before the
sampled reaction changes it: the collision density's change in $\Sigma_t$
cancels the estimator's $1/\Sigma_t$. With either estimator, a neutron's
reaction-rate, production, heating or damage-energy score is also multiplied
by the change to its own response, $\Sigma'_x/\Sigma_x$. The
spread of the replica results includes the Monte Carlo noise of the shared
histories, which is estimated and subtracted
([Clements et al. 2024](#references)).

The MF=33 cross-section covariances are sampled, and where a nuclide's data
carries its resonance parameters (MF=2 and MF=32), each replica draws those and
rebuilds the resonance cross sections from the draw. Heating and damage energy
move with the reactions that deposit them, apart from fission's share, which
stays nominal. Secondary distributions, $\bar\nu$ and fission spectra are not
perturbed, and photon data carries no covariance: a photon inherits its
neutron's ratios times the change to the cross section of the reaction that
made it. The ratios are carried through every tracking mode, survival biasing,
weight windows and MPI. GPU runs, and overlay, mesh and per-nuclide tallies,
are refused.

A transmutation with the "independent" method takes its uncertainty from its
one transport: the replica weights above give each replica's rates, the
per-history covariance of the tallies gives their statistical spread, and each
replica's schedule is solved again from them. The "coupled" method cannot do
this, because each step's rates were tallied in compositions that the earlier
steps' noise and data had already moved. Each replica instead repeats the whole
coupled calculation, as in Total Monte Carlo
([Koning and Rochman 2008](#references)): its own transport seed, its own cross
sections and isomeric production curves in every transport, and its own
half-lives, decay branching ratios and fission yields in every solve. The
spread over the replicas then carries the feedback through the compositions
exactly, with nothing linearised. A replica transported on its own cross
sections follows its own histories whatever its seed, so its spread holds the
transport's noise as well, which is why the coupled method carries the cross
sections only beside the statistical source.

## References

- J. Amanatides and A. Woo, "A fast voxel traversal algorithm for ray
  tracing", Eurographics 1987 Technical Papers (1987),
  [doi:10.2312/egtp.19871000](https://doi.org/10.2312/egtp.19871000).
- L. Ballabio, J. Källne and G. Gorini, "Relativistic calculation of fusion
  product spectra for thermonuclear plasmas", Nucl. Fusion 38 (1998)
  1723-1735.
- F. Biggs, L. B. Mendelsohn and J. B. Mann, "Hartree-Fock Compton profiles
  for the elements", At. Data Nucl. Data Tables 16 (1975) 201-309.
- T. F. Chan, G. H. Golub and R. J. LeVeque, "Updating formulae and a
  pairwise algorithm for computing sample variances", Stanford University
  report STAN-CS-79-773 (1979).
- Y. Chen and U. Fischer, "Rigorous MCNP based shutdown dose rate
  calculations: computational scheme, verification calculations and
  application to ITER", Fusion Eng. Des. 63-64 (2002) 107-114.
- K. B. Clements, G. Geraci, A. J. Olson and T. S. Palmer, "A variance
  deconvolution estimator for efficient uncertainty quantification in Monte
  Carlo radiation transport applications", J. Quant. Spectrosc. Radiat.
  Transf. 319 (2024) 108958.
- D. W. Dudt and E. Kolemen, "DESC: A stellarator equilibrium solver", Phys.
  Plasmas 27 (2020) 102513,
  [doi:10.1063/5.0020743](https://doi.org/10.1063/5.0020743).
- C. Fausser et al., "Tokamak D-T neutron source models for different plasma
  physics confinement modes", Fusion Eng. Des. 87 (2012) 787,
  [doi:10.1016/j.fusengdes.2012.02.025](https://doi.org/10.1016/j.fusengdes.2012.02.025).
- S. Harper, P. Romano, B. Forget and K. S. Smith, "Threadsafe dynamic
  neighbor lists for Monte Carlo ray tracing", Nucl. Sci. Eng. 194
  (2020) 1009-1015.
- T. Kaltiaisenaho, "Implementing a photon physics model in Serpent 2",
  M.Sc. thesis, Aalto University (2016).
- A. J. Koning and D. Rochman, "Towards sustainable nuclear energy: Putting
  nuclear physics to work", Ann. Nucl. Energy 35 (2008) 2024-2030,
  [doi:10.1016/j.anucene.2008.06.004](https://doi.org/10.1016/j.anucene.2008.06.004).
- Q. Pan, L. Wang, Y. Cai, X. Liu and J. Xiong, "Density-extrapolation Global
  Variance Reduction (DeGVR) method for large-scale radiation field
  calculation", Comput. Math. Appl. 143 (2023) 10-22.
- M. Pusa, "Higher-order Chebyshev rational approximation method and
  application to burnup equations", Nucl. Sci. Eng. 182 (2016) 297-318,
  [doi:10.13182/NSE15-26](https://doi.org/10.13182/NSE15-26).
- H. Rief, "Generalized Monte Carlo perturbation algorithms for correlated
  sampling and a second-order Taylor series approach", Ann. Nucl. Energy 11
  (1984) 455-476.
- F. Salvat, J. M. Fernández-Varea and J. Sempau, "PENELOPE-2011: a code
  system for Monte Carlo simulation of electron and photon transport",
  OECD/NEA (2011).
- S. M. Seltzer and M. J. Berger, "Bremsstrahlung energy spectra from
  electrons with kinetic energy 1 keV-10 GeV incident on screened nuclei and
  orbital electrons of neutral atoms with Z = 1-100", At. Data Nucl. Data
  Tables 35 (1986) 345-418,
  [doi:10.1016/0092-640X(86)90014-8](https://doi.org/10.1016/0092-640X(86)90014-8).
