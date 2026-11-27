# Method

This page describes what YANI computes and in what order. The
[overview](index.md) gives the governing equation. Where a cheaper option was
tried and measured, the result is recorded here as well.

## Overview

A run has six stages:

1. **Network.** The transmutation chain gives each nuclide's decay modes,
   reaction products and fission yields. YANI finds the products the material
   populates over the schedule by solving it, and loads cross sections for
   those and the composition only
   ([Which nuclides get loaded](#which-nuclides-get-loaded)).
2. **Rates.** Each reaction rate per atom is the pointwise cross section folded
   against the spectrum ([The collapse](#the-collapse)), with an optional
   self-shielding correction ([Self-shielding](#self-shielding)). Fission
   yields and isomeric splits are weighted by the same spectrum.
3. **Matrix.** Decay constants and reaction rates are assembled into the sparse
   matrix $A$ ([Building the matrix](#building-the-matrix)).
4. **Solve.** Each step of the schedule is one matrix exponential
   ([The solve](#the-solve), [Schedules](#schedules)).
5. **Derived quantities.** Activity, decay heat, photon emission, contact dose
   and clearance indices are calculated from each step's inventory and the
   decay data
   ([Derived quantities](#derived-quantities)).
6. **Uncertainty.** If requested, stages 2 to 5 are repeated with perturbed
   inputs ([Nuclear-data uncertainty](#nuclear-data-uncertainty)).

## The solve

One timestep is a matrix exponential, $\mathbf{N}(t) = e^{At}\mathbf{N}(0)$,
evaluated by CRAM at order 48 following [Pusa 2010][pusa2010] and
[Pusa 2015][pusa2015], in the incomplete partial fraction form.

$A$ is built from decay constants and $\sigma\phi$ rates and does not depend on
the composition, so it is constant over a step and the exponential is solved
exactly. Without self-shielding, step length therefore does not affect accuracy
for a fixed spectrum: one year in a single step and one year in 365 steps agree
to round-off.

With self-shielding the rates do depend on the composition, through the flux
depression the material's own resonances cause. Each step's rates are collapsed
at the composition the step starts from and held over the step, so the step
length sets how closely the rates follow the burnup, and refining the schedule
converges at first order. On a natural tungsten first wall over five full-power
years, ten steps put Re within 0.05% and Os within 0.25% of a converged
schedule.

What a long step does not capture is a change in the spectrum caused by the
changing composition. `transmute()` takes the flux as an input and does not
solve for it. If the spectrum would harden or soften appreciably over a
campaign, split the schedule into several pulses, each with its own `source`.

The 24 poles are evaluated sequentially, because the incomplete partial fraction
form is a recurrence. The sparse LU runs single-threaded so that the order of
floating point additions, and therefore the result, does not depend on the
thread count. The symbolic factorization is computed once and reused for every
pole.

Most of a transmutation network is acyclic, since decay chains only run one
way. The linear solve at each pole is therefore done block by block over the
strongly connected components of the network, in topological order. Blocks
without a cycle are solved by substitution, and only blocks with a cycle (for
example a capture followed by a decay back to the parent) use the sparse LU.

After the exponential, two adjustments are made:

- A nuclide with no parent present in the step is set to its closed-form value
  $N_i(0)\,e^{A_{ii}t}$ instead of the CRAM value. Both describe the same
  equation, and the closed form is exact, so a stable nuclide that is only
  being burned does not accumulate round-off.
- Densities at or below $10^{-30}$ atoms/b-cm are dropped.

## Building the matrix

The coefficients of $A$ come from two sources:

- **The transmutation network** gives decay constants, decay branching,
  reaction products and fission yields. Each of these can come from a different
  library through four independent settings. See
  [Nuclear data settings](usage.md#nuclear-data-settings).
- **The collapse** gives each $\sigma\phi$ (see [The collapse](#the-collapse)).

$A$ has one column per parent nuclide $j$:

- **Decay.** $\lambda_j = \ln 2 / T_{1/2}$ is added to the diagonal as a loss,
  and each decay mode adds $b\,\lambda_j$ to its daughter's row, where $b$ is
  the mode's branching ratio. Emitted light particles are also produced, so an
  alpha decay produces He4. A mode with no daughter in the chain, such as
  spontaneous fission, removes atoms without producing any.
- **Reactions.** Each reaction rate $R$ is added to the diagonal as a loss, and
  $b\,R$ is added to each product's row. The light ejectiles named by the
  reaction (H1, H2, H3, He3, He4) are also produced, so gas production is part
  of the inventory. A reaction whose product is the target itself cancels
  against its own loss term.
- **Fission.** The fission rate $R_f$ is a loss, and each fission product $p$
  gains $R_f\,Y(p)$, where $Y$ is the spectrum-weighted yield described in
  [Fission yields](#fission-yields).

Stable nuclides have no decay term and are removed only by reactions.

### The collapse

Each $\sigma\phi$ is the pointwise cross section folded against the spectrum:

$$
\sigma_{\text{eff}} = \frac{\sum_g \sigma_g \phi_g}{\sum_g \phi_g},
\qquad
R = 10^{-24}\,\sigma_{\text{eff}} \sum_g \phi_g = 10^{-24} \int \sigma(E)\,\psi(E)\,\mathrm{d}E
$$

where $\psi$ is the piecewise-constant flux density given by the group fluxes.
The total flux cancels, so the rate is the integral of the cross section over
the spectrum. The cross section is used at the energy resolution of the
evaluation, so the group structure of the spectrum is the only averaging
applied.

Each $\sigma_g$ is the integral of the cross section over the group divided by
the group width. The integral uses the trapezoid rule over the group edges and
every evaluated point inside the group. Cross sections are interpolated
linearly between points, so the trapezoid rule is exact.

At the ends of the evaluated energy range:

- Below the first energy, a threshold reaction is zero and any other reaction
  takes its first value.
- Above the last energy, the cross section is zero, while the group keeps its
  full width. This would make the rate too low, so a run raises an error if
  more than 0.1% of the flux lies above the top of a nuclide's evaluation. See
  [Energy range](libraries.md#energy-range).

Cross sections are taken at the material's temperature. A temperature between
two in the data is served by blending the two that bracket it, linearly in
temperature, and one outside the data's range is refused, naming the range. If
the material has no temperature set, the lowest temperature in the data is
used, 250 K on the published libraries.

Within each group the flux is assumed flat in energy. This assumption has an
error of its own. Compared with a pointwise reference for the FNS foil
reactions on VITAMIN-J-175, flat within-group flux gives a 3.7% mean error on
the hard FNS spectrum (21% with $1/E$ within each group). With a 10% $1/E$ tail
added to the spectrum, the error is 28% for flat and 3.2% for $1/E$. Since
neither shape is better in both cases, YANI uses the flat shape. Finer groups
reduce this error for both shapes.

### Fission yields

Fission yields are tabulated at a few incident energies, typically thermal,
500 keV and 14 MeV, and the yield depends on the energy of the neutron causing
the fission. YANI uses the independent yields (MT=454) and combines the
tabulated sets as

$$
Y(p) = \sum_k c_k\,Y_k(p), \qquad
c_k \propto \int \sigma_f(E)\,\psi(E)\,h_k(E)\,\mathrm{d}E
$$

where $h_k$ is a hat function equal to 1 at the $k$-th tabulated energy and
falling linearly to 0 at the neighbouring tabulated energies, held constant
beyond the first and last. Each set is weighted by the fraction of fissions
near its energy, and the weights are normalised to sum to one. The
interpolation between tabulated energies is always linear, regardless of the
interpolation law given in the evaluation.

The set of reachable fission products is the union over all tabulated sets. A
nuclide with yields and a nonzero fission rate but no weights raises an error.

### Which nuclides get loaded

YANI does not truncate the network at a fixed number of reaction or decay
steps. A fixed depth drops real nuclides when set too low and loads unneeded
data when set too high. Nor does it load cross sections for everything the
composition can reach: from almost any composition that is one large connected
network, 465 nuclides with neutron data on ENDF/B-VIII.1 from Fe56, Al27 or
anything from Be9 up, and hundreds of MB, where a year's irradiation of a 316
steel needs 99 of them.

Instead, YANI finds the products that matter by solving. It loads the
composition's cross sections, solves the schedule, and loads every product with
neutron reactions that, at any state of the schedule, reached
`product_threshold` (default $10^{-12}$) as its atom fraction or as its share
of the activity, the decay heat or the decay photon line power. It then solves
again with those products reacting, and repeats until a solve populates nothing
new above the threshold. Decay shares are tested as well as atom fractions
because a rare product can carry a decay quantity: Na22 in an irradiated
Li4SiO4, at an atom fraction near $10^{-15}$, carries nearly all of the contact
dose a month after shutdown, and its own 28 kb thermal (n,p) burns it out
twelvefold.

A product left out is still produced, decays and is reported; only its own
neutron reactions are dropped. Its atom fraction and its share of each decay
quantity are then off by less than the threshold, and what its reactions would
have made is bounded by the summed peak atom fraction of the products left out.
`get_product_screening_info` reports that sum as `dropped_fraction`, beside
what was loaded, what was left out and how many solves the screening took. A
threshold of 0 loads every product the solve populates at all, which is exact.

## Isomeric branching

When a reaction leaves part of its product in a metastable state, the
metastable state is a separate nuclide with its own half-life and decay chain.
The branching fraction is stored as a function of incident energy and folded
against the spectrum when rates are calculated, so runs with different spectra
on the same target can have different branching fractions.
`get_isomeric_branching()` returns the split used in each step. See
[Weighted, from a solve](usage.md#weighted-from-a-solve).

## Self-shielding

The collapse assumes the given spectrum is the flux inside the material. Often
the spectrum is the flux in the surrounding field instead. Inside a lump of
material, the flux is depressed across resonances where the total cross section
is large, so the true reaction rate is lower than the unshielded rate.

The correction changes the shape of the flux within each group and preserves
the group totals, since those come from a measurement or a transport
calculation. It is applied as a per-group, per-reaction weighting. A nuclide
with no resonances in a group is unchanged.

The flux depression and the escape probability both depend on how easily a
neutron leaves the lump, which depends on its geometry. A `Material` has no
geometry, so the user gives a shape or a mean chord length $4V/S$. YANI never
infers a shape. Without one, no shielding is applied and the result is
bit-identical to a run without self-shielding.

The flux inside the lump is found by solving

$$
\left(\Sigma_t(u) + \Sigma_e\right)\phi(u) =
\int_{u-\Delta u}^{u} \Sigma_s(u')\,\phi(u')\,\frac{e^{-(u-u')}}{1-\alpha}\,\mathrm{d}u' + \Sigma_e
$$

from high to low energy in lethargy, so the in-scattering source uses the flux
already calculated at higher energies. It makes no assumption that resonances
are narrow.

The narrow-resonance approximation, which uses the asymptotic $1/E$ form for
the in-scattering source, was implemented and tested but is not included. It
over-shields nuclides with strong elastic resonances, where in-scattering fills
in the capture dips. On a $1/E$ field at 50 µm it is 41% low on Co59, 63% low on
W186 and 16% low on Mn55, and within 1% on Rh103, Fe56, Eu151 and Nb93. Against
measurement, W186(n,γ) in FNG-tung is 1.29 b. The slowing-down solve gives C/E
0.80 and the narrow-resonance approximation gives 0.46, which is twice as far
from the measurement as no correction. Whether the approximation is accurate
for a given run depends on its spectrum and composition, which a user cannot
easily check in advance, so it is not offered. See
[Self-shielding](usage.md#self-shielding).

## Schedules

A schedule is a list of steps, each with a duration and a rate, and each step is
one matrix exponential. A `Cooldown` is a step with a rate of zero. Its matrix
contains only decay terms, and it uses the same solver as an irradiation step.
The inventory is reported at the start and at the end of every step, so the
output times are the cumulative sums of the step durations. To get a decay
curve, use many short cooldown steps, which are cheap; `cooldown_steps()`
generates them.

On `Material.transmute`, rates are calculated once for each distinct spectrum.
The collapse gives each reaction rate per unit flux from the shape of the
spectrum, and each pulse multiplies these by its own `rate`. Pulses that share
a `source` share one collapse. With self-shielding on, the flux depression
depends on what the material is made of, so every irradiation step after the
first collapses again at the composition it starts from; the first step uses
the starting composition's collapse. Without self-shielding the rates do not
depend on the composition and each spectrum is collapsed once.

### Coupled to transport

In yamc, `Model.simulate_transmutation` uses a transport calculation instead of
the collapse. Each material's reaction rates are tallied directly as
continuous-energy track-length estimates of $\sigma(E)\,\ell$, scored at the
energy of each track, so there is no group structure. In the unresolved
resonance range, $\sigma$ is the probability-table band the transport sampled
for that flight (Levitt, Nucl. Sci. Eng. 49, 450 (1972)), not the smooth
average: inside a band the flux and the cross section are anti-correlated, so
scoring the average would over-count the rate in a material that shields
itself there. Capture and the other disappearance channels take the band's
capture factor, fission channels its fission factor and elastic its elastic
factor. The rate per target atom is

$$
R = \frac{\langle \sigma \ell \rangle}{10^{24}\,V} \times S
$$

where $\langle \sigma \ell \rangle$ is the mean per source particle in b-cm,
$V$ is the material volume in cm³ and $S$ is the pulse's source rate in n/s.
For this reason, `rate` is a source rate in yamc and a total flux in
`Material.transmute`. Fission-yield weights are tallied in the same way, with
the hat functions applied on each track.

There are two methods:

- **independent** runs transport once, with the starting composition, and
  scales the per-source-particle rates by each step's source rate. It is the
  faster method and the only one that supports nuclear-data uncertainty.
- **coupled** runs transport at the start of every irradiation step with the
  current compositions, and updates the materials in the geometry after each
  step. Rates are held at their beginning-of-step values for the whole step,
  with no predictor-corrector, so split any step long enough for burnup to
  change the flux.

Both methods start with a short scouting transport that gives each material's
flux spectrum. The products whose cross sections are loaded and tallied are
chosen on it by the same screening as
[Which nuclides get loaded](#which-nuclides-get-loaded).

## Derived quantities

Each derived quantity is a sum over nuclides of atom density times a per-atom
value from the decay data. With $N_i$ in atoms/b-cm and $V$ in cm³, the number
of atoms is $10^{24} N_i V$.

**Activity** is $A_i = \lambda_i\,10^{24} N_i V$ in Bq, with
$\lambda_i = \ln 2 / T_{1/2}$ from the same chain used in the solve. Stable
nuclides are omitted.

**Decay heat** is $P_i = A_i\,\bar E_i$, where $\bar E_i$ is the mean energy
released per decay, the sum of three MT=457 average energies:

- light particles: beta, conversion and Auger electrons
- electromagnetic: gamma and X-rays
- heavy particles: alpha, protons, neutrons and fragments

Neutrino energy is excluded because it is not deposited. `component=` returns
the beta, gamma or alpha part on its own. If a nuclide that produces heat has
no component split in its data, an error is raised. Some decay records contain
$Q/3$ placeholder energies instead of evaluated ones; see
[Some decay records are placeholders](libraries.md#some-decay-records-are-placeholders).

**Specific values** replace $V$: a per-volume unit sets it to 1 (times
$10^6$ for /m³), and a per-mass unit to $1/\rho$ (times 1000 for /kg), so
neither needs the material volume. Any other factor in the unit, an SI prefix
or a curie, is a constant multiplier on top.

**Decay photon lines** come from the MF=8 discrete spectra of the gamma and
X-ray radiation types. Each line's emission rate per atom is $\lambda_i$ times
its intensity per decay, and the line spectrum is that rate times the number of
atoms, summed over nuclides. Lines at the same energy are merged. The lines are
not binned, so the spectrum can be used directly as a photon source.

**Decay photon continua** are the parts of decay photon spectra given in the
evaluation as a density over energy instead of as lines, for example the
photons from spontaneous fission or the emission of nuclides far from
stability. Each continuum keeps its own energy grid and ENDF interpolation law
and is not resampled or merged. Its emission rate is the exact integral under
that law. These are evaluated decay spectra. Bremsstrahlung is not modelled.

### Clearance index

A clearance, exemption or disposal limit set gives a limit $L_i$ for each
radionuclide, in Bq/g, Ci/m³ or Bq depending on the regulation. The index is

$$
I = \sum_i \frac{a_i}{L_i}
$$

where $a_i$ is the nuclide's activity in the same units. The material meets the
set's limits when $I$ is below the set's threshold, normally 1. The activities
use the half-lives of the transmutation chain, the same ones as `activity()`.
Specific activities are divided by the mass of the whole material, so stable
nuclides must remain in the inventory.

The limit tables and the arithmetic come from the
[radiological-material-clearance-finder](https://github.com/fusion-energy/radiological_material_clearance_finder)
crate. Depending on the set, the limit used for a nuclide can be adjusted:

- a nuclide the table does not list takes the set's catch-all limit, if it has
  one, and is listed in `defaulted`;
- where a parent's limit already includes a daughter in secular equilibrium,
  the daughter's activity is left out (`excluded`), or the part the parent
  accounts for is credited and only the rest is assessed (`credited`);
- activity with no limit and no catch-all is not in the index and is reported
  in `uncovered` and `uncovered_fraction`.

A low index with a large `uncovered_fraction` does not show that the material
meets the limits. See [Clearance and disposal limits](usage.md#clearance-and-disposal-limits).

### Contact dose

The contact dose rate treats the material as a half-space that emits photons
uniformly. At the surface, the uncollided flux at energy $E$ from a uniform
isotropic source of $q$ photons/cm³/s is $q / 2\mu(E)$, where $\mu$ is the
material's linear attenuation coefficient. The size of the material cancels, so
the result needs no volume or distance. Summed over all lines of all nuclides:

$$
\dot D = \frac{B}{2} \sum_i 10^{24} N_i \sum_\ell S_{i\ell}\,
\frac{E_\ell\,(\mu_\text{en}/\rho)_\text{air}(E_\ell)}{\mu(E_\ell)}
$$

for absorbed dose in air, where $S_{i\ell}$ is the emission rate of line $\ell$
per atom and $\mu = \sum_e \rho_e\,(\mu/\rho)_e$ is calculated from each
element's partial density and its NIST XCOM mass attenuation coefficient.
$(\mu_\text{en}/\rho)_\text{air}$ is from NIST SRD 126. For effective dose,
$E\,(\mu_\text{en}/\rho)_\text{air}$ is replaced by the ICRP-116 photon
fluence-to-effective-dose coefficient for the AP geometry. Both tables are
interpolated log-log, and lines outside the energy range of both tables are
not counted. Continua are integrated exactly under their interpolation law,
using Gauss-Legendre moments of the response.

The build-up factor $B$ (default 2) accounts for photons that scatter and still
reach the surface. It is the largest source of error: compared with a photon
transport calculation of the same half-space, it gives a result 16 to 17% high
for Co60 in steel. The method follows UKAEA-CCFE-RE(21)02, Appendix C.7.1.

Dose rates away from the material need a photon transport calculation. The
photon lines can be used as its source, and `dose_coefficients()` provides the
ICRP-116 and ICRP-74 coefficients, including H*(10), to fold with its flux
tally.

### Displacement damage

Unlike the quantities above, displacement damage is a reaction quantity, not a
decay one: it accrues during pulses and cooldowns add nothing. With
`displacement_damage=True`, each nuclide's damage-energy cross section (MT=444)
is folded against the pulse spectrum by the same collapse as the reaction
rates, self-shielding included. Each irradiation step adds, per element $e$,
that rate averaged over the element's own nuclides, times the step's flux and
duration:

$$
\Delta E_{\text{dam},e} = \frac{\sum_{i \in e} \left(N_i^{0} + N_i^{1}\right)\bar\sigma_{444,i}}
{\sum_{i \in e} \left(N_i^{0} + N_i^{1}\right)}\;\phi\,\Delta t
$$

where $N^{0}$ and $N^{1}$ are the atom densities at the start and end of the
step. Averaging the two ends is the trapezoidal rule on the damage deposited,
so a nuclide that burns out over a step contributes less to it and nothing
after, and a product element that appears within the step is still defined.
Displacements per atom follow the NRT model (Norgett, Robinson and Torrens
1975; ASTM E521):

$$
\text{dpa}_e = \frac{0.8\,E_{\text{dam},e}}{2\,E_{d,e}}
$$

Only the linear branch is applied. MT=444 is already integrated over the recoil
spectrum, so the per-recoil threshold steps (no displacement below $E_d$, one
up to $2E_d/0.8$) cannot be applied to it, and in a fast spectrum the recoils
they would change carry a negligible share of the damage energy.

The material total is the atom-fraction-weighted sum over elements,
$\sum_e f_e\,\text{dpa}_e$, so each element's damage energy is converted with
its own $E_d$ before the elements are combined, and the damage-energy total is
weighted the same way. This is the elemental superposition of NRT, which is
defined for a monatomic target: each element's recoils are treated as slowing
down among atoms of their own kind. It reduces to the elemental value for a
pure element, and leaves out the transfer of recoil energy between elements in
a cascade, which the polyatomic displacement functions of Parkin and Coulter
(1981) describe and a single $E_d$ per element cannot.

$E_d$ is the spatially averaged displacement threshold energy: the ASTM E521
recommended value where the standard gives one, otherwise the average from
Table 2.4 of the OECD-NEA report *Primary Radiation Damage in Materials*
(NEA/NSC/DOC(2015)9). An element in neither has no default, since published
values spread too widely to pick one silently, and must be given through
`displacement_energies=`.

Nothing missing is counted as zero silently. A nuclide of the starting
composition with no MT=444, or an element of it with no $E_d$, is refused. A
product with no MT=444 (often a short-lived one the library does not publish)
is left out and listed in `without_damage_energy`. A product element with no
$E_d$ (hydrogen and helium from $(n,p)$ and $(n,\alpha)$, rhenium from
tungsten) still adds to the damage-energy total but gets no dpa, and is listed
in `without_displacement_energy`. Both lists carry the largest atom fraction
each reached on an irradiated step, and `get_displacement_damage_info` also
reports the $E_d$ and its source for every element. The same conversion is
used for the damage-energy tally of a transport run, so the two paths cannot
drift apart. See [Displacement damage](usage.md#displacement-damage).

## Nuclear-data uncertainty

The solve is deterministic and every entry of the matrix is a rate, so
uncertainty is propagated by perturbing the inputs to those rates and repeating
the solve. This is exact to all orders in the matrix exponential: nothing is
linearised, the sandwich rule is not used, and the solver is unchanged.

### Folding the covariance

A reaction rate is linear in the cross section, and MF=33 gives a covariance
that is constant on each interval of the evaluation's energy grid. Together
these give

$$
\operatorname{Cov}(R_i, R_j) = \sum_{k,l} C_{ij}[k,l]\; r_i[k]\; r_j[l]
$$

where $r_i[k]$ is reaction $i$'s rate restricted to covariance interval $k$.
This is exact and does not project the covariance onto the flux group
structure, since the integration is done on the covariance grid. Each block is
folded on its own grid and the contributions are summed.

LB=8 blocks are the exception. An LB=8 block is a short-range variance in
barns squared, not relative to the cross section. ENDF-102 section 33.2.2.2
states that the average over a subinterval $\Delta E_j$ of its interval
$\Delta E_k$ has variance $F_k\,\Delta E_k/\Delta E_j$ and is uncorrelated with
any other subinterval. Its contribution to a rate therefore depends on how the
flux varies within $\Delta E_k$, which is the only place the flux groups enter.
The fold splits $\Delta E_k$ at the group boundaries, where the flux density
$\psi$ is constant, so the rule applies exactly to each piece and the variance
of the rate is $F_k\,\Delta E_k \sum_j \psi_j^2\,\Delta E_j$. A flux that is
flat over the whole interval gives the plain absolute diagonal, which is the
smallest this term can be for a given flux in the interval. The more the flux
is concentrated within an LB=8 interval, the larger the term: ENDF/B-VIII.1
Cr52 (n,p) is 17.7% for a flux flat over the tape's [14, 16] MeV interval and
22.5% for a flux in a single 0.2 MeV group at 14.1 MeV.

A covariance grid does not have to cover the whole flux range. The part of a
rate from outside the grid has no stated uncertainty, so it is included in the
denominator but not the numerator, and the relative uncertainty is smaller
than the covariance grid alone would suggest. `rate_fraction_covered` records
the fraction of each rate that has a stated uncertainty: the rate from energies
where the reaction's diagonal variance, summed over its blocks, is nonzero,
divided by the rate over the whole flux range. An interval with a variance of
zero counts as uncovered. The rate used is the rate in the fold, dilute on a
dilute run and shielded on a self-shielded run. On a transport run, the
tallied rate is different: the fraction there is of the dilute rate over the
tally spectrum, the covered fraction of the tallied rate is not calculated, and
the production-weighted `rate_fraction_covered_total` is `None`.

### Drawing a replica

For the cross sections, a replica draws each nuclide's evaluation once, on its
covariance cells (the energy intervals of its blocks), and every spectrum, step
and channel reads its rate off that one draw, so one evaluation is one
uncertainty wherever it is used. The relative cells are drawn as a multivariate
lognormal with mean one and the evaluation's own covariance $C$:

$$
\Sigma = \ln\left(1 + C\right), \qquad L L^{T} = \Sigma, \qquad
z \sim N(0, I), \qquad m_k = \exp\left((Lz)_k - \tfrac{1}{2}\Sigma_{kk}\right)
$$

with the logarithm taken element by element, so each multiplier $m_k$ has mean
one, $\operatorname{Cov}(m_k, m_l) = C_{kl}$, and no perturbed cross section is
negative. A rate is linear in the cross section, so the perturbed rate is exact
rather than linearised:

$$
R_i' = R_i + \sum_k r_i[k]\,\left(m_k - 1\right)
$$

with $r_i[k]$ the partial rate of the fold above. Absolute blocks (LB=0) are
drawn as normal shifts in barns, and LB=8 short-range variances as independent
normal pieces over the flux groups that cut their intervals; a rate that reads
one of these, or that subtracts reactions, can come out negative, and is then
set to zero and counted in `rates_floored`. A normal draw of the relative
cells, $R_i(1 + \delta_i)$, can give negative rates: on the FNS decay-heat
benchmark it sent 8.3% of all sampled rates below zero (13% on iron), and
setting them to zero biases the mean upward.

A covariance no lognormal can carry, two fully correlated cells with different
sigmas or an anticorrelation with $1 + C_{kl} \le 0$, is drawn at the nearest
one that keeps every sigma: where $\Sigma$ is not positive semi-definite its
correlation matrix is replaced by the nearest correlation matrix, and the
nuclide is listed in `lognormal_not_carried` with how far its correlations
moved. That is a property of the distribution rather than a defect of the
data, so it does not set `has_gaps`.

The evaluation gives a covariance but no distribution, so the lognormal shape
is an assumption. For a channel with sigma near 1 or above, this assumption
determines the tails. Such channels are heavy-tailed and their sampled spread
converges slowly: at 1024 replicas, the maximum the driver adds automatically,
the median sample standard deviation is 0.98 of sigma at $\sigma = 1$, 0.92 at
2, 0.83 at 3 and 0.50 at 9. TENDL-2017 has channels at $10^4$% and above. The
report lists every sampled channel of a populated nuclide with an evaluated
relative sigma of at least 1 in `sigma_at_least_one` and at least 10 in
`sigma_at_least_ten`. Channels with sigma of 1 or more on nuclides outside the
populated bound (described below) are listed in
`sigma_at_least_one_outside_bound`, since a draw on such a channel can be
orders of magnitude above nominal and populate the nuclide.

Each factor comes from an eigendecomposition of a correlation matrix, scaled
back by the standard deviations, so every cell is factorised as accurately as
its own variance allows: cyclic Jacobi for small blocks, and a blocked
tridiagonalisation with divide and conquer for large ones, whose eigenvectors
are signed by a fixed rule. Each runs in a fixed order, so a factor depends
only on the matrix and is the same on any thread count. Cholesky would require
positive definite matrices, and MF=33 matrices are often not even positive
semi-definite as evaluated.

Such a matrix has no square root, so it is repaired before it is sampled.
Clipping its negative eigenvalues would only ever add variance. Instead, the
correlation matrix of the cells with a positive stated variance,
$R = D^{-1/2} C D^{-1/2}$, is replaced by the nearest correlation matrix and
rescaled by the evaluated sigmas, so every such cell is sampled at its
evaluated sigma exactly and only correlations move, by as little as a positive
semi-definite matrix allows. A cell stated with a negative variance, or with a
zero one and a covariance to another cell, has no correlation to keep and is
held at nominal. A channel that folds several cells reads their correlations,
so its sampled sigma can still move, either way.

A matrix counts as repaired when any of the following holds:

- $R$, over the cells with a positive variance, has an eigenvalue below
  $-m \cdot 10^{-12}$, where $m$ is the number of those cells;
- a cell has a negative variance;
- a cell has zero variance and a nonzero covariance with another cell.

$R$ is congruent to that block of $C$, so one is positive semi-definite exactly
when the other is. On $R$ the round-off of the decomposition is about $m$ times
machine epsilon regardless of the cell sigmas, so a positive semi-definite
matrix whose smallest eigenvalue is a few ulps below zero does not count as
repaired. The test does not compare $\lambda_\text{min}$ of $C$ with its
$\lambda_\text{max}$, because TENDL-2017 has channels with relative sigma up to
$5.6 \times 10^8$ (variance near $3 \times 10^{17}$), and one such channel
would set the threshold far above a real repair among the other channels.

The repair is reported in these fields:

- `covariance_repaired` lists the nuclides with a repaired channel that a draw
  can move (one with a positive rate on a spectrum used in the schedule). Any
  entry sets `has_gaps` to true.
- The report covers the nuclides the material can populate. The fold uses every
  nuclide whose data the run loaded, the composition and the products that
  reach `product_threshold`, so a nuclide is reported only when an upper bound
  on its density over the schedule, at nominal rates, reaches the solver floor
  of $10^{-30}$ atoms/b-cm. A nuclide outside the bound cannot change any nominal
  density by that much, but a replica's rates on a wide channel can be orders
  of magnitude above nominal. Repaired nuclides outside the bound with a
  channel a draw can move are listed in `covariance_repaired_outside_bound`,
  and any entry also sets `has_gaps` to true.
- `covariance_repairs` records, for each repaired populated nuclide and
  spectrum, $\lambda_\text{min}$ of $R$ before the repair,
  `largest_correlation_change` and `correlation_frobenius_change` (the largest
  single change and the Frobenius change of $R$), the `cells` repaired and the
  `held_cells`, whether the repair `converged`, and each channel's evaluated
  variance (as stated, even when negative) and evaluated sigma beside the sigma
  it is sampled at.
- `worst_sigma_change` is the largest
  $|\sigma_\text{sampled} / \sigma_\text{evaluated} - 1|$ over the repaired
  channels a draw can move, and is infinite where a repair gave a spread to a
  channel whose stated variance is zero or negative. Every cell with a positive
  stated variance keeps its sigma, so a change here comes from a channel that
  folds several cells, whose correlations moved, or from a held cell.
- `rate_weighted_sigma_change` is the weighted mean of each channel's
  $|\sigma_\text{sampled} / \sigma_\text{evaluated} - 1|$. Each channel is
  weighted by its rate at unit flux, its spectrum's fluence in the schedule and
  its parent's initial density. Weighting the change instead of the sigma stops
  a wide channel with a small rate from dominating. Because of the initial
  density, this is a first-generation measure: produced nuclides have no
  weight, and their repairs appear in the other fields.

The sampled sigma is read from the factorization for every matrix, so matrices
below the repair threshold show their round-off there.

The other sources are sampled using their own stated sigma:

- **Flux.** Drawn once per replica and group, from the pulse's `flux_std_dev`
  or through the factor of its `flux_covariance`, and shared by every rate
  collapsed against that spectrum, the isomeric splits and the `(n,n')` rates
  the branching overlay folds from it included. A rate is linear in the flux,
  so the perturbed rate is the collapse's per-group terms reweighted exactly,
  without repeating the collapse.
- **Half-lives.** Drawn per nuclide from the sigma in the decay data.
- **Decay energies.** Drawn from the sigma on each beta, gamma and alpha
  component that has one, or on the total if no component has a sigma.
- **Decay photons.** Per spectrum, one draw from the sigma on its
  normalisation (FD for lines, FC for a continuum) scales all of its lines or
  continuum points, so the lines of one spectrum move together. Each line then
  takes its own draw from its intensity sigma (dRI, which MT=457 states
  relative to FD and without FD's sigma, so nothing is counted twice) and one
  from its energy sigma (dER). A nuclide's gamma decay energy follows the
  draw, by the drawn change in its photon energy per decay (each line's energy
  times its intensity, plus each continuum's energy integral), in place of a
  `decay_energy` draw of it; the part of it the tabulated spectra do not carry
  is held, and the beta and alpha parts keep their own draws.
- **Fission yields.** Each independent yield (MT=454) the tape states a DY
  for is drawn on the tape's own product, and the draws are summed onto the
  chain's products the way the converter summed the nominal yields: a product
  with no decay data lands on its stand-in. The mapping is not stored, so it
  is rebuilt with the converter's rule and checked against the nominal yields.
  An actinide that borrows another's yields shares that evaluation's draw.
- **Isomeric branching.** Where the branching subsection carries the MF=40
  covariance of a reaction's final-state productions
  (`branching_covariance.arrow`), each state's production is drawn as a
  lognormal multiplier on the covariance's own energy intervals, folded with
  the same spectrum and weight as the nominal split, and the split is remade
  from the drawn productions by the nominal branching rule. Where the list
  names its ground state, the shares are the drawn productions over their
  drawn sum and the reaction total stays the one the cross-section draw gives,
  so a factor common to every state cancels and the uncertainty of the total
  is not counted twice. An isomer listed alone, and every `(n,n')` isomer, is
  an absolute production: it is made at its drawn value and the ground state
  takes the rest of the total, so its own MF=40 sigma replaces the MF=33 sigma
  of its reaction that it carries with the cross sections alone, and adding
  this source can lower its sigma. A draw above the drawn total is scaled back
  to it and counted in `isomer_branchings_clipped`. On a transport run the
  tally also scores each stated state's production on the MF=40 intervals, so
  the same draw re-splits the tallied partials; a tally that scored no
  intervals holds every split.
- **Resonance parameters.** Where a nuclide's folder carries its resonance
  parameters and their covariance (MF=2 and MF=32, `resonance_parameters.arrow`,
  in the published libraries), each replica draws the parameters themselves,
  rebuilds the resonance range's cross sections from them and Doppler broadens
  the change to the temperature in use, so that range's uncertainty is exact
  in the parameters rather than first order. A nuclide whose parameters cannot
  be sampled falls back to the first-order MF=32 rows in `covariance.arrow`;
  `resonance_parameters` in the report says which was used, and why.
- **Tallied rates.** On a transport run, drawn jointly from their per-history
  covariance.

The decay data gives an expected value and a standard deviation for each value,
with no distribution or correlation (ENDF-102 section 29.1), and each nuclide is
sampled independently. A half-life, decay energy, photon value or fission
yield is its nominal value times a
lognormal factor with mean one and variance equal to the squared relative
sigma, so every draw is positive and the ensemble has the stated mean and sigma
exactly, with no floor. The independence between a nuclide's decay-energy
components and between different nuclides' half-lives is an assumption, since
the data gives no correlation. A sigma that cannot be sampled (one on a value
of zero, such as a decay energy or a yield, or one that is not finite) is held
at nominal and listed in `half_life_uncertainty_not_carried`,
`decay_energy_uncertainty_not_carried`,
`decay_photon_line_uncertainty_not_carried` or
`fission_yield_uncertainty_not_carried`, all counted as gaps.

Decay branching ratios are drawn per parent, for every parent with two or more
modes and a stated sigma. MT=457 gives each mode a ratio and a sigma with no
covariance between modes, and the ratios sum to a fixed total $T$, so the
deviations of a draw must sum to zero. Such a zero-sum covariance exists only
when no mode's sigma exceeds the sum of the others'. Of those that realise
every stated sigma, the draw takes the one of largest entropy, which assumes no
correlation the sum rule does not force; for two or three modes it is the only
one. When the largest sigma exceeds the sum of the others', that mode is drawn
at their sum, the most the sum rule allows.

Every moving mode but one is a logit-normal on $[0, T]$ whose mean and
standard deviation are fitted to the evaluated ratio and its sigma, so a small
branch with a large relative sigma is drawn skewed and never below zero. The
modes are coupled through a Gaussian copula so that they carry the target
covariance, and the last mode is their complement, which keeps the total. A
single mode stating no sigma is that complement when it is the largest mode or
when only one other mode states a sigma, as an evaluator writing a dominant
mode as one minus the others intends. A mode is held at its ratio only when
its stated sigma is one no ratio with its mean can have,
$\sigma^2 \ge b (T - b)$ for a ratio $b$, or when it states none and is not
the complement; the other modes share what is left. A draw whose complement would fall below zero is pulled back
towards nominal and counted in `decay_branchings_floored`.

A parent held whole is listed in `no_decay_branching_uncertainty` (no mode
states a sigma, or one does beside two or more that do not) or
`decay_branchings_too_wide` (every mode that could move states a sigma no ratio
can have). A parent drawn short of its stated sigmas is listed in
`decay_branchings_partly_held` (some modes held) or
`decay_branchings_sigma_adjusted` (a mode drawn more than 0.1% from its stated
sigma, because the largest exceeds the sum of the others' or the complement
needs a correlation non-negative ratios cannot carry). Each of the four counts
as a gap. Only the inventory changes: a sampled parent's lines and decay energy
per decay use its nominal branching.

The flux draw is a lognormal too, by the cross sections' transform: each bin's
flux is scaled by $\exp(y_g - \Sigma_{gg}/2)$ with $y \sim N(0, \Sigma)$ and
$\Sigma = \ln(1 + C)$ element by element, $C$ the relative covariance of the
pulse's `flux_std_dev` or `flux_covariance`. No sampled flux is negative, even
for a bin whose relative error exceeds 100%, and nothing is floored; a
`flux_covariance` no lognormal carries is drawn at the nearest one and listed
in `flux_lognormal_not_carried`. The tallied-rate draw stays normal, because it
carries the per-history correlations between rates: a tallied rate drawn below
zero is set to zero and counted in `statistical_floored`, which biases the mean
of that source upward, so a nonzero count means the normal distribution is a
poor description of the tally.

Seeds depend only on their arguments. A cross-section, half-life,
decay-branching, decay-energy, decay-photon or isomeric-branching draw depends
only on `(seed, replica, nuclide)`, the nuclide being the parent for an
isomeric split: not on the number of replicas, the order they ran in, or the
other nuclides in the material. Resonance parameters add the range,
`(seed, replica, nuclide, range)`, and fission yields are keyed on the
evaluation. Nuclides linked by an applied cross-material block are drawn
jointly from `(seed, replica, group)`, the group being every linked nuclide of
the run, so their draw depends on which partners the run holds, though their
distribution does not. The flux draw depends on `(seed, replica, spectrum)`,
where spectrum is the spectrum's index among the schedule's spectra. The
statistical draw depends on `(seed, replica, material)`, the material being its
slot in the yamc geometry's material store rather than its id: the model fixes
the slot, while an id the user did not set depends on what else the process
built. It is made jointly over that material's tallied rates, so adding or
removing a tallied rate changes the other draws of that material. A given seed
always reproduces a run.

### Derived quantities under uncertainty

Activity, decay heat, contact dose and every photon line are calculated once
per replica, with that replica's half-lives, and the spread is taken over the
results. They are not combined from per-nuclide sigmas.

Adding in quadrature would be wrong. Every Mn56 atom in an irradiated iron foil
came from an Fe56 atom, so the two densities are anticorrelated and their
spreads partly cancel. Quadrature would count variance that is not there.
Calculating once from the mean inventory would also be wrong: activity and
decay heat are linear in $N$, so this gives the correct mean but a spread of
zero. Contact dose is not linear in the densities at all, since a replica that
produces more of an emitter also absorbs more of its photons.

The replica's half-lives must be used here as well as in the solve. A
saturated activity is $\lambda N = R$, which depends very little on its own
half-life. Using the nominal $\lambda$ would give it the full spread of
$N = R / \lambda$. A line's intensity per atom is its emission per decay times
$\lambda$, so it is also rescaled with the replica's $\lambda$, which keeps the
emission per decay at the value the decay photon draw then starts from. Decay
energies and decay photons are only used here: they are sampled where decay
heat, the photon spectrum and contact dose are calculated and affect nothing
else. The photon draw moves line energies as well, so the photon
spectrum matches a line across replicas on its nominal energy.

### What is not propagated

Nine sources are propagated: activation cross sections (with resonance
parameters where the library carries them), the flux spectrum (on
`Material.transmute`), half-lives, decay branching ratios as described above,
decay energies, decay photon lines and continua, independent fission yields,
isomeric branching from MF=40, and the statistical error of tallies (on a
transport run). Everything listed below is held at its evaluated or nominal
value in every replica. Most of these have their own uncertainty or model
error, which is then missing from the sigma, so the sigma is underestimated,
and for some results this is the largest term. Two can go either way, because
holding them also removes a feedback that reduces the spread: self-shielding,
and, on `Material.transmute`, the response of the flux to the cross sections
of a material that shapes its own flux. Their entries below say which way each
goes. The covariance blocks that are skipped because they correlate an
evaluation with one the run does not hold can also go either way: dropping a
covariance term lowers the variance of a sum when the term is positive and
raises it when the term is negative.

- **Decay branching ratios the draw holds.** A mode whose stated sigma no
  ratio with its mean can have, a mode stating no sigma that is not the
  complement of the rest, and a parent with no sigma at all are held at their
  evaluated ratios. The draw cannot reproduce a stated sigma the sum rule or
  non-negative ratios cannot carry, and reports where it fell short.
- **Fission yield correlations and energy weights.** No evaluation publishes a
  correlation between yields, so each is drawn independently and a draw's
  yields are not renormalised to the nominal total. A parent whose tape yields
  do not sum back to the chain's (`fission_yields_mapping_mismatch`) is held.
  On `Material.transmute` the weights that combine a nuclide's yield sets by
  incident energy come from the nominal spectrum in every replica, so a flux
  draw does not change them; on a transport run with `cross_sections` on, each
  replica tallies its own.
- **Isomeric branching without MF=40.** The split of a reaction product
  between the ground state and an isomer comes from MF=9 or MF=10. Where an
  evaluation gives its uncertainty, in MF=40, the `isomer_branching` source
  draws it, on `Material.transmute` and on a transport run alike. A split whose
  evaluation states no MF=40 on any of its states is held and listed in
  `no_isomer_branching_uncertainty`, and MF=40 blocks that cannot be used
  (another evaluation's partner, a partner that is not an MF=10 production, NC,
  absolute and short-range layouts, a state the branching rule folds no curve
  for) are counted in `isomer_branching_blocks_not_used`. With the branching
  overlay configured (`transmutation_branch_ratios`), a flux draw on
  `Material.transmute` moves each split with the groups its states are
  produced in, and re-makes the `(n,n')` rates the overlay adds to reach a
  metastable state, including In115 to In115m. Those channels have no
  group-averaged cross section and no cross-section covariance of their own.
- **Resonance-parameter covariance (MF=32) neither sampled nor written.**
  ENDF-6 gives the cross-section covariance in the resonance range as an MF=32
  part plus the MF=33 part, and many evaluations put the whole resonance-range
  uncertainty in MF=32 and leave MF=33 at zero there. Where the library
  carries the resonance parameters, the draw samples them exactly (above);
  otherwise the converter's first-order MF=32 rows in `covariance.arrow` are
  folded like any MF=33 block. What is left held is a range whose formalism
  `endf` does not reconstruct, and data converted before either was written,
  on which the fold sees MF=33 only and a capture rate driven by resonance
  flux can have a sigma near zero. For capture in a $1/E$ field, NJOY ERRORR gives
  ENDF/B-VIII.1 W186 1.53% with MF=32 and 0.00% from MF=33 alone. On such data
  the coverage report shows this: W186's capture block states zero variance
  from $10^{-5}$ eV to 10 keV, where nearly all the capture rate is, so on the
  FNS spectrum its `rate_fraction_covered` is about 0.07, against 1 on the
  published library, which samples the resonance parameters. W186 is still
  listed under `perturbed`, since the block's other intervals are used. The
  same check gives 6.32% against 0.01% on the JEFF-4.0 Ag109 tape and 4.71%
  against 0.00% on the TENDL-2025 Co59 tape.
- **MF=33 blocks that are not used.** These are:
    - blocks whose `MAT1` names an evaluation the run does not hold, or that
      correlate a reaction with no relative covariance of its own; a block
      naming an evaluation the run does hold is applied, and the two nuclides
      are drawn jointly;
    - blocks whose partner is not a cross section (`XMF1` other than 0 or 3);
    - NC blocks that cannot be derived (LTY 1 to 4, and the LTY=0 blocks
      counted in `skipped_nc`);
    - lumped reactions (MT=851 to 870) whose block is stated in barns rather
      than relative, which says nothing of how the sum splits;
    - blocks on partial levels (MT=600 to 849 and 875 to 891) where the chain
      uses the total and no LTY=0 NC block or lump names them.

    A block naming the evaluation's own MAT is folded, as are an LTY=0 NC
    block's derivation from the reactions it names, a lump with a single
    component, and a relative lump of several: each component with no
    covariance of its own takes the lump's relative perturbation, so the
    components move together and their sum moves as the lump states.
    `skipped_cross_material`, `skipped_other_file`, `skipped_nc` and
    `lumped_covariance_not_assignable` count the first four kinds, on reactions
    the fold reaches, and all set `has_gaps`; no library yani reads states a
    partner that is not a cross section, an NC block with LTY 1 to 4, or a lump
    in barns. Partial levels are not counted, and the rates they would cover
    count as uncovered in `rate_fraction_covered_total`. A nuclide whose blocks
    are all of these kinds, or all on reactions the chain does not use, is
    listed under `no_covariance_data`, which sets `has_gaps`. Tungsten's
    `(n,2n)` is lumped: ENDF/B-VIII.1, JEFF-4.0 and FENDL-3.2d give it only
    together with `(n,2np)`, as MT 852, and it is sampled through that lump.
    JEFF-4.0 Be9 gives its `(n,2n)` only on the levels MT=875 to 890, and
    ENDF/B-VIII.1 Ca40 gives its `(n,p)` and `(n,a)` only on MT=600 and 800, so
    Ca40 is listed under `perturbed` through its capture alone.
- **Self-shielding.** With a shape or a chord, every replica uses the flux
  depression calculated from the evaluated cross sections, at the composition
  each step starts from in the nominal run. Holding it fixed
  drops two effects with opposite signs. A larger capture cross section would
  deepen its own dip, which narrows the spread, and this is not included. The
  elastic and total covariance does not reach the correction either, which
  would widen the spread where MF=33 gives a resonance-range uncertainty for
  elastic scattering. Which effect is larger depends on the evaluation. For a
  0.1 mm Au197 foil under $1/E$ on VITAMIN-J-175, ENDF/B-VIII.1 gives no
  resonance-range elastic uncertainty, and the held sigma is 1.8 times too large
  (1.70% held against 0.96% with the feedback). TENDL-2017 does give one, and
  there the held sigma is too small: 5.34% held against 6.07% with the feedback
  and the elastic term together (2.23% with the feedback alone). These figures
  fold the shielded partials, as the fold does.
- **Response of the flux to the cross sections, on `Material.transmute`.**
  There the flux is the one the pulse is given, so a replica's cross sections
  move the rates and leave the flux as it is. For a trace activation product
  this is exact to first order in its own cross section. For a material that
  shapes its own flux it is not: a larger Li6(n,t) cross section in a breeder
  depresses its own flux, the same narrowing feedback as self-shielding, so
  holding the flux overstates that rate's sigma. Nor do the cross sections of
  the materials the neutrons passed through (a shield, a multiplier, a
  breeder) change the flux, which understates the sigma of a product behind
  many mean free paths of steel. The spectrum's own uncertainty is only what
  the pulse is given. yamc's `Model.simulate_transmutation` carries both: with
  `cross_sections` on, its one transport gives every history a likelihood-ratio
  weight per replica, so each replica's rates, isomeric partials, fission-yield
  weights and flux are what a transport with that replica's data would tally,
  every material's cross sections included (`flux_response` is
  `"replica_weights"` in the report, and `transport_replicas` the count). The
  count is fixed before the transport, `samples` or 256 at the default
  `convergence`, and the Monte Carlo noise each replica's shift carries is
  measured from the even half of the histories and removed from every sigma
  (`noise_removed`, `noisy`). Only the independent method supports
  `data_uncertainty`.
- **Decay photon spectrum covariance and continuum shape.** MT=457 can state a
  covariance between a spectrum's lines (LCOV), and no library ships one. A
  continuum's shape is held, and only its normalisation is drawn. The decay
  data also leave unstated whether a spectrum's line intensities share a part
  (ENDF/B-VIII.1 folds each spectrum's normalisation sigma into every line's),
  whether a normalisation and its lines are correlated, and whether a nuclide's
  gamma and x-ray spectra are. None of these is assumed: every photon output,
  and the gamma part of decay heat, is evaluated with all of them taken as zero
  (`Estimate.std_dev`) and as one (`Estimate.std_dev_correlated`), which bounds
  every non-negative correlation, and `decay_photon_spectra_folded` names the
  spectra written the ENDF/B way, where most of that range comes from. The
  decay data give a sigma on 99.6% of ENDF/B-VIII.1 gamma lines, and for one
  emitter as a trace in iron the contact dose from its line data alone has a
  relative sigma of 0.014% for Co60, 1.09 to 1.75% for Mn56 and 0.82 to 1.70%
  for W187 on ENDF/B-VIII.1, from the independent end to the fully correlated
  one.
- **Dose constants and build-up.** Contact dose uses the same photon
  attenuation coefficients, the same response (air energy absorption, or the
  ICRP-116 coefficients for effective dose) and the same constant build-up
  factor in every replica. None of these tables gives a per-value uncertainty.
  The build-up factor is a modelling choice with a larger error: the default of
  2 is 16 to 17% high for Co60 in steel compared with a photon transport
  calculation of the same half-space.
- **Material composition and natural abundances.** Element and impurity
  fractions, density and natural isotopic abundances are the same in every
  replica. For activation driven by a trace impurity this is often the largest
  missing term: in a 316L-like steel with 0.1 wt% cobalt, the contact dose at
  10 years is 99.6% Co60 and changes by 0.69% per 1% change in the cobalt
  fraction. A material contains only the elements it is given, and the bundled
  PNNL compendium's `Steel, Stainless 316L` has no cobalt, niobium, tantalum or
  silver.
- **Displacement damage beyond the starting composition's reaction deposits.**
  MT=444 has no covariance of its own, so a replica moves it through the drawn
  cross sections of the reactions that deposit it. The share no reaction's
  deposit reproduces (fission's above all), the MT=444 of transmutation
  products, short-range (LB=8) covariance and the displacement threshold
  energies $E_d$ are held at nominal. The NRT model itself is a model choice
  with no uncertainty attached. `get_data_uncertainty_info(...)["damage_energy"]`
  reports the share explained per starting nuclide.

`not_perturbed` in `get_data_uncertainty_info` lists these on every run:
isomeric branching (MF=9/MF=10) of a reaction with no stated MF=40 covariance,
the MT=5 product multiplicities (MF=6 MT=5), cross-material blocks naming an
evaluation the run does not hold, non-cross-section and underivable NC MF=33
blocks, lumped MF=33 blocks stated in barns, resonance-parameter covariance
(MF=32) neither sampled nor in `covariance.arrow`, the decay photon spectrum
covariance and continuum shape, the photon attenuation, energy-absorption and
fluence-to-dose coefficients, the contact-dose build-up factor, and the
material composition, density, natural abundances and atomic masses. It also
adds an entry for each source a run turns off (half-life, decay energy, decay
photon lines, fission yield, decay branching, isomeric branching, activation
cross section), the per-branch decay emission of a parent whose branching was
sampled, the flux spectrum on `Material.transmute` when that source is off or
some or all spectra have no sigma, the tallied-rate statistics when a transport
run does not sample them, and the self-shielding correction when shielding is
on. The MT=5 multiplicities are shares of the MT=5 rate: the rate is drawn
with the cross sections, and its split among the residuals and light
particles is held.

Some held inputs on this page have no entry of their own: MF=33 blocks on
partial levels (MT=600-849, 875-891), for which the chain has no rate, so they
are neither listed nor counted; the cross-section covariance of the overlay's
`(n,n')` channels; the fission-yield weights held at the nominal spectrum
under a flux draw; and, on `Material.transmute`, the response of the flux to
the cross sections of the materials the neutrons passed through.

`has_gaps` only considers the sources the run perturbs. It is true when any of
these occurs:

- a nuclide with no usable MF=33 block;
- a skipped cross-material, non-cross-section or NC block;
- a mirrored pair whose copies disagree;
- an unassignable lump;
- a block with an unsupported or malformed layout;
- a channel whose partial rates add up to more or less than its rate
  (`partials_above_rate`, `partials_below_rate`);
- a derived channel whose opposing terms have no stated covariance
  (`derived_opposing_uncorrelated`);
- a covariance repaired beyond round-off on a channel a draw can move, inside
  or outside the populated bound;
- a cross-material covariance drawn at the closest admissible one
  (`cross_material_adjusted`);
- a perturbed evaluation its own library documents a problem with
  (`covariance_warnings`);
- a spectrum with no flux sigma, or a rate under a spectrum with one that the
  flux draw cannot move (`flux_rates_without_terms`);
- a reachable unstable nuclide with no half-life sigma, or with a decay energy
  or decay photon data but no sigma on it;
- a half-life, decay-energy or decay photon sigma that cannot be sampled;
- a fissioning parent with no yield sigma, a yield sigma that cannot be
  sampled, or yields that do not map back (`fission_yields_mapping_mismatch`);
- a multi-mode parent whose branching was held, wholly or in part, or drawn
  short of its stated sigmas;
- a reachable isomeric split with no MF=40 to draw, an MF=40 block not used,
  or an MF=40 matrix repaired to the nearest correlation matrix.

Of the inputs in the not-propagated list, the skipped and unassignable MF=33
blocks set `has_gaps`, and the partial-level blocks set it only through
`no_covariance_data`, when they are all a nuclide has. A reported sigma is the
spread from the propagated sources, with everything in this list held
fixed. See [Nuclear-data uncertainty](usage.md#nuclear-data-uncertainty).

## Reproducibility

These three properties are checked by tests:

- **Thread count does not change the result.** A 1-thread and a 7-thread run
  agree to the last bit, including the per-replica ensemble, the sigmas and the
  truncation counters.
- **A seed reproduces a run.** See the seeding rules above.
- **Optimisations do not change the result.** Every performance change to the
  collapse is checked to be bit-identical to the previous implementation, by
  hex-float comparison of inventories and against a reference of the per-group
  terms, one-group rates and fission-yield weights.

Without `data_uncertainty`, no covariance is read from disk, no matrix is
folded or factorised, and no replica is solved. The means are bit-identical
to a build without uncertainty support.

[pusa2010]: https://doi.org/10.13182/NSE09-14
[pusa2015]: https://doi.org/10.13182/NSE15-26
