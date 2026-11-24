# Why YANI

Four things, in order of how often they decide the question: it has to be
easy to run, the numbers have to be right, it has to be fast enough to use in
a loop, and it has to bend to the problem you actually have rather than the
one a tool author imagined.

## Easy

**`pip install yani`, on Linux, macOS or Windows.** The wheel carries the
compiled solver. There is no compiler to invoke, no transport stack to build
first, and no system package to hunt down.

**Nuclear data downloads on demand.** Point a setting at a library keyword and
the sections your calculation actually needs are fetched and cached on first
use, not a whole evaluation up front. See
[Where the data comes from](usage.md#where-the-data-comes-from).

**The license stack is fully permissive.** YANI is dual-licensed under MIT or
Apache-2.0, at your option, which is the usual arrangement in the Rust ecosystem
and means a review board that wants Apache-2.0's explicit patent grant can have
it without asking anyone. Nothing in the dependency chain forces a copyleft
license onto whatever you build with it, so using it in a closed pipeline or a
commercial tool is not a licensing question, just an engineering one.

**It runs in a browser too, with nothing installed at all.** The solver is Rust
and compiles to `wasm32` cleanly, so
[the browser build](https://fusion-neutronics.github.io/yani-online/) is that
same solver against the same continuous-energy data, not a reduced version of
it. There is no server, no account and no upload: the calculation runs on your
own machine, and the page is a single file you can save and open offline. It is
the fastest way to try YANI before installing anything, and the easiest way to
hand a colleague a worked example, since a link carries the whole setup.

**A citable material library ships in the wheel.** The PNNL Compendium
(PNNL-15870 Rev. 2), 410 named materials with their compositions and
densities, is bundled rather than a separate download, so a standard
shielding or structural material is `yani.materials.pnnl.material(key="Steel,
Stainless 316", volume=...)` away instead of a retyping exercise from someone
else's PDF. See [Materials](usage.md#materials).

## Accurate

**Continuous-energy cross sections, not a multigroup collapse.** A multigroup
library has already averaged the cross section over each group using someone
else's assumed spectrum shape, and resonance structure inside a group is gone
before you ever see it. YANI folds your actual spectrum against the pointwise
cross section directly, at the energy resolution the evaluation was measured
at, so the only averaging is the one your own spectrum implies. This matters
most exactly where it is usually skipped: narrow resonances that a coarse
group structure smears across a bin.

**An uncertainty on the answer, not just the answer.** Pass
`data_uncertainty=` and every nuclide density comes back with a standard
deviation beside it. The activation cross sections are resampled from the
evaluation's own ENDF MF=33 covariance, folded against your spectrum, and
their resonance parameters from MF=32 where the library carries them; the
half-lives, decay energies and decay branching ratios from the decay data's
own sigmas; the isomeric splits from MF=40; and the
schedule is re-solved for each sample. That is exact to all orders in the
matrix exponential: nothing is linearized, and the sandwich rule is not used,
so correlations between nuclides survive. When a replica moves a channel, the
channel's product moves against its target and together with the nuclides the
product decays to, and a quantity summed over them keeps both: the spreads of
the first pair partly cancel and those of the second add. If your spectrum
arrives from a Monte Carlo run with an error on it, hand it to the pulse as
`flux_std_dev`, or as a full `flux_covariance`, and that propagates too, as its
own source, so you can see which source dominates.

**Every quantity you would put in a report carries a band from the same
ensemble.** Activity, decay heat, contact dose and every decay photon line come
back as an `Estimate`, a nominal value with the ensemble's spread beside it.
Each is evaluated once per replica, with that replica's own half-lives, and
summed inside that replica: adding the per-nuclide sigmas in quadrature
double-counts a variance that partly cancels, and contact dose is not even
linear in the densities, since a replica that makes more of an emitter also
absorbs more of it. Each line's emission per decay and energy are drawn from
the decay data's own sigmas, so a photon line's band is its emitters' activity
and its own intensity together. The attenuation, response and build-up behind
contact dose are the same in every replica. A calculated band beside a measured one says whether the
disagreement is larger than the propagated data allows.

**A sigma of zero says which kind of zero it is.** The hard part of an
uncertainty is not producing one, it is knowing what it left out. A nuclide
whose evaluation publishes no covariance and a nuclide whose covariance is
genuinely small would otherwise both report `0.0`, and only one of those is
reassuring. `get_data_uncertainty_info` keeps them apart: which nuclides were
perturbed, which have no usable published covariance, what share of each
reaction rate carries a nonzero stated variance, which evaluated matrices were
not positive semi-definite and had to be repaired (a repair that keeps every
evaluated sigma and moves only correlations, with each channel's evaluated
sigma beside the one sampled), and
which sources were held at nominal
([What is not propagated](method.md#what-is-not-propagated) explains each one).
On ENDF/B-VIII.1 the second point is not academic (48% of evaluations carry
cross-section covariance, against 100% of TENDL-2025), so the same run on two
libraries can give two very different sigmas, and the report is what tells you
why. What it separates is an evaluation with no MF=33 from one whose MF=33 is
small, and a capture whose MF=33 is zero over the resonance range, because the
evaluation keeps that uncertainty in resonance-parameter covariance (MF=32),
reads as uncovered there when the data carries no resonance parameters
(ENDF/B-VIII.1 W186 about 0.07 on the FNS spectrum from MF=33 alone, and 1 on
the published library, which samples them). `rate_fraction_covered_total`
weights coverage by reaction rate and by parent density instead of counting
evaluations.

**Resonance self-shielding from a slowing-down solve, with a warning when you
skip it.** Give a lump its shape, `yani.shapes.FoilLump(thickness=0.1)`, or its
mean chord, and the flux inside it is depressed where the total cross section is
large. The solve assumes nothing about resonances being narrow: the cheaper
narrow-resonance approximation is not offered, because it over-shields strong
elastic scatterers badly enough to be worse than applying no correction at all.
Leave the correction out, which is the default since a material carries no
geometry, and the run still reports what it skipped:
`would_shield` in `get_self_shielding_info` names the resonance absorbers it left
uncorrected and how strongly their own resonances could bite. That is a screen
rather than a bound, since it carries no geometry, so it says which answers to
distrust rather than by how much. See
[Self-shielding](usage.md#self-shielding).

**Gamma lines, not a binned photon response.** `decay_photon_spectrum()`
returns discrete energy/intensity pairs, not a spectrum pre-collapsed onto a
group structure. A binned response answers "how much energy in this
window"; lines answer "which nuclide is that", which is what you need for
identifying an isotope from an emitted spectrum or feeding a photon transport
run without inheriting someone else's binning choice.

**Displacement damage from the same solve, with an uncertainty.** Pass
`displacement_damage=True` and the run also returns damage energy and NRT dpa,
per element and for the material, cumulative over the schedule. The MT=444
damage-energy cross section is folded against your spectrum by the same
continuous-energy collapse as the reaction rates, self-shielding included, and
at each step's composition, so transmutation products count as they build up.
With `data_uncertainty=` the dpa gets a band from the same ensemble as
everything else, and the report says how much of each nuclide's damage rate the
sampled reactions account for. See
[Displacement damage](usage.md#displacement-damage).

**Clearance and disposal limits from the same inventory.**
`clearance_index()` checks a material against the UK (EPR 2016, IRR 2017),
German (StrlSchV), US (NRC 10 CFR 61, Fetter), EU Basic Safety Standards and
IAEA GSR Part 3 limit sets, with the transmutation chain's own half-lives, and
reports the activity no limit in the set covers so a low index cannot hide a
gap. See [Clearance and disposal limits](usage.md#clearance-and-disposal-limits).

**Production routes come back with a share on each one.**
`get_production_routes()` lists the routes into a product and weights each by
what the material's own reactions drove over the step. A chain on its own can do
neither: it offers `Os190(n,a)` for W187 as readily as `W186(n,gamma)`, and
nothing in a tungsten foil is osmium, and the branching on one of its edges is a
share of that channel alone, so it cannot rank two routes. The result has the
shape of a published pathway table. See
[Production routes](usage.md#production-routes).

**Isomeric branching is tracked, not folded into the ground state, and it is
flux-weighted rather than a fixed number.** A reaction that leaves a fraction
of its product in a metastable state is modeled as a branch to that state,
with its own half-life and its own decay chain, rather than lumped into the
ground-state inventory. The branching fraction itself is stored as a curve
against incident energy, not a single evaluation-average ratio, and is folded
against your actual spectrum at rate-compute time, so two calculations with
different spectra but the same target nuclide can legitimately get different
branching fractions, because they should. `Ag110_m1` dominating the activity
of a silver foil for years after `Ag110` itself has decayed away, as in the
[getting started](getting_started.md) example, is the kind of result that
disappears if branching is not tracked. `get_isomeric_branching()` reports the
split each step was solved with, which separates a disagreement caused by a
cross section from one caused by a branching ratio. See
[Weighted, from a solve](usage.md#weighted-from-a-solve).

**A reaction rate can be read back per energy group, not just as one number.**
`get_reaction_rate_spectrum()` resolves one channel's rate onto the groups of the
spectrum that drove it, and the entries sum to the collapsed rate because both
come from the same walk of the same cross sections. A capture cross section spans
decades, so an effective one-group value of tens of millibarns against a
14 MeV-dominated spectrum is either fast capture or resonance capture, and only
the breakdown says which. That is the difference between a disagreement that
belongs to the resonance processing and one that belongs to the fast cross
section. Set against where a covariance states a nonzero variance, it also
shows, to group resolution, where in energy the share that
`rate_fraction_covered` reports as one number comes from, and on a shielded run
it says which groups the depression moved. See
[Where in energy a rate came from](usage.md#where-in-energy-a-rate-came-from).

**Verification and validation runs against every open benchmark we have
found**, [CoNDERC][conderc] among them. The rest is a todo, honestly labeled:
if you know of an open, reproducible benchmark this list is missing, tell us
and we will add a reproducible V&V script for it.

[conderc]: https://nds.iaea.org/conderc/

## Fast

**No transport solve stands between your spectrum and an answer.** Point a
`Material` at whatever spectrum you already have and call `transmute()`; there
is no geometry to build and no particle transport run to get through first.
YANI folds continuous-energy cross sections against your spectrum directly,
as one function call.

**Parsed nuclear data stays loaded in RAM, not reloaded from disk.** Cross
section libraries and transmutation chains are parsed once per source, a
library keyword or a file path, into a process-wide cache keyed on exactly
that, and reused from then on. Looping over many materials, spectra or
schedule variants in one script pays the parse cost once, not once per
`transmute()` call, and a later call that needs a wider slice of the same
source (say, more reactions) tops the cache up rather than re-parsing it from
scratch. The cross sections a `transmute()` needs are kept on the material
itself as well, so a second call on the same material does no file reading at
all. A sweep over thousands of distinct compositions hands them back with
`release_nuclear_data()`. Downloads are cached separately, to disk in the
platform cache directory (`yani.cache_dir()`), so a second run of the script
does not refetch.

**The chain that gets built is exactly the one your material can reach, no
manual depth setting required.** Rather than truncating the transmutation
network at a fixed number of reaction or decay steps from the starting
nuclides, a value that silently drops real nuclides if set too low and loads
data no calculation needs if set too high, YANI finds the products that
matter by solving: it loads the composition's cross sections, solves the
schedule, loads every product that reached `product_threshold` (1e-12 by
default) as an atom fraction or as a share of the activity, decay heat or
decay photon power, and repeats until nothing new crosses it. A year's
irradiation of a 316 steel against ENDF/B-VIII.1 loads 99 nuclides that way,
of the 465 its network can reach. A product left out is still made, decays and is reported, and only its own
reactions are dropped, by an amount `get_product_screening_info` bounds;
`product_threshold=0` loads every product the solve populates, which is exact.

**Nuclear data is a binary columnar format, not text to parse.** Cross
sections and transmutation chains are stored as Arrow, so loading them is
reading typed columns straight into memory rather than parsing fixed-width
text records the way an ENDF tape has to be. Converting from ENDF or ACE is a
one-time step; every calculation after that reads the fast format.

**Many materials in one call.** `yani.transmute(materials=..., schedules=...)`
solves a whole mesh, a component split into regions or a sweep over
compositions together. The chain is read once, each nuclide's cross sections
are decoded once for every material that needs them, a collapse runs once per
distinct spectrum, composition, temperature and shielding, and the solves run
in parallel. Each material gets exactly what its own `transmute()` would give
it. See [Several materials at once](usage.md#several-materials-at-once).

## Flexible

**The same solver targets native and the browser.** The Rust core builds for
`wasm32` as a first-class target, not a native tool with a JS wrapper bolted
on afterward, so the identical solver behind `pip install yani` can also run
client-side with no backend to stand up. Where a calculation runs is a
deployment choice, not a rewrite.

**Four independently sourced transmutation subsections, mixed at will.**
`transmutation_reactions`, `transmutation_decay_data`,
`transmutation_fission_yields` and `transmutation_branch_ratios` are each
their own setting, so a network can mix libraries by subsection instead of
committing to one library for everything. A TENDL reactions network borrowing
decay data and fission yields from ENDF/B-VIII.1, the usual arrangement since
TENDL publishes neither, is a first-class configuration, not a data file
hack. See [Nuclear data settings](usage.md#nuclear-data-settings).

**A schedule can mix spectra, not just magnitudes.** Every `Pulse` carries its
own `source`, so a campaign whose neutron spectrum changes shape between
phases, not just intensity, is expressed directly instead of approximated
with one averaged spectrum. A fusion reactor's operating life is the obvious
case: DD-phase pulses and DT-phase pulses each get their own spectrum in the
same `PulseSchedule`, back to back, and the network sees the real spectrum
shape for each phase. See [Schedules](usage.md#schedules).

**Any spectrum group structure works**, not just a preset list. A `Histogram`
takes whatever ascending list of energy boundaries your spectrum happens to
use, of whatever length, and named structures like `CCFE-709` are a
convenience for a handful of common cases, never a requirement. See
[Named group structures](usage.md#named-group-structures).

**Local ENDF files, local Arrow files and premade Arrow files, in any
combination.** Convert your own ENDF evaluations, point settings at your own
converted directories, or use a downloaded keyword, and mix all three within
one calculation: `neutron_cross_section_data` alone can be set per-nuclide, so
one network can pull cross sections from a local file for the isotope you care
about and from a downloaded library for everything else. See
[Making your own data](usage.md#making-your-own-data) and
[Nuclear data settings](usage.md#nuclear-data-settings).
