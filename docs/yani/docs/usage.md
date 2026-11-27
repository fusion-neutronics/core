# Usage

The examples on this page assume the nuclear data settings are made, as in
[Getting started](getting_started.md): `neutron_cross_section_data` and the
three `transmutation_*` subsections have no default, and a calculation that
reads an unset one is refused (see [Nuclear data settings](#nuclear-data-settings)).

## Schedules

An irradiation history is one `PulseSchedule` built from `Pulse` (irradiation) and
`Cooldown` (decay only) steps, in order. Every step carries a duration, either
seconds as a plain number or a `(value, unit)` tuple:

| unit | accepted spellings |
| --- | --- |
| seconds | `s`, `sec`, `second`, `seconds` |
| minutes | `min`, `minute`, `minutes` |
| hours | `h`, `hr`, `hour`, `hours` |
| days | `d`, `day`, `days` |
| years | `a`, `y`, `yr`, `year`, `years` (365.25 days) |

<!-- doctest: skip -->
```python
schedule = yani.PulseSchedule(steps=[
    yani.Pulse(rate=1e14, duration=(2, "h"), source=spectrum),
    yani.Cooldown(duration=(30, "min")),
    yani.Pulse(rate=5e13, duration=(2, "h"), source=spectrum),   # different rate
    yani.Cooldown(duration=(1, "a")),
])
```

Each pulse gets its own `rate` and its own `source`, so a campaign whose spectrum
or magnitude changes between phases is expressed directly. The results carry one
material per step, so the number of results is the number of steps.

A decay curve wants points at cumulative times, while `Cooldown` takes the
duration of its own step, so a sweep means differencing the times by hand.
`cooldown_steps()` does that for you:

<!-- doctest: skip -->
```python
schedule = yani.PulseSchedule(steps=[
    yani.Pulse(rate=1e14, duration=(2, "h"), source=spectrum),
    *yani.cooldown_steps(start=(1, "h"), stop=(10, "a"), n=40),
])
```

The steps land at cumulative times from `start` to `stop`, log-spaced by
default: decay is exponential, so a linear sweep spends its points on the flat
tail and misses the early fall. `spacing="linear"` is the other one, and
anything else raises.

!!! note "What `rate` means depends on the entry point"
    For standalone `Material.transmute()`, `rate` is the **total flux magnitude**
    in n/cm²/s: the spectrum supplies the shape and `rate` supplies the size. For
    the coupled `Model.simulate_transmutation()` in yamc, the same field is a
    **source rate** in n/s, because there a transport solve turns emission into
    flux. The `Pulse` docstring gives the latter.

## Spectra

The spectrum rides on the pulse as a `NeutronSource` whose energy distribution is
a `Histogram` of group boundaries and per-group values:

<!-- doctest: skip -->
```python
energy_groups = [1e-5, 0.625, 1e5, 2e7]   # eV, ascending, n+1 boundaries
multigroup_flux = [1e12, 5e12, 1e14]      # n groups: thermal, epithermal, fast

spectrum = yani.NeutronSource(
    energy=yani.sources.Histogram(boundaries=energy_groups, probabilities=multigroup_flux)
)
pulse = yani.Pulse(rate=sum(multigroup_flux), duration=(1, "h"), source=spectrum)
```

The `Histogram` normalizes the shape, so a flux spectrum can be passed in with
its own magnitudes and the total handed to `rate`. The source's position and
direction are ignored, since nothing is transported.

### Named group structures

**Any group structure works.** The boundaries argument is just an ascending list
of energies in eV, of whatever length your spectrum happens to have, and nothing
about YANI prefers one structure over another. The example above uses three
groups.

What the names below add is convenience for a handful of common structures, so
that if your spectrum is already on one of them you do not type out its
boundaries. They are not privileged, and using one is not required:

<!-- doctest: skip -->
```python
spectrum = yani.NeutronSource(
    energy=yani.sources.Histogram(boundaries="CCFE-709", probabilities=flux_709)   # 709 values
)
```

| name | groups | range | typical use |
| --- | ---: | --- | --- |
| `XMAS-172` | 172 | 1.00001e-5 eV to 19.6403 MeV | the European (XMAS/JEF) lattice set |
| `VITAMIN-J-175` | 175 | 1e-5 eV to 19.64 MeV | the long-standing neutron transport standard |
| `SCALE-252` | 252 | 0 eV to 20 MeV | SCALE/AMPX |
| `TRIPOLI-315` | 315 | 1e-5 eV to 19.64 MeV | TRIPOLI |
| `SHEM-361` | 361 | 0 eV to 19.6403 MeV | resonance self-shielding in thermal lattices |
| `LLNL-616` | 616 | 1e-5 eV to 20 MeV | LLNL |
| `CCFE-709` | 709 | 1e-5 eV to 1 GeV | fine-group fusion activation spectra |
| `SCALE-999` | 999 | 1e-5 eV to 20 MeV | fine-group SCALE/AMPX |
| `UKAEA-1102` | 1102 | 1e-5 eV to 1 GeV | the finer companion to `CCFE-709`, same 1 GeV top |
| `ECCO-1968` | 1968 | 1.00001e-5 eV to 19.64033 MeV | ECCO/ERANOS fine group |
| `VITAMIN-J-42` | 42 | 1 keV to 50 MeV | photon, with resolution at the 511 keV and Co60 lines |
| `CCFE-24-PHOTON` | 24 | 1 keV to 20 MeV | coarse photon |

Two of them reach far past the evaluations. `CCFE-709` and `UKAEA-1102` top out
at 1 GeV, while most ENDF/B-VIII.1 evaluations stop at 20 MeV, so a spectrum on
one of those structures with flux in its top groups is refused, naming the
nuclide and the energy its evaluation stops at. Leaving those groups empty is
the normal case and costs nothing. Which library reaches how far is in
[Energy range](libraries.md#energy-range).

Passing a name that is not one of these raises with the list of the ones that
are, so a typo never silently becomes something else. Note that `SCALE-252`
and `SHEM-361` start at exactly 0 eV; treat their bottom bin as open-ended.

If you need the edges themselves, to bin your own data, to plot a spectrum
against them, or to fold a cross section over them, ask for them by name:

<!-- doctest: skip -->
```python
edges = yani.group_structure(name="CCFE-709")
len(edges)              # 710, one more than the group count
edges[0], edges[-1]     # 1e-05 eV, 1e+09 eV

yani.group_structure_names()   # every name in the table above
```

The list is a copy, and the same edges a `Histogram` built on that name bins on.

Cross sections are collapsed against each distinct spectrum once and then scaled
by each step's `rate`, rather than recollapsed every step.

## Materials

A composition dict plus a density. Element symbols expand over natural
abundance, nuclide names are taken literally, and chemical formulas are parsed:

<!-- doctest: skip -->
```python
foil = yani.Material(composition={"Ag": 1.0}, density=10.49, volume=1.0)
enriched_li = yani.Material(
    composition={"Li": yani.enriched(fraction=1.0, target="Li6", percent=60.0)},
    density=0.534,
    volume=10.0,
)
breeder = yani.Material(composition={"Li4SiO4": 1.0}, density=2.4, volume=100.0)
```

For a standard shielding or structural material, the PNNL Compendium
(PNNL-15870 Rev. 2) ships with the package, 410 materials keyed by the report's
own names, so the composition and density come from a citable source rather
than from your own retyping:

<!-- doctest: skip -->
```python
steel = yani.materials.pnnl.material(key="Steel, Stainless 316", volume=1000.0)
concrete = yani.materials.pnnl.material(key="Concrete, Ordinary (NIST)", volume=1e6)

list(yani.materials.pnnl)          # every name in the compendium
```

Pass `volume` at construction for either route: it is what turns the solver's
atom densities into the atom counts that activity and decay heat need.

Two other constructors cover compositions you assemble rather than look up.
`Material.mix_materials` blends existing materials by atom, mass or volume
fraction and works out the mixed density; `Material.from_atom_densities` takes
absolute atoms/barn-cm per nuclide and stores them verbatim, for an inventory
you already hold as number densities:

<!-- doctest: skip -->
```python
eurofer = yani.Material(composition={"Fe": 0.9, "Cr": 0.09, "W": 0.01}, density=7.8)
water = yani.Material(composition={"H2O": 1.0}, density=1.0)
blanket = yani.Material.mix_materials(
    materials=[eurofer, water], fractions=[0.8, 0.2], fraction_type="volume"
)
blanket.volume = 5e4

measured = yani.Material.from_atom_densities(
    {"Fe56": 7.7e-2, "Co59": 5.0e-6}, volume=1.0
)
```

`density` on a `from_atom_densities` material is the sum of the values given.

`transmute()` loads the cross sections it needs into the material and leaves
them there, so calling it again on the same material does no file reading at
all. That is the right trade for a material you transmute more than once and the
wrong one for a sweep over thousands of distinct compositions, each of which
would otherwise hold its nuclides' data for as long as it lives: those of the
composition and of the products the irradiation populates, a few dozen for a
pure metal and up to a couple of hundred for a steel. Call
`material.release_nuclear_data()` when you are done with one; the material stays
usable, and the next call that needs the data loads it again.

## Results

`transmute()` returns a `TransmutationResults` -- the same object the coupled
`Model.simulate_transmutation()` returns in yamc -- keyed by the material's `id`.
Every material has one from construction: the `id` passed to it, or else a
fresh one that no earlier yani material in the process was given (yamc numbers
its own materials separately).
`step_materials()` unpacks the results into `list[Material]`, one per step,
each holding the inventory at the end of that step and keeping the original's
`id`. Extensive quantities need the material's `volume` in cm³:

<!-- doctest: skip -->
```python
results = material.transmute(schedule=schedule)
steps = results.step_materials(material_id=material.id)

for step, mat in enumerate(steps, start=1):
    print(step, mat.activity(), "Bq", mat.decay_heat(), "W")

final = steps[-1]
final.activity(by_nuclide=True)      # dict[str, float], Bq per nuclide
final.decay_heat(by_nuclide=True)    # dict[str, float], W per nuclide
final.nuclides                       # list[(name, fraction)]
final.get_atoms_per_barn_cm()        # dict[str, float], the solver's own units
```

Stable nuclides contribute nothing to activity or decay heat and are omitted from
the by-nuclide dictionaries.

`units=` gives the result in the unit you report in. The default is the total,
in Bq or W, which needs `volume`. A per-volume unit needs neither `volume` nor
`density`, and a per-mass unit needs only `density`. Per mass is per kg of the
material you started from at every step, not of that step's inventory, so a
per-kg series has the same shape as the total. SI prefixes work, and activity
also takes curies:

<!-- doctest: skip -->
```python
final.activity(units="Bq/kg")             # per mass, as handbooks quote it
final.activity(units="kBq/g")              # per mass, needs only density
final.activity(units="Bq/cm3")             # per volume, no volume needed
final.activity(units="TBq")                # a total, needs volume
final.activity(units="Ci")                 # curies
final.decay_heat(units="W/kg")             # per mass
final.decay_heat(units="MW/m3")            # per volume
final.decay_heat(units="MeV/s")            # a total in another energy unit
final.decay_photon_spectrum(units="1/(s cm3)")  # photons/s/cm3
```

| quantity | default | per volume | per mass | other totals |
| --- | --- | --- | --- | --- |
| `activity` | `"Bq"` | `"Bq/cm3"`, `"GBq/m3"` | `"Bq/g"`, `"Bq/kg"`, `"kBq/g"` | `"kBq"`, `"TBq"`, `"Ci"`, `"mCi"` |
| `decay_heat` | `"W"` | `"W/cm3"`, `"MW/m3"` | `"W/g"`, `"W/kg"` | `"kW"`, `"MeV/s"` |
| `decay_photon_spectrum` | `"1/s"` | `"1/(s cm3)"` | `"1/(s kg)"` | |
| `decay_photon_continua` | `"1/(s eV)"` | `"1/(s eV cm3)"` | `"1/(s eV kg)"` | |

A unit of the wrong kind, such as `decay_heat(units="Bq/kg")`, raises a
`ValueError` naming what was expected.

`component=` splits decay heat into the recoverable parts the decay data gives,
`"beta"`, `"gamma"` or `"alpha"`. The gamma heat is the part that leaves a thin
component, while the beta and alpha heat is deposited where the decay happens:

<!-- doctest: skip -->
```python
final.decay_heat(component="gamma")                  # W, the photons alone
final.decay_heat(component="beta", by_nuclide=True)  # dict[str, float], W
```

A nuclide making decay heat whose data carries no split, which data converted
before the split was recorded does not, raises rather than understating the
component.

`contact_dose()` is the third of these, and the one that needs no `volume`: it
is the dose someone receives with a hand on the material, from the material's
own decay photons, and a bigger lump of the same material reads the same at
contact.

<!-- doctest: skip -->
```python
final.contact_dose()                          # Gy/h, absorbed dose in air
final.contact_dose(by_nuclide=True)           # dict[str, float], Gy/h per nuclide
final.contact_dose(dose_quantity="effective") # Sv/h, ICRP-116 effective dose
final.contact_dose(units="uGy/h")             # the same dose in another rate unit
final.contact_dose(dose_quantity="effective", units="mSv/h")
```

A contact dose is already per unit of something, so `units=` takes only a rate
of the same kind: Gy for the absorbed dose in air and Sv for the effective
dose, with any prefix and time unit (`"mGy/h"`, `"uSv/h"`, `"mSv/a"` for a
year, `"Gy/s"`).

It is a slab estimate rather than a transport result. The material is taken to
be a half-space, so half the photons emitted at any depth head for the surface,
and the ones that arrive are the ones the material did not attenuate itself.
Integrating over depth cancels the geometry, which is why no distance and no
volume appear in the answer. A `build_up` factor, 2.0 by default, stands in for
the photons that scatter on the way out and still arrive.

The two NIST tabulations it folds against are public, so a response function of
your own can be built from the same data: `mu/rho` for any element from Z = 1 to
100, and `mu_en/rho` for air. Both read log-log between their tabulated points.

<!-- doctest: skip -->
```python
iron = yani.data.mass_attenuation_coefficient(element="Fe")   # or by atomic number, 26
iron.interpolate(energy=1.0e6)                               # cm2/g at 1 MeV
air = yani.data.mass_energy_absorption_coefficient(material="air")
```

Fluence-to-dose coefficients for folding with a transport tally come from
`yani.data.dose_coefficients()`. Effective dose is the default, from ICRP-116 or
ICRP-74 for an irradiation geometry; `dose_quantity="ambient"` gives the ambient
dose equivalent H*(10) from ICRP-74, the quantity most area-monitoring
regulations still reference. H*(10) is defined in an aligned and expanded field,
so it takes no `geometry`:

<!-- doctest: skip -->
```python
effective = yani.data.dose_coefficients("neutron", geometry="AP")
h10 = yani.data.dose_coefficients("neutron", dose_quantity="ambient")
```

The H*(10) tables cover 1 meV to 20 MeV for neutrons and 10 keV to 10 MeV for
photons and are not extrapolated, so a particle outside that range scores
nothing. H*(10) is meant to over-estimate effective dose, but for high-energy
neutrons it reads under it.

The default quantity follows UKAEA-CCFE-RE(21)02, Appendix C.7.1. Bremsstrahlung from
decay electrons is not modelled, so a strong beta emitter reads low at contact.

The decay photon line spectrum comes back as the `(x, p)` pair the source
distributions take, in photons per second, so it feeds straight into a photon
transport run in yamc:

<!-- doctest: skip -->
```python
energies, intensities = final.decay_photon_spectrum()
```

Part of some decay spectra is given in the data as a density over energy rather
than as lines: the spontaneous-fission photons of an actinide, or the whole
photon emission of a nuclide far from stability. `decay_photon_spectrum()` holds
the lines only, and `decay_photon_continua()` returns the rest, one
`PhotonContinuum` per nuclide and continuum. Its rates are per eV, so they are
not line rates and cannot be added to the spectrum above; `emission_rate` is
their integral. It takes `units=` as a density per eV:

<!-- doctest: skip -->
```python
for continuum in final.decay_photon_continua(units="1/(s eV cm3)"):
    continuum.nuclide          # e.g. "Cf252"
    continuum.energies         # eV, the continuum's own grid
    continuum.rates            # photons/s/eV/cm3 on that grid
    continuum.interpolation    # the ENDF law between points, e.g. "linear-linear"
    continuum.emission_rate    # photons/s/cm3, the integral
```

### Inventory and gas-production diagnostics

Composition getters include the initial state at index zero; `times` is the
matching cumulative-time array. Unknown material IDs or states return `None`:

```python
print(results.material_ids, results.num_steps, results.times)
co60 = results.get_nuclide_evolution(material_id=material.id, nuclide="Co60")
state = results.get_material(material_id=material.id, step=1)
if state is not None:
  print(state.get_nuclide_names())

gas = results.get_gas_production(material_id=material.id)
if gas is not None:
  print(list(zip(results.times, gas["H"], gas["He"])))
screening = results.get_product_screening_info(material_id=material.id)
if screening is not None:
  print(screening["loaded"], screening["below_threshold"], screening["unavailable"])
print(results.data_gaps)
```

Gas production is in appm, relative to the material's initial atom count.
By default the initial gas is subtracted; `produced=False` returns the gas
present, including what was there initially. A consumed gas can therefore have
negative net production. A chain missing one of the five gas nuclides raises
rather than reporting its unknown contribution as zero.

With `data_uncertainty=` the gas comes with a nuclear-data spread as well,
one `Estimate` per gas nuclide and total at a single step:

<!-- doctest: skip -->
```python
gas = results.get_gas_production_uncertainty(material_id=material.id, step=1)
if gas is not None:                  # None unless data_uncertainty was given
    gas["He"].nominal, gas["He"].std_dev
```

Each replica's gas is taken from its own inventory against the one initial
inventory, and the `"H"` and `"He"` totals are summed within a replica before
the spread is taken, so they are not the quadrature sum of their parts.

Product screening omits the neutron reactions of sufficiently small products,
not their production or decay. `data_gaps` reports missing-data assumptions
separately, so a small or zero activity is not a substitute for checking it.

### Displacement damage

`displacement_damage=True` also computes the damage the irradiation does to the
material's lattice: the damage energy deposited per atom, and NRT displacements
per atom (dpa), per element and for the material. Each getter returns one
cumulative value per state, aligned with `times`, starting from zero:

<!-- doctest: skip -->
```python
iron = yani.Material(composition={"Fe": 1.0}, density=7.874, volume=1.0)
schedule = yani.PulseSchedule(steps=[
    yani.Pulse(rate=sum(multigroup_flux), duration=(1, "a"), source=spectrum),
    yani.Cooldown(duration=(1, "d")),
])
results = iron.transmute(schedule=schedule, displacement_damage=True)

results.get_dpa(material_id=iron.id)                  # [0.0, 6.49, 6.49]
results.get_dpa(material_id=iron.id, element="Fe")    # the iron's own dpa
results.get_damage_energy(material_id=iron.id)        # eV per atom: [0.0, 649.4, 649.4]
results.get_displacement_damage_info(material_id=iron.id)
```

Each nuclide's MT=444 damage-energy cross section is folded against the pulse
spectrum by the same collapse as the reaction rates, self-shielding included,
at each step's composition; cooldowns add nothing. An element's dpa is
`0.8 * E_damage / (2 * E_d)` (the NRT model, ASTM E521), with the displacement
threshold energy `E_d` from ASTM E521, or from the OECD-NEA 2015 report
"Primary Radiation Damage in Materials" for an element E521 does not cover.
`displacement_energies={"Fe": 40.0}` replaces them, and an element in neither
must be given there. The material total weights the elements by atom fraction.
`get_displacement_damage_info` names the `E_d` used for each element and its
source, the transmutation products whose data has no MT=444
(`without_damage_energy`) and the product elements with no `E_d`, hydrogen and
helium above all (`without_displacement_energy`), each with the largest atom
fraction it reached. Without the flag, MT=444 is not fetched and nothing about
the solve changes.

Pass a `DataUncertainty` as well and the dpa and damage energy come with a
nuclear-data spread, one `Estimate` per step:

<!-- doctest: skip -->
```python
results = iron.transmute(
    schedule=schedule,
    displacement_damage=True,
    data_uncertainty=yani.DataUncertainty(seed=42),
)
dpa = results.get_dpa_uncertainty(material_id=iron.id, step=1)
if dpa is not None:                 # None unless both were asked for
    dpa.nominal, dpa.std_dev        # 6.49, 0.158: 2.4%
results.get_damage_energy_uncertainty(material_id=iron.id, step=1, element="Fe")
```

Each replica's dpa is accumulated over its own inventory. Under
`cross_sections`, MT=444 has no covariance of its own, but it is the sum over
reactions of each cross section times the damage energy that reaction deposits,
so a draw moves it by each reaction's deposit times its drawn change. Elastic
and inelastic scattering, which carry most of it, are drawn with the activation
channels as one evaluation. What no reaction's deposit reproduces, fission's
share above all, is held, and `get_data_uncertainty_info(...)["damage_energy"]`
gives, per nuclide of the starting composition, the share of its MT=444 rate the
deposits reproduce (`explained`) and the relative sigma of its drawn rate
(`sigma`); for this iron, Fe56 is explained in full with a sigma of 2.8%. Under
`flux_spectrum`, the damage energy moves with the replica's flux draw. The
damage energy of transmutation products, short-range (LB=8) covariance and the
displacement threshold energies are held at nominal. Asking for damage widens
what is drawn for each nuclide of the starting composition, so for one seed the
draws differ from those of a run without `displacement_damage`.

### Clearance and disposal limits

A material can also be checked against the regulatory clearance, exemption and
disposal limits directly, from yani 0.19.0. `clearance_index()` divides
each radionuclide's activity by its limit in a set and sums the ratios; the
material meets the limits when the sum is below the set's threshold, normally 1.
The sets cover the UK (EPR 2016, IRR 2017), Germany (StrlSchV), the US (NRC 10
CFR 61, Fetter), the EU Basic Safety Standards and IAEA GSR Part 3, and come
from the
[radiological-material-clearance-finder](https://github.com/fusion-energy/radiological_material_clearance_finder)
crate, whose documentation describes each one and where its numbers come from.

The half-lives are the transmutation chain's, the same ones `activity()` and
`decay_heat()` use. The composition is enough for the Bq/g and Ci/m3 sets; only
the total activity sets need `volume`. Stable isotopes have to stay in the
material, since they are most of its mass and so most of the Bq/g denominator.

<!-- doctest: skip -->
```python
result = final.clearance_index("UK_EPR16_out_of_scope")
result.index           # float, sum of activity-to-limit ratios
result.clearable       # bool, index below the set's threshold
result.by_nuclide      # dict[str, float], each nuclide's ratio, largest first
result.uncovered       # dict[str, float], activity no limit in the set covers
print(result)          # a short report of the dominant nuclides

results = final.clearance_indices()   # dict[str, ClearanceResult], every set
routes = [name for name, r in results.items() if r.clearable]
```

`metal=True` applies the NRC's activated metal rows, `apply_default_limit` and
`exclude_daughters` switch off the catch-all limit and the secular equilibrium
credit that some tables define, and `uncovered_fraction` is the number to check
before trusting a comfortable index.

The rest of a `ClearanceResult` says how the index was reached:

- `limits_used` is the limit applied to each nuclide after any metal, per-gram,
  dynamic or secular-equilibrium adjustment, and `assessed_activity(nuclide)`
  the activity charged against it: the nuclide's total less any part a parent
  already accounted for.
- `defaulted` lists the nuclides that took the set's catch-all limit,
  `unlimited` those the source explicitly places no limit on, and
  `uncovered_activity` the total activity no limit covers, in the set's units.
- `out_of_scope` is true when the regulation excludes the material outright;
  for the UK sets that means every radionuclide present is shorter lived than
  100 s.
- `limit_set` and `material_name` are carried through for reporting, and
  `to_json("clearance.json")` saves the whole result as JSON; with no path,
  `to_json()` returns the JSON as bytes, so `json.loads(result.to_json())`
  gives it as plain Python.

### Several materials at once

`yani.transmute()` is the plural of `Material.transmute()`, for the cells of a
mesh, a component broken into regions, or a sweep over compositions. Each
material gets exactly the answer `material.transmute(schedule)` would give it,
and the result is one `TransmutationResults` keyed by each material's `id`, so
the ids must be distinct. Every material is built with its own, so two share one
only when one was given it explicitly or came out of a transmutation result: the
materials a result returns keep their original's `id`, so they share it with the
original and with each other.

<!-- doctest: skip -->
```python
inner = yani.Material(composition={"Fe": 1.0}, density=7.87, volume=10.0)
outer = yani.Material(composition={"Fe": 1.0}, density=7.87, volume=10.0)

results = yani.transmute(materials=[inner, outer], schedules=[schedule, schedule])
results.get_source_rates(material_id=outer.id)   # list[float], each step's flux, n/cm2/s
results.collapse_reuse                           # {"performed": ..., "requested": ...}
```

Pass one schedule for every material, or one per material in the same order.
Each schedule carries that material's own spectra and flux magnitudes, but all
of them must share one timeline: the same durations, irradiating on the same
steps. What a Python loop would repeat is done once: the chain is read once,
each nuclide's cross sections are decoded once for every material that needs
them, and a collapse runs once per distinct spectrum, composition, temperature
and shielding, so cells of one steel that saw the same spectrum share it
whatever their flux magnitudes. `collapse_reuse` says how many collapses that
saved, and the solves then run in parallel. `data_uncertainty`,
`self_shielding_chord` and `self_shielding_shape` apply to every material as
`Material.transmute()` applies them to one, and the same seed perturbs a given
evaluation the same way in every material.

`get_source_rates()` gives the rate each step drove a material at, zero for a
cooldown. On a spectrum solve that is the material's own flux magnitude in
n/cm²/s; on a yamc transport run it is the source strength in n/s.

### Saving results

`to_arrow()` writes a `TransmutationResults` to an Arrow IPC file, and
`TransmutationResults.from_arrow()` reads it back with every getter answering
as before, to the bit. Without a path, `to_arrow()` returns the file's bytes,
for a browser or Pyodide session with no file system to write to.

A calculation too big to hold at once (thousands of mesh cells, a long
schedule, an uncertainty run) can be split into parts, each saved and dropped
before the next:

<!-- doctest: skip -->
```python
for part, cells in enumerate([cells[:500], cells[500:]]):
    results = yani.transmute(materials=cells, schedules=schedule)
    results.to_arrow(f"part_{part}.arrow")

results = yani.TransmutationResults.from_arrow("part_0.arrow")
data = results.to_arrow()                       # bytes
yani.TransmutationResults.from_arrow(data)
```

Every file has the same columns, one row per stored number: `material_id`,
`material_name`, `step`, `quantity` (`"density"`, `"reaction_rate"` or
`"source_rate"`), `nuclide`, `reaction`, `product` and `value`. So the parts
read as one table without yani:

<!-- doctest: skip -->
```python
import polars as pl

pl.scan_ipc("part_*.arrow").filter(pl.col("quantity") == "density").collect()
```

The file holds the results, not the nuclear data behind them. Three getters
read that data again: `get_production_routes` and `get_isomeric_branching` walk
the transmutation chain, and `get_reaction_rate_spectrum` reads the initial
composition's cross sections. The file pins both, the chain by a hash of its
contents and each nuclide by the file it was read from, and `from_arrow()`
reattaches them from the configured `yani.transmutation_*` sources and those
files. Where that data is not the same or is no longer there, it warns once,
and those three getters raise rather than answer from other data. Everything
else answers from the file.

### Continuous processing

`Processing` describes continuous feeds, extraction and transfers between
materials during a step. Pass it to the batch `transmute()` function, not to
`Material.transmute()`, so every source and destination is part of one solve:

```python
processing = yani.Processing()
processing.add_transfer(
  material=inner,
  component="H3",
  rate=1e-4,
  rate_units="1/s",
  destination=outer,
)
processing.add_feed(
  material=inner,
  composition={"Li6": 0.6, "Li7": 0.4},
  rate=1e-3,
  rate_units="g/s",
  steps=[0],
)
processed = yani.transmute(
  materials=[inner, outer], schedules=schedule, processing=processing,
)
print(processing.material_ids)
```

The transfer removes a fraction of the named component per second; omitting
`destination` removes it from the system. An element name selects every isotope,
and omitting `component` transfers the whole composition. Feed compositions are
atom fractions, normalized internally. Absolute feed units such as `g/s`,
`kg/d`, `mol/h` and `atoms/s` need the receiving material's volume.

`steps` selects zero-based schedule steps. Omitted, a term applies to every
step, including cooldowns. Rates can also be tables of `(times, rates)`, and
`proportional_to="Li6"` makes a feed follow a nuclide's current inventory with
rate units such as `1/s`; the full forms are in the
[API reference](api.md#continuous-processing).

## Self-shielding

A lump of a resonance absorber shields itself: the flux inside it is depressed
exactly where the total cross section is large, so the reaction rate is lower
than the same material spread thin. Nothing is corrected unless you ask, because
a `Material` carries no geometry to infer a size from. Give the lump a shape, or
state its mean chord `4V/S` in cm yourself:

<!-- doctest: skip -->
```python
results = foil.transmute(
    schedule=schedule,
    self_shielding_shape=yani.shapes.FoilLump(thickness=0.1),   # cm
)
```

`SphereLump()` and `CubeLump()` are fixed by the material's `volume` and take no
arguments, while `FoilLump(thickness=)`, `CylinderLump(radius=)` and
`WireLump(radius=)` carry the one dimension a volume cannot imply. Give a shape
or `self_shielding_chord=`, not both. A sphere has the least surface for its
volume, so it has the longest chord and shields more than any other shape of the
same size: an upper bound rather than a safe default, which is why there is no
default at all.

The flux inside the lump comes from a slowing-down solve, which assumes nothing
about resonances being narrow. The cheaper narrow-resonance approximation is
deliberately not offered, because it over-shields strong elastic scatterers
badly enough to be worse than applying no correction at all: on W186(n,gamma) in
FNG-tung, measured at 1.29 b, the slowing-down solve gives C/E 0.80, no
correction gives 0.86 and narrow resonance gives 0.46.

What was done comes back on the results, and a run that shielded nothing says so
rather than staying silent:

<!-- doctest: skip -->
```python
shielding = results.get_self_shielding_info(material_id=foil.id)
if shielding is not None:              # None for a coupled solve or unknown id
    shielding["method"], shielding["chord_cm"]   # how, and the chord used
    shielding["shielded"]              # nuclides the correction reached
    shielding["not_shielded"]          # and why each of the others was left
    shielding["strongest_factor"]      # smallest factor any group average took
```

A `chord_cm` of `None` means the run was dilute and nothing was corrected. A
`strongest_factor` of `1.0` means the correction ran and changed nothing in
practice. A dilute run fills `would_shield` instead: nuclides whose own
resonances are structured enough to have suppressed a reaction, each mapped to
how strongly.

`would_shield` is an indicator, not a correction and not a bound. There is no
geometry in it. The weight is `1 / (1 + N * sigma_x)` on that one reaction and
that nuclide's own density, which fixes the background at 1/cm, while the
correction proper uses `1 / chord_cm` against the material's total with
in-scattering. So it can sit either side of the real factor: a lump thinner than
a centimetre of chord shields less, and a material whose other nuclides dominate
the total across a resonance dips the flux further than one reaction can
express. On the FNS tungsten foil it reads 0.634 for W186, against a
slowing-down correction that gives 0.750 at a 1 mm chord and saturates at 0.730.

Read it as "this answer may be high, and this is a resonance absorber", which
is the warning a dilute run should carry rather than silence. For the size of
the effect, ask for a shielded run.

## Production routes

`TransmutationChain` carries the topology behind the solve: which nuclide becomes
which, and how. `reactions` holds the neutron-induced edges and `decays` the
radioactive ones, both keyed by parent and both giving
`(kind, target, branching)`:

<!-- doctest: skip -->
```python
chain = yani.TransmutationChain.from_arrow("transmutation-endf-b8.1-sfr.arrow")

chain.reactions["W186"]   # [('(n,2n)', 'W185', 1.0), ..., ('(n,a)', 'Hf183', 1.0)]
chain.decays["Hf183"]     # [('beta-', 'Ta183', 1.0)]
chain.decays["W185_m1"]   # [('IT', 'W185', 1.0)]
```

Stable nuclides have no decay modes and are absent from `decays`, as they are
from `half_lives`. A `target` of `None` means the channel names no single
product: fission on the reaction side, a product outside the chain on the decay
side. Decay modes carry the evaluation's own spellings (`"beta-"`, `"ec/beta+"`,
`"alpha"`, `"IT"`, `"sf"`, and emissions written as `"beta-,n"`), and the
branchings for one parent sum to 1.

The two share a tuple shape, so a route walk can put them in one edge list. With
reactions alone the walk stops at the first product, one step short of most real
routes:

<!-- doctest: skip -->
```python
edges = {}
for parent, es in chain.reactions.items():
    edges.setdefault(parent, []).extend(es)
for parent, es in chain.decays.items():
    edges.setdefault(parent, []).extend(es)

for kind1, via, _ in edges["W186"]:              # two-step routes from W186 to Ta183
    for kind2, end, _ in edges.get(via, []):
        if end == "Ta183":
            print(f"W186{kind1}{via}({kind2}){end}")

# W186(n,3n)W184((n,np))Ta183
# W186(n,4n)W183((n,p))Ta183
# W186(n,a)Hf183(beta-)Ta183      <- the one that needs the decay edge
```

The branching on an edge is a share of its own channel, not a rate: it says how a
channel splits, not how often the channel fires. Two routes into the same product
are not weighted against each other by these numbers alone.

Note that the chain this object holds is the base three-part merge. The isomeric
branching overlay is applied inside the solve, so a `(n,2n)` edge here names the
ground state where the overlay would split it between ground state and isomer.

### Weighted, from a solve

The walk above can enumerate routes but not rank them, and the branching it
reads is the file's, not the one the solve used. `get_production_routes()`
answers the same question from a solve that has happened. Asked for the Ta183 of
the example above:

<!-- doctest: skip -->
```python
routes = results.get_production_routes(material_id=mid, product="Ta183", step=0)
for route in routes or []:          # None if the material or the step is unknown
    print(f"{route['route']:<32} {route['share']:.1%}")

# W183(n,p)Ta183                    72.4%
# W186(n,a)Hf183(BETA-)Ta183        16.1%   <- the one that needs the decay edge
# W184(n,np)Ta183                    7.1%
# W184(n,d)Ta183                     4.4%
```

Each entry also carries `steps`, the same route as `(parent, kind, target)`
triples with the chain's own spellings (`('Hf183', 'beta-', 'Ta183')`), where the
`route` string uppercases decay kinds the way the published pathway tables print
them.

The walk starts from the nuclides the material began with, which a bare chain
cannot do. Asked what makes W187, a chain lists
`Os190(n,a)` and `Ir192(n,npa)` as readily as `W186(n,gamma)`, and nothing in a
tungsten foil is osmium.

`share` is the fraction of that product's production arriving down the route: the
atoms it starts from, times what its reaction drove per atom of its parent over
the step, times the branching of every decay it passes through, so a route
through a 1% branch delivers 1% of what the reaction made. Reaction steps carry
the step duration too, so `reaction_depth=2` is in the same units as the default
1 and comes out smaller by roughly a factor of the fluence. `production` is the
same number unnormalised, in atoms per barn-cm, because 100% of almost nothing
and 100% of the inventory read alike otherwise.

The flux-weighted isomeric branching comes from the same place:

<!-- doctest: skip -->
```python
branching = results.get_isomeric_branching(material_id=mid, step=0) or []
branching[0]
# {'parent': 'W186', 'reaction': '(n,2n)', 'production': 9.35e-14,
#  'split': [('W185_m1', 0.535), ('W185', 0.465)]}
```

Channels come back ordered by what they made, the channel's rate times its
parent's atom density at the start of the step, rather than by rate alone. A
rate is per atom of its parent, so ordering on that promotes whatever sits on a
trace isotope: on the FNS tungsten foil `W180 (n,2n)` has the highest per-atom
rate of any channel in the foil, and W180 is 0.12% of it, so weighted by what it
actually made the channel falls to fifth, two orders of magnitude below the
`W186 (n,2n)` carrying most of that foil's decay heat. A parent the step did not
start with has a production of `0.0` and sorts last rather than being dropped.

Which state a reaction leaves its product in is energy dependent, so the one
number describing a spectrum is the branching collapsed against it, and that
exists only inside a solve. In the chain file that same channel reads
`W186 (n,2n) -> W185 1.000000` and `-> W185_m1 0.000000`, a placeholder the
overlay replaces at solve time. Only channels landing in more than one final
state are returned, since a single-product channel has no branching to report and
listing it at 1.0 buries the ones that do.

This is what separates a disagreement caused by a cross section from one caused
by a branching ratio. On a foil whose decay heat comes from an isomer, getting it
wrong is a factor of several.

`get_branching_report()` says how each channel's split was read from the
evaluation over the step's spectrum: as shares of the reaction's total, or as
absolute productions with the ground state taking the rest, and how much of the
parent's removal rate rests on anything the evaluation does not give:

<!-- doctest: skip -->
```python
report = results.get_branching_report(material_id=mid, step=0)
if report is not None:
    channel = report["channels"][0]
    channel["parent"], channel["reaction"], channel["representation"]
    channel["states"]            # each final state and its share of the reaction
    channel["clipped_share"]     # of the parent's removal rate
    report["dropped"]            # channels that could not be folded, and why
    report["unmodelled_mt5"]     # (nuclide, share, reason) where MT=5's residuals are not modelled
```

A run refuses when the clipped or held production of its channels, each parent
weighted by its density in the material, is more than 0.1% of the material's
neutron removal rate; each channel's `clipped_share` and `extrapolated_share`
are of its own parent's removal. Weighting by the material keeps one bad list
on a product present at a trace from refusing every irradiation whose network
reaches it, while a material made of that parent is held to the same 0.1%.

MT=5, `(n,anything)`, is the `(n,X)` reaction. The reactions subsection carries
its products from the evaluation's MF=6 MT=5: each residual with its isomeric
state, folded as a share of the MT=5 total at each energy (`file` 6,
`representation` `"share"`), and each light particle H1, H2, H3, He3 and He4 as
its multiplicity times that total (`representation` `"multiplicity"`, whose
`share` is the multiplicity folded over the spectrum and can exceed one). Their
gas is therefore in the inventory. In `TransmutationChain.reactions` an `(n,X)`
edge's branching is NaN, since its split depends on the spectrum.

`unmodelled_mt5` lists what is left: a parent whose evaluation gives MT=5's
light particles but not its residuals (ENDF/B-VIII.1's Fe58 and zirconium
isotopes, among others), each with its share of the parent's removal rate and
the reason. A run refuses when the material's own nuclides lose more than 0.1%
of the material's removal rate that way. A reactions subsection converted
before MT=5 was carried has no `(n,X)` at all, and its MT=5 is reported, as it
was, rather than refused.

### Where in energy a rate came from

A one-group rate cannot say which part of the spectrum made it.
`get_reaction_rate_spectrum()` resolves one channel's rate onto the groups of
the spectrum that drove it:

<!-- doctest: skip -->
```python
capture = results.get_reaction_rate_spectrum(
    material_id=mid, nuclide="W186", kind="(n,gamma)", step=0
) or {}

edges = capture["boundaries"]     # group boundaries in eV, one longer than rates
per_group = capture["rates"]      # each group's contribution, 1/s

sum(per_group)                    # the rate get_reaction_rates() reports
```

The two agree because they come from the same walk of the same cross sections,
self-shielding included, so this is a decomposition of the rate the solve used
rather than a second estimate of it.

The attribution is what the single number hides. A capture cross section spans
decades, and an effective one-group value of tens of millibarns against a
spectrum that is almost all fast is either a fast-capture rate or a
resonance-region rate. Which one it is decides whether a disagreement belongs to
the resonance processing or to the fast cross section, and those are different
data and different fixes:

<!-- doctest: skip -->
```python
below_100_keV = sum(r for lo, r in zip(edges, per_group) if lo < 1.0e5)
print(f"{below_100_keV / sum(per_group):.1%} of the capture rate is resonance region")
```

It is also what to read beside `rate_fraction_covered`: set against the energies
where a channel's covariance states a nonzero variance, it shows, to the
resolution of the groups, where in energy the covered and uncovered parts of
the rate the run actually used sit, which the share gives only as one number.
On a shielded run it also shows which groups the flux depression moved, which
`strongest_factor` gives only as a worst case.

Nothing is stored for it. One reaction over a 709-group structure is cheap to
walk when asked, and keeping the breakdown for every channel would be tens of
megabytes, so a run that never asks pays nothing. `None` comes back for a step
that drove no flux, for a nuclide or channel the data does not carry, for
`(n,n')`, whose rate comes from the branching overlay rather than from a group
average, and for a coupled yamc run, which scores its rates at the collision
energy and keeps no group structure to resolve them onto.

## Inspecting nuclear data

The bundled names, natural abundances and nuclide-name parser need no nuclear
data download:

```python
print(yani.data.element_names()["Fe"])
print(yani.data.reaction_names()[102])
print(yani.data.atomic_number("Fe"), yani.data.atomic_symbol(26))
print(yani.data.split_nuclide("Ta180_m1"))
print(yani.data.element_nuclides()["Li"])
print(yani.data.natural_abundance()["Li6"])
print(yani.data.natural_abundance_records()["Li6"])
print(yani.cache_dir())
```

Natural-abundance records retain the reference's stated uncertainty or interval;
an absent value is `None`, not zero. These reference uncertainties are not
automatically propagated by a transmutation calculation.

Converted data can be inspected without running a solve. Pass the individual
nuclide or element directory written by the converter, not the library root:

```python
nuclide_info = yani.read_nuclide_from_arrow("converted/Fe56.arrow")
element_info = yani.read_element_from_arrow("converted/Fe.arrow")
print(nuclide_info, element_info)
```

For a long-running inspection process, `yani.data.clear_nuclide_cache()` clears
the in-memory nuclide cache, not the downloaded files. Release data held by a
material with `material.release_nuclear_data()` when the material is no longer
needed for the current sweep.

## Nuclear data settings

Five module-level settings, read and written like attributes:

| setting | supplies | decides | left unset |
| --- | --- | --- | --- |
| `neutron_cross_section_data` | continuous-energy cross sections per nuclide | how fast each channel runs | an irradiation is refused, naming the setting |
| `transmutation_reactions` | which reaction on which nuclide gives which product, and its Q | which products exist to be made | none: a calculation that reads it is refused, naming the setting |
| `transmutation_decay_data` | half-lives, decay modes, mean decay energies, photon lines | how the inventory decays and what it emits | none: a calculation that reads it is refused, naming the setting |
| `transmutation_fission_yields` | fission product yields | what a fission makes | none: a calculation that reads it is refused, naming the setting |
| `transmutation_branch_ratios` | isomeric branching against incident energy | which product is left in a metastable state | no overlay, so every product lands in its ground state |

Each takes a library keyword or a path, and each can point somewhere different,
so a network can mix libraries by subsection: a TENDL reactions network
borrowing ENDF/B-VIII.1 decay data is the usual arrangement. The first two are
easily confused, and are not alternatives: the cross sections decide how fast
each reaction runs, and the reactions network, which holds no cross sections,
decides which reactions and products exist. See
[The two that both say reactions](#the-two-that-both-say-reactions). They are global
process state, not per-call arguments. See
[Nuclear data libraries](libraries.md) for which library publishes which
subsection, and how much is in each.

There is also a `photon_cross_section_data` setting, the same one yamc has, which
selects photon interaction data for photon transport. No yani calculation reads
it: the photon attenuation behind `contact_dose()` is the built-in NIST XCOM
tabulation, so it can stay unset. It is independent of
`neutron_cross_section_data`, and neither falls back to the other.

Note which way the last column goes. Only `transmutation_branch_ratios` has a
default. The other four name the data a result is made from, so none is chosen
for you and nothing is downloaded from a library you did not pick: leave one
alone and a calculation that reads it stops with a `ValueError` naming each
unset setting, the hosted keywords and a line to copy. What each calculation
reads:

| calculation | reads |
| --- | --- |
| a schedule with an irradiating step (`Material.transmute()`, `yani.transmute()`) | `neutron_cross_section_data` and all three `transmutation_*` subsections above |
| a schedule that only cools, and `activity()`, `decay_heat()`, `contact_dose()`, `clearance_index()` and the other decay quantities | `transmutation_decay_data` |

A decay-only schedule folds no cross sections and runs without
`neutron_cross_section_data`.

### The two that both say reactions

`neutron_cross_section_data` and `transmutation_reactions` are halves of one
calculation rather than alternatives, and the rate collapse is where they meet.
For each nuclide, the network lists the channels it has; each channel's name is
mapped to an MT number; and the cross sections are what that MT is priced with.
So the network decides **which channels are considered at all** and the cross
sections decide **how fast each one runs**. Neither substitutes for the other:
swap the library behind `transmutation_reactions` and products appear or
disappear, swap the one behind `neutron_cross_section_data` and the same
products arrive at different rates.

Two consequences follow from that loop, one quiet and one not:

* A channel the network names whose MT the cross sections do not carry is
  skipped. `(n,n')` is the standing example: it has no transport MT, so its rate
  comes from the branching overlay's partials rather than from a group average,
  which is why it is absent from a `get_reaction_rate_spectrum()` answer.
* A missing `neutron_cross_section_data` is refused, per the column above, and
  so is a network pointed at the wrong material: before the solve, naming what the
  material holds against what the network drives.

`transmutation_reactions` and `transmutation_fission_yields` additionally accept
`False`, which turns that subsection off. A decay-only calculation carries no
reaction rates, so it needs no reaction topology; a material nothing in which
fissions needs no yields, and neither does a library that publishes none of its
own. Off is not the same as unset: `None` leaves the setting unset, so a calculation
that reads it is refused, whereas `False` says the subsection is absent on
purpose. Each reads back as it was assigned: the source, `False` or `None`. A rate that then needs it is
refused when the burnup matrix is built, naming the nuclide and the reaction,
rather than being solved as though the reaction produced nothing.

A reactions network built for one material and pointed at another is refused on
the same terms. Networks are often scoped to the nuclides they were built from,
because walking a whole library to activate one foil is wasted work. The cost is
that two scoped networks look alike from the outside: two conversions writing to
the same output path leave the second network under a filename that names the
first. Solving against the wrong one does not fail. Every step runs, no rate is
negative, and the composition comes back as it went in, so the mistake surfaces
as a decay heat of exactly zero much later with nothing to point at. An
irradiated schedule whose material has no drivable nuclide is therefore refused
up front, naming both what the material holds and what the network has reactions
for. A decay-only schedule is exempt, since driving nothing is correct for it.

## Where the data comes from

A keyword is resolved on first use, once per process, to the library's current
published release: a dated folder such as `2026-10-10.1` with a manifest listing
every file in it with its size and sha256. The sections needed are downloaded
from that release, each checked against the manifest (a byte range against its
own hash) before it is kept, then reused. A file that does not match is refused
with an error naming it, never cached. Only the sections activation reads are
fetched, never a whole library, so this is a small fraction of a transport data
set. A load of many nuclides fetches them in concurrent batches and prints a
`Downloaded k of N nuclides` counter as it goes.

The cache is a `yamc` folder in the platform's per-user cache directory, shared
with yamc, and `yani.cache_dir()` returns it:

| platform | cache directory |
| --- | --- |
| Linux | `~/.cache/yamc`, or `$XDG_CACHE_HOME/yamc` when that is set |
| macOS | `~/Library/Caches/yamc` |
| Windows | `%LOCALAPPDATA%\yamc` |

Each release keeps a folder of its own, laid out as the published tree:

```
<cache dir>/<keyword>/<release>/manifest.json
<cache dir>/<keyword>/<release>/neutron/<Nuclide>.arrow/...
<cache dir>/<keyword>/<release>/transmutation/<subsection>.arrow/...
```

There is no setting that moves the cache. To read data from somewhere else,
point the setting at a local directory instead (below), which bypasses the
cache. When a newer release is published, the next run downloads what it needs
from the new release, says so once, and leaves the older folder in place; every
file of one library a run reads comes from a single release. With no connection,
yani uses the newest cached release that already holds every file the run needs
and says which; only a file missing from the cache is then an error.

The release each keyword resolved to is recorded on the results, as
`TransmutationResults.data_releases`: keyword to `release`, `manifest_sha256`,
`format_version`, `offline`, `converters`, `locally_processed` and `library`, so
a result says exactly which data produced it. For a published release
`locally_processed` is `False` and `library` is empty. `converters` is every build of the converters that wrote the
release's files, as its manifest lists them, such as
`"yani-python 0.29.0 (core 1db838a3...)"`. Each entry names the wheel that did
the converting, so data a yamc converter wrote reads `"yamc-python ..."`, and
the version is `unknown` when that wheel's distribution metadata was not
installed. It is empty for a release published before manifests listed them.

Every setting also accepts a **path** to a local converted directory instead of
a keyword:

<!-- doctest: skip -->
```python
yani.transmutation_reactions = "/data/my-network.arrow"
```

`neutron_cross_section_data` additionally accepts a **dict** keyed by nuclide,
to mix sources nuclide by nuclide. The four `transmutation_*` settings take a single
value each, since a network is assembled per subsection rather than per nuclide:

<!-- doctest: skip -->
```python
yani.neutron_cross_section_data = {
    "Fe56": "tendl-2025",
    "Li6": "endf-b8.1",
    "Be9": "/data/Be9.arrow",
}
```

### A whole library in one folder

A folder laid out as a published release, with its `manifest.json` beside the
`neutron/` and `transmutation/` folders it describes, is a release folder, and
its one path serves every setting. Cross sections are read from `neutron/`, and
each transmutation subsection from `transmutation/<subsection>.arrow/`:

<!-- doctest: skip -->
```python
yani.neutron_cross_section_data = "/data/endf-b8.1"
yani.transmutation_decay_data = "/data/endf-b8.1"
yani.transmutation_reactions = "/data/endf-b8.1"
yani.transmutation_fission_yields = "/data/endf-b8.1"
```

A copy of every file a published release's `manifest.json` lists, from
`https://yamc-data.xsplot.com/<keyword>/<release>/`, with the manifest itself
at the top, is such a folder, and so is a release folder in the cache
(`<cache dir>/<keyword>/<release>/`) for the files it holds.

Nothing is downloaded. The manifest is read as a published release's is: a
`format_version` this build does not read is refused with the advice to upgrade,
and each file a load reads is checked against the size the manifest gives it. A
subsection the manifest does not list is refused, naming the subsections the
folder does carry, and so is a file the manifest lists that the folder does not
hold. `TransmutationResults.data_releases` records the folder under its absolute
path, with its `release` and `manifest_sha256`, and `offline` set to `False`.

A load checks sizes only. `yani.verify_library` is the full check: it re-hashes
every file the manifest lists and looks for files it does not list, which is
worth running on a library kept for offline use or copied between machines:

<!-- doctest: skip -->
```python
import pprint

report = yani.verify_library("/data/irdff-ii")
pprint.pprint(report, sort_dicts=False)
```

```text
{'keyword': 'irdff-ii',
 'release': '2026-10-12.1',
 'format_version': 2,
 'manifest_sha256': '547003e2d63c167978be82915986206c581af7373c6f71ca0bb78732a1e182c5',
 'file_count': 3,
 'total_bytes': 6508798,
 'missing': [],
 'changed': [],
 'unlisted': [],
 'ok': True}
```

`ok` is `True` when `missing`, `changed` and `unlisted` are all empty. An
unlisted file fails the check, because results identify the library by the
manifest's sha256 and a file the manifest does not list is data that identity
does not cover. A release folder in the cache holds only the files runs have
downloaded, so the rest are reported missing.

Wherever a reactions or fission-yields source resolves, the folder must hold
`reactions.arrow` or `fission_yields.arrow`: one without it is refused, naming
the folder, rather than read as a network with no reactions or no yields, which
would make every rate through it zero. Set the subsection to `False` to leave
it out. A bare name that is both a keyword and a folder in the working directory
(`endf-b8.1` with a folder `./endf-b8.1` beside it) is refused rather than
guessed at; write `./endf-b8.1` or an absolute path to mean the folder.

To build the data yourself from ENDF tapes rather than download it, the
converters ship on this wheel: see [Making your own data](#making-your-own-data).

## Making your own data

The wheel carries the library builder and the converters, so producing the Arrow
data is a function call rather than a separate toolchain.

### A whole library

`generate_library` builds one of the published libraries from its evaluated
source archives into a folder laid out exactly as the published release, with its
`manifest.json` written last:

<!-- doctest: skip -->
```python
report = yani.generate_library("endf-b8.1", "/data/endf-b8.1")
print(report["release"], report["file_count"], report["njoy"])
```

It downloads each source archive and checks it against the size and sha256 it is
pinned to, unpacks it, converts every evaluation, builds the transmutation chain,
reads every folder back, and writes each particle's `index.json` and then the
manifest. The folder then serves every setting, as a downloaded release does
(see [A whole library in one folder](#a-whole-library-in-one-folder)):

<!-- doctest: skip -->
```python
yani.neutron_cross_section_data = "/data/endf-b8.1"
yani.transmutation_decay_data = "/data/endf-b8.1"
yani.transmutation_reactions = "/data/endf-b8.1"
yani.transmutation_fission_yields = "/data/endf-b8.1"
yani.transmutation_branch_ratios = "/data/endf-b8.1"
```

The libraries are `"endf-b8.1"`, `"jeff-4.0"`, `"jendl-5.0"`, `"tendl-2025"`,
`"tendl-2017"`, `"fendl-3.2d"` and `"irdff-ii"`. Each is built from its own
evaluations only, never with another library's data, so a library publishes the
parts it can build from what it evaluates:

| keyword | transport parts | transmutation parts |
| --- | --- | --- |
| `endf-b8.1`, `jendl-5.0` | neutron, photon, thermal | decay, reactions, fission_yields, branching |
| `jeff-4.0` | neutron, thermal | decay, reactions, fission_yields, branching |
| `tendl-2025`, `tendl-2017` | neutron, photonuclear | reactions, branching |
| `fendl-3.2d` | neutron, photon | reactions, branching |
| `irdff-ii` | none | branching |

`thermal` is the thermal scattering tables (`thermal/c_H_in_H2O.arrow/` and so
on). Like `photon` and `photonuclear`, it is there for transport in yamc: no
yani calculation reads it, so a build for activation alone can leave it out
with `parts`.

A library without decay data of its own (TENDL, FENDL, IRDFF-II) writes each
excited product level of its branching as the evaluation gives it, by level and
excitation energy, and a run names the isomer against the decay data it is given:
TENDL branching run with `transmutation_decay_data = "endf-b8.1"` is matched to
ENDF/B-VIII.1's isomers, and the branching report says so.

The keyword arguments narrow a build, for testing or for the part you need:

<!-- doctest: skip -->
```python
# The transmutation chain alone, which needs no NJOY and takes minutes.
yani.generate_library("endf-b8.1", "/data/endf-b8.1-chain",
                      parts=["decay", "reactions", "fission_yields", "branching"])

# Two nuclides and an element (every Fe evaluation, and Fe photon data), at
# two temperatures.
yani.generate_library("endf-b8.1", "/data/endf-b8.1-few",
                      nuclides=["Li6", "Li7", "Fe"], temperatures=[293.6, 600.0])
```

- `parts`: of `"neutron"`, `"photon"`, `"photonuclear"`, `"thermal"`,
  `"decay"`, `"reactions"`, `"fission_yields"` and `"branching"`; the default is
  every part the library's release carries, and a part it does not carry is
  refused.
- `nuclides`: nuclides and element symbols, and thermal tables by name
  (`"c_H_in_H2O"`) or by the element they bind (`"H"` is every table binding
  hydrogen). A name the library does not have is refused. The chain is always
  built whole.
- `temperatures`: in Kelvin, by default 250, 293.6, 600, 900, 1200 and 2500 K.
  The 0 K grid is always kept.
- `njoy_exec`: the NJOY to run. The default is the `njoy` the
  [njoy2016](https://shimwell.github.io/wheels) wheel installs beside this
  Python, then the one on PATH. Only the neutron and thermal parts need NJOY;
  FENDL needs the IAEA-NDS fork, and the builder refuses any other for it.
- `downloads`: a folder of source archives fetched by hand, searched first.

Running it again resumes: a folder finished for this release is kept and only the
rest are converted, so an interrupted build continues where it stopped. The
release stamp is the date the build started, kept on every resume. A folder holds
one library built by one build of the package: the same version compiled from the
same core commit, with the same dirty flag. A resume by any other build is
refused rather than mixed.

The build is recorded in the release. The manifest's `provenance` carries
`builder` (`"yani-python 0.28.0"`), `builder_commit` (the core commit the wheel
was compiled from, `yani.__commit__`) and `builder_dirty`
(`yani.__commit_dirty__`), both `null` for a wheel built outside a git checkout,
beside `njoy`. Every `version.json` and `provenance.json` carries the same as
one `converter` stamp, such as `"yani-python 0.28.0 (core 1db838a3...)"`, with
`, dirty` after the hash for a modified tree, and the manifest lists each
distinct stamp as `converters`. The single-evaluation converters below write
the same stamp.

A wheel compiled from a tree with uncommitted changes (`yani.__commit_dirty__`
is `True`) writes data that no commit describes, so it builds only a partial
library: one given `parts`, `nuclides` or `temperatures`. A whole library from
such a wheel is refused before anything is written. Commit the changes and
rebuild the wheel to build a whole library. Wheels installed from PyPI are
built from a clean commit and are never refused for this.

No manifest is written unless every evaluation converted (or is one NJOY is known
not to process, declared in the builder with the reason), every folder reads back,
and the chain loads. An evaluation that fails is reported with its reason, which
is also kept in `<path>/.build/failed/`, and running again retries only those.
The returned dict says what the build did:

- `keyword`, `release`, `path`, `parts`;
- `converted` and `reused`: `"<particle>/<Name>"` keys;
- `not_converted`: the declared failures, with NJOY's reason for each;
- `stale_declarations`: declared failures that converted after all;
- `njoy`: the NJOY version used, or `None`;
- `chain_nuclides`, `manifest_sha256`, `file_count` and `total_bytes`.

`verify_library` re-hashes the finished folder against its manifest, as for any
library folder.

A full build is long, most of it NJOY: on a 32-core machine about 50 minutes for
ENDF/B-VIII.1, 9 hours for JENDL-5 and a day or more for each TENDL. The builder
checks the free disk a full build needs before it starts.

### Single evaluations

The converters behind the builder take one evaluation, or one set of tapes, at a
time:

<!-- doctest: skip -->
```python
# Cross sections for the rate collapse, from ENDF (needs NJOY) or from ACE.
yani.convert_neutron_xs(
    input_path="n-026_Fe_056.endf",
    output_dir="out/", njoy_exec="njoy")

# The transmutation network, from decay, fission-yield and neutron evaluations.
yani.convert_transmutation(
    decay_files=decay, fpy_files=fpy, neutron_files=neutron,
    output_path="out/transmutation_tendl-2025.arrow",
)

# Isomeric branching, as an overlay on the network.
yani.convert_branching(
    neutron_files=neutron, decay_files=decay,
    output_path="out/branching.arrow",
)
```

`convert_transmutation`, `convert_branching` and `radionuclide_production` read
every material in each file, so a tape can be passed as it is published:
JEFF-4.0's decay and fission yield tapes, or IRDFF-II's neutron tape of 70
materials, need no splitting first. `convert_neutron_xs` runs NJOY in a hidden
folder inside `output_dir`, removed when the conversion ends, so that disk, not
the system's temporary folder, needs room for NJOY's tapes.

A run that reads a transmutation subsection `convert_transmutation` wrote, given
by its directory rather than a keyword, records it in
`TransmutationResults.data_releases` under the directory's absolute path, with
`locally_processed` set to `True`, `library` and `release` the `library` and
`data_version` the converter recorded, and `manifest_sha256` the sha256 of its
files. Data processed locally, such as the UKDD and UKFY libraries FISPACT-II
pairs with TENDL, is then named in the result as what it is, not passed off as a
published release.

Both network converters record which nuclides carry reactions of their own, as
`parents` on each subsection of the manifest they write. A nuclide that appears
only as somebody else's product is not a parent. That distinction is what lets
an irradiated solve refuse a network scoped to a different material instead of
returning the starting composition unchanged.

### What the converters report about the library

Neither converter corrects its input. What they do is say what the input claims
that cannot be right, so that a number carrying your decay heat can be looked up
before it is believed.

The `decay` subsection's `provenance.json` carries two records:

* `decay_energy_placeholders` names every nuclide whose average decay energies
  are the Q/3 stand-in rather than an evaluated scheme, and the rule used to
  detect them. Listed by name rather than counted, because the question a reader
  asks is whether a nuclide carrying heat in their inventory is one of them.
* `decay_inconsistencies` lists, per kind, the records that cannot all be true:
  flagged unstable with a half-life of zero, branching ratios that do not sum to
  one, and isomeric transitions whose light and electromagnetic averages do not
  add up to the transition's Q. That last one matters most for decay heat, since
  an isomeric transition emits no neutrino and so must pay out its whole Q.
  ENDF/B-VIII.1 books Hf177m1 at 1.52 MeV against a 1.32 MeV transition.

`convert_branching` returns a dict rather than writing one, since a build script
is what decides whether to publish:

* `level_routes` counts how each excited production level was matched to an
  isomeric state: by energy, by energy within a tenth, by level index, as the
  only candidate, or not at all. Given no decay files, nothing is matched: each
  level is written as the evaluation gives it (route `as_evaluated`) and a run
  matches it against the decay data it is given. A level matched by index is right only while
  two level schemes happen to agree, so a rebuild against another decay library
  moving levels out of `energy` is the regression the plain counts hide.
* `flagged_levels` is one line per level worth reading before publishing.
* `partial_sum_mismatches` is one line per reaction whose MF=10 partial cross
  sections do not reconstruct its MF=3 total, or whose MF=9 yields do not sum to
  one, by more than 2% of the nonelastic cross section. The rates are shared out
  in the partials' proportions either way, so this does not move a yani answer;
  it moves a code that folds the partials as they stand, and it points at the
  evaluation. TENDL-2017's Ir191 (n,2n) partials sum to 95% of MF=3 at 14 MeV
  because the file lists two of Ir190's three states.

### Filling placeholder decay energies

Where the decay library has no evaluated scheme for a nuclide and another
library does, the second one's average energies can be substituted:

<!-- doctest: skip -->
```python
yani.convert_transmutation(
    decay_files=decay, fpy_files=fpy, neutron_files=neutron,
    output_path="out/transmutation_endf-b8.1.arrow",
    library="endf-b8.1",
    decay_fill_files=jendl_decay, decay_fill_library="jendl-5.0",
)
```

Only a placeholder is replaced, only from a record that is not itself a
placeholder, and only when the two half-lives agree within 25%, which is what
keeps a differently-assigned isomer out. Only the mean energy and its
uncertainty move: half-lives, decay modes and spectra stay as the decay library
has them, so the network's topology is untouched. Every substitution is written
to `provenance.json` as `decay_energy_fill`, with the value before and after, and
each nuclide's `decay_energy_source` reads `evaluation`, `placeholder` or
`filled:<library>`.

Without `decay_fill_files` nothing is substituted and the numbers are
byte-identical to what the decay library published, so the placeholder list is
worth reading even when you do not act on it.

`convert_neutron_transport` and `convert_photon` are present too, since all the
wheels share one bindings crate, but they write sections only transport reads.
`convert_photon` carries the Compton profile, density effect and bremsstrahlung
tabulations it adds to every element (they are in no evaluation), so it takes the
photo-atomic file and an output folder, and the same call gives the same data
from yamc:

<!-- doctest: skip -->
```python
yani.convert_photon("photoat-026_Fe_000.endf", "out/photon",
                    relaxation_path="atom-026_Fe_000.endf")
```

`convert_photonuclear` keeps a photonuclear evaluation (TENDL's gamma
sublibrary) as evaluated, its MF=3 cross sections on their own grids and every
original section, without NJOY.

NJOY is needed for the ENDF route and nothing avoids it: an ENDF evaluation
describes the resonance region with resonance parameters rather than pointwise
cross sections, so something has to reconstruct and Doppler broaden it.
`source_format="ace"` reads an already-processed table and needs no NJOY, at the
cost of one temperature and no heating sections.

## Nuclear-data uncertainty

Pass a `DataUncertainty` and every nuclide density comes back with a standard
deviation beside it:

<!-- doctest: skip -->
```python
results = material.transmute(
    schedule=schedule,
    data_uncertainty=yani.DataUncertainty(seed=42),
)

mid = material.id
mean = results.get_nuclide_density(material_id=mid, nuclide="Mn56", step=1)
sigma = results.get_nuclide_uncertainty(material_id=mid, nuclide="Mn56", step=1)
```

The activation cross sections are resampled from their ENDF MF=33 covariance,
folded against your own spectrum, and the schedule is re-solved per sample. That
is exact to all orders in the matrix exponential; the solver is untouched and
only its input changes. Omitting the argument costs nothing at all: no
covariance is read, nothing is folded, and the inventories are bit-identical.

Every draw is a pure function of the seed, the sample and what is drawn: a
nuclide for the cross sections (the linked group, where a cross-material block
joins two evaluations), half-lives, decay branching, decay energies, decay
photons and isomeric splits, an evaluation for the fission yields, a spectrum
for the flux, and a material's tallied rates, jointly, for the statistical
error. So a seed reproduces a run regardless of sample count or iteration
order. Leave `samples` unset and the driver adds samples, 64 at a time, until
every tracked standard deviation is known to `convergence`, the standard error
of a sigma relative to that sigma (0.05 unless set). It is judged at every step
on each nuclide density within 1e-6 of that step's largest and on the total
activity, decay heat and decay photon line rate, leaving out any sigma under
0.01% of its value. At least 128 samples run whatever the target and at most
1024; a run the cap stops sets `hit_cap`, and `get_data_uncertainty_info` lists
every sigma that missed, worst first, under `unconverged`. A fixed `samples`
runs exactly that many and lists what missed all the same, and a transport run
fixes its count before the transport, 256 at the default target. The photon
spectrum line by line and the contact dose are not judged, since their dose
quantity and build-up are chosen when they are read; every `Estimate` carries
its own `std_dev_standard_error`.

Nine sources can be perturbed, and `DataUncertainty.available_sources()` names
them:

- `cross_sections`: the activation cross sections, from their ENDF MF=33
  covariance folded against your spectrum. Where the library carries the
  evaluation's resonance parameters (MF=2 and MF=32, in
  `resonance_parameters.arrow`), each replica draws the parameters themselves
  from their covariance and rebuilds the resonance range's cross sections from
  the draw, so the resonance-range uncertainty is exact in the parameters
  rather than first order. A nuclide whose parameters cannot be sampled keeps
  the first-order MF=32 rows of `covariance.arrow`, and the report says which
  and why.
- `flux_spectrum`: the spectrum itself, from the error you hand the pulse
  (below). A spectrum with no stated error contributes nothing.
- `half_life`: the half-life of every unstable nuclide the material can reach,
  from the decay data's own sigma on it. A replica's half-lives are used in its
  solve and in the activity, decay heat and dose evaluated from it, so a
  saturated activity stays as insensitive to its own half-life as it physically
  is.
- `decay_branching`: the decay branching ratios of every reachable parent with
  two or more modes and a stated sigma, however many modes it has. Each draw
  keeps the parent's total and every ratio between zero and it, and each mode
  takes its evaluated ratio as its mean and its stated sigma as its spread, so
  a small branch with a large relative sigma is drawn skewed rather than held.
  A mode is held at its ratio only when its stated sigma is one no ratio with
  that mean can have, or it states none and is not the complement of the rest;
  the report names every parent short of its stated sigmas, and why.
- `decay_energy`: each nuclide's mean decay energy, from the sigma on each beta,
  gamma and alpha component that states one, or on the total when no component
  states a sigma. A decay energy never enters the solve, so this moves decay
  heat and nothing else.
- `decay_photon_lines`: each decay photon spectrum's normalisation, one draw
  per spectrum shared by all its lines, and each line's own intensity and
  energy, from the decay data's sigmas on them. No photon enters the solve, so
  this moves the photon spectrum, the contact dose and, through the gamma decay
  energy, which follows each replica's drawn lines in place of a `decay_energy`
  draw, the decay heat. Each of these has a range rather than one spread,
  `Estimate.std_dev` to `Estimate.std_dev_correlated`, because the decay data
  do not state how a nuclide's photon intensities are correlated.
- `fission_yield`: the independent yields of each parent the run gives a
  fission rate, from the DY the evaluation states on each, drawn independently
  since no evaluation states a correlation. A parent with yields that never
  fissions in the run (an actinide reached by capture from a structural
  material, say) is not drawn and is not counted as a gap; the transport-coupled
  path still lists every reachable parent with yields. It needs the evaluated yields, which the published fission
  yield libraries carry; a chain without them lists every parent under
  `no_fission_yield_uncertainty`.
- `isomer_branching`: the isomeric split of every reachable reaction whose
  evaluation states the MF=40 covariance of its final states' production (the
  branching subsection's `branching_covariance.arrow`). Each state's production
  is drawn on the covariance's own grid, folded against the same spectrum as
  the nominal split, and the split remade from the draw. Where the list names
  its ground state only the shares move and the reaction total stays the
  cross sections' own, so the total's uncertainty is not counted twice. An
  isomer listed alone, and every `(n,n')` isomer, is made at its drawn
  production, so its MF=40 sigma replaces the MF=33 sigma of its reaction
  that it carries under `cross_sections` alone, and can be the smaller. It
  applies on `Material.transmute` and on a transport run, where the tally
  scores each state's production on the covariance's intervals.
- `statistical`: the Monte Carlo error of transport-tallied reaction rates, from
  their per-history covariance, on a yamc `Model.simulate_transmutation` run
  with the independent method. A spectrum run's rates are a deterministic
  collapse, so on `Material.transmute` it has nothing to act on and is left out
  of the report's `sources`, as `flux_spectrum` is on a transport run.

`sources=` restricts a run to some of them, which is how a contribution is
measured. Naming a source this build cannot perturb raises. The decay branching
ratios that source does not sample, the isomeric splits of reactions with no
MF=40 covariance, the material composition and the
other inputs listed under
[What is not propagated](method.md#what-is-not-propagated) stay at their
evaluated or nominal values.

A spectrum that came from a Monte Carlo run carries a statistical error of its
own. Hand it to the pulse, per bin, in the same order and units as the histogram
values, and it is sampled alongside the cross sections:

<!-- doctest: skip -->
```python
flux_sigma = [2e10, 8e10, 1e12]         # one per group, as the histogram has

pulse = yani.Pulse(
    rate=sum(multigroup_flux),
    duration=(1, "h"),
    source=spectrum,
    flux_std_dev=flux_sigma,
)
```

Omitting it is the common case, since a spectrum taken from a published
reference set carries no stated error. A run then reports the omission in
`spectra_without_flux_sigma` of `get_data_uncertainty_info` rather than letting
the flux read as known exactly.

A per-bin sigma treats the bins as independent, and a tally's bins are not: the
same histories score them, so they move together, and a per-bin sigma
understates the error on any rate that sums over a band of them. Where the
correlations are known, give the pulse the full covariance of the histogram
values as `flux_covariance` instead of `flux_std_dev`. It has to be symmetric
and positive semi-definite, and is checked.

Because a zero sigma could mean either "well known" or "nothing published", the
two are separated in `get_data_uncertainty_info` for a nuclide with no usable
MF=33 block for any channel the chain drives (none on the tape, or only
underivable NC blocks, blocks to an evaluation the run does not hold, lumped
blocks stated in barns, partial-level or unsupported ones):

<!-- doctest: skip -->
```python
info = results.get_data_uncertainty_info(material_id=mid)
if info is not None:               # None unless data_uncertainty was passed
    info["perturbed"]              # had usable MF=33 covariance
    info["no_covariance_data"]     # no usable MF=33 block for a driven channel
    info["rate_fraction_covered"]  # share of each rate with a nonzero stated variance
    info["rate_fraction_covered_total"]  # ... and over the run, weighted by production
    info["partials_above_rate"]    # channels whose sigma is overstated
    info["partials_below_rate"]    # channels whose sigma is understated
    info["half_lives_perturbed"]   # unstable nuclides whose half-life was sampled
    info["no_half_life_uncertainty"]     # ... and those whose data states no sigma
    info["decay_energies_perturbed"]     # nuclides whose decay energy was sampled
    info["no_decay_energy_uncertainty"]  # decay energies held for want of a sigma
    info["decay_branchings_perturbed"]   # multi-mode parents whose split was drawn
    info["half_life_uncertainty_not_carried"]     # a stated sigma no draw can
    info["decay_energy_uncertainty_not_carried"]  # carry, held; see the method page
    info["decay_photon_lines_perturbed"]  # nuclides whose photon data was drawn
    info["no_decay_photon_line_uncertainty"]  # photon data held for want of a sigma
    info["fission_yields_perturbed"]      # fissioning parents whose yields were drawn
    info["fission_yields_mapping_mismatch"]  # yields that did not map back, held
    info["isomer_branchings_perturbed"]   # reactions whose isomeric split was drawn
    info["no_isomer_branching_uncertainty"]  # splits held, no MF=40 on any state
    info["isomer_production_sigma"]       # MF=40's folded sigma per product
    info["resonance_parameters"]   # per nuclide: parameters sampled, or first-order rows and why
    info["rates_floored"]          # rate draws floored at zero, see the method page;
                                   # also statistical_floored
    info["skipped_cross_material"] # {nuclide: blocks to an evaluation not in the run}
    info["skipped_other_file"]     # {nuclide: blocks whose partner is not a cross section}
    info["mirrored_disagree"]      # pairs stored both ways whose copies differ
    info["not_perturbed"]          # every input held at nominal, see below
    info["sources"]                # the sources that applied to this run
    info["has_gaps"]               # True if something was left out or is inconsistent
```

`not_perturbed` names every input held at nominal on every run, from any
resonance-parameter covariance (MF=32) neither sampled nor in `covariance.arrow`
and the MT=5 product multiplicities to the photon spectrum covariance, dose
coefficients and the material composition, plus the run-dependent entries: a
source the run switched off, spectra without a flux sigma, tallied-rate
statistics a transport run did not draw, the self-shielding correction, and the
per-branch decay emission of a parent whose branching was drawn.
[What is not propagated](method.md#what-is-not-propagated)
explains each one, and names the few held inputs with no entry of their own,
among them MF=33 blocks on partial levels (MT=600-849, 875-891), which are
neither listed nor counted unless an LTY=0 NC block names them, because the
chain drives no rate for them. `skipped_cross_material`, `skipped_nc` and
`lumped_covariance_not_assignable` count the cross-material blocks naming an
evaluation the run does not hold, the underivable NC blocks and the lumped
blocks stated in barns, none of which is used. A multi-mode parent
whose branching was held whole is named in `no_decay_branching_uncertainty` or
`decay_branchings_too_wide`, and one that was drawn short of its stated sigmas
in `decay_branchings_partly_held` (some modes held) or
`decay_branchings_sigma_adjusted` (a mode drawn more than 0.1% from its stated
sigma, because the sum rule or non-negative ratios cannot carry it). MF=40
blocks that could not be used are counted per parent in
`isomer_branching_blocks_not_used`. `has_gaps` looks only at the sources the
run perturbs, and is True when one of them met a nuclide with no usable MF=33
block, a skipped or unassignable MF=33 block, a block whose layout is
unsupported or malformed, a channel whose partial rates add up to more, or
less, than its rate (`partials_above_rate`, `partials_below_rate`, below), a
derived channel resting on an unstated covariance between opposing terms, a
covariance repaired past round-off on a channel a draw can move (inside the
populated bound or outside it), a cross-material covariance drawn at the
closest admissible one (`cross_material_adjusted`), an evaluation its own
library warns about (`covariance_warnings`), a spectrum with no flux sigma or a
rate under one that the flux draw cannot move, a reachable unstable nuclide
with no stated half-life sigma, or with a decay energy or decay photon data but
no stated sigma on it, a half-life, decay-energy or photon sigma no draw can
carry, a fissioning parent with no yield sigma, a yield sigma no draw can carry
or yields that do not map back, a multi-mode parent whose branching was held or
drawn short of its stated sigmas, or an isomeric split with no MF=40 to draw,
an MF=40 block not used or an MF=40 matrix repaired. Partial-level
blocks set it only through `no_covariance_data`, when they are all a nuclide
has.

Read `rate_fraction_covered_total` before any sigma above it. A count of
nuclides with MF=33 measures how much covariance exists. This measures how much
of it lands on the reactions the run drove, weighted by rate and by the parent's
own density.

Tungsten is a case where the coverage rests on a lump. ENDF/B-VIII.1, JEFF-4.0
and FENDL-3.2d state the `(n,2n)` that makes 98% of a tungsten foil's decay heat
only lumped with `(n,2np)`, as the uncertainty of their sum (MT 852). Each
component takes the lump's relative perturbation, so the two move together and
their sum moves as the lump states. On ENDF/B-VIII.1 and the FNS spectrum a
tungsten foil's `rate_fraction_covered_total` is 1, and
`sources=["cross_sections"]` gives its decay heat a spread of about 3% at the
first measured cooling time, nearly all of it W185m's.

Covered means the evaluation states a nonzero variance there, not that a
covariance grid spans it. A grid can run across the whole range with a variance
of zero on some intervals, and rate from those counts as uncovered, the same as
rate from outside the grid. ENDF/B-VIII.1 W186 `(n,gamma)` is the case: its
MF=33 block states zero from 1e-5 eV to 10 keV, where nearly all of a capture
rate is, because the evaluation keeps that uncertainty in resonance-parameter
covariance (MF=32). The published libraries carry the resonance parameters,
which are sampled and cover that range, so on the FNS spectrum W186 `(n,gamma)`
reads 1; on data with MF=33 alone it reads 0.07, and W186 stays under
`perturbed` for the intervals above 10 keV.

The share is of the rate the fold divides by: the rate from energies with a
nonzero stated variance, over the rate across the flux range, both with the
dilute cross section on a dilute run and the shielded one on a self-shielded
run. Shielding depresses the resonance range, which is where capture blocks
often state zero, so a shielded run's shares can sit well away from the dilute
ones. The total weights each channel's share by the production the run
actually drove, so it is the share of that production coming from covered
energies. A transport run is the exception: its shares are of the dilute rate
over the tally spectrum, the covered share of the tallied production is not
computed, and `rate_fraction_covered_total` is `None` there rather than a
figure weighting that production by dilute shares. The per-channel shares are
still given, and the relative sigma is diluted by a different amount than they
say.

`partials_above_rate` lists, with their ratio, the channels whose partial rates
the covariance was weighted with add up to more than the rate it was divided
by. The two were then computed different ways, and the relative sigma is
overstated. A grafted `(n,n')`, whose rate is the metastables' MF=10
production while its partials are MT 4's, is one cause. The `1/E` within-group
weight is another: there the part of a group a covariance edge cuts
off is weighted by its share of the group's energy width, so the parts need not
add up to the group's rate even on a dilute run, and by a lot: Fe56 `(n,p)` on a
three-group spectrum whose fast group holds its 4.3 MeV covariance edge sums to
about ten times its rate. A tallied rate on a transport run is a third. The
coverage share is of the fold's own rate, not of the listed rate, and a channel
in either map makes `rate_fraction_covered_total` `None`.

`partials_below_rate` is the same check the other way. A covariance grid that
spans the whole flux range leaves no rate outside it, so its partial rates must
add up to the rate, and a shortfall understates the sigma. The `1/E` weight
gives one for a reaction falling with energy when a covariance edge cuts a
group. A grid that stops short of the flux range cannot be checked this way,
since rate from outside it rightly leaves its partials short. Both maps count
towards `has_gaps`.

Three maps say which blocks were read but not folded. `skipped_cross_material`
counts, per nuclide, the blocks correlating one of its reactions with a
reaction of another evaluation that were not applied: the evaluation they name
is not in the run, or one of the two reactions has no relative covariance to
correlate. Where the run holds both evaluations the block is applied, and the
nuclides it links are drawn jointly, one draw per replica shared by every
material, counted in `applied_cross_material`; where the stated covariance is
not admissible beside each evaluation's own, the closest admissible one is
drawn and `cross_material_adjusted` says by how much (ENDF/B-VIII.1 U235 and
U238 fission read about 0.31), which counts towards `has_gaps`. A block naming
its partner by the evaluation's own MAT is not one of them; ENDF-102 allows
that spelling alongside `MAT1 = 0`, and both are folded. `skipped_other_file` counts the
blocks whose partner is not a cross section (`XMF1` other than 0 or 3, or a
final state). Both count only blocks on a reaction the run drives, once per
nuclide however many spectra the run has. `mirrored_disagree` is keyed
`"Nuclide (n,a) (n,b)"`: a pair stored in both orientations is folded once,
from the lower MT's section, and the other copy is checked against its
transpose. A difference above 1e-5 of the pair's largest entry is listed with
its size. JEFF-4.0 Be9 stores 130 pairs both ways, and they agree exactly. All
three count towards `has_gaps`.

Both weights are needed. Rate alone, without the density of the parent each rate
belongs to, counts a channel on a trace isotope the same as one on the bulk: on
an iron foil that is 32% against 96%.

For a decay-only schedule there is no production to weight by, so the value is
`None`, not zero. It is `None` on a transport run too, as above, and when
either partials map lists a channel.

Activity, decay heat and contact dose come back with a band from the same
ensemble, each as an `Estimate` holding the unperturbed value and the ensemble's
spread on it:

<!-- doctest: skip -->
```python
activity = results.get_activity_uncertainty(material_id=mid, step=1)
activity.nominal                    # the unperturbed run, always present
activity.mean, activity.std_dev     # None below two replicas
activity.relative_std_dev

results.get_decay_heat_uncertainty(material_id=mid, step=1)
results.get_contact_dose_uncertainty(material_id=mid, step=1, dose_quantity="effective")
results.get_activity_uncertainty(material_id=mid, step=1, by_nuclide=True)   # dict[str, Estimate]
```

Each is evaluated once per replica and summed within that replica, so the
correlations between nuclides survive. Building the same number out of the
per-nuclide sigmas is wrong in a specific direction: evaluating from the mean
inventory gives no spread at all, and adding sigmas in quadrature double-counts
a variance that partly cancels, since every Mn56 atom in an irradiated iron foil
came out of an Fe56 atom. `mean` and `std_dev` are `None` below two replicas,
because a spread over one sample is unmeasured rather than zero.

With `displacement_damage=True`, the dpa and damage energy come with a band
too, from `get_dpa_uncertainty` and `get_damage_energy_uncertainty`: see
[Displacement damage](#displacement-damage).

The decay photon spectrum comes back line by line, as `LineEstimate`s over the
union of the lines the unperturbed run and every replica emit:

<!-- doctest: skip -->
```python
lines = results.get_decay_photon_spectrum_uncertainty(material_id=mid, step=1)
for line in lines or []:
    line.energy, line.nominal, line.std_dev
    line.energy_mean, line.energy_std_dev  # the energy each replica drew
    line.emitting              # replicas that emitted it at all
```

A line a replica does not emit counts as a zero in it, which is the only rule
under which two lines' spreads are taken over the same sample. `emitting` keeps
that zero-fill visible, so a dim line and an intermittent one stay
distinguishable.

Every one of these bands is evaluated with each replica's own half-lives and,
for decay heat, its own decay energies. With `decay_photon_lines` on, each
replica also draws every line's emission per decay and energy, so a line's band
is its emitters' activity and its own intensity together, and lines are matched
across replicas on their nominal `energy`. Contact dose holds its attenuation
coefficients, response and build-up factor the same in every replica.

`get_uncertainty_inventories(material_id=mid, step=step)` is still there for a
quantity these four do not cover, and returns every replica's full inventory to
take the spread over yourself.

### Where an uncertainty comes from

`DataUncertainty(attribution=True)` also says where each nuclide's sigma comes
from. The run then resamples each source alone as well as all of them together,
so it costs more than one without:

<!-- doctest: skip -->
```python
results = material.transmute(
    schedule=schedule,
    data_uncertainty=yani.DataUncertainty(seed=42, attribution=True),
)
breakdown = results.get_uncertainty_breakdown(
    material_id=mid, nuclide="Mn56", step=1
)
if breakdown is not None:          # None unless attribution was asked for
    breakdown["variance"]          # the total, the square of the sigma above
    breakdown["by_source"]         # each source alone; these sum to the total
    breakdown["unattributed"]      # what the sum leaves, small when it holds
    breakdown["contributors"]      # (source, nuclide, reaction, variance), largest first
```

The sources are independent, so `by_source` is how much of the variance is
cross sections, half-lives, flux and so on, and `unattributed` is interaction
and sampling noise. `contributors` is first order and says which evaluation to
look at: within the cross sections a nuclide's whole evaluation has a
`reaction` of `None` and each channel names its own, and a half-life or a
parent's decay branching (its whole split) has `None`. The total stays the resampled one.

On a yamc transport run with the `statistical` source,
`get_reaction_rate_uncertainty(material_id=mid, step=step)` lists each tallied
rate with its standard deviation, as `(nuclide, reaction, target, rate,
std_dev)` in 1/s per atom. The rates were scored by the same histories and are
correlated; the inventory sigmas carry those correlations, and these standard
deviations alone do not.

### Reading ensemble diagnostics

The density sigma, its sampling error and the replica noise are different
quantities. After a run with `data_uncertainty`, inspect them explicitly:

```python
sigma_evolution = results.get_nuclide_uncertainty_evolution(
  material_id=material.id, nuclide="Co60",
)
sigma_error = results.get_nuclide_uncertainty_standard_error(
  material_id=material.id, nuclide="Co60", step=1,
)
replica_noise = results.get_nuclide_uncertainty_noise(
  material_id=material.id, nuclide="Co60", step=1,
)
activity_estimate = results.get_activity_uncertainty(material_id=material.id, step=1)
if activity_estimate is not None:
  print(activity_estimate.std_dev, activity_estimate.std_dev_range)
  print(activity_estimate.replica_spread, activity_estimate.noise_dominated)
```

The evolution includes the initial state, matching `get_nuclide_evolution()`.
Unknown IDs or unavailable uncertainty results return `None`; they do not
establish zero uncertainty. The noise tuple holds the replica spread and the
Monte Carlo noise removed from it. `std_dev_range` describes how well the
estimated sigma is settled, not a confidence interval on the physical value.

## Limits worth knowing

- **Without self-shielding, a step is exact at any length, for a fixed
  spectrum.** The burnup matrix is built from decay constants and `sigma*phi`
  rates and does not depend on the composition, so it is constant over a step
  and the matrix exponential is solved exactly. One year in a single step and
  one year in 365 agree to round-off. Shortening steps to chase accuracy buys
  nothing here.

  **With self-shielding, the rates follow the composition step by step.** The
  flux depression depends on what the material is made of, so each step is
  collapsed at the composition it starts from and its rates are held over the
  step. Shorter steps then follow the burnup more closely, converging at first
  order, and each shielded step costs one collapse.

  What a long step **does** miss is the spectrum changing as the composition does,
  which `transmute()` cannot see because the flux is your input rather than
  something it solves for. If the field would harden or soften appreciably over
  the campaign, split the schedule and give each pulse its own spectrum. That is
  a statement about the physics you are feeding it, not about the solver.
- `data_uncertainty` covers the activation cross sections (with their
  resonance parameters where the library carries them), the half-lives, the
  decay branching ratios, the decay energies, the decay photon lines
  and continuum normalisations, the fission yields, the isomeric branching
  where an evaluation states MF=40, and the flux spectrum
  when a pulse carries `flux_std_dev` or `flux_covariance`. Without one the
  flux is taken as exact, since nothing is transported here. The decay
  branching ratios the draw holds, isomeric splits with no MF=40,
  resonance-parameter covariance (MF=32) neither sampled nor in
  `covariance.arrow`, cross-material blocks to an evaluation the run does not
  hold, underivable NC, barn-stated lumped and partial-level MF=33 blocks, the self-shielding correction, the photon
  spectrum covariance, the dose constants and the material composition are held at
  their evaluated or nominal values;
  [What is not propagated](method.md#what-is-not-propagated) says how much each
  can matter.
- Nothing is self-shielded unless you give a shape or a chord, so a dilute run
  of a resonance absorber reads high. `would_shield` in `get_self_shielding_info`
  flags which nuclides on the run that skipped it, and how strongly their own
  resonances could bite. It is a screen, not a bound: only a shielded run says
  by how much.
- The network is only as complete as the chain you configure. A product whose
  parent reaction is missing from `transmutation_reactions` never appears, and
  neither does any route through it.
- A route table covers one step, to the depth you ask for. Each route is weighted
  by what its own reactions drove over that step, which is right while the
  intermediates barely burn. A route needing more reactions than `reaction_depth`
  or more decays than `decay_depth` does not appear, and the shares are shares of
  the routes that did. Raise the depths before reading an absence as impossible.
- Nuclides many decades below the largest inventory carry no significant figures.
  Treat a deep trace as a bound, not a number.
