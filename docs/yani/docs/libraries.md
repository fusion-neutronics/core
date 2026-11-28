# Nuclear data libraries

Six libraries are available by keyword. Point a setting at one and the sections it
needs are downloaded on first use from the library's current published release,
verified against that release's manifest, and cached on disk (`yani.cache_dir()`
says where); only the sections activation reads are fetched, never a whole
library. Releases, the cache layout and offline use are in
[Where the data comes from](usage.md#where-the-data-comes-from).

<!-- doctest: skip -->
```python
import yani

yani.neutron_cross_section_data = "tendl-2025"
yani.transmutation_reactions = "tendl-2025"
yani.transmutation_decay_data = "endf-b8.1"
yani.transmutation_fission_yields = "endf-b8.1"
```

| keyword | what it is | cite |
| --- | --- | --- |
| [`endf-b8.1`][cite-endf-b8.1] | ENDF/B-VIII.1. | [Nobre et al. 2026](#cite-endf-b8-1) |
| [`jeff-4.0`][cite-jeff-4.0] | JEFF-4.0. | [JEFF-4.0 data sets](#cite-jeff-4-0) |
| [`jendl-5.0`][cite-jendl-5.0] | JENDL-5. | [Iwamoto et al. 2023](#cite-jendl-5-0) |
| [`tendl-2025`][cite-tendl-2025] | TENDL-2025. The widest isotope coverage of the six, but its lithium [breeds no tritium](#tendl-2025s-lithium-breeds-no-tritium). | [Koning and Rochman 2012; Koning et al. 2019](#cite-tendl) |
| [`tendl-2017`][cite-tendl-2017] | TENDL-2017, for comparing a result against the older release. | [Koning and Rochman 2012; Koning et al. 2019](#cite-tendl) |
| [`fendl-3.2d`][cite-fendl-3.2d] | FENDL-3.2d. Cross sections, plus `reactions` and `branching` subsections. | [Schnabel et al. 2024](#cite-fendl-3-2d) |

The evaluators who made these libraries ask to be cited wherever their data is
used: see [Cite the nuclear data you use](#cite-the-nuclear-data-you-use).

Four more keywords each supply one transmutation subsection and no cross
sections: [`irdff-ii`][cite-irdff-ii] (isomeric branching),
[`ukdd-2012`][cite-ukdd-2012] and [`ukdd-2020`][cite-ukdd-2020] (decay data)
and [`ukfy-4.2`][cite-ukfy-4.2] (fission yields). See
[UKAEA decay data and UKFY fission yields](#ukaea-decay-data-and-ukfy-fission-yields).

Every setting also takes a path to a converted directory instead of a keyword,
including a folder that holds a whole library laid out as a release, which
serves every setting at once (see
[A whole library in one folder](usage.md#a-whole-library-in-one-folder)), and
`neutron_cross_section_data` takes a dict keyed by nuclide. Which setting does
what, and what each one does when left alone, is in
[Nuclear data settings](usage.md#nuclear-data-settings). Only
`transmutation_branch_ratios` has a default; the others are refused when a
calculation reads them unset.

--8<-- "cite-nuclear-data.md"

## Asking yani what a library holds

`yani.available_nuclear_data_libraries()` answers from each library's current
release: a dict keyed by keyword, giving the nuclides with neutron cross
sections, which of them carry each kind of covariance, the thermal scattering
tables, the elements with photo-atomic data, the targets with photonuclear
data, which transmutation subsections it publishes and the upstream files it
was built from. It reads each release's manifest, so no data is downloaded,
and the manifests are cached after the first call.

<!-- doctest: skip -->
```python
libraries = yani.available_nuclear_data_libraries()
libraries["tendl-2025"]["decay"]              # False: take decay from another library
"Fe56" in libraries["tendl-2025"]["photonuclear"]
libraries["irdff-ii"]["isomeric_branching"]   # True
sorted(libraries["endf-b8.1"]["neutron_covariances"])
# ['angular_distribution', 'cross_section', 'fission_spectrum', 'nubar', 'resonance_parameters']
```

Every list is complete, so an empty one means the library has none of that.
A library that could not be looked up (no network and nothing cached) has
`None` throughout and the reason under `"error"`. The tables below are the same
information, summarised.

## What each library publishes

A calculation needs cross sections and four transmutation subsections, and each
is set on its own, so a network can take each part from wherever publishes it.
Hover a column heading for what it holds; the list under the table says the same.

!!! note "Cross sections or reactions?"
    There is one set of cross sections, and it is the first column. The
    `reactions` subsection is not a second set: it holds no cross sections, only
    the network of which reaction on which nuclide makes which product. The
    cross sections decide **how fast** each reaction runs and the network decides
    **which** reactions and products exist, so a calculation needs both, and the
    two can come from different libraries. More in
    [The two that both say reactions](usage.md#the-two-that-both-say-reactions).

| library | <abbr title="neutron_cross_section_data: cross sections sigma(E) for each nuclide, folded with the spectrum into reaction rates. Decides how fast each reaction runs. The only cross sections a calculation uses.">cross sections</abbr> | <abbr title="transmutation_decay_data: half-lives, decay modes, decay energies and decay photon lines. Decides how the inventory decays and what it emits.">`decay`</abbr> | <abbr title="transmutation_reactions: the network of which reaction on which nuclide makes which product, with its Q. Holds no cross sections. Decides which products can be made at all.">`reactions`</abbr> | <abbr title="transmutation_fission_yields: which fission products a fission makes, and how many. Only read when something fissions.">`fission_yields`</abbr> | <abbr title="transmutation_branch_ratios: how a reaction's product splits between ground and metastable states, against incident energy.">`branching`</abbr> |
| --- | --- | :---: | :---: | :---: | :---: |
| [`endf-b8.1`][cite-endf-b8.1] | neutron + photoatomic | yes | yes | yes | yes |
| [`jeff-4.0`][cite-jeff-4.0] | neutron | yes | yes | yes | yes |
| [`jendl-5.0`][cite-jendl-5.0] | neutron + photoatomic | yes | yes | yes | yes |
| [`fendl-3.2d`][cite-fendl-3.2d] | neutron + photoatomic | -- | yes | -- | yes |
| [`tendl-2025`][cite-tendl-2025] | neutron + photonuclear (next release, data only) | -- | yes | -- | yes |
| [`tendl-2017`][cite-tendl-2017] | neutron + photonuclear (next release, data only) | -- | yes | -- | yes |

- **cross sections**, set by `neutron_cross_section_data`: cross sections
  against energy for each nuclide, folded with the spectrum into reaction rates.
  They decide how fast each reaction runs.
- **`decay`**, set by `transmutation_decay_data`: half-lives, decay modes, decay
  energies and decay photon lines. They decide how the inventory decays and what
  it emits.
- **`reactions`**, set by `transmutation_reactions`: which reaction on which
  nuclide makes which product, and its Q. It decides which products can be made
  at all, and holds no cross sections.
- **`fission_yields`**, set by `transmutation_fission_yields`: which fission
  products a fission makes, and how many.
- **`branching`**, set by `transmutation_branch_ratios`: how a reaction's
  product splits between its ground and metastable states.

[TENDL][cite-tendl] has neutron and photonuclear evaluations, but no decay or fission-yield
sublibrary, so those two must come from one of the three that do. It is still
what you want for `reactions`, because it covers far more parent nuclides, and
that is what decides whether an activation product appears in your inventory at
all. `endf-b8.1`, `jeff-4.0` and `jendl-5.0` each supply a complete network from
one library.

[FENDL-3.2d][cite-fendl-3.2d] publishes the same two subsections as TENDL, both built against
[ENDF/B-VIII.1][cite-endf-b8.1] decay data, so it too takes its `decay` and `fission_yields` from
elsewhere. Its `branching` subsection is the main reason to use it: 54 of its
evaluations carry isomer production (MF=9 or MF=10), 15 of them with MF=40
covariances.

Pointing a subsection at a library that does not publish it fails immediately,
naming what that library does have, rather than solving to an inventory that is
quietly missing a route.

The `cross sections` column lists different kinds of evaluated data. YANI's
activation rates use neutron cross sections. Photoatomic data matters when a
decay photon spectrum is transported in yamc; it is not the photonuclear data
in TENDL. JEFF's photonuclear sublibrary is not processed as photoatomic data,
and FENDL's photoatomic data carries no atomic relaxation.

### UKAEA decay data and UKFY fission yields

Three keywords supply one transmutation subsection each, and no cross sections:
the decay and fission-yield libraries UKAEA processes for FISPACT-II, from
[UKAEA's FISPACT-II nuclear data repository](https://git.oecd-nea.org/fispact/nuclear_data).

| keyword | what it is | subsection | cite |
| --- | --- | --- | --- |
| [`ukdd-2012`][cite-ukdd-2012] | UKAEA decay_2012: 3,875 nuclides. | `decay` | [UKAEA (CC BY 4.0)](#cite-ukdd) |
| [`ukdd-2020`][cite-ukdd-2020] | UKAEA decay_2020, its update from newer evaluations (JEFF-3.3, JENDL/DDF-2015, ENDF/B-VIII.0 among them): 4,035 nuclides. | `decay` | [UKAEA (CC BY 4.0); Katakura and Minato 2016](#cite-ukdd) |
| [`ukfy-4.2`][cite-ukfy-4.2] | UKFY-4.2, neutron-induced fission yields evaluated by R.W. Mills (NNL): 223 parents, each at 37 incident energies from 1e-5 eV to 150 MeV. | `fission_yields` | [UKAEA (CC BY 4.0); evaluator R.W. Mills](#cite-ukfy-4-2) |

[TENDL][cite-tendl] has no decay or fission-yield evaluations, so a TENDL
network takes them from another library, and these are the ones FISPACT-II
reads. A calculation set up with them reads the same data as a FISPACT-II
calculation on the same TENDL release, which is what a comparison with
published FISPACT-II results needs: those in the IAEA CoNDERC archives were run
with TENDL-2017 and decay_2012.

<!-- doctest: skip -->
```python
yani.neutron_cross_section_data = "tendl-2017"
yani.transmutation_reactions = "tendl-2017"
yani.transmutation_branch_ratios = "tendl-2017"
yani.transmutation_decay_data = "ukdd-2012"
yani.transmutation_fission_yields = "ukfy-4.2"
```

Each is the source of its one subsection only. Named as cross sections or as
another subsection, it is refused before anything is downloaded, with a message
naming the setting that does take it. `data_releases` records each library
of a mixed network like this one, and each one in it is to be cited: see
[Cite the nuclear data you use](#cite-the-nuclear-data-you-use).

UKFY-4.2 is large for a yields table: its `fission_yields` subsection is
about 0.5 GB, against 2.8 MB for ENDF/B-VIII.1's, because every parent is
evaluated at 37 energies. The first run that reads it downloads it, and later
runs read it from the cache.

### Photonuclear data in the next TENDL release

The next [TENDL-2025][cite-tendl-2025] and [TENDL-2017][cite-tendl-2017] regeneration processes `TENDL-g.tgz` as well
as the neutron archive. These gamma evaluations describe photonuclear
reactions on nuclides, not photoatomic interactions on elements. The published
`2026-10-12.1` release does not contain the new files; they become available
only after the regenerated release is validated and published.

Output lives under `photonuclear/{Nuclide}.arrow/`:

- `photonuclear.arrow` holds MF=3 cross sections on their original energy grids,
  Q values and ENDF interpolation, in eV and barns.
- `photonuclear_sections.arrow` preserves every original ENDF section,
  including outgoing-particle distributions.
- `version.json` identifies `particle="photonuclear"` and `scope="evaluated"`.

The particle index and release manifest list the available evaluations and
files. These are evaluated data, not transport-ready secondary distributions.
`Material.transmute()` remains neutron-flux driven and does not consume these
tables, and yamc does not yet sample photonuclear collisions. In particular,
TENDL gamma data cannot replace a photoatomic `photon/` library.

For a local evaluation, the converter and readback are available on this wheel:

<!-- doctest: skip -->
```python
folder = yani.convert_photonuclear(
    input_path="g-Fe056.tendl", output_dir="out/photonuclear",
    library="tendl-2025",
)
summary = yani.read_photonuclear_from_arrow(folder)
print(summary["name"], summary["mts"])
```

The readback checks the Arrow cross sections against the preserved MF=3 source.
Neither neutron NJOY processing nor auxiliary atomic tabulations are needed.

## How big each library is

Counted from published release `2026-10-12.1`, not from the forthcoming
regeneration. `--` means that release does not publish the category, rather
than publishing an empty one. Photonuclear evaluation counts will come from
the new release's index; the photoatomic counts below do not include them.

| library | neutron | photoatomic elements | reaction edges | branching channels | decay nuclides | covariance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [`endf-b8.1`][cite-endf-b8.1] | 557 | 100 | 12,444 | 917 | 3,820 | 48% |
| [`jeff-4.0`][cite-jeff-4.0] | 593 | -- | 81,045 | 8,362 | 3,851 | 99% |
| [`jendl-5.0`][cite-jendl-5.0] | 795 | 100 | 145,117 | 5,397 | 4,070 | 13% |
| [`fendl-3.2d`][cite-fendl-3.2d] | 192 | 61 | 27,814 | 652 | -- | 35% |
| [`tendl-2025`][cite-tendl-2025] | 2,848 | -- | 415,154 | 40,934 | -- | 100% |
| [`tendl-2017`][cite-tendl-2017] | 2,801 | -- | 417,434 | 41,185 | -- | 100% |

`neutron` and `photoatomic elements` are evaluations, one per nuclide and one per element.
`reaction edges` is how many `(parent, reaction, product)` triples the network
holds, which is the number that decides whether a product is reachable at all;
the isomeric overlay adds more at solve time, 12,817 rather than 12,444 on
`endf-b8.1`. Most of each count is the residual production of MT=5, the
`(n,X)` edges described under
[Weighted, from a solve](usage.md#weighted-from-a-solve): 7,269 of
ENDF/B-VIII.1's 12,444 and 347,357 of TENDL-2025's 415,154. `covariance` is the
share of neutron evaluations carrying cross-section covariance.

The spread between the two ends is the thing worth noticing: TENDL carries five
times the evaluations of ENDF/B-VIII.1 and thirty-three times the reaction
edges, and ENDF/B-VIII.1 carries the decay data that TENDL has none of. That is
why the usual arrangement is a TENDL `reactions` network with ENDF/B-VIII.1
`decay` and `fission_yields` beside it.

Mixing that way asks the decay subsection to cover everything the reactions
network can reach, and the published pairings do: the TENDL-2025 network puts
3,366 nuclides in play, and both ENDF/B-VIII.1 and JENDL-5 carry decay data for
every one of them. It is worth knowing what the miss would look like, because a
hand-built or scoped chain can produce it. A product with no decay entry gets no
decay term in the burnup matrix, so it accumulates and never decays, reports zero
activity and zero heat, and is indistinguishable from a genuinely stable
product.

### Covariance is presence, not coverage

The percentage counts evaluations that carry cross-section covariance, which is
not the same as covariance landing where your run needs it. Read
`rate_fraction_covered_total` before any sigma: it weights each channel by the
production it drove, so a channel stated with no covariance, or only on partial
levels, reads as uncovered however many of the element's isotopes carry MF=33.
[Nuclear-data uncertainty](usage.md#nuclear-data-uncertainty) has the cases.

Depth varies as much as presence. [JENDL-5][cite-jendl-5.0] carries covariance for only 13% of its
evaluations, and its Fe56 covers 56 channels against [ENDF/B-VIII.1][cite-endf-b8.1]'s 7.

Three covariance families reach the data. The cross sections' MF=33 is folded
against your spectrum. Resonance-parameter covariance (MF=32), where many
evaluations keep their whole resonance-range uncertainty, is used where the
library carries the parameters themselves (`resonance_parameters.arrow`): each
replica draws them and rebuilds the resonance range from the draw. A nuclide
whose parameters cannot be sampled falls back to the first-order MF=32 blocks
the converter writes into `covariance.arrow`. On data with neither, a capture
rate driven by resonance flux can carry a sigma near zero on a library whose
MF=33 is zero there, and the coverage report shows that case: a zero-valued
MF=33 interval counts as uncovered, so such a capture reads a
`rate_fraction_covered` well below 1 (about 0.07 for ENDF/B-VIII.1 W186 on the
FNS spectrum, with MF=33 alone). The isomeric splits' MF=40 covariance, in the
branching subsection's `branching_covariance.arrow`, is drawn by the
`isomer_branching` source. The ENDF parser also reads angular-distribution
covariance (MF=34), and nothing consumes it yet. The decay data's own sigmas on
half-lives, decay energies, decay branching ratios (any number of modes),
decay photon lines and fission yields are carried as well, and propagated.
Everything held at nominal is listed under [What is not
propagated](method.md#what-is-not-propagated).

### Energy range

Every library reaches past 20 MeV for at least some evaluations, and only
[TENDL-2025][cite-tendl-2025] is uniform. Counting evaluations by the ceiling they were converted
to:

| library | ceiling |
| --- | --- |
| [`endf-b8.1`][cite-endf-b8.1] | 452 at 20 MeV, 34 at 150, 34 at 200, 28 at 30, rest at 28 to 65 |
| [`jeff-4.0`][cite-jeff-4.0] | 491 at 200 MeV, 65 at 20, 16 at 150, 10 at 30, rest at 60 to 90 |
| [`jendl-5.0`][cite-jendl-5.0] | 579 at 200 MeV, 216 at 20 |
| [`fendl-3.2d`][cite-fendl-3.2d] | 109 at 200 MeV, 72 at 150, 11 at 60 to 65, none at 20 |
| [`tendl-2025`][cite-tendl-2025] | 200 MeV throughout |
| [`tendl-2017`][cite-tendl-2017] | 2,780 at 200 MeV, 14 at 20, 4 at 150, rest at 30 to 60 |

So ENDF/B-VIII.1 is the only one that is mostly a 20 MeV library, and even its
Fe56 and W186 reach 150 MeV. A spectrum whose flux reaches above a nuclide's
ceiling is **refused before the collapse**, naming the nuclide and the energy
its evaluation stops at, because there is no cross section up there to fold
against and any answer would be an invention. So check the ceiling of the
nuclides that carry your reaction rather than the library's headline.

Groups above the ceiling that carry no flux cost nothing, which is what makes a
padded spectrum safe: the fold stops at the last tabulated point and contributes
zero above it, rather than carrying the last value forward to 1 GeV. Those are
different answers whenever the top groups are populated, and the second one is
wrong: a threshold reaction whose cross section is still rising at 20 MeV would
otherwise be priced at its 20 MeV value across two more decades.

### Some decay records are placeholders

A nuclide whose decay scheme nobody has evaluated still gets a decay file. The
conversions from the Nuclear Wallet Cards and from NUBASE carry a half-life and
the decay modes and, in place of measured average energies, book a third of each
beta or electron-capture branch's Q to the light particles, a third to the
photons and the last third to the neutrino. That is not a rough estimate:
electron capture hands most of Q to the neutrino, so for an EC emitter the
placeholder can be several times the recoverable energy. [ENDF/B-VIII.1][cite-endf-b8.1] books
Sn111 at 1.63 MeV per decay against the 0.69 MeV [JENDL-5][cite-jendl-5.0] evaluates from its
decay scheme.

| library | placeholder decay records |
| --- | ---: |
| [`endf-b8.1`][cite-endf-b8.1] | 1144 |
| [`jeff-4.0`][cite-jeff-4.0] | 1529 |
| [`jendl-5.0`][cite-jendl-5.0] | 377 |

They are exotic nuclides, so most runs never touch one, but decay heat at short
cooling times is carried by exactly the short-lived products that tend to be
unevaluated: on the FNS tin, cadmium, palladium and nickel foils these records
carry 12 to 35% of the heat in the first minutes. The converter labels them
rather than hiding them, and can substitute another library's evaluation for
them; see
[Filling placeholder decay energies](usage.md#filling-placeholder-decay-energies).

### Some reactions do not list the isomer

A reaction that can leave its product in a metastable state has to say so in
the evaluation, one entry per final state. One that lists the ground state
alone is not merely less accurate: the isomer is absent from any network built
from it, so no code can make it, and a measurement that sees its decay heat
cannot be reproduced by any means. Nothing about such a channel looks wrong
from outside. It is present, its cross section is reasonable, and only the
state list is short.

Two channels where that decides an answer, both from the FNS decay-heat
benchmark:

| channel | `jeff-4.0` | `endf-b8.1` | `jendl-5.0` | `tendl-2017` | `tendl-2025` |
| --- | :---: | :---: | :---: | :---: | :---: |
| `Os190 (n,n')` to Os190m, 1706 keV | yes | no file | yes | **no** | yes |
| `Ir191 (n,2n)` to Ir190m2, 377 keV | yes | yes | yes | **no** | yes |

The heat from an osmium foil five minutes after a 14 MeV irradiation is
Os190m, so on `tendl-2017` the calculated heat is a third of the measurement
and no solver setting recovers it. On `tendl-2025` it is within about 20%.
That is a larger effect than any of the differences above, and it is invisible
in the cross sections.

So a library can be asked what states it lists, before a run rather than after:

<!-- doctest: skip -->
```python
channels = yani.radionuclide_production(neutron)
for channel in channels:
    if channel["reaction"] == "(n,2n)":
        print(channel["parent"], [s["excitation_energy_eV"] for s in channel["states"]])
```

The excitation energy is what compares between libraries. The level index beside
it is the evaluation's own numbering and is not comparable: Ir190's 377 keV
isomer is level 3 in ENDF/B-VIII.1 and level 37 in JEFF-4.0 and TENDL-2025.

### TENDL-2025's lithium breeds no tritium

TENDL-2019 and TENDL-2023 carried ENDF/B-VIII.0's lithium unchanged, two Los
Alamos evaluations: Li6 by G.M. Hale (2017), which gives `(n,p)` and `(n,t)` as
MT=103 and MT=105, and Li7 by P.G. Young (1988). [TENDL-2025][cite-tendl-2025] replaces both with
TALYS output, written as ENDF by TEFAL. Its Li6 file has cross sections for MT
1 to 5, 51 to 55, 91 and 102 only, and its Li7 file for MT 1 to 5, 16, 51 to 54,
91 and 102, so neither has a triton channel: the tritons MT=5 produces, and the
MT=205 derived from them, begin at 30 MeV. Li6's resonance range is a
six-resonance multi-level Breit-Wigner fit with neutron and gamma widths only,
and its thermal total cross section is 0.485 b, where ENDF/B-VIII.1's is
938.9 b, almost all of it `(n,t)`.

So lithium on `tendl-2025` makes no tritium below 30 MeV, which is every energy
a fusion or fission system reaches. ENDF/B-VIII.1 at 294 K has Li6 `(n,t)` at
938.1 b at 0.0253 eV and 0.0259 b at 14.06 MeV, and Li7 triton production
(MT=205) at 0.301 b at 14.06 MeV; TENDL-2025 has zero for all three. In yani, a
Li6 foil's H3 production averaged over a 0.02 to 0.03 eV group is 949 b on
`endf-b8.1` and 0 on `tendl-2025`, and over a 14.0 to 14.1 MeV group 0.0259 b
and 0. Transport shows it at the scale of a blanket: a 100 cm sphere of natural
lithium around a 14.06 MeV source has a tritium breeding ratio of 1.2169 +/-
0.0021 on `endf-b8.1` and exactly 0 on `tendl-2025` in yamc (100,000 histories).

For a lithium-bearing breeder, blanket or shield, take both the cross sections
and the `reactions` network from another library. The network is what holds
the `(n,t)` channel: with `transmutation_reactions = "tendl-2025"`, Li6 makes no
tritium even when its cross sections come from `endf-b8.1`.

### Temperatures and size

All six are converted at the same six temperatures, 250, 293.6, 600, 900, 1200
and 2500 K, so a library swap never changes which temperature is available.

Size is per nuclide rather than per library, because only the nuclides a run
loads are fetched: those in your material and the products its irradiation
populates. Fe56 is 56 MB on `endf-b8.1` and 19 MB on `tendl-2025`; U235 is
179 MB and 387 MB, 346 MB of the latter its cross-section covariance, which
only an uncertainty run reads. The chains are not small: 88 MB for a TENDL
network against 67 MB for JENDL-5's complete one, most of it the `(n,X)`
residual production in `reactions`.

### TENDL chains are not standalone

`TransmutationChain` cannot open a TENDL or FENDL chain on its own, because it
has no `decay` subsection and the loader wants a complete one. Set the subsections
individually instead, which is the arrangement the tables above describe:

<!-- doctest: skip -->
```python
yani.transmutation_reactions = "tendl-2025"
yani.transmutation_branch_ratios = "tendl-2025"
yani.transmutation_decay_data = "endf-b8.1"
yani.transmutation_fission_yields = "endf-b8.1"
```
