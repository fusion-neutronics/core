# Nuclear data libraries

Six libraries are available by keyword. Point a setting at one and the sections it
needs are downloaded on first use from the library's current published release,
verified against that release's manifest, and cached on disk under
``yamc.cache_dir()``. Releases, the cache layout and offline use are described
in [Nuclear data](nuclear_data.md#cross-sections-from-a-library-keyword).

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"
```

| keyword | what it is | cite |
|---------|-------------------------------------------------------------------|---------|
| [``endf-b8.1``][cite-endf-b8.1] | ENDF/B-VIII.1. | [Nobre et al. 2026](#cite-endf-b8-1) |
| [``jeff-4.0``][cite-jeff-4.0] | JEFF-4.0. | [JEFF-4.0 data sets](#cite-jeff-4-0) |
| [``jendl-5.0``][cite-jendl-5.0] | JENDL-5. | [Iwamoto et al. 2023](#cite-jendl-5-0) |
| [``tendl-2025``][cite-tendl-2025] | TENDL-2025. The widest isotope coverage of the six, but it has no hydrogen or helium, and its lithium has a tritium production problem: see [Hydrogen and helium in TENDL-2025](#hydrogen-and-helium-in-tendl-2025) and [Lithium in TENDL-2025](#lithium-in-tendl-2025). | [Koning and Rochman 2012; Koning et al. 2019](#cite-tendl) |
| [``tendl-2017``][cite-tendl-2017] | TENDL-2017, for comparing a result against the older release. | [Koning and Rochman 2012; Koning et al. 2019](#cite-tendl) |
| [``fendl-3.2d``][cite-fendl-3.2d] | FENDL-3.2d. Cross sections, plus ``reactions`` and ``branching`` subsections. | [Schnabel et al. 2024](#cite-fendl-3-2d) |

The evaluators who made these libraries ask to be cited wherever their data is
used: see [Cite the nuclear data you use](#cite-the-nuclear-data-you-use).

Four more keywords each supply one transmutation subsection and no cross
sections: [``irdff-ii``][cite-irdff-ii] (isomeric branching),
[``ukdd-2012``][cite-ukdd-2012] and [``ukdd-2020``][cite-ukdd-2020] (decay data)
and [``ukfy-4.2``][cite-ukfy-4.2] (fission yields). See
[UKAEA decay data and UKFY fission yields](#ukaea-decay-data-and-ukfy-fission-yields).

Every setting also takes a path instead of a keyword, including a folder that
holds a whole library laid out as a release, which serves every setting at once
(see [A whole library in one folder](nuclear_data.md#a-whole-library-in-one-folder)),
and ``neutron_cross_section_data`` and ``photon_cross_section_data``
additionally take a directory or a dict, keyed by nuclide and by element
respectively; the four
``transmutation_*`` parts take one value each, since a network is assembled per
subsection rather than per nuclide. Which setting does what, and what each does
when left alone, is in [Nuclear data settings](nuclear_data.md#nuclear-data-settings).
Each library can also be built locally from its evaluated sources with
``yamc.generate_library``, into such a folder (see
[Building a library yourself](nuclear_data.md#building-a-library-yourself)).

--8<-- "cite-nuclear-data.md"

## Asking yamc what a library holds

``yamc.available_nuclear_data_libraries()`` answers from each library's
current release: a dict keyed by keyword, giving the nuclides with neutron
cross sections, which of them carry each kind of covariance, the thermal
scattering tables, the elements with photo-atomic data, the targets with
photonuclear data, which transmutation subsections it publishes and the
upstream files it was built from. It reads each release's manifest, so no data
is downloaded, and the manifests are cached after the first call.

<!-- doctest: skip -->
```python
libraries = yamc.available_nuclear_data_libraries()
libraries["endf-b8.1"]["photon"][:3]          # ['Ac', 'Ag', 'Al']
libraries["tendl-2025"]["photon"]             # []: no photo-atomic data
len(libraries["tendl-2025"]["photonuclear"])  # photonuclear targets
"Fe56" in libraries["jeff-4.0"]["neutron_covariances"]["cross_section"]
libraries["irdff-ii"]["isomeric_branching"]   # True
```

Every list is complete, so an empty one means the library has none of that.
A library that could not be looked up (no network and nothing cached) has
``None`` throughout and the reason under ``"error"``. The tables below are the
same information, summarised.

## What each library publishes

Hover a column heading for what it holds; the list under the table says the same.

!!! note "Cross sections or reactions?"
    There is one set of cross sections, and it is the first column: transport
    samples collisions from it, and transmutation folds the same neutron cross
    sections with the spectrum to get its reaction rates. The ``reactions``
    subsection is not a second set: it holds no cross sections, only the network
    of which reaction on which nuclide makes which product. The cross sections
    decide **how fast** each reaction runs and the network decides **which**
    reactions and products exist, so an activation run needs both, and the two
    can come from different libraries. More in
    [The two that both say reactions](nuclear_data.md#the-two-that-both-say-reactions).

| library | <abbr title="neutron_cross_section_data and photon_cross_section_data: the cross sections transport samples collisions from. The neutron ones are also folded with the spectrum into transmutation reaction rates, so they decide how fast each reaction runs. The only cross sections a calculation uses.">cross sections</abbr> | <abbr title="transmutation_decay_data: half-lives, decay modes, decay energies and decay photon lines. Decides how the inventory decays and the photon source of a shutdown dose run.">``decay``</abbr> | <abbr title="transmutation_reactions: the network of which reaction on which nuclide makes which activation product, with its Q. Holds no cross sections. Decides which products can be made at all.">``reactions``</abbr> | <abbr title="transmutation_fission_yields: which fission products a fission makes, and how many. Only read when something fissions.">``fission_yields``</abbr> | <abbr title="transmutation_branch_ratios: how a reaction's product splits between ground and metastable states, against incident energy.">``branching``</abbr> |
| --- | --- | :---: | :---: | :---: | :---: |
| [``endf-b8.1``][cite-endf-b8.1] | neutron + photoatomic | yes | yes | yes | yes |
| [``jeff-4.0``][cite-jeff-4.0] | neutron | yes | yes | yes | yes |
| [``jendl-5.0``][cite-jendl-5.0] | neutron + photoatomic | yes | yes | yes | yes |
| [``fendl-3.2d``][cite-fendl-3.2d] | neutron + photoatomic | -- | yes | -- | yes |
| [``tendl-2025``][cite-tendl-2025] | neutron + photonuclear (next release, data only) | -- | yes | -- | yes |
| [``tendl-2017``][cite-tendl-2017] | neutron + photonuclear (next release, data only) | -- | yes | -- | yes |

- **cross sections**: neutron data per nuclide, set by
  ``neutron_cross_section_data``, and photoatomic data per element, set by
  ``photon_cross_section_data``. Transport samples collisions from them, and the
  neutron ones also give transmutation its reaction rates.
- **``decay``**, set by ``transmutation_decay_data``: half-lives, decay modes,
  decay energies and decay photon lines.
- **``reactions``**, set by ``transmutation_reactions``: which reaction on which
  nuclide makes which product, and its Q. It decides which activation products
  can be made at all, and holds no cross sections.
- **``fission_yields``**, set by ``transmutation_fission_yields``: which fission
  products a fission makes, and how many.
- **``branching``**, set by ``transmutation_branch_ratios``: how a reaction's
  product splits between its ground and metastable states.

``yamc.neutron_cross_section_data`` selects a library's neutron data and
``yamc.photon_cross_section_data`` its photoatomic data, independently. The
photonuclear entries describe additional evaluated data, not another photon
transport library.
The neutron setting is not transport-only: ``Material.transmute()`` and
``Model.simulate_transmutation()`` fold the same cross sections against the
spectrum to get their reaction rates, so an activation run needs it as much as a
transport run does. The four remaining columns are the transmutation subsections,
each set independently through its own ``yamc.transmutation_*`` setting. A library
with no subsections still works for transport, it just cannot supply a chain.

[TENDL][cite-tendl] has neutron and photonuclear evaluations, but no decay or fission-yield evaluations. Take
those from a library that has them and source ``reactions`` and ``branching`` from
TENDL, whose parent coverage is far wider. [FENDL-3.2d][cite-fendl-3.2d] publishes the same two
subsections, built against [ENDF/B-VIII.1][cite-endf-b8.1] decay data; its ``branching`` is the
reason to use it, since 54 of its evaluations carry isomer production (MF=9 or
MF=10), 15 of them with MF=40 covariances. Pointing a
transmutation part at a library that does not provide it fails immediately with a
message listing the subsections that library does provide.

``endf-b8.1``, ``jeff-4.0`` and ``jendl-5.0`` each supply a complete,
single-library transmutation chain.

### UKAEA decay data and UKFY fission yields

Three keywords supply one transmutation subsection each, and no cross sections:
the decay and fission-yield libraries UKAEA processes for FISPACT-II, from
[UKAEA's FISPACT-II nuclear data repository](https://git.oecd-nea.org/fispact/nuclear_data).

| keyword | what it is | subsection | cite |
| --- | --- | --- | --- |
| [``ukdd-2012``][cite-ukdd-2012] | UKAEA decay_2012: 3,875 nuclides. | ``decay`` | [UKAEA (CC BY 4.0)](#cite-ukdd) |
| [``ukdd-2020``][cite-ukdd-2020] | UKAEA decay_2020, its update from newer evaluations (JEFF-3.3, JENDL/DDF-2015, ENDF/B-VIII.0 among them): 4,035 nuclides. | ``decay`` | [UKAEA (CC BY 4.0); Katakura and Minato 2016](#cite-ukdd) |
| [``ukfy-4.2``][cite-ukfy-4.2] | UKFY-4.2, neutron-induced fission yields evaluated by R.W. Mills (NNL): 223 parents, each at 37 incident energies from 1e-5 eV to 150 MeV. | ``fission_yields`` | [UKAEA (CC BY 4.0); evaluator R.W. Mills](#cite-ukfy-4-2) |

[TENDL][cite-tendl] has no decay or fission-yield evaluations, so a TENDL
network takes them from another library, and these are the ones FISPACT-II
reads. A calculation set up with them reads the same data as a FISPACT-II
calculation on the same TENDL release, which is what a comparison with
published FISPACT-II results needs: those in the IAEA CoNDERC archives were run
with TENDL-2017 and decay_2012.

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "tendl-2017"
yamc.transmutation_reactions = "tendl-2017"
yamc.transmutation_branch_ratios = "tendl-2017"
yamc.transmutation_decay_data = "ukdd-2012"
yamc.transmutation_fission_yields = "ukfy-4.2"
```

Each is the source of its one subsection only. Named as cross sections or as
another subsection, it is refused before anything is downloaded, with a message
naming the setting that does take it. ``data_releases`` records each library
of a mixed network like this one, and each one in it is to be cited: see
[Cite the nuclear data you use](#cite-the-nuclear-data-you-use).

UKFY-4.2 is large for a yields table: its ``fission_yields`` subsection is
about 0.5 GB, against 2.8 MB for ENDF/B-VIII.1's, because every parent is
evaluated at 37 energies. The first run that reads it downloads it, and later
runs read it from the cache.

### Photon data is selected separately

``yamc.photon_cross_section_data`` selects photoatomic photon data, and
``yamc.neutron_cross_section_data`` selects neutron data. Both start unset and
neither falls back on the other, so a coupled or D1S run sets both, a
``PhotonSource`` run needs only the photon setting, and a neutron-only run only
the neutron one. Photon-only and coupled transport load the required elements
automatically; an extra ``material.read_nuclear_data(...)`` call is not needed
when the global setting names the intended library. Material construction itself
downloads nothing: data is loaded when a calculation needs it.

```python
import yamc

yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"
water = yamc.Material(composition={"H1": 2.0, "O16": 1.0}, density=1.0)
```

Only three libraries provide photoatomic data for photon transport:

- [``endf-b8.1``][cite-endf-b8.1] and [``jendl-5.0``][cite-jendl-5.0] publish the photoatomic and atomic-relaxation
  pair a coupled run needs. JENDL-5's is EPICS2017 adopted by JAEA, so it is the
  same data ENDF/B-VIII.1 carries.
- [``fendl-3.2d``][cite-fendl-3.2d] publishes photoatomic cross sections with no atomic relaxation,
  so absorption emits no fluorescence or Auger electrons; the loader warns
  about missing relaxation where it matters.
- [``jeff-4.0``][cite-jeff-4.0] and both [TENDL][cite-tendl] releases publish no photoatomic data at all. JEFF's
  photon sublibrary is photonuclear, which is a different quantity.

There is no implicit switch to another photon library. A photon run with
``photon_cross_section_data = "tendl-2025"`` or ``"tendl-2017"`` fails for
missing photoatomic data, even after the new photonuclear files are published.
A run that takes its neutron data from TENDL takes its photon data from one of
the three libraries above, named in ``photon_cross_section_data``. A single
material can also be given an explicit per-element ``photon_data`` map on
[read_nuclear_data](materials.md#loading-nuclear-data), which takes precedence
over the global selection.

### Photonuclear data in the next TENDL release

The next [TENDL-2025][cite-tendl-2025] and [TENDL-2017][cite-tendl-2017] regeneration adds the gamma evaluations from
``TENDL-g.tgz``. These are per-nuclide photonuclear reactions, not the
per-element photoatomic interactions above. Release ``2026-10-12.1`` does not
contain this new output; it becomes available only after the regenerated
release is validated and published.

The new folders are ``photonuclear/{Nuclide}.arrow/``. Each contains:

- ``photonuclear.arrow``: the MF=3 cross sections on their original energy grids,
  with Q values and ENDF interpolation, in eV and barns.
- ``photonuclear_sections.arrow``: every original ENDF section, including
  outgoing-particle distributions.
- ``version.json``: provenance identifying ``particle="photonuclear"`` and
  ``scope="evaluated"``.

``photonuclear/index.json`` lists the evaluations, and the release manifest
lists and hashes the files. This is data processing, not photonuclear collision
sampling: neither ``PhotonSource`` nor coupled photon transport consumes these
tables yet. A new ``photonuclear/`` folder does not supply a photoatomic
``photon/`` folder.

## How big each library is

Counted from each library's published release ``2026-10-12.1``. These counts do
not describe the forthcoming regeneration. ``--`` means that release does not
publish the category, rather than publishing an empty one.

| library | neutron | photoatomic elements | reaction edges | branching channels | decay nuclides | covariance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [``endf-b8.1``][cite-endf-b8.1] | 557 | 100 | 12,444 | 917 | 3,820 | 48% |
| [``jeff-4.0``][cite-jeff-4.0] | 593 | -- | 81,045 | 8,362 | 3,851 | 99% |
| [``jendl-5.0``][cite-jendl-5.0] | 795 | 100 | 145,117 | 5,397 | 4,070 | 13% |
| [``fendl-3.2d``][cite-fendl-3.2d] | 192 | 61 | 27,814 | 652 | -- | 35% |
| [``tendl-2025``][cite-tendl-2025] | 2,848 | -- | 415,154 | 40,934 | -- | 100% |
| [``tendl-2017``][cite-tendl-2017] | 2,801 | -- | 417,434 | 41,185 | -- | 100% |

``neutron`` and ``photoatomic elements`` are evaluations, one per nuclide and one per element.
``reaction edges`` is how many ``(parent, reaction, product)`` triples the network
holds, which is what decides whether a product is reachable at all. Most of them
are the products of ``(n,X)`` (MT=5), its residuals and light particles, one
triple each. ``covariance`` is the share of neutron evaluations carrying
cross-section covariance.

The two ends of the table differ widely. TENDL carries five times the
evaluations of ENDF/B-VIII.1 and over thirty times the reaction edges, and has none
of the decay data. That is why a TENDL ``reactions`` network with
ENDF/B-VIII.1 ``decay`` and ``fission_yields`` beside it is the usual arrangement.

### Covariance is presence, not coverage

The percentage counts evaluations that carry cross-section covariance, which is
not the same as covariance landing where a given run needs it. Depth varies as
much as presence: [JENDL-5][cite-jendl-5.0] carries covariance for 13% of its evaluations, and its
Fe56 covers 56 channels against [ENDF/B-VIII.1][cite-endf-b8.1]'s 7.

Six covariance families reach the converted data, and three of them are sampled.
The cross sections' MF=33 is drawn by the ``cross_sections`` source.
Resonance-parameter covariance (MF=32), where an evaluation often keeps its
resolved-resonance uncertainty while leaving MF=33 at zero over that range, is
used two ways: where a nuclide's folder carries the parameters themselves
(``resonance_parameters.arrow``, in the published libraries), each sample draws
them and rebuilds the resonance range from the draw; otherwise the converter's
first-order MF=32 rows in ``covariance.arrow`` are folded like MF=33. Data with
neither arrives with no capture, elastic or fission uncertainty below the top of
its resolved resonances, and the uncertainty report counts rate from energies
where the stated variance is zero as uncovered, so that gap shows in
``rate_fraction_covered``. The radionuclide-production covariance (MF=40, the
covariance of each product state's production cross section) is in the branching
subsection's ``branching_covariance.arrow`` and drawn by the
``isomer_branching`` source. ENDF/B-VIII.1, FENDL-3.2d and TENDL state no
correlation in MF=40 between a product's states, or with the reaction's MF=33,
so the split's uncertainty is only partly stated. The angular-distribution
covariance (MF=34, ``angular_covariance.arrow``), the fission multiplicity
covariance (MF=31, ``nubar_covariance.arrow``) and the fission spectrum
covariance (MF=35, ``spectrum_covariance.arrow``) are converted and published
too, and ``Model.data_uncertainty_coverage()`` reports which nuclides carry
them, but nothing samples them yet.

Beyond covariance, the decay data carries each nuclide's half-life with the
standard deviation its evaluation states, and its decay energy with a standard
deviation built from the evaluation's sigmas on the beta, gamma and alpha
components. The evaluation states no correlation between them, so they are
treated as independent and the total's sigma is their quadrature sum. The
``half_life`` and ``decay_energy`` uncertainty sources draw from these. The
converter also stores the sigma each evaluation states on a decay mode's
branching ratio (``branching_ratio_uncertainty`` in ``decay_modes.arrow``),
which the ``decay_branching`` source draws from for parents with any number of
modes. Decay data converted before that column existed carries no branching
sigma. The converter also stores each evaluation's fission yields with their
stated sigmas (``yield_uncertainties`` in
``fission_yields/evaluated_yields.arrow``), which the ``fission_yield`` source
draws, and the sigmas stated on decay photon lines, which the
``decay_photon_lines`` source draws. [What it does not cover](materials.md#what-it-does-not-cover) lists what a
nuclear-data uncertainty run holds at nominal.

### Energy range

Every library reaches past 20 MeV for at least some evaluations, and only
[TENDL-2025][cite-tendl-2025] takes every evaluation to the same ceiling. Counting evaluations by
the ceiling they were converted to:

| library | ceiling |
| --- | --- |
| [``endf-b8.1``][cite-endf-b8.1] | 452 at 20 MeV, 34 at 150, 34 at 200, 28 at 30, rest at 28 to 65 |
| [``jeff-4.0``][cite-jeff-4.0] | 491 at 200 MeV, 65 at 20, 16 at 150, 10 at 30, rest at 60 to 90 |
| [``jendl-5.0``][cite-jendl-5.0] | 579 at 200 MeV, 216 at 20 |
| [``fendl-3.2d``][cite-fendl-3.2d] | 109 at 200 MeV, 72 at 150, 11 at 60 to 65, none at 20 |
| [``tendl-2025``][cite-tendl-2025] | 200 MeV throughout |
| [``tendl-2017``][cite-tendl-2017] | 2,780 at 200 MeV, 14 at 20, 4 at 150, 2 at 30, 1 at 60 |

ENDF/B-VIII.1 is the only one that is mostly a 20 MeV library, and even its Fe56
and W186 reach 150 MeV. For a high-energy source, check the ceiling of the
nuclides that carry the reaction rather than the library's headline figure.

Transport and transmutation treat the ceiling differently. In transport a particle above a nuclide's ceiling finds no
cross section up there, and that reads as a rate too low rather than as an
error. In a transmutation collapse, from yamc 0.11.0, it is refused: a group
spectrum whose flux reaches past an evaluation's last energy point is rejected
before the fold, naming the nuclide and the energy its evaluation stops at,
because there is nothing up there to fold against and any answer would be
invented. Earlier releases carried the last tabulated value forward and folded
against that, which gives a wrong answer whenever the top groups are populated.
Groups above the ceiling that carry no flux are accepted, so a padded spectrum
is safe.

### What a library can quietly lack

A library can lack two things that neither the tables above nor its cross
sections reveal. Both matter for decay heat and shutdown dose rather than for
transport.

The first is placeholder mean decay energies. A nuclide whose decay scheme
nobody has evaluated still gets a decay file, with a third of each beta or
electron-capture branch's Q booked to the light particles and a third to the
photons. For an electron-capture emitter that is meaningless, since the
neutrino carries almost all of Q: [ENDF/B-VIII.1][cite-endf-b8.1] books Sn111 at 1.63 MeV per
decay against the 0.69 MeV [JENDL-5][cite-jendl-5.0] evaluates from its decay scheme. The counts
are 1144 records for `endf-b8.1`, 1529 for `jeff-4.0` and 375 for
`jendl-5.0`. They are exotic nuclides, but early decay heat is carried by
exactly the short-lived products that tend to be unevaluated. Every one is
named in the decay subsection's `provenance.json`, in data converted by
yamc 0.10.0 or newer; libraries published before that carry the numbers
without the record that says which of them are placeholders.

The second is reactions that do not list the isomer they make. A reaction that can leave
its product in a metastable state has to say so, one entry per final state. One
that lists the ground state alone puts the isomer outside the network
altogether, so no code can make it. `Os190 (n,n')` to Os190m at 1706 keV and
`Ir191 (n,2n)` to Ir190m2 at 377 keV are both carried by `jeff-4.0` and
`tendl-2025` and both absent from `tendl-2017`, which is why an osmium foil's
heat five minutes after a 14 MeV irradiation comes out at a third of the
measured value on that library. Nothing about such a channel looks wrong from
outside: it is present, its cross section is reasonable, and only the state
list is short.

### Lithium in TENDL-2025

[TENDL-2025][cite-tendl-2025]'s Li6 and Li7 produce no tritium below 30 MeV. For a breeder
blanket, a lithium-bearing shield, or anything else where lithium's reactions
matter, use another library.

TENDL-2019 and TENDL-2023 carried the ENDF/B-VIII.0 lithium evaluations
unchanged: Li6 from LANL (G.M. Hale, 2017), with its (n,t) reaction as MT=105,
and Li7 from LANL (P.G. Young, 1988). TENDL-2025 replaces both with TALYS
output, written by TEFAL. Its Li6 has MT=1 to 5, 51 to 55, 91 and 102 in MF=3
and no other cross section, and its Li7 has MT=1 to 5, 16, 51 to 54, 91 and
102. Neither has an (n,t) channel, so the only tritons left are those of MT=5,
which starts at 30 MeV in both. Li6's resolved resonance range is a
six-resonance MLBW fit with neutron and gamma widths only, so its thermal total
is 0.485 b, where ENDF/B-VIII.1 has 938.9 b, 938.1 b of it (n,t).

A 100 cm sphere of natural lithium (0.534 g/cm3, 294 K) with a 14.06 MeV point
source at its centre, 100,000 histories, yamc 0.16.0 on release
``2026-10-12.1``, per source neutron:

| library | tritium (TBR) | from Li6 | from Li7 | absorption |
| --- | ---: | ---: | ---: | ---: |
| [``endf-b8.1``][cite-endf-b8.1] | 1.2169 +/- 0.0021 | 0.4624 | 0.7546 | 0.4788 |
| [``jeff-4.0``][cite-jeff-4.0] | 1.2164 +/- 0.0021 | 0.4617 | 0.7547 | 0.4781 |
| [``tendl-2017``][cite-tendl-2017] | 1.2164 +/- 0.0021 | 0.4617 | 0.7547 | 0.4781 |
| [``fendl-3.2d``][cite-fendl-3.2d] | 1.2157 +/- 0.0021 | 0.4610 | 0.7548 | 0.4774 |
| [``jendl-5.0``][cite-jendl-5.0] | 1.2120 +/- 0.0020 | 0.4750 | 0.7370 | 0.4925 |
| [``tendl-2025``][cite-tendl-2025] | 0 | 0 | 0 | 0.0000054 |

The tritium tally is not the only thing that changes. With no (n,t) and next to
no capture (Li6's is 1.5e-10 b at thermal, against 0.0385 b in ENDF/B-VIII.1),
lithium absorbs almost nothing on TENDL-2025: the sphere's flux comes out 3.7%
higher and its neutron heating 43.5% lower than on ENDF/B-VIII.1, which counts
the 4.78 MeV each Li6(n,t) releases. TENDL-2025 misses tritium production
cross sections in Li6 and Li7. This is a known issue with the cross section
library.

Transmutation is affected the same way, through either setting. A litre of
natural lithium irradiated for 30 days at 1e14 n/cm2/s, in a spectrum with 30%
of its flux between 13.5 and 14.5 MeV and the rest slowed down, makes 392 appm
of tritium with ``endf-b8.1`` as both ``neutron_cross_section_data`` and
``transmutation_reactions``, and none with ``tendl-2025`` as either.

The ``tendl-2025`` keyword serves the library as published; yamc does not patch
or replace these two evaluations.

### Hydrogen and helium in TENDL-2025

[TENDL-2025][cite-tendl-2025] has no hydrogen or helium. Its neutron archive, ``TENDL-n.tgz``,
holds 2850 evaluations and none of them is H1, H2, H3, He3 or He4, so a
material containing either element (water, polyethylene, a hydride, a helium
coolant) cannot be modelled on ``tendl-2025``. This is a gap in the library as
published. Use another library for a model that contains them;
``tendl-2017`` carries all five.

### Temperatures and size

All six are converted at the same six temperatures, 250, 293.6, 600, 900, 1200 and
2500 K, so a library swap never changes which temperature is available.

Size is per nuclide rather than per library, because only the nuclides in the
model are fetched. Fe56 is 56 MB on ``endf-b8.1`` and 18 MB on ``tendl-2025``;
U235 is 171 MB and 43 MB. Transport pulls the ``distributions`` section on top of
what an activation run needs, which is 15 MB of Fe56's 56 MB on ENDF/B-VIII.1.
